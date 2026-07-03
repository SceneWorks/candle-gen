//! `Fp8Linear` / `Int8Linear` (sc-9299) — the two 8-bit linear layers over the [`CublasLt`] compute
//! leg. Each holds a *statically* quantized weight (per-tensor scale, done once at construction) and
//! quantizes the activation **dynamically** per forward (v1: amax→scale→cast in pure candle ops, a
//! fused kernel is a later optimization). This is the layer a provider crate would swap in for an
//! fp8 fast tier or an INT8-ConvRot checkpoint. This layer is just the GEMM (`X·(stored W)ᵀ`); a
//! ConvRot checkpoint's stored weight is the *rotated* `W·R`, so its consume path applies the online
//! `RHT(x)` activation rotation ([`super::convrot`]) upstream before calling this layer (sc-9601) — the
//! GEMM itself stays rotation-agnostic. The per-output-channel dequant fold runs **on-device** when the
//! device's cuBLASLt supports int8→f32 output (sc-9601 perf), else it falls back to the exact host fold.
//!
//! Both are `#[cfg(feature = "cuda")]` — they own a `CublasLt` handle. The weight-quant / act-quant
//! helpers they build on are pure candle ops (see [`super::cublaslt`]) and compile everywhere.

use super::cublaslt::{
    quantize_activation_fp8, quantize_activation_int8, quantize_weight_fp8, quantize_weight_int8,
    quantize_weight_int8_per_channel, CublasLt,
};
use candle_core::{Device, Result, Tensor};
use std::sync::Arc;

/// An fp8 E4M3 linear: `y = (X · Wᵀ) · scale_w · scale_x` with a per-tensor-quantized weight and
/// dynamic per-tensor activation quant. Optional bias added back in the output dtype.
pub struct Fp8Linear {
    w_fp8: Tensor, // (N, K) F8E4M3
    scale_w: f32,
    bias: Option<Tensor>,
    lt: Arc<CublasLt>,
}

impl Fp8Linear {
    /// Quantize a dense `(N, K)` weight to fp8 E4M3 once and bind it to a cuBLASLt handle.
    pub fn new(weight: &Tensor, bias: Option<Tensor>, lt: Arc<CublasLt>) -> Result<Self> {
        let qw = quantize_weight_fp8(weight)?;
        Ok(Self {
            w_fp8: qw.q,
            scale_w: qw.scale,
            bias,
            lt,
        })
    }

    /// Build sharing an existing handle for the device (constructs a fresh handle when `None`).
    pub fn from_device(weight: &Tensor, bias: Option<Tensor>, dev: &Device) -> Result<Self> {
        Self::new(weight, bias, Arc::new(CublasLt::new(dev)?))
    }

    /// `x`: `(..., K)`; flattened to `(M, K)` for the GEMM, then reshaped back. Output bf16.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (flat, restore) = flatten_tokens(x)?;
        let qx = quantize_activation_fp8(&flat)?;
        let y = self
            .lt
            .matmul_fp8(&self.w_fp8, self.scale_w, &qx.q, qx.scale)?;
        let y = restore(y)?;
        match &self.bias {
            Some(b) => y.broadcast_add(&b.to_dtype(y.dtype())?),
            None => Ok(y),
        }
    }
}

/// The weight dequant granularity of an [`Int8Linear`] — the sc-9300 extension from a single
/// per-tensor scalar to a `[out]` per-output-channel vector. Both fold `scale_w · scale_x` onto the
/// exact int32 accumulate; per-channel simply carries one scale per output row (the granularity the
/// community INT8-ConvRot checkpoints store, `{base}.weight_scale` `[out, 1]`).
enum WeightScale {
    /// One scale for the whole weight (sc-9299): `dequant = q · scale`.
    PerTensor(f32),
    /// One scale per output row (sc-9300): `dequant[o, :] = q[o, :] · scale[o]`.
    PerChannel(Vec<f32>),
}

/// An int8 IGEMM linear: exact int32 accumulate, dequant scale folded on the candle side. Same
/// dynamic-activation-quant contract as [`Fp8Linear`]. The weight scale is per-tensor ([`Self::new`],
/// sc-9299) or **per-output-channel** ([`Self::from_per_channel_parts`], sc-9300 — the community
/// INT8-ConvRot consume path, where the checkpoint ships int8 codes + a `[out]` row scale). The forward
/// is a plain IGEMM + per-row dequant — the exact `X·Wᵀ` for a per-channel-quantized weight. NB a
/// ConvRot checkpoint's stored weight is the *rotated* `W·R`, so the consume path applies the online
/// `RHT(x)` activation rotation ([`super::convrot`], sc-9601) upstream before this layer — this layer is
/// rotation-agnostic, computing `X·(stored W)ᵀ` either way. The per-channel dequant fold runs on-device
/// (int32 IGEMM → `cast_i32_f32` → candle float fold) when the device supports it, else the host fold.
pub struct Int8Linear {
    w_i8: Tensor, // (N, K) int codes carried in F32 — the resident weight for the non-staged path
    /// A **pre-staged** on-device `i8` weight (sc-9300): the ConvRot consume path stages the `(N, K)`
    /// codes once so the resident weight is native `i8` (1 byte/elem), not an 8×-larger I64/F32 tensor
    /// (a 12B DiT's 224 int8 projections otherwise blow VRAM). When set, the per-channel forward uses
    /// the staged matmul; `w_i8` then holds only the small CPU source (kept for shape queries).
    w_staged: Option<super::cublaslt::DevInt8>,
    scale_w: WeightScale,
    bias: Option<Tensor>,
    lt: Arc<CublasLt>,
    /// Whether this device can fold the dequant **on-device** (sc-9601) — i.e. cast the int32 accumulate
    /// `i32 → f32` via the vendored `cast_i32_f32` kernel. When set, the per-channel staged forward folds
    /// on-device (no int32→host round-trip); otherwise it falls back to the exact host fold. Probed once.
    ondevice_dequant: bool,
}

impl Int8Linear {
    /// Per-tensor int8 (sc-9299): quantize a dense `(N, K)` weight once and bind a cuBLASLt handle.
    pub fn new(weight: &Tensor, bias: Option<Tensor>, lt: Arc<CublasLt>) -> Result<Self> {
        let qw = quantize_weight_int8(weight)?;
        Ok(Self {
            w_i8: qw.q,
            w_staged: None,
            scale_w: WeightScale::PerTensor(qw.scale),
            bias,
            lt,
            ondevice_dequant: false, // per-tensor forward uses the host fold (tests)
        })
    }

    pub fn from_device(weight: &Tensor, bias: Option<Tensor>, dev: &Device) -> Result<Self> {
        Self::new(weight, bias, Arc::new(CublasLt::new(dev)?))
    }

    /// **Per-output-channel int8 from a dense weight** (sc-9300) — quantize `(N, K)` to int8 with a
    /// per-row scale. The from-dense twin of [`Self::from_per_channel_parts`] (used by numerics tests
    /// and any dense→int8 fold); a real ConvRot checkpoint uses the parts constructor instead.
    pub fn new_per_channel(
        weight: &Tensor,
        bias: Option<Tensor>,
        lt: Arc<CublasLt>,
    ) -> Result<Self> {
        let qw = quantize_weight_int8_per_channel(weight)?;
        Ok(Self {
            w_i8: qw.q,
            w_staged: None,
            scale_w: WeightScale::PerChannel(qw.scale),
            bias,
            lt,
            ondevice_dequant: false, // from-dense (no staged weight) uses the host fold (tests)
        })
    }

    /// **Per-output-channel int8 straight from the on-disk parts** (sc-9300, the ConvRot consume path):
    /// `w_i8` is the checkpoint's `(N, K)` int8 codes (carried in any dtype the caller narrows at the
    /// stage), `scale_w` its `[N]` per-output-row `weight_scale`. No re-quantization — the stored codes
    /// and their row scales are used as-is. (For a ConvRot checkpoint the codes are a *rotated* weight,
    /// needing the online `x·R` leg upstream — sc-9601.)
    pub fn from_per_channel_parts(
        w_i8: Tensor,
        scale_w: Vec<f32>,
        bias: Option<Tensor>,
        lt: Arc<CublasLt>,
    ) -> Result<Self> {
        let n = w_i8.dims2()?.0;
        if scale_w.len() != n {
            candle_core::bail!(
                "Int8Linear::from_per_channel_parts: scale_w len {} != weight rows {n}",
                scale_w.len()
            );
        }
        // Pre-stage the codes to a resident native-`i8` device buffer (1 byte/elem) so the 224
        // projections of a 12B DiT don't hold their codes as 8×-larger I64 tensors on the GPU.
        let w_staged = Some(lt.stage_int8(&w_i8)?);
        // Probe once whether the on-device int32→f32 dequant fast path is available (sc-9601).
        let ondevice_dequant = lt.supports_ondevice_int8_dequant();
        Ok(Self {
            w_i8,
            w_staged,
            scale_w: WeightScale::PerChannel(scale_w),
            bias,
            lt,
            ondevice_dequant,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (flat, restore) = flatten_tokens(x)?;
        let qx = quantize_activation_int8(&flat)?;
        let y = match (&self.scale_w, &self.w_staged) {
            (WeightScale::PerChannel(s), Some(w)) if self.ondevice_dequant => self
                .lt
                .matmul_int8_per_channel_staged_ondevice(w, s, &qx.q, qx.scale)?,
            (WeightScale::PerChannel(s), Some(w)) => self
                .lt
                .matmul_int8_per_channel_staged(w, s, &qx.q, qx.scale)?,
            (WeightScale::PerChannel(s), None) => self
                .lt
                .matmul_int8_per_channel(&self.w_i8, s, &qx.q, qx.scale)?,
            (WeightScale::PerTensor(s), _) => {
                self.lt.matmul_int8(&self.w_i8, *s, &qx.q, qx.scale)?
            }
        };
        let y = restore(y)?;
        match &self.bias {
            Some(b) => y.broadcast_add(&b.to_dtype(y.dtype())?),
            None => Ok(y),
        }
    }
}

/// Collapse leading dims to a `(M, K)` matrix and return a closure that restores the original
/// leading shape on the `(M, N)` output.
fn flatten_tokens(x: &Tensor) -> Result<(Tensor, impl Fn(Tensor) -> Result<Tensor>)> {
    let dims = x.dims().to_vec();
    let k = *dims.last().expect("linear input has a last dim");
    let m: usize = dims[..dims.len() - 1].iter().product();
    let flat = x.reshape((m, k))?;
    let lead = dims[..dims.len() - 1].to_vec();
    let restore = move |y: Tensor| -> Result<Tensor> {
        let n = y.dim(1)?;
        let mut out_shape = lead.clone();
        out_shape.push(n);
        y.reshape(out_shape)
    };
    Ok((flat, restore))
}
