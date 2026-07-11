//! The shared **packed-load** module (sc-9086, epic 9083) — the candle twin of
//! `mlx_gen::quant::{lin, embedding}` ([[the MLX Group-B template]], sc-8669). Every provider
//! crate's loader packed-**detects** with one call: a pre-quantized MLX tier (epic 8506, e.g.
//! `SceneWorks/z-image-turbo-mlx`) stores each quantized `Linear` / token embedding as the packed
//! triple `{base}.weight` (u32 codes) + `{base}.scales` + `{base}.biases`; [`lin`] / [`embedding`]
//! auto-detect it by the presence of `{base}.scales` (and/or the component `config.json`'s
//! `quantization: { bits, group_size }` block — [`PackedConfig`]) and build the quantized module
//! **directly from the packed parts** via the sc-9085 [`repack`] seam. Absent `.scales`, the dense
//! path is taken **unchanged**, so one loader serves both a dense bf16 and a pre-quantized snapshot.
//!
//! **The quantized forward dequantizes the weight into a *dense* matmul (sc-7702).** It does NOT
//! take candle's int8 `QMatMul` fast path (`fast_mmvq`/`fast_mmq`), which quantizes the *activation*
//! to per-32-element `q8_1`: a single outlier text feature (gpt-oss ±10⁴) sets a block's int8 scale
//! and zeros the co-located channels, so a Q4 denoise diverges to NaN — a solid-black render.
//! Dequantizing the weight to the activation dtype and running a plain matmul keeps the activation
//! full-precision, so uniform Q4 renders coherently. The resident footprint stays the small
//! quantized [`QTensor`]; the dequant is per-forward. This mirrors the Lens DiT quant (sc-5117) and
//! is regression-tested in [`tests`] (`q4_packed_forward_survives_outlier_activations`, CUDA-gated).
//!
//! **Idempotent [`QLinear::quantize`].** Crates call `quantize(bits)` after a dense load today; on a
//! QLinear that already loaded packed (`Quantized`) that call is a **no-op** — it does not
//! re-quantize (mirroring MLX's `AdaptableLinear::quantize` no-op-when-`Quantized`). So a loader can
//! packed-detect *and* keep an unconditional post-load `quantize` pass, and the two compose.
//!
//! **Bit-width, group size & repack.** MLX packs group-wise **affine** (`w = scale·q + bias`). The
//! z-image / flux tiers use group size 64, the default that the shape-inferring
//! [`repack::mlx_packed_bits`] and `lin`/`embedding`/`from_packed` assume; the boogu tier packs at
//! group **32** (sc-9410), which the shapes can't disambiguate, so its loaders pass the group size
//! explicitly (the `*_gs` entry points, read from `config.json`'s `quantization.group_size`,
//! [`PackedConfig`]). Q4 repacks **losslessly** into GGML `Q4_1` (same affine form; one MLX group of
//! `g` splits into `g / 32` `Q4_1` blocks). Q8 has no affine GGML container, so the Q8 tier is
//! materialized to its exact MLX grid and re-quantized to symmetric `Q8_0` (the accepted sc-9085
//! double-quant, 0.56 % mean relative RMS on the real z-image Q8 tier). See [`repack`] for the
//! byte-level details.

pub mod repack;

// The ConvRot online rotation leg (sc-9601): the regular-Hadamard activation rotation `RHT(x) = x·R`
// a community INT8-ConvRot checkpoint needs before the int8 IGEMM (its stored weight is `W·R`). Pure
// candle ops — builds everywhere (CPU tests + CUDA).
pub mod convrot;

// The cuBLASLt 8-bit GEMM compute leg (sc-9299 spike, epic 9083's 8-bit pivot): fp8 E4M3 + int8
// IGEMM matmul over cudarc's raw cublasLt sys bindings, plus the `Fp8Linear`/`Int8Linear` linear
// layers with dynamic per-tensor activation quant. The `CublasLt` handle is cuda-only; the small
// activation/weight quant helpers are pure candle ops and build everywhere.
pub mod cublaslt;
// The 8-bit linear layers own a `CublasLt` handle → cuda-only.
#[cfg(feature = "cuda")]
pub mod eight_bit_linear;

pub use convrot::{convrot_rotate, is_power_of_four, regular_hadamard};
pub use repack::{
    dequant_mlx_q4_reference, dequant_mlx_q4_reference_gs, dequant_mlx_q8, dequant_mlx_q8_gs,
    f16_exact, mlx_packed_bits, mlx_packed_bits_gs, pack_mlx_affine, repack_mlx_q4_to_q4_1,
    repack_mlx_q4_to_q4_1_gs, MLX_GROUP_SIZE,
};

#[cfg(feature = "cuda")]
pub use cublaslt::CublasLt;
pub use cublaslt::{
    quantize_activation_fp8, quantize_activation_int8, quantize_weight_fp8, quantize_weight_int8,
    quantize_weight_int8_per_channel, PerChannelInt8Weight, QuantizedActivation, F8E4M3_MAX,
    I8_MAX,
};
#[cfg(feature = "cuda")]
pub use eight_bit_linear::{Fp8Linear, Int8Linear};

use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Embedding, Linear, Module, VarBuilder};
use gen_core::Quant;

/// The GGUF block type a load-time [`Quant`] level maps to when quantizing a *dense* weight in place
/// — `Q4_0` / `Q8_0` (block size 32). Shared with the per-crate seams (Lens sc-5117, FLUX.2 sc-5917):
/// the single source of truth for the family's `Quant → GgmlDType` mapping. The **packed** path uses
/// `Q4_1` instead (the affine container the MLX tiers repack into losslessly — [`repack`]).
pub fn ggml_dtype(quant: Quant) -> GgmlDType {
    match quant {
        Quant::Q4 => GgmlDType::Q4_0,
        Quant::Q8 => GgmlDType::Q8_0,
    }
}

/// GGUF block size for `Q4_0`/`Q8_0` (the candle-core legacy quants). A [`QLinear`] folded with the
/// SAM3/SeedVR2 skip predicate is quantized only when its contraction (`in_features`) divides this;
/// otherwise it stays dense — the reference predicate that leaves e.g. SeedVR2's `vid_in.proj` (in=132)
/// and SAM3's `2→256`/`4→256`/`258→256` projections full-precision.
pub const QUANT_BLOCK: usize = 32;

/// A component's `quantization` manifest block — the candle twin of `mlx_gen_flux2::config::Flux2Quant`
/// (generalized out of the dormant per-crate `Flux2Quant`, sc-9086). An install-time convert job
/// writes `quantization: { "bits", "group_size" }` into a packed component's `config.json`; its
/// presence is a redundant (with the `.scales` sibling) packed-detect key and carries the group size
/// for crates that would rather read it than infer it from shapes. Kept `i32`-typed to match the JSON
/// the MLX converter emits (and mlx-gen's `Flux2Quant`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackedConfig {
    pub bits: i32,
    pub group_size: i32,
}

impl PackedConfig {
    /// Parse the `quantization` block out of a component `config.json` value — `None` when the block
    /// is absent (a dense component) or missing `bits` (nothing identifies it as packed). Detects a
    /// packed tier without touching the safetensors: `PackedConfig::from_config(cfg).is_some()` ⇔ the
    /// loader should take the packed path.
    ///
    /// **`group_size` absent ⇒ default to [`MLX_GROUP_SIZE`] (64), never silent dense (sc-9410).** A
    /// packed component that carries `bits` but omits `group_size` is still packed — u32 codes that a
    /// dense fallback would load as garbage. MLX's own default group size is 64 (the z-image/flux
    /// tiers), so an absent `group_size` means "the default 64", not "dense". Returning `None` here
    /// would silently degrade the whole component to the dense path over bit-packed nibbles.
    pub fn from_config(cfg: &serde_json::Value) -> Option<Self> {
        let q = cfg.get("quantization")?;
        let bits = q.get("bits")?.as_i64()? as i32;
        // Present `quantization.bits` ⇒ packed; a missing `group_size` defaults to MLX's 64 rather
        // than degrading the component to a dense read of u32 codes.
        let group_size = q
            .get("group_size")
            .and_then(|g| g.as_i64())
            .map(|g| g as i32)
            .unwrap_or(MLX_GROUP_SIZE as i32);
        Some(Self { bits, group_size })
    }
}

/// How a [`QLinear::Quantized`] arm computes its matmul — the **load-bearing** knob unified out of the
/// four per-crate seams (F-025 / sc-9005). Both strategies compute `x·Wᵀ + b` over the *same* GGUF
/// `QTensor` weight; they differ only in whether the activation stays full-precision.
///
/// - [`Self::DequantDense`] — dequantize the weight to the activation dtype and run a *dense* matmul,
///   keeping the activation full-precision. **The sc-7702 fix.** candle's int8 `QMatMul` fast path
///   (`fast_mmvq`/`fast_mmq`) quantizes the *activation* to per-32-element `q8_1`, so a single outlier
///   text feature (gpt-oss ±10⁴) sets a block's int8 scale and zeros the co-located channels → a Q4
///   denoise diverges to NaN (a solid-black render). Dequant-to-dense keeps uniform Q4 coherent. The
///   Lens DiT (sc-5117) and every packed-tier load use this.
/// - [`Self::Int8Fast`] — route the (f32) activation through candle's `QMatMul::forward`, which takes
///   the int8 fast path on CUDA for `batch > 8`. Faster, but corrupts under the outlier activations
///   [`Self::DequantDense`] guards against — safe **only** where the activations are known well-scaled
///   (FLUX.2's f32 DiT/TE, SAM3's heads, SeedVR2's DiT — all GPU-validated near-lossless). A model
///   with outlier activations on an `Int8Fast` seam reproduces the sc-7702 failure; that is exactly the
///   drift F-025 makes explicit rather than leaving four identically-named types silently disagree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatmulStrategy {
    /// Dequantize the weight to a dense matmul (sc-7702-safe; the default).
    DequantDense,
    /// candle's int8 `QMatMul::forward` fast path (activation-quant; unsafe under outliers).
    Int8Fast,
}

/// A dense `[out, in]` projection stored in one of two equivalent layouts — the second **load-bearing**
/// drift unified out of the four seams (F-025). Both forward to `x·Wᵀ + b`; they differ in weight
/// storage and the per-forward GEMM shape, which must stay byte-identical to each site's prior code.
///
/// - [`Self::Linear`] — a `candle_nn::Linear` holding the `[out, in]` weight (Lens / FLUX.2). The
///   forward is `candle_nn::Linear::forward` (a `broadcast_matmul` for rank > 2 inputs).
/// - [`Self::Transposed`] — the weight **pre-transposed** to a contiguous `[in, out]` at load
///   (SAM3 / SeedVR2, sc-8997/F-017: re-transposing per forward materialized a fresh copy of the whole
///   weight every call). The forward flattens all leading dims into one 2-D GEMM `[lead,in]@[in,out]`
///   and reshapes back — candle's `matmul` rejects the non-contiguous broadcasted rhs a high-rank input
///   otherwise produces, and the flattened GEMM is faster.
#[derive(Clone)]
pub enum DenseLinear {
    Linear(Linear),
    Transposed {
        /// Pre-transposed `[in, out]`, contiguous.
        weight_t: Tensor,
        bias: Option<Tensor>,
    },
}

impl DenseLinear {
    /// The recovered torch-native `[out, in]` weight and its `[out]` bias — used by the quantize fold
    /// (which needs the `[out, in]` layout GGUF quantization expects) and shape queries. For
    /// [`Self::Transposed`] this transposes `weight_t` back once (not per forward).
    fn out_in_weight_bias(&self) -> Result<(Tensor, Option<Tensor>)> {
        match self {
            Self::Linear(l) => Ok((l.weight().clone(), l.bias().cloned())),
            Self::Transposed { weight_t, bias } => Ok((weight_t.t()?.contiguous()?, bias.clone())),
        }
    }

    /// `in_features` (the contraction / last-dim) — the axis the quantize-skip predicate tests.
    fn in_features(&self) -> Result<usize> {
        match self {
            Self::Linear(l) => l.weight().dim(1),
            Self::Transposed { weight_t, .. } => weight_t.dim(0),
        }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Linear(l) => l.forward(x),
            Self::Transposed { weight_t, bias } => {
                let (in_dim, out_dim) = (weight_t.dim(0)?, weight_t.dim(1)?);
                let dims = x.dims().to_vec();
                let lead: usize = dims[..dims.len() - 1].iter().product();
                let y = x.contiguous()?.reshape((lead, in_dim))?.matmul(weight_t)?;
                let mut out_shape = dims[..dims.len() - 1].to_vec();
                out_shape.push(out_dim);
                let y = y.reshape(out_shape)?;
                match bias {
                    Some(b) => y.broadcast_add(b),
                    None => Ok(y),
                }
            }
        }
    }
}

/// The stored quantized weight. `QTensor` is not `Clone` and `QMatMul::from_qtensor` consumes it, so
/// the two [`MatmulStrategy`] arms keep the weight in the shape each forward needs, built once at fold
/// time: [`Self::Dequant`] holds the `QTensor` behind an `Arc` (dequantized to a dense matmul per
/// forward — the sc-7702 path); [`Self::Matmul`] holds the resident `QMatMul` (candle's int8 fast path,
/// itself `Arc`-backed). Both wrap the same GGUF blocks; only the forward compute differs. The `Arc`
/// makes [`QLinear`] `Clone` (SAM3's video model shares one quantized backbone across two heads, F-028).
#[derive(Clone)]
pub enum QuantWeight {
    /// The GGUF `QTensor` (dequantized to the activation dtype per forward). [`MatmulStrategy::DequantDense`].
    Dequant(std::sync::Arc<QTensor>),
    /// The resident `QMatMul` over the same GGUF blocks. [`MatmulStrategy::Int8Fast`].
    Matmul(QMatMul),
}

impl QuantWeight {
    /// The GGUF block type of the stored weight — used by tests to assert the packed `Q4_1` / folded
    /// `Q4_0`/`Q8_0` container survived the unification.
    pub fn dtype(&self) -> GgmlDType {
        match self {
            Self::Dequant(w) => w.dtype(),
            Self::Matmul(m) => match m {
                QMatMul::QTensor(w) => w.dtype(),
                // Dense/TensorF16 fallbacks never occur here (we always build from a QTensor).
                _ => GgmlDType::F32,
            },
        }
    }
}

/// How the quantized forward flattens the activation and casts back — the two remaining per-site
/// forward-shape drifts (F-025), kept explicit so each site stays byte-identical. Only consulted for
/// [`MatmulStrategy::Int8Fast`] (the dequant-dense path routes through `candle_nn::Linear`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QuantForward {
    /// Flatten all leading dims into one 2-D GEMM and reshape back to `[.., out_dim]` (SAM3 / SeedVR2),
    /// rather than feeding the raw activation to `QMatMul` (FLUX.2, whose activation is already 2-D).
    pub flatten_leading: bool,
    /// Cast the output back to the input dtype after the (f32) matmul + bias. SAM3 runs pure f32 so it
    /// skips the cast (`false`); FLUX.2 / SeedVR2 flow bf16 through the DiT and cast back (`true`).
    pub cast_back: bool,
}

/// A `Linear` projection that is **dense** or **GGUF-quantized** — the ONE shared seam every
/// quant-capable candle provider crate builds on (F-025 / sc-9005, consolidating the four drifted
/// copies in `candle-gen-flux2` / `candle-gen-lens` / `candle-gen-sam3` / `candle-gen-seedvr2`). The
/// quantized weight is a GGUF `QTensor` (`Q4_1` from a packed tier, or `Q4_0`/`Q8_0` from a load-time
/// fold); its two **load-bearing** behaviors — the matmul [`MatmulStrategy`] (dequant-dense vs int8
/// fast, the sc-7702 knob) and the dense weight [`DenseLinear`] layout — are explicit so no site's
/// numerics change. Built dense (`Self::linear*`), packed (`Self::from_packed*` / [`lin`]), or
/// packed-detected ([`Self::linear_detect`]); [`Self::quantize`] / [`Self::quantize_onto`] fold a dense
/// one and are a no-op on an already-quantized one. Every arm computes `x·Wᵀ + b`. `Clone` is cheap
/// (candle tensors, `QMatMul`, and the `Arc<QTensor>` dequant weight are all `Arc`-backed) — SAM3's
/// video model clones a once-quantized backbone to share it across two heads (F-028).
#[derive(Clone)]
pub enum QLinear {
    Dense(DenseLinear),
    Quantized {
        /// The GGUF-quantized weight (`Q4_1` from a packed tier, or `Q4_0`/`Q8_0` from a load-time
        /// quantize), stored in the form its [`MatmulStrategy`] needs ([`QuantWeight`]).
        weight: QuantWeight,
        /// The bias kept full-precision (`None` for bias-less projections).
        bias: Option<Tensor>,
        /// Per-site forward flatten / dtype-cast behavior (only consulted for [`MatmulStrategy::Int8Fast`]).
        fwd: QuantForward,
    },
}

// `QMatMul`/`QTensor` are not `Debug`, so summarize rather than derive — lets `QLinear` drop into the
// many `#[derive(Debug)]` vendored provider modules (e.g. the candle SDXL UNet, sc-9416) that hold it.
impl std::fmt::Debug for QLinear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dense(_) => f.write_str("QLinear::Dense"),
            Self::Quantized { weight, bias, .. } => f
                .debug_struct("QLinear::Quantized")
                .field("dtype", &weight.dtype())
                .field("bias", &bias.is_some())
                .finish(),
        }
    }
}

impl QLinear {
    /// A biased dense `[out, in]` projection from `vb` (`{prefix}.weight` + `{prefix}.bias`).
    pub fn linear(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self::Dense(DenseLinear::Linear(candle_nn::linear(
            in_dim, out_dim, vb,
        )?)))
    }

    /// A bias-less dense `[out, in]` projection from `vb` (`{prefix}.weight`).
    pub fn linear_no_bias(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self::Dense(DenseLinear::Linear(candle_nn::linear_no_bias(
            in_dim, out_dim, vb,
        )?)))
    }

    /// A dense projection wrapping an already-built [`DenseLinear`] — the entry point for the
    /// pre-transposed `[in, out]` layout (SAM3 / SeedVR2, sc-8997/F-017) and for tests.
    pub fn from_dense(dense: DenseLinear) -> Self {
        Self::Dense(dense)
    }

    /// **Packed-detecting** `[out, in]` loader: if `{base}.scales` is present in `vb` (a pre-quantized
    /// MLX tier), build a packed [`Self::Quantized`] straight from the packed parts on `vb`'s device via
    /// the shared [`lin`] — **no dense weight is materialized**. Otherwise the **dense** path is taken
    /// unchanged (`{base}.weight` [+ `{base}.bias`]), to be optionally folded later by
    /// [`Self::quantize`]. `base` is the full dotted key prefix (e.g. `attn.to_out.0`), so the
    /// `.scales`/`.biases` siblings survive any `to_out.0`-style key nesting — build the base string
    /// first, then detect (never `.pp()` past the scales sibling). Used by Lens / FLUX.2, whose packed
    /// forward is [`MatmulStrategy::DequantDense`] (the sc-7702 fix).
    pub fn linear_detect(
        in_dim: usize,
        out_dim: usize,
        vb: &VarBuilder,
        base: &str,
        bias: bool,
    ) -> Result<Self> {
        lin(vb, base, in_dim, out_dim, bias)
    }

    /// As [`Self::linear_detect`], but at an explicit MLX packed `group_size` (sc-9410 / sc-9416) — the
    /// packed branch repacks at `group_size` (read from the component `config.json`'s
    /// `quantization.group_size`); the dense branch is unchanged. The SDXL MLX tiers pack at the default
    /// 64, but a loader that reads the group from config threads it here rather than assume 64.
    pub fn linear_detect_gs(
        in_dim: usize,
        out_dim: usize,
        vb: &VarBuilder,
        base: &str,
        bias: bool,
        group_size: usize,
    ) -> Result<Self> {
        lin_gs(vb, base, in_dim, out_dim, bias, group_size)
    }

    /// Build a `Quantized` projection directly from an MLX packed triple (`wq` u32 codes + `scales` +
    /// `biases`) on `device` at the default group size 64 — Q4 via the lossless `Q4_1` repack, Q8 via
    /// dequant → `Q8_0` re-quant (bit-width inferred from the shapes). Uses [`MatmulStrategy::DequantDense`]
    /// (the sc-7702-safe forward every packed tier needs). `bias` is the optional dense `{base}.bias`,
    /// kept full-precision. No dense weight is ever materialized on the Q4 path (the whole point: the
    /// packed footprint lands on `device` directly). See [`Self::from_packed_gs`] for a non-64 group
    /// tier (boogu packs at 32).
    pub fn from_packed(
        wq: &Tensor,
        scales: &Tensor,
        biases: &Tensor,
        bias: Option<Tensor>,
        device: &Device,
    ) -> Result<Self> {
        Self::from_packed_gs(wq, scales, biases, bias, MLX_GROUP_SIZE, device)
    }

    /// As [`Self::from_packed`], but at an explicit MLX `group_size` (sc-9410) — the boogu tier packs
    /// at group 32 (the z-image / flux tiers at the default 64). The group size is not recoverable from
    /// the packed shapes alone (see [`repack`]), so a non-64 tier must pass it (from its component
    /// `config.json`'s `quantization.group_size`).
    pub fn from_packed_gs(
        wq: &Tensor,
        scales: &Tensor,
        biases: &Tensor,
        bias: Option<Tensor>,
        group_size: usize,
        device: &Device,
    ) -> Result<Self> {
        let weight = repack_packed_weight(wq, scales, biases, group_size, device)?;
        Ok(Self::Quantized {
            weight: QuantWeight::Dequant(std::sync::Arc::new(weight)),
            bias,
            fwd: QuantForward {
                flatten_leading: false,
                cast_back: true,
            },
        })
    }

    /// `x·Wᵀ + b`. `Dense` delegates to its [`DenseLinear`] layout. `Quantized` runs its
    /// [`MatmulStrategy`]: `DequantDense` dequantizes the weight (and bias) to the activation dtype and
    /// runs a dense matmul (sc-7702); `Int8Fast` casts the activation to f32 and routes it through
    /// candle's `QMatMul::forward` (with per-site leading-dim flatten / dtype cast-back).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(l) => l.forward(x),
            Self::Quantized {
                weight: QuantWeight::Dequant(weight),
                bias,
                ..
            } => {
                let in_dtype = x.dtype();
                let w = weight.dequantize(x.device())?.to_dtype(in_dtype)?;
                let bias = match bias {
                    Some(b) => Some(b.to_dtype(in_dtype)?),
                    None => None,
                };
                Linear::new(w, bias).forward(x)
            }
            Self::Quantized {
                weight: QuantWeight::Matmul(matmul),
                bias,
                fwd,
            } => {
                let in_dtype = x.dtype();
                let (mut y, out_shape) = if fwd.flatten_leading {
                    let dims = x.dims().to_vec();
                    let in_features = *dims.last().expect("linear input has rank >= 1");
                    let lead: usize = dims[..dims.len() - 1].iter().product();
                    let xf = x
                        .reshape((lead, in_features))?
                        .to_dtype(DType::F32)?
                        .contiguous()?;
                    let y2 = matmul.forward(&xf)?; // [lead, out]
                    let out_features = y2.dim(1)?;
                    let mut out_shape = dims;
                    *out_shape.last_mut().unwrap() = out_features;
                    (y2, Some(out_shape))
                } else {
                    let xf = x.to_dtype(DType::F32)?.contiguous()?;
                    (matmul.forward(&xf)?, None)
                };
                if let Some(shape) = out_shape {
                    y = y.reshape(shape)?;
                }
                if let Some(b) = bias {
                    y = y.broadcast_add(b)?;
                }
                if fwd.cast_back {
                    y = y.to_dtype(in_dtype)?;
                }
                Ok(y)
            }
        }
    }

    /// Fold a **dense** projection to `Q4_0`/`Q8_0` in place on its current device — **idempotent**: a
    /// no-op when already `Quantized` (from a load-time quantize or a packed-tier load), so a loader can
    /// packed-detect *and* keep an unconditional post-load `quantize` pass. Uses
    /// [`MatmulStrategy::DequantDense`] (the sc-7702-safe path; Lens's fold). The weight is quantized on
    /// the CPU and placed back on its original device via `QTensor::quantize_onto`; the bias stays
    /// full-precision. See [`Self::quantize_int8_fast`] for the int8-fast fold (FLUX.2 / SAM3 / SeedVR2)
    /// and [`Self::quantize_onto`] to land the fold on an explicit device.
    pub fn quantize(&mut self, quant: Quant) -> Result<()> {
        self.fold(
            quant,
            None,
            MatmulStrategy::DequantDense,
            QuantForward {
                flatten_leading: false,
                cast_back: true,
            },
            false,
        )
    }

    /// As [`Self::quantize`] but lands the folded `QTensor` on an explicit `device` (FLUX.2's
    /// CPU-stage-then-quantize-onto-GPU path, sc-7460) and uses the [`MatmulStrategy::Int8Fast`] forward
    /// (FLUX.2's f32 DiT/TE, GPU-validated). Idempotent on an already-quantized projection.
    pub fn quantize_onto(&mut self, quant: Quant, device: &Device) -> Result<()> {
        self.fold(
            quant,
            Some(device.clone()),
            MatmulStrategy::Int8Fast,
            QuantForward {
                flatten_leading: false,
                cast_back: true,
            },
            false,
        )
    }

    /// As [`Self::quantize`] (the sc-7702-safe [`MatmulStrategy::DequantDense`] arm) but lands the
    /// folded `QTensor` on an explicit `device` — SD3.5's CPU-stage-then-quantize-onto-GPU path
    /// (sc-8504), where the ~8 B dense MMDiT is staged in system RAM and each projection is folded
    /// *onto* the GPU so the dense projection weight never lands there. This is the fourth fold
    /// combination: [`Self::quantize`] is dequant-dense on the current device, [`Self::quantize_onto`]
    /// is int8-fast onto an explicit device, [`Self::quantize_int8_fast`] is int8-fast on the current
    /// device, and this is dequant-dense onto an explicit device. Feeds the SAME f32 CPU source to
    /// `QTensor::quantize_onto` as [`Self::quantize`], so the `Q4_0`/`Q8_0` blocks are bit-identical
    /// between the in-place and CPU-staged folds — only the dense on-device transient differs.
    /// Idempotent on an already-quantized projection.
    pub fn quantize_dequant_onto(&mut self, quant: Quant, device: &Device) -> Result<()> {
        self.fold(
            quant,
            Some(device.clone()),
            MatmulStrategy::DequantDense,
            QuantForward {
                flatten_leading: false,
                cast_back: true,
            },
            false,
        )
    }

    /// Fold to [`MatmulStrategy::Int8Fast`] in place on the current device, **skipping** any projection
    /// whose `in_features` is not a multiple of the 32-wide GGUF block (SAM3 / SeedVR2's reference
    /// predicate — it leaves e.g. SeedVR2's `vid_in.proj` in=132 and SAM3's `2→256`/`4→256`/`258→256`
    /// projections dense). `flatten_leading` / `cast_back` capture the site's forward shape (SAM3:
    /// flatten, no cast-back — pure f32; SeedVR2: flatten + cast-back — bf16 DiT). Idempotent.
    pub fn quantize_int8_fast(
        &mut self,
        quant: Quant,
        skip_indivisible: bool,
        flatten_leading: bool,
        cast_back: bool,
    ) -> Result<()> {
        self.fold(
            quant,
            None,
            MatmulStrategy::Int8Fast,
            QuantForward {
                flatten_leading,
                cast_back,
            },
            skip_indivisible,
        )
    }

    /// The shared dense→quantized fold behind [`Self::quantize`] / [`Self::quantize_onto`] /
    /// [`Self::quantize_int8_fast`]. `onto` picks the target device (`None` = the weight's current
    /// device); `skip_indivisible` leaves an `in_features % 32 != 0` projection dense (the SAM3/SeedVR2
    /// predicate). No-op when already `Quantized`.
    fn fold(
        &mut self,
        quant: Quant,
        onto: Option<Device>,
        strategy: MatmulStrategy,
        fwd: QuantForward,
        skip_indivisible: bool,
    ) -> Result<()> {
        let Self::Dense(dense) = self else {
            return Ok(());
        };
        if skip_indivisible && !dense.in_features()?.is_multiple_of(QUANT_BLOCK) {
            return Ok(());
        }
        let (w_out_in, bias) = dense.out_in_weight_bias()?;
        let device = onto.unwrap_or(w_out_in.device().clone());
        let w_cpu = w_out_in.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        // `QTensor::quantize_onto` reads the tensor's RAW backing storage (its `flatten_all` is a
        // zero-copy reshape on a strides-contiguous layout), not the logical view: on CUDA an
        // offset view silently quantizes the WRONG rows (flux2's ComfyUI in-place DiT loaded every
        // double block's to_k/to_v as copies of to_q — an incoherent render with per-layer weight
        // parity ~1.0, sc-11028); on the CPU any strict subview trips a length assert (a panic).
        // A dim-0 narrow keeps contiguous strides, so `.contiguous()` upstream is a no-op — the
        // view shape reaches here (and `Tensor::copy` is no help: it clones the whole backing
        // buffer, layout included). Materialize the source into owned zero-offset storage
        // (`force_contiguous`) unless it already covers EXACTLY its whole backing buffer.
        let covers_storage = {
            let (storage, layout) = w_cpu.storage_and_layout();
            let full_len = match &*storage {
                candle_core::Storage::Cpu(s) => s.as_slice::<f32>().map(<[f32]>::len).ok(),
                // Non-CPU cannot happen (`to_device` above); force the copy if it somehow does.
                _ => None,
            };
            layout.start_offset() == 0
                && w_cpu.is_contiguous()
                && full_len == Some(w_cpu.elem_count())
        };
        let w_cpu = if covers_storage {
            w_cpu
        } else {
            w_cpu.force_contiguous()?
        };
        let qtensor = QTensor::quantize_onto(&w_cpu, ggml_dtype(quant), &device)?;
        let weight = match strategy {
            MatmulStrategy::DequantDense => QuantWeight::Dequant(std::sync::Arc::new(qtensor)),
            MatmulStrategy::Int8Fast => QuantWeight::Matmul(QMatMul::from_qtensor(qtensor)?),
        };
        // The bias follows the weight's device and is promoted to f32 for the post-matmul add.
        let bias = match bias {
            Some(b) => Some(b.to_device(&device)?.to_dtype(DType::F32)?),
            None => None,
        };
        *self = Self::Quantized { weight, bias, fwd };
        Ok(())
    }

    /// Move a still-**dense** projection (weight + optional bias) to `device`, in place. A no-op when
    /// already quantized (that weight already lives on its device). Used by the CPU-staged quant path
    /// for the leaves it must keep dense — e.g. FLUX.2's control branch `control_img_in` (260
    /// in-features is not a multiple of the block 32, so it can't quantize) (sc-7460).
    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        if let Self::Dense(dense) = self {
            match dense {
                DenseLinear::Linear(l) => {
                    let w = l.weight().to_device(device)?;
                    let b = match l.bias() {
                        Some(b) => Some(b.to_device(device)?),
                        None => None,
                    };
                    *dense = DenseLinear::Linear(Linear::new(w, b));
                }
                DenseLinear::Transposed { weight_t, bias } => {
                    let w = weight_t.to_device(device)?;
                    let b = match bias {
                        Some(b) => Some(b.to_device(device)?),
                        None => None,
                    };
                    *dense = DenseLinear::Transposed {
                        weight_t: w,
                        bias: b,
                    };
                }
            }
        }
        Ok(())
    }

    /// Whether this projection loaded (or was folded) to a quantized weight.
    pub fn is_quantized(&self) -> bool {
        matches!(self, Self::Quantized { .. })
    }

    /// The [`MatmulStrategy`] of a quantized projection (`None` when dense) — used by tests to assert a
    /// site kept its intended (dequant-dense vs int8-fast) forward after the F-025 unification.
    pub fn matmul_strategy(&self) -> Option<MatmulStrategy> {
        match self {
            Self::Quantized {
                weight: QuantWeight::Dequant(_),
                ..
            } => Some(MatmulStrategy::DequantDense),
            Self::Quantized {
                weight: QuantWeight::Matmul(_),
                ..
            } => Some(MatmulStrategy::Int8Fast),
            Self::Dense(_) => None,
        }
    }
}

impl Module for QLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        QLinear::forward(self, x)
    }
}

/// A token embedding that is **dense** (the loaded `[vocab, hidden]` table) or **GGUF-quantized**
/// (the table stored as a `QTensor`, dequantized per forward). The TE `embed_tokens` is packed in the
/// MLX tiers, so the packed-load path needs the embedding analogue of [`QLinear`]. The forward is the
/// same index-select as `candle_nn::Embedding`.
pub enum QEmbedding {
    Dense(Embedding),
    Quantized {
        /// The GGUF-quantized `[vocab, hidden]` table; dequantized to `out_dtype` per forward, then
        /// index-selected.
        table: QTensor,
        hidden_size: usize,
        /// The dtype the dequantized table is cast to before index-select — the dense embedding
        /// table's dtype (i.e. `vb.dtype()`). Mirrors how [`QLinear::forward`] casts its dequantized
        /// weight to the activation dtype, so a packed bf16 text-encoder embedding yields bf16 rows
        /// exactly as the dense path would (dtype parity). Defaults to `F32` (the `QTensor`'s natural
        /// dequant dtype) for [`Self::from_packed`] callers that don't specify one.
        out_dtype: DType,
    },
}

impl QEmbedding {
    /// A dense `[vocab, hidden]` embedding from `vb` (`{prefix}.weight`).
    pub fn embedding(vocab: usize, hidden: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self::Dense(candle_nn::embedding(vocab, hidden, vb)?))
    }

    /// Build a `Quantized` embedding directly from an MLX packed triple on `device` (Q4 lossless
    /// repack / Q8 re-quant, as [`QLinear::from_packed`]). The `[vocab, hidden]` table's `hidden`
    /// (the last dim, the group axis) is `scales.cols · group_size`. The forward dequantizes to
    /// `F32` (the `QTensor`'s natural dtype); use [`Self::from_packed_dtype`] to match a non-f32
    /// dense-path output dtype (dtype parity).
    pub fn from_packed(
        wq: &Tensor,
        scales: &Tensor,
        biases: &Tensor,
        device: &Device,
    ) -> Result<Self> {
        Self::from_packed_dtype(wq, scales, biases, device, DType::F32)
    }

    /// As [`Self::from_packed_dtype`], but at an explicit MLX `group_size` (sc-9410, the boogu
    /// group-32 tier). See [`QLinear::from_packed_gs`].
    pub fn from_packed_dtype_gs(
        wq: &Tensor,
        scales: &Tensor,
        biases: &Tensor,
        device: &Device,
        out_dtype: DType,
        group_size: usize,
    ) -> Result<Self> {
        let table = repack_packed_weight(wq, scales, biases, group_size, device)?;
        let hidden = table.shape().dims()[1];
        Ok(Self::Quantized {
            table,
            hidden_size: hidden,
            out_dtype,
        })
    }

    /// As [`Self::from_packed`], but the forward dequantizes to `out_dtype` (the dense-path table
    /// dtype, `vb.dtype()`) — so a packed bf16 embedding yields bf16 rows exactly as the dense path
    /// would, mirroring [`QLinear::forward`]'s activation-dtype cast.
    pub fn from_packed_dtype(
        wq: &Tensor,
        scales: &Tensor,
        biases: &Tensor,
        device: &Device,
        out_dtype: DType,
    ) -> Result<Self> {
        Self::from_packed_dtype_gs(wq, scales, biases, device, out_dtype, MLX_GROUP_SIZE)
    }

    /// Index-select the embedding rows for `indexes`. Dense delegates to `candle_nn::Embedding`;
    /// quantized dequantizes the table and casts it to `out_dtype` (the dense-path table dtype) once,
    /// then index-selects — the same shape *and* dtype contract as the dense forward.
    pub fn forward(&self, indexes: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(e) => e.forward(indexes),
            Self::Quantized {
                table,
                hidden_size,
                out_dtype,
            } => {
                let w = table.dequantize(indexes.device())?.to_dtype(*out_dtype)?;
                Embedding::new(w, *hidden_size).forward(indexes)
            }
        }
    }

    pub fn is_quantized(&self) -> bool {
        matches!(self, Self::Quantized { .. })
    }
}

/// Repack an MLX packed triple into a resident [`QTensor`] on `device`: **Q4** via the lossless
/// `Q4_1` repack ([`repack_mlx_q4_to_q4_1`]); **Q8** via the exact-grid dequant + a `Q8_0` re-quant
/// (no affine 8-bit GGML container — the accepted sc-9085 double-quant, 0.56 % mean rel RMS). The
/// bit-width is inferred from the packed shapes ([`mlx_packed_bits`]).
///
/// **Public so a per-crate loader that can't use [`QLinear`] directly can still reuse the exact
/// Q4/Q8 dispatch (sc-9457).** The Lens gpt-oss encoder packs its fused MoE experts as a *3-D*
/// `[E, out, in/g]` affine triple that neither [`lin`] nor [`QLinear`] consumes (both are 2-D);
/// it slices each expert's `[out, in/g]` triple and calls this to build the resident `QTensor` it
/// wraps in a `QMatMul`, so the packed Q4→Q4_1 / Q8→Q8_0 conversion has a single implementation.
pub fn repack_packed_weight(
    wq: &Tensor,
    scales: &Tensor,
    biases: &Tensor,
    group_size: usize,
    device: &Device,
) -> Result<QTensor> {
    let (wq_cols, s_cols) = (wq.dims2()?.1, scales.dims2()?.1);
    match mlx_packed_bits_gs(wq_cols, s_cols, group_size) {
        4 => repack_mlx_q4_to_q4_1_gs(wq, scales, biases, group_size, device),
        8 => {
            let grid = dequant_mlx_q8_gs(wq, scales, biases, group_size)?;
            // `quantize_onto` needs a CPU source; `dequant_mlx_q8_gs` already returns on the CPU.
            QTensor::quantize_onto(&grid, GgmlDType::Q8_0, device)
        }
        b => candle_core::bail!(
            "unsupported MLX packed bit-width {b} (wq {wq_cols}, scales {s_cols}, group {group_size})"
        ),
    }
}

/// Load `{base}` as a [`QLinear`] — **packed** when the `{base}.scales` sibling is present in `vb`
/// (a pre-quantized MLX tier: build the quantized weight straight from the packed parts), else
/// **dense** (`{base}.weight`, path unchanged). `bias` additionally loads the dense `{base}.bias`
/// (distinct from the packed path's own `{base}.biases`, which is always loaded packed). The candle
/// twin of `mlx_gen::quant::lin`: one loader serves both a dense bf16 and a packed snapshot, with no
/// `quantization` manifest to read. `vb`'s dtype is the dense-path weight dtype; the packed path
/// builds on `vb`'s device.
pub fn lin(
    vb: &VarBuilder,
    base: &str,
    in_dim: usize,
    out_dim: usize,
    bias: bool,
) -> Result<QLinear> {
    lin_gs(vb, base, in_dim, out_dim, bias, MLX_GROUP_SIZE)
}

/// As [`lin`], but at an explicit MLX `group_size` (sc-9410) — the packed branch repacks at
/// `group_size` (the boogu tier's 32; z-image / flux default to 64). The dense branch is unchanged.
pub fn lin_gs(
    vb: &VarBuilder,
    base: &str,
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    group_size: usize,
) -> Result<QLinear> {
    let scales_key = format!("{base}.scales");
    if vb.contains_tensor(&scales_key) {
        let device = vb.device().clone();
        // The u32 packed codes must load at their native `U32` (a cast to the vb's float dtype would
        // reinterpret the bit-packed nibbles); the scales/biases upcast bf16 → f32 exactly.
        let wq = vb.get_unchecked_dtype(&format!("{base}.weight"), DType::U32)?;
        let scales = vb.get_unchecked_dtype(&scales_key, DType::F32)?;
        let biases = vb.get_unchecked_dtype(&format!("{base}.biases"), DType::F32)?;
        let bias = if bias {
            Some(vb.get_unchecked_dtype(&format!("{base}.bias"), vb.dtype())?)
        } else {
            None
        };
        return QLinear::from_packed_gs(&wq, &scales, &biases, bias, group_size, &device);
    }
    if bias {
        QLinear::linear(in_dim, out_dim, vb.pp(base))
    } else {
        QLinear::linear_no_bias(in_dim, out_dim, vb.pp(base))
    }
}

/// Load `{base}` as a [`QEmbedding`] — packed when `{base}.scales` is present, else dense (the
/// embedding analogue of [`lin`]; the candle twin of `mlx_gen::quant::embedding`). The TE
/// `embed_tokens` is packed in the MLX tiers, so this closes the packed-detect surface over both the
/// projections and the token embedding.
pub fn embedding(vb: &VarBuilder, base: &str, vocab: usize, hidden: usize) -> Result<QEmbedding> {
    embedding_gs(vb, base, vocab, hidden, MLX_GROUP_SIZE)
}

/// As [`embedding`], but at an explicit MLX `group_size` (sc-9410, the boogu group-32 tier).
pub fn embedding_gs(
    vb: &VarBuilder,
    base: &str,
    vocab: usize,
    hidden: usize,
    group_size: usize,
) -> Result<QEmbedding> {
    let scales_key = format!("{base}.scales");
    if vb.contains_tensor(&scales_key) {
        let device = vb.device().clone();
        let wq = vb.get_unchecked_dtype(&format!("{base}.weight"), DType::U32)?;
        let scales = vb.get_unchecked_dtype(&scales_key, DType::F32)?;
        let biases = vb.get_unchecked_dtype(&format!("{base}.biases"), DType::F32)?;
        // Dequantize the table to the dense-path table dtype (`vb.dtype()`), so a packed bf16
        // text-encoder embedding yields bf16 rows exactly as the dense path would (dtype parity).
        return QEmbedding::from_packed_dtype_gs(
            &wq,
            &scales,
            &biases,
            &device,
            vb.dtype(),
            group_size,
        );
    }
    QEmbedding::embedding(vocab, hidden, vb.pp(base))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::safetensors::MmapedSafetensors;

    // ---- helpers -----------------------------------------------------------------------------

    /// Cosine similarity over all elements (f64), the quant-parity metric.
    fn cosine(a: &Tensor, b: &Tensor) -> f32 {
        let a = a.to_dtype(DType::F32).unwrap().flatten_all().unwrap();
        let b = b.to_dtype(DType::F32).unwrap().flatten_all().unwrap();
        let a = a.to_vec1::<f32>().unwrap();
        let b = b.to_vec1::<f32>().unwrap();
        let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (x, y) in a.iter().zip(b.iter()) {
            dot += (*x as f64) * (*y as f64);
            na += (*x as f64) * (*x as f64);
            nb += (*y as f64) * (*y as f64);
        }
        (dot / (na.sqrt() * nb.sqrt() + 1e-12)) as f32
    }

    /// Test-side MLX packer: per-element 4-bit codes → MLX u32 words (LSB-first nibbles).
    fn pack_mlx_q4(codes: &[u8]) -> Vec<u32> {
        codes
            .chunks_exact(8)
            .map(|c| {
                c.iter()
                    .enumerate()
                    .fold(0u32, |acc, (i, &q)| acc | ((q as u32 & 0xF) << (4 * i)))
            })
            .collect()
    }

    /// Build a real MLX Q4 packed triple for an `[out, in]` weight (group 64) with f16-exact,
    /// per-group-distinct scales/biases and position-dependent codes — the packed-parts fixtures the
    /// `from_packed` / detect loaders consume, plus the exact affine grid they must reproduce.
    fn q4_fixture(out_dim: usize, in_dim: usize) -> (Tensor, Tensor, Tensor, Vec<f32>) {
        let dev = Device::Cpu;
        let codes: Vec<u8> = (0..out_dim * in_dim)
            .map(|i| ((i * 7 + i / 13) % 16) as u8)
            .collect();
        let groups = out_dim * in_dim / MLX_GROUP_SIZE;
        let scales: Vec<f32> = (0..groups).map(|g| 0.0625 * (g as f32 + 1.0)).collect();
        let biases: Vec<f32> = (0..groups).map(|g| -0.5 - 0.25 * g as f32).collect();
        let gpr = in_dim / MLX_GROUP_SIZE;
        let grid: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| {
                let (row, col) = (i / in_dim, i % in_dim);
                let g = row * gpr + col / MLX_GROUP_SIZE;
                scales[g] * codes[i] as f32 + biases[g]
            })
            .collect();
        let wq = Tensor::from_vec(pack_mlx_q4(&codes), (out_dim, in_dim / 8), &dev).unwrap();
        let s = Tensor::from_vec(scales, (out_dim, gpr), &dev).unwrap();
        let b = Tensor::from_vec(biases, (out_dim, gpr), &dev).unwrap();
        (wq, s, b, grid)
    }

    // ---- packed-vs-dense forward parity ------------------------------------------------------

    /// A packed-loaded `QLinear` forward matches a *dense* linear built from the SAME affine grid the
    /// pack represents — bit-exact (the repack is lossless, and both forwards dequant-to-dense-matmul).
    #[test]
    fn packed_qlinear_forward_matches_dense_grid() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (64, 128);
        let (wq, s, b, grid) = q4_fixture(out_dim, in_dim);

        let packed = QLinear::from_packed(&wq, &s, &b, None, &dev)?;
        assert!(packed.is_quantized());
        let dense = QLinear::Dense(DenseLinear::Linear(Linear::new(
            Tensor::from_vec(grid, (out_dim, in_dim), &dev)?,
            None,
        )));

        let x = Tensor::randn(0f32, 1f32, (4, in_dim), &dev)?;
        let cos = cosine(&packed.forward(&x)?, &dense.forward(&x)?);
        assert!(cos > 0.99999, "packed vs dense-grid cosine {cos:.6}");
        Ok(())
    }

    /// A packed-loaded `QEmbedding` reproduces the affine grid rows exactly (index-select over the
    /// dequantized table == the dense grid table).
    #[test]
    fn packed_qembedding_matches_dense_grid() -> Result<()> {
        let dev = Device::Cpu;
        let (vocab, hidden) = (32, 128);
        let (wq, s, b, grid) = q4_fixture(vocab, hidden);

        let packed = QEmbedding::from_packed(&wq, &s, &b, &dev)?;
        assert!(packed.is_quantized());
        let dense = QEmbedding::Dense(Embedding::new(
            Tensor::from_vec(grid, (vocab, hidden), &dev)?,
            hidden,
        ));

        let idx = Tensor::from_vec(vec![0u32, 5, 31, 12, 5], (5,), &dev)?;
        let (p, d) = (packed.forward(&idx)?, dense.forward(&idx)?);
        let dev_max = (p.sub(&d)?).abs()?.max_all()?.to_scalar::<f32>()?;
        assert_eq!(dev_max, 0.0, "packed embedding deviates from dense grid");
        Ok(())
    }

    /// The packed `QEmbedding` forward output dtype matches the dense embedding path's — a bf16
    /// dense table yields bf16 rows, so the packed path (loaded with the same `vb.dtype()`) must too.
    /// The dense path here goes through the same VarBuilder-detect loader (`embedding`), so it holds
    /// the `[base].weight` in the vb's bf16 dtype exactly as a real bf16 text-encoder would.
    #[test]
    fn packed_qembedding_forward_dtype_matches_dense() -> Result<()> {
        let dev = Device::Cpu;
        let (vocab, hidden) = (32, 128);
        let (wq, s, b, grid) = q4_fixture(vocab, hidden);

        // A packed table and a dense table, both written to safetensors and loaded through the
        // `embedding` detect-loader at bf16 vb dtype — the dense table is the reference dtype path.
        let mut map: std::collections::HashMap<String, Tensor> = std::collections::HashMap::new();
        map.insert("emb.weight".into(), wq);
        map.insert("emb.scales".into(), s);
        map.insert("emb.biases".into(), b);
        map.insert(
            "dense.weight".into(),
            Tensor::from_vec(grid, (vocab, hidden), &dev)?,
        );

        let tmp = std::env::temp_dir().join(format!(
            "sc9086_emb_dtype_{}.safetensors",
            std::process::id()
        ));
        candle_core::safetensors::save(&map, &tmp)?;
        // SAFETY: we just wrote this file and nothing else touches it during the test.
        let st = unsafe { MmapedSafetensors::new(&tmp)? };
        let vb = VarBuilder::from_backend(Box::new(st), DType::BF16, dev.clone());

        let packed = embedding(&vb, "emb", vocab, hidden)?;
        assert!(
            packed.is_quantized(),
            "`.scales` present ⇒ packed embedding"
        );
        let dense = embedding(&vb, "dense", vocab, hidden)?;
        assert!(!dense.is_quantized(), "no `.scales` ⇒ dense embedding");

        let idx = Tensor::from_vec(vec![0u32, 5, 31, 12, 5], (5,), &dev)?;
        let p = packed.forward(&idx)?;
        let d = dense.forward(&idx)?;
        assert_eq!(d.dtype(), DType::BF16, "dense bf16 embedding yields bf16");
        assert_eq!(
            p.dtype(),
            d.dtype(),
            "packed embedding forward dtype must match the dense path (dtype parity)"
        );

        std::fs::remove_file(&tmp).ok();
        Ok(())
    }

    // ---- idempotent quantize ------------------------------------------------------------------

    /// `quantize` is a no-op on an already-packed `QLinear` — it must NOT re-quantize (double-quantize)
    /// a weight that loaded packed. The stored `Q4_1` weight and the forward stay unchanged.
    #[test]
    fn quantize_is_noop_on_packed() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (64, 128);
        let (wq, s, b, _grid) = q4_fixture(out_dim, in_dim);

        let mut packed = QLinear::from_packed(&wq, &s, &b, None, &dev)?;
        let x = Tensor::randn(0f32, 1f32, (4, in_dim), &dev)?;
        let before = packed.forward(&x)?;

        // The weight is Q4_1 (packed), not the Q4_0 a load-time quantize would emit.
        match &packed {
            QLinear::Quantized { weight, .. } => assert_eq!(weight.dtype(), GgmlDType::Q4_1),
            _ => panic!("expected packed Quantized"),
        }

        packed.quantize(Quant::Q4)?; // must no-op, not re-quantize to Q4_0
        match &packed {
            QLinear::Quantized { weight, .. } => assert_eq!(
                weight.dtype(),
                GgmlDType::Q4_1,
                "quantize re-quantized a packed weight (Q4_1 → {:?})",
                weight.dtype()
            ),
            _ => panic!("quantize turned a packed linear dense"),
        }
        let after = packed.forward(&x)?;
        let dev_max = (before.sub(&after)?).abs()?.max_all()?.to_scalar::<f32>()?;
        assert_eq!(dev_max, 0.0, "no-op quantize changed the forward");
        Ok(())
    }

    /// **sc-11028 regression.** `QTensor::quantize{,_onto}` reads the tensor's RAW backing storage
    /// (its `flatten_all` is a zero-copy reshape on a strides-contiguous layout), ignoring a view's
    /// start offset. Folding a dim-0 narrow (a fused-qkv row chunk — `.contiguous()` is a no-op on
    /// it) therefore used to quantize the WRONG rows: flux2's ComfyUI in-place DiT loaded every
    /// double block's to_k/to_v as a copy of to_q — an incoherent render while per-layer weight
    /// parity read ~1.0 (the view API is offset-aware; the fold was not). The fold must
    /// materialize offset views into owned storage first: each folded chunk's forward must track
    /// the matching chunk of the dense fused weight (Q8, near-lossless) — chunks 1/2 fail without
    /// the guard, chunk 0 (offset 0) vacuously passes.
    #[test]
    fn fold_materializes_offset_views() -> Result<()> {
        let dev = Device::Cpu;
        let (d, in_dim) = (64usize, 32usize);
        let fused = Tensor::randn(0f32, 1f32, (3 * d, in_dim), &dev)?;
        let x = Tensor::randn(0f32, 1f32, (4, in_dim), &dev)?;
        for (i, name) in ["q", "k", "v"].iter().enumerate() {
            // The trap shape: a dim-0 narrow keeps contiguous strides, so `.contiguous()` no-ops
            // and the chunk stays a VIEW into `fused` at storage offset i·d·in_dim.
            let view = fused.narrow(0, i * d, d)?.contiguous()?;
            assert!(
                view.is_contiguous(),
                "narrow must look contiguous (the trap)"
            );
            let dense_ref = Linear::new(view.copy()?, None).forward(&x)?;
            for (label, onto) in [("quantize", false), ("quantize_onto", true)] {
                let mut lin = QLinear::Dense(DenseLinear::Linear(Linear::new(view.clone(), None)));
                if onto {
                    lin.quantize_onto(Quant::Q8, &dev)?;
                } else {
                    lin.quantize(Quant::Q8)?;
                }
                let cos = cosine(&lin.forward(&x)?, &dense_ref);
                assert!(
                    cos > 0.999,
                    "chunk {name} via {label}: cos {cos:.6} — an offset view quantized the wrong \
                     rows (sc-11028)"
                );
            }
        }
        Ok(())
    }

    /// The dense→quantize path stays idempotent too (a second `quantize` is a no-op, not a panic).
    #[test]
    fn quantize_dense_then_idempotent() -> Result<()> {
        let dev = Device::Cpu;
        let w = Tensor::randn(0f32, 1f32, (64, 32), &dev)?;
        let mut lin = QLinear::Dense(DenseLinear::Linear(Linear::new(w, None)));
        lin.quantize(Quant::Q8)?;
        assert!(lin.is_quantized());
        lin.quantize(Quant::Q8)?; // no-op, must not error
        assert!(matches!(lin, QLinear::Quantized { bias: None, .. }));
        Ok(())
    }

    /// `quantize_dequant_onto` folds a dense projection to the **dequant-dense** arm (sc-7702-safe)
    /// onto an explicit device — the SD3.5 CPU-stage path (sc-8504). It must NOT take the int8-fast
    /// arm `quantize_onto` uses, and its `Q4_0`/`Q8_0` blocks (and thus forward) must be bit-identical
    /// to the in-place [`QLinear::quantize`] fold of the same weight.
    #[test]
    fn quantize_dequant_onto_is_dequant_dense_and_matches_in_place() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (64, 32);
        let w = Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev)?;
        let b = Tensor::randn(0f32, 1f32, (out_dim,), &dev)?;

        for quant in [Quant::Q4, Quant::Q8] {
            // In-place dequant-dense fold (the reference).
            let mut in_place =
                QLinear::Dense(DenseLinear::Linear(Linear::new(w.clone(), Some(b.clone()))));
            in_place.quantize(quant)?;

            // CPU-stage → dequant-dense onto the SAME device.
            let mut staged =
                QLinear::Dense(DenseLinear::Linear(Linear::new(w.clone(), Some(b.clone()))));
            staged.quantize_dequant_onto(quant, &dev)?;
            assert_eq!(
                staged.matmul_strategy(),
                Some(MatmulStrategy::DequantDense),
                "quantize_dequant_onto must use the sc-7702-safe dequant-dense arm, not int8-fast"
            );

            // The forwards are bit-identical (same f32 CPU source → same GGUF blocks → same dequant).
            let x = Tensor::randn(0f32, 1f32, (4, in_dim), &dev)?;
            let a = in_place.forward(&x)?.flatten_all()?.to_vec1::<f32>()?;
            let c = staged.forward(&x)?.flatten_all()?.to_vec1::<f32>()?;
            for (p, r) in a.iter().zip(c.iter()) {
                assert_eq!(
                    p.to_bits(),
                    r.to_bits(),
                    "{quant:?} quantize_dequant_onto forward differs from in-place quantize"
                );
            }
        }
        Ok(())
    }

    // ---- PackedConfig detect ------------------------------------------------------------------

    #[test]
    fn packed_config_detects_quantization_block() {
        let packed = serde_json::json!({ "quantization": { "bits": 4, "group_size": 64 } });
        assert_eq!(
            PackedConfig::from_config(&packed),
            Some(PackedConfig {
                bits: 4,
                group_size: 64
            })
        );
        let dense = serde_json::json!({ "hidden_size": 2048 });
        assert_eq!(PackedConfig::from_config(&dense), None);
    }

    /// **`quantization.bits` present but `group_size` absent ⇒ default to 64, NOT silent dense
    /// (sc-9410).** A packed component missing `group_size` still stores u32 codes; degrading it to a
    /// dense read would load garbage. The existing group-64 / group-32 behavior stays byte-identical —
    /// only the *absent* case changed (it used to return `None`).
    #[test]
    fn packed_config_defaults_absent_group_size_to_64() {
        let no_gs = serde_json::json!({ "quantization": { "bits": 4 } });
        assert_eq!(
            PackedConfig::from_config(&no_gs),
            Some(PackedConfig {
                bits: 4,
                group_size: MLX_GROUP_SIZE as i32
            }),
            "absent group_size must default to the MLX group size (64), not degrade to dense"
        );
        // Explicit group sizes are unchanged.
        assert_eq!(
            PackedConfig::from_config(
                &serde_json::json!({ "quantization": { "bits": 4, "group_size": 32 } })
            ),
            Some(PackedConfig {
                bits: 4,
                group_size: 32
            })
        );
        // Still `None` when there is nothing marking it packed (no `bits`).
        assert_eq!(
            PackedConfig::from_config(&serde_json::json!({ "quantization": { "group_size": 64 } })),
            None
        );
    }

    // ---- packed-detect over a VarBuilder ------------------------------------------------------

    /// `lin` / `embedding` route to the packed path when the `.scales` sibling is present in the
    /// VarBuilder, and to the dense path otherwise — the one-call packed-detect contract. Uses an
    /// in-memory safetensors so no weights are needed.
    #[test]
    fn lin_and_embedding_packed_detect() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (64, 128);
        let (wq, s, b, _grid) = q4_fixture(out_dim, in_dim);
        let mut map: std::collections::HashMap<String, Tensor> = std::collections::HashMap::new();
        map.insert("proj.weight".into(), wq);
        map.insert("proj.scales".into(), s);
        map.insert("proj.biases".into(), b);
        // A dense sibling with no `.scales`.
        map.insert(
            "dense.weight".into(),
            Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev)?,
        );

        let tmp =
            std::env::temp_dir().join(format!("sc9086_detect_{}.safetensors", std::process::id()));
        candle_core::safetensors::save(&map, &tmp)?;
        // SAFETY: we just wrote this file and nothing else touches it during the test.
        let st = unsafe { MmapedSafetensors::new(&tmp)? };
        let vb = VarBuilder::from_backend(Box::new(st), DType::F32, dev.clone());

        let packed = lin(&vb, "proj", in_dim, out_dim, false)?;
        assert!(packed.is_quantized(), "`.scales` present ⇒ packed path");
        let dense = lin(&vb, "dense", in_dim, out_dim, false)?;
        assert!(!dense.is_quantized(), "no `.scales` ⇒ dense path unchanged");

        let emb = embedding(&vb, "proj", out_dim, in_dim)?;
        assert!(emb.is_quantized(), "`.scales` present ⇒ packed embedding");

        std::fs::remove_file(&tmp).ok();
        Ok(())
    }

    // ---- sc-7702 outlier / no-fast_mmq regression --------------------------------------------

    /// **sc-7702 regression (CUDA only).** The shared `QLinear` forward must stay accurate for
    /// activations with massive outliers — the gpt-oss text features (±10⁴) that made a Q4 DiT render
    /// solid black through candle's int8 `QMatMul` fast path (`fast_mmq`, batch > 8): that path
    /// quantizes the *activation* to per-32-element `q8_1`, so one outlier sets a block scale that
    /// rounds every co-located channel to **zero**. The packed-load `QLinear` must NOT take that path;
    /// it dequantizes the weight to a *dense* matmul, keeping the activation full-precision. Here each
    /// block's outlier sits on a **zero-weight** channel (carries no signal — the reference is built
    /// purely from its block-mates): the dequant path tracks the f32 reference, the raw int8 path
    /// collapses to ~0. A revert to `QMatMul::forward` inside `QLinear::forward` fails the `> 0.99`
    /// assert. This exercises the shared **dequant-on-forward** compute path in `QLinear::forward` —
    /// the byte-identical path a packed `Q4_1` load and a load-time `Q4_0` quantize both feed (the
    /// weight is dequantized to a dense matmul; the int8 activation fast path stays off), so a
    /// `Q4_0` `QLinear` built via the load-time `quantize` covers the packed forward too. Skips on
    /// CPU (the int8 MMVQ/MMQ path is CUDA-only).
    #[test]
    fn q4_packed_forward_survives_outlier_activations() -> Result<()> {
        use candle_core::quantized::QMatMul;
        let dev = match crate::default_device() {
            Ok(d) if !matches!(d, Device::Cpu) => d,
            _ => {
                eprintln!(
                    "SKIP q4_packed_forward_survives_outlier_activations: needs a CUDA device"
                );
                return Ok(());
            }
        };
        let (in_dim, out_dim, m, blk) = (256usize, 256usize, 64usize, 32usize); // m>8 ⇒ MMQ path

        // A dense weight whose per-32-block channel-0 is ZEROED (the outlier channel carries no
        // signal). Built on the CPU (the `quantize_onto` source requirement) then moved to `dev`.
        let mut w = Tensor::randn(0f32, 0.1f32, (out_dim, in_dim), &Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for o in 0..out_dim {
            for b in 0..(in_dim / blk) {
                w[o * in_dim + b * blk] = 0.0;
            }
        }
        let w_cpu = Tensor::from_vec(w, (out_dim, in_dim), &Device::Cpu)?;
        let w = w_cpu.to_device(&dev)?;

        // Activation: ~N(0,1) with a +30000 outlier on each block's channel-0.
        let mut x = Tensor::randn(0f32, 1f32, (m, in_dim), &dev)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for r in 0..m {
            for b in 0..(in_dim / blk) {
                x[r * in_dim + b * blk] = 30000.0;
            }
        }
        let x = Tensor::from_vec(x, (m, in_dim), &dev)?;
        let reference = x.matmul(&w.t()?)?; // f32 dense ground truth on `dev`

        // Dequant-on-forward path: a Q4 `QLinear` built via the load-time `quantize` — the shortest
        // way to a real QTensor, and the byte-identical `QLinear::forward` compute path a packed
        // `from_packed` load also feeds (both dequant the weight to a dense matmul).
        let qt = QTensor::quantize_onto(&w_cpu, GgmlDType::Q4_0, &dev)?;
        let mut lin = QLinear::Dense(DenseLinear::Linear(Linear::new(w, None)));
        lin.quantize(Quant::Q4)?;
        let dequant = lin.forward(&x)?;

        // Raw int8 `QMatMul` over the same Q4 weight — the path the black-render bug took.
        let int8 = QMatMul::from_qtensor(qt)?.forward(&x)?;

        let dq = cosine(&dequant, &reference);
        let q8 = cosine(&int8, &reference);
        eprintln!("sc-7702 packed outlier cosine: dequant(QLinear)={dq:.4}  int8(QMatMul)={q8:.4}");
        assert!(
            dq > 0.99,
            "packed QLinear Q4 forward must track f32 under outlier activations (cosine {dq:.4}) — \
             the int8 activation-quant path must not return inside QLinear::forward (sc-7702)"
        );
        assert!(
            dq > q8 + 0.05,
            "vacuous test: the raw int8 path ({q8:.4}) did not degrade vs dequant ({dq:.4})"
        );
        Ok(())
    }

    // ---- F-025 / sc-9005 unified-seam drifts --------------------------------------------------

    /// A dense [`DenseLinear::Linear`] forward equals a plain `candle_nn::Linear` exactly (the Lens /
    /// FLUX.2 dense layout is a pass-through).
    #[test]
    fn dense_linear_layout_equals_plain_linear() -> Result<()> {
        let dev = Device::Cpu;
        let (in_dim, out_dim) = (32, 64);
        let w = Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev)?;
        let b = Tensor::randn(0f32, 1f32, out_dim, &dev)?;
        let plain = Linear::new(w.clone(), Some(b.clone()));
        let ql = QLinear::Dense(DenseLinear::Linear(plain.clone()));

        let x = Tensor::randn(0f32, 1f32, (4, in_dim), &dev)?;
        let dev_max = (ql.forward(&x)?.sub(&plain.forward(&x)?)?)
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert_eq!(dev_max, 0.0, "Dense(Linear) must equal plain Linear");
        Ok(())
    }

    /// A pre-transposed [`DenseLinear::Transposed`] forward (SAM3 / SeedVR2 layout) equals the
    /// `[out,in]` `candle_nn::Linear` result over a rank-3 activation — the flatten-GEMM path is
    /// numerically the same `x·Wᵀ + b`.
    #[test]
    fn dense_transposed_layout_matches_linear_on_nd() -> Result<()> {
        let dev = Device::Cpu;
        let (in_dim, out_dim) = (32, 48);
        let w = Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev)?; // [out, in]
        let b = Tensor::randn(0f32, 1f32, out_dim, &dev)?;
        let transposed = QLinear::from_dense(DenseLinear::Transposed {
            weight_t: w.t()?.contiguous()?,
            bias: Some(b.clone()),
        });
        let plain = Linear::new(w, Some(b));

        // rank-3 activation (the SeedVR2/SAM3 case that goes through the leading-dim flatten).
        let x = Tensor::randn(0f32, 1f32, (2usize, 5usize, in_dim), &dev)?;
        let got = transposed.forward(&x)?;
        let want = plain.forward(&x)?;
        assert_eq!(got.dims(), want.dims());
        let cos = cosine(&got, &want);
        assert!(cos > 0.99999, "transposed vs linear cosine {cos:.6}");
        Ok(())
    }

    /// The int8-fast fold builds a [`QuantWeight::Matmul`] (candle's `QMatMul` path), quantizes/forwards
    /// near-losslessly at Q8, and reports [`MatmulStrategy::Int8Fast`] — the FLUX.2 / SAM3 / SeedVR2
    /// strategy, kept distinct from the dequant-dense (Lens) path.
    #[test]
    fn int8_fast_fold_quantizes_and_reports_strategy() -> Result<()> {
        let dev = Device::Cpu;
        let (in_dim, out_dim) = (32, 64);
        let w = Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev)?;
        let b = Tensor::randn(0f32, 1f32, out_dim, &dev)?;
        let mut lin = QLinear::Dense(DenseLinear::Linear(Linear::new(w, Some(b))));
        let x = Tensor::randn(0f32, 1f32, (4, in_dim), &dev)?;
        let dense = lin.forward(&x)?;

        lin.quantize_int8_fast(Quant::Q8, false, false, true)?;
        assert_eq!(lin.matmul_strategy(), Some(MatmulStrategy::Int8Fast));
        assert!(matches!(
            lin,
            QLinear::Quantized {
                weight: QuantWeight::Matmul(_),
                ..
            }
        ));
        let q = lin.forward(&x)?;
        assert_eq!(q.dims(), dense.dims());
        let cos = cosine(&dense, &q);
        assert!(cos > 0.999, "int8-fast Q8 cosine {cos:.5}");
        Ok(())
    }

    /// `quantize` (the Lens fold) yields the [`MatmulStrategy::DequantDense`] (sc-7702-safe) path.
    #[test]
    fn dequant_dense_fold_reports_strategy() -> Result<()> {
        let dev = Device::Cpu;
        let w = Tensor::randn(0f32, 1f32, (64, 32), &dev)?;
        let mut lin = QLinear::Dense(DenseLinear::Linear(Linear::new(w, None)));
        lin.quantize(Quant::Q4)?;
        assert_eq!(lin.matmul_strategy(), Some(MatmulStrategy::DequantDense));
        Ok(())
    }

    /// The SAM3/SeedVR2 skip predicate: an `in_features` not a multiple of 32 stays dense; a divisible
    /// one folds. Idempotent (a second fold is a no-op).
    #[test]
    fn int8_fast_skip_predicate_and_idempotent() -> Result<()> {
        let dev = Device::Cpu;
        // in=20, not a multiple of 32 → stays dense under the skip predicate.
        let odd = Tensor::randn(0f32, 1f32, (64, 20), &dev)?;
        let mut lin = QLinear::from_dense(DenseLinear::Transposed {
            weight_t: odd.t()?.contiguous()?,
            bias: None,
        });
        lin.quantize_int8_fast(Quant::Q8, true, true, false)?;
        assert!(matches!(lin, QLinear::Dense(_)), "in=20 must stay dense");

        // in=32 → folds; a second call is a no-op.
        let w = Tensor::randn(0f32, 1f32, (64, 32), &dev)?;
        let mut lin = QLinear::from_dense(DenseLinear::Transposed {
            weight_t: w.t()?.contiguous()?,
            bias: None,
        });
        lin.quantize_int8_fast(Quant::Q8, true, true, false)?;
        assert!(lin.is_quantized(), "in=32 must fold");
        lin.quantize_int8_fast(Quant::Q8, true, true, false)?; // idempotent
        assert_eq!(lin.matmul_strategy(), Some(MatmulStrategy::Int8Fast));
        Ok(())
    }

    /// `quantize_onto` lands the folded weight on an explicit device (here the CPU) with the int8-fast
    /// strategy and an f32-promoted bias — FLUX.2's CPU-stage → quantize-onto path.
    #[test]
    fn quantize_onto_explicit_device_int8_fast() -> Result<()> {
        let dev = Device::Cpu;
        let w = Tensor::randn(0f32, 1f32, (64, 32), &dev)?;
        let b = Tensor::randn(0f32, 1f32, 64, &dev)?;
        let mut lin = QLinear::Dense(DenseLinear::Linear(Linear::new(w, Some(b))));
        lin.quantize_onto(Quant::Q8, &dev)?;
        assert_eq!(lin.matmul_strategy(), Some(MatmulStrategy::Int8Fast));
        match &lin {
            QLinear::Quantized { bias: Some(b), .. } => {
                assert_eq!(b.dtype(), DType::F32, "bias promoted to f32")
            }
            _ => panic!("expected quantized with bias"),
        }
        // Idempotent.
        lin.quantize_onto(Quant::Q8, &dev)?;
        assert!(lin.is_quantized());
        Ok(())
    }

    /// `to_device` moves a dense `Transposed` projection and is a no-op on a quantized one.
    #[test]
    fn to_device_moves_dense_only() -> Result<()> {
        let dev = Device::Cpu;
        let w = Tensor::randn(0f32, 1f32, (64, 32), &dev)?;
        let mut lin = QLinear::from_dense(DenseLinear::Transposed {
            weight_t: w.t()?.contiguous()?,
            bias: None,
        });
        lin.to_device(&dev)?; // dense: rebuilds in place
        assert!(matches!(lin, QLinear::Dense(_)));
        lin.quantize(Quant::Q8)?;
        lin.to_device(&dev)?; // quantized: no-op, must not error
        assert!(lin.is_quantized());
        Ok(())
    }
}
