//! SD3.5 MMDiT quantization seam — **two routes to a quantized MMDiT**, both built on the shared
//! [`candle_gen::quant`] packed-load module (sc-9086) and candle-core's first-class GGUF quant:
//!
//! - **Packed tier (sc-9414, the fast path).** The hosted `SceneWorks/sd3.5-*-mlx` q4/q8 tiers store
//!   each quantized DiT `Linear` as the MLX packed triple (`{base}.weight` u32 codes, `{base}.scales`,
//!   `{base}.biases`; group size 64, the MLX default the shared loaders assume). [`QLinear::linear_detect`]
//!   packed-**detects** the `.scales` sibling and builds the quantized weight **straight from the packed
//!   parts** on the DiT device via the shared [`candle_gen::quant::lin`] loader (Q4 → `Q4_1` lossless
//!   repack, Q8 → `Q8_0` requant). **No dense bf16 weight is ever materialized** — the q4 MMDiT lands
//!   directly from the packed parts, with no dense staging *and* no load-then-quantize pass.
//!
//! - **Dense → quantize (the legacy path, unchanged; sc-7879).** When the snapshot is a dense bf16 tier
//!   (the stock diffusers snapshot; `.scales` absent), each DiT projection loads dense and
//!   [`crate::transformer::Sd3Transformer::quantize`] / [`quantize_onto`](crate::transformer::Sd3Transformer::quantize_onto)
//!   folds it to `Q4_0`/`Q8_0` in place **after** the (dense) weights — and any adapter merge — have
//!   loaded. Both fold entry points are **no-ops** on an already-packed projection (idempotent), so a
//!   packed-detect load and the unconditional post-load fold pass compose: an MLX-packed weight is never
//!   double-quantized.
//!
//! **The quantized matmul DEQUANTIZES the weight and runs a *dense* matmul — it does NOT take candle's
//! int8 `QMatMul` fast path (sc-7702).** That fast path (CUDA `fast_mmvq`/`fast_mmq`) quantizes the
//! *activation* to per-32-element `q8_1` blocks; a single large activation outlier sets a block's int8
//! scale and rounds every co-located channel to zero, which made the Lens Q4 DiT render solid black.
//! Dequantizing the weight to a dense matmul keeps the activation full-precision, so **uniform Q4
//! renders coherently** (GPU-verified on Blackwell for Lens). The load-time-quantized [`Self::Quantized`]
//! arm owns this behavior directly; the packed [`Self::Packed`] arm delegates to the shared
//! [`candle_gen::quant::QLinear`], which owns the identical dequant-on-forward compute path.
//!
//! **Quantize from CPU, store on the DiT's device.** The legacy fold's `QTensor::quantize_onto`
//! requires the source on the CPU, so each weight round-trips device→CPU→`quantize_onto(dev)`; the
//! resulting `QTensor` lives on the target device and the dense copy is dropped. The packed path skips
//! this entirely — the shared repack builds the `QTensor` straight from the packed parts on the DiT
//! device.
//!
//! **Two fold entry points, same numerics.** SD3.5 Large's DiT (~8 B params) *fits* the GPU dense
//! transiently, so the original [`QLinear::quantize`] builds dense on the target device and folds in
//! place. [`QLinear::quantize_onto`] (sc-8504, the FLUX.2-dev CPU-stage pattern) instead takes an
//! explicit target `device`: build the dense DiT on a **CPU** VarBuilder, then fold each projection
//! *onto* the GPU so the dense projection weight never lands on the GPU at all. Both feed the same f32
//! CPU source to `QTensor::quantize_onto`, so the `Q4_0`/`Q8_0` blocks are **bit-identical** between the
//! in-place and CPU-staged paths; the only difference is that the dense GPU transient is gone. Both are
//! no-ops on an already-packed projection, so the packed tier composes with either.
//!
//! **Which projections pack.** The MLX q4/q8 tier packs *every* DiT `Linear`, including the small AdaLN
//! modulation linears (`norm1.linear`, `norm1_context.linear`, `norm_out.linear`) and the
//! timestep/text embedders. The compute-heavy projections (attention q/k/v/out, the joint `add_*`, the
//! GELU MLP, the image-only `attn2`, `context_embedder`, `proj_out`) load as [`Self::Packed`] and keep
//! their Q4/Q8 footprint resident. The chaos-sensitive AdaLN / embedder linears load through
//! [`linear_detect_dense`] — packed-**detected** the same way, but **dequantized to a full-precision
//! dense [`Linear`]** so they stay a dense-typed leaf exactly as on the dense tier (matching the
//! deliberate "AdaLN modulation linears stay dense" choice of the legacy fold, which never enumerates
//! them). The per-head q/k RMSNorms, the patchify conv, and the learned pos-embed table carry no
//! `.scales` and stay dense as before.
//!
//! **Text encoders & VAE.** This seam is the **MMDiT** only. The three text encoders (CLIP-L, CLIP-G,
//! T5) and the VAE are stored **dense bf16** in every tier (byte-identical across `bf16/`, `q4/`, `q8/`
//! — the `sd3.5-*-mlx` converter only packs `transformer/`), so they are not touched here.

use candle_gen::candle_core::quantized::{GgmlDType, QTensor};
use candle_gen::candle_core::{DType, Device, Result, Tensor};
use candle_gen::candle_nn::{Conv2d, Linear, Module, RmsNorm, VarBuilder};
use candle_gen::gen_core::Quant;
use candle_gen::quant as shared;

/// The GGUF block type a [`Quant`] level maps to when quantizing a *dense* weight in place — `Q4_0` /
/// `Q8_0` (block size 32, the candle-core default GGUF quant). Every SD3.5 DiT projection contraction
/// is divisible by 32 (`inner_dim`: Large 2432, Medium 1536; `ff_hidden`: 9728 / 6144;
/// `joint_attention_dim` 4096), so the last-dim block check always passes. Shared single source of
/// truth with the Lens/FLUX.2 DiT quant. The **packed** path uses `Q4_1` instead (the affine container
/// the MLX tiers repack into losslessly — [`candle_gen::quant::repack`]).
pub fn ggml_dtype(quant: Quant) -> GgmlDType {
    match quant {
        Quant::Q4 => GgmlDType::Q4_0,
        Quant::Q8 => GgmlDType::Q8_0,
    }
}

/// Bytes-per-parameter of a GGUF block type, **including** the per-32-element block scale overhead.
/// `Q4_0` packs 32 weights into a 18-byte block (16 nibbles + one f16 scale) ⇒ 0.5625 B/param;
/// `Q8_0` packs 32 weights into a 34-byte block (32 int8 + one f16 scale) ⇒ 1.0625 B/param. Used by
/// [`crate::memory`] so the `minMemoryGb` estimate reflects the real on-device quantized footprint
/// (not the idealized 0.5 / 1.0).
pub fn bytes_per_param(quant: Quant) -> f64 {
    match quant {
        // 18 bytes / 32 weights.
        Quant::Q4 => 18.0 / 32.0,
        // 34 bytes / 32 weights.
        Quant::Q8 => 34.0 / 32.0,
    }
}

/// A Linear projection that is **dense** (the loaded bf16/f32 weight), **load-time GGUF-quantized** (the
/// `Q4_0`/`Q8_0` weight blocks + the bias, folded from a dense weight — sc-7879), or **packed** (loaded
/// directly from an MLX-packed tier through the shared [`candle_gen::quant::QLinear`] — sc-9414). Built
/// dense ([`Self::linear`] / [`Self::linear_no_bias`]) or packed-detected ([`Self::linear_detect`]);
/// [`Self::quantize`] / [`Self::quantize_onto`] transition a dense one to load-time-quantized in place
/// and are a **no-op** on an already-quantized *or* packed one. All three forwards compute `x·Wᵀ + b`
/// via a dense matmul (sc-7702).
pub enum QLinear {
    Dense(Linear),
    Quantized {
        /// The GGUF-quantized weight (`Q4_0`/`Q8_0`); dequantized to the activation dtype per forward.
        weight: QTensor,
        /// The bias kept full-precision (`None` for the bias-less projections, if any).
        bias: Option<Tensor>,
    },
    /// Loaded straight from an MLX-packed tier via the shared module — the sc-9414 fast path (no dense
    /// bf16 staging, no load-then-quantize). The inner [`shared::QLinear`] holds the resident
    /// `Q4_1`/`Q8_0` weight and **dequantizes-on-forward** into a dense matmul (sc-7702, *not* the int8
    /// `QMatMul` fast path).
    Packed(shared::QLinear),
}

impl QLinear {
    /// A biased dense `[out, in]` projection from `vb` (`{prefix}.weight` + `{prefix}.bias`).
    pub fn linear(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self::Dense(candle_gen::candle_nn::linear(
            in_dim, out_dim, vb,
        )?))
    }

    /// A bias-less dense `[out, in]` projection from `vb` (`{prefix}.weight`).
    pub fn linear_no_bias(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self::Dense(candle_gen::candle_nn::linear_no_bias(
            in_dim, out_dim, vb,
        )?))
    }

    /// **Packed-detecting** `[out, in]` loader (sc-9414): if `{base}.scales` is present in `vb` (a
    /// pre-quantized MLX tier), build a [`Self::Packed`] straight from the packed parts on `vb`'s device
    /// via the shared [`candle_gen::quant::lin`] — **no dense weight is materialized**. Otherwise the
    /// **dense** path is taken unchanged (`{base}.weight` [+ `{base}.bias`]), to be optionally folded
    /// later by [`Self::quantize`] / [`Self::quantize_onto`].
    ///
    /// `base` is the full dotted key prefix (e.g. `to_out.0`), so the `.scales`/`.biases` siblings
    /// survive any `to_out.0`-style key nesting — the key-remap trap: build the base string first, then
    /// detect (never `.pp()` past the scales sibling). The SD3.5 DiT tier packs at group size 64 (the
    /// MLX default the shared `lin` assumes), so no explicit group size is threaded.
    pub fn linear_detect(
        in_dim: usize,
        out_dim: usize,
        vb: &VarBuilder,
        base: &str,
        bias: bool,
    ) -> Result<Self> {
        if vb.contains_tensor(&format!("{base}.scales")) {
            return Ok(Self::Packed(shared::lin(vb, base, in_dim, out_dim, bias)?));
        }
        let sub = vb.pp(base);
        if bias {
            Self::linear(in_dim, out_dim, sub)
        } else {
            Self::linear_no_bias(in_dim, out_dim, sub)
        }
    }

    /// `x·Wᵀ + b`. All three arms run a **dense** matmul. `Dense` delegates to `candle_nn::Linear`;
    /// `Quantized` dequantizes its `Q4_0`/`Q8_0` weight (and bias) to the activation dtype and delegates
    /// likewise; `Packed` delegates to the shared dequant-on-forward `QLinear`. Dequantizing to a dense
    /// matmul — rather than candle's int8 `QMatMul` fast path — is the sc-7702 fix (see the module docs).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(l) => l.forward(x),
            Self::Quantized { weight, bias } => {
                let in_dtype = x.dtype();
                let w = weight.dequantize(x.device())?.to_dtype(in_dtype)?;
                let bias = match bias {
                    Some(b) => Some(b.to_dtype(in_dtype)?),
                    None => None,
                };
                Linear::new(w, bias).forward(x)
            }
            Self::Packed(l) => l.forward(x),
        }
    }

    /// Fold a dense projection to `Q4_0`/`Q8_0` in place. **Idempotent**: a no-op if already load-time
    /// -quantized *or* packed (an MLX-packed weight must not be double-quantized). The weight is
    /// quantized on the CPU and placed back on its **original** device via `QTensor::quantize_onto`; the
    /// bias is kept full-precision for the (dense) post-matmul add. The in-place path: the dense weight
    /// already lives on the (GPU) compute device.
    pub fn quantize(&mut self, quant: Quant) -> Result<()> {
        let device = match self {
            Self::Dense(l) => l.weight().device().clone(),
            // Already quantized (load-time fold) or packed (MLX tier) — no re-quantize.
            Self::Quantized { .. } | Self::Packed(_) => return Ok(()),
        };
        self.quantize_onto(quant, &device)
    }

    /// Fold a dense projection to `Q4_0`/`Q8_0` **onto `device`** in place. **Idempotent**: a no-op if
    /// already load-time-quantized *or* packed. The CPU-stage path (sc-8504): when the dense `QLinear`
    /// was built on a CPU VarBuilder, this round-trips the weight through the CPU (the `quantize_onto`
    /// source requirement) and lands the resulting `QTensor` on `device` (the GPU), so the dense
    /// projection never lives on the GPU. The bias is moved to `device` and kept full-precision.
    ///
    /// Numerically identical to [`Self::quantize`]: both feed the same f32 CPU source to
    /// `QTensor::quantize_onto`, so the `Q4_0`/`Q8_0` blocks are bit-for-bit the same regardless of
    /// where the dense weight started.
    pub fn quantize_onto(&mut self, quant: Quant, device: &Device) -> Result<()> {
        let Self::Dense(l) = self else {
            // Already quantized (load-time fold) or packed (MLX tier) — no re-quantize.
            return Ok(());
        };
        let w_cpu = l.weight().to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        let weight = QTensor::quantize_onto(&w_cpu, ggml_dtype(quant), device)?;
        let bias = match l.bias() {
            Some(b) => Some(b.to_device(device)?),
            None => None,
        };
        *self = Self::Quantized { weight, bias };
        Ok(())
    }

    /// Move a still-**dense** projection (weight + optional bias) to `device`, in place. A no-op once
    /// quantized *or* packed (the `QTensor` already lives on its device).
    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        if let Self::Dense(l) = self {
            let w = l.weight().to_device(device)?;
            let b = match l.bias() {
                Some(b) => Some(b.to_device(device)?),
                None => None,
            };
            *self = Self::Dense(Linear::new(w, b));
        }
        Ok(())
    }

    /// Whether this projection loaded directly from an MLX-packed tier (the sc-9414 no-dense-staging
    /// path). Distinguishes a packed load from a load-time `quantize` fold — used by the loaders/tests to
    /// assert a packed tier fired the packed path (and did not silently fall back to dense).
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn is_packed(&self) -> bool {
        matches!(self, Self::Packed(_))
    }
}

impl Module for QLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        QLinear::forward(self, x)
    }
}

/// **Packed-detecting dense** `[out, in]` loader (sc-9414) for the chaos-sensitive AdaLN modulation +
/// timestep/text embedder linears — the leaves the legacy fold deliberately keeps **dense** (it never
/// enumerates them for `quantize`). On a packed MLX tier these keys carry a `.scales` sibling (the
/// converter packs *every* DiT `Linear`), so this **dequantizes the packed parts to a full-precision
/// dense [`Linear`]** — matching the dense-tier type + the "AdaLN stays dense" numerics — instead of
/// keeping them resident as a `QTensor`. Absent `.scales`, the plain dense path is taken unchanged
/// (`{base}.weight` [+ `{base}.bias`]), byte-identical to the pre-sc-9414 loader.
///
/// The dequantized weight is built through the shared [`candle_gen::quant::lin`] repack (Q4 → lossless
/// `Q4_1`, Q8 → `Q8_0` requant, same as a packed projection) then materialized to `dtype` on `device`,
/// so an AdaLN linear holds the exact same values a packed projection at that key would — only stored
/// densely. `base` is the full dotted key prefix so the `.scales`/`.biases` siblings survive nesting.
pub fn linear_detect_dense(
    in_dim: usize,
    out_dim: usize,
    vb: &VarBuilder,
    base: &str,
    bias: bool,
) -> Result<Linear> {
    if !vb.contains_tensor(&format!("{base}.scales")) {
        let sub = vb.pp(base);
        return if bias {
            candle_gen::candle_nn::linear(in_dim, out_dim, sub)
        } else {
            candle_gen::candle_nn::linear_no_bias(in_dim, out_dim, sub)
        };
    }
    // Packed tier: build the quantized weight via the shared repack, then dequantize it to a dense
    // full-precision leaf at the dense-path dtype so the modulation linear stays a dense `Linear`.
    let device = vb.device().clone();
    let dtype = vb.dtype();
    let q = shared::lin(vb, base, in_dim, out_dim, bias)?;
    // The shared packed load is a `Quantized` projection whose weight is the dequant-dense
    // `QuantWeight::Dequant` arm (a packed tier is sc-7702-safe by construction; F-025 / sc-9005).
    let shared::QLinear::Quantized {
        weight: shared::QuantWeight::Dequant(weight),
        bias: qbias,
        ..
    } = q
    else {
        // `shared::lin` returns `Quantized`/`Dequant` whenever `.scales` is present (checked above), so
        // this is unreachable; fall back to a dense read rather than panicking if the contract changes.
        let sub = vb.pp(base);
        return if bias {
            candle_gen::candle_nn::linear(in_dim, out_dim, sub)
        } else {
            candle_gen::candle_nn::linear_no_bias(in_dim, out_dim, sub)
        };
    };
    let w = weight.dequantize(&device)?.to_dtype(dtype)?;
    let b = match qbias {
        Some(b) => Some(b.to_device(&device)?.to_dtype(dtype)?),
        None => None,
    };
    Ok(Linear::new(w, b))
}

/// Move a dense [`Linear`] (weight + optional bias) to `device` — the CPU-stage migration of a
/// dense-kept leaf (AdaLN modulation linears, the timestep/text embedders, the AdaLN-continuous head).
pub fn linear_to(l: &Linear, device: &Device) -> Result<Linear> {
    let w = l.weight().to_device(device)?;
    let b = match l.bias() {
        Some(b) => Some(b.to_device(device)?),
        None => None,
    };
    Ok(Linear::new(w, b))
}

/// Move a dense [`Conv2d`] (the patchify conv) to `device`, preserving its stride/padding config.
pub fn conv2d_to(c: &Conv2d, device: &Device) -> Result<Conv2d> {
    let w = c.weight().to_device(device)?;
    let b = match c.bias() {
        Some(b) => Some(b.to_device(device)?),
        None => None,
    };
    Ok(Conv2d::new(w, b, *c.config()))
}

/// Move a dense [`RmsNorm`] (the per-head q/k norms) to `device` at `eps`.
pub fn rms_norm_to(n: &RmsNorm, eps: f64, device: &Device) -> Result<RmsNorm> {
    Ok(RmsNorm::new(n.weight().to_device(device)?, eps))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::safetensors::MmapedSafetensors;
    use std::collections::HashMap;

    /// A dense `[out, in]` `QLinear` straight from explicit weight/bias tensors (no VarBuilder), so a
    /// test can capture the dense output and quantize the *same* weights for a 1:1 comparison.
    fn dense_from(w: &Tensor, b: Option<&Tensor>) -> QLinear {
        QLinear::Dense(Linear::new(w.clone(), b.cloned()))
    }

    /// Cosine similarity over all elements (f64).
    fn cosine(a: &Tensor, b: &Tensor) -> f32 {
        let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (x, y) in a.iter().zip(b.iter()) {
            dot += (*x as f64) * (*y as f64);
            na += (*x as f64) * (*x as f64);
            nb += (*y as f64) * (*y as f64);
        }
        (dot / (na.sqrt() * nb.sqrt() + 1e-12)) as f32
    }

    /// A `[64, 32]` projection (in=32 = one Q4_0/Q8_0 block per row) quantizes and forwards
    /// near-losslessly at Q8 / coherently at Q4 vs the dense f32 result — the per-linear analog of the
    /// full-DiT quant parity, on CPU with no weights.
    fn quant_roundtrip(quant: Quant, min_cos: f32) {
        let dev = Device::Cpu;
        let (in_dim, out_dim) = (32usize, 64usize);
        let w = Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev).unwrap();
        let b = Tensor::randn(0f32, 1f32, (out_dim,), &dev).unwrap();
        let mut lin = dense_from(&w, Some(&b));

        let x = Tensor::randn(0f32, 1f32, (4, in_dim), &dev).unwrap();
        let dense = lin.forward(&x).unwrap();

        lin.quantize(quant).unwrap();
        assert!(
            matches!(lin, QLinear::Quantized { .. }),
            "must be quantized"
        );
        let q = lin.forward(&x).unwrap();

        // Quantized output stays finite and tracks the dense reference.
        for v in q.flatten_all().unwrap().to_vec1::<f32>().unwrap() {
            assert!(v.is_finite(), "{quant:?} produced a non-finite output");
        }
        let cos = cosine(&dense, &q);
        assert!(cos > min_cos, "{quant:?} cosine {cos:.5} ≤ {min_cos}");
    }

    #[test]
    fn q8_is_near_lossless() {
        quant_roundtrip(Quant::Q8, 0.999);
    }

    #[test]
    fn q4_stays_coherent() {
        quant_roundtrip(Quant::Q4, 0.95);
    }

    /// `quantize` is idempotent — a second call on an already-quantized linear is a no-op, not a panic
    /// (the DiT's quantize pass runs uniformly over every `QLinear`).
    #[test]
    fn quantize_is_idempotent() {
        let dev = Device::Cpu;
        let w = Tensor::randn(0f32, 1f32, (64, 32), &dev).unwrap();
        let mut lin = dense_from(&w, None);
        lin.quantize(Quant::Q8).unwrap();
        lin.quantize(Quant::Q8).unwrap(); // no-op, must not error
        assert!(matches!(lin, QLinear::Quantized { bias: None, .. }));
    }

    /// The quantize→dequantize round-trip error is bounded: dequantizing the stored blocks recovers
    /// the dense weight within the block's quant step (Q8 tight, Q4 coarser).
    #[test]
    fn dequant_round_trip_error_is_bounded() {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (64usize, 64usize);
        let w = Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev).unwrap();
        for (quant, max_rel) in [(Quant::Q8, 0.05f32), (Quant::Q4, 0.30f32)] {
            let w_cpu = w.to_dtype(DType::F32).unwrap();
            let qt = QTensor::quantize_onto(&w_cpu, ggml_dtype(quant), &dev).unwrap();
            let recon = qt.dequantize(&dev).unwrap();
            let num = (&w - &recon)
                .unwrap()
                .sqr()
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap()
                .sqrt();
            let den = w
                .sqr()
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap()
                .sqrt();
            let rel = num / den;
            assert!(
                rel < max_rel,
                "{quant:?} relative recon error {rel:.4} ≥ {max_rel}"
            );
        }
    }

    /// Block-scale overhead is accounted for: Q4 ≈ 0.5625 B/param, Q8 ≈ 1.0625 B/param, and both are
    /// below bf16's 2.0.
    #[test]
    fn bytes_per_param_includes_block_scale() {
        assert!((bytes_per_param(Quant::Q4) - 0.5625).abs() < 1e-9);
        assert!((bytes_per_param(Quant::Q8) - 1.0625).abs() < 1e-9);
        assert!(bytes_per_param(Quant::Q4) < bytes_per_param(Quant::Q8));
        assert!(bytes_per_param(Quant::Q8) < 2.0);
    }

    // ---- packed-tier detect (sc-9414) -----------------------------------------------------------

    /// Build an MLX group-64 Q4 packed triple (`weight` u32 codes + f32 `scales`/`biases`) for a
    /// `[out_dim, in_dim]` linear, plus the exact dense affine grid it represents — the fixture the
    /// packed-detect path consumes and the reference it must reproduce. The scales/biases are chosen
    /// f16-exact (dyadic) so the `Q4_1` repack's f16 cast is lossless and the grid match is exact.
    /// Mirrors the shared/Lens packed fixture.
    fn q4_packed(out_dim: usize, in_dim: usize) -> (Tensor, Tensor, Tensor, Vec<f32>) {
        let dev = Device::Cpu;
        const G: usize = 64;
        let codes: Vec<u8> = (0..out_dim * in_dim)
            .map(|i| ((i * 7 + i / 13) % 16) as u8)
            .collect();
        let groups = out_dim * in_dim / G;
        let scales: Vec<f32> = (0..groups).map(|g| 0.0625 * (g as f32 + 1.0)).collect();
        let biases: Vec<f32> = (0..groups).map(|g| -0.5 - 0.25 * g as f32).collect();
        let gpr = in_dim / G;
        let grid: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| {
                let (row, col) = (i / in_dim, i % in_dim);
                let g = row * gpr + col / G;
                scales[g] * codes[i] as f32 + biases[g]
            })
            .collect();
        let words: Vec<u32> = codes
            .chunks_exact(8)
            .map(|c| {
                c.iter()
                    .enumerate()
                    .fold(0u32, |acc, (i, &q)| acc | ((q as u32 & 0xF) << (4 * i)))
            })
            .collect();
        let wq = Tensor::from_vec(words, (out_dim, in_dim / 8), &dev).unwrap();
        let s = Tensor::from_vec(scales, (out_dim, gpr), &dev).unwrap();
        let b = Tensor::from_vec(biases, (out_dim, gpr), &dev).unwrap();
        (wq, s, b, grid)
    }

    /// A VarBuilder over an in-memory packed safetensors map, exercising the real `contains_tensor` /
    /// `get_unchecked_dtype` detect path (not a hand-built enum). No external temp-file crate — writes
    /// to the system temp dir under a per-process unique name (the Lens-quant test pattern).
    fn vb_from_map(tag: &str, map: HashMap<String, Tensor>) -> VarBuilder<'static> {
        let tmp = std::env::temp_dir().join(format!(
            "sc9414_{tag}_{}_{}.safetensors",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        candle_gen::candle_core::safetensors::save(&map, &tmp).unwrap();
        // SAFETY: we just wrote this file and nothing else touches it during the test.
        let st = unsafe { MmapedSafetensors::new(&tmp).unwrap() };
        VarBuilder::from_backend(Box::new(st), DType::F32, Device::Cpu)
    }

    fn packed_map(
        base: &str,
        wq: &Tensor,
        s: &Tensor,
        b: &Tensor,
        bias: Option<&Tensor>,
    ) -> HashMap<String, Tensor> {
        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert(format!("{base}.weight"), wq.clone());
        map.insert(format!("{base}.scales"), s.clone());
        map.insert(format!("{base}.biases"), b.clone());
        if let Some(bi) = bias {
            map.insert(format!("{base}.bias"), bi.clone());
        }
        map
    }

    /// **Packed-detect fires on the SD3 key layout** (sc-9414). A `to_out.0`-nested packed linear (the
    /// key-remap trap: the `.scales`/`.biases` siblings must survive the `to_out.0` nesting) is detected
    /// and loaded as [`QLinear::Packed`] — NOT silently as dense — while a dense sibling with no
    /// `.scales` stays `Dense`. The packed forward matches a dense linear built from the exact affine
    /// grid the packed parts represent.
    #[test]
    fn packed_detect_fires_on_sd3_layout_and_leaves_dense_unchanged() {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (64usize, 128usize); // in_dim = 2 groups of 64
        let (wq, scales, biases, grid) = q4_packed(out_dim, in_dim);
        let bias = Tensor::randn(0f32, 1f32, (out_dim,), &dev).unwrap();

        // A representative nested key from the SD3 DiT surface, plus a dense sibling.
        let mut map = packed_map("attn.to_out.0", &wq, &scales, &biases, Some(&bias));
        map.insert(
            "attn.to_q.weight".into(),
            Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev).unwrap(),
        );
        map.insert(
            "attn.to_q.bias".into(),
            Tensor::randn(0f32, 1f32, (out_dim,), &dev).unwrap(),
        );
        let vb = vb_from_map("detect", map);
        let attn = vb.pp("attn");

        // `to_out.0` — packed-detected through the remapped base (never `.pp("0")` past the sibling).
        let lin = QLinear::linear_detect(in_dim, out_dim, &attn, "to_out.0", true).unwrap();
        assert!(
            lin.is_packed(),
            "packed tier must load as Packed, not a silent dense fallback"
        );

        // A dense sibling stays dense (path unchanged).
        let dense_sib = QLinear::linear_detect(in_dim, out_dim, &attn, "to_q", true).unwrap();
        assert!(
            matches!(dense_sib, QLinear::Dense(_)),
            "no `.scales` ⇒ dense"
        );
        assert!(!dense_sib.is_packed());

        // Reference dense linear from the exact grid + the same bias.
        let w_ref = Tensor::from_vec(grid, (out_dim, in_dim), &dev).unwrap();
        let ref_lin = Linear::new(w_ref, Some(bias));
        let x = Tensor::randn(0f32, 1f32, (4, in_dim), &dev).unwrap();
        let packed_out = lin.forward(&x).unwrap();
        let ref_out = ref_lin.forward(&x).unwrap();
        let cos = cosine(&ref_out, &packed_out);
        assert!(
            cos > 0.9999,
            "packed forward must reproduce the affine grid it repacks (cosine {cos:.6})"
        );
    }

    /// **`quantize` / `quantize_onto` are no-ops on a packed projection** (sc-9414): a packed-loaded
    /// `QLinear` must not be re-quantized by the unconditional post-load fold pass — it stays `Packed`,
    /// so an MLX-packed weight is never double-quantized. Covers BOTH fold entry points.
    #[test]
    fn quantize_is_noop_on_packed() {
        let (out_dim, in_dim) = (64usize, 128usize);
        let (wq, scales, biases, _grid) = q4_packed(out_dim, in_dim);
        let vb = vb_from_map(
            "noop",
            packed_map("context_embedder", &wq, &scales, &biases, None),
        );

        let mut lin =
            QLinear::linear_detect(in_dim, out_dim, &vb, "context_embedder", false).unwrap();
        assert!(lin.is_packed());
        lin.quantize(Quant::Q4).unwrap();
        assert!(
            lin.is_packed(),
            "quantize must not touch a packed projection"
        );
        lin.quantize_onto(Quant::Q8, &Device::Cpu).unwrap();
        assert!(
            lin.is_packed(),
            "quantize_onto must not touch a packed projection"
        );
    }

    /// **`linear_detect_dense` packed-detects but yields a full-precision dense [`Linear`]** (sc-9414) —
    /// the AdaLN / embedder leaves. On a packed tier it dequantizes the packed parts to a dense weight
    /// (so the modulation linear stays a dense-typed, chaos-safe leaf); its forward matches the exact
    /// affine grid the parts represent. Absent `.scales` it is a plain dense read.
    #[test]
    fn linear_detect_dense_dequantizes_packed_to_dense() {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (48usize, 64usize); // 1 group of 64
        let (wq, scales, biases, grid) = q4_packed(out_dim, in_dim);
        let bias = Tensor::randn(0f32, 1f32, (out_dim,), &dev).unwrap();
        let vb = vb_from_map(
            "adaln",
            packed_map("norm1.linear", &wq, &scales, &biases, Some(&bias)),
        );

        let lin = linear_detect_dense(in_dim, out_dim, &vb, "norm1.linear", true).unwrap();
        // A dense Linear (f32, the vb dtype), not a resident QTensor.
        assert_eq!(lin.weight().dims(), &[out_dim, in_dim]);
        assert_eq!(lin.weight().dtype(), DType::F32);

        let w_ref = Tensor::from_vec(grid, (out_dim, in_dim), &dev).unwrap();
        let ref_lin = Linear::new(w_ref, Some(bias));
        let x = Tensor::randn(0f32, 1f32, (4, in_dim), &dev).unwrap();
        let got = lin.forward(&x).unwrap();
        let want = ref_lin.forward(&x).unwrap();
        let cos = cosine(&want, &got);
        assert!(
            cos > 0.9999,
            "AdaLN dense-detect must match the grid (cosine {cos:.6})"
        );

        // Absent .scales ⇒ plain dense read.
        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert(
            "d.weight".into(),
            Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev).unwrap(),
        );
        map.insert(
            "d.bias".into(),
            Tensor::randn(0f32, 1f32, (out_dim,), &dev).unwrap(),
        );
        let vb2 = vb_from_map("adaln_dense", map);
        let lin2 = linear_detect_dense(in_dim, out_dim, &vb2, "d", true).unwrap();
        assert_eq!(lin2.weight().dims(), &[out_dim, in_dim]);
    }
}
