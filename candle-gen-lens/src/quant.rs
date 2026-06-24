//! Lens DiT load-time Q4/Q8 quantization seam (sc-5117) — the candle twin of
//! `mlx-gen-lens`'s `AdaptableLinear` quant path (sc-3175), built on **candle-core's first-class GGUF
//! quantization** (`QTensor`, the epic's "reuse the quant path, don't build a bespoke seam"). A [`QLinear`]
//! is a `Linear` that is **either** dense (bf16/f32) **or** GGUF-quantized; the DiT swaps its
//! compute-heavy projections to `QLinear` and [`crate::transformer::LensTransformer::quantize`] folds
//! each one to `Q4_0`/`Q8_0` in place after the (dense) weights — and any adapter merge — have loaded.
//!
//! **The quantized matmul dequantizes the weight and runs a *dense* matmul — it does NOT take
//! candle's int8 `QMatMul` fast path (sc-7702).** That fast path (CUDA `fast_mmvq`/`fast_mmq`)
//! quantizes the *activation* to `q8_1` per 32-element block; gpt-oss's massive outlier text
//! activations (±10⁴) blow out a block's int8 scale and zero the co-located channels, so the Q4 DiT
//! denoise diverges to NaN within a few steps — a solid-black render (Q8 only masks it with more
//! weight bits). Dequantizing the weight to a dense matmul keeps the activation full-precision, so
//! **uniform Q4 renders coherently** — GPU-verified on Blackwell: the *same* Q4 weights render black
//! through the int8 path and a clean image through this dequant path. Each forward dequantizes the
//! stored `Q4_0`/`Q8_0` blocks to the activation dtype on the fly, so the resident weight footprint
//! stays the small quantized one (the point of the story) while the matmul sees full-precision
//! activations. The surrounding DiT keeps flowing bf16 between layers exactly as the dense path does.
//!
//! **Quantize from CPU, store on the DiT's device.** `QTensor::quantize_onto` requires the source on
//! the CPU, so each weight round-trips device→CPU→`quantize_onto(dev)`; the resulting `QTensor` lives
//! on the original device (CPU or CUDA) and the dense copy is dropped. This mirrors mlx's build-dense-
//! then-`quantize()`-in-place ordering, so the transient load-time peak holds the dense DiT briefly
//! before the quantized blocks replace it (the steady-state resident footprint is the quantized one —
//! the point of the story).

use candle_gen::candle_core::quantized::{GgmlDType, QTensor};
use candle_gen::candle_core::{DType, Device, Result, Tensor};
use candle_gen::candle_nn::{Linear, Module, VarBuilder};
use candle_gen::gen_core::Quant;

/// The GGUF block type a [`Quant`] level maps to — `Q4_0` / `Q8_0` (block size 32, the candle-core
/// default GGUF quant). Every Lens DiT projection has an `in_features` divisible by 32
/// (128 / 1536 / 4096 / 11520), so the last-dim block check always passes. Shared with the gpt-oss
/// encoder quant (sc-5111) — its 2880-wide contraction is also ÷32 (but not ÷256, so only these
/// 32-block quants apply), the single source of truth for the family's `Quant → GgmlDType` mapping.
pub fn ggml_dtype(quant: Quant) -> GgmlDType {
    match quant {
        Quant::Q4 => GgmlDType::Q4_0,
        Quant::Q8 => GgmlDType::Q8_0,
    }
}

/// A Linear projection that is **dense** (the loaded bf16/f32 weight) or **GGUF-quantized** (the
/// `Q4_0`/`Q8_0` weight blocks + the bias, dequantized to a dense matmul each forward — sc-7702).
/// Built dense; [`Self::quantize`] transitions it to quantized in place. The dense and quantized
/// forwards are the same `x·Wᵀ + b`.
pub enum QLinear {
    Dense(Linear),
    Quantized {
        /// The GGUF-quantized weight (`Q4_0`/`Q8_0`); dequantized to the activation dtype per forward.
        weight: QTensor,
        /// The bias kept full-precision (`None` for the bias-less SwiGLU MLPs).
        bias: Option<Tensor>,
    },
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

    /// `x·Wᵀ + b`. Both arms run a **dense** matmul: `Dense` delegates to `candle_nn::Linear`;
    /// `Quantized` dequantizes its `Q4_0`/`Q8_0` weight (and bias) to the activation dtype and
    /// delegates likewise. Dequantizing to a dense matmul — rather than candle's int8 `QMatMul` fast
    /// path — is the sc-7702 fix: that path's per-32-block `q8_1` activation quant overflows on
    /// gpt-oss's outlier activations and the Q4 denoise diverges to NaN (see the module docs).
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
        }
    }

    /// Fold a dense projection to `Q4_0`/`Q8_0` in place (idempotent — a no-op if already quantized).
    /// The weight is quantized on the CPU and placed back on its original device via
    /// `QTensor::quantize_onto`; the bias is kept full-precision for the (dense) post-matmul add.
    pub fn quantize(&mut self, quant: Quant) -> Result<()> {
        let Self::Dense(l) = self else {
            return Ok(());
        };
        let device = l.weight().device().clone();
        let w_cpu = l.weight().to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        let weight = QTensor::quantize_onto(&w_cpu, ggml_dtype(quant), &device)?;
        let bias = l.bias().cloned();
        *self = Self::Quantized { weight, bias };
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A dense `[out, in]` `QLinear` straight from explicit weight/bias tensors (no VarBuilder), so a
    /// test can capture the dense output and quantize the *same* weights for a 1:1 comparison.
    fn dense_from(w: &Tensor, b: Option<&Tensor>) -> QLinear {
        QLinear::Dense(Linear::new(w.clone(), b.cloned()))
    }

    /// A `[64, 32]` projection (in=32 = one Q4_0/Q8_0 block per row) quantizes and forwards
    /// near-losslessly at Q8 / coherently at Q4 vs the dense f32 result — the per-linear analog of the
    /// full-DiT `dit_quant_parity` gate, runnable on CPU with no weights.
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

        let a = dense.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let c = q.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let (mut dot, mut na, mut nc) = (0f64, 0f64, 0f64);
        for (p, r) in a.iter().zip(c.iter()) {
            dot += (*p as f64) * (*r as f64);
            na += (*p as f64) * (*p as f64);
            nc += (*r as f64) * (*r as f64);
        }
        let cos = (dot / (na.sqrt() * nc.sqrt() + 1e-12)) as f32;
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

    /// Cosine similarity over all elements (f64), for the outlier regression below.
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

    /// **sc-7702 regression (CUDA only).** The quantized [`QLinear`] forward must stay accurate for
    /// activations with massive outliers — the gpt-oss text features (±10⁴) that made the Q4 DiT
    /// render solid black. candle's int8 `QMatMul` fast path (`fast_mmq`, batch > 8) quantizes the
    /// *activation* to per-32-element `q8_1`, so one outlier sets a block scale that rounds every
    /// co-located channel to **zero**. The fix dequantizes the weight to a *dense* matmul, keeping the
    /// activation full-precision. Here each block's outlier sits on a **zero-weight** channel (so it
    /// carries no signal — the reference output is built purely from its block-mates): the dequant
    /// path tracks the f32 reference, the int8 path collapses to ~0. A revert to the int8 path inside
    /// `QLinear::forward` fails the `> 0.99` assert.
    ///
    /// Skips on CPU (the int8 MMVQ/MMQ path is CUDA-only; CPU `QMatMul` already dequantizes).
    #[test]
    fn q4_forward_survives_outlier_activations() {
        use candle_gen::candle_core::quantized::QMatMul;
        let dev = match candle_gen::default_device() {
            Ok(d) if !matches!(d, Device::Cpu) => d,
            _ => {
                eprintln!("SKIP q4_forward_survives_outlier_activations: needs a CUDA device");
                return;
            }
        };
        let (in_dim, out_dim, m, blk) = (256usize, 256usize, 64usize, 32usize); // m>8 ⇒ MMQ path

        // Weight: small random, but every block's channel-0 (the outlier channel) is ZEROED — so the
        // outlier carries no signal and the reference output is built purely from its block-mates.
        let mut w = Tensor::randn(0f32, 0.1f32, (out_dim, in_dim), &dev)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for o in 0..out_dim {
            for b in 0..(in_dim / blk) {
                w[o * in_dim + b * blk] = 0.0;
            }
        }
        let w = Tensor::from_vec(w, (out_dim, in_dim), &dev).unwrap();

        // Activation: ~N(0,1), with a +30000 outlier on each block's channel-0.
        let mut x = Tensor::randn(0f32, 1f32, (m, in_dim), &dev)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for r in 0..m {
            for b in 0..(in_dim / blk) {
                x[r * in_dim + b * blk] = 30000.0;
            }
        }
        let x = Tensor::from_vec(x, (m, in_dim), &dev).unwrap();

        let reference = x.matmul(&w.t().unwrap()).unwrap(); // f32 dense ground truth

        // Production path: QLinear Q4 (dequant-on-forward).
        let mut lin = QLinear::Dense(Linear::new(w.clone(), None));
        lin.quantize(Quant::Q4).unwrap();
        let dequant = lin.forward(&x).unwrap();

        // Raw int8 `QMatMul` over the same Q4 weight — the path the black-render bug took.
        let w_cpu = w.to_device(&Device::Cpu).unwrap();
        let qt = QTensor::quantize_onto(&w_cpu, ggml_dtype(Quant::Q4), &dev).unwrap();
        let int8 = QMatMul::from_qtensor(qt).unwrap().forward(&x).unwrap();

        let dq = cosine(&dequant, &reference);
        let q8 = cosine(&int8, &reference);
        eprintln!("sc-7702 outlier cosine: dequant(QLinear)={dq:.4}  int8(QMatMul)={q8:.4}");
        assert!(
            dq > 0.99,
            "QLinear Q4 forward must track f32 under outlier activations (cosine {dq:.4}) — the int8 \
             activation-quant path must not return inside QLinear::forward (sc-7702)"
        );
        assert!(
            dq > q8 + 0.05,
            "vacuous test: the raw int8 path ({q8:.4}) did not degrade vs dequant ({dq:.4}). If a \
             candle bump fixed its q8_1 activation quant, the QLinear dequant workaround can be revisited"
        );
    }
}
