//! Lens DiT quantization seam — **two routes to a quantized DiT**, both built on the shared
//! [`candle_gen::quant`] packed-load module (sc-9086) and candle-core's first-class GGUF quant:
//!
//! - **Packed tier (sc-9413, the fast path).** The hosted `SceneWorks/lens-mlx` / `lens-turbo-mlx`
//!   q4/q8 tiers store each quantized DiT `Linear` as the MLX packed triple `{base}.weight` (u32
//!   codes) + `{base}.scales` + `{base}.biases` (group size 64, the default). [`QLinear::linear_detect`]
//!   packed-**detects** the `.scales` sibling and builds the quantized weight **straight from the
//!   packed parts** on the DiT device via the shared [`candle_gen::quant::lin`] loader (Q4 → `Q4_1`
//!   lossless repack, Q8 → `Q8_0` requant). **No dense bf16 weight is ever materialized** — the q4 DiT
//!   lands directly from the packed parts, with no dense staging *and* no load-then-quantize pass.
//!
//! - **Dense → quantize (the legacy path, unchanged; sc-5117).** When the snapshot is a dense bf16
//!   tier (the stock `SceneWorks/Lens` diffusers snapshot; `.scales` absent), each DiT projection loads
//!   dense and [`crate::transformer::LensTransformer::quantize`] folds it to `Q4_0`/`Q8_0` in place
//!   **after** the (dense) weights — and any adapter merge — have loaded. [`QLinear::quantize`] is a
//!   **no-op** on an already-packed projection (idempotent), so a packed-detect load and the
//!   unconditional post-load `quantize` pass compose: an MLX-packed weight is never double-quantized.
//!
//! **The quantized matmul dequantizes the weight and runs a *dense* matmul — it does NOT take candle's
//! int8 `QMatMul` fast path (sc-7702).** That fast path (CUDA `fast_mmvq`/`fast_mmq`) quantizes the
//! *activation* to `q8_1` per 32-element block; gpt-oss's massive outlier text activations (±10⁴) blow
//! out a block's int8 scale and zero the co-located channels, so the Q4 DiT denoise diverges to NaN
//! within a few steps — a solid-black render (Q8 only masks it with more weight bits). Dequantizing the
//! weight to a dense matmul keeps the activation full-precision, so **uniform Q4 renders coherently** —
//! GPU-verified on Blackwell. The load-time-quantized [`Self::Quantized`] arm owns this behavior
//! directly; the packed [`Self::Packed`] arm delegates to the shared [`candle_gen::quant::QLinear`],
//! which owns the identical dequant-on-forward compute path.
//!
//! **Quantize from CPU, store on the DiT's device.** The legacy fold's `QTensor::quantize_onto`
//! requires the source on the CPU, so each weight round-trips device→CPU→`quantize_onto(dev)`; the
//! resulting `QTensor` lives on the original device and the dense copy is dropped. The packed path
//! skips this entirely — the shared repack builds the `QTensor` straight from the packed parts on the
//! DiT device.
//!
//! **Text encoder & VAE.** This seam is the **DiT** only. The gpt-oss text encoder
//! ([`crate::text_encoder`]) has its own expert quant seam, and the Flux.2 VAE stays f32. The hosted
//! `SceneWorks/lens-mlx` tier packs its `text_encoder/` experts in a *3-D* per-expert MLX affine format
//! (`model.layers.*.mlp.experts.{gate_up_proj,down_proj}` + `.scales`/`.biases`), which is **not** the
//! 2-D `Linear` shape the shared loaders consume — so the encoder carries a dedicated 3-D fused-expert
//! packed loader ([`crate::text_encoder::GptOssTextEncoder::new_quant`], sc-9457) that slices each
//! expert and delegates the Q4→`Q4_1` / Q8→`Q8_0` repack to the shared
//! [`candle_gen::quant::repack_packed_weight`] seam. With that landed, a pure `lens-mlx` snapshot now
//! loads packed **end-to-end** (DiT + encoder + VAE) — the encoder no longer needs the dense
//! `SceneWorks/Lens` MXFP4 snapshot (which stays the load path for the dense diffusers tier).

use candle_gen::candle_core::quantized::{GgmlDType, QTensor};
use candle_gen::candle_core::{DType, Device, Result, Tensor};
use candle_gen::candle_nn::{Linear, Module, VarBuilder};
use candle_gen::gen_core::Quant;
use candle_gen::quant as shared;

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

/// A Linear projection that is **dense** (the loaded bf16/f32 weight), **load-time GGUF-quantized**
/// (the `Q4_0`/`Q8_0` weight blocks + the bias, folded from a dense weight — sc-5117), or **packed**
/// (loaded directly from an MLX-packed tier through the shared [`candle_gen::quant::QLinear`] — sc-9413).
/// Built dense ([`Self::linear`] / [`Self::linear_no_bias`]) or packed-detected ([`Self::linear_detect`]);
/// [`Self::quantize`] transitions a dense one to load-time-quantized in place and is a **no-op** on an
/// already-quantized *or* packed one. All three forwards compute `x·Wᵀ + b` via a dense matmul (sc-7702).
pub enum QLinear {
    Dense(Linear),
    Quantized {
        /// The GGUF-quantized weight (`Q4_0`/`Q8_0`); dequantized to the activation dtype per forward.
        weight: QTensor,
        /// The bias kept full-precision (`None` for the bias-less SwiGLU MLPs).
        bias: Option<Tensor>,
    },
    /// Loaded straight from an MLX-packed tier via the shared module — the sc-9413 fast path (no dense
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

    /// **Packed-detecting** `[out, in]` loader (sc-9413): if `{base}.scales` is present in `vb` (a
    /// pre-quantized MLX tier), build a [`Self::Packed`] straight from the packed parts on `vb`'s device
    /// via the shared [`candle_gen::quant::lin`] — **no dense weight is materialized**. Otherwise the
    /// **dense** path is taken unchanged (`{base}.weight` [+ `{base}.bias`]), to be optionally folded
    /// later by [`Self::quantize`].
    ///
    /// `base` is the full dotted key prefix (e.g. `to_out.0`), so the `.scales`/`.biases` siblings
    /// survive any `to_out.0`-style key nesting — the key-remap trap: build the base string first, then
    /// detect (never `.pp()` past the scales sibling). The Lens DiT tier packs at group size 64 (the MLX
    /// default the shared `lin` assumes), so no explicit group size is threaded.
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
    /// matmul — rather than candle's int8 `QMatMul` fast path — is the sc-7702 fix: that path's
    /// per-32-block `q8_1` activation quant overflows on gpt-oss's outlier activations and the Q4 denoise
    /// diverges to NaN (see the module docs).
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
    /// quantized on the CPU and placed back on its original device via `QTensor::quantize_onto`; the bias
    /// is kept full-precision for the (dense) post-matmul add.
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

    /// Whether this projection holds a quantized weight (load-time-folded or packed), i.e. no dense
    /// weight.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn is_quantized(&self) -> bool {
        matches!(self, Self::Quantized { .. } | Self::Packed(_))
    }

    /// Whether this projection loaded directly from an MLX-packed tier (the sc-9413 no-dense-staging
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

    /// Cosine similarity over all elements (f64).
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

    // ---- sc-9413 packed-detect (from the MLX-packed tier, no dense staging) -------------------

    /// Test-side MLX Q4 packer: per-element 4-bit codes → MLX u32 words (LSB-first nibbles), group 64
    /// (the Lens DiT tier's group size). Returns `(wq [out, in/8] u32, scales [out, in/64], biases
    /// [out, in/64], affine grid [out, in])` — the exact packed-parts fixture the detect loaders consume
    /// plus the affine grid they reproduce.
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

    /// **Packed-detect fires on the Lens DiT key layout, incl. the `attn.to_out.0` nesting (sc-9413).**
    /// Writes a safetensors mimicking the real Lens DiT packed layout — an `attn.to_out.0` group-64
    /// packed triple (the key remap that would silently fall back to dense if the loader `.pp()`'d past
    /// the `.scales` sibling) and a dense `attn.img_qkv` sibling — and loads both through
    /// `linear_detect`. The `.scales`/`.biases` siblings must survive the `to_out.0` base string,
    /// `to_out.0` must load `Packed`, the dense sibling stays `Dense`, and the packed forward must
    /// reproduce the affine grid bit-exactly (not a silent dense fallback).
    #[test]
    fn linear_detect_fires_on_to_out_remap_and_leaves_dense_unchanged() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (64usize, 128usize);
        let (wq, s, b, grid) = q4_packed(out_dim, in_dim);

        let mut map: HashMap<String, Tensor> = HashMap::new();
        // The `attn.to_out.0` packed triple (the nested key the Lens DiT threads as a single base).
        map.insert("attn.to_out.0.weight".into(), wq);
        map.insert("attn.to_out.0.scales".into(), s);
        map.insert("attn.to_out.0.biases".into(), b);
        // A dense sibling (`img_qkv`) with no `.scales` → the dense path must stay unchanged.
        map.insert(
            "attn.img_qkv.weight".into(),
            Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev)?,
        );

        let tmp =
            std::env::temp_dir().join(format!("sc9413_detect_{}.safetensors", std::process::id()));
        candle_gen::candle_core::safetensors::save(&map, &tmp)?;
        // SAFETY: we just wrote this file and nothing else touches it during the test.
        let st = unsafe { MmapedSafetensors::new(&tmp)? };
        let vb = VarBuilder::from_backend(Box::new(st), DType::F32, dev.clone());
        let attn = vb.pp("attn");

        // `to_out.0` — packed-detected through the remapped base (never `.pp("0")` past the sibling).
        let packed = QLinear::linear_detect(in_dim, out_dim, &attn, "to_out.0", false)?;
        assert!(packed.is_packed(), "`.scales` under to_out.0 ⇒ packed load");
        assert!(packed.is_quantized());

        // `img_qkv` — dense (no `.scales`), path unchanged.
        let dense = QLinear::linear_detect(in_dim, out_dim, &attn, "img_qkv", false)?;
        assert!(!dense.is_packed(), "no `.scales` ⇒ dense path unchanged");
        assert!(matches!(dense, QLinear::Dense(_)));

        // The packed forward reproduces the affine grid (bit-exact repack + dequant-on-forward).
        let grid_lin = QLinear::Dense(Linear::new(
            Tensor::from_vec(grid, (out_dim, in_dim), &dev)?,
            None,
        ));
        let x = Tensor::randn(0f32, 1f32, (4, in_dim), &dev)?;
        let cos = cosine(&packed.forward(&x)?, &grid_lin.forward(&x)?);
        assert!(cos > 0.99999, "packed vs affine-grid cosine {cos:.6}");

        std::fs::remove_file(&tmp).ok();
        Ok(())
    }

    /// `quantize` is a **no-op** on a packed projection — an MLX-packed weight must never be
    /// double-quantized. The projection stays `Packed` and its forward is unchanged, so a loader can
    /// packed-detect *and* keep the unconditional `LensTransformer::quantize` pass (sc-9413).
    #[test]
    fn quantize_is_noop_on_packed() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (64usize, 128usize);
        let (wq, s, b, _grid) = q4_packed(out_dim, in_dim);

        let packed = shared::QLinear::from_packed(&wq, &s, &b, None, &dev)?;
        let mut lin = QLinear::Packed(packed);
        let x = Tensor::randn(0f32, 1f32, (4, in_dim), &dev)?;
        let before = lin.forward(&x)?;

        lin.quantize(Quant::Q4)?; // must no-op, not re-quantize
        assert!(lin.is_packed(), "quantize changed a packed projection");
        let after = lin.forward(&x)?;
        let dev_max = (before.sub(&after)?).abs()?.max_all()?.to_scalar::<f32>()?;
        assert_eq!(dev_max, 0.0, "no-op quantize changed the packed forward");
        Ok(())
    }

    /// A packed-loaded `QLinear` forward matches a load-time-quantized fold of the SAME affine grid the
    /// pack represents — the packed `Q4_1` repack and the legacy `Q4_0` fold both feed the shared
    /// dequant-on-forward compute, so both track the dense grid closely. Confirms the packed path is a
    /// genuine quantized load (not the dense weight) and interoperable with the legacy fold's parity.
    #[test]
    fn packed_matches_legacy_fold_of_same_grid() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (64usize, 128usize);
        let (wq, s, b, grid) = q4_packed(out_dim, in_dim);

        let packed = QLinear::Packed(shared::QLinear::from_packed(&wq, &s, &b, None, &dev)?);
        assert!(packed.is_packed());

        let mut folded = QLinear::Dense(Linear::new(
            Tensor::from_vec(grid, (out_dim, in_dim), &dev)?,
            None,
        ));
        folded.quantize(Quant::Q4)?;

        let x = Tensor::randn(0f32, 1f32, (4, in_dim), &dev)?;
        let cos = cosine(&packed.forward(&x)?, &folded.forward(&x)?);
        assert!(cos > 0.99, "packed vs legacy-fold cosine {cos:.5}");
        Ok(())
    }
}
