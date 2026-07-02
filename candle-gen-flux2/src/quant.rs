//! FLUX.2 Q4/Q8 quantization seam — the candle twin of `mlx-gen-flux2`'s packed-weight path (sc-5917),
//! built on candle-core's GGUF quant. A [`QLinear`] is a `Linear` that is **dense** (f32),
//! **load-time-quantized** (a `QMatMul` folded from a dense weight), or **packed** (loaded directly
//! from an MLX-packed tier through the shared [`candle_gen::quant`] module — sc-9086/sc-9087).
//!
//! **Two load routes to a quantized dev DiT/TE (both avoid a dense GPU copy):**
//!
//! - **Packed tier (sc-9087, the fast path).** The hosted `SceneWorks/flux2-dev-mlx` q4/q8 tiers store
//!   each quantized projection as the MLX packed triple `{base}.weight` (u32 codes) + `{base}.scales` +
//!   `{base}.biases`. [`QLinear::linear_detect`] packed-**detects** the `.scales` sibling and builds the
//!   quantized weight **straight from the packed parts** on the GPU via the shared
//!   [`candle_gen::quant::lin`] loader (Q4→`Q4_1` lossless, Q8→`Q8_0` requant). **No dense bf16 weight
//!   is ever materialized** — this kills the ~105 GB dense CPU-staging peak (the whole point of sc-9087;
//!   the q4 DiT lands ~18 GB directly). The packed forward **dequantizes the weight into a dense matmul**
//!   (sc-7702), *not* candle's int8 `QMatMul` fast path.
//!
//! - **Dense → quantize-onto (the legacy path, unchanged).** When the snapshot is a dense bf16/f32 tier
//!   (`.scales` absent — klein, or a dev fixture, or the pre-tier dev weights), the loader stages the
//!   dense weights in **system RAM** and [`QLinear::quantize_onto`] folds each projection **onto** the
//!   GPU via [`QTensor::quantize_onto`] (which needs a CPU source). This is the ~105 GB peak the packed
//!   path replaces; it is retained for dense tiers and small fixtures.
//!
//! [`QLinear::quantize_onto`] is a **no-op** on an already-packed projection (idempotent), so a loader
//! can packed-detect *and* keep the unconditional post-load `quantize` pass — the two compose. The small
//! dense leaves that stay full precision (RMSNorms, the token embedding) are moved to the GPU alongside
//! via [`rms_norm_to`] / [`Tensor::to_device`].
//!
//! **The quantized matmul runs in f32.** candle's CPU `QMatMul` and the CUDA dmmv fallback need an
//! f32 activation; FLUX.2 already flows f32, so the cast is a no-op here. The bias (when present) is
//! kept f32 and added after the matmul.

use candle_gen::candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_gen::candle_core::{DType, Device, Result, Tensor};
use candle_gen::candle_nn::{Linear, Module, RmsNorm, VarBuilder};
use candle_gen::gen_core::Quant;
use candle_gen::quant as shared;

/// The GGUF block type a [`Quant`] level maps to — `Q4_0` / `Q8_0` (block size 32). Every dev TE/DiT
/// projection's `in_features` is divisible by 32 (128 / 256 / 4096 / 5120 / 6144 / 15360 / 24576 /
/// 32768), so the last-dim block check always passes. Shared mapping with the Lens DiT quant (sc-5117).
pub fn ggml_dtype(quant: Quant) -> GgmlDType {
    match quant {
        Quant::Q4 => GgmlDType::Q4_0,
        Quant::Q8 => GgmlDType::Q8_0,
    }
}

/// A Linear projection that is **dense** (the loaded f32 weight), **load-time GGUF-quantized** (a
/// `QMatMul` folded from a dense weight + full-precision bias), or **packed** (loaded directly from an
/// MLX-packed tier through the shared [`candle_gen::quant::QLinear`], sc-9087). Built dense
/// ([`Self::linear`]) or packed-detected ([`Self::linear_detect`]); [`Self::quantize_onto`] transitions
/// a dense one to load-time-quantized on a target device and is a **no-op** on an already-packed one.
/// All three forwards compute `x·Wᵀ + b`.
pub enum QLinear {
    Dense(Linear),
    Quantized {
        matmul: QMatMul,
        /// Bias kept in f32 (added after the f32 `QMatMul`); `None` for the bias-less projections.
        bias: Option<Tensor>,
    },
    /// Loaded straight from an MLX-packed tier via the shared module — the sc-9087 fast path (no dense
    /// bf16 staging). The inner [`shared::QLinear`] holds the resident `Q4_1`/`Q8_0` weight and
    /// **dequantizes-on-forward** into a dense matmul (sc-7702, *not* the int8 `QMatMul` fast path).
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

    /// **Packed-detecting** `[out, in]` loader (sc-9087): if `{base}.scales` is present in `vb` (a
    /// pre-quantized MLX tier), build a [`Self::Packed`] straight from the packed parts on `vb`'s device
    /// via the shared [`candle_gen::quant::lin`] — **no dense weight is materialized**. Otherwise the
    /// **dense** path is taken unchanged (`{base}.weight` [+ `{base}.bias`]), to be optionally folded
    /// later by [`Self::quantize_onto`]. `base` is the full dotted key prefix (e.g. `attn.to_out.0`),
    /// so the `.scales`/`.biases` siblings survive any `to_out.0`-style key nesting (the sc-8670 remap
    /// trap: build the base string first, then detect — never `.pp()` past the scales sibling).
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

    /// `x·Wᵀ + b`. Dense delegates to `candle_nn::Linear`; load-time-quantized casts the input to f32,
    /// runs the GGUF `QMatMul`, adds the f32 bias, and casts back; packed delegates to the shared
    /// dequant-on-forward `QLinear` (sc-7702).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(l) => l.forward(x),
            Self::Quantized { matmul, bias } => {
                let in_dtype = x.dtype();
                let xf = x.to_dtype(DType::F32)?.contiguous()?;
                let out = matmul.forward(&xf)?;
                let out = match bias {
                    Some(b) => out.broadcast_add(b)?,
                    None => out,
                };
                out.to_dtype(in_dtype)
            }
            Self::Packed(l) => l.forward(x),
        }
    }

    /// Fold a dense projection to `Q4_0`/`Q8_0` **onto `device`** in place. **Idempotent**: a no-op if
    /// already load-time-quantized *or* packed (an MLX-packed weight must not be double-quantized). The
    /// dense weight is round-tripped to the CPU (the `quantize_onto` source requirement) and the
    /// resulting `QTensor` lives on `device`; the dense copy is dropped. The bias is promoted to f32 on
    /// `device`. Unlike Lens's same-device `quantize`, this targets an explicit device so a dense-tier
    /// dev model can stage dense in system RAM and land quantized on the GPU.
    pub fn quantize_onto(&mut self, quant: Quant, device: &Device) -> Result<()> {
        let Self::Dense(l) = self else {
            return Ok(());
        };
        let w_cpu = l.weight().to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        let qtensor = QTensor::quantize_onto(&w_cpu, ggml_dtype(quant), device)?;
        let matmul = QMatMul::from_qtensor(qtensor)?;
        let bias = match l.bias() {
            Some(b) => Some(b.to_device(device)?.to_dtype(DType::F32)?),
            None => None,
        };
        *self = Self::Quantized { matmul, bias };
        Ok(())
    }

    /// Move a still-**dense** projection (weight + optional bias) to `device`, in place. A no-op when
    /// already quantized or packed (those weights already live on their device). Used by the CPU-staged
    /// quant path for the leaves it must keep dense — e.g. the control branch's `control_img_in` (260
    /// in-features is not a multiple of the Q4_0/Q8_0 block 32, so it can't quantize) (sc-7460).
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

    /// Whether this projection is quantized (load-time-folded or packed) — i.e. holds no dense weight.
    pub fn is_quantized(&self) -> bool {
        matches!(self, Self::Quantized { .. } | Self::Packed(_))
    }

    /// Whether this projection loaded directly from an MLX-packed tier (the sc-9087 no-dense-staging
    /// path). Distinguishes a packed load from a load-time `quantize_onto` fold — used by the loaders to
    /// assert that a packed tier took the packed path (and by tests).
    pub fn is_packed(&self) -> bool {
        matches!(self, Self::Packed(_))
    }
}

/// A token embedding that is **dense** (`candle_nn::Embedding`) or **packed** (loaded straight from an
/// MLX-packed tier's `embed_tokens` triple via the shared [`candle_gen::quant::QEmbedding`], sc-9087).
/// The dev Mistral TE `embed_tokens` is packed in the q4/q8 tiers, so this closes the packed-detect
/// surface over the token embedding too (no dense embedding table materialized on the packed path).
pub enum QEmbedding {
    Dense(candle_gen::candle_nn::Embedding),
    Packed(shared::QEmbedding),
}

impl QEmbedding {
    /// **Packed-detecting** `[vocab, hidden]` embedding loader (sc-9087): packed when `{base}.scales`
    /// is present in `vb` (build straight from the packed parts via the shared
    /// [`candle_gen::quant::embedding`], dequantized to `vb.dtype()` — dtype parity with the dense
    /// table), else dense (`{base}.weight`, path unchanged). `base` is the full dotted key prefix.
    pub fn detect(vb: &VarBuilder, base: &str, vocab: usize, hidden: usize) -> Result<Self> {
        if vb.contains_tensor(&format!("{base}.scales")) {
            return Ok(Self::Packed(shared::embedding(vb, base, vocab, hidden)?));
        }
        Ok(Self::Dense(candle_gen::candle_nn::embedding(
            vocab,
            hidden,
            vb.pp(base),
        )?))
    }

    /// Move a still-**dense** table to `device` in place (a no-op when packed — the packed table already
    /// lives on its device). The CPU-staged dense path carries the token embedding to the GPU here.
    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        if let Self::Dense(e) = self {
            *self = Self::Dense(candle_gen::candle_nn::Embedding::new(
                e.embeddings().to_device(device)?,
                e.hidden_size(),
            ));
        }
        Ok(())
    }

    /// Index-select the embedding rows for `indexes`.
    pub fn forward(&self, indexes: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(e) => e.forward(indexes),
            Self::Packed(e) => e.forward(indexes),
        }
    }

    pub fn is_packed(&self) -> bool {
        matches!(self, Self::Packed(_))
    }
}

/// Rebuild a dense `RmsNorm` on `device` at `eps` (a no-op-cost move when already there). Used by the
/// CPU-staged dev quant path to carry the full-precision norms onto the GPU alongside the quantized
/// projections.
pub fn rms_norm_to(n: &RmsNorm, eps: f64, device: &Device) -> Result<RmsNorm> {
    Ok(RmsNorm::new(n.weight().to_device(device)?, eps))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dense_from(w: &Tensor, b: Option<&Tensor>) -> QLinear {
        QLinear::Dense(Linear::new(w.clone(), b.cloned()))
    }

    /// A `[64, 32]` projection quantizes and forwards near-losslessly at Q8 / coherently at Q4 vs the
    /// dense f32 result — the per-linear analog of the full-model quant parity, on CPU with no weights.
    fn quant_roundtrip(quant: Quant, min_cos: f32) {
        let dev = Device::Cpu;
        let (in_dim, out_dim) = (32usize, 64usize);
        let w = Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev).unwrap();
        let b = Tensor::randn(0f32, 1f32, (out_dim,), &dev).unwrap();
        let mut lin = dense_from(&w, Some(&b));

        let x = Tensor::randn(0f32, 1f32, (4, in_dim), &dev).unwrap();
        let dense = lin.forward(&x).unwrap();

        lin.quantize_onto(quant, &dev).unwrap();
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

    #[test]
    fn quantize_is_idempotent() {
        let dev = Device::Cpu;
        let w = Tensor::randn(0f32, 1f32, (64, 32), &dev).unwrap();
        let mut lin = dense_from(&w, None);
        lin.quantize_onto(Quant::Q8, &dev).unwrap();
        lin.quantize_onto(Quant::Q8, &dev).unwrap(); // no-op, must not error
        assert!(matches!(lin, QLinear::Quantized { bias: None, .. }));
    }

    // ---- sc-9087 packed-detect (from an MLX-packed tier, no dense staging) --------------------

    use candle_gen::candle_core::safetensors::MmapedSafetensors;
    use std::collections::HashMap;

    /// Test-side MLX Q4 packer: per-element 4-bit codes → MLX u32 words (LSB-first nibbles), group 64.
    /// Returns `(wq [out, in/8] u32, scales [out, in/64], biases [out, in/64], affine grid [out, in])`.
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

    fn cosine(a: &Tensor, b: &Tensor) -> f32 {
        let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (x, y) in a.iter().zip(&b) {
            dot += (*x as f64) * (*y as f64);
            na += (*x as f64) * (*x as f64);
            nb += (*y as f64) * (*y as f64);
        }
        (dot / (na.sqrt() * nb.sqrt() + 1e-12)) as f32
    }

    /// Write a safetensors mimicking the flux2-dev packed key layout — including the `attn.to_out.0`
    /// nesting the loader remaps — and load it through the packed-detecting `linear_detect`. The
    /// `.scales`/`.biases` siblings must survive the `to_out.0` base string (the sc-8670 remap trap),
    /// the projection must load `Packed` (no dense staging), and its forward must match the affine grid.
    #[test]
    fn linear_detect_packed_survives_to_out_remap() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (64usize, 128usize);
        let (wq, s, b, grid) = q4_packed(out_dim, in_dim);

        let mut map: HashMap<String, Tensor> = HashMap::new();
        // The `attn.to_out.0` packed triple (the remap the flux2 loader threads as a single base).
        map.insert("attn.to_out.0.weight".into(), wq);
        map.insert("attn.to_out.0.scales".into(), s);
        map.insert("attn.to_out.0.biases".into(), b);
        // A dense sibling (`to_q`) with no `.scales` → the dense path must stay unchanged.
        map.insert(
            "attn.to_q.weight".into(),
            Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev)?,
        );

        let tmp =
            std::env::temp_dir().join(format!("sc9087_detect_{}.safetensors", std::process::id()));
        candle_gen::candle_core::safetensors::save(&map, &tmp)?;
        // SAFETY: we just wrote this file and nothing else touches it during the test.
        let st = unsafe { MmapedSafetensors::new(&tmp)? };
        let vb = VarBuilder::from_backend(Box::new(st), DType::F32, dev.clone());
        let attn = vb.pp("attn");

        // `to_out.0` — packed-detected through the remapped base (never `.pp("0")` past the sibling).
        let packed = QLinear::linear_detect(in_dim, out_dim, &attn, "to_out.0", false)?;
        assert!(packed.is_packed(), "`.scales` under to_out.0 ⇒ packed load");
        assert!(packed.is_quantized());

        // `to_q` — dense (no `.scales`), path unchanged.
        let dense = QLinear::linear_detect(in_dim, out_dim, &attn, "to_q", false)?;
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

    /// `quantize_onto` is a **no-op** on a packed projection — an MLX-packed weight must never be
    /// double-quantized. The projection stays `Packed` and its forward is unchanged.
    #[test]
    fn quantize_onto_is_noop_on_packed() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (64usize, 128usize);
        let (wq, s, b, _grid) = q4_packed(out_dim, in_dim);

        let packed = shared::QLinear::from_packed(&wq, &s, &b, None, &dev)?;
        let mut lin = QLinear::Packed(packed);
        let x = Tensor::randn(0f32, 1f32, (4, in_dim), &dev)?;
        let before = lin.forward(&x)?;

        lin.quantize_onto(Quant::Q4, &dev)?; // must no-op, not re-quantize
        assert!(lin.is_packed(), "quantize_onto changed a packed projection");
        let after = lin.forward(&x)?;
        let dev_max = (before.sub(&after)?).abs()?.max_all()?.to_scalar::<f32>()?;
        assert_eq!(
            dev_max, 0.0,
            "no-op quantize_onto changed the packed forward"
        );
        Ok(())
    }

    /// The packed-detecting embedding loader fires on `{base}.scales` and reproduces the affine grid
    /// rows (the dev Mistral TE `embed_tokens` is packed in the q4/q8 tiers).
    #[test]
    fn embedding_detect_packed() -> Result<()> {
        let dev = Device::Cpu;
        let (vocab, hidden) = (32usize, 128usize);
        let (wq, s, b, grid) = q4_packed(vocab, hidden);

        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert("embed_tokens.weight".into(), wq);
        map.insert("embed_tokens.scales".into(), s);
        map.insert("embed_tokens.biases".into(), b);
        let tmp =
            std::env::temp_dir().join(format!("sc9087_emb_{}.safetensors", std::process::id()));
        candle_gen::candle_core::safetensors::save(&map, &tmp)?;
        // SAFETY: freshly written, single-reader.
        let st = unsafe { MmapedSafetensors::new(&tmp)? };
        let vb = VarBuilder::from_backend(Box::new(st), DType::F32, dev.clone());

        let emb = QEmbedding::detect(&vb, "embed_tokens", vocab, hidden)?;
        assert!(emb.is_packed(), "`.scales` ⇒ packed embedding");

        let dense_table = candle_gen::candle_nn::Embedding::new(
            Tensor::from_vec(grid, (vocab, hidden), &dev)?,
            hidden,
        );
        let idx = Tensor::from_vec(vec![0u32, 5, 31, 12, 5], (5,), &dev)?;
        let p = emb.forward(&idx)?;
        let d = dense_table.forward(&idx)?;
        let dev_max = (p.sub(&d)?).abs()?.max_all()?.to_scalar::<f32>()?;
        assert_eq!(
            dev_max, 0.0,
            "packed embedding deviates from the affine grid"
        );

        std::fs::remove_file(&tmp).ok();
        Ok(())
    }
}
