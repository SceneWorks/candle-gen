//! Ideogram 4 packed-load seam (sc-9412, sc-9089 umbrella) — the candle twin of the flux2-dev
//! conversion (sc-9087) and the direct sibling of krea (sc-9411), built on the shared
//! [`candle_gen::quant`] packed-load module (sc-9086).
//!
//! Ideogram ships a **pre-quantized** MLX tier (`SceneWorks/ideogram-4-mlx`, q4/q8; no bf16) whose
//! snapshots store each quantized `Linear` as the MLX packed triple `{base}.weight` (u32 codes) +
//! `{base}.scales` + `{base}.biases` (bf16). Both the DiT (`transformer/`,
//! `unconditional_transformer/`) and the Qwen3-VL text encoder are packed; the VAE stays dense. The
//! group size is 64 (the MLX default — the ideogram converter emits **no** `quantization` block in
//! `config.json`, so detection keys purely on the `{base}.scales` sibling, and the group size defaults
//! to [`candle_gen::quant::MLX_GROUP_SIZE`], exactly as the shared VarBuilder `lin`/`embedding` do).
//! It is threaded through the shared group-size-aware loaders
//! ([`candle_gen::quant::QLinear::from_packed_gs`] / `QEmbedding::from_packed_dtype_gs`, Q4 → `Q4_1`
//! lossless repack, Q8 → `Q8_0`) so a future tier that packs at a different group (and carries the
//! `quantization.group_size` block, sc-9474) is honoured — **not** a hardcoded 64. **No dense weight is
//! ever materialized** on the packed path.
//!
//! Absent `.scales` (a dense projection MLX left dense — the DiT norms and the small
//! `embed_image_indicator` table — or an all-dense bf16 tier) the **dense** path is taken
//! **unchanged** (`candle_nn::Linear` / `Embedding` from `{base}.weight`), so one crate serves both a
//! dense bf16 and a packed q4/q8 snapshot.
//!
//! **The projection type is the SHARED [`candle_gen::quant::AdaptLinear`] (sc-11104).** It wraps a
//! frozen base — dense (`candle_nn::Linear`) or MLX-packed ([`candle_gen::quant::QLinear`],
//! dequant-on-forward, sc-7702) — plus zero or more **forward-time additive LoRA residuals**. With no
//! residual attached its forward is byte-identical to the bare base, so the plain T2I path is
//! unchanged. On **both** tiers the bundled TurboTime LoRA rides as an *unmerged* residual
//! (`y = base(x) + Σ scale·((x·A)·B)`) — never folded into a base weight, so a packed base stays
//! quantized and a dense base stays a clean disk-backed mmap (eviction-friendly)
//! ([`crate::adapters::install_turbo_lora_additive`]). The DiT loader is a thin `MmapedSafetensors`
//! wrapper ([`crate::loader::Weights`]), so this seam builds the base from **raw tensors** pulled
//! through `Weights` (`from_packed_gs` / `Linear::new`) wrapped via
//! [`candle_gen::quant::AdaptLinear::from_packed`] / `from_dense`, rather than the VarBuilder-detecting
//! `candle_gen::quant::lin`. The Qwen3-VL text encoder loads via a `VarBuilder`, so it routes straight
//! through the shared `lin`/`embedding` detectors instead ([`crate::text_encoder`]). Both compute
//! paths are identical: the shared dequant-on-forward `QLinear` (sc-7702 — *not* candle's int8
//! `QMatMul` fast path, whose q8_1 activation quant NaNs on outlier features).

use candle_gen::candle_core::{Result, Tensor};
use candle_gen::candle_nn::{Embedding, Module};
use candle_gen::quant as shared;

/// A Linear projection with a frozen base (**dense** or MLX-**packed**) plus optional **forward-time
/// additive LoRA/LoKr residuals** — the shared [`candle_gen::quant::AdaptLinear`] (sc-11104). Built
/// dense ([`QLinear::from_dense`]) or packed ([`QLinear::from_packed`], from the raw MLX triple via the
/// shared [`candle_gen::quant::QLinear::from_packed_gs`]); the loader ([`crate::loader::linear_detect`])
/// picks the arm. The TurboTime LoRA is pushed post-load as a forward-time residual on **both** tiers
/// ([`crate::adapters::install_turbo_lora_additive`]) — the base is never mutated.
pub use candle_gen::quant::AdaptLinear as QLinear;

/// A token embedding that is **dense** (`candle_nn::Embedding`) or **packed** (loaded straight from the
/// MLX-packed tier's triple via the shared [`candle_gen::quant::QEmbedding`], sc-9412). The DiT's small
/// `embed_image_indicator` table stays **dense** in the hosted q4/q8 tiers (only the projections are
/// packed there), so on the DiT this always takes the dense arm; the packed arm closes the
/// packed-detect surface for future-proofing + guards a silent dense read of u32 codes. (The Qwen3-VL
/// TE's `embed_tokens` **is** packed, but the TE loads through the shared VarBuilder `embedding`
/// detector, not this enum — see [`crate::text_encoder`].)
pub enum QEmbedding {
    Dense(Embedding),
    Packed(shared::QEmbedding),
}

impl QEmbedding {
    /// Wrap an already-loaded dense [`Embedding`].
    pub fn dense(embedding: Embedding) -> Self {
        Self::Dense(embedding)
    }

    /// Build a **packed** embedding from the MLX packed triple at `group_size`, dequantized to
    /// `out_dtype` (the dense-path table dtype — dtype parity), on `wq`'s device.
    pub fn packed(
        wq: &Tensor,
        scales: &Tensor,
        biases: &Tensor,
        out_dtype: candle_gen::candle_core::DType,
        group_size: usize,
    ) -> Result<Self> {
        let device = wq.device().clone();
        Ok(Self::Packed(shared::QEmbedding::from_packed_dtype_gs(
            wq, scales, biases, &device, out_dtype, group_size,
        )?))
    }

    /// Index-select the embedding rows for `indexes`.
    pub fn forward(&self, indexes: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(e) => e.forward(indexes),
            Self::Packed(e) => e.forward(indexes),
        }
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn is_packed(&self) -> bool {
        matches!(self, Self::Packed(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::{DType, Device};
    use candle_gen::candle_nn::Linear;

    /// The Ideogram MLX tier's quant group size (64 — the MLX default; the ideogram converter emits no
    /// `quantization` block, so the loaders default to this via [`candle_gen::quant::MLX_GROUP_SIZE`]).
    const G: usize = 64;

    /// Test-side MLX Q4 packer at group `G`: per-element 4-bit codes → MLX u32 words (LSB-first
    /// nibbles). Returns `(wq [out, in/8] u32, scales [out, in/G], biases [out, in/G], affine grid)` —
    /// the exact packed-parts fixture the loaders consume plus the affine grid they reproduce.
    fn q4_packed(out_dim: usize, in_dim: usize) -> (Tensor, Tensor, Tensor, Vec<f32>) {
        let dev = Device::Cpu;
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

    /// A group-64 packed [`QLinear`] (shared `AdaptLinear`, packed base, no residual) forward matches a
    /// dense linear built from the SAME affine grid the pack represents — bit-exact (the Q4 → Q4_1
    /// repack is lossless, both forwards dequant-to-dense-matmul). Uses an `input_proj`-style shape.
    #[test]
    fn packed_qlinear_matches_dense_grid() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (128usize, 256usize);
        let (wq, s, b, grid) = q4_packed(out_dim, in_dim);

        let base = shared::QLinear::from_packed_gs(&wq, &s, &b, None, G, &dev)?;
        let packed = QLinear::from_packed(base, in_dim, out_dim);
        assert!(packed.is_packed(), "group-64 triple ⇒ packed load");
        let dense = QLinear::from_dense(
            Linear::new(Tensor::from_vec(grid, (out_dim, in_dim), &dev)?, None),
            in_dim,
            out_dim,
        );
        assert!(!dense.is_packed());

        let x = Tensor::randn(0f32, 1f32, (4, in_dim), &dev)?;
        let cos = cosine(&packed.forward(&x)?, &dense.forward(&x)?);
        assert!(
            cos > 0.99999,
            "group-64 packed vs affine-grid cosine {cos:.6}"
        );
        Ok(())
    }

    /// A group-64 packed `QEmbedding` reproduces the affine grid rows exactly, dequantized to the
    /// requested dtype (the future-proof path for a DiT tier that ever packs `embed_image_indicator`).
    #[test]
    fn packed_qembedding_matches_dense_grid() -> Result<()> {
        let dev = Device::Cpu;
        let (vocab, hidden) = (64usize, 128usize);
        let (wq, s, b, grid) = q4_packed(vocab, hidden);

        let packed = QEmbedding::packed(&wq, &s, &b, DType::F32, G)?;
        assert!(packed.is_packed(), "group-64 triple ⇒ packed embedding");
        let dense = QEmbedding::dense(Embedding::new(
            Tensor::from_vec(grid, (vocab, hidden), &dev)?,
            hidden,
        ));

        let idx = Tensor::from_vec(vec![0u32, 5, 63, 12, 5], (5,), &dev)?;
        let p = packed.forward(&idx)?;
        let d = dense.forward(&idx)?;
        let dev_max = (p.sub(&d)?).abs()?.max_all()?.to_scalar::<f32>()?;
        assert_eq!(
            dev_max, 0.0,
            "group-64 packed embedding deviates from the grid"
        );
        Ok(())
    }
}
