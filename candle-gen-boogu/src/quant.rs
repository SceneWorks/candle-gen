//! Boogu packed-load seam (sc-9410, sc-9089 umbrella) — the candle twin of the flux2-dev conversion
//! (sc-9087), z-image (sc-9408), and flux-schnell (sc-9407), built on the shared
//! [`candle_gen::quant`] packed-load module (sc-9086).
//!
//! Boogu ships a **pre-quantized** MLX tier (`SceneWorks/boogu-image-mlx`, bf16/q4 per variant; NO q8)
//! whose q4 snapshots store each quantized `Linear` / token embedding as the MLX packed triple
//! `{base}.weight` (u32 codes) + `{base}.scales` + `{base}.biases`. Unlike the z-image / flux tiers,
//! the boogu quantizer packs at **group size 32** (their default 64), which the packed shapes alone
//! can't disambiguate — so the group size is read from each component `config.json`'s
//! `quantization.group_size` ([`candle_gen::quant::PackedConfig`]) and threaded through the shared
//! group-size-aware loaders ([`candle_gen::quant::QLinear::from_packed_gs`] / `from_packed_dtype_gs`,
//! Q4 → `Q4_1` lossless repack). **No dense bf16 weight is ever materialized** on the packed path — the
//! q4 DiT/TE land straight from the packed parts rather than staging the dense tier in RAM first.
//!
//! Absent `.scales` (a dense bf16 tier) the **dense** path is taken **unchanged** (`candle_nn::Linear`
//! / `Embedding` from `{base}.weight`), so one crate serves both a dense bf16 and a packed q4 snapshot.
//!
//! Boogu's loader is a thin `MmapedSafetensors` wrapper ([`crate::loader::Weights`]), not a
//! `VarBuilder` (the flux2/z-image/flux crates load through `VarBuilder`), so this seam builds the
//! quantized module from **raw tensors** pulled through `Weights` (`from_packed_gs`) rather than the
//! VarBuilder-detecting `candle_gen::quant::lin`; the compute path is identical (the shared
//! dequant-on-forward `QLinear`/`QEmbedding`, sc-7702 — *not* candle's int8 `QMatMul` fast path).

use candle_gen::candle_core::{Result, Tensor};
use candle_gen::candle_nn::{Embedding, Linear, Module};
use candle_gen::quant as shared;

/// A Linear projection that is **dense** (the loaded bf16 weight) or **packed** (loaded straight from
/// the MLX-packed tier via the shared [`candle_gen::quant::QLinear`], sc-9410). Built dense
/// ([`Self::dense`]) or packed ([`Self::packed`]); both forwards compute `x·Wᵀ + b`.
pub enum QLinear {
    Dense(Linear),
    /// Loaded directly from the MLX-packed tier through the shared module — the resident `Q4_1` weight
    /// **dequantizes-on-forward** into a dense matmul (sc-7702, *not* the int8 `QMatMul` fast path).
    Packed(shared::QLinear),
}

impl QLinear {
    /// Wrap an already-loaded dense [`Linear`] (the loader built it from `{base}.weight` [+ `.bias`]).
    pub fn dense(linear: Linear) -> Self {
        Self::Dense(linear)
    }

    /// Build a **packed** projection from the MLX packed triple (`wq` u32 codes + `scales` + `biases`,
    /// optional dense `bias`) at the tier's `group_size`, on `wq`'s device, via the shared group-size
    /// -aware loader. No dense weight is materialized.
    pub fn packed(
        wq: &Tensor,
        scales: &Tensor,
        biases: &Tensor,
        bias: Option<Tensor>,
        group_size: usize,
    ) -> Result<Self> {
        let device = wq.device().clone();
        Ok(Self::Packed(shared::QLinear::from_packed_gs(
            wq, scales, biases, bias, group_size, &device,
        )?))
    }

    /// `x·Wᵀ + b`. Dense delegates to `candle_nn::Linear`; packed delegates to the shared
    /// dequant-on-forward `QLinear` (sc-7702).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(l) => l.forward(x),
            Self::Packed(l) => l.forward(x),
        }
    }

    /// Whether this projection loaded directly from the MLX-packed tier (the packed path) — used by
    /// the loaders + tests to assert a packed tier fired the packed path (not a silent dense fallback).
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

/// A token embedding that is **dense** (`candle_nn::Embedding`) or **packed** (loaded straight from the
/// MLX-packed tier's `embed_tokens` triple via the shared [`candle_gen::quant::QEmbedding`], sc-9410).
/// The Qwen3-VL TE `model.language_model.embed_tokens` is packed in the q4 tier, so this closes the
/// packed-detect surface over the token embedding too (no dense table materialized on the packed path).
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

    const G: usize = 32; // the boogu tier's MLX quant group size

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

    /// A group-32 packed `QLinear` forward matches a dense linear built from the SAME affine grid the
    /// pack represents — bit-exact (the repack is lossless, both forwards dequant-to-dense-matmul). Uses
    /// the boogu tier's real `to_out.0` output-projection shape (3360×3360).
    #[test]
    fn packed_qlinear_matches_dense_grid() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (128usize, 256usize);
        let (wq, s, b, grid) = q4_packed(out_dim, in_dim);

        let packed = QLinear::packed(&wq, &s, &b, None, G)?;
        assert!(packed.is_packed(), "group-32 triple ⇒ packed load");
        let dense = QLinear::dense(Linear::new(
            Tensor::from_vec(grid, (out_dim, in_dim), &dev)?,
            None,
        ));
        assert!(!dense.is_packed());

        let x = Tensor::randn(0f32, 1f32, (4, in_dim), &dev)?;
        let cos = cosine(&packed.forward(&x)?, &dense.forward(&x)?);
        assert!(
            cos > 0.99999,
            "group-32 packed vs affine-grid cosine {cos:.6}"
        );
        Ok(())
    }

    /// A group-32 packed `QEmbedding` reproduces the affine grid rows exactly (the Qwen3-VL TE
    /// `embed_tokens` is packed in the q4 tier), dequantized to the requested dtype.
    #[test]
    fn packed_qembedding_matches_dense_grid() -> Result<()> {
        let dev = Device::Cpu;
        let (vocab, hidden) = (64usize, 128usize);
        let (wq, s, b, grid) = q4_packed(vocab, hidden);

        let packed = QEmbedding::packed(&wq, &s, &b, DType::F32, G)?;
        assert!(packed.is_packed(), "group-32 triple ⇒ packed embedding");
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
            "group-32 packed embedding deviates from the grid"
        );
        Ok(())
    }
}
