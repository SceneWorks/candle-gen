//! FLUX.1 [schnell] packed-load seam (sc-9407, sc-9089 umbrella) — the candle twin of the flux2-dev
//! conversion (sc-9087) and the z-image conversion (sc-9408), built on the shared
//! [`candle_gen::quant`] packed-load module (sc-9086).
//!
//! FLUX.1 [schnell] ships a **pre-quantized** MLX tier (`SceneWorks/flux1-schnell-mlx`, epic 8506)
//! whose q4/q8 snapshots store each quantized `Linear` / token embedding as the MLX packed triple
//! `{base}.weight` (u32 codes) + `{base}.scales` + `{base}.biases`. [`QLinear::linear_detect`] /
//! [`QEmbedding::detect`] packed-**detect** the `.scales` sibling and build the quantized weight
//! **straight from the packed parts** on the target device via the shared [`candle_gen::quant::lin`] /
//! `embedding` loaders (Q4 → `Q4_1` lossless repack, Q8 → `Q8_0` requant). **No dense bf16 weight is
//! ever materialized** on the packed path — the q4 DiT lands ~6.3 GB directly rather than staging a
//! ~24 GB dense DiT in RAM first.
//!
//! Absent `.scales` (a dense bf16 tier — the stock black-forest-labs `FLUX.1-schnell` BFL snapshot)
//! the loader falls back to the **stock** BFL-layout `candle-transformers` path unchanged (see
//! [`crate::pipeline`]), so one crate serves both a dense BFL snapshot and a packed diffusers snapshot.
//!
//! **The packed forward dequantizes the weight into a dense matmul (sc-7702)** — it does *not* take
//! candle's int8 `QMatMul` fast path (`fast_mmq`), so a Q4 denoise stays coherent. The whole
//! `Dense | Quantized` Linear seam now lives once in `candle-gen` (F-025 / sc-9005): FLUX.1's packed
//! projections load through the shared [`candle_gen::quant::QLinear`]'s
//! [`MatmulStrategy::DequantDense`](candle_gen::quant::MatmulStrategy::DequantDense) arm, which owns
//! that behavior. This module re-exports that shared `QLinear` under the crate-local name every vendored
//! diffusers DiT / T5 / CLIP builds its projections from, and keeps the thin dense-or-packed
//! [`QEmbedding`] wrapper + the [`dequant_packed_to_dense`] VAE helper the packed loaders need.
//!
//! FLUX.1 has no *dense-tier on-the-fly* quant path (the only quantized tier is the pre-packed MLX
//! one), so — like the z-image seam and unlike the flux2 seam — this crate never folds a dense weight;
//! it only packed-detects. The [`QLinear::quantize`] / `quantize_onto` folds on the shared type are
//! simply never called here.

use candle_gen::candle_core::{Device, Result, Tensor};
use candle_gen::candle_nn::{Embedding, Module, VarBuilder};
use candle_gen::quant as shared;

// The `Dense | Quantized` Linear seam now lives once in `candle-gen` (F-025 / sc-9005). FLUX.1's packed
// projections take the shared [`MatmulStrategy::DequantDense`] arm (sc-7702-safe, no int8 activation
// quant), built by `QLinear::linear_detect` on an MLX tier. Re-export the shared type under the
// crate-local `QLinear` name the vendored DiT / T5 / CLIP reference.
pub use candle_gen::quant::QLinear;

/// A token embedding that is **dense** (`candle_nn::Embedding`) or **packed** (loaded straight from an
/// MLX-packed tier's embedding triple via the shared [`candle_gen::quant::QEmbedding`], sc-9407). The
/// FLUX packed tier packs the CLIP `token_embedding`/`position_embedding`, the T5 `shared` token table,
/// and the T5 per-block `relative_attention_bias` table, so this closes the packed-detect surface over
/// every embedding table too (no dense table materialized on the packed path).
pub enum QEmbedding {
    Dense(Embedding),
    Packed(shared::QEmbedding),
}

impl QEmbedding {
    /// **Packed-detecting** `[vocab, hidden]` embedding loader (sc-9407): packed when `{base}.scales`
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

    /// Index-select the embedding rows for `indexes`.
    pub fn forward(&self, indexes: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(e) => e.forward(indexes),
            Self::Packed(e) => e.forward(indexes),
        }
    }

    /// Whether this embedding loaded directly from an MLX-packed tier — used by the loaders/tests to
    /// assert a packed tier fired the packed path (and did not silently fall back to dense).
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn is_packed(&self) -> bool {
        matches!(self, Self::Packed(_))
    }
}

/// Dequantize an MLX-packed triple (`{base}.weight` u32 + `{base}.scales` + `{base}.biases`) to a
/// **dense** `[out, in]` tensor on `device` at `out_dtype` (Q4 via the exact affine grid, Q8 via the
/// exact-grid dequant — [`candle_gen::quant::repack`]). Used only by the VAE loader
/// ([`crate::pipeline`]): the FLUX packed tier quantizes just the 8 tiny (512×64) mid-block attention
/// projections, so — rather than vendor a whole diffusers VAE for negligible weights — the packed VAE
/// weights are dequantized to dense here and handed to a **stock** diffusers `AutoencoderKL` through a
/// `VarBuilder::from_tensors` overlay (the same pragmatic path z-image took, sc-9408). This still loads
/// *from the packed parts* (no dense bf16 tier downloaded); the ~2 MB one-time dequant is trivial next to
/// the DiT/TE the packed path serves natively.
pub fn dequant_packed_to_dense(
    vb: &VarBuilder,
    base: &str,
    device: &Device,
    out_dtype: candle_gen::candle_core::DType,
) -> Result<Tensor> {
    use candle_gen::candle_core::DType;
    let wq = vb.get_unchecked_dtype(&format!("{base}.weight"), DType::U32)?;
    let scales = vb.get_unchecked_dtype(&format!("{base}.scales"), DType::F32)?;
    let biases = vb.get_unchecked_dtype(&format!("{base}.biases"), DType::F32)?;
    let (wq_cols, s_cols) = (wq.dims2()?.1, scales.dims2()?.1);
    let grid = match shared::mlx_packed_bits(wq_cols, s_cols) {
        4 => shared::dequant_mlx_q4_reference(&wq, &scales, &biases)?,
        8 => shared::dequant_mlx_q8(&wq, &scales, &biases)?,
        b => candle_gen::candle_core::bail!("flux vae: unsupported MLX packed bit-width {b}"),
    };
    grid.to_device(device)?.to_dtype(out_dtype)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::safetensors::MmapedSafetensors;
    use candle_gen::candle_core::{DType, Device};
    use candle_gen::candle_nn::Linear;
    use candle_gen::quant::{DenseLinear, MatmulStrategy};
    use std::collections::HashMap;

    /// Test-side MLX Q4 packer: per-element 4-bit codes → MLX u32 words (LSB-first nibbles), group 64.
    /// Returns `(wq [out, in/8] u32, scales [out, in/64], biases [out, in/64], affine grid [out, in])` —
    /// the exact packed-parts fixture the detect loaders consume, plus the affine grid they reproduce.
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

    /// **Packed-detect fires on the FLUX diffusers key layout (incl. the `attn.to_out.0` nesting).**
    /// Writes a safetensors mimicking the real FLUX DiT packed layout — a `to_out.0` triple (the key
    /// remap that would silently fall back to dense if the loader `.pp()`'d past the `.scales` sibling)
    /// and a dense `to_q` sibling — and loads both through `linear_detect`. The `.scales`/`.biases`
    /// siblings must survive the `to_out.0` base string, `to_out.0` must load quantized via the
    /// sc-7702-safe **dequant-dense** arm (the packed forward flux uses), the dense sibling stays
    /// `Dense`, and the packed forward must reproduce the affine grid bit-exactly.
    #[test]
    fn linear_detect_fires_on_to_out_remap_and_leaves_dense_unchanged() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (64usize, 128usize);
        let (wq, s, b, grid) = q4_packed(out_dim, in_dim);

        let mut map: HashMap<String, Tensor> = HashMap::new();
        // The `attn.to_out.0` packed triple (the nested key the FLUX DiT threads as one base).
        map.insert("attn.to_out.0.weight".into(), wq);
        map.insert("attn.to_out.0.scales".into(), s);
        map.insert("attn.to_out.0.biases".into(), b);
        // A dense sibling (`to_q`) with no `.scales` → the dense path must stay unchanged.
        map.insert(
            "attn.to_q.weight".into(),
            Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev)?,
        );

        let tmp =
            std::env::temp_dir().join(format!("sc9407_detect_{}.safetensors", std::process::id()));
        candle_gen::candle_core::safetensors::save(&map, &tmp)?;
        // SAFETY: we just wrote this file and nothing else touches it during the test.
        let st = unsafe { MmapedSafetensors::new(&tmp)? };
        let vb = VarBuilder::from_backend(Box::new(st), DType::F32, dev.clone());
        let attn = vb.pp("attn");

        // `to_out.0` — packed-detected through the remapped base (never `.pp("0")` past the sibling).
        // The packed load must take the sc-7702-safe dequant-dense arm, NOT the int8-fast fold arm.
        let packed = QLinear::linear_detect(in_dim, out_dim, &attn, "to_out.0", false)?;
        assert_eq!(
            packed.matmul_strategy(),
            Some(MatmulStrategy::DequantDense),
            "`.scales` under to_out.0 ⇒ packed dequant-dense load"
        );

        // `to_q` — dense (no `.scales`), path unchanged.
        let dense = QLinear::linear_detect(in_dim, out_dim, &attn, "to_q", false)?;
        assert!(!dense.is_quantized(), "no `.scales` ⇒ dense path unchanged");
        assert!(matches!(dense, QLinear::Dense(_)));

        // The packed forward reproduces the affine grid (bit-exact repack + dequant-on-forward).
        let grid_lin = QLinear::from_dense(DenseLinear::Linear(Linear::new(
            Tensor::from_vec(grid, (out_dim, in_dim), &dev)?,
            None,
        )));
        let x = Tensor::randn(0f32, 1f32, (4, in_dim), &dev)?;
        let cos = cosine(&packed.forward(&x)?, &grid_lin.forward(&x)?);
        assert!(cos > 0.99999, "packed vs affine-grid cosine {cos:.6}");

        std::fs::remove_file(&tmp).ok();
        Ok(())
    }

    /// The packed-detecting embedding loader fires on `{base}.scales` and reproduces the affine grid
    /// rows (the CLIP `token_embedding` / T5 `shared` are packed in the q4/q8 tiers), while a
    /// `.scales`-less table stays dense.
    #[test]
    fn embedding_detect_fires_and_matches_grid() -> Result<()> {
        let dev = Device::Cpu;
        let (vocab, hidden) = (32usize, 128usize);
        let (wq, s, b, grid) = q4_packed(vocab, hidden);

        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert("shared.weight".into(), wq);
        map.insert("shared.scales".into(), s);
        map.insert("shared.biases".into(), b);
        map.insert(
            "dense.weight".into(),
            Tensor::from_vec(grid.clone(), (vocab, hidden), &dev)?,
        );
        let tmp =
            std::env::temp_dir().join(format!("sc9407_emb_{}.safetensors", std::process::id()));
        candle_gen::candle_core::safetensors::save(&map, &tmp)?;
        // SAFETY: freshly written, single-reader.
        let st = unsafe { MmapedSafetensors::new(&tmp)? };
        let vb = VarBuilder::from_backend(Box::new(st), DType::F32, dev.clone());

        let emb = QEmbedding::detect(&vb, "shared", vocab, hidden)?;
        assert!(emb.is_packed(), "`.scales` ⇒ packed embedding");
        let dense = QEmbedding::detect(&vb, "dense", vocab, hidden)?;
        assert!(!dense.is_packed(), "no `.scales` ⇒ dense embedding");

        let idx = Tensor::from_vec(vec![0u32, 5, 31, 12, 5], (5,), &dev)?;
        let p = emb.forward(&idx)?;
        let d = dense.forward(&idx)?;
        let dev_max = (p.sub(&d)?).abs()?.max_all()?.to_scalar::<f32>()?;
        assert_eq!(
            dev_max, 0.0,
            "packed embedding deviates from the dense grid"
        );

        std::fs::remove_file(&tmp).ok();
        Ok(())
    }
}
