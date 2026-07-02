//! Chroma DiT packed-load seam (sc-9409, sc-9089 umbrella) — the candle twin of the flux-schnell
//! conversion (sc-9407) and the z-image conversion (sc-9408), built on the shared
//! [`candle_gen::quant`] packed-load module (sc-9086).
//!
//! Chroma ships a **pre-quantized** MLX tier (`SceneWorks/chroma1-{base,hd,flash}-mlx`, epic 8506)
//! whose q4/q8 snapshots store each quantized DiT `Linear` as the MLX packed triple `{base}.weight`
//! (u32 codes) + `{base}.scales` + `{base}.biases` (plus the dense `{base}.bias`).
//! [`QLinear::linear_detect`] packed-**detects** the `.scales` sibling and builds the quantized weight
//! **straight from the packed parts** on the target device via the shared [`candle_gen::quant::lin_gs`]
//! loader (Q4 → `Q4_1` lossless repack, Q8 → `Q8_0` requant). **No dense bf16 weight is ever
//! materialized** on the packed path — the q4 DiT lands ~5.4 GB directly rather than staging the
//! ~17.8 GB dense DiT first.
//!
//! Absent `.scales` (a dense bf16 tier — the stock diffusers `Chroma1-*` snapshot, *and* every
//! non-quantized Linear inside a packed tier: the `x_embedder`/`context_embedder`/`proj_out` and the
//! whole `distilled_guidance_layer` Approximator ship dense even in the q4/q8 checkpoints) the loader
//! falls back to the **stock** `candle_nn::linear` path unchanged, so the same [`crate::transformer`]
//! serves both a dense snapshot and a packed one, and the mixed dense/packed DiT loads with one call
//! site per projection.
//!
//! **The packed forward dequantizes the weight into a dense matmul (sc-7702)** — it does *not* take
//! candle's int8 `QMatMul` fast path (`fast_mmq`), so a Q4 denoise stays coherent. The compute path is
//! the shared [`candle_gen::quant::QLinear`], which already owns that behavior; this module is only the
//! thin dense-or-packed **enum + detect** wrapper the vendored Chroma DiT builds its projections from
//! (mirroring `candle-gen-flux/src/quant.rs`). Chroma has no *dense-tier on-the-fly* quant path (the
//! only quantized tier is the pre-packed MLX one), so — like the flux-schnell and z-image seams — there
//! is no `quantize_onto`/load-time fold arm here.
//!
//! Chroma's T5-XXL text encoder and the AutoencoderKL VAE are **dense bf16 in every tier** (the MLX
//! convert job quantizes only the transformer — the `text_encoder/` + `vae/` safetensors are byte-for-
//! byte identical across the bf16/q4/q8 snapshots), so they keep the stock [`crate::text`] /
//! [`crate::vae`] loaders unchanged; only the DiT threads through this packed-detect seam.

use candle_gen::candle_core::{Result, Tensor};
use candle_gen::candle_nn::{Linear, Module, VarBuilder};
use candle_gen::quant as shared;

/// A Linear projection that is **dense** (the loaded bf16/f32 weight) or **packed** (loaded straight
/// from an MLX-packed tier via the shared [`candle_gen::quant::QLinear`], sc-9409). Built dense
/// ([`Self::linear`]) or packed-detected ([`Self::linear_detect`] / [`Self::linear_detect_gs`]); both
/// forwards compute `x·Wᵀ + b`.
pub enum QLinear {
    Dense(Linear),
    /// Loaded directly from an MLX-packed tier through the shared module — the resident `Q4_1`/`Q8_0`
    /// weight **dequantizes-on-forward** into a dense matmul (sc-7702, *not* the int8 `QMatMul` fast
    /// path).
    Packed(shared::QLinear),
}

impl QLinear {
    /// A biased dense `[out, in]` projection from `vb` (`{prefix}.weight` + `{prefix}.bias`).
    pub fn linear(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self::Dense(candle_gen::candle_nn::linear(
            in_dim, out_dim, vb,
        )?))
    }

    /// **Packed-detecting** `[out, in]` loader at the default MLX group size 64 — see
    /// [`Self::linear_detect_gs`]. Every Chroma packed tier ships `quantization.group_size = 64`, so
    /// this is the common call; the `_gs` form threads a non-64 group size read from `config.json`.
    #[cfg(test)]
    pub fn linear_detect(
        in_dim: usize,
        out_dim: usize,
        vb: &VarBuilder,
        base: &str,
        bias: bool,
    ) -> Result<Self> {
        Self::linear_detect_gs(in_dim, out_dim, vb, base, bias, shared::MLX_GROUP_SIZE)
    }

    /// **Packed-detecting** `[out, in]` loader at an explicit MLX `group_size` (sc-9409): if
    /// `{base}.scales` is present in `vb` (a pre-quantized MLX tier), build a [`Self::Packed`] straight
    /// from the packed parts on `vb`'s device via the shared [`candle_gen::quant::lin_gs`] — **no dense
    /// weight is materialized**. Otherwise the **dense** path is taken unchanged (`{base}.weight`
    /// [+ `{base}.bias`]).
    ///
    /// `base` is the full dotted key prefix (e.g. `attn.to_out.0`), so the `.scales`/`.biases`/`.bias`
    /// siblings survive any `to_out.0`-style key nesting: build the base string first, then detect —
    /// never `.pp()` past the scales sibling (the key-remap trap the shared loader guards). The
    /// `linear_detect_fires_on_to_out_remap*` test pins this on the real Chroma `attn.to_out.0` layout.
    pub fn linear_detect_gs(
        in_dim: usize,
        out_dim: usize,
        vb: &VarBuilder,
        base: &str,
        bias: bool,
        group_size: usize,
    ) -> Result<Self> {
        if vb.contains_tensor(&format!("{base}.scales")) {
            return Ok(Self::Packed(shared::lin_gs(
                vb, base, in_dim, out_dim, bias, group_size,
            )?));
        }
        let sub = vb.pp(base);
        Self::linear(in_dim, out_dim, sub)
    }

    /// `x·Wᵀ + b`. Dense delegates to `candle_nn::Linear`; packed delegates to the shared
    /// dequant-on-forward `QLinear` (sc-7702).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(l) => l.forward(x),
            Self::Packed(l) => l.forward(x),
        }
    }

    /// Whether this projection loaded directly from an MLX-packed tier (the packed path). Distinguishes
    /// a packed load from the dense path — used by the loader tests to assert a packed tier fired the
    /// packed path (and did not silently fall back to dense).
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
    use candle_gen::candle_core::{DType, Device};
    use std::collections::HashMap;

    /// Test-side MLX Q4 packer: per-element 4-bit codes → MLX u32 words (LSB-first nibbles), group 64.
    /// Returns `(wq [out, in/8] u32, scales [out, in/64], biases [out, in/64], affine grid [out, in])`
    /// — the exact packed-parts fixture the detect loaders consume, plus the affine grid they reproduce.
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

    /// **Packed-detect fires on the Chroma DiT key layout (incl. the `attn.to_out.0` nesting and a
    /// biased projection).** Writes a safetensors mimicking the real Chroma packed layout — a
    /// `to_out.0` triple *with* a dense `.bias` (the key remap that would silently fall back to dense if
    /// the loader `.pp()`'d past the `.scales` sibling) and a dense `to_q` sibling — and loads both
    /// through `linear_detect`. The `.scales`/`.biases`/`.bias` siblings must survive the `to_out.0`
    /// base string, `to_out.0` must load `Packed`, the dense sibling stays `Dense`, and the packed
    /// biased forward must reproduce the affine grid (+ bias) bit-exactly.
    #[test]
    fn linear_detect_fires_on_to_out_remap_and_leaves_dense_unchanged() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (64usize, 128usize);
        let (wq, s, b, grid) = q4_packed(out_dim, in_dim);
        let bias_vec: Vec<f32> = (0..out_dim).map(|i| 0.01 * i as f32).collect();

        let mut map: HashMap<String, Tensor> = HashMap::new();
        // The `attn.to_out.0` packed triple + dense `.bias` (the nested key the Chroma DiT threads as
        // one base — every Chroma DiT projection is biased).
        map.insert("attn.to_out.0.weight".into(), wq);
        map.insert("attn.to_out.0.scales".into(), s);
        map.insert("attn.to_out.0.biases".into(), b);
        map.insert(
            "attn.to_out.0.bias".into(),
            Tensor::from_vec(bias_vec.clone(), (out_dim,), &dev)?,
        );
        // A dense sibling (`to_q`) with no `.scales` → the dense path must stay unchanged.
        map.insert(
            "attn.to_q.weight".into(),
            Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev)?,
        );
        map.insert(
            "attn.to_q.bias".into(),
            Tensor::zeros((out_dim,), DType::F32, &dev)?,
        );

        let tmp =
            std::env::temp_dir().join(format!("sc9409_detect_{}.safetensors", std::process::id()));
        candle_gen::candle_core::safetensors::save(&map, &tmp)?;
        // SAFETY: we just wrote this file and nothing else touches it during the test.
        let st = unsafe { MmapedSafetensors::new(&tmp)? };
        let vb = VarBuilder::from_backend(Box::new(st), DType::F32, dev.clone());
        let attn = vb.pp("attn");

        // `to_out.0` — packed-detected through the remapped base (never `.pp("0")` past the sibling).
        let packed = QLinear::linear_detect(in_dim, out_dim, &attn, "to_out.0", true)?;
        assert!(packed.is_packed(), "`.scales` under to_out.0 ⇒ packed load");

        // `to_q` — dense (no `.scales`), path unchanged.
        let dense = QLinear::linear_detect(in_dim, out_dim, &attn, "to_q", true)?;
        assert!(!dense.is_packed(), "no `.scales` ⇒ dense path unchanged");
        assert!(matches!(dense, QLinear::Dense(_)));

        // The packed forward reproduces the affine grid + bias (bit-exact repack + dequant-on-forward).
        let grid_lin = QLinear::Dense(Linear::new(
            Tensor::from_vec(grid, (out_dim, in_dim), &dev)?,
            Some(Tensor::from_vec(bias_vec, (out_dim,), &dev)?),
        ));
        let x = Tensor::randn(0f32, 1f32, (4, in_dim), &dev)?;
        let cos = cosine(&packed.forward(&x)?, &grid_lin.forward(&x)?);
        assert!(cos > 0.99999, "packed vs affine-grid cosine {cos:.6}");

        std::fs::remove_file(&tmp).ok();
        Ok(())
    }

    /// `linear_detect_gs` threads a non-64 group size to the shared loader (the sc-9410 boogu path;
    /// Chroma itself is group 64, but the seam must not hard-code 64). A group-32 packed triple round-
    /// trips through the packed forward, proving the group size is honored end to end.
    #[test]
    fn linear_detect_gs_threads_group_size() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (32usize, 64usize);
        // group 32 fixture: 2 groups per row.
        const G: usize = 32;
        let codes: Vec<u8> = (0..out_dim * in_dim)
            .map(|i| ((i * 5 + 3) % 16) as u8)
            .collect();
        let groups = out_dim * in_dim / G;
        let scales: Vec<f32> = (0..groups).map(|g| 0.05 * (g as f32 + 1.0)).collect();
        let biases: Vec<f32> = (0..groups).map(|g| -0.3 - 0.1 * g as f32).collect();
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
        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert(
            "p.weight".into(),
            Tensor::from_vec(words, (out_dim, in_dim / 8), &dev)?,
        );
        map.insert(
            "p.scales".into(),
            Tensor::from_vec(scales, (out_dim, gpr), &dev)?,
        );
        map.insert(
            "p.biases".into(),
            Tensor::from_vec(biases, (out_dim, gpr), &dev)?,
        );

        let tmp =
            std::env::temp_dir().join(format!("sc9409_gs_{}.safetensors", std::process::id()));
        candle_gen::candle_core::safetensors::save(&map, &tmp)?;
        // SAFETY: freshly written, single-reader.
        let st = unsafe { MmapedSafetensors::new(&tmp)? };
        let vb = VarBuilder::from_backend(Box::new(st), DType::F32, dev.clone());

        let packed = QLinear::linear_detect_gs(in_dim, out_dim, &vb, "p", false, G)?;
        assert!(packed.is_packed());
        let grid_lin = QLinear::Dense(Linear::new(
            Tensor::from_vec(grid, (out_dim, in_dim), &dev)?,
            None,
        ));
        let x = Tensor::randn(0f32, 1f32, (3, in_dim), &dev)?;
        let cos = cosine(&packed.forward(&x)?, &grid_lin.forward(&x)?);
        assert!(cos > 0.99999, "group-32 packed vs grid cosine {cos:.6}");

        std::fs::remove_file(&tmp).ok();
        Ok(())
    }
}
