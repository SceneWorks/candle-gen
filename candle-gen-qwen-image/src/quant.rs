//! Qwen-Image DiT packed-load seam (sc-9415, sc-9089 umbrella) — the candle twin of the chroma
//! conversion (sc-9409) and the boogu/krea Qwen-VL conversions (sc-9410/9411), built on the shared
//! [`candle_gen::quant`] packed-load module (sc-9086).
//!
//! Qwen-Image ships **pre-quantized** MLX tiers (`SceneWorks/qwen-image-mlx` +
//! `qwen-image-edit-2511-mlx`, epic 8506) whose q4/q8 snapshots store each quantized DiT `Linear` as
//! the MLX packed triple `{base}.weight` (u32 codes) + `{base}.scales` + `{base}.biases` (plus the
//! dense `{base}.bias`). [`QLinear::linear_detect_gs`] packed-**detects** the `.scales` sibling and
//! builds the quantized weight **straight from the packed parts** on the target device via the shared
//! [`candle_gen::quant::lin_gs`] loader (Q4 → `Q4_1` lossless repack, Q8 → `Q8_0` requant). **No dense
//! bf16 weight is ever materialized** on the packed path — the q4 DiT lands ~5 GB directly rather than
//! staging the ~41 GB dense DiT first.
//!
//! **The projection type is now the SHARED [`candle_gen::quant::AdaptLinear`] (sc-11091).** It used to
//! be a per-crate dense-or-packed enum here; it is now re-exported as [`QLinear`] so the DiT
//! ([`crate::transformer`]) keeps its call sites unchanged, and — new for the edit lane — it carries an
//! optional **forward-time additive LoRA/LoKr residual**, so the **Qwen-Image-Edit-2511-Lightning**
//! distill (and user LoRAs) can apply on a **packed q4/q8** edit tier with the base kept packed (the
//! deltas ride unmerged, never folded into u32 codes — [`crate::adapters::install_additive`]). A dense
//! Edit snapshot still **folds** the delta (bit-exact) via [`crate::adapters::merge_adapters`]; with no
//! adapter the forward is byte-identical to before.
//!
//! **Which sub-models pack (audited against the real tier headers, sc-9415):** only the
//! `transformer/` is packed — 846 packed `Linear` triples at group size 64 (every attn/ff/embedder/
//! `proj_out` projection; the RMSNorm weights `norm_q`/`norm_k`/`txt_norm` stay dense, they are not
//! `Linear`). The `text_encoder/` (the fused Qwen2.5-VL **language model `model.*` AND the vision
//! tower `visual.*`**) and the `vae/` ship **dense bf16 in every tier**, so [`crate::text_encoder`] /
//! [`crate::vision`] keep their stock `candle_nn` loaders and [`guard_dense`] trips loudly if a TE
//! weight ever unexpectedly grows a `.scales` sibling.
//!
//! **The packed forward dequantizes the weight into a dense matmul (sc-7702)** — it does *not* take
//! candle's int8 `QMatMul` fast path (`fast_mmq`), so a Q4 denoise stays coherent. That behavior lives
//! in the shared [`candle_gen::quant::QLinear`] that [`AdaptLinear`]'s packed arm wraps.

use candle_gen::candle_core::Result;
use candle_gen::candle_nn::VarBuilder;

/// The residual-capable Qwen-Image DiT projection — a frozen dense/packed base plus optional
/// forward-time additive LoRA/LoKr residuals. **Now the shared [`candle_gen::quant::AdaptLinear`]**
/// (sc-11091), re-exported so `crate::quant::QLinear` keeps resolving for the DiT loader / transformer
/// / additive installer. Built dense ([`QLinear::linear`] / [`QLinear::linear_no_bias`]) or packed-
/// detecting ([`QLinear::linear_detect_gs`]); the base forward computes `x·Wᵀ (+ b)` and each attached
/// adapter adds `scale·((x·A)·B)` AS-IS — no dense weight materialized, so a packed q4/q8 base keeps
/// its footprint while the Lightning distill (or a user LoRA) applies *unmerged*.
pub use candle_gen::quant::AdaptLinear as QLinear;

/// Guard a **dense** VarBuilder sub-tree against an unexpected MLX-packed weight: error loudly if
/// `{base}.scales` is present under `vb` (sc-9415, the boogu `linear_guard_dense` precedent). The
/// Qwen-Image MLX tiers keep the whole fused Qwen2.5-VL text encoder (LM `model.*` **and** the vision
/// tower `visual.*`) dense bf16 — only the DiT packs. So the TE / vision loaders read `{base}.weight`
/// as their float dtype; if a future tier ever packed a TE weight, that u32 code stream would be
/// silently reinterpreted as bf16 garbage. This makes that a hard load error naming the offending key,
/// so a TE-packing tier is forced to add a real packed path rather than render noise.
pub fn guard_dense(vb: &VarBuilder, base: &str) -> Result<()> {
    if vb.contains_tensor(&format!("{base}.scales")) {
        candle_gen::candle_core::bail!(
            "qwen-image: `{base}.scales` present — this weight is MLX-packed, but the loader here is \
             the dense text-encoder/vision path (the Qwen-Image MLX tiers keep the whole Qwen2.5-VL \
             encoder dense; only the DiT packs). Reading its u32 codes as bf16 would be silent \
             garbage. A tier that packs the text encoder must add a real packed path (sc-9415)."
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::safetensors::MmapedSafetensors;
    use candle_gen::candle_core::{DType, Device, Tensor};
    use candle_gen::candle_nn::Linear;
    use std::collections::HashMap;

    /// Test-side MLX Q4 packer: per-element 4-bit codes → MLX u32 words (LSB-first nibbles), group `G`.
    /// Returns `(wq [out, in/8] u32, scales [out, in/G], biases [out, in/G], affine grid [out, in])` —
    /// the exact packed-parts fixture the detect loaders consume, plus the affine grid they reproduce.
    fn q4_packed(out_dim: usize, in_dim: usize, g: usize) -> (Tensor, Tensor, Tensor, Vec<f32>) {
        let dev = Device::Cpu;
        let codes: Vec<u8> = (0..out_dim * in_dim)
            .map(|i| ((i * 7 + i / 13) % 16) as u8)
            .collect();
        let groups = out_dim * in_dim / g;
        let scales: Vec<f32> = (0..groups).map(|gi| 0.0625 * (gi as f32 + 1.0)).collect();
        let biases: Vec<f32> = (0..groups).map(|gi| -0.5 - 0.25 * gi as f32).collect();
        let gpr = in_dim / g;
        let grid: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| {
                let (row, col) = (i / in_dim, i % in_dim);
                let gi = row * gpr + col / g;
                scales[gi] * codes[i] as f32 + biases[gi]
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

    /// **Packed-detect fires on the real Qwen-Image DiT key layout (the `attn.to_out.0` nesting + a
    /// biased projection) and leaves a dense sibling unchanged.** `to_out.0` must load packed, the dense
    /// `to_q` sibling stays dense, and the packed biased forward reproduces the affine grid (+ bias)
    /// bit-exactly. The shared `AdaptLinear` load path (sc-11091).
    #[test]
    fn linear_detect_fires_on_to_out_remap_and_leaves_dense_unchanged() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (64usize, 128usize);
        let (wq, s, b, grid) = q4_packed(out_dim, in_dim, 64);
        let bias_vec: Vec<f32> = (0..out_dim).map(|i| 0.01 * i as f32).collect();

        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert("attn.to_out.0.weight".into(), wq);
        map.insert("attn.to_out.0.scales".into(), s);
        map.insert("attn.to_out.0.biases".into(), b);
        map.insert(
            "attn.to_out.0.bias".into(),
            Tensor::from_vec(bias_vec.clone(), (out_dim,), &dev)?,
        );
        map.insert(
            "attn.to_q.weight".into(),
            Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev)?,
        );
        map.insert(
            "attn.to_q.bias".into(),
            Tensor::zeros((out_dim,), DType::F32, &dev)?,
        );

        let tmp =
            std::env::temp_dir().join(format!("sc9415_detect_{}.safetensors", std::process::id()));
        candle_gen::candle_core::safetensors::save(&map, &tmp)?;
        // SAFETY: we just wrote this file and nothing else touches it during the test.
        let st = unsafe { MmapedSafetensors::new(&tmp)? };
        let vb = VarBuilder::from_backend(Box::new(st), DType::F32, dev.clone());
        let attn = vb.pp("attn");

        let packed = QLinear::linear_detect(in_dim, out_dim, &attn, "to_out.0", true)?;
        assert!(packed.is_packed(), "`.scales` under to_out.0 ⇒ packed load");

        let dense = QLinear::linear_detect(in_dim, out_dim, &attn, "to_q", true)?;
        assert!(!dense.is_packed(), "no `.scales` ⇒ dense path unchanged");

        let grid_lin = QLinear::from_dense(
            Linear::new(
                Tensor::from_vec(grid, (out_dim, in_dim), &dev)?,
                Some(Tensor::from_vec(bias_vec, (out_dim,), &dev)?),
            ),
            in_dim,
            out_dim,
        );
        let x = Tensor::randn(0f32, 1f32, (4, in_dim), &dev)?;
        let cos = cosine(&packed.forward(&x)?, &grid_lin.forward(&x)?);
        assert!(cos > 0.99999, "packed vs affine-grid cosine {cos:.6}");

        std::fs::remove_file(&tmp).ok();
        Ok(())
    }

    /// `linear_detect_gs` threads the config `group_size` to the shared loader (sc-9415). A group-32
    /// packed triple round-trips through the packed forward, proving the group size is honored end to
    /// end (Qwen-Image tiers are group 64, but the seam must not hard-code 64).
    #[test]
    fn linear_detect_gs_threads_group_size() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (32usize, 64usize);
        let (wq, s, b, grid) = q4_packed(out_dim, in_dim, 32);
        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert("p.weight".into(), wq);
        map.insert("p.scales".into(), s);
        map.insert("p.biases".into(), b);

        let tmp =
            std::env::temp_dir().join(format!("sc9415_gs_{}.safetensors", std::process::id()));
        candle_gen::candle_core::safetensors::save(&map, &tmp)?;
        // SAFETY: freshly written, single-reader.
        let st = unsafe { MmapedSafetensors::new(&tmp)? };
        let vb = VarBuilder::from_backend(Box::new(st), DType::F32, dev.clone());

        let packed = QLinear::linear_detect_gs(in_dim, out_dim, &vb, "p", false, 32)?;
        assert!(packed.is_packed());
        let grid_lin = QLinear::from_dense(
            Linear::new(Tensor::from_vec(grid, (out_dim, in_dim), &dev)?, None),
            in_dim,
            out_dim,
        );
        let x = Tensor::randn(0f32, 1f32, (3, in_dim), &dev)?;
        let cos = cosine(&packed.forward(&x)?, &grid_lin.forward(&x)?);
        assert!(cos > 0.99999, "group-32 packed vs grid cosine {cos:.6}");

        std::fs::remove_file(&tmp).ok();
        Ok(())
    }

    /// The bias-less packed detect: `norm_out.linear` loads bias-less on the DiT. A packed triple with
    /// no `.bias` sibling detects packed and reproduces the (un-biased) affine grid.
    #[test]
    fn linear_detect_no_bias_packed() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (64usize, 128usize);
        let (wq, s, b, grid) = q4_packed(out_dim, in_dim, 64);
        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert("linear.weight".into(), wq);
        map.insert("linear.scales".into(), s);
        map.insert("linear.biases".into(), b);

        let tmp =
            std::env::temp_dir().join(format!("sc9415_nobias_{}.safetensors", std::process::id()));
        candle_gen::candle_core::safetensors::save(&map, &tmp)?;
        // SAFETY: freshly written, single-reader.
        let st = unsafe { MmapedSafetensors::new(&tmp)? };
        let vb = VarBuilder::from_backend(Box::new(st), DType::F32, dev.clone());

        let packed = QLinear::linear_detect(in_dim, out_dim, &vb, "linear", false)?;
        assert!(packed.is_packed(), "bias-less packed triple ⇒ packed load");
        let grid_lin = QLinear::from_dense(
            Linear::new(Tensor::from_vec(grid, (out_dim, in_dim), &dev)?, None),
            in_dim,
            out_dim,
        );
        let x = Tensor::randn(0f32, 1f32, (2, in_dim), &dev)?;
        let cos = cosine(&packed.forward(&x)?, &grid_lin.forward(&x)?);
        assert!(cos > 0.99999, "bias-less packed vs grid cosine {cos:.6}");

        std::fs::remove_file(&tmp).ok();
        Ok(())
    }

    /// The **vision-tower / text-encoder dense guard** (sc-9415): [`guard_dense`] errors loudly when a
    /// weight it expects dense unexpectedly grows a `.scales` sibling (a packed weight on a path that
    /// reads bf16), and is a no-op on a genuinely-dense weight.
    #[test]
    fn guard_dense_errors_on_unexpected_scales() -> Result<()> {
        let dev = Device::Cpu;
        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert(
            "visual.blocks.0.attn.qkv.weight".into(),
            Tensor::zeros((8, 4), DType::U32, &dev)?,
        );
        map.insert(
            "visual.blocks.0.attn.qkv.scales".into(),
            Tensor::zeros((8, 1), DType::F32, &dev)?,
        );
        map.insert(
            "visual.blocks.0.attn.proj.weight".into(),
            Tensor::randn(0f32, 1f32, (8, 8), &dev)?,
        );

        let tmp =
            std::env::temp_dir().join(format!("sc9415_guard_{}.safetensors", std::process::id()));
        candle_gen::candle_core::safetensors::save(&map, &tmp)?;
        // SAFETY: freshly written, single-reader.
        let st = unsafe { MmapedSafetensors::new(&tmp)? };
        let vb = VarBuilder::from_backend(Box::new(st), DType::F32, dev.clone());

        assert!(
            guard_dense(&vb, "visual.blocks.0.attn.qkv").is_err(),
            "a `.scales` sibling on a dense-path weight must error"
        );
        assert!(guard_dense(&vb, "visual.blocks.0.attn.proj").is_ok());

        std::fs::remove_file(&tmp).ok();
        Ok(())
    }
}
