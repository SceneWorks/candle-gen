//! Wan2.2 packed-load seam (sc-10025, sc-9089 umbrella twin for the wan crate) — the candle mirror of
//! the MLX Wan quant-matrix tiers (sc-9941 TI2V-5B / sc-9942 T2V-A14B / sc-9943 I2V-A14B, epic 8506),
//! built on the shared [`candle_gen::quant`] packed-load module (sc-9086).
//!
//! The Wan MLX tiers (`SceneWorks/wan2.2-{ti2v-5b,t2v-a14b,i2v-a14b}-mlx`, q4/q8/bf16) store each
//! quantized DiT attention / feed-forward `Linear` as an MLX packed triple: `{base}.weight` (u32
//! codes), `{base}.scales`, and `{base}.biases` (the dense `{base}.bias` rides alongside where present).
//! The hosted tiers pack at group 64 (MLX's default, the group the MLX build's `quantize_wan_transformer`
//! writes). **No dense weight is materialized** on the packed path — the resident `Q4_1`/`Q8_0` weight
//! dequantizes-on-forward (sc-7702), *not* candle's int8 `QMatMul` fast path.
//!
//! ## The residual-capable linear is now the SHARED core (sc-11091)
//!
//! The DiT projection type — a frozen dense/packed base plus zero or more **forward-time additive
//! LoRA/LoKr residuals** (sc-10094, epic 10043's ComfyUI-style unmerged branch) — used to be a
//! per-crate `QLinear` struct here. It has been **hoisted into [`candle_gen::quant::AdaptLinear`]**
//! (sc-11091) and merged with `candle-gen-anima`'s copy; this crate re-exports it as [`QLinear`] so the
//! DiT loader / transformer / additive installer keep their `crate::quant::QLinear` references
//! unchanged. The additive mechanism + its parity tests now live in `candle-gen/src/quant/adapt.rs`.
//! The Wan-native LoKr **fold** (dense base) still routes through `crate::adapters::merge_adapters`;
//! the packed additive path pushes plain LoRA residuals (LoKr/LoHa on packed stay rejected —
//! sc-10050/10051).
//!
//! ## Per-component packed / dense split (mirrors the MLX build: only the DiT experts quantize)
//!
//! | Component | Tier file | Packed surface |
//! |---|---|---|
//! | **`WanTransformer` DiT** (5B, or both A14B MoE experts) | `model` / `high_noise_model` / `low_noise_model` | **PACKED** (attn `to_q/k/v` + `to_out.0`, ffn `net.0.proj` + `net.2`, condition-embedder + `proj_out`) |
//! | **UMT5-XXL TE** | `t5_encoder.safetensors` | dense in the tier (the MLX build keeps the T5 dense) |
//! | **z16 Wan VAE** (3-D conv) | `vae.safetensors` | dense (3-D convs are never MLX-affine-packed) |
//!
//! The detect is by `{base}.scales` presence, not a hardcoded per-key list: one crate serves both the
//! current dense `Wan-AI/*-Diffusers` checkpoint (no `.scales` ⇒ every leaf dense, byte-identical to
//! before) and a packed tier. Actually ingesting the hosted `SceneWorks/wan2.2-*-mlx` file layout is
//! the separate loader effort tracked by sc-10026; the tests below validate the wiring with synthetic
//! packed fixtures on the crate's real DiT key layout.

use candle_gen::candle_core::Result;
use candle_gen::candle_nn::VarBuilder;

/// The Wan MLX tiers' quant group size (the hosted q4/q8 tiers pack at 64, MLX's default and the group
/// the MLX build's `quantize_wan_transformer` writes). The seam detects at this default; sc-10026 threads
/// the per-component config `group_size` at real tier ingestion.
pub const GROUP_SIZE: usize = candle_gen::quant::MLX_GROUP_SIZE; // 64

/// The residual-capable Wan DiT projection — a frozen dense/packed base plus stacked forward-time
/// additive LoRA residuals (sc-10094). **Now the shared [`candle_gen::quant::AdaptLinear`]** (sc-11091),
/// re-exported here so `crate::quant::QLinear` keeps resolving for the DiT loader, transformer, and
/// additive installer. Built dense ([`QLinear::linear`] / [`QLinear::linear_no_bias`]) or packed-
/// detecting ([`QLinear::linear_detect`] / [`QLinear::linear_detect_gs`]); the base forward computes
/// `x·Wᵀ (+ b)` and each attached adapter adds `scale·((x·A)·B)` AS-IS — no dense weight materialized,
/// so a packed q4/q8 base keeps its footprint while the Lightning distill (or a user LoRA) applies
/// *unmerged*.
pub use candle_gen::quant::AdaptLinear as QLinear;

/// Guard a **dense** VarBuilder sub-tree against an unexpected MLX-packed weight: error loudly if
/// `{base}.scales` is present under `vb` (sc-10025, the qwen `guard_dense` precedent). The Wan MLX tiers
/// keep the z16 3-D-conv VAE dense (only the DiT experts pack), so the VAE loaders read `{base}.weight`
/// as their float dtype; if a future tier ever packed a conv, that u32 code stream would be silently
/// reinterpreted as garbage. This makes that a hard load error naming the offending key.
#[cfg_attr(not(test), allow(dead_code))]
pub fn guard_dense(vb: &VarBuilder, base: &str) -> Result<()> {
    if vb.contains_tensor(&format!("{base}.scales")) {
        candle_gen::candle_core::bail!(
            "wan: `{base}.scales` present — this weight is MLX-packed, but the loader here is the dense \
             z16 VAE path (the Wan MLX tiers keep the 3-D-conv VAE dense; only the DiT experts pack). \
             Reading its u32 codes as a float would be silent garbage. A tier that packs the VAE must \
             add a real packed path (sc-10025)."
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::safetensors::MmapedSafetensors;
    use candle_gen::candle_core::{DType, Device, Tensor};
    use candle_gen::candle_nn::{Linear, Module};
    use candle_gen::quant as shared;
    use std::collections::HashMap;

    /// Test-side MLX Q4 packer at [`GROUP_SIZE`] (64): per-element 4-bit codes → MLX u32 words
    /// (LSB-first nibbles). Returns `(wq [out, in/8] u32, scales [out, in/G], biases [out, in/G], affine
    /// grid [out, in])` — the exact packed-parts fixture the loaders consume plus the grid they reproduce.
    fn q4_packed(out_dim: usize, in_dim: usize) -> (Tensor, Tensor, Tensor, Vec<f32>) {
        let dev = Device::Cpu;
        let g = GROUP_SIZE;
        let codes: Vec<u8> = (0..out_dim * in_dim)
            .map(|i| ((i * 7 + i / 13) % 16) as u8)
            .collect();
        let groups = out_dim * in_dim / g;
        let scales: Vec<f32> = (0..groups).map(|k| 0.0625 * (k as f32 + 1.0)).collect();
        let biases: Vec<f32> = (0..groups).map(|k| -0.5 - 0.25 * k as f32).collect();
        let gpr = in_dim / g;
        let grid: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| {
                let (row, col) = (i / in_dim, i % in_dim);
                let k = row * gpr + col / g;
                scales[k] * codes[i] as f32 + biases[k]
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
        let a = a
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let b = b
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (x, y) in a.iter().zip(&b) {
            dot += (*x as f64) * (*y as f64);
            na += (*x as f64) * (*x as f64);
            nb += (*y as f64) * (*y as f64);
        }
        (dot / (na.sqrt() * nb.sqrt() + 1e-12)) as f32
    }

    /// `linear_detect` fires the **packed** path on the real Wan DiT `attn1.to_out.0` key layout (the
    /// `to_out.0` nesting the diffusers checkpoint uses, with a dense `.bias`), a leaf with no `.scales`
    /// stays **dense**, and the packed forward matches the affine grid the pack represents (bit-exact
    /// repack + dequant-on-forward). The residual-capable shared `AdaptLinear` load path.
    #[test]
    fn linear_detect_packed_on_dit_key_layout() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (128usize, 256usize);
        let (wq, s, b, grid) = q4_packed(out_dim, in_dim);
        // A dense bias for the packed to_out.0 (the tier ships `to_out.0.bias` alongside the triple).
        let out_bias = Tensor::randn(0f32, 1f32, (out_dim,), &dev)?;

        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert("blocks.0.attn1.to_out.0.weight".into(), wq);
        map.insert("blocks.0.attn1.to_out.0.scales".into(), s);
        map.insert("blocks.0.attn1.to_out.0.biases".into(), b);
        map.insert("blocks.0.attn1.to_out.0.bias".into(), out_bias.clone());
        // A dense projection (`to_q`) — no `.scales`; the seam must take the dense arm unchanged.
        map.insert(
            "blocks.0.attn1.to_q.weight".into(),
            Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev)?,
        );
        map.insert(
            "blocks.0.attn1.to_q.bias".into(),
            Tensor::randn(0f32, 1f32, (out_dim,), &dev)?,
        );

        let tmp =
            std::env::temp_dir().join(format!("sc10025_dit_{}.safetensors", std::process::id()));
        candle_gen::candle_core::safetensors::save(&map, &tmp)?;
        // SAFETY: freshly written, single-reader for the test.
        let st = unsafe { MmapedSafetensors::new(&tmp)? };
        let vb = VarBuilder::from_backend(Box::new(st), DType::F32, dev.clone());
        let blk = vb.pp("blocks.0.attn1");

        let packed = QLinear::linear_detect(in_dim, out_dim, &blk, "to_out.0", true)?;
        assert!(
            packed.is_packed(),
            "`.scales` under to_out.0 ⇒ packed load (not a silent dense fallback)"
        );
        let dense = QLinear::linear_detect(in_dim, out_dim, &blk, "to_q", true)?;
        assert!(
            !dense.is_packed(),
            "no `.scales` ⇒ dense to_q, path unchanged"
        );

        // The packed forward reproduces the affine grid (+ the dense bias) bit-exactly.
        let grid_lin = QLinear::from_dense(
            Linear::new(
                Tensor::from_vec(grid, (out_dim, in_dim), &dev)?,
                Some(out_bias),
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

    /// The dense arm of `linear_detect` is byte-identical to the legacy `candle_nn::linear` read — a dense
    /// checkpoint (no `.scales` anywhere) loads every leaf dense, unchanged. Confirms the current dense
    /// `Wan-AI/*-Diffusers` checkpoint path is untouched.
    #[test]
    fn linear_detect_dense_path_unchanged() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (32usize, 64usize);
        let w = Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev)?;
        let b = Tensor::randn(0f32, 1f32, (out_dim,), &dev)?;

        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert("proj.weight".into(), w.clone());
        map.insert("proj.bias".into(), b.clone());
        let tmp =
            std::env::temp_dir().join(format!("sc10025_dense_{}.safetensors", std::process::id()));
        candle_gen::candle_core::safetensors::save(&map, &tmp)?;
        // SAFETY: freshly written, single-reader.
        let st = unsafe { MmapedSafetensors::new(&tmp)? };
        let vb = VarBuilder::from_backend(Box::new(st), DType::F32, dev.clone());

        let lin = QLinear::linear_detect(in_dim, out_dim, &vb, "proj", true)?;
        assert!(!lin.is_packed(), "no `.scales` ⇒ dense");
        // Reference: the exact legacy read.
        let ref_lin = Linear::new(w, Some(b));
        let x = Tensor::randn(0f32, 1f32, (4, in_dim), &dev)?;
        let dev_max = (lin.forward(&x)?.sub(&ref_lin.forward(&x)?)?)
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert_eq!(
            dev_max, 0.0,
            "dense arm deviates from the legacy linear read"
        );

        std::fs::remove_file(&tmp).ok();
        Ok(())
    }

    /// The shared [`candle_gen::quant::embedding_gs`] fires the packed path on the UMT5 `shared.scales`
    /// sibling and reproduces the affine grid rows exactly; a leaf with no `.scales` loads dense. This is
    /// the loader the UMT5 encoder uses for its `shared` embedding.
    #[test]
    fn shared_embedding_packed_detect_on_umt5_layout() -> Result<()> {
        let dev = Device::Cpu;
        let (vocab, hidden) = (64usize, 128usize);
        let (wq, s, b, grid) = q4_packed(vocab, hidden);

        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert("shared.weight".into(), wq);
        map.insert("shared.scales".into(), s);
        map.insert("shared.biases".into(), b);
        map.insert(
            "dense_shared.weight".into(),
            Tensor::from_vec(grid.clone(), (vocab, hidden), &dev)?,
        );
        let tmp =
            std::env::temp_dir().join(format!("sc10025_emb_{}.safetensors", std::process::id()));
        candle_gen::candle_core::safetensors::save(&map, &tmp)?;
        // SAFETY: freshly written, single-reader.
        let st = unsafe { MmapedSafetensors::new(&tmp)? };
        let vb = VarBuilder::from_backend(Box::new(st), DType::F32, dev.clone());

        let packed = shared::embedding_gs(&vb, "shared", vocab, hidden, GROUP_SIZE)?;
        let dense = shared::embedding_gs(&vb, "dense_shared", vocab, hidden, GROUP_SIZE)?;
        // Row-select all ids and compare against the affine grid (bit-exact repack for the packed arm;
        // identity for the dense arm).
        let ids = Tensor::arange(0u32, vocab as u32, &dev)?;
        let grid_t = Tensor::from_vec(grid, (vocab, hidden), &dev)?;
        for (label, e) in [("packed", &packed), ("dense", &dense)] {
            let rows = e.forward(&ids)?;
            let dev_max = (rows.sub(&grid_t)?).abs()?.max_all()?.to_scalar::<f32>()?;
            assert_eq!(
                dev_max, 0.0,
                "{label} embed rows deviate from the affine grid"
            );
        }

        std::fs::remove_file(&tmp).ok();
        Ok(())
    }

    /// `quantize` is a **no-op** on a packed `shared::QLinear` — an MLX-packed weight must never be
    /// double-quantized. The stored `Q4_1` weight and the forward stay unchanged.
    #[test]
    fn packed_quantize_is_noop() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (64usize, 128usize);
        let (wq, s, b, _grid) = q4_packed(out_dim, in_dim);

        let mut packed = shared::QLinear::from_packed(&wq, &s, &b, None, &dev)?;
        let x = Tensor::randn(0f32, 1f32, (4, in_dim), &dev)?;
        let before = packed.forward(&x)?;
        packed.quantize(candle_gen::gen_core::Quant::Q4)?; // must no-op, not re-quantize
        let after = packed.forward(&x)?;
        let dev_max = (before.sub(&after)?).abs()?.max_all()?.to_scalar::<f32>()?;
        assert_eq!(dev_max, 0.0, "no-op quantize changed the packed forward");
        Ok(())
    }

    /// `guard_dense` errors loudly when a `.scales` sibling appears where a dense weight is expected (a
    /// z16 Wan VAE conv), and passes cleanly otherwise. The guard so a tier that ever packs a conv doesn't
    /// silently load u32-code garbage.
    #[test]
    fn guard_dense_errors_on_packed_conv() -> Result<()> {
        let dev = Device::Cpu;
        let (wq, s, b, _grid) = q4_packed(16, 64);

        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert("conv_in.weight".into(), wq);
        map.insert("conv_in.scales".into(), s);
        map.insert("conv_in.biases".into(), b);
        map.insert(
            "conv_out.weight".into(),
            Tensor::randn(0f32, 1f32, (8, 8), &dev)?,
        );
        let tmp =
            std::env::temp_dir().join(format!("sc10025_guard_{}.safetensors", std::process::id()));
        candle_gen::candle_core::safetensors::save(&map, &tmp)?;
        // SAFETY: freshly written, single-reader.
        let st = unsafe { MmapedSafetensors::new(&tmp)? };
        let vb = VarBuilder::from_backend(Box::new(st), DType::F32, dev.clone());

        assert!(
            guard_dense(&vb, "conv_in").is_err(),
            "guard must error on a `.scales` sibling where a dense conv is expected"
        );
        guard_dense(&vb, "conv_out")?; // clean dense leaf passes

        std::fs::remove_file(&tmp).ok();
        Ok(())
    }
}
