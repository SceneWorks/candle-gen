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
//! **Which sub-models pack (audited against the real tier headers, sc-9415):** only the
//! `transformer/` is packed — 846 packed `Linear` triples at group size 64 (every attn/ff/embedder/
//! `proj_out` projection; the RMSNorm weights `norm_q`/`norm_k`/`txt_norm` stay dense, they are not
//! `Linear`). The `text_encoder/` (the fused Qwen2.5-VL **language model `model.*` AND the vision
//! tower `visual.*`**) and the `vae/` ship **dense bf16 in every tier** — the MLX convert job
//! quantizes only the transformer, so those safetensors carry **zero `.scales`** (verified: the q4/q8
//! TE index lists 0 scales across all 729 keys, incl. the 390 `visual.*` keys). The multi-modal
//! "vision tower stays bf16" rule (this epic) therefore holds here in the strongest form: the whole
//! TE stays dense, and [`crate::text_encoder`] / [`crate::vision`] keep their stock `candle_nn` loaders
//! unchanged. [`crate::text_encoder::QwenTextEncoder::new`] additionally **guards** (errors loudly) if a
//! TE weight ever unexpectedly grows a `.scales` sibling, rather than silently reading u32 codes as
//! bf16 (the boogu `linear_guard_dense` precedent) — a future TE-packing tier must add a real packed
//! path, not fall through.
//!
//! Absent `.scales` (a dense bf16 tier — the stock diffusers `Qwen/Qwen-Image` snapshot, *and* the
//! separate InstantX `QwenControlNet` / alibaba-pai `QwenFunControlBranch` control checkpoints, which
//! are **not** part of the MLX tiers and are genuinely-deferred to sc-9517 for their own packed tiers)
//! the loader falls back to the **stock** `candle_nn::linear` path unchanged, so the same
//! [`crate::transformer`] serves both a dense snapshot and a packed one, and the mixed dense/packed DiT
//! loads with one call site per projection.
//!
//! **The packed forward dequantizes the weight into a dense matmul (sc-7702)** — it does *not* take
//! candle's int8 `QMatMul` fast path (`fast_mmq`), so a Q4 denoise stays coherent. The compute path is
//! the shared [`candle_gen::quant::QLinear`], which already owns that behavior; this module is only the
//! thin dense-or-packed **enum + detect** wrapper the vendored Qwen-Image DiT builds its projections
//! from (mirroring `candle-gen-chroma/src/quant.rs`). Qwen-Image has no *dense-tier on-the-fly* quant
//! path (the only quantized tier is the pre-packed MLX one), so — like the chroma seam — there is no
//! `quantize_onto`/load-time fold arm here.

use candle_gen::candle_core::{Result, Tensor};
use candle_gen::candle_nn::{Linear, Module, VarBuilder};
use candle_gen::quant as shared;

/// A Linear projection that is **dense** (the loaded bf16/f32 weight) or **packed** (loaded straight
/// from an MLX-packed tier via the shared [`candle_gen::quant::QLinear`], sc-9415). Built dense
/// ([`Self::linear`] / [`Self::linear_no_bias`]) or packed-detected ([`Self::linear_detect_gs`]); both
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

    /// A bias-less dense `[out, in]` projection from `vb` (`{prefix}.weight`) — the DiT `norm_out.linear`
    /// loads bias-less (the checkpoint bias is ignored).
    pub fn linear_no_bias(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self::Dense(candle_gen::candle_nn::linear_no_bias(
            in_dim, out_dim, vb,
        )?))
    }

    /// **Packed-detecting** `[out, in]` loader at an explicit MLX `group_size` (sc-9415): if
    /// `{base}.scales` is present in `vb` (a pre-quantized MLX tier), build a [`Self::Packed`] straight
    /// from the packed parts on `vb`'s device via the shared [`candle_gen::quant::lin_gs`] — **no dense
    /// weight is materialized**. Otherwise the **dense** path is taken unchanged (`{base}.weight`
    /// [+ `{base}.bias`]).
    ///
    /// `base` is the full dotted key prefix relative to `vb` (e.g. `attn.to_out.0`), so the
    /// `.scales`/`.biases`/`.bias` siblings survive any `to_out.0`-style key nesting: build the base
    /// string first, then detect — never `.pp()` past the scales sibling (the key-remap trap the shared
    /// loader guards). The `linear_detect_fires_on_to_out_remap*` test pins this on the real Qwen-Image
    /// `attn.to_out.0` layout.
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
        if bias {
            Self::linear(in_dim, out_dim, sub)
        } else {
            Self::linear_no_bias(in_dim, out_dim, sub)
        }
    }

    /// **Packed-detecting** `[out, in]` loader at the default MLX group size 64 (a thin
    /// [`Self::linear_detect_gs`]) — used by the tests and any caller that doesn't thread a config
    /// group size. Every Qwen-Image packed tier ships `quantization.group_size = 64`.
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
    use candle_gen::candle_core::{DType, Device};
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
    /// biased projection) and leaves a dense sibling unchanged.** Writes a safetensors mimicking the
    /// real Qwen-Image packed layout — a `to_out.0` triple *with* a dense `.bias` (the key remap that
    /// would silently fall back to dense if the loader `.pp()`'d past the `.scales` sibling) and a dense
    /// `to_q` sibling — and loads both through `linear_detect`. The `.scales`/`.biases`/`.bias` siblings
    /// must survive the `to_out.0` base string, `to_out.0` must load `Packed`, the dense sibling stays
    /// `Dense`, and the packed biased forward must reproduce the affine grid (+ bias) bit-exactly.
    #[test]
    fn linear_detect_fires_on_to_out_remap_and_leaves_dense_unchanged() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (64usize, 128usize);
        let (wq, s, b, grid) = q4_packed(out_dim, in_dim, 64);
        let bias_vec: Vec<f32> = (0..out_dim).map(|i| 0.01 * i as f32).collect();

        let mut map: HashMap<String, Tensor> = HashMap::new();
        // The `attn.to_out.0` packed triple + dense `.bias` (the nested key the Qwen DiT threads as one
        // base — the joint-attention `to_out` projection is biased).
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
            std::env::temp_dir().join(format!("sc9415_detect_{}.safetensors", std::process::id()));
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
        let grid_lin = QLinear::Dense(Linear::new(
            Tensor::from_vec(grid, (out_dim, in_dim), &dev)?,
            None,
        ));
        let x = Tensor::randn(0f32, 1f32, (2, in_dim), &dev)?;
        let cos = cosine(&packed.forward(&x)?, &grid_lin.forward(&x)?);
        assert!(cos > 0.99999, "bias-less packed vs grid cosine {cos:.6}");

        std::fs::remove_file(&tmp).ok();
        Ok(())
    }

    /// The **vision-tower / text-encoder dense guard** (sc-9415): [`guard_dense`] errors loudly when a
    /// weight it expects dense unexpectedly grows a `.scales` sibling (a packed weight on a path that
    /// reads bf16), and is a no-op on a genuinely-dense weight. The Qwen-Image MLX tiers keep the whole
    /// Qwen2.5-VL encoder dense, so the guard is a defensive tripwire, not a live branch.
    #[test]
    fn guard_dense_errors_on_unexpected_scales() -> Result<()> {
        let dev = Device::Cpu;
        let mut map: HashMap<String, Tensor> = HashMap::new();
        // A packed weight where only a dense one is expected (a "vision tower packed" tier).
        map.insert(
            "visual.blocks.0.attn.qkv.weight".into(),
            Tensor::zeros((8, 4), DType::U32, &dev)?,
        );
        map.insert(
            "visual.blocks.0.attn.qkv.scales".into(),
            Tensor::zeros((8, 1), DType::F32, &dev)?,
        );
        // A genuinely-dense sibling.
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

        // The packed weight trips the guard (loud error, not a silent u32-as-bf16 read).
        assert!(
            guard_dense(&vb, "visual.blocks.0.attn.qkv").is_err(),
            "a `.scales` sibling on a dense-path weight must error"
        );
        // The dense weight passes.
        assert!(guard_dense(&vb, "visual.blocks.0.attn.proj").is_ok());

        std::fs::remove_file(&tmp).ok();
        Ok(())
    }
}
