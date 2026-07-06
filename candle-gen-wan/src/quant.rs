//! Wan2.2 packed-load seam (sc-10025, sc-9089 umbrella twin for the wan crate) — the candle mirror of
//! the MLX Wan quant-matrix tiers (sc-9941 TI2V-5B / sc-9942 T2V-A14B / sc-9943 I2V-A14B, epic 8506),
//! built on the shared [`candle_gen::quant`] packed-load module (sc-9086). Direct sibling of the
//! qwen-image (sc-9415), sdxl (sc-9416) and ltx-video (sc-9417) seams — this one follows the qwen
//! `QLinear::linear_detect` shape (dims-aware dense arm via `candle_nn::linear`, so the dense path stays
//! byte-identical AND works on the crate's `VarMap`/shape-synthesizing test backends).
//!
//! The Wan MLX tiers (`SceneWorks/wan2.2-{ti2v-5b,t2v-a14b,i2v-a14b}-mlx`, q4/q8/bf16) store each
//! quantized DiT attention / feed-forward `Linear` as an MLX packed triple: `{base}.weight` (u32
//! codes), `{base}.scales`, and `{base}.biases` (the dense `{base}.bias` rides alongside where present).
//! The hosted tiers pack at group 64 (MLX's default, the group the MLX build's `quantize_wan_transformer`
//! writes). **No dense weight is materialized** on the packed path — the resident `Q4_1`/`Q8_0` weight
//! dequantizes-on-forward (sc-7702), *not* candle's int8 `QMatMul` fast path.
//!
//! ## Per-component packed / dense split (mirrors the MLX build: only the DiT experts quantize)
//!
//! | Component | Tier file | Packed surface |
//! |---|---|---|
//! | **`WanTransformer` DiT** (5B, or both A14B MoE experts) | `model` / `high_noise_model` / `low_noise_model` | **PACKED** (attn `to_q/k/v` + `to_out.0`, ffn `net.0.proj` + `net.2`, condition-embedder + `proj_out`) |
//! | **UMT5-XXL TE** | `t5_encoder.safetensors` | dense in the tier (the MLX build keeps the T5 dense) |
//! | **z16 Wan VAE** (3-D conv) | `vae.safetensors` | dense (3-D convs are never MLX-affine-packed) |
//!
//! The shared [`WanTransformer`](crate::transformer::WanTransformer) is the DiT for the TI2V-5B and for
//! **both** A14B experts, so routing its Linear surface through the packed-detect loader covers all
//! three quant-matrix models at once. The UMT5 encoder is dense in the hosted tier — routing it through
//! the shared [`candle_gen::quant::embedding_gs`] / this loader only future-proofs the surface + closes
//! the guard (mirrors qwen routing its dense Qwen2.5-VL TE). The detect is by `{base}.scales` presence,
//! not a hardcoded per-key list: one crate serves both the current dense `Wan-AI/*-Diffusers` checkpoint
//! (no `.scales` ⇒ every leaf dense, byte-identical to before) and a packed tier.
//!
//! ## Scope boundary — packed-detect seam only; tier *ingestion* is the follow-up (sc-10026)
//!
//! This seam makes every DiT / UMT5 Linear (and the UMT5 `shared` embedding) **packed-detect** on the
//! crate's OWN diffusers key layout at [`GROUP_SIZE`] (64): the current dense checkpoint has no
//! `.scales`, so every leaf takes the dense arm **byte-identically** to before, and a `.scales` sibling
//! at any of those keys fires the packed path. **Actually loading the hosted `SceneWorks/wan2.2-*-mlx`
//! q4/q8 tiers is a separate loader effort** — those tiers ship the MLX file layout
//! (`high_noise_model.safetensors` / `low_noise_model.safetensors` / `t5_encoder.safetensors` /
//! `vae.safetensors`, not the diffusers `transformer/` + `transformer_2/` + `text_encoder/` + `vae/`
//! component dirs) and MLX key names, and thread the config `group_size` — so resolving the tier
//! subdir, remapping the layout/keys, **and the real packed GPU video render** are deferred to and
//! tracked by **sc-10026**. The tests below validate the wiring with **synthetic** packed fixtures on
//! the crate's real DiT key layout — they prove the packed-detect seam fires, not a real tier ingest.

use candle_gen::candle_core::{Result, Tensor};
use candle_gen::candle_nn::{Linear, Module, VarBuilder};
use candle_gen::quant as shared;

/// The Wan MLX tiers' quant group size (the hosted q4/q8 tiers pack at 64, MLX's default and the group
/// the MLX build's `quantize_wan_transformer` writes). The seam detects at this default; sc-10026 threads
/// the per-component config `group_size` at real tier ingestion.
pub const GROUP_SIZE: usize = shared::MLX_GROUP_SIZE; // 64

/// A Linear projection — a **base** weight (dense, the legacy per-crate path, or **packed** straight from
/// an MLX-packed tier via the shared [`candle_gen::quant::QLinear`]) plus zero or more forward-time
/// **additive LoRA/LoKr adapters** (sc-10094). Built dense ([`Self::linear`] / [`Self::linear_no_bias`]) or
/// packed-detected ([`Self::linear_detect`]); the base forward computes `x·Wᵀ + b`, and each attached
/// adapter adds a residual `scale·((x·A)·B)` (LoRA) / `scale·x·ΔWᵀ` (LoKr) computed AS-IS against the base
/// weight — **no dense weight is materialized**, so a packed q4/q8 base keeps its footprint while the
/// mandatory Lightning distill (or a user LoRA) applies *unmerged* (the ComfyUI-style branch epic 10043
/// delivers; the candle twin of mlx-gen's `AdaptableLinear`). With no adapter the forward is byte-identical
/// to the pre-sc-10094 dense/packed path.
pub struct QLinear {
    base: Base,
    in_features: usize,
    out_features: usize,
    /// Forward-time additive residuals applied in push order (adapters stack). Empty on the pure
    /// inference/dense path.
    adapters: Vec<Adapter>,
}

/// The frozen base weight behind a [`QLinear`] — **dense** or **MLX-packed**. Both compute `x·Wᵀ + b`; the
/// packed arm dequantizes-on-forward into a dense matmul (sc-7702, *not* the int8 `QMatMul` fast path).
enum Base {
    Dense(Linear),
    Packed(shared::QLinear),
}

/// A forward-time additive adapter residual (sc-10094) — the base weight is **never mutated** (the
/// ComfyUI-style unmerged branch, candle twin of mlx-gen's `Adapter`). Factors are held **f32** and cast to
/// the activation dtype per forward (they are tiny — `[in,r]`/`[r,out]` — so the cast is cheap).
enum Adapter {
    /// LoRA residual `scale·(x·a)·b`: `a` `[in, rank]` (= `downᵀ`), `b` `[rank, out]` (= `upᵀ` with the
    /// `alpha/rank` ratio folded in at resolution). The **deferred two-small-matmul** form — never the
    /// `[out,in]` product — so it stays memory-free on any quant (a q4 base keeps its q4 footprint).
    Lora { a: Tensor, b: Tensor, scale: f64 },
    /// LoKr / full-delta residual `scale·x·δᵀ`: `delta` `[out, in]` f32 reconstructed from the Kronecker
    /// factors at the base's shape. It materializes a dense `[out,in]` delta, so it is **dense-base only**
    /// (a packed tier rejects LoKr/LoHa — deferred to sc-10050/10051); kept for dense additive==folded
    /// parity.
    Lokr { delta: Tensor, scale: f64 },
}

impl Base {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Base::Dense(l) => l.forward(x),
            Base::Packed(l) => l.forward(x),
        }
    }

    fn is_packed(&self) -> bool {
        matches!(self, Base::Packed(_))
    }
}

impl Adapter {
    /// The residual this adapter adds to the base forward, in the activation dtype of `x`.
    fn residual(&self, x: &Tensor) -> Result<Tensor> {
        let xd = x.dtype();
        match self {
            // Two small matmuls: `(x·a)·b`, never the `[out,in]` product (memory-free on any quant).
            Adapter::Lora { a, b, scale } => {
                let r = x
                    .broadcast_matmul(&a.to_dtype(xd)?)?
                    .broadcast_matmul(&b.to_dtype(xd)?)?;
                r * *scale
            }
            // Reconstructed dense delta: `x·δᵀ` (dense base only).
            Adapter::Lokr { delta, scale } => {
                let r = x.broadcast_matmul(&delta.to_dtype(xd)?.t()?)?;
                r * *scale
            }
        }
    }
}

impl QLinear {
    fn wrap_dense(l: Linear, in_dim: usize, out_dim: usize) -> Self {
        Self {
            base: Base::Dense(l),
            in_features: in_dim,
            out_features: out_dim,
            adapters: Vec::new(),
        }
    }

    /// A biased dense `[out, in]` projection from `vb` (`{prefix}.weight` + `{prefix}.bias`), shape-checked
    /// exactly like the legacy `transformer::linear` — so it loads unchanged on the crate's `VarMap`-backed
    /// test fixtures.
    pub fn linear(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self::wrap_dense(
            candle_gen::candle_nn::linear(in_dim, out_dim, vb)?,
            in_dim,
            out_dim,
        ))
    }

    /// A bias-less dense `[out, in]` projection from `vb` (`{prefix}.weight`) — the UMT5 q/k/v/o + FFN
    /// projections load bias-less (the legacy `text_encoder::linear_no_bias`).
    pub fn linear_no_bias(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self::wrap_dense(
            candle_gen::candle_nn::linear_no_bias(in_dim, out_dim, vb)?,
            in_dim,
            out_dim,
        ))
    }

    /// **Packed-detecting** `[out, in]` loader at an explicit MLX `group_size` (sc-10025): if
    /// `{base}.scales` is present under `vb` (a pre-quantized MLX tier), build a [`Self::Packed`] straight
    /// from the packed parts on `vb`'s device via the shared [`candle_gen::quant::lin_gs`] — **no dense
    /// weight is materialized**. Otherwise the **dense** path is taken unchanged (`{base}.weight`
    /// [+ `{base}.bias`], shape-checked).
    ///
    /// `base` is the full dotted key prefix relative to `vb` (e.g. `to_out.0`), so the
    /// `.scales`/`.biases`/`.bias` siblings survive any `to_out.0`-style nesting: build the base string
    /// first, then detect — never `.pp()` past the scales sibling (the key-remap trap the shared loader
    /// guards).
    pub fn linear_detect_gs(
        in_dim: usize,
        out_dim: usize,
        vb: &VarBuilder,
        base: &str,
        bias: bool,
        group_size: usize,
    ) -> Result<Self> {
        if vb.contains_tensor(&format!("{base}.scales")) {
            return Ok(Self {
                base: Base::Packed(shared::lin_gs(vb, base, in_dim, out_dim, bias, group_size)?),
                in_features: in_dim,
                out_features: out_dim,
                adapters: Vec::new(),
            });
        }
        let sub = vb.pp(base);
        if bias {
            Self::linear(in_dim, out_dim, sub)
        } else {
            Self::linear_no_bias(in_dim, out_dim, sub)
        }
    }

    /// **Packed-detecting** `[out, in]` loader at the default MLX [`GROUP_SIZE`] (64) — the seam entry the
    /// DiT / UMT5 call sites use (every hosted Wan tier packs at 64; sc-10026 threads the config group at
    /// real ingestion). Thin wrapper over [`Self::linear_detect_gs`].
    pub fn linear_detect(
        in_dim: usize,
        out_dim: usize,
        vb: &VarBuilder,
        base: &str,
        bias: bool,
    ) -> Result<Self> {
        Self::linear_detect_gs(in_dim, out_dim, vb, base, bias, GROUP_SIZE)
    }

    /// `x·Wᵀ + b` plus every attached additive adapter residual, in push order (sc-10094). The base
    /// (dense `candle_nn::Linear` or the shared dequant-on-forward packed `QLinear`, sc-7702) is used
    /// AS-IS; each adapter adds `scale·((x·A)·B)` (LoRA) / `scale·x·ΔWᵀ` (LoKr). With no adapter the
    /// output is byte-identical to the pre-sc-10094 base forward.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut y = self.base.forward(x)?;
        for a in &self.adapters {
            y = (y + a.residual(x)?)?;
        }
        Ok(y)
    }

    /// Whether this projection loaded directly from an MLX-packed tier (the packed path) — used by the
    /// tests to assert a packed tier fired the packed path (not a silent dense fallback), and by the
    /// additive-adapter router to reject LoKr/LoHa on a packed base (sc-10094; deferred to sc-10050/10051).
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn is_packed(&self) -> bool {
        self.base.is_packed()
    }

    /// The base projection's `(out_features, in_features)` — the shape a LoKr delta is reconstructed at,
    /// recoverable even from a packed base (the logical dims are captured at construction, not read back
    /// from the quantized weight).
    pub fn base_shape(&self) -> (usize, usize) {
        (self.out_features, self.in_features)
    }

    /// The projection's contraction (`in_features`) — the last-dim an `[in, rank]` LoRA `a` factor
    /// contracts against.
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// The projection's output width (`out_features`).
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Attach a **LoRA** residual `scale·(x·a)·b` (sc-10094): `a` `[in, rank]` (= `downᵀ`), `b`
    /// `[rank, out]` (= `upᵀ` with the `alpha/rank` ratio already folded in), `scale` the caller's
    /// per-adapter strength. Multiple pushes stack. Valid on **any** base (dense or packed) — the base
    /// weight is untouched, so a packed q4/q8 tier keeps its footprint.
    pub fn push_lora(&mut self, a: Tensor, b: Tensor, scale: f64) {
        self.adapters.push(Adapter::Lora { a, b, scale });
    }

    /// Attach a **full-delta** residual `scale·x·δᵀ` (sc-10094) from a reconstructed `[out, in]` LoKr/LoHa
    /// delta. Dense-base only — a packed tier rejects LoKr/LoHa upstream (the delta would need the base's
    /// dense grid); kept for dense additive==folded parity.
    pub fn push_delta(&mut self, delta: Tensor, scale: f64) {
        self.adapters.push(Adapter::Lokr { delta, scale });
    }

    /// Whether any additive adapter is attached.
    pub fn is_adapted(&self) -> bool {
        !self.adapters.is_empty()
    }
}

impl Module for QLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        QLinear::forward(self, x)
    }
}

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
    use candle_gen::candle_core::{DType, Device};
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
    /// repack + dequant-on-forward).
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
        let grid_lin = QLinear::wrap_dense(
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

    /// Build a packed wan [`QLinear`] on the `p` key via the round-trip the DiT loader takes
    /// (`linear_detect` on a written `.scales`/`.biases`/`.weight` triple) — the packed base the additive
    /// tests attach a LoRA onto.
    fn packed_qlinear(out_dim: usize, in_dim: usize) -> (QLinear, Vec<f32>) {
        let dev = Device::Cpu;
        let (wq, s, b, grid) = q4_packed(out_dim, in_dim);
        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert("p.weight".into(), wq);
        map.insert("p.scales".into(), s);
        map.insert("p.biases".into(), b);
        let tmp =
            std::env::temp_dir().join(format!("sc10094_packed_{}.safetensors", std::process::id()));
        candle_gen::candle_core::safetensors::save(&map, &tmp).unwrap();
        // SAFETY: freshly written, single-reader for the test.
        let st = unsafe { MmapedSafetensors::new(&tmp).unwrap() };
        let vb = VarBuilder::from_backend(Box::new(st), DType::F32, dev);
        let q = QLinear::linear_detect(in_dim, out_dim, &vb, "p", false).unwrap();
        std::fs::remove_file(&tmp).ok();
        (q, grid)
    }

    /// The additive LoRA branch (`base(x) + scale·(x·a)·b`, `a = downᵀ`, `b = upᵀ` with `alpha/rank`
    /// folded in) reproduces the **folded** merge `x·(W + δ)ᵀ` with `δ = (alpha/rank)·scale·(up·down)`,
    /// on a dense f32 base — the sc-10094 additive==folded parity (tight in f32).
    #[test]
    fn additive_lora_matches_folded_dense() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim, rank) = (48usize, 64usize, 4usize);
        let w = Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev)?;
        let down = Tensor::randn(0f32, 1f32, (rank, in_dim), &dev)?; // A [rank, in]
        let up = Tensor::randn(0f32, 1f32, (out_dim, rank), &dev)?; // B [out, rank]
        let (alpha, user_scale) = (8.0f64, 0.7f64);
        let ratio = alpha / rank as f64; // 2.0

        // Additive: a = downᵀ [in, rank], b = (upᵀ)·ratio [rank, out], scale = user strength.
        let a = down.t()?.contiguous()?;
        let b = (up.t()?.contiguous()? * ratio)?;
        let mut q = QLinear::wrap_dense(Linear::new(w.clone(), None), in_dim, out_dim);
        q.push_lora(a, b, user_scale);
        assert!(q.is_adapted());

        // Folded: δ = ratio·user_scale·(up·down); W_merged = W + δ.
        let delta = ((up.matmul(&down)?) * (ratio * user_scale))?;
        let folded = QLinear::wrap_dense(Linear::new((w + delta)?, None), in_dim, out_dim);

        let x = Tensor::randn(0f32, 1f32, (3, in_dim), &dev)?;
        let dev_max = (q.forward(&x)?.sub(&folded.forward(&x)?)?)
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(dev_max < 1e-4, "additive vs folded deviates by {dev_max}");
        Ok(())
    }

    /// A scale-0 additive LoRA is a byte-exact no-op — the adapter is attached but contributes nothing,
    /// so the forward equals the un-adapted base exactly.
    #[test]
    fn additive_scale_zero_is_noop() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim, rank) = (32usize, 40usize, 4usize);
        let w = Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev)?;
        let base = QLinear::wrap_dense(Linear::new(w.clone(), None), in_dim, out_dim);
        let mut adapted = QLinear::wrap_dense(Linear::new(w, None), in_dim, out_dim);
        adapted.push_lora(
            Tensor::randn(0f32, 1f32, (in_dim, rank), &dev)?,
            Tensor::randn(0f32, 1f32, (rank, out_dim), &dev)?,
            0.0,
        );
        let x = Tensor::randn(0f32, 1f32, (3, in_dim), &dev)?;
        let dev_max = (adapted.forward(&x)?.sub(&base.forward(&x)?)?)
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert_eq!(dev_max, 0.0, "scale-0 additive LoRA must be an exact no-op");
        Ok(())
    }

    /// A LoRA applied additively onto a **packed** base runs with no error, keeps the base packed (no
    /// dense weight materialized), and **shifts** the output away from the un-adapted packed forward —
    /// the core sc-10094 acceptance on a quantized tier. The output stays finite.
    #[test]
    fn additive_lora_on_packed_shifts_and_finite() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim, rank) = (64usize, 128usize, 8usize);
        let (packed_base, _grid) = packed_qlinear(out_dim, in_dim);
        assert!(packed_base.is_packed());

        let mut adapted = {
            let (q, _) = packed_qlinear(out_dim, in_dim);
            q
        };
        // Non-degenerate factors so the residual is a real shift.
        let a = (Tensor::randn(0f32, 1f32, (in_dim, rank), &dev)? * 0.1)?;
        let b = (Tensor::randn(0f32, 1f32, (rank, out_dim), &dev)? * 0.1)?;
        adapted.push_lora(a, b, 1.0);
        assert!(adapted.is_packed(), "adapter must not un-pack the base");
        assert!(adapted.is_adapted());

        let x = Tensor::randn(0f32, 1f32, (4, in_dim), &dev)?;
        let base_y = packed_base.forward(&x)?;
        let adapted_y = adapted.forward(&x)?;
        let shift = (adapted_y.sub(&base_y)?)
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(
            shift > 1e-4,
            "additive LoRA on packed did not shift ({shift})"
        );
        let maxabs = adapted_y.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(maxabs.is_finite(), "packed additive output non-finite");
        Ok(())
    }
}
