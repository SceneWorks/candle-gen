//! The full Ideogram 4 DiT: token composition (`[text ; image]`), scalar-`t` AdaLN conditioning, 34
//! blocks, and the affine-less final layer. Port of `Ideogram4Transformer.forward`.
//!
//! Token roles (`indicator`): `LLM_TOKEN_INDICATOR = 3` (text), `OUTPUT_IMAGE_INDICATOR = 2`
//! (image). Text positions carry the projected Qwen3-VL features (`llm_cond_proj`); image positions
//! carry the patchified noise latents (`input_proj`). Both streams live in one sequence, mixed every
//! block by full (segment-masked) attention + interleaved 3D MRoPE.

use candle_gen::candle_core::{DType, Result, Tensor, D};

use super::block::Ideogram4Block;
use super::mrope::Ideogram4MRoPE;
use super::rmsnorm;
use crate::config::Ideogram4DitConfig;
use crate::loader::{embedding_detect, linear_detect, Weights};
use crate::quant::{QEmbedding, QLinear};

/// Token role constants (upstream `ideogram4.constants`).
const OUTPUT_IMAGE_INDICATOR: i64 = 2;
const LLM_TOKEN_INDICATOR: i64 = 3;

/// `llm_cond_norm` and the final LayerNorm both use eps 1e-6 (upstream).
const COND_NORM_EPS: f64 = 1e-6;
const FINAL_NORM_EPS: f64 = 1e-6;

pub struct Ideogram4Transformer {
    input_proj: QLinear,
    llm_cond_norm: Tensor,
    llm_cond_proj: QLinear,
    t_mlp_in: QLinear,
    t_mlp_out: QLinear,
    adaln_proj: QLinear,
    embed_image_indicator: QEmbedding,
    rotary_emb: Ideogram4MRoPE,
    layers: Vec<Ideogram4Block>,
    final_adaln: QLinear,
    final_linear: QLinear,
    /// Sinusoidal frequencies for the `t` embedding (`[1, emb_dim/2]`, f32).
    t_freqs: Tensor,
    dtype: DType,
    /// Per-render step-invariant-tensor cache (sc-8992 / F-012). The role masks (from `indicator`), the
    /// segment attention mask (from `segment_ids`), and the MRoPE `(cos, sin)` tables (from
    /// `position_ids`) depend only on the fixed packing geometry — not on σ / `t` / the current latent —
    /// so they are identical across every denoise step (×2 under CFG). This crate previously rebuilt the
    /// `[B,1,L,L]` mask in a host loop and round-tripped `indicator` device→host **every** forward.
    /// Cache them keyed on the (loop-invariant) inputs' host contents. `Mutex` (not `RefCell`): the DiT
    /// is used behind a shared cache and must stay `Send + Sync`.
    cond_cache: std::sync::Mutex<Option<PreparedCond>>,
}

/// The step-invariant conditioning tensors prepared once per render (sc-8992). `seg_mask = None` when
/// every token shares a segment id — the additive mask is provably all-zeros, so the per-block add is
/// skipped entirely (softmax over `scores + 0` == softmax over `scores`, so the step is byte-identical).
struct PreparedCond {
    b: usize,
    l: usize,
    indicator: Vec<i64>,
    segment_ids: Vec<i64>,
    position_ids: Vec<f32>,
    llm_mask: Tensor,
    img_mask: Tensor,
    img_idx: Tensor,
    cos: Tensor,
    sin: Tensor,
    seg_mask: Option<Tensor>,
}

impl Ideogram4Transformer {
    /// Load a DiT from a component dir of `.safetensors` (top-level keys: `input_proj.*`,
    /// `layers.{i}.*`, `final_layer.*`, …). `w`'s dtype is the DiT compute dtype (bf16).
    pub fn load(w: &Weights, cfg: &Ideogram4DitConfig) -> Result<Self> {
        let head_dim = cfg.emb_dim / cfg.num_heads;
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            layers.push(Ideogram4Block::load(
                w,
                &format!("layers.{i}"),
                cfg.num_heads,
                head_dim,
                cfg.norm_eps,
            )?);
        }
        // Sinusoidal freqs: half = emb_dim/2, lf = ln(1e4)/(half-1), f[d] = exp(-lf·d).
        let half = cfg.emb_dim / 2;
        let lf = (1e4f32).ln() / (half as f32 - 1.0);
        let t_freqs: Vec<f32> = (0..half).map(|d| (-lf * d as f32).exp()).collect();
        let t_freqs = Tensor::from_vec(t_freqs, (1, half), w.device())?;

        let embed_image_indicator = embedding_detect(w, "embed_image_indicator")?;

        Ok(Self {
            input_proj: linear_detect(w, "input_proj", true)?,
            llm_cond_norm: w.get("llm_cond_norm.weight")?,
            llm_cond_proj: linear_detect(w, "llm_cond_proj", true)?,
            t_mlp_in: linear_detect(w, "t_embedding.mlp_in", true)?,
            t_mlp_out: linear_detect(w, "t_embedding.mlp_out", true)?,
            adaln_proj: linear_detect(w, "adaln_proj", true)?,
            embed_image_indicator,
            rotary_emb: Ideogram4MRoPE::new(
                head_dim,
                cfg.rope_theta,
                cfg.mrope_section,
                w.device(),
            )?,
            layers,
            final_adaln: linear_detect(w, "final_layer.adaln_modulation", true)?,
            final_linear: linear_detect(w, "final_layer.linear", true)?,
            t_freqs,
            dtype: w.dtype(),
            cond_cache: std::sync::Mutex::new(None),
        })
    }

    /// The DiT's compute device (every weight loaded onto it) — the device a resolved additive residual
    /// factor is moved to before it is pushed onto a projection (sc-11104).
    pub fn device(&self) -> candle_gen::candle_core::Device {
        self.t_freqs.device().clone()
    }

    /// Walk **every** adaptable projection in the DiT with its canonical diffusers-style dotted path
    /// (sc-11104): the top-level `input_proj` / `llm_cond_proj` / `t_embedding.mlp_{in,out}` /
    /// `adaln_proj`, each block's `layers.{i}.*`, and the final layer's `final_layer.{adaln_modulation,
    /// linear}`. These are the keys a prefix-stripped TurboTime-LoRA module resolves against, so the
    /// additive installer ([`crate::adapters::install_turbo_lora_additive`]) can push a residual onto any
    /// matched projection while leaving the base packed. Ordered by walk, top-level first.
    pub fn visit_adaptable_mut(
        &mut self,
        f: &mut dyn FnMut(&str, &mut QLinear) -> Result<()>,
    ) -> Result<()> {
        f("input_proj", &mut self.input_proj)?;
        f("llm_cond_proj", &mut self.llm_cond_proj)?;
        f("t_embedding.mlp_in", &mut self.t_mlp_in)?;
        f("t_embedding.mlp_out", &mut self.t_mlp_out)?;
        f("adaln_proj", &mut self.adaln_proj)?;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.visit_adaptable_mut(&format!("layers.{i}"), f)?;
        }
        f("final_layer.adaln_modulation", &mut self.final_adaln)?;
        f("final_layer.linear", &mut self.final_linear)?;
        Ok(())
    }

    /// Sinusoidal scalar-`t` embedding → MLP. `t`: `[B]` in `[0,1]` → `[B, emb_dim]`.
    fn t_embedding(&self, t: &Tensor) -> Result<Tensor> {
        let scaled = (t.to_dtype(DType::F32)? * 1e4)?; // [B]
        let emb = scaled.unsqueeze(1)?.broadcast_mul(&self.t_freqs)?; // [B, half]
        let emb = Tensor::cat(&[emb.sin()?, emb.cos()?], D::Minus1)?.to_dtype(self.dtype)?;
        let h = self.t_mlp_in.forward(&emb)?.silu()?;
        self.t_mlp_out.forward(&h)
    }

    /// Velocity prediction `[B, L, in_channels]` (f32). Inputs follow the upstream packing:
    /// `llm_features [B,L,llm_dim]`, `x [B,L,in_ch]`, `t [B]`, `position_ids [B,L,3]`,
    /// `segment_ids [B,L]`, `indicator [B,L]`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        llm_features: &Tensor,
        x: &Tensor,
        t: &Tensor,
        position_ids: &Tensor,
        segment_ids: &Tensor,
        indicator: &Tensor,
    ) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;
        // The role masks, MRoPE tables, and segment mask are step-invariant (fixed packing geometry),
        // so build them once per render and reuse across every step / CFG pass (sc-8992). `seg_mask`
        // is `None` for the always-uniform-segment path this pipeline drives — the mask is all-zeros,
        // so the per-block add is skipped (byte-identical after softmax).
        let (llm_mask, img_mask, img_idx, cos, sin, seg_mask) =
            self.prepared_cond(indicator, segment_ids, position_ids, b, l)?;

        let llm_features = llm_features
            .to_dtype(self.dtype)?
            .broadcast_mul(&llm_mask)?;
        let x = x.to_dtype(self.dtype)?.broadcast_mul(&img_mask)?;
        let x = self.input_proj.forward(&x)?.broadcast_mul(&img_mask)?;

        let t_cond = self.t_embedding(t)?.unsqueeze(1)?; // [B,1,emb]
        let adaln_input = self.adaln_proj.forward(&t_cond)?.silu()?; // [B,1,adaln]

        let llm = rmsnorm(&llm_features, &self.llm_cond_norm, COND_NORM_EPS)?;
        let llm = self.llm_cond_proj.forward(&llm)?.broadcast_mul(&llm_mask)?;

        let mut h = (&x + &llm)?;
        h = (h + self.embed_image_indicator.forward(&img_idx)?)?;

        for layer in &self.layers {
            h = layer.forward(&h, &cos, &sin, seg_mask.as_ref(), &adaln_input)?;
        }

        // Final layer: scale = 1 + adaln(silu(c)); linear(layernorm_no_affine(h) · scale).
        let scale = (self.final_adaln.forward(&adaln_input.silu()?)? + 1.0)?;
        let normed = layer_norm_no_affine(&h, FINAL_NORM_EPS)?;
        let out = self.final_linear.forward(&normed.broadcast_mul(&scale)?)?;
        out.to_dtype(DType::F32)
    }

    /// Build (or reuse) the step-invariant conditioning tensors for this render (sc-8992): the role
    /// masks (`indicator`), the MRoPE `(cos, sin)` (`position_ids`), and the segment attention mask
    /// (`segment_ids`; `None` when all segment ids are equal → the mask is all-zeros and the per-block
    /// add is skipped). Recomputed only when the loop-invariant inputs change; otherwise the Arc-backed
    /// handles are cloned. The construction is identical to computing it inline, so every step is
    /// byte-identical.
    #[allow(clippy::type_complexity)]
    fn prepared_cond(
        &self,
        indicator: &Tensor,
        segment_ids: &Tensor,
        position_ids: &Tensor,
        b: usize,
        l: usize,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, Option<Tensor>)> {
        let ind: Vec<i64> = indicator
            .to_dtype(DType::I64)?
            .flatten_all()?
            .to_vec1::<i64>()?;
        let seg: Vec<i64> = segment_ids
            .to_dtype(DType::I64)?
            .flatten_all()?
            .to_vec1::<i64>()?;
        let pos: Vec<f32> = position_ids
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        let mut guard = candle_gen::lock_recover(&self.cond_cache);
        if let Some(c) = guard.as_ref() {
            if c.b == b
                && c.l == l
                && c.indicator == ind
                && c.segment_ids == seg
                && c.position_ids == pos
            {
                return Ok((
                    c.llm_mask.clone(),
                    c.img_mask.clone(),
                    c.img_idx.clone(),
                    c.cos.clone(),
                    c.sin.clone(),
                    c.seg_mask.clone(),
                ));
            }
        }

        let (llm_mask, img_mask, img_idx) =
            role_tensors(&ind, b, l, self.dtype, indicator.device())?;
        let (cos, sin) = self.rotary_emb.forward(position_ids)?;
        let seg_mask = segment_mask(&seg, b, l, indicator.device())?;

        *guard = Some(PreparedCond {
            b,
            l,
            indicator: ind,
            segment_ids: seg,
            position_ids: pos,
            llm_mask: llm_mask.clone(),
            img_mask: img_mask.clone(),
            img_idx: img_idx.clone(),
            cos: cos.clone(),
            sin: sin.clone(),
            seg_mask: seg_mask.clone(),
        });
        Ok((llm_mask, img_mask, img_idx, cos, sin, seg_mask))
    }
}

/// No-affine LayerNorm over the last dim (computed in f32 for stability, cast back to `x`'s dtype).
fn layer_norm_no_affine(x: &Tensor, eps: f64) -> Result<Tensor> {
    let dt = x.dtype();
    let x = x.to_dtype(DType::F32)?;
    let mean = x.mean_keepdim(D::Minus1)?;
    let xc = x.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
    let normed = xc.broadcast_div(&(var + eps)?.sqrt()?)?;
    normed.to_dtype(dt)
}

/// From `indicator` (host `[B·L]`, row-major): `(llm_mask [B,L,1], img_mask [B,L,1]` at `dtype`,
/// `img_idx [B,L]` u32). `img_idx` = 1 at image tokens, 0 elsewhere (the `embed_image_indicator`
/// lookup index). Takes the already-host-read `indicator` slice so the device→host round-trip happens
/// once per render, not once per step (sc-8992).
fn role_tensors(
    ind: &[i64],
    b: usize,
    l: usize,
    dtype: DType,
    dev: &candle_gen::candle_core::Device,
) -> Result<(Tensor, Tensor, Tensor)> {
    let n = b * l;
    let mut llm = vec![0f32; n];
    let mut img = vec![0f32; n];
    let mut idx = vec![0u32; n];
    for (p, &v) in ind.iter().enumerate().take(n) {
        if v == LLM_TOKEN_INDICATOR {
            llm[p] = 1.0;
        }
        if v == OUTPUT_IMAGE_INDICATOR {
            img[p] = 1.0;
            idx[p] = 1;
        }
    }
    Ok((
        Tensor::from_vec(llm, (b, l, 1), dev)?.to_dtype(dtype)?,
        Tensor::from_vec(img, (b, l, 1), dev)?.to_dtype(dtype)?,
        Tensor::from_vec(idx, (b, l), dev)?,
    ))
}

/// Additive attention mask `[B, 1, L, L]` (f32): `0` where two tokens share a `segment_id`, `-inf`
/// otherwise (full bidirectional attention within a packed sample — not causal). Takes the
/// already-host-read `seg` slice (sc-8992).
///
/// Returns `None` when **every** token shares one segment id — the mask would be all-zeros, so the
/// caller skips the per-block additive step entirely (`softmax(scores + 0) == softmax(scores)`, so the
/// step is byte-identical). This pipeline always packs a single uniform segment, so `None` is the hot
/// path and the ~`B·L²`-element allocation + per-block broadcast-add are avoided.
fn segment_mask(
    seg: &[i64],
    b: usize,
    l: usize,
    dev: &candle_gen::candle_core::Device,
) -> Result<Option<Tensor>> {
    let uniform = seg.iter().all(|&s| Some(s) == seg.first().copied());
    if uniform {
        return Ok(None);
    }
    let mut data = vec![0f32; b * l * l];
    for bi in 0..b {
        for i in 0..l {
            for j in 0..l {
                if seg[bi * l + i] != seg[bi * l + j] {
                    data[(bi * l + i) * l + j] = f32::NEG_INFINITY;
                }
            }
        }
    }
    Ok(Some(Tensor::from_vec(data, (b, 1, l, l), dev)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Device;

    /// A uniform segment (every token shares one id) yields `None` — the caller skips the per-block
    /// additive mask, which is the always-taken path in this pipeline (sc-8992).
    #[test]
    fn segment_mask_uniform_is_none() {
        let dev = Device::Cpu;
        assert!(segment_mask(&[7, 7, 7, 7], 1, 4, &dev).unwrap().is_none());
        // A single-token sequence is trivially uniform.
        assert!(segment_mask(&[3], 1, 1, &dev).unwrap().is_none());
    }

    /// A non-uniform segment builds the additive `[B,1,L,L]` mask: `0` within a segment, `-inf` across.
    #[test]
    fn segment_mask_non_uniform_places_neg_inf_across_segments() {
        let dev = Device::Cpu;
        // Tokens 0,1 in segment 0; tokens 2,3 in segment 1.
        let m = segment_mask(&[0, 0, 1, 1], 1, 4, &dev).unwrap().unwrap();
        assert_eq!(m.dims(), &[1, 1, 4, 4]);
        let v = m.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let at = |i: usize, j: usize| v[i * 4 + j];
        // Same-segment pairs are 0; cross-segment pairs are -inf.
        assert_eq!(at(0, 1), 0.0);
        assert_eq!(at(2, 3), 0.0);
        assert!(at(0, 2).is_infinite() && at(0, 2) < 0.0);
        assert!(at(3, 1).is_infinite() && at(3, 1) < 0.0);
    }
}
