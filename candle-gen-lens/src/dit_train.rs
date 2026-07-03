//! Vendored, training-adapted Lens DiT (sc-5147) — the candle twin of `mlx-gen-lens`'s trainable
//! transformer and the Lens analog of [`candle-gen-wan`'s `dit_train`](../../candle-gen-wan/src/dit_train.rs).
//!
//! A faithful copy of [`crate::transformer`]'s `LensTransformer` with the four attention projections
//! (`img_qkv` / `txt_qkv` / `to_out.0` / `to_add_out`) held as [`LoraLinear`] so the native LoRA/LoKr
//! trainer can splice a trainable residual into each — the stock
//! [`JointAttention`](crate::transformer) builds them from frozen [`QLinear`](crate::quant) with no
//! seam. Only the structs that *own* a projection (or the block / model that owns them) are vendored;
//! the frozen, **already-composable** pieces ([`TimeEmbed`], [`NormOut`], [`modulate`], [`gated`],
//! [`build_joint_mask`]) are **reused** from [`crate::transformer`] so the two stay in lockstep.
//!
//! **Three deviations, all forced by candle autograd** (the epic-5164 fused-ops trap, see
//! [[candle-fused-ops-no-backward]] / `candle-gen-wan::dit_train`). The stock DiT uses three fused
//! `CustomOp`s with **no backward** that silently zero every upstream adapter gradient:
//! `softmax_last_dim` → the composable [`softmax`](candle_nn::ops::softmax) (f32); `RmsNorm::forward`
//! (the QK norms + the four block norms) → [`RmsNorm::forward_diff`] (the numerically-identical
//! composable LayerNorm path); and `apply_rope` (candle's fused `rope_i`) → [`apply_rope_diff`] (the
//! same interleaved rotation in plain ops — the Wan lesson, its `rope_i` on the q/k path zeroed every
//! attention factor's grad).
//! The frozen front-end (`img_in` / `txt_in` / the per-layer text norms / `time_embed`) and the head
//! (`norm_out` / `proj_out`) sit **upstream/downstream** of every adapter, so they stay frozen `Linear`
//! and the text norms reuse `forward_diff` only so the gradient flows *through* them to earlier blocks.
//!
//! With no adapter installed the vendored forward is bit-identical to the stock forward (the
//! `parity_tests` gate pins this). `forward` returns the **raw** patch-space velocity (no sign flip) —
//! Lens feeds the transformer output to the flow-match step *without* negation (opposite of Z-Image),
//! so the trainer regresses the raw velocity toward `noise − x0` (see [`crate::training`]).

use candle_gen::candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_gen::candle_nn::ops::softmax;
use candle_gen::candle_nn::{linear, linear_no_bias, rms_norm, Linear, RmsNorm, VarBuilder};
use candle_gen::train::gradient_checkpoint::Segment;
use candle_gen::train::lora::{lora_linear, LoraHost, LoraLinear};

use crate::rope::LensRope;
use crate::transformer::{
    build_joint_mask, gated, modulate, LensDitConfig, NormOut, TimeEmbed, EPS, TXT_NORM_EPS,
};

/// The Lens LoRA/LoKr target suffixes — the fused dual-stream attention projections, matching the
/// torch `lens_train_runner` `DEFAULT_LORA_TARGET_MODULES` (sc-2218) and the inference merge surface
/// ([`crate::adapters`]). `img_qkv` / `txt_qkv` are the **fused** `[3·inner, inner]` projections (a
/// LoRA on them adapts the whole fused weight — no q/k/v split); `to_out.0` is the first element of
/// diffusers' `to_out` `ModuleList`, so its path segment literally contains the `.0`.
pub const LENS_ATTN_TARGETS: [&str; 4] = ["img_qkv", "txt_qkv", "to_out.0", "to_add_out"];

/// Composable interleaved RoPE — the differentiable twin of [`crate::rope::apply_rope`], which wraps
/// candle's fused `rope_i` (a `CustomOp` with NO backward). Applies the **same** interleaved rotation
/// `out[2k] = x[2k]·cos_k − x[2k+1]·sin_k`, `out[2k+1] = x[2k]·sin_k + x[2k+1]·cos_k` over the half
/// tables `cos`/`sin` `[S, head_dim/2]`, so with the same tables it equals `rope_i` (the `parity_tests`
/// gate pins vendored == stock through this path). `x`: `[B, H, S, head_dim]`.
fn apply_rope_diff(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let dtype = x.dtype();
    let (b, h, s, d) = x.dims4()?;
    let half = d / 2;
    let xf = x.to_dtype(DType::F32)?.reshape((b, h, s, half, 2))?;
    let x0 = xf.narrow(4, 0, 1)?.squeeze(4)?; // even lanes [B,H,S,half]
    let x1 = xf.narrow(4, 1, 1)?.squeeze(4)?; // odd lanes
    let cos = cos.reshape((1, 1, s, half))?;
    let sin = sin.reshape((1, 1, s, half))?;
    let o0 = (x0.broadcast_mul(&cos)? - x1.broadcast_mul(&sin)?)?;
    let o1 = (x0.broadcast_mul(&sin)? + x1.broadcast_mul(&cos)?)?;
    // Re-interleave: stack the two lanes on a new trailing axis → [B,H,S,half,2] → [B,H,S,d].
    Tensor::stack(&[&o0, &o1], 4)?
        .reshape((b, h, s, d))?
        .to_dtype(dtype)
}

// ==================== TrainJointAttention (LoRA seam) ====================

/// Lens joint (dual-stream) attention with the four projections held as [`LoraLinear`]. Numerically
/// identical to the stock [`JointAttention`](crate::transformer) with no adapter installed, except it
/// runs the composable softmax / QK `forward_diff` / [`apply_rope_diff`] (the stock fused ops have no
/// backward). The projections carry a **bias** (Lens fused QKV is biased), so they wrap [`lora_linear`].
struct TrainJointAttention {
    img_qkv: LoraLinear,
    txt_qkv: LoraLinear,
    to_out: LoraLinear,
    to_add_out: LoraLinear,
    norm_q: RmsNorm,
    norm_k: RmsNorm,
    norm_added_q: RmsNorm,
    norm_added_k: RmsNorm,
    heads: usize,
    head_dim: usize,
}

impl TrainJointAttention {
    fn new(cfg: &LensDitConfig, vb: VarBuilder) -> Result<Self> {
        let inner = cfg.inner_dim;
        let hd = cfg.head_dim;
        Ok(Self {
            img_qkv: lora_linear(inner, 3 * inner, vb.pp("img_qkv"))?,
            txt_qkv: lora_linear(inner, 3 * inner, vb.pp("txt_qkv"))?,
            to_out: lora_linear(inner, inner, vb.pp("to_out").pp("0"))?,
            to_add_out: lora_linear(inner, inner, vb.pp("to_add_out"))?,
            norm_q: rms_norm(hd, EPS, vb.pp("norm_q"))?,
            norm_k: rms_norm(hd, EPS, vb.pp("norm_k"))?,
            norm_added_q: rms_norm(hd, EPS, vb.pp("norm_added_q"))?,
            norm_added_k: rms_norm(hd, EPS, vb.pp("norm_added_k"))?,
            heads: cfg.num_heads,
            head_dim: hd,
        })
    }

    fn visit_lora_mut(
        &mut self,
        f: &mut dyn FnMut(&mut LoraLinear) -> candle_gen::Result<()>,
    ) -> candle_gen::Result<()> {
        f(&mut self.img_qkv)?;
        f(&mut self.txt_qkv)?;
        f(&mut self.to_out)?;
        f(&mut self.to_add_out)?;
        Ok(())
    }

    /// Fused QKV → `(q, k, v)` each `[B, seq, heads, head_dim]`. Mirrors
    /// [`JointAttention::qkv`](crate::transformer).
    fn qkv(&self, lin: &LoraLinear, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (b, s, _) = x.dims3()?;
        let (h, hd) = (self.heads, self.head_dim);
        let t = lin.forward(x)?.reshape((b, s, 3, h, hd))?;
        let q = t.narrow(2, 0, 1)?.squeeze(2)?.contiguous()?;
        let k = t.narrow(2, 1, 1)?.squeeze(2)?.contiguous()?;
        let v = t.narrow(2, 2, 1)?.squeeze(2)?.contiguous()?;
        Ok((q, k, v))
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        img: &Tensor,
        txt: &Tensor,
        img_cos: &Tensor,
        img_sin: &Tensor,
        txt_cos: &Tensor,
        txt_sin: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let (b, img_seq, _) = img.dims3()?;
        let txt_seq = txt.dim(1)?;
        let (h, hd) = (self.heads, self.head_dim);

        let (iq, ik, iv) = self.qkv(&self.img_qkv, img)?;
        let (tq, tk, tv) = self.qkv(&self.txt_qkv, txt)?;

        // QK RMSNorm over head_dim (composable `forward_diff` — the fused `forward` has no backward).
        let iq = self.norm_q.forward_diff(&iq)?;
        let ik = self.norm_k.forward_diff(&ik)?;
        let tq = self.norm_added_q.forward_diff(&tq)?;
        let tk = self.norm_added_k.forward_diff(&tk)?;

        // To heads-first `[B, heads, seq, head_dim]`, then interleaved RoPE on q/k.
        let bhsd = |x: &Tensor| -> Result<Tensor> { x.transpose(1, 2)?.contiguous() };
        let iq = apply_rope_diff(&bhsd(&iq)?, img_cos, img_sin)?;
        let ik = apply_rope_diff(&bhsd(&ik)?, img_cos, img_sin)?;
        let iv = bhsd(&iv)?;
        let tq = apply_rope_diff(&bhsd(&tq)?, txt_cos, txt_sin)?;
        let tk = apply_rope_diff(&bhsd(&tk)?, txt_cos, txt_sin)?;
        let tv = bhsd(&tv)?;

        // Joint `[img, txt]` (image first) over the sequence axis.
        let q = Tensor::cat(&[&iq, &tq], 2)?;
        let k = Tensor::cat(&[&ik, &tk], 2)?;
        let v = Tensor::cat(&[&iv, &tv], 2)?;
        let scale = (hd as f64).powf(-0.5);
        // Composable SDPA in f32 (NOT the fused `softmax_last_dim` — that CustomOp has no backward).
        // i32-overflow guard (sc-9116): the joint `[img, txt]` scores `[B, heads, joint, joint]` reach
        // `i32::MAX` at large edit/joint sequences (the F-003 class; training runs small today, but the
        // math is identical to the inference twin), so the shared budgeted helper chunks over the query
        // rows (byte-identical for common sizes). The softmax closure preserves the exact f32-upcast
        // composable `softmax(_, D::Minus1)` so the backward is unchanged.
        let qd = q.dtype();
        let o = candle_gen::sdpa_budgeted_bhsd(
            &q,
            &k,
            &v,
            scale,
            mask,
            |s| softmax(&s.to_dtype(DType::F32)?, D::Minus1)?.to_dtype(qd),
            candle_gen::ATTN_SCORES_BUDGET,
        )?; // [B, heads, joint, head_dim]
        let joint = img_seq + txt_seq;
        let o = o.transpose(1, 2)?.reshape((b, joint, h * hd))?;

        let img_o = o.narrow(1, 0, img_seq)?.contiguous()?;
        let txt_o = o.narrow(1, img_seq, txt_seq)?.contiguous()?;
        Ok((
            self.to_out.forward(&img_o)?,
            self.to_add_out.forward(&txt_o)?,
        ))
    }
}

// ==================== TrainGateMlp (frozen) ====================

/// SwiGLU `GateMLP` (`w2(silu(w1·x) · w3·x)`, bias-less) — the frozen twin of
/// [`GateMlp`](crate::transformer) built from plain `Linear` (not the inference `QLinear`; the MLPs are
/// never adapter targets, and a dense `Linear` equals `QLinear::Dense` exactly).
struct TrainGateMlp {
    w1: Linear,
    w2: Linear,
    w3: Linear,
}

impl TrainGateMlp {
    fn new(inner: usize, hidden: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            w1: linear_no_bias(inner, hidden, vb.pp("w1"))?,
            w2: linear_no_bias(hidden, inner, vb.pp("w2"))?,
            w3: linear_no_bias(inner, hidden, vb.pp("w3"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.w1.forward(x)?.silu()?;
        let up = self.w3.forward(x)?;
        self.w2.forward(&gate.mul(&up)?)
    }
}

// ==================== TrainBlock ====================

/// One Lens dual-stream MMDiT block: per-stream AdaLN modulation around the joint attention (`mod1`)
/// and the SwiGLU MLP (`mod2`), with gated residuals. Byte-faithful to
/// [`LensTransformerBlock`](crate::transformer) with the trainable attention spliced in and the four
/// block norms run via `forward_diff`. Returns `(encoder, hidden)` — the reference block's order.
struct TrainBlock {
    img_mod: Linear,
    txt_mod: Linear,
    img_norm1: RmsNorm,
    img_norm2: RmsNorm,
    txt_norm1: RmsNorm,
    txt_norm2: RmsNorm,
    attn: TrainJointAttention,
    img_mlp: TrainGateMlp,
    txt_mlp: TrainGateMlp,
}

impl TrainBlock {
    fn new(cfg: &LensDitConfig, vb: VarBuilder) -> Result<Self> {
        let inner = cfg.inner_dim;
        let hidden = cfg.mlp_hidden();
        Ok(Self {
            img_mod: linear(inner, 6 * inner, vb.pp("img_mod").pp("1"))?,
            txt_mod: linear(inner, 6 * inner, vb.pp("txt_mod").pp("1"))?,
            img_norm1: rms_norm(inner, EPS, vb.pp("img_norm1"))?,
            img_norm2: rms_norm(inner, EPS, vb.pp("img_norm2"))?,
            txt_norm1: rms_norm(inner, EPS, vb.pp("txt_norm1"))?,
            txt_norm2: rms_norm(inner, EPS, vb.pp("txt_norm2"))?,
            attn: TrainJointAttention::new(cfg, vb.pp("attn"))?,
            img_mlp: TrainGateMlp::new(inner, hidden, vb.pp("img_mlp"))?,
            txt_mlp: TrainGateMlp::new(inner, hidden, vb.pp("txt_mlp"))?,
        })
    }

    fn visit_lora_mut(
        &mut self,
        f: &mut dyn FnMut(&mut LoraLinear) -> candle_gen::Result<()>,
    ) -> candle_gen::Result<()> {
        self.attn.visit_lora_mut(f)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        hidden: &Tensor,  // image [B, img_seq, inner]
        encoder: &Tensor, // text  [B, txt_seq, inner]
        temb: &Tensor,    // [B, inner]
        img_cos: &Tensor,
        img_sin: &Tensor,
        txt_cos: &Tensor,
        txt_sin: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let act = temb.silu()?;
        let img_mod = self.img_mod.forward(&act)?;
        let txt_mod = self.txt_mod.forward(&act)?;
        let n = img_mod.dim(D::Minus1)? / 2;
        let (im0, im1) = (
            img_mod.narrow(D::Minus1, 0, n)?,
            img_mod.narrow(D::Minus1, n, n)?,
        );
        let (tm0, tm1) = (
            txt_mod.narrow(D::Minus1, 0, n)?,
            txt_mod.narrow(D::Minus1, n, n)?,
        );

        // attention path
        let (img_n, img_g1) = modulate(&self.img_norm1.forward_diff(hidden)?, &im0)?;
        let (txt_n, txt_g1) = modulate(&self.txt_norm1.forward_diff(encoder)?, &tm0)?;
        let (img_attn, txt_attn) = self
            .attn
            .forward(&img_n, &txt_n, img_cos, img_sin, txt_cos, txt_sin, mask)?;
        let hidden = gated(hidden, &img_g1, &img_attn)?;
        let encoder = gated(encoder, &txt_g1, &txt_attn)?;

        // feed-forward path (SwiGLU)
        let (img_n2, img_g2) = modulate(&self.img_norm2.forward_diff(&hidden)?, &im1)?;
        let hidden = gated(&hidden, &img_g2, &self.img_mlp.forward(&img_n2)?)?;
        let (txt_n2, txt_g2) = modulate(&self.txt_norm2.forward_diff(&encoder)?, &tm1)?;
        let encoder = gated(&encoder, &txt_g2, &self.txt_mlp.forward(&txt_n2)?)?;

        Ok((encoder, hidden))
    }
}

// ==================== LensTransformerTrain ====================

/// The constant side tensors the per-block stack + head consume, produced once by
/// [`LensTransformerTrain::forward_pre_main`]. None depend on an adapter `Var` (the front-end is
/// frozen), so the gradient-checkpointing path captures them as detached constants shared across every
/// recomputed block.
pub struct PreCtx {
    /// `[B, inner]` timestep embedding (drives every block's AdaLN + the head).
    pub temb: Tensor,
    img_cos: Tensor,
    img_sin: Tensor,
    txt_cos: Tensor,
    txt_sin: Tensor,
    mask: Option<Tensor>,
}

/// The vendored, trainable twin of [`LensTransformer`](crate::transformer). Built from the *same*
/// `transformer/` safetensors keys, so it loads the real DiT weights unchanged and, with no adapter
/// installed, reproduces the stock forward bit-for-bit (`parity_tests`).
pub struct LensTransformerTrain {
    img_in: Linear,
    txt_norm: Vec<RmsNorm>, // per-layer text front-end RMSNorm (eps 1e-5), run via forward_diff
    txt_in: Linear,
    time_embed: TimeEmbed,
    blocks: Vec<TrainBlock>,
    norm_out: NormOut,
    proj_out: Linear,
    rope: LensRope,
    cfg: LensDitConfig,
    device: Device,
    dtype: DType,
}

impl LensTransformerTrain {
    /// Load from a diffusers `transformer/` weight set at `dtype` (bf16 production / f32 gate). Same key
    /// layout as [`LensTransformer::new`](crate::transformer::LensTransformer::new).
    pub fn new(cfg: &LensDitConfig, vb: VarBuilder) -> Result<Self> {
        let inner = cfg.inner_dim;
        let mut txt_norm = Vec::with_capacity(cfg.num_text_layers);
        for i in 0..cfg.num_text_layers {
            txt_norm.push(rms_norm(
                cfg.enc_hidden_dim,
                TXT_NORM_EPS,
                vb.pp("txt_norm").pp(i),
            )?);
        }
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            blocks.push(TrainBlock::new(cfg, vb.pp("transformer_blocks").pp(i))?);
        }
        Ok(Self {
            img_in: linear(cfg.in_channels, inner, vb.pp("img_in"))?,
            txt_norm,
            txt_in: linear(cfg.txt_in_dim(), inner, vb.pp("txt_in"))?,
            time_embed: TimeEmbed::new(cfg, vb.pp("time_text_embed"))?,
            blocks,
            norm_out: NormOut::new(cfg, vb.pp("norm_out"))?,
            proj_out: linear(
                inner,
                cfg.patch_size * cfg.patch_size * cfg.out_channels,
                vb.pp("proj_out"),
            )?,
            rope: LensRope::new(cfg.rope_theta, cfg.axes_dims_rope),
            cfg: *cfg,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    pub fn config(&self) -> &LensDitConfig {
        &self.cfg
    }

    /// The **pre-main** forward: embed the image latents + the multi-layer text front-end into the two
    /// token streams `(hidden, encoder)` and build the timestep + RoPE constants ([`PreCtx`]). Every
    /// piece here is frozen and upstream of all adapters, so the gradient-checkpointing path runs it
    /// **detached** — there are no upstream adapter grads to stitch (the
    /// [`checkpointed_backward`](candle_gen::train::gradient_checkpoint::checkpointed_backward) input
    /// cotangent is discarded). Mirrors the front-end of
    /// [`LensTransformer::forward`](crate::transformer::LensTransformer::forward).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_pre_main(
        &self,
        hidden_states: &Tensor,
        text_feats: &[Tensor],
        text_valid: Option<&Tensor>,
        timestep: f32,
        frame: usize,
        h: usize,
        w: usize,
    ) -> Result<(Tensor, Tensor, PreCtx)> {
        assert_eq!(
            text_feats.len(),
            self.cfg.num_text_layers,
            "expected {} text-feature layers, got {}",
            self.cfg.num_text_layers,
            text_feats.len()
        );
        let img_len = hidden_states.dim(1)?;
        let txt_len = text_feats[0].dim(1)?;

        let hidden = self.img_in.forward(hidden_states)?;

        let mut normed = Vec::with_capacity(self.cfg.num_text_layers);
        for (i, feat) in text_feats.iter().enumerate() {
            normed.push(self.txt_norm[i].forward_diff(feat)?);
        }
        let normed_refs: Vec<&Tensor> = normed.iter().collect();
        let encoder = self
            .txt_in
            .forward(&Tensor::cat(&normed_refs, D::Minus1)?)?;

        let temb = self
            .time_embed
            .forward(timestep, &self.device, self.dtype)?;
        let (img_cos, img_sin) = self.rope.img_cos_sin(frame, h, w, &self.device)?;
        let (txt_cos, txt_sin) = self.rope.txt_cos_sin(txt_len, h, w, &self.device)?;
        let mask = match text_valid {
            Some(valid) => Some(build_joint_mask(valid, img_len, self.dtype, &self.device)?),
            None => None,
        };

        Ok((
            hidden,
            encoder,
            PreCtx {
                temb,
                img_cos,
                img_sin,
                txt_cos,
                txt_sin,
                mask,
            },
        ))
    }

    /// One [`Segment`] per transformer block, each mapping the dual-stream state `[hidden, encoder] →
    /// [hidden, encoder]` using `ctx`'s timestep + RoPE constants. Fed to
    /// [`checkpointed_backward`](candle_gen::train::gradient_checkpoint::checkpointed_backward) so each
    /// of the 48 blocks is recomputed in the backward — bounding the peak to **one** block's transient
    /// weight gradients (candle's matmul backward materializes a grad for the frozen base weight too, so
    /// a dense 48-block backward would hold ~48 layers of weight-grads at once; the Wan lesson). The
    /// trainer appends a final `[hidden, encoder] → [loss]` segment ([`velocity_out`](Self::velocity_out)
    /// + the flow-match regression).
    pub fn main_block_segments<'a>(&'a self, ctx: &'a PreCtx) -> Vec<Segment<'a>> {
        self.blocks
            .iter()
            .map(|blk| -> Segment<'a> {
                Box::new(move |st: &[Tensor]| {
                    // state = [hidden, encoder]; block returns (encoder, hidden).
                    let (e, h) = blk.forward(
                        &st[0],
                        &st[1],
                        &ctx.temb,
                        &ctx.img_cos,
                        &ctx.img_sin,
                        &ctx.txt_cos,
                        &ctx.txt_sin,
                        ctx.mask.as_ref(),
                    )?;
                    Ok(vec![h, e])
                })
            })
            .collect()
    }

    /// The **post-main** head: AdaLN-modulated `norm_out` (by `temb`) → `proj_out` → the **raw**
    /// patch-space velocity `[B, img_len, patch²·out_channels]` (= 128). `hidden` is the last block's
    /// image-stream output. No sign flip and no unpatchify (the 2×2 (un)patchify lives in the VAE shim).
    pub fn velocity_out(&self, hidden: &Tensor, ctx: &PreCtx) -> Result<Tensor> {
        let hidden = self.norm_out.forward(hidden, &ctx.temb)?;
        self.proj_out.forward(&hidden)
    }

    /// Dense forward (the parity path): `pre_main` → the per-block stack → `velocity_out`. Mirrors
    /// [`LensTransformer::forward`](crate::transformer::LensTransformer::forward) and returns the same
    /// `[B, img_len, 128]` raw velocity. The `parity_tests` gate pins this == stock with no adapter.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        text_feats: &[Tensor],
        text_valid: Option<&Tensor>,
        timestep: f32,
        frame: usize,
        h: usize,
        w: usize,
    ) -> Result<Tensor> {
        let (mut hidden, mut encoder, ctx) =
            self.forward_pre_main(hidden_states, text_feats, text_valid, timestep, frame, h, w)?;
        for blk in &self.blocks {
            let (e, hs) = blk.forward(
                &hidden,
                &encoder,
                &ctx.temb,
                &ctx.img_cos,
                &ctx.img_sin,
                &ctx.txt_cos,
                &ctx.txt_sin,
                ctx.mask.as_ref(),
            )?;
            encoder = e;
            hidden = hs;
        }
        self.velocity_out(&hidden, &ctx)
    }
}

impl LoraHost for LensTransformerTrain {
    fn visit_lora_mut(
        &mut self,
        f: &mut dyn FnMut(&mut LoraLinear) -> candle_gen::Result<()>,
    ) -> candle_gen::Result<()> {
        for blk in self.blocks.iter_mut() {
            blk.visit_lora_mut(f)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod parity_tests {
    //! Pin the vendored trainable DiT to the stock [`LensTransformer`](crate::transformer): built from
    //! the *same* `VarMap`-backed weights with no adapter installed, the two must produce a bit-identical
    //! forward — the regression guard that the `LoraLinear` swap + composable softmax/rms/rope changed
    //! nothing.
    use super::*;
    use crate::transformer::LensTransformer;
    use candle_gen::candle_core::{Device, Tensor};
    use candle_gen::candle_nn::{VarBuilder, VarMap};

    /// A tiny Lens-shaped config (2 layers, 2 heads × 8, 1 text layer) — exercises every vendored path
    /// cheaply on CPU while keeping the real arithmetic invariants: `patch² · out = 4·8 = 32 = in`,
    /// `inner = heads · head_dim`, and `Σ axes_dims_rope = head_dim` (each axis even).
    fn tiny_cfg() -> LensDitConfig {
        LensDitConfig {
            patch_size: 2,
            in_channels: 32,
            out_channels: 8,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            inner_dim: 16,
            enc_hidden_dim: 12,
            num_text_layers: 1,
            timestep_channels: 16,
            axes_dims_rope: [2, 2, 4],
            rope_theta: 10_000.0,
        }
    }

    #[test]
    fn vendored_dit_matches_stock_forward() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        // Vendored built first; the stock model reads the SAME varmap params, so any output difference
        // is a forward-logic difference, not a weight one.
        let vendored = LensTransformerTrain::new(&cfg, vb.clone()).unwrap();
        let stock = LensTransformer::new(&cfg, vb).unwrap();
        // Randomize every shared var so the forward runs on nontrivial weights (a fresh VarMap is
        // zero-init, which would make the comparison vacuous).
        for v in vm.all_vars() {
            v.set(&Tensor::randn(0f32, 0.1f32, v.dims(), &dev).unwrap())
                .unwrap();
        }

        // image latents [1, frame·h·w, in_channels]; frame=1, grid 2×2 → 4 image tokens; 3 text tokens.
        let (frame, h, w) = (1usize, 2usize, 2usize);
        let img_len = frame * h * w;
        let hidden = Tensor::randn(0f32, 1f32, (1, img_len, cfg.in_channels), &dev).unwrap();
        let feat = Tensor::randn(0f32, 1f32, (1, 3, cfg.enc_hidden_dim), &dev).unwrap();
        let feats = vec![feat];

        let y_v = vendored
            .forward(&hidden, &feats, None, 0.3, frame, h, w)
            .unwrap();
        let y_s = stock
            .forward(&hidden, &feats, None, 0.3, frame, h, w)
            .unwrap();

        assert_eq!(y_v.dims(), y_s.dims());
        let diff = (y_v - y_s)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            diff < 1e-5,
            "vendored Lens DiT diverged from stock by {diff}"
        );
    }

    /// The [`LoraHost`] walk reaches exactly `4 × num_layers` projections — the four fused/output
    /// attention `LoraLinear`s in every block.
    #[test]
    fn lora_host_visits_every_attention_projection() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let mut model = LensTransformerTrain::new(&cfg, vb).unwrap();
        let mut paths: Vec<String> = Vec::new();
        model
            .visit_lora_mut(&mut |lin| {
                paths.push(lin.path().to_string());
                Ok(())
            })
            .unwrap();
        assert_eq!(paths.len(), 4 * cfg.num_layers);
        for suffix in LENS_ATTN_TARGETS {
            assert!(
                paths
                    .iter()
                    .any(|p| p == suffix || p.ends_with(&format!(".{suffix}"))),
                "no visited projection matched suffix {suffix}"
            );
        }
        assert!(paths.contains(&"transformer_blocks.0.attn.img_qkv".to_string()));
        assert!(paths.contains(&"transformer_blocks.0.attn.to_out.0".to_string()));
        assert!(paths.contains(&"transformer_blocks.0.attn.to_add_out".to_string()));
    }

    /// The crux of the vendoring: with LoRA installed, a `loss.backward()` must deliver a non-zero
    /// gradient to **every** adapter factor — proving the fused→composable swaps (softmax / QK
    /// `forward_diff` / [`apply_rope_diff`]) actually let grads flow back to the `img_qkv` / `txt_qkv`
    /// / `to_out.0` / `to_add_out` adapters in every block (the stock fused ops would silently zero
    /// them). The loss is over **both** final streams (image + text); the real trainer regresses only
    /// the image-stream velocity, under which the *last* block's `to_add_out` legitimately gets no grad
    /// (the final text stream is architecturally discarded — true in the torch reference too), so a
    /// both-streams loss is what exercises every projection's composable backward. Factors are forced
    /// nonzero so the residual is live everywhere.
    #[test]
    fn backward_reaches_every_adapter_factor() {
        use candle_gen::train::lora::build_lora_targets;

        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let mut model = LensTransformerTrain::new(&cfg, vb).unwrap();
        for v in vm.all_vars() {
            v.set(&Tensor::randn(0f32, 0.1f32, v.dims(), &dev).unwrap())
                .unwrap();
        }
        let targets: Vec<String> = LENS_ATTN_TARGETS.iter().map(|s| s.to_string()).collect();
        let set = build_lora_targets(&mut model, &targets, 4, 4.0, 0, &dev).unwrap();
        assert_eq!(set.vars.len(), 2 * 4 * cfg.num_layers); // (A,B) × 4 projections × layers
                                                            // Force every factor (incl. the zero-init B) nonzero so the residual is live everywhere.
        for v in &set.vars {
            v.set(&Tensor::randn(0f32, 0.02f32, v.dims(), &dev).unwrap())
                .unwrap();
        }

        let (frame, h, w) = (1usize, 2usize, 2usize);
        let img_len = frame * h * w;
        let hs = Tensor::randn(0f32, 1f32, (1, img_len, cfg.in_channels), &dev).unwrap();
        let feat = Tensor::randn(0f32, 1f32, (1, 3, cfg.enc_hidden_dim), &dev).unwrap();

        // Run the block stack via the public segments API and form a loss over both final streams.
        let (hidden, encoder, ctx) = model
            .forward_pre_main(&hs, &[feat], None, 0.3, frame, h, w)
            .unwrap();
        let mut state = vec![hidden, encoder];
        for seg in model.main_block_segments(&ctx) {
            state = seg(&state).unwrap();
        }
        let loss = (state[0].sqr().unwrap().sum_all().unwrap()
            + state[1].sqr().unwrap().sum_all().unwrap())
        .unwrap();
        let grads = loss.backward().unwrap();

        for (i, var) in set.vars.iter().enumerate() {
            let g = grads
                .get(var.as_tensor())
                .unwrap_or_else(|| panic!("adapter factor {i} received NO gradient"));
            let mag = g
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            assert!(mag > 0.0, "adapter factor {i} got an all-zero gradient");
        }
    }
}
