//! The Qwen-Image **dual-stream MMDiT** (60 blocks). Port of `mlx-gen-qwen-image`'s `transformer/`,
//! run in candle bf16 (the native checkpoint dtype; ~41 GB).
//!
//! Shape anchors: `inner_dim = 3072` (24 heads × 128), `in_channels = 64`, `out_channels = 16`,
//! `joint_attention_dim = 3584`. Conditioning is **timestep-only** (no text pooling). Each block runs
//! both an image and a text stream with per-stream AdaLN modulation (`img_mod`/`txt_mod` → 2 sets of
//! shift/scale/gate), a JOINT attention over the **`[txt, img]`** sequence (text first) with
//! interleaved 3-axis RoPE (see [`crate::rope`]), and a GELU-tanh FFN per stream.
//!
//! Parity-load-bearing: all LayerNorms are affine-free, eps 1e-6; q/k RMSNorm is per-head (128-dim),
//! eps 1e-6; the top-level `txt_norm` is a standard RMSNorm applied before `txt_in`; `norm_out.linear`
//! is loaded **bias-less** (the checkpoint bias is ignored); the timestep proj scales by ×1000 inside
//! the sinusoid; all the other Linears are **biased**.

use candle_gen::candle_core::{DType, Device, Result, Tensor, D};
use candle_gen::candle_nn::{ops::softmax_last_dim, rms_norm, Module, RmsNorm, VarBuilder};

use crate::config::TransformerConfig;
use crate::quant::QLinear;
use crate::rope::{apply_rope, QwenRope, RopeCache};

const EPS: f64 = 1e-6;

/// Affine-free LayerNorm over the last axis (dtype-preserving; computed in f32).
fn layer_norm(x: &Tensor) -> Result<Tensor> {
    let dt = x.dtype();
    let xf = x.to_dtype(DType::F32)?;
    let mean = xf.mean_keepdim(D::Minus1)?;
    let xc = xf.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
    xc.broadcast_div(&(var + EPS)?.sqrt()?)?.to_dtype(dt)
}

/// Split a `[B, 3·inner]` modulation chunk into `(shift, scale, gate)`, each `[B, 1, inner]`.
fn chunk3(m: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
    let inner = m.dim(D::Minus1)? / 3;
    let shift = m.narrow(D::Minus1, 0, inner)?.unsqueeze(1)?;
    let scale = m.narrow(D::Minus1, inner, inner)?.unsqueeze(1)?;
    let gate = m.narrow(D::Minus1, 2 * inner, inner)?.unsqueeze(1)?;
    Ok((shift, scale, gate))
}

/// AdaLN-zero modulate: returns `(x·(1+scale) + shift, gate)`.
fn modulate(x: &Tensor, m: &Tensor) -> Result<(Tensor, Tensor)> {
    let (shift, scale, gate) = chunk3(m)?;
    let out = x.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(&shift)?;
    Ok((out, gate))
}

/// `modulate` with optional per-token timestep selection (`zero_cond_t`, Qwen-Image-Edit-2511). With
/// `index = None` this is exactly [`modulate`]. With `index = Some` the `m` chunk carries a doubled
/// batch `[real_t ; zero_t]` (`[2, 3·inner]`); each image token picks the real-`t` half where
/// `index == 0` (noise) and the `t 0` half where `index == 1` (conditioning) — the diffusers
/// `_modulate(index)`. Blended via `real + (zero − real)·index` (bit-equivalent for a 0/1 index).
fn modulate_sel(x: &Tensor, m: &Tensor, index: Option<&Tensor>) -> Result<(Tensor, Tensor)> {
    let Some(index) = index else {
        return modulate(x, m);
    };
    let inner = m.dim(D::Minus1)? / 3;
    let blend = index.unsqueeze(2)?.to_dtype(m.dtype())?; // [1, seq, 1]
    let pick = |slot: usize| -> Result<Tensor> {
        let real = m
            .narrow(0, 0, 1)?
            .narrow(D::Minus1, slot * inner, inner)?
            .unsqueeze(1)?; // [1,1,inner]
        let zero = m
            .narrow(0, 1, 1)?
            .narrow(D::Minus1, slot * inner, inner)?
            .unsqueeze(1)?; // [1,1,inner]
        real.broadcast_add(&zero.broadcast_sub(&real)?.broadcast_mul(&blend)?) // [1, seq, inner]
    };
    let shift = pick(0)?;
    let scale = pick(1)?;
    let gate = pick(2)?;
    let out = x.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(&shift)?;
    Ok((out, gate))
}

/// `x + gate·y`.
fn gated(x: &Tensor, gate: &Tensor, y: &Tensor) -> Result<Tensor> {
    x + y.broadcast_mul(gate)?
}

/// Sinusoidal timestep embedding `[1, dim]` from the raw sigma — the ×1000 scale is applied inside
/// the argument (diffusers `timestep · 1000`); `[cos | sin]`, base 10000.
fn timestep_embedding(sigma: f32, dim: usize, device: &Device) -> Result<Tensor> {
    let half = dim / 2;
    let ln = 10000f32.ln();
    let mut cos = vec![0f32; half];
    let mut sin = vec![0f32; half];
    for k in 0..half {
        let freq = (-ln * k as f32 / half as f32).exp();
        let arg = sigma * 1000.0 * freq;
        cos[k] = arg.cos();
        sin[k] = arg.sin();
    }
    let cos = Tensor::from_vec(cos, (1, half), device)?;
    let sin = Tensor::from_vec(sin, (1, half), device)?;
    Tensor::cat(&[&cos, &sin], D::Minus1)
}

/// SDPA over `[B,H,S,D]` q/k/v → `[B, S, H·D]`. scale = `head_dim^-0.5`. Delegates to the shared
/// i32-overflow-safe [`candle_gen::sdpa_budgeted_bhsd`] (sc-9570), which chunks over the query rows once
/// the `[B,H,Sq,Sk]` scores tensor would exceed [`candle_gen::ATTN_SCORES_BUDGET`] (the candle CUDA
/// i32-index limit). The Qwen MMDiT runs ONE joint attention over the `[txt, noise(, ref)]` sequence
/// (24 heads); the dual-latent edit path grows fastest and at ≳1280² trips the guard. The
/// `softmax_last_dim` closure keeps the exact fused softmax; each query row's softmax is independent, so
/// the chunked result is byte-identical to the single pass. This crate does the head-merge here.
fn attention(q: &Tensor, k: &Tensor, v: &Tensor, head_dim: usize) -> Result<Tensor> {
    let (b, _h, s, _d) = q.dims4()?;
    let scale = (head_dim as f64).powf(-0.5);
    let o = candle_gen::sdpa_budgeted_bhsd(
        q,
        k,
        v,
        scale,
        None,
        softmax_last_dim,
        candle_gen::ATTN_SCORES_BUDGET,
    )?; // [B,H,S,D]
    let (_b, h, _s, d) = o.dims4()?;
    o.transpose(1, 2)?.reshape((b, s, h * d))
}

/// Reshape `[B,S,inner]` → `[B,H,S,head_dim]`, applying per-head RMSNorm (over head_dim) for q/k.
fn to_heads(x: &Tensor, heads: usize, head_dim: usize, norm: Option<&RmsNorm>) -> Result<Tensor> {
    let (b, s, _) = x.dims3()?;
    let x = x.reshape((b, s, heads, head_dim))?;
    let x = match norm {
        Some(n) => n.forward(&x)?,
        None => x,
    };
    x.transpose(1, 2)?.contiguous()
}

struct TimeEmbed {
    linear_1: QLinear,
    linear_2: QLinear,
    channels: usize,
}

impl TimeEmbed {
    fn new(cfg: &TransformerConfig, vb: VarBuilder, gs: usize) -> Result<Self> {
        let inner = cfg.inner_dim();
        let te = vb.pp("timestep_embedder");
        Ok(Self {
            linear_1: QLinear::linear_detect_gs(
                cfg.timestep_channels,
                inner,
                &te,
                "linear_1",
                true,
                gs,
            )?,
            linear_2: QLinear::linear_detect_gs(inner, inner, &te, "linear_2", true, gs)?,
            channels: cfg.timestep_channels,
        })
    }

    fn forward(&self, sigma: f32, device: &Device, dtype: DType) -> Result<Tensor> {
        let emb = timestep_embedding(sigma, self.channels, device)?.to_dtype(dtype)?;
        let h = self.linear_1.forward(&emb)?.silu()?;
        self.linear_2.forward(&h)
    }

    /// Visit the two timestep-embedder projections (`{prefix}.timestep_embedder.linear_{1,2}`) — not
    /// adapted by the Lightning distill, but part of the general adaptable surface (sc-11091).
    fn visit_adaptable_mut(
        &mut self,
        prefix: &str,
        f: &mut dyn FnMut(&str, &mut QLinear) -> candle_gen::Result<()>,
    ) -> candle_gen::Result<()> {
        f(
            &format!("{prefix}.timestep_embedder.linear_1"),
            &mut self.linear_1,
        )?;
        f(
            &format!("{prefix}.timestep_embedder.linear_2"),
            &mut self.linear_2,
        )?;
        Ok(())
    }
}

/// GELU-tanh feed-forward (`net.0.proj → gelu → net.2`).
struct FeedForward {
    proj_in: QLinear,
    proj_out: QLinear,
}

impl FeedForward {
    fn new(inner: usize, hidden: usize, vb: VarBuilder, gs: usize) -> Result<Self> {
        Ok(Self {
            // `net.0.proj` and `net.2` nest under the ff base — pass the full dotted base so the
            // `.scales`/`.biases` siblings survive the key remap (never `.pp()` past a scales sibling).
            proj_in: QLinear::linear_detect_gs(inner, hidden, &vb, "net.0.proj", true, gs)?,
            proj_out: QLinear::linear_detect_gs(hidden, inner, &vb, "net.2", true, gs)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.proj_out.forward(&self.proj_in.forward(x)?.gelu()?)
    }

    /// Visit the two feed-forward projections (`{prefix}.net.0.proj`, `{prefix}.net.2`) — the diffusers
    /// keys the Lightning distill's stream-MLP LoRA targets (sc-11091).
    fn visit_adaptable_mut(
        &mut self,
        prefix: &str,
        f: &mut dyn FnMut(&str, &mut QLinear) -> candle_gen::Result<()>,
    ) -> candle_gen::Result<()> {
        f(&format!("{prefix}.net.0.proj"), &mut self.proj_in)?;
        f(&format!("{prefix}.net.2"), &mut self.proj_out)?;
        Ok(())
    }
}

struct JointAttention {
    to_q: QLinear,
    to_k: QLinear,
    to_v: QLinear,
    to_out: QLinear,
    add_q: QLinear,
    add_k: QLinear,
    add_v: QLinear,
    to_add_out: QLinear,
    norm_q: RmsNorm,
    norm_k: RmsNorm,
    norm_added_q: RmsNorm,
    norm_added_k: RmsNorm,
    heads: usize,
    head_dim: usize,
}

impl JointAttention {
    fn new(cfg: &TransformerConfig, vb: VarBuilder, gs: usize) -> Result<Self> {
        let inner = cfg.inner_dim();
        let hd = cfg.head_dim;
        let lin = |base: &str| QLinear::linear_detect_gs(inner, inner, &vb, base, true, gs);
        Ok(Self {
            to_q: lin("to_q")?,
            to_k: lin("to_k")?,
            to_v: lin("to_v")?,
            // `to_out` nests at `to_out.0` — the full base keeps the `.scales`/`.biases` siblings.
            to_out: lin("to_out.0")?,
            add_q: lin("add_q_proj")?,
            add_k: lin("add_k_proj")?,
            add_v: lin("add_v_proj")?,
            to_add_out: lin("to_add_out")?,
            norm_q: rms_norm(hd, EPS, vb.pp("norm_q"))?,
            norm_k: rms_norm(hd, EPS, vb.pp("norm_k"))?,
            norm_added_q: rms_norm(hd, EPS, vb.pp("norm_added_q"))?,
            norm_added_k: rms_norm(hd, EPS, vb.pp("norm_added_k"))?,
            heads: cfg.num_heads,
            head_dim: hd,
        })
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
    ) -> Result<(Tensor, Tensor)> {
        let (h, hd) = (self.heads, self.head_dim);
        let txt_seq = txt.dim(1)?;

        let iq = apply_rope(
            &to_heads(&self.to_q.forward(img)?, h, hd, Some(&self.norm_q))?,
            img_cos,
            img_sin,
        )?;
        let ik = apply_rope(
            &to_heads(&self.to_k.forward(img)?, h, hd, Some(&self.norm_k))?,
            img_cos,
            img_sin,
        )?;
        let iv = to_heads(&self.to_v.forward(img)?, h, hd, None)?;
        let tq = apply_rope(
            &to_heads(&self.add_q.forward(txt)?, h, hd, Some(&self.norm_added_q))?,
            txt_cos,
            txt_sin,
        )?;
        let tk = apply_rope(
            &to_heads(&self.add_k.forward(txt)?, h, hd, Some(&self.norm_added_k))?,
            txt_cos,
            txt_sin,
        )?;
        let tv = to_heads(&self.add_v.forward(txt)?, h, hd, None)?;

        // Joint over the sequence, text first.
        let q = Tensor::cat(&[&tq, &iq], 2)?;
        let k = Tensor::cat(&[&tk, &ik], 2)?;
        let v = Tensor::cat(&[&tv, &iv], 2)?;
        // Chunk the joint attention over query rows when the [B,H,Sq,Sk] scores tensor would exceed the
        // candle CUDA i32-index limit (long edit/joint sequences >~1024²); numerically identical to a
        // single pass, and a no-op single pass for the txt2img / control sizes (sc-6217).
        let o = attention(&q, &k, &v, hd)?; // [B, seq, h·hd]
        let seq = o.dim(1)?;
        let txt_o = o.narrow(1, 0, txt_seq)?.contiguous()?;
        let img_o = o.narrow(1, txt_seq, seq - txt_seq)?.contiguous()?;
        Ok((
            self.to_out.forward(&img_o)?,
            self.to_add_out.forward(&txt_o)?,
        ))
    }

    /// Visit every adaptable joint-attention projection (`{prefix}.{to_q,to_k,to_v,to_out.0,
    /// add_q_proj,add_k_proj,add_v_proj,to_add_out}`) — the diffusers key each maps to, so a resolved
    /// LoRA factor routes 1:1 (sc-11091). The RMSNorms are not `Linear` and are never adapted.
    fn visit_adaptable_mut(
        &mut self,
        prefix: &str,
        f: &mut dyn FnMut(&str, &mut QLinear) -> candle_gen::Result<()>,
    ) -> candle_gen::Result<()> {
        f(&format!("{prefix}.to_q"), &mut self.to_q)?;
        f(&format!("{prefix}.to_k"), &mut self.to_k)?;
        f(&format!("{prefix}.to_v"), &mut self.to_v)?;
        f(&format!("{prefix}.to_out.0"), &mut self.to_out)?;
        f(&format!("{prefix}.add_q_proj"), &mut self.add_q)?;
        f(&format!("{prefix}.add_k_proj"), &mut self.add_k)?;
        f(&format!("{prefix}.add_v_proj"), &mut self.add_v)?;
        f(&format!("{prefix}.to_add_out"), &mut self.to_add_out)?;
        Ok(())
    }
}

struct Block {
    img_mod: QLinear,
    txt_mod: QLinear,
    attn: JointAttention,
    img_ff: FeedForward,
    txt_ff: FeedForward,
}

impl Block {
    fn new(cfg: &TransformerConfig, vb: VarBuilder, gs: usize) -> Result<Self> {
        let inner = cfg.inner_dim();
        let ff_hidden = inner * 4;
        Ok(Self {
            // `img_mod`/`txt_mod` nest the projection at `.1` — pass the full dotted base.
            img_mod: QLinear::linear_detect_gs(inner, 6 * inner, &vb, "img_mod.1", true, gs)?,
            txt_mod: QLinear::linear_detect_gs(inner, 6 * inner, &vb, "txt_mod.1", true, gs)?,
            attn: JointAttention::new(cfg, vb.pp("attn"), gs)?,
            img_ff: FeedForward::new(inner, ff_hidden, vb.pp("img_mlp"), gs)?,
            txt_ff: FeedForward::new(inner, ff_hidden, vb.pp("txt_mlp"), gs)?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        hidden: &Tensor,
        encoder: &Tensor,
        temb: &Tensor,
        img_cos: &Tensor,
        img_sin: &Tensor,
        txt_cos: &Tensor,
        txt_sin: &Tensor,
        // `Some` only on the Qwen-Image-Edit-2511 `zero_cond_t` path: then `temb` is the doubled
        // `[real_t ; zero_t]` and the image stream selects modulation per token (0 = noise, 1 = cond).
        modulate_index: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let act = temb.silu()?; // [1, inner] (or [2, inner] under zero_cond_t)
        let img_mod = self.img_mod.forward(&act)?; // [1 or 2, 6·inner]
                                                   // The text stream always uses the real-timestep modulation (row 0 under zero_cond_t).
        let txt_act = match modulate_index {
            Some(_) => act.narrow(0, 0, 1)?,
            None => act.clone(),
        };
        let txt_mod = self.txt_mod.forward(&txt_act)?; // [1, 6·inner]
        let half = img_mod.dim(D::Minus1)? / 2;
        let (im0, im1) = (
            img_mod.narrow(D::Minus1, 0, half)?,
            img_mod.narrow(D::Minus1, half, half)?,
        );
        let (tm0, tm1) = (
            txt_mod.narrow(D::Minus1, 0, half)?,
            txt_mod.narrow(D::Minus1, half, half)?,
        );

        // attention path
        let (img_n, img_g1) = modulate_sel(&layer_norm(hidden)?, &im0, modulate_index)?;
        let (txt_n, txt_g1) = modulate(&layer_norm(encoder)?, &tm0)?;
        let (img_attn, txt_attn) = self
            .attn
            .forward(&img_n, &txt_n, img_cos, img_sin, txt_cos, txt_sin)?;
        let hidden = gated(hidden, &img_g1, &img_attn)?;
        let encoder = gated(encoder, &txt_g1, &txt_attn)?;

        // feed-forward path
        let (img_n2, img_g2) = modulate_sel(&layer_norm(&hidden)?, &im1, modulate_index)?;
        let hidden = gated(&hidden, &img_g2, &self.img_ff.forward(&img_n2)?)?;
        let (txt_n2, txt_g2) = modulate(&layer_norm(&encoder)?, &tm1)?;
        let encoder = gated(&encoder, &txt_g2, &self.txt_ff.forward(&txt_n2)?)?;

        Ok((encoder, hidden))
    }

    /// Visit every adaptable projection in this block under `{prefix}` (`transformer_blocks.{i}`): the
    /// two AdaLN modulation projections (`img_mod.1` / `txt_mod.1`), the joint attention, and the two
    /// stream MLPs — the full per-block adaptable surface (sc-11091). The Lightning distill adapts the
    /// attention + MLP set (12/block); user LoRAs may also target the mod projections.
    fn visit_adaptable_mut(
        &mut self,
        prefix: &str,
        f: &mut dyn FnMut(&str, &mut QLinear) -> candle_gen::Result<()>,
    ) -> candle_gen::Result<()> {
        f(&format!("{prefix}.img_mod.1"), &mut self.img_mod)?;
        f(&format!("{prefix}.txt_mod.1"), &mut self.txt_mod)?;
        self.attn
            .visit_adaptable_mut(&format!("{prefix}.attn"), f)?;
        self.img_ff
            .visit_adaptable_mut(&format!("{prefix}.img_mlp"), f)?;
        self.txt_ff
            .visit_adaptable_mut(&format!("{prefix}.txt_mlp"), f)?;
        Ok(())
    }
}

/// AdaLayerNorm-Continuous output head: `silu(temb) → linear (bias-less) → (scale, shift)`, then
/// `(1+scale)·LN(x) + shift`.
struct NormOut {
    linear: QLinear,
}

impl NormOut {
    fn new(cfg: &TransformerConfig, vb: VarBuilder, gs: usize) -> Result<Self> {
        let inner = cfg.inner_dim();
        // The checkpoint ships a bias, but the fork loads this bias-less (packed-detect bias-less too).
        Ok(Self {
            linear: QLinear::linear_detect_gs(inner, 2 * inner, &vb, "linear", false, gs)?,
        })
    }

    fn forward(&self, x: &Tensor, temb: &Tensor) -> Result<Tensor> {
        let p = self.linear.forward(&temb.silu()?)?;
        let inner = p.dim(D::Minus1)? / 2;
        let scale = p.narrow(D::Minus1, 0, inner)?.unsqueeze(1)?;
        let shift = p.narrow(D::Minus1, inner, inner)?.unsqueeze(1)?;
        layer_norm(x)?
            .broadcast_mul(&(scale + 1.0)?)?
            .broadcast_add(&shift)
    }

    /// Visit the output-head projection (`{prefix}.linear`) (sc-11091).
    fn visit_adaptable_mut(
        &mut self,
        prefix: &str,
        f: &mut dyn FnMut(&str, &mut QLinear) -> candle_gen::Result<()>,
    ) -> candle_gen::Result<()> {
        f(&format!("{prefix}.linear"), &mut self.linear)
    }
}

/// The Qwen-Image MMDiT.
pub struct QwenTransformer {
    img_in: QLinear,
    txt_norm: RmsNorm,
    txt_in: QLinear,
    time_embed: TimeEmbed,
    blocks: Vec<Block>,
    norm_out: NormOut,
    proj_out: QLinear,
    rope: QwenRope,
    rope_cache: RopeCache,
    device: Device,
    dtype: DType,
}

impl QwenTransformer {
    /// Build the MMDiT from a `transformer/` VarBuilder at the default MLX group size 64. A **dense**
    /// diffusers snapshot (no `.scales`) loads unchanged; a pre-quantized MLX tier at group 64 loads
    /// packed. Callers that read the packed `group_size` from `transformer/config.json` (a non-64 tier)
    /// use [`Self::new_gs`].
    pub fn new(cfg: &TransformerConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_gs(cfg, vb, candle_gen::quant::MLX_GROUP_SIZE)
    }

    /// Build the MMDiT at an explicit MLX `group_size` (sc-9415), read from the packed
    /// `transformer/config.json`'s `quantization.group_size` (`SceneWorks/qwen-image-mlx` +
    /// `qwen-image-edit-2511-mlx` ship group 64). Every DiT `Linear` packed-**detects** its `.scales`
    /// sibling and loads straight from the packed parts when present, else the dense path unchanged —
    /// so a dense diffusers snapshot and a q4/q8 packed snapshot both load through this one call. The
    /// only quantized weights in the tier are the DiT projections; the RMSNorm weights stay dense.
    pub fn new_gs(cfg: &TransformerConfig, vb: VarBuilder, gs: usize) -> Result<Self> {
        let inner = cfg.inner_dim();
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            blocks.push(Block::new(cfg, vb.pp("transformer_blocks").pp(i), gs)?);
        }
        Ok(Self {
            img_in: QLinear::linear_detect_gs(cfg.in_channels, inner, &vb, "img_in", true, gs)?,
            txt_norm: rms_norm(cfg.joint_attention_dim, cfg.eps, vb.pp("txt_norm"))?,
            txt_in: QLinear::linear_detect_gs(
                cfg.joint_attention_dim,
                inner,
                &vb,
                "txt_in",
                true,
                gs,
            )?,
            time_embed: TimeEmbed::new(cfg, vb.pp("time_text_embed"), gs)?,
            blocks,
            norm_out: NormOut::new(cfg, vb.pp("norm_out"), gs)?,
            // proj_out maps to the packed velocity (patch²·out_channels = 64 = in_channels).
            proj_out: QLinear::linear_detect_gs(inner, cfg.in_channels, &vb, "proj_out", true, gs)?,
            rope: QwenRope::new(cfg),
            rope_cache: RopeCache::new(),
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Whether the DiT loaded from an MLX-packed tier (probed on `proj_out`, which packs together with
    /// every other DiT projection in a q4/q8 tier). Gates the edit lane's additive-vs-fold adapter route
    /// (sc-11091): a packed base attaches LoRA residuals unmerged; a dense base folds `W += δ`.
    pub fn is_packed(&self) -> bool {
        self.proj_out.is_packed()
    }

    /// The device the DiT weights live on — the forward-time residual factors are read on the CPU and
    /// moved here at install (else the residual matmul is a device mismatch).
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Walk every adaptable projection, invoking `f(path, &mut QLinear)` once each with the projection's
    /// canonical diffusers dotted path (`img_in`, `txt_in`, `time_text_embed.timestep_embedder.linear_*`,
    /// per-block `transformer_blocks.{i}.*`, `norm_out.linear`, `proj_out`). The additive installer
    /// ([`crate::adapters::install_additive`]) pushes a resolved LoRA/LoKr residual onto each matched
    /// projection (sc-11091).
    pub fn visit_adaptable_mut(
        &mut self,
        f: &mut dyn FnMut(&str, &mut QLinear) -> candle_gen::Result<()>,
    ) -> candle_gen::Result<()> {
        f("img_in", &mut self.img_in)?;
        f("txt_in", &mut self.txt_in)?;
        self.time_embed.visit_adaptable_mut("time_text_embed", f)?;
        for (i, blk) in self.blocks.iter_mut().enumerate() {
            blk.visit_adaptable_mut(&format!("transformer_blocks.{i}"), f)?;
        }
        self.norm_out.visit_adaptable_mut("norm_out", f)?;
        f("proj_out", &mut self.proj_out)?;
        Ok(())
    }

    /// Predict velocity. `hidden_states` `[1, img_seq, 64]`, `encoder_hidden_states`
    /// `[1, txt_seq, 3584]`, `timestep` = raw sigma, `(lat_h, lat_w)` = the packed token grid.
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        timestep: f32,
        lat_h: usize,
        lat_w: usize,
    ) -> Result<Tensor> {
        // The plain path is `forward_control` with no residuals — byte-identical (the match below is
        // inert when `residuals = None`), so the txt2img parity path has a single source of truth.
        self.forward_control(
            hidden_states,
            encoder_hidden_states,
            timestep,
            lat_h,
            lat_w,
            None,
            0.0,
        )
    }

    /// [`forward`] with optional ControlNet residual injection (sc-5489): after base block `i` the
    /// residual `residuals[i / interval]` (pre-scaled by `control_scale`) is added to the image stream,
    /// where `interval = ceil(num_blocks / num_residuals)` (60 base blocks, 5 control residuals →
    /// interval 12) — the diffusers `QwenImageTransformer2DModel` `index_block // interval_control`
    /// pattern. `residuals = None` (or empty) is byte-identical to the plain forward.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_control(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        timestep: f32,
        lat_h: usize,
        lat_w: usize,
        residuals: Option<&[Tensor]>,
        control_scale: f32,
    ) -> Result<Tensor> {
        let temb = self
            .time_embed
            .forward(timestep, &self.device, self.dtype)?;
        let mut hidden = self.img_in.forward(hidden_states)?;
        let encoder = self.txt_norm.forward(encoder_hidden_states)?;
        let mut encoder = self.txt_in.forward(&encoder)?;

        let txt_seq = encoder.dim(1)?;
        // Step-invariant (fixed grid), so cache the RoPE tables per render (sc-8992).
        let (img_cos, img_sin, txt_cos, txt_sin) =
            self.rope_cache
                .tables(&self.rope, &[(lat_h, lat_w)], txt_seq, &self.device)?;

        // Treat an empty slice as "no control" so the group index can't underflow. Pre-scale the (few)
        // control residuals once, before the 60-block loop.
        let residuals = residuals.filter(|r| !r.is_empty());
        let interval = residuals.map(|r| self.blocks.len().div_ceil(r.len().max(1)));
        let scaled: Option<Vec<Tensor>> = match residuals {
            Some(res) => Some(
                res.iter()
                    .map(|r| r * control_scale as f64)
                    .collect::<Result<Vec<_>>>()?,
            ),
            None => None,
        };

        for (i, block) in self.blocks.iter().enumerate() {
            let (e, h) = block.forward(
                &hidden, &encoder, &temb, &img_cos, &img_sin, &txt_cos, &txt_sin, None,
            )?;
            encoder = e;
            // After each base block, add the pre-scaled control residual for this block's group:
            // diffusers `hidden_states = hidden_states + controlnet_block_samples[i // interval]`.
            hidden = match (&scaled, interval) {
                (Some(res), Some(interval)) => {
                    let idx = (i / interval).min(res.len() - 1);
                    (h + &res[idx])?
                }
                _ => h,
            };
        }

        let hidden = self.norm_out.forward(&hidden, &temb)?;
        self.proj_out.forward(&hidden)
    }

    /// Qwen-Image-**Edit** dual-latent forward (sc-5487). `hidden_states` `[1, noise_seq + ref_seq, 64]`
    /// is the noise latents concatenated with the packed reference latents (the caller concatenates and
    /// slices back the noise prefix from the returned velocity); `cond_grids` lists each reference's
    /// `(latent_h, latent_w)` so the 3-axis RoPE spans `[noise] + references` (the grid index drives the
    /// frame axis). `zero_cond_t` (Edit-2511): double the timestep to `[t, 0]` and modulate the
    /// conditioning tokens as clean (t = 0) via the per-token `modulate_index`; `false` (the original
    /// Edit / 2509) runs a single timestep over the whole sequence. Returns the velocity over the
    /// **full** sequence `[1, noise_seq + ref_seq, 64]`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_edit(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        timestep: f32,
        lat_h: usize,
        lat_w: usize,
        cond_grids: &[(usize, usize)],
        zero_cond_t: bool,
    ) -> Result<Tensor> {
        let img_seq = hidden_states.dim(1)?;
        let mut hidden = self.img_in.forward(hidden_states)?;
        let encoder = self.txt_norm.forward(encoder_hidden_states)?;
        let mut encoder = self.txt_in.forward(&encoder)?;
        let txt_seq = encoder.dim(1)?;

        // 3-axis RoPE over the noise grid then each reference grid.
        let mut grids = Vec::with_capacity(1 + cond_grids.len());
        grids.push((lat_h, lat_w));
        grids.extend_from_slice(cond_grids);
        // Step-invariant (fixed noise + reference grids), so cache the RoPE tables per render (sc-8992).
        let (img_cos, img_sin, txt_cos, txt_sin) =
            self.rope_cache
                .tables(&self.rope, &grids, txt_seq, &self.device)?;

        // zero_cond_t: double the temb to [real_t ; zero_t] and build the per-token select index.
        let zc = zero_cond_t && !cond_grids.is_empty();
        let temb_real = self
            .time_embed
            .forward(timestep, &self.device, self.dtype)?;
        let (temb, modulate_index) = if zc {
            let temb_zero = self.time_embed.forward(0.0, &self.device, self.dtype)?;
            let temb2 = Tensor::cat(&[&temb_real, &temb_zero], 0)?;
            let idx = build_modulate_index(lat_h * lat_w, cond_grids, img_seq, &self.device)?;
            (temb2, Some(idx))
        } else {
            (temb_real.clone(), None)
        };

        for block in &self.blocks {
            let (e, h) = block.forward(
                &hidden,
                &encoder,
                &temb,
                &img_cos,
                &img_sin,
                &txt_cos,
                &txt_sin,
                modulate_index.as_ref(),
            )?;
            encoder = e;
            hidden = h;
        }

        // norm_out uses only the real-timestep embedding (the fork's temb[:B]).
        let hidden = self.norm_out.forward(&hidden, &temb_real)?;
        self.proj_out.forward(&hidden)
    }
}

/// The per-token timestep selector for `zero_cond_t` (Qwen-Image-Edit-2511): `0` for the noise latent
/// tokens (`latent_h·latent_w`), `1` for every conditioning-image token (`Σ h·w` over the reference
/// grids). Shape `[1, img_seq]` f32 — diffusers `[[0]*prod(shapes[0]) + [1]*Σ prod(shapes[1:])]`.
fn build_modulate_index(
    noise_len: usize,
    cond_grids: &[(usize, usize)],
    img_seq: usize,
    device: &Device,
) -> Result<Tensor> {
    let cond_len: usize = cond_grids.iter().map(|(h, w)| h * w).sum();
    debug_assert_eq!(
        noise_len + cond_len,
        img_seq,
        "modulate index spans the full image sequence"
    );
    let mut row = vec![0f32; noise_len];
    row.extend(std::iter::repeat_n(1f32, cond_len));
    Tensor::from_vec(row, (1, img_seq), device)
}

/// The Qwen-Image **2512-Fun-Controlnet-Union** VACE control branch (sc-8350) — the candle port of the
/// alibaba-pai `Qwen-Image-2512-Fun-Controlnet-Union` `QwenImageControlTransformer2DModel` (the
/// `VideoX-Fun` VACE family, the same shape as the FLUX.2 / Z-Image Fun-Controlnet-Union branches).
/// This is the sole Qwen control branch; it **replaced** the retired InstantX `QwenImageControlNetModel`
/// shape on the Qwen control path (that dead lane was removed in sc-9868 — MLX twin retired in sc-8267,
/// worker repointed InstantX→2512-Fun in sc-8350).
///
/// Unlike that retired InstantX ControlNet (an independent mini-transformer with a zero-init
/// `controlnet_x_embedder` ADDed onto `img_in(x)`, emitting per-block residuals the base ADDs at a
/// fixed interval), the 2512-Fun branch is **VACE-style**: a `control_img_in` patch embedder
/// (`132 → inner`) feeds a control state `c` threaded through `N` control blocks that reuse the base
/// [`Block`] math (and the base modulation / RoPE / timestep), seeded at block 0 by
/// `c = before_proj(c) + img_embed`. Each control block emits a hint via a zero-init `after_proj`; the
/// base transformer adds `hints[n]·control_scale` into its image stream **after** the base block at
/// `control_layers[n]` (`[0, 12, 24, 36, 48]` — 5 hints across the 60-layer MMDiT). `control_scale = 0`
/// is byte-identical to the base forward (`+0`).
///
/// The control blocks are the *same* [`Block`] as the base (identical on-disk keys), so the loader
/// reuses [`Block::new`].
pub struct QwenFunControlBranch {
    /// `control_img_in`: 132 → inner. Biased patch embedder for the packed 132-ch control context.
    control_img_in: QLinear,
    /// The `N` control blocks (same math as the base dual-stream block; reuse the base RoPE / temb).
    blocks: Vec<Block>,
    /// Zero-init per-block hint projection (`inner → inner`), one per control block (`after_proj`).
    after_proj: Vec<QLinear>,
    /// Zero-init `before_proj` on control block 0 (`inner → inner`): `c = before_proj(c) + img_embed`.
    before_proj: QLinear,
    /// Base block indices each control hint injects into (`control_layers`); `places[n]` is the base
    /// index for hint `n`.
    places: Vec<usize>,
}

impl QwenFunControlBranch {
    /// Load from the 2512-Fun checkpoint. `control_layers` must contain `0` (`before_proj` lives on
    /// control block 0). Keys: `control_img_in.{weight,bias}`, `control_blocks.{i}.*` (a base block +
    /// `after_proj` for every `i`, plus `before_proj` on `i == 0`).
    pub fn new(
        cfg: &TransformerConfig,
        control_layers: &[usize],
        control_in_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let inner = cfg.inner_dim();
        // Every projection here loads through `QLinear::linear_detect_gs`, which packed-detects
        // per-key: the caller-provided `controlnet` path may be the dense alibaba-pai
        // `Qwen-Image-2512-Fun-Controlnet-Union` checkpoint **or** the shared packed control tier
        // `SceneWorks/qwen-image-2512-fun-controlnet-union` (bf16/q8/q4 — each q4/q8 subdir a single
        // `model.safetensors` of packed `{base}.weight` u32 + `.scales` + `.biases` triples). A dense
        // checkpoint takes the dense path (group size inert); a packed tier packed-detects with **no
        // code change** (sc-9869). The Qwen-Image tiers pack at the MLX default group size 64.
        let gs = candle_gen::quant::MLX_GROUP_SIZE;
        let n = control_layers.len();
        let mut blocks = Vec::with_capacity(n);
        let mut after_proj = Vec::with_capacity(n);
        for i in 0..n {
            let blk = vb.pp("control_blocks").pp(i);
            blocks.push(Block::new(cfg, blk.clone(), gs)?);
            after_proj.push(QLinear::linear_detect_gs(
                inner,
                inner,
                &blk,
                "after_proj",
                true,
                gs,
            )?);
        }
        Ok(Self {
            control_img_in: QLinear::linear_detect_gs(
                control_in_dim,
                inner,
                &vb,
                "control_img_in",
                true,
                gs,
            )?,
            before_proj: QLinear::linear_detect_gs(
                inner,
                inner,
                &vb.pp("control_blocks").pp(0),
                "before_proj",
                true,
                gs,
            )?,
            blocks,
            after_proj,
            places: control_layers.to_vec(),
        })
    }

    /// Number of control hints (= control layers); drives the base injection sites.
    pub fn num_hints(&self) -> usize {
        self.blocks.len()
    }

    /// The hint index injected at base block `idx`, or `None`. Mirrors the fork's
    /// `control_layers_mapping`.
    pub fn hint_index(&self, idx: usize) -> Option<usize> {
        self.places.iter().position(|&p| p == idx)
    }

    /// Run the VACE control stack → the per-block hints (pre-scale), one per control layer. The fork's
    /// `forward_control`: `c = control_img_in(control_context)`; block 0 seeds
    /// `c = before_proj(c) + img_embed`; each control block runs the *base* block math (reusing the base
    /// modulation / RoPE / timestep) and threads its own text stream to the next; `hint[i] =
    /// after_proj(c_after_block_i)`. The threaded text is local to the control stack — only the
    /// image-stream hints leave.
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        img_embed: &Tensor,
        encoder_embed: &Tensor,
        control_context: &Tensor,
        temb: &Tensor,
        img_cos: &Tensor,
        img_sin: &Tensor,
        txt_cos: &Tensor,
        txt_sin: &Tensor,
    ) -> Result<Vec<Tensor>> {
        // c = control_img_in(control_context); seed block 0 with `before_proj(c) + img_embed`.
        let c0 = self.control_img_in.forward(control_context)?;
        let mut c = (self.before_proj.forward(&c0)? + img_embed)?;
        let mut encoder = encoder_embed.clone();
        let mut hints = Vec::with_capacity(self.blocks.len());
        for (block, ap) in self.blocks.iter().zip(&self.after_proj) {
            let (e, new_c) =
                block.forward(&c, &encoder, temb, img_cos, img_sin, txt_cos, txt_sin, None)?;
            encoder = e;
            // hint[i] = after_proj(c_after_block_i) (zero-init projection; the fork's `c_skip`).
            hints.push(ap.forward(&new_c)?);
            c = new_c;
        }
        Ok(hints)
    }
}

impl QwenTransformer {
    /// [`forward`](Self::forward) with the **2512-Fun-Controlnet-Union** VACE control branch (sc-8350).
    /// Identical to the T2I forward, plus: the control branch's per-block hints are computed once from
    /// the post-embedder image + text streams (reusing the base modulation / RoPE / timestep), and after
    /// base block `control_layers[n]` the hint `hints[n]·control_scale` is added to the image stream —
    /// the fork's `QwenImageControlTransformer2DModel.forward`. `control = None` is **byte-identical** to
    /// the plain forward, as is `control_scale = 0` (the zero-init `after_proj` + `+0` injection).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_fun_control(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        timestep: f32,
        lat_h: usize,
        lat_w: usize,
        control: Option<(&QwenFunControlBranch, &Tensor)>,
        control_scale: f32,
    ) -> Result<Tensor> {
        let temb = self
            .time_embed
            .forward(timestep, &self.device, self.dtype)?;
        let mut hidden = self.img_in.forward(hidden_states)?;
        let encoder = self.txt_norm.forward(encoder_hidden_states)?;
        let mut encoder = self.txt_in.forward(&encoder)?;

        let txt_seq = encoder.dim(1)?;
        // Step-invariant (fixed grid), so cache the RoPE tables per render (sc-8992).
        let (img_cos, img_sin, txt_cos, txt_sin) =
            self.rope_cache
                .tables(&self.rope, &[(lat_h, lat_w)], txt_seq, &self.device)?;

        // VACE control hints (sc-8350): computed once from the post-embedder image + text streams,
        // before the base block loop (the fork's `forward_control`), then injected per block. The hints
        // are pre-scaled by `control_scale` once (the scalar is the same across all hints and blocks);
        // `control = None` or `control_scale = 0` → no injection (byte-identical base).
        let scaled_hints: Option<Vec<Tensor>> = match control {
            Some((branch, cc)) => {
                let hints = branch.forward(
                    &hidden, &encoder, cc, &temb, &img_cos, &img_sin, &txt_cos, &txt_sin,
                )?;
                Some(
                    hints
                        .iter()
                        .map(|h| h * control_scale as f64)
                        .collect::<Result<Vec<_>>>()?,
                )
            }
            None => None,
        };

        for (i, block) in self.blocks.iter().enumerate() {
            let (e, h) = block.forward(
                &hidden, &encoder, &temb, &img_cos, &img_sin, &txt_cos, &txt_sin, None,
            )?;
            encoder = e;
            // After base block `i`, add the pre-scaled hint for this block (if `i` is a control layer) —
            // the fork's `hidden_states = hidden_states + hints[block_id] * context_scale`.
            hidden = match (&scaled_hints, control) {
                (Some(hints), Some((branch, _))) => match branch.hint_index(i) {
                    Some(n) => (h + &hints[n])?,
                    None => h,
                },
                _ => h,
            };
        }

        let hidden = self.norm_out.forward(&hidden, &temb)?;
        self.proj_out.forward(&hidden)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunked_attention_matches_single_pass() {
        // Per-query-row softmax is independent, so chunking over query rows (forced via a tiny budget)
        // must match the single pass bit-for-bit — the guard for the i32-overflow fix (sc-6217).
        // Retargeted onto the shared `candle_gen::sdpa_budgeted_bhsd` (sc-9570) with this crate's exact
        // `softmax_last_dim` closure and no mask.
        let dev = Device::Cpu;
        let (b, h, s, d) = (1usize, 2usize, 7usize, 4usize);
        let q = Tensor::randn(0f32, 1f32, (b, h, s, d), &dev).unwrap();
        let k = Tensor::randn(0f32, 1f32, (b, h, s, d), &dev).unwrap();
        let v = Tensor::randn(0f32, 1f32, (b, h, s, d), &dev).unwrap();
        let scale = (d as f64).powf(-0.5);
        let sm = |x: &Tensor| softmax_last_dim(x);
        // Huge budget → single pass; tiny budget (1) → single-row chunks; a MID-SIZE budget forces
        // multi-row chunks + a remainder (block=3 over s=7 → 3,3,1) — the sc-9116 hardening ask.
        let single =
            candle_gen::sdpa_budgeted_bhsd(&q, &k, &v, scale, None, sm, usize::MAX).unwrap();
        let a = single.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // budget = b·h·s·block = 1·2·7·3 = 42 → block = 42/(1·2·7) = 3.
        for budget in [1usize, 42] {
            let chunked =
                candle_gen::sdpa_budgeted_bhsd(&q, &k, &v, scale, None, sm, budget).unwrap();
            assert_eq!(single.dims(), chunked.dims());
            let c = chunked.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            for (x, y) in a.iter().zip(&c) {
                assert!(
                    (x - y).abs() < 1e-6,
                    "chunked attention diverged at budget {budget}: {x} vs {y}"
                );
            }
        }
    }

    use candle_gen::candle_nn::var_builder::SimpleBackend;
    use std::sync::Mutex;

    /// A deterministic random [`SimpleBackend`] for the no-weights control tests: returns a small
    /// reproducible normal tensor of the requested shape for any key (a fresh seeded RNG per
    /// `VarBuilder`, advanced per `get` so distinct keys get distinct — but reproducible — tensors).
    /// `Mutex` (not `RefCell`) because `SimpleBackend: Send + Sync`.
    struct RandomBackend {
        rng: Mutex<rand::rngs::StdRng>,
    }

    impl RandomBackend {
        fn new(seed: u64) -> Self {
            use rand::SeedableRng;
            Self {
                rng: Mutex::new(rand::rngs::StdRng::seed_from_u64(seed)),
            }
        }
    }

    impl SimpleBackend for RandomBackend {
        fn get(
            &self,
            s: candle_gen::candle_core::Shape,
            _name: &str,
            _h: candle_gen::candle_nn::Init,
            dtype: DType,
            dev: &Device,
        ) -> candle_gen::candle_core::Result<Tensor> {
            let n: usize = s.elem_count();
            let mut rng = candle_gen::lock_recover(&self.rng);
            // Small magnitude keeps the tiny DiT numerically sane (and norm-out + RMSNorm stable).
            let data: Vec<f32> = candle_gen::seeded_normal_vec(&mut rng, n)
                .into_iter()
                .map(|v| 0.05f32 * v)
                .collect();
            Tensor::from_vec(data, s, dev)?.to_dtype(dtype)
        }

        fn get_unchecked(
            &self,
            _name: &str,
            _dtype: DType,
            _dev: &Device,
        ) -> candle_gen::candle_core::Result<Tensor> {
            candle_gen::candle_core::bail!("RandomBackend requires a shape; use get")
        }

        fn contains_tensor(&self, name: &str) -> bool {
            // A **dense** random backend: it synthesizes a `{base}.weight`/`.bias` for any key via
            // `get`, but has **no** MLX-packed `.scales`/`.biases` sibling. So report `false` for the
            // packed-detect markers — otherwise `QLinear::linear_detect_gs` would take the packed path
            // and call `get_unchecked_dtype` (unsupported here). This keeps the no-weights control
            // wiring tests on the dense path (the InstantX / 2512-Fun control checkpoints are dense).
            !(name.ends_with(".scales") || name.ends_with(".biases"))
        }
    }

    /// A tiny Qwen config (4 base blocks, 2 heads × 8) for the no-weights control wiring tests.
    fn tiny_cfg() -> TransformerConfig {
        TransformerConfig {
            in_channels: 8,
            out_channels: 2,
            num_layers: 4,
            num_heads: 2,
            head_dim: 8,
            joint_attention_dim: 12,
            timestep_channels: 16,
            axes_dim: [2, 3, 3],
            rope_theta: 10_000.0,
            eps: 1e-6,
        }
    }

    fn random_vb(seed: u64) -> VarBuilder<'static> {
        VarBuilder::from_backend(Box::new(RandomBackend::new(seed)), DType::F32, Device::Cpu)
    }

    /// scale=0 (and `control = None`) reproduce the plain base forward **byte-exact** — the zero-init
    /// `after_proj` plus the `+0` injection means the control branch contributes nothing. This is the
    /// load-bearing parity guarantee (the base T2I/Edit path is untouched by the new lane).
    #[test]
    fn fun_control_scale_zero_is_byte_exact_base() {
        let cfg = tiny_cfg();
        let dev = Device::Cpu;
        // Base transformer + a 2512-Fun control branch (2 control layers for the tiny model) from
        // independent random backends.
        let transformer = QwenTransformer::new(&cfg, random_vb(1)).unwrap();
        let control_layers = [0usize, 2];
        let control_in_dim = 132; // the real packed control-context width (independent of the tiny DiT)
        let branch =
            QwenFunControlBranch::new(&cfg, &control_layers, control_in_dim, random_vb(2)).unwrap();

        let (lat_h, lat_w) = (2usize, 3usize);
        let seq = lat_h * lat_w;
        let txt_seq = 5usize;
        let hidden = Tensor::randn(0f32, 1f32, (1, seq, cfg.in_channels), &dev).unwrap();
        let encoder =
            Tensor::randn(0f32, 1f32, (1, txt_seq, cfg.joint_attention_dim), &dev).unwrap();
        let control_cond = Tensor::randn(0f32, 1f32, (1, seq, control_in_dim), &dev).unwrap();
        let sigma = 0.7f32;

        let base = transformer
            .forward(&hidden, &encoder, sigma, lat_h, lat_w)
            .unwrap();

        // control = None ≡ base.
        let none = transformer
            .forward_fun_control(&hidden, &encoder, sigma, lat_h, lat_w, None, 0.0)
            .unwrap();
        // control = Some(branch) with scale 0 ≡ base (the injection is `+ hint·0`).
        let scaled0 = transformer
            .forward_fun_control(
                &hidden,
                &encoder,
                sigma,
                lat_h,
                lat_w,
                Some((&branch, &control_cond)),
                0.0,
            )
            .unwrap();

        let b = base.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let n = none.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let z = scaled0.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(b, n, "control=None must be byte-exact base");
        assert_eq!(b, z, "control_scale=0 must be byte-exact base");

        // And a non-zero scale actually changes the output (the lane is wired, not inert).
        let scaled1 = transformer
            .forward_fun_control(
                &hidden,
                &encoder,
                sigma,
                lat_h,
                lat_w,
                Some((&branch, &control_cond)),
                1.0,
            )
            .unwrap();
        let s1 = scaled1.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(
            b.iter().zip(&s1).any(|(x, y)| (x - y).abs() > 1e-6),
            "control_scale=1 must change the output (branch must be wired)"
        );
    }

    /// **Forward-time additive install wiring (sc-11091).** A LoRA over two real Lightning projections
    /// (`transformer_blocks.0.attn.to_q` + `…img_mlp.net.0.proj`) installs as forward-time residuals via
    /// [`crate::adapters::install_additive`], applies (`report.applied == 2`), and **shifts** the DiT
    /// forward vs the un-adapted base (identical weights); a scale-0 install is a byte-exact no-op, and
    /// an adapter surface that matches nothing errors (never renders unadapted). The base here is the
    /// tiny **dense** DiT — install_additive is base-agnostic (it pushes residuals, never folds), so the
    /// packed-footprint / stays-packed property is proven at the `AdaptLinear` unit level
    /// (`candle-gen/src/quant/adapt.rs`); the real q4/q8 GPU render is the offload re-measure follow-up.
    #[test]
    fn install_additive_applies_shifts_and_zero_is_noop() {
        use candle_gen::candle_core::safetensors::save;
        use candle_gen::gen_core::{AdapterKind, AdapterSpec};
        use std::collections::HashMap as Map;

        let cfg = tiny_cfg();
        let dev = Device::Cpu;
        let inner = cfg.inner_dim(); // 16
        let ff_hidden = inner * 4; // 64
        let rank = 4usize;

        // A LoRA over two real projections; small factors so the residual is a clean, finite shift.
        let mk =
            |r: usize, c: usize| (Tensor::randn(0f32, 1f32, (r, c), &dev).unwrap() * 0.1).unwrap();
        let mut lora: Map<String, Tensor> = Map::new();
        for (path, out) in [
            ("transformer_blocks.0.attn.to_q", inner),
            ("transformer_blocks.0.img_mlp.net.0.proj", ff_hidden),
        ] {
            lora.insert(format!("{path}.lora_down.weight"), mk(rank, inner)); // A [rank, in]
            lora.insert(format!("{path}.lora_up.weight"), mk(out, rank)); // B [out, rank]
        }
        let tmp = std::env::temp_dir().join(format!(
            "sc11091_lora_{}_{}.safetensors",
            std::process::id(),
            cfg.num_layers
        ));
        save(&lora, &tmp).unwrap();

        // Base (un-adapted) and adapted DiTs share identical weights (same random seed).
        let base_dit = QwenTransformer::new(&cfg, random_vb(7)).unwrap();
        let mut adapted = QwenTransformer::new(&cfg, random_vb(7)).unwrap();
        let spec = AdapterSpec::new(tmp.clone(), 1.0, AdapterKind::Lora);
        let report = crate::adapters::install_additive(&mut adapted, &[spec]).unwrap();
        assert_eq!(
            report.applied, 2,
            "both projections must receive a residual"
        );
        assert!(
            report.skipped_targets.is_empty(),
            "no target should be unrouted"
        );

        // Same inputs through both: the adapter must SHIFT the forward, and stay finite.
        let (lat_h, lat_w) = (2usize, 3usize);
        let seq = lat_h * lat_w;
        let hidden = Tensor::randn(0f32, 1f32, (1, seq, cfg.in_channels), &dev).unwrap();
        let encoder = Tensor::randn(0f32, 1f32, (1, 5, cfg.joint_attention_dim), &dev).unwrap();
        let b = base_dit
            .forward(&hidden, &encoder, 0.7, lat_h, lat_w)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let a = adapted
            .forward(&hidden, &encoder, 0.7, lat_h, lat_w)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(
            a.iter().all(|v| v.is_finite()),
            "additive DiT forward must stay finite"
        );
        assert!(
            b.iter().zip(&a).any(|(x, y)| (x - y).abs() > 1e-5),
            "the additive LoRA must shift the DiT forward"
        );

        // scale 0 ⇒ byte-exact base (the mutation anchor: break the scale bake and this breaks).
        let mut zero = QwenTransformer::new(&cfg, random_vb(7)).unwrap();
        let spec0 = AdapterSpec::new(tmp.clone(), 0.0, AdapterKind::Lora);
        crate::adapters::install_additive(&mut zero, &[spec0]).unwrap();
        let z = zero
            .forward(&hidden, &encoder, 0.7, lat_h, lat_w)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(b, z, "scale-0 additive install must be a byte-exact no-op");

        std::fs::remove_file(&tmp).ok();

        // An adapter surface that matches NO DiT module errors (never renders unadapted silently).
        let mut miss_map: Map<String, Tensor> = Map::new();
        miss_map.insert(
            "transformer_blocks.0.attn.no_such_proj.lora_down.weight".into(),
            mk(rank, inner),
        );
        miss_map.insert(
            "transformer_blocks.0.attn.no_such_proj.lora_up.weight".into(),
            mk(inner, rank),
        );
        let miss = std::env::temp_dir().join(format!(
            "sc11091_miss_{}_{}.safetensors",
            std::process::id(),
            cfg.num_heads
        ));
        save(&miss_map, &miss).unwrap();
        let mut dit2 = QwenTransformer::new(&cfg, random_vb(7)).unwrap();
        let miss_spec = AdapterSpec::new(miss.clone(), 1.0, AdapterKind::Lora);
        assert!(
            crate::adapters::install_additive(&mut dit2, &[miss_spec]).is_err(),
            "an all-miss adapter surface must error, not render unadapted"
        );
        std::fs::remove_file(&miss).ok();
    }

    /// The branch wiring: 2 control layers → 2 hints, injected at base blocks `[0, 2]` (and only there).
    #[test]
    fn fun_control_branch_hint_wiring() {
        let cfg = tiny_cfg();
        let control_layers = [0usize, 2];
        let branch = QwenFunControlBranch::new(&cfg, &control_layers, 132, random_vb(3)).unwrap();
        assert_eq!(branch.num_hints(), 2);
        assert_eq!(branch.hint_index(0), Some(0));
        assert_eq!(branch.hint_index(2), Some(1));
        assert_eq!(branch.hint_index(1), None);
        assert_eq!(branch.hint_index(3), None);
    }
}
