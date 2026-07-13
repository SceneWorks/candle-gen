//! SANA **Linear Diffusion Transformer trunk** — component-for-component candle port of diffusers
//! `SanaTransformer2DModel` / `SanaTransformerBlock`, mirroring the mlx-gen-sana trunk (mlx-gen #613,
//! story sc-8487). This is sc-11778, epic 11776 — the candle/CUDA sibling of that MLX port.
//!
//! Port target: `Efficient-Large-Model/Sana_1600M_1024px_diffusers` (the 1.6B model). We write the
//! **f32 image inference path** (the port-playbook's "f32-or-split" convention; the whole trunk runs
//! f32 like the sibling DC-AE decoder, and the linear-attention `1/(Σ+eps)` normalizer is f32 in the
//! reference regardless). No training path.
//!
//! ## Architecture (the four story pillars)
//!
//!  - **ReLU linear self-attention** (`attn1`, `SanaLinearAttnProcessor2_0`) — O(N) attention:
//!    `ReLU(Q),ReLU(K)`, then the reference's `value`-padded-ones-row normalizer collapsed to the
//!    algebraically identical numerator/denominator split `num = (V·Kᵀ)·Q`, `den = (Σ_n K)·Q`,
//!    divided with a `1/(·+1e-15)` normalizer. This is the **same shared hard primitive the DC-AE
//!    spike wrote once** — [`crate::dc_ae::relu_linear_attention`] — reused verbatim here (the trunk's
//!    `attn1` is the plain single-scale case: no multiscale QKV projections). `attention_bias=false`
//!    for SANA-1.6B → `to_q/k/v` bias-free; `to_out.0` carries a bias.
//!  - **Cross-attention** (`attn2`, standard softmax SDPA) to the caption embeddings — `to_q/k/v` all
//!    bias-carrying, KV from the projected+normed caption.
//!  - **Mix-FFN** (`ff`, `GLUMBConv`) — `conv_inverted(1×1) → SiLU → conv_depth(3×3 depthwise) →
//!    gated SiLU → conv_point(1×1, no bias)`. The 3×3 depthwise conv is the token-mixer. This reuses
//!    the spike's shared [`crate::dc_ae::glu_mbconv_core`] (the DC-AE `GLUMBConv` and this Mix-FFN are
//!    the same gated inverted-bottleneck; the block here owns its own residual + modulation-gate, so
//!    it uses the bare core with `norm_type=None, residual_connection=False`).
//!  - **NoPE** — `interpolation_scale=None` ⇒ `patch_embed` carries no `pos_embed`; the conv patchify
//!    (here `patch_size=1`, a 1×1 conv) plus the Mix-FFN depthwise conv provide all locality.
//!
//! Per-block adaLN-single modulation `(shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp,
//! gate_mlp)` comes from `block.scale_shift_table[6,dim] + timestep_emb.reshape(B,6,-1)`; the
//! timestep path is `Timesteps(256) → timestep_embedder(MLP) = embedded_timestep`, then
//! `time_embed.linear(SiLU(embedded_timestep)) → [B, 6·dim]`. Output: `SanaModulatedNorm`
//! (affine-free LayerNorm + `scale_shift_table[2,dim] + embedded_timestep`) → `proj_out` →
//! unpatchify to `[B, out_channels, H, W]` (32 channels = the DC-AE f32c32 latent, so the trunk's
//! output feeds [`crate::dc_ae::DcAeDecoder::decode`] directly).
//!
//! **Layout: token `[B, N, C]` for the attention/Linear/LayerNorm ops, NCHW `[B, C, H, W]` for the
//! conv ops** — the same split the reference uses, except candle stays NCHW-native for the convs
//! where MLX transposed to NHWC (numerically identical: the conv weights load `[O, I/groups, kH, kW]`
//! as-is and the Mix-FFN grid is arranged into the same spatial order). Tensor keys are the diffusers
//! `SanaTransformer2DModel` names exactly, so a converted checkpoint (or the committed tiny golden)
//! loads unchanged.

use candle_gen::candle_core::{DType, Result, Tensor, D};
use candle_gen::candle_nn::ops::{silu, softmax_last_dim};
use candle_gen::candle_nn::{Conv2d, Linear, Module};
use candle_gen::Weights;

use crate::config::SanaTransformerConfig;
use crate::dc_ae::{conv, glu_mbconv_core, relu_linear_attention};

// ----------------------------------------------------------------------------------------------
// Shared scalar / norm primitives (f32).
// ----------------------------------------------------------------------------------------------

/// `nn.Linear` (weight `[out, in]`, optional bias) applied over the last axis — via candle's
/// batched [`Linear`], which handles the `[B, N, in]` token layout. Weights stored f32.
fn linear(w: &Weights, prefix: &str, bias: bool) -> candle_gen::Result<Linear> {
    let weight = w
        .require(&format!("{prefix}.weight"))?
        .to_dtype(DType::F32)?;
    let b = if bias {
        Some(w.require(&format!("{prefix}.bias"))?.to_dtype(DType::F32)?)
    } else {
        None
    };
    Ok(Linear::new(weight, b))
}

/// Affine-free LayerNorm over the last axis, f32 (diffusers `norm1`/`norm2`/`norm_out` are all
/// `elementwise_affine=False` — the adaLN modulation supplies the affine externally).
fn layer_norm_affine_free(x: &Tensor, eps: f64) -> Result<Tensor> {
    let dt = x.dtype();
    let xf = x.to_dtype(DType::F32)?;
    let mean = xf.mean_keepdim(D::Minus1)?;
    let centered = xf.broadcast_sub(&mean)?;
    let var = centered.sqr()?.mean_keepdim(D::Minus1)?;
    let normed = centered.broadcast_div(&(var + eps)?.sqrt()?)?;
    normed.to_dtype(dt)
}

/// `RMSNorm(elementwise_affine=True, bias=False)` over the last axis, f32 reduction (diffusers
/// `caption_norm`, and — Sprint — the `rms_norm_across_heads` qk-norm). `weight` is `[C]`.
fn rms_norm_last(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let dt = x.dtype();
    let xf = x.to_dtype(DType::F32)?;
    let var = xf.sqr()?.mean_keepdim(D::Minus1)?;
    let normed = xf.broadcast_div(&(var + eps)?.sqrt()?)?.to_dtype(dt)?;
    normed.broadcast_mul(weight)
}

/// adaLN-single affine `norm · (1 + scale) + shift` (diffusers `hidden * (1 + scale) + shift`).
/// `scale`/`shift` are `[B, 1, dim]`, `norm` is `[B, N, dim]` — broadcast over the token axis.
fn modulate(norm: &Tensor, scale: &Tensor, shift: &Tensor) -> Result<Tensor> {
    let gained = norm.broadcast_mul(&(scale + 1.0)?)?;
    gained.broadcast_add(shift)
}

/// Sinusoidal timestep embedding (diffusers `Timesteps`, `flip_sin_to_cos=True` ⇒ `[cos | sin]`).
/// `t` is `[B]`; returns `[B, dim]`. For SANA: `dim=256, max_period=10000, downscale_freq_shift=0`.
fn timestep_sincos(
    t: &Tensor,
    dim: usize,
    max_period: f64,
    downscale_freq_shift: f64,
) -> Result<Tensor> {
    let half = dim / 2;
    let neg_log = -(max_period.ln()) as f32;
    let denom = (half as f64 - downscale_freq_shift) as f32;
    let freqs: Vec<f32> = (0..half)
        .map(|i| ((i as f32) * neg_log / denom).exp())
        .collect();
    let dev = t.device();
    let f = Tensor::from_vec(freqs, (1, half), dev)?; // [1, half]
    let b = t.dim(0)?;
    let t = t.to_dtype(DType::F32)?.reshape((b, 1))?; // [B, 1]
    let emb = t.broadcast_mul(&f)?; // [B, half]
    Tensor::cat(&[&emb.cos()?, &emb.sin()?], 1) // [B, dim]  — cos first
}

// ----------------------------------------------------------------------------------------------
// ReLU linear self-attention (attn1) — reuses the DC-AE spike's shared primitive.
// ----------------------------------------------------------------------------------------------

/// `SanaLinearAttnProcessor2_0`: ReLU linear attention over the token axis. Input/output `[B, N, C]`.
struct LinearSelfAttn {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    /// Sprint `qk_norm = "rms_norm_across_heads"`: RMSNorm over the full projected query / key (the
    /// whole `inner_dim`), applied BEFORE the head split and the ReLU. `None` for base SANA.
    norm_q: Option<Tensor>,
    norm_k: Option<Tensor>,
    heads: usize,
    attn_eps: f64,
    qk_norm_eps: f64,
}

impl LinearSelfAttn {
    fn load(w: &Weights, prefix: &str, cfg: &SanaTransformerConfig) -> candle_gen::Result<Self> {
        let (norm_q, norm_k) = if cfg.qk_norm {
            (
                Some(
                    w.require(&format!("{prefix}.norm_q.weight"))?
                        .to_dtype(DType::F32)?,
                ),
                Some(
                    w.require(&format!("{prefix}.norm_k.weight"))?
                        .to_dtype(DType::F32)?,
                ),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            // attention_bias=false → q/k/v bias-free; to_out.0 carries a bias.
            to_q: linear(w, &format!("{prefix}.to_q"), false)?,
            to_k: linear(w, &format!("{prefix}.to_k"), false)?,
            to_v: linear(w, &format!("{prefix}.to_v"), false)?,
            to_out: linear(w, &format!("{prefix}.to_out.0"), true)?,
            norm_q,
            norm_k,
            heads: cfg.num_attention_heads as usize,
            attn_eps: cfg.attn_eps as f64,
            qk_norm_eps: cfg.attn_qk_norm_eps as f64,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, n, _c) = x.dims3()?;

        // qk_norm (Sprint): RMSNorm over the full inner_dim BEFORE the head split (diffusers applies
        // `attn.norm_q(query)` / `attn.norm_k(key)` to the `[B,N,inner]` projection).
        let q = self.to_q.forward(x)?;
        let q = match &self.norm_q {
            Some(g) => rms_norm_last(&q, g, self.qk_norm_eps)?,
            None => q,
        };
        let k = self.to_k.forward(x)?;
        let k = match &self.norm_k {
            Some(g) => rms_norm_last(&k, g, self.qk_norm_eps)?,
            None => k,
        };
        let v = self.to_v.forward(x)?;
        let inner = q.dim(D::Minus1)?;
        let hd = inner / self.heads;

        // [B,N,inner] → [B,heads,hd,N] (diffusers transpose(1,2).unflatten(1,(heads,-1))). This is the
        // exact `[B, groups, head_dim, N]` layout the shared spike primitive consumes.
        let to_hdn = |a: &Tensor| -> Result<Tensor> {
            a.reshape((b, n, self.heads, hd))?
                .permute((0, 2, 3, 1))?
                .contiguous()
        };
        let q = to_hdn(&q)?.relu()?; // φ(q) = ReLU(q)
        let k = to_hdn(&k)?.relu()?; // φ(k) = ReLU(k)
        let v = to_hdn(&v)?;

        // Reused verbatim from the DC-AE spike (sc-11777): num = (V·Kᵀ)·Q, den = (Σ_n K)·Q, / (·+eps).
        let out = relu_linear_attention(&q, &k, &v, self.attn_eps)?; // [B,heads,hd,N]

        // [B,heads,hd,N] → [B,N,inner]
        let out = out.permute((0, 3, 1, 2))?.reshape((b, n, inner))?;
        let out = self.to_out.forward(&out)?;

        // The reference clips `to_out` to fp16's representable range as an overflow guard, but only
        // when the input dtype was fp16 (`if original_dtype == torch.float16`). Our f32 path is
        // unchanged; kept for fidelity if a caller ever runs this trunk in f16.
        if x.dtype() == DType::F16 {
            out.clamp(-65504.0f32, 65504.0f32)
        } else {
            Ok(out)
        }
    }
}

// ----------------------------------------------------------------------------------------------
// Standard cross-attention (attn2) to the caption embedding.
// ----------------------------------------------------------------------------------------------

struct CrossAttn {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    norm_q: Option<Tensor>,
    norm_k: Option<Tensor>,
    heads: usize,
    qk_norm_eps: f64,
}

impl CrossAttn {
    fn load(w: &Weights, prefix: &str, cfg: &SanaTransformerConfig) -> candle_gen::Result<Self> {
        let (norm_q, norm_k) = if cfg.qk_norm {
            (
                Some(
                    w.require(&format!("{prefix}.norm_q.weight"))?
                        .to_dtype(DType::F32)?,
                ),
                Some(
                    w.require(&format!("{prefix}.norm_k.weight"))?
                        .to_dtype(DType::F32)?,
                ),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            to_q: linear(w, &format!("{prefix}.to_q"), true)?,
            to_k: linear(w, &format!("{prefix}.to_k"), true)?,
            to_v: linear(w, &format!("{prefix}.to_v"), true)?,
            to_out: linear(w, &format!("{prefix}.to_out.0"), true)?,
            norm_q,
            norm_k,
            heads: cfg.num_cross_attention_heads as usize,
            qk_norm_eps: cfg.attn_qk_norm_eps as f64,
        })
    }

    /// `x` (query) `[B, N, dim]`, `kv` (caption) `[B, M, dim]`.
    fn forward(&self, x: &Tensor, kv: &Tensor) -> Result<Tensor> {
        let (b, n, _) = x.dims3()?;
        let m = kv.dim(1)?;

        let q = self.to_q.forward(x)?;
        let q = match &self.norm_q {
            Some(g) => rms_norm_last(&q, g, self.qk_norm_eps)?,
            None => q,
        };
        let k = self.to_k.forward(kv)?;
        let k = match &self.norm_k {
            Some(g) => rms_norm_last(&k, g, self.qk_norm_eps)?,
            None => k,
        };
        let v = self.to_v.forward(kv)?;
        let inner = q.dim(D::Minus1)?;
        let hd = inner / self.heads;
        let scale = 1.0 / (hd as f64).sqrt();

        // [B,len,inner] → [B,heads,len,hd]
        let split = |a: &Tensor, len: usize| -> Result<Tensor> {
            a.reshape((b, len, self.heads, hd))?
                .permute((0, 2, 1, 3))?
                .contiguous()
        };
        let q = split(&q, n)?; // [B,H,N,hd]
        let k = split(&k, m)?; // [B,H,M,hd]
        let v = split(&v, m)?; // [B,H,M,hd]

        // Softmax SDPA in f32 (caption seq is short; full attention).
        let scores = q
            .matmul(&k.transpose(D::Minus1, D::Minus2)?.contiguous()?)?
            .affine(scale, 0.0)?; // [B,H,N,M]
        let probs = softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?; // [B,H,N,hd]

        let ctx = ctx.permute((0, 2, 1, 3))?.reshape((b, n, inner))?;
        self.to_out.forward(&ctx)
    }
}

// ----------------------------------------------------------------------------------------------
// GLUMBConv Mix-FFN (block `ff`: norm_type=None, residual_connection=False) — reuses the spike core.
// ----------------------------------------------------------------------------------------------

struct MixFfn {
    conv_inverted: Conv2d, // 1×1, inner → 2·hidden (+bias)
    conv_depth: Conv2d,    // 3×3 depthwise, 2·hidden → 2·hidden (+bias)
    conv_point: Conv2d,    // 1×1, hidden → inner (no bias)
    hidden: usize,
}

impl MixFfn {
    fn load(w: &Weights, prefix: &str, cfg: &SanaTransformerConfig) -> candle_gen::Result<Self> {
        let inner = cfg.inner_dim() as usize;
        let hidden = (cfg.mlp_ratio * inner as f32) as usize;
        Ok(Self {
            conv_inverted: conv(w, &format!("{prefix}.conv_inverted"), 1, 0, 1, true)?,
            conv_depth: conv(w, &format!("{prefix}.conv_depth"), 1, 1, 2 * hidden, true)?,
            conv_point: conv(w, &format!("{prefix}.conv_point"), 1, 0, 1, false)?,
            hidden,
        })
    }

    /// `grid` is NCHW `[B, inner, H, W]`. Returns NCHW `[B, inner, H, W]`.
    fn forward(&self, grid: &Tensor) -> Result<Tensor> {
        glu_mbconv_core(
            &self.conv_inverted,
            &self.conv_depth,
            &self.conv_point,
            self.hidden,
            grid,
        )
    }
}

// ----------------------------------------------------------------------------------------------
// SanaTransformerBlock.
// ----------------------------------------------------------------------------------------------

struct SanaBlock {
    scale_shift_table: Tensor, // [6, dim]
    attn1: LinearSelfAttn,
    attn2: CrossAttn,
    ff: MixFfn,
    norm_eps: f64,
}

impl SanaBlock {
    fn load(w: &Weights, prefix: &str, cfg: &SanaTransformerConfig) -> candle_gen::Result<Self> {
        Ok(Self {
            scale_shift_table: w
                .require(&format!("{prefix}.scale_shift_table"))?
                .to_dtype(DType::F32)?,
            attn1: LinearSelfAttn::load(w, &format!("{prefix}.attn1"), cfg)?,
            attn2: CrossAttn::load(w, &format!("{prefix}.attn2"), cfg)?,
            ff: MixFfn::load(w, &format!("{prefix}.ff"), cfg)?,
            norm_eps: cfg.norm_eps as f64,
        })
    }

    /// `hidden` `[B, N, dim]` (N = H·W tokens), `caption` `[B, M, dim]`, `temb` `[B, 6·dim]`.
    fn forward(
        &self,
        hidden: &Tensor,
        caption: &Tensor,
        temb: &Tensor,
        h: usize,
        w: usize,
    ) -> Result<Tensor> {
        let (b, n, dim) = hidden.dims3()?;

        // 1. Modulation: scale_shift_table[None] + temb.reshape(B,6,-1) → chunk(6) along axis 1.
        let ss = self.scale_shift_table.reshape((1, 6, dim))?;
        let modg = ss.broadcast_add(&temb.reshape((b, 6, dim))?)?; // [B,6,dim]
        let chunk = |i: usize| -> Result<Tensor> { modg.narrow(1, i, 1) }; // [B,1,dim]
        let (shift_msa, scale_msa, gate_msa) = (chunk(0)?, chunk(1)?, chunk(2)?);
        let (shift_mlp, scale_mlp, gate_mlp) = (chunk(3)?, chunk(4)?, chunk(5)?);

        // 2. Self linear-attention.
        let norm_h = layer_norm_affine_free(hidden, self.norm_eps)?;
        let norm_h = modulate(&norm_h, &scale_msa, &shift_msa)?;
        let attn_out = self.attn1.forward(&norm_h)?;
        let hidden = (hidden + gate_msa.broadcast_mul(&attn_out)?)?;

        // 3. Cross-attention (no pre-norm in SANA — attn2 reads `hidden` directly).
        let cross = self.attn2.forward(&hidden, caption)?;
        let hidden = (cross + hidden)?;

        // 4. Mix-FFN. norm2 → modulate → un-flatten to NCHW [B,dim,H,W] → GLUMBConv → flatten → gate.
        let norm_h = layer_norm_affine_free(&hidden, self.norm_eps)?;
        let norm_h = modulate(&norm_h, &scale_mlp, &shift_mlp)?;
        // tokens [B,N,dim] → grid [B,H,W,dim] → NCHW [B,dim,H,W] (candle-native; MLX kept NHWC — same
        // spatial order, so the depthwise conv sees identical neighbourhoods).
        let grid = norm_h
            .reshape((b, h, w, dim))?
            .permute((0, 3, 1, 2))?
            .contiguous()?;
        let ff = self.ff.forward(&grid)?; // [B,dim,H,W]
        let ff = ff.permute((0, 2, 3, 1))?.reshape((b, n, dim))?; // [B,N,dim]
        hidden + gate_mlp.broadcast_mul(&ff)?
    }
}

// ----------------------------------------------------------------------------------------------
// Full trunk.
// ----------------------------------------------------------------------------------------------

/// SANA Linear-DiT trunk (`SanaTransformer2DModel`).
pub struct SanaTransformer {
    cfg: SanaTransformerConfig,
    patch_embed: Conv2d, // proj: in → inner (kernel/stride = patch_size)
    // timestep path (AdaLayerNormSingle.emb + .linear, or — Sprint — the combined
    // timestep+guidance embedder).
    ts_embedder_1: Linear,
    ts_embedder_2: Linear,
    time_linear: Linear, // → 6·inner
    /// Sprint: the extra guidance embedder (`SanaCombinedTimestepGuidanceEmbeddings`). `None` for base.
    guidance_embedder: Option<(Linear, Linear)>,
    // caption path
    caption_proj_1: Linear,
    caption_proj_2: Linear,
    caption_norm: Tensor, // RMSNorm weight [inner]
    blocks: Vec<SanaBlock>,
    scale_shift_table: Tensor, // [2, inner] (output modulated norm)
    proj_out: Linear,
}

impl SanaTransformer {
    pub fn from_weights(w: &Weights, cfg: SanaTransformerConfig) -> candle_gen::Result<Self> {
        let p = cfg.patch_size as usize;
        let patch_embed = conv(w, "patch_embed.proj", p, 0, 1, true)?;
        let mut blocks = Vec::with_capacity(cfg.num_layers as usize);
        for i in 0..cfg.num_layers {
            blocks.push(SanaBlock::load(
                w,
                &format!("transformer_blocks.{i}"),
                &cfg,
            )?);
        }
        // The Sprint guidance variant (`SanaCombinedTimestepGuidanceEmbeddings`) drops the `.emb.`
        // nesting AdaLayerNormSingle introduces and adds a parallel `guidance_embedder`.
        let (ts1_key, ts2_key, guidance_embedder) = if cfg.guidance_embeds {
            (
                "time_embed.timestep_embedder.linear_1",
                "time_embed.timestep_embedder.linear_2",
                Some((
                    linear(w, "time_embed.guidance_embedder.linear_1", true)?,
                    linear(w, "time_embed.guidance_embedder.linear_2", true)?,
                )),
            )
        } else {
            (
                "time_embed.emb.timestep_embedder.linear_1",
                "time_embed.emb.timestep_embedder.linear_2",
                None,
            )
        };
        Ok(Self {
            patch_embed,
            ts_embedder_1: linear(w, ts1_key, true)?,
            ts_embedder_2: linear(w, ts2_key, true)?,
            time_linear: linear(w, "time_embed.linear", true)?,
            guidance_embedder,
            caption_proj_1: linear(w, "caption_projection.linear_1", true)?,
            caption_proj_2: linear(w, "caption_projection.linear_2", true)?,
            caption_norm: w.require("caption_norm.weight")?.to_dtype(DType::F32)?,
            blocks,
            scale_shift_table: w.require("scale_shift_table")?.to_dtype(DType::F32)?,
            proj_out: linear(w, "proj_out", true)?,
            cfg,
        })
    }

    /// Forward one denoise step.
    ///
    /// * `latent_nchw` — `[B, in_channels, H, W]` (channels-first, diffusers-native).
    /// * `caption` — `[B, M, caption_channels]` caption embedding (M = 300 for SANA-1.6B).
    /// * `timestep` — `[B]` scalar timestep(s).
    ///
    /// Returns the noise prediction `[B, out_channels, H, W]` (channels-first), where
    /// `out_channels == 32` matches the DC-AE f32c32 latent so the output feeds
    /// [`crate::dc_ae::DcAeDecoder::decode`] directly.
    pub fn forward(
        &self,
        latent_nchw: &Tensor,
        caption: &Tensor,
        timestep: &Tensor,
    ) -> Result<Tensor> {
        self.forward_with_guidance(latent_nchw, caption, timestep, None)
    }

    /// [`Self::forward`] with an optional **embedded guidance scalar** (SANA-Sprint).
    ///
    /// * `guidance` — `[B]` the CFG-free guidance scalar (already multiplied by `guidance_embeds_scale`
    ///   by the caller). `Some` only for a Sprint-config trunk (`guidance_embeds = true`); `None` runs
    ///   the base AdaLN-single path. Sprint feeds the scale as an embedded conditioning input — it is
    ///   NOT classifier-free guidance (no uncond forward).
    pub fn forward_with_guidance(
        &self,
        latent_nchw: &Tensor,
        caption: &Tensor,
        timestep: &Tensor,
        guidance: Option<&Tensor>,
    ) -> Result<Tensor> {
        let cfg = &self.cfg;
        let dim = cfg.inner_dim() as usize;
        let (b, _c, height, width) = latent_nchw.dims4()?;
        let p = cfg.patch_size as usize;
        let (ph, pw) = (height / p, width / p);
        let n = ph * pw;

        // 1. Patch embed (NCHW, candle-native). [B,C,H,W] → conv → [B,dim,ph,pw] → tokens [B,N,dim].
        let x = self
            .patch_embed
            .forward(&latent_nchw.to_dtype(DType::F32)?)?; // [B,dim,ph,pw]
        let mut hidden = x.reshape((b, dim, n))?.transpose(1, 2)?.contiguous()?; // [B,N,dim]

        // 2. Timestep embedding → embedded_timestep emb [B,dim] and modulation temb [B,6·dim].
        let ts_proj = timestep_sincos(timestep, 256, 10_000.0, 0.0)?; // [B,256]
        let timesteps_emb = self
            .ts_embedder_2
            .forward(&silu(&self.ts_embedder_1.forward(&ts_proj)?)?)?; // [B,dim]
        let emb = match (&self.guidance_embedder, guidance) {
            (Some((g1, g2)), Some(g)) => {
                // Sprint: conditioning = timesteps_emb + guidance_emb (the guidance scalar through the
                // same sincos(256) projection + a parallel MLP), exactly as diffusers
                // `SanaCombinedTimestepGuidanceEmbeddings`.
                let g_proj = timestep_sincos(g, 256, 10_000.0, 0.0)?;
                let guidance_emb = g2.forward(&silu(&g1.forward(&g_proj)?)?)?;
                (timesteps_emb + guidance_emb)?
            }
            _ => timesteps_emb,
        };
        let temb = self.time_linear.forward(&silu(&emb)?)?; // [B,6·dim]

        // 3. Caption projection + RMSNorm.
        let cap = self.caption_proj_1.forward(caption)?;
        let cap = self.caption_proj_2.forward(&cap.gelu()?)?; // GELU(approximate="tanh")
        let caption = rms_norm_last(&cap, &self.caption_norm, cfg.caption_norm_eps as f64)?;

        // 4. Transformer blocks.
        for block in &self.blocks {
            hidden = block.forward(&hidden, &caption, &temb, ph, pw)?;
        }

        // 5. Output: SanaModulatedNorm(embedded_timestep) → proj_out → unpatchify.
        let ss = self.scale_shift_table.reshape((1, 2, dim))?;
        let modg = ss.broadcast_add(&emb.reshape((b, 1, dim))?)?; // [B,2,dim]
        let shift = modg.narrow(1, 0, 1)?; // [B,1,dim]
        let scale = modg.narrow(1, 1, 1)?; // [B,1,dim]
        let normed = layer_norm_affine_free(&hidden, cfg.norm_eps as f64)?;
        let hidden = modulate(&normed, &scale, &shift)?;

        let out = self.proj_out.forward(&hidden)?; // [B,N, p·p·out_channels]
                                                   // unpatchify: [B,ph,pw,p,p,oc] → permute(0,5,1,3,2,4) → [B,oc,ph·p,pw·p].
        let oc = cfg.out_channels as usize;
        out.reshape((b, ph, pw, p, p, oc))?
            .permute((0, 5, 1, 3, 2, 4))?
            .reshape((b, oc, ph * p, pw * p))
    }

    pub fn config(&self) -> &SanaTransformerConfig {
        &self.cfg
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Device;
    use std::collections::HashMap;

    /// Deterministic pseudo-random fill (LCG) — reproducible on any backend, no rand dep. Matches the
    /// convention of the DC-AE spike's tests.
    fn det(shape: &[usize], seed: u64, dev: &Device) -> Tensor {
        let n: usize = shape.iter().product();
        let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((s >> 33) as f64) / ((1u64 << 31) as f64) - 1.0;
            v.push(u as f32);
        }
        Tensor::from_vec(v, shape, dev).unwrap()
    }

    #[test]
    fn timestep_sincos_matches_reference() {
        // Independent hand computation of diffusers `Timesteps(256, flip_sin_to_cos=True,
        // downscale_freq_shift=0)`: freq[i] = exp(-ln(10000)·i/128), emb = [cos(t·freq) | sin(t·freq)].
        let dev = Device::Cpu;
        let t_val = 0.6f32;
        let t = Tensor::from_vec(vec![t_val], (1,), &dev).unwrap();
        let dim = 256usize;
        let got = timestep_sincos(&t, dim, 10_000.0, 0.0)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        let half = dim / 2;
        let neg_log = -(10_000f32.ln());
        let mut want = vec![0f32; dim];
        for i in 0..half {
            let freq = (neg_log * i as f32 / half as f32).exp();
            want[i] = (t_val * freq).cos();
            want[half + i] = (t_val * freq).sin();
        }
        let max_d = got
            .iter()
            .zip(want.iter())
            .fold(0f32, |a, (g, w)| a.max((g - w).abs()));
        assert!(max_d < 1e-6, "timestep_sincos mismatch; max|Δ|={max_d}");
    }

    // -- synthetic weight-map builder (covers every key the base trunk's `from_weights` requires) ---

    struct Emit<'a> {
        map: HashMap<String, Tensor>,
        seed: u64,
        dev: &'a Device,
    }
    impl<'a> Emit<'a> {
        fn new(dev: &'a Device) -> Self {
            Self {
                map: HashMap::new(),
                seed: 1,
                dev,
            }
        }
        fn t(&mut self, shape: &[usize], key: String) {
            self.seed += 1;
            // scale down so a deep stack of matmuls stays in a sane range (finiteness smoke, not parity)
            let base = det(shape, self.seed, self.dev);
            self.map.insert(key, base.affine(0.2, 0.0).unwrap());
        }
        fn linear(&mut self, p: &str, out: usize, inn: usize, bias: bool) {
            self.t(&[out, inn], format!("{p}.weight"));
            if bias {
                self.t(&[out], format!("{p}.bias"));
            }
        }
        fn conv(&mut self, p: &str, o: usize, i_over_g: usize, k: usize, bias: bool) {
            self.t(&[o, i_over_g, k, k], format!("{p}.weight"));
            if bias {
                self.t(&[o], format!("{p}.bias"));
            }
        }
    }

    /// Build a synthetic (deterministic) weight map covering every key the BASE trunk requires for
    /// `cfg`, then load it through the real [`Weights`] path.
    fn synthetic_trunk_weights(cfg: &SanaTransformerConfig, dev: &Device) -> Weights {
        let inner = cfg.inner_dim() as usize;
        let cross_inner = (cfg.num_cross_attention_heads * cfg.cross_attention_head_dim) as usize;
        let p = cfg.patch_size as usize;
        let hidden = (cfg.mlp_ratio * inner as f32) as usize;
        let oc = cfg.out_channels as usize;
        let mut e = Emit::new(dev);

        e.conv("patch_embed.proj", inner, cfg.in_channels as usize, p, true);
        e.linear(
            "time_embed.emb.timestep_embedder.linear_1",
            inner,
            256,
            true,
        );
        e.linear(
            "time_embed.emb.timestep_embedder.linear_2",
            inner,
            inner,
            true,
        );
        e.linear("time_embed.linear", 6 * inner, inner, true);
        e.linear(
            "caption_projection.linear_1",
            inner,
            cfg.caption_channels as usize,
            true,
        );
        e.linear("caption_projection.linear_2", inner, inner, true);
        e.t(&[inner], "caption_norm.weight".to_string());
        e.t(&[2, inner], "scale_shift_table".to_string());
        e.linear("proj_out", p * p * oc, inner, true);

        for i in 0..cfg.num_layers {
            let bp = format!("transformer_blocks.{i}");
            e.t(&[6, inner], format!("{bp}.scale_shift_table"));
            // attn1 (linear self-attn): q/k/v no bias, to_out.0 bias.
            e.linear(&format!("{bp}.attn1.to_q"), inner, inner, false);
            e.linear(&format!("{bp}.attn1.to_k"), inner, inner, false);
            e.linear(&format!("{bp}.attn1.to_v"), inner, inner, false);
            e.linear(&format!("{bp}.attn1.to_out.0"), inner, inner, true);
            // attn2 (cross): all bias; KV projected from the caption (dim = inner).
            e.linear(&format!("{bp}.attn2.to_q"), cross_inner, inner, true);
            e.linear(&format!("{bp}.attn2.to_k"), cross_inner, inner, true);
            e.linear(&format!("{bp}.attn2.to_v"), cross_inner, inner, true);
            e.linear(&format!("{bp}.attn2.to_out.0"), inner, cross_inner, true);
            // ff (GLUMBConv Mix-FFN): conv_inverted 1×1 → 2·hidden, conv_depth 3×3 depthwise,
            // conv_point 1×1 hidden → inner (no bias).
            e.conv(
                &format!("{bp}.ff.conv_inverted"),
                2 * hidden,
                inner,
                1,
                true,
            );
            e.conv(&format!("{bp}.ff.conv_depth"), 2 * hidden, 1, 3, true);
            e.conv(&format!("{bp}.ff.conv_point"), inner, hidden, 1, false);
        }
        Weights::from_map(e.map)
    }

    fn small_cfg() -> SanaTransformerConfig {
        SanaTransformerConfig {
            in_channels: 4,
            out_channels: 4,
            num_attention_heads: 2,
            attention_head_dim: 8, // inner = 16
            num_layers: 2,
            num_cross_attention_heads: 2,
            cross_attention_head_dim: 8, // cross inner = 16
            caption_channels: 6,
            mlp_ratio: 2.5,
            patch_size: 1,
            norm_eps: 1e-6,
            caption_norm_eps: 1e-5,
            attn_qk_norm_eps: 1e-5,
            attn_eps: 1e-15,
            guidance_embeds: false,
            guidance_embeds_scale: 0.1,
            qk_norm: false,
        }
    }

    #[test]
    fn trunk_forward_shape_finite_cpu() {
        // A random-weight forward through every trunk primitive (patch embed → ReLU linear self-attn →
        // cross-attn → GLUMBConv Mix-FFN → adaLN-single → output modnorm → unpatchify): finite, right
        // shape, non-degenerate. The CPU analogue of the GPU SANA-res smoke. (Real numeric parity lives
        // in tests/transformer_parity.rs against the committed diffusers golden.)
        let dev = Device::Cpu;
        let cfg = small_cfg();
        let w = synthetic_trunk_weights(&cfg, &dev);
        let model = SanaTransformer::from_weights(&w, cfg.clone()).unwrap();

        let (h, wd) = (4usize, 4usize);
        let latent = det(&[1, cfg.in_channels as usize, h, wd], 101, &dev);
        let caption = det(&[1, 3, cfg.caption_channels as usize], 202, &dev);
        let timestep = Tensor::from_vec(vec![0.7f32], (1,), &dev).unwrap();

        let out = model.forward(&latent, &caption, &timestep).unwrap();
        assert_eq!(out.dims(), &[1, cfg.out_channels as usize, h, wd]);
        let flat = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(
            flat.iter().all(|v| v.is_finite()),
            "non-finite trunk forward"
        );
        let (lo, hi) = flat
            .iter()
            .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
        assert!(
            hi - lo > 1e-5,
            "trunk output is constant — graph degenerate: [{lo}, {hi}]"
        );
    }

    /// **GPU full-trunk forward at SANA-1.6B config and latent resolution** (32×32 latent = 1024px).
    /// Builds the FULL `sana_1600m` trunk (20 blocks, inner 2240 — real ~1.6B-param footprint) with
    /// synthetic weights on CUDA sm_120, forwards a `[1,32,32,32]` latent + `[1,300,2304]` caption,
    /// and asserts finite / correct shape / non-degenerate while reporting peak VRAM. cuda-gated;
    /// needs `--release` (and an idle GPU for a clean baseline).
    ///
    /// Run:
    ///   cargo test -p candle-gen-sana --lib --features cuda --release -- --ignored --nocapture \
    ///       gpu_trunk_forward_sana1600m
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "GPU SANA-1.6B full-trunk forward — run on CUDA sm_120 with --release"]
    fn gpu_trunk_forward_sana1600m() {
        use candle_gen::testkit::VramProbe;
        let dev = Device::new_cuda(0).expect("cuda device");
        let cfg = SanaTransformerConfig::sana_1600m();

        let mut probe = VramProbe::start(0);
        let load = probe.phase();
        let w = synthetic_trunk_weights(&cfg, &dev);
        let model = SanaTransformer::from_weights(&w, cfg.clone()).unwrap();
        probe.end_load(load);

        let latent = det(&[1, cfg.in_channels as usize, 32, 32], 7, &dev);
        let caption = det(&[1, 300, cfg.caption_channels as usize], 8, &dev);
        let timestep = Tensor::from_vec(vec![0.6f32], (1,), &dev).unwrap();

        let run = probe.phase();
        let out = model.forward(&latent, &caption, &timestep).unwrap();
        let _ = out.sum_all().unwrap().to_scalar::<f32>().unwrap(); // force eval
        probe.end_gen(run);

        assert_eq!(out.dims(), &[1, cfg.out_channels as usize, 32, 32]);
        let flat = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(
            flat.iter().all(|v| v.is_finite()),
            "non-finite SANA-res trunk forward"
        );
        let (lo, hi) = flat
            .iter()
            .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
        assert!(
            hi - lo > 1e-4,
            "degenerate SANA-res trunk output: [{lo}, {hi}]"
        );
        println!(
            "SANA-1.6B f32 32² trunk forward OK on CUDA: range=[{lo:.4}, {hi:.4}]  VRAM: {}",
            probe.report()
        );
    }
}
