//! FLUX.1-dev **Fun-Controlnet-Union** control branch (sc-8412) — the candle (Windows/CUDA) sibling of
//! `mlx-gen-flux`'s `control_transformer` (sc-8238). Port of the diffusers `FluxControlNetModel` as
//! shipped by `Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0`: a small partial copy of the base
//! FLUX.1 MMDiT (the checkpoint ships `num_layers = 6` double blocks and `num_single_layers = 0`) that
//! ingests the VAE-encoded control image (a pose skeleton / canny / depth map — **input-agnostic**, no
//! discrete mode index in 2.0) and emits one per-block residual, injected into the frozen base 19-layer
//! double stream at `interval = ceil(19 / 6) = 4` (see [`crate::ip_dit::IpFlux::forward_control`]).
//!
//! This follows the standard diffusers ControlNet shape: an **independent** mini-transformer with its
//! own `x_embedder` / `context_embedder` / `time_text_embed` and a zero-init `controlnet_x_embedder`
//! that adds the encoded control image into the image stream; each of its blocks' output is projected by
//! a zero-init `controlnet_blocks[i]` Linear into a residual. The N residuals are returned (pre-scale)
//! for the base transformer to add.
//!
//! ## Diffusers vs BFL key layout (why this is a self-contained block)
//! The candle base FLUX DiT ([`crate::ip_dit::IpFlux`] / the stock `candle-transformers` `Flux`) is the
//! **original BFL** key layout (`img_in`/`txt_in`, fused `img_attn.qkv`, single `img_mod`). The Shakker
//! control checkpoint is the **diffusers** layout (`x_embedder`/`context_embedder`, split `attn.to_q`/
//! `to_k`/`to_v` + `add_*_proj`, `norm1.linear` AdaLN-Zero, `ff.net.0.proj`). The two are
//! computationally equivalent but key-incompatible, so the control branch ports the diffusers
//! [`JointBlock`] here rather than reusing the BFL [`crate::ip_dit`] block — matching the mlx port
//! (where base + control happen to *share* the diffusers block because the mlx base is diffusers-layout
//! too). The control RoPE is the **same** FLUX RoPE the base uses ([`crate::ip_dit::EmbedNd`] over
//! `cat(txt_ids, img_ids)`), so the residuals align 1:1 with the base image tokens.
//!
//! Adapters (LoRA/LoKr) target the **base** transformer only — the control branch is never an adapter
//! target (mirrors the mlx / FLUX.2 / Z-Image / Qwen control ports).

use candle_core::{DType, Result, Tensor, D};
use candle_nn::{LayerNorm, Linear, Module, RmsNorm, VarBuilder};

use crate::ip_dit::{
    apply_rope, control_residual_interval, scaled_dot_product_attention, timestep_embedding,
    Config, DitImageInjector, EmbedNd, IpFlux,
};

/// The diffusers LayerNorm / RMS epsilons (FLUX defaults).
const LN_EPS: f64 = 1e-6;
const RMS_EPS: f64 = 1e-6;
/// FLUX dev pooled CLIP width + T5 context width + packed latent channels (the control branch's three
/// input projections). Pinned because the Shakker checkpoint is dev-only.
const POOLED_DIM: usize = 768;
const CONTEXT_DIM: usize = 4096;
const PACKED_CHANNELS: usize = 64;

/// The control branch's structural dims, derived from the base FLUX [`Config`] so the branch shares the
/// base's inner width / head layout (FLUX dev: hidden 3072, heads 24, head_dim 128, mlp 12288). Threaded
/// so the CI parity test can build a tiny base + a matching tiny branch.
#[derive(Clone, Copy, Debug)]
struct Dims {
    hidden: usize,
    heads: usize,
    head_dim: usize,
    mlp: usize,
}

impl Dims {
    fn from_config(cfg: &Config) -> Self {
        let hidden = cfg.hidden_size;
        let heads = cfg.num_heads;
        Self {
            hidden,
            heads,
            head_dim: hidden / heads,
            mlp: (hidden as f64 * cfg.mlp_ratio) as usize,
        }
    }
}

/// The Shakker `FLUX.1-dev-ControlNet-Union-Pro-2.0` config: a 6-block partial copy of the base 19-layer
/// FLUX.1 MMDiT, identical inner dims, `num_single_layers = 0`, guidance-embedded (FLUX.1-dev). The 2.0
/// checkpoint has NO `num_mode` / condition-type embedding (input-agnostic — the kind is which
/// preprocessed image is fed, not a forward branch).
#[derive(Clone, Copy, Debug)]
pub struct FluxControlNetConfig {
    /// Number of control double blocks shipped in the checkpoint (Shakker 2.0 = 6).
    pub num_layers: usize,
    /// FLUX.1-dev carries an embedded guidance scalar (the control branch mirrors the base).
    pub supports_guidance: bool,
}

impl FluxControlNetConfig {
    /// The shipped Shakker Union-Pro-2.0: `num_layers = 6`, guidance-embedded (dev).
    pub fn shakker_union_pro_2_0() -> Self {
        Self {
            num_layers: 6,
            supports_guidance: true,
        }
    }
}

/// `LayerNorm` with no affine params (elementwise-affine = False, the diffusers AdaLN-Zero base norm).
fn layer_norm_no_affine(
    dim: usize,
    dtype: DType,
    device: &candle_core::Device,
) -> Result<LayerNorm> {
    let ws = Tensor::ones(dim, dtype, device)?;
    Ok(LayerNorm::new_no_bias(ws, LN_EPS))
}

/// `silu(emb) @ linear` → split into `chunks` modulation params, the diffusers `AdaLayerNormZero`. The
/// base LayerNorm is parameter-free; the shift/scale come from `emb`.
struct AdaLayerNormZero {
    linear: Linear,
    norm: LayerNorm,
    chunks: usize,
}

impl AdaLayerNormZero {
    fn new(chunks: usize, d: Dims, vb: VarBuilder) -> Result<Self> {
        let linear = candle_nn::linear(d.hidden, chunks * d.hidden, vb.pp("linear"))?;
        let norm = layer_norm_no_affine(d.hidden, vb.dtype(), vb.device())?;
        Ok(Self {
            linear,
            norm,
            chunks,
        })
    }

    /// 6-chunk variant (double-block image/text stream): returns
    /// `(normed, gate_msa, shift_mlp, scale_mlp, gate_mlp)`.
    fn forward_six(
        &self,
        hidden: &Tensor,
        emb: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        debug_assert_eq!(self.chunks, 6);
        let p = emb.silu()?.apply(&self.linear)?.chunk(6, D::Minus1)?;
        // shift = p[0], scale = p[1]; normed = norm(hidden) * (1 + scale) + shift.
        let normed = self.norm.forward(hidden)?;
        let normed = normed
            .broadcast_mul(&(p[1].unsqueeze(1)? + 1.0)?)?
            .broadcast_add(&p[0].unsqueeze(1)?)?;
        Ok((
            normed,
            p[2].clone(),
            p[3].clone(),
            p[4].clone(),
            p[5].clone(),
        ))
    }
}

/// The diffusers joint (double-block) attention: split `to_q`/`to_k`/`to_v` + `to_out.0` for the image
/// stream, `add_{q,k,v}_proj` + `to_add_out` for the text stream, RMS-norm on q/k (`norm_q`/`norm_k`,
/// `norm_added_q`/`norm_added_k`), joint RoPE over `cat(txt, img)`.
struct JointAttention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    add_q: Linear,
    add_k: Linear,
    add_v: Linear,
    to_add_out: Linear,
    norm_q: RmsNorm,
    norm_k: RmsNorm,
    norm_added_q: RmsNorm,
    norm_added_k: RmsNorm,
    heads: usize,
    head_dim: usize,
}

impl JointAttention {
    fn new(d: Dims, vb: VarBuilder) -> Result<Self> {
        let lin = |n: &str, vb: &VarBuilder| candle_nn::linear(d.hidden, d.hidden, vb.pp(n));
        let rms = |n: &str, vb: &VarBuilder| -> Result<RmsNorm> {
            Ok(RmsNorm::new(vb.get(d.head_dim, n)?, RMS_EPS))
        };
        Ok(Self {
            to_q: lin("to_q", &vb)?,
            to_k: lin("to_k", &vb)?,
            to_v: lin("to_v", &vb)?,
            to_out: candle_nn::linear(d.hidden, d.hidden, vb.pp("to_out").pp("0"))?,
            add_q: lin("add_q_proj", &vb)?,
            add_k: lin("add_k_proj", &vb)?,
            add_v: lin("add_v_proj", &vb)?,
            to_add_out: lin("to_add_out", &vb)?,
            norm_q: rms("norm_q.weight", &vb)?,
            norm_k: rms("norm_k.weight", &vb)?,
            norm_added_q: rms("norm_added_q.weight", &vb)?,
            norm_added_k: rms("norm_added_k.weight", &vb)?,
            heads: d.heads,
            head_dim: d.head_dim,
        })
    }

    /// Project `x` through `(q, k, v)`, reshape to `[B, heads, seq, head_dim]`, RMS-norm q/k. Returns the
    /// per-head q/k/v (pre-RoPE).
    fn qkv(
        &self,
        x: &Tensor,
        q: &Linear,
        k: &Linear,
        v: &Linear,
        nq: &RmsNorm,
        nk: &RmsNorm,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (b, s, _) = x.dims3()?;
        let (heads, head_dim) = (self.heads, self.head_dim);
        let to_heads =
            |t: Tensor| -> Result<Tensor> { t.reshape((b, s, heads, head_dim))?.transpose(1, 2) };
        let q = to_heads(x.apply(q)?)?.apply(nq)?;
        let k = to_heads(x.apply(k)?)?.apply(nk)?;
        let v = to_heads(x.apply(v)?)?;
        Ok((q, k, v))
    }

    /// Joint attention over `cat(txt, img)` with the shared FLUX RoPE `pe`. Returns
    /// `(attn_img, attn_txt)` — the image and text attention outputs (post `to_out`/`to_add_out`).
    fn forward(&self, hidden: &Tensor, encoder: &Tensor, pe: &Tensor) -> Result<(Tensor, Tensor)> {
        let (q, k, v) = self.qkv(
            hidden,
            &self.to_q,
            &self.to_k,
            &self.to_v,
            &self.norm_q,
            &self.norm_k,
        )?;
        let (eq, ek, ev) = self.qkv(
            encoder,
            &self.add_q,
            &self.add_k,
            &self.add_v,
            &self.norm_added_q,
            &self.norm_added_k,
        )?;
        // Concatenate text + image on the sequence axis (diffusers order: text first).
        let q = Tensor::cat(&[&eq, &q], 2)?;
        let k = Tensor::cat(&[&ek, &k], 2)?;
        let v = Tensor::cat(&[&ev, &v], 2)?;
        // RoPE the joint q/k, then SDPA, then split back into text/image.
        let q = apply_rope(&q, pe)?.contiguous()?;
        let k = apply_rope(&k, pe)?.contiguous()?;
        let out = scaled_dot_product_attention(&q, &k, &v)?; // [B, heads, seq, head_dim]
        let out = out.transpose(1, 2)?.flatten_from(2)?; // [B, seq, hidden]
        let txt_seq = encoder.dim(1)?;
        let img_seq = hidden.dim(1)?;
        let attn_txt = out.narrow(1, 0, txt_seq)?.apply(&self.to_add_out)?;
        let attn_img = out.narrow(1, txt_seq, img_seq)?.apply(&self.to_out)?;
        Ok((attn_img, attn_txt))
    }
}

/// diffusers `FeedForward` (`net.0.proj` → activation → `net.2`). The image stream uses exact GELU
/// (`gelu_erf`), the context stream the tanh approximation (`gelu`) — mirroring the mlx port.
struct FeedForward {
    lin1: Linear,
    lin2: Linear,
    approx: bool,
}

impl FeedForward {
    fn new(approx: bool, d: Dims, vb: VarBuilder) -> Result<Self> {
        let lin1 = candle_nn::linear(d.hidden, d.mlp, vb.pp("net").pp("0").pp("proj"))?;
        let lin2 = candle_nn::linear(d.mlp, d.hidden, vb.pp("net").pp("2"))?;
        Ok(Self { lin1, lin2, approx })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.apply(&self.lin1)?;
        let x = if self.approx {
            x.gelu()?
        } else {
            x.gelu_erf()?
        };
        x.apply(&self.lin2)
    }
}

/// One diffusers FLUX joint (double-stream) transformer block — the `transformer_blocks.{i}` the
/// control checkpoint ships. AdaLN-Zero modulation on both streams, joint attention, gated FF.
struct JointBlock {
    norm1: AdaLayerNormZero,
    norm1_context: AdaLayerNormZero,
    attn: JointAttention,
    ff: FeedForward,
    ff_context: FeedForward,
    /// The diffusers `norm2` — a parameter-free LayerNorm shared by both streams (same hidden dim).
    /// Built once at load so the hot per-step FF path reuses it instead of re-allocating the device
    /// `ones` weight every invocation (sc-9039).
    norm2: LayerNorm,
}

impl JointBlock {
    fn new(d: Dims, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm1: AdaLayerNormZero::new(6, d, vb.pp("norm1"))?,
            norm1_context: AdaLayerNormZero::new(6, d, vb.pp("norm1_context"))?,
            attn: JointAttention::new(d, vb.pp("attn"))?,
            ff: FeedForward::new(false, d, vb.pp("ff"))?,
            ff_context: FeedForward::new(true, d, vb.pp("ff_context"))?,
            norm2: layer_norm_no_affine(d.hidden, vb.dtype(), vb.device())?,
        })
    }

    /// `(encoder, hidden)` → `(encoder', hidden')` (matching the mlx return order).
    fn forward(
        &self,
        hidden: &Tensor,
        encoder: &Tensor,
        emb: &Tensor,
        pe: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (norm_hidden, gate_msa, shift_mlp, scale_mlp, gate_mlp) =
            self.norm1.forward_six(hidden, emb)?;
        let (norm_encoder, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp) =
            self.norm1_context.forward_six(encoder, emb)?;
        let (attn_img, attn_txt) = self.attn.forward(&norm_hidden, &norm_encoder, pe)?;
        let hidden = apply_norm_ff(
            hidden,
            &attn_img,
            &gate_msa,
            &shift_mlp,
            &scale_mlp,
            &gate_mlp,
            &self.ff,
            &self.norm2,
        )?;
        let encoder = apply_norm_ff(
            encoder,
            &attn_txt,
            &c_gate_msa,
            &c_shift_mlp,
            &c_scale_mlp,
            &c_gate_mlp,
            &self.ff_context,
            &self.norm2,
        )?;
        Ok((encoder, hidden))
    }
}

/// The AdaLN-Zero post-attention residual + gated FF, shared by the image/text streams:
/// `h = h + gate_msa·attn; h = h + gate_mlp·ff(norm(h)·(1+scale_mlp)+shift_mlp)`.
#[allow(clippy::too_many_arguments)]
fn apply_norm_ff(
    hidden: &Tensor,
    attn: &Tensor,
    gate_msa: &Tensor,
    shift_mlp: &Tensor,
    scale_mlp: &Tensor,
    gate_mlp: &Tensor,
    ff: &FeedForward,
    norm: &LayerNorm,
) -> Result<Tensor> {
    let hidden = (hidden + attn.broadcast_mul(&gate_msa.unsqueeze(1)?)?)?;
    // The diffusers `norm2` (parameter-free, elementwise-affine = False), built once at load and
    // reused here rather than re-allocated each call (sc-9039).
    let normed = norm
        .forward(&hidden)?
        .broadcast_mul(&(scale_mlp.unsqueeze(1)? + 1.0)?)?
        .broadcast_add(&shift_mlp.unsqueeze(1)?)?;
    let ff_out = ff.forward(&normed)?;
    hidden.broadcast_add(&ff_out.broadcast_mul(&gate_mlp.unsqueeze(1)?)?)
}

/// `silu → linear_1 → silu → linear_2` MLP timestep/guidance/text embedder (diffusers
/// `TimestepEmbedding` / `PixArtAlphaTextProjection`-style two-layer MLP).
struct MlpEmbedder {
    lin1: Linear,
    lin2: Linear,
}

impl MlpEmbedder {
    fn new(in_dim: usize, hidden: usize, vb: VarBuilder) -> Result<Self> {
        let lin1 = candle_nn::linear(in_dim, hidden, vb.pp("linear_1"))?;
        let lin2 = candle_nn::linear(hidden, hidden, vb.pp("linear_2"))?;
        Ok(Self { lin1, lin2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.apply(&self.lin1)?.silu()?.apply(&self.lin2)
    }
}

/// diffusers `CombinedTimestepGuidanceTextEmbeddings`: sinusoidal time + guidance projections summed
/// with the pooled-text projection. `time_text_embed.{timestep,guidance,text}_embedder`.
struct TimeTextEmbed {
    timestep: MlpEmbedder,
    text: MlpEmbedder,
    guidance: Option<MlpEmbedder>,
}

impl TimeTextEmbed {
    fn new(
        supports_guidance: bool,
        hidden: usize,
        pooled_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            timestep: MlpEmbedder::new(256, hidden, vb.pp("timestep_embedder"))?,
            text: MlpEmbedder::new(pooled_dim, hidden, vb.pp("text_embedder"))?,
            guidance: if supports_guidance {
                Some(MlpEmbedder::new(256, hidden, vb.pp("guidance_embedder"))?)
            } else {
                None
            },
        })
    }

    /// `timestep` and `guidance` are the raw scalars (already ×1 — the sinusoid embedder internally
    /// scales by 1000 via [`timestep_embedding`]). `pooled` is the CLIP pooled `[B, 768]`.
    fn forward(
        &self,
        timestep: &Tensor,
        guidance: Option<&Tensor>,
        pooled: &Tensor,
    ) -> Result<Tensor> {
        let dtype = pooled.dtype();
        let mut out = self
            .timestep
            .forward(&timestep_embedding(timestep, 256, dtype)?)?;
        if let (Some(g_in), Some(g)) = (self.guidance.as_ref(), guidance) {
            out = (out + g_in.forward(&timestep_embedding(g, 256, dtype)?)?)?;
        }
        out = (out + self.text.forward(pooled)?)?;
        Ok(out)
    }
}

/// The FLUX.1 ControlNet control transformer (the trainable branch). Holds its own input projections +
/// N double blocks + N zero-init residual projections; emits the per-block residuals for the base
/// transformer ([`IpFlux::forward_control`]).
pub struct FluxControlNet {
    x_embedder: Linear,
    context_embedder: Linear,
    time_text_embed: TimeTextEmbed,
    /// Zero-init projection of the packed control latent (`64 → hidden`), added to `x_embedder(x)`.
    controlnet_x_embedder: Linear,
    blocks: Vec<JointBlock>,
    /// Zero-init per-block residual projections (`hidden → hidden`).
    controlnet_blocks: Vec<Linear>,
    pe_embedder: EmbedNd,
}

impl FluxControlNet {
    /// Load from the Shakker Union-Pro-2.0 checkpoint (standard diffusers layout — un-prefixed keys for
    /// the real single-file `diffusion_pytorch_model.safetensors`). `base_cfg` is the base FLUX [`Config`]
    /// the branch must share dims with (hidden / heads / RoPE axes / theta); `cfg` pins `num_layers` (= 6).
    pub fn new(base_cfg: &Config, cfg: &FluxControlNetConfig, vb: VarBuilder) -> Result<Self> {
        let d = Dims::from_config(base_cfg);
        let (pooled_dim, context_dim, packed) = (POOLED_DIM, CONTEXT_DIM, PACKED_CHANNELS);
        let x_embedder = candle_nn::linear(packed, d.hidden, vb.pp("x_embedder"))?;
        let context_embedder = candle_nn::linear(context_dim, d.hidden, vb.pp("context_embedder"))?;
        let time_text_embed = TimeTextEmbed::new(
            cfg.supports_guidance,
            d.hidden,
            pooled_dim,
            vb.pp("time_text_embed"),
        )?;
        let controlnet_x_embedder =
            candle_nn::linear(packed, d.hidden, vb.pp("controlnet_x_embedder"))?;
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        let mut controlnet_blocks = Vec::with_capacity(cfg.num_layers);
        let vb_b = vb.pp("transformer_blocks");
        let vb_cb = vb.pp("controlnet_blocks");
        for i in 0..cfg.num_layers {
            blocks.push(JointBlock::new(d, vb_b.pp(i))?);
            controlnet_blocks.push(candle_nn::linear(d.hidden, d.hidden, vb_cb.pp(i))?);
        }
        // The same FLUX RoPE the base uses (axes_dim + theta from the base config), so the control
        // residuals share the base image-token positions.
        let pe_embedder = EmbedNd::new(d.head_dim, base_cfg.theta, base_cfg.axes_dim.clone());
        Ok(Self {
            x_embedder,
            context_embedder,
            time_text_embed,
            controlnet_x_embedder,
            blocks,
            controlnet_blocks,
            pe_embedder,
        })
    }

    /// Number of control residuals (= control layers); drives the base injection interval.
    pub fn num_residuals(&self) -> usize {
        self.controlnet_blocks.len()
    }

    /// Run the control branch → the per-block residuals (pre-scale), one per control layer.
    ///
    /// `hidden_states`: the current packed **noise** latents `[B, img_seq, 64]` (the controlnet sees the
    /// same latents the base does this step). `control_cond`: the packed VAE-encoded control image
    /// `[B, img_seq, 64]` (constant across steps). `prompt_embeds`/`pooled`: the same text features the
    /// base forward uses. `timesteps`/`guidance`: the scheduler timestep + embedded guidance scalar.
    /// `img_ids`/`txt_ids`: the shared FLUX position ids (so the control RoPE == the base RoPE).
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        control_cond: &Tensor,
        prompt_embeds: &Tensor,
        pooled: &Tensor,
        img_ids: &Tensor,
        txt_ids: &Tensor,
        timesteps: &Tensor,
        guidance: Option<&Tensor>,
    ) -> Result<Vec<Tensor>> {
        // `x_embedder(x) + controlnet_x_embedder(control_cond)` (diffusers `hidden_states =
        // hidden_states + self.controlnet_x_embedder(controlnet_cond)`). The zero-init
        // `controlnet_x_embedder` means an untrained / scale-0 branch starts as a no-op.
        let mut hidden = (hidden_states.apply(&self.x_embedder)?
            + control_cond.apply(&self.controlnet_x_embedder)?)?;
        let mut encoder = prompt_embeds.apply(&self.context_embedder)?;
        let emb = self.time_text_embed.forward(timesteps, guidance, pooled)?;
        let pe = {
            let ids = Tensor::cat(&[txt_ids, img_ids], 1)?;
            ids.apply(&self.pe_embedder)?
        };

        let mut residuals = Vec::with_capacity(self.blocks.len());
        for (block, cn) in self.blocks.iter().zip(&self.controlnet_blocks) {
            let (e, h) = block.forward(&hidden, &encoder, &emb, &pe)?;
            encoder = e;
            hidden = h;
            // residual[i] = controlnet_blocks[i](hidden_after_block_i) (diffusers zero-init proj).
            residuals.push(hidden.apply(cn)?);
        }
        Ok(residuals)
    }
}

/// The FLUX.1-dev base MMDiT + its Fun-Controlnet-Union control branch (sc-8412). Composes the
/// parity-proven [`IpFlux`] base with a [`FluxControlNet`]; [`forward`](Self::forward) computes the
/// control residuals once and threads them (+ an optional identity injector — compose-ready) into the
/// base double stream.
pub struct FluxControlTransformer {
    base: IpFlux,
    branch: FluxControlNet,
}

impl FluxControlTransformer {
    pub fn new(base: IpFlux, branch: FluxControlNet) -> Self {
        Self { base, branch }
    }

    /// Read-only access to the base DiT.
    pub fn base(&self) -> &IpFlux {
        &self.base
    }

    /// Number of control double blocks (residuals); `interval = ceil(num_double / num_residuals)`.
    pub fn num_residuals(&self) -> usize {
        self.branch.num_residuals()
    }

    /// The injection interval over the base double blocks for this branch (`ceil(19/6) = 4`).
    pub fn residual_interval(&self) -> usize {
        control_residual_interval(self.base.num_double_blocks(), self.branch.num_residuals())
    }

    /// Control forward (no identity injector): the convenience entry the generator wires.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        img: &Tensor,
        img_ids: &Tensor,
        txt: &Tensor,
        txt_ids: &Tensor,
        timesteps: &Tensor,
        pooled: &Tensor,
        guidance: Option<&Tensor>,
        control_cond: &Tensor,
        control_scale: f64,
    ) -> Result<Tensor> {
        self.forward_composed(
            img,
            img_ids,
            txt,
            txt_ids,
            timesteps,
            pooled,
            guidance,
            control_cond,
            control_scale,
            None,
        )
    }

    /// Control forward THAT ALSO threads an optional identity injector (PuLID / XLabs IP-Adapter) — the
    /// **compose-ready** entry. The control residuals are computed once from the branch, then injected
    /// into the base double stream alongside the injector seam in [`IpFlux::forward_control`]. With
    /// `injector = None` this is the plain control forward.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_composed(
        &self,
        img: &Tensor,
        img_ids: &Tensor,
        txt: &Tensor,
        txt_ids: &Tensor,
        timesteps: &Tensor,
        pooled: &Tensor,
        guidance: Option<&Tensor>,
        control_cond: &Tensor,
        control_scale: f64,
        injector: Option<&dyn DitImageInjector>,
    ) -> Result<Tensor> {
        let residuals = self.branch.forward(
            img,
            control_cond,
            txt,
            pooled,
            img_ids,
            txt_ids,
            timesteps,
            guidance,
        )?;
        self.base.forward_control(
            img,
            img_ids,
            txt,
            txt_ids,
            timesteps,
            pooled,
            guidance,
            injector,
            Some((&residuals, control_scale)),
        )
    }
}

/// The control kinds the Shakker Union-Pro-2.0 checkpoint admits: **pose / canny / depth**
/// (input-agnostic — the kind is which preprocessed image is fed, not a forward branch). A unit-testable
/// free function so the policy is testable without a loaded generator.
pub fn accepts_control_kind(kind: &str) -> bool {
    matches!(kind, "pose" | "canny" | "depth")
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// A tiny base FLUX [`Config`] for the GPU-free control tests: inner 16 (heads 2 · head_dim 8),
    /// `axes_dim` summing to 8, with the real fixed input widths (in 64 / context 4096 / vec 768) the
    /// control branch's three input projections require.
    fn tiny_base_cfg() -> Config {
        Config {
            in_channels: 64,
            vec_in_dim: 768,
            context_in_dim: 4096,
            hidden_size: 16,
            mlp_ratio: 2.0,
            num_heads: 2,
            depth: 1,
            depth_single_blocks: 1,
            axes_dim: vec![2, 2, 4],
            theta: 10_000,
            qkv_bias: true,
            guidance_embed: true,
        }
    }

    #[test]
    fn shakker_config_is_6_layers_guidance() {
        let cfg = FluxControlNetConfig::shakker_union_pro_2_0();
        assert_eq!(cfg.num_layers, 6);
        assert!(cfg.supports_guidance);
        // The injection interval over FLUX.1's 19 base double blocks: ceil(19/6) = 4.
        assert_eq!(control_residual_interval(19, cfg.num_layers), 4);
    }

    #[test]
    fn injection_interval_math_matches_diffusers_ceil() {
        // ceil(num_double / num_residuals): 19/6 = 4, plus the degenerate-count guards.
        assert_eq!(control_residual_interval(19, 6), 4);
        assert_eq!(control_residual_interval(19, 1), 19);
        assert_eq!(control_residual_interval(19, 19), 1);
        assert_eq!(control_residual_interval(19, 0), 19); // clamped to 1 divisor
                                                          // The diffusers index mapping `i / interval` keeps every base block in range of the 6 residuals.
        let interval = control_residual_interval(19, 6);
        for i in 0..19 {
            assert!((i / interval).min(5) < 6);
        }
    }

    #[test]
    fn accepted_control_kinds_are_pose_canny_depth() {
        assert!(accepts_control_kind("pose"));
        assert!(accepts_control_kind("canny"));
        assert!(accepts_control_kind("depth"));
        assert!(!accepts_control_kind("scribble"));
        assert!(!accepts_control_kind("normal"));
    }

    /// A tiny synthetic control branch (2 layers) forwards to residual shapes `[B, img_seq, hidden]`,
    /// one per layer, and zero-init `controlnet_blocks` + zero-init `controlnet_x_embedder` make the
    /// untrained branch emit all-zero residuals (scale-0-equivalent ⇒ base unchanged). GPU-free.
    #[test]
    fn control_branch_zero_init_residuals_are_zero() -> Result<()> {
        use candle_nn::VarMap;
        let dev = Device::Cpu;
        let dtype = DType::F32;
        let vm = VarMap::new();
        // Build the branch with all params at their loaded values, then zero the two zero-init layers
        // (controlnet_x_embedder + controlnet_blocks.*) — exactly the diffusers init that makes an
        // untrained branch a no-op. The rest are random (VarMap default init), proving the zero comes
        // from the zero-init projections, not from a dead forward.
        let base_cfg = tiny_base_cfg();
        let cfg = FluxControlNetConfig {
            num_layers: 2,
            supports_guidance: true,
        };
        let vb = VarBuilder::from_varmap(&vm, dtype, &dev);
        let _branch = FluxControlNet::new(&base_cfg, &cfg, vb)?; // proves load wiring
                                                                 // Zero the zero-init layers (controlnet_x_embedder + controlnet_blocks.*) — exactly the diffusers
                                                                 // init that makes an untrained branch a no-op.
        {
            let mut data = vm.data().lock().unwrap();
            for (name, var) in data.iter_mut() {
                if name.starts_with("controlnet_x_embedder")
                    || name.starts_with("controlnet_blocks")
                {
                    var.set(&Tensor::zeros(var.shape(), dtype, &dev)?)?;
                }
            }
        }
        // Rebuild from the (now partially-zeroed) varmap.
        let vb2 = VarBuilder::from_varmap(&vm, dtype, &dev);
        let branch2 = FluxControlNet::new(&base_cfg, &cfg, vb2)?;

        let (b, img_seq, txt_seq) = (1usize, 4usize, 3usize);
        let hidden = Tensor::randn(0f32, 1f32, (b, img_seq, 64), &dev)?;
        let control = Tensor::randn(0f32, 1f32, (b, img_seq, 64), &dev)?;
        let txt = Tensor::randn(0f32, 1f32, (b, txt_seq, 4096), &dev)?;
        let pooled = Tensor::randn(0f32, 1f32, (b, 768), &dev)?;
        // 2x2 latent grid → img_seq = 4; ids are [seq, 3].
        let img_ids = Tensor::zeros((b, img_seq, 3), dtype, &dev)?;
        let txt_ids = Tensor::zeros((b, txt_seq, 3), dtype, &dev)?;
        let ts = Tensor::full(0.5f32, b, &dev)?;
        let g = Tensor::full(3.5f32, b, &dev)?;

        let residuals = branch2.forward(
            &hidden,
            &control,
            &txt,
            &pooled,
            &img_ids,
            &txt_ids,
            &ts,
            Some(&g),
        )?;
        assert_eq!(residuals.len(), 2, "one residual per control layer");
        for r in &residuals {
            assert_eq!(r.dims(), &[b, img_seq, base_cfg.hidden_size]);
            let max = r.abs()?.max_all()?.to_scalar::<f32>()?;
            assert!(
                max == 0.0,
                "zero-init controlnet_blocks ⇒ all-zero residual, got max {max}"
            );
        }
        Ok(())
    }

    /// sc-9039: hoisting the parameter-free `norm2` out of the per-step FF hot loop must be
    /// bit-identical to constructing it fresh each call. A no-affine LayerNorm is deterministic
    /// given (dim, eps), so a norm built once at load equals one built per invocation, byte-for-byte.
    #[test]
    fn hoisted_norm2_is_bit_identical_to_fresh() -> Result<()> {
        let dev = Device::Cpu;
        let dtype = DType::F32;
        let dim = 16usize;
        let x = Tensor::randn(0f32, 1f32, (1, 4, dim), &dev)?;

        // The hoisted norm: built once (as `JointBlock::norm2` is at load).
        let hoisted = layer_norm_no_affine(dim, dtype, &dev)?;
        let out_hoisted = hoisted.forward(&x)?;

        // The old behaviour: a fresh norm constructed inside the call.
        let fresh = layer_norm_no_affine(dim, dtype, &dev)?;
        let out_fresh = fresh.forward(&x)?;

        // Byte-for-byte equal (no tolerance): identical eps + unit weight ⇒ identical output.
        let diff = (out_hoisted - out_fresh)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert_eq!(
            diff, 0.0,
            "hoisted no-affine LayerNorm must match a fresh one exactly"
        );
        Ok(())
    }
}
