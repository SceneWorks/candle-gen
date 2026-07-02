//! Vendored **diffusers-layout** FLUX.1 MMDiT (`FluxTransformer2DModel`) with the shared packed-load
//! seam (sc-9407) — the candle twin of the flux2-dev vendored transformer (sc-9087) and the z-image
//! `packed_dit` (sc-9408).
//!
//! **Why a vendored diffusers DiT.** The stock txt2img path ([`crate::pipeline`]) uses
//! `candle-transformers`'s **BFL-layout** `flux::model::Flux` (`double_blocks`/`single_blocks`, fused
//! `img_attn.qkv`). The pre-quantized MLX tier (`SceneWorks/flux1-schnell-mlx`, epic 8506) ships the
//! **diffusers** `FluxTransformer2DModel` layout instead (`transformer_blocks`/`single_transformer_blocks`,
//! split `attn.to_q`/`to_k`/`to_v`, `norm1.linear` AdaLN-Zero, `ff.net.0.proj`) — the exact layout the
//! FLUX control branch already ports ([`crate::control`]). So the packed path vendors a minimal
//! diffusers DiT here, building every `Linear` through the packed-detecting [`crate::quant::QLinear`] so
//! q4/q8 load straight from the packed parts (no dense staging), while a dense diffusers snapshot (bf16,
//! no `.scales`) loads through the same code unchanged. Numerically this is the diffusers formulation of
//! the same FLUX.1 forward the stock BFL model computes (both consume the same VAE-packed latents / T5
//! context / CLIP pooled vector and emit the same velocity); the CI parity test pins a dense build of
//! this DiT against the stock `candle-transformers` `Flux` at 1e-4 on shared weights.
//!
//! The RoPE / SDPA / timestep-embedding helpers are **reused** from [`crate::ip_dit`] (the same FLUX
//! RoPE the BFL model and the control branch use), so this file adds no numerics of its own beyond the
//! diffusers block wiring. The diffusers double block is structurally the [`crate::control::JointBlock`]
//! (AdaLN-Zero modulation, split-projection joint attention, gated FF); the single block + the AdaLN
//! output head are added here.

use candle_gen::candle_core::{DType, Result, Tensor, D};
use candle_gen::candle_nn::{LayerNorm, Module, RmsNorm, VarBuilder};

use crate::ip_dit::{
    apply_rope, scaled_dot_product_attention, timestep_embedding, Config, EmbedNd,
};
use crate::quant::QLinear;

/// diffusers FLUX LayerNorm / RMS epsilon.
const LN_EPS: f64 = 1e-6;
const RMS_EPS: f64 = 1e-6;
/// Pooled CLIP width / T5 context width / packed latent channels (the DiT's three input widths).
const POOLED_DIM: usize = 768;
const CONTEXT_DIM: usize = 4096;

/// The DiT structural dims derived from the shared FLUX [`Config`]: hidden 3072, heads 24, head_dim 128,
/// mlp 12288 (`mlp_ratio` 4.0). Shared with the BFL model via the reused config, so this fork cannot
/// drift on the FLUX hyperparameters.
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

/// A parameter-free `LayerNorm` (elementwise-affine = False) — the diffusers AdaLN base norm.
fn layer_norm_no_affine(
    dim: usize,
    dtype: DType,
    device: &candle_gen::candle_core::Device,
) -> Result<LayerNorm> {
    let ws = Tensor::ones(dim, dtype, device)?;
    Ok(LayerNorm::new_no_bias(ws, LN_EPS))
}

/// `silu(emb) @ linear` → split into `chunks` modulation params — the diffusers `AdaLayerNormZero`
/// (6-chunk, double block) / `AdaLayerNormZeroSingle` (3-chunk, single block). The base LayerNorm is
/// parameter-free; the shift/scale/gate come from `emb`.
struct AdaLayerNormZero {
    linear: QLinear,
    norm: LayerNorm,
}

impl AdaLayerNormZero {
    fn new(chunks: usize, d: Dims, vb: &VarBuilder, prefix: &str) -> Result<Self> {
        let linear = QLinear::linear_detect(
            d.hidden,
            chunks * d.hidden,
            vb,
            &format!("{prefix}.linear"),
            true,
        )?;
        Ok(Self {
            linear,
            norm: layer_norm_no_affine(d.hidden, vb.dtype(), vb.device())?,
        })
    }

    /// 6-chunk (double block): `(normed, gate_msa, shift_mlp, scale_mlp, gate_mlp)`.
    fn forward_six(
        &self,
        hidden: &Tensor,
        emb: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let p = self.linear.forward(&emb.silu()?)?.chunk(6, D::Minus1)?;
        let normed = self
            .norm
            .forward(hidden)?
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

    /// 3-chunk (single block): `(normed, gate)`.
    fn forward_three(&self, hidden: &Tensor, emb: &Tensor) -> Result<(Tensor, Tensor)> {
        let p = self.linear.forward(&emb.silu()?)?.chunk(3, D::Minus1)?;
        let normed = self
            .norm
            .forward(hidden)?
            .broadcast_mul(&(p[1].unsqueeze(1)? + 1.0)?)?
            .broadcast_add(&p[0].unsqueeze(1)?)?;
        Ok((normed, p[2].clone()))
    }
}

/// The diffusers joint (double-block) attention: split `to_q`/`to_k`/`to_v` + `to_out.0` for the image
/// stream, `add_{q,k,v}_proj` + `to_add_out` for the text stream, RMS-norm on q/k, joint RoPE over
/// `cat(txt, img)`. Structurally identical to [`crate::control::JointBlock`]'s attention but built with
/// the packed-detecting [`QLinear`].
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
    fn new(d: Dims, vb: &VarBuilder) -> Result<Self> {
        let lin = |n: &str| QLinear::linear_detect(d.hidden, d.hidden, vb, n, true);
        let rms =
            |n: &str| -> Result<RmsNorm> { Ok(RmsNorm::new(vb.get(d.head_dim, n)?, RMS_EPS)) };
        Ok(Self {
            to_q: lin("to_q")?,
            to_k: lin("to_k")?,
            to_v: lin("to_v")?,
            // `to_out.0`: the packed `.scales`/`.biases` siblings sit under the full `to_out.0` prefix,
            // so pass that whole base to `linear_detect` (never `.pp("0")` past the sibling).
            to_out: QLinear::linear_detect(d.hidden, d.hidden, vb, "to_out.0", true)?,
            add_q: lin("add_q_proj")?,
            add_k: lin("add_k_proj")?,
            add_v: lin("add_v_proj")?,
            to_add_out: lin("to_add_out")?,
            norm_q: rms("norm_q.weight")?,
            norm_k: rms("norm_k.weight")?,
            norm_added_q: rms("norm_added_q.weight")?,
            norm_added_k: rms("norm_added_k.weight")?,
            heads: d.heads,
            head_dim: d.head_dim,
        })
    }

    fn qkv(
        &self,
        x: &Tensor,
        q: &QLinear,
        k: &QLinear,
        v: &QLinear,
        nq: &RmsNorm,
        nk: &RmsNorm,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (b, s, _) = x.dims3()?;
        let (heads, head_dim) = (self.heads, self.head_dim);
        let to_heads =
            |t: Tensor| -> Result<Tensor> { t.reshape((b, s, heads, head_dim))?.transpose(1, 2) };
        let q = to_heads(q.forward(x)?)?.apply(nq)?;
        let k = to_heads(k.forward(x)?)?.apply(nk)?;
        let v = to_heads(v.forward(x)?)?;
        Ok((q, k, v))
    }

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
        let q = Tensor::cat(&[&eq, &q], 2)?;
        let k = Tensor::cat(&[&ek, &k], 2)?;
        let v = Tensor::cat(&[&ev, &v], 2)?;
        let q = apply_rope(&q, pe)?.contiguous()?;
        let k = apply_rope(&k, pe)?.contiguous()?;
        let out = scaled_dot_product_attention(&q, &k, &v)?;
        let out = out.transpose(1, 2)?.flatten_from(2)?;
        let txt_seq = encoder.dim(1)?;
        let img_seq = hidden.dim(1)?;
        let attn_txt = self.to_add_out.forward(&out.narrow(1, 0, txt_seq)?)?;
        let attn_img = self.to_out.forward(&out.narrow(1, txt_seq, img_seq)?)?;
        Ok((attn_img, attn_txt))
    }
}

/// diffusers `FeedForward` (`net.0.proj` → activation → `net.2`). The image stream uses exact GELU
/// (`gelu_erf`), the context stream the tanh approximation (`gelu`) — mirroring the mlx port + control.
struct FeedForward {
    lin1: QLinear,
    lin2: QLinear,
    approx: bool,
}

impl FeedForward {
    fn new(approx: bool, d: Dims, vb: &VarBuilder) -> Result<Self> {
        let lin1 = QLinear::linear_detect(d.hidden, d.mlp, vb, "net.0.proj", true)?;
        let lin2 = QLinear::linear_detect(d.mlp, d.hidden, vb, "net.2", true)?;
        Ok(Self { lin1, lin2, approx })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.lin1.forward(x)?;
        let x = if self.approx {
            x.gelu()?
        } else {
            x.gelu_erf()?
        };
        self.lin2.forward(&x)
    }
}

/// One diffusers FLUX joint (double-stream) block (`transformer_blocks.{i}`).
struct JointBlock {
    norm1: AdaLayerNormZero,
    norm1_context: AdaLayerNormZero,
    attn: JointAttention,
    ff: FeedForward,
    ff_context: FeedForward,
    norm2: LayerNorm,
}

impl JointBlock {
    fn new(d: Dims, vb: &VarBuilder) -> Result<Self> {
        Ok(Self {
            norm1: AdaLayerNormZero::new(6, d, vb, "norm1")?,
            norm1_context: AdaLayerNormZero::new(6, d, vb, "norm1_context")?,
            attn: JointAttention::new(d, &vb.pp("attn"))?,
            ff: FeedForward::new(false, d, &vb.pp("ff"))?,
            ff_context: FeedForward::new(true, d, &vb.pp("ff_context"))?,
            norm2: layer_norm_no_affine(d.hidden, vb.dtype(), vb.device())?,
        })
    }

    /// `(hidden, encoder)` → `(hidden', encoder')`.
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
        Ok((hidden, encoder))
    }
}

/// The AdaLN-Zero post-attention residual + gated FF (shared by the image/text streams):
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
    let normed = norm
        .forward(&hidden)?
        .broadcast_mul(&(scale_mlp.unsqueeze(1)? + 1.0)?)?
        .broadcast_add(&shift_mlp.unsqueeze(1)?)?;
    let ff_out = ff.forward(&normed)?;
    hidden.broadcast_add(&ff_out.broadcast_mul(&gate_mlp.unsqueeze(1)?)?)
}

/// One diffusers FLUX single-stream block (`single_transformer_blocks.{i}`): AdaLN-Zero-Single
/// modulation, split `attn.to_q`/`to_k`/`to_v` (+ RMS q/k norm), a parallel `proj_mlp` (hidden →
/// 4·hidden), and a fused `proj_out` (hidden + 4·hidden → hidden) over `cat(attn, gelu(mlp))`. The
/// single blocks operate on the concatenated `cat(txt, img)` stream (like the BFL single blocks).
struct SingleBlock {
    norm: AdaLayerNormZero,
    to_q: QLinear,
    to_k: QLinear,
    to_v: QLinear,
    norm_q: RmsNorm,
    norm_k: RmsNorm,
    proj_mlp: QLinear,
    proj_out: QLinear,
    heads: usize,
    head_dim: usize,
}

impl SingleBlock {
    fn new(d: Dims, vb: &VarBuilder) -> Result<Self> {
        let attn = vb.pp("attn");
        Ok(Self {
            // `norm` here is AdaLayerNormZeroSingle: `norm.linear` emits 3·hidden.
            norm: AdaLayerNormZero::new(3, d, vb, "norm")?,
            to_q: QLinear::linear_detect(d.hidden, d.hidden, &attn, "to_q", true)?,
            to_k: QLinear::linear_detect(d.hidden, d.hidden, &attn, "to_k", true)?,
            to_v: QLinear::linear_detect(d.hidden, d.hidden, &attn, "to_v", true)?,
            norm_q: RmsNorm::new(attn.get(d.head_dim, "norm_q.weight")?, RMS_EPS),
            norm_k: RmsNorm::new(attn.get(d.head_dim, "norm_k.weight")?, RMS_EPS),
            proj_mlp: QLinear::linear_detect(d.hidden, d.mlp, vb, "proj_mlp", true)?,
            proj_out: QLinear::linear_detect(d.hidden + d.mlp, d.hidden, vb, "proj_out", true)?,
            heads: d.heads,
            head_dim: d.head_dim,
        })
    }

    fn forward(&self, hidden: &Tensor, emb: &Tensor, pe: &Tensor) -> Result<Tensor> {
        let (norm_hidden, gate) = self.norm.forward_three(hidden, emb)?;
        let (b, s, _) = norm_hidden.dims3()?;
        let (heads, head_dim) = (self.heads, self.head_dim);
        let to_heads =
            |t: Tensor| -> Result<Tensor> { t.reshape((b, s, heads, head_dim))?.transpose(1, 2) };
        let q = to_heads(self.to_q.forward(&norm_hidden)?)?.apply(&self.norm_q)?;
        let k = to_heads(self.to_k.forward(&norm_hidden)?)?.apply(&self.norm_k)?;
        let v = to_heads(self.to_v.forward(&norm_hidden)?)?;
        let q = apply_rope(&q, pe)?.contiguous()?;
        let k = apply_rope(&k, pe)?.contiguous()?;
        let attn = scaled_dot_product_attention(&q, &k, &v)?;
        let attn = attn.transpose(1, 2)?.flatten_from(2)?; // [b, s, hidden]
        let mlp = self.proj_mlp.forward(&norm_hidden)?.gelu()?;
        let out = self
            .proj_out
            .forward(&Tensor::cat(&[&attn, &mlp], D::Minus1)?)?;
        // Residual with the single-block gate.
        hidden.broadcast_add(&out.broadcast_mul(&gate.unsqueeze(1)?)?)
    }
}

/// `silu → linear_1 → silu → linear_2` MLP (diffusers `TimestepEmbedding` / text projection).
struct MlpEmbedder {
    lin1: QLinear,
    lin2: QLinear,
}

impl MlpEmbedder {
    fn new(in_dim: usize, hidden: usize, vb: &VarBuilder) -> Result<Self> {
        Ok(Self {
            lin1: QLinear::linear_detect(in_dim, hidden, vb, "linear_1", true)?,
            lin2: QLinear::linear_detect(hidden, hidden, vb, "linear_2", true)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.lin2.forward(&self.lin1.forward(x)?.silu()?)
    }
}

/// diffusers `CombinedTimestep(Guidance)TextEmbeddings`: sinusoidal time (+ optional guidance)
/// projection summed with the pooled-text projection. `time_text_embed.{timestep,guidance,text}_embedder`.
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
        vb: &VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            timestep: MlpEmbedder::new(256, hidden, &vb.pp("timestep_embedder"))?,
            text: MlpEmbedder::new(pooled_dim, hidden, &vb.pp("text_embedder"))?,
            guidance: if supports_guidance {
                Some(MlpEmbedder::new(256, hidden, &vb.pp("guidance_embedder"))?)
            } else {
                None
            },
        })
    }

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

/// diffusers `AdaLayerNormContinuous` output head (`norm_out`): `silu(emb) @ linear` → `(shift, scale)`,
/// `norm(hidden)·(1+scale)+shift`, then the `proj_out` to the packed latent channels.
struct OutputHead {
    norm_linear: QLinear,
    norm: LayerNorm,
    proj_out: QLinear,
}

impl OutputHead {
    fn new(d: Dims, out_channels: usize, vb: &VarBuilder) -> Result<Self> {
        Ok(Self {
            norm_linear: QLinear::linear_detect(
                d.hidden,
                2 * d.hidden,
                vb,
                "norm_out.linear",
                true,
            )?,
            norm: layer_norm_no_affine(d.hidden, vb.dtype(), vb.device())?,
            proj_out: QLinear::linear_detect(d.hidden, out_channels, vb, "proj_out", true)?,
        })
    }

    fn forward(&self, hidden: &Tensor, emb: &Tensor) -> Result<Tensor> {
        // diffusers chunks the AdaLNContinuous projection as (shift, scale) — shift FIRST.
        let p = self
            .norm_linear
            .forward(&emb.silu()?)?
            .chunk(2, D::Minus1)?;
        let normed = self
            .norm
            .forward(hidden)?
            .broadcast_mul(&(p[1].unsqueeze(1)? + 1.0)?)?
            .broadcast_add(&p[0].unsqueeze(1)?)?;
        self.proj_out.forward(&normed)
    }
}

/// The vendored diffusers FLUX.1 MMDiT — 19 double `transformer_blocks` + 38 single
/// `single_transformer_blocks`, packed-detecting every `Linear` through [`QLinear`]. Consumes the same
/// packed VAE latents / T5 context / CLIP pooled vector as the stock BFL model and emits the same
/// `[B, img_seq, 64]` velocity.
pub struct PackedFluxDit {
    x_embedder: QLinear,
    context_embedder: QLinear,
    time_text_embed: TimeTextEmbed,
    double_blocks: Vec<JointBlock>,
    single_blocks: Vec<SingleBlock>,
    output: OutputHead,
    pe_embedder: EmbedNd,
}

impl PackedFluxDit {
    /// Load the diffusers FLUX DiT from `vb` (rooted at the `transformer/` component). `cfg` is the
    /// shared FLUX [`Config`] (`Config::schnell()` / `Config::dev()`); `num_double`/`num_single` come
    /// from the component `config.json` (`num_layers` / `num_single_layers` = 19 / 38 for FLUX.1).
    pub fn new(cfg: &Config, num_double: usize, num_single: usize, vb: VarBuilder) -> Result<Self> {
        let d = Dims::from_config(cfg);
        let x_embedder =
            QLinear::linear_detect(cfg.in_channels, d.hidden, &vb, "x_embedder", true)?;
        let context_embedder =
            QLinear::linear_detect(CONTEXT_DIM, d.hidden, &vb, "context_embedder", true)?;
        let time_text_embed = TimeTextEmbed::new(
            cfg.guidance_embed,
            d.hidden,
            POOLED_DIM,
            &vb.pp("time_text_embed"),
        )?;
        let mut double_blocks = Vec::with_capacity(num_double);
        let vb_d = vb.pp("transformer_blocks");
        for i in 0..num_double {
            double_blocks.push(JointBlock::new(d, &vb_d.pp(i))?);
        }
        let mut single_blocks = Vec::with_capacity(num_single);
        let vb_s = vb.pp("single_transformer_blocks");
        for i in 0..num_single {
            single_blocks.push(SingleBlock::new(d, &vb_s.pp(i))?);
        }
        let output = OutputHead::new(d, cfg.in_channels, &vb)?;
        let pe_embedder = EmbedNd::new(d.head_dim, cfg.theta, cfg.axes_dim.clone());
        Ok(Self {
            x_embedder,
            context_embedder,
            time_text_embed,
            double_blocks,
            single_blocks,
            output,
            pe_embedder,
        })
    }

    /// FLUX velocity forward. `img`: packed latents `[B, img_seq, 64]`. `txt`: T5 context
    /// `[B, txt_seq, 4096]`. `pooled`: CLIP pooled `[B, 768]`. `timesteps`: `[B]` raw σ (the sinusoid
    /// embedder scales ×1000). `guidance`: `[B]` embedded guidance (dev only). `img_ids`/`txt_ids`:
    /// `[B, seq, 3]` FLUX position ids. Returns `[B, img_seq, 64]`.
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
    ) -> Result<Tensor> {
        let mut hidden = self.x_embedder.forward(img)?;
        let mut encoder = self.context_embedder.forward(txt)?;
        let emb = self.time_text_embed.forward(timesteps, guidance, pooled)?;
        let pe = Tensor::cat(&[txt_ids, img_ids], 1)?.apply(&self.pe_embedder)?;

        for block in &self.double_blocks {
            let (h, e) = block.forward(&hidden, &encoder, &emb, &pe)?;
            hidden = h;
            encoder = e;
        }

        // The single blocks run on the concatenated `cat(txt, img)` stream.
        let txt_seq = encoder.dim(1)?;
        let mut joint = Tensor::cat(&[&encoder, &hidden], 1)?;
        for block in &self.single_blocks {
            joint = block.forward(&joint, &emb, &pe)?;
        }
        let img_seq = hidden.dim(1)?;
        let hidden = joint.narrow(1, txt_seq, img_seq)?;

        self.output.forward(&hidden, &emb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Device;
    use candle_gen::candle_nn::VarMap;

    /// A tiny FLUX [`Config`] for the GPU-free DiT smoke test: inner 16 (heads 2 · head_dim 8),
    /// `axes_dim` summing to 8, real input widths (in 64 / context 4096 / pooled 768).
    fn tiny_cfg(guidance: bool) -> Config {
        Config {
            in_channels: 64,
            vec_in_dim: 768,
            context_in_dim: 4096,
            hidden_size: 16,
            mlp_ratio: 4.0,
            num_heads: 2,
            depth: 0,
            depth_single_blocks: 0,
            axes_dim: vec![2, 2, 4],
            theta: 10_000,
            qkv_bias: true,
            guidance_embed: guidance,
        }
    }

    /// The vendored diffusers DiT loads (dense, no `.scales`) and forwards to the FLUX velocity shape
    /// `[B, img_seq, 64]` with finite values — exercising the double/single/output-head wiring and the
    /// `cat(txt,img)` single-block plumbing end-to-end. `guidance_embed = true` also exercises the dev
    /// guidance-embedder path. A 2-double / 2-single tiny config keeps it GPU-free.
    #[test]
    fn packed_dit_dense_forward_shape_and_finite() -> Result<()> {
        let dev = Device::Cpu;
        let cfg = tiny_cfg(true);
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let dit = PackedFluxDit::new(&cfg, 2, 2, vb)?;

        let (b, img_seq, txt_seq) = (1usize, 4usize, 3usize);
        let img = Tensor::randn(0f32, 1f32, (b, img_seq, 64), &dev)?;
        let txt = Tensor::randn(0f32, 1f32, (b, txt_seq, 4096), &dev)?;
        let pooled = Tensor::randn(0f32, 1f32, (b, 768), &dev)?;
        let img_ids = Tensor::zeros((b, img_seq, 3), DType::F32, &dev)?;
        let txt_ids = Tensor::zeros((b, txt_seq, 3), DType::F32, &dev)?;
        let ts = Tensor::full(0.5f32, b, &dev)?;
        let g = Tensor::full(3.5f32, b, &dev)?;

        let out = dit.forward(&img, &img_ids, &txt, &txt_ids, &ts, &pooled, Some(&g))?;
        assert_eq!(out.dims(), &[b, img_seq, 64]);
        let max = out.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(max.is_finite(), "DiT output must be finite, got max {max}");
        Ok(())
    }
}
