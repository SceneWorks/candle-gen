//! Cosmos-Predict2 DiT — the Anima image transformer, the candle transcription of `mlx-gen-anima`'s
//! `transformer.rs` (itself from diffusers `transformer_cosmos.py::CosmosTransformer3DModel` + the
//! `Cosmos-2.0-Diffusion-2B-Text2Image` config). Weight keys are the **original Cosmos** names
//! (`{prefix}.blocks.N.*`, `{prefix}.x_embedder.proj.1`, `{prefix}.t_embedder.1.*`,
//! `{prefix}.final_layer.*`) — the single-file bf16 checkpoint loads unchanged. `prefix` is detected
//! per file (`net` for the base cut, `model.diffusion_model` for turbo/aesthetic; see [`crate::loader`]).
//!
//! Ported pieces: `CosmosPatchEmbed`, `CosmosTimestepEmbedding`/`CosmosEmbedding`,
//! `CosmosAdaLayerNorm(Zero)`, `CosmosAttention` (q/k RMSNorm + half-split RoPE on self-attn),
//! `CosmosTransformerBlock`, final layer. **Skipped** (config-off for Anima): learnable pos-embed,
//! cross-attn projection, ControlNet hooks. Heads == kv_heads (MHA, no GQA repeat in the DiT — the
//! GQA 16/8 lives only in the Qwen3 encoder).

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::Result;

use crate::adapt::AdaptLinear;
use crate::config::DitConfig;
use crate::nn::{apply_rope_half, layer_norm_no_affine, modulate, rms_norm, sdpa, timestep_sincos};
use crate::rope::cosmos_image_rope;

/// q/k RMSNorm eps in diffusers `Attention` (`qk_norm="rms_norm"`, default `eps=1e-5`).
const ATTN_QK_NORM_EPS: f64 = 1e-5;
/// LayerNorm / time-embed-norm eps (`elementwise_affine=false, eps=1e-6`).
const NORM_EPS: f64 = 1e-6;
/// Sinusoidal timestep-embedding `max_period` (`Timesteps` default).
const TIME_MAX_PERIOD: f64 = 10000.0;

// Every DiT projection is a bias-less, **packed-detecting** [`AdaptLinear`] (`crate::adapt`): dense
// `{name}.weight`, or an MLX-packed `{name}.weight` u32 codes + `.scales` + `.biases` (auto-detected by
// `{name}.scales`), plus zero or more forward-time LoRA residuals. The packed forward dequantizes to a
// dense matmul per step (sc-7702 `DequantDense` — CPU-capable, coherent at Q4; NOT the CUDA-only int8
// `QMatMul` fast path). Anima's tiers pack ONLY the DiT (the conditioner / Qwen3 TE / VAE stay dense
// bf16 — the sc-10517 dense-TE precedent), all at MLX group size 64. With no adapter attached the
// forward is byte-identical to the pre-sc-10640 bare linear.

/// `CosmosEmbedding`: sinusoidal time_proj → `CosmosTimestepEmbedding` (`temb`, 3·hidden) + `RMSNorm`
/// (`embedded_timestep`, hidden).
struct TimeEmbed {
    linear_1: AdaptLinear,
    linear_2: AdaptLinear,
    norm: Tensor,
    hidden: usize,
    device: Device,
}

impl TimeEmbed {
    fn new(vb: &VarBuilder, cfg: &DitConfig) -> Result<Self> {
        Ok(Self {
            linear_1: AdaptLinear::detect(vb, "t_embedder.1.linear_1")?,
            linear_2: AdaptLinear::detect(vb, "t_embedder.1.linear_2")?,
            norm: vb.get_unchecked("t_embedding_norm.weight")?,
            hidden: cfg.hidden_size(),
            device: vb.device().clone(),
        })
    }

    /// `sigma`: `[B]`. Returns `(temb [B, 3·hidden], embedded [B, hidden])` in `dtype`.
    fn forward(&self, sigma: &Tensor, dtype: DType) -> Result<(Tensor, Tensor)> {
        let proj = timestep_sincos(sigma, self.hidden, TIME_MAX_PERIOD, dtype, &self.device)?;
        let temb = self
            .linear_2
            .forward(&self.linear_1.forward(&proj)?.silu()?)?;
        let embedded = rms_norm(&proj, &self.norm, NORM_EPS)?;
        Ok((temb, embedded))
    }

    /// Visit this embedding's two adaptable projections for the additive-adapter walk (sc-10640).
    fn visit_adaptable_mut(
        &mut self,
        prefix: &str,
        f: &mut dyn FnMut(&str, &mut AdaptLinear) -> Result<()>,
    ) -> Result<()> {
        f(&format!("{prefix}.linear_1"), &mut self.linear_1)?;
        f(&format!("{prefix}.linear_2"), &mut self.linear_2)?;
        Ok(())
    }
}

/// `CosmosAdaLayerNormZero` (norm1/2/3): LayerNorm(no affine) then `(1+scale)·norm + shift`, plus a
/// `gate`. `linear_2` emits `3·hidden` (shift|scale|gate), added to `temb`.
struct AdaLayerNormZero {
    linear_1: AdaptLinear,
    linear_2: AdaptLinear,
}

impl AdaLayerNormZero {
    fn new(vb: &VarBuilder, prefix: &str) -> Result<Self> {
        Ok(Self {
            linear_1: AdaptLinear::detect(vb, &format!("{prefix}.1"))?,
            linear_2: AdaptLinear::detect(vb, &format!("{prefix}.2"))?,
        })
    }

    /// Returns `(modulated_norm [B,S,H], gate [B,1,H])`.
    fn forward(
        &self,
        hidden: &Tensor,
        embedded: &Tensor,
        temb: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let e = self
            .linear_2
            .forward(&self.linear_1.forward(&embedded.silu()?)?)?;
        let e = (e + temb)?; // [B, 3H]
        let parts = e.chunk(3, 1)?; // shift, scale, gate
        let shift = parts[0].unsqueeze(1)?;
        let scale = parts[1].unsqueeze(1)?;
        let gate = parts[2].unsqueeze(1)?;
        let normed = layer_norm_no_affine(hidden, NORM_EPS)?;
        Ok((modulate(&normed, &scale, &shift)?, gate))
    }

    /// Visit the modulation's two adaptable projections (`{prefix}.1`, `{prefix}.2`) for the
    /// additive-adapter walk (sc-10640) — the adaLN down/up pair the official LoRA targets.
    fn visit_adaptable_mut(
        &mut self,
        prefix: &str,
        f: &mut dyn FnMut(&str, &mut AdaptLinear) -> Result<()>,
    ) -> Result<()> {
        f(&format!("{prefix}.1"), &mut self.linear_1)?;
        f(&format!("{prefix}.2"), &mut self.linear_2)?;
        Ok(())
    }
}

/// `CosmosAdaLayerNorm` (final `norm_out`): LayerNorm(no affine) then `(1+scale)·norm + shift`.
/// `linear_2` emits `2·hidden` (shift|scale), added to `temb[..., :2·hidden]`.
struct AdaLayerNorm {
    linear_1: AdaptLinear,
    linear_2: AdaptLinear,
    hidden: usize,
}

impl AdaLayerNorm {
    fn new(vb: &VarBuilder, prefix: &str, cfg: &DitConfig) -> Result<Self> {
        Ok(Self {
            linear_1: AdaptLinear::detect(vb, &format!("{prefix}.1"))?,
            linear_2: AdaptLinear::detect(vb, &format!("{prefix}.2"))?,
            hidden: cfg.hidden_size(),
        })
    }

    fn forward(&self, hidden: &Tensor, embedded: &Tensor, temb: &Tensor) -> Result<Tensor> {
        let e = self
            .linear_2
            .forward(&self.linear_1.forward(&embedded.silu()?)?)?;
        let e = (e + temb.narrow(1, 0, 2 * self.hidden)?)?; // + temb[:, :2H]
        let parts = e.chunk(2, 1)?; // shift, scale
        let shift = parts[0].unsqueeze(1)?;
        let scale = parts[1].unsqueeze(1)?;
        let normed = layer_norm_no_affine(hidden, NORM_EPS)?;
        modulate(&normed, &scale, &shift)
    }

    /// Visit the final modulation's two adaptable projections (`{prefix}.1`, `{prefix}.2`) for the
    /// additive-adapter walk (sc-10640).
    fn visit_adaptable_mut(
        &mut self,
        prefix: &str,
        f: &mut dyn FnMut(&str, &mut AdaptLinear) -> Result<()>,
    ) -> Result<()> {
        f(&format!("{prefix}.1"), &mut self.linear_1)?;
        f(&format!("{prefix}.2"), &mut self.linear_2)?;
        Ok(())
    }
}

/// `CosmosAttention` — self (attn1: q/k/v from hidden, RoPE) or cross (attn2: q from hidden, k/v from
/// text, no RoPE). Per-head q/k RMSNorm; heads == kv_heads (no GQA repeat for Anima).
struct Attention {
    to_q: AdaptLinear,
    to_k: AdaptLinear,
    to_v: AdaptLinear,
    to_out: AdaptLinear,
    norm_q: Tensor,
    norm_k: Tensor,
    heads: usize,
    head_dim: usize,
    scale: f64,
}

impl Attention {
    fn new(vb: &VarBuilder, prefix: &str, cfg: &DitConfig) -> Result<Self> {
        let hd = cfg.attention_head_dim;
        Ok(Self {
            to_q: AdaptLinear::detect(vb, &format!("{prefix}.q_proj"))?,
            to_k: AdaptLinear::detect(vb, &format!("{prefix}.k_proj"))?,
            to_v: AdaptLinear::detect(vb, &format!("{prefix}.v_proj"))?,
            to_out: AdaptLinear::detect(vb, &format!("{prefix}.output_proj"))?,
            norm_q: vb.get_unchecked(&format!("{prefix}.q_norm.weight"))?,
            norm_k: vb.get_unchecked(&format!("{prefix}.k_norm.weight"))?,
            heads: cfg.num_attention_heads,
            head_dim: hd,
            scale: (hd as f64).powf(-0.5),
        })
    }

    /// `hidden`: `[B, Sq, H]`. `encoder`: `Some([B, Sk, Ctx])` for cross-attn (else self-attn on
    /// `hidden`). `rope`: `Some((cos,sin))` applies half-split RoPE (self-attn only).
    fn forward(
        &self,
        hidden: &Tensor,
        encoder: Option<&Tensor>,
        rope: Option<(&Tensor, &Tensor)>,
    ) -> Result<Tensor> {
        let (b, sq, _) = hidden.dims3()?;
        let kv_src = encoder.unwrap_or(hidden);
        let sk = kv_src.dim(1)?;

        let q = self
            .to_q
            .forward(hidden)?
            .reshape((b, sq, self.heads, self.head_dim))?;
        let k = self
            .to_k
            .forward(kv_src)?
            .reshape((b, sk, self.heads, self.head_dim))?;
        let v = self
            .to_v
            .forward(kv_src)?
            .reshape((b, sk, self.heads, self.head_dim))?;

        // per-head q/k RMSNorm (over the head_dim).
        let q = rms_norm(&q, &self.norm_q, ATTN_QK_NORM_EPS)?;
        let k = rms_norm(&k, &self.norm_k, ATTN_QK_NORM_EPS)?;

        let (q, k) = match rope {
            Some((cos, sin)) => (
                apply_rope_half(&q, cos, sin)?,
                apply_rope_half(&k, cos, sin)?,
            ),
            None => (q, k),
        };

        // [b,s,h,hd] -> [b,h,s,hd]
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;
        let o = sdpa(&q, &k, &v, self.scale, None)?;
        let o = o
            .transpose(1, 2)?
            .reshape((b, sq, self.heads * self.head_dim))?;
        Ok(self.to_out.forward(&o)?)
    }

    /// Visit this attention's four adaptable projections (`{prefix}.{q,k,v,output}_proj`) for the
    /// additive-adapter walk (sc-10640).
    fn visit_adaptable_mut(
        &mut self,
        prefix: &str,
        f: &mut dyn FnMut(&str, &mut AdaptLinear) -> Result<()>,
    ) -> Result<()> {
        f(&format!("{prefix}.q_proj"), &mut self.to_q)?;
        f(&format!("{prefix}.k_proj"), &mut self.to_k)?;
        f(&format!("{prefix}.v_proj"), &mut self.to_v)?;
        f(&format!("{prefix}.output_proj"), &mut self.to_out)?;
        Ok(())
    }
}

/// `FeedForward(mult=4, activation="gelu")` — `net.2(gelu_exact(net.0.proj(x)))`, no bias.
struct FeedForward {
    proj_in: AdaptLinear,  // mlp.layer1
    proj_out: AdaptLinear, // mlp.layer2
}

impl FeedForward {
    fn new(vb: &VarBuilder, prefix: &str) -> Result<Self> {
        Ok(Self {
            proj_in: AdaptLinear::detect(vb, &format!("{prefix}.layer1"))?,
            proj_out: AdaptLinear::detect(vb, &format!("{prefix}.layer2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // gelu_exact = erf GELU (candle `gelu_erf`), matching `mlx_gen::nn::gelu_exact`.
        let h = self.proj_in.forward(x)?.gelu_erf()?;
        Ok(self.proj_out.forward(&h)?)
    }

    /// Visit the FFN's two adaptable projections (`{prefix}.layer1`, `{prefix}.layer2`) for the
    /// additive-adapter walk (sc-10640).
    fn visit_adaptable_mut(
        &mut self,
        prefix: &str,
        f: &mut dyn FnMut(&str, &mut AdaptLinear) -> Result<()>,
    ) -> Result<()> {
        f(&format!("{prefix}.layer1"), &mut self.proj_in)?;
        f(&format!("{prefix}.layer2"), &mut self.proj_out)?;
        Ok(())
    }
}

/// `CosmosTransformerBlock`: gated self-attn → gated cross-attn → gated FF.
struct Block {
    norm1: AdaLayerNormZero,
    attn1: Attention,
    norm2: AdaLayerNormZero,
    attn2: Attention,
    norm3: AdaLayerNormZero,
    ff: FeedForward,
}

impl Block {
    fn new(vb: &VarBuilder, prefix: &str, cfg: &DitConfig) -> Result<Self> {
        Ok(Self {
            norm1: AdaLayerNormZero::new(vb, &format!("{prefix}.adaln_modulation_self_attn"))?,
            attn1: Attention::new(vb, &format!("{prefix}.self_attn"), cfg)?,
            norm2: AdaLayerNormZero::new(vb, &format!("{prefix}.adaln_modulation_cross_attn"))?,
            attn2: Attention::new(vb, &format!("{prefix}.cross_attn"), cfg)?,
            norm3: AdaLayerNormZero::new(vb, &format!("{prefix}.adaln_modulation_mlp"))?,
            ff: FeedForward::new(vb, &format!("{prefix}.mlp"))?,
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        encoder: &Tensor,
        embedded: &Tensor,
        temb: &Tensor,
        rope: (&Tensor, &Tensor),
    ) -> Result<Tensor> {
        // 1. self attention (RoPE)
        let (normed, gate) = self.norm1.forward(hidden, embedded, temb)?;
        let attn = self.attn1.forward(&normed, None, Some(rope))?;
        let hidden = (hidden + gate.broadcast_mul(&attn)?)?;
        // 2. cross attention (no RoPE). No attention mask over the conditioner's 512-token output: the
        // diffusers reference leaves the zero-padded positions UNMASKED (condition_embedder_anima.py:346,
        // transformer_cosmos.py:204 SDPA attn_mask=None). Padded keys are zero vectors, not −inf. Do NOT
        // "fix" this into a mask — it would introduce a conditioning divergence.
        let (normed, gate) = self.norm2.forward(&hidden, embedded, temb)?;
        let attn = self.attn2.forward(&normed, Some(encoder), None)?;
        let hidden = (hidden + gate.broadcast_mul(&attn)?)?;
        // 3. feed forward
        let (normed, gate) = self.norm3.forward(&hidden, embedded, temb)?;
        let ff = self.ff.forward(&normed)?;
        Ok((hidden + gate.broadcast_mul(&ff)?)?)
    }

    /// Visit this block's 16 adaptable projections (self/cross attn q/k/v/out, the three adaLN
    /// modulation `.1`/`.2` pairs, ff layer1/layer2) for the additive-adapter walk (sc-10640).
    fn visit_adaptable_mut(
        &mut self,
        prefix: &str,
        f: &mut dyn FnMut(&str, &mut AdaptLinear) -> Result<()>,
    ) -> Result<()> {
        self.norm1
            .visit_adaptable_mut(&format!("{prefix}.adaln_modulation_self_attn"), f)?;
        self.attn1
            .visit_adaptable_mut(&format!("{prefix}.self_attn"), f)?;
        self.norm2
            .visit_adaptable_mut(&format!("{prefix}.adaln_modulation_cross_attn"), f)?;
        self.attn2
            .visit_adaptable_mut(&format!("{prefix}.cross_attn"), f)?;
        self.norm3
            .visit_adaptable_mut(&format!("{prefix}.adaln_modulation_mlp"), f)?;
        self.ff.visit_adaptable_mut(&format!("{prefix}.mlp"), f)?;
        Ok(())
    }
}

/// The full Cosmos-Predict2 DiT.
pub struct CosmosDiT {
    patch_embed: AdaptLinear, // x_embedder.proj.1
    time_embed: TimeEmbed,
    blocks: Vec<Block>,
    norm_out: AdaLayerNorm,
    proj_out: AdaptLinear, // final_layer.linear
    cfg: DitConfig,
    device: Device,
}

impl CosmosDiT {
    /// `vb` is a VarBuilder rooted at the checkpoint's DiT prefix (`"net"` or
    /// `"model.diffusion_model"`); keys are the original Cosmos names.
    pub fn new(vb: &VarBuilder, cfg: DitConfig) -> Result<Self> {
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            blocks.push(Block::new(vb, &format!("blocks.{i}"), &cfg)?);
        }
        Ok(Self {
            patch_embed: AdaptLinear::detect(vb, "x_embedder.proj.1")?,
            time_embed: TimeEmbed::new(vb, &cfg)?,
            blocks,
            norm_out: AdaLayerNorm::new(vb, "final_layer.adaln_modulation", &cfg)?,
            proj_out: AdaptLinear::detect(vb, "final_layer.linear")?,
            cfg,
            device: vb.device().clone(),
        })
    }

    pub fn config(&self) -> &DitConfig {
        &self.cfg
    }

    /// Whether the DiT loaded from a **packed** MLX tier (its projections are quantized). Probed on
    /// `proj_out` (`final_layer.linear`, in=hidden), like `candle-gen-wan`'s `is_packed` — a hidden-wide
    /// projection that is always group-64-packable, so it packs whenever the tier does. NOT `patch_embed`:
    /// its `in` (17·pt·ph·pw = 68) is not a multiple of the group size, so a converter may leave it dense
    /// even in a packed tier. A dense checkpoint packs none.
    pub fn is_packed(&self) -> bool {
        self.proj_out.is_packed()
    }

    /// The device the DiT weights live on — residual factors are read on CPU and moved here.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Walk every adaptable projection, invoking `f(path, &mut AdaptLinear)` once each with the
    /// projection's DiT-root-relative dotted path (`blocks.<i>.*` + the globals). The additive-adapter
    /// installer routes forward-time residuals through this (sc-10640; the candle analog of mlx-gen's
    /// `AdaptableHost` and of `candle-gen-wan`'s `visit_adaptable_mut`). Covers EVERY projection — not
    /// just the canonical 448 — so a residual routes exactly the surface the dense fold would.
    pub fn visit_adaptable_mut(
        &mut self,
        f: &mut dyn FnMut(&str, &mut AdaptLinear) -> Result<()>,
    ) -> Result<()> {
        f("x_embedder.proj.1", &mut self.patch_embed)?;
        self.time_embed.visit_adaptable_mut("t_embedder.1", f)?;
        for (i, blk) in self.blocks.iter_mut().enumerate() {
            blk.visit_adaptable_mut(&format!("blocks.{i}"), f)?;
        }
        self.norm_out
            .visit_adaptable_mut("final_layer.adaln_modulation", f)?;
        f("final_layer.linear", &mut self.proj_out)?;
        Ok(())
    }

    /// Patchify a `[B, C, 1, Hl, Wl]` latent (`C=17` after mask concat) to `[B, seq, C·pt·ph·pw]`.
    fn patchify(&self, x: &Tensor) -> Result<Tensor> {
        let (pt, ph, pw) = self.cfg.patch_size;
        let (b, c, t, hl, wl) = x.dims5()?;
        // reshape (B, C, T/pt, pt, Hl/ph, ph, Wl/pw, pw)
        let x = x.reshape(&[b, c, t / pt, pt, hl / ph, ph, wl / pw, pw])?;
        // permute (0,2,4,6,1,3,5,7) -> (B, T/pt, Hl/ph, Wl/pw, C, pt, ph, pw)
        let x = x
            .permute(&[0usize, 2, 4, 6, 1, 3, 5, 7][..])?
            .contiguous()?;
        let seq = (t / pt) * (hl / ph) * (wl / pw);
        Ok(x.reshape((b, seq, c * pt * ph * pw))?)
    }

    /// Inverse of [`patchify`] on `[B, seq, ph·pw·pt·out_ch]` → `[B, out_ch, 1, Hl, Wl]`.
    fn unpatchify(&self, x: &Tensor, pe_t: usize, pe_h: usize, pe_w: usize) -> Result<Tensor> {
        let (pt, ph, pw) = self.cfg.patch_size;
        let oc = self.cfg.out_channels;
        let b = x.dim(0)?;
        // [B, seq, ph*pw*pt*oc] -> [B, pe_t, pe_h, pe_w, ph, pw, pt, oc]
        let x = x.reshape(&[b, pe_t, pe_h, pe_w, ph, pw, pt, oc])?;
        // permute (0,7,1,6,2,4,3,5) -> [B, oc, pe_t, pt, pe_h, ph, pe_w, pw]
        let x = x
            .permute(&[0usize, 7, 1, 6, 2, 4, 3, 5][..])?
            .contiguous()?;
        // collapse patch pairs -> [B, oc, pe_t*pt, pe_h*ph, pe_w*pw]
        Ok(x.reshape((b, oc, pe_t * pt, pe_h * ph, pe_w * pw))?)
    }

    /// Denoise forward. `latents`: `[B, 16, 1, Hl, Wl]`. `sigma`: `[B]`. `encoder`: `[B, 512,
    /// text_embed_dim]`. Returns the velocity `[B, 16, 1, Hl, Wl]` in `dtype`.
    pub fn forward(
        &self,
        latents: &Tensor,
        sigma: &Tensor,
        encoder: &Tensor,
        dtype: DType,
    ) -> Result<Tensor> {
        let latents = latents.to_dtype(dtype)?;
        let (b, _c, t, hl, wl) = latents.dims5()?;
        let (pt, ph, pw) = self.cfg.patch_size;
        let (pe_t, pe_h, pe_w) = (t / pt, hl / ph, wl / pw);

        // 1. concat the (all-zeros) padding-mask channel -> [B,17,1,Hl,Wl].
        let hidden = if self.cfg.concat_padding_mask {
            let pad = Tensor::zeros((b, 1, t, hl, wl), dtype, &self.device)?;
            Tensor::cat(&[&latents, &pad], 1)?
        } else {
            latents
        };

        // 2. RoPE for this latent grid (per-axis OOD-guarded).
        let rope = cosmos_image_rope(&self.cfg, pe_t, pe_h, pe_w, &self.device)?;

        // 3. patchify + patch-embed -> [B, seq, hidden].
        let hidden = self.patch_embed.forward(&self.patchify(&hidden)?)?;

        // 4. time embedding.
        let (temb, embedded) = self.time_embed.forward(sigma, dtype)?;

        // 5. transformer blocks.
        let mut hidden = hidden;
        for block in &self.blocks {
            hidden = block.forward(&hidden, encoder, &embedded, &temb, (&rope.cos, &rope.sin))?;
        }

        // 6. output norm + projection + unpatchify.
        let hidden = self.norm_out.forward(&hidden, &embedded, &temb)?;
        let hidden = self.proj_out.forward(&hidden)?;
        self.unpatchify(&hidden, pe_t, pe_h, pe_w)
    }
}

#[cfg(test)]
mod tests {
    //! Quant path (Q4 packed) exercised on candle's **CPU** backend: `AdaptLinear::detect` packed-detects
    //! an MLX-packed triple and the dequant-dense forward reproduces the affine grid — proving pack →
    //! load → forward on CPU, no CUDA. (The CUDA-only path is the int8 fast GEMM, which Anima does NOT
    //! use.)
    use super::*;
    use candle_gen::candle_core::safetensors::MmapedSafetensors;
    use candle_gen::candle_core::Device;
    use candle_gen::candle_nn::{Linear, Module};
    use candle_gen::quant::MLX_GROUP_SIZE;
    use std::collections::HashMap;

    /// Test-side MLX Q4 packer (group `g`): per-element 4-bit codes → u32 words (LSB-first nibbles).
    /// Returns `(wq [out, in/8] u32, scales [out, in/g], biases [out, in/g], affine grid [out, in])` —
    /// the exact packed-parts fixture `lin()` consumes, plus the affine grid the dequant reproduces.
    fn q4_packed(out_dim: usize, in_dim: usize, g: usize) -> (Tensor, Tensor, Tensor, Vec<f32>) {
        let dev = Device::Cpu;
        let codes: Vec<u8> = (0..out_dim * in_dim)
            .map(|i| ((i * 7 + i / 13) % 16) as u8)
            .collect();
        let gpr = in_dim / g;
        let groups = out_dim * gpr;
        let scales: Vec<f32> = (0..groups).map(|gi| 0.0625 * (gi as f32 + 1.0)).collect();
        let biases: Vec<f32> = (0..groups).map(|gi| -0.5 - 0.25 * gi as f32).collect();
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

    /// A packed DiT projection (`.scales` present) loads packed and its dequant-dense forward matches
    /// the affine grid bit-exact on CPU; a dense sibling (no `.scales`) stays dense. Group size 64
    /// (Anima's tier), the `AdaptLinear::detect` default. This is the pack → load → forward CPU validation.
    #[test]
    fn packed_dit_linear_loads_and_forwards_on_cpu() {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (64usize, 128usize); // in divisible by group 64
        let (wq, s, b, grid) = q4_packed(out_dim, in_dim, MLX_GROUP_SIZE);

        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert("blocks.0.self_attn.q_proj.weight".into(), wq);
        map.insert("blocks.0.self_attn.q_proj.scales".into(), s);
        map.insert("blocks.0.self_attn.q_proj.biases".into(), b);
        // A dense sibling (no `.scales`) — the dense path must stay unchanged.
        map.insert(
            "blocks.0.self_attn.k_proj.weight".into(),
            Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev).unwrap(),
        );

        let tmp = std::env::temp_dir().join(format!("anima_q4_{}.safetensors", std::process::id()));
        candle_gen::candle_core::safetensors::save(&map, &tmp).unwrap();
        // SAFETY: just-written file, nothing else touches it during the test.
        let st = unsafe { MmapedSafetensors::new(&tmp).unwrap() };
        let vb = VarBuilder::from_backend(Box::new(st), DType::F32, dev.clone());

        let packed = AdaptLinear::detect(&vb, "blocks.0.self_attn.q_proj").unwrap();
        assert!(packed.is_packed(), "`.scales` ⇒ packed load");
        assert_eq!(packed.base_shape(), (out_dim, in_dim));
        let dense = AdaptLinear::detect(&vb, "blocks.0.self_attn.k_proj").unwrap();
        assert!(!dense.is_packed(), "no `.scales` ⇒ dense path");

        // The packed dequant-dense forward reproduces the affine grid on CPU.
        let grid_lin = Linear::new(
            Tensor::from_vec(grid, (out_dim, in_dim), &dev).unwrap(),
            None,
        );
        let x = Tensor::randn(0f32, 1f32, (4usize, in_dim), &dev).unwrap();
        let cos = cosine(&packed.forward(&x).unwrap(), &grid_lin.forward(&x).unwrap());
        assert!(
            cos > 0.99999,
            "packed vs affine-grid cosine {cos:.6} (CPU dequant-dense)"
        );
        let _ = std::fs::remove_file(&tmp);
    }
}
