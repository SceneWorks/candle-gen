//! The top-level Lens DiT (`LensTransformer2DModel`): multi-layer text front-end → `img_in` +
//! timestep embedding → 48 dual-stream blocks → `AdaLayerNormContinuous` + `proj_out` back to patch
//! space. Image-stream output only (the text stream is discarded after the last block).
//!
//! Candle port of `mlx-gen-lens/src/dit/transformer.rs`, a near-twin of `candle-gen-qwen-image`'s
//! MMDiT. The Lens-specific pieces are the **multi-layer text front-end** (4 captured gpt-oss layers,
//! per-layer RMSNorm eps 1e-5 → channel-concat → `txt_in`), the **fused** attention projections (in
//! [`super::attention`]), the **`[img, txt]`** join order, the **SwiGLU** GateMLP, and a **biased**
//! `norm_out.linear` (the Lens checkpoint uses the bias the qwen fork dropped).

use candle_gen::candle_core::{DType, Device, Result, Tensor, D};
use candle_gen::candle_nn::{linear, rms_norm, Linear, Module, RmsNorm, VarBuilder};

use super::block::LensTransformerBlock;
use super::rope::LensRope3d;

/// AdaLayerNormContinuous / per-layer text-norm epsilon for the affine-free LayerNorm.
const LN_EPS: f64 = 1e-6;
/// The per-layer text front-end RMSNorm epsilon (`txt_norm.{i}`).
const TXT_NORM_EPS: f64 = 1e-5;

/// The Lens-Turbo / Lens `transformer/config.json` values.
#[derive(Clone, Copy, Debug)]
pub struct LensDitConfig {
    pub patch_size: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub inner_dim: usize,
    pub enc_hidden_dim: usize,
    pub axes_dims_rope: [usize; 3],
    pub num_text_layers: usize,
}

impl LensDitConfig {
    pub fn lens() -> Self {
        Self {
            patch_size: 2,
            in_channels: 128,
            out_channels: 32,
            num_layers: 48,
            num_heads: 24,
            head_dim: 64,
            inner_dim: 1536,
            enc_hidden_dim: 2880,
            axes_dims_rope: [8, 28, 28],
            num_text_layers: 4, // selected_layer_index = (5, 11, 17, 23)
        }
    }

    /// SwiGLU GateMLP hidden width (`inner/3·8`).
    fn mlp_hidden(&self) -> usize {
        self.inner_dim / 3 * 8
    }
}

/// Affine-free LayerNorm over the last axis (dtype-preserving; computed in f32, eps 1e-6).
fn layer_norm(x: &Tensor) -> Result<Tensor> {
    let dt = x.dtype();
    let xf = x.to_dtype(DType::F32)?;
    let mean = xf.mean_keepdim(D::Minus1)?;
    let xc = xf.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
    xc.broadcast_div(&(var + LN_EPS)?.sqrt()?)?.to_dtype(dt)
}

/// Sinusoidal timestep projection (`Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0,
/// scale=1000)`): `[B] → [B, 256]` as `[cos | sin]`, base 10000.
fn timestep_proj(timesteps: &[f32], device: &Device) -> Result<Tensor> {
    let (proj_dim, scale, max_period) = (256usize, 1000f32, 10000f32);
    let half = proj_dim / 2;
    let ln = max_period.ln();
    let b = timesteps.len();
    let mut data = vec![0f32; b * proj_dim];
    for (bi, &t) in timesteps.iter().enumerate() {
        for k in 0..half {
            let freq = (-ln * k as f32 / half as f32).exp();
            let arg = t * freq * scale;
            data[bi * proj_dim + k] = arg.cos(); // flip_sin_to_cos → [cos | sin]
            data[bi * proj_dim + half + k] = arg.sin();
        }
    }
    Tensor::from_vec(data, (b, proj_dim), device)
}

/// `temb = linear_2(silu(linear_1(proj(t))))`, `[B] → [B, inner]`.
struct TimeEmbed {
    linear_1: Linear,
    linear_2: Linear,
}

impl TimeEmbed {
    fn new(inner: usize, vb: VarBuilder) -> Result<Self> {
        let te = vb.pp("timestep_embedder");
        Ok(Self {
            linear_1: linear(256, inner, te.pp("linear_1"))?,
            linear_2: linear(inner, inner, te.pp("linear_2"))?,
        })
    }

    fn forward(&self, timesteps: &[f32], device: &Device, dtype: DType) -> Result<Tensor> {
        let proj = timestep_proj(timesteps, device)?.to_dtype(dtype)?;
        let h = self.linear_1.forward(&proj)?.silu()?;
        self.linear_2.forward(&h)
    }
}

/// `AdaLayerNormContinuous`: affine-free LayerNorm scaled/shifted by `linear(silu(temb))`. The Lens
/// checkpoint's `norm_out.linear` **carries a bias** (the reference uses it). `[scale | shift]` →
/// `LN(x)·(1+scale) + shift`.
struct NormOut {
    linear: Linear,
}

impl NormOut {
    fn new(inner: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            linear: linear(inner, 2 * inner, vb.pp("linear"))?,
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
}

/// The Lens denoising DiT.
pub struct LensTransformer {
    img_in: Linear,
    txt_norm: Vec<RmsNorm>, // per-layer text-feature RMSNorm (eps 1e-5)
    txt_in: Linear,
    time_embed: TimeEmbed,
    blocks: Vec<LensTransformerBlock>,
    norm_out: NormOut,
    proj_out: Linear,
    rope: LensRope3d,
    cfg: LensDitConfig,
    device: Device,
    dtype: DType,
}

impl LensTransformer {
    /// Load from a diffusers `transformer/` weight set (the `VarBuilder` dtype is the working dtype —
    /// bf16 production / f32 parity gate).
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
        let mlp_hidden = cfg.mlp_hidden();
        for i in 0..cfg.num_layers {
            blocks.push(LensTransformerBlock::new(
                inner,
                cfg.num_heads,
                cfg.head_dim,
                mlp_hidden,
                vb.pp("transformer_blocks").pp(i),
            )?);
        }
        Ok(Self {
            img_in: linear(cfg.in_channels, inner, vb.pp("img_in"))?,
            txt_in: linear(cfg.enc_hidden_dim * cfg.num_text_layers, inner, vb.pp("txt_in"))?,
            time_embed: TimeEmbed::new(inner, vb.pp("time_text_embed"))?,
            txt_norm,
            blocks,
            norm_out: NormOut::new(inner, vb.pp("norm_out"))?,
            // proj_out maps to the packed patch velocity (patch²·out_channels = 128 = in_channels).
            proj_out: linear(
                inner,
                cfg.patch_size * cfg.patch_size * cfg.out_channels,
                vb.pp("proj_out"),
            )?,
            rope: LensRope3d::new(10000.0, cfg.axes_dims_rope),
            cfg: *cfg,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Forward.
    ///
    /// - `hidden_states`: `[B, img_len, in_channels]` patchified image latents (`img_len = frame·h·w`).
    /// - `text_feats`: the `num_text_layers` captured gpt-oss layers, each `[B, txt_len, enc_hidden_dim]`.
    /// - `text_valid`: optional `[B, txt_len]` (1 = valid) → additive joint attention mask; `None` =
    ///   all text valid (the single-prompt path).
    /// - `timestep`: `[B]` in `[0, 1]`.
    /// - `(frame, h, w)`: the latent grid shape (`img_len = frame·h·w`).
    ///
    /// Returns `[B, img_len, patch²·out_channels]` (= 128) patch-space velocity.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        text_feats: &[Tensor],
        text_valid: Option<&Tensor>,
        timestep: &Tensor,
        frame: usize,
        h: usize,
        w: usize,
    ) -> Result<Tensor> {
        assert_eq!(
            text_feats.len(),
            self.cfg.num_text_layers,
            "expected {} text-feature layers, got {}",
            self.cfg.num_text_layers,
            text_feats.len()
        );
        let (b, img_len, _) = hidden_states.dims3()?;
        let txt_len = text_feats[0].dim(1)?;

        let mut hidden = self.img_in.forward(hidden_states)?;

        // Multi-layer text front-end: per-layer RMSNorm (eps 1e-5) → channel-concat → txt_in.
        let mut normed = Vec::with_capacity(self.cfg.num_text_layers);
        for (i, feat) in text_feats.iter().enumerate() {
            normed.push(self.txt_norm[i].forward(feat)?);
        }
        let normed_refs: Vec<&Tensor> = normed.iter().collect();
        let mut enc = self.txt_in.forward(&Tensor::cat(&normed_refs, D::Minus1)?)?;

        let ts: Vec<f32> = timestep.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let temb = self.time_embed.forward(&ts, &self.device, self.dtype)?;

        let (img_cos, img_sin, txt_cos, txt_sin) =
            self.rope.forward(frame, h, w, txt_len, &self.device)?;

        let mask = match text_valid {
            Some(valid) => Some(build_joint_mask(valid, img_len, b, self.dtype)?),
            None => None,
        };

        for block in &self.blocks {
            let (e, hs) = block.forward(
                &hidden,
                &enc,
                &temb,
                &img_cos,
                &img_sin,
                &txt_cos,
                &txt_sin,
                mask.as_ref(),
            )?;
            enc = e;
            hidden = hs;
        }

        let hidden = self.norm_out.forward(&hidden, &temb)?;
        self.proj_out.forward(&hidden)
    }
}

/// Additive joint attention mask `[B, 1, 1, img_len + txt_len]`: image tokens always valid; text
/// positions follow `text_valid` (1 = valid). Padded positions get a large-negative additive term so
/// the softmax masks them out (`(valid − 1)·1e9`, valid → 0).
fn build_joint_mask(text_valid: &Tensor, img_len: usize, b: usize, dtype: DType) -> Result<Tensor> {
    let txt_len = text_valid.dim(1)?;
    let dev = text_valid.device();
    let img_ones = Tensor::ones((b, img_len), DType::F32, dev)?;
    let valid = Tensor::cat(&[&img_ones, &text_valid.to_dtype(DType::F32)?], 1)?;
    let additive = ((valid - 1.0)? * 1e9f64)?; // valid → 0, invalid → -1e9
    additive
        .reshape((b, 1, 1, img_len + txt_len))?
        .to_dtype(dtype)
}
