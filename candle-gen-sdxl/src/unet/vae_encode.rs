//! Vendored, **training-only** SDXL VAE moments-encoder (sc-5165).
//!
//! The trainer caches each image's clean latent `x0 = mean(VAE.encode(image)) × 0.13025`. candle's
//! stock `AutoEncoderKL` ([`candle_transformers`]) exposes only `encode() -> DiagonalGaussianDistribution`
//! whose `mean` field is **private** — the one public accessor, `sample()`, draws candle's device RNG
//! (non-portable, the very thing sc-3673 banned) and adds the sampling noise we explicitly do **not**
//! want (the user's deterministic-`.mean` decision). And `encoder`/`quant_conv` are private too, so
//! there is no public path to the moments. We pin candle (not fork), so the fix is to vendor the
//! encode path — a byte-faithful copy of candle's `vae::Encoder::{new,forward}` + `quant_conv`, reusing
//! the VAE building blocks already vendored for the UNet ([`DownEncoderBlock2D`]/[`UNetMidBlock2D`]).
//!
//! This is **encode-only and frozen**: no adapter, no gradient checkpointing, no decoder (the trainer
//! never decodes). Inference's VAE *decode* stays the stock `AutoEncoderKL`, untouched. Loaded f32 for
//! a clean latent mean; the latents are cached once then frozen.

use candle_core::{Result, Tensor};
use candle_nn::{self as nn, Module};

use super::unet_2d_blocks::{
    DownEncoderBlock2D, DownEncoderBlock2DConfig, UNetMidBlock2D, UNetMidBlock2DConfig,
};

/// SDXL VAE config (`stabilityai/stable-diffusion-xl-base-1.0/vae/config.json`, shared by the
/// fp16-fix VAE): 4 down stages 128/256/512/512, 2 layers each, 4 latent channels, 32 group-norm
/// groups, with the `quant_conv` head. Matches candle's `AutoEncoderKLConfig` for SDXL.
const BLOCK_OUT_CHANNELS: [usize; 4] = [128, 256, 512, 512];
const LAYERS_PER_BLOCK: usize = 2;
const LATENT_CHANNELS: usize = 4;
const NORM_NUM_GROUPS: usize = 32;
const IN_CHANNELS: usize = 3;

/// The frozen VAE encode path that yields the **deterministic latent mean** (scaled). A faithful
/// replica of candle's private `vae::Encoder` + `quant_conv`, so it loads the stock VAE safetensors
/// (`encoder.*` / `quant_conv.*` keys) unchanged.
#[derive(Debug)]
pub struct VaeMomentsEncoder {
    conv_in: nn::Conv2d,
    down_blocks: Vec<DownEncoderBlock2D>,
    mid_block: UNetMidBlock2D,
    conv_norm_out: nn::GroupNorm,
    conv_out: nn::Conv2d,
    quant_conv: Option<nn::Conv2d>,
    scale: f64,
}

impl VaeMomentsEncoder {
    /// Build from a `VarBuilder` over the SDXL VAE safetensors, at the given residual `scale`
    /// (`0.13025` for SDXL). `vs` is the root: the encoder reads `vs.pp("encoder")` and the head reads
    /// `vs.pp("quant_conv")`, exactly as candle's `AutoEncoderKL::new`.
    pub fn new(vs: nn::VarBuilder, scale: f64) -> Result<Self> {
        let conv_cfg = nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let vs_enc = vs.pp("encoder");
        let conv_in = nn::conv2d(
            IN_CHANNELS,
            BLOCK_OUT_CHANNELS[0],
            3,
            conv_cfg,
            vs_enc.pp("conv_in"),
        )?;

        let mut down_blocks = Vec::with_capacity(BLOCK_OUT_CHANNELS.len());
        let vs_db = vs_enc.pp("down_blocks");
        for index in 0..BLOCK_OUT_CHANNELS.len() {
            let out_channels = BLOCK_OUT_CHANNELS[index];
            let in_channels = if index > 0 {
                BLOCK_OUT_CHANNELS[index - 1]
            } else {
                BLOCK_OUT_CHANNELS[0]
            };
            let is_final = index + 1 == BLOCK_OUT_CHANNELS.len();
            let cfg = DownEncoderBlock2DConfig {
                num_layers: LAYERS_PER_BLOCK,
                resnet_eps: 1e-6,
                resnet_groups: NORM_NUM_GROUPS,
                add_downsample: !is_final,
                downsample_padding: 0,
                ..Default::default()
            };
            down_blocks.push(DownEncoderBlock2D::new(
                vs_db.pp(index.to_string()),
                in_channels,
                out_channels,
                cfg,
            )?);
        }

        let last = *BLOCK_OUT_CHANNELS.last().unwrap();
        let mid_cfg = UNetMidBlock2DConfig {
            resnet_eps: 1e-6,
            output_scale_factor: 1.,
            attn_num_head_channels: None,
            resnet_groups: Some(NORM_NUM_GROUPS),
            ..Default::default()
        };
        let mid_block = UNetMidBlock2D::new(vs_enc.pp("mid_block"), last, None, mid_cfg)?;
        let conv_norm_out =
            nn::group_norm(NORM_NUM_GROUPS, last, 1e-6, vs_enc.pp("conv_norm_out"))?;
        // `double_z`: the encoder emits 2·latent channels (mean ‖ logvar).
        let conv_out = nn::conv2d(
            last,
            2 * LATENT_CHANNELS,
            3,
            conv_cfg,
            vs_enc.pp("conv_out"),
        )?;
        let quant_conv = Some(nn::conv2d(
            2 * LATENT_CHANNELS,
            2 * LATENT_CHANNELS,
            1,
            Default::default(),
            vs.pp("quant_conv"),
        )?);

        Ok(Self {
            conv_in,
            down_blocks,
            mid_block,
            conv_norm_out,
            conv_out,
            quant_conv,
            scale,
        })
    }

    /// Encode a VAE-input image `[B, 3, H, W]` (RGB, `[-1, 1]`) to the **scaled deterministic latent
    /// mean** `[B, 4, H/8, W/8]` = `mean(moments) × scale`. Faithful to candle's `Encoder::forward` +
    /// `quant_conv`, then the diffusers `DiagonalGaussianDistribution` mean = `moments.chunk(2)[0]`.
    /// No sampling — deterministic, launch-portable (no device RNG).
    pub fn encode_mean(&self, x: &Tensor) -> Result<Tensor> {
        let mut xs = x.apply(&self.conv_in)?;
        for down_block in self.down_blocks.iter() {
            xs = xs.apply(down_block)?;
        }
        let xs = self
            .mid_block
            .forward(&xs, None)?
            .apply(&self.conv_norm_out)?;
        let xs = nn::ops::silu(&xs)?.apply(&self.conv_out)?;
        let parameters = match &self.quant_conv {
            Some(quant_conv) => quant_conv.forward(&xs)?,
            None => xs,
        };
        // [mean ‖ logvar] split on the channel axis; keep the mean half.
        let mean = parameters
            .chunk(2, 1)?
            .into_iter()
            .next()
            .expect("chunk(2) yields two halves");
        mean * self.scale
    }
}
