// Vendored verbatim from candle-transformers (sc-11154 / F-081); only
// `scaled_dot_product_attention` is changed (budgeted). Dead-code/clippy allowances keep the
// faithful upstream copy warning-clean under the workspace `-D warnings` gate.
#![allow(dead_code)]
//! Z-Image VAE (AutoEncoderKL) - Diffusers Format
//!
//! This VAE implementation uses the diffusers weight naming format,
//! which is different from the Flux autoencoder original format.
//!
//! Key differences from Flux autoencoder:
//! 1. Weight paths: `encoder.down_blocks.{i}.resnets.{j}.*` vs `encoder.down.{i}.block.{j}.*`
//! 2. Attention naming: `to_q/to_k/to_v/to_out.0.*` vs `q/k/v/proj_out.*`
//! 3. Shortcut naming: `conv_shortcut.*` vs `nin_shortcut.*`

use candle_gen::candle_core::{Module, Result, Tensor, D};
use candle_gen::candle_nn::{conv2d, group_norm, Conv2d, Conv2dConfig, GroupNorm, VarBuilder};

// ==================== Config ====================

/// VAE configuration. The upstream `serde(default=…)` deserialization is dropped in the vendored copy
/// (sc-11154 / F-081) — FLUX.1 only ever builds this via [`VaeConfig::z_image`], never from a JSON
/// component config — so the crate needs no serde-derive wiring.
#[derive(Debug, Clone)]
pub struct VaeConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub latent_channels: usize,
    pub block_out_channels: Vec<usize>,
    pub layers_per_block: usize,
    pub scaling_factor: f64,
    pub shift_factor: f64,
    pub norm_num_groups: usize,
}

impl Default for VaeConfig {
    fn default() -> Self {
        Self::z_image()
    }
}

impl VaeConfig {
    /// Create configuration for Z-Image VAE
    pub fn z_image() -> Self {
        Self {
            in_channels: 3,
            out_channels: 3,
            latent_channels: 16,
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            scaling_factor: 0.3611,
            shift_factor: 0.1159,
            norm_num_groups: 32,
        }
    }
}

// ==================== Attention ====================

fn scaled_dot_product_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    // i32-overflow guard (sc-11154 / F-081): the VAE mid-block is a single-head spatial self-attention
    // over `H·W` tokens, so a 2048² decode (`H·W = 256² = 65536`) materializes a `[B, 1, 65536, 65536]`
    // scores tensor of `≈ 4.3e9 > i32::MAX`. candle's CUDA kernels index scores with i32 and silently
    // corrupt the tail — a garbage decode after a correct denoise. Fold the singleton head into the
    // batch and chunk over the query rows via the shared helper: a single un-chunked pass (byte-identical
    // to the pre-guard `softmax(q·kᵀ·scale)·v`) below budget, exact fused `softmax_last_dim` preserved.
    let dim = q.dim(D::Minus1)?;
    let scale_factor = 1.0 / (dim as f64).sqrt();
    let (b, heads, sq, d) = q.dims4()?;
    let sk = k.dim(2)?;
    let qf = q.reshape((b * heads, sq, d))?;
    let kf = k.reshape((b * heads, sk, d))?;
    let vf = v.reshape((b * heads, sk, d))?;
    let o = candle_gen::sdpa_budgeted_flat(
        &qf,
        &kf,
        &vf,
        scale_factor,
        candle_gen::candle_nn::ops::softmax_last_dim,
        candle_gen::ATTN_SCORES_BUDGET,
    )?;
    o.reshape((b, heads, sq, d))
}

/// VAE Attention block (diffusers format)
///
/// Note: VAE attention uses Linear with bias (2D weight shape)
/// Unlike Transformer attention which uses linear_no_bias
#[derive(Debug, Clone)]
struct Attention {
    group_norm: GroupNorm,
    to_q: candle_gen::candle_nn::Linear,
    to_k: candle_gen::candle_nn::Linear,
    to_v: candle_gen::candle_nn::Linear,
    to_out: candle_gen::candle_nn::Linear,
}

impl Attention {
    fn new(channels: usize, num_groups: usize, vb: VarBuilder) -> Result<Self> {
        let group_norm = group_norm(num_groups, channels, 1e-6, vb.pp("group_norm"))?;
        // VAE attention uses Linear with bias
        let to_q = candle_gen::candle_nn::linear(channels, channels, vb.pp("to_q"))?;
        let to_k = candle_gen::candle_nn::linear(channels, channels, vb.pp("to_k"))?;
        let to_v = candle_gen::candle_nn::linear(channels, channels, vb.pp("to_v"))?;
        let to_out = candle_gen::candle_nn::linear(channels, channels, vb.pp("to_out").pp("0"))?;
        Ok(Self {
            group_norm,
            to_q,
            to_k,
            to_v,
            to_out,
        })
    }
}

impl Module for Attention {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let (b, c, h, w) = xs.dims4()?;

        // GroupNorm
        let xs = xs.apply(&self.group_norm)?;

        // (B, C, H, W) -> (B, H, W, C) -> (B*H*W, C)
        let xs = xs.permute((0, 2, 3, 1))?.reshape((b * h * w, c))?;

        // Linear projections
        let q = xs.apply(&self.to_q)?; // (B*H*W, C)
        let k = xs.apply(&self.to_k)?;
        let v = xs.apply(&self.to_v)?;

        // Reshape for attention: (B*H*W, C) -> (B, H*W, C) -> (B, 1, H*W, C)
        let q = q.reshape((b, h * w, c))?.unsqueeze(1)?;
        let k = k.reshape((b, h * w, c))?.unsqueeze(1)?;
        let v = v.reshape((b, h * w, c))?.unsqueeze(1)?;

        // Scaled dot-product attention
        let xs = scaled_dot_product_attention(&q, &k, &v)?;

        // (B, 1, H*W, C) -> (B*H*W, C)
        let xs = xs.squeeze(1)?.reshape((b * h * w, c))?;

        // Output projection
        let xs = xs.apply(&self.to_out)?;

        // (B*H*W, C) -> (B, H, W, C) -> (B, C, H, W)
        let xs = xs.reshape((b, h, w, c))?.permute((0, 3, 1, 2))?;

        // Residual connection
        xs + residual
    }
}

// ==================== ResnetBlock2D ====================

/// ResNet block (diffusers format)
#[derive(Debug, Clone)]
struct ResnetBlock2D {
    norm1: GroupNorm,
    conv1: Conv2d,
    norm2: GroupNorm,
    conv2: Conv2d,
    conv_shortcut: Option<Conv2d>,
}

impl ResnetBlock2D {
    fn new(
        in_channels: usize,
        out_channels: usize,
        num_groups: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let conv_cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };

        let norm1 = group_norm(num_groups, in_channels, 1e-6, vb.pp("norm1"))?;
        let conv1 = conv2d(in_channels, out_channels, 3, conv_cfg, vb.pp("conv1"))?;
        let norm2 = group_norm(num_groups, out_channels, 1e-6, vb.pp("norm2"))?;
        let conv2 = conv2d(out_channels, out_channels, 3, conv_cfg, vb.pp("conv2"))?;

        let conv_shortcut = if in_channels != out_channels {
            Some(conv2d(
                in_channels,
                out_channels,
                1,
                Default::default(),
                vb.pp("conv_shortcut"),
            )?)
        } else {
            None
        };

        Ok(Self {
            norm1,
            conv1,
            norm2,
            conv2,
            conv_shortcut,
        })
    }
}

impl Module for ResnetBlock2D {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let h = xs
            .apply(&self.norm1)?
            .apply(&candle_gen::candle_nn::Activation::Swish)?
            .apply(&self.conv1)?
            .apply(&self.norm2)?
            .apply(&candle_gen::candle_nn::Activation::Swish)?
            .apply(&self.conv2)?;

        match &self.conv_shortcut {
            Some(conv) => xs.apply(conv)? + h,
            None => xs + h,
        }
    }
}

// ==================== DownEncoderBlock2D ====================

#[derive(Debug, Clone)]
struct Downsample2D {
    conv: Conv2d,
}

impl Downsample2D {
    fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        let conv_cfg = Conv2dConfig {
            stride: 2,
            padding: 0,
            ..Default::default()
        };
        let conv = conv2d(channels, channels, 3, conv_cfg, vb.pp("conv"))?;
        Ok(Self { conv })
    }
}

impl Module for Downsample2D {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // Manual padding: (0, 1, 0, 1) for right=1, bottom=1
        let xs = xs.pad_with_zeros(D::Minus1, 0, 1)?; // width: right
        let xs = xs.pad_with_zeros(D::Minus2, 0, 1)?; // height: bottom
        xs.apply(&self.conv)
    }
}

#[derive(Debug, Clone)]
struct DownEncoderBlock2D {
    resnets: Vec<ResnetBlock2D>,
    downsampler: Option<Downsample2D>,
}

impl DownEncoderBlock2D {
    fn new(
        in_channels: usize,
        out_channels: usize,
        num_layers: usize,
        num_groups: usize,
        add_downsample: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut resnets = Vec::with_capacity(num_layers);
        let vb_resnets = vb.pp("resnets");

        for i in 0..num_layers {
            let in_c = if i == 0 { in_channels } else { out_channels };
            resnets.push(ResnetBlock2D::new(
                in_c,
                out_channels,
                num_groups,
                vb_resnets.pp(i),
            )?);
        }

        let downsampler = if add_downsample {
            Some(Downsample2D::new(
                out_channels,
                vb.pp("downsamplers").pp("0"),
            )?)
        } else {
            None
        };

        Ok(Self {
            resnets,
            downsampler,
        })
    }
}

impl Module for DownEncoderBlock2D {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut h = xs.clone();
        for resnet in &self.resnets {
            h = h.apply(resnet)?;
        }
        if let Some(ds) = &self.downsampler {
            h = h.apply(ds)?;
        }
        Ok(h)
    }
}

// ==================== UpDecoderBlock2D ====================

#[derive(Debug, Clone)]
struct Upsample2D {
    conv: Conv2d,
}

impl Upsample2D {
    fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        let conv_cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let conv = conv2d(channels, channels, 3, conv_cfg, vb.pp("conv"))?;
        Ok(Self { conv })
    }
}

impl Module for Upsample2D {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (_, _, h, w) = xs.dims4()?;
        xs.upsample_nearest2d(h * 2, w * 2)?.apply(&self.conv)
    }
}

#[derive(Debug, Clone)]
struct UpDecoderBlock2D {
    resnets: Vec<ResnetBlock2D>,
    upsampler: Option<Upsample2D>,
}

impl UpDecoderBlock2D {
    fn new(
        in_channels: usize,
        out_channels: usize,
        num_layers: usize, // decoder has num_layers + 1 resnets per block
        num_groups: usize,
        add_upsample: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut resnets = Vec::with_capacity(num_layers + 1);
        let vb_resnets = vb.pp("resnets");

        for i in 0..=num_layers {
            let in_c = if i == 0 { in_channels } else { out_channels };
            resnets.push(ResnetBlock2D::new(
                in_c,
                out_channels,
                num_groups,
                vb_resnets.pp(i),
            )?);
        }

        let upsampler = if add_upsample {
            Some(Upsample2D::new(out_channels, vb.pp("upsamplers").pp("0"))?)
        } else {
            None
        };

        Ok(Self { resnets, upsampler })
    }
}

impl Module for UpDecoderBlock2D {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut h = xs.clone();
        for resnet in &self.resnets {
            h = h.apply(resnet)?;
        }
        if let Some(us) = &self.upsampler {
            h = h.apply(us)?;
        }
        Ok(h)
    }
}

// ==================== UNetMidBlock2D ====================

#[derive(Debug, Clone)]
struct UNetMidBlock2D {
    resnet_0: ResnetBlock2D,
    attention: Attention,
    resnet_1: ResnetBlock2D,
}

impl UNetMidBlock2D {
    fn new(channels: usize, num_groups: usize, vb: VarBuilder) -> Result<Self> {
        let resnet_0 =
            ResnetBlock2D::new(channels, channels, num_groups, vb.pp("resnets").pp("0"))?;
        let attention = Attention::new(channels, num_groups, vb.pp("attentions").pp("0"))?;
        let resnet_1 =
            ResnetBlock2D::new(channels, channels, num_groups, vb.pp("resnets").pp("1"))?;
        Ok(Self {
            resnet_0,
            attention,
            resnet_1,
        })
    }
}

impl Module for UNetMidBlock2D {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.resnet_0)?
            .apply(&self.attention)?
            .apply(&self.resnet_1)
    }
}

// ==================== Encoder ====================

/// VAE Encoder
#[derive(Debug, Clone)]
pub struct Encoder {
    conv_in: Conv2d,
    down_blocks: Vec<DownEncoderBlock2D>,
    mid_block: UNetMidBlock2D,
    conv_norm_out: GroupNorm,
    conv_out: Conv2d,
}

impl Encoder {
    pub fn new(cfg: &VaeConfig, vb: VarBuilder) -> Result<Self> {
        let conv_cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let conv_in = conv2d(
            cfg.in_channels,
            cfg.block_out_channels[0],
            3,
            conv_cfg,
            vb.pp("conv_in"),
        )?;

        let mut down_blocks = Vec::with_capacity(cfg.block_out_channels.len());
        let vb_down = vb.pp("down_blocks");

        for (i, &out_channels) in cfg.block_out_channels.iter().enumerate() {
            let in_channels = if i == 0 {
                cfg.block_out_channels[0]
            } else {
                cfg.block_out_channels[i - 1]
            };
            let add_downsample = i < cfg.block_out_channels.len() - 1;
            down_blocks.push(DownEncoderBlock2D::new(
                in_channels,
                out_channels,
                cfg.layers_per_block,
                cfg.norm_num_groups,
                add_downsample,
                vb_down.pp(i),
            )?);
        }

        let mid_channels = *cfg.block_out_channels.last().unwrap();
        let mid_block = UNetMidBlock2D::new(mid_channels, cfg.norm_num_groups, vb.pp("mid_block"))?;

        let conv_norm_out = group_norm(
            cfg.norm_num_groups,
            mid_channels,
            1e-6,
            vb.pp("conv_norm_out"),
        )?;
        let conv_out = conv2d(
            mid_channels,
            2 * cfg.latent_channels,
            3,
            conv_cfg,
            vb.pp("conv_out"),
        )?;

        Ok(Self {
            conv_in,
            down_blocks,
            mid_block,
            conv_norm_out,
            conv_out,
        })
    }
}

impl Module for Encoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut h = xs.apply(&self.conv_in)?;
        for block in &self.down_blocks {
            h = h.apply(block)?;
        }
        h.apply(&self.mid_block)?
            .apply(&self.conv_norm_out)?
            .apply(&candle_gen::candle_nn::Activation::Swish)?
            .apply(&self.conv_out)
    }
}

// ==================== Decoder ====================

/// VAE Decoder
#[derive(Debug, Clone)]
pub struct Decoder {
    conv_in: Conv2d,
    mid_block: UNetMidBlock2D,
    up_blocks: Vec<UpDecoderBlock2D>,
    conv_norm_out: GroupNorm,
    conv_out: Conv2d,
}

impl Decoder {
    pub fn new(cfg: &VaeConfig, vb: VarBuilder) -> Result<Self> {
        let conv_cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let mid_channels = *cfg.block_out_channels.last().unwrap();

        let conv_in = conv2d(
            cfg.latent_channels,
            mid_channels,
            3,
            conv_cfg,
            vb.pp("conv_in"),
        )?;
        let mid_block = UNetMidBlock2D::new(mid_channels, cfg.norm_num_groups, vb.pp("mid_block"))?;

        // Decoder up_blocks order is reversed from encoder down_blocks
        let reversed_channels: Vec<usize> = cfg.block_out_channels.iter().rev().cloned().collect();
        let mut up_blocks = Vec::with_capacity(reversed_channels.len());
        let vb_up = vb.pp("up_blocks");

        for (i, &out_channels) in reversed_channels.iter().enumerate() {
            let in_channels = if i == 0 {
                mid_channels
            } else {
                reversed_channels[i - 1]
            };
            let add_upsample = i < reversed_channels.len() - 1;
            up_blocks.push(UpDecoderBlock2D::new(
                in_channels,
                out_channels,
                cfg.layers_per_block,
                cfg.norm_num_groups,
                add_upsample,
                vb_up.pp(i),
            )?);
        }

        let final_channels = *reversed_channels.last().unwrap();
        let conv_norm_out = group_norm(
            cfg.norm_num_groups,
            final_channels,
            1e-6,
            vb.pp("conv_norm_out"),
        )?;
        let conv_out = conv2d(
            final_channels,
            cfg.out_channels,
            3,
            conv_cfg,
            vb.pp("conv_out"),
        )?;

        Ok(Self {
            conv_in,
            mid_block,
            up_blocks,
            conv_norm_out,
            conv_out,
        })
    }
}

impl Module for Decoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut h = xs.apply(&self.conv_in)?.apply(&self.mid_block)?;
        for block in &self.up_blocks {
            h = h.apply(block)?;
        }
        h.apply(&self.conv_norm_out)?
            .apply(&candle_gen::candle_nn::Activation::Swish)?
            .apply(&self.conv_out)
    }
}

// ==================== DiagonalGaussian ====================

/// Diagonal Gaussian distribution sampling (VAE reparameterization trick)
#[derive(Debug, Clone)]
pub struct DiagonalGaussian {
    sample: bool,
}

impl DiagonalGaussian {
    pub fn new(sample: bool) -> Self {
        Self { sample }
    }
}

impl Module for DiagonalGaussian {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let chunks = xs.chunk(2, 1)?; // Split along channel dimension
        let mean = &chunks[0];
        let logvar = &chunks[1];

        if self.sample {
            let std = (logvar * 0.5)?.exp()?;
            mean + (std * mean.randn_like(0., 1.)?)?
        } else {
            Ok(mean.clone())
        }
    }
}

// ==================== AutoEncoderKL ====================

/// Z-Image VAE (AutoEncoderKL) - Diffusers Format
#[derive(Debug, Clone)]
pub struct AutoEncoderKL {
    encoder: Encoder,
    decoder: Decoder,
    reg: DiagonalGaussian,
    scale_factor: f64,
    shift_factor: f64,
}

impl AutoEncoderKL {
    pub fn new(cfg: &VaeConfig, vb: VarBuilder) -> Result<Self> {
        let encoder = Encoder::new(cfg, vb.pp("encoder"))?;
        let decoder = Decoder::new(cfg, vb.pp("decoder"))?;
        let reg = DiagonalGaussian::new(true);

        Ok(Self {
            encoder,
            decoder,
            reg,
            scale_factor: cfg.scaling_factor,
            shift_factor: cfg.shift_factor,
        })
    }

    /// Encode image to latent space
    /// xs: (B, 3, H, W) RGB image, range [-1, 1]
    /// Returns: (B, latent_channels, H/8, W/8)
    pub fn encode(&self, xs: &Tensor) -> Result<Tensor> {
        let z = xs.apply(&self.encoder)?.apply(&self.reg)?;
        (z - self.shift_factor)? * self.scale_factor
    }

    /// Decode latent to image
    /// xs: (B, latent_channels, H/8, W/8)
    /// Returns: (B, 3, H, W) RGB image, range [-1, 1]
    pub fn decode(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = ((xs / self.scale_factor)? + self.shift_factor)?;
        xs.apply(&self.decoder)
    }

    /// Get scaling factor
    pub fn scale_factor(&self) -> f64 {
        self.scale_factor
    }

    /// Get shift factor
    pub fn shift_factor(&self) -> f64 {
        self.shift_factor
    }
}

impl Module for AutoEncoderKL {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.decode(&self.encode(xs)?)
    }
}

#[cfg(test)]
mod parity_tests {
    //! Pin the vendored diffusers FLUX VAE (`AutoEncoderKL`) to the stock candle-transformers
    //! `z_image::vae` module (sc-11154 / F-081): built from the *same* `VarMap`-backed weights, the
    //! deterministic `decode` forward must be bit-identical, proving the only edited line
    //! (`scaled_dot_product_attention`, a single un-chunked pass below `ATTN_SCORES_BUDGET`) changed
    //! nothing numerically.
    use super::{AutoEncoderKL, VaeConfig};
    use candle_gen::candle_core::{DType, Device, Tensor};
    use candle_gen::candle_nn::{VarBuilder, VarMap};
    use candle_transformers::models::z_image::vae as stock;

    #[test]
    fn vendored_flux_diffusers_vae_decode_matches_stock() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let cfg = VaeConfig::z_image();
        let vendored = AutoEncoderKL::new(&cfg, vb.clone()).unwrap();
        let stock_vae = stock::AutoEncoderKL::new(&stock::VaeConfig::z_image(), vb).unwrap();

        let z = Tensor::randn(0f32, 1f32, (1, cfg.latent_channels, 8, 8), &dev).unwrap();
        let y_v = vendored.decode(&z).unwrap();
        let y_s = stock_vae.decode(&z).unwrap();
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
            "vendored FLUX diffusers VAE decode diverged from stock by {diff}"
        );
    }
}
