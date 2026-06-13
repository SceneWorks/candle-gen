//! The FLUX.2 **AutoencoderKL-Flux2** decoder (decode-only — txt2img needs no encode). Ported from
//! `mlx-gen-flux2`'s `vae.rs`, run in candle-native **NCHW** f32.
//!
//! FLUX.2 differs from a plain AutoencoderKL in two ways, both **outside** the conv decoder:
//! 1. **2×2 pack/patchify** between the 32-ch VAE latent and the 128-ch transformer space — the
//!    decoder itself is a standard 32-ch AutoencoderKL; the unpatchify (`[B,128,h,w] → [B,32,2h,2w]`)
//!    happens here before `decode`.
//! 2. **BatchNorm-stats normalization** of the packed 128-ch space (`bn.running_mean/var`): the
//!    transformer works in a `(x − mean)/std`-normalized space, so decode first **de-normalizes**
//!    `x·std + mean` (`std = sqrt(running_var + 1e-4)`).
//!
//! `scaling_factor = 1.0`, `shift_factor = 0.0` (identity), so there is no `z/scale + shift` step —
//! the bn-stats step replaces it. GroupNorm eps is **1e-6** (not SDXL's 1e-5). The decoder is the
//! diffusers tree: `conv_in → mid(resnet,attn,resnet) → 4 up_blocks(3 resnets each, 3 upsamplers) →
//! groupnorm/silu → conv_out`.

use candle_gen::candle_core::{DType, Result, Tensor};
use candle_gen::candle_nn::{
    conv2d, group_norm, linear, Conv2d, Conv2dConfig, GroupNorm, Linear, Module, VarBuilder,
};

const GN_GROUPS: usize = 32;
const GN_EPS: f64 = 1e-6;
const BN_EPS: f64 = 1e-4;
const BLOCK_OUT: [usize; 4] = [128, 256, 512, 512];
const LATENT_CHANNELS: usize = 32;
/// Decoder up_blocks have `layers_per_block + 1 = 3` resnets each.
const DECODER_RESNETS: usize = 3;

fn conv3x3(in_c: usize, out_c: usize, vb: VarBuilder) -> Result<Conv2d> {
    let cfg = Conv2dConfig {
        padding: 1,
        stride: 1,
        ..Default::default()
    };
    conv2d(in_c, out_c, 3, cfg, vb)
}

fn conv1x1(in_c: usize, out_c: usize, vb: VarBuilder) -> Result<Conv2d> {
    conv2d(in_c, out_c, 1, Conv2dConfig::default(), vb)
}

/// Diffusers ResnetBlock2D (temb-free): `gn→silu→conv1 → gn→silu→conv2 + shortcut`.
struct Resnet {
    norm1: GroupNorm,
    conv1: Conv2d,
    norm2: GroupNorm,
    conv2: Conv2d,
    shortcut: Option<Conv2d>,
}

impl Resnet {
    fn new(in_c: usize, out_c: usize, vb: VarBuilder) -> Result<Self> {
        let norm1 = group_norm(GN_GROUPS, in_c, GN_EPS, vb.pp("norm1"))?;
        let conv1 = conv3x3(in_c, out_c, vb.pp("conv1"))?;
        let norm2 = group_norm(GN_GROUPS, out_c, GN_EPS, vb.pp("norm2"))?;
        let conv2 = conv3x3(out_c, out_c, vb.pp("conv2"))?;
        let shortcut = if in_c != out_c {
            Some(conv1x1(in_c, out_c, vb.pp("conv_shortcut"))?)
        } else {
            None
        };
        Ok(Self {
            norm1,
            conv1,
            norm2,
            conv2,
            shortcut,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.norm1.forward(x)?.silu()?;
        let h = self.conv1.forward(&h)?;
        let h = self.norm2.forward(&h)?.silu()?;
        let h = self.conv2.forward(&h)?;
        let res = match &self.shortcut {
            Some(c) => c.forward(x)?,
            None => x.clone(),
        };
        h + res
    }
}

/// Single-head spatial self-attention in the mid block (diffusers `Attention`).
struct MidAttention {
    norm: GroupNorm,
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    channels: usize,
}

impl MidAttention {
    fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm: group_norm(GN_GROUPS, channels, GN_EPS, vb.pp("group_norm"))?,
            q: linear(channels, channels, vb.pp("to_q"))?,
            k: linear(channels, channels, vb.pp("to_k"))?,
            v: linear(channels, channels, vb.pp("to_v"))?,
            out: linear(channels, channels, vb.pp("to_out").pp("0"))?,
            channels,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, h, w) = x.dims4()?;
        let normed = self.norm.forward(x)?;
        // (B, C, H, W) -> (B, H*W, C)
        let seq = normed
            .reshape((b, c, h * w))?
            .transpose(1, 2)?
            .contiguous()?;
        let q = self.q.forward(&seq)?;
        let k = self.k.forward(&seq)?;
        let v = self.v.forward(&seq)?;
        let scale = (self.channels as f64).powf(-0.5);
        let attn = (q.matmul(&k.transpose(1, 2)?)? * scale)?;
        let attn = candle_gen::candle_nn::ops::softmax_last_dim(&attn)?;
        let o = attn.matmul(&v)?; // (B, HW, C)
        let o = self.out.forward(&o)?;
        // back to (B, C, H, W) and residual
        let o = o.transpose(1, 2)?.reshape((b, c, h, w))?;
        x + o
    }
}

/// Nearest-2× upsample + 3×3 conv.
struct Upsampler {
    conv: Conv2d,
}

impl Upsampler {
    fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            conv: conv3x3(channels, channels, vb.pp("conv"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_, _, h, w) = x.dims4()?;
        let up = x.upsample_nearest2d(h * 2, w * 2)?;
        self.conv.forward(&up)
    }
}

struct UpBlock {
    resnets: Vec<Resnet>,
    upsampler: Option<Upsampler>,
}

impl UpBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = x.clone();
        for r in &self.resnets {
            h = r.forward(&h)?;
        }
        if let Some(u) = &self.upsampler {
            h = u.forward(&h)?;
        }
        Ok(h)
    }
}

/// The FLUX.2 VAE: decode-only, plus the bn-stats de-normalization and 2×2 unpatchify wrapper.
pub struct Flux2Vae {
    bn_mean: Tensor, // [1,128,1,1]
    bn_std: Tensor,  // [1,128,1,1]
    post_quant_conv: Conv2d,
    conv_in: Conv2d,
    mid_resnet0: Resnet,
    mid_attn: MidAttention,
    mid_resnet1: Resnet,
    up_blocks: Vec<UpBlock>,
    conv_norm_out: GroupNorm,
    conv_out: Conv2d,
}

impl Flux2Vae {
    /// Build from a `vae/` VarBuilder (diffusers AutoencoderKLFlux2 keys, f32).
    pub fn new(vb: VarBuilder) -> Result<Self> {
        // bn stats live at the top level (packed 128-ch space).
        let bn_mean = vb.get(128, "bn.running_mean")?.reshape((1, 128, 1, 1))?;
        let bn_var = vb.get(128, "bn.running_var")?;
        let bn_std = (bn_var + BN_EPS)?.sqrt()?.reshape((1, 128, 1, 1))?;

        let post_quant_conv = conv1x1(LATENT_CHANNELS, LATENT_CHANNELS, vb.pp("post_quant_conv"))?;

        let dec = vb.pp("decoder");
        let top = *BLOCK_OUT.last().unwrap(); // 512
        let conv_in = conv3x3(LATENT_CHANNELS, top, dec.pp("conv_in"))?;

        let mid = dec.pp("mid_block");
        let mid_resnet0 = Resnet::new(top, top, mid.pp("resnets").pp("0"))?;
        let mid_attn = MidAttention::new(top, mid.pp("attentions").pp("0"))?;
        let mid_resnet1 = Resnet::new(top, top, mid.pp("resnets").pp("1"))?;

        // Decoder up_blocks iterate the reversed block_out channels.
        let reversed: Vec<usize> = BLOCK_OUT.iter().rev().copied().collect(); // [512,512,256,128]
        let mut up_blocks = Vec::with_capacity(reversed.len());
        let mut prev = top;
        for (i, &out_c) in reversed.iter().enumerate() {
            let ub = dec.pp("up_blocks").pp(i);
            let mut resnets = Vec::with_capacity(DECODER_RESNETS);
            for j in 0..DECODER_RESNETS {
                let in_c = if j == 0 { prev } else { out_c };
                resnets.push(Resnet::new(in_c, out_c, ub.pp("resnets").pp(j))?);
            }
            let is_final = i == reversed.len() - 1;
            let upsampler = if is_final {
                None
            } else {
                Some(Upsampler::new(out_c, ub.pp("upsamplers").pp("0"))?)
            };
            up_blocks.push(UpBlock { resnets, upsampler });
            prev = out_c;
        }

        let conv_norm_out = group_norm(GN_GROUPS, prev, GN_EPS, dec.pp("conv_norm_out"))?;
        let conv_out = conv3x3(prev, 3, dec.pp("conv_out"))?;

        Ok(Self {
            bn_mean,
            bn_std,
            post_quant_conv,
            conv_in,
            mid_resnet0,
            mid_attn,
            mid_resnet1,
            up_blocks,
            conv_norm_out,
            conv_out,
        })
    }

    /// Decode packed transformer latents `[1, 128, lat_h, lat_w]` (NCHW) → RGB image `[1, 3, H, W]`
    /// in `[-1, 1]`. De-normalizes the bn-stats space, unpatchifies 128→32 (doubling spatial), then
    /// runs the standard AutoencoderKL decode.
    pub fn decode_packed(&self, packed: &Tensor) -> Result<Tensor> {
        let packed = packed.to_dtype(DType::F32)?;
        // De-normalize: x·std + mean (broadcast over [1,128,1,1]).
        let denorm = packed
            .broadcast_mul(&self.bn_std)?
            .broadcast_add(&self.bn_mean)?;
        let latents = unpatchify(&denorm)?; // [1, 32, 2h, 2w]
        let z = self.post_quant_conv.forward(&latents)?;
        self.decode(&z)
    }

    fn decode(&self, z: &Tensor) -> Result<Tensor> {
        let mut h = self.conv_in.forward(z)?;
        h = self.mid_resnet0.forward(&h)?;
        h = self.mid_attn.forward(&h)?;
        h = self.mid_resnet1.forward(&h)?;
        for ub in &self.up_blocks {
            h = ub.forward(&h)?;
        }
        let h = self.conv_norm_out.forward(&h)?.silu()?;
        self.conv_out.forward(&h)
    }
}

/// 2×2 unpatchify (NCHW): `[B, 128, h, w] → [B, 32, 2h, 2w]`. The 128 channel axis splits as
/// `(c=32, ph=2, pw=2)` (c outermost), matching the fork's channel order `c·4 + ph·2 + pw`.
fn unpatchify(x: &Tensor) -> Result<Tensor> {
    let (b, c, h, w) = x.dims4()?;
    let c4 = c / 4;
    // [B,128,h,w] -> [B, c4, 2, 2, h, w] -> [B, c4, h, 2, w, 2] -> [B, c4, 2h, 2w]
    x.reshape((b, c4, 2, 2, h, w))?
        .permute((0, 1, 4, 2, 5, 3))?
        .reshape((b, c4, h * 2, w * 2))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Device;

    /// The 2×2 unpatchify channel/spatial order matches the fork's `c·4 + ph·2 + pw` pinning.
    #[test]
    fn unpatchify_channel_spatial_order() {
        // Build [1, 8, 1, 1] where channel value = c*4+ph*2+pw for c4=2.
        let data: Vec<f32> = (0..8).map(|x| x as f32).collect();
        let x = Tensor::from_vec(data, (1, 8, 1, 1), &Device::Cpu).unwrap();
        let out = unpatchify(&x).unwrap(); // [1, 2, 2, 2]
        assert_eq!(out.dims(), &[1, 2, 2, 2]);
        let v = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // out[0, c, ph, pw] == c*4 + ph*2 + pw. Flattened over (c, ph, pw) row-major:
        // c=0: (0,0)=0,(0,1)=1,(1,0)=2,(1,1)=3 ; c=1: 4,5,6,7
        assert_eq!(v, vec![0., 1., 2., 3., 4., 5., 6., 7.]);
    }
}
