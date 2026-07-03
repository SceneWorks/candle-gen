//! Chroma's **AutoencoderKL** decoder — the candle (Windows/CUDA) port, decode-only (txt2img).
//!
//! Chroma reuses the FLUX 16-channel AutoencoderKL (the mlx provider literally calls
//! `mlx_gen_flux::load_vae`). It is a *standard* diffusers AutoencoderKL — unlike `candle-gen-flux2`'s
//! VAE there is **no** 2×2 pack and **no** BatchNorm-stats space (the 2×2 patchify lives between the
//! DiT's packed tokens and this 16-ch latent and is handled by `flux::sampling::unpack` in the
//! pipeline). The only deltas vs. the flux2 decoder this is adapted from are `LATENT_CHANNELS 32→16`
//! and the diffusers `z/scaling + shift` un-scale (FLUX `scaling_factor = 0.3611`,
//! `shift_factor = 0.1159`), folded into [`Vae::decode`] (pipeline-unscale + optional
//! `post_quant_conv` + decoder). The real Chroma/FLUX.1 16-ch VAE ships **no** `post_quant_conv`
//! (matching the mlx parity reference `mlx_gen_flux::load_vae`), so it is loaded/applied only when a
//! snapshot happens to carry one — see [`Vae`].
//!
//! The decoder tree is the diffusers layout: `conv_in → mid(resnet, attn, resnet) → 4 up_blocks
//! (3 resnets each, upsampler on all but the last) → groupnorm/silu → conv_out`. GroupNorm eps 1e-6,
//! single-head mid attention. Runs in candle-native NCHW f32.

use candle_gen::candle_core::{DType, Result, Tensor};
use candle_gen::candle_nn::{
    conv2d, group_norm, linear, ops::softmax_last_dim, Conv2d, Conv2dConfig, GroupNorm, Linear,
    Module, VarBuilder,
};

const GN_GROUPS: usize = 32;
const GN_EPS: f64 = 1e-6;
const BLOCK_OUT: [usize; 4] = [128, 256, 512, 512];
const LATENT_CHANNELS: usize = 16;
/// Decoder up_blocks have `layers_per_block + 1 = 3` resnets each.
const DECODER_RESNETS: usize = 3;
/// FLUX AutoencoderKL scaling/shift (the Chroma `vae/config.json` values). The DiT works in the
/// pre-scaled latent; decode un-scales `z/scale + shift` before the conv decoder.
const SCALING_FACTOR: f64 = 0.3611;
const SHIFT_FACTOR: f64 = 0.1159;

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
        let shortcut = if in_c != out_c {
            Some(conv1x1(in_c, out_c, vb.pp("conv_shortcut"))?)
        } else {
            None
        };
        Ok(Self {
            norm1: group_norm(GN_GROUPS, in_c, GN_EPS, vb.pp("norm1"))?,
            conv1: conv3x3(in_c, out_c, vb.pp("conv1"))?,
            norm2: group_norm(GN_GROUPS, out_c, GN_EPS, vb.pp("norm2"))?,
            conv2: conv3x3(out_c, out_c, vb.pp("conv2"))?,
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
        // i32-overflow guard (sc-9116): the single-head spatial scores `[B, HW, HW]` reach `65536² ≈
        // 4.3e9 > i32::MAX` at a 2048² decode, so chunk over the query rows (byte-identical for the
        // common sizes). Shared helper; softmax closure keeps the exact fused `softmax_last_dim`.
        let o = candle_gen::sdpa_budgeted_flat(
            &q,
            &k,
            &v,
            scale,
            softmax_last_dim,
            candle_gen::ATTN_SCORES_BUDGET,
        )?; // (B, HW, C)
        let o = self.out.forward(&o)?;
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

/// The Chroma/FLUX 16-channel AutoencoderKL — decode-only (txt2img). Build from a `vae/` VarBuilder
/// (diffusers AutoencoderKL keys, f32).
pub struct Vae {
    /// The `post_quant_conv` 1×1 conv, **when the snapshot ships one**. The real Chroma VAE
    /// (`lodestones/Chroma1-*`, mirrored verbatim into the `SceneWorks/chroma1-*-mlx` tiers) is a FLUX.1
    /// 16-ch AutoencoderKL that carries **no** `post_quant_conv` — matching the mlx parity reference
    /// (`mlx_gen_flux::load_vae`), whose decode never applies one. It is loaded (and applied) only if
    /// present, so a snapshot that does ship it stays byte-exact and one that omits it decodes directly
    /// from the un-scaled latent (sc-9409: unblocks the real Chroma render, which previously errored on
    /// the missing `post_quant_conv.weight` key).
    post_quant_conv: Option<Conv2d>,
    conv_in: Conv2d,
    mid_resnet0: Resnet,
    mid_attn: MidAttention,
    mid_resnet1: Resnet,
    up_blocks: Vec<UpBlock>,
    conv_norm_out: GroupNorm,
    conv_out: Conv2d,
}

impl Vae {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        // Optional: the real Chroma/FLUX.1 16-ch VAE ships no `post_quant_conv` (see the field docs).
        let post_quant_conv = if vb.contains_tensor("post_quant_conv.weight") {
            Some(conv1x1(
                LATENT_CHANNELS,
                LATENT_CHANNELS,
                vb.pp("post_quant_conv"),
            )?)
        } else {
            None
        };
        let dec = vb.pp("decoder");
        let top = *BLOCK_OUT.last().unwrap(); // 512
        let conv_in = conv3x3(LATENT_CHANNELS, top, dec.pp("conv_in"))?;

        let mid = dec.pp("mid_block");
        let mid_resnet0 = Resnet::new(top, top, mid.pp("resnets").pp("0"))?;
        let mid_attn = MidAttention::new(top, mid.pp("attentions").pp("0"))?;
        let mid_resnet1 = Resnet::new(top, top, mid.pp("resnets").pp("1"))?;

        // Decoder up_blocks iterate the reversed block_out channels [512,512,256,128].
        let reversed: Vec<usize> = BLOCK_OUT.iter().rev().copied().collect();
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

    /// Decode a 16-channel latent `[1, 16, H/8, W/8]` (NCHW, the pre-scaled diffusion latent) → an RGB
    /// image `[1, 3, H, W]` in `[-1, 1]`. Folds the diffusers pipeline un-scale (`z/scale + shift`)
    /// into the VAE decode (`post_quant_conv → decoder`), matching candle's `flux::autoencoder`.
    pub fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        let z = ((latents.to_dtype(DType::F32)? / SCALING_FACTOR)? + SHIFT_FACTOR)?;
        // `post_quant_conv` only when the snapshot ships it (the real Chroma VAE omits it — the mlx
        // parity reference likewise never applies one).
        let z = match &self.post_quant_conv {
            Some(pqc) => pqc.forward(&z)?,
            None => z,
        };
        let mut h = self.conv_in.forward(&z)?;
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
