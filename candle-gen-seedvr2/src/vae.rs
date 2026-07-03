//! SeedVR2 3D causal video VAE — candle port of `mlx-gen-seedvr2/src/vae.rs`.
//!
//! Encoder maps `(B,3,T,H,W)` → `(B,16,T',H',W')` (spatial /8; temporal `ceil(T/4)` via two
//! temporal-stride down blocks); decoder inverts it. Conv weights are torch `[O,I,kT,kH,kW]` (loaded
//! as-is — [`CausalConv3d`] uses the per-temporal-tap slice as a conv2d kernel). GroupNorm (32 groups,
//! eps 1e-6) runs in f32. Parity-critical details (causal first-frame pad, asymmetric down pad, the
//! pixel-shuffle upsampler) mirror the reference.

use candle_gen::candle_core::{Result, Tensor};

use crate::config::VaeConfig;
use crate::conv3d::CausalConv3d;
use crate::nn;
use crate::weights::Weights;

type CResult<T> = candle_gen::Result<T>;

/// `[out,in]`-weight dense layer (the VAE attention projections). Stores the weight pre-transposed to
/// `[in,out]` so the per-forward matmul has no transpose/copy (sc-8997/F-017).
struct Linear {
    wt: Tensor,
    b: Tensor,
}
impl Linear {
    fn load(w: &Weights, prefix: &str) -> CResult<Self> {
        let weight = w.require(&format!("{prefix}.weight"))?; // [out, in]
        Ok(Self {
            wt: nn::transpose_weight(weight)?, // [in, out], once at load (sc-8997/F-017)
            b: w.require(&format!("{prefix}.bias"))?.clone(),
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        nn::linear(x, &self.wt, Some(&self.b))
    }
}

/// GroupNorm over an NCTHW tensor (channels in dim 1), f32 — the candle twin of the reference `gn`.
fn gn(x: &Tensor, w: &Tensor, b: &Tensor, groups: usize, eps: f64) -> Result<Tensor> {
    nn::group_norm(x, w, b, groups, eps)
}

/// `norm1 → silu → conv1 → norm2 → silu → conv2` + (1³ conv) skip when channels differ.
struct ResnetBlock3d {
    norm1_w: Tensor,
    norm1_b: Tensor,
    conv1: CausalConv3d,
    norm2_w: Tensor,
    norm2_b: Tensor,
    conv2: CausalConv3d,
    shortcut: Option<CausalConv3d>,
    groups: usize,
    eps: f64,
}
impl ResnetBlock3d {
    fn load(w: &Weights, prefix: &str, cfg: &VaeConfig) -> CResult<Self> {
        let shortcut = if w.get(&format!("{prefix}.conv_shortcut.weight")).is_some() {
            Some(CausalConv3d::load(
                w,
                &format!("{prefix}.conv_shortcut"),
                (1, 1, 1),
                (0, 0, 0),
                false,
            )?)
        } else {
            None
        };
        Ok(Self {
            norm1_w: w.require(&format!("{prefix}.norm1.weight"))?.clone(),
            norm1_b: w.require(&format!("{prefix}.norm1.bias"))?.clone(),
            conv1: CausalConv3d::load(w, &format!("{prefix}.conv1"), (1, 1, 1), (1, 1, 1), false)?,
            norm2_w: w.require(&format!("{prefix}.norm2.weight"))?.clone(),
            norm2_b: w.require(&format!("{prefix}.norm2.bias"))?.clone(),
            conv2: CausalConv3d::load(w, &format!("{prefix}.conv2"), (1, 1, 1), (1, 1, 1), false)?,
            shortcut,
            groups: cfg.group_norm_groups,
            eps: cfg.group_norm_eps,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = match &self.shortcut {
            Some(s) => s.forward(x)?,
            None => x.clone(),
        };
        let h = gn(x, &self.norm1_w, &self.norm1_b, self.groups, self.eps)?;
        let h = self.conv1.forward(&nn::silu(&h)?)?;
        let h = gn(&h, &self.norm2_w, &self.norm2_b, self.groups, self.eps)?;
        let h = self.conv2.forward(&nn::silu(&h)?)?;
        h + residual
    }
}

/// Per-frame single-head spatial self-attention (head_dim = C). NCTHW I/O.
struct Attention3d {
    gn_w: Tensor,
    gn_b: Tensor,
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    groups: usize,
    eps: f64,
}
impl Attention3d {
    fn load(w: &Weights, prefix: &str, cfg: &VaeConfig) -> CResult<Self> {
        Ok(Self {
            gn_w: w.require(&format!("{prefix}.group_norm.weight"))?.clone(),
            gn_b: w.require(&format!("{prefix}.group_norm.bias"))?.clone(),
            to_q: Linear::load(w, &format!("{prefix}.to_q"))?,
            to_k: Linear::load(w, &format!("{prefix}.to_k"))?,
            to_v: Linear::load(w, &format!("{prefix}.to_v"))?,
            to_out: Linear::load(w, &format!("{prefix}.to_out.0"))?,
            groups: cfg.group_norm_groups,
            eps: cfg.group_norm_eps,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, t, h, wd) = x.dims5()?;
        let residual = x.clone();
        // (B,C,T,H,W) -> (B,T,C,H,W) -> (B*T, C, H*W)
        let xs = x
            .permute((0, 2, 1, 3, 4))?
            .contiguous()?
            .reshape((b * t, c, h * wd))?;
        // GroupNorm over channels (dim 1), then to channels-last for the projections.
        let xn = gn(&xs, &self.gn_w, &self.gn_b, self.groups, self.eps)?
            .transpose(1, 2)?
            .contiguous()?; // [B*T, H*W, C]

        // Single-head full-frame spatial self-attention: q/k/v are `[B*T, H*W, C]`.
        let q = self.to_q.forward(&xn)?; // [B*T, H*W, C]
        let k = self.to_k.forward(&xn)?;
        let v = self.to_v.forward(&xn)?;
        let scale = (c as f64).powf(-0.5);
        // i32-overflow guard (sc-9116): this is the seedvr2 *VAE* mid-block — FULL-FRAME (not the DiT's
        // windowed attn), and decode is NOT tiled. seedvr2 is an upscaler (max_size 4096, spatial_scale
        // 8 → latent side up to 512 → HW up to 262144), so the single-head scores `[B*T, HW, HW]` reach
        // `262144² ≈ 6.9e10 ≫ i32::MAX` (overflows above ~1650² output), silently corrupting the tail
        // rows on the candle CUDA kernels. Chunk over the query rows (byte-identical below budget); the
        // softmax closure matches `nn::sdpa`'s exact fused `softmax_last_dim`.
        let o = candle_gen::sdpa_budgeted_flat(
            &q,
            &k,
            &v,
            scale,
            candle_gen::candle_nn::ops::softmax_last_dim,
            candle_gen::ATTN_SCORES_BUDGET,
        )?; // [B*T, H*W, C]
        let o = self.to_out.forward(&o)?;
        // back to NCTHW
        let o = o
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, t, c, h, wd))?
            .permute((0, 2, 1, 3, 4))?
            .contiguous()?;
        o + residual
    }
}

/// Stride-2 down sampler: asymmetric `(0,1)` H/W pad then a causal conv (spatial-only `kt=1`, or
/// temporal `kt=3, st=2`).
struct Downsample3d {
    conv: CausalConv3d,
}
impl Downsample3d {
    fn load(w: &Weights, prefix: &str, temporal: bool) -> CResult<Self> {
        let (st, pt) = if temporal { (2, 1) } else { (1, 0) };
        Ok(Self {
            conv: CausalConv3d::load(w, &format!("{prefix}.conv"), (st, 2, 2), (pt, 0, 0), false)?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // (0,1) pad on H (dim 3) and W (dim 4).
        let xp = x.pad_with_zeros(3, 0, 1)?.pad_with_zeros(4, 0, 1)?;
        self.conv.forward(&xp)
    }
}

/// Pixel-shuffle upsampler: `upscale_conv` (1³, C→C·sf²·tf) → reshape/transpose → `conv` (3³, causal).
struct Upsample3d {
    upscale_conv: CausalConv3d,
    conv: CausalConv3d,
    sf: usize,
    tf: usize,
}
impl Upsample3d {
    fn load(w: &Weights, prefix: &str, temporal: bool) -> CResult<Self> {
        Ok(Self {
            upscale_conv: CausalConv3d::load(
                w,
                &format!("{prefix}.upscale_conv"),
                (1, 1, 1),
                (0, 0, 0),
                false,
            )?,
            conv: CausalConv3d::load(w, &format!("{prefix}.conv"), (1, 1, 1), (1, 1, 1), true)?,
            sf: 2,
            tf: if temporal { 2 } else { 1 },
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, t, h, wd) = x.dims5()?;
        let x = self.upscale_conv.forward(x)?; // (B, C·sf²·tf, T, H, W)
        let (sf, tf) = (self.sf, self.tf);
        // (B, sf, sf, tf, C, T, H, W) -> (B, C, T, tf, H, sf, W, sf) -> (B, C, T·tf, H·sf, W·sf)
        let x = x
            .reshape(&[b, sf, sf, tf, c, t, h, wd][..])?
            .permute([0usize, 4, 5, 3, 6, 1, 7, 2])?
            .contiguous()?
            .reshape((b, c, t * tf, h * sf, wd * sf))?;
        let x = if t == 1 && tf > 1 {
            x.narrow(2, 0, 1)?
        } else {
            x
        };
        self.conv.forward(&x)
    }
}

/// `num_resnets` resnets then an optional sampler.
struct DownBlock3d {
    resnets: Vec<ResnetBlock3d>,
    downsampler: Option<Downsample3d>,
}
impl DownBlock3d {
    fn load(
        w: &Weights,
        prefix: &str,
        n: usize,
        temporal: bool,
        sample: bool,
        cfg: &VaeConfig,
    ) -> CResult<Self> {
        let resnets = (0..n)
            .map(|i| ResnetBlock3d::load(w, &format!("{prefix}.resnets.{i}"), cfg))
            .collect::<CResult<Vec<_>>>()?;
        let downsampler = if sample {
            Some(Downsample3d::load(
                w,
                &format!("{prefix}.downsamplers.0"),
                temporal,
            )?)
        } else {
            None
        };
        Ok(Self {
            resnets,
            downsampler,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = x.clone();
        for r in &self.resnets {
            h = r.forward(&h)?;
        }
        if let Some(d) = &self.downsampler {
            h = d.forward(&h)?;
        }
        Ok(h)
    }
}

struct UpBlock3d {
    resnets: Vec<ResnetBlock3d>,
    upsampler: Option<Upsample3d>,
}
impl UpBlock3d {
    fn load(
        w: &Weights,
        prefix: &str,
        n: usize,
        temporal: bool,
        sample: bool,
        cfg: &VaeConfig,
    ) -> CResult<Self> {
        let resnets = (0..n)
            .map(|i| ResnetBlock3d::load(w, &format!("{prefix}.resnets.{i}"), cfg))
            .collect::<CResult<Vec<_>>>()?;
        let upsampler = if sample {
            Some(Upsample3d::load(
                w,
                &format!("{prefix}.upsamplers.0"),
                temporal,
            )?)
        } else {
            None
        };
        Ok(Self { resnets, upsampler })
    }
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

/// `resnet → attention → resnet` at constant channels.
struct MidBlock3d {
    resnet0: ResnetBlock3d,
    attn: Attention3d,
    resnet1: ResnetBlock3d,
}
impl MidBlock3d {
    fn load(w: &Weights, prefix: &str, cfg: &VaeConfig) -> CResult<Self> {
        Ok(Self {
            resnet0: ResnetBlock3d::load(w, &format!("{prefix}.resnets.0"), cfg)?,
            attn: Attention3d::load(w, &format!("{prefix}.attentions.0"), cfg)?,
            resnet1: ResnetBlock3d::load(w, &format!("{prefix}.resnets.1"), cfg)?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.resnet0.forward(x)?;
        let h = self.attn.forward(&h)?;
        self.resnet1.forward(&h)
    }
}

struct Encoder3d {
    conv_in: CausalConv3d,
    down_blocks: Vec<DownBlock3d>,
    mid: MidBlock3d,
    norm_out_w: Tensor,
    norm_out_b: Tensor,
    conv_out: CausalConv3d,
    groups: usize,
    eps: f64,
}
impl Encoder3d {
    fn load(w: &Weights, cfg: &VaeConfig) -> CResult<Self> {
        let n = cfg.enc_layers_per_block;
        // down0 spatial-only; down1/down2 temporal; down3 no sampler.
        let down_blocks = vec![
            DownBlock3d::load(w, "encoder.down_blocks.0", n, false, true, cfg)?,
            DownBlock3d::load(w, "encoder.down_blocks.1", n, true, true, cfg)?,
            DownBlock3d::load(w, "encoder.down_blocks.2", n, true, true, cfg)?,
            DownBlock3d::load(w, "encoder.down_blocks.3", n, false, false, cfg)?,
        ];
        Ok(Self {
            conv_in: CausalConv3d::load(w, "encoder.conv_in", (1, 1, 1), (1, 1, 1), false)?,
            down_blocks,
            mid: MidBlock3d::load(w, "encoder.mid_block", cfg)?,
            norm_out_w: w.require("encoder.conv_norm_out.weight")?.clone(),
            norm_out_b: w.require("encoder.conv_norm_out.bias")?.clone(),
            conv_out: CausalConv3d::load(w, "encoder.conv_out", (1, 1, 1), (1, 1, 1), false)?,
            groups: cfg.group_norm_groups,
            eps: cfg.group_norm_eps,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = self.conv_in.forward(x)?;
        for d in &self.down_blocks {
            h = d.forward(&h)?;
        }
        h = self.mid.forward(&h)?;
        h = gn(
            &h,
            &self.norm_out_w,
            &self.norm_out_b,
            self.groups,
            self.eps,
        )?;
        self.conv_out.forward(&nn::silu(&h)?)
    }
}

struct Decoder3d {
    conv_in: CausalConv3d,
    mid: MidBlock3d,
    up_blocks: Vec<UpBlock3d>,
    norm_out_w: Tensor,
    norm_out_b: Tensor,
    conv_out: CausalConv3d,
    groups: usize,
    eps: f64,
}
impl Decoder3d {
    fn load(w: &Weights, cfg: &VaeConfig) -> CResult<Self> {
        let n = cfg.dec_layers_per_block;
        // up0/up1 temporal; up2 spatial-only; up3 no sampler.
        let up_blocks = vec![
            UpBlock3d::load(w, "decoder.up_blocks.0", n, true, true, cfg)?,
            UpBlock3d::load(w, "decoder.up_blocks.1", n, true, true, cfg)?,
            UpBlock3d::load(w, "decoder.up_blocks.2", n, false, true, cfg)?,
            UpBlock3d::load(w, "decoder.up_blocks.3", n, false, false, cfg)?,
        ];
        Ok(Self {
            conv_in: CausalConv3d::load(w, "decoder.conv_in", (1, 1, 1), (1, 1, 1), false)?,
            mid: MidBlock3d::load(w, "decoder.mid_block", cfg)?,
            up_blocks,
            norm_out_w: w.require("decoder.conv_norm_out.weight")?.clone(),
            norm_out_b: w.require("decoder.conv_norm_out.bias")?.clone(),
            conv_out: CausalConv3d::load(w, "decoder.conv_out", (1, 1, 1), (1, 1, 1), false)?,
            groups: cfg.group_norm_groups,
            eps: cfg.group_norm_eps,
        })
    }
    fn forward(&self, z: &Tensor) -> Result<Tensor> {
        let mut h = self.conv_in.forward(z)?;
        h = self.mid.forward(&h)?;
        for u in &self.up_blocks {
            h = u.forward(&h)?;
        }
        h = gn(
            &h,
            &self.norm_out_w,
            &self.norm_out_b,
            self.groups,
            self.eps,
        )?;
        self.conv_out.forward(&nn::silu(&h)?)
    }
}

/// The SeedVR2 3D causal video VAE.
pub struct Seedvr2Vae {
    encoder: Encoder3d,
    decoder: Decoder3d,
    scaling_factor: f64,
    latent_channels: usize,
    pub spatial_scale: usize,
}

impl Seedvr2Vae {
    pub fn from_weights(w: &Weights) -> CResult<Self> {
        let cfg = VaeConfig::seedvr2();
        Ok(Self {
            encoder: Encoder3d::load(w, &cfg)?,
            decoder: Decoder3d::load(w, &cfg)?,
            scaling_factor: cfg.scaling_factor,
            latent_channels: cfg.latent_channels,
            spatial_scale: cfg.spatial_scale,
        })
    }

    /// `(B,3,T,H,W)` → scaled mean latent `(B,16,T',H',W')`. A 4-D `(B,3,H,W)` input gains `T=1`.
    pub fn encode(&self, x: &Tensor) -> Result<Tensor> {
        let x = if x.rank() == 4 {
            x.unsqueeze(2)?
        } else {
            x.clone()
        };
        let h = self.encoder.forward(&x)?; // (B,32,T',H',W')
        let mean = h.narrow(1, 0, self.latent_channels)?; // first 16 channels
        mean * self.scaling_factor
    }

    /// `(B,16,T',H',W')` → `(B,3,T,H,W)`. A 4-D latent gains `T=1`.
    pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
        let z = if z.rank() == 4 {
            z.unsqueeze(2)?
        } else {
            z.clone()
        };
        let z = (z * (1.0 / self.scaling_factor))?;
        self.decoder.forward(&z)
    }
}

#[cfg(test)]
mod shuffle_tests {
    use super::*;
    use candle_gen::candle_core::Device;

    /// The exact reshape/permute the temporal [`Upsample3d`] uses, in isolation.
    fn pixel_shuffle(x: &Tensor, sf: usize, tf: usize) -> candle_gen::Result<Tensor> {
        let (b, cc, t, h, wd) = x.dims5()?;
        let c = cc / (sf * sf * tf);
        Ok(x.reshape(&[b, sf, sf, tf, c, t, h, wd][..])?
            .permute([0usize, 4, 5, 3, 6, 1, 7, 2])?
            .contiguous()?
            .reshape((b, c, t * tf, h * sf, wd * sf))?)
    }

    /// Verify the candle reshape/permute realises the intended depth→(space,time) mapping for
    /// **distinct** per-(channel,frame) values: out[cc, ti·tf+ft, hi·sf+s1, wi·sf+s2] ==
    /// x[((s1·sf+s2)·tf+ft)·c+cc, ti, hi, wi]. Identical frames can't catch a temporal-mix bug here.
    #[test]
    fn pixel_shuffle_index_mapping() -> candle_gen::Result<()> {
        let dev = Device::Cpu;
        let (sf, tf, c, t, h, wd) = (2usize, 2usize, 2usize, 3usize, 2usize, 2usize);
        let cc = c * sf * sf * tf;
        // unique value per (channel, t, h, w)
        let n = cc * t * h * wd;
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let x = Tensor::from_vec(data.clone(), (1, cc, t, h, wd), &dev)?;
        let y = pixel_shuffle(&x, sf, tf)?;
        let yv = y.flatten_all()?.to_vec1::<f32>()?;
        let (ot, oh, ow) = (t * tf, h * sf, wd * sf);
        let xat = |ch: usize, ti: usize, hi: usize, wi: usize| -> f32 {
            data[((ch * t + ti) * h + hi) * wd + wi]
        };
        let yat = |co: usize, to: usize, ho: usize, wo: usize| -> f32 {
            yv[((co * ot + to) * oh + ho) * ow + wo]
        };
        for s1 in 0..sf {
            for s2 in 0..sf {
                for ft in 0..tf {
                    for cci in 0..c {
                        let ch = ((s1 * sf + s2) * tf + ft) * c + cci;
                        for ti in 0..t {
                            for hi in 0..h {
                                for wi in 0..wd {
                                    let got = yat(cci, ti * tf + ft, hi * sf + s1, wi * sf + s2);
                                    let exp = xat(ch, ti, hi, wi);
                                    assert_eq!(got, exp, "shuffle mismatch ch={ch}");
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
