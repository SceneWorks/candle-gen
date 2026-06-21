//! The **`AutoencoderKLWan`** (z48, `is_residual`) decoder — a port of diffusers
//! `autoencoder_kl_wan.py`, decode-only. Causal-Conv3d temporal VAE: latent `[B,48,T,H,W]` →
//! `[B,3, 1+(T-1)·4, 16H, 16W]` in `[-1,1]`.
//!
//! diffusers streams the decode frame-by-frame with a `feat_cache` (the causal temporal cache);
//! that is mathematically identical to a single pass over all `T` frames with the causal
//! left-padding ([`crate::conv3d`]) — except the temporal **upsampling**, where the first latent
//! frame is passed through un-doubled and the rest are doubled via the `time_conv` channel
//! interleave (the `first_chunk` rule). We reproduce that here in one pass.
//!
//! `WanRMS_norm` is a **channel-L2 normalization** over the channel axis (`x / max(‖x‖₂, 1e-12) ·
//! √C · γ`), NOT GroupNorm; weights ship as `.gamma`.

use candle_gen::candle_core::{DType, Error, Result, Tensor};
use candle_gen::candle_nn::ops::softmax_last_dim;
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::tiling::{
    budgeted_plan, TileCandidates, TilingBudgetError, TilingConfig, VaeTiling,
};

use crate::config::{VaeConfig, LATENTS_MEAN, LATENTS_STD};
use crate::conv3d::{CausalConv3d, Ctx};

const NORM_EPS: f64 = 1e-12;

/// Channel-L2 norm (`F.normalize(dim=channel) · √C · γ`). Works on 4-D `[N,C,H,W]` and 5-D
/// `[B,C,T,H,W]` tensors (channel axis 1). `pub(crate)` so the z16 [`crate::vae16`] sibling reuses it.
pub(crate) struct ChanNorm {
    gamma: Tensor, // [C]
    sqrt_c: f64,
}

impl ChanNorm {
    pub(crate) fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        let gamma = vb.get_unchecked("gamma")?.flatten_all()?;
        Ok(Self {
            gamma,
            sqrt_c: (channels as f64).sqrt(),
        })
    }

    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let l2 = x.sqr()?.sum_keepdim(1)?.sqrt()?.clamp(NORM_EPS, 1e30)?;
        let normed = (x.broadcast_div(&l2)? * self.sqrt_c)?;
        let c = self.gamma.dim(0)?;
        let gshape = match x.rank() {
            5 => vec![1, c, 1, 1, 1],
            4 => vec![1, c, 1, 1],
            _ => vec![1, c],
        };
        normed.broadcast_mul(&self.gamma.reshape(gshape)?)
    }
}

/// A native 2-D conv applied per video frame (resample / attention 1×1 convs). `pub(crate)` for the
/// z16 [`crate::vae16`] sibling.
pub(crate) struct Conv2dW {
    w: Tensor,
    b: Tensor, // [1,O,1,1]
    pad: usize,
}

impl Conv2dW {
    pub(crate) fn load(
        in_c: usize,
        out_c: usize,
        k: usize,
        pad: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            w: vb.get((out_c, in_c, k, k), "weight")?.contiguous()?,
            b: vb.get(out_c, "bias")?.reshape((1, out_c, 1, 1))?,
            pad,
        })
    }
    /// `x`: `[N, C, H, W]`.
    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.conv2d(&self.w, self.pad, 1, 1, 1)?.broadcast_add(&self.b)
    }
}

pub(crate) fn causal(
    in_c: usize,
    out_c: usize,
    kernel: (usize, usize, usize),
    vb: VarBuilder,
) -> Result<CausalConv3d> {
    CausalConv3d::load(in_c, out_c, kernel, vb)
}

pub(crate) struct Resnet {
    norm1: ChanNorm,
    conv1: CausalConv3d,
    norm2: ChanNorm,
    conv2: CausalConv3d,
    shortcut: Option<CausalConv3d>,
}

impl Resnet {
    pub(crate) fn new(in_c: usize, out_c: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm1: ChanNorm::new(in_c, vb.pp("norm1"))?,
            conv1: causal(in_c, out_c, (3, 3, 3), vb.pp("conv1"))?,
            norm2: ChanNorm::new(out_c, vb.pp("norm2"))?,
            conv2: causal(out_c, out_c, (3, 3, 3), vb.pp("conv2"))?,
            shortcut: if in_c != out_c {
                Some(causal(in_c, out_c, (1, 1, 1), vb.pp("conv_shortcut"))?)
            } else {
                None
            },
        })
    }

    pub(crate) fn forward(&self, x: &Tensor, ctx: &Ctx) -> Result<Tensor> {
        let h = match &self.shortcut {
            Some(c) => c.forward(x, ctx)?,
            None => x.clone(),
        };
        let y = self.conv1.forward(&self.norm1.forward(x)?.silu()?, ctx)?;
        let y = self.conv2.forward(&self.norm2.forward(&y)?.silu()?, ctx)?;
        y + h
    }

    pub(crate) fn reset_cache(&self) {
        self.conv1.reset_cache();
        self.conv2.reset_cache();
        if let Some(c) = &self.shortcut {
            c.reset_cache();
        }
    }
}

pub(crate) struct MidAttn {
    norm: ChanNorm,
    qkv: Conv2dW,
    proj: Conv2dW,
    channels: usize,
}

impl MidAttn {
    pub(crate) fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm: ChanNorm::new(channels, vb.pp("norm"))?,
            qkv: Conv2dW::load(channels, channels * 3, 1, 0, vb.pp("to_qkv"))?,
            proj: Conv2dW::load(channels, channels, 1, 0, vb.pp("proj"))?,
            channels,
        })
    }

    /// `x`: `[B,C,T,H,W]`. Per-frame spatial self-attention.
    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, t, h, w) = x.dims5()?;
        let merged = x
            .permute((0, 2, 1, 3, 4))?
            .reshape((b * t, c, h, w))?
            .contiguous()?;
        let xn = self.norm.forward(&merged)?;
        let qkv = self.qkv.forward(&xn)?; // [BT,3C,H,W]
        let qkv = qkv
            .reshape((b * t, 1, 3 * c, h * w))?
            .permute((0, 1, 3, 2))?
            .contiguous()?; // [BT,1,HW,3C]
        let q = qkv.narrow(3, 0, c)?.contiguous()?;
        let k = qkv.narrow(3, c, c)?.contiguous()?;
        let v = qkv.narrow(3, 2 * c, c)?.contiguous()?;
        let scale = (self.channels as f64).powf(-0.5);
        let scores = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
        let attn = softmax_last_dim(&scores)?;
        let o = attn.matmul(&v)?; // [BT,1,HW,C]
        let o = o
            .squeeze(1)?
            .permute((0, 2, 1))?
            .reshape((b * t, c, h, w))?
            .contiguous()?;
        let o = self.proj.forward(&o)?;
        let o = o
            .reshape((b, t, c, h, w))?
            .permute((0, 2, 1, 3, 4))?
            .contiguous()?;
        x + o
    }
}

/// Parameter-free `DupUp3D` shortcut (channel-duplicate upsample). `first_chunk` drops the leading
/// `factor_t-1` temporal frames to align with the causal main-path temporal expansion.
struct Dup {
    out_c: usize,
    factor_t: usize,
    factor_s: usize,
    repeats: usize,
}

impl Dup {
    fn new(in_c: usize, out_c: usize, factor_t: usize, factor_s: usize) -> Self {
        let factor = factor_t * factor_s * factor_s;
        Self {
            out_c,
            factor_t,
            factor_s,
            repeats: out_c * factor / in_c,
        }
    }

    fn apply(&self, x: &Tensor, ctx: &Ctx) -> Result<Tensor> {
        let (b, c, t, h, w) = x.dims5()?;
        // repeat_interleave channels: [B,C,T,H,W] → [B,C,repeats,T,H,W] → [B,C*repeats,T,H,W].
        let x = x
            .unsqueeze(2)?
            .broadcast_as((b, c, self.repeats, t, h, w))?
            .reshape((b, c * self.repeats, t, h, w))?;
        let (ft, fs) = (self.factor_t, self.factor_s);
        let x = x
            .reshape(&[b, self.out_c, ft, fs, fs, t, h, w][..])?
            .permute(&[0usize, 1, 5, 2, 6, 3, 7, 4][..])? // [B,out,t,ft,h,fs,w,fs]
            .reshape((b, self.out_c, t * ft, h * fs, w * fs))?
            .contiguous()?;
        // Drop the leading ft-1 duplicated frames so the shortcut aligns with the causal main-path
        // temporal expansion (the "first frame un-doubled" rule). Single pass: always (the clip's
        // leading frames). Streaming: only on the first latent frame — later chunks keep all t·ft
        // frames, matching the temporal upsampler which doubles them.
        let drop_leading = ft > 1 && (!ctx.streaming || ctx.first_chunk);
        if drop_leading {
            let tt = x.dim(2)?;
            x.narrow(2, ft - 1, tt - (ft - 1))
        } else {
            Ok(x)
        }
    }
}

pub(crate) enum Upsampler {
    /// Temporal (3D): `time_conv` doubling + spatial 2× conv.
    Temporal {
        time_conv: CausalConv3d,
        resample: Conv2dW,
    },
    /// Spatial-only (2D): nearest-2× + conv.
    Spatial { resample: Conv2dW },
}

impl Upsampler {
    /// Per-frame nearest-2× upsample then the 3×3 resample conv. `x`: `[B,C,T,H,W]`.
    fn spatial(resample: &Conv2dW, x: &Tensor) -> Result<Tensor> {
        let (b, c, t, h, w) = x.dims5()?;
        let merged = x
            .permute((0, 2, 1, 3, 4))?
            .reshape((b * t, c, h, w))?
            .contiguous()?;
        let up = merged.upsample_nearest2d(h * 2, w * 2)?;
        let y = resample.forward(&up)?;
        let (_, oc, oh, ow) = y.dims4()?;
        y.reshape((b, t, oc, oh, ow))?
            .permute((0, 2, 1, 3, 4))?
            .contiguous()
    }

    /// Double `t` frames → `2t` via the channel-interleave of `time_conv` (a `[B,2C,t,H,W]` output).
    fn double_temporal(time_conv: &CausalConv3d, x: &Tensor, ctx: &Ctx) -> Result<Tensor> {
        let (b, c, t, h, w) = x.dims5()?;
        let tc = time_conv.forward(x, ctx)?; // [B,2C,t,H,W]
        tc.reshape((b, 2, c, t, h, w))?
            .permute((0, 2, 3, 1, 4, 5))? // [B,C,t,2,H,W]
            .reshape((b, c, 2 * t, h, w))?
            .contiguous()
    }

    pub(crate) fn forward(&self, x: &Tensor, ctx: &Ctx) -> Result<Tensor> {
        match self {
            Upsampler::Spatial { resample } => Self::spatial(resample, x),
            Upsampler::Temporal {
                time_conv,
                resample,
            } => {
                let x_t = if ctx.streaming {
                    // Per-frame: the first latent frame passes un-doubled (and never touches the
                    // time_conv cache, matching the single-pass `first`); every later frame is
                    // doubled through the streaming time_conv.
                    if ctx.first_chunk {
                        x.clone()
                    } else {
                        Self::double_temporal(time_conv, x, ctx)?
                    }
                } else {
                    // Single pass: frame 0 un-doubled, frames 1.. doubled in one time_conv call.
                    let t = x.dim(2)?;
                    if t > 1 {
                        let first = x.narrow(2, 0, 1)?;
                        let rest = x.narrow(2, 1, t - 1)?;
                        let doubled = Self::double_temporal(time_conv, &rest, ctx)?;
                        Tensor::cat(&[&first, &doubled], 2)?
                    } else {
                        x.narrow(2, 0, 1)?
                    }
                };
                Self::spatial(resample, &x_t)
            }
        }
    }

    pub(crate) fn reset_cache(&self) {
        if let Upsampler::Temporal { time_conv, .. } = self {
            time_conv.reset_cache();
        }
    }
}

struct UpBlock {
    resnets: Vec<Resnet>,
    upsampler: Option<Upsampler>,
    dup: Option<Dup>,
}

impl UpBlock {
    fn forward(&self, x: &Tensor, ctx: &Ctx) -> Result<Tensor> {
        let x_copy = x.clone();
        let mut h = x.clone();
        for r in &self.resnets {
            h = r.forward(&h, ctx)?;
        }
        if let Some(up) = &self.upsampler {
            h = up.forward(&h, ctx)?;
        }
        if let Some(dup) = &self.dup {
            h = (h + dup.apply(&x_copy, ctx)?)?;
        }
        Ok(h)
    }

    fn reset_cache(&self) {
        for r in &self.resnets {
            r.reset_cache();
        }
        if let Some(up) = &self.upsampler {
            up.reset_cache();
        }
    }
}

pub struct WanVae {
    mean: Tensor, // [1,48,1,1,1]
    std: Tensor,
    post_quant_conv: CausalConv3d,
    conv_in: CausalConv3d,
    mid_resnet0: Resnet,
    mid_attn: MidAttn,
    mid_resnet1: Resnet,
    up_blocks: Vec<UpBlock>,
    norm_out: ChanNorm,
    conv_out: CausalConv3d,
    patch_size: usize,
    out_channels: usize,
}

impl WanVae {
    pub fn new(cfg: &VaeConfig, vb: VarBuilder) -> Result<Self> {
        let device = vb.device();
        let mean = Tensor::from_vec(LATENTS_MEAN.to_vec(), (1, 48, 1, 1, 1), device)?;
        let std = Tensor::from_vec(LATENTS_STD.to_vec(), (1, 48, 1, 1, 1), device)?;
        let post_quant_conv = causal(cfg.z_dim, cfg.z_dim, (1, 1, 1), vb.pp("post_quant_conv"))?;

        let dec = vb.pp("decoder");
        // dims = [base*4] + base*[reversed dim_mult] = [1024, 1024,1024,512,256] for base=256.
        let b = cfg.base_dim;
        let dims = [b * 4, b * 4, b * 4, b * 2, b];
        let conv_in = causal(cfg.z_dim, dims[0], (3, 3, 3), dec.pp("conv_in"))?;

        let mid = dec.pp("mid_block");
        let mid_resnet0 = Resnet::new(dims[0], dims[0], mid.pp("resnets").pp("0"))?;
        let mid_attn = MidAttn::new(dims[0], mid.pp("attentions").pp("0"))?;
        let mid_resnet1 = Resnet::new(dims[0], dims[0], mid.pp("resnets").pp("1"))?;

        // temperal_upsample = [true, true, false]; up_flag = i != 3.
        let temporal = [true, true, false, false];
        let mut up_blocks = Vec::with_capacity(4);
        for i in 0..4 {
            let (in_c, out_c) = (dims[i], dims[i + 1]);
            let up_flag = i != 3;
            let ub = dec.pp("up_blocks").pp(i);
            let mut resnets = Vec::with_capacity(cfg.num_res_blocks + 1);
            let mut cur = in_c;
            for j in 0..(cfg.num_res_blocks + 1) {
                resnets.push(Resnet::new(cur, out_c, ub.pp("resnets").pp(j))?);
                cur = out_c;
            }
            let (upsampler, dup) = if up_flag {
                let up = if temporal[i] {
                    Upsampler::Temporal {
                        time_conv: causal(
                            out_c,
                            out_c * 2,
                            (3, 1, 1),
                            ub.pp("upsampler").pp("time_conv"),
                        )?,
                        resample: Conv2dW::load(
                            out_c,
                            out_c,
                            3,
                            1,
                            ub.pp("upsampler").pp("resample").pp("1"),
                        )?,
                    }
                } else {
                    Upsampler::Spatial {
                        resample: Conv2dW::load(
                            out_c,
                            out_c,
                            3,
                            1,
                            ub.pp("upsampler").pp("resample").pp("1"),
                        )?,
                    }
                };
                let factor_t = if temporal[i] { 2 } else { 1 };
                (Some(up), Some(Dup::new(in_c, out_c, factor_t, 2)))
            } else {
                (None, None)
            };
            up_blocks.push(UpBlock {
                resnets,
                upsampler,
                dup,
            });
        }

        let norm_out = ChanNorm::new(dims[4], dec.pp("norm_out"))?;
        let conv_out = causal(
            dims[4],
            cfg.conv_out_channels,
            (3, 3, 3),
            dec.pp("conv_out"),
        )?;

        Ok(Self {
            mean,
            std,
            post_quant_conv,
            conv_in,
            mid_resnet0,
            mid_attn,
            mid_resnet1,
            up_blocks,
            norm_out,
            conv_out,
            patch_size: cfg.patch_size,
            out_channels: cfg.out_channels,
        })
    }

    /// Decode latents `[B,48,T,H,W]` → RGB frames `[B,3, 1+(T-1)·4, 16H, 16W]` in `[-1,1]`.
    ///
    /// **Streams one latent frame at a time** (sc-5176): the original single pass decoded every frame
    /// at once, spiking VAE memory ~60 GB on a 320²×17 clip (OOM). Each `CausalConv3d` carries its
    /// causal `feat_cache` across frames, so this is bit-equivalent to [`Self::decode_full`] while
    /// bounding peak memory to ~one frame's activations. Frame 0 expands to 1 output frame, each later
    /// latent frame to 4 (the two temporal upsamplers) — total `1+(T-1)·4`.
    pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
        let z = self.unnormalize(z)?;
        let t_lat = z.dim(2)?;
        self.reset_caches();
        let mut out: Option<Tensor> = None;
        for i in 0..t_lat {
            let zi = z.narrow(2, i, 1)?.contiguous()?;
            let oi = self.decode_inner(&zi, &Ctx::streaming(i == 0))?;
            out = Some(match out {
                Some(o) => Tensor::cat(&[&o, &oi], 2)?,
                None => oi,
            });
        }
        self.reset_caches();
        out.expect("decode needs >= 1 latent frame")
            .clamp(-1f32, 1f32)
    }

    /// Single-pass decode over all frames (the original path). Retained for the streaming-parity test
    /// (`decode` must match this bit-for-bit); not used in production (it OOMs on real clips).
    pub fn decode_full(&self, z: &Tensor) -> Result<Tensor> {
        let z = self.unnormalize(z)?;
        self.decode_inner(&z, &Ctx::single_pass())?
            .clamp(-1f32, 1f32)
    }

    /// Decode with **spatial tiling** for memory-bounded high-resolution video (`cfg`) — the candle
    /// half of sc-7111 (follow-on to the LTX budgeted decode, sc-7076/sc-6894).
    ///
    /// Unlike mlx-gen-wan's *single-pass* z48 decode (which needed full spatial **+ temporal** budgeted
    /// tiling, sc-6894), candle-gen-wan's [`decode`](Self::decode) already **streams one latent frame at
    /// a time** — so the temporal axis is already memory-bounded (peak ≈ one frame's decode, not the
    /// whole clip). The only remaining spike is a single **high-resolution frame** decoded through the
    /// z48 vae22 decoder; spatial tiling caps that. So this tiles **only the spatial axes** and keeps
    /// the streaming temporal bound: each spatial tile is itself decoded via the per-frame streaming
    /// [`decode`], preserving the causal feat-cache semantics.
    ///
    /// Shares the pure [`gen_core::tiling`](candle_gen::gen_core::tiling) geometry verbatim with the LTX
    /// half ([`VaeTiling::WAN22`]: ×16 spatial, ×4 **causal** temporal): splits the latent into
    /// overlapping spatial tiles, decodes each through the streaming [`decode`], and trapezoidally
    /// blends them into the full video by pad-and-accumulate (bounded peak = one tile's streaming decode
    /// plus the full-output `output`/`weights` buffers). Falls back to single-pass [`decode`] when `cfg`
    /// does not fire for these dims. `cfg` is expected to carry **spatial** tiling only (the budgeted
    /// selector never tiles the temporal axis here); a temporal `cfg` would split the causal stream at
    /// tile boundaries and is not bit-exact vs. the streaming [`decode`].
    pub fn decode_tiled(&self, z: &Tensor, cfg: &TilingConfig) -> Result<Tensor> {
        let (_b, _c, f, h, w) = z.dims5()?;
        if !cfg.needs_tiling(VaeTiling::WAN22, f as i32, h as i32, w as i32) {
            return self.decode(z);
        }
        let plan = cfg.plan(VaeTiling::WAN22, f as i32, h as i32, w as i32);
        let dev = z.device();

        // Full-size accumulators (mirrors the LTX/mlx pad-and-accumulate); add each tile in turn.
        // `output` carries the batch; `weights` stays b=1 and broadcasts on the final divide. With a
        // spatial-only `cfg`, `plan.t` is a single full-extent temporal tile (all-ones mask), so the
        // temporal loop runs once and each `decode` call streams the whole clip.
        let mut output: Option<Tensor> = None; // [B, 3, out_f, out_h, out_w]
        let mut weights: Option<Tensor> = None; // [1, 1, out_f, out_h, out_w]

        for t in &plan.t {
            for hh in &plan.h {
                for ww in &plan.w {
                    let tile = z
                        .narrow(2, t.start as usize, (t.end - t.start) as usize)?
                        .narrow(3, hh.start as usize, (hh.end - hh.start) as usize)?
                        .narrow(4, ww.start as usize, (ww.end - ww.start) as usize)?;
                    let dec = self.decode(&tile)?; // [B, 3, td, hd, wd] (streamed, clamped)
                    let (_, _, td, hd, wd) = dec.dims5()?;
                    let at = td.min((t.out_stop - t.out_start) as usize);
                    let ah = hd.min((hh.out_stop - hh.out_start) as usize);
                    let aw = wd.min((ww.out_stop - ww.out_start) as usize);

                    // 1-D trapezoidal masks → outer product [1, 1, at, ah, aw].
                    let tm = Tensor::from_slice(&t.mask[..at], (1, 1, at, 1, 1), dev)?;
                    let hm = Tensor::from_slice(&hh.mask[..ah], (1, 1, 1, ah, 1), dev)?;
                    let wm = Tensor::from_slice(&ww.mask[..aw], (1, 1, 1, 1, aw), dev)?;
                    let blend = tm.broadcast_mul(&hm)?.broadcast_mul(&wm)?;

                    let dec = dec.narrow(2, 0, at)?.narrow(3, 0, ah)?.narrow(4, 0, aw)?;
                    let weighted = dec.broadcast_mul(&blend)?;

                    // Place each tile at its output offset by zero-padding to the full output shape.
                    let (pt0, pt1) = (
                        t.out_start as usize,
                        plan.out_f as usize - (t.out_start as usize + at),
                    );
                    let (ph0, ph1) = (
                        hh.out_start as usize,
                        plan.out_h as usize - (hh.out_start as usize + ah),
                    );
                    let (pw0, pw1) = (
                        ww.out_start as usize,
                        plan.out_w as usize - (ww.out_start as usize + aw),
                    );
                    let pad5 = |x: &Tensor| -> Result<Tensor> {
                        x.pad_with_zeros(2, pt0, pt1)?
                            .pad_with_zeros(3, ph0, ph1)?
                            .pad_with_zeros(4, pw0, pw1)
                    };
                    let weighted_full = pad5(&weighted)?;
                    let blend_full = pad5(&blend)?;

                    output = Some(match output {
                        None => weighted_full,
                        Some(acc) => acc.add(&weighted_full)?,
                    });
                    weights = Some(match weights {
                        None => blend_full,
                        Some(acc) => acc.add(&blend_full)?,
                    });
                }
            }
        }

        let output =
            output.ok_or_else(|| Error::Msg("wan vae22: tile-decode plan had no tiles".into()))?;
        let weights =
            weights.ok_or_else(|| Error::Msg("wan vae22: tile-decode plan had no tiles".into()))?;
        // Normalize by the summed blend weight (clamped away from 0), broadcasting [1,1,F,H,W] over C.
        output.broadcast_div(&weights.maximum(1e-8f64)?)
    }

    /// **Memory-bounded** decode (sc-7111): derive the decoded output dims from the latent geometry
    /// (z48 vae22: ×16 spatial, ×4 **causal** temporal ⇒ `out_f = 1 + (T_lat−1)·4`), pick a budgeted
    /// **spatial** tiling via [`auto_tiling_budgeted_wan22`], and run [`decode_tiled`](Self::decode_tiled)
    /// — or the streaming [`decode`](Self::decode) when a single high-res frame already fits the VRAM
    /// budget. An over-budget decode returns a **catchable** error here instead of OOM-ing the worker.
    /// The candle z48 analogue of the LTX [`decode_budgeted`](crate::vae) wiring (sc-7076).
    pub fn decode_budgeted(&self, z: &Tensor) -> Result<Tensor> {
        let (_b, _c, f, h, w) = z.dims5()?;
        let out_f = 1 + (f as i32 - 1) * VaeTiling::WAN22.temporal_scale; // causal ×4
        let out_h = h as i32 * VaeTiling::WAN22.spatial_scale; // ×16
        let out_w = w as i32 * VaeTiling::WAN22.spatial_scale;
        match auto_tiling_budgeted_wan22(out_h, out_w, out_f)? {
            Some(cfg) => self.decode_tiled(z, &cfg),
            None => self.decode(z),
        }
    }

    /// `z_pixel = z·std + mean` in f32 (the inverse of the encoder's per-channel normalize).
    fn unnormalize(&self, z: &Tensor) -> Result<Tensor> {
        z.to_dtype(DType::F32)?
            .broadcast_mul(&self.std)?
            .broadcast_add(&self.mean)
    }

    /// The decoder graph for one chunk (`ctx.streaming` selects the per-frame `feat_cache` path).
    fn decode_inner(&self, z: &Tensor, ctx: &Ctx) -> Result<Tensor> {
        let mut h = self.post_quant_conv.forward(z, ctx)?;
        h = self.conv_in.forward(&h, ctx)?;
        h = self.mid_resnet0.forward(&h, ctx)?;
        h = self.mid_attn.forward(&h)?;
        h = self.mid_resnet1.forward(&h, ctx)?;
        for ub in &self.up_blocks {
            h = ub.forward(&h, ctx)?;
        }
        let h = self.norm_out.forward(&h)?.silu()?;
        let h = self.conv_out.forward(&h, ctx)?; // [B,12,T',H8,W8]
        self.unpatchify(&h)
    }

    /// Drop every streaming `feat_cache` (called around the [`Self::decode`] frame loop).
    fn reset_caches(&self) {
        self.post_quant_conv.reset_cache();
        self.conv_in.reset_cache();
        self.mid_resnet0.reset_cache();
        self.mid_resnet1.reset_cache();
        for ub in &self.up_blocks {
            ub.reset_cache();
        }
        self.conv_out.reset_cache();
    }

    /// 12 → 3 channels, 2× spatial (inverse of the encoder's 2×2 patchify).
    fn unpatchify(&self, x: &Tensor) -> Result<Tensor> {
        let p = self.patch_size;
        let (b, _cp, t, h, w) = x.dims5()?;
        let c = self.out_channels;
        x.reshape(&[b, c, p, p, t, h, w][..])?
            .permute(&[0usize, 1, 4, 5, 3, 6, 2][..])? // [B,c,T,H,p,W,p]
            .reshape((b, c, t, h * p, w * p))?
            .contiguous()
    }
}

// --- sc-7111 / sc-6894: budgeted z48 vae22 spatial-tiling for the candle decode -------------------
//
// Mirrors the candle LTX half (sc-7076, `candle-gen-ltx::vae`): the shared `gen_core::tiling::
// budgeted_plan` selector + a z48-vae22 cost model + a CUDA-VRAM budget source. The selector and tile
// geometry are byte-identical to the mlx side (pure gen-core); only the cost CONSTANTS and the budget
// source are backend-specific.
//
// **Structural difference from mlx-gen-wan** (why this is a spatial-only mirror, not a 1:1 port): the
// candle `decode` STREAMS one latent frame at a time (`WanVae::decode`), so the temporal axis is
// already memory-bounded — peak ≈ one frame's decode, not the whole clip. mlx-gen-wan's single-pass
// decode needed full spatial+temporal budgeted tiling (sc-6894); candle only needs to cap the spike of
// a single **high-resolution frame**, so the candidate grid below carries **no temporal candidates**
// and the cost model's per-tile term is per-output-**pixel** (one streamed frame's activations), not
// per-output-voxel (the whole tile) as in the mlx vae22 model.

const GIB_F64: f64 = 1024.0 * 1024.0 * 1024.0;
/// Fraction of total VRAM treated as safe (matches the candle-gen-ltx 0.85 / seedvr2 convention).
const WAN22_VAE_BUDGET_SAFE_FRAC: f64 = 0.85;
/// Fallback budget when neither the env override nor `nvidia-smi` yields a value.
const WAN22_VAE_DEFAULT_BUDGET_GIB: f64 = 16.0;

// Cost-model constants. **CUDA-CALIBRATED (sc-7148)** — fit from real-weight peak-VRAM anchors measured
// by `tests/vae_decode_sweep.rs` on an RTX PRO 6000 Blackwell (sm_120, CUDA 12.9, f32, device-level
// `nvidia-smi` peak) over real Wan2.2-TI2V-5B z48 weights. The five anchors (output WxHxF / largest
// spatial tile → measured peak):
//   768²×13  single-pass (full 768² frame)  → 29.17 GiB
//   1280²×13 single-pass (full 1280² frame) → 76.46 GiB   (the high-res single-frame spike)
//   1280²×13 tiled 256px                    →  6.30 GiB   (accumulator-dominated)
//   1280²×13 tiled 512px                    → 22.87 GiB
//   1280²×61 tiled 256px                    → 17.21 GiB   (accum ≈149 B/voxel from the out_vox slope)
// Because `decode` STREAMS one latent frame at a time, the per-tile activation spike scales with ONE
// output frame's area (`tile_h·tile_w`), not the whole tile's volume — so the model is `ACCUM·out_vox +
// FRAME·frame_px`. The accumulator term (≈149 B/voxel, far above the mlx z48's 40 — candle's tiled
// decode juggles more full-output buffers: the streaming per-tile `cat` plus the pad-and-accumulate
// blend) and the per-frame term (≈50 000 B/px for a full frame, rising to ≈90 000 B/px for the smaller
// tiles whose per-px decoder overhead amortizes worse) are rounded to the **conservative**
// (over-predicting) side; the model reproduces every anchor at ratio 1.12–1.88× (never under).
//
// **The old placeholders (40 / 12 000) badly UNDER-predicted** — they forecast ~18 GiB for the 1280²
// single-frame spike that really peaks 76 GiB, so the selector could have OK'd a single-pass / large
// tile that OOMs. (The story's "even the conservative placeholder never OOMs" assumed the seed
// over-estimated; the CUDA measurement shows it under-estimated by ~4×.) Re-run the sweep after a
// decoder or candle-allocator change. See the `wan22_decode_peak_matches_cuda_anchors` test below.
const WAN22_VAE_ACCUM_BYTES_PER_VOXEL: f64 = 160.0;
const WAN22_VAE_FRAME_BYTES_PER_OUT_PX: f64 = 92_000.0;

/// Candidate spatial tile sizes (output px, multiples of the vae22 ×16 scale, overlap 64). Matches the
/// mlx `VAE22_SPATIAL_PX` grid.
const WAN22_VAE_SPATIAL_PX: [i32; 8] = [768, 640, 512, 448, 384, 320, 256, 192];
/// **No temporal candidates** — the candle decode is already temporally streaming-bounded, so the
/// budgeted selector only ever tiles the spatial axes (see the module note above).
const WAN22_VAE_TEMPORAL_FR: [(i32, i32); 0] = [];

/// Estimated concurrent VRAM peak (GiB) of the streaming z48 vae22 decode while assembling an `out_*`
/// video, the largest spatial tile spanning `tile_h·tile_w` output px. `ACCUM·out_voxels` is the
/// unavoidable full-output accumulator floor; `FRAME·tile_h·tile_w` is one streamed frame's decoder
/// activations (independent of `out_f`/`tile_f` — the temporal axis streams). Single-pass is
/// `tile_* == out_*`; a zero tile yields the accumulator-only floor (the gen-core `budgeted_plan`
/// contract). `tile_f` is unused: the temporal axis is never tiled here.
fn estimated_wan22_decode_peak_gib(
    out_f: i64,
    out_h: i64,
    out_w: i64,
    _tile_f: i64,
    tile_h: i64,
    tile_w: i64,
) -> f64 {
    let out_voxels = (out_f * out_h * out_w) as f64;
    let frame_px = (tile_h * tile_w) as f64;
    (WAN22_VAE_ACCUM_BYTES_PER_VOXEL * out_voxels + WAN22_VAE_FRAME_BYTES_PER_OUT_PX * frame_px)
        / GIB_F64
}

/// Total VRAM (GiB) read from `nvidia-smi` (min across GPUs) — the SceneWorks worker convention.
/// `None` off-CUDA (e.g. the Mac dev host, where the budget falls back to the env override / default).
/// Mirrors `candle-gen-ltx::vae` / `candle-gen-seedvr2::video` (de-dupe into candle-gen core is a
/// tracked follow-up).
fn nvidia_smi_total_gib() -> Option<f64> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let min_mb = text
        .lines()
        .filter_map(|line| line.trim().parse::<f64>().ok())
        .filter(|&mb| mb > 0.0)
        .fold(f64::INFINITY, f64::min);
    min_mb.is_finite().then_some(min_mb / 1024.0)
}

/// The safe peak-GiB budget for the z48 vae22 decode tiler. Resolved in order: `WAN_VAE_BUDGET_GIB`
/// env override (positive float — the deterministic injection point for the worker/tests) → total VRAM
/// × [`WAN22_VAE_BUDGET_SAFE_FRAC`] (via `nvidia-smi`) → [`WAN22_VAE_DEFAULT_BUDGET_GIB`].
pub fn wan22_vae_safe_budget_gib() -> f64 {
    if let Ok(raw) = std::env::var("WAN_VAE_BUDGET_GIB") {
        if let Ok(gib) = raw.trim().parse::<f64>() {
            if gib > 0.0 {
                return gib;
            }
        }
    }
    match nvidia_smi_total_gib() {
        Some(total) => total * WAN22_VAE_BUDGET_SAFE_FRAC,
        None => WAN22_VAE_DEFAULT_BUDGET_GIB,
    }
}

/// **Memory-budgeted** spatial tiling for the z48 vae22 decode — routes the shared [`budgeted_plan`]
/// selector through the vae22 cost model. Caller passes the **output** dims. `Ok(None)` → a single
/// high-res frame already fits (streaming single-pass); `Ok(Some)` → the largest spatial tile that
/// fits; `Err` → a catchable over-budget signal returned before the decode (not an OOM).
pub fn auto_tiling_budgeted_wan22(
    height: i32,
    width: i32,
    out_frames: i32,
) -> Result<Option<TilingConfig>> {
    plan_wan22_tiling(height, width, out_frames, wan22_vae_safe_budget_gib())
}

/// Pure vae22 spatial tile selector (the `safe_gib` ceiling injected so it is unit-testable without a
/// GPU). Supplies the vae22 candidate grid + cost model to the shared [`budgeted_plan`]; same
/// `Ok(None)` / `Ok(Some)` / catchable-`Err` contract as the LTX half.
fn plan_wan22_tiling(
    height: i32,
    width: i32,
    out_frames: i32,
    safe_gib: f64,
) -> Result<Option<TilingConfig>> {
    let candidates = TileCandidates {
        spatial_px: &WAN22_VAE_SPATIAL_PX,
        spatial_overlap_px: 64,
        temporal: &WAN22_VAE_TEMPORAL_FR,
    };
    budgeted_plan(
        height,
        width,
        out_frames,
        safe_gib,
        candidates,
        estimated_wan22_decode_peak_gib,
    )
    .map_err(|e| match e {
        TilingBudgetError::AccumulatorsExceedBudget {
            projected_gib,
            safe_gib,
        } => Error::Msg(format!(
            "wan z48 vae22 decode: assembling a {width}×{height}×{out_frames} video needs ~{projected_gib:.0} \
             GB just for the output buffers, over the ~{safe_gib:.0} GB safe VRAM budget. Reduce the \
             resolution or frame count."
        )),
        TilingBudgetError::SmallestTileExceedsBudget {
            projected_gib,
            safe_gib,
        } => Error::Msg(format!(
            "wan z48 vae22 decode: a {width}×{height}×{out_frames} video peaks at ~{projected_gib:.0} GB even \
             with the smallest spatial tile, over the ~{safe_gib:.0} GB safe VRAM budget. Reduce the \
             resolution or frame count."
        )),
    })
}

#[cfg(test)]
mod budget_tests {
    use super::*;

    #[test]
    fn wan22_tiling_single_pass_when_small() {
        // A short, low-res clip fits a single streaming decode → no tiling.
        assert!(plan_wan22_tiling(256, 256, 25, 60.0).unwrap().is_none());
    }

    #[test]
    fn wan22_tiling_bounds_high_res_frame_peak() {
        // A 1280×1280 frame's single-pass activation spike (~76 GiB measured) blows a 48 GiB-class
        // budget; the selector must tile spatially and keep the recomputed peak under the safe ceiling
        // (bounded/catchable). 48 GiB (not 24) because the CUDA-calibrated accumulator floor for a
        // 1280²×97 video is ~24 GiB on its own — a 24 GiB tier genuinely cannot assemble it (the model
        // correctly returns AccumulatorsExceedBudget there; see the unfittable test).
        let safe = 48.0 * 0.85;
        let cfg = plan_wan22_tiling(1280, 1280, 97, safe)
            .unwrap()
            .expect("high-res frame must tile spatially");
        // Spatial-only: the selector never tiles the temporal axis here.
        assert!(cfg.spatial.is_some(), "expected a spatial tile");
        assert!(
            cfg.temporal.is_none(),
            "candle decode streams temporally — no temporal tiling"
        );
        let th = cfg
            .spatial
            .map(|s| (s.tile_px as i64).min(1280))
            .unwrap_or(1280);
        let peak = estimated_wan22_decode_peak_gib(97, 1280, 1280, 97, th, th);
        assert!(peak <= safe, "chosen peak {peak:.1} over safe {safe:.1}");
    }

    #[test]
    fn wan22_tiling_errors_when_unfittable() {
        // 4K × 257 frames under 8 GiB: the output accumulators alone blow it → catchable, not an OOM.
        assert!(plan_wan22_tiling(2160, 3840, 257, 8.0).is_err());
    }

    #[test]
    fn wan22_budget_env_override_wins() {
        // The deterministic injection point the worker/tests use. (Set/clear in-process.)
        std::env::set_var("WAN_VAE_BUDGET_GIB", "42.5");
        assert_eq!(wan22_vae_safe_budget_gib(), 42.5);
        std::env::remove_var("WAN_VAE_BUDGET_GIB");
    }

    /// sc-7148: the calibrated streaming cost model must stay **conservative** against the real CUDA
    /// peak-VRAM anchors (RTX PRO 6000 Blackwell, sm_120, f32, real Wan2.2-TI2V-5B z48 weights) it was
    /// fit from — `estimated ≥ measured` for every anchor (never under-predict ⇒ the selector never OKs
    /// a tile / single-pass that OOMs), and not absurdly over (≤ 2.5×). Regenerate with `cargo test -p
    /// candle-gen-wan --features cuda --release --test vae_decode_sweep -- --ignored --nocapture` after
    /// a decoder or candle-allocator change.
    #[test]
    fn wan22_decode_peak_matches_cuda_anchors() {
        // (out_f, out_h, out_w, tile_h, tile_w, measured_peak_gib). `tile_f` is unused by the streaming
        // model (passed as out_f). Single-pass ⇒ tile == out (the full-frame spike).
        let anchors: [(i64, i64, i64, i64, i64, f64); 5] = [
            (13, 768, 768, 768, 768, 29.1748),
            (13, 1280, 1280, 1280, 1280, 76.4561),
            (13, 1280, 1280, 256, 256, 6.2998),
            (13, 1280, 1280, 512, 512, 22.8672),
            (61, 1280, 1280, 256, 256, 17.2109),
        ];
        for (of, oh, ow, th, tw, measured) in anchors {
            let est = estimated_wan22_decode_peak_gib(of, oh, ow, of, th, tw);
            assert!(
                est >= measured,
                "under-predicts {ow}x{oh}x{of} tile {tw}x{th}: est {est:.2} < measured {measured:.2} GiB"
            );
            assert!(
                est <= measured * 2.5,
                "over-predicts {ow}x{oh}x{of} tile {tw}x{th}: est {est:.2} > 2.5x measured {measured:.2} GiB"
            );
        }
    }
}
