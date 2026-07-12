//! The **Wan 2.1 `AutoencoderKLWan`** (z16, stride 4×8×8) — the temporal VAE used by **both** A14B
//! MoE variants (`wan2_2_t2v_14b` / `wan2_2_i2v_14b`, sc-5174). Decode (always) + encode (I2V
//! channel-concat conditioning), ported from the diffusers checkpoint
//! (`Wan-AI/Wan2.2-T2V-A14B-Diffusers/vae`).
//!
//! Distinct from the 5B's z48 [`crate::vae`] `AutoencoderKLWan` on three structural axes (`vae/config.json`):
//!  - **z16, base_dim 96** (`dim_mult [1,2,4,4]`) vs the z48 base 256.
//!  - **non-residual** — no `DupUp3D`/`AvgDown3D` block-level shortcuts (the z48's `is_residual`).
//!  - **no spatial patchify** — `conv_out` emits 3 channels directly (the z48 unpatchifies a 2×2 grid),
//!    so the spatial scale is **8×** (3 up/down stages), not 16×.
//!  - diffusers names the up-sampler `up_blocks.N.upsamplers.0.…` (plural) vs the z48's `upsampler.…`.
//!
//! It reuses the proven z48 building blocks ([`ChanNorm`], [`Conv2dW`], [`Resnet`], [`MidAttn`],
//! [`Upsampler`], [`causal`]) and the from-scratch [`CausalConv3d`](crate::conv3d) — only the encoder's
//! stride-2 spatial/temporal downsamplers are new here. Decode **streams one latent frame at a time**
//! (the sc-5176 fix, bit-equivalent to a single pass via the causal `feat_cache`); encode mirrors the
//! diffusers **chunked** causal encode (frame 0 alone, then 4-frame chunks). Everything runs **f32**.

use std::sync::Mutex;

use candle_gen::candle_core::{DType, Result, Tensor};
use candle_gen::candle_nn::VarBuilder;

use crate::config::{Vae16Config, LATENTS16_MEAN, LATENTS16_STD};
use crate::conv3d::{CausalConv3d, Ctx};
use crate::vae::{causal, ChanNorm, Conv2dW, MidAttn, Resnet, Upsampler};

/// One z16 decoder up-stage: residual blocks then an optional spatial/temporal upsampler (no `Dup`
/// residual — the z16 VAE is non-residual).
struct UpBlock {
    resnets: Vec<Resnet>,
    upsampler: Option<Upsampler>,
}

impl UpBlock {
    fn forward(&self, x: &Tensor, ctx: &Ctx) -> Result<Tensor> {
        let mut h = x.clone();
        for r in &self.resnets {
            h = r.forward(&h, ctx)?;
        }
        if let Some(up) = &self.upsampler {
            h = up.forward(&h, ctx)?;
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

/// Encoder spatial 2× downsample: `ZeroPad2d((0,1,0,1))` per frame + a stride-2 3×3 conv (the diffusers
/// `WanResample` `resample.1`). Operates per-frame (no temporal cache).
struct SpatialDown {
    w: Tensor,
    b: Tensor, // [1, O, 1, 1]
}

impl SpatialDown {
    fn load(in_c: usize, out_c: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            w: vb.get((out_c, in_c, 3, 3), "weight")?.contiguous()?,
            b: vb.get(out_c, "bias")?.reshape((1, out_c, 1, 1))?,
        })
    }

    /// `x`: `[B,C,T,H,W]` → `[B,C,T,H/2,W/2]`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, t, h, w) = x.dims5()?;
        let merged = x
            .permute((0, 2, 1, 3, 4))?
            .reshape((b * t, c, h, w))?
            .contiguous()?;
        // ZeroPad2d((left,right,top,bottom)) = (0,1,0,1): pad right (W, dim 3) + bottom (H, dim 2).
        let padded = merged.pad_with_zeros(2, 0, 1)?.pad_with_zeros(3, 0, 1)?;
        let y = padded.conv2d(&self.w, 0, 2, 1, 1)?.broadcast_add(&self.b)?;
        let (_, oc, oh, ow) = y.dims4()?;
        y.reshape((b, t, oc, oh, ow))?
            .permute((0, 2, 1, 3, 4))?
            .contiguous()
    }
}

/// Encoder temporal 2× downsample (`time_conv`, a causal stride-2 `(3,1,1)` conv). Chunked like the
/// diffusers `WanResample` `downsample3d`: the **first chunk** stashes its last frame and passes
/// through un-downsampled; later chunks prepend the stash as the single causal left-context frame
/// (`causal = kt − st = 1`, fully covered by the cache → no zero-pad) and run the stride-2 conv.
struct TemporalDown {
    w: Tensor,                    // [O, I, 3, 1, 1]
    b: Tensor,                    // [1, O, 1, 1, 1]
    cache: Mutex<Option<Tensor>>, // previous chunk's last frame
}

impl TemporalDown {
    fn load(in_c: usize, out_c: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            w: vb.get((out_c, in_c, 3, 1, 1), "weight")?.contiguous()?,
            b: vb.get(out_c, "bias")?.reshape((1, out_c, 1, 1, 1))?,
            cache: Mutex::new(None),
        })
    }

    fn reset_cache(&self) {
        // sc-9015 / F-031: recover from a poisoned lock (reset-on-miss streaming cache).
        *candle_gen::lock_recover(&self.cache) = None;
    }

    fn forward(&self, x: &Tensor, ctx: &Ctx) -> Result<Tensor> {
        let last = x.narrow(2, x.dim(2)? - 1, 1)?.contiguous()?;
        if ctx.first_chunk {
            // First chunk: passthrough (no temporal downsample), stash the last frame for next chunk.
            // sc-9015 / F-031: recover from a poisoned lock (overwrite-on-miss streaming cache).
            *candle_gen::lock_recover(&self.cache) = Some(last);
            return Ok(x.clone());
        }
        // sc-9015 / F-031: recover from a poisoned lock; the `.expect` below is on the cached
        // `Option`, not the lock (a warmed cache is a real precondition on the non-first chunk).
        let prev = candle_gen::lock_recover(&self.cache)
            .clone()
            .expect("TemporalDown: non-first chunk needs a warmed cache");
        let xcat = Tensor::cat(&[&prev, x], 2)?; // T+1 frames; cache supplies the 1 causal context frame
        let out = self.strided_conv(&xcat)?;
        *candle_gen::lock_recover(&self.cache) = Some(last);
        Ok(out)
    }

    /// Stride-2, kernel-3 temporal conv over `[B,C,Tc,H,W]` (1×1 spatial) → `[B,O,(Tc-3)/2+1,H,W]`.
    /// Three taps `out[o] = Σ_k W[:,:,k]·x[2o+k]`, each a per-frame 1×1 conv2d, summed.
    fn strided_conv(&self, xcat: &Tensor) -> Result<Tensor> {
        let (b, c, tc, h, w) = xcat.dims5()?;
        let out_t = (tc - 3) / 2 + 1;
        let dev = xcat.device();
        let mut acc: Option<Tensor> = None;
        for k in 0..3 {
            let idx: Vec<u32> = (0..out_t).map(|o| (2 * o + k) as u32).collect();
            let sel = Tensor::from_vec(idx, out_t, dev)?;
            let frames = xcat.index_select(&sel, 2)?; // [B,C,out_t,H,W]
            let merged = frames
                .permute((0, 2, 1, 3, 4))?
                .reshape((b * out_t, c, h, w))?
                .contiguous()?;
            let wk = self.w.narrow(2, k, 1)?.squeeze(2)?.contiguous()?; // [O,I,1,1]
            let yk = merged.conv2d(&wk, 0, 1, 1, 1)?;
            acc = Some(match acc {
                Some(a) => (a + yk)?,
                None => yk,
            });
        }
        let y = acc.expect("kernel 3 has >= 1 tap");
        let (_, oc, oh, ow) = y.dims4()?;
        y.reshape((b, out_t, oc, oh, ow))?
            .permute((0, 2, 1, 3, 4))?
            .contiguous()?
            .broadcast_add(&self.b)
    }
}

/// One encoder down-stage entry: a residual block or a spatial/temporal downsample.
enum DownLayer {
    Res(Resnet),
    Down {
        spatial: SpatialDown,
        temporal: Option<TemporalDown>,
    },
}

impl DownLayer {
    fn forward(&self, x: &Tensor, ctx: &Ctx) -> Result<Tensor> {
        match self {
            DownLayer::Res(r) => r.forward(x, ctx),
            DownLayer::Down { spatial, temporal } => {
                let x = spatial.forward(x)?;
                match temporal {
                    Some(t) => t.forward(&x, ctx),
                    None => Ok(x),
                }
            }
        }
    }

    fn reset_cache(&self) {
        match self {
            DownLayer::Res(r) => r.reset_cache(),
            DownLayer::Down { temporal, .. } => {
                if let Some(t) = temporal {
                    t.reset_cache();
                }
            }
        }
    }
}

/// The z16 encoder (`conv_in → flat down_blocks → mid → norm/SiLU/conv_out`) → `2·z` moments. Chunked
/// causal: drive each chunk with [`Ctx::streaming`]; the convs carry their `feat_cache`.
struct Encoder {
    conv_in: CausalConv3d,
    down_blocks: Vec<DownLayer>,
    mid_resnet0: Resnet,
    mid_attn: MidAttn,
    mid_resnet1: Resnet,
    norm_out: ChanNorm,
    conv_out: CausalConv3d,
}

impl Encoder {
    fn new(cfg: &Vae16Config, vb: VarBuilder) -> Result<Self> {
        let b = cfg.base_dim;
        // dim_mult [1,2,4,4] → stage dims [96,192,384,384]; downsample after stages 0,1,2.
        let stage_dim = [b, b * 2, b * 4, b * 4];
        let temporal_down = [false, true, true];

        let conv_in = causal(3, b, (3, 3, 3), vb.pp("conv_in"))?;
        let mut down_blocks = Vec::new();
        let mut idx = 0usize;
        for (s, &out_d) in stage_dim.iter().enumerate() {
            let in_d = if s == 0 { b } else { stage_dim[s - 1] };
            for j in 0..cfg.num_res_blocks {
                let rin = if j == 0 { in_d } else { out_d };
                down_blocks.push(DownLayer::Res(Resnet::new(
                    rin,
                    out_d,
                    vb.pp("down_blocks").pp(idx),
                )?));
                idx += 1;
            }
            if s < 3 {
                let db = vb.pp("down_blocks").pp(idx);
                let spatial = SpatialDown::load(out_d, out_d, db.pp("resample").pp("1"))?;
                let temporal = if temporal_down[s] {
                    Some(TemporalDown::load(out_d, out_d, db.pp("time_conv"))?)
                } else {
                    None
                };
                down_blocks.push(DownLayer::Down { spatial, temporal });
                idx += 1;
            }
        }

        let mid = vb.pp("mid_block");
        let mid_dim = b * 4;
        Ok(Self {
            conv_in,
            down_blocks,
            mid_resnet0: Resnet::new(mid_dim, mid_dim, mid.pp("resnets").pp("0"))?,
            mid_attn: MidAttn::new(mid_dim, mid.pp("attentions").pp("0"))?,
            mid_resnet1: Resnet::new(mid_dim, mid_dim, mid.pp("resnets").pp("1"))?,
            norm_out: ChanNorm::new(mid_dim, vb.pp("norm_out"))?,
            conv_out: causal(mid_dim, 2 * cfg.z_dim, (3, 3, 3), vb.pp("conv_out"))?,
        })
    }

    fn forward(&self, x: &Tensor, ctx: &Ctx) -> Result<Tensor> {
        let mut h = self.conv_in.forward(x, ctx)?;
        for layer in &self.down_blocks {
            h = layer.forward(&h, ctx)?;
        }
        h = self.mid_resnet0.forward(&h, ctx)?;
        h = self.mid_attn.forward(&h)?;
        h = self.mid_resnet1.forward(&h, ctx)?;
        let h = self.norm_out.forward(&h)?.silu()?;
        self.conv_out.forward(&h, ctx)
    }

    fn reset_cache(&self) {
        self.conv_in.reset_cache();
        for layer in &self.down_blocks {
            layer.reset_cache();
        }
        self.mid_resnet0.reset_cache();
        self.mid_resnet1.reset_cache();
        self.conv_out.reset_cache();
    }
}

/// Reject a conditioning-video temporal length the chunked causal encode cannot consume without
/// dropping frames (F-126, sc-11220). The encode reads frame 0 alone then 4-frame chunks, i.e.
/// exactly `1 + 4·k` frames; any other `t` leaves the trailing `(t - 1) % 4` frames unencoded.
fn require_aligned_encode_frames(t: usize) -> Result<()> {
    if t == 0 || !(t - 1).is_multiple_of(4) {
        return Err(candle_gen::candle_core::Error::Msg(format!(
            "WanVae16::encode: frame count must be 1 + 4·k (got {t}); the causal encode would \
             silently drop the trailing {} frame(s)",
            t.saturating_sub(1) % 4
        )));
    }
    Ok(())
}

/// The Wan 2.1 z16 VAE: a decoder (always) plus an optional encoder (I2V conditioning), with
/// per-channel latent normalization.
pub struct WanVae16 {
    mean: Tensor, // [1,16,1,1,1]
    std: Tensor,
    post_quant_conv: CausalConv3d,
    conv_in: CausalConv3d,
    mid_resnet0: Resnet,
    mid_attn: MidAttn,
    mid_resnet1: Resnet,
    up_blocks: Vec<UpBlock>,
    norm_out: ChanNorm,
    conv_out: CausalConv3d,
    encoder: Option<(Encoder, CausalConv3d)>, // (encoder, quant_conv)
    z_dim: usize,
}

impl WanVae16 {
    /// Build a **decode-only** z16 VAE from a diffusers `vae/` snapshot (T2V — no I2V conditioning).
    pub fn new(cfg: &Vae16Config, vb: VarBuilder) -> Result<Self> {
        Self::build(cfg, vb, false)
    }

    /// Build a z16 VAE **with the encoder** (I2V — the conditioning image's first-frame latent).
    pub fn new_with_encoder(cfg: &Vae16Config, vb: VarBuilder) -> Result<Self> {
        Self::build(cfg, vb, true)
    }

    fn build(cfg: &Vae16Config, vb: VarBuilder, with_encoder: bool) -> Result<Self> {
        let device = vb.device();
        let z = cfg.z_dim;
        let mean = Tensor::from_vec(LATENTS16_MEAN.to_vec(), (1, z, 1, 1, 1), device)?;
        let std = Tensor::from_vec(LATENTS16_STD.to_vec(), (1, z, 1, 1, 1), device)?;
        let post_quant_conv = causal(z, z, (1, 1, 1), vb.pp("post_quant_conv"))?;

        let dec = vb.pp("decoder");
        let b = cfg.base_dim;
        // Per-up-block resnet output dims base·[4,4,2,1]; the spatial resample halves channels into the
        // next block's input. temperal_upsample = reversed([false,true,true]) = [true,true,false].
        let resnet_out = [b * 4, b * 4, b * 2, b];
        let has_up = [true, true, true, false];
        let temporal = [true, true, false, false];
        let conv_in = causal(z, resnet_out[0], (3, 3, 3), dec.pp("conv_in"))?;

        let mid = dec.pp("mid_block");
        let mid_dim = b * 4;
        let mid_resnet0 = Resnet::new(mid_dim, mid_dim, mid.pp("resnets").pp("0"))?;
        let mid_attn = MidAttn::new(mid_dim, mid.pp("attentions").pp("0"))?;
        let mid_resnet1 = Resnet::new(mid_dim, mid_dim, mid.pp("resnets").pp("1"))?;

        let mut up_blocks = Vec::with_capacity(4);
        let mut block_in = resnet_out[0]; // conv_in output feeds up_block 0
        for i in 0..4 {
            let out_c = resnet_out[i];
            let ub = dec.pp("up_blocks").pp(i);
            let mut resnets = Vec::with_capacity(cfg.num_res_blocks + 1);
            let mut cur = block_in;
            for j in 0..(cfg.num_res_blocks + 1) {
                resnets.push(Resnet::new(cur, out_c, ub.pp("resnets").pp(j))?);
                cur = out_c;
            }
            let upsampler = if has_up[i] {
                let us = ub.pp("upsamplers").pp("0");
                let resample = Conv2dW::load(out_c, out_c / 2, 3, 1, us.pp("resample").pp("1"))?;
                Some(if temporal[i] {
                    Upsampler::Temporal {
                        time_conv: causal(out_c, out_c * 2, (3, 1, 1), us.pp("time_conv"))?,
                        resample,
                    }
                } else {
                    Upsampler::Spatial { resample }
                })
            } else {
                None
            };
            up_blocks.push(UpBlock { resnets, upsampler });
            block_in = out_c / 2; // the resample halves channels into the next block
        }

        let norm_out = ChanNorm::new(b, dec.pp("norm_out"))?;
        let conv_out = causal(b, cfg.out_channels, (3, 3, 3), dec.pp("conv_out"))?;

        let encoder = if with_encoder {
            Some((
                Encoder::new(cfg, vb.pp("encoder"))?,
                causal(2 * z, 2 * z, (1, 1, 1), vb.pp("quant_conv"))?,
            ))
        } else {
            None
        };

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
            encoder,
            z_dim: z,
        })
    }

    /// Decode latents `[B,16,T,H,W]` → RGB frames `[B,3, 1+(T-1)·4, 8H, 8W]` in `[-1,1]`. **Streams one
    /// latent frame at a time** (sc-5176): bit-equivalent to a single pass (the causal `feat_cache`) but
    /// bounds peak memory to ~one frame's activations — the 14B's heavier clips would otherwise OOM the
    /// VAE-decode stage exactly as the 5B did.
    pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
        let z = self.unnormalize(z)?;
        let t_lat = z.dim(2)?;
        self.reset_caches();
        // Collect the per-frame decoded chunks and `cat` once (sc-9037): cat-ing onto a growing
        // accumulator each iteration re-copies every prior frame → O(T²) copy traffic and briefly
        // holds old+new. A single `Tensor::cat` at the end is O(T) and equivalent (same frames, same
        // order along the temporal axis).
        let mut chunks: Vec<Tensor> = Vec::with_capacity(t_lat);
        for i in 0..t_lat {
            let zi = z.narrow(2, i, 1)?.contiguous()?;
            chunks.push(self.decode_inner(&zi, &Ctx::streaming(i == 0))?);
        }
        self.reset_caches();
        assert!(!chunks.is_empty(), "decode needs >= 1 latent frame");
        Tensor::cat(&chunks, 2)?.clamp(-1f32, 1f32)
    }

    /// Single-pass decode over all frames (the original path). Retained for the streaming-parity test;
    /// not used in production (it spikes VAE memory on real clips).
    pub fn decode_full(&self, z: &Tensor) -> Result<Tensor> {
        let z = self.unnormalize(z)?;
        self.decode_inner(&z, &Ctx::single_pass())?
            .clamp(-1f32, 1f32)
    }

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
        self.conv_out.forward(&h, ctx) // [B,3,T',8H,8W] — no unpatchify (z16 has no spatial patchify)
    }

    /// Encode a conditioning video `[B,3,T,H,W]` (`T = 1 + 4·k`, values in `[-1,1]`) → normalized latent
    /// `[B,16,T_lat,H/8,W/8]`. Mirrors the diffusers **chunked** causal encode (frame 0 alone, then
    /// 4-frame chunks; the convs carry their `feat_cache`), then `quant_conv` → take the posterior mean →
    /// `(μ − mean)/std`. Requires encoder weights ([`Self::new_with_encoder`]).
    pub fn encode(&self, video: &Tensor) -> Result<Tensor> {
        let (encoder, quant_conv) = self.encoder.as_ref().ok_or_else(|| {
            candle_gen::candle_core::Error::Msg("WanVae16: encode needs encoder weights".into())
        })?;
        let t = video.dim(2)?;
        // The chunked causal encode consumes exactly `1 + 4·(num_chunks-1)` frames (frame 0 alone,
        // then 4-frame chunks). For `t % 4 != 1` the trailing `(t - 1) % 4` frames would silently
        // vanish from the latent. All in-repo callers pre-align, but this method (and `Scail2Job`)
        // is `pub`, so reject the unaligned length with a typed error rather than dropping frames.
        require_aligned_encode_frames(t)?;
        let num_chunks = 1 + (t - 1) / 4;
        encoder.reset_cache();
        // Collect the per-chunk encoded features and `cat` once (sc-9037): cat-ing onto a growing
        // accumulator each iteration re-copies every prior chunk → O(T²) copy traffic. A single
        // `Tensor::cat` at the end is O(T) and equivalent (same chunks, same temporal order).
        let mut chunks: Vec<Tensor> = Vec::with_capacity(num_chunks);
        for i in 0..num_chunks {
            let chunk = if i == 0 {
                video.narrow(2, 0, 1)?
            } else {
                video.narrow(2, 1 + 4 * (i - 1), 4)?
            }
            .contiguous()?;
            chunks.push(encoder.forward(&chunk, &Ctx::streaming(i == 0))?);
        }
        encoder.reset_cache();
        assert!(!chunks.is_empty(), "encode needs >= 1 frame");
        let out = Tensor::cat(&chunks, 2)?;
        // quant_conv (1×1×1) over the full moments, take the mean (first z channels), normalize.
        let moments = quant_conv.forward(&out, &Ctx::single_pass())?;
        let mu = moments.narrow(1, 0, self.z_dim)?;
        mu.broadcast_sub(&self.mean)?.broadcast_div(&self.std)
    }

    /// `z_pixel = z·std + mean` in f32 (the inverse of the encoder's per-channel normalize).
    fn unnormalize(&self, z: &Tensor) -> Result<Tensor> {
        z.to_dtype(DType::F32)?
            .broadcast_mul(&self.std)?
            .broadcast_add(&self.mean)
    }

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
}

#[cfg(test)]
mod tests {
    //! sc-9037 (F-053): the streaming decode/encode loops used to accumulate frame chunks by
    //! `cat`-ing onto a growing accumulator each iteration — O(T²) copy traffic. The fix collects
    //! the chunks into a `Vec<Tensor>` and does a single `Tensor::cat` at the end (O(T)). This must
    //! be **bit-identical** to the old incremental accumulation (same frames, same temporal order,
    //! no boundary blend — the streaming loops here are a plain temporal concatenation). These tests
    //! pin that equivalence at the tensor level so the refactor is provably output-preserving
    //! without needing real VAE weights.

    use candle_gen::candle_core::{DType, Device, Tensor};

    /// Old accumulator: `out = cat([out, chunk], dim)` folded over the chunk list (the pre-sc-9037
    /// pattern). Returns `None` for an empty list (mirrors the old `Option<Tensor>` seed).
    fn incremental_cat(chunks: &[Tensor], dim: usize) -> Option<Tensor> {
        let mut out: Option<Tensor> = None;
        for c in chunks {
            out = Some(match out {
                Some(o) => Tensor::cat(&[&o, c], dim).unwrap(),
                None => c.clone(),
            });
        }
        out
    }

    /// A known sequence of temporal chunks shaped like the streaming decode output
    /// `[B, C, t_i, H, W]` with per-chunk frame counts mimicking `1 + (T-1)·4` (frame 0 → 1 output
    /// frame, later latent frames → 4). Deterministic ascending values so any reorder/duplication
    /// would change the bytes.
    fn make_chunks(dev: &Device) -> Vec<Tensor> {
        let (b, c, h, w) = (1usize, 3usize, 2usize, 2usize);
        let per_chunk_frames = [1usize, 4, 4, 4, 4]; // 5 latent frames → 17 output frames
        let mut base = 0f32;
        let mut chunks = Vec::new();
        for &tf in &per_chunk_frames {
            let n = b * c * tf * h * w;
            let data: Vec<f32> = (0..n).map(|k| base + k as f32).collect();
            base += n as f32;
            chunks.push(Tensor::from_vec(data, (b, c, tf, h, w), dev).unwrap());
        }
        chunks
    }

    /// The single-cat replacement is byte-for-byte identical to the incremental accumulator over a
    /// realistic multi-frame chunk sequence (the decode temporal axis, dim 2).
    #[test]
    fn single_cat_matches_incremental_accumulation() {
        let dev = Device::Cpu;
        let chunks = make_chunks(&dev);

        let single = Tensor::cat(&chunks, 2).unwrap();
        let incremental = incremental_cat(&chunks, 2).expect("non-empty");

        assert_eq!(single.dims(), incremental.dims(), "shape must match");
        let a = single
            .flatten_all()
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let b = incremental
            .flatten_all()
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        // Bit-identical (concatenation copies bytes verbatim — no arithmetic, no blend).
        assert_eq!(a, b, "single-cat must be bit-identical to incremental cat");
    }

    /// The total output frame count along the temporal axis is `1 + (T-1)·4` — the streaming
    /// contract — and is preserved by the single cat.
    #[test]
    fn single_cat_preserves_frame_count() {
        let dev = Device::Cpu;
        let chunks = make_chunks(&dev);
        let single = Tensor::cat(&chunks, 2).unwrap();
        let t_lat = chunks.len();
        assert_eq!(single.dim(2).unwrap(), 1 + (t_lat - 1) * 4);
    }

    /// F-126 (sc-11220): `encode` must reject a temporal length it cannot consume without dropping
    /// trailing frames. Aligned `1 + 4·k` lengths pass; any other (including 0) errors, and the
    /// message reports how many frames would have been dropped.
    #[test]
    fn require_aligned_encode_frames_gates_unaligned_lengths() {
        for t in [1usize, 5, 9, 17, 33] {
            assert!(
                super::require_aligned_encode_frames(t).is_ok(),
                "t={t} is 1+4k and must be accepted"
            );
        }
        // t=0 and every t with (t-1)%4 != 0 are rejected.
        assert!(super::require_aligned_encode_frames(0).is_err());
        for (t, dropped) in [(2usize, 1usize), (3, 2), (4, 3), (18, 1), (20, 3)] {
            let err = super::require_aligned_encode_frames(t)
                .expect_err("unaligned length must be rejected");
            let msg = err.to_string();
            assert!(
                msg.contains(&format!("{dropped}")) && msg.contains(&format!("got {t}")),
                "message reports dropped count and t: {msg}"
            );
        }
    }
}
