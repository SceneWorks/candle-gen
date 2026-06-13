//! **Causal 3-D convolution** for the Wan temporal VAE — candle ships no `conv3d`, and because video
//! has `T > 1` the conv3d does *not* reduce to a single conv2d (unlike a single-image VAE). Instead a
//! `kD×kH×kW` kernel is decomposed into `kD` conv2d "taps": the temporal axis is causally left-padded
//! by `kD-1` zero frames, and the output is `Σ_{kd} conv2d(x_pad[:, :, kd : kd+T], W[:, :, kd])`.
//!
//! This reproduces diffusers' `WanCausalConv3d` exactly (its `_padding = (·, ·, ·, ·, 2·pad_t, 0)`
//! left-pad + VALID conv, temporal stride 1), and is mathematically identical to the reference's
//! frame-by-frame `feat_cache` streaming (a causal conv over the whole clip == the cache-streamed
//! one) — so a single pass over all `T` frames matches.

use candle_gen::candle_core::{Result, Tensor};
use candle_gen::candle_nn::VarBuilder;

/// A causal Conv3d loaded from a diffusers `[O, I, kD, kH, kW]` weight. Temporal stride is always 1
/// in the Wan decoder; spatial padding is "same" (`(kH-1)/2`), temporal padding is causal (left
/// `kD-1`).
pub struct CausalConv3d {
    weight: Tensor, // [O, I, kD, kH, kW]
    bias: Tensor,   // [1, O, 1, 1, 1]
    kd: usize,
    spatial_pad: usize,
}

impl CausalConv3d {
    /// Load from `vb` with an explicit kernel `(kD, kH, kW)` (kH == kW) and channel counts.
    pub fn load(
        in_c: usize,
        out_c: usize,
        kernel: (usize, usize, usize),
        vb: VarBuilder,
    ) -> Result<Self> {
        let (kd, kh, kw) = kernel;
        let weight = vb.get((out_c, in_c, kd, kh, kw), "weight")?.contiguous()?;
        let bias = vb.get(out_c, "bias")?.reshape((1, out_c, 1, 1, 1))?;
        Ok(Self {
            weight,
            bias,
            kd,
            spatial_pad: (kh - 1) / 2,
        })
    }

    /// `x`: `[B, C, T, H, W]` → `[B, O, T, H, W]` (spatial "same", temporal causal).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, t, h, w) = x.dims5()?;
        // Causal temporal left-pad by kD-1 zero frames.
        let xpad = if self.kd > 1 {
            x.pad_with_zeros(2, self.kd - 1, 0)?
        } else {
            x.clone()
        };
        let mut acc: Option<Tensor> = None;
        for kd in 0..self.kd {
            // Tap weight W[:, :, kd] → [O, I, kH, kW].
            let wk = self.weight.narrow(2, kd, 1)?.squeeze(2)?.contiguous()?;
            // The T frames this tap convolves: x_pad[:, :, kd : kd+T].
            let frames = xpad.narrow(2, kd, t)?;
            // Merge (B, T) into the conv2d batch axis: [B, C, T, H, W] → [B*T, C, H, W].
            let merged = frames
                .permute((0, 2, 1, 3, 4))?
                .reshape((b * t, c, h, w))?
                .contiguous()?;
            let y = merged.conv2d(&wk, self.spatial_pad, 1, 1, 1)?;
            acc = Some(match acc {
                Some(a) => (a + y)?,
                None => y,
            });
        }
        let y = acc.expect("kD >= 1");
        let (_, o, hp, wp) = y.dims4()?;
        let y = y
            .reshape((b, t, o, hp, wp))?
            .permute((0, 2, 1, 3, 4))?
            .contiguous()?;
        y.broadcast_add(&self.bias)
    }
}
