//! Generalized **CausalConv3d** for the SeedVR2 3D causal video VAE — candle ships no `conv3d`, so a
//! `kT×kH×kW` kernel is decomposed into `kT` conv2d "taps" summed over the temporal axis (the proven
//! candle pattern: `candle-gen-wan`/`-svd` + candle-transformers `qwen3_vl::conv3d_temporal_2`). This
//! generalizes the Wan/SVD temporal-only convs to SeedVR2's needs:
//!   * spatial kernels + spatial stride (the stride-2 downsample convs) + symmetric spatial padding,
//!   * temporal stride 2 (the temporal downsamplers),
//!   * **first-frame-repeat** causal temporal padding — NOT zero-pad — matching the mflux reference
//!     `common/conv3d.py` (`causal_pad = use_padding_causal ? 2·pt : kt-1`, repeat frame 0).
//!
//! NCTHW in/out. Weight is torch-layout `[O, I, kT, kH, kW]` (loaded as-is: candle conv2d's per-tap
//! kernel is exactly `W[:, :, kd] = [O, I, kH, kW]`). All SeedVR2 convs have `ph == pw` and `sh == sw`
//! (candle conv2d takes one symmetric padding/stride); the asymmetric `(0,1)` downsample pad is added
//! by the caller (`vae::Downsample3d`) before the conv, exactly like the reference.

use candle_gen::candle_core::{Result, Tensor};

use crate::weights::Weights;

pub struct CausalConv3d {
    weight: Tensor, // [O, I, kT, kH, kW]
    bias: Tensor,   // [1, O, 1, 1, 1]
    kt: usize,
    st: usize,
    sh: usize,
    #[allow(dead_code)]
    sw: usize,
    pt: usize,
    ph: usize,
    #[allow(dead_code)]
    pw: usize,
    use_padding_causal: bool,
}

impl CausalConv3d {
    /// Load `{prefix}.weight` (`[O,I,kT,kH,kW]`) + `{prefix}.bias` (`[O]`). `stride`/`padding` are
    /// `(t,h,w)`; `use_padding_causal` selects the `2·pt` causal-pad rule (else `kt-1`).
    pub fn load(
        w: &Weights,
        prefix: &str,
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
        use_padding_causal: bool,
    ) -> candle_gen::Result<Self> {
        let weight = w.require(&format!("{prefix}.weight"))?.clone();
        let (_o, _i, kt, _kh, _kw) = weight.dims5()?;
        let bias = w.require(&format!("{prefix}.bias"))?;
        let o = bias.dim(0)?;
        let bias = bias.reshape((1, o, 1, 1, 1))?;
        debug_assert_eq!(padding.1, padding.2, "seedvr2 conv: ph == pw");
        debug_assert_eq!(stride.1, stride.2, "seedvr2 conv: sh == sw");
        Ok(Self {
            weight,
            bias,
            kt,
            st: stride.0,
            sh: stride.1,
            sw: stride.2,
            pt: padding.0,
            ph: padding.1,
            pw: padding.2,
            use_padding_causal,
        })
    }

    /// `x`: `[B, C, T, H, W]` → `[B, O, T_out, H_out, W_out]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, _t, h, w) = x.dims5()?;
        // Temporal causal padding (repeat the first frame), or symmetric zero-pad for kt==1.
        let xpad = if self.kt > 1 {
            let causal_pad = if self.use_padding_causal {
                2 * self.pt
            } else {
                self.kt - 1
            };
            if causal_pad > 0 {
                let first = x.narrow(2, 0, 1)?; // [B,C,1,H,W]
                let rep = first.broadcast_as((b, c, causal_pad, h, w))?.contiguous()?;
                Tensor::cat(&[&rep, x], 2)?
            } else {
                x.clone()
            }
        } else if self.pt > 0 {
            x.pad_with_zeros(2, self.pt, self.pt)?
        } else {
            x.clone()
        };
        let tpad = xpad.dim(2)?;
        // VALID temporal conv with temporal stride → T_out output frames.
        let t_out = (tpad - self.kt) / self.st + 1;
        let dev = x.device();

        let mut acc: Option<Tensor> = None;
        for kd in 0..self.kt {
            // Tap weight W[:, :, kd] → [O, I, kH, kW] (one conv2d).
            let wk = self.weight.narrow(2, kd, 1)?.squeeze(2)?.contiguous()?;
            // The T_out frames this tap convolves: x_pad[:, :, kd + o·st], o in 0..T_out.
            let frames = if self.st == 1 {
                xpad.narrow(2, kd, t_out)?
            } else {
                let idx: Vec<u32> = (0..t_out).map(|o| (kd + o * self.st) as u32).collect();
                let idx = Tensor::from_vec(idx, t_out, dev)?;
                xpad.index_select(&idx, 2)?
            };
            // Merge (B, T_out) into the conv2d batch axis: [B, C, T_out, H, W] → [B·T_out, C, H, W].
            let merged = frames
                .permute((0, 2, 1, 3, 4))?
                .reshape((b * t_out, c, h, w))?
                .contiguous()?;
            let y = merged.conv2d(&wk, self.ph, self.sh, 1, 1)?; // [B·T_out, O, H_out, W_out]
            acc = Some(match acc {
                Some(a) => (a + y)?,
                None => y,
            });
        }
        let y = acc.expect("kt >= 1");
        let (_, o, hp, wp) = y.dims4()?;
        let y = y
            .reshape((b, t_out, o, hp, wp))?
            .permute((0, 2, 1, 3, 4))?
            .contiguous()?;
        y.broadcast_add(&self.bias)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::{DType, Device};

    fn conv(
        weight: Tensor,
        bias: Tensor,
        st: usize,
        ph: usize,
        pt: usize,
        upc: bool,
    ) -> CausalConv3d {
        let (_o, _i, kt, _kh, _kw) = weight.dims5().unwrap();
        let o = bias.dim(0).unwrap();
        CausalConv3d {
            weight,
            bias: bias.reshape((1, o, 1, 1, 1)).unwrap(),
            kt,
            st,
            sh: 1,
            sw: 1,
            pt,
            ph,
            pw: ph,
            use_padding_causal: upc,
        }
    }

    /// A 3×3×3 causal conv (stride 1, spatial pad 1) preserves T,H,W (causal repeat = 2 frames).
    #[test]
    fn k333_preserves_dims() -> Result<()> {
        let dev = Device::Cpu;
        let (o, i) = (4usize, 3usize);
        let c = conv(
            Tensor::randn(0f32, 1.0, (o, i, 3, 3, 3), &dev)?,
            Tensor::zeros(o, DType::F32, &dev)?,
            1,
            1,
            1,
            false,
        );
        let x = Tensor::randn(0f32, 1.0, (1, i, 5, 8, 8), &dev)?;
        let y = c.forward(&x)?;
        assert_eq!(y.dims(), &[1, o, 5, 8, 8]);
        Ok(())
    }

    /// A temporal-stride-2 downsample conv (3×3×3, causal pad 2) halves T (ceil) — the encoder
    /// temporal down block.
    #[test]
    fn temporal_stride2_halves_t() -> Result<()> {
        let dev = Device::Cpu;
        let (o, i) = (2usize, 2usize);
        // stride (2,1,1), spatial pad 1 here for shape simplicity.
        let c = conv(
            Tensor::randn(0f32, 1.0, (o, i, 3, 3, 3), &dev)?,
            Tensor::zeros(o, DType::F32, &dev)?,
            2,
            1,
            1,
            false,
        );
        let x = Tensor::randn(0f32, 1.0, (1, i, 9, 4, 4), &dev)?;
        let y = c.forward(&x)?; // T_out = (9+2-3)/2 + 1 = 5
        assert_eq!(y.dims(), &[1, o, 5, 4, 4]);
        Ok(())
    }

    /// A 1×1×1 conv is pointwise and preserves all dims (the resnet shortcut / upscale conv).
    #[test]
    fn k111_pointwise() -> Result<()> {
        let dev = Device::Cpu;
        let (o, i) = (5usize, 3usize);
        let c = conv(
            Tensor::randn(0f32, 1.0, (o, i, 1, 1, 1), &dev)?,
            Tensor::zeros(o, DType::F32, &dev)?,
            1,
            0,
            0,
            false,
        );
        let x = Tensor::randn(0f32, 1.0, (1, i, 6, 3, 3), &dev)?;
        let y = c.forward(&x)?;
        assert_eq!(y.dims(), &[1, o, 6, 3, 3]);
        Ok(())
    }
}
