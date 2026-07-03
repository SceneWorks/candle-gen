//! LTX-2.3 **audio VAE decoder** (sc-5495) — candle (NCHW) port of `mlx-gen-ltx` `audio_vae.rs`.
//!
//! A 2-D conv autoencoder *decoder*, **causal on the height (time) axis**, PixelNorm, `ch 128`,
//! `ch_mult (1,2,4)`, `z_channels 8`, `out_ch 2` (stereo), `num_res_blocks 2`, nearest-2× causal
//! upsample. Decodes the audio latent `(B, 8, T, 16)` → mel spectrogram `(B, 2, 4T−3, 64)` that the
//! [`crate::vocoder`] turns into a waveform.
//!
//! Loaded from the dense `ltx-2.3-22b-distilled.safetensors` under the `audio_vae.` prefix (the same
//! checkpoint the video stack uses — no separate download). Conv weights are the checkpoint-native
//! PyTorch layout `[O, I, kH, kW]`, so candle's `conv2d` consumes them directly. `mid_block_add_attention`
//! is **false** for the shipped checkpoint (no `mid.attn_1` weights), so the mid block is
//! `ResnetBlock → ResnetBlock`. Runs **f32** (a post-sampling quality island).

use candle_gen::candle_core::{Result, Tensor, D};
use candle_gen::candle_nn::ops::silu;
use candle_gen::candle_nn::VarBuilder;

use crate::config::AudioVaeConfig;

/// PixelNorm epsilon (`build_normalization_layer(..., NormType.PIXEL)` → `PixelNorm(eps=1e-6)`).
const PIXEL_EPS: f64 = 1e-6;

/// Per-location RMS over the channel axis (dim 1, NCHW): `x / sqrt(mean(x², C) + eps)`. No learned γ.
fn pixel_norm(x: &Tensor) -> Result<Tensor> {
    let c = x.dim(1)?;
    let mean = (x.sqr()?.sum_keepdim(1)? / c as f64)?;
    let denom = (mean + PIXEL_EPS)?.sqrt()?;
    x.broadcast_div(&denom)
}

/// A 2-D convolution with asymmetric (causal-on-height) or symmetric padding applied manually, then
/// `conv2d(padding=0)`. NCHW; weight `[O, I, kH, kW]`.
struct CausalConv2d {
    w: Tensor, // [O, I, kH, kW]
    b: Tensor, // [1, O, 1, 1]
    pad_top: usize,
    pad_bottom: usize,
    pad_left: usize,
    pad_right: usize,
}

impl CausalConv2d {
    /// `causal_height = true` → pad the full `kH−1` on top (time is causal); width is symmetric.
    fn load(vb: &VarBuilder, prefix: &str, causal_height: bool) -> Result<Self> {
        // Audio-VAE convs are never MLX-affine-packed; guard against an unexpected `.scales` sibling
        // (sc-9417).
        let w = crate::quant::guard_no_scales(vb, prefix, vb.dtype())?.contiguous()?; // (O, I, kH, kW)
        let dims = w.dims();
        let (out_c, kh, kw) = (dims[0], dims[2], dims[3]);
        let b = vb
            .get_unchecked(&format!("{prefix}.bias"))?
            .reshape((1, out_c, 1, 1))?;
        let (ph, pw) = (kh - 1, kw - 1);
        let (pad_top, pad_bottom) = if causal_height {
            (ph, 0)
        } else {
            (ph / 2, ph - ph / 2)
        };
        Ok(Self {
            w,
            b,
            pad_top,
            pad_bottom,
            pad_left: pw / 2,
            pad_right: pw - pw / 2,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        if self.pad_top + self.pad_bottom > 0 {
            x = x.pad_with_zeros(2, self.pad_top, self.pad_bottom)?;
        }
        if self.pad_left + self.pad_right > 0 {
            x = x.pad_with_zeros(3, self.pad_left, self.pad_right)?;
        }
        let y = x.conv2d(&self.w, 0, 1, 1, 1)?;
        y.broadcast_add(&self.b)
    }
}

/// 2-D ResNet block (`ResnetBlock`): PixelNorm → SiLU → conv → PixelNorm → SiLU → conv, plus a
/// 1×1 `nin_shortcut` when `in != out`.
struct ResnetBlock {
    conv1: CausalConv2d,
    conv2: CausalConv2d,
    nin_shortcut: Option<CausalConv2d>,
}

impl ResnetBlock {
    fn load(vb: &VarBuilder, prefix: &str) -> Result<Self> {
        let nin_key = format!("{prefix}.nin_shortcut.conv.weight");
        let nin_shortcut = if vb.contains_tensor(&nin_key) {
            Some(CausalConv2d::load(
                vb,
                &format!("{prefix}.nin_shortcut.conv"),
                true,
            )?)
        } else {
            None
        };
        Ok(Self {
            conv1: CausalConv2d::load(vb, &format!("{prefix}.conv1.conv"), true)?,
            conv2: CausalConv2d::load(vb, &format!("{prefix}.conv2.conv"), true)?,
            nin_shortcut,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.conv1.forward(&silu(&pixel_norm(x)?)?)?;
        let h = self.conv2.forward(&silu(&pixel_norm(&h)?)?)?;
        let shortcut = match &self.nin_shortcut {
            Some(c) => c.forward(x)?,
            None => x.clone(),
        };
        shortcut + h
    }
}

/// Upsample stage (`Upsample`): nearest-2× (H & W) → causal conv → **drop the first time element**
/// (undoes the encoder's causal padding, keeping length `2n−1`).
struct Upsample {
    conv: CausalConv2d,
}

impl Upsample {
    fn load(vb: &VarBuilder, prefix: &str) -> Result<Self> {
        Ok(Self {
            conv: CausalConv2d::load(vb, &format!("{prefix}.conv.conv"), true)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_, _, h, w) = x.dims4()?;
        let up = x.upsample_nearest2d(2 * h, 2 * w)?;
        let y = self.conv.forward(&up)?;
        // Drop the first element along the causal (height/time) axis.
        let th = y.dim(2)?;
        y.narrow(2, 1, th - 1)
    }
}

/// One decoder up-level: `num_res_blocks + 1` ResnetBlocks + optional Upsample.
struct UpLevel {
    blocks: Vec<ResnetBlock>,
    upsample: Option<Upsample>,
}

/// The LTX-2.3 audio VAE decoder. `conv_in → mid → up-levels → PixelNorm → SiLU → conv_out`.
pub struct AudioDecoder {
    conv_in: CausalConv2d,
    mid_block_1: ResnetBlock,
    mid_block_2: ResnetBlock,
    up: Vec<UpLevel>, // index = level (0..num_resolutions), run high→low (reversed) at decode
    conv_out: CausalConv2d,
    mean_of_means: Tensor, // (128,)
    std_of_means: Tensor,
    out_ch: usize,
    mel_bins: usize,
    downsample_factor: usize,
}

impl AudioDecoder {
    /// Build from a `VarBuilder` positioned at the `audio_vae` prefix + the [`AudioVaeConfig`].
    pub fn load(vb: &VarBuilder, cfg: &AudioVaeConfig) -> Result<Self> {
        let num_res = cfg.num_resolutions();
        let mut up = Vec::with_capacity(num_res);
        for level in 0..num_res {
            let mut blocks = Vec::with_capacity((cfg.num_res_blocks + 1) as usize);
            for i_block in 0..(cfg.num_res_blocks + 1) {
                blocks.push(ResnetBlock::load(
                    vb,
                    &format!("decoder.up.{level}.block.{i_block}"),
                )?);
            }
            let up_key = format!("decoder.up.{level}.upsample.conv.conv.weight");
            let upsample = if vb.contains_tensor(&up_key) {
                Some(Upsample::load(vb, &format!("decoder.up.{level}.upsample"))?)
            } else {
                None
            };
            up.push(UpLevel { blocks, upsample });
        }

        Ok(Self {
            conv_in: CausalConv2d::load(vb, "decoder.conv_in.conv", true)?,
            mid_block_1: ResnetBlock::load(vb, "decoder.mid.block_1")?,
            mid_block_2: ResnetBlock::load(vb, "decoder.mid.block_2")?,
            up,
            conv_out: CausalConv2d::load(vb, "decoder.conv_out.conv", true)?,
            mean_of_means: vb
                .get_unchecked("per_channel_statistics.mean-of-means")?
                .contiguous()?,
            std_of_means: vb
                .get_unchecked("per_channel_statistics.std-of-means")?
                .contiguous()?,
            out_ch: cfg.out_ch as usize,
            mel_bins: cfg.mel_bins as usize,
            downsample_factor: crate::config::AUDIO_LATENT_DOWNSAMPLE_FACTOR as usize,
        })
    }

    /// Denormalize the latent `(B, C, T, F)` (NCHW): patchify over `(C, F)` per time step, `·std +
    /// mean`, unpatchify. Matches `AudioPatchifier` + `PerChannelStatistics.un_normalize`.
    fn denormalize(&self, sample: &Tensor) -> Result<Tensor> {
        let (b, c, t, f) = sample.dims4()?;
        // (B,C,T,F) → (B,T,C,F) → (B,T,C·F). C-major flatten matches the reference patchifier.
        let patched = sample
            .permute((0, 2, 1, 3))?
            .reshape((b, t, c * f))?
            .contiguous()?;
        let denorm = patched
            .broadcast_mul(&self.std_of_means)?
            .broadcast_add(&self.mean_of_means)?;
        // unpatchify: (B,T,C·F) → (B,T,C,F) → (B,C,T,F).
        denorm
            .reshape((b, t, c, f))?
            .permute((0, 2, 1, 3))?
            .contiguous()
    }

    /// Decode an audio latent `(B, z=8, T, 16)` (NCHW) → mel spectrogram `(B, out_ch=2, T', 64)`.
    pub fn decode(&self, latent: &Tensor) -> Result<Tensor> {
        let latent = latent.to_dtype(candle_gen::candle_core::DType::F32)?;
        let (_, _, frames, _latent_mel) = latent.dims4()?;
        let sample = self.denormalize(&latent)?;

        let mut h = self.conv_in.forward(&sample)?;
        h = self.mid_block_1.forward(&h)?;
        h = self.mid_block_2.forward(&h)?;

        // Up path, high → low level (reversed); upsample on levels != 0.
        for (idx, stage) in self.up.iter().enumerate().rev() {
            for block in &stage.blocks {
                h = block.forward(&h)?;
            }
            if idx != 0 {
                if let Some(up) = &stage.upsample {
                    h = up.forward(&h)?;
                }
            }
        }

        h = self.conv_out.forward(&silu(&pixel_norm(&h)?)?)?;

        // Crop/pad to the causal target frame count + target mel bins, channels = out_ch.
        let target_frames = {
            let f = frames * self.downsample_factor;
            f.saturating_sub(self.downsample_factor - 1).max(1)
        };
        let target_mel = if self.mel_bins > 0 {
            self.mel_bins
        } else {
            h.dim(D::Minus1)?
        };
        self.adjust(&h, target_frames, target_mel)
    }

    /// Crop-then-pad NCHW `(B, C, time, freq)` to `(B, out_ch, target_time, target_freq)`.
    fn adjust(&self, x: &Tensor, target_time: usize, target_freq: usize) -> Result<Tensor> {
        let (_, c, cur_t, cur_f) = x.dims4()?;
        let crop_t = cur_t.min(target_time);
        let crop_f = cur_f.min(target_freq);
        let crop_c = c.min(self.out_ch);
        let mut x = x
            .narrow(1, 0, crop_c)?
            .narrow(2, 0, crop_t)?
            .narrow(3, 0, crop_f)?;
        let pad_t = target_time.saturating_sub(x.dim(2)?);
        let pad_f = target_freq.saturating_sub(x.dim(3)?);
        if pad_t > 0 {
            x = x.pad_with_zeros(2, 0, pad_t)?;
        }
        if pad_f > 0 {
            x = x.pad_with_zeros(3, 0, pad_f)?;
        }
        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Device;

    #[test]
    fn pixel_norm_unit_rms() {
        // (1, C=4, 1, 1): RMS over channels → unit RMS output.
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 4, 1, 1), &Device::Cpu).unwrap();
        let y = pixel_norm(&x).unwrap();
        let s = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let ms = s.iter().map(|v| v * v).sum::<f32>() / 4.0;
        assert!((ms - 1.0).abs() < 1e-4, "rms² = {ms}");
    }
}
