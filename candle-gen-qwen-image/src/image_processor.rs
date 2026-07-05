//! Qwen2-VL image processor — candle (Windows/CUDA) port of `mlx-gen-qwen-image`'s
//! [`crate::image_processor`], used by Qwen-Image-Edit's reference flow. Pipeline:
//! `smart_resize` → PIL-compatible BICUBIC resize → `/255` → CLIP normalize → temporal-repeat →
//! patchify → `(N, 1176)` pixel_values + the `(grid_t, grid_h, grid_w)` grid.
//!
//! The PIL-exact resampler ([`resize_bicubic_u8`]) lives in gen-core ([`candle_gen::gen_core::imageops`]);
//! this module is otherwise pure index math, so it is unit-tested on CPU without any model weights.

use candle_gen::candle_core::{Device, Tensor};
use candle_gen::gen_core::imageops::resize_bicubic_u8;
use candle_gen::{CandleError, Result};

/// `(grid_t, grid_h, grid_w)` for one image, in **patch** units (`image_px / patch_size`).
pub type Grid = [i32; 3];

pub const OPENAI_CLIP_MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
pub const OPENAI_CLIP_STD: [f32; 3] = [0.268_629_54, 0.261_302_6, 0.275_777_1];

/// RGB uint8 image in HWC layout.
pub struct ImageInput<'a> {
    pub data: &'a [u8],
    pub height: usize,
    pub width: usize,
}

/// Patchified output ready for the vision encoder.
pub struct ProcessedImage {
    /// `(grid_t·grid_h·grid_w, channel·temporal·patch·patch)` = `(N, 1176)`, f32, on the device.
    pub pixel_values: Tensor,
    /// `[grid_t, grid_h, grid_w]` (patch units).
    pub grid: Grid,
}

#[derive(Debug, Clone)]
pub struct QwenImageProcessor {
    pub min_pixels: i64,
    pub max_pixels: i64,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub merge_size: usize,
}

impl Default for QwenImageProcessor {
    fn default() -> Self {
        Self {
            min_pixels: 56 * 56,
            max_pixels: 28 * 28 * 1280,
            patch_size: 14,
            temporal_patch_size: 2,
            merge_size: 2,
        }
    }
}

impl QwenImageProcessor {
    /// Integer target dims: round each side to a multiple of `patch_size*merge_size`, then clamp the
    /// pixel count into `[min_pixels, max_pixels]`. Uses Python's round-half-to-even
    /// (`round_ties_even`) so `.5` cases match the fork exactly.
    pub fn smart_resize(&self, height: usize, width: usize) -> (usize, usize) {
        let factor = (self.patch_size * self.merge_size) as f64;
        let (h, w) = (height as f64, width as f64);
        let (minp, maxp) = (self.min_pixels as f64, self.max_pixels as f64);

        let mut h_bar = (h / factor).round_ties_even() * factor;
        let mut w_bar = (w / factor).round_ties_even() * factor;
        if h_bar * w_bar > maxp {
            let beta = ((h * w) / maxp).sqrt();
            h_bar = factor.max((h / beta / factor).floor() * factor);
            w_bar = factor.max((w / beta / factor).floor() * factor);
        } else if h_bar * w_bar < minp {
            let beta = (minp / (h * w)).sqrt();
            h_bar = (h * beta / factor).ceil() * factor;
            w_bar = (w * beta / factor).ceil() * factor;
        }
        (h_bar as usize, w_bar as usize)
    }

    /// `smart_resize` → BICUBIC resize → CLIP-normalize → temporal-repeat → patchify, returning
    /// `pixel_values [N, 1176]` (on `device`) + the `[grid_t, grid_h, grid_w]` grid (`grid_t == 1`).
    pub fn preprocess(&self, image: ImageInput, device: &Device) -> Result<ProcessedImage> {
        if image.height == 0 || image.width == 0 {
            return Err(CandleError::Msg(format!(
                "qwen image processor: zero dimension ({}x{})",
                image.width, image.height
            )));
        }
        if image.data.len() != image.height * image.width * 3 {
            return Err(CandleError::Msg(format!(
                "qwen image processor: pixel buffer {} != {}x{}x3",
                image.data.len(),
                image.width,
                image.height
            )));
        }
        let (rh, rw) = self.smart_resize(image.height, image.width);

        // Resize on the uint8 image (matching PIL), producing f32 HWC in [0,255].
        let resized: Vec<f32> = if (image.height, image.width) == (rh, rw) {
            image.data.iter().map(|&p| p as f32).collect()
        } else {
            resize_bicubic_u8(image.data, image.height, image.width, rh, rw)?
        };

        // /255, CLIP-normalize, laid out as [t, c, rh, rw] (the single frame repeated across the
        // temporal axis, mirroring the fork's `np.repeat`).
        let (c, t) = (3usize, self.temporal_patch_size);
        let plane = rh * rw;
        let mut chw = vec![0f32; t * c * plane];
        for ch in 0..c {
            let (mean, std) = (OPENAI_CLIP_MEAN[ch], OPENAI_CLIP_STD[ch]);
            for y in 0..rh {
                for x in 0..rw {
                    let v = (resized[(y * rw + x) * c + ch] / 255.0 - mean) / std;
                    let chw_idx = ch * plane + y * rw + x;
                    for frame in 0..t {
                        chw[frame * c * plane + chw_idx] = v;
                    }
                }
            }
        }

        // Patchify to the fork's (grid_h·grid_w, channel·temporal·patch²) layout. The fork reshapes
        // to (1, t, c, gh/m, m, p, gw/m, m, p), transposes (0,3,6,4,7,2,1,5,8), and flattens; that
        // makes the **row** order (merger-block-row, merger-block-col, within-row, within-col) — so
        // each `merge²` patch group is contiguous (the merger relies on this) — and the **feature**
        // order (channel, temporal, patch_y, patch_x), matching the PyTorch conv weight's flatten.
        let p = self.patch_size;
        let m = self.merge_size;
        let (gh, gw) = (rh / p, rw / p);
        let feat = c * t * p * p; // 1176
        let mut pixel_values = vec![0f32; gh * gw * feat];
        let mut row = 0usize;
        for bh in 0..gh / m {
            for bw in 0..gw / m {
                for mr in 0..m {
                    for mc in 0..m {
                        let gy = bh * m + mr;
                        let gx = bw * m + mc;
                        let mut f = row * feat;
                        for ch in 0..c {
                            for ft in 0..t {
                                for py in 0..p {
                                    for px in 0..p {
                                        let sy = gy * p + py;
                                        let sx = gx * p + px;
                                        pixel_values[f] =
                                            chw[ft * c * plane + ch * plane + sy * rw + sx];
                                        f += 1;
                                    }
                                }
                            }
                        }
                        row += 1;
                    }
                }
            }
        }

        let pixel_values =
            Tensor::from_vec(pixel_values, (gh * gw, feat), device).map_err(CandleError::from)?;
        Ok(ProcessedImage {
            pixel_values,
            grid: [1, gh as i32, gw as i32],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smart_resize_matches_reference_cases() {
        let p = QwenImageProcessor::default();
        assert_eq!(p.smart_resize(56, 84), (56, 84)); // already aligned -> no-op
        assert_eq!(p.smart_resize(200, 150), (196, 140)); // downscale
        assert_eq!(p.smart_resize(20, 20), (56, 56)); // upscale to min_pixels
    }

    #[test]
    fn preprocess_shape_and_grid() {
        let proc = QwenImageProcessor::default();
        // 56x84 is already /28-aligned → grid (1, 56/14, 84/14) = (1, 4, 6), N = 24 patches.
        let img = vec![128u8; 56 * 84 * 3];
        let out = proc
            .preprocess(
                ImageInput {
                    data: &img,
                    height: 56,
                    width: 84,
                },
                &Device::Cpu,
            )
            .unwrap();
        assert_eq!(out.grid, [1, 4, 6]);
        assert_eq!(out.pixel_values.dims(), &[24, 1176]);
        // A flat-gray image yields a constant per channel after normalize; all entries finite.
        let v = out
            .pixel_values
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(v.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn preprocess_rejects_bad_buffer() {
        let proc = QwenImageProcessor::default();
        let img = vec![0u8; 10];
        assert!(proc
            .preprocess(
                ImageInput {
                    data: &img,
                    height: 8,
                    width: 8,
                },
                &Device::Cpu,
            )
            .is_err());
    }
}
