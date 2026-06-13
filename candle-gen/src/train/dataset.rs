//! Dataset preparation for training (sc-5165) — the candle twin of `mlx_gen::train::dataset`.
//!
//! Unlike the MLX harness (which operates on a pre-decoded `Image` so the core crate stays
//! decode-free), candle-gen links the `image` crate, so this module owns the full file → tensor path:
//! decode → center-crop to a square → resize to the bucketed edge (Lanczos, matching the Python
//! kernel's `Image.LANCZOS`) → normalize to `[-1, 1]` as a VAE-input tensor `[1, 3, edge, edge]`
//! (RGB, channel-first). Resolution is bucketed to a multiple of 32 (the latent grid must tile
//! cleanly), mirroring the Python `bucket_resolution`.

use std::path::Path;

use candle_core::{Device, Tensor};
use image::imageops::FilterType;

use crate::{CandleError, Result};

/// Floor `resolution` to a multiple of 32 (the training bucket). `0` → the `512` default; otherwise
/// `(res/32)*32` with a 32-px floor so a tiny nonzero input never collapses to 0. Mirrors the Python
/// `bucket_resolution` (and the MLX twin).
pub fn bucket_resolution(resolution: u32) -> u32 {
    if resolution == 0 {
        return 512;
    }
    ((resolution / 32) * 32).max(32)
}

/// Decode `path`, center-crop to its largest centered square, resize to `edge`×`edge` (Lanczos), and
/// return a VAE-input tensor `[1, 3, edge, edge]` (RGB, channel-first) normalized to `[-1, 1]` on
/// `device` — exactly the Python `_load_training_image` + `_image_to_tensor` pipeline
/// (`array/127.5 - 1.0`, `permute(2,0,1).unsqueeze(0)`).
pub fn load_image_tensor(path: &Path, edge: u32, device: &Device) -> Result<Tensor> {
    let img = image::open(path)
        .map_err(|e| CandleError::Msg(format!("open image {}: {e}", path.display())))?
        .to_rgb8();
    let (w, h) = img.dimensions();
    let side = w.min(h);
    let (x0, y0) = ((w - side) / 2, (h - side) / 2);
    let cropped = image::imageops::crop_imm(&img, x0, y0, side, side).to_image();
    let resized = image::imageops::resize(&cropped, edge, edge, FilterType::Lanczos3);

    let (ew, eh) = (edge as usize, edge as usize);
    let mut data = vec![0f32; 3 * eh * ew];
    for (x, y, px) in resized.enumerate_pixels() {
        let (x, y) = (x as usize, y as usize);
        for c in 0..3 {
            // channel-first [3, H, W]; RGB; [-1, 1].
            data[c * eh * ew + y * ew + x] = px[c] as f32 / 127.5 - 1.0;
        }
    }
    Ok(Tensor::from_vec(data, (1, 3, eh, ew), &Device::Cpu)?.to_device(device)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucket_floors_to_multiple_of_32() {
        assert_eq!(bucket_resolution(1024), 1024);
        assert_eq!(bucket_resolution(1000), 992);
        assert_eq!(bucket_resolution(1023), 992);
        assert_eq!(bucket_resolution(0), 512);
        assert_eq!(bucket_resolution(16), 32);
        assert_eq!(bucket_resolution(512), 512);
    }
}
