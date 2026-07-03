//! Pipeline glue for Wan txt2video: latent geometry, deterministic CPU-seeded noise (sc-3673),
//! classifier-free guidance, and the frames → `gen_core::Image` conversion.

use candle_gen::candle_core::{DType, Device, Result, Tensor};
use candle_gen::gen_core::Image;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::config::{VAE_STRIDE_SPATIAL, VAE_STRIDE_TEMPORAL};

/// Latent dims `(t_lat, h_lat, w_lat)` for `frames × height × width`.
pub fn latent_dims(frames: u32, width: u32, height: u32) -> (usize, usize, usize) {
    let t_lat = (frames - 1) / VAE_STRIDE_TEMPORAL + 1;
    let h_lat = height / VAE_STRIDE_SPATIAL;
    let w_lat = width / VAE_STRIDE_SPATIAL;
    (t_lat as usize, h_lat as usize, w_lat as usize)
}

/// Deterministic N(0,1) latent noise `[1, 48, t_lat, h_lat, w_lat]` (f32) — CPU `StdRng` (ChaCha),
/// launch-portable per seed.
pub fn create_noise(
    seed: u64,
    z_dim: usize,
    t_lat: usize,
    h_lat: usize,
    w_lat: usize,
    device: &Device,
) -> Result<Tensor> {
    let n = z_dim * t_lat * h_lat * w_lat;
    let mut rng = StdRng::seed_from_u64(seed);
    let data = candle_gen::seeded_normal_vec(&mut rng, n);
    Tensor::from_vec(data, (1, z_dim, t_lat, h_lat, w_lat), device)
}

/// Classifier-free guidance: `uncond + g·(cond − uncond)`.
pub fn cfg(cond: &Tensor, uncond: &Tensor, guidance: f64) -> Result<Tensor> {
    uncond + (cond - uncond)?.affine(guidance, 0.0)?
}

/// Decoded video `[1, 3, T, H, W]` in `[-1, 1]` → one RGB8 [`Image`] per frame.
pub fn frames_to_images(decoded: &Tensor) -> Result<Vec<Image>> {
    let u8s = ((decoded.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?
        .to_dtype(DType::U8)?
        .to_device(&Device::Cpu)?;
    let (_b, c, t, h, w) = u8s.dims5()?;
    let frames = u8s.squeeze(0)?; // [3,T,H,W]
    let mut out = Vec::with_capacity(t);
    for ti in 0..t {
        let frame = frames.narrow(1, ti, 1)?.squeeze(1)?; // [3,H,W]
        let pixels = frame.permute((1, 2, 0))?.flatten_all()?.to_vec1::<u8>()?;
        debug_assert_eq!(c, 3);
        out.push(Image {
            width: w as u32,
            height: h as u32,
            pixels,
        });
    }
    Ok(out)
}
