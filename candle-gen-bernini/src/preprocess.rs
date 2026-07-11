//! Source-media preprocessing — decoded conditioning [`Image`]s (the worker owns file I/O) → resized
//! `[-1,1]` pixel tensors → [`WanVae16::encode`] normalized z16 latents, the `videos`/`images` lists
//! [`crate::forward::guided_velocity`] consumes. The candle sibling of `mlx-gen-bernini/src/preprocess.rs`.
//!
//! Reuses [`candle_gen_wan::wan14b::preprocess_i2v_image`] (cover-fit resize to the target W×H,
//! RGB→`[-1,1]` CHW). Conditioning is resized to the **output** geometry here (a faithful first
//! approximation; the reference resizes each source to its own aspect under `max_image_size` — refine if
//! a media-mode parity gap appears).

use candle_gen::candle_core::{Device, Tensor};
use candle_gen::gen_core::Image;
use candle_gen::{CandleError, Result as CResult};

use candle_gen_wan::vae16::WanVae16;
use candle_gen_wan::wan14b::preprocess_i2v_image;

/// One conditioning image `[1, 16, 1, H/8, W/8]` (z16, normalized): resize → `[-1,1]` `[1,3,1,H,W]` →
/// [`WanVae16::encode`]. The batch axis is kept (the candle DiT latents are batch-first).
pub fn encode_image(
    vae: &WanVae16,
    image: &Image,
    width: u32,
    height: u32,
    device: &Device,
) -> CResult<Tensor> {
    let video = preprocess_i2v_image(image, width, height, device)?; // [1, 3, 1, H, W] in [-1,1]
    Ok(vae.encode(&video)?)
}

/// One conditioning video clip `[1, 16, T_lat, H/8, W/8]`: each frame resized to `[1,3,1,H,W]`, stacked
/// on the temporal axis → `[1,3,T,H,W]` (T must be `1 + 4k`) → [`WanVae16::encode`].
pub fn encode_videoclip(
    vae: &WanVae16,
    frames: &[Image],
    width: u32,
    height: u32,
    device: &Device,
) -> CResult<Tensor> {
    if frames.is_empty() {
        return Err(CandleError::Msg(
            "bernini_renderer: empty conditioning video clip".into(),
        ));
    }
    if frames.len() % 4 != 1 {
        return Err(CandleError::Msg(format!(
            "bernini_renderer: video-clip frame count must be 1 + 4·k (got {})",
            frames.len()
        )));
    }
    let mut per_frame = Vec::with_capacity(frames.len());
    for f in frames {
        per_frame.push(preprocess_i2v_image(f, width, height, device)?); // [1, 3, 1, H, W]
    }
    let refs: Vec<&Tensor> = per_frame.iter().collect();
    let video = Tensor::cat(&refs, 2)?; // [1, 3, T, H, W]
    Ok(vae.encode(&video)?)
}
