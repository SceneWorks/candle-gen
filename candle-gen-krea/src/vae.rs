//! Krea 2's VAE — the **Qwen-Image** `AutoencoderKLQwenImage` (16 latent channels), reused wholesale
//! from [`candle_gen_qwen_image::vae::QwenVae`]. Port of `mlx-gen-krea`'s `vae.rs` (which reuses
//! `mlx-gen-qwen-image`'s `QwenVae`).
//!
//! The published `krea/Krea-2-Turbo` `vae/config.json` declares `_class_name =
//! "AutoencoderKLQwenImage"` and `_name_or_path = "Qwen/Qwen-Image"`, and the reference loads
//! `AutoencoderKLQwenImage.from_pretrained("Qwen/Qwen-Image", subfolder="vae")` — so the Krea
//! snapshot's `vae/` weights are byte-identical to Qwen-Image's and load through the same module
//! (the provider→provider VAE-reuse precedent: boogu→z-image, kolors→sdxl, ideogram→flux2).
//!
//! De-normalization is **per-channel** `latents_mean`/`latents_std` (a 16-vector, NOT a scalar
//! scale/shift), already baked into [`QwenVae::decode`] (`(z·std) + mean → post_quant_conv → decoder`).

use std::path::Path;

use candle_gen::candle_core::{DType, Device};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::Result;

pub use candle_gen_qwen_image::vae::{QwenVae, QwenVaeEncoder};

/// VAE spatial compression factor (`ae.compression`) — 3 spatial-downsample stages = 8×. With
/// `patch_size = 2` this gives the pipeline's W/H alignment `compression · patch = 16`.
pub const VAE_COMPRESSION: u32 = 8;
/// VAE latent channel count (`ae.channels` = the DiT's `z_dim`).
pub const VAE_CHANNELS: u32 = 16;

/// Build a [`VarBuilder`] over every `.safetensors` in the snapshot's `vae/` dir at f32 (the decode is
/// precision-sensitive; the published `vae/` is f32).
fn vae_varbuilder(dir: &Path, device: &Device) -> Result<VarBuilder<'static>> {
    candle_gen::load_sorted_mmap(dir, DType::F32, device, "krea")
}

/// Load the Qwen-Image VAE (decode) from a Krea snapshot's `vae/` dir. `root` is the **snapshot root**
/// (the `vae/` subdir is joined internally), matching [`crate::config::Krea2Config::from_snapshot`].
pub fn load_vae(root: impl AsRef<Path>, device: &Device) -> Result<QwenVae> {
    let vb = vae_varbuilder(&root.as_ref().join("vae"), device)?;
    Ok(QwenVae::new(vb)?)
}

/// Load the Qwen-Image VAE **encoder** from a Krea snapshot's `vae/` dir (epic 10871 / sc-10877) — the
/// image-edit path VAE-encodes each source reference into the normalized 16-ch latent the DiT's `img_in`
/// consumes ([`QwenVaeEncoder::encode`], the inverse of [`QwenVae::decode`]'s de-normalize). Weights are
/// the same `vae/` dir as [`load_vae`]; the encoder reads the `encoder.*` / `quant_conv` subtrees.
pub fn load_vae_encoder(root: impl AsRef<Path>, device: &Device) -> Result<QwenVaeEncoder> {
    let vb = vae_varbuilder(&root.as_ref().join("vae"), device)?;
    Ok(QwenVaeEncoder::new(vb)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vae_constants_match_qwen_image() {
        assert_eq!(VAE_COMPRESSION, 8);
        assert_eq!(VAE_CHANNELS, 16);
        assert_eq!(VAE_COMPRESSION * 2, 16);
    }

    #[test]
    fn load_vae_errors_cleanly_on_missing_dir() {
        match load_vae("/nonexistent-krea-snapshot", &Device::Cpu) {
            Err(e) => assert!(!e.to_string().is_empty()),
            Ok(_) => panic!("expected a load error for a missing snapshot dir"),
        }
    }
}
