//! The latent→pixel decode seam (epic 7840, sc-7844 / candle dup sc-7853).
//!
//! Every candle image engine ends sampling with `vae.decode(latent)` called inline. To let a single
//! generation optionally route that final step through NVIDIA **PiD** — a pixel-diffusion decoder
//! that decodes *and* super-resolves in one pass — instead of the native VAE, without N bespoke
//! per-engine swaps, the decode step is expressed against this one trait. `candle-gen-pid` implements
//! it for PiD (the `PidDecoder`); a provider passes `Some(&pid)` at its decode call site when the
//! per-generation `req.use_pid` toggle is set, and the native VAE decode otherwise (the byte-exact
//! default). This is the candle twin of `mlx_gen::decoder::LatentDecoder` — a core-crate trait, NOT a
//! gen-core contract type (the gen-core seam is the `LoadSpec::pid` / `GenerationRequest::use_pid`
//! fields), so it lives here rather than requiring a gen-core pin bump.

use candle_core::Tensor;

use crate::Result;

/// Decodes a model's final **unpacked** latent into a decoded image tensor — the input a provider's
/// tensor→[`gen_core::Image`](gen_core::Image) step then turns into an image.
///
/// Contract:
/// - The input is the engine's unpacked latent in its latent space's native layout (e.g. Qwen/FLUX
///   16-ch `[1, C, H/8, W/8]`, SDXL 4-ch) — the same **normalized** tensor the native VAE decode
///   receives. Each implementor is tied to one latent space.
/// - The output is an `f32` tensor ready for the provider's image conversion.
/// - The output's spatial size **may exceed** the VAE-native size: PiD decodes and super-resolves in
///   a single pass. Callers must read dimensions from the returned tensor, never assume
///   `latent · spatial_scale`.
pub trait LatentDecoder {
    /// Decode `latents` to a decoded image tensor.
    fn decode(&self, latents: &Tensor) -> Result<Tensor>;
}
