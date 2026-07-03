//! Lens VAE decode (sc-5113) — a thin shim over the **already-ported** Flux.2 `AutoencoderKLFlux2`
//! ([`candle_gen_flux2::vae::Flux2Vae`]). The Lens latent space *is* the Flux.2 one (32-ch latent,
//! 2×2 patchify into the 128-ch transformer space, BatchNorm-stats normalization), so the whole
//! `LensPipeline._decode` reduces to: reshape the DiT output into the packed NCHW grid and call the
//! shared decode.
//!
//! ## Why the reshape is the whole shim
//! The reference `_decode` does `rearrange(b (h w) (c p1 p2) -> b c (h p1) (w p2))` then
//! `_patchify_latents` (re-pack 2×2) → bn de-normalize → `_unpatchify_latents` → `vae.decode`. The
//! rearrange-then-patchify pair is an **identity** that collapses to a plain reshape from
//! `[B, h·w, 128]` to the packed grid `[B, 128, h, w]` (the DiT's 128 channels already carry the
//! `c·4 + p1·2 + p2` packing, exactly [`Flux2Vae::decode_packed`]'s expected channel order). The
//! bn de-normalize (`x·std + mean`, `std = √(running_var + 1e-4)`), the 2×2 unpatchify, and the
//! AutoencoderKL decode are then the shared Flux.2 path verbatim — only the checkpoint differs
//! (the Lens `vae/` snapshot, loaded into the same `Flux2Vae`).

use candle_gen::candle_core::{DType, Result, Tensor};

pub use candle_gen_flux2::vae::Flux2Vae;

/// Decode the Lens DiT output into an image. `dit_out`: `[B, h·w, 128]` (the final denoised
/// patch-space latents after the last sampling step, NOT a per-step velocity); `(latent_h, latent_w)`
/// is the packed latent grid
/// (`= height/16, width/16`). Returns `[B, 3, H, W]` (NCHW) in ~`[−1, 1]`, where `H = latent_h·16`,
/// `W = latent_w·16` (2× unpatchify × 8× VAE upsample).
pub fn decode(
    vae: &Flux2Vae,
    dit_out: &Tensor,
    latent_h: usize,
    latent_w: usize,
) -> Result<Tensor> {
    let (b, _, c) = dit_out.dims3()?; // [B, h·w, 128]
    let packed = dit_out
        .reshape((b, latent_h, latent_w, c))?
        .permute((0, 3, 1, 2))? // [B, h, w, 128] → [B, 128, h, w] (NCHW)
        .contiguous()?;
    vae.decode_packed(&packed)
}

/// Convert a decoded image `[B, 3, H, W]` in `[−1, 1]` to `u8` `[0, 255]` (`(x.clamp(−1,1)+1)·127.5`),
/// matching the reference `_to_pil` quantization.
pub fn to_uint8(image: &Tensor) -> Result<Tensor> {
    let x = image.to_dtype(DType::F32)?.clamp(-1f32, 1f32)?;
    ((x + 1.0)? * 127.5)?.to_dtype(DType::U8)
}

/// Encode an RGB image `[B, 3, H, W]` (NCHW, ~`[−1, 1]`) into the packed Lens DiT latent
/// `[B, h·w, 128]` — the **inverse** of [`decode`], for the native training-latent path (sc-5147). The
/// shared [`Flux2Vae::encode_packed`] does the neural encode → 2×2 patchify → bn-normalize into the
/// packed grid `[B, 128, h, w]`; this shim only flattens that grid to the DiT's `[B, h·w, 128]`
/// sequence, exactly mirroring how [`decode`] reshapes the DiT output *back* to the packed grid (the
/// rearrange/patchify pair collapses to a transpose, so no explicit re-pack is needed). Returns
/// `(x0, latent_h, latent_w)` with `latent_h = H/16, latent_w = W/16` — the grid the DiT forward
/// consumes as `(frame = 1, h, w)`. Requires a [`Flux2Vae`] built with [`Flux2Vae::new_with_encoder`].
///
/// This is the deterministic posterior-**mean** latent (`encode_packed` uses the mean). The torch
/// `lens_train_runner._encode_latents` samples the posterior; the mean is the reproducible choice for
/// cached training latents (the flow-match target dominates the small posterior variance) and is what
/// the parity gate pins.
pub fn encode(vae: &Flux2Vae, image: &Tensor) -> Result<(Tensor, usize, usize)> {
    let packed = vae.encode_packed(image)?; // [B, 128, h, w]
    let (b, c, h, w) = packed.dims4()?;
    let x0 = packed
        .permute((0, 2, 3, 1))? // [B, h, w, 128]
        .reshape((b, h * w, c))? // [B, h·w, 128]
        .contiguous()?;
    Ok((x0, h, w))
}
