//! The PiD [`LatentDecoder`] — the per-generation decoder that swaps an engine's `vae.decode(latent)`
//! for a super-resolving PiD pixel-diffusion decode (the sc-7844 seam, candle mirror). It carries this
//! generation's caption embeddings (+ degrade σ + SR scale), so `decode(latents)` stays the unchanged
//! trait method (the engine already holds the conditioning). Faithful to `from_clean.py`: PiD consumes
//! the **normalized** VAE latent directly; the output resolution is `latent_grid · vae_compression · scale`.

use candle_gen::candle_core::{DType, Tensor};
use candle_gen::gen_core::runtime::CancelFlag;
use candle_gen::{LatentDecoder, Result};

use crate::lq::PidNet;
use crate::sampler::Sampler;

/// A PiD decoder bound to one generation's caption + σ + scale.
pub struct PidDecoder {
    net: PidNet,
    sampler: Sampler,
    /// `[1, L, txt_embed_dim]` caption embeddings for this generation (from [`crate::caption`]), f32.
    caption_embs: Tensor,
    /// Degrade σ fed to the LQ gate (0 for a clean-latent decode).
    sigma: f32,
    /// Spatial SR factor (4× for the released students).
    scale: i32,
    /// VAE spatial compression (latent grid → pixel grid; 8 for the catalog VAEs).
    vae_compression: i32,
    /// Per-decode RNG seed for the sampler's noise + per-step ε.
    seed: u64,
    /// Cooperative cancellation for the multi-second 4-step decode (F-006). Bound at decoder-mint time
    /// from `req.cancel`. `None` ⇒ uncancellable (direct struct-API construction).
    cancel: Option<CancelFlag>,
    /// Spatial-tiling policy `(tile_edge_px, overlap_px)` for large outputs (sc-10087). Bound at mint
    /// time from the budget plan ([`crate::budget::plan_tile_edge`]); `decode` tiles when the output
    /// exceeds `tile_edge` on either axis (which on CUDA is what avoids the sysmem-fallback silent hang),
    /// else takes the exact whole-image path. `None` ⇒ always whole-image.
    tile: Option<(i32, i32)>,
}

impl PidDecoder {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        net: PidNet,
        sampler: Sampler,
        caption_embs: Tensor,
        sigma: f32,
        scale: i32,
        vae_compression: i32,
        seed: u64,
    ) -> Self {
        Self {
            net,
            sampler,
            caption_embs,
            sigma,
            scale,
            vae_compression,
            seed,
            cancel: None,
            tile: None,
        }
    }

    /// Bind a cooperative cancellation handle (F-006). Callers that go through
    /// [`crate::resolve_pid_decoder`] get this wired from `req.cancel` automatically.
    pub fn with_cancel(mut self, cancel: CancelFlag) -> Self {
        self.cancel = Some(cancel);
        self
    }

    /// Enable spatial tiling for large outputs (sc-10087): [`LatentDecoder::decode`] tiles the per-step
    /// velocity forward with `tile_edge`-px tiles (feather `overlap`) whenever the output exceeds
    /// `tile_edge` on either axis, else takes the exact whole-image path. Callers through
    /// [`crate::resolve_pid_decoder`] get this wired from [`crate::budget::plan_tile_edge`] automatically.
    pub fn with_tiling(mut self, tile_edge: i32, overlap: i32) -> Self {
        self.tile = Some((tile_edge, overlap));
        self
    }

    /// The output pixel resolution `(H, W)` for a latent grid `[.., .., zH, zW]`.
    pub fn target_hw(&self, latents: &Tensor) -> Result<(usize, usize)> {
        let (_, _, zh, zw) = latents.dims4()?;
        let f = (self.vae_compression * self.scale) as usize;
        Ok((zh * f, zw * f))
    }

    /// Spatially-tiled decode (sc-10087): same result geometry + seeded noise as [`LatentDecoder::decode`],
    /// but the per-step velocity forward runs on overlapping `tile`-px pixel windows (feather-blended), so
    /// the whole-image `PidNet::forward` VRAM peak never materializes. `tile`/`overlap` are output-pixel
    /// units. Used above the budget threshold at the decode seam, and by the A/B harness.
    pub fn decode_tiled(&self, latents: &Tensor, tile: i32, overlap: i32) -> Result<Tensor> {
        let b = latents.dim(0)?;
        let (th, tw) = self.target_hw(latents)?;
        let latents = latents.to_dtype(DType::F32)?;
        let sigma = Tensor::from_vec(vec![self.sigma; b], (b,), latents.device())?;
        self.sampler.sample_tiled(
            &self.net,
            &self.caption_embs,
            &latents,
            &sigma,
            b,
            th,
            tw,
            self.seed,
            tile,
            overlap,
            self.cancel.as_ref(),
        )
    }
}

impl LatentDecoder for PidDecoder {
    /// `latents`: the normalized VAE latent `[B, C, zH, zW]`. Returns super-resolved pixels
    /// `[B, 3, zH·vae_compression·scale, zW·vae_compression·scale]` in `[-1, 1]` (f32).
    fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        let b = latents.dim(0)?;
        let (th, tw) = self.target_hw(latents)?;
        // The PiD net runs f32; an engine may hand an f16/bf16 sampler latent, so cast here (a no-op
        // when already f32). Matches the validated `from_clean` handoff (the normalized VAE latent).
        let latents = latents.to_dtype(DType::F32)?;
        let sigma = Tensor::from_vec(vec![self.sigma; b], (b,), latents.device())?;
        // Tile when a policy is set and the output exceeds one tile on either axis (sc-10087); otherwise
        // the exact whole-image path — byte-identical to the pre-tiling decode for small outputs.
        if let Some((edge, overlap)) = self.tile {
            if th as i32 > edge || tw as i32 > edge {
                return self.sampler.sample_tiled(
                    &self.net,
                    &self.caption_embs,
                    &latents,
                    &sigma,
                    b,
                    th,
                    tw,
                    self.seed,
                    edge,
                    overlap,
                    self.cancel.as_ref(),
                );
            }
        }
        self.sampler.sample(
            &self.net,
            &self.caption_embs,
            &latents,
            &sigma,
            b,
            th,
            tw,
            self.seed,
            self.cancel.as_ref(),
        )
    }
}
