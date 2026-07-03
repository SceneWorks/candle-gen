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
        }
    }

    /// Bind a cooperative cancellation handle (F-006). Callers that go through
    /// [`crate::resolve_pid_decoder`] get this wired from `req.cancel` automatically.
    pub fn with_cancel(mut self, cancel: CancelFlag) -> Self {
        self.cancel = Some(cancel);
        self
    }

    /// The output pixel resolution `(H, W)` for a latent grid `[.., .., zH, zW]`.
    pub fn target_hw(&self, latents: &Tensor) -> Result<(usize, usize)> {
        let (_, _, zh, zw) = latents.dims4()?;
        let f = (self.vae_compression * self.scale) as usize;
        Ok((zh * f, zw * f))
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
