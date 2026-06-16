//! # candle-gen-pulid
//!
//! PuLID-FLUX face-identity provider for [`candle-gen`](candle_gen) — the candle (Windows/CUDA) sibling
//! of `mlx-gen-pulid` (epic 5480, sc-5492). It ports the PuLID-FLUX stack on top of the candle FLUX.1
//! backbone, retiring the torch PuLID adapter off-Mac:
//!
//!   * [`eva_clip`] — the EVA02-CLIP-L-14-336 visual tower producing `id_cond_vit` (768-d) + 5 hidden
//!     states from the background-whitened aligned face crop ([`candle_gen_face`]'s `face_features_image`).
//!   * [`idformer`] — the perceiver-resampler fusing the ArcFace embedding + the EVA features into the
//!     32-token `id_embedding`.
//!   * [`ca`] — the 20 `PerceiverAttentionCA` modules injected into the FLUX DiT image stream via
//!     `candle-gen-flux`'s post-block [`DitImageInjector`](candle_gen_flux::DitImageInjector) seam.
//!   * [`pulid_flux`] — the end-to-end [`PulidFlux`](pulid_flux::PulidFlux) provider composing the
//!     above with the candle FLUX.1-dev backbone + the `gen-core` `FaceEmbedder`.
//!
//! Like the other candle identity providers (InstantID, the IP-Adapters), [`PulidFlux`] is a plain
//! struct driven **directly** by the worker (a bespoke reference stream), NOT a gen-core-registered
//! [`Generator`](candle_gen::gen_core::Generator) — the registered `flux1_*` descriptors stay
//! txt2img-only so the worker keeps the rest on the appropriate path.

pub mod ca;
pub mod eva_clip;
pub mod idformer;
pub mod pulid_flux;

pub use pulid_flux::{PulidFlux, PulidFluxPaths, PulidFluxRequest, DEFAULT_ID_WEIGHT};

/// PuLID-FLUX real-weight GPU validation (sc-5492) — env-driven, `#[ignore]`d integration test (the
/// analog of the InstantID / IP-Adapter Phase-5 harnesses).
#[cfg(test)]
mod validate;

/// Force-link hook so a consumer reaching this provider only indirectly keeps the rlib (and any future
/// registrations) linked — the same pattern as `candle_gen_flux::force_link`.
pub fn force_link() {}
