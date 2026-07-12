//! # candle-gen-pid — NVIDIA PiD (Pixel Diffusion Decoder)
//!
//! An optional, super-resolving replacement for an engine's VAE decode step (epic 7840, candle dup
//! sc-7853). PiD denoises directly in high-resolution pixel space, **decoding and upsampling in one
//! 4-step pass**. It is tied to a *latent space*, not a model: one `PixDiT_T2I` student topology
//! serves the whole image catalog, parameterized per latent space by a checkpoint + channel count +
//! latent norm (see [`registry`]).
//!
//! This crate implements the core [`candle_gen::LatentDecoder`] trait (the seam from sc-7844's candle
//! mirror), so a PiD-eligible engine can swap `vae.decode(latent)` for `pid.decode(latent)` at its
//! decode call site when the per-generation toggle is set — without N bespoke per-engine ports.
//! [`PidEngine`] is the load-once / decode-many entry a provider holds; [`PidEngine::decoder`] mints a
//! per-generation [`PidDecoder`] bound to that generation's caption + degrade σ + seed.
//!
//! This is the Windows/CUDA + Linux sibling of `mlx-gen-pid` — a faithful re-expression of the same
//! math in candle idioms (`candle_nn::Linear`/`ops::rms_norm`, `candle_gen::sdpa_budgeted_bhsd`,
//! native-NCHW conv2d, `candle_gen::seed` noise). The registry/config numerics are byte-identical to
//! the MLX port; the compute is validated to the same matmul floor.
//!
//! ## License
//! PiD weights are NVIDIA NSCLv1 (non-commercial). The NC restriction flows to PiD-decoded output —
//! it must be surfaced/labeled as research/evaluation-only at the worker/web layer (Phase 3).

pub mod backbone;
pub mod budget;
pub mod caption;
pub mod config;
pub mod decoder;
pub mod engine;
pub mod gemma2;
pub mod lq;
pub(crate) mod nn;
pub mod registry;
pub mod sampler;
pub mod tiling;

pub use backbone::PixDiT;
pub use caption::CaptionEncoder;
pub use config::{CaptionConfig, PidConfig, RopeMode, SampleType, SamplerConfig};
pub use decoder::PidDecoder;
pub use engine::{
    flow_capture_for_request, resolve_pid_decoder, resolve_pid_decoder_at_sigma,
    resolve_pid_decoder_for_fields, PidEngine,
};
pub use gemma2::{Gemma2, Gemma2Config};
pub use lq::{LqAdapter, PidNet};
pub use registry::{lookup, BackboneSpec, CkptType, LatentNorm};
pub use sampler::Sampler;
