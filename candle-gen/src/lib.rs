//! # candle-gen
//!
//! The **candle** tensor-backend core for SceneWorks generative inference — the Windows/CUDA
//! sibling of [`mlx-gen`](https://github.com/michaeltrefry/mlx-gen) (Apple MLX). Both crates
//! implement the **same** backend-neutral [`gen_core`] contract (epic 3720): the `Generator` /
//! `Trainer` / `Captioner` / `Transform` traits, the request/output types, and the link-time
//! model registry. A consumer pins one backend by SHA and links its provider crates; the provider
//! crates self-register via `inventory`, so adding a model is purely additive (no central match).
//!
//! This crate owns the candle-specific seam: device/dtype selection across the CPU (default),
//! Metal (`metal` feature, Mac), and CUDA (`cuda` feature, Windows) backends, plus the
//! [`CandleError`] → [`gen_core::Error`] bridge that lets a provider crate's
//! `Generator::generate` (whose signature is `gen_core::Result`) keep using `?` on the candle
//! `Result`s that do the actual tensor work.
//!
//! **Phase 1 (sc-4946) is a scaffold:** the contract + capability surface + device plumbing are
//! wired and proven to compile/register, but the real SDXL pipeline lands in a later slice.

// Re-export the backend-neutral contract so downstream provider crates resolve `gen_core::…`
// through `candle_gen::gen_core` (single gen-core resolution — see the skew gate). Mirrors how
// mlx-gen re-exports gen_core for mlx-gen-sdxl.
pub use gen_core;
// Re-export the link-time registration macros so provider crates call `candle_gen::register_*!`
// (the candle twin of `mlx_gen::register_generators!`). They emit `gen_core`'s `inventory::submit!`
// + the `Into::into` error bridge, so providers drop their hand-written `_registered` adapters.
pub use gen_core::{
    register_captioner, register_generators, register_image_embedder, register_text_embedder,
    register_trainer,
};
// Re-export the candle backend so provider crates share this crate's exact candle build.
pub use candle_core;
pub use candle_nn;

// Shared VRAM-budget probe (sc-9014 / F-030): the trusted-path `nvidia-smi` resolver the video-VAE
// decode tilers (seedvr2/wan/ltx) route through, instead of each spawning a bare
// `Command::new("nvidia-smi")` that Windows resolves via the process search order (a PATH-hijack
// vector). Resolves an absolute path from System32 / CUDA_PATH once and caches it.
pub mod gpu;

// The MLX-packed → GGML repack seam (sc-9085 spike → sc-9086, epic 9083): lets the candle lane
// load the hosted MLX quant tiers (epic 8506) directly — no dense staging, no second artifact
// matrix. Provider crates' packed-detect loaders build on this.
pub mod quant;

// The shared native training harness (epic 5164 / sc-5165) — the candle twin of `mlx_gen::train`.
// Provider crates (sdxl/z-image/wan/lens) build their `gen_core::Trainer` on top of this.
pub mod train;

// The unified sampler/scheduler framework backend (epic 7114 P2, sc-7119): the gen-core `LatentOps`
// impl over `candle_core::Tensor`. The candle twin of `mlx_gen::MlxLatentOps`; lets every candle
// provider crate drive the shared gen-core callback samplers.
pub mod sampler;
pub use sampler::{
    curated_sampler_names, curated_scheduler_names, menu_with_aliases, resolve_flow_schedule,
    resolve_schedule, run_av_curated_sampler, run_curated_sampler, run_flow_sampler, AvLatents,
    CandleAvLatentOps, CandleLatentOps,
};

use thiserror::Error;

/// The candle-backed crate error. gen-core cannot name candle types, so device/tensor failures
/// arrive boxed in [`gen_core::Error::Backend`] via the [`From`] bridge below. This mirrors
/// mlx-gen's `From<mlx_gen::Error> for gen_core::Error` seam — legal under the orphan rule because
/// the source type ([`CandleError`]) is local to this crate.
#[derive(Debug, Error)]
pub enum CandleError {
    /// A candle op (matmul, conv, device alloc, …) failed.
    #[error("candle op failed: {0}")]
    Candle(#[from] candle_core::Error),

    /// A contextual message (config/validation/shape errors).
    #[error("{0}")]
    Msg(String),

    /// Cooperative cancellation tripped mid-generation (the request's `CancelFlag`). Kept a typed
    /// variant — NOT a `Msg` — so a provider's rich-`Result` body can `return Err(CandleError::Canceled)`
    /// between denoise steps and the [`From`] bridge lifts it to the contract-load-bearing
    /// [`gen_core::Error::Canceled`] (the worker + gen-core-testkit conformance suite key off the typed
    /// variant, sc-4481). Mirrors mlx-gen's `Error::Canceled`.
    #[error("cancelled")]
    Canceled,
}

impl From<CandleError> for gen_core::Error {
    fn from(e: CandleError) -> Self {
        match e {
            // candle's Error is `Send + Sync + 'static`, so it boxes straight into Backend.
            CandleError::Candle(c) => gen_core::Error::backend(c),
            CandleError::Msg(s) => gen_core::Error::Msg(s),
            // Preserve the typed cancellation signal across the bridge (do NOT stringify to Msg).
            CandleError::Canceled => gen_core::Error::Canceled,
        }
    }
}

/// Reverse bridge: lift a backend-neutral [`gen_core::Error`] back into [`CandleError`]. The unified
/// curated-sampler driver ([`sampler::run_curated_sampler`]) runs over the gen-core `Sampler` trait
/// (which returns `gen_core::Result`), so a provider's rich-`Result` denoise body needs this to `?` the
/// driver's output. The load-bearing arm is `Canceled -> Canceled`: a cooperative cancel tripped inside
/// the driver's `denoise` callback surfaces as `gen_core::Error::Canceled` and MUST stay the typed
/// [`CandleError::Canceled`] (not a stringified `Msg`) so the worker + conformance suite key off it.
/// Mirrors mlx-gen's `From<gen_core::Error> for mlx_gen::Error`.
impl From<gen_core::Error> for CandleError {
    fn from(e: gen_core::Error) -> Self {
        match e {
            gen_core::Error::Canceled => CandleError::Canceled,
            gen_core::Error::MissingTensor(s) => CandleError::Msg(format!("missing tensor: {s}")),
            gen_core::Error::Unsupported(s) => CandleError::Msg(format!("unsupported: {s}")),
            gen_core::Error::Io(io) => CandleError::Msg(io.to_string()),
            gen_core::Error::Backend(b) => CandleError::Msg(b.to_string()),
            gen_core::Error::Msg(s) => CandleError::Msg(s),
        }
    }
}

impl From<String> for CandleError {
    fn from(s: String) -> Self {
        CandleError::Msg(s)
    }
}

impl From<&str> for CandleError {
    fn from(s: &str) -> Self {
        CandleError::Msg(s.to_string())
    }
}

/// Crate-wide result over [`CandleError`] (the rich candle-side `Result`; provider `Generator`
/// bodies bridge the tail into `gen_core::Result` via `?` + the [`From`] above).
pub type Result<T> = std::result::Result<T, CandleError>;

/// The process-default compute device, selected at compile time by feature:
/// CUDA (`cuda`) → Metal (`metal`) → CPU (default). Exercising this proves candle links and a
/// `Device` constructs on whatever backend the build selected (CPU/Metal on Mac).
pub fn default_device() -> Result<candle_core::Device> {
    #[cfg(feature = "cuda")]
    let dev = candle_core::Device::new_cuda(0)?;
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    let dev = candle_core::Device::new_metal(0)?;
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    let dev = candle_core::Device::Cpu;
    Ok(dev)
}

/// The default dense compute dtype for the selected backend: `F16` on the GPU backends
/// (Metal/CUDA — the SDXL family is fp16), `F32` on CPU (Mac default; half-precision CPU kernels
/// are slow/unsupported). Providers override per-component as needed (e.g. an fp32 VAE).
pub fn default_dtype() -> candle_core::DType {
    #[cfg(any(feature = "cuda", feature = "metal"))]
    {
        candle_core::DType::F16
    }
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    {
        candle_core::DType::F32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_device_constructs() {
        // CPU on the default Mac build; Metal/CUDA when those features are on. Proves candle is
        // linked and a Device is constructible — the scaffold's "candle actually builds" check.
        let dev = default_device().expect("default device constructs");
        // A trivial tensor op on the device proves the backend is live, not just named.
        let t = candle_core::Tensor::zeros((2, 2), default_dtype(), &dev).expect("alloc");
        assert_eq!(t.dims(), &[2, 2]);
    }

    #[test]
    fn candle_error_bridges_to_backend() {
        // A candle error must box into gen_core::Error::Backend (the parity-critical seam).
        let bad =
            candle_core::Tensor::zeros((2, 3), candle_core::DType::F32, &candle_core::Device::Cpu)
                .unwrap()
                .matmul(
                    &candle_core::Tensor::zeros(
                        (4, 5),
                        candle_core::DType::F32,
                        &candle_core::Device::Cpu,
                    )
                    .unwrap(),
                );
        let candle_err = CandleError::from(bad.unwrap_err());
        let neutral: gen_core::Error = candle_err.into();
        assert!(matches!(neutral, gen_core::Error::Backend(_)));
    }
}
