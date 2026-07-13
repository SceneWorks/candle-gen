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
//! [`CandleError`] ⇄ [`gen_core::Error`] bridges that let a provider crate's
//! `Generator::generate` (whose signature is `gen_core::Result`) keep using `?` on the candle
//! `Result`s that do the actual tensor work.
//!
//! Beyond that seam this crate is the **shared commons** for the candle provider crates — the single
//! audited home for machinery that would otherwise be hand-copied across ~25 providers and drift. The
//! provider crates (`candle-gen-sdxl`, `-z-image`, `-flux`, `-wan`, `-ltx`, `-lens`, `-krea`, …) build
//! their `gen_core::Generator` / `Trainer` on top of these modules:
//!
//! - [`loader`] — sorted-`.safetensors` → unsafe-mmap [`VarBuilder`](candle_nn::VarBuilder) loading,
//!   the single concentrated `unsafe` mmap surface (sc-8999).
//! - [`weights`] — the non-`VarBuilder` safetensors key→[`Tensor`](candle_core::Tensor) weight map
//!   (dtype-coerce, duplicate-key policy, prefix-filtered reads) shared by the IP-Adapter / ControlNet /
//!   PuLID loads (sc-9044).
//! - [`seed`] — seed derivation + launch-portable CPU-first seeded-noise helpers, so every provider
//!   draws reproducible noise identically across providers and backends (sc-7792).
//! - [`sampler`] — the gen-core `LatentOps` impl over [`Tensor`](candle_core::Tensor) plus the unified
//!   curated / flow / two-stream (A/V) sampler drivers and schedule resolvers (epic 7114).
//! - [`train`] — the shared native training harness (LoRA/LoKr adapters, optimizers, LR schedules,
//!   dataset bucketing, manual gradient checkpointing, and the inference-side [`train::merge`] adapter
//!   reconstruction), epic 5164.
//! - [`quant`] — the MLX-packed → GGML repack seam that lets the candle lane consume the hosted MLX
//!   quant tiers directly (epic 9083).
//! - [`vae_tiling`] — the budgeted video-VAE tile/blend/accumulate driver shared by the wan/ltx
//!   decoders (sc-9006).
//! - [`gpu`] — the trusted-path `nvidia-smi` VRAM-budget probe (sc-9014).
//! - [`sync`] — poison-tolerant [`lock_recover`] for the shared generator/component caches (sc-9015).

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

// Shared sorted-`.safetensors` → unsafe-mmap loader (sc-8999 / F-019): the single audited home for
// the `list a snapshot component dir, sort deterministically, error-if-empty, unsafe-mmap into a
// VarBuilder` idiom that was hand-copied ~34 times across the provider crates. Concentrates the
// `unsafe` mmap surface (also aids F-062) and the SAFETY invariant in one place.
pub mod loader;
pub use loader::{
    component_vb, load_one_tensor, load_one_tensor_sharded, load_path_mmap, load_sorted_mmap,
    mmap_var_builder, resolve_weight_files, sorted_safetensors,
};

// Shared i32-overflow-safe scaled-dot-product attention (sc-9116 / epic 8979): the F-003 query-row
// chunking guard — hoisted from the per-crate flux2/chroma/lens/qwen-image copies — so the remaining
// audited DiT/VAE attention sites (sdxl/z-image/sd3/svd/scail2/ideogram/krea/lens-train + the
// chroma/flux2/qwen-image VAE mid-blocks) share ONE guarded copy. candle CUDA kernels index elements
// with i32; a scores tensor over `i32::MAX` silently corrupts its tail at large render sizes.
pub mod attention;
pub use attention::{sdpa_budgeted_bhsd, sdpa_budgeted_flat, ATTN_SCORES_BUDGET};

// Shared Qwen3-VL text-encoder grounding helpers (sc-11205 / F-118): the MRoPE / vision-splice
// machinery (`Rotary` 1-D RoPE table, GQA `repeat_kv`, `<|image_pad|>` `image_blocks`, the vision-embed
// `replace_seq`/`slice_seq`, the 3-D interleaved `mrope_positions` + `mrope_cos_sin`, and the additive
// `causal_mask`) that the Boogu (Qwen3-VL-8B) and Krea (Qwen3-VL-4B) condition encoders both need on
// their image-grounded edit paths. Was byte-identical between `candle-gen-boogu` and `candle-gen-krea`
// (~250 lines of parity-critical grounding, fixed twice); hoisted here so both draw from one copy.
pub mod grounding;

// The latent→pixel decode seam (epic 7840, sc-7853): the `LatentDecoder` trait a provider routes its
// final `vae.decode(latent)` through so a per-generation `req.use_pid` toggle can swap in NVIDIA PiD
// (`candle-gen-pid`) without N bespoke per-engine ports. The candle twin of `mlx_gen::decoder`.
pub mod decoder;
pub use decoder::LatentDecoder;

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
    resolve_schedule, run_av_curated_sampler, run_curated_sampler, run_flow_sampler,
    run_scm_sampler, AvLatents, CandleAvLatentOps, CandleLatentOps, ScmScheduler,
    SCM_DEFAULT_INTERMEDIATE_TIMESTEP, SCM_DEFAULT_MAX_TIMESTEP, SCM_SIGMA_DATA,
};

// Shared seed-derivation + launch-portable seeded-noise helpers (sc-7792 consolidation / F-059,
// sc-9043): the per-image batch seed (`base + index`), the ancestral-step RNG salt, and the CPU-first
// noise draw. Previously hand-copied into every provider crate — a single divergent copy silently
// breaks cross-provider / cross-backend reproducibility, so they get one home here.
pub mod seed;
pub use seed::{
    for_each_image_seed, image_seed, seeded_noise_nchw, seeded_normal_vec, STEP_RNG_SALT,
};

// Shared budgeted video-VAE tiling machinery (sc-9006 / F-026): the tile/narrow/blend/pad-accumulate/
// normalize DRIVER + the `<PREFIX>_VAE_BUDGET_GIB` budget resolver + the budgeted-plan selector that
// were copied byte-near-identically between candle-gen-wan (z48 vae22) and candle-gen-ltx. The pure
// tile GEOMETRY stays in `gen_core::tiling`; this module owns the candle-side execution of a plan,
// parameterized by each VAE's cost model + decode closure so the per-VAE numerics are unchanged.
pub mod vae_tiling;

// Shared safetensors key→`Tensor` weight map (sc-9044 / F-060): the non-`VarBuilder` loader (float
// dtype-coerce, hard duplicate-key policy, prefix-filtered header-only reads) that the SDXL IP-Adapter/
// ControlNet loads AND the FLUX-family IP-Adapter / PuLID EVA-CLIP towers all share. It had drifted into
// `candle-gen-sdxl`, making that pipeline crate a de-facto commons crate that PuLID/FLUX pulled the whole
// ~12k-LOC SDXL crate in for. Hoisted here; `candle-gen-sdxl::weights` re-exports it for compatibility.
pub mod weights;
pub use weights::Weights;

// Shared native WAN-VAE (`Wan2.1` z16 3D-causal-conv autoencoder) → diffusers key remap (epic 10451):
// both the Qwen-Image (sc-10830) and Wan2.2 (sc-10909) in-place ComfyUI lanes read the *same physical*
// Wan2.1 16-channel VAE, stored with native WAN-VAE keys. The rename lives here (the `weights` module's
// F-060 posture) rather than duplicated in each pipeline crate's `comfyui` seam.
pub mod comfyui_vae;
pub use comfyui_vae::remap_vae_wan_to_diffusers;

// Poison-tolerant locking + read-through helper for the shared generator/component caches (sc-9015 /
// F-031; `cached` sc-7792): a panic while holding a cache `Mutex` (e.g. a CUDA OOM lifted to a panic
// mid-decode) poisons it, after which a plain `.lock().unwrap()` panics forever — one transient
// failure wedges a long-lived worker lane into a permanent panic loop. `lock_recover` treats a
// poisoned overwrite-on-miss cache as usable; `cached` is the lazy `components()` read-through built
// on it, collapsing the byte-identical lock-check-load-store scaffold across the provider crates.
pub mod sync;
pub use sync::{cached, lock_recover};

// Shared test-support helpers (sc-9055 / F-069): the PPM read/write, cosine, env-path, GPU peak-VRAM,
// and HF-Hub-cache resolution helpers that had been hand-copied — and had drifted — across ~16
// `#[cfg(test)]` validation modules in the provider crates. Also folds the F-071/sc-9057 `$HF_HOME`
// cache-resolution harmonization. Gated behind the `testkit` feature so this test-only surface never
// compiles into a production build; provider crates enable it as a dev-dependency feature.
#[cfg(feature = "testkit")]
pub mod testkit;

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
        // linked and a Device is constructible on whatever backend the build selected.
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
