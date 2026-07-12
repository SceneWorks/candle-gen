//! The candle SDXL **txt2img** pipeline (sc-3675) — the proven epic-3494 prototype
//! (`D:\sceneworks-candle-spike\src\bin\candle_sdxl.rs`) lifted out of its standalone CLI/PNG shell
//! and into the backend-neutral [`gen_core::Generator`] contract.
//!
//! What changed vs the spike, and what deliberately did **not**:
//! - **Components** (the GO-validated path): dual CLIP (CLIP-L + CLIP-bigG) loaded **f16** (sc-3674;
//!   the spike used f32) and encoded; UNet **f16**; VAE **f16** with the `madebyollin/sdxl-vae-fp16-fix`
//!   (f16 SDXL VAE NaNs without it); VAE scale **0.13025** (the diffusers SDXL value, not candle's
//!   hardcoded SD1.5 0.18215).
//! - **Perf (sc-3674)**: the UNet attention runs through fused **flash-attention** when the crate is
//!   built `--features flash-attn` AND the runtime toggle ([`crate::set_flash_attn`], default on) is
//!   set — on Blackwell sm_120 that cut steady-state from ~0.32 to ~0.21 s/step and peak VRAM ~21.6→18
//!   GiB. The build feature is the opt-in; the toggle is what the SceneWorks UI exposes.
//! - **Peak VRAM (sc-4987)**: two structural levers on top of sc-3674's 18 GiB high-water mark, both
//!   targeting torch-parity (~9 GiB) at 1024². (1) **Staged sequential load** — each CLIP encoder is
//!   loaded, run, and **dropped** before the next, and *both* are gone before the UNet/VAE even load
//!   (text embeddings are seed-independent, computed once up front), so the dual CLIP (~1.6 GiB f16)
//!   never sits resident through denoise/decode. (2) **VAE tiling** — the VAE decode at 1024² is the
//!   tallest single allocation; [`tile_blend_decode`] splits the latent into overlapping 64² latent
//!   tiles (512² output), decodes each, and trapezoidally blends the seams (diffusers'
//!   `enable_vae_tiling`), bounding the decode peak to one tile. Gated by [`crate::vae_tiling_enabled`]
//!   (default on) and only *fires* above 512² output (the geometry policy lives in [`gen_core::tiling`]).
//! - **Deterministic seeding + non-ancestral scheduler (sc-3673)**: initial noise is drawn from a
//!   fixed-algorithm CPU RNG (`StdRng`) seeded by `seed` and moved to the device — NOT candle's CUDA
//!   `device.set_seed`, whose seed→noise mapping was not portable across launch environments and
//!   occasionally collapsed the sample (sc-3498). The sampler is **DDIM (eta=0)**, non-ancestral, so
//!   there is no per-step stochastic noise. Net: generation is a pure function of `(seed, request)`.
//! - **CLI/`emit_event`/PNG/sidecar removed**: progress is `on_progress(Progress::Step/Decoding)`,
//!   cancellation is `req.cancel` → typed [`gen_core::Error::Canceled`], and each image is returned as a
//!   `gen_core::Image` (RGB8) — the worker owns asset writes (no candle-specific worker code).
//! - **Weights come from `spec.weights` (the SDXL snapshot dir)**, not a hardcoded HF repo: UNet +
//!   both text encoders load from the snapshot's component subdirs. The two **model-agnostic** inputs
//!   — the fp16-VAE-fix and the CLIP-L/bigG `tokenizer.json`s — still resolve via `hf-hub` (cached),
//!   exactly as the spike.
//!
//! - **Component caching (sc-5037)**: the seed/prompt/resolution-independent [`Components`] (UNet +
//!   VAE) are loaded once and **cached on the generator** across `generate` calls (keyed by the
//!   flash-attn setting), so back-to-back requests skip the ~7 GiB UNet/VAE disk re-read. This is
//!   reconciled with the sc-4987 staged load rather than reverting it: CLIP stays
//!   load-on-demand-and-free (only one encoder resident at a time), and the generator computes the
//!   text embeddings *before* acquiring the cached UNet/VAE — so the cold-call ordering (CLIP freed
//!   before UNet/VAE load) and the ~8.7 GiB peak are preserved; the cache holds only UNet+VAE
//!   resident between calls (a latency win, not a peak-VRAM regression).
//!
//! - **RealVisXL (sc-3677)**: RealVisXL_V5.0 (`SG161222/RealVisXL_V5.0`) shares the SDXL architecture
//!   AND ships the standard diffusers multi-component tree with the *same* component filenames this
//!   pipeline already resolves — `unet/diffusion_pytorch_model.fp16.safetensors`,
//!   `text_encoder{,_2}/model.fp16.safetensors`. So it loads through this exact snapshot path
//!   unmodified; the single-file root checkpoints it also publishes are not needed and no single-file
//!   loader was added (the [`snapshot_file`] component layout is present, not absent). The model-
//!   agnostic VAE-fix + CLIP tokenizers and the production defaults below ([`DEFAULT_STEPS`],
//!   [`DEFAULT_GUIDANCE`], [`VAE_SCALE`]) are shared, matching the Python `SdxlDiffusersAdapter`; the
//!   one accepted sampler difference (DDIM eta=0 vs the adapter's euler_ancestral) is the sc-3673
//!   launch-portable-determinism choice. Parity is locked by `tests/conformance.rs::realvisxl_conformance`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use candle_gen::candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::sampling::{
    schedule_sigmas, AlphaSchedule, DiscreteModelSampling, LightningPolicy, SamplerPolicy,
    Scheduler,
};
use candle_gen::gen_core::tiling::{TilingConfig, VaeTiling};
use candle_gen::gen_core::{self, AdapterSpec, GenerationRequest, Image, PidWeights, Progress};
// Shared per-image batch seed (`base + index`) — one home in `candle-gen` (sc-9043 / F-059).
use candle_gen::{CandleError, LatentDecoder, Result};
use candle_gen_pid::{PidDecoder, PidEngine};

/// The PiD backbone (latent-space) tag for SDXL (epic 7840 / sc-7853): SDXL's own `sdxl` VP-frame
/// student (4× SR). Kolors reuses this crate's decode seam via the same `sdxl` tag (shared VAE).
/// Re-exported (sc-8373) so `candle-gen-instantid` loads the SAME `sdxl` student — InstantID composes
/// the SDXL VAE, so there is no InstantID-specific PiD checkpoint.
pub const PID_BACKBONE: &str = "sdxl";
use candle_transformers::models::stable_diffusion::unet_2d::{
    BlockConfig, UNet2DConditionModel, UNet2DConditionModelConfig,
};
use candle_transformers::models::stable_diffusion::vae::AutoEncoderKL;
use candle_transformers::models::stable_diffusion::{self, StableDiffusionConfig};

// The vendored, packed-detecting SDXL UNet (sc-5165 / sc-9416): its Linear surface routes through the
// shared `candle_gen::quant` seam, so it loads a pre-quantized MLX tier (SceneWorks/sdxl-base-mlx
// q4/q8) straight from the packed parts. Aliased to avoid clashing with the stock (dense) UNet type.
use crate::unet::{sdxl_unet_config, UNet2DConditionModel as VendoredUNet};
use candle_gen::quant::{PackedConfig, MLX_GROUP_SIZE};
use rand::{rngs::StdRng, SeedableRng};
use tokenizers::Tokenizer;

/// diffusers SDXL VAE `scaling_factor` (candle's example hardcodes the SD1.5 value 0.18215 for `Xl`;
/// 0.13025 is the diffusers-correct one and is what produced correctly-exposed output in the spike).
pub(crate) const VAE_SCALE: f64 = 0.13025;
/// Production SDXL defaults (the SceneWorks `sdxl` row): 30 steps, CFG 7.0 — used when the request
/// omits them.
const DEFAULT_STEPS: usize = 30;
const DEFAULT_GUIDANCE: f64 = 7.0;
/// The sampler an **omitted** `req.sampler` resolves to (sc-10826): the curated `ddim` solver — a
/// k-diffusion DDIM (eta=0, non-ancestral) over the SDXL ε-schedule, driven by the unified
/// [`gen_core::sampling`] framework via [`Pipeline::denoise_curated`]. It **replaces** the native
/// candle-transformers `DDIMScheduler` inference loop, which rendered a ghosted, translucent
/// double-exposure (guidance-invariant) on the default path while every curated solver — including
/// this curated `ddim` — is clean. Being eta=0 / non-ancestral it keeps the sc-3673 launch-portable
/// determinism the native default targeted (generation stays a pure function of `(seed, request)`),
/// and `ddim` is part of the advertised curated vocabulary ([`candle_gen::curated_sampler_names`]),
/// so it remains a valid selection.
const DEFAULT_SAMPLER: &str = "ddim";

/// The curated solver a txt2img render resolves `req.sampler` to (sc-10826), factored out of
/// [`Pipeline::render`] so the default-routing rule is unit-testable without a GPU:
/// - `Some(LIGHTNING_SAMPLER)` ⇒ `None` — the `lightning` render owns its own few-step path.
/// - `None` (omitted) ⇒ `Some(DEFAULT_SAMPLER)` — the curated `ddim` solver (the fix: the native
///   candle-transformers DDIM loop that ghosted is gone).
/// - any other `Some(name)` ⇒ `Some(name)` verbatim — the curated solver by that name (an unknown
///   name euler-falls-back inside `run_curated_sampler`, N3).
fn resolve_sampler(sampler: Option<&str>) -> Option<&str> {
    if sampler == Some(LIGHTNING_SAMPLER) {
        None
    } else {
        Some(sampler.unwrap_or(DEFAULT_SAMPLER))
    }
}

/// The few-step **Lightning** sampler id (sc-6128) — diffusers Euler-trailing, selected per request
/// via `req.sampler` and advertised in [`crate::descriptor`]'s `samplers`. The SceneWorks worker forces
/// it for the `realvisxl_lightning` model id; distilled Lightning checkpoints (RealVisXL Lightning /
/// SDXL-Lightning) render correctly in 2–8 steps through this schedule, where DDIM at the same step
/// count produces mush.
pub(crate) const LIGHTNING_SAMPLER: &str = "lightning";
/// Lightning's few-step default when the request omits `steps` — matches `mlx-gen-sdxl`'s
/// `accel_defaults("lightning")` (4 steps, CFG off). The worker typically sends an explicit count
/// (the AC eyeballs ~5).
const LIGHTNING_DEFAULT_STEPS: usize = 4;
/// SDXL's `scaled_linear` β endpoints + train-step count (the diffusers SDXL scheduler defaults — the
/// same values `DDIMSchedulerConfig::default()` and `sampler::EulerAncestralSampler` carry). The
/// Lightning policy's σ table is built from these.
const SDXL_BETA_START: f32 = 0.00085;
const SDXL_BETA_END: f32 = 0.012;
const SDXL_TRAIN_STEPS: usize = 1000;

/// Build SDXL's ε-prediction α-cumprod schedule (`scaled_linear` β over 1000 train steps) — the
/// [`DiscreteModelSampling`] source the curated unified-sampler path integrates over. Shared by the
/// txt2img [`Pipeline::denoise_curated`] (sc-7124), the Lightning policy, and the conditioned
/// [`crate::ip_provider`] curated denoise (sc-7297), so they speak one SDXL noise schedule.
pub(crate) fn sdxl_alpha_schedule() -> Result<AlphaSchedule> {
    Ok(AlphaSchedule::scaled_linear(
        SDXL_TRAIN_STEPS,
        SDXL_BETA_START,
        SDXL_BETA_END,
    ))
}

/// Build the SDXL-**Lightning** sampler *policy* (sc-6128) for `num_steps`: diffusers
/// `EulerDiscreteScheduler(timestep_spacing="trailing", final_sigmas_type="zero")`, ε-prediction. The
/// schedule math is the backend-neutral [`gen_core::sampling::LightningPolicy`] — the **same** policy
/// the `mlx-gen-sdxl` `LightningSampler` drives, so no candle gen-core pin bump is needed and the two
/// backends share the reference trailing-spacing + interpolated σ table. The candle side is only the
/// ~5-line tensor application in [`Pipeline::denoise_lightning`].
fn lightning_policy(num_steps: usize) -> Result<LightningPolicy> {
    let sched = sdxl_alpha_schedule()?;
    Ok(LightningPolicy::new(&sched, SDXL_TRAIN_STEPS, num_steps))
}

/// The fp16-stable SDXL VAE (the base VAE NaNs in f16). Model-agnostic across every SDXL checkpoint,
/// so it is fetched by repo id rather than read from the per-model snapshot.
pub(crate) const VAE_FIX_REPO: &str = "madebyollin/sdxl-vae-fp16-fix";
pub(crate) const VAE_FIX_FILE: &str = "diffusion_pytorch_model.safetensors";

/// Immutable commit SHAs pinning the three runtime `hf-hub` downloads on the SDXL render path
/// (sc-9013 / F-029). Every `hf_get` resolves against one of these — never the mutable `main`
/// default — so a compromised or force-pushed upstream cannot silently swap the fp16-fix VAE weights
/// or the CLIP tokenizations out from under a cold-cache generation. The trio is model-agnostic (they
/// do not vary per SDXL checkpoint), so a single pin per repo covers every request. Bump these
/// deliberately (and re-validate) if an upstream fix is ever needed. SHAs captured 2026-07-02:
/// - `madebyollin/sdxl-vae-fp16-fix`
/// - `openai/clip-vit-large-patch14`
/// - `laion/CLIP-ViT-bigG-14-laion2B-39B-b160k`
const HUB_PINS: &[(&str, &str)] = &[
    (VAE_FIX_REPO, "207b116dae70ace3637169f1ddd2434b91b3a8cd"),
    (
        "openai/clip-vit-large-patch14",
        "32bd64288804d66eefd0ccbe215aa642df71cc41",
    ),
    (
        "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
        "743c27bd53dfe508a0ade0f50698f99b39d03bec",
    ),
];

/// The pinned immutable revision for a runtime hub `repo`, or an error if the repo is not in
/// [`HUB_PINS`] — an unpinned runtime download on the render path is a supply-chain risk (F-029), so
/// `hf_get` refuses to resolve one rather than silently falling back to the mutable `main` revision.
fn hub_revision(repo: &str) -> Result<&'static str> {
    HUB_PINS
        .iter()
        .find(|(id, _)| *id == repo)
        .map(|(_, rev)| *rev)
        .ok_or_else(|| CandleError::Msg(format!("no pinned hub revision for repo {repo:?}")))
}

/// The SDXL VAE's tiling geometry (sc-4987): the decoder upsamples latents ×8 spatially, and an image
/// VAE has **no temporal axis** — so temporal scale 1, non-causal (the `[B, 4, h, w]` latent is tiled
/// on the two spatial axes only, with the singleton temporal axis a no-op in [`TilingConfig::plan`]).
const SDXL_VAE_TILING: VaeTiling = VaeTiling {
    spatial_scale: 8,
    temporal_scale: 1,
    causal_temporal: false,
};

/// The SDXL VAE tiling policy (sc-4987) — diffusers' `enable_vae_tiling` defaults: **512² output
/// tiles (64² latent) with 128 px overlap (16 latent, the 0.25 overlap-factor)**. `needs_tiling` then
/// fires only when an output axis exceeds 512 px, so 512² renders stay monolithic (latent 64 is not
/// `> 64`) and 1024² tiles into a 3×3 grid stepping 48 latent — bounding the decode peak to one 512²
/// tile while the 16-latent overlap + trapezoidal blend keeps seams invisible.
fn sdxl_tiling_config() -> TilingConfig {
    TilingConfig::spatial_only(512, 128)
}

/// Which of the two SDXL CLIP encoders — selects the tokenizer repo, the snapshot weights subpath,
/// and which `StableDiffusionConfig` clip config to use.
pub(crate) enum Clip {
    /// CLIP-L (`text_encoder/`) — `openai/clip-vit-large-patch14` tokenizer.
    L,
    /// OpenCLIP bigG (`text_encoder_2/`) — `laion/CLIP-ViT-bigG-14-laion2B-39B-b160k` tokenizer.
    BigG,
}

impl Clip {
    /// `(tokenizer repo, snapshot weights subpath)`.
    pub(crate) fn sources(&self) -> (&'static str, &'static str) {
        match self {
            Clip::L => (
                "openai/clip-vit-large-patch14",
                "text_encoder/model.fp16.safetensors",
            ),
            Clip::BigG => (
                "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
                "text_encoder_2/model.fp16.safetensors",
            ),
        }
    }

    /// The encoder's diffusers component subdir (`text_encoder` / `text_encoder_2`) — the base for its
    /// `config.json` (packed-detect) and its **packed** weight file (`model.safetensors`, not the
    /// dense `.fp16` name).
    pub(crate) fn subdir(&self) -> &'static str {
        match self {
            Clip::L => "text_encoder",
            Clip::BigG => "text_encoder_2",
        }
    }

    /// The vendored CLIP config for this encoder (sc-9527): CLIP-L (`text_encoder/`) vs OpenCLIP bigG
    /// (`text_encoder_2/`). Mirrors the stock `clip::Config::sdxl()` / `sdxl2()` the pipeline uses.
    pub(crate) fn vendored_config(&self) -> crate::clip::Config {
        match self {
            Clip::L => crate::clip::Config::sdxl(),
            Clip::BigG => crate::clip::Config::sdxl2(),
        }
    }
}

/// The two SDXL CLIP tokenizers (CLIP-L + CLIP-bigG), loaded+parsed **once** and cached on the
/// generator, reused across every `text_embeddings` call (sc-8991 / F-011) instead of re-reading the
/// `tokenizer.json` files from the hf-hub cache and re-parsing them on each encode. Model-agnostic (the
/// repos are fixed by [`Clip::sources`]), so a single pair serves every SDXL snapshot the generator
/// renders. These carry no VRAM, so caching them does not affect the sc-4987 CLIP-weight peak lever.
pub(crate) struct SdxlTokenizers {
    tok_l: Tokenizer,
    tok_g: Tokenizer,
}

impl SdxlTokenizers {
    /// Load both CLIP tokenizers from their (pinned, hf-hub-cached) repos. Call once per generator.
    pub(crate) fn load() -> Result<Self> {
        let (tok_l_repo, _) = Clip::L.sources();
        let (tok_g_repo, _) = Clip::BigG.sources();
        let tok_l = Tokenizer::from_file(hf_get(tok_l_repo, "tokenizer.json")?)
            .map_err(|e| CandleError::Msg(format!("load tokenizer {tok_l_repo}: {e}")))?;
        let tok_g = Tokenizer::from_file(hf_get(tok_g_repo, "tokenizer.json")?)
            .map_err(|e| CandleError::Msg(format!("load tokenizer {tok_g_repo}: {e}")))?;
        Ok(Self { tok_l, tok_g })
    }
}

/// Resolve a file from a (cached) HF repo — used only for the model-agnostic tokenizers + fp16-VAE-fix.
///
/// sc-9013 / F-029: the download is pinned to an immutable commit SHA ([`hub_revision`]) rather than
/// the hub's mutable `main` default, so an upstream force-push / compromise cannot silently alter the
/// weights or tokenization at request time. A repo with no pin is rejected up front.
pub(crate) fn hf_get(repo: &str, path: &str) -> Result<PathBuf> {
    use hf_hub::api::sync::Api;
    use hf_hub::{Repo, RepoType};
    let revision = hub_revision(repo)?;
    Api::new()
        .and_then(|api| {
            api.repo(Repo::with_revision(
                repo.to_string(),
                RepoType::Model,
                revision.to_string(),
            ))
            .get(path)
        })
        .map_err(|e| CandleError::Msg(format!("hf-hub fetch {repo}/{path}@{revision}: {e}")))
}

/// A txt2img pipeline handle. sc-4987 made loading **staged**: this carries only the
/// `StableDiffusionConfig` (the per-request latent dims), the snapshot `root`, and the compute
/// device/dtype — the heavy components (CLIP, UNet, VAE) are loaded *inside* [`generate`] in the
/// order they are needed and dropped as soon as they are not, so the dual CLIP is freed before the
/// UNet/VAE ever allocate. (Pre-sc-4987 this struct held all four components resident at once.)
pub(crate) struct Pipeline {
    config: StableDiffusionConfig,
    root: PathBuf,
    device: Device,
    dtype: DType,
    /// LoRA/LoKr adapters to merge into the UNet at component-load time (sc-5165). Fixed for the
    /// generator's lifetime (they come from the `LoadSpec`), so they do not enter the component cache
    /// key — only flash-attn does. Empty ⇒ the stock mmap `build_unet` path (zero regression).
    adapters: Vec<AdapterSpec>,
    /// The `LoadSpec::pid` component captured at load (epic 7840 / sc-7853), built into the cached
    /// [`Components`] so the PiD engine loads once alongside the UNet/VAE. `None` ⇒ native VAE decode.
    pid_spec: Option<PidWeights>,
}

/// The seed- and prompt-independent heavy components (UNet + f16 VAE), `Arc`-shared so they can be
/// **cached on the generator across `generate` calls** (sc-5037) and cheaply cloned out from under
/// the cache lock for a render. SDXL's UNet/VAE are resolution-agnostic (`build_unet`/`build_vae`
/// read only the fixed `unet`/`autoencoder` sub-configs, never the latent dims), so a single cached
/// pair serves every request size; the only construction input that varies is flash-attn, which the
/// generator keys the cache on. CLIP is deliberately **not** here — it stays load-on-demand-and-free
/// (the sc-4987 peak-VRAM lever), so caching the UNet/VAE does not make the dual CLIP resident.
#[derive(Clone)]
pub(crate) struct Components {
    pub(crate) unet: SdxlUnet,
    pub(crate) vae: Arc<AutoEncoderKL>,
    /// Optional NVIDIA PiD super-resolving decoder (epic 7840 / sc-7853); None ⇒ native VAE decode.
    pub(crate) pid: Option<Arc<PidEngine>>,
}

/// The SDXL denoise UNet, in one of two builds that share the txt2img `forward(x, t, ehs)` contract
/// (sc-9416):
///
/// - [`Self::Stock`] — the stock candle-transformers `UNet2DConditionModel`, built for a **dense**
///   diffusers snapshot (bf16/f16). Byte-identical to pre-sc-9416, incl. the fused flash-attention
///   path; this is the default for every dense SDXL/RealVisXL checkpoint (zero regression).
/// - [`Self::Vendored`] — the crate's vendored UNet, whose Linear surface packed-detects through the
///   shared `candle_gen::quant`. Built **only** for a pre-quantized MLX tier
///   (`SceneWorks/sdxl-base-mlx` q4/q8), where the whole attention/FF/proj/time-embed Linear surface
///   loads straight from the packed `{weight u32, scales, biases}` parts (no dense staging) and the
///   convolutions + norms stay dense. Runs the math attention (the vendored flash path is a stub).
///
/// Both are `Arc`-shared so the seed/prompt-independent UNet is cached across `generate` calls (sc-5037)
/// and cheaply cloned per render.
#[derive(Clone)]
pub(crate) enum SdxlUnet {
    Stock(Arc<UNet2DConditionModel>),
    Vendored(Arc<VendoredUNet>),
}

impl SdxlUnet {
    /// The txt2img denoise forward, dispatched to whichever build. Both compute the SDXL ε-prediction
    /// `[B, 4, h, w]` for `(latents, timestep, dual-CLIP embeddings)` — the packed vendored UNet is
    /// pinned bit-identical to the stock UNet on a dense build by the vendored-vs-stock parity test.
    pub(crate) fn forward(
        &self,
        xs: &Tensor,
        timestep: f64,
        encoder_hidden_states: &Tensor,
    ) -> Result<Tensor> {
        match self {
            Self::Stock(u) => Ok(u.forward(xs, timestep, encoder_hidden_states)?),
            Self::Vendored(u) => Ok(u.forward(xs, timestep, encoder_hidden_states)?),
        }
    }
}

/// One SDXL CLIP text encoder, in one of two builds that share the `forward(ids) -> last hidden`
/// contract (sc-9527):
///
/// - [`Self::Stock`] — the stock candle-transformers `ClipTextTransformer`, built for a **dense**
///   diffusers snapshot. Byte-identical to the pre-sc-9527 txt2img path (zero regression on every
///   dense SDXL/RealVisXL checkpoint).
/// - [`Self::Vendored`] — the crate's vendored CLIP tower, whose Linear surface packed-detects through
///   `candle_gen::quant`. Built **only** for a pre-quantized MLX tier (`SceneWorks/sdxl-base-mlx`
///   q4/q8), where every attention / MLP `Linear` loads straight from the packed
///   `{weight u32, scales, biases}` parts (no dense staging).
///
/// The vendored tower is pinned bit-identical to the stock one on a dense build by the
/// `clip::tests::vendored_dense_matches_stock` parity test.
enum CandleModule {
    Stock(stable_diffusion::clip::ClipTextTransformer),
    Vendored(crate::clip::ClipTextTransformer),
}

impl CandleModule {
    /// The last-hidden-state forward `ids [B, S] -> [B, S, embed_dim]`, dispatched to whichever build.
    fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        match self {
            Self::Stock(m) => Ok(m.forward(ids)?),
            Self::Vendored(m) => Ok(m.forward(ids)?),
        }
    }
}

impl Pipeline {
    /// Build the (light) pipeline handle for the SDXL snapshot `root` at the given device/dtype (f16)
    /// and request dims. This does **no** weight I/O — the config's only request-dependent fields are
    /// the latent dims; the heavy components load lazily in [`generate`].
    pub(crate) fn load(
        root: &Path,
        device: &Device,
        dtype: DType,
        width: u32,
        height: u32,
        adapters: &[AdapterSpec],
        pid_spec: Option<PidWeights>,
    ) -> Result<Self> {
        // The config's only request-dependent fields are the latent dims; the component configs
        // (clip/clip2/unet/autoencoder) are fixed for SDXL.
        let config = StableDiffusionConfig::sdxl(None, Some(height as usize), Some(width as usize));
        Ok(Self {
            config,
            root: root.to_path_buf(),
            device: device.clone(),
            dtype,
            adapters: adapters.to_vec(),
            pid_spec,
        })
    }

    /// SDXL dual-CLIP conditioning: encode `prompt` (cond) and `uncond` through both encoders, stack
    /// `[uncond, cond]` on the batch axis, and concatenate the two encoders on the feature axis —
    /// shape `[2, tokens, 2048]`, cast to the compute dtype. Mirrors the spike's `text_embeddings`.
    ///
    /// sc-4987: each encoder is loaded, run, and dropped **inside** [`encode_one`] before the next is
    /// loaded — so the two CLIP encoders are never co-resident, and both are gone when this returns.
    /// sc-5037: the generator calls this **before** acquiring the (possibly cached-resident) UNet/VAE,
    /// preserving the cold-call ordering (CLIP freed before they load); on a warm call the UNet/VAE are
    /// already resident, but only one CLIP encoder is ever resident at a time (`build_unet`+VAE ≈ 7 GiB
    /// + one CLIP ≤ 1.4 GiB stays under the denoise-time peak, so the sc-4987 high-water is preserved).
    pub(crate) fn text_embeddings(
        &self,
        toks: &SdxlTokenizers,
        prompt: &str,
        uncond: &str,
    ) -> Result<Tensor> {
        let l = self.encode_one(Clip::L, &toks.tok_l, prompt, uncond)?;
        let g = self.encode_one(Clip::BigG, &toks.tok_g, prompt, uncond)?;
        Ok(Tensor::cat(&[l, g], D::Minus1)?)
    }

    /// Load one CLIP encoder, encode `[uncond, cond]` through it (padded to its
    /// `max_position_embeddings`), and return the embeddings — the encoder weights are loaded into a
    /// local and **dropped when this function returns** (sc-4987), freeing its VRAM before the next
    /// encoder / the UNet load.
    fn encode_one(
        &self,
        which: Clip,
        tokenizer: &Tokenizer,
        prompt: &str,
        uncond: &str,
    ) -> Result<Tensor> {
        let (_tok_repo, weights_sub) = which.sources();
        let clip_cfg = match which {
            Clip::L => &self.config.clip,
            Clip::BigG => self
                .config
                .clip2
                .as_ref()
                .ok_or_else(|| CandleError::Msg("sdxl config missing clip2".into()))?,
        };
        // The tokenizer is now loaded+parsed ONCE on the generator (sc-8991 / F-011) and passed in,
        // rather than re-read from `hf_get(tok_repo, ...)` per encode. The CLIP *weights* still load and
        // drop inside this function (the sc-4987 peak-VRAM lever); only the tiny tokenizer is cached.
        //
        // sc-9527 (sc-9089j follow-up to sc-9416): the MLX SDXL tiers ALSO pack the dual CLIP text
        // encoders (a `quantization` block in `text_encoder{,_2}/config.json` + `.scales`-packed Linears
        // under `model.safetensors`). The txt2img conditioning uses only each encoder's last hidden
        // state (`forward`), so we build the **vendored, packed-detecting** CLIP tower when the tier is
        // packed — every Linear (attn q/k/v/out_proj, MLP fc1/fc2) loads straight from the packed parts —
        // and the stock dense builder otherwise (byte-identical, pinned by the vendored-vs-stock parity
        // test). The `group_size` is threaded from the component config (sc-9410).
        let text_model: CandleModule = match detect_packed_clip(&self.root, &which)? {
            Some((packed_file, group_size)) => {
                let vs = candle_gen::mmap_var_builder(&[packed_file], self.dtype, &self.device)?;
                let tower = crate::clip::ClipTextTransformer::new_gs(
                    vs,
                    &which.vendored_config(),
                    group_size,
                )?;
                CandleModule::Vendored(tower)
            }
            None => {
                // sc-3674: load CLIP at the compute dtype (f16), not the spike's F32. The fp16
                // safetensors load directly, the forward runs f16 (diffusers loads CLIP fp16 too), and it
                // halves the text-encoder VRAM (CLIP-bigG ~2.8→1.4 GiB) with no visible quality change.
                // The embeddings are cast to `dtype` below.
                let stock = stable_diffusion::build_clip_transformer(
                    clip_cfg,
                    snapshot_file(&self.root, weights_sub)?,
                    &self.device,
                    self.dtype,
                )?;
                CandleModule::Stock(stock)
            }
        };

        let vocab = tokenizer.get_vocab(true);
        let pad_token = clip_cfg
            .pad_with
            .clone()
            .unwrap_or_else(|| "<|endoftext|>".into());
        let pad_id = *vocab
            .get(pad_token.as_str())
            .ok_or_else(|| CandleError::Msg(format!("pad token {pad_token:?} not in vocab")))?;

        let encode = |text: &str| -> Result<Tensor> {
            let mut tokens = tokenizer
                .encode(text, true)
                .map_err(|e| CandleError::Msg(format!("tokenize: {e}")))?
                .get_ids()
                .to_vec();
            let max = clip_cfg.max_position_embeddings;
            if tokens.len() > max {
                return Err(CandleError::Msg(format!(
                    "prompt too long: {} tokens > {max}",
                    tokens.len()
                )));
            }
            while tokens.len() < max {
                tokens.push(pad_id);
            }
            Ok(Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?)
        };

        let cond = text_model.forward(&encode(prompt)?)?;
        let uncond = text_model.forward(&encode(uncond)?)?;
        Ok(Tensor::cat(&[uncond, cond], 0)?.to_dtype(self.dtype)?)
        // `text_model` drops here, freeing this encoder's weights before the caller loads the next
        // (sc-4987). The `tokenizer` is borrowed from the generator's cache and outlives this call.
    }

    /// Load the heavy [`Components`] (UNet + f16 VAE) for the given flash-attn setting. The UNet reads
    /// from the snapshot (fused flash-attention when built `--features flash-attn` AND `use_flash_attn`
    /// — sc-3674); the f16-stable VAE (`madebyollin/sdxl-vae-fp16-fix`) resolves via `hf-hub`. The
    /// generator owns the caching of the result across calls (sc-5037); this is the cache-miss loader.
    pub(crate) fn load_components(&self, use_flash_attn: bool) -> Result<Components> {
        // sc-9416: a **packed** MLX tier (`SceneWorks/sdxl-base-mlx` q4/q8) ships its UNet under the
        // non-`.fp16` filename with a `quantization` block in `unet/config.json` and `.scales`-packed
        // Linear weights. Detect it and load the vendored packed-detecting UNet straight from the packed
        // parts (no dense staging); every dense snapshot keeps the stock build below, unchanged.
        let unet = match self.detect_packed_unet()? {
            Some((packed_file, group_size)) => {
                // sc-9528: a packed tier WITH a trained LoRA/LoKr now folds via dequant → merge →
                // keep-dense (option a). The delta cannot be added to the u32-packed weights directly
                // and SDXL merges (not residuals) for chaos-sensitivity, so the adapted Linears are
                // dequantized, the delta is folded through `adapters::merge_adapters` (shared, verbatim),
                // and only those adapted layers are served dense — every unadapted layer stays packed.
                let vendored = if self.adapters.is_empty() {
                    self.load_packed_unet(&packed_file, group_size)?
                } else {
                    self.load_packed_unet_with_adapters(&packed_file, group_size)?
                };
                SdxlUnet::Vendored(Arc::new(vendored))
            }
            None => {
                let unet_file =
                    snapshot_file(&self.root, "unet/diffusion_pytorch_model.fp16.safetensors")?;
                if use_flash_attn {
                    // The fused flash-attn kernel never materializes the full `[B·H, S, S]` scores
                    // tensor, so it does not hit the i32-overflow (sc-11154 / F-081); keep the stock
                    // candle UNet so the fused kernel is used (byte-identical to pre-sc-5165).
                    let unet = if self.adapters.is_empty() {
                        self.config
                            .build_unet(unet_file, &self.device, 4, true, self.dtype)?
                    } else {
                        self.build_unet_with_adapters(&unet_file, true)?
                    };
                    SdxlUnet::Stock(Arc::new(unet))
                } else {
                    // Math-path attention (the default — no `flash-attn` feature): the stock candle
                    // UNet materializes a full `[B·H, S, S]` scores tensor that overflows i32 at
                    // ≥ ~1664² (2048²: `2·10·16384² ≈ 5.4e9 > i32::MAX`) and silently corrupts on CUDA
                    // (sc-11154 / F-081). Route the dense load through the vendored UNet, whose math
                    // attention is the i32-overflow-safe `sdpa_budgeted_flat` — bit-identical to the
                    // stock forward per `vendored_unet_matches_stock_forward`.
                    let vendored = if self.adapters.is_empty() {
                        self.load_dense_vendored_unet(&unet_file)?
                    } else {
                        self.load_dense_vendored_unet_with_adapters(&unet_file)?
                    };
                    SdxlUnet::Vendored(Arc::new(vendored))
                }
            }
        };
        let vae = self.config.build_vae(
            hf_get(VAE_FIX_REPO, VAE_FIX_FILE)?,
            &self.device,
            self.dtype,
        )?;
        // Load the optional PiD super-resolving decoder once (epic 7840 / sc-7853) when the caller
        // opted in via `LoadSpec::pid`; SDXL's own `sdxl` latent-space student. `None` ⇒ native VAE.
        let pid = match self.pid_spec.as_ref() {
            Some(spec) => Some(Arc::new(PidEngine::from_spec(
                spec,
                PID_BACKBONE,
                &self.device,
            )?)),
            None => None,
        };
        Ok(Components {
            unet,
            vae: Arc::new(vae),
            pid,
        })
    }

    /// Detect a pre-quantized MLX SDXL tier at this pipeline's `root` — the free
    /// [`detect_packed_unet`] (shared with the InstantID/edit/IP-Adapter UNet loader, sc-10813) keyed
    /// on `self.root`.
    fn detect_packed_unet(&self) -> Result<Option<(PathBuf, usize)>> {
        detect_packed_unet(&self.root)
    }

    /// Build the vendored packed-detecting SDXL UNet from a packed MLX-tier `unet/` checkpoint
    /// (sc-9416). One mmap'd VarBuilder feeds the whole UNet; `linear_detect` in the vendored
    /// attention/FF/proj/time-embed sites builds a quantized `QLinear` straight from each packed
    /// `{weight, scales, biases}` triple, while the convolutions + norms load dense. No adapter is
    /// installed (the packed tier is inference-only), so the four attention projections' `LoraLinear`
    /// bases are their packed `QLinear` and the forward is exactly `x·Wᵀ + b` (dequant-on-forward).
    fn load_packed_unet(&self, unet_file: &Path, _group_size: usize) -> Result<VendoredUNet> {
        let vs =
            candle_gen::mmap_var_builder(&[unet_file.to_path_buf()], self.dtype, &self.device)?;
        // The vendored `new` threads the default MLX group size (64) — validated == the config group in
        // `detect_packed_unet` — through its packed-detecting leaves; `sdxl_unet_config` is the canonical
        // 3-block SDXL geometry (`use_linear_projection = true`, matching the packed `proj_in/out`).
        Ok(VendoredUNet::new(vs, 4, 4, false, sdxl_unet_config())?)
    }

    /// Build the vendored packed UNet from a packed MLX-tier checkpoint with the [`AdapterSpec`]s folded
    /// in (sc-9528 — the sc-9089j follow-up deferred from sc-9416). Since a trained LoRA/LoKr delta
    /// cannot be added to the u32-packed weights, and SDXL merges rather than adds a forward-time
    /// residual (chaos-sensitivity, see [`crate::adapters`]), the packed Linears are **dequantized** to
    /// dense f32, the delta is **folded** through [`crate::adapters::merge_adapters`] (shared, verbatim —
    /// no LoRA/LoKr math reimplemented), and only the adapted layers are **kept dense** while every
    /// unadapted layer stays packed ([`crate::packed_adapters`]). The resulting mixed (dense-adapted +
    /// packed-unadapted) tensor map feeds the vendored UNet through a `VarBuilder::from_tensors`, whose
    /// per-Linear `linear_detect_gs` packed-detects each layer by its `.scales` sibling — so an adapted
    /// layer routes through the dense arm and an unadapted one keeps its packed footprint.
    fn load_packed_unet_with_adapters(
        &self,
        unet_file: &Path,
        group_size: usize,
    ) -> Result<VendoredUNet> {
        // The vendored UNet's top-level constructor threads only the default MLX group 64 through its
        // blocks; a non-64 tier would dequantize/repartition at `group_size` while the UNet reads at 64.
        // Refuse it loudly (mirrors `detect_packed_unet`) rather than silently mis-fold.
        crate::packed_adapters::assert_group_size_supported(group_size)?;
        let raw = candle_gen::candle_core::safetensors::load(unet_file, &Device::Cpu)?;
        let merged =
            crate::packed_adapters::fold_adapters_into_packed_map(raw, &self.adapters, group_size)?;
        // `from_tensors` serves the u32 packed weights via the vendored seam's `get_unchecked_dtype`
        // (exactly as the mmap path) and the dense merged weights via the vb dtype — so the packed vs
        // dense fork is taken per Linear as loaded. `false` = no flash-attn on the packed path.
        let vs = VarBuilder::from_tensors(merged, self.dtype, &self.device);
        Ok(VendoredUNet::new(vs, 4, 4, false, sdxl_unet_config())?)
    }

    /// Build the UNet with the [`AdapterSpec`]s merged into its weights (sc-5165). The base UNet
    /// tensors are loaded onto **CPU** at their native dtype, the LoRA/LoKr deltas are folded in
    /// ([`crate::adapters::merge_adapters`], f32 math), then the stock candle UNet is built from the
    /// merged map — each tensor moved to the device and cast to the load dtype as the VarBuilder serves
    /// it, so the merged weights live on the GPU exactly like the mmap path and peak GPU is unchanged.
    ///
    /// We merge into the dense weights rather than add a forward-time residual because SDXL's sampler
    /// is chaos-sensitive (`(W+δ)·x` ≠ `W·x + δ·x` to ~1 ULP); see [`crate::adapters`].
    fn build_unet_with_adapters(
        &self,
        unet_file: &Path,
        use_flash_attn: bool,
    ) -> Result<UNet2DConditionModel> {
        let mut tensors = candle_gen::candle_core::safetensors::load(unet_file, &Device::Cpu)?;
        crate::adapters::merge_adapters(&mut tensors, &self.adapters)?;
        // `build_unet` reads `StableDiffusionConfig`'s private `unet` config + builds its own
        // VarBuilder from a file path, so it cannot be fed a merged map; replicate its two lines with
        // the canonical SDXL UNet config (pinned to candle's `StableDiffusionConfig::sdxl_`).
        let vs = VarBuilder::from_tensors(tensors, self.dtype, &self.device);
        let unet = UNet2DConditionModel::new(vs, 4, 4, use_flash_attn, sdxl_stock_unet_config())?;
        Ok(unet)
    }

    /// Build the **dense** SDXL UNet through the vendored stack (sc-11154 / F-081). The vendored math
    /// attention routes through the shared i32-overflow-safe [`candle_gen::sdpa_budgeted_flat`], so the
    /// dense (no-flash) path no longer materializes an over-`i32::MAX` `[B·H, S, S]` scores tensor at
    /// the advertised 2048² envelope (the stock candle UNet does — silent CUDA corruption). The
    /// vendored copy is bit-identical to the stock forward (`vendored_unet_matches_stock_forward`), and
    /// the diffusers fp16 checkpoint layout is shared: its `linear_detect` leaves see no `.scales`
    /// siblings and load every Linear dense (the same code path the packed loader takes for un-packed
    /// tensors). No adapter is installed, so each attention projection's base is its dense Linear.
    fn load_dense_vendored_unet(&self, unet_file: &Path) -> Result<VendoredUNet> {
        let vs =
            candle_gen::mmap_var_builder(&[unet_file.to_path_buf()], self.dtype, &self.device)?;
        Ok(VendoredUNet::new(vs, 4, 4, false, sdxl_unet_config())?)
    }

    /// Dense vendored UNet with the [`AdapterSpec`]s merged in (sc-11154 / F-081) — the adapted
    /// counterpart of [`Self::load_dense_vendored_unet`]. Mirrors [`Self::build_unet_with_adapters`]
    /// (base tensors on CPU → [`crate::adapters::merge_adapters`] f32 fold → `from_tensors`) but targets
    /// the vendored stack so the i32-overflow-safe budgeted attention applies. SDXL merges the delta
    /// into the dense weights (not a forward residual) for chaos-sensitivity; see [`crate::adapters`].
    fn load_dense_vendored_unet_with_adapters(&self, unet_file: &Path) -> Result<VendoredUNet> {
        let mut tensors = candle_gen::candle_core::safetensors::load(unet_file, &Device::Cpu)?;
        crate::adapters::merge_adapters(&mut tensors, &self.adapters)?;
        let vs = VarBuilder::from_tensors(tensors, self.dtype, &self.device);
        Ok(VendoredUNet::new(vs, 4, 4, false, sdxl_unet_config())?)
    }

    /// Render `req` against pre-resolved `text_embeddings` and (caller-cached, sc-5037) `unet`/`vae`,
    /// emitting per-step progress and honoring `req.cancel`. Returns one `gen_core::Image` per
    /// `req.count` (each with seed `base_seed + index`). The denoise+decode here is unchanged from
    /// sc-4987 — only the component *ownership* moved out to the generator so it can cache them.
    pub(crate) fn render(
        &self,
        req: &GenerationRequest,
        text_embeddings: &Tensor,
        unet: &SdxlUnet,
        vae: &AutoEncoderKL,
        pid: Option<&PidEngine>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Vec<Image>> {
        // sc-6128: a `lightning` request runs the few-step Euler-trailing path. Every other render —
        // the omitted-sampler default AND every explicit curated solver name (incl. `ddim`) — routes
        // through the unified curated `Sampler` over `DiscreteModelSampling` (EPS), epic 7114 P4 / sc-7124.
        //
        // sc-10826: the omitted-sampler default previously ran candle-transformers' native
        // `DDIMScheduler` inference loop, which rendered a ghosted, translucent double-exposure
        // (guidance-invariant) while every curated solver — including the curated `ddim` — is clean.
        // So the default now maps to `DEFAULT_SAMPLER` (the curated `ddim`), and the native loop is
        // gone. `ddim` no longer diverts to the native path; it takes the curated `ddim` solver like
        // every other name. Determinism is preserved (curated `ddim` is eta=0 / non-ancestral).
        let lightning = req.sampler.as_deref() == Some(LIGHTNING_SAMPLER);
        // The curated solver name for a non-lightning render: the request's name, or `ddim` by default
        // (`resolve_sampler`). `None` ⇒ a `lightning` render (its own path). An unknown name falls back
        // to euler inside `run_curated_sampler` (N3 — never a hard fail).
        let curated: Option<&str> = resolve_sampler(req.sampler.as_deref());
        let steps = req.steps.map(|s| s as usize).unwrap_or(if lightning {
            LIGHTNING_DEFAULT_STEPS
        } else {
            DEFAULT_STEPS
        });
        let guidance = req.guidance.map(|g| g as f64).unwrap_or(DEFAULT_GUIDANCE);
        // Lightning is a distilled, classifier-free sampler — it never runs CFG (the worker sends
        // guidance 1.0 for `realvisxl_lightning`, and CFG on a CFG-free checkpoint degrades output).
        // The DDIM path honors the request guidance exactly as before.
        let use_guide = !lightning && guidance > 1.0;
        let base_seed = req.seed.unwrap_or_else(gen_core::default_seed);
        let total = steps as u32;
        let (lat_h, lat_w) = (self.config.height / 8, self.config.width / 8);

        // Resolve the decode seam once for the whole batch (epic 7840 / sc-7853): a per-generation PiD
        // decoder bound to this prompt when `req.use_pid` is set (errors if requested but not loaded),
        // else `None` → the native SDXL VAE decode. Shared across `count` images (same prompt).
        let pid_decoder =
            candle_gen_pid::resolve_pid_decoder(pid, req, base_seed, crate::MODEL_ID)?;

        // Lightning precompute (seed-independent): the trailing-Euler policy + the cond-only text
        // embedding (row 1 of the `[uncond, cond]` dual-CLIP stack — Lightning runs one conditioned
        // forward per step, so the uncond row is unused). Built once, reused for every image.
        let lightning_ctx = if lightning {
            Some((
                lightning_policy(steps)?,
                text_embeddings.narrow(0, 1, 1)?.contiguous()?,
            ))
        } else {
            None
        };

        candle_gen::for_each_image_seed(base_seed, req.count, |seed| {
            // sc-3673 — deterministic, launch-portable initial noise: draw N(0,1) from a
            // fixed-algorithm CPU RNG (`StdRng`, ChaCha-based) seeded by `seed`, build the latent on
            // CPU, then move it to the compute device. This replaces candle's CUDA `device.set_seed`
            // + on-device `randn`, whose seed→noise mapping was NOT portable across launch
            // environments and occasionally collapsed the sample to garbage (sc-3498). Paired with the
            // non-ancestral solver (DDIM, or the deterministic Lightning Euler), the whole generation
            // is a pure function of `(seed, request)` — same seed ⇒ same image, any launch.
            let n = 4 * lat_h * lat_w;
            let mut rng = StdRng::seed_from_u64(seed);
            let noise = candle_gen::seeded_normal_vec(&mut rng, n);
            let init = Tensor::from_vec(noise, (1, 4, lat_h, lat_w), &Device::Cpu)?
                .to_device(&self.device)?;

            let latents = if let Some((policy, cond)) = &lightning_ctx {
                self.denoise_lightning(&init, policy, cond, unet, &req.cancel, on_progress, total)?
            } else if let Some(name) = curated {
                // The default path (sc-10826): `curated` is `Some` for every non-lightning render, so
                // the omitted-sampler default (→ the curated `ddim`) and every explicit curated name
                // run this one unified sampler. The native candle-transformers DDIM loop is gone.
                self.denoise_curated(
                    req,
                    name,
                    &init,
                    text_embeddings,
                    unet,
                    steps,
                    use_guide,
                    guidance,
                    seed,
                    on_progress,
                )?
            } else {
                // Unreachable by construction: `curated` is `Some` whenever `lightning_ctx` is `None`
                // (a non-lightning render always resolves a curated name via `DEFAULT_SAMPLER`). A
                // typed error rather than an `unwrap`/`unreachable!` so a future routing change can't
                // silently fall through to a broken (or removed) path.
                return Err(CandleError::Msg(
                    "sdxl: no denoise path selected (neither lightning nor curated) — a routing bug"
                        .into(),
                ));
            };

            on_progress(Progress::Decoding);
            self.decode(vae, pid_decoder.as_ref(), &latents)
        })
    }

    /// The SDXL-**Lightning** few-step denoise (sc-6128) — diffusers Euler-trailing, ε-prediction,
    /// **CFG-off**. Distilled Lightning checkpoints are trained classifier-free, so this runs a single
    /// conditioned UNet forward per step (no uncond batch, no CFG combine).
    ///
    /// The latents live in diffusers' un-normalized **σ-space** (kept f32, unlike the DDIM path's f16
    /// latents): the prior is `unit_noise · σ_max`, the model input is `x/√(σ²+1)` cast to the UNet
    /// dtype, and each step is the deterministic Euler update `x ← x + ε·(σ_{i+1} − σ_i)` in f32. That
    /// update is the candle tensor application of [`gen_core::sampling`]'s neutral [`LightningPolicy`]
    /// coefficients (`a_x = 1`, `a_noise = 0`), mirroring `mlx-gen-sdxl`'s `apply_step` — so the two
    /// backends share one reference schedule.
    ///
    /// `init` is the seeded unit-normal noise (CPU `StdRng` → device, f32; the sc-3673 launch-portable
    /// contract); `cond` is the cond-only text embedding `[1, T, 2048]`. Returns latents in the compute
    /// dtype (f16) for the shared [`decode`](Self::decode).
    #[allow(clippy::too_many_arguments)]
    fn denoise_lightning(
        &self,
        init: &Tensor,
        policy: &LightningPolicy,
        cond: &Tensor,
        unet: &SdxlUnet,
        cancel: &gen_core::runtime::CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
        total: u32,
    ) -> Result<Tensor> {
        // σ-space prior: unit noise · the largest σ (init_noise_sigma for trailing spacing).
        let mut latents = init.affine(policy.init_noise_scale() as f64, 0.0)?;
        for i in 0..policy.num_steps() {
            if cancel.is_cancelled() {
                return Err(CandleError::Canceled);
            }
            let c = policy.coeffs(i);
            // Model-input scaling x/√(σ²+1), cast to the UNet compute dtype (f16). CFG-off ⇒ batch 1.
            let x_in = latents.affine(c.c_in as f64, 0.0)?.to_dtype(self.dtype)?;
            let eps = unet
                .forward(&x_in, c.timestep as f64, cond)?
                .to_dtype(DType::F32)?;
            // Euler ε-pred step in f32: x + ε·(σ_{i+1} − σ_i) (a_x = 1, a_noise = 0, deterministic).
            latents = (latents + eps.affine(c.a_out as f64, 0.0)?)?;
            on_progress(Progress::Step {
                current: i as u32 + 1,
                total,
            });
        }
        // The shared `decode` expects the compute dtype (f16), like the DDIM loop's latents.
        Ok(latents.to_dtype(self.dtype)?)
    }

    /// The **curated** ε/DDPM denoise (epic 7114 P4, sc-7124) — an ADDITIVE option alongside the native
    /// DDIM default and Lightning. Drives the unified [`gen_core::sampling::Sampler`] (`euler` /
    /// `euler_ancestral` / `heun` / `dpmpp_2m` / `dpmpp_sde` / `uni_pc` / `lcm`) over a
    /// [`DiscreteModelSampling`] (SDXL ε-prediction, `scaled_linear` β over 1000 train steps). The
    /// `scheduler` axis (`normal` default / `karras` / `sgm_uniform` / …) picks the σ schedule via
    /// [`candle_gen::resolve_schedule`]. Latents live in k-diffusion VE σ-space (prior = unit noise ·
    /// σ_max), kept f32 (like the Lightning path); the [`DiscreteModelSampling`] recombines ε → x0 and
    /// supplies the `1/√(σ²+1)` input scaling, so the closure just runs the UNet + CFG and returns raw ε.
    /// The native DDIM/Lightning defaults are untouched, so this never affects the N1 default-parity gate.
    #[allow(clippy::too_many_arguments)]
    fn denoise_curated(
        &self,
        req: &GenerationRequest,
        sampler: &str,
        init: &Tensor,
        text_embeddings: &Tensor,
        unet: &SdxlUnet,
        steps: usize,
        use_guide: bool,
        guidance: f64,
        seed: u64,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Tensor> {
        let sched = sdxl_alpha_schedule()?;
        let ms = DiscreteModelSampling::sdxl(&sched);
        // Native curated schedule = ComfyUI's SDXL default (`normal`); the scheduler axis overrides it.
        let native = schedule_sigmas(Scheduler::Normal, &ms, steps);
        let sigmas = candle_gen::resolve_schedule(req.scheduler.as_deref(), &ms, steps, &native);
        // VE prior: unit noise · σ_max (sigmas[0]); kept f32 through the sampler (cast to f16 per eval).
        let latents = (init * sigmas[0] as f64)?;
        let out = candle_gen::run_curated_sampler(
            Some(sampler),
            &ms,
            &sigmas,
            latents,
            seed,
            &req.cancel,
            on_progress,
            |x_in, t| -> Result<Tensor> {
                // `x_in` is already `1/√(σ²+1)`-scaled by `denoise()`; `t` is the nearest training-step
                // index the UNet embeds. CFG batches/combines exactly like the native DDIM path.
                let model_in = if use_guide {
                    Tensor::cat(&[x_in, x_in], 0)?
                } else {
                    x_in.clone()
                };
                let model_in = model_in.to_dtype(self.dtype)?;
                let noise_pred = unet.forward(&model_in, t as f64, text_embeddings)?;
                let eps = if use_guide {
                    let chunks = noise_pred.chunk(2, 0)?;
                    let (uncond, cond) = (&chunks[0], &chunks[1]);
                    (uncond + ((cond - uncond)? * guidance)?)?
                } else {
                    noise_pred
                };
                // Raw ε in f32 so the DiscreteModelSampling x0 recombine + solver math stay f32.
                Ok(eps.to_dtype(DType::F32)?)
            },
        )?;
        // The shared `decode` expects the compute dtype (f16), like the DDIM/Lightning latents.
        Ok(out.to_dtype(self.dtype)?)
    }

    /// Decode latents to an RGB8 [`Image`], either through the native VAE or — when a PiD decoder
    /// resolved (epic 7840 / sc-7853) — the super-resolving PiD student (emits a larger `[1,3,4H,4W]`
    /// tensor). Both produce `[-1, 1]` pixels; [`to_image`](Self::to_image) reads the size from the
    /// tensor (never `latent*8`).
    ///
    /// **Latent convention (sc-7848 parity — NOT zero-transform on candle):** the PiD `sdxl` student
    /// trained on the **0.13025-normalized** latent — the scaled sampler output `latents`. In candle the
    /// VAE de-scale happens here in the pipeline (`latents / VAE_SCALE`) rather than inside `vae.decode`
    /// (unlike the qwen/flux families, whose VAE de-normalizes internally), so PiD gets `latents`
    /// (normalized) while the VAE gets the de-scaled raw latent. This matches MLX, where `vae.decode`
    /// de-scales internally and both paths receive that same normalized tensor.
    fn decode(
        &self,
        vae: &AutoEncoderKL,
        pid: Option<&PidDecoder>,
        latents: &Tensor,
    ) -> Result<Image> {
        let img = match pid {
            Some(pid) => pid.decode(latents)?,
            None => self.decode_image(vae, &(latents / VAE_SCALE)?)?,
        };
        self.to_image(&img)
    }

    /// Convert a decoded pixel tensor `[1, 3, H, W]` in `[-1, 1]` → RGB8 [`Image`] (`x/2 + 0.5`, clamp,
    /// ×255). Shared by the native VAE decode and the PiD super-resolving decode; the output size is
    /// read from the tensor, never assumed (PiD may be 4× the VAE-native size).
    fn to_image(&self, img: &Tensor) -> Result<Image> {
        let img = ((img / 2.)? + 0.5)?.clamp(0f32, 1f32)?;
        let img = (img * 255.)?
            .to_dtype(DType::U8)?
            .i(0)?
            .to_device(&Device::Cpu)?;
        let (c, h, w) = img.dims3()?;
        if c != 3 {
            return Err(CandleError::Msg(format!("expected 3 channels, got {c}")));
        }
        let pixels = img.permute((1, 2, 0))?.flatten_all()?.to_vec1::<u8>()?;
        Ok(Image {
            width: w as u32,
            height: h as u32,
            pixels,
        })
    }

    /// Decode the already-unscaled latent to an image tensor `[1, 3, H, W]` via the shared
    /// [`tiled_vae_decode`] — tiled (sc-4987) when [`crate::vae_tiling_enabled`] is set AND the output
    /// exceeds the tiling threshold (512²); otherwise the monolithic `AutoEncoderKL::decode`.
    fn decode_image(&self, vae: &AutoEncoderKL, unscaled: &Tensor) -> Result<Tensor> {
        tiled_vae_decode(vae, unscaled)
    }
}

/// Decode an already-unscaled SDXL latent `[1, 4, h, w]` to an image tensor `[1, 3, H, W]`, applying the
/// sc-4987 budgeted VAE tiling when [`crate::vae_tiling_enabled`] is set AND the output exceeds the
/// tiling threshold (512²); otherwise the monolithic `AutoEncoderKL::decode`. The non-tiling path is
/// byte-identical to a bare `vae.decode`, so ≤512² renders and the conformance suite are unaffected.
///
/// This is the single decode seam for **every** SDXL lane (F-061 / sc-9045): the registered
/// [`Pipeline::decode`] and the bespoke [`crate::denoise::decode_image`] (trainer preview, IP / edit
/// providers) both route through it, so all lanes get the same bounded-peak decode at identical
/// resolutions instead of the bespoke providers decoding 1024² monolithically.
pub(crate) fn tiled_vae_decode(vae: &AutoEncoderKL, unscaled: &Tensor) -> Result<Tensor> {
    if crate::vae_tiling_enabled() {
        let cfg = sdxl_tiling_config();
        let (_, _, h, w) = unscaled.dims4()?;
        if cfg.needs_tiling(SDXL_VAE_TILING, 1, h as i32, w as i32) {
            return tile_blend_decode(
                unscaled,
                SDXL_VAE_TILING,
                &cfg,
                |tile| Ok(vae.decode(tile)?),
            );
        }
    }
    Ok(vae.decode(unscaled)?)
}

/// Tiled VAE decode with trapezoidal seam blending (sc-4987) — the candle port of mlx-gen's
/// `tile_decode_accumulate`, specialized to a 4-D image latent `[B, C, h, w]` (no temporal axis).
///
/// Splits `unscaled` (the already-`/VAE_SCALE` latent) into the overlapping spatial tiles planned by
/// [`TilingConfig::plan`], decodes each via `decode_tile`, and accumulates `Σ(maskᵢ·decodeᵢ)` and
/// `Σ maskᵢ` into full-size output/weight buffers, returning `output / max(weights, 1e-8)`. Because
/// the tiles overlap and the per-axis masks are a partition of unity, the blend is exact for an
/// identity decode (the CPU unit test) and seam-free for the real VAE (the overlap absorbs the
/// boundary-conv mismatch). Peak memory is bounded by **one tile's** decode — the win — plus the two
/// full-size (but f32, ~12 MiB at 1024²) accumulators.
///
/// Accumulation is in f32: `decode_tile` runs f16, but the blend divide wants the mask precision and
/// f32 at output resolution is negligible. The returned tensor is `[1, 3, out_h, out_w]` f32, which
/// the caller's `/2 + 0.5 / clamp / ×255` post-processing consumes identically to the f16 mono path.
fn tile_blend_decode(
    unscaled: &Tensor,
    vae_tiling: VaeTiling,
    cfg: &TilingConfig,
    decode_tile: impl Fn(&Tensor) -> Result<Tensor>,
) -> Result<Tensor> {
    let device = unscaled.device();
    let (_b, _c, h, w) = unscaled.dims4()?;
    // f = 1: an image latent has no temporal axis, so the plan's single temporal tile is a no-op and
    // we iterate the spatial (h × w) tiles only.
    let plan = cfg.plan(vae_tiling, 1, h as i32, w as i32);
    let (out_h, out_w) = (plan.out_h as usize, plan.out_w as usize);

    let mut output: Option<Tensor> = None; // [1, 3, out_h, out_w] f32
    let mut weights: Option<Tensor> = None; // [1, 1, out_h, out_w] f32
    for hh in &plan.h {
        for ww in &plan.w {
            let tile = unscaled
                .narrow(2, hh.start as usize, (hh.end - hh.start) as usize)?
                .narrow(3, ww.start as usize, (ww.end - ww.start) as usize)?;
            let dec = decode_tile(&tile)?.to_dtype(DType::F32)?;

            // Clip the decoded tile + masks to the planned output span (guards the VAE returning a
            // pixel or two over/under the latent×scale span; for SDXL's exact ×8 this is a no-op).
            let (_, _, dh, dw) = dec.dims4()?;
            let ah = dh.min((hh.out_stop - hh.out_start) as usize);
            let aw = dw.min((ww.out_stop - ww.out_start) as usize);
            let dec = dec.narrow(2, 0, ah)?.narrow(3, 0, aw)?;

            // 1-D trapezoidal masks → outer product, each broadcasting along its own (h / w) axis.
            let hm = Tensor::from_slice(&hh.mask[..ah], (1, 1, ah, 1), device)?;
            let wm = Tensor::from_slice(&ww.mask[..aw], (1, 1, 1, aw), device)?;
            let blend = hm.broadcast_mul(&wm)?; // [1, 1, ah, aw]
            let weighted = dec.broadcast_mul(&blend)?; // [1, 3, ah, aw]

            // Place each tile at its (out_start) offset by zero-padding to the full output shape, then
            // add — the bounded-peak accumulate (mirrors the reference's full-size output+weights).
            let (pad_top, pad_bottom) =
                (hh.out_start as usize, out_h - (hh.out_start as usize + ah));
            let (pad_left, pad_right) =
                (ww.out_start as usize, out_w - (ww.out_start as usize + aw));
            let weighted_full = weighted
                .pad_with_zeros(2, pad_top, pad_bottom)?
                .pad_with_zeros(3, pad_left, pad_right)?;
            let blend_full = blend
                .pad_with_zeros(2, pad_top, pad_bottom)?
                .pad_with_zeros(3, pad_left, pad_right)?;

            output = Some(match output {
                None => weighted_full,
                Some(acc) => (acc + weighted_full)?,
            });
            weights = Some(match weights {
                None => blend_full,
                Some(acc) => (acc + blend_full)?,
            });
        }
    }

    let output = output.ok_or_else(|| CandleError::Msg("vae tiling produced no tiles".into()))?;
    let weights = weights.ok_or_else(|| CandleError::Msg("vae tiling produced no tiles".into()))?;
    // Normalize by the summed blend weight (floored to avoid a divide-by-zero at any gap; the plan's
    // coverage invariant guarantees weights > 0 everywhere, so the floor never actually engages).
    Ok(output.broadcast_div(&weights.clamp(1e-8f32, f32::MAX)?)?)
}

/// The canonical SDXL UNet config — a verbatim copy of the `unet` block candle's
/// `StableDiffusionConfig::sdxl_` builds (pinned to the workspace candle rev; values from
/// `stabilityai/stable-diffusion-xl-base-1.0/unet/config.json`). `build_unet` reads that config from a
/// **private** field and only accepts a file path, so the adapter-merge path — which must feed a
/// merged in-memory tensor map — reconstructs it here. The no-adapter path still goes through
/// `build_unet`, so this is exercised only when adapters are present; the conformance lane (sc-5165
/// verify) renders an adapted image, which would NaN/garble if these drifted from the stock config.
fn sdxl_stock_unet_config() -> UNet2DConditionModelConfig {
    let bc = |out_channels, use_cross_attn, attention_head_dim| BlockConfig {
        out_channels,
        use_cross_attn,
        attention_head_dim,
    };
    UNet2DConditionModelConfig {
        blocks: vec![
            bc(320, None, 5),
            bc(640, Some(2), 10),
            bc(1280, Some(10), 20),
        ],
        center_input_sample: false,
        cross_attention_dim: 2048,
        downsample_padding: 1,
        flip_sin_to_cos: true,
        freq_shift: 0.,
        layers_per_block: 2,
        mid_block_scale_factor: 1.,
        norm_eps: 1e-5,
        norm_num_groups: 32,
        // The pipeline builds `StableDiffusionConfig::sdxl(None, …)`, i.e. unsliced attention.
        sliced_attention_size: None,
        use_linear_projection: true,
    }
}

/// Detect a **packed** MLX-tier CLIP encoder `which` in the snapshot at `root` (sc-9527, sc-9089j
/// follow-up to sc-9416): `Some((packed_weight_file, group_size))` when
/// `text_encoder{,_2}/config.json` carries a `quantization` block ([`PackedConfig`]) AND the packed
/// weight file (`model.safetensors`, not the dense `.fp16` name) exists, else `None` — a dense
/// diffusers snapshot loads through the stock builder unchanged. A missing config (e.g. a bare
/// single-file checkpoint) is treated as dense; the downstream loader gives the precise "missing X"
/// error. `group_size` is threaded from the config (defaulting to 64 via [`PackedConfig`], never
/// silent-dense — the sc-9410 rule) into the vendored CLIP's Linear seam.
pub(crate) fn detect_packed_clip(root: &Path, which: &Clip) -> Result<Option<(PathBuf, usize)>> {
    let dir = which.subdir();
    let cfg_path = root.join(dir).join("config.json");
    if !cfg_path.is_file() {
        return Ok(None);
    }
    let bytes = std::fs::read(&cfg_path)
        .map_err(|e| CandleError::Msg(format!("sdxl: read {dir}/config.json: {e}")))?;
    let cfg: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|e| CandleError::Msg(format!("sdxl: parse {dir}/config.json: {e}")))?;
    let Some(packed) = PackedConfig::from_config(&cfg) else {
        return Ok(None);
    };
    let file = snapshot_file(root, &format!("{dir}/model.safetensors"))?;
    Ok(Some((file, packed.group_size as usize)))
}

/// Detect a pre-quantized MLX SDXL tier at `root` (sc-9416): `Some((unet_file, group_size))` when
/// `unet/config.json` carries a `quantization` block ([`PackedConfig`]) and the packed weight file
/// (`diffusion_pytorch_model.safetensors`, not the dense `.fp16` name) exists, else `None` (a dense
/// diffusers snapshot — the stock/dense build). Errors on a packed tier whose group size the vendored
/// UNet's Linear seam does not thread (only 64 today) rather than silently repacking at the wrong grid.
///
/// Shared by the base txt2img load ([`Pipeline::load_components`], via the [`Pipeline::detect_packed_unet`]
/// method wrapper) AND the InstantID/edit/IP-Adapter vendored-UNet loader ([`crate::loaders::load_instantid_unet`],
/// sc-10813) — both take the packed vs dense fork from the SAME `unet/config.json` probe so a q4/q8 tier
/// serves the edit / inpaint / IP-Adapter lanes, not just plain txt2img.
pub(crate) fn detect_packed_unet(root: &Path) -> Result<Option<(PathBuf, usize)>> {
    let cfg_path = root.join("unet/config.json");
    if !cfg_path.is_file() {
        return Ok(None);
    }
    let bytes = std::fs::read(&cfg_path)
        .map_err(|e| CandleError::Msg(format!("sdxl: read unet/config.json: {e}")))?;
    let cfg: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|e| CandleError::Msg(format!("sdxl: parse unet/config.json: {e}")))?;
    let Some(packed) = PackedConfig::from_config(&cfg) else {
        return Ok(None);
    };
    let group_size = packed.group_size as usize;
    // The vendored UNet's Linear seam threads the default MLX group size (64) through its leaf
    // constructors; a non-64 tier would repack on the wrong grid, so refuse it loudly. The SDXL MLX
    // tiers all pack at 64, so this never fires on a real tier.
    if group_size != MLX_GROUP_SIZE {
        // sc-9528 kept this loud reject: the vendored UNet's top-level `new` → blocks → leaves chain
        // threads only the default group 64 (the leaf `*_gs` constructors exist, but wiring a non-64
        // group through the many nested block constructors is the same infeasibility lens/sd3 hit in
        // sc-9474). A non-64 SDXL MLX tier does not exist today; refuse it rather than repack on the
        // wrong grid. The adapter fold ([`crate::packed_adapters`]) asserts gs==64 for the same reason.
        return Err(CandleError::Msg(format!(
            "sdxl: packed tier group_size {group_size} unsupported (only {MLX_GROUP_SIZE}); \
             a non-64 SDXL tier needs the group threaded through the UNet blocks (sc-9528)"
        )));
    }
    let file = snapshot_file(root, "unet/diffusion_pytorch_model.safetensors")?;
    Ok(Some((file, group_size)))
}

/// Resolve a component file inside the SDXL snapshot dir, erroring clearly if absent (e.g. a
/// single-file RealVisXL checkpoint that lacks the diffusers multi-component tree — sc-3677).
pub(crate) fn snapshot_file(root: &Path, sub: &str) -> Result<PathBuf> {
    let p = root.join(sub);
    if !p.is_file() {
        return Err(CandleError::Msg(format!(
            "sdxl snapshot is missing {sub} (expected a diffusers multi-component snapshot at {})",
            root.display()
        )));
    }
    Ok(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// sc-9416: `detect_packed_unet` returns `Some((file, group_size))` for a snapshot whose
    /// `unet/config.json` carries a `quantization` block AND the packed weight file exists, and `None`
    /// for a dense snapshot (no block) — the packed/dense fork the base txt2img load takes. GPU-free.
    #[test]
    fn detect_packed_unet_reads_quantization_block() {
        let tmp = std::env::temp_dir().join(format!("sc9416_detect_{}", std::process::id()));
        let unet_dir = tmp.join("unet");
        std::fs::create_dir_all(&unet_dir).unwrap();
        // A packed config + a (stub) packed weight file at the non-.fp16 name.
        std::fs::write(
            unet_dir.join("config.json"),
            br#"{"quantization": {"bits": 4, "group_size": 64}, "cross_attention_dim": 2048}"#,
        )
        .unwrap();
        std::fs::write(
            unet_dir.join("diffusion_pytorch_model.safetensors"),
            b"stub",
        )
        .unwrap();

        let pipe = Pipeline {
            config: StableDiffusionConfig::sdxl(None, Some(1024), Some(1024)),
            root: tmp.clone(),
            device: Device::Cpu,
            dtype: DType::F32,
            adapters: vec![],
            pid_spec: None,
        };
        let got = pipe.detect_packed_unet().unwrap();
        assert!(got.is_some(), "a quantization block ⇒ packed tier");
        assert_eq!(got.unwrap().1, 64, "group_size threaded from config");

        // A dense config (no quantization block) ⇒ None (the stock build).
        std::fs::write(
            unet_dir.join("config.json"),
            br#"{"cross_attention_dim": 2048, "sample_size": 128}"#,
        )
        .unwrap();
        assert!(
            pipe.detect_packed_unet().unwrap().is_none(),
            "no quantization block ⇒ dense (stock) build"
        );

        // A bits-only block still packs (group defaults to 64, not silent-dense — the sc-9410 rule).
        std::fs::write(
            unet_dir.join("config.json"),
            br#"{"quantization": {"bits": 8}}"#,
        )
        .unwrap();
        assert_eq!(
            pipe.detect_packed_unet().unwrap().map(|(_, g)| g),
            Some(64),
            "bits-only ⇒ packed at the default group 64"
        );

        std::fs::remove_dir_all(&tmp).ok();
    }

    /// A packed tier whose group size is not the seam's threaded 64 is rejected loudly (sc-9416 /
    /// sc-9528) rather than silently repacking on the wrong grid.
    #[test]
    fn detect_packed_unet_rejects_non_64_group() {
        let tmp = std::env::temp_dir().join(format!("sc9416_detect_g32_{}", std::process::id()));
        let unet_dir = tmp.join("unet");
        std::fs::create_dir_all(&unet_dir).unwrap();
        std::fs::write(
            unet_dir.join("config.json"),
            br#"{"quantization": {"bits": 4, "group_size": 32}}"#,
        )
        .unwrap();
        std::fs::write(
            unet_dir.join("diffusion_pytorch_model.safetensors"),
            b"stub",
        )
        .unwrap();
        let pipe = Pipeline {
            config: StableDiffusionConfig::sdxl(None, Some(1024), Some(1024)),
            root: tmp.clone(),
            device: Device::Cpu,
            dtype: DType::F32,
            adapters: vec![],
            pid_spec: None,
        };
        assert!(
            pipe.detect_packed_unet().is_err(),
            "a non-64 group_size must be rejected, not silently mis-repacked"
        );
        std::fs::remove_dir_all(&tmp).ok();
    }

    /// sc-10826: an **omitted** sampler must resolve to the curated `ddim` solver — the native
    /// candle-transformers `DDIMScheduler` inference loop (which rendered a ghosted, translucent
    /// double-exposure on the default path) is removed, so the default now runs the same unified
    /// curated framework every named sampler does. `lightning` keeps its own path (`None`), and an
    /// explicit curated name passes through unchanged. This pins the routing rule the ghost fix hinges
    /// on, GPU-free — the human-eyeball coherence check is the `realvisxl_lightning` GPU smoke with
    /// `RVXL_SAMPLER=` (engine default) + real CFG.
    #[test]
    fn omitted_sampler_routes_to_curated_ddim_not_native() {
        assert_eq!(
            resolve_sampler(None),
            Some("ddim"),
            "omitted ⇒ curated ddim"
        );
        assert_eq!(
            resolve_sampler(Some("ddim")),
            Some("ddim"),
            "ddim ⇒ curated ddim"
        );
        assert_eq!(
            resolve_sampler(Some("dpmpp_2m")),
            Some("dpmpp_2m"),
            "an explicit curated name passes through"
        );
        assert_eq!(
            resolve_sampler(Some(LIGHTNING_SAMPLER)),
            None,
            "lightning takes its own few-step path, not the curated framework"
        );
        // The default is a genuinely-advertised curated solver, so it never silently euler-falls-back
        // or targets a removed native path.
        assert_eq!(DEFAULT_SAMPLER, "ddim");
        assert!(
            candle_gen::curated_sampler_names().contains(&DEFAULT_SAMPLER),
            "DEFAULT_SAMPLER must be in the advertised curated menu"
        );
    }

    /// sc-3677 parity: the production txt2img values the candle lane resolves an omitted field to
    /// must match the SceneWorks `SdxlDiffusersAdapter` reference (30 steps, CFG 7.0), and the
    /// VAE un-scale must be the diffusers-correct SDXL `scaling_factor` (0.13025 — NOT candle's
    /// hardcoded SD1.5 0.18215). `sdxl` and `realvisxl` map to this one engine, so this pins the
    /// shared default surface both ids inherit. GPU-free (asserts the constants directly).
    #[test]
    fn parity_defaults_match_diffusers_adapter() {
        assert_eq!(DEFAULT_STEPS, 30);
        // float consts: compare with an epsilon (clippy's float_cmp would reject `==`).
        assert!((DEFAULT_GUIDANCE - 7.0).abs() < f64::EPSILON);
        assert!((VAE_SCALE - 0.13025).abs() < f64::EPSILON);
    }

    /// sc-9013 / F-029: every runtime `hf-hub` download on the SDXL render path must be pinned to an
    /// immutable commit SHA — never the mutable `main` default. Assert each of the three render-path
    /// repos (the fp16-fix VAE + both CLIP tokenizers) has a pin that is a 40-char lowercase hex SHA
    /// (not `"main"`/a branch), and that `hub_revision` resolves each; an unpinned repo is rejected.
    #[test]
    fn render_path_hub_downloads_are_pinned_to_immutable_shas() {
        let render_repos = [VAE_FIX_REPO, Clip::L.sources().0, Clip::BigG.sources().0];
        for repo in render_repos {
            let rev = hub_revision(repo).expect("render-path repo must be pinned");
            assert_ne!(
                rev, "main",
                "{repo} is pinned to the mutable default revision"
            );
            assert_eq!(
                rev.len(),
                40,
                "{repo} pin is not a 40-char commit SHA: {rev:?}"
            );
            assert!(
                rev.chars()
                    .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
                "{repo} pin is not a lowercase hex SHA: {rev:?}"
            );
        }
        // An unknown repo must be refused, not silently resolved against `main`.
        assert!(hub_revision("some/unpinned-repo").is_err());
    }

    /// sc-5165: the adapter-merge path reconstructs the SDXL UNet config locally (since `build_unet`
    /// reads it from a private field of `StableDiffusionConfig`). Pin its canonical values so an
    /// accidental edit — which would silently build the adapter-path UNet differently from the
    /// no-adapter `build_unet` path — is caught here, not only on the GPU conformance lane.
    #[test]
    fn sdxl_stock_unet_config_pins_canonical_values() {
        let c = sdxl_stock_unet_config();
        assert_eq!(c.blocks.len(), 3);
        assert_eq!(c.blocks[0].out_channels, 320);
        assert_eq!(c.blocks[0].use_cross_attn, None);
        assert_eq!(c.blocks[0].attention_head_dim, 5);
        assert_eq!(c.blocks[1].out_channels, 640);
        assert_eq!(c.blocks[1].use_cross_attn, Some(2));
        assert_eq!(c.blocks[1].attention_head_dim, 10);
        assert_eq!(c.blocks[2].out_channels, 1280);
        assert_eq!(c.blocks[2].use_cross_attn, Some(10));
        assert_eq!(c.blocks[2].attention_head_dim, 20);
        assert_eq!(c.cross_attention_dim, 2048);
        assert_eq!(c.layers_per_block, 2);
        assert_eq!(c.norm_num_groups, 32);
        assert_eq!(c.downsample_padding, 1);
        assert!(c.use_linear_projection);
        assert!(!c.center_input_sample);
        assert!(c.flip_sin_to_cos);
        assert_eq!(c.sliced_attention_size, None);
    }

    /// sc-6128: the Lightning policy is diffusers `EulerDiscreteScheduler(timestep_spacing="trailing",
    /// final_sigmas_type="zero")` built from the SDXL `scaled_linear` betas. Pin the trailing timesteps
    /// (the hand-computable `round(arange(N, 0, −N/steps)) − 1`), the σ-max prior scale, and the
    /// final-step zero-σ landing — the candle wrapper of the gen-core policy (no GPU/weights).
    #[test]
    fn lightning_policy_is_trailing_euler_with_zero_final() {
        let p = lightning_policy(5).unwrap();
        assert_eq!(p.num_steps(), 5);
        // Trailing spacing for 5 steps over 1000 train timesteps: round([1000,800,600,400,200]) − 1.
        let ts: Vec<f32> = (0..5).map(|i| p.coeffs(i).timestep).collect();
        assert_eq!(ts, vec![999.0, 799.0, 599.0, 399.0, 199.0]);
        // init_noise_scale = the largest σ (σ at the near-train-end first step) — well above 1 for SDXL.
        assert!(
            p.init_noise_scale() > 10.0,
            "σ_max should be the large trailing σ, got {}",
            p.init_noise_scale()
        );
        // c_in = 1/√(σ²+1) ∈ (0, 1] and the conditioning timestep descends across the schedule.
        let c0 = p.coeffs(0);
        assert!(c0.c_in > 0.0 && c0.c_in <= 1.0);
        assert!(c0.timestep > p.coeffs(4).timestep);
        // `final_sigmas_type="zero"`: the last step's σ_{i+1} is 0, so a_out = 0 − σ_last < 0 — the
        // step drives the latent the rest of the way to the clean sample.
        assert!(
            p.coeffs(4).a_out < 0.0,
            "final a_out should bring σ→0, got {}",
            p.coeffs(4).a_out
        );
        // The deterministic Euler step injects no noise.
        assert!((0..5).all(|i| p.coeffs(i).a_noise == 0.0));
    }

    /// sc-6128: the policy guards a degenerate 0-step request (the real `steps>=1` floor is the
    /// generator's `validate`), so `lightning_policy(0)` still yields a usable 1-step schedule rather
    /// than panicking on a `/0`.
    #[test]
    fn lightning_policy_clamps_zero_steps() {
        assert_eq!(lightning_policy(0).unwrap().num_steps(), 1);
    }

    /// The tiled blend (slice → mask → pad → accumulate → normalize) must exactly reconstruct the
    /// input under an **identity** decode at spatial-scale 1 — every output position is
    /// `Σ(maskᵢ·xᵢ) / Σ maskᵢ = x`, regardless of the (overlapping) trapezoidal mask values. This
    /// covers the candle accumulation math on CPU without a GPU/VAE; the per-axis tiling geometry
    /// itself is unit-tested in `gen_core::tiling`.
    #[test]
    fn tile_blend_identity_roundtrip() {
        let device = Device::Cpu;
        // 1×1 spatial scale so out dims == latent dims and an identity decode is shape-preserving.
        let vae = VaeTiling {
            spatial_scale: 1,
            temporal_scale: 1,
            causal_temporal: false,
        };
        // A small grid with overlapping tiles: 4-wide tiles, 2 overlap, over a 10×10 field → 4 tiles
        // per axis, exercising left/right ramps and the interior all-ones region.
        let cfg = TilingConfig::spatial_only(4, 2);
        let (h, w) = (10usize, 10usize);
        let vals: Vec<f32> = (0..(h * w) as i64).map(|i| i as f32).collect();
        let input = Tensor::from_vec(vals.clone(), (1, 1, h, w), &device).unwrap();

        // Sanity: tiling actually fires for this config/size.
        assert!(cfg.needs_tiling(vae, 1, h as i32, w as i32));

        let out = tile_blend_decode(&input, vae, &cfg, |tile| Ok(tile.clone())).unwrap();
        assert_eq!(out.dims4().unwrap(), (1, 1, h, w));
        let got = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (g, e) in got.iter().zip(vals.iter()) {
            assert!((g - e).abs() < 1e-4, "blend reconstruction off: {g} vs {e}");
        }
    }

    /// Below the tiling threshold (a 64² latent → 512² output, the conformance render size) the plan
    /// produces a **single** tile, so the tiled path is a no-op pass-through identical to a monolithic
    /// decode — the guarantee that 512² output is unchanged by sc-4987.
    #[test]
    fn no_tiling_below_threshold() {
        let cfg = sdxl_tiling_config();
        // 64² latent = 512² output: not > the 64-latent tile, so tiling must NOT fire.
        assert!(!cfg.needs_tiling(SDXL_VAE_TILING, 1, 64, 64));
        // 128² latent = 1024² output: must fire.
        assert!(cfg.needs_tiling(SDXL_VAE_TILING, 1, 128, 128));
    }

    /// F-061 / sc-9045: the bespoke `denoise::decode_image` (trainer preview, IP / edit providers) and
    /// the registered `Pipeline::decode` now share the single [`tiled_vae_decode`] seam. This asserts
    /// the seam's gate is a pure function of the tiling flag + latent size — so both callers make the
    /// **same** tiled-vs-monolithic decision at identical resolutions. Combined with
    /// `tile_blend_identity_roundtrip` (tiling is exact for an identity decode) and
    /// `no_tiling_below_threshold` (≤512² stays monolithic ⇒ byte-identical to a bare decode), the two
    /// SDXL lanes are guaranteed the same output for in-memory cases and the same bounded peak on large
    /// latents. A real-VAE decode-parity check runs on the GPU conformance lane (no CPU VAE fixture).
    #[test]
    fn tiled_decode_gate_is_shared_and_size_driven() {
        let cfg = sdxl_tiling_config();
        // The decision `tiled_vae_decode` makes for a given latent is `enabled && needs_tiling`.
        // With the flag off, no latent tiles (registered + bespoke both decode monolithically).
        let gate =
            |enabled: bool, h: i32, w: i32| enabled && cfg.needs_tiling(SDXL_VAE_TILING, 1, h, w);
        assert!(
            !gate(false, 128, 128),
            "flag off ⇒ never tile (monolithic, byte-identical)"
        );
        assert!(
            !gate(true, 64, 64),
            "512² output ⇒ single tile ⇒ monolithic"
        );
        assert!(
            gate(true, 128, 128),
            "1024² output ⇒ tiled (bounded peak) on both lanes"
        );
    }
}
