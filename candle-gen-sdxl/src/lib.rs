//! # candle-gen-sdxl
//!
//! The **Stable Diffusion XL** provider crate for [`candle-gen`](candle_gen) — the candle
//! (Windows/CUDA) sibling of `mlx-gen-sdxl`. It implements the backend-neutral
//! [`gen_core::Generator`] contract and self-registers via `inventory`, so linking this crate
//! makes `gen_core::load("sdxl", …)` resolve the candle SDXL generator.
//!
//! **txt2img (sc-3675 + sc-3673):** [`SdxlGenerator::generate`] runs the GO-validated epic-3494
//! prototype ([`pipeline`]) through the contract: dual CLIP → UNet (real CFG) → f16 VAE, emitting
//! `Progress` and honoring `req.cancel`, with **deterministic CPU-seeded noise + the non-ancestral
//! DDIM sampler** (sc-3673) so output is launch-portable per seed.
//!
//! **LoRA/LoKr (sc-5165):** [`load`] accepts `spec.adapters` and merges a trained adapter's delta into
//! the UNet weights at component load ([`adapters`] + [`pipeline`]) — the inference half of the native
//! candle trainer, closing the train→infer loop. The descriptor advertises the wired surface
//! (txt2img, negative prompt, guidance, `ddim`, the few-step `lightning` sampler, **LoRA/LoKr**) — NOT
//! the full mlx-gen-sdxl conditioning / accel-sampler surface — so the worker routes the rest to the
//! Python fallback (sc-3678) rather than the candle backend silently dropping a control. The
//! descriptor's `backend` is `"candle"` and `mac_only` is `false` (Windows/CUDA target).
//!
//! **Lightning (sc-6128):** a `req.sampler == "lightning"` request runs the few-step Euler-trailing
//! denoise ([`pipeline`]) — diffusers `EulerDiscreteScheduler(timestep_spacing="trailing")`, ε-pred,
//! `final_sigmas_type="zero"`, **CFG-off** — reusing the backend-neutral `gen_core::sampling`
//! `LightningPolicy` (the same schedule `mlx-gen-sdxl`'s `LightningSampler` drives). This makes the
//! `realvisxl_lightning` model id renderable on the candle (Windows) lane at ~5 steps; base SDXL is
//! unaffected (it keeps the DDIM default).
//!
//! Perf (sc-3674): CLIP loads f16 and the UNet attention runs through fused **flash-attention** when
//! built `--features flash-attn` and the runtime toggle ([`set_flash_attn`], default on) is set.
//!
//! Peak VRAM (sc-4987): the dual CLIP is loaded/run/freed before the UNet+VAE load (staged
//! sequential load), and the VAE decode tiles + blends above 512² output ([`set_vae_tiling`], default
//! on) — together targeting torch-parity peak VRAM at 1024².
//!
//! Component caching across `generate` calls (sc-5037 — a latency win, in tension with sc-4987's
//! mid-call frees) is wired. **RealVisXL + parity (sc-3677):** RealVisXL_V5.0 ships the standard
//! diffusers tree with the *same* `.fp16.safetensors` component filenames as SDXL-base, so it loads
//! through this identical path unmodified (no single-file loader needed); parity with the Python
//! `SdxlDiffusersAdapter` is locked by the CPU parity tests here + `tests/conformance.rs`
//! (`sdxl_conformance` / `realvisxl_conformance` on the CUDA lane). See [`pipeline`] for the layout
//! finding and the one accepted sampler difference (DDIM vs euler_ancestral, sc-3673).

mod pipeline;
// The PiD backbone (latent-space) tag (epic 7840 / sc-8373), re-exported so `candle-gen-instantid`
// loads the same `sdxl` student through its own `with_pid` (it composes the SDXL VAE).
pub use pipeline::PID_BACKBONE;

// The vendored, packed-detecting SDXL CLIP text-encoder tower (sc-9527, sc-9089j follow-up to the
// sc-9416 UNet packed-load): its Linear surface (attn q/k/v/out_proj, MLP fc1/fc2, bigG
// text_projection) routes through `candle_gen::quant` so the packed MLX tier's dual CLIP loads
// straight from the packed parts. A pure superset — a dense snapshot takes the stock path unchanged.
pub mod clip;

// Inference-side LoRA/LoKr adapter merge (sc-5165) — folds a trained adapter's delta into the dense
// UNet weights before the stock UNet is built (`pipeline` calls this on the adapter path). The candle
// twin of `mlx-gen-sdxl::adapters`; closes the train→infer loop with the trainer below.
mod adapters;
// The merge entry point + its report are public: the worker can introspect what merged (the candle
// analog of `mlx-gen-sdxl::apply_sdxl_adapters`'s report), and the trainer-verify lane
// (`tests/trainer_e2e.rs`) asserts a trained adapter merges into every target with nothing skipped.
pub use adapters::{merge_adapters, AdditiveReport, MergeReport};
// Adapters on a **packed** (pre-quantized MLX tier) UNet (sc-11103, epic 10765): the distill LoRA rides
// the packed Linears **additively** (`adapters::install_additive`, no dequant — the q4/q8 footprint
// survives) and any conv LoRA folds into the dense convs (`adapters::fold_conv_adapters`). This retired
// the sc-9528 dequant→fold→keep-dense path (`packed_adapters.rs`), which dequantized the FF (the bulk of
// the UNet) and defeated the point of the packed tier for SDXL-Lightning / RealVisXL-Lightning.

// IP-Adapter (sc-5491, epic 5480): the perceiver `Resampler` (`image_proj.*` → image/identity tokens)
// + the decoupled cross-attn K/V pairs (`ip_adapter.*`), the candle twin of `mlx-gen-sdxl::ip_adapter`.
// Built here (not in the InstantID glue crate) so the SDXL IP-Adapter-Plus path (sc-5488) reuses them.
// sc-5488 adds the CLIP-ViT image preprocessing + the `IpImageEncoder` (CLIP image encoder → Resampler).
pub mod ip_adapter;
// CLIP ViT vision tower (sc-5488) — the IP-Adapter image encoder (ViT-H/14 for SDXL, ViT-L/14-336 for
// Kolors), the one net-new model the general IP-Adapter port needs (candle-gen had only the text CLIP).
pub mod vision_encoder;
pub use vision_encoder::{ClipVisionEncoder, VisionConfig};
// The safetensors key→Tensor map for the IP-Adapter / ControlNet loads (non-VarBuilder weights).
// Hoisted to the `candle-gen` core crate (F-060, sc-9044); re-exported here for source compatibility.
pub mod weights;

// Euler / Euler-ancestral sampler (sc-5491) — the InstantID/diffusers-SDXL solver the InstantID
// denoise loop runs (the txt2img `pipeline` runs the unified curated framework: DDIM eta=0 by
// default, sc-10826). Port of `mlx-gen-sdxl::sampler::EulerSampler`, with the ancestral noise
// injected (the loop owns the seeded RNG, sc-3673) and the schedule scalars in host f64 (no Python
// parity to hold ULP-for-ULP).
pub mod sampler;
pub use sampler::EulerAncestralSampler;

// InstantID denoise loop + the SDXL conditioning/prior/control/decode helpers (sc-5491) — the candle
// twin of `mlx-gen-sdxl::pipeline`'s `denoise_ip_control` family, composing the IP-Adapter UNet, the
// IdentityNet ControlNet, and the euler-ancestral sampler. Driven by the `candle-gen-instantid` glue.
pub mod denoise;
pub use denoise::{
    decode_image, denoise_curated, denoise_ip_control, denoise_ip_multi_control,
    preprocess_control_image, seeded_prior, seeded_sigma_prior, text_time_ids, ControlContext,
    Denoiser,
};

// SDXL dual-CLIP conditioning (sc-5491) — penultimate hidden (cross-attn) + pooled text-embeds
// (add_embedding), the micro-conditioning the txt2img `pipeline` skips but `forward_instantid` needs.
pub mod conditioning;
pub use conditioning::SdxlConditioner;

// SDXL component loaders for InstantID (sc-5491) — the vendored UNet (+ add_embedding), the fp16-fix
// VAE, and a diffusers ControlNet, built from an SDXL snapshot. The candle twins of mlx-gen-sdxl's
// load_unet_dtype/load_vae/load_controlnet.
pub mod loaders;
pub use loaders::{
    load_instantid_unet, load_instantid_unet_with_adapters, load_sdxl_controlnet, load_sdxl_vae,
};

// The SDXL VAE type the loader returns, re-exported so the `candle-gen-instantid` glue can hold one as
// a field + pass it to `decode_image` without depending on candle-transformers directly.
pub use candle_transformers::models::stable_diffusion::vae::AutoEncoderKL;

// The vendored UNet itself, re-exported so the `candle-gen-instantid` glue can hold one + drive its
// InstantID surface (install_ip_adapter / set_ip_context / forward_instantid via the denoise loop).
// `sdxl_unet_config` + `UNet2DConditionModelConfig`/`BlockConfig` are re-exported too so the Kolors
// IP-Adapter provider (sc-5488) can build the same vendored stack from the SDXL-family Kolors UNet.
pub use unet::{sdxl_unet_config, BlockConfig, UNet2DConditionModel, UNet2DConditionModelConfig};

// SDXL IP-Adapter-Plus reference-image provider (sc-5488, epic 5480) — the [`ip_adapter`] +
// [`denoise`] stack composed without a face embedder / ControlNet: CLIP ViT-H image tokens → pure-IP
// denoise. The reference-conditioning sibling of the InstantID glue crate, but for plain SDXL/RealVisXL.
pub mod ip_provider;
pub use ip_provider::{
    IpAdapterSdxl, IpAdapterSdxlPaths, IpAdapterSdxlRequest, DEFAULT_IP_ADAPTER_SCALE,
};

/// SDXL IP-Adapter-Plus real-weight GPU validation (sc-5488) — env-driven, `#[ignore]`d integration
/// test (the analog of the InstantID Phase-5 harness).
#[cfg(test)]
mod ip_validate;

// SDXL img2img / inpaint / outpaint edit provider (sc-6037, epic 5480) — pixel-conditioned editing,
// the candle twin of the `mlx-gen-sdxl` edit path and the provider half that unblocks the worker
// img2img/edit/inpaint routing (sc-5487). Reuses the IP/InstantID denoise stack with the IP branch
// inert (no install) + the deterministic VAE moments-encoder init + a per-step inpaint mask blend.
pub mod edit_provider;
pub use edit_provider::{
    SdxlEdit, SdxlEditPaths, SdxlEditRequest, DEFAULT_EDIT_STRENGTH, DEFAULT_INPAINT_STRENGTH,
};

/// SDXL edit (img2img / inpaint / outpaint) real-weight GPU validation (sc-6037) — env-driven,
/// `#[ignore]`d integration test (the analog of the IP-Adapter Phase-5 harness).
#[cfg(test)]
mod edit_validate;

// Vendored, training-adapted SDXL UNet + VAE-encode stack (sc-5165) — used by the native LoRA/LoKr
// trainer below. Inference continues to use the stock candle-transformers UNet via `pipeline`; the
// vendored copy retains some unused upstream surface (decoder blocks, the additional-residuals
// forward), hence `allow(dead_code)`.
#[allow(dead_code)]
mod unet;

// The InstantID inference surface built on the vendored UNet (sc-5491, epic 5480): the SDXL ControlNet
// branch (the IdentityNet) — an encoder copy + conditioning embedding + zero-conv heads producing the
// scaled down/mid residuals the InstantID UNet adds in. Re-exported so the `candle-gen-instantid` glue
// crate composes it; also the SDXL ControlNet building block sc-5489 reuses.
pub use unet::{ControlNet, ControlNetConfig, ControlResiduals};

// The native candle SDXL LoRA/LoKr trainer (sc-5165) — implements `gen_core::Trainer` and
// self-registers via `inventory` (kept linked by `force_link`), so `gen_core::load_trainer("sdxl", …)`
// resolves the candle trainer.
mod training;
pub use training::{load_trainer, trainer_descriptor, SdxlTrainer};

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::{DType, Device};
use candle_gen::gen_core::{
    self, AdapterSpec, Capabilities, GenerationOutput, GenerationRequest, Generator, LoadSpec,
    Modality, ModelDescriptor, PidWeights, Progress, Quant, WeightsSource,
};

use pipeline::{Components, Pipeline};

/// Registry id — matches the SceneWorks worker's `payload.model` (`MODEL_TABLE["sdxl"]`). The
/// worker maps both `sdxl` and `realvisxl` onto engine id `"sdxl"`, so — exactly like
/// `mlx-gen-sdxl` — this crate registers a SINGLE descriptor under `"sdxl"`.
pub const MODEL_ID: &str = "sdxl";

/// SDXL works in latent space at /8: both dims must be multiples of 8.
const SIZE_MULTIPLE: u32 = 8;

/// Process-global flash-attention runtime toggle (sc-3674). This switch was designed to decide
/// whether a flash-attn-capable build actually *uses* the fused kernels, so the SceneWorks UI can
/// expose it (defaulted on) and the worker flips it from settings — without recompiling. Mirrors
/// `mlx-gen-sdxl::set_compile_glue`. **sc-9032:** the `flash-attn` cargo feature it was ANDed with
/// was a no-op alias (`= ["cuda"]`, no `candle-flash-attn` wired) and was removed; `components` now
/// hard-codes the flash path off, so this toggle is retained as public worker API but is inert until
/// the fused-kernel slice lands and re-gates the load on it. Default **on**.
static FLASH_ATTN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Enable/disable flash-attention for subsequently-loaded pipelines (sc-3674). Process-global; the
/// worker calls this from its `backend_candle`/flash setting at startup. Inert since sc-9032 removed
/// the no-op `flash-attn` feature — no fused kernels are compiled in (retained as worker API).
pub fn set_flash_attn(on: bool) {
    FLASH_ATTN.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Whether flash-attention is currently enabled (the runtime toggle, [`set_flash_attn`]). Since
/// sc-9032 the pipeline hard-codes the flash path off (the no-op `flash-attn` feature was removed),
/// so this returning `true` does not enable anything.
pub fn flash_attn_enabled() -> bool {
    FLASH_ATTN.load(std::sync::atomic::Ordering::Relaxed)
}

/// Process-global VAE-tiling runtime toggle (sc-4987). When on, the VAE decode tiles the latent into
/// overlapping 64²-latent (512²-output) tiles and trapezoidally blends the seams — bounding the
/// decode's peak VRAM to one tile (the tallest single allocation at 1024², for torch-parity). Unlike
/// flash-attn there is no build feature: it is pure candle, so the switch alone decides. It only
/// *fires* above 512² output (smaller renders stay monolithic), so leaving it on is free at/below
/// 512². The SceneWorks worker/UI drives it; default **on** to hit the <12 GiB target out of the box.
static VAE_TILING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Enable/disable VAE tiling for subsequent decodes (sc-4987). Process-global; the worker drives it
/// from its backend setting. Off restores the monolithic single-pass decode (higher peak VRAM).
pub fn set_vae_tiling(on: bool) {
    VAE_TILING.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Whether VAE tiling is currently enabled (the runtime toggle, [`set_vae_tiling`]). The pipeline
/// additionally only tiles when the output exceeds the 512² threshold, so this returning `true` does
/// not change ≤512² output.
pub fn vae_tiling_enabled() -> bool {
    VAE_TILING.load(std::sync::atomic::Ordering::Relaxed)
}

/// A loaded candle SDXL generator. Loading is **lazy**: `load` does no file I/O (registry
/// introspection against a missing path still resolves), and the heavy UNet/VAE are built on the
/// first [`generate`](Generator::generate) call. sc-5037: those [`Components`] are then **cached** in
/// `components` and reused across subsequent calls (keyed by the flash-attn setting), so back-to-back
/// requests skip the ~7 GiB UNet/VAE disk re-read. CLIP is intentionally not cached — it stays
/// load-on-demand-and-free (the sc-4987 peak-VRAM lever), so the cache is a latency win that does not
/// raise the ~8.7 GiB high-water mark.
pub struct SdxlGenerator {
    descriptor: ModelDescriptor,
    root: PathBuf,
    device: Device,
    dtype: DType,
    /// LoRA/LoKr adapters to merge into the UNet weights at first-`generate` component load (sc-5165).
    /// Fixed for the generator's lifetime (from the `LoadSpec`), so they sit outside the component
    /// cache key. Empty ⇒ the stock no-adapter UNet load.
    adapters: Vec<AdapterSpec>,
    /// The `LoadSpec::pid` component captured at load (epic 7840 / sc-7853), threaded into the lazy
    /// component build so the PiD engine loads once alongside the UNet/VAE. `None` when not opted in.
    pid_spec: Option<PidWeights>,
    /// Cached UNet+VAE + the flash-attn flag they were built with. `Mutex` because `Generator` is
    /// shared and `generate` takes `&self`; the lock is held only to read/populate the cache (a
    /// cheap `Arc` clone or a one-time load), never across the denoise.
    components: Mutex<Option<(bool, Components)>>,
    /// Cached dual CLIP tokenizers, loaded+parsed once and reused across `generate` calls (sc-8991 /
    /// F-011) rather than re-reading `tokenizer.json` from the hf-hub cache on every text encode. Shared
    /// behind an `Arc` (model-agnostic); `lock_recover` mirrors the components-cache poison recovery.
    tokenizers: Mutex<Option<Arc<pipeline::SdxlTokenizers>>>,
}

impl SdxlGenerator {
    /// Get the cached UNet/VAE, loading (and caching) them on a miss. Keyed by the effective
    /// flash-attn setting (`build_unet` bakes it in, sc-3674), so flipping [`set_flash_attn`] between
    /// calls rebuilds rather than serving a stale UNet. The lock is held over the cache-miss load
    /// (concurrent first-callers serialize on it) but released before the caller's denoise.
    fn components(&self, pipe: &Pipeline) -> gen_core::Result<Components> {
        // sc-9032: the `flash-attn` cargo feature (a no-op alias for `cuda`) was removed. No fused
        // candle-flash-attn kernel is wired, so the flash path is never taken — `false` here is
        // byte-identical to the old `cfg!(feature = "flash-attn") && flash_attn_enabled()`, which
        // always resolved false in every buildable config. The `set_flash_attn` runtime toggle stays
        // (public worker API) but is inert until the fused-kernel slice lands and re-gates this.
        let flash = false;
        // sc-9015 / F-031: recover from a poisoned lock (overwrite-on-miss cache; a prior panic
        // while locked must not turn every later `generate` into a panic).
        let mut guard = candle_gen::lock_recover(&self.components);
        if let Some((cached_flash, comps)) = guard.as_ref() {
            if *cached_flash == flash {
                return Ok(comps.clone());
            }
        }
        let comps = pipe.load_components(flash)?;
        *guard = Some((flash, comps.clone()));
        Ok(comps)
    }

    /// Get the cached dual CLIP tokenizers, loading (and caching) them on a miss (sc-8991 / F-011). The
    /// tokenizers are model-agnostic (fixed hf-hub repos) so a single pair serves every request; parsing
    /// them once here removes the tens-of-ms `tokenizer.json` re-parse the per-encode load did. The
    /// shared [`candle_gen::cached`] read-through recovers a poisoned lock internally (the F-031 idiom).
    fn tokenizers(&self) -> gen_core::Result<Arc<pipeline::SdxlTokenizers>> {
        candle_gen::cached(&self.tokenizers, || {
            Ok(Arc::new(pipeline::SdxlTokenizers::load()?))
        })
    }
}

impl Generator for SdxlGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        // The shared capability floor (count/size range/guidance/negative/sampler/conditioning):
        // since the descriptor advertises NO conditioning, any conditioning entry is rejected here.
        self.descriptor
            .capabilities
            .validate_request(MODEL_ID, req)?;
        // Model-specific floor on top of the shared one (mirrors mlx-gen-sdxl::validate_request).
        if req.prompt.is_empty() {
            return Err(gen_core::Error::Msg(
                "sdxl: prompt must not be empty".into(),
            ));
        }
        // An explicit `steps: Some(0)` would VAE-decode pure scaled noise — reject loudly (a derived
        // 0 from img2img strength would be a legitimate no-op, but this is txt2img-only).
        if req.steps == Some(0) {
            return Err(gen_core::Error::Msg(
                "sdxl: steps must be >= 1 (an explicit 0 renders undenoised noise)".into(),
            ));
        }
        if !req.width.is_multiple_of(SIZE_MULTIPLE) || !req.height.is_multiple_of(SIZE_MULTIPLE) {
            return Err(gen_core::Error::Msg(format!(
                "sdxl: width/height must be multiples of {SIZE_MULTIPLE} (got {}x{})",
                req.width, req.height
            )));
        }
        // F-116: `lightning` (sc-6128) runs its own fixed Euler-trailing schedule and never consults
        // `req.scheduler` — so a caller-selected NON-DEFAULT curated scheduler (karras / sgm_uniform /
        // …) would be silently dropped (the F-004 shape). Reject that combination loudly instead of
        // misleading. Only a recognized non-default scheduler is rejected; the default `normal`, `None`,
        // and any unrecognized value (e.g. the native-fallback `discrete` alias) all pass — a worker
        // that always populates `scheduler:"normal"` alongside `lightning` must not hard-error, and
        // every one of these resolves to lightning's own trailing schedule anyway (nothing dropped).
        if req.sampler.as_deref() == Some(pipeline::LIGHTNING_SAMPLER) {
            if let Some(sched) = req.scheduler.as_deref() {
                let recognized = gen_core::sampling::Scheduler::from_name(sched);
                if recognized.is_some_and(|s| s != gen_core::sampling::Scheduler::Normal) {
                    return Err(gen_core::Error::Msg(format!(
                        "sdxl: the `lightning` sampler uses its own fixed trailing schedule and \
                         ignores the `scheduler` axis (got `{sched}`) — omit `scheduler` (or use \
                         `normal`) for lightning"
                    )));
                }
            }
        }
        Ok(())
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.validate(req)?;
        // The rich-`CandleError` tail — including the typed `Canceled` — bridges into
        // `gen_core::Error` via `?` (the From bridge). The light `Pipeline` handle carries this
        // request's latent dims; the heavy UNet/VAE come from the cache.
        let pipe = Pipeline::load(
            &self.root,
            &self.device,
            self.dtype,
            req.width,
            req.height,
            &self.adapters,
            self.pid_spec.clone(),
        )?;
        // Encode text FIRST (loads + frees CLIP) so the cold-call ordering — CLIP gone before the
        // UNet/VAE are resident — is preserved (sc-4987); only then acquire the cached UNet/VAE
        // (sc-5037). On a warm call the UNet/VAE are already resident, but CLIP loads one encoder at a
        // time, so the footprint stays under the denoise-time peak.
        let negative = req.negative_prompt.as_deref().unwrap_or("");
        let tokenizers = self.tokenizers()?;
        let text_embeddings = pipe.text_embeddings(&tokenizers, &req.prompt, negative)?;
        let components = self.components(&pipe)?;
        let images = pipe.render(
            req,
            &text_embeddings,
            &components.unet,
            &components.vae,
            components.pid.as_deref(),
            on_progress,
        )?;
        Ok(GenerationOutput::Images(images))
    }
}

/// SDXL's identity + the surface candle wires: real classifier-free guidance (negative prompt + CFG
/// scale), txt2img, `ddim`, the few-step **`lightning`** sampler (sc-6128 — Euler-trailing, CFG-off,
/// for distilled Lightning checkpoints), and **LoRA/LoKr** (sc-5165 — load-time merge of a trained
/// adapter into the UNet weights, see [`load`] + [`pipeline`]). No conditioning is advertised, and the
/// other acceleration samplers (lcm/hyper) remain the Python fallback's job (sc-3678) until candle
/// wires them — so the descriptor never promises a path `generate` can't serve (the false-capability
/// trap). Two backend-correct deviations from `mlx-gen-sdxl`: `backend = "candle"` and
/// `mac_only = false`.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID,
        family: "sdxl",
        // The tensor backend whose provider crate registered this engine (sc-3723). MLX sets "mlx".
        backend: "candle",
        modality: Modality::Image,
        capabilities: Capabilities {
            // SDXL uses real classifier-free guidance: honors the negative prompt + a CFG scale.
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: false,
            // txt2img only in sc-3675 — img2img/inpaint/control land later; advertising none means
            // the shared `validate_request` rejects any conditioning, and the worker keeps those
            // shapes on the Python path (sc-3678).
            conditioning: vec![],
            // sc-5165: a trained LoRA/LoKr adapter is merged into the UNet weights at load (`load` +
            // `pipeline::Pipeline::load_components`). Advertised so the worker routes adapter jobs here
            // rather than to the Python fallback.
            supports_lora: true,
            supports_lokr: true,
            // DDIM (eta=0) is the deterministic, launch-portable DEFAULT (sc-3673); `lightning`
            // (sc-6128) is the few-step Euler-trailing path for distilled checkpoints. epic 7114 P4
            // (sc-7124) added the curated ε/DDPM menu (euler / euler_ancestral / heun / dpmpp_2m /
            // dpmpp_sde / uni_pc / lcm + ddim) over `DiscreteModelSampling`, plus the curated σ-schedule
            // axis (normal / karras / sgm_uniform / …). sc-10826: every non-`lightning` render — the
            // omitted-sampler default (→ curated `ddim`) AND every named solver — now routes the curated
            // EPS path; the native candle-transformers DDIM inference loop (which ghosted on the default)
            // is gone, so `ddim` and the default take the clean curated `ddim` solver. `lightning` keeps
            // its native few-step path. The legacy `discrete` scheduler alias falls back to the native schedule.
            samplers: candle_gen::menu_with_aliases(
                candle_gen::curated_sampler_names(),
                &["lightning"],
            ),
            schedulers: candle_gen::menu_with_aliases(
                candle_gen::curated_scheduler_names(),
                &["discrete"],
            ),
            supported_guidance_methods: vec![],
            min_size: 512,
            max_size: 2048,
            max_count: 8,
            // candle is the Windows/CUDA backend — NOT Mac-only (the MLX provider sets this true).
            mac_only: false,
            // Packed q4/q8 MLX-tier inference (sc-9416 UNet + sc-9527 dual-CLIP + sc-9528 adapter fold)
            // is wired end-to-end, so advertise Q4/Q8 (sc-10767, epic 9083 full-catalog parity). The
            // tier is packed-detected from disk (`detect_packed_unet` / `detect_packed_clip`); the
            // LoadSpec `quant` overlay is an advisory no-op on an already-packed tier (as with
            // boogu/flux2-dev). bf16 tiers stay dense (Quant::None), verbatim.
            supported_quants: &[Quant::Q4, Quant::Q8],
            supports_kv_cache: false,
            requires_sigma_shift: false,
        },
    }
}

/// Construct the (lazy) candle SDXL generator from a [`LoadSpec`]. `spec.weights` must be a
/// [`WeightsSource::Dir`] pointing at a `stabilityai/stable-diffusion-xl-base-1.0`-layout snapshot
/// (the diffusers multi-component tree: `text_encoder/`, `text_encoder_2/`, `unet/`, …).
///
/// `spec.adapters` (sc-5165) are LoRA/LoKr adapters to **merge into the UNet weights** — folded in at
/// the first `generate`'s component load (this `load` stays lazy: no file I/O here), via
/// [`adapters::merge_adapters`](crate::pipeline). PEFT (the candle trainer's format) + kohya LoRA and
/// PEFT/kohya LoKr are supported; an adapter that matches no UNet target errors at that first
/// `generate` rather than rendering an unadapted image silently.
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(
                "sdxl expects a snapshot directory (text_encoder/ text_encoder_2/ unet/ …), not a \
                 single .safetensors file"
                    .into(),
            ));
        }
    };
    // SDXL is fp16 (the production reference dtype) regardless of the CPU-default dtype; the device
    // is the backend selected at compile time (CUDA on Windows, Metal/CPU on Mac).
    let device = candle_gen::default_device()?;
    Ok(Box::new(SdxlGenerator {
        descriptor: descriptor(),
        root,
        device,
        dtype: DType::F16,
        adapters: spec.adapters.clone(),
        // PiD is an optional aux decoder (epic 7840 / sc-7853): capture the load-spec component (if
        // any) so the lazy component build loads the engine once. Unlike adapters, it is not rejected
        // — `None` simply keeps the byte-exact native-VAE path.
        pid_spec: spec.pid.clone(),
        components: Mutex::new(None),
        tokenizers: Mutex::new(None),
    }))
}

// Link-time self-registration into gen-core's model registry. Linking this crate makes
// `gen_core::load("sdxl", …)` resolve the candle generator — no central match statement to edit.
candle_gen::register_generators! { descriptor => load }

/// Force-link hook. A consumer that only reaches this provider *through* the `gen_core` registry
/// references nothing in this crate directly, so the linker (MSVC in particular, on a release
/// build) discards the whole rlib — taking the `inventory::submit!` registration above with it, and
/// `gen_core::load("sdxl", …)` then fails with "no generator registered". Referencing this no-op
/// from the consumer keeps the crate linked so the registration survives. The SceneWorks worker
/// force-links each provider crate for exactly this reason (e.g. `sensenova_jobs`).
pub fn force_link() {}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::registry;
    use candle_gen::gen_core::{Conditioning, ConditioningKind, Image, LoadSpec, WeightsSource};

    /// The seam under test: this provider's `inventory::submit!` is linked into the test binary,
    /// so resolving `"sdxl"` through gen-core's registry returns OUR candle generator. `load` is
    /// lazy, so a nonexistent weights dir still resolves (no file I/O until `generate`).
    #[test]
    fn sdxl_registers_and_resolves_as_candle() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry::load("sdxl", &spec).expect("candle sdxl is registered");
        assert_eq!(g.descriptor().id, "sdxl");
        assert_eq!(g.descriptor().backend, "candle");
        assert_eq!(g.descriptor().modality, Modality::Image);
    }

    #[test]
    fn descriptor_advertises_only_wired_txt2img_surface() {
        let d = descriptor();
        assert!(d.capabilities.supports_negative_prompt);
        assert!(d.capabilities.supports_guidance);
        assert!(!d.capabilities.supports_true_cfg);
        assert!(!d.capabilities.mac_only);
        // txt2img: no conditioning advertised. sc-5165: LoRA/LoKr ARE now wired (load-time merge), so
        // they are advertised — the worker routes adapter jobs to candle. sc-6128: the few-step
        // `lightning` accel sampler is wired too (the lcm/hyper accel samplers still are not).
        assert!(d.capabilities.conditioning.is_empty());
        assert!(d.capabilities.supports_lora);
        assert!(d.capabilities.supports_lokr);
        // sc-10767 (epic 9083): the packed q4/q8 MLX-tier inference path (sc-9416/9527/9528) is now
        // advertised, so the worker keeps quant tier-selects on the candle lane rather than deferring.
        assert_eq!(d.capabilities.supported_quants, &[Quant::Q4, Quant::Q8]);
        // sc-7124: the curated ε/DDPM sampler menu + the native `lightning` alias; `ddim` is part of
        // the curated vocabulary and (sc-10826) is now the omitted-sampler default's solver too. The
        // curated scheduler axis + `discrete`.
        assert_eq!(
            d.capabilities.samplers,
            candle_gen::menu_with_aliases(candle_gen::curated_sampler_names(), &["lightning"])
        );
        assert!(d.capabilities.samplers.contains(&"ddim"));
        assert!(d.capabilities.samplers.contains(&"dpmpp_2m"));
        assert!(d.capabilities.samplers.contains(&"lightning"));
        assert_eq!(
            d.capabilities.schedulers,
            candle_gen::menu_with_aliases(candle_gen::curated_scheduler_names(), &["discrete"])
        );
    }

    /// sc-3677 parity: the worker maps BOTH `sdxl` and `realvisxl` onto this single descriptor, so
    /// the contract surface it reads (capability advertisement + request validation) is identical for
    /// the two model ids. This pins the parity-relevant shape the Python `SdxlDiffusersAdapter` path
    /// is reconciled against — dims policy (min/max size, the latent-/8 size multiple), batch ceiling,
    /// and the deterministic `ddim` sampler. The accepted *differences* (DDIM vs the adapter's
    /// euler_ancestral default, sc-3673; the txt2img-only surface routing conditioning/LoRA to the
    /// Python fallback, sc-3678) are documented in the crate docs + tests/conformance.rs.
    #[test]
    fn realvisxl_shares_the_sdxl_contract_surface() {
        let d = descriptor();
        assert_eq!(d.family, "sdxl");
        assert_eq!(d.backend, "candle");
        assert_eq!(d.capabilities.min_size, 512);
        assert_eq!(d.capabilities.max_size, 2048);
        assert_eq!(d.capabilities.max_count, 8);
        // sc-7124: the curated sampler menu (incl. `ddim`) + the native `lightning` alias.
        assert_eq!(
            d.capabilities.samplers,
            candle_gen::menu_with_aliases(candle_gen::curated_sampler_names(), &["lightning"])
        );
        // SDXL works in latent space at /8 — the size policy both ids share (validate rejects
        // non-multiples). Anchored here so a change to the alignment is a parity-visible diff.
        assert_eq!(SIZE_MULTIPLE, 8);
    }

    /// sc-3674: the flash-attn runtime toggle defaults on and round-trips (what the worker/UI drive).
    #[test]
    fn flash_attn_toggle_roundtrips() {
        assert!(
            flash_attn_enabled(),
            "flash-attn runtime toggle defaults on"
        );
        set_flash_attn(false);
        assert!(!flash_attn_enabled());
        set_flash_attn(true);
        assert!(flash_attn_enabled());
    }

    /// sc-4987: the VAE-tiling runtime toggle defaults on (to hit the <12 GiB target out of the box)
    /// and round-trips — what the worker/UI drive.
    #[test]
    fn vae_tiling_toggle_roundtrips() {
        assert!(
            vae_tiling_enabled(),
            "vae-tiling runtime toggle defaults on"
        );
        set_vae_tiling(false);
        assert!(!vae_tiling_enabled());
        set_vae_tiling(true);
        assert!(vae_tiling_enabled());
    }

    /// A txt2img request passes validation; unsupported shapes are rejected clearly (not silently
    /// served). Uses the lazy generator so no weights are needed.
    #[test]
    fn validate_accepts_txt2img_and_rejects_unsupported() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry::load("sdxl", &spec).unwrap();

        let ok = GenerationRequest {
            prompt: "a rusty robot holding a lit candle".into(),
            guidance: Some(7.0),
            negative_prompt: Some("blurry".into()),
            ..Default::default()
        };
        assert!(g.validate(&ok).is_ok());

        // Empty prompt, non-multiple-of-8 size, explicit 0 steps, and any conditioning are rejected.
        for bad in [
            GenerationRequest::default(), // empty prompt
            GenerationRequest {
                prompt: "x".into(),
                width: 1020,
                ..Default::default()
            },
            GenerationRequest {
                prompt: "x".into(),
                steps: Some(0),
                ..Default::default()
            },
            GenerationRequest {
                prompt: "x".into(),
                conditioning: vec![Conditioning::Reference {
                    image: Image::default(),
                    strength: None,
                }],
                ..Default::default()
            },
        ] {
            assert!(g.validate(&bad).is_err(), "should reject: {bad:?}");
        }
        // Sanity: the rejected conditioning above is a kind the descriptor does not advertise.
        assert!(!descriptor()
            .capabilities
            .accepts(ConditioningKind::Reference));
    }

    /// sc-6128: `validate` accepts the advertised `lightning` sampler (the worker forces it for
    /// `realvisxl_lightning`) and still rejects an unadvertised one — the shared `validate_request`
    /// only passes a named sampler that is in `descriptor().samplers`. GPU-free (lazy generator).
    #[test]
    fn validate_accepts_lightning_sampler() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry::load("sdxl", &spec).unwrap();

        let lightning = GenerationRequest {
            prompt: "a rusty robot holding a lit candle".into(),
            sampler: Some("lightning".into()),
            steps: Some(5),
            guidance: Some(1.0),
            ..Default::default()
        };
        assert!(g.validate(&lightning).is_ok());

        // sc-7124: a curated ε/DDPM sampler is now advertised and accepted.
        let curated = GenerationRequest {
            prompt: "x".into(),
            sampler: Some("dpmpp_2m".into()),
            scheduler: Some("karras".into()),
            ..Default::default()
        };
        assert!(g.validate(&curated).is_ok());

        // An unadvertised sampler is still rejected (not silently downgraded).
        let bogus = GenerationRequest {
            prompt: "x".into(),
            sampler: Some("not_a_sampler".into()),
            ..Default::default()
        };
        assert!(g.validate(&bogus).is_err());

        // F-116: `lightning` + a NON-DEFAULT curated scheduler (`karras`) is rejected — lightning
        // ignores the scheduler axis, so honoring the selection is impossible and silently dropping it
        // would mislead.
        let lightning_sched = GenerationRequest {
            prompt: "x".into(),
            sampler: Some("lightning".into()),
            scheduler: Some("karras".into()),
            ..Default::default()
        };
        assert!(g.validate(&lightning_sched).is_err());

        // …but `lightning` with the DEFAULT `normal` scheduler PASSES — a worker that always populates
        // `scheduler:"normal"` alongside lightning must not hard-error, and `normal` resolves to
        // lightning's own trailing schedule anyway (nothing dropped).
        let lightning_normal = GenerationRequest {
            prompt: "x".into(),
            sampler: Some("lightning".into()),
            scheduler: Some("normal".into()),
            ..Default::default()
        };
        assert!(g.validate(&lightning_normal).is_ok());

        // …as does `lightning` with no scheduler at all, or the unrecognized native-fallback `discrete`
        // alias — both likewise resolve to lightning's own trailing schedule.
        let lightning_none = GenerationRequest {
            prompt: "x".into(),
            sampler: Some("lightning".into()),
            scheduler: None,
            ..Default::default()
        };
        assert!(g.validate(&lightning_none).is_ok());
        let lightning_discrete = GenerationRequest {
            prompt: "x".into(),
            sampler: Some("lightning".into()),
            scheduler: Some("discrete".into()),
            ..Default::default()
        };
        assert!(g.validate(&lightning_discrete).is_ok());
    }

    /// sc-5165: `load` now ACCEPTS LoRA/LoKr adapters — it carries them for a load-time merge into the
    /// UNet weights at the first `generate`. `load` stays lazy (no file I/O), so a nonexistent adapter
    /// path still resolves here; an unresolvable / mis-formatted adapter errors only when `generate`
    /// first builds the UNet. (The merge math + format routing are covered in `adapters`'s tests.)
    #[test]
    fn load_accepts_lora_adapters() {
        use candle_gen::gen_core::{AdapterKind, AdapterSpec};
        let spec = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_adapters(vec![
            AdapterSpec::new("/lora.safetensors".into(), 1.0, AdapterKind::Lora),
        ]);
        let g = load(&spec).expect("load accepts adapters (lazy; merge defers to generate)");
        assert!(g.descriptor().capabilities.supports_lora);
        assert!(g.descriptor().capabilities.supports_lokr);
    }

    #[test]
    fn load_rejects_single_file_source() {
        let spec = LoadSpec::new(WeightsSource::File("/tmp/sdxl.safetensors".into()));
        let err = load(&spec).err().expect("expected an error").to_string();
        assert!(err.contains("snapshot directory"), "got: {err}");
    }
}
