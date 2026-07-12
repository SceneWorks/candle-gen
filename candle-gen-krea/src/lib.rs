//! # candle-gen-krea
//!
//! The **Krea 2** provider crate for [`candle-gen`](candle_gen) — the candle (Windows/CUDA) sibling of
//! `mlx-gen-krea`. Registers two generator ids over **one architecture** (only the DiT weights differ,
//! distilled vs base — the Boogu base/turbo precedent):
//!
//! * **`krea_2_turbo`** — the user-facing text-to-image model: a 12B **dense single-stream**
//!   rectified-flow / v-param DiT (28 gated single-stream blocks, hidden 6144, GQA 48Q/12KV, head_dim
//!   128, SwiGLU 16384, 3-axis interleaved RoPE `[32,48,48]`, `DoubleSharedModulation`, and a
//!   `text_fusion` front-end that aggregates the 12 selected Qwen3-VL hidden layers) driven by a
//!   Qwen3-VL-4B condition encoder and the Qwen-Image VAE. TDM-distilled few-step (8 steps),
//!   **CFG-free** (guidance inert), up to 2048².
//! * **`krea_2_raw`** (sc-9994 / epic 9992) — the undistilled 12B base run as a **full classifier-free
//!   guidance** generator: a real guidance scale + optional user negative prompt, 52 steps, resolution-
//!   dynamic mu ([`pipeline::render_base`]). The SAME id is also the Krea LoRA *training* base (Path 1:
//!   one id, both roles — generator + trainer registries). Two DiT forwards/step (cond vs uncond).
//!
//! **Reuse:** the VAE is `candle_gen_qwen_image::vae::QwenVae` (the exact `AutoencoderKLQwenImage`
//! Qwen-Image ships — per-channel `latents_mean`/`latents_std` de-norm) — reused verbatim, as
//! `mlx-gen-krea` reuses `mlx-gen-qwen-image`'s `QwenVae`. The Qwen3-VL-4B condition encoder
//! ([`text_encoder`]), the single-stream DiT ([`transformer`]), and the rectified-flow sampler
//! ([`schedule`]) are ported here.
//!
//! `backend = "candle"`, `mac_only = false`. Apache-2.0; Krea 2 Community License (non-commercial use
//! satisfies it). The packed q4/q8/bf16 turnkey loads per-tier via `loader::linear_detect` (sc-9411);
//! the descriptor advertises `supported_quants: [Q4, Q8]` so the worker's A-B quant toggle engages
//! (sc-9607).

pub mod adapters;
pub mod config;
pub mod convert;
pub mod loader;
pub mod pipeline;
pub mod quant;
pub mod schedule;
pub mod text_encoder;
pub mod tokenizer;
pub mod transformer;
pub mod vae;
pub mod vision;

// The candle Krea LoRA/LoKr trainer (sc-7577) + its vendored composable-op trainable DiT. Private
// (reached through gen-core's trainer registry by id, like the SDXL/Z-Image trainers); the
// `inventory::submit!` in `training` is kept linked by [`force_link`].
mod train_dit;
mod training;

// The pose-ControlNet control branch (sc-8460 spike / sc-8462, epic 8459): a trainable N-block side
// branch over the frozen DiT with zero-init per-block residual injection. Public so the spike's
// trainer/inference example binaries can drive it; the worker route is a later story.
pub mod control;

// The callable control-branch trainer (sc-8462): the spike CLI's training loop lifted into a
// reusable `ControlTrainer` so the ControlNet Training Studio worker driver (epic 10159 B2) can
// drive a run and stream its progress. Kept gen-core-neutral for the later MLX training lane.
pub mod control_train;

// The gen_core `Trainer` adapter for Krea pose-ControlNet (sc-10163, epic 10159 B2): registers
// `krea_2_control` so the studio drives control-branch training through the same `load_trainer` path
// LoRA uses. Private (reached via the registry by id); its `register_trainer!` is kept linked by
// [`force_link`], like the LoRA trainer in `training`.
mod control_trainer;

// The Krea 2 Turbo pose-ControlNet **inference** provider (sc-8464, epic 8459): loads a trained
// control-branch overlay on the frozen Turbo base and renders a pose-conditioned image. The
// deployable form of the sc-8460 spike inference harness; the worker `KreaControl` route calls it.
pub mod control_provider;

// Shared test-only tiny-DiT fixture (training + control tests).
#[cfg(test)]
mod testfix;

pub use adapters::{
    install_additive, merge_adapters, merge_into_weights, AdditiveReport, MergeReport,
};
pub use config::Krea2Config;
pub use control_provider::{
    Krea2Control, Krea2ControlPaths, Krea2ControlRequest, DEFAULT_CONTROL_SCALE,
};
pub use pipeline::Components;
pub use schedule::{krea_sigmas, turbo_sigmas, TURBO_MU, TURBO_STEPS};
pub use text_encoder::{KreaTeConfig, KreaTextEncoder};
pub use tokenizer::KreaTokenizer;
// The composable trainable DiT, exposed for the sc-8460 control-branch spike binaries (the branch
// injects into its block stack; its forward is the spike's inference surface).
pub use train_dit::KreaTrainDit;
pub use transformer::Krea2Transformer;
pub use vae::{load_vae, QwenVae};

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::Device;
use candle_gen::gen_core::{
    self, AdapterSpec, Capabilities, Conditioning, ConditioningKind, GenerationOutput,
    GenerationRequest, Generator, Image, LoadSpec, Modality, ModelDescriptor, PidWeights, Progress,
    Quant, WeightsSource,
};

/// Registry id for the Krea 2 Turbo text-to-image variant. Matches the SceneWorks worker's
/// `payload.model` and the manifest `engine_id` (sc-7572).
pub const KREA_2_TURBO_ID: &str = "krea_2_turbo";

/// Registry id for the undistilled **Raw** full-CFG text-to-image variant (sc-9994 / epic 9992). The
/// SAME string as the Krea LoRA *trainer* base (`crate::training::KREA_2_RAW_ID`) — Path 1 makes one id
/// both the training base and a first-class generator; the trainer + generator live in separate
/// registries so the shared id never collides. Matches the worker `payload.model` + manifest `engine_id`.
pub const KREA_2_RAW_ID: &str = "krea_2_raw";

/// Registry id for the **image-edit** variant (epic 10871 / sc-11085). Kontext-style instruction edit
/// over one or two source references (scene = image 1, person = image 2) on the undistilled full-CFG
/// base. The engine (pipeline `render_edit` + edit components) landed via #416 but was unreachable
/// through the `Generator` seam until this id was registered — the candle mirror of the mlx-gen #693
/// `krea_2_edit` seam. Matches the worker `payload.model` + manifest `engine_id`.
pub const KREA_2_EDIT_ID: &str = "krea_2_edit";

/// Surface tag for the **distilled Turbo image-edit** (`krea_2_turbo_edit`, sc-11640). Not a registered
/// `Generator` id — the CFG-free distilled edit is driven through the worker's bespoke
/// `generate_candle_krea_edit_stream` lane, which calls [`pipeline::render_edit`] with `distilled = true`
/// directly. Named here so the shared edit path (PiD decode-seam errors, sc-11197) reports the right
/// surface for the Turbo edit vs the Raw [`KREA_2_EDIT_ID`].
pub const KREA_2_TURBO_EDIT_ID: &str = "krea_2_turbo_edit";

/// patch_size(2)·vae_downsample(8) = 16 — patchify requires latent dims divisible by this.
const SIZE_MULTIPLE: u32 = 16;
/// Resolution bounds (W/H). Turbo renders up to 2048²; the catalog/worker gate the UI options tighter.
const RES_MIN: u32 = 256;
const RES_MAX: u32 = 2048;
/// Max images per request (the image-model standard, shared with the other families).
const MAX_COUNT: u32 = 8;

/// A lazily-loaded Krea 2 Turbo generator. The components (tokenizer + Qwen3-VL-4B TE + single-stream
/// DiT + Qwen-Image VAE) load on the first `generate` and are cached.
pub struct KreaGenerator {
    descriptor: ModelDescriptor,
    root: PathBuf,
    device: Device,
    /// LoRA/LoKr adapters merged into the DiT weights at component-load (sc-7836). Fixed for this
    /// generator instance; empty ⇒ the stock unadapted build.
    adapters: Vec<AdapterSpec>,
    /// The `LoadSpec::pid` component captured at load (epic 7840 / sc-7853), threaded into the lazy
    /// component build so the PiD engine loads once alongside the base model. `None` when not opted in.
    pid_spec: Option<PidWeights>,
    /// The community **INT8-ConvRot** DiT single-file checkpoint (sc-9300), captured at load when the
    /// spec selected the ConvRot consume path (a `WeightsSource::File` on `LoadSpec::text_encoder` — see
    /// [`build`]). `Some` ⇒ the lazy component build takes the DiT from this int8 checkpoint
    /// ([`pipeline::load_components_convrot`]) and the tokenizer / Qwen3-VL TE / Qwen-Image VAE from the
    /// canonical `root` snapshot; `None` ⇒ the dense/packed `transformer/` snapshot path (unchanged).
    convrot_dit: Option<PathBuf>,
    components: Mutex<Option<Arc<Components>>>,
    /// The image-edit-only components (Qwen-Image VAE **encoder** + Qwen3-VL **vision tower**), loaded
    /// lazily on the first edit so the txt2img (Turbo/Raw) paths keep their footprint (epic 10871).
    /// Only ever populated for the `krea_2_edit` id.
    edit_components: Mutex<Option<Arc<pipeline::EditComponents>>>,
}

impl KreaGenerator {
    fn components(&self) -> gen_core::Result<Arc<Components>> {
        candle_gen::cached(&self.components, || {
            // ConvRot consume path (sc-9300): when a ConvRot DiT was selected, the DiT is taken from
            // the int8 single-file checkpoint while everything else loads from the canonical snapshot.
            // The sm_89 compute-capability floor is enforced inside `load_components_convrot` (locked
            // decision 7). LoRA/LoKr and PiD overlays are not wired through the ConvRot variant
            // (rejected at `build`).
            Ok(match self.convrot_dit.as_ref() {
                Some(convrot_dit) => Arc::new(pipeline::load_components_convrot(
                    &self.root,
                    convrot_dit,
                    &self.device,
                )?),
                None => Arc::new(pipeline::load_components(
                    &self.root,
                    &self.device,
                    &self.adapters,
                    self.pid_spec.as_ref(),
                )?),
            })
        })
    }

    /// The image-edit-only components (VAE encoder + Qwen3-VL vision tower), loaded once on the first
    /// edit and cached (epic 10871). Both read weights that already ship in the Krea 2 snapshot.
    fn edit_components(&self) -> gen_core::Result<Arc<pipeline::EditComponents>> {
        candle_gen::cached(&self.edit_components, || {
            Ok(Arc::new(pipeline::load_edit_components(
                &self.root,
                &self.device,
            )?))
        })
    }
}

impl Generator for KreaGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        let id = self.descriptor.id;
        if req.prompt.trim().is_empty() {
            return Err(gen_core::Error::Msg(format!(
                "{id}: prompt must not be empty"
            )));
        }
        self.descriptor.capabilities.validate_request(id, req)?;
        if req.steps == Some(0) {
            return Err(gen_core::Error::Msg(format!("{id}: steps must be >= 1")));
        }
        if !req.width.is_multiple_of(SIZE_MULTIPLE) || !req.height.is_multiple_of(SIZE_MULTIPLE) {
            return Err(gen_core::Error::Msg(format!(
                "{id}: width/height must be multiples of {SIZE_MULTIPLE} (got {}x{})",
                req.width, req.height
            )));
        }
        // The Edit variant needs 1..=2 source references (scene, then person). The capability floor
        // above (empty `conditioning` on Turbo/Raw) already rejects any Reference/MultiReference on the
        // txt2img ids; only `krea_2_edit` advertises them, so resolve + count-check here.
        if self.descriptor.id == KREA_2_EDIT_ID {
            resolve_edit_references(req)?;
        }
        Ok(())
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.validate(req)?;
        let comps = self.components()?;
        // Variant read off the descriptor id (the mlx-gen-krea `generate_impl` branch): Edit = Kontext
        // instruction edit over 1-2 references on the full-CFG base; Raw = full-CFG undistilled txt2img
        // (52-step, dynamic-mu, two forwards/step); Turbo = CFG-free distilled (8-step, one forward).
        // One generator struct, three render paths.
        let images = if self.descriptor.id == KREA_2_EDIT_ID {
            // Extract the fixed-order source set (scene = image 1, person = image 2) from the request
            // conditioning, load the edit-only components lazily, and run the Kontext edit path.
            let references: Vec<Image> =
                resolve_edit_references(req)?.into_iter().cloned().collect();
            let edit = self.edit_components()?;
            // The registered `krea_2_edit` seam is the undistilled full-CFG edit (`distilled = false`);
            // the CFG-free distilled Turbo edit (sc-11640) is driven through the worker's bespoke
            // `generate_candle_krea_edit_stream` lane (which calls `render_edit(distilled = true)`
            // directly), matching how every candle edit lane bypasses the registered seam.
            pipeline::render_edit(
                &comps,
                &edit,
                req,
                &references,
                false,
                &self.device,
                on_progress,
            )?
        } else if self.descriptor.id == KREA_2_RAW_ID {
            pipeline::render_base(&comps, req, &self.device, on_progress)?
        } else {
            pipeline::render(&comps, req, &self.device, on_progress)?
        };
        Ok(GenerationOutput::Images(images))
    }
}

/// Krea 2 Turbo identity + capabilities — constructible without loading weights (registry
/// introspection / capability advertisement). Distilled few-step text-to-image: **CFG-free** (the TDM
/// distillation baked the guided velocity into the weights, so no guidance / unconditional branch), no
/// user negative prompt, no img2img/control conditioning on the Turbo checkpoint.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: KREA_2_TURBO_ID,
        family: "krea_2",
        backend: "candle",
        modality: Modality::Image,
        capabilities: Capabilities {
            supports_negative_prompt: false,
            // CFG-free distilled student (like Ideogram Turbo / Boogu Turbo / SDXL-Lightning).
            supports_guidance: false,
            supports_true_cfg: false,
            conditioning: Vec::new(),
            // LoRA/LoKr wired (sc-7836): a trained `krea_2_raw` adapter merges into the dense DiT
            // attention projections at load ([`adapters::merge_into_weights`]), closing the candle
            // train→infer loop.
            supports_lora: true,
            supports_lokr: true,
            // Rectified-flow v-param over the unified curated-sampler framework (epic 7114). The
            // native distilled loop stays the byte-exact default (`req.sampler == None`).
            samplers: candle_gen::curated_sampler_names(),
            schedulers: candle_gen::curated_scheduler_names(),
            supported_guidance_methods: vec![],
            min_size: RES_MIN,
            max_size: RES_MAX,
            max_count: MAX_COUNT,
            mac_only: false,
            // sc-9607: advertise the packed tiers so the worker's A-B quant toggle engages off-Mac.
            // The resolved q4/q8/bf16 turnkey subdir self-describes its tier (`loader::linear_detect`,
            // sc-9411); `build` no-ops the requested quant, and it composes with a merged LoRA overlay.
            supported_quants: &[Quant::Q4, Quant::Q8],
            supports_kv_cache: false,
            requires_sigma_shift: false,
        },
    }
}

/// Krea 2 **Raw** identity + capabilities (sc-9994 / epic 9992) — the undistilled 12B DiT run with
/// **true classifier-free guidance** (two DiT forwards/step: cond vs uncond) at 52 steps, unlike the
/// CFG-free distilled Turbo. Same architecture / snapshot layout as Turbo (only the DiT weights differ,
/// distilled vs base), so it shares [`build`] + the whole [`pipeline`]. Exposes a real guidance scale
/// AND a user negative prompt (unlike Turbo / Boogu base, which fixes the uncond to the empty prompt).
/// NOT guidance-distilled, so `supports_true_cfg` stays false — the two-forward CFG IS the guidance
/// (the Boogu-base precedent). Derived from [`descriptor`] so the shared surface (family / backend /
/// samplers / quants / size / LoRA) stays in lockstep with Turbo.
pub fn raw_descriptor() -> ModelDescriptor {
    let mut d = descriptor();
    d.id = KREA_2_RAW_ID;
    d.capabilities.supports_negative_prompt = true;
    d.capabilities.supports_guidance = true;
    d.capabilities.supports_true_cfg = false;
    d
}

/// Krea 2 **Edit** identity + capabilities (epic 10871 / sc-11085) — the Kontext-style instruction-edit
/// variant. Derived from [`raw_descriptor`] (the edit runs the undistilled **full-CFG** loop from pure
/// noise, with the references as in-context conditioning), so it inherits the Raw surface — real
/// guidance + a user negative prompt, packed quants, LoRA/LoKr (the edit LoRA merges through the shared
/// [`build`] adapter path) — and additionally advertises the source-reference conditioning:
/// [`ConditioningKind::Reference`] for a single source and [`ConditioningKind::MultiReference`] for two
/// (scene = image 1, person = image 2; [`pipeline::MAX_EDIT_REFERENCES`]).
pub fn edit_descriptor() -> ModelDescriptor {
    let mut d = raw_descriptor();
    d.id = KREA_2_EDIT_ID;
    d.capabilities.conditioning = vec![
        ConditioningKind::Reference,
        ConditioningKind::MultiReference,
    ];
    d
}

/// The image-edit source references, in fixed order (scene = image 1, person = image 2; sc-10878) —
/// collected from both [`Conditioning::Reference`] (a single source) and [`Conditioning::MultiReference`]
/// (two sources). At least one and at most [`pipeline::MAX_EDIT_REFERENCES`] is required; zero or more
/// than the cap is an error. Borrows from `req.conditioning`; the generate path clones the resolved set
/// into the owned `&[Image]` the pipeline consumes. Pure so it is unit-testable without weights.
fn resolve_edit_references(req: &GenerationRequest) -> gen_core::Result<Vec<&Image>> {
    let mut refs: Vec<&Image> = Vec::new();
    for c in &req.conditioning {
        match c {
            Conditioning::Reference { image, .. } => refs.push(image),
            Conditioning::MultiReference { images } => refs.extend(images.iter()),
            _ => {} // the capability floor already rejects the other conditioning kinds.
        }
    }
    if refs.is_empty() {
        return Err(gen_core::Error::Msg(format!(
            "{KREA_2_EDIT_ID}: an instruction edit requires at least one source reference image \
             (scene = image 1, person = image 2)"
        )));
    }
    if refs.len() > pipeline::MAX_EDIT_REFERENCES {
        return Err(gen_core::Error::Msg(format!(
            "{KREA_2_EDIT_ID}: at most {} references are supported (scene = image 1, person = image \
             2); got {}",
            pipeline::MAX_EDIT_REFERENCES,
            refs.len()
        )));
    }
    Ok(refs)
}

/// sc-9300 ConvRot selection: decode whether a [`LoadSpec`] selects the community INT8-ConvRot DiT
/// consume path, returning the DiT single-file checkpoint when it does. ConvRot rides the shared,
/// already-optional [`LoadSpec::text_encoder`] field as a [`WeightsSource::File`]; a [`WeightsSource::Dir`]
/// there is a mis-shaped spec (ConvRot is a single file) and errors. `None` on `text_encoder` ⇒ the
/// dense/packed snapshot path. Extracted from [`build`] so the routing decision is unit-testable on CPU
/// without loading weights.
fn convrot_selector(spec: &LoadSpec, id: &str) -> gen_core::Result<Option<PathBuf>> {
    match spec.text_encoder.as_ref() {
        Some(WeightsSource::File(p)) => Ok(Some(p.clone())),
        Some(WeightsSource::Dir(_)) => Err(gen_core::Error::Msg(format!(
            "candle {id}: LoadSpec::text_encoder selects the INT8-ConvRot DiT and must be a single \
             .safetensors file (WeightsSource::File), not a directory"
        ))),
        None => Ok(None),
    }
}

fn build(spec: &LoadSpec, descriptor: ModelDescriptor) -> gen_core::Result<Box<dyn Generator>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(format!(
                "{} expects a snapshot directory (transformer/ text_encoder/ vae/ tokenizer/), not a \
                 single .safetensors file",
                descriptor.id
            )));
        }
    };
    // sc-9300 seam: select the community **INT8-ConvRot** DiT consume path when the spec carries a
    // ConvRot DiT single-file checkpoint. It rides the shared, already-optional `LoadSpec::text_encoder`
    // field as a `WeightsSource::File` — the canonical Krea 2 snapshot (`spec.weights`, a `Dir`) still
    // supplies the tokenizer / Qwen3-VL TE / Qwen-Image VAE / config + all non-quantized surface, and
    // only the DiT weights are taken from the int8 checkpoint (`pipeline::load_components_convrot`,
    // which enforces the sm_89 compute-cap floor). This reuses an existing extensibility point (the same
    // pattern LTX uses to ride an aux path on `text_encoder`) rather than growing the shared
    // `WeightsSource` enum with a ConvRot variant — which would force a new match arm across every
    // provider in candle-gen AND the worker plus a gen-core pin bump. Only Krea reads this; every other
    // engine ignores `text_encoder` unchanged. `None`/`Dir` here ⇒ the dense/packed snapshot path below.
    let convrot_dit = convrot_selector(spec, descriptor.id)?;
    // LoRA/LoKr adapters are accepted and merged into the DiT at first `generate` (sc-7836); the merge
    // (`adapters::merge_into_weights`) is lazy, so a nonexistent adapter path still loads here.
    //
    // sc-9607: `spec.quantize` (Q4/Q8) is ACCEPTED and no-ops — the resolved per-tier turnkey is
    // already MLX-packed and `loader::linear_detect` builds each `QLinear::Quantized` straight from the
    // packed parts (sc-9411), composing with the adapter overlay (an adapter-merged projection stays
    // dense and takes priority). No on-the-fly quant pass runs; the requested quant is recipe-only.
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "candle {} does not support ControlNet / IP-Adapter overlays",
            descriptor.id
        )));
    }
    // The ConvRot consume path (sc-9300) is DiT-only and does not thread LoRA/LoKr or PiD overlays — the
    // int8 checkpoint replaces the dense transformer wholesale. Reject the combination up front so the
    // worker gets a clear error instead of silently dropping the overlay.
    if convrot_dit.is_some() && (!spec.adapters.is_empty() || spec.pid.is_some()) {
        return Err(gen_core::Error::Unsupported(format!(
            "candle {}: the INT8-ConvRot DiT path does not support LoRA/LoKr adapters or a PiD decoder \
             overlay",
            descriptor.id
        )));
    }
    let device = candle_gen::default_device()?;
    Ok(Box::new(KreaGenerator {
        descriptor,
        root,
        device,
        adapters: spec.adapters.clone(),
        // PiD is an optional aux decoder (epic 7840 / sc-7853): capture the load-spec component (if any)
        // so the lazy component build loads the engine once. `None` keeps the byte-exact native path.
        pid_spec: spec.pid.clone(),
        convrot_dit,
        components: Mutex::new(None),
        edit_components: Mutex::new(None),
    }))
}

/// Construct a lazy candle Krea 2 **Turbo** generator. `spec.weights` must be a [`WeightsSource::Dir`]
/// pointing at a candle-readable (bf16) Krea 2 snapshot (`transformer/ text_encoder/ vae/ tokenizer/`).
///
/// **INT8-ConvRot (sc-9300).** To load the community int8-quantized DiT instead of the snapshot's dense
/// `transformer/`, pass the ConvRot DiT single-file checkpoint as
/// `spec.text_encoder = Some(WeightsSource::File(convrot_dit.safetensors))` while keeping
/// `spec.weights = WeightsSource::Dir(canonical_snapshot)` (which supplies the tokenizer / TE / VAE /
/// config). The ConvRot path enforces the sm_89 compute-cap floor and does not combine with LoRA/LoKr
/// or PiD overlays.
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    build(spec, descriptor())
}

/// Construct a lazy candle Krea 2 **Raw** generator (`krea_2_raw`, sc-9994 / epic 9992). Identical
/// snapshot assembly to [`load`] — the Raw + Turbo turnkeys share the exact architecture / weight layout
/// (only distilled-vs-base DiT weights differ), so one [`build`] serves both — but stores the CFG-capable
/// [`raw_descriptor`] so `generate` runs the full-CFG [`pipeline::render_base`] path. Accepts the same
/// LoRA/LoKr, PiD, and packed-quant surface as Turbo; the ConvRot / ControlNet rejections are shared.
pub fn load_raw(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    build(spec, raw_descriptor())
}

/// Construct a lazy candle Krea 2 **Edit** generator (`krea_2_edit`, epic 10871 / sc-11085). Identical
/// snapshot assembly to [`load`] / [`load_raw`] — one [`build`] serves all three ids — but stores the
/// [`edit_descriptor`] so `generate` routes the reference-conditioned [`pipeline::render_edit`] path and
/// lazily loads the edit-only components (VAE encoder + vision tower). The edit LoRA rides the shared
/// `spec.adapters` merge path, exactly like a Raw-trained adapter on the txt2img ids.
pub fn load_edit(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    build(spec, edit_descriptor())
}

// Link-time registration: all three variants register here — `krea_2_turbo` (distilled, CFG-free),
// `krea_2_raw` (undistilled, full-CFG; sc-9994 / epic 9992), and `krea_2_edit` (Kontext instruction
// edit over 1-2 references; epic 10871 / sc-11085).
candle_gen::register_generators! {
    descriptor => load,
    raw_descriptor => load_raw,
    edit_descriptor => load_edit,
}

/// Force-link hook (keeps the `inventory::submit!` registrations — the `krea_2_turbo`, `krea_2_raw`,
/// and `krea_2_edit` generators and the `krea_2_raw` trainer — from being dead-stripped).
pub fn force_link() {}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::registry;

    #[test]
    fn registers_krea_2_turbo_as_candle() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry::load(KREA_2_TURBO_ID, &spec).expect("krea_2_turbo is registered");
        assert_eq!(g.descriptor().id, KREA_2_TURBO_ID);
        assert_eq!(g.descriptor().family, "krea_2");
        assert_eq!(g.descriptor().backend, "candle");
        assert!(!g.descriptor().capabilities.mac_only);
    }

    // --- Raw (undistilled, full-CFG) variant — sc-9994 / epic 9992 ---

    #[test]
    fn registers_krea_2_raw_as_candle() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry::load(KREA_2_RAW_ID, &spec).expect("krea_2_raw is registered");
        assert_eq!(g.descriptor().id, KREA_2_RAW_ID);
        assert_eq!(g.descriptor().family, "krea_2");
        assert_eq!(g.descriptor().backend, "candle");
        assert!(!g.descriptor().capabilities.mac_only);
    }

    #[test]
    fn raw_descriptor_is_krea_2_raw_and_cfg_capable() {
        let d = raw_descriptor();
        assert_eq!(d.id, KREA_2_RAW_ID);
        // The generator id MUST equal the LoRA-trainer base id (Path 1: one id, both roles).
        assert_eq!(KREA_2_RAW_ID, crate::training::KREA_2_RAW_ID);
        assert_eq!(d.family, "krea_2");
        assert_eq!(d.backend, "candle");
        assert_eq!(d.modality, Modality::Image);
        // Undistilled base: real CFG guidance + a user negative prompt (unlike Turbo / Boogu base).
        assert!(d.capabilities.supports_guidance);
        assert!(d.capabilities.supports_negative_prompt);
        // Not guidance-distilled: no separate embedded-guidance axis, so no true_cfg toggle.
        assert!(!d.capabilities.supports_true_cfg);
        // Shared surface stays in lockstep with Turbo (derived from `descriptor()`).
        assert!(d.capabilities.supports_lora && d.capabilities.supports_lokr);
        assert_eq!(d.capabilities.supported_quants, &[Quant::Q4, Quant::Q8]);
        assert_eq!(d.capabilities.samplers, descriptor().capabilities.samplers);
        assert!(!d.capabilities.mac_only);
        assert_eq!(pipeline::RAW_STEPS, 52);
        assert_eq!(pipeline::RAW_GUIDANCE, 3.5);
    }

    #[test]
    fn raw_validate_accepts_guidance_and_negative_prompt() {
        // The CFG floor that rejects these on Turbo must ACCEPT them on Raw.
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry::load(KREA_2_RAW_ID, &spec).unwrap();
        let ok = GenerationRequest {
            prompt: "a red apple on a wooden table".into(),
            width: 1024,
            height: 1024,
            guidance: Some(3.5),
            negative_prompt: Some("blurry, lowres".into()),
            ..Default::default()
        };
        assert!(g.validate(&ok).is_ok());
    }

    #[test]
    fn load_raw_rejects_single_file_like_turbo() {
        // Same snapshot loader as Turbo — a single-file weights source is rejected the same way.
        let file = LoadSpec::new(WeightsSource::File("/tmp/x.safetensors".into()));
        assert!(load_raw(&file).is_err());
        // A LoRA `LoadSpec` on the Raw id is accepted + lazy, exactly like Turbo (sc-7836 wiring).
        let dir = LoadSpec::new(WeightsSource::Dir("/snap".into()));
        assert!(load_raw(&dir).is_ok());
    }

    #[test]
    fn descriptor_surface_is_cfg_free_turbo() {
        let d = descriptor();
        assert_eq!(d.id, KREA_2_TURBO_ID);
        assert_eq!(d.modality, Modality::Image);
        assert!(!d.capabilities.supports_guidance);
        assert!(!d.capabilities.supports_negative_prompt);
        assert!(d.capabilities.conditioning.is_empty());
        // LoRA/LoKr merge wired (sc-7836); packed Q4/Q8 tiers advertised (sc-9607).
        assert!(d.capabilities.supports_lora);
        assert!(d.capabilities.supports_lokr);
        assert_eq!(d.capabilities.supported_quants, &[Quant::Q4, Quant::Q8]);
        assert_eq!(d.capabilities.max_size, 2048);
        assert_eq!(TURBO_STEPS, 8);
    }

    #[test]
    fn validate_accepts_txt2img_and_rejects_bad() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry::load(KREA_2_TURBO_ID, &spec).unwrap();
        let ok = GenerationRequest {
            prompt: "a red apple on a wooden table".into(),
            width: 1024,
            height: 1024,
            ..Default::default()
        };
        assert!(g.validate(&ok).is_ok());
        for bad in [
            GenerationRequest::default(),
            GenerationRequest {
                prompt: "x".into(),
                width: 1000,
                height: 1024,
                ..Default::default()
            },
            GenerationRequest {
                prompt: "x".into(),
                width: 1024,
                height: 1024,
                steps: Some(0),
                ..Default::default()
            },
        ] {
            assert!(g.validate(&bad).is_err(), "should reject: {bad:?}");
        }
    }

    /// F-154 (sc-11210): the empty-prompt guard rejects a whitespace-only prompt (`trim().is_empty()`),
    /// matching the chroma and krea control-provider siblings — a whitespace prompt would otherwise
    /// reach the TE as an effectively-empty sequence.
    #[test]
    fn validate_rejects_whitespace_only_prompt() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry::load(KREA_2_TURBO_ID, &spec).unwrap();
        for ws in ["   ", "\t", "\n", " \t\n "] {
            let req = GenerationRequest {
                prompt: ws.into(),
                width: 1024,
                height: 1024,
                ..Default::default()
            };
            assert!(
                g.validate(&req).is_err(),
                "whitespace-only prompt {ws:?} must be rejected"
            );
        }
    }

    #[test]
    fn validate_rejects_guidance_and_negative_prompt() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry::load(KREA_2_TURBO_ID, &spec).unwrap();
        let base = GenerationRequest {
            prompt: "x".into(),
            width: 512,
            height: 512,
            ..Default::default()
        };
        assert!(g
            .validate(&GenerationRequest {
                guidance: Some(3.5),
                ..base.clone()
            })
            .is_err());
        assert!(g
            .validate(&GenerationRequest {
                negative_prompt: Some("y".into()),
                ..base
            })
            .is_err());
    }

    #[test]
    fn load_accepts_lora_rejects_single_file_and_unwired_surfaces() {
        use candle_gen::gen_core::{AdapterKind, AdapterSpec};
        let file = LoadSpec::new(WeightsSource::File("/tmp/q.safetensors".into()));
        assert!(load(&file).is_err());
        // LoRA/LoKr now wired (sc-7836): a LoRA `LoadSpec` is accepted (lazily — the merge happens at
        // first `generate`), so `load` resolves rather than rejecting.
        let lora = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_adapters(vec![
            AdapterSpec::new("/lora.safetensors".into(), 1.0, AdapterKind::Lora),
        ]);
        assert!(load(&lora).is_ok(), "LoRA load is wired + lazy (sc-7836)");
        // sc-9607: a Q4/Q8 `spec.quantize` is now ACCEPTED (a no-op on the already-packed tier) — load
        // proceeds past the quant check and constructs lazily, exactly like the LoRA case above.
        let quant = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_quant(Quant::Q8);
        assert!(
            load(&quant).is_ok(),
            "Q4/Q8 quant is accepted + lazy (sc-9607)"
        );
    }

    // sc-9300: the ConvRot consume path is reachable through the LoadSpec API. The selector routes a
    // `WeightsSource::File` on `text_encoder` to the INT8-ConvRot DiT (`load_components_convrot`), a
    // plain `Dir` weights spec to the dense/packed snapshot path (`load_components`), and rejects the
    // mis-shaped / incompatible combinations. These assert the routing decision on CPU (no weights).
    #[test]
    fn convrot_selector_routes_file_to_convrot_dir_to_dense() {
        // A `Dir`-only spec (canonical snapshot, no ConvRot DiT) ⇒ the dense/packed path.
        let dense = LoadSpec::new(WeightsSource::Dir("/snap".into()));
        assert_eq!(
            convrot_selector(&dense, KREA_2_TURBO_ID).unwrap(),
            None,
            "a Dir-only spec dispatches to the dense/packed snapshot path"
        );
        // A ConvRot DiT single-file on `text_encoder` ⇒ the ConvRot path, carrying the DiT checkpoint.
        let convrot = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_convrot_text_encoder();
        assert_eq!(
            convrot_selector(&convrot, KREA_2_TURBO_ID).unwrap(),
            Some(PathBuf::from("/krea2_int8_convrot.safetensors")),
            "a File on text_encoder selects the ConvRot DiT consume path"
        );
        // A `Dir` on `text_encoder` is not a valid ConvRot selector (ConvRot is a single file).
        let bad = LoadSpec {
            text_encoder: Some(WeightsSource::Dir("/te_dir".into())),
            ..LoadSpec::new(WeightsSource::Dir("/snap".into()))
        };
        assert!(
            convrot_selector(&bad, KREA_2_TURBO_ID).is_err(),
            "a Dir on text_encoder is a mis-shaped ConvRot selector and errors"
        );
    }

    #[test]
    fn load_accepts_convrot_and_rejects_convrot_with_overlays() {
        use candle_gen::gen_core::{AdapterKind, AdapterSpec};
        // A ConvRot-selecting spec loads (lazily — the int8 DiT + snapshot load at first `generate`).
        let convrot = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_convrot_text_encoder();
        assert!(
            load(&convrot).is_ok(),
            "a ConvRot LoadSpec is accepted + lazy (sc-9300)"
        );
        // ConvRot does not thread LoRA/LoKr — the int8 checkpoint replaces the dense DiT wholesale.
        let convrot_lora = LoadSpec::new(WeightsSource::Dir("/snap".into()))
            .with_convrot_text_encoder()
            .with_adapters(vec![AdapterSpec::new(
                "/lora.safetensors".into(),
                1.0,
                AdapterKind::Lora,
            )]);
        assert!(
            load(&convrot_lora).is_err(),
            "ConvRot + LoRA is rejected (the int8 DiT path is not adapter-wired)"
        );
        // ConvRot does not thread a PiD decoder overlay either.
        let convrot_pid = LoadSpec::new(WeightsSource::Dir("/snap".into()))
            .with_convrot_text_encoder()
            .with_pid(
                WeightsSource::File("/pid.safetensors".into()),
                WeightsSource::Dir("/gemma".into()),
            );
        assert!(
            load(&convrot_pid).is_err(),
            "ConvRot + PiD is rejected (the int8 DiT path is not PiD-wired)"
        );
    }

    // --- Edit (Kontext instruction edit, full-CFG) variant — epic 10871 / sc-11085 ---

    #[test]
    fn registers_krea_2_edit_as_candle() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry::load(KREA_2_EDIT_ID, &spec).expect("krea_2_edit is registered");
        assert_eq!(g.descriptor().id, KREA_2_EDIT_ID);
        assert_eq!(g.descriptor().family, "krea_2");
        assert_eq!(g.descriptor().backend, "candle");
        assert!(!g.descriptor().capabilities.mac_only);
    }

    #[test]
    fn edit_descriptor_is_cfg_capable_and_advertises_references() {
        let d = edit_descriptor();
        assert_eq!(d.id, KREA_2_EDIT_ID);
        assert_eq!(d.family, "krea_2");
        assert_eq!(d.backend, "candle");
        assert_eq!(d.modality, Modality::Image);
        // Derived from the Raw surface: real CFG guidance + a user negative prompt (the edit runs the
        // undistilled full-CFG loop with the references as in-context conditioning).
        assert!(d.capabilities.supports_guidance);
        assert!(d.capabilities.supports_negative_prompt);
        assert!(!d.capabilities.supports_true_cfg);
        // And it advertises BOTH single- and two-reference conditioning (Turbo/Raw advertise neither).
        assert_eq!(
            d.capabilities.conditioning,
            vec![
                ConditioningKind::Reference,
                ConditioningKind::MultiReference
            ]
        );
        assert!(descriptor().capabilities.conditioning.is_empty());
        assert!(raw_descriptor().capabilities.conditioning.is_empty());
        // Shared surface stays in lockstep with Raw/Turbo (derived from `raw_descriptor()`).
        assert!(d.capabilities.supports_lora && d.capabilities.supports_lokr);
        assert_eq!(d.capabilities.supported_quants, &[Quant::Q4, Quant::Q8]);
    }

    #[test]
    fn load_edit_rejects_single_file_accepts_dir_and_lora() {
        use candle_gen::gen_core::{AdapterKind, AdapterSpec};
        // Same snapshot loader as Turbo/Raw — a single-file weights source is rejected.
        let file = LoadSpec::new(WeightsSource::File("/tmp/x.safetensors".into()));
        assert!(load_edit(&file).is_err());
        // A plain snapshot dir loads lazily.
        let dir = LoadSpec::new(WeightsSource::Dir("/snap".into()));
        assert!(load_edit(&dir).is_ok());
        // The edit LoRA rides the shared `spec.adapters` merge path (accepted + lazy).
        let lora = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_adapters(vec![
            AdapterSpec::new("/edit_lora.safetensors".into(), 1.0, AdapterKind::Lora),
        ]);
        assert!(load_edit(&lora).is_ok(), "edit LoRA load is wired + lazy");
    }

    fn ref_image(w: u32, h: u32) -> Image {
        Image {
            width: w,
            height: h,
            pixels: vec![0u8; (w * h * 3) as usize],
        }
    }

    #[test]
    fn resolve_edit_references_single_and_pair_fixed_order() {
        // A single `Reference` → one source (the scene).
        let one = GenerationRequest {
            prompt: "make it autumn".into(),
            conditioning: vec![Conditioning::Reference {
                image: ref_image(2, 2),
                strength: None,
            }],
            ..Default::default()
        };
        let refs = resolve_edit_references(&one).unwrap();
        assert_eq!(refs.len(), 1);
        assert_eq!((refs[0].width, refs[0].height), (2, 2));

        // A two-image `MultiReference` → scene (image 1) then person (image 2), order preserved.
        let two = GenerationRequest {
            prompt: "put the person in the scene".into(),
            conditioning: vec![Conditioning::MultiReference {
                images: vec![ref_image(4, 4), ref_image(6, 6)],
            }],
            ..Default::default()
        };
        let refs = resolve_edit_references(&two).unwrap();
        assert_eq!(refs.len(), 2);
        assert_eq!((refs[0].width, refs[0].height), (4, 4), "scene = image 1");
        assert_eq!((refs[1].width, refs[1].height), (6, 6), "person = image 2");
    }

    #[test]
    fn resolve_edit_references_rejects_zero_and_over_cap() {
        // Zero references → error (an edit needs a source).
        let none = GenerationRequest {
            prompt: "make it autumn".into(),
            ..Default::default()
        };
        assert!(resolve_edit_references(&none).is_err());

        // Three references → past the fixed-order cap (scene, person).
        let three = GenerationRequest {
            prompt: "x".into(),
            conditioning: vec![Conditioning::MultiReference {
                images: vec![ref_image(2, 2), ref_image(2, 2), ref_image(2, 2)],
            }],
            ..Default::default()
        };
        let err = resolve_edit_references(&three).unwrap_err().to_string();
        assert!(err.contains("at most 2"), "got: {err}");
    }

    /// Test helper: attach a ConvRot DiT single-file selector on `text_encoder` (sc-9300).
    trait WithConvRot {
        fn with_convrot_text_encoder(self) -> Self;
    }
    impl WithConvRot for LoadSpec {
        fn with_convrot_text_encoder(mut self) -> Self {
            self.text_encoder = Some(WeightsSource::File(
                "/krea2_int8_convrot.safetensors".into(),
            ));
            self
        }
    }
}
