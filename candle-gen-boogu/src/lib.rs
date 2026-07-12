//! # candle-gen-boogu
//!
//! The **Boogu-Image-0.1** provider crate for [`candle-gen`](candle_gen) — the candle (Windows/CUDA)
//! sibling of `mlx-gen-boogu`. Registers three engine ids:
//!
//! * **`boogu_image`** — the Base variant: a 10.3B Lumina-Image-2.0 / OmniGen2-lineage hybrid MMDiT
//!   (8 double + 32 single + 2 refiner layers, GQA, 3-axis interleaved RoPE) with true-CFG, driven by
//!   a Qwen3-VL-8B condition encoder and a FLUX.1 16-channel VAE. 50-step rectified-flow Euler over a
//!   static-shift (`mu = 1.15`) schedule, routed through the unified curated-sampler framework.
//! * **`boogu_image_turbo`** — the same Base weights-arch + a DMD-distilled few-step (4) sampler loop,
//!   CFG-free (guidance inert). The default fast surface.
//! * **`boogu_image_edit`** — text+image-to-image with one or more reference images (sc-7523 single,
//!   sc-7645 multi up to 5): each source ([`ConditioningKind::Reference`] /
//!   [`ConditioningKind::MultiReference`]) is VAE-encoded into the DiT's spatial reference sequence
//!   (`forward_edit`) **and** read by the Qwen3-VL **vision tower** so the MLLM "sees" it
//!   (image-conditioned instruction features). Same true-CFG / static-shift schedule as Base.
//!
//! **Reuse:** the FLUX.1 VAE is `candle-transformers`' `z_image::vae::AutoEncoderKL` (the exact 16-ch
//! AutoencoderKL Z-Image ships, scaling 0.3611 / shift 0.1159) — reused verbatim, as `mlx-gen-boogu`
//! reuses `mlx-gen-z-image`'s `Vae`. The Qwen3-VL-8B condition encoder, its vision tower, and the
//! hybrid DiT are ported here.
//!
//! `backend = "candle"`, `mac_only = false`. Apache-2.0, ungated.

pub mod config;
pub mod loader;
pub mod pipeline;
pub mod quant;
pub mod text_encoder;
pub mod tokenizer;
pub mod transformer;
pub mod vision;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::Device;
use candle_gen::gen_core::{
    self, Capabilities, Conditioning, ConditioningKind, GenerationOutput, GenerationRequest,
    Generator, Image, LoadSpec, Modality, ModelDescriptor, PidWeights, Progress, Quant,
    WeightsSource,
};

use pipeline::{Components, EditComponents};

/// Registry id for the Base text-to-image variant (true-CFG).
pub const BOOGU_IMAGE_ID: &str = "boogu_image";
/// Registry id for the Turbo variant (DMD few-step, CFG-free).
pub const BOOGU_IMAGE_TURBO_ID: &str = "boogu_image_turbo";
/// Registry id for the instruction image-edit variant (single- or multi-reference TI2I).
pub const BOOGU_IMAGE_EDIT_ID: &str = "boogu_image_edit";

/// Patch(2)·ae_scale(8) = 16 — `patchify` requires latent dims divisible by this.
const SIZE_MULTIPLE: u32 = 16;

/// Maximum reference images the Edit lane accepts — the DiT's `image_index_embedding` row count (the
/// OmniGen2-lineage `[5, hidden]` parameter supports up to 5 distinct reference index slots).
const MAX_EDIT_REFERENCES: usize = 5;

/// The curated samplers the Turbo DMD student stays coherent under (the stochastic / re-noising
/// solvers — `lcm` most of all; real-weight survey sc-7491). The student was distilled against a
/// stochastic (re-noised) trajectory, so the curated stochastic solvers match its training regime;
/// the deterministic ODE solvers feed the few-step student out-of-regime latents, so they stay off
/// the menu. A selected name routes `render_turbo` through the unified `run_flow_sampler` over the
/// DMD σ grid (sc-9009); unset stays the byte-exact native DMD loop. Mirrors `mlx-gen-boogu`'s
/// `TURBO_SAMPLERS`.
const TURBO_SAMPLERS: &[&str] = &["lcm", "euler_ancestral", "dpmpp_sde"];

/// Which Boogu sampler path a generator drives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Variant {
    /// Base — true-CFG text-to-image.
    Base,
    /// Turbo — CFG-free DMD few-step text-to-image.
    Turbo,
    /// Edit — TI2I (true-CFG) with one or more reference images VAE-encoded + vision-conditioned.
    Edit,
}

/// A lazily-loaded Boogu generator. [`Variant`] selects the sampler path. The shared T2I components
/// load on the first `generate`; the Edit-only components (vision tower + VAE encoder) load lazily on
/// the first edit, so the T2I paths keep their footprint.
pub struct BooguGenerator {
    descriptor: ModelDescriptor,
    root: PathBuf,
    variant: Variant,
    device: Device,
    /// The `LoadSpec::pid` component captured at load (epic 7840 / sc-7853), threaded into the lazy
    /// component build so the PiD engine loads once alongside the base model. `None` when not opted in.
    pid_spec: Option<PidWeights>,
    components: Mutex<Option<Arc<Components>>>,
    edit_components: Mutex<Option<Arc<EditComponents>>>,
}

impl BooguGenerator {
    fn components(&self) -> gen_core::Result<Arc<Components>> {
        candle_gen::cached(&self.components, || {
            Ok(Arc::new(pipeline::load_components(
                &self.root,
                &self.device,
                self.pid_spec.as_ref(),
            )?))
        })
    }

    fn edit_components(&self) -> gen_core::Result<Arc<EditComponents>> {
        candle_gen::cached(&self.edit_components, || {
            Ok(Arc::new(pipeline::load_edit_components(
                &self.root,
                &self.device,
            )?))
        })
    }
}

impl Generator for BooguGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        let id = self.descriptor.id;
        self.descriptor.capabilities.validate_request(id, req)?;
        if req.prompt.trim().is_empty() {
            return Err(gen_core::Error::Msg(format!(
                "{id}: prompt must not be empty"
            )));
        }
        if req.steps == Some(0) {
            return Err(gen_core::Error::Msg(format!("{id}: steps must be >= 1")));
        }
        if !req.width.is_multiple_of(SIZE_MULTIPLE) || !req.height.is_multiple_of(SIZE_MULTIPLE) {
            return Err(gen_core::Error::Msg(format!(
                "{id}: width/height must be multiples of {SIZE_MULTIPLE} (got {}x{})",
                req.width, req.height
            )));
        }
        // The Edit variant needs 1..=5 source references; the capability floor already rejects any
        // Reference/MultiReference on Base/Turbo (their `conditioning` surface is empty).
        if self.variant == Variant::Edit {
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
        let images = match self.variant {
            Variant::Turbo => pipeline::render_turbo(&comps, req, &self.device, on_progress)?,
            Variant::Base => pipeline::render_base(&comps, req, &self.device, on_progress)?,
            Variant::Edit => {
                let references = resolve_edit_references(req)?;
                let edit = self.edit_components()?;
                pipeline::render_edit(&comps, &edit, req, &references, &self.device, on_progress)?
            }
        };
        Ok(GenerationOutput::Images(images))
    }
}

/// The img2img/instruction-edit source images, in order — collected from both
/// [`Conditioning::Reference`] (single) and [`Conditioning::MultiReference`] (multi). At least one and
/// at most [`MAX_EDIT_REFERENCES`] (the DiT's `image_index_embedding` row count) is required; zero or
/// more than the cap is an error.
fn resolve_edit_references(req: &GenerationRequest) -> gen_core::Result<Vec<&Image>> {
    let mut refs: Vec<&Image> = Vec::new();
    for c in &req.conditioning {
        match c {
            Conditioning::Reference { image, .. } => refs.push(image),
            Conditioning::MultiReference { images } => refs.extend(images.iter()),
            _ => {} // the capability floor already rejects other conditioning kinds.
        }
    }
    if refs.is_empty() {
        return Err(gen_core::Error::Msg(
            "boogu_image_edit: an instruction edit requires at least one source reference image"
                .into(),
        ));
    }
    if refs.len() > MAX_EDIT_REFERENCES {
        return Err(gen_core::Error::Msg(format!(
            "boogu_image_edit: at most {MAX_EDIT_REFERENCES} reference images are supported (got {})",
            refs.len()
        )));
    }
    Ok(refs)
}

/// Boogu Base descriptor — true-CFG text-to-image; no user negative prompt (the CFG-negative is the
/// model's own fixed empty/drop instruction); no img2img/control conditioning on the Base checkpoint.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: BOOGU_IMAGE_ID,
        family: "boogu",
        backend: "candle",
        modality: Modality::Image,
        capabilities: Capabilities {
            supports_negative_prompt: false,
            supports_guidance: true,
            supports_true_cfg: false,
            conditioning: Vec::new(),
            supports_lora: false,
            supports_lokr: false,
            // Base is rectified-flow Euler over the static-shift schedule, routed through the unified
            // curated-sampler framework (epic 7114).
            samplers: candle_gen::curated_sampler_names(),
            schedulers: candle_gen::curated_scheduler_names(),
            supported_guidance_methods: vec![],
            min_size: 256,
            max_size: 2048,
            max_count: 8,
            mac_only: false,
            // sc-9607: advertise the packed tiers so the worker's A-B quant toggle engages off-Mac.
            // The resolved `base/`-`-q4/`-`-bf16/` turnkey subdir self-describes its tier
            // (`loader::linear_detect`, sc-9410, group-size-aware); `build` no-ops the requested quant.
            // Turbo + edit inherit this via `descriptor()`.
            supported_quants: &[Quant::Q4, Quant::Q8],
            supports_kv_cache: false,
            requires_sigma_shift: false,
        },
    }
}

/// Boogu Turbo descriptor — same base, CFG-free DMD few-step; guidance is inert. The advertised
/// sampler menu is the DMD-compatible stochastic subset ([`TURBO_SAMPLERS`]); a selected sampler or
/// scheduler routes the few-step denoise through the unified curated framework over the DMD σ grid
/// (sc-9009), while unset keeps the byte-exact native DMD student loop.
pub fn descriptor_turbo() -> ModelDescriptor {
    let mut d = descriptor();
    d.id = BOOGU_IMAGE_TURBO_ID;
    d.capabilities.supports_guidance = false;
    d.capabilities.samplers = TURBO_SAMPLERS.to_vec();
    d
}

/// Boogu Edit descriptor — same true-CFG surface as the Base path plus one or more img2img/instruction
/// -edit source images ([`ConditioningKind::Reference`] for a single source, or
/// [`ConditioningKind::MultiReference`] for up to [`MAX_EDIT_REFERENCES`]): each source is read by the
/// Qwen3-VL vision tower (semantic edit) and VAE-encoded into the DiT's spatial reference sequence.
pub fn descriptor_edit() -> ModelDescriptor {
    let mut d = descriptor();
    d.id = BOOGU_IMAGE_EDIT_ID;
    d.capabilities.conditioning = vec![
        ConditioningKind::Reference,
        ConditioningKind::MultiReference,
    ];
    d
}

fn build(
    spec: &LoadSpec,
    descriptor: ModelDescriptor,
    variant: Variant,
) -> gen_core::Result<Box<dyn Generator>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(format!(
                "{} expects a snapshot directory (mllm/ transformer/ vae/), not a single \
                 .safetensors file",
                descriptor.id
            )));
        }
    };
    if !spec.adapters.is_empty() {
        return Err(gen_core::Error::Unsupported(format!(
            "candle {} does not accept user LoRA/LoKr adapters",
            descriptor.id
        )));
    }
    // sc-9607: `spec.quantize` (Q4/Q8) is ACCEPTED and no-ops — the resolved per-tier turnkey is
    // already MLX-packed and `loader::linear_detect` builds each `QLinear::Quantized` straight from the
    // packed parts (sc-9410, group-size-aware), so no on-the-fly quant pass runs. Advertising
    // `supported_quants` lets the worker's A-B tier toggle engage; the requested quant is recipe-only.
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "candle {} does not support ControlNet / IP-Adapter overlays",
            descriptor.id
        )));
    }
    let device = candle_gen::default_device()?;
    Ok(Box::new(BooguGenerator {
        descriptor,
        root,
        variant,
        device,
        // PiD is an optional aux decoder (epic 7840 / sc-7853): capture the load-spec component (if
        // any) so the lazy component build loads the engine once. Unlike adapters/control above, it is
        // not rejected — `None` simply keeps the byte-exact native-VAE path.
        pid_spec: spec.pid.clone(),
        components: Mutex::new(None),
        edit_components: Mutex::new(None),
    }))
}

/// Construct a lazy candle Boogu **Base** generator. `spec.weights` must be a [`WeightsSource::Dir`]
/// pointing at a candle-readable (bf16) Boogu snapshot (`mllm/ transformer/ vae/`).
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    build(spec, descriptor(), Variant::Base)
}

/// Construct a lazy candle Boogu **Turbo** generator (DMD few-step, CFG-free).
pub fn load_turbo(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    build(spec, descriptor_turbo(), Variant::Turbo)
}

/// Construct a lazy candle Boogu **Edit** generator (single-reference TI2I, true-CFG).
pub fn load_edit(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    build(spec, descriptor_edit(), Variant::Edit)
}

candle_gen::register_generators! {
    descriptor => load,
    descriptor_turbo => load_turbo,
    descriptor_edit => load_edit,
}

/// Force-link hook (keeps the `inventory::submit!` registrations from being dead-stripped).
pub fn force_link() {}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::registry;

    #[test]
    fn registers_all_three_ids_as_candle() {
        for id in [BOOGU_IMAGE_ID, BOOGU_IMAGE_TURBO_ID, BOOGU_IMAGE_EDIT_ID] {
            let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
            let g = registry::load(id, &spec).unwrap_or_else(|_| panic!("{id} is registered"));
            assert_eq!(g.descriptor().id, id);
            assert_eq!(g.descriptor().family, "boogu");
            assert_eq!(g.descriptor().backend, "candle");
            assert!(!g.descriptor().capabilities.mac_only);
        }
    }

    #[test]
    fn descriptor_surfaces() {
        let b = descriptor();
        assert!(b.capabilities.supports_guidance);
        assert!(!b.capabilities.supports_negative_prompt);
        assert!(b.capabilities.conditioning.is_empty());
        // sc-9607: packed tiers advertised so the worker A-B toggle engages; turbo + edit inherit it.
        assert_eq!(b.capabilities.supported_quants, &[Quant::Q4, Quant::Q8]);
        let t = descriptor_turbo();
        assert_eq!(t.id, BOOGU_IMAGE_TURBO_ID);
        assert!(!t.capabilities.supports_guidance);
        assert_eq!(t.capabilities.samplers, TURBO_SAMPLERS.to_vec());
        assert_eq!(t.capabilities.supported_quants, &[Quant::Q4, Quant::Q8]);
        assert_eq!(
            descriptor_edit().capabilities.supported_quants,
            &[Quant::Q4, Quant::Q8]
        );
    }

    #[test]
    fn turbo_advertised_menu_is_honored_curated_names() {
        // sc-9009: every advertised Turbo sampler must be a real curated solver name — the routing in
        // `render_turbo` hands `req.sampler` to `run_flow_sampler`, whose N3 fallback silently
        // substitutes Euler for an unknown name, which would resurrect the silent-ignore trap.
        let curated = candle_gen::curated_sampler_names();
        for s in TURBO_SAMPLERS {
            assert!(
                curated.contains(s),
                "advertised turbo sampler {s:?} is not a curated solver: {curated:?}"
            );
        }
        // The scheduler axis is advertised (inherited from Base) and honored by the same routing.
        let t = descriptor_turbo();
        assert_eq!(
            t.capabilities.schedulers,
            candle_gen::curated_scheduler_names()
        );
    }

    #[test]
    fn descriptor_edit_adds_reference() {
        let d = descriptor_edit();
        assert_eq!(d.id, BOOGU_IMAGE_EDIT_ID);
        assert!(d.capabilities.supports_guidance);
        // Edit advertises both single- and multi-reference conditioning.
        assert!(d
            .capabilities
            .conditioning
            .contains(&ConditioningKind::Reference));
        assert!(d
            .capabilities
            .conditioning
            .contains(&ConditioningKind::MultiReference));
        // Base/Turbo keep an empty conditioning surface (only Edit advertises references).
        assert!(descriptor().capabilities.conditioning.is_empty());
        assert!(descriptor_turbo().capabilities.conditioning.is_empty());
    }

    #[test]
    fn edit_validate_reference_count() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry::load(BOOGU_IMAGE_EDIT_ID, &spec).unwrap();
        let img = |w: u32, h: u32| Image {
            width: w,
            height: h,
            pixels: vec![0u8; (w * h * 3) as usize],
        };
        let one_ref = || Conditioning::Reference {
            image: img(512, 512),
            strength: None,
        };
        let base = GenerationRequest {
            prompt: "make it autumn".into(),
            width: 512,
            height: 512,
            ..Default::default()
        };
        // No reference → error.
        assert!(g.validate(&base).is_err());
        // A single `Reference` → ok.
        let one = GenerationRequest {
            conditioning: vec![one_ref()],
            ..base.clone()
        };
        assert!(g.validate(&one).is_ok());
        // Two references (now supported, up to 5) → ok.
        let two = GenerationRequest {
            conditioning: vec![one_ref(), one_ref()],
            ..base.clone()
        };
        assert!(g.validate(&two).is_ok());
        // A `MultiReference` with the max 5 images → ok.
        let five = GenerationRequest {
            conditioning: vec![Conditioning::MultiReference {
                images: (0..5).map(|_| img(512, 512)).collect(),
            }],
            ..base.clone()
        };
        assert!(g.validate(&five).is_ok());
        // Six references → error (past the `image_index_embedding` cap).
        let six = GenerationRequest {
            conditioning: vec![Conditioning::MultiReference {
                images: (0..6).map(|_| img(512, 512)).collect(),
            }],
            ..base
        };
        assert!(g.validate(&six).is_err());
    }

    #[test]
    fn base_rejects_reference_conditioning() {
        // Base has no conditioning surface, so the capability floor rejects a Reference.
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry::load(BOOGU_IMAGE_ID, &spec).unwrap();
        let r = GenerationRequest {
            prompt: "x".into(),
            width: 512,
            height: 512,
            conditioning: vec![Conditioning::Reference {
                image: Image {
                    width: 512,
                    height: 512,
                    pixels: vec![0u8; 512 * 512 * 3],
                },
                strength: None,
            }],
            ..Default::default()
        };
        assert!(g.validate(&r).is_err());
    }

    #[test]
    fn validate_accepts_txt2img_and_rejects_bad() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry::load(BOOGU_IMAGE_ID, &spec).unwrap();
        let ok = GenerationRequest {
            prompt: "a red apple on a wooden table".into(),
            guidance: Some(4.0),
            ..Default::default()
        };
        assert!(g.validate(&ok).is_ok());
        for bad in [
            GenerationRequest::default(),
            GenerationRequest {
                prompt: "x".into(),
                width: 1000,
                ..Default::default()
            },
            GenerationRequest {
                prompt: "x".into(),
                steps: Some(0),
                ..Default::default()
            },
        ] {
            assert!(g.validate(&bad).is_err(), "should reject: {bad:?}");
        }
    }

    /// F-154 (sc-11210): the empty-prompt guard rejects a whitespace-only prompt (`trim().is_empty()`),
    /// matching the chroma/krea-control siblings — a whitespace prompt otherwise reaches the TE as an
    /// effectively-empty sequence.
    #[test]
    fn validate_rejects_whitespace_only_prompt() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry::load(BOOGU_IMAGE_ID, &spec).unwrap();
        for ws in ["   ", "\t", "\n", " \t\n "] {
            let req = GenerationRequest {
                prompt: ws.into(),
                guidance: Some(4.0),
                ..Default::default()
            };
            assert!(
                g.validate(&req).is_err(),
                "whitespace-only prompt {ws:?} must be rejected"
            );
        }
    }

    #[test]
    fn load_rejects_single_file_and_unwired_surfaces() {
        use candle_gen::gen_core::{AdapterKind, AdapterSpec};
        let file = LoadSpec::new(WeightsSource::File("/tmp/q.safetensors".into()));
        assert!(load(&file).is_err());
        let lora = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_adapters(vec![
            AdapterSpec::new("/lora.safetensors".into(), 1.0, AdapterKind::Lora),
        ]);
        assert!(matches!(
            load(&lora).err().expect("err"),
            gen_core::Error::Unsupported(_)
        ));
    }
}
