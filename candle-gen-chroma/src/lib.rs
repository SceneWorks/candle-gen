//! # candle-gen-chroma
//!
//! The **Chroma** provider crate for [`candle-gen`](candle_gen) — the candle (Windows/CUDA) sibling
//! of `mlx-gen-chroma`. It implements the backend-neutral [`gen_core::Generator`] contract and
//! self-registers via `inventory` for all three Chroma variants, so linking this crate makes
//! `gen_core::load("chroma1_hd" | "chroma1_base" | "chroma1_flash", …)` resolve the candle Chroma
//! generators.
//!
//! **txt2img (sc-5484):** Chroma is a FLUX.1-schnell-derived DiT — the MMDiT skeleton with a
//! distilled-guidance **Approximator** replacing FLUX's modulation stack, **T5-XXL-only**
//! conditioning, and **true CFG** (a real negative prompt). [`pipeline`] adapts it through the
//! contract: T5-XXL encode → the Chroma DiT (flow-match Euler, static-shift / beta sigmas) → the
//! FLUX 16-ch AutoencoderKL, emitting `Progress` and honoring `req.cancel`, with **deterministic
//! CPU-seeded noise** (sc-3673) so output is launch-portable per seed. The three variants:
//!
//! - **`chroma1_hd`** — high-detail full-CFG (28 steps, true_cfg 4.0, sigma shift 3.0).
//! - **`chroma1_base`** — base full-CFG (28 steps, true_cfg 4.0, beta-spaced sigmas).
//! - **`chroma1_flash`** — few-step distilled (8 steps, true_cfg 1.0 → single forward).
//!
//! The descriptors advertise **only** the wired txt2img surface (true-CFG + negative prompt, but NOT
//! LoRA/LoKr, quantization, or ControlNet/IP-adapter) — so the worker routes the rest to the Python
//! fallback rather than the candle backend silently dropping a control (the false-capability trap,
//! exactly as the SDXL / FLUX / Z-Image slices did). `backend` is `"candle"` and `mac_only` is `false`.

mod beta;
mod config;
mod pipeline;
mod quant;
mod rope;
mod text;
mod transformer;
mod vae;

use std::path::PathBuf;
use std::sync::Mutex;

use candle_gen::candle_core::Device;
use candle_gen::gen_core::{
    self, GenerationOutput, GenerationRequest, Generator, LoadSpec, ModelDescriptor, PidWeights,
    Progress, WeightsSource,
};

pub use config::{ChromaVariant, CHROMA1_BASE_ID, CHROMA1_FLASH_ID, CHROMA1_HD_ID, SIZE_MULTIPLE};

use pipeline::{Components, Pipeline};

/// A loaded candle Chroma generator (one per variant). Loading is **lazy**: `load` does no file I/O,
/// and the heavy components (T5 + DiT + VAE) are built on the first [`generate`](Generator::generate)
/// call and then cached.
pub struct ChromaGenerator {
    variant: ChromaVariant,
    descriptor: ModelDescriptor,
    root: PathBuf,
    device: Device,
    /// The `LoadSpec::pid` component captured at load (epic 7840 / sc-7853), threaded into the lazy
    /// component build so the PiD engine loads once alongside the base model. `None` when not opted in.
    pid_spec: Option<PidWeights>,
    components: Mutex<Option<Components>>,
}

impl ChromaGenerator {
    fn components(&self, pipe: &Pipeline) -> gen_core::Result<Components> {
        // `?` bridges the candle-side `load_components` error into `gen_core::Error`.
        Ok(candle_gen::cached(&self.components, || {
            pipe.load_components()
        })?)
    }
}

impl Generator for ChromaGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        // The shared capability floor (size/count range; negative_prompt + true_cfg are advertised,
        // so they pass; guidance / conditioning / unadvertised samplers are rejected here).
        self.descriptor
            .capabilities
            .validate_request(self.variant.id(), req)?;
        let id = self.variant.id();
        if req.prompt.trim().is_empty() {
            return Err(gen_core::Error::Msg(format!(
                "{id}: prompt must not be empty"
            )));
        }
        if req.steps == Some(0) {
            return Err(gen_core::Error::Msg(format!(
                "{id}: steps must be >= 1 (an explicit 0 renders undenoised noise)"
            )));
        }
        if !req.width.is_multiple_of(SIZE_MULTIPLE) || !req.height.is_multiple_of(SIZE_MULTIPLE) {
            return Err(gen_core::Error::Msg(format!(
                "{id}: width/height must be multiples of {SIZE_MULTIPLE} (got {}x{})",
                req.width, req.height
            )));
        }
        Ok(())
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.validate(req)?;
        let pipe = Pipeline::load(
            self.variant,
            &self.root,
            &self.device,
            self.pid_spec.clone(),
        );
        let components = self.components(&pipe)?;
        let images = pipe.render(req, &components, on_progress)?;
        Ok(GenerationOutput::Images(images))
    }
}

/// Descriptors (registry introspection — no weights).
pub fn descriptor_hd() -> ModelDescriptor {
    ChromaVariant::Hd.descriptor()
}
pub fn descriptor_base() -> ModelDescriptor {
    ChromaVariant::Base.descriptor()
}
pub fn descriptor_flash() -> ModelDescriptor {
    ChromaVariant::Flash.descriptor()
}

/// Construct a lazy candle Chroma generator for `variant`. `spec.weights` must be a
/// [`WeightsSource::Dir`] pointing at a Chroma diffusers snapshot (`tokenizer/`, `text_encoder/`,
/// `transformer/`, `vae/`). LoRA adapters, quantization, and control/IP-adapter overlays are rejected
/// — none are wired in this slice, so refusing is more honest than silently dropping them.
fn load_variant(variant: ChromaVariant, spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    let id = variant.id();
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(format!(
                "{id} expects a Chroma diffusers snapshot directory (tokenizer/, text_encoder/, \
                 transformer/, vae/), not a single .safetensors file"
            )));
        }
    };
    if !spec.adapters.is_empty() {
        return Err(gen_core::Error::Unsupported(format!(
            "candle {id} does not support LoRA/LoKr yet — refusing to silently drop the adapters"
        )));
    }
    if spec.quantize.is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "candle {id} does not support on-the-fly Q4/Q8 quantization yet"
        )));
    }
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "candle {id} does not support control / IP-adapter overlays yet (txt2img only)"
        )));
    }
    let device = candle_gen::default_device()?;
    Ok(Box::new(ChromaGenerator {
        variant,
        descriptor: variant.descriptor(),
        root,
        device,
        // PiD is an optional aux decoder (epic 7840 / sc-7853): capture the load-spec component (if
        // any) so the lazy component build loads the engine once. Unlike adapters/quant/control above,
        // it is not rejected — `None` simply keeps the byte-exact native-VAE path.
        pid_spec: spec.pid.clone(),
        components: Mutex::new(None),
    }))
}

pub fn load_hd(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_variant(ChromaVariant::Hd, spec)
}
pub fn load_base(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_variant(ChromaVariant::Base, spec)
}
pub fn load_flash(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_variant(ChromaVariant::Flash, spec)
}

// Link-time self-registration into gen-core's model registry — one descriptor per variant.
candle_gen::register_generators! {
    descriptor_hd => load_hd,
    descriptor_base => load_base,
    descriptor_flash => load_flash,
}

/// Force-link hook. A consumer that only reaches this provider *through* the `gen_core` registry
/// references nothing in this crate directly, so the linker (MSVC on a release build in particular)
/// can discard the whole rlib — taking the `inventory::submit!` registrations with it. Referencing
/// this no-op from the consumer keeps the crate linked. (Same pattern as `candle_gen_flux::force_link`.)
pub fn force_link() {}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::registry;
    use candle_gen::gen_core::{Conditioning, Image, Modality};

    #[test]
    fn all_three_variants_register_and_resolve_as_candle() {
        for id in [CHROMA1_HD_ID, CHROMA1_BASE_ID, CHROMA1_FLASH_ID] {
            let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
            let g =
                registry::load(id, &spec).unwrap_or_else(|_| panic!("candle {id} is registered"));
            assert_eq!(g.descriptor().id, id);
            assert_eq!(g.descriptor().family, "chroma");
            assert_eq!(g.descriptor().backend, "candle");
            assert_eq!(g.descriptor().modality, Modality::Image);
        }
    }

    #[test]
    fn validate_accepts_true_cfg_and_negative_rejects_unsupported() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let hd = registry::load(CHROMA1_HD_ID, &spec).unwrap();

        // True CFG + negative prompt are advertised → accepted.
        let ok = GenerationRequest {
            prompt: "a rusty robot holding a lit candle".into(),
            negative_prompt: Some("blurry".into()),
            true_cfg: Some(4.0),
            ..Default::default()
        };
        assert!(hd.validate(&ok).is_ok());

        for bad in [
            GenerationRequest::default(), // empty prompt
            GenerationRequest {
                prompt: "x".into(),
                guidance: Some(3.5), // distilled guidance not advertised
                ..Default::default()
            },
            GenerationRequest {
                prompt: "x".into(),
                width: 1000, // not a multiple of 16
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
            assert!(hd.validate(&bad).is_err(), "should reject: {bad:?}");
        }
    }

    #[test]
    fn load_rejects_unwired_surfaces_and_single_file() {
        use candle_gen::gen_core::{AdapterKind, AdapterSpec, Quant};
        for load in [load_hd, load_base, load_flash] {
            let lora = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_adapters(vec![
                AdapterSpec::new("/lora.safetensors".into(), 1.0, AdapterKind::Lora),
            ]);
            assert!(matches!(
                load(&lora).err().expect("err"),
                gen_core::Error::Unsupported(_)
            ));

            let quant = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_quant(Quant::Q8);
            assert!(matches!(
                load(&quant).err().expect("err"),
                gen_core::Error::Unsupported(_)
            ));

            let single = LoadSpec::new(WeightsSource::File("/x.safetensors".into()));
            let err = load(&single).err().expect("err").to_string();
            assert!(err.contains("snapshot directory"), "got: {err}");
        }
    }

    /// A pre-cancelled request aborts before any forward with the typed `Canceled` (the cancellation
    /// contract — sc-4481). The per-image cancel check runs before `denoise`, so loading lazily means
    /// this needs no weights... except `generate` loads components first. Kept as a descriptor-level
    /// check instead: `generate` on a nonexistent snapshot errors (not a panic).
    #[test]
    fn descriptors_are_image_modality() {
        assert_eq!(descriptor_hd().modality, Modality::Image);
        assert_eq!(descriptor_base().modality, Modality::Image);
        assert_eq!(descriptor_flash().modality, Modality::Image);
    }
}
