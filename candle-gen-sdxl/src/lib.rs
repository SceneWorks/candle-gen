//! # candle-gen-sdxl
//!
//! The **Stable Diffusion XL** provider crate for [`candle-gen`](candle_gen) — the candle
//! (Windows/CUDA) sibling of `mlx-gen-sdxl`. It implements the backend-neutral
//! [`gen_core::Generator`] contract and self-registers via `inventory`, so linking this crate
//! makes `gen_core::load("sdxl", …)` resolve the candle SDXL generator.
//!
//! **Phase 1 (sc-4946) is a scaffold:** the [`descriptor`] advertises the SDXL capability surface
//! (real classifier-free guidance: negative prompt + CFG scale; img2img/inpaint/control
//! conditioning; LoRA/LoKr; the Euler-Ancestral + few-step acceleration samplers), mirroring
//! `mlx-gen-sdxl`, but [`SdxlGenerator::generate`] is a stub that returns
//! [`gen_core::Error::Unsupported`]. The real candle UNet/CLIP/VAE pipeline lands in a later slice
//! (sc-3675). The descriptor's `backend` is `"candle"` and `mac_only` is `false` (this backend
//! targets Windows/CUDA — unlike the Mac-only MLX provider).

use candle_gen::gen_core::registry::ModelRegistration;
use candle_gen::gen_core::{
    self, Capabilities, ConditioningKind, GenerationOutput, GenerationRequest, Generator, LoadSpec,
    Modality, ModelDescriptor, Progress, Quant,
};

/// Registry id — matches the SceneWorks worker's `payload.model` (`MODEL_TABLE["sdxl"]`). The
/// worker maps both `sdxl` and `realvisxl` onto engine id `"sdxl"`, so — exactly like
/// `mlx-gen-sdxl` — this crate registers a SINGLE descriptor under `"sdxl"`.
pub const MODEL_ID: &str = "sdxl";

/// A loaded (scaffold) candle SDXL generator. Today it carries only its [`ModelDescriptor`]; the
/// real component fields (dual CLIP encoders, UNet, VAE, sampler) land with the pipeline slice.
pub struct SdxlGenerator {
    descriptor: ModelDescriptor,
}

impl Generator for SdxlGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        // The shared capability floor (count/size/cfg/sampler/conditioning) — model-specific
        // checks layer on top once the pipeline exists.
        self.descriptor.capabilities.validate_request(MODEL_ID, req)
    }

    fn generate(
        &self,
        _req: &GenerationRequest,
        _on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        Err(gen_core::Error::Unsupported(
            "candle SDXL pipeline not yet implemented (scaffold; sc-3675)".into(),
        ))
    }
}

/// SDXL's identity + capabilities — constructible without loading weights (registry
/// introspection). Mirrors `mlx-gen-sdxl`'s descriptor (real CFG, the same conditioning + sampler
/// surface) with two backend-correct deviations: `backend = "candle"` and `mac_only = false`.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID,
        family: "sdxl",
        // The tensor backend whose provider crate registered this engine (sc-3723). MLX sets
        // "mlx"; this is the candle sibling.
        backend: "candle",
        modality: Modality::Image,
        capabilities: Capabilities {
            // SDXL uses real classifier-free guidance: honors the negative prompt + a CFG scale.
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: false,
            // img2img Reference + masked inpaint/outpaint (Mask) + tile-ControlNet (Control) —
            // mirrors mlx-gen-sdxl. (Advertised surface; wired by the pipeline slice.)
            conditioning: vec![
                ConditioningKind::Reference,
                ConditioningKind::Mask,
                ConditioningKind::Control,
            ],
            supports_lora: true,
            supports_lokr: true,
            // `euler_ancestral` is the production default; `lcm`/`lightning`/`hyper` are the
            // few-step acceleration samplers (each paired with its acceleration LoRA at load).
            samplers: vec!["euler_ancestral", "lcm", "lightning", "hyper"],
            schedulers: vec!["discrete"],
            min_size: 512,
            max_size: 2048,
            max_count: 8,
            // candle is the Windows/CUDA backend — NOT Mac-only (the MLX provider sets this true).
            mac_only: false,
            // On-the-fly Q4/Q8, mirroring mlx-gen-sdxl's advertised surface (read by the worker
            // capability advertisement, sc-3723).
            supported_quants: &[Quant::Q4, Quant::Q8],
            supports_kv_cache: false,
            requires_sigma_shift: false,
        },
    }
}

/// Construct the (scaffold) candle SDXL generator. The real loader will read the SDXL diffusers
/// snapshot from `spec.weights`; today it only stamps the descriptor.
pub fn load(_spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    Ok(Box::new(SdxlGenerator {
        descriptor: descriptor(),
    }))
}

// Link-time self-registration into gen-core's model registry. Linking this crate makes
// `gen_core::load("sdxl", …)` resolve the candle generator — no central match statement to edit.
inventory::submit! {
    ModelRegistration { descriptor, load }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::registry;
    use candle_gen::gen_core::{LoadSpec, WeightsSource};

    /// The seam under test: this provider's `inventory::submit!` is linked into the test binary,
    /// so resolving `"sdxl"` through gen-core's registry returns OUR candle generator.
    #[test]
    fn sdxl_registers_and_resolves_as_candle() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry::load("sdxl", &spec).expect("candle sdxl is registered");
        assert_eq!(g.descriptor().id, "sdxl");
        assert_eq!(g.descriptor().backend, "candle");
        assert_eq!(g.descriptor().modality, Modality::Image);
    }

    #[test]
    fn descriptor_advertises_real_cfg() {
        let d = descriptor();
        assert!(d.capabilities.supports_negative_prompt);
        assert!(d.capabilities.supports_guidance);
        assert!(!d.capabilities.supports_true_cfg);
        assert!(!d.capabilities.mac_only);
    }

    /// The scaffold's `generate` is intentionally an `Unsupported` stub (the pipeline is sc-3675).
    #[test]
    fn generate_is_unsupported_scaffold() {
        let g = SdxlGenerator {
            descriptor: descriptor(),
        };
        let req = GenerationRequest {
            prompt: "a test".into(),
            ..Default::default()
        };
        let mut noop = |_p: Progress| {};
        let err = g.generate(&req, &mut noop).unwrap_err();
        assert!(matches!(err, gen_core::Error::Unsupported(_)));
    }
}
