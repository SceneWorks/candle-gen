//! # candle-gen-seedvr2
//!
//! The **SeedVR2** provider crate for [`candle-gen`](candle_gen) — the Windows/CUDA sibling of
//! `mlx-gen-seedvr2` (epic 4811 / sc-5157). A native-candle port of the ByteDance one-step
//! diffusion-transformer super-resolution upscaler:
//!
//! 1. **DiT** ([`dit`]) — a dual-stream MMDiT with adaptive **spatiotemporal window attention**
//!    (`window=(T,H,W)=(4,3,3)`, shifted on odd layers), 3D axial RoPE, QK-norm, SwiGLU, AdaLN.
//! 2. **3D causal video VAE** ([`vae`]) — `CausalConv3d` (candle has no conv3d → conv2d temporal-sum,
//!    see [`conv3d`]) encoder/decoder with `temporal_down/up_blocks=2`, GroupNorm, per-frame attention.
//! 3. **One-step Euler** + a precomputed negative-prompt embedding (bundled, no runtime text encoder).
//!
//! **This slice (sc-5157): the 3B engine + image-mode upscale.** Registered under `seedvr2` (alias) +
//! `seedvr2_3b`, `Modality::Image`, a `Reference` LR image → `GenerationOutput::Images`. Video mode
//! (the 5-D temporal pass) is sc-5926; 7B + int8/int4 quant is sc-5927; worker wiring/gating is
//! sc-5928. `backend = "candle"`, `mac_only = false`.

pub mod color;
pub mod config;
pub mod conv3d;
pub mod convert;
pub mod dit;
pub mod nn;
pub mod pipeline;
pub mod vae;
pub mod weights;

use std::path::PathBuf;
use std::sync::Mutex;

use candle_gen::candle_core::{DType, Device};
use candle_gen::gen_core::registry::ModelRegistration;
use candle_gen::gen_core::{
    self, default_seed, Capabilities, Conditioning, ConditioningKind, GenerationOutput,
    GenerationRequest, Generator, Image, LoadSpec, Modality, ModelDescriptor, Precision, Progress,
    WeightsSource,
};

use config::{DitConfig, VAE_SCALE};
use pipeline::Seedvr2Pipeline;

pub const MODEL_ID: &str = "seedvr2";
pub const MODEL_ID_3B: &str = "seedvr2_3b";
const DIT_FILE_3B: &str = "seedvr2_ema_3b_fp16.safetensors";

fn descriptor_for(id: &'static str) -> ModelDescriptor {
    ModelDescriptor {
        id,
        family: "seedvr2",
        backend: "candle",
        modality: Modality::Image, // video (VideoClip) lands in sc-5926
        capabilities: Capabilities {
            supports_negative_prompt: false, // precomputed neg-embed; no prompt surface
            supports_guidance: false,        // one-step, guidance fixed at 1.0
            supports_true_cfg: false,
            conditioning: vec![ConditioningKind::Reference], // the LR input image
            supports_lora: false,
            supports_lokr: false,
            samplers: vec!["seedvr2_euler"],
            schedulers: vec!["seedvr2_euler"],
            min_size: VAE_SCALE,
            max_size: 4096,
            max_count: 8,
            mac_only: false,
            supports_kv_cache: false,
            requires_sigma_shift: false,
            supported_quants: &[], // Linear-only DiT quant is sc-5927
        },
    }
}

pub fn descriptor() -> ModelDescriptor {
    descriptor_for(MODEL_ID)
}
pub fn descriptor_3b() -> ModelDescriptor {
    descriptor_for(MODEL_ID_3B)
}

/// The lazy candle SeedVR2 generator (one-step image upscaler). The pipeline is loaded on first
/// `generate` and cached behind a `Mutex` for the worker's `Arc<dyn Generator>` reuse.
pub struct Seedvr2Generator {
    descriptor: ModelDescriptor,
    root: PathBuf,
    device: Device,
    dtype: DType,
    pipe: Mutex<Option<std::sync::Arc<Seedvr2Pipeline>>>,
}

/// The LR input image carried by the request's `Reference` conditioning.
fn reference_image(req: &GenerationRequest) -> Option<&Image> {
    req.conditioning.iter().find_map(|c| match c {
        Conditioning::Reference { image, .. } => Some(image),
        _ => None,
    })
}

impl Seedvr2Generator {
    fn pipeline(&self) -> gen_core::Result<std::sync::Arc<Seedvr2Pipeline>> {
        let mut guard = self.pipe.lock().expect("seedvr2 pipeline mutex poisoned");
        if let Some(p) = guard.as_ref() {
            return Ok(p.clone());
        }
        let cfg = DitConfig::seedvr2_3b();
        let p = std::sync::Arc::new(Seedvr2Pipeline::load(
            &self.root,
            DIT_FILE_3B,
            &cfg,
            self.dtype,
            &self.device,
        )?);
        *guard = Some(p.clone());
        Ok(p)
    }
}

impl Generator for Seedvr2Generator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        self.descriptor
            .capabilities
            .validate_request(self.descriptor.id, req)?;
        if reference_image(req).is_none() {
            return Err(gen_core::Error::Msg(format!(
                "{}: image upscale requires a Reference (LR input) image",
                self.descriptor.id
            )));
        }
        if !req.width.is_multiple_of(VAE_SCALE) || !req.height.is_multiple_of(VAE_SCALE) {
            return Err(gen_core::Error::Msg(format!(
                "{}: width/height must be multiples of {VAE_SCALE} (got {}x{})",
                self.descriptor.id, req.width, req.height
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
        let image = reference_image(req).expect("validated");
        let pipe = self.pipeline()?;
        let base_seed = req.seed.unwrap_or_else(default_seed);
        let softness = req.softness.unwrap_or(0.0);

        let mut out = Vec::with_capacity(req.count as usize);
        for i in 0..req.count {
            if req.cancel.is_cancelled() {
                return Err(gen_core::Error::Canceled);
            }
            on_progress(Progress::Step {
                current: 1,
                total: 1,
            });
            let seed = base_seed.wrapping_add(i as u64);
            let img = pipe.generate(
                image,
                req.width as usize,
                req.height as usize,
                seed,
                softness,
            )?;
            on_progress(Progress::Decoding);
            out.push(img);
        }
        Ok(GenerationOutput::Images(out))
    }
}

/// Construct a lazy candle SeedVR2 generator. `spec.weights` must be a [`WeightsSource::Dir`] pointing
/// at a raw `numz/SeedVR2_comfyUI` checkpoint dir (`ema_vae_fp16.safetensors` + the 3B DiT file).
/// Adapters / quantization (sc-5927) / control overlays are rejected.
fn load_with(spec: &LoadSpec, id: &'static str) -> gen_core::Result<Box<dyn Generator>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(format!(
                "{id}: expects a numz/SeedVR2_comfyUI checkpoint directory, not a single file"
            )))
        }
    };
    if !spec.adapters.is_empty() {
        return Err(gen_core::Error::Unsupported(format!(
            "{id}: LoRA/LoKr adapters are not part of SeedVR2"
        )));
    }
    if spec.quantize.is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "{id}: int8/int4 quantization lands in sc-5927"
        )));
    }
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "{id}: ControlNet / IP-Adapter conditioning is not part of SeedVR2"
        )));
    }
    let device = candle_gen::default_device()?;
    // Precision::Bf16 is the "native dense" sentinel → bf16 on GPU, f32 on CPU (the parity dtype);
    // Fp32 forces full precision everywhere.
    let dtype = match spec.precision {
        Precision::Fp32 => DType::F32,
        Precision::Bf16 if device.is_cpu() => DType::F32,
        Precision::Bf16 => DType::BF16,
    };
    Ok(Box::new(Seedvr2Generator {
        descriptor: descriptor_for(id),
        root,
        device,
        dtype,
        pipe: Mutex::new(None),
    }))
}

fn load_registered(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_with(spec, MODEL_ID)
}
fn load_registered_3b(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_with(spec, MODEL_ID_3B)
}

inventory::submit! {
    ModelRegistration { descriptor, load: load_registered }
}
inventory::submit! {
    ModelRegistration { descriptor: descriptor_3b, load: load_registered_3b }
}

/// Force-link hook (keeps the `inventory::submit!` registration from being dead-stripped).
pub fn force_link() {}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::registry;

    #[test]
    fn descriptor_is_seedvr2_image() {
        let d = descriptor();
        assert_eq!(d.id, MODEL_ID);
        assert_eq!(d.family, "seedvr2");
        assert_eq!(d.backend, "candle");
        assert_eq!(d.modality, Modality::Image);
        assert!(!d.capabilities.mac_only);
        assert!(!d.capabilities.supports_guidance);
        assert!(d.capabilities.accepts(ConditioningKind::Reference));
        assert_eq!(d.capabilities.min_size, VAE_SCALE);
    }

    #[test]
    fn both_ids_resolve_in_registry() {
        for id in [MODEL_ID, MODEL_ID_3B] {
            let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent/seedvr2".into()));
            let g = registry::load(id, &spec).expect("seedvr2 is registered");
            assert_eq!(g.descriptor().family, "seedvr2");
            assert_eq!(g.descriptor().backend, "candle");
        }
    }

    #[test]
    fn load_rejects_unwired_surfaces() {
        use candle_gen::gen_core::Quant;
        let quant = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_quant(Quant::Q8);
        assert!(matches!(
            load_with(&quant, MODEL_ID).err().expect("err"),
            gen_core::Error::Unsupported(_)
        ));
        let file = LoadSpec::new(WeightsSource::File("/w.safetensors".into()));
        assert!(load_with(&file, MODEL_ID).is_err());
    }
}
