//! SCAIL-2 provider: capability surface, registration, snapshot/config resolution, and the
//! [`Generator`] entrypoint — the candle (Windows/CUDA) sibling of `mlx-gen-scail2`'s pipeline.
//!
//! [`Generator::generate`] maps the [`GenerationRequest`] conditioning onto the SCAIL-2 inputs and runs
//! the live [`crate::generate`] denoise pipeline: the primary **reference character** is a
//! [`Conditioning::Reference`] image paired with its color-coded [`Conditioning::Mask`]; the **driving
//! video + per-frame color masks** are a `ControlClip`; `video_mode == "replacement"` toggles the
//! cross-identity `replace_flag` (else animation). Inference adapters (`spec.adapters`) — LoRA / LoKr /
//! LoHa, the lightx2v lightning diff-patch, and the Bias-Aware DPO refinement LoRA — are folded into the
//! dense DiT before build ([`crate::adapters`], sc-6838). Multi-reference awaits the worker request
//! contract (sc-5583: gen-core has no way to pair an extra reference image with its color-coded mask —
//! `Conditioning::MultiReference` carries images only); [`crate::generate`] already supports extra
//! characters via [`crate::generate::CharacterRef`], so until that contract lands `MultiReference` is
//! deliberately NOT advertised and [`Generator::validate`] rejects it loudly (sc-8985).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::{
    self, AdapterSpec, Capabilities, Conditioning, ConditioningKind, GenerationOutput,
    GenerationRequest, Generator, Image, LoadSpec, Modality, ModelDescriptor, Progress, Quant,
    WeightsSource,
};
use candle_gen::{CandleError, Result as CResult};
use candle_gen_wan::config::{TextEncoderConfig, Vae16Config, MAX_AREA_14B};
use candle_gen_wan::scheduler::Sampler;
use candle_gen_wan::text_encoder::Umt5Encoder;
use candle_gen_wan::vae16::WanVae16;

use crate::clip::{ClipVisionConfig, ScailClip};
use crate::config::Scail2Config;
use crate::generate::{CharacterRef, Components, Scail2Job};
use crate::model::Scail2Dit;

/// Default driving-segment window + clean-history overlap (upstream `scail.py` defaults).
const SEGMENT_LEN: usize = 81;
const SEGMENT_OVERLAP: usize = 5;
/// Upstream `generate()` sampler defaults: 40 steps, shift 5.0, guide 5.0, 16 fps.
const DEFAULT_STEPS: u32 = 40;
const DEFAULT_SHIFT: f32 = 5.0;
const DEFAULT_GUIDANCE: f32 = 5.0;
const DEFAULT_FPS: u32 = 16;

/// SceneWorks/engine model id (matches `mlx-gen-scail2` so a consumer resolves the same engine across
/// backends). A still image is `num_frames == 1`.
pub const MODEL_ID: &str = "scail2_14b";

/// Stable identity + advertised capabilities for SCAIL-2 (Wan2.1-14B I2V end-to-end character
/// animation: reference image + driving video + color-coded masks → animated/identity-replaced video;
/// plain single-scale CFG; packed-token conditioning + per-source RoPE + CLIP image cross-attn).
/// `backend = "candle"`, `mac_only = false` (the off-Mac CUDA lane).
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID,
        family: "scail2",
        backend: "candle",
        modality: Modality::Video,
        capabilities: Capabilities {
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: false,
            // Reference character image (Reference) + its color-coded segmentation mask (Mask); the
            // driving video + its per-frame color masks map to ControlClip. `MultiReference` (extra
            // characters) is deliberately NOT advertised: gen-core's `Conditioning::MultiReference`
            // carries images only, with no way to pair each extra reference with its required
            // color-coded mask, so the request contract can't reach `Scail2Job.additional` yet —
            // sc-5583 tracks the paired ref+mask contract + worker plumbing (sc-8985: advertising it
            // let multi-ref requests validate, render for minutes, and silently drop the extras).
            conditioning: vec![
                ConditioningKind::Reference,
                ConditioningKind::Mask,
                ConditioningKind::ControlClip,
            ],
            // Inference LoRA / LoKr / LoHa + the lightx2v lightning diff-patch + the Bias-Aware DPO
            // refinement LoRA, merged into the dense DiT before build (sc-6838,
            // [`crate::adapters::merge_adapters`]).
            supports_lora: true,
            supports_lokr: true,
            // candle's FlowScheduler is UniPC/Euler; "dpm++" resolves to UniPC (bh2). Advertised to
            // match the mlx-gen-scail2 descriptor for cross-backend routing parity.
            samplers: vec!["unipc", "dpm++"],
            schedulers: Vec::new(),
            supported_guidance_methods: vec![],
            min_size: 32,
            max_size: 1280,
            max_count: 1,
            mac_only: false,
            supported_quants: &[] as &[Quant],
            supports_kv_cache: false,
            requires_sigma_shift: false,
        },
    }
}

/// Load all `.safetensors` in the snapshot subdir `sub` as one f32 mmapped [`VarBuilder`].
fn component_vb(root: &Path, device: &Device, sub: &str) -> CResult<VarBuilder<'static>> {
    candle_gen::component_vb(root, sub, DType::F32, device, "scail2")
}

/// The loaded SCAIL-2 model: resolved config + snapshot dir, with the heavy components (DiT / VAE /
/// UMT5 / CLIP) loaded lazily on first generate and cached.
pub struct Scail2 {
    descriptor: ModelDescriptor,
    config: Scail2Config,
    root: PathBuf,
    device: Device,
    /// Inference adapters (LoRA / LoKr / LoHa / lightx2v lightning diff-patch) folded into the DiT
    /// before build; empty for the stock path (sc-6838).
    adapters: Vec<AdapterSpec>,
    components: Mutex<Option<Arc<Components>>>,
}

impl Scail2 {
    /// Build the DiT [`VarBuilder`] over the `transformer/` snapshot. With no adapters this is the
    /// stock f32 mmap build — **byte-identical** to the pre-sc-6838 path (the empty-adapter regression
    /// gate). With adapters, the base tensors are loaded to a CPU map, each delta is folded in
    /// ([`crate::adapters::merge_adapters`], f32 math — merge not residual, the chaos-sensitive-sampler
    /// rationale), the **whole map is cast to f32 on the CPU**, then the DiT is built from it.
    ///
    /// The host-side f32 cast is load-bearing for memory: SCAIL-2's DiT is f32, so a bf16 base tensor
    /// served through `from_tensors(F32, gpu)` would cast bf16→f32 *on the GPU*, and candle's CUDA
    /// caching allocator retains the freed bf16 staging blocks — ~28 GiB piled on top of the ~56 GiB
    /// f32 DiT, OOM-ing at the VAE-decode peak even on a 96 GiB card. Casting host-side (host RAM is
    /// ample, the map is transient) makes `get` a pure f32 host→device move, so the GPU footprint
    /// matches the stock mmap path exactly. (The Wan-14B merge path doesn't need this — its DiT is
    /// bf16, so `from_tensors` never casts on the GPU.)
    fn transformer_vb(&self) -> CResult<VarBuilder<'static>> {
        if self.adapters.is_empty() {
            return component_vb(&self.root, &self.device, "transformer");
        }
        let dir = self.root.join("transformer");
        let files = candle_gen::sorted_safetensors(&dir, "scail2")?;
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        for f in &files {
            let part = candle_gen::candle_core::safetensors::load(f, &Device::Cpu)?;
            tensors.extend(part);
        }
        // Discard the merge report — the silent twin (`candle-gen-z-image`'s
        // `transformer_vb_with_adapters`) does the same; a mismatched adapter surface already errors
        // inside `merge_adapters`, so library code stays quiet on stderr (sc-9035 / F-051).
        crate::adapters::merge_adapters(&mut tensors, &self.adapters)?;
        // Cast host-side so `from_tensors` does no GPU-side bf16→f32 staging (see the doc note above).
        for v in tensors.values_mut() {
            if v.dtype() != DType::F32 {
                *v = v.to_dtype(DType::F32)?;
            }
        }
        Ok(VarBuilder::from_tensors(tensors, DType::F32, &self.device))
    }

    fn load_components(&self) -> CResult<Components> {
        let te = Umt5Encoder::new(
            &TextEncoderConfig::umt5_xxl(),
            component_vb(&self.root, &self.device, "text_encoder")?,
        )?;
        let dit = Scail2Dit::new(self.transformer_vb()?, &self.config)?;
        let vae = WanVae16::new_with_encoder(
            &Vae16Config::wan21(),
            component_vb(&self.root, &self.device, "vae")?,
        )?;
        let clip = ScailClip::new(
            component_vb(&self.root, &self.device, "clip")?,
            &ClipVisionConfig::vit_h_14(),
        )?;
        let tok = crate::generate::build_tokenizer(&self.root, &TextEncoderConfig::umt5_xxl())?;
        Ok(Components {
            te,
            dit,
            vae,
            clip,
            tok,
        })
    }

    fn components(&self) -> CResult<Arc<Components>> {
        candle_gen::cached(&self.components, || Ok(Arc::new(self.load_components()?)))
    }
}

/// Construct a candle SCAIL-2 generator. `spec.weights` must be a [`WeightsSource::Dir`] pointing at a
/// snapshot with `text_encoder/`, `transformer/` (the converted SCAIL2Model DiT), `vae/` (z16 Wan VAE
/// with encoder), `clip/` (open-CLIP ViT-H/14 visual tower), and `tokenizer/tokenizer.json`. Inference
/// adapters (`spec.adapters` — LoRA / LoKr / LoHa / lightx2v lightning diff-patch / Bias-Aware DPO) are
/// merged into the dense DiT before build (sc-6838); on-the-fly quantization is still rejected.
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(
                "scail2: expected a snapshot directory (text_encoder/ transformer/ vae/ clip/ \
                 tokenizer/), not a single .safetensors file"
                    .into(),
            ));
        }
    };
    if spec.quantize.is_some() {
        return Err(gen_core::Error::Unsupported(
            "candle scail2 does not support on-the-fly Q4/Q8 quantization yet".into(),
        ));
    }
    if !root.exists() {
        return Err(gen_core::Error::Msg(format!(
            "scail2: snapshot dir does not exist: {}",
            root.display()
        )));
    }
    let config = Scail2Config::from_model_dir(&root)?;
    let device = candle_gen::default_device()?;
    Ok(Box::new(Scail2 {
        descriptor: descriptor(),
        config,
        root,
        device,
        adapters: spec.adapters.clone(),
        components: Mutex::new(None),
    }))
}

candle_gen::register_generators! { descriptor => load }

/// Force-link hook (keeps the `inventory::submit!` registration from being dead-stripped).
pub fn force_link() {}

impl Generator for Scail2 {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        // Actionable multi-reference rejection first, so the caller learns WHY (pending contract,
        // sc-5583) rather than the generic capability-floor "not supported" from `validate_request`
        // (which also rejects it now that `MultiReference` is unadvertised, sc-8985).
        reject_multi_reference(self.descriptor.id, req)?;
        reject_zero_steps(self.descriptor.id, req)?;
        reject_over_area(self.descriptor.id, req)?;
        self.descriptor
            .capabilities
            .validate_request(self.descriptor.id, req)
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.validate(req)?;
        Ok(self.run(req, on_progress)?)
    }
}

/// Reject `MultiReference` conditioning loudly instead of letting a multi-character request render
/// for minutes and silently drop the extra characters (sc-8985). The engine core already supports
/// extra characters ([`crate::generate::CharacterRef`] / `Scail2Job.additional`), but gen-core's
/// `Conditioning::MultiReference` carries images only — there is no way to pair each extra reference
/// with its required color-coded segmentation mask until the paired ref+mask request contract lands
/// (sc-5583).
fn reject_multi_reference(id: &str, req: &GenerationRequest) -> gen_core::Result<()> {
    if req
        .conditioning
        .iter()
        .any(|c| matches!(c, Conditioning::MultiReference { .. }))
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{id}: MultiReference (extra reference characters) is not supported yet — each extra \
             character needs its own color-coded segmentation mask and the paired reference+mask \
             request contract is pending (sc-5583); pass exactly one Reference + Mask"
        )));
    }
    Ok(())
}

/// Reject an explicit `steps: Some(0)` loudly instead of running zero denoise iterations and
/// VAE-decoding the pure prior — on video that is MINUTES of GPU time for garbage (sc-9016, F-032).
/// Mirrors the registered `SdxlGenerator::validate` steps floor; this worker-driven video path has no
/// gen-core steps floor upstream of it. A `None` legitimately falls through to `DEFAULT_STEPS`.
fn reject_zero_steps(id: &str, req: &GenerationRequest) -> gen_core::Result<()> {
    if req.steps == Some(0) {
        return Err(gen_core::Error::Msg(format!(
            "{id}: steps must be >= 1 (an explicit 0 renders undenoised noise)"
        )));
    }
    Ok(())
}

/// Reject an over-area request loudly instead of letting the 14B DiT run for minutes and OOM. SCAIL-2's
/// DiT runs **f32** (≈ 56 GiB resident) with a packed conditioning sequence >2× the plain token count,
/// so a far-over-envelope request (e.g. 1280×1280×81) validates and dies with an opaque CUDA OOM at the
/// VAE-decode peak. Reject past the shared A14B cap with an actionable message, mirroring the A14B MoE
/// lane (`wan14b.rs`, sc-9028 / F-044); the incident class F-090 (sc-11215) left this lane open. `max_size`
/// alone only bounds each edge, so 1280×1280 (both ≤ 1280) slips through without the area check.
fn reject_over_area(id: &str, req: &GenerationRequest) -> gen_core::Result<()> {
    let area = req.width as usize * req.height as usize;
    if area > MAX_AREA_14B {
        return Err(gen_core::Error::Msg(format!(
            "{id}: width×height ({}×{} = {area} px) exceeds the max area {MAX_AREA_14B} px \
             (704×1280); reduce the resolution",
            req.width, req.height
        )));
    }
    Ok(())
}

/// The first conditioning input matching `f`.
fn find_conditioning<'a, T>(
    req: &'a GenerationRequest,
    f: impl Fn(&'a Conditioning) -> Option<T>,
) -> Option<T> {
    req.conditioning.iter().find_map(f)
}

impl Scail2 {
    /// Map the request conditioning onto a [`Scail2Job`] and run the denoise pipeline.
    fn run(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> CResult<GenerationOutput> {
        let reference = find_conditioning(req, |c| match c {
            Conditioning::Reference { image, .. } => Some(image),
            _ => None,
        })
        .ok_or_else(|| {
            CandleError::Msg("scail2: a Reference character image is required".into())
        })?;
        let ref_mask = find_conditioning(req, |c| match c {
            Conditioning::Mask { image } => Some(image),
            _ => None,
        })
        .ok_or_else(|| {
            CandleError::Msg(
                "scail2: a Mask (the reference character's color-coded segmentation mask) is required"
                    .into(),
            )
        })?;
        let driving = req.control_clip().ok_or_else(|| {
            CandleError::Msg(
                "scail2: a ControlClip (driving video frames + per-frame color masks) is required"
                    .into(),
            )
        })?;

        let first: &Image = driving.frames.first().ok_or_else(|| {
            CandleError::Msg("scail2: the ControlClip has no driving frames".into())
        })?;
        let width = if req.width > 0 {
            req.width
        } else {
            first.width
        };
        let height = if req.height > 0 {
            req.height
        } else {
            first.height
        };

        let neg = req.negative_prompt.clone().unwrap_or_default();
        let job = Scail2Job {
            prompt: &req.prompt,
            negative_prompt: &neg,
            width,
            height,
            reference: CharacterRef {
                image: reference,
                mask: ref_mask,
            },
            // Extra characters await the paired ref+mask request contract (sc-5583); `validate`
            // rejects `MultiReference` until then (sc-8985).
            additional: Vec::new(),
            driving_frames: driving.frames,
            driving_masks: driving.mask,
            replace_flag: req.video_mode.as_deref() == Some("replacement"),
            seed: req.seed.unwrap_or_else(gen_core::default_seed),
            steps: req.steps.unwrap_or(DEFAULT_STEPS) as usize,
            shift: req.scheduler_shift.unwrap_or(DEFAULT_SHIFT) as f64,
            guidance: req.guidance.unwrap_or(DEFAULT_GUIDANCE) as f64,
            sampler: Sampler::parse(req.sampler.as_deref()),
            fps: req.fps.unwrap_or(DEFAULT_FPS),
            segment_len: SEGMENT_LEN,
            segment_overlap: SEGMENT_OVERLAP,
        };
        let comps = self.components()?;
        let te_cfg = TextEncoderConfig::umt5_xxl();
        crate::generate::generate(
            &comps,
            &te_cfg,
            &job,
            &|| req.cancel.is_cancelled(),
            on_progress,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::registry;

    #[test]
    fn registers_and_resolves_as_candle_video() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        // The snapshot dir doesn't exist, so `load` errors — but the engine must be REGISTERED (the
        // registry resolves the id to this provider's `load`).
        let err = registry::load(MODEL_ID, &spec).err().expect("dir missing");
        assert!(
            err.to_string().contains("does not exist"),
            "expected a missing-dir error from the scail2 loader, got: {err}"
        );
    }

    #[test]
    fn descriptor_surface() {
        let d = descriptor();
        assert_eq!(d.id, MODEL_ID);
        assert_eq!(d.family, "scail2");
        assert_eq!(d.backend, "candle");
        assert_eq!(d.modality, Modality::Video);
        assert!(!d.capabilities.mac_only);
        assert!(d.capabilities.supports_guidance);
        assert!(d.capabilities.supports_negative_prompt);
        assert!(!d.capabilities.supports_true_cfg);
        assert!(d.capabilities.supports_lora);
        assert!(d.capabilities.supports_lokr);
        assert!(d.capabilities.accepts(ConditioningKind::Reference));
        assert!(d.capabilities.accepts(ConditioningKind::Mask));
        assert!(d.capabilities.accepts(ConditioningKind::ControlClip));
        // MultiReference is deliberately NOT advertised until the paired ref+mask request contract
        // lands (sc-5583) — advertising it silently dropped the extra characters (sc-8985).
        assert!(!d.capabilities.accepts(ConditioningKind::MultiReference));
        assert!(d.capabilities.samplers.contains(&"unipc"));
    }

    #[test]
    fn multi_reference_is_rejected_loudly() {
        let img = Image {
            width: 64,
            height: 64,
            pixels: vec![0u8; 64 * 64 * 3],
        };
        let req = GenerationRequest {
            prompt: "a character".into(),
            width: 64,
            height: 64,
            count: 1,
            conditioning: vec![Conditioning::MultiReference {
                images: vec![img.clone(), img],
            }],
            ..Default::default()
        };
        // The dedicated guard fires with the actionable pending-contract message.
        let err = reject_multi_reference(MODEL_ID, &req).expect_err("err");
        assert!(matches!(err, gen_core::Error::Unsupported(_)), "got: {err}");
        let msg = err.to_string();
        assert!(msg.contains("MultiReference"), "got: {msg}");
        assert!(msg.contains("sc-5583"), "got: {msg}");
        // Backstop: with `MultiReference` unadvertised, the shared capability floor rejects it too.
        assert!(
            descriptor()
                .capabilities
                .validate_request(MODEL_ID, &req)
                .is_err(),
            "the capability floor must reject unadvertised MultiReference conditioning"
        );
        // A request without MultiReference passes the guard (the floor still enforces the rest).
        let single = GenerationRequest {
            conditioning: Vec::new(),
            ..req
        };
        assert!(reject_multi_reference(MODEL_ID, &single).is_ok());
    }

    #[test]
    fn zero_steps_is_rejected_loudly() {
        // An explicit `steps: Some(0)` is a fast, actionable error — NOT minutes of video decoded from
        // undenoised prior noise (sc-9016, F-032).
        let zero = GenerationRequest {
            prompt: "a character".into(),
            steps: Some(0),
            ..Default::default()
        };
        let err = reject_zero_steps(MODEL_ID, &zero).expect_err("steps==0 must be rejected");
        let msg = err.to_string();
        assert!(msg.contains("steps must be >= 1"), "got: {msg}");
        // A valid step count and an unset (default) step count both pass this guard.
        let valid = GenerationRequest {
            steps: Some(40),
            ..zero.clone()
        };
        assert!(reject_zero_steps(MODEL_ID, &valid).is_ok());
        let unset = GenerationRequest {
            steps: None,
            ..zero
        };
        assert!(reject_zero_steps(MODEL_ID, &unset).is_ok());
    }

    #[test]
    fn over_area_is_rejected_loudly() {
        // A far-over-envelope request (1280×1280, both edges ≤ `max_size` so `max_size` alone lets it
        // through) must be a fast, actionable rejection — NOT minutes of the f32 14B DiT running to an
        // opaque CUDA OOM (F-090 / sc-11215, mirroring the A14B MoE lane's sc-9028 guard).
        assert_eq!(704 * 1280, MAX_AREA_14B);
        let over = GenerationRequest {
            prompt: "a character".into(),
            width: 1280,
            height: 1280,
            ..Default::default()
        };
        let err = reject_over_area(MODEL_ID, &over).expect_err("over-area must be rejected");
        let msg = err.to_string();
        assert!(msg.contains("max area"), "message names the cap: {msg}");
        // Exactly at the cap (1280×704 = 901 120 px) and a small in-bounds request both pass the guard.
        let at_cap = GenerationRequest {
            width: 1280,
            height: 704,
            ..over.clone()
        };
        assert!(reject_over_area(MODEL_ID, &at_cap).is_ok());
        let small = GenerationRequest {
            width: 512,
            height: 512,
            ..over
        };
        assert!(reject_over_area(MODEL_ID, &small).is_ok());
    }

    #[test]
    fn load_rejects_single_file_and_unwired_surfaces() {
        use candle_gen::gen_core::{AdapterKind, AdapterSpec};
        // single-file source
        let f = LoadSpec::new(WeightsSource::File("/tmp/w.safetensors".into()));
        assert!(load(&f).is_err());
        // LoRA adapters are now ACCEPTED (sc-6838) — `load` proceeds past the adapter check and fails
        // only on the missing snapshot dir, NOT with an Unsupported("LoRA") error.
        let lora = LoadSpec::new(WeightsSource::Dir("/nonexistent".into())).with_adapters(vec![
            AdapterSpec::new("/lora.safetensors".into(), 1.0, AdapterKind::Lora),
        ]);
        let err = load(&lora).err().expect("missing dir");
        assert!(
            !matches!(err, gen_core::Error::Unsupported(_)),
            "got: {err}"
        );
        assert!(err.to_string().contains("does not exist"), "got: {err}");
        // on-the-fly quant is still rejected
        let quant = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_quant(Quant::Q8);
        assert!(matches!(
            load(&quant).err().expect("err"),
            gen_core::Error::Unsupported(_)
        ));
    }
}
