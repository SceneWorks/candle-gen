//! The **Wan-VACE** controllable-video provider (`wan_vace`, Wan2.1-VACE-14B) — the candle
//! (Windows/CUDA) sibling of `mlx-gen-wan`'s `wan_vace`. Registers as `backend = "candle"`,
//! [`Modality::Video`].
//!
//! VACE is **mode-agnostic at the engine boundary**, exactly like diffusers `WanVACEPipeline`: the
//! SceneWorks worker builds the per-mode control video + mask (replace_person = the person-region-
//! neutralized clip + the person mask; extend/bridge = the source frames at the kept positions + a
//! generated-span mask) and passes them as one [`Conditioning::ControlClip`]. This provider
//! VAE-encodes the inactive/reactive split + unfolds the mask into the 96-channel control latent
//! ([`crate::vace::prepare_video_latents`] / [`prepare_masks`](crate::vace::prepare_masks)) and runs
//! the CFG VACE denoise loop ([`denoise_vace`](crate::vace::denoise_vace)). Reference images (from
//! [`Conditioning::Reference`]) become leading latent frames and are dropped after denoise (diffusers
//! `latents[:, :, num_reference_images:]`).
//!
//! **Snapshot layout** (diffusers): `transformer/` (the VACE DiT, diffusers tensor names), `text_encoder/`
//! (UMT5-XXL), `vae/` (the z16 Wan VAE — needs the encoder for the control encode), `tokenizer/`.
//!
//! **Dtypes:** UMT5 + VAE run **f32**; the VACE DiT runs **bf16** (norms/modulation upcast to f32) —
//! the candle Wan regime. LoRA / on-the-fly quantization are **deferred** (rejected at load).

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::{
    self, Capabilities, Conditioning, ConditioningKind, GenerationOutput, GenerationRequest,
    Generator, Image, LoadSpec, Modality, ModelDescriptor, Progress, Quant, WeightsSource,
};
use candle_gen::{CandleError, Result as CResult};

use crate::config::{
    TextEncoderConfig, Vae16Config, WanVaceConfig, DEFAULT_FPS_VACE, DEFAULT_GUIDANCE_VACE,
    DEFAULT_STEPS_VACE, MAX_AREA_14B, MODEL_ID_VACE, NEGATIVE_FALLBACK, SIZE_MULTIPLE_14B,
    VACE_FLOW_SHIFT, VAE16_STRIDE_TEMPORAL,
};
use crate::pipeline::{create_noise, frames_to_images};
use crate::rope::WanRope;
use crate::scheduler::Sampler;
use crate::text_encoder::Umt5Encoder;
use crate::vace::{
    build_vace_control, denoise_vace, prepare_masks, prepare_video_latents, WanVaceTransformer,
};
use crate::vae16::WanVae16;
use crate::wan14b::preprocess_i2v_image;

const DIT_DTYPE: DType = DType::BF16;
const ENC_DTYPE: DType = DType::F32;
const VAE_DTYPE: DType = DType::F32;
const Z_DIM: usize = 16;
const VAE_T: usize = VAE16_STRIDE_TEMPORAL as usize;

#[derive(Clone)]
struct Components {
    te: Arc<Umt5Encoder>,
    dit: Arc<WanVaceTransformer>,
    vae: Arc<WanVae16>,
    /// UMT5 tokenizer, loaded+parsed **once** at component load and reused across encodes (sc-8991 /
    /// F-011) instead of re-parsing `tokenizer.json` per prompt/branch.
    tok: Arc<candle_gen::gen_core::tokenizer::TextTokenizer>,
}

struct Pipeline {
    te_cfg: TextEncoderConfig,
    vace_cfg: WanVaceConfig,
    vae_cfg: Vae16Config,
    root: PathBuf,
    device: Device,
}

impl Pipeline {
    fn load(root: &Path, device: &Device) -> Self {
        Self {
            te_cfg: TextEncoderConfig::umt5_xxl(),
            vace_cfg: WanVaceConfig::vace_14b(),
            vae_cfg: Vae16Config::wan21(),
            root: root.to_path_buf(),
            device: device.clone(),
        }
    }

    fn component_vb(&self, sub: &str, dtype: DType) -> CResult<VarBuilder<'static>> {
        // Shared Wan component loader (sc-9000 / F-020); the crafted snapshot description stays local.
        crate::text_encode::component_vb(
            &self.root,
            sub,
            dtype,
            &self.device,
            "wan-vace",
            "Wan2.1-VACE-14B diffusers",
        )
    }

    fn load_components(&self) -> CResult<Components> {
        let te = Umt5Encoder::new(&self.te_cfg, self.component_vb("text_encoder", ENC_DTYPE)?)?;
        let dit =
            WanVaceTransformer::new(&self.vace_cfg, self.component_vb("transformer", DIT_DTYPE)?)?;
        // The control encode needs the VAE encoder.
        let vae = WanVae16::new_with_encoder(&self.vae_cfg, self.component_vb("vae", VAE_DTYPE)?)?;
        let tok = crate::text_encode::build_umt5_tokenizer(&self.root, &self.te_cfg, "wan-vace")?;
        Ok(Components {
            te: Arc::new(te),
            dit: Arc::new(dit),
            vae: Arc::new(vae),
            tok: Arc::new(tok),
        })
    }

    /// Tokenize + UMT5-encode `prompt` → `[1, 512, 4096]` (f32), zero-padded to `max_length` (the DiT
    /// cross-attends over the 512-padded context — the same rule as the base Wan). Shared Wan
    /// text-encode routine (sc-9000 / F-020).
    fn encode(&self, comps: &Components, prompt: &str) -> CResult<Tensor> {
        crate::text_encode::umt5_encode_padded(
            &comps.tok,
            &self.te_cfg,
            &comps.te,
            prompt,
            &self.device,
            ENC_DTYPE,
            "wan-vace",
        )
    }

    /// Stack a list of frame [`Image`]s → a `[1, 3, F, H, W]` clip in `[-1, 1]` (the Wan VAE input
    /// convention), via the per-frame cover-fit resize + center-crop.
    fn preprocess_clip(&self, frames: &[Image], width: u32, height: u32) -> CResult<Tensor> {
        if frames.is_empty() {
            return Err(CandleError::Msg(
                "wan-vace: control clip has no frames".into(),
            ));
        }
        let planes: Vec<Tensor> = frames
            .iter()
            .map(|im| preprocess_i2v_image(im, width, height, &self.device)) // [1,3,1,H,W]
            .collect::<CResult<_>>()?;
        let refs: Vec<&Tensor> = planes.iter().collect();
        Ok(Tensor::cat(&refs, 2)?) // [1,3,F,H,W]
    }

    fn render(
        &self,
        req: &GenerationRequest,
        comps: &Components,
        on_progress: &mut dyn FnMut(Progress),
    ) -> CResult<(Vec<Image>, u32)> {
        let clip = req
            .control_clip()
            .ok_or_else(|| CandleError::Msg("wan-vace: requires a ControlClip".into()))?;
        let width = req.width;
        let height = req.height;
        let steps = req
            .steps
            .map(|s| s as usize)
            .unwrap_or(DEFAULT_STEPS_VACE as usize);
        let shift = req
            .scheduler_shift
            .map(|s| s as f64)
            .unwrap_or(VACE_FLOW_SHIFT);
        let sampler = Sampler::parse(req.sampler.as_deref());
        let seed = req.seed.unwrap_or_else(gen_core::default_seed);
        let guidance = req.guidance.unwrap_or(DEFAULT_GUIDANCE_VACE) as f64;
        let cfg_disabled = guidance <= 1.0;
        let fps = req.fps.unwrap_or(DEFAULT_FPS_VACE);

        // Control video [-1,1] + mask [0,1] (diffusers `clamp((m+1)/2)`), each [1,3,F,H,W].
        let control_video = self.preprocess_clip(clip.frames, width, height)?;
        let mask = self.preprocess_clip(clip.mask, width, height)?;
        let mask = ((mask + 1.0)? * 0.5)?; // (m+1)/2 ∈ [0,1]

        // Reference images (optional) → [1,3,1,H,W] each.
        let references: Vec<Tensor> = req
            .conditioning
            .iter()
            .filter_map(|c| match c {
                Conditioning::Reference { image, .. } => Some(image),
                _ => None,
            })
            .map(|im| preprocess_i2v_image(im, width, height, &self.device))
            .collect::<CResult<_>>()?;
        let num_ref = references.len();

        // Stage 1: UMT5 text encode + project to the DiT context.
        let pos = self.encode(comps, &req.prompt)?;
        let ctx_pos = comps.dit.embed_text(&pos)?;
        let ctx_neg = if cfg_disabled {
            None
        } else {
            let neg = req.negative_prompt.as_deref().unwrap_or(NEGATIVE_FALLBACK);
            Some(comps.dit.embed_text(&self.encode(comps, neg)?)?)
        };

        // Stage 2: z16 VAE-encode the control + mask → the 96-ch control latent.
        let patch_h = self.vace_cfg.base.patch.1;
        let video_latents = prepare_video_latents(&comps.vae, &control_video, &mask, &references)?;
        let mask_latents = prepare_masks(&mask, patch_h, num_ref)?;
        let control = build_vace_control(&video_latents, &mask_latents)?;
        let (_, _, t_total, h_lat, w_lat) = control.dims5()?;

        // Per-vace-layer control scale (diffusers `conditioning_scale`); default 1.0.
        let scales = vec![req.control_scale.unwrap_or(1.0); self.vace_cfg.vace_layers.len()];

        // RoPE for the (ref-extended) token grid.
        let (pt, ph, pw) = self.vace_cfg.base.patch;
        let (ppf, pph, ppw) = (t_total / pt, h_lat / ph, w_lat / pw);
        let (cos, sin) = WanRope::new(&self.vace_cfg.base).cos_sin(ppf, pph, ppw, &self.device)?;

        // Seeded init noise [1, 16, T_total, h, w].
        let init_noise = create_noise(seed, Z_DIM, t_total, h_lat, w_lat, &self.device)?;

        // Stage 3: CFG VACE denoise.
        let total = steps as u32;
        let mut on_step = |i: usize| {
            on_progress(Progress::Step {
                current: i as u32,
                total,
            })
        };
        let latents = denoise_vace(
            &comps.dit,
            &control,
            &scales,
            sampler,
            steps,
            shift,
            guidance,
            &ctx_pos,
            ctx_neg.as_ref(),
            &init_noise,
            &cos,
            &sin,
            &req.cancel,
            &mut on_step,
        )?;

        // Drop the leading reference latent frames (diffusers `latents[:, :, num_reference_images:]`).
        let latents = if num_ref > 0 {
            latents.narrow(2, num_ref, t_total - num_ref)?
        } else {
            latents
        };

        // Stage 4: z16 VAE decode → RGB frames.
        on_progress(Progress::Decoding);
        let decoded = comps.vae.decode(&latents)?;
        let images = frames_to_images(&decoded)?;
        Ok((images, fps))
    }
}

/// A loaded Wan-VACE generator. Heavy components (UMT5, the VACE DiT, the z16 VAE) are loaded lazily on
/// the first `generate` and cached.
pub struct WanVaceGenerator {
    descriptor: ModelDescriptor,
    root: PathBuf,
    device: Device,
    components: Mutex<Option<Components>>,
}

impl WanVaceGenerator {
    fn components(&self, pipe: &Pipeline) -> gen_core::Result<Components> {
        // `cached` recovers a poisoned lock (sc-9015) internally; `?` bridges the candle-side
        // `load_components` error into `gen_core::Error`.
        Ok(candle_gen::cached(&self.components, || {
            pipe.load_components()
        })?)
    }
}

impl Generator for WanVaceGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        self.descriptor
            .capabilities
            .validate_request(MODEL_ID_VACE, req)?;
        if req.prompt.is_empty() {
            return Err(gen_core::Error::Msg(
                "wan-vace: prompt must not be empty".into(),
            ));
        }
        if req.steps == Some(0) {
            return Err(gen_core::Error::Msg("wan-vace: steps must be >= 1".into()));
        }
        if !req.width.is_multiple_of(SIZE_MULTIPLE_14B)
            || !req.height.is_multiple_of(SIZE_MULTIPLE_14B)
        {
            return Err(gen_core::Error::Msg(format!(
                "wan-vace: width/height must be multiples of {SIZE_MULTIPLE_14B} (got {}x{})",
                req.width, req.height
            )));
        }
        // The 14B VACE DiT (14B params + a 96-ch control stream) fails opaquely (OOM) far over the
        // envelope. Reject past the shared A14B cap with an actionable message, mirroring the A14B
        // MoE lane (`wan14b.rs`, sc-9028 / F-044); the incident class F-090 (sc-11215) left open here.
        let area = req.width as usize * req.height as usize;
        if area > MAX_AREA_14B {
            return Err(gen_core::Error::Msg(format!(
                "wan-vace: width×height ({}×{} = {area} px) exceeds the max area {MAX_AREA_14B} px \
                 (704×1280); reduce the resolution",
                req.width, req.height
            )));
        }
        let clip = req.control_clip().ok_or_else(|| {
            gen_core::Error::Msg(
                "wan-vace: needs a ControlClip (the masked control video — the worker builds it per \
                 mode: replace_person / extend / bridge)"
                    .into(),
            )
        })?;
        if clip.frames.len() != clip.mask.len() {
            return Err(gen_core::Error::Msg(format!(
                "wan-vace: control frames ({}) and mask frames ({}) length mismatch",
                clip.frames.len(),
                clip.mask.len()
            )));
        }
        // Control clip frame count must be 1 + 4·k (one z16 VAE temporal chunk + groups of 4).
        if clip.frames.len() % VAE_T != 1 {
            return Err(gen_core::Error::Msg(format!(
                "wan-vace: control clip frame count must be 1 + 4·k (got {})",
                clip.frames.len()
            )));
        }
        // The VACE output frame count is driven solely by the ControlClip (`render` derives the
        // temporal length from `clip.frames`, never `req.frames`). A request carrying a `frames`
        // that disagrees with the clip would silently render the clip's length instead — the F-043
        // silently-ignored-input class (sc-9027). Reject the disagreement with an actionable error.
        if let Some(f) = req.frames {
            if f as usize != clip.frames.len() {
                return Err(gen_core::Error::Msg(format!(
                    "wan-vace: req.frames ({f}) disagrees with the ControlClip frame count ({}); \
                     the VACE output length is driven by the control clip — omit `frames` or set it \
                     to {}",
                    clip.frames.len(),
                    clip.frames.len()
                )));
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
        let pipe = Pipeline::load(&self.root, &self.device);
        let components = self.components(&pipe)?;
        let (frames, fps) = pipe.render(req, &components, on_progress)?;
        Ok(GenerationOutput::Video {
            frames,
            fps,
            audio: None,
        })
    }
}

/// Wan-VACE descriptor — CFG (guidance + negative prompt), UniPC/Euler samplers, a `ControlClip` (the
/// universal VACE form the worker builds per mode) + optional `Reference` images. LoRA / quant deferred.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID_VACE,
        family: "wan",
        backend: "candle",
        modality: Modality::Video,
        capabilities: Capabilities {
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: false,
            conditioning: vec![ConditioningKind::ControlClip, ConditioningKind::Reference],
            supports_lora: false,
            supports_lokr: false,
            // Curated `uni_pc` (sc-7296) → Wan's native UniPC; `euler` flow Euler. Legacy `unipc` alias.
            samplers: vec!["uni_pc", "euler", "unipc"],
            schedulers: vec![],
            supported_guidance_methods: vec![],
            min_size: 16,
            max_size: 1280,
            max_count: 1,
            mac_only: false,
            supported_quants: &[] as &[Quant],
            supports_kv_cache: false,
            requires_sigma_shift: false,
        },
    }
}

/// Construct a lazy candle Wan-VACE generator. `spec.weights` must be a [`WeightsSource::Dir`] pointing
/// at a Wan2.1-VACE-14B diffusers snapshot (`text_encoder/`, `transformer/`, `vae/`, `tokenizer/`).
/// LoRA / on-the-fly quantization are rejected (not wired).
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(
                "wan_vace expects a snapshot directory (text_encoder/ transformer/ vae/ tokenizer/), \
                 not a single .safetensors file"
                    .into(),
            ));
        }
    };
    if !spec.adapters.is_empty() {
        return Err(gen_core::Error::Unsupported(
            "candle wan_vace does not support LoRA/LoKr yet".into(),
        ));
    }
    if spec.quantize.is_some() {
        return Err(gen_core::Error::Unsupported(
            "candle wan_vace does not support on-the-fly Q4/Q8 quantization yet".into(),
        ));
    }
    let device = candle_gen::default_device()?;
    Ok(Box::new(WanVaceGenerator {
        descriptor: descriptor(),
        root,
        device,
        components: Mutex::new(None),
    }))
}

candle_gen::register_generators! { descriptor => load }

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::registry;
    use candle_gen::gen_core::ReplacementMode;

    fn control_req() -> GenerationRequest {
        let frame = Image {
            width: 64,
            height: 64,
            pixels: vec![0u8; 64 * 64 * 3],
        };
        GenerationRequest {
            prompt: "a person walking".into(),
            width: 64,
            height: 64,
            guidance: Some(5.0),
            conditioning: vec![Conditioning::ControlClip {
                frames: vec![
                    frame.clone(),
                    frame.clone(),
                    frame.clone(),
                    frame.clone(),
                    frame.clone(),
                ],
                mask: vec![
                    frame.clone(),
                    frame.clone(),
                    frame.clone(),
                    frame.clone(),
                    frame,
                ],
                masking_strength: 1.0,
                start_frame: 0,
                mode: ReplacementMode::default(),
            }],
            ..Default::default()
        }
    }

    #[test]
    fn registers_as_candle_video() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry::load(MODEL_ID_VACE, &spec).expect("wan_vace is registered");
        assert_eq!(g.descriptor().id, MODEL_ID_VACE);
        assert_eq!(g.descriptor().family, "wan");
        assert_eq!(g.descriptor().backend, "candle");
        assert_eq!(g.descriptor().modality, Modality::Video);
        assert!(!g.descriptor().capabilities.mac_only);
    }

    #[test]
    fn descriptor_surface() {
        let d = descriptor();
        assert!(d.capabilities.supports_guidance);
        assert!(d.capabilities.supports_negative_prompt);
        assert!(d.capabilities.accepts(ConditioningKind::ControlClip));
        assert!(d.capabilities.accepts(ConditioningKind::Reference));
        assert!(d.capabilities.samplers.contains(&"uni_pc")); // curated spelling (sc-7296)
        assert!(d.capabilities.samplers.contains(&"unipc")); // legacy alias retained
        assert!(!d.capabilities.supports_lora);
    }

    #[test]
    fn validate_accepts_control_clip_and_rejects_bad_shapes() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry::load(MODEL_ID_VACE, &spec).unwrap();
        assert!(g.validate(&control_req()).is_ok());

        // No control clip.
        let mut no_clip = control_req();
        no_clip.conditioning.clear();
        assert!(g.validate(&no_clip).is_err());

        // Frame count not 1 + 4·k (4 frames).
        let frame = Image {
            width: 64,
            height: 64,
            pixels: vec![0u8; 64 * 64 * 3],
        };
        let mut bad_count = control_req();
        bad_count.conditioning = vec![Conditioning::ControlClip {
            frames: vec![frame.clone(), frame.clone(), frame.clone(), frame.clone()],
            mask: vec![frame.clone(), frame.clone(), frame.clone(), frame.clone()],
            masking_strength: 1.0,
            start_frame: 0,
            mode: ReplacementMode::default(),
        }];
        assert!(g.validate(&bad_count).is_err());

        // Size not a multiple of 16.
        let mut bad_size = control_req();
        bad_size.width = 70;
        assert!(g.validate(&bad_size).is_err());
    }

    /// F-124 (sc-11220): the VACE output length is driven solely by the ControlClip. A `req.frames`
    /// that agrees with the 5-frame clip passes; one that disagrees is rejected (rather than silently
    /// rendering the clip's length — the F-043 silently-ignored-input class).
    #[test]
    fn validate_rejects_frames_disagreeing_with_clip() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry::load(MODEL_ID_VACE, &spec).unwrap();

        // Agrees with the 5-frame control clip → accepted.
        let mut ok = control_req();
        ok.frames = Some(5);
        assert!(g.validate(&ok).is_ok());

        // Disagrees → rejected with a message that names both counts.
        let mut bad = control_req();
        bad.frames = Some(33);
        let err = g
            .validate(&bad)
            .expect_err("frames disagreeing with the clip must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("33") && msg.contains('5'),
            "names counts: {msg}"
        );
    }

    /// The shared A14B area cap is enforced on the VACE lane too (F-090 / sc-11215): an at-cap request
    /// passes and a grid-aligned over-cap request is rejected by the area check with an actionable
    /// message that names the cap — mirroring `wan14b.rs`'s `validate_enforces_max_area` (sc-9028).
    #[test]
    fn validate_enforces_max_area() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry::load(MODEL_ID_VACE, &spec).unwrap();

        // Exactly at the cap (1280×704 = 901 120 px, both multiples of 16) is accepted.
        assert_eq!(704 * 1280, MAX_AREA_14B);
        let mut at_cap = control_req();
        at_cap.width = 1280;
        at_cap.height = 704;
        assert!(g.validate(&at_cap).is_ok());

        // Over the cap while both edges stay within `max_size` (1280×1280 = 1 638 400 px, both
        // grid-aligned and ≤ 1280) is rejected specifically by the area check, with an actionable
        // message naming the cap.
        let mut over = control_req();
        over.width = 1280;
        over.height = 1280;
        let err = g.validate(&over).expect_err("over-area must be rejected");
        let msg = err.to_string();
        assert!(msg.contains("max area"), "message names the cap: {msg}");
    }

    #[test]
    fn load_rejects_adapters_quant_and_single_file() {
        use candle_gen::gen_core::{AdapterKind, AdapterSpec};
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
        let file = LoadSpec::new(WeightsSource::File("/w.safetensors".into()));
        assert!(load(&file).is_err());
    }
}
