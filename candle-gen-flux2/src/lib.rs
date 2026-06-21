//! # candle-gen-flux2
//!
//! The **FLUX.2-klein** provider crate for [`candle-gen`](candle_gen) — the candle (Windows/CUDA)
//! sibling of `mlx-gen-flux2`. Unlike FLUX.1 (sc-3694), FLUX.2 has **no** `candle-transformers`
//! reference: the MMDiT transformer ([`transformer`]), the 32-channel 2×2-patchify VAE ([`vae`]), the
//! Qwen3 prompt-embeds text path ([`text_encoder`]), the 4-axis RoPE ([`pos_embed`]) and the
//! flow-match geometry ([`pipeline`]) are all ported here from the macOS provider.
//!
//! **txt2img (sc-3695):** [`Flux2Generator::generate`] runs Qwen3 (hidden states 9/18/27 → 12288-wide
//! `prompt_embeds`) → the MMDiT (8 joint + 24 fused-single blocks, distilled **4-step** flow-match
//! Euler, guidance 1.0) → the AutoencoderKL-Flux2 decoder, registered under `"flux2_klein_9b"`. Same
//! deterministic CPU-seeded-noise contract (sc-3673); the Qwen chat-template tokenization reuses
//! gen-core's [`TextTokenizer`] with [`ChatTemplate::QwenInstructNoThink`].
//!
//! **Sampling (epic 7114 P4, sc-7123):** both denoise loops (txt2img [`Pipeline::render`] and the edit
//! path [`Flux2Edit`]) route through the unified curated sampler/scheduler driver
//! (`candle_gen::run_flow_sampler` / `resolve_flow_schedule`). FLUX.2 is a rectified-flow engine using
//! the `Sigma` convention but embeds σ×1000, so the predict closure feeds `sigma * 1000.0` to the
//! transformer; the guidance>1 CFG blend (and, on the edit path, the joint `[target, refs]` concat)
//! lives inside that closure. The descriptor advertises the curated sampler/scheduler menus; the default
//! (unset sampler/scheduler) path is the N1 no-op — euler over the native empirical-mu flow-match schedule.
//!
//! **First-slice surface:** txt2img only. The mlx provider's edit variants (`flux2_klein_9b_edit`,
//! `flux2_klein_9b_kv_edit` — single/multi Reference, the reference-K/V cache), LoRA/LoKr, and Q4/Q8
//! quantization are **not** wired here; they are a follow-up. The descriptor advertises only the
//! txt2img surface so the worker routes the rest to the Python fallback. `backend = "candle"`,
//! `mac_only = false`.

pub mod config;
pub mod edit_provider;
pub mod pipeline;
pub mod pos_embed;
pub mod text_encoder;
pub mod transformer;
pub mod vae;

pub use edit_provider::{Flux2Edit, Flux2EditPaths, Flux2EditRequest};

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::{DType, Device, IndexOp, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::registry::ModelRegistration;
use candle_gen::gen_core::sampling::TimestepConvention;
use candle_gen::gen_core::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use candle_gen::gen_core::{
    self, Capabilities, GenerationOutput, GenerationRequest, Generator, Image, LoadSpec, Modality,
    ModelDescriptor, Progress, Quant, WeightsSource,
};
use candle_gen::{CandleError, Result as CResult};

use config::{Flux2Config, DEFAULT_GUIDANCE, DEFAULT_STEPS, FLUX2_KLEIN_9B_ID, SIZE_MULTIPLE};
use text_encoder::Qwen3TextEncoder;
use transformer::Flux2Transformer;
use vae::Flux2Vae;

/// Qwen3 `<|endoftext|>` pad token id (FLUX.2 text encoder).
const QWEN_PAD_TOKEN_ID: i32 = 151643;

/// The loaded FLUX.2 components, `Arc`-shared so the generator caches them across `generate` calls.
#[derive(Clone)]
struct Components {
    te: Arc<Qwen3TextEncoder>,
    transformer: Arc<Flux2Transformer>,
    vae: Arc<Flux2Vae>,
}

/// A txt2img pipeline handle: snapshot root + device + the f32 compute dtype. `pub(crate)` so the
/// edit provider ([`edit_provider`]) reuses the snapshot mmap + prompt-encode scaffolding.
pub(crate) struct Pipeline {
    pub(crate) cfg: Flux2Config,
    pub(crate) root: PathBuf,
    pub(crate) device: Device,
    pub(crate) dtype: DType,
}

impl Pipeline {
    pub(crate) fn load(root: &Path, device: &Device) -> Self {
        Self {
            cfg: Flux2Config::klein_9b(),
            root: root.to_path_buf(),
            device: device.clone(),
            // FLUX.2 runs the reference math in f32 (the Qwen3 encoder + the MMDiT). The 9b weights
            // are large but the math is parity-sensitive; a bf16 pass is a follow-up optimization.
            dtype: DType::F32,
        }
    }

    /// mmap a VarBuilder over every `.safetensors` in the snapshot subdir `sub`.
    pub(crate) fn component_vb(&self, sub: &str) -> CResult<VarBuilder<'static>> {
        let dir = self.root.join(sub);
        if !dir.is_dir() {
            return Err(CandleError::Msg(format!(
                "flux2 snapshot is missing the {sub}/ component dir (expected a FLUX.2-klein \
                 diffusers snapshot at {})",
                self.root.display()
            )));
        }
        let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
            .map_err(|e| CandleError::Msg(format!("flux2: read {sub}/: {e}")))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        files.sort();
        if files.is_empty() {
            return Err(CandleError::Msg(format!(
                "flux2: no .safetensors in {sub}/ (at {})",
                dir.display()
            )));
        }
        // SAFETY: mmap of read-only weight files; standard candle loading path.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, self.dtype, &self.device)? };
        Ok(vb)
    }

    fn load_components(&self) -> CResult<Components> {
        let te = Qwen3TextEncoder::new(&self.cfg, self.component_vb("text_encoder")?)?;
        let transformer = Flux2Transformer::new(&self.cfg, self.component_vb("transformer")?)?;
        let vae = Flux2Vae::new(self.component_vb("vae")?)?;
        Ok(Components {
            te: Arc::new(te),
            transformer: Arc::new(transformer),
            vae: Arc::new(vae),
        })
    }

    /// Tokenize + encode the prompt to `prompt_embeds` `[1, 512, 12288]` (f32).
    pub(crate) fn encode(&self, te: &Qwen3TextEncoder, prompt: &str) -> CResult<Tensor> {
        let tok = TextTokenizer::from_file(
            self.root.join("tokenizer/tokenizer.json"),
            TokenizerConfig {
                max_length: self.cfg.max_sequence_length,
                pad_token_id: QWEN_PAD_TOKEN_ID,
                chat_template: ChatTemplate::QwenInstructNoThink,
                pad_to_max_length: true,
            },
        )
        .map_err(|e| CandleError::Msg(format!("flux2: load tokenizer: {e}")))?;
        let out = tok
            .tokenize(prompt)
            .map_err(|e| CandleError::Msg(format!("flux2: tokenize: {e}")))?;
        let len = out.ids.len();
        let ids: Vec<u32> = out.ids.iter().map(|&i| i as u32).collect();
        let mask: Vec<i64> = out.mask.iter().map(|&m| m as i64).collect();
        let input_ids = Tensor::from_vec(ids, (1, len), &self.device)?;
        let attn_mask = Tensor::from_vec(mask, (1, len), &self.device)?;
        Ok(te.prompt_embeds(&input_ids, &attn_mask)?)
    }

    fn render(
        &self,
        req: &GenerationRequest,
        comps: &Components,
        on_progress: &mut dyn FnMut(Progress),
    ) -> CResult<Vec<Image>> {
        let steps = req
            .steps
            .map(|s| s as usize)
            .unwrap_or(DEFAULT_STEPS as usize);
        let base_seed = req.seed.unwrap_or_else(gen_core::default_seed);
        let guidance = req.guidance.unwrap_or(DEFAULT_GUIDANCE);
        let (lat_h, lat_w) = pipeline::latent_dims(req.width, req.height);

        // Prompt embeds are seed-independent: encode once.
        let prompt_embeds = self.encode(&comps.te, &req.prompt)?;
        // Classifier-free negative only when guidance > 1 (distilled klein runs CFG-free at 1.0).
        let negative = if guidance > 1.0 {
            let neg = req.negative_prompt.as_deref().unwrap_or(" ");
            Some(self.encode(&comps.te, neg)?)
        } else {
            None
        };

        let img_ids = pipeline::prepare_grid_ids(lat_h, lat_w);
        let txt_ids = pipeline::prepare_text_ids(self.cfg.max_sequence_length);

        // Curated sampler/scheduler routing (epic 7114 P4, sc-7123). The NATIVE schedule is the legacy
        // empirical-mu flow-match sigmas (descending, trailing 0.0); the same `mu` feeds the curated
        // scheduler axis so `normal`/`karras`/etc. honor the resolution-dependent shift. The default path
        // (sampler/scheduler unset) is the N1 no-op — euler over the native schedule reproduces the legacy
        // `euler_step` flow-match loop within tolerance.
        let mu = pipeline::compute_mu(pipeline::image_seq_len(req.width, req.height), steps);
        let (native, _timesteps) = pipeline::schedule(steps, req.width, req.height);
        let sigmas =
            candle_gen::resolve_flow_schedule(req.scheduler.as_deref(), mu, steps, &native);

        let mut images = Vec::with_capacity(req.count as usize);
        for index in 0..req.count {
            let seed = base_seed.wrapping_add(index as u64);
            let latents =
                pipeline::create_noise(&self.cfg, seed, req.width, req.height, &self.device)?;

            // The driver does cancel + progress + the euler/curated integrator step. The forward (and the
            // guidance>1 CFG blend) lives inside `predict` so a multi-eval solver re-runs it. FLUX.2 uses
            // the Sigma convention but the model embeds σ×1000, so feed `sigma * 1000.0` to the transformer.
            let latents = candle_gen::run_flow_sampler(
                req.sampler.as_deref(),
                TimestepConvention::Sigma,
                &sigmas,
                latents,
                seed,
                &req.cancel,
                on_progress,
                |latents, sigma| -> CResult<Tensor> {
                    let ts = sigma * 1000.0;
                    let v = comps.transformer.forward(
                        latents,
                        &prompt_embeds,
                        &img_ids,
                        &txt_ids,
                        ts,
                    )?;
                    match &negative {
                        Some(neg) => {
                            let vn = comps
                                .transformer
                                .forward(latents, neg, &img_ids, &txt_ids, ts)?;
                            // vn + guidance·(v − vn)
                            Ok((&vn + ((&v - &vn)? * guidance as f64)?)?)
                        }
                        None => Ok(v),
                    }
                },
            )?;

            on_progress(Progress::Decoding);
            let packed = pipeline::unpack_latents(&latents, req.width, req.height)?;
            let decoded = comps.vae.decode_packed(&packed)?; // [1,3,H,W] in [-1,1]
            images.push(to_image(&decoded)?);
        }
        Ok(images)
    }
}

/// Map a decoded `[1, 3, H, W]` tensor in `[-1, 1]` to an RGB8 [`Image`].
pub(crate) fn to_image(decoded: &Tensor) -> CResult<Image> {
    let img = ((decoded.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?.to_dtype(DType::U8)?;
    let img = img.i(0)?.to_device(&Device::Cpu)?;
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

/// A loaded candle FLUX.2 generator. Loading is lazy; components build on the first `generate` and
/// are cached.
pub struct Flux2Generator {
    descriptor: ModelDescriptor,
    root: PathBuf,
    device: Device,
    components: Mutex<Option<Components>>,
}

impl Flux2Generator {
    fn components(&self, pipe: &Pipeline) -> gen_core::Result<Components> {
        let mut guard = self
            .components
            .lock()
            .expect("flux2 components cache mutex poisoned");
        if let Some(c) = guard.as_ref() {
            return Ok(c.clone());
        }
        let c = pipe.load_components()?;
        *guard = Some(c.clone());
        Ok(c)
    }
}

impl Generator for Flux2Generator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        self.descriptor
            .capabilities
            .validate_request(FLUX2_KLEIN_9B_ID, req)?;
        if req.prompt.is_empty() {
            return Err(gen_core::Error::Msg(
                "flux2_klein_9b: prompt must not be empty".into(),
            ));
        }
        if req.steps == Some(0) {
            return Err(gen_core::Error::Msg(
                "flux2_klein_9b: steps must be >= 1 (an explicit 0 renders undenoised noise)"
                    .into(),
            ));
        }
        if !req.width.is_multiple_of(SIZE_MULTIPLE) || !req.height.is_multiple_of(SIZE_MULTIPLE) {
            return Err(gen_core::Error::Msg(format!(
                "flux2_klein_9b: width/height must be multiples of {SIZE_MULTIPLE} (got {}x{})",
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
        let pipe = Pipeline::load(&self.root, &self.device);
        let components = self.components(&pipe)?;
        let images = pipe.render(req, &components, on_progress)?;
        Ok(GenerationOutput::Images(images))
    }
}

/// FLUX.2-klein-9b txt2img descriptor — the surface sc-3695 wires. Guidance is advertised (klein
/// defaults to 1.0 / CFG-free, but >1.0 runs a classifier-free negative pass); no negative-prompt-
/// only, no conditioning (edit/Reference deferred), no LoRA/quant.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: FLUX2_KLEIN_9B_ID,
        family: "flux2",
        backend: "candle",
        modality: Modality::Image,
        capabilities: Capabilities {
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: false,
            // txt2img only in this slice — the mlx edit/Reference surface is deferred.
            conditioning: vec![],
            supports_lora: false,
            supports_lokr: false,
            // Curated sampler/scheduler menu (epic 7114 P4, sc-7123). The legacy `flow_match_euler`
            // scheduler alias is retained and falls back to the native schedule via the N3 path.
            samplers: candle_gen::curated_sampler_names(),
            schedulers: candle_gen::menu_with_aliases(
                candle_gen::curated_scheduler_names(),
                &["flow_match_euler"],
            ),
            min_size: 256,
            max_size: 2048,
            max_count: 8,
            mac_only: false,
            supported_quants: &[] as &[Quant],
            supports_kv_cache: false,
            // FLUX.2 uses the empirical-mu shifted flow-match schedule.
            requires_sigma_shift: true,
        },
    }
}

/// Construct a lazy candle FLUX.2 generator. `spec.weights` must be a [`WeightsSource::Dir`] pointing
/// at a `black-forest-labs/FLUX.2-klein-9B` diffusers snapshot (`text_encoder/`, `transformer/`,
/// `vae/`, `tokenizer/`). Adapters / quantization / control overlays are rejected (not wired).
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(
                "flux2_klein_9b expects a snapshot directory (text_encoder/ transformer/ vae/ \
                 tokenizer/), not a single .safetensors file"
                    .into(),
            ));
        }
    };
    if !spec.adapters.is_empty() {
        return Err(gen_core::Error::Unsupported(
            "candle flux2_klein_9b does not support LoRA/LoKr yet".into(),
        ));
    }
    if spec.quantize.is_some() {
        return Err(gen_core::Error::Unsupported(
            "candle flux2_klein_9b does not support on-the-fly Q4/Q8 quantization yet".into(),
        ));
    }
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(gen_core::Error::Unsupported(
            "candle flux2_klein_9b does not support control / IP-adapter / edit yet (txt2img only)"
                .into(),
        ));
    }
    let device = candle_gen::default_device()?;
    Ok(Box::new(Flux2Generator {
        descriptor: descriptor(),
        root,
        device,
        components: Mutex::new(None),
    }))
}

// Link-time self-registration into gen-core's model registry.
inventory::submit! {
    ModelRegistration { descriptor, load }
}

/// Force-link hook (keeps the `inventory::submit!` registration from being dead-stripped when the
/// crate is reached only through the registry). Same pattern as the other providers.
pub fn force_link() {}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::registry;
    use candle_gen::gen_core::ConditioningKind;

    #[test]
    fn registers_and_resolves_as_candle() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry::load(FLUX2_KLEIN_9B_ID, &spec).expect("flux2 is registered");
        assert_eq!(g.descriptor().id, FLUX2_KLEIN_9B_ID);
        assert_eq!(g.descriptor().family, "flux2");
        assert_eq!(g.descriptor().backend, "candle");
        assert_eq!(g.descriptor().modality, Modality::Image);
    }

    #[test]
    fn descriptor_advertises_only_wired_txt2img_surface() {
        let d = descriptor();
        assert!(d.capabilities.supports_guidance);
        assert!(d.capabilities.requires_sigma_shift);
        assert!(!d.capabilities.mac_only);
        assert!(d.capabilities.conditioning.is_empty());
        assert!(!d.capabilities.supports_lora);
        assert!(!d.capabilities.supports_kv_cache);
        assert!(d.capabilities.supported_quants.is_empty());
        assert!(!d.capabilities.accepts(ConditioningKind::Reference));
    }

    #[test]
    fn validate_accepts_txt2img_and_rejects_unsupported() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry::load(FLUX2_KLEIN_9B_ID, &spec).unwrap();
        let ok = GenerationRequest {
            prompt: "a rusty robot holding a lit candle".into(),
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

    #[test]
    fn load_rejects_unwired_surfaces() {
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
    }

    #[test]
    fn load_rejects_single_file_source() {
        let spec = LoadSpec::new(WeightsSource::File("/tmp/flux2.safetensors".into()));
        let err = load(&spec).err().expect("expected an error").to_string();
        assert!(err.contains("snapshot directory"), "got: {err}");
    }
}
