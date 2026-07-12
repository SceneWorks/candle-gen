//! # candle-gen-ltx
//!
//! The **LTX-2.3 (distilled 22B)** text-to-video provider for [`candle-gen`](candle_gen) — the candle
//! (Windows/CUDA) sibling of `mlx-gen-ltx`. LTX has **no** `candle-transformers` reference: the
//! `AVTransformer3DModel` video DiT ([`transformer`]), the `CausalVideoAutoencoder` temporal VAE
//! ([`vae`], on a from-scratch [`conv3d`]), the **Gemma-3-12B** text encoder ([`gemma`]) +
//! per-token-RMS aggregation + 8-layer learnable-register connector ([`text_encoder`], [`connector`])
//! are all ported here. The distilled rectified-flow denoise runs through the unified
//! `candle_gen::run_av_curated_sampler` over the fixed [`STAGE1_SIGMAS`](config::STAGE1_SIGMAS)
//! schedule (epic 7114), so no per-crate scheduler module is needed.
//!
//! **txt2video+audio (sc-3698 / sc-5495):** [`LtxGenerator::generate`] runs Gemma-3-12B → video +
//! audio text projections → connectors → the 48-layer dual-modal [`AvDiT`](transformer::AvDiT) (split
//! 3-D RoPE, per-head gated attention, adaLN-single, bidirectional cross-modal attention) joint
//! denoise → the temporal VAE decoder (frames) **plus** the [`AudioDecoder`](audio_vae::AudioDecoder)
//! → [`LtxVocoder`](vocoder::LtxVocoder) → a synchronized 48 kHz stereo `AudioTrack`. Registered under
//! `"ltx_2_3_distilled"`; single-stage distilled denoise (no CFG). **Deferred** to follow-up stories:
//! the 2-stage latent upsampler, I2V conditioning, prompt-enhance, LoRA/IC-LoRA, and fp8/quant.
//!
//! **Dtypes:** the DiT, connector, text projection, and Gemma encoder run **bf16** (the checkpoint's
//! native dtype; 22B+12B does not fit f32 on a single 96 GB GPU); the VAE runs **f32**; attention and
//! norms upcast to f32. `backend = "candle"`, `mac_only = false`.
//!
//! **Weights:** `spec.weights` points at an LTX-2.3 snapshot dir (the
//! `ltx-2.3-22b-distilled.safetensors` single-file checkpoint bundling DiT + VAE + projection +
//! connector). The Gemma-3-12B encoder + its `tokenizer.json` live in a separate snapshot, located via
//! the `LTX_GEMMA_DIR` env var (falling back to `<root>/text_encoder`).

pub mod audio_vae;
pub mod config;
pub mod connector;
pub mod conv3d;
pub mod gemma;
pub mod pipeline;
pub mod quant;
pub mod rope;
pub mod text_encoder;
pub mod tier;
pub mod transformer;
pub mod vae;
pub mod vocoder;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::{
    self, AudioTrack, Capabilities, GenerationOutput, GenerationRequest, Generator, Image,
    LoadSpec, Modality, ModelDescriptor, Progress, Quant, WeightsSource,
};
use candle_gen::{run_av_curated_sampler, AvLatents, CandleError, Result as CResult};

use audio_vae::AudioDecoder;
use config::{
    compute_audio_frames, AudioVaeConfig, AvConfig, ConnectorConfig, GemmaConfig, VocoderConfig,
    DEFAULT_FPS, DEFAULT_FRAMES, DEFAULT_HEIGHT, DEFAULT_WIDTH, MODEL_ID, NATIVE_STEPS,
    STAGE1_SIGMAS, TEXT_MAX_LENGTH,
};
use gemma::GemmaEncoder;
use text_encoder::LtxTextEncoder;
use transformer::AvDiT;
use vae::LtxVideoVae;
use vocoder::LtxVocoder;

const DIT_DTYPE: DType = DType::BF16;
const VAE_DTYPE: DType = DType::F32;
const SIZE_MULTIPLE: u32 = config::SPATIAL_SCALE as u32;

#[derive(Clone)]
struct Components {
    te: Arc<LtxTextEncoder>,
    avdit: Arc<AvDiT>,
    vae: Arc<LtxVideoVae>,
    /// Audio decode chain — `None` on the packed MLX tier path (sc-9545), which is **video-only**: the
    /// tier's audio-VAE + vocoder ship in a different key layout (channels-last convs, no `decoder.`/
    /// `vocoder.` prefix) that is a separate ingestion slice (follow-up), and the sc-9417 render AC is a
    /// video render. The audio latent stream still runs through the joint AvDiT (cross-modal coupling
    /// keeps the video coherent); only the audio VAE→vocoder decode is skipped.
    audio: Option<AudioChain>,
    tokenizer: Arc<tokenizers::Tokenizer>,
}

#[derive(Clone)]
struct AudioChain {
    decoder: Arc<AudioDecoder>,
    vocoder: Arc<LtxVocoder>,
    sample_rate: u32,
}

struct Pipeline {
    av_cfg: AvConfig,
    gemma_cfg: GemmaConfig,
    conn_cfg: ConnectorConfig,
    audio_conn_cfg: ConnectorConfig,
    audio_vae_cfg: AudioVaeConfig,
    vocoder_cfg: VocoderConfig,
    root: PathBuf,
    device: Device,
    /// Gemma-encoder override from `LoadSpec::text_encoder` (sc-8827); see [`Pipeline::gemma_dir`].
    gemma_override: Option<PathBuf>,
}

impl Pipeline {
    fn load(root: &Path, device: &Device, gemma_override: Option<PathBuf>) -> Self {
        Self {
            av_cfg: AvConfig::ltx_2_3(),
            gemma_cfg: GemmaConfig::gemma_3_12b(),
            conn_cfg: ConnectorConfig::ltx_2_3(),
            audio_conn_cfg: ConnectorConfig::ltx_2_3_audio(),
            audio_vae_cfg: AudioVaeConfig::ltx_2_3(),
            vocoder_cfg: VocoderConfig::ltx_2_3(),
            root: root.to_path_buf(),
            device: device.clone(),
            gemma_override,
        }
    }

    /// The single full **dense bf16** LTX-2.3 checkpoint in `root` — the 22B model bundling DiT + VAE +
    /// audio-VAE + vocoder + projection (not a LoRA / upscaler / fp8 variant). Handles both the base
    /// `Lightricks/LTX-2.3` (`ltx-2.3-22b-distilled*.safetensors`) and full-model fine-tunes whose file
    /// is named differently (e.g. the eros merge's `10Eros_v1_bf16.safetensors`, sc-5495): the snapshot
    /// may carry several `.safetensors` (bf16 + fp8 variants), so prefer `distilled`, then a `bf16`
    /// dense file, then the largest remaining — fp8/mixed are skipped (candle loads the bf16 weights).
    fn ltx_checkpoint(&self) -> CResult<PathBuf> {
        let lname = |p: &Path| {
            p.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_ascii_lowercase()
        };
        let mut cands: Vec<PathBuf> = std::fs::read_dir(&self.root)
            .map_err(|e| CandleError::Msg(format!("ltx: read snapshot dir: {e}")))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                let name = lname(p);
                name.ends_with(".safetensors")
                    && !name.contains("lora")
                    && !name.contains("upscaler")
                    && !name.contains("fp8")
                    && !name.contains("mixed")
            })
            .collect();
        cands.sort();
        if cands.is_empty() {
            return Err(CandleError::Msg(format!(
                "ltx: no dense LTX-2.3 `.safetensors` checkpoint in {} (expected e.g. \
                 `ltx-2.3-22b-distilled.safetensors` or a `*_bf16.safetensors` full-model fine-tune)",
                self.root.display()
            )));
        }
        if let Some(p) = cands.iter().find(|p| lname(p).contains("distilled")) {
            return Ok(p.clone());
        }
        if let Some(p) = cands.iter().find(|p| lname(p).contains("bf16")) {
            return Ok(p.clone());
        }
        // No name hint — the full dense model dwarfs any aux file, so take the largest.
        Ok(cands
            .into_iter()
            .max_by_key(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0))
            .expect("cands non-empty"))
    }

    /// The Gemma-3-12B encoder snapshot dir. A `LoadSpec::text_encoder` override (sc-8827) wins; then
    /// `$LTX_GEMMA_DIR`; then `<root>/text_encoder`.
    fn gemma_dir(&self) -> CResult<PathBuf> {
        if let Some(p) = &self.gemma_override {
            if !p.is_dir() {
                return Err(CandleError::Msg(format!(
                    "ltx: LoadSpec text_encoder path is not a directory: {}",
                    p.display()
                )));
            }
            return Ok(p.clone());
        }
        if let Ok(p) = std::env::var("LTX_GEMMA_DIR") {
            return Ok(PathBuf::from(p));
        }
        let fallback = self.root.join("text_encoder");
        if fallback.is_dir() {
            return Ok(fallback);
        }
        Err(CandleError::Msg(
            "ltx: set LTX_GEMMA_DIR to a google/gemma-3-12b-it snapshot (or place it at \
             <root>/text_encoder)"
                .into(),
        ))
    }

    fn safetensors_in(dir: &Path) -> CResult<Vec<PathBuf>> {
        // Shared sorted-`.safetensors` resolver (sc-8999 / F-019).
        candle_gen::sorted_safetensors(dir, "ltx")
    }

    fn load_components(&self) -> CResult<Components> {
        // sc-9545: a packed MLX split-tier subdir (`.../q4` or `.../q8`) is ingested through the
        // remapping VarBuilders in `tier` so the sc-9417 packed-detect seam fires on the real tier
        // weights with no dense staging; the single-bundle dense checkpoint keeps the legacy path below.
        if let Some(paths) = tier::TierPaths::detect(&self.root, self.gemma_override.as_deref()) {
            return self.load_components_tier(&paths);
        }

        let ltx_file = self.ltx_checkpoint()?;
        let gemma_dir = self.gemma_dir()?;
        let gemma_files = Self::safetensors_in(&gemma_dir)?;

        // Two builders over the single LTX file: bf16 (DiT + projection + connector), f32 (VAE).
        let ltx_files = [ltx_file];
        let vb_bf16 = candle_gen::mmap_var_builder(&ltx_files, DIT_DTYPE, &self.device)?;
        let vb_f32 = candle_gen::mmap_var_builder(&ltx_files, VAE_DTYPE, &self.device)?;
        let gemma_vb = candle_gen::mmap_var_builder(&gemma_files, DIT_DTYPE, &self.device)?
            .pp("language_model.model");

        let dit_vb = vb_bf16.pp("model.diffusion_model");
        let avdit = AvDiT::new(dit_vb.clone(), &self.av_cfg)?;
        let te = LtxTextEncoder::new_av(
            gemma_vb,
            vb_bf16.clone(),
            dit_vb,
            &self.gemma_cfg,
            &self.conn_cfg,
            &self.audio_conn_cfg,
        )?;
        let vae = LtxVideoVae::new(vb_f32.pp("vae"), config::LATENT_CHANNELS, 4)?;
        // The audio VAE decoder + vocoder run f32 (post-sampling quality islands).
        let audio_decoder = AudioDecoder::load(&vb_f32.pp("audio_vae"), &self.audio_vae_cfg)?;
        let vocoder = LtxVocoder::load(vb_f32, &self.device, &self.vocoder_cfg)?;
        let audio_sample_rate = self.vocoder_cfg.final_sample_rate() as u32;

        let tok_path = gemma_dir.join("tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
            .map_err(|e| CandleError::Msg(format!("ltx: load gemma tokenizer: {e}")))?;

        Ok(Components {
            te: Arc::new(te),
            avdit: Arc::new(avdit),
            vae: Arc::new(vae),
            audio: Some(AudioChain {
                decoder: Arc::new(audio_decoder),
                vocoder: Arc::new(vocoder),
                sample_rate: audio_sample_rate,
            }),
            tokenizer: Arc::new(tokenizer),
        })
    }

    /// Load the DiT (packed) + connectors/text-projection (dense) + video VAE (dense) + Gemma TE
    /// straight from the split MLX packed tier (sc-9545). The DiT builder applies the crate→tier key
    /// remap so [`crate::quant::qlinear`]'s packed-detect fires on the real `.scales` siblings; the
    /// group_size is read + validated from `quantize_config.json` (AC). **Video-only**: the tier's
    /// audio-VAE + vocoder are a separate ingestion slice (channels-last, differently-prefixed) tracked
    /// as a follow-up — the audio latent stream still flows through the joint AvDiT, only its final
    /// VAE→vocoder decode is skipped.
    fn load_components_tier(&self, paths: &tier::TierPaths) -> CResult<Components> {
        // Read + validate the tier's group_size (AC): errors loudly if a tier ever ships a group the
        // packed loaders don't repack at, rather than mis-aligning the MLX→GGML repack.
        let _group = paths.validate_group_size()?;

        let dit_vb = paths.dit_vb(DIT_DTYPE, &self.device)?;
        let conn_vb = paths.connector_vb(DIT_DTYPE, &self.device)?;
        let vae_vb = paths.vae_vb(VAE_DTYPE, &self.device)?;
        let gemma_vb = paths.gemma_vb(DIT_DTYPE, &self.device)?;

        // The DiT loader roots at `model.diffusion_model.` (the remap strips it); the connector loader
        // is handed a `model.diffusion_model.`-prefixed builder too (the remap strips it), and the text
        // projection sits at the connector-file root (also reached through that builder).
        let dit_root = dit_vb.pp("model.diffusion_model");
        let conn_root = conn_vb.pp("model.diffusion_model");
        let avdit = AvDiT::new(dit_root.clone(), &self.av_cfg)?;
        let te = LtxTextEncoder::new_av(
            gemma_vb,
            conn_root.clone(),
            conn_root,
            &self.gemma_cfg,
            &self.conn_cfg,
            &self.audio_conn_cfg,
        )?;
        let vae = LtxVideoVae::new(vae_vb.pp("vae"), config::LATENT_CHANNELS, 4)?;

        let tok_path = paths.tokenizer_path();
        let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
            .map_err(|e| CandleError::Msg(format!("ltx tier: load gemma tokenizer: {e}")))?;

        Ok(Components {
            te: Arc::new(te),
            avdit: Arc::new(avdit),
            vae: Arc::new(vae),
            audio: None,
            tokenizer: Arc::new(tokenizer),
        })
    }

    /// Tokenize `prompt` with the Gemma tokenizer (BOS, right-truncate then **left-pad** to
    /// `TEXT_MAX_LENGTH`), returning `(input_ids [1, 256] u32, mask01 [256])`.
    fn tokenize(&self, tok: &tokenizers::Tokenizer, prompt: &str) -> CResult<(Tensor, Vec<u32>)> {
        let enc = tok
            .encode(prompt, true)
            .map_err(|e| CandleError::Msg(format!("ltx: tokenize: {e}")))?;
        let mut ids: Vec<u32> = enc.get_ids().to_vec();
        let max = TEXT_MAX_LENGTH;
        if ids.len() > max {
            ids.truncate(max);
        }
        let nv = ids.len();
        let pad = max - nv;
        let mut padded = vec![0u32; pad];
        padded.extend_from_slice(&ids);
        let mut mask = vec![0u32; pad];
        mask.extend(std::iter::repeat_n(1u32, nv));
        let input_ids = Tensor::from_vec(padded, (1, max), &self.device)?;
        Ok((input_ids, mask))
    }

    fn render(
        &self,
        req: &GenerationRequest,
        comps: &Components,
        on_progress: &mut dyn FnMut(Progress),
    ) -> CResult<(Vec<Image>, u32, Option<AudioTrack>)> {
        let frames = req.frames.unwrap_or(DEFAULT_FRAMES);
        let fps = req.fps.unwrap_or(DEFAULT_FPS);
        let seed = req.seed.unwrap_or_else(gen_core::default_seed);

        // Text encode → video (1,256,4096) + audio (1,256,2048) contexts (one Gemma pass).
        let (input_ids, mask01) = self.tokenize(&comps.tokenizer, &req.prompt)?;
        let (video_ctx, audio_ctx) = comps.te.encode_both(&input_ids, &mask01)?;

        // Latent geometry + position grids (video 3-axis, audio 1-axis time).
        let (t_lat, h_lat, w_lat) = pipeline::latent_dims(frames, req.width, req.height);
        let af = compute_audio_frames(frames as usize, fps as f64).max(1);
        let video_grid = rope::create_position_grid(t_lat, h_lat, w_lat, fps as f32, &self.device)?;
        let audio_grid = rope::create_audio_position_grid(af, &self.device)?;

        let vlat = pipeline::create_noise(seed, t_lat, h_lat, w_lat, &self.device)?;
        let alat = pipeline::create_audio_noise(seed, af, &self.device)?;

        // Unified curated sampling over the JOINT video+audio streams (epic 7114 P4, sc-7125). LTX is
        // distilled rectified-flow with the fixed `STAGE1_SIGMAS` schedule, so per decision 3b it exposes
        // the SAMPLER axis but NO scheduler axis (the baked σ schedule is the native default). The
        // default `euler` reproduces the legacy per-stream `to_denoised`→`euler_step` loop exactly (the
        // FLOW `x0 = x − σ·v` recombine + euler == the native scheduler), the N1 no-op. Both streams are
        // velocity-prediction (`Sigma` convention); the AvDiT couples them via cross-modal attention each
        // forward, so the per-step model eval (flatten → AvDiT → unflatten) lives inside the closure.
        let out = run_av_curated_sampler(
            req.sampler.as_deref(),
            &STAGE1_SIGMAS[..],
            AvLatents {
                video: vlat,
                audio: alat,
            },
            seed,
            &req.cancel,
            on_progress,
            |av, sigma| -> CResult<AvLatents> {
                let vflat = pipeline::flatten_latent(&av.video)?;
                let aflat = pipeline::flatten_audio_latent(&av.audio)?;
                let (vvel, avel) = comps.avdit.forward(
                    &vflat,
                    &aflat,
                    sigma as f64,
                    &video_ctx,
                    &audio_ctx,
                    &video_grid,
                    &audio_grid,
                )?;
                Ok(AvLatents {
                    video: pipeline::unflatten_latent(
                        &vvel.to_dtype(DType::F32)?,
                        t_lat,
                        h_lat,
                        w_lat,
                    )?,
                    audio: pipeline::unflatten_audio_latent(&avel.to_dtype(DType::F32)?, af)?,
                })
            },
        )?;
        let vlat = out.video;
        let alat = out.audio;

        on_progress(Progress::Decoding);
        // sc-7076 — memory-bounded + catchable VAE decode (budgeted tiling), replacing the single-pass
        // full-video decode that OOMs the worker on large/long outputs.
        let decoded = comps.vae.decode_budgeted(&vlat)?;
        let images = pipeline::frames_to_images(&decoded)?;
        // Audio decode only when the audio chain is loaded (the dense bundle); the packed MLX tier is
        // video-only (sc-9545) — its audio VAE/vocoder are a separate ingestion slice.
        let audio = match &comps.audio {
            Some(chain) => Some(pipeline::decode_audio_track(
                &chain.decoder,
                &chain.vocoder,
                &alat,
                chain.sample_rate,
            )?),
            None => None,
        };
        Ok((images, fps, audio))
    }
}

pub struct LtxGenerator {
    descriptor: ModelDescriptor,
    root: PathBuf,
    device: Device,
    /// Optional Gemma-encoder snapshot dir from `LoadSpec::text_encoder` (sc-8827); overrides the
    /// `$LTX_GEMMA_DIR` env var / `<root>/text_encoder` fallback in [`Pipeline::gemma_dir`].
    gemma_override: Option<PathBuf>,
    components: Mutex<Option<Components>>,
}

impl LtxGenerator {
    fn components(&self, pipe: &Pipeline) -> gen_core::Result<Components> {
        // `cached` recovers a poisoned lock (sc-9015) internally; `?` bridges the candle-side
        // `load_components` error into `gen_core::Error`.
        Ok(candle_gen::cached(&self.components, || {
            pipe.load_components()
        })?)
    }
}

impl Generator for LtxGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        self.descriptor
            .capabilities
            .validate_request(MODEL_ID, req)?;
        if req.prompt.is_empty() {
            return Err(gen_core::Error::Msg("ltx: prompt must not be empty".into()));
        }
        if !req.width.is_multiple_of(SIZE_MULTIPLE) || !req.height.is_multiple_of(SIZE_MULTIPLE) {
            return Err(gen_core::Error::Msg(format!(
                "ltx: width/height must be multiples of {SIZE_MULTIPLE} (got {}x{})",
                req.width, req.height
            )));
        }
        if let Some(f) = req.frames {
            if f == 0 || f % config::TEMPORAL_SCALE as u32 != 1 {
                return Err(gen_core::Error::Msg(format!(
                    "ltx: frames must satisfy frames % {} == 1 (got {f})",
                    config::TEMPORAL_SCALE
                )));
            }
        }
        // Bound the AvDiT denoise sequence length (F-131, sc-11234). The checks above bound only the
        // frame *shape*, never its magnitude, so a huge frame count (e.g. `frames: 2001`, which
        // satisfies `% 8 == 1`) at a large resolution produced ~400k latent tokens and OOM'd deep in
        // the 22B denoise loop rather than failing catchably here. The video latent token count
        // `t_lat · h_lat · w_lat` is the memory driver (self-attn working set + per-token q/k/v across
        // 48 layers); cap it against the GPU envelope. Uses the effective frame count (the render
        // default when `None`) and the already-validated (mult-of-32) width/height.
        let eff_frames = req.frames.unwrap_or(DEFAULT_FRAMES);
        let (t_lat, h_lat, w_lat) = pipeline::latent_dims(eff_frames, req.width, req.height);
        let tokens = t_lat * h_lat * w_lat;
        let max_tokens = config::max_latent_tokens();
        if tokens > max_tokens {
            return Err(gen_core::Error::Msg(format!(
                "ltx: request too large — {eff_frames} frames at {}x{} is {tokens} latent tokens, \
                 over the {max_tokens}-token cap (the 22B AvDiT denoise loop would exceed the GPU \
                 memory envelope). Reduce the frame count or resolution, or raise \
                 LTX_MAX_LATENT_TOKENS for a larger-VRAM device.",
                req.width, req.height
            )));
        }
        // `req.steps` (sc-9027 / F-043): the distilled model bakes the fixed `STAGE1_SIGMAS` σ waypoints
        // into training, so the only supported step count is `NATIVE_STEPS`. Reject any other explicit
        // override with a clear diagnostic instead of silently running the baked schedule — a
        // `steps: 30` request must not quietly render at 8 steps. `None` uses the distilled default.
        if let Some(s) = req.steps {
            if s != NATIVE_STEPS {
                return Err(gen_core::Error::Msg(format!(
                    "ltx: this distilled model runs a fixed {NATIVE_STEPS}-step schedule and cannot \
                     honor steps={s}; omit `steps` (or pass {NATIVE_STEPS}) to use the baked schedule"
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
        let pipe = Pipeline::load(&self.root, &self.device, self.gemma_override.clone());
        let components = self.components(&pipe)?;
        let (frames, fps, audio) = pipe.render(req, &components, on_progress)?;
        Ok(GenerationOutput::Video { frames, fps, audio })
    }
}

/// LTX-2.3 distilled txt2video descriptor — single-stage rectified-flow (no CFG / negative prompt;
/// guidance is distilled in). The denoise step count is FIXED at [`NATIVE_STEPS`] (the baked
/// `STAGE1_SIGMAS` schedule); an explicit non-native `req.steps` is rejected in `validate` rather than
/// silently ignored (sc-9027 / F-043). Synchronized audio is produced (sc-5495, the joint video+audio
/// streams); I2V / upsampler / LoRA / quant remain deferred.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID,
        family: "ltx",
        backend: "candle",
        modality: Modality::Video,
        capabilities: Capabilities {
            supports_negative_prompt: false,
            supports_guidance: false,
            supports_true_cfg: false,
            conditioning: vec![],
            supports_lora: false,
            supports_lokr: false,
            // Unified curated SAMPLER menu (epic 7114 P4, sc-7125) over the joint video+audio streams +
            // the legacy `rectified-flow` alias (falls back to euler). Per decision 3b: sampler-only, NO
            // scheduler axis — LTX is distilled with the fixed `STAGE1_SIGMAS` schedule; `euler` is the
            // recommended default (the byte-faithful N1 path). The rest are exposed for ComfyUI parity.
            samplers: candle_gen::menu_with_aliases(
                candle_gen::curated_sampler_names(),
                &["rectified-flow"],
            ),
            schedulers: vec![],
            supported_guidance_methods: vec![],
            min_size: SIZE_MULTIPLE,
            max_size: 1280,
            max_count: 1,
            mac_only: false,
            supported_quants: &[] as &[Quant],
            supports_kv_cache: false,
            requires_sigma_shift: false,
        },
    }
}

/// Construct a lazy candle LTX-2.3 generator. `spec.weights` is an LTX-2.3 snapshot dir (the
/// `ltx-2.3-22b-distilled.safetensors` checkpoint); the Gemma encoder is located via `LTX_GEMMA_DIR`.
/// Adapters / quantization / conditioning are rejected (not wired).
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(p) => p
            .parent()
            .map(|d| d.to_path_buf())
            .unwrap_or_else(|| p.clone()),
    };
    if !spec.adapters.is_empty() {
        return Err(gen_core::Error::Unsupported(
            "candle ltx does not support LoRA/LoKr yet".into(),
        ));
    }
    if spec.quantize.is_some() {
        return Err(gen_core::Error::Unsupported(
            "candle ltx does not support on-the-fly Q4/Q8 quantization yet".into(),
        ));
    }
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(gen_core::Error::Unsupported(
            "candle ltx does not support image / I2V conditioning yet (txt2video only)".into(),
        ));
    }
    // sc-8827: the Gemma encoder location may ride the spec (`LoadSpec::text_encoder`) so the caller
    // does not have to mutate the process-global `$LTX_GEMMA_DIR`; `None` keeps the env / `<root>`
    // fallback in `gemma_dir`.
    let gemma_override = spec.text_encoder.as_ref().map(|src| match src {
        WeightsSource::Dir(p) | WeightsSource::File(p) => p.clone(),
    });
    let device = candle_gen::default_device()?;
    Ok(Box::new(LtxGenerator {
        descriptor: descriptor(),
        root,
        device,
        gemma_override,
        components: Mutex::new(None),
    }))
}

candle_gen::register_generators! { descriptor => load }

/// Force-link hook (keeps the `inventory::submit!` registration from being dead-stripped).
pub fn force_link() {}

#[allow(dead_code)]
fn _defaults_referenced() {
    let _ = (DEFAULT_WIDTH, DEFAULT_HEIGHT, GemmaEncoder::forward);
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::registry;

    #[test]
    fn registers_and_resolves_as_candle_video() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry::load(MODEL_ID, &spec).expect("ltx is registered");
        assert_eq!(g.descriptor().id, MODEL_ID);
        assert_eq!(g.descriptor().family, "ltx");
        assert_eq!(g.descriptor().backend, "candle");
        assert_eq!(g.descriptor().modality, Modality::Video);
    }

    #[test]
    fn gemma_dir_prefers_spec_text_encoder_over_env() {
        // sc-8827: a `LoadSpec::text_encoder` override drives the Gemma-encoder location, so the
        // worker does not have to mutate the process-global `$LTX_GEMMA_DIR`. An existing dir is
        // returned as-is (ahead of any env value); a nonexistent override errors with the spec-side
        // message. Uses a unique env value that is NOT a real dir so a fallthrough would differ.
        let real = std::env::temp_dir().join("ltx_gemma_spec_ok");
        let _ = std::fs::create_dir_all(&real);
        let pipe = Pipeline::load(
            Path::new("/nonexistent/root"),
            &Device::Cpu,
            Some(real.clone()),
        );
        assert_eq!(pipe.gemma_dir().unwrap(), real);
        std::fs::remove_dir_all(&real).ok();

        let bad = Pipeline::load(
            Path::new("/nonexistent/root"),
            &Device::Cpu,
            Some(PathBuf::from("/nonexistent/ltx_gemma")),
        );
        let err = bad.gemma_dir().unwrap_err().to_string();
        assert!(err.contains("LoadSpec text_encoder"), "got: {err}");
    }

    #[test]
    fn ltx_checkpoint_selects_base_distilled_and_eros_bf16() {
        // Helper: a temp dir seeded with `files`, then `ltx_checkpoint()`'s chosen file name.
        let pick = |tag: &str, files: &[&str]| -> String {
            let dir = std::env::temp_dir().join(format!("ltx_ckpt_{tag}"));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            for f in files {
                std::fs::write(dir.join(f), b"x").unwrap();
            }
            let pipe = Pipeline::load(&dir, &Device::Cpu, None);
            let got = pipe.ltx_checkpoint().unwrap();
            let name = got.file_name().unwrap().to_str().unwrap().to_owned();
            std::fs::remove_dir_all(&dir).unwrap();
            name
        };
        // Base `Lightricks/LTX-2.3`: the distilled file wins over dev / lora / upscaler.
        assert_eq!(
            pick(
                "base",
                &[
                    "ltx-2.3-22b-dev.safetensors",
                    "ltx-2.3-22b-distilled.safetensors",
                    "ltx-2.3-22b-distilled-lora-384.safetensors",
                    "ltx-2.3-spatial-upscaler-x2.safetensors",
                ],
            ),
            "ltx-2.3-22b-distilled.safetensors"
        );
        // Eros merge: the dense `_bf16` file wins; the fp8 / mixed variants are skipped.
        assert_eq!(
            pick(
                "eros",
                &[
                    "10Eros_v1_bf16.safetensors",
                    "10Eros_v1-fp8mixed_learned.safetensors",
                    "10Eros_v1_fp8_transformer.safetensors",
                ],
            ),
            "10Eros_v1_bf16.safetensors"
        );
    }

    #[test]
    fn descriptor_surface() {
        let d = descriptor();
        assert!(!d.capabilities.supports_guidance);
        assert!(!d.capabilities.supports_negative_prompt);
        assert!(!d.capabilities.mac_only);
        assert!(d.capabilities.conditioning.is_empty());
        // sc-7125: curated sampler menu + the legacy `rectified-flow` alias; NO scheduler axis (3b).
        assert!(d.capabilities.samplers.contains(&"rectified-flow"));
        assert!(d.capabilities.samplers.contains(&"euler"));
        assert!(d.capabilities.samplers.contains(&"dpmpp_2m"));
        assert!(d.capabilities.schedulers.is_empty());
    }

    #[test]
    fn validate_accepts_txt2video_and_rejects_unsupported() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry::load(MODEL_ID, &spec).unwrap();
        let ok = GenerationRequest {
            prompt: "a cat walking across a sunny garden".into(),
            width: 704,
            height: 480,
            frames: Some(49),
            sampler: Some("rectified-flow".into()),
            ..Default::default()
        };
        assert!(g.validate(&ok).is_ok());
        for bad in [
            GenerationRequest::default(),
            GenerationRequest {
                prompt: "x".into(),
                frames: Some(48), // not ≡ 1 (mod 8)
                ..Default::default()
            },
            GenerationRequest {
                prompt: "x".into(),
                width: 700, // not a multiple of 32
                ..Default::default()
            },
        ] {
            assert!(g.validate(&bad).is_err(), "should reject: {bad:?}");
        }
    }

    /// sc-9027 / F-043: the distilled schedule is fixed, so `render` runs exactly `NATIVE_STEPS`
    /// (`STAGE1_SIGMAS.len() − 1`) denoise steps and never resamples for an arbitrary `req.steps`.
    #[test]
    fn native_steps_matches_baked_schedule() {
        assert_eq!(NATIVE_STEPS as usize, STAGE1_SIGMAS.len() - 1);
        assert_eq!(NATIVE_STEPS, 8);
    }

    /// `req.steps` is no longer silently ignored: `None` (distilled default) and an explicit
    /// `Some(NATIVE_STEPS)` are accepted; any other override is rejected with a diagnostic rather than
    /// quietly running the baked 8-step schedule.
    #[test]
    fn validate_honors_or_rejects_req_steps() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry::load(MODEL_ID, &spec).unwrap();
        let base = GenerationRequest {
            prompt: "a cat walking across a sunny garden".into(),
            width: 704,
            height: 480,
            frames: Some(49),
            ..Default::default()
        };
        // Default (None) → distilled schedule.
        assert!(g.validate(&base).is_ok());
        // Explicit native step count is honored.
        assert!(g
            .validate(&GenerationRequest {
                steps: Some(NATIVE_STEPS),
                ..base.clone()
            })
            .is_ok());
        // A non-native override (the F-043 `steps: 30` case) is rejected, not silently ignored.
        for s in [1u32, 4, 7, 9, 30, 50] {
            assert!(
                g.validate(&GenerationRequest {
                    steps: Some(s),
                    ..base.clone()
                })
                .is_err(),
                "steps={s} must be rejected"
            );
        }
    }

    /// F-131 / sc-11234: `validate` bounds the video latent token count (`t_lat · h_lat · w_lat`),
    /// so a huge frame count that passes the `% 8 == 1` shape check but would OOM the 22B AvDiT
    /// denoise loop is rejected catchably up front instead of blowing up mid-render. An in-bounds
    /// long clip still passes.
    #[test]
    fn validate_rejects_unbounded_frame_count() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry::load(MODEL_ID, &spec).unwrap();
        let base = GenerationRequest {
            prompt: "a cat walking across a sunny garden".into(),
            width: 1280,
            height: 1280,
            ..Default::default()
        };
        // The finding's pathological case: 2001 frames satisfies `% 8 == 1` (shape-valid) but is
        // ~400k latent tokens at 1280² — far over the cap.
        assert_eq!(
            2001 % config::TEMPORAL_SCALE as u32,
            1,
            "shape-valid frame count"
        );
        let huge = GenerationRequest {
            frames: Some(2001),
            ..base.clone()
        };
        let err = g.validate(&huge).unwrap_err().to_string();
        assert!(
            err.contains("latent tokens") && err.contains("cap"),
            "over-cap request rejected with a clear message: {err}"
        );

        // The token count is the actual driver: computing it here mirrors `validate`.
        let (t, h, w) = pipeline::latent_dims(2001, 1280, 1280);
        assert!(
            t * h * w > config::max_latent_tokens(),
            "2001@1280² exceeds the cap"
        );

        // A generous but in-bounds clip still validates: 129 frames at 704×480 → t_lat 17 ·
        // (22·15) = 5610 latent tokens, comfortably under the 131072 cap.
        let ok = GenerationRequest {
            frames: Some(129),
            width: 704,
            height: 480,
            ..base
        };
        assert!(
            g.validate(&ok).is_ok(),
            "an in-bounds long clip must pass: {ok:?}"
        );
    }
}
