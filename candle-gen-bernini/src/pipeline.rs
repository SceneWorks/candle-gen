//! The native Bernini **renderer** provider (sc-10994, the candle/CUDA sibling of `mlx-gen-bernini`'s
//! renderer, sc-4706): loads the dual-expert Wan2.2-T2V-A14B snapshot (converted from
//! `ByteDance/Bernini-Diffusers`, or a stock Wan2.2-T2V-A14B tier) + the UMT5/VAE/tokenizer, and runs
//! the boundary-switched, APG-guided denoise in **spatial latent space**, decoding to an image (1
//! frame) or video.
//!
//! Mirrors [`candle_gen_wan::wan14b`]'s dual-expert staging (UMT5 → high/low experts → z16 VAE) with the
//! plain-CFG combine replaced by the Bernini renderer's guided velocity: for the caption-only render the
//! default mode is `t2v_apg` (APG in x-space; the reference's `resolve_mode(None,false,false)`), with
//! plain CFG (`t2v`) selectable via `video_mode="t2v"`. Dual-expert switch: **high-noise** expert while
//! the integer timestep `≥ switch_dit_boundary·1000`, **low-noise** below — and on the first low-noise
//! step all omegas are scaled once by `OMEGA_SCALE` (the reference's `omega_scale`).
//!
//! **Scope (sc-11004):** every renderer guidance mode runs through [`crate::forward`]'s token-axis packed
//! conditioning — the text-only `t2v`/`t2v_apg` **and** the packed source-id conditioning modes
//! (`v2v`/`v2v_chain`/`v2v_apg`/`r2v_apg`/`rv2v`). A `Reference` (i2i), `MultiReference` (r2v refs), or
//! `VideoClip` (v2v/rv2v) is VAE-encoded to a z16 source latent, patch-embedded with its own source-id
//! RoPE, and packed on the DiT token axis with the noisy target; the target tokens are sliced back out to
//! the velocity. The Qwen2.5-VL planner / MAR / ViT-guidance path (the `*_wapg` modes) remains a follow-up
//! (the planner seams are sc-10995).
//!
//! **Dtypes:** UMT5 + z16 VAE run **f32**; the two experts run **bf16** (norms/modulation upcast to
//! f32); APG runs f32. `backend = "candle"`, `mac_only = false`. Q4/Q8 is a **packed tier** (the two
//! experts load through the sc-10025 packed-detect seam), streamed one expert at a time.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use candle_gen::gen_core::{
    self, Capabilities, Conditioning, ConditioningKind, GenerationOutput, GenerationRequest,
    Generator, LoadSpec, Modality, ModelDescriptor, Progress, Quant, WeightsSource,
};
use candle_gen::{CandleError, Result as CResult};

use candle_gen_wan::config::{
    TextEncoderConfig, TransformerConfig, Vae16Config, DEFAULT_FRAMES_14B, MAX_AREA_14B,
    NEGATIVE_FALLBACK, NUM_TRAIN_TIMESTEPS, SIZE_MULTIPLE_14B, VAE16_STRIDE_SPATIAL,
    VAE16_STRIDE_TEMPORAL,
};
use candle_gen_wan::pipeline::{create_noise, frames_to_images};
use candle_gen_wan::scheduler::{flow_sigmas, FlowScheduler, Sampler};
use candle_gen_wan::text_encoder::Umt5Encoder;
use candle_gen_wan::transformer::WanTransformer;
use candle_gen_wan::vae16::WanVae16;

use crate::config::{resolve_mode, BerniniKnobs, Defaults};
use crate::forward::{guided_velocity, num_momentum_buffers, GuidanceParams, PackedForward};
use crate::guidance::MomentumBuffer;
use crate::preprocess::{encode_image, encode_videoclip};

/// The experts run bf16 (the diffusers weights load as bf16, matching the 5B/14B regime); UMT5 + VAE f32.
const DIT_DTYPE: DType = DType::BF16;
const ENC_DTYPE: DType = DType::F32;
const VAE_DTYPE: DType = DType::F32;
/// The A14B DiT emits 16-channel latents (z16 VAE).
const Z_DIM: usize = 16;

/// SceneWorks/engine model id (matches `mlx-gen-bernini`'s renderer so a consumer resolves the same
/// engine across backends). A still image is `num_frames == 1`.
pub const MODEL_ID: &str = "bernini_renderer";

/// Stable identity + advertised capabilities for the Bernini renderer (Wan2.2-A14B dual-expert with
/// APG guidance; caption→video/image). `backend = "candle"`, `mac_only = false`.
///
/// Advertises the **packed source-id conditioning** surface (sc-11004): a single `Reference` (i2i), a
/// `MultiReference` set (r2v reference images), or a `VideoClip` (v2v/rv2v) — each VAE-encoded to a z16
/// source latent and packed on the DiT token axis with its own source-id RoPE. The text-only render
/// (t2v/t2v_apg) stays the default when no conditioning is present. Q4/Q8 are **packed tiers**
/// (pre-quantized; the loaders read the `.scales` siblings).
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID,
        family: "bernini",
        backend: "candle",
        modality: Modality::Video,
        capabilities: Capabilities {
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: false,
            conditioning: vec![
                ConditioningKind::Reference,
                ConditioningKind::MultiReference,
                ConditioningKind::VideoClip,
            ],
            // LoRA/quant-adapter surface is a follow-on; the renderer ships dense bf16 / packed q4/q8.
            supports_lora: false,
            supports_lokr: false,
            // Curated `uni_pc` (sc-7296) → Wan's native UniPC; `euler` flow Euler. Legacy `unipc` alias.
            samplers: vec!["uni_pc", "euler", "unipc"],
            schedulers: Vec::new(),
            supported_guidance_methods: vec![],
            min_size: 16,
            max_size: 1280,
            max_count: 1,
            mac_only: false,
            supported_quants: &[Quant::Q4, Quant::Q8],
            supports_kv_cache: false,
            requires_sigma_shift: false,
        },
    }
}

/// The heavy resident components, loaded lazily on first generate and cached.
struct Components {
    te: Umt5Encoder,
    /// `transformer/` — the **high-noise** expert (timestep ≥ boundary).
    high: WanTransformer,
    /// `transformer_2/` — the **low-noise** expert (timestep < boundary).
    low: WanTransformer,
    vae: WanVae16,
    /// UMT5 tokenizer, parsed **once** at component load and reused across the pos/neg encodes.
    tok: TextTokenizer,
}

/// A loaded Bernini renderer: resolved Bernini knobs + the snapshot dir, with the heavy components
/// (UMT5, the two experts, the z16 VAE) loaded lazily on the first `generate` and cached.
pub struct BerniniRenderer {
    descriptor: ModelDescriptor,
    knobs: BerniniKnobs,
    root: PathBuf,
    device: Device,
    components: Mutex<Option<Arc<Components>>>,
}

impl BerniniRenderer {
    fn component_vb(&self, sub: &str, dtype: DType) -> CResult<VarBuilder<'static>> {
        candle_gen::component_vb(&self.root, sub, dtype, &self.device, "bernini_renderer")
    }

    fn load_components(&self) -> CResult<Components> {
        let te = Umt5Encoder::new(
            &TextEncoderConfig::umt5_xxl(),
            self.component_vb("text_encoder", ENC_DTYPE)?,
        )?;
        let dit_cfg = TransformerConfig::t2v_14b();
        // Sequential/streaming load: build (and, for a packed tier, dequant-detect) one expert's
        // VarBuilder at a time so only one expert's staging is resident at a time (mirrors wan14b).
        // transformer/ = high-noise, transformer_2/ = low-noise (diffusers WanPipeline convention).
        let high = WanTransformer::new(&dit_cfg, self.component_vb("transformer", DIT_DTYPE)?)?;
        let low = WanTransformer::new(&dit_cfg, self.component_vb("transformer_2", DIT_DTYPE)?)?;
        // Build the VAE **with its encoder**: the packed source-id conditioning modes (i2i/v2v/r2v,
        // sc-11004) VAE-encode the Reference/MultiReference/VideoClip media into z16 source latents. The
        // text-only render (t2v/t2v_apg) only uses the decoder, but the encoder is resident either way
        // (one component build, cached; mirrors wan14b I2V).
        let vae = WanVae16::new_with_encoder(
            &Vae16Config::wan21(),
            self.component_vb("vae", VAE_DTYPE)?,
        )?;
        let tok = build_tokenizer(&self.root)?;
        Ok(Components {
            te,
            high,
            low,
            vae,
            tok,
        })
    }

    fn components(&self) -> CResult<Arc<Components>> {
        candle_gen::cached(&self.components, || Ok(Arc::new(self.load_components()?)))
    }

    /// Tokenize + UMT5-encode `prompt` → `[1, 512, 4096]` (f32), zero-padded/truncated to `max_length`
    /// (the DiT cross-attends over the 512-padded context — the Wan training convention, sc-3697). The
    /// empty-prompt guard (sc-7078) emits one pad token so a 0-length sequence never reaches the CUDA
    /// embedding gather. Replicates `candle_gen_wan`'s crate-private `umt5_encode_padded`.
    fn encode(&self, comps: &Components, prompt: &str) -> CResult<Tensor> {
        let te_cfg = TextEncoderConfig::umt5_xxl();
        let out = comps
            .tok
            .tokenize(prompt)
            .map_err(|e| CandleError::Msg(format!("bernini_renderer: tokenize: {e}")))?;
        let mut ids: Vec<u32> = out.ids.iter().map(|&i| i as u32).collect();
        if ids.is_empty() {
            ids.push(te_cfg.pad_token_id as u32);
        }
        let len = ids.len();
        let input_ids = Tensor::from_vec(ids, (1, len), &self.device)?;
        let embeds = comps.te.encode(&input_ids)?.to_dtype(ENC_DTYPE)?; // [1, S, 4096]
        let max_len = te_cfg.max_length;
        let dim = embeds.dim(2)?;
        let padded = match len.cmp(&max_len) {
            std::cmp::Ordering::Less => {
                let pad = Tensor::zeros((1, max_len - len, dim), ENC_DTYPE, &self.device)?;
                Tensor::cat(&[&embeds, &pad], 1)?
            }
            std::cmp::Ordering::Greater => embeds.narrow(1, 0, max_len)?,
            std::cmp::Ordering::Equal => embeds,
        };
        Ok(padded)
    }

    fn render(
        &self,
        req: &GenerationRequest,
        comps: &Components,
        on_progress: &mut dyn FnMut(Progress),
    ) -> CResult<GenerationOutput> {
        let frames = req.frames.unwrap_or(DEFAULT_FRAMES_14B).max(1);
        let width = req.width;
        let height = req.height;
        let steps = req
            .steps
            .map(|s| s as usize)
            .unwrap_or(Defaults::STEPS)
            .max(1);
        let seed = req.seed.unwrap_or_else(gen_core::default_seed);
        let fps = req.fps.unwrap_or(Defaults::FPS);
        let sampler = Sampler::parse(req.sampler.as_deref());
        // The reference builds the scheduler with `flow_shift = config.shift` (the Bernini knob).
        let shift = req
            .scheduler_shift
            .map(|s| s as f64)
            .unwrap_or(self.knobs.shift as f64);

        // Conditioning split (mirrors mlx): VideoClip → videos, Reference/MultiReference → images. The
        // presence of each drives `resolve_mode`, which picks the guidance [`Mode`] (text-only ⇒
        // `t2v_apg`, images ⇒ `v2v`, video ⇒ `v2v_apg`, both ⇒ `rv2v`, overridable via `video_mode`).
        let has_video = req
            .conditioning
            .iter()
            .any(|c| matches!(c, Conditioning::VideoClip { .. }));
        let has_image = req.conditioning.iter().any(|c| {
            matches!(
                c,
                Conditioning::Reference { .. } | Conditioning::MultiReference { .. }
            )
        });
        let mode = resolve_mode(req.video_mode.as_deref(), has_video, has_image);

        // VAE-encode the source media → z16 conditioning latents (each `[1, 16, T, H8, W8]`). VideoClips
        // become `videos`; Reference / MultiReference images become `images`. The packed forward assigns
        // each its source-id RoPE (videos first) and concatenates them on the DiT token axis.
        let mut videos: Vec<Tensor> = Vec::new();
        let mut images: Vec<Tensor> = Vec::new();
        for c in &req.conditioning {
            match c {
                Conditioning::VideoClip { frames: fr, .. } => {
                    videos.push(encode_videoclip(
                        &comps.vae,
                        fr,
                        width,
                        height,
                        &self.device,
                    )?);
                }
                Conditioning::Reference { image, .. } => {
                    images.push(encode_image(
                        &comps.vae,
                        image,
                        width,
                        height,
                        &self.device,
                    )?);
                }
                Conditioning::MultiReference { images: imgs } => {
                    for im in imgs {
                        images.push(encode_image(&comps.vae, im, width, height, &self.device)?);
                    }
                }
                other => {
                    return Err(CandleError::Msg(format!(
                        "bernini_renderer: unsupported conditioning kind {:?} (only Reference / \
                         MultiReference / VideoClip are packed as source-id conditioning)",
                        other.kind()
                    )));
                }
            }
        }

        // --- Text encode (pos + neg) once; project to each expert's context (per-expert embedder) ---
        let pos = self.encode(comps, &req.prompt)?;
        let neg_prompt = req.negative_prompt.as_deref().unwrap_or(NEGATIVE_FALLBACK);
        let neg = self.encode(comps, neg_prompt)?;
        let high_pos = comps.high.embed_text(&pos)?;
        let low_pos = comps.low.embed_text(&pos)?;
        let high_neg = comps.high.embed_text(&neg)?;
        let low_neg = comps.low.embed_text(&neg)?;

        // --- Latent geometry (z16 strides). RoPE is now built per token-segment inside `PackedForward`
        // (each source + the target gets its own source-id-shifted table), so no shared cos/sin here. ---
        let t_lat = ((frames - 1) / VAE16_STRIDE_TEMPORAL + 1) as usize;
        let h_lat = (height / VAE16_STRIDE_SPATIAL) as usize;
        let w_lat = (width / VAE16_STRIDE_SPATIAL) as usize;
        let dit_cfg = TransformerConfig::t2v_14b();

        let mut latents = create_noise(seed, Z_DIM, t_lat, h_lat, w_lat, &self.device)?;
        let mut sched = FlowScheduler::new(sampler, steps, shift);
        // The APG x-space conversion needs this step's flow sigma; the schedule matches the scheduler's.
        let sigmas = flow_sigmas(steps, shift);
        let boundary_ts = self.knobs.switch_dit_boundary as f64 * NUM_TRAIN_TIMESTEPS as f64;
        let total = steps as u32;

        // The packed-forward engine drives **every** mode uniformly: the target (and each conditioning
        // source) is patch-embedded with its source-id RoPE, packed on the token axis, run through one
        // `forward_packed`, and the target tokens sliced back out. For the text-only modes the source set
        // is empty and this reduces to the plain per-expert forward (byte-identical to sc-10994's path).
        let pf = PackedForward::new(
            dit_cfg,
            self.knobs.max_trained_src_id,
            self.knobs.interpolate_src_id,
        );
        // APG momentum buffers persist across steps (0 for the plain modes, 1 for the single-cond `*_apg`
        // modes, 2 for the chained `r2v_apg`).
        let mut mbufs: Vec<MomentumBuffer> = (0..num_momentum_buffers(mode))
            .map(|_| MomentumBuffer::new(Defaults::MOMENTUM))
            .collect();
        let mut switched = false;
        let mut omega_txt = req.guidance.unwrap_or(Defaults::OMEGA_TXT);
        let mut omega_vid = Defaults::OMEGA_VID;
        let mut omega_img = Defaults::OMEGA_IMG;

        // `i` drives the scheduler timestep, the progress counter, AND the flow-sigma lookup (the *_apg
        // modes' x-space conversion) — not a plain element walk over one slice, so the range loop is correct.
        #[allow(clippy::needless_range_loop)]
        for i in 0..steps {
            if req.cancel.is_cancelled() {
                return Err(CandleError::Canceled);
            }
            let t = sched.timestep(i);
            // MoE: high-noise expert at/above the boundary timestep, low-noise below — switching the
            // transformer AND its per-expert text contexts together. `select_expert` scales omega_txt once
            // on the first low-noise step (its own unit test); scale the video/image omegas the same way.
            let prev = switched;
            let (expert, ctx_pos, ctx_neg) =
                match select_expert(t, boundary_ts, &mut switched, &mut omega_txt) {
                    Expert::High => (&comps.high, &high_pos, &high_neg),
                    Expert::Low => (&comps.low, &low_pos, &low_neg),
                };
            if switched && !prev {
                omega_vid *= Defaults::OMEGA_SCALE;
                omega_img *= Defaults::OMEGA_SCALE;
            }
            let g = GuidanceParams {
                omega_vid,
                omega_img,
                omega_txt,
                eta: Defaults::ETA,
                norm_threshold: [Defaults::NORM_THRESHOLD, Defaults::NORM_THRESHOLD],
            };
            let v = guided_velocity(
                &pf, mode, expert, &latents, &videos, &images, t, sigmas[i], ctx_pos, ctx_neg, &g,
                &mut mbufs,
            )?;
            latents = sched.step(&v, &latents)?; // 16-channel latent (out_dim 16)
            on_progress(Progress::Step {
                current: i as u32 + 1,
                total,
            });
        }

        on_progress(Progress::Decoding);
        let decoded = comps.vae.decode(&latents)?;
        let out_images = frames_to_images(&decoded)?;

        // num_frames == 1 ⇒ a still image (t2i). A single latent frame still decodes to one VAE
        // temporal chunk; the still image is the first frame, matching the reference's single-frame PNG.
        if frames == 1 {
            let first = out_images.into_iter().next().ok_or_else(|| {
                CandleError::Msg("bernini_renderer: VAE decode produced no frames".into())
            })?;
            Ok(GenerationOutput::Images(vec![first]))
        } else {
            Ok(GenerationOutput::Video {
                frames: out_images,
                fps,
                audio: None,
            })
        }
    }
}

/// Which dual-expert transformer a denoise step routes through: **high-noise** at/above the boundary
/// timestep (`transformer/`), **low-noise** below (`transformer_2/`) — the diffusers WanPipeline
/// convention mirrored by [`Components`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Expert {
    High,
    Low,
}

/// Pure dual-expert selection for one denoise step (sc-10994) — Bernini's most model-specific renderer
/// delta, factored out of [`BerniniRenderer::render`] so it is unit-testable without GPU weights.
///
/// High-noise expert while the integer timestep `t ≥ boundary_ts`, low-noise below. On the FIRST
/// low-noise step (the high→low transition) all omegas are scaled once by [`Defaults::OMEGA_SCALE`] (the
/// reference's `omega_scale`) via the `switched` latch, and never again on subsequent low steps.
/// `render` maps the returned [`Expert`] onto the resident experts + per-expert text contexts, so this
/// stays behaviorally identical to the previous inline switch.
fn select_expert(t: f64, boundary_ts: f64, switched: &mut bool, omega: &mut f32) -> Expert {
    if t >= boundary_ts {
        Expert::High
    } else {
        if !*switched {
            *switched = true;
            *omega *= Defaults::OMEGA_SCALE;
        }
        Expert::Low
    }
}

/// Build the Bernini renderer UMT5 tokenizer from `root/tokenizer/tokenizer.json` **once** (reused
/// across the pos/neg encodes). Byte-identical [`TokenizerConfig`] to `candle_gen_wan`'s Wan loader.
fn build_tokenizer(root: &Path) -> CResult<TextTokenizer> {
    let te_cfg = TextEncoderConfig::umt5_xxl();
    TextTokenizer::from_file(
        root.join("tokenizer/tokenizer.json"),
        TokenizerConfig {
            max_length: te_cfg.max_length,
            pad_token_id: te_cfg.pad_token_id,
            chat_template: ChatTemplate::None,
            pad_to_max_length: false,
        },
    )
    .map_err(|e| CandleError::Msg(format!("bernini_renderer: load tokenizer: {e}")))
}

/// Load the Bernini renderer from a converted snapshot directory (`text_encoder/`, `transformer/`,
/// `transformer_2/`, `vae/`, `tokenizer/`, + optional `bernini_renderer.json` knobs). The dual-expert
/// snapshot is a Wan2.2-T2V-A14B diffusers layout produced by [`crate::convert`] from
/// `ByteDance/Bernini-Diffusers` (or a stock Wan2.2-T2V-A14B tier for a raw-render validation).
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(
                "bernini_renderer: expected a snapshot directory (text_encoder/ transformer/ \
                 transformer_2/ vae/ tokenizer/), not a single .safetensors file"
                    .into(),
            ))
        }
    };
    // Control / VACE / IP-adapter overlays are not part of the renderer surface.
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(gen_core::Error::Unsupported(
            "bernini_renderer does not support control / VACE / IP-adapter overlays".into(),
        ));
    }
    // `spec.quantize` is a no-op tier-select marker (mirrors wan14b/ltx): a q4/q8 A14B tier is
    // pre-packed (the packed-detect loaders read its `.scales`), a dense tier loads dense — so it does
    // NOT reject here; both experts load through the sc-10025 packed-detect seam.
    let knobs = BerniniKnobs::from_dir(&root);
    let device = candle_gen::default_device()?;
    Ok(Box::new(BerniniRenderer {
        descriptor: descriptor(),
        knobs,
        root,
        device,
        components: Mutex::new(None),
    }))
}

// Link-time self-registration into candle-gen's model registry (epic 3720).
candle_gen::register_generators! { descriptor => load }

/// Force-link hook (keeps the `inventory::submit!` registration from being dead-stripped).
pub fn force_link() {}

impl Generator for BerniniRenderer {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        let id = self.descriptor.id;
        self.descriptor.capabilities.validate_request(id, req)?;
        if req.prompt.is_empty() {
            return Err(gen_core::Error::Msg(format!(
                "{id}: prompt must not be empty"
            )));
        }
        if req.steps == Some(0) {
            return Err(gen_core::Error::Msg(format!(
                "{id}: steps must be >= 1 (an explicit 0 renders undenoised noise)"
            )));
        }
        if !req.width.is_multiple_of(SIZE_MULTIPLE_14B)
            || !req.height.is_multiple_of(SIZE_MULTIPLE_14B)
        {
            return Err(gen_core::Error::Msg(format!(
                "{id}: width/height must be multiples of {SIZE_MULTIPLE_14B} (got {}x{})",
                req.width, req.height
            )));
        }
        let area = req.width as usize * req.height as usize;
        if area > MAX_AREA_14B {
            return Err(gen_core::Error::Msg(format!(
                "{id}: width×height ({}×{} = {area} px) exceeds the max area {MAX_AREA_14B} px \
                 (704×1280); reduce the resolution",
                req.width, req.height
            )));
        }
        if let Some(f) = req.frames {
            if f == 0 || f % 4 != 1 {
                return Err(gen_core::Error::Msg(format!(
                    "{id}: num_frames must be 1 + 4·k (got {f})"
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
        let comps = self.components()?;
        Ok(self.render(req, &comps, on_progress)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::registry;

    #[test]
    fn registers_and_resolves_as_candle_video() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        // load is lazy (components build on first generate), so the registry resolves + `load` succeeds
        // even for a missing dir; the descriptor identity is what we pin here.
        let g = registry::load(MODEL_ID, &spec).expect("bernini_renderer is registered");
        assert_eq!(g.descriptor().id, MODEL_ID);
        assert_eq!(g.descriptor().family, "bernini");
        assert_eq!(g.descriptor().backend, "candle");
        assert_eq!(g.descriptor().modality, Modality::Video);
        assert!(!g.descriptor().capabilities.mac_only);
    }

    #[test]
    fn descriptor_surface() {
        let d = descriptor();
        assert_eq!(d.id, MODEL_ID);
        assert!(d.capabilities.supports_guidance);
        assert!(d.capabilities.supports_negative_prompt);
        assert!(!d.capabilities.supports_true_cfg);
        // Q4 and Q8 packed tiers both supported (streamed one expert at a time).
        assert!(d.capabilities.supported_quants.contains(&Quant::Q4));
        assert!(d.capabilities.supported_quants.contains(&Quant::Q8));
        assert!(d.capabilities.samplers.contains(&"uni_pc"));
        assert!(d.capabilities.samplers.contains(&"unipc")); // legacy alias
                                                             // Packed source-id conditioning surface (sc-11004): reference image / multi-reference / video clip.
        assert!(d.capabilities.accepts(ConditioningKind::Reference));
        assert!(d.capabilities.accepts(ConditioningKind::MultiReference));
        assert!(d.capabilities.accepts(ConditioningKind::VideoClip));
    }

    #[test]
    fn load_rejects_single_file_and_overlays() {
        // single-file source
        let f = LoadSpec::new(WeightsSource::File("/tmp/w.safetensors".into()));
        assert!(load(&f).is_err());
        // Quant is accepted (no-op packed-tier marker): load is lazy, so it succeeds past the marker.
        let quant = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_quant(Quant::Q8);
        assert!(
            load(&quant).is_ok(),
            "q8 is a packed-tier select marker, not rejected"
        );
        let quant4 = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_quant(Quant::Q4);
        assert!(load(&quant4).is_ok());
    }

    #[test]
    fn validate_enforces_surface() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry::load(MODEL_ID, &spec).unwrap();
        let ok = GenerationRequest {
            prompt: "a cat walking across a sunny garden".into(),
            width: 256,
            height: 256,
            guidance: Some(4.0),
            frames: Some(17),
            sampler: Some("uni_pc".into()),
            ..Default::default()
        };
        assert!(g.validate(&ok).is_ok());
        for bad in [
            // empty prompt
            GenerationRequest::default(),
            // frames not ≡ 1 (mod 4)
            GenerationRequest {
                prompt: "x".into(),
                width: 256,
                height: 256,
                frames: Some(16),
                ..Default::default()
            },
            // size not a multiple of 16
            GenerationRequest {
                prompt: "x".into(),
                width: 300,
                height: 256,
                ..Default::default()
            },
            // explicit zero steps
            GenerationRequest {
                prompt: "x".into(),
                width: 256,
                height: 256,
                steps: Some(0),
                ..Default::default()
            },
            // over the max-area envelope
            GenerationRequest {
                prompt: "x".into(),
                width: 1280,
                height: 1024,
                frames: Some(17),
                sampler: Some("uni_pc".into()),
                ..Default::default()
            },
        ] {
            assert!(g.validate(&bad).is_err(), "should reject: {bad:?}");
        }
    }

    /// The dual-expert switch + omega-scale-once latch (the story AC's "expert selection" test). Asserts
    /// (a) high→low transition happens exactly at the boundary, (b) omega scales exactly once on the
    /// first sub-boundary step and not again, (c) high expert above the boundary.
    #[test]
    fn select_expert_switches_at_boundary_and_scales_omega_once() {
        let boundary =
            BerniniKnobs::default().switch_dit_boundary as f64 * NUM_TRAIN_TIMESTEPS as f64; // 0.875 * 1000 = 875.0
        let base = Defaults::OMEGA_TXT;

        let mut switched = false;
        let mut omega = base;

        // (c) Above the boundary → high expert; no switch, omega untouched.
        assert_eq!(
            select_expert(900.0, boundary, &mut switched, &mut omega),
            Expert::High
        );
        assert!(!switched);
        assert_eq!(omega, base);

        // (a) Exactly AT the boundary is still high-noise (the switch is `t >= boundary_ts`).
        assert_eq!(
            select_expert(boundary, boundary, &mut switched, &mut omega),
            Expert::High
        );
        assert!(!switched);
        assert_eq!(omega, base);

        // (a) First step below the boundary → low expert, latch flips.
        assert_eq!(
            select_expert(boundary - 1.0, boundary, &mut switched, &mut omega),
            Expert::Low
        );
        assert!(switched);
        // (b) omega scaled exactly once.
        assert_eq!(omega, base * Defaults::OMEGA_SCALE);

        // (b) Subsequent low-noise steps do NOT scale omega again.
        let after_first = omega;
        assert_eq!(
            select_expert(100.0, boundary, &mut switched, &mut omega),
            Expert::Low
        );
        assert_eq!(omega, after_first);
        assert_eq!(
            select_expert(0.0, boundary, &mut switched, &mut omega),
            Expert::Low
        );
        assert_eq!(omega, after_first);
    }

    /// Scheduler wiring: the `boundary_ts` axis and the `flow_sigmas(steps, shift)` the APG x-space
    /// conversion indexes both share the `FlowScheduler`'s internal sigma indexing — `timestep(i) ==
    /// σ_i · NUM_TRAIN_TIMESTEPS`. Guards the sigma-vs-index alignment the APG conversion depends on.
    #[test]
    fn flow_sigmas_align_with_scheduler_timesteps() {
        let steps = 8;
        let shift = 3.0_f64;
        let sched = FlowScheduler::new(Sampler::UniPC, steps, shift);
        let sigmas = flow_sigmas(steps, shift);
        assert_eq!(sigmas.len(), steps + 1); // terminal 0.0
        for (i, &sigma) in sigmas.iter().enumerate().take(steps) {
            // sched.timestep uses the f64 sigmas; flow_sigmas is the f32-cast schedule the APG loop
            // indexes at the same `i`. They align within f32 rounding on the 0..1000 timestep axis.
            let expected = sigma as f64 * NUM_TRAIN_TIMESTEPS as f64;
            assert!(
                (sched.timestep(i) - expected).abs() < 1e-3,
                "step {i}: timestep {} vs σ_i·N {expected}",
                sched.timestep(i)
            );
        }
        // boundary_ts (render's expert switch) lives on that same 0..1000 timestep axis.
        let boundary_ts =
            BerniniKnobs::default().switch_dit_boundary as f64 * NUM_TRAIN_TIMESTEPS as f64;
        assert!((boundary_ts - 875.0).abs() < 1e-9);
    }
}
