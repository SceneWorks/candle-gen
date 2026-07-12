//! # candle-gen-qwen-image
//!
//! The **Qwen-Image** provider crate for [`candle-gen`](candle_gen) — the candle (Windows/CUDA)
//! sibling of `mlx-gen-qwen-image`. Qwen-Image has **no** `candle-transformers` reference: the
//! 60-layer dual-stream MMDiT ([`transformer`]), the causal-Conv3d VAE ([`vae`]), and the Qwen2.5-VL
//! prompt-embeds path ([`text_encoder`]) are all ported here from the macOS provider.
//!
//! **txt2img (sc-3696):** [`QwenImageGenerator::generate`] runs Qwen2.5-VL (last normed hidden state,
//! 34 template tokens dropped → 3584-wide `prompt_embeds`) → the MMDiT (interleaved 3-axis RoPE,
//! dynamic-μ flow-match Euler, **true CFG** with norm-rescale) → the AutoencoderKLQwenImage decoder,
//! registered under `"qwen_image"`. Deterministic CPU-seeded noise (sc-3673); tokenization reuses
//! gen-core's [`TextTokenizer`] with [`ChatTemplate::QwenImage`].
//!
//! **Dtypes:** the Qwen2.5-VL encoder runs in **f32** (the fork rounds only the embeds to bf16) and
//! the 20B MMDiT in **bf16** (its native checkpoint dtype) — ~74 GB resident, which fits the 96 GB
//! Blackwell where an all-f32 load (~113 GB) would not.
//!
//! **First-slice surface:** txt2img only. The mlx provider's img2img / Edit / ControlNet / Lightning
//! / LoRA / quantization surface is **deferred** and rejected. `backend = "candle"`, `mac_only = false`.

// Qwen-Image-Edit inference adapter merge (sc-6220, epic 5480): fold a LoRA/LoKr `.safetensors` delta
// into the dense MMDiT weights at load — the Qwen-Image-Edit-2511-Lightning few-step distill, plus
// general Qwen-family LoRA/LoKr. Consumed by `edit::QwenEdit::load`.
pub mod adapters;
pub mod config;
// Shared scaffolding for the control lane (`control_fun`): the component loader, prompt encoder,
// control-image preprocessor, and VAE-output converter, parameterized by an error `label` (sc-9011,
// F-074). De-duplicates what used to be verbatim copies; the lane's outputs are preserved exactly.
mod control_common;
// Qwen-Image **2512-Fun-Controlnet-Union** (VACE) control — the candle structural-control lane
// (sc-8350, mirrors mlx sc-8267). A `control_img_in` patch embedder feeds a control state threaded
// through 5 VACE control blocks (seeded by `before_proj`), each emitting a zero-init `after_proj` hint
// the base 2512 MMDiT adds at `control_layers = [0, 12, 24, 36, 48]`. Input-agnostic (pose/canny/depth
// share one path, no mode index). A bespoke provider the worker drives directly. This is the sole Qwen
// control engine — the retired InstantX ControlNet lane (`control`) was removed in sc-9868 (its MLX
// twin was retired in sc-8267 and the worker repointed InstantX→2512-Fun in sc-8350).
pub mod control_fun;
// Qwen-Image-Edit (img2img / reference) — the candle edit lane (sc-5487, epic 5480). The Qwen2.5-VL
// vision tower + image processor + VL splice turn a reference image + edit prompt into vision-
// conditioned prompt embeds (Slice A); the dual-latent `QwenEdit` provider (Slice B) VAE-encodes each
// reference, concatenates it after the noise, and denoises with the reference grids in the RoPE.
pub mod edit;
pub mod image_processor;
pub mod pipeline;
// ComfyUI single-file Qwen-Image → in-memory remap seam (epic 10451 Phase 2b): strip the
// `model.diffusion_model.` prefix + upcast the plain `fp8_e4m3fn` DiT to bf16 (sc-10670), and remap the
// native WAN-VAE keys of the tree's `vae/qwen_image_vae.safetensors` to the diffusers schema (sc-10830)
// — making a user's existing ComfyUI Qwen-Image DiT + VAE loadable in place via
// `VarBuilder::from_tensors`. The Qwen2.5-VL text encoder (scaled-fp8, sc-10671) and tokenizer still
// come from a resident Qwen-Image diffusers snapshot. Entry point: [`load_from_comfyui_dit`].
mod comfyui;
// Qwen-Image DiT packed-load seam (sc-9415, sc-9089 umbrella): route every DiT `Linear` through the
// shared `candle_gen::quant` packed-detect so the pre-quantized MLX tiers (`SceneWorks/qwen-image-mlx`
// + `qwen-image-edit-2511-mlx` q4/q8) load straight from the packed parts. The fused Qwen2.5-VL text
// encoder (LM + vision tower) and the VAE stay dense in every tier (see the module docs).
pub mod quant;
pub mod rope;
pub mod text_encoder;
pub mod transformer;
pub mod vae;
pub mod vision;
pub mod vision_language;
pub mod vl_tokenizer;

pub use control_fun::{
    QwenFunControl, QwenFunControlPaths, QwenFunControlRequest, CONTROL_IN_DIM, CONTROL_LAYERS,
    DEFAULT_CONTROL_SCALE,
};
pub use edit::{QwenEdit, QwenEditPaths, QwenEditRequest};
pub use vision_language::{load_vision_language_encoder, QwenVisionLanguageEncoder};

/// Qwen-Image 2512-Fun-Controlnet-Union (VACE) real-weight GPU validation (sc-8350) — env-driven,
/// `#[ignore]`d.
#[cfg(test)]
mod control_fun_validate;

/// Qwen-Image-Edit vision-language encoder real-weight GPU validation (sc-5487) — env-driven, `#[ignore]`d.
#[cfg(test)]
mod vision_validate;

/// Qwen-Image-Edit full provider real-weight GPU validation (sc-5487) — env-driven, `#[ignore]`d.
#[cfg(test)]
mod edit_validate;

/// In-place ComfyUI Qwen-Image VAE (native WAN-VAE keys) real-weight GPU validation (sc-10830) —
/// env-driven, `#[ignore]`d.
#[cfg(test)]
mod comfyui_vae_validate;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::{DType, Device, IndexOp, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use candle_gen::gen_core::{
    self, Capabilities, GenerationOutput, GenerationRequest, Generator, Image, LoadSpec, Modality,
    ModelDescriptor, OffloadPolicy, PidWeights, Progress, Quant, WeightsSource,
};
use candle_gen::{CandleError, LatentDecoder, Result as CResult};
use candle_gen_pid::PidEngine;

/// The PiD backbone (latent-space) tag for the Qwen-Image VAE — resolves to the `qwenimage` `2kto4k`
/// student + 4× SR (`candle_gen_pid::registry`). Shared with Krea (which reuses [`vae::QwenVae`]).
const PID_BACKBONE: &str = "qwenimage";

use config::{
    TextEncoderConfig, TransformerConfig, DEFAULT_GUIDANCE, DEFAULT_STEPS, MODEL_ID,
    NEGATIVE_FALLBACK, SIZE_MULTIPLE,
};
use text_encoder::QwenTextEncoder;
use transformer::QwenTransformer;
use vae::QwenVae;

/// The transformer is the 20B bottleneck — keep it bf16 (native dtype). The encoder + VAE run f32.
const DIT_DTYPE: DType = DType::BF16;
const ENC_DTYPE: DType = DType::F32;

#[derive(Clone)]
struct Components {
    te: Arc<QwenTextEncoder>,
    transformer: Arc<QwenTransformer>,
    vae: Arc<QwenVae>,
    /// Qwen tokenizer, loaded+parsed **once** at component load and reused across every prompt/branch
    /// encode (sc-8991 / F-011) instead of re-parsing `tokenizer.json` per request.
    tokenizer: Arc<TextTokenizer>,
    /// Optional NVIDIA PiD super-resolving decoder (epic 7840 / sc-7853), loaded once when the model
    /// was loaded with `LoadSpec::pid`. `None` ⇒ the native `QwenVae` decode (the default path).
    pid: Option<Arc<PidEngine>>,
}

struct Pipeline {
    te_cfg: TextEncoderConfig,
    dit_cfg: TransformerConfig,
    root: PathBuf,
    device: Device,
    /// The `LoadSpec::pid` component (converted PiD checkpoint + gemma dir), if the caller opted in.
    pid_spec: Option<PidWeights>,
    /// An in-place ComfyUI Qwen-Image DiT single-file (epic 10451 Phase 2b, sc-10670). When set, the
    /// transformer is built from this file (prefix-strip + fp8→bf16, see [`comfyui`]) instead of the
    /// snapshot's `transformer/` dir; the text encoder / tokenizer still come from `root`. `None`
    /// on the registry path (dense/packed snapshot transformer).
    comfyui_dit: Option<PathBuf>,
    /// An in-place ComfyUI Qwen-Image VAE single-file (epic 10451 Phase 2b, sc-10830, `vae/
    /// qwen_image_vae.safetensors`, native WAN-VAE keys). When set, the VAE is built from this file
    /// (key-remapped, see [`comfyui::remap_vae_wan_to_diffusers`]); when `None` the VAE falls back to
    /// the snapshot's `vae/` dir. Independent of `comfyui_dit` (either can be in place).
    comfyui_vae: Option<PathBuf>,
}

impl Pipeline {
    fn load(root: &Path, device: &Device, pid_spec: Option<PidWeights>) -> Self {
        Self {
            te_cfg: TextEncoderConfig::qwen_image(),
            dit_cfg: TransformerConfig::qwen_image(),
            root: root.to_path_buf(),
            device: device.clone(),
            pid_spec,
            comfyui_dit: None,
            comfyui_vae: None,
        }
    }

    /// Same as [`load`](Self::load) but with the transformer (and, when `comfyui_vae` is set, the VAE)
    /// sourced from in-place ComfyUI single-files (sc-10670 / sc-10830). `root` is the resident
    /// Qwen-Image diffusers snapshot that supplies the text encoder / tokenizer (and the VAE when
    /// `comfyui_vae` is `None`).
    fn load_comfyui(
        root: &Path,
        device: &Device,
        comfyui_dit: PathBuf,
        comfyui_vae: Option<PathBuf>,
    ) -> Self {
        Self {
            te_cfg: TextEncoderConfig::qwen_image(),
            dit_cfg: TransformerConfig::qwen_image(),
            root: root.to_path_buf(),
            device: device.clone(),
            pid_spec: None,
            comfyui_dit: Some(comfyui_dit),
            comfyui_vae,
        }
    }

    fn component_vb(&self, sub: &str, dtype: DType) -> CResult<VarBuilder<'static>> {
        let dir = self.root.join(sub);
        if !dir.is_dir() {
            return Err(CandleError::Msg(format!(
                "qwen-image snapshot is missing the {sub}/ dir (expected a Qwen-Image diffusers \
                 snapshot at {})",
                self.root.display()
            )));
        }
        // Shared sorted-`.safetensors` → mmap (sc-8999 / F-019); the crafted "missing dir" message
        // above stays local (it names the expected Qwen-Image snapshot).
        let files = candle_gen::sorted_safetensors(&dir, "qwen-image")?;
        candle_gen::mmap_var_builder(&files, dtype, &self.device)
    }

    fn load_components(&self) -> CResult<Components> {
        // The fused Qwen2.5-VL text encoder (LM + vision tower) ships DENSE bf16 in every tier — the
        // MLX convert job quantizes only the transformer — so the TE loader is unchanged (it guards
        // against an unexpected `.scales`; see `text_encoder`). The DiT packed-detects: read the packed
        // `group_size` from `transformer/config.json` (default 64 when dense/absent, never silent dense
        // — `candle_gen::quant::PackedConfig` resolves a missing group_size to 64).
        let te = QwenTextEncoder::new(&self.te_cfg, self.component_vb("text_encoder", ENC_DTYPE)?)?;
        let transformer = match &self.comfyui_dit {
            // In-place ComfyUI DiT single-file (sc-10670): remap keys + upcast fp8→bf16 into an
            // in-memory tensor map, then build via `VarBuilder::from_tensors` (the same in-memory path
            // the packed/adapter loads use). Dense bf16 after the upcast, so the default group size.
            Some(dit_file) => {
                let dit_map = candle_gen::candle_core::safetensors::load(dit_file, &Device::Cpu)?;
                let dit_map = comfyui::remap_and_cast_comfyui_dit(dit_map, DIT_DTYPE)?;
                let dit_vb = VarBuilder::from_tensors(dit_map, DIT_DTYPE, &self.device);
                // Dense bf16 after the upcast — the group size is inert on the dense path (no `.scales`
                // siblings in the map), so the shared default.
                QwenTransformer::new_gs(&self.dit_cfg, dit_vb, candle_gen::quant::MLX_GROUP_SIZE)?
            }
            None => {
                let gs = transformer_group_size(&self.root.join("transformer"));
                QwenTransformer::new_gs(
                    &self.dit_cfg,
                    self.component_vb("transformer", DIT_DTYPE)?,
                    gs,
                )?
            }
        };
        let vae = match &self.comfyui_vae {
            // In-place ComfyUI VAE single-file (sc-10830): remap the native WAN-VAE keys to the
            // diffusers schema in memory, then build via `VarBuilder::from_tensors` at ENC_DTYPE — the
            // f32 upcast happens on `get` (the map's bf16 → f32), byte-matching the snapshot VAE mmap.
            Some(vae_file) => {
                let vae_map = candle_gen::candle_core::safetensors::load(vae_file, &Device::Cpu)?;
                let vae_map = comfyui::remap_vae_wan_to_diffusers(vae_map)?;
                let vae_vb = VarBuilder::from_tensors(vae_map, ENC_DTYPE, &self.device);
                QwenVae::new(vae_vb)?
            }
            None => QwenVae::new(self.component_vb("vae", ENC_DTYPE)?)?,
        };
        let tokenizer = TextTokenizer::from_file(
            self.root.join("tokenizer/tokenizer.json"),
            TokenizerConfig {
                max_length: self.te_cfg.max_length,
                pad_token_id: self.te_cfg.pad_token_id,
                chat_template: ChatTemplate::QwenImage,
                pad_to_max_length: false,
            },
        )
        .map_err(|e| CandleError::Msg(format!("qwen-image: load tokenizer: {e}")))?;
        // Load the optional PiD super-resolving decoder once (epic 7840 / sc-7853) when the caller
        // opted in via `LoadSpec::pid`; otherwise `None` and the render path uses the native QwenVae.
        let pid = match &self.pid_spec {
            Some(spec) => Some(Arc::new(PidEngine::from_spec(
                spec,
                PID_BACKBONE,
                &self.device,
            )?)),
            None => None,
        };
        Ok(Components {
            te: Arc::new(te),
            transformer: Arc::new(transformer),
            vae: Arc::new(vae),
            tokenizer: Arc::new(tokenizer),
            pid,
        })
    }

    /// Tokenize + encode `prompt` → `prompt_embeds` `[1, seq, 3584]` at the DiT dtype (bf16). `tok` is
    /// the cached component tokenizer (sc-8991 / F-011) — parsed once at load, reused across encodes.
    fn encode(&self, te: &QwenTextEncoder, tok: &TextTokenizer, prompt: &str) -> CResult<Tensor> {
        let out = tok
            .tokenize(prompt)
            .map_err(|e| CandleError::Msg(format!("qwen-image: tokenize: {e}")))?;
        let len = out.ids.len();
        let ids: Vec<u32> = out.ids.iter().map(|&i| i as u32).collect();
        let input_ids = Tensor::from_vec(ids, (1, len), &self.device)?;
        Ok(te.prompt_embeds(&input_ids)?.to_dtype(DIT_DTYPE)?)
    }

    fn render(
        &self,
        req: &GenerationRequest,
        comps: &Components,
        on_progress: &mut dyn FnMut(Progress),
    ) -> CResult<Vec<Image>> {
        let guidance = req.guidance.unwrap_or(DEFAULT_GUIDANCE);

        let pos_embeds = self.encode(&comps.te, &comps.tokenizer, &req.prompt)?;
        // True CFG: build the negative branch unless guidance is a no-op (≤ 1.0).
        let neg_embeds = if guidance > 1.0 {
            let neg = req.negative_prompt.as_deref().unwrap_or(NEGATIVE_FALLBACK);
            Some(self.encode(&comps.te, &comps.tokenizer, neg)?)
        } else {
            None
        };

        // The DiT / VAE / optional PiD are already resident in the cross-request-cached `comps`; run the
        // shared denoise + decode tail (byte-identical to the sequential path).
        self.denoise_and_decode(
            req,
            &comps.transformer,
            &comps.vae,
            comps.pid.as_deref(),
            &pos_embeds,
            neg_embeds.as_ref(),
            guidance,
            on_progress,
        )
    }

    /// The shared denoise + decode tail (epic 10765 Phase 1c, sc-10867): given already-encoded prompt
    /// embeds and the just-resident DiT / VAE / optional PiD decoder, run the per-image flow sampler and
    /// decode. Called by BOTH the resident [`render`](Self::render) (which borrows the components out of
    /// the cross-request-cached [`Components`]) and the sequential
    /// [`render_sequential`](Self::render_sequential) (which owns just-loaded components after the text
    /// encoder was dropped). Byte-identical between the two paths — only the load/free schedule of the
    /// components differs, not this code. The borrows (`&QwenTransformer` / `&QwenVae` /
    /// `Option<&PidEngine>`) let both an `Arc`-cached and an owned component feed the same loop.
    #[allow(clippy::too_many_arguments)]
    fn denoise_and_decode(
        &self,
        req: &GenerationRequest,
        transformer: &QwenTransformer,
        vae: &QwenVae,
        pid: Option<&PidEngine>,
        pos_embeds: &Tensor,
        neg_embeds: Option<&Tensor>,
        guidance: f32,
        on_progress: &mut dyn FnMut(Progress),
    ) -> CResult<Vec<Image>> {
        let steps = resolve_steps(req.steps);
        let base_seed = req.seed.unwrap_or_else(gen_core::default_seed);
        let (lat_h, lat_w) = pipeline::latent_dims(req.width, req.height);

        // Routed through the unified curated sampler/scheduler framework (epic 7114 P4, sc-7123): the
        // `scheduler` axis picks the σ schedule over the production dynamic-μ shift (`native` = the
        // legacy `qwen_sigmas`), the `sampler` axis picks the integrator. The DEFAULT (`euler` over the
        // native schedule) is the N1 no-op — algebraically the legacy `euler_step` loop. The model is
        // fed the raw sigma (`Sigma` convention), and Qwen-Image is **true CFG** (a positive + negative
        // forward + norm-rescaled blend per step), so the whole pos/neg/blend lives inside the `predict`
        // closure — a multi-eval solver re-runs the whole closure.
        let native = pipeline::qwen_sigmas(steps, req.width, req.height);
        let mu = pipeline::qwen_mu(req.width, req.height);
        let sigmas =
            candle_gen::resolve_flow_schedule(req.scheduler.as_deref(), mu, steps, &native);

        // Resolve the decode seam once for the whole batch (epic 7840 / sc-7853): a per-generation PiD
        // decoder bound to this prompt when `req.use_pid` is set (errors if requested but not loaded),
        // else `None` → the native QwenVae decode. Built before the loop so all `count` images share it
        // (same prompt), mirroring the MLX `decode_and_collect` seam.
        let pid_decoder = candle_gen_pid::resolve_pid_decoder(pid, req, base_seed, MODEL_ID)?;

        candle_gen::for_each_image_seed(base_seed, req.count, |seed| {
            let latents = pipeline::create_noise(seed, req.width, req.height, &self.device)?
                .to_dtype(DIT_DTYPE)?;

            let latents = candle_gen::run_flow_sampler(
                req.sampler.as_deref(),
                gen_core::sampling::TimestepConvention::Sigma,
                &sigmas,
                latents,
                seed,
                &req.cancel,
                on_progress,
                |latents, sigma| -> CResult<Tensor> {
                    let pos = transformer.forward(latents, pos_embeds, sigma, lat_h, lat_w)?;
                    match neg_embeds {
                        Some(neg) => {
                            let neg = transformer.forward(latents, neg, sigma, lat_h, lat_w)?;
                            Ok(pipeline::compute_guided_noise(&pos, &neg, guidance)?)
                        }
                        None => Ok(pos),
                    }
                },
            )?;

            on_progress(Progress::Decoding);
            let lat = pipeline::unpack_latents(&latents, req.width, req.height)?;
            // PiD (super-resolving) decode when the toggle resolved one; else the native VAE. Both
            // consume the same normalized `[1,16,H/8,W/8]` latent (QwenVae de-normalizes internally,
            // and PiD is trained on that normalized latent) — a zero-transform seam. PiD returns a
            // larger `[1,3,4H,4W]` tensor; `to_image` reads the size from it.
            let decoded = match &pid_decoder {
                Some(pid) => pid.decode(&lat)?,
                None => vae.decode(&lat)?,
            };
            to_image(&decoded)
        })
    }

    /// Load ONLY the Qwen2.5-VL text encoder + its tokenizer for the sequential-residency path (epic
    /// 10765 Phase 1c, sc-10867) — dropped right after the encode so the ~8 GB encoder frees before the
    /// DiT loads. Same loads as [`load_components`](Self::load_components), minus the DiT / VAE / PiD.
    fn load_te_seq(&self) -> CResult<(QwenTextEncoder, TextTokenizer)> {
        let te = QwenTextEncoder::new(&self.te_cfg, self.component_vb("text_encoder", ENC_DTYPE)?)?;
        let tokenizer = TextTokenizer::from_file(
            self.root.join("tokenizer/tokenizer.json"),
            TokenizerConfig {
                max_length: self.te_cfg.max_length,
                pad_token_id: self.te_cfg.pad_token_id,
                chat_template: ChatTemplate::QwenImage,
                pad_to_max_length: false,
            },
        )
        .map_err(|e| CandleError::Msg(format!("qwen-image: load tokenizer: {e}")))?;
        Ok((te, tokenizer))
    }

    /// Load ONLY the DiT for the sequential path (sc-10867) — loaded after the text encoder was dropped,
    /// so it reuses the encoder's freed allocator pool (capping peak at DiT+VAE, not TE+DiT+VAE). Same
    /// packed-detect load (`transformer_group_size` → `QwenTransformer::new_gs`) as
    /// [`load_components`](Self::load_components).
    fn load_transformer_seq(&self) -> CResult<QwenTransformer> {
        let gs = transformer_group_size(&self.root.join("transformer"));
        Ok(QwenTransformer::new_gs(
            &self.dit_cfg,
            self.component_vb("transformer", DIT_DTYPE)?,
            gs,
        )?)
    }

    /// Load ONLY the VAE for the sequential path (sc-10867). Small relative to the DiT, so it stays
    /// co-resident with the DiT through decode (splitting them further buys ~nothing).
    fn load_vae_seq(&self) -> CResult<QwenVae> {
        Ok(QwenVae::new(self.component_vb("vae", ENC_DTYPE)?)?)
    }

    /// Load the optional PiD super-resolving decoder for the sequential path (sc-10867) when the caller
    /// opted in via `LoadSpec::pid`; else `None` (the native QwenVae decode). Same load as
    /// [`load_components`](Self::load_components).
    fn load_pid_seq(&self) -> CResult<Option<PidEngine>> {
        match &self.pid_spec {
            Some(spec) => Ok(Some(PidEngine::from_spec(
                spec,
                PID_BACKBONE,
                &self.device,
            )?)),
            None => Ok(None),
        }
    }

    /// Sequential-residency render (epic 10765 Phase 1c, sc-10867): load the Qwen2.5-VL text encoder →
    /// encode → DROP it → load the DiT + VAE (+ optional PiD) → denoise/decode. Peak VRAM is bounded to
    /// the DiT+VAE working set instead of TE+DiT+VAE, reclaiming the ~8 GB Qwen2.5-VL encoder, so a card
    /// that OOMs the resident path can still render. Output is **bit-identical** to
    /// [`render`](Self::render) — the SAME [`encode`](Self::encode) and the SAME shared
    /// [`denoise_and_decode`](Self::denoise_and_decode) tail run; only the load/free schedule differs.
    ///
    /// Selected by the generator from `LoadSpec::offload_policy` (the worker fit-gate sets `Sequential`)
    /// or the `CANDLE_GEN_OFFLOAD=sequential` env override (the GPU A/B harness). Because it drops the
    /// components, it does NOT populate the generator's `Components` cache — repeat requests reload from
    /// the (page-cached) snapshot; that reload cost is the deliberate trade for the lower peak, which is
    /// why it is opt-in per the fit-gate rather than the default.
    fn render_sequential(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> CResult<Vec<Image>> {
        let guidance = req.guidance.unwrap_or(DEFAULT_GUIDANCE);

        // Phase 1 — text encode, then DROP the encoder (scoped) so the ~8 GB Qwen2.5-VL frees before the
        // DiT loads. Delegates to the SAME `encode` the resident path uses, so tokenization + the
        // conditioning tensors are byte-identical.
        let (pos_embeds, neg_embeds) = {
            let (te, tok) = self.load_te_seq()?;
            let pos = self.encode(&te, &tok, &req.prompt)?;
            // True CFG: build the negative branch unless guidance is a no-op (≤ 1.0).
            let neg = if guidance > 1.0 {
                let neg = req.negative_prompt.as_deref().unwrap_or(NEGATIVE_FALLBACK);
                Some(self.encode(&te, &tok, neg)?)
            } else {
                None
            };
            (pos, neg)
        };

        // Phase 2 — load the DiT (reusing the encoder's freed pool) + the VAE + the optional PiD decoder.
        let transformer = self.load_transformer_seq()?;
        let vae = self.load_vae_seq()?;
        let pid = self.load_pid_seq()?;

        // Phase 3 — the shared denoise + decode tail, identical to `render`'s.
        self.denoise_and_decode(
            req,
            &transformer,
            &vae,
            pid.as_ref(),
            &pos_embeds,
            neg_embeds.as_ref(),
            guidance,
            on_progress,
        )
    }
}

/// The MLX packed `group_size` for the DiT, read from `transformer/config.json`'s `quantization`
/// block (a packed tier — `SceneWorks/qwen-image-mlx` + `qwen-image-edit-2511-mlx` ship group 64).
/// Absent config / no `quantization` block (a dense diffusers snapshot) ⇒ the shared default 64
/// ([`candle_gen::quant::PackedConfig`] already resolves a missing/absent `group_size` to 64, so a
/// packed tier with only `bits` still loads packed — never a silent dense read of u32 codes). The
/// group size is inert on the dense path, so a dense snapshot is byte-identical regardless.
pub(crate) fn transformer_group_size(dit_dir: &Path) -> usize {
    std::fs::read_to_string(dit_dir.join("config.json"))
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        .and_then(|v| candle_gen::quant::PackedConfig::from_config(&v))
        .map(|pc| pc.group_size as usize)
        .unwrap_or(candle_gen::quant::MLX_GROUP_SIZE)
}

/// Whether the DiT tier is MLX-**packed** (q4/q8), read from `transformer/config.json`'s `quantization`
/// block — the MLX convert job stamps it on every quantized tier (`SceneWorks/qwen-image-*-mlx`), and a
/// dense diffusers snapshot has none. Gates the edit lane's adapter route (sc-11091): a packed base
/// attaches LoRA/LoKr as forward-time additive residuals (base kept packed, no dense reload), while a
/// dense base folds `W += δ` bit-exactly. A misread is safe by construction — the additive path is
/// correct on a dense base too (it just isn't bit-identical to the fold there), and per-projection
/// packed-detect (`.scales`) still governs the actual weight load.
pub(crate) fn transformer_is_packed(dit_dir: &Path) -> bool {
    std::fs::read_to_string(dit_dir.join("config.json"))
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        .and_then(|v| candle_gen::quant::PackedConfig::from_config(&v))
        .is_some()
}

fn to_image(decoded: &Tensor) -> CResult<Image> {
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

/// Whether the sequential-residency offload path is force-enabled by env (epic 10765 Phase 1c,
/// sc-10867). Reads `CANDLE_GEN_OFFLOAD`: `sequential` (case-insensitive) selects the phased
/// load→encode→drop path regardless of `LoadSpec::offload_policy`; unset or any other value defers to
/// the spec (the worker fit-gate sets the policy in production). Kept as the override the two-process
/// GPU A/B harness drives, mirroring the candle-gen-flux toggle (sc-10769/sc-10821).
fn sequential_offload_enabled() -> bool {
    std::env::var("CANDLE_GEN_OFFLOAD")
        .map(|value| value.trim().eq_ignore_ascii_case("sequential"))
        .unwrap_or(false)
}

pub struct QwenImageGenerator {
    descriptor: ModelDescriptor,
    root: PathBuf,
    device: Device,
    /// The `LoadSpec::pid` component captured at load (epic 7840 / sc-7853), threaded into the lazy
    /// component build so the PiD engine loads once alongside the base model. `None` when not opted in.
    pid_spec: Option<PidWeights>,
    /// Component-residency policy captured from `LoadSpec::offload_policy` (epic 10765 Phase 1c,
    /// sc-10867). `Sequential` routes `generate` through `render_sequential` (load→encode→drop the
    /// Qwen2.5-VL text encoder before the DiT), capping peak VRAM at the cost of the components cache;
    /// `Resident` (default) keeps the cached path. The worker's fit-gate sets this when it predicts the
    /// resident TE+DiT+VAE sum won't fit but the DiT+VAE working set will. Inert on the in-place ComfyUI
    /// DiT lane (that transformer is a single file, not `root/transformer/` — sequential defers there).
    offload_policy: OffloadPolicy,
    /// An in-place ComfyUI Qwen-Image DiT single-file (epic 10451 Phase 2b, sc-10670), set only by
    /// [`load_from_comfyui_dit`]. When present, the lazy component build sources the transformer from
    /// this file (prefix-strip + fp8→bf16) and the TE/tokenizer from [`Self::root`]; `None` on the
    /// registry path.
    comfyui_dit: Option<PathBuf>,
    /// An in-place ComfyUI Qwen-Image VAE single-file (epic 10451 Phase 2b, sc-10830), set only by
    /// [`load_from_comfyui_dit`]. When present the VAE is read from this file (native WAN-VAE keys
    /// remapped); `None` ⇒ the snapshot's `vae/`.
    comfyui_vae: Option<PathBuf>,
    components: Mutex<Option<Components>>,
}

impl QwenImageGenerator {
    fn components(&self, pipe: &Pipeline) -> gen_core::Result<Components> {
        // `?` bridges the candle-side `load_components` error into `gen_core::Error`.
        Ok(candle_gen::cached(&self.components, || {
            pipe.load_components()
        })?)
    }
}

impl Generator for QwenImageGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        self.descriptor
            .capabilities
            .validate_request(MODEL_ID, req)?;
        if req.prompt.is_empty() {
            return Err(gen_core::Error::Msg(
                "qwen_image: prompt must not be empty".into(),
            ));
        }
        if req.steps == Some(0) {
            return Err(gen_core::Error::Msg(
                "qwen_image: steps must be >= 1".into(),
            ));
        }
        if !req.width.is_multiple_of(SIZE_MULTIPLE) || !req.height.is_multiple_of(SIZE_MULTIPLE) {
            return Err(gen_core::Error::Msg(format!(
                "qwen_image: width/height must be multiples of {SIZE_MULTIPLE} (got {}x{})",
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
        let pipe = match &self.comfyui_dit {
            Some(dit_file) => Pipeline::load_comfyui(
                &self.root,
                &self.device,
                dit_file.clone(),
                self.comfyui_vae.clone(),
            ),
            None => Pipeline::load(&self.root, &self.device, self.pid_spec.clone()),
        };
        // Sequential-residency offload (epic 10765 Phase 1c, sc-10867): when selected, load→encode→drop
        // the ~8 GB Qwen2.5-VL text encoder before the DiT so peak VRAM is DiT+VAE, not TE+DiT+VAE —
        // letting a card that OOMs the resident path render. Output is bit-identical; it bypasses the
        // components cache (it drops what it loads). Driven by `LoadSpec::offload_policy` (the worker
        // fit-gate sets `Sequential`); `CANDLE_GEN_OFFLOAD=sequential` is an env override kept for the
        // GPU A/B harness. Never taken on the in-place ComfyUI DiT lane — that transformer is a single
        // file, not `root/transformer/`, so the per-phase loaders can't source it (advisory: falls back
        // to resident). The default stays the resident, cross-request-cached path.
        let sequential = self.comfyui_dit.is_none()
            && (self.offload_policy == OffloadPolicy::Sequential || sequential_offload_enabled());
        let images = if sequential {
            pipe.render_sequential(req, on_progress)?
        } else {
            let components = self.components(&pipe)?;
            pipe.render(req, &components, on_progress)?
        };
        Ok(GenerationOutput::Images(images))
    }
}

/// Resolve the sampling step count for a base txt2img render: honor a caller-supplied `steps`,
/// otherwise fall back to [`DEFAULT_STEPS`]. The base Qwen-Image is a non-distilled 20B flow-match
/// model, so the default is a production count (sc-9046 / F-076) — the few-step distilled/Lightning
/// count lives only on the gated Edit path.
fn resolve_steps(requested: Option<u32>) -> usize {
    requested
        .map(|s| s as usize)
        .unwrap_or(DEFAULT_STEPS as usize)
}

/// Qwen-Image txt2img descriptor — the surface sc-3696 wires: true-CFG txt2img with a negative
/// prompt; no conditioning (img2img/Edit deferred), no LoRA/quant, no Lightning sampler.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID,
        family: "qwen-image",
        backend: "candle",
        modality: Modality::Image,
        capabilities: Capabilities {
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: true,
            conditioning: vec![],
            supports_lora: false,
            supports_lokr: false,
            samplers: candle_gen::curated_sampler_names(),
            schedulers: candle_gen::menu_with_aliases(
                candle_gen::curated_scheduler_names(),
                &["flow_match_euler"],
            ),
            supported_guidance_methods: vec![],
            min_size: 256,
            max_size: 2048,
            max_count: 8,
            mac_only: false,
            supported_quants: &[] as &[Quant],
            supports_kv_cache: false,
            requires_sigma_shift: true,
        },
    }
}

/// Construct a lazy candle Qwen-Image generator. `spec.weights` must be a [`WeightsSource::Dir`]
/// pointing at a `Qwen/Qwen-Image` diffusers snapshot (`text_encoder/`, `transformer/`, `vae/`,
/// `tokenizer/`). Adapters / quantization / control overlays are rejected (not wired).
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(
                "qwen_image expects a snapshot directory (text_encoder/ transformer/ vae/ \
                 tokenizer/), not a single .safetensors file"
                    .into(),
            ));
        }
    };
    if !spec.adapters.is_empty() {
        return Err(gen_core::Error::Unsupported(
            "candle qwen_image does not support LoRA/LoKr yet".into(),
        ));
    }
    if spec.quantize.is_some() {
        return Err(gen_core::Error::Unsupported(
            "candle qwen_image does not support on-the-fly Q4/Q8 quantization yet".into(),
        ));
    }
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(gen_core::Error::Unsupported(
            "candle qwen_image does not support control / Edit yet (txt2img only)".into(),
        ));
    }
    let device = candle_gen::default_device()?;
    Ok(Box::new(QwenImageGenerator {
        descriptor: descriptor(),
        root,
        device,
        // PiD is an optional aux decoder (epic 7840 / sc-7853): capture the load-spec component (if
        // any) so the lazy component build loads the engine once. Unlike adapters/quant/control above,
        // it is not rejected — `None` simply keeps the byte-exact native-VAE path.
        pid_spec: spec.pid.clone(),
        // Component-residency policy (epic 10765 Phase 1c, sc-10867) — `Sequential` routes generate
        // through `render_sequential`. Captured at load; defaults to `Resident`.
        offload_policy: spec.offload_policy,
        comfyui_dit: None,
        comfyui_vae: None,
        components: Mutex::new(None),
    }))
}

/// Construct a lazy candle Qwen-Image generator that reads its **DiT** (and optionally its **VAE**) in
/// place from existing ComfyUI single-files (epic 10451 Phase 2b, sc-10670 / sc-10830) — no copy, no
/// re-download. `transformer_file` is the user's ComfyUI Qwen-Image DiT
/// (`diffusion_models/qwen_image_*_fp8_e4m3fn.safetensors`, keyed `model.diffusion_model.*`, plain
/// fp8); its keys are prefix-stripped and its weights upcast to bf16 in memory ([`comfyui`]) at
/// component build.
///
/// `vae_file`, when `Some`, is the tree's `vae/qwen_image_vae.safetensors` (native WAN-VAE keys,
/// remapped to the diffusers schema by [`comfyui::remap_vae_wan_to_diffusers`]); when `None` the VAE
/// comes from `snapshot_dir`'s `vae/`. `snapshot_dir` is a resident Qwen-Image diffusers snapshot
/// supplying the Qwen2.5-VL text encoder (still snapshot-sourced — it is scaled-fp8, sc-10671) and the
/// tokenizer (and the VAE when `vae_file` is `None`). txt2img only; no adapters / control / PiD.
pub fn load_from_comfyui_dit(
    transformer_file: impl Into<PathBuf>,
    snapshot_dir: impl Into<PathBuf>,
    vae_file: Option<PathBuf>,
) -> gen_core::Result<Box<dyn Generator>> {
    let device = candle_gen::default_device()?;
    Ok(Box::new(QwenImageGenerator {
        descriptor: descriptor(),
        root: snapshot_dir.into(),
        device,
        pid_spec: None,
        // The in-place ComfyUI DiT lane keeps everything resident (its transformer is a single file,
        // not `root/transformer/`, so the sequential per-phase loaders don't apply).
        offload_policy: OffloadPolicy::Resident,
        comfyui_dit: Some(transformer_file.into()),
        comfyui_vae: vae_file,
        components: Mutex::new(None),
    }))
}

candle_gen::register_generators! { descriptor => load }

/// Force-link hook (keeps the `inventory::submit!` registration from being dead-stripped).
pub fn force_link() {}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::registry;
    use candle_gen::gen_core::ConditioningKind;

    #[test]
    fn registers_and_resolves_as_candle() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry::load(MODEL_ID, &spec).expect("qwen_image is registered");
        assert_eq!(g.descriptor().id, MODEL_ID);
        assert_eq!(g.descriptor().family, "qwen-image");
        assert_eq!(g.descriptor().backend, "candle");
    }

    /// Sequential-residency GPU validation (epic 10765 Phase 1c, sc-10867). ONE probed generation whose
    /// mode is the `CANDLE_GEN_OFFLOAD` env the generator reads; prints the device peak VRAM and writes
    /// the raw RGB pixels to `QWEN_OUT`. Run it TWICE in SEPARATE processes (resident vs sequential) and
    /// compare: the pixel files must be byte-identical (parity) and the sequential peak materially lower
    /// (the ~8 GB Qwen2.5-VL encoder dropped before the DiT loads). Two processes are REQUIRED — candle's
    /// cudarc caching allocator never returns pages to the driver, so a second in-process run would reuse
    /// the first run's pool and read the same peak. Ignored by default; needs a real-file (hardlink-
    /// staged, not raw-HF-symlink) Qwen-Image snapshot in `QWEN_IMAGE_SNAPSHOT` + a CUDA device.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "needs QWEN_IMAGE_SNAPSHOT (a real-file Qwen-Image snapshot dir w/ tokenizer.json) + a CUDA GPU"]
    fn qwen_image_probed_generate_for_offload_ab() {
        let dir = std::env::var("QWEN_IMAGE_SNAPSHOT")
            .expect("set QWEN_IMAGE_SNAPSHOT to a real-file (hardlink-staged) Qwen-Image snapshot");
        let out = std::env::var("QWEN_OUT").expect("set QWEN_OUT to the pixel-dump path");
        // Two ways to select sequential residency, both exercised by the A/B runner:
        //   - env `CANDLE_GEN_OFFLOAD=sequential` (the override, sc-10769/sc-10867), OR
        //   - `QWEN_OFFLOAD_MODE=spec-sequential` → drive it through `LoadSpec::offload_policy`
        //     (the worker-facing contract, sc-10821/sc-10867), with CANDLE_GEN_OFFLOAD UNSET.
        let mut spec = LoadSpec::new(WeightsSource::Dir(dir.into()));
        let spec_mode = std::env::var("QWEN_OFFLOAD_MODE").unwrap_or_default();
        if spec_mode == "spec-sequential" {
            spec = spec.with_offload_policy(OffloadPolicy::Sequential);
        }
        let req = GenerationRequest {
            prompt: "a rusty robot holding a lit candle, studio lighting".into(),
            width: 1024,
            height: 1024,
            steps: Some(8),
            seed: Some(42),
            count: 1,
            ..Default::default()
        };
        let sampler = candle_gen::testkit::PeakSampler::start(0);
        let g = load(&spec).expect("load qwen_image");
        let output = g.generate(&req, &mut |_| {}).expect("generate");
        let peak_mib = sampler.stop();
        let img = match output {
            GenerationOutput::Images(mut v) => v.remove(0),
            other => panic!("expected images, got {other:?}"),
        };
        std::fs::write(&out, &img.pixels).expect("write pixels");
        let env_mode = std::env::var("CANDLE_GEN_OFFLOAD").unwrap_or_default();
        let mode = if spec_mode == "spec-sequential" {
            "spec-sequential"
        } else if env_mode.eq_ignore_ascii_case("sequential") {
            "env-sequential"
        } else {
            "resident"
        };
        eprintln!(
            "SEQ_AB mode={mode} peak_mib={peak_mib} bytes={} {}x{} out={out}",
            img.pixels.len(),
            img.width,
            img.height
        );
    }

    /// `transformer_group_size` reads the packed `transformer/config.json`'s `quantization.group_size`
    /// (the `SceneWorks/qwen-image-mlx` tiers ship 64), defaults a `bits`-only block to the shared 64
    /// (never a silent dense read of u32 codes), and defaults a dense/absent config to 64. This is the
    /// group size threaded into `QwenTransformer::new_gs` for the packed-detect load (sc-9415).
    #[test]
    fn transformer_group_size_reads_quantization_block() {
        let tmp = std::env::temp_dir().join(format!("sc9415_gscfg_{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();

        // The real qwen-image-mlx tier: bits 4, group 64.
        std::fs::write(
            tmp.join("config.json"),
            r#"{ "num_layers": 60, "quantization": { "bits": 4, "group_size": 64 } }"#,
        )
        .unwrap();
        assert_eq!(transformer_group_size(&tmp), 64);

        // A non-64 packed tier is honored end to end.
        std::fs::write(
            tmp.join("config.json"),
            r#"{ "quantization": { "bits": 4, "group_size": 32 } }"#,
        )
        .unwrap();
        assert_eq!(transformer_group_size(&tmp), 32);

        // `bits`-only (no group_size) ⇒ the shared default 64 (PackedConfig resolves it), NOT a dense
        // read of packed nibbles.
        std::fs::write(
            tmp.join("config.json"),
            r#"{ "quantization": { "bits": 8 } }"#,
        )
        .unwrap();
        assert_eq!(transformer_group_size(&tmp), 64);

        // A dense snapshot (no `quantization` block) ⇒ default 64 (inert on the dense path).
        std::fs::write(tmp.join("config.json"), r#"{ "num_layers": 60 }"#).unwrap();
        assert_eq!(transformer_group_size(&tmp), 64);

        // A genuinely-absent config ⇒ default 64.
        assert_eq!(
            transformer_group_size(&tmp.join("missing")),
            candle_gen::quant::MLX_GROUP_SIZE
        );

        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn descriptor_surface() {
        let d = descriptor();
        assert!(d.capabilities.supports_guidance);
        assert!(d.capabilities.supports_true_cfg);
        assert!(d.capabilities.supports_negative_prompt);
        assert!(d.capabilities.requires_sigma_shift);
        assert!(!d.capabilities.mac_only);
        assert!(d.capabilities.conditioning.is_empty());
        assert!(!d.capabilities.accepts(ConditioningKind::Reference));
        assert!(!d.capabilities.supports_lora);
        // Unified curated sampler/scheduler menu (epic 7114 P4, sc-7123): the full curated sampler menu,
        // and the curated scheduler menu plus the legacy `flow_match_euler` alias (N3 fallback).
        assert_eq!(
            d.capabilities.samplers,
            candle_gen::curated_sampler_names(),
            "samplers expose the curated menu"
        );
        assert!(
            d.capabilities.schedulers.contains(&"flow_match_euler"),
            "schedulers keep the legacy alias"
        );
        for s in candle_gen::curated_scheduler_names() {
            assert!(
                d.capabilities.schedulers.contains(&s),
                "scheduler menu missing {s}"
            );
        }
    }

    #[test]
    fn validate_accepts_txt2img_and_rejects_unsupported() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry::load(MODEL_ID, &spec).unwrap();
        let ok = GenerationRequest {
            prompt: "a rusty robot holding a lit candle".into(),
            guidance: Some(4.0),
            negative_prompt: Some("blurry".into()),
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

    /// sc-9046 (F-076): the base txt2img path is a non-distilled 20B model, so an omitted `steps`
    /// must resolve to the production [`DEFAULT_STEPS`] (not the old 4-step distilled count), while an
    /// explicit caller-supplied count is always honored verbatim.
    #[test]
    fn resolve_steps_defaults_to_production_and_honors_explicit() {
        // Omitted → production default (currently 30), never the few-step distilled count.
        assert_eq!(resolve_steps(None), DEFAULT_STEPS as usize);
        assert!(
            resolve_steps(None) >= 20,
            "base default must be a production step count, not a distilled few-step count"
        );
        // Explicit values pass through untouched — including a legitimate low count if a caller
        // deliberately asks for it (e.g. a merged Lightning adapter, wired elsewhere).
        assert_eq!(resolve_steps(Some(4)), 4);
        assert_eq!(resolve_steps(Some(50)), 50);
        assert_eq!(resolve_steps(Some(1)), 1);
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
        let spec = LoadSpec::new(WeightsSource::File("/tmp/q.safetensors".into()));
        let err = load(&spec).err().expect("expected an error").to_string();
        assert!(err.contains("snapshot directory"), "got: {err}");
    }

    /// sc-8647: a `Qwen/Qwen-Image-2512` snapshot is a structural drop-in for the original
    /// `Qwen/Qwen-Image` — same diffusers layout (`text_encoder/ transformer/ vae/ tokenizer/`),
    /// same 60-layer MMDiT, same Qwen2.5-VL text encoder, same Qwen2 BPE tokenizer (the worker's
    /// `DERIVED_TOKENIZER_OVERLAYS` materializes `tokenizer/tokenizer.json` for 2512 too). The
    /// candle loader keys nothing on the repo string — it loads the dir structurally — so a
    /// 2512-shaped snapshot is accepted exactly like the base. Pin that: a synthetic 2512 snapshot
    /// dir loads, and the per-release config used is byte-identical to the base config.
    #[test]
    fn loads_qwen_image_2512_shaped_snapshot() {
        // The 2512 base reuses the base config verbatim (sc-8271 parity); the candle loader uses
        // these for the DiT + text encoder regardless of which snapshot dir is supplied.
        assert_eq!(
            TransformerConfig::qwen_image_2512(),
            TransformerConfig::qwen_image(),
            "2512 MMDiT config must be a verbatim drop-in (same 60-layer dual-stream MMDiT)"
        );
        assert_eq!(
            TextEncoderConfig::qwen_image_2512(),
            TextEncoderConfig::qwen_image(),
            "2512 text-encoder config must be a verbatim drop-in (same Qwen2.5-VL + BPE tokenizer)"
        );

        // A 2512 snapshot ships the identical diffusers directory layout; the worker overlays a
        // built `tokenizer/tokenizer.json`. Build that shape and confirm the loader accepts it (no
        // repo-string gate rejects 2512) and that `Pipeline::load` resolves the tokenizer path that
        // `encode` reads.
        let tmp = std::env::temp_dir().join(format!("qwen2512_snap_{}", std::process::id()));
        for sub in ["text_encoder", "transformer", "vae", "tokenizer"] {
            std::fs::create_dir_all(tmp.join(sub)).unwrap();
        }
        std::fs::write(tmp.join("tokenizer/tokenizer.json"), b"{}").unwrap();

        let spec = LoadSpec::new(WeightsSource::Dir(tmp.clone()));
        let g = load(&spec).expect("a 2512-shaped snapshot dir must load like the base");
        assert_eq!(g.descriptor().id, MODEL_ID);

        let pipe = Pipeline::load(&tmp, &Device::Cpu, None);
        assert!(
            pipe.root.join("tokenizer/tokenizer.json").is_file(),
            "loader must resolve the overlaid tokenizer.json under tokenizer/"
        );

        std::fs::remove_dir_all(&tmp).ok();
    }
}
