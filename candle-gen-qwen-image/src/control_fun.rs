//! The Qwen-Image **2512-Fun-Controlnet-Union** (VACE) control provider (sc-8350) — the candle
//! (Windows/CUDA) sibling of `mlx-gen-qwen-image`'s `QwenImageControl` (mlx sc-8267 / PR #604).
//! Structural control (pose / canny / depth) on the **Qwen-Image-2512** base via the alibaba-pai
//! `Qwen-Image-2512-Fun-Controlnet-Union` checkpoint (Apache-2.0, ungated).
//!
//! Unlike the InstantX [`crate::control::QwenControl`] (an independent mini-transformer
//! emitting residuals the base ADDs at a fixed interval), this is **VACE-style**: a `control_img_in`
//! patch embedder feeds a control state threaded through 5 control blocks that reuse the base block
//! math (seeded at block 0 by `before_proj(c) + img_embed`); each emits a zero-init `after_proj` hint
//! the base 60-layer MMDiT adds into its image stream at `control_layers = [0, 12, 24, 36, 48]` scaled
//! by the request's control scale — [`QwenTransformer::forward_fun_control`].
//!
//! **Input-agnostic** (sc-8250): pose, canny, and depth differ only by the preprocessor-produced
//! control image fed to [`QwenFunControl::generate`] — there is no mode index and no per-kind branch.
//! v1 is pose/canny/depth-from-prompt (no img2img-with-control compose yet).
//!
//! Like the InstantX lane, this is a plain struct driven **directly** by the worker (a bespoke stream,
//! like `candle_gen_sdxl::IpAdapterSdxl`), not a gen-core-registered generator — the registered
//! `qwen_image` descriptor stays txt2img-only. The InstantX lane ([`crate::control`]) is kept intact;
//! the worker retirement of InstantX is **Phase B** (sc-8246, a different repo).

use std::path::{Path, PathBuf};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::runtime::CancelFlag;
use candle_gen::gen_core::{Image, Progress};
use candle_gen::{CandleError, Result};

use crate::config::{TextEncoderConfig, TransformerConfig, NEGATIVE_FALLBACK};
use crate::control_common;
use crate::pipeline;
use crate::text_encoder::QwenTextEncoder;
use crate::transformer::{QwenFunControlBranch, QwenTransformer};
use crate::vae::{QwenVae, QwenVaeEncoder};

/// The transformer + control branch run bf16 (native dtype); the encoder + VAE run f32.
const DIT_DTYPE: DType = DType::BF16;
const ENC_DTYPE: DType = DType::F32;
/// Error-message prefix for this lane (shared [`control_common`] helpers thread it through).
const LABEL: &str = "qwen fun-control";

/// The 2512-Fun Union injects 5 VACE hints into the base 60-layer MMDiT at these base block indices
/// (the alibaba-pai `config/qwenimage_control.yaml` `control_layers`, interval 12). `0` must be present
/// — `before_proj` lives on control block 0.
pub const CONTROL_LAYERS: [usize; 5] = [0, 12, 24, 36, 48];
/// Packed control-context channels (`control_img_in` in-features): `[control_latent(16) | mask(1) |
/// inpaint(16)]` × the 2×2 patch = `33 · 4 = 132`.
pub const CONTROL_IN_DIM: usize =
    (crate::config::LATENT_CHANNELS * 2 + 1) * crate::config::PATCH * crate::config::PATCH;
/// Default conditioning scale on the VACE hints.
pub const DEFAULT_CONTROL_SCALE: f32 = 1.0;

/// Paths to the Qwen-Image 2512-Fun control checkpoints.
pub struct QwenFunControlPaths {
    /// The `Qwen/Qwen-Image-2512` diffusers snapshot dir (`text_encoder/`, `transformer/`, `vae/`,
    /// `tokenizer/`).
    pub qwen_base: PathBuf,
    /// The alibaba-pai `Qwen-Image-2512-Fun-Controlnet-Union` checkpoint — a single `.safetensors`
    /// file or a dir of shards.
    pub controlnet: PathBuf,
}

/// One Qwen-Image 2512-Fun (pose/canny/depth) generation request. The control **kind** is implicit in
/// the control image passed to [`QwenFunControl::generate`] (input-agnostic — no mode field).
#[derive(Clone)]
pub struct QwenFunControlRequest {
    pub prompt: String,
    pub negative: String,
    pub width: u32,
    pub height: u32,
    pub steps: usize,
    /// True-CFG guidance scale.
    pub guidance: f32,
    /// Conditioning scale on the VACE hints (`0` ≡ base txt2img).
    pub control_scale: f32,
    pub seed: u64,
    pub cancel: CancelFlag,
}

impl Default for QwenFunControlRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            negative: String::new(),
            width: 1024,
            height: 1024,
            steps: 30,
            guidance: 4.0,
            control_scale: DEFAULT_CONTROL_SCALE,
            seed: 0,
            cancel: CancelFlag::default(),
        }
    }
}

/// Resolve the 2512-Fun control weight file(s) from a dir-or-file path → the list of `.safetensors`
/// shards to mmap (the checkpoint is a single `Qwen-Image-2512-Fun-Controlnet-Union-….safetensors`,
/// or a dir of shards).
fn resolve_controlnet_files(path: &Path) -> Result<Vec<PathBuf>> {
    // Shared file-or-dir resolver (sc-8999 / F-019): single `.safetensors` → itself, a dir → its
    // sorted shards, a missing path → the crafted `{label}: no .safetensors ...` error.
    candle_gen::resolve_weight_files(path, "qwen fun-control")
}

/// The loaded Qwen-Image 2512-Fun control model: the reused base text encoder / DiT / VAE-decoder, plus
/// the VAE encoder (to encode the control hint) and the VACE control branch.
pub struct QwenFunControl {
    device: Device,
    te: QwenTextEncoder,
    /// Qwen tokenizer, loaded+parsed **once** at load and reused across encodes (sc-8991 / F-011)
    /// instead of re-parsing `tokenizer.json` per prompt/branch.
    tokenizer: candle_gen::gen_core::tokenizer::TextTokenizer,
    transformer: QwenTransformer,
    controlnet: QwenFunControlBranch,
    vae: QwenVae,
    vae_encoder: QwenVaeEncoder,
}

impl QwenFunControl {
    /// Load the base Qwen-Image-2512 components + the VAE encoder + the 2512-Fun VACE control branch.
    pub fn load(paths: &QwenFunControlPaths) -> Result<Self> {
        let device = candle_gen::default_device()?;
        let root = paths.qwen_base.clone();
        // The 2512 base reuses the base config verbatim (sc-8647 / sc-8271 parity).
        let te_cfg = TextEncoderConfig::qwen_image_2512();
        let dit_cfg = TransformerConfig::qwen_image_2512();

        let te = QwenTextEncoder::new(
            &te_cfg,
            control_common::component_vb(&root, "text_encoder", ENC_DTYPE, &device, LABEL)?,
        )?;
        // The base 2512 MMDiT packed-detects (a packed MLX base tier loads straight from the packed
        // parts; a dense base snapshot unchanged) at the `group_size` read from `transformer/config.json`.
        let gs = crate::transformer_group_size(&root.join("transformer"));
        let transformer = QwenTransformer::new_gs(
            &dit_cfg,
            control_common::component_vb(&root, "transformer", DIT_DTYPE, &device, LABEL)?,
            gs,
        )?;
        let vae = QwenVae::new(control_common::component_vb(
            &root, "vae", ENC_DTYPE, &device, LABEL,
        )?)?;
        let vae_encoder = QwenVaeEncoder::new(control_common::component_vb(
            &root, "vae", ENC_DTYPE, &device, LABEL,
        )?)?;

        let cn_files = resolve_controlnet_files(&paths.controlnet)?;
        let cn_vb = candle_gen::mmap_var_builder(&cn_files, DIT_DTYPE, &device)?;
        let controlnet =
            QwenFunControlBranch::new(&dit_cfg, &CONTROL_LAYERS, CONTROL_IN_DIM, cn_vb)?;

        let tokenizer = control_common::load_tokenizer(&root, &te_cfg, LABEL)?;
        Ok(Self {
            device,
            te,
            tokenizer,
            transformer,
            controlnet,
            vae,
            vae_encoder,
        })
    }

    /// Tokenize + encode `prompt` → `prompt_embeds` `[1, seq, 3584]` at the DiT dtype (bf16). Mirrors
    /// the txt2img `Pipeline::encode`.
    fn encode(&self, prompt: &str) -> Result<Tensor> {
        control_common::encode(
            &self.tokenizer,
            &self.te,
            &self.device,
            DIT_DTYPE,
            prompt,
            LABEL,
        )
    }

    /// Structural-control generation: condition the base MMDiT on `control` (a preprocessed pose / canny
    /// / depth image at the request size — input-agnostic, no kind argument) via the 2512-Fun VACE
    /// branch. VAE-encodes + packs the control hint to the 132-ch control context once, then runs the
    /// control denoise (the control branch runs on both CFG passes).
    pub fn generate(
        &self,
        req: &QwenFunControlRequest,
        control: &Image,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        if req.cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }
        let (lat_h, lat_w) = pipeline::latent_dims(req.width, req.height);

        let pos = self.encode(&req.prompt)?;
        let neg = if req.guidance > 1.0 {
            let n = if req.negative.trim().is_empty() {
                NEGATIVE_FALLBACK
            } else {
                req.negative.as_str()
            };
            Some(self.encode(n)?)
        } else {
            None
        };

        // VAE-encode the control image → 16-ch latent, then pack the 132-ch control context (control
        // latent + zero mask + zero inpaint, 2×2-packed). Constant across denoise steps + the batch.
        let control_img = control_common::preprocess_control_image(
            control,
            req.width,
            req.height,
            &self.device,
            LABEL,
        )?;
        let control_latent = self.vae_encoder.encode(&control_img)?;
        let control_cond =
            pipeline::pack_fun_control_context(&control_latent, req.width, req.height)?
                .to_dtype(DIT_DTYPE)?;

        // Routed through the unified curated sampler/scheduler framework (epic 7114): the `native`
        // schedule is the production `qwen_sigmas`, `mu` steers the (non-default) curated scheduler. The
        // bespoke control provider has no `req.sampler`/`req.scheduler` surface, so both stay `None` (the
        // N1 default: `euler` over the native schedule). The model is fed the raw sigma (`Sigma`
        // convention); the VACE branch + true-CFG pos/neg/blend all live inside the `predict` closure.
        let native = pipeline::qwen_sigmas(req.steps, req.width, req.height);
        let mu = pipeline::qwen_mu(req.width, req.height);
        let sigmas = candle_gen::resolve_flow_schedule(None, mu, req.steps, &native);
        let latents = pipeline::create_noise(req.seed, req.width, req.height, &self.device)?
            .to_dtype(DIT_DTYPE)?;

        let latents = candle_gen::run_flow_sampler(
            None,
            candle_gen::gen_core::sampling::TimestepConvention::Sigma,
            &sigmas,
            latents,
            req.seed,
            &req.cancel,
            on_progress,
            |latents, sigma| -> Result<Tensor> {
                let pos_v = self.transformer.forward_fun_control(
                    latents,
                    &pos,
                    sigma,
                    lat_h,
                    lat_w,
                    Some((&self.controlnet, &control_cond)),
                    req.control_scale,
                )?;
                match &neg {
                    Some(neg) => {
                        let neg_v = self.transformer.forward_fun_control(
                            latents,
                            neg,
                            sigma,
                            lat_h,
                            lat_w,
                            Some((&self.controlnet, &control_cond)),
                            req.control_scale,
                        )?;
                        Ok(pipeline::compute_guided_noise(
                            &pos_v,
                            &neg_v,
                            req.guidance,
                        )?)
                    }
                    None => Ok(pos_v),
                }
            },
        )?;

        on_progress(Progress::Decoding);
        let lat = pipeline::unpack_latents(&latents, req.width, req.height)?;
        let decoded = self.vae.decode(&lat)?;
        control_common::to_image(&decoded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::Image;

    #[test]
    fn request_defaults() {
        let r = QwenFunControlRequest::default();
        assert_eq!((r.width, r.height), (1024, 1024));
        assert_eq!(r.steps, 30);
        assert_eq!(r.control_scale, DEFAULT_CONTROL_SCALE);
        assert!(!r.cancel.is_cancelled());
    }

    /// The shipped 2512-Fun Union: 5 control layers at `[0, 12, 24, 36, 48]` across the 60-layer base
    /// MMDiT (interval 12, `0` present for `before_proj`), control context 132 = (16·2 + 1)·4.
    #[test]
    fn control_layout_matches_fork() {
        assert_eq!(CONTROL_LAYERS, [0, 12, 24, 36, 48]);
        assert_eq!(CONTROL_LAYERS.len(), 5);
        assert!(CONTROL_LAYERS.contains(&0), "before_proj lives on block 0");
        assert_eq!(CONTROL_IN_DIM, 132);
        let base = TransformerConfig::qwen_image_2512();
        // 5 hints evenly spaced across 60 base blocks at interval 12.
        assert_eq!(base.num_layers, 60);
        for (n, &p) in CONTROL_LAYERS.iter().enumerate() {
            assert_eq!(
                p,
                n * 12,
                "control layer {n} should inject at base block {}",
                n * 12
            );
            assert!(p < base.num_layers, "injection index in range");
        }
    }

    #[test]
    fn controlnet_file_resolution() {
        let dir = std::env::temp_dir().join(format!("qwen_fun_cn_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // Empty dir → error.
        assert!(resolve_controlnet_files(&dir).is_err());
        // A single file path resolves to itself.
        let f = dir.join("Qwen-Image-2512-Fun-Controlnet-Union.safetensors");
        std::fs::write(&f, b"x").unwrap();
        assert_eq!(resolve_controlnet_files(&f).unwrap(), vec![f.clone()]);
        // A dir of shards resolves to the sorted shard list.
        let g = dir.join("model-00002.safetensors");
        std::fs::write(&g, b"y").unwrap();
        let got = resolve_controlnet_files(&dir).unwrap();
        assert_eq!(got, vec![f, g]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// This lane's control-image preprocessing goes through the shared [`control_common`] helper with
    /// this lane's `LABEL`; the numeric behavior is unchanged from the pre-dedup verbatim copy.
    #[test]
    fn control_preprocess_shape_and_range() {
        let img = Image {
            width: 16,
            height: 8,
            pixels: vec![255u8; 16 * 8 * 3],
        };
        let t = control_common::preprocess_control_image(&img, 16, 8, &Device::Cpu, LABEL).unwrap();
        assert_eq!(t.dims(), &[1, 3, 8, 16]);
        // 255 → 255/127.5 - 1 = 1.0
        let v = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(v.iter().all(|x| (x - 1.0).abs() < 1e-4));
        // size mismatch errors loudly, with this lane's label.
        let e = control_common::preprocess_control_image(&img, 32, 8, &Device::Cpu, LABEL)
            .unwrap_err()
            .to_string();
        assert!(e.starts_with("qwen fun-control:"), "got: {e}");
    }

    /// The 132-ch control context packs to `[1, seq, 132]` and reduces to `[control_latent | 0 | 0]`:
    /// the mask (channel 16) and the inpaint latents (channels 17..33) of every packed token are zero
    /// in the pose/canny/depth-only layout, while the control latent (channels 0..16) carries through.
    #[test]
    fn fun_control_context_packs_and_zero_pads() {
        let (w, h) = (32u32, 16u32);
        let (l8h, l8w) = ((h / 8) as usize, (w / 8) as usize); // 2 x 4
                                                               // A non-zero 16-ch control latent.
        let latent = Tensor::ones((1, 16, l8h, l8w), DType::F32, &Device::Cpu).unwrap();
        let ctx = pipeline::pack_fun_control_context(&latent, w, h).unwrap();
        let (lat_h, lat_w) = pipeline::latent_dims(w, h); // h/16, w/16 = 1 x 2
        assert_eq!(ctx.dims(), &[1, lat_h * lat_w, 132]);
        // Reshape the packed 132 features back to [33, 2, 2] per token and check the channel layout:
        // channels 0..16 (control latent) are 1.0, channel 16 (mask) + 17..33 (inpaint) are 0.0.
        let v = ctx.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let seq = lat_h * lat_w;
        for tok in 0..seq {
            for ch in 0..33 {
                for sub in 0..4 {
                    let val = v[tok * 132 + ch * 4 + sub];
                    if ch < 16 {
                        assert_eq!(val, 1.0, "control latent channel {ch} should be 1.0");
                    } else {
                        assert_eq!(val, 0.0, "mask/inpaint channel {ch} should be 0.0");
                    }
                }
            }
        }
    }
}
