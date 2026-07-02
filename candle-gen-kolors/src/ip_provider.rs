//! Kolors **IP-Adapter-Plus** provider (sc-5488, epic 5480) — reference-image (identity) conditioning
//! on Kolors, the candle (Windows/CUDA) sibling of the `mlx-gen-kolors` IP-Adapter path. It reuses the
//! vendored SDXL IP stack the InstantID/IP-Adapter slices built ([`candle_gen_sdxl`]) with the Kolors
//! conditioning the txt2img [`crate::pipeline`] already speaks:
//!
//! - the reference image's identity tokens come from the **CLIP ViT-L/14-336** image encoder
//!   ([`ClipVisionEncoder`] at [`VisionConfig::vit_l_14_336`]) → the IP-Adapter "plus"
//!   [`Resampler`] (`image_proj.*`, [`ResamplerConfig::kolors_plus`]) — the 1024-d penultimate → 16×2048
//!   tokens (vs the SDXL slice's ViT-H/14 1280-d tower);
//! - the decoupled cross-attention K/V (`ip_adapter.*`, 70 pairs) are installed into the **vendored
//!   SDXL [`UNet2DConditionModel`]** exactly as for SDXL/InstantID;
//! - the text path is **ChatGLM3-6B** (the Kolors encoder) projected to the cross-attention width by the
//!   Kolors `encoder_hid_proj` (4096→2048), and the denoise runs the Kolors **leading-Euler** sampler
//!   ([`KolorsEulerSampler`]) — NOT the SDXL EulerAncestral — so the output matches the Kolors txt2img
//!   numerics; the IP tokens are merely injected at `ip_scale` alongside the projected text context.
//!
//! Why the vendored SDXL UNet (not [`crate::unet::KolorsUNet`])? The txt2img Kolors UNet composes the
//! *stock* candle-transformers cross-attn blocks, which expose no decoupled-cross-attention seam. The
//! vendored SDXL stack is the only candle UNet carrying the IP install; since Kolors is an SDXL-family
//! UNet, its checkpoint loads into it 1:1 — the two Kolors deltas are handled outside the block stack:
//! the **5632** `add_embedding` (via [`UNet2DConditionModel::with_add_embedding`]) and the
//! `encoder_hid_proj` context projection (applied here, before [`UNet2DConditionModel::forward_instantid`],
//! since the vendored UNet's context arrives already at the cross-attention width).
//!
//! Two candle divergences carried from the SDXL slice hold: **CFG is uncond-first** (`[negative,
//! prompt]`, the Kolors txt2img convention), and the IP tokens live on the UNet
//! ([`UNet2DConditionModel::set_ip_context`], set once before the denoise — so
//! [`generate`](IpAdapterKolors::generate) takes `&mut self`). The uncond IP row is **literal zero
//! tokens** ([`IpImageEncoder::zeros_tokens`]).

use std::path::{Path, PathBuf};

use candle_gen::candle_core::{DType, Device, IndexOp, Tensor};
use candle_gen::candle_nn::{self as nn, Linear, Module, VarBuilder};
use candle_gen::gen_core::runtime::CancelFlag;
use candle_gen::gen_core::sampling::{schedule_sigmas, DiscreteModelSampling, Scheduler};
use candle_gen::gen_core::{Image, Progress};
use candle_gen::{CandleError, Result};
use candle_transformers::models::stable_diffusion::vae::AutoEncoderKL;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};

use candle_gen_sdxl::ip_adapter::{load_ip_kv_pairs, IpImageEncoder, Resampler, ResamplerConfig};
use candle_gen_sdxl::vision_encoder::{check_layer_count, ClipVisionEncoder, VisionConfig};
use candle_gen_sdxl::weights::Weights;
use candle_gen_sdxl::{denoise_curated, sdxl_unet_config, UNet2DConditionModel};

use crate::chatglm3::ChatGlmModel;
use crate::config::ChatGlmConfig;
use crate::pipeline::{curated_route, kolors_alpha_schedule, sdxl_vae_config, VAE_SCALE};
use crate::sampler::KolorsEulerSampler;
use crate::tokenizer::KolorsTokenizer;

/// The IP-Adapter compute dtype. Kolors runs the whole stack at **f32** (the candle port recipe — a
/// single matmul dtype; = mlx's "f32 activations over bf16 weights"), so the vendored UNet, the CLIP
/// image encoder, the Resampler, and the SDXL VAE all load at f32 here too.
const DTYPE: DType = DType::F32;

/// Kolors `add_embedding` dims (the Kolors `unet/config.json`): `addition_time_embed_dim = 256`,
/// `projection_class_embeddings_input_dim = 5632` (pooled 4096 + 6·256) — vs SDXL's 2816. The vendored
/// UNet needs the `add_embedding` head the plain `forward` omits.
const ADDITION_TIME_EMBED_DIM: usize = 256;
const PROJECTION_INPUT_DIM: usize = 5632;

/// ChatGLM3 context width (`encoder_hid_proj` in-features).
const CONTEXT_DIM: usize = 4096;
/// The SDXL/Kolors UNet cross-attention width (`encoder_hid_proj` out-features = the IP token width).
const CROSS_ATTENTION_DIM: usize = 2048;

/// The Kolors ViT-L/14-336 CLIP crop size (the IP-Adapter image tower).
const KOLORS_IP_IMAGE_SIZE: usize = 336;

/// The IP-Adapter-Plus bundle file inside a `Kwai-Kolors/Kolors-IP-Adapter-Plus` snapshot: the
/// `image_proj` Resampler + the 70 `ip_adapter.{n}.to_k_ip/to_v_ip` decoupled-attn pairs.
const IP_BUNDLE_FILE: &str = "ip_adapter_plus_general.safetensors";

/// Default `ip_adapter_scale` for Kolors IP-Adapter-Plus (the mlx `KolorsGenerator` registry's
/// `IP_DEFAULT_SCALE`; matches the torch Kolors IP-Adapter default — 0.6, vs SDXL's 0.7).
pub const DEFAULT_IP_ADAPTER_SCALE: f32 = 0.6;

/// Paths to the Kolors IP-Adapter-Plus checkpoints.
pub struct IpAdapterKolorsPaths {
    /// The `Kwai-Kolors/Kolors-diffusers` snapshot dir (`tokenizer/`, `text_encoder/` ChatGLM3-6B,
    /// `unet/` SDXL-family UNet, `vae/` SDXL VAE).
    pub kolors_base: PathBuf,
    /// The `Kwai-Kolors/Kolors-IP-Adapter-Plus` snapshot dir (`image_encoder/` CLIP ViT-L/14-336 +
    /// `ip_adapter_plus_general.safetensors`).
    pub ip_adapter: PathBuf,
}

/// One Kolors IP-Adapter generation request.
#[derive(Clone)]
pub struct IpAdapterKolorsRequest {
    pub prompt: String,
    pub negative: String,
    pub width: u32,
    pub height: u32,
    pub steps: usize,
    /// Classifier-free guidance scale.
    pub guidance: f32,
    /// IP-Adapter scale (the decoupled cross-attn weight on the image tokens).
    pub ip_adapter_scale: f32,
    /// Curated unified-sampler selection (epic 7114, sc-7297). `None` (or `euler_discrete`) keeps the
    /// bespoke leading-Euler default byte-exact (N1); a curated
    /// [`Solver`](candle_gen::gen_core::sampling::Solver) name routes the IP denoise
    /// through [`denoise_curated`] over the Kolors [`DiscreteModelSampling`].
    pub sampler: Option<String>,
    /// Curated σ-schedule selection (epic 7114). `None` ⇒ the native leading schedule; a [`Scheduler`]
    /// name re-shapes σ. A non-default scheduler alone also engages the curated path.
    pub scheduler: Option<String>,
    pub seed: u64,
    /// Cooperative cancellation, checked before each denoise step (the engine contract).
    pub cancel: CancelFlag,
}

impl Default for IpAdapterKolorsRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            negative: String::new(),
            width: 1024,
            height: 1024,
            steps: 50,
            guidance: 5.0,
            ip_adapter_scale: DEFAULT_IP_ADAPTER_SCALE,
            sampler: None,
            scheduler: None,
            seed: 0,
            cancel: CancelFlag::default(),
        }
    }
}

/// mmap an f32 [`VarBuilder`] over every `.safetensors` in `dir` (the ChatGLM3 encoder + UNet ship
/// sharded or single-file) — mirrors the txt2img pipeline's loader.
fn f32_vb(dir: &Path, device: &Device) -> Result<VarBuilder<'static>> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| CandleError::Msg(format!("kolors-ip: read {}: {e}", dir.display())))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(CandleError::Msg(format!(
            "kolors-ip: no .safetensors found in {} (expected a Kolors-diffusers snapshot)",
            dir.display()
        )));
    }
    // SAFETY: mmap of read-only weight files; the standard candle loading path.
    Ok(unsafe { VarBuilder::from_mmaped_safetensors(&files, DTYPE, device)? })
}

/// Resolve the CLIP image-encoder weight file from the IP-Adapter snapshot's `image_encoder/` dir:
/// `model.safetensors` then `model.fp16.safetensors` (the diffusers `image_encoder/` layout).
fn resolve_image_encoder(dir: &Path) -> Result<PathBuf> {
    if dir.is_file() {
        return Ok(dir.to_path_buf());
    }
    for name in ["model.safetensors", "model.fp16.safetensors"] {
        let p = dir.join(name);
        if p.is_file() {
            return Ok(p);
        }
    }
    Err(CandleError::Msg(format!(
        "kolors-ip: CLIP image encoder not found under {} (expected model.safetensors)",
        dir.display()
    )))
}

/// Loaded Kolors IP-Adapter model: the ChatGLM3 tokenizer + encoder, the `encoder_hid_proj` context
/// projection, the vendored SDXL UNet (with the IP K/V pairs installed + the 5632 `add_embedding`), the
/// CLIP ViT-L/14-336 image-token source, and the f32 SDXL VAE.
pub struct IpAdapterKolors {
    tokenizer: KolorsTokenizer,
    chatglm: ChatGlmModel,
    /// Kolors-only: project the ChatGLM3 context (4096) to the cross-attention width (2048), applied
    /// here (the vendored UNet has no `encoder_hid_proj`, unlike [`crate::unet::KolorsUNet`]).
    encoder_hid_proj: Linear,
    unet: UNet2DConditionModel,
    ip_encoder: IpImageEncoder,
    vae: AutoEncoderKL,
    device: Device,
}

impl IpAdapterKolors {
    /// Load the Kolors backbone (ChatGLM3 + SDXL-family UNet into the vendored stack + SDXL VAE) + the
    /// CLIP ViT-L/14-336 image encoder + the IP-Adapter-Plus Resampler, installing the decoupled-cross-
    /// attn K/V pairs into the UNet.
    pub fn load(paths: &IpAdapterKolorsPaths) -> Result<Self> {
        let device = candle_gen::default_device()?;
        let base = paths.kolors_base.as_path();

        let tokenizer = KolorsTokenizer::from_dir(base.join("tokenizer"))?;
        let chatglm = ChatGlmModel::new(
            ChatGlmConfig::chatglm3_6b(),
            f32_vb(&base.join("text_encoder"), &device)?,
        )?;

        // Vendored SDXL UNet from the Kolors `unet/` weights. One mmap'd VarBuilder feeds the UNet body,
        // the 5632 `add_embedding` head, and the `encoder_hid_proj` (all in the same checkpoint).
        let vs = f32_vb(&base.join("unet"), &device)?;
        let mut unet = UNet2DConditionModel::new(vs.clone(), 4, 4, false, sdxl_unet_config())?
            .with_add_embedding(vs.clone(), ADDITION_TIME_EMBED_DIM, PROJECTION_INPUT_DIM)?;
        let encoder_hid_proj =
            nn::linear(CONTEXT_DIM, CROSS_ATTENTION_DIM, vs.pp("encoder_hid_proj"))?;

        // IP-Adapter-Plus bundle: the Resampler (`image_proj.*`) + the decoupled K/V pairs
        // (`ip_adapter.*`), both at the UNet dtype.
        let bundle = paths.ip_adapter.join(IP_BUNDLE_FILE);
        let ipa = Weights::from_file(&bundle, &device, DTYPE)
            .map_err(|e| CandleError::Msg(format!("kolors-ip: load bundle {bundle:?}: {e}")))?;
        let resampler =
            Resampler::from_weights(&ipa, "image_proj", &ResamplerConfig::kolors_plus())?;
        unet.install_ip_adapter(load_ip_kv_pairs(&ipa)?)?;

        // CLIP ViT-L/14-336 image encoder (`vision_model.*`).
        let enc_cfg = VisionConfig::vit_l_14_336();
        let enc_path = resolve_image_encoder(&paths.ip_adapter.join("image_encoder"))?;
        let enc_w = Weights::from_file(&enc_path, &device, DTYPE).map_err(|e| {
            CandleError::Msg(format!(
                "kolors-ip: load CLIP image encoder {enc_path:?}: {e}"
            ))
        })?;
        check_layer_count(&enc_w, &enc_cfg)?;
        let encoder = ClipVisionEncoder::from_weights(&enc_w, &enc_cfg)?;
        let ip_encoder = IpImageEncoder::new(encoder, resampler, KOLORS_IP_IMAGE_SIZE);

        let vae = AutoEncoderKL::new(f32_vb(&base.join("vae"), &device)?, 3, 3, sdxl_vae_config())?;

        Ok(Self {
            tokenizer,
            chatglm,
            encoder_hid_proj,
            unet,
            ip_encoder,
            vae,
            device,
        })
    }

    /// Reference-image T2I: condition the Kolors generation on `reference`'s CLIP-ViT-L/14-336 identity
    /// tokens at `req.ip_adapter_scale`, denoising with the Kolors leading-Euler sampler.
    pub fn generate(
        &mut self,
        req: &IpAdapterKolorsRequest,
        reference: &Image,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        if req.cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }
        let use_guide = req.guidance > 1.0;

        // Everything that borrows `&self`, computed into owned values BEFORE the `&mut self.unet`
        // `set_ip_context` (so the disjoint-field borrows don't overlap — the SDXL/InstantID pattern).
        // CFG batch is [neg, pos] = uncond-first (the Kolors txt2img convention); without guidance only
        // the positive branch is built.
        let (pos_ctx, pos_pooled) = self.encode(&req.prompt)?;
        let (context, pooled) = if use_guide {
            let (neg_ctx, neg_pooled) = self.encode(&req.negative)?;
            (
                Tensor::cat(&[&neg_ctx, &pos_ctx], 0)?,
                Tensor::cat(&[&neg_pooled, &pos_pooled], 0)?,
            )
        } else {
            (pos_ctx, pos_pooled)
        };
        let batch = if use_guide { 2 } else { 1 };
        // Project the ChatGLM3 context to the cross-attention width once up front — the vendored UNet
        // (unlike `KolorsUNet`) has no `encoder_hid_proj`, so its context must already be 2048-wide.
        let projected = self.encoder_hid_proj.forward(&context)?;
        let time_ids = self.build_time_ids(batch, req.height, req.width)?;
        let ip_tokens = self.ip_tokens(reference, use_guide)?;

        let (lat_h, lat_w) = ((req.height / 8) as usize, (req.width / 8) as usize);

        // Set the IP image tokens on the UNet (constant across the denoise) — picked up by BOTH the
        // native and curated denoise paths via `forward_instantid`'s decoupled-attn branch.
        self.unet
            .set_ip_context(Some(&ip_tokens), req.ip_adapter_scale as f64)?;

        // Curated unified-sampler path (epic 7114, sc-7297): a curated solver name (≠ the native
        // `euler_discrete`) OR a curated scheduler routes the IP denoise through the additive
        // k-diffusion `denoise_curated`, which threads the decoupled-attn IP tokens (set above) through
        // the curated solver. The native leading-Euler default stays byte-exact (N1). The decision is
        // the shared [`curated_route`] (sc-8984) so the three Kolors entry points can't drift.
        let curated = curated_route(req.sampler.as_deref(), req.scheduler.as_deref());

        let latents = if let Some(sampler_name) = curated {
            // k-diffusion VE-σ sampling over the Kolors `DiscreteModelSampling`. A scheduler-only curated
            // run keeps `euler_discrete` (a non-solver alias) ⇒ the driver's euler fallback (N3).
            let sched = kolors_alpha_schedule()?;
            let ms = DiscreteModelSampling::sdxl(&sched);
            let native = schedule_sigmas(Scheduler::Normal, &ms, req.steps);
            let sigmas =
                candle_gen::resolve_schedule(req.scheduler.as_deref(), &ms, req.steps, &native);
            let prior = (self.initial_noise(req.seed, lat_h, lat_w)? * sigmas[0] as f64)?;
            // Pure IP: no ControlNet branch (`controls = &[]`); `projected` is the UNet cross-attn
            // conditioning (and fills the unused `controlnet_encoder` slot).
            denoise_curated(
                &self.unet,
                Some(sampler_name),
                &ms,
                &sigmas,
                prior,
                &projected,
                &pooled,
                &time_ids,
                req.guidance as f64,
                DTYPE,
                req.seed,
                &req.cancel,
                on_progress,
                &[],
                &projected,
            )?
        } else {
            let sampler = KolorsEulerSampler::new(req.steps).map_err(CandleError::Msg)?;
            let noise = self.initial_noise(req.seed, lat_h, lat_w)?;
            let mut latents = (noise * sampler.init_noise_sigma() as f64)?;
            let total = sampler.num_steps() as u32;
            for i in 0..sampler.num_steps() {
                if req.cancel.is_cancelled() {
                    return Err(CandleError::Canceled);
                }
                let scaled = (&latents / sampler.scale_in(i) as f64)?;
                let model_in = if use_guide {
                    Tensor::cat(&[&scaled, &scaled], 0)?
                } else {
                    scaled
                };
                let eps = self.unet.forward_instantid(
                    &model_in,
                    sampler.timestep(i) as f64,
                    &projected,
                    &pooled,
                    &time_ids,
                    None, // pure IP — no ControlNet down residuals
                    None, // … and no mid residual
                )?;
                let eps = if use_guide {
                    let ch = eps.chunk(2, 0)?;
                    let (uncond, cond) = (&ch[0], &ch[1]);
                    (uncond + ((cond - uncond)? * req.guidance as f64)?)?
                } else {
                    eps
                };
                latents = (&latents + (eps * sampler.step_dt(i) as f64)?)?;
                on_progress(Progress::Step {
                    current: i as u32 + 1,
                    total,
                });
            }
            latents
        };

        on_progress(Progress::Decoding);
        self.decode(&latents)
    }

    /// Encode one prompt → `(context [1, 256, 4096], pooled [1, 4096])` via the ChatGLM3 encoder.
    fn encode(&self, prompt: &str) -> Result<(Tensor, Tensor)> {
        let tokens = self.tokenizer.encode(prompt)?;
        Ok(self.chatglm.encode_prompt(&tokens)?)
    }

    /// The SDXL micro-conditioning `time_ids` = `(H, W, 0, 0, H, W)` per row, f32 `[batch, 6]` (the
    /// Kolors txt2img value — original == target, no crop).
    fn build_time_ids(&self, batch: usize, height: u32, width: u32) -> Result<Tensor> {
        let (hf, wf) = (height as f32, width as f32);
        let row = [hf, wf, 0.0, 0.0, hf, wf];
        let mut v = Vec::with_capacity(batch * 6);
        for _ in 0..batch {
            v.extend_from_slice(&row);
        }
        Ok(Tensor::from_vec(v, (batch, 6), &self.device)?)
    }

    /// Build the CFG-batched IP tokens from the reference image. **Uncond-first**: under CFG the uncond
    /// row is literal **zero tokens** (the IP-Adapter convention) stacked *before* the positive row.
    fn ip_tokens(&self, reference: &Image, use_guide: bool) -> Result<Tensor> {
        let tokens = self.ip_encoder.tokens(reference, &self.device)?; // [1, 16, 2048]
        if use_guide {
            let zeros = self.ip_encoder.zeros_tokens(&self.device)?;
            Ok(Tensor::cat(&[&zeros, &tokens], 0)?) // uncond (zeros) first, then cond
        } else {
            Ok(tokens)
        }
    }

    /// sc-3673 deterministic, launch-portable initial noise `[1, 4, lat_h, lat_w]`: N(0,1) from a
    /// fixed-algorithm CPU RNG seeded by `seed`, moved to the device (matches the txt2img pipeline).
    fn initial_noise(&self, seed: u64, lat_h: usize, lat_w: usize) -> Result<Tensor> {
        let n = 4 * lat_h * lat_w;
        let mut rng = StdRng::seed_from_u64(seed);
        let noise: Vec<f32> = (0..n).map(|_| StandardNormal.sample(&mut rng)).collect();
        Ok(Tensor::from_vec(noise, (1, 4, lat_h, lat_w), &Device::Cpu)?.to_device(&self.device)?)
    }

    /// VAE-decode latents `[1, 4, H/8, W/8]` → an RGB8 [`Image`] (un-scale by [`VAE_SCALE`], `x/2 + 0.5`,
    /// clamp, ×255) — the txt2img pipeline's decode.
    fn decode(&self, latents: &Tensor) -> Result<Image> {
        let unscaled = (latents / VAE_SCALE)?;
        let img = self.vae.decode(&unscaled)?;
        let img = ((img / 2.)? + 0.5)?.clamp(0f32, 1f32)?;
        let img = (img * 255.)?
            .to_dtype(DType::U8)?
            .i(0)?
            .to_device(&Device::Cpu)?;
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
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The request defaults match the Kolors IP-Adapter production knobs (1024², 50 steps, CFG 5.0, ip
    /// scale 0.6).
    #[test]
    fn request_defaults() {
        let r = IpAdapterKolorsRequest::default();
        assert_eq!((r.width, r.height), (1024, 1024));
        assert_eq!(r.steps, 50);
        assert_eq!(r.guidance, 5.0);
        assert_eq!(r.ip_adapter_scale, DEFAULT_IP_ADAPTER_SCALE);
        assert_eq!(DEFAULT_IP_ADAPTER_SCALE, 0.6);
        // The curated knobs default to None ⇒ the bespoke leading-Euler path (N1 byte-exact).
        assert!(r.sampler.is_none() && r.scheduler.is_none());
        assert!(!r.cancel.is_cancelled());
    }

    /// The Kolors `add_embedding` projection input is 5632 (pooled 4096 + 6·256) — vs SDXL's 2816 —
    /// pinning the one numeric delta that lets the vendored SDXL UNet carry the Kolors checkpoint.
    #[test]
    fn kolors_add_embedding_dims() {
        assert_eq!(ADDITION_TIME_EMBED_DIM, 256);
        assert_eq!(PROJECTION_INPUT_DIM, 4096 + 6 * 256);
        assert_eq!(CONTEXT_DIM, 4096);
        assert_eq!(CROSS_ATTENTION_DIM, 2048);
    }

    /// `resolve_image_encoder`: a directory resolves `model.safetensors`; a direct file is used as-is; a
    /// missing dir errors loudly.
    #[test]
    fn image_encoder_resolution() {
        let dir = std::env::temp_dir().join(format!("candle_kolors_ip_enc_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        assert!(resolve_image_encoder(&dir).is_err());
        let f = dir.join("model.safetensors");
        std::fs::write(&f, b"x").unwrap();
        assert_eq!(resolve_image_encoder(&dir).unwrap(), f);
        assert_eq!(resolve_image_encoder(&f).unwrap(), f);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
