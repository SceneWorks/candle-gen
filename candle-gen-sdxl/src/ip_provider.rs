//! SDXL **IP-Adapter-Plus** provider (sc-5488, epic 5480) — reference-image (identity) conditioning
//! on SDXL/RealVisXL, the candle (Windows/CUDA) sibling of the `mlx-gen-sdxl` IP-Adapter path. It is
//! the [`crate::ip_adapter`] + [`crate::denoise`] stack the InstantID port (sc-5491) built, composed
//! **without** a face embedder and **without** a ControlNet:
//!
//! - the reference image's identity tokens come from the **CLIP ViT-H/14 image encoder**
//!   ([`ClipVisionEncoder`]) → the IP-Adapter "plus" [`Resampler`] (`image_proj.*`), not ArcFace;
//! - the decoupled cross-attention K/V (`ip_adapter.*`) are installed into the vendored UNet exactly
//!   as for InstantID;
//! - the denoise is [`denoise_ip_multi_control`] with an **empty** control set — pure IP, no
//!   IdentityNet/OpenPose residuals.
//!
//! The two candle divergences carried from sc-5491 hold: **CFG is uncond-first** (`[negative,
//! prompt]`), and the IP tokens live on the UNet ([`UNet2DConditionModel::set_ip_context`], set once
//! before the denoise — so [`generate`](IpAdapterSdxl::generate) takes `&mut self`). The one
//! IP-Adapter-specific difference from InstantID: the uncond IP row is **literal zero tokens**
//! ([`IpImageEncoder::zeros_tokens`]), not `Resampler(zeros)`.

use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::runtime::CancelFlag;
use candle_gen::gen_core::sampling::{schedule_sigmas, DiscreteModelSampling, Scheduler, Solver};
use candle_gen::gen_core::{Image, Progress};
// Shared ancestral-step RNG salt (`seed + STEP_RNG_SALT`) — one home in `candle-gen` (sc-9043 / F-059).
// `LatentDecoder` is the decode seam the optional PiD student implements (epic 7840, sc-8044).
use candle_gen::gen_core::PidWeights;
use candle_gen::{CandleError, LatentDecoder, Result, STEP_RNG_SALT};
use candle_gen_pid::{PidDecoder, PidEngine};

use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::denoise::{
    decode_image, denoise_curated, denoise_ip_multi_control, seeded_prior, seeded_sigma_prior,
    text_time_ids, Denoiser,
};
use crate::ip_adapter::{load_ip_kv_pairs, IpImageEncoder, Resampler, ResamplerConfig};
use crate::loaders::{load_instantid_unet, load_sdxl_vae};
use crate::pipeline::sdxl_alpha_schedule;
use crate::sampler::EulerAncestralSampler;
use crate::unet::UNet2DConditionModel;
use crate::vision_encoder::{check_layer_count, ClipVisionEncoder, VisionConfig};
use crate::weights::Weights;
use crate::{conditioning::SdxlConditioner, AutoEncoderKL, PID_BACKBONE};

/// The IP-Adapter compute dtype — fp16, matching the production SDXL path (the VAE is the f16-stable
/// `madebyollin/sdxl-vae-fp16-fix`; the CLIP image encoder runs at this dtype too).
const DTYPE: DType = DType::F16;

/// Default `ip_adapter_scale` for SDXL IP-Adapter-Plus (the worker's `ipAdapterScale` default, matching
/// the torch `SdxlDiffusersAdapter`).
pub const DEFAULT_IP_ADAPTER_SCALE: f32 = 0.7;

/// Paths to the SDXL IP-Adapter-Plus checkpoints.
pub struct IpAdapterSdxlPaths {
    /// SDXL base snapshot dir (`unet/`, `text_encoder{,_2}/`, …).
    pub sdxl_base: PathBuf,
    /// The IP-Adapter-Plus bundle (`ip-adapter-plus_sdxl_vit-h.safetensors`: `image_proj.*` Resampler +
    /// `ip_adapter.*` K/V pairs).
    pub ip_adapter: PathBuf,
    /// The CLIP ViT-H/14 image encoder — a dir (`model(.fp16).safetensors`) or the file directly
    /// (`h94/IP-Adapter` `models/image_encoder`).
    pub image_encoder: PathBuf,
}

/// One SDXL IP-Adapter generation request.
#[derive(Clone)]
pub struct IpAdapterSdxlRequest {
    pub prompt: String,
    pub negative: String,
    pub width: u32,
    pub height: u32,
    pub steps: usize,
    /// Classifier-free guidance scale.
    pub guidance: f32,
    /// IP-Adapter scale (the decoupled cross-attn weight on the image tokens).
    pub ip_adapter_scale: f32,
    /// Curated unified-sampler selection (epic 7114, sc-7297). `None` (or `euler_ancestral`) keeps the
    /// bespoke ancestral default byte-exact (N1); a curated [`Solver`] name routes the IP denoise
    /// through [`denoise_curated`] over the SDXL [`DiscreteModelSampling`].
    pub sampler: Option<String>,
    /// Curated σ-schedule selection (epic 7114). `None` ⇒ the discrete default; a [`Scheduler`] name
    /// re-shapes σ. A non-default scheduler alone also engages the curated path.
    pub scheduler: Option<String>,
    pub seed: u64,
    /// Opt into the PiD super-resolving decoder (epic 7840, sc-8044): when `true` **and** the model was
    /// loaded with [`with_pid`](IpAdapterSdxl::with_pid), the final latent is decoded by the `sdxl` PiD
    /// student (4× SR → 2K/4K) instead of the native VAE. `false` (default) keeps the byte-exact VAE decode.
    pub use_pid: bool,
    /// Cooperative cancellation, checked before each denoise step (the engine contract).
    pub cancel: CancelFlag,
}

impl Default for IpAdapterSdxlRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            negative: String::new(),
            width: 1024,
            height: 1024,
            steps: 30,
            guidance: 5.0,
            ip_adapter_scale: DEFAULT_IP_ADAPTER_SCALE,
            sampler: None,
            scheduler: None,
            seed: 0,
            use_pid: false,
            cancel: CancelFlag::default(),
        }
    }
}

/// Resolve the CLIP image-encoder weight file from a dir-or-file path: a file is used directly; a dir
/// resolves `model.safetensors` then `model.fp16.safetensors` (the diffusers `image_encoder/` layout).
fn resolve_image_encoder(path: &Path) -> Result<PathBuf> {
    if path.is_file() {
        return Ok(path.to_path_buf());
    }
    for name in ["model.safetensors", "model.fp16.safetensors"] {
        let p = path.join(name);
        if p.is_file() {
            return Ok(p);
        }
    }
    Err(CandleError::Msg(format!(
        "ip-adapter: CLIP image encoder not found under {} (expected a model.safetensors or a \
         direct .safetensors file)",
        path.display()
    )))
}

/// Reject `steps == 0` loudly instead of running zero denoise iterations and VAE-decoding the pure
/// scaled prior noise — a fast typed error, not GPU time burned on garbage (sc-9016, F-032). Mirrors
/// the registered `SdxlGenerator::validate` steps floor; this worker-driven IP path has no gen-core
/// capability floor upstream of it.
fn reject_zero_steps(steps: usize) -> Result<()> {
    if steps == 0 {
        return Err(CandleError::Msg(
            "sdxl ip-adapter: steps must be >= 1 (an explicit 0 renders undenoised noise)".into(),
        ));
    }
    Ok(())
}

/// Loaded SDXL IP-Adapter model: the vendored SDXL UNet (with the IP K/V pairs installed + the
/// `add_embedding` head) + the dual-CLIP conditioner + the CLIP image encoder/Resampler token source +
/// the f16 VAE + the ancestral sampler.
pub struct IpAdapterSdxl {
    conditioner: SdxlConditioner,
    unet: UNet2DConditionModel,
    ip_encoder: IpImageEncoder,
    vae: AutoEncoderKL,
    sampler: EulerAncestralSampler,
    /// Optional PiD super-resolving decoder (epic 7840, sc-8044), attached via [`with_pid`](Self::with_pid).
    /// Composes the SDXL VAE, so it loads the SAME `sdxl` student ([`PID_BACKBONE`]) as the registered
    /// SDXL provider.
    pid: Option<PidEngine>,
    device: Device,
}

impl IpAdapterSdxl {
    /// Load the SDXL backbone + dual-CLIP conditioner + CLIP ViT-H image encoder + IP-Adapter-Plus
    /// Resampler, installing the decoupled-cross-attn K/V pairs into the UNet.
    pub fn load(paths: &IpAdapterSdxlPaths) -> Result<Self> {
        let device = candle_gen::default_device()?;
        let root = paths.sdxl_base.as_path();

        let conditioner = SdxlConditioner::load(root, &device, DTYPE)?;
        let mut unet = load_instantid_unet(root, &device, DTYPE)?;

        // IP-Adapter-Plus bundle: the Resampler (`image_proj.*`) + the decoupled K/V pairs
        // (`ip_adapter.*`), both at the UNet dtype.
        let ipa = Weights::from_file(&paths.ip_adapter, &device, DTYPE).map_err(|e| {
            CandleError::Msg(format!(
                "ip-adapter: load bundle {:?}: {e}",
                paths.ip_adapter
            ))
        })?;
        let resampler =
            Resampler::from_weights(&ipa, "image_proj", &ResamplerConfig::plus_sdxl_vit_h())?;
        unet.install_ip_adapter(load_ip_kv_pairs(&ipa)?)?;

        // CLIP ViT-H/14 image encoder (`vision_model.*`).
        let enc_cfg = VisionConfig::vit_h_14();
        let enc_path = resolve_image_encoder(&paths.image_encoder)?;
        let enc_w = Weights::from_file(&enc_path, &device, DTYPE).map_err(|e| {
            CandleError::Msg(format!(
                "ip-adapter: load CLIP image encoder {enc_path:?}: {e}"
            ))
        })?;
        check_layer_count(&enc_w, &enc_cfg)?;
        let encoder = ClipVisionEncoder::from_weights(&enc_w, &enc_cfg)?;
        let ip_encoder = IpImageEncoder::new(encoder, resampler, enc_cfg.image_size);

        let vae = load_sdxl_vae(&device, DTYPE)?;
        Ok(Self {
            conditioner,
            unet,
            ip_encoder,
            vae,
            sampler: EulerAncestralSampler::sdxl(),
            pid: None,
            device,
        })
    }

    /// Attach the optional PiD super-resolving decoder (epic 7840, sc-8044). Same [`PidWeights`] load-spec
    /// as the registry SDXL provider; the IP-Adapter composes the SDXL VAE so it loads the **same**
    /// [`PID_BACKBONE`] (`sdxl`) student. A `use_pid = true` request then decodes through it (4× SR) instead
    /// of the native VAE; without it, `use_pid` errors loudly. Call after [`load`](Self::load).
    pub fn with_pid(mut self, pid: &PidWeights) -> Result<Self> {
        self.pid = Some(PidEngine::from_spec(pid, PID_BACKBONE, &self.device)?);
        Ok(self)
    }

    /// Mint the per-generation PiD decoder when the request opted in (`use_pid`) and a student is loaded;
    /// `None` keeps the native VAE decode. Errors loudly if `use_pid` is set without a prior
    /// [`with_pid`](Self::with_pid). A clean-latent (σ=0) decoder bound to the prompt + seed; the request
    /// cancel threads in for a cancellable SR decode.
    fn pid_decoder_for(&self, req: &IpAdapterSdxlRequest) -> Result<Option<PidDecoder>> {
        // Route through the shared guarded seam (sc-11242 / F-091) so the SR decode is budgeted
        // (F-013 sc-9095) and spatially tiled (sc-10087). Clean-latent σ=0 decode, single image.
        candle_gen_pid::resolve_pid_decoder_for_fields(
            self.pid.as_ref(),
            req.use_pid,
            &req.prompt,
            1,
            req.width,
            req.height,
            &req.cancel,
            req.seed,
            "sdxl ip-adapter",
            0.0,
        )
    }

    /// Reference-image T2I: condition the SDXL generation on `reference`'s CLIP-ViT-H identity tokens
    /// at `req.ip_adapter_scale` (no ControlNet — pure IP).
    pub fn generate(
        &mut self,
        req: &IpAdapterSdxlRequest,
        reference: &Image,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        if req.cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }
        reject_zero_steps(req.steps)?;
        let cfg_on = req.guidance > 1.0;

        // Everything that borrows `&self`, computed into owned values BEFORE the `&mut self.unet`
        // `set_ip_context` (so the disjoint-field borrows don't overlap — the InstantID pattern).
        let (conditioning, pooled) = self
            .conditioner
            .encode(&req.prompt, &req.negative, cfg_on)?;
        let batch = conditioning.dim(0)?;
        let time_ids = text_time_ids(batch, &self.device, DTYPE)?;
        let ip_tokens = self.ip_tokens(reference, cfg_on)?;

        // Set the IP image tokens on the UNet (constant across the denoise — phase 2c/2e design) —
        // picked up by BOTH the native ancestral and the curated denoise paths.
        self.unet
            .set_ip_context(Some(&ip_tokens), req.ip_adapter_scale as f64)?;

        // Curated unified-sampler path (epic 7114, sc-7297): a curated solver name (≠ the bespoke
        // `euler_ancestral` default) OR a non-discrete scheduler routes the IP denoise through the
        // additive k-diffusion `denoise_curated` (the decoupled-attn IP tokens, set above, threaded
        // through the curated solver). The bespoke ancestral default stays byte-exact (N1).
        let sampler_name = req.sampler.as_deref().unwrap_or("euler_ancestral");
        let scheduler_curated = req
            .scheduler
            .as_deref()
            .and_then(Scheduler::from_name)
            .is_some();
        let sampler_curated =
            Solver::from_name(sampler_name).is_some() && sampler_name != "euler_ancestral";

        let latents = if sampler_curated || scheduler_curated {
            let ms = DiscreteModelSampling::sdxl(&sdxl_alpha_schedule()?);
            let native = schedule_sigmas(Scheduler::Normal, &ms, req.steps);
            let sigmas =
                candle_gen::resolve_schedule(req.scheduler.as_deref(), &ms, req.steps, &native);
            let prior =
                seeded_sigma_prior(req.seed, req.width, req.height, sigmas[0], &self.device)?;
            // Pure IP: no ControlNet branch (`controls = &[]`); `conditioning` is the UNet cross-attn
            // context (and fills the unused `controlnet_encoder` slot). IP tokens preconditioned above.
            denoise_curated(
                &self.unet,
                Some(sampler_name),
                &ms,
                &sigmas,
                prior,
                &conditioning,
                &pooled,
                &time_ids,
                req.guidance as f64,
                DTYPE,
                req.seed,
                &req.cancel,
                on_progress,
                &[],
                &conditioning,
            )?
        } else {
            let prior = self.seeded_prior_with(req.seed, req.width, req.height)?;
            let d = Denoiser {
                unet: &self.unet,
                sampler: &self.sampler,
            };
            let steps = self.sampler.timesteps(req.steps, self.sampler.max_time());
            let mut rng = StdRng::seed_from_u64(req.seed.wrapping_add(STEP_RNG_SALT));
            denoise_ip_multi_control(
                &d,
                prior,
                &conditioning,
                &pooled,
                &time_ids,
                req.guidance as f64,
                &steps,
                &mut rng,
                &req.cancel,
                on_progress,
                &[],           // pure IP — no ControlNet branches
                &conditioning, // controlnet_encoder is unused with no controls
            )?
        };
        on_progress(Progress::Decoding);
        // Decode the final latent: native SDXL VAE by default, or the `sdxl` PiD student (4× SR) when this
        // generation opted in (`req.use_pid`) and `with_pid` loaded one (epic 7840, sc-8044).
        let pid_decoder = self.pid_decoder_for(req)?;
        let pid_ref = pid_decoder.as_ref().map(|d| d as &dyn LatentDecoder);
        decode_image(&self.vae, &latents, pid_ref)
    }

    /// Build the CFG-batched IP tokens from the reference image. **Uncond-first**: under CFG the uncond
    /// row is literal **zero tokens** (the IP-Adapter convention — `IPAdapter` zeros the image-embed
    /// output, not the Resampler input) stacked *before* the positive row.
    fn ip_tokens(&self, reference: &Image, cfg_on: bool) -> Result<Tensor> {
        let tokens = self.ip_encoder.tokens(reference, &self.device)?; // [1, 16, 2048]
        if cfg_on {
            let zeros = self.ip_encoder.zeros_tokens(&self.device)?;
            Ok(Tensor::cat(&[&zeros, &tokens], 0)?) // uncond (zeros) first, then cond
        } else {
            Ok(tokens)
        }
    }

    /// Seed a `StdRng` and sample the prior latents for a `width × height` render (the prior stream is
    /// keyed by `seed`; the per-step ancestral noise stream by `seed + STEP_RNG_SALT`).
    fn seeded_prior_with(&self, seed: u64, width: u32, height: u32) -> Result<Tensor> {
        let mut rng = StdRng::seed_from_u64(seed);
        seeded_prior(&self.sampler, &mut rng, width, height, &self.device, DTYPE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The request defaults match the SDXL IP-Adapter production knobs (1024², 30 steps, ip scale 0.7).
    #[test]
    fn request_defaults() {
        let r = IpAdapterSdxlRequest::default();
        assert_eq!((r.width, r.height), (1024, 1024));
        assert_eq!(r.steps, 30);
        assert_eq!(r.ip_adapter_scale, DEFAULT_IP_ADAPTER_SCALE);
        // The curated knobs default to None ⇒ the bespoke ancestral path (N1 byte-exact).
        assert!(r.sampler.is_none() && r.scheduler.is_none());
        assert!(!r.cancel.is_cancelled());
    }

    /// `steps == 0` is rejected with a fast, actionable error (never decoded as undenoised noise);
    /// a valid step count passes (sc-9016, F-032).
    #[test]
    fn zero_steps_is_rejected() {
        let err = reject_zero_steps(0).expect_err("steps==0 must be rejected");
        assert!(err.to_string().contains("steps must be >= 1"), "got: {err}");
        assert!(reject_zero_steps(1).is_ok());
        assert!(reject_zero_steps(30).is_ok());
    }

    /// `resolve_image_encoder`: a directory resolves `model.safetensors`; a missing dir errors loudly.
    #[test]
    fn image_encoder_resolution() {
        let dir = std::env::temp_dir().join(format!(
            "candle_ipadapter_enc_{}_{}",
            std::process::id(),
            "t"
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // No weight file yet → error.
        assert!(resolve_image_encoder(&dir).is_err());
        // Create a model.safetensors stand-in → resolves to it.
        let f = dir.join("model.safetensors");
        std::fs::write(&f, b"x").unwrap();
        assert_eq!(resolve_image_encoder(&dir).unwrap(), f);
        // A direct file path is used as-is.
        assert_eq!(resolve_image_encoder(&f).unwrap(), f);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
