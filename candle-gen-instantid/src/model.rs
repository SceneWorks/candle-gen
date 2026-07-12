//! InstantID provider (sc-5491, epic 5480) — identity-preserving SDXL T2I, the candle (Windows/CUDA)
//! sibling of `mlx-gen-instantid::model`. Composes the candle SDXL building blocks
//! ([`candle_gen_sdxl`]) + the native face stack ([`candle_gen_face`]) the earlier phases built, rather
//! than re-implementing them. One denoise step applies **both** identity paths driven by the reference
//! face (the vendored `StableDiffusionXLInstantIDPipeline`):
//! - the **face IP tokens** (ArcFace 512 → 16×2048 via the InstantID [`Resampler`]) injected into every
//!   UNet cross-attention at `ip_adapter_scale`;
//! - the **IdentityNet** (a stock SDXL ControlNet) on the 5-keypoint `draw_kps` control image, its
//!   cross-attention conditioned on the *same* face tokens, residuals added into the UNet.
//!
//! **Two candle-specific divergences from the mlx crate** (the design set in sc-5491 phases 2c–2e):
//!  1. **CFG is uncond-first** (`[negative, prompt]`), matching the candle txt2img + `denoise`
//!     convention — so the uncond face row is `Resampler(zeros)` stacked *first*. (mlx is positive-first.)
//!  2. **The face IP tokens live on the UNet** ([`UNet2DConditionModel::set_ip_context`]), set once per
//!     generation before the denoise loop — so the `generate*` methods take `&mut self`. mlx instead
//!     threads the tokens into `forward_with_ip_control` each step. Determinism is a seeded CPU `StdRng`
//!     (the launch-portable sc-3673 contract), seeded per request.

use std::path::{Path, PathBuf};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::runtime::CancelFlag;
use candle_gen::gen_core::sampling::{
    schedule_sigmas, AlphaSchedule, DiscreteModelSampling, Scheduler, Solver,
};
use candle_gen::gen_core::{
    AdapterSpec, DetectedFace, FaceEmbedder, Image, PidWeights, Progress, WeightsSource,
};
// Shared ancestral-step RNG salt (`seed + STEP_RNG_SALT`) — one home in `candle-gen` (sc-9043 / F-059).
// `LatentDecoder` is the decode seam the optional PiD student implements (epic 7840, sc-8373).
use candle_gen::{CandleError, LatentDecoder, Result, STEP_RNG_SALT};

use candle_gen_face::CandleFaceAnalysis;
use candle_gen_pid::{PidDecoder, PidEngine};
use candle_gen_sdxl::ip_adapter::{load_ip_kv_pairs, Resampler, ResamplerConfig};
use candle_gen_sdxl::weights::Weights;
use candle_gen_sdxl::{
    decode_image, denoise_curated, denoise_ip_multi_control, load_instantid_unet,
    load_instantid_unet_with_adapters, load_sdxl_controlnet, load_sdxl_vae,
    preprocess_control_image, seeded_prior, seeded_sigma_prior, text_time_ids, AutoEncoderKL,
    ControlContext, ControlNet, Denoiser, EulerAncestralSampler, SdxlConditioner,
    UNet2DConditionModel, PID_BACKBONE,
};

use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::kps;
use crate::openpose::{self, BodyPoint, STICKWIDTH};
use crate::resample::resize_lanczos_u8;
use crate::restore;

/// The InstantID compute dtype — fp16, matching the production SDXL path (the VAE is the f16-stable
/// `madebyollin/sdxl-vae-fp16-fix`; the face stack runs f32 inside [`candle_gen_face`]).
const DTYPE: DType = DType::F16;

/// ArcFace embedding width.
const EMBEDDING_DIM: usize = 512;
/// Number of face landmarks `draw_kps` indexes (`[left_eye, right_eye, nose, mouth_left, mouth_right]`).
const FACE_KP_COUNT: usize = 5;

/// Default `ip_adapter_scale` (the vendored pipeline's `set_ip_adapter_scale(0.8)`).
pub const DEFAULT_IP_SCALE: f32 = 0.8;
/// Default IdentityNet `controlnet_conditioning_scale` (the vendored default 0.8).
pub const DEFAULT_CONTROLNET_SCALE: f32 = 0.8;
/// Default OpenPose `controlnet_conditioning_scale` in pose mode (the worker's `openPoseScale` 0.7).
pub const DEFAULT_OPENPOSE_SCALE: f32 = 0.7;
/// The no-face-visible OpenPose scale floor (`instantid_adapter.py:425` `max(openPoseScale, 0.85)`).
const NO_FACE_OPENPOSE_FLOOR: f32 = 0.85;
/// Default face-restoration prompt (sc-3380), **gender-neutral by design** (the worker's hardcoded
/// "the woman's face" is a latent bug; the native port neutralizes it). Callers may override.
pub const FACE_RESTORE_PROMPT: &str =
    "close-up portrait of the face, soft natural light, photorealistic, sharp focus";
/// The face-restore crop padding factor (`instantid_adapter.py:483` `* 1.9`).
const FACE_RESTORE_CROP_PAD: f32 = 1.9;

/// SDXL ε-prediction α-cumprod schedule params (`scaled_linear` β over 1000 train steps) — the source
/// for the curated unified-sampler path's [`DiscreteModelSampling`] (epic 7114, sc-7297). The same
/// values the base candle SDXL pipeline uses; InstantID rides the stock SDXL noise schedule.
const SDXL_TRAIN_STEPS: usize = 1000;
const SDXL_BETA_START: f32 = 0.00085;
const SDXL_BETA_END: f32 = 0.012;

/// Reject a caller-supplied `kps` slice shorter than [`FACE_KP_COUNT`] with a typed error (without it,
/// `kps::draw_kps` would panic on a truncated landmark list — F-079).
fn validate_kps(kps: &[(f32, f32)]) -> Result<()> {
    if kps.len() < FACE_KP_COUNT {
        return Err(CandleError::Msg(format!(
            "instantid: need {FACE_KP_COUNT} face keypoints, got {}",
            kps.len()
        )));
    }
    Ok(())
}

/// Reject `steps == 0` with a typed error instead of running zero denoise iterations and VAE-decoding
/// the pure scaled prior noise (sc-9016, F-032). Mirrors the registered `SdxlGenerator::validate` steps
/// floor; the worker-driven InstantID path has no gen-core capability floor upstream of it. The default
/// entries (`generate`, `generate_angle`, `generate_with_kps`) funnel through `generate_with`; the
/// worker-driven pose path (`generate_pose`, `generate_pose_with`, sc-3117) bypasses `generate_with` and
/// runs its own `run_identity_denoise`, so it calls `check_steps` directly. Both `run_identity_denoise`
/// callers are guarded.
fn check_steps(steps: usize) -> Result<()> {
    if steps == 0 {
        return Err(CandleError::Msg(
            "instantid: steps must be >= 1 (an explicit 0 renders undenoised noise)".into(),
        ));
    }
    Ok(())
}

/// Reject a non-512-d ArcFace embedding with a typed error.
fn check_embedding(embedding: &[f32]) -> Result<()> {
    if embedding.len() != EMBEDDING_DIM {
        return Err(CandleError::Msg(format!(
            "instantid: ArcFace embedding must be {EMBEDDING_DIM}-d, got {}",
            embedding.len()
        )));
    }
    Ok(())
}

/// Paths to the InstantID checkpoints.
pub struct InstantIdPaths {
    /// SDXL base snapshot dir (`unet/`, `text_encoder{,_2}/`, …).
    pub sdxl_base: PathBuf,
    /// IdentityNet `ControlNetModel` — a dir (`diffusion_pytorch_model(.fp16).safetensors`) or a file.
    pub identitynet: WeightsSource,
    /// Converted `ip-adapter.safetensors` (`image_proj.*` Resampler + `ip_adapter.*` K/V pairs).
    pub ip_adapter: PathBuf,
    /// User LoRA/LoKr style/character adapters to merge onto the SDXL UNet at load (sc-6038).
    /// InstantID runs on a stock SDXL (RealVisXL) UNet, so SDXL-family LoRAs apply on top of the
    /// IdentityNet + face IP-Adapter. Empty (the common case) is a no-op. Distinct from
    /// [`ip_adapter`](Self::ip_adapter), which is the InstantID identity IP-Adapter.
    pub adapters: Vec<AdapterSpec>,
}

/// One InstantID generation request.
#[derive(Clone)]
pub struct InstantIdRequest {
    pub prompt: String,
    pub negative: String,
    pub width: u32,
    pub height: u32,
    pub steps: usize,
    /// Classifier-free guidance scale (the vendored default 5.0).
    pub guidance: f32,
    pub ip_adapter_scale: f32,
    pub controlnet_scale: f32,
    /// OpenPose `controlnet_conditioning_scale` — used only by [`InstantId::generate_pose`].
    pub openpose_scale: f32,
    /// Curated unified-sampler selection (epic 7114, sc-7297). `None` (or `euler_ancestral`) keeps
    /// InstantID's bespoke ancestral default byte-exact (N1); a curated [`Solver`] name routes the
    /// dual-conditioning denoise through [`denoise_curated`] over the SDXL [`DiscreteModelSampling`].
    pub sampler: Option<String>,
    /// Curated σ-schedule selection (epic 7114). `None` ⇒ the discrete default; a [`Scheduler`] name
    /// re-shapes σ over the schedule. A non-default scheduler alone also engages the curated path.
    pub scheduler: Option<String>,
    pub seed: u64,
    /// Opt into the PiD super-resolving decoder (epic 7840, sc-8373) for this generation: when `true`
    /// **and** the model was loaded with [`with_pid`](InstantId::with_pid), the final latent is decoded
    /// by the `sdxl` PiD student (4× SR → 2K/4K) instead of the native VAE. `false` (the default) keeps
    /// the byte-exact VAE decode. The face-restore re-render always stays on the VAE regardless (its
    /// paste-back assumes a native-resolution crop) — see [`restore_face`](InstantId::restore_face).
    pub use_pid: bool,
    /// Cooperative cancellation, checked before each denoise step + between phases (the engine
    /// contract every provider honors). `Clone` shares the flag so the caller keeps a cancel handle.
    pub cancel: CancelFlag,
}

impl Default for InstantIdRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            negative: String::new(),
            width: 1024,
            height: 1024,
            steps: 30,
            guidance: 5.0,
            ip_adapter_scale: DEFAULT_IP_SCALE,
            controlnet_scale: DEFAULT_CONTROLNET_SCALE,
            openpose_scale: DEFAULT_OPENPOSE_SCALE,
            sampler: None,
            scheduler: None,
            seed: 0,
            use_pid: false,
            cancel: CancelFlag::default(),
        }
    }
}

/// Loaded InstantID model: the SDXL backbone (the vendored UNet with the face IP K/V pairs installed +
/// the `add_embedding` head) + the dual-CLIP conditioner + the IdentityNet + the face Resampler + the
/// f16 VAE + the ancestral sampler, plus optional OpenPose CN (pose mode) and face-analysis stack.
pub struct InstantId {
    conditioner: SdxlConditioner,
    unet: UNet2DConditionModel,
    identitynet: ControlNet,
    /// The OpenPose ControlNet for pose mode (sc-3117), attached via [`with_openpose`](Self::with_openpose).
    openpose: Option<ControlNet>,
    resampler: Resampler,
    vae: AutoEncoderKL,
    sampler: EulerAncestralSampler,
    /// SDXL ε-prediction α-cumprod schedule (`scaled_linear`), built once at load — the
    /// [`DiscreteModelSampling`] source for the curated unified-sampler path (epic 7114, sc-7297).
    alpha_schedule: AlphaSchedule,
    face: Option<CandleFaceAnalysis>,
    /// Optional PiD super-resolving decoder (epic 7840, sc-8373), attached via [`with_pid`](Self::with_pid).
    /// `Some` ⇒ a `req.use_pid` generation decodes the final SDXL latent through the `sdxl` PiD student
    /// (4× SR) instead of the native VAE. InstantID composes the SDXL VAE, so it loads the SAME `sdxl`
    /// checkpoint ([`PID_BACKBONE`]) as the registered SDXL provider — there is no InstantID-specific PiD.
    pid: Option<PidEngine>,
    device: Device,
}

impl InstantId {
    /// Load the SDXL backbone + dual-CLIP conditioner + IdentityNet + face Resampler, installing the
    /// decoupled-cross-attn K/V pairs into the UNet. The face-analysis stack attaches separately via
    /// [`with_face`](Self::with_face).
    pub fn load(paths: &InstantIdPaths) -> Result<Self> {
        let device = candle_gen::default_device()?;
        let root = paths.sdxl_base.as_path();

        let conditioner = SdxlConditioner::load(root, &device, DTYPE)?;
        // User LoRA/LoKr (sc-6038) folds into the dense UNet weights before the IP-Adapter K/V pairs
        // install below — SDXL-family LoRAs apply to the InstantID RealVisXL backbone just like the
        // registry SDXL path. Empty is the mmap fast path.
        let mut unet = if paths.adapters.is_empty() {
            load_instantid_unet(root, &device, DTYPE)?
        } else {
            load_instantid_unet_with_adapters(root, &device, DTYPE, &paths.adapters)?
        };
        let identitynet = load_sdxl_controlnet(&paths.identitynet, &device, DTYPE)?;

        // Face IP-Adapter: the Resampler (`image_proj.*`) + the decoupled K/V pairs (`ip_adapter.*`),
        // both from the converted bundle (loaded directly at the UNet dtype).
        let ipa = Weights::from_file(&paths.ip_adapter, &device, DTYPE).map_err(|e| {
            CandleError::Msg(format!(
                "instantid: load ip-adapter {:?} (run tools/convert_instantid.py): {e}",
                paths.ip_adapter
            ))
        })?;
        let resampler =
            Resampler::from_weights(&ipa, "image_proj", &ResamplerConfig::instantid_face())?;
        let pairs = load_ip_kv_pairs(&ipa)?;
        unet.install_ip_adapter(pairs)?;

        let vae = load_sdxl_vae(&device, DTYPE)?;
        Ok(Self {
            conditioner,
            unet,
            identitynet,
            openpose: None,
            resampler,
            vae,
            sampler: EulerAncestralSampler::sdxl(),
            alpha_schedule: AlphaSchedule::scaled_linear(
                SDXL_TRAIN_STEPS,
                SDXL_BETA_START,
                SDXL_BETA_END,
            ),
            face: None,
            pid: None,
            device,
        })
    }

    /// Attach the OpenPose ControlNet for pose mode (sc-3117) — a stock diffusers SDXL ControlNet
    /// (`xinsir/controlnet-openpose-sdxl-1.0`), loaded via the same [`load_sdxl_controlnet`] as
    /// IdentityNet. Required by [`generate_pose`](Self::generate_pose).
    pub fn with_openpose(mut self, openpose: &WeightsSource) -> Result<Self> {
        self.openpose = Some(load_sdxl_controlnet(openpose, &self.device, DTYPE)?);
        Ok(self)
    }

    /// Attach the native face-analysis stack (SCRFD detector + ArcFace embedder) so [`generate`] can
    /// take a raw reference image. `dir` holds `scrfd_10g.safetensors` + `arcface_iresnet100.safetensors`
    /// (the [`candle_gen_face`] layout). The stack loads onto this model's device.
    pub fn with_face(mut self, dir: &Path) -> Result<Self> {
        self.face = Some(candle_gen_face::load_on(dir, &self.device)?);
        Ok(self)
    }

    /// Attach the optional PiD super-resolving decoder (epic 7840, sc-8373). `pid` is the same
    /// [`PidWeights`] load-spec the registry SDXL provider consumes (`LoadSpec::pid`): a
    /// [`WeightsSource::File`] converted `sdxl` student checkpoint + a [`WeightsSource::Dir`] Gemma-2
    /// caption-encoder snapshot. InstantID composes the SDXL VAE, so it loads the **same** [`PID_BACKBONE`]
    /// (`sdxl`) tag — there is no InstantID-specific PiD checkpoint. After this, a request with
    /// `use_pid = true` decodes through the student (4× SR → 2K/4K) instead of the native VAE; without it,
    /// `use_pid` errors loudly (the engine never silently falls back). Call after [`load`](Self::load).
    pub fn with_pid(mut self, pid: &PidWeights) -> Result<Self> {
        self.pid = Some(PidEngine::from_spec(pid, PID_BACKBONE, &self.device)?);
        Ok(self)
    }

    /// Mint the per-generation PiD decoder when the request opted in (`use_pid`) and a student is
    /// loaded; `None` keeps the native VAE decode. Errors loudly if `use_pid` is set without a prior
    /// [`with_pid`](Self::with_pid) (never a silent VAE fallback — matches the registry SDXL provider's
    /// `resolve_pid_decoder` contract). InstantID runs a full denoise (no `from_ldm` early-stop), so the
    /// latent sits at the clean σ=0; the prompt is the PiD caption and the seed comes from the request.
    /// The request cancel threads into the decoder so the ~4-step SR decode stays cancellable per step.
    fn pid_decoder_for(&self, req: &InstantIdRequest) -> Result<Option<PidDecoder>> {
        // Route through the shared guarded seam (sc-11242 / F-091) so the SR decode is budgeted
        // (F-013 sc-9095) and spatially tiled (sc-10087) — a large `use_pid` decode (4× SR) otherwise
        // ran the whole-image forward and reproduced the CUDA sysmem-fallback silent hang. InstantID
        // runs a full denoise to the clean σ=0 latent (count=1, single image).
        candle_gen_pid::resolve_pid_decoder_for_fields(
            self.pid.as_ref(),
            req.use_pid,
            &req.prompt,
            1,
            req.width,
            req.height,
            &req.cancel,
            req.seed,
            "instantid",
            0.0,
        )
    }

    /// Detect + embed the largest face in `image` (the reference): bbox + 5 kps + 512-d ArcFace
    /// embedding. Requires [`with_face`](Self::with_face).
    pub fn largest_face(&self, image: &Image) -> Result<DetectedFace> {
        let face = self.face.as_ref().ok_or_else(|| {
            CandleError::Msg("instantid: face stack not attached (with_face)".into())
        })?;
        face.largest_face(image)
            .map_err(|e| CandleError::Msg(e.to_string()))
    }

    /// Full T2I: letterbox the reference to the output size (the sc-2009 kps-distortion rule), detect
    /// the largest face, then generate. Requires [`with_face`](Self::with_face).
    pub fn generate(
        &mut self,
        req: &InstantIdRequest,
        reference: &Image,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        let canvas = kps::letterbox(reference, req.width, req.height);
        let face = self.largest_face(&canvas)?;
        let kps: Vec<(f32, f32)> = face.kps.iter().map(|p| (p[0], p[1])).collect();
        self.generate_with(req, &face.embedding, &kps, on_progress)
    }

    /// **Multi-view angle generation** (sc-3117) from the canonical [`kps::VIEW_ANGLE_KPS`] pack: the
    /// reference supplies identity (its ArcFace embedding), the pack supplies the IdentityNet pose. The
    /// canvas is square (`req.width` = side; `req.height` ignored). Requires [`with_face`](Self::with_face).
    pub fn generate_angle(
        &mut self,
        req: &InstantIdRequest,
        reference: &Image,
        view_angle: &str,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        let side = req.width;
        let view = kps::view_angle_kps(view_angle, side).ok_or_else(|| {
            CandleError::Msg(format!(
                "instantid: unknown view angle {view_angle:?} (see VIEW_ANGLE_KPS)"
            ))
        })?;
        let canvas = kps::letterbox(reference, side, side);
        let face = self.largest_face(&canvas)?;
        let kps: Vec<(f32, f32)> = view.to_vec();
        let sq = InstantIdRequest {
            width: side,
            height: side,
            ..req.clone()
        };
        self.generate_with(&sq, &face.embedding, &kps, on_progress)
    }

    /// **Multi-view angle generation from caller-supplied landmarks** (sc-4425): identical square-canvas
    /// pipeline, but the 5-point kps come from the caller (`kps_norm`, normalized `0.0..=1.0`) so
    /// SceneWorks owns the angle presets. Requires [`with_face`](Self::with_face).
    pub fn generate_with_kps(
        &mut self,
        req: &InstantIdRequest,
        reference: &Image,
        kps_norm: &[(f32, f32)],
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        validate_kps(kps_norm)?;
        let side = req.width;
        let kps: Vec<(f32, f32)> = kps_norm
            .iter()
            .map(|(x, y)| (x * side as f32, y * side as f32))
            .collect();
        let canvas = kps::letterbox(reference, side, side);
        let face = self.largest_face(&canvas)?;
        let sq = InstantIdRequest {
            width: side,
            height: side,
            ..req.clone()
        };
        self.generate_with(&sq, &face.embedding, &kps, on_progress)
    }

    /// Core generate from a precomputed ArcFace `embedding` (512-d) + 5 `kps` (output-canvas pixel
    /// coords) — the face-stack-independent path (also the engine seam: `ip_adapter_scale = 0` +
    /// `controlnet_scale = 0` reduces to plain SDXL txt2img).
    pub fn generate_with(
        &mut self,
        req: &InstantIdRequest,
        embedding: &[f32],
        kps: &[(f32, f32)],
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        if req.cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }
        check_steps(req.steps)?;
        check_embedding(embedding)?;
        validate_kps(kps)?;
        let cfg_on = req.guidance > 1.0;

        // Everything that borrows `&self`, computed into owned values BEFORE the `&mut self.unet`
        // `set_ip_context` (so the disjoint-field borrows don't overlap).
        let (conditioning, pooled) = self
            .conditioner
            .encode(&req.prompt, &req.negative, cfg_on)?;
        let batch = conditioning.dim(0)?;
        let time_ids = text_time_ids(batch, &self.device, DTYPE)?;
        let face_tokens = self.face_tokens(embedding, cfg_on)?;
        let kps_image = kps::draw_kps(req.width, req.height, kps);
        let id_cond_embed =
            self.cond_embed(&self.identitynet, &kps_image, req.width, req.height, cfg_on)?;

        // Set the face IP tokens on the UNet (constant across the denoise — phase 2c/2e design).
        self.unet
            .set_ip_context(Some(&face_tokens), req.ip_adapter_scale as f64)?;

        let control_ctx = ControlContext {
            controlnet: &self.identitynet,
            cond_embed: id_cond_embed,
            scale: req.controlnet_scale as f64,
        };
        let latents = self.run_identity_denoise(
            req,
            req.width,
            req.height,
            std::slice::from_ref(&control_ctx),
            &conditioning,
            &pooled,
            &time_ids,
            &face_tokens, // the IdentityNet cross-attn conditioning = the face tokens
            on_progress,
        )?;
        on_progress(Progress::Decoding);
        // Decode the final latent: the native SDXL VAE by default, or the `sdxl` PiD student (4× SR)
        // when this generation opted in (`req.use_pid`) and `with_pid` loaded one (epic 7840, sc-8373).
        let pid_decoder = self.pid_decoder_for(req)?;
        let pid_ref = pid_decoder.as_ref().map(|d| d as &dyn LatentDecoder);
        decode_image(&self.vae, &latents, pid_ref)
    }

    /// Build the CFG-batched face tokens from a 512-d ArcFace `embedding`. **Uncond-first**: under CFG
    /// the uncond row is `Resampler(zeros)` (the zero embedding through the Resampler — the reference's
    /// `_encode_prompt_image_emb`, NOT literal zero tokens) stacked *before* the positive row.
    fn face_tokens(&self, embedding: &[f32], cfg_on: bool) -> Result<Tensor> {
        let embed = Tensor::from_vec(embedding.to_vec(), (1, 1, EMBEDDING_DIM), &self.device)?
            .to_dtype(DTYPE)?;
        let input = if cfg_on {
            let z = Tensor::zeros((1, 1, EMBEDDING_DIM), DTYPE, &self.device)?;
            Tensor::cat(&[&z, &embed], 0)? // uncond (zeros) first, then cond
        } else {
            embed
        };
        self.resampler.forward(&input) // [B, 16, 2048]
    }

    /// Preprocess a control image → the step-invariant ControlNet conditioning embedding, CFG-batched to
    /// match the UNet input (the same control on both rows; the IdentityNet conditions the uncond row
    /// too). Returns the owned `cond_embed` (so the caller can hold a `&self.controlnet` borrow + the
    /// `&mut self.unet` IP-set disjointly).
    fn cond_embed(
        &self,
        controlnet: &ControlNet,
        image: &Image,
        width: u32,
        height: u32,
        cfg_on: bool,
    ) -> Result<Tensor> {
        let c = preprocess_control_image(image, width, height, &self.device)?.to_dtype(DTYPE)?;
        let control = if cfg_on {
            Tensor::cat(&[&c, &c], 0)?
        } else {
            c
        };
        controlnet.embed_cond(&control)
    }

    /// Seed a `StdRng` and sample the prior latents for a `width × height` render (split out so both
    /// the prior and the per-step ancestral noise — drawn from a *separate* per-step stream — are
    /// reproducible). The prior stream is keyed by `seed`; the step stream by `seed + STEP_RNG_SALT`.
    fn seeded_prior_with(&self, seed: u64, width: u32, height: u32) -> Result<Tensor> {
        let mut rng = StdRng::seed_from_u64(seed);
        seeded_prior(&self.sampler, &mut rng, width, height, &self.device, DTYPE)
    }

    /// Run the InstantID dual-conditioning denoise — the bespoke ancestral default (byte-exact N1) or,
    /// when the request names a curated sampler/scheduler (epic 7114, sc-7297), the additive
    /// k-diffusion [`denoise_curated`] over the SDXL [`DiscreteModelSampling`]. BOTH carry the SAME dual
    /// conditioning: the IdentityNet (+ OpenPose) ControlNet residuals via `controls` + the face IP
    /// tokens already set on the UNet ([`UNet2DConditionModel::set_ip_context`] — the shared
    /// precondition of `denoise_ip_multi_control` and `denoise_curated`). A curated name only swaps the
    /// integrator; `euler_ancestral` (the default) and no curated knob stay on the vendored ancestral
    /// loop. `controls` is one branch for [`Self::generate_with`], two (IdentityNet + OpenPose) for pose
    /// mode; `controlnet_encoder` is the face tokens both branches cross-attend to.
    #[allow(clippy::too_many_arguments)]
    fn run_identity_denoise(
        &self,
        req: &InstantIdRequest,
        width: u32,
        height: u32,
        controls: &[ControlContext],
        conditioning: &Tensor,
        pooled: &Tensor,
        time_ids: &Tensor,
        controlnet_encoder: &Tensor,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Tensor> {
        let sampler_name = req.sampler.as_deref().unwrap_or("euler_ancestral");
        let scheduler_curated = req
            .scheduler
            .as_deref()
            .and_then(Scheduler::from_name)
            .is_some();
        // A curated solver name (other than the bespoke `euler_ancestral` default) OR a non-discrete
        // scheduler routes to the additive k-diffusion path; everything else stays byte-exact ancestral
        // (the N1 default gate). Mirrors the base candle SDXL pipeline's curated decision.
        let sampler_curated =
            Solver::from_name(sampler_name).is_some() && sampler_name != "euler_ancestral";
        if sampler_curated || scheduler_curated {
            // Curated unified-sampler path (epic 7114, sc-7297): k-diffusion VE-σ sampling over the SDXL
            // `DiscreteModelSampling`. The latents live in raw σ-space (`ε·σ_max`); `denoise_curated`
            // applies the IdentityNet residuals (`controls`) + the face IP tokens (preconditioned on the
            // UNet) each step, exactly as the ancestral loop — only the integrator differs.
            let ms = DiscreteModelSampling::sdxl(&self.alpha_schedule);
            // Native curated schedule = ComfyUI's SDXL default (`normal`); the scheduler axis overrides.
            let native = schedule_sigmas(Scheduler::Normal, &ms, req.steps);
            let sigmas =
                candle_gen::resolve_schedule(req.scheduler.as_deref(), &ms, req.steps, &native);
            let prior = seeded_sigma_prior(req.seed, width, height, sigmas[0], &self.device)?;
            denoise_curated(
                &self.unet,
                Some(sampler_name),
                &ms,
                &sigmas,
                prior,
                conditioning,
                pooled,
                time_ids,
                req.guidance as f64,
                DTYPE,
                req.seed,
                &req.cancel,
                on_progress,
                controls,
                controlnet_encoder,
            )
        } else {
            // Bespoke ancestral default (byte-exact N1): InstantID's production EulerAncestral over the
            // seeded prior, dual conditioning via `denoise_ip_multi_control` (a single branch is
            // bit-identical to the historical `denoise_ip_control`).
            let prior = self.seeded_prior_with(req.seed, width, height)?;
            let d = Denoiser {
                unet: &self.unet,
                sampler: &self.sampler,
            };
            let steps = self.sampler.timesteps(req.steps, self.sampler.max_time());
            let mut rng = StdRng::seed_from_u64(req.seed.wrapping_add(STEP_RNG_SALT));
            denoise_ip_multi_control(
                &d,
                prior,
                conditioning,
                pooled,
                time_ids,
                req.guidance as f64,
                &steps,
                &mut rng,
                &req.cancel,
                on_progress,
                controls,
                controlnet_encoder,
            )
        }
    }

    /// **Pose mode** (sc-3117): generate the character in one pose on a square canvas. The OpenPose
    /// skeleton drives the body; IdentityNet + the face IP tokens anchor the face when the head is
    /// visible. `req.width` is the side; requires [`with_face`](Self::with_face) +
    /// [`with_openpose`](Self::with_openpose).
    pub fn generate_pose(
        &mut self,
        req: &InstantIdRequest,
        reference: &Image,
        keypoints: &[BodyPoint],
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        let side = req.width;
        let canvas = kps::letterbox(reference, side, side);
        let face = self.largest_face(&canvas)?;
        // Place the reference's 5 face landmarks at the pose's head box (when the head is visible).
        let face_kps = openpose::face_box_from_keypoints(keypoints)
            .map(|(cx, cy, face_h_frac)| place_face_kps(&face, cx, cy, face_h_frac, side));
        let sq = InstantIdRequest {
            width: side,
            height: side,
            ..req.clone()
        };
        self.generate_pose_with(
            &sq,
            &face.embedding,
            face_kps.as_deref(),
            keypoints,
            on_progress,
        )
    }

    /// Core pose-mode generate (face-stack-independent): MultiControlNet over `[IdentityNet(face_kps),
    /// OpenPose(skeleton)]` + the face IP tokens. `Some` `face_kps` (head visible) drives IdentityNet +
    /// IP at the request scales; `None` (back/occluded) zeroes them and boosts OpenPose to
    /// `max(openpose_scale, 0.85)`. Requires [`with_openpose`](Self::with_openpose).
    pub fn generate_pose_with(
        &mut self,
        req: &InstantIdRequest,
        embedding: &[f32],
        face_kps: Option<&[(f32, f32)]>,
        keypoints: &[BodyPoint],
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        if req.cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }
        check_steps(req.steps)?;
        check_embedding(embedding)?;
        if let Some(kps) = face_kps {
            validate_kps(kps)?;
        }
        if self.openpose.is_none() {
            return Err(CandleError::Msg(
                "instantid: pose mode needs the OpenPose ControlNet (with_openpose)".into(),
            ));
        }
        let side = req.width;
        let cfg_on = req.guidance > 1.0;

        let (conditioning, pooled) = self
            .conditioner
            .encode(&req.prompt, &req.negative, cfg_on)?;
        let batch = conditioning.dim(0)?;
        let time_ids = text_time_ids(batch, &self.device, DTYPE)?;
        let face_tokens = self.face_tokens(embedding, cfg_on)?;
        let skeleton = openpose::draw_bodypose(side, side, keypoints, STICKWIDTH);

        // Face landmark control image + per-branch scales. No visible face ⇒ blank kps, IdentityNet + IP
        // zeroed, OpenPose boosted (the shared seed/prompt carry hair/wardrobe continuity).
        let (face_image, id_scale, op_scale, ip_scale) = match face_kps {
            Some(kps) => (
                kps::draw_kps(side, side, kps),
                req.controlnet_scale as f64,
                req.openpose_scale as f64,
                req.ip_adapter_scale as f64,
            ),
            None => (
                Image {
                    width: side,
                    height: side,
                    pixels: vec![0u8; (side as usize) * (side as usize) * 3],
                },
                0.0,
                req.openpose_scale.max(NO_FACE_OPENPOSE_FLOOR) as f64,
                0.0,
            ),
        };
        let id_cond_embed = self.cond_embed(&self.identitynet, &face_image, side, side, cfg_on)?;
        let op_cond_embed = {
            let op = self.openpose.as_ref().expect("openpose checked above");
            self.cond_embed(op, &skeleton, side, side, cfg_on)?
        };

        // No-face mode zeros the IP scale; the head-visible path uses the request scale. Set on the
        // UNet (constant across the denoise) before the shared `run_identity_denoise`.
        self.unet.set_ip_context(Some(&face_tokens), ip_scale)?;

        let op = self.openpose.as_ref().expect("openpose checked above");
        // MultiControlNet branch order matches the reference: [IdentityNet(kps), OpenPose(skeleton)].
        let controls = [
            ControlContext {
                controlnet: &self.identitynet,
                cond_embed: id_cond_embed,
                scale: id_scale,
            },
            ControlContext {
                controlnet: op,
                cond_embed: op_cond_embed,
                scale: op_scale,
            },
        ];
        let latents = self.run_identity_denoise(
            req,
            side,
            side,
            &controls,
            &conditioning,
            &pooled,
            &time_ids,
            &face_tokens, // both branches' cross-attn conditioning = the face tokens
            on_progress,
        )?;
        on_progress(Progress::Decoding);
        // Decode the final latent: the native SDXL VAE by default, or the `sdxl` PiD student (4× SR)
        // when this generation opted in (`req.use_pid`) and `with_pid` loaded one (epic 7840, sc-8373).
        let pid_decoder = self.pid_decoder_for(req)?;
        let pid_ref = pid_decoder.as_ref().map(|d| d as &dyn LatentDecoder);
        decode_image(&self.vae, &latents, pid_ref)
    }

    /// **Face-restoration pass** (sc-3380): ADetailer-style identity recovery at full-body framing.
    /// Detect the largest face in `base`, crop it with `1.9×` padding, re-render that crop through the
    /// single-control [`generate_with`] path with the reference `embedding`, then paste it back with a
    /// feathered elliptical mask. A no-op (returns `base`) when no face is found or the crop is
    /// degenerate. Requires [`with_face`](Self::with_face).
    pub fn restore_face(
        &mut self,
        req: &InstantIdRequest,
        base: &Image,
        embedding: &[f32],
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        if req.cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }
        // Detect (no embed — only the box/landmarks are used; the identity is the `embedding` param).
        let dets = {
            let face = self.face.as_ref().ok_or_else(|| {
                CandleError::Msg("instantid: face stack not attached (with_face)".into())
            })?;
            face.detect(base)
                .map_err(|e| CandleError::Msg(e.to_string()))?
        };
        let Some(f) = dets.first() else {
            return Ok(base.clone()); // no face to restore — leave the base untouched
        };

        // Crop box: a square-ish window around the face center, padded ×1.9, clamped to the image.
        let [x1, y1, x2, y2] = f.bbox;
        let (cx, cy) = ((x1 + x2) / 2.0, (y1 + y2) / 2.0);
        let half = (x2 - x1).max(y2 - y1) * FACE_RESTORE_CROP_PAD / 2.0;
        let ax = (cx - half).max(0.0) as usize;
        let ay = (cy - half).max(0.0) as usize;
        let right = (cx + half).min(base.width as f32) as usize;
        let bottom = (cy + half).min(base.height as f32) as usize;
        let (crop_w, crop_h) = (right.saturating_sub(ax), bottom.saturating_sub(ay));
        if crop_w < 16 || crop_h < 16 {
            return Ok(base.clone()); // degenerate crop — skip
        }

        // Re-place the detected face's 5 kps into the crop, scaled to the square re-render side.
        let side = req.width;
        let (sx, sy) = (side as f32 / crop_w as f32, side as f32 / crop_h as f32);
        let kps: Vec<(f32, f32)> = f
            .kps
            .iter()
            .map(|&[kx, ky]| ((kx - ax as f32) * sx, (ky - ay as f32) * sy))
            .collect();

        // Re-render the crop (IdentityNet only) imposing the reference identity, then downscale back.
        // Force the native VAE here regardless of the caller's `use_pid` (epic 7840, sc-8373): the
        // paste-back below resizes the re-render as exactly `side×side`, but a PiD decode would emit it
        // at 4× — so PiD-decoding the crop would corrupt the resize. The base image keeps whatever decode
        // the top-level generate used.
        let restore_req = InstantIdRequest {
            width: side,
            height: side,
            use_pid: false,
            ..req.clone()
        };
        let restored = self.generate_with(&restore_req, embedding, &kps, on_progress)?;
        let small_f = resize_lanczos_u8(
            &restored.pixels,
            side as usize,
            side as usize,
            crop_h,
            crop_w,
        );
        let small: Vec<u8> = small_f.iter().map(|&v| v as u8).collect();

        // Feathered elliptical paste-back onto a copy of the base.
        let alpha = restore::feather_mask(crop_w, crop_h);
        let mut out = base.clone();
        restore::paste_alpha(&mut out, &small, crop_w, crop_h, ax, ay, &alpha);
        Ok(out)
    }
}

/// Re-place a detected face's 5-point landmarks at a pose's head box (the vendored
/// `instantid_adapter.py::_run_pose` / `_normalized_kps`): normalize the reference kps to its detected
/// bbox, then scale/translate to the head box `(cx, cy)` (normalized canvas coords) at height
/// `face_h_frac` of the canvas, preserving the face aspect. Returns canvas-pixel coords. A free fn (no
/// `&self`) so it composes cleanly under the `&mut self` generate.
fn place_face_kps(
    face: &DetectedFace,
    cx: f64,
    cy: f64,
    face_h_frac: f64,
    side: u32,
) -> Vec<(f32, f32)> {
    let [x1, y1, x2, y2] = face.bbox;
    let (ox, oy) = (x1 as f64, y1 as f64);
    let sw = (x2 - x1).max(1.0) as f64;
    let sh = (y2 - y1).max(1.0) as f64;
    let aspect = sw / sh;
    let canvas = side as f64;
    let face_h = canvas * face_h_frac;
    let face_w = face_h * aspect;
    face.kps
        .iter()
        .map(|&[kx, ky]| {
            let nx = (kx as f64 - ox) / sw;
            let ny = (ky as f64 - oy) / sh;
            let px = cx * canvas + (nx - 0.5) * face_w;
            let py = cy * canvas + (ny - 0.5) * face_h;
            (px as f32, py as f32)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_kps_rejects_short_slices() {
        for len in 0..FACE_KP_COUNT {
            let kps = vec![(0.0f32, 0.0f32); len];
            let err = validate_kps(&kps).unwrap_err().to_string();
            assert!(
                err.contains("need 5 face keypoints"),
                "len {len} got: {err}"
            );
        }
        assert!(validate_kps(&[(0.0, 0.0); FACE_KP_COUNT]).is_ok());
        assert!(validate_kps(&[(0.0, 0.0); FACE_KP_COUNT + 1]).is_ok());
    }

    #[test]
    fn check_embedding_enforces_512() {
        assert!(check_embedding(&vec![0.0; 512]).is_ok());
        assert!(check_embedding(&vec![0.0; 511]).is_err());
        assert!(check_embedding(&[]).is_err());
    }

    /// `steps == 0` is rejected with a fast, actionable error (never decoded as undenoised noise);
    /// a valid step count passes (sc-9016, F-032).
    #[test]
    fn check_steps_rejects_zero() {
        let err = check_steps(0).unwrap_err().to_string();
        assert!(err.contains("steps must be >= 1"), "got: {err}");
        assert!(check_steps(1).is_ok());
        assert!(check_steps(30).is_ok());
    }

    /// The worker-driven pose entry (`generate_pose`/`generate_pose_with`, sc-3117) bypasses
    /// `generate_with` and runs its own `run_identity_denoise`, so it must guard `steps == 0` itself.
    /// It applies `check_steps(req.steps)` right after the cancel check — assert that call rejects a
    /// zero-step pose request before any denoise/VAE-decode of undenoised noise (sc-9016, F-032).
    #[test]
    fn pose_entry_rejects_zero_steps() {
        let req = InstantIdRequest {
            steps: 0,
            ..Default::default()
        };
        // The exact guard `generate_pose_with` now runs on its request.
        let err = check_steps(req.steps).unwrap_err().to_string();
        assert!(err.contains("steps must be >= 1"), "got: {err}");
        // A worker-typical pose step count passes the same guard.
        let ok = InstantIdRequest {
            steps: 30,
            ..Default::default()
        };
        assert!(check_steps(ok.steps).is_ok());
    }

    /// `place_face_kps` centers the face landmarks at the head box and scales by the face-height frac:
    /// the bbox center maps to `(cx, cy)·side`, and the landmark spread scales with `face_h_frac`.
    #[test]
    fn place_face_kps_centers_and_scales() {
        // A face whose bbox is [10,10,30,30] (20×20, square) with kps at the bbox center + corners.
        let face = DetectedFace {
            bbox: [10.0, 10.0, 30.0, 30.0],
            kps: [
                [20.0, 20.0],
                [10.0, 10.0],
                [30.0, 30.0],
                [20.0, 10.0],
                [10.0, 30.0],
            ],
            det_score: 1.0,
            embedding: Vec::new(),
        };
        let side = 100u32;
        // Head box centered at (0.5, 0.5) of the canvas, face height 0.5 of the canvas.
        let placed = place_face_kps(&face, 0.5, 0.5, 0.5, side);
        // The bbox-center landmark (20,20) → canvas center (50,50).
        assert!((placed[0].0 - 50.0).abs() < 1e-3 && (placed[0].1 - 50.0).abs() < 1e-3);
        // aspect = 1, face_h = face_w = 50. The corner (10,10) is at norm (-0.5,-0.5) of the bbox →
        // 50 + (-0.5)*50 = 25.
        assert!((placed[1].0 - 25.0).abs() < 1e-3 && (placed[1].1 - 25.0).abs() < 1e-3);
        // The corner (30,30) is at norm (+0.5,+0.5) → 75.
        assert!((placed[2].0 - 75.0).abs() < 1e-3 && (placed[2].1 - 75.0).abs() < 1e-3);
    }
}
