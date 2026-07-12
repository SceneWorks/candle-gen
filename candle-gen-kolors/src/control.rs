//! Kolors **ControlNet (strict pose)** provider (sc-5489, epic 5480) — reference-pose conditioning on
//! Kolors, the candle (Windows/CUDA) sibling of the `mlx-gen-kolors` ControlNet path. It reuses the
//! same vendored SDXL stack the Kolors IP-Adapter slice ([`crate::ip_provider`]) stands on, swapping the
//! IP-Adapter overlay for a diffusers **`ControlNetModel`**:
//!
//! - a rendered OpenPose skeleton (the worker draws it at the request size) is normalized to `[0,1]`
//!   and embedded ONCE by the ControlNet's `controlnet_cond_embedding` conv stack
//!   ([`ControlNet::embed_cond`], step-invariant);
//! - each denoise step the Kolors ControlNet — an SDXL-family encoder copy ([`ControlNet`], built from
//!   [`ControlNetConfig::kolors`]) — emits the per-down-block + mid [`ControlResiduals`] (scaled by
//!   `control_scale`), which the vendored SDXL [`UNet2DConditionModel::forward_instantid`] adds into its
//!   skip connections + mid output (the same residual seam InstantID rides on);
//! - the text path is **ChatGLM3-6B** (the Kolors encoder); the denoise runs the Kolors **leading-Euler**
//!   sampler ([`KolorsEulerSampler`]) — NOT the SDXL EulerAncestral — so the numerics match Kolors txt2img.
//!
//! The Kolors `ControlNetModel` carries its **own** `encoder_hid_proj` (4096→2048), trained separately
//! from the UNet's — so the raw ChatGLM3 context is projected **twice**: once by the UNet's
//! `encoder_hid_proj` (for the base cross-attentions) and once by the ControlNet's (for the control
//! branch's cross-attentions). No IP-Adapter K/V is installed, so [`UNet2DConditionModel::forward_instantid`]
//! runs as a plain SDXL UNet + control residuals (the decoupled-attn branch is `None`-guarded).
//!
//! Like the SDXL/InstantID/Kolors-IP lanes, **CFG is uncond-first** (`[negative, prompt]`, the Kolors
//! txt2img convention), and the control branch runs on **both** CFG passes (the diffusers
//! `guess_mode=False` rule). The provider is a plain struct driven **directly** by the worker (a bespoke
//! pose stream, like [`crate::ip_provider::IpAdapterKolors`]), not a gen-core-registered generator — the
//! registered `kolors` descriptor stays txt2img-only.

use std::path::{Path, PathBuf};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::{self as nn, Linear, Module, VarBuilder};
use candle_gen::gen_core::runtime::CancelFlag;
use candle_gen::gen_core::{Image, PidWeights, Progress};
use candle_gen::{CandleError, Result};
use candle_gen_pid::{PidDecoder, PidEngine};
use candle_transformers::models::stable_diffusion::vae::AutoEncoderKL;

use candle_gen_sdxl::{
    denoise_curated, preprocess_control_image, sdxl_unet_config, ControlContext, ControlNet,
    ControlNetConfig, UNet2DConditionModel,
};

use crate::chatglm3::ChatGlmModel;
use crate::common::{self, CuratedSetup};
use crate::config::ChatGlmConfig;
use crate::pipeline::{curated_route, sdxl_vae_config};
use crate::sampler::KolorsEulerSampler;
use crate::tokenizer::KolorsTokenizer;

/// The control compute dtype. Kolors runs the whole stack at **f32** (the candle port recipe — a single
/// matmul dtype), so the vendored UNet, the ControlNet, and the SDXL VAE all load at f32 here too.
const DTYPE: DType = DType::F32;

/// Kolors `add_embedding` dims (the Kolors `unet/config.json` AND the `Kolors-ControlNet-*/config.json`):
/// `addition_time_embed_dim = 256`, `projection_class_embeddings_input_dim = 5632` (pooled 4096 + 6·256)
/// — vs SDXL's 2816. The vendored UNet needs the `add_embedding` head the plain `forward` omits; the
/// ControlNet builds its matching head from [`ControlNetConfig::kolors`].
const ADDITION_TIME_EMBED_DIM: usize = 256;
const PROJECTION_INPUT_DIM: usize = 5632;

/// ChatGLM3 context width (the `encoder_hid_proj` in-features, on BOTH the UNet and the ControlNet).
const CONTEXT_DIM: usize = 4096;
/// The SDXL/Kolors UNet cross-attention width (the `encoder_hid_proj` out-features).
const CROSS_ATTENTION_DIM: usize = 2048;

/// Default ControlNet conditioning scale (the strict-pose tier — parity with the Qwen control slice and
/// the mlx Kolors ControlNet path).
pub const DEFAULT_CONTROL_SCALE: f32 = 1.0;

/// Paths to the Kolors ControlNet checkpoints.
pub struct KolorsControlPaths {
    /// The `Kwai-Kolors/Kolors-diffusers` snapshot dir (`tokenizer/`, `text_encoder/` ChatGLM3-6B,
    /// `unet/` SDXL-family UNet, `vae/` SDXL VAE).
    pub kolors_base: PathBuf,
    /// The `Kwai-Kolors/Kolors-ControlNet-Pose` checkpoint — a single `.safetensors` file or a dir
    /// (`diffusion_pytorch_model.safetensors`).
    pub controlnet: PathBuf,
}

/// One Kolors ControlNet (strict-pose) generation request.
#[derive(Clone)]
pub struct KolorsControlRequest {
    pub prompt: String,
    pub negative: String,
    pub width: u32,
    pub height: u32,
    pub steps: usize,
    /// Classifier-free guidance scale.
    pub guidance: f32,
    /// ControlNet conditioning scale on the pose residuals.
    pub control_scale: f32,
    /// Curated unified-sampler selection (epic 7114, sc-7297). `None` (or `euler_discrete`) keeps the
    /// bespoke leading-Euler default byte-exact (N1); a curated
    /// [`Solver`](candle_gen::gen_core::sampling::Solver) name routes the pose-control
    /// denoise through [`denoise_curated`] over the Kolors [`DiscreteModelSampling`].
    pub sampler: Option<String>,
    /// Curated σ-schedule selection (epic 7114). `None` ⇒ the native leading schedule; a [`Scheduler`]
    /// name re-shapes σ. A non-default scheduler alone also engages the curated path.
    pub scheduler: Option<String>,
    pub seed: u64,
    /// Opt into the PiD super-resolving decoder (epic 7840, sc-8044): when `true` **and** the model was
    /// loaded with [`with_pid`](KolorsControl::with_pid), the final latent is decoded by the `sdxl` PiD
    /// student (4× SR → 2K/4K) instead of the native SDXL VAE (Kolors composes the SDXL VAE). `false`
    /// (default) keeps the VAE decode.
    pub use_pid: bool,
    /// Cooperative cancellation, checked before each denoise step (the engine contract).
    pub cancel: CancelFlag,
}

impl Default for KolorsControlRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            negative: String::new(),
            width: 1024,
            height: 1024,
            steps: 50,
            guidance: 5.0,
            control_scale: DEFAULT_CONTROL_SCALE,
            sampler: None,
            scheduler: None,
            seed: 0,
            use_pid: false,
            cancel: CancelFlag::default(),
        }
    }
}

/// mmap an f32 [`VarBuilder`] over every `.safetensors` in `dir` (the ChatGLM3 encoder + UNet ship
/// sharded or single-file) — mirrors the txt2img pipeline / IP-Adapter loaders.
fn f32_vb(dir: &Path, device: &Device) -> Result<VarBuilder<'static>> {
    candle_gen::load_sorted_mmap(dir, DTYPE, device, "kolors-control")
}

/// Resolve the ControlNet weight **file** from a dir-or-file path (the diffusers `ControlNetModel`
/// layout: a single `diffusion_pytorch_model(.fp16).safetensors`).
fn resolve_controlnet_file(path: &Path) -> Result<PathBuf> {
    if path.is_file() {
        return Ok(path.to_path_buf());
    }
    for name in [
        "diffusion_pytorch_model.safetensors",
        "diffusion_pytorch_model.fp16.safetensors",
    ] {
        let p = path.join(name);
        if p.is_file() {
            return Ok(p);
        }
    }
    Err(CandleError::Msg(format!(
        "kolors-control: ControlNet weights not found under {} (expected a \
         diffusion_pytorch_model.safetensors or a direct .safetensors file)",
        path.display()
    )))
}

/// Loaded Kolors ControlNet model: the ChatGLM3 tokenizer + encoder, the UNet's `encoder_hid_proj`
/// context projection, the vendored SDXL UNet (NO IP installed — plain SDXL + control residuals), the
/// Kolors ControlNet + its OWN `encoder_hid_proj`, and the f32 SDXL VAE.
pub struct KolorsControl {
    tokenizer: KolorsTokenizer,
    chatglm: ChatGlmModel,
    /// The UNet's ChatGLM3 context projection (4096 → 2048), applied before the base cross-attentions
    /// (the vendored UNet has no `encoder_hid_proj`, unlike [`crate::unet::KolorsUNet`]).
    encoder_hid_proj: Linear,
    unet: UNet2DConditionModel,
    /// The ControlNet's OWN ChatGLM3 context projection (4096 → 2048), trained separately from the
    /// UNet's, applied before the control branch's cross-attentions.
    cn_encoder_hid_proj: Linear,
    controlnet: ControlNet,
    vae: AutoEncoderKL,
    /// Optional PiD super-resolving decoder (epic 7840, sc-8044), attached via [`with_pid`](Self::with_pid).
    /// Kolors composes the SDXL VAE, so it loads the `sdxl` student (same tag as the base Kolors provider).
    pid: Option<PidEngine>,
    device: Device,
}

impl KolorsControl {
    /// Load the Kolors backbone (ChatGLM3 + SDXL-family UNet into the vendored stack + SDXL VAE) + the
    /// Kolors `ControlNetModel` (encoder copy + its own `encoder_hid_proj`). No IP-Adapter K/V is
    /// installed — the control branch is the only conditioning overlay.
    pub fn load(paths: &KolorsControlPaths) -> Result<Self> {
        let device = candle_gen::default_device()?;
        let base = paths.kolors_base.as_path();

        let tokenizer = KolorsTokenizer::from_dir(base.join("tokenizer"))?;
        let chatglm = ChatGlmModel::new(
            ChatGlmConfig::chatglm3_6b(),
            f32_vb(&base.join("text_encoder"), &device)?,
        )?;

        // Vendored SDXL UNet from the Kolors `unet/` weights + the 5632 `add_embedding` head + the UNet's
        // `encoder_hid_proj` (all in the same checkpoint). NOTE: no `install_ip_adapter` — `forward_instantid`
        // then runs as a plain SDXL UNet (its decoupled-attn branch is `None`-guarded) + control residuals.
        let vs = f32_vb(&base.join("unet"), &device)?;
        let unet = UNet2DConditionModel::new(vs.clone(), 4, 4, false, sdxl_unet_config())?
            .with_add_embedding(vs.clone(), ADDITION_TIME_EMBED_DIM, PROJECTION_INPUT_DIM)?;
        let encoder_hid_proj =
            nn::linear(CONTEXT_DIM, CROSS_ATTENTION_DIM, vs.pp("encoder_hid_proj"))?;

        // Kolors ControlNet (a diffusers SDXL-family `ControlNetModel`) + its OWN `encoder_hid_proj`.
        let cn_file = resolve_controlnet_file(&paths.controlnet)?;
        let cn_vb = candle_gen::mmap_var_builder(&[cn_file], DTYPE, &device)?;
        let cn_encoder_hid_proj = nn::linear(
            CONTEXT_DIM,
            CROSS_ATTENTION_DIM,
            cn_vb.pp("encoder_hid_proj"),
        )?;
        let controlnet = ControlNet::new(cn_vb, &ControlNetConfig::kolors())?;

        let vae = AutoEncoderKL::new(f32_vb(&base.join("vae"), &device)?, 3, 3, sdxl_vae_config())?;

        Ok(Self {
            tokenizer,
            chatglm,
            encoder_hid_proj,
            unet,
            cn_encoder_hid_proj,
            controlnet,
            vae,
            pid: None,
            device,
        })
    }

    /// Attach the optional PiD super-resolving decoder (epic 7840, sc-8044). Same [`PidWeights`] load-spec
    /// as the registry Kolors/SDXL provider; Kolors composes the SDXL VAE, so it loads the `sdxl` student.
    /// A `use_pid = true` request then decodes through it (4× SR) instead of the native VAE; without it,
    /// `use_pid` errors loudly. Call after [`load`](Self::load).
    pub fn with_pid(mut self, pid: &PidWeights) -> Result<Self> {
        // Kolors reuses the SDXL VAE latent space, so the PiD backbone tag is `sdxl` (the base Kolors
        // provider's `pipeline::PID_BACKBONE`).
        self.pid = Some(PidEngine::from_spec(pid, "sdxl", &self.device)?);
        Ok(self)
    }

    /// Mint the per-generation PiD decoder when the request opted in (`use_pid`) and a student is loaded;
    /// `None` keeps the native VAE decode. Errors loudly if `use_pid` is set without a prior
    /// [`with_pid`](Self::with_pid). A clean-latent (σ=0) decoder bound to the prompt + seed; the request
    /// cancel threads in for a cancellable SR decode.
    fn pid_decoder_for(&self, req: &KolorsControlRequest) -> Result<Option<PidDecoder>> {
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
            "kolors control",
            0.0,
        )
    }

    /// Strict-pose T2I: condition the Kolors generation on `skeleton` (a rendered OpenPose image at the
    /// request size) via the Kolors ControlNet, denoising with the Kolors leading-Euler sampler. The
    /// worker renders the skeleton; this embeds it once, then runs the control denoise (the control
    /// branch runs on both CFG passes).
    pub fn generate(
        &self,
        req: &KolorsControlRequest,
        skeleton: &Image,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        if req.cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }
        common::reject_zero_steps("kolors control", req.steps)?;
        let use_guide = req.guidance > 1.0;

        // CFG batch is [neg, pos] = uncond-first (the Kolors txt2img convention); without guidance only
        // the positive branch is built. Shared CFG-concat (sc-9001); the ChatGLM3 encode stays local.
        let (context, pooled, batch) =
            common::cfg_batch_context(&req.prompt, &req.negative, use_guide, |p| self.encode(p))?;

        // Two SEPARATE ChatGLM3 → cross-attention projections: the UNet's `encoder_hid_proj` feeds the
        // base cross-attentions; the ControlNet's own (separately-trained) `encoder_hid_proj` feeds the
        // control branch's. Both project the raw 4096-wide context to 2048 up front. (This dual
        // projection is the control lane's genuine drift — NOT shared.)
        let projected = self.encoder_hid_proj.forward(&context)?;
        let cn_context = self.cn_encoder_hid_proj.forward(&context)?;
        let time_ids = common::build_time_ids(&self.device, batch, req.height, req.width)?;

        // The pose skeleton → `[batch, 3, H, W]` in `[0,1]` (the diffusers control-image normalization,
        // NOT a VAE's `[-1,1]`), CFG-batched (same control on both rows). `embed_cond` is step-invariant,
        // so the conditioning embedding is computed ONCE here.
        let control = preprocess_control_image(skeleton, req.width, req.height, &self.device)?
            .to_dtype(DTYPE)?;
        let control = if use_guide {
            Tensor::cat(&[&control, &control], 0)?
        } else {
            control
        };
        let cond_embed = self.controlnet.embed_cond(&control)?;
        let control_scale = req.control_scale as f64;
        let (lat_h, lat_w) = ((req.height / 8) as usize, (req.width / 8) as usize);

        // Curated unified-sampler path (epic 7114, sc-7297): a curated solver name (≠ the native
        // `euler_discrete`) OR a curated scheduler routes the pose-control denoise through the
        // additive k-diffusion `denoise_curated`, which threads the Kolors ControlNet residuals through
        // the curated solver. The native leading-Euler default stays byte-exact (N1). The decision is
        // the shared [`curated_route`] (sc-8984) so the three Kolors entry points can't drift.
        let curated = curated_route(req.sampler.as_deref(), req.scheduler.as_deref());

        let latents = if let Some(sampler_name) = curated {
            // k-diffusion VE-σ sampling over the Kolors `DiscreteModelSampling`. A scheduler-only curated
            // run keeps `euler_discrete` (a non-solver alias) ⇒ the driver's euler fallback (N3). Shared
            // curated-σ setup (sc-9001) from the SAME launch-portable seeded noise as the native path.
            let noise = common::initial_noise(&self.device, req.seed, lat_h, lat_w)?;
            let setup = CuratedSetup::new(req.scheduler.as_deref(), req.steps, &noise)?;
            // The control lane's genuine drift: the ControlNet residual context threaded into the solver.
            let control_ctx = ControlContext {
                controlnet: &self.controlnet,
                cond_embed,
                scale: control_scale,
            };
            // No IP installed on this UNet ⇒ `forward_instantid`'s decoupled branch is inert; the
            // ControlNet cross-attends to its OWN text projection (`cn_context`), the UNet to `projected`.
            denoise_curated(
                &self.unet,
                Some(sampler_name),
                &setup.model_sampling,
                &setup.sigmas,
                setup.prior,
                &projected,
                &pooled,
                &time_ids,
                req.guidance as f64,
                DTYPE,
                req.seed,
                &req.cancel,
                on_progress,
                std::slice::from_ref(&control_ctx),
                &cn_context,
            )?
        } else {
            let sampler = KolorsEulerSampler::new(req.steps).map_err(CandleError::Msg)?;
            let noise = common::initial_noise(&self.device, req.seed, lat_h, lat_w)?;
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
                let t = sampler.timestep(i) as f64;
                // Control residuals from the Kolors ControlNet (its own context projection), scaled by
                // `control_scale`, then added into the UNet skip + mid via `forward_instantid`.
                let res = self.controlnet.forward(
                    &model_in,
                    &cond_embed,
                    t,
                    &cn_context,
                    &pooled,
                    &time_ids,
                    control_scale,
                )?;
                let eps = self.unet.forward_instantid(
                    &model_in,
                    t,
                    &projected,
                    &pooled,
                    &time_ids,
                    Some(res.down.as_slice()),
                    Some(&res.mid),
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
        // Decode the final latent: native SDXL VAE by default, or the `sdxl` PiD student (4× SR) when this
        // generation opted in (`req.use_pid`) and `with_pid` loaded one (epic 7840, sc-8044). Kolors
        // composes the SDXL VAE, so it shares the `sdxl` student with the base Kolors provider.
        let pid_decoder = self.pid_decoder_for(req)?;
        common::decode(&self.vae, pid_decoder.as_ref(), &latents)
    }

    /// Encode one prompt → `(context [1, 256, 4096], pooled [1, 4096])` via the ChatGLM3 encoder. Stays
    /// local (borrows `&self.tokenizer` / `&self.chatglm`); passed as a closure to the shared
    /// [`common::cfg_batch_context`], so the ChatGLM3 plumbing is per-site and only the CFG convention
    /// is shared.
    fn encode(&self, prompt: &str) -> Result<(Tensor, Tensor)> {
        let tokens = self.tokenizer.encode(prompt)?;
        Ok(self.chatglm.encode_prompt(&tokens)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The request defaults match the Kolors production knobs (1024², 50 steps, CFG 5.0, control 1.0).
    #[test]
    fn request_defaults() {
        let r = KolorsControlRequest::default();
        assert_eq!((r.width, r.height), (1024, 1024));
        assert_eq!(r.steps, 50);
        assert_eq!(r.guidance, 5.0);
        assert_eq!(r.control_scale, DEFAULT_CONTROL_SCALE);
        assert_eq!(DEFAULT_CONTROL_SCALE, 1.0);
        // The curated knobs default to None ⇒ the bespoke leading-Euler path (N1 byte-exact).
        assert!(r.sampler.is_none() && r.scheduler.is_none());
        assert!(!r.cancel.is_cancelled());
    }

    /// The Kolors `add_embedding` projection input is 5632 (pooled 4096 + 6·256) — vs SDXL's 2816 —
    /// shared by the vendored UNet AND the ControlNet's matching head (`ControlNetConfig::kolors`).
    #[test]
    fn kolors_add_embedding_dims() {
        assert_eq!(ADDITION_TIME_EMBED_DIM, 256);
        assert_eq!(PROJECTION_INPUT_DIM, 4096 + 6 * 256);
        assert_eq!(CONTEXT_DIM, 4096);
        assert_eq!(CROSS_ATTENTION_DIM, 2048);
        assert_eq!(
            ControlNetConfig::kolors().projection_class_embeddings_input_dim,
            PROJECTION_INPUT_DIM
        );
    }

    /// `resolve_controlnet_file`: a directory resolves `diffusion_pytorch_model.safetensors`; a direct
    /// file is used as-is; a missing dir errors loudly.
    #[test]
    fn controlnet_file_resolution() {
        let dir = std::env::temp_dir().join(format!("candle_kolors_cn_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        assert!(resolve_controlnet_file(&dir).is_err());
        let f = dir.join("diffusion_pytorch_model.safetensors");
        std::fs::write(&f, b"x").unwrap();
        assert_eq!(resolve_controlnet_file(&dir).unwrap(), f);
        assert_eq!(resolve_controlnet_file(&f).unwrap(), f);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
