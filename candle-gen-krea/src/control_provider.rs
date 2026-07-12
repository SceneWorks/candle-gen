//! Krea 2 Turbo **pose-ControlNet inference** provider (sc-8464, epic 8459) — candle (Windows/CUDA).
//!
//! The deployable sibling of the sc-8460 spike harness (`examples/krea-control-infer.rs`): loads the
//! frozen Krea 2 Turbo base (through the composable [`KreaTrainDit`] — the same forward the branch
//! trains against) plus a trained [`ControlBranch`](crate::control::ControlBranch) overlay, and renders
//! the standard 8-step CFG-free Turbo denoise conditioned on a rendered OpenPose skeleton.
//!
//! **How it conditions:** the pose skeleton is VAE-encoded (Qwen-Image VAE) into a control latent, then
//! [`forward_with_control`](crate::control::forward_with_control) — a drop-in for the base
//! `dit.forward` — adds the branch residual into the frozen main stream after each of the first N
//! single-stream blocks, scaled by `control_scale` and RMS-clamped at τ (the S0 recipe: τ = 0.15,
//! applied identically train/infer). `control_scale = 0` is engine-proven **byte-identical** to the
//! un-branched base generation at the same seed (the spike's identity contract).
//!
//! Bespoke provider (NOT gen-core-registered), worker-invoked by name — the candle pattern for
//! conditioned surfaces (mirrors [`crate::control_train`]'s trainer and the FLUX.2 control provider).
//! Krea 2 Turbo is CFG-free + distilled few-step: a single guidance-inert forward per step, no
//! negative pass. The base loads bf16 dense (the 12B + 3.3B branch + TE + VAE fit a 96 GB card); v1 is
//! bf16-only (no quant tier — the studio-trained overlays are bf16). `generate` takes `&self` so one
//! load serves many poses; the residual clamp is a fixed recipe constant set at load, not a knob.

use std::path::PathBuf;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::runtime::CancelFlag;
use candle_gen::gen_core::sampling::TimestepConvention;
use candle_gen::gen_core::{Image, Progress};
use candle_gen::train::flow_match::component_vb;
use candle_gen::{CandleError, Result};
use candle_gen_qwen_image::vae::{QwenVae, QwenVaeEncoder};
use rand::{rngs::StdRng, SeedableRng};

use crate::config::Krea2Config;
use crate::control::{forward_with_control, ControlBranch, DEFAULT_RESIDUAL_CLAMP};
use crate::loader::Weights;
use crate::pipeline::to_image;
use crate::pipeline::MAX_TEXT_TOKENS;
use crate::text_encoder::{KreaTeConfig, KreaTextEncoder};
use crate::tokenizer::KreaTokenizer;
use crate::train_dit::KreaTrainDit;
use crate::{load_vae, turbo_sigmas, TURBO_STEPS};

/// Qwen-Image VAE 8× spatial compression (latent side = pixels / 8).
const SPATIAL_SCALE: u32 = 8;
/// Latent channel count (Qwen-Image VAE).
const LATENT_CHANNELS: usize = 16;
/// Width/height must be a multiple of this (VAE 8× × 2×2 patchify), matching the base txt2img guard.
const SIZE_MULTIPLE: u32 = 16;

/// Default `control_scale` for the distilled CFG-free Turbo base. The S0 spike found the usable band
/// ~0.5–0.75 (widening to ~0.7–0.9 with more data); ship a comfortable mid default and hard-cap the
/// exposed range ≤ 0.85 (over-drive haloes to halftone above that). The worker applies the cap.
pub const DEFAULT_CONTROL_SCALE: f32 = 0.6;

/// Paths to the Krea 2 control checkpoints: the Krea 2 Turbo diffusers snapshot dir (`text_encoder/`,
/// `transformer/`, `vae/`, `tokenizer/`) + the trained control-branch overlay (a single `.safetensors`).
pub struct Krea2ControlPaths {
    /// Krea 2 Turbo diffusers snapshot dir (the deployed base the overlay applies on).
    pub root: PathBuf,
    /// The trained control-branch overlay checkpoint (`.safetensors`, e.g. `control_step5000.safetensors`).
    pub control: PathBuf,
}

/// One Krea 2 strict-pose control request. Krea 2 Turbo is CFG-free (no guidance / negative pass) —
/// the only conditioning knob beyond the prompt is `control_scale`.
#[derive(Clone)]
pub struct Krea2ControlRequest {
    pub prompt: String,
    pub width: u32,
    pub height: u32,
    pub steps: usize,
    /// How strongly the control branch locks the base (S0 usable ~0.5–0.85). `0.0` ⇒ base passthrough
    /// (byte-identical to un-branched generation at the same seed).
    pub control_scale: f32,
    pub seed: u64,
    /// Cooperative cancellation, checked before each denoise step (the engine contract).
    pub cancel: CancelFlag,
}

impl Default for Krea2ControlRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            width: 1024,
            height: 1024,
            steps: TURBO_STEPS,
            control_scale: DEFAULT_CONTROL_SCALE,
            seed: 0,
            cancel: CancelFlag::default(),
        }
    }
}

/// A loaded Krea 2 control model: the Qwen3-VL text encoder + the frozen composable Turbo DiT + the
/// trained control branch + the Qwen-Image VAE (decode) with its encoder (control-image encode). All
/// bf16/f32 dense — one load serves many poses.
pub struct Krea2Control {
    device: Device,
    tokenizer: KreaTokenizer,
    te: KreaTextEncoder,
    dit: KreaTrainDit,
    branch: ControlBranch,
    vae: QwenVae,
    vae_encoder: QwenVaeEncoder,
}

impl Krea2Control {
    /// Load the frozen Turbo base (bf16 composable DiT + f32 TE), the trained control-branch overlay
    /// (frozen, RMS-clamped at the recipe τ), and the Qwen-Image VAE with its encoder. The TE/DiT
    /// weight readers are dropped after construction; the model holds only the built components.
    pub fn load(paths: &Krea2ControlPaths) -> Result<Self> {
        let device = candle_gen::default_device()?;
        let cfg = Krea2Config::from_snapshot(&paths.root)?;

        // Text encoder (f32, exactly the pipeline's) — Qwen3-VL language tower.
        let tokenizer = KreaTokenizer::from_snapshot(&paths.root, &device)?;
        let te_cfg = KreaTeConfig::from_snapshot(&paths.root)?;
        let te_w = Weights::from_dir(&paths.root.join("text_encoder"), &device, DType::F32)?;
        let te = KreaTextEncoder::load(&te_w, "language_model", &te_cfg, MAX_TEXT_TOKENS)?;
        drop(te_w);

        // Frozen base DiT (bf16, composable — the train-time forward the branch was trained against).
        let dit_w = Weights::from_dir(&paths.root.join("transformer"), &device, DType::BF16)?;
        let dit = KreaTrainDit::load(&dit_w, &cfg)?;
        drop(dit_w);

        // Trained control branch. Freeze (detach weight reads so the sampler builds no autograd graph)
        // and set the fixed recipe residual clamp — identical to train time (S0: τ = 0.15).
        let mut branch = ControlBranch::from_checkpoint(&paths.control, &cfg, &device)?;
        branch.freeze();
        branch.set_residual_clamp(Some(DEFAULT_RESIDUAL_CLAMP));

        // VAE decode (final latent → pixels) + encoder (pose skeleton → control latent).
        let vae = load_vae(&paths.root, &device)?;
        let vae_encoder = QwenVaeEncoder::new(component_vb(
            &paths.root,
            "vae",
            &device,
            DType::F32,
            "krea control infer",
        )?)?;

        Ok(Self {
            device,
            tokenizer,
            te,
            dit,
            branch,
            vae,
            vae_encoder,
        })
    }

    /// Generate one strict-pose-conditioned image from a rendered OpenPose skeleton. The `control_image`
    /// must already be at the request's `width`×`height` — the worker driver renders the skeleton
    /// (square-canonical, the same `openpose_skeleton` renderer training used) at exactly the provider's
    /// output dims, so no resize happens here.
    pub fn generate(
        &self,
        req: &Krea2ControlRequest,
        control_image: &Image,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        if req.cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }
        validate_request(req)?;

        // Prompt embeds + control latent are seed-independent: encode once.
        let context = self
            .te
            .forward(&self.tokenizer.encode_prompt(&req.prompt, MAX_TEXT_TOKENS)?)?;
        let ctrl_nchw = control_image_to_nchw(control_image, req.width, req.height, &self.device)?;
        let ctrl_latent = self.vae_encoder.encode(&ctrl_nchw)?;
        let scale = req.control_scale as f64;

        // Seeded initial noise — the pipeline's CPU-RNG discipline (sc-3673).
        let (lat_h, lat_w) = (
            (req.height / SPATIAL_SCALE) as usize,
            (req.width / SPATIAL_SCALE) as usize,
        );
        let mut rng = StdRng::seed_from_u64(req.seed);
        let noise = candle_gen::seeded_normal_vec(&mut rng, LATENT_CHANNELS * lat_h * lat_w);
        let noise = Tensor::from_vec(noise, (1, LATENT_CHANNELS, lat_h, lat_w), &Device::Cpu)?
            .to_device(&self.device)?;

        // 8-step CFG-free Turbo denoise (raw sigma timestep, Euler `x + v·Δσ`). The control forward is a
        // drop-in for `dit.forward`; `scale == 0` short-circuits to the base forward inside it.
        let sigmas = turbo_sigmas(req.steps);
        let latent = candle_gen::run_flow_sampler(
            None,
            TimestepConvention::Sigma,
            &sigmas,
            noise,
            req.seed,
            &req.cancel,
            on_progress,
            |x, timestep| -> Result<Tensor> {
                let t = Tensor::from_vec(vec![timestep], (1,), &self.device)?;
                let v = forward_with_control(
                    &self.dit,
                    &self.branch,
                    x,
                    &t,
                    &context,
                    &ctrl_latent,
                    scale,
                )?;
                Ok(v.to_dtype(DType::F32)?)
            },
        )?;

        on_progress(Progress::Decoding);
        let decoded = self.vae.decode(&latent)?.to_dtype(DType::F32)?; // [1, 3, H, W] in [-1, 1]
        to_image(&decoded)
    }
}

/// Validate the seed-independent request knobs before any tensor work. The empty-prompt guard mirrors
/// the registered txt2img `validate` (an empty prompt reaches the TE as a zero-length sequence and
/// surfaces as a deep tensor-shape error instead of a clean validation error).
fn validate_request(req: &Krea2ControlRequest) -> Result<()> {
    if req.prompt.trim().is_empty() {
        return Err(CandleError::Msg("krea control: prompt is required".into()));
    }
    if !req.width.is_multiple_of(SIZE_MULTIPLE) || !req.height.is_multiple_of(SIZE_MULTIPLE) {
        return Err(CandleError::Msg(format!(
            "krea control: width/height must be multiples of {SIZE_MULTIPLE} (got {}x{})",
            req.width, req.height
        )));
    }
    if req.steps == 0 {
        return Err(CandleError::Msg("krea control: steps must be >= 1".into()));
    }
    Ok(())
}

/// The rendered OpenPose skeleton (HWC RGB u8, already at `width`×`height`) → `[1, 3, H, W]` f32 in
/// `[-1, 1]`, channel-first — the exact normalization `candle_gen::train::dataset::load_image_tensor`
/// produces at train time, so the VAE-encoded control latent is identical to what the branch was
/// trained on. The worker driver renders the control map at the provider's output dims, so a size
/// mismatch is a wiring bug, not a resize case (the lib carries no image codec) — it errors loudly.
fn control_image_to_nchw(
    image: &Image,
    width: u32,
    height: u32,
    device: &Device,
) -> Result<Tensor> {
    let (iw, ih) = (image.width, image.height);
    if (iw, ih) != (width, height) {
        return Err(CandleError::Msg(format!(
            "krea control: control image {iw}x{ih} must match the render size {width}x{height}"
        )));
    }
    let (rw, rh) = (width as usize, height as usize);
    if image.pixels.len() != rw * rh * 3 {
        return Err(CandleError::Msg(format!(
            "krea control: control pixel buffer {} != {width}x{height}x3",
            image.pixels.len()
        )));
    }
    let mut data = vec![0f32; 3 * rh * rw];
    for y in 0..rh {
        for x in 0..rw {
            let base = (y * rw + x) * 3;
            for c in 0..3 {
                // HWC u8 [0,255] → channel-first [3, H, W]; [-1, 1].
                data[c * rh * rw + y * rw + x] = image.pixels[base + c] as f32 / 127.5 - 1.0;
            }
        }
    }
    Ok(Tensor::from_vec(data, (1, 3, rh, rw), &Device::Cpu)?.to_device(device)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The request defaults match the Turbo control production knobs (1024², 8 CFG-free steps,
    /// control scale 0.6).
    #[test]
    fn request_defaults() {
        let r = Krea2ControlRequest::default();
        assert_eq!((r.width, r.height), (1024, 1024));
        assert_eq!(r.steps, TURBO_STEPS);
        assert_eq!(r.control_scale, DEFAULT_CONTROL_SCALE);
        assert!(!r.cancel.is_cancelled());
    }

    /// The empty-prompt guard: an empty or whitespace-only prompt is a clean validation error; a real
    /// prompt with valid size passes.
    #[test]
    fn validate_request_rejects_empty_prompt() {
        let empty = Krea2ControlRequest::default();
        assert!(validate_request(&empty)
            .unwrap_err()
            .to_string()
            .contains("prompt is required"));

        let whitespace = Krea2ControlRequest {
            prompt: " \t\n".into(),
            ..Default::default()
        };
        assert!(validate_request(&whitespace)
            .unwrap_err()
            .to_string()
            .contains("prompt is required"));

        let ok = Krea2ControlRequest {
            prompt: "a dancer mid-leap".into(),
            ..Default::default()
        };
        assert!(validate_request(&ok).is_ok());
    }

    /// The size/steps guards fire.
    #[test]
    fn validate_request_keeps_size_and_steps_guards() {
        let odd = Krea2ControlRequest {
            prompt: "a dancer".into(),
            height: 1000,
            ..Default::default()
        };
        assert!(validate_request(&odd)
            .unwrap_err()
            .to_string()
            .contains("multiples"));

        let zero_steps = Krea2ControlRequest {
            prompt: "a dancer".into(),
            steps: 0,
            ..Default::default()
        };
        assert!(validate_request(&zero_steps)
            .unwrap_err()
            .to_string()
            .contains("steps"));
    }
}
