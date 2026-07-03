//! Z-Image **img2img / edit** provider (sc-6595, epic 5480) — pixel-conditioned editing on
//! Z-Image-Turbo, the candle (Windows/CUDA) sibling of the `mlx-gen-z-image` img2img path (on Mac the
//! registered `z_image_turbo` generator's `Conditioning::Reference` route, [`mlx_gen_z_image`]). The
//! candle `z_image_turbo` descriptor advertises **txt2img only**, so — exactly like the strict-pose
//! Fun-ControlNet ([`crate::control`]) — this is a **bespoke provider driven directly by the worker**
//! (`generate_candle_zimage_edit_stream`), NOT gen-core-registered. The registered descriptor stays
//! honest (no img2img promise it can't serve through the registry path).
//!
//! It reuses the validated txt2img machinery (the **stock** candle-transformers Z-Image DiT + Qwen3
//! encoder + AutoencoderKL VAE + the flow-match Euler schedule, [`crate::pipeline`]) with a
//! strength-derived img2img init:
//!  1. **VAE-encode** the source to its clean latent **mean** — deterministic. candle's
//!     `AutoEncoderKL::encode` runs `DiagonalGaussian{sample:true}`, i.e. it draws `randn` from the
//!     *device* RNG, which is not launch-portable (breaks sc-3673). So, like [`crate::control`], we run
//!     the `Encoder` directly and take the distribution mean: `(mean − shift) · scale`.
//!  2. **Blend** with seeded init noise at the start sigma: `x_t = (1 − σ_start)·clean + σ_start·noise`
//!     (the flow-match interpolation — the fork's `add_noise_by_interpolation`).
//!  3. **Denoise** the reduced schedule `init_time_step..steps` and decode.
//!
//! **Strength is the Z-Image upstream "structure-preservation" convention** (the fork's `init_time_step`,
//! matched here for Mac/Windows parity — NOT the SDXL "more strength ⇒ more change" knob): for strength
//! in `(0, 1]`, `init_time_step = max(1, floor(steps·strength))`, the denoise runs `init_time_step..steps`,
//! and the init is noised to `sigma[init_time_step]`. **Higher strength → later start → fewer steps →
//! output stays CLOSER to the source.** Default [`DEFAULT_EDIT_STRENGTH`] (0.6), matching the worker's
//! `advanced.strength` default. (The candle txt2img schedule is linear — `set_timesteps(steps, Some(mu))`,
//! see [`crate::pipeline`] — so the *image* won't be bit-identical to the MLX static-shift-3.0 path, but
//! the strength knob means the same thing on both backends, and strength→0 reduces to candle txt2img.)
//!
//! **Distilled, no CFG** (Z-Image-Turbo): a single DiT forward per step, no negative prompt / guidance;
//! the predicted velocity is negated before the Euler step (the Z-Image sign convention, see
//! [`crate::pipeline`]). **Deterministic** (sc-3673): the init noise is drawn from a seeded CPU `StdRng`
//! then moved to the device; the flow-match Euler step injects no per-step noise, so generation is a pure
//! function of `(seed, request, source)`. `generate` takes `&self` and returns one [`Image`] (the worker
//! loops `count`), so one loaded model serves many edits.

use std::path::{Path, PathBuf};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::runtime::CancelFlag;
use candle_gen::gen_core::{Image, Progress};
use candle_gen::{CandleError, Result};
use candle_transformers::models::z_image::preprocess::prepare_inputs;
use candle_transformers::models::z_image::scheduler::{
    calculate_shift, FlowMatchEulerDiscreteScheduler, SchedulerConfig, BASE_IMAGE_SEQ_LEN,
    BASE_SHIFT, MAX_IMAGE_SEQ_LEN, MAX_SHIFT,
};
use candle_transformers::models::z_image::text_encoder::{TextEncoderConfig, ZImageTextEncoder};
use candle_transformers::models::z_image::transformer::{
    Config as DitConfig, ZImageTransformer2DModel,
};
use candle_transformers::models::z_image::vae::{AutoEncoderKL, Encoder as VaeEncoder, VaeConfig};

// Shared Z-Image plumbing (loader/decode/preprocess/tokenizer/seed) — one home (sc-9002 / F-022).
use crate::common::{self, ResizePolicy, ENC_DTYPE, PATCH_SIZE, SPATIAL_SCALE};

/// The transformer + latents run bf16 (Z-Image native, the validated candle txt2img dtype); the VAE
/// encoder runs f32 (the encode path's dtype) and its mean is cast to bf16 for the init latent.
const DTYPE: DType = DType::BF16;

/// Z-Image works at /8 then patchifies /2, so both image dims must be multiples of 16 for a clean
/// patchify (the txt2img `validate` floor).
const SIZE_MULTIPLE: u32 = 16;

/// Z-Image-Turbo is guidance-distilled to a fixed 4-step schedule (the txt2img default).
const DEFAULT_STEPS: usize = 4;

/// img2img default strength — the worker's `advanced.strength` default (`resolve_zimage_edit_init`,
/// torch `ZImageImg2ImgPipeline` 0.6). With the fork's `init_time_step`, higher → closer to the source.
pub const DEFAULT_EDIT_STRENGTH: f32 = 0.6;

/// img2img start step (the Z-Image fork's `init_time_step`): for strength in `(0, 1]`,
/// `max(1, floor(steps·strength))`; otherwise 0 (full regeneration from the max-σ prior). Higher strength
/// → later start → fewer denoise steps → output stays closer to the init image — the upstream Z-Image
/// convention, matched here so the strength knob behaves identically on the Mac (MLX) and Windows
/// (candle) lanes. `floor` because Python `int(steps · strength)` truncates toward zero for `s ≥ 0`.
fn init_time_step(num_steps: usize, strength: f32) -> usize {
    if strength > 0.0 {
        let s = strength.clamp(0.0, 1.0);
        ((num_steps as f32 * s) as usize).max(1)
    } else {
        0
    }
}

/// Paths to the Z-Image edit checkpoints — just the base `Tongyi-MAI/Z-Image-Turbo` snapshot (img2img
/// reuses the Turbo weights; no extra checkpoint, unlike the Fun-ControlNet overlay).
pub struct ZImageEditPaths {
    /// The `Tongyi-MAI/Z-Image-Turbo` base snapshot dir (`tokenizer/`, `text_encoder/`, `transformer/`,
    /// `vae/`).
    pub base: PathBuf,
}

/// One Z-Image img2img / edit request. No negative/guidance — Z-Image-Turbo is guidance-distilled.
#[derive(Clone)]
pub struct ZImageEditRequest {
    pub prompt: String,
    pub width: u32,
    pub height: u32,
    pub steps: usize,
    /// Denoise strength in `[0, 1]` — the Z-Image structure-preservation knob (higher ⇒ closer to the
    /// source; see [`init_time_step`]).
    pub strength: f32,
    pub seed: u64,
    /// Cooperative cancellation, checked before each denoise step (the engine contract).
    pub cancel: CancelFlag,
}

impl Default for ZImageEditRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            width: 1024,
            height: 1024,
            steps: DEFAULT_STEPS,
            strength: DEFAULT_EDIT_STRENGTH,
            seed: 0,
            cancel: CancelFlag::default(),
        }
    }
}

/// Loaded Z-Image img2img model: the Qwen3 tokenizer + text encoder, the **stock** Z-Image DiT (the same
/// txt2img transformer — img2img is just a different init + start step), the decode VAE, and a VAE
/// encoder (deterministic mean encode of the source).
pub struct ZImageEdit {
    device: Device,
    text_encoder: ZImageTextEncoder,
    /// Qwen tokenizer, loaded+parsed **once** at load and reused across encodes (sc-8991 / F-011)
    /// instead of re-parsing `tokenizer.json` per prompt.
    tokenizer: candle_gen::gen_core::tokenizer::TextTokenizer,
    transformer: ZImageTransformer2DModel,
    vae: AutoEncoderKL,
    vae_encoder: VaeEncoder,
    vae_shift: f64,
    vae_scale: f64,
}

impl ZImageEdit {
    /// Load the base Z-Image components (Qwen3 encoder + stock DiT + VAE) + a VAE encoder for the source.
    /// The transformer + VAE decode run bf16 (the validated txt2img dtype); the VAE encoder runs f32. The
    /// DiT honors the process-global accelerated-attention toggle, exactly as the txt2img pipeline does,
    /// so img2img uses the same attention path.
    pub fn load(paths: &ZImageEditPaths) -> Result<Self> {
        let device = candle_gen::default_device()?;
        let root = paths.base.clone();

        let text_encoder = ZImageTextEncoder::new(
            &TextEncoderConfig::z_image(),
            component_vb(&root, "text_encoder", DTYPE, &device)?,
        )?;

        let mut dit_cfg = DitConfig::z_image_turbo();
        // sc-9032: no-op `flash-attn` feature removed; accelerated dispatch is never wired behind a
        // build feature, so this is always off (was `cfg!(feature = "flash-attn") && …`).
        dit_cfg.set_use_accelerated_attn(false);
        let transformer = ZImageTransformer2DModel::new(
            &dit_cfg,
            component_vb(&root, "transformer", DTYPE, &device)?,
        )?;

        let vae_cfg = VaeConfig::z_image();
        let vae = AutoEncoderKL::new(&vae_cfg, component_vb(&root, "vae", DTYPE, &device)?)?;
        let vae_encoder = VaeEncoder::new(
            &vae_cfg,
            component_vb(&root, "vae", ENC_DTYPE, &device)?.pp("encoder"),
        )?;

        let tokenizer = common::build_tokenizer(&root, "z-image edit")?;
        Ok(Self {
            device,
            text_encoder,
            tokenizer,
            transformer,
            vae,
            vae_encoder,
            vae_shift: vae_cfg.shift_factor,
            vae_scale: vae_cfg.scaling_factor,
        })
    }

    /// img2img: regenerate `source` toward `req.prompt` at `req.strength`, denoising with the distilled
    /// flow-match Euler schedule (no CFG). VAE-encodes the source once into the clean init latent, blends
    /// it with seeded noise at the start sigma, then runs the reduced `init_time_step..steps` loop.
    pub fn generate(
        &self,
        req: &ZImageEditRequest,
        source: &Image,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        if req.cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }
        if !req.width.is_multiple_of(SIZE_MULTIPLE) || !req.height.is_multiple_of(SIZE_MULTIPLE) {
            return Err(CandleError::Msg(format!(
                "z-image edit: width/height must be multiples of {SIZE_MULTIPLE} (got {}x{})",
                req.width, req.height
            )));
        }
        let steps = req.steps.max(1);
        let lat_h = (req.height / SPATIAL_SCALE) as usize;
        let lat_w = (req.width / SPATIAL_SCALE) as usize;

        // Text embeddings (bf16, the txt2img path) — seed- and source-independent.
        let cap = self.text_embeddings(&req.prompt)?;

        // Deterministic clean source latent (mean encode × scale), constant across the count loop.
        let clean = self.encode_source(source, req.width, req.height)?; // (1, 16, lat_h, lat_w) bf16

        // Flow-match Euler schedule — the txt2img construction (Some(mu) ⇒ linear sigmas; see pipeline.rs).
        let image_seq_len = ((lat_h as u32 / PATCH_SIZE) * (lat_w as u32 / PATCH_SIZE)) as usize;
        let mu = calculate_shift(
            image_seq_len,
            BASE_IMAGE_SEQ_LEN,
            MAX_IMAGE_SEQ_LEN,
            BASE_SHIFT,
            MAX_SHIFT,
        );
        let mut scheduler = FlowMatchEulerDiscreteScheduler::new(SchedulerConfig::z_image_turbo());
        scheduler.set_timesteps(steps, Some(mu));

        // img2img start step + the sigma to noise the source to. `sigmas` has `steps + 1` entries
        // (indices 0..=steps); `start == steps` ⇒ σ_start = 0 ⇒ x_t = clean and an empty denoise loop
        // (the honest result of a max-strength structure-preserving edit: the source VAE round-trip).
        let start = init_time_step(steps, req.strength);
        let sigma_start = scheduler.sigmas[start];

        // Deterministic, launch-portable init noise (sc-3673, shared [`common::seed_noise`]).
        let noise = common::seed_noise(req.seed, lat_h, lat_w, &self.device, DTYPE)?;

        // Flow-match interpolation blend: x_t = (1 − σ_start)·clean + σ_start·noise.
        let x_t = ((clean * (1.0 - sigma_start))? + (noise * sigma_start)?)?;

        // prepare_inputs pads cap_feats (+ mask) and adds the frame axis → latents (1,16,1,lat_h,lat_w).
        let prepared = prepare_inputs(&x_t, std::slice::from_ref(&cap), &self.device)?;
        let cap_feats = prepared.cap_feats;
        let cap_mask = prepared.cap_mask;
        let mut latents = prepared.latents;

        // Reduced schedule: run steps `start..steps`. Reading scheduler.sigmas/timesteps directly (both
        // pub) and doing the Euler step inline is byte-identical to the txt2img loop's
        // `current_timestep_normalized()` + `step()` — it just starts at `start` instead of 0.
        let total = (steps - start) as u32;
        for step_i in start..steps {
            if req.cancel.is_cancelled() {
                return Err(CandleError::Canceled);
            }
            // The DiT timestep convention is 1−σ (the scheduler's `current_timestep_normalized`).
            let t_norm = (1000.0 - scheduler.timesteps[step_i]) / 1000.0;
            let t = Tensor::from_vec(vec![t_norm as f32], (1,), &self.device)?;
            // The Z-Image DiT velocity is negated before the flow-match Euler step (the sign convention).
            let velocity = self
                .transformer
                .forward(&latents, &t, &cap_feats, &cap_mask)?
                .neg()?;
            // Euler step: x_{i+1} = x_i + (σ_{i+1} − σ_i)·velocity (= scheduler.step).
            let dt = scheduler.sigmas[step_i + 1] - scheduler.sigmas[step_i];
            latents = (latents + (velocity * dt)?)?;
            on_progress(Progress::Step {
                current: (step_i - start) as u32 + 1,
                total,
            });
        }

        on_progress(Progress::Decoding);
        self.decode(&latents)
    }

    /// Prompt → `cap_feats` `(seq, 2560)` at bf16 via the Qwen3 encoder + the shared Qwen chat template
    /// ([`common::prompt_ids`] + [`common::encode_ids`]).
    fn text_embeddings(&self, prompt: &str) -> Result<Tensor> {
        let ids = common::prompt_ids(&self.tokenizer, prompt, "z-image edit")?;
        common::encode_ids(&ids, &self.device, DTYPE, |input_ids| {
            self.text_encoder.forward(input_ids)
        })
    }

    /// VAE-encode `source` (LANCZOS-resized to the render size, normalized to `[-1, 1]` NCHW) to the
    /// deterministic clean latent `(1, 16, H/8, W/8)` bf16: the distribution **mean** (not a sampled
    /// draw), mapped to latent space as `(mean − shift) · scale` via the shared [`common::encode_mean`].
    /// The candle `AutoEncoderKL::encode` samples via the device RNG (not launch-portable, sc-3673), so
    /// the raw `Encoder` is run here instead — the same deterministic path [`crate::control`] uses.
    fn encode_source(&self, source: &Image, width: u32, height: u32) -> Result<Tensor> {
        // `ResizeAlways`: the edit provider always LANCZOS-fits the source to the render size.
        let img = common::preprocess_image(
            source,
            width,
            height,
            ResizePolicy::ResizeAlways,
            &self.device,
            "z-image edit",
        )?; // f32 (1,3,H,W) [-1,1]
        common::encode_mean(
            &self.vae_encoder,
            &img,
            self.vae_shift,
            self.vae_scale,
            DTYPE,
        )
    }

    /// VAE-decode the final latents `(1, 16, 1, h, w)` → an RGB8 [`Image`] (the shared txt2img decode).
    /// The edit lane does not carry a PiD decoder (base txt2img is the shipping PiD path, epic 7840 /
    /// sc-7853); native VAE decode.
    fn decode(&self, latents: &Tensor) -> Result<Image> {
        common::decode(&self.vae, None, latents)
    }
}

/// mmap a [`VarBuilder`] over every `.safetensors` in `root/sub` at `dtype` (the txt2img loader, shared
/// shape with [`crate::control`]'s `component_vb`).
fn component_vb(
    root: &Path,
    sub: &str,
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder<'static>> {
    candle_gen::component_vb(root, sub, dtype, device, "z-image edit")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_defaults() {
        let r = ZImageEditRequest::default();
        assert_eq!((r.width, r.height), (1024, 1024));
        assert_eq!(r.steps, DEFAULT_STEPS);
        assert_eq!(r.strength, DEFAULT_EDIT_STRENGTH);
        assert_eq!(DEFAULT_EDIT_STRENGTH, 0.6);
        assert!(!r.cancel.is_cancelled());
    }

    /// The Z-Image "structure-preservation" strength → start-step law (the fork's `init_time_step`):
    /// `max(1, floor(steps·strength))` for strength > 0, else 0. Higher strength → later start → fewer
    /// steps run. Pure function, no GPU — the cross-backend-parity contract with the MLX lane.
    #[test]
    fn init_time_step_is_fork_convention() {
        // strength 0 / negative ⇒ full regeneration from step 0.
        assert_eq!(init_time_step(4, 0.0), 0);
        assert_eq!(init_time_step(4, -1.0), 0);
        // floor(steps·strength), min 1.
        assert_eq!(init_time_step(4, 0.05), 1); // floor(0.2)=0 → max(1,0)=1
        assert_eq!(init_time_step(4, 0.25), 1); // floor(1.0)=1
        assert_eq!(init_time_step(4, 0.6), 2); // floor(2.4)=2 (the default)
        assert_eq!(init_time_step(4, 0.75), 3); // floor(3.0)=3
        assert_eq!(init_time_step(4, 1.0), 4); // floor(4.0)=4 == steps ⇒ empty loop, source round-trip
                                               // clamp above 1.
        assert_eq!(init_time_step(4, 2.0), 4);
        // Higher strength is monotonically a later (or equal) start ⇒ fewer (or equal) steps.
        let starts: Vec<usize> = [0.1, 0.3, 0.5, 0.7, 0.9]
            .iter()
            .map(|&s| init_time_step(8, s))
            .collect();
        assert!(starts.windows(2).all(|w| w[0] <= w[1]), "{starts:?}");
    }

    /// The init blend reduces to the txt2img prior at the loop start: σ_start = sigmas[start], and the
    /// reduced denoise runs `start..steps`. Anchors the schedule indices the generate loop reads (no GPU).
    #[test]
    fn schedule_start_index_and_sigma() {
        let mu = calculate_shift(
            4096,
            BASE_IMAGE_SEQ_LEN,
            MAX_IMAGE_SEQ_LEN,
            BASE_SHIFT,
            MAX_SHIFT,
        );
        let mut s = FlowMatchEulerDiscreteScheduler::new(SchedulerConfig::z_image_turbo());
        s.set_timesteps(DEFAULT_STEPS, Some(mu));
        // sigmas has steps + 1 entries, starts at 1.0, ends at 0.
        assert_eq!(s.sigmas.len(), DEFAULT_STEPS + 1);
        assert!((s.sigmas[0] - 1.0).abs() < 1e-9);
        assert!(s.sigmas[DEFAULT_STEPS].abs() < 1e-12);
        // At the default strength 0.6 the loop starts at index 2, runs 2 steps, and noises to sigmas[2].
        let start = init_time_step(DEFAULT_STEPS, DEFAULT_EDIT_STRENGTH);
        assert_eq!(start, 2);
        assert_eq!(DEFAULT_STEPS - start, 2);
        assert!(s.sigmas[start] > 0.0 && s.sigmas[start] < 1.0);
        // Max strength ⇒ start == steps ⇒ σ_start == 0 (x_t = clean) and an empty loop.
        let full = init_time_step(DEFAULT_STEPS, 1.0);
        assert_eq!(full, DEFAULT_STEPS);
        assert!(s.sigmas[full].abs() < 1e-12);
    }

    /// The edit source preprocess (shared [`common::preprocess_image`] with `ResizeAlways`) resizes to
    /// the render size and maps `[0,255] → [-1,1]` in CHW f32. A solid white source ⇒ all ≈ 1.0; a
    /// non-multiple buffer errors.
    #[test]
    fn source_preprocess_shape_and_range() {
        let img = Image {
            width: 8,
            height: 8,
            pixels: vec![255u8; 8 * 8 * 3],
        };
        let t = common::preprocess_image(
            &img,
            16,
            16,
            ResizePolicy::ResizeAlways,
            &Device::Cpu,
            "z-image edit",
        )
        .unwrap();
        assert_eq!(t.dims(), &[1, 3, 16, 16]); // resized to the render size
        let v = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(v.iter().all(|x| (x - 1.0).abs() < 1e-3)); // 255 → 1.0
        let bad = Image {
            width: 8,
            height: 8,
            pixels: vec![0u8; 8 * 8 * 3 - 1],
        };
        assert!(common::preprocess_image(
            &bad,
            8,
            8,
            ResizePolicy::ResizeAlways,
            &Device::Cpu,
            "z-image edit"
        )
        .is_err());
    }
}
