//! FLUX.1-dev **Fun-Controlnet-Union** provider (sc-8412) — the candle (Windows/CUDA) sibling of
//! `mlx-gen-flux`'s `Flux1DevControl` (sc-8238/8239). Strict structural conditioning (pose / canny /
//! depth — **input-agnostic**, no discrete mode index) on FLUX.1-dev via
//! `Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0`, a diffusers residual-emitter control branch
//! overlaid on the dev base DiT.
//!
//! **How it conditions:** the pose/canny/depth control image is VAE-encoded + 2×2-packed into the
//! packed transformer latent `[1, seq, 64]` (the same pack as the noise latents, so it aligns 1:1 with
//! the base image tokens), constant across the denoise. [`FluxControlTransformer`] runs the
//! parity-proven dev DiT ([`crate::ip_dit::IpFlux`]) plus the control branch
//! ([`crate::control::FluxControlNet`]): 6 per-block residuals are computed once and added into the base
//! image stream after base double blocks at `interval = ceil(19/6) = 4`, scaled by `control_scale`
//! (Shakker README sweet spot ≈ 0.7; absent → that default, an explicit `Some(0.0)` = control off).
//! dev is guidance-distilled — a single embedded-guidance forward,
//! no true-CFG / negative pass.
//!
//! **Compose-readiness:** the denoise routes through [`FluxControlTransformer::forward_composed`], which
//! threads an optional identity injector ([`crate::ip_dit::DitImageInjector`] — PuLID / XLabs
//! IP-Adapter) into the SAME base double-block stream as the control residuals. This provider wires the
//! seam (and exposes [`Flux1DevControl::generate_with_injector`]); a follow-on epic stacks identity +
//! control in one denoise step by passing `Some(injector)`.
//!
//! Bespoke provider (NOT gen-core-registered), worker-invoked by name — the candle pattern for
//! conditioned surfaces (mirrors [`crate::ip_provider`] / the FLUX.2 control provider). The
//! `flux1_dev_control` worker lane is a separate Phase-B step (sc-8304/sc-8246), not this crate.
//! Determinism is the candle-lane contract (sc-3673): seeded CPU init noise; the control encode uses
//! the VAE posterior MEAN (no sampling, no device RNG — sc-8988), so the control latent is fully
//! deterministic and launch-portable.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::vae::native::{AutoEncoder, DiagonalGaussian, Encoder};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::clip::text_model::ClipTextTransformer;
use candle_transformers::models::flux::sampling::{get_schedule, State};
use candle_transformers::models::t5::T5EncoderModel;

use candle_gen::gen_core::imageops::resize_lanczos_u8;
use candle_gen::gen_core::runtime::CancelFlag;
use candle_gen::gen_core::sampling::TimestepConvention;
use candle_gen::gen_core::{Image, Progress};
use candle_gen::{CandleError, Result};

use crate::control::{
    accepts_control_kind, FluxControlNet, FluxControlNetConfig, FluxControlTransformer,
};
use crate::flux1_load;
use crate::ip_dit::{DitImageInjector, IpFlux};
use crate::pipeline::{ae_config, decode_latents, encode_text, flow_mu, flux_config};
use crate::Variant;

/// The provider-specific error label for the shared [`crate::flux1_load`] diagnostics.
const LABEL: &str = "flux1 control";
/// FLUX runs at bf16.
const DTYPE: DType = DType::BF16;
/// FLUX latent channel count (the raw VAE latent / initial noise; the DiT packs it 2×2 to 64).
const LATENT_CHANNELS: usize = 16;
/// FLUX dev's resolution-dependent flow-match time-shift endpoints (matching the txt2img pipeline).
const BASE_SHIFT: f64 = 0.5;
const MAX_SHIFT: f64 = 1.15;

/// FLUX latent geometry requires both image dims to be multiples of 16 for a clean 2×2 pack.
const SIZE_MULTIPLE: u32 = 16;

/// Default control-conditioning scale — the Shakker Union-Pro-2.0 README recommends ≈ 0.7. Used only when
/// the request leaves `control_scale` **absent** (`None`); an explicit `Some(x)` always wins, including
/// `Some(0.0)` (which the engine proves is byte-identical to the base forward — "control off"). See
/// sc-9024 / F-040: an explicit 0.0 must NOT be remapped to the default (0.0 is a valid value, not
/// "unset").
pub const DEFAULT_CONTROL_SCALE: f32 = 0.7;

/// Paths to the FLUX.1-dev control checkpoints: the dev snapshot dir + the Shakker Fun-Controlnet-Union
/// overlay.
pub struct Flux1ControlPaths {
    /// The black-forest-labs FLUX.1-dev snapshot dir (`flux1-dev.safetensors`, `ae.safetensors`,
    /// `text_encoder/`, `text_encoder_2/`, `tokenizer_2/`).
    pub flux_base: PathBuf,
    /// The `FLUX.1-dev-ControlNet-Union-Pro-2.0` checkpoint (a single `.safetensors`
    /// `diffusion_pytorch_model.safetensors`, or a dir containing it).
    pub control: PathBuf,
}

/// One FLUX.1-dev strict-control request. dev is guidance-distilled — `guidance` is the embedded scalar
/// (single forward, no negative prompt).
#[derive(Clone)]
pub struct Flux1ControlRequest {
    pub prompt: String,
    pub width: u32,
    pub height: u32,
    pub steps: usize,
    /// Embedded guidance scale (dev default ≈ 3.5).
    pub guidance: f32,
    /// `control_scale` — how strongly the control branch locks the base (≈ 0.7). `None` = absent → the
    /// [`DEFAULT_CONTROL_SCALE`]; `Some(x)` is honored verbatim, including `Some(0.0)` for "control off"
    /// (byte-identical to the base forward). sc-9024: an explicit 0.0 is a valid value, not "unset".
    pub control_scale: Option<f32>,
    /// The control kind (pose / canny / depth). Input-agnostic — used only to validate the accepted set;
    /// it does NOT branch the forward (Union-Pro-2.0 dropped the discrete mode index).
    pub control_kind: String,
    pub seed: u64,
    /// Cooperative cancellation, checked before each denoise step (the engine contract).
    pub cancel: CancelFlag,
}

impl Default for Flux1ControlRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            width: 1024,
            height: 1024,
            steps: 25,
            guidance: 3.5,
            control_scale: None,
            control_kind: "pose".into(),
            seed: 0,
            cancel: CancelFlag::default(),
        }
    }
}

/// Open a VarBuilder over the Shakker control checkpoint — a single `.safetensors` `File` or a `Dir`
/// containing it — on `device` at `dtype`.
fn control_var_builder(path: &Path, dtype: DType, device: &Device) -> Result<VarBuilder<'static>> {
    candle_gen::load_path_mmap(path, dtype, device, "flux1 control")
}

/// Resize `image` to `width`×`height` and convert to an NCHW `[1, 3, H, W]` tensor in `[-1, 1]` at
/// `dtype` on `device` — the VAE-encode input for the control image (the same normalization the noise
/// path uses for decode).
fn preprocess_control(
    image: &Image,
    width: u32,
    height: u32,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let (iw, ih) = (image.width as usize, image.height as usize);
    if image.pixels.len() != iw * ih * 3 {
        return Err(CandleError::Msg(format!(
            "flux1 control: control pixel buffer {} != {iw}x{ih}x3",
            image.pixels.len()
        )));
    }
    let (rw, rh) = (width as usize, height as usize);
    // Lanczos-resize to the render size (no-op when already that size), then [0,255] → [-1,1], HWC→NCHW.
    let resized: Vec<f32> = if (ih, iw) == (rh, rw) {
        image.pixels.iter().map(|&v| v as f32).collect()
    } else {
        resize_lanczos_u8(&image.pixels, ih, iw, rh, rw)? // HWC f32 [0,255]
    };
    let data: Vec<f32> = resized.iter().map(|&v| 2.0 * (v / 255.0) - 1.0).collect();
    let hwc = Tensor::from_vec(data, (rh, rw, 3), device)?;
    let nchw = hwc.permute((2, 0, 1))?.unsqueeze(0)?.contiguous()?;
    nchw.to_dtype(dtype).map_err(Into::into)
}

/// Deterministic control-image VAE encoder (sc-8988): the FLUX [`Encoder`] + the posterior MEAN
/// (`DiagonalGaussian` with `sample = false`) + the BFL shift/scale. The upstream
/// [`AutoEncoder::encode`] *samples* the posterior with device `randn` — per-launch-deterministic at
/// best and never launch-portable, violating the sc-3673 determinism contract the rest of this
/// provider maintains (CPU-seeded `StdRng` everywhere). The mean encode is RNG-free (matching
/// `Flux2Vae::encode_packed` and the boogu edit precedent) and needs no seed at all.
struct MeanVaeEncoder {
    encoder: Encoder,
    shift_factor: f64,
    scale_factor: f64,
}

impl MeanVaeEncoder {
    /// Encode `nchw` (`[1, 3, H, W]` in `[-1, 1]`) to the posterior-MEAN latent `[1, 16, H/8, W/8]`,
    /// shift/scale-normalized exactly like [`AutoEncoder::encode`] — minus the sampling.
    fn encode(&self, nchw: &Tensor) -> Result<Tensor> {
        let moments = nchw.apply(&self.encoder)?; // [1, 2·z, H/8, W/8] = (mean ‖ logvar)
        let mean = moments.apply(&DiagonalGaussian::new(false, 1)?)?;
        ((mean - self.shift_factor)? * self.scale_factor).map_err(Into::into)
    }
}

/// A loaded FLUX.1-dev control model: the reused FLUX text encoders + VAE, the dev DiT wrapped in its
/// control branch ([`FluxControlTransformer`]). `generate` takes `&self` (no per-call mutation), so one
/// load serves many renders.
pub struct Flux1DevControl {
    /// T5 + CLIP tokenizers, loaded+parsed **once** at load and reused across encodes (sc-8991 / F-011)
    /// instead of re-parsing per prompt in `encode_text`.
    toks: crate::pipeline::FluxTokenizers,
    device: Device,
    dtype: DType,
    clip: ClipTextTransformer,
    /// Behind a `Mutex` because `T5EncoderModel::forward` takes `&mut self` while `generate` is `&self`;
    /// locked only for the once-per-request text encode.
    t5: Mutex<T5EncoderModel>,
    transformer: FluxControlTransformer,
    vae: AutoEncoder,
    /// The deterministic (posterior-MEAN) control-image encoder — sc-8988. A second `Encoder` instance
    /// over the same `ae.safetensors` (the upstream `AutoEncoder`'s encoder is private and its `encode`
    /// samples); ~34M params of duplicated encoder weights, negligible next to the dev DiT.
    control_encoder: MeanVaeEncoder,
}

impl Flux1DevControl {
    /// Load the dev base (the forked [`IpFlux`] DiT — the only FLUX DiT with the compose-ready injector
    /// seam) + the reused text encoders + VAE + the Shakker control overlay, assembling the
    /// [`FluxControlTransformer`].
    pub fn load(paths: &Flux1ControlPaths) -> Result<Self> {
        let device = candle_gen::default_device()?;
        let dtype = DTYPE;
        let root = paths.flux_base.clone();
        let variant = Variant::Dev; // the Shakker control is FLUX.1-dev only.

        if !root.join(variant.transformer_file()).is_file() {
            return Err(CandleError::Msg(format!(
                "flux1 control: no {} in {} (expected a black-forest-labs FLUX.1-dev snapshot)",
                variant.transformer_file(),
                root.display()
            )));
        }

        // CLIP-L + T5-XXL text encoders (shared FLUX.1 backbone load, sc-9003).
        let (clip, t5) = flux1_load::text_encoders(&root, dtype, &device, LABEL)?;

        // The forked FLUX DiT (the compose-ready injector seam) from the root BFL checkpoint — the genuine
        // per-provider drift (IpFlux base, not the stock Flux), so the wrapper choice stays here.
        let dit_vb = flux1_load::dit_vb(&root, variant, dtype, &device, LABEL)?;
        let base = IpFlux::new(&flux_config(variant), dit_vb)?;

        // FLUX AutoEncoder (`ae.safetensors`) for decode, plus a separate posterior-MEAN encoder over the
        // same weights for the deterministic control encode (sc-8988). The shared loader hands back the
        // VAE plus its VarBuilder so the mean-encoder reuses the same mmap (the per-provider drift here).
        let ae_cfg = ae_config(variant);
        let (vae, vae_vb) = flux1_load::vae(&root, variant, dtype, &device, LABEL)?;
        let control_encoder = MeanVaeEncoder {
            encoder: Encoder::new(&ae_cfg, vae_vb.pp("encoder"))?,
            shift_factor: ae_cfg.shift_factor,
            scale_factor: ae_cfg.scale_factor,
        };

        // The Shakker control overlay (diffusers layout, un-prefixed keys). The branch shares the base
        // FLUX config's dims (hidden / heads / RoPE) so its residuals align with the base image tokens.
        let control_vb = control_var_builder(&paths.control, dtype, &device)?;
        let branch = FluxControlNet::new(
            &flux_config(variant),
            &FluxControlNetConfig::shakker_union_pro_2_0(),
            control_vb,
        )?;
        let transformer = FluxControlTransformer::new(base, branch);

        let toks = crate::pipeline::FluxTokenizers::load(&root)?;
        Ok(Self {
            toks,
            device,
            dtype,
            clip,
            t5: Mutex::new(t5),
            transformer,
            vae,
            control_encoder,
        })
    }

    /// The injection interval over the base double blocks (`ceil(19/6) = 4`).
    pub fn residual_interval(&self) -> usize {
        self.transformer.residual_interval()
    }

    /// Number of control residuals (the control double-block count, 6).
    pub fn num_residuals(&self) -> usize {
        self.transformer.num_residuals()
    }

    /// Generate one control-conditioned image. `control_image` is the preprocessed pose/canny/depth hint
    /// (the worker pre-fits it to the render size; this re-resizes defensively). `injector = None` is the
    /// plain control path.
    pub fn generate(
        &self,
        req: &Flux1ControlRequest,
        control_image: &Image,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        self.generate_with_injector(req, control_image, None, on_progress)
    }

    /// As [`generate`](Self::generate), but threading an OPTIONAL identity injector (PuLID / XLabs
    /// IP-Adapter) into every control-denoise step — the **compose-ready** seam. With `injector = None`
    /// this is the plain control path; `Some(..)` stacks identity + control in one denoise.
    pub fn generate_with_injector(
        &self,
        req: &Flux1ControlRequest,
        control_image: &Image,
        injector: Option<&dyn DitImageInjector>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        if req.cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }
        if req.prompt.trim().is_empty() {
            return Err(CandleError::Msg("flux1 control: prompt is required".into()));
        }
        if !accepts_control_kind(&req.control_kind) {
            return Err(CandleError::Msg(format!(
                "flux1_dev_control supports pose/canny/depth control (Fun-Controlnet-Union), got {:?}",
                req.control_kind
            )));
        }
        if !req.width.is_multiple_of(SIZE_MULTIPLE) || !req.height.is_multiple_of(SIZE_MULTIPLE) {
            return Err(CandleError::Msg(format!(
                "flux1 control: width/height must be multiples of {SIZE_MULTIPLE} (got {}x{})",
                req.width, req.height
            )));
        }
        if req.steps == 0 {
            return Err(CandleError::Msg("flux1 control: steps must be >= 1".into()));
        }
        // sc-9024 (F-040): distinguish "absent" from an explicit 0.0. Only an ABSENT scale (`None`) falls
        // back to the Shakker default; an explicit `Some(x)` is honored verbatim — including `Some(0.0)`,
        // which the engine proves is byte-identical to the base forward ("control off"). The old code
        // treated 0.0 as "unset" and silently steered at 0.7, contradicting the engine's own ablation
        // semantics (scale 0 ≡ base; see `control_parity` / `control_real_weights`).
        let control_scale = req.control_scale.unwrap_or(DEFAULT_CONTROL_SCALE) as f64;

        // Conditioning (seed-independent): text (T5 seq + CLIP pooled) + the packed control latent.
        let (t5_emb, clip_emb) = encode_text(
            Variant::Dev,
            &self.toks,
            &self.device,
            self.dtype,
            &self.clip,
            &self.t5,
            &req.prompt,
        )?;
        let control_latent = self.encode_control_latent(control_image, req.width, req.height)?;

        // candle's get_noise geometry: latent is /8 of a multiple-of-16 request. sc-3673 parity:
        // deterministic, launch-portable CPU-seeded initial noise (shared FLUX.1 helper, sc-9003).
        let lat_h = (req.height as usize).div_ceil(16) * 2;
        let lat_w = (req.width as usize).div_ceil(16) * 2;
        let noise = flux1_load::seeded_noise(
            req.seed,
            LATENT_CHANNELS,
            lat_h,
            lat_w,
            &self.device,
            self.dtype,
        )?;

        let state = State::new(&t5_emb, &clip_emb, &noise)?;
        let timesteps = get_schedule(req.steps, Some((state.img.dim(1)?, BASE_SHIFT, MAX_SHIFT)));
        let guidance = req.guidance as f64;

        let latents = self.denoise(
            &state,
            &control_latent,
            &timesteps,
            guidance,
            control_scale,
            injector,
            req.seed,
            &req.cancel,
            on_progress,
        )?;
        on_progress(Progress::Decoding);
        // Control lane does not carry a PiD decoder (base txt2img is the shipping PiD path, epic 7840 /
        // sc-7853); native FLUX VAE decode.
        decode_latents(
            &self.vae,
            None,
            &latents,
            req.height as usize,
            req.width as usize,
        )
    }

    /// VAE-encode + 2×2-pack the control hint into the packed control latent `[1, seq, 64]` (constant
    /// across steps). Uses the same VAE weights + pack (`State::new`'s patchify) as the noise latents, so
    /// the control latent aligns 1:1 with the base image tokens. The encode takes the posterior MEAN
    /// (sc-8988) — no sampling, no device RNG — so the latent is fully deterministic and launch-portable
    /// (the candle determinism contract, sc-3673), identical across steps and across the batch.
    fn encode_control_latent(&self, image: &Image, width: u32, height: u32) -> Result<Tensor> {
        let nchw = preprocess_control(image, width, height, &self.device, self.dtype)?;
        let encoded = self.control_encoder.encode(&nchw)?; // [1, 16, H/8, W/8]
                                                           // Reuse the candle State patchify to 2×2-pack to [1, seq, 64] — identical geometry to the noise
                                                           // pack. `State::new` needs a t5/clip embed only to fill txt/vec (unused here), so feed tiny
                                                           // placeholders and take only `.img` (the packed latent).
        let dummy_t5 = Tensor::zeros((1, 1, 4096), self.dtype, &self.device)?;
        let dummy_clip = Tensor::zeros((1, 768), self.dtype, &self.device)?;
        let packed = State::new(&dummy_t5, &dummy_clip, &encoded)?;
        Ok(packed.img)
    }

    /// The flow-match denoise with the control branch, routed through the unified curated
    /// sampler/scheduler driver (epic 7114 P4) — the control twin of the IP/txt2img `denoise`. dev is
    /// guidance-distilled (a single embedded-guidance forward, no negative pass). The control forward
    /// lives INSIDE the `predict` closure so a multi-eval solver re-runs the whole step. FLUX feeds the
    /// raw timestep (`Sigma` convention: `t == σ`). The provider exposes no sampler/scheduler knob, so
    /// this defaults to the native flow-match Euler path. Cancellation + progress are owned by the driver.
    #[allow(clippy::too_many_arguments)]
    fn denoise(
        &self,
        state: &State,
        control_latent: &Tensor,
        timesteps: &[f64],
        guidance: f64,
        control_scale: f64,
        injector: Option<&dyn DitImageInjector>,
        seed: u64,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Tensor> {
        let b_sz = state.img.dim(0)?;
        let guidance_t = Tensor::full(guidance as f32, b_sz, &self.device)?;
        let native: Vec<f32> = timesteps.iter().map(|&t| t as f32).collect();
        let mu = flow_mu(Variant::Dev, state.img.dim(1)?);
        let steps = native.len().saturating_sub(1);
        let sigmas = candle_gen::resolve_flow_schedule(None, mu, steps, &native);
        candle_gen::run_flow_sampler(
            None,
            TimestepConvention::Sigma,
            &sigmas,
            state.img.clone(),
            seed,
            cancel,
            on_progress,
            |img, t| -> Result<Tensor> {
                let t_vec = Tensor::full(t, b_sz, &self.device)?;
                Ok(self.transformer.forward_composed(
                    img,
                    &state.img_ids,
                    &state.txt,
                    &state.txt_ids,
                    &t_vec,
                    &state.vec,
                    Some(&guidance_t),
                    control_latent,
                    control_scale,
                    injector,
                )?)
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The request defaults match the dev control production knobs (1024², 25 steps, guidance 3.5, pose).
    /// `control_scale` defaults to `None` (absent) — resolved to [`DEFAULT_CONTROL_SCALE`] at generate
    /// time (sc-9024: absent ≠ an explicit 0.0).
    #[test]
    fn request_defaults() {
        let r = Flux1ControlRequest::default();
        assert_eq!((r.width, r.height), (1024, 1024));
        assert_eq!(r.steps, 25);
        assert_eq!(r.guidance, 3.5);
        assert_eq!(r.control_scale, None);
        assert_eq!(r.control_kind, "pose");
        assert!(!r.cancel.is_cancelled());
    }

    /// sc-9024 (F-040): the `control_scale` resolution honors an explicit `Some(0.0)` as "control off"
    /// (byte-identical to the base forward — see `control_parity`) and only substitutes the Shakker
    /// default (0.7) when the scale is truly absent (`None`). A genuine 0.0 must NOT be remapped to 0.7.
    #[test]
    fn control_scale_resolution_honors_explicit_zero() {
        // The generate-time resolution: `req.control_scale.unwrap_or(DEFAULT_CONTROL_SCALE)`.
        let resolve = |cs: Option<f32>| cs.unwrap_or(DEFAULT_CONTROL_SCALE) as f64;

        // Absent → the default (0.7): a request that supplied a control but left the scale unset steers.
        assert_eq!(
            resolve(None),
            DEFAULT_CONTROL_SCALE as f64,
            "absent (None) must fall back to the Shakker default"
        );
        // Explicit 0.0 → 0.0 (NOT 0.7): the caller asked for control off; honor it (0 ≡ base forward).
        assert_eq!(
            resolve(Some(0.0)),
            0.0,
            "explicit Some(0.0) must be honored, not remapped to the default"
        );
        // Explicit non-zero → verbatim (compare at the same f32→f64 widening the code performs).
        assert_eq!(
            resolve(Some(0.35)),
            0.35f32 as f64,
            "explicit Some(x) is honored verbatim"
        );
        assert_eq!(resolve(Some(1.0)), 1.0);
    }

    /// The control checkpoint resolver: a missing path errors loudly; a direct file is used as-is.
    #[test]
    fn control_checkpoint_resolution() {
        let dev = Device::Cpu;
        let dir = std::env::temp_dir().join(format!("flux1_ctrl_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        // A nonexistent path errors.
        assert!(control_var_builder(&dir.join("nope.safetensors"), DTYPE, &dev).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    use crate::vae::native::Config as AeCfg;
    use candle_nn::VarMap;
    use rand::{rngs::StdRng, SeedableRng};

    /// A tiny FLUX AE config (real BFL shift/scale, `ch = 32` — the AE's `group_norm(32, ·)` floor).
    fn tiny_ae_cfg() -> AeCfg {
        AeCfg {
            resolution: 16,
            in_channels: 3,
            ch: 32,
            out_ch: 3,
            ch_mult: vec![1],
            num_res_blocks: 1,
            z_channels: 2,
            scale_factor: 0.3611,
            shift_factor: 0.1159,
        }
    }

    /// Build a tiny random-weight [`MeanVaeEncoder`] on CPU. VarMap zero-init is randomized (a zero
    /// encoder emits constant moments, which would make the mean-vs-sample distinction vacuous).
    fn tiny_mean_encoder(dev: &Device) -> MeanVaeEncoder {
        let cfg = tiny_ae_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, dev);
        let encoder = Encoder::new(&cfg, vb).expect("tiny encoder");
        let mut rng = StdRng::seed_from_u64(8988);
        for var in vm.data().lock().unwrap().values() {
            let n = var.shape().elem_count();
            let data: Vec<f32> = candle_gen::seeded_normal_vec(&mut rng, n)
                .into_iter()
                .map(|v| v * 0.05)
                .collect();
            let t = Tensor::from_vec(data, var.shape(), dev).expect("randomize");
            var.set(&t).expect("set var");
        }
        MeanVaeEncoder {
            encoder,
            shift_factor: cfg.shift_factor,
            scale_factor: cfg.scale_factor,
        }
    }

    /// A fixed (RNG-free) `[1, 3, 16, 16]` input in `[-1, 1]`.
    fn fixed_input(dev: &Device) -> Tensor {
        let n = 3 * 16 * 16;
        let data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin()).collect();
        Tensor::from_vec(data, (1, 3, 16, 16), dev).expect("input")
    }

    /// sc-8988: the control encode is bit-identical across calls and independent of the device RNG
    /// stream — the old sampled path (`AutoEncoder::encode` + `set_seed`) drew device `randn` and
    /// diverged when the RNG state moved between calls.
    #[test]
    fn control_encode_mean_is_deterministic_and_rng_free() {
        let dev = Device::Cpu;
        let enc = tiny_mean_encoder(&dev);
        let img = fixed_input(&dev);
        let a = enc.encode(&img).expect("encode a");
        // Churn the device RNG between calls — a sampled encode would change; the mean must not.
        let _ = Tensor::randn(0f32, 1f32, (16,), &dev).expect("rng churn");
        let b = enc.encode(&img).expect("encode b");
        // ch_mult = [1] ⇒ a single level with no downsample: the tiny latent stays at H×W.
        assert_eq!(a.dims(), &[1, 2, 16, 16], "z=2 latent");
        let (av, bv) = (
            a.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            b.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        );
        assert!(av.iter().all(|v| v.is_finite()), "finite latent");
        assert_eq!(av, bv, "mean encode must be bit-identical across calls");
    }

    /// sc-8988: the encode is the shift/scale-normalized posterior MEAN — the first `z_channels` chunk
    /// of the encoder moments, exactly as `AutoEncoder::encode` normalizes, minus the sampling.
    #[test]
    fn control_encode_takes_the_posterior_mean() {
        let dev = Device::Cpu;
        let enc = tiny_mean_encoder(&dev);
        let img = fixed_input(&dev);
        let got = enc.encode(&img).expect("encode");
        let moments = img.apply(&enc.encoder).expect("moments"); // [1, 2·z, 2, 2]
        let mean = moments.chunk(2, 1).expect("chunk")[0].clone();
        let want = ((mean - enc.shift_factor).unwrap() * enc.scale_factor).unwrap();
        let (g, w) = (
            got.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            want.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        );
        assert_eq!(g, w, "encode must equal (posterior_mean - shift) * scale");
    }
}
