//! FLUX.2-dev **strict-pose ControlNet** provider (sc-7460, epic 6564) — the candle (Windows/CUDA)
//! sibling of mlx-gen-flux2's `flux2_dev_control` (sc-2292). Strict pose (and the union's
//! canny/depth/hed/mlsd/scribble/gray) on FLUX.2-dev via `alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union`,
//! a VACE-style control branch overlaid on the dev base DiT.
//!
//! **How it conditions:** the pose/union control image is VAE-encoded into the packed transformer
//! latent ([`Flux2Vae::encode_packed`]), packed to `[1, seq, 128]`, then concatenated with a zero
//! inpaint **mask** (4) + a zero **inpaint latent** (128) → the 260-ch control context (the union
//! ControlNet's pose-only channel layout). [`Flux2ControlTransformer`] runs the parity-proven dev DiT
//! plus the control branch ([`Flux2ControlBranch`]): per-block hints are computed once from the
//! post-embedder streams and added into the base image stream after base double blocks `[0, 2, 4, 6]`,
//! scaled by `control_scale` (dev README sweet spot 0.65–0.80). The control context is clean + constant
//! across the denoise (encoded once). dev is guidance-distilled — a single embedded-guidance forward,
//! no true-CFG / negative pass.
//!
//! Bespoke provider (NOT gen-core-registered), worker-invoked by name — the candle pattern for
//! conditioned surfaces (mirrors [`crate::edit_provider`] / the SDXL control providers). The dev base
//! is the 32B flagship, so it loads via the CPU-stage → quantize-onto-GPU path ([`crate::quant`]); the
//! ~8 GB bf16 control overlay loads dense on the device and quantizes in place. Determinism is the
//! candle-lane contract (sc-3673): the seeded CPU init noise reuses [`pipeline::create_noise`].

use std::path::{Path, PathBuf};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::runtime::CancelFlag;
use candle_gen::gen_core::sampling::TimestepConvention;
use candle_gen::gen_core::{Image, Progress, Quant};
use candle_gen::{CandleError, Result};

use crate::config::{Flux2Variant, DEFAULT_GUIDANCE_DEV, DEFAULT_STEPS_DEV, SIZE_MULTIPLE};
use crate::edit_provider::preprocess_ref;
use crate::text_encoder::Qwen3TextEncoder;
use crate::transformer::{Flux2ControlBranch, Flux2ControlTransformer, Flux2Transformer};
use crate::vae::Flux2Vae;
use crate::{pipeline, to_image, Pipeline};

/// Default `control_context_scale` — the dev Fun-Controlnet-Union README sweet spot is 0.65–0.80; the
/// mlx worker defaults to 0.75. Strong pose lock without over-constraining the base.
pub const DEFAULT_CONTROL_SCALE: f32 = 0.75;

/// Paths to the FLUX.2-dev control checkpoints: the dev snapshot dir (`text_encoder/`, `transformer/`,
/// `vae/`, `tokenizer/`) + the Fun-Controlnet-Union overlay (a single `.safetensors` file, e.g.
/// `FLUX.2-dev-Fun-Controlnet-Union-2602.safetensors`, or a dir holding it).
pub struct Flux2ControlPaths {
    /// FLUX.2-dev diffusers snapshot dir.
    pub root: PathBuf,
    /// The Fun-Controlnet-Union control checkpoint (`.safetensors` file or a dir containing it).
    pub control: PathBuf,
}

/// One FLUX.2-dev strict-pose control request. dev is guidance-distilled — `guidance` is the embedded
/// scalar (single forward, no negative prompt).
#[derive(Clone)]
pub struct Flux2ControlRequest {
    pub prompt: String,
    pub width: u32,
    pub height: u32,
    pub steps: usize,
    /// Embedded guidance scale (dev default ≈ 4.0).
    pub guidance: f32,
    /// `control_context_scale` — how strongly the control branch locks the base (≈ 0.65–0.80).
    pub control_scale: f32,
    pub seed: u64,
    /// Cooperative cancellation, checked before each denoise step (the engine contract).
    pub cancel: CancelFlag,
}

impl Default for Flux2ControlRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            width: 1024,
            height: 1024,
            steps: DEFAULT_STEPS_DEV as usize,
            guidance: DEFAULT_GUIDANCE_DEV,
            control_scale: DEFAULT_CONTROL_SCALE,
            seed: 0,
            cancel: CancelFlag::default(),
        }
    }
}

/// A loaded FLUX.2-dev control model: the Mistral text encoder + the dev DiT wrapped in its control
/// branch ([`Flux2ControlTransformer`]) + the VAE **with the encoder** (the control-image encode).
/// `generate` takes `&self` (no per-call mutation), so one load serves many renders.
pub struct Flux2Control {
    pipe: Pipeline,
    te: Qwen3TextEncoder,
    transformer: Flux2ControlTransformer,
    vae: Flux2Vae,
}

impl Flux2Control {
    /// Load the dev base (CPU-stage → quantize-onto-GPU when `quant` is set; dense otherwise — fixture
    /// only, the 32B does not fit dense), the Fun-Controlnet-Union control overlay (dense on the device
    /// → quantized in place), and the VAE with its encoder. `quant` (Q4/Q8) is required in practice for
    /// the real 32B weights.
    pub fn load(paths: &Flux2ControlPaths, quant: Option<Quant>) -> Result<Self> {
        let device = candle_gen::default_device()?;
        let pipe = Pipeline::load(Flux2Variant::Dev, quant, &paths.root, &device);

        // Base DiT + Mistral TE: the dev quant path stages them dense in CPU RAM and quantizes each
        // projection onto the GPU (the dense 32B never lands on the GPU); the dense path loads on-device.
        let (te, base) = match quant {
            Some(q) => {
                let cpu = Device::Cpu;
                let mut te =
                    Qwen3TextEncoder::new(&pipe.cfg, pipe.component_vb_on("text_encoder", &cpu)?)?;
                te.quantize(q, &device)?;
                let mut base =
                    Flux2Transformer::new(&pipe.cfg, pipe.component_vb_on("transformer", &cpu)?)?;
                base.quantize(q, &device)?;
                (te, base)
            }
            None => (
                Qwen3TextEncoder::new(&pipe.cfg, pipe.component_vb("text_encoder")?)?,
                Flux2Transformer::new(&pipe.cfg, pipe.component_vb("transformer")?)?,
            ),
        };

        // The control overlay is small (~8 GB bf16) and fits on the GPU; load it dense on-device and
        // quantize in place (the 260-ch `control_img_in` stays dense — 260 ∤ 32).
        let control_vb = control_var_builder(&paths.control, pipe.dtype, &device)?;
        let mut branch = Flux2ControlBranch::new(&pipe.cfg, control_vb)?;
        if let Some(q) = quant {
            branch.quantize(q, &device)?;
        }
        let transformer = Flux2ControlTransformer::new(base, branch);

        let vae = Flux2Vae::new_with_encoder(pipe.component_vb("vae")?)?;
        Ok(Self {
            pipe,
            te,
            transformer,
            vae,
        })
    }

    /// Generate one strict-pose-conditioned image. `control_image` is the pose/union skeleton (the
    /// worker pre-fits it to the render size; this re-resizes defensively).
    pub fn generate(
        &self,
        req: &Flux2ControlRequest,
        control_image: &Image,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        if req.cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }
        if !req.width.is_multiple_of(SIZE_MULTIPLE) || !req.height.is_multiple_of(SIZE_MULTIPLE) {
            return Err(CandleError::Msg(format!(
                "flux2 control: width/height must be multiples of {SIZE_MULTIPLE} (got {}x{})",
                req.width, req.height
            )));
        }
        if req.steps == 0 {
            return Err(CandleError::Msg("flux2 control: steps must be >= 1".into()));
        }

        let device = &self.pipe.device;
        let cfg = &self.pipe.cfg;

        // Prompt embeds (text-only Mistral) + the packed 260-ch control context are seed-independent:
        // encode once.
        let prompt_embeds = self.pipe.encode(&self.te, &req.prompt)?;
        let control_context = self.encode_control_context(control_image, req.width, req.height)?;

        let (lat_h, lat_w) = pipeline::latent_dims(req.width, req.height);
        let img_ids = pipeline::prepare_grid_ids(lat_h, lat_w);
        let txt_ids = pipeline::prepare_text_ids(cfg.max_sequence_length);

        // Curated sampler/scheduler routing (epic 7114 P4) — the same driver the txt2img/edit paths use.
        // No per-request sampler/scheduler knob, so this runs the default (`None`) euler over the native
        // empirical-mu schedule (the N1 no-op reproducing the legacy flow-match loop within tolerance).
        let mu = pipeline::compute_mu(pipeline::image_seq_len(req.width, req.height), req.steps);
        let (native, _timesteps) = pipeline::schedule(req.steps, req.width, req.height);
        let sigmas = candle_gen::resolve_flow_schedule(None, mu, req.steps, &native);

        let latents = pipeline::create_noise(cfg, req.seed, req.width, req.height, device)?;
        // The driver does cancel + progress + the integrator step. The control forward lives inside the
        // predict closure so a multi-eval solver re-runs it. dev is guidance-distilled: a single forward
        // feeding the embedded guidance scalar (no negative pass). FLUX.2 embeds σ×1000.
        let latents = candle_gen::run_flow_sampler(
            None,
            TimestepConvention::Sigma,
            &sigmas,
            latents,
            req.seed,
            &req.cancel,
            on_progress,
            |latents, sigma| -> Result<Tensor> {
                let ts = sigma * 1000.0;
                Ok(self.transformer.forward(
                    latents,
                    &prompt_embeds,
                    &img_ids,
                    &txt_ids,
                    ts,
                    Some(req.guidance),
                    &control_context,
                    req.control_scale,
                )?)
            },
        )?;

        on_progress(Progress::Decoding);
        let packed = pipeline::unpack_latents(&latents, req.width, req.height)?;
        let decoded = self.vae.decode_packed(&packed)?; // [1,3,H,W] in [-1,1]
        to_image(&decoded)
    }

    /// Build the packed control context `[1, seq, 260]` from the pose/union control image (the fork's
    /// `pipeline_flux2_control.py`): VAE-encode → 2×2 patchify → BN-normalize → pack (`control_latents`,
    /// 128), concatenated with a zero inpaint **mask** (4) + a zero **inpaint latent** (128). For pure
    /// pose (no inpaint image / mask) the fork's mask + inpaint latent are all-zero. `seq` equals the
    /// target latent sequence (built at the same `width`/`height`), so the control context aligns 1:1
    /// with the base image tokens.
    fn encode_control_context(&self, image: &Image, width: u32, height: u32) -> Result<Tensor> {
        let cfg = &self.pipe.cfg;
        let device = &self.pipe.device;
        let nchw = preprocess_ref(image, width, height, device, self.pipe.dtype)?;
        let packed = self.vae.encode_packed(&nchw)?; // [1, 128, H/16, W/16]
        let control_packed = pipeline::pack_nchw(&packed)?; // [1, seq, 128]
        let (_, seq, _) = control_packed.dims3()?;
        // Union pose-only layout: zero mask (in_channels / num_latent_channels = 128/32 = 4, the 2×2
        // patch) + zero inpaint latent (in_channels, 128) → concatenated on the channel axis = 260.
        let in_ch = cfg.in_channels;
        let mask_ch = in_ch / cfg.num_latent_channels;
        let mask = Tensor::zeros((1, seq, mask_ch), DType::F32, device)?;
        let inpaint = Tensor::zeros((1, seq, in_ch), DType::F32, device)?;
        let cc = Tensor::cat(&[&control_packed, &mask, &inpaint], 2)?;
        debug_assert_eq!(
            cc.dim(2)?,
            crate::transformer::CONTROL_IN_DIM,
            "control context must be 260ch"
        );
        Ok(cc)
    }
}

/// Open a VarBuilder over the Fun-Controlnet-Union checkpoint — a single `.safetensors` `File` or a
/// `Dir` containing the `.safetensors` shards — on `device` at `dtype`.
fn control_var_builder(path: &Path, dtype: DType, device: &Device) -> Result<VarBuilder<'static>> {
    let files: Vec<PathBuf> = if path.is_dir() {
        let mut f: Vec<PathBuf> = std::fs::read_dir(path)
            .map_err(|e| CandleError::Msg(format!("flux2 control: read {}: {e}", path.display())))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        f.sort();
        f
    } else {
        vec![path.to_path_buf()]
    };
    if files.is_empty() {
        return Err(CandleError::Msg(format!(
            "flux2 control: no .safetensors at {} (expected the FLUX.2-dev-Fun-Controlnet-Union \
             checkpoint)",
            path.display()
        )));
    }
    // SAFETY: mmap of read-only weight files; the standard candle loading path.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, device)? };
    Ok(vb)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The request defaults match the dev control production knobs (1024², 28 steps, guidance 4.0,
    /// control scale 0.75).
    #[test]
    fn request_defaults() {
        let r = Flux2ControlRequest::default();
        assert_eq!((r.width, r.height), (1024, 1024));
        assert_eq!(r.steps, DEFAULT_STEPS_DEV as usize);
        assert_eq!(r.guidance, DEFAULT_GUIDANCE_DEV);
        assert_eq!(r.control_scale, DEFAULT_CONTROL_SCALE);
        assert!(!r.cancel.is_cancelled());
    }
}
