//! SANA text-to-image sampling pipeline (epic 11776, story sc-11780 — **the candle-gen half**).
//!
//! Composes the three already-merged native SANA components into one end-to-end prompt→image path,
//! the Windows/CUDA + Linux sibling of `mlx-gen-sana::pipeline` (mlx sc-8489):
//!
//! ```text
//!  prompt ─▶ SanaTextEncoder (sc-11779: CHI → gemma-2-2b-it last-hidden) ─▶ [1, 300, 2304]
//!         ─▶ SanaTransformer  (sc-11778: Linear-DiT trunk, velocity prediction) ─▶ [1, 32, h, w]
//!         ─▶ DcAeDecoder      (sc-11777: DC-AE f32c32 decode)                   ─▶ [1, 3, 1024, 1024]
//! ```
//!
//! driven by the **unified flow-matching scheduler** (epic 7114): the σ schedule is built by
//! [`gen_core::sampling::build_flow_sigmas`] and integrated by [`candle_gen::run_flow_sampler`] — the
//! SAME machinery the sibling candle flow-match families use (`candle-gen-z-image`, `candle-gen-sd3`).
//! No bespoke scheduler.
//!
//! ## Sampler / shift / timestep convention (mirrored from `mlx-gen-sana::pipeline`)
//!
//! * **Flow-match Euler, static shift 3.0.** `Sana_1600M_1024px_diffusers` ships a
//!   `FlowMatchEulerDiscreteScheduler` with `shift = 3.0` and `use_dynamic_shifting = false`, so the
//!   native schedule is `build_flow_sigmas(steps, ln(3))` (resolution-independent, `exp(mu) = shift`).
//!   An unset `scheduler` keeps that byte-exact; a curated epic-7114 name re-shapes σ over the same
//!   `mu = ln(3)` via [`candle_gen::resolve_flow_schedule`].
//! * **Timestep convention.** The unified sampler hands the predict closure `ms.timestep(σ) = σ`
//!   ([`TimestepConvention::Sigma`]); the SANA trunk embeds the diffusers-scale timestep `σ · 1000`
//!   (`num_train_timesteps`), so the closure scales it before the forward (identical to SD3's MMDiT).
//!   The Euler update itself stays in σ-space (`x += (σ_{t+1} − σ_t) · v`).
//!
//! ## CFG
//!
//! Base SANA is a **true-CFG** model. Each step runs the trunk TWICE — cond (prompt) + uncond
//! (negative/empty prompt) — and combines `pred = uncond + scale · (cond − uncond)` (diffusers
//! `SanaPipeline.__call__` default `guidance_scale = 4.5`). When `guidance_scale <= 1.0` the uncond
//! forward is skipped (CFG off), matching diffusers' `do_classifier_free_guidance = guidance_scale > 1.0`.
//!
//! ## DC-AE latent scaling
//!
//! diffusers `SanaPipeline` decodes `latents / vae.config.scaling_factor` (the DC-AE
//! `scaling_factor = 0.41407`, [`DcAeConfig::scaling_factor`]); [`DcAeDecoder::decode`] expects the
//! **already-unscaled** latent, so the division is applied here before decode. The decoder emits NCHW
//! `[1, 3, H, W]` in `[-1, 1]`, mapped to RGB8 (`clip(x·0.5 + 0.5)·255`).

use std::path::{Path, PathBuf};

use candle_gen::candle_core::{DType, Device, IndexOp, Tensor};
use candle_gen::gen_core::sampling::{build_flow_sigmas, TimestepConvention};
use candle_gen::gen_core::{CancelFlag, Image, Progress};
use candle_gen::{resolve_flow_schedule, run_flow_sampler, CandleError, Result, Weights};
use candle_gen_pid::{Gemma2, Gemma2Config};

use crate::config::{DcAeConfig, SanaTransformerConfig};
use crate::dc_ae::DcAeDecoder;
use crate::text_encoder::SanaTextEncoder;
use crate::transformer::SanaTransformer;

/// DC-AE f32c32 latent channel count (the SANA trunk's `out_channels`).
pub const LATENT_CHANNELS: usize = 32;
/// DC-AE deep-compression spatial downsample (latent edge is image/32).
pub const SPATIAL_SCALE: u32 = 32;
/// diffusers `num_train_timesteps` — the SANA trunk embeds `sigma * 1000`.
pub const NUM_TRAIN_TIMESTEPS: f32 = 1000.0;
/// SANA-1.6B static flow-match shift (`scheduler_config.json` `shift = 3.0`, no dynamic shifting).
pub const SCHEDULE_SHIFT: f32 = 3.0;
/// diffusers `SanaPipeline` default `num_inference_steps`.
pub const DEFAULT_STEPS: usize = 20;
/// diffusers `SanaPipeline` default `guidance_scale`.
pub const DEFAULT_GUIDANCE: f32 = 4.5;

/// Seeded txt2img latent noise — shape `[1, 32, height/32, width/32]`, f32. diffusers
/// `randn_tensor([B, 32, H/32, W/32])`; we draw f32 on CPU (launch-portable, sc-3673) then move to
/// `device`. (`init_noise_sigma = 1.0` for flow-match, so the latent is the raw normal draw.)
pub fn create_noise(device: &Device, seed: u64, width: u32, height: u32) -> Result<Tensor> {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let (lh, lw) = (
        (height / SPATIAL_SCALE) as usize,
        (width / SPATIAL_SCALE) as usize,
    );
    Ok(candle_gen::seeded_noise_nchw(
        &mut rng,
        LATENT_CHANNELS,
        lh,
        lw,
        device,
    )?)
}

/// Build the descending flow-match σ schedule for SANA (static shift 3.0), honoring a curated
/// epic-7114 `scheduler` name (which re-shapes σ over the same `mu = ln(shift)`). An unset / unknown /
/// native-aliased name returns the byte-exact `build_flow_sigmas(steps, ln(3))` schedule.
pub fn sana_sigmas(scheduler_name: Option<&str>, steps: usize) -> Vec<f32> {
    let mu = SCHEDULE_SHIFT.ln();
    let native = build_flow_sigmas(steps, mu);
    resolve_flow_schedule(scheduler_name, mu, steps, &native)
}

/// One flow-match Euler denoise with **true CFG** + progress + cooperative cancellation. Each step
/// runs the SANA trunk twice (cond + uncond) and combines `uncond + scale·(cond − uncond)`; the Euler
/// step then advances the latents in σ-space. The trunk timestep is `σ·1000`. When `guidance_scale`
/// is `<= 1.0` the uncond branch is skipped (CFG off, one forward per step; diffusers parity).
#[allow(clippy::too_many_arguments)]
pub fn denoise_cfg(
    transformer: &SanaTransformer,
    sigmas: &[f32],
    sampler_name: Option<&str>,
    seed: u64,
    latents: Tensor,
    cond: &Tensor,
    uncond: Option<&Tensor>,
    guidance_scale: f32,
    device: &Device,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Tensor> {
    let predict = |x: &Tensor, timestep: f32| -> Result<Tensor> {
        // The unified flow sampler hands `timestep = σ`; the SANA trunk embeds `σ·1000`.
        let t = Tensor::from_vec(vec![timestep * NUM_TRAIN_TIMESTEPS], (1,), device)?;
        let pred_cond = transformer.forward(x, cond, &t)?;
        match uncond {
            Some(uc) if guidance_scale > 1.0 => {
                let pred_uncond = transformer.forward(x, uc, &t)?;
                // pred = uncond + scale·(cond − uncond).
                let delta = (&pred_cond - &pred_uncond)?;
                Ok((&pred_uncond + (delta * guidance_scale as f64)?)?)
            }
            _ => Ok(pred_cond),
        }
    };
    run_flow_sampler(
        sampler_name,
        TimestepConvention::Sigma,
        sigmas,
        latents,
        seed,
        cancel,
        on_progress,
        predict,
    )
}

/// DC-AE-decode the final `[1, 32, H/32, W/32]` latent → an RGB8 [`Image`]. diffusers `SanaPipeline`
/// divides by `vae.config.scaling_factor` before decode; the decoder emits NCHW `[1, 3, H, W]` in
/// `[-1, 1]`, mapped to `[0, 255]` u8.
pub fn decode_to_image(decoder: &DcAeDecoder, cfg: &DcAeConfig, latents: &Tensor) -> Result<Image> {
    let unscaled = (latents / cfg.scaling_factor as f64)?; // diffusers: latents / scaling_factor
    let decoded = decoder.decode(&unscaled)?; // [1, 3, H, W] NCHW, f32 in [-1, 1]
    let rgb = (((decoded * 0.5)? + 0.5)?.clamp(0f32, 1f32)? * 255.0)?;
    let rgb = rgb
        .round()?
        .to_dtype(DType::U8)?
        .i(0)?
        .to_device(&Device::Cpu)?; // [3, H, W]
    let (c, h, w) = rgb.dims3()?;
    if c != 3 {
        return Err(CandleError::Msg(format!("expected 3 channels, got {c}")));
    }
    let pixels = rgb.permute((1, 2, 0))?.flatten_all()?.to_vec1::<u8>()?;
    Ok(Image {
        width: w as u32,
        height: h as u32,
        pixels,
    })
}

/// The composed SANA text-to-image pipeline: text encoder + trunk + DC-AE decoder, with the DC-AE
/// config (for the latent `scaling_factor`). A clean `generate` entrypoint mirroring the sibling candle
/// flow-match pipelines. Base SANA-1.6B only (true-CFG flow-match Euler); the CFG-free SCM/Sprint
/// distill is not ported on the candle side.
pub struct SanaPipeline {
    text_encoder: SanaTextEncoder,
    transformer: SanaTransformer,
    decoder: DcAeDecoder,
    dc_ae_cfg: DcAeConfig,
}

/// One text-to-image request for [`SanaPipeline::generate`]. `None` fields fall back to the diffusers
/// `SanaPipeline` defaults (`steps = 20`, `guidance = 4.5`, `seed = 0`, empty negative prompt).
#[derive(Clone, Debug)]
pub struct SanaGenerateRequest<'a> {
    pub prompt: &'a str,
    pub negative_prompt: Option<&'a str>,
    pub height: u32,
    pub width: u32,
    pub steps: Option<usize>,
    pub guidance_scale: Option<f32>,
    pub seed: Option<u64>,
    /// Optional curated epic-7114 sampler name (e.g. `"euler"`, `"dpmpp_2m"`); `None` = native Euler.
    pub sampler: Option<&'a str>,
    /// Optional curated epic-7114 scheduler name re-shaping σ over the same `mu = ln(shift)`.
    pub scheduler: Option<&'a str>,
}

impl<'a> SanaGenerateRequest<'a> {
    /// A 1024px request for `prompt` with all diffusers defaults.
    pub fn new(prompt: &'a str) -> Self {
        Self {
            prompt,
            negative_prompt: None,
            height: 1024,
            width: 1024,
            steps: None,
            guidance_scale: None,
            seed: None,
            sampler: None,
            scheduler: None,
        }
    }
}

impl SanaPipeline {
    /// Compose the base SANA-1.6B pipeline from its three already-constructed components plus the
    /// DC-AE config (used for the latent `scaling_factor`).
    pub fn new(
        text_encoder: SanaTextEncoder,
        transformer: SanaTransformer,
        decoder: DcAeDecoder,
        dc_ae_cfg: DcAeConfig,
    ) -> Self {
        Self {
            text_encoder,
            transformer,
            decoder,
            dc_ae_cfg,
        }
    }

    /// Assemble the pipeline from an `Efficient-Large-Model/Sana_1600M_1024px_diffusers`-shaped
    /// snapshot directory (the whole-repo HF snapshot: `transformer/ vae/ text_encoder/ tokenizer/`).
    ///
    /// Everything runs **f32** (the parity precision + the dense-GEMM-safe path, matching the DC-AE
    /// decoder and the trunk's f32 forward): the transformer, DC-AE, and gemma-2-2b-it caption encoder
    /// are all coerced to f32 on load. The component file selection ([`resolve_component_files`]) picks
    /// the fp32 (non-`fp16`) safetensors and tolerates both single-file and sharded checkpoints, so the
    /// raw diffusers tree loads without a curated allow-list.
    pub fn from_diffusers_snapshot(root: &Path, device: &Device) -> Result<Self> {
        let trunk_files = resolve_component_files(&root.join("transformer"))?;
        let trunk_w = Weights::from_files(&trunk_files, device, DType::F32)?;
        let trunk = SanaTransformer::from_weights(&trunk_w, SanaTransformerConfig::sana_1600m())?;

        let dcfg = DcAeConfig::sana_f32c32();
        let vae_files = resolve_component_files(&root.join("vae"))?;
        let vae_w = Weights::from_files(&vae_files, device, DType::F32)?;
        let decoder = DcAeDecoder::from_weights(&vae_w, dcfg.clone())?;

        let te = load_text_encoder(root, device)?;

        Ok(Self::new(te, trunk, decoder, dcfg))
    }

    /// Run the full prompt→image pipeline with caller-supplied cancellation + progress (the seam the
    /// gen-core `Generator` adapter wires into the contract). Encodes the prompt (and the negative
    /// prompt when CFG is active) ONCE, seeds the DC-AE latent, runs the flow-match Euler denoise over
    /// the SANA trunk with true CFG, then DC-AE-decodes to an RGB8 [`Image`].
    pub fn generate_with(
        &self,
        req: &SanaGenerateRequest<'_>,
        device: &Device,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        let steps = req.steps.unwrap_or(DEFAULT_STEPS);
        let guidance = req.guidance_scale.unwrap_or(DEFAULT_GUIDANCE);
        let seed = req.seed.unwrap_or(0);

        // Conditioning is seed-independent — encode once. Cond = the prompt; uncond = the negative
        // prompt (empty string when unset), used only when CFG is active. diffusers gates CFG on
        // `do_classifier_free_guidance = guidance_scale > 1.0`.
        let cond = self.text_encoder.encode(req.prompt)?;
        let cfg_on = guidance > 1.0;
        let uncond = if cfg_on {
            let neg = req.negative_prompt.unwrap_or("");
            Some(self.text_encoder.encode(neg)?)
        } else {
            None
        };

        // Static shift=3.0 schedule (scheduler_config.json), resolution-independent. An unset scheduler
        // keeps it byte-exact; a curated name re-shapes σ over the same mu=ln(3).
        let sigmas = sana_sigmas(req.scheduler, steps);

        let latents = create_noise(device, seed, req.width, req.height)?;
        let latents = denoise_cfg(
            &self.transformer,
            &sigmas,
            req.sampler,
            seed,
            latents,
            &cond,
            uncond.as_ref(),
            guidance,
            device,
            cancel,
            on_progress,
        )?;
        on_progress(Progress::Decoding);
        decode_to_image(&self.decoder, &self.dc_ae_cfg, &latents)
    }

    /// Convenience [`SanaPipeline::generate_with`] with a no-op cancel + progress (examples / tests).
    pub fn generate(&self, req: &SanaGenerateRequest<'_>, device: &Device) -> Result<Image> {
        let cancel = CancelFlag::default();
        let mut noop = |_: Progress| {};
        self.generate_with(req, device, &cancel, &mut noop)
    }
}

/// Load the gemma-2-2b-it caption encoder from a diffusers SANA snapshot. The gemma **weights** live in
/// `text_encoder/` (fp32 shards) and the gemma **tokenizer** in `tokenizer/tokenizer.json` (the
/// `Sana_1600M_1024px_diffusers` layout), so we build [`SanaTextEncoder`] directly rather than via
/// [`SanaTextEncoder::from_snapshot`] (which expects the tokenizer co-located under the weights dir).
fn load_text_encoder(root: &Path, device: &Device) -> Result<SanaTextEncoder> {
    let te_files = resolve_component_files(&root.join("text_encoder"))?;
    let gw = Weights::from_files(&te_files, device, DType::F32)?;
    // The diffusers SANA `text_encoder/` saves the Gemma2Model UN-prefixed (`embed_tokens.weight`,
    // `layers.0.…`); PiD's `SceneWorks/gemma-2-2b-it` mirror wraps it under `model.`. Pick whichever
    // this snapshot uses so both layouts load.
    let prefix = if gw.contains("embed_tokens.weight") {
        ""
    } else {
        "model."
    };
    let gemma = Gemma2::from_weights(&gw, prefix, &Gemma2Config::gemma_2_2b())?;
    // Prefer the sibling `tokenizer/` dir (the diffusers layout); fall back to a co-located file.
    let tok = {
        let t1 = root.join("tokenizer").join("tokenizer.json");
        if t1.is_file() {
            t1
        } else {
            root.join("text_encoder").join("tokenizer.json")
        }
    };
    SanaTextEncoder::new(gemma, tok)
}

/// Whether a safetensors filename is a shard of a multi-file checkpoint (`…-00001-of-00002.safetensors`).
fn is_shard(name: &str) -> bool {
    name.contains("-of-")
}

/// Select the usable `.safetensors` files in a diffusers component dir. The raw
/// `Sana_1600M_1024px_diffusers` tree ships BOTH fp32 and `fp16` copies, and the transformer ships a
/// single-file AND a sharded fp32 copy — loading all of them would collide on duplicate keys. Policy:
///
/// - drop any `fp16` copy (we run f32 everywhere; `Weights::from_files` coerces on load anyway);
/// - if the dir holds a sharded checkpoint (`…-of-…`), use the shard set (the diffusers-native fp32
///   split); otherwise use the remaining single file(s).
///
/// Deterministically sorted, and a hard error if nothing usable is found.
pub fn resolve_component_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let rd = std::fs::read_dir(dir)
        .map_err(|e| CandleError::Msg(format!("read component dir {dir:?}: {e}")))?;
    let mut candidates = Vec::new();
    for entry in rd {
        let path = entry
            .map_err(|e| CandleError::Msg(format!("read entry in {dir:?}: {e}")))?
            .path();
        if path.extension().and_then(|e| e.to_str()) != Some("safetensors") {
            continue;
        }
        let name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_lowercase();
        // Drop fp16 copies (we load everything f32; keeps single-vs-sharded selection unambiguous).
        if name.contains("fp16") {
            continue;
        }
        candidates.push(path);
    }
    let sharded: Vec<PathBuf> = candidates
        .iter()
        .filter(|p| {
            p.file_name()
                .and_then(|s| s.to_str())
                .map(is_shard)
                .unwrap_or(false)
        })
        .cloned()
        .collect();
    let mut chosen = if sharded.is_empty() {
        candidates
    } else {
        sharded
    };
    if chosen.is_empty() {
        return Err(CandleError::Msg(format!(
            "no usable (non-fp16) .safetensors found in {dir:?}"
        )));
    }
    chosen.sort();
    Ok(chosen)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Device;

    #[test]
    fn noise_shape_is_batch1_32ch() {
        let dev = Device::Cpu;
        let n = create_noise(&dev, 0, 1024, 1024).unwrap();
        assert_eq!(n.dims(), &[1, 32, 32, 32]);
        let n = create_noise(&dev, 0, 512, 1024).unwrap();
        // width 512 → latent w = 16; height 1024 → latent h = 32.
        assert_eq!(n.dims(), &[1, 32, 32, 16]);
    }

    #[test]
    fn noise_is_seed_deterministic() {
        let dev = Device::Cpu;
        let a = create_noise(&dev, 7, 256, 256).unwrap();
        let b = create_noise(&dev, 7, 256, 256).unwrap();
        let c = create_noise(&dev, 8, 256, 256).unwrap();
        let v = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(v(&a), v(&b), "same seed reproduces");
        assert_ne!(v(&a), v(&c), "diff seed differs");
    }

    #[test]
    fn static_shift_schedule_matches_diffusers() {
        // SANA-1.6B: FlowMatchEulerDiscreteScheduler shift=3.0, no dynamic shifting. The native (unset
        // scheduler) path must reproduce the diffusers static-shift σ table exactly.
        let s = sana_sigmas(None, 4);
        let expected = [1.0_f32, 0.9, 0.75, 0.5, 0.0];
        assert_eq!(s.len(), 5);
        for (got, want) in s.iter().zip(expected) {
            assert!((got - want).abs() < 1e-5, "got {got} want {want}");
        }
    }

    #[test]
    fn curated_scheduler_reshapes_but_stays_descending_to_zero() {
        // A curated epic-7114 scheduler name re-shapes σ over the same mu=ln(3): still descending,
        // trailing 0, and distinct from the native ramp (so the knob has an effect).
        let native = sana_sigmas(None, 12);
        let karras = sana_sigmas(Some("karras"), 12);
        assert_eq!(*karras.last().unwrap(), 0.0);
        assert!(karras.windows(2).all(|w| w[0] >= w[1]));
        assert_ne!(karras, native);
    }

    #[test]
    fn resolve_component_files_prefers_shards_and_drops_fp16() {
        use std::fs::File;
        let dir = std::env::temp_dir().join(format!("sana_rcf_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // Mimic the diffusers transformer dir: single bf16 + fp32 shards + an fp16 copy + non-weights.
        for f in [
            "diffusion_pytorch_model.safetensors",
            "diffusion_pytorch_model-00001-of-00002.safetensors",
            "diffusion_pytorch_model-00002-of-00002.safetensors",
            "diffusion_pytorch_model.fp16.safetensors",
            "config.json",
            "diffusion_pytorch_model.safetensors.index.json",
        ] {
            File::create(dir.join(f)).unwrap();
        }
        let chosen = resolve_component_files(&dir).unwrap();
        let names: Vec<String> = chosen
            .iter()
            .map(|p| p.file_name().unwrap().to_str().unwrap().to_string())
            .collect();
        assert_eq!(
            names,
            vec![
                "diffusion_pytorch_model-00001-of-00002.safetensors".to_string(),
                "diffusion_pytorch_model-00002-of-00002.safetensors".to_string(),
            ],
            "shards win, single + fp16 dropped"
        );

        // A single-file component dir (the vae layout: one fp32 + one fp16) → the single fp32 file.
        let vdir = dir.join("vae");
        std::fs::create_dir_all(&vdir).unwrap();
        for f in [
            "diffusion_pytorch_model.safetensors",
            "diffusion_pytorch_model.fp16.safetensors",
        ] {
            File::create(vdir.join(f)).unwrap();
        }
        let chosen = resolve_component_files(&vdir).unwrap();
        assert_eq!(chosen.len(), 1);
        assert_eq!(
            chosen[0].file_name().unwrap().to_str().unwrap(),
            "diffusion_pytorch_model.safetensors"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
