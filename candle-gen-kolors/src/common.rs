//! Shared Kolors pipeline scaffolding (sc-9001 / F-021) — the numerics that were copy-pasted verbatim
//! across the three Kolors entry points: txt2img ([`crate::pipeline::Pipeline`]), pose-control
//! ([`crate::control::KolorsControl`]) and IP-Adapter ([`crate::ip_provider::IpAdapterKolors`]).
//!
//! Before this module the `time_ids`, initial-noise, decode, CFG-batched-encode and curated-σ-prior
//! blocks lived three times each (`control.rs:386-433`, `ip_provider.rs:370-429` vs
//! `pipeline.rs:305-351`); the routing drift among them was the root cause of the crate's one
//! behavioral bug (sc-8984 / F-004). Extracting them here gives ONE home for the shared numerics while
//! every genuine per-entry-point difference — the control branch, the IP image tokens, the ChatGLM3
//! encode plumbing, and the CFG combine — stays explicit at the call site (passed as params / a
//! closure), so nothing is flattened into a single "does-everything" driver.
//!
//! These are free functions taking an explicit `&Device` / `&AutoEncoderKL` rather than methods on a
//! shared struct, so the IP provider (whose `generate` is `&mut self` for `set_ip_context`) can call
//! them from inside its disjoint-field borrow region without fighting the borrow checker.

use candle_gen::candle_core::{DType, Device, IndexOp, Tensor};
use candle_gen::gen_core::sampling::{schedule_sigmas, DiscreteModelSampling, Scheduler};
use candle_gen::gen_core::Image;
use candle_gen::{CandleError, LatentDecoder, Result};
use candle_gen_pid::PidDecoder;
use candle_transformers::models::stable_diffusion::vae::AutoEncoderKL;
use rand::{rngs::StdRng, SeedableRng};

use crate::pipeline::{kolors_alpha_schedule, VAE_SCALE};

/// Reject `steps == 0` loudly instead of the silent 1-step render it would otherwise produce: the
/// curated unified-sampler path feeds `req.steps` into gen-core `schedule_sigmas`, which clamps
/// `steps.max(1)` — so an explicit 0 silently becomes a single-step decode of near-pure noise (the
/// native `KolorsEulerSampler` already errors, but the curated branch does not). A fast typed error
/// mirrors the sibling bespoke lanes (`reject_zero_steps` in sdxl-IP / scail2 / instantid, sc-9016,
/// F-032); these worker-driven Kolors paths have no gen-core capability floor upstream. Shared by
/// both the ControlNet and IP-Adapter entry points so they can't drift (sc-11182, F-102).
pub(crate) fn reject_zero_steps(id: &str, steps: usize) -> Result<()> {
    if steps == 0 {
        return Err(CandleError::Msg(format!(
            "{id}: steps must be >= 1 (an explicit 0 renders undenoised noise)"
        )));
    }
    Ok(())
}

/// The SDXL micro-conditioning `time_ids` = `(H, W, 0, 0, H, W)` per row, f32 `[batch, 6]` (the Kolors
/// txt2img value — original == target, no crop). Shared verbatim by all three entry points.
pub(crate) fn build_time_ids(
    device: &Device,
    batch: usize,
    height: u32,
    width: u32,
) -> Result<Tensor> {
    let (hf, wf) = (height as f32, width as f32);
    let row = [hf, wf, 0.0, 0.0, hf, wf];
    let mut v = Vec::with_capacity(batch * 6);
    for _ in 0..batch {
        v.extend_from_slice(&row);
    }
    Ok(Tensor::from_vec(v, (batch, 6), device)?)
}

/// sc-3673 deterministic, launch-portable initial noise `[1, 4, lat_h, lat_w]`: N(0,1) from a
/// fixed-algorithm CPU RNG (`StdRng`, ChaCha) seeded by `seed`, moved to the device. Shared verbatim by
/// all three entry points so a seed reproduces the same noise regardless of the conditioning lane.
pub(crate) fn initial_noise(
    device: &Device,
    seed: u64,
    lat_h: usize,
    lat_w: usize,
) -> Result<Tensor> {
    let n = 4 * lat_h * lat_w;
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = candle_gen::seeded_normal_vec(&mut rng, n);
    Ok(Tensor::from_vec(noise, (1, 4, lat_h, lat_w), &Device::Cpu)?.to_device(device)?)
}

/// Decode latents `[1, 4, H/8, W/8]` → an RGB8 [`Image`], either through the native SDXL VAE or — when
/// a PiD decoder resolved (epic 7840 / sc-7853, the `sdxl` student, reused because Kolors reuses the
/// SDXL VAE) — the super-resolving PiD student (emits a larger `[1,3,4H,4W]` tensor). Both yield
/// `[-1, 1]` pixels; [`to_image`] reads the size from the tensor. Shared by all three entry points
/// (control/IP pass `None`).
///
/// **Latent convention (sc-7848 parity — NOT zero-transform on candle):** the PiD `sdxl` student
/// trained on the **0.13025-normalized** latent (the scaled sampler output `latents`), so PiD gets
/// `latents` while the VAE gets the pipeline-de-scaled raw latent (`latents / VAE_SCALE`) — candle
/// de-scales here, not inside `vae.decode`, unlike the qwen/flux families. Matches MLX.
pub(crate) fn decode(
    vae: &AutoEncoderKL,
    pid: Option<&PidDecoder>,
    latents: &Tensor,
) -> Result<Image> {
    let img = match pid {
        Some(pid) => pid.decode(latents)?,
        None => vae.decode(&(latents / VAE_SCALE)?)?,
    };
    to_image(&img)
}

/// Convert a decoded pixel tensor `[1, 3, H, W]` in `[-1, 1]` → RGB8 [`Image`] (`x/2 + 0.5`, clamp,
/// ×255). Shared by the native VAE decode and the PiD super-resolving decode; the output size is read
/// from the tensor, never assumed (PiD may be 4× the VAE-native size).
pub(crate) fn to_image(img: &Tensor) -> Result<Image> {
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

/// CFG-batch a prompt / negative pair into `(context, pooled, batch)` using the caller's ChatGLM3
/// `encode` closure. The two Kolors bespoke providers ([`crate::control`] / [`crate::ip_provider`])
/// and txt2img all share this exact structure — encode the positive prompt, and under CFG encode the
/// negative and `cat([neg, pos], 0)` (uncond-first, the Kolors convention). Without guidance only the
/// positive branch is built and `batch == 1`.
///
/// The ChatGLM3 encode itself stays at the call site (as `encode`) so each site keeps its exact
/// tokenizer/encoder plumbing — txt2img threads `Components`, the two providers borrow `&self` fields.
/// This helper owns ONLY the CFG concat convention, which was identical across the three.
pub(crate) fn cfg_batch_context<F>(
    prompt: &str,
    negative: &str,
    use_guide: bool,
    mut encode: F,
) -> Result<(Tensor, Tensor, usize)>
where
    F: FnMut(&str) -> Result<(Tensor, Tensor)>,
{
    let (pos_ctx, pos_pooled) = encode(prompt)?;
    if use_guide {
        let (neg_ctx, neg_pooled) = encode(negative)?;
        let context = Tensor::cat(&[&neg_ctx, &pos_ctx], 0)?;
        let pooled = Tensor::cat(&[&neg_pooled, &pos_pooled], 0)?;
        Ok((context, pooled, 2))
    } else {
        Ok((pos_ctx, pos_pooled, 1))
    }
}

/// The shared curated-path σ/prior setup (epic 7114, sc-7124/sc-7297) — identical across the three
/// entry points: build the Kolors ε-prediction [`DiscreteModelSampling`] over the `scaled_linear`
/// schedule, resolve the σ-table from the requested `scheduler` (defaulting to ComfyUI `normal`), and
/// lift the launch-portable seeded noise into VE σ-space (`noise · σ_max`).
///
/// Returned as an owned handle: the [`DiscreteModelSampling`] (`sdxl` copies out of the schedule, so
/// it owns its data), the resolved σ-table, and the VE-σ prior — all ready to feed
/// `candle_gen::run_curated_sampler` / `candle_gen_sdxl::denoise_curated` directly.
pub(crate) struct CuratedSetup {
    pub(crate) model_sampling: DiscreteModelSampling,
    pub(crate) sigmas: Vec<f32>,
    pub(crate) prior: Tensor,
}

impl CuratedSetup {
    /// Build the [`DiscreteModelSampling`] + σ-table + VE-σ prior for `steps` at the given `scheduler`,
    /// from the shared launch-portable seeded `noise` (already `[1, 4, lat_h, lat_w]`). The prior is
    /// `noise · σ_max`.
    pub(crate) fn new(scheduler: Option<&str>, steps: usize, noise: &Tensor) -> Result<Self> {
        let sched = kolors_alpha_schedule()?;
        let model_sampling = DiscreteModelSampling::sdxl(&sched);
        // Native curated schedule = ComfyUI's default (`normal`); the scheduler axis overrides it.
        let native = schedule_sigmas(Scheduler::Normal, &model_sampling, steps);
        let sigmas = candle_gen::resolve_schedule(scheduler, &model_sampling, steps, &native);
        // VE prior: unit noise · σ_max (sigmas[0]); kept f32 through the sampler.
        let prior = (noise * sigmas[0] as f64)?;
        Ok(Self {
            model_sampling,
            sigmas,
            prior,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu() -> Device {
        Device::Cpu
    }

    /// `steps == 0` is a fast typed error on BOTH Kolors bespoke lanes (it would otherwise clamp to a
    /// silent 1-step render on the curated path); a valid step count passes (sc-11182, F-102).
    #[test]
    fn reject_zero_steps_floors_both_lanes() {
        let err = reject_zero_steps("kolors control", 0).expect_err("steps==0 must be rejected");
        assert!(err.to_string().contains("steps must be >= 1"), "{err}");
        let err = reject_zero_steps("kolors ip-adapter", 0).expect_err("steps==0 must be rejected");
        assert!(err.to_string().contains("kolors ip-adapter"), "{err}");
        assert!(reject_zero_steps("kolors control", 1).is_ok());
        assert!(reject_zero_steps("kolors ip-adapter", 30).is_ok());
    }

    #[test]
    fn time_ids_rows_are_h_w_0_0_h_w() {
        let d = cpu();
        let t = build_time_ids(&d, 2, 1024, 768).unwrap();
        assert_eq!(t.dims(), &[2, 6]);
        let v = t.to_vec2::<f32>().unwrap();
        // Every row = (H, W, 0, 0, H, W); both CFG rows identical.
        assert_eq!(v[0], vec![1024.0, 768.0, 0.0, 0.0, 1024.0, 768.0]);
        assert_eq!(v[1], v[0]);
        // batch == 1 keeps a single row.
        let one = build_time_ids(&d, 1, 512, 512).unwrap();
        assert_eq!(one.dims(), &[1, 6]);
    }

    #[test]
    fn initial_noise_shape_and_seed_determinism() {
        let d = cpu();
        let a = initial_noise(&d, 42, 16, 24).unwrap();
        assert_eq!(a.dims(), &[1, 4, 16, 24]);
        // Same seed ⇒ byte-identical noise (launch-portable, sc-3673).
        let b = initial_noise(&d, 42, 16, 24).unwrap();
        assert_eq!(
            a.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            b.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
        // A different seed diverges.
        let c = initial_noise(&d, 43, 16, 24).unwrap();
        assert_ne!(
            a.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            c.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
    }

    #[test]
    fn cfg_batch_context_doubles_under_guidance() {
        let d = cpu();
        // A stand-in "encode": prompt → distinct constant tensors so we can see the concat order.
        let enc = |p: &str| -> Result<(Tensor, Tensor)> {
            let val = if p == "pos" { 1.0f32 } else { 2.0f32 };
            let ctx = Tensor::full(val, (1, 256, 4096), &d)?;
            let pooled = Tensor::full(val, (1, 4096), &d)?;
            Ok((ctx, pooled))
        };

        // With guidance: batch == 2, uncond-first (neg row before pos row).
        let (ctx, pooled, batch) = cfg_batch_context("pos", "neg", true, enc).unwrap();
        assert_eq!(batch, 2);
        assert_eq!(ctx.dims(), &[2, 256, 4096]);
        assert_eq!(pooled.dims(), &[2, 4096]);
        // Row 0 is the negative (2.0), row 1 the positive (1.0).
        let p = pooled.to_vec2::<f32>().unwrap();
        assert_eq!(p[0][0], 2.0);
        assert_eq!(p[1][0], 1.0);

        // Without guidance: batch == 1, positive-only.
        let (ctx1, _pooled1, batch1) = cfg_batch_context("pos", "neg", false, enc).unwrap();
        assert_eq!(batch1, 1);
        assert_eq!(ctx1.dims(), &[1, 256, 4096]);
    }

    #[test]
    fn curated_setup_prior_scales_noise_by_sigma_max() {
        let d = cpu();
        let noise = initial_noise(&d, 7, 8, 8).unwrap();
        let setup = CuratedSetup::new(None, 20, &noise).unwrap();
        // sigmas descend from σ_max to (near) 0 (length is steps + 1, trailing 0).
        assert_eq!(setup.sigmas.len(), 21);
        assert!(setup.sigmas[0] > setup.sigmas[setup.sigmas.len() - 1]);
        // prior == noise · sigmas[0], elementwise.
        let expected = (&noise * setup.sigmas[0] as f64).unwrap();
        assert_eq!(
            setup.prior.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            expected.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
        // The owned DiscreteModelSampling is carried on the handle.
        let _ms = &setup.model_sampling;
    }
}
