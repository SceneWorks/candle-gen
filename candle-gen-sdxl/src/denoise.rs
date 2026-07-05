//! InstantID denoise loop + the SDXL conditioning/prior/control/decode helpers (sc-5491, epic 5480) —
//! the candle twin of the `denoise_ip_control` / `denoise_ip_multi_control` family in
//! `mlx-gen-sdxl::pipeline`. These tie the 2a–2d building blocks (the IP-Adapter cross-attn on the
//! UNet, the IdentityNet [`ControlNet`], the [`EulerAncestralSampler`]) into the per-step denoise the
//! `candle-gen-instantid` glue crate (phase 3) drives.
//!
//! **Two structural divergences from the MLX pipeline, both following the candle design already set in
//! the earlier phases:**
//!  1. **The IP face tokens live on the UNet, set once.** mlx threads `(tokens, scale)` into
//!     `forward_with_ip_control` every step; the candle UNet stores the decoupled K/V + tokens on each
//!     `CrossAttention` (2c) because the face tokens are constant across the denoise. So this loop's
//!     **precondition** is that the caller has called [`UNet2DConditionModel::set_ip_context`] once
//!     before it (and `install_ip_adapter` at load) — the loop itself only runs
//!     [`UNet2DConditionModel::forward_instantid`], which picks up whatever IP context is set (inert if
//!     cleared). One loop therefore serves IdentityNet-only, IdentityNet+OpenPose (multi-control), and
//!     (cleared IP + no controls) plain SDXL.
//!  2. **Determinism is a seeded CPU `StdRng`, threaded explicitly.** The caller seeds one `StdRng`
//!     per image and passes `&mut` it to [`seeded_prior`] (the init latents) and this loop (each
//!     ancestral step's noise) — one continuous stream, so generation is a pure function of
//!     `(seed, request)`, launch-portable (the sc-3673 contract the txt2img path already follows).
//!     mlx instead seeds its process-global RNG; see [`EulerAncestralSampler`].
//!
//! CFG batch convention matches the candle txt2img pipeline: **row 0 = uncond, row 1 = cond**, and
//! `eps = eps_uncond + cfg·(eps_cond − eps_uncond)` — so the caller must batch `conditioning` /
//! `pooled` / `time_ids` / the face tokens in that order.

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_transformers::models::stable_diffusion::vae::AutoEncoderKL;
use rand::rngs::StdRng;
use rand::SeedableRng;

use candle_gen::gen_core::runtime::CancelFlag;
use candle_gen::gen_core::sampling::DiscreteModelSampling;
use candle_gen::gen_core::{Image, Progress};
use candle_gen::{CandleError, LatentDecoder, Result};

use crate::pipeline::VAE_SCALE;
use crate::sampler::EulerAncestralSampler;
use crate::unet::{ControlNet, ControlResiduals, UNet2DConditionModel};

/// VAE spatial downscale (the latent is image/8 per side).
pub const SPATIAL_SCALE: u32 = 8;
/// Latent channel count.
pub const LATENT_CHANNELS: usize = 4;

/// The SDXL micro-conditioning `time_ids`, hardcoded `[512, 512, 0, 0, 512, 512]` per row — the
/// vendored `StableDiffusionXL.generate_latents` quirk InstantID inherits (it does NOT pass the real
/// original/crop/target sizes). `batch` rows, in `dtype` (so it concatenates with the f16 pooled text
/// embeds inside the UNet/ControlNet `add_embedding`).
pub fn text_time_ids(batch: usize, device: &Device, dtype: DType) -> Result<Tensor> {
    let row = [512.0f32, 512.0, 0.0, 0.0, 512.0, 512.0];
    let mut v = Vec::with_capacity(batch * 6);
    for _ in 0..batch {
        v.extend_from_slice(&row);
    }
    Ok(Tensor::from_vec(v, (batch, 6), device)?.to_dtype(dtype)?)
}

/// Draw `n` unit-normal f32 from the seeded `StdRng` stream and build an NCHW `[1, C, h, w]` tensor on
/// CPU (so the draw sequence is device- and launch-independent — sc-3673), then move to `device`. The
/// shared draw used by both the prior and each ancestral step.
fn draw_noise(rng: &mut StdRng, c: usize, h: usize, w: usize, device: &Device) -> Result<Tensor> {
    Ok(candle_gen::seeded_noise_nchw(rng, c, h, w, device)?)
}

/// Sample the prior latents `noise · σ_last · rsqrt(σ_last²+1)` for a `width × height` render: draw
/// unit-normal noise from the seeded `rng`, scale via [`EulerAncestralSampler::scale_prior_noise`], and
/// return NCHW `[1, 4, height/8, width/8]` cast to `dtype` (f16 for production). The caller seeds `rng`
/// once and reuses it for the per-step ancestral noise, so the whole render is a pure function of the
/// seed.
pub fn seeded_prior(
    sampler: &EulerAncestralSampler,
    rng: &mut StdRng,
    width: u32,
    height: u32,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let (lh, lw) = (
        (height / SPATIAL_SCALE) as usize,
        (width / SPATIAL_SCALE) as usize,
    );
    let noise = draw_noise(rng, LATENT_CHANNELS, lh, lw, device)?;
    Ok(sampler.scale_prior_noise(&noise)?.to_dtype(dtype)?)
}

/// Draw the curated **VE σ-space** prior latents `noise · σ_max` for a `width × height` render — the
/// launch-portable (sc-3673) seeded CPU draw the curated [`denoise_curated`] path starts from, vs the
/// ancestral [`seeded_prior`]'s `σ·rsqrt(σ²+1)` scaling. `seed` keys the draw; the result is f32 (the
/// σ-space math stays f32 — [`denoise_curated`] casts to the UNet compute dtype per eval). Returns NCHW
/// `[1, 4, height/8, width/8]` on `device`.
pub fn seeded_sigma_prior(
    seed: u64,
    width: u32,
    height: u32,
    sigma_max: f32,
    device: &Device,
) -> Result<Tensor> {
    let (lh, lw) = (
        (height / SPATIAL_SCALE) as usize,
        (width / SPATIAL_SCALE) as usize,
    );
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = draw_noise(&mut rng, LATENT_CHANNELS, lh, lw, device)?;
    Ok((noise * sigma_max as f64)?)
}

/// Preprocess a ControlNet control image (the InstantID kps / OpenPose skeleton) for the candle UNet:
/// normalize `[0,255] → [0,1]` (the diffusers control-image processor's `do_normalize=False`, NOT the
/// `[-1,1]` of a VAE init) and lay out **NCHW** `[1, 3, H, W]` f32 (candle conv order, vs mlx NHWC).
///
/// Requires the image already at the render size — the InstantID renderers (`draw_kps`,
/// `draw_bodypose`) emit at the target dims, so no resize is needed here; a general resizing
/// ControlNet preprocessor (arbitrary user control images) is the broader sc-5489 surface. A
/// mismatched size errors loudly rather than silently stretching.
pub fn preprocess_control_image(
    image: &Image,
    target_width: u32,
    target_height: u32,
    device: &Device,
) -> Result<Tensor> {
    let (iw, ih) = (image.width as usize, image.height as usize);
    if image.pixels.len() != iw * ih * 3 {
        return Err(CandleError::Msg(format!(
            "sdxl control image pixel buffer {} != {iw}x{ih}x3",
            image.pixels.len()
        )));
    }
    if (image.width, image.height) != (target_width, target_height) {
        return Err(CandleError::Msg(format!(
            "sdxl control image is {}x{} but the render is {target_width}x{target_height}; the \
             InstantID kps/pose renderers draw at the target size (resize is the sc-5489 general \
             ControlNet surface, not this path)",
            image.width, image.height
        )));
    }
    let data: Vec<f32> = image.pixels.iter().map(|&p| p as f32 / 255.0).collect();
    // HWC → CHW, batch 1.
    let hwc = Tensor::from_vec(data, (ih, iw, 3), device)?;
    Ok(hwc.permute((2, 0, 1))?.unsqueeze(0)?.contiguous()?)
}

/// Decode final latents `[1, 4, h, w]` to an RGB8 [`Image`], through the native VAE by default or —
/// when a `pid` decoder is supplied (epic 7840, sc-8373) — the super-resolving PiD student. Both
/// produce `[-1, 1]` pixels; the common tail is `x/2 + 0.5`, clamp, ×255 (the candle txt2img
/// post-process), reading the output size from the decoded tensor (never `latent·8`, since PiD emits a
/// larger `[1, 3, 4H, 4W]`).
///
/// **Latent convention (sc-7848 parity):** the native VAE path un-scales by [`VAE_SCALE`] (candle
/// de-scales in the pipeline, not inside `vae.decode`), while the PiD `sdxl` student was trained on the
/// **0.13025-normalized** latent — the scaled sampler output — so it receives `latents` unchanged.
/// This mirrors [`crate::pipeline::Pipeline::decode`] exactly.
///
/// The native VAE decode routes through the shared [`crate::pipeline::tiled_vae_decode`] (F-061 /
/// sc-9045), so every bespoke lane that calls this — the trainer preview, the IP / edit / InstantID
/// providers — gets the same sc-4987 budgeted VAE tiling the registered
/// [`crate::pipeline::Pipeline::decode`] path uses. The tiling only bounds peak VRAM on large latents
/// (>512² output); ≤512² decodes are byte-identical to the prior monolithic path. `pid = None` is
/// byte-identical to the pre-sc-8373 signature.
pub fn decode_image(
    vae: &AutoEncoderKL,
    latents: &Tensor,
    pid: Option<&dyn LatentDecoder>,
) -> Result<Image> {
    let img = match pid {
        // PiD decodes (and 4× super-resolves) the normalized latent directly — no VAE de-scale.
        Some(pid) => pid.decode(latents)?,
        None => {
            let unscaled = (latents / VAE_SCALE)?;
            crate::pipeline::tiled_vae_decode(vae, &unscaled)?
        }
    };
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

/// The UNet + sampler for one denoise run (borrowed from the loaded model).
pub struct Denoiser<'a> {
    /// The InstantID UNet — its IP face tokens must already be set
    /// ([`UNet2DConditionModel::set_ip_context`]) and the decoupled K/V installed
    /// ([`UNet2DConditionModel::install_ip_adapter`]).
    pub unet: &'a UNet2DConditionModel,
    /// The ancestral sampler (SDXL's `EulerAncestralSampler::sdxl()`).
    pub sampler: &'a EulerAncestralSampler,
}

/// One ControlNet branch for the denoise loop: the loaded branch + the precomputed (step-invariant)
/// conditioning embedding for its fixed control image ([`ControlNet::embed_cond`], computed once) + its
/// `conditioning_scale`. InstantID uses one (IdentityNet on the kps image) or two (+ OpenPose) of these.
///
/// **Batch contract:** `cond_embed` is added to `conv_in(x_unet)` inside [`ControlNet::forward`], so it
/// must be batched to match the UNet input — **batch 2 when CFG is on** (the same control replicated to
/// the uncond + cond rows; the IdentityNet conditions both). The caller CFG-replicates the control image
/// before [`ControlNet::embed_cond`]. `controlnet_encoder` (passed to the denoise fns) is likewise
/// CFG-batched.
pub struct ControlContext<'a> {
    pub controlnet: &'a ControlNet,
    pub cond_embed: Tensor,
    pub scale: f64,
}

/// Run the InstantID denoise with CFG and **multiple** ControlNet branches whose residuals are summed
/// (the diffusers `MultiControlNetModel` rule) before injection — the engine for InstantID pose mode
/// (`controls = [IdentityNet(kps), OpenPose(skeleton)]`). A single-element `controls` is bit-identical
/// to [`denoise_ip_control`]; an empty `controls` runs plain IP (or, with the IP context cleared, plain
/// SDXL).
///
/// `steps` is the `(t, t_prev)` schedule ([`EulerAncestralSampler::timesteps`]); `controlnet_encoder` is
/// the cross-attention conditioning every branch shares (the face tokens for InstantID — the vendored
/// pipeline passes the same `prompt_image_emb` to every sub-ControlNet). The face IP tokens are NOT a
/// parameter: they are already set on `d.unet` (the precondition above). Progress is emitted per step;
/// `cancel` is checked between steps; the ancestral noise comes from `rng`.
#[allow(clippy::too_many_arguments)]
pub fn denoise_ip_multi_control(
    d: &Denoiser,
    mut latents: Tensor,
    conditioning: &Tensor,
    pooled: &Tensor,
    time_ids: &Tensor,
    cfg: f64,
    steps: &[(f64, f64)],
    rng: &mut StdRng,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    controls: &[ControlContext],
    controlnet_encoder: &Tensor,
) -> Result<Tensor> {
    // A zero-step schedule is a no-op (img2img at strength ≤ 1/steps) — return the init latents,
    // never invoking the σ=0 ancestral step that would NaN. (txt2img always has steps ≥ 1.)
    if steps.is_empty() {
        return Ok(latents);
    }
    let cfg_on = cfg > 1.0;
    let total = steps.len() as u32;
    let device = latents.device().clone();
    let (lat_c, lat_h, lat_w) = {
        let (_, c, h, w) = latents.dims4()?;
        (c, h, w)
    };

    for (i, &(t, t_prev)) in steps.iter().enumerate() {
        if cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }
        // Euler-ancestral folds the input renormalization into its step (the sampler's `step` applies
        // `rsqrt(σ_prev²+1)`), so the UNet input is the raw latents — no `scale_model_input`. CFG runs
        // the cond + uncond rows in one batched forward.
        let x_unet = if cfg_on {
            Tensor::cat(&[&latents, &latents], 0)?
        } else {
            latents.clone()
        };

        // Sum each branch's (already `scale`'d) residuals — the MultiControlNet rule. One branch ⇒ the
        // single residual unchanged; zero ⇒ `None` (no injection). All branches share the face tokens
        // as their cross-attention conditioning (`controlnet_encoder`).
        let mut combined: Option<ControlResiduals> = None;
        for cc in controls {
            let res = cc.controlnet.forward(
                &x_unet,
                &cc.cond_embed,
                t,
                controlnet_encoder,
                pooled,
                time_ids,
                cc.scale,
            )?;
            combined = Some(match combined {
                None => res,
                Some(prev) => prev.add(&res)?,
            });
        }

        // The InstantID UNet forward: the `add_embedding` micro-conditioning + the decoupled IP branch
        // (from the set context) + the (summed) control residuals.
        let eps = match &combined {
            Some(r) => d.unet.forward_instantid(
                &x_unet,
                t,
                conditioning,
                pooled,
                time_ids,
                Some(r.down.as_slice()),
                Some(&r.mid),
            )?,
            None => {
                d.unet
                    .forward_instantid(&x_unet, t, conditioning, pooled, time_ids, None, None)?
            }
        };

        // Classifier-free guidance: row 0 = uncond, row 1 = cond (the candle txt2img convention).
        let eps = if cfg_on {
            let chunks = eps.chunk(2, 0)?;
            let (uncond, cond) = (&chunks[0], &chunks[1]);
            (uncond + ((cond - uncond)? * cfg)?)?
        } else {
            eps
        };

        // One continuing seeded stream → launch-portable determinism. The noise is unused at the final
        // step (σ_up = 0) but still drawn so the stream advances identically regardless of schedule.
        let noise = draw_noise(rng, lat_c, lat_h, lat_w, &device)?;
        latents = d.sampler.step(&eps, &latents, t, t_prev, &noise)?;
        on_progress(Progress::Step {
            current: i as u32 + 1,
            total,
        });
    }
    Ok(latents)
}

/// Run the InstantID denoise with CFG and a **single** ControlNet branch (the IdentityNet on the kps
/// image) — the non-pose InstantID path. A thin wrapper over [`denoise_ip_multi_control`].
#[allow(clippy::too_many_arguments)]
pub fn denoise_ip_control(
    d: &Denoiser,
    latents: Tensor,
    conditioning: &Tensor,
    pooled: &Tensor,
    time_ids: &Tensor,
    cfg: f64,
    steps: &[(f64, f64)],
    rng: &mut StdRng,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    control: &ControlContext,
    controlnet_encoder: &Tensor,
) -> Result<Tensor> {
    denoise_ip_multi_control(
        d,
        latents,
        conditioning,
        pooled,
        time_ids,
        cfg,
        steps,
        rng,
        cancel,
        on_progress,
        std::slice::from_ref(control),
        controlnet_encoder,
    )
}

/// Curated unified-sampler **conditioned** denoise (epic 7114, sc-7297) — the **additive** k-diffusion
/// alternative to InstantID's bespoke ancestral default and Kolors' bespoke leading-Euler default, the
/// candle twin of `mlx-gen-sdxl::denoise_curated`. Drives any curated [`gen_core::sampling::Solver`]
/// (`euler` / `euler_ancestral` / `heun` / `dpmpp_2m` / `dpmpp_sde` / `uni_pc` / `lcm` / `ddim`) over a
/// [`DiscreteModelSampling`] (ε-prediction) + a resolved σ schedule through the shared
/// [`candle_gen::run_curated_sampler`], threading the SAME dual conditioning the bespoke loops carry:
/// the summed ControlNet residuals (`controls`, the MultiControlNet rule) + the decoupled IP branch.
///
/// **The IP face/image tokens are NOT a parameter** — like [`denoise_ip_multi_control`], they are
/// preconditioned on `unet` ([`UNet2DConditionModel::set_ip_context`], set once before this call; inert
/// when cleared). The loop runs [`UNet2DConditionModel::forward_instantid`], which picks up whatever IP
/// context is set, so one function serves ControlNet-only, IP-only, dual (InstantID), and plain modes.
///
/// The latents live in raw k-diffusion VE σ-space (`x = ε·σ_max` at the start — built by the caller);
/// the UNet input is `x/√(σ²+1)` ([`DiscreteModelSampling::input_scale`]), cast to `dtype` (f16 for
/// InstantID, f32 for Kolors) inside the predict closure. The conditioning `timestep` is the nearest
/// training index ([`DiscreteModelSampling::timestep`]) — ComfyUI's behaviour for a discrete model under
/// a curated solver. The output is returned in `dtype`, ready for the caller's decode.
///
/// `conditioning` / `pooled` / `time_ids` / `controlnet_encoder` are already **CFG-batched** by the
/// caller (`[uncond, cond]`, the candle convention), as is each `ControlContext::cond_embed`; only the
/// single-row σ-space latent is CFG-replicated here. `controlnet_encoder` is the cross-attention
/// conditioning every branch shares (the face tokens for InstantID; the ControlNet's own text
/// projection for Kolors) — unused when `controls` is empty (`control_encoder=None` in mlx terms).
///
/// N1: selected only when the request names a curated sampler/scheduler; the bespoke ancestral /
/// leading-Euler defaults stay byte-exact (never entered here). Cancellation + progress route through
/// the driver's `denoise` callback; the ancestral / dpmpp_sde noise comes from the seeded `seed`.
#[allow(clippy::too_many_arguments)]
pub fn denoise_curated(
    unet: &UNet2DConditionModel,
    sampler_name: Option<&str>,
    ms: &DiscreteModelSampling,
    sigmas: &[f32],
    latents: Tensor,
    conditioning: &Tensor,
    pooled: &Tensor,
    time_ids: &Tensor,
    cfg: f64,
    dtype: DType,
    seed: u64,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    controls: &[ControlContext],
    controlnet_encoder: &Tensor,
) -> Result<Tensor> {
    let cfg_on = cfg > 1.0;
    let out = candle_gen::run_curated_sampler(
        sampler_name,
        ms,
        sigmas,
        latents,
        seed,
        cancel,
        on_progress,
        |x_in, timestep| -> Result<Tensor> {
            // `x_in` is the `1/√(σ²+1)`-scaled latent (f32 from the solver); cast to the UNet compute
            // dtype, then CFG-batch the single row to [uncond, cond].
            let x = x_in.to_dtype(dtype)?;
            let x_unet = if cfg_on {
                Tensor::cat(&[&x, &x], 0)?
            } else {
                x
            };
            let t = timestep as f64;

            // Sum each branch's (already `scale`'d) residuals — the MultiControlNet rule. One branch ⇒
            // the single residual; zero ⇒ `None` (no injection). All branches share `controlnet_encoder`.
            let mut combined: Option<ControlResiduals> = None;
            for cc in controls {
                let res = cc.controlnet.forward(
                    &x_unet,
                    &cc.cond_embed,
                    t,
                    controlnet_encoder,
                    pooled,
                    time_ids,
                    cc.scale,
                )?;
                combined = Some(match combined {
                    None => res,
                    Some(prev) => prev.add(&res)?,
                });
            }

            // The micro-conditioning forward + the decoupled IP branch (from the set context) + the
            // (summed) control residuals.
            let eps = match &combined {
                Some(r) => unet.forward_instantid(
                    &x_unet,
                    t,
                    conditioning,
                    pooled,
                    time_ids,
                    Some(r.down.as_slice()),
                    Some(&r.mid),
                )?,
                None => {
                    unet.forward_instantid(&x_unet, t, conditioning, pooled, time_ids, None, None)?
                }
            };

            // CFG combine: row 0 = uncond, row 1 = cond (the candle convention). Raw ε in f32 so the
            // `DiscreteModelSampling` x0 recombine + solver math stay f32.
            let eps = if cfg_on {
                let chunks = eps.chunk(2, 0)?;
                let (uncond, cond) = (&chunks[0], &chunks[1]);
                (uncond + ((cond - uncond)? * cfg)?)?
            } else {
                eps
            };
            Ok(eps.to_dtype(DType::F32)?)
        },
    )?;
    // Return in the compute dtype, ready for the caller's decode (matches the bespoke loops' latents).
    Ok(out.to_dtype(dtype)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unet::{BlockConfig, ControlNetConfig, UNet2DConditionModelConfig};
    use candle_core::Device;
    use candle_nn::{VarBuilder, VarMap};
    use rand::SeedableRng;

    /// A tiny SDXL-shaped UNet config: one basic + one cross-attn down block, cross-attn mid, mirrored
    /// up. Cross-attns: down1 (1) + mid (1) + up0 (2) = 4 (so the IP install consumes 4 pairs). Same
    /// shape the 2c `forward_instantid` test uses, so the InstantID forward is well-exercised here.
    fn unet_cfg() -> UNet2DConditionModelConfig {
        UNet2DConditionModelConfig {
            center_input_sample: false,
            flip_sin_to_cos: true,
            freq_shift: 0.,
            blocks: vec![
                BlockConfig {
                    out_channels: 32,
                    use_cross_attn: None,
                    attention_head_dim: 8,
                },
                BlockConfig {
                    out_channels: 64,
                    use_cross_attn: Some(1),
                    attention_head_dim: 8,
                },
            ],
            layers_per_block: 1,
            downsample_padding: 1,
            mid_block_scale_factor: 1.,
            norm_num_groups: 32,
            norm_eps: 1e-5,
            cross_attention_dim: 16,
            use_linear_projection: false,
        }
    }

    /// addition_time_embed_dim = 8, projection_input_dim = pooled(16) + time_ids_len(2)·8 = 32.
    const ADD_TIME_DIM: usize = 8;
    const PROJ_DIM: usize = 32;
    const CROSS_DIM: usize = 16;

    /// Build a tiny InstantID UNet (add_embedding loaded, 4 IP K/V pairs installed). Returns it ready
    /// for `set_ip_context`.
    fn build_unet(vb: VarBuilder, dev: &Device) -> UNet2DConditionModel {
        let mut unet = UNet2DConditionModel::new(vb.clone(), 4, 4, false, unet_cfg())
            .unwrap()
            .with_add_embedding(vb, ADD_TIME_DIM, PROJ_DIM)
            .unwrap();
        // inner = 64 for every cross-attn in this config (all blocks at 64 channels); K/V map
        // cross_attention_dim(16) → inner(64).
        let pair = || {
            (
                Tensor::randn(0f32, 1f32, (64, CROSS_DIM), dev).unwrap(),
                Tensor::randn(0f32, 1f32, (64, CROSS_DIM), dev).unwrap(),
            )
        };
        unet.install_ip_adapter(vec![pair(), pair(), pair(), pair()])
            .unwrap();
        unet
    }

    /// CFG-batched ([uncond, cond]) conditioning for a `latent`-sized render: text `[2, S, 16]`, pooled
    /// `[2, 16]`, time_ids `[2, 6→2]`, IP tokens `[2, 3, 16]`.
    struct Cond {
        text: Tensor,
        pooled: Tensor,
        time_ids: Tensor,
        ip_tokens: Tensor,
    }
    fn cond(dev: &Device) -> Cond {
        Cond {
            text: Tensor::randn(0f32, 1f32, (2, 5, CROSS_DIM), dev).unwrap(),
            pooled: Tensor::randn(0f32, 1f32, (2, CROSS_DIM), dev).unwrap(),
            time_ids: Tensor::randn(0f32, 1f32, (2, 2), dev).unwrap(),
            ip_tokens: Tensor::randn(0f32, 1f32, (2, 3, CROSS_DIM), dev).unwrap(),
        }
    }

    fn finite(t: &Tensor) -> bool {
        t.flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|v| v.is_finite())
    }
    fn vals(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    /// `text_time_ids` lays out `[512,512,0,0,512,512]` per row at the requested dtype/shape.
    #[test]
    fn text_time_ids_layout() {
        let dev = Device::Cpu;
        let t = text_time_ids(2, &dev, DType::F32).unwrap();
        assert_eq!(t.dims(), &[2, 6]);
        let v = vals(&t);
        assert_eq!(&v[..6], &[512., 512., 0., 0., 512., 512.]);
        assert_eq!(&v[6..], &[512., 512., 0., 0., 512., 512.]);
    }

    /// `seeded_prior` shape = `[1, 4, h/8, w/8]`, finite, and the SAME seed reproduces it bit-for-bit
    /// (the launch-portable determinism) while a different seed changes it.
    #[test]
    fn seeded_prior_shape_and_determinism() {
        let dev = Device::Cpu;
        let s = EulerAncestralSampler::sdxl();
        let mut r1 = StdRng::seed_from_u64(7);
        let p1 = seeded_prior(&s, &mut r1, 64, 48, &dev, DType::F32).unwrap();
        assert_eq!(p1.dims(), &[1, 4, 6, 8]); // 48/8=6, 64/8=8
        assert!(finite(&p1));
        let mut r2 = StdRng::seed_from_u64(7);
        let p2 = seeded_prior(&s, &mut r2, 64, 48, &dev, DType::F32).unwrap();
        assert_eq!(vals(&p1), vals(&p2));
        let mut r3 = StdRng::seed_from_u64(8);
        let p3 = seeded_prior(&s, &mut r3, 64, 48, &dev, DType::F32).unwrap();
        assert_ne!(vals(&p1), vals(&p3));
    }

    /// `preprocess_control_image`: `[0,255]→[0,1]`, NCHW `[1,3,H,W]`, and a wrong-size image errors.
    #[test]
    fn preprocess_control_normalizes_and_checks_size() {
        let dev = Device::Cpu;
        // 2×1 RGB: (255,255,255), (0,0,0).
        let img = Image {
            width: 2,
            height: 1,
            pixels: vec![255, 255, 255, 0, 0, 0],
        };
        let t = preprocess_control_image(&img, 2, 1, &dev).unwrap();
        assert_eq!(t.dims(), &[1, 3, 1, 2]);
        // CHW: channel 0 = [px0_r, px1_r] = [1.0, 0.0].
        let v = vals(&t);
        assert_eq!(v[0], 1.0);
        assert_eq!(v[1], 0.0);
        // Mismatched target size errors (no silent stretch).
        assert!(preprocess_control_image(&img, 4, 4, &dev).is_err());
        // Wrong buffer length errors.
        let bad = Image {
            width: 4,
            height: 4,
            pixels: vec![0u8; 8],
        };
        assert!(preprocess_control_image(&bad, 4, 4, &dev).is_err());
    }

    /// The denoise loop (no ControlNet): runs a few ancestral steps on a tiny InstantID UNet, preserves
    /// the latent shape, stays finite, is deterministic for a fixed seed, differs with CFG on vs off,
    /// and honors a pre-cancelled flag.
    #[test]
    fn denoise_loop_no_control() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let mut unet = build_unet(vb, &dev);
        let c = cond(&dev);
        unet.set_ip_context(Some(&c.ip_tokens), 0.8).unwrap();

        let sampler = EulerAncestralSampler::sdxl();
        let steps = sampler.timesteps(3, sampler.max_time());
        let d = Denoiser {
            unet: &unet,
            sampler: &sampler,
        };
        let latent0 = Tensor::randn(0f32, 1f32, (1, 4, 8, 8), &dev).unwrap();

        let run = |seed: u64, cfg: f64| -> Tensor {
            let mut rng = StdRng::seed_from_u64(seed);
            let cancel = CancelFlag::new();
            let mut prog = |_p: Progress| {};
            denoise_ip_multi_control(
                &d,
                latent0.clone(),
                &c.text,
                &c.pooled,
                &c.time_ids,
                cfg,
                &steps,
                &mut rng,
                &cancel,
                &mut prog,
                &[],
                &c.text,
            )
            .unwrap()
        };

        let a = run(11, 5.0);
        assert_eq!(a.dims(), &[1, 4, 8, 8]);
        assert!(finite(&a));
        // Determinism: same seed ⇒ identical.
        let b = run(11, 5.0);
        assert_eq!(vals(&a), vals(&b));
        // A different CFG scale (both batched, the real InstantID path) changes the result — the
        // `eps_uncond + cfg·(eps_cond − eps_uncond)` combine is actually applied.
        let strong = run(11, 9.0);
        assert!(vals(&a)
            .iter()
            .zip(vals(&strong).iter())
            .any(|(x, y)| (x - y).abs() > 1e-4));

        // A pre-cancelled flag stops before any step.
        let mut rng = StdRng::seed_from_u64(11);
        let cancel = CancelFlag::new();
        cancel.cancel();
        let mut prog = |_p: Progress| {};
        let err = denoise_ip_multi_control(
            &d,
            latent0.clone(),
            &c.text,
            &c.pooled,
            &c.time_ids,
            5.0,
            &steps,
            &mut rng,
            &cancel,
            &mut prog,
            &[],
            &c.text,
        );
        assert!(matches!(err, Err(CandleError::Canceled)));
    }

    /// The denoise loop WITH a single ControlNet branch whose geometry matches the UNet (same block
    /// config): the residual injection shapes line up (no error), and a positive `scale` changes the
    /// output vs `scale = 0` (the residuals actually reach the UNet). This is the CPU guard for the
    /// ControlNet→UNet residual integration before the GPU/real-weight validation (phase 5).
    #[test]
    fn denoise_loop_with_controlnet() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let mut unet = build_unet(vb.clone(), &dev);
        let c = cond(&dev);
        unet.set_ip_context(Some(&c.ip_tokens), 0.8).unwrap();

        // ControlNet over the SAME tiny UNet geometry, so its down residuals match the UNet's skips
        // 1:1. cond_block_out_channels length 4 ⇒ 3 stride-2 ⇒ 8× downsample (control 64² → latent 8²).
        let cn_cfg = ControlNetConfig {
            unet: unet_cfg(),
            addition_time_embed_dim: ADD_TIME_DIM,
            projection_class_embeddings_input_dim: PROJ_DIM,
            conditioning_channels: 3,
            cond_block_out_channels: vec![4, 8, 16, 32],
        };
        let cn = ControlNet::new(vb, &cn_cfg).unwrap();
        // A control image at the render size (64² for the 8² latent), CFG-batched (batch 2) to match
        // the CFG UNet input — the same kps control on both the uncond and cond rows.
        let control = Tensor::randn(0f32, 1f32, (2, 3, 64, 64), &dev).unwrap();
        let cond_embed = cn.embed_cond(&control).unwrap();

        let sampler = EulerAncestralSampler::sdxl();
        let steps = sampler.timesteps(2, sampler.max_time());
        let d = Denoiser {
            unet: &unet,
            sampler: &sampler,
        };
        let latent0 = Tensor::randn(0f32, 1f32, (1, 4, 8, 8), &dev).unwrap();

        let run = |scale: f64| -> Tensor {
            let cc = ControlContext {
                controlnet: &cn,
                cond_embed: cond_embed.clone(),
                scale,
            };
            let mut rng = StdRng::seed_from_u64(3);
            let cancel = CancelFlag::new();
            let mut prog = |_p: Progress| {};
            denoise_ip_control(
                &d,
                latent0.clone(),
                &c.text,
                &c.pooled,
                &c.time_ids,
                5.0,
                &steps,
                &mut rng,
                &cancel,
                &mut prog,
                &cc,
                &c.text, // the face tokens are the ControlNet cross-attn conditioning
            )
            .unwrap()
        };

        let active = run(0.9);
        assert_eq!(active.dims(), &[1, 4, 8, 8]);
        assert!(finite(&active));
        // scale = 0 ⇒ the zero-conv-scaled residuals vanish ⇒ the control has no effect.
        let inactive = run(0.0);
        assert!(
            vals(&active)
                .iter()
                .zip(vals(&inactive).iter())
                .any(|(x, y)| (x - y).abs() > 1e-4),
            "a positive ControlNet scale must change the denoise output"
        );
    }

    /// The curated conditioned denoise (sc-7297): drive a `DiscreteModelSampling` over a tiny σ schedule
    /// with a single ControlNet branch + the set IP context. Preserves the latent shape, stays finite,
    /// is deterministic for a fixed seed, and a positive ControlNet scale changes the output vs `0`
    /// (the residuals reach the UNet through the curated path too). The CPU guard for the
    /// `run_curated_sampler` wiring before the GPU/real-weight smoke (phase 5).
    #[test]
    fn denoise_curated_loop_with_controlnet() {
        use candle_gen::gen_core::sampling::{AlphaSchedule, DiscreteModelSampling};

        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let mut unet = build_unet(vb.clone(), &dev);
        let c = cond(&dev);
        unet.set_ip_context(Some(&c.ip_tokens), 0.8).unwrap();

        let cn_cfg = ControlNetConfig {
            unet: unet_cfg(),
            addition_time_embed_dim: ADD_TIME_DIM,
            projection_class_embeddings_input_dim: PROJ_DIM,
            conditioning_channels: 3,
            cond_block_out_channels: vec![4, 8, 16, 32],
        };
        let cn = ControlNet::new(vb, &cn_cfg).unwrap();
        let control = Tensor::randn(0f32, 1f32, (2, 3, 64, 64), &dev).unwrap();
        let cond_embed = cn.embed_cond(&control).unwrap();

        // A short discrete σ schedule (trailing 0) over the SDXL ε contract.
        let sched = AlphaSchedule::scaled_linear(1000, 0.00085, 0.012);
        let ms = DiscreteModelSampling::sdxl(&sched);
        let sigmas = vec![8.0_f32, 4.0, 2.0, 1.0, 0.5, 0.0];
        // VE σ-space prior: unit noise · σ_max.
        let prior =
            (Tensor::randn(0f32, 1f32, (1, 4, 8, 8), &dev).unwrap() * sigmas[0] as f64).unwrap();

        let run = |scale: f64| -> Tensor {
            let cc = ControlContext {
                controlnet: &cn,
                cond_embed: cond_embed.clone(),
                scale,
            };
            let cancel = CancelFlag::new();
            let mut prog = |_p: Progress| {};
            denoise_curated(
                &unet,
                Some("euler"),
                &ms,
                &sigmas,
                prior.clone(),
                &c.text,
                &c.pooled,
                &c.time_ids,
                5.0,
                DType::F32,
                3,
                &cancel,
                &mut prog,
                std::slice::from_ref(&cc),
                &c.text, // the face/control cross-attn conditioning
            )
            .unwrap()
        };

        let active = run(0.9);
        assert_eq!(active.dims(), &[1, 4, 8, 8]);
        assert!(finite(&active));
        // Determinism: same seed ⇒ identical.
        assert_eq!(vals(&active), vals(&run(0.9)));
        // scale = 0 ⇒ the zero-conv-scaled residuals vanish ⇒ the control has no effect.
        let inactive = run(0.0);
        assert!(
            vals(&active)
                .iter()
                .zip(vals(&inactive).iter())
                .any(|(x, y)| (x - y).abs() > 1e-4),
            "a positive ControlNet scale must change the curated denoise output"
        );
    }

    /// A pre-cancelled flag stops the curated denoise before any model eval (the driver bridges
    /// `gen_core::Error::Canceled` ⇄ `CandleError::Canceled`).
    #[test]
    fn denoise_curated_honors_cancel() {
        use candle_gen::gen_core::sampling::{AlphaSchedule, DiscreteModelSampling};

        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let mut unet = build_unet(vb, &dev);
        let c = cond(&dev);
        unet.set_ip_context(Some(&c.ip_tokens), 0.8).unwrap();

        let sched = AlphaSchedule::scaled_linear(1000, 0.00085, 0.012);
        let ms = DiscreteModelSampling::sdxl(&sched);
        let sigmas = vec![8.0_f32, 4.0, 0.0];
        let prior = Tensor::randn(0f32, 1f32, (1, 4, 8, 8), &dev).unwrap();
        let cancel = CancelFlag::new();
        cancel.cancel();
        let mut prog = |_p: Progress| {};
        let err = denoise_curated(
            &unet,
            Some("euler"),
            &ms,
            &sigmas,
            prior,
            &c.text,
            &c.pooled,
            &c.time_ids,
            5.0,
            DType::F32,
            3,
            &cancel,
            &mut prog,
            &[],
            &c.text,
        );
        assert!(matches!(err, Err(CandleError::Canceled)));
    }
}
