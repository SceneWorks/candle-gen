//! Krea 2 text-to-image pipeline (sc-7580/sc-7582) — tokenize → Qwen3-VL-4B condition-encode (the
//! 12-layer select stack) → DiT (text_fusion aggregator + single-stream denoise) → Qwen-Image VAE
//! decode. Port of `mlx-gen-krea`'s `pipeline.rs` (the reference `sampling.py::sample`). Two render
//! surfaces share this one pipeline:
//! - **Turbo** ([`render`]) — the distilled few-step **CFG-free** path (one DiT forward/step).
//! - **Raw** ([`render_base`], sc-9994 / epic 9992) — the undistilled 12B DiT with **true
//!   classifier-free guidance** (two DiT forwards/step: cond vs uncond) + optional user negative
//!   prompt at 52 steps, resolution-dynamic mu ([`base_schedule`]). The Boogu base/turbo precedent.
//!
//! **CFG-free (Turbo).** The TDM distillation baked the guided velocity into the weights, so there is
//! no unconditional branch (`guidance == 0` in the reference) — one DiT forward per step. Per-sample
//! `B = 1`: one prompt → no padding → the DiT runs the full valid context.
//!
//! **Rectified-flow v-param Euler.** The DiT consumes the raw sigma as its timestep
//! ([`TimestepConvention::Sigma`]; it scales ×1000 internally) and predicts the flow velocity
//! directly, so the core [`candle_gen::run_flow_sampler`] Euler step `x + v·(σ_{i+1} − σ_i)` is exactly
//! the reference `img += (tprev − tcurr)·v`. The native exponential-mu schedule
//! ([`crate::schedule::turbo_sigmas`]) is the byte-exact default; a per-generation curated
//! sampler/scheduler (epic 7114) reshapes over the same mu.

use std::path::Path;
use std::sync::Arc;

use candle_gen::candle_core::{DType, Device, IndexOp, Tensor};
use candle_gen::gen_core::imageops::resize_bicubic_u8;
use candle_gen::gen_core::sampling::TimestepConvention;
use candle_gen::gen_core::{self, AdapterSpec, GenerationRequest, Image, PidWeights, Progress};
use candle_gen::{CandleError, LatentDecoder, Result};
use candle_gen_pid::PidEngine;
use candle_gen_qwen_image::vae::QwenVae;
use rand::{rngs::StdRng, SeedableRng};

/// The PiD backbone (latent-space) tag for Krea (epic 7840 / sc-7853). Krea reuses the Qwen-Image VAE,
/// so its latent space is `qwenimage` — the same `2kto4k` 4× student Qwen-Image resolves.
const PID_BACKBONE: &str = "qwenimage";

use crate::config::Krea2Config;
use crate::loader::Weights;
use crate::schedule::{dynamic_mu, krea_sigmas, turbo_sigmas, TURBO_MU, TURBO_STEPS};
use crate::text_encoder::{KreaTeConfig, KreaTextEncoder};
use crate::transformer::Krea2Transformer;
use crate::vae::{load_vae, load_vae_encoder, QwenVaeEncoder};

/// Component compute dtypes. The Qwen3-VL TE runs in **f32** (parity-grade for this encoder, shared
/// with the ideogram/boogu ports); the 12B DiT runs **bf16** (native on candle's CUDA backend); the
/// Qwen-Image VAE runs **f32** (decode-precision-sensitive).
const TE_DTYPE: DType = DType::F32;
const DIT_DTYPE: DType = DType::BF16;

/// VAE spatial downscale (the latent is image/8 per side) and latent channel count.
const SPATIAL_SCALE: u32 = 8;
const LATENT_CHANNELS: usize = 16;

/// Raw (undistilled, full-CFG) generation defaults — the reference `sampling.py` Raw preset (sc-7566
/// spike), mirroring mlx-gen-krea `DEFAULT_RAW_STEPS` / `DEFAULT_RAW_GUIDANCE` (sc-9994). 52 steps,
/// guidance 3.5, resolution-dynamic mu ([`base_schedule`]); the SceneWorks manifest `default_steps` /
/// `defaults.guidanceScale` mirror these.
pub const RAW_STEPS: usize = 52;
pub const RAW_GUIDANCE: f32 = 3.5;

/// The image-edit reference cap (epic 10871 / sc-10878). The edit LoRA was trained on **one or two**
/// references in a **fixed order** — scene = image 1, person = image 2 — and the authors note swapping
/// the order degrades results. The engine's `forward_edit` is generic over N references (each at its own
/// RoPE frame), but the trained contract is 1..=2, so the pipeline caps here.
pub const MAX_EDIT_REFERENCES: usize = 2;

/// Max prompt tokens the Qwen3-VL RoPE table is sized for (generous; Krea prompts + the 34-token
/// template prefix are short). Enforced up front by [`crate::tokenizer::KreaTokenizer::encode_prompt`]
/// so an over-length prompt returns a clear length error instead of an opaque tensor-shape error deep
/// in the condition encoder (sc-9047).
pub(crate) const MAX_TEXT_TOKENS: usize = 1024;

/// Max tokens for the **image-grounded edit** conditioning (epic 10871 / sc-10880). Far larger than
/// [`MAX_TEXT_TOKENS`] because the edit template embeds one `<|image_pad|>` per merged vision token —
/// a single ~1 MP reference is ~1000 tokens, and two references push past 2000. Unlike the t2i path,
/// this is NOT bounded by the encoder's RoPE-table size: the grounded [`KreaTextEncoder::forward_with_images`]
/// builds a fresh interleaved-MRoPE table sized to the actual sequence, so the only cap is this guard
/// against a pathologically large reference set. (The t2i cap stays 1024 — short prompts only.)
pub(crate) const MAX_EDIT_TOKENS: usize = 8192;

/// The loaded Krea 2 Turbo components, `Arc`-shared so the generator caches them across `generate`.
pub struct Components {
    tok: crate::tokenizer::KreaTokenizer,
    te: KreaTextEncoder,
    dit: Krea2Transformer,
    vae: Arc<QwenVae>,
    /// Optional NVIDIA PiD super-resolving decoder (epic 7840 / sc-7853), loaded once when the model
    /// was loaded with `LoadSpec::pid`. `None` ⇒ the native `QwenVae` decode (the default path).
    pid: Option<Arc<PidEngine>>,
}

/// Load all Turbo components from a Krea 2 snapshot (`tokenizer/ text_encoder/ transformer/ vae/`).
///
/// `adapters` (when non-empty) are trained `krea_2_raw` LoRA/LoKr `.safetensors` merged into the dense
/// DiT attention projections at load (sc-7836, [`crate::adapters::merge_into_weights`]) — **merge, not
/// residual** (the flow-match sampler is chaos-sensitive). Empty ⇒ the stock unadapted build.
pub fn load_components(
    root: &Path,
    device: &Device,
    adapters: &[AdapterSpec],
    pid_spec: Option<&PidWeights>,
) -> Result<Components> {
    let tok = crate::tokenizer::KreaTokenizer::from_snapshot(root, device)?;

    let te_cfg = KreaTeConfig::from_snapshot(root)?;
    let te_w = Weights::from_dir(&root.join("text_encoder"), device, TE_DTYPE)?;
    let te = KreaTextEncoder::load(&te_w, "language_model", &te_cfg, MAX_TEXT_TOKENS)?;

    let cfg = Krea2Config::from_snapshot(root)?;
    let mut dit_w = Weights::from_dir(&root.join("transformer"), device, DIT_DTYPE)?;
    crate::convert::validate_transformer(&dit_w, &cfg)?;
    // Fold any LoRA/LoKr adapters into the targeted dense weights before the DiT reads them. A
    // non-empty spec that matches no target is a hard error inside `merge_into_weights` (the worker
    // then falls back rather than silently rendering unadapted).
    crate::adapters::merge_into_weights(&mut dit_w, &cfg, adapters)?;
    let dit = Krea2Transformer::load(&dit_w, &cfg)?;

    let vae = load_vae(root, device)?;

    // Load the optional PiD super-resolving decoder once (epic 7840 / sc-7853) when the caller opted
    // in via `LoadSpec::pid`; Krea shares the Qwen-Image VAE latent space (`qwenimage` student).
    let pid = match pid_spec {
        Some(spec) => Some(Arc::new(PidEngine::from_spec(spec, PID_BACKBONE, device)?)),
        None => None,
    };

    Ok(Components {
        tok,
        te,
        dit,
        vae: Arc::new(vae),
        pid,
    })
}

/// Load Turbo components with the DiT taken from a **single-file INT8-ConvRot checkpoint** (sc-9300)
/// instead of the snapshot's `transformer/` dir. The tokenizer / Qwen3-VL TE / Qwen-Image VAE still come
/// from the canonical `root` snapshot (the ConvRot artifact quantizes only the DiT). `convrot_dit` is
/// the native-mmdit-keyed `.safetensors` file; the DiT's 28 blocks' attn+mlp load as per-output-channel
/// int8 (cuBLASLt IGEMM on CUDA), everything else dense bf16.
///
/// **Coherent as of sc-9601.** The checkpoint's int8 weights are the *rotated* `W·R` (regular-Hadamard,
/// group 256); each ConvRot projection now applies the matching online `RHT(x)` activation rotation
/// ([`candle_gen::quant::convrot`]) before the int8 IGEMM, so `RHT(x)·(W·R)ᵀ = x·Wᵀ` and the render is
/// coherent (the sc-9300 A/B NO-GO was the missing online leg — arXiv 2512.03673 / ComfyUI ConvRot,
/// clean-room from the paper + the checkpoint format). The per-channel dequant fold runs on-device
/// (sc-9601 perf). Worker wiring as a shipping generator variant stays deferred (sc-9092 pattern).
///
/// **sm_89 floor (locked decision 7 / sc-9300).** The int8 IGEMM tier is only offered on compute
/// capability ≥ 8.9 (RTX 40-series and up). On CUDA, this errors up front if the device is below the
/// floor rather than rendering on a card the marketing contract excludes; on non-CUDA it is a no-op
/// (the CPU dequant-dense fallback is for tests, not a shipping path).
pub fn load_components_convrot(
    root: &Path,
    convrot_dit: &Path,
    device: &Device,
) -> Result<Components> {
    ensure_int8_floor(device)?;

    let tok = crate::tokenizer::KreaTokenizer::from_snapshot(root, device)?;

    let te_cfg = KreaTeConfig::from_snapshot(root)?;
    let te_w = Weights::from_dir(&root.join("text_encoder"), device, TE_DTYPE)?;
    let te = KreaTextEncoder::load(&te_w, "language_model", &te_cfg, MAX_TEXT_TOKENS)?;

    let cfg = Krea2Config::from_snapshot(root)?;
    let dit_w = Weights::from_convrot_file(convrot_dit, device, DIT_DTYPE)?;
    crate::convert::validate_transformer(&dit_w, &cfg)?;
    let dit = Krea2Transformer::load(&dit_w, &cfg)?;

    let vae = load_vae(root, device)?;

    Ok(Components {
        tok,
        te,
        dit,
        vae: Arc::new(vae),
        // The INT8-ConvRot path is a deferred non-shipping variant (see the fn docs); PiD is not wired
        // through it. The shipping `load_components` path carries the optional decoder.
        pid: None,
    })
}

/// Enforce the INT8-ConvRot sm_89 compute-capability floor (locked decision 7). Reuses the sc-9299
/// cuBLASLt compute-cap probe (`meets_fp8_floor` ⇔ capability ≥ 8.9). A non-CUDA device is allowed (the
/// CPU dequant path is test-only). On CUDA below the floor this errors with the marketing contract.
#[cfg(feature = "cuda")]
fn ensure_int8_floor(device: &Device) -> Result<()> {
    if device.is_cuda() {
        let lt = candle_gen::quant::CublasLt::new(device)
            .map_err(|e| CandleError::Msg(format!("krea convrot: cublasLt probe: {e}")))?;
        if !lt
            .meets_fp8_floor()
            .map_err(|e| CandleError::Msg(format!("krea convrot: compute-cap probe: {e}")))?
        {
            let cap = lt.compute_cap().unwrap_or((0, 0));
            return Err(CandleError::Msg(format!(
                "krea INT8-ConvRot requires compute capability >= 8.9 (RTX 40-series+); this device is \
                 sm_{}{} — the ConvRot variant is not offered on older cards",
                cap.0, cap.1
            )));
        }
    }
    Ok(())
}

/// Non-CUDA build: the int8 floor is vacuous (the CPU dequant-dense fallback is test-only).
#[cfg(not(feature = "cuda"))]
fn ensure_int8_floor(_device: &Device) -> Result<()> {
    Ok(())
}

/// Render the **Turbo** (CFG-free, few-step rectified-flow Euler) text-to-image path for `req`.
pub fn render(
    comps: &Components,
    req: &GenerationRequest,
    device: &Device,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Vec<Image>> {
    let steps = req.steps.map(|s| s as usize).unwrap_or(TURBO_STEPS);
    let base_seed = req.seed.unwrap_or_else(gen_core::default_seed);

    // Condition encoding (seed-independent): the 12 selected Qwen3-VL hidden layers, stacked +
    // prefix-dropped → the DiT's text_fusion context [1, n_tok, 12, 2560]. CFG-free, B=1.
    let context = comps
        .te
        .forward(&comps.tok.encode_prompt(&req.prompt, MAX_TEXT_TOKENS)?)?;

    // Native exponential-mu Turbo sigmas are the byte-exact default; a curated scheduler reshapes over
    // the same mu. Raw sigma → DiT timestep, raw velocity → Euler `x + v·(σ_{i+1} − σ_i)`.
    let native = turbo_sigmas(steps);
    let sigmas = candle_gen::resolve_flow_schedule(
        req.scheduler.as_deref(),
        TURBO_MU as f32,
        steps,
        &native,
    );

    // Resolve the decode seam once for the whole batch (epic 7840 / sc-7853): a per-generation PiD
    // decoder bound to this prompt when `req.use_pid` is set (errors if requested but not loaded), else
    // `None` → the native QwenVae decode. Shared across `count` images (same prompt).
    let pid_decoder = candle_gen_pid::resolve_pid_decoder(
        comps.pid.as_deref(),
        req,
        base_seed,
        crate::KREA_2_TURBO_ID,
    )?;

    candle_gen::for_each_image_seed(base_seed, req.count, |seed| {
        let noise = init_noise(req.height, req.width, seed, device)?;
        let lat = candle_gen::run_flow_sampler(
            req.sampler.as_deref(),
            TimestepConvention::Sigma,
            &sigmas,
            noise,
            seed,
            &req.cancel,
            on_progress,
            |x, timestep| -> Result<Tensor> {
                let t = Tensor::from_vec(vec![timestep], (1,), device)?;
                let v = comps.dit.forward(x, &t, &context)?;
                Ok(v.to_dtype(DType::F32)?)
            },
        )?;
        on_progress(Progress::Decoding);
        // PiD (super-resolving) decode when the toggle resolved one; else the native VAE. Both consume
        // the same normalized `[1,16,H/8,W/8]` latent (a zero-transform seam); PiD returns a larger
        // `[1,3,4H,4W]` tensor and `to_image` reads the size from it.
        let decoded = match &pid_decoder {
            Some(pid) => pid.decode(&lat)?,
            None => comps.vae.decode(&lat)?.to_dtype(DType::F32)?,
        };
        to_image(&decoded)
    })
}

/// Render the **Raw** (undistilled, full classifier-free-guidance) rectified-flow text-to-image path
/// for `req` (`krea_2_raw`, epic 9992 / sc-9994) — the CFG sibling of [`render`]. Two DiT forwards per
/// step, the conditional (positive prompt) and the unconditional (the user negative prompt, or `""`
/// when none), combined by the **reference** `sampling.py:129` formula via [`krea_cfg_combine`]
/// (`v = cond + guidance·(cond − uncond)`, NOT the textbook `uncond + g·Δ`: Krea's guidance is offset by
/// one). `guidance ≤ 0` short-circuits to a single conditional forward (the uncond context is never
/// encoded), matching the reference `cfg = guidance > 0`. Unlike Turbo's fixed `mu = 1.15`, the schedule
/// is resolution-**dynamic** ([`base_schedule`]). Everything else — the Qwen3-VL condition encode, the
/// PiD/native decode seam, and the per-seed batch loop — is identical to [`render`].
pub fn render_base(
    comps: &Components,
    req: &GenerationRequest,
    device: &Device,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Vec<Image>> {
    let steps = req.steps.map(|s| s as usize).unwrap_or(RAW_STEPS);
    let base_seed = req.seed.unwrap_or_else(gen_core::default_seed);
    let guidance = req.guidance.unwrap_or(RAW_GUIDANCE);

    // Positive (conditional) condition encoding (seed-independent): the 12 selected Qwen3-VL hidden
    // layers → the DiT's text_fusion context [1, n_tok, 12, 2560].
    let context = comps
        .te
        .forward(&comps.tok.encode_prompt(&req.prompt, MAX_TEXT_TOKENS)?)?;

    // Unconditional context, encoded ONLY when CFG is active (reference `cfg = guidance > 0`). An
    // absent / empty negative prompt defaults to `""` (reference `negative_prompts = [""] * n`).
    let neg_context = if guidance > 0.0 {
        let negative = req.negative_prompt.as_deref().unwrap_or_default();
        Some(
            comps
                .te
                .forward(&comps.tok.encode_prompt(negative, MAX_TEXT_TOKENS)?)?,
        )
    } else {
        None
    };

    // Resolution-dynamic Raw sigma schedule (mu from the image-token count); a curated scheduler
    // reshapes over the same dynamic mu. Raw sigma → DiT timestep, raw velocity → Euler
    // `x + v·(σ_{i+1} − σ_i)`.
    let sigmas = base_schedule(steps, req.width, req.height, req.scheduler.as_deref());

    // Resolve the decode seam once for the whole batch (epic 7840 / sc-7853): a per-generation PiD
    // decoder bound to this prompt when `req.use_pid` is set (errors if requested but not loaded), else
    // `None` → the native QwenVae decode.
    let pid_decoder = candle_gen_pid::resolve_pid_decoder(
        comps.pid.as_deref(),
        req,
        base_seed,
        crate::KREA_2_RAW_ID,
    )?;

    candle_gen::for_each_image_seed(base_seed, req.count, |seed| {
        let noise = init_noise(req.height, req.width, seed, device)?;
        let lat = candle_gen::run_flow_sampler(
            req.sampler.as_deref(),
            TimestepConvention::Sigma,
            &sigmas,
            noise,
            seed,
            &req.cancel,
            on_progress,
            |x, timestep| -> Result<Tensor> {
                let t = Tensor::from_vec(vec![timestep], (1,), device)?;
                let cond = comps.dit.forward(x, &t, &context)?;
                // Two-forward CFG when a negative context was prepared (guidance > 0); else the bare
                // conditional velocity. Combined by the shared reference formula (`krea_cfg_combine`).
                let v = match &neg_context {
                    Some(nc) => {
                        let uncond = comps.dit.forward(x, &t, nc)?;
                        krea_cfg_combine(&cond, &uncond, guidance)?
                    }
                    None => cond,
                };
                Ok(v.to_dtype(DType::F32)?)
            },
        )?;
        on_progress(Progress::Decoding);
        let decoded = match &pid_decoder {
            Some(pid) => pid.decode(&lat)?,
            None => comps.vae.decode(&lat)?.to_dtype(DType::F32)?,
        };
        to_image(&decoded)
    })
}

/// The image-edit-only components (epic 10871), loaded lazily on the first edit so the txt2img paths
/// keep their footprint. Holds the two extra wires the edit LoRA's dual-conditioning contract needs:
/// the Qwen-Image VAE **encoder** (source references → in-context VAE tokens, sc-10877) and the
/// Qwen3-VL **vision tower** (source references → image-grounded conditioning, sc-10880).
pub struct EditComponents {
    vae_encoder: QwenVaeEncoder,
    vision: crate::vision::VisionTower,
}

/// Load the image-edit-only components from a Krea 2 snapshot (epic 10871). Both read weights that
/// already ship in the snapshot — the VAE encoder from `vae/`, the vision tower from the `visual.*`
/// subtree of `text_encoder/`.
pub fn load_edit_components(root: &Path, device: &Device) -> Result<EditComponents> {
    Ok(EditComponents {
        vae_encoder: load_vae_encoder(root, device)?,
        vision: crate::vision::load_vision_tower(root, device)?,
    })
}

/// Render the **image-edit** path (epic 10871 / sc-10877, sc-10878) — a source reference (or two:
/// scene = image 1, person = image 2) plus an instruction produce an edited image with identity/subject
/// preserved. Kontext-style: each reference is VAE-encoded at the **target** resolution and prepended to
/// the DiT sequence at its own RoPE frame ([`Krea2Transformer::forward_edit`]); the denoise runs the Raw
/// true-CFG loop **from pure noise** (the source is in-context conditioning, NOT a noised init — the key
/// difference from an img2img latent-blend). `references` is the fixed-order source set (1 or 2).
///
/// **Dual conditioning (R2).** BOTH wires the edit LoRA was trained against are supplied: (a) the
/// in-context VAE tokens (references VAE-encoded at target res, seq-concat in `forward_edit`), and (b)
/// the image-grounded Qwen3-VL encoding (references run through the vision tower, spliced into the
/// condition encoder — [`KreaTextEncoder::forward_with_images`]). The vision tower runs once per
/// reference and is reused across the positive/negative instruction encodings.
pub fn render_edit(
    comps: &Components,
    edit: &EditComponents,
    req: &GenerationRequest,
    references: &[Image],
    // `true` = the CFG-free distilled Turbo edit (`krea_2_turbo_edit`, sc-11640): few-step
    // `turbo_schedule` (fixed mu), guidance forced to 0 (a single conditional forward). `false` = the
    // undistilled Raw edit (`krea_2_edit`, epic 10871): the resolution-dynamic `base_schedule` under
    // full CFG. Mirrors mlx-gen-krea `render_edit(distilled)`.
    distilled: bool,
    device: &Device,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Vec<Image>> {
    check_reference_count(references.len())?;
    let steps =
        req.steps
            .map(|s| s as usize)
            .unwrap_or(if distilled { TURBO_STEPS } else { RAW_STEPS });
    let base_seed = req.seed.unwrap_or_else(gen_core::default_seed);
    // Turbo edit is CFG-free: force guidance 0 so the unconditional grounded encode is skipped and the
    // sampler runs a single conditional forward. Raw edit honors the request guidance (default 3.5).
    let guidance = if distilled {
        0.0
    } else {
        req.guidance.unwrap_or(RAW_GUIDANCE)
    };

    // Wire (b): run the vision tower ONCE per reference (shared across the pos/neg instruction encodings).
    let grounding = encode_vision(&edit.vision, references, device)?;

    // Positive (+ optional unconditional) image-grounded conditioning: the instruction is tokenized with
    // the vision blocks, and the tower's merged embeds + deepstack features are spliced in.
    let context = grounding.condition(&comps.tok, &comps.te, &req.prompt)?;
    let neg_context = if guidance > 0.0 {
        let negative = req.negative_prompt.as_deref().unwrap_or_default();
        Some(grounding.condition(&comps.tok, &comps.te, negative)?)
    } else {
        None
    };

    // Wire (a): VAE-encode each reference at the TARGET resolution → the normalized 16-ch latent (static
    // across steps). Fixed order preserved (scene, then person — sc-10878).
    let ref_latents =
        encode_references(&edit.vae_encoder, references, req.width, req.height, device)?;

    // Turbo edit runs the distilled few-step `turbo_schedule` (fixed mu) the CFG-free student expects;
    // Raw edit runs the resolution-dynamic `base_schedule` (undistilled, like `render_base`).
    let sigmas = if distilled {
        turbo_schedule(steps, req.scheduler.as_deref())
    } else {
        base_schedule(steps, req.width, req.height, req.scheduler.as_deref())
    };

    // Resolve the decode seam once for the whole batch (epic 7840 / sc-7853, sc-11197): a per-generation
    // PiD decoder bound to this request when `req.use_pid` is set (errors if requested but not loaded),
    // else `None` → the native QwenVae decode. The edit id honors `use_pid` exactly like its `render` /
    // `render_base` txt2img siblings — an edit generator loaded with a PiD engine (`spec.pid`) is no
    // longer a dead wire. `model_id` names the surface for the error message (Raw vs distilled Turbo edit).
    let pid_decoder = candle_gen_pid::resolve_pid_decoder(
        comps.pid.as_deref(),
        req,
        base_seed,
        edit_pid_model_id(distilled),
    )?;

    candle_gen::for_each_image_seed(base_seed, req.count, |seed| {
        let noise = init_noise(req.height, req.width, seed, device)?;
        let lat = candle_gen::run_flow_sampler(
            req.sampler.as_deref(),
            TimestepConvention::Sigma,
            &sigmas,
            noise,
            seed,
            &req.cancel,
            on_progress,
            |x, timestep| -> Result<Tensor> {
                let t = Tensor::from_vec(vec![timestep], (1,), device)?;
                let cond = comps.dit.forward_edit(x, &t, &context, &ref_latents)?;
                let v = match &neg_context {
                    Some(nc) => {
                        let uncond = comps.dit.forward_edit(x, &t, nc, &ref_latents)?;
                        krea_cfg_combine(&cond, &uncond, guidance)?
                    }
                    None => cond,
                };
                Ok(v.to_dtype(DType::F32)?)
            },
        )?;
        on_progress(Progress::Decoding);
        // PiD (super-resolving) decode when the toggle resolved one; else the native VAE. Both consume
        // the same normalized `[1,16,H/8,W/8]` latent (a zero-transform seam); PiD returns a larger
        // `[1,3,4H,4W]` tensor and `to_image` reads the size from it — matching `render` / `render_base`.
        let decoded = match &pid_decoder {
            Some(pid) => pid.decode(&lat)?,
            None => comps.vae.decode(&lat)?.to_dtype(DType::F32)?,
        };
        to_image(&decoded)
    })
}

/// VAE-encode each source `references` image at the target `(width, height)` → the normalized 16-ch
/// latent `[1, 16, H/8, W/8]` the DiT's `img_in` consumes (same space as the noise). Fixed order is
/// preserved (the returned vec is in `references` order — scene, then person; sc-10878).
fn encode_references(
    vae_encoder: &QwenVaeEncoder,
    references: &[Image],
    width: u32,
    height: u32,
    device: &Device,
) -> Result<Vec<Tensor>> {
    let mut out = Vec::with_capacity(references.len());
    for im in references {
        let pixels = image_to_pixels(im, width, height, device)?;
        out.push(vae_encoder.encode(&pixels)?);
    }
    Ok(out)
}

/// Validate the source-reference count against the fixed-order [`MAX_EDIT_REFERENCES`] cap (sc-10878):
/// at least one (an edit needs a source) and at most two (scene, then person). Pure so it is unit-testable.
fn check_reference_count(n: usize) -> Result<()> {
    if n == 0 {
        return Err(CandleError::Msg(
            "krea edit: at least one source reference image is required".into(),
        ));
    }
    if n > MAX_EDIT_REFERENCES {
        return Err(CandleError::Msg(format!(
            "krea edit: at most {MAX_EDIT_REFERENCES} references are supported (scene = image 1, \
             person = image 2); got {n}"
        )));
    }
    Ok(())
}

/// The vision-tower output for the source references (epic 10871 / sc-10880), shared across the
/// positive/negative instruction encodings: each reference's merged image embeds + 3 deepstack features
/// + patch grid, and the per-reference `<|image_pad|>` placeholder count for the tokenizer.
struct Grounding {
    embeds: Vec<Tensor>,
    deepstack: Vec<Vec<Tensor>>,
    grids: Vec<[i32; 3]>,
    n_per_ref: Vec<usize>,
}

impl Grounding {
    /// Image-grounded condition encoding for one instruction: tokenize `prompt` with the reference
    /// vision blocks, then splice the tower's embeds/deepstack into the condition encoder.
    fn condition(
        &self,
        tok: &crate::tokenizer::KreaTokenizer,
        te: &KreaTextEncoder,
        prompt: &str,
    ) -> Result<Tensor> {
        let ids = tok.encode_with_images(prompt, &self.n_per_ref, MAX_EDIT_TOKENS)?;
        Ok(te.forward_with_images(&ids, &self.embeds, &self.deepstack, &self.grids)?)
    }
}

/// Run the Qwen3-VL vision tower over each source reference (in fixed order — sc-10878) → the shared
/// [`Grounding`]. Preprocess + tower `forward` per reference (epic 10871 / sc-10880).
fn encode_vision(
    vision: &crate::vision::VisionTower,
    references: &[Image],
    device: &Device,
) -> Result<Grounding> {
    let merge = vision.config().spatial_merge_size;
    let mut g = Grounding {
        embeds: Vec::with_capacity(references.len()),
        deepstack: Vec::with_capacity(references.len()),
        grids: Vec::with_capacity(references.len()),
        n_per_ref: Vec::with_capacity(references.len()),
    };
    for im in references {
        let (embeds, deepstack, grid) = crate::vision::encode_image(
            vision,
            &im.pixels,
            im.height as usize,
            im.width as usize,
            device,
        )?;
        g.n_per_ref
            .push(crate::vision::merged_token_count(grid, merge));
        g.embeds.push(embeds);
        g.deepstack.push(deepstack);
        g.grids.push(grid);
    }
    Ok(g)
}

/// One RGB8 [`Image`] (HWC, `[0, 255]`) resized to `(target_w, target_h)` → the VAE input tensor
/// `[1, 3, H, W]` in `[-1, 1]` (f32, on `device`). BICUBIC resample (gen-core's PIL-exact
/// [`resize_bicubic_u8`], the same used by the vision preprocess), then `(x/255)·2 − 1`.
fn image_to_pixels(im: &Image, target_w: u32, target_h: u32, device: &Device) -> Result<Tensor> {
    let (w, h) = (target_w as usize, target_h as usize);
    if im.pixels.len() != (im.width * im.height * 3) as usize {
        return Err(CandleError::Msg(format!(
            "krea edit: reference pixel buffer {} != {}x{}x3",
            im.pixels.len(),
            im.width,
            im.height
        )));
    }
    let resized: Vec<f32> = if (im.width, im.height) == (target_w, target_h) {
        im.pixels.iter().map(|&p| p as f32).collect()
    } else {
        resize_bicubic_u8(&im.pixels, im.height as usize, im.width as usize, h, w)?
    };
    // HWC [0,255] → CHW [-1,1].
    let mut chw = vec![0f32; 3 * h * w];
    for c in 0..3 {
        for y in 0..h {
            for x in 0..w {
                chw[c * h * w + y * w + x] = resized[(y * w + x) * 3 + c] / 127.5 - 1.0;
            }
        }
    }
    Ok(Tensor::from_vec(chw, (1, 3, h, w), device)?)
}

/// The **Raw** flow-match sigma schedule for `steps` at a given resolution: the exponential-mu shift
/// with a resolution-**dynamic** `mu` interpolated in image-token count ([`dynamic_mu`]), unlike the
/// Turbo fixed `mu = 1.15`. Length `steps + 1`, descending with a trailing `0.0`; a curated scheduler
/// (epic 7114) reshapes over the same dynamic mu. Mirrors mlx-gen-krea `base_schedule`.
pub fn base_schedule(steps: usize, width: u32, height: u32, scheduler: Option<&str>) -> Vec<f32> {
    // Image token count = (W/16)·(H/16) (latent /8 then patch /2) — the reference `x.shape[1]`.
    let seq_len = (width as f64 / 16.0) * (height as f64 / 16.0);
    let mu = dynamic_mu(seq_len);
    let native = krea_sigmas(steps, mu);
    candle_gen::resolve_flow_schedule(scheduler, mu as f32, steps, &native)
}

/// The **Turbo** flow-match sigma schedule for `steps`: the TDM-distilled fixed-mu (`TURBO_MU = 1.15`)
/// exponential-shift `turbo_sigmas`, optionally reshaped by a curated scheduler over the same mu. Length
/// `steps + 1`, descending with a trailing `0.0`. The distilled CFG-free Turbo edit (`krea_2_turbo_edit`,
/// sc-11640) denoises on this trajectory the few-step student was trained on, NOT the resolution-dynamic
/// `base_schedule`. Mirrors mlx-gen-krea `turbo_schedule` + the `render_turbo` inline build.
pub fn turbo_schedule(steps: usize, scheduler: Option<&str>) -> Vec<f32> {
    let native = turbo_sigmas(steps);
    candle_gen::resolve_flow_schedule(scheduler, TURBO_MU as f32, steps, &native)
}

/// The surface id for the image-edit PiD decode-seam error message (sc-11197): the registered Raw edit
/// ([`crate::KREA_2_EDIT_ID`]) vs the distilled Turbo edit ([`crate::KREA_2_TURBO_EDIT_ID`], driven
/// through the worker's bespoke lane with `distilled = true`). Pure so the routing is unit-testable.
fn edit_pid_model_id(distilled: bool) -> &'static str {
    if distilled {
        crate::KREA_2_TURBO_EDIT_ID
    } else {
        crate::KREA_2_EDIT_ID
    }
}

/// Krea's classifier-free-guidance velocity combine — the reference `sampling.py:129`
/// `v = v_cond + guidance·(v_cond − v_uncond)`, **NOT** the standard `v_uncond + g·Δ`. Krea's guidance
/// is offset by one: the standard form applies one full step LESS guidance, and at `guidance = 1.0`
/// collapses to exactly `v_cond` (zero effective CFG — the washed-out-render trap). Single source of
/// truth so the Raw inference path (sc-9994) and the trainer preview (`training::render_sample`) can
/// never drift again (mirrors the mlx-gen sc-10009 dedupe). The caller runs this only for
/// `guidance > 0` (a single conditional forward otherwise, matching `cfg = guidance > 0`).
pub(crate) fn krea_cfg_combine(
    v_cond: &Tensor,
    v_uncond: &Tensor,
    guidance: f32,
) -> Result<Tensor> {
    let guided = ((v_cond - v_uncond)? * guidance as f64)?;
    Ok((v_cond + guided)?)
}

/// Seeded initial Gaussian latent noise `[1, 16, H/8, W/8]` (f32; the VAE's 8× spatial compression).
/// Deterministic, launch-portable CPU RNG (sc-3673 parity), exactly as the z-image/ideogram/boogu
/// providers. The model layer offsets `seed` per image in a batch (reference `seed + i`).
fn init_noise(height: u32, width: u32, seed: u64, device: &Device) -> Result<Tensor> {
    let (lat_h, lat_w) = (
        (height / SPATIAL_SCALE) as usize,
        (width / SPATIAL_SCALE) as usize,
    );
    let n = LATENT_CHANNELS * lat_h * lat_w;
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = candle_gen::seeded_normal_vec(&mut rng, n);
    Ok(
        Tensor::from_vec(noise, (1, LATENT_CHANNELS, lat_h, lat_w), &Device::Cpu)?
            .to_device(device)?,
    )
}

/// Convert a decoded pixel tensor `[1, 3, H, W]` in `[-1, 1]` (f32) → RGB8 [`Image`]. Shared by the
/// native VAE decode (`QwenVae::decode` applies the per-channel `z·std + mean` de-normalize internally)
/// and the PiD super-resolving decode (which already emits `[-1, 1]` pixels, possibly at 4× the size).
/// The reference `clamp(-1,1)·0.5 + 0.5` denormalize is the `(x+1)·127.5` below; the output size is read
/// from the tensor, never assumed (PiD may be larger than VAE-native).
pub(crate) fn to_image(decoded: &Tensor) -> Result<Image> {
    let img = ((decoded.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?.to_dtype(DType::U8)?;
    let img = img.i(0)?.to_device(&Device::Cpu)?;
    let (c, h, w) = img.dims3()?;
    if c != 3 {
        return Err(CandleError::Msg(format!(
            "krea: expected 3 channels, got {c}"
        )));
    }
    let pixels = img.permute((1, 2, 0))?.flatten_all()?.to_vec1::<u8>()?;
    Ok(Image {
        width: w as u32,
        height: h as u32,
        pixels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `image_to_pixels` normalizes an RGB8 image to `[1, 3, H, W]` in `[-1, 1]` and, when the source is
    /// already at the target size, maps pixel `v` → `v/127.5 − 1` per channel with no resample (a mid-gray
    /// 128 → ~0.0039, black 0 → −1, white 255 → +1).
    #[test]
    fn image_to_pixels_normalizes_and_shapes() {
        // 2×2 image, channel-constant: R=0, G=128, B=255.
        let mut px = Vec::new();
        for _ in 0..4 {
            px.extend_from_slice(&[0u8, 128, 255]);
        }
        let im = Image {
            width: 2,
            height: 2,
            pixels: px,
        };
        let t = image_to_pixels(&im, 2, 2, &Device::Cpu).unwrap();
        assert_eq!(t.dims(), &[1, 3, 2, 2]);
        let v = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // CHW: first 4 = R (−1), next 4 = G (~0), last 4 = B (+1).
        assert!(v[0..4].iter().all(|&x| (x - (-1.0)).abs() < 1e-6));
        assert!(v[4..8].iter().all(|&x| x.abs() < 0.01));
        assert!(v[8..12].iter().all(|&x| (x - 1.0).abs() < 1e-6));
    }

    #[test]
    fn image_to_pixels_rejects_bad_buffer() {
        let im = Image {
            width: 4,
            height: 4,
            pixels: vec![0u8; 10],
        };
        assert!(image_to_pixels(&im, 4, 4, &Device::Cpu).is_err());
    }

    #[test]
    fn reference_count_enforces_fixed_order_cap() {
        // 0 → error (an edit needs a source); 1 and 2 ok; 3 → error (past the scene/person cap).
        assert!(check_reference_count(0).is_err());
        assert!(check_reference_count(1).is_ok());
        assert!(check_reference_count(2).is_ok());
        let err = check_reference_count(3).unwrap_err().to_string();
        assert!(err.contains('2'), "names the cap: {err}");
        assert!(
            err.contains("scene") && err.contains("person"),
            "documents order: {err}"
        );
    }

    /// The Krea CFG combine is the reference `cond + g·(cond − uncond)`, not the standard
    /// `uncond + g·Δ`. With cond = 2, uncond = 1 (Δ = 1): g = 1 → 3 (the standard form would give 2 —
    /// exactly `cond` — which is why the shared default `sample_guidance_scale = 1.0` washed previews
    /// out, sc-10009); a larger g pushes further from cond, away from uncond; g = 0 → cond.
    #[test]
    fn cfg_combine_is_reference_offset_by_one() {
        let cond = Tensor::from_vec(vec![2.0f32], (1,), &Device::Cpu).unwrap();
        let uncond = Tensor::from_vec(vec![1.0f32], (1,), &Device::Cpu).unwrap();
        for (g, want) in [(1.0f32, 3.0f32), (3.5, 5.5), (0.0, 2.0)] {
            let v = krea_cfg_combine(&cond, &uncond, g).unwrap();
            let got = v.to_vec1::<f32>().unwrap()[0];
            assert!((got - want).abs() < 1e-5, "g={g}: got {got}, want {want}");
        }
    }

    /// The edit path names the right surface for the PiD decode-seam error (sc-11197): the registered
    /// Raw edit id when undistilled, the distilled Turbo-edit tag when `distilled = true`.
    #[test]
    fn edit_pid_model_id_names_the_surface() {
        assert_eq!(edit_pid_model_id(false), crate::KREA_2_EDIT_ID);
        assert_eq!(edit_pid_model_id(true), crate::KREA_2_TURBO_EDIT_ID);
    }

    /// The image-edit decode seam honors `req.use_pid` exactly like `render` / `render_base` (sc-11197,
    /// F-088). Before the fix `render_edit` never consulted `use_pid` and always decoded with the native
    /// VAE — the descriptor-accepts / render-drops trap. This asserts the shared seam `render_edit` now
    /// calls (`resolve_pid_decoder` with the edit id): `use_pid = true` with no loaded engine is a hard
    /// error (no longer silently dropped), and `use_pid = false` resolves to the native path (`None`).
    #[test]
    fn edit_decode_seam_honors_use_pid() {
        // use_pid requested but no PiD engine loaded → the same hard error the txt2img siblings raise
        // (previously `render_edit` swallowed the flag and returned a native-resolution image).
        let want_pid = GenerationRequest {
            prompt: "make it autumn".into(),
            width: 512,
            height: 512,
            use_pid: true,
            ..Default::default()
        };
        // (`PidDecoder` is not `Debug`, so match the seam result rather than `unwrap_err`.)
        let err =
            match candle_gen_pid::resolve_pid_decoder(None, &want_pid, 0, edit_pid_model_id(false))
            {
                Ok(_) => panic!("use_pid = true with no loaded PiD engine must be a hard error"),
                Err(e) => e.to_string(),
            };
        assert!(
            err.contains(crate::KREA_2_EDIT_ID) && err.contains("use_pid"),
            "use_pid on the edit id must error when unloaded, naming the surface: {err}"
        );

        // use_pid unset → the native QwenVae decode (no decoder resolved), decoupled from any engine.
        let no_pid = GenerationRequest {
            prompt: "make it autumn".into(),
            width: 512,
            height: 512,
            use_pid: false,
            ..Default::default()
        };
        let resolved =
            candle_gen_pid::resolve_pid_decoder(None, &no_pid, 0, edit_pid_model_id(false));
        assert!(
            matches!(resolved, Ok(None)),
            "use_pid = false must resolve to the native decode path (None)"
        );
    }

    /// The Raw schedule uses the resolution-dynamic mu (vs Turbo's fixed 1.15): at 1024² the image-token
    /// count is `(1024/16)² = 4096`, so `mu = dynamic_mu(4096) = 0.90625`, and the native (unscheduled)
    /// sigmas match the reference `timesteps(seq_len=4096)` — a descending `[1.0 … 0.0]` of length
    /// `steps + 1`. Distinct from `turbo_sigmas`, confirming Raw is not on the distilled fixed-mu curve.
    #[test]
    fn base_schedule_is_resolution_dynamic() {
        let sig = base_schedule(4, 1024, 1024, None);
        assert_eq!(sig.len(), 5);
        assert_eq!(sig.first().copied(), Some(1.0));
        assert_eq!(sig.last().copied(), Some(0.0));
        // Reference `timesteps(seq_len=4096, steps=4)` at f64 precision (narrowed to the f32 the sampler
        // stores) — the same values schedule.rs asserts for the dynamic-mu path.
        let want = [1.0f64, 0.88130659, 0.71223223, 0.45205718, 0.0];
        for (i, (&g, w)) in sig.iter().zip(want).enumerate() {
            assert!((g as f64 - w).abs() < 1e-5, "sigma[{i}] = {g}, want {w}");
        }
        assert_ne!(
            sig,
            turbo_sigmas(4),
            "Raw dynamic-mu differs from Turbo fixed-mu"
        );
    }
}
