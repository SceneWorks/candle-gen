//! Krea 2 **Turbo** text-to-image pipeline (sc-7580/sc-7582) — tokenize → Qwen3-VL-4B condition-encode
//! (the 12-layer select stack) → DiT (text_fusion aggregator + single-stream denoise) → Qwen-Image VAE
//! decode. Port of `mlx-gen-krea`'s `pipeline.rs` (the reference `sampling.py::sample` Turbo path).
//!
//! **CFG-free.** The TDM distillation baked the guided velocity into the weights, so there is no
//! unconditional branch (`guidance == 0` in the reference) — one DiT forward per step. Per-sample
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
use crate::schedule::{turbo_sigmas, TURBO_MU, TURBO_STEPS};
use crate::text_encoder::{KreaTeConfig, KreaTextEncoder};
use crate::transformer::Krea2Transformer;
use crate::vae::load_vae;

/// Component compute dtypes. The Qwen3-VL TE runs in **f32** (parity-grade for this encoder, shared
/// with the ideogram/boogu ports); the 12B DiT runs **bf16** (native on candle's CUDA backend); the
/// Qwen-Image VAE runs **f32** (decode-precision-sensitive).
const TE_DTYPE: DType = DType::F32;
const DIT_DTYPE: DType = DType::BF16;

/// VAE spatial downscale (the latent is image/8 per side) and latent channel count.
const SPATIAL_SCALE: u32 = 8;
const LATENT_CHANNELS: usize = 16;

/// Max prompt tokens the Qwen3-VL RoPE table is sized for (generous; Krea prompts + the 34-token
/// template prefix are short). Enforced up front by [`crate::tokenizer::KreaTokenizer::encode_prompt`]
/// so an over-length prompt returns a clear length error instead of an opaque tensor-shape error deep
/// in the condition encoder (sc-9047).
pub(crate) const MAX_TEXT_TOKENS: usize = 1024;

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

    let mut images = Vec::with_capacity(req.count as usize);
    for index in 0..req.count {
        let seed = base_seed.wrapping_add(index as u64);
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
        images.push(to_image(&decoded)?);
    }
    Ok(images)
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
fn to_image(decoded: &Tensor) -> Result<Image> {
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
