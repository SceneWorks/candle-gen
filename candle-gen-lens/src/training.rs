//! The candle **Lens LoRA/LoKr trainer** (sc-5147) — the candle twin of the worker's Python torch
//! `lens_train_runner.py`, implementing the backend-neutral
//! [`gen_core::Trainer`](candle_gen::gen_core::train::Trainer) with `backend = "candle"`. Together with
//! the inference cutover it retires `/opt/lens-venv` + `INCLUDE_LENS` — the last Python holdout for Lens
//! (epic 3482 / 5164). It reuses the shared [`candle_gen::train`] harness the SDXL/Z-Image/Wan stories
//! established, building on [`crate::dit_train`]'s vendored trainable DiT and [`crate::vae`]'s encode
//! shim.
//!
//! Registered under `"lens"` — the **non-distilled** `microsoft/Lens` base (the de-distill lesson,
//! sc-1583; a LoRA trained here applies cleanly to `lens_turbo`, same architecture).
//!
//! ## The Lens recipe (from `lens_train_runner.py`)
//!
//! Cache → loop → save, on the **flow-match** objective:
//!  - **Flow-match, no negation.** `x_t = (1−t)·x0 + t·noise`, `target = noise − x0`; the DiT's **raw**
//!    velocity is regressed toward it (Lens feeds the transformer output to the scheduler *without*
//!    negation — opposite of Z-Image). The timestep `t ∈ (0, 1)` is fed to the DiT **directly** (no
//!    `1 − σ`, no `·1000`).
//!  - **gpt-oss text front-end, cached + frozen.** Each caption is gpt-oss-encoded and its 4 selected
//!    layers ([`DEFAULT_SELECTED_LAYERS`] = 5/11/17/23) captured + cropped at [`TXT_OFFSET`] (the
//!    harmony-preamble offset) — exactly the inference `encode_one`. Cached once; the encoder is dropped
//!    before the DiT loads.
//!  - **Latents from a neural VAE encode.** Each image is `Flux2Vae`-encoded to the packed DiT latent
//!    `[1, S, 128]` ([`crate::vae::encode`], posterior mean) and cached.
//!  - **Targets:** the fused dual-stream attention projections [`LENS_ATTN_TARGETS`]
//!    (`img_qkv`/`txt_qkv`/`to_out.0`/`to_add_out`); train only the adapter, freeze the gpt-oss encoder
//!    + VAE + DiT base.
//!  - **Save** a diffusers-format `.safetensors` (bare dotted PEFT keys for LoRA / `lokr_w*` + metadata
//!    for LoKr) that the inference merge ([`crate::adapters`]) loads unchanged.
//!
//! The 48-block backward always runs **gradient-checkpointed** (candle's matmul backward materializes a
//! grad for the frozen base weight too, so a dense 48-block backward holds ~48 layers of weight-grads at
//! once — the Wan lesson). Adapter factors / loss / grads / optimizer state stay f32 (master weights);
//! the frozen base + activation stream follow `train_dtype` (bf16 default).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_gen::candle_core::backprop::GradStore;
use candle_gen::candle_core::{DType, Device, Tensor, Var};
use candle_gen::candle_nn::VarBuilder;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};

use candle_gen::gen_core::train::{
    NetworkType, Trainer, TrainerDescriptor, TrainingConfig, TrainingOutput, TrainingProgress,
    TrainingRequest,
};
use candle_gen::gen_core::{self, LoadSpec, Modality, WeightsSource};
use candle_gen::train::checkpoint::{checkpoint_filename, file_stem};
use candle_gen::train::dataset::{bucket_resolution, load_image_tensor};
use candle_gen::train::gradient_checkpoint::checkpointed_backward;
use candle_gen::train::lora::{
    build_lokr_targets, build_lora_targets, save_lokr, save_lora_peft, AdapterKind, LoraSet,
};
use candle_gen::train::optim::{accumulate_grads, clip_grad_norm, scale_grads, TrainOptimizer};
use candle_gen::train::schedule::{lr_multiplier, schedule_updates};
use candle_gen::{CandleError, Result};

use crate::dit_train::{LensTransformerTrain, LENS_ATTN_TARGETS};
use crate::text::{LensTokenizer, TXT_OFFSET};
use crate::text_encoder::{Config as EncoderConfig, GptOssTextEncoder, DEFAULT_SELECTED_LAYERS};
use crate::transformer::LensDitConfig;
use crate::vae::{encode as vae_encode, Flux2Vae};
use crate::{DEFAULT_DATE, MODEL_ID_BASE};

/// gpt-oss is encoded at bf16 for caching (it only produces the cached, frozen features; kept f32 in
/// the cache and dropped before the DiT loads).
const ENC_DTYPE: DType = DType::BF16;

/// Recognized `timestep_type` values (`linear`/`uniform`/`weighted` + the `sigmoid` default); anything
/// else is rejected rather than silently defaulted (matching the Z-Image / Wan trainers).
const TIMESTEP_TYPES: [&str; 4] = ["sigmoid", "linear", "uniform", "weighted"];
/// Recognized `timestep_bias` values — the high/low tilts plus the neutral default.
const TIMESTEP_BIASES: [&str; 9] = [
    "balanced",
    "none",
    "neutral",
    "high",
    "high_noise",
    "favor_high_noise",
    "low",
    "low_noise",
    "favor_low_noise",
];
/// Recognized `loss_type` values — `mae`/`l1` select MAE, `mse`/`l2` the MSE default.
const LOSS_TYPES: [&str; 4] = ["mse", "l2", "mae", "l1"];

/// `"bf16"`/`"bfloat16"` → [`DType::BF16`] (the default — halves the activation working set); anything
/// else → [`DType::F32`]. Adapter factors / loss / grads stay f32 (master weights).
fn parse_compute_dtype(s: &str) -> DType {
    let t = s.trim();
    if t.eq_ignore_ascii_case("bf16") || t.eq_ignore_ascii_case("bfloat16") {
        DType::BF16
    } else {
        DType::F32
    }
}

fn normalize_cfg(s: &str) -> String {
    s.trim().to_ascii_lowercase().replace([' ', '-'], "_")
}

/// Sample a flow-match timestep `t ∈ (0, 1)` — `sigmoid(randn)` by default, `uniform` for linear,
/// `(uniform + sigmoid(randn))/2` for weighted; bias `high` → `√t`, `low` → `t²`. Clamped to
/// `[1e-3, 1−1e-3]`. Deterministic in `seed` (the sc-3673 CPU `StdRng` discipline). Unlike Wan there is
/// no expert band — Lens trains one model over the full `(0, 1)` range.
fn sample_timestep(timestep_type: &str, timestep_bias: &str, seed: u64) -> f64 {
    let mut rng = StdRng::seed_from_u64(seed);
    let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
    let t = match normalize_cfg(timestep_type).as_str() {
        "linear" | "uniform" => rng.random::<f32>(),
        "weighted" => {
            let base = rng.random::<f32>();
            let z: f32 = StandardNormal.sample(&mut rng);
            (base + sigmoid(z)) / 2.0
        }
        _ => {
            let z: f32 = StandardNormal.sample(&mut rng);
            sigmoid(z)
        }
    };
    let t = match normalize_cfg(timestep_bias).as_str() {
        "high" | "high_noise" | "favor_high_noise" => t.sqrt(),
        "low" | "low_noise" | "favor_low_noise" => t * t,
        _ => t,
    };
    (t as f64).clamp(1e-3, 1.0 - 1e-3)
}

/// `(x_t, target)` for one sample at flow-match `t`: `x_t = (1−t)·x0 + t·noise`, `target = noise − x0`
/// (the **raw** velocity Lens trains toward — NO sign flip). All in f32.
fn build_batch(x0: &Tensor, noise: &Tensor, t: f64) -> Result<(Tensor, Tensor)> {
    let x_t = ((x0 * (1.0 - t))? + (noise * t)?)?;
    let target = (noise - x0)?;
    Ok((x_t, target))
}

/// Flow-match velocity loss in f32: `mean((v − target)²)` (MSE) or `mean|v − target|` (MAE). `v` is the
/// DiT's raw f32 velocity output `[1, S, 128]`.
fn velocity_loss(
    v: &Tensor,
    target: &Tensor,
    mae: bool,
) -> candle_gen::candle_core::Result<Tensor> {
    let diff = (v.to_dtype(DType::F32)? - target)?;
    if mae {
        diff.abs()?.mean_all()
    } else {
        diff.sqr()?.mean_all()
    }
}

/// Deterministic `N(0, 1)` noise of the given shape (seeded CPU `StdRng`, sc-3673), moved to `device`.
fn sample_noise(shape: &[usize], seed: u64, device: &Device) -> Result<Tensor> {
    let n: usize = shape.iter().product();
    let mut rng = StdRng::seed_from_u64(seed);
    let data: Vec<f32> = (0..n).map(|_| StandardNormal.sample(&mut rng)).collect();
    Ok(Tensor::from_vec(data, shape, &Device::Cpu)?.to_device(device)?)
}

/// One micro-step's forward+backward over the installed adapter `Var`s: build the noised latent at `t`,
/// predict the **raw** velocity through the (LoRA-adapted) DiT, regress it toward `noise − x0`, and
/// return `(loss, grads)` keyed by `lora_vars`. `(h, w)` is the (constant, per-resolution) latent grid;
/// `text_feats` are the cached, frozen gpt-oss features (any dtype — cast to `compute_dtype` here). A
/// free function so the tests can drive it against a tiny DiT.
///
/// `use_checkpoint` selects the **gradient-checkpointed** backward — required at scale, not just a memory
/// lever: candle's matmul backward materializes a gradient for the *frozen* base weight too, so a dense
/// 48-block backward holds ~48 layers of base-weight grads at once. The checkpointed path runs the
/// adapter-free pre-main forward detached, then segments the per-block stack so only one block's
/// transient weight-grads are live at a time (see [`LensTransformerTrain::main_block_segments`]). Both
/// paths yield the same adapter grads (the `dense_and_checkpoint_grads_match` test pins this).
#[allow(clippy::too_many_arguments)]
fn compute_loss_grads(
    dit: &LensTransformerTrain,
    lora_vars: &[Var],
    x0: &Tensor,
    text_feats: &[Tensor],
    h: usize,
    w: usize,
    t: f64,
    noise: &Tensor,
    mae: bool,
    compute_dtype: DType,
    use_checkpoint: bool,
) -> Result<(f32, GradStore)> {
    let (x_t, target) = build_batch(x0, noise, t)?;
    let x_t = x_t.to_dtype(compute_dtype)?;
    let feats: Vec<Tensor> = text_feats
        .iter()
        .map(|f| f.to_dtype(compute_dtype))
        .collect::<candle_gen::candle_core::Result<_>>()?;
    let timestep = t as f32; // fed to the DiT directly (no 1−σ, no ·1000)

    if use_checkpoint {
        // Pre-main (img/txt embeds, frozen) has no adapters → its `(hidden, encoder)` boundary is a
        // detached constant; the input cotangent is discarded.
        let (hidden, encoder, ctx) = dit.forward_pre_main(&x_t, &feats, None, timestep, 1, h, w)?;
        let hidden_d = hidden.detach();
        let encoder_d = encoder.detach();
        let mut segs = dit.main_block_segments(&ctx);
        // Final segment: head → raw velocity (NO negation) → flow-match regression → [loss].
        let target_owned = target.clone();
        let ctx_ref = &ctx;
        segs.push(Box::new(move |st: &[Tensor]| {
            let v = dit.velocity_out(&st[0], ctx_ref)?;
            Ok(vec![velocity_loss(&v, &target_owned, mae)?])
        }));
        checkpointed_backward(&segs, &[hidden_d, encoder_d], lora_vars)
    } else {
        // Dense backward (tiny models / tests only — see the `use_checkpoint` note re: OOM at scale).
        let v = dit.forward(&x_t, &feats, None, timestep, 1, h, w)?;
        let loss = velocity_loss(&v, &target, mae)?;
        let loss_val = loss.to_dtype(DType::F32)?.to_scalar::<f32>()?;
        let grads = loss.backward()?;
        Ok((loss_val, grads))
    }
}

/// Resolve the sorted `.safetensors` files in the snapshot component subdir `sub`.
fn component_files(root: &Path, sub: &str) -> Result<Vec<PathBuf>> {
    let dir = root.join(sub);
    if !dir.is_dir() {
        return Err(CandleError::Msg(format!(
            "lens trainer: snapshot missing the {sub}/ component directory (at {})",
            root.display()
        )));
    }
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .map_err(|e| CandleError::Msg(format!("lens trainer: read {sub}/: {e}")))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(CandleError::Msg(format!(
            "lens trainer: no .safetensors in {sub}/ (at {})",
            dir.display()
        )));
    }
    Ok(files)
}

/// Build a [`VarBuilder`] over the snapshot component subdir `sub` at `dtype`.
fn component_vb(
    root: &Path,
    sub: &str,
    device: &Device,
    dtype: DType,
) -> Result<VarBuilder<'static>> {
    let files = component_files(root, sub)?;
    // SAFETY: mmap of read-only weight files; standard candle loading path.
    Ok(unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, device)? })
}

/// gpt-oss-encode `caption` → its 4 captured layers cropped at [`TXT_OFFSET`], each `[1, s, 2880]`
/// (f32, cached). Mirrors the inference `encode_one` (single prompt, unpadded). A caption whose token
/// length is `≤ TXT_OFFSET` (the harmony preamble alone) yields length-0 features — surfaced as an error
/// (an empty caption is a dataset bug, not silently trained on zero text).
fn encode_caption(
    tokenizer: &LensTokenizer,
    encoder: &GptOssTextEncoder,
    caption: &str,
    device: &Device,
) -> Result<Vec<Tensor>> {
    let ids = tokenizer
        .encode(caption, DEFAULT_DATE)
        .map_err(|e| CandleError::Msg(format!("lens trainer: tokenize caption: {e}")))?;
    let l = ids.len();
    if l <= TXT_OFFSET {
        return Err(CandleError::Msg(format!(
            "lens trainer: caption {caption:?} tokenizes to {l} tokens (≤ the {TXT_OFFSET}-token \
             harmony preamble) — it carries no text features"
        )));
    }
    let input_ids = Tensor::from_vec(ids, (1, l), device)?;
    let layers = encoder.capture(&input_ids, &DEFAULT_SELECTED_LAYERS)?;
    let s = l - TXT_OFFSET;
    layers
        .iter()
        .map(|f| {
            Ok(f.narrow(1, TXT_OFFSET, s)?
                .to_dtype(DType::F32)?
                .contiguous()?)
        })
        .collect()
}

/// The config's target-module suffixes (default [`LENS_ATTN_TARGETS`]).
fn resolve_target_suffixes(cfg: &TrainingConfig) -> Vec<String> {
    if cfg.lora_target_modules.is_empty() {
        LENS_ATTN_TARGETS.iter().map(|s| s.to_string()).collect()
    } else {
        cfg.lora_target_modules.clone()
    }
}

/// Write the adapter `.safetensors`: LoRA with **bare** dotted keys (empty prefix — Lens DiT keys are
/// bare diffusers paths, what [`crate::adapters`] reads), LoKr with bare keys + metadata.
fn save_adapter(set: &LoraSet, path: &Path) -> Result<()> {
    let meta = HashMap::new();
    match set.kind {
        AdapterKind::Lora => save_lora_peft(set, "", &meta, path),
        AdapterKind::Lokr => save_lokr(set, &meta, path),
    }
}

fn create_output_dir(dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir)
        .map_err(|e| CandleError::Msg(format!("create output dir {}: {e}", dir.display())))
}

/// Install LoRA/LoKr adapters on `dit` for the resolved `suffixes`.
fn install_adapters(
    dit: &mut LensTransformerTrain,
    cfg: &TrainingConfig,
    suffixes: &[String],
    device: &Device,
) -> Result<LoraSet> {
    match cfg.network_type {
        NetworkType::Lora => {
            build_lora_targets(dit, suffixes, cfg.rank, cfg.alpha, cfg.seed, device)
        }
        NetworkType::Lokr => build_lokr_targets(
            dit,
            suffixes,
            cfg.rank,
            cfg.alpha,
            cfg.decompose_factor,
            cfg.seed,
            device,
        ),
    }
}

/// Identity + capabilities of the candle Lens trainer: LoRA + LoKr, `backend = "candle"`.
pub fn trainer_descriptor() -> TrainerDescriptor {
    TrainerDescriptor {
        id: MODEL_ID_BASE,
        family: "lens",
        backend: "candle",
        modality: Modality::Image,
        supports_lora: true,
        supports_lokr: true,
    }
}

/// A loaded candle Lens trainer. Loading is **lazy** — the gpt-oss encoder / VAE / DiT are built inside
/// [`train`](Trainer::train) at the request's compute dtype.
pub struct LensTrainer {
    descriptor: TrainerDescriptor,
    root: PathBuf,
    device: Device,
}

/// Construct the (lazy) candle Lens trainer from a [`LoadSpec`] whose `weights` is the `microsoft/Lens`
/// snapshot directory (`tokenizer/ text_encoder/ transformer/ vae/`).
pub fn load_trainer(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    let root =
        match &spec.weights {
            WeightsSource::Dir(p) => p.clone(),
            WeightsSource::File(_) => return Err(CandleError::Msg(
                "lens trainer expects a snapshot directory (tokenizer/ text_encoder/ transformer/ \
                 vae/), not a single .safetensors file"
                    .into(),
            )),
        };
    Ok(Box::new(LensTrainer {
        descriptor: trainer_descriptor(),
        root,
        device: candle_gen::default_device()?,
    }))
}

fn load_trainer_registered(spec: &LoadSpec) -> gen_core::Result<Box<dyn Trainer>> {
    load_trainer(spec).map_err(Into::into)
}

// Link-time self-registration into gen-core's trainer registry (kept linked by `crate::force_link`).
inventory::submit! {
    gen_core::registry::TrainerRegistration { descriptor: trainer_descriptor, load: load_trainer_registered }
}

impl Trainer for LensTrainer {
    fn descriptor(&self) -> &TrainerDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &TrainingRequest) -> gen_core::Result<()> {
        self.validate_impl(req).map_err(Into::into)
    }

    fn train(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> gen_core::Result<TrainingOutput> {
        self.train_impl(req, on_progress).map_err(Into::into)
    }
}

impl LensTrainer {
    /// Reject a request before any expensive load (mirrors the Wan / Z-Image trainers' guards).
    fn validate_impl(&self, req: &TrainingRequest) -> Result<()> {
        let cfg = &req.config;
        if req.items.is_empty() {
            return Err(CandleError::Msg("lens trainer: dataset is empty".into()));
        }
        if cfg.rank == 0 {
            return Err(CandleError::Msg("lens trainer: rank must be > 0".into()));
        }
        if cfg.steps == 0 {
            return Err(CandleError::Msg("lens trainer: steps must be > 0".into()));
        }
        if !TrainOptimizer::is_supported(&cfg.optimizer) {
            return Err(CandleError::Msg(format!(
                "lens trainer: optimizer '{}' is not available (supported: adamw, adam, rose, prodigy)",
                cfg.optimizer
            )));
        }
        if !TIMESTEP_TYPES.contains(&normalize_cfg(&cfg.timestep_type).as_str()) {
            return Err(CandleError::Msg(format!(
                "lens trainer: timestep_type '{}' is not recognized (supported: {})",
                cfg.timestep_type,
                TIMESTEP_TYPES.join(", ")
            )));
        }
        if !TIMESTEP_BIASES.contains(&normalize_cfg(&cfg.timestep_bias).as_str()) {
            return Err(CandleError::Msg(format!(
                "lens trainer: timestep_bias '{}' is not recognized (supported: {})",
                cfg.timestep_bias,
                TIMESTEP_BIASES.join(", ")
            )));
        }
        if !LOSS_TYPES.contains(&normalize_cfg(&cfg.loss_type).as_str()) {
            return Err(CandleError::Msg(format!(
                "lens trainer: loss_type '{}' is not recognized (supported: {})",
                cfg.loss_type,
                LOSS_TYPES.join(", ")
            )));
        }
        Ok(())
    }

    fn train_impl(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> Result<TrainingOutput> {
        self.validate_impl(req)?;
        let cfg = &req.config;
        let device = &self.device;
        on_progress(TrainingProgress::Preparing);
        let edge = bucket_resolution(cfg.resolution);
        let compute_dtype = parse_compute_dtype(&cfg.train_dtype);
        let dit_cfg = LensDitConfig::lens();

        // --- load + cache: gpt-oss caption features (frozen) + VAE latent means (f32) ---
        on_progress(TrainingProgress::LoadingModel);
        let tokenizer =
            LensTokenizer::from_file(self.root.join("tokenizer").join("tokenizer.json"))?;
        // gpt-oss is the caching workhorse (dense bf16, ~40 GB transient) — built then dropped.
        let encoder = GptOssTextEncoder::new(
            &EncoderConfig::gpt_oss_20b(),
            component_vb(&self.root, "text_encoder", device, ENC_DTYPE)?,
        )?;
        let vae = Flux2Vae::new_with_encoder(component_vb(&self.root, "vae", device, DType::F32)?)?;

        let total = req.items.len() as u32;
        let mut cache: Vec<(Tensor, Vec<Tensor>)> = Vec::with_capacity(req.items.len());
        let mut grid: Option<(usize, usize)> = None;
        for (i, item) in req.items.iter().enumerate() {
            if req.cancel.is_cancelled() {
                break;
            }
            on_progress(TrainingProgress::Caching {
                current: i as u32 + 1,
                total,
            });
            let img = load_image_tensor(&item.image_path, edge, device)?; // [1,3,edge,edge] in [-1,1]
            let (x0, lh, lw) = vae_encode(&vae, &img)?; // [1, S, 128] packed latent (mean), f32
            let feats = encode_caption(&tokenizer, &encoder, &item.caption, device)?;
            grid.get_or_insert((lh, lw));
            cache.push((x0, feats));
        }
        drop(encoder);
        drop(vae);
        if cache.is_empty() {
            if req.cancel.is_cancelled() {
                return Err(CandleError::Canceled);
            }
            return Err(CandleError::Msg(
                "lens trainer: no usable dataset items".into(),
            ));
        }
        let (lat_h, lat_w) = grid.expect("cache is non-empty");

        // --- build the trainable DiT (transformer/) + install the adapter ---
        let suffixes = resolve_target_suffixes(cfg);
        let mut dit = LensTransformerTrain::new(
            &dit_cfg,
            component_vb(&self.root, "transformer", device, compute_dtype)?,
        )?;
        let set = install_adapters(&mut dit, cfg, &suffixes, device)?;
        let accum = cfg.gradient_accumulation.max(1);
        let weight_decay = if cfg.optimizer.eq_ignore_ascii_case("adam") {
            0.0
        } else {
            cfg.weight_decay
        };
        let mae = matches!(normalize_cfg(&cfg.loss_type).as_str(), "mae" | "l1");
        let mut opt = TrainOptimizer::from_config(
            &cfg.optimizer,
            set.vars.clone(),
            cfg.learning_rate,
            weight_decay,
        )?;
        let (total_updates, warmup_updates) =
            schedule_updates(cfg.steps.max(1), accum, cfg.lr_warmup_steps);

        // --- train loop ---
        // The 48-block dense backward is infeasible (candle materializes a grad for every frozen base
        // weight too), so training always uses the gradient-checkpointed backward.
        let use_checkpoint = true;
        let mut accumulated: Option<GradStore> = None;
        let mut micro = 0u32;
        let mut update_idx = 0u32;
        let mut last_loss = 0.0f32;
        let mut steps_run = 0u32;
        for step in 1..=cfg.steps {
            if req.cancel.is_cancelled() {
                break;
            }
            let (x0, feats) = &cache[((step - 1) as usize) % cache.len()];
            let t = sample_timestep(
                &cfg.timestep_type,
                &cfg.timestep_bias,
                cfg.seed.wrapping_mul(0x9E37_79B9).wrapping_add(step as u64),
            );
            let noise = sample_noise(
                x0.dims(),
                cfg.seed.wrapping_add(step as u64).wrapping_mul(2) + 1,
                device,
            )?;
            let (loss, grads) = compute_loss_grads(
                &dit,
                &set.vars,
                x0,
                feats,
                lat_h,
                lat_w,
                t,
                &noise,
                mae,
                compute_dtype,
                use_checkpoint,
            )?;
            last_loss = loss;
            steps_run = step;

            accumulate_grads(&mut accumulated, grads, &set.vars)?;
            micro += 1;
            if micro.is_multiple_of(accum) {
                apply_update(
                    &mut opt,
                    &mut accumulated,
                    &set,
                    accum,
                    cfg,
                    update_idx,
                    total_updates,
                    warmup_updates,
                )?;
                update_idx += 1;
            }

            on_progress(TrainingProgress::Training {
                step,
                total: cfg.steps,
                loss: last_loss,
            });

            if cfg.save_every > 0 && step % cfg.save_every == 0 && step != cfg.steps {
                create_output_dir(&req.output_dir)?;
                let name = checkpoint_filename(file_stem(&req.file_name), step);
                save_adapter(&set, &req.output_dir.join(name))?;
                on_progress(TrainingProgress::Checkpoint { step });
            }
        }

        if steps_run == 0 {
            return Err(CandleError::Canceled);
        }
        // Flush any pending (sub-`accum`) accumulation so the final partial step is applied.
        if accumulated.is_some() {
            apply_update(
                &mut opt,
                &mut accumulated,
                &set,
                accum,
                cfg,
                update_idx,
                total_updates,
                warmup_updates,
            )?;
        }

        // --- save the final adapter ---
        on_progress(TrainingProgress::Saving);
        create_output_dir(&req.output_dir)?;
        let path = req.output_dir.join(&req.file_name);
        save_adapter(&set, &path)?;
        Ok(TrainingOutput {
            adapter_path: path,
            steps: steps_run,
            final_loss: last_loss,
        })
    }
}

/// Fire one optimizer update: LR-schedule, average the accumulated grads, clip, step.
#[allow(clippy::too_many_arguments)]
fn apply_update(
    opt: &mut TrainOptimizer,
    accumulated: &mut Option<GradStore>,
    set: &LoraSet,
    accum: u32,
    cfg: &TrainingConfig,
    update_idx: u32,
    total_updates: u32,
    warmup_updates: u32,
) -> Result<()> {
    let mult = lr_multiplier(cfg.lr_scheduler, update_idx, total_updates, warmup_updates);
    opt.set_lr_scaled(mult);
    let mut avg = accumulated
        .take()
        .expect("apply_update called with a pending accumulation");
    scale_grads(&mut avg, &set.vars, 1.0 / accum as f64)?;
    clip_grad_norm(&mut avg, &set.vars, 1.0)?;
    opt.step(&avg)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_nn::{VarBuilder, VarMap};
    use candle_gen::gen_core::registry;

    /// A tiny Lens-shaped DiT config (2 layers, 2 heads × 8, 1 text layer) — exercises the real
    /// flow-match forward+backward on CPU. Mirrors `dit_train`'s tiny cfg (Σ axes = head_dim).
    fn tiny_cfg() -> LensDitConfig {
        LensDitConfig {
            patch_size: 2,
            in_channels: 32,
            out_channels: 8,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            inner_dim: 16,
            enc_hidden_dim: 12,
            num_text_layers: 1,
            timestep_channels: 16,
            axes_dims_rope: [2, 2, 4],
            rope_theta: 10_000.0,
        }
    }

    /// Randomize every var in a fresh `VarMap` — a zero patch/img_in weight makes `hidden ≡ 0` and the
    /// adapter grads vacuously zero; real training loads nonzero weights, so the tiny tests must too.
    fn randomize_base(vm: &VarMap, dev: &Device) {
        for v in vm.all_vars() {
            v.set(&Tensor::randn(0f32, 0.1f32, v.dims(), dev).unwrap())
                .unwrap();
        }
    }

    /// Tiny synthetic inputs: a packed latent `[1, h·w, in_channels]`, one text-feature layer, noise,
    /// and the latent grid `(h, w)`.
    fn tiny_inputs(
        cfg: &LensDitConfig,
        dev: &Device,
    ) -> (Tensor, Vec<Tensor>, Tensor, usize, usize) {
        let (h, w) = (2usize, 2usize);
        let img_len = h * w;
        let x0 = Tensor::randn(0f32, 1f32, (1, img_len, cfg.in_channels), dev).unwrap();
        let feat = Tensor::randn(0f32, 1f32, (1, 3, cfg.enc_hidden_dim), dev).unwrap();
        let noise = Tensor::randn(0f32, 1f32, (1, img_len, cfg.in_channels), dev).unwrap();
        (x0, vec![feat], noise, h, w)
    }

    /// `build_batch`: `x_t = (1−t)x0 + t·noise`, `target = noise − x0` (raw, no negation).
    #[test]
    fn build_batch_math() {
        let dev = Device::Cpu;
        let x0 = Tensor::from_vec(vec![2.0f32, 4.0], (1, 2), &dev).unwrap();
        let noise = Tensor::from_vec(vec![1.0f32, 0.0], (1, 2), &dev).unwrap();
        let (x_t, target) = build_batch(&x0, &noise, 0.25).unwrap();
        assert_eq!(x_t.to_vec2::<f32>().unwrap(), vec![vec![1.75, 3.0]]);
        assert_eq!(target.to_vec2::<f32>().unwrap(), vec![vec![-1.0, -4.0]]);
    }

    /// Timestep sampling is deterministic and in `(0, 1)`.
    #[test]
    fn timestep_is_in_range_and_deterministic() {
        for seed in [0u64, 1, 42, 9999] {
            let a = sample_timestep("sigmoid", "balanced", seed);
            let b = sample_timestep("sigmoid", "balanced", seed);
            assert_eq!(a, b, "same seed reproduces");
            assert!(a > 0.0 && a < 1.0, "t out of range: {a}");
        }
    }

    /// The keystone training gate: a real flow-match forward+backward over the tiny DiT with nonzero
    /// LoRA factors yields a finite loss and a gradient on **every** adapter `Var` (save the last block's
    /// `to_add_out`, whose text-stream output the image-velocity head discards — see `dit_train`).
    #[test]
    fn backward_reaches_lora_factors() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let mut dit = LensTransformerTrain::new(&cfg, vb).unwrap();
        randomize_base(&vm, &dev);
        let suffixes: Vec<String> = LENS_ATTN_TARGETS.iter().map(|s| s.to_string()).collect();
        let set = build_lora_targets(&mut dit, &suffixes, 4, 8.0, 7, &dev).unwrap();
        // Move B off its zero-init so both A and B grads are nonzero.
        for v in &set.vars {
            v.set(&Tensor::randn(0f32, 0.02f32, v.as_tensor().dims(), &dev).unwrap())
                .unwrap();
        }
        let (x0, feats, noise, h, w) = tiny_inputs(&cfg, &dev);
        let (loss, grads) = compute_loss_grads(
            &dit,
            &set.vars,
            &x0,
            &feats,
            h,
            w,
            0.5,
            &noise,
            false,
            DType::F32,
            false,
        )
        .unwrap();
        assert!(loss.is_finite(), "loss must be finite, got {loss}");
        let mut saw_nonzero = false;
        for v in &set.vars {
            if let Some(g) = grads.get(v.as_tensor()) {
                let gv = g.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                assert!(gv.iter().all(|x| x.is_finite()), "non-finite gradient");
                if gv.iter().any(|x| x.abs() > 1e-9) {
                    saw_nonzero = true;
                }
            }
        }
        assert!(saw_nonzero, "backprop is not reaching the adapter factors");
        assert_eq!(set.vars.len(), 4 * 2 * cfg.num_layers); // 4 projections × 2 factors × layers
    }

    /// The correctness gate for the gradient-checkpointed backward (the path real training always uses):
    /// it must reproduce the dense `loss.backward()` grads (mod float reassociation) on the tiny DiT.
    #[test]
    fn dense_and_checkpoint_grads_match() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let mut dit = LensTransformerTrain::new(&cfg, vb).unwrap();
        randomize_base(&vm, &dev);
        let suffixes: Vec<String> = LENS_ATTN_TARGETS.iter().map(|s| s.to_string()).collect();
        let set = build_lora_targets(&mut dit, &suffixes, 4, 8.0, 7, &dev).unwrap();
        for v in &set.vars {
            v.set(&Tensor::randn(0f32, 0.02f32, v.as_tensor().dims(), &dev).unwrap())
                .unwrap();
        }
        let (x0, feats, noise, h, w) = tiny_inputs(&cfg, &dev);
        let (loss_d, g_d) = compute_loss_grads(
            &dit,
            &set.vars,
            &x0,
            &feats,
            h,
            w,
            0.5,
            &noise,
            false,
            DType::F32,
            false,
        )
        .unwrap();
        let (loss_c, g_c) = compute_loss_grads(
            &dit,
            &set.vars,
            &x0,
            &feats,
            h,
            w,
            0.5,
            &noise,
            false,
            DType::F32,
            true,
        )
        .unwrap();
        assert!(
            (loss_d - loss_c).abs() < 1e-4,
            "loss: dense {loss_d} vs checkpoint {loss_c}"
        );
        let mut saw_nonzero = false;
        for (i, v) in set.vars.iter().enumerate() {
            // A var with no dense grad (the discarded last-block to_add_out) is skipped in both paths.
            let (Some(a), Some(b)) = (g_d.get(v.as_tensor()), g_c.get(v.as_tensor())) else {
                continue;
            };
            let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let b = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            assert_eq!(a.len(), b.len());
            for (x, y) in a.iter().zip(b.iter()) {
                assert!(
                    (x - y).abs() < 1e-4,
                    "grad mismatch for var {i} (dense {x} vs checkpoint {y})"
                );
                if x.abs() > 1e-6 {
                    saw_nonzero = true;
                }
            }
        }
        assert!(saw_nonzero, "expected nonzero adapter grads to compare");
    }

    /// A few optimizer steps on a fixed batch lower the loss — the step descends the flow-match
    /// objective end to end through the harness.
    #[test]
    fn one_optimizer_step_descends() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let mut dit = LensTransformerTrain::new(&cfg, vb).unwrap();
        randomize_base(&vm, &dev);
        let suffixes: Vec<String> = LENS_ATTN_TARGETS.iter().map(|s| s.to_string()).collect();
        let set = build_lora_targets(&mut dit, &suffixes, 4, 8.0, 7, &dev).unwrap();
        for v in &set.vars {
            v.set(&Tensor::randn(0f32, 0.02f32, v.as_tensor().dims(), &dev).unwrap())
                .unwrap();
        }
        let (x0, feats, noise, h, w) = tiny_inputs(&cfg, &dev);
        let mut opt = TrainOptimizer::from_config("adamw", set.vars.clone(), 1e-2, 0.0).unwrap();
        let loss_at = |dit: &LensTransformerTrain| {
            compute_loss_grads(
                dit,
                &set.vars,
                &x0,
                &feats,
                h,
                w,
                0.5,
                &noise,
                false,
                DType::F32,
                false,
            )
            .unwrap()
        };
        let (loss0, mut grads) = loss_at(&dit);
        for _ in 0..5 {
            clip_grad_norm(&mut grads, &set.vars, 1.0).unwrap();
            opt.step(&grads).unwrap();
            grads = loss_at(&dit).1;
        }
        let (loss1, _) = loss_at(&dit);
        assert!(
            loss1 < loss0,
            "5 steps on a fixed batch should lower the loss: {loss0} -> {loss1}"
        );
    }

    /// The trainer self-registers and resolves through gen-core's trainer registry as the candle Lens
    /// trainer; `load_trainer` is lazy, so a nonexistent weights dir still resolves.
    #[test]
    fn trainer_registers_and_resolves_as_candle() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let t = registry::load_trainer(MODEL_ID_BASE, &spec)
            .expect("candle lens trainer is registered");
        assert_eq!(t.descriptor().id, MODEL_ID_BASE);
        assert_eq!(t.descriptor().backend, "candle");
        assert_eq!(t.descriptor().modality, Modality::Image);
        assert!(t.descriptor().supports_lora && t.descriptor().supports_lokr);
    }

    /// `validate` rejects an empty dataset, zero rank/steps, an unsupported optimizer, and unrecognized
    /// timestep/loss knobs — before any load.
    #[test]
    fn validate_rejects_bad_requests() {
        use candle_gen::gen_core::runtime::CancelFlag;
        use candle_gen::gen_core::train::TrainingItem;
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let t = registry::load_trainer(MODEL_ID_BASE, &spec).unwrap();
        let base = TrainingRequest {
            items: vec![TrainingItem {
                image_path: "/img.png".into(),
                caption: "x".into(),
            }],
            config: TrainingConfig::default(),
            output_dir: "/out".into(),
            file_name: "a.safetensors".into(),
            trigger_words: vec![],
            cancel: CancelFlag::new(),
        };
        assert!(t.validate(&base).is_ok());
        let bad = |mutate: &dyn Fn(&mut TrainingRequest)| {
            let mut r = base.clone();
            mutate(&mut r);
            assert!(t.validate(&r).is_err());
        };
        bad(&|r| r.items.clear());
        bad(&|r| r.config.rank = 0);
        bad(&|r| r.config.steps = 0);
        bad(&|r| r.config.optimizer = "lion".into());
        bad(&|r| r.config.timestep_type = "bogus".into());
        bad(&|r| r.config.loss_type = "huber".into());
    }
}
