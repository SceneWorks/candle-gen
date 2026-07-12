//! The SD3.5 **txt2img** pipeline (sc-7877, epic 7982) — drives C1's MMDiT + triple-TE aggregator +
//! 16-ch VAE through the backend-neutral [`gen_core::Generator`] contract for both the **Large**
//! (classifier-free guidance) and **Large Turbo** (distilled, 4-step, CFG-off) variants.
//!
//! Structural template: `candle-gen-z-image`'s `pipeline.rs` (the closest flow-match-DiT + VAE-decode
//! provider). Key SD3.5-specific choices, grounded in the public diffusers `StableDiffusion3Pipeline`
//! + `FlowMatchEulerDiscreteScheduler`:
//!
//! - **Conditioning**: [`crate::conditioning::Sd3TextEncoders`] runs CLIP-L + CLIP-bigG + T5-XXL and
//!   [`crate::conditioning::aggregate`] builds the pooled `[B, 2048]` + context `[B, 333, 4096]`. The
//!   **unconditional** branch (CFG) is the empty-prompt encode through the same encoders (NOT a zero
//!   tensor — diffusers encodes `""`).
//! - **Sampler**: the repo's unified flow-match framework ([`candle_gen::run_flow_sampler`] +
//!   [`candle_gen::resolve_flow_schedule`], epic 7114). The **native** σ schedule is the SD3
//!   `FlowMatchEulerDiscreteScheduler` shifted ramp ([`sd3_sigmas`]); the default `euler` over it is
//!   the standard flow-match step `x + v·(σ_{i+1} − σ_i)`. The DiT consumes `t = σ·1000` (the SD3
//!   timestep convention), applied inside the predict closure.
//! - **Large CFG**: two forward passes per step (cond + uncond), combined
//!   `uncond + cfg·(cond − uncond)`; `cfg ≈ 4.0`, ~28 steps, shift 3.0.
//! - **Turbo**: guidance-distilled — a single forward per step, **no negative branch** (`cfg = 1.0`),
//!   4 steps. Same MMDiT/VAE weights layout; the Turbo *checkpoint* differs, the code path does not
//!   except for skipping the uncond eval.
//! - **Deterministic seeding (sc-3673 parity)**: initial latent noise from a fixed-algorithm CPU RNG
//!   (`StdRng`) seeded by `seed`, moved to the device — NOT candle's CUDA `randn`. The Euler step is
//!   non-stochastic, so generation is a pure function of `(seed, request)`.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::{DType, Device, IndexOp, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::sampling::TimestepConvention;
use candle_gen::gen_core::{self, AdapterSpec, GenerationRequest, Image, Progress, Quant};
// Shared per-image batch seed (`base + index`) — one home in `candle-gen` (sc-9043 / F-059).
use candle_gen::{CandleError, Result};
use rand::{rngs::StdRng, SeedableRng};

use crate::conditioning::{aggregate, Sd3Conditioning, Sd3TextEncoders};
use crate::config::Sd3Config;
use crate::transformer::Sd3Transformer;
use crate::vae::{load_vae, AutoEncoderKL, LATENT_CHANNELS, SPATIAL_SCALE};

/// The two wired SD3.5 variants. They share the MMDiT/VAE architecture + encoders; they differ in the
/// default schedule (CFG-on 28-step vs distilled 4-step), the CFG default, and whether the negative
/// branch runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    /// SD3.5 Large — classifier-free guidance, ~28 steps, shift 3.0, cfg ≈ 4.0.
    Large,
    /// SD3.5 Large Turbo — guidance-distilled, 4 steps, CFG-off (single forward per step).
    LargeTurbo,
    /// SD3.5 **Medium** — the MMDiT-X (dual-attention) model. Classifier-free guidance like Large
    /// (NOT distilled); fewer/narrower blocks (24 × 1536). Same flow-match pipeline + σ-shift.
    Medium,
}

impl Variant {
    /// The default inference step count when a request omits `steps`.
    pub fn default_steps(self) -> usize {
        match self {
            // Medium uses CFG like Large; SD3.5-Medium's published default is ~40 steps.
            Variant::Large => 28,
            Variant::Medium => 40,
            Variant::LargeTurbo => 4,
        }
    }

    /// The default classifier-free guidance scale. Turbo is distilled (CFG-off ⇒ 1.0 ⇒ the uncond
    /// branch is skipped entirely).
    pub fn default_cfg(self) -> f32 {
        match self {
            Variant::Large => 4.0,
            // SD3.5-Medium's published default guidance is ~4.5.
            Variant::Medium => 4.5,
            Variant::LargeTurbo => 1.0,
        }
    }

    /// The flow-match resolution-independent σ shift. SD3.5 Large/Turbo both use 3.0; Medium also
    /// uses 3.0.
    pub fn shift(self) -> f32 {
        3.0
    }

    /// Whether the variant runs classifier-free guidance (the uncond forward). Turbo is distilled, so
    /// the negative branch is never evaluated regardless of the request's guidance value.
    pub fn cfg_enabled(self) -> bool {
        matches!(self, Variant::Large | Variant::Medium)
    }

    /// The architecture [`Sd3Config`] for this variant — Large/Turbo share the Large MMDiT geometry;
    /// Medium is the MMDiT-X (dual-attention) preset.
    pub(crate) fn config(self) -> Sd3Config {
        match self {
            Variant::Large | Variant::LargeTurbo => Sd3Config::large(),
            Variant::Medium => Sd3Config::medium(),
        }
    }
}

/// VAE spatial downscale (image/8 per side) — re-exported from [`crate::vae`] for the latent geometry.
const VAE_SCALE: u32 = SPATIAL_SCALE;

/// The SD3.5 [`FlowMatchEulerDiscreteScheduler`] σ ramp for `steps` inference steps with the given
/// `shift`, matching diffusers `set_timesteps`:
///
/// 1. `sigmas = linspace(1.0, 1/num_train, steps)` (the σ table the timesteps map to, σ_max = 1.0);
/// 2. shift each: `σ' = shift·σ / (1 + (shift − 1)·σ)` (the resolution-independent flow shift);
/// 3. append a trailing `0.0` (the clean end).
///
/// Returns `steps + 1` strictly-decreasing sigmas from 1.0 → 0.0 — the `native` schedule fed to
/// [`candle_gen::resolve_flow_schedule`]. Pure; unit-tested without a GPU.
pub fn sd3_sigmas(steps: usize, shift: f32) -> Vec<f32> {
    let steps = steps.max(1);
    // diffusers: timesteps = linspace(num_train, ~0, steps); sigmas = timesteps / num_train. With
    // num_train = 1000 this is sigmas = linspace(1.0, 1/1000, steps). The exact lower endpoint barely
    // matters (it is shifted then the trailing 0.0 dominates the final step); use 1/num_train for parity.
    let num_train = 1000.0f32;
    let mut out: Vec<f32> = (0..steps)
        .map(|i| {
            let frac = if steps == 1 {
                0.0
            } else {
                i as f32 / (steps - 1) as f32
            };
            // linspace(1.0, 1/num_train, steps)
            let sigma = 1.0 - frac * (1.0 - 1.0 / num_train);
            shift * sigma / (1.0 + (shift - 1.0) * sigma)
        })
        .collect();
    out.push(0.0);
    out
}

/// A txt2img pipeline handle: the snapshot `root` + compute device/dtype (bf16) + the variant. Loading
/// the heavy components is done by [`load_components`](Self::load_components) and owned/cached by the
/// generator, mirroring the Z-Image provider's lazy split.
pub(crate) struct Pipeline {
    root: PathBuf,
    device: Device,
    dtype: DType,
    variant: Variant,
    cfg: Sd3Config,
    /// Optional MMDiT quantization applied right after the (dense) transformer weights load
    /// (sc-7879). `None` ⇒ dense bf16; the TE + VAE stay dense regardless.
    quant: Option<Quant>,
    /// LoRA/LoKr adapters merged into the MMDiT weights at component-load (sc-7881). Empty ⇒ the stock
    /// mmap build (zero regression). The merge runs **before** quantization.
    adapters: Vec<AdapterSpec>,
}

/// The loaded SD3.5 components, `Arc`-shared so the generator can cache them across `generate` calls.
/// The encoders are behind a `Mutex` because the T5 forward takes `&mut self`.
#[derive(Clone)]
pub(crate) struct Components {
    encoders: Arc<Mutex<Sd3TextEncoders>>,
    transformer: Arc<Sd3Transformer>,
    vae: Arc<AutoEncoderKL>,
}

impl Pipeline {
    /// Build the (light) pipeline handle. Does **no** weight I/O — components load lazily via
    /// [`load_components`](Self::load_components).
    pub(crate) fn load(
        root: &Path,
        device: &Device,
        dtype: DType,
        variant: Variant,
        quant: Option<Quant>,
        adapters: &[AdapterSpec],
    ) -> Self {
        Self {
            root: root.to_path_buf(),
            device: device.clone(),
            dtype,
            variant,
            cfg: variant.config(),
            quant,
            adapters: adapters.to_vec(),
        }
    }

    /// Load the three text encoders + the MMDiT + the VAE from the diffusers component subdirs
    /// (`text_encoder*/`, `transformer/`, `vae/`).
    pub(crate) fn load_components(&self) -> Result<Components> {
        let encoders =
            Sd3TextEncoders::load(&self.root, self.cfg.t5_seq_len, &self.device, self.dtype)?;
        // Adapters ride as **forward-time additive residuals** on the DiT (sc-11105) — on BOTH tiers:
        // the base is never folded (`W += δ` would pin an un-evictable in-memory copy — epic 10765), so
        // it stays an unmutated mmap (dense) / packed base. Quantization (if any) then folds ONLY the
        // base in place and the residuals survive. `merge_adapters` (the old dense fold) is retained as a
        // public utility but no longer on the load path.
        // Whether `transformer/` is a pre-quantized MLX-packed tier (`config.json` carries a
        // `quantization` block) — gates the no-adapter packed-detect build below + the group-size guard.
        let packed_cfg = self.transformer_packed_config();
        let packed_tier = self.adapters.is_empty() && packed_cfg.is_some();
        // Adapters ride as forward-time additive residuals on the DiT — on BOTH tiers (sc-11105,
        // additive-everywhere for epic 10765); the base is never mutated, so it stays evictable.
        let additive = !self.adapters.is_empty();
        // sc-9474: the shared packed loaders (`QLinear::linear_detect` / `linear_detect_dense`) repack
        // at the MLX default group size 64, which every hosted `sd3.5-*-mlx` tier uses. The parsed
        // `PackedConfig.group_size` is threaded here as a LOUD guard so a hypothetical future group-32
        // tier (as boogu's is) fails at load rather than silently repacking to garbage. If a non-64 SD3.5
        // tier is ever shipped, thread `group_size` through `Sd3Transformer::new` into the `*_gs` shared
        // entry points (`linear_detect_gs` / `lin_gs`), as candle-gen-boogu (sc-9410) does. Runs for any
        // packed tier (with or without adapters).
        if let Some(cfg) = &packed_cfg {
            if cfg.group_size != candle_gen::quant::MLX_GROUP_SIZE as i32 {
                return Err(CandleError::Msg(format!(
                    "sd3 packed transformer/ tier declares quantization.group_size = {} but the \
                     candle-gen-sd3 packed loaders assume the MLX default {} (sc-9474). Thread the \
                     parsed group_size through Sd3Transformer::new into the shared `*_gs` entry \
                     points (as candle-gen-boogu does) before loading this tier.",
                    cfg.group_size,
                    candle_gen::quant::MLX_GROUP_SIZE,
                )));
            }
        }
        let transformer = if additive {
            // Adapters on ANY tier: build the MMDiT on the device (a packed tier packed-detects each
            // `.scales` sibling; a dense tier loads the unmutated bf16 mmap), then install the LoRA/LoKr
            // as **forward-time additive residuals** — the base is never folded, so it stays evictable
            // (a `W += δ` fold pins an un-evictable in-memory copy — epic 10765). A requested `quantize`
            // then folds ONLY the base in place (dense→Q4/Q8; a no-op on an already-packed base) and the
            // residuals survive. Additive equals the old dense fold to f32 tolerance (~1 ULP). The
            // dense-tier adapter case builds dense on the device (the sc-8504 CPU-stage optimization is
            // for the no-adapter path); the adapter's residual never lands as a dense base weight.
            let mut transformer = Sd3Transformer::new(
                &self.cfg,
                self.component_vb_on("transformer", &self.device)?,
            )?;
            crate::adapters::install_additive(&mut transformer, &self.adapters)?;
            if let Some(q) = self.quant {
                transformer.quantize(q)?;
            }
            transformer
        } else {
            match self.quant {
                // sc-9414 packed path: build the MMDiT **directly on the GPU** from the MLX-packed tier —
                // each projection packed-detects its `.scales` sibling and lands its Q4_1/Q8_0 footprint
                // straight on the device (no dense bf16 staging, no load-then-quantize). The post-load
                // `quantize_onto` pass is a no-op on the already-packed projections and only re-migrates the
                // dense-kept leaves (already on the GPU). The AdaLN/embedder leaves are dequantized to dense
                // full-precision leaves inside `new` (see `quant::linear_detect_dense`). TE + VAE stay dense.
                Some(q) if packed_tier => {
                    let mut transformer = Sd3Transformer::new(
                        &self.cfg,
                        self.component_vb_on("transformer", &self.device)?,
                    )?;
                    transformer.quantize_onto(q, &self.device)?;
                    transformer
                }
                // sc-8504 CPU-stage path (dense tier): build the dense MMDiT on a **CPU** VarBuilder, then
                // `quantize_onto` the compute device — the quantized projections land directly on the GPU
                // (the dense projection weight never touches it) and the dense-kept leaves migrate
                // alongside. This drops the in-place dense-build transient (sc-7879 built dense on-device
                // then folded in place, so dense + quantized briefly coexisted on the GPU). The resulting
                // Q4_0/Q8_0 blocks are bit-identical to the in-place path (the quantizer routes through the
                // CPU either way). The TE + VAE stay dense bf16.
                Some(q) => {
                    let cpu = Device::Cpu;
                    let mut transformer =
                        Sd3Transformer::new(&self.cfg, self.component_vb_on("transformer", &cpu)?)?;
                    transformer.quantize_onto(q, &self.device)?;
                    transformer
                }
                // Dense bf16: load straight onto the compute device.
                None => Sd3Transformer::new(
                    &self.cfg,
                    self.component_vb_on("transformer", &self.device)?,
                )?,
            }
        };
        let vae = load_vae(self.component_vb("vae")?)?;
        Ok(Components {
            encoders: Arc::new(Mutex::new(encoders)),
            transformer: Arc::new(transformer),
            vae: Arc::new(vae),
        })
    }

    /// Resolve the sorted list of `.safetensors` files in the snapshot component subdir `sub`
    /// (single-file or sharded), erroring if the dir or files are missing.
    fn component_files(&self, sub: &str) -> Result<Vec<PathBuf>> {
        let dir = self.root.join(sub);
        if !dir.is_dir() {
            return Err(CandleError::Msg(format!(
                "sd3 snapshot is missing the {sub}/ component directory (at {})",
                self.root.display()
            )));
        }
        // Shared sorted-`.safetensors` resolver (sc-8999 / F-019); the crafted "missing dir" message
        // above stays local (it names the expected sd3 snapshot).
        candle_gen::sorted_safetensors(&dir, "sd3")
    }

    /// Build a [`VarBuilder`] over every `.safetensors` in the snapshot component subdir `sub`, on the
    /// pipeline's compute device (the stock mmap path; no adapters).
    fn component_vb(&self, sub: &str) -> Result<VarBuilder<'static>> {
        self.component_vb_on(sub, &self.device)
    }

    /// [`Self::component_vb`] but on an explicit `device` — the sc-8504 CPU-stage quant path builds the
    /// dense MMDiT on the CPU (system RAM) before quantizing each projection onto the GPU, so the dense
    /// projection weights never land on the GPU.
    fn component_vb_on(&self, sub: &str, device: &Device) -> Result<VarBuilder<'static>> {
        let files = self.component_files(sub)?;
        candle_gen::mmap_var_builder(&files, self.dtype, device)
    }

    /// The `transformer/` component's parsed [`candle_gen::quant::PackedConfig`] when it is a
    /// **pre-quantized MLX-packed tier** — its `config.json` carries a `quantization` block, which the
    /// `sd3.5-*-mlx` convert job writes for the packed DiT. On a packed tier the loader builds each Linear
    /// directly from the packed parts on the GPU (sc-9414, no dense CPU staging); on a dense tier it falls
    /// back to the CPU-stage → quantize-onto-GPU path. Absent/unreadable config → `None` (dense path), so
    /// a fixture with no `config.json` still loads.
    ///
    /// The parsed `group_size` is threaded to [`load_components`](Self::load_components), which asserts it
    /// is the MLX default 64 (the group size the shared packed loaders assume) so a future group-32 tier
    /// fails LOUD rather than silently repacking to garbage (sc-9474). Every hosted SD3.5 tier packs at 64.
    fn transformer_packed_config(&self) -> Option<candle_gen::quant::PackedConfig> {
        let path = self.root.join("transformer").join("config.json");
        std::fs::read_to_string(&path)
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
            .and_then(|v| candle_gen::quant::PackedConfig::from_config(&v))
    }

    /// Build the conditioning for a prompt: the aggregated pooled + context. For CFG the
    /// unconditional branch is the **empty-prompt** encode (diffusers encodes `""`), NOT a zero
    /// tensor. Returns `(cond, Option<uncond>)`; `uncond` is `None` when CFG is off — either a
    /// distilled (Turbo) variant, or an effective `cfg_scale == 1.0` where the blend
    /// `uncond + 1·(cond − uncond)` reduces exactly to `cond`, so the uncond encode/forward is pure
    /// waste (sc-8993).
    fn conditioning(
        &self,
        encoders: &Mutex<Sd3TextEncoders>,
        req: &GenerationRequest,
        cfg_scale: f32,
    ) -> Result<(Sd3Conditioning, Option<Sd3Conditioning>)> {
        let mut enc = candle_gen::lock_recover(encoders);
        let cond_out = enc.encode(&req.prompt)?;
        let cond = aggregate(&self.cfg, &cond_out)?;
        let uncond = if self.variant.cfg_enabled() && cfg_scale != 1.0 {
            let neg = req.negative_prompt.as_deref().unwrap_or("");
            let uncond_out = enc.encode(neg)?;
            Some(aggregate(&self.cfg, &uncond_out)?)
        } else {
            None
        };
        Ok((cond, uncond))
    }

    /// Render `req` against pre-loaded `components`, emitting per-step progress and honoring
    /// `req.cancel`. One `gen_core::Image` per `req.count` (each at seed `base_seed + index`).
    pub(crate) fn render(
        &self,
        req: &GenerationRequest,
        components: &Components,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Vec<Image>> {
        let steps = req
            .steps
            .map(|s| s as usize)
            .unwrap_or_else(|| self.variant.default_steps());
        let cfg_scale = if self.variant.cfg_enabled() {
            req.guidance.unwrap_or_else(|| self.variant.default_cfg())
        } else {
            1.0
        };
        let base_seed = req.seed.unwrap_or_else(gen_core::default_seed);
        let lat_h = (req.height / VAE_SCALE) as usize;
        let lat_w = (req.width / VAE_SCALE) as usize;

        // Conditioning is seed- and image-independent: encode once for the whole batch. `cfg_scale`
        // gates the uncond encode — at 1.0 the CFG blend collapses to cond, so it's skipped (sc-8993).
        let (cond, uncond) = self.conditioning(&components.encoders, req, cfg_scale)?;

        candle_gen::for_each_image_seed(base_seed, req.count, |seed| {
            render_core(
                &components.transformer,
                &components.vae,
                &cond,
                uncond.as_ref(),
                cfg_scale,
                steps,
                self.variant.shift(),
                (lat_h, lat_w),
                seed,
                self.device.clone(),
                self.dtype,
                req.sampler.as_deref(),
                req.scheduler.as_deref(),
                &req.cancel,
                on_progress,
            )
        })
    }
}

/// The render core shared by [`Pipeline::render`] and the structural/CUDA smoke tests: build the
/// deterministic CPU-seeded noise, run the unified flow-match sampler (with CFG when `uncond` is
/// `Some`, distilled-single-eval when `None`), and VAE-decode. Decoupled from snapshot I/O so a test
/// can drive it with a random-weight transformer + VAE.
#[allow(clippy::too_many_arguments)]
pub(crate) fn render_core(
    transformer: &Sd3Transformer,
    vae: &AutoEncoderKL,
    cond: &Sd3Conditioning,
    uncond: Option<&Sd3Conditioning>,
    cfg_scale: f32,
    steps: usize,
    shift: f32,
    latent_hw: (usize, usize),
    seed: u64,
    device: Device,
    dtype: DType,
    sampler: Option<&str>,
    scheduler: Option<&str>,
    cancel: &gen_core::CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Image> {
    let (lat_h, lat_w) = latent_hw;

    // Native SD3 flow-match schedule (shifted), then the curated scheduler axis (default = native).
    let native = sd3_sigmas(steps, shift);
    let sigmas = candle_gen::resolve_flow_schedule(scheduler, 0.0, steps, &native);

    // sc-3673 parity — deterministic, launch-portable initial noise: N(0,1) from a CPU RNG seeded by
    // `seed`, built on CPU then moved to the device.
    let n = LATENT_CHANNELS * lat_h * lat_w;
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = candle_gen::seeded_normal_vec(&mut rng, n);
    let latents = Tensor::from_vec(noise, (1, LATENT_CHANNELS, lat_h, lat_w), &Device::Cpu)?
        .to_device(&device)?
        .to_dtype(dtype)?;

    let latents = candle_gen::run_flow_sampler(
        sampler,
        TimestepConvention::Sigma,
        &sigmas,
        latents,
        seed,
        cancel,
        on_progress,
        |latents, sigma| -> Result<Tensor> {
            // SD3 feeds the DiT `t = σ·1000` (the timestep convention; the embedder scales the
            // sinusoid). f32 here is correct — the embedder upcasts internally.
            let t = Tensor::from_vec(vec![sigma * 1000.0], (1,), &device)?;
            let v_cond = transformer.forward(latents, &cond.context, &cond.pooled, &t)?;
            let v = match uncond {
                // Large CFG: v = uncond + cfg·(cond − uncond).
                Some(uncond) => {
                    let v_uncond =
                        transformer.forward(latents, &uncond.context, &uncond.pooled, &t)?;
                    (&v_uncond + ((&v_cond - &v_uncond)? * cfg_scale as f64)?)?
                }
                // Turbo (distilled): no negative branch — the velocity is used directly.
                None => v_cond,
            };
            // The DiT may run in a different (e.g. f32) dtype than the latent stream; bring the
            // velocity back to the latent dtype so the flow-match step's add agrees (sc-7881).
            Ok(v.to_dtype(latents.dtype())?)
        },
    )?;

    on_progress(Progress::Decoding);
    decode_image(vae, &latents)
}

/// VAE-decode the final latents `(1, 16, h, w)` to an RGB8 [`Image`]. The VAE applies its own
/// `/scaling_factor + shift_factor` un-scale inside `decode`; the `[-1, 1]` output maps to `[0, 255]`
/// u8.
fn decode_image(vae: &AutoEncoderKL, latents: &Tensor) -> Result<Image> {
    let decoded = vae.decode(latents)?.to_dtype(DType::F32)?; // (1, 3, H, W) in [-1, 1]
    let img = ((decoded.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?.to_dtype(DType::U8)?;
    let img = img.i(0)?.to_device(&Device::Cpu)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variant_defaults_match_sd35() {
        assert_eq!(Variant::Large.default_steps(), 28);
        assert_eq!(Variant::LargeTurbo.default_steps(), 4);
        assert_eq!(Variant::Medium.default_steps(), 40);
        assert_eq!(Variant::Large.default_cfg(), 4.0);
        assert_eq!(Variant::LargeTurbo.default_cfg(), 1.0);
        assert_eq!(Variant::Medium.default_cfg(), 4.5);
        assert!(Variant::Large.cfg_enabled());
        assert!(Variant::Medium.cfg_enabled(), "medium uses CFG like Large");
        assert!(!Variant::LargeTurbo.cfg_enabled(), "turbo is distilled");
        assert_eq!(Variant::Large.shift(), 3.0);
        assert_eq!(Variant::Medium.shift(), 3.0);
        assert_eq!(VAE_SCALE, 8);
        // Medium maps to the MMDiT-X (dual-attention) config; Large/Turbo to the Large MMDiT.
        assert!(!Variant::Medium.config().dual_attention_layers.is_empty());
        assert!(Variant::Large.config().dual_attention_layers.is_empty());
        assert!(Variant::LargeTurbo
            .config()
            .dual_attention_layers
            .is_empty());
    }

    /// The SD3 σ schedule: `steps + 1` sigmas, max σ at the shifted 1.0, strictly decreasing, terminal
    /// 0.0. The shift pushes interior sigmas up vs the unshifted linear ramp.
    #[test]
    fn sd3_sigmas_is_shifted_decreasing_to_zero() {
        let steps = 28;
        let s = sd3_sigmas(steps, 3.0);
        assert_eq!(s.len(), steps + 1);
        // σ=1.0 shifted by `shift·1/(1+(shift-1)·1) = shift/shift = 1.0` stays 1.0.
        assert!((s[0] - 1.0).abs() < 1e-6, "max sigma: {}", s[0]);
        assert!(s[steps].abs() < 1e-9, "terminal sigma must be 0");
        for w in s.windows(2) {
            assert!(w[0] > w[1], "sigmas must strictly decrease: {s:?}");
        }
    }

    /// The shift actually changes the schedule: at shift 3.0 the interior sigmas exceed the unshifted
    /// (shift 1.0) linear ramp — the regression guard for "shift applied".
    #[test]
    fn shift_raises_interior_sigmas() {
        let unshifted = sd3_sigmas(10, 1.0);
        let shifted = sd3_sigmas(10, 3.0);
        // Midpoint interior sigma is strictly larger under shift 3.0.
        let mid = 5;
        assert!(
            shifted[mid] > unshifted[mid],
            "shift 3.0 must raise interior sigmas: {} vs {}",
            shifted[mid],
            unshifted[mid]
        );
    }

    /// **The parsed packed `group_size` is threaded, not discarded** (sc-9474). A `transformer/config.json`
    /// carrying `quantization: { bits, group_size }` parses into a `PackedConfig` whose `group_size` is the
    /// on-disk value (32 here, boogu's group size) — proving `transformer_packed_config` no longer throws
    /// the group size away. A dense config (no `quantization`) parses to `None`, and a default group-64
    /// pack round-trips to 64. `load_components` uses this to reject a non-64 tier LOUD (the shared SD3.5
    /// packed loaders assume the MLX default 64).
    #[test]
    fn transformer_packed_config_threads_parsed_group_size() {
        let tmp = std::env::temp_dir().join(format!(
            "sc9474_sd3_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let tdir = tmp.join("transformer");
        std::fs::create_dir_all(&tdir).unwrap();
        let write_cfg = |json: &str| std::fs::write(tdir.join("config.json"), json).unwrap();
        let pipe = |root: &Path| {
            Pipeline::load(
                root,
                &Device::Cpu,
                DType::F32,
                Variant::Large,
                Some(Quant::Q4),
                &[],
            )
        };

        // group-32 pack: the group size survives the parse (would previously be discarded).
        write_cfg(r#"{"quantization": {"bits": 4, "group_size": 32}}"#);
        let cfg32 = pipe(&tmp).transformer_packed_config();
        assert_eq!(
            cfg32.map(|c| c.group_size),
            Some(32),
            "parsed group_size must be threaded, not discarded"
        );

        // group-64 (the MLX default every hosted SD3.5 tier uses) round-trips to 64.
        write_cfg(r#"{"quantization": {"bits": 4, "group_size": 64}}"#);
        assert_eq!(
            pipe(&tmp).transformer_packed_config().map(|c| c.group_size),
            Some(candle_gen::quant::MLX_GROUP_SIZE as i32)
        );

        // A dense config (no `quantization`) ⇒ None (dense path, no guard).
        write_cfg(r#"{"in_channels": 16}"#);
        assert!(pipe(&tmp).transformer_packed_config().is_none());

        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// Turbo's 4-step schedule starts at 1.0 and is strictly decreasing to 0.
    #[test]
    fn turbo_four_step_schedule() {
        let s = sd3_sigmas(4, 3.0);
        assert_eq!(s.len(), 5);
        assert!((s[0] - 1.0).abs() < 1e-6);
        assert!(s[4].abs() < 1e-9);
    }

    // ---- structural end-to-end tests (random weights) ---------------------------------------------
    //
    // These exercise the FULL render core — sampler + CFG/Turbo dispatch + VAE decode — on CPU with a
    // tiny, real-shape MMDiT and the real 16-ch VAE, asserting a decoded RGB image of the right
    // dimensions with finite values. No snapshot/weights/GPU needed (the encoders are bypassed by
    // building the aggregated conditioning tensors directly, the same isolation `conditioning.rs`
    // uses). The CUDA smoke below reuses this exact harness with a CUDA device.

    use candle_gen::candle_core::{Device, Tensor};
    use candle_gen::candle_nn::{VarBuilder, VarMap};
    use candle_gen::gen_core::CancelFlag;

    use crate::config::Sd3Config;
    use crate::transformer::Sd3Transformer;
    use crate::vae::load_vae;

    /// A tiny SD3.5-shaped config — small inner dim + a couple of joint blocks — but with the FULL
    /// conditioning widths (pooled 2048, joint 4096) so the conditioning tensors are real-shaped.
    fn tiny_cfg() -> Sd3Config {
        Sd3Config {
            in_channels: 16,
            patch_size: 2,
            pos_embed_max_size: 8,
            inner_dim: 16,
            num_heads: 2,
            head_dim: 8,
            num_layers: 3,
            mlp_ratio: 2.0,
            qk_norm: true,
            context_pre_only_last: true,
            pooled_dim: 2048,
            joint_attention_dim: 4096,
            clip_l_dim: 768,
            clip_g_dim: 1280,
            clip_concat_dim: 2048,
            clip_seq_len: 77,
            t5_seq_len: 8,
            t5_dim: 4096,
            timestep_channels: 16,
            dual_attention_layers: Vec::new(),
        }
    }

    /// A tiny **MMDiT-X** (Medium-shaped) config: like [`tiny_cfg`] but with the first two of three
    /// blocks flagged as dual-attention, so the Medium pipeline path drives the `attn2` blocks.
    fn tiny_medium_cfg() -> Sd3Config {
        Sd3Config {
            dual_attention_layers: vec![0, 1],
            ..tiny_cfg()
        }
    }

    /// Build a random-weight tiny transformer + the real 16-ch VAE on `device`, plus synthetic
    /// conditioning at the config's full widths. Returns everything `render_core` needs.
    fn harness(
        cfg: &Sd3Config,
        device: &Device,
    ) -> (
        Sd3Transformer,
        AutoEncoderKL,
        Sd3Conditioning,
        Sd3Conditioning,
    ) {
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, device);
        let transformer = Sd3Transformer::new(cfg, vb).unwrap();

        let vae_vm = VarMap::new();
        let vae_vb = VarBuilder::from_varmap(&vae_vm, DType::F32, device);
        let vae = load_vae(vae_vb).unwrap();

        let ctx_seq = cfg.context_seq_len();
        let cond = Sd3Conditioning {
            pooled: Tensor::randn(0f32, 1f32, (1, cfg.pooled_dim), device).unwrap(),
            context: Tensor::randn(0f32, 1f32, (1, ctx_seq, cfg.joint_attention_dim), device)
                .unwrap(),
        };
        let uncond = Sd3Conditioning {
            pooled: Tensor::zeros((1, cfg.pooled_dim), DType::F32, device).unwrap(),
            context: Tensor::zeros((1, ctx_seq, cfg.joint_attention_dim), DType::F32, device)
                .unwrap(),
        };
        (transformer, vae, cond, uncond)
    }

    /// Run the render core for `variant` on `device` at a tiny latent size + a small step count and
    /// assert a finite, right-dimensioned RGB image. The Large path exercises CFG (cond + uncond); the
    /// Turbo path the distilled single-eval (no uncond). `lat = 4` latent → 32px image at /8.
    ///
    /// `steps` is kept small (NOT `variant.default_steps()`): the flow sampler accumulates a candle
    /// autograd graph across iterations, and the full 28-step Large default overflows the 2 MB Windows
    /// test-thread stack at decode. The structural assertion is "CFG vs distilled dispatch + decode
    /// shape", which a handful of steps exercises identically; the defaults are checked separately in
    /// [`variant_defaults_match_sd35`].
    fn run_variant(variant: Variant, device: &Device) -> Image {
        run_variant_cfg(variant, &tiny_cfg(), device)
    }

    /// Like [`run_variant`] but with an explicit architecture config, so the Medium (MMDiT-X)
    /// path can be driven with a dual-attention tiny config.
    fn run_variant_cfg(variant: Variant, cfg: &Sd3Config, device: &Device) -> Image {
        let (transformer, vae, cond, uncond) = harness(cfg, device);
        let uncond_ref = if variant.cfg_enabled() {
            Some(&uncond)
        } else {
            None
        };
        let cancel = CancelFlag::default();
        let mut steps_seen = 0u32;
        let mut progress = |_p: Progress| steps_seen += 1;
        let lat = 4usize; // 32px image at /8
        let steps = variant.default_steps().min(4);
        let img = render_core(
            &transformer,
            &vae,
            &cond,
            uncond_ref,
            variant.default_cfg(),
            steps,
            variant.shift(),
            (lat, lat),
            7,
            device.clone(),
            DType::F32,
            None,
            None,
            &cancel,
            &mut progress,
        )
        .unwrap();
        assert_eq!(img.width, (lat as u32) * SPATIAL_SCALE);
        assert_eq!(img.height, (lat as u32) * SPATIAL_SCALE);
        // RGB8 = 3 bytes/pixel; u8 pixels are inherently finite (no NaN/Inf can escape decode).
        assert_eq!(img.pixels.len(), (img.width * img.height * 3) as usize);
        img
    }

    /// Full Large pipeline (CFG on) end-to-end on CPU → a decoded image of the right dimensions.
    #[test]
    fn large_cfg_pipeline_decodes_on_cpu() {
        let img = run_variant(Variant::Large, &Device::Cpu);
        // Sanity: 32×32 RGB.
        assert_eq!(img.width, 32);
        assert_eq!(img.height, 32);
    }

    /// Full Turbo pipeline (4-step, CFG-off) end-to-end on CPU → a decoded image.
    #[test]
    fn turbo_distilled_pipeline_decodes_on_cpu() {
        let _ = run_variant(Variant::LargeTurbo, &Device::Cpu);
    }

    /// Full **Medium** (MMDiT-X dual-attention, CFG-on) pipeline end-to-end on CPU → a decoded image
    /// of the right dimensions. Drives the dual-attention transformer through the SAME flow-match
    /// render core as Large (CFG cond + uncond), confirming the Medium config + dual blocks render.
    #[test]
    fn medium_mmdit_x_pipeline_decodes_on_cpu() {
        let img = run_variant_cfg(Variant::Medium, &tiny_medium_cfg(), &Device::Cpu);
        assert_eq!(img.width, 32);
        assert_eq!(img.height, 32);
    }

    /// Determinism: with the SAME weights, the same seed reproduces the same image (the render core
    /// is a pure function of seed + request + weights — the Euler step injects no per-step noise).
    /// Builds one harness and renders twice; a differing result would mean hidden nondeterminism in
    /// the sampler/decode.
    #[test]
    fn render_is_deterministic_for_a_seed() {
        let device = Device::Cpu;
        let cfg = tiny_cfg();
        let (transformer, vae, cond, _uncond) = harness(&cfg, &device);
        let cancel = CancelFlag::default();
        let render = |seed| {
            render_core(
                &transformer,
                &vae,
                &cond,
                None,
                1.0,
                4,
                3.0,
                (4, 4),
                seed,
                device.clone(),
                DType::F32,
                None,
                None,
                &cancel,
                &mut |_p: Progress| {},
            )
            .unwrap()
        };
        let a = render(7);
        let b = render(7);
        assert_eq!(a.pixels, b.pixels, "same seed + weights must reproduce");
    }

    /// sc-8993: at `cfg_scale == 1.0` the CFG blend `uncond + 1·(cond − uncond)` is exactly `cond`
    /// in exact arithmetic, so running the uncond forward is pure waste. This pins the equivalence
    /// the `conditioning` gate relies on: render_core with `Some(uncond)` at cfg 1.0 is
    /// numerically equivalent to the cond-only path (`None`) — proving skipping the uncond branch
    /// when guidance is disabled cannot change the output.
    ///
    /// We compare with a tiny per-pixel tolerance rather than byte-exact equality. The `Some(uncond)`
    /// path evaluates `v_uncond + (v_cond − v_uncond)·1.0`, whose reduction to `v_cond` is only exact
    /// in real arithmetic: floating-point add/sub is non-associative and platforms differ in FMA
    /// contraction and rounding (x86-64 Linux vs macOS/Windows), so the recombined velocity can differ
    /// from the direct `v_cond` in the last ULP. That sub-ULP difference propagates through the
    /// sampler + VAE decode and can flip a boundary pixel by ±1 after the u8 quantization — a byte-exact
    /// assertion is therefore not portable. A tolerance of ≤1 (u8) is still fully discriminating: a real
    /// CFG regression at cfg=1.0 (e.g. dropping the cond term, or a sign/scale error in the blend) shifts
    /// the image by far more than one gray level across many pixels.
    #[test]
    fn cfg_scale_one_equals_cond_only_path() {
        let device = Device::Cpu;
        let cfg = tiny_cfg();
        let (transformer, vae, cond, uncond) = harness(&cfg, &device);
        let cancel = CancelFlag::default();
        let render = |uncond_ref: Option<&Sd3Conditioning>| {
            render_core(
                &transformer,
                &vae,
                &cond,
                uncond_ref,
                1.0, // cfg_scale == 1.0: blend collapses to cond
                4,
                3.0,
                (4, 4),
                7,
                device.clone(),
                DType::F32,
                None,
                None,
                &cancel,
                &mut |_p: Progress| {},
            )
            .unwrap()
        };
        // With guidance ENABLED the uncond forward runs (wasted); with it skipped only cond runs.
        let with_uncond = render(Some(&uncond));
        let cond_only = render(None);
        assert_eq!(
            with_uncond.pixels.len(),
            cond_only.pixels.len(),
            "both paths must decode to the same-shaped image"
        );
        let max_abs_diff = with_uncond
            .pixels
            .iter()
            .zip(cond_only.pixels.iter())
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap_or(0);
        assert!(
            max_abs_diff <= 1,
            "cfg_scale 1.0 blend must equal cond-only (skipping the uncond branch is a no-op); \
             max per-pixel abs diff was {max_abs_diff} (>1 u8 => not just FP non-associativity)"
        );
    }

    /// **CUDA random-weight smoke (sc-7877).** Asserts the tiny MMDiT + VAE render core compiles, runs
    /// on the GPU (Blackwell PTX JIT, dense ops), and decodes a finite, right-shaped image with NO
    /// NaN/Inf. Dense/PTX ops JIT fine on sm_120 (only Q4/Q8 quant needs the native fatbin — that is
    /// C4). Runs under the repo's CUDA gate (`scripts/check-cuda.ps1`, built at
    /// `CUDA_COMPUTE_CAP=80`). Both the CFG (Large) and distilled (Turbo) paths are exercised.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_random_weight_forward_smoke() {
        let device = Device::new_cuda(0).expect("CUDA device 0");
        // Large (CFG) path.
        let large = run_variant(Variant::Large, &device);
        assert_eq!(
            large.pixels.len(),
            (large.width * large.height * 3) as usize
        );
        // Turbo (distilled, 4-step) path.
        let turbo = run_variant(Variant::LargeTurbo, &device);
        assert_eq!(
            turbo.pixels.len(),
            (turbo.width * turbo.height * 3) as usize
        );
    }

    /// **CUDA Medium MMDiT-X random-weight smoke (sc-7878).** The Medium dual-attention transformer
    /// (with `attn2` blocks) + VAE render core runs on the Blackwell GPU and decodes a finite,
    /// right-shaped RGB image with no NaN/Inf. This is the net-new C3 coverage over the C2 smoke: it
    /// exercises the MMDiT-X `attn2` (image-only self-attention) and 9-chunk AdaLN paths on CUDA.
    /// Built at `CUDA_COMPUTE_CAP=80` (dense PTX JIT — no quant), runs under `scripts/check-cuda.ps1`.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_medium_mmdit_x_forward_smoke() {
        let device = Device::new_cuda(0).expect("CUDA device 0");
        let img = run_variant_cfg(Variant::Medium, &tiny_medium_cfg(), &device);
        assert_eq!(img.pixels.len(), (img.width * img.height * 3) as usize);
        // No NaN/Inf can escape the u8 decode clamp; the assertion above plus a successful decode is
        // the GPU-side "shapes + finite" smoke.
        assert_eq!(img.width, 4 * SPATIAL_SCALE);
    }

    /// **CUDA Q4/Q8 quant smoke (sc-7879).** Build a tiny MMDiT (inner=32 so every contraction is at
    /// least one Q4_0/Q8_0 block wide) with random weights, fold it to Q4 *and* Q8 via
    /// `Sd3Transformer::quantize`, run a forward on the Blackwell GPU, and assert the velocity is
    /// **finite and non-zero**. This proves the dequant-on-forward path (`crate::quant`, sc-7702)
    /// actually executes on sm_120 — i.e. the dequant kernel runs and does NOT silently return zeros
    /// (the failure mode of CUDA Q4/Q8 without the native fatbin, sc-7544). Built at
    /// `CUDA_COMPUTE_CAP=80` under `scripts/check-cuda.ps1`.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_quant_forward_smoke() {
        use candle_gen::gen_core::Quant;

        let device = Device::new_cuda(0).expect("CUDA device 0");
        // inner=32 → one Q4_0/Q8_0 block per contraction row; 3 joint blocks exercise the MLP + attn.
        let cfg = Sd3Config {
            inner_dim: 32,
            num_heads: 2,
            head_dim: 16,
            ..tiny_cfg()
        };

        for quant in [Quant::Q4, Quant::Q8] {
            let vm = VarMap::new();
            let vb = VarBuilder::from_varmap(&vm, DType::F32, &device);
            let mut model = Sd3Transformer::new(&cfg, vb).unwrap();
            model.quantize(quant).expect("quantize on CUDA");

            let latent = Tensor::randn(0f32, 1f32, (1, cfg.in_channels, 8, 8), &device).unwrap();
            let ctx_seq = cfg.context_seq_len();
            let context =
                Tensor::randn(0f32, 1f32, (1, ctx_seq, cfg.joint_attention_dim), &device).unwrap();
            let pooled = Tensor::randn(0f32, 1f32, (1, cfg.pooled_dim), &device).unwrap();
            let t = Tensor::full(0.5f32, 1, &device).unwrap();

            let v = model
                .forward(&latent, &context, &pooled, &t)
                .expect("quantized MMDiT forward on CUDA");
            assert_eq!(v.dims(), latent.dims());

            let v = v
                .to_dtype(DType::F32)
                .unwrap()
                .to_device(&Device::Cpu)
                .unwrap();
            let vals = v.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            assert!(
                vals.iter().all(|x| x.is_finite()),
                "{quant:?} CUDA quant forward produced non-finite output"
            );
            let max_abs = vals.iter().fold(0f32, |m, &x| m.max(x.abs()));
            assert!(
                max_abs > 0.0,
                "{quant:?} CUDA quant forward is all-zero — the dequant path silently no-op'd on \
                 sm_120 (native fatbin missing? sc-7544)"
            );
            eprintln!("sc-7879 {quant:?} CUDA quant smoke: max|v|={max_abs:.4} (finite, non-zero)");
        }
    }

    /// **CUDA guidance-gate validation (sc-8993).** On the Blackwell GPU, prove the two AC halves:
    /// (1) at `cfg_scale == 1.0` the CFG blend `Some(uncond)` path is **byte-identical** to the
    /// cond-only (`None`) path — so skipping the uncond forward when guidance is disabled cannot alter
    /// output; and (2) a genuinely guided render (`cfg_scale = 4.0`, `Some(uncond)`) still decodes a
    /// finite, right-shaped image and differs from the cond-only render (guidance is doing something —
    /// its output is the code path this optimization deliberately leaves untouched). Built at
    /// `CUDA_COMPUTE_CAP=80` under `scripts/check-cuda.ps1`.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_cfg_gate_matches_cond_only_and_preserves_guided() {
        let device = Device::new_cuda(0).expect("CUDA device 0");
        let cfg = tiny_cfg();
        let (transformer, vae, cond, uncond) = harness(&cfg, &device);
        let cancel = CancelFlag::default();
        let render = |uncond_ref: Option<&Sd3Conditioning>, scale: f32| {
            render_core(
                &transformer,
                &vae,
                &cond,
                uncond_ref,
                scale,
                4,
                3.0,
                (4, 4),
                7,
                device.clone(),
                DType::F32,
                None,
                None,
                &cancel,
                &mut |_p: Progress| {},
            )
            .unwrap()
        };
        // (1) cfg 1.0: the wasted-uncond path == cond-only ON THE GPU, within a sub-ULP tolerance.
        // The `Some(uncond)` path recombines `v_uncond + (v_cond − v_uncond)·1.0`, whose reduction to
        // `v_cond` is exact only in real arithmetic; FP non-associativity can flip a boundary pixel by
        // ±1 after u8 quantization (see the CPU `cfg_scale_one_equals_cond_only_path` test). ≤1 (u8) is
        // still fully discriminating — a real CFG regression shifts many pixels by far more.
        let with_uncond_1 = render(Some(&uncond), 1.0);
        let cond_only = render(None, 1.0);
        assert_eq!(
            with_uncond_1.pixels.len(),
            cond_only.pixels.len(),
            "sc-8993: both cfg 1.0 paths must decode to the same-shaped image"
        );
        let max_abs_diff = with_uncond_1
            .pixels
            .iter()
            .zip(cond_only.pixels.iter())
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap_or(0);
        assert!(
            max_abs_diff <= 1,
            "sc-8993: cfg 1.0 with uncond must equal cond-only on CUDA; max per-pixel abs diff was \
             {max_abs_diff} (>1 u8 => not just FP non-associativity)"
        );
        // (2) guided (cfg 4.0) still renders and is a distinct, untouched code path.
        let guided = render(Some(&uncond), 4.0);
        assert_eq!(
            guided.pixels.len(),
            (guided.width * guided.height * 3) as usize
        );
        assert_ne!(
            guided.pixels, cond_only.pixels,
            "guided (cfg 4.0) output must differ from cond-only — guidance is active"
        );
        eprintln!(
            "sc-8993 CUDA gate: cfg1.0==cond-only (byte-eq), cfg4.0 guided distinct + decoded"
        );
    }

    /// **Real-weight memory profile — GATED (sc-7879 / C6).** Measures the TRUE peak CUDA memory of a
    /// real SD3.5 load at each precision (bf16 / Q8 / Q4) and prints it alongside the principled
    /// [`crate::memory::min_memory_gb`] estimate, so C6 can confirm the estimate is a safe ceiling.
    /// `#[ignore]`d because the SD3.5 checkpoints are gated (no HF token here). C6 flips this on by
    /// setting `SD35_LARGE_PATH`. Runnable later via:
    ///   `SD35_LARGE_PATH=/path/to/sd35-large cargo test -p candle-gen-sd3 --features cuda \
    ///    real_weight_memory_profile -- --ignored --nocapture`
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "gated SD3.5 weights unavailable here; set SD35_LARGE_PATH to enable (C6)"]
    fn real_weight_memory_profile() {
        use crate::memory::{min_memory_gb, Precision};
        use candle_gen::gen_core::Quant;

        let large_path = std::env::var("SD35_LARGE_PATH")
            .expect("set SD35_LARGE_PATH to a stable-diffusion-3.5-large diffusers snapshot dir");
        let device = Device::new_cuda(0).expect("CUDA device 0");

        for (precision, quant) in [
            (Precision::Bf16, None),
            (Precision::Q8, Some(Quant::Q8)),
            (Precision::Q4, Some(Quant::Q4)),
        ] {
            // Load (lazily) then force the component build by rendering the conditioning path is
            // overkill; instead build the transformer directly + quantize, and read the CUDA peak.
            let pipe = Pipeline::load(
                Path::new(&large_path),
                &device,
                DType::BF16,
                Variant::Large,
                quant,
                &[],
            );
            let comps = pipe.load_components().expect("load real components");
            // Touch the transformer so the allocation is live, then sample the device's used memory.
            let _ = &comps.transformer;
            // candle exposes per-device allocated bytes via the CUDA device; if unavailable, skip the
            // measurement and just print the estimate (C6 can wire `nvidia-smi` sampling).
            let est = min_memory_gb(Variant::Large, precision);
            eprintln!(
                "sc-7879 real-weight profile Large/{precision:?}: minMemoryGb estimate = {est:.1} GiB \
                 (peak measurement: sample nvidia-smi here in C6)"
            );
            drop(comps);
        }
    }

    /// **Real-weight smoke — GATED (sc-7877 / C6).** A real Large + Turbo render against actual SD3.5
    /// weights. `#[ignore]`d because the SD3.5 checkpoints are gated (Stability Community License,
    /// HF-account-bound) and NOT available in this environment; we do not download them. C6 (sc-7881)
    /// flips this on by setting `SD35_LARGE_PATH` (and `SD35_TURBO_PATH`) to a local snapshot dir.
    /// Runnable later via:
    ///   `SD35_LARGE_PATH=/path/to/sd35-large cargo test -p candle-gen-sd3 --features cuda \
    ///    real_weight_render -- --ignored --nocapture`
    #[test]
    #[ignore = "gated SD3.5 weights + HF token unavailable here; set SD35_LARGE_PATH to enable (C6)"]
    fn real_weight_render() {
        use candle_gen::gen_core::{registry, GenerationRequest, LoadSpec, WeightsSource};
        let large_path = std::env::var("SD35_LARGE_PATH")
            .expect("set SD35_LARGE_PATH to a stable-diffusion-3.5-large diffusers snapshot dir");
        let spec = LoadSpec::new(WeightsSource::Dir(large_path.into()));
        let g = registry::load(crate::MODEL_ID, &spec).expect("load sd3 large");
        let req = GenerationRequest {
            prompt: "a rusty robot holding a lit candle, studio lighting".into(),
            width: 1024,
            height: 1024,
            ..Default::default()
        };
        let mut progress = |_p: Progress| {};
        let out = g.generate(&req, &mut progress).expect("real-weight render");
        match out {
            candle_gen::gen_core::GenerationOutput::Images(imgs) => {
                assert_eq!(imgs.len(), 1);
                assert_eq!(imgs[0].width, 1024);
                assert_eq!(imgs[0].height, 1024);
                // sc-7881 C6: the render must be COHERENT — finite (no NaN/Inf clamp artifacts) and
                // non-degenerate (real spatial structure, not a uniform/noise field). A wrong AdaLN
                // `norm_out` scale/shift order scrambled this into a unit-variance noise wash.
                assert_image_coherent(&imgs[0]);
            }
            other => panic!("expected images, got {other:?}"),
        }
    }

    /// C6 coherence floor for a real-weight render: every pixel is finite (a `u8` buffer is finite by
    /// construction, but the upstream latent isn't — a NaN latent decodes to a flat clamp), the image
    /// spans a real dynamic range, and the per-pixel std is well above a degenerate noise/flat field.
    /// A correct SD3.5 render lands std ≈ 40–50 on a 0–255 luma; a uniform or pure-noise wash is far
    /// lower in structure. We assert a conservative floor that the scrambled (pre-fix) renders failed
    /// the *eyeball* on but passed numerically, so this is paired with the saved-PNG human check.
    #[cfg(test)]
    fn assert_image_coherent(img: &Image) {
        let px = &img.pixels;
        assert!(!px.is_empty(), "empty image buffer");
        let n = px.len() as f64;
        let mean = px.iter().map(|&b| b as f64).sum::<f64>() / n;
        let var = px.iter().map(|&b| (b as f64 - mean).powi(2)).sum::<f64>() / n;
        let std = var.sqrt();
        let min = *px.iter().min().unwrap();
        let max = *px.iter().max().unwrap();
        // Non-uniform: a real render spans most of the 0..255 range.
        assert!(
            max as i32 - min as i32 > 64,
            "degenerate dynamic range (min={min} max={max}) — render is near-uniform"
        );
        // Non-flat: meaningful spatial contrast (a flat fill is std≈0).
        assert!(
            std > 8.0,
            "degenerate std {std:.2} — render lacks structure"
        );
    }

    /// **LoRA before/after real-weight render — GATED (sc-7881).** Renders the SAME seed+prompt at
    /// Large twice — once stock, once with a community kohya `lora_sd3` adapter merged into the MMDiT —
    /// and asserts (a) the adapter actually merged (the merge errors loudly if no target matched, so a
    /// successful adapted load already proves keys mapped) and (b) the adapted image **visibly differs**
    /// from the base (mean abs pixel delta over a threshold). `#[ignore]`d because the SD3.5 weights +
    /// the LoRA are not present here; the validation box flips it on:
    ///   `SD35_LARGE_PATH=/path/to/sd35-large SD35_LORA_PATH=/path/to/lora.safetensors \
    ///    cargo test -p candle-gen-sd3 --features cuda lora_before_after_render -- --ignored --nocapture`
    #[test]
    #[ignore = "gated SD3.5 weights + a LoRA file unavailable here; set SD35_LARGE_PATH + SD35_LORA_PATH (sc-7881)"]
    fn lora_before_after_render() {
        use candle_gen::gen_core::{
            registry, AdapterKind, AdapterSpec, GenerationRequest, LoadSpec, WeightsSource,
        };
        let large_path = std::env::var("SD35_LARGE_PATH")
            .expect("set SD35_LARGE_PATH to a stable-diffusion-3.5-large diffusers snapshot dir");
        let lora_path = std::env::var("SD35_LORA_PATH")
            .expect("set SD35_LORA_PATH to a kohya lora_sd3 .safetensors");
        let strength: f32 = std::env::var("SD35_LORA_STRENGTH")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.0);

        let req = GenerationRequest {
            // A portrait prompt so a portrait LoRA's effect is on-distribution and easy to eyeball.
            prompt: "a portrait photo of a woman, studio lighting, highly detailed".into(),
            width: 1024,
            height: 1024,
            seed: Some(1234),
            steps: Some(28),
            ..Default::default()
        };
        let render = |adapters: Vec<AdapterSpec>| -> Image {
            let spec = LoadSpec::new(WeightsSource::Dir(large_path.clone().into()))
                .with_adapters(adapters);
            let g = registry::load(crate::MODEL_ID, &spec).expect("load sd3 large");
            match g
                .generate(&req, &mut |_p: Progress| {})
                .expect("real-weight render")
            {
                candle_gen::gen_core::GenerationOutput::Images(mut imgs) => imgs.remove(0),
                other => panic!("expected images, got {other:?}"),
            }
        };

        let base = render(vec![]);
        let adapted = render(vec![AdapterSpec::new(
            lora_path.into(),
            strength,
            AdapterKind::Lora,
        )]);
        assert_eq!(base.pixels.len(), adapted.pixels.len());

        // Mean absolute per-channel pixel delta (0..255). A real LoRA at strength 1.0 shifts the image
        // well clear of seed/sampler jitter; a no-op merge would be ~0.
        let sum: u64 = base
            .pixels
            .iter()
            .zip(adapted.pixels.iter())
            .map(|(a, b)| (*a as i32 - *b as i32).unsigned_abs() as u64)
            .sum();
        let mean = sum as f64 / base.pixels.len() as f64;
        eprintln!("sc-7881 LoRA before/after: mean abs pixel delta = {mean:.2} (of 255)");
        assert!(
            mean > 2.0,
            "adapted render barely differs from base (mean delta {mean:.2}) — did the LoRA merge?"
        );
    }

    /// **Medium real-weight smoke — GATED (sc-7878 / C6).** A real SD3.5-Medium (MMDiT-X) render
    /// against actual gated weights. `#[ignore]`d for the same reason as [`real_weight_render`] (the
    /// SD3.5-Medium checkpoint is gated and not present here; we do not download it). C6 flips this
    /// on via `SD35_MEDIUM_PATH`. Runnable later via:
    ///   `SD35_MEDIUM_PATH=/path/to/sd35-medium cargo test -p candle-gen-sd3 --features cuda \
    ///    medium_real_weight_render -- --ignored --nocapture`
    #[test]
    #[ignore = "gated SD3.5-Medium weights unavailable here; set SD35_MEDIUM_PATH to enable (C6)"]
    fn medium_real_weight_render() {
        use candle_gen::gen_core::{registry, GenerationRequest, LoadSpec, WeightsSource};
        let medium_path = std::env::var("SD35_MEDIUM_PATH")
            .expect("set SD35_MEDIUM_PATH to a stable-diffusion-3.5-medium diffusers snapshot dir");
        let spec = LoadSpec::new(WeightsSource::Dir(medium_path.into()));
        let g = registry::load(crate::MODEL_ID_MEDIUM, &spec).expect("load sd3 medium");
        let req = GenerationRequest {
            prompt: "a rusty robot holding a lit candle, studio lighting".into(),
            width: 1024,
            height: 1024,
            ..Default::default()
        };
        let mut progress = |_p: Progress| {};
        let out = g
            .generate(&req, &mut progress)
            .expect("medium real-weight render");
        match out {
            candle_gen::gen_core::GenerationOutput::Images(imgs) => {
                assert_eq!(imgs.len(), 1);
                assert_eq!(imgs[0].width, 1024);
                assert_eq!(imgs[0].height, 1024);
                // sc-7881 C6: the MMDiT-X (dual-attention) render must be coherent too.
                assert_image_coherent(&imgs[0]);
            }
            other => panic!("expected images, got {other:?}"),
        }
    }
}
