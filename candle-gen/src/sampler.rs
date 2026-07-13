//! The candle backend impl of the unified `gen_core::sampling::LatentOps` (epic 7114 P2, sc-7119) —
//! the tensor primitives the unified curated samplers (Euler / Heun / DPM++ 2M·SDE / UniPC /
//! ancestral / LCM / DDIM, sc-7117) are written against, over `candle_core::Tensor`. The candle twin
//! of mlx-gen's `MlxLatentOps`.
//!
//! Carries the same byte-parity convention so a migrated engine's DEFAULT sampler stays bit-identical
//! (the N1 gate): `scale(x, 1.0)` and `axpy(1.0, x, b, y) = x + y·b` elide the multiply-by-one.
//! `randn_like` uses the same per-step subkey derivation as mlx-gen's `StepRng` (D6 determinism),
//! drawn with a seeded CPU `StdRng` + `StandardNormal` so the noise is launch-portable and matched to
//! the latent's dtype/shape/device (the sc-3673 / sc-5179 initial-noise lineage). Cross-backend bitwise
//! equality of the draw is NOT a goal (the RNG differs from mlx).

use candle_core::Tensor;
use gen_core::guidance::GuidanceOps;
use gen_core::sampling::{LatentOps, TimestepConvention};
use gen_core::{CancelFlag, Progress};
use rand::{rngs::StdRng, SeedableRng};

use crate::{CandleError, Result};

/// Lift a `candle_core::Result` into the backend-neutral `gen_core::Result` (the `LatentOps` trait is
/// declared in gen-core, so its methods return `gen_core::Result`). Routes through the existing
/// candle-gen bridge: `candle_core::Error -> CandleError -> gen_core::Error::Backend`.
#[inline]
fn ge<T>(r: candle_core::Result<T>) -> gen_core::Result<T> {
    r.map_err(|e| CandleError::from(e).into())
}

/// The candle backend impl of [`gen_core::sampling::LatentOps`]. See the module docs.
#[derive(Clone, Copy, Debug, Default)]
pub struct CandleLatentOps;

impl LatentOps for CandleLatentOps {
    type Latent = Tensor;

    fn scale(&self, x: &Tensor, scale: f32) -> gen_core::Result<Tensor> {
        if scale == 1.0 {
            return Ok(x.clone());
        }
        ge(x.affine(scale as f64, 0.0))
    }

    fn add(&self, a: &Tensor, b: &Tensor) -> gen_core::Result<Tensor> {
        ge(a.add(b))
    }

    fn sub(&self, a: &Tensor, b: &Tensor) -> gen_core::Result<Tensor> {
        ge(a.sub(b))
    }

    fn axpy(&self, a: f32, x: &Tensor, b: f32, y: &Tensor) -> gen_core::Result<Tensor> {
        // Byte-parity with mlx-gen's apply_step a_x==1 branch: emit `x + y·b` (multiply-by-one elided).
        let yb = ge(y.affine(b as f64, 0.0))?;
        if a == 1.0 {
            return ge(x.add(&yb));
        }
        let ax = ge(x.affine(a as f64, 0.0))?;
        ge(ax.add(&yb))
    }

    fn randn_like(&self, x: &Tensor, seed: u64, step: usize) -> gen_core::Result<Tensor> {
        // Same per-step subkey derivation as mlx-gen's StepRng (de-correlate steps; `+1` keeps step 0
        // off the raw init-noise seed). Seeded CPU StdRng draw -> launch-portable, then matched to the
        // latent's shape/device/dtype.
        let sub = seed.wrapping_add(0x9E37_79B9_7F4A_7C15_u64.wrapping_mul(step as u64 + 1));
        let mut rng = StdRng::seed_from_u64(sub);
        let data = crate::seeded_normal_vec(&mut rng, x.elem_count());
        let noise = ge(Tensor::from_vec(data, x.shape().clone(), x.device()))?;
        ge(noise.to_dtype(x.dtype()))
    }
}

/// Normalize gen-core's (possibly negative) reduction `axes` to candle's `usize` dims for `x`.
fn candle_dims(x: &Tensor, axes: &[i32]) -> Vec<usize> {
    let r = x.rank() as i32;
    axes.iter()
        .map(|&a| (if a < 0 { a + r } else { a }) as usize)
        .collect()
}

/// The guidance-axis op extension (epic 7434 P2, sc-7440) over `candle_core::Tensor` — the candle
/// twin of mlx-gen's `MlxLatentOps` impl, backing the backend-neutral [`gen_core::guidance`] library
/// (cfg / cfg_rescale / full APG). candle is EAGER, so no `eval` boundary applies.
///
/// The reductions return the **keepdim** tensor (reduced shape); candle has NO implicit broadcasting,
/// so `mul`/`div` use `broadcast_mul`/`broadcast_div` to let the library's mixed full×reduced combines
/// broadcast (equal shapes broadcast to themselves). Negative `axes` are normalized via
/// [`candle_dims`]; `shape` is unused (the `Tensor` carries its own).
impl GuidanceOps for CandleLatentOps {
    fn mul(&self, a: &Tensor, b: &Tensor) -> gen_core::Result<Tensor> {
        ge(a.broadcast_mul(b))
    }

    fn div(&self, a: &Tensor, b: &Tensor) -> gen_core::Result<Tensor> {
        ge(a.broadcast_div(b))
    }

    fn clamp_min(&self, x: &Tensor, s: f32) -> gen_core::Result<Tensor> {
        ge(x.clamp(s, f32::INFINITY))
    }

    fn clamp_max(&self, x: &Tensor, s: f32) -> gen_core::Result<Tensor> {
        ge(x.clamp(f32::NEG_INFINITY, s))
    }

    fn select_positive(&self, sel: &Tensor, a: &Tensor, b: &Tensor) -> gen_core::Result<Tensor> {
        let mask = ge(sel.gt(0f32))?;
        ge(mask.where_cond(a, b))
    }

    fn norm_over(&self, x: &Tensor, _shape: &[usize], axes: &[i32]) -> gen_core::Result<Tensor> {
        let dims = candle_dims(x, axes);
        let sq = ge(x.sqr())?;
        let summed = ge(sq.sum_keepdim(dims.as_slice()))?;
        ge(summed.sqrt())
    }

    fn dot_over(
        &self,
        a: &Tensor,
        b: &Tensor,
        _shape: &[usize],
        axes: &[i32],
    ) -> gen_core::Result<Tensor> {
        let dims = candle_dims(a, axes);
        let prod = ge(a.broadcast_mul(b))?;
        ge(prod.sum_keepdim(dims.as_slice()))
    }
}

// =================================================================================================
// Curated-sampler driver (epic 7114 P4, sc-7123): the per-engine adoption seam. The candle twin of
// mlx-gen's `run_curated_sampler` / `run_flow_sampler` / `resolve_schedule` (`mlx-gen/src/sampler.rs`),
// adapted for candle's EAGER evaluation (no `eval()` boundary — candle runs each op as it is built,
// so the cancel check + progress emit in the `denoise` callback already land per model eval).
// =================================================================================================

/// Drive a curated gen-core unified [`gen_core::sampling::Sampler`] over ANY prediction type — the
/// generalized core behind [`run_flow_sampler`], the per-engine adoption seam.
///
/// An engine supplies its [`gen_core::sampling::ModelSampling`] (`FlowModelSampling` for the
/// rectified-flow cohort, `DiscreteModelSampling` for the ε/DDPM cohort — SDXL/Kolors,
/// `EdmModelSampling` for the v-prediction outliers — SVD), its σ schedule, and its model forward (as
/// `predict`). The `ModelSampling` recombines the raw model output into a denoised `x0` estimate and
/// supplies the `c_in` input scaling, so the curated solver (Euler / Heun / DPM++ 2M·SDE / UniPC /
/// ancestral / LCM / DDIM) never sees the prediction type — it integrates `x0` in k-diffusion sigma
/// space regardless. This is what lets one solver library serve flow, EPS, and EDM engines alike.
///
/// - `sampler_name`: the canonical curated solver name. Unknown / `None` / a non-solver alias falls
///   back to plain Euler (N3 — never hard-fail a generation over a sampling knob).
/// - `ms`: the engine's prediction-type + noise-schedule contract.
/// - `sigmas`: the descending schedule, length `num_steps + 1`, trailing `0.0`.
/// - `predict(x_in, timestep)`: the engine's model forward returning the RAW (already CFG-combined)
///   output the prediction type expects — velocity for FLOW, ε for EPS, v for V. `x_in` is the
///   `c_in`-scaled latent ([`gen_core::sampling::ModelSampling::input_scale`]; identity for FLOW) and
///   `timestep` is the conditioning value the model embeds at this σ
///   ([`gen_core::sampling::ModelSampling::timestep`]). Any per-step CFG combine, velocity negation,
///   reference-latent concat, or adapter injection lives INSIDE this closure — the solver only sees
///   the combined output, so a multi-eval solver (heun / dpmpp) re-runs the whole closure each eval.
///
/// Cancellation and progress route through the `denoise` callback, the sole per-eval hook the
/// callback-form `Sampler` exposes; `cancel` bridges `gen_core::Error::Canceled` ⇄
/// `CandleError::Canceled`. Progress is reported as the count of schedule nodes already descended past,
/// robust to the multi-eval solvers (heun / dpmpp_sde call this twice per step; the count stays
/// monotone and ≤ total).
#[allow(clippy::too_many_arguments)]
pub fn run_curated_sampler(
    sampler_name: Option<&str>,
    ms: &dyn gen_core::sampling::ModelSampling,
    sigmas: &[f32],
    latents: Tensor,
    seed: u64,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    mut predict: impl FnMut(&Tensor, f32) -> Result<Tensor>,
) -> Result<Tensor> {
    use gen_core::sampling::{denoise as gc_denoise, sampler_by_name, Euler, Sampler};

    let ops = CandleLatentOps;
    let total = sigmas.len().saturating_sub(1).max(1) as u32;
    // N3: a curated name routes to its solver; an unknown name / non-solver alias falls back to Euler.
    let sampler: Box<dyn Sampler<CandleLatentOps>> = sampler_name
        .and_then(sampler_by_name::<CandleLatentOps>)
        .unwrap_or_else(|| Box::new(Euler));

    let mut denoise_fn = |x: &Tensor, sigma: f32| -> gen_core::Result<Tensor> {
        if cancel.is_cancelled() {
            return Err(gen_core::Error::Canceled);
        }
        // Progress as the count of schedule nodes already descended past — robust to the multi-eval
        // solvers (heun / dpmpp_sde call this twice per step; the count stays monotone and ≤ total).
        let current = (sigmas.iter().filter(|&&s| s > sigma).count() as u32 + 1).min(total);
        on_progress(Progress::Step { current, total });
        gc_denoise(&ops, ms, x, sigma, |xin, t| {
            predict(xin, t).map_err(Into::into)
        })
    };

    sampler
        .sample(&ops, ms, &mut denoise_fn, latents, sigmas, seed)
        .map_err(CandleError::from)
}

/// Drive a curated solver over a flow-match (rectified-flow) sigma schedule — the thin
/// [`run_curated_sampler`] wrapper for the FLOW cohort (Lens / FLUX / Qwen / Chroma / Z-Image / FLUX.2).
/// `conv` selects whether the model is fed the raw sigma ([`TimestepConvention::Sigma`]) or `1 − σ`
/// ([`TimestepConvention::OneMinusSigma`]); `predict` returns the RAW (already CFG-combined) velocity.
/// `euler` over FLOW reproduces the legacy flow-match Euler loop within the N1 tolerance.
///
/// The time-shift lives entirely in `sigmas` (resolved by [`resolve_flow_schedule`]), so
/// `FlowModelSampling::new(conv)` (mu = 0) is the correct integration contract here — its `timestep` /
/// `denoised_coeffs` are mu-independent.
#[allow(clippy::too_many_arguments)]
pub fn run_flow_sampler(
    sampler_name: Option<&str>,
    conv: TimestepConvention,
    sigmas: &[f32],
    latents: Tensor,
    seed: u64,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    predict: impl FnMut(&Tensor, f32) -> Result<Tensor>,
) -> Result<Tensor> {
    let ms = gen_core::sampling::FlowModelSampling::new(conv);
    run_curated_sampler(
        sampler_name,
        &ms,
        sigmas,
        latents,
        seed,
        cancel,
        on_progress,
        predict,
    )
}

/// Resolve a descending σ schedule honoring a per-generation curated `scheduler`, over ANY
/// [`gen_core::sampling::ModelSampling`] — the engine-side counterpart to the `sampler` knob: the
/// *scheduler* picks where the steps land, the *sampler* picks how each step advances. An unset /
/// unknown / native-aliased name returns `native` verbatim (the N1 byte-exact default); a curated name
/// builds the schedule via [`gen_core::sampling::schedule_sigmas`] over `ms`, which reads its σ-table /
/// timestep↔sigma map — so `normal` / `karras` / `sgm_uniform` / `simple` / `beta` / `ddim_uniform` /
/// `exponential` land correctly for the ε/DDPM (`DiscreteModelSampling`), EDM (`EdmModelSampling`), and
/// flow (`FlowModelSampling`) contracts alike. A curated scheduler may return a length other than
/// `steps + 1` (`ddim_uniform` / `beta` re-stride), which simply changes the effective step count — the
/// same behaviour ComfyUI / diffusers have.
pub fn resolve_schedule(
    scheduler_name: Option<&str>,
    ms: &dyn gen_core::sampling::ModelSampling,
    steps: usize,
    native: &[f32],
) -> Vec<f32> {
    use gen_core::sampling::{schedule_sigmas, Scheduler};
    match scheduler_name.and_then(Scheduler::from_name) {
        Some(sched) => schedule_sigmas(sched, ms, steps),
        None => native.to_vec(),
    }
}

/// Resolve the descending flow sigma schedule for an engine, honoring a per-generation curated
/// `scheduler` selection (epic 7114 scheduler axis) — the FLOW [`resolve_schedule`] wrapper.
///
/// - `scheduler_name`: the canonical curated scheduler name (`normal` / `simple` / `karras` /
///   `exponential` / `sgm_uniform` / `beta` / `ddim_uniform`). `None`, an unknown name, or a native
///   alias (e.g. `flow_match` / `flow_match_euler`) falls back to `native` (N3 — never hard-fail a
///   generation over a scheduling knob; the engine's native schedule is the byte-exact default).
/// - `mu`: the engine's time-shift (`compute_mu(image_seq_len, steps)` for the dynamic-shift models,
///   `shift.ln()` for a static-shift model, `0.0` for an unshifted one). The curated schedule is built
///   over a [`gen_core::sampling::FlowModelSampling::with_shift`] carrying this `mu` so `normal` /
///   `sgm_uniform` / … stay consistent with the engine's resolution-/config-dependent shift instead of
///   degrading to a linear σ ramp (which would starve a high-shift model of high-noise steps).
/// - `steps`: the denoise step count.
/// - `native`: the engine's exact native schedule (length `steps + 1`, trailing `0.0`), returned
///   verbatim on the default path so the per-engine N1 default-parity gate holds.
///
/// Schedule construction is **convention-independent** — the σ schedule is the same noise-fraction ramp
/// however the model consumes σ — so this always builds with [`TimestepConvention::Sigma`]; the engine's
/// own conditioning convention is applied separately by [`run_flow_sampler`].
pub fn resolve_flow_schedule(
    scheduler_name: Option<&str>,
    mu: f32,
    steps: usize,
    native: &[f32],
) -> Vec<f32> {
    let ms = gen_core::sampling::FlowModelSampling::with_shift(TimestepConvention::Sigma, mu);
    resolve_schedule(scheduler_name, &ms, steps, native)
}

/// The curated unified-framework **sampler** menu (epic 7114 decision 2) as capability strings — every
/// [`gen_core::sampling::Solver`] name, in menu order. A migrated engine advertises this in its
/// [`gen_core::generator::Capabilities`] `samplers` list (plus any legacy alias it still honors, e.g.
/// `flow_match`) so the per-generation `sampler` knob can select any curated integrator.
pub fn curated_sampler_names() -> Vec<&'static str> {
    gen_core::sampling::Solver::ALL
        .iter()
        .map(|s| s.name())
        .collect()
}

/// The curated unified-framework **scheduler** menu (epic 7114 decision 2) as capability strings —
/// every [`gen_core::sampling::Scheduler`] name, in menu order. Engines that expose the sigma-schedule
/// axis advertise this in their `schedulers` list; selecting one builds the schedule via
/// [`gen_core::sampling::schedule_sigmas`].
pub fn curated_scheduler_names() -> Vec<&'static str> {
    gen_core::sampling::Scheduler::ALL
        .iter()
        .map(|s| s.name())
        .collect()
}

/// The curated `menu` plus any legacy `aliases` an engine still honors (deduped, preserving order).
/// Each alias falls back to euler / the engine's native schedule through the N3 path in
/// [`run_curated_sampler`] / [`resolve_schedule`], so it stays valid against
/// [`gen_core::generator::Capabilities::validate_request`] without changing behaviour. A convenience for
/// building a migrated engine's `samplers` / `schedulers` capability lists (e.g.
/// `menu_with_aliases(curated_sampler_names(), &["flow_match_euler"])`).
pub fn menu_with_aliases(
    mut menu: Vec<&'static str>,
    aliases: &[&'static str],
) -> Vec<&'static str> {
    for &a in aliases {
        if !menu.contains(&a) {
            menu.push(a);
        }
    }
    menu
}

// =================================================================================================
// Joint two-stream (video+audio) curated sampling — LTX's cross-modal denoise (epic 7114 P4, sc-7125).
// The candle twin of mlx-gen's `AvLatents` / `MlxAvLatentOps` / `run_av_curated_sampler`.
// =================================================================================================

/// A joint video+audio latent pair — the [`gen_core::sampling::LatentOps::Latent`] for LTX's
/// cross-modal denoise, whose two streams are integrated **together** by one curated solver each step
/// (the AvDiT couples them via cross-modal attention). The single-`Tensor` [`CandleLatentOps`] cannot
/// represent this, so the two-stream variant exists.
#[derive(Clone)]
pub struct AvLatents {
    pub video: Tensor,
    pub audio: Tensor,
}

/// [`gen_core::sampling::LatentOps`] over [`AvLatents`] — applies each solver op to BOTH streams, so the
/// gen-core curated solvers (Euler / Heun / DPM++ 2M·SDE / UniPC / ancestral / DDIM) drive LTX's joint
/// video+audio denoise. Each per-stream op reuses [`CandleLatentOps`], so the byte-parity rules
/// (`scale(x, 1)` / `axpy(1, …)` elide the multiply) hold per stream.
#[derive(Clone, Copy, Debug, Default)]
pub struct CandleAvLatentOps;

impl LatentOps for CandleAvLatentOps {
    type Latent = AvLatents;

    fn scale(&self, x: &AvLatents, scale: f32) -> gen_core::Result<AvLatents> {
        Ok(AvLatents {
            video: CandleLatentOps.scale(&x.video, scale)?,
            audio: CandleLatentOps.scale(&x.audio, scale)?,
        })
    }

    fn add(&self, a: &AvLatents, b: &AvLatents) -> gen_core::Result<AvLatents> {
        Ok(AvLatents {
            video: CandleLatentOps.add(&a.video, &b.video)?,
            audio: CandleLatentOps.add(&a.audio, &b.audio)?,
        })
    }

    fn sub(&self, a: &AvLatents, b: &AvLatents) -> gen_core::Result<AvLatents> {
        Ok(AvLatents {
            video: CandleLatentOps.sub(&a.video, &b.video)?,
            audio: CandleLatentOps.sub(&a.audio, &b.audio)?,
        })
    }

    fn axpy(&self, a: f32, x: &AvLatents, b: f32, y: &AvLatents) -> gen_core::Result<AvLatents> {
        Ok(AvLatents {
            video: CandleLatentOps.axpy(a, &x.video, b, &y.video)?,
            audio: CandleLatentOps.axpy(a, &x.audio, b, &y.audio)?,
        })
    }

    fn randn_like(&self, x: &AvLatents, seed: u64, step: usize) -> gen_core::Result<AvLatents> {
        // Distinct subkeys per stream (the audio seed is XOR-shifted) so the two streams' stochastic
        // noise is decorrelated; each reuses the per-step `StepRng`-equivalent derivation.
        Ok(AvLatents {
            video: CandleLatentOps.randn_like(&x.video, seed, step)?,
            audio: CandleLatentOps.randn_like(&x.audio, seed ^ 0xA5A5_5A5A_C3C3_3C3C, step)?,
        })
    }
}

/// Drive a curated unified solver over LTX's **joint video+audio** flow-match schedule — the two-stream
/// sibling of [`run_flow_sampler`] (epic 7114 P4, sc-7125). The model is velocity-prediction over the
/// FLOW [`TimestepConvention::Sigma`] convention for BOTH streams; `predict(av_in, sigma)` returns the
/// raw `(video_velocity, audio_velocity)` as an [`AvLatents`]. Cancel + progress route through the
/// `denoise` callback exactly as [`run_curated_sampler`]. Used for LTX's distilled T2V+A path (the
/// per-token-σ I2V path with its post-step mask blend stays native).
#[allow(clippy::too_many_arguments)]
pub fn run_av_curated_sampler(
    sampler_name: Option<&str>,
    sigmas: &[f32],
    latents: AvLatents,
    seed: u64,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    mut predict: impl FnMut(&AvLatents, f32) -> Result<AvLatents>,
) -> Result<AvLatents> {
    use gen_core::sampling::{
        denoise as gc_denoise, sampler_by_name, Euler, FlowModelSampling, Sampler,
    };

    let ops = CandleAvLatentOps;
    let ms = FlowModelSampling::new(TimestepConvention::Sigma);
    let total = sigmas.len().saturating_sub(1).max(1) as u32;
    let sampler: Box<dyn Sampler<CandleAvLatentOps>> = sampler_name
        .and_then(sampler_by_name::<CandleAvLatentOps>)
        .unwrap_or_else(|| Box::new(Euler));

    let mut denoise_fn = |x: &AvLatents, sigma: f32| -> gen_core::Result<AvLatents> {
        if cancel.is_cancelled() {
            return Err(gen_core::Error::Canceled);
        }
        let current = (sigmas.iter().filter(|&&s| s > sigma).count() as u32 + 1).min(total);
        on_progress(Progress::Step { current, total });
        gc_denoise(&ops, &ms, x, sigma, |xin, t| {
            predict(xin, t).map_err(Into::into)
        })
    };

    sampler
        .sample(&ops, &ms, &mut denoise_fn, latents, sigmas, seed)
        .map_err(CandleError::from)
}

// =================================================================================================
// SCM / TrigFlow continuous-time-consistency sampler — the CFG-free, 1–4 step SANA-Sprint loop
// (epic 11776, story sc-11781; the candle twin of `mlx-gen-sana::scm` + `denoise_sprint`).
//
// ## Why this is NOT the unified flow-match sampler
//
// The epic-7114 unified framework ([`run_flow_sampler`] / the `Solver` menu) integrates in **sigma
// space** with an `x0 = k_x·x + k_out·output` recombination. SCM is a different parameterization: it
// works in **angle / atan space** (`t ∈ [0, π/2]`, the TrigFlow continuous-time-consistency map),
// predicts `x0 = cos(s)·x − sin(s)·output`, and **renoises** to the next angle
// `x_t = cos(t)·x0 + sin(t)·noise·σ_data`. The model output itself is recombined trigonometrically
// *before* the scheduler step. None of that maps onto the flow-match `ModelSampling` / `Solver`
// contract, so SCM is a dedicated scheduler + driver here (the consistency analog of the curated
// solver menu). What it DOES reuse is the run-loop contract — per-step cancel + monotone progress —
// and [`CandleLatentOps::randn_like`]'s per-step subkey derivation for the between-step renoise, so
// the stochastic noise matches the rest of candle-gen's D6 determinism convention.
//
// ## Guidance axis (epic 7434)
//
// SANA-Sprint is **CFG-free**: a single trunk forward per step with the guidance scale fed as an
// *embedded scalar* (the trunk's guidance embedder). There is no cond/uncond pair to combine, so the
// epic-7434 guidance library (`cfg` / `cfg_rescale` / `apg` / `cfg_pp` — all *combine* operators) does
// not apply; Sprint slots onto the guidance axis as the embedded / CFG-free case. The embedded
// guidance forward lives INSIDE the caller's `predict` closure, so this driver stays trunk-agnostic.
// =================================================================================================

/// diffusers `SCMScheduler` default `max_timesteps` — the Sprint default `π/2`.
pub const SCM_DEFAULT_MAX_TIMESTEP: f32 = std::f32::consts::FRAC_PI_2;
/// diffusers Sprint default `intermediate_timesteps` (only consulted when `num_steps == 2`).
pub const SCM_DEFAULT_INTERMEDIATE_TIMESTEP: f32 = 1.3;
/// diffusers `SCMScheduler` `sigma_data` — the std-dev of the multi-step renoise (`0.5` for Sprint).
pub const SCM_SIGMA_DATA: f32 = 0.5;

/// A SANA-Sprint SCM (TrigFlow consistency) denoising schedule: the descending angle timesteps
/// `t ∈ [max_timesteps … 0]`, length `num_steps + 1` (the trailing `0` is the clean endpoint).
///
/// Mirrors diffusers `SCMScheduler.set_timesteps`:
/// * `num_steps == 2` → `[max_timesteps, intermediate_timesteps, 0]`;
/// * otherwise → `linspace(max_timesteps, 0, num_steps + 1)`.
#[derive(Clone, Debug)]
pub struct ScmScheduler {
    /// Descending angle timesteps, length `num_steps + 1`.
    pub timesteps: Vec<f32>,
    /// `sigma_data` (the renoise std-dev; `0.5` for Sprint).
    pub sigma_data: f32,
}

impl ScmScheduler {
    /// Build the Sprint schedule for `num_steps` (the diffusers `set_timesteps` default path:
    /// `max_timesteps = π/2`, `intermediate_timesteps = 1.3`). 1–4 steps is the Sprint operating band.
    pub fn new(num_steps: usize) -> Self {
        Self::with_timesteps(
            num_steps,
            SCM_DEFAULT_MAX_TIMESTEP,
            SCM_DEFAULT_INTERMEDIATE_TIMESTEP,
        )
    }

    /// Build the schedule for `num_steps` with explicit `max_timesteps` / `intermediate_timesteps`
    /// (the latter only used for the `num_steps == 2` two-point schedule, matching diffusers).
    pub fn with_timesteps(
        num_steps: usize,
        max_timesteps: f32,
        intermediate_timesteps: f32,
    ) -> Self {
        let timesteps = if num_steps == 2 {
            vec![max_timesteps, intermediate_timesteps, 0.0]
        } else {
            let n = num_steps.max(1);
            (0..=n)
                .map(|i| max_timesteps * (1.0 - i as f32 / n as f32))
                .collect()
        };
        Self {
            timesteps,
            sigma_data: SCM_SIGMA_DATA,
        }
    }

    /// Wrap an explicit descending angle schedule (length `num_steps + 1`).
    pub fn from_timesteps(timesteps: Vec<f32>) -> Self {
        Self {
            timesteps,
            sigma_data: SCM_SIGMA_DATA,
        }
    }

    /// Number of denoise iterations (loop count) = `timesteps.len() - 1`.
    pub fn num_steps(&self) -> usize {
        self.timesteps.len().saturating_sub(1)
    }

    /// Whether this is a true single-step schedule (one renoise-free step). diffusers skips the
    /// between-step noise when `num_steps == 1`.
    pub fn is_single_step(&self) -> bool {
        self.num_steps() <= 1
    }

    /// The **SCM conditioning timestep** the trunk embeds at step `i` (diffusers
    /// `scm_timestep = sin(t)/(cos(t)+sin(t))` where `t` is the angle timestep). The trunk consumes
    /// this, NOT the raw angle.
    pub fn scm_timestep(&self, i: usize) -> f32 {
        let t = self.timesteps[i];
        let (s, c) = (t.sin(), t.cos());
        s / (c + s)
    }

    /// The model-input scale at step `i` (diffusers `sqrt(scm_t² + (1 - scm_t)²)`, applied to
    /// `latents / sigma_data`).
    pub fn input_scale(&self, i: usize) -> f32 {
        let st = self.scm_timestep(i);
        (st * st + (1.0 - st) * (1.0 - st)).sqrt()
    }
}

/// Drive the SCM / TrigFlow **CFG-free few-step** consistency loop — the SANA-Sprint denoise. One
/// trunk forward per step (no CFG uncond pass), a faithful port of diffusers `SanaSprintPipeline`'s
/// denoise + `SCMScheduler.step`:
///
/// 1. pre-scale the seed latent by `sigma_data` (diffusers `latents = latents * sigma_data`);
/// 2. per step `i` over the angle schedule `s = timesteps[i]`:
///    * `scm_t = sin(s)/(cos(s)+sin(s))`; model input = `(latents / σ_data) · sqrt(scm_t²+(1−scm_t)²)`;
///    * ONE `predict(lat_in, scm_t)` (the caller's embedded-guidance trunk forward — CFG-free);
///    * recombine the raw output trigonometrically, `· σ_data`;
///    * `x0 = cos(s)·latents − sin(s)·model_output`; renoise to the next angle
///      `latents = cos(t')·x0 + sin(t')·noise·σ_data` (skipped on the final / single-step transition);
/// 3. return `denoised / σ_data` (the diffusers decode input).
///
/// `predict(lat_in, scm_t)` returns the RAW trunk output; the trigflow recombination + renoise live
/// here, so the driver is trunk-agnostic (the embedded guidance scalar is captured by the closure).
/// Cancel + monotone progress mirror the [`run_curated_sampler`] run-loop contract.
pub fn run_scm_sampler(
    scheduler: &ScmScheduler,
    latents: Tensor,
    seed: u64,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    mut predict: impl FnMut(&Tensor, f32) -> Result<Tensor>,
) -> Result<Tensor> {
    let ops = CandleLatentOps;
    let sd = scheduler.sigma_data as f64;
    let n = scheduler.num_steps();
    let total = n.max(1) as u32;
    let single_step = scheduler.is_single_step();

    // diffusers: latents = latents * sigma_data (the SCM prior std-dev).
    let mut latents = latents.affine(sd, 0.0)?;
    let mut denoised = latents.clone();

    for i in 0..n {
        if cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }
        on_progress(Progress::Step {
            current: (i as u32 + 1).min(total),
            total,
        });

        let s = scheduler.timesteps[i];
        let t_next = scheduler.timesteps[i + 1];
        let scm_t = scheduler.scm_timestep(i);
        let in_scale = scheduler.input_scale(i) as f64;

        // model input = (latents / σ_data) · sqrt(scm_t² + (1-scm_t)²).
        let lat_in = latents.affine(in_scale / sd, 0.0)?;
        let raw = predict(&lat_in, scm_t)?;

        // diffusers trigflow recombination (uses the SCALED lat_in, not the raw latent):
        //   noise_pred = ((1-2·scm_t)·lat_in + (1-2·scm_t+2·scm_t²)·raw) / in_scale · σ_data.
        let a = 1.0 - 2.0 * scm_t;
        let b = 1.0 - 2.0 * scm_t + 2.0 * scm_t * scm_t;
        let combined = lat_in
            .affine(a as f64, 0.0)?
            .add(&raw.affine(b as f64, 0.0)?)?;
        let model_output = combined.affine(sd / in_scale, 0.0)?;

        // SCMScheduler.step: pred_x0 = cos(s)·latents − sin(s)·model_output.
        let pred_x0 = latents
            .affine(s.cos() as f64, 0.0)?
            .sub(&model_output.affine(s.sin() as f64, 0.0)?)?;
        denoised = pred_x0.clone();

        // Renoise to the next angle (skipped on the final / single-step transition, matching diffusers
        // `if len(self.timesteps) > 1`). The per-step subkey reuses the shared D6 derivation.
        latents = if single_step {
            pred_x0
        } else {
            let noise = ops
                .randn_like(&latents, seed, i)
                .map_err(CandleError::from)?
                .affine(sd, 0.0)?;
            pred_x0
                .affine(t_next.cos() as f64, 0.0)?
                .add(&noise.affine(t_next.sin() as f64, 0.0)?)?
        };
    }

    // diffusers: latents = denoised / sigma_data (the decode input).
    Ok(denoised.affine(1.0 / sd, 0.0)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use gen_core::sampling::{
        build_flow_sigmas, compute_mu, denoise, image_seq_len, sampler_by_name, Euler,
        FlowModelSampling, Sampler, TimestepConvention,
    };

    fn t(v: &[f32]) -> Tensor {
        Tensor::from_vec(v.to_vec(), v.len(), &Device::Cpu).unwrap()
    }
    fn vec1(x: &Tensor) -> Vec<f32> {
        x.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }
    fn max_abs(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| (x - y).abs())
            .fold(0.0_f32, f32::max)
    }
    /// A constant flow velocity (independent of x) for the exactness check.
    fn const_v() -> Tensor {
        t(&[0.37, -0.12, 0.8, -0.5])
    }

    #[test]
    fn candle_scale_add_sub() {
        let ops = CandleLatentOps;
        let a = t(&[1.0, 2.0, 3.0]);
        let b = t(&[0.5, -1.0, 4.0]);
        assert_eq!(vec1(&ops.scale(&a, 2.0).unwrap()), vec![2.0, 4.0, 6.0]);
        assert_eq!(vec1(&ops.scale(&a, 1.0).unwrap()), vec1(&a)); // 1.0 -> clone
        assert_eq!(vec1(&ops.add(&a, &b).unwrap()), vec![1.5, 1.0, 7.0]);
        assert_eq!(vec1(&ops.sub(&a, &b).unwrap()), vec![0.5, 3.0, -1.0]);
    }

    #[test]
    fn candle_axpy_a1_byte_parity() {
        let ops = CandleLatentOps;
        let x = t(&[0.3, -1.2, 2.5]);
        let y = t(&[0.7, 0.1, -0.4]);
        let got = ops.axpy(1.0, &x, 0.25, &y).unwrap();
        let want = x.add(&y.affine(0.25, 0.0).unwrap()).unwrap();
        assert_eq!(vec1(&got), vec1(&want), "axpy a=1 not byte-identical");
        // General a: 2·x + (−3)·y.
        let got2 = ops.axpy(2.0, &x, -3.0, &y).unwrap();
        let want2 = x
            .affine(2.0, 0.0)
            .unwrap()
            .add(&y.affine(-3.0, 0.0).unwrap())
            .unwrap();
        assert_eq!(vec1(&got2), vec1(&want2));
    }

    #[test]
    fn candle_randn_like_deterministic_shaped() {
        let ops = CandleLatentOps;
        let x = t(&[0.0, 0.0, 0.0, 0.0, 0.0]);
        let a = ops.randn_like(&x, 42, 0).unwrap();
        assert_eq!(a.dims(), x.dims());
        assert_eq!(a.dtype(), x.dtype());
        assert_eq!(vec1(&a), vec1(&ops.randn_like(&x, 42, 0).unwrap()));
        assert_ne!(vec1(&a), vec1(&ops.randn_like(&x, 42, 1).unwrap()));
        assert_ne!(vec1(&a), vec1(&ops.randn_like(&x, 43, 0).unwrap()));
    }

    #[test]
    fn candle_euler_integrates_constant_velocity_exactly() {
        // The rectified-flow ODE dx/dσ = v (constant) is linear, so the unified Euler over
        // CandleLatentOps must land EXACTLY on x_init − v·σ_0. Proves scale/sub/axpy compose right.
        let ops = CandleLatentOps;
        let ms = FlowModelSampling::new(TimestepConvention::Sigma);
        let sigmas = build_flow_sigmas(10, compute_mu(image_seq_len(1024, 1024), 10));
        let v = const_v();
        let x_init = t(&[0.3, -1.1, 2.0, 0.05]);
        let mut dn = |x: &Tensor, s: f32| denoise(&ops, &ms, x, s, |_xin, _t| Ok(v.clone()));
        let out = Euler
            .sample(&ops, &ms, &mut dn, x_init.clone(), &sigmas, 0)
            .unwrap();
        let want = x_init
            .sub(&v.affine(sigmas[0] as f64, 0.0).unwrap())
            .unwrap();
        assert!(max_abs(&vec1(&out), &vec1(&want)) < 1e-4);
    }

    #[test]
    fn candle_drives_every_curated_solver_to_finite_output() {
        // The P2 deliverable: every gen-core curated sampler runs end-to-end over candle Tensor.
        let ops = CandleLatentOps;
        let ms = FlowModelSampling::new(TimestepConvention::Sigma);
        let sigmas = build_flow_sigmas(6, compute_mu(image_seq_len(512, 512), 6));
        let x_init = t(&[0.2, -0.5, 1.0, 0.3]);
        for name in [
            "euler",
            "euler_ancestral",
            "heun",
            "dpmpp_2m",
            "dpmpp_sde",
            "uni_pc",
            "lcm",
            "ddim",
            "er_sde",
            "dpmpp_2m_sde",
        ] {
            let sampler = sampler_by_name::<CandleLatentOps>(name).expect("known solver");
            // Smooth velocity field v = 0.25·x + 0.1.
            let mut dn =
                |x: &Tensor, s: f32| denoise(&ops, &ms, x, s, |xin, _t| ge(xin.affine(0.25, 0.1)));
            let out = sampler
                .sample(&ops, &ms, &mut dn, x_init.clone(), &sigmas, 7)
                .unwrap();
            assert!(
                vec1(&out).iter().all(|v| v.is_finite()),
                "{name} produced non-finite output"
            );
        }
    }

    // --- Curated-sampler driver (epic 7114 P4, sc-7123): the per-engine adoption seam ---------------

    use super::{run_curated_sampler, run_flow_sampler};
    use candle_core::DType;
    use gen_core::sampling::{AlphaSchedule, DiscreteModelSampling, EdmModelSampling, Scheduler};
    use gen_core::{CancelFlag, Progress};

    /// N1 keystone: `run_flow_sampler` with the default `euler` over a FLOW schedule reproduces the
    /// legacy inline flow-match loop `img += v·(σ_{i+1} − σ_i)` within tolerance (the `to_d` round-trip
    /// is an f32-cancellation). This is the per-engine default-parity contract every flow engine relies
    /// on after routing its denoise through the driver.
    #[test]
    fn run_flow_sampler_euler_matches_inline_flow_loop() {
        let sigmas = build_flow_sigmas(8, compute_mu(image_seq_len(1024, 1024), 8));
        let x_init = t(&[0.3, -1.1, 2.0, 0.05, -0.4, 1.7]);
        // A reference flow velocity `v = 0.3·x + 0.1` (matches the gen-core byte-equiv stub).
        let velocity = |x: &Tensor| -> Result<Tensor> { Ok(x.affine(0.3, 0.1)?) };

        // Legacy inline loop: img += v·(σ_{i+1} − σ_i) over the descending schedule.
        let mut legacy = x_init.clone();
        for w in sigmas.windows(2) {
            let v = velocity(&legacy).unwrap();
            legacy = (&legacy + (v * (w[1] - w[0]) as f64).unwrap()).unwrap();
        }

        // Unified driver, default euler (sampler_name = None).
        let cancel = CancelFlag::new();
        let mut progress = |_p: Progress| {};
        let out = run_flow_sampler(
            None,
            TimestepConvention::Sigma,
            &sigmas,
            x_init,
            0,
            &cancel,
            &mut progress,
            |xin, _t| velocity(xin),
        )
        .unwrap();
        assert!(
            max_abs(&vec1(&out), &vec1(&legacy)) < 1e-4,
            "driver euler diverged from inline flow loop"
        );
    }

    /// Keystone: `run_curated_sampler` over a `DiscreteModelSampling` (ε prediction) with `euler`
    /// reproduces the legacy Kolors/diffusers Euler step `x + ε·(σ_{i+1} − σ_i)` EXACTLY for a constant
    /// ε field — the rectified integral is `x_init − ε·σ_0` (the `to_d` round-trip cancels). This is the
    /// equivalence the DDPM cohort's (sc-7124) curated path relies on.
    #[test]
    fn run_curated_sampler_eps_euler_matches_legacy_discrete_step() {
        let sched = AlphaSchedule::scaled_linear(1000, 0.00085, 0.012);
        let ms = DiscreteModelSampling::sdxl(&sched);
        let sigmas = vec![8.0_f32, 4.0, 2.0, 1.0, 0.5, 0.0];
        let x_init = t(&[0.3, -1.1, 2.0]);
        let eps = [0.7_f32, -0.2, 0.4];
        let cancel = CancelFlag::new();
        let mut progress = |_p: Progress| {};
        let out = run_curated_sampler(
            Some("euler"),
            &ms,
            &sigmas,
            x_init.clone(),
            0,
            &cancel,
            &mut progress,
            |_xin, _t| Ok(t(&eps)),
        )
        .unwrap();
        for ((g, &x0), &e) in vec1(&out).iter().zip(&vec1(&x_init)).zip(&eps) {
            let want = x0 - e * sigmas[0]; // x_init − ε·σ_0
            assert!((g - want).abs() < 2e-3, "eps euler: got {g} want {want}");
        }
    }

    /// Keystone: `run_curated_sampler` drives a v-prediction `EdmModelSampling` (SVD's contract, sc-7125)
    /// to finite output over every curated solver — proving the driver is prediction-type-agnostic.
    #[test]
    fn run_curated_sampler_v_prediction_edm_is_finite_every_solver() {
        let ms = EdmModelSampling::svd();
        let sigmas = vec![80.0_f32, 20.0, 5.0, 1.0, 0.2, 0.0];
        let x_init = t(&[0.2, -0.5, 1.0, 0.3]);
        let cancel = CancelFlag::new();
        let mut progress = |_p: Progress| {};
        for name in [
            "euler",
            "euler_ancestral",
            "heun",
            "dpmpp_2m",
            "dpmpp_sde",
            "uni_pc",
            "ddim",
        ] {
            let out = run_curated_sampler(
                Some(name),
                &ms,
                &sigmas,
                x_init.clone(),
                7,
                &cancel,
                &mut progress,
                // A mild v "model": v = 0.1·x_in (the input is c_in-scaled, keeping it bounded).
                |xin, _t| Ok(xin.affine(0.1, 0.0)?),
            )
            .unwrap();
            assert!(
                vec1(&out).iter().all(|v| v.is_finite()),
                "{name} (v-pred/EDM) produced non-finite output"
            );
        }
    }

    /// N3: an unknown / unset sampler name falls back to euler (never hard-fails), and a curated name
    /// routes to a genuinely different solver (so the knob has an effect).
    #[test]
    fn driver_unknown_sampler_falls_back_to_euler() {
        let sigmas = build_flow_sigmas(6, compute_mu(image_seq_len(512, 512), 6));
        let x0 = t(&[0.2, -0.5, 1.0, 0.3]);
        let run = |name: Option<&str>| {
            let cancel = CancelFlag::new();
            let mut p = |_p: Progress| {};
            run_flow_sampler(
                name,
                TimestepConvention::Sigma,
                &sigmas,
                x0.clone(),
                7,
                &cancel,
                &mut p,
                |xin, _t| Ok(xin.affine(0.25, 0.1)?),
            )
            .unwrap()
        };
        // Unknown name == default == explicit euler.
        assert_eq!(vec1(&run(Some("nope"))), vec1(&run(None)));
        assert_eq!(vec1(&run(Some("euler"))), vec1(&run(None)));
        // A real solver swap differs from euler.
        assert_ne!(vec1(&run(Some("heun"))), vec1(&run(None)));
    }

    /// The driver bridges cancellation: a flag tripped before the first eval surfaces as the typed
    /// `CandleError::Canceled` (not a stringified `Msg`).
    #[test]
    fn driver_cancellation_bridges_to_typed_canceled() {
        let sigmas = build_flow_sigmas(4, compute_mu(image_seq_len(512, 512), 4));
        let cancel = CancelFlag::new();
        cancel.cancel();
        let mut progress = |_p: Progress| {};
        let err = run_flow_sampler(
            None,
            TimestepConvention::Sigma,
            &sigmas,
            t(&[0.1, 0.2, 0.3, 0.4]),
            0,
            &cancel,
            &mut progress,
            |xin, _t| Ok(xin.clone()),
        )
        .unwrap_err();
        assert!(matches!(err, CandleError::Canceled));
    }

    /// `resolve_flow_schedule` returns the native schedule verbatim for the default/native-alias path
    /// (N1 byte-exact), and a curated scheduler name builds a distinct descending-to-0 schedule.
    #[test]
    fn resolve_flow_schedule_default_is_native_curated_differs() {
        use super::resolve_flow_schedule;
        let mu = compute_mu(image_seq_len(1024, 1024), 12);
        let native = build_flow_sigmas(12, mu);
        // Default + native alias => byte-identical native.
        assert_eq!(resolve_flow_schedule(None, mu, 12, &native), native);
        assert_eq!(
            resolve_flow_schedule(Some("flow_match"), mu, 12, &native),
            native
        );
        // A curated scheduler builds a real schedule (descending, trailing 0), distinct from native.
        let karras = resolve_flow_schedule(Some("karras"), mu, 12, &native);
        assert_eq!(*karras.last().unwrap(), 0.0);
        assert!(karras.windows(2).all(|w| w[0] >= w[1]));
        assert_ne!(karras, native);
        // Every curated scheduler resolves to a valid descending-to-0 schedule.
        for s in Scheduler::ALL {
            let sigs = resolve_flow_schedule(Some(s.name()), mu, 12, &native);
            assert!(
                sigs.len() >= 2 && *sigs.last().unwrap() == 0.0,
                "{}",
                s.name()
            );
        }
    }

    /// The curated menus expose exactly the gen-core vocabulary (decision 2).
    #[test]
    fn curated_menus_match_vocabulary() {
        use super::{curated_sampler_names, curated_scheduler_names};
        assert_eq!(
            curated_sampler_names(),
            vec![
                "euler",
                "euler_ancestral",
                "heun",
                "dpmpp_2m",
                "dpmpp_sde",
                "uni_pc",
                "lcm",
                "ddim",
                "er_sde",
                "dpmpp_2m_sde"
            ]
        );
        assert_eq!(
            curated_scheduler_names(),
            vec![
                "normal",
                "simple",
                "karras",
                "exponential",
                "sgm_uniform",
                "beta",
                "ddim_uniform",
                "beta57"
            ]
        );
    }

    /// Sanity: the driver runs at the engines' real dtype (bf16) without panicking and stays finite.
    #[test]
    fn driver_runs_in_bf16() {
        let sigmas = build_flow_sigmas(4, compute_mu(image_seq_len(512, 512), 4));
        let x0 = t(&[0.2, -0.5, 1.0, 0.3]).to_dtype(DType::BF16).unwrap();
        let cancel = CancelFlag::new();
        let mut progress = |_p: Progress| {};
        let out = run_flow_sampler(
            Some("dpmpp_2m"),
            TimestepConvention::Sigma,
            &sigmas,
            x0,
            7,
            &cancel,
            &mut progress,
            |xin, _t| Ok(xin.affine(0.25, 0.1)?),
        )
        .unwrap();
        assert_eq!(out.dtype(), DType::BF16);
        assert!(vec1(&out.to_dtype(DType::F32).unwrap())
            .iter()
            .all(|v| v.is_finite()));
    }

    // --- Two-stream AV driver (epic 7114 P4, sc-7125): LTX's joint video+audio denoise --------------

    use super::{run_av_curated_sampler, AvLatents};

    /// N1 keystone: `run_av_curated_sampler` default euler over a constant per-stream velocity lands
    /// exactly on `x_init − v·σ_0` per stream (the rectified-flow integral) — proving the two-stream
    /// `CandleAvLatentOps` + driver reproduce the legacy per-stream LTX `euler_step`.
    #[test]
    fn run_av_curated_sampler_euler_matches_legacy_per_stream() {
        let sigmas = vec![1.0_f32, 0.75, 0.5, 0.25, 0.0];
        let v_video = [0.7_f32, -0.2, 0.4];
        let v_audio = [0.1_f32, 0.5];
        let init = AvLatents {
            video: t(&[0.3, -1.1, 2.0]),
            audio: t(&[0.05, -0.4]),
        };
        let cancel = CancelFlag::new();
        let mut progress = |_p: Progress| {};
        let out = run_av_curated_sampler(
            None,
            &sigmas,
            init,
            0,
            &cancel,
            &mut progress,
            |_x, _sigma| {
                Ok(AvLatents {
                    video: t(&v_video),
                    audio: t(&v_audio),
                })
            },
        )
        .unwrap();
        for ((g, &x0), &v) in vec1(&out.video)
            .iter()
            .zip(&[0.3_f32, -1.1, 2.0])
            .zip(&v_video)
        {
            assert!((g - (x0 - v * sigmas[0])).abs() < 2e-3, "video: got {g}");
        }
        for ((g, &x0), &v) in vec1(&out.audio).iter().zip(&[0.05_f32, -0.4]).zip(&v_audio) {
            assert!((g - (x0 - v * sigmas[0])).abs() < 2e-3, "audio: got {g}");
        }
    }

    /// Every curated solver drives the two-stream AV latents to finite output (the stochastic ones too).
    #[test]
    fn run_av_curated_sampler_every_solver_is_finite() {
        let sigmas = build_flow_sigmas(6, compute_mu(image_seq_len(512, 512), 6));
        let cancel = CancelFlag::new();
        let mut progress = |_p: Progress| {};
        for name in [
            "euler",
            "euler_ancestral",
            "heun",
            "dpmpp_2m",
            "dpmpp_sde",
            "uni_pc",
            "ddim",
        ] {
            let init = AvLatents {
                video: t(&[0.2, -0.5, 1.0, 0.3]),
                audio: t(&[0.1, -0.2]),
            };
            let out = run_av_curated_sampler(
                Some(name),
                &sigmas,
                init,
                7,
                &cancel,
                &mut progress,
                |x, _s| {
                    Ok(AvLatents {
                        video: x.video.affine(0.2, 0.0)?,
                        audio: x.audio.affine(0.2, 0.0)?,
                    })
                },
            )
            .unwrap();
            assert!(
                vec1(&out.video).iter().all(|v| v.is_finite())
                    && vec1(&out.audio).iter().all(|v| v.is_finite()),
                "{name} (AV two-stream) produced non-finite output"
            );
        }
    }

    // =============================================================================================
    // SCM / TrigFlow consistency sampler (sc-11781) — the CFG-free few-step SANA-Sprint loop.
    // =============================================================================================

    fn approx(a: f32, b: f32, tol: f32, what: &str) {
        assert!((a - b).abs() < tol, "{what}: got {a} want {b}");
    }

    #[test]
    fn scm_two_step_schedule_is_diffusers_intermediate() {
        // diffusers set_timesteps(2) → [π/2, 1.3, 0].
        let s = ScmScheduler::new(2);
        assert_eq!(s.timesteps.len(), 3);
        approx(s.timesteps[0], SCM_DEFAULT_MAX_TIMESTEP, 1e-4, "max");
        approx(s.timesteps[1], 1.3, 1e-6, "intermediate");
        approx(s.timesteps[2], 0.0, 1e-6, "end");
        assert_eq!(s.num_steps(), 2);
        approx(s.sigma_data, 0.5, 1e-6, "sigma_data");
    }

    #[test]
    fn scm_four_step_schedule_is_linspace() {
        // diffusers set_timesteps(4) → linspace(π/2, 0, 5) = [π/2, 3π/8, π/4, π/8, 0].
        let s = ScmScheduler::new(4);
        assert_eq!(s.timesteps.len(), 5);
        for (i, &tt) in s.timesteps.iter().enumerate() {
            approx(
                tt,
                SCM_DEFAULT_MAX_TIMESTEP * (1.0 - i as f32 / 4.0),
                1e-5,
                "ts",
            );
        }
        assert!(s.timesteps.windows(2).all(|w| w[0] > w[1]));
    }

    #[test]
    fn scm_timestep_and_input_scale_match_diffusers_formula() {
        let s = ScmScheduler::new(2);
        // Endpoints: t = π/2 → sin=1, cos=0 → scm = 1; input scale = sqrt(1+0) = 1.
        approx(s.scm_timestep(0), 1.0, 1e-5, "scm at pi/2");
        approx(s.input_scale(0), 1.0, 1e-5, "scale at pi/2");
        // INDEPENDENT hard-coded reference for the mid-angle t=1.3 (Python: sin(1.3)=0.963558,
        // cos(1.3)=0.267499 → scm = 0.782708; input_scale = sqrt(0.782708²+0.217292²) = 0.812310).
        // Non-circular: proves the formula matches reality, not just that the impl matches the formula.
        approx(s.scm_timestep(1), 0.782708, 1e-5, "scm at t=1.3");
        approx(s.input_scale(1), 0.812310, 1e-5, "input_scale at t=1.3");
    }

    #[test]
    fn scm_single_step_flag() {
        assert!(ScmScheduler::new(1).is_single_step());
        assert!(!ScmScheduler::new(2).is_single_step());
        assert!(!ScmScheduler::new(4).is_single_step());
    }

    /// The driver runs EXACTLY one `predict` per step (CFG-free: no uncond pass), reports `num_steps`
    /// progress events, and produces a finite, non-degenerate latent.
    #[test]
    fn run_scm_sampler_one_forward_per_step_finite() {
        let latents = t(&[0.3, -1.1, 2.0, 0.05, -0.4, 1.7]);
        let scheduler = ScmScheduler::new(2);
        let cancel = CancelFlag::new();
        let mut forwards = 0usize;
        let mut steps_seen = 0usize;
        let mut progress = |p: Progress| {
            if matches!(p, Progress::Step { .. }) {
                steps_seen += 1;
            }
        };
        // A trunk stub: a small affine of the (scaled) input, scm_t-dependent so the two steps differ.
        let out = run_scm_sampler(
            &scheduler,
            latents,
            7,
            &cancel,
            &mut progress,
            |xin, scm_t| {
                forwards += 1;
                Ok(xin.affine(0.2 * scm_t as f64 + 0.1, 0.05)?)
            },
        )
        .unwrap();
        assert_eq!(forwards, 2, "CFG-free: exactly one trunk forward per step");
        assert_eq!(steps_seen, 2, "reports exactly num_steps progress events");
        let v = vec1(&out);
        assert!(v.iter().all(|x| x.is_finite()), "SCM latent non-finite");
        let (lo, hi) = (
            v.iter().cloned().fold(f32::INFINITY, f32::min),
            v.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        );
        assert!(hi - lo > 1e-6, "SCM latent is constant — graph degenerate");
    }

    /// Single-step (num_steps = 1) takes exactly one forward and skips the renoise draw.
    #[test]
    fn run_scm_sampler_single_step_one_forward() {
        let latents = t(&[0.5, -0.2, 0.9, 1.3]);
        let scheduler = ScmScheduler::new(1);
        assert!(scheduler.is_single_step());
        let cancel = CancelFlag::new();
        let mut forwards = 0usize;
        let mut progress = |_p: Progress| {};
        let out = run_scm_sampler(&scheduler, latents, 1, &cancel, &mut progress, |xin, _t| {
            forwards += 1;
            Ok(xin.affine(0.3, 0.0)?)
        })
        .unwrap();
        assert_eq!(forwards, 1, "single-step SCM runs exactly one forward");
        assert!(vec1(&out).iter().all(|x| x.is_finite()));
    }

    /// Determinism: the same seed + schedule + stub reproduces the latent bit-for-bit (the per-step
    /// renoise is seed-derived), and a different seed diverges (the multi-step noise actually mixes in).
    #[test]
    fn run_scm_sampler_is_seed_deterministic() {
        let run = |seed: u64| -> Vec<f32> {
            let scheduler = ScmScheduler::new(4);
            let cancel = CancelFlag::new();
            let mut progress = |_p: Progress| {};
            let out = run_scm_sampler(
                &scheduler,
                t(&[0.3, -1.1, 2.0, 0.05]),
                seed,
                &cancel,
                &mut progress,
                |xin, scm_t| Ok(xin.affine(0.15 * scm_t as f64 + 0.2, 0.0)?),
            )
            .unwrap();
            vec1(&out)
        };
        assert_eq!(run(7), run(7), "same seed reproduces");
        assert_ne!(run(7), run(8), "different seed diverges (renoise mixes in)");
    }
}

/// Real-backend (candle `Tensor`, CPU) parity for the gen-core guidance library over
/// [`CandleLatentOps`] (sc-7440) — the candle twin of mlx-gen's `guidance_ops_tests`. Proves the
/// trait impl reproduces the bespoke Lens/Bernini combine and `apg @ eta=1/nt=0/no-momentum == cfg`
/// on real `Tensor`, not just `CpuLatentOps`.
#[cfg(test)]
mod guidance_ops_tests {
    use super::*;
    use candle_core::Device;
    use gen_core::guidance::{cfg, cfg_rescale, normalized_guidance};

    /// A [B=1, seq=2, C=3] tensor (per-token channel-axis geometry, the Lens case).
    fn t(data: &[f32]) -> Tensor {
        Tensor::from_vec(data.to_vec(), (1usize, 2, 3), &Device::Cpu).unwrap()
    }

    fn max_abs(a: &Tensor, b: &Tensor) -> f32 {
        let av = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let bv = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        av.iter()
            .zip(&bv)
            .map(|(&x, &y)| (x - y).abs())
            .fold(0.0_f32, f32::max)
    }

    fn pair() -> (Tensor, Tensor) {
        let cond = t(&[3.0, 4.0, 0.0, 1.0, 2.0, 2.0]);
        let uncond = t(&[0.5, -1.0, 0.25, 0.1, 0.3, -0.2]);
        (cond, uncond)
    }

    #[test]
    fn cfg_matches_hand_combine() {
        let ops = CandleLatentOps;
        let (cond, uncond) = pair();
        let got = cfg(&ops, &cond, &uncond, 4.0).unwrap();
        let want = uncond
            .add(&cond.sub(&uncond).unwrap().affine(4.0, 0.0).unwrap())
            .unwrap();
        assert!(
            max_abs(&got, &want) < 1e-4,
            "cfg over candle != hand combine"
        );
    }

    /// gen-core `cfg_rescale` over candle == the bespoke Lens formula (per-token channel-axis rescale).
    #[test]
    fn cfg_rescale_matches_lens_formula() {
        let ops = CandleLatentOps;
        let (cond, uncond) = pair();
        let shape = [1usize, 2, 3];
        let got = cfg_rescale(&ops, &cond, &uncond, 2.0, &shape, &[-1]).unwrap();
        // Hand-rolled Lens reference: comb rescaled to ‖cond‖/‖comb‖ over the channel axis (2).
        let comb = uncond
            .add(&cond.sub(&uncond).unwrap().affine(2.0, 0.0).unwrap())
            .unwrap();
        let norm = |x: &Tensor| x.sqr().unwrap().sum_keepdim(2).unwrap().sqrt().unwrap();
        let cond_norm = norm(&cond);
        let comb_norm = norm(&comb);
        let denom = comb_norm.clamp(1e-12, f32::INFINITY).unwrap();
        let ratio = cond_norm.broadcast_div(&denom).unwrap();
        let want = comb.broadcast_mul(&ratio).unwrap();
        assert!(
            max_abs(&got, &want) < 1e-4,
            "cfg_rescale over candle != Lens formula"
        );
    }

    /// The Bernini invariant on real `Tensor`: APG at eta=1, nt=0, no momentum == plain CFG.
    #[test]
    fn apg_reduces_to_cfg_on_candle() {
        let ops = CandleLatentOps;
        let (cond, uncond) = pair();
        let shape = [1usize, 2, 3];
        let got =
            normalized_guidance(&ops, &cond, &uncond, 4.0, None, 1.0, 0.0, &shape, &[-1]).unwrap();
        let want = cfg(&ops, &cond, &uncond, 4.0).unwrap();
        assert!(
            max_abs(&got, &want) < 1e-4,
            "apg(eta=1,nt=0) over candle != cfg"
        );
    }

    /// eta=0 ⇒ the APG delta is orthogonal to the conditional base: (nd · cond) per token ≈ 0.
    #[test]
    fn apg_eta0_is_orthogonal_on_candle() {
        let ops = CandleLatentOps;
        let (cond, uncond) = pair();
        let shape = [1usize, 2, 3];
        let got =
            normalized_guidance(&ops, &cond, &uncond, 1.0, None, 0.0, 0.0, &shape, &[-1]).unwrap();
        let nd = got.sub(&uncond).unwrap();
        let dot = nd.mul(&cond).unwrap().sum_keepdim(2).unwrap();
        let zeros = dot.zeros_like().unwrap();
        assert!(
            max_abs(&dot, &zeros) < 1e-3,
            "eta=0 not orthogonal to cond on candle"
        );
    }
}
