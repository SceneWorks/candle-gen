//! The distilled few-step pixel-diffusion sampler — faithful port of
//! `pid_distill_model.py::_student_sample_loop` (+ `_velocity_to_x0`). The released students run the
//! **SDE / velocity-prediction** schedule (`student_t_list=[0.999,0.866,0.634,0.342,0.0]`,
//! `fm_timescale=1000`, cfg 1 — no classifier-free guidance). PiD denoises directly in high-res
//! **pixel** space: `noise`/`x` are `[B, 3, H, W]` at the *output* resolution, conditioned on the LQ
//! latent + caption + degrade σ.
//!
//! Per step `(t_cur, t_next)`: `v = net(x, t_cur·timescale, …)`, `x0 = x − t_cur·v`; then for an SDE
//! interior step `x = (1−t_next)·x0 + t_next·ε` (fresh noise), and the final `t_next=0` step takes
//! `x = x0`. Output is clamped to `[-1, 1]`.
//!
//! The step math is RNG-free and deterministic — [`Sampler::run`] takes the initial noise and the
//! per-step ε injected, so it parity-tests bit-for-bit against the torch loop. [`Sampler::sample`]
//! draws them from a seeded CPU `StdRng` (`candle_gen::seed`, launch-portable) for production —
//! cross-backend RNG does not match torch/MLX, so this is a same-backend decode.

use candle_gen::candle_core::Tensor;
use candle_gen::gen_core::runtime::CancelFlag;
use candle_gen::seed::seeded_normal_vec;
use candle_gen::{CandleError, Result};
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::config::{SampleType, SamplerConfig};
use crate::lq::PidNet;

/// The distilled few-step sampler.
pub struct Sampler {
    t_list: Vec<f32>,
    timescale: f32,
    sde: bool,
}

impl Sampler {
    pub fn new(cfg: &SamplerConfig) -> Self {
        Self {
            t_list: cfg.student_t_list.clone(),
            timescale: cfg.fm_timescale,
            sde: cfg.sample_type == SampleType::Sde,
        }
    }

    /// Number of denoising steps (`len(t_list) − 1`).
    pub fn steps(&self) -> usize {
        self.t_list.len().saturating_sub(1)
    }

    /// Number of fresh-noise draws the SDE loop consumes (one per interior step with `t_next>0`).
    pub fn num_eps(&self) -> usize {
        if !self.sde {
            return 0;
        }
        (1..self.t_list.len())
            .filter(|&i| self.t_list[i] > 0.0)
            .count()
    }

    /// velocity-prediction `x0 = x − t·v`.
    fn velocity_to_x0(x: &Tensor, v: &Tensor, t: f32) -> Result<Tensor> {
        Ok((x - (v * t as f64)?)?)
    }

    /// Deterministic loop with the initial `noise` and the per-step `eps` injected (one ε per SDE
    /// interior step, in order). `caption`/`lq_latent`/`sigma` condition the net every step.
    ///
    /// `cancel` is checked at each of the ~4 step boundaries (F-006) — candle is eager, so a check per
    /// boundary is sufficient to interrupt this multi-second decode. Returns [`CandleError::Canceled`].
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        net: &PidNet,
        noise: &Tensor,
        eps: &[Tensor],
        caption: &Tensor,
        lq_latent: &Tensor,
        sigma: &Tensor,
        cancel: Option<&CancelFlag>,
    ) -> Result<Tensor> {
        let b = noise.dim(0)?;
        let device = noise.device();
        let mut x = noise.clone();
        let mut ei = 0usize;
        for i in 0..self.steps() {
            if cancel.is_some_and(CancelFlag::is_cancelled) {
                return Err(CandleError::Canceled);
            }
            let t_cur = self.t_list[i];
            let t_next = self.t_list[i + 1];
            let t_scaled = Tensor::from_vec(vec![t_cur * self.timescale; b], (b,), device)?;
            let v = net.forward(&x, &t_scaled, caption, lq_latent, sigma)?;
            if t_next > 0.0 {
                if self.sde {
                    let x0 = Self::velocity_to_x0(&x, &v, t_cur)?;
                    let e = &eps[ei];
                    ei += 1;
                    x = ((x0 * (1.0 - t_next) as f64)? + (e * t_next as f64)?)?;
                } else {
                    // ODE: x = x + (t_next − t_cur)·v (velocity prediction).
                    x = (&x + (&v * (t_next - t_cur) as f64)?)?;
                }
            } else {
                x = Self::velocity_to_x0(&x, &v, t_cur)?;
            }
        }
        Ok(x.clamp(-1.0f32, 1.0f32)?)
    }

    /// Production entry: draw the initial noise + per-step ε from a seeded CPU `StdRng` (launch-
    /// portable), then run the loop. Returns clamped pixels `[B, 3, H, W]`.
    #[allow(clippy::too_many_arguments)]
    pub fn sample(
        &self,
        net: &PidNet,
        caption: &Tensor,
        lq_latent: &Tensor,
        sigma: &Tensor,
        b: usize,
        h: usize,
        w: usize,
        seed: u64,
        cancel: Option<&CancelFlag>,
    ) -> Result<Tensor> {
        let device = lq_latent.device();
        let mut rng = StdRng::seed_from_u64(seed);
        let mut draw = || -> Result<Tensor> {
            let v = seeded_normal_vec(&mut rng, b * 3 * h * w);
            Ok(Tensor::from_vec(v, (b, 3, h, w), device)?)
        };
        let noise = draw()?;
        let mut eps = Vec::with_capacity(self.num_eps());
        for _ in 0..self.num_eps() {
            eps.push(draw()?);
        }
        self.run(net, &noise, &eps, caption, lq_latent, sigma, cancel)
    }
}
