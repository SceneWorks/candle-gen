//! Kolors scheduler — the candle port of `mlx-gen-kolors`'s `sampler.rs`: a faithful reproduction of
//! diffusers `EulerDiscreteScheduler` with the Kolors config (`scaled_linear` betas β₀=0.00085,
//! β₁=0.014, **num_train_timesteps=1100**, `timestep_spacing="leading"`, `steps_offset=1`,
//! `interpolation_type="linear"`, `final_sigmas_type="zero"`, epsilon prediction, `s_churn=0`).
//!
//! This is the non-ancestral Euler — **no per-step RNG** — so a denoise run is fully deterministic
//! given the initial latents (paired with the sc-3673 CPU-seeded noise, generation is a pure function
//! of `(seed, request)`). The schedule math is pure host f64/f32 (GPU-free, unit-tested below); the
//! pipeline applies it with candle tensor ops:
//!  - **leading** timesteps: `(arange(0,N)·(1100//N)).round()[::-1] + steps_offset`;
//!  - leading ⇒ `init_noise_sigma = (max_sigma² + 1)^0.5`;
//!  - `scale_model_input`: divide latents by `√(σ²+1)` before the UNet;
//!  - Euler step (ε-pred, γ=0): `x_next = x + ε·(σ_next − σ)`.

/// Kolors' `num_train_timesteps` — the length of the `scaled_linear` schedule the sampler interpolates
/// over. `num_steps` must lie in `1..=NUM_TRAIN_TIMESTEPS`.
pub const NUM_TRAIN_TIMESTEPS: usize = 1100;

const BETA_START: f64 = 0.00085;
const BETA_END: f64 = 0.014;
const STEPS_OFFSET: i64 = 1;

/// Kolors' EulerDiscrete (leading) sampler over the 1100-step `scaled_linear` schedule.
#[derive(Clone, Debug)]
pub struct KolorsEulerSampler {
    /// Interpolated sigmas at the (effective) timesteps, length `num_steps + 1` (trailing `0.0`).
    sigmas: Vec<f32>,
    /// The (float) leading timesteps fed to the UNet, length `num_steps`.
    timesteps: Vec<f32>,
    init_noise_sigma: f32,
}

impl KolorsEulerSampler {
    /// Build the Kolors schedule for `num_steps` denoise steps. Errors on `num_steps == 0` (divide by
    /// zero) or `> NUM_TRAIN_TIMESTEPS` (every leading timestep would collapse to a single value).
    pub fn new(num_steps: usize) -> Result<Self, String> {
        if num_steps == 0 {
            return Err("kolors sampler: num_steps must be >= 1".into());
        }
        if num_steps > NUM_TRAIN_TIMESTEPS {
            return Err(format!(
                "kolors sampler: num_steps must be <= {NUM_TRAIN_TIMESTEPS} (got {num_steps})"
            ));
        }

        // scaled_linear betas → alphas → alphas_cumprod (f64 throughout, matching diffusers).
        let n_train = NUM_TRAIN_TIMESTEPS;
        let (b0, b1) = (BETA_START.sqrt(), BETA_END.sqrt());
        let mut acp = 1.0f64;
        let mut full: Vec<f64> = Vec::with_capacity(n_train);
        for i in 0..n_train {
            let frac = if n_train == 1 {
                0.0
            } else {
                i as f64 / (n_train - 1) as f64
            };
            let beta = (b0 + (b1 - b0) * frac).powi(2);
            acp *= 1.0 - beta;
            // Per-train-step Karras sigma √((1-ᾱ)/ᾱ).
            full.push(((1.0 - acp) / acp).sqrt());
        }

        // leading: timesteps = (arange(0,N)·step_ratio).round()[::-1] + steps_offset.
        let step_ratio = (n_train / num_steps) as i64;
        let timesteps: Vec<f32> = (0..num_steps)
            .rev()
            .map(|j| ((j as i64 * step_ratio) + STEPS_OFFSET) as f32)
            .collect();

        // np.interp(timesteps, arange(0, N), full), then append 0 (final_sigmas_type="zero").
        let interp = |t: f32| -> f32 {
            let tt = (t as f64).clamp(0.0, (n_train - 1) as f64);
            let lo = tt.floor() as usize;
            let hi = (lo + 1).min(n_train - 1);
            let frac = tt - lo as f64;
            (full[lo] * (1.0 - frac) + full[hi] * frac) as f32
        };
        let mut sigmas: Vec<f32> = timesteps.iter().map(|&t| interp(t)).collect();
        let max_sigma = sigmas.iter().copied().fold(0.0_f32, f32::max);
        sigmas.push(0.0);

        Ok(Self {
            sigmas,
            timesteps,
            // leading spacing ⇒ init_noise_sigma = (max_sigma² + 1)^0.5.
            init_noise_sigma: (max_sigma * max_sigma + 1.0).sqrt(),
        })
    }

    /// The number of denoise steps.
    pub fn num_steps(&self) -> usize {
        self.timesteps.len()
    }

    /// The leading float timestep fed to the UNet time embedding at step `i`.
    pub fn timestep(&self, i: usize) -> f32 {
        self.timesteps[i]
    }

    /// `init_noise_sigma` — the initial latents are `noise · init_noise_sigma`.
    pub fn init_noise_sigma(&self) -> f32 {
        self.init_noise_sigma
    }

    /// The `scale_model_input` divisor at step `i`: `√(σ_i² + 1)` (latents are divided by this before
    /// the UNet, putting them in the model's expected scale).
    pub fn scale_in(&self, i: usize) -> f32 {
        let s = self.sigmas[i] as f64;
        (s * s + 1.0).sqrt() as f32
    }

    /// The Euler step delta at step `i`: `σ_next − σ_cur` (the latent update is `x += ε · dt`).
    pub fn step_dt(&self, i: usize) -> f32 {
        self.sigmas[i + 1] - self.sigmas[i]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_bad_step_counts() {
        assert!(KolorsEulerSampler::new(0).is_err());
        assert!(KolorsEulerSampler::new(NUM_TRAIN_TIMESTEPS + 1).is_err());
        assert!(KolorsEulerSampler::new(1).is_ok());
        assert!(KolorsEulerSampler::new(NUM_TRAIN_TIMESTEPS).is_ok());
    }

    #[test]
    fn leading_timesteps_descend_with_offset() {
        // N=50, step_ratio = 1100/50 = 22, leading timesteps = (j·22)[::-1] + 1.
        let s = KolorsEulerSampler::new(50).unwrap();
        assert_eq!(s.num_steps(), 50);
        // First (j=49): 49·22 + 1 = 1079; last (j=0): 0·22 + 1 = 1.
        assert!((s.timestep(0) - 1079.0).abs() < 1e-4, "{}", s.timestep(0));
        assert!((s.timestep(49) - 1.0).abs() < 1e-4, "{}", s.timestep(49));
        for i in 0..49 {
            assert!(s.timestep(i) > s.timestep(i + 1), "timesteps must descend");
        }
    }

    #[test]
    fn sigmas_descend_to_zero_and_scale_in_starts_above_one() {
        let s = KolorsEulerSampler::new(30).unwrap();
        // sigmas has length num_steps + 1, trailing 0.
        assert_eq!(s.sigmas.len(), 31);
        assert!(s.sigmas[30].abs() < 1e-9, "final sigma is 0");
        for w in s.sigmas.windows(2) {
            assert!(w[0] >= w[1], "sigmas must be non-increasing");
        }
        // init_noise_sigma = sqrt(max_sigma^2 + 1) > max_sigma > 1 (the Karras sigmas exceed 1 near t=1100).
        assert!(s.init_noise_sigma() > 1.0);
        // scale_in at step 0 = sqrt(sigma_0^2 + 1) >= 1, and is 1 at the trailing zero sigma.
        assert!(s.scale_in(0) > 1.0);
    }

    #[test]
    fn euler_step_dt_is_negative() {
        // The schedule descends, so every Euler delta (σ_next − σ_cur) is ≤ 0.
        let s = KolorsEulerSampler::new(10).unwrap();
        for i in 0..s.num_steps() {
            assert!(s.step_dt(i) <= 0.0, "dt at {i} should be <= 0");
        }
    }
}
