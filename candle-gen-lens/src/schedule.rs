//! Lens sampling schedule + CFG (sc-5114). The schedule is the core flow-match Euler verbatim: the
//! Lens `compute_empirical_mu` is **byte-identical** to gen-core's [`compute_mu`] (same calibrated
//! constants + `>4300` branch), and the Lens `linspace(1, 1/n, n)` → dynamic-shift `set_timesteps`
//! is exactly [`build_flow_sigmas`], exposed as [`lens_sigmas`]. Only two pieces are Lens-specific:
//!
//! 1. **Timestep convention** — Lens feeds the transformer the *shifted sigma* directly (the
//!    reference `timestep / 1000`, where `scheduler.timesteps = shifted_sigma · 1000`), **not** the
//!    `1 − sigma` other DiT families use.
//! 2. **Norm-rescaled CFG** — [`cfg_rescale`]: `comb = uncond + g·(cond − uncond)`, then rescale
//!    `comb` to carry `cond`'s per-token (channel-axis) L2 norm.
//!
//! The denoise loop itself runs through the unified `candle_gen::run_flow_sampler` (epic 7114): its
//! default `euler` over the native [`lens_sigmas`] schedule is the N1 no-op that reproduces the legacy
//! per-crate flow-match Euler step within tolerance. **Lens is the standard-guidance family, NOT
//! true-CFG**: Turbo = 4-step / guidance 1.0 (≈ no CFG), base = 20-step / guidance 5.0.

use candle_gen::candle_core::{DType, Result, Tensor, D};
use candle_gen::gen_core::sampling::{build_flow_sigmas, compute_mu};

/// Per-variant sampling defaults (`num_steps`, `guidance_scale`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LensSamplingDefaults {
    pub num_steps: usize,
    pub guidance_scale: f32,
}

/// `microsoft/Lens-Turbo`: distilled **4 steps, guidance 1.0** (≈ no CFG).
pub const TURBO: LensSamplingDefaults = LensSamplingDefaults {
    num_steps: 4,
    guidance_scale: 1.0,
};
/// `microsoft/Lens` (base): **20 steps, guidance 5.0**.
pub const BASE: LensSamplingDefaults = LensSamplingDefaults {
    num_steps: 20,
    guidance_scale: 5.0,
};

/// The Lens empirical time-shift `mu`, fit from the latent token count `latent_h · latent_w` (==
/// the reference `compute_empirical_mu(seq_len, num_steps)`). Exposed so the unified scheduler axis
/// (sc-7123) can build a curated `normal`/`karras`/… schedule over the SAME shift the native schedule
/// uses (`resolve_flow_schedule`).
pub fn lens_mu(num_steps: usize, latent_h: usize, latent_w: usize) -> f32 {
    compute_mu(latent_h * latent_w, num_steps)
}

/// Build the Lens flow-match sigma schedule for `num_steps` at the given latent grid (length
/// `num_steps + 1`, descending, trailing `0.0`). The empirical time-shift `mu` is fit from the latent
/// token count `latent_h · latent_w` (== the reference `compute_empirical_mu(seq_len, num_steps)`).
pub fn lens_sigmas(num_steps: usize, latent_h: usize, latent_w: usize) -> Vec<f32> {
    build_flow_sigmas(num_steps, lens_mu(num_steps, latent_h, latent_w))
}

/// Norm-rescaled classifier-free guidance (the reference per-step CFG).
///
/// `cond`/`uncond`: `[B, seq, C]` predictions. Returns `comb · (‖cond‖ / ‖comb‖)` per token
/// (channel-axis L2 norm), with `comb = uncond + g·(cond − uncond)`; where `‖comb‖ = 0` the scale is
/// `1` (matching the reference `torch.where(comb_norm > 0, cond_norm / comb_norm.clamp_min(1e-12), 1)`).
pub fn cfg_rescale(cond: &Tensor, uncond: &Tensor, guidance: f32) -> Result<Tensor> {
    let comb = (uncond + ((cond - uncond)? * guidance as f64)?)?;
    let cond_norm = l2_over_channels(cond)?; // [B, seq, 1]
    let comb_norm = l2_over_channels(&comb)?;
    let ratio = cond_norm.broadcast_div(&comb_norm.maximum(1e-12)?)?;
    let scale = comb_norm
        .gt(0f64)?
        .where_cond(&ratio, &Tensor::ones_like(&comb_norm)?)?;
    comb.broadcast_mul(&scale)
}

/// Per-token L2 norm over the last (channel) axis, keepdim: `sqrt(sum(x², -1))`. Computed in f32 for
/// stability, cast back to `x`'s dtype so it composes with a bf16 denoise loop. No epsilon inside the
/// `sqrt` — the reference uses `torch.norm` and guards the divide separately (see [`cfg_rescale`]).
fn l2_over_channels(x: &Tensor) -> Result<Tensor> {
    let dt = x.dtype();
    let xf = x.to_dtype(DType::F32)?;
    xf.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?.to_dtype(dt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Device;

    #[test]
    fn variant_defaults() {
        assert_eq!(TURBO.num_steps, 4);
        assert_eq!(TURBO.guidance_scale, 1.0);
        assert_eq!(BASE.num_steps, 20);
        assert_eq!(BASE.guidance_scale, 5.0);
    }

    #[test]
    fn sigmas_descend_to_zero() {
        for n in [4usize, 20] {
            let s = lens_sigmas(n, 64, 64);
            assert_eq!(s.len(), n + 1, "n={n} length");
            assert_eq!(*s.last().unwrap(), 0.0, "n={n} trailing 0");
            assert!((s[0] - 1.0).abs() < 1e-4, "n={n} start ~1: {}", s[0]);
            assert!(s[..n].windows(2).all(|w| w[0] > w[1]), "n={n} descending");
        }
    }

    #[test]
    fn cfg_rescale_carries_cond_norm() {
        let dev = Device::Cpu;
        let cond = Tensor::from_vec(
            vec![3.0f32, 4.0, 0.0, 0.0, 1.0, 2.0, 2.0, 0.0],
            (1, 2, 4),
            &dev,
        )
        .unwrap();
        let uncond = Tensor::from_vec(
            vec![0.5f32, -0.5, 1.0, 0.0, -1.0, 0.0, 0.5, 0.5],
            (1, 2, 4),
            &dev,
        )
        .unwrap();
        let out = cfg_rescale(&cond, &uncond, 2.0).unwrap();
        // Per-token output L2 norm must equal cond's per-token L2 norm (token0: 5, token1: 3).
        let on = l2_over_channels(&out)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!((on[0] - 5.0).abs() < 1e-4, "token0 norm {}", on[0]);
        assert!((on[1] - 3.0).abs() < 1e-4, "token1 norm {}", on[1]);
    }

    /// sc-8993: at guidance == 1.0 the CFG combine `uncond + (cond − uncond)·1` is exactly `cond`, and
    /// the rescale ratio `cond_norm / comb_norm` is 1, so `cfg_rescale(cond, uncond, 1.0) == cond` for
    /// ANY uncond. This is the algebraic justification for skipping the uncond encode/forward when
    /// guidance is disabled — the denoise loop's cond-only path returns bit-identical output.
    #[test]
    fn cfg_rescale_at_guidance_one_is_cond_for_any_uncond() {
        let dev = Device::Cpu;
        let cond = Tensor::from_vec(
            vec![3.0f32, 4.0, 0.0, -2.0, 1.0, 2.0, 2.0, 7.0],
            (1, 2, 4),
            &dev,
        )
        .unwrap();
        // A deliberately unrelated uncond — the result must ignore it entirely at guidance 1.0.
        let uncond = Tensor::from_vec(
            vec![9.5f32, -0.5, 1.0, 3.0, -1.0, 8.0, 0.5, -4.5],
            (1, 2, 4),
            &dev,
        )
        .unwrap();
        let out = cfg_rescale(&cond, &uncond, 1.0).unwrap();
        let diff = (&out - &cond)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            diff < 1e-5,
            "cfg_rescale(cond, uncond, 1.0) must equal cond; max |diff| = {diff}"
        );
    }
}
