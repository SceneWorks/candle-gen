//! Adaptive-Projected-Guidance (APG) for the Bernini renderer's `*_apg` modes — the candle sibling of
//! `mlx-gen-bernini/src/guidance.rs`.
//!
//! Like the mlx seam, the math lives once in the backend-neutral
//! [`gen_core::guidance`](candle_gen::gen_core::guidance); this module only injects candle's
//! [`CandleLatentOps`](candle_gen::sampler::CandleLatentOps) backend and Bernini's reduction geometry.
//!
//! **Reduction geometry.** The candle DiT latent is `[B, C, T, H, W]` (5-D, batch-first). The reference
//! reduces the x-space L2 norm + projection over `dim=[-1,-2,-4]` = channels + spatial, *excluding* the
//! temporal axis (per frame). On `[B, C, T, H, W]` that is axes `[C=1, H=3, W=4]` — [`APG_DIMS`]. (The
//! mlx layout is `[C, T, H, W]` with no batch, so it uses `[0, 2, 3]`; same channels+spatial set.)

use candle_gen::candle_core::Tensor;
use candle_gen::gen_core::guidance as core;
use candle_gen::sampler::CandleLatentOps;
use candle_gen::Result as CResult;

/// APG reduction dims on a `[B, C, T, H, W]` candle velocity (channels + spatial, per frame) — the
/// reference's `dim=[-1,-2,-4]` on its `[B, C, T, H, W]` layout (which excludes the temporal axis).
pub const APG_DIMS: &[i32] = &[1, 3, 4];

/// Persistent momentum accumulator for one APG stream (`running = diff + momentum·running`) — the
/// shared [`gen_core::guidance::MomentumBuffer`] over a candle [`Tensor`]. One per guidance term,
/// allocated **before** the denoise loop so the running average carries across steps.
pub type MomentumBuffer = core::MomentumBuffer<Tensor>;

/// Single-condition APG: `uncond + scale · normalize_diff(cond − uncond, base = cond)`
/// (`normalized_guidance`), over Bernini's per-frame [`APG_DIMS`] geometry. With `eta = 1`,
/// `norm_threshold = 0`, and no momentum this is exactly plain CFG `uncond + scale·(cond − uncond)`.
pub fn normalized_guidance(
    cond: &Tensor,
    uncond: &Tensor,
    scale: f32,
    buf: Option<&mut MomentumBuffer>,
    eta: f32,
    norm_threshold: f32,
) -> CResult<Tensor> {
    Ok(core::normalized_guidance(
        &CandleLatentOps,
        cond,
        uncond,
        scale,
        buf,
        eta,
        norm_threshold,
        &[], // shape unused on candle (the Tensor carries its own).
        APG_DIMS,
    )?)
}

/// Chained APG over an ordered list of predictions (`normalized_guidance_chain`). With
/// `bases = [uncond, preds[0], preds[1], …]`, accumulates
/// `result = uncond + Σ_i scales[i] · normalize_diff(preds[i] − bases[i], base = preds[i])`, each term
/// with its own momentum buffer and norm threshold. Used by `r2v_apg` over `[x_I, x_TI]`.
pub fn normalized_guidance_chain(
    uncond: &Tensor,
    preds: &[Tensor],
    scales: &[f32],
    bufs: &mut [MomentumBuffer],
    eta: f32,
    norm_thresholds: &[f32],
) -> CResult<Tensor> {
    Ok(core::normalized_guidance_chain(
        &CandleLatentOps,
        uncond,
        preds,
        scales,
        bufs,
        eta,
        norm_thresholds,
        &[],
        APG_DIMS,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::{Device, Tensor};

    fn deterministic(seed: f32) -> Tensor {
        // A varied, finite [B=1, C=4, T=2, H=2, W=2] tensor.
        let n = 4 * 2 * 2 * 2;
        let v: Vec<f32> = (0..n)
            .map(|i| ((i as f32 * 7.0 + seed * 13.0) % 11.0) - 5.0)
            .collect();
        Tensor::from_vec(v, (1, 4, 2, 2, 2), &Device::Cpu).unwrap()
    }

    fn max_abs(a: &Tensor, b: &Tensor) -> f32 {
        (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    /// eta=1, norm_threshold=0, no momentum ⇒ APG is EXACTLY plain CFG `uncond + scale·(cond−uncond)`.
    /// This is the load-bearing equivalence the renderer's `t2v` (plain) vs `t2v_apg` split relies on.
    #[test]
    fn apg_reduces_to_plain_cfg_at_eta1_no_clamp() {
        let cond = deterministic(1.0);
        let uncond = deterministic(2.0);
        let scale = 4.0f32;
        let got = normalized_guidance(&cond, &uncond, scale, None, 1.0, 0.0).unwrap();
        let want = (&uncond + ((&cond - &uncond).unwrap() * scale as f64).unwrap()).unwrap();
        assert!(
            max_abs(&got, &want) < 1e-4,
            "eta=1/nt=0 must equal plain CFG"
        );
        assert_eq!(got.dims(), cond.dims(), "shape preserved");
    }

    /// The momentum buffer accumulates across steps (`running = diff + momentum·running`); the first
    /// call returns the diff unchanged. Pins the per-step carry the denoise loop depends on.
    #[test]
    fn momentum_carries_across_steps() {
        // Two-step run with momentum vs a single-step run must differ (the buffer changed the result).
        let uncond = deterministic(2.0);
        let c1 = deterministic(3.0);
        let c2 = deterministic(4.0);
        let mut buf = MomentumBuffer::new(0.5);
        let _ = normalized_guidance(&c1, &uncond, 3.0, Some(&mut buf), 0.8, 0.0).unwrap();
        let with_hist = normalized_guidance(&c2, &uncond, 3.0, Some(&mut buf), 0.8, 0.0).unwrap();
        let fresh = normalized_guidance(&c2, &uncond, 3.0, None, 0.8, 0.0).unwrap();
        assert!(
            max_abs(&with_hist, &fresh) > 1e-4,
            "momentum history must change the second step"
        );
    }

    /// Chained APG (the `r2v_apg` shape) runs over two predictions with per-term buffers and preserves
    /// shape — pins the chain plumbing (the per-term forward math is covered by the gen_core tests).
    #[test]
    fn chain_runs_and_keeps_shape() {
        let uncond = deterministic(2.0);
        let preds = [deterministic(3.0), deterministic(4.0)];
        let mut bufs = [MomentumBuffer::new(0.3), MomentumBuffer::new(0.3)];
        let out =
            normalized_guidance_chain(&uncond, &preds, &[2.0, 5.0], &mut bufs, 0.6, &[1.0, 0.0])
                .unwrap();
        assert_eq!(out.dims(), uncond.dims());
    }
}
