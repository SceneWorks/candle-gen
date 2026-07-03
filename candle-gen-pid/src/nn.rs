//! Crate-internal candle helpers shared across the PiD backbone / LQ adapter / Gemma-2 encoder:
//! bias-detecting Linear load and an f32-computed RMSNorm. These stand in for the `mlx_gen::nn` /
//! `mlx_gen::adapters` helpers the MLX port leans on — candle-gen's core crate has no such module, so
//! the provider crates (and this one) build directly on `candle_nn`.
//!
//! The whole PiD net runs **f32** (the MLX backbone's stated parity target and the dense-GEMM-safe
//! path): it avoids the mixed-dtype promotions the MLX code relies on and the f16-outlier NaN risk,
//! and image-decode memory is trivial vs a video VAE. So `rms` here is a plain f32 normalize; it still
//! upcasts defensively so a caller that hands a half tensor stays correct.

use candle_gen::candle_core::{DType, Tensor};
use candle_gen::candle_nn::ops::rms_norm as candle_rms_norm;
use candle_gen::candle_nn::Linear;
use candle_gen::{Result, Weights};

/// Load an `[out, in]` Linear (+ optional `{prefix}.bias`) as a [`candle_nn::Linear`]. The PiD/Gemma
/// checkpoints store torch `[out, in]` weights, which candle's `Linear` consumes directly (it applies
/// `x · Wᵀ`) — no transpose, unlike the MLX port's NHWC conv weights.
pub fn linear(w: &Weights, prefix: &str) -> Result<Linear> {
    let weight = w.require(&format!("{prefix}.weight"))?;
    let bias_key = format!("{prefix}.bias");
    let bias = if w.contains(&bias_key) {
        Some(w.require(&bias_key)?)
    } else {
        None
    };
    Ok(Linear::new(weight, bias))
}

/// `x · rsqrt(mean(x²)+eps) · weight` over the last axis (all in f32). Mirrors the MLX `rms`
/// (which upcasts to f32 for the reduction — load-bearing over the stack's ~60 norms). `candle_nn`'s
/// `rms_norm` applies `weight` directly (no `+1`), matching PiD's `RMSNorm`; Gemma's `(1+weight)` is
/// pre-folded at load by the caller.
pub fn rms(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let dt = x.dtype();
    let xf = x.to_dtype(DType::F32)?.contiguous()?;
    let wf = weight.to_dtype(DType::F32)?;
    Ok(candle_rms_norm(&xf, &wf, eps)?.to_dtype(dt)?)
}
