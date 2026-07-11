//! Small candle neural-net helpers shared by the Bernini planner modules (the Qwen2.5-VL backbone +
//! the connector) — the candle twins of the `mlx_gen::nn` ops the MLX port uses. Kept composable (no
//! fused-kernel dispatch) so they match the MLX numerics on CPU and CUDA alike (mirrors
//! `candle-gen-anima/src/nn.rs`).

use candle_gen::candle_core::{Tensor, D};
use candle_gen::candle_nn::{Linear, VarBuilder};
use candle_gen::Result;

/// Bias-less dense `Linear` loaded from `{name}.weight` (shape read from disk via `get_unchecked`, so a
/// packed/dense weight of any `[out, in]` loads unchanged) — the candle twin of the MLX `lin` helper.
pub fn lin(vb: &VarBuilder, name: &str) -> Result<Linear> {
    Ok(Linear::new(
        vb.get_unchecked(&format!("{name}.weight"))?,
        None,
    ))
}

/// `Linear` **with** bias loaded from `{name}.weight` + `{name}.bias`.
pub fn lin_bias(vb: &VarBuilder, name: &str) -> Result<Linear> {
    Ok(Linear::new(
        vb.get_unchecked(&format!("{name}.weight"))?,
        Some(vb.get_unchecked(&format!("{name}.bias"))?),
    ))
}

/// Plain RMSNorm `w · x / sqrt(mean(x²) + eps)` over the last dim (matches `mlx_rs::fast::rms_norm`).
/// `w` broadcasts over the leading dims. Computed in the input dtype (the planner parity goldens are
/// f32; the real loader upcasts internally if it runs bf16).
pub fn rms_norm(x: &Tensor, w: &Tensor, eps: f64) -> Result<Tensor> {
    let mean = x.sqr()?.mean_keepdim(D::Minus1)?; // [.., 1]
    let denom = (mean + eps)?.sqrt()?;
    Ok(x.broadcast_div(&denom)?.broadcast_mul(w)?)
}

/// Torch `nn.LayerNorm` over the last dim: `(x − mean)/sqrt(var + eps)`, with **biased** variance
/// (divide by N, no Bessel) matching torch, then optional affine `·w + b`. The candle twin of
/// `mlx_rs::fast::layer_norm`; `w`/`b` `None` is the `elementwise_affine=False` variant (the clip-diff
/// `FinalLayer`'s `norm_final`). Reductions run in the input dtype (the goldens are f32).
pub fn layer_norm(x: &Tensor, w: Option<&Tensor>, b: Option<&Tensor>, eps: f64) -> Result<Tensor> {
    let mean = x.mean_keepdim(D::Minus1)?;
    let centered = x.broadcast_sub(&mean)?;
    let var = centered.sqr()?.mean_keepdim(D::Minus1)?;
    let mut y = centered.broadcast_div(&(var + eps)?.sqrt()?)?;
    if let Some(w) = w {
        y = y.broadcast_mul(w)?;
    }
    if let Some(b) = b {
        y = y.broadcast_add(b)?;
    }
    Ok(y)
}
