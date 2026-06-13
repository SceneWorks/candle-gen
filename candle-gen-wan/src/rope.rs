//! Wan's **3-axis (frame, height, width) interleaved RoPE** for the DiT, a port of diffusers
//! `WanRotaryPosEmbed`. `head_dim = 128` splits as `h_dim = w_dim = 2·(128//6) = 42` and
//! `t_dim = 128 − 84 = 44`; each axis contributes `dim/2` frequencies → `22 + 21 + 21 = 64`
//! per token (= `head_dim/2`). θ = 10000, positions are the raw grid indices (no centering).
//!
//! diffusers builds `cos`/`sin` with `repeat_interleave` (pairs `2k`, `2k+1` equal) and applies
//! `out[2k] = x[2k]·cos_k − x[2k+1]·sin_k`, `out[2k+1] = x[2k]·sin_k + x[2k+1]·cos_k` — exactly
//! candle's interleaved `rope_i` over the de-duplicated half tables `cos_k`/`sin_k`.

use candle_gen::candle_core::{DType, Device, Result, Tensor};
use candle_gen::candle_nn::rotary_emb::rope_i;

use crate::config::TransformerConfig;

pub struct WanRope {
    theta: f64,
    t_dim: usize,
    a_dim: usize, // height == width axis dim
    half: usize,  // 64
}

impl WanRope {
    pub fn new(cfg: &TransformerConfig) -> Self {
        let a_dim = 2 * (cfg.head_dim / 6); // 42
        let t_dim = cfg.head_dim - 2 * a_dim; // 44
        Self {
            theta: cfg.rope_theta,
            t_dim,
            a_dim,
            half: cfg.head_dim / 2,
        }
    }

    /// Per-axis inverse frequencies `theta^{-(2k)/D}`, `k = 0..D/2`.
    fn inv_freq(&self, dim: usize) -> Vec<f64> {
        (0..dim / 2)
            .map(|k| 1.0 / self.theta.powf((2 * k) as f64 / dim as f64))
            .collect()
    }

    /// Build `(cos, sin)` `[L, 64]` for the image-token grid `(ppf, pph, ppw)` in row-major
    /// `(f, h, w)` order (matching the patch-embed token flatten).
    pub fn cos_sin(
        &self,
        ppf: usize,
        pph: usize,
        ppw: usize,
        dev: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let inv_t = self.inv_freq(self.t_dim); // 22
        let inv_a = self.inv_freq(self.a_dim); // 21
        let (n_t, n_h) = (inv_t.len(), inv_a.len());
        let l = ppf * pph * ppw;
        let mut cos = vec![0f32; l * self.half];
        let mut sin = vec![0f32; l * self.half];
        for f in 0..ppf {
            for h in 0..pph {
                for w in 0..ppw {
                    let row = (f * pph + h) * ppw + w;
                    for (j, slot) in (0..self.half).enumerate() {
                        // Band layout: [t(0..22) | h(22..43) | w(43..64)].
                        let ang = if j < n_t {
                            f as f64 * inv_t[j]
                        } else if j < n_t + n_h {
                            h as f64 * inv_a[j - n_t]
                        } else {
                            w as f64 * inv_a[j - n_t - n_h]
                        };
                        let off = row * self.half + slot;
                        cos[off] = ang.cos() as f32;
                        sin[off] = ang.sin() as f32;
                    }
                }
            }
        }
        Ok((
            Tensor::from_vec(cos, (l, self.half), dev)?,
            Tensor::from_vec(sin, (l, self.half), dev)?,
        ))
    }
}

/// Apply interleaved RoPE to `x` `[B, H, S, head_dim]` with `cos`/`sin` `[S, head_dim/2]`. Computed
/// in f32 (cos/sin are f32), cast back to `x`'s dtype.
pub fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let dtype = x.dtype();
    let xf = x.to_dtype(DType::F32)?.contiguous()?;
    rope_i(&xf, cos, sin)?.to_dtype(dtype)
}
