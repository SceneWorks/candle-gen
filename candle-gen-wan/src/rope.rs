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

/// Wan RoPE base (matches [`WanRope::new`]'s `rope_theta` for the shipped configs). Used by the
/// source-id phase, which reads `head_dim` directly rather than a `TransformerConfig`.
const ROPE_THETA: f64 = 10000.0;

/// Compose the Bernini **source-id** rotary phase (`use_src_id_rotary_emb`) onto a precomputed spatial
/// RoPE `(cos, sin)` (each `[L, head_dim/2]` from [`WanRope::cos_sin`]) — the candle sibling of
/// `mlx-gen-bernini`'s `rope::apply_source_id`.
///
/// Upstream (`transformer_wan.py:282-289`) computes a per-source phase
/// `get_1d_rotary_pos_embed(head_dim, pos=source_id)` — a complex unit-modulus vector
/// `e^{i·source_id·ω_k}` of width `head_dim/2` with `ω_k = θ^(-2k/head_dim)` (θ = 10000) — and
/// **complex-multiplies** it into the spatial RoPE `freqs`. A complex multiply of unit-modulus
/// exponentials is an angle add, so per lane `k`: `θ_final[p,k] = θ_spatial[p,k] + source_id·ω_k`.
///
/// The per-lane `(cos(source_id·ω_k), sin(source_id·ω_k))` is computed in **f64** host-side (matching
/// the reference's `freqs_dtype=torch.float64`), then folded in with f32 candle ops. `source_id = 0.0`
/// returns the inputs unchanged (the noisy target keeps the plain spatial RoPE). Feed the result to
/// [`apply_rope`] exactly as the spatial table would be.
pub fn apply_source_id(
    cos_sp: &Tensor,
    sin_sp: &Tensor,
    source_id: f64,
    head_dim: usize,
) -> Result<(Tensor, Tensor)> {
    if source_id == 0.0 {
        return Ok((cos_sp.clone(), sin_sp.clone()));
    }
    let half_d = head_dim / 2;
    // Per-lane phase (cos(source_id·ω_k), sin(source_id·ω_k)), ω_k = θ^(-2k/head_dim), computed f64.
    let mut cos_id = vec![0f32; half_d];
    let mut sin_id = vec![0f32; half_d];
    for k in 0..half_d {
        let inv = ROPE_THETA.powf(-((2 * k) as f64) / head_dim as f64);
        let ang = source_id * inv;
        cos_id[k] = ang.cos() as f32;
        sin_id[k] = ang.sin() as f32;
    }
    let dev = cos_sp.device();
    // Shape [1, half_d] broadcasts over the L (token) axis of the spatial `[L, half_d]` tables.
    let cos_id = Tensor::from_vec(cos_id, (1, half_d), dev)?;
    let sin_id = Tensor::from_vec(sin_id, (1, half_d), dev)?;
    // Complex multiply (cos_sp + i·sin_sp)·(cos_id + i·sin_id).
    let cos_out = cos_sp
        .broadcast_mul(&cos_id)?
        .sub(&sin_sp.broadcast_mul(&sin_id)?)?;
    let sin_out = sin_sp
        .broadcast_mul(&cos_id)?
        .add(&cos_sp.broadcast_mul(&sin_id)?)?;
    Ok((cos_out, sin_out))
}

/// Assign source-ids to `n` conditioning sources (the noisy target separately keeps id 0) — the candle
/// sibling of `mlx-gen-bernini`'s `rope::assign_source_ids`. Mirrors upstream `_make_sids`
/// (`wan_diffusion.py:369-374`): ids start at 1; when `interpolate` is on and `n > max_trained`, the ids
/// are evenly spread into the trained range `[1, max_trained]` via `linspace(1, max_trained, n)`
/// (fractional) instead of extrapolating past the largest id seen in training; otherwise they are the
/// integers `1..=n`.
pub fn assign_source_ids(n: usize, max_trained: f64, interpolate: bool) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if interpolate && n as f64 > max_trained {
        if n == 1 {
            return vec![1.0];
        }
        let step = (max_trained - 1.0) / (n as f64 - 1.0);
        (0..n).map(|i| 1.0 + step * i as f64).collect()
    } else {
        (1..=n).map(|i| i as f64).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Device;

    fn tiny_rope() -> WanRope {
        WanRope::new(&TransformerConfig::t2v_14b()) // head_dim 128, θ 10000
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

    /// `source_id = 0` is the identity phase (the noisy target keeps the plain spatial RoPE).
    #[test]
    fn source_id_zero_is_identity() {
        let dev = Device::Cpu;
        let (cos, sin) = tiny_rope().cos_sin(2, 3, 4, &dev).unwrap();
        let (cos0, sin0) = apply_source_id(&cos, &sin, 0.0, 128).unwrap();
        assert_eq!(max_abs(&cos, &cos0), 0.0);
        assert_eq!(max_abs(&sin, &sin0), 0.0);
    }

    /// Composing a unit-modulus phase keeps `cos²+sin² = 1` per lane, so RoPE stays an orthogonal
    /// rotation (norm-preserving) for any source_id — an independent golden on the phase math.
    #[test]
    fn source_id_phase_is_norm_preserving() {
        let dev = Device::Cpu;
        let (cos, sin) = tiny_rope().cos_sin(1, 4, 4, &dev).unwrap(); // L = 16
        let (cos3, sin3) = apply_source_id(&cos, &sin, 3.0, 128).unwrap();
        let unit = (cos3.sqr().unwrap() + sin3.sqr().unwrap()).unwrap();
        let ones = Tensor::ones(unit.dims(), unit.dtype(), &dev).unwrap();
        assert!(max_abs(&unit, &ones) < 1e-5, "cos²+sin² ≠ 1");
    }

    /// Lane 0 has `ω_0 = θ^0 = 1`, so the phase for `source_id s` is exactly `(cos s, sin s)`; at
    /// grid position 0 the spatial angle is 0, so `cos'[0,0]=cos(s)`, `sin'[0,0]=sin(s)`. An
    /// independent closed-form golden (no torch/mlx) on the source-id complex multiply.
    #[test]
    fn source_id_phase_matches_manual_complex_multiply() {
        let dev = Device::Cpu;
        let (cos, sin) = tiny_rope().cos_sin(1, 1, 1, &dev).unwrap(); // L = 1
        let s = 2.5_f64;
        let (cos2, sin2) = apply_source_id(&cos, &sin, s, 128).unwrap();
        let c = cos2.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];
        let si = sin2.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];
        assert!((c - s.cos() as f32).abs() < 1e-5, "cos lane0 = {c}");
        assert!((si - s.sin() as f32).abs() < 1e-5, "sin lane0 = {si}");
    }

    /// A non-zero source id actually shifts the table (guards against a silent identity), and the shift
    /// is exactly the documented per-lane angle add at a chosen non-trivial grid position.
    #[test]
    fn source_id_shifts_the_table() {
        let dev = Device::Cpu;
        let rope = tiny_rope();
        let (cos, sin) = rope.cos_sin(2, 2, 2, &dev).unwrap();
        let (cos1, _sin1) = apply_source_id(&cos, &sin, 1.0, 128).unwrap();
        assert!(max_abs(&cos, &cos1) > 1e-4, "source_id 1 must shift cos");
        // Reconstruct lane `k` at token `row` from the documented angle add and check it matches.
        let (row, k, half) = (5usize, 7usize, 64usize);
        let cs = cos.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let sn = sin.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let (theta_sp_cos, theta_sp_sin) = (cs[row * half + k], sn[row * half + k]);
        let inv = 10000f64.powf(-((2 * k) as f64) / 128.0);
        let (pc, ps) = ((1.0 * inv).cos() as f32, (1.0 * inv).sin() as f32);
        let want_cos = theta_sp_cos * pc - theta_sp_sin * ps;
        let got_cos = cos1.flatten_all().unwrap().to_vec1::<f32>().unwrap()[row * half + k];
        assert!((got_cos - want_cos).abs() < 1e-5, "{got_cos} vs {want_cos}");
    }

    /// `assign_source_ids`: integers in-range, `linspace(1, max_trained, n)` when interpolating past the
    /// trained range, integers past the range when interpolation is off. Ported from mlx-gen-bernini.
    #[test]
    fn assign_source_ids_integer_and_interpolated() {
        assert_eq!(assign_source_ids(3, 5.0, true), vec![1.0, 2.0, 3.0]);
        assert_eq!(
            assign_source_ids(5, 5.0, true),
            vec![1.0, 2.0, 3.0, 4.0, 5.0]
        );
        let ids = assign_source_ids(9, 5.0, true);
        assert_eq!(ids.len(), 9);
        assert_eq!(ids[0], 1.0);
        assert_eq!(*ids.last().unwrap(), 5.0);
        assert!((ids[1] - 1.5).abs() < 1e-9);
        assert_eq!(
            assign_source_ids(7, 5.0, false),
            (1..=7).map(|i| i as f64).collect::<Vec<_>>()
        );
        assert_eq!(assign_source_ids(0, 5.0, true), Vec::<f64>::new());
    }
}
