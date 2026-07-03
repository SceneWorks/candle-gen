//! 2-D NTK-aware image RoPE + 1-D text RoPE — host-computed f32 `(cos, sin)` tables, then the
//! interleaved rotation applied to q/k. Faithful port of `pixeldit_official.py`'s
//! `precompute_freqs_cis_2d_ntk`, `fetch_pos_text`, and `apply_rotary_emb`.
//!
//! The reference packs each per-axis angle into an interleaved real `[N, head_dim/2, 2]` (cos, sin)
//! tensor where consecutive dim-pairs alternate the x-axis and y-axis rotation (element `2j` = x,
//! `2j+1` = y). We compute the `cos`/`sin` halves on the host (deterministic f64 math → f32) and
//! rotate the interleaved `(real, imag)` pairs exactly as `apply_rotary_emb` does. The whole net runs
//! f32, so the tables and the rotation are all f32 (no upcast dance).

use candle_gen::candle_core::{Device, Tensor};
use candle_gen::Result;

/// Host `(cos, sin)` tables `[L, head_dim/2]` (f32) for the 2-D NTK-aware image RoPE.
///
/// Mirrors `precompute_freqs_cis_2d_ntk(dim=head_dim, height=hs, width=ws, ref_grid_h, ref_grid_w,
/// theta=10000, scale=16)`. Token order is row-major over `(hs, ws)` with `ws` fastest; dim-pair
/// `m=2j` rotates by the x-axis (width) angle, `m=2j+1` by the y-axis (height) angle.
#[allow(clippy::too_many_arguments)]
pub fn rope_2d_ntk(
    head_dim: i32,
    hs: i32,
    ws: i32,
    ref_grid_h: i32,
    ref_grid_w: i32,
    theta: f32,
    scale: f32,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let dim = head_dim as f64;
    let dim_axis = dim / 2.0;
    let ntk_exp = if dim_axis > 2.0 {
        dim_axis / (dim_axis - 2.0)
    } else {
        1.0
    };
    let h_scale = hs as f64 / ref_grid_h as f64;
    let w_scale = ws as f64 / ref_grid_w as f64;
    let h_theta = theta as f64 * h_scale.powf(ntk_exp);
    let w_theta = theta as f64 * w_scale.powf(ntk_exp);

    let lin = |n: i32, idx: i32| -> f64 {
        if n <= 1 {
            0.0
        } else {
            idx as f64 * scale as f64 / (n - 1) as f64
        }
    };
    let n_pairs = (head_dim / 4) as usize; // dim//4 complex pairs per axis -> dim//2 real angles
    let freqs_w: Vec<f64> = (0..n_pairs)
        .map(|j| 1.0 / w_theta.powf((4 * j) as f64 / dim))
        .collect();
    let freqs_h: Vec<f64> = (0..n_pairs)
        .map(|j| 1.0 / h_theta.powf((4 * j) as f64 / dim))
        .collect();

    let half = (head_dim / 2) as usize;
    let l = (hs * ws) as usize;
    let mut cos = vec![0f32; l * half];
    let mut sin = vec![0f32; l * half];
    for r in 0..hs {
        let yp = lin(hs, r);
        for c in 0..ws {
            let xp = lin(ws, c);
            let p = (r * ws + c) as usize;
            for m in 0..half {
                let j = m / 2;
                let angle = if m % 2 == 0 {
                    xp * freqs_w[j]
                } else {
                    yp * freqs_h[j]
                };
                cos[p * half + m] = angle.cos() as f32;
                sin[p * half + m] = angle.sin() as f32;
            }
        }
    }
    Ok((
        Tensor::from_vec(cos, (l, half), device)?,
        Tensor::from_vec(sin, (l, half), device)?,
    ))
}

/// Host `(cos, sin)` tables `[length, head_dim/2]` (f32) for the 1-D text RoPE.
///
/// Mirrors `fetch_pos_text`: `freqs[m] = theta^(-2m/head_dim)`, `angle[l,m] = l·freqs[m]`.
pub fn rope_1d_text(
    head_dim: i32,
    length: i32,
    theta: f32,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let half = (head_dim / 2) as usize;
    let freqs: Vec<f64> = (0..half)
        .map(|m| 1.0 / (theta as f64).powf((2 * m) as f64 / head_dim as f64))
        .collect();
    let len = length as usize;
    let mut cos = vec![0f32; len * half];
    let mut sin = vec![0f32; len * half];
    for l in 0..len {
        for m in 0..half {
            let angle = l as f64 * freqs[m];
            cos[l * half + m] = angle.cos() as f32;
            sin[l * half + m] = angle.sin() as f32;
        }
    }
    Ok((
        Tensor::from_vec(cos, (len, half), device)?,
        Tensor::from_vec(sin, (len, half), device)?,
    ))
}

/// Apply interleaved RoPE to `q`/`k` in `[B, H, S, D]` with `cos`/`sin` `[S, D/2]`. Pairs
/// `(x[2i], x[2i+1])` as `(real, imag)` and rotates by `cos/sin[i]` — bit-equivalent to
/// `apply_rotary_emb`'s `_rotate`. The head axis is a pure broadcast, so applying after the
/// `[B,H,S,D]` transpose is identical to the reference's pre-transpose apply.
pub fn apply_rope(q: &Tensor, k: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<(Tensor, Tensor)> {
    let (s, half) = cos.dims2()?;
    let cos = cos.reshape((1, 1, s, half))?;
    let sin = sin.reshape((1, 1, s, half))?;
    let one = |x: &Tensor| -> Result<Tensor> {
        let (b, h, seq, hd) = x.dims4()?;
        let x5 = x.reshape((b, h, seq, hd / 2, 2))?;
        let real = x5.narrow(4, 0, 1)?.reshape((b, h, seq, hd / 2))?;
        let imag = x5.narrow(4, 1, 1)?.reshape((b, h, seq, hd / 2))?;
        // (real·cos − imag·sin, imag·cos + real·sin), then re-interleave on a new last axis.
        let out0 = (real.broadcast_mul(&cos)? - imag.broadcast_mul(&sin)?)?;
        let out1 = (imag.broadcast_mul(&cos)? + real.broadcast_mul(&sin)?)?;
        let stacked = Tensor::stack(&[&out0, &out1], 4)?; // [b,h,seq,hd/2,2]
        Ok(stacked.reshape((b, h, seq, hd))?)
    };
    Ok((one(q)?, one(k)?))
}
