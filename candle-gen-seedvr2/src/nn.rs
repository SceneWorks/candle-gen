//! Shared nn helpers for the SeedVR2 candle port, matching the MLX reference semantics:
//! GroupNorm/RMSNorm compute in **f32** (cast back to the input dtype), dense scaled-dot-product
//! attention, a `[out,in]`-weight linear (`y = x·Wᵀ + b`), and the tanh-GELU.

use candle_gen::candle_core::{DType, Result, Tensor, D};
use candle_gen::candle_nn::ops::softmax_last_dim;

/// Dense layer `y = x·Wᵀ (+ b)` given the weight **already transposed** to `[in, out]` (the callers
/// transpose the loaded `[out,in]` weight once at construction — sc-8997/F-017; re-transposing per
/// forward materialized a fresh contiguous copy of the whole weight every call). Flattens all leading
/// dims into one 2-D GEMM and reshapes back — candle's `matmul` rejects the non-contiguous broadcasted
/// rhs that `broadcast_matmul` produces for a high-rank `x` (e.g. the 5-D patchified tokens), and the
/// flattened GEMM is faster.
pub fn linear(x: &Tensor, wt: &Tensor, b: Option<&Tensor>) -> Result<Tensor> {
    let (in_dim, out_dim) = (wt.dim(0)?, wt.dim(1)?); // [in, out]
    let dims = x.dims().to_vec();
    let lead: usize = dims[..dims.len() - 1].iter().product();
    let y = x.contiguous()?.reshape((lead, in_dim))?.matmul(wt)?; // [lead, out]
    let mut out_shape = dims[..dims.len() - 1].to_vec();
    out_shape.push(out_dim);
    let y = y.reshape(out_shape)?;
    match b {
        Some(b) => y.broadcast_add(b),
        None => Ok(y),
    }
}

/// Pre-transpose a loaded `[out,in]` weight to the contiguous `[in,out]` layout [`linear`] consumes,
/// so the per-forward GEMM has no transpose/copy (sc-8997/F-017).
pub fn transpose_weight(w: &Tensor) -> Result<Tensor> {
    w.t()?.contiguous()
}

/// GroupNorm over `[N, C, *spatial]` (channels in dim 1, any trailing rank), computed in f32 with a
/// learnable `[C]` weight/bias. Matches mlx's channels-last f32 GroupNorm bit-for-bit at f32.
pub fn group_norm(x: &Tensor, w: &Tensor, b: &Tensor, groups: usize, eps: f64) -> Result<Tensor> {
    let sh = x.dims().to_vec();
    let (n, c) = (sh[0], sh[1]);
    let rest: usize = sh[2..].iter().product();
    let dt = x.dtype();
    let xf = x.to_dtype(DType::F32)?;
    let g = xf.reshape((n, groups, (c / groups) * rest))?;
    let mean = g.mean_keepdim(2)?;
    let centered = g.broadcast_sub(&mean)?;
    let var = centered.sqr()?.mean_keepdim(2)?;
    let normed = centered
        .broadcast_div(&var.affine(1.0, eps)?.sqrt()?)?
        .reshape(sh.clone())?;
    // affine: reshape [C] → [1, C, 1, …] to broadcast over the trailing dims.
    let mut ws = vec![1usize; sh.len()];
    ws[1] = c;
    let wv = w.to_dtype(DType::F32)?.reshape(ws.clone())?;
    let bv = b.to_dtype(DType::F32)?.reshape(ws)?;
    normed.broadcast_mul(&wv)?.broadcast_add(&bv)?.to_dtype(dt)
}

/// RMSNorm over the last dim with a `[dim]` weight, computed in f32.
pub fn rms_norm(x: &Tensor, w: &Tensor, eps: f64) -> Result<Tensor> {
    let dt = x.dtype();
    let xf = x.to_dtype(DType::F32)?;
    let ms = xf.sqr()?.mean_keepdim(D::Minus1)?;
    let normed = xf.broadcast_div(&ms.affine(1.0, eps)?.sqrt()?)?;
    normed.broadcast_mul(&w.to_dtype(DType::F32)?)?.to_dtype(dt)
}

/// SiLU (x·sigmoid(x)).
pub fn silu(x: &Tensor) -> Result<Tensor> {
    let sig = (x.neg()?.exp()? + 1.0)?.recip()?;
    x.mul(&sig)
}

/// tanh-approximation GELU (the reference `gelu_tanh`).
pub fn gelu_tanh(x: &Tensor) -> Result<Tensor> {
    x.gelu()
}

/// Dense scaled-dot-product attention over `[B, H, S, D]` (no mask): `softmax(q·kᵀ·scale)·v`.
pub fn sdpa(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64) -> Result<Tensor> {
    let q = q.contiguous()?;
    let kt = k.transpose(D::Minus2, D::Minus1)?.contiguous()?; // [B,H,D,S]
    let scores = (q.matmul(&kt)? * scale)?;
    let attn = softmax_last_dim(&scores)?;
    attn.matmul(&v.contiguous()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Device;

    /// The old `linear` re-derived `[in,out]` inside the forward via `w.t()?.contiguous()?`. The
    /// sc-8997/F-017 refactor moves that transpose to load time (`transpose_weight`) so the per-forward
    /// matmul is a plain GEMM. This asserts the two are **bit-identical** for a high-rank (5-D, like the
    /// patchified DiT tokens) input across every leading dim, so the perf refactor changed nothing
    /// numerically. It also structurally checks `transpose_weight` produces exactly `[in,out]` from a
    /// `[out,in]` weight (the "transposed exactly once, at load" invariant).
    fn linear_old(x: &Tensor, w: &Tensor, b: Option<&Tensor>) -> Result<Tensor> {
        let wt = w.t()?.contiguous()?; // [in, out] — the pre-fix per-forward transpose+copy
        let (in_dim, out_dim) = (wt.dim(0)?, wt.dim(1)?);
        let dims = x.dims().to_vec();
        let lead: usize = dims[..dims.len() - 1].iter().product();
        let y = x.contiguous()?.reshape((lead, in_dim))?.matmul(&wt)?;
        let mut out_shape = dims[..dims.len() - 1].to_vec();
        out_shape.push(out_dim);
        let y = y.reshape(out_shape)?;
        match b {
            Some(b) => y.broadcast_add(b),
            None => Ok(y),
        }
    }

    #[test]
    fn linear_pretransposed_matches_pre_fix_bit_identical() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (7usize, 5usize);
        // torch-native `[out, in]` weight, as loaded from the checkpoint.
        let w = Tensor::randn(0f32, 1.0, (out_dim, in_dim), &dev)?;
        let b = Tensor::randn(0f32, 1.0, out_dim, &dev)?;
        // 5-D input (B, T, H, W, in) — the same high-rank shape the DiT patchify feeds the linear.
        let x = Tensor::randn(0f32, 1.0, (2usize, 3usize, 2usize, 2usize, in_dim), &dev)?;

        // `transpose_weight` is exactly `[in, out]` (transposed exactly once, at load).
        let wt = transpose_weight(&w)?;
        assert_eq!(wt.dims(), &[in_dim, out_dim]);

        for bias in [None, Some(&b)] {
            let old = linear_old(&x, &w, bias)?.flatten_all()?.to_vec1::<f32>()?;
            let new = linear(&x, &wt, bias)?.flatten_all()?.to_vec1::<f32>()?;
            assert_eq!(old.len(), new.len());
            for (o, n) in old.iter().zip(new.iter()) {
                assert_eq!(
                    o.to_bits(),
                    n.to_bits(),
                    "linear output changed: {o} vs {n}"
                );
            }
        }
        Ok(())
    }

    /// Brute-force GroupNorm over NCTHW (joint over c/g, T, H, W) — confirms the reshape groups
    /// channels correctly and fully normalizes at T>1 (a mis-grouped reshape would under-normalize
    /// and explode downstream — the candle SeedVR2 video failure mode).
    #[test]
    fn group_norm_matches_bruteforce_t_gt_1() -> Result<()> {
        let dev = Device::Cpu;
        let (n, c, t, h, wd, groups) = (1usize, 8usize, 3usize, 2usize, 2usize, 4usize);
        let x = Tensor::randn(0f32, 2.0, (n, c, t, h, wd), &dev)?;
        let w = Tensor::randn(0f32, 1.0, c, &dev)?;
        let b = Tensor::randn(0f32, 1.0, c, &dev)?;
        let eps = 1e-6;
        let got = group_norm(&x, &w, &b, groups, eps)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        let xv = x.flatten_all()?.to_vec1::<f32>()?; // [n,c,t,h,w] row-major
        let wv = w.to_vec1::<f32>()?;
        let bv = b.to_vec1::<f32>()?;
        let rest = t * h * wd;
        let cpg = c / groups;
        let idx = |ci: usize, k: usize| ci * rest + k; // n=1
        let mut exp = vec![0f32; c * rest];
        for gr in 0..groups {
            // collect group elements (cpg channels × rest)
            let mut vals = Vec::with_capacity(cpg * rest);
            for ci in gr * cpg..(gr + 1) * cpg {
                for k in 0..rest {
                    vals.push(xv[idx(ci, k)]);
                }
            }
            let mean = vals.iter().sum::<f32>() / vals.len() as f32;
            let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / vals.len() as f32;
            let std = (var + eps as f32).sqrt();
            for ci in gr * cpg..(gr + 1) * cpg {
                for k in 0..rest {
                    let norm = (xv[idx(ci, k)] - mean) / std;
                    exp[idx(ci, k)] = norm * wv[ci] + bv[ci];
                }
            }
        }
        let max_err = got
            .iter()
            .zip(exp.iter())
            .map(|(a, e)| (a - e).abs())
            .fold(0f32, f32::max);
        assert!(max_err < 1e-3, "group_norm wrong at T>1: max_err={max_err}");
        Ok(())
    }
}
