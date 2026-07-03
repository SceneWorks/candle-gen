//! Shared tensor helpers (linear loaders, f32-island norms, scaled-dot-product attention) used by the
//! CLIP visual tower ([`crate::clip`]) and the DiT ([`crate::model`]). These mirror the conventions in
//! `candle-gen-wan`'s transformer (which keeps them `pub(crate)`): norms + softmax upcast to f32.

use candle_gen::candle_core::{DType, Result, Tensor, D};
use candle_gen::candle_nn::ops::softmax_last_dim;
use candle_gen::candle_nn::{Linear, VarBuilder};

/// A biased `[out, in]` Linear (`nn.Linear`) loaded by dotted name under `vb`.
pub(crate) fn linear(in_c: usize, out_c: usize, vb: VarBuilder) -> Result<Linear> {
    Ok(Linear::new(
        vb.get((out_c, in_c), "weight")?,
        Some(vb.get(out_c, "bias")?),
    ))
}

/// A `Conv3d`/`Conv2d` patch embed with `kernel == stride` read as a `[out, in·∏kernel]` dense Linear
/// (the non-overlapping conv is a patchify + linear; the patchify feature order matches the conv
/// weight flatten order). `kernel` = the spatial kernel dims (e.g. `[1, 2, 2]` or `[14, 14]`).
pub(crate) fn conv_as_linear(
    out_c: usize,
    in_c: usize,
    kernel: &[usize],
    weight_name: &str,
    bias_name: Option<&str>,
    vb: &VarBuilder,
) -> Result<Linear> {
    let kernel_numel: usize = kernel.iter().product();
    let infeat = in_c * kernel_numel;
    // The raw conv weight is `[out, in, k...]`; flatten the kernel dims into `in`.
    let mut shape = vec![out_c, in_c];
    shape.extend_from_slice(kernel);
    let w = vb.get(shape, weight_name)?.reshape((out_c, infeat))?;
    let b = match bias_name {
        Some(n) => Some(vb.get(out_c, n)?),
        None => None,
    };
    Ok(Linear::new(w, b))
}

/// LayerNorm over the last dim with affine weight+bias, computed in f32 (returns f32).
pub(crate) fn ln_affine(x: &Tensor, w: &Tensor, b: &Tensor, eps: f64) -> Result<Tensor> {
    let xf = x.to_dtype(DType::F32)?;
    let mean = xf.mean_keepdim(D::Minus1)?;
    let xc = xf.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
    let n = xc.broadcast_div(&(var + eps)?.sqrt()?)?;
    n.broadcast_mul(&w.to_dtype(DType::F32)?)?
        .broadcast_add(&b.to_dtype(DType::F32)?)
}

/// Non-affine LayerNorm (`WanLayerNorm(elementwise_affine=False)`) over the last dim, in f32.
pub(crate) fn ln_no_affine(x: &Tensor, eps: f64) -> Result<Tensor> {
    let xf = x.to_dtype(DType::F32)?;
    let mean = xf.mean_keepdim(D::Minus1)?;
    let xc = xf.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
    xc.broadcast_div(&(var + eps)?.sqrt()?)
}

/// RMSNorm over the last dim (qk-norm) with affine weight, computed in f32, cast back to `x`'s dtype.
pub(crate) fn rms(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let dt = x.dtype();
    let xf = x.to_dtype(DType::F32)?;
    let var = xf.sqr()?.mean_keepdim(D::Minus1)?;
    let normed = xf.broadcast_div(&(var + eps)?.sqrt()?)?;
    normed
        .broadcast_mul(&weight.to_dtype(DType::F32)?)?
        .to_dtype(dt)
}

/// Scaled-dot-product attention. `q,k,v`: `[B, H, S*, d]`; softmax upcast to f32.
///
/// i32-overflow guard (sc-9116): the packed video-DiT scores `[B, H, S, S]` (S = target+ref+pose+mask
/// tokens) exceed `i32::MAX` at 1024²+ / longer clips (e.g. a 64×64 latent → ~25k tokens → `~40·25k² ≈
/// 2.5e10`), silently corrupting the tail rows on the candle CUDA kernels. The shared budgeted helper
/// chunks over the query rows (byte-identical for the common sizes); the softmax closure preserves the
/// exact f32-upcast `softmax_last_dim`.
pub(crate) fn sdpa(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64) -> Result<Tensor> {
    let dt = q.dtype();
    candle_gen::sdpa_budgeted_bhsd(
        q,
        k,
        v,
        scale,
        None,
        |s| softmax_last_dim(&s.to_dtype(DType::F32)?)?.to_dtype(dt),
        candle_gen::ATTN_SCORES_BUDGET,
    )
}

/// Patchify `[C, T, H, W]` with patch `(pt, ph, pw)` → tokens `[L, C·pt·ph·pw]` (feature order
/// `(c, pt, ph, pw)`, token order `(t, h, w)`), plus the patch grid `(tg, hg, wg)`. The non-overlapping
/// `Conv3d` patch-embed (read as a `[dim, C·∏patch]` Linear) consumes these tokens directly, with the
/// feature flatten matching the conv weight flatten `(in, kt, kh, kw)`.
pub(crate) fn patchify(
    x: &Tensor,
    patch: (usize, usize, usize),
) -> Result<(Tensor, (usize, usize, usize))> {
    let (c, t, h, w) = x.dims4()?;
    let (pt, ph, pw) = patch;
    let (tg, hg, wg) = (t / pt, h / ph, w / pw);
    let tok = x
        .reshape(&[c, tg, pt, hg, ph, wg, pw][..])?
        .permute([1, 3, 5, 0, 2, 4, 6])? // [tg, hg, wg, c, pt, ph, pw]
        .contiguous()?
        .reshape((tg * hg * wg, c * pt * ph * pw))?;
    Ok((tok, (tg, hg, wg)))
}

/// Inverse of [`patchify`]: tokens `[L, out·pt·ph·pw]` for grid `(tg, hg, wg)` → `[out, tg·pt, hg·ph,
/// wg·pw]`.
pub(crate) fn unpatchify(
    tok: &Tensor,
    grid: (usize, usize, usize),
    out: usize,
    patch: (usize, usize, usize),
) -> Result<Tensor> {
    let (tg, hg, wg) = grid;
    let (pt, ph, pw) = patch;
    // The head projects each token to `pt·ph·pw·out` features in `(pt, ph, pw, out)` order — the
    // channel (`out`) is INNERMOST — matching upstream `u.view(*grid, *patch_size, c)` +
    // `einsum('fhwpqrc->cfphqwr')` and the validated `mlx-gen-wan::unpatchify`. The previous
    // `(out, pt, ph, pw)` split (channel OUTERMOST) round-trips correctly only for `out == 1`
    // (so the C=1 unit test missed it) but for `out == 16` scrambles channels ↔ patch-pixels,
    // decoding to pure structured noise (sc-7078).
    tok.reshape(&[tg, hg, wg, pt, ph, pw, out][..])?
        .permute([6, 0, 3, 1, 4, 2, 5])? // [out, tg, pt, hg, ph, wg, pw]
        .contiguous()?
        .reshape((out, tg * pt, hg * ph, wg * pw))
}
