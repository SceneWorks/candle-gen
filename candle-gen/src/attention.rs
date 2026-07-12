//! Shared **i32-overflow-safe** scaled-dot-product attention (sc-9116 / epic 8979, the F-003 sweep).
//!
//! candle's CUDA elementwise/softmax/matmul kernels index tensor elements with **i32**. A scores
//! (or probs) tensor whose element count exceeds `i32::MAX` (~2.147e9) silently corrupts its tail —
//! the trailing query rows get garbage attention (a near-zero / wrong context) with no error. For a
//! DiT operating on image latent tokens, or a VAE spatial self-attention operating on `H·W` pixels,
//! the sequence length scales with the rendered resolution, so at the advertised max render sizes
//! (2048² image → ~16k DiT tokens, or a VAE bottleneck `H·W` → ~65k) the `[…,Sq,Sk]` scores tensor
//! blows past `i32::MAX`:
//!
//! - SDXL UNet self-attn @ 2048²: `8·65536² ≈ 3.4e10`.
//! - SD3 / Z-Image / Ideogram / Krea DiT @ 2048²: `~24·16384² ≈ 6e9`.
//! - chroma / flux2 / qwen-image VAE mid-block @ 2048²: `65536² ≈ 4.3e9`.
//!
//! F-003 (sc-8983) first fixed this for the flux2 / chroma / lens / qwen-image DiT transformers with a
//! per-crate `attention_budgeted` helper that chunks over the **query rows** — each query row's softmax
//! is over all keys and is independent of the other rows, so the chunked result is numerically identical
//! to the single pass, and the chunking only ever engages on the over-budget buckets. This module hoists
//! that pattern into the shared commons so the remaining audited sites (sc-9116) share one guarded copy
//! instead of a growing pile of near-identical per-crate copies.
//!
//! Two shapes are covered:
//! - [`sdpa_budgeted_bhsd`] — the 4-D DiT shape `[B, H, Sq, D]` (heads explicit), optional additive mask.
//! - [`sdpa_budgeted_flat`] — the 3-D shape `[N, Sq, D]` where `N` folds `B·H` (SDXL, whose attention
//!   reshapes heads into the batch dim) **or** `B` for a single-head VAE spatial attention (`N=B`,
//!   `Sq=H·W`).
//!
//! Both take a caller-supplied `softmax` closure so each site keeps its exact softmax semantics — the
//! composable `softmax(_, D::Minus1)` (grad-carrying, used by the trainers), the fused `softmax_last_dim`,
//! or an f32-upcast variant — unchanged. The helpers only wrap the scores matmul + softmax + value matmul
//! in the budgeted query-row chunking; the numerics of a single block are exactly the caller's.

use candle_core::{Result, Tensor, D};

/// Max elements in a single attention scores tensor before the query rows are chunked. candle CUDA
/// kernels index elements with **i32**, so a scores/probs tensor exceeding `i32::MAX` (~2.147e9)
/// silently corrupts its tail. 1.0e9 keeps each chunk well under the limit while leaving every render
/// size whose single-pass scores are already `≤ 1e9` a single un-chunked pass (byte-identical to the
/// pre-guard path). Matches the per-crate F-003 budget so the two never diverge.
pub const ATTN_SCORES_BUDGET: usize = 1_000_000_000;

/// The largest query-block length whose `[…, block, Sk]` scores element count stays within `budget`.
/// Returns the whole `sq` (single un-chunked pass) when the full scores tensor already fits — so the
/// common sizes are the unchanged single matmul+softmax+matmul. `rows_per_query` is the product of all
/// the leading (non-query, non-key) dims times `sk` — i.e. the element count contributed by ONE query
/// row (`B·H·Sk` for the 4-D shape, `N·Sk` for the flat shape).
fn query_block(rows_per_query: usize, sq: usize, budget: usize) -> usize {
    if rows_per_query.saturating_mul(sq) <= budget {
        sq
    } else {
        (budget / rows_per_query.max(1)).max(1)
    }
}

/// i32-overflow-safe SDPA over the 4-D shape `q,k,v: [B, H, Sq, D]` (k/v key length `Sk` may differ
/// from `Sq`), returning the attention output `[B, H, Sq, D]` (the caller does its own
/// transpose/head-merge, so this is a drop-in around an existing `matmul → softmax → matmul`).
///
/// `scale` multiplies the scores (`head_dim^-0.5` at the call sites). `mask`, if given, is an additive
/// bias broadcast onto the scores AFTER scaling; it must broadcast over the query rows (e.g. `[B,1,1,Sk]`
/// or `[B,1,Sq,Sk]` with a per-row layout that narrows consistently — the common `[B,1,1,Sk]` and the
/// full `[B,1,Sq,Sk]` both do). `softmax` is applied to each scores block over its last dim; pass the
/// exact closure the call site used (`softmax_last_dim`, composable `softmax(_, D::Minus1)`, or an
/// f32-upcast wrapper) so the numerics are unchanged.
///
/// When `budget` is large enough for the full `[B,H,Sq,Sk]` scores tensor this is a single pass,
/// byte-identical to the un-guarded `(q·kᵀ·scale) → softmax → ·v`. Otherwise it chunks over the query
/// rows; since each query row's softmax is over all keys and independent of the others, the chunked
/// result is numerically identical (up to the associativity-free `cat`).
pub fn sdpa_budgeted_bhsd(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    mask: Option<&Tensor>,
    softmax: impl Fn(&Tensor) -> Result<Tensor>,
    budget: usize,
) -> Result<Tensor> {
    let (b, h, sq, _d) = q.dims4()?;
    let sk = k.dim(2)?;
    let q = q.contiguous()?;
    let k_t = k.transpose(2, 3)?.contiguous()?;
    let v = v.contiguous()?;

    let block = query_block(b * h * sk, sq, budget);
    if block >= sq {
        let mut scores = (q.matmul(&k_t)? * scale)?;
        if let Some(m) = mask {
            scores = scores.broadcast_add(m)?;
        }
        let probs = softmax(&scores)?;
        return probs.matmul(&v);
    }

    let mut blocks = Vec::new();
    let mut start = 0;
    while start < sq {
        let len = block.min(sq - start);
        let mut scores = (q.narrow(2, start, len)?.matmul(&k_t)? * scale)?;
        if let Some(m) = mask {
            // A `[B,1,1,Sk]` mask broadcasts identically onto every query chunk; a per-query
            // `[B,1,Sq,Sk]` mask must be narrowed to the same rows so each chunk sees its own slice.
            let m = if m.dim(2)? == sq {
                m.narrow(2, start, len)?
            } else {
                m.clone()
            };
            scores = scores.broadcast_add(&m)?;
        }
        let probs = softmax(&scores)?;
        blocks.push(probs.matmul(&v)?); // [B,H,len,D]
        start += len;
    }
    Tensor::cat(&blocks, 2) // [B,H,Sq,D]
}

/// i32-overflow-safe SDPA over the 3-D shape `q,k,v: [N, Sq, D]`, returning `[N, Sq, D]`. `N` folds the
/// leading dims — `B·H` for SDXL's head-into-batch attention, or `B` for a single-head VAE spatial
/// self-attention where `Sq = H·W`. `scale`/`softmax` behave as in [`sdpa_budgeted_bhsd`]; there is no
/// mask parameter (none of the flat-shape call sites use one). A drop-in around an existing
/// `q.matmul(kᵀ)·scale → softmax → ·v` on the 3-D tensors.
pub fn sdpa_budgeted_flat(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    softmax: impl Fn(&Tensor) -> Result<Tensor>,
    budget: usize,
) -> Result<Tensor> {
    let (n, sq, _d) = q.dims3()?;
    let sk = k.dim(1)?;
    let q = q.contiguous()?;
    let k_t = k.transpose(D::Minus1, D::Minus2)?.contiguous()?;
    let v = v.contiguous()?;

    let block = query_block(n * sk, sq, budget);
    if block >= sq {
        let scores = (q.matmul(&k_t)? * scale)?;
        let probs = softmax(&scores)?;
        return probs.matmul(&v);
    }

    let mut blocks = Vec::new();
    let mut start = 0;
    while start < sq {
        let len = block.min(sq - start);
        let scores = (q.narrow(1, start, len)?.matmul(&k_t)? * scale)?;
        let probs = softmax(&scores)?;
        blocks.push(probs.matmul(&v)?); // [N,len,D]
        start += len;
    }
    Tensor::cat(&blocks, 1) // [N,Sq,D]
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::ops::softmax_last_dim;

    fn approx_eq(a: &Tensor, b: &Tensor) {
        assert_eq!(a.dims(), b.dims());
        let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (x, y) in a.iter().zip(&b) {
            assert!(
                (x - y).abs() < 1e-5,
                "chunked attention diverged: {x} vs {y}"
            );
        }
    }

    #[test]
    fn bhsd_chunked_matches_single_pass() {
        // Per-query-row softmax is independent, so chunking over query rows (forced via a tiny budget)
        // must match the single un-chunked pass — the i32-overflow guard invariant (sc-9116).
        let dev = Device::Cpu;
        let (b, h, s, d) = (1usize, 2usize, 7usize, 4usize);
        let q = Tensor::randn(0f32, 1f32, (b, h, s, d), &dev).unwrap();
        let k = Tensor::randn(0f32, 1f32, (b, h, s, d), &dev).unwrap();
        let v = Tensor::randn(0f32, 1f32, (b, h, s, d), &dev).unwrap();
        let sm = |x: &Tensor| softmax_last_dim(x);
        let single = sdpa_budgeted_bhsd(&q, &k, &v, 0.5, None, sm, usize::MAX).unwrap();
        // Full single pass, tiny budget → single-row chunks, and a MID-SIZE budget forcing multi-row
        // chunks + a remainder (block=3 over s=7 → chunks 3,3,1) — the sc-9116 test-hardening ask.
        approx_eq(
            &single,
            &sdpa_budgeted_bhsd(&q, &k, &v, 0.5, None, sm, 1).unwrap(),
        );
        approx_eq(
            &single,
            // budget = b·h·sk·block = 1·2·7·3 = 42 → block = 42/(1·2·7) = 3.
            &sdpa_budgeted_bhsd(&q, &k, &v, 0.5, None, sm, 42).unwrap(),
        );
    }

    #[test]
    fn bhsd_masked_chunked_matches_single_pass() {
        // With an additive `[B,1,1,Sk]` mask the chunked path must still match: the mask broadcasts
        // identically onto every query chunk.
        let dev = Device::Cpu;
        let (b, h, s, d) = (1usize, 2usize, 7usize, 4usize);
        let q = Tensor::randn(0f32, 1f32, (b, h, s, d), &dev).unwrap();
        let k = Tensor::randn(0f32, 1f32, (b, h, s, d), &dev).unwrap();
        let v = Tensor::randn(0f32, 1f32, (b, h, s, d), &dev).unwrap();
        let mask = Tensor::randn(0f32, 1f32, (b, 1, 1, s), &dev).unwrap();
        let sm = |x: &Tensor| softmax_last_dim(x);
        let single = sdpa_budgeted_bhsd(&q, &k, &v, 0.5, Some(&mask), sm, usize::MAX).unwrap();
        approx_eq(
            &single,
            &sdpa_budgeted_bhsd(&q, &k, &v, 0.5, Some(&mask), sm, 42).unwrap(),
        );
    }

    #[test]
    fn bhsd_per_query_mask_chunked_matches_single_pass() {
        // A FULL per-query `[B,1,Sq,Sk]` additive mask (ideogram's `[B,1,L,L]` shape) exercises the
        // narrow-slice branch: each query chunk must see its OWN mask rows (`narrow(2, start, len)`),
        // not the whole mask. A mid-size budget (block=3 over s=7 → 3,3,1) forces the multi-row narrow.
        let dev = Device::Cpu;
        let (b, h, s, d) = (1usize, 2usize, 7usize, 4usize);
        let q = Tensor::randn(0f32, 1f32, (b, h, s, d), &dev).unwrap();
        let k = Tensor::randn(0f32, 1f32, (b, h, s, d), &dev).unwrap();
        let v = Tensor::randn(0f32, 1f32, (b, h, s, d), &dev).unwrap();
        // Distinct per-(query,key) bias so a wrong (un-narrowed / mis-aligned) slice would diverge.
        let mask = Tensor::randn(0f32, 1f32, (b, 1, s, s), &dev).unwrap();
        let sm = |x: &Tensor| softmax_last_dim(x);
        let single = sdpa_budgeted_bhsd(&q, &k, &v, 0.5, Some(&mask), sm, usize::MAX).unwrap();
        approx_eq(
            &single,
            &sdpa_budgeted_bhsd(&q, &k, &v, 0.5, Some(&mask), sm, 1).unwrap(),
        );
        approx_eq(
            &single,
            &sdpa_budgeted_bhsd(&q, &k, &v, 0.5, Some(&mask), sm, 42).unwrap(),
        );
    }

    #[test]
    fn guard_fires_at_advertised_sizes_sc11154() {
        // sc-11154 / F-081: the five newly-swept sites overflow i32 *within* their advertised,
        // `validate`-accepted envelopes. Assert the shared budget (`ATTN_SCORES_BUDGET`) engages the
        // query-row chunking at each site's advertised over-threshold size — and, critically, does NOT
        // chunk a comfortably in-budget size (so the common path stays the byte-identical single pass).
        // `rows_per_query` is the element count contributed by ONE query row (`N·Sk` flat, `B·H·Sk` 4-D).
        let b = ATTN_SCORES_BUDGET;

        // (a) stock SDXL UNet self-attn @ 2048² (heads-into-batch flat): N = B·H = 2·10, Sk = Sq = 16384
        // → 2·10·16384² ≈ 5.4e9. Chunk length must be < Sq.
        assert!(query_block(2 * 10 * 16384, 16384, b) < 16384);
        // (b) FLUX.1 VAE mid-block @ 2048² (single-head flat): N = 1, Sk = Sq = 65536 → 65536² ≈ 4.3e9.
        assert!(query_block(65536, 65536, b) < 65536);
        // (c) boogu Qwen3-VL ViT at a ~3.0 MP reference (4-D): B·H = 1·16, Sk = Sq = 11585 → 16·11585².
        assert!(query_block(16 * 11585, 11585, b) < 11585);
        // (d) krea grounded TE at the inclusive 8192-token cap (4-D): B·H = 1·32, Sk = Sq = 8192 → 2^31.
        assert!(query_block(32 * 8192, 8192, b) < 8192);
        // (e) sensenova ~8.2k-token image prefill (4-D), heads = 32: 32·8192² > i32::MAX.
        assert!(query_block(32 * 8192, 8200, b) < 8200);

        // Below the budget every one of these families runs a SINGLE un-chunked pass (block == Sq). A
        // 512² SDXL attn (Sq = 4096, N = 20 → 20·4096² ≈ 3.4e8) and a 1024² FLUX VAE (Sq = 16384 →
        // 16384² ≈ 2.7e8) both fit, so the guard is a no-op there.
        assert_eq!(query_block(20 * 4096, 4096, b), 4096);
        assert_eq!(query_block(16384, 16384, b), 16384);
    }

    #[test]
    fn flat_chunked_matches_single_pass() {
        // The 3-D (heads-folded / single-head VAE) shape, same invariant.
        let dev = Device::Cpu;
        let (n, s, d) = (3usize, 7usize, 4usize);
        let q = Tensor::randn(0f32, 1f32, (n, s, d), &dev).unwrap();
        let k = Tensor::randn(0f32, 1f32, (n, s, d), &dev).unwrap();
        let v = Tensor::randn(0f32, 1f32, (n, s, d), &dev).unwrap();
        let sm = |x: &Tensor| softmax_last_dim(x);
        let single = sdpa_budgeted_flat(&q, &k, &v, 0.5, sm, usize::MAX).unwrap();
        approx_eq(
            &single,
            &sdpa_budgeted_flat(&q, &k, &v, 0.5, sm, 1).unwrap(),
        );
        approx_eq(
            &single,
            // budget = n·sk·block = 3·7·3 = 63 → block = 63/(3·7) = 3.
            &sdpa_budgeted_flat(&q, &k, &v, 0.5, sm, 63).unwrap(),
        );
    }
}
