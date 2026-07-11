//! Krea 2 DiT 3-axis (t, h, w) unified RoPE — port of `mlx-gen-krea`'s `transformer/rope.rs` (the
//! reference `mmdit.py` `PositionalEncoding` + `rope` + `ropeapply`).
//!
//! Two facts from the reference, both parity-critical:
//!  1. **Complex *interleaved* rotation** (GPT-J / "lumina"): adjacent dims `(2k, 2k+1)` form a complex
//!     pair `x[2k] + i·x[2k+1]` rotated by `e^{iθ_k}`, *not* the half-split `[x1,x2]→[-x2,x1]`. (The
//!     same op as `candle-gen-boogu`'s `apply_interleaved_rope`.)
//!  2. **Three position axes with UNEQUAL sub-dims** `axes_dims_rope = [32,48,48]` (boogu's are equal,
//!     so its table builder doesn't generalize). The head-dim freq index `k ∈ [0, head_dim/2)` is split
//!     into three contiguous blocks of `axes[i]/2`, each block `i` using its own inverse frequencies
//!     `θ^(−2j/axes[i])` over its own position axis.
//!
//! **Position scheme** (reference `sampling.py::prepare`): text tokens are all `(0,0,0)`; image patch
//! tokens are `(0, row, col)` — the t-axis is **always 0**, so only the h/w axes carry position and the
//! text tokens get identity RoPE. The joint `[text; image]` table is applied to the whole single-stream
//! sequence (the text-fusion blocks use no RoPE).
//!
//! Inverse frequencies + angles are built on the host in **f64** (the reference's `rope` uses
//! `torch.float64`), then the `cos`/`sin` tables are materialized as f32 (`rope(...).float()`).

use candle_gen::candle_core::{DType, Device, Result, Tensor, D};

/// Precomputed `cos`/`sin` rotary tables for one forward pass, laid out `[1, cap_len + img_len,
/// head_dim/2]` (f32) in joint `[text; image]` order.
pub struct RopeTables {
    cos: Tensor,
    sin: Tensor,
}

impl RopeTables {
    /// Build the joint table for a text-to-image forward: `cap_len` text positions `(0,0,0)` followed
    /// by an `h_tokens × w_tokens` row-major image grid `(0, row, col)`. `axes` are the per-axis RoPE
    /// sub-dims (`[t,h,w]`, summing to `head_dim`); `theta` is `rope_theta`.
    pub fn build_t2i(
        cap_len: usize,
        h_tokens: usize,
        w_tokens: usize,
        axes: [usize; 3],
        theta: f64,
        device: &Device,
    ) -> Result<Self> {
        let mut positions = Vec::with_capacity(cap_len + h_tokens * w_tokens);
        for _ in 0..cap_len {
            positions.push((0.0, 0.0, 0.0));
        }
        for r in 0..h_tokens {
            for c in 0..w_tokens {
                positions.push((0.0, r as f64, c as f64));
            }
        }
        from_positions(&positions, axes, theta, device)
    }

    /// Build the joint table for a **Kontext-style edit** forward (epic 10871 / sc-10877): `cap_len`
    /// text positions `(0,0,0)`, then `n_refs` reference-image grids, then the target (noise) grid — the
    /// sequence order `[text, refs…, target]` from the reference `ComfyUI-Krea2Edit` node (refs BEFORE
    /// the noise, unlike Qwen-Edit). All grids share the target `h_tokens × w_tokens` shape (references
    /// are VAE-encoded at the target resolution). The `(frame, row, col)` position tuple carries the
    /// reference index on the **t-axis** (sub-dim `axes[0]`, unused/`0` in t2i): text = `(0,0,0)`, the
    /// target = `(0, row, col)` (frame 0, identical to t2i), and reference `i` = `(i+1, row, col)`.
    ///
    /// `n_refs == 0` reduces to `[text, target]` — **byte-identical** to [`Self::build_t2i`] (the target
    /// grid sits at frame 0, exactly the t2i image grid).
    pub fn build_edit(
        cap_len: usize,
        h_tokens: usize,
        w_tokens: usize,
        n_refs: usize,
        axes: [usize; 3],
        theta: f64,
        device: &Device,
    ) -> Result<Self> {
        let img_len = h_tokens * w_tokens;
        let mut positions = Vec::with_capacity(cap_len + (n_refs + 1) * img_len);
        for _ in 0..cap_len {
            positions.push((0.0, 0.0, 0.0));
        }
        // References first (frame i+1), then the target/noise grid (frame 0).
        for i in 0..n_refs {
            let frame = (i + 1) as f64;
            for r in 0..h_tokens {
                for c in 0..w_tokens {
                    positions.push((frame, r as f64, c as f64));
                }
            }
        }
        for r in 0..h_tokens {
            for c in 0..w_tokens {
                positions.push((0.0, r as f64, c as f64));
            }
        }
        from_positions(&positions, axes, theta, device)
    }

    /// `(cos, sin)` for the full joint `[text; image]` sequence (the single-stream blocks).
    pub fn joint(&self) -> (Tensor, Tensor) {
        (self.cos.clone(), self.sin.clone())
    }
}

/// Build the `cos`/`sin` tables from 3-axis positions. For freq block `i` (sub-dim `axes[i]`, so
/// `axes[i]/2` complex freqs) the inverse frequencies are `θ^(−2j/axes[i])` (`j ∈ [0, axes[i]/2)`),
/// each multiplied by that token's position on axis `i`. Computed in f64, stored f32.
fn from_positions(
    positions: &[(f64, f64, f64)],
    axes: [usize; 3],
    theta: f64,
    device: &Device,
) -> Result<RopeTables> {
    // Per-axis inverse frequencies in f64 (reference `rope`: `1 / (theta ** (arange(0,d,2)/d))`).
    let inv: Vec<Vec<f64>> = axes
        .iter()
        .map(|&d| {
            (0..d / 2)
                .map(|j| 1.0 / theta.powf((2 * j) as f64 / d as f64))
                .collect()
        })
        .collect();
    let half: usize = axes.iter().map(|d| d / 2).sum(); // head_dim/2

    let total = positions.len();
    let mut cos = vec![0f32; total * half];
    let mut sin = vec![0f32; total * half];
    for (t, &(p0, p1, p2)) in positions.iter().enumerate() {
        let pos = [p0, p1, p2];
        let mut k = 0usize; // running freq index across the three concatenated blocks
        for (axis, freqs) in inv.iter().enumerate() {
            for &f in freqs {
                let angle = pos[axis] * f;
                cos[t * half + k] = angle.cos() as f32;
                sin[t * half + k] = angle.sin() as f32;
                k += 1;
            }
        }
    }

    Ok(RopeTables {
        cos: Tensor::from_vec(cos, (1, total, half), device)?,
        sin: Tensor::from_vec(sin, (1, total, half), device)?,
    })
}

/// Apply the complex-interleaved rotary embedding to `x` in `[b, s, heads, head_dim]` layout.
///
/// `cos`/`sin` are `[1, s, head_dim/2]` (f32, broadcast over heads). For each adjacent pair
/// `(x[2k], x[2k+1])`:
///   `out[2k]   = x[2k]·cos_k − x[2k+1]·sin_k`
///   `out[2k+1] = x[2k]·sin_k + x[2k+1]·cos_k`
/// Computed in f32 (the reference upcasts), then cast back to `x`'s dtype.
pub fn apply_interleaved_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let dt = x.dtype();
    let (b, s, h, hd) = x.dims4()?;
    let half = hd / 2;

    // cos/sin: [1, s, half] → [1, s, 1, half] (broadcast over heads + the pair axis). They arrive as
    // `narrow`ed slices of the rope table, so contiguate before the reshape.
    let cos = cos.contiguous()?.reshape((1, s, 1, half))?;
    let sin = sin.contiguous()?.reshape((1, s, 1, half))?;

    // [b, s, h, hd] → [b, s, h, half, 2]; the last axis holds the (even, odd) complex pair.
    let xr = x.to_dtype(DType::F32)?.reshape((b, s, h, half, 2))?;
    let xe = xr.narrow(4, 0, 1)?.contiguous()?.reshape((b, s, h, half))?;
    let xo = xr.narrow(4, 1, 1)?.contiguous()?.reshape((b, s, h, half))?;

    let out_e = (xe.broadcast_mul(&cos)? - xo.broadcast_mul(&sin)?)?;
    let out_o = (xe.broadcast_mul(&sin)? + xo.broadcast_mul(&cos)?)?;

    // Re-interleave: stack on a new trailing axis → [b, s, h, half, 2] → [b, s, h, hd].
    let out = Tensor::stack(&[&out_e, &out_o], D::Minus1)?.reshape((b, s, h, hd))?;
    out.to_dtype(dt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Device;

    // head_dim 128 = sum([32,48,48]) ⇒ half = 16 + 24 + 24 = 64.
    const AXES: [usize; 3] = [32, 48, 48];
    const THETA: f64 = 1000.0;

    #[test]
    fn t2i_table_shape_and_text_identity() {
        let dev = Device::Cpu;
        let (cap, ht, wt) = (5usize, 4usize, 3usize);
        let r = RopeTables::build_t2i(cap, ht, wt, AXES, THETA, &dev).unwrap();
        let (cos, sin) = r.joint();
        let total = cap + ht * wt;
        assert_eq!(cos.dims(), &[1, total, 64]);
        assert_eq!(sin.dims(), &[1, total, 64]);
        // Text tokens are (0,0,0) → identity RoPE: cos = 1, sin = 0 everywhere in the text block.
        let cos_text = cos
            .narrow(1, 0, cap)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let sin_text = sin
            .narrow(1, 0, cap)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(cos_text.iter().all(|&v| (v - 1.0).abs() < 1e-6));
        assert!(sin_text.iter().all(|&v| v.abs() < 1e-6));
    }

    #[test]
    fn build_edit_zero_refs_equals_t2i() {
        // With no references, `[text, target]` is byte-identical to the t2i `[text, image]` table.
        let dev = Device::Cpu;
        let (cap, ht, wt) = (5usize, 4usize, 3usize);
        let t2i = RopeTables::build_t2i(cap, ht, wt, AXES, THETA, &dev).unwrap();
        let edit = RopeTables::build_edit(cap, ht, wt, 0, AXES, THETA, &dev).unwrap();
        for (a, b) in [(t2i.cos, edit.cos), (t2i.sin, edit.sin)] {
            assert_eq!(a.dims(), b.dims());
            let av = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let bv = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            assert_eq!(
                av, bv,
                "build_edit(n_refs=0) must equal build_t2i byte-for-byte"
            );
        }
    }

    #[test]
    fn build_edit_places_refs_at_successive_frames() {
        // Sequence `[text(cap), ref0(img), ref1(img), target(img)]`. The frame (t-axis, sub-dim 32 →
        // 16 freqs) distinguishes each reference; text + target stay at frame 0.
        let dev = Device::Cpu;
        let (cap, ht, wt, n_refs) = (2usize, 2usize, 2usize, 2usize);
        let img_len = ht * wt;
        let r = RopeTables::build_edit(cap, ht, wt, n_refs, AXES, THETA, &dev).unwrap();
        let (cos, _sin) = r.joint();
        let total = cap + (n_refs + 1) * img_len;
        assert_eq!(cos.dims(), &[1, total, 64]);

        // The first `axes[0]/2 = 16` freqs are the frame (t-axis). Frame 0 → cos = 1 over that block.
        let frame_block = |tok: usize| -> Vec<f32> {
            cos.narrow(1, tok, 1)
                .unwrap()
                .narrow(2, 0, 16)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        };
        // Text token (frame 0) and target token (last block, frame 0) → identity in the frame block.
        assert!(frame_block(0).iter().all(|&v| (v - 1.0).abs() < 1e-6));
        assert!(frame_block(total - 1)
            .iter()
            .all(|&v| (v - 1.0).abs() < 1e-6));
        // ref0 (frame 1) and ref1 (frame 2) rotate the frame block → NOT all ones.
        let ref0_tok = cap; // first reference token
        let ref1_tok = cap + img_len;
        assert!(frame_block(ref0_tok)
            .iter()
            .any(|&v| (v - 1.0).abs() > 1e-4));
        assert!(frame_block(ref1_tok)
            .iter()
            .any(|&v| (v - 1.0).abs() > 1e-4));
        // ref0 and ref1 differ (frame 1 vs 2).
        assert_ne!(frame_block(ref0_tok), frame_block(ref1_tok));
    }

    #[test]
    fn interleaved_rope_roundtrips_shape() {
        let dev = Device::Cpu;
        let r = RopeTables::build_t2i(2, 2, 2, AXES, THETA, &dev).unwrap();
        let (cos, sin) = r.joint();
        // [b, s, heads, head_dim] with s = 2 + 4 = 6, head_dim 128.
        let x = Tensor::randn(0f32, 1.0, (1, 6, 3, 128), &dev).unwrap();
        let y = apply_interleaved_rope(&x, &cos, &sin).unwrap();
        assert_eq!(y.dims(), x.dims());
        // Text tokens get identity rotation, so those rows are unchanged.
        let x0 = x
            .narrow(1, 0, 2)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let y0 = y
            .narrow(1, 0, 2)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let maxd = x0
            .iter()
            .zip(&y0)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            maxd < 1e-5,
            "text tokens should be identity under RoPE, maxd={maxd}"
        );
    }
}
