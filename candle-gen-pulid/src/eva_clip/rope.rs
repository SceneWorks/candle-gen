//! EVA `VisionRotaryEmbeddingFast` — the 2-D vision RoPE applied to the patch tokens of each attention
//! block. Candle port of `eva_clip/rope.py` (the MLX sibling's `rope.rs`).
//!
//! TWO things make this distinct from the half-split text RoPE:
//!   1. **Interleaved** `rotate_half`: pairs are adjacent — `out[2i] = -x[2i+1]; out[2i+1] = x[2i]`
//!      (einops `rearrange(x,'(d r)->d r',r=2)` then `stack(-x2,x1)`), with the per-freq table
//!      duplicated adjacently. NOT the `[-x2, x1]` half-split form.
//!   2. **2-D** table: the first half of each head dim is driven by the patch *row*, the second half by
//!      the patch *column*.
//!
//! The freqs are deterministic (no weights); we rebuild them on the host (theta-pow seed, f64) and cast
//! to f32 — the checkpoint's `rope.freqs_cos/sin` buffers are redundant.

use candle_core::{DType, Device, Tensor};

/// The fixed RoPE cos/sin tables `[grid²=576, head_dim=64]` for EVA02-CLIP-L-14-336.
pub struct VisionRope {
    cos: Tensor,
    sin: Tensor,
}

impl VisionRope {
    /// Rebuild the `VisionRotaryEmbeddingFast` cos/sin tables for a square patch grid, on `device`.
    ///
    /// `head_dim` = 64, `grid` = image/patch = 24, `pt_seq_len` = 16 (the pretraining grid; EVA
    /// interpolates frequencies via `t = arange(grid)/grid * pt_seq_len`).
    pub fn build(
        head_dim: usize,
        grid: usize,
        pt_seq_len: usize,
        theta: f64,
        device: &Device,
    ) -> candle_core::Result<Self> {
        let half = head_dim / 2; // rope dim = head_dim/2 = 32
        let nfreq = half / 2; // 16 base frequencies (arange(0,half,2))
                              // freqs[j] = 1 / theta^(2j / half)
        let freqs: Vec<f64> = (0..nfreq)
            .map(|j| 1.0 / theta.powf((2 * j) as f64 / half as f64))
            .collect();
        // t[i] = i/grid * pt_seq_len
        let t: Vec<f64> = (0..grid)
            .map(|i| i as f64 / grid as f64 * pt_seq_len as f64)
            .collect();
        // per-axis table rg[pos, d] for d in 0..half: rg[pos, 2j] = rg[pos, 2j+1] = t[pos]*freqs[j]
        let axis_table = |pos: usize| -> Vec<f64> {
            let mut row = vec![0.0f64; half];
            for j in 0..nfreq {
                let val = t[pos] * freqs[j];
                row[2 * j] = val;
                row[2 * j + 1] = val;
            }
            row
        };
        // full[h*grid+w, :] = concat(axis_table(h), axis_table(w)) → [576, 64]
        let n = grid * grid;
        let mut cos = vec![0.0f32; n * head_dim];
        let mut sin = vec![0.0f32; n * head_dim];
        for h in 0..grid {
            let th = axis_table(h);
            for w in 0..grid {
                let tw = axis_table(w);
                let base = (h * grid + w) * head_dim;
                for d in 0..half {
                    let (vh, vw) = (th[d], tw[d]);
                    cos[base + d] = vh.cos() as f32;
                    sin[base + d] = vh.sin() as f32;
                    cos[base + half + d] = vw.cos() as f32;
                    sin[base + half + d] = vw.sin() as f32;
                }
            }
        }
        Ok(Self {
            cos: Tensor::from_vec(cos, (n, head_dim), device)?,
            sin: Tensor::from_vec(sin, (n, head_dim), device)?,
        })
    }

    /// Apply RoPE to patch-token q/k: `x·cos + rotate_half_interleaved(x)·sin`, computed in f32.
    /// `x`: `[B, heads, grid², head_dim]` (CLS token already sliced off by the caller).
    pub fn apply(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let orig = x.dtype();
        let xf = x.to_dtype(DType::F32)?;
        let (n, hd) = (self.cos.dim(0)?, self.cos.dim(1)?);
        let cos = self.cos.reshape((1, 1, n, hd))?;
        let sin = self.sin.reshape((1, 1, n, hd))?;
        let rot = rotate_half_interleaved(&xf)?;
        let out = (xf.broadcast_mul(&cos)? + rot.broadcast_mul(&sin)?)?;
        out.to_dtype(orig)
    }
}

/// Interleaved `rotate_half`: reshape the last dim `(.., d/2, 2)`, then rebuild `(-x_odd, x_even)`.
/// Result: `out[2i] = -x[2i+1]`, `out[2i+1] = x[2i]`.
fn rotate_half_interleaved(x: &Tensor) -> candle_core::Result<Tensor> {
    let dims = x.dims().to_vec();
    let last = *dims.last().unwrap();
    let mut pair = dims.clone();
    *pair.last_mut().unwrap() = last / 2;
    pair.push(2);
    let xr = x.reshape(pair)?; // [.., d/2, 2]
    let ax = xr.rank() - 1;
    let even = xr.narrow(ax, 0, 1)?; // [.., d/2, 1]
    let odd = xr.narrow(ax, 1, 1)?;
    let rotated = Tensor::cat(&[&odd.neg()?, &even], ax)?; // (-odd, even)
    rotated.reshape(dims)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The interleaved rotate swaps adjacent pairs with a sign: `[a,b,c,d] → [-b,a,-d,c]`.
    #[test]
    fn rotate_half_interleaved_swaps_pairs() {
        let dev = Device::Cpu;
        let x = Tensor::from_vec(vec![1f32, 2., 3., 4.], (1, 1, 1, 4), &dev).unwrap();
        let r = rotate_half_interleaved(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(r, vec![-2.0, 1.0, -4.0, 3.0]);
    }

    /// The cos/sin tables have the expected `[grid², head_dim]` shape and the first position (t=0) is
    /// all `cos=1, sin=0` (no rotation at the origin patch).
    #[test]
    fn rope_tables_shape_and_origin() {
        let dev = Device::Cpu;
        let r = VisionRope::build(64, 24, 16, 10000.0, &dev).unwrap();
        assert_eq!(r.cos.dims(), &[576, 64]);
        let cos0 = r
            .cos
            .narrow(0, 0, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let sin0 = r
            .sin
            .narrow(0, 0, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(cos0.iter().all(|&v| (v - 1.0).abs() < 1e-6));
        assert!(sin0.iter().all(|&v| v.abs() < 1e-6));
    }
}
