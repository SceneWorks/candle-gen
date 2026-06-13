//! Lens complex axial RoPE (`LensEmbedRope`, `theta=10000`, `axes_dim=(8,28,28)`, `scale_rope=True`).
//! Candle port of `mlx-gen-lens/src/dit/rope.rs`, sharing the construction of
//! `candle-gen-qwen-image`'s `QwenRope` (only the axes differ: `8 + 28 + 28 = 64 = head_dim` →
//! `4 + 14 + 14 = 32` complex pairs). Produces **interleaved** cos/sin tables for the image and text
//! streams; `freqs_cis = polar(1, angle) = cos + i·sin`, so the reference's complex `view_as_complex`
//! apply is reproduced by candle's interleaved `rope_i` in [`super::attention`].
//!
//! Image stream: frame axis at positions `0..frame`, height/width at **centered** positions
//! `hi - (h - h/2)` / `wi - (w - w/2)` (`scale_rope`). Text stream: a single scalar position
//! `max(h/2, w/2) + t` across all 32 pair-frequencies. Angles are computed **host-side** in f32.

use candle_gen::candle_core::{DType, Device, Result, Tensor};
use candle_gen::candle_nn::rotary_emb::rope_i;

/// `(img_cos, img_sin, txt_cos, txt_sin)`: image tables `[frame·h·w, head_dim/2]`, text tables
/// `[txt_seq, head_dim/2]`.
pub type RopeTables = (Tensor, Tensor, Tensor, Tensor);

/// Lens 3-axis (frame/height/width) RoPE table builder.
pub struct LensRope3d {
    theta: f32,
    axes_dim: [usize; 3],
    half: usize,
}

impl LensRope3d {
    pub fn new(theta: f32, axes_dim: [usize; 3]) -> Self {
        Self {
            theta,
            axes_dim,
            half: axes_dim.iter().sum::<usize>() / 2,
        }
    }

    /// The Lens default: θ=10000, axes `(8, 28, 28)` (Σ = 64 = head_dim, Σ/2 = 32 pairs).
    pub fn lens() -> Self {
        Self::new(10000.0, [8, 28, 28])
    }

    /// Per-axis frequencies `ω_d[k] = theta^{-(2k)/d}`, `k ∈ 0..d/2`.
    fn omega_axis(&self, dim: usize) -> Vec<f32> {
        (0..dim / 2)
            .map(|k| 1.0f32 / self.theta.powf((2 * k) as f32 / dim as f32))
            .collect()
    }

    /// Build the `(img_cos, img_sin, txt_cos, txt_sin)` tables for a single `(frame, h, w)` image grid
    /// and a text sequence of `txt_seq` tokens.
    pub fn forward(
        &self,
        frame: usize,
        h: usize,
        w: usize,
        txt_seq: usize,
        device: &Device,
    ) -> Result<RopeTables> {
        let o0 = self.omega_axis(self.axes_dim[0]);
        let o1 = self.omega_axis(self.axes_dim[1]);
        let o2 = self.omega_axis(self.axes_dim[2]);
        let half = self.half; // 4 + 14 + 14 = 32

        let total = frame * h * w;
        let mut img_cos = vec![0f32; total * half];
        let mut img_sin = vec![0f32; total * half];
        // height/width centered positions (scale_rope): hp ∈ [-(h - h/2), …, h/2 - 1].
        let h_off = (h - h / 2) as i64;
        let w_off = (w - w / 2) as i64;
        for f in 0..frame {
            for hi in 0..h {
                let hp = hi as i64 - h_off;
                for wi in 0..w {
                    let wp = wi as i64 - w_off;
                    let row = (f * h * w + hi * w + wi) * half;
                    let mut j = 0;
                    for &fr in &o0 {
                        let a = f as f32 * fr;
                        img_cos[row + j] = a.cos();
                        img_sin[row + j] = a.sin();
                        j += 1;
                    }
                    for &fr in &o1 {
                        let a = hp as f32 * fr;
                        img_cos[row + j] = a.cos();
                        img_sin[row + j] = a.sin();
                        j += 1;
                    }
                    for &fr in &o2 {
                        let a = wp as f32 * fr;
                        img_cos[row + j] = a.cos();
                        img_sin[row + j] = a.sin();
                        j += 1;
                    }
                }
            }
        }

        // Text stream: scalar position max(h/2, w/2) + t across all 32 pair-frequencies.
        let txt_base = (h / 2).max(w / 2) as i64;
        let all_omega: Vec<f32> = o0.iter().chain(&o1).chain(&o2).copied().collect();
        let mut txt_cos = vec![0f32; txt_seq * half];
        let mut txt_sin = vec![0f32; txt_seq * half];
        for t in 0..txt_seq {
            let p = (txt_base + t as i64) as f32;
            let row = t * half;
            for (j, &fr) in all_omega.iter().enumerate() {
                let a = p * fr;
                txt_cos[row + j] = a.cos();
                txt_sin[row + j] = a.sin();
            }
        }

        Ok((
            Tensor::from_vec(img_cos, (total, half), device)?,
            Tensor::from_vec(img_sin, (total, half), device)?,
            Tensor::from_vec(txt_cos, (txt_seq, half), device)?,
            Tensor::from_vec(txt_sin, (txt_seq, half), device)?,
        ))
    }
}

/// Interleaved complex RoPE on `x` `[B, heads, seq, head_dim]` with `cos`/`sin` `[seq, head_dim/2]`.
/// Pairs `(x_2i, x_2i+1)` are rotated by `(cos_i, sin_i)` (candle's `rope_i`, the same interleaved
/// convention the qwen-image port validated). The rotation runs in f32 and is cast back to `x`'s dtype.
pub fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let dtype = x.dtype();
    let xf = x.to_dtype(DType::F32)?.contiguous()?;
    rope_i(&xf, cos, sin)?.to_dtype(dtype)
}
