//! FLUX.2's **4-axis RoPE** over `(t, h, w, layer)` position ids. Port of `mlx-gen-flux2`'s
//! `pos_embed.rs`. Each of the 4 axes (`axes_dim = [32,32,32,32]`) contributes `dim/2 = 16`
//! frequencies; the four 16-wide blocks concatenate into a `head_dim/2 = 64`-wide `(cos, sin)`
//! table. Frequencies `ω[j] = theta^{-(2j)/dim}` (θ = 2000) are computed **host-side** in f32 (no
//! MLX-op parity contract for FLUX.2), then the table is materialized on-device.
//!
//! Application is **interleaved** (GPT-J style): lanes `2i` / `2i+1` of each head are the real /
//! imaginary parts rotated together by `(cos[i], sin[i])` — `out0 = r·cos − i·sin`,
//! `out1 = i·cos + r·cos`. candle's [`candle_nn::rotary_emb::rope_i`] implements exactly this
//! interleaved rotation for `cos`/`sin` of width `head_dim/2`, so it is reused for the apply.

use candle_gen::candle_core::{Device, Result, Tensor};
use candle_gen::candle_nn::rotary_emb::rope_i;

use crate::config::Flux2Config;

/// Builds the FLUX.2 4-axis RoPE `(cos, sin)` tables and applies them to q/k.
pub struct Flux2PosEmbed {
    theta: f32,
    axes_dim: [usize; 4],
    /// `sum(axes_dim) / 2` — the cos/sin table width (= `head_dim / 2`, 64 for klein).
    half: usize,
}

impl Flux2PosEmbed {
    pub fn new(cfg: &Flux2Config) -> Self {
        Self {
            theta: cfg.rope_theta,
            axes_dim: cfg.axes_dim,
            half: cfg.axes_dim.iter().sum::<usize>() / 2,
        }
    }

    /// The cos/sin table width (`head_dim / 2`).
    pub fn half(&self) -> usize {
        self.half
    }

    /// Build the `(cos, sin)` tables `[seq, half]` (f32) for the ordered list of 4-axis position
    /// `ids` (each `[t, h, w, layer]`). Computed host-side, then moved to `device`.
    pub fn cos_sin(&self, ids: &[[i64; 4]], device: &Device) -> Result<(Tensor, Tensor)> {
        let seq = ids.len();
        let mut cos = vec![0f32; seq * self.half];
        let mut sin = vec![0f32; seq * self.half];
        for (s, id) in ids.iter().enumerate() {
            let mut col = 0usize;
            for (axis, &dim) in self.axes_dim.iter().enumerate() {
                let pos = id[axis] as f32;
                let half_axis = dim / 2;
                for j in 0..half_axis {
                    let omega = 1.0f32 / self.theta.powf((2 * j) as f32 / dim as f32);
                    let angle = pos * omega;
                    cos[s * self.half + col] = angle.cos();
                    sin[s * self.half + col] = angle.sin();
                    col += 1;
                }
            }
        }
        let cos = Tensor::from_vec(cos, (seq, self.half), device)?;
        let sin = Tensor::from_vec(sin, (seq, self.half), device)?;
        Ok((cos, sin))
    }

    /// Apply the interleaved RoPE to `x` `[B, H, S, head_dim]` with `cos`/`sin` `[S, head_dim/2]`
    /// (f32, contiguous). Returns the rotated tensor in `x`'s dtype.
    pub fn apply(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        // rope_i wants f32 cos/sin and a contiguous x; do the rotation in f32 then cast back.
        let xf = x
            .to_dtype(candle_gen::candle_core::DType::F32)?
            .contiguous()?;
        let rotated = rope_i(&xf, cos, sin)?;
        rotated.to_dtype(dtype)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cos_sin_shapes_and_known_values() {
        let cfg = Flux2Config::klein_9b();
        let pe = Flux2PosEmbed::new(&cfg);
        assert_eq!(pe.half(), 64);
        // A position-0 id: every angle is 0 → cos 1, sin 0 across all 64 lanes.
        let (cos, sin) = pe.cos_sin(&[[0, 0, 0, 0]], &Device::Cpu).unwrap();
        assert_eq!(cos.dims(), &[1, 64]);
        let cos_v = cos.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let sin_v = sin.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for c in &cos_v {
            assert!((c - 1.0).abs() < 1e-6);
        }
        for s in &sin_v {
            assert!(s.abs() < 1e-6);
        }
    }

    #[test]
    fn first_lane_of_each_axis_uses_base_frequency() {
        // For axis position p and j=0, ω=1, so angle == p; the first lane of each 16-wide axis block
        // (cols 0,16,32,48) reflects that axis's position directly.
        let cfg = Flux2Config::klein_9b();
        let pe = Flux2PosEmbed::new(&cfg);
        let (cos, _) = pe.cos_sin(&[[1, 2, 3, 4]], &Device::Cpu).unwrap();
        let v = cos.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((v[0] - 1.0f32.cos()).abs() < 1e-5, "t axis @ pos 1");
        assert!((v[16] - 2.0f32.cos()).abs() < 1e-5, "h axis @ pos 2");
        assert!((v[32] - 3.0f32.cos()).abs() < 1e-5, "w axis @ pos 3");
        assert!((v[48] - 4.0f32.cos()).abs() < 1e-5, "layer axis @ pos 4");
    }

    /// Interleaved RoPE at position 0 is the identity (cos 1, sin 0).
    #[test]
    fn apply_at_pos_zero_is_identity() {
        let cfg = Flux2Config::klein_9b();
        let pe = Flux2PosEmbed::new(&cfg);
        let (cos, sin) = pe.cos_sin(&[[0, 0, 0, 0]], &Device::Cpu).unwrap();
        // x: [B=1, H=1, S=1, D=128]
        let x = Tensor::arange(0f32, 128f32, &Device::Cpu)
            .unwrap()
            .reshape((1, 1, 1, 128))
            .unwrap();
        let y = Flux2PosEmbed::apply(&x, &cos, &sin).unwrap();
        let xv = x.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let yv = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (a, b) in xv.iter().zip(&yv) {
            assert!((a - b).abs() < 1e-5);
        }
    }
}
