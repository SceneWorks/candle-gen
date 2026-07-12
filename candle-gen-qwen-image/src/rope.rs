//! Qwen-Image's **3-axis (frame, height, width) interleaved RoPE** for the MMDiT. Port of
//! `mlx-gen-qwen-image`'s `transformer/rope.rs`. Each axis (`axes_dim = [16, 56, 56]`) contributes
//! `dim/2` frequencies → `8 + 28 + 28 = 64` per token (= `head_dim/2`). θ = 10000, `scale_rope` (the
//! height/width positions are centered). Frequencies and angles are computed **host-side** in f32.
//!
//! - **Image tokens** at grid `(h, w)`: the frame axis uses position 0 (single image), height/width
//!   use **centered** positions `h - (latent_h - latent_h/2)` / `w - (latent_w - latent_w/2)`.
//! - **Text tokens** at index `t`: a single scalar position `txt_base + t`
//!   (`txt_base = max(latent_h/2, latent_w/2)`) applied across **all 64** frequencies.
//!
//! Application is **interleaved** (lanes `2i`/`2i+1` are the real/imag pair), via candle's `rope_i`.

use candle_gen::candle_core::{DType, Device, Result, Tensor};
use candle_gen::candle_nn::rotary_emb::rope_i;

use crate::config::TransformerConfig;

pub struct QwenRope {
    theta: f32,
    axes_dim: [usize; 3],
    half: usize,
}

impl QwenRope {
    pub fn new(cfg: &TransformerConfig) -> Self {
        Self {
            theta: cfg.rope_theta,
            axes_dim: cfg.axes_dim,
            half: cfg.axes_dim.iter().sum::<usize>() / 2,
        }
    }

    /// The 64-wide concatenated frequency vector `[ω_frame(8), ω_h(28), ω_w(28)]`,
    /// `ω_d[k] = theta^{-(2k)/d}`.
    fn omega(&self) -> Vec<f32> {
        let mut all = Vec::with_capacity(self.half);
        for &dim in &self.axes_dim {
            for k in 0..dim / 2 {
                all.push(1.0f32 / self.theta.powf((2 * k) as f32 / dim as f32));
            }
        }
        all
    }

    /// Image-token `(cos, sin)` `[lat_h·lat_w, 64]` (row-major over the grid). The single-grid (T2I)
    /// case of [`img_cos_sin_multi`](Self::img_cos_sin_multi) — grid index 0 → frame position 0.
    pub fn img_cos_sin(
        &self,
        lat_h: usize,
        lat_w: usize,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        self.img_cos_sin_multi(&[(lat_h, lat_w)], device)
    }

    /// Image-token `(cos, sin)` over one-or-more grids (the Qwen-Image-Edit dual-latent path): the
    /// noise grid (index 0) followed by each reference grid (index 1, 2, …) in sequence order. The
    /// grid **index** drives the frame-axis position (so a reference's frame freqs differ from the
    /// noise's), while height/width stay per-grid **centered**. Concatenated `[Σ h_i·w_i, 64]`.
    pub fn img_cos_sin_multi(
        &self,
        grids: &[(usize, usize)],
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let omega = self.omega();
        let (n_f, n_h) = (self.axes_dim[0] / 2, self.axes_dim[1] / 2); // 8, 28
        let total: usize = grids.iter().map(|(h, w)| h * w).sum();
        let mut cos = vec![0f32; total * self.half];
        let mut sin = vec![0f32; total * self.half];
        let mut off = 0usize;
        for (idx, &(lat_h, lat_w)) in grids.iter().enumerate() {
            let h_center = (lat_h - lat_h / 2) as i64;
            let w_center = (lat_w - lat_w / 2) as i64;
            for h in 0..lat_h {
                for w in 0..lat_w {
                    let row = off + h * lat_w + w;
                    let hp = h as i64 - h_center;
                    let wp = w as i64 - w_center;
                    for (j, &om) in omega.iter().enumerate() {
                        // frame band → grid index, height band → hp, width band → wp.
                        let pos = if j < n_f {
                            idx as i64
                        } else if j < n_f + n_h {
                            hp
                        } else {
                            wp
                        } as f32;
                        let a = pos * om;
                        cos[row * self.half + j] = a.cos();
                        sin[row * self.half + j] = a.sin();
                    }
                }
            }
            off += lat_h * lat_w;
        }
        Ok((
            Tensor::from_vec(cos, (total, self.half), device)?,
            Tensor::from_vec(sin, (total, self.half), device)?,
        ))
    }

    /// Text-token `(cos, sin)` `[txt_seq, 64]`: scalar position `txt_base + t` across all freqs. The
    /// single-grid case of [`txt_cos_sin_multi`](Self::txt_cos_sin_multi).
    pub fn txt_cos_sin(
        &self,
        txt_seq: usize,
        lat_h: usize,
        lat_w: usize,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        self.txt_cos_sin_multi(txt_seq, &[(lat_h, lat_w)], device)
    }

    /// Text-token `(cos, sin)` for the dual-latent path: `txt_base = max_i(max(h_i/2, w_i/2))` over
    /// every grid, then position `txt_base + t` across all freqs.
    pub fn txt_cos_sin_multi(
        &self,
        txt_seq: usize,
        grids: &[(usize, usize)],
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let omega = self.omega();
        let txt_base = grids
            .iter()
            .map(|(h, w)| (h / 2).max(w / 2))
            .max()
            .unwrap_or(0) as i64;
        let mut cos = vec![0f32; txt_seq * self.half];
        let mut sin = vec![0f32; txt_seq * self.half];
        for t in 0..txt_seq {
            let pos = (txt_base + t as i64) as f32;
            for (j, &om) in omega.iter().enumerate() {
                let a = pos * om;
                cos[t * self.half + j] = a.cos();
                sin[t * self.half + j] = a.sin();
            }
        }
        Ok((
            Tensor::from_vec(cos, (txt_seq, self.half), device)?,
            Tensor::from_vec(sin, (txt_seq, self.half), device)?,
        ))
    }
}

/// Per-render RoPE-table cache (sc-8992 / F-012). The image + text `(cos, sin)` tables depend only on
/// the fixed token grids (`[(lat_h, lat_w), …]`) and `txt_seq` — not on σ / the current latent — so
/// they are identical across every denoise step (×2 under CFG, and the control branch re-derives the
/// same grid each step). Cache them keyed on `(grids, txt_seq)` and rebuild only when the geometry
/// changes; hits Arc-clone the stored handles. Byte-identical to recomputing.
///
/// `Mutex` (not `RefCell`) because the transformers are shared as `Arc<…>` and must stay `Send + Sync`.
pub struct RopeCache {
    slot: std::sync::Mutex<Option<RopeCacheEntry>>,
}

struct RopeCacheEntry {
    grids: Vec<(usize, usize)>,
    txt_seq: usize,
    img_cos: Tensor,
    img_sin: Tensor,
    txt_cos: Tensor,
    txt_sin: Tensor,
}

impl RopeCache {
    pub fn new() -> Self {
        Self {
            slot: std::sync::Mutex::new(None),
        }
    }

    /// Build (or reuse) the image + text RoPE tables for `grids` / `txt_seq`. Recomputed only when the
    /// geometry changes vs the cached entry. Construction is identical to calling the `QwenRope`
    /// builders inline, so every step is byte-identical.
    #[allow(clippy::type_complexity)]
    pub fn tables(
        &self,
        rope: &QwenRope,
        grids: &[(usize, usize)],
        txt_seq: usize,
        device: &Device,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let mut guard = candle_gen::lock_recover(&self.slot);
        if let Some(c) = guard.as_ref() {
            if c.grids.as_slice() == grids && c.txt_seq == txt_seq {
                return Ok((
                    c.img_cos.clone(),
                    c.img_sin.clone(),
                    c.txt_cos.clone(),
                    c.txt_sin.clone(),
                ));
            }
        }
        let (img_cos, img_sin) = rope.img_cos_sin_multi(grids, device)?;
        let (txt_cos, txt_sin) = rope.txt_cos_sin_multi(txt_seq, grids, device)?;
        *guard = Some(RopeCacheEntry {
            grids: grids.to_vec(),
            txt_seq,
            img_cos: img_cos.clone(),
            img_sin: img_sin.clone(),
            txt_cos: txt_cos.clone(),
            txt_sin: txt_sin.clone(),
        });
        Ok((img_cos, img_sin, txt_cos, txt_sin))
    }
}

impl Default for RopeCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Apply interleaved RoPE to `x` `[B, H, S, head_dim]` with `cos`/`sin` `[S, head_dim/2]`.
pub fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let dtype = x.dtype();
    let xf = x.to_dtype(DType::F32)?.contiguous()?;
    rope_i(&xf, cos, sin)?.to_dtype(dtype)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn omega_is_64_wide_with_band_layout() {
        let r = QwenRope::new(&TransformerConfig::qwen_image());
        assert_eq!(r.half, 64);
        let om = r.omega();
        assert_eq!(om.len(), 64);
        // First freq of each band is theta^0 = 1.
        assert!((om[0] - 1.0).abs() < 1e-6, "frame band base");
        assert!((om[8] - 1.0).abs() < 1e-6, "height band base");
        assert!((om[36] - 1.0).abs() < 1e-6, "width band base");
    }

    #[test]
    fn img_frame_axis_is_zero_angle() {
        let r = QwenRope::new(&TransformerConfig::qwen_image());
        let (cos, sin) = r.img_cos_sin(4, 4, &Device::Cpu).unwrap();
        assert_eq!(cos.dims(), &[16, 64]);
        // Frame band (first 8 lanes) has position 0 → cos 1, sin 0 for every token.
        let cv = cos.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let sv = sin.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for tok in 0..16 {
            for j in 0..8 {
                assert!((cv[tok * 64 + j] - 1.0).abs() < 1e-6);
                assert!(sv[tok * 64 + j].abs() < 1e-6);
            }
        }
    }

    #[test]
    fn apply_rope_at_zero_is_identity() {
        let r = QwenRope::new(&TransformerConfig::qwen_image());
        // txt position 0 with lat 2x2 → txt_base = max(1,1)=1, so not zero; build an explicit zero table.
        let cos = Tensor::ones((3, 64), DType::F32, &Device::Cpu).unwrap();
        let sin = Tensor::zeros((3, 64), DType::F32, &Device::Cpu).unwrap();
        let x = Tensor::arange(0f32, 3.0 * 128.0, &Device::Cpu)
            .unwrap()
            .reshape((1, 1, 3, 128))
            .unwrap();
        let y = apply_rope(&x, &cos, &sin).unwrap();
        let xv = x.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let yv = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (a, b) in xv.iter().zip(&yv) {
            assert!((a - b).abs() < 1e-5);
        }
        let _ = &r;
    }
}
