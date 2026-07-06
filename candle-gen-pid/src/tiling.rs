//! Spatial tiling for the PiD pixel-space decode (sc-10087, candle mirror of `mlx-gen-pid::tiling`).
//!
//! At large output resolutions (≥~6144²) a single whole-image `PidNet::forward` overflows VRAM; on
//! Windows/NVIDIA the driver's default-on CUDA System-Memory-Fallback turns that OOM into silent paging
//! to host RAM → throughput collapses, the GPU idles, the decode never finishes. We attack the dominant
//! peak term directly: split the pixel grid into overlapping tiles and run the **velocity** forward one
//! tile at a time, feather-blending the per-tile predictions into a full-resolution `v`.
//!
//! ## Fidelity note — an *approximation*, not a transparent split
//! The `PixDiT` is globally self-attentive at both stages, so tiling drops cross-tile attention within a
//! step. PiD's global structure comes from the small, whole **LQ latent** conditioning (each tile gets
//! its aligned latent slice), so the pixel diffusion is mostly local detail synthesis — overlap + feather
//! blend is near-seamless. Validated on Metal by the sc-10087 real-weight A/B (tiled vs whole at
//! 1024-native → 4096²: no measurable seam, PSNR 34.75 dB); this candle path mirrors that engine exactly.
//!
//! ## Design (mirror of the MLX port)
//! - The 4-step SDE loop stays **whole-image** ([`crate::sampler::Sampler::run_tiled`]): `x`, the seeded
//!   `noise`, and each ε remain full-resolution, so the sampler math + the launch-portable RNG sequence
//!   (sc-3673) are byte-for-byte unchanged. Only the per-step forward is tiled.
//! - Tiles align to `F = H / zH` (the pixel→latent factor, 32 for the 4× qwenimage student), a multiple
//!   of `patch_size`, so each tile's LQ slice + patch grid are integer and consistent.
//! - candle is eager, so (unlike MLX) no per-tile materialization call is needed — each tile's activations
//!   drop at end of iteration, bounding peak to (resident full-res buffers) + (one tile's forward).

use candle_gen::candle_core::{Device, Tensor};
use candle_gen::Result;

use crate::lq::PidNet;

/// A spatial tile's pixel-space extent `[y0,y1) × [x0,x1)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpatialTile {
    pub y0: i32,
    pub y1: i32,
    pub x0: i32,
    pub x1: i32,
}

/// Tile a length-`n` axis into full-`tile`-size windows at stride `tile-overlap`; the final window's
/// start is clamped so it stays full size and ends at `n`. One window covers `n <= tile`. Mirrors
/// `mlx-gen-pid::tiling::tile_axis` / seedvr2.
fn tile_axis(n: i32, tile: i32, overlap: i32) -> Vec<(i32, i32)> {
    if n <= tile {
        return vec![(0, n)];
    }
    let stride = (tile - overlap).max(1);
    let mut out = Vec::new();
    let mut s = 0;
    loop {
        let e = (s + tile).min(n);
        let start = (e - tile).max(0);
        if out.last() != Some(&(start, e)) {
            out.push((start, e));
        }
        if e >= n {
            break;
        }
        s += stride;
    }
    out
}

/// The grid of overlapping tiles covering an `h × w` grid. `tile`/`overlap` are rounded **down** to a
/// multiple of `align` (the pixel→latent factor); `tile` is floored to at least `align`.
pub fn plan_tiles(h: i32, w: i32, tile: i32, overlap: i32, align: i32) -> Vec<SpatialTile> {
    let align = align.max(1);
    let tile = (tile / align * align).max(align);
    let overlap = (overlap / align * align).clamp(0, tile - align);
    let ys = tile_axis(h, tile, overlap);
    let xs = tile_axis(w, tile, overlap);
    let mut out = Vec::with_capacity(ys.len() * xs.len());
    for &(y0, y1) in &ys {
        for &(x0, x1) in &xs {
            out.push(SpatialTile { y0, y1, x0, x1 });
        }
    }
    out
}

/// Linear taper along one axis: ramp up over the first `overlap` px when `fade_start`, down over the last
/// `overlap` when `fade_end`, 1 in between; floored positive. Mirrors seedvr2 / the MLX port.
fn axis_ramp(len: i32, fade_start: bool, fade_end: bool, overlap: i32) -> Vec<f32> {
    let ov = overlap.max(1);
    (0..len)
        .map(|i| {
            let mut w = 1.0f32;
            if fade_start && i < ov {
                w = w.min((i as f32 + 1.0) / (ov as f32 + 1.0));
            }
            if fade_end && i >= len - ov {
                w = w.min((len - i) as f32 / (ov as f32 + 1.0));
            }
            w.max(1e-4)
        })
        .collect()
}

/// Separable per-pixel feather weights `(th·tw)` for a tile, tapering to ~0 over `overlap` px on each edge
/// abutting a neighbor and staying 1 at outer image edges. `w = ry·rx`.
fn feather_weight(
    th: i32,
    tw: i32,
    fade_top: bool,
    fade_bottom: bool,
    fade_left: bool,
    fade_right: bool,
    overlap: i32,
) -> Vec<f32> {
    let ry = axis_ramp(th, fade_top, fade_bottom, overlap);
    let rx = axis_ramp(tw, fade_left, fade_right, overlap);
    let mut out = vec![0f32; (th * tw) as usize];
    for y in 0..th as usize {
        for x in 0..tw as usize {
            out[y * tw as usize + x] = ry[y] * rx[x];
        }
    }
    out
}

/// Slice `a`'s axis `axis` to `[start, end)`.
fn slice_axis(a: &Tensor, axis: usize, start: i32, end: i32) -> Result<Tensor> {
    Ok(a.narrow(axis, start as usize, (end - start) as usize)?)
}

/// One tiled **velocity** forward: compute `v = net(x, t, …)` over the whole `[B,3,H,W]` grid by running
/// the net on overlapping pixel tiles (each with its aligned `lq_latent` slice) and feather-blending the
/// per-tile predictions. `overlap` is the feather width (px). candle is eager, so each tile's activations
/// drop at end of iteration → peak is bounded to the resident buffers + one tile's forward.
#[allow(clippy::too_many_arguments)]
pub fn forward_tiled(
    net: &PidNet,
    x: &Tensor,
    t_scaled: &Tensor,
    caption: &Tensor,
    lq_latent: &Tensor,
    sigma: &Tensor,
    tile: i32,
    overlap: i32,
) -> Result<Tensor> {
    let (_, _, h, w) = x.dims4()?;
    let (h, w) = (h as i32, w as i32);
    let z_h = lq_latent.dim(2)? as i32;
    let f = (h / z_h).max(1);
    let plan = plan_tiles(h, w, tile, overlap, f);

    // Whole-image single tile → the plain forward (exact, no blend).
    if plan.len() == 1 {
        return net.forward(x, t_scaled, caption, lq_latent, sigma);
    }

    let device: &Device = x.device();
    let mut acc: Option<Tensor> = None; // [B,3,H,W]
    let mut wsum: Option<Tensor> = None; // [1,1,H,W]
    for tl in &plan {
        let (th, tw) = (tl.y1 - tl.y0, tl.x1 - tl.x0);
        let x_tile = slice_axis(&slice_axis(x, 2, tl.y0, tl.y1)?, 3, tl.x0, tl.x1)?;
        let lq_tile = slice_axis(
            &slice_axis(lq_latent, 2, tl.y0 / f, tl.y1 / f)?,
            3,
            tl.x0 / f,
            tl.x1 / f,
        )?;
        let v_tile = net.forward(&x_tile, t_scaled, caption, &lq_tile, sigma)?; // [B,3,th,tw]

        let wvec = feather_weight(th, tw, tl.y0 > 0, tl.y1 < h, tl.x0 > 0, tl.x1 < w, overlap);
        let weight = Tensor::from_vec(wvec, (1, 1, th as usize, tw as usize), device)?;
        // place the weighted tile at (y0,x0) in a full-res canvas via zero-pad on H then W.
        let wv = v_tile
            .broadcast_mul(&weight)?
            .pad_with_zeros(2, tl.y0 as usize, (h - tl.y1) as usize)?
            .pad_with_zeros(3, tl.x0 as usize, (w - tl.x1) as usize)?;
        let wp = weight
            .pad_with_zeros(2, tl.y0 as usize, (h - tl.y1) as usize)?
            .pad_with_zeros(3, tl.x0 as usize, (w - tl.x1) as usize)?;
        acc = Some(match acc {
            Some(a) => (a + wv)?,
            None => wv,
        });
        wsum = Some(match wsum {
            Some(a) => (a + wp)?,
            None => wp,
        });
    }
    Ok(acc
        .expect("≥1 tile")
        .broadcast_div(&wsum.expect("≥1 tile"))?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_axis_covers_without_gaps_and_stays_full_size() {
        let ax = tile_axis(4096, 2048, 256);
        assert_eq!(ax.first().unwrap().0, 0);
        assert_eq!(ax.last().unwrap().1, 4096);
        for &(s, e) in &ax {
            assert_eq!(e - s, 2048);
        }
        for pair in ax.windows(2) {
            assert!(pair[1].0 < pair[0].1, "windows overlap");
        }
    }

    #[test]
    fn single_window_when_grid_fits() {
        assert_eq!(tile_axis(2048, 2048, 256), vec![(0, 2048)]);
        assert_eq!(tile_axis(1024, 2048, 256), vec![(0, 1024)]);
    }

    #[test]
    fn plan_tiles_aligns_to_factor() {
        let plan = plan_tiles(4096, 4096, 2000, 250, 32);
        assert!(!plan.is_empty());
        for t in &plan {
            for v in [t.y0, t.y1, t.x0, t.x1] {
                assert_eq!(v % 32, 0, "tile edge {v} aligned to 32");
            }
        }
    }

    #[test]
    fn whole_image_is_one_tile() {
        assert_eq!(plan_tiles(4096, 4096, 8192, 256, 32).len(), 1);
    }

    #[test]
    fn feather_tapers_only_abutting_edges() {
        let w = feather_weight(64, 64, true, true, true, true, 16);
        assert!(w[0] < 0.1);
        assert!(w[(32 * 64 + 32) as usize] > 0.9);
        let e = feather_weight(64, 64, false, true, false, true, 16);
        assert!(e[0] > 0.9);
    }
}
