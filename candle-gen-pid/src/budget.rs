//! Memory budget for the PiD super-resolving decode (F-013 sc-9095, tiling sc-10087 — candle mirror).
//!
//! PiD denoises directly in **high-resolution pixel space**: `noise`/`x`/each per-step ε is a full
//! `[B,3,H,W]` f32 tensor at the *super-resolved output* resolution, and on top of those the patch
//! stream runs `(H/patch)·(W/patch)` tokens through the MMDiT blocks. A `max_size`-legal 1536²/2048²
//! request super-resolves 4× to 6144²/8192², whose whole-image forward exceeds VRAM on a smaller GPU. On
//! Windows/NVIDIA the driver's default-on CUDA System-Memory-Fallback turns that OOM into **silent**
//! paging to host RAM → the GPU idles and the decode never finishes (sc-10087).
//!
//! The fix is [spatial tiling](crate::tiling): run the per-step velocity forward on overlapping tiles.
//! This module sizes it — splitting the peak into the two terms tiling treats differently:
//!   * **resident full-res buffers** (`x` + `noise` + ε + blend accumulators, at the *output* resolution,
//!     alive across the whole decode) — tiling does **not** shrink these;
//!   * **per-forward activations** (pixel-space transients + patch-stream working set, at the *forward*
//!     grid) — tiling shrinks these to one tile.
//!
//! [`plan_tile_edge`] picks the largest tile whose per-forward term fits `safe_gib`; [`guard`] refuses
//! only when even the resident buffers plus a minimum tile won't fit. Unlike the MLX mirror there is no
//! command-buffer watchdog on CUDA, so untiling keys on memory alone.

use candle_gen::{CandleError, Result};

use crate::config::PidConfig;

/// 1 GiB in bytes.
const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// Bytes per f32 element (PiD runs its pixel/patch tensors in f32).
const F32_BYTES: f64 = 4.0;

/// Full-resolution `[B,3,H,W]` f32 tensors held **resident** across the whole decode, at the *output*
/// resolution — the running `x`, the seeded `noise`, the per-step SDE ε (3 for the 4-step distill), and
/// (under tiling) the `acc` blend accumulator. `6` is deliberately conservative. **Tiling does not reduce
/// this term** — it is the floor and the reason the guard still exists as a backstop.
const RESIDENT_FULL_TENSORS: f64 = 6.0;

/// Pixel-space transient `[B,3,h,w]` tensors live during a **single forward** at the *forward* grid —
/// shrinks to one tile under tiling.
const FWD_PIXEL_TENSORS: f64 = 3.0;

/// Multiplier on the single-forward patch-token working set (`tokens · hidden`) — the several block
/// activation temporaries. Scales with the *forward* grid (tiled → one tile).
const PATCH_ACT_BLOCK_MULT: f64 = 8.0;

/// Tile edges are chosen as multiples of this (also the feather/overlap granularity) — a multiple of the
/// pixel→latent factor (32 for the 4× qwenimage student) and of `patch_size` (16).
pub const TILE_ALIGN: i32 = 512;

/// Smallest tile edge the planner will drop to; below this the approximation degrades and [`guard`]
/// refuses.
pub const MIN_TILE_EDGE: i32 = 1024;

/// Default feather overlap (px) between tiles — the seedvr2 default (sc-5201) / the sc-10087 A/B (2048 px
/// tiles / 256 px overlap → no measurable seam).
pub const DEFAULT_TILE_OVERLAP: i32 = 256;

/// Preferred tile edge when tiling is engaged on a GPU with memory to spare (the sc-10087 A/B-validated
/// 2048 px tile; attention is superlinear, so many small forwards beat few large ones). Shrunk only when
/// memory forces it (see [`plan_tile_edge`]).
pub const PREFERRED_TILE_EDGE: i32 = 2048;

/// GiB of the resident full-res buffers for a `[b,3,th,tw]` output — the term tiling can't shrink.
fn resident_full_gib(b: f64, th: f64, tw: f64) -> f64 {
    RESIDENT_FULL_TENSORS * b * 3.0 * th * tw * F32_BYTES / GIB
}

/// GiB of a single forward's activations over an `fh × fw` grid (the whole output, or one tile).
fn forward_activations_gib(b: f64, fh: f64, fw: f64, patch_size: i32, hidden: i32) -> f64 {
    let pixels = FWD_PIXEL_TENSORS * b * 3.0 * fh * fw * F32_BYTES;
    let tokens = (fh / patch_size as f64) * (fw / patch_size as f64);
    let patch_act = PATCH_ACT_BLOCK_MULT * b * tokens * hidden as f64 * F32_BYTES;
    (pixels + patch_act) / GIB
}

/// Estimated concurrent peak (GiB) of a **whole-image** (untiled) PiD decode producing `[b,3,th,tw]`:
/// the resident full-res buffers plus a single forward over the whole output grid. Pure.
pub fn estimated_decode_peak_gib(b: i32, th: i32, tw: i32, patch_size: i32, hidden: i32) -> f64 {
    let (b, th, tw) = (b as f64, th as f64, tw as f64);
    resident_full_gib(b, th, tw) + forward_activations_gib(b, th, tw, patch_size, hidden)
}

/// Estimated concurrent peak (GiB) of a **tiled** decode of `[b,3,th,tw]` with square `tile`-px tiles:
/// the same resident full-res buffers plus one forward over a single tile (clamped to the output).
pub fn tiled_peak_gib(b: i32, th: i32, tw: i32, tile: i32, patch_size: i32, hidden: i32) -> f64 {
    let (bf, thf, twf) = (b as f64, th as f64, tw as f64);
    let fh = tile.min(th) as f64;
    let fw = tile.min(tw) as f64;
    resident_full_gib(bf, thf, twf) + forward_activations_gib(bf, fh, fw, patch_size, hidden)
}

/// A chosen tiling plan for a decode of output `th × tw`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TilePlan {
    /// Square tile edge (px, multiple of [`TILE_ALIGN`]).
    pub edge: i32,
    /// Feather overlap (px) between neighbouring tiles.
    pub overlap: i32,
    /// `true` when the whole output fits in one tile within budget → decode the exact whole-image path.
    pub whole_fits: bool,
}

/// Pick a tiling plan for a `[b,3,th,tw]` decode under `safe_gib`. Untile when the whole-image peak fits
/// budget (no CUDA watchdog to also satisfy — unlike the MLX mirror). When tiling, prefer the validated
/// [`PREFERRED_TILE_EDGE`], shrinking only if memory forces it (the per-forward term must fit what's left
/// after the resident buffers), floored at [`MIN_TILE_EDGE`]. Pure.
pub fn plan_tile_edge(
    b: i32,
    th: i32,
    tw: i32,
    patch_size: i32,
    hidden: i32,
    safe_gib: f64,
) -> TilePlan {
    let out_edge = th.max(tw);
    if estimated_decode_peak_gib(b, th, tw, patch_size, hidden) <= safe_gib {
        return TilePlan {
            edge: out_edge,
            overlap: DEFAULT_TILE_OVERLAP,
            whole_fits: true,
        };
    }
    let resident = resident_full_gib(b as f64, th as f64, tw as f64);
    let avail = (safe_gib - resident).max(0.0);
    let per_px2 = forward_activations_gib(b as f64, 1.0, 1.0, patch_size, hidden); // GiB at 1×1
    let mem_edge = if per_px2 > 0.0 {
        (avail / per_px2).max(0.0).sqrt() as i32
    } else {
        out_edge
    };
    let edge = (mem_edge
        .min(PREFERRED_TILE_EDGE)
        .min(out_edge)
        .max(MIN_TILE_EDGE)
        / TILE_ALIGN
        * TILE_ALIGN)
        .max(TILE_ALIGN);
    TilePlan {
        edge,
        overlap: DEFAULT_TILE_OVERLAP,
        whole_fits: false,
    }
}

/// The PiD decode budget backstop (F-013 + sc-10087): with tiling, a large output is *tiled* rather than
/// refused, so this refuses only when even a [`MIN_TILE_EDGE`] tile plus the resident full-res buffers
/// exceeds `safe_gib`. `model_id` only labels the error. Pure (budget injected).
pub fn guard(
    model_id: &str,
    count: u32,
    width: u32,
    height: u32,
    scale: i32,
    cfg: &PidConfig,
    safe_gib: f64,
) -> Result<()> {
    let tw = (width * scale as u32) as i32;
    let th = (height * scale as u32) as i32;
    let b = count.max(1) as i32;
    let floor = tiled_peak_gib(b, th, tw, MIN_TILE_EDGE, cfg.patch_size, cfg.hidden_size);
    if floor > safe_gib {
        return Err(CandleError::Msg(format!(
            "{model_id}: a PiD decode at {tw}×{th} (super-resolved {scale}× from {width}×{height}) \
             needs ~{floor:.0} GB even tiled at the minimum {MIN_TILE_EDGE}px tile — the output-resolution \
             noise/latent buffers alone exceed this machine's ~{safe_gib:.0} GB safe budget. Reduce the \
             resolution or disable PiD for this request."
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const PATCH: i32 = 16;
    const HIDDEN: i32 = 1536;

    #[test]
    fn whole_peak_grows_with_area() {
        let base = estimated_decode_peak_gib(1, 512, 512, PATCH, HIDDEN);
        let dbl = estimated_decode_peak_gib(1, 1024, 1024, PATCH, HIDDEN);
        assert!((dbl - 4.0 * base).abs() / base < 1e-9);
    }

    #[test]
    fn tiling_cuts_the_forward_term() {
        let whole = estimated_decode_peak_gib(1, 6144, 6144, PATCH, HIDDEN);
        let tiled = tiled_peak_gib(1, 6144, 6144, 2048, PATCH, HIDDEN);
        assert!(
            tiled < whole * 0.5,
            "tiled {tiled:.1} vs whole {whole:.1} GiB"
        );
        assert!(tiled > resident_full_gib(1.0, 6144.0, 6144.0));
    }

    #[test]
    fn plan_untiles_when_whole_fits_vram() {
        // Ample VRAM → whole-image (no watchdog constraint on CUDA, so even 6144² untiles if it fits).
        let plan = plan_tile_edge(1, 6144, 6144, PATCH, HIDDEN, 100.0);
        assert!(plan.whole_fits, "6144² fits 100 GiB → whole: {plan:?}");
    }

    #[test]
    fn plan_tiles_when_whole_busts_vram() {
        // A 6144² decode that doesn't fit an 8 GiB card must tile (the sc-10087 CUDA repro).
        let plan = plan_tile_edge(1, 6144, 6144, PATCH, HIDDEN, 8.0);
        assert!(!plan.whole_fits, "6144² must tile under 8 GiB: {plan:?}");
        assert!(plan.edge <= PREFERRED_TILE_EDGE);
        assert!(plan.edge >= MIN_TILE_EDGE);
        assert_eq!(plan.edge % TILE_ALIGN, 0);
    }

    #[test]
    fn guard_admits_large_output_that_tiling_can_handle() {
        let cfg = PidConfig::sr4x();
        assert!(
            guard("qwenimage", 1, 2048, 2048, 4, &cfg, 14.0).is_ok(),
            "8192² should tile, not refuse, under 14 GiB"
        );
    }

    #[test]
    fn guard_refuses_when_even_a_min_tile_wont_fit() {
        let cfg = PidConfig::sr4x();
        let err = guard("qwenimage", 1, 2048, 2048, 4, &cfg, 2.0)
            .expect_err("8192² must refuse under a 2 GiB budget");
        let msg = err.to_string();
        assert!(
            matches!(err, CandleError::Msg(_)),
            "typed refusal, got {err:?}"
        );
        assert!(msg.contains("minimum"), "names the min-tile floor: {msg}");
    }
}
