//! Shared **budgeted video-VAE tiling machinery** (sc-9006 / F-026) — the single home for the
//! tile/narrow/blend/pad-accumulate/normalize DRIVER and the VRAM-budget selector that were copied
//! byte-near-identically between [`candle-gen-wan`]'s z48 vae22 decode and [`candle-gen-ltx`]'s LTX
//! decode (and echoed in [`candle-gen-seedvr2`]).
//!
//! The tile **geometry** already lives in the pure [`gen_core::tiling`] module ([`budgeted_plan`],
//! [`TilePlan`], [`AxisTile`], [`VaeTiling`]) and is shared verbatim by every backend. What was still
//! duplicated is the candle-side *execution* of a plan:
//!
//!  - [`decode_tiled`] — split a latent into the plan's overlapping tiles, decode each through a
//!    caller-supplied closure, trapezoidally blend the results into the full output by
//!    pad-and-accumulate, and normalize by the summed blend weight. ~80 lines, identical in both.
//!  - [`safe_budget_gib`] — the `<PREFIX>_VAE_BUDGET_GIB` env override → `nvidia-smi` total × safe-frac
//!    → default-fallback resolver (0.85 safe-frac, 16 GiB default in both).
//!  - [`plan_tiling`] — feed a per-VAE candidate grid + cost model to [`budgeted_plan`] and map the
//!    [`TilingBudgetError`] into a human-readable, catchable [`candle_core::Error`].
//!
//! **What stays per-VAE** (this module parameterizes, never merges, these):
//!  - the [`VaeTiling`] variant (WAN22 ×16 spatial / ×4 causal temporal vs LTX ×32 / ×8 causal);
//!  - the decode closure (each VAE's own decoder graph);
//!  - the candidate grid ([`TileCandidates`]) — notably wan carries **no temporal candidates** (its
//!    `decode` already streams one latent frame at a time, so only the spatial axes are ever tiled),
//!    whereas ltx carries a full spatial + temporal grid;
//!  - the cost model closure (wan's streaming `ACCUM·out_vox + FRAME·frame_px` vs ltx's
//!    `FIXED + ACCUM·out_vox + TILE·tile_vox`) and its calibrated constants;
//!  - the env-var name / label used in budget resolution + error strings.
//!
//! The driver's tiling **decisions and numerics are unchanged** from the two hand-copied bodies: same
//! narrow offsets, same trapezoidal outer-product blend, same `pad_with_zeros` placement, same
//! `maximum(1e-8)` normalize — so decoded output is byte-identical for a given plan + decode closure.

use candle_core::{Error, Result, Tensor};

use gen_core::tiling::{
    budgeted_plan, TileCandidates, TilePlan, TilingBudgetError, TilingConfig, VaeTiling,
};

/// Execute a budgeted tile `plan` for a 5-D video latent `[B, C, F, H, W]`: for each tile in the
/// plan, `narrow` the latent, decode it via `decode_fn`, trapezoidally blend, and pad-and-accumulate
/// into the full output, then normalize by the summed blend weight.
///
/// This is the generic form of the identical `WanVae::decode_tiled` / `LtxVideoVae::decode_tiled`
/// bodies (sc-9006). It is parameterized only by:
///  - `vae` — the [`VaeTiling`] geometry (spatial/temporal scale + causal-vs-non-causal mapping);
///  - `label` — a short crate tag for the "plan had no tiles" error (e.g. `"wan z48 vae22"`);
///  - `decode_fn` — the caller's single-tile decoder (`WanVae::decode` streams per-frame; `LtxVideoVae::decode`
///    is single-pass) — it MUST return `[B, 3, out_f, out_h, out_w]` for the tile's latent extent.
///
/// When `cfg` does not fire for these latent dims ([`TilingConfig::needs_tiling`] is false) the whole
/// latent is decoded in one `decode_fn` call (no tiling), exactly as the per-crate copies did.
///
/// The tile loop, narrow offsets, `[1,1,at,ah,aw]` trapezoidal outer-product blend, `pad_with_zeros`
/// placement, and `maximum(1e-8)` weight normalize are byte-for-byte the prior hand-copied logic, so
/// output is unchanged for a given `plan` + `decode_fn`.
pub fn decode_tiled<F>(
    vae: VaeTiling,
    label: &str,
    latent: &Tensor,
    cfg: &TilingConfig,
    decode_fn: F,
) -> Result<Tensor>
where
    F: Fn(&Tensor) -> Result<Tensor>,
{
    let (_b, _c, f, h, w) = latent.dims5()?;
    if !cfg.needs_tiling(vae, f as i32, h as i32, w as i32) {
        return decode_fn(latent);
    }
    let plan = cfg.plan(vae, f as i32, h as i32, w as i32);
    blend_plan(label, latent, &plan, decode_fn)
}

/// The pure pad-and-accumulate tile blender (split out of [`decode_tiled`] so unit tests can drive a
/// known `plan` + synthetic decode closure without a [`TilingConfig`]). Loops `plan.t × plan.h ×
/// plan.w`, narrows each latent tile, decodes it, blends via the trapezoidal outer-product mask, and
/// accumulates into the full-output `output`/`weights` buffers, finally normalizing.
fn blend_plan<F>(label: &str, latent: &Tensor, plan: &TilePlan, decode_fn: F) -> Result<Tensor>
where
    F: Fn(&Tensor) -> Result<Tensor>,
{
    let dev = latent.device();

    // Full-size accumulators (pad-and-accumulate each tile). `output` carries the batch; `weights`
    // stays `b=1` and broadcasts on the final divide.
    let mut output: Option<Tensor> = None; // [B, 3, out_f, out_h, out_w]
    let mut weights: Option<Tensor> = None; // [1, 1, out_f, out_h, out_w]

    for t in &plan.t {
        for hh in &plan.h {
            for ww in &plan.w {
                let tile = latent
                    .narrow(2, t.start as usize, (t.end - t.start) as usize)?
                    .narrow(3, hh.start as usize, (hh.end - hh.start) as usize)?
                    .narrow(4, ww.start as usize, (ww.end - ww.start) as usize)?;
                let dec = decode_fn(&tile)?; // [B, 3, td, hd, wd]
                let (_, _, td, hd, wd) = dec.dims5()?;
                let at = td.min((t.out_stop - t.out_start) as usize);
                let ah = hd.min((hh.out_stop - hh.out_start) as usize);
                let aw = wd.min((ww.out_stop - ww.out_start) as usize);

                // 1-D trapezoidal masks → outer product [1, 1, at, ah, aw].
                let tm = Tensor::from_slice(&t.mask[..at], (1, 1, at, 1, 1), dev)?;
                let hm = Tensor::from_slice(&hh.mask[..ah], (1, 1, 1, ah, 1), dev)?;
                let wm = Tensor::from_slice(&ww.mask[..aw], (1, 1, 1, 1, aw), dev)?;
                let blend = tm.broadcast_mul(&hm)?.broadcast_mul(&wm)?;

                let dec = dec.narrow(2, 0, at)?.narrow(3, 0, ah)?.narrow(4, 0, aw)?;
                let weighted = dec.broadcast_mul(&blend)?;

                // Place each tile at its output offset by zero-padding to the full output shape.
                // `out_start + a* <= out_stop <= out_*`, so the right pad never underflows.
                let (pt0, pt1) = (
                    t.out_start as usize,
                    plan.out_f as usize - (t.out_start as usize + at),
                );
                let (ph0, ph1) = (
                    hh.out_start as usize,
                    plan.out_h as usize - (hh.out_start as usize + ah),
                );
                let (pw0, pw1) = (
                    ww.out_start as usize,
                    plan.out_w as usize - (ww.out_start as usize + aw),
                );
                let pad5 = |x: &Tensor| -> Result<Tensor> {
                    x.pad_with_zeros(2, pt0, pt1)?
                        .pad_with_zeros(3, ph0, ph1)?
                        .pad_with_zeros(4, pw0, pw1)
                };
                let weighted_full = pad5(&weighted)?;
                let blend_full = pad5(&blend)?;

                output = Some(match output {
                    None => weighted_full,
                    Some(acc) => acc.add(&weighted_full)?,
                });
                weights = Some(match weights {
                    None => blend_full,
                    Some(acc) => acc.add(&blend_full)?,
                });
            }
        }
    }

    let output =
        output.ok_or_else(|| Error::Msg(format!("{label}: tile-decode plan had no tiles")))?;
    let weights =
        weights.ok_or_else(|| Error::Msg(format!("{label}: tile-decode plan had no tiles")))?;
    // Normalize by the summed blend weight (clamped away from 0), broadcasting [1,1,F,H,W] over C.
    output.broadcast_div(&weights.maximum(1e-8f64)?)
}

/// Resolve the safe peak-GiB budget for a video-VAE decode tiler. Resolved in order:
///  1. the `env_var` override (a positive float — the deterministic injection point for the
///     worker/tests, e.g. `WAN_VAE_BUDGET_GIB` / `LTX_VAE_BUDGET_GIB`);
///  2. total VRAM × `safe_frac` via the shared trusted-path `nvidia-smi` probe
///     ([`crate::gpu::nvidia_smi_min_total_gib`] — an absolute System32/CUDA_PATH binary, never a
///     bare `PATH` lookup; sc-9014 / F-030);
///  3. `default_gib`.
///
/// Both callers pass `safe_frac = 0.85` and `default_gib = 16.0` today, but they stay per-caller
/// parameters (not baked-in constants) so a future VAE with a different memory profile can differ
/// without re-forking the resolver.
pub fn safe_budget_gib(env_var: &str, safe_frac: f64, default_gib: f64) -> f64 {
    if let Ok(raw) = std::env::var(env_var) {
        if let Ok(gib) = raw.trim().parse::<f64>() {
            if gib > 0.0 {
                return gib;
            }
        }
    }
    match crate::gpu::nvidia_smi_min_total_gib() {
        Some(total) => total * safe_frac,
        None => default_gib,
    }
}

/// Route a per-VAE `candidates` grid + `cost_fn` cost model through the shared [`budgeted_plan`]
/// selector and map the [`TilingBudgetError`] into a human-readable, catchable [`candle_core::Error`]
/// tagged with `label` (e.g. `"wan z48 vae22 decode"`). Caller passes the **output** dims.
///
/// `Ok(None)` → a single-pass decode already fits the budget; `Ok(Some(cfg))` → the largest tiling
/// that fits; `Err` → an over-budget signal returned **before** the decode (a catchable error, not an
/// OOM). This is the generic form of the identical `plan_wan22_tiling` / `plan_ltx_tiling` bodies.
pub fn plan_tiling<F>(
    label: &str,
    height: i32,
    width: i32,
    out_frames: i32,
    safe_gib: f64,
    candidates: TileCandidates<'_>,
    cost_fn: F,
) -> Result<Option<TilingConfig>>
where
    F: Fn(i64, i64, i64, i64, i64, i64) -> f64,
{
    budgeted_plan(height, width, out_frames, safe_gib, candidates, cost_fn).map_err(|e| match e {
        TilingBudgetError::AccumulatorsExceedBudget {
            projected_gib,
            safe_gib,
        } => Error::Msg(format!(
            "{label}: assembling a {width}×{height}×{out_frames} video needs ~{projected_gib:.0} GB \
             just for the output buffers, over the ~{safe_gib:.0} GB safe VRAM budget. Reduce the \
             resolution or frame count."
        )),
        TilingBudgetError::SmallestTileExceedsBudget {
            projected_gib,
            safe_gib,
        } => Error::Msg(format!(
            "{label}: a {width}×{height}×{out_frames} video peaks at ~{projected_gib:.0} GB even with \
             the smallest tile, over the ~{safe_gib:.0} GB safe VRAM budget. Reduce the resolution or \
             frame count."
        )),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    // A synthetic "VAE": the decode closure is identity-scaled so the blended output is exactly
    // reconstructible, letting us assert the DRIVER's blend/stitch is correct independent of any real
    // decoder. Uses `VaeTiling::WAN` (×8 spatial / ×4 non-causal temporal) with a plan that fires.

    /// Decode = spatial ×8, temporal ×4 nearest-neighbour upsample (matches `VaeTiling::WAN` output
    /// geometry) of a constant-per-tile latent, so the trapezoidal blend of overlapping tiles must
    /// reconstruct the same constant field everywhere (partition-of-unity property of the masks).
    fn upsample_wan(tile: &Tensor) -> Result<Tensor> {
        let (b, _c, f, h, w) = tile.dims5()?;
        // The latent is a constant field (see the test); emit a [B,3, f*4, h*8, w*8] constant of the
        // same value. Read the constant from element 0.
        let val: f32 = tile.flatten_all()?.to_vec1::<f32>()?[0];
        Tensor::full(val, (b, 3, f * 4, h * 8, w * 8), tile.device())
    }

    #[test]
    fn blend_reconstructs_constant_field() {
        // A constant latent decoded to a constant field: after trapezoidal blend + normalize, every
        // output voxel must equal the constant (the masks form a partition of unity over the output).
        let dev = Device::Cpu;
        let cfg = TilingConfig::spatial_only(64, 16); // tile 64px, overlap 16 (output px)
                                                      // 16×16 latent → 128×128 output (×8), forced to tile (>8 threshold) into overlapping tiles.
        let latent = Tensor::full(0.7f32, (1, 4, 1, 16, 16), &dev).unwrap();
        let out = decode_tiled(VaeTiling::WAN, "test wan", &latent, &cfg, upsample_wan).unwrap();
        assert_eq!(out.dims(), &[1, 3, 4, 128, 128]);
        let v = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let (min, max) = v
            .iter()
            .fold((f32::MAX, f32::MIN), |(lo, hi), &x| (lo.min(x), hi.max(x)));
        assert!(
            (min - 0.7).abs() < 1e-4 && (max - 0.7).abs() < 1e-4,
            "blend must reconstruct the constant 0.7 everywhere; got [{min}, {max}]"
        );
    }

    #[test]
    fn no_tiling_when_below_threshold() {
        // Below the WAN needs_tiling threshold (h=w=8), the driver decodes the whole latent once.
        let dev = Device::Cpu;
        let cfg = TilingConfig::spatial_only(64, 16);
        let latent = Tensor::full(0.5f32, (1, 4, 2, 8, 8), &dev).unwrap();
        let out = decode_tiled(VaeTiling::WAN, "test wan", &latent, &cfg, upsample_wan).unwrap();
        // Single decode: [1,3, 2*4, 8*8, 8*8].
        assert_eq!(out.dims(), &[1, 3, 8, 64, 64]);
    }

    #[test]
    fn synthetic_cost_model_selects_deterministic_plan() {
        // A synthetic cost model + candidate grid drives the budgeted selector to a deterministic
        // decision: a tiny budget must force the smallest tile; a huge budget must return None
        // (single-pass fits). Cost = bytes-per-output-voxel-only so the choice is easy to reason about.
        const PX: [i32; 3] = [512, 256, 128];
        const FR: [(i32, i32); 0] = [];
        let cost = |of: i64, oh: i64, ow: i64, _tf: i64, th: i64, tw: i64| -> f64 {
            // accumulator floor (per out voxel) + per-tile term (per tile pixel).
            let out_vox = (of * oh * ow) as f64;
            let tile_px = (th * tw) as f64;
            (10.0 * out_vox + 1000.0 * tile_px) / (1024.0 * 1024.0 * 1024.0)
        };
        let cand = || TileCandidates {
            spatial_px: &PX,
            spatial_overlap_px: 32,
            temporal: &FR,
        };

        // For 1024²×49: accumulator floor (zero tile) ≈ 0.48 GiB; single-pass ≈ 1.46 GiB. So a
        // budget above single-pass returns None; between floor and single-pass forces a spatial tile;
        // below the floor is a catchable Err.
        // Huge budget → single-pass fits → None.
        let none = plan_tiling("t", 1024, 1024, 49, 1e6, cand(), cost).unwrap();
        assert!(none.is_none(), "huge budget should not tile");

        // Between the accumulator floor and single-pass → tiles (Some, spatial set, no temporal).
        let some = plan_tiling("t", 1024, 1024, 49, 1.0, cand(), cost)
            .unwrap()
            .expect("mid budget must tile");
        assert!(some.spatial.is_some());
        assert!(some.temporal.is_none(), "no temporal candidates supplied");

        // Impossible budget (accumulator floor alone exceeds it) → catchable Err, not a panic/OOM.
        assert!(plan_tiling("t", 4096, 4096, 257, 0.001, cand(), cost).is_err());
    }

    #[test]
    fn safe_budget_env_override_wins() {
        // The deterministic injection point the worker/tests use. (Set/clear in-process.)
        std::env::set_var("CG_TEST_VAE_BUDGET_GIB", "42.5");
        assert_eq!(safe_budget_gib("CG_TEST_VAE_BUDGET_GIB", 0.85, 16.0), 42.5);
        std::env::remove_var("CG_TEST_VAE_BUDGET_GIB");
        // With no env + (on a CPU/CI box) no nvidia-smi, falls back to the default.
        // (On a GPU box this returns total×frac; either way it is > 0.)
        assert!(safe_budget_gib("CG_TEST_VAE_BUDGET_GIB_UNSET", 0.85, 16.0) > 0.0);
    }
}
