//! LTX-2.3 **video VAE decoder** (`CausalVideoAutoencoder`, latent 128-ch, patch 4, 8× temporal /
//! 32× spatial) — port of mlx-gen-ltx `vae.rs` (`LTX2VideoDecoder`). T2V needs only `decode`; the
//! encoder (I2V) is deferred.
//!
//! Decode: denormalize `latent·std + mean` → `conv_in 128→1024` → 9 up_blocks (`Res` groups +
//! `DepthToSpace` upsamplers) → pixel-norm (eps 1e-8) → SiLU → `conv_out 128→48` → unpatchify(×4).
//! All convs are non-causal (frame-replication temporal pad). pixel_norm = `x/√(mean(x² over C)+eps)`
//! (no √C, no γ). Runs **f32**.
//!
//! Block execution order (the config `decoder_blocks` list is encoder-order; the decoder reverses
//! it): `Res(2), Up(2,2,2), Res(2), Up(2,2,2), Res(4), Up(2,1,1), Res(6), Up(1,2,2), Res(4)`. Each
//! `Up` with temporal stride 2 doubles then drops the first frame, so latent T=7 → 49 pixel frames;
//! spatial 15 → 480 px (×2×2×2 then unpatchify ×4).

use candle_gen::candle_core::{Result, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::tiling::{TileCandidates, TilingConfig, VaeTiling};
use candle_gen::vae_tiling;

use crate::conv3d::CausalConv3d;

const DEC_NORM_EPS: f64 = 1e-8;

/// `x / sqrt(mean(x² over C, keepdims) + eps)` — LTX PixelNorm (channel axis = 1, no √C, no γ).
fn pixel_norm(x: &Tensor) -> Result<Tensor> {
    let c = x.dim(1)?;
    let sumsq = x.sqr()?.sum_keepdim(1)?;
    let mean = (sumsq / c as f64)?;
    let denom = (mean + DEC_NORM_EPS)?.sqrt()?;
    x.broadcast_div(&denom)
}

/// Decoder residual block (`ResnetBlock3DSimple`): pixel-norm → SiLU → conv → pixel-norm → SiLU →
/// conv → residual add. Channels constant (no shortcut).
struct DecResBlock {
    conv1: CausalConv3d,
    conv2: CausalConv3d,
}

impl DecResBlock {
    fn load(vb: VarBuilder, prefix: &str) -> Result<Self> {
        Ok(Self {
            conv1: CausalConv3d::load(vb.clone(), &format!("{prefix}.conv1.conv"))?,
            conv2: CausalConv3d::load(vb, &format!("{prefix}.conv2.conv"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = candle_gen::candle_nn::ops::silu(&pixel_norm(x)?)?;
        let h = self.conv1.forward(&h, false)?;
        let h = candle_gen::candle_nn::ops::silu(&pixel_norm(&h)?)?;
        let h = self.conv2.forward(&h, false)?;
        h + x
    }
}

/// `DepthToSpaceUpsample` (residual=false): conv → depth-to-space → (st>1) drop first temporal frame.
struct DepthToSpace {
    conv: CausalConv3d,
    st: usize,
    sh: usize,
    sw: usize,
}

impl DepthToSpace {
    fn load(vb: VarBuilder, prefix: &str, stride: (usize, usize, usize)) -> Result<Self> {
        Ok(Self {
            conv: CausalConv3d::load(vb, &format!("{prefix}.conv.conv"))?,
            st: stride.0,
            sh: stride.1,
            sw: stride.2,
        })
    }

    /// `(B, C·st·sh·sw, D, H, W) -> (B, C, D·st, H·sh, W·sw)`.
    fn depth_to_space(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c_packed, d, h, w) = x.dims5()?;
        let (st, sh, sw) = (self.st, self.sh, self.sw);
        let c = c_packed / (st * sh * sw);
        let x = x.reshape([b, c, st, sh, sw, d, h, w].as_slice())?;
        // transpose to (B, C, D, st, H, sh, W, sw) = axes [0,1,5,2,6,3,7,4].
        let x = x.permute([0usize, 1, 5, 2, 6, 3, 7, 4].as_slice())?;
        x.reshape((b, c, d * st, h * sh, w * sw))?.contiguous()
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv.forward(x, false)?;
        let x = self.depth_to_space(&x)?;
        if self.st > 1 {
            let t = x.dim(2)?;
            x.narrow(2, 1, t - 1)
        } else {
            Ok(x)
        }
    }
}

enum UpLayer {
    Res(Vec<DecResBlock>),
    Up(DepthToSpace),
}

/// One decoder block in execution order: a res group of `n` blocks, or an upsampler with `stride`.
enum DBlock {
    Res(usize),
    Up((usize, usize, usize)),
}

/// The fixed LTX-2.3 decoder block order (config `decoder_blocks` already reversed to execution order).
const DECODER_BLOCKS: [DBlock; 9] = [
    DBlock::Res(2),
    DBlock::Up((2, 2, 2)),
    DBlock::Res(2),
    DBlock::Up((2, 2, 2)),
    DBlock::Res(4),
    DBlock::Up((2, 1, 1)),
    DBlock::Res(6),
    DBlock::Up((1, 2, 2)),
    DBlock::Res(4),
];

/// `(B, C·p², F, H, W) -> (B, C, F, H·p, W·p)` (spatial-only unpatchify, patch_size_t = 1).
fn unpatchify(x: &Tensor, p: usize) -> Result<Tensor> {
    let (b, c_packed, f, h, w) = x.dims5()?;
    let c = c_packed / (p * p);
    // (B, C, 1, p, p, F, H, W) -> transpose (0,1,5,2,6,4,7,3) -> (B, C, F, H·p, W·p).
    let x = x.reshape([b, c, 1, p, p, f, h, w].as_slice())?;
    let x = x.permute([0usize, 1, 5, 2, 6, 4, 7, 3].as_slice())?;
    x.reshape((b, c, f, h * p, w * p))?.contiguous()
}

/// The LTX-2.3 video VAE (decoder only, T2V).
pub struct LtxVideoVae {
    conv_in: CausalConv3d,
    up_blocks: Vec<UpLayer>,
    conv_out: CausalConv3d,
    mean: Tensor, // [1, 128, 1, 1, 1]
    std: Tensor,  // [1, 128, 1, 1, 1]
    patch_size: usize,
}

impl LtxVideoVae {
    /// Build from a VarBuilder rooted at the `vae.` prefix of the checkpoint.
    pub fn new(vb: VarBuilder, latent_channels: usize, patch_size: usize) -> Result<Self> {
        let dec = vb.pp("decoder");
        let mut up_blocks = Vec::with_capacity(DECODER_BLOCKS.len());
        for (idx, block) in DECODER_BLOCKS.iter().enumerate() {
            let prefix = format!("up_blocks.{idx}");
            up_blocks.push(match block {
                DBlock::Res(n) => {
                    let mut blocks = Vec::with_capacity(*n);
                    for j in 0..*n {
                        blocks.push(DecResBlock::load(
                            dec.clone(),
                            &format!("{prefix}.res_blocks.{j}"),
                        )?);
                    }
                    UpLayer::Res(blocks)
                }
                DBlock::Up(stride) => {
                    UpLayer::Up(DepthToSpace::load(dec.clone(), &prefix, *stride)?)
                }
            });
        }
        let stats = vb.pp("per_channel_statistics");
        let mean = stats
            .get_unchecked("mean-of-means")?
            .reshape((1, latent_channels, 1, 1, 1))?;
        let std = stats
            .get_unchecked("std-of-means")?
            .reshape((1, latent_channels, 1, 1, 1))?;
        Ok(Self {
            conv_in: CausalConv3d::load(dec.clone(), "conv_in.conv")?,
            up_blocks,
            conv_out: CausalConv3d::load(dec, "conv_out.conv")?,
            mean,
            std,
            patch_size,
        })
    }

    /// Decode a normalized latent `[B, 128, F', H', W']` → video `[B, 3, F, 32·H', 32·W']` in ~[-1,1].
    pub fn decode(&self, latent: &Tensor) -> Result<Tensor> {
        // Denormalize: x · std + mean.
        let x =
            (latent.broadcast_mul(&self.std)? + self.mean.broadcast_as(latent.shape())?.clone())?;
        let mut x = self.conv_in.forward(&x, false)?;
        for layer in &self.up_blocks {
            x = match layer {
                UpLayer::Res(blocks) => {
                    let mut h = x;
                    for b in blocks {
                        h = b.forward(&h)?;
                    }
                    h
                }
                UpLayer::Up(u) => u.forward(&x)?,
            };
        }
        let x = pixel_norm(&x)?;
        let x = candle_gen::candle_nn::ops::silu(&x)?;
        let x = self.conv_out.forward(&x, false)?;
        unpatchify(&x, self.patch_size)
    }

    /// Decode with **tiling** for memory-bounded large/long-video decode (`cfg`) — the candle port of
    /// mlx-gen-ltx `LtxVideoVae::decode_tiled` (sc-7076 / sc-6894). Splits the latent into overlapping
    /// spatial/temporal tiles via the shared pure `gen_core::tiling` geometry (`VaeTiling::LTX`: ×32
    /// spatial, ×8 **causal** temporal), decodes each tile through [`decode`](Self::decode), and
    /// trapezoidally blends them into the full video by pad-and-accumulate (bounded peak = one tile's
    /// decode + the full-output `output`/`weights` buffers). Falls back to single-pass [`decode`] when
    /// `cfg` does not fire for these dims.
    ///
    /// Numerically mirrors the parity-validated mlx version op-for-op; candle's eager evaluation makes
    /// the reference's per-tile `mx.eval` (the peak-bounding barrier) unnecessary. **NOTE (CUDA-gated):**
    /// the spatial-tiling path is straightforward, but temporal tiling crosses the causal-Conv3d frame
    /// boundary (each tile's leading edge replicates the *tile's* first frame) — the gen_core causal
    /// temporal mapping handles the geometry, but byte-parity vs. a full single-pass decode must be
    /// confirmed on real weights + CUDA (the Mac dev host can only compile-check this).
    pub fn decode_tiled(&self, latent: &Tensor, cfg: &TilingConfig) -> Result<Tensor> {
        // The tile/narrow/blend/pad-accumulate/normalize DRIVER is shared with the wan half in
        // `candle_gen::vae_tiling::decode_tiled` (sc-9006 / F-026). What stays ltx-specific: the
        // `VaeTiling::LTX` geometry (×32 spatial / ×8 causal temporal) and the single-pass `decode`
        // closure. Unlike wan, `cfg` may carry a temporal tile (ltx `decode` is not per-frame
        // streaming); the shared driver's `plan.t` loop handles both.
        vae_tiling::decode_tiled(VaeTiling::LTX, "ltx vae", latent, cfg, |tile| {
            self.decode(tile)
        })
    }

    /// **Memory-bounded** decode (sc-7076): derive the decoded output dims from the latent geometry
    /// (LTX VAE: ×32 spatial, ×8 **causal** temporal ⇒ `out_f = 1 + (T_lat−1)·8`), pick a budgeted
    /// tiling via [`auto_tiling_budgeted_ltx`], and run [`decode_tiled`](Self::decode_tiled) — or the
    /// single-pass [`decode`](Self::decode) when the whole decode already fits the VRAM budget. An
    /// over-budget decode returns a **catchable** error here instead of OOM-ing the worker. The candle
    /// analogue of mlx-gen-ltx `decode_to_frames`'s internal budgeting.
    pub fn decode_budgeted(&self, latent: &Tensor) -> Result<Tensor> {
        let (_b, _c, f, h, w) = latent.dims5()?;
        let out_f = 1 + (f as i32 - 1) * VaeTiling::LTX.temporal_scale; // causal ×8
        let out_h = h as i32 * VaeTiling::LTX.spatial_scale; // ×32
        let out_w = w as i32 * VaeTiling::LTX.spatial_scale;
        match auto_tiling_budgeted_ltx(out_h, out_w, out_f)? {
            Some(cfg) => self.decode_tiled(latent, &cfg),
            None => self.decode(latent),
        }
    }
}

// --- sc-7076 / sc-6894: budgeted LTX VAE decode (candle) ------------------------------------------
//
// The shared budgeted-tiling DRIVER + budget resolver + selector now live ONCE in
// `candle_gen::vae_tiling` (sc-9006 / F-026, de-duped from the byte-near-identical wan/ltx copies);
// this module supplies only the LTX-specific cost CONSTANTS, the spatial+temporal candidate grid, and
// the cost model. The tile geometry itself is pure gen-core (`gen_core::tiling`), byte-identical to
// the mlx side.

const GIB_F64: f64 = 1024.0 * 1024.0 * 1024.0;
/// Env override read by the shared [`vae_tiling::safe_budget_gib`] resolver.
const LTX_VAE_BUDGET_ENV: &str = "LTX_VAE_BUDGET_GIB";
/// Fraction of total VRAM treated as safe (matches the mlx 0.85 + candle-gen-seedvr2 convention).
const LTX_VAE_BUDGET_SAFE_FRAC: f64 = 0.85;
/// Fallback budget when neither the env override nor `nvidia-smi` yields a value.
const LTX_VAE_DEFAULT_BUDGET_GIB: f64 = 16.0;

// Cost-model constants. **CUDA-CALIBRATED (sc-7148)** — fit from real-weight peak-VRAM anchors measured
// by `tests/vae_decode_sweep.rs` on an RTX PRO 6000 Blackwell (sm_120, CUDA 12.9, f32, device-level
// `nvidia-smi` peak), replacing the mlx-Metal placeholders (sc-6894). The five anchors (output WxHxF /
// largest-tile px·fr → measured peak):
//   512²×25  single-pass                    →  5.99 GiB   (≈635 B/out-voxel above the floor)
//   768²×25  single-pass                    → 10.86 GiB
//   1024²×25 single-pass                    → 17.27 GiB
//   1280²×121 tiled 256px/64fr              → 14.83 GiB   (accumulator-dominated: tiny per-tile term)
//   1280²×121 tiled 512px/64fr              → 16.39 GiB
// The peak splits into a fixed floor (resident VAE decoder + CUDA context, ≈2.2 GiB baseline), a
// per-output-voxel accumulator term (the f32 `output`/`weights` buffers + pad/add transients, fit
// ≈54 B/voxel from the tiled anchors), and a per-tile activation term that scales with the largest
// tile's output volume (the single-pass anchors imply ≈635 B/voxel for the decoder's 1024-channel
// stack). Constants are rounded to the **conservative** (over-predicting) side so the budgeted selector
// never picks a tile that OOMs — the model reproduces every anchor at ratio 1.12–1.65× (never under).
// The placeholders (40/300) under-predicted single-pass by ~1.9×; re-run the sweep after a decoder or
// candle-allocator change. See the `ltx_decode_peak_matches_cuda_anchors` regression test below.
const LTX_VAE_FIXED_BYTES: f64 = 2.7e9;
const LTX_VAE_ACCUM_BYTES_PER_VOXEL: f64 = 80.0;
const LTX_VAE_TILE_BYTES_PER_OUT_VOXEL: f64 = 620.0;

/// Candidate spatial tile sizes (output px, multiples of the LTX ×32 scale, overlap 64).
const LTX_VAE_SPATIAL_PX: [i32; 8] = [768, 640, 512, 448, 384, 320, 256, 192];
/// Candidate temporal tiles `(tile_frames, overlap_frames)` in output frames.
const LTX_VAE_TEMPORAL_FR: [(i32, i32); 4] = [(96, 24), (64, 16), (48, 16), (24, 8)];

/// Estimated concurrent peak (GiB) of an LTX decode whose largest tile spans `tile_*` output voxels
/// while assembling an `out_*` video. `FIXED + ACCUM·out_vox + TILE·tile_vox`. Single-pass is
/// `tile_* == out_*`; a zero tile is the accumulator+fixed floor.
fn estimated_ltx_decode_peak_gib(
    out_f: i64,
    out_h: i64,
    out_w: i64,
    tile_f: i64,
    tile_h: i64,
    tile_w: i64,
) -> f64 {
    let out_voxels = (out_f * out_h * out_w) as f64;
    let tile_voxels = (tile_f * tile_h * tile_w) as f64;
    (LTX_VAE_FIXED_BYTES
        + LTX_VAE_ACCUM_BYTES_PER_VOXEL * out_voxels
        + LTX_VAE_TILE_BYTES_PER_OUT_VOXEL * tile_voxels)
        / GIB_F64
}

/// The safe peak-GiB budget for the LTX decode tiler. Resolved in order: `LTX_VAE_BUDGET_GIB` env
/// override (positive float — the deterministic injection point for the worker/tests) → total VRAM ×
/// [`LTX_VAE_BUDGET_SAFE_FRAC`] (via the shared trusted-path `nvidia-smi` probe
/// [`candle_gen::gpu::nvidia_smi_min_total_gib`] — an absolute System32/CUDA_PATH binary, never a bare
/// `PATH` lookup; sc-9014 / F-030) → [`LTX_VAE_DEFAULT_BUDGET_GIB`].
pub fn ltx_vae_safe_budget_gib() -> f64 {
    vae_tiling::safe_budget_gib(
        LTX_VAE_BUDGET_ENV,
        LTX_VAE_BUDGET_SAFE_FRAC,
        LTX_VAE_DEFAULT_BUDGET_GIB,
    )
}

/// **Memory-budgeted** tiling for the LTX VAE decode — routes the shared [`budgeted_plan`] selector
/// through the LTX cost model. Caller passes the **output** dims. `Ok(None)` → single-pass already
/// fits; `Err` → a catchable over-budget signal returned before the decode (not an OOM).
pub fn auto_tiling_budgeted_ltx(
    height: i32,
    width: i32,
    out_frames: i32,
) -> Result<Option<TilingConfig>> {
    plan_ltx_tiling(height, width, out_frames, ltx_vae_safe_budget_gib())
}

/// Pure LTX tile selector (the `safe_gib` ceiling injected so it is unit-testable without a GPU).
fn plan_ltx_tiling(
    height: i32,
    width: i32,
    out_frames: i32,
    safe_gib: f64,
) -> Result<Option<TilingConfig>> {
    // Shared budgeted selector + error mapping (sc-9006 / F-026); ltx-specific: the spatial+temporal
    // candidate grid and the `estimated_ltx_decode_peak_gib` cost model.
    let candidates = TileCandidates {
        spatial_px: &LTX_VAE_SPATIAL_PX,
        spatial_overlap_px: 64,
        temporal: &LTX_VAE_TEMPORAL_FR,
    };
    vae_tiling::plan_tiling(
        "ltx vae decode",
        height,
        width,
        out_frames,
        safe_gib,
        candidates,
        estimated_ltx_decode_peak_gib,
    )
}

#[cfg(test)]
mod budget_tests {
    use super::*;

    #[test]
    fn ltx_tiling_single_pass_when_small() {
        // A short, low-res clip fits a single-pass decode → no tiling.
        assert!(plan_ltx_tiling(256, 256, 25, 60.0).unwrap().is_none());
    }

    #[test]
    fn ltx_tiling_bounds_moderate_res_peak() {
        // 1280×1280×121 single-pass would peak ~66 GB; on a 48 GiB-class budget it must tile and keep
        // the recomputed peak under the safe ceiling (bounded/catchable). Cost constants are the
        // CUDA-calibrated (sc-7148) anchors; this checks the SELECTOR logic, not the calibration itself
        // (that is covered by `ltx_decode_peak_matches_cuda_anchors`).
        let safe = 48.0 * 0.85;
        let cfg = plan_ltx_tiling(1280, 1280, 121, safe)
            .unwrap()
            .expect("moderate-res LTX must tile");
        let th = cfg
            .spatial
            .map(|s| (s.tile_px as i64).min(1280))
            .unwrap_or(1280);
        let tw = th;
        let tf = cfg
            .temporal
            .map(|t| (t.tile_frames as i64).min(121))
            .unwrap_or(121);
        let peak = estimated_ltx_decode_peak_gib(121, 1280, 1280, tf, th, tw);
        assert!(peak <= safe, "chosen peak {peak:.1} over safe {safe:.1}");
    }

    #[test]
    fn ltx_tiling_errors_when_unfittable() {
        // 4K × 257 frames under 8 GiB: output accumulators (+ fixed floor) alone blow it → catchable.
        assert!(plan_ltx_tiling(2160, 3840, 257, 8.0).is_err());
    }

    #[test]
    fn ltx_budget_env_override_wins() {
        // The deterministic injection point the worker/tests use. (Set/clear in-process.)
        std::env::set_var("LTX_VAE_BUDGET_GIB", "42.5");
        assert_eq!(ltx_vae_safe_budget_gib(), 42.5);
        std::env::remove_var("LTX_VAE_BUDGET_GIB");
    }

    /// sc-7148: the calibrated cost model must stay **conservative** against the real CUDA peak-VRAM
    /// anchors (RTX PRO 6000 Blackwell, sm_120, f32) it was fit from — `estimated ≥ measured` for every
    /// anchor (never under-predict ⇒ the selector never OKs a tile that OOMs), and not absurdly over
    /// (≤ 2.5×). Regenerate the anchors with `cargo test -p candle-gen-ltx --features cuda --release
    /// --test vae_decode_sweep -- --ignored --nocapture` after a decoder or candle-allocator change.
    #[test]
    fn ltx_decode_peak_matches_cuda_anchors() {
        // (out_f, out_h, out_w, tile_f, tile_h, tile_w, measured_peak_gib). Single-pass ⇒ tile == out.
        let anchors: [(i64, i64, i64, i64, i64, i64, f64); 5] = [
            (25, 512, 512, 25, 512, 512, 5.9873),
            (25, 768, 768, 25, 768, 768, 10.8623),
            (25, 1024, 1024, 25, 1024, 1024, 17.2686),
            (121, 1280, 1280, 64, 256, 256, 14.8311),
            (121, 1280, 1280, 64, 512, 512, 16.3936),
        ];
        for (of, oh, ow, tf, th, tw, measured) in anchors {
            let est = estimated_ltx_decode_peak_gib(of, oh, ow, tf, th, tw);
            assert!(
                est >= measured,
                "under-predicts {ow}x{oh}x{of} tile {tw}x{th}x{tf}: est {est:.2} < measured {measured:.2} GiB"
            );
            assert!(
                est <= measured * 2.5,
                "over-predicts {ow}x{oh}x{of} tile {tw}x{th}x{tf}: est {est:.2} > 2.5x measured {measured:.2} GiB"
            );
        }
    }
}
