//! SPLIT 3-D RoPE (double-precision) + the position grid — port of mlx-gen-ltx `positions.rs`
//! (`create_position_grid`) and `rope.rs` (`_precompute_freqs_cis_double_precision` +
//! `apply_split_rotary_emb`), themselves ports of the LTX `models/ltx/rope.py`.
//!
//! The frequency grid is built in **f64** on the host (the reference accumulates 682 log-spaced
//! frequencies in numpy float64 and only down-casts the final cos/sin to f32 — bf16 positions
//! degrade video quality), emitting f32 tensors. Each head covers a distinct 64-wide slice of the
//! 2048 padded frequencies, so cos/sin genuinely vary per head — this is NOT candle's stock `rope`
//! (same freqs per head); the rotate-halves is done manually in [`apply_split_rope`].

use std::f64::consts::PI;

use candle_gen::candle_core::{DType, Device, Result, Tensor, D};

use crate::config::{SPATIAL_SCALE, TEMPORAL_SCALE};

/// Build the f32 position grid `[1, 3, T, 2]` (last axis `[start, end]`) for a latent `(frames,
/// h, w)` token grid, C-major over `(frame, height, width)`. Frame axis: `(t+e)·TEMPORAL_SCALE`,
/// causal first-frame fix `max(0, px+1-TEMPORAL_SCALE)`, then `÷ fps`. Spatial axes: `(coord+e)·
/// SPATIAL_SCALE`.
pub fn create_position_grid(
    frames: usize,
    height: usize,
    width: usize,
    fps: f32,
    device: &Device,
) -> Result<Tensor> {
    let hw = height * width;
    let num_patches = frames * hw;
    let ts = TEMPORAL_SCALE as i64;
    let ss = SPATIAL_SCALE as i64;
    // Row-major (3, T, 2): axis d, patch p, endpoint e.
    let mut data = vec![0f32; 3 * num_patches * 2];
    for p in 0..num_patches {
        let t = (p / hw) as i64;
        let rem = p % hw;
        let h = (rem / width) as i64;
        let w = (rem % width) as i64;
        for e in 0..2i64 {
            let frame_pix = (t + e) * ts;
            let mut frame_f = frame_pix as f32;
            frame_f = (frame_f + 1.0 - ts as f32).max(0.0);
            frame_f /= fps;
            let height_f = ((h + e) * ss) as f32;
            let width_f = ((w + e) * ss) as f32;
            let pe = p * 2 + e as usize;
            data[pe] = frame_f; // d=0
            data[num_patches * 2 + pe] = height_f; // d=1
            data[2 * num_patches * 2 + pe] = width_f; // d=2
        }
    }
    Tensor::from_vec(data, (1, 3, num_patches, 2), device)
}

/// Apply SPLIT (half-rotation) RoPE: `x` is `[B, H, T, D]`, `cos`/`sin` are `[1, H, T, D/2]`.
/// Splits the last axis into halves `(a, b)` and rotates `[a·cos − b·sin, b·cos + a·sin]`
/// (GPT-NeoX form). Computes in f32, casts back to `x`'s dtype.
pub fn apply_split_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let in_dtype = x.dtype();
    let x = x.to_dtype(DType::F32)?;
    let half = x.dim(D::Minus1)? / 2;
    let a = x.narrow(D::Minus1, 0, half)?;
    let b = x.narrow(D::Minus1, half, half)?;
    let out_a = (a.broadcast_mul(cos)? - b.broadcast_mul(sin)?)?;
    let out_b = (b.broadcast_mul(cos)? + a.broadcast_mul(sin)?)?;
    Tensor::cat(&[&out_a, &out_b], D::Minus1)?.to_dtype(in_dtype)
}

/// 1-D split RoPE for the connector: positions `t/max_pos` over `arange(seq)` → `(cos, sin)` each
/// f32 `[1, num_heads, seq, head_dim/2]`. n_pos_dims = 1, so `n_elem = 2`, `num_indices = dim/2`.
pub fn precompute_connector_freqs(
    seq: usize,
    dim: usize,
    theta: f64,
    max_pos: i32,
    num_heads: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let n_pos_dims = 1usize;
    let n_elem = 2 * n_pos_dims; // 2
    let num_indices = (dim / n_elem).max(1);
    let step = if num_indices == 1 {
        0.0
    } else {
        1.0 / (num_indices - 1) as f64
    };
    let indices: Vec<f64> = (0..num_indices)
        .map(|i| theta.powf(i as f64 * step) * (PI / 2.0))
        .collect();
    let current = num_indices * n_pos_dims;
    let expected = dim / 2;
    let pad_size = expected.saturating_sub(current);
    let head_half = expected / num_heads;

    let total = num_heads * seq * head_half;
    let mut cos_out = vec![0f32; total];
    let mut sin_out = vec![0f32; total];
    for t in 0..seq {
        // position = raw index t, scaled by max_pos, *2-1 (mlx connector `rope`).
        let scaled = t as f64 / max_pos as f64 * 2.0 - 1.0;
        for h in 0..num_heads {
            for p in 0..head_half {
                let flat = h * head_half + p;
                let (c, s) = if flat < pad_size {
                    (1.0f32, 0.0f32)
                } else {
                    let k = flat - pad_size;
                    let i = k / n_pos_dims;
                    let ang = scaled * indices[i];
                    (ang.cos() as f32, ang.sin() as f32)
                };
                let o = (h * seq + t) * head_half + p;
                cos_out[o] = c;
                sin_out[o] = s;
            }
        }
    }
    let shape = (1, num_heads, seq, head_half);
    Ok((
        Tensor::from_vec(cos_out, shape, device)?,
        Tensor::from_vec(sin_out, shape, device)?,
    ))
}

/// N-dim SPLIT RoPE — split-RoPE `(cos, sin)` tables over an arbitrary number of position axes
/// (1 for audio/cross-modal, 3 for video). `positions` is `[1, D, T, 2]`; `max_pos`
/// carries one maximum per axis (`len == D`). Each token's per-axis scaled position is
/// `mid/max_pos·2−1`; the padded `dim/2` frequency slots are filled C-major over `(freq_index, axis)`.
pub fn precompute_split_freqs_nd(
    positions: &Tensor,
    dim: usize,
    theta: f64,
    max_pos: &[i32],
    num_heads: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let (_b, n_pos_dims, seq, _two) = positions.dims4()?;
    assert_eq!(
        n_pos_dims,
        max_pos.len(),
        "split-rope: positions axes must match max_pos length"
    );
    let pos = positions.flatten_all()?.to_vec1::<f32>()?;
    let idx = |d: usize, t: usize, e: usize| (d * seq + t) * 2 + e;

    let n_elem = 2 * n_pos_dims;
    let num_indices = (dim / n_elem).max(1);
    let step = if num_indices == 1 {
        0.0
    } else {
        1.0 / (num_indices - 1) as f64
    };
    let indices: Vec<f64> = (0..num_indices)
        .map(|i| theta.powf(i as f64 * step) * (PI / 2.0))
        .collect();

    let current = num_indices * n_pos_dims;
    let expected = dim / 2;
    let pad_size = expected.saturating_sub(current);
    let head_half = expected / num_heads;

    let total = num_heads * seq * head_half;
    let mut cos_out = vec![0f32; total];
    let mut sin_out = vec![0f32; total];
    for t in 0..seq {
        let mut scaled = vec![0f64; n_pos_dims];
        for (d, s) in scaled.iter_mut().enumerate() {
            let start = pos[idx(d, t, 0)] as f64;
            let end = pos[idx(d, t, 1)] as f64;
            let mid = (start + end) / 2.0;
            *s = mid / max_pos[d] as f64 * 2.0 - 1.0;
        }
        for h in 0..num_heads {
            for p in 0..head_half {
                let flat = h * head_half + p;
                let (c, s) = if flat < pad_size {
                    (1.0f32, 0.0f32)
                } else {
                    let k = flat - pad_size;
                    let i = k / n_pos_dims;
                    let d = k % n_pos_dims;
                    let ang = scaled[d] * indices[i];
                    (ang.cos() as f32, ang.sin() as f32)
                };
                let o = (h * seq + t) * head_half + p;
                cos_out[o] = c;
                sin_out[o] = s;
            }
        }
    }
    let shape = (1, num_heads, seq, head_half);
    Ok((
        Tensor::from_vec(cos_out, shape, device)?,
        Tensor::from_vec(sin_out, shape, device)?,
    ))
}

/// Extract the time (frame) axis of a position grid `[1, D, T, 2]` → `[1, 1, T, 2]` (`time_axis`).
pub fn time_axis(grid: &Tensor) -> Result<Tensor> {
    let (_b, _d, t, _two) = grid.dims4()?;
    grid.narrow(1, 0, 1)?.reshape((1, 1, t, 2))
}

/// The audio RoPE position grid `[1, 1, T, 2]` (`[start, end]` timestamps in seconds). For latent
/// frame `t`: `mel = max(t·4 + 1 − 4, 0)` (start) / `(t+1)·4 + 1 − 4` (end); `time = mel · hop /
/// sample_rate` — the causal-aligned mel→second map (`create_audio_position_grid`). Always f32.
pub fn create_audio_position_grid(audio_frames: usize, device: &Device) -> Result<Tensor> {
    use crate::config::{
        AUDIO_HOP_LENGTH, AUDIO_LATENT_DOWNSAMPLE_FACTOR, AUDIO_LATENT_SAMPLE_RATE,
    };
    let df = AUDIO_LATENT_DOWNSAMPLE_FACTOR as f32;
    let hop = AUDIO_HOP_LENGTH as f32;
    let sr = AUDIO_LATENT_SAMPLE_RATE as f32;
    let time = |latent: i64| -> f32 {
        let mel = ((latent as f32) * df + 1.0 - df).max(0.0);
        (mel * hop) / sr
    };
    let t = audio_frames;
    let mut data = vec![0f32; t * 2];
    for f in 0..t {
        data[f * 2] = time(f as i64);
        data[f * 2 + 1] = time(f as i64 + 1);
    }
    Tensor::from_vec(data, (1, 1, t, 2), device)
}
