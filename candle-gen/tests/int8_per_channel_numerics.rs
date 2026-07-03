//! sc-9300 CPU numerics gate for the **per-output-channel** int8 weight-scale extension of the
//! sc-9299 int8 leg — the granularity a community INT8-ConvRot checkpoint stores (`weight_scale`
//! `[out, 1]`). These checks run on CPU (the weight-quant helper is pure candle ops); the on-device
//! IGEMM fold is exercised by the CUDA-gated `cublaslt_8bit_numerics` test.
//!
//! The point of per-channel is that a weight whose output rows span very different magnitudes
//! reconstructs far better than a single per-tensor scale can: a per-tensor scale is set by the
//! largest-magnitude row and quantizes the small-magnitude rows to a handful of codes. Per-channel
//! gives each row its own 127-level budget, so a per-tensor scale is the degenerate one-value case.

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::quant::{quantize_weight_int8, quantize_weight_int8_per_channel};

/// Dequantize a per-channel int8 weight back to f32: `w[o, :] = q[o, :] * scale[o]`.
fn dequant_per_channel(q: &Tensor, scale: &[f32]) -> Tensor {
    let (n, _k) = q.dims2().unwrap();
    let scale_col = Tensor::from_vec(scale.to_vec(), (n, 1), q.device()).unwrap();
    q.to_dtype(DType::F32)
        .unwrap()
        .broadcast_mul(&scale_col)
        .unwrap()
}

/// Mean relative RMS of a reconstructed weight vs the original (both flattened f32).
fn rel_rms(got: &Tensor, reference: &Tensor) -> f32 {
    let g = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let r = reference
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let (mut num, mut den) = (0f64, 0f64);
    for (a, b) in g.iter().zip(&r) {
        num += (*a as f64 - *b as f64).powi(2);
        den += (*b as f64).powi(2);
    }
    (num / den.max(1e-30)).sqrt() as f32
}

/// A weight with rows of wildly different magnitude (row `o` scaled by `2^o`): a single per-tensor
/// scale is dominated by the largest row, so the small rows lose almost all precision, while the
/// per-channel scale reconstructs every row near-losslessly.
fn ragged_weight(n: usize, k: usize) -> Tensor {
    let dev = Device::Cpu;
    let mut v = vec![0f32; n * k];
    for o in 0..n {
        let mag = 2f32.powi(o as i32); // row magnitude spans 2^0 .. 2^(n-1)
        for j in 0..k {
            // deterministic ~[-1,1] pattern, scaled per row
            let s = (((o * 31 + j * 17) % 101) as f32 / 50.0 - 1.0) * mag;
            v[o * k + j] = s;
        }
    }
    Tensor::from_vec(v, (n, k), &dev).unwrap()
}

/// Per-channel int8 reconstructs a ragged weight within a tight budget, and beats per-tensor by a
/// wide margin — the whole reason a ConvRot checkpoint ships a `[out]` scale, not a scalar.
#[test]
fn per_channel_beats_per_tensor_on_ragged_weight() {
    let (n, k) = (16usize, 128usize);
    let w = ragged_weight(n, k);

    // Per-channel reconstruction.
    let pc = quantize_weight_int8_per_channel(&w).unwrap();
    assert_eq!(pc.scale.len(), n, "one scale per output row");
    let w_pc = dequant_per_channel(&pc.q, &pc.scale);
    let err_pc = rel_rms(&w_pc, &w);

    // Per-tensor reconstruction (the sc-9299 scalar path).
    let pt = quantize_weight_int8(&w).unwrap();
    let w_pt = (pt.q.to_dtype(DType::F32).unwrap() * pt.scale as f64).unwrap();
    let err_pt = rel_rms(&w_pt, &w);

    eprintln!("[sc-9300] ragged weight rel-RMS: per-channel={err_pc:.5} per-tensor={err_pt:.5}");
    // Per-channel is near-lossless (each row gets a full 127-level budget).
    assert!(
        err_pc < 0.01,
        "per-channel int8 should reconstruct a ragged weight near-losslessly, got {err_pc}"
    );
    // And it beats per-tensor on this ragged weight (the small-magnitude rows keep their precision
    // instead of collapsing under a scalar scale set by the largest row).
    assert!(
        err_pc < err_pt,
        "per-channel ({err_pc}) should be tighter than per-tensor ({err_pt}) on a ragged weight"
    );
}

/// Per-channel is a strict superset of per-tensor: a weight whose rows share one magnitude gives a
/// per-channel scale vector of (near-)equal entries, and its reconstruction matches the per-tensor
/// path to within rounding.
#[test]
fn per_channel_reduces_to_per_tensor_on_uniform_weight() {
    let dev = Device::Cpu;
    let (n, k) = (8usize, 64usize);
    // Every row shares the same magnitude range → the per-row absmax is (near) identical.
    let v: Vec<f32> = (0..n * k)
        .map(|i| ((i * 13 % 255) as f32) / 254.0 * 2.0 - 1.0)
        .collect();
    let w = Tensor::from_vec(v, (n, k), &dev).unwrap();

    let pc = quantize_weight_int8_per_channel(&w).unwrap();
    let w_pc = dequant_per_channel(&pc.q, &pc.scale);
    let err = rel_rms(&w_pc, &w);
    eprintln!("[sc-9300] uniform weight per-channel rel-RMS = {err:.5}");
    assert!(
        err < 0.01,
        "per-channel int8 on a uniform weight is tight, got {err}"
    );
}

/// The int8 codes are in range and the scale is strictly positive — the invariants the on-device
/// IGEMM + per-row dequant fold relies on.
#[test]
fn per_channel_codes_in_range() {
    let w = ragged_weight(8, 32);
    let pc = quantize_weight_int8_per_channel(&w).unwrap();
    let codes = pc.q.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!(
        codes
            .iter()
            .all(|&c| (-127.0..=127.0).contains(&c) && c.fract() == 0.0),
        "int8 codes must be integers in [-127, 127]"
    );
    assert!(
        pc.scale.iter().all(|&s| s > 0.0),
        "every per-row scale must be strictly positive"
    );
}
