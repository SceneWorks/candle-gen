//! sc-9299 numerics gate for the cuBLASLt 8-bit GEMM leg. CUDA-only (the wrapper needs a live Lt
//! handle); a graceful no-op on CPU/Metal builds. Two checks the story calls out:
//!
//! 1. **Parity vs f32** for well-conditioned inputs — fp8 and int8 both reconstruct `X·Wᵀ` within
//!    their per-tensor-quant error budget (fp8 ≈ a few %, int8 tighter for in-range data).
//! 2. **The sc-7702 activation-outlier scenario** — one feature blown up to ±1e4 (the gpt-oss
//!    outlier). Per-tensor int8 sets the whole scale off that single value and zeros the co-located
//!    in-range channels → large error. This is EXPECTED to degrade WITHOUT rotation; it is the
//!    motivation for the ConvRot consume story (sc-9300), recorded here, not "fixed".

#![cfg(feature = "cuda")]

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::quant::cublaslt::CublasLt;

/// splitmix64-hashed deterministic pseudo-random in ~[-1, 1] (launch-portable; no device RNG).
fn pseudo_random(n: usize, seed: u64) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let mut z = (i as u64)
                .wrapping_add(seed)
                .wrapping_add(0x9E37_79B9_7F4A_7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            (z as f64 / u64::MAX as f64) as f32 * 2.0 - 1.0
        })
        .collect()
}

/// Mean relative RMS error of `got` vs `reference` (both flattened f32).
fn rel_rms(got: &[f32], reference: &[f32]) -> f32 {
    let (mut num, mut den) = (0f64, 0f64);
    for (g, r) in got.iter().zip(reference) {
        num += (*g as f64 - *r as f64).powi(2);
        den += (*r as f64).powi(2);
    }
    (num / den.max(1e-30)).sqrt() as f32
}

fn f32_reference(x: &Tensor, w: &Tensor) -> Vec<f32> {
    // (M,K) · (N,K)ᵀ = (M,N)
    let y = x.matmul(&w.t().unwrap().contiguous().unwrap()).unwrap();
    y.to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

#[test]
fn wellconditioned_parity_vs_f32() {
    let dev = match Device::cuda_if_available(0) {
        Ok(d @ Device::Cuda(_)) => d,
        _ => {
            eprintln!("[sc-9299] no CUDA device; skipping numerics gate");
            return;
        }
    };
    let lt = CublasLt::new(&dev).unwrap();
    eprintln!(
        "[sc-9299] device compute cap = {:?}",
        lt.compute_cap().unwrap()
    );

    // Small, 16-aligned DiT-ish shape.
    let (m, k, n) = (32usize, 256usize, 256usize);
    let x = Tensor::from_vec(pseudo_random(m * k, 1), (m, k), &dev).unwrap();
    let w = Tensor::from_vec(pseudo_random(n * k, 2), (n, k), &dev).unwrap();
    let reference = f32_reference(&x, &w);

    // fp8 path.
    let qw = candle_gen::quant::quantize_weight_fp8(&w).unwrap();
    let qx = candle_gen::quant::quantize_activation_fp8(&x).unwrap();
    let y_fp8 = lt
        .matmul_fp8(&qw.q, qw.scale, &qx.q, qx.scale)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let err_fp8 = rel_rms(&y_fp8, &reference);
    eprintln!("[sc-9299] fp8  rel-RMS vs f32 = {err_fp8:.4}");

    // int8 path.
    let qwi = candle_gen::quant::quantize_weight_int8(&w).unwrap();
    let qxi = candle_gen::quant::quantize_activation_int8(&x).unwrap();
    let y_i8 = lt
        .matmul_int8(&qwi.q, qwi.scale, &qxi.q, qxi.scale)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let err_i8 = rel_rms(&y_i8, &reference);
    eprintln!("[sc-9299] int8 rel-RMS vs f32 = {err_i8:.4}");

    // Per-tensor 8-bit on well-conditioned data lands well under 10% for both. fp8 E4M3 (3 mantissa
    // bits) is the looser of the two here; int8 (7-bit magnitude) is tighter for in-range data.
    assert!(err_fp8 < 0.10, "fp8 rel-RMS too high: {err_fp8}");
    assert!(err_i8 < 0.10, "int8 rel-RMS too high: {err_i8}");
}

#[test]
fn int8_per_channel_parity_vs_f32() {
    let dev = match Device::cuda_if_available(0) {
        Ok(d @ Device::Cuda(_)) => d,
        _ => {
            eprintln!("[sc-9300] no CUDA device; skipping per-channel numerics gate");
            return;
        }
    };
    let lt = CublasLt::new(&dev).unwrap();

    // A weight whose output rows span very different magnitudes — the case per-channel exists for.
    let (m, k, n) = (32usize, 256usize, 256usize);
    let x = Tensor::from_vec(pseudo_random(m * k, 5), (m, k), &dev).unwrap();
    let mut wv = pseudo_random(n * k, 6);
    for row in 0..n {
        let mag = 1.0 + (row as f32) * 0.5; // rows 1x .. ~128x
        for j in 0..k {
            wv[row * k + j] *= mag;
        }
    }
    let w = Tensor::from_vec(wv, (n, k), &dev).unwrap();
    let reference = f32_reference(&x, &w);

    // Per-channel int8 weight quant (the ConvRot granularity), on-device IGEMM + per-row dequant.
    let qw = candle_gen::quant::quantize_weight_int8_per_channel(&w).unwrap();
    let qx = candle_gen::quant::quantize_activation_int8(&x).unwrap();
    let y_pc = lt
        .matmul_int8_per_channel(&qw.q, &qw.scale, &qx.q, qx.scale)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let err_pc = rel_rms(&y_pc, &reference);

    // Per-TENSOR int8 on the SAME ragged weight — the scalar scale is dominated by the largest row,
    // so the small-magnitude rows lose precision and the matmul error is materially worse.
    let qw_pt = candle_gen::quant::quantize_weight_int8(&w).unwrap();
    let y_pt = lt
        .matmul_int8(&qw_pt.q, qw_pt.scale, &qx.q, qx.scale)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let err_pt = rel_rms(&y_pt, &reference);

    eprintln!("[sc-9300] int8 X·Wᵀ rel-RMS vs f32: per-channel={err_pc:.4} per-tensor={err_pt:.4}");
    // Per-channel lands well under 10% on the ragged weight (per-tensor does not).
    assert!(err_pc < 0.10, "per-channel int8 rel-RMS too high: {err_pc}");
    assert!(
        err_pc < err_pt,
        "per-channel ({err_pc}) must beat per-tensor ({err_pt}) on a magnitude-ragged weight"
    );
}

/// sc-9601 perf: the **on-device** per-channel dequant (`matmul_int8_per_channel_staged_ondevice`,
/// int32 IGEMM + on-device `i32 → f32` cast + candle float fold) must match the exact int32→host fold
/// (`matmul_int8_per_channel_staged`) — same numbers, no host round-trip. Also asserts the on-device
/// path is actually available (the vendored `cast_i32_f32` kernel is present), so the fast path is the
/// one that runs, not a silent fallback to the host fold.
#[test]
fn int8_per_channel_ondevice_matches_host_fold() {
    let dev = match Device::cuda_if_available(0) {
        Ok(d @ Device::Cuda(_)) => d,
        _ => {
            eprintln!("[sc-9601] no CUDA device; skipping on-device dequant gate");
            return;
        }
    };
    let lt = CublasLt::new(&dev).unwrap();
    assert!(
        lt.supports_ondevice_int8_dequant(),
        "sc-9601: this device must cast i32→f32 on-device (vendored cast_i32_f32) for the fast path"
    );

    let (m, k, n) = (48usize, 512usize, 384usize);
    let x = Tensor::from_vec(pseudo_random(m * k, 11), (m, k), &dev).unwrap();
    let mut wv = pseudo_random(n * k, 12);
    for row in 0..n {
        let mag = 1.0 + (row as f32) * 0.25;
        for j in 0..k {
            wv[row * k + j] *= mag;
        }
    }
    let w = Tensor::from_vec(wv, (n, k), &dev).unwrap();

    let qw = candle_gen::quant::quantize_weight_int8_per_channel(&w).unwrap();
    let qx = candle_gen::quant::quantize_activation_int8(&x).unwrap();
    let staged = lt.stage_int8(&qw.q).unwrap();

    let y_host = lt
        .matmul_int8_per_channel_staged(&staged, &qw.scale, &qx.q, qx.scale)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let y_dev = lt
        .matmul_int8_per_channel_staged_ondevice(&staged, &qw.scale, &qx.q, qx.scale)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    let err = rel_rms(&y_dev, &y_host);
    eprintln!("[sc-9601] on-device vs host int8 fold rel-RMS = {err:.2e} (both bf16 output)");
    // Both fold the SAME exact int32 accumulate (one on host, one via the f32-output epilogue) and cast
    // to bf16 — they must agree to bf16 rounding (~1e-3), not merely "close".
    assert!(
        err < 5e-3,
        "on-device dequant must match the host fold to bf16 rounding, got rel-RMS {err}"
    );
}

#[test]
fn int8_degrades_on_activation_outlier_no_rotation() {
    let dev = match Device::cuda_if_available(0) {
        Ok(d @ Device::Cuda(_)) => d,
        _ => {
            eprintln!("[sc-9299] no CUDA device; skipping outlier gate");
            return;
        }
    };
    let lt = CublasLt::new(&dev).unwrap();

    let (m, k, n) = (32usize, 256usize, 256usize);
    let base = pseudo_random(m * k, 3); // the in-range signal (~±1)
    let mut xv = base.clone();
    // sc-7702 scenario: a single feature blown up to ±1e4 (the gpt-oss activation outlier), in
    // column 0 of every token row so the per-tensor activation scale is dominated by it.
    for row in 0..m {
        xv[row * k] = if row % 2 == 0 { 1.0e4 } else { -1.0e4 };
    }
    // The in-range-only activation: outlier column zeroed. This is the co-located signal a correct
    // (rotated) path preserves — the contribution per-tensor int8 is expected to annihilate.
    let mut inrange = base.clone();
    for row in 0..m {
        inrange[row * k] = 0.0;
    }

    let x = Tensor::from_vec(xv, (m, k), &dev).unwrap();
    let x_inrange = Tensor::from_vec(inrange, (m, k), &dev).unwrap();
    // The outlier-only activation (everything but column 0 zeroed) — the part int8 DOES keep.
    let mut only_out = vec![0f32; m * k];
    for row in 0..m {
        only_out[row * k] = if row % 2 == 0 { 1.0e4 } else { -1.0e4 };
    }
    let x_only_outlier = Tensor::from_vec(only_out, (m, k), &dev).unwrap();
    let w = Tensor::from_vec(pseudo_random(n * k, 4), (n, k), &dev).unwrap();

    // True in-range contribution: what the co-located features SHOULD add to the output.
    let true_inrange = f32_reference(&x_inrange, &w);
    // The pure outlier-column contribution (what int8 preserves faithfully).
    let outlier_contrib = f32_reference(&x_only_outlier, &w);

    let qwi = candle_gen::quant::quantize_weight_int8(&w).unwrap();
    let qxi = candle_gen::quant::quantize_activation_int8(&x).unwrap();
    // absmax/127 ≈ 1e4/127 ≈ 79 — every in-range (±1) feature rounds to 0 after dividing by ~79.
    eprintln!(
        "[sc-9299] int8 activation scale under outlier = {:.2} (in-range features round to 0)",
        qxi.scale
    );
    let y_i8 = lt
        .matmul_int8(&qwi.q, qwi.scale, &qxi.q, qxi.scale)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    // Recovered in-range contribution = int8 output − the outlier-column contribution. Because the
    // in-range channels quantized to 0, this residual is ~0, NOT the true in-range signal → the
    // relative error against `true_inrange` is ~100%. That collapse is the sc-7702 failure and the
    // whole reason ConvRot (sc-9300) rotates the outlier energy across channels before quantizing.
    let recovered_inrange: Vec<f32> = y_i8
        .iter()
        .zip(&outlier_contrib)
        .map(|(y, o)| y - o)
        .collect();
    let err = rel_rms(&recovered_inrange, &true_inrange);
    eprintln!("[sc-9299] int8 in-range rel-RMS under activation outlier = {err:.4} (EXPECTED ~1.0 — in-range signal annihilated; motivates sc-9300 ConvRot)");

    // Assert the DEGRADATION. If this ever came back small, per-tensor int8 would be outlier-safe
    // and the ConvRot motivation would be void. Recorded, not fixed.
    assert!(
        err > 0.5,
        "expected int8 to annihilate the in-range signal under the sc-7702 outlier (got rel-RMS \
         {err}); if this fails, per-tensor int8 is outlier-safe and ConvRot is unnecessary"
    );
}
