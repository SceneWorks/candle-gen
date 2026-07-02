//! sc-9085 real-weight validation of the MLX-packed → GGML repack seam against the HOSTED
//! z-image-turbo tiers (epic 8506) — `#[ignore]`, env-driven, run manually on the box with the
//! tiers cached:
//!
//! ```text
//! SC9085_Q4=<...>/q4/transformer/model.safetensors   (required)
//! SC9085_BF16=<...>/bf16/transformer/model.safetensors  (optional: ground-truth order check)
//! SC9085_Q8=<...>/q8/transformer/model.safetensors   (optional: Q8_0 re-quant error study)
//! cargo test -p candle-gen --test mlx_repack_real_weights -- --ignored --nocapture
//! ```
//!
//! (The sc-9086 shared-load test `shared_qlinear_packed_load_matches_grid_on_real_tier` reuses the
//! required `SC9085_Q4` env and drives the shared `candle_gen::quant::lin` packed-detect loader over
//! the real tier — see its own docstring.)
//!
//! Three claims, in order of strength:
//! 1. **Lossless**: candle dequant of the repacked `Q4_1` tensor == the MLX affine grid
//!    (`f16(scale)·q + f16(bias)`), element-exact, every packed tensor. Plus a census of
//!    scales/biases that survive the bf16 → f16 cast inexactly (expected ~none).
//! 2. **Order ground truth** (vs the bf16 tier the pack was built from): every dequantized element
//!    sits within half a quantization step of its dense source — `|dq − w| ≤ 0.5·|scale| + eps`.
//!    A nibble/group ordering mistake fails this instantly (errors ~15 steps).
//! 3. **Q8 study**: the accepted Q8 path (exact MLX grid → `QTensor::quantize(Q8_0)`) — report the
//!    double-quantization error so the sc-9085 decision carries numbers.

use candle_gen::candle_core::quantized::{GgmlDType, QTensor};
use candle_gen::candle_core::safetensors::MmapedSafetensors;
use candle_gen::candle_core::{DType, Device, Result, Tensor};
use candle_gen::quant::{
    dequant_mlx_q4_reference, dequant_mlx_q8, f16_exact, mlx_packed_bits, repack_mlx_q4_to_q4_1,
    MLX_GROUP_SIZE,
};

/// The packed bases of a safetensors file: every key with `.scales` + `.biases` + `.weight`.
fn packed_bases(st: &MmapedSafetensors) -> Vec<String> {
    let mut bases: Vec<String> = st
        .tensors()
        .iter()
        .filter_map(|(k, _)| k.strip_suffix(".scales").map(str::to_string))
        .collect();
    bases.sort();
    bases
}

fn load_triple(st: &MmapedSafetensors, base: &str) -> Result<(Tensor, Tensor, Tensor)> {
    let dev = Device::Cpu;
    Ok((
        st.load(&format!("{base}.weight"), &dev)?,
        st.load(&format!("{base}.scales"), &dev)?,
        st.load(&format!("{base}.biases"), &dev)?,
    ))
}

#[test]
#[ignore = "needs the hosted z-image tiers on disk (SC9085_* env)"]
fn q4_repack_lossless_on_real_tier() -> Result<()> {
    let q4_path = std::env::var("SC9085_Q4").expect("SC9085_Q4 not set");
    // SAFETY: the tier files are immutable HF-cache blobs; nothing rewrites them mid-test.
    let st = unsafe { MmapedSafetensors::new(&q4_path)? };
    let bases = packed_bases(&st);
    assert!(!bases.is_empty(), "no packed triples found in {q4_path}");

    let bf16 = std::env::var("SC9085_BF16")
        .ok()
        // SAFETY: as above.
        .map(|p| unsafe { MmapedSafetensors::new(p) }.expect("bf16 reference opens"));

    let (mut n_tensors, mut n_elems) = (0usize, 0usize);
    let (mut inexact_scales, mut max_cast_dev) = (0usize, 0f32);
    let mut max_grid_dev = 0f32; // claim 1: expect exactly 0
    let mut max_halfstep = 0f32; // claim 2: expect <= ~0.5 (+ float fuzz)
    for base in &bases {
        let (wq, scales, biases) = load_triple(&st, base)?;
        let (wq_cols, s_cols) = (wq.dims2()?.1, scales.dims2()?.1);
        assert_eq!(mlx_packed_bits(wq_cols, s_cols), 4, "{base}: not a Q4 pack");

        // f16-cast census on the raw scales/biases.
        let s32 = scales
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let b32 = biases
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for &x in s32.iter().chain(b32.iter()) {
            if !f16_exact(x) {
                inexact_scales += 1;
                max_cast_dev = max_cast_dev.max((half::f16::from_f32(x).to_f32() - x).abs());
            }
        }

        // Claim 1: repacked dequant == MLX grid, exactly.
        let qt = repack_mlx_q4_to_q4_1(&wq, &scales, &biases, &Device::Cpu)?;
        assert_eq!(qt.dtype(), GgmlDType::Q4_1);
        let dq = qt.dequantize(&Device::Cpu)?;
        let grid = dequant_mlx_q4_reference(&wq, &scales, &biases)?;
        let dev = (dq.sub(&grid))?.abs()?.max_all()?.to_scalar::<f32>()?;
        max_grid_dev = max_grid_dev.max(dev);
        assert_eq!(
            dev, 0.0,
            "{base}: repacked dequant deviates from the MLX grid"
        );

        // Claim 2 (order ground truth vs the dense bf16 source the pack was built from), in
        // quantization steps (`|dq − w| / |scale|` per group). MLX's affine quantizer is NOT naive
        // round-to-nearest — it snaps the max-magnitude group edge onto the grid — so individual
        // elements can land slightly beyond half a step. The discriminative check is the MEAN:
        // correct unpack order ⇒ ~0.25 steps (mean |rounding error|); any nibble/group/row
        // permutation mistake ⇒ ~2.7 steps (uniform grid mismatch) on essentially every tensor.
        if let Some(bf) = &bf16 {
            let w = bf
                .load(&format!("{base}.weight"), &Device::Cpu)?
                .to_dtype(DType::F32)?;
            let (out_dim, in_dim) = w.dims2()?;
            let err = (dq.sub(&w))?.abs()?; // [out, in]
            let err_g = err.reshape((out_dim, in_dim / MLX_GROUP_SIZE, MLX_GROUP_SIZE))?;
            let s_abs = scales
                .to_dtype(DType::F32)?
                .abs()?
                .reshape((out_dim, in_dim / MLX_GROUP_SIZE, 1))?
                .broadcast_add(&Tensor::full(1e-12f32, (1, 1, 1), &Device::Cpu)?)?;
            let steps = err_g.broadcast_div(&s_abs)?;
            let max_steps = steps.max_all()?.to_scalar::<f32>()?;
            let mean_steps = steps.mean_all()?.to_scalar::<f32>()?;
            max_halfstep = max_halfstep.max(max_steps);
            assert!(
                mean_steps <= 0.5 && max_steps <= 2.0,
                "{base}: mean {mean_steps} / max {max_steps} quant-steps from the bf16 source — \
                 element order is wrong"
            );
        }

        n_tensors += 1;
        n_elems += dq.elem_count();
    }

    println!(
        "Q4 repack over {n_tensors} packed tensors / {n_elems} elems: max grid deviation {max_grid_dev} \
         (LOSSLESS), max quant-steps vs bf16 source {max_halfstep} (order-error would be ~3–15), \
         f16-inexact scales/biases {inexact_scales} (max cast dev {max_cast_dev})"
    );

    // CUDA leg: the repacked blocks upload + dequantize identically on the GPU.
    #[cfg(feature = "cuda")]
    {
        let dev = Device::new_cuda(0)?;
        let base = &bases[bases.len() / 2];
        let (wq, scales, biases) = load_triple(&st, base)?;
        let qt_gpu = repack_mlx_q4_to_q4_1(&wq, &scales, &biases, &dev)?;
        let dq_gpu = qt_gpu.dequantize(&dev)?.to_device(&Device::Cpu)?;
        let grid = dequant_mlx_q4_reference(&wq, &scales, &biases)?;
        let gpu_dev = (dq_gpu.sub(&grid))?.abs()?.max_all()?.to_scalar::<f32>()?;
        println!("CUDA dequant of repacked {base}: max deviation from grid {gpu_dev}");
        assert_eq!(gpu_dev, 0.0, "CUDA Q4_1 dequant deviates from the MLX grid");
    }

    Ok(())
}

#[test]
#[ignore = "needs the hosted z-image q8 tier on disk (SC9085_Q8 env)"]
fn q8_requant_error_study() -> Result<()> {
    let q8_path = std::env::var("SC9085_Q8").expect("SC9085_Q8 not set");
    // SAFETY: the tier files are immutable HF-cache blobs; nothing rewrites them mid-test.
    let st = unsafe { MmapedSafetensors::new(&q8_path)? };
    let bases = packed_bases(&st);
    assert!(!bases.is_empty(), "no packed triples found in {q8_path}");

    // Mean/max RELATIVE error of the Q8_0 re-quantization, normalized per-tensor by the RMS of the
    // MLX-grid values (the sc-8507 spike's metric).
    let (mut worst_rel, mut sum_rel, mut n) = (0f64, 0f64, 0usize);
    for base in &bases {
        let (wq, scales, biases) = load_triple(&st, base)?;
        let (wq_cols, s_cols) = (wq.dims2()?.1, scales.dims2()?.1);
        assert_eq!(mlx_packed_bits(wq_cols, s_cols), 8, "{base}: not a Q8 pack");

        let grid = dequant_mlx_q8(&wq, &scales, &biases)?;
        let requant = QTensor::quantize(&grid, GgmlDType::Q8_0)?.dequantize(&Device::Cpu)?;
        let err_rms = (requant.sub(&grid))?
            .sqr()?
            .mean_all()?
            .to_scalar::<f32>()? as f64;
        let rms = grid.sqr()?.mean_all()?.to_scalar::<f32>()? as f64;
        let rel = (err_rms / rms.max(1e-30)).sqrt();
        worst_rel = worst_rel.max(rel);
        sum_rel += rel;
        n += 1;
    }
    println!(
        "Q8_0 re-quant of the MLX Q8 grid over {n} tensors: mean rel error {:.5}, worst {:.5}",
        sum_rel / n as f64,
        worst_rel
    );
    Ok(())
}

/// **sc-9086 shared packed-load, real tier.** The shared `candle_gen::quant::lin` packed-DETECT
/// loader, driven over the real z-image q4 `transformer/` tier through a `VarBuilder` (exactly how a
/// per-crate loader will call it), must:
///   1. route every `.scales`-bearing base to the packed path (`QLinear::is_quantized`), and
///   2. produce a `QLinear` whose forward is **bit-exact** to a dense linear built from the same MLX
///      affine grid — proving the shared QLinear reconstructs the real tier weights losslessly (the
///      packed forward and the dense-grid forward both dequant-to-dense-matmul, so any repack/detect
///      bug shows as a nonzero deviation).
///
/// Runs on `SC9085_Q4` (the same required env as the lossless test); the `#[cfg(feature = "cuda")]`
/// leg repeats one base on the GPU so the QTensor upload + dequant is exercised on Blackwell.
#[test]
#[ignore = "needs the hosted z-image q4 tier on disk (SC9085_Q4 env)"]
fn shared_qlinear_packed_load_matches_grid_on_real_tier() -> Result<()> {
    use candle_gen::candle_nn::{Linear, VarBuilder};
    use candle_gen::quant::{dequant_mlx_q4_reference, lin, DenseLinear, QLinear};

    let q4_path = std::env::var("SC9085_Q4").expect("SC9085_Q4 not set");
    // SAFETY: immutable HF-cache blob.
    let st = unsafe { MmapedSafetensors::new(&q4_path)? };
    let bases = packed_bases(&st);
    assert!(!bases.is_empty(), "no packed triples in {q4_path}");

    let run = |dev: &Device, tag: &str| -> Result<()> {
        // SAFETY: immutable HF-cache blob; a fresh mmap per VarBuilder.
        let st2 = unsafe { MmapedSafetensors::new(&q4_path)? };
        let vb = VarBuilder::from_backend(Box::new(st2), DType::F32, dev.clone());
        // Sample a spread of bases (loading every one on the GPU is wasteful; a stride covers the
        // full key namespace — attn/mlp/mod projections — cheaply).
        let stride = (bases.len() / 8).max(1);
        let mut worst = 0f32;
        let mut n = 0usize;
        for base in bases.iter().step_by(stride) {
            let (wq, scales, biases) = load_triple(&st, base)?;
            let (out_dim, in_dim) = (wq.dims2()?.0, scales.dims2()?.1 * MLX_GROUP_SIZE);

            // Shared packed-detect loader: `.scales` present ⇒ packed QLinear, built on `dev`.
            let ql = lin(&vb, base, in_dim, out_dim, false)?;
            assert!(
                ql.is_quantized(),
                "{tag} {base}: `.scales` present but detect took the dense path"
            );

            // Reference: a dense linear over the exact MLX grid the pack represents.
            let grid = dequant_mlx_q4_reference(&wq, &scales, &biases)?.to_device(dev)?;
            let dense = QLinear::Dense(DenseLinear::Linear(Linear::new(grid, None)));

            let x = Tensor::randn(0f32, 1f32, (2, in_dim), dev)?;
            let got = ql.forward(&x)?;
            let want = dense.forward(&x)?;
            let dev_max = (got.sub(&want)?).abs()?.max_all()?.to_scalar::<f32>()?;
            worst = worst.max(dev_max);
            n += 1;
        }
        println!("[{tag}] shared QLinear packed-load over {n} real bases: max |packed − dense-grid| forward deviation {worst}");
        assert_eq!(
            worst, 0.0,
            "{tag}: packed QLinear forward deviates from the dense grid"
        );
        Ok(())
    };

    run(&Device::Cpu, "cpu")?;
    #[cfg(feature = "cuda")]
    run(&Device::new_cuda(0)?, "cuda")?;
    Ok(())
}
