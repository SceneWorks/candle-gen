//! sc-8523 spike, phase 1 (speed ceiling): is candle's EXISTING int8 GEMM path (`fast_mmq` /
//! `fast_mmvq`, the one sc-7702 turned off for accuracy) actually FASTER than the production
//! dequant-to-dense path and plain bf16 cuBLAS on this GPU? ConvRot's entire premise on the candle
//! lane is int8 tensor-core speed; if the int8 path doesn't win here, rotation has nothing to buy
//! and the spike is a NO-GO without writing any rotation code.
//!
//! Measures, per DiT-representative (M tokens, K in, N out) shape and per quant dtype:
//!   1. `bf16`   — dense bf16 matmul (cuBLAS; the bf16-tier baseline)
//!   2. `dequant`— per-forward `QTensor::dequantize` + dense bf16 matmul (production `QLinear`,
//!                 sc-7702; the plain Q4/Q8-tier baseline)
//!   3. `int8`   — `QMatMul::forward` (activation q8_1-quantized on the fly + int8 MMQ kernels;
//!                 what ConvRot would make safe)
//!
//! fp8 note: candle has NO fp8 GEMM path (`F8E4M3` is storage-only; no cublasLt fp8 wiring), so
//! the epic's "vs fp8" baseline is unavailable on this lane — recorded as a spike finding; the
//! meaningful fight is int8 vs bf16 vs dequant.
//!
//! ```text
//! (vcvars) cargo run --release --example convrot_w8a8_bench -p candle-gen --features cuda
//! ```

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with --features cuda");
}

#[cfg(feature = "cuda")]
fn main() -> candle_gen::candle_core::Result<()> {
    use candle_gen::candle_core::quantized::{GgmlDType, QMatMul, QTensor};
    use candle_gen::candle_core::{DType, Device, Module, Tensor};
    use std::time::Instant;

    let dev = Device::new_cuda(0)?;

    // (label, M tokens, K in, N out) — z-image DiT attn/ffn @1024² (4096 tokens, dim 3840),
    // flux2-dev joint attn (dim 15360), and a small-batch tail case (MMVQ territory).
    let shapes: &[(&str, usize, usize, usize)] = &[
        ("zimage-attn  ", 4096, 3840, 3840),
        ("zimage-ffn   ", 4096, 3840, 10240),
        ("flux2-joint  ", 4096, 15360, 3840),
        ("small-batch  ", 8, 3840, 3840),
    ];
    let dtypes = [GgmlDType::Q4_0, GgmlDType::Q4_1, GgmlDType::Q8_0];

    let time = |f: &mut dyn FnMut() -> candle_gen::candle_core::Result<()>|
        -> candle_gen::candle_core::Result<f64> {
        for _ in 0..3 {
            f()?;
        }
        dev.synchronize()?;
        let t = Instant::now();
        let iters = 20;
        for _ in 0..iters {
            f()?;
        }
        dev.synchronize()?;
        Ok(t.elapsed().as_secs_f64() * 1e3 / iters as f64)
    };

    println!("GPU: {dev:?}   (ms/iter over 20, 3 warmup; TFLOP/s in parens)");
    println!(
        "{:<14} {:>6} {:>6} {:>6}  {:<6} {:>16} {:>16} {:>16}  int8-vs-best-dense",
        "shape", "M", "K", "N", "wq", "bf16", "dequant+mm", "int8 QMatMul"
    );
    for &(label, m, k, n) in shapes {
        // Weight staged on CPU f32 (quantize_onto contract), activation bf16 on the GPU.
        let w_cpu = Tensor::randn(0f32, 1f32, (n, k), &Device::Cpu)?;
        let w_bf16 = w_cpu.to_device(&dev)?.to_dtype(DType::BF16)?;
        let x = Tensor::randn(0f32, 1f32, (m, k), &dev)?.to_dtype(DType::BF16)?;
        let wt = w_bf16.t()?.contiguous()?;

        let flops = 2.0 * m as f64 * k as f64 * n as f64;
        let tf = |ms: f64| flops / (ms * 1e-3) / 1e12;

        let t_bf16 = time(&mut || x.matmul(&wt).map(|_| ()))?;

        for wq_dtype in dtypes {
            let qt = std::sync::Arc::new(QTensor::quantize_onto(&w_cpu, wq_dtype, &dev)?);
            let qmm = QMatMul::from_arc(qt.clone())?;

            // Production QLinear path: dequantize per forward, dense matmul in the activation dtype.
            let t_dq = time(&mut || {
                let wd = qt
                    .dequantize(&dev)?
                    .to_dtype(DType::BF16)?
                    .t()?
                    .contiguous()?;
                x.matmul(&wd).map(|_| ())
            })?;

            // The int8 path (fast_mmq / fast_mmvq): activation q8_1-quantized inside.
            let t_i8 = time(&mut || qmm.forward(&x).map(|_| ()))?;

            let best_dense = t_bf16.min(t_dq);
            println!(
                "{label} {m:>6} {k:>6} {n:>6}  {:<6} {:>8.3} ({:>5.1}) {:>8.3} ({:>5.1}) {:>8.3} ({:>5.1})  {:>5.2}x",
                format!("{wq_dtype:?}"),
                t_bf16, tf(t_bf16),
                t_dq, tf(t_dq),
                t_i8, tf(t_i8),
                best_dense / t_i8
            );
        }
    }
    Ok(())
}
