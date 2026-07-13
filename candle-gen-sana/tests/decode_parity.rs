//! DC-AE decoder **parity** gate vs the mlx-gen #612 / diffusers `AutoencoderDC` reference (spike
//! sc-11777 GO/NO-GO).
//!
//! `#[ignore]`d: needs the real `dc-ae-f32c32-sana-1.0` weights (~1.25 GB) and a golden
//! (`latent` + reference raw-decoder `image`) produced by the same `dump_dcae_golden.py` the mlx port
//! (sc-8486) used. This decodes the SAME latent through the candle port and checks it reproduces the
//! reference output. The mlx port hit `mean_rel ≈ 0.005` single-pass; this gate holds the candle port
//! to the same ballpark (candle CUDA f32 matmul carries a similar reduced-precision floor).
//!
//! Run (CUDA):
//!   SANA_DCAE_WEIGHTS=/path/diffusion_pytorch_model.safetensors \
//!   SANA_DCAE_GOLDEN=/path/dcae_golden.safetensors \
//!   cargo test -p candle-gen-sana --test decode_parity --features cuda --release -- --ignored --nocapture

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::Weights;
use candle_gen_sana::{DcAeConfig, DcAeDecoder};

fn device() -> Device {
    #[cfg(feature = "cuda")]
    {
        Device::new_cuda(0).expect("cuda device")
    }
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        Device::new_metal(0).expect("metal device")
    }
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    {
        Device::Cpu
    }
}

/// `max|Δ| / max|ref|`.
fn peak_rel(got: &Tensor, want: &Tensor) -> f32 {
    let diff = (got - want)
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    let denom = want
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    diff / denom.max(1e-12)
}

/// `Σ|Δ| / Σ|ref|`.
fn mean_rel(got: &Tensor, want: &Tensor) -> f32 {
    let num = (got - want)
        .unwrap()
        .abs()
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    let den = want
        .abs()
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    num / den.max(1e-12)
}

#[test]
#[ignore = "needs dc-ae-f32c32-sana-1.0 weights + dump_dcae_golden.py golden"]
fn decode_matches_reference() {
    let dev = device();
    let weights_path = std::env::var("SANA_DCAE_WEIGHTS").expect("set SANA_DCAE_WEIGHTS");
    let golden_path = std::env::var("SANA_DCAE_GOLDEN").expect("set SANA_DCAE_GOLDEN");

    let golden = Weights::from_file(std::path::Path::new(&golden_path), &dev, DType::F32)
        .expect("load golden");
    let latent = golden.require("latent").expect("golden latent"); // [1,32,32,32] NCHW
    let want = golden.require("image").expect("golden image"); // [1,3,1024,1024] NCHW

    let weights = Weights::from_file(std::path::Path::new(&weights_path), &dev, DType::F32)
        .expect("load weights");
    let decoder = DcAeDecoder::from_weights(&weights, DcAeConfig::sana_f32c32()).expect("build");
    let got = decoder.decode(&latent).expect("decode"); // [1,3,1024,1024] NCHW

    assert_eq!(got.dims(), want.dims(), "shape");
    let peak = peak_rel(&got, &want);
    let mean = mean_rel(&got, &want);
    println!("DC-AE candle decode parity vs mlx-gen #612 reference: peak_rel={peak:.5}  mean_rel={mean:.5}");

    // Dense f32 path. `mean_rel` is the faithfulness gate — a port bug (wrong transpose/op order/
    // layout) wrecks the mean; reduced-precision matmul rounding does not. The mlx port measured
    // mean_rel ≈ 0.005; the candle CUDA f32 path should land in the same ballpark.
    assert!(
        mean < 1e-2,
        "mean_rel {mean} too high — that IS a port bug, not rounding"
    );
    // `peak_rel` is looser by design (the linear-attention 1/(Σ+eps) normalizer amplifies a handful
    // of interior pixels' reduced-precision matmul noise); mlx set this ceiling at 0.10.
    assert!(
        peak < 0.10,
        "peak_rel {peak} above the attention-normalizer precision ceiling"
    );
}
