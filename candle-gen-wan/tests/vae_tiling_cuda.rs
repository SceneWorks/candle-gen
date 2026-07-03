//! Real-weight CUDA validation of the z48 vae22 **spatial-tiling** decode (sc-7111).
//!
//! Loads only the VAE component of a Wan2.2-TI2V-5B snapshot (not the 33 GB DiT), so it is cheap, and
//! exercises the new [`WanVae::decode_tiled`] / [`WanVae::decode_budgeted`] paths on real z48 weights.
//! `#[ignore]`d by default — needs the CUDA backend, a snapshot dir, and the weights:
//!
//! ```text
//! set WAN_SNAPSHOT=C:\Users\…\models--Wan-AI--Wan2.2-TI2V-5B-Diffusers\snapshots\<hash>
//! cargo test -p candle-gen-wan --features cuda --release --test vae_tiling_cuda -- --ignored --nocapture
//! ```
//!
//! NOTE: the z48 decoder has **global per-frame spatial attention** (`MidAttn` softmaxes over all
//! H·W), so spatial tiling is intentionally an *approximation* — each tile attends only within itself
//! and the overlapping trapezoidal blend softens the seams. So `decode_tiled` is **not** bit-exact vs.
//! the single-pass `decode`; this test asserts shape/finiteness/range exactly, the budgeted-routing
//! equivalences exactly, and only a loose closeness bound on tiled-vs-untiled (plus it prints the
//! measured cosine / max-abs-diff for the anchor record).
#![cfg(feature = "cuda")]

use std::path::PathBuf;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::tiling::TilingConfig;
use candle_gen_wan::config::VaeConfig;
use candle_gen_wan::vae::WanVae;

/// Load just the VAE from `$WAN_SNAPSHOT/vae` onto CUDA:0, or `None` if the env var is unset.
fn load_vae() -> Option<(WanVae, Device)> {
    let snap = std::env::var("WAN_SNAPSHOT").ok()?;
    let dev = Device::new_cuda(0).expect("cuda:0");
    let f = PathBuf::from(snap)
        .join("vae")
        .join("diffusion_pytorch_model.safetensors");
    // SAFETY: mmap of a read-only, process-owned weight file resolved from `$WAN_SNAPSHOT`; not
    // mutated behind the mapping — the standard candle loading path.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[f], DType::F32, &dev).unwrap() };
    let vae = WanVae::new(&VaeConfig::ti2v_5b(), vb).unwrap();
    Some((vae, dev))
}

fn assert_finite_and_in_range(t: &Tensor, label: &str) {
    let v = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!(
        v.iter().all(|x| x.is_finite()),
        "{label}: produced non-finite values"
    );
    let (lo, hi) = v
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &x| {
            (lo.min(x), hi.max(x))
        });
    assert!(
        lo >= -1.01 && hi <= 1.01,
        "{label}: out of [-1,1] range: [{lo}, {hi}]"
    );
}

/// Cosine similarity + max-abs-diff between two equally-shaped tensors (on host).
fn agreement(a: &Tensor, b: &Tensor) -> (f32, f32) {
    let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let b = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(a.len(), b.len());
    let (mut dot, mut na, mut nb, mut maxd) = (0f64, 0f64, 0f64, 0f32);
    for (&x, &y) in a.iter().zip(&b) {
        dot += (x * y) as f64;
        na += (x * x) as f64;
        nb += (y * y) as f64;
        maxd = maxd.max((x - y).abs());
    }
    ((dot / (na.sqrt() * nb.sqrt() + 1e-12)) as f32, maxd)
}

#[test]
#[ignore = "needs WAN_SNAPSHOT (a Wan2.2-TI2V-5B snapshot dir) + CUDA weights"]
fn wan_z48_spatial_tiling_decode() {
    let Some((vae, dev)) = load_vae() else {
        eprintln!("WAN_SNAPSHOT unset — skipping");
        return;
    };

    // 640²×5-frame output (40×40 latent, 2 latent frames). 640 > 512 ⇒ a single-pass high-res frame.
    let z = Tensor::randn(0f32, 1f32, (1, 48, 2, 40, 40), &dev).unwrap();

    // Baseline: the streaming single-pass decode.
    let base = vae.decode(&z).unwrap();
    assert_eq!(base.dims(), &[1, 3, 5, 640, 640], "unexpected output shape");
    assert_finite_and_in_range(&base, "decode");

    // Spatial tiling: 256 px tiles (16 latent), 64 px overlap ⇒ several tiles per axis.
    let cfg = TilingConfig::spatial_only(256, 64);
    let tiled = vae.decode_tiled(&z, &cfg).unwrap();
    assert_eq!(tiled.dims(), base.dims(), "tiled shape != baseline");
    assert_finite_and_in_range(&tiled, "decode_tiled");

    let (cos, maxd) = agreement(&base, &tiled);
    eprintln!("[sc-7111] decode vs decode_tiled(256/64): cosine={cos:.5} max_abs_diff={maxd:.4}");
    // Global-attention tiling perturbs the result but the overlapping trapezoidal blend keeps it
    // close. Measured anchor on real TI2V-5B z48 weights (640²×5, 256/64 px tiles): cosine=0.99982,
    // max_abs_diff=0.19 (the diff concentrates at tile seams / high-frequency detail). 0.99 floor
    // leaves margin while still cratering on a blend/offset regression.
    assert!(
        cos > 0.99,
        "tiled decode diverged from baseline: cosine={cos:.5}"
    );

    // Budgeted routing — huge budget ⇒ single-pass ⇒ bit-exact with `decode`.
    std::env::set_var("WAN_VAE_BUDGET_GIB", "100000");
    let big = vae.decode_budgeted(&z).unwrap();
    let (_, maxd_big) = agreement(&base, &big);
    assert!(
        maxd_big < 1e-6,
        "decode_budgeted(huge budget) must equal single-pass decode, max_abs_diff={maxd_big}"
    );

    // Budgeted routing — tight budget ⇒ must tile, stay finite/in-range, and differ from single-pass.
    // 8 GiB (not the old 0.5): the CUDA-calibrated cost model (sc-7148) puts even the smallest spatial
    // tile of this 640²×5 decode at ~3.5 GiB, so 0.5 GiB is genuinely infeasible (SmallestTileExceeds-
    // Budget); 8 GiB forces tiling (single-pass peaks ~35 GiB by the model) while staying fittable.
    std::env::set_var("WAN_VAE_BUDGET_GIB", "8");
    let small = vae.decode_budgeted(&z).unwrap();
    std::env::remove_var("WAN_VAE_BUDGET_GIB");
    assert_eq!(small.dims(), base.dims());
    assert_finite_and_in_range(&small, "decode_budgeted(tiny)");
    let (_, maxd_small) = agreement(&base, &small);
    assert!(
        maxd_small > 0.0,
        "tiny-budget decode_budgeted should have tiled (differ from single-pass)"
    );
}
