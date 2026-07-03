//! sc-7297 / epic 7114 — curated-sampler smoke for the candle Kolors **conditioned** sub-providers
//! (the Windows/CUDA twin of `mlx-gen-kolors/tests/conditioned_curated_smoke.rs`).
//!
//! Validates that routing the Kolors ControlNet-pose ([`KolorsControl`]) and IP-Adapter
//! ([`IpAdapterKolors`]) denoises through the curated k-diffusion path (the shared
//! `candle_gen_sdxl::denoise_curated`, threading the ControlNet residuals / IP decoupled-attn tokens)
//! renders coherently — i.e. a curated solver does NOT destabilize the strong conditioning these modes
//! were originally fixed-sampler-locked for.
//!
//! Unlike the mlx provider (one registered generator dispatching every mode, incl. a combined-pose
//! tier), the candle Kolors conditioned modes are SEPARATE struct APIs the worker drives directly, so
//! this drives them directly too — there is no combined-pose tier to cover.
//!
//! Runs against the REAL weights (no torch, no golden artifacts). Run (PowerShell, MSVC vcvars + CUDA):
//!
//! ```text
//! $env:KOLORS_SNAPSHOT   = "<Kolors-diffusers snapshot dir>"
//! $env:KOLORS_CONTROLNET = "<Kolors-ControlNet-Pose snapshot dir or .safetensors>"
//! $env:KOLORS_IP_ADAPTER = "<Kolors-IP-Adapter-Plus snapshot dir>"
//! cargo test -p candle-gen-kolors --features cuda --release --test conditioned_curated_smoke -- --ignored --nocapture
//! ```
//!
//! Gate (directional): for each conditioned mode and each curated solver —
//!   (1) the render is **coherent** (not collapsed to noise/flat — the destabilization failure mode), and
//!   (2) it **differs** from the bespoke `euler_discrete` default (a real solver swap, not a silent no-op).
//! The default path itself stays byte-exact (covered by the unit gates); here we prove the curated route
//! is wired through the conditioning and behaves.

use candle_gen::gen_core::{Image, Progress};
use candle_gen::testkit::env_path_opt as env_path;
use candle_gen_kolors::{
    IpAdapterKolors, IpAdapterKolorsPaths, IpAdapterKolorsRequest, KolorsControl,
    KolorsControlPaths, KolorsControlRequest,
};

const SIZE: u32 = 512;
const STEPS: usize = 8;
const SEED: u64 = 7;
const PROMPT: &str = "a person standing in a sunlit park, photorealistic, sharp focus, high detail";
const NEGATIVE: &str = "lowres, blurry, deformed, disfigured, cartoon, painting";

/// A deterministic synthetic RGB8 image (pose skeleton / reference stand-in). The conditioning
/// *adherence* numbers would only be meaningful with real images, but the coherence + distinctness gate
/// holds either way (it gates the solver's stability, not adherence).
fn synthetic_image() -> Image {
    let (h, w) = (SIZE as usize, SIZE as usize);
    let mut px = vec![0u8; h * w * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            px[i] = (x * 255 / (w - 1)) as u8;
            px[i + 1] = (y * 255 / (h - 1)) as u8;
            px[i + 2] = ((x ^ y) % 256) as u8;
        }
    }
    Image {
        width: w as u32,
        height: h as u32,
        pixels: px,
    }
}

/// (std, distinct-level-count, mean horizontal-adjacent-|Δ|) over an RGB8 buffer.
fn image_stats(px: &[u8], w: u32) -> (f32, usize, f32) {
    let n = px.len() as f64;
    let mean = px.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = px.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
    let mut seen = [false; 256];
    for &v in px {
        seen[v as usize] = true;
    }
    let distinct = seen.iter().filter(|&&b| b).count();
    let stride = (w * 3) as usize;
    let (mut adj_sum, mut adj_n) = (0f64, 0u64);
    for (i, &v) in px.iter().enumerate() {
        if i >= 3 && i % stride >= 3 {
            adj_sum += (v as i32 - px[i - 3] as i32).unsigned_abs() as f64;
            adj_n += 1;
        }
    }
    (
        var.sqrt() as f32,
        distinct,
        (adj_sum / adj_n.max(1) as f64) as f32,
    )
}

fn assert_coherent(img: &Image, label: &str) {
    assert_eq!(
        img.pixels.len(),
        (img.width * img.height * 3) as usize,
        "{label}: pixel buffer is RGB8 HWC"
    );
    let (std, distinct, adj) = image_stats(&img.pixels, img.width);
    println!("[smoke] {label:<32} std={std:.1} distinct={distinct} adjΔ={adj:.1}");
    assert!(std > 10.0, "{label}: image is near-flat (std {std:.1})");
    assert!(
        distinct > 24,
        "{label}: too few distinct levels ({distinct})"
    );
    assert!(
        adj < 60.0,
        "{label}: not spatially smooth — looks like destabilized noise (adjΔ {adj:.1})"
    );
}

fn frac_diff(a: &Image, b: &Image) -> f32 {
    let differ = a
        .pixels
        .iter()
        .zip(&b.pixels)
        .filter(|(x, y)| (**x as i16 - **y as i16).abs() > 4)
        .count();
    differ as f32 / a.pixels.len() as f32
}

/// Gate one curated render against the bespoke default: coherent + a real swap (distinct from default).
fn assert_curated(default: &Image, img: &Image, tag: &str) {
    assert_coherent(img, tag);
    let frac = frac_diff(default, img);
    println!(
        "[smoke] {tag:<32} differs from default in {:.2}% of pixels",
        frac * 100.0
    );
    assert!(
        frac > 0.01,
        "{tag}: curated solver must differ from the euler_discrete default (real swap); differ {frac}"
    );
}

/// ControlNet-pose, curated: the pose branch residuals thread the curated solver.
#[test]
#[ignore = "needs Kolors-diffusers + Kolors-ControlNet-Pose snapshots (set KOLORS_SNAPSHOT/KOLORS_CONTROLNET) + CUDA"]
fn controlnet_curated_is_coherent_and_distinct() {
    let (Some(base), Some(cn)) = (env_path("KOLORS_SNAPSHOT"), env_path("KOLORS_CONTROLNET"))
    else {
        eprintln!("skipping: set KOLORS_SNAPSHOT + KOLORS_CONTROLNET");
        return;
    };
    let model = KolorsControl::load(&KolorsControlPaths {
        kolors_base: base,
        controlnet: cn,
    })
    .expect("load KolorsControl");
    let pose = synthetic_image();

    let req = |sampler: Option<&str>| KolorsControlRequest {
        prompt: PROMPT.into(),
        negative: NEGATIVE.into(),
        width: SIZE,
        height: SIZE,
        steps: STEPS,
        guidance: 5.0,
        control_scale: 0.7,
        sampler: sampler.map(Into::into),
        scheduler: None,
        seed: SEED,
        ..Default::default()
    };
    let mut noop = |_p: Progress| {};

    let default = model
        .generate(&req(None), &pose, &mut noop)
        .expect("default generate");
    assert_coherent(&default, "controlnet/default(euler_discrete)");
    for sampler in ["euler", "heun", "dpmpp_2m"] {
        let img = model
            .generate(&req(Some(sampler)), &pose, &mut noop)
            .expect("curated generate");
        assert_curated(&default, &img, &format!("controlnet/{sampler}"));
    }
}

/// IP-Adapter, curated: the decoupled-attn image tokens thread the curated solver.
#[test]
#[ignore = "needs Kolors-diffusers + Kolors-IP-Adapter-Plus snapshots (set KOLORS_SNAPSHOT/KOLORS_IP_ADAPTER) + CUDA"]
fn ip_adapter_curated_is_coherent_and_distinct() {
    let (Some(base), Some(ip)) = (env_path("KOLORS_SNAPSHOT"), env_path("KOLORS_IP_ADAPTER"))
    else {
        eprintln!("skipping: set KOLORS_SNAPSHOT + KOLORS_IP_ADAPTER");
        return;
    };
    let mut model = IpAdapterKolors::load(&IpAdapterKolorsPaths {
        kolors_base: base,
        ip_adapter: ip,
    })
    .expect("load IpAdapterKolors");
    let reference = synthetic_image();

    let req = |sampler: Option<&str>| IpAdapterKolorsRequest {
        prompt: PROMPT.into(),
        negative: NEGATIVE.into(),
        width: SIZE,
        height: SIZE,
        steps: STEPS,
        guidance: 5.0,
        ip_adapter_scale: 0.6,
        sampler: sampler.map(Into::into),
        scheduler: None,
        seed: SEED,
        ..Default::default()
    };
    let mut noop = |_p: Progress| {};

    let default = model
        .generate(&req(None), &reference, &mut noop)
        .expect("default generate");
    assert_coherent(&default, "ip_adapter/default(euler_discrete)");
    for sampler in ["euler", "heun", "dpmpp_2m"] {
        let img = model
            .generate(&req(Some(sampler)), &reference, &mut noop)
            .expect("curated generate");
        assert_curated(&default, &img, &format!("ip_adapter/{sampler}"));
    }
}
