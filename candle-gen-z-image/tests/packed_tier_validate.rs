//! Z-Image **packed-tier** real-weight GPU validation (sc-9408, sc-9089 umbrella).
//!
//! Loads the pre-quantized MLX-packed `SceneWorks/z-image-turbo-mlx` q4 (and, when present, q8) tier
//! **directly from the packed parts** (no dense bf16 staging) through the registered `z_image_turbo`
//! generator, and asserts it renders a **coherent, non-degenerate** image — the end-to-end proof that
//! the packed-detect path fired (a silent fall-back to dense would fail to load the u32-packed weights,
//! and a broken packed forward would render solid black / NaN).
//!
//! `#[ignore]`d (needs a real GPU + the cached packed tier). On the Windows/Blackwell box (v143 vcvars
//! + CUDA on PATH), point at the **tier subdir** (the packed snapshot nests `bf16/`, `q4/`, `q8/`):
//!
//! ```text
//! set Z_IMAGE_PACKED_Q4=D:\.cache\huggingface\hub\models--SceneWorks--z-image-turbo-mlx\snapshots\<hash>\q4
//! set Z_IMAGE_PACKED_Q8=...\q8    (optional)
//! cargo test -p candle-gen-z-image --features cuda --release --test packed_tier_validate -- --ignored --nocapture
//! ```
#![cfg(feature = "cuda")]

use std::path::PathBuf;
use std::time::Instant;

use candle_gen::gen_core::{
    self, GenerationOutput, GenerationRequest, Image, LoadSpec, Progress, WeightsSource,
};

/// Basic non-degeneracy: the render is not solid-black / constant (a broken packed forward — NaN or
/// zeroed activations — decodes to a flat image), and has some spread of pixel values.
fn assert_coherent(img: &Image, tag: &str) {
    assert_eq!(
        img.pixels.len(),
        (img.width * img.height * 3) as usize,
        "{tag}: RGB buffer size mismatch"
    );
    let min = *img.pixels.iter().min().unwrap();
    let max = *img.pixels.iter().max().unwrap();
    let mean = img.pixels.iter().map(|&p| p as f64).sum::<f64>() / img.pixels.len() as f64;
    let var = img
        .pixels
        .iter()
        .map(|&p| (p as f64 - mean).powi(2))
        .sum::<f64>()
        / img.pixels.len() as f64;
    eprintln!(
        "[{tag}] {}x{} pixel min={min} max={max} mean={mean:.1} std={:.1}",
        img.width,
        img.height,
        var.sqrt()
    );
    assert!(
        max > min + 16,
        "{tag}: render is (near-)constant [{min}, {max}] — packed forward likely degenerate (black)"
    );
    assert!(
        var.sqrt() > 8.0,
        "{tag}: pixel std {:.1} too low — degenerate render",
        var.sqrt()
    );
}

fn render_tier(env: &str, tag: &str) {
    let Ok(dir) = std::env::var(env) else {
        eprintln!("SKIP {tag}: set {env} to the packed tier subdir");
        return;
    };
    candle_gen_z_image::force_link();
    let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from(&dir)));
    let gen = gen_core::registry::load("z_image_turbo", &spec).expect("z_image_turbo registered");

    let req = GenerationRequest {
        prompt: "a photo of a rusty robot holding a lit candle, cinematic lighting".into(),
        width: 512,
        height: 512,
        count: 1,
        seed: Some(42),
        steps: Some(4),
        ..Default::default()
    };

    let t = Instant::now();
    let mut on_progress = |_p: Progress| {};
    let out = gen
        .generate(&req, &mut on_progress)
        .unwrap_or_else(|e| panic!("{tag}: packed-tier generate failed: {e}"));
    let secs = t.elapsed().as_secs_f32();
    eprintln!("[{tag}] load+render wall-clock {secs:.1}s (cold: includes packed load)");

    let images = match out {
        GenerationOutput::Images(imgs) => imgs,
        GenerationOutput::Video { .. } => panic!("{tag}: expected images"),
    };
    assert_eq!(images.len(), 1, "{tag}: expected 1 image");
    assert_coherent(&images[0], tag);

    // Write the render next to the tier so it can be eyeballed.
    if let Some(buf) =
        image::RgbImage::from_raw(images[0].width, images[0].height, images[0].pixels.clone())
    {
        let out_path = std::env::temp_dir().join(format!("z_image_packed_{tag}.png"));
        let _ = buf.save(&out_path);
        eprintln!("[{tag}] wrote {}", out_path.display());
    }
}

/// The q4 packed tier renders a coherent image straight from the packed parts (the primary sc-9408
/// deliverable — the tier is cached, so this is the routine GPU check).
#[test]
#[ignore = "needs Z_IMAGE_PACKED_Q4 (packed q4 tier subdir) + a CUDA GPU; run with --features cuda --ignored"]
fn packed_q4_renders_coherent() {
    render_tier("Z_IMAGE_PACKED_Q4", "q4");
}

/// The q8 packed tier renders a coherent image (double-quant Q8_0 path); only runs when the q8 tier is
/// present locally.
#[test]
#[ignore = "needs Z_IMAGE_PACKED_Q8 (packed q8 tier subdir) + a CUDA GPU; run with --features cuda --ignored"]
fn packed_q8_renders_coherent() {
    render_tier("Z_IMAGE_PACKED_Q8", "q8");
}
