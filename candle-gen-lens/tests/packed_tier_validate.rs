//! Lens **packed-tier** real-weight GPU validation (sc-9457, sc-9089 umbrella).
//!
//! Loads a pre-quantized MLX-packed `SceneWorks/lens-mlx` / `lens-turbo-mlx` q4 (and, when present, q8)
//! tier **directly from the packed parts** — the DiT linears via `linear_detect` (sc-9413) AND the
//! gpt-oss text-encoder fused MoE experts via the 3-D per-expert affine loader (sc-9457) — through the
//! registered `lens_turbo` generator, and asserts it renders a **coherent, non-degenerate** image. This
//! is the end-to-end proof that the encoder now loads packed too, so a pure `lens-mlx` snapshot loads
//! packed end-to-end (DiT + encoder + VAE) with no MXFP4 dependency: a silent fall-back to dense would
//! fail to read the u32-packed expert codes, and a broken packed forward would render solid black / NaN
//! (the sc-7702 int8-activation-outlier failure mode the dequant path avoids).
//!
//! The `LoadSpec` sets **no** quant level, so this also proves the encoder's packed-detect is
//! quant-independent (it fires on the `.scales` sibling, not on the requested `Quant`).
//!
//! `#[ignore]`d (needs a real GPU + the cached packed tier). On the Windows/Blackwell box (v143 vcvars
//! and CUDA on PATH), point at the **tier subdir** — the packed snapshot nests `bf16/`, `q4/`, `q8/`,
//! each a full diffusers snapshot with `tokenizer/ text_encoder/ transformer/ vae/`:
//!
//! ```text
//! set LENS_PACKED_Q4=D:\.cache\huggingface\hub\models--SceneWorks--lens-turbo-mlx\snapshots\<hash>\q4
//! set LENS_PACKED_Q8=...\q8    (optional)
//! cargo test -p candle-gen-lens --features cuda --release --test packed_tier_validate -- --ignored --nocapture
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
        eprintln!("SKIP {tag}: set {env} to the packed lens tier subdir (…/q4 or …/q8)");
        return;
    };
    candle_gen_lens::force_link();
    // No `.with_quant(...)`: the packed tier is detected from the on-disk `.scales` siblings, so this
    // proves the DiT + encoder packed-load is quant-independent (a pure lens-mlx snapshot loads packed
    // end-to-end with no quant request).
    let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from(&dir)));
    let gen = gen_core::registry::load("lens_turbo", &spec).expect("lens_turbo registered");
    eprintln!(
        "[{tag}] engine id={} backend={}",
        gen.descriptor().id,
        gen.descriptor().backend
    );

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
        let out_path = std::env::temp_dir().join(format!("lens_packed_{tag}.png"));
        let _ = buf.save(&out_path);
        eprintln!("[{tag}] wrote {}", out_path.display());
    }
}

/// The q4 packed tier renders a coherent image straight from the packed parts (the primary sc-9457
/// deliverable — DiT + encoder both packed; the tier is cached, so this is the routine GPU check).
#[test]
#[ignore = "needs LENS_PACKED_Q4 (packed q4 tier subdir) + a CUDA GPU; run with --features cuda --ignored"]
fn packed_q4_renders_coherent() {
    render_tier("LENS_PACKED_Q4", "q4");
}

/// The q8 packed tier renders a coherent image (double-quant Q8_0 experts + DiT); only runs when the q8
/// tier is present locally.
#[test]
#[ignore = "needs LENS_PACKED_Q8 (packed q8 tier subdir) + a CUDA GPU; run with --features cuda --ignored"]
fn packed_q8_renders_coherent() {
    render_tier("LENS_PACKED_Q8", "q8");
}
