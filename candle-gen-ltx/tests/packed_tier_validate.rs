//! LTX-2.3 **packed-tier** real-weight GPU video-render validation (sc-9545, sc-9089 umbrella).
//!
//! Loads the pre-quantized MLX-packed `SceneWorks/ltx-2.3-mlx` q4 (and, when present, q8) tier
//! **directly from the split packed parts** (no dense bf16 staging) through the registered
//! `ltx_2_3_distilled` generator, and asserts it renders a **coherent, non-degenerate** short video —
//! the end-to-end proof that the sc-9417 packed-detect seam fired on the REAL remapped tier keys (a
//! silent dense fall-back would fail to load the u32-packed transformer weights, and a broken packed
//! forward would decode to solid-black / NaN frames).
//!
//! This is the story sc-9545 render AC that sc-9417 could not satisfy: the tier ships split
//! per-component safetensors (`transformer` / `connector` / `vae_decoder` / gemma shards) with the DiT
//! keys remapped (`to_out.0`↔`to_out`, `ff.net.0.proj`↔`ff.proj_in`, `ff.net.2`↔`ff.proj_out`,
//! `linear_1/2`↔`linear1/2`), ingested by `candle_gen_ltx::tier`.
//!
//! `#[ignore]`d (needs a real GPU + the cached packed tier). On the Windows/Blackwell box (v143 vcvars
//! with CUDA on PATH), point at the **q4 tier subdir** (the packed snapshot nests `gemma/`, `q4/`,
//! `q8/`; the gemma sibling is auto-resolved, or override with `LTX_GEMMA_DIR`):
//!
//! ```text
//! set LTX_PACKED_Q4=D:\.cache\huggingface\hub\models--SceneWorks--ltx-2.3-mlx\snapshots\<hash>\q4
//! set LTX_PACKED_Q8=...\q8    (optional)
//! cargo test -p candle-gen-ltx --features cuda --release --test packed_tier_validate -- --ignored --nocapture
//! ```
#![cfg(feature = "cuda")]

use std::path::PathBuf;
use std::time::Instant;

use candle_gen::gen_core::{
    self, GenerationOutput, GenerationRequest, Image, LoadSpec, Progress, WeightsSource,
};

/// Basic per-frame non-degeneracy: the frame is not solid-black / constant (a broken packed forward —
/// NaN or zeroed activations — decodes to a flat frame) and has some spread of pixel values.
fn assert_frame_coherent(img: &Image, tag: &str) {
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
        "{tag}: frame is (near-)constant [{min}, {max}] — packed forward likely degenerate (black)"
    );
    assert!(
        var.sqrt() > 8.0,
        "{tag}: pixel std {:.1} too low — degenerate frame",
        var.sqrt()
    );
}

fn render_tier(env: &str, tag: &str) {
    let Ok(dir) = std::env::var(env) else {
        eprintln!("SKIP {tag}: set {env} to the packed tier subdir (e.g. …/q4)");
        return;
    };
    candle_gen_ltx::force_link();
    let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from(&dir)));
    let gen =
        gen_core::registry::load("ltx_2_3_distilled", &spec).expect("ltx_2_3_distilled registered");

    // Short + low-res + the baked distilled step schedule (8) to bound time + VRAM for the 22B model:
    // 9 frames (2 latent frames), 256×256, seed 42.
    let req = GenerationRequest {
        prompt: "a fluffy cat walking across a sunny garden, gentle camera pan, cinematic".into(),
        width: 256,
        height: 256,
        count: 1,
        seed: Some(42),
        frames: Some(9),
        sampler: Some("rectified-flow".into()),
        ..Default::default()
    };

    let t = Instant::now();
    let mut on_progress = |_p: Progress| {};
    let out = gen
        .generate(&req, &mut on_progress)
        .unwrap_or_else(|e| panic!("{tag}: packed-tier generate failed: {e}"));
    let secs = t.elapsed().as_secs_f32();
    eprintln!("[{tag}] load+render wall-clock {secs:.1}s (cold: includes packed tier load)");

    let (frames, fps) = match out {
        GenerationOutput::Video { frames, fps, .. } => (frames, fps),
        GenerationOutput::Images(_) => panic!("{tag}: expected video, got images"),
    };
    assert!(!frames.is_empty(), "{tag}: no frames rendered");
    eprintln!("[{tag}] {} frame(s) @ {fps}fps", frames.len());
    for (i, f) in frames.iter().enumerate() {
        assert_frame_coherent(f, &format!("{tag}#{i}"));
    }

    // Write the first + middle frames next to temp so they can be eyeballed.
    for &i in &[0usize, frames.len() / 2] {
        if let Some(buf) =
            image::RgbImage::from_raw(frames[i].width, frames[i].height, frames[i].pixels.clone())
        {
            let out_path = std::env::temp_dir().join(format!("ltx_packed_{tag}_frame{i:03}.png"));
            let _ = buf.save(&out_path);
            eprintln!("[{tag}] wrote {}", out_path.display());
        }
    }
}

/// The q4 packed tier renders a coherent short video straight from the split packed parts (the primary
/// sc-9545 deliverable — the sc-9417 render AC).
#[test]
#[ignore = "needs LTX_PACKED_Q4 (packed q4 tier subdir) + a CUDA GPU; run with --features cuda --ignored"]
fn packed_q4_renders_coherent_video() {
    render_tier("LTX_PACKED_Q4", "q4");
}

/// The q8 packed tier renders a coherent short video (double-quant Q8_0 path); only runs when the q8
/// tier is present locally.
#[test]
#[ignore = "needs LTX_PACKED_Q8 (packed q8 tier subdir) + a CUDA GPU; run with --features cuda --ignored"]
fn packed_q8_renders_coherent_video() {
    render_tier("LTX_PACKED_Q8", "q8");
}
