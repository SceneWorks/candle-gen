//! GPU smoke for the `bernini_renderer` raw t2v render (sc-10994). `#[ignore]`d — needs a dual-expert
//! Wan2.2-T2V-A14B candle snapshot on disk + a CUDA device. The Bernini renderer IS Wan2.2-T2V-A14B
//! (finetuned), so a stock `SceneWorks/wan2.2-t2v-a14b-candle` tier exercises the full renderer
//! machinery (UMT5 → dual-expert boundary switch → APG guidance → z16 VAE decode) end to end.
//!
//! ```sh
//! # point at a diffusers-layout dual-expert snapshot dir (text_encoder/ transformer/ transformer_2/ vae/ tokenizer/)
//! export SCENEWORKS_BERNINI_SNAPSHOT="E:/huggingface/hub/models--SceneWorks--wan2.2-t2v-a14b-candle/snapshots/<hash>/q4"
//! export SCENEWORKS_BERNINI_OUT="D:/tmp/bernini_render.png"   # optional; defaults to the temp dir
//! # vcvars 14.44 + CUDA_COMPUTE_CAP=120, then:
//! cargo test -p candle-gen-bernini --release --features cuda gpu_smoke_raw_render -- --ignored --nocapture
//! ```

use std::time::Instant;

use candle_gen::gen_core::registry;
use candle_gen::gen_core::{
    GenerationOutput, GenerationRequest, LoadSpec, Progress, WeightsSource,
};

#[test]
#[ignore = "GPU smoke: needs a dual-expert Wan2.2-T2V-A14B candle snapshot + CUDA device"]
fn gpu_smoke_raw_render() {
    candle_gen_bernini::force_link();

    let snapshot = std::env::var("SCENEWORKS_BERNINI_SNAPSHOT").expect(
        "set SCENEWORKS_BERNINI_SNAPSHOT to a dual-expert Wan2.2-T2V-A14B candle snapshot dir",
    );
    let out_path = std::env::var("SCENEWORKS_BERNINI_OUT").unwrap_or_else(|_| {
        std::env::temp_dir()
            .join("bernini_render.png")
            .to_string_lossy()
            .into_owned()
    });

    let spec = LoadSpec::new(WeightsSource::Dir(snapshot.clone().into()));
    let gen = registry::load("bernini_renderer", &spec).expect("load bernini_renderer");
    eprintln!("[bernini-gpu] loaded bernini_renderer from {snapshot}");

    // A single-frame (t2i) raw render: 256x256, a handful of steps so the dual-expert boundary (0.875)
    // is crossed (high→low). frames=1 ⇒ 1 latent frame ⇒ a still image.
    let req = GenerationRequest {
        prompt: "a red panda sitting on a mossy log in a sunlit forest, cinematic".into(),
        width: 256,
        height: 256,
        frames: Some(1),
        steps: Some(12),
        seed: Some(42),
        sampler: Some("uni_pc".into()),
        ..Default::default()
    };

    let mut last_step = 0u32;
    let mut on_progress = |p: Progress| {
        if let Progress::Step { current, total } = p {
            if current == 1 || current == total || current % 4 == 0 {
                eprintln!("[bernini-gpu] step {current}/{total}");
            }
            last_step = current;
        } else {
            eprintln!("[bernini-gpu] {p:?}");
        }
    };

    let t0 = Instant::now();
    let out = gen
        .generate(&req, &mut on_progress)
        .expect("bernini_renderer generate");
    let dt = t0.elapsed();

    let image = match out {
        GenerationOutput::Images(mut imgs) => imgs.remove(0),
        GenerationOutput::Video { mut frames, .. } => frames.remove(0),
    };
    assert!(
        image.width >= 16 && image.height >= 16,
        "decoded frame has real dims"
    );
    assert_eq!(
        image.pixels.len(),
        image.width as usize * image.height as usize * 3,
        "decoded frame is a full RGB8 buffer"
    );
    // Not all-constant (a black/uniform frame would be a broken decode).
    let first = image.pixels[0];
    let varied = image.pixels.iter().any(|&p| p != first);
    assert!(varied, "decoded frame must not be a single flat color");

    image::save_buffer(
        &out_path,
        &image.pixels,
        image.width,
        image.height,
        image::ColorType::Rgb8,
    )
    .expect("save png");

    eprintln!(
        "[bernini-gpu] DONE raw t2v render {}x{} in {:.1}s ({} steps) -> {out_path}",
        image.width,
        image.height,
        dt.as_secs_f32(),
        last_step,
    );
}
