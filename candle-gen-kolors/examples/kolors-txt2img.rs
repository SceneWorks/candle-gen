//! Kolors txt2img smoke driver — exercises the full candle-gen seam end-to-end on a real GPU:
//! `gen_core::registry::load("kolors", …)` resolves THIS crate's inventory-registered generator, runs
//! [`Generator::generate`] against a local Kolors snapshot, and writes each `gen_core::Image` to PNG.
//!
//! The human-eyeball check behind sc-5485 (the worker, not this example, owns asset writes in
//! production). Build with the CUDA backend on the Windows/Blackwell box:
//!
//! ```text
//! cargo run --release --example kolors-txt2img --features cuda -- \
//!   --snapshot "C:\Users\…\models--Kwai-Kolors--Kolors-diffusers\snapshots\<hash>" \
//!   --prompt "一只猫 / a photo of a cat holding a lit candle" --seed 42 --out out.png
//! ```
//!
//! The snapshot is a Kolors diffusers tree (`tokenizer/` with a materialized `tokenizer.json`,
//! `text_encoder/` ChatGLM3-6B, `unet/`, `vae/`). Defaults: 50 steps / CFG 5.0 / 1024².

use std::path::PathBuf;

use candle_gen::gen_core::{
    self, GenerationOutput, GenerationRequest, LoadSpec, Progress, WeightsSource,
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn arg(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1).cloned())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    let snapshot = arg(&args, "--snapshot")
        .or_else(|| std::env::var("KOLORS_SNAPSHOT").ok())
        .ok_or("pass --snapshot <dir> (or set KOLORS_SNAPSHOT) pointing at a Kolors snapshot")?;
    let prompt = arg(&args, "--prompt").unwrap_or_else(|| {
        "a photo of a rusty robot holding a lit candle, dramatic cinematic lighting, highly detailed"
            .into()
    });
    let negative = arg(&args, "--negative");
    // Curated sampler/scheduler axes (epic 7114 / sc-8984): e.g. `--sampler dpmpp_2m`, or
    // `--scheduler karras` ALONE (the previously-dropped scheduler-only curated request).
    let sampler = arg(&args, "--sampler");
    let scheduler = arg(&args, "--scheduler");
    let steps: Option<u32> = arg(&args, "--steps").and_then(|s| s.parse().ok());
    let guidance: Option<f32> = arg(&args, "--guidance").and_then(|s| s.parse().ok());
    let seed: u64 = arg(&args, "--seed")
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);
    let width: u32 = arg(&args, "--width")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024);
    let height: u32 = arg(&args, "--height")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024);
    let count: u32 = arg(&args, "--count")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let out = arg(&args, "--out")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("kolors_smoke.png"));

    println!(
        "[smoke] snapshot={snapshot}\n[smoke] engine=kolors {width}x{height} steps={steps:?} guidance={guidance:?} seed={seed} count={count} sampler={sampler:?} scheduler={scheduler:?}\n[smoke] prompt={prompt:?}"
    );

    // Force-link the provider so its `inventory::submit!` registration survives the linker.
    candle_gen_kolors::force_link();

    let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from(&snapshot)));
    let gen = gen_core::registry::load("kolors", &spec)?;
    println!(
        "[smoke] resolved engine id={} backend={}",
        gen.descriptor().id,
        gen.descriptor().backend
    );

    let req = GenerationRequest {
        prompt,
        negative_prompt: negative,
        width,
        height,
        count,
        seed: Some(seed),
        steps,
        guidance,
        sampler,
        scheduler,
        ..Default::default()
    };

    let mut on_progress = |p: Progress| match p {
        Progress::Step { current, total } => {
            print!("\r[smoke] step {current}/{total}   ");
            use std::io::Write;
            let _ = std::io::stdout().flush();
        }
        Progress::Decoding => println!("\n[smoke] decoding"),
    };
    let t_call = std::time::Instant::now();
    let output = gen.generate(&req, &mut on_progress)?;
    let gen_s = t_call.elapsed().as_secs_f32();
    let images = match output {
        GenerationOutput::Images(imgs) => imgs,
        GenerationOutput::Video { .. } => return Err("expected images, got video".into()),
    };
    println!("[smoke] {} image(s) in {gen_s:.1}s", images.len());

    for (i, img) in images.iter().enumerate() {
        let path = if images.len() == 1 {
            out.clone()
        } else {
            out.with_file_name(format!(
                "{}_{i}.png",
                out.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("kolors_smoke")
            ))
        };
        let buf = image::RgbImage::from_raw(img.width, img.height, img.pixels.clone())
            .ok_or("invalid RGB buffer dimensions")?;
        buf.save(&path)?;
        println!(
            "[smoke] wrote {} ({}x{})",
            path.display(),
            img.width,
            img.height
        );
    }
    Ok(())
}
