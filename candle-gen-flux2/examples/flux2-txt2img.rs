//! FLUX.2-klein txt2img smoke driver — resolves THIS crate's inventory-registered generator through
//! `gen_core::registry::load("flux2_klein_9b", …)`, runs a real `generate` against a local FLUX.2
//! snapshot, and writes the `gen_core::Image` to PNG. The human-eyeball check behind sc-3695.
//!
//! ```text
//! cargo run --release --example flux2-txt2img --features cuda -- \
//!   --snapshot "C:\Users\…\models--black-forest-labs--FLUX.2-klein-9B\snapshots\<hash>" \
//!   --prompt "a photo of a rusty robot holding a lit candle" --steps 4 --seed 42 --out out.png
//! ```

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
        .or_else(|| std::env::var("FLUX2_SNAPSHOT").ok())
        .ok_or(
            "pass --snapshot <dir> (or set FLUX2_SNAPSHOT) pointing at a FLUX.2-klein snapshot",
        )?;
    let prompt = arg(&args, "--prompt").unwrap_or_else(|| {
        "a photo of a rusty robot holding a lit candle, dramatic cinematic lighting, highly detailed"
            .into()
    });
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
    let out = arg(&args, "--out")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("flux2_smoke.png"));

    println!(
        "[smoke] snapshot={snapshot}\n[smoke] {width}x{height} steps={steps:?} guidance={guidance:?} seed={seed}\n[smoke] prompt={prompt:?}"
    );

    candle_gen_flux2::force_link();
    let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from(&snapshot)));
    let gen = gen_core::registry::load("flux2_klein_9b", &spec)?;
    println!(
        "[smoke] resolved engine id={} backend={}",
        gen.descriptor().id,
        gen.descriptor().backend
    );

    let req = GenerationRequest {
        prompt,
        width,
        height,
        count: 1,
        seed: Some(seed),
        steps,
        guidance,
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
    let t0 = std::time::Instant::now();
    let output = gen.generate(&req, &mut on_progress)?;
    let secs = t0.elapsed().as_secs_f32();
    let images = match output {
        GenerationOutput::Images(imgs) => imgs,
        GenerationOutput::Video { .. } => return Err("expected images, got video".into()),
    };
    println!("[smoke] {} image(s) in {secs:.1}s", images.len());

    let img = &images[0];
    let buf = image::RgbImage::from_raw(img.width, img.height, img.pixels.clone())
        .ok_or("invalid RGB buffer dimensions")?;
    buf.save(&out)?;
    println!(
        "[smoke] wrote {} ({}x{})",
        out.display(),
        img.width,
        img.height
    );
    Ok(())
}
