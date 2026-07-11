//! Bernini renderer smoke driver (sc-11004) — resolves the `bernini_renderer` engine through
//! `gen_core::registry::load`, feeds a `Reference` conditioning image so `resolve_mode` picks a
//! **packed source-id conditioning** mode (i2i → `v2v`), and runs a real conditioned `generate` against
//! a local Wan2.2-T2V-A14B (or Bernini) snapshot: the reference is VAE-encoded to a z16 source latent,
//! patch-embedded with its source-id RoPE, packed on the DiT token axis with the noisy target, run
//! through `forward_packed`, and the target tokens sliced back out + decoded. Writes the output PNG(s).
//!
//! ```text
//! cargo run --release --example bernini-render --features cuda -- \
//!   --snapshot "E:\huggingface\hub\models--SceneWorks--wan2.2-t2v-a14b-candle\snapshots\<hash>\q4" \
//!   --prompt "a serene mountain lake at golden hour, cinematic" \
//!   --width 256 --height 256 --frames 1 --steps 4 --seed 42 --out bernini_render_smoke
//! ```
//!
//! Omit `--image` to render from a synthetic gradient reference (still exercises the packed forward).
//! Pass `--mode t2v_apg` to run the text-only path instead.

use std::path::PathBuf;

use candle_gen::gen_core::{
    self, Conditioning, GenerationOutput, GenerationRequest, Image, LoadSpec, Progress,
    WeightsSource,
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn arg(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1).cloned())
}

fn load_image(path: &str) -> Result<Image> {
    let rgb = image::open(path)?.to_rgb8();
    let (width, height) = (rgb.width(), rgb.height());
    Ok(Image {
        width,
        height,
        pixels: rgb.into_raw(),
    })
}

/// A synthetic RGB gradient reference (so the smoke needs no external file).
fn synth_image(w: u32, h: u32) -> Image {
    let mut pixels = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            pixels.push((255 * x / w.max(1)) as u8);
            pixels.push((255 * y / h.max(1)) as u8);
            pixels.push(128);
        }
    }
    Image {
        width: w,
        height: h,
        pixels,
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let snapshot = arg(&args, "--snapshot")
        .or_else(|| std::env::var("BERNINI_SNAPSHOT").ok())
        .ok_or("pass --snapshot <dir> (or set BERNINI_SNAPSHOT)")?;
    let prompt = arg(&args, "--prompt")
        .unwrap_or_else(|| "a serene mountain lake at golden hour, cinematic".into());
    let mode = arg(&args, "--mode"); // e.g. t2v_apg to force the text-only path
    let steps: Option<u32> = arg(&args, "--steps").and_then(|s| s.parse().ok());
    let guidance: Option<f32> = arg(&args, "--guidance").and_then(|s| s.parse().ok());
    let seed: u64 = arg(&args, "--seed")
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);
    let width: u32 = arg(&args, "--width")
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);
    let height: u32 = arg(&args, "--height")
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);
    let frames: Option<u32> = arg(&args, "--frames")
        .and_then(|s| s.parse().ok())
        .or(Some(1));
    let out = arg(&args, "--out").unwrap_or_else(|| "bernini_render_smoke".into());

    // A Reference image → resolve_mode picks the i2i packed conditioning mode (v2v). --mode t2v_apg
    // forces the text-only path (no conditioning).
    let text_only = mode.as_deref() == Some("t2v_apg") || mode.as_deref() == Some("t2v");
    let conditioning = if text_only {
        vec![]
    } else {
        let image = match arg(&args, "--image") {
            Some(p) => load_image(&p)?,
            None => synth_image(width, height),
        };
        vec![Conditioning::Reference {
            image,
            strength: None,
        }]
    };

    println!(
        "[smoke] snapshot={snapshot}\n[smoke] {width}x{height} frames={frames:?} steps={steps:?} \
         guidance={guidance:?} seed={seed} mode={mode:?} conditioned={}\n[smoke] prompt={prompt:?}",
        !conditioning.is_empty()
    );

    candle_gen_bernini::force_link();
    candle_gen_wan::force_link();
    let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from(&snapshot)));
    let gen = gen_core::registry::load("bernini_renderer", &spec)?;
    println!(
        "[smoke] resolved engine id={} backend={} modality={:?} conditioning={:?}",
        gen.descriptor().id,
        gen.descriptor().backend,
        gen.descriptor().modality,
        gen.descriptor().capabilities.conditioning,
    );

    let req = GenerationRequest {
        prompt,
        width,
        height,
        count: 1,
        seed: Some(seed),
        steps,
        guidance,
        frames,
        video_mode: mode,
        conditioning,
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

    std::fs::create_dir_all(&out)?;
    let frames_out = match output {
        GenerationOutput::Images(imgs) => imgs,
        GenerationOutput::Video { frames, .. } => frames,
    };
    for (i, f) in frames_out.iter().enumerate() {
        let buf = image::RgbImage::from_raw(f.width, f.height, f.pixels.clone())
            .ok_or("invalid RGB buffer dimensions")?;
        buf.save(PathBuf::from(&out).join(format!("frame_{i:03}.png")))?;
    }
    println!(
        "[smoke] wrote {} frame(s) to {}/ ({}x{}) in {secs:.1}s",
        frames_out.len(),
        out,
        frames_out[0].width,
        frames_out[0].height
    );
    Ok(())
}
