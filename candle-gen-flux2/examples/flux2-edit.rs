//! FLUX.2 **edit** smoke / validation driver (sc-5487 klein, sc-7460 dev) — exercises the bespoke
//! [`candle_gen_flux2::Flux2Edit`] reference-edit provider against a local snapshot on a real GPU, and
//! writes the outputs for an eyeball check.
//!
//! ```text
//! # klein (distilled, dense):
//! cargo run --release --example flux2-edit --features cuda -- \
//!   --snapshot "C:\…\FLUX.2-klein-9B" \
//!   [--reference portrait.png] --prompt "give them a red wizard hat" --steps 4 --seed 42 --out edit.png
//!
//! # dev (32B, embedded guidance, Q4 — single + multi reference, sc-7460):
//! cargo run --release --example flux2-edit --features cuda -- --variant dev --quant q4 \
//!   --snapshot "C:\…\FLUX.2-dev" --reference a.png [--reference2 b.png] \
//!   --prompt "place the subject on a beach at sunset" --steps 28 --guidance 4 --seed 42 --out edit.png
//! ```
//!
//! **klein** (with no `--reference`) first txt2img-generates a base portrait (so it is self-contained)
//! and edits it, then runs a same-seed/same-prompt txt2img baseline (no reference) and prints the
//! mean-abs pixel diff — a decisive ablation that the reference conditions the output. **dev** requires
//! `--reference` (a second 32B load for a txt2img baseline is avoided); it renders single-reference +
//! multi-reference (`[ref, ref2|ref]` — the multi-ref token concat) and ablates against a gray dummy
//! reference on the *same* loaded model. Both check the pre/mid cancel contract.

use std::path::PathBuf;

use candle_gen::gen_core::runtime::CancelFlag;
use candle_gen::gen_core::{
    self, GenerationOutput, GenerationRequest, Image, LoadSpec, Progress, Quant, WeightsSource,
};
use candle_gen_flux2::{Flux2Edit, Flux2EditPaths, Flux2EditRequest};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn arg(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1).cloned())
}

fn save(img: &Image, path: &PathBuf) -> Result<()> {
    let buf = image::RgbImage::from_raw(img.width, img.height, img.pixels.clone())
        .ok_or("invalid RGB buffer dimensions")?;
    buf.save(path)?;
    Ok(())
}

/// Mean absolute per-channel pixel difference between two same-size RGB8 images.
fn mean_abs_diff(a: &Image, b: &Image) -> f64 {
    assert_eq!(a.pixels.len(), b.pixels.len(), "size mismatch");
    let sum: u64 = a
        .pixels
        .iter()
        .zip(&b.pixels)
        .map(|(&x, &y)| (x as i32 - y as i32).unsigned_abs() as u64)
        .sum();
    sum as f64 / a.pixels.len() as f64
}

fn mean_std(img: &Image) -> (f64, f64) {
    let n = img.pixels.len() as f64;
    let mean = img.pixels.iter().map(|&p| p as f64).sum::<f64>() / n;
    let var = img
        .pixels
        .iter()
        .map(|&p| (p as f64 - mean).powi(2))
        .sum::<f64>()
        / n;
    (mean, var.sqrt())
}

fn load_image(path: &str) -> Result<Image> {
    let img = image::open(path)?.to_rgb8();
    Ok(Image {
        width: img.width(),
        height: img.height(),
        pixels: img.into_raw(),
    })
}

/// A flat mid-gray dummy reference (the ablation control — a content-free reference).
fn gray_dummy(w: u32, h: u32) -> Image {
    Image {
        width: w,
        height: h,
        pixels: vec![128u8; (w * h * 3) as usize],
    }
}

/// txt2img via the inventory-registered generator `engine_id` (klein's baseline + self-generated
/// reference). `quant` is honored for dev.
#[allow(clippy::too_many_arguments)]
fn txt2img(
    engine_id: &str,
    snapshot: &str,
    quant: Option<Quant>,
    prompt: &str,
    w: u32,
    h: u32,
    steps: u32,
    seed: u64,
) -> Result<Image> {
    let mut spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from(snapshot)));
    if let Some(q) = quant {
        spec = spec.with_quant(q);
    }
    let gen = gen_core::registry::load(engine_id, &spec)?;
    let req = GenerationRequest {
        prompt: prompt.to_owned(),
        width: w,
        height: h,
        count: 1,
        seed: Some(seed),
        steps: Some(steps),
        ..Default::default()
    };
    let mut noop = |_: Progress| {};
    match gen.generate(&req, &mut noop)? {
        GenerationOutput::Images(mut imgs) => imgs.pop().ok_or_else(|| "no image".into()),
        GenerationOutput::Video { .. } => Err("expected images".into()),
    }
}

fn step_progress(tag: &'static str) -> impl FnMut(Progress) {
    move |p: Progress| {
        if let Progress::Step { current, total } = p {
            print!("\r[{tag}] step {current}/{total}   ");
            use std::io::Write;
            let _ = std::io::stdout().flush();
        }
    }
}

struct Common {
    snapshot: String,
    prompt: String,
    steps: u32,
    guidance: f32,
    seed: u64,
    width: u32,
    height: u32,
    out: PathBuf,
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let dev_variant = matches!(arg(&args, "--variant").as_deref(), Some("dev"));
    let snapshot = arg(&args, "--snapshot")
        .or_else(|| std::env::var("FLUX2_SNAPSHOT").ok())
        .ok_or("pass --snapshot <dir> (or set FLUX2_SNAPSHOT)")?;
    let default_steps = if dev_variant { 28 } else { 4 };
    let default_guidance = if dev_variant { 4.0 } else { 1.0 };
    let c = Common {
        snapshot,
        prompt: arg(&args, "--prompt")
            .unwrap_or_else(|| "make the person wear a bright red wizard hat".into()),
        steps: arg(&args, "--steps")
            .and_then(|s| s.parse().ok())
            .unwrap_or(default_steps),
        guidance: arg(&args, "--guidance")
            .and_then(|s| s.parse().ok())
            .unwrap_or(default_guidance),
        seed: arg(&args, "--seed")
            .and_then(|s| s.parse().ok())
            .unwrap_or(42),
        width: arg(&args, "--width")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1024),
        height: arg(&args, "--height")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1024),
        out: arg(&args, "--out")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("flux2_edit.png")),
    };

    candle_gen_flux2::force_link();

    if dev_variant {
        let quant = match arg(&args, "--quant").as_deref() {
            Some("q8") => Some(Quant::Q8),
            Some("q4") | None => Some(Quant::Q4),
            Some(other) => return Err(format!("--quant must be q4|q8 (got {other})").into()),
        };
        run_dev(&args, &c, quant)
    } else {
        run_klein(&args, &c)
    }
}

/// dev (sc-7460): single + multi reference edit on the 32B flagship (embedded guidance, Q4). Requires
/// `--reference` (avoids a second 32B txt2img load); the ablation is against a gray dummy on the same
/// loaded model.
fn run_dev(args: &[String], c: &Common, quant: Option<Quant>) -> Result<()> {
    let ref_path = arg(args, "--reference")
        .ok_or("dev edit requires --reference <png> (a second 32B txt2img load is avoided)")?;
    let reference = load_image(&ref_path)?;
    println!(
        "[edit-dev] reference={ref_path} ({}x{}) quant={quant:?}",
        reference.width, reference.height
    );
    let reference2 = match arg(args, "--reference2") {
        Some(p) => {
            let r = load_image(&p)?;
            println!("[edit-dev] reference2={p} ({}x{})", r.width, r.height);
            r
        }
        None => reference.clone(),
    };
    println!(
        "[edit-dev] {}x{} steps={} guidance={} seed={}\n[edit-dev] prompt={:?}",
        c.width, c.height, c.steps, c.guidance, c.seed, c.prompt
    );

    let model = Flux2Edit::load_dev(
        &Flux2EditPaths {
            root: PathBuf::from(&c.snapshot),
        },
        quant,
    )?;
    let req = Flux2EditRequest {
        prompt: c.prompt.clone(),
        negative: String::new(),
        width: c.width,
        height: c.height,
        steps: c.steps as usize,
        guidance: c.guidance,
        seed: c.seed,
        // Native VAE: this example exercises the edit pipeline, not the optional PiD SR (sc-8044).
        use_pid: false,
        cancel: CancelFlag::new(),
    };

    // 1) Single-reference edit.
    let mut prog = step_progress("edit-dev:1ref");
    let t0 = std::time::Instant::now();
    let single = model.generate(&req, std::slice::from_ref(&reference), &mut prog)?;
    let (m, s) = mean_std(&single);
    println!(
        "\n[edit-dev] single-ref done in {:.1}s (mean {m:.1} / std {s:.1})",
        t0.elapsed().as_secs_f32()
    );
    save(&single, &c.out)?;
    println!("[edit-dev] wrote {}", c.out.display());

    // 2) Multi-reference edit ([ref, ref2]) — the multi-ref token concat (grids at t=10, t=20).
    let mut prog = step_progress("edit-dev:2ref");
    let t1 = std::time::Instant::now();
    let multi = model.generate(&req, &[reference.clone(), reference2], &mut prog)?;
    let (mm, ms) = mean_std(&multi);
    let multi_out = PathBuf::from(format!("{}_multi.png", c.out.display()));
    println!(
        "\n[edit-dev] multi-ref done in {:.1}s (mean {mm:.1} / std {ms:.1})",
        t1.elapsed().as_secs_f32()
    );
    save(&multi, &multi_out)?;
    println!("[edit-dev] wrote {}", multi_out.display());

    // 3) Ablation: single-ref vs a gray dummy reference (same loaded model) → the diff must be decisive
    // (the reference conditions the output).
    let mut noop = |_: Progress| {};
    let dummy = model.generate(
        &req,
        std::slice::from_ref(&gray_dummy(c.width, c.height)),
        &mut noop,
    )?;
    save(
        &dummy,
        &PathBuf::from(format!("{}_graydummy.png", c.out.display())),
    )?;
    let diff = mean_abs_diff(&single, &dummy);
    println!("[edit-dev] ablation: mean|ref − graydummy| = {diff:.2} (decisive when >> 0)");

    cancel_contract("edit-dev", &model, &req, std::slice::from_ref(&reference));
    Ok(())
}

/// klein (sc-5487): distilled reference edit (dense). Self-contained — txt2img-generates the reference
/// and the no-reference baseline when `--reference` is absent.
fn run_klein(args: &[String], c: &Common) -> Result<()> {
    let reference = match arg(args, "--reference") {
        Some(path) => {
            println!("[edit] reference={path}");
            load_image(&path)?
        }
        None => {
            let base_prompt =
                "a photorealistic studio portrait of a young woman with long red hair, \
                               neutral background, soft lighting";
            println!("[edit] no --reference; txt2img-generating a base portrait first");
            let base = txt2img(
                "flux2_klein_9b",
                &c.snapshot,
                None,
                base_prompt,
                c.width,
                c.height,
                c.steps,
                1,
            )?;
            let ref_path = PathBuf::from(format!("{}_reference.png", c.out.display()));
            save(&base, &ref_path)?;
            println!("[edit] wrote {}", ref_path.display());
            base
        }
    };
    println!(
        "[edit] {}x{} steps={} guidance={} seed={}\n[edit] prompt={:?}",
        c.width, c.height, c.steps, c.guidance, c.seed, c.prompt
    );

    // Ablation baseline FIRST, while no edit model is resident (two 9B models do not co-reside).
    let noref = txt2img(
        "flux2_klein_9b",
        &c.snapshot,
        None,
        &c.prompt,
        c.width,
        c.height,
        c.steps,
        c.seed,
    )?;
    save(
        &noref,
        &PathBuf::from(format!("{}_noref.png", c.out.display())),
    )?;

    let model = Flux2Edit::load(&Flux2EditPaths {
        root: PathBuf::from(&c.snapshot),
    })?;
    let req = Flux2EditRequest {
        prompt: c.prompt.clone(),
        negative: String::new(),
        width: c.width,
        height: c.height,
        steps: c.steps as usize,
        guidance: c.guidance,
        seed: c.seed,
        // Native VAE: this example exercises the edit pipeline, not the optional PiD SR (sc-8044).
        use_pid: false,
        cancel: CancelFlag::new(),
    };
    let mut prog = step_progress("edit");
    let t0 = std::time::Instant::now();
    let edited = model.generate(&req, std::slice::from_ref(&reference), &mut prog)?;
    println!("\n[edit] edit done in {:.1}s", t0.elapsed().as_secs_f32());
    save(&edited, &c.out)?;
    println!("[edit] wrote {}", c.out.display());

    let diff = mean_abs_diff(&edited, &noref);
    println!("[edit] ablation: mean|edit − noref| = {diff:.2} (decisive when >> 0)");

    cancel_contract("edit", &model, &req, std::slice::from_ref(&reference));
    Ok(())
}

/// Pre-cancel and mid-denoise cancel must both return `Canceled`.
fn cancel_contract(tag: &str, model: &Flux2Edit, req: &Flux2EditRequest, refs: &[Image]) {
    let pre = Flux2EditRequest {
        cancel: CancelFlag::new(),
        ..req.clone()
    };
    pre.cancel.cancel();
    let mut noop = |_: Progress| {};
    let pre_ok = matches!(
        model.generate(&pre, refs, &mut noop),
        Err(candle_gen::CandleError::Canceled)
    );
    println!(
        "[{tag}] pre-cancel → Canceled: {}",
        if pre_ok { "PASS" } else { "FAIL" }
    );

    let mid = Flux2EditRequest {
        cancel: CancelFlag::new(),
        ..req.clone()
    };
    let mid_flag = mid.cancel.clone();
    let mut cancel_after_1 = |p: Progress| {
        if let Progress::Step { current, .. } = p {
            if current >= 1 {
                mid_flag.cancel();
            }
        }
    };
    let mid_ok = matches!(
        model.generate(&mid, refs, &mut cancel_after_1),
        Err(candle_gen::CandleError::Canceled)
    );
    println!(
        "[{tag}] mid-cancel → Canceled: {}",
        if mid_ok { "PASS" } else { "FAIL" }
    );
}
