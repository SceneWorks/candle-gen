//! FLUX.2-dev **strict-pose ControlNet** smoke / validation driver (sc-7460) — exercises the bespoke
//! [`candle_gen_flux2::Flux2Control`] provider (the `FLUX.2-dev-Fun-Controlnet-Union` VACE branch on
//! the dev base) against a local dev snapshot + the control checkpoint on a real GPU, and writes the
//! outputs for an eyeball check. The human-validation behind the candle FLUX.2-dev control lane.
//!
//! ```text
//! cargo run --release --example flux2-control --features cuda -- \
//!   --snapshot "C:\…\FLUX.2-dev" \
//!   --control  "C:\…\FLUX.2-dev-Fun-Controlnet-Union-2602.safetensors" \
//!   [--pose pose.png] --quant q4 \
//!   --prompt "a knight in ornate armor, dramatic lighting" --steps 28 --seed 42 --out control.png
//! ```
//!
//! With no `--pose`, the harness draws a simple synthetic OpenPose-style stick figure (so it is fully
//! self-contained). It runs a decisive **ablation**: the same prompt+seed at `control_scale = 0` (which
//! the engine proves is byte-identical to the base txt2img forward) vs `control_scale = 0.75` — the
//! mean-abs pixel diff must be large, proving the pose skeleton actually conditions the output. It also
//! checks the pre-cancel / mid-denoise cancel contract.

use std::path::PathBuf;

use candle_gen::gen_core::runtime::CancelFlag;
use candle_gen::gen_core::{Image, Progress, Quant};
use candle_gen_flux2::{Flux2Control, Flux2ControlPaths, Flux2ControlRequest};

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

/// Image mean / std over all channels (a coherence sanity check — degenerate outputs have std ≈ 0).
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

/// Draw a thick white line (Bresenham + a square brush) onto an RGB8 buffer.
#[allow(clippy::too_many_arguments)]
fn draw_line(px: &mut [u8], w: usize, h: usize, x0: i32, y0: i32, x1: i32, y1: i32, r: i32) {
    let (dx, dy) = ((x1 - x0).abs(), -(y1 - y0).abs());
    let (sx, sy) = (if x0 < x1 { 1 } else { -1 }, if y0 < y1 { 1 } else { -1 });
    let (mut err, mut x, mut y) = (dx + dy, x0, y0);
    loop {
        for by in -r..=r {
            for bx in -r..=r {
                let (px_, py_) = (x + bx, y + by);
                if px_ >= 0 && px_ < w as i32 && py_ >= 0 && py_ < h as i32 {
                    let i = (py_ as usize * w + px_ as usize) * 3;
                    px[i] = 255;
                    px[i + 1] = 255;
                    px[i + 2] = 255;
                }
            }
        }
        if x == x1 && y == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }
}

/// A crude OpenPose-style stick figure on a black background — a legitimate pose control input for the
/// union ControlNet (head + spine + arms + legs). Centered, sized to the canvas.
fn synthetic_pose(w: u32, h: u32) -> Image {
    let (wu, hu) = (w as usize, h as usize);
    let mut px = vec![0u8; wu * hu * 3];
    let cx = w as i32 / 2;
    let head = h as i32 / 5;
    let neck = h as i32 * 30 / 100;
    let hip = h as i32 * 60 / 100;
    let span = w as i32 / 5;
    let r = (w.min(h) / 110).max(2) as i32;
    // head (a small ring approximated by a short vertical segment) + spine
    draw_line(&mut px, wu, hu, cx, head, cx, neck, r + 2);
    draw_line(&mut px, wu, hu, cx, neck, cx, hip, r);
    // arms: shoulders → elbows → hands
    draw_line(&mut px, wu, hu, cx, neck, cx - span, neck + span, r);
    draw_line(
        &mut px,
        wu,
        hu,
        cx - span,
        neck + span,
        cx - span - span / 2,
        hip,
        r,
    );
    draw_line(&mut px, wu, hu, cx, neck, cx + span, neck + span, r);
    draw_line(
        &mut px,
        wu,
        hu,
        cx + span,
        neck + span,
        cx + span + span / 2,
        hip,
        r,
    );
    // legs: hips → knees → feet
    draw_line(&mut px, wu, hu, cx, hip, cx - span / 2, hip + span, r);
    draw_line(
        &mut px,
        wu,
        hu,
        cx - span / 2,
        hip + span,
        cx - span / 2,
        h as i32 * 92 / 100,
        r,
    );
    draw_line(&mut px, wu, hu, cx, hip, cx + span / 2, hip + span, r);
    draw_line(
        &mut px,
        wu,
        hu,
        cx + span / 2,
        hip + span,
        cx + span / 2,
        h as i32 * 92 / 100,
        r,
    );
    Image {
        width: w,
        height: h,
        pixels: px,
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let snapshot = arg(&args, "--snapshot")
        .or_else(|| std::env::var("FLUX2_DEV_SNAPSHOT").ok())
        .ok_or("pass --snapshot <dir> at a FLUX.2-dev snapshot")?;
    let control = arg(&args, "--control")
        .or_else(|| std::env::var("FLUX2_CONTROL").ok())
        .ok_or("pass --control <path> at the FLUX.2-dev-Fun-Controlnet-Union .safetensors")?;
    let prompt = arg(&args, "--prompt").unwrap_or_else(|| {
        "a knight in ornate steel armor standing in a courtyard, dramatic cinematic lighting, \
         highly detailed"
            .into()
    });
    let steps: u32 = arg(&args, "--steps")
        .and_then(|s| s.parse().ok())
        .unwrap_or(28);
    let guidance: f32 = arg(&args, "--guidance")
        .and_then(|s| s.parse().ok())
        .unwrap_or(4.0);
    let control_scale: f32 = arg(&args, "--control-scale")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.75);
    let seed: u64 = arg(&args, "--seed")
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);
    let width: u32 = arg(&args, "--width")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024);
    let height: u32 = arg(&args, "--height")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024);
    let quant = match arg(&args, "--quant").as_deref() {
        Some("q4") => Some(Quant::Q4),
        Some("q8") => Some(Quant::Q8),
        Some(other) => return Err(format!("--quant must be q4|q8 (got {other})").into()),
        None => Some(Quant::Q4),
    };
    let out = arg(&args, "--out")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("flux2_dev_control.png"));

    let pose = match arg(&args, "--pose") {
        Some(path) => {
            let img = image::open(&path)?.to_rgb8();
            println!("[control] pose={path} ({}x{})", img.width(), img.height());
            Image {
                width: img.width(),
                height: img.height(),
                pixels: img.into_raw(),
            }
        }
        None => {
            println!("[control] no --pose; drawing a synthetic stick-figure skeleton");
            let p = synthetic_pose(width, height);
            save(&p, &PathBuf::from(format!("{}_pose.png", out.display())))?;
            p
        }
    };

    println!(
        "[control] {width}x{height} steps={steps} guidance={guidance} scale={control_scale} \
         quant={quant:?} seed={seed}\n[control] prompt={prompt:?}"
    );

    let model = Flux2Control::load(
        &Flux2ControlPaths {
            root: PathBuf::from(&snapshot),
            control: PathBuf::from(&control),
        },
        quant,
    )?;

    let base_req = Flux2ControlRequest {
        prompt: prompt.clone(),
        width,
        height,
        steps: steps as usize,
        guidance,
        control_scale,
        seed,
        // Native VAE: this example exercises the control pipeline, not the optional PiD SR (sc-8044).
        use_pid: false,
        cancel: CancelFlag::new(),
    };
    let mut on_progress = |p: Progress| {
        if let Progress::Step { current, total } = p {
            print!("\r[control] step {current}/{total}   ");
            use std::io::Write;
            let _ = std::io::stdout().flush();
        }
    };

    // 1) The pose-conditioned render.
    let t0 = std::time::Instant::now();
    let controlled = model.generate(&base_req, &pose, &mut on_progress)?;
    println!(
        "\n[control] render done in {:.1}s",
        t0.elapsed().as_secs_f32()
    );
    save(&controlled, &out)?;
    let (m, s) = mean_std(&controlled);
    println!(
        "[control] wrote {} (mean {m:.1} / std {s:.1})",
        out.display()
    );

    // 2) Ablation: scale = 0 (engine-proven byte-identical to the base txt2img forward) vs the pose
    // render → the diff must be decisive (the pose skeleton conditions the output).
    let zero_req = Flux2ControlRequest {
        control_scale: 0.0,
        cancel: CancelFlag::new(),
        ..base_req.clone()
    };
    let mut noop = |_: Progress| {};
    let base = model.generate(&zero_req, &pose, &mut noop)?;
    save(
        &base,
        &PathBuf::from(format!("{}_scale0.png", out.display())),
    )?;
    let diff = mean_abs_diff(&controlled, &base);
    println!(
        "[control] ablation: mean|scale=0.75 − scale=0| = {diff:.2} (decisive when >> 0; the pose \
         conditions the output)"
    );

    // 3) Cancel contract: pre-cancel and mid-denoise cancel both return Canceled.
    let pre = Flux2ControlRequest {
        cancel: CancelFlag::new(),
        ..base_req.clone()
    };
    pre.cancel.cancel();
    let pre_ok = matches!(
        model.generate(&pre, &pose, &mut noop),
        Err(candle_gen::CandleError::Canceled)
    );
    println!(
        "[control] pre-cancel → Canceled: {}",
        if pre_ok { "PASS" } else { "FAIL" }
    );

    let mid = Flux2ControlRequest {
        cancel: CancelFlag::new(),
        ..base_req.clone()
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
        model.generate(&mid, &pose, &mut cancel_after_1),
        Err(candle_gen::CandleError::Canceled)
    );
    println!(
        "[control] mid-cancel → Canceled: {}",
        if mid_ok { "PASS" } else { "FAIL" }
    );

    Ok(())
}
