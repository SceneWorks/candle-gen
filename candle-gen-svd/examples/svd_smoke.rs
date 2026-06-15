//! GPU smoke / validation harness for the candle SVD-XT provider (sc-5493).
//!
//! Loads the `svd_xt` engine on the default device (CUDA when built `--features cuda`), runs a real
//! img2vid generation from a synthetic source frame, and checks the output is non-degenerate (the
//! requested frame count + size, frame-0 not constant, motion across the clip). Dumps a few frames as
//! PNG for visual inspection. Size / frame-count / steps / decode-chunk are `SVD_*` env overridable.
//!
//! Usage: `cargo run -p candle-gen-svd --features cuda --example svd_smoke -- <snapshot_dir> [out_dir]`

use std::path::PathBuf;
use std::time::Instant;

use candle_gen::gen_core::{
    registry, Conditioning, GenerationOutput, GenerationRequest, Image, LoadSpec, Progress,
    WeightsSource,
};

fn synthetic_image(w: u32, h: u32) -> Image {
    let (wu, hu) = (w as usize, h as usize);
    let mut pixels = vec![0u8; wu * hu * 3];
    for y in 0..hu {
        for x in 0..wu {
            let i = (y * wu + x) * 3;
            pixels[i] = (x * 255 / wu) as u8; // R gradient L→R
            pixels[i + 1] = (y * 255 / hu) as u8; // G gradient T→B
            pixels[i + 2] = 96; // B const
                                // A bright square near the centre (gives the motion model something to move).
            let (cx, cy) = (wu / 2, hu / 2);
            if x.abs_diff(cx) < 70 && y.abs_diff(cy) < 70 {
                pixels[i] = 255;
                pixels[i + 1] = 240;
                pixels[i + 2] = 210;
            }
        }
    }
    Image {
        width: w,
        height: h,
        pixels,
    }
}

fn write_png(path: &PathBuf, img: &Image) -> Result<(), Box<dyn std::error::Error>> {
    let buf = image::RgbImage::from_raw(img.width, img.height, img.pixels.clone())
        .ok_or("frame buffer size mismatch")?;
    buf.save(path)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    candle_gen_svd::force_link();
    let mut args = std::env::args().skip(1);
    let root = PathBuf::from(
        args.next()
            .expect("usage: svd_smoke <snapshot_dir> [out_dir]"),
    );
    let out_dir = PathBuf::from(args.next().unwrap_or_else(|| "svd_smoke_out".to_string()));
    std::fs::create_dir_all(&out_dir)?;

    // Env-overridable so a fast functional smoke (small res, chunked decode) and a native-res run share
    // one harness. Defaults: 512x512 / 14 frames / 25 steps / decode chunk 8 — tractable in well under
    // the 97 GB cap. Bump SVD_W/SVD_H to 1024/576 for the native SVD-XT resolution.
    let envu = |k: &str, d: u32| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(d)
    };
    let (w, h) = (envu("SVD_W", 512), envu("SVD_H", 512));
    let n_frames = envu("SVD_FRAMES", 14);
    let steps = envu("SVD_STEPS", 25);
    let chunk = envu("SVD_CHUNK", 8);
    eprintln!("config: {w}x{h} frames={n_frames} steps={steps} decode_chunk={chunk}");
    let src = synthetic_image(w, h);

    let spec = LoadSpec::new(WeightsSource::Dir(root));
    let gen = registry::load("svd_xt", &spec)?;
    eprintln!(
        "loaded engine id={} backend={}",
        gen.descriptor().id,
        gen.descriptor().backend
    );

    let req = GenerationRequest {
        width: w,
        height: h,
        frames: Some(n_frames),
        steps: Some(steps),
        conditioning_fps: Some(7),
        motion_bucket_id: Some(127.0),
        noise_aug_strength: Some(0.02),
        // Chunked VAE decode (production worker default) — avoids the full-clip decode memory spike.
        decode_chunk_size: Some(chunk),
        seed: Some(42),
        conditioning: vec![Conditioning::Reference {
            image: src,
            strength: None,
        }],
        ..Default::default()
    };

    let t0 = Instant::now();
    let mut last = 0u32;
    let out = gen.generate(&req, &mut |p| {
        if let Progress::Step { current, total } = p {
            if current != last {
                last = current;
                eprintln!(
                    "  step {current}/{total} ({:.1}s)",
                    t0.elapsed().as_secs_f32()
                );
            }
        }
    })?;
    let elapsed = t0.elapsed().as_secs_f32();

    let GenerationOutput::Video { frames, fps, audio } = out else {
        panic!("svd_xt did not return Video");
    };
    eprintln!(
        "DONE in {elapsed:.1}s: {} frames @ {fps}fps, {}x{}, audio={}",
        frames.len(),
        frames[0].width,
        frames[0].height,
        audio.is_some()
    );

    // Non-degeneracy checks: the requested frame count + size, frame-0 not constant, motion across clip.
    assert_eq!(frames.len(), n_frames as usize, "frame count");
    assert_eq!((frames[0].width, frames[0].height), (w, h), "frame size");
    let f0 = &frames[0].pixels;
    let (mn, mx) = (*f0.iter().min().unwrap(), *f0.iter().max().unwrap());
    eprintln!("frame0 pixel range: {mn}..{mx}");
    assert!(mx > mn, "frame0 is constant — pipeline likely dead/NaN→0");
    let fl = &frames[frames.len() - 1].pixels;
    let motion: u64 = f0
        .iter()
        .zip(fl)
        .map(|(a, b)| (*a as i32 - *b as i32).unsigned_abs() as u64)
        .sum();
    let per_px = motion as f64 / f0.len() as f64;
    eprintln!("frame0→frameN mean abs diff/channel: {per_px:.3}");
    assert!(per_px > 0.1, "no motion across the clip (frames identical)");

    let n = frames.len();
    for &i in &[0usize, n / 2, n - 1] {
        let p = out_dir.join(format!("svd_frame_{i:02}.png"));
        write_png(&p, &frames[i])?;
        eprintln!("wrote {}", p.display());
    }
    eprintln!("SVD GPU smoke OK");
    Ok(())
}
