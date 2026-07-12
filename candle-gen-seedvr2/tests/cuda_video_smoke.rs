//! Real-checkpoint CUDA video smoke / functional validation for the SeedVR2 video mode (sc-5926).
//!
//! `#[ignore]` by default (needs the weights + a GPU build). Run on the Blackwell box with:
//! ```text
//! set SEEDVR2_CKPT=D:\sceneworks-seedvr2-validate\ckpt
//! cargo test -p candle-gen-seedvr2 --features cuda --release --test cuda_video_smoke -- --ignored --nocapture
//! ```
//! `SEEDVR2_CKPT` is a dir holding `ema_vae_fp16.safetensors` + `seedvr2_ema_3b_fp16.safetensors`.
//! Optional: `SEEDVR2_DTYPE=bf16` (default f32).
//!
//! A *functional* validation (does video mode run E2E on CUDA and produce a faithful, frame-count-
//! preserving, seam-free, temporally-coherent upscale?), not a bit-exact parity check. It exercises
//! the two non-trivial orchestration paths sc-5926 adds on top of the (already validated) 5-D model
//! pass:
//!   1. **temporal chunking + overlap cross-fade** — a 20-frame clip with `chunk=16` → two chunks
//!      `[0,16]`/`[4,20]` whose 4-frame overlap is cross-faded. Asserts the frame count is preserved,
//!      every frame is finite/non-degenerate + structurally faithful, the chunk-boundary frame-to-frame
//!      delta is NOT a seam spike vs the interior, and the chunked path is at least as temporally
//!      coherent as the independent per-frame path.
//!   2. **HD spatial tiling** ([`Seedvr2Pipeline::run_frame_tiled`]) — one frame upscaled in
//!      overlapping feather-blended tiles must match the untiled single-pass upscale (high correlation),
//!      proving the tile partition + feather assembly is numerically sound.

use candle_gen::candle_core::DType;
use candle_gen::gen_core::{imageops, Image};
use candle_gen_seedvr2::config::DitConfig;
use candle_gen_seedvr2::pipeline::Seedvr2Pipeline;
use candle_gen_seedvr2::video;

const DIT_FILE: &str = "seedvr2_ema_3b_fp16.safetensors";

/// A deterministic structured frame (gradients + checkerboard + circles), translated by `(dx,dy)` so a
/// clip has real, *coherent* motion for the temporal path to act on.
fn synth_frame(side: usize, dx: i32, dy: i32) -> Image {
    let mut pixels = vec![0u8; side * side * 3];
    let s = side as i32;
    for y in 0..side {
        for x in 0..side {
            let i = (y * side + x) * 3;
            let sx = ((x as i32 - dx).rem_euclid(s)) as usize;
            let sy = ((y as i32 - dy).rem_euclid(s)) as usize;
            let check = (((sx / 12) + (sy / 12)) % 2) as u8 * 90;
            let cx = side as f32 / 2.0;
            let dr = (((sx as f32 - cx).powi(2) + (sy as f32 - cx).powi(2)).sqrt() * 0.18).sin();
            pixels[i] = (sx * 255 / side) as u8; // R gradient
            pixels[i + 1] = (40 + check as usize).min(255) as u8; // G checkerboard
            pixels[i + 2] = (((dr + 1.0) * 0.5) * 255.0) as u8; // B rings
        }
    }
    Image {
        width: side as u32,
        height: side as u32,
        pixels,
    }
}

/// A `n`-frame clip translating ≤ `motion` px/frame (the "realistic motion" of the acceptance bar).
fn synth_clip(side: usize, n: usize, motion: i32) -> Vec<Image> {
    (0..n)
        .map(|t| synth_frame(side, t as i32 * motion, (t as i32 * motion) / 2))
        .collect()
}

fn pearson(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (
        a.iter().map(|&v| v as f64).sum::<f64>() / n,
        b.iter().map(|&v| v as f64).sum::<f64>() / n,
    );
    let (mut cov, mut va, mut vb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b.iter()) {
        let (dx, dy) = (x as f64 - ma, y as f64 - mb);
        cov += dx * dy;
        va += dx * dx;
        vb += dy * dy;
    }
    cov / (va.sqrt() * vb.sqrt()).max(1e-12)
}

/// Mean |byte difference| between two equal-size RGB8 frames (a frame-to-frame change proxy).
fn frame_delta(a: &Image, b: &Image) -> f64 {
    let n = a.pixels.len().min(b.pixels.len());
    let sum: u64 = (0..n)
        .map(|i| (a.pixels[i] as i32 - b.pixels[i] as i32).unsigned_abs() as u64)
        .sum();
    sum as f64 / n.max(1) as f64
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if v.is_empty() {
        0.0
    } else {
        v[v.len() / 2]
    }
}

fn load_pipe() -> Option<(Seedvr2Pipeline, DType)> {
    let ckpt = match std::env::var("SEEDVR2_CKPT") {
        Ok(p) => p,
        Err(_) => {
            eprintln!("SKIP: set SEEDVR2_CKPT to a numz/SeedVR2_comfyUI checkpoint dir");
            return None;
        }
    };
    let dtype = match std::env::var("SEEDVR2_DTYPE").as_deref() {
        Ok("bf16") => DType::BF16,
        _ => DType::F32,
    };
    let device = candle_gen::default_device().expect("device");
    eprintln!(
        "[seedvr2-video] device={device:?} dtype={dtype:?} budget={:.1} GiB ckpt={ckpt}",
        video::safe_budget_gib()
    );
    let cfg = DitConfig::seedvr2_3b();
    let pipe = Seedvr2Pipeline::load(&ckpt, DIT_FILE, &cfg, dtype, &device).expect("load pipeline");
    Some((pipe, dtype))
}

#[test]
#[ignore = "needs SEEDVR2_CKPT weights + a CUDA build"]
fn cuda_video_chunk_overlap_smoke() {
    let Some((pipe, _dt)) = load_pipe() else {
        return;
    };

    let (src, tgt) = (256usize, 512usize); // 2× upscale; 512 is ÷16
    let n = 20usize;
    let clip = synth_clip(src, n, 2); // ≤2 px/frame — realistic motion

    // chunk=16 forces TWO chunks ([0,16],[4,20]) so the 4-frame overlap cross-fade is exercised.
    // Collect real per-chunk progress (sc-11227): must be monotonic, 1-based, and end at (total,total)
    // — NOT the old fixed `Step { 1, 1 }` placeholder.
    let t0 = std::time::Instant::now();
    let mut steps: Vec<(usize, usize)> = Vec::new();
    let out = {
        let mut on_step = |done: usize, total: usize| steps.push((done, total));
        pipe.generate_video(&clip, tgt, tgt, 42, 0.0, Some(16), None, Some(&mut on_step))
            .expect("generate_video")
    };
    assert!(!steps.is_empty(), "progress must be reported per chunk");
    let total = steps[0].1;
    assert!(
        total >= 2,
        "chunk=16 over 20 frames → ≥2 chunks, got {total}"
    );
    assert_eq!(steps.first().copied(), Some((1, total)), "1-based start");
    assert_eq!(steps.last().copied(), Some((total, total)), "ends at total");
    assert!(
        steps.windows(2).all(|w| w[1].0 > w[0].0),
        "progress strictly increasing (real, not placeholder): {steps:?}"
    );

    // sc-11227: a cancel tripped mid-clip is honored promptly (per chunk), surfacing the typed
    // `Canceled` rather than running the whole (minutes-to-hours) upscale to completion.
    let cancel = candle_gen::gen_core::CancelFlag::new();
    cancel.cancel();
    let canceled = pipe.generate_video(&clip, tgt, tgt, 42, 0.0, Some(16), Some(&cancel), None);
    assert!(
        matches!(canceled, Err(candle_gen::CandleError::Canceled)),
        "a tripped cancel flag must stop the upscale loop early with typed Canceled"
    );
    eprintln!(
        "[seedvr2-video] {n}×{src}→{tgt} chunked in {:?} -> {} frames",
        t0.elapsed(),
        out.len()
    );

    // frame count preserved + dims + non-degenerate per frame.
    assert_eq!(out.len(), n, "frame count must be preserved");
    let ranges: Vec<(u8, u8)> = out
        .iter()
        .map(|f| {
            (
                *f.pixels.iter().min().unwrap(),
                *f.pixels.iter().max().unwrap(),
            )
        })
        .collect();
    eprintln!("[seedvr2-video] per-frame (min,max) = {ranges:?}");
    for (i, f) in out.iter().enumerate() {
        assert_eq!(
            (f.width, f.height),
            (tgt as u32, tgt as u32),
            "frame {i} dims"
        );
        let (mn, mx) = ranges[i];
        assert!(
            mx > mn,
            "frame {i} is constant (degenerate): min={mn} max={mx}"
        );
    }

    // structural faithfulness: each output frame correlates with the bicubic upscale of its LR frame.
    let mut min_corr = 1.0f64;
    for (i, f) in out.iter().enumerate() {
        let base = imageops::resize_bicubic_u8(&clip[i].pixels, src, src, tgt, tgt).unwrap();
        let of: Vec<f32> = f.pixels.iter().map(|&v| v as f32).collect();
        min_corr = min_corr.min(pearson(&of, &base));
    }
    eprintln!("[seedvr2-video] min per-frame corr_vs_bicubic = {min_corr:.4}");
    assert!(
        min_corr > 0.7,
        "a frame is not structurally faithful (corr={min_corr:.4})"
    );

    // seam check: the chunk-boundary frame-to-frame delta must not spike vs the interior median.
    // With chunk=16/overlap=4 the seam (if any) would land in the second chunk's leading frames.
    let deltas: Vec<f64> = (1..out.len())
        .map(|i| frame_delta(&out[i - 1], &out[i]))
        .collect();
    let med = median(deltas.clone());
    let max_delta = deltas.iter().cloned().fold(0.0f64, f64::max);
    eprintln!("[seedvr2-video] frame-Δ median={med:.3} max={max_delta:.3} all={deltas:?}");
    assert!(
        max_delta <= (med * 4.0).max(med + 8.0),
        "a chunk seam is visible: max frame-Δ {max_delta:.3} ≫ median {med:.3}"
    );

    // coherence: the temporal (chunked) path should be at least as smooth as the independent
    // per-frame path (same anchored seed). Force per-frame via a budget override where ONE 512²
    // frame fits but an 8-frame chunk doesn't (18 GiB > weights+1-frame for both bf16 & f32).
    std::env::set_var("SEEDVR2_BUDGET_GIB", "18");
    let per_frame = pipe
        .generate_video(&clip, tgt, tgt, 42, 0.0, None, None, None)
        .expect("generate_video per-frame fallback");
    std::env::remove_var("SEEDVR2_BUDGET_GIB");
    assert_eq!(per_frame.len(), n, "fallback frame count must be preserved");
    let pf_deltas: Vec<f64> = (1..per_frame.len())
        .map(|i| frame_delta(&per_frame[i - 1], &per_frame[i]))
        .collect();
    let pf_mean = pf_deltas.iter().sum::<f64>() / pf_deltas.len() as f64;
    let ck_mean = deltas.iter().sum::<f64>() / deltas.len() as f64;
    eprintln!(
        "[seedvr2-video] coherence: chunked mean-Δ={ck_mean:.3} vs per-frame mean-Δ={pf_mean:.3}"
    );
    assert!(
        ck_mean <= pf_mean * 1.5 + 1.0,
        "chunked path is markedly less coherent than per-frame ({ck_mean:.3} vs {pf_mean:.3})"
    );
}

/// Isolation diagnostic: VAE round-trip (encode → decode, NO DiT) on a MOVING 16-frame clip. If the
/// decoded clip is in range + faithful, the VAE handles distinct T'=4 latents fine and the moving-clip
/// failure is in the DiT (temporal windowing at tp=4); if it explodes, it's a VAE integration bug.
#[test]
#[ignore = "needs SEEDVR2_CKPT weights + a CUDA build"]
fn cuda_video_vae_roundtrip_smoke() {
    use candle_gen::candle_core::Tensor;
    let Some((pipe, _dt)) = load_pipe() else {
        return;
    };
    let (src, tgt) = (256usize, 512usize);
    let clip = synth_clip(src, 16, 2);
    // Build (1,3,16,512,512) via the public preprocess + temporal cat.
    let per: Vec<Tensor> = clip
        .iter()
        .map(|f| {
            pipe.preprocess(f, tgt, tgt, 0.0)
                .unwrap()
                .unsqueeze(2)
                .unwrap()
        })
        .collect();
    let refs: Vec<&Tensor> = per.iter().collect();
    let x = Tensor::cat(&refs, 2).unwrap(); // (1,3,16,512,512)
    let latent = pipe.vae.encode(&x).unwrap();
    let decoded = pipe.vae.decode(&latent).unwrap(); // (1,3,16,512,512), NO DiT
    let v = decoded
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let (mn, mx) = v
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &x| {
            (a.min(x), b.max(x))
        });
    eprintln!(
        "[seedvr2-video] VAE-roundtrip (no DiT) decoded min={mn:.3} max={mx:.3} latentT={}",
        latent.dims()[2]
    );
    // A correct no-DiT reconstruction lands near [-1,1] (some overshoot); the pre-fix bug gave ±625.
    assert!(
        mn > -15.0 && mx < 15.0,
        "VAE round-trip explodes on a moving clip (min={mn:.3} max={mx:.3}) — VAE bug, not DiT"
    );
}

/// Isolation diagnostic: an 8-frame clip of IDENTICAL frames should decode to ~8 copies of the
/// single-frame (T=1) upscale. Encoder temporal convs see identical frames (≈ the T=1 repeat), so any
/// divergence isolates the decoder's T>1 temporal-upsample structure from motion/distinct-frame
/// effects.
#[test]
#[ignore = "needs SEEDVR2_CKPT weights + a CUDA build"]
fn cuda_video_identical_frames_smoke() {
    let Some((pipe, _dt)) = load_pipe() else {
        return;
    };
    let (src, tgt) = (256usize, 512usize);
    let frame = synth_frame(src, 0, 0);
    let img = pipe
        .generate(&frame, tgt, tgt, 11, 0.0)
        .expect("T=1 baseline");
    let clip: Vec<Image> = (0..8).map(|_| frame.clone()).collect();
    let vid = pipe
        .generate_video(&clip, tgt, tgt, 11, 0.0, Some(8), None, None)
        .expect("identical-frames video");
    assert_eq!(vid.len(), 8);
    let base: Vec<f32> = img.pixels.iter().map(|&v| v as f32).collect();
    for (i, f) in vid.iter().enumerate() {
        let fv: Vec<f32> = f.pixels.iter().map(|&v| v as f32).collect();
        let corr = pearson(&fv, &base);
        let (mn, mx) = (
            *f.pixels.iter().min().unwrap(),
            *f.pixels.iter().max().unwrap(),
        );
        eprintln!("[seedvr2-video] identical frame {i}: corr_vs_T1={corr:.4} min={mn} max={mx}");
        assert!(
            corr > 0.9,
            "identical-frames video frame {i} diverges from the T=1 upscale (corr={corr:.4}) — \
             decoder T>1 structural bug"
        );
    }
}

#[test]
#[ignore = "needs SEEDVR2_CKPT weights + a CUDA build"]
fn cuda_video_hd_tiling_smoke() {
    let Some((pipe, dt)) = load_pipe() else {
        return;
    };

    // One mid-res frame, upscaled two ways: untiled single pass vs spatially tiled (256-px tiles,
    // overlap 64). The feather-blended tiled result must closely match the single-pass result.
    let (src, tgt) = (256usize, 512usize);
    let frame = synth_frame(src, 0, 0);

    let untiled = pipe
        .generate(&frame, tgt, tgt, 7, 0.0)
        .expect("single-pass");

    let processed = {
        // (1,3,1,H,W) preprocessed clip the tiler consumes.
        let p = pipe.preprocess(&frame, tgt, tgt, 0.0).expect("preprocess");
        p.unsqueeze(2).expect("add T axis")
    };
    let t0 = std::time::Instant::now();
    let decoded = pipe
        .run_frame_tiled(&processed, 7, 256, 64)
        .expect("run_frame_tiled");
    eprintln!(
        "[seedvr2-video] HD tiling {tgt}² (256-px tiles) in {:?} -> {:?}",
        t0.elapsed(),
        decoded.dims()
    );

    // decoded is (1,3,1,H,W) in [-1,1]; bring to RGB8 the same way the pipeline does.
    let u8s = ((decoded.clamp(-1f32, 1f32).unwrap() + 1.0).unwrap() * 127.5)
        .unwrap()
        .to_dtype(DType::U8)
        .unwrap()
        .to_device(&candle_gen::candle_core::Device::Cpu)
        .unwrap();
    let chw = u8s.squeeze(0).unwrap().squeeze(1).unwrap(); // (3,H,W)
    let tiled_px = chw
        .permute((1, 2, 0))
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<u8>()
        .unwrap();
    let _ = dt;

    assert_eq!(tiled_px.len(), tgt * tgt * 3, "tiled frame size");
    let a: Vec<f32> = untiled.pixels.iter().map(|&v| v as f32).collect();
    let b: Vec<f32> = tiled_px.iter().map(|&v| v as f32).collect();
    let corr = pearson(&a, &b);
    eprintln!("[seedvr2-video] tiled-vs-untiled corr = {corr:.4}");
    assert!(
        corr > 0.9,
        "spatial tiling diverges from the single-pass upscale (corr={corr:.4}) — feather/partition bug"
    );
}
