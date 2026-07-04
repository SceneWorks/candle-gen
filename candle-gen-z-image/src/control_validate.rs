//! Z-Image Fun-ControlNet (strict-pose) real-weight GPU validation (sc-5489, epic 5480) — an
//! env-driven, `#[ignore]`d integration test that drives the REAL [`ZImageControl`] stack on the
//! deployed hardware (a `Tongyi-MAI/Z-Image-Turbo` snapshot + the
//! `alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1` checkpoint + a rendered pose skeleton). The
//! Z-Image sibling of the Qwen/Kolors strict-pose harnesses.
//!
//! **Gate.** A strict-pose ControlNet should make the generation *follow* the skeleton, so the metric
//! is a with-control vs no-control ablation: generate twice at one seed — **with** control
//! (`control_scale > 0`) and **without** (`control_scale = 0` → the VACE hints contribute zero, so the
//! forward reduces to the base Z-Image txt2img) — and assert the outputs differ meaningfully. Plus the
//! cancel contract. The "does it match the pose" judgement is the eyeball check on the written PPMs.
//!
//! Run (after deploying weights into a local dir):
//! ```text
//! set ZIMG_CTRL_BASE=...\Z-Image-Turbo          # tokenizer/ text_encoder/ transformer/ vae/
//! set ZIMG_CTRL_NET=...\Z-Image-Turbo-Fun-Controlnet-Union-2.1.safetensors   # file or dir
//! set ZIMG_CTRL_POSE=...\skeleton.ppm           # a rendered OpenPose skeleton at the request size (P6)
//! set ZIMG_CTRL_OUT=...\out
//! cargo test -p candle-gen-z-image --features cuda --release control_validate::real_weight -- --ignored --nocapture
//! ```
//!
//! **Base mode (sc-8680).** A second `#[ignore]`d test, `real_weight_control_base`, drives the
//! **undistilled base** control path (shift-6.0, ~50-step, real CFG) — the candle sibling of the MLX base
//! control variant. Point `ZIMG_CTRL_BASE` at a `Tongyi-MAI/Z-Image` (non-Turbo) snapshot and
//! `ZIMG_CTRL_NET` at the base `Z-Image-Fun-Controlnet-Union-2.1` checkpoint (a **dir** exercises the
//! deterministic overlay resolution — it must pick the Union file, not a Tile-lite sibling), then run
//! `control_validate::real_weight_control_base -- --ignored --nocapture`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use candle_gen::gen_core::runtime::CancelFlag;
use candle_gen::gen_core::Progress;
use candle_gen::testkit::{env_path, mean_abs_diff, read_ppm, write_ppm};

use crate::control::{ZImageControl, ZImageControlPaths, ZImageControlRequest};

#[test]
#[ignore = "real-weight GPU validation; set ZIMG_CTRL_BASE/ZIMG_CTRL_NET/ZIMG_CTRL_POSE/ZIMG_CTRL_OUT"]
fn real_weight_control() {
    run_control_validation(false, "turbo", 8, None, None);
}

/// Base-mode (sc-8680) real-weight validation: point `ZIMG_CTRL_BASE` at a `Tongyi-MAI/Z-Image`
/// (non-Turbo) snapshot and `ZIMG_CTRL_NET` at the base `Z-Image-Fun-Controlnet-Union-2.1` checkpoint
/// (a **dir** exercises the Union-vs-Tile-lite overlay resolution). Runs the undistilled shift-6.0
/// ~50-step CFG path (guidance 4.0 + a negative prompt) — the ablation gate + cancel contract are
/// identical.
#[test]
#[ignore = "real-weight GPU validation (base mode); set ZIMG_CTRL_BASE/ZIMG_CTRL_NET/ZIMG_CTRL_POSE/ZIMG_CTRL_OUT"]
fn real_weight_control_base() {
    run_control_validation(
        true,
        "base",
        50,
        Some(4.0),
        Some("blurry, low quality, deformed"),
    );
}

/// The shared control-validation harness (sc-8680): loads the model in the requested mode (`base`),
/// runs a with-control vs no-control (scale 0) ablation, checks the pre-/mid-denoise cancel contract,
/// and asserts the control path meaningfully changes the output. `steps`/`guidance`/`negative` tune the
/// per-mode request (Turbo ignores guidance/negative; base uses them).
fn run_control_validation(
    base: bool,
    tag: &str,
    steps: usize,
    guidance: Option<f32>,
    negative: Option<&str>,
) {
    let out_dir = env_path("ZIMG_CTRL_OUT");
    std::fs::create_dir_all(&out_dir).ok();

    let paths = ZImageControlPaths {
        snapshot: env_path("ZIMG_CTRL_BASE"),
        control: env_path("ZIMG_CTRL_NET"),
        base,
    };
    let skeleton = read_ppm(&env_path("ZIMG_CTRL_POSE"));
    println!(
        "[{tag}] pose skeleton {}x{}; loading ZImageControl (base={base}) …",
        skeleton.width, skeleton.height
    );

    let t0 = std::time::Instant::now();
    let model = ZImageControl::load(&paths).expect("load ZImageControl");
    println!("[{tag}] loaded in {:?}", t0.elapsed());

    let req = ZImageControlRequest {
        prompt: "a person standing, full body, photorealistic, studio lighting, sharp focus".into(),
        width: skeleton.width,
        height: skeleton.height,
        steps,
        control_scale: 1.0,
        guidance,
        negative_prompt: negative.map(str::to_string),
        seed: 12345,
        // Native VAE: this harness validates the pose-control pipeline, not the optional PiD SR (sc-8044).
        use_pid: false,
        cancel: CancelFlag::new(),
    };

    let mut noop = |_p: Progress| {};

    // With control.
    let t = std::time::Instant::now();
    let out_ctrl = model
        .generate(&req, &skeleton, &mut noop)
        .expect("generate (control)");
    println!("[{tag}][control] {:?}", t.elapsed());
    write_ppm(
        &out_dir.join(format!("zimage_control_{tag}.ppm")),
        &out_ctrl,
    );

    // Without control (scale 0 → the VACE hints contribute zero → plain Z-Image at the same seed/prompt).
    let plain_req = ZImageControlRequest {
        control_scale: 0.0,
        ..req.clone()
    };
    let t = std::time::Instant::now();
    let out_plain = model
        .generate(&plain_req, &skeleton, &mut noop)
        .expect("generate (no control)");
    println!("[{tag}][no-control] {:?}", t.elapsed());
    write_ppm(
        &out_dir.join(format!("zimage_no_control_{tag}.ppm")),
        &out_plain,
    );

    let diff = mean_abs_diff(&out_ctrl, &out_plain);
    println!("=== Z-Image Fun-ControlNet validation ({tag}) ===");
    println!("  mean abs pixel diff (control vs no-control): {diff:.2}");
    println!("  outputs: {}", out_dir.display());

    // Pre-cancel.
    let cancelled = ZImageControlRequest {
        cancel: {
            let c = CancelFlag::new();
            c.cancel();
            c
        },
        ..req.clone()
    };
    let pre = model.generate(&cancelled, &skeleton, &mut noop);
    assert!(
        matches!(pre, Err(candle_gen::CandleError::Canceled)),
        "pre-cancel must return Canceled"
    );
    println!("[{tag}][cancel:pre] Err(Canceled) ✓");

    // Mid-denoise cancel on step 3.
    let mid = CancelFlag::new();
    let mid_req = ZImageControlRequest {
        cancel: mid.clone(),
        ..req.clone()
    };
    let seen = Arc::new(AtomicUsize::new(0));
    let seen_cb = seen.clone();
    let mut cancel_at_3 = move |p: Progress| {
        if let Progress::Step { current, .. } = p {
            seen_cb.store(current as usize, Ordering::SeqCst);
            if current >= 3 {
                mid.cancel();
            }
        }
    };
    let res = model.generate(&mid_req, &skeleton, &mut cancel_at_3);
    assert!(
        matches!(res, Err(candle_gen::CandleError::Canceled)),
        "mid-cancel must return Canceled"
    );
    let steps_seen = seen.load(Ordering::SeqCst);
    assert!(
        (3..=4).contains(&steps_seen),
        "mid-cancel should stop right after step 3 (saw {steps_seen})"
    );
    println!("[{tag}][cancel:mid] Err(Canceled) after {steps_seen} steps ✓");

    // The gate: the control path meaningfully changes the output (it actually conditions the image).
    assert!(
        diff > 5.0,
        "control vs no-control mean diff {diff:.2} too small — control may not be wired"
    );
    println!(
        "[{tag}] Z-Image Fun-ControlNet validation PASS ✅ (eyeball the PPMs for pose adherence)"
    );
}
