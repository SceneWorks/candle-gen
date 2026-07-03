//! Qwen-Image ControlNet (strict-pose) real-weight GPU validation (sc-5489, epic 5480) — an env-driven,
//! `#[ignore]`d integration test that drives the REAL [`QwenControl`] stack on the deployed hardware (a
//! `Qwen/Qwen-Image` snapshot + the InstantX `Qwen-Image-ControlNet-Union` checkpoint + a rendered pose
//! skeleton). The analog of the IP-Adapter Phase-5 harnesses.
//!
//! **Gate.** A strict-pose ControlNet should make the generation *follow* the skeleton, so the metric is
//! a with-control vs no-control ablation: generate twice at one seed — **with** control
//! (`control_scale > 0`) and **without** (`control_scale = 0` → the control residuals are zeroed, so the
//! forked forward is byte-identical to plain txt2img) — and assert the outputs differ meaningfully (the
//! control path actually conditions the image). Plus the cancel contract. The "does it match the pose"
//! judgement is the eyeball check on the written PPMs.
//!
//! Run (after deploying weights into a local dir):
//! ```text
//! set QWEN_CTRL_BASE=...\Qwen-Image          # diffusers snapshot (text_encoder/ transformer/ vae/ tokenizer/)
//! set QWEN_CTRL_NET=...\qwen_controlnet_union.safetensors   # InstantX Union (file or dir)
//! set QWEN_CTRL_POSE=...\skeleton.ppm         # a rendered OpenPose skeleton at the request size (P6)
//! set QWEN_CTRL_OUT=...\out
//! cargo test -p candle-gen-qwen-image --features cuda --release control_validate::real_weight -- --ignored --nocapture
//! ```

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use candle_gen::gen_core::runtime::CancelFlag;
use candle_gen::gen_core::Progress;
use candle_gen::testkit::{env_path, mean_abs_diff, read_ppm, write_ppm};

use crate::control::{QwenControl, QwenControlPaths, QwenControlRequest};

#[test]
#[ignore = "real-weight GPU validation; set QWEN_CTRL_BASE/QWEN_CTRL_NET/QWEN_CTRL_POSE/QWEN_CTRL_OUT"]
fn real_weight_control() {
    let out_dir = env_path("QWEN_CTRL_OUT");
    std::fs::create_dir_all(&out_dir).ok();

    let paths = QwenControlPaths {
        qwen_base: env_path("QWEN_CTRL_BASE"),
        controlnet: env_path("QWEN_CTRL_NET"),
    };
    let skeleton = read_ppm(&env_path("QWEN_CTRL_POSE"));
    println!(
        "pose skeleton {}x{}; loading QwenControl …",
        skeleton.width, skeleton.height
    );

    let t0 = std::time::Instant::now();
    let model = QwenControl::load(&paths).expect("load QwenControl");
    println!("loaded in {:?}", t0.elapsed());

    let base = QwenControlRequest {
        prompt: "a person standing, full body, photorealistic, studio lighting, sharp focus".into(),
        negative: "blurry, lowres, deformed, extra limbs, watermark".into(),
        width: skeleton.width,
        height: skeleton.height,
        steps: 20,
        guidance: 4.0,
        control_scale: 1.0,
        seed: 12345,
        cancel: CancelFlag::new(),
    };

    let mut noop = |_p: Progress| {};

    // With control.
    let t = std::time::Instant::now();
    let out_ctrl = model
        .generate(&base, &skeleton, &mut noop)
        .expect("generate (control)");
    println!("[control] {:?}", t.elapsed());
    write_ppm(&out_dir.join("qwen_control.ppm"), &out_ctrl);

    // Without control (scale 0 → control residuals zeroed → plain txt2img at the same seed/prompt).
    let plain_req = QwenControlRequest {
        control_scale: 0.0,
        ..base.clone()
    };
    let t = std::time::Instant::now();
    let out_plain = model
        .generate(&plain_req, &skeleton, &mut noop)
        .expect("generate (no control)");
    println!("[no-control] {:?}", t.elapsed());
    write_ppm(&out_dir.join("qwen_no_control.ppm"), &out_plain);

    let diff = mean_abs_diff(&out_ctrl, &out_plain);
    println!("=== Qwen-Image ControlNet validation ===");
    println!("  mean abs pixel diff (control vs no-control): {diff:.2}");
    println!("  outputs: {}", out_dir.display());

    // Pre-cancel.
    let cancelled = QwenControlRequest {
        cancel: {
            let c = CancelFlag::new();
            c.cancel();
            c
        },
        ..base.clone()
    };
    let pre = model.generate(&cancelled, &skeleton, &mut noop);
    assert!(
        matches!(pre, Err(candle_gen::CandleError::Canceled)),
        "pre-cancel must return Canceled"
    );
    println!("[cancel:pre] Err(Canceled) ✓");

    // Mid-denoise cancel on step 3.
    let mid = CancelFlag::new();
    let mid_req = QwenControlRequest {
        cancel: mid.clone(),
        ..base.clone()
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
    println!("[cancel:mid] Err(Canceled) after {steps_seen} steps ✓");

    // The gate: the control path meaningfully changes the output (it actually conditions the image).
    assert!(
        diff > 5.0,
        "control vs no-control mean diff {diff:.2} too small — control may not be wired"
    );
    println!("Qwen-Image ControlNet validation PASS ✅ (eyeball the PPMs for pose adherence)");
}
