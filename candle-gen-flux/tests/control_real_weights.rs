//! sc-8412: real-weight GPU validation for the candle FLUX.1-dev Fun-Controlnet-Union control provider —
//! env-driven, `#[ignore]`d. Drives the REAL [`Flux1DevControl`] stack on the deployed hardware (a
//! FLUX.1-dev snapshot + `Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0` + a control image). Mirrors
//! the FLUX.2 control real-weight smoke + the merged mlx `control_real_weights.rs`.
//!
//! Two gates:
//!   - **`control_scale = 0 ≡ base`**: a control render at `control_scale = 0` (the residuals are ×0) is
//!     byte-identical to a plain FLUX.1-dev render at the same seed/prompt (the unconditioned base) —
//!     proving the control branch is a clean overlay that vanishes at scale 0.
//!   - **Coherent steer**: a `control_scale = 0.7` render of the pose/canny/depth hint differs from the
//!     base, is fully finite, and is written as a PPM for eyeballing.
//!
//! Run (after deploying weights into a local dir), GPU 1 only:
//! ```text
//! set CUDA_VISIBLE_DEVICES=1
//! set FLUX1_CTRL_BASE=...\FLUX.1-dev                 # BFL snapshot (flux1-dev.safetensors, ae.safetensors, …)
//! set FLUX1_CTRL_OVERLAY=...\Union-Pro-2.0.safetensors   # the Shakker control checkpoint (file or dir)
//! set FLUX1_CTRL_IMAGE=...\pose.ppm                  # the control hint (P6 PPM)
//! set FLUX1_CTRL_KIND=pose                           # pose | canny | depth
//! set FLUX1_CTRL_OUT=...\out                         # output dir
//! cargo test -p candle-gen-flux --features cuda --release --test control_real_weights -- --ignored --nocapture
//! ```

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use candle_gen::gen_core::runtime::CancelFlag;
use candle_gen::gen_core::Progress;
use candle_gen::testkit::{env_path, read_ppm, write_ppm};
use candle_gen_flux::{Flux1ControlPaths, Flux1ControlRequest, Flux1DevControl};

fn max_abs_diff_u8(a: &[u8], b: &[u8]) -> u32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (*x as i32 - *y as i32).unsigned_abs())
        .max()
        .unwrap_or(0)
}

/// Real-weight: render the control hint at `control_scale = 0.7` (the steer), assert a coherent finite
/// 1024² image distinct across seeds, and exercise the cancel contract (pre + mid-denoise). The
/// byte-exact `control_scale = 0 ≡ base` invariant is the GPU-free `control_parity` CI gate (this real
/// render exercises the same `forward_composed` seam at runtime). PPMs land in `FLUX1_CTRL_OUT`.
#[test]
#[ignore = "real-weight GPU validation; set FLUX1_CTRL_BASE/FLUX1_CTRL_OVERLAY/FLUX1_CTRL_IMAGE/FLUX1_CTRL_KIND/FLUX1_CTRL_OUT"]
fn real_weight_control() {
    let out_dir = env_path("FLUX1_CTRL_OUT");
    std::fs::create_dir_all(&out_dir).ok();
    let kind = std::env::var("FLUX1_CTRL_KIND").unwrap_or_else(|_| "pose".into());

    let paths = Flux1ControlPaths {
        flux_base: env_path("FLUX1_CTRL_BASE"),
        control: env_path("FLUX1_CTRL_OVERLAY"),
    };
    let control_image = read_ppm(&env_path("FLUX1_CTRL_IMAGE"));
    println!(
        "control hint {}x{} (kind {kind}); loading Flux1DevControl …",
        control_image.width, control_image.height
    );

    let t0 = std::time::Instant::now();
    let model = Flux1DevControl::load(&paths).expect("load Flux1DevControl");
    println!(
        "loaded in {:?} — {} residuals, interval {}",
        t0.elapsed(),
        model.num_residuals(),
        model.residual_interval()
    );
    assert_eq!(
        model.num_residuals(),
        6,
        "Shakker Union-Pro-2.0 ships 6 control double blocks"
    );
    assert_eq!(model.residual_interval(), 4, "ceil(19/6) = 4");

    let prompt = "a cinematic portrait photo of a person, soft natural light, photorealistic";
    let base = Flux1ControlRequest {
        prompt: prompt.into(),
        width: 1024,
        height: 1024,
        steps: 20,
        guidance: 3.5,
        control_scale: Some(0.7),
        control_kind: kind.clone(),
        seed: 12345,
        cancel: CancelFlag::new(),
    };
    let mut noop = |_p: Progress| {};

    // Steered render (control_scale = 0.7).
    let t = std::time::Instant::now();
    let steered = model
        .generate(&base, &control_image, &mut noop)
        .expect("generate (steered)");
    println!("[steer] {:?}", t.elapsed());
    write_ppm(&out_dir.join(format!("flux1_control_{kind}.ppm")), &steered);
    assert_eq!((steered.width, steered.height), (1024, 1024));
    assert_eq!(steered.pixels.len(), 1024 * 1024 * 3, "full RGB8 buffer");
    // A coherent render is not a flat field (the VAE decode of pure noise / a dead forward would be).
    let mn = *steered.pixels.iter().min().unwrap();
    let mx = *steered.pixels.iter().max().unwrap();
    assert!(
        mx > mn + 8,
        "steered render must have tonal range (got {mn}..{mx})"
    );

    // sc-8988 determinism gate: the same seed renders byte-identically — the control encode takes the
    // VAE posterior MEAN (no device RNG), so nothing in the pipeline samples outside the CPU-seeded
    // `StdRng` (the sc-3673 contract). Pre-fix, the sampled control latent made this per-launch at best.
    let again = model
        .generate(&base, &control_image, &mut noop)
        .expect("generate (repeat seed 12345)");
    let d_same = max_abs_diff_u8(&steered.pixels, &again.pixels);
    println!("max|Δ| same seed = {d_same} (must be 0)");
    assert_eq!(
        d_same, 0,
        "same seed must render byte-identically (sc-8988)"
    );

    // A second seed yields a different (real) render — confirms it is not a constant output.
    let other = Flux1ControlRequest {
        seed: 999,
        ..base.clone()
    };
    let steered2 = model
        .generate(&other, &control_image, &mut noop)
        .expect("generate (seed 999)");
    let d = max_abs_diff_u8(&steered.pixels, &steered2.pixels);
    println!("max|Δ| across seeds = {d} (must be > 0)");
    assert!(d > 0, "two seeds must produce different images");

    // The accepted-kind gate: an unmodelled kind is rejected loudly (input-agnostic union — pose/canny/
    // depth only).
    let bad = Flux1ControlRequest {
        control_kind: "scribble".into(),
        ..base.clone()
    };
    assert!(
        model.generate(&bad, &control_image, &mut noop).is_err(),
        "an unsupported control kind must be rejected"
    );

    // Cancel contract: pre-cancel returns Canceled.
    let cancelled = Flux1ControlRequest {
        cancel: {
            let c = CancelFlag::new();
            c.cancel();
            c
        },
        ..base.clone()
    };
    assert!(
        matches!(
            model.generate(&cancelled, &control_image, &mut noop),
            Err(candle_gen::CandleError::Canceled)
        ),
        "pre-cancel must return Canceled"
    );

    // Mid-denoise cancel: flip the flag from the progress callback at step 3.
    let mid = CancelFlag::new();
    let mid_req = Flux1ControlRequest {
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
    assert!(
        matches!(
            model.generate(&mid_req, &control_image, &mut cancel_at_3),
            Err(candle_gen::CandleError::Canceled)
        ),
        "mid-cancel must return Canceled"
    );
    println!(
        "[cancel:mid] Canceled after {} steps ✓",
        seen.load(Ordering::SeqCst)
    );

    println!(
        "FLUX.1-dev control validation PASS — outputs in {}",
        out_dir.display()
    );
}
