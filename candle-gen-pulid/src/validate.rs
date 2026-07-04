//! PuLID-FLUX real-weight GPU validation (sc-5492, epic 5480) — `#[ignore]`d by default.
//!
//! The unit tests in the sub-modules only prove shape/schedule with tiny random tensors. This drives the
//! REAL PuLID-FLUX stack (FLUX.1-dev + `guozinan/PuLID` + the converted EVA02-CLIP-L-14-336 + the native
//! SCRFD/ArcFace/BiSeNet face dir) on the GPU and asserts **identity recovery** — the ArcFace cosine
//! between the reference face and the generated face — for the id-conditioned render vs the `id_weight=0`
//! ablation (= plain FLUX), plus a pixel-diff sanity and pre-/mid-denoise cancellation. It re-embeds each
//! output through a fresh `candle-gen-face` stack, so a broken inference path (≈0 cosine) is caught
//! quantitatively, not just visually.
//!
//! Env-driven so no real weights live in the repo. Run (PowerShell, MSVC 14.44 vcvars +
//! `CUDA_COMPUTE_CAP=120`):
//! ```text
//! $env:PULID_FLUX_BASE = "<black-forest-labs/FLUX.1-dev snapshot dir>"
//! $env:PULID_WEIGHTS   = "<guozinan/PuLID>/pulid_flux_v0.9.1.safetensors"
//! $env:PULID_EVA       = "<converted EVA02-CLIP-L-14-336 .safetensors>"
//! $env:PULID_FACE_DIR  = "<dir: scrfd_10g + arcface_iresnet100 + bisenet_parsing>"
//! $env:PULID_REF       = "<reference face .ppm (P6)>"
//! $env:PULID_OUT       = "<output dir for the .ppm renders>"
//! cargo test -p candle-gen-pulid --features cuda --release validate::real_weight -- --ignored --nocapture
//! ```

#![cfg(test)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use candle_gen::gen_core::runtime::CancelFlag;
use candle_gen::gen_core::{FaceEmbedder, Image, Progress};
use candle_gen::testkit::{cosine, env_path, read_ppm, write_ppm};
use candle_gen::CandleError;

use crate::pulid_flux::{PulidFlux, PulidFluxPaths, PulidFluxRequest};

/// Mean absolute per-byte difference between two equal-size renders (the injection-changes-output sanity).
fn mean_abs_diff(a: &Image, b: &Image) -> f64 {
    assert_eq!(a.pixels.len(), b.pixels.len(), "render size mismatch");
    let sum: u64 = a
        .pixels
        .iter()
        .zip(&b.pixels)
        .map(|(x, y)| (*x as i32 - *y as i32).unsigned_abs() as u64)
        .sum();
    sum as f64 / a.pixels.len() as f64
}

/// Re-embed the largest face in `img` (via a fresh face stack) and report its cosine to `ref_emb`.
/// Returns `-1.0` (with a warning) when no face is detected.
fn output_cosine(face: &dyn FaceEmbedder, ref_emb: &[f32], img: &Image, tag: &str) -> f32 {
    match face.largest_face(img) {
        Ok(f) => {
            let c = cosine(ref_emb, &f.embedding);
            eprintln!(
                "[{tag}] face detected (det={:.3}) cosine={c:.4}",
                f.det_score
            );
            c
        }
        Err(e) => {
            eprintln!("[{tag}] WARNING: no face detected in output ({e}) — cosine n/a");
            -1.0
        }
    }
}

/// A coarse step-progress printer so a slow GPU run shows life under `--nocapture`.
fn make_progress() -> impl FnMut(Progress) {
    move |p: Progress| {
        if let Progress::Step { current, total } = p {
            if current == 1 || current % 5 == 0 || current == total {
                eprintln!("    step {current}/{total}");
            }
        }
    }
}

#[test]
#[ignore = "real-weight GPU validation; set PULID_* env + run with --features cuda --release"]
fn real_weight_pulid() {
    let out_dir = env_path("PULID_OUT");
    std::fs::create_dir_all(&out_dir).unwrap();

    let paths = PulidFluxPaths {
        flux_base: env_path("PULID_FLUX_BASE"),
        pulid_weights: env_path("PULID_WEIGHTS"),
        eva_weights: env_path("PULID_EVA"),
        face_dir: env_path("PULID_FACE_DIR"),
    };

    eprintln!("loading PulidFlux (FLUX.1-dev + PuLID + EVA-CLIP + face stack) ...");
    let t0 = Instant::now();
    let model = PulidFlux::load(&paths).expect("PulidFlux::load");
    // A separate face stack for re-embedding the outputs (same weights; the model's is private).
    let face = candle_gen_face::load_on(
        &env_path("PULID_FACE_DIR"),
        &candle_gen::default_device().unwrap(),
    )
    .expect("load face stack for re-embedding");
    eprintln!("loaded in {:?}", t0.elapsed());

    let reference = read_ppm(&env_path("PULID_REF"));
    let ref_emb = face
        .largest_face(&reference)
        .expect("detect a face in the reference image")
        .embedding;
    eprintln!(
        "reference {}x{} — ArcFace embedding {}-d",
        reference.width,
        reference.height,
        ref_emb.len()
    );

    let steps: usize = std::env::var("PULID_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(25);
    let base = PulidFluxRequest {
        prompt: "portrait of a person, color photo, cinematic lighting, sharp focus, high detail"
            .to_owned(),
        width: 1024,
        height: 1024,
        steps,
        guidance: 4.0,
        id_weight: 1.0,
        sampler: None,
        scheduler: None,
        seed: 12345,
        // Native VAE: this harness validates the face-identity pipeline, not the optional PiD SR (sc-8044).
        use_pid: false,
        cancel: CancelFlag::new(),
    };

    // --- 1) id-conditioned render (id_weight 1.0) ---
    let t = Instant::now();
    let with_id = model
        .generate(&base, &reference, &mut make_progress())
        .expect("pulid generate (id_weight 1.0)");
    eprintln!("[id] {:?}", t.elapsed());
    write_ppm(&out_dir.join("pulid_id.ppm"), &with_id);
    let id_cos = output_cosine(&face, &ref_emb, &with_id, "id");

    // --- 2) no-id ablation (id_weight 0.0 ⇒ plain FLUX, same seed/prompt) ---
    let no_id_req = PulidFluxRequest {
        id_weight: 0.0,
        ..base.clone()
    };
    let t = Instant::now();
    let no_id = model
        .generate(&no_id_req, &reference, &mut make_progress())
        .expect("pulid generate (id_weight 0.0)");
    eprintln!("[no-id] {:?}", t.elapsed());
    write_ppm(&out_dir.join("pulid_no_id.ppm"), &no_id);
    let no_id_cos = output_cosine(&face, &ref_emb, &no_id, "no-id");

    let diff = mean_abs_diff(&with_id, &no_id);

    // --- 3) Cancel — pre-denoise (flag set before the call) ---
    let pre = PulidFluxRequest {
        cancel: {
            let c = CancelFlag::new();
            c.cancel();
            c
        },
        ..base.clone()
    };
    let r = model.generate(&pre, &reference, &mut make_progress());
    assert!(
        matches!(r, Err(CandleError::Canceled)),
        "pre-cancel must return Err(Canceled), got {r:?}"
    );
    eprintln!("[cancel:pre] Err(Canceled) ✓");

    // --- 4) Cancel — mid-denoise (flip the flag from the progress callback on the 3rd step) ---
    let cancel = CancelFlag::new();
    let seen = Arc::new(AtomicUsize::new(0));
    let mut prog = {
        let cancel = cancel.clone();
        let seen = seen.clone();
        move |p: Progress| {
            if let Progress::Step { .. } = p {
                if seen.fetch_add(1, Ordering::Relaxed) >= 2 {
                    cancel.cancel();
                }
            }
        }
    };
    let mid = PulidFluxRequest {
        cancel: cancel.clone(),
        ..base.clone()
    };
    let r = model.generate(&mid, &reference, &mut prog);
    assert!(
        matches!(r, Err(CandleError::Canceled)),
        "mid-denoise cancel must return Err(Canceled), got {r:?}"
    );
    eprintln!(
        "[cancel:mid] Err(Canceled) after {} steps ✓",
        seen.load(Ordering::Relaxed)
    );

    // --- Identity-recovery gate ---
    eprintln!(
        "\n=== PuLID-FLUX validation ===\n  id   cosine: {id_cos:.4}\n  no-id cosine: {no_id_cos:.4}\n  pixel diff (id vs no-id): {diff:.2}\n  outputs: {}",
        out_dir.display()
    );
    // The PuLID id render must recover the reference identity (mlx envelope ≈0.7); >0.45 is a
    // conservative pass bar that still catches a broken inference path (random faces ≈0).
    assert!(
        id_cos > 0.45,
        "id cosine too low ({id_cos}) — PuLID inference likely broken"
    );
    // And it must beat the no-id ablation by a clear margin (the identity is the PuLID path, not the
    // prompt) — and visibly change the pixels.
    assert!(
        id_cos > no_id_cos + 0.1,
        "id cosine ({id_cos}) not clearly above the no-id ablation ({no_id_cos}) — id conditioning weak"
    );
    assert!(
        diff > 5.0,
        "id vs no-id pixel diff too small ({diff}) — injection had no effect"
    );
    eprintln!("PuLID-FLUX validation PASS ✅ (eyeball pulid_id.ppm vs pulid_no_id.ppm)");
}
