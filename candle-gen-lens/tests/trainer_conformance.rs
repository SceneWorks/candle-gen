//! Real-weight gen-core **Trainer contract** conformance for the candle `lens` trainer (sc-5147,
//! epic 3720 / sc-4895) — the candle twin of `mlx-gen-lens/tests/trainer_conformance.rs`.
//!
//! Drives the actual [`LensTrainer`](candle_gen_lens::training) through the backend-neutral checks
//! (capability honesty, `TrainingProgress` monotonicity, typed cancellation before any step, registry
//! round-trip). `#[ignore]` + `cfg(feature = "cuda")` because it needs the real `microsoft/Lens` weights
//! (`LENS_BASE_SNAPSHOT` or the HF cache) — including the ~40 GB gpt-oss encoder — and a CUDA GPU.
//! On the Windows/Blackwell box (v143 vcvars + CUDA on PATH):
//!
//! ```text
//! set LENS_BASE_SNAPSHOT=C:\Users\…\models--microsoft--Lens\snapshots\<hash>
//! cargo test -p candle-gen-lens --features cuda --release --test trainer_conformance -- --ignored --nocapture
//! ```
//!
//! The profile is forced to **bf16** (the DiT's native dtype) and gradient-checkpointing on (the
//! trainer always checkpoints the 48-block backward).
#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};

// Force-link the provider so its `inventory::submit!` trainer registration survives into the test
// binary (the registry round-trip check resolves `lens` through it).
use candle_gen_lens as _;

use candle_gen::gen_core::{self, LoadSpec, TrainingItem, WeightsSource};
use gen_core_testkit::TrainerProfile;

/// The `microsoft/Lens` base snapshot dir — `LENS_BASE_SNAPSHOT` or the first HF-cache snapshot.
fn snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("LENS_BASE_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .expect("USERPROFILE/HOME");
    let snaps =
        PathBuf::from(home).join(".cache/huggingface/hub/models--microsoft--Lens/snapshots");
    std::fs::read_dir(&snaps)
        .expect("HF cache snapshots dir (set LENS_BASE_SNAPSHOT to override)")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir")
}

/// Two solid-colour swatch PNGs + captions in `dir` (mirrors the trainer e2e dataset). Captions are
/// long enough to clear the gpt-oss harmony preamble so they carry text features.
fn make_dataset(dir: &Path) -> Vec<TrainingItem> {
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).unwrap();
    let mut items = Vec::new();
    for (i, color) in [[200u8, 40, 40], [40, 80, 200]].iter().enumerate() {
        let mut img = image::RgbImage::new(96, 96);
        for px in img.pixels_mut() {
            *px = image::Rgb(*color);
        }
        let path = dir.join(format!("img{i}.png"));
        img.save(&path).unwrap();
        items.push(TrainingItem {
            image_path: path,
            caption: format!("a solid colour swatch number {i}"),
        });
    }
    items
}

#[test]
#[ignore = "needs real microsoft/Lens weights (LENS_BASE_SNAPSHOT or HF cache) + a CUDA GPU; run with --features cuda --ignored"]
fn lens_trainer_satisfies_gen_core_contract() {
    assert_eq!(candle_gen_lens::MODEL_ID_BASE, "lens");
    let tmp = std::env::temp_dir().join("candle_lens_trainer_conformance");
    let items = make_dataset(&tmp.join("data"));
    let mut profile = TrainerProfile::cheap(items, tmp.join("out"));
    // The Lens DiT trains at its native bf16; the trainer always gradient-checkpoints the 48-block
    // backward, so set the flag for honesty.
    profile.config.train_dtype = "bf16".to_string();
    profile.config.gradient_checkpointing = true;
    let snap = snapshot();

    gen_core_testkit::trainer_conformance(
        || {
            let spec = LoadSpec::new(WeightsSource::Dir(snap.clone()));
            gen_core::load_trainer("lens", &spec).expect("load lens trainer")
        },
        &profile,
    );
}
