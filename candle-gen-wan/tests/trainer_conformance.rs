//! Real-weight gen-core **Trainer contract** conformance for the candle `wan2_2_t2v_14b` MoE trainer
//! (sc-5167, epic 3720 / sc-4895) — the candle twin of `mlx-gen-wan/tests/trainer_conformance.rs`.
//!
//! Drives the actual [`WanMoeTrainer`](candle_gen_wan::training) through the backend-neutral checks
//! (capability honesty, `TrainingProgress` monotonicity, typed cancellation before any step, registry
//! round-trip). `#[ignore]` + `cfg(feature = "cuda")` because it needs the real
//! `Wan-AI/Wan2.2-T2V-A14B-Diffusers` weights (`WAN_T2V_14B_SNAPSHOT` or the HF cache) and a CUDA GPU.
//! On the Windows/Blackwell box (v143 vcvars + CUDA on PATH):
//!
//! ```text
//! set WAN_T2V_14B_SNAPSHOT=C:\Users\…\models--Wan-AI--Wan2.2-T2V-A14B-Diffusers\snapshots\<hash>
//! cargo test -p candle-gen-wan --features cuda --release --test trainer_conformance -- --ignored --nocapture
//! ```
//!
//! The profile is forced to **bf16** (the cheap default is f32; two 14B experts at f32 would be ~56 GB).
#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};

// Force-link the provider so its `inventory::submit!` trainer registration survives into the test
// binary (the registry round-trip check resolves `wan2_2_t2v_14b` through it).
use candle_gen_wan as _;

use candle_gen::gen_core::{self, LoadSpec, TrainingItem, WeightsSource};
use gen_core_testkit::TrainerProfile;

/// The Wan A14B (T2V) base snapshot dir — `WAN_T2V_14B_SNAPSHOT` or the first HF-cache snapshot.
fn snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("WAN_T2V_14B_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .expect("USERPROFILE/HOME");
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--Wan-AI--Wan2.2-T2V-A14B-Diffusers/snapshots");
    std::fs::read_dir(&snaps)
        .expect("HF cache snapshots dir (set WAN_T2V_14B_SNAPSHOT to override)")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir")
}

/// Two solid-colour swatch PNGs + captions in `dir` (mirrors the trainer e2e dataset).
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
#[ignore = "needs real Wan2.2-T2V-A14B weights (WAN_T2V_14B_SNAPSHOT or HF cache) + a CUDA GPU; run with --features cuda --ignored"]
fn wan_t2v_14b_trainer_satisfies_gen_core_contract() {
    assert_eq!(candle_gen_wan::config::MODEL_ID_T2V_14B, "wan2_2_t2v_14b");
    let tmp = std::env::temp_dir().join("candle_wan_trainer_conformance");
    let items = make_dataset(&tmp.join("data"));
    let mut profile = TrainerProfile::cheap(items, tmp.join("out"));
    // Two 14B experts at f32 (the cheap default) would be ~56 GB; train at the model's native bf16.
    profile.config.train_dtype = "bf16".to_string();
    // The trainer always gradient-checkpoints the 14B backward; set the flag for honesty.
    profile.config.gradient_checkpointing = true;
    let snap = snapshot();

    gen_core_testkit::trainer_conformance(
        || {
            let spec = LoadSpec::new(WeightsSource::Dir(snap.clone()));
            gen_core::load_trainer("wan2_2_t2v_14b", &spec).expect("load wan2_2_t2v_14b trainer")
        },
        &profile,
    );
}
