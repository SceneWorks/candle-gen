//! gen-core contract conformance for the candle Kolors provider (sc-4481, epic 3692 / sc-5485).
//!
//! Runs the backend-neutral [`gen_core_testkit`] suite — validate-honesty, progress monotonicity,
//! typed cancellation, seed-determinism, registry round-trip — against the real candle generator. The
//! **seed-determinism** check is the regression guard for the deterministic CPU-seeded noise (sc-3673
//! parity) the pipeline relies on.
//!
//! It drives a real `generate`, so it needs the CUDA backend + a local Kolors snapshot and is
//! `#[ignore]`d by default. On the Windows/Blackwell box (v143 vcvars + CUDA on PATH):
//!
//! ```text
//! set KOLORS_SNAPSHOT=C:\Users\…\models--Kwai-Kolors--Kolors-diffusers\snapshots\<hash>
//! cargo test -p candle-gen-kolors --features cuda --release --test conformance -- --ignored
//! ```
//!
//! The snapshot must carry the materialized `tokenizer/tokenizer.json` (ChatGLM3 ships only a slow SP
//! tokenizer — run Kolors' `tools/build_kolors_tokenizer.py` once). As with the SDXL/FLUX/Z-Image/
//! Chroma slices: same-backend determinism only; cross-backend pixel equality vs `mlx-gen-kolors` is
//! NOT a goal (RNG algorithms differ).
#![cfg(feature = "cuda")]

use std::path::PathBuf;

use candle_gen::gen_core::{LoadSpec, WeightsSource};
use gen_core_testkit::{conformance, Profile};

#[test]
#[ignore = "needs KOLORS_SNAPSHOT (a Kwai-Kolors/Kolors-diffusers snapshot dir with a materialized tokenizer.json) + a CUDA GPU; run with --features cuda --ignored"]
fn kolors_conformance() {
    let snap = std::env::var("KOLORS_SNAPSHOT")
        .expect("set KOLORS_SNAPSHOT to a Kwai-Kolors/Kolors-diffusers snapshot dir");
    let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from(snap)));

    // 512² (== the descriptor's min_size) and a tiny step count keep the suite's ~4 generate() calls
    // cheap — it verifies contract behavior, not image quality. guidance defaults to 5.0, exercising
    // the dual-branch CFG path. Both dims are multiples of the /8 alignment.
    let profile = Profile {
        width: 512,
        height: 512,
        steps: 4,
        ..Profile::cheap()
    };
    conformance(|| candle_gen_kolors::load(&spec).unwrap(), &profile);
}
