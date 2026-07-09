//! gen-core contract conformance for the candle Wan provider (sc-4481, epic 3692 / sc-3697).
//!
//! Runs the backend-neutral [`gen_core_testkit`] suite — validate-honesty, progress monotonicity,
//! typed cancellation, seed-determinism, registry round-trip — against the real candle generator.
//! Drives a real `generate`, so it needs the CUDA backend + a local Wan2.2-TI2V-5B snapshot and is
//! `#[ignore]`d by default:
//!
//! ```text
//! set WAN_SNAPSHOT=C:\Users\…\models--Wan-AI--Wan2.2-TI2V-5B-Diffusers\snapshots\<hash>
//! cargo test -p candle-gen-wan --features cuda --release --test conformance -- --ignored
//! ```
#![cfg(feature = "cuda")]

use std::path::PathBuf;

use candle_gen::gen_core::{
    GenerationOutput, GenerationRequest, Generator, LoadSpec, ModelDescriptor, Progress,
    WeightsSource,
};
use gen_core_testkit::{conformance, Profile};

/// A test-only wrapper that pins `frames` to a tiny count so the suite's ~4 `generate()` calls stay
/// affordable on the 5B DiT (the `Profile` can't set `frames`). 5 frames → 2 latent frames.
struct TinyClip(Box<dyn Generator>);

impl Generator for TinyClip {
    fn descriptor(&self) -> &ModelDescriptor {
        self.0.descriptor()
    }
    fn validate(&self, req: &GenerationRequest) -> candle_gen::gen_core::Result<()> {
        self.0.validate(req)
    }
    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> candle_gen::gen_core::Result<GenerationOutput> {
        let mut r = req.clone();
        r.frames = Some(5);
        self.0.generate(&r, on_progress)
    }
}

#[test]
#[ignore = "needs WAN_SNAPSHOT (a Wan2.2-TI2V-5B snapshot dir) + a CUDA GPU"]
fn wan_conformance() {
    let snap = std::env::var("WAN_SNAPSHOT")
        .expect("set WAN_SNAPSHOT to a Wan2.2-TI2V-5B-Diffusers snapshot dir");
    let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from(snap)));

    // 512² / 4 steps / 5 frames keeps the suite affordable. Verifies contract behavior, not quality.
    // 512² (not 256²) is the smallest size the descriptor now accepts: the 5B's z48 vae22 renders
    // garbage below a ~15×15 latent grid (min_size 480, sc-10306), and the validate-honesty check
    // exercises `generate` at the profile size.
    let profile = Profile {
        width: 512,
        height: 512,
        steps: 4,
        ..Profile::cheap()
    };
    conformance(
        || Box::new(TinyClip(candle_gen_wan::load(&spec).unwrap())),
        &profile,
    );
}
