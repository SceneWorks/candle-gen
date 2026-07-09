//! Shared seed-derivation + launch-portable seeded-noise helpers (sc-7792 consolidation / F-059,
//! sc-9043).
//!
//! # Why these live here
//!
//! Every candle provider crate reproduces the *same* determinism conventions, and a single divergent
//! copy silently breaks cross-provider (and cross-backend, vs `mlx-gen`) reproducibility:
//!
//!  * **Per-image batch seed** — image `index` of a `count`-image request renders at
//!    `base_seed + index` (wrapping at the `u64` ceiling), so the *n*-th image of a batch reproduces
//!    in isolation as a single `count: 1` render at that derived seed. This mirrors the SceneWorks
//!    `SdxlDiffusersAdapter` per-image increment and `mlx-gen`'s `seed + i` convention. It used to be
//!    a hand-copied `pub(crate) fn image_seed` in six crates (chroma/flux/kolors/sd3/sdxl/z-image).
//!
//!  * **Ancestral-step RNG salt** — the conditioned SDXL/InstantID/PuLID lanes key the *prior* noise
//!    stream by `seed` and the per-step ancestral noise stream by `seed + STEP_RNG_SALT`, so the two
//!    streams never collide (the first ancestral draw would otherwise re-derive the prior). The salt
//!    is the golden-ratio odd constant `0x9E37_79B9_7F4A_7C15`, previously copied verbatim into three
//!    files (sdxl `edit_provider`/`ip_provider`, instantid `model`).
//!
//!  * **Launch-portable noise draw** — deterministic `N(0, 1)` noise is drawn from a seeded CPU
//!    `StdRng` into a flat `Vec<f32>` *on CPU*, then moved to the compute device (sc-3673). Drawing on
//!    CPU keeps the sample sequence device- and launch-independent (candle's CUDA RNG would not match
//!    the CPU/Metal draw). The flat-draw primitive was reimplemented at ~40 sites; the NCHW form was a
//!    byte-identical `draw_noise` in sdxl `denoise`/`edit_provider`.
//!
//! Keeping one home means the reproducibility law is stated once and every new crate reuses it instead
//! of copying the block again.
//!
//! # gen-core lift (sc-7794)
//!
//! sc-7794 will eventually lift [`image_seed`] + [`STEP_RNG_SALT`] *up* into `gen_core` so both the
//! candle and mlx backends share one home. Until that lands, these live here as the candle-local single
//! source of truth; when 7794 merges, the callers (now a single `candle_gen::seed::*` import, not the
//! scattered copies) reroute to the `gen_core` equivalents and this module drops the lifted items.

use candle_core::{Device, Tensor};
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};

/// The per-image seed within a batch: image `index` of a `count`-image request renders at
/// `base_seed + index` (wrapping at the `u64` ceiling). Mirrors the SceneWorks `SdxlDiffusersAdapter`
/// per-image increment and `mlx-gen`'s `seed + i` convention, so the *n*-th image of a batch
/// reproduces in isolation as a single `count: 1` render at that derived seed. A pure function so the
/// law is unit-testable without a GPU.
#[inline]
pub fn image_seed(base_seed: u64, index: u32) -> u64 {
    base_seed.wrapping_add(index as u64)
}

/// The salt added to the request seed to key the per-step **ancestral** noise stream away from the
/// **prior** noise stream (the prior is keyed by `seed`, the steps by `seed + STEP_RNG_SALT`) —
/// otherwise the first ancestral draw would re-derive the prior latents. The golden-ratio odd constant
/// (`⌊2⁶⁴ / φ⌋`); its high bit density keeps the two `StdRng` streams well separated. Shared by the
/// conditioned SDXL/InstantID/PuLID lanes; the launch-portable determinism the worker relies on.
pub const STEP_RNG_SALT: u64 = 0x9E37_79B9_7F4A_7C15;

/// Draw `n` unit-normal (`N(0, 1)`) `f32` from the seeded `rng` stream on **CPU** — the launch-portable
/// primitive (sc-3673): the sample sequence stays device- and launch-independent because it never
/// touches the GPU RNG. Callers reshape/scale/cast and move to the device. The building block every
/// seeded-noise draw shares.
#[inline]
pub fn seeded_normal_vec(rng: &mut StdRng, n: usize) -> Vec<f32> {
    (0..n).map(|_| StandardNormal.sample(rng)).collect()
}

/// Draw an NCHW `[1, c, h, w]` unit-normal `f32` tensor from the seeded `rng` on CPU (so the draw
/// sequence is device- and launch-independent — sc-3673), then move to `device`. The shared draw used
/// by the SDXL prior and each ancestral step (formerly a byte-identical `draw_noise` in `denoise.rs` /
/// `edit_provider.rs`).
#[inline]
pub fn seeded_noise_nchw(
    rng: &mut StdRng,
    c: usize,
    h: usize,
    w: usize,
    device: &Device,
) -> candle_core::Result<Tensor> {
    let noise = seeded_normal_vec(rng, c * h * w);
    Tensor::from_vec(noise, (1, c, h, w), &Device::Cpu)?.to_device(device)
}

/// Run `body` once per image in a `count`-image batch, deriving each per-image seed from `base_seed`
/// via [`image_seed`], and collect the results (sc-7792).
///
/// This owns the outer frame every batch image render repeats — the `0..count` loop, the
/// [`image_seed`] derivation, and the `Vec` collect — while the model body (noise / schedule /
/// predict / decode, plus its own progress emits) stays hand-written in the closure, which captures
/// `on_progress` and the loaded components from the enclosing scope. Generic over the produced item
/// `I` and the error type `E`, so both the `gen_core::Result` and candle-side [`crate::Result`]
/// providers share it.
///
/// **Why `base_seed` is a parameter, not re-derived from the request.** [`gen_core::default_seed`] is
/// *non-deterministic* (it draws from the wall clock), so it must be resolved exactly **once** per
/// generation — `req.seed.unwrap_or_else(gen_core::default_seed)` — at the call site. Most renders
/// also feed that same `base_seed` to a per-generation PiD decoder (epic 7840) resolved before the
/// loop; re-drawing it here would seed the decoder and the image loop from two different clocks. So
/// the caller owns the one draw and passes it in. The single-output video providers (wan / ltx / svd)
/// don't loop over a count and don't use this.
#[inline]
pub fn for_each_image_seed<I, E>(
    base_seed: u64,
    count: u32,
    mut body: impl FnMut(u64) -> Result<I, E>,
) -> Result<Vec<I>, E> {
    let mut out = Vec::with_capacity(count as usize);
    for index in 0..count {
        out.push(body(image_seed(base_seed, index))?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    /// Per-image seed in a `count`-batch is `base + index` (wrapping), so image *n* reproduces in
    /// isolation at that derived seed — the mlx `seed + i` convention. Pure function, no GPU.
    #[test]
    fn image_seed_is_base_plus_index() {
        assert_eq!(image_seed(42, 0), 42);
        assert_eq!(image_seed(42, 1), 43);
        assert_eq!(image_seed(42, 7), 49);
        // Wrap at the u64 ceiling.
        assert_eq!(image_seed(u64::MAX, 1), 0);
    }

    /// The ancestral-step salt is the exact golden-ratio constant the three copies pinned — a change
    /// here would silently break reproducibility of every conditioned SDXL/InstantID/PuLID render.
    #[test]
    fn step_rng_salt_is_pinned() {
        assert_eq!(STEP_RNG_SALT, 0x9E37_79B9_7F4A_7C15);
    }

    /// The seeded flat draw is a pure function of the seed and length — same seed ⇒ same sequence,
    /// and the salted stream diverges from the base stream (the prior-vs-step separation law).
    #[test]
    fn seeded_normal_vec_is_deterministic() {
        let a = seeded_normal_vec(&mut StdRng::seed_from_u64(7), 16);
        let b = seeded_normal_vec(&mut StdRng::seed_from_u64(7), 16);
        assert_eq!(a, b);
        let salted = seeded_normal_vec(
            &mut StdRng::seed_from_u64(7u64.wrapping_add(STEP_RNG_SALT)),
            16,
        );
        assert_ne!(a, salted);
    }

    /// Byte-identity guard for the ~40-site migration (sc-9452): the shared primitive must draw the
    /// **exact same sequence in the exact same order** as the inline
    /// `(0..n).map(|_| StandardNormal.sample(&mut rng)).collect()` that every provider crate previously
    /// hand-rolled. A regression here (a different fill order, a re-seed, an extra draw) would silently
    /// change every migrated site's seed→noise mapping. We reproduce the old inline expression verbatim
    /// and assert equality.
    #[test]
    fn seeded_normal_vec_matches_inline_draw() {
        let n = 257; // deliberately not a round number / power of two
        for seed in [0u64, 1, 42, 8988, u64::MAX] {
            let via_primitive = seeded_normal_vec(&mut StdRng::seed_from_u64(seed), n);
            // The literal pre-migration inline draw.
            let mut rng = StdRng::seed_from_u64(seed);
            let inline: Vec<f32> = (0..n).map(|_| StandardNormal.sample(&mut rng)).collect();
            assert_eq!(via_primitive, inline, "byte-identity broke for seed {seed}");
        }
    }

    /// Pinned first-N golden values: locks the concrete `StdRng` + `StandardNormal` draw so a dependency
    /// bump that silently changes the RNG algorithm or the normal transform is caught. Any change to
    /// these bytes breaks reproducibility of every migrated noise-draw site and must be deliberate.
    #[test]
    fn seeded_normal_vec_first_values_are_pinned() {
        let v = seeded_normal_vec(&mut StdRng::seed_from_u64(42), 4);
        let expected = [0.069_427_915_f32, 0.132_938_12, 0.262_576_37, -0.225_300_88];
        for (got, want) in v.iter().zip(expected.iter()) {
            assert!(
                (got - want).abs() < 1e-6,
                "pinned draw drifted: got {got}, want {want}"
            );
        }
    }

    /// The NCHW draw has the requested shape, lands on the target (CPU here) device, and is a pure
    /// function of the seed. GPU-free.
    #[test]
    fn seeded_noise_nchw_shape_and_determinism() {
        let mut r1 = StdRng::seed_from_u64(3);
        let mut r2 = StdRng::seed_from_u64(3);
        let a = seeded_noise_nchw(&mut r1, 4, 2, 3, &Device::Cpu).unwrap();
        let b = seeded_noise_nchw(&mut r2, 4, 2, 3, &Device::Cpu).unwrap();
        assert_eq!(a.dims(), &[1, 4, 2, 3]);
        let av = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let bv = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(av, bv);
    }

    /// `for_each_image_seed` runs the body once per `count`, feeding `image_seed(base, index)` each
    /// time, and collects the results in order. GPU-free.
    #[test]
    fn for_each_image_seed_derives_per_image_seed_and_collects() {
        let seeds: Vec<u64> = for_each_image_seed(100, 3, Ok::<u64, ()>).unwrap();
        assert_eq!(seeds, vec![100, 101, 102]);
    }

    /// `count == 0` yields an empty batch, and a body error short-circuits the loop (no further calls).
    #[test]
    fn for_each_image_seed_empty_batch_and_error_short_circuit() {
        let empty: Vec<u64> = for_each_image_seed(1, 0, Ok::<u64, ()>).unwrap();
        assert!(empty.is_empty());

        let mut calls = 0;
        let r: Result<Vec<u64>, &str> = for_each_image_seed(1, 5, |s| {
            calls += 1;
            if calls == 2 {
                Err("boom")
            } else {
                Ok(s)
            }
        });
        assert_eq!(r, Err("boom"));
        assert_eq!(calls, 2, "stops at the first error");
    }
}
