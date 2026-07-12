//! sc-11671: the Bernini MAR reveal-order / per-step-noise RNGs match torch + numpy bit-for-bit.
//!
//! Golden (`tools/dump_bernini_mar_mt19937_golden.py`, torch 2.13 CPU + numpy 2.4 — an independent
//! oracle, no Bernini model math): for a fixed seed + shapes it dumps the reveal permutation from numpy
//! legacy `np.random.shuffle` and the per-step flow-match base noise from `torch.randn` (drawn
//! sequentially across the planning steps). Asserts:
//!   - [`numpy_shuffle`] is **bit-exact** (integer equality) to numpy's shuffle, and
//!   - [`torch_step_noise`] matches `torch.randn`'s `normal_fill` within a tight f32 tolerance
//!     (the uniforms are bit-exact; only the Box–Muller transcendentals can differ by ~1 ULP).
//!
//! CPU, no cuda / weights.

mod common;

use common::Golden;

use candle_gen_bernini::mar::mar_schedule;
use candle_gen_bernini::rng::{numpy_shuffle, torch_step_noise};

#[test]
fn mar_reveal_order_matches_numpy_shuffle_bit_exact() {
    let g = Golden::load("mar_mt19937_golden");
    let seed: u32 = g.meta_req("seed").parse().unwrap();
    let n_query: usize = g.meta_req("n_query").parse().unwrap();

    let rust_order = numpy_shuffle(n_query, seed);
    let golden_order = g.i32("order");

    assert_eq!(
        rust_order, golden_order,
        "numpy Fisher–Yates reveal permutation must be bit-exact to np.random.shuffle"
    );
    // Sanity: it really is a permutation of [0, n_query).
    let mut sorted = rust_order.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, (0..n_query as i32).collect::<Vec<_>>());
}

#[test]
fn mar_step_noise_matches_torch_randn() {
    let g = Golden::load("mar_mt19937_golden");
    let seed: u32 = g.meta_req("seed").parse().unwrap();
    let n_query: i32 = g.meta_req("n_query").parse().unwrap();
    let planning_step: usize = g.meta_req("planning_step").parse().unwrap();
    let in_channels: usize = g.meta_req("in_channels").parse().unwrap();
    let noise_steps: Vec<usize> = g
        .meta_req("noise_steps")
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().unwrap())
        .collect();

    let order = numpy_shuffle(n_query as usize, seed);
    let schedule = mar_schedule(n_query, planning_step, &order);
    let per_step = torch_step_noise(&schedule, in_channels, seed);
    assert_eq!(per_step.len(), planning_step, "one entry per planning step");

    // Every step the reference drew noise on (and only those) must carry a non-empty draw.
    let rust_drawn: Vec<usize> = per_step
        .iter()
        .enumerate()
        .filter(|(_, d)| !d.is_empty())
        .map(|(s, _)| s)
        .collect();
    assert_eq!(
        rust_drawn, noise_steps,
        "the non-skip steps must match torch's drawn steps exactly"
    );

    let mut peak = 0f32;
    for &step in &noise_steps {
        let got = &per_step[step];
        let golden = g.f32(&format!("noise.{step}"));
        let shape = g.shape(&format!("noise.{step}"));
        assert_eq!(
            got.len(),
            shape.iter().product::<usize>(),
            "step {step} noise length"
        );
        assert_eq!(
            got.len() % in_channels,
            0,
            "step {step} noise is a whole number of rows"
        );
        let d = got
            .iter()
            .zip(&golden)
            .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()));
        peak = peak.max(d);
    }
    println!(
        "torch.randn parity: peak |Δ| across {} steps = {peak:.3e}",
        noise_steps.len()
    );
    // The 24-bit float uniforms are bit-exact; Box–Muller's sqrtf/lnf/cosf/sinf can differ by ~1 ULP.
    assert!(
        peak < 1e-5,
        "per-step FM noise peak |Δ| {peak:.3e} exceeds 1e-5 vs torch.randn"
    );
}
