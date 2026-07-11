//! The planner→renderer handoff matches the reference (near-bit, f32) — sc-10995, candle port of the
//! mlx lane's `handoff_parity`. A tiny `MLPConnector` + `mask_tokens` with random f32 weights (dumped
//! from the reference `post_process_input_embeds` / `feat_from_planner_to_renderer` + the 4-stream
//! extraction), reused byte-for-byte. Validates the mask selection + the `for_gen` integration
//! end-to-end. CPU, f32.

mod common;

use common::{errors, flat_f32, Golden};

use candle_gen::candle_core::Device;
use candle_gen_bernini::connector::MlpConnector;
use candle_gen_bernini::mar::{four_streams, post_process_input_embeds};

fn check(name: &str, got: &[f32], want: &[f32], tol: f32) {
    let (max_diff, rel) = errors(got, want);
    println!("{name:>14}: peak|Δ|={max_diff:.3e} peak-rel={rel:.3e}");
    assert!(rel < tol, "{name} peak-rel {rel} exceeds {tol:.1e}");
}

#[test]
fn handoff_matches_reference_f32() {
    let dev = Device::Cpu;
    let g = Golden::load("handoff_golden");
    let vb = g.var_builder(&dev);
    let conn = MlpConnector::new(vb.pp("model.connector")).expect("connector");

    // mask_token = mask_tokens[:, :1] -> [1, 1, H].
    let mask_tokens = g.tensor("model.mask_tokens", &dev);
    let h = mask_tokens.dim(2).unwrap();
    let mask_token = mask_tokens
        .narrow(1, 0, 1)
        .unwrap()
        .reshape((1, 1, h))
        .unwrap();

    let cond_gen = g.bools_from_i32("io.cond_gen_mask");
    let uncond_gen = g.bools_from_i32("io.uncond_gen_mask");

    // --- post_process: gen slots set to mask_token ---
    let cond_input = g.tensor("io.cond_input", &dev);
    let pp = post_process_input_embeds(&cond_input, &cond_gen, &mask_token).unwrap();
    check(
        "post_process",
        &flat_f32(&pp),
        &g.f32("out.post_processed"),
        1e-5,
    );

    // --- 4 streams ---
    let cond_hidden = g.tensor("io.cond_hidden", &dev);
    let uncond_hidden = g.tensor("io.uncond_hidden", &dev);
    let s = four_streams(&cond_hidden, &cond_gen, &uncond_hidden, &uncond_gen, &conn).unwrap();
    check(
        "wtxt_wvit",
        &flat_f32(&s.wtxt_wvit),
        &g.f32("out.wtxt_wvit"),
        5e-3,
    );
    check(
        "wtxt_wovit",
        &flat_f32(&s.wtxt_wovit),
        &g.f32("out.wtxt_wovit"),
        5e-3,
    );
    check(
        "wotxt_wvit",
        &flat_f32(&s.wotxt_wvit),
        &g.f32("out.wotxt_wvit"),
        5e-3,
    );
    check(
        "wotxt_wovit",
        &flat_f32(&s.wotxt_wovit),
        &g.f32("out.wotxt_wovit"),
        5e-3,
    );
}
