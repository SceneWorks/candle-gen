//! The renderer's ViT-conditioned guidance-combine modes match the reference (f32) — sc-10995, candle
//! port of the mlx lane's `vit_guidance_parity`. Synthetic-fixture parity over random `[1, n, C]`
//! target-sliced packed-token predictions: `apg_delta` (v-space projection) + the combine arms
//! (`vae_txt_vit` / `_wapg` / `rv2v_wapg` / `r2v_wapg`). Pure elementwise + a single-scalar projection
//! per delta, so this matches to the f32 floor. CPU, weight-free.

mod common;

use common::{errors, flat_f32, Golden};

use candle_gen::candle_core::Device;
use candle_gen_bernini::guidance::apg_delta;
use candle_gen_bernini::vit_guidance::{rv2v_chain, vae_txt_vit};

fn check(name: &str, got: &[f32], want: &[f32], tol: f32) {
    let (max_diff, rel) = errors(got, want);
    println!("{name:>12}: peak|Δ|={max_diff:.3e} peak-rel={rel:.3e}");
    assert!(rel < tol, "{name} peak-rel exceeds {tol:.1e}");
}

#[test]
fn vit_guidance_matches_reference() {
    let dev = Device::Cpu;
    let g = Golden::load("vit_guidance_golden");
    let t = |k: &str| g.tensor(k, &dev);
    let (w_img, w_txt, w_tgt, w_vid) = (4.5f32, 4.0, 3.0, 1.25);

    // bare apg_delta (projection only): apg_delta(img - base, ref = img, 0.2, 1.0).
    let delta = (&t("io.img") - &t("io.base")).unwrap();
    let apg = apg_delta(&delta, &t("io.img"), 0.2, 1.0).expect("apg_delta");
    check("apg_only", &flat_f32(&apg), &g.f32("out.apg_only"), 1e-5);

    // vae_txt_vit (plain) + vae_txt_vit_wapg (apg, ref = "to" pred).
    let vtv_plain = vae_txt_vit(
        &t("io.base"),
        &t("io.img"),
        &t("io.txt"),
        &t("io.vit"),
        w_img,
        w_txt,
        w_tgt,
        false,
    )
    .unwrap();
    check(
        "vtv_plain",
        &flat_f32(&vtv_plain),
        &g.f32("out.vtv_plain"),
        1e-5,
    );
    let vtv_apg = vae_txt_vit(
        &t("io.base"),
        &t("io.img"),
        &t("io.txt"),
        &t("io.vit"),
        w_img,
        w_txt,
        w_tgt,
        true,
    )
    .unwrap();
    check("vtv_apg", &flat_f32(&vtv_apg), &g.f32("out.vtv_apg"), 1e-5);

    // rv2v_wapg (plain) + r2v_wapg (apg, ref = "from" pred).
    let rv2v_plain = rv2v_chain(
        &t("io.base"),
        &t("io.eps_v"),
        &t("io.eps_vi"),
        &t("io.eps_vti"),
        &t("io.eps_vtic"),
        w_vid,
        w_img,
        w_txt,
        w_tgt,
        false,
    )
    .unwrap();
    check(
        "rv2v_plain",
        &flat_f32(&rv2v_plain),
        &g.f32("out.rv2v_plain"),
        1e-5,
    );
    let r2v_apg = rv2v_chain(
        &t("io.base"),
        &t("io.eps_v"),
        &t("io.eps_vi"),
        &t("io.eps_vti"),
        &t("io.eps_vtic"),
        w_vid,
        w_img,
        w_txt,
        w_tgt,
        true,
    )
    .unwrap();
    check("r2v_apg", &flat_f32(&r2v_apg), &g.f32("out.r2v_apg"), 1e-5);
}
