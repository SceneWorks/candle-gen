//! sc-10995: the connector + clip-diff head match the reference (near-bit, f32) — candle port of the
//! mlx lane's `clip_diff_parity`. Synthetic-fixture golden (`tools/dump_bernini_clip_diff_golden.py`):
//! tiny `MLPConnector` + `SimpleMLPAdaLN` + `FlowMatchScheduler` with random weights. Exercises
//! `for_gen`/`for_vit`, the net forward (TimestepEmbedder, adaLN ResBlocks, FinalLayer), and a full
//! triple-CFG `sample()` denoise — all f32. Tolerances reflect the cross-backend f32 matmul floor.
//! CPU, no cuda/weights.

mod common;

use common::{errors, flat_f32, Golden};

use candle_gen::candle_core::Device;
use candle_gen_bernini::clip_diff::DiffLossFm;
use candle_gen_bernini::connector::MlpConnector;

#[test]
fn clip_diff_matches_reference_f32() {
    let dev = Device::Cpu;
    let g = Golden::load("clip_diff_golden");
    let depth: usize = g.meta_req("depth").parse().unwrap();
    let hidden: usize = g.meta_req("hidden").parse().unwrap();
    let shift: f32 = g.meta_req("shift").parse().unwrap();
    let steps: usize = g.meta_req("steps").parse().unwrap();
    let txt_cfg: f32 = g.meta_req("txt_cfg").parse().unwrap();
    let img_cfg: f32 = g.meta_req("img_cfg").parse().unwrap();

    let vb = g.var_builder(&dev);

    // --- connector ---
    let conn = MlpConnector::new(vb.pp("conn")).expect("connector");
    let cx = g.tensor("io.conn_x", &dev);
    let (_, gen_rel) = errors(
        &flat_f32(&conn.for_gen(&cx).unwrap()),
        &g.f32("out.for_gen"),
    );
    let (_, vit_rel) = errors(
        &flat_f32(&conn.for_vit(&cx).unwrap()),
        &g.f32("out.for_vit"),
    );
    println!("for_gen rel={gen_rel:.3e}  for_vit rel={vit_rel:.3e}");
    assert!(gen_rel < 5e-3, "for_gen rel {gen_rel:.3e}");
    assert!(vit_rel < 5e-3, "for_vit rel {vit_rel:.3e}");

    // --- clip-diff net forward ---
    let mut head = DiffLossFm::new(vb.pp("net"), depth, hidden, shift).expect("head");
    let nx = g.tensor("io.net_x", &dev);
    let nt = g.tensor("io.net_t", &dev);
    let nc = g.tensor("io.net_c", &dev);
    let (_, net_rel) = errors(
        &flat_f32(&head.forward(&nx, &nt, &nc).unwrap()),
        &g.f32("out.net"),
    );
    println!("net rel={net_rel:.3e}");
    assert!(net_rel < 5e-3, "net rel {net_rel:.3e}");

    // --- full triple-CFG sample() ---
    let z = g.tensor("io.z", &dev);
    let noise = g.tensor("io.noise_base", &dev);
    let sample = head
        .sample(&z, txt_cfg, steps, Some(img_cfg), &noise)
        .expect("sample");
    let (_, s_rel) = errors(&flat_f32(&sample), &g.f32("out.sample"));
    println!("sample rel={s_rel:.3e}");
    assert!(s_rel < 5e-3, "sample rel {s_rel:.3e}");
}
