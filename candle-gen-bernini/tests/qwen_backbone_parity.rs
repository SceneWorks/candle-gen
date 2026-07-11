//! The native Qwen2.5-VL-7B planner backbone matches the reference forward (near-bit, f32) — sc-10995,
//! candle port of the mlx lane's `qwen_backbone_parity`. A tiny structurally-faithful two-layer decoder
//! with random weights (dumped from the reference by `tools/dump_bernini_qwen_backbone_golden.py`),
//! reused here byte-for-byte. Exercises the 3-D MRoPE channel stitch, QKV-bias projections, GQA repeat,
//! the external additive 4D mask, the residual stack, and the HF `hidden_states[-2]` tap — without the
//! 14 GB checkpoint. CPU, f32 throughout.

mod common;

use common::{errors, flat_f32, Golden};

use candle_gen::candle_core::{DType, Device};
use candle_gen_bernini::qwen2_5_vl::{Qwen25VlText, QwenVlTextConfig};

fn config_from_meta(g: &Golden) -> QwenVlTextConfig {
    let sec: Vec<usize> = g
        .meta_req("mrope_section")
        .split(',')
        .map(|s| s.parse::<usize>().unwrap())
        .collect();
    QwenVlTextConfig {
        hidden_size: g.meta_req("hidden_size").parse().unwrap(),
        num_layers: g.meta_req("num_hidden_layers").parse().unwrap(),
        num_heads: g.meta_req("num_attention_heads").parse().unwrap(),
        num_kv_heads: g.meta_req("num_key_value_heads").parse().unwrap(),
        head_dim: g.meta_req("head_dim").parse().unwrap(),
        intermediate_size: g.meta_req("intermediate_size").parse().unwrap(),
        rms_norm_eps: g.meta_req("rms_norm_eps").parse().unwrap(),
        rope_theta: g.meta_req("rope_theta").parse().unwrap(),
        mrope_section: [sec[0], sec[1], sec[2]],
    }
}

#[test]
fn qwen_backbone_matches_reference_f32() {
    let dev = Device::Cpu;
    let g = Golden::load("qwen_backbone_golden");
    let cfg = config_from_meta(&g);

    // Load the backbone from the fixture's `w.model.*` namespace (all f32).
    let vb = g.var_builder(&dev);
    let backbone = Qwen25VlText::new(cfg.clone(), vb.pp("w.model")).expect("backbone");

    let embeds = g.tensor("io.embeds", &dev);
    // position_ids are stored I32 → build an I64 [3,L] tensor.
    let pid = g.i64("io.position_ids");
    let pshape = g.shape("io.position_ids");
    let position_ids =
        candle_gen::candle_core::Tensor::from_vec(pid, (pshape[0], pshape[1]), &dev).unwrap();
    let mask = g.tensor("io.mask", &dev);

    // 1. MRoPE table golden — the net-new 3D rotary stitch, compared to torch's assembled cos/sin.
    let (cos, sin) = backbone.mrope_cos_sin(&position_ids, DType::F32).unwrap();
    let (cos_abs, cos_rel) = errors(&flat_f32(&cos), &g.f32("out.cos"));
    let (sin_abs, sin_rel) = errors(&flat_f32(&sin), &g.f32("out.sin"));
    println!("mrope cos: peak|Δ|={cos_abs:.3e} rel={cos_rel:.3e}  sin: peak|Δ|={sin_abs:.3e} rel={sin_rel:.3e}");
    assert!(
        cos_rel < 1e-3 && sin_rel < 1e-3,
        "MRoPE table must match torch"
    );

    // 2. Penultimate hidden state golden — the full forward through both layers.
    let all = backbone.forward(&embeds, &position_ids, &mask).unwrap();
    assert_eq!(all.len(), cfg.num_layers + 1, "hidden-state count = N+1");
    let penult = &all[all.len() - 2];
    let (abs, rel) = errors(&flat_f32(penult), &g.f32("out.penultimate"));
    println!("penultimate: peak|Δ|={abs:.3e}  peak-rel={rel:.3e}");
    assert!(
        rel < 5e-3,
        "penultimate within the f32 matmul floor (rel {rel:.3e})"
    );

    // 3. The convenience accessor returns the same [-2] tensor.
    let p2 = backbone.penultimate(&embeds, &position_ids, &mask).unwrap();
    let (_, rel2) = errors(&flat_f32(&p2), &flat_f32(penult));
    assert!(rel2 < 1e-6, "penultimate() == forward()[-2]");
}
