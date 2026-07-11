//! sc-10995: the MAR semantic-planning loop (`sample_vit_embed`) matches the reference trajectory
//! (near-bit, f32) — candle port of the mlx lane's `mar_parity`. Golden
//! (`tools/dump_bernini_mar_golden.py`) composes a tiny structurally-faithful Qwen2.5-VL text backbone
//! with an `MLPConnector` and a `SimpleMLPAdaLN`, then drives them through the reference
//! `sample_vit_embed` loop and `feat_from_planner_to_renderer` with the **injected** reveal `order` and
//! per-step FM noise (so the trajectory is deterministic). The chosen order exercises a single-token
//! reveal, the reference's `nonzero().sum()==0` skip (a lone token-0 reveal), and two multi-token
//! reveals. Asserts the filled `pred_vit_embed` and the 4 handoff streams. CPU, f32 — no cuda/weights.

mod common;

use common::{errors, flat_f32, Golden};

use candle_gen::candle_core::{Device, Tensor};
use candle_gen::gen_core::CancelFlag;
use candle_gen_bernini::clip_diff::DiffLossFm;
use candle_gen_bernini::connector::MlpConnector;
use candle_gen_bernini::mar::{sample_vit_embed, StreamState, VitCfg};
use candle_gen_bernini::qwen2_5_vl::{Qwen25VlText, QwenVlTextConfig};

fn stream(g: &Golden, name: &str, dev: &Device) -> StreamState {
    let pshape = g.shape(&format!("io.{name}.pos"));
    let pos = Tensor::from_vec(
        g.i64(&format!("io.{name}.pos")),
        (pshape[0], pshape[1]),
        dev,
    )
    .unwrap();
    StreamState {
        input_embeds: g.tensor(&format!("io.{name}.input"), dev),
        position_ids: pos,
        mask: g.tensor(&format!("io.{name}.mask4d"), dev),
        gen_idx: g
            .i32(&format!("io.{name}.gen_idx"))
            .into_iter()
            .map(|x| x as u32)
            .collect(),
    }
}

#[test]
fn mar_loop_matches_reference_f32() {
    let dev = Device::Cpu;
    let g = Golden::load("mar_golden");

    let sec: Vec<usize> = g
        .meta_req("mrope_section")
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    let u = |k: &str| g.meta_req(k).parse::<usize>().unwrap();
    let cfg = QwenVlTextConfig {
        hidden_size: u("hidden"),
        num_layers: u("layers"),
        num_heads: u("heads"),
        num_kv_heads: u("kv_heads"),
        head_dim: u("head_dim"),
        intermediate_size: u("intermediate"),
        rms_norm_eps: g.meta_req("eps").parse().unwrap(),
        rope_theta: g.meta_req("theta").parse().unwrap(),
        mrope_section: [sec[0], sec[1], sec[2]],
    };
    let depth = u("depth");
    let hidden = u("hidden");
    let shift: f32 = g.meta_req("shift").parse().unwrap();

    let vb = g.var_builder(&dev);
    let backbone = Qwen25VlText::new(cfg, vb.pp("w.model")).expect("backbone");
    let connector = MlpConnector::new(vb.pp("conn")).expect("connector");
    let mut clip_diff = DiffLossFm::new(vb.pp("net"), depth, hidden, shift).expect("clip_diff");

    let vit_cfg = VitCfg {
        planning_step: u("planning_step"),
        vit_denoising_step: u("vit_denoising_step"),
        vit_txt_cfg: g.meta_req("vit_txt_cfg").parse().unwrap(),
        vit_img_cfg: g.meta_req("vit_img_cfg").parse().unwrap(),
    };
    let order = g.i32("io.order");
    let mask_token = g.tensor("io.mask_token", &dev);
    let step_noise: Vec<Tensor> = (0..vit_cfg.planning_step)
        .map(|s| g.tensor(&format!("io.noise.{s}"), &dev))
        .collect();

    let cond = stream(&g, "cond", &dev);
    let uncond = stream(&g, "uncond", &dev);
    let imgcond = stream(&g, "imgcond", &dev);
    let cancel = CancelFlag::new();

    let out = sample_vit_embed(
        &backbone,
        &connector,
        &mut clip_diff,
        &cond,
        &uncond,
        &imgcond,
        &vit_cfg,
        &order,
        &step_noise,
        &cancel,
        &mask_token,
    )
    .expect("sample_vit_embed");

    let checks: [(&str, &Tensor); 5] = [
        ("pred_vit_embed", &out.pred_vit_embed),
        ("wtxt_wvit", &out.wtxt_wvit),
        ("wtxt_wovit", &out.wtxt_wovit),
        ("wotxt_wvit", &out.wotxt_wvit),
        ("wotxt_wovit", &out.wotxt_wovit),
    ];
    for (name, got) in checks {
        let (abs, rel) = errors(&flat_f32(got), &g.f32(&format!("out.{name}")));
        println!("{name:>14}: peak|Δ|={abs:.3e}  peak-rel={rel:.3e}");
        assert!(rel < 5e-3, "{name} peak-rel {rel:.3e} exceeds 5e-3");
    }
}
