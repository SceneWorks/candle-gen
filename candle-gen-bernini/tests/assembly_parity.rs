//! sc-10995: the planner-input assembly glue matches the reference (bit-exact) — candle port of the
//! mlx lane's `assembly_parity`. Synthetic-fixture golden (`tools/dump_bernini_assembly_golden.py`):
//!   - `format_mllm_inputs_embeds` — token embedding + `masked_scatter` of the ViT visual features into
//!     the visual slots (input-ViT ∪ gen-ViT).
//!   - `concat_with_zero_init` — prepend the UMT5 prompt embeds, then zero-pad / truncate to
//!     `max_sequence_length` (both branches).
//!
//! These are exact host/tensor ops → bit-for-bit equality. CPU, no cuda/weights.

mod common;

use common::{errors, flat_f32, Golden};

use candle_gen::candle_core::Device;
use candle_gen_bernini::assembly::{concat_with_zero_init, format_mllm_inputs_embeds};
use candle_gen_bernini::qwen2_5_vl::{Qwen25VlText, QwenVlTextConfig};

fn assert_exact(name: &str, got: &[f32], want: &[f32]) {
    let (abs, _) = errors(got, want);
    println!("{name:>14}: max|Δ|={abs:.3e}");
    assert!(abs < 1e-6, "{name} max|Δ| {abs:.3e} not bit-exact");
}

#[test]
fn assembly_matches_reference() {
    let dev = Device::Cpu;
    let g = Golden::load("assembly_golden");
    let hidden: usize = g.meta_req("hidden").parse().unwrap();
    let max_seq: usize = g.meta_req("max_seq").parse().unwrap();

    // Minimal backbone (0 layers) — only the token embedding is exercised.
    let cfg = QwenVlTextConfig {
        hidden_size: hidden,
        num_layers: 0,
        num_heads: 2,
        num_kv_heads: 1,
        head_dim: 8,
        intermediate_size: 32,
        rms_norm_eps: 1e-6,
        rope_theta: 1_000_000.0,
        mrope_section: [1, 2, 1],
    };
    let vb = g.var_builder(&dev);
    let backbone = Qwen25VlText::new(cfg, vb.pp("model")).expect("backbone");

    // --- format_mllm_inputs_embeds ---
    let input_ids: Vec<i64> = g.i64("io.input_ids");
    let visual_embeds = g.tensor("io.visual_embeds", &dev);
    let vin = g.bools_from_i32("io.visual_input_mask");
    let vout = g.bools_from_i32("io.visual_output_mask");
    let got = format_mllm_inputs_embeds(&backbone, &input_ids, Some(&visual_embeds), &vin, &vout)
        .expect("format_mllm");
    assert_exact("format_mllm", &flat_f32(&got), &g.f32("out.format_mllm"));
    // sc-11148 / F-079: the embeds must land on the backbone's device (the ids are built there so
    // `embed_tokens.index_select` doesn't cross devices). On a CUDA backbone this catches a regression
    // to a hardcoded CPU id tensor.
    assert!(
        got.device().same_device(backbone.device()),
        "format_mllm embeds must be on the backbone device"
    );

    // no-visual path == plain token embedding (scatter of nothing is a no-op).
    let none =
        format_mllm_inputs_embeds(&backbone, &input_ids, None, &vin, &vout).expect("no visual");
    assert_eq!(none.dims(), &[1, input_ids.len(), hidden]);
    assert!(
        none.device().same_device(backbone.device()),
        "no-visual embeds must be on the backbone device"
    );

    // --- concat_with_zero_init (pad + truncate) ---
    let t5 = g.tensor("io.t5", &dev);
    let pad = concat_with_zero_init(&t5, &g.tensor("io.stream_short", &dev), max_seq).expect("pad");
    assert_exact("concat_pad", &flat_f32(&pad), &g.f32("out.concat_pad"));
    let trunc =
        concat_with_zero_init(&t5, &g.tensor("io.stream_long", &dev), max_seq).expect("trunc");
    assert_exact(
        "concat_trunc",
        &flat_f32(&trunc),
        &g.f32("out.concat_trunc"),
    );
}

/// sc-11148 / F-079: on CUDA the planner-input tensors (token ids embedded by the backbone, plus the
/// MRoPE position ids and the additive attention mask built host-side on CPU) must all live on the
/// backbone's device. Before the fix `format_mllm_inputs_embeds` built its ids on `Device::Cpu`, so
/// `embed_tokens.index_select` hard-errored with `DeviceMismatchBinaryOp` — every `bernini` generate
/// failed at the first planner forward. This runs a tiny synthetic backbone directly on CUDA (no real
/// weights) so the CUDA gate catches any regression to a hardcoded-CPU planner tensor.
#[cfg(feature = "cuda")]
#[test]
fn planner_tensors_land_on_cuda_backbone() {
    use std::collections::HashMap;

    use candle_gen::candle_core::{DType, Tensor};
    use candle_gen::candle_nn::VarBuilder;
    use candle_gen_bernini::{build_attention_mask_4d, mrope_position_ids, MRopeConfig};

    let dev = Device::new_cuda(0).expect("cuda device 0");
    let (vocab, hidden) = (32usize, 16usize);

    // Zero-layer backbone: only `embed_tokens` / `norm` are read, so the token embedding is exercised.
    let cfg = QwenVlTextConfig {
        hidden_size: hidden,
        num_layers: 0,
        num_heads: 2,
        num_kv_heads: 1,
        head_dim: 8,
        intermediate_size: 32,
        rms_norm_eps: 1e-6,
        rope_theta: 1_000_000.0,
        mrope_section: [1, 2, 1],
    };
    let mut w: HashMap<String, Tensor> = HashMap::new();
    w.insert(
        "model.embed_tokens.weight".into(),
        Tensor::randn(0f32, 0.2f32, (vocab, hidden), &dev).unwrap(),
    );
    w.insert(
        "model.norm.weight".into(),
        Tensor::ones(hidden, DType::F32, &dev).unwrap(),
    );
    let vb = VarBuilder::from_tensors(w, DType::F32, &dev);
    let backbone = Qwen25VlText::new(cfg, vb.pp("model")).expect("backbone");
    assert!(
        backbone.device().same_device(&dev),
        "backbone weights on cuda"
    );

    // The token-embedding path: pre-fix this hard-errored with DeviceMismatchBinaryOp.
    let input_ids: Vec<i64> = vec![1, 5, 9, 3, 7];
    let l = input_ids.len();
    let vin = vec![false; l];
    let vout = vec![false; l];
    let embeds =
        format_mllm_inputs_embeds(&backbone, &input_ids, None, &vin, &vout).expect("embed on cuda");
    assert!(
        embeds.device().same_device(&dev),
        "embeds must be on the cuda backbone device"
    );

    // MRoPE ids + attention mask are built on CPU then moved (as `build_stream` does) — the moved
    // tensors must be usable alongside the cuda embeds.
    let pos = mrope_position_ids(&input_ids, &[], &[], &MRopeConfig::default())
        .unwrap()
        .to_device(&dev)
        .unwrap();
    let mask = build_attention_mask_4d(&[0; 5], &[0; 5])
        .unwrap()
        .to_device(&dev)
        .unwrap();
    assert!(pos.device().same_device(&dev), "position ids moved to cuda");
    assert!(mask.device().same_device(&dev), "mask moved to cuda");
}
