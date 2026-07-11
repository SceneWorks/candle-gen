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

    // no-visual path == plain token embedding (scatter of nothing is a no-op).
    let none =
        format_mllm_inputs_embeds(&backbone, &input_ids, None, &vin, &vout).expect("no visual");
    assert_eq!(none.dims(), &[1, input_ids.len(), hidden]);

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
