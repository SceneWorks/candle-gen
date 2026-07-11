//! sc-7580 / sc-7582 — committed-fixture **cross-backend** parity for the candle Krea 2 Qwen3-VL-4B
//! text encoder against the **transformers** `Qwen3VLTextModel` forward (an independent graph), at tiny
//! dims. The fixture is the SAME one `mlx-gen-krea` validates against, so candle and mlx agree on the
//! reference contract.
//!
//! Exercises bias-less GQA (decoupled head_dim: q_proj 128-wide while hidden is 64), per-head q/k
//! RMSNorm, HF half-split RoPE, the causal mask, and the select-layer hidden-state stack +
//! template-prefix slice — the `context` the DiT consumes.

use std::path::Path;

use candle_gen::candle_core::{DType, Device, Result, Tensor};
use candle_gen_krea::loader::Weights;
use candle_gen_krea::{KreaTeConfig, KreaTextEncoder};

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/te_golden.safetensors"
);

fn vec_f32(x: &Tensor) -> Vec<f32> {
    x.to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

fn cosine(a: &Tensor, b: &Tensor) -> f32 {
    let a = vec_f32(a);
    let b = vec_f32(b);
    let dot: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
    vec_f32(a)
        .iter()
        .zip(&vec_f32(b))
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

/// Tiny config matching `tools/dump_krea_te_golden.py`.
fn tiny_te_config() -> KreaTeConfig {
    KreaTeConfig {
        num_layers: 6,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 32,
        rms_norm_eps: 1e-6,
        rope_theta: 5_000_000.0,
        select_hidden: vec![2, 4],
        prefix_tokens: 3,
        image_token_id: 151655,
        mrope_section: [24, 20, 20],
    }
}

#[test]
fn te_matches_reference() -> Result<()> {
    let w = Weights::from_file(Path::new(FIXTURE), &Device::Cpu, DType::F32)
        .unwrap_or_else(|e| panic!("load te fixture: {e}"));
    let cfg = tiny_te_config();
    let te = KreaTextEncoder::load(&w, "language_model", &cfg, 64)?;

    // The fixture's `in.attention_mask` is all-ones (no padding), so the candle causal-only forward
    // matches; `input_ids` keep their on-disk int dtype.
    let input_ids = w.get_raw("in.input_ids")?.to_dtype(DType::U32)?;
    let hiddens = te.forward(&input_ids)?;
    let want = w.get("out.hiddens")?;
    assert_eq!(hiddens.dims(), want.dims(), "stacked-context shape");

    let c = cosine(&hiddens, &want);
    println!(
        "Krea TE parity: cosine={c:.7} max_abs={:e}",
        max_abs_diff(&hiddens, &want)
    );
    assert!(c > 0.999, "TE cosine {c:.7} <= 0.999");
    assert!(
        max_abs_diff(&hiddens, &want) < 2e-2,
        "TE stacked context diverged beyond 2e-2 (cosine {c:.7})"
    );
    Ok(())
}
