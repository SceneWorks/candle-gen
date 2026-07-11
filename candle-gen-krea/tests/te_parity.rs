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

/// **Image-grounded smoke** (epic 10871 / sc-10880) on the committed tiny TE — no vision-tower weights
/// needed. Drives the real `forward_with_images` plumbing (the `<|image_pad|>` splice, the 3-D
/// interleaved MRoPE, deepstack injection at layers 0/1/2, and the select-layer stack + prefix-drop) on
/// real (tiny) LM weights with synthetic vision features. `image_token_id` is overridden to a small
/// in-vocab id so the tiny embedding table is valid. Asserts the grounded context is correctly shaped +
/// finite, and — the key check — that changing the spliced vision embeds *changes* the output (the
/// splice actually feeds through the decoder), which the weight-free helper tests can't show.
#[test]
fn grounded_forward_splices_vision_through() -> Result<()> {
    let w = Weights::from_file(Path::new(FIXTURE), &Device::Cpu, DType::F32)
        .unwrap_or_else(|e| panic!("load te fixture: {e}"));
    // image_token_id = 0 (always in-vocab); filler/text tokens use 1 so they're never mistaken for the
    // image placeholder. mrope_section is inert-safe at head_dim 32 (all freqs < the section bounds).
    let mut cfg = tiny_te_config();
    cfg.image_token_id = 0;
    let te = KreaTextEncoder::load(&w, "language_model", &cfg, 64)?;
    let hidden = w.get("out.hiddens")?.dim(3)?; // the LM hidden width the vision embeds must match

    // input_ids: 4 text, then a 4-token `<|image_pad|>` block (a 2×2 merged grid → grid [1,4,4]), then
    // 2 text. S = 10 > prefix 3, so the image block survives the prefix-drop.
    let ids: Vec<u32> = vec![1, 1, 1, 1, 0, 0, 0, 0, 1, 1];
    let input_ids = Tensor::from_vec(ids, (1, 10), &Device::Cpu)?;
    let grid = [1i32, 4, 4]; // merged (4/2)·(4/2) = 4 = the block length
    let n = 4usize;

    let mk = |seed: f32| -> Result<(Vec<Tensor>, Vec<Vec<Tensor>>)> {
        let embeds = (Tensor::ones((n, hidden), DType::F32, &Device::Cpu)? * seed as f64)?;
        let ds: Vec<Tensor> = (0..3)
            .map(|k| {
                Tensor::ones((n, hidden), DType::F32, &Device::Cpu)
                    .map(|t| (t * (0.01 * (k + 1) as f64)).unwrap())
            })
            .collect::<std::result::Result<_, _>>()?;
        Ok((vec![embeds], vec![ds]))
    };

    let (e_a, ds_a) = mk(0.5)?;
    let out_a = te.forward_with_images(&input_ids, &e_a, &ds_a, &[grid])?;
    assert_eq!(out_a.dim(0)?, 1, "batch");
    assert_eq!(
        out_a.dim(1)?,
        10 - cfg.prefix_tokens,
        "prefix-dropped length"
    );
    assert_eq!(out_a.dim(2)?, cfg.select_hidden.len(), "select-layer stack");
    assert_eq!(out_a.dim(3)?, hidden, "hidden width");
    assert!(
        vec_f32(&out_a).iter().all(|v| v.is_finite()),
        "grounded context finite"
    );

    // Different vision embeds → different grounded context (the splice feeds through the decoder).
    let (e_b, ds_b) = mk(-0.7)?;
    let out_b = te.forward_with_images(&input_ids, &e_b, &ds_b, &[grid])?;
    let d = max_abs_diff(&out_a, &out_b);
    println!("grounded smoke: vision-embed-change Δ={d:e}");
    assert!(
        d > 1e-4,
        "changing the vision embeds must change the grounded context (Δ={d:e})"
    );
    Ok(())
}
