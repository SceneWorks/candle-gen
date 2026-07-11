//! sc-7580 / sc-7582 — committed-fixture **cross-backend** parity for the candle Krea 2 single-stream
//! DiT against the Krea-published reference (`github.com/krea-ai/krea-2` `mmdit.py` `SingleStreamDiT`),
//! at tiny dims. The fixtures are the SAME ones `mlx-gen-krea` validates against (random seeded weights
//! remapped to the diffusers checkpoint keys + the reference outputs), so candle and mlx agree on the
//! exact reference contract — the cross-platform parity AC. candle CPU runs f32, so the tolerance is
//! tighter than the mlx Metal path; keep the mlx 2e-2 / cosine>0.999 bar.

use std::path::Path;

use candle_gen::candle_core::{Device, Result, Tensor};
use candle_gen_krea::loader::Weights;
use candle_gen_krea::transformer::block::{SingleStreamBlock, TextFusionTransformer};
use candle_gen_krea::transformer::rope::RopeTables;
use candle_gen_krea::{Krea2Config, Krea2Transformer};

const FIX: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/");

// Tiny config shared by the dump script (`mmdit` derives axes [8,12,12] from head_dim 32).
const HEADS: usize = 4;
const KV: usize = 2;
const HEAD_DIM: usize = 32;
const HIDDEN: usize = 128;
const TXT_HEADS: usize = 2;
const EPS: f64 = 1e-5;

fn load(name: &str) -> Weights {
    let path = format!("{FIX}{name}");
    Weights::from_file(
        Path::new(&path),
        &Device::Cpu,
        candle_gen::candle_core::DType::F32,
    )
    .unwrap_or_else(|e| panic!("load fixture {name}: {e}"))
}

fn vec_f32(x: &Tensor) -> Vec<f32> {
    x.to_dtype(candle_gen::candle_core::DType::F32)
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

/// The #1 parity risk localized: the 3-axis interleaved RoPE table for the DiT's joint positions
/// (`cap_len` text `(0,0,0)` + an `ht×wt` grid `(0,row,col)`) must match the reference cos/sin exactly.
#[test]
fn rope_matches_reference() -> Result<()> {
    let g = load("rope_golden.safetensors");
    // meta = [n_tok, ht, wt, ax0, ax1, ax2]; theta fixed at 1000 (see the dump).
    let (cap, ht, wt) = (5usize, 4usize, 4usize);
    let r = RopeTables::build_t2i(cap, ht, wt, [8, 12, 12], 1000.0, &Device::Cpu)?;
    let (cos, sin) = r.joint();

    let want_cos = g.get("cos")?;
    let want_sin = g.get("sin")?;
    assert_eq!(cos.dims(), want_cos.dims(), "cos shape");
    let dc = max_abs_diff(&cos, &want_cos);
    let ds = max_abs_diff(&sin, &want_sin);
    assert!(dc < 1e-5, "rope cos diverged (max abs {dc:e})");
    assert!(ds < 1e-5, "rope sin diverged (max abs {ds:e})");
    Ok(())
}

/// One `SingleStreamBlock`: DoubleSharedModulation (6-factor pre/post), the sigmoid-gated GQA attention
/// with interleaved RoPE, and the SwiGLU FFN.
#[test]
fn single_block_matches_reference() -> Result<()> {
    let w = load("single_block_golden.safetensors");
    let blk = SingleStreamBlock::load(&w, "blk", HEADS, KV, HEAD_DIM, HIDDEN, EPS)?;
    let y = blk.forward(
        &w.get("in.x")?,
        &w.get("in.tvec")?,
        &w.get("in.cos")?,
        &w.get("in.sin")?,
    )?;
    let want = w.get("out.y")?;
    assert_eq!(y.dims(), want.dims());
    let c = cosine(&y, &want);
    println!(
        "single_block parity: cosine={c:.7} max_abs={:e}",
        max_abs_diff(&y, &want)
    );
    assert!(c > 0.999, "single block cosine {c:.7} <= 0.999");
    assert!(
        max_abs_diff(&y, &want) < 2e-2,
        "single block diverged beyond 2e-2 (cosine {c:.7})"
    );
    Ok(())
}

/// The `TextFusionTransformer`: layer-axis aggregation (attention across the stacked layers) →
/// `projector` 12→1 collapse → token-axis refiner blocks.
#[test]
fn text_fusion_matches_reference() -> Result<()> {
    let w = load("text_fusion_golden.safetensors");
    let tf = TextFusionTransformer::load(&w, 2, 2, TXT_HEADS, TXT_HEADS, HEAD_DIM, EPS)?;
    let y = tf.forward(&w.get("in.x")?)?;
    let want = w.get("out.y")?;
    assert_eq!(y.dims(), want.dims());
    let c = cosine(&y, &want);
    println!(
        "text_fusion parity: cosine={c:.7} max_abs={:e}",
        max_abs_diff(&y, &want)
    );
    assert!(c > 0.999, "text_fusion cosine {c:.7} <= 0.999");
    assert!(
        max_abs_diff(&y, &want) < 2e-2,
        "text_fusion diverged beyond 2e-2 (cosine {c:.7})"
    );
    Ok(())
}

/// Tiny config matching `tools/dump_krea_dit_golden.py::dump_dit` (the SwiGLU inner dims are read from
/// the weights, so `intermediate_size` is documentary).
fn tiny_dit_config() -> Krea2Config {
    Krea2Config {
        in_channels: 16,
        patch_size: 2,
        hidden_size: 128,
        num_attention_heads: 4,
        num_kv_heads: 2,
        attention_head_dim: 32,
        num_layers: 2,
        intermediate_size: 384,
        norm_eps: 1e-5,
        axes_dims_rope: [8, 12, 12],
        rope_theta: 1000.0,
        timestep_embed_dim: 64,
        num_text_layers: 3,
        num_layerwise_text_blocks: 2,
        num_refiner_text_blocks: 2,
        text_hidden_dim: 64,
        text_intermediate_size: 256,
        text_num_attention_heads: 2,
        text_num_kv_heads: 2,
    }
}

/// **Edit-forward integrated smoke** (epic 10871 / sc-10877, sc-10878) on the committed tiny DiT — no
/// 12B download needed. Exercises the real `forward_edit` plumbing (ref patch-embed through `img_in`,
/// the `[text, refs…, target]` seq-concat, `build_edit` RoPE frames, the single-stream stack, the
/// target-tail slice, unpatchify) end to end on real (tiny) weights:
///  1. **Zero refs ≡ the base forward** — `forward_edit(…, &[])` must equal `forward(…)` to f32 noise
///     (`build_edit(n_refs=0) ≡ build_t2i`, and `[ctx, target]` == `[ctx, img]`).
///  2. **A reference influences the target** — one ref yields the same-shaped, finite velocity that
///     *differs* from the base (the in-context tokens actually condition the output).
///  3. **Two refs** — the scene/person path runs and stays finite + correctly shaped (sc-10878).
#[test]
fn edit_forward_reduces_to_base_and_refs_influence() -> Result<()> {
    let w = load("dit_golden.safetensors");
    let cfg = tiny_dit_config();
    let dit = Krea2Transformer::load(&w, &cfg)?;
    let latent = w.get("in.latent")?;
    let timestep = w.get("in.timestep")?;
    let context = w.get("in.context")?;

    let base = dit.forward(&latent, &timestep, &context)?;

    // 1. Zero references ≡ the base forward (byte-close: same ops, same RoPE table).
    let edit0 = dit.forward_edit(&latent, &timestep, &context, &[])?;
    assert_eq!(edit0.dims(), base.dims(), "zero-ref edit shape");
    let d0 = max_abs_diff(&edit0, &base);
    assert!(
        d0 < 1e-5,
        "forward_edit(&[]) must equal forward() (max abs {d0:e})"
    );

    // A synthetic reference latent at the target resolution (a transform of the noise latent — any
    // same-shaped normalized latent exercises the ref path).
    let ref0 = ((&latent * 0.5)? + 0.1)?;

    // 2. One reference: same shape, finite, and materially different from the base.
    let edit1 = dit.forward_edit(&latent, &timestep, &context, std::slice::from_ref(&ref0))?;
    assert_eq!(edit1.dims(), base.dims(), "one-ref edit shape");
    assert!(
        vec_f32(&edit1).iter().all(|v| v.is_finite()),
        "one-ref edit finite"
    );
    let d1 = max_abs_diff(&edit1, &base);
    println!("edit-forward smoke: zero-ref Δ={d0:e} · one-ref Δ={d1:e}");
    assert!(
        d1 > 1e-4,
        "a reference must influence the target velocity (Δ={d1:e})"
    );

    // 3. Two references (scene = image 1, person = image 2): runs, correct shape, finite.
    let ref1 = ((&latent * (-0.3))? + 0.2)?;
    let edit2 = dit.forward_edit(&latent, &timestep, &context, &[ref0, ref1])?;
    assert_eq!(edit2.dims(), base.dims(), "two-ref edit shape");
    assert!(
        vec_f32(&edit2).iter().all(|v| v.is_finite()),
        "two-ref edit finite"
    );
    Ok(())
}

/// Full `SingleStreamDiT` forward: img patch-embed, the custom timestep embedding + shared modulation,
/// text fusion + `txt_in`, the joint single-stream stack under 3-axis RoPE, the final layer, and
/// unpatchify — end to end vs the reference velocity.
#[test]
fn dit_matches_reference() -> Result<()> {
    let w = load("dit_golden.safetensors");
    let cfg = tiny_dit_config();
    cfg.validate().unwrap();
    let dit = Krea2Transformer::load(&w, &cfg)?;
    let velocity = dit.forward(
        &w.get("in.latent")?,
        &w.get("in.timestep")?,
        &w.get("in.context")?,
    )?;
    let want = w.get("out.velocity")?;
    assert_eq!(velocity.dims(), want.dims(), "velocity shape");
    let c = cosine(&velocity, &want);
    println!(
        "full-DiT parity: cosine={c:.7} max_abs={:e}",
        max_abs_diff(&velocity, &want)
    );
    assert!(c > 0.999, "full-DiT cosine {c:.7} <= 0.999");
    assert!(
        max_abs_diff(&velocity, &want) < 2e-2,
        "full-DiT velocity diverged beyond 2e-2 (cosine {c:.7})"
    );
    Ok(())
}
