//! SANA Linear-DiT **trunk** parity gate vs diffusers `SanaTransformer2DModel` (sc-11778), the candle
//! mirror of `mlx-gen-sana`'s `transformer_parity.rs` (mlx-gen #613, sc-8487).
//!
//! Tests:
//!
//!  * [`trunk_matches_diffusers_tiny`] — DEFAULT (not `#[ignore]`d). Loads the SMALL committed golden
//!    (`tests/fixtures/sana_transformer_golden.safetensors`, ~74 KB, dumped from a faithful random-init
//!    diffusers `SanaTransformer2DModel` at reduced dim/depth by `mlx-gen/tools/
//!    dump_sana_transformer_golden.py`): the exact torch weights + inputs + reference noise
//!    prediction. The candle trunk loads those weights and must reproduce the diffusers output. This
//!    is a REAL end-to-end numeric parity check (ReLU linear self-attn + cross-attn + GLUMBConv
//!    Mix-FFN + adaLN-single + NoPE) that runs in CI without the ~1.6B-param real checkpoint. The
//!    SAME golden the MLX port parity-checks against — so the two backends are pinned to one torch
//!    reference. MLX hit `mean_rel ≈ 0.00052` on the real path; this committed f32 tiny golden is even
//!    tighter (a port bug diverges by orders of magnitude; f32 rounding does not).
//!
//!  * [`sprint_trunk_matches_diffusers_tiny`] — DEFAULT. Same, for the SANA-Sprint guidance-embed /
//!    qk-norm superset (`tests/fixtures/sana_sprint_trunk_golden.safetensors`) via
//!    [`SanaTransformer::forward_with_guidance`].
//!
//!  * [`trunk_matches_diffusers_real`] — `#[ignore]`d, gated behind `SANA_TRANSFORMER_WEIGHTS` +
//!    `SANA_TRANSFORMER_GOLDEN` (the real `Sana_1600M_1024px_diffusers` transformer + a `--real`
//!    golden from the dump tool). Characterises full-model parity against the 0.00052 ballpark when
//!    the real weights are stageable.
//!
//! Parity metrics mirror `decode_parity.rs`: `mean_rel = Σ|Δ|/Σ|ref|`, `peak_rel = max|Δ|/max|ref|`.

use std::collections::HashMap;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::Weights;
use candle_gen_sana::{SanaTransformer, SanaTransformerConfig};

fn mean_rel(got: &Tensor, want: &Tensor) -> f32 {
    let num = (got - want)
        .unwrap()
        .abs()
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    let den = want
        .abs()
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    num / den.max(1e-12)
}

fn peak_rel(got: &Tensor, want: &Tensor) -> f32 {
    let peak = (got - want)
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    let denom = want
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    peak / denom.max(1e-12)
}

/// Split the dump tool's `w.`-prefixed weights into a [`Weights`] with the bare diffusers key names,
/// and return the input/output tensors alongside.
fn split_golden(golden: &Weights) -> (Weights, Tensor, Tensor, Tensor, Tensor) {
    let mut map = HashMap::new();
    for key in golden.keys() {
        if let Some(rest) = key.strip_prefix("w.") {
            map.insert(rest.to_string(), golden.require(key).unwrap());
        }
    }
    let latent = golden.require("input.latent").expect("input.latent");
    let caption = golden.require("input.caption").expect("input.caption");
    let timestep = golden.require("input.timestep").expect("input.timestep");
    let want = golden.require("output.sample").expect("output.sample");
    (Weights::from_map(map), latent, caption, timestep, want)
}

/// Tiny config matching `dump_sana_transformer_golden.py`'s tiny instance.
fn tiny_config() -> SanaTransformerConfig {
    SanaTransformerConfig {
        in_channels: 4,
        out_channels: 4,
        num_attention_heads: 2,
        attention_head_dim: 8, // inner = 16
        num_layers: 2,
        num_cross_attention_heads: 2,
        cross_attention_head_dim: 8,
        caption_channels: 24,
        mlp_ratio: 2.5,
        patch_size: 1,
        norm_eps: 1e-6,
        caption_norm_eps: 1e-5,
        attn_qk_norm_eps: 1e-5,
        attn_eps: 1e-15,
        guidance_embeds: false,
        guidance_embeds_scale: 0.1,
        qk_norm: false,
    }
}

#[test]
fn trunk_matches_diffusers_tiny() {
    let golden_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/sana_transformer_golden.safetensors"
    );
    let golden = Weights::from_file(golden_path.as_ref(), &Device::Cpu, DType::F32)
        .expect("load tiny golden");
    let (weights, latent, caption, timestep, want) = split_golden(&golden);

    let model = SanaTransformer::from_weights(&weights, tiny_config()).expect("build trunk");
    let got = model
        .forward(&latent, &caption, &timestep)
        .expect("forward");

    assert_eq!(got.dims(), want.dims(), "shape");
    let mean = mean_rel(&got, &want);
    let peak = peak_rel(&got, &want);
    println!("SANA trunk parity (tiny f32): mean_rel={mean:.6}  peak_rel={peak:.6}");

    // f32 path, 2-block tiny model: a port bug (wrong transpose / op order / modulation chunk order /
    // wrong qkv split) diverges by orders of magnitude. The clean port sits at f32 matmul noise.
    assert!(
        mean < 5e-3,
        "mean_rel {mean} too high — that IS a port bug, not rounding"
    );
    assert!(peak < 5e-2, "peak_rel {peak} above the precision ceiling");
}

/// Tiny SANA-**Sprint** config (guidance embedder + qk-norm ON), matching `dump_sana_sprint_golden.py`.
fn tiny_sprint_config() -> SanaTransformerConfig {
    SanaTransformerConfig {
        guidance_embeds: true,
        guidance_embeds_scale: 0.1,
        qk_norm: true,
        ..tiny_config()
    }
}

#[test]
fn sprint_trunk_matches_diffusers_tiny() {
    let golden_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/sana_sprint_trunk_golden.safetensors"
    );
    let golden = Weights::from_file(golden_path.as_ref(), &Device::Cpu, DType::F32)
        .expect("load tiny Sprint golden");
    let (weights, latent, caption, timestep, want) = split_golden(&golden);
    let guidance = golden.require("input.guidance").expect("input.guidance");

    let model = SanaTransformer::from_weights(&weights, tiny_sprint_config())
        .expect("build Sprint trunk (guidance embedder + qk-norm keys)");
    let got = model
        .forward_with_guidance(&latent, &caption, &timestep, Some(&guidance))
        .expect("forward_with_guidance");

    assert_eq!(got.dims(), want.dims(), "shape");
    let mean = mean_rel(&got, &want);
    let peak = peak_rel(&got, &want);
    println!("SANA-Sprint trunk parity (tiny f32): mean_rel={mean:.6}  peak_rel={peak:.6}");
    assert!(
        mean < 5e-3,
        "mean_rel {mean} too high — a port bug in the guidance-embed / qk-norm path"
    );
    assert!(peak < 5e-2, "peak_rel {peak} above the precision ceiling");
}

/// Full-model parity vs the real `Sana_1600M_1024px_diffusers` transformer + a `--real` golden
/// (mean_rel target ≈ 0.00052, the number the MLX port hit). Gated: needs the ~1.6B checkpoint.
///
/// Run:
///   SANA_TRANSFORMER_WEIGHTS=/path/to/Sana_1600M_1024px_diffusers \
///   SANA_TRANSFORMER_GOLDEN=/path/to/sana_transformer_real.safetensors \
///   cargo test -p candle-gen-sana --test transformer_parity -- --ignored --nocapture
#[test]
#[ignore = "needs Sana_1600M_1024px_diffusers transformer + a --real golden (SANA_TRANSFORMER_WEIGHTS / SANA_TRANSFORMER_GOLDEN)"]
fn trunk_matches_diffusers_real() {
    let weights_dir =
        std::env::var("SANA_TRANSFORMER_WEIGHTS").expect("set SANA_TRANSFORMER_WEIGHTS");
    let golden_path =
        std::env::var("SANA_TRANSFORMER_GOLDEN").expect("set SANA_TRANSFORMER_GOLDEN");

    let golden = Weights::from_file(golden_path.as_ref(), &Device::Cpu, DType::F32)
        .expect("load real golden");
    let latent = golden.require("input.latent").expect("latent");
    let caption = golden.require("input.caption").expect("caption");
    let timestep = golden.require("input.timestep").expect("timestep");
    let want = golden.require("output.sample").expect("output");

    // Real transformer checkpoint: one or more diffusion_pytorch_model*.safetensors under transformer/.
    let dir = std::path::Path::new(&weights_dir).join("transformer");
    let mut shards: Vec<std::path::PathBuf> = std::fs::read_dir(&dir)
        .expect("read transformer dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "safetensors").unwrap_or(false))
        .collect();
    shards.sort();
    let weights =
        Weights::from_files(&shards, &Device::Cpu, DType::F32).expect("load real weights");

    let model = SanaTransformer::from_weights(&weights, SanaTransformerConfig::sana_1600m())
        .expect("build");
    let got = model
        .forward(&latent, &caption, &timestep)
        .expect("forward");

    assert_eq!(got.dims(), want.dims(), "shape");
    let mean = mean_rel(&got, &want);
    let peak = peak_rel(&got, &want);
    println!("SANA trunk parity (real f32): mean_rel={mean:.6}  peak_rel={peak:.6}");
    // Real single step; the per-step drift band the MLX port observed across a 20-step gen is ~3.4%.
    assert!(
        mean < 3.4e-2,
        "mean_rel {mean} above the per-step drift band"
    );
}
