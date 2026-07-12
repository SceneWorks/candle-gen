//! Krea 2 pose-ControlNet **train/infer divergence diagnostic** (sc-8460 step-500 probe).
//!
//! For one real (target, pose, caption) sample and a few σ values the 8-step Turbo sampler actually
//! visits, this compares — at one or more resolutions —
//!  1. the TRAINING branch application (graph-tracked Vars, exactly as `control_loss_grads`'s dense
//!     path applies it) vs the INFERENCE application (frozen branch, as `krea-control-infer`):
//!     relative velocity difference must be ~0 (bf16 tolerance);
//!  2. per-injection-point `‖residual‖ / ‖main image tokens‖` ratios (is the branch swamping the
//!     stream, and does that depend on resolution?);
//!  3. the branched vs un-branched base velocity (how much the branch bends the prediction).
//!
//! ```text
//! cargo run -p candle-gen-krea --example krea-control-diag --features cuda --release -- \
//!   --snapshot <krea snapshot> --ckpt control_step500.safetensors \
//!   --data D:\sceneworks-pose-controlnet\spike --manifest val_manifest.jsonl --id 000015 \
//!   --sizes 512,1024
//! ```

use std::path::PathBuf;

use candle_gen::candle_core::{DType, Tensor};
use candle_gen::train::flow_match::{component_vb, sample_noise};
use candle_gen_krea::control::{forward_with_control, probe_forward, ControlBranch};
use candle_gen_krea::loader::Weights;
// The crate's single canonical prompt-token cap (sc-11205 / F-120) — no longer a per-example const.
use candle_gen_krea::pipeline::MAX_TEXT_TOKENS;
use candle_gen_krea::{
    turbo_sigmas, Krea2Config, KreaTeConfig, KreaTextEncoder, KreaTokenizer, KreaTrainDit,
};
use candle_gen_qwen_image::vae::QwenVaeEncoder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let get = |key: &str, def: &str| -> String {
        argv.iter()
            .position(|a| a == key)
            .and_then(|i| argv.get(i + 1).cloned())
            .unwrap_or_else(|| def.to_string())
    };
    let snapshot = PathBuf::from(get("--snapshot", "D:/models/Krea-2-Turbo"));
    let ckpt = PathBuf::from(get("--ckpt", "control_step500.safetensors"));
    let data = PathBuf::from(get("--data", "D:/sceneworks-pose-controlnet/spike"));
    let manifest = get("--manifest", "val_manifest.jsonl");
    let id = get("--id", "000015");
    let sizes: Vec<u32> = get("--sizes", "512,1024")
        .split(',')
        .map(|s| s.parse().expect("--sizes"))
        .collect();
    let seed: u64 = get("--seed", "42").parse()?;
    let residual_clamp: f64 = get(
        "--residual-clamp",
        &candle_gen_krea::control::DEFAULT_RESIDUAL_CLAMP.to_string(),
    )
    .parse()?;

    let device = candle_gen::default_device()?;

    // Locate the sample row.
    let text = std::fs::read_to_string(data.join(&manifest))?;
    let row = text
        .lines()
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .find(|v| v.get("id").and_then(|s| s.as_str()) == Some(id.as_str()))
        .ok_or_else(|| format!("id {id} not in {manifest}"))?;
    let field = |k: &str| row.get(k).and_then(|s| s.as_str()).unwrap().to_string();
    let (target_p, pose_p, caption) = (field("target"), field("pose"), field("caption"));
    eprintln!("sample {id}: {caption:?}");

    // Condition encode once (f32 TE), then drop.
    let tokenizer = KreaTokenizer::from_snapshot(&snapshot, &device)?;
    let te_cfg = KreaTeConfig::from_snapshot(&snapshot)?;
    let te_w = Weights::from_dir(&snapshot.join("text_encoder"), &device, DType::F32)?;
    let te = KreaTextEncoder::load(&te_w, "language_model", &te_cfg, MAX_TEXT_TOKENS)?;
    let context = te.forward(&tokenizer.encode_prompt(&caption, MAX_TEXT_TOKENS)?)?;
    drop(te);
    drop(te_w);

    // Per-size latents through the shared VAE encoder, then drop it.
    let vae_enc = QwenVaeEncoder::new(component_vb(
        &snapshot,
        "vae",
        &device,
        DType::F32,
        "krea control diag",
    )?)?;
    let mut per_size = Vec::new();
    for &size in &sizes {
        let tgt =
            candle_gen::train::dataset::load_image_tensor(&data.join(&target_p), size, &device)?;
        let pose =
            candle_gen::train::dataset::load_image_tensor(&data.join(&pose_p), size, &device)?;
        per_size.push((size, vae_enc.encode(&tgt)?, vae_enc.encode(&pose)?));
    }
    drop(vae_enc);

    // Frozen base DiT + two instances of the SAME checkpoint: train-mode and frozen (infer-mode).
    let cfg = Krea2Config::from_snapshot(&snapshot)?;
    let dit_w = Weights::from_dir(&snapshot.join("transformer"), &device, DType::BF16)?;
    let dit = KreaTrainDit::load(&dit_w, &cfg)?;
    drop(dit_w);
    let clamp = (residual_clamp > 0.0).then_some(residual_clamp);
    let mut b_train = ControlBranch::from_checkpoint(&ckpt, &cfg, &device)?;
    b_train.set_residual_clamp(clamp);
    let mut b_infer = ControlBranch::from_checkpoint(&ckpt, &cfg, &device)?;
    b_infer.freeze();
    b_infer.set_residual_clamp(clamp);
    eprintln!("residual clamp tau: {clamp:?}");
    eprintln!(
        "branch: {} blocks, {:.2}B params, ckpt {}",
        b_train.num_blocks(),
        b_train.num_params() as f64 / 1e9,
        ckpt.display()
    );

    // σ values the 8-step Turbo schedule actually visits: first, middle, late.
    let sig = turbo_sigmas(8);
    let sigmas = [sig[0], sig[3], sig[6]];

    let norm = |t: &Tensor| -> Result<f64, Box<dyn std::error::Error>> {
        Ok((t
            .to_dtype(DType::F32)?
            .sqr()?
            .sum_all()?
            .to_scalar::<f32>()? as f64)
            .sqrt())
    };

    for (size, x0, ctrl) in &per_size {
        for (si, &sigma) in sigmas.iter().enumerate() {
            let noise = sample_noise(x0.dims(), seed.wrapping_add(si as u64), &device)?;
            let x_t = ((x0 * (1.0 - sigma as f64))? + (noise * sigma as f64)?)?;
            let t = Tensor::from_vec(vec![sigma], (1,), &device)?;

            // (1) training-path forward (graph-tracked) — dropped right after the norm read. The
            // retained one-forward graph OOMs at 1024², so the train/infer equality check runs at
            // ≤512² only (the paths are resolution-independent code).
            let v_train = if *size <= 512 {
                Some(
                    forward_with_control(&dit, &b_train, &x_t, &t, &context, ctrl, 1.0)?
                        .to_dtype(DType::F32)?
                        .detach(),
                )
            } else {
                None
            };
            // (2) inference-path forward + per-injection norms + base velocity.
            let (report, v_infer, v_base) =
                probe_forward(&dit, &b_infer, &x_t, &t, &context, ctrl, 1.0)?;
            let v_infer = v_infer.to_dtype(DType::F32)?;
            let v_base = v_base.to_dtype(DType::F32)?;

            let n_base = norm(&v_base)?;
            let d_paths = match &v_train {
                Some(vt) => format!(
                    "{:.3e}",
                    norm(&(vt - &v_infer)?)? / (norm(&v_infer)? + 1e-9)
                ),
                None => "skipped".to_string(),
            };
            let d_branch = norm(&(&v_infer - &v_base)?)? / (n_base + 1e-9);
            println!(
                "size {size} sigma {sigma:.4}: |v_base| {n_base:.2}  train-vs-infer rel {d_paths}  branched-vs-base rel {d_branch:.3}"
            );
            let pre: Vec<String> = report
                .iter()
                .map(|(p, _, m)| format!("{:.3}", p / (m + 1e-9)))
                .collect();
            let post: Vec<String> = report
                .iter()
                .map(|(_, q, m)| format!("{:.3}", q / (m + 1e-9)))
                .collect();
            println!(
                "  res/main pre-clamp [{}] post-clamp [{}] (inject offset {})",
                pre.join(", "),
                post.join(", "),
                b_infer.inject_offset()
            );
        }
    }
    Ok(())
}
