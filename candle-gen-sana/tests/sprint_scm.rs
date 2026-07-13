//! SANA-Sprint **SCM/TrigFlow CFG-free few-step** integration gate (sc-11781, epic 11776) — drives
//! the committed tiny SANA-Sprint golden trunk (`tests/fixtures/sana_sprint_trunk_golden.safetensors`,
//! the SAME fixture `transformer_parity.rs` numerically validates) through
//! [`candle_gen_sana::denoise_sprint`] over the shared [`candle_gen::run_scm_sampler`] loop. Proves the
//! Sprint pipeline half end-to-end WITHOUT the ~1.6B real checkpoint: the embedded-guidance trunk
//! forward wires into the SCM loop, the loop takes exactly `num_steps` single forwards (CFG-free — no
//! uncond pass), and the denoised latent is finite + non-degenerate. The full 1024² real-weight run is
//! the `candle-gen-sana-sprint-txt2img` example (the GPU-validation harness).

use std::collections::HashMap;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::{CancelFlag, Progress};
use candle_gen::{ScmScheduler, Weights};
use candle_gen_sana::{denoise_sprint, SanaTransformer, SanaTransformerConfig};

/// Tiny SANA-Sprint config (guidance embedder + qk-norm ON), matching the fixture the parity test uses.
fn tiny_sprint_config() -> SanaTransformerConfig {
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
        guidance_embeds: true,
        guidance_embeds_scale: 0.1,
        qk_norm: true,
    }
}

/// Build the tiny Sprint trunk from the committed golden (the `w.`-prefixed weights only).
fn tiny_sprint_trunk() -> SanaTransformer {
    let golden_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/sana_sprint_trunk_golden.safetensors"
    );
    let golden = Weights::from_file(golden_path.as_ref(), &Device::Cpu, DType::F32)
        .expect("load tiny Sprint golden");
    let mut map = HashMap::new();
    for key in golden.keys() {
        if let Some(rest) = key.strip_prefix("w.") {
            map.insert(rest.to_string(), golden.require(key).unwrap());
        }
    }
    SanaTransformer::from_weights(&Weights::from_map(map), tiny_sprint_config())
        .expect("build Sprint trunk (guidance embedder + qk-norm keys)")
}

/// Deterministic pseudo-random fill (LCG) — reproducible, no rand dep.
fn det(shape: &[usize], seed: u64) -> Tensor {
    let n: usize = shape.iter().product();
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = ((s >> 33) as f64) / ((1u64 << 31) as f64) - 1.0;
        v.push(u as f32);
    }
    Tensor::from_vec(v, shape, &Device::Cpu).unwrap()
}

fn stats(t: &Tensor) -> (f32, f32) {
    let v = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    (
        v.iter().cloned().fold(f32::INFINITY, f32::min),
        v.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
    )
}

/// 2-step Sprint SCM denoise over the real tiny trunk: the loop runs exactly `num_steps` forwards
/// (CFG-free), keeps the trunk in/out channel + spatial shape, and yields a finite, non-degenerate
/// latent.
#[test]
fn sprint_scm_2step_finite_nondegenerate() {
    let dev = Device::Cpu;
    let cfg = tiny_sprint_config();
    let trunk = tiny_sprint_trunk();

    let (lat_h, lat_w) = (6usize, 4usize); // non-square → catches an axis swap.
    let latents = det(&[1, cfg.out_channels as usize, lat_h, lat_w], 0);
    let cond = det(&[1, 7, cfg.caption_channels as usize], 100);

    let scheduler = ScmScheduler::new(2);
    assert_eq!(scheduler.num_steps(), 2);

    let cancel = CancelFlag::default();
    let mut steps_seen = 0usize;
    let mut on_progress = |p: Progress| {
        if matches!(p, Progress::Step { .. }) {
            steps_seen += 1;
        }
    };

    let denoised = denoise_sprint(
        &trunk,
        &scheduler,
        7,
        latents,
        &cond,
        4.5, // guidance_scale (embedded, CFG-free)
        cfg.guidance_embeds_scale,
        &dev,
        &cancel,
        &mut on_progress,
    )
    .expect("Sprint SCM denoise");

    assert_eq!(
        denoised.dims(),
        &[1, cfg.out_channels as usize, lat_h, lat_w],
        "SCM denoise keeps the trunk in/out channel + spatial shape"
    );
    assert_eq!(
        steps_seen,
        scheduler.num_steps(),
        "SCM loop reports exactly num_steps progress events"
    );
    let (lo, hi) = stats(&denoised);
    assert!(lo.is_finite() && hi.is_finite(), "non-finite: [{lo}, {hi}]");
    assert!(hi - lo > 1e-6, "SCM latent is constant — graph degenerate");
}

/// Single-step Sprint (num_steps = 1) skips the renoise and produces a finite latent.
#[test]
fn sprint_scm_single_step_finite() {
    let dev = Device::Cpu;
    let cfg = tiny_sprint_config();
    let trunk = tiny_sprint_trunk();

    let latents = det(&[1, cfg.out_channels as usize, 4, 4], 3);
    let cond = det(&[1, 5, cfg.caption_channels as usize], 11);
    let scheduler = ScmScheduler::new(1);
    assert!(scheduler.is_single_step());

    let cancel = CancelFlag::default();
    let mut steps = 0usize;
    let out = denoise_sprint(
        &trunk,
        &scheduler,
        1,
        latents,
        &cond,
        4.5,
        cfg.guidance_embeds_scale,
        &dev,
        &cancel,
        &mut |p| {
            if matches!(p, Progress::Step { .. }) {
                steps += 1;
            }
        },
    )
    .expect("single-step Sprint SCM denoise");
    assert_eq!(steps, 1, "single-step SCM runs exactly one step");
    let (lo, hi) = stats(&out);
    assert!(lo.is_finite() && hi.is_finite());
}

/// Determinism: same seed reproduces the denoised latent; a 4-step run's renoise actually mixes the
/// per-step seed in, so a different seed diverges.
#[test]
fn sprint_scm_seed_determinism() {
    let dev = Device::Cpu;
    let cfg = tiny_sprint_config();
    let trunk = tiny_sprint_trunk();
    let cond = det(&[1, 5, cfg.caption_channels as usize], 42);

    let run = |seed: u64| -> Vec<f32> {
        let latents = det(&[1, cfg.out_channels as usize, 4, 4], 9);
        let scheduler = ScmScheduler::new(4);
        let cancel = CancelFlag::default();
        let out = denoise_sprint(
            &trunk,
            &scheduler,
            seed,
            latents,
            &cond,
            4.5,
            cfg.guidance_embeds_scale,
            &dev,
            &cancel,
            &mut |_| {},
        )
        .unwrap();
        out.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    };
    assert_eq!(run(7), run(7), "same seed reproduces");
    assert_ne!(run(7), run(8), "different seed diverges (renoise mixes in)");
}
