//! Krea 2 **pose-ControlNet inference hook** — the sc-8460 spike validation harness (epic 8459).
//!
//! Loads the frozen Krea 2 Turbo base (through the composable [`KreaTrainDit`] — the same forward
//! the branch trains against) plus an optional trained
//! [`ControlBranch`](candle_gen_krea::control::ControlBranch) checkpoint, and renders the standard
//! 8-step CFG-free Turbo denoise from (pose PNG, prompt, control_scale, seed).
//!
//! `--scale 0` (or omitting `--ckpt`) never runs the branch — **byte-identical** to the
//! un-branched base generation with the same seed (the spike's identity contract).
//!
//! ```text
//! cargo run -p candle-gen-krea --example krea-control-infer --features cuda --release -- \
//!   --snapshot <krea-2-turbo snapshot dir> --ckpt control_step200.safetensors \
//!   --pose pose.png --prompt "a person dancing" --scale 1.0 --seed 42 --out out.png
//! ```
//!
//! Flags: `--snapshot <dir>` `--ckpt <safetensors>` (optional) `--pose <png>` (required with a
//! ckpt at nonzero scale) `--prompt <str>` `--scale F` (default 1.0) `--seed N` `--steps N`
//! (default 8) `--size N` (square, default 1024) `--out <png>`.

use std::path::PathBuf;

use candle_gen::candle_core::{DType, Tensor};
use candle_gen::gen_core::runtime::CancelFlag;
use candle_gen::gen_core::sampling::TimestepConvention;
use candle_gen::gen_core::Progress;
use candle_gen::train::flow_match::component_vb;
use candle_gen_krea::control::{forward_with_control, ControlBranch};
use candle_gen_krea::loader::Weights;
// The crate's single canonical prompt-token cap (sc-11205 / F-120) — no longer a per-example const.
use candle_gen_krea::pipeline::MAX_TEXT_TOKENS;
use candle_gen_krea::{
    load_vae, turbo_sigmas, Krea2Config, KreaTeConfig, KreaTextEncoder, KreaTokenizer, KreaTrainDit,
};
use candle_gen_qwen_image::vae::QwenVaeEncoder;
use rand::{rngs::StdRng, SeedableRng};

const LATENT_CHANNELS: usize = 16;
const SPATIAL_SCALE: u32 = 8;

struct Args {
    snapshot: PathBuf,
    ckpt: Option<PathBuf>,
    pose: Option<PathBuf>,
    prompt: String,
    scale: f64,
    seed: u64,
    steps: usize,
    size: u32,
    out: PathBuf,
    /// Residual RMS clamp τ (0 = off; default `control::DEFAULT_RESIDUAL_CLAMP`).
    residual_clamp: f64,
}

fn parse_args() -> Args {
    let mut a = Args {
        snapshot: PathBuf::from("D:/models/Krea-2-Turbo"),
        ckpt: None,
        pose: None,
        prompt: "a person standing in a colorful room".into(),
        scale: 1.0,
        seed: 42,
        steps: 8,
        size: 1024,
        out: PathBuf::from("krea_control_render.png"),
        residual_clamp: candle_gen_krea::control::DEFAULT_RESIDUAL_CLAMP,
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let key = argv[i].as_str();
        let val = || {
            argv.get(i + 1)
                .unwrap_or_else(|| panic!("missing value for {key}"))
                .clone()
        };
        match key {
            "--snapshot" => a.snapshot = val().into(),
            "--ckpt" => a.ckpt = Some(val().into()),
            "--pose" => a.pose = Some(val().into()),
            "--prompt" => a.prompt = val(),
            "--scale" => a.scale = val().parse().expect("--scale"),
            "--seed" => a.seed = val().parse().expect("--seed"),
            "--steps" => a.steps = val().parse().expect("--steps"),
            "--size" => a.size = val().parse().expect("--size"),
            "--out" => a.out = val().into(),
            "--residual-clamp" => a.residual_clamp = val().parse().expect("--residual-clamp"),
            other => panic!("unknown flag {other}"),
        }
        i += 2;
    }
    a
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = parse_args();
    let device = candle_gen::default_device()?;
    let use_branch = a.ckpt.is_some() && a.scale != 0.0;

    // Condition encoding (f32 TE, exactly the pipeline's) — encoder dropped after.
    let tokenizer = KreaTokenizer::from_snapshot(&a.snapshot, &device)?;
    let te_cfg = KreaTeConfig::from_snapshot(&a.snapshot)?;
    let te_w = Weights::from_dir(&a.snapshot.join("text_encoder"), &device, DType::F32)?;
    let te = KreaTextEncoder::load(&te_w, "language_model", &te_cfg, MAX_TEXT_TOKENS)?;
    let context = te.forward(&tokenizer.encode_prompt(&a.prompt, MAX_TEXT_TOKENS)?)?;
    drop(te);
    drop(te_w);

    // Control latent: VAE-encode the pose skeleton (only when the branch will actually run).
    let ctrl_latent = if use_branch {
        let pose = a
            .pose
            .as_ref()
            .ok_or("--pose is required with --ckpt at nonzero --scale")?;
        let enc = QwenVaeEncoder::new(component_vb(
            &a.snapshot,
            "vae",
            &device,
            DType::F32,
            "krea control infer",
        )?)?;
        let img = candle_gen::train::dataset::load_image_tensor(pose, a.size, &device)?;
        Some(enc.encode(&img)?)
    } else {
        None
    };

    // Frozen base DiT (bf16, composable — the train-time forward) + optional branch.
    let cfg = Krea2Config::from_snapshot(&a.snapshot)?;
    let dit_w = Weights::from_dir(&a.snapshot.join("transformer"), &device, DType::BF16)?;
    let dit = KreaTrainDit::load(&dit_w, &cfg)?;
    let branch = match (&a.ckpt, use_branch) {
        (Some(p), true) => {
            let mut b = ControlBranch::from_checkpoint(p, &cfg, &device)?;
            // Inference: detach weight reads so the sampler loop builds no autograd graph.
            b.freeze();
            let clamp = (a.residual_clamp > 0.0).then_some(a.residual_clamp);
            b.set_residual_clamp(clamp);
            eprintln!("residual clamp tau: {clamp:?}");
            eprintln!(
                "branch: {} blocks, {:.2}B params, control_scale {}",
                b.num_blocks(),
                b.num_params() as f64 / 1e9,
                a.scale
            );
            Some(b)
        }
        _ => {
            eprintln!("no branch (baseline generation)");
            None
        }
    };
    drop(dit_w);

    // Seeded initial noise — the pipeline's `init_noise` (sc-3673 CPU RNG discipline).
    let lat = (a.size / SPATIAL_SCALE) as usize;
    let mut rng = StdRng::seed_from_u64(a.seed);
    let noise = candle_gen::seeded_normal_vec(&mut rng, LATENT_CHANNELS * lat * lat);
    let noise = Tensor::from_vec(
        noise,
        (1, LATENT_CHANNELS, lat, lat),
        &candle_gen::candle_core::Device::Cpu,
    )?
    .to_device(&device)?;

    // 8-step CFG-free Turbo denoise (raw sigma timestep, Euler `x + v·Δσ`).
    let sigmas = turbo_sigmas(a.steps);
    let cancel = CancelFlag::new();
    let mut on_progress = |p: Progress| {
        if let Progress::Step { current, total } = p {
            eprintln!("step {current}/{total}");
        }
    };
    let lat = candle_gen::run_flow_sampler(
        None,
        TimestepConvention::Sigma,
        &sigmas,
        noise,
        a.seed,
        &cancel,
        &mut on_progress,
        |x, timestep| {
            let t = Tensor::from_vec(vec![timestep], (1,), &device)?;
            let v = match (&branch, &ctrl_latent) {
                (Some(b), Some(c)) => forward_with_control(&dit, b, x, &t, &context, c, a.scale)?,
                _ => dit.forward(x, &t, &context)?,
            };
            Ok(v.to_dtype(DType::F32)?)
        },
    )?;

    // Native Qwen-Image VAE decode -> PNG.
    eprintln!("decoding…");
    let vae = load_vae(&a.snapshot, &device)?;
    let decoded = vae.decode(&lat)?.to_dtype(DType::F32)?; // [1, 3, H, W] in [-1, 1]
    let img = ((decoded.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?.to_dtype(DType::U8)?;
    let img = img
        .squeeze(0)?
        .to_device(&candle_gen::candle_core::Device::Cpu)?;
    let (c, h, w) = img.dims3()?;
    if c != 3 {
        return Err(format!("expected 3 channels, got {c}").into());
    }
    let pixels = img.permute((1, 2, 0))?.flatten_all()?.to_vec1::<u8>()?;
    let buf = image::RgbImage::from_raw(w as u32, h as u32, pixels).ok_or("bad image buffer")?;
    buf.save(&a.out)?;
    eprintln!("wrote {} ({w}x{h})", a.out.display());
    Ok(())
}
