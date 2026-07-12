//! Krea 2 pose-ControlNet **provider** smoke (sc-8464, epic 8459) — drives the packaged
//! [`Krea2Control`](candle_gen_krea::Krea2Control) exactly as the worker `KreaControl` route does:
//! load the Turbo snapshot + a trained control-branch overlay once, then render one pose-conditioned
//! image from a skeleton PNG.
//!
//! This is the deployable-path sibling of `krea-control-infer` (which stays the low-level byte-identity
//! diagnostic): it validates the public provider API end-to-end against the sc-8460 spike checkpoint
//! before the worker lane wires it. Reproduce the spike's pose-lock:
//!
//! ```text
//! cargo run -p candle-gen-krea --example krea-control-provider --features cuda --release -- \
//!   --snapshot <krea-2-turbo snapshot dir> --ckpt control_step5000.safetensors \
//!   --pose pose.png --prompt "a person dancing" --scale 0.6 --seed 42 --out out.png
//! ```
//!
//! Flags: `--snapshot <dir>` `--ckpt <safetensors>` (required) `--pose <png>` (required)
//! `--prompt <str>` `--scale F` (default 0.6) `--seed N` `--steps N` (default 8) `--size N`
//! (square, default 1024) `--out <png>`.

use std::path::PathBuf;

use candle_gen::gen_core::runtime::CancelFlag;
use candle_gen::gen_core::{Image, Progress};
use candle_gen_krea::{
    Krea2Control, Krea2ControlPaths, Krea2ControlRequest, DEFAULT_CONTROL_SCALE,
};

struct Args {
    snapshot: PathBuf,
    ckpt: PathBuf,
    pose: PathBuf,
    prompt: String,
    scale: f32,
    seed: u64,
    steps: usize,
    size: u32,
    out: PathBuf,
}

fn parse_args() -> Args {
    let mut a = Args {
        snapshot: PathBuf::from("D:/models/Krea-2-Turbo"),
        ckpt: PathBuf::new(),
        pose: PathBuf::new(),
        prompt: "a person standing in a colorful room".into(),
        scale: DEFAULT_CONTROL_SCALE,
        seed: 42,
        steps: 8,
        size: 1024,
        out: PathBuf::from("krea_control_provider.png"),
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
            "--ckpt" => a.ckpt = val().into(),
            "--pose" => a.pose = val().into(),
            "--prompt" => a.prompt = val(),
            "--scale" => a.scale = val().parse().expect("--scale"),
            "--seed" => a.seed = val().parse().expect("--seed"),
            "--steps" => a.steps = val().parse().expect("--steps"),
            "--size" => a.size = val().parse().expect("--size"),
            "--out" => a.out = val().into(),
            other => panic!("unknown flag {other}"),
        }
        i += 2;
    }
    assert!(!a.ckpt.as_os_str().is_empty(), "--ckpt is required");
    assert!(!a.pose.as_os_str().is_empty(), "--pose is required");
    a
}

/// Load a skeleton PNG into a gen_core `Image` (HWC RGB u8) at the render size — the provider requires
/// the control image already at `size`×`size` (the worker driver renders it there; the lib carries no
/// codec). The spike poses are square-canonical, so a direct resize matches the train-time letterbox.
fn load_pose(path: &PathBuf, size: u32) -> Result<Image, Box<dyn std::error::Error>> {
    let rgb = image::open(path)?.to_rgb8();
    let resized = image::imageops::resize(&rgb, size, size, image::imageops::FilterType::Lanczos3);
    Ok(Image {
        width: size,
        height: size,
        pixels: resized.into_raw(),
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = parse_args();
    let pose = load_pose(&a.pose, a.size)?;

    let model = Krea2Control::load(&Krea2ControlPaths {
        root: a.snapshot,
        control: a.ckpt,
        adapters: Vec::new(),
    })?;
    eprintln!(
        "loaded Krea2Control; rendering {}x{} @ scale {}",
        a.size, a.size, a.scale
    );

    let req = Krea2ControlRequest {
        prompt: a.prompt,
        width: a.size,
        height: a.size,
        steps: a.steps,
        control_scale: a.scale,
        seed: a.seed,
        cancel: CancelFlag::new(),
    };
    let mut on_progress = |p: Progress| {
        if let Progress::Step { current, total } = p {
            eprintln!("step {current}/{total}");
        }
    };
    let out = model.generate(&req, &pose, &mut on_progress)?;

    let buf = image::RgbImage::from_raw(out.width, out.height, out.pixels)
        .ok_or("bad output image buffer")?;
    buf.save(&a.out)?;
    eprintln!("wrote {} ({}x{})", a.out.display(), out.width, out.height);
    Ok(())
}
