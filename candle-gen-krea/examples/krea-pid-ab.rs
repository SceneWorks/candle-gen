//! GPU A/B validation harness for the NVIDIA **PiD** super-resolving decoder on the Krea 2 Turbo
//! candle provider (epic 7840 / sc-7853). Renders the same prompt twice — once through the native
//! Qwen-Image VAE (the byte-exact default) and once through PiD (`req.use_pid = true`, decodes +
//! super-resolves 4× in one pass) — and writes both PNGs for eyeballing.
//!
//! ```text
//! cargo run -p candle-gen-krea --example krea-pid-ab --features cuda --release -- \
//!   <base_snapshot_dir> <pid_qwenimage.safetensors> <gemma-2-2b-it_dir> "<prompt>" [W] [H] [seed]
//! ```
//! With no args it resolves the three weight locations from the local HF cache
//! (`D:/.cache/huggingface`). Krea reuses the Qwen-Image VAE, so its PiD latent space is `qwenimage`.

use candle_gen::gen_core::{
    registry, GenerationOutput, GenerationRequest, LoadSpec, Progress, WeightsSource,
};

/// First existing path from a set of candidates (the HF-cache snapshot layout varies by machine).
fn first_existing(cands: &[&str]) -> Option<String> {
    cands
        .iter()
        .find(|p| std::path::Path::new(p).exists())
        .map(|p| p.to_string())
}

fn mean_u8(pixels: &[u8]) -> f64 {
    if pixels.is_empty() {
        return 0.0;
    }
    pixels.iter().map(|&p| p as f64).sum::<f64>() / pixels.len() as f64
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    candle_gen_krea::force_link();
    let a: Vec<String> = std::env::args().collect();

    let base = a.get(1).cloned().or_else(|| first_existing(&[
        "D:/.cache/huggingface/hub/models--krea--Krea-2-Turbo/snapshots/1161245028ef398cd0a951101b2bbf486464f841",
    ])).ok_or("base Krea snapshot dir not found (pass as arg 1)")?;
    let pid_ckpt = a.get(2).cloned().or_else(|| first_existing(&[
        "D:/.cache/huggingface/hub/models--SceneWorks--pid-qwenimage/snapshots/39d7b0a9003a3fc934d36d8b5658b2d8ea9c1231/pid_qwenimage_2kto4k.safetensors",
    ])).ok_or("pid_qwenimage safetensors not found (pass as arg 2)")?;
    let gemma = a.get(3).cloned().or_else(|| first_existing(&[
        "D:/.cache/huggingface/hub/models--google--gemma-2-2b-it/snapshots/299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8",
    ])).ok_or("gemma-2-2b-it snapshot dir not found (pass as arg 3)")?;

    let prompt = a.get(4).cloned().unwrap_or_else(|| {
        "a red fox sitting in a snowy pine forest at dawn, sharp fur detail".into()
    });
    let width: u32 = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(512);
    let height: u32 = a.get(6).and_then(|s| s.parse().ok()).unwrap_or(512);
    let seed: u64 = a.get(7).and_then(|s| s.parse().ok()).unwrap_or(7);

    eprintln!("base : {base}");
    eprintln!("pid  : {pid_ckpt}");
    eprintln!("gemma: {gemma}");

    // Load once with the PiD aux decoder attached; `req.use_pid` selects it per render.
    let spec = LoadSpec::new(WeightsSource::Dir(base.into())).with_pid(
        WeightsSource::File(pid_ckpt.into()),
        WeightsSource::Dir(gemma.into()),
    );
    let gen = registry::load("krea_2_turbo", &spec)?;

    let mut on_progress = |p: Progress| match p {
        Progress::Step { current, total } => eprintln!("  step {current}/{total}"),
        Progress::Decoding => eprintln!("  decoding…"),
    };

    let base_req = GenerationRequest {
        prompt: prompt.clone(),
        width,
        height,
        count: 1,
        seed: Some(seed),
        ..Default::default()
    };

    // --- A: native VAE baseline ---
    eprintln!("\n[A] native VAE decode ({width}x{height})…");
    let GenerationOutput::Images(mut imgs) = gen.generate(&base_req, &mut on_progress)? else {
        return Err("expected images".into());
    };
    let vae_img = imgs.pop().ok_or("no VAE image")?;
    eprintln!(
        "    VAE: {}x{}  mean={:.1}",
        vae_img.width,
        vae_img.height,
        mean_u8(&vae_img.pixels)
    );
    image::RgbImage::from_raw(vae_img.width, vae_img.height, vae_img.pixels.clone())
        .ok_or("bad VAE buffer")?
        .save("krea_pid_ab_vae.png")?;

    // --- B: PiD super-resolving decode ---
    eprintln!(
        "\n[B] PiD decode (use_pid=true; expect {}x{})…",
        width * 4,
        height * 4
    );
    let pid_req = GenerationRequest {
        use_pid: true,
        ..base_req
    };
    let GenerationOutput::Images(mut imgs) = gen.generate(&pid_req, &mut on_progress)? else {
        return Err("expected images".into());
    };
    let pid_img = imgs.pop().ok_or("no PiD image")?;
    eprintln!(
        "    PiD: {}x{}  mean={:.1}",
        pid_img.width,
        pid_img.height,
        mean_u8(&pid_img.pixels)
    );
    image::RgbImage::from_raw(pid_img.width, pid_img.height, pid_img.pixels.clone())
        .ok_or("bad PiD buffer")?
        .save("krea_pid_ab_pid.png")?;

    // Assertions: PiD is exactly 4× the VAE side, non-degenerate color range.
    assert_eq!(
        pid_img.width,
        vae_img.width * 4,
        "PiD must be 4× the VAE width"
    );
    assert_eq!(
        pid_img.height,
        vae_img.height * 4,
        "PiD must be 4× the VAE height"
    );
    let (lo, hi) = pid_img
        .pixels
        .iter()
        .fold((255u8, 0u8), |(lo, hi), &p| (lo.min(p), hi.max(p)));
    assert!(
        hi - lo > 32,
        "PiD output must span a real color range (got {lo}..{hi})"
    );

    eprintln!(
        "\nOK — wrote krea_pid_ab_vae.png ({}²) + krea_pid_ab_pid.png ({}²).",
        width,
        width * 4
    );
    Ok(())
}
