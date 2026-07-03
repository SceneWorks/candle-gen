//! GPU A/B validation for the NVIDIA PiD `sdxl` decoder (epic 7840 / sc-7853): render once through the
//! native SDXL VAE and once through PiD (`use_pid`), write both PNGs. Confirms the SDXL de-norm seam
//! (PiD gets the 0.13025-normalized latent, not the de-scaled raw latent).
//!
//! `cargo run -p candle-gen-sdxl --example sdxl-pid-ab --features cuda --release -- [model] [base] [pid] [gemma] [W] [H] [seed]`

use candle_gen::gen_core::{
    registry, GenerationOutput, GenerationRequest, LoadSpec, Progress, WeightsSource,
};

fn me(c: &[&str]) -> Option<String> {
    c.iter()
        .find(|p| std::path::Path::new(p).exists())
        .map(|p| p.to_string())
}
fn mean(px: &[u8]) -> f64 {
    if px.is_empty() {
        0.0
    } else {
        px.iter().map(|&p| p as f64).sum::<f64>() / px.len() as f64
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    candle_gen_sdxl::force_link();
    let a: Vec<String> = std::env::args().collect();
    let model = a.get(1).cloned().unwrap_or_else(|| "sdxl".into());
    let base = a.get(2).cloned().or_else(|| me(&["D:/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b"])).ok_or("base snapshot not found")?;
    let pid = a.get(3).cloned().or_else(|| me(&["D:/.cache/huggingface/hub/models--SceneWorks--pid-sdxl/snapshots/70b494831561dc2c181f04a7f057260b8785419a/pid_sdxl_2kto4k.safetensors"])).ok_or("pid-sdxl not found")?;
    let gemma = a.get(4).cloned().or_else(|| me(&["D:/.cache/huggingface/hub/models--google--gemma-2-2b-it/snapshots/299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8"])).ok_or("gemma not found")?;
    let w: u32 = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(512);
    let h: u32 = a.get(6).and_then(|s| s.parse().ok()).unwrap_or(512);
    let seed: u64 = a.get(7).and_then(|s| s.parse().ok()).unwrap_or(7);
    let prompt = "a red fox sitting in a snowy pine forest at dawn, sharp fur detail".to_string();

    eprintln!("model {model}\nbase  {base}\npid   {pid}\ngemma {gemma}");
    let spec = LoadSpec::new(WeightsSource::Dir(base.into())).with_pid(
        WeightsSource::File(pid.into()),
        WeightsSource::Dir(gemma.into()),
    );
    let gen = registry::load(&model, &spec)?;
    let mut op = |p: Progress| match p {
        Progress::Step { current, total } => eprintln!("  step {current}/{total}"),
        Progress::Decoding => eprintln!("  decoding…"),
    };
    let req = GenerationRequest {
        prompt,
        width: w,
        height: h,
        count: 1,
        seed: Some(seed),
        ..Default::default()
    };

    eprintln!("[A] VAE {w}x{h}…");
    let GenerationOutput::Images(mut i) = gen.generate(&req, &mut op)? else {
        return Err("images".into());
    };
    let v = i.pop().ok_or("no vae img")?;
    eprintln!("  VAE {}x{} mean={:.1}", v.width, v.height, mean(&v.pixels));
    image::RgbImage::from_raw(v.width, v.height, v.pixels.clone())
        .ok_or("buf")?
        .save(format!("{model}_pid_vae.png"))?;

    eprintln!("[B] PiD (expect {}x{})…", w * 4, h * 4);
    let preq = GenerationRequest {
        use_pid: true,
        ..req
    };
    let GenerationOutput::Images(mut i) = gen.generate(&preq, &mut op)? else {
        return Err("images".into());
    };
    let p = i.pop().ok_or("no pid img")?;
    eprintln!("  PiD {}x{} mean={:.1}", p.width, p.height, mean(&p.pixels));
    image::RgbImage::from_raw(p.width, p.height, p.pixels.clone())
        .ok_or("buf")?
        .save(format!("{model}_pid_pid.png"))?;

    assert_eq!(p.width, v.width * 4, "PiD must be 4× the VAE width");
    let (lo, hi) = p
        .pixels
        .iter()
        .fold((255u8, 0u8), |(l, h), &x| (l.min(x), h.max(x)));
    assert!(hi - lo > 32, "degenerate color range {lo}..{hi}");
    eprintln!("OK — wrote {model}_pid_vae.png + {model}_pid_pid.png");
    Ok(())
}
