//! sc-9300 A/B harness: render the same prompt+seed on the canonical **bf16** Krea 2 DiT and the
//! community **INT8-ConvRot** DiT, report per-render wall time + a pixel-difference summary, and write
//! both PNGs. Optionally also renders the sc-9411 **Q4 packed** tier if a packed snapshot dir is given.
//!
//! ```text
//! cargo run -p candle-gen-krea --example krea-convrot-ab --features cuda --release -- \
//!   <canonical_snapshot_dir> <convrot_dit.safetensors> "<prompt>" [W] [H] [steps] [seed] [q4_snapshot_dir]
//! ```
//! The canonical snapshot supplies tokenizer / Qwen3-VL TE / Qwen-Image VAE for every variant; only the
//! DiT weights differ. sm_120 (Blackwell) here — see the sc-9300 PR for the sm_89-audience caveat.
//!
//! **sc-9601: the ConvRot render is now COHERENT.** The checkpoint's stored int8 weight is the rotated
//! `W·R` (regular-Hadamard, group 256); the consume path applies the matching online `RHT(x)` before the
//! int8 IGEMM (sc-9601), so it reconstructs `X·Wᵀ` and renders on par with the Q4 tier (target PSNR ≥ ~20
//! dB vs bf16, up from the sc-9300 NO-GO's ≈ 8 dB). This harness is the honest A/B measurement of that GO.

use std::path::PathBuf;
use std::time::Instant;

use candle_gen::gen_core::{GenerationRequest, Image, Progress};
use candle_gen_krea::pipeline;

fn render(
    comps: &pipeline::Components,
    req: &GenerationRequest,
    device: &candle_gen::candle_core::Device,
) -> Image {
    let mut noop = |_p: Progress| {};
    let imgs = pipeline::render(comps, req, device, &mut noop).expect("render");
    imgs.into_iter().next().expect("one image")
}

/// Mean absolute per-channel difference (0..255) and max, plus a rough PSNR — the honest pixel A/B.
fn pixel_stats(a: &Image, b: &Image) -> (f64, u8, f64) {
    assert_eq!(a.pixels.len(), b.pixels.len(), "same buffer size");
    let (mut sum, mut mx, mut sq) = (0u64, 0u8, 0f64);
    for (x, y) in a.pixels.iter().zip(&b.pixels) {
        let d = x.abs_diff(*y);
        sum += d as u64;
        mx = mx.max(d);
        sq += (d as f64) * (d as f64);
    }
    let n = a.pixels.len() as f64;
    let mse = sq / n;
    let psnr = if mse == 0.0 {
        f64::INFINITY
    } else {
        20.0 * (255.0f64).log10() - 10.0 * mse.log10()
    };
    (sum as f64 / n, mx, psnr)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    candle_gen_krea::force_link();
    let a: Vec<String> = std::env::args().collect();
    let snapshot = PathBuf::from(
        a.get(1)
            .cloned()
            .unwrap_or_else(|| "D:/models/Krea-2-Turbo".into()),
    );
    let convrot = PathBuf::from(
        a.get(2)
            .cloned()
            .unwrap_or_else(|| "D:/krea2_turbo_int8_convrot.safetensors".into()),
    );
    let prompt = a
        .get(3)
        .cloned()
        .unwrap_or_else(|| "a red apple on a wooden table, studio lighting".into());
    let width: u32 = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(1024);
    let height: u32 = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(1024);
    let steps: u32 = a.get(6).and_then(|s| s.parse().ok()).unwrap_or(8);
    let seed: u64 = a.get(7).and_then(|s| s.parse().ok()).unwrap_or(42);
    let q4_dir = a.get(8).cloned();

    let device = candle_gen::default_device()?;
    eprintln!("device: {device:?}");

    let req = GenerationRequest {
        prompt: prompt.clone(),
        width,
        height,
        count: 1,
        seed: Some(seed),
        steps: Some(steps),
        ..Default::default()
    };

    // Canonical bf16.
    let t0 = Instant::now();
    let bf16 = pipeline::load_components(&snapshot, &device, &[], None)?;
    let load_bf16 = t0.elapsed();
    let t1 = Instant::now();
    let img_bf16 = render(&bf16, &req, &device);
    let r_bf16 = t1.elapsed();
    drop(bf16);

    // INT8-ConvRot.
    let t2 = Instant::now();
    let cr = pipeline::load_components_convrot(&snapshot, &convrot, &device)?;
    let load_cr = t2.elapsed();
    let t3 = Instant::now();
    let img_cr = render(&cr, &req, &device);
    let r_cr = t3.elapsed();
    drop(cr);

    let (mad, mx, psnr) = pixel_stats(&img_bf16, &img_cr);

    save(&img_bf16, "krea_ab_bf16.png")?;
    save(&img_cr, "krea_ab_convrot.png")?;

    // Optional Q4 packed tier.
    let mut q4_line =
        String::from("Q4 packed:      (not run — pass a packed snapshot dir as arg 8)");
    if let Some(dir) = q4_dir {
        let t4 = Instant::now();
        let q4 = pipeline::load_components(&PathBuf::from(&dir), &device, &[], None)?;
        let load_q4 = t4.elapsed();
        let t5 = Instant::now();
        let img_q4 = render(&q4, &req, &device);
        let r_q4 = t5.elapsed();
        let (mad4, mx4, psnr4) = pixel_stats(&img_bf16, &img_q4);
        save(&img_q4, "krea_ab_q4.png")?;
        q4_line = format!(
            "Q4 packed:      load {:?}  render {:?}  vs-bf16 MAD {:.2} max {} PSNR {:.1} dB",
            load_q4, r_q4, mad4, mx4, psnr4
        );
    }

    println!("\n==== sc-9601 Krea 2 A/B ({width}x{height}, {steps} steps, seed {seed}) ====");
    println!("prompt: {prompt}");
    println!("canonical bf16: load {load_bf16:?}  render {r_bf16:?}");
    println!("INT8-ConvRot:   load {load_cr:?}  render {r_cr:?}  vs-bf16 MAD {mad:.2} max {mx} PSNR {psnr:.1} dB");
    println!("{q4_line}");
    println!("(pngs: krea_ab_bf16.png / krea_ab_convrot.png)");
    Ok(())
}

fn save(img: &Image, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let buf = image::RgbImage::from_raw(img.width, img.height, img.pixels.clone())
        .ok_or("bad image buffer")?;
    buf.save(path)?;
    Ok(())
}
