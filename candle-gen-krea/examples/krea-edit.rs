//! GPU-validation harness for the candle Krea 2 **Edit** lane (`krea_2_edit`, epic 10871 / sc-11085;
//! P4.2 identity validation, sc-10886). The candle twin of the MLX `krea_edit_smoke` (mlx-gen #702):
//! load the Krea 2 Raw snapshot with the `krea2_identity_edit` LoRA folded into the DiT, take one or two
//! reference PNGs + an instruction, render an edited image through the production `krea_2_edit` Generator
//! seam (`registry::load` → `Generator::generate` → `pipeline::render_edit`), and write a PNG.
//!
//! Two-reference order is **fixed**: scene = image 1, person = image 2 (the LoRA's trained layout;
//! swapping degrades results). One ref → `Conditioning::Reference`; two → `Conditioning::MultiReference`.
//!
//! **R5 ablation:** pass the LoRA arg as `none` (or empty) to load WITHOUT the identity LoRA — the dual
//! conditioning (in-context VAE tokens + Qwen3-VL grounding) still runs but is inert/off-distribution,
//! the degraded mode the worker R5 gate blocks. Used for the epic-10871 P4.2 dual-vs-inert delta.
//!
//! ```text
//! # two-reference face-prominent (scene, then person), with the edit LoRA
//! cargo run -p candle-gen-krea --example krea-edit --features cuda --release -- \
//!   E:\...\Krea-2-Raw scene.png,person.png "a close-up of the woman ... in this street" \
//!   1024 1024 16 42 out.png E:\...\krea2_identity_edit_v1_1_r128.safetensors 3.0
//! ```
//! Arg order: <snapshot_dir> <ref.png[,ref2.png]> <instruction> [width=1024] [height=1024] \
//!            [steps=16] [seed=42] [out=krea_edit.png] [lora=none] [guidance=3.0] [lora_scale=1.0]

use candle_gen::gen_core::{
    registry, AdapterKind, AdapterSpec, Conditioning, GenerationOutput, GenerationRequest, Image,
    LoadSpec, Progress, WeightsSource,
};
use image::imageops::FilterType;

/// Snap `n` down to the nearest multiple of 16 (the engine's `SIZE_MULTIPLE`), floored at 256 (`RES_MIN`).
fn snap16(n: u32) -> u32 {
    (n - n % 16).max(256)
}

/// Load a reference PNG/JPG and snap it to a multiple-of-16 RGB8 [`Image`].
fn load_reference(path: &str) -> Result<Image, Box<dyn std::error::Error>> {
    let img = image::open(path)?.to_rgb8();
    let (rw, rh) = (snap16(img.width()), snap16(img.height()));
    let img = if (rw, rh) != (img.width(), img.height()) {
        eprintln!(
            "snapping reference {path} {}x{} -> {rw}x{rh} (multiple of 16)",
            img.width(),
            img.height()
        );
        image::imageops::resize(&img, rw, rh, FilterType::Lanczos3)
    } else {
        img
    };
    Ok(Image {
        width: rw,
        height: rh,
        pixels: img.into_raw(),
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    candle_gen_krea::force_link();

    let a: Vec<String> = std::env::args().collect();
    let snapshot = a
        .get(1)
        .cloned()
        .unwrap_or_else(|| "D:/models/Krea-2-Raw".into());
    let ref_arg = a
        .get(2)
        .cloned()
        .unwrap_or_else(|| "scene.png,person.png".into());
    let instruction = a
        .get(3)
        .cloned()
        .unwrap_or_else(|| "keep the person's face and identity exactly the same".into());
    let width: u32 = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(1024);
    let height: u32 = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(1024);
    // Match the sc-10886 recipe: 16-step Raw CFG (0 → the engine's 52-step Raw default).
    let steps: u32 = a.get(6).and_then(|s| s.parse().ok()).unwrap_or(16);
    let seed: u64 = a.get(7).and_then(|s| s.parse().ok()).unwrap_or(42);
    let out = a.get(8).cloned().unwrap_or_else(|| "krea_edit.png".into());
    // The `krea2_identity_edit` LoRA path; `none`/empty → the R5 no-LoRA ablation.
    let lora = a.get(9).cloned().unwrap_or_else(|| "none".into());
    let guidance: f32 = a.get(10).and_then(|s| s.parse().ok()).unwrap_or(3.0);
    let lora_scale: f32 = a.get(11).and_then(|s| s.parse().ok()).unwrap_or(1.0);

    // References in fixed order (scene = image 1, person = image 2), each snapped to a mult-of-16 buffer.
    let references: Vec<Image> = ref_arg
        .split(',')
        .map(|p| load_reference(p.trim()))
        .collect::<Result<_, _>>()?;

    let conditioning = if references.len() == 1 {
        vec![Conditioning::Reference {
            image: references.into_iter().next().unwrap(),
            strength: None,
        }]
    } else {
        eprintln!(
            "two-reference edit (scene = image 1, person = image 2), {} refs",
            references.len()
        );
        vec![Conditioning::MultiReference { images: references }]
    };

    let mut spec = LoadSpec::new(WeightsSource::Dir(snapshot.clone().into()));
    let no_lora = lora.trim().is_empty() || lora.trim().eq_ignore_ascii_case("none");
    if no_lora {
        eprintln!("[edit] NO edit LoRA (R5 ablation — dual conditioning present but inert)");
    } else {
        eprintln!("[edit] edit LoRA {lora} (scale {lora_scale})");
        spec.adapters = vec![AdapterSpec::new(lora.into(), lora_scale, AdapterKind::Lora)];
    }
    eprintln!("[edit] loading krea_2_edit from {snapshot}");
    let gen = registry::load("krea_2_edit", &spec)?;

    let req = GenerationRequest {
        prompt: instruction.clone(),
        width,
        height,
        count: 1,
        seed: Some(seed),
        steps: if steps == 0 { None } else { Some(steps) },
        guidance: Some(guidance),
        conditioning,
        ..Default::default()
    };
    eprintln!(
        "[edit] '{instruction}' ({width}x{height}, {steps} steps, g={guidance}, seed={seed})"
    );

    let mut on_progress = |p: Progress| match p {
        Progress::Step { current, total } => eprintln!("step {current}/{total}"),
        Progress::Decoding => eprintln!("decoding…"),
    };

    let GenerationOutput::Images(images) = gen.generate(&req, &mut on_progress)? else {
        return Err("expected images".into());
    };
    let result = images.into_iter().next().ok_or("no image")?;

    let buf = image::RgbImage::from_raw(result.width, result.height, result.pixels)
        .ok_or("bad image buffer")?;
    buf.save(&out)?;
    eprintln!("[edit] wrote {out} ({width}x{height})");
    Ok(())
}
