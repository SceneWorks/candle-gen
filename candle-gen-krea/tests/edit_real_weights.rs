//! epic 10871 — candle Krea 2 **image-edit** end-to-end real-weight smoke (the Windows/CUDA twin of the
//! mlx-gen-krea edit validation). Drives the engine pipeline directly (`load_components` +
//! `load_edit_components` + `render_edit`) rather than the `Generator` contract, because the SceneWorks
//! `KreaEdit` routing/descriptor is P3 (sc-10882/10883) — the engine capability is exercised here.
//!
//! The dual-conditioning contract needs BOTH wires ON to be on-distribution (R8), so this loads the
//! community identity-edit LoRA (`conradlocke/krea2-identity-edit`) into the Raw DiT and feeds a real
//! source image. `#[ignore]` — needs the real Krea 2 Raw snapshot, the edit LoRA, a source image, and a
//! CUDA GPU:
//! ```sh
//! KREA_RAW_DIR=D:\models\Krea-2-Raw \
//! KREA_EDIT_LORA=D:\models\krea2-identity-edit.safetensors \
//! KREA_EDIT_SOURCE=D:\fixtures\person.ppm \
//!   cargo test -p candle-gen-krea --release --features cuda --test edit_real_weights -- --ignored --nocapture
//! ```
//! Optionally set `KREA_EDIT_SOURCE_B` for the two-reference (scene = image 1, person = image 2) path.

use std::path::PathBuf;
use std::time::Instant;

use candle_gen::gen_core::{AdapterKind, AdapterSpec, GenerationRequest, Image};
use candle_gen_krea::pipeline::{load_components, load_edit_components, render_edit};

/// (std, distinct-level-count, mean horizontal-adjacent-|Δ|) — a coherent natural image has a broad
/// histogram AND spatial smoothness; pure noise has a high adjacent Δ and flat std.
fn image_stats(px: &[u8], w: u32) -> (f32, usize, f32) {
    let n = px.len() as f64;
    let mean = px.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = px.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
    let mut seen = [false; 256];
    for &v in px {
        seen[v as usize] = true;
    }
    let distinct = seen.iter().filter(|&&b| b).count();
    let stride = (w * 3) as usize;
    let (mut adj_sum, mut adj_n) = (0f64, 0u64);
    for (i, &v) in px.iter().enumerate() {
        if i >= 3 && i % stride >= 3 {
            adj_sum += (v as i32 - px[i - 3] as i32).unsigned_abs() as f64;
            adj_n += 1;
        }
    }
    (
        var.sqrt() as f32,
        distinct,
        (adj_sum / adj_n.max(1) as f64) as f32,
    )
}

fn is_coherent(img: &Image) -> bool {
    let (std, distinct, adj) = image_stats(&img.pixels, img.width);
    std > 10.0 && distinct > 24 && adj < 60.0
}

/// Minimal binary-PPM (P6) reader → an [`Image`] (RGB8). Keeps the fixture format identical to the qwen
/// edit smoke without depending on an image decoder in this test.
fn read_ppm(path: &PathBuf) -> Image {
    let bytes = std::fs::read(path).expect("read source PPM");
    assert_eq!(&bytes[0..2], b"P6", "expected a binary P6 PPM");
    // Parse the 3 whitespace-separated header numbers (width, height, maxval) after the magic.
    let mut it = bytes[2..].iter().copied();
    let mut nums = Vec::with_capacity(3);
    let mut cur = String::new();
    let mut consumed = 2usize;
    for b in it.by_ref() {
        consumed += 1;
        if b.is_ascii_whitespace() {
            if !cur.is_empty() {
                nums.push(cur.parse::<usize>().unwrap());
                cur.clear();
                if nums.len() == 3 {
                    break;
                }
            }
        } else {
            cur.push(b as char);
        }
    }
    let (w, h) = (nums[0], nums[1]);
    let pixels = bytes[consumed..consumed + w * h * 3].to_vec();
    Image {
        width: w as u32,
        height: h as u32,
        pixels,
    }
}

fn save(img: &Image, name: &str) {
    let dir = std::env::temp_dir().join("krea_edit_smoke");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{name}.png"));
    image::save_buffer(
        &path,
        &img.pixels,
        img.width,
        img.height,
        image::ExtendedColorType::Rgb8,
    )
    .unwrap();
    eprintln!("  saved {}", path.display());
}

/// The engine edit AC on the real GPU: load the Raw DiT with the identity-edit LoRA merged, feed the
/// source reference(s) + an instruction, and render a coherent edited image (a flow-sign / seq-concat /
/// RoPE-frame / vision-splice bug yields noise → fails the coherence gate). Two references when
/// `KREA_EDIT_SOURCE_B` is set (scene = image 1, person = image 2 — sc-10878).
#[test]
#[ignore = "needs KREA_RAW_DIR + KREA_EDIT_LORA + KREA_EDIT_SOURCE (a P6 PPM); --features cuda"]
fn edit_renders_coherent_with_dual_conditioning() {
    candle_gen_krea::force_link();
    let (Ok(root), Ok(lora), Ok(source)) = (
        std::env::var("KREA_RAW_DIR"),
        std::env::var("KREA_EDIT_LORA"),
        std::env::var("KREA_EDIT_SOURCE"),
    ) else {
        eprintln!("skipping: set KREA_RAW_DIR + KREA_EDIT_LORA + KREA_EDIT_SOURCE");
        return;
    };
    let root = PathBuf::from(root);
    let device = candle_gen::default_device().expect("device");

    // References in fixed order: scene = image 1, person = image 2 (sc-10878).
    let mut references = vec![read_ppm(&PathBuf::from(&source))];
    if let Ok(b) = std::env::var("KREA_EDIT_SOURCE_B") {
        references.push(read_ppm(&PathBuf::from(b)));
    }
    // Target resolution: `KREA_EDIT_SIZE=WxH` (both multiples of 16) overrides; else the reference's
    // size. A smaller target keeps the DiT joint sequence (and its memory/time) light — the vision
    // grounding still runs at the reference's native resolution regardless.
    let (w, h) = std::env::var("KREA_EDIT_SIZE")
        .ok()
        .and_then(|s| {
            let (a, b) = s.split_once('x')?;
            Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
        })
        .unwrap_or((references[0].width, references[0].height));
    let steps = std::env::var("KREA_EDIT_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20u32);
    let prompt = std::env::var("KREA_EDIT_PROMPT")
        .unwrap_or_else(|_| "make the person smile warmly, keep their identity".into());

    let t_load = Instant::now();
    let comps = load_components(
        &root,
        &device,
        &[AdapterSpec::new(lora.into(), 1.0, AdapterKind::Lora)],
        None,
    )
    .expect("load Krea components + edit LoRA");
    let edit = load_edit_components(&root, &device).expect("load edit components");
    let load_s = t_load.elapsed().as_secs_f32();

    let req = GenerationRequest {
        prompt,
        width: w,
        height: h,
        count: 1,
        seed: Some(0),
        steps: Some(steps),
        guidance: Some(3.5),
        ..Default::default()
    };

    let t_gen = Instant::now();
    let imgs =
        render_edit(&comps, &edit, &req, &references, &device, &mut |_| {}).expect("render_edit");
    let gen_s = t_gen.elapsed().as_secs_f32();

    assert_eq!(imgs.len(), 1, "count=1 → one image");
    let img = &imgs[0];
    assert_eq!((img.width, img.height), (w, h), "output dims");
    let (std, distinct, adj) = image_stats(&img.pixels, img.width);
    eprintln!(
        "[krea edit {w}x{h} refs={}] load {load_s:.1}s · render {gen_s:.1}s · std={std:.1} \
         distinct={distinct} adjΔ={adj:.1} coherent={}",
        references.len(),
        is_coherent(img)
    );
    save(img, &format!("edit_{w}x{h}_refs{}", references.len()));
    assert!(
        is_coherent(img),
        "edit render must be a coherent image, not noise (std={std:.1} distinct={distinct} adjΔ={adj:.1})"
    );
    // When the target matches the source size, the edit must differ from the source — an inert
    // conditioning path would echo the input. (Skipped when KREA_EDIT_SIZE rescaled the target.)
    if (w, h) == (references[0].width, references[0].height) {
        let diff = references[0]
            .pixels
            .iter()
            .zip(&img.pixels)
            .filter(|(a, b)| a != b)
            .count();
        assert!(diff > 0, "the edit must change the source, not echo it");
    }
}
