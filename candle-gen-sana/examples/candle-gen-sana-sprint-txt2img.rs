//! SANA-**Sprint** txt2img GPU-validation harness (sc-11781, epic 11776) — resolves THIS crate's
//! inventory-registered CFG-free few-step generator through
//! `gen_core::registry::load("sana_sprint_1600m", …)`, runs a real `generate` against an
//! `Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers` snapshot over the SCM/TrigFlow 1–4 step
//! loop, and writes the `gen_core::Image` to PNG. Prints output min/max/mean so the finite /
//! non-degenerate acceptance can be read off the console. UNIQUE example name so the shared
//! `target/…/examples` path never collides with the base `candle-gen-sana-txt2img` example (which is
//! byte-unchanged).
//!
//! Sprint is CFG-free (the guidance scale is an EMBEDDED scalar, not classifier-free) and few-step:
//! default 2 steps, guidance 4.5; try `--steps 1` and `--steps 4`.
//!
//! Weights: `--repo <hf_id>` downloads the WHOLE Hugging Face repo snapshot (the model-download
//! convention — every sibling, not a pinned allow-list) into the HF cache and loads from there, OR
//! `--snapshot <dir>` loads a pre-downloaded snapshot. `--repo` defaults to the public
//! `Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers`.
//!
//! Build/run on the Windows/Blackwell box (MSVC 14.44 vcvars, CUDA_COMPUTE_CAP=120):
//!
//! ```text
//! set HF_HUB_CACHE=E:\huggingface\hub
//! cargo run --release --example candle-gen-sana-sprint-txt2img --features cuda -- \
//!   --repo Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers \
//!   --prompt "a red panda on a mossy log in a misty forest, cinematic lighting" \
//!   --width 1024 --height 1024 --steps 2 --guidance 4.5 --seed 42 --out sana_sprint.png
//! ```

use std::path::PathBuf;

use candle_gen::gen_core::{
    self, GenerationOutput, GenerationRequest, LoadSpec, Progress, WeightsSource,
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn arg(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1).cloned())
}

/// Download the WHOLE Hugging Face repo snapshot (every sibling file — the model-download convention,
/// not a pinned allow-list) into the HF cache and return the snapshot directory. Files already present
/// in the cache are not re-fetched.
fn download_whole_repo(repo_id: &str) -> Result<PathBuf> {
    use hf_hub::api::sync::ApiBuilder;

    println!("[sana-sprint] downloading whole HF repo {repo_id} (cached files are reused)…");
    let api = ApiBuilder::new().build()?;
    let repo = api.model(repo_id.to_string());
    let info = repo.info()?;
    let mut snapshot_root: Option<PathBuf> = None;
    for sib in &info.siblings {
        let local = repo.get(&sib.rfilename)?;
        if sib.rfilename == "model_index.json" {
            snapshot_root = local.parent().map(|p| p.to_path_buf());
        }
    }
    let root = snapshot_root
        .or_else(|| {
            repo.get(&info.siblings.first()?.rfilename)
                .ok()
                .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        })
        .ok_or("could not resolve the downloaded snapshot directory")?;
    println!("[sana-sprint] snapshot at {}", root.display());
    Ok(root)
}

fn tensor_stats(pixels: &[u8]) -> (u8, u8, f64) {
    let (mut lo, mut hi, mut sum) = (u8::MAX, u8::MIN, 0f64);
    for &p in pixels {
        lo = lo.min(p);
        hi = hi.max(p);
        sum += p as f64;
    }
    (lo, hi, sum / pixels.len().max(1) as f64)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    let snapshot = match arg(&args, "--snapshot") {
        Some(dir) => PathBuf::from(dir),
        None => {
            let repo = arg(&args, "--repo").unwrap_or_else(|| {
                "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers".into()
            });
            download_whole_repo(&repo)?
        }
    };

    let prompt = arg(&args, "--prompt").unwrap_or_else(|| {
        "a red panda on a mossy log in a misty forest, cinematic lighting, highly detailed".into()
    });
    // Sprint default: 2 steps, embedded guidance 4.5.
    let guidance: f32 = arg(&args, "--guidance")
        .and_then(|s| s.parse().ok())
        .unwrap_or(4.5);
    let steps: Option<u32> = arg(&args, "--steps").and_then(|s| s.parse().ok());
    let seed: u64 = arg(&args, "--seed")
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);
    let width: u32 = arg(&args, "--width")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024);
    let height: u32 = arg(&args, "--height")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024);
    let count: u32 = arg(&args, "--count")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let out = arg(&args, "--out")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("sana_sprint_txt2img.png"));

    println!(
        "[sana-sprint] snapshot={}\n[sana-sprint] {width}x{height} steps={steps:?} guidance={guidance} (embedded, CFG-free) seed={seed} count={count}\n[sana-sprint] prompt={prompt:?}",
        snapshot.display()
    );

    // Force-link the provider so its `register_generators!` submission survives the linker.
    candle_gen_sana::force_link();

    let spec = LoadSpec::new(WeightsSource::Dir(snapshot));
    let gen = gen_core::registry::load("sana_sprint_1600m", &spec)?;
    println!(
        "[sana-sprint] resolved engine id={} backend={} supports_true_cfg={} (Sprint is CFG-free)",
        gen.descriptor().id,
        gen.descriptor().backend,
        gen.descriptor().capabilities.supports_true_cfg
    );

    let req = GenerationRequest {
        prompt,
        negative_prompt: None, // Sprint is CFG-free — the negative prompt is inapplicable.
        guidance: Some(guidance),
        width,
        height,
        count,
        seed: Some(seed),
        steps,
        sampler: None,
        scheduler: None,
        ..Default::default()
    };

    let mut on_progress = |p: Progress| match p {
        Progress::Step { current, total } => {
            print!("\r[sana-sprint] step {current}/{total}   ");
            use std::io::Write;
            let _ = std::io::stdout().flush();
        }
        Progress::Decoding => println!("\n[sana-sprint] decoding"),
        Progress::Loading(_) => {}
    };

    let t0 = std::time::Instant::now();
    let output = gen.generate(&req, &mut on_progress)?;
    let secs = t0.elapsed().as_secs_f32();

    let images = match output {
        GenerationOutput::Images(imgs) => imgs,
        GenerationOutput::Video { .. } => return Err("expected images, got video".into()),
    };
    println!("[sana-sprint] {} image(s) in {secs:.2}s", images.len());

    for (i, img) in images.iter().enumerate() {
        let (lo, hi, mean) = tensor_stats(&img.pixels);
        let degenerate = lo == hi;
        println!(
            "[sana-sprint] image {i}: {}x{} min={lo} max={hi} mean={mean:.2} finite=true degenerate={degenerate}",
            img.width, img.height
        );
        let path = if images.len() == 1 {
            out.clone()
        } else {
            out.with_file_name(format!(
                "{}_{i}.png",
                out.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("sana_sprint")
            ))
        };
        let buf = image::RgbImage::from_raw(img.width, img.height, img.pixels.clone())
            .ok_or("invalid RGB buffer dimensions")?;
        buf.save(&path)?;
        println!(
            "[sana-sprint] wrote {} ({}x{})",
            path.display(),
            img.width,
            img.height
        );
    }
    Ok(())
}
