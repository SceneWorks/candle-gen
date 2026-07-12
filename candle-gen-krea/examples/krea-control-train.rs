//! Krea 2 **pose-ControlNet control-branch trainer** — the sc-8460 spike harness (epic 8459).
//!
//! Trains the [`candle_gen_krea::control::ControlBranch`] (N trainable copies of the first N
//! single-stream DiT blocks + zero-init residual projections) against the **frozen** Krea 2 Turbo
//! base on (target image, pose skeleton, caption) triples, with the same rectified-flow velocity
//! objective the Krea LoRA trainer uses, sampled across the full noise schedule.
//!
//! ```text
//! cargo run -p candle-gen-krea --example krea-control-train --features cuda --release -- \
//!   --snapshot D:\.cache\huggingface\hub\models--krea--Krea-2-Turbo\snapshots\<sha> \
//!   --data D:\sceneworks-pose-controlnet\spike\train \
//!   --out  D:\krea-control-ckpt --steps 200 --batch 1 --lr 1e-4 --save-every 100
//! ```
//!
//! Dataset layout: `<data>/manifest.jsonl` with rows
//! `{"id","target","pose","caption",...}` (paths relative to the manifest dir), images square.
//! `--synth N` first writes N synthetic (gradient target, stick-figure pose) pairs + manifest into
//! `--data` — the self-contained smoke path when the real COCO-pose data isn't staged yet.
//!
//! Flags: `--snapshot <dir>` `--data <dir>` `--out <dir>` `--steps N` `--batch N` (gradient
//! accumulation — memory-flat) `--lr F` `--save-every N` `--n-blocks N` (default 7)
//! `--resolution N` (default 512) `--seed N` `--branch-dtype bf16|f32` (default bf16)
//! `--resume <ckpt.safetensors>` `--synth N`
//! `--timestep-type sigmoid|uniform|linear|weighted` (default uniform)
//! `--warmup-steps N` (linear LR warmup from 0; default 0 = off)
//! `--checkpoint true|false` (gradient-checkpointed backward, default true — the dense backward
//! OOMs ≥ 512² on a 96 GB card).

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_gen::candle_core::DType;
use candle_gen::train::flow_match::component_vb;
use candle_gen_krea::control::ControlBranch;
use candle_gen_krea::control_train::{
    ControlSample, ControlTrainConfig, ControlTrainer, TrainEvent,
};
use candle_gen_krea::loader::Weights;
// The crate's single canonical prompt-token cap (sc-11205 / F-120) — no longer a per-example const.
use candle_gen_krea::pipeline::MAX_TEXT_TOKENS;
use candle_gen_krea::{Krea2Config, KreaTeConfig, KreaTextEncoder, KreaTokenizer, KreaTrainDit};
use candle_gen_qwen_image::vae::QwenVaeEncoder;

struct Args {
    snapshot: PathBuf,
    data: PathBuf,
    out: PathBuf,
    steps: u32,
    batch: u32,
    lr: f32,
    save_every: u32,
    n_blocks: usize,
    resolution: u32,
    seed: u64,
    branch_dtype: DType,
    resume: Option<PathBuf>,
    synth: usize,
    timestep_type: String,
    /// Linear LR warmup from 0 over the first N steps (0 = off).
    warmup_steps: u32,
    /// Gradient-checkpointed backward (default on — dense OOMs ≥ 512² on a 96 GB card).
    checkpoint: bool,
    /// Residual RMS clamp τ (0 = off). Default `control::DEFAULT_RESIDUAL_CLAMP` — prevents the
    /// block-0 stream-overwrite degeneracy the step-500 probe found (sc-8460).
    residual_clamp: f64,
    /// Branch block `i` injects into main block `i + offset` (default
    /// `control::DEFAULT_INJECT_OFFSET` = 1: skip main block 0, the degenerate site).
    inject_offset: usize,
    /// lr multiplier for the injection-projection group (default 0.1).
    proj_lr_mult: f32,
    /// Decoupled weight decay for the injection-projection group (default 0.05).
    proj_weight_decay: f32,
}

fn parse_args() -> Args {
    let mut a = Args {
        snapshot: PathBuf::from("D:/models/Krea-2-Turbo"),
        data: PathBuf::from("./control-data"),
        out: PathBuf::from("./control-ckpt"),
        steps: 200,
        batch: 1,
        lr: 1e-4,
        save_every: 100,
        n_blocks: 7,
        resolution: 512,
        seed: 42,
        branch_dtype: DType::BF16,
        resume: None,
        synth: 0,
        timestep_type: "uniform".into(),
        warmup_steps: 0,
        checkpoint: true,
        residual_clamp: candle_gen_krea::control::DEFAULT_RESIDUAL_CLAMP,
        inject_offset: candle_gen_krea::control::DEFAULT_INJECT_OFFSET,
        proj_lr_mult: 0.1,
        proj_weight_decay: 0.05,
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
            "--data" => a.data = val().into(),
            "--out" => a.out = val().into(),
            "--steps" => a.steps = val().parse().expect("--steps"),
            "--batch" => a.batch = val().parse::<u32>().expect("--batch").max(1),
            "--lr" => a.lr = val().parse().expect("--lr"),
            "--save-every" => a.save_every = val().parse().expect("--save-every"),
            "--n-blocks" => a.n_blocks = val().parse().expect("--n-blocks"),
            "--resolution" => a.resolution = val().parse().expect("--resolution"),
            "--seed" => a.seed = val().parse().expect("--seed"),
            "--branch-dtype" => {
                a.branch_dtype = match val().to_ascii_lowercase().as_str() {
                    "bf16" | "bfloat16" => DType::BF16,
                    "f32" | "fp32" | "float32" => DType::F32,
                    other => panic!("--branch-dtype must be bf16|f32, got {other}"),
                }
            }
            "--resume" => a.resume = Some(val().into()),
            "--synth" => a.synth = val().parse().expect("--synth"),
            "--timestep-type" => a.timestep_type = val(),
            "--warmup-steps" => a.warmup_steps = val().parse().expect("--warmup-steps"),
            "--residual-clamp" => a.residual_clamp = val().parse().expect("--residual-clamp"),
            "--inject-offset" => a.inject_offset = val().parse().expect("--inject-offset"),
            "--proj-lr-mult" => a.proj_lr_mult = val().parse().expect("--proj-lr-mult"),
            "--proj-weight-decay" => {
                a.proj_weight_decay = val().parse().expect("--proj-weight-decay")
            }
            "--checkpoint" => {
                a.checkpoint = match val().to_ascii_lowercase().as_str() {
                    "true" | "1" | "on" => true,
                    "false" | "0" | "off" => false,
                    other => panic!("--checkpoint must be true|false, got {other}"),
                }
            }
            other => panic!("unknown flag {other}"),
        }
        i += 2;
    }
    a
}

struct Row {
    id: String,
    target: String,
    pose: String,
    caption: String,
}

fn read_manifest(data: &Path) -> Vec<Row> {
    let path = data.join("manifest.jsonl");
    let text =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let v: serde_json::Value =
                serde_json::from_str(l).unwrap_or_else(|e| panic!("manifest row: {e}"));
            let field = |k: &str| {
                v.get(k)
                    .and_then(|s| s.as_str())
                    .unwrap_or_else(|| panic!("manifest row missing string field {k:?}: {l}"))
                    .to_string()
            };
            let target = field("target");
            // Encode-cache key: the manifest id, falling back to the target filename stem.
            let id = v
                .get("id")
                .and_then(|s| s.as_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| {
                    Path::new(&target)
                        .file_stem()
                        .map(|s| s.to_string_lossy().into_owned())
                        .unwrap_or_else(|| target.clone())
                });
            Row {
                id,
                target,
                pose: field("pose"),
                caption: field("caption"),
            }
        })
        .collect()
}

// ── synthetic smoke data ─────────────────────────────────────────────────────────────────────

fn draw_line(img: &mut image::RgbImage, p0: (f32, f32), p1: (f32, f32), c: [u8; 3], th: i32) {
    let (w, h) = (img.width() as i32, img.height() as i32);
    let steps = ((p1.0 - p0.0).abs().max((p1.1 - p0.1).abs()) as i32).max(1);
    for s in 0..=steps {
        let t = s as f32 / steps as f32;
        let x = (p0.0 + (p1.0 - p0.0) * t) as i32;
        let y = (p0.1 + (p1.1 - p0.1) * t) as i32;
        for dx in -th..=th {
            for dy in -th..=th {
                let (px, py) = (x + dx, y + dy);
                if px >= 0 && px < w && py >= 0 && py < h {
                    img.put_pixel(px as u32, py as u32, image::Rgb(c));
                }
            }
        }
    }
}

/// A stick figure at `(cx, cy)` with per-sample limb angles; drawn in `c` on `img`.
fn draw_figure(img: &mut image::RgbImage, cx: f32, cy: f32, scale: f32, i: usize, c: [u8; 3]) {
    let a = (i as f32) * 0.7;
    let head = (cx, cy - 0.35 * scale);
    let hip = (cx, cy + 0.1 * scale);
    // head "circle"
    for k in 0..24 {
        let t0 = (k as f32) / 24.0 * std::f32::consts::TAU;
        let t1 = ((k + 1) as f32) / 24.0 * std::f32::consts::TAU;
        let r = 0.08 * scale;
        draw_line(
            img,
            (head.0 + r * t0.cos(), head.1 + r * t0.sin()),
            (head.0 + r * t1.cos(), head.1 + r * t1.sin()),
            c,
            3,
        );
    }
    let neck = (cx, cy - 0.25 * scale);
    draw_line(img, neck, hip, c, 4); // spine
    let arm = 0.22 * scale;
    draw_line(
        img,
        neck,
        (
            neck.0 - arm * (0.9 + 0.3 * a.sin()),
            neck.1 + arm * a.cos().abs(),
        ),
        c,
        4,
    );
    draw_line(
        img,
        neck,
        (
            neck.0 + arm * (0.9 + 0.3 * a.cos()),
            neck.1 + arm * a.sin().abs(),
        ),
        c,
        4,
    );
    let leg = 0.3 * scale;
    draw_line(img, hip, (hip.0 - leg * 0.5, hip.1 + leg), c, 4);
    draw_line(img, hip, (hip.0 + leg * 0.5, hip.1 + leg), c, 4);
}

/// Write `count` synthetic (target, pose, caption) pairs + `manifest.jsonl` into `dir`.
fn synth_pairs(dir: &Path, count: usize) {
    std::fs::create_dir_all(dir).expect("create --data dir");
    let edge = 1024u32;
    let mut manifest = String::new();
    for i in 0..count {
        let mut target = image::RgbImage::from_fn(edge, edge, |x, y| {
            let r = ((x as f32 / edge as f32) * 200.0) as u8;
            let g = ((y as f32 / edge as f32) * 200.0) as u8;
            let b = ((i * 40) % 255) as u8;
            image::Rgb([r, g, b])
        });
        let cx = edge as f32 * (0.35 + 0.3 * ((i as f32 * 0.37).sin().abs()));
        let cy = edge as f32 * 0.5;
        let scale = edge as f32 * 0.6;
        draw_figure(&mut target, cx, cy, scale, i, [240, 220, 200]);

        let mut pose = image::RgbImage::from_pixel(edge, edge, image::Rgb([0, 0, 0]));
        draw_figure(&mut pose, cx, cy, scale, i, [255, 255, 255]);

        let (tname, pname) = (format!("target_{i}.png"), format!("pose_{i}.png"));
        target.save(dir.join(&tname)).expect("save target");
        pose.save(dir.join(&pname)).expect("save pose");
        manifest.push_str(&format!(
            "{}\n",
            serde_json::json!({
                "id": format!("synth_{i}"),
                "target": tname,
                "pose": pname,
                "caption": format!("a person standing in a colorful room, synthetic pose sample {i}"),
                "coco_image_id": 0,
            })
        ));
    }
    std::fs::write(dir.join("manifest.jsonl"), manifest).expect("write manifest");
    eprintln!("synthesized {count} pairs into {}", dir.display());
}

// ── main ─────────────────────────────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = parse_args();
    if a.synth > 0 {
        synth_pairs(&a.data, a.synth);
    }
    let rows = read_manifest(&a.data);
    if rows.is_empty() {
        return Err("manifest is empty".into());
    }
    std::fs::create_dir_all(&a.out)?;
    let device = candle_gen::default_device()?;
    let edge = candle_gen::train::dataset::bucket_resolution(a.resolution);
    eprintln!(
        "krea control trainer: {} items, {} steps, batch {}, lr {}, n_blocks {}, res {edge}, branch dtype {:?}",
        rows.len(),
        a.steps,
        a.batch,
        a.lr,
        a.n_blocks,
        a.branch_dtype
    );

    // ── cache: VAE-encode targets + poses, TE-encode captions. The cache lives on the **CPU**
    // (and is persisted to `<data>/.encode-cache/` keyed by manifest id + resolution, so a relaunch
    // skips the encode pass entirely); each sample is copied to the device inside the step loop just
    // before use. A GPU-resident cache scales with dataset size — at 5k items it filled the card and
    // WDDM-paged the whole run (~400 s/step). Latents stay f32 (the flow-match mix runs in f32);
    // the caption stack is stored bf16 (the DiT casts it to bf16 at forward anyway — identical
    // values, half the RAM: ~3 MB/item → ~16 GB CPU RAM at 5k items).
    let t_cache = Instant::now();
    let cpu = candle_gen::candle_core::Device::Cpu;
    let cache_dir = a.data.join(".encode-cache");
    std::fs::create_dir_all(&cache_dir)?;
    let cache_paths: Vec<PathBuf> = rows
        .iter()
        .map(|r| cache_dir.join(format!("{}_{edge}.safetensors", r.id)))
        .collect();

    // Load the encoders only if at least one item is missing from the disk cache.
    let encoders = if cache_paths.iter().any(|p| !p.exists()) {
        let vae_enc = QwenVaeEncoder::new(component_vb(
            &a.snapshot,
            "vae",
            &device,
            DType::F32,
            "krea control trainer",
        )?)?;
        let tokenizer = KreaTokenizer::from_snapshot(&a.snapshot, &device)?;
        let te_cfg = KreaTeConfig::from_snapshot(&a.snapshot)?;
        let te_w = Weights::from_dir(&a.snapshot.join("text_encoder"), &device, DType::F32)?;
        let te = KreaTextEncoder::load(&te_w, "language_model", &te_cfg, MAX_TEXT_TOKENS)?;
        Some((vae_enc, tokenizer, te, te_w))
    } else {
        eprintln!(
            "encode cache complete under {} — skipping the encoders",
            cache_dir.display()
        );
        None
    };

    // (x0 latent f32, control latent f32, caption stack bf16), all on CPU.
    let mut cache: Vec<ControlSample> = Vec::with_capacity(rows.len());
    for (i, row) in rows.iter().enumerate() {
        let path = &cache_paths[i];
        let sample = if path.exists() {
            ControlSample::load(path)?
        } else {
            let (vae_enc, tokenizer, te, _) = encoders
                .as_ref()
                .expect("encoders loaded when any item is uncached");
            let target = candle_gen::train::dataset::load_image_tensor(
                &a.data.join(&row.target),
                edge,
                &device,
            )?;
            let pose = candle_gen::train::dataset::load_image_tensor(
                &a.data.join(&row.pose),
                edge,
                &device,
            )?;
            let x0 = vae_enc.encode(&target)?.to_device(&cpu)?;
            let ctrl = vae_enc.encode(&pose)?.to_device(&cpu)?;
            let ids = tokenizer.encode_prompt(&row.caption, MAX_TEXT_TOKENS)?;
            // (1, L, layers, hidden) -> the unbatched (L, layers, hidden) stack
            // `control_loss_grads` consumes (it re-adds the batch axis), stored bf16 on CPU.
            let cap = te
                .forward(&ids)?
                .squeeze(0)?
                .to_dtype(DType::BF16)?
                .to_device(&cpu)?;
            let sample = ControlSample { x0, ctrl, cap };
            sample.save(path)?;
            sample
        };
        cache.push(sample);
        if (i + 1) % 100 == 0 || i + 1 == rows.len() {
            eprintln!("cached {}/{}", i + 1, rows.len());
        }
    }
    drop(encoders);
    eprintln!("cache done in {:.1}s", t_cache.elapsed().as_secs_f32());

    // ── frozen base DiT (bf16) + trainable branch ──
    let cfg = Krea2Config::from_snapshot(&a.snapshot)?;
    let dit_w = Weights::from_dir(&a.snapshot.join("transformer"), &device, DType::BF16)?;
    let dit = KreaTrainDit::load(&dit_w, &cfg)?;

    let (mut branch, start_step) = match &a.resume {
        Some(ckpt) => {
            let b = ControlBranch::from_checkpoint(ckpt, &cfg, &device)?;
            let meta = ckpt.with_extension("json");
            let step = std::fs::read_to_string(&meta)
                .ok()
                .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
                .and_then(|v| v.get("step").and_then(|s| s.as_u64()))
                .unwrap_or(0) as u32;
            eprintln!(
                "resumed {} ({} blocks, {} params) at step {step}",
                ckpt.display(),
                b.num_blocks(),
                b.num_params()
            );
            (b, step)
        }
        None => {
            let b = ControlBranch::from_base(
                &dit_w,
                &cfg,
                a.n_blocks,
                a.branch_dtype,
                a.inject_offset,
            )?;
            eprintln!(
                "fresh branch: {} blocks, {:.2}B trainable params, inject offset {}",
                b.num_blocks(),
                b.num_params() as f64 / 1e9,
                b.inject_offset()
            );
            (b, 0)
        }
    };
    drop(dit_w);

    let clamp = (a.residual_clamp > 0.0).then_some(a.residual_clamp);
    branch.set_residual_clamp(clamp);
    eprintln!("residual clamp tau: {clamp:?}");

    // The trainable target, optimizer groups, accumulation, clip, warmup, checkpointing, and
    // telemetry now live in the reusable `ControlTrainer` (sc-8462) — this example is a thin CLI over
    // it. The numerics are unchanged from the spike; the trainer streams `TrainEvent`s we render to
    // stdout + a JSONL log (the same lines the spike printed).
    let cfg_train = ControlTrainConfig {
        lr: a.lr,
        proj_lr_mult: a.proj_lr_mult,
        proj_weight_decay: a.proj_weight_decay,
        batch: a.batch,
        max_steps: a.steps,
        warmup_steps: a.warmup_steps,
        timestep_type: a.timestep_type.clone(),
        seed: a.seed,
        grad_checkpoint: a.checkpoint,
        mae: false,
        compute_dtype: DType::BF16,
        save_every: a.save_every,
        resolution: edge,
        // This spike CLI trains pose control; the studio path sets this from the job's control_type.
        control_type: Some("pose".into()),
    };
    eprintln!(
        "optimizer groups: body lr {} | proj lr {} wd {}",
        a.lr,
        a.lr * a.proj_lr_mult,
        a.proj_weight_decay
    );

    let log_path = a.out.join("train_log.jsonl");
    let mut log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)?;
    let total = start_step + a.steps;

    let mut trainer = ControlTrainer::new(
        dit,
        branch,
        cache,
        cfg_train,
        a.out.clone(),
        start_step,
        device,
    )?;
    trainer.run(|ev| match ev {
        TrainEvent::Step(r) => {
            println!(
                "step {:>5}/{total} loss {:.5} grad_norm {:.3} lr {:.2e} {:.2}s",
                r.step, r.loss, r.grad_norm, r.lr, r.secs
            );
            let _ = writeln!(
                log,
                "{}",
                serde_json::json!({"step": r.step, "loss": r.loss, "grad_norm": r.grad_norm, "secs": r.secs, "lr": r.lr})
            );
        }
        TrainEvent::Telemetry { step, pre, post } => {
            println!("telemetry step {step:>5}: res/main pre-clamp {pre:?} post-clamp {post:?}");
            let _ = writeln!(
                log,
                "{}",
                serde_json::json!({"step": step, "telemetry_pre": pre, "telemetry_post": post})
            );
        }
        TrainEvent::Checkpoint { path, .. } => {
            eprintln!("checkpoint -> {}", path.display());
        }
    })?;
    eprintln!("training complete; log at {}", log_path.display());
    Ok(())
}
