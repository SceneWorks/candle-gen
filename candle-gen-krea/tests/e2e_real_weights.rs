//! sc-7582 — candle Krea 2 **Turbo** end-to-end real-weight smoke (the Windows/CUDA twin of
//! `mlx-gen-krea`'s `e2e_real_weights.rs`). Loads the full registered engine (`krea_2_turbo`:
//! tokenizer + Qwen3-VL-4B TE + single-stream DiT + Qwen-Image VAE), renders a 1024² image through the
//! `Generator` contract, gates programmatic coherence (a velocity-sign or schedule-direction bug yields
//! pure noise → fails the smoothness gate), and saves the PNG for eyeballing against the mlx render.
//!
//! `#[ignore]` — needs the real snapshot (~12 B params; bf16 ≈ 24 GB resident). Run on the Windows GPU:
//! ```sh
//! KREA_TURBO_DIR=D:\models\Krea-2-Turbo \
//!   cargo test -p candle-gen-krea --release --features cuda --test e2e_real_weights -- --ignored --nocapture
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_gen::candle_core::{safetensors, DType, Device, Tensor};
use candle_gen::gen_core::{
    registry, AdapterKind, AdapterSpec, GenerationOutput, GenerationRequest, Image, LoadSpec,
    WeightsSource,
};
use candle_gen_krea::loader::Weights;
use candle_gen_krea::{merge_adapters, merge_into_weights, Krea2Config};

const PROMPT: &str =
    "A medium-shot photograph of a red fox sitting in a snowy forest at golden hour.";

fn snapshot() -> Option<PathBuf> {
    std::env::var("KREA_TURBO_DIR").ok().map(PathBuf::from)
}

/// (std, distinct-level-count, mean horizontal-adjacent-|Δ|) over an RGB8 buffer — a coherent natural
/// image has a broad histogram AND spatial smoothness; pure noise has a high adjacent Δ and flat std.
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

/// A real Turbo render has a broad histogram (`std`/`distinct`) and spatial smoothness (`adjΔ`); pure
/// noise (the failure mode of a flow-sign / schedule-direction bug) fails the `adjΔ` gate.
fn is_coherent(img: &Image) -> bool {
    let (std, distinct, adj) = image_stats(&img.pixels, img.width);
    std > 10.0 && distinct > 24 && adj < 60.0
}

fn save(img: &Image, name: &str) {
    let dir = std::env::temp_dir().join("krea_turbo_smoke");
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

fn render(width: u32, height: u32) {
    candle_gen_krea::force_link();
    let Some(root) = snapshot() else {
        eprintln!("skipping: set KREA_TURBO_DIR");
        return;
    };

    // The same `load` the `krea_2_turbo` registry entry dispatches to (registration is unit-tested in
    // `tests::registers_krea_2_turbo_as_candle`).
    let spec = LoadSpec::new(WeightsSource::Dir(root));
    let t_load = Instant::now();
    let gen = registry::load("krea_2_turbo", &spec).expect("load krea_2_turbo engine");
    let load_s = t_load.elapsed().as_secs_f32();

    let req = GenerationRequest {
        prompt: PROMPT.into(),
        width,
        height,
        count: 1,
        seed: Some(0),
        steps: Some(8),
        ..Default::default()
    };

    let t_gen = Instant::now();
    let out = gen.generate(&req, &mut |_| {}).expect("generate");
    let gen_s = t_gen.elapsed().as_secs_f32();

    let GenerationOutput::Images(imgs) = out else {
        panic!("expected GenerationOutput::Images");
    };
    assert_eq!(imgs.len(), 1, "count=1 → one image");
    let img = &imgs[0];
    assert_eq!((img.width, img.height), (width, height), "output dims");

    let (std, distinct, adj) = image_stats(&img.pixels, img.width);
    eprintln!(
        "[krea_2_turbo {width}x{height} 8-step] load {load_s:.1}s · render {gen_s:.1}s · \
         std={std:.1} distinct={distinct} adjΔ={adj:.1} coherent={}",
        is_coherent(img)
    );
    save(img, &format!("fox_{width}x{height}_s8"));
    assert!(
        is_coherent(img),
        "Turbo render must be a coherent image, not noise (std={std:.1} distinct={distinct} adjΔ={adj:.1})"
    );
}

#[test]
#[ignore = "needs the real Krea 2 Turbo snapshot (set KREA_TURBO_DIR)"]
fn turbo_engine_renders_coherent_1024() {
    render(1024, 1024);
}

/// sc-9592 regression lock: at 1536² the DiT scores tensor is `48·9216² ≈ 4.1e9 > i32::MAX`, which
/// faulted (CUDA_ERROR_ILLEGAL_ADDRESS) before the sc-9116 query-row chunking guarded `sdpa`.
#[test]
#[ignore = "needs the real snapshot (KREA_TURBO_DIR); larger footprint — run if it fits"]
fn turbo_engine_renders_coherent_1536() {
    render(1536, 1536);
}

#[test]
#[ignore = "needs the real snapshot (KREA_TURBO_DIR); larger footprint — run if it fits"]
fn turbo_engine_renders_coherent_2048() {
    render(2048, 2048);
}

// ── sc-7836 inference-side LoRA/LoKr adapter merge ────────────────────────────────────────────────

/// Write a synthetic bare-dotted `krea_2_raw`-format LoRA covering **every** attention projection
/// (`transformer_blocks.<i>.attn.<to_q|to_k|to_v|to_out.0>`) of the real DiT — the same 112-target
/// surface the trainer (sc-7838) emits — with small random factors (so a merge perturbs but does not
/// destroy the distilled few-step render). `alpha = rank` ⇒ the spec `scale` is the effective strength.
/// Returns the number of targeted modules. Stands in for a real trained adapter (sc-7837 does the real
/// Raw→Turbo round trip); here we exercise the engine **merge + render** path on the real weights.
fn build_synth_adapter(path: &Path, cfg: &Krea2Config, rank: usize, sigma: f32) -> usize {
    let dev = Device::Cpu;
    let (hidden, q, kv) = (cfg.hidden_size, cfg.q_dim(), cfg.kv_dim());
    let mut map: HashMap<String, Tensor> = HashMap::new();
    let mut count = 0usize;
    for i in 0..cfg.num_layers {
        for (proj, out_f, in_f) in [
            ("to_q", q, hidden),
            ("to_k", kv, hidden),
            ("to_v", kv, hidden),
            ("to_out.0", hidden, q),
        ] {
            let base = format!("transformer_blocks.{i}.attn.{proj}");
            let a = Tensor::randn(0f32, sigma, (rank, in_f), &dev).unwrap(); // A [rank, in]
            let b = Tensor::randn(0f32, sigma, (out_f, rank), &dev).unwrap(); // B [out, rank]
            map.insert(format!("{base}.lora_A.weight"), a);
            map.insert(format!("{base}.lora_B.weight"), b);
            map.insert(
                format!("{base}.alpha"),
                Tensor::from_vec(vec![rank as f32], (1,), &dev).unwrap(),
            );
            count += 1;
        }
    }
    safetensors::save(&map, path).unwrap();
    count
}

/// `merge_into_weights` against the **real** DiT key set: a 112-target synthetic adapter must merge
/// every target with nothing skipped (the AC's "every trained target merges, nothing skipped"). GPU-free
/// (reads the attention surface on the CPU), so it runs without `--features cuda`.
#[test]
#[ignore = "needs the real Krea 2 Turbo snapshot (set KREA_TURBO_DIR)"]
fn adapter_merges_every_attention_target() {
    let Some(root) = snapshot() else {
        eprintln!("skipping: set KREA_TURBO_DIR");
        return;
    };
    let cfg = Krea2Config::from_snapshot(&root).expect("parse transformer config");
    let mut w = Weights::from_dir(&root.join("transformer"), &Device::Cpu, DType::BF16)
        .expect("mmap transformer/");

    let path = std::env::temp_dir().join("krea_synth_lora_merge.safetensors");
    let n = build_synth_adapter(&path, &cfg, 4, 0.01);
    let report = merge_into_weights(
        &mut w,
        &cfg,
        &[AdapterSpec::new(path.clone(), 1.0, AdapterKind::Lora)],
    )
    .expect("merge");
    std::fs::remove_file(&path).ok();

    eprintln!(
        "[krea merge] targets={n} merged={} skipped={}",
        report.merged, report.skipped_keys
    );
    assert_eq!(report.merged, n, "every attention target must merge");
    assert_eq!(report.skipped_keys, 0, "nothing may be skipped");
    assert_eq!(n, cfg.num_layers * 4, "112-target surface (28 blocks × 4)");
}

/// Render `req` against `krea_2_turbo` with `adapters` merged, returning the single image.
fn render_with(root: &Path, width: u32, height: u32, adapters: Vec<AdapterSpec>) -> Image {
    let spec = LoadSpec::new(WeightsSource::Dir(root.to_path_buf())).with_adapters(adapters);
    let gen = registry::load("krea_2_turbo", &spec).expect("load krea_2_turbo engine");
    let req = GenerationRequest {
        prompt: PROMPT.into(),
        width,
        height,
        count: 1,
        seed: Some(0),
        steps: Some(8),
        ..Default::default()
    };
    let out = gen.generate(&req, &mut |_| {}).expect("generate");
    let GenerationOutput::Images(mut imgs) = out else {
        panic!("expected GenerationOutput::Images");
    };
    assert_eq!(imgs.len(), 1, "count=1 → one image");
    imgs.remove(0)
}

/// The sc-7836 engine AC on the real GPU: a `krea_2_raw`-format adapter loads + merges at
/// `krea_2_turbo` inference; **scale 0 ≡ base byte-exact** (the LoRA neutral element), and a non-zero
/// scale yields a finite, correctly-sized, coherent image that *differs* from the base (the merge
/// actually moved the weights). Synthetic adapter — the real trained Raw→Turbo round trip is sc-7837.
#[test]
#[ignore = "needs the real Krea 2 Turbo snapshot (KREA_TURBO_DIR); --features cuda"]
fn turbo_engine_applies_lora_adapter() {
    candle_gen_krea::force_link();
    let Some(root) = snapshot() else {
        eprintln!("skipping: set KREA_TURBO_DIR");
        return;
    };
    let cfg = Krea2Config::from_snapshot(&root).expect("parse transformer config");
    let path = std::env::temp_dir().join("krea_synth_lora_render.safetensors");
    build_synth_adapter(&path, &cfg, 4, 0.01);

    let base = render_with(&root, 1024, 1024, vec![]);
    let zero = render_with(
        &root,
        1024,
        1024,
        vec![AdapterSpec::new(path.clone(), 0.0, AdapterKind::Lora)],
    );
    // The strong, deterministic half of the AC: a scale-0 merge is the identity.
    assert_eq!(
        base.pixels, zero.pixels,
        "scale-0 adapter merge must be byte-exact with the base render"
    );

    let adapted = render_with(
        &root,
        1024,
        1024,
        vec![AdapterSpec::new(path.clone(), 1.0, AdapterKind::Lora)],
    );
    std::fs::remove_file(&path).ok();

    assert_eq!((adapted.width, adapted.height), (1024, 1024), "output dims");
    let (std, distinct, adj) = image_stats(&adapted.pixels, adapted.width);
    let diff = base
        .pixels
        .iter()
        .zip(&adapted.pixels)
        .filter(|(a, b)| a != b)
        .count();
    eprintln!(
        "[krea adapter render] std={std:.1} distinct={distinct} adjΔ={adj:.1} coherent={} \
         changed_px={diff}/{}",
        is_coherent(&adapted),
        adapted.pixels.len()
    );
    save(&base, "fox_base_1024");
    save(&adapted, "fox_adapter_s1_1024");
    assert!(
        is_coherent(&adapted),
        "adapted render must be a coherent image (std={std:.1} distinct={distinct} adjΔ={adj:.1})"
    );
    assert!(diff > 0, "a non-zero-scale adapter must change the render");
}

// ── sc-8776 ai-toolkit LoKr sniff + widened-surface merge ─────────────────────────────────────────

const AITOOLKIT_LOKR_ENV: &str = "KREA_REALISM_LOKR";

/// Rewrite an ai-toolkit native module path (`diffusion_model.blocks.N.attn.wq`, `…mlp.down`, …) to the
/// diffusers DiT key the merge folds into — the test-side mirror of the crate-private
/// `normalize_native_krea_path`, kept explicit here so the real-file assertion is independent of it.
fn resolve_native_module(module: &str) -> String {
    let m = module.strip_prefix("diffusion_model.").unwrap_or(module);
    let mut p = if let Some(r) = m.strip_prefix("blocks.") {
        format!("transformer_blocks.{r}")
    } else if let Some(r) = m.strip_prefix("txtfusion.") {
        format!("text_fusion.{r}")
    } else {
        m.to_string()
    };
    p = p.replace(".mlp.", ".ff.");
    p.replace(".attn.wq", ".attn.to_q")
        .replace(".attn.wk", ".attn.to_k")
        .replace(".attn.wv", ".attn.to_v")
        .replace(".attn.wo", ".attn.to_out.0")
        .replace(".attn.gate", ".attn.to_gate")
}

/// The keystone sc-8776 validation on the **real** ostris ai-toolkit LoKr
/// (`realism_engine_krea2_v2.safetensors`, 768 tensors, no `networkType`, full `lokr_w1`/`lokr_w2` +
/// per-target `.alpha`, over `attn.{wq,wk,wv,wo,gate}` + `mlp.{down,gate,up}` across the single-stream
/// **and** text_fusion blocks). Synthesizes a zero base of the exact per-target `[out,in]` implied by
/// each module's Kronecker factors (`out = w1_r·w2_r`, `in = w1_c·w2_c`), then folds the real file in
/// through the public [`merge_adapters`] with the worker's `AdapterKind::Lora` classification. Every one
/// of the 256 targets must merge with **nothing skipped** — the "no adapter target modules matched"
/// failure this story fixes. GPU-free and model-snapshot-free: needs only the adapter file itself.
#[test]
#[ignore = "needs the real ai-toolkit LoKr (set KREA_REALISM_LOKR to the .safetensors)"]
fn real_aitoolkit_lokr_merges_full_surface() {
    let Ok(adapter) = std::env::var(AITOOLKIT_LOKR_ENV) else {
        eprintln!("skipping: set {AITOOLKIT_LOKR_ENV} to realism_engine_krea2_v2.safetensors");
        return;
    };
    let dev = Device::Cpu;
    let tensors = safetensors::load(&adapter, &dev).expect("load ai-toolkit LoKr");

    // Group each module's Kronecker leg dims → the base weight's [out, in].
    #[derive(Default)]
    struct Dims {
        w1: Option<(usize, usize)>,
        w2_out: Option<usize>,
        w2_in: Option<usize>,
    }
    enum Leg {
        W1(usize, usize),
        W1a(usize),
        W1b(usize),
        W2(usize, usize),
        W2a(usize),
        W2b(usize),
    }
    let mut mods: HashMap<String, Dims> = HashMap::new();
    for (k, t) in &tensors {
        let d = t.dims();
        let (module, leg) = if let Some(m) = k.strip_suffix(".lokr_w1") {
            (m, Leg::W1(d[0], d[1]))
        } else if let Some(m) = k.strip_suffix(".lokr_w1_a") {
            (m, Leg::W1a(d[0]))
        } else if let Some(m) = k.strip_suffix(".lokr_w1_b") {
            (m, Leg::W1b(d[1]))
        } else if let Some(m) = k.strip_suffix(".lokr_w2") {
            (m, Leg::W2(d[0], d[1]))
        } else if let Some(m) = k.strip_suffix(".lokr_w2_a") {
            (m, Leg::W2a(d[0]))
        } else if let Some(m) = k.strip_suffix(".lokr_w2_b") {
            (m, Leg::W2b(d[1]))
        } else {
            continue; // .alpha
        };
        let e = mods.entry(module.to_string()).or_default();
        match leg {
            Leg::W1(r, c) => e.w1 = Some((r, c)),
            Leg::W1a(r) => e.w1 = Some((r, e.w1.map_or(0, |(_, c)| c))),
            Leg::W1b(c) => {
                let r = e.w1.map_or(0, |(r, _)| r);
                e.w1 = Some((r, c));
            }
            Leg::W2(o, i) => {
                e.w2_out = Some(o);
                e.w2_in = Some(i);
            }
            Leg::W2a(o) => e.w2_out = Some(o),
            Leg::W2b(i) => e.w2_in = Some(i),
        }
    }

    let mut base: HashMap<String, Tensor> = HashMap::new();
    for (module, dims) in &mods {
        let (w1r, w1c) = dims.w1.expect("every module has a w1 leg");
        let (w2o, w2i) = (dims.w2_out.unwrap(), dims.w2_in.unwrap());
        let (out_f, in_f) = (w1r * w2o, w1c * w2i);
        let key = format!("{}.weight", resolve_native_module(module));
        base.insert(
            key,
            Tensor::zeros((out_f, in_f), DType::BF16, &dev).unwrap(),
        );
    }
    let n = base.len();
    eprintln!("[krea ai-toolkit LoKr] modules={n}");

    let report = merge_adapters(
        &mut base,
        &[AdapterSpec::new(adapter.into(), 1.0, AdapterKind::Lora)],
    )
    .expect("merge must not error (the story's failure)");
    eprintln!(
        "[krea ai-toolkit LoKr] merged={} skipped={}",
        report.merged, report.skipped_keys
    );
    assert_eq!(n, 256, "full ai-toolkit surface = 28×8 + 2×8 + 2×8");
    assert_eq!(report.merged, n, "every ai-toolkit target must merge");
    assert_eq!(report.skipped_keys, 0, "nothing may be skipped");
}

/// The GPU parity smoke: render `krea_2_turbo` with the **real** ai-toolkit LoKr merged (classified
/// `Lora`, as the worker does), asserting a coherent image that differs from the un-adapted base — the
/// candle side of "the same adapter renders on MLX" (sc-8776). Needs both the Turbo snapshot and the
/// adapter file; eyeball `fox_realism_*` against the mlx render.
#[test]
#[ignore = "needs the real Krea 2 Turbo snapshot (KREA_TURBO_DIR) + KREA_REALISM_LOKR; --features cuda"]
fn turbo_engine_applies_aitoolkit_lokr() {
    candle_gen_krea::force_link();
    let (Some(root), Ok(adapter)) = (snapshot(), std::env::var(AITOOLKIT_LOKR_ENV)) else {
        eprintln!("skipping: set KREA_TURBO_DIR and {AITOOLKIT_LOKR_ENV}");
        return;
    };
    let adapter = PathBuf::from(adapter);

    let base = render_with(&root, 1024, 1024, vec![]);
    let adapted = render_with(
        &root,
        1024,
        1024,
        vec![AdapterSpec::new(adapter, 1.0, AdapterKind::Lora)],
    );
    assert_eq!((adapted.width, adapted.height), (1024, 1024), "output dims");
    let (std, distinct, adj) = image_stats(&adapted.pixels, adapted.width);
    let diff = base
        .pixels
        .iter()
        .zip(&adapted.pixels)
        .filter(|(a, b)| a != b)
        .count();
    eprintln!(
        "[krea ai-toolkit render] std={std:.1} distinct={distinct} adjΔ={adj:.1} coherent={} \
         changed_px={diff}/{}",
        is_coherent(&adapted),
        adapted.pixels.len()
    );
    save(&base, "fox_realism_base_1024");
    save(&adapted, "fox_realism_lokr_1024");
    assert!(
        is_coherent(&adapted),
        "ai-toolkit LoKr render must be coherent (std={std:.1} distinct={distinct} adjΔ={adj:.1})"
    );
    assert!(diff > 0, "the LoKr merge must change the render");
}
