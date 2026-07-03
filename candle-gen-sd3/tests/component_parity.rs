//! SD3.5 diffusers component-parity harness (sc-9076, epic 8979 — the F-001 / F-002 follow-up).
//!
//! Validates the `candle-gen-sd3` port **component-by-component** against a HuggingFace `diffusers` +
//! `transformers` reference, for a fixed prompt / fixed seed / fixed timestep. This is the repeatable
//! bit-exact-ish harness the epic's Highs section asked for: "fix first, then re-validate SD3.5 with a
//! diffusers component-parity run." It catches F-001 / F-002-class regressions numerically rather than
//! by eyeballing an A/B render.
//!
//! The two correctness bugs this specifically re-validates:
//!  * **F-002** (`conditioning.rs`) — the CLIP **pooled** conditioning must be taken at the FIRST EOS
//!    token (HF `argmax` pooling), NOT the last pad slot. Covered by the `clip_l_pooled` /
//!    `clip_g_pooled` / `pooled` checks: a wrong-token pool diverges from the diffusers pooled output.
//!  * **F-001** (`transformer.rs`) — the MMDiT **final** joint block's context-AdaLN chunk order
//!    (`AdaLayerNormContinuous`: `scale, shift = chunk(2)`, scale FIRST). A swapped order scrambles the
//!    predicted velocity into spatial noise. Covered end-to-end by the `dit_velocity` one-step check.
//!
//! ## How it runs
//! Two halves:
//!  1. **Reference** (committed, regenerated out-of-band): `tests/parity/gen_reference.py` dumps every
//!     intermediate tensor to `tests/parity/reference/<tag>_reference.safetensors` + a manifest. Run it
//!     once in the `sd35env` venv; the tensors are small (~10 MB) so they can be committed OR
//!     regenerated locally (see the module `gen_reference.py` header). This test resolves the reference
//!     from `$SD35_PARITY_REF` (a dir) or the in-crate `tests/parity/reference/` default.
//!  2. **Candle** (this test): loads the SAME SD3.5 snapshot via the crate's public component APIs
//!     (`Sd3TextEncoders`, `aggregate`, `Sd3Transformer`), runs the identical fixed inputs, and compares
//!     each component to the reference with **cosine** + **max-abs-diff** against the documented
//!     tolerances baked into the reference manifest.
//!
//! ## Gating & weight resolution (F-069)
//! The real-weight test is `#[ignore]`d (it needs the multi-GB SD3.5 snapshot + the generated
//! reference) and resolves the snapshot via the shared `candle_gen::testkit` HF-cache resolver
//! (`$HF_HUB_CACHE` → `$HF_HOME/hub` → `<home>/.cache/huggingface/hub`), so it never silently no-ops on
//! a missing cache — it panics with an actionable message. Run it explicitly:
//!
//! ```text
//! export HF_HOME=D:/.cache/huggingface
//! # (once) generate the reference in the sd35env venv:
//! python candle-gen-sd3/tests/parity/gen_reference.py \
//!     --model stabilityai/stable-diffusion-3.5-large --out candle-gen-sd3/tests/parity/reference
//! # then, from the workspace root:
//! cargo test -p candle-gen-sd3 --test component_parity -- --ignored --nocapture
//! ```
//!
//! The encoder / conditioning checks run in **f32 on CPU** to match the reference precision exactly.
//! Because loading the ~9.5 GB T5 (f32/CPU) and the 8B DiT in one process is slow, the two halves are
//! **independently skippable** so a run fits a sane budget on the one-GPU box:
//!  * `$SD35_PARITY_SKIP_DIT=1` — run only the F-002 encoder/conditioning half (no GPU build needed).
//!  * `$SD35_PARITY_SKIP_ENCODERS=1` (+ `$SD35_PARITY_CUDA=1`) — run only the F-001 DiT check on GPU;
//!    it reads context/pooled/latent from the reference so it needs no live encoders.
//!
//! The DiT one-step check defaults to **f32 on CPU** (exact but very heavy); `$SD35_PARITY_CUDA=1`
//! runs it in **bf16 on GPU 0** (map a physical card with `CUDA_VISIBLE_DEVICES`). In bf16 the max-abs
//! band widens to a documented value (`$SD35_PARITY_DIT_MAX_ABS`, default 0.25) — the **cosine floor**
//! is the real F-001 guard and is unchanged. See [`load_dit`].
//!
//! ## Validated result (SD3.5-large, this branch)
//! Encoders (f32/CPU): clip_l_pooled cos=1.000000, clip_g_pooled cos=0.999987, clip_l_penultimate
//! cos=1.000000, clip_g_penultimate cos=0.999909, t5_hidden cos=0.999999, pooled cos=0.999992,
//! context cos=0.999999 — all PASS. DiT (bf16/CUDA): dit_velocity cos=0.999226 (F-001 final-AdaLN
//! correct). The harness ALSO caught a real bug: bigG was padded with eos (49407) instead of its
//! configured pad token `!` (0) — before the fix, context cos was 0.98 / clip_g_penultimate cos 0.40;
//! fixed in conditioning.rs (`resolve_clip_pad_id`, sc-9076).

use std::path::{Path, PathBuf};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::testkit::{cosine, require_hf_snapshot_dir};
use candle_gen_sd3::conditioning::{aggregate, EncoderOutputs, Sd3TextEncoders};
use candle_gen_sd3::config::Sd3Config;
use candle_gen_sd3::transformer::Sd3Transformer;

/// The SD3.5 repo whose snapshot the reference was generated from.
const MODEL_REPO: &str = "stabilityai/stable-diffusion-3.5-large";
/// The reference tag (basename) `gen_reference.py --tag` writes.
const TAG: &str = "sd35_large";

/// A single component's parity outcome.
struct Parity {
    name: String,
    cosine: f32,
    max_abs: f32,
    cosine_min: f32,
    max_abs_max: f32,
}

impl Parity {
    fn passed(&self) -> bool {
        self.cosine >= self.cosine_min && self.max_abs <= self.max_abs_max
    }
    fn report(&self) -> String {
        format!(
            "{:<20} cosine={:.6} (>= {:.4})  max_abs={:.3e} (<= {:.1e})  {}",
            self.name,
            self.cosine,
            self.cosine_min,
            self.max_abs,
            self.max_abs_max,
            if self.passed() { "PASS" } else { "FAIL" },
        )
    }
}

/// Resolve the reference directory: `$SD35_PARITY_REF` if set, else the in-crate default.
fn reference_dir() -> PathBuf {
    if let Ok(d) = std::env::var("SD35_PARITY_REF") {
        return PathBuf::from(d);
    }
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("parity")
        .join("reference")
}

/// Load the reference `.safetensors` (all f32) into a name→Tensor map on CPU.
fn load_reference() -> std::collections::HashMap<String, Tensor> {
    let path = reference_dir().join(format!("{TAG}_reference.safetensors"));
    assert!(
        path.is_file(),
        "reference {} missing — generate it with tests/parity/gen_reference.py in the sd35env venv \
         (see the test module header)",
        path.display()
    );
    candle_gen::candle_core::safetensors::load(&path, &Device::Cpu)
        .unwrap_or_else(|e| panic!("load reference {}: {e}", path.display()))
}

/// Read the tolerances table from the reference manifest so the Rust side and Python side share ONE
/// source of truth (no duplicated constants). Returns (cosine_min, max_abs) for `name`.
fn tolerance(name: &str) -> (f32, f32) {
    let path = reference_dir().join(format!("{TAG}_manifest.json"));
    let txt = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read manifest {}: {e}", path.display()));
    let v: serde_json::Value = serde_json::from_str(&txt).expect("parse manifest json");
    let t = &v["tolerances"][name];
    let cos = t["cosine_min"]
        .as_f64()
        .unwrap_or_else(|| panic!("manifest tolerances missing cosine_min for {name}"))
        as f32;
    let ma = t["max_abs"]
        .as_f64()
        .unwrap_or_else(|| panic!("manifest tolerances missing max_abs for {name}"))
        as f32;
    (cos, ma)
}

/// Flatten a tensor to a CPU f32 Vec for the metric helpers.
fn flat(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32)
        .unwrap()
        .to_device(&Device::Cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

/// Max absolute per-element difference between two equal-length flattened tensors.
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "shape mismatch in max_abs_diff");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

/// Build a [`Parity`] for `name` comparing candle `got` to reference `want`.
fn compare(name: &str, got: &Tensor, want: &Tensor) -> Parity {
    assert_eq!(
        got.dims(),
        want.dims(),
        "{name}: candle shape {:?} != reference shape {:?}",
        got.dims(),
        want.dims()
    );
    let g = flat(got);
    let w = flat(want);
    let (cosine_min, max_abs_max) = tolerance(name);
    Parity {
        name: name.to_string(),
        cosine: cosine(&g, &w),
        max_abs: max_abs_diff(&g, &w),
        cosine_min,
        max_abs_max,
    }
}

/// Load the MMDiT from the snapshot `transformer/` dir. Defaults to f32 on CPU (exact, matches the
/// reference precision); with `$SD35_PARITY_CUDA=1` loads bf16 on CUDA:1 to validate the GPU path.
fn load_dit(root: &Path, cfg: &Sd3Config) -> (Sd3Transformer, Device, DType) {
    let cuda = std::env::var("SD35_PARITY_CUDA").ok().as_deref() == Some("1");
    let (device, dtype) = if cuda {
        (
            Device::new_cuda(0)
                .expect("SD35_PARITY_CUDA=1 but no CUDA device 0 (CUDA_VISIBLE_DEVICES?)"),
            DType::BF16,
        )
    } else {
        (Device::Cpu, DType::F32)
    };
    let dir = root.join("transformer");
    let files = candle_gen::sorted_safetensors(&dir, "sd3-parity")
        .unwrap_or_else(|e| panic!("resolve transformer safetensors: {e}"));
    let vb = candle_gen::mmap_var_builder(&files, dtype, &device)
        .unwrap_or_else(|e| panic!("mmap transformer: {e}"));
    let dit = Sd3Transformer::new(cfg, vb).unwrap_or_else(|e| panic!("build MMDiT: {e}"));
    (dit, device, dtype)
}

/// The full component-parity run. `#[ignore]`d — opt in with `--ignored`. Requires the SD3.5 snapshot
/// in the HF cache and the diffusers reference (see the module header). Prints a per-component parity
/// table and asserts every component passes its documented tolerance.
#[test]
#[ignore = "real-weight parity: needs the SD3.5 snapshot + the diffusers reference (see module header)"]
fn sd35_component_parity_vs_diffusers() {
    let root = require_hf_snapshot_dir(MODEL_REPO);
    let refs = load_reference();
    let cfg = Sd3Config::large();

    // The two heavy halves are independently skippable so a run fits a sane wall-clock budget on the
    // one-GPU box (loading the ~9.5 GB T5 in f32 on CPU AND the 8B DiT in one process is slow):
    //   * `$SD35_PARITY_SKIP_ENCODERS=1` -> run only the F-001 DiT check (reads context/pooled/latent
    //     from the reference, so it needs no live encoders — pairs with `$SD35_PARITY_CUDA=1`).
    //   * `$SD35_PARITY_SKIP_DIT=1`      -> run only the F-002 encoder/conditioning half (no GPU build
    //     needed). At least one half must run.
    let skip_enc = std::env::var("SD35_PARITY_SKIP_ENCODERS").ok().as_deref() == Some("1");
    let skip_dit = std::env::var("SD35_PARITY_SKIP_DIT").ok().as_deref() == Some("1");
    assert!(
        !(skip_enc && skip_dit),
        "SD35_PARITY_SKIP_ENCODERS and SD35_PARITY_SKIP_DIT are both set — nothing to validate"
    );

    let mut results: Vec<Parity> = Vec::new();
    let mut dit_label = "SKIPPED";

    // -- F-002 surface: text encoders + conditioning, in f32 on CPU (matches reference precision) --
    if !skip_enc {
        let cpu = Device::Cpu;
        let mut encoders = Sd3TextEncoders::load(&root, cfg.t5_seq_len, &cpu, DType::F32)
            .unwrap_or_else(|e| panic!("load SD3.5 text encoders: {e}"));
        let prompt = read_prompt();
        let enc: EncoderOutputs = encoders
            .encode(&prompt)
            .unwrap_or_else(|e| panic!("encode prompt: {e}"));

        // Projected pooled CLIP embeds (F-002: first-EOS pooling).
        results.push(compare(
            "clip_l_pooled",
            &enc.clip_l_pooled,
            &refs["clip_l_pooled"],
        ));
        results.push(compare(
            "clip_g_pooled",
            &enc.clip_g_pooled,
            &refs["clip_g_pooled"],
        ));
        // Penultimate hidden states (feed the joint context; the sc-9076 bigG pad-token fix lives here).
        results.push(compare(
            "clip_l_penultimate",
            &enc.clip_l_hidden,
            &refs["clip_l_penultimate"],
        ));
        results.push(compare(
            "clip_g_penultimate",
            &enc.clip_g_hidden,
            &refs["clip_g_penultimate"],
        ));
        results.push(compare("t5_hidden", &enc.t5_hidden, &refs["t5_hidden"]));

        // Aggregated pooled + context (the diffusers encode_prompt concat/pad/order).
        let cond = aggregate(&cfg, &enc).unwrap_or_else(|e| panic!("aggregate conditioning: {e}"));
        results.push(compare("pooled", &cond.pooled, &refs["pooled"]));
        results.push(compare("context", &cond.context, &refs["context"]));
    }

    // -- F-001 surface: the one-step DiT velocity (exercises the final context-AdaLN end to end) --
    if !skip_dit {
        let t0 = std::time::Instant::now();
        eprintln!("[parity] loading MMDiT ...");
        let (dit, device, dtype) = load_dit(&root, &cfg);
        eprintln!(
            "[parity] MMDiT loaded in {:.1}s",
            t0.elapsed().as_secs_f32()
        );
        dit_label = if device.is_cuda() {
            "bf16/CUDA"
        } else {
            "f32/CPU"
        };
        // The reference dumped the exact fixed latent + the context/pooled it fed the DiT — use them
        // verbatim so this check isolates the MMDiT forward from the encoders (validated separately).
        let latent = refs["dit_latent_in"]
            .to_device(&device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let ctx = refs["context"].to_device(&device).unwrap();
        let pooled = refs["pooled"].to_device(&device).unwrap();
        let timestep = Tensor::new(&[read_timestep()], &device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let t1 = std::time::Instant::now();
        eprintln!("[parity] DiT forward ...");
        let vel = dit
            .forward(&latent, &ctx, &pooled, &timestep)
            .unwrap_or_else(|e| panic!("DiT forward: {e}"));
        eprintln!("[parity] DiT forward in {:.1}s", t1.elapsed().as_secs_f32());
        // Precision-aware max-abs: the manifest tolerance is the f32-vs-f32 band. When the DiT ran in
        // **bf16 on GPU** (the default fast path, `$SD35_PARITY_CUDA=1`) vs the f32 reference, the
        // velocity drifts more per element after 38 joint blocks (a few % of its ~3.6 range), so the
        // max-abs bound widens to a documented bf16 band (`$SD35_PARITY_DIT_MAX_ABS`, default 0.25).
        // The **cosine floor is unchanged** — it is the actual F-001 (final context-AdaLN) guard; a
        // swapped scale/shift collapses cosine far below 1 regardless of precision.
        let mut p = compare("dit_velocity", &vel, &refs["dit_velocity"]);
        if device.is_cuda() {
            let band = std::env::var("SD35_PARITY_DIT_MAX_ABS")
                .ok()
                .and_then(|s| s.parse::<f32>().ok())
                .unwrap_or(0.25);
            p.max_abs_max = p.max_abs_max.max(band);
        }
        results.push(p);
    }

    // -- Report + assert --------------------------------------------------------------------------
    eprintln!("\n=== SD3.5 component-parity vs diffusers ({MODEL_REPO}) ===");
    eprintln!(
        "  encoders: {}   DiT: {}",
        if skip_enc { "SKIPPED" } else { "f32/CPU" },
        dit_label
    );
    for p in &results {
        eprintln!("  {}", p.report());
    }
    eprintln!("=== (F-002 = clip_*_pooled + pooled + context; F-001 = dit_velocity) ===\n");

    let failed: Vec<&Parity> = results.iter().filter(|p| !p.passed()).collect();
    assert!(
        failed.is_empty(),
        "component parity FAILED for: {}",
        failed
            .iter()
            .map(|p| p.name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );
}

/// The fixed prompt — read from the reference manifest so it stays in lockstep with the Python side.
fn read_prompt() -> String {
    let path = reference_dir().join(format!("{TAG}_manifest.json"));
    let txt = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read manifest {}: {e}", path.display()));
    let v: serde_json::Value = serde_json::from_str(&txt).expect("parse manifest json");
    v["prompt"].as_str().expect("manifest prompt").to_string()
}

/// The fixed DiT timestep (in the 0..1000 convention), read from the reference manifest.
fn read_timestep() -> f32 {
    let path = reference_dir().join(format!("{TAG}_manifest.json"));
    let txt = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read manifest {}: {e}", path.display()));
    let v: serde_json::Value = serde_json::from_str(&txt).expect("parse manifest json");
    v["timestep"].as_f64().expect("manifest timestep") as f32
}

/// A tensor-shape sanity guard that runs WITHOUT weights (not `#[ignore]`d): confirms the reference
/// artifact, if present, has the shapes the harness expects. Skips gracefully when the reference has
/// not been generated (so a fresh checkout's `cargo test` stays green). This keeps the harness wiring
/// honest even on a box without the SD3.5 weights.
#[test]
fn reference_manifest_shapes_are_consistent_if_present() {
    let manifest = reference_dir().join(format!("{TAG}_manifest.json"));
    if !manifest.is_file() {
        eprintln!(
            "skip: no reference manifest at {} (run gen_reference.py to enable the parity harness)",
            manifest.display()
        );
        return;
    }
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&manifest).unwrap()).unwrap();
    let t = &v["tensors"];
    // Aggregated conditioning shapes at the SD3.5 large defaults.
    assert_eq!(
        t["pooled"].as_array().unwrap().len(),
        2,
        "pooled is [B, 2048]"
    );
    assert_eq!(t["pooled"][1].as_u64().unwrap(), 2048);
    assert_eq!(
        t["context"][2].as_u64().unwrap(),
        4096,
        "context hidden = 4096"
    );
    assert_eq!(
        t["context"][1].as_u64().unwrap(),
        (77 + v["t5_len"].as_u64().unwrap()),
        "context seq = 77 + t5_len"
    );
    // DiT velocity is [1, 16, H, W].
    assert_eq!(
        t["dit_velocity"][1].as_u64().unwrap(),
        16,
        "16 latent channels"
    );
    // Every tolerance entry has both bounds.
    for name in [
        "clip_l_pooled",
        "clip_g_pooled",
        "pooled",
        "context",
        "dit_velocity",
    ] {
        assert!(
            v["tolerances"][name]["cosine_min"].is_number(),
            "tolerance cosine_min present for {name}"
        );
        assert!(
            v["tolerances"][name]["max_abs"].is_number(),
            "tolerance max_abs present for {name}"
        );
    }
}
