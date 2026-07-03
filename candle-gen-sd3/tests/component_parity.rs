//! SD3.5 diffusers component-parity harness (sc-9076, epic 8979 — the F-001 / F-002 follow-up;
//! extended to Medium (MMDiT-X) + Large-Turbo in sc-9580).
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
//! ## Variants (sc-9580)
//! The harness is **tag / config / repo-driven** via [`VARIANTS`], so one code path validates every
//! SD3.5 variant against its own committed golden reference:
//!  * `sd35_large`       — `stabilityai/stable-diffusion-3.5-large` (`Sd3Config::large()`, 38 joint
//!    blocks, no dual-attention). The original sc-9076 coverage.
//!  * `sd35_medium`      — `stabilityai/stable-diffusion-3.5-medium` (`Sd3Config::medium()`, the
//!    **MMDiT-X** model: 24 blocks, the first 13 carrying the second image-only self-attention
//!    (`dual_attention_layers=[0..=12]`) and a 9-chunk `SD35AdaLayerNormZeroX` `norm1`). This is the
//!    highest-value add — the `dit_velocity` one-step check exercises the dual-attention joint blocks
//!    the Large run never touches, end to end (including F-001's final context-AdaLN on this variant).
//!  * `sd35_large_turbo` — `stabilityai/stable-diffusion-3.5-large-turbo` (guidance-distilled sibling;
//!    shares the Large MMDiT geometry `Sd3Config::large()`, different checkpoint). Confirms the port
//!    reads the distilled weights with the same numerical fidelity.
//!
//! ## How it runs
//! Two halves:
//!  1. **Reference** (committed, regenerated out-of-band): `tests/parity/gen_reference.py` dumps every
//!     intermediate tensor to `tests/parity/reference/<tag>_reference.safetensors` + a manifest. Run it
//!     once per variant in the parity venv; the tensors are small (~10 MB) so they can be committed OR
//!     regenerated locally (see the module `gen_reference.py` header). This test resolves the reference
//!     from `$SD35_PARITY_REF` (a dir) or the in-crate `tests/parity/reference/` default.
//!  2. **Candle** (this test): loads the SAME SD3.5 snapshot via the crate's public component APIs
//!     (`Sd3TextEncoders`, `aggregate`, `Sd3Transformer`), runs the identical fixed inputs, and compares
//!     each component to the reference with **cosine** + **max-abs-diff** against the documented
//!     tolerances baked into that variant's reference manifest.
//!
//! ## Gating & weight resolution (F-069)
//! Each variant's real-weight test is `#[ignore]`d (it needs the multi-GB SD3.5 snapshot + the
//! generated reference) and resolves the snapshot via the shared `candle_gen::testkit` HF-cache
//! resolver (`$HF_HUB_CACHE` → `$HF_HOME/hub` → `<home>/.cache/huggingface/hub`), so it never silently
//! no-ops on a missing cache — it panics with an actionable message. Run them explicitly:
//!
//! ```text
//! export HF_HOME=D:/.cache/huggingface
//! # (once, per variant) generate the reference in the parity venv:
//! python candle-gen-sd3/tests/parity/gen_reference.py \
//!     --model stabilityai/stable-diffusion-3.5-medium --tag sd35_medium \
//!     --out candle-gen-sd3/tests/parity/reference
//! # then, from the workspace root, one variant at a time:
//! cargo test -p candle-gen-sd3 --test component_parity sd35_medium_component_parity -- --ignored --nocapture
//! ```
//!
//! The encoder / conditioning checks run in **f32 on CPU** to match the reference precision exactly.
//! Because loading the ~9.5 GB T5 (f32/CPU) and the DiT in one process is slow, the two halves are
//! **independently skippable** so a run fits a sane budget on the one-GPU box:
//!  * `$SD35_PARITY_SKIP_DIT=1` — run only the F-002 encoder/conditioning half (no GPU build needed).
//!  * `$SD35_PARITY_SKIP_ENCODERS=1` (+ `$SD35_PARITY_CUDA=1`) — run only the F-001 DiT check on GPU;
//!    it reads context/pooled/latent from the reference so it needs no live encoders.
//!
//! The DiT one-step check defaults to **f32 on CPU** (exact but very heavy — matches the reference
//! essentially bit-exactly); `$SD35_PARITY_CUDA=1` runs it in **bf16 on GPU 0** (map a physical card
//! with `CUDA_VISIBLE_DEVICES`). In bf16 both bounds relax to documented bands
//! (`$SD35_PARITY_DIT_MAX_ABS` default 1.5, `$SD35_PARITY_DIT_COSINE_MIN` default 0.975) — the cosine
//! floor is still the real F-001 guard (a swapped final-AdaLN collapses cosine to ~0). The strict f32
//! manifest floor (asserted when the DiT runs f32/CPU) is the real correctness gate. See the DiT block
//! below and [`load_dit`].
//!
//! ## Validated result
//! Encoders (f32/CPU) are identical across all three variants (they share the triple-TE stack):
//! clip_l_pooled cos=1.000000, clip_g_pooled cos=0.999987, clip_l_penultimate cos=1.000000,
//! clip_g_penultimate cos=0.999909, t5_hidden cos=0.999999, pooled cos=0.999992, context cos=0.999999 —
//! all PASS.
//!
//! DiT one-step velocity:
//!  * **f32/CPU** (the strict correctness gate) — Large cos=1.000000, **Medium (MMDiT-X)**
//!    cos=1.000000 max_abs=2.2e-5, **Large-Turbo** cos=1.000000 max_abs=1.9e-4. The Medium dual-
//!    attention joint blocks and every variant's final context-AdaLN (F-001) are bit-exact vs diffusers.
//!  * **bf16/CUDA** (the fast path) — Large cos=0.999226, Medium cos=0.997696 (max_abs 0.55),
//!    Large-Turbo cos=0.976093 (max_abs 1.39). Turbo's distilled weights drift most in bf16; its f32
//!    match is perfect, so this is precision, not a bug.
//!
//! The Large harness ALSO caught a real bug in sc-9076: bigG was padded with eos (49407) instead of its
//! configured pad token `!` (0) — fixed in conditioning.rs (`resolve_clip_pad_id`).

use std::path::{Path, PathBuf};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::testkit::{cosine, require_hf_snapshot_dir};
use candle_gen_sd3::conditioning::{aggregate, EncoderOutputs, Sd3TextEncoders};
use candle_gen_sd3::config::Sd3Config;
use candle_gen_sd3::transformer::Sd3Transformer;

/// A parity variant: the reference `--tag`, the SD3.5 HF repo the snapshot + reference come from, and
/// the `Sd3Config` preset the candle side builds its components with. Large and Large-Turbo share the
/// Large MMDiT geometry; Medium is the MMDiT-X (dual-attention) preset — driving the config off the tag
/// is what lets the SAME code path validate the dual-attention joint blocks.
struct Variant {
    /// The reference basename `gen_reference.py --tag` wrote (e.g. `sd35_medium`).
    tag: &'static str,
    /// The SD3.5 repo whose snapshot the reference was generated from (and the candle side loads).
    repo: &'static str,
    /// Builds the architecture config for this variant.
    config: fn() -> Sd3Config,
}

/// The SD3.5 variants the harness covers. Adding a variant = one row here + its committed golden pair
/// (`gen_reference.py --model <repo> --tag <tag>`); the per-variant `#[test]` wrappers below dispatch
/// into the shared [`run_parity`].
const VARIANTS: &[Variant] = &[
    Variant {
        tag: "sd35_large",
        repo: "stabilityai/stable-diffusion-3.5-large",
        config: Sd3Config::large,
    },
    Variant {
        // The MMDiT-X model — exercises the dual-attention joint blocks (the highest-value add).
        tag: "sd35_medium",
        repo: "stabilityai/stable-diffusion-3.5-medium",
        config: Sd3Config::medium,
    },
    Variant {
        // Guidance-distilled sibling; shares the Large MMDiT geometry, different checkpoint.
        tag: "sd35_large_turbo",
        repo: "stabilityai/stable-diffusion-3.5-large-turbo",
        config: Sd3Config::large,
    },
];

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

/// Load a variant's reference `.safetensors` (all f32) into a name→Tensor map on CPU.
fn load_reference(tag: &str) -> std::collections::HashMap<String, Tensor> {
    let path = reference_dir().join(format!("{tag}_reference.safetensors"));
    assert!(
        path.is_file(),
        "reference {} missing — generate it with tests/parity/gen_reference.py in the parity venv \
         (see the test module header)",
        path.display()
    );
    candle_gen::candle_core::safetensors::load(&path, &Device::Cpu)
        .unwrap_or_else(|e| panic!("load reference {}: {e}", path.display()))
}

/// Parse a variant's reference manifest JSON.
fn manifest(tag: &str) -> serde_json::Value {
    let path = reference_dir().join(format!("{tag}_manifest.json"));
    let txt = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read manifest {}: {e}", path.display()));
    serde_json::from_str(&txt).expect("parse manifest json")
}

/// Read the tolerances table from a variant's reference manifest so the Rust side and Python side
/// share ONE source of truth (no duplicated constants). Returns (cosine_min, max_abs) for `name`.
fn tolerance(tag: &str, name: &str) -> (f32, f32) {
    let v = manifest(tag);
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

/// Build a [`Parity`] for `name` comparing candle `got` to reference `want` under `tag`'s tolerances.
fn compare(tag: &str, name: &str, got: &Tensor, want: &Tensor) -> Parity {
    assert_eq!(
        got.dims(),
        want.dims(),
        "{name}: candle shape {:?} != reference shape {:?}",
        got.dims(),
        want.dims()
    );
    let g = flat(got);
    let w = flat(want);
    let (cosine_min, max_abs_max) = tolerance(tag, name);
    Parity {
        name: name.to_string(),
        cosine: cosine(&g, &w),
        max_abs: max_abs_diff(&g, &w),
        cosine_min,
        max_abs_max,
    }
}

/// Load the MMDiT from the snapshot `transformer/` dir. Defaults to f32 on CPU (exact, matches the
/// reference precision); with `$SD35_PARITY_CUDA=1` loads bf16 on CUDA:0 to validate the GPU path.
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

/// The full component-parity run for one [`Variant`]. Requires the variant's SD3.5 snapshot in the HF
/// cache and the diffusers reference (see the module header). Prints a per-component parity table and
/// asserts every component passes its documented tolerance.
fn run_parity(variant: &Variant) {
    let Variant { tag, repo, config } = variant;
    let root = require_hf_snapshot_dir(repo);
    let refs = load_reference(tag);
    let cfg = config();

    // The two heavy halves are independently skippable so a run fits a sane wall-clock budget on the
    // one-GPU box (loading the ~9.5 GB T5 in f32 on CPU AND the DiT in one process is slow):
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
        let prompt = read_prompt(tag);
        let enc: EncoderOutputs = encoders
            .encode(&prompt)
            .unwrap_or_else(|e| panic!("encode prompt: {e}"));

        // Projected pooled CLIP embeds (F-002: first-EOS pooling).
        results.push(compare(
            tag,
            "clip_l_pooled",
            &enc.clip_l_pooled,
            &refs["clip_l_pooled"],
        ));
        results.push(compare(
            tag,
            "clip_g_pooled",
            &enc.clip_g_pooled,
            &refs["clip_g_pooled"],
        ));
        // Penultimate hidden states (feed the joint context; the sc-9076 bigG pad-token fix lives here).
        results.push(compare(
            tag,
            "clip_l_penultimate",
            &enc.clip_l_hidden,
            &refs["clip_l_penultimate"],
        ));
        results.push(compare(
            tag,
            "clip_g_penultimate",
            &enc.clip_g_hidden,
            &refs["clip_g_penultimate"],
        ));
        results.push(compare(
            tag,
            "t5_hidden",
            &enc.t5_hidden,
            &refs["t5_hidden"],
        ));

        // Aggregated pooled + context (the diffusers encode_prompt concat/pad/order).
        let cond = aggregate(&cfg, &enc).unwrap_or_else(|e| panic!("aggregate conditioning: {e}"));
        results.push(compare(tag, "pooled", &cond.pooled, &refs["pooled"]));
        results.push(compare(tag, "context", &cond.context, &refs["context"]));
    }

    // -- F-001 surface: the one-step DiT velocity (exercises the final context-AdaLN end to end; on
    //    Medium this is the MMDiT-X path — the dual-attention joint blocks + 9-chunk norm1) --
    if !skip_dit {
        let t0 = std::time::Instant::now();
        eprintln!("[parity] loading MMDiT ({tag}) ...");
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
        let timestep = Tensor::new(&[read_timestep(tag)], &device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let t1 = std::time::Instant::now();
        eprintln!("[parity] DiT forward ...");
        let vel = dit
            .forward(&latent, &ctx, &pooled, &timestep)
            .unwrap_or_else(|e| panic!("DiT forward: {e}"));
        eprintln!("[parity] DiT forward in {:.1}s", t1.elapsed().as_secs_f32());
        // Precision-aware tolerances. The manifest bounds are the **f32-vs-f32** band; in f32/CPU EVERY
        // variant's DiT matches the diffusers reference essentially bit-exactly, which is what actually
        // proves the port correct (validated f32/CPU dit_velocity: Large/Medium/Turbo all cos=1.000000,
        // max_abs 2e-5..2e-4 — including the Medium MMDiT-X dual-attention path). When the DiT ran in
        // **bf16 on GPU** (the default fast path, `$SD35_PARITY_CUDA=1`) vs the f32 reference, the
        // velocity drifts per element after the 24-38 joint blocks, so BOTH bounds relax to a documented
        // bf16 band:
        //   * max-abs -> `$SD35_PARITY_DIT_MAX_ABS` (default 1.5). The velocity absmax is ~4; the
        //     **guidance-distilled Turbo** checkpoint has the widest bf16 drift (its distilled weights
        //     carry larger activations that bf16 rounds more coarsely — measured max_abs ~1.39, vs ~0.55
        //     Medium / ~0.3 Large).
        //   * cosine  -> `$SD35_PARITY_DIT_COSINE_MIN` (default 0.975). This stays the F-001 (final
        //     context-AdaLN) guard: a swapped scale/shift scrambles the velocity into noise and collapses
        //     cosine to ~0, far below this floor, regardless of precision. Measured bf16/CUDA cosines:
        //     Large 0.9992, Medium 0.9977, Turbo 0.9761 (Turbo is the worst-conditioned in bf16; its f32
        //     cosine is a perfect 1.0). Run the DiT half in f32/CPU (drop `$SD35_PARITY_CUDA`) to assert
        //     the strict manifest floor instead — that is the real correctness gate.
        let mut p = compare(tag, "dit_velocity", &vel, &refs["dit_velocity"]);
        if device.is_cuda() {
            let band = std::env::var("SD35_PARITY_DIT_MAX_ABS")
                .ok()
                .and_then(|s| s.parse::<f32>().ok())
                .unwrap_or(1.5);
            p.max_abs_max = p.max_abs_max.max(band);
            let cos_floor = std::env::var("SD35_PARITY_DIT_COSINE_MIN")
                .ok()
                .and_then(|s| s.parse::<f32>().ok())
                .unwrap_or(0.975);
            p.cosine_min = p.cosine_min.min(cos_floor);
        }
        results.push(p);
    }

    // -- Report + assert --------------------------------------------------------------------------
    eprintln!("\n=== SD3.5 component-parity vs diffusers ({repo}) [{tag}] ===");
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
        "component parity FAILED for {tag}: {}",
        failed
            .iter()
            .map(|p| p.name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );
}

/// **sc-9581 A/B**: prove the sc-9076 bigG pad-token fix (bigG pads with its configured `!` = 0,
/// not eos 49407) demonstrably changes the SD3.5 conditioning AND the step-0 DiT velocity — i.e.
/// the fix matters end-to-end, not just at the parity metric.
///
/// For the fixed prompt it encodes twice through the SAME loaded encoders:
///  * **fixed**  — `encode` (bigG pad = resolved `pad_g` = `!` = 0, sc-9076);
///  * **pre-fix** — `encode_with_pad_g(.., 49407)` (bigG padded with eos, the pre-sc-9076 behavior).
///
/// then aggregates each to the joint `context` and asserts:
///  1. fixed↔pre-fix context **differs** (cosine < 1, non-trivial max-abs) — the fix is not a no-op;
///  2. fixed is **closer to the diffusers reference** than pre-fix is (cosine_fixed > cosine_prefix),
///     i.e. the fix moves the conditioning toward the correct diffusers output;
///  3. (optional, `$SD35_PARITY_CUDA=1`) the one-step DiT **velocity** built from the two contexts
///     also differs — so the fix propagates to a different latent at step 0.
///
/// Runs against the **Large** variant ([`VARIANTS`]`[0]`) — the checkpoint the sc-9076 pad fix was
/// found on. Gating mirrors the main parity test (needs the SD3.5 snapshot + the diffusers
/// reference). The DiT half is skipped unless `$SD35_PARITY_CUDA=1` (needs the GPU build of the 8B
/// MMDiT). Requires the `sc9581-ab` feature (the `encode_with_pad_g` hook). Run:
/// ```text
/// export HF_HOME=D:/.cache/huggingface
/// cargo test -p candle-gen-sd3 --features sc9581-ab,cuda --test component_parity \
///     sd35_bigg_pad_token_ab -- --ignored --nocapture
/// ```
#[test]
#[cfg(feature = "sc9581-ab")]
#[ignore = "sc-9581 A/B: needs the SD3.5 snapshot + the diffusers reference (see module header)"]
fn sd35_bigg_pad_token_ab() {
    // The pad fix was found on SD3.5-large; run the A/B against that variant's snapshot + reference.
    let variant = &VARIANTS[0];
    let tag = variant.tag;
    let root = require_hf_snapshot_dir(variant.repo);
    let refs = load_reference(tag);
    let cfg = (variant.config)();
    let cpu = Device::Cpu;

    let mut encoders = Sd3TextEncoders::load(&root, cfg.t5_seq_len, &cpu, DType::F32)
        .unwrap_or_else(|e| panic!("load SD3.5 text encoders: {e}"));
    // Sanity: the resolved bigG pad IS `!` = 0 (the fix), distinct from eos 49407.
    assert_eq!(
        encoders.pad_g(),
        0,
        "sc-9076: bigG must resolve pad_token `!` = 0 (got {})",
        encoders.pad_g()
    );
    let prompt = read_prompt(tag);

    let enc_fixed = encoders
        .encode(&prompt)
        .unwrap_or_else(|e| panic!("encode (fixed pad_g=0): {e}"));
    let enc_prefix = encoders
        .encode_with_pad_g(&prompt, 49407)
        .unwrap_or_else(|e| panic!("encode (pre-fix pad_g=eos): {e}"));

    let ctx_fixed = aggregate(&cfg, &enc_fixed).unwrap().context;
    let ctx_prefix = aggregate(&cfg, &enc_prefix).unwrap().context;
    let ref_ctx = &refs["context"];

    let f = flat(&ctx_fixed);
    let p = flat(&ctx_prefix);
    let r = flat(ref_ctx);

    let cos_fixed_vs_prefix = cosine(&f, &p);
    let maxabs_fixed_vs_prefix = max_abs_diff(&f, &p);
    let cos_fixed_vs_ref = cosine(&f, &r);
    let cos_prefix_vs_ref = cosine(&p, &r);

    // Also isolate the bigG penultimate (the tensor the pad actually corrupts).
    let gf = flat(&enc_fixed.clip_g_hidden);
    let gp = flat(&enc_prefix.clip_g_hidden);
    let gr = flat(&refs["clip_g_penultimate"]);
    let cos_g_fixed_vs_prefix = cosine(&gf, &gp);
    let cos_g_fixed_vs_ref = cosine(&gf, &gr);
    let cos_g_prefix_vs_ref = cosine(&gp, &gr);

    eprintln!("\n=== sc-9581 bigG pad-token A/B (fixed `!`=0 vs pre-fix eos=49407) ===");
    eprintln!("  prompt: {prompt:?}");
    eprintln!("  clip_g_penultimate:");
    eprintln!("    fixed↔pre-fix   cosine={cos_g_fixed_vs_prefix:.6}");
    eprintln!("    fixed↔diffusers cosine={cos_g_fixed_vs_ref:.6}");
    eprintln!("    prefix↔diffusers cosine={cos_g_prefix_vs_ref:.6}");
    eprintln!("  joint context [1, 333, 4096]:");
    eprintln!(
        "    fixed↔pre-fix   cosine={cos_fixed_vs_prefix:.6}  max_abs={maxabs_fixed_vs_prefix:.3e}"
    );
    eprintln!("    fixed↔diffusers cosine={cos_fixed_vs_ref:.6}");
    eprintln!("    prefix↔diffusers cosine={cos_prefix_vs_ref:.6}");

    // (1) the fix is NOT a no-op for this (short) prompt: the pad slots differ.
    assert!(
        cos_fixed_vs_prefix < 0.9999,
        "fix appears to be a no-op: fixed↔pre-fix context cosine={cos_fixed_vs_prefix:.6} (expected < 0.9999 for a padded prompt)"
    );
    assert!(
        maxabs_fixed_vs_prefix > 1e-4,
        "fixed and pre-fix contexts are identical (max_abs={maxabs_fixed_vs_prefix:.3e})"
    );
    // (2) the fix moves the conditioning TOWARD diffusers.
    assert!(
        cos_fixed_vs_ref > cos_prefix_vs_ref,
        "fix did not improve diffusers parity: fixed↔ref cosine={cos_fixed_vs_ref:.6} !> prefix↔ref cosine={cos_prefix_vs_ref:.6}"
    );
    assert!(
        cos_g_fixed_vs_ref > cos_g_prefix_vs_ref,
        "fix did not improve bigG penultimate parity: fixed={cos_g_fixed_vs_ref:.6} !> prefix={cos_g_prefix_vs_ref:.6}"
    );

    // (3) optional: prove it reaches the latent — the one-step DiT velocity differs.
    if std::env::var("SD35_PARITY_CUDA").ok().as_deref() == Some("1") {
        let (dit, device, dtype) = load_dit(&root, &cfg);
        let latent = refs["dit_latent_in"]
            .to_device(&device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let pooled = refs["pooled"]
            .to_device(&device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let timestep = Tensor::new(&[read_timestep(tag)], &device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let cf = ctx_fixed
            .to_device(&device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let cp = ctx_prefix
            .to_device(&device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let vel_fixed = dit.forward(&latent, &cf, &pooled, &timestep).unwrap();
        let vel_prefix = dit.forward(&latent, &cp, &pooled, &timestep).unwrap();
        let vf = flat(&vel_fixed);
        let vp = flat(&vel_prefix);
        let vel_cos = cosine(&vf, &vp);
        let vel_maxabs = max_abs_diff(&vf, &vp);
        eprintln!("  step-0 DiT velocity [1,16,H,W] (bf16/CUDA):");
        eprintln!("    fixed↔pre-fix   cosine={vel_cos:.6}  max_abs={vel_maxabs:.3e}");
        assert!(
            vel_maxabs > 1e-3,
            "fix did not change the step-0 latent velocity (max_abs={vel_maxabs:.3e})"
        );
    }
    eprintln!("=== sc-9581 A/B PASS: bigG pad fix changes conditioning AND improves diffusers parity ===\n");
}

/// The fixed prompt for `tag` — read from its reference manifest so it stays in lockstep with Python.
fn read_prompt(tag: &str) -> String {
    manifest(tag)["prompt"]
        .as_str()
        .expect("manifest prompt")
        .to_string()
}

/// The fixed DiT timestep (in the 0..1000 convention) for `tag`, read from its reference manifest.
fn read_timestep(tag: &str) -> f32 {
    manifest(tag)["timestep"]
        .as_f64()
        .expect("manifest timestep") as f32
}

// -- Per-variant real-weight parity tests. Each is `#[ignore]`d — opt in with `--ignored`. Dispatch
//    into the shared `run_parity`; select one variant by name, e.g.
//    `cargo test -p candle-gen-sd3 --test component_parity sd35_medium_component_parity -- --ignored`.

/// SD3.5-large parity (the original sc-9076 coverage).
#[test]
#[ignore = "real-weight parity: needs the SD3.5-large snapshot + the diffusers reference (see module header)"]
fn sd35_large_component_parity_vs_diffusers() {
    run_parity(&VARIANTS[0]);
}

/// SD3.5-medium parity — the **MMDiT-X** (dual-attention) variant (sc-9580, highest-value add).
#[test]
#[ignore = "real-weight parity: needs the SD3.5-medium snapshot + the diffusers reference (see module header)"]
fn sd35_medium_component_parity_vs_diffusers() {
    run_parity(&VARIANTS[1]);
}

/// SD3.5-large-turbo parity — the guidance-distilled Large sibling (sc-9580).
#[test]
#[ignore = "real-weight parity: needs the SD3.5-large-turbo snapshot + the diffusers reference (see module header)"]
fn sd35_large_turbo_component_parity_vs_diffusers() {
    run_parity(&VARIANTS[2]);
}

/// A tensor-shape sanity guard that runs WITHOUT weights (not `#[ignore]`d): for every [`VARIANTS`]
/// entry whose reference is present, confirms the manifest has the shapes the harness expects. Skips
/// gracefully when a reference has not been generated (so a fresh checkout's `cargo test` stays green).
/// This keeps the harness wiring honest even on a box without the SD3.5 weights, and guards each
/// committed golden pair.
#[test]
fn reference_manifests_are_consistent_if_present() {
    let mut checked = 0usize;
    for variant in VARIANTS {
        let tag = variant.tag;
        let manifest_path = reference_dir().join(format!("{tag}_manifest.json"));
        if !manifest_path.is_file() {
            eprintln!(
                "skip {tag}: no reference manifest at {} (run gen_reference.py to enable it)",
                manifest_path.display()
            );
            continue;
        }
        checked += 1;
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&manifest_path).unwrap()).unwrap();
        assert_eq!(v["tag"].as_str(), Some(tag), "{tag}: manifest tag mismatch");
        let t = &v["tensors"];
        // Aggregated conditioning shapes are shared across all SD3.5 variants (triple-TE 2048 pooled /
        // 4096 joint context).
        assert_eq!(
            t["pooled"].as_array().unwrap().len(),
            2,
            "{tag}: pooled is [B, 2048]"
        );
        assert_eq!(t["pooled"][1].as_u64().unwrap(), 2048, "{tag}: pooled 2048");
        assert_eq!(
            t["context"][2].as_u64().unwrap(),
            4096,
            "{tag}: context hidden = 4096"
        );
        assert_eq!(
            t["context"][1].as_u64().unwrap(),
            (77 + v["t5_len"].as_u64().unwrap()),
            "{tag}: context seq = 77 + t5_len"
        );
        // DiT velocity is [1, 16, H, W].
        assert_eq!(
            t["dit_velocity"][1].as_u64().unwrap(),
            16,
            "{tag}: 16 latent channels"
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
                "{tag}: tolerance cosine_min present for {name}"
            );
            assert!(
                v["tolerances"][name]["max_abs"].is_number(),
                "{tag}: tolerance max_abs present for {name}"
            );
        }
    }
    eprintln!(
        "reference manifest consistency: checked {checked}/{} variant(s)",
        VARIANTS.len()
    );
}
