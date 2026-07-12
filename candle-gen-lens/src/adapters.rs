//! Lens DiT inference-side adapter merge (sc-5116) — load a trained LoRA/LoKr `.safetensors` and fold
//! its delta into the dense `transformer/` weights **before** the [`crate::transformer::LensTransformer`]
//! is built. The candle twin of `mlx-gen-lens::adapters` (sc-3174), structurally identical to
//! `candle-gen-z-image::adapters` (sc-5166) re-homed onto the Lens DiT key namespace.
//!
//! **Fused QKV — no q/k/v split.** The Lens trainer's `DEFAULT_LORA_TARGET_MODULES` are
//! `img_qkv` / `txt_qkv` / `to_out.0` / `to_add_out` on `LensTransformer2DModel`. `img_qkv` / `txt_qkv`
//! are **single fused** `[3·inner, inner]` projections that the trainer targets as one module each, so
//! a LoRA/LoKr on them merges into the **whole** fused weight — there is no q/k/v split-then-apply seam
//! here (that BFL fused→split path is FLUX.2's, sc-2743). The by-key merge then handles all four
//! targets uniformly: each is a 2-D Linear weight at `{path}.weight`, exactly like Z-Image's
//! `to_q`/`to_k`/`to_v`/`to_out.0`.
//!
//! **Merge, don't residual** (same rationale as SDXL / Z-Image): inference has no need to keep the
//! factors trainable, so it folds `W += δ` into the dense weight and reproduces the merged-weight
//! forward exactly. The Lens flow-match sampler is chaos-sensitive — `(W+δ)·x` differs from the
//! residual `W·x + δ·x` by ~1 ULP, which cascades to a visibly different image. The delta is
//! reconstructed with the **same** f32 math the trainer's forward uses
//! ([`reconstruct_lora_delta`] / [`reconstruct_lokr_delta`]), so a candle-trained adapter round-trips.
//!
//! **Merge at the safetensors-key level.** The DiT reads its `transformer/` keys 1:1, so
//! `transformer_blocks.{i}.attn.{img_qkv,txt_qkv,to_out.0,to_add_out}.weight` is a valid base key for
//! every Linear an adapter targets. The DiT family writes **bare dotted** PEFT keys (no
//! `base_model.model.transformer.` prefix — that is SDXL-specific); on read we also tolerate the common
//! community prefixes ([`PEFT_PREFIXES`]) and a kohya `lora_transformer_<flat>` flattening resolved
//! against the base key set.
//!
//! LoRA/LoKr are **DiT-only** for Lens (the gpt-oss text encoder and Flux.2 VAE are not adapter
//! targets, matching the trainer). The same `LensTransformer` serves both `lens` and `lens_turbo`
//! (identical architecture), so a LoRA trained on base `microsoft/Lens` applies cleanly to `Lens-Turbo`.
//! Out-of-surface keys are **counted and surfaced** in [`MergeReport`], never silently dropped.

use std::collections::{BTreeMap, HashMap, HashSet};

use candle_gen::candle_core::{DType, Tensor};
use candle_gen::gen_core::weightsmeta as wmeta;
use candle_gen::gen_core::{AdapterKind, AdapterSpec};
use candle_gen::quant::LokrFactors;
use candle_gen::train::lora::{reconstruct_lokr_delta, reconstruct_lora_delta, LoraAdapterMeta};
// The shared adapter-merge skeleton (sc-8998 / F-018): the format-parsing + merge-report primitives
// this crate previously hand-copied. Only the DiT-specific key→module resolution stays local below.
use candle_gen::train::merge::{
    build_kohya_table, merge_into, no_target_matched, read_adapter, read_scalar, AdapterFile,
    LoraTriple, Role,
};
// Re-exported so `candle_gen_lens::MergeReport` (the crate's public surface) keeps resolving.
pub use candle_gen::train::merge::MergeReport;
use candle_gen::{CandleError, Result};

use crate::transformer::LensTransformer;

/// PEFT key prefixes tolerated on read, longest-first. The candle/torch Lens trainer writes **bare**
/// dotted paths (no prefix), but community adapters and `peft.save_pretrained()` wrap the DiT under
/// one of these; stripping them yields the same dotted module path. A key matching none is taken
/// as-is (bare).
const PEFT_PREFIXES: [&str; 4] = [
    "base_model.model.transformer.",
    "base_model.model.",
    "diffusion_model.",
    "transformer.",
];

/// kohya / community flattened-module LoRA prefix (the DiT analog of SDXL's `lora_unet_`). The
/// `_`-flattened stem is resolved against the base DiT key table (diffusers names contain `_`, so the
/// flattening is ambiguous without it).
const KOHYA_PREFIX: &str = "lora_transformer_";

/// LoKr per-module factor suffixes, longest-first so `.lokr_w1_a` wins over `.lokr_w1`.
const LOKR_SUFFIXES: [&str; 6] = [
    ".lokr_w1_a",
    ".lokr_w1_b",
    ".lokr_w1",
    ".lokr_w2_a",
    ".lokr_w2_b",
    ".lokr_w2",
];

/// Strip the longest matching PEFT prefix, or return the key unchanged (bare dotted path).
fn strip_peft_prefix(key: &str) -> &str {
    for p in PEFT_PREFIXES {
        if let Some(rem) = key.strip_prefix(p) {
            return rem;
        }
    }
    key
}

/// Map one LoRA key to `(dit_dotted_path, role)`, or `None` if outside the DiT merge surface. kohya
/// (`lora_transformer_<flat>…`) resolves the flattened stem via `table`; PEFT/bare resolve directly
/// after the optional prefix strip.
fn classify_lora_key(key: &str, table: &BTreeMap<String, String>) -> Option<(String, Role)> {
    if let Some(rem) = key.strip_prefix(KOHYA_PREFIX) {
        for (suf, role) in [
            (".lora_down.weight", Role::Down),
            (".lora_up.weight", Role::Up),
            (".alpha", Role::Alpha),
        ] {
            if let Some(stem) = rem.strip_suffix(suf) {
                return table.get(stem).map(|d| (d.clone(), role));
            }
        }
        return None;
    }
    let rem = strip_peft_prefix(key);
    for (suf, role) in [
        (".lora_A.default.weight", Role::Down),
        (".lora_B.default.weight", Role::Up),
        (".lora_A.weight", Role::Down),
        (".lora_B.weight", Role::Up),
        (".alpha", Role::Alpha),
    ] {
        if let Some(path) = rem.strip_suffix(suf) {
            return Some((path.to_string(), role));
        }
    }
    None
}

/// Map one LoKr factor key to `(dit_dotted_path, factor_name)`, or `None` if out of surface.
fn classify_lokr_key(
    key: &str,
    table: &BTreeMap<String, String>,
) -> Option<(String, &'static str)> {
    for suf in LOKR_SUFFIXES {
        if let Some(stem) = key.strip_suffix(suf) {
            let factor = &suf[1..]; // drop the leading '.'
            return if let Some(flat) = stem.strip_prefix(KOHYA_PREFIX) {
                table.get(flat).map(|d| (d.clone(), factor))
            } else {
                Some((strip_peft_prefix(stem).to_string(), factor))
            };
        }
    }
    None
}

/// Merge one LoRA file into `base` at `scale`: classify every key, fold complete `(down, up)` pairs
/// into `{path}.weight`. `rank` is `A`'s leading dim; `alpha` is the per-target `.alpha` tensor when
/// present, else the `lora_adapter_metadata` blob's `alpha_pattern`/`lora_alpha` (the diffusers / PEFT
/// `save_lora_adapter` format ships no `.alpha` tensor — sc-5374), else `rank`. Half-pairs and
/// conv-shaped (4-D) factors are surfaced as skipped.
fn merge_lora_file(
    base: &mut HashMap<String, Tensor>,
    af: &AdapterFile,
    scale: f32,
    table: &BTreeMap<String, String>,
    report: &mut MergeReport,
) -> Result<()> {
    let mut triples: BTreeMap<String, LoraTriple> = BTreeMap::new();
    for (key, t) in &af.tensors {
        match classify_lora_key(key, table) {
            Some((path, Role::Down)) => triples.entry(path).or_default().down = Some(t.clone()),
            Some((path, Role::Up)) => triples.entry(path).or_default().up = Some(t.clone()),
            Some((path, Role::Alpha)) => {
                triples.entry(path).or_default().alpha = Some(read_scalar(key, "alpha", t)?)
            }
            None => report.skipped_keys += 1,
        }
    }

    // sc-5374: diffusers / PEFT `save_lora_adapter` files ship no per-target `.alpha` tensor —
    // `lora_alpha`/`r` (+ per-module overrides) live in the `lora_adapter_metadata` header blob.
    // `None` for kohya / candle-trainer files (those carry a `.alpha` tensor, used exactly as before).
    let cfg = LoraAdapterMeta::from_file_metadata(&af.meta);
    for (path, t) in triples {
        let (Some(down), Some(up)) = (t.down, t.up) else {
            report.skipped_keys += 1; // half-pair (partner targeted a non-routable module)
            continue;
        };
        if down.dims().len() != 2 || up.dims().len() != 2 {
            report.skipped_keys += 1; // conv-shaped LoRA — out of surface
            continue;
        }
        let base_key = format!("{path}.weight");
        if !base.contains_key(&base_key) {
            report.skipped_keys += 1;
            continue;
        }
        // per-target `.alpha` tensor → `alpha_pattern`/`lora_alpha` blob → factor rank (last resort).
        let (cfg_alpha, cfg_rank) = cfg.as_ref().map_or((None, None), |c| c.effective(&path));
        let rank = cfg_rank.unwrap_or(down.dims()[0] as f32);
        let alpha = t.alpha.or(cfg_alpha).unwrap_or(rank);
        let delta = reconstruct_lora_delta(&down, &up, alpha, rank, scale)?;
        merge_into(base, &base_key, &delta, report)?;
    }
    Ok(())
}

/// Merge one LoKr file into `base` at `scale`: `rank`/`alpha` from file metadata (alpha defaults to
/// rank), per-module factors grouped, `δ = (alpha/rank)·kron(w1,w2)·scale` reconstructed and merged.
fn merge_lokr_file(
    base: &mut HashMap<String, Tensor>,
    af: &AdapterFile,
    scale: f32,
    table: &BTreeMap<String, String>,
    report: &mut MergeReport,
) -> Result<()> {
    let rank = af
        .meta
        .get("rank")
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(1.0);
    let alpha = af
        .meta
        .get("alpha")
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(rank);

    let mut grouped: BTreeMap<String, BTreeMap<&'static str, Tensor>> = BTreeMap::new();
    for (key, t) in &af.tensors {
        match classify_lokr_key(key, table) {
            Some((path, factor)) => {
                grouped.entry(path).or_default().insert(factor, t.clone());
            }
            None => report.skipped_keys += 1,
        }
    }

    for (path, f) in grouped {
        let base_key = format!("{path}.weight");
        let Some(w) = base.get(&base_key) else {
            report.skipped_keys += 1;
            continue;
        };
        if w.dims().len() != 2 {
            report.skipped_keys += 1; // conv LoKr — out of surface
            continue;
        }
        let (out_f, in_f) = (w.dims()[0], w.dims()[1]);
        let delta = reconstruct_lokr_delta(
            f.get("lokr_w1"),
            f.get("lokr_w1_a"),
            f.get("lokr_w1_b"),
            f.get("lokr_w2"),
            f.get("lokr_w2_a"),
            f.get("lokr_w2_b"),
            alpha,
            rank,
            scale,
            (out_f, in_f),
        )?;
        merge_into(base, &base_key, &delta, report)?;
    }
    Ok(())
}

/// Fold every adapter spec in `specs` into the base DiT tensor `map` (CPU, native dtype) at each
/// spec's `scale` — LoRA and LoKr, merged into the dense weights (`W += δ`). Returns the
/// [`MergeReport`]; errors if a non-empty spec list matches **no** target (a format / prefix
/// misconfiguration — the worker should then fall back rather than render an unadapted image silently).
pub fn merge_adapters(
    map: &mut HashMap<String, Tensor>,
    specs: &[AdapterSpec],
) -> Result<MergeReport> {
    if specs.is_empty() {
        return Ok(MergeReport::default());
    }
    let table = build_kohya_table(map, &[2]);
    let mut report = MergeReport::default();
    for spec in specs {
        let af = read_adapter(&spec.path)?;
        match spec.kind {
            AdapterKind::Lokr => merge_lokr_file(map, &af, spec.scale, &table, &mut report)?,
            AdapterKind::Lora => {
                // The file metadata is authoritative — a Lora-declared LoKr file has no lora_A/B keys
                // and would merge nothing; surface the mismatch loudly rather than no-op.
                if af.declares_lokr() {
                    return Err(CandleError::Msg(format!(
                        "lens: adapter {} declared Lora but its metadata says networkType=lokr",
                        spec.path.display()
                    )));
                }
                merge_lora_file(map, &af, spec.scale, &table, &mut report)?;
            }
        }
    }
    if report.merged == 0 {
        return Err(no_target_matched(
            "lens",
            "expected bare/PEFT `<path>.lora_A/B.weight` (LoRA) or `<module>.lokr_w1/w2` with \
             networkType=lokr (LoKr) over the DiT attention projections (img_qkv / txt_qkv / \
             to_out.0 / to_add_out). Conv-layer / text-encoder adapters are out of surface",
            specs.len(),
        ));
    }
    Ok(report)
}

// ---- Forward-time additive (unmerged) install on a PACKED tier (sc-11105) ------------------------
//
// The packed q4/q8 Lens tier (`SceneWorks/lens-mlx` / `lens-turbo-mlx`) has **no dense `W`** to fold a
// delta into — the [`merge_adapters`] path above (`W += δ`) errors on u32 codes, which is why a user LoRA
// on a packed tier used to force a full dense build. Instead, [`install_additive`] attaches each LoRA/LoKr
// as a **forward-time residual** on the DiT's shared [`candle_gen::quant::AdaptLinear`] projections:
// `y = base(x) + Σ scale·((x·A)·B)`, the base kept packed. So a user adapter applies on the q4/q8 tier at
// the base's footprint. The dense tier keeps folding (bit-exact) via [`merge_adapters`]. This mirrors
// qwen-image-edit / z-image (sc-11091 / sc-11105); the resolver reuses this crate's exact
// [`classify_lora_key`]/[`classify_lokr_key`] + [`LoraAdapterMeta::effective`] so the additive residual
// equals the fold to f32 tolerance.

/// A resolved LoRA residual pending attachment: `a = downᵀ` `[in, rank]`, `b = upᵀ·(alpha/rank)`
/// `[rank, out]`, `scale` the user strength. Read on CPU; moved to the DiT device at push.
struct PendingLora {
    a: Tensor,
    b: Tensor,
    scale: f64,
}

/// A LoKr module's raw factors + the FULL `(alpha/rank)·strength` scale, pending the projection's
/// `[out, in]` to build the structured Kronecker factors ([`LokrFactors`], the vec-trick — never the
/// dense delta).
struct PendingLokr {
    w1: Option<Tensor>,
    w1_a: Option<Tensor>,
    w1_b: Option<Tensor>,
    w2: Option<Tensor>,
    w2_a: Option<Tensor>,
    w2_b: Option<Tensor>,
    scale: f64,
}

/// A report of a forward-time additive install (sc-11105) — the packed-tier analog of [`MergeReport`].
#[derive(Debug, Default)]
pub struct AdditiveReport {
    /// Projections that received a residual (one per `(path, file)` hit; multiple stack).
    pub applied: usize,
    /// Resolved target paths present in the adapter file(s) but absent from the DiT surface.
    pub skipped_targets: Vec<String>,
    /// Adapter-file keys outside the LoRA/LoKr surface, half-pairs, or shape-mismatched factors.
    pub skipped_keys: usize,
}

/// Resolve one LoRA file into per-path [`PendingLora`] (`a = downᵀ`, `b = upᵀ·ratio`). Mirrors
/// [`merge_lora_file`]'s classify + effective alpha/rank **exactly** (incl. the kohya `table`),
/// producing UNMERGED factors instead of a folded delta — so the additive residual equals the folded
/// delta to f32 tolerance.
fn resolve_lora_file(
    af: &AdapterFile,
    scale: f32,
    table: &BTreeMap<String, String>,
    pending: &mut BTreeMap<String, Vec<PendingLora>>,
    skipped_keys: &mut usize,
) -> Result<()> {
    let mut triples: BTreeMap<String, LoraTriple> = BTreeMap::new();
    for (key, t) in &af.tensors {
        match classify_lora_key(key, table) {
            Some((path, Role::Down)) => triples.entry(path).or_default().down = Some(t.clone()),
            Some((path, Role::Up)) => triples.entry(path).or_default().up = Some(t.clone()),
            Some((path, Role::Alpha)) => {
                triples.entry(path).or_default().alpha = Some(read_scalar(key, "alpha", t)?)
            }
            None => *skipped_keys += 1,
        }
    }
    let cfg = LoraAdapterMeta::from_file_metadata(&af.meta);
    for (path, t) in triples {
        let (Some(down), Some(up)) = (t.down, t.up) else {
            *skipped_keys += 1; // half-pair (partner targeted a non-routable module)
            continue;
        };
        if down.dims().len() != 2 || up.dims().len() != 2 {
            *skipped_keys += 1; // conv-shaped LoRA — out of surface
            continue;
        }
        let (cfg_alpha, cfg_rank) = cfg.as_ref().map_or((None, None), |c| c.effective(&path));
        let rank = cfg_rank.unwrap_or(down.dims()[0] as f32) as f64;
        if rank == 0.0 {
            *skipped_keys += 1;
            continue;
        }
        let alpha = t.alpha.or(cfg_alpha).unwrap_or(rank as f32) as f64;
        let ratio = alpha / rank;
        // a = downᵀ [in, rank]; b = upᵀ·ratio [rank, out]. f32, contiguous for the matmul.
        let a = down.to_dtype(DType::F32)?.t()?.contiguous()?;
        let b = (up.to_dtype(DType::F32)?.t()?.contiguous()? * ratio)?;
        pending.entry(path).or_default().push(PendingLora {
            a,
            b,
            scale: scale as f64,
        });
    }
    Ok(())
}

/// Resolve one LoKr file into per-path [`PendingLokr`] with the FULL `(alpha/rank)·scale` baked (the
/// structured residual carries no separate scale field — the two-conventions trap). Mirrors
/// [`merge_lokr_file`]'s rank/alpha read (file metadata, alpha defaults to rank) + the kohya `table`.
fn resolve_lokr_file(
    af: &AdapterFile,
    scale: f32,
    table: &BTreeMap<String, String>,
    pending: &mut BTreeMap<String, Vec<PendingLokr>>,
    skipped_keys: &mut usize,
) -> Result<()> {
    let rank = af
        .meta
        .get("rank")
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(1.0);
    let alpha = af
        .meta
        .get("alpha")
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(rank);
    let full = (alpha as f64 / rank as f64) * scale as f64;
    let mut grouped: BTreeMap<String, BTreeMap<&'static str, Tensor>> = BTreeMap::new();
    for (key, t) in &af.tensors {
        match classify_lokr_key(key, table) {
            Some((path, factor)) => {
                grouped.entry(path).or_default().insert(factor, t.clone());
            }
            None => *skipped_keys += 1,
        }
    }
    for (path, f) in grouped {
        pending.entry(path).or_default().push(PendingLokr {
            w1: f.get("lokr_w1").cloned(),
            w1_a: f.get("lokr_w1_a").cloned(),
            w1_b: f.get("lokr_w1_b").cloned(),
            w2: f.get("lokr_w2").cloned(),
            w2_a: f.get("lokr_w2_a").cloned(),
            w2_b: f.get("lokr_w2_b").cloned(),
            scale: full,
        });
    }
    Ok(())
}

/// Install `specs` as **forward-time additive residuals** on a **packed** DiT (sc-11105): resolve each
/// LoRA/LoKr file into unmerged factors, then walk the DiT once pushing residuals onto matched
/// projections — the base is never dequantized or folded, so a q4/q8 tier keeps its footprint while the
/// user adapter applies. A **LoHa** (no allocation-free structured form) and an **untagged third-party
/// LyCORIS** adapter are rejected with a pointer to the dense tier. Like [`merge_adapters`], a non-empty
/// spec set that matches **no** target errors (never renders unadapted).
pub fn install_additive(
    dit: &mut LensTransformer,
    specs: &[AdapterSpec],
) -> Result<AdditiveReport> {
    let mut report = AdditiveReport::default();

    // The kohya `flattened → dotted` table, built from the DiT's own projection paths (all Linear,
    // rank-2), so a `lora_transformer_<flat>` key resolves exactly as the dense fold's
    // `build_kohya_table` would (the additive-path analog with no base tensor map at hand).
    let mut paths: Vec<String> = Vec::new();
    dit.visit_adaptable_mut(&mut |path, _lin| {
        paths.push(path.to_string());
        Ok(())
    })?;
    let table: BTreeMap<String, String> = paths
        .iter()
        .map(|p| (p.replace('.', "_"), p.clone()))
        .collect();

    let mut pending_lora: BTreeMap<String, Vec<PendingLora>> = BTreeMap::new();
    let mut pending_lokr: BTreeMap<String, Vec<PendingLokr>> = BTreeMap::new();

    for spec in specs {
        let af = read_adapter(&spec.path)?;
        if wmeta::keys_contain_loha(af.tensors.keys().map(String::as_str)) {
            return Err(CandleError::Msg(format!(
                "lens: a LoHa adapter cannot apply on a packed (q4/q8) tier — its Hadamard product \
                 `(w1_a·w1_b) ⊙ (w2_a·w2_b)` has no allocation-free structured form (unlike LoKr's \
                 Kronecker vec-trick), so it would materialize a full [out,in] delta per target. Use a \
                 dense bf16 snapshot (where it folds into the weight) or a plain LoRA/LoKr. Offending \
                 file: {}",
                spec.path.display()
            )));
        }
        // Untagged LyCORIS LoKr (`lokr_*` keys, no `networkType=lokr` stamp, not declared) carries its
        // scale per-module in a way this additive resolver doesn't thread; reject on packed rather than
        // silently mis-scale (the dense tier folds it via `merge_adapters`).
        let untagged_lokr = !af.declares_lokr()
            && spec.kind != AdapterKind::Lokr
            && wmeta::keys_contain_lokr(af.tensors.keys().map(String::as_str));
        if untagged_lokr {
            return Err(CandleError::Msg(format!(
                "lens: an untagged third-party LyCORIS LoKr cannot apply additively on a packed (q4/q8) \
                 tier — use a dense bf16 snapshot (where `merge_adapters` folds it), a PEFT-stamped LoKr \
                 (`networkType=lokr`), or a plain LoRA. Offending file: {}",
                spec.path.display()
            )));
        }
        if spec.kind == AdapterKind::Lokr || af.declares_lokr() {
            resolve_lokr_file(
                &af,
                spec.scale,
                &table,
                &mut pending_lokr,
                &mut report.skipped_keys,
            )?;
        } else {
            resolve_lora_file(
                &af,
                spec.scale,
                &table,
                &mut pending_lora,
                &mut report.skipped_keys,
            )?;
        }
    }

    // Attach: walk the DiT once, pushing any resolved residual for each projection's canonical path. The
    // factors are read on the CPU but the base weight lives on the DiT device (CUDA on a packed tier),
    // so they are moved onto it at push. A factor whose dims don't match the projection is surfaced as a
    // skipped key, never a crashing forward (the additive analog of the fold path's shape guard).
    let device = dit.device().clone();
    let mut matched: HashSet<String> = HashSet::new();
    let mut applied = 0usize;
    let mut skipped_keys = 0usize;
    dit.visit_adaptable_mut(&mut |path, lin| {
        let (out_f, in_f) = lin.base_shape();
        if let Some(list) = pending_lora.get(path) {
            matched.insert(path.to_string());
            for p in list {
                if p.a.dims()[0] != in_f || p.b.dims()[1] != out_f {
                    skipped_keys += 1;
                    continue;
                }
                lin.push_lora(p.a.to_device(&device)?, p.b.to_device(&device)?, p.scale);
                applied += 1;
            }
        }
        if let Some(list) = pending_lokr.get(path) {
            matched.insert(path.to_string());
            for p in list {
                match LokrFactors::build(
                    p.scale,
                    (out_f, in_f),
                    p.w1.as_ref(),
                    p.w1_a.as_ref(),
                    p.w1_b.as_ref(),
                    p.w2.as_ref(),
                    None, // no tucker/CP `lokr_t2` on the peft LoKr surface
                    p.w2_a.as_ref(),
                    p.w2_b.as_ref(),
                )? {
                    Some(factors) => {
                        lin.push_lokr_structured(factors.to_device(&device)?);
                        applied += 1;
                    }
                    None => {
                        return Err(CandleError::Msg(format!(
                            "lens: LoKr target `{path}` is not deferrable on a packed tier (a tucker/CP \
                             `lokr_t2`, or a base that does not factor as a·b × c·d) — no \
                             allocation-free structured form. Use a dense bf16 snapshot."
                        )));
                    }
                }
            }
        }
        Ok(())
    })?;
    report.applied = applied;
    report.skipped_keys += skipped_keys;

    // Pending targets absent from the DiT surface are surfaced, never silently dropped.
    for path in pending_lora.keys().chain(pending_lokr.keys()) {
        if !matched.contains(path) {
            report.skipped_targets.push(path.clone());
        }
    }
    // A non-empty spec set that adapted nothing is a format/prefix misconfiguration — fail loudly rather
    // than render an unadapted image (the additive twin of `merge_adapters`' zero-match guard).
    if !specs.is_empty() && report.applied == 0 {
        return Err(no_target_matched(
            "lens",
            "expected bare/PEFT `<path>.lora_A/B.weight` (LoRA) or `<module>.lokr_w1/w2` with \
             networkType=lokr (LoKr) over the DiT attention projections (img_qkv / txt_qkv / to_out.0 / \
             to_add_out) + SwiGLU MLPs. Conv-layer / text-encoder adapters are out of surface",
            specs.len(),
        ));
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::{DType, Device};

    /// A tiny stand-in for the base Lens DiT tensor map: the fused `img_qkv` + the `to_out.0`
    /// projection + one conv (4-D) weight that a 2-D adapter must never touch.
    fn base_map() -> HashMap<String, Tensor> {
        let dev = Device::Cpu;
        let mut m = HashMap::new();
        // Fused QKV is [3·inner, inner] — a LoRA on it merges whole (no q/k/v split).
        m.insert(
            "transformer_blocks.0.attn.img_qkv.weight".into(),
            Tensor::zeros((12, 4), DType::BF16, &dev).unwrap(),
        );
        m.insert(
            "transformer_blocks.0.attn.to_out.0.weight".into(),
            Tensor::zeros((4, 4), DType::BF16, &dev).unwrap(),
        );
        // a conv weight (4-D) — must never be merged by a 2-D LoRA.
        m.insert(
            "vae_like_conv.weight".into(),
            Tensor::zeros((4, 4, 3, 3), DType::BF16, &dev).unwrap(),
        );
        m
    }

    fn t2(data: &[f32], r: usize, c: usize) -> Tensor {
        Tensor::from_vec(data.to_vec(), (r, c), &Device::Cpu).unwrap()
    }

    /// Bare dotted (the trainer's format), prefixed PEFT, and kohya flattened all resolve to the same
    /// dotted Lens DiT path — including the `to_out.0` ModuleList index.
    #[test]
    fn classify_lora_resolves_bare_peft_and_kohya() {
        let table = build_kohya_table(&base_map(), &[2]);
        // bare dotted (what the trainer writes for the DiT).
        let (p, _) =
            classify_lora_key("transformer_blocks.0.attn.img_qkv.lora_A.weight", &table).unwrap();
        assert_eq!(p, "transformer_blocks.0.attn.img_qkv");
        // PEFT-prefixed (community / peft.save_pretrained).
        let (p, r) = classify_lora_key(
            "transformer.transformer_blocks.0.attn.img_qkv.lora_B.weight",
            &table,
        )
        .unwrap();
        assert_eq!(p, "transformer_blocks.0.attn.img_qkv");
        assert!(matches!(r, Role::Up));
        // `.default.` infix.
        assert!(matches!(
            classify_lora_key(
                "base_model.model.transformer.transformer_blocks.0.attn.txt_qkv.lora_B.default.weight",
                &table,
            )
            .unwrap()
            .1,
            Role::Up
        ));
        // kohya flattened stem, incl. the `.0` of to_out.0 → `to_out_0`.
        let (p, _) = classify_lora_key(
            "lora_transformer_transformer_blocks_0_attn_to_out_0.lora_down.weight",
            &table,
        )
        .unwrap();
        assert_eq!(p, "transformer_blocks.0.attn.to_out.0");
        // text-encoder keys are out of surface.
        assert!(classify_lora_key(
            "lora_te1_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight",
            &table
        )
        .is_none());
    }

    /// Bare-dotted LoRA on the **fused** `img_qkv` merges into the whole `[3·inner, inner]` weight
    /// (`W += (alpha/rank)·scale·B·A`); base+delta is exact in f32 — the headline fused-QKV-no-split case.
    #[test]
    fn merge_lora_fused_qkv_folds_whole_weight() {
        let mut map = base_map();
        // A [rank=2, in=4], B [out=12, rank=2] — note out spans the full fused 3·inner.
        let down = t2(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], 2, 4);
        let mut up_data = vec![0f32; 24];
        up_data[0] = 2.0; // row 0 (a q row), rank-0
        up_data[2 * 2 + 1] = 3.0; // row 2 (another q row), rank-1
        up_data[8 * 2] = 5.0; // row 8 (a v row), rank-0 — proves the delta spans past q into k/v
        let up = t2(&up_data, 12, 2);
        let af = AdapterFile {
            tensors: HashMap::from([
                (
                    "transformer_blocks.0.attn.img_qkv.lora_A.weight".to_string(),
                    down.clone(),
                ),
                (
                    "transformer_blocks.0.attn.img_qkv.lora_B.weight".to_string(),
                    up.clone(),
                ),
                (
                    "transformer_blocks.0.attn.img_qkv.alpha".to_string(),
                    Tensor::from_vec(vec![4.0f32], (1,), &Device::Cpu).unwrap(),
                ),
            ]),
            meta: HashMap::new(),
        };
        let table = build_kohya_table(&map, &[2]);
        let mut report = MergeReport::default();
        // scale 1.0; alpha 4, rank 2 ⇒ effective 2.0. ΔW = 2.0·(B·A) over the full [12, 4] fused weight.
        merge_lora_file(&mut map, &af, 1.0, &table, &mut report).unwrap();
        assert_eq!(report.merged, 1);
        let merged = map
            .get("transformer_blocks.0.attn.img_qkv.weight")
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let expected = reconstruct_lora_delta(&down, &up, 4.0, 2.0, 1.0).unwrap(); // base is zero
        assert_eq!(merged.dims(), &[12, 4]);
        let diff = (merged - expected)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff < 1e-2, "merged fused weight off by {diff}"); // bf16 base round-trip tolerance
    }

    /// sc-5374: a diffusers-format LoRA with NO per-target `.alpha` tensor but a `lora_adapter_metadata`
    /// blob (`lora_alpha = 16`, `r = 8`) merges into the fused `img_qkv` at the metadata-derived
    /// `(16/8)·scale = 2.0`, not the old `alpha = rank` default. Base is zero ⇒ merged IS the delta.
    #[test]
    fn merge_lora_honors_lora_adapter_metadata_alpha() {
        let dev = Device::Cpu;
        let mut map = base_map();
        let path = "transformer_blocks.0.attn.img_qkv";
        let down = Tensor::randn(0f32, 1f32, (8, 4), &dev).unwrap(); // A [r=8, in=4]
        let up = Tensor::randn(0f32, 1f32, (12, 8), &dev).unwrap(); // B [out=12, r=8] (fused qkv)
        let af = AdapterFile {
            tensors: HashMap::from([
                (format!("{path}.lora_A.weight"), down.clone()),
                (format!("{path}.lora_B.weight"), up.clone()),
            ]),
            meta: HashMap::from([(
                "lora_adapter_metadata".to_string(),
                r#"{"lora_alpha": 16, "r": 8}"#.to_string(),
            )]),
        };
        let table = build_kohya_table(&map, &[2]);
        let mut report = MergeReport::default();
        merge_lora_file(&mut map, &af, 1.0, &table, &mut report).unwrap();
        assert_eq!(report.merged, 1);
        let merged = map
            .get(&format!("{path}.weight"))
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        assert_eq!(merged.dims(), &[12, 4]);
        let expected = reconstruct_lora_delta(&down, &up, 16.0, 8.0, 1.0).unwrap();
        let diff = (&merged - &expected)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff < 1e-4, "metadata-alpha merge off by {diff}");
        // The pre-sc-5374 default (alpha = rank ⇒ scale 1.0) would diverge by a factor of 2.
        let buggy = reconstruct_lora_delta(&down, &up, 8.0, 8.0, 1.0).unwrap();
        let gap = (&merged - &buggy)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            gap > 1e-3,
            "metadata alpha must differ from alpha=rank (gap {gap})"
        );
    }

    /// A conv-shaped LoRA (4-D factors) is surfaced as skipped, never merged into the conv weight.
    #[test]
    fn merge_skips_conv_shaped_lora() {
        let mut map = base_map();
        let dev = Device::Cpu;
        let af = AdapterFile {
            tensors: HashMap::from([
                (
                    "vae_like_conv.lora_A.weight".to_string(),
                    Tensor::zeros((2, 4, 3, 3), DType::F32, &dev).unwrap(),
                ),
                (
                    "vae_like_conv.lora_B.weight".to_string(),
                    Tensor::zeros((4, 2, 1, 1), DType::F32, &dev).unwrap(),
                ),
            ]),
            meta: HashMap::new(),
        };
        let table = build_kohya_table(&map, &[2]);
        let mut report = MergeReport::default();
        merge_lora_file(&mut map, &af, 1.0, &table, &mut report).unwrap();
        assert_eq!(report.merged, 0);
        assert_eq!(report.skipped_keys, 1); // the (down,up) pair, dropped as a conv shape
    }

    /// LoKr merges `δ = (alpha/rank)·kron(w1,w2)` into the fused dense weight, reading rank/alpha from
    /// meta. w1 [3,1] ⊗ w2 [4,4] → [12,4] = the `img_qkv` shape (whole-weight fused merge).
    #[test]
    fn merge_lokr_folds_kron_delta() {
        let mut map = base_map();
        let w1 = t2(&[1.0, 2.0, 0.5], 3, 1);
        let w2 = Tensor::from_vec(
            (0..16).map(|i| (i as f32) * 0.1).collect::<Vec<_>>(),
            (4, 4),
            &Device::Cpu,
        )
        .unwrap();
        let af = AdapterFile {
            tensors: HashMap::from([
                (
                    "transformer_blocks.0.attn.img_qkv.lokr_w1".to_string(),
                    w1.clone(),
                ),
                (
                    "transformer_blocks.0.attn.img_qkv.lokr_w2".to_string(),
                    w2.clone(),
                ),
            ]),
            meta: HashMap::from([
                ("networkType".to_string(), "lokr".to_string()),
                ("rank".to_string(), "2".to_string()),
                ("alpha".to_string(), "2".to_string()),
            ]),
        };
        let table = build_kohya_table(&map, &[2]);
        let mut report = MergeReport::default();
        merge_lokr_file(&mut map, &af, 1.0, &table, &mut report).unwrap();
        assert_eq!(report.merged, 1);
        let merged = map
            .get("transformer_blocks.0.attn.img_qkv.weight")
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let expected = reconstruct_lokr_delta(
            Some(&w1),
            None,
            None,
            Some(&w2),
            None,
            None,
            2.0,
            2.0,
            1.0,
            (12, 4),
        )
        .unwrap();
        let diff = (merged - expected)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff < 1e-2, "merged lokr weight off by {diff}");
    }

    /// The keystone train→infer round-trip: a PEFT `.safetensors` written by the **actual trainer**
    /// path ([`candle_gen::train::lora::save_lora_peft`] with the DiT's empty prefix) is read back
    /// through the public [`merge_adapters`] entry, and the merged weight equals the trained delta.
    #[test]
    fn roundtrip_trainer_peft_file_merges() {
        use candle_gen::candle_nn::Linear;
        use candle_gen::train::lora::{build_lora_targets, save_lora_peft, LoraHost, LoraLinear};

        struct Host(LoraLinear);
        impl LoraHost for Host {
            fn visit_lora_mut(
                &mut self,
                f: &mut dyn FnMut(&mut LoraLinear) -> candle_gen::Result<()>,
            ) -> candle_gen::Result<()> {
                f(&mut self.0)
            }
        }

        let dev = Device::Cpu;
        let path = "transformer_blocks.3.attn.to_add_out";
        let base_w = Tensor::zeros((4, 4), DType::F32, &dev).unwrap();
        let mut host = Host(LoraLinear::from_linear(
            Linear::new(base_w, None),
            4,
            4,
            path.into(),
        ));

        // rank 2, alpha 4 ⇒ effective 2.0. Force B (vars[1]) nonzero so ΔW ≠ 0 (zero-init B no-ops).
        let set =
            build_lora_targets(&mut host, &["to_add_out".to_string()], 2, 4.0, 7, &dev).unwrap();
        let up_randn = Tensor::randn(0f32, 1f32, (4, 2), &dev).unwrap();
        set.vars[1].set(&up_randn).unwrap(); // vars = [down(A), up(B)]

        // Write the real PEFT file the DiT trainer emits (empty prefix → bare dotted keys), then
        // merge it through the public entry point.
        let file = std::env::temp_dir().join(format!(
            "candle_lens_lora_roundtrip_{}.safetensors",
            std::process::id()
        ));
        save_lora_peft(&set, "", &HashMap::new(), &file).unwrap();

        let mut map = HashMap::new();
        map.insert(
            format!("{path}.weight"),
            Tensor::zeros((4, 4), DType::BF16, &dev).unwrap(),
        );
        let report = merge_adapters(
            &mut map,
            &[AdapterSpec::new(file.clone(), 1.0, AdapterKind::Lora)],
        );
        std::fs::remove_file(&file).ok();
        let report = report.unwrap();

        assert_eq!(
            report.merged, 1,
            "the trained to_add_out adapter must merge"
        );
        let expected = reconstruct_lora_delta(
            set.vars[0].as_tensor(),
            set.vars[1].as_tensor(),
            4.0,
            2.0,
            1.0,
        )
        .unwrap();
        let merged = map[&format!("{path}.weight")].to_dtype(DType::F32).unwrap();
        let diff = (&merged - &expected)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            diff < 1e-2,
            "round-trip merge diverged from the trained delta by {diff}"
        );
        let mag = expected
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(mag > 0.0, "forced-nonzero B must yield a non-trivial delta");
    }

    /// A non-empty spec list that matches nothing is a loud error (not a silent unadapted render).
    #[test]
    fn merge_lora_file_unresolvable_key_merges_nothing() {
        let mut map = base_map();
        let af = AdapterFile {
            tensors: HashMap::from([(
                "lora_transformer_nonexistent_module.lora_down.weight".to_string(),
                t2(&[0.0, 0.0], 1, 2),
            )]),
            meta: HashMap::new(),
        };
        let table = build_kohya_table(&map, &[2]);
        let mut report = MergeReport::default();
        merge_lora_file(&mut map, &af, 1.0, &table, &mut report).unwrap();
        assert_eq!(report.merged, 0);
        assert!(report.skipped_keys >= 1);
    }

    /// **Additive == folded parity at the resolver level (sc-11105).** The unmerged factors
    /// [`resolve_lora_file`] produces reproduce the folded `x·(W + δ)ᵀ` on a dense
    /// [`candle_gen::quant::AdaptLinear`] base — using this crate's exact [`classify_lora_key`] +
    /// effective alpha/rank — so the packed additive path and the dense fold path agree to f32 tolerance
    /// (the parity the ~1-ULP Lens sampler sensitivity needs).
    #[test]
    fn resolve_lora_matches_fold_on_dense() {
        use candle_gen::candle_nn::Linear;
        use candle_gen::quant::AdaptLinear;
        let dev = Device::Cpu;
        let (out_dim, in_dim, rank) = (16usize, 12usize, 3usize);
        let path = "transformer_blocks.0.attn.to_out.0";
        let down = Tensor::randn(0f32, 1f32, (rank, in_dim), &dev).unwrap();
        let up = Tensor::randn(0f32, 1f32, (out_dim, rank), &dev).unwrap();
        let (alpha, scale) = (6.0f32, 0.8f32); // ratio = alpha/rank = 2.0
        let af = AdapterFile {
            tensors: HashMap::from([
                (format!("{path}.lora_A.weight"), down.clone()),
                (format!("{path}.lora_B.weight"), up.clone()),
                (
                    format!("{path}.alpha"),
                    Tensor::from_vec(vec![alpha], (1,), &dev).unwrap(),
                ),
            ]),
            meta: HashMap::new(),
        };
        let table: BTreeMap<String, String> = BTreeMap::new();
        let mut pending: BTreeMap<String, Vec<PendingLora>> = BTreeMap::new();
        let mut skipped = 0usize;
        resolve_lora_file(&af, scale, &table, &mut pending, &mut skipped).unwrap();
        assert_eq!(skipped, 0, "clean LoRA resolves with no skipped keys");
        let p = &pending[path][0];
        assert_eq!(p.a.dims(), &[in_dim, rank], "a = downᵀ [in, rank]");
        assert_eq!(p.b.dims(), &[rank, out_dim], "b = upᵀ·ratio [rank, out]");

        let w = Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev).unwrap();
        let mut additive = AdaptLinear::from_dense(Linear::new(w.clone(), None), in_dim, out_dim);
        additive.push_lora(p.a.clone(), p.b.clone(), p.scale);
        let delta = reconstruct_lora_delta(&down, &up, alpha, rank as f32, scale).unwrap();
        let folded =
            AdaptLinear::from_dense(Linear::new((w + delta).unwrap(), None), in_dim, out_dim);

        let x = Tensor::randn(0f32, 1f32, (2usize, in_dim), &dev).unwrap();
        let d = (additive.forward(&x).unwrap() - folded.forward(&x).unwrap())
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(d < 1e-4, "resolved additive != folded (max diff {d})");
    }

    /// A **LoHa** file (`hada_*`) and an untagged third-party **LyCORIS LoKr** (`lokr_*`, no
    /// `networkType` stamp) are detected by the key sniffers [`install_additive`] rejects on a packed
    /// tier. sc-11105.
    #[test]
    fn loha_and_untagged_lokr_are_detected() {
        assert!(wmeta::keys_contain_loha(
            ["transformer_blocks.0.attn.to_out.0.hada_w1_a"].into_iter()
        ));
        assert!(wmeta::keys_contain_lokr(
            ["transformer_blocks.0.attn.to_out.0.lokr_w1"].into_iter()
        ));
    }

    /// `declares_lokr` reads the `networkType` metadata that the `Lora`-over-LoKr guard keys off.
    #[test]
    fn declares_lokr_reads_network_type() {
        let lokr = AdapterFile {
            tensors: HashMap::new(),
            meta: HashMap::from([("networkType".to_string(), "lokr".to_string())]),
        };
        let lora = AdapterFile {
            tensors: HashMap::new(),
            meta: HashMap::new(),
        };
        assert!(lokr.declares_lokr());
        assert!(!lora.declares_lokr());
    }
}
