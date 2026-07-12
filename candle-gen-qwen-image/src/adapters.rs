//! Qwen-Image-Edit inference-side adapter merge (sc-6220, epic 5480) — fold a LoRA/LoKr
//! `.safetensors` delta into the dense MMDiT transformer weights **before** [`crate::transformer::
//! QwenTransformer`] is built. The candle twin of `mlx-gen-qwen-image`'s adapter consumption (the
//! `AdaptableHost for QwenTransformer` module map) realized in the by-key-merge style the candle
//! `candle-gen-sdxl::adapters` already uses.
//!
//! **Primary consumer:** the **Qwen-Image-Edit-2511-Lightning** few-step distill (lightx2v) — a LoRA
//! over the per-block joint-attention + stream-MLP linears, merged at scale 1.0 so the 4-step
//! lightning schedule produces a clean edit. General Qwen-family LoRA/LoKr ride the same path.
//!
//! **Merge, don't residual** (same rationale as the SDXL merge): the flow-match Euler denoise is
//! precision-sensitive, so folding the delta into the dense weight (`W += δ`) reproduces the merged
//! forward `(W+δ)·x` exactly, with no per-step residual op. The delta is reconstructed with the same
//! f32 math the trainer's forward uses ([`reconstruct_lora_delta`] / [`reconstruct_lokr_delta`] /
//! [`reconstruct_loha_delta`]), so a candle-trained adapter round-trips.
//!
//! **Merge at the safetensors-key level.** The candle MMDiT reads the diffusers transformer keys 1:1
//! (`transformer_blocks.{i}.attn.to_q.weight`, `…img_mlp.net.0.proj.weight`, …), so a LoRA's
//! prefix-stripped dotted module path resolves `{path}.weight` directly — no per-module routing table.
//! Formats resolved (the Qwen-family conventions; `gen-core`'s [`wmeta::COMMON_LORA_PREFIXES`]):
//!  - **PEFT / diffusers / bare LoRA** — `‹prefix›‹path›.lora_A/B[.default].weight` **or**
//!    `‹prefix›‹path›.lora_down/up.weight` (+ optional `‹path›.alpha`), where `‹prefix›` is
//!    `transformer.` / `diffusion_model.` / none. `lora_down`==`lora_A`, `lora_up`==`lora_B`. The
//!    lightx2v Lightning LoRA is the **bare-path + down/up + per-module-`.alpha`** form. The scaling is
//!    the per-target `.alpha` tensor (kohya / candle-trainer) or — when absent — `lora_alpha`/`r`
//!    (+ `alpha_pattern`/`rank_pattern`) in the `lora_adapter_metadata` blob (diffusers
//!    `save_lora_adapter`), else `rank`.
//!  - **LoKr** — PEFT-stamped `‹path›.lokr_w1`/`lokr_w2` (+ low-rank `_a`/`_b`) with `networkType=lokr`
//!    and `rank`/`alpha` in file metadata, reconstructing `δ = (alpha/rank)·kron(w1,w2)`.
//!  - **Third-party LyCORIS** — untagged `lokr_*` / `hada_*` (no `networkType` stamp), reconstructed
//!    per-module at the lycoris scale.
//!
//! **Linear-only.** Every Qwen MMDiT target is a `Linear` (the model has no conv layers), so — unlike
//! the SDXL merge — there is no conv-LoRA surface; a non-2-D factor or a factor that resolves to no
//! module is surfaced in [`MergeReport`], never silently dropped.

use std::collections::{BTreeMap, HashMap, HashSet};

use candle_gen::candle_core::{DType, Tensor};
use candle_gen::gen_core::weightsmeta as wmeta;
use candle_gen::gen_core::{AdapterKind, AdapterSpec};
use candle_gen::quant::LokrFactors;
use candle_gen::train::lora::{reconstruct_lokr_delta, reconstruct_lora_delta, LoraAdapterMeta};

use crate::transformer::QwenTransformer;
// The shared adapter-merge skeleton (sc-8998 / F-018): the format-parsing + merge-report + third-party
// LyCORIS engine this crate previously hand-copied. Only the Qwen-specific key→module resolution
// (bare/prefixed dotted paths, no kohya table) stays local below.
use candle_gen::train::merge::{
    merge_into, merge_one_thirdparty, no_target_matched, parse_loha_thirdparty,
    parse_lokr_thirdparty, read_adapter, read_scalar, AdapterFile, LoraTriple, Role,
};
// Re-exported so `candle_gen_qwen_image::MergeReport` (the crate's public surface) keeps resolving.
pub use candle_gen::train::merge::MergeReport;
use candle_gen::{CandleError, Result};

/// LoKr per-module factor suffixes, longest-first so `.lokr_w1_a` wins over `.lokr_w1`.
const LOKR_SUFFIXES: [&str; 6] = [
    ".lokr_w1_a",
    ".lokr_w1_b",
    ".lokr_w1",
    ".lokr_w2_a",
    ".lokr_w2_b",
    ".lokr_w2",
];

/// Strip a leading Qwen LoRA namespace prefix (`transformer.` / `diffusion_model.`), if present —
/// leaving the bare diffusers module path that resolves directly against the base transformer keys.
/// A bare key (the lightx2v Lightning convention) and a LoKr factor key (always bare) pass through.
fn strip_lora_prefix(key: &str) -> &str {
    for p in wmeta::COMMON_LORA_PREFIXES {
        if let Some(rest) = key.strip_prefix(p) {
            return rest;
        }
    }
    key
}

/// Map one LoRA key to `(module_path, role)`, or `None` if outside the merge surface. Strips the
/// optional namespace prefix, then matches both the PEFT (`lora_A`/`lora_B`, optional `.default.`
/// infix) and the diffusers/kohya (`lora_down`/`lora_up`) factor namings, plus the per-module
/// `.alpha` (which is often bare even when the factors are prefixed).
fn classify_lora_key(key: &str) -> Option<(String, Role)> {
    let rem = strip_lora_prefix(key);
    for (suf, role) in [
        (".lora_A.default.weight", Role::Down),
        (".lora_B.default.weight", Role::Up),
        (".lora_A.weight", Role::Down),
        (".lora_B.weight", Role::Up),
        (".lora_down.weight", Role::Down),
        (".lora_up.weight", Role::Up),
        (".alpha", Role::Alpha),
    ] {
        if let Some(path) = rem.strip_suffix(suf) {
            return Some((path.to_string(), role));
        }
    }
    None
}

/// Map one (PEFT-stamped) LoKr factor key to `(module_path, factor_name)`, or `None` if out of
/// surface. Strips the optional namespace prefix; the factor name keeps its leading `.` dropped.
fn classify_lokr_key(key: &str) -> Option<(String, &'static str)> {
    for suf in LOKR_SUFFIXES {
        if let Some(stem) = key.strip_suffix(suf) {
            let factor = &suf[1..]; // drop the leading '.'
            return Some((strip_lora_prefix(stem).to_string(), factor));
        }
    }
    None
}

/// Merge one LoRA file into `base` at `scale`: classify every key, fold complete `(down, up)` pairs
/// into `{path}.weight`. `rank` is `A`'s leading dim; `alpha` is the per-target `.alpha` tensor when
/// present, else the `lora_adapter_metadata` blob's `alpha_pattern`/`lora_alpha`, else `rank`.
/// Linear-only (the Qwen MMDiT has no convs): a non-2-D pair, a half-pair, or an unresolved module is
/// surfaced as skipped.
fn merge_lora_file(
    base: &mut HashMap<String, Tensor>,
    af: &AdapterFile,
    scale: f32,
    report: &mut MergeReport,
) -> Result<()> {
    let mut triples: BTreeMap<String, LoraTriple> = BTreeMap::new();
    for (key, t) in &af.tensors {
        match classify_lora_key(key) {
            Some((path, Role::Down)) => triples.entry(path).or_default().down = Some(t.clone()),
            Some((path, Role::Up)) => triples.entry(path).or_default().up = Some(t.clone()),
            Some((path, Role::Alpha)) => {
                triples.entry(path).or_default().alpha = Some(read_scalar(key, t)?)
            }
            None => report.skipped_keys += 1,
        }
    }

    // PEFT/diffusers `save_lora_adapter` files carry no per-target `.alpha` tensor — `lora_alpha`/`r`
    // (+ per-module overrides) live in the `lora_adapter_metadata` blob. `None` for kohya / candle-
    // trainer / lightx2v files (those ship a `.alpha` tensor), in which case the per-target `.alpha`
    // or the factor rank is used.
    let cfg = LoraAdapterMeta::from_file_metadata(&af.meta);
    for (path, t) in triples {
        let (Some(down), Some(up)) = (t.down, t.up) else {
            report.skipped_keys += 1; // half-pair (partner targeted a non-routable module)
            continue;
        };
        if down.dims().len() != 2 || up.dims().len() != 2 {
            report.skipped_keys += 1; // Linear-only surface (the MMDiT has no conv weights)
            continue;
        }
        let base_key = format!("{path}.weight");
        if !base.contains_key(&base_key) {
            report.skipped_keys += 1;
            continue;
        }
        // Effective scaling: per-target `.alpha` tensor → `alpha_pattern`/`lora_alpha` blob → factor
        // rank. The denominator is the blob `r`/`rank_pattern` when given, else `A`'s leading dim.
        let (cfg_alpha, cfg_rank) = cfg.as_ref().map_or((None, None), |c| c.effective(&path));
        let rank = cfg_rank.unwrap_or(down.dims()[0] as f32);
        let alpha = t.alpha.or(cfg_alpha).unwrap_or(rank);
        let delta = reconstruct_lora_delta(&down, &up, alpha, rank, scale)?;
        merge_into(base, &base_key, &delta, report)?;
    }
    Ok(())
}

/// Merge one (PEFT-stamped) LoKr file into `base` at `scale`: `rank`/`alpha` from file metadata
/// (alpha defaults to rank), per-module factors grouped, `δ = (alpha/rank)·kron(w1,w2)·scale`
/// reconstructed and merged. Linear-only.
fn merge_lokr_file(
    base: &mut HashMap<String, Tensor>,
    af: &AdapterFile,
    scale: f32,
    report: &mut MergeReport,
) -> Result<()> {
    let (rank, alpha) = wmeta::parse_rank_alpha(
        af.meta.get("rank").map(String::as_str),
        af.meta.get("alpha").map(String::as_str),
    );

    let mut grouped: BTreeMap<String, BTreeMap<&'static str, Tensor>> = BTreeMap::new();
    for (key, t) in &af.tensors {
        match classify_lokr_key(key) {
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
            report.skipped_keys += 1; // Linear-only surface
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

/// Merge a third-party LyCORIS **LoKr** file (`lokr_*` keys, per-module `.alpha`, no `networkType`
/// stamp) into `base` at `scale`, via the shared [`parse_lokr_thirdparty`] +
/// [`merge_one_thirdparty`]. Each raw key resolves by prefix-strip (Qwen has no kohya table); the
/// per-module lycoris scale is baked into the delta closure. Linear-only.
fn merge_lokr_thirdparty(
    base: &mut HashMap<String, Tensor>,
    af: &AdapterFile,
    scale: f32,
    report: &mut MergeReport,
) -> Result<()> {
    for (raw, g) in parse_lokr_thirdparty(af)? {
        merge_one_thirdparty(
            base,
            Some(strip_lora_prefix(&raw)),
            |bs| g.delta(bs, scale),
            report,
        )?;
    }
    Ok(())
}

/// Merge a third-party LyCORIS **LoHa** file (`hada_*` keys) into `base` at `scale`.
fn merge_loha_thirdparty(
    base: &mut HashMap<String, Tensor>,
    af: &AdapterFile,
    scale: f32,
    report: &mut MergeReport,
) -> Result<()> {
    for (raw, g) in parse_loha_thirdparty(af)? {
        merge_one_thirdparty(
            base,
            Some(strip_lora_prefix(&raw)),
            |bs| g.delta(bs, scale),
            report,
        )?;
    }
    Ok(())
}

/// Fold every adapter spec in `specs` into the base MMDiT tensor `map` (CPU, native dtype) at each
/// spec's `scale` — LoRA and LoKr, merged into the dense weights (`W += δ`). Returns the
/// [`MergeReport`]; errors if a non-empty spec list matches **no** target (a format / prefix
/// misconfiguration — the worker should then fail rather than render an unadapted image silently).
pub fn merge_adapters(
    map: &mut HashMap<String, Tensor>,
    specs: &[AdapterSpec],
) -> Result<MergeReport> {
    if specs.is_empty() {
        return Ok(MergeReport::default());
    }
    let mut report = MergeReport::default();
    for spec in specs {
        let af = read_adapter(&spec.path)?;
        // Untagged LyCORIS: `lokr_*` / `hada_*` keys without a `networkType=lokr` stamp, so the
        // caller's declared `kind` can't label them — detect + route by keys before the kind match.
        if !af.declares_lokr() && wmeta::keys_contain_lokr(af.tensors.keys().map(String::as_str)) {
            merge_lokr_thirdparty(map, &af, spec.scale, &mut report)?;
            continue;
        }
        if wmeta::keys_contain_loha(af.tensors.keys().map(String::as_str)) {
            merge_loha_thirdparty(map, &af, spec.scale, &mut report)?;
            continue;
        }
        match spec.kind {
            AdapterKind::Lokr => merge_lokr_file(map, &af, spec.scale, &mut report)?,
            AdapterKind::Lora => {
                // The file metadata is authoritative — a Lora-declared LoKr file has no lora_A/B keys
                // and would merge nothing; surface the mismatch loudly rather than no-op.
                if af.declares_lokr() {
                    return Err(CandleError::Msg(format!(
                        "qwen edit: adapter {} declared Lora but its metadata says networkType=lokr",
                        spec.path.display()
                    )));
                }
                merge_lora_file(map, &af, spec.scale, &mut report)?;
            }
        }
    }
    if report.merged == 0 {
        return Err(no_target_matched(
            "qwen edit",
            "expected diffusers/PEFT `‹transformer.|diffusion_model.›‹path›.lora_A/B|lora_down/up.\
             weight` (+ optional `.alpha`) over the MMDiT `transformer_blocks.{i}.{attn.*,img_mlp.*,\
             txt_mlp.*}` modules, `‹path›.lokr_w1/w2` with networkType=lokr (LoKr), or untagged \
             LyCORIS `lokr_*` / `hada_*`",
            specs.len(),
        ));
    }
    Ok(report)
}

// ---- Forward-time additive (unmerged) install on a PACKED tier (sc-11091) ------------------------
//
// The packed q4/q8 Edit tier (`SceneWorks/qwen-image-edit-2511-mlx`) has **no dense `W`** to fold a
// delta into — the `merge_adapters` path above `W += δ` errors on u32 codes. Instead, [`install_additive`]
// attaches each LoRA/LoKr as a **forward-time residual** on the shared [`crate::quant::QLinear`]
// (`= candle_gen::quant::AdaptLinear`): `y = base(x) + Σ scale·((x·A)·B)`, the base kept packed. So the
// Qwen-Image-Edit-2511-Lightning distill (all 720 attn+MLP Linears) applies on the q4/q8 tier at the
// base's footprint, not a dense reload. The dense tier keeps folding (bit-exact) via `merge_adapters`.

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

/// A report of a forward-time additive install (sc-11091) — the packed-tier analog of [`MergeReport`].
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
/// [`merge_lora_file`]'s classify + effective alpha/rank **exactly**, producing UNMERGED factors
/// instead of a folded delta — so the additive residual equals the folded delta to f32 tolerance.
fn resolve_lora_file(
    af: &AdapterFile,
    scale: f32,
    pending: &mut BTreeMap<String, Vec<PendingLora>>,
    skipped_keys: &mut usize,
) -> Result<()> {
    let mut triples: BTreeMap<String, LoraTriple> = BTreeMap::new();
    for (key, t) in &af.tensors {
        match classify_lora_key(key) {
            Some((path, Role::Down)) => triples.entry(path).or_default().down = Some(t.clone()),
            Some((path, Role::Up)) => triples.entry(path).or_default().up = Some(t.clone()),
            Some((path, Role::Alpha)) => {
                triples.entry(path).or_default().alpha = Some(read_scalar(key, t)?)
            }
            None => *skipped_keys += 1,
        }
    }
    let cfg = LoraAdapterMeta::from_file_metadata(&af.meta);
    for (path, t) in triples {
        let (Some(down), Some(up)) = (t.down, t.up) else {
            *skipped_keys += 1; // half-pair
            continue;
        };
        if down.dims().len() != 2 || up.dims().len() != 2 {
            *skipped_keys += 1; // Linear-only surface (the MMDiT has no conv weights)
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

/// Resolve one (PEFT-stamped) LoKr file into per-path [`PendingLokr`] with the FULL `(alpha/rank)·scale`
/// baked (the structured residual carries no separate scale field — the two-conventions trap). Mirrors
/// [`merge_lokr_file`]'s rank/alpha; the factors stay small until built against the projection shape.
fn resolve_lokr_file(
    af: &AdapterFile,
    scale: f32,
    pending: &mut BTreeMap<String, Vec<PendingLokr>>,
    skipped_keys: &mut usize,
) -> Result<()> {
    let (rank, alpha) = wmeta::parse_rank_alpha(
        af.meta.get("rank").map(String::as_str),
        af.meta.get("alpha").map(String::as_str),
    );
    let full = (alpha as f64 / rank as f64) * scale as f64;
    let mut grouped: BTreeMap<String, BTreeMap<&'static str, Tensor>> = BTreeMap::new();
    for (key, t) in &af.tensors {
        match classify_lokr_key(key) {
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

/// Install `specs` as **forward-time additive residuals** on a **packed** DiT (sc-11091): resolve each
/// LoRA/LoKr file into unmerged factors, then walk the DiT once pushing residuals onto matched
/// projections — the base is never dequantized or folded, so a q4/q8 tier keeps its footprint while the
/// Lightning distill (or a user LoRA) applies. A **LoHa** (no allocation-free structured form) and an
/// **untagged third-party LyCORIS** adapter are rejected with a pointer to the dense tier. Like
/// [`merge_adapters`], a non-empty spec set that matches **no** target errors (never renders unadapted).
pub fn install_additive(
    dit: &mut QwenTransformer,
    specs: &[AdapterSpec],
) -> Result<AdditiveReport> {
    let mut pending_lora: BTreeMap<String, Vec<PendingLora>> = BTreeMap::new();
    let mut pending_lokr: BTreeMap<String, Vec<PendingLokr>> = BTreeMap::new();
    let mut report = AdditiveReport::default();

    for spec in specs {
        let af = read_adapter(&spec.path)?;
        if wmeta::keys_contain_loha(af.tensors.keys().map(String::as_str)) {
            return Err(CandleError::Msg(format!(
                "qwen edit: a LoHa adapter cannot apply on a packed (q4/q8) Edit tier — its Hadamard \
                 product `(w1_a·w1_b) ⊙ (w2_a·w2_b)` has no allocation-free structured form (unlike \
                 LoKr's Kronecker vec-trick), so it would materialize a full [out,in] delta per target. \
                 Use the dense (bf16) tier (where it folds into the weight) or a plain LoRA/LoKr. \
                 sc-10051. Offending file: {}",
                spec.path.display()
            )));
        }
        // Untagged LyCORIS LoKr (`lokr_*` keys, no `networkType=lokr` stamp, not declared) carries its
        // scale per-module in a way this additive resolver doesn't thread; reject on packed rather than
        // silently mis-scale (the dense tier folds it via `merge_lokr_thirdparty`).
        let untagged_lokr = !af.declares_lokr()
            && spec.kind != AdapterKind::Lokr
            && wmeta::keys_contain_lokr(af.tensors.keys().map(String::as_str));
        if untagged_lokr {
            return Err(CandleError::Msg(format!(
                "qwen edit: an untagged third-party LyCORIS LoKr cannot apply additively on a packed \
                 (q4/q8) Edit tier — use the dense (bf16) tier (where `merge_adapters` folds it), a \
                 PEFT-stamped LoKr (`networkType=lokr`), or a plain LoRA. Offending file: {}",
                spec.path.display()
            )));
        }
        if spec.kind == AdapterKind::Lokr || af.declares_lokr() {
            resolve_lokr_file(&af, spec.scale, &mut pending_lokr, &mut report.skipped_keys)?;
        } else {
            resolve_lora_file(&af, spec.scale, &mut pending_lora, &mut report.skipped_keys)?;
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
                            "qwen edit: LoKr target `{path}` is not deferrable on a packed tier (a \
                             tucker/CP `lokr_t2`, or a base that does not factor as a·b × c·d) — no \
                             allocation-free structured form. Use the dense (bf16) tier. sc-10050."
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
            "qwen edit",
            "expected diffusers/PEFT `‹transformer.|diffusion_model.›‹path›.lora_A/B|lora_down/up.\
             weight` (+ optional `.alpha`) over the MMDiT `transformer_blocks.{i}.{attn.*,img_mlp.*,\
             txt_mlp.*}` modules, or `‹path›.lokr_w1/w2` with networkType=lokr (LoKr)",
            specs.len(),
        ));
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::{DType, Device};

    /// A tiny stand-in for the base MMDiT tensor map: two per-block attention Linears + one MLP Linear.
    fn base_map() -> HashMap<String, Tensor> {
        let dev = Device::Cpu;
        let mut m = HashMap::new();
        for key in [
            "transformer_blocks.0.attn.to_q.weight",
            "transformer_blocks.0.attn.to_out.0.weight",
            "transformer_blocks.0.img_mlp.net.0.proj.weight",
        ] {
            m.insert(
                key.to_string(),
                Tensor::zeros((4, 4), DType::BF16, &dev).unwrap(),
            );
        }
        m
    }

    fn t2(data: &[f32], r: usize, c: usize) -> Tensor {
        Tensor::from_vec(data.to_vec(), (r, c), &Device::Cpu).unwrap()
    }

    /// **Additive == folded parity at the resolver level (sc-11091).** The unmerged factors
    /// [`resolve_lora_file`] produces (`a = downᵀ`, `b = upᵀ·(alpha/rank)`, pushed at the user `scale`)
    /// reproduce the folded `x·(W + δ)ᵀ` on a dense base — using the qwen-edit crate's exact
    /// `classify_lora_key` + effective alpha/rank — so the **packed additive** path and the **dense
    /// fold** path agree to f32 tolerance. This is the acceptance parity the sampler's ~1-ULP
    /// sensitivity needs (`candle_gen::train::lora::reconstruct_lora_delta`).
    #[test]
    fn resolve_lora_matches_fold_on_dense() {
        use crate::quant::QLinear;
        use candle_gen::candle_nn::Linear;
        let dev = Device::Cpu;
        let (out_dim, in_dim, rank) = (16usize, 12usize, 3usize);
        let path = "transformer_blocks.0.attn.to_q";
        let down = Tensor::randn(0f32, 1f32, (rank, in_dim), &dev).unwrap(); // A [rank, in]
        let up = Tensor::randn(0f32, 1f32, (out_dim, rank), &dev).unwrap(); // B [out, rank]
        let (alpha, scale) = (6.0f32, 0.8f32); // ratio = alpha/rank = 2.0
        let af = AdapterFile {
            tensors: HashMap::from([
                (format!("{path}.lora_down.weight"), down.clone()),
                (format!("{path}.lora_up.weight"), up.clone()),
                (
                    format!("{path}.alpha"),
                    Tensor::from_vec(vec![alpha], (1,), &dev).unwrap(),
                ),
            ]),
            meta: HashMap::new(),
        };
        let mut pending: BTreeMap<String, Vec<PendingLora>> = BTreeMap::new();
        let mut skipped = 0usize;
        resolve_lora_file(&af, scale, &mut pending, &mut skipped).unwrap();
        assert_eq!(skipped, 0, "clean LoRA resolves with no skipped keys");
        let p = &pending[path][0];
        assert_eq!(p.a.dims(), &[in_dim, rank], "a = downᵀ [in, rank]");
        assert_eq!(p.b.dims(), &[rank, out_dim], "b = upᵀ·ratio [rank, out]");

        // Additive: base W + the resolved residual.
        let w = Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev).unwrap();
        let mut additive = QLinear::from_dense(Linear::new(w.clone(), None), in_dim, out_dim);
        additive.push_lora(p.a.clone(), p.b.clone(), p.scale);

        // Folded: δ = (alpha/rank)·scale·(B·A); W_merged = W + δ.
        let delta = reconstruct_lora_delta(&down, &up, alpha, rank as f32, scale).unwrap();
        let folded = QLinear::from_dense(Linear::new((w + delta).unwrap(), None), in_dim, out_dim);

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

    /// A **LoHa** file is rejected by [`install_additive`]'s resolve preamble before any DiT walk (no
    /// allocation-free structured form on a packed tier), and an untagged third-party **LyCORIS LoKr**
    /// likewise — both point the caller at the dense tier (sc-11091). Read via an in-memory
    /// [`AdapterFile`] would bypass `read_adapter`, so these are exercised end-to-end in the DiT install
    /// test (`crate::transformer` tests); here we assert the key detectors classify them.
    #[test]
    fn loha_and_untagged_lokr_are_detected() {
        use candle_gen::gen_core::weightsmeta as wmeta;
        assert!(wmeta::keys_contain_loha(
            ["transformer_blocks.0.attn.to_q.hada_w1_a"].into_iter()
        ));
        assert!(wmeta::keys_contain_lokr(
            ["transformer_blocks.0.attn.to_q.lokr_w1"].into_iter()
        ));
    }

    /// The lightx2v Lightning shape: bare dotted path + `lora_down`/`lora_up` + per-module `.alpha`.
    #[test]
    fn classify_resolves_bare_down_up_and_alpha() {
        assert!(matches!(
            classify_lora_key("transformer_blocks.0.attn.to_q.lora_down.weight").unwrap(),
            (p, Role::Down) if p == "transformer_blocks.0.attn.to_q"
        ));
        assert!(matches!(
            classify_lora_key("transformer_blocks.0.attn.to_q.lora_up.weight").unwrap(),
            (p, Role::Up) if p == "transformer_blocks.0.attn.to_q"
        ));
        assert!(matches!(
            classify_lora_key("transformer_blocks.0.attn.to_out.0.alpha").unwrap(),
            (p, Role::Alpha) if p == "transformer_blocks.0.attn.to_out.0"
        ));
    }

    /// PEFT spelling with a `transformer.` namespace prefix + `lora_A`/`lora_B` (+ `.default.` infix).
    #[test]
    fn classify_strips_namespace_prefix_and_peft_naming() {
        let (p, role) =
            classify_lora_key("transformer.transformer_blocks.5.img_mlp.net.2.lora_A.weight")
                .unwrap();
        assert_eq!(p, "transformer_blocks.5.img_mlp.net.2");
        assert!(matches!(role, Role::Down));
        assert!(matches!(
            classify_lora_key(
                "diffusion_model.transformer_blocks.5.txt_mlp.net.2.lora_B.default.weight"
            )
            .unwrap()
            .1,
            Role::Up
        ));
        // A non-LoRA key is out of surface.
        assert!(classify_lora_key("transformer_blocks.0.attn.norm_q.weight").is_none());
    }

    /// The Lightning merge: a bare down/up + per-module `.alpha` LoRA folds `W += (alpha/rank)·B·A`.
    #[test]
    fn merge_lightning_shape_folds_expected_delta() {
        let mut map = base_map();
        let path = "transformer_blocks.0.attn.to_q";
        let down = t2(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], 2, 4); // A [rank=2, in=4]
        let up = t2(&[2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0], 4, 2); // B [out=4, rank=2]
        let af = AdapterFile {
            tensors: HashMap::from([
                (format!("{path}.lora_down.weight"), down.clone()),
                (format!("{path}.lora_up.weight"), up.clone()),
                (
                    format!("{path}.alpha"),
                    Tensor::from_vec(vec![4.0f32], (1,), &Device::Cpu).unwrap(),
                ),
            ]),
            meta: HashMap::new(),
        };
        let mut report = MergeReport::default();
        merge_lora_file(&mut map, &af, 1.0, &mut report).unwrap();
        assert_eq!(report.merged, 1);
        assert_eq!(report.skipped_keys, 0);
        // alpha 4 / rank 2 = 2.0; base is zero, so the merged weight IS ΔW = 2·(B·A).
        let merged = map[&format!("{path}.weight")].to_dtype(DType::F32).unwrap();
        let expected = reconstruct_lora_delta(&down, &up, 4.0, 2.0, 1.0).unwrap();
        let diff = (merged - expected)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff < 1e-2, "merged weight off by {diff}"); // bf16 base round-trip tolerance
    }

    /// The user-`scale` knob: the same adapter at scale 0.5 yields half the delta.
    #[test]
    fn merge_honors_user_scale() {
        let mut map = base_map();
        let path = "transformer_blocks.0.attn.to_q";
        let down = Tensor::randn(0f32, 1f32, (2, 4), &Device::Cpu).unwrap();
        let up = Tensor::randn(0f32, 1f32, (4, 2), &Device::Cpu).unwrap();
        let af = AdapterFile {
            tensors: HashMap::from([
                (format!("{path}.lora_down.weight"), down.clone()),
                (format!("{path}.lora_up.weight"), up.clone()),
            ]),
            meta: HashMap::new(),
        };
        let mut report = MergeReport::default();
        merge_lora_file(&mut map, &af, 0.5, &mut report).unwrap();
        let merged = map[&format!("{path}.weight")].to_dtype(DType::F32).unwrap();
        // alpha defaults to rank (2) ⇒ effective scale = (2/2)·0.5 = 0.5.
        let expected = reconstruct_lora_delta(&down, &up, 2.0, 2.0, 0.5).unwrap();
        let diff = (merged - expected)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff < 1e-2, "scaled merge off by {diff}");
    }

    /// PEFT LoKr (`networkType=lokr`, rank/alpha in metadata) folds the kron delta into the dense weight.
    #[test]
    fn merge_lokr_folds_kron_delta() {
        let mut map = base_map();
        let path = "transformer_blocks.0.attn.to_q";
        let w1 = t2(&[1.0, 0.0, 0.0, 1.0], 2, 2);
        let w2 = t2(&[0.5, 0.0, 0.0, 0.5], 2, 2);
        let af = AdapterFile {
            tensors: HashMap::from([
                (format!("{path}.lokr_w1"), w1.clone()),
                (format!("{path}.lokr_w2"), w2.clone()),
            ]),
            meta: HashMap::from([
                ("networkType".to_string(), "lokr".to_string()),
                ("rank".to_string(), "2".to_string()),
                ("alpha".to_string(), "2".to_string()),
            ]),
        };
        let mut report = MergeReport::default();
        merge_lokr_file(&mut map, &af, 1.0, &mut report).unwrap();
        assert_eq!(report.merged, 1);
        let merged = map[&format!("{path}.weight")].to_dtype(DType::F32).unwrap();
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
            (4, 4),
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

    /// An untagged third-party LyCORIS LoKr (no `networkType`) is detected by keys + merged.
    #[test]
    fn merge_thirdparty_lokr_routes_and_merges() {
        let mut map = base_map();
        let path = "transformer_blocks.0.attn.to_q";
        let af = AdapterFile {
            tensors: HashMap::from([
                // both-full ⇒ lycoris scale 1.0.
                (format!("{path}.lokr_w1"), t2(&[1.0, 0.0, 0.0, 1.0], 2, 2)),
                (format!("{path}.lokr_w2"), t2(&[0.5, 0.0, 0.0, 0.5], 2, 2)),
            ]),
            meta: HashMap::new(), // no stamp → third-party
        };
        assert!(!af.declares_lokr());
        assert!(wmeta::keys_contain_lokr(
            af.tensors.keys().map(String::as_str)
        ));
        // `merge_adapters` reads the file from disk; drive the in-memory third-party path directly.
        let mut report = MergeReport::default();
        merge_lokr_thirdparty(&mut map, &af, 1.0, &mut report).unwrap();
        assert_eq!(report.merged, 1);
    }

    /// A third-party LoHa (`hada_*`) routes through the Hadamard merge into the resolved Linear.
    #[test]
    fn merge_thirdparty_loha_routes_and_merges() {
        let mut map = base_map();
        let path = "transformer_blocks.0.attn.to_q";
        let af = AdapterFile {
            tensors: HashMap::from([
                (
                    format!("{path}.hada_w1_a"),
                    t2(&[0.5, 0.1, -0.2, 0.3], 4, 1),
                ),
                (
                    format!("{path}.hada_w1_b"),
                    t2(&[0.4, -0.1, 0.2, 0.6], 1, 4),
                ),
                (
                    format!("{path}.hada_w2_a"),
                    t2(&[0.2, 0.0, 0.1, -0.3], 4, 1),
                ),
                (
                    format!("{path}.hada_w2_b"),
                    t2(&[1.0, 0.5, -0.5, 0.25], 1, 4),
                ),
            ]),
            meta: HashMap::new(),
        };
        assert!(wmeta::keys_contain_loha(
            af.tensors.keys().map(String::as_str)
        ));
        let mut report = MergeReport::default();
        merge_loha_thirdparty(&mut map, &af, 1.0, &mut report).unwrap();
        assert_eq!(report.merged, 1);
        let merged = map[&format!("{path}.weight")]
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(merged.iter().all(|v| v.is_finite()));
    }

    /// An empty spec list merges nothing (no error); the production edit path.
    #[test]
    fn merge_adapters_empty_is_noop() {
        let mut map = base_map();
        let report = merge_adapters(&mut map, &[]).unwrap();
        assert_eq!(report, MergeReport::default());
    }

    /// A non-empty LoRA file that matches no MMDiT module merges nothing (the loud-error precondition).
    #[test]
    fn merge_lora_file_matches_nothing_when_off_surface() {
        let mut map = base_map();
        let af = AdapterFile {
            tensors: HashMap::from([
                (
                    "transformer_blocks.99.attn.to_q.lora_down.weight".to_string(),
                    t2(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 2, 4),
                ),
                (
                    "transformer_blocks.99.attn.to_q.lora_up.weight".to_string(),
                    t2(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 4, 2),
                ),
            ]),
            meta: HashMap::new(),
        };
        let mut report = MergeReport::default();
        merge_lora_file(&mut map, &af, 1.0, &mut report).unwrap();
        assert_eq!(report.merged, 0);
        assert!(report.skipped_keys >= 1);
    }
}
