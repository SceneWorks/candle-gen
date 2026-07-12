//! Anima LoRA/LoKr adapter consumption on the candle lane (sc-10525, the candle twin of MLX sc-10521).
//!
//! **Merge, don't residual** at the safetensors-key level, before the DiT + conditioner are built: the
//! Anima DiT single-file checkpoint holds BOTH the Cosmos DiT (`{prefix}.blocks.*` + globals) AND the
//! bundled `AnimaTextConditioner` (`{prefix}.llm_adapter.blocks.*`) under one root (`prefix` = `net`
//! for the base cut, `model.diffusion_model` for turbo/aesthetic). A trained adapter carries BOTH
//! stacks under ComfyUI's `diffusion_model.` namespace (PEFT `lora_A.weight`/`lora_B.weight`), so after
//! stripping the namespace every target path (`blocks.*` for the DiT, `llm_adapter.blocks.*` for the
//! conditioner) folds into the **same** base key `{prefix}.{path}.weight` — no separate DiT/conditioner
//! routing is needed, because the conditioner already lives under `{prefix}.llm_adapter.` in the base.
//!
//! **The verified trap (sc-10274 class).** `anima-turbo-lora-v0.2` is **508** target pairs (**448** DiT
//! and **60** `llm_adapter.*`); `anima-greg-rutkowski-style` is **448** DiT-only. The merge is
//! **strict**: every target in the file MUST resolve to a base `{prefix}.{path}.weight` key, else it
//! hard-errors naming the unrouted targets — a DiT-only walk that skipped the 60 conditioner targets
//! cannot silently load at partial strength. (For `anima-turbo-lora-v0.2` specifically all 60
//! conditioner `lora_B` are zero-init, so their delta is numerically inert for THIS file — but the
//! guard is about the MECHANISM: a non-zero conditioner LoRA, e.g. the shipped `anima-rl-v0.1`, must
//! not silently fold at partial strength.) No `alpha`/`rank` in the PEFT metadata (`__metadata__ ==
//! {"format":"pt"}`, zero `.alpha` tensors) means α = r means scale 1.0, folded via the same f32 math
//! a candle trainer's forward would use.

use std::collections::{BTreeMap, HashMap, HashSet};

use candle_gen::candle_core::{DType, Tensor};
use candle_gen::gen_core::{AdapterKind, AdapterSpec};
use candle_gen::train::lora::{reconstruct_lokr_delta, reconstruct_lora_delta};
use candle_gen::train::merge::{
    merge_into, no_target_matched, read_adapter, read_scalar, AdapterFile, LoraTriple, MergeReport,
    Role,
};
use candle_gen::{CandleError, Result};

use crate::adapt::{AdaptLinear, LokrFactors};
use crate::conditioner::AnimaTextConditioner;
use crate::transformer::CosmosDiT;

/// LoRA-key namespace prefixes an Anima adapter may carry, longest-first so the more specific PEFT form
/// wins (the trained files use `diffusion_model.`; a bare-key candle-trained adapter matches `""`).
const LORA_PREFIXES: [&str; 5] = [
    "base_model.model.diffusion_model.",
    "base_model.model.",
    "diffusion_model.",
    "transformer.",
    "",
];

/// LoKr per-module factor suffixes, longest-first so `.lokr_w1_a` wins over `.lokr_w1`.
const LOKR_SUFFIXES: [&str; 6] = [
    ".lokr_w1_a",
    ".lokr_w1_b",
    ".lokr_w1",
    ".lokr_w2_a",
    ".lokr_w2_b",
    ".lokr_w2",
];

fn strip_lora_prefix(key: &str) -> &str {
    for p in LORA_PREFIXES {
        if let Some(rem) = key.strip_prefix(p) {
            return rem;
        }
    }
    key
}

/// Map one LoRA key to `(module_path, role)` (path is namespace-stripped, e.g. `blocks.0.self_attn.q_proj`
/// or `llm_adapter.blocks.0.cross_attn.k_proj`). Splits on the `.lora_{A,B,down,up}.weight` / `.alpha`
/// suffix so `adaln_modulation_self_attn.1`'s trailing `.1` (a real path segment) survives.
fn classify_lora_key(key: &str) -> Option<(String, Role)> {
    let rem = strip_lora_prefix(key);
    for (suf, role) in [
        (".lora_A.weight", Role::Down),
        (".lora_B.weight", Role::Up),
        (".lora_A.default.weight", Role::Down),
        (".lora_B.default.weight", Role::Up),
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

/// Map one LoKr factor key to `(module_path, factor_name)`.
fn classify_lokr_key(key: &str) -> Option<(String, &'static str)> {
    for suf in LOKR_SUFFIXES {
        if let Some(stem) = key.strip_suffix(suf) {
            return Some((strip_lora_prefix(stem).to_string(), &suf[1..]));
        }
    }
    None
}

/// Fold one LoRA file into `base` (keys `{prefix}.{path}.weight`) at `scale`. Every complete `(A,B)`
/// target is either routed (folded, path pushed to `routed`) or recorded in `unrouted` (its base key is
/// absent) — the caller turns a non-empty `unrouted` into a hard error (strict, no silent partial).
fn merge_lora_file(
    base: &mut HashMap<String, Tensor>,
    af: &AdapterFile,
    prefix: &str,
    scale: f32,
    report: &mut MergeReport,
    routed: &mut Vec<String>,
    unrouted: &mut Vec<String>,
) -> Result<()> {
    let mut triples: BTreeMap<String, LoraTriple> = BTreeMap::new();
    for (key, t) in &af.tensors {
        match classify_lora_key(key) {
            Some((path, Role::Down)) => triples.entry(path).or_default().down = Some(t.clone()),
            Some((path, Role::Up)) => triples.entry(path).or_default().up = Some(t.clone()),
            Some((path, Role::Alpha)) => {
                triples.entry(path).or_default().alpha = Some(read_scalar(key, "alpha", t)?)
            }
            None => report.skipped_keys += 1,
        }
    }

    for (path, t) in triples {
        let (Some(down), Some(up)) = (t.down, t.up) else {
            report.skipped_keys += 1; // half-pair
            continue;
        };
        if down.dims().len() != 2 || up.dims().len() != 2 {
            report.skipped_keys += 1; // Anima adapts Linears only
            continue;
        }
        let base_key = format!("{prefix}.{path}.weight");
        if !base.contains_key(&base_key) {
            unrouted.push(path);
            continue;
        }
        let rank = down.dims()[0] as f32;
        // No PEFT `.alpha` ⇒ α = rank ⇒ (alpha/rank) = 1 ⇒ scale 1.0 fold.
        let alpha = t.alpha.unwrap_or(rank);
        let delta = reconstruct_lora_delta(&down, &up, alpha, rank, scale)?;
        merge_into(base, &base_key, &delta, report)?;
        routed.push(path);
    }
    Ok(())
}

/// Fold one LoKr file into `base` at `scale` — `rank`/`alpha` from the file's `__metadata__` (default
/// `rank = 1`, `alpha = rank`), per-module factors grouped, `δ = (alpha/rank)·kron(w1,w2)·scale`.
fn merge_lokr_file(
    base: &mut HashMap<String, Tensor>,
    af: &AdapterFile,
    prefix: &str,
    scale: f32,
    report: &mut MergeReport,
    routed: &mut Vec<String>,
    unrouted: &mut Vec<String>,
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
        match classify_lokr_key(key) {
            Some((path, factor)) => {
                grouped.entry(path).or_default().insert(factor, t.clone());
            }
            None => report.skipped_keys += 1,
        }
    }

    for (path, f) in grouped {
        let base_key = format!("{prefix}.{path}.weight");
        let Some(w) = base.get(&base_key) else {
            unrouted.push(path);
            continue;
        };
        if w.dims().len() != 2 {
            report.skipped_keys += 1;
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
        routed.push(path);
    }
    Ok(())
}

/// Fold every adapter in `specs` into the Anima DiT+conditioner base weight map `base` (CPU), keyed
/// `{prefix}.{path}.weight`, stacked and mixed LoRA/LoKr. **Strict**: a target whose base key is absent
/// (an unrouted `llm_adapter.*` / `blocks.*`) is a hard error naming the unrouted paths — never a silent
/// partial fold (the sc-10274 guard). A spec list that routes zero targets also hard-errors. Returns the
/// [`MergeReport`] (`report.merged` = the routed-target count: 508 for the turbo LoRA, 448 for a
/// DiT-only style LoRA).
pub fn apply_anima_adapters(
    base: &mut HashMap<String, Tensor>,
    prefix: &str,
    specs: &[AdapterSpec],
) -> Result<MergeReport> {
    let mut report = MergeReport::default();
    let mut routed: Vec<String> = Vec::new();
    let mut unrouted: Vec<String> = Vec::new();

    for spec in specs {
        let af = read_adapter(&spec.path)?;
        let is_lokr = matches!(spec.kind, AdapterKind::Lokr) || af.declares_lokr();
        if is_lokr {
            merge_lokr_file(
                base,
                &af,
                prefix,
                spec.scale,
                &mut report,
                &mut routed,
                &mut unrouted,
            )?;
        } else {
            merge_lora_file(
                base,
                &af,
                prefix,
                spec.scale,
                &mut report,
                &mut routed,
                &mut unrouted,
            )?;
        }
    }

    if !unrouted.is_empty() {
        unrouted.sort();
        return Err(CandleError::Msg(format!(
            "anima: {} adapter target(s) did not route to a base module under prefix {prefix:?} \
             (no silent partial fold — sc-10274). First unrouted: {}",
            unrouted.len(),
            unrouted
                .iter()
                .take(5)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        )));
    }
    if routed.is_empty() {
        return Err(no_target_matched(
            "anima",
            "expected ComfyUI `diffusion_model.<path>.lora_A/B.weight` (DiT `blocks.*` + conditioner \
             `llm_adapter.blocks.*`)",
            specs.len(),
        ));
    }
    Ok(report)
}

// ---- Unmerged additive-LoRA path (sc-10640, epic 10043) -----------------------------------------
//
// A packed q4/q8 DiT has no dense `.weight` to fold `B·A` into (`apply_anima_adapters` folds the delta
// before the model is built), so on a packed tier the adapters apply as **forward-time residuals** on
// the *already-built* model: `y = base(x) + scale·((x·A)·B)` ([`AdaptLinear::push_lora`]), the packed
// codes used AS-IS — no dense weight is materialized, so a q4 base keeps its q4 footprint. This is the
// candle twin of the MLX additive branch (sc-10578) and the direct analog of `candle-gen-wan`'s
// `install_additive` (sc-10094). Both the 448 DiT targets (packed) and the 60 conditioner targets (dense
// bf16 — Anima packs ONLY the DiT) install uniformly through this one path.
//
// **LoKr → structured Kronecker residual (sc-10713).** A LoKr also applies unmerged on a packed base, via
// the Kronecker vec-trick `vec(w1·reshape(x)·w2ᵀ)` ([`LokrFactors`], the candle port of mlx-gen's
// `Adapter::LokrStructured`): the small `[a,c]`/`[b,d]` factors are applied WITHOUT ever forming the
// `[out,in]` delta, so a LoKr costs the same memory as a plain LoRA on any tier. The FULL
// `(alpha/rank)·strength` scale is baked into `w2` (the structured residual carries no separate scale) —
// distinct from the fold path's `reconstruct_lokr_delta`, which bakes only `alpha/rank` and rides
// `strength` in the merge scale (the two-conventions trap, sc-10578). A LoKr module with no 2-D matrix
// form (a conv/tucker factor, or a factor/base mismatch) is not deferrable and is rejected on packed
// rather than materialized.
//
// **LoHa → rejected on packed (sc-10713 / sc-10678 policy call).** The Hadamard product
// `(w1_a·w1_b) ⊙ (w2_a·w2_b)` has NO Kronecker structure and therefore no allocation-free deferred form:
// applying it requires forming the two full `[out,in]` matrices before the elementwise multiply. On a
// packed tier that is exactly the ≈3.9 GB memory-lie sc-10640 forbids (a q4 DiT + 448 dense deltas > the
// bf16 tier), so a LoHa on a packed base is a hard, actionable error — never a silent materialization. No
// official Anima LoHa exists, so nothing is user-blocked; a bespoke deferred form is unwarranted. On a
// dense tier a LoKr still folds through [`apply_anima_adapters`] (via `reconstruct_lokr_delta`), but a
// LoHa is likewise unhandled there: the dense path imports no LoHa reconstruction, so a LoHa's `hada_*`
// keys route to zero targets and hit the `no_target_matched` hard error — never a fold — matching the
// packed path's loud LoHa rejection. (Again moot: no official Anima LoHa exists.)

/// A LoRA residual resolved from a file, pending attachment to a host projection: `a = downᵀ`
/// `[in, rank]`, `b = upᵀ·(alpha/rank)` `[rank, out]` (the `alpha/rank` ratio folded into `b`, matching
/// the fold path's split), `scale` = the spec's user strength. Held f32 (the merge dtype).
struct PendingLora {
    a: Tensor,
    b: Tensor,
    scale: f64,
}

/// True when a file carries **LoHa** (Hadamard-product) factors (`hada_*` keys). LoHa has no
/// `AdapterKind` variant (gen-core knows only `Lora`/`Lokr`), so it is detected by keys alone. On a
/// packed base a LoHa is **rejected** (its Hadamard product has no allocation-free deferred form — see
/// the module docs / sc-10713); on a dense base it is likewise unhandled — the dense fold path
/// [`apply_anima_adapters`] reconstructs only LoRA/LoKr, so a LoHa's `hada_*` keys route to zero targets
/// and error out at target-matching rather than folding. Moot in practice: no official Anima LoHa exists.
fn is_loha(af: &AdapterFile) -> bool {
    af.tensors
        .keys()
        .any(|k| k.contains(".hada_") || k.starts_with("hada_"))
}

/// True when a file carries **LoKr** (Kronecker) factors, is stamped `networkType=lokr`, or is declared
/// `AdapterKind::Lokr`. On a packed base a LoKr installs as a **structured residual** (the vec-trick,
/// [`resolve_lokr_file`] → [`LokrFactors`]); on a dense base it folds through [`apply_anima_adapters`].
fn is_lokr(af: &AdapterFile, kind: AdapterKind) -> bool {
    kind == AdapterKind::Lokr
        || af.declares_lokr()
        || af
            .tensors
            .keys()
            .any(|k| k.contains(".lokr_") || k.starts_with("lokr_"))
}

/// A LoKr residual resolved from a file, pending attachment to a host projection as a **structured**
/// (Kronecker vec-trick) residual. Holds the small per-module factors (full `lokr_w1`/`lokr_w2` or their
/// low-rank `w_a`/`w_b` legs) at **f32**, plus the fully-effective `scale` (`(alpha/rank)·strength`,
/// derived once per file — the FULL scale the structured residual bakes into `w2`, unlike the fold path's
/// `alpha/rank`-only delta, sc-10578). The `[a,c]`/`[b,d]` [`LokrFactors`] are built at attach time, when
/// the target projection's `[out,in]` base shape is known (so a·b==out, c·d==in can be checked without
/// materializing anything).
struct PendingLokr {
    w1: Option<Tensor>,
    w1_a: Option<Tensor>,
    w1_b: Option<Tensor>,
    w2: Option<Tensor>,
    w2_a: Option<Tensor>,
    w2_b: Option<Tensor>,
    scale: f64,
}

/// Resolve one LoKr file's per-module factor groups into [`PendingLokr`]s at the fully-effective scale,
/// mirroring [`merge_lokr_file`]'s `rank`/`alpha` read (`af.meta`, defaulting `rank=1`, `alpha=rank`) and
/// factor grouping — but producing the **unmerged** small factors for the structured residual instead of
/// a folded `[out,in]` delta. The `(alpha/rank)·strength` scale is baked here (the structured residual
/// has no separate scale), NOT `alpha/rank` alone as the fold path does (the two-conventions trap).
fn resolve_lokr_file(
    af: &AdapterFile,
    scale: f32,
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
    // FULL effective scale — mirrors `reconstruct_lokr_delta`'s `(alpha/rank)·scale`, but baked into the
    // structured factors rather than split across a delta (alpha/rank) + an Adapter scale (strength).
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
        // f32 factors for the vec-trick build (the merge/train dtype), matching `reconstruct_lokr_delta`.
        let get = |k: &str| -> Result<Option<Tensor>> {
            Ok(f.get(k).map(|t| t.to_dtype(DType::F32)).transpose()?)
        };
        pending.entry(path).or_default().push(PendingLokr {
            w1: get("lokr_w1")?,
            w1_a: get("lokr_w1_a")?,
            w1_b: get("lokr_w1_b")?,
            w2: get("lokr_w2")?,
            w2_a: get("lokr_w2_a")?,
            w2_b: get("lokr_w2_b")?,
            scale: full,
        });
    }
    Ok(())
}

/// Resolve one LoRA file's complete `(down, up[, alpha])` groups into per-path [`PendingLora`]s at
/// `scale`, mirroring [`merge_lora_file`]'s classification + `alpha/rank` split — but producing the
/// **unmerged** `a`/`b` factors instead of a folded delta.
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
                triples.entry(path).or_default().alpha = Some(read_scalar(key, "alpha", t)?)
            }
            None => *skipped_keys += 1,
        }
    }
    for (path, t) in triples {
        let (Some(down), Some(up)) = (t.down, t.up) else {
            *skipped_keys += 1; // half-pair
            continue;
        };
        if down.dims().len() != 2 || up.dims().len() != 2 {
            *skipped_keys += 1; // Anima adapts Linears only
            continue;
        }
        let rank = down.dims()[0] as f64;
        if rank == 0.0 {
            *skipped_keys += 1;
            continue;
        }
        // No PEFT `.alpha` ⇒ α = rank ⇒ ratio 1.0 (the missing-alpha convention), same as the fold.
        let alpha = t.alpha.map(|a| a as f64).unwrap_or(rank);
        let ratio = alpha / rank;
        // a = downᵀ [in, rank]; b = (upᵀ·ratio) [rank, out]. f32, contiguous for the matmul.
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

/// Install `specs` as **forward-time additive residuals** on an already-built (packed) DiT + conditioner
/// — the packed-tier path where [`apply_anima_adapters`] can't fold (sc-10640). The DiT (packed) and
/// conditioner (dense bf16) projections are visited uniformly; each resolved target pushes a residual
/// onto its projection, the packed codes untouched. **Strict**, exactly like the fold path: a target
/// present in a file but absent from the host surface is a hard error naming the unrouted paths (the
/// sc-10274 no-silent-partial guard), and a spec set that routes zero targets also hard-errors.
///
/// **LoRA** installs as `scale·((x·A)·B)` ([`AdaptLinear::push_lora`]). **LoKr** installs as a
/// **structured Kronecker residual** ([`AdaptLinear::push_lokr_structured`], sc-10713) — the vec-trick,
/// no `[out,in]` materialized; a LoKr module with no deferrable 2-D form on a packed base is a hard,
/// actionable error rather than a silent materialization. **LoHa** on a packed base is a hard, actionable
/// error (its Hadamard product has no allocation-free form — see the module docs). Returns the
/// [`MergeReport`] (`report.merged` = the residual-installed target count: 508 for the turbo LoRA, 448
/// for a DiT-only style LoRA).
pub fn install_anima_residuals(
    dit: &mut CosmosDiT,
    conditioner: &mut AnimaTextConditioner,
    specs: &[AdapterSpec],
) -> Result<MergeReport> {
    // The factors are read on the CPU; the base weight lives on the DiT's device (CUDA on a packed
    // tier), so they are moved onto it during the attach — else the residual matmul is a device mismatch.
    let device = dit.device().clone();
    let mut pending_lora: BTreeMap<String, Vec<PendingLora>> = BTreeMap::new();
    let mut pending_lokr: BTreeMap<String, Vec<PendingLokr>> = BTreeMap::new();
    let mut report = MergeReport::default();
    for spec in specs {
        let af = read_adapter(&spec.path)?;
        // LoHa (Hadamard) has no allocation-free structured form — reject on the packed tier (never a
        // silent materialization of the ~3.9 GB per-target delta the packed path exists to avoid).
        if is_loha(&af) {
            return Err(CandleError::Msg(format!(
                "anima: a LoHa adapter cannot apply on a quantized (packed q4/q8) Anima tier — its \
                 Hadamard product (w1_a·w1_b) ⊙ (w2_a·w2_b) has no allocation-free structured form (unlike \
                 LoKr's Kronecker vec-trick), so it would have to materialize a full [out,in] delta per \
                 target (≈3.9 GB over the 448 DiT targets, more than the bf16 tier it was meant to shrink). \
                 Use the dense (bf16) tier, where it folds into the weight, or a plain LoRA/LoKr, which \
                 applies additively on any tier. sc-10713. Offending file: {}",
                spec.path.display()
            )));
        }
        if is_lokr(&af, spec.kind) {
            resolve_lokr_file(&af, spec.scale, &mut pending_lokr, &mut report.skipped_keys)?;
        } else {
            resolve_lora_file(&af, spec.scale, &mut pending_lora, &mut report.skipped_keys)?;
        }
    }

    // Attach: walk the DiT + conditioner once, pushing any resolved residual for each projection's path.
    // A LoRA factor whose dims don't match the target projection is surfaced as a skipped key, never a
    // crashing forward (the additive analog of the fold path's `merge_into` shape guard). A LoKr that
    // does not decompose the target's `[out,in]` via the vec-trick (a conv/tucker or shape-mismatched
    // LoKr) is a HARD error on the packed tier — it has no allocation-free form, and materializing it
    // would defeat the packed tier (sc-10713), so it must not be silently dropped or force a dense delta.
    let mut matched: HashSet<String> = HashSet::new();
    {
        let pending_lora = &pending_lora;
        let pending_lokr = &pending_lokr;
        let matched = &mut matched;
        let report = &mut report;
        let device = &device;
        let mut visit = |path: &str, lin: &mut AdaptLinear| -> Result<()> {
            let (out_f, in_f) = lin.base_shape();
            if let Some(list) = pending_lora.get(path) {
                matched.insert(path.to_string());
                for p in list {
                    if p.a.dims()[0] != in_f || p.b.dims()[1] != out_f {
                        report.skipped_keys += 1; // shape-mismatched factor for this projection
                        continue;
                    }
                    lin.push_lora(p.a.to_device(device)?, p.b.to_device(device)?, p.scale);
                    report.merged += 1;
                }
            }
            if let Some(list) = pending_lokr.get(path) {
                matched.insert(path.to_string());
                for p in list {
                    // Build the small `[a,c]`/`[b,d]` Kronecker factors against the now-known base shape —
                    // never the `[out,in]` delta. `None` ⇒ not deferrable on packed ⇒ hard reject.
                    match LokrFactors::build(
                        p.scale,
                        (out_f, in_f),
                        p.w1.as_ref(),
                        p.w1_a.as_ref(),
                        p.w1_b.as_ref(),
                        p.w2.as_ref(),
                        None, // peft LoKr never carries a tucker `lokr_t2` (lycoris conv-only).
                        p.w2_a.as_ref(),
                        p.w2_b.as_ref(),
                    )? {
                        Some(f) => {
                            lin.push_lokr_structured(f.to_device(device)?);
                            report.merged += 1;
                        }
                        None => {
                            return Err(CandleError::Msg(format!(
                                "anima: a LoKr target ({path}) has no allocation-free structured form on a \
                                 quantized (packed q4/q8) tier — its factors do not decompose the \
                                 [out,in]=[{out_f},{in_f}] base via the Kronecker vec-trick (a conv/tucker \
                                 or shape-mismatched LoKr). Materializing the full [out,in] delta would \
                                 defeat the packed tier (the ≈3.9 GB memory-lie sc-10640 forbids). Use the \
                                 dense (bf16) tier, where it folds, or a plain LoRA. sc-10713."
                            )));
                        }
                    }
                }
            }
            Ok(())
        };
        dit.visit_adaptable_mut(&mut visit)?;
        conditioner.visit_adaptable_mut(&mut visit)?;
    }

    // Strict: a resolved target absent from the host surface is a hard error (no silent partial residual
    // — sc-10274), and a spec set that routes nothing is a misconfiguration.
    let mut unrouted: Vec<String> = pending_lora
        .keys()
        .chain(pending_lokr.keys())
        .filter(|p| !matched.contains(*p))
        .cloned()
        .collect();
    unrouted.sort();
    unrouted.dedup();
    if !unrouted.is_empty() {
        return Err(CandleError::Msg(format!(
            "anima: {} adapter target(s) did not route to a DiT/conditioner projection (no silent \
             partial residual — sc-10274). First unrouted: {}",
            unrouted.len(),
            unrouted
                .iter()
                .take(5)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        )));
    }
    if report.merged == 0 {
        return Err(no_target_matched(
            "anima",
            "expected ComfyUI `diffusion_model.<path>.lora_A/B.weight` (DiT `blocks.*` + conditioner \
             `llm_adapter.blocks.*`) on a packed tier",
            specs.len(),
        ));
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::{DType, Device};

    const PFX: &str = "net";

    /// A base map with one target of each of the three routing classes: a DiT attention projection, a
    /// DiT adaLN modulation `.1` (the `.N`-suffix path segment), and a conditioner (`llm_adapter.*`).
    fn base_map(out: usize, inp: usize) -> HashMap<String, Tensor> {
        let dev = Device::Cpu;
        let mut m = HashMap::new();
        for path in [
            "blocks.0.self_attn.q_proj",
            "blocks.0.adaln_modulation_self_attn.1",
            "llm_adapter.blocks.0.cross_attn.k_proj",
        ] {
            m.insert(
                format!("{PFX}.{path}.weight"),
                Tensor::zeros((out, inp), DType::F32, &dev).unwrap(),
            );
        }
        m
    }

    fn rand_t(r: usize, c: usize, seed: u64) -> Tensor {
        // Deterministic small values so the fold is exactly reproducible in the assertion.
        let n = r * c;
        let mut s = seed;
        let data: Vec<f32> = (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / u32::MAX as f32) * 0.02 - 0.01
            })
            .collect();
        Tensor::from_vec(data, (r, c), &Device::Cpu).unwrap()
    }

    fn write_lora(
        dir: &std::path::Path,
        paths: &[&str],
        out: usize,
        inp: usize,
        rank: usize,
    ) -> std::path::PathBuf {
        let mut m = HashMap::new();
        for (i, p) in paths.iter().enumerate() {
            m.insert(
                format!("diffusion_model.{p}.lora_A.weight"),
                rand_t(rank, inp, 100 + i as u64),
            );
            m.insert(
                format!("diffusion_model.{p}.lora_B.weight"),
                rand_t(out, rank, 500 + i as u64),
            );
        }
        let path = dir.join("lora.safetensors");
        candle_gen::candle_core::safetensors::save(&m, &path).unwrap();
        path
    }

    /// Injected weight == base + B·A **bit-exact** (f32) for a DiT target, an adaLN target, and a
    /// conditioner target — the weight-level LoRA property, provable with no GPU. Also asserts the
    /// routed count == number of targets (all three classes routed).
    #[test]
    fn lora_fold_is_base_plus_ba_for_all_three_target_classes() {
        let (out, inp, rank) = (8usize, 6usize, 2usize);
        let dir = std::env::temp_dir().join(format!("anima_lora_fold_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let targets = [
            "blocks.0.self_attn.q_proj",
            "blocks.0.adaln_modulation_self_attn.1",
            "llm_adapter.blocks.0.cross_attn.k_proj",
        ];
        let lora_path = write_lora(&dir, &targets, out, inp, rank);

        // Recover the exact A/B written, to compute the expected B@A.
        let af = read_adapter(&lora_path).unwrap();
        let mut base = base_map(out, inp);
        let spec = AdapterSpec::new(lora_path, 1.0, AdapterKind::Lora);
        let report = apply_anima_adapters(&mut base, PFX, std::slice::from_ref(&spec)).unwrap();
        assert_eq!(report.merged, 3, "all three target classes must route");

        for p in targets {
            let a = af
                .tensors
                .get(&format!("diffusion_model.{p}.lora_A.weight"))
                .unwrap();
            let b = af
                .tensors
                .get(&format!("diffusion_model.{p}.lora_B.weight"))
                .unwrap();
            let expected = b.matmul(a).unwrap(); // B[out,r] @ A[r,in] = [out,in], base is zeros
            let got = base.get(&format!("{PFX}.{p}.weight")).unwrap();
            let diff = (got - &expected)
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            assert!(
                diff < 1e-6,
                "{p}: merged != base + B·A (max abs diff {diff})"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Strict routing: a target whose base key is absent (a DiT-only base missing the `llm_adapter.*`
    /// module) is a HARD ERROR — never a silent partial fold (the sc-10274 mutation guard).
    #[test]
    fn unrouted_target_is_a_hard_error() {
        let (out, inp, rank) = (8usize, 6usize, 2usize);
        let dir = std::env::temp_dir().join(format!("anima_lora_unrouted_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let lora_path = write_lora(
            &dir,
            &[
                "blocks.0.self_attn.q_proj",
                "llm_adapter.blocks.0.cross_attn.k_proj",
            ],
            out,
            inp,
            rank,
        );
        // DiT-only base: drop the conditioner module → the llm_adapter target cannot route.
        let mut base = base_map(out, inp);
        base.remove(&format!(
            "{PFX}.llm_adapter.blocks.0.cross_attn.k_proj.weight"
        ));

        let spec = AdapterSpec::new(lora_path, 1.0, AdapterKind::Lora);
        let err = apply_anima_adapters(&mut base, PFX, std::slice::from_ref(&spec))
            .expect_err("a DiT-only base must reject the llm_adapter target, not silently drop it");
        assert!(
            err.to_string().contains("did not route"),
            "expected an unrouted-target error, got: {err}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A synthesized LoKr (no official Anima LoKr exists) loads + applies: `δ = kron(w1, w2)` folded
    /// into the base, and a stacked LoRA+LoKr sums both deltas.
    #[test]
    fn synthesized_lokr_loads_and_stacks_with_lora() {
        let dev = Device::Cpu;
        let dir = std::env::temp_dir().join(format!("anima_lokr_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path_target = "blocks.0.self_attn.q_proj";

        // A full-factor LoKr: w1 [2,3], w2 [4,2] ⇒ kron = [8,6]. No metadata ⇒ rank = alpha = 1 (the
        // merge defaults) ⇒ scale 1.0; the `AdapterKind::Lokr` spec below drives the LoKr dispatch (no
        // `networkType` stamp needed — candle's `save` writes no header metadata).
        let w1 = rand_t(2, 3, 7);
        let w2 = rand_t(4, 2, 9);
        let mut lm = HashMap::new();
        lm.insert(format!("diffusion_model.{path_target}.lokr_w1"), w1.clone());
        lm.insert(format!("diffusion_model.{path_target}.lokr_w2"), w2.clone());
        let lokr_path = dir.join("lokr.safetensors");
        candle_gen::candle_core::safetensors::save(&lm, &lokr_path).unwrap();

        let mut base = HashMap::new();
        base.insert(
            format!("{PFX}.{path_target}.weight"),
            Tensor::zeros((8usize, 6usize), DType::F32, &dev).unwrap(),
        );

        // LoKr alone folds kron(w1, w2).
        let lokr_spec = AdapterSpec::new(lokr_path.clone(), 1.0, AdapterKind::Lokr);
        let r = apply_anima_adapters(&mut base, PFX, std::slice::from_ref(&lokr_spec)).unwrap();
        assert_eq!(r.merged, 1, "the LoKr target must route");
        let after_lokr = base
            .get(&format!("{PFX}.{path_target}.weight"))
            .unwrap()
            .clone();
        let l2 = after_lokr
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            l2 > 0.0,
            "LoKr delta must be non-zero (kron of non-zero factors)"
        );

        // Stacked: fold a LoRA onto the SAME base on top of the LoKr → the two deltas sum.
        let lora_path = write_lora(&dir, &[path_target], 8, 6, 2);
        let lora_spec = AdapterSpec::new(lora_path, 1.0, AdapterKind::Lora);
        let r2 = apply_anima_adapters(&mut base, PFX, std::slice::from_ref(&lora_spec)).unwrap();
        assert_eq!(r2.merged, 1);
        let stacked = base.get(&format!("{PFX}.{path_target}.weight")).unwrap();
        // stacked - after_lokr == the LoRA delta (non-zero) ⇒ additive stacking, not overwrite.
        let added = (stacked - &after_lokr)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            added > 0.0,
            "stacked LoRA delta must add on top of the LoKr, not replace it"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `is_lokr` fires on `lokr_` factor keys and a declared `Lokr` kind (→ structured residual path);
    /// `is_loha` fires on `hada_` keys (→ reject-on-packed); a plain LoRA is neither (→ LoRA residual
    /// path). The three kinds are routed disjointly in `install_anima_residuals` (sc-10713).
    #[test]
    fn is_lokr_and_is_loha_route_the_three_adapter_kinds() {
        let dir = std::env::temp_dir().join(format!("anima_islokr_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        // A LoKr file (`lokr_` keys) → `is_lokr` true even when the spec kind is Lora (keys win); not LoHa.
        let mut lm = HashMap::new();
        lm.insert(
            "diffusion_model.blocks.0.self_attn.q_proj.lokr_w1".to_string(),
            rand_t(2, 3, 1),
        );
        let lokr_path = dir.join("lokr.safetensors");
        candle_gen::candle_core::safetensors::save(&lm, &lokr_path).unwrap();
        let lokr_af = read_adapter(&lokr_path).unwrap();
        assert!(is_lokr(&lokr_af, AdapterKind::Lora), "lokr_ keys ⇒ is_lokr");
        assert!(!is_loha(&lokr_af), "lokr_ keys ⇒ not is_loha");

        // A LoHa file (`hada_` keys) → `is_loha` true; not LoKr.
        let mut hm = HashMap::new();
        hm.insert(
            "diffusion_model.blocks.0.self_attn.q_proj.hada_w1_a".to_string(),
            rand_t(2, 3, 2),
        );
        let loha_path = dir.join("loha.safetensors");
        candle_gen::candle_core::safetensors::save(&hm, &loha_path).unwrap();
        let loha_af = read_adapter(&loha_path).unwrap();
        assert!(is_loha(&loha_af), "hada_ keys ⇒ is_loha");
        assert!(
            !is_lokr(&loha_af, AdapterKind::Lora),
            "hada_ keys ⇒ not is_lokr"
        );

        // A plain LoRA (`lora_A`/`lora_B`) → neither (reaches the LoRA residual path); but a declared
        // `Lokr` kind → `is_lokr` regardless of keys.
        let lora_path = write_lora(&dir, &["blocks.0.self_attn.q_proj"], 8, 6, 2);
        let lora_af = read_adapter(&lora_path).unwrap();
        assert!(
            !is_lokr(&lora_af, AdapterKind::Lora) && !is_loha(&lora_af),
            "plain LoRA ⇒ neither"
        );
        assert!(
            is_lokr(&lora_af, AdapterKind::Lokr),
            "declared Lokr kind ⇒ is_lokr"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `resolve_lokr_file` groups a file's per-module factors and bakes the FULL `(alpha/rank)·strength`
    /// scale (the structured-residual convention, sc-10578) — not `alpha/rank` alone as the fold path
    /// does. Metadata `rank=2, alpha=8` + strength 0.5 ⇒ `full = (8/2)·0.5 = 2.0`. Constructs the
    /// `AdapterFile` directly (its `meta` is the header `__metadata__` the reader would surface).
    #[test]
    fn resolve_lokr_file_bakes_full_alpha_rank_strength_scale() {
        let path = "blocks.0.self_attn.q_proj";
        let tensors = HashMap::from([
            (format!("diffusion_model.{path}.lokr_w1"), rand_t(2, 3, 11)),
            (format!("diffusion_model.{path}.lokr_w2"), rand_t(4, 2, 12)),
        ]);
        let meta = HashMap::from([
            ("rank".to_string(), "2".to_string()),
            ("alpha".to_string(), "8".to_string()),
        ]);
        let af = AdapterFile { tensors, meta };

        let mut pending: BTreeMap<String, Vec<PendingLokr>> = BTreeMap::new();
        let mut skipped = 0usize;
        resolve_lokr_file(&af, 0.5, &mut pending, &mut skipped).unwrap();
        let list = pending.get(path).expect("the LoKr target must resolve");
        assert_eq!(list.len(), 1);
        // full = (alpha/rank)·strength = (8/2)·0.5 = 2.0. Dropping the `·strength` term (a real
        // two-conventions bug) would make this 4.0 — a non-vacuous check.
        assert!(
            (list[0].scale - 2.0).abs() < 1e-9,
            "full scale must be (alpha/rank)·strength = 2.0, got {}",
            list[0].scale
        );
        assert!(list[0].w1.is_some() && list[0].w2.is_some());
    }

    /// `resolve_lora_file` produces the **unmerged** factors the additive residual needs: `a = downᵀ`
    /// `[in, rank]`, `b = upᵀ·(alpha/rank)` `[rank, out]`. With no PEFT alpha the ratio is 1.0, so `b`
    /// equals `upᵀ` exactly — pinning the transpose + scale the residual matmul `(x·a)·b` expects.
    #[test]
    fn resolve_lora_file_factors_are_transposed_and_alpha_scaled() {
        let (out, inp, rank) = (8usize, 6usize, 2usize);
        let dir = std::env::temp_dir().join(format!("anima_resolve_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = "blocks.0.self_attn.q_proj";
        let lora_path = write_lora(&dir, &[path], out, inp, rank);
        let af = read_adapter(&lora_path).unwrap();

        let mut pending: BTreeMap<String, Vec<PendingLora>> = BTreeMap::new();
        let mut skipped = 0usize;
        resolve_lora_file(&af, 1.0, &mut pending, &mut skipped).unwrap();

        let list = pending.get(path).expect("the target must resolve");
        assert_eq!(list.len(), 1);
        let p = &list[0];
        assert_eq!(p.a.dims(), &[inp, rank], "a = downᵀ [in, rank]");
        assert_eq!(p.b.dims(), &[rank, out], "b = upᵀ [rank, out]");
        assert_eq!(p.scale, 1.0);

        // b must equal upᵀ exactly (ratio 1.0, no PEFT alpha).
        let up = af
            .tensors
            .get(&format!("diffusion_model.{path}.lora_B.weight"))
            .unwrap();
        let want_b = up.t().unwrap().contiguous().unwrap();
        let dev_max = (p.b.sub(&want_b).unwrap())
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            dev_max < 1e-6,
            "b must equal upᵀ (ratio 1.0), max diff {dev_max}"
        );

        // With an explicit `.alpha` ≠ rank, `b` picks up the `alpha/rank` ratio. This is the non-vacuous
        // case: dropping `* ratio` in `resolve_lora_file` makes THIS assertion fail (the ratio-1.0 case
        // above cannot catch it).
        let mut m2 = HashMap::new();
        m2.insert(
            format!("diffusion_model.{path}.lora_A.weight"),
            rand_t(rank, inp, 42),
        );
        let up2 = rand_t(out, rank, 43);
        m2.insert(format!("diffusion_model.{path}.lora_B.weight"), up2.clone());
        m2.insert(
            format!("diffusion_model.{path}.alpha"),
            Tensor::from_vec(vec![(2 * rank) as f32], (1,), &Device::Cpu).unwrap(), // ratio 2.0
        );
        let ap = dir.join("lora_alpha.safetensors");
        candle_gen::candle_core::safetensors::save(&m2, &ap).unwrap();
        let af2 = read_adapter(&ap).unwrap();
        let mut pending2: BTreeMap<String, Vec<PendingLora>> = BTreeMap::new();
        let mut skipped2 = 0usize;
        resolve_lora_file(&af2, 1.0, &mut pending2, &mut skipped2).unwrap();
        let p2 = &pending2.get(path).unwrap()[0];
        let want_b2 = (up2.t().unwrap().contiguous().unwrap() * 2.0f64).unwrap();
        let d2 = (p2.b.sub(&want_b2).unwrap())
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            d2 < 1e-6,
            "b must equal upᵀ·(alpha/rank=2.0), max diff {d2}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
