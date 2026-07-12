//! Wan inference-side adapter merge (sc-5167) — load a trained LoRA/LoKr `.safetensors` and fold its
//! delta into the dense DiT-expert weights **before** the stock [`WanTransformer`](crate::transformer)
//! is built. The Wan twin of [`candle-gen-sdxl::adapters`] / `candle-gen-z-image::adapters`, and the
//! closing half of the native-trainer loop: a LoRA/LoKr produced by [`crate::training`]'s Wan MoE
//! trainer now actually loads in candle inference.
//!
//! **Merge, don't residual** (the chaos-sensitive-sampler argument from the SDXL/Z-Image ports), at the
//! **safetensors-key level** before construction: load the expert's base weights into a
//! `HashMap<String,Tensor>` on CPU, add `δ` to `{path}.weight`, then `VarBuilder::from_tensors`. The
//! stock Wan DiT reads diffusers keys 1:1, so `{path}.weight` is a valid base key for every attention
//! projection an adapter targets (`blocks.{i}.attn1/attn2.{to_q,to_k,to_v,to_out.0}`). The delta is
//! reconstructed with the **same** f32 math the trainer's forward uses
//! ([`reconstruct_lora_delta`](candle_gen::train::lora::reconstruct_lora_delta) /
//! [`reconstruct_lokr_delta`](candle_gen::train::lora::reconstruct_lokr_delta)), so a candle-trained
//! adapter round-trips exactly.
//!
//! **MoE.** The A14B is two experts (`transformer/` high-noise, `transformer_2/` low-noise). A trained
//! Wan MoE LoRA ships as a `{stem}.high_noise` / `{stem}.low_noise` pair; the worker tags each
//! [`AdapterSpec`] with [`MoeExpert`](candle_gen::gen_core::MoeExpert) so the high file merges onto the
//! high expert and the low onto the low. This module merges whatever specs it is handed into one map;
//! the per-expert routing (filter by `moe_expert`) lives in [`crate::wan14b`].
//!
//! **Key conventions.** The candle trainer writes **bare** dotted PEFT/LoKr keys (no prefix). Community
//! Wan LoRAs carry a `diffusion_model.` / `transformer.` namespace (the diffusers/sd-scripts exports) or
//! the kohya `lora_unet_<flattened>` form; all resolve. Out-of-surface keys are counted in
//! [`MergeReport`] (so a zero-match spec list hard-errors rather than silently no-op'ing), but the
//! populated report is *discarded* at the call site — F-051 (sc-9035) ratified silent library-side
//! merges (no per-merge stderr), matching the Z-Image/sd3/qwen-image-edit twins.

use std::collections::{BTreeMap, HashMap, HashSet};

use candle_gen::candle_core::{DType, Tensor};
use candle_gen::gen_core::{AdapterKind, AdapterSpec, MoeExpert};
use candle_gen::train::lora::{reconstruct_lokr_delta, reconstruct_lora_delta};
// The shared adapter-merge skeleton (sc-8998 / F-018): the format-parsing + merge-report primitives
// this crate previously hand-copied. Only the Wan-specific key→module resolution stays local below.
use candle_gen::train::merge::{
    build_kohya_table, merge_into, no_target_matched, read_adapter, read_scalar, AdapterFile,
    LoraTriple, Role,
};
// Re-exported so `candle_gen_wan::MergeReport` (the crate's public surface) keeps resolving.
pub use candle_gen::train::merge::MergeReport;
use candle_gen::{CandleError, Result};

use crate::transformer::WanTransformer;

/// LoRA-key namespace prefixes a Wan adapter may carry, longest-first so the more specific peft form
/// wins. The candle trainer writes bare keys (matched by the trailing `""`).
const LORA_PREFIXES: [&str; 5] = [
    "base_model.model.diffusion_model.",
    "base_model.model.",
    "diffusion_model.",
    "transformer.",
    "",
];
/// kohya / sd-scripts community LoRA key prefix (the flattened-module form).
const KOHYA_PREFIX: &str = "lora_unet_";

/// LoKr per-module factor suffixes, longest-first so `.lokr_w1_a` wins over `.lokr_w1`.
const LOKR_SUFFIXES: [&str; 6] = [
    ".lokr_w1_a",
    ".lokr_w1_b",
    ".lokr_w1",
    ".lokr_w2_a",
    ".lokr_w2_b",
    ".lokr_w2",
];

/// Strip the longest matching [`LORA_PREFIXES`] namespace from a dotted key (or return it unchanged for
/// a bare key).
fn strip_lora_prefix(key: &str) -> &str {
    for p in LORA_PREFIXES {
        if let Some(rem) = key.strip_prefix(p) {
            return rem;
        }
    }
    key
}

/// Translate a **native-Wan** LoRA module path into the crate's **diffusers** key namespace (sc-10026).
/// Community / distill Wan LoRAs — notably the mandatory lightx2v **Lightning** distill — are keyed on
/// the native `WanModel` submodule names (`self_attn`/`cross_attn`, bare `q`/`k`/`v`/`o`, the FFN
/// `Sequential` indices `ffn.0`/`ffn.2`), but the candle [`WanTransformer`](crate::transformer) is a port
/// of the diffusers `WanTransformer3DModel` (`attn1`/`attn2`, `to_q`/`to_k`/`to_v`/`to_out.0`,
/// `ffn.net.0.proj`/`ffn.net.2`). Without this rename a native-keyed LoRA resolves to a path no
/// projection carries → the additive install matches nothing (the mlx Wan path needs no translation — it
/// *is* native-keyed). A path already in the diffusers namespace passes through unchanged: none of the
/// native tokens (`self_attn`/`cross_attn`/trailing bare `q/k/v/o`/`ffn.0`/`ffn.2`) appear in a diffusers
/// key (its attention leaf is `to_q`…`to_out.0`, never a bare `q`).
fn translate_wan_native_key(path: &str) -> String {
    let mut p = path
        .replace(".self_attn.", ".attn1.")
        .replace(".cross_attn.", ".attn2.");
    // Native FFN `Sequential` indices → the diffusers `FeedForward` submodule names.
    if let Some(base) = p.strip_suffix(".ffn.0") {
        p = format!("{base}.ffn.net.0.proj");
    } else if let Some(base) = p.strip_suffix(".ffn.2") {
        p = format!("{base}.ffn.net.2");
    }
    // Native bare attention projection names → diffusers `to_*`, matching only the trailing segment. A
    // diffusers key's attention leaf is `to_q`/`to_k`/`to_v` or `to_out`.`0` — none ends in a bare
    // `.q`/`.k`/`.v`/`.o`, so this can never double-map an already-diffusers key.
    for (native, diff) in [
        (".q", ".to_q"),
        (".k", ".to_k"),
        (".v", ".to_v"),
        (".o", ".to_out.0"),
    ] {
        if let Some(base) = p.strip_suffix(native) {
            p = format!("{base}{diff}");
            break;
        }
    }
    p
}

/// Map one LoRA key to `(diffusers_dotted_path, role)`, or `None` if outside the DiT merge surface.
/// kohya (`lora_unet_<flat>…`) resolves the flattened stem via `table`; the dotted forms (bare or
/// namespaced) resolve directly, with a native-Wan → diffusers submodule rename
/// ([`translate_wan_native_key`]) so the lightx2v Lightning distill (native-keyed) lands on the
/// diffusers projections.
fn classify_lora_key(key: &str, table: &BTreeMap<String, String>) -> Option<(String, Role)> {
    // A bundled text-encoder adapter (`lora_te*` / `…text_encoder.…`) is never a DiT target — reject it
    // up front so the permissive dotted branch below (which accepts a bare path) can't mis-route it.
    if key.starts_with("lora_te") || key.contains("text_encoder") {
        return None;
    }
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
            return Some((translate_wan_native_key(path), role));
        }
    }
    None
}

/// Map one LoKr factor key to `(diffusers_dotted_path, factor_name)`, or `None` if out of surface.
fn classify_lokr_key(
    key: &str,
    table: &BTreeMap<String, String>,
) -> Option<(String, &'static str)> {
    for suf in LOKR_SUFFIXES {
        if let Some(stem) = key.strip_suffix(suf) {
            let factor = &suf[1..];
            return if let Some(flat) = stem.strip_prefix(KOHYA_PREFIX) {
                table.get(flat).map(|d| (d.clone(), factor))
            } else {
                Some((translate_wan_native_key(strip_lora_prefix(stem)), factor))
            };
        }
    }
    None
}

/// Merge one LoRA file into `base` at `scale`: classify every key, fold complete `(down, up)` pairs
/// into `{path}.weight`. `rank` is `A`'s leading dim; `alpha` the per-target `.alpha` (default `rank`).
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

    for (path, t) in triples {
        let (Some(down), Some(up)) = (t.down, t.up) else {
            report.skipped_keys += 1; // half-pair
            continue;
        };
        if down.dims().len() != 2 || up.dims().len() != 2 {
            report.skipped_keys += 1; // Wan adapts attention Linears only (no conv surface)
            continue;
        }
        let base_key = format!("{path}.weight");
        if !base.contains_key(&base_key) {
            report.skipped_keys += 1;
            continue;
        }
        let rank = down.dims()[0] as f32;
        let alpha = t.alpha.unwrap_or(rank);
        let delta = reconstruct_lora_delta(&down, &up, alpha, rank, scale)?;
        merge_into(base, &base_key, &delta, report)?;
    }
    Ok(())
}

/// Merge one LoKr file into `base` at `scale`: `rank`/`alpha` from file metadata, per-module factors
/// grouped, `δ = (alpha/rank)·kron(w1,w2)·scale` reconstructed and merged.
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
    }
    Ok(())
}

/// Fold every adapter spec in `specs` into one expert's base DiT tensor `map` (CPU, native dtype) at
/// each spec's `scale` — LoRA and LoKr, merged into the dense weights (`W += δ`). Returns the
/// [`MergeReport`]; errors if a non-empty spec list matches **no** target (a format/prefix
/// misconfiguration — the worker should fall back rather than render an unadapted video silently).
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
                if af.declares_lokr() {
                    return Err(CandleError::Msg(format!(
                        "wan: adapter {} declared Lora but its metadata says networkType=lokr",
                        spec.path.display()
                    )));
                }
                merge_lora_file(map, &af, spec.scale, &table, &mut report)?;
            }
        }
    }
    if report.merged == 0 {
        return Err(no_target_matched(
            "wan",
            "expected PEFT `[diffusion_model.|transformer.]<path>.lora_A/B.weight` or kohya \
             `lora_unet_<flat>.lora_down/up.weight` (LoRA), or `<module>.lokr_w1/w2` with \
             networkType=lokr (LoKr), targeting `blocks.<i>.attn1/attn2.{to_q,to_k,to_v,to_out.0}`",
            specs.len(),
        ));
    }
    Ok(report)
}

// ---- Unmerged additive-LoRA path (sc-10094, epic 10043) -----------------------------------------
//
// A packed q4/q8 tier has no dense `W` to fold `δ` into (`merge_adapters` above folds `W += δ` before
// the DiT is built), so on a packed tier the mandatory Lightning distill (and user LoRAs) apply as a
// **forward-time residual** on the packed [`QLinear`] instead: `y = base(x) + scale·((x·A)·B)`
// ([`QLinear::push_lora`](crate::quant::QLinear::push_lora)), the base weight used AS-IS — no dense
// weight is materialized, so a q4 base keeps its q4 footprint. The candle twin of mlx-gen's
// `apply_wan_adapters_additive` (sc-10044). **LoKr/LoHa on a packed tier is rejected** (its residual
// needs the base's dense grid — deferred to sc-10050/10051); on a dense tier those still fold through
// [`merge_adapters`]. `install_additive` inverts the mlx flow (walk the host once, look each projection
// up in the resolved map) — the same result, but no path→field string routing.

/// Report of an additive-adapter install (sc-10094): how many projections received a residual, plus the
/// resolved target paths that matched **no** projection on the host and the off-surface adapter keys —
/// both surfaced, never silently dropped. The caller aggregates `applied` across the A14B's two experts
/// and raises the zero-match error (a non-empty adapter set that adapts nothing is a misconfiguration).
#[derive(Debug, Default)]
pub struct AdditiveReport {
    /// Projections that received a residual (one per `(path, file)` hit; multiple stack).
    pub applied: usize,
    /// Resolved target paths present in the adapter file(s) but absent from the host DiT surface.
    pub skipped_targets: Vec<String>,
    /// Adapter-file keys outside the LoRA surface (text-encoder keys, half-pairs, non-2-D factors).
    pub skipped_keys: usize,
}

/// A LoRA residual resolved from a file, pending attachment to a host projection: `a = downᵀ`
/// `[in, rank]`, `b = upᵀ·(alpha/rank)` `[rank, out]` (the `alpha/rank` ratio folded into `b`, matching
/// the fold path's split), `scale` = the spec's user strength. Held f32 (the trainer/merge dtype).
struct PendingLora {
    a: Tensor,
    b: Tensor,
    scale: f64,
}

/// True when a file carries LoKr/LoHa factors (or is stamped/declared LoKr) — such an adapter cannot
/// apply additively on a **packed** tier (its residual needs the base's dense grid; deferred to
/// sc-10050/10051), and on a dense tier it belongs on the [`merge_adapters`] fold path.
fn is_lokr_or_loha(af: &AdapterFile, kind: AdapterKind) -> bool {
    kind == AdapterKind::Lokr
        || af.declares_lokr()
        || af.tensors.keys().any(|k| {
            k.contains(".lokr_")
                || k.contains(".hada_")
                || k.starts_with("lokr_")
                || k.starts_with("hada_")
        })
}

/// Resolve one LoRA file's complete `(down, up[, alpha])` groups into per-path [`PendingLora`]s at
/// `scale`, mirroring [`merge_lora_file`]'s classification + `alpha/rank` split — but producing the
/// **unmerged** `a`/`b` factors instead of folding a delta. `table` is the host-derived kohya
/// `flat→dotted` map (built from [`WanTransformer::adaptable_paths`], since a packed tier has no dense
/// base map to build it from).
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
    for (path, t) in triples {
        let (Some(down), Some(up)) = (t.down, t.up) else {
            *skipped_keys += 1; // half-pair
            continue;
        };
        if down.dims().len() != 2 || up.dims().len() != 2 {
            *skipped_keys += 1; // Wan adapts Linears only (no conv surface)
            continue;
        }
        let rank = down.dims()[0] as f64;
        if rank == 0.0 {
            *skipped_keys += 1;
            continue;
        }
        let alpha = t.alpha.map(|a| a as f64).unwrap_or(rank);
        let ratio = alpha / rank;
        // a = downᵀ [in, rank]; b = upᵀ·(alpha/rank) [rank, out]. f32, contiguous for the matmul.
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

/// Install `specs` as forward-time additive residuals on an already-built [`WanTransformer`] (sc-10094)
/// — the packed-tier path where [`merge_adapters`] can't fold. Shared (`moe_expert == None`) specs apply
/// to any `expert`; expert-tagged specs apply only to their expert (the A14B MoE routing), shared ones
/// stacked first. **LoKr/LoHa on a packed tier is a hard, actionable error** (deferred to
/// sc-10050/10051); on a dense base such an adapter belongs on [`merge_adapters`]. Returns the per-call
/// [`AdditiveReport`] **without** raising the zero-match error — the caller aggregates `applied` across
/// the two experts and decides (a High-only spec legitimately adapts nothing on the Low expert).
pub fn install_additive(
    dit: &mut WanTransformer,
    specs: &[AdapterSpec],
    expert: MoeExpert,
) -> Result<AdditiveReport> {
    let packed = dit.is_packed();
    let table: BTreeMap<String, String> = dit
        .adaptable_paths()
        .into_iter()
        .map(|p| (p.replace('.', "_"), p))
        .collect();

    // Resolve shared specs first, then this expert's — so shared residuals stack before expert ones
    // (mlx pass-1/pass-2 order).
    let ordered = specs
        .iter()
        .filter(|s| s.moe_expert.is_none())
        .chain(specs.iter().filter(|s| s.moe_expert == Some(expert)));

    let mut pending: BTreeMap<String, Vec<PendingLora>> = BTreeMap::new();
    let mut report = AdditiveReport::default();
    for spec in ordered {
        let af = read_adapter(&spec.path)?;
        if is_lokr_or_loha(&af, spec.kind) {
            let why = if packed {
                "on a quantized (packed q4/q8) Wan tier are not supported yet — use the bf16 tier \
                 (where they fold into the dense weight), or a plain LoRA (which applies additively \
                 on any tier); tracked in sc-10050 (LoKr) / sc-10051 (LoHa)"
            } else {
                "reached the additive path on a dense base — fold them via merge_adapters instead"
            };
            return Err(CandleError::Msg(format!(
                "wan: LoKr/LoHa adapters {why}. Offending file: {}",
                spec.path.display()
            )));
        }
        resolve_lora_file(
            &af,
            spec.scale,
            &table,
            &mut pending,
            &mut report.skipped_keys,
        )?;
    }

    // Attach: walk the host once, pushing any resolved residual for each projection's path. A factor
    // whose dims don't match the target projection (`a` `[in, r]`, `b` `[r, out]`) is surfaced as a
    // skipped key, never a crashing forward — the additive analog of the fold path's `merge_into`
    // shape guard (a mismatched community LoRA target is skipped, not fatal). The factors are read on the
    // CPU (`read_adapter`) but the base weight lives on the DiT's device (e.g. CUDA on a packed tier), so
    // they are moved onto it once here — else the forward-time residual matmul is a device mismatch.
    let device = dit.device().clone();
    let mut matched: HashSet<String> = HashSet::new();
    dit.visit_adaptable_mut(&mut |path, lin| {
        if let Some(list) = pending.get(path) {
            matched.insert(path.to_string());
            let (out_f, in_f) = lin.base_shape();
            for p in list {
                if p.a.dims()[0] != in_f || p.b.dims()[1] != out_f {
                    report.skipped_keys += 1; // shape-mismatched factor for this projection
                    continue;
                }
                lin.push_lora(p.a.to_device(&device)?, p.b.to_device(&device)?, p.scale);
                report.applied += 1;
            }
        }
        Ok(())
    })?;
    // Pending targets absent from the host surface are surfaced, never silently dropped.
    for path in pending.keys() {
        if !matched.contains(path) {
            report.skipped_targets.push(path.clone());
        }
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::{DType, Device};

    /// A tiny stand-in for one expert's DiT tensor map: the four attention projections of block 0.
    fn base_map() -> HashMap<String, Tensor> {
        let dev = Device::Cpu;
        let mut m = HashMap::new();
        for leaf in ["attn1.to_q", "attn1.to_out.0", "attn2.to_k"] {
            m.insert(
                format!("blocks.0.{leaf}.weight"),
                Tensor::zeros((4, 4), DType::F16, &dev).unwrap(),
            );
        }
        m
    }

    fn t2(data: &[f32], r: usize, c: usize) -> Tensor {
        Tensor::from_vec(data.to_vec(), (r, c), &Device::Cpu).unwrap()
    }

    /// The bare candle-trainer key, a `diffusion_model.`-namespaced community key, and a kohya
    /// flattened stem all resolve to the same dotted path.
    #[test]
    fn classify_lora_resolves_bare_namespaced_and_kohya() {
        let table = build_kohya_table(&base_map(), &[2]);
        let (p, _) = classify_lora_key("blocks.0.attn1.to_q.lora_A.weight", &table).unwrap();
        assert_eq!(p, "blocks.0.attn1.to_q");
        let (p, _) = classify_lora_key(
            "diffusion_model.blocks.0.attn1.to_q.lora_down.weight",
            &table,
        )
        .unwrap();
        assert_eq!(p, "blocks.0.attn1.to_q");
        let (p, _) =
            classify_lora_key("lora_unet_blocks_0_attn1_to_out_0.lora_up.weight", &table).unwrap();
        assert_eq!(p, "blocks.0.attn1.to_out.0");
        // text-encoder keys are out of the DiT surface.
        assert!(
            classify_lora_key("lora_te_text_model_layers_0_q.lora_down.weight", &table).is_none()
        );
    }

    /// A **native-Wan**-keyed LoRA (the lightx2v Lightning distill: `self_attn`/`cross_attn`, bare
    /// `q/k/v/o`, `ffn.0`/`ffn.2`, under a `diffusion_model.` namespace) is translated onto the crate's
    /// diffusers projections (sc-10026) — without this the mandatory 4-step distill matches no projection
    /// and the additive install zero-matches.
    #[test]
    fn classify_lora_translates_native_wan_keys() {
        let table = build_kohya_table(&base_map(), &[2]);
        for (key, want) in [
            (
                "diffusion_model.blocks.0.self_attn.q.lora_down.weight",
                "blocks.0.attn1.to_q",
            ),
            (
                "diffusion_model.blocks.3.cross_attn.k.lora_up.weight",
                "blocks.3.attn2.to_k",
            ),
            (
                "diffusion_model.blocks.1.cross_attn.o.lora_down.weight",
                "blocks.1.attn2.to_out.0",
            ),
            (
                "diffusion_model.blocks.2.self_attn.v.lora_up.weight",
                "blocks.2.attn1.to_v",
            ),
            (
                "diffusion_model.blocks.0.ffn.0.lora_down.weight",
                "blocks.0.ffn.net.0.proj",
            ),
            (
                "diffusion_model.blocks.0.ffn.2.lora_up.weight",
                "blocks.0.ffn.net.2",
            ),
        ] {
            let (p, _) = classify_lora_key(key, &table).expect(key);
            assert_eq!(p, want, "native key {key}");
        }
        // An already-diffusers key is untouched by the translator (no native tokens to rename).
        let (p, _) = classify_lora_key("blocks.5.attn2.to_out.0.lora_A.weight", &table).unwrap();
        assert_eq!(p, "blocks.5.attn2.to_out.0");
    }

    /// PEFT LoRA merges into `W += (alpha/rank)·scale·B·A`; base is zero so the merged weight IS ΔW.
    #[test]
    fn merge_lora_folds_expected_delta() {
        let mut map = base_map();
        let down = t2(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], 2, 4);
        let up = t2(&[2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0], 4, 2);
        let af = AdapterFile {
            tensors: HashMap::from([
                (
                    "blocks.0.attn1.to_q.lora_A.weight".to_string(),
                    down.clone(),
                ),
                ("blocks.0.attn1.to_q.lora_B.weight".to_string(), up.clone()),
                (
                    "blocks.0.attn1.to_q.alpha".to_string(),
                    Tensor::from_vec(vec![4.0f32], (1,), &Device::Cpu).unwrap(),
                ),
            ]),
            meta: HashMap::new(),
        };
        let table = build_kohya_table(&map, &[2]);
        let mut report = MergeReport::default();
        merge_lora_file(&mut map, &af, 1.0, &table, &mut report).unwrap();
        assert_eq!(report.merged, 1);
        let merged = map
            .get("blocks.0.attn1.to_q.weight")
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let expected = reconstruct_lora_delta(&down, &up, 4.0, 2.0, 1.0).unwrap();
        let diff = (merged - expected)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff < 1e-4, "merged weight off by {diff}");
    }

    /// LoKr merges `δ = (alpha/rank)·kron(w1,w2)` into the dense weight, reading rank/alpha from meta.
    #[test]
    fn merge_lokr_folds_kron_delta() {
        let mut map = base_map();
        let w1 = t2(&[1.0, 0.0, 0.0, 1.0], 2, 2);
        let w2 = t2(&[0.5, 0.0, 0.0, 0.5], 2, 2);
        let af = AdapterFile {
            tensors: HashMap::from([
                ("blocks.0.attn2.to_k.lokr_w1".to_string(), w1.clone()),
                ("blocks.0.attn2.to_k.lokr_w2".to_string(), w2.clone()),
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
            .get("blocks.0.attn2.to_k.weight")
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
        assert!(diff < 1e-4, "merged lokr weight off by {diff}");
    }

    /// sc-9027 / F-043: a partially-matching LoRA (one on-surface target + one off-surface) merges the
    /// hit and counts the miss in [`MergeReport::skipped_keys`], so the merge machinery distinguishes a
    /// partial match from a total miss (the latter hard-errors). The caller discards the report (F-051),
    /// so this asserts the report contents directly from the merge, not any call-site side effect.
    #[test]
    fn merge_lora_partial_match_reports_skipped() {
        let mut map = base_map();
        let down = t2(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], 2, 4);
        let up = t2(&[2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0], 4, 2);
        let af = AdapterFile {
            tensors: HashMap::from([
                // On-surface: block 0 attn1.to_q exists in `base_map`.
                (
                    "blocks.0.attn1.to_q.lora_A.weight".to_string(),
                    down.clone(),
                ),
                ("blocks.0.attn1.to_q.lora_B.weight".to_string(), up.clone()),
                // Off-surface: block 99 is not in the base map — the miss must be counted, not dropped.
                ("blocks.99.attn1.to_q.lora_A.weight".to_string(), down),
                ("blocks.99.attn1.to_q.lora_B.weight".to_string(), up),
            ]),
            meta: HashMap::new(),
        };
        let table = build_kohya_table(&map, &[2]);
        let mut report = MergeReport::default();
        merge_lora_file(&mut map, &af, 1.0, &table, &mut report).unwrap();
        assert_eq!(report.merged, 1, "the on-surface target merges");
        assert!(
            report.skipped_keys >= 1,
            "the off-surface target is surfaced as skipped, not silently dropped"
        );
    }

    /// A non-empty spec list that matches nothing surfaces as zero-merged (the public entry then errors).
    #[test]
    fn merge_lora_nothing_matched_is_zero() {
        let mut map = base_map();
        let af = AdapterFile {
            tensors: HashMap::from([(
                "blocks.99.attn1.to_q.lora_A.weight".to_string(),
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

    // ---- additive-adapter install (sc-10094) ----------------------------------------------------

    use crate::config::TransformerConfig;
    use crate::rope::WanRope;
    use candle_gen::candle_core::safetensors as cst;
    use candle_gen::candle_nn::{VarBuilder, VarMap};

    /// A tiny Wan-shaped config (z16, 2 layers, 1 head, head_dim 128) — the dit_train test shape,
    /// exercising every vendored DiT path cheaply on CPU.
    fn tiny_cfg() -> TransformerConfig {
        TransformerConfig {
            in_channels: 16,
            out_channels: 16,
            num_layers: 2,
            num_heads: 1,
            head_dim: 128,
            dim: 128,
            ffn_dim: 256,
            freq_dim: 256,
            text_dim: 64,
            patch: (1, 2, 2),
            eps: 1e-6,
            rope_theta: 10000.0,
            rope_max_seq_len: 1024,
        }
    }

    /// A fresh, **randomized** dense base tensor map for `cfg` — build a `WanTransformer` once to
    /// populate every key, randomize all vars (a zero patch kernel makes the forward vacuous), then
    /// snapshot the varmap to a CPU f32 map both the additive and the fold path can build from.
    fn random_base_map(cfg: &TransformerConfig, dev: &Device) -> HashMap<String, Tensor> {
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, dev);
        let _ = WanTransformer::new(cfg, vb).unwrap(); // populate keys/shapes
        for v in vm.all_vars() {
            v.set(&Tensor::randn(0f32, 0.1f32, v.dims(), dev).unwrap())
                .unwrap();
        }
        let data = vm.data().lock().unwrap();
        data.iter()
            .map(|(k, v)| (k.clone(), v.as_tensor().clone()))
            .collect()
    }

    fn dit_from_map(
        cfg: &TransformerConfig,
        map: HashMap<String, Tensor>,
        dev: &Device,
    ) -> WanTransformer {
        WanTransformer::new(cfg, VarBuilder::from_tensors(map, DType::F32, dev)).unwrap()
    }

    /// Fixed DiT inputs `(latent, umt5, cos, sin)` so two DiTs can be compared on the *same* forward.
    fn fixed_inputs(cfg: &TransformerConfig, dev: &Device) -> (Tensor, Tensor, Tensor, Tensor) {
        let latent = Tensor::randn(0f32, 1f32, (1, cfg.in_channels, 1, 4, 4), dev).unwrap();
        let umt5 = Tensor::randn(0f32, 1f32, (1, 3, cfg.text_dim), dev).unwrap();
        let (cos, sin) = WanRope::new(cfg).cos_sin(1, 2, 2, dev).unwrap();
        (latent, umt5, cos, sin)
    }

    fn dit_forward(dit: &WanTransformer, inp: &(Tensor, Tensor, Tensor, Tensor)) -> Tensor {
        let (latent, umt5, cos, sin) = inp;
        let ctx = dit.embed_text(umt5).unwrap();
        dit.forward(latent, &ctx, 500.0, cos, sin).unwrap()
    }

    fn max_abs_dev(a: &Tensor, b: &Tensor) -> f32 {
        a.sub(b)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    /// Write a PEFT LoRA `.safetensors` targeting `paths` (each `[out,in]` = `[dim,dim]`), rank `r`,
    /// per-target `.alpha`.
    fn write_lora(
        paths: &[&str],
        dim: usize,
        r: usize,
        alpha: f32,
        tag: &str,
    ) -> std::path::PathBuf {
        let dev = Device::Cpu;
        let mut m: HashMap<String, Tensor> = HashMap::new();
        for p in paths {
            m.insert(
                format!("{p}.lora_A.weight"),
                Tensor::randn(0f32, 1f32, (r, dim), &dev).unwrap(),
            );
            m.insert(
                format!("{p}.lora_B.weight"),
                Tensor::randn(0f32, 1f32, (dim, r), &dev).unwrap(),
            );
            m.insert(
                format!("{p}.alpha"),
                Tensor::from_vec(vec![alpha], (1,), &dev).unwrap(),
            );
        }
        let path = std::env::temp_dir().join(format!(
            "sc10094_lora_{tag}_{}.safetensors",
            std::process::id()
        ));
        cst::save(&m, &path).unwrap();
        path
    }

    /// The additive install on a dense base reproduces the `merge_adapters` fold: same base weights +
    /// same LoRA → the two DiT forwards match tightly (f32), on the *same* inputs. The sc-10094
    /// additive==folded acceptance, end-to-end through the vendored DiT (not just the bare `QLinear`).
    #[test]
    fn install_additive_matches_merge_fold() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let base = random_base_map(&cfg, &dev);
        // Square (dim×dim) attention projections — the uniform-shape write_lora fixture; the FFN
        // net.2 (in=ffn_dim≠out=dim) needs a shape-matched factor, covered by the shape guard.
        let targets = [
            "blocks.0.attn1.to_q",
            "blocks.1.attn2.to_v",
            "blocks.1.attn1.to_k",
        ];
        let lora = write_lora(&targets, cfg.dim, 4, 8.0, "parity");
        let specs = vec![AdapterSpec::new(lora.clone(), 0.8, AdapterKind::Lora)];

        // Additive: build from the base, install forward-time residuals.
        let mut dit_add = dit_from_map(&cfg, base.clone(), &dev);
        let report = install_additive(&mut dit_add, &specs, MoeExpert::High).unwrap();
        assert_eq!(report.applied, targets.len(), "every target adapts");
        assert!(
            report.skipped_targets.is_empty(),
            "no target missed the host"
        );

        // Fold: merge the same delta into the base map, build from the merged map.
        let mut merged = base.clone();
        merge_adapters(&mut merged, &specs).unwrap();
        let dit_fold = dit_from_map(&cfg, merged, &dev);

        let inp = fixed_inputs(&cfg, &dev);
        let dev_max = max_abs_dev(&dit_forward(&dit_add, &inp), &dit_forward(&dit_fold, &inp));
        assert!(
            dev_max < 1e-3,
            "additive DiT forward vs folded deviates by {dev_max}"
        );
        std::fs::remove_file(&lora).ok();
    }

    /// A LoRA installs additively onto the tiny DiT and **shifts** the forward vs the un-adapted base
    /// (same inputs); the report accounts for every target with no host miss.
    #[test]
    fn install_additive_applies_and_shifts() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let base = random_base_map(&cfg, &dev);
        let targets = ["blocks.0.attn1.to_q", "blocks.1.attn2.to_v"];
        let lora = write_lora(&targets, cfg.dim, 4, 8.0, "shift");
        let specs = vec![AdapterSpec::new(lora.clone(), 1.0, AdapterKind::Lora)];

        let dit_base = dit_from_map(&cfg, base.clone(), &dev);
        let mut dit_add = dit_from_map(&cfg, base, &dev);
        let report = install_additive(&mut dit_add, &specs, MoeExpert::High).unwrap();
        assert_eq!(report.applied, 2);
        assert!(report.skipped_targets.is_empty());

        let inp = fixed_inputs(&cfg, &dev);
        let shift = max_abs_dev(&dit_forward(&dit_base, &inp), &dit_forward(&dit_add, &inp));
        assert!(
            shift > 1e-4,
            "additive LoRA did not shift the DiT forward ({shift})"
        );
        std::fs::remove_file(&lora).ok();
    }

    /// MoE routing: a High-tagged spec adapts nothing when installed for the Low expert, and adapts its
    /// targets when installed for the High expert.
    #[test]
    fn install_additive_moe_routing() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let base = random_base_map(&cfg, &dev);
        let lora = write_lora(&["blocks.0.attn1.to_q"], cfg.dim, 4, 4.0, "moe");
        let specs =
            vec![AdapterSpec::new(lora.clone(), 1.0, AdapterKind::Lora)
                .with_moe_expert(MoeExpert::High)];

        let mut low = dit_from_map(&cfg, base.clone(), &dev);
        let low_report = install_additive(&mut low, &specs, MoeExpert::Low).unwrap();
        assert_eq!(
            low_report.applied, 0,
            "a High spec must not touch the Low expert"
        );

        let mut high = dit_from_map(&cfg, base, &dev);
        let high_report = install_additive(&mut high, &specs, MoeExpert::High).unwrap();
        assert_eq!(high_report.applied, 1, "a High spec adapts the High expert");
        std::fs::remove_file(&lora).ok();
    }

    /// A LoKr adapter is rejected by the additive path with an actionable error (LoKr/LoHa fold on the
    /// dense path; a packed tier rejects them entirely — sc-10050/10051).
    #[test]
    fn install_additive_rejects_lokr() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let base = random_base_map(&cfg, &dev);
        let mut m: HashMap<String, Tensor> = HashMap::new();
        m.insert(
            "blocks.0.attn1.to_q.lokr_w1".into(),
            Tensor::randn(0f32, 1f32, (cfg.dim, cfg.dim), &dev).unwrap(),
        );
        m.insert(
            "blocks.0.attn1.to_q.lokr_w2".into(),
            Tensor::from_vec(vec![1.0f32], (1, 1), &dev).unwrap(),
        );
        let path =
            std::env::temp_dir().join(format!("sc10094_lokr_{}.safetensors", std::process::id()));
        cst::save(&m, &path).unwrap();
        let specs = vec![AdapterSpec::new(path.clone(), 1.0, AdapterKind::Lokr)];

        let mut dit = dit_from_map(&cfg, base, &dev);
        let err = install_additive(&mut dit, &specs, MoeExpert::High).unwrap_err();
        assert!(
            err.to_string().contains("LoKr/LoHa"),
            "expected a LoKr/LoHa rejection, got: {err}"
        );
        std::fs::remove_file(&path).ok();
    }
}
