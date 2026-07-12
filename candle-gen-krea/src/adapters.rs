//! Krea 2 inference-side adapter merge (sc-7836) — load a trained `krea_2_raw` LoRA/LoKr
//! `.safetensors` and fold its delta into the dense single-stream DiT (`transformer/`) weights
//! **before** [`crate::transformer::Krea2Transformer`] is built. The candle twin of the MLX
//! inference-merge seam (sc-7578's *engine* half) and the closing half of the native-trainer loop: a
//! LoRA produced by [`crate::training`]'s `krea_2_raw` trainer now actually loads in candle `krea_2_turbo`
//! inference. Structurally identical to [`candle_gen_z_image::merge_adapters`] (the same DiT key
//! namespace), so the well-exercised z-image classify/merge core carries over verbatim.
//!
//! **Merge, don't residual** (same rationale as Z-Image / SDXL): inference has no need to keep the
//! factors trainable, so it folds `W += δ` into the dense weight and reproduces the merged-weight
//! forward exactly. The flow-match sampler is chaos-sensitive — `(W+δ)·x` ≠ `W·x + δ·x` to ~1 ULP — so
//! a live residual would drift. The delta is reconstructed with the **same** f32 math the trainer's
//! forward uses ([`reconstruct_lora_delta`] / [`reconstruct_lokr_delta`]), so a candle-trained adapter
//! round-trips exactly.
//!
//! **Merge at the safetensors-key level.** The DiT reads its `transformer/` keys 1:1, so `{path}.weight`
//! is a valid base key for every Linear an adapter targets. The candle `krea_2_raw` trainer's own
//! default surface is the single-stream blocks' attention projections
//! (`to_q`/`to_k`/`to_v`/`to_out.0`, [`crate::train_dit::KREA_ATTN_TARGETS`]), but the *merge* surface is
//! wider — the full set of adaptable Linears MLX's host exposes (attention incl. `to_gate` + the SwiGLU
//! FFN `ff.<gate|up|down>`, across the single-stream `transformer_blocks` **and** the `text_fusion`
//! blocks, `merge_surface_keys`) — so an ai-toolkit LoKr that adapts gate + FFN folds in fully
//! (sc-8776). The Krea trainer writes **bare dotted** PEFT keys (`save_lora_peft(set, "", …)` — no
//! `base_model.model.unet.` prefix); on read we also tolerate the common community prefixes
//! ([`PEFT_PREFIXES`]), the ai-toolkit native `diffusion_model.blocks…`/`wq`/`mlp` naming
//! ([`normalize_native_krea_path`]), and a kohya `lora_transformer_<flat>` flattening resolved against
//! the base key set.
//!
//! **Family-match policy:** a `family: krea_2` adapter (`baseModel: krea_2_raw`) applies on
//! `krea_2_turbo` — there is **no base-model gating** here (the Lens / Z-Image precedent; base-model
//! gating is a `wan-video`-only worker concern). The candle engine merges whatever DiT-targeting
//! factors the file carries.
//!
//! Out-of-surface keys are **counted and surfaced** in [`MergeReport`], never silently dropped:
//! text-encoder LoRAs (this is a DiT-only merge) and conv-shaped (4-D) factors (the merge folds only
//! into the 2-D Linear projections).

use std::collections::{BTreeMap, HashMap, HashSet};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::{AdapterKind, AdapterSpec};
use candle_gen::quant::{AdaptLinear, LokrFactors};
use candle_gen::train::lora::{
    reconstruct_lokr_delta, reconstruct_lora_delta, LoraAdapterMeta, LoraLinear,
};
// The shared adapter-merge skeleton (sc-8998 / F-018): the format-parsing + merge-report primitives
// this crate previously hand-copied. Only the Krea-specific key→module resolution (ai-toolkit native
// rename + the bespoke file-meta-or-lycoris LoKr grouping) stays local below.
use candle_gen::train::merge::{
    build_kohya_table, merge_into, no_target_matched, read_adapter, read_scalar, AdapterFile,
    LoraTriple, Role,
};
// Re-exported so `candle_gen_krea::MergeReport` (the crate's public surface) keeps resolving.
pub use candle_gen::train::merge::MergeReport;
use candle_gen::{CandleError, Result};

use crate::config::Krea2Config;
use crate::loader::Weights;

/// PEFT key prefixes tolerated on read, longest-first. The candle Krea trainer writes **bare** dotted
/// paths (no prefix), but community adapters and `peft.save_pretrained()` wrap the DiT under one of
/// these; stripping them yields the same dotted module path. A key matching none is taken as-is (bare).
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

/// Rewrite a native Krea-2 / ai-toolkit (ostris) module path to the diffusers names the base DiT keys
/// use (sc-8185). ai-toolkit keys its LoRAs to the **raw checkpoint layout** — `blocks`/`txtfusion`
/// containers, `attn.{wq,wk,wv,wo,gate}`, an `mlp` FFN — whereas SceneWorks' converter/trainer (and the
/// base DiT tensor keys this merge folds into) use `transformer_blocks`/`text_fusion`,
/// `attn.{to_q,to_k,to_v,to_out.0,to_gate}`, `ff`. A path already in diffusers form is returned
/// unchanged (none of the replacements match it), so this is a no-op for our own LoRAs.
fn normalize_native_krea_path(path: &str) -> String {
    // Container (leading segment): native `blocks`/`txtfusion` → diffusers `transformer_blocks`/
    // `text_fusion`. `transformer_blocks.`/`text_fusion.` don't start with `blocks.`/`txtfusion.`, so
    // an already-diffusers path is untouched.
    let mut p = if let Some(rest) = path.strip_prefix("blocks.") {
        format!("transformer_blocks.{rest}")
    } else if let Some(rest) = path.strip_prefix("txtfusion.") {
        format!("text_fusion.{rest}")
    } else {
        path.to_string()
    };
    // FFN container, then the attention leaf names.
    p = p.replace(".mlp.", ".ff.");
    p = p
        .replace(".attn.wq", ".attn.to_q")
        .replace(".attn.wk", ".attn.to_k")
        .replace(".attn.wv", ".attn.to_v")
        .replace(".attn.wo", ".attn.to_out.0")
        .replace(".attn.gate", ".attn.to_gate");
    p
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
            return Some((normalize_native_krea_path(path), role));
        }
    }
    None
}

/// Resolve a LoKr module *stem* (the key with its `.lokr_*` / `.alpha` suffix already removed) to a
/// DiT dotted path: a kohya `lora_transformer_<flat>` stem via `table`, else PEFT/bare/native resolved
/// through the prefix strip + ai-toolkit rename. Shared by the factor and `.alpha` classifiers so a
/// per-target `.alpha` groups under the same path as its factors.
fn resolve_lokr_module(stem: &str, table: &BTreeMap<String, String>) -> Option<String> {
    if let Some(flat) = stem.strip_prefix(KOHYA_PREFIX) {
        table.get(flat).cloned()
    } else {
        Some(normalize_native_krea_path(strip_peft_prefix(stem)))
    }
}

/// Map one LoKr factor key to `(dit_dotted_path, factor_name)`, or `None` if out of surface.
fn classify_lokr_key(
    key: &str,
    table: &BTreeMap<String, String>,
) -> Option<(String, &'static str)> {
    for suf in LOKR_SUFFIXES {
        if let Some(stem) = key.strip_suffix(suf) {
            let factor = &suf[1..]; // drop the leading '.'
            return resolve_lokr_module(stem, table).map(|d| (d, factor));
        }
    }
    None
}

/// `true` if any tensor key is a LoKr factor (`*.lokr_w…`), regardless of `networkType` metadata —
/// how a **third-party** LyCORIS LoKr (ai-toolkit / kohya / lycoris-lib) is recognized (those files
/// ship the Kronecker factors but not the peft `networkType=lokr` stamp). Mirrors MLX `is_lokr_keys`.
fn has_lokr_keys(af: &AdapterFile) -> bool {
    af.tensors
        .keys()
        .any(|k| LOKR_SUFFIXES.iter().any(|s| k.ends_with(s)))
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
                triples.entry(path).or_default().alpha = Some(read_scalar(key, t)?)
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

/// One module's grouped LoKr factors plus its optional per-target `.alpha` scalar (the ai-toolkit /
/// lycoris layout — vs SceneWorks' peft LoKr which stamps one file-level `rank`/`alpha`).
#[derive(Default)]
struct LokrGroup {
    factors: BTreeMap<&'static str, Tensor>,
    alpha: Option<f32>,
}

impl LokrGroup {
    /// The LyCORIS factorization rank (`lora_dim`) derived from whichever factor is decomposed —
    /// `lokr_w1_a` is `[out_l, dim]` (dim = trailing), else `lokr_w2_a` is `[out_k, dim]`. `None` when
    /// **both** legs are full matrices: lycoris then forces `alpha = lora_dim` ⇒ scale 1 (ai-toolkit
    /// `LokrModule.__init__`: `if use_w1 and use_w2: alpha = lora_dim`). Mirrors MLX `ThirdPartyLokr`.
    fn rank(&self) -> Option<f32> {
        if let Some(a) = self.factors.get("lokr_w1_a") {
            return Some(a.dims()[1] as f32);
        }
        self.factors.get("lokr_w2_a").map(|a| a.dims()[1] as f32)
    }
}

/// Merge one LoKr file into `base` at `scale`, `δ = (alpha/rank)·kron(w1,w2)·scale` per module.
///
/// Two `(alpha, rank)` sources, matching MLX's split (`parse_lokr` vs `parse_lokr_thirdparty`):
/// SceneWorks' / candle-trainer **peft** LoKr stamps one file-level `rank`/`alpha` (alpha defaults to
/// rank) applied to every target — preferred when present so a candle-trained adapter round-trips.
/// A **third-party** LyCORIS file (ai-toolkit / lycoris) stamps neither and instead carries a
/// per-target `.alpha` tensor; then rank/alpha/scale are derived **per module** ([`LokrGroup`]):
/// rank from a decomposed factor's inner dim, alpha from the `.alpha` tensor, and the both-full case
/// (`rank() == None`) forced to scale 1 — the ai-toolkit convention (sc-8776).
fn merge_lokr_file(
    base: &mut HashMap<String, Tensor>,
    af: &AdapterFile,
    scale: f32,
    table: &BTreeMap<String, String>,
    report: &mut MergeReport,
) -> Result<()> {
    let file_rank = af.meta.get("rank").and_then(|s| s.parse::<f32>().ok());
    let file_alpha = af.meta.get("alpha").and_then(|s| s.parse::<f32>().ok());
    let has_file_meta = file_rank.is_some() || file_alpha.is_some();

    let mut grouped: BTreeMap<String, LokrGroup> = BTreeMap::new();
    for (key, t) in &af.tensors {
        // Per-target `.alpha` scalar (ai-toolkit / lycoris) — group under the same DiT path as its
        // factors so it can inform that module's scale.
        if let Some(stem) = key.strip_suffix(".alpha") {
            match resolve_lokr_module(stem, table) {
                Some(path) => grouped.entry(path).or_default().alpha = Some(read_scalar(key, t)?),
                None => report.skipped_keys += 1,
            }
            continue;
        }
        match classify_lokr_key(key, table) {
            Some((path, factor)) => {
                grouped
                    .entry(path)
                    .or_default()
                    .factors
                    .insert(factor, t.clone());
            }
            None => report.skipped_keys += 1,
        }
    }

    for (path, g) in grouped {
        let base_key = format!("{path}.weight");
        let Some(w) = base.get(&base_key) else {
            report.skipped_keys += 1;
            continue;
        };
        if w.dims().len() != 2 {
            report.skipped_keys += 1; // conv LoKr — out of surface
            continue;
        }
        // A group with only an `.alpha` (its factors targeted a non-routable module) can't be
        // reconstructed — surface it rather than erroring on the missing `w1` leg.
        if !g.factors.contains_key("lokr_w1") && !g.factors.contains_key("lokr_w1_a") {
            report.skipped_keys += 1;
            continue;
        }
        let (out_f, in_f) = (w.dims()[0], w.dims()[1]);
        // File-level peft metadata (candle-trainer) applied uniformly; else lycoris per-target.
        let (alpha, rank) = if has_file_meta {
            let rank = file_rank.unwrap_or(1.0);
            (file_alpha.unwrap_or(rank), rank)
        } else {
            match g.rank() {
                Some(r) => (g.alpha.unwrap_or(r), r),
                None => (1.0, 1.0), // both factors full ⇒ lycoris scale 1
            }
        };
        let delta = reconstruct_lokr_delta(
            g.factors.get("lokr_w1"),
            g.factors.get("lokr_w1_a"),
            g.factors.get("lokr_w1_b"),
            g.factors.get("lokr_w2"),
            g.factors.get("lokr_w2_a"),
            g.factors.get("lokr_w2_b"),
            alpha,
            rank,
            scale,
            (out_f, in_f),
        )?;
        merge_into(base, &base_key, &delta, report)?;
    }
    Ok(())
}

/// Fold every adapter spec in `specs` into the base DiT tensor `map` (CPU, native dtype) at each spec's
/// `scale` — LoRA and LoKr, merged into the dense weights (`W += δ`). Returns the [`MergeReport`];
/// errors if a non-empty spec list matches **no** target (a format / prefix misconfiguration — the
/// worker should then fall back rather than render an unadapted image silently).
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
                        "krea: adapter {} declared Lora but its metadata says networkType=lokr",
                        spec.path.display()
                    )));
                }
                // A third-party LyCORIS LoKr (ai-toolkit / lycoris, sc-8776) carries `lokr_*` keys but
                // NO `networkType` stamp, so `classify_adapter` can't know to set kind=Lokr — sniff the
                // keys and route to the LoKr merge, mirroring MLX's `is_lokr_keys` autoprefix branch.
                if has_lokr_keys(&af) {
                    merge_lokr_file(map, &af, spec.scale, &table, &mut report)?;
                } else {
                    merge_lora_file(map, &af, spec.scale, &table, &mut report)?;
                }
            }
        }
    }
    if report.merged == 0 {
        return Err(no_target_matched(
            "krea",
            "expected bare/PEFT `<path>.lora_A/B.weight` (LoRA) or `<module>.lokr_w1/w2` (LoKr — \
             declared networkType=lokr OR sniffed by key) over the DiT attention (to_q|to_k|to_v|\
             to_gate|to_out.0) and SwiGLU FFN (ff.gate|ff.up|ff.down) projections across the \
             single-stream transformer_blocks and text_fusion blocks. Conv-layer / text-encoder \
             adapters are out of surface",
            specs.len(),
        ));
    }
    Ok(report)
}

/// The dense base-weight keys the merge targets: every adaptable Linear the merge surface can fold a
/// delta into — per block, the attention projections (`attn.<to_q|to_k|to_v|to_gate|to_out.0>`) and
/// the SwiGLU FFN (`ff.<gate|up|down>`), across the single-stream `transformer_blocks` **and** the
/// `text_fusion` `layerwise_blocks` / `refiner_blocks`. This is the full 256-Linear surface MLX's Krea
/// [`AdaptableHost`] exposes (sc-8776); the candle `krea_2_raw` trainer's own default is a subset
/// ([`crate::train_dit::KREA_ATTN_TARGETS`], the 112 attention matrices), but an ai-toolkit LoKr adapts `to_gate` + the
/// FFN too, so preloading only the attention subset would silently skip half its targets. Preloading
/// exactly this surface (rather than the whole 12B model — norms, embedders, modulation stay out)
/// bounds the merge's transient host memory while letting every trained target resolve; a key absent
/// from a given build (e.g. a 1-block test snapshot) is skipped by [`merge_into_weights`]'s
/// `w.contains` guard.
fn merge_surface_keys(cfg: &Krea2Config) -> Vec<String> {
    let mut keys = Vec::new();
    let mut block = |prefix: &str| {
        for target in ["to_q", "to_k", "to_v", "to_gate", "to_out.0"] {
            keys.push(format!("{prefix}.attn.{target}.weight"));
        }
        for target in ["gate", "up", "down"] {
            keys.push(format!("{prefix}.ff.{target}.weight"));
        }
    };
    for i in 0..cfg.num_layers {
        block(&format!("transformer_blocks.{i}"));
    }
    for i in 0..cfg.num_layerwise_text_blocks {
        block(&format!("text_fusion.layerwise_blocks.{i}"));
    }
    for i in 0..cfg.num_refiner_text_blocks {
        block(&format!("text_fusion.refiner_blocks.{i}"));
    }
    keys
}

/// Merge the LoRA/LoKr `specs` into the DiT `Weights` `w` (sc-7836): preload the attention-projection
/// base weights (`merge_surface_keys`) onto the CPU, fold each adapter's delta in
/// ([`merge_adapters`], f32 math matching the trainer), and install the result as `w`'s overlay so the
/// subsequent `Krea2Transformer::load` reads the merged weights. A no-op (empty overlay) when `specs`
/// is empty — the stock unadapted build. The engine's adapter-merge entry; [`crate::pipeline`] calls it
/// at component-load, and it is public so a real-weight smoke can assert the merge surface directly.
pub fn merge_into_weights(
    w: &mut Weights,
    cfg: &Krea2Config,
    specs: &[AdapterSpec],
) -> Result<MergeReport> {
    if specs.is_empty() {
        return Ok(MergeReport::default());
    }
    let mut map: HashMap<String, Tensor> = HashMap::new();
    for key in merge_surface_keys(cfg) {
        if w.contains(&key) {
            // Packed-aware base: on a packed tier the surface `{base}.weight` is u32 codes, so
            // reconstruct the dense grid from the packed triple before folding the delta (sc-9411).
            // On a dense tier this is the plain CPU weight. `merge_adapters` folds each targeted
            // delta into this base in place.
            map.insert(key.clone(), w.get_cpu_merge_base(&key)?);
        }
    }

    // Snapshot the preloaded base identities so, after the merge, we can install into the overlay
    // ONLY the projections a delta actually folded into (sc-9411 adapter compose). `merge_into`
    // replaces a merged key's tensor with a fresh one (new `TensorId`), so a changed id ⇔ merged.
    // Keeping untargeted keys out of the overlay lets them stay **packed** on a packed tier (the
    // overlay would otherwise force the whole reconstructed-dense DiT resident); on a dense tier the
    // untargeted keys are identical to the mmap, so dropping them is a pure memory win.
    let base_ids: HashMap<String, _> = map.iter().map(|(k, t)| (k.clone(), t.id())).collect();
    let report = merge_adapters(&mut map, specs)?;
    map.retain(|k, t| base_ids.get(k).is_none_or(|&id| t.id() != id));
    w.set_overlay(map);
    Ok(report)
}

// ---- Forward-time additive (unmerged) install on a PACKED tier (sc-11105) ------------------------
//
// On a **packed** q4/q8 Krea tier, [`merge_into_weights`] reconstructs each adapted projection's dense
// weight from the packed parts and installs it as a dense overlay — so a user LoRA forces those
// projections resident-dense. [`install_additive`] instead attaches each LoRA/LoKr as a **forward-time
// residual** on the DiT's shared [`candle_gen::quant::AdaptLinear`] projections: `y = base(x) + Σ
// scale·((x·A)·B)`, the base kept packed. So a user adapter applies on the q4/q8 tier at the base's
// footprint. The dense tier keeps folding (bit-exact) via [`merge_into_weights`]. The resolver reuses
// this crate's exact [`classify_lora_key`] / [`classify_lokr_key`] (incl. the ai-toolkit native rename)
// + the peft/lycoris [`LokrGroup`] alpha/rank rule, so the additive residual equals the fold to f32
// tolerance. Krea attention is **split** q/k/v (no fused-QKV), so each factor attaches 1:1.

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
/// [`merge_lora_file`]'s classify (incl. the ai-toolkit native rename) + effective alpha/rank exactly,
/// producing UNMERGED factors — so the additive residual equals the folded delta to f32 tolerance.
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

/// Resolve one LoKr file into per-path [`PendingLokr`] with the FULL `(alpha/rank)·scale` baked. Mirrors
/// [`merge_lokr_file`]'s two `(alpha, rank)` sources exactly — file-level peft metadata applied
/// uniformly, else the per-module lycoris [`LokrGroup`] rule (rank from a decomposed factor's inner dim,
/// alpha from the `.alpha` tensor, both-full ⇒ scale 1).
fn resolve_lokr_file(
    af: &AdapterFile,
    scale: f32,
    table: &BTreeMap<String, String>,
    pending: &mut BTreeMap<String, Vec<PendingLokr>>,
    skipped_keys: &mut usize,
) -> Result<()> {
    let file_rank = af.meta.get("rank").and_then(|s| s.parse::<f32>().ok());
    let file_alpha = af.meta.get("alpha").and_then(|s| s.parse::<f32>().ok());
    let has_file_meta = file_rank.is_some() || file_alpha.is_some();

    let mut grouped: BTreeMap<String, LokrGroup> = BTreeMap::new();
    for (key, t) in &af.tensors {
        if let Some(stem) = key.strip_suffix(".alpha") {
            match resolve_lokr_module(stem, table) {
                Some(path) => grouped.entry(path).or_default().alpha = Some(read_scalar(key, t)?),
                None => *skipped_keys += 1,
            }
            continue;
        }
        match classify_lokr_key(key, table) {
            Some((path, factor)) => {
                grouped
                    .entry(path)
                    .or_default()
                    .factors
                    .insert(factor, t.clone());
            }
            None => *skipped_keys += 1,
        }
    }

    for (path, g) in grouped {
        // A group with only an `.alpha` (its factors targeted a non-routable module) can't be built.
        if !g.factors.contains_key("lokr_w1") && !g.factors.contains_key("lokr_w1_a") {
            *skipped_keys += 1;
            continue;
        }
        let (alpha, rank) = if has_file_meta {
            let rank = file_rank.unwrap_or(1.0);
            (file_alpha.unwrap_or(rank), rank)
        } else {
            match g.rank() {
                Some(r) => (g.alpha.unwrap_or(r), r),
                None => (1.0, 1.0), // both factors full ⇒ lycoris scale 1
            }
        };
        let full = (alpha as f64 / rank as f64) * scale as f64;
        pending.entry(path).or_default().push(PendingLokr {
            w1: g.factors.get("lokr_w1").cloned(),
            w1_a: g.factors.get("lokr_w1_a").cloned(),
            w1_b: g.factors.get("lokr_w1_b").cloned(),
            w2: g.factors.get("lokr_w2").cloned(),
            w2_a: g.factors.get("lokr_w2_a").cloned(),
            w2_b: g.factors.get("lokr_w2_b").cloned(),
            scale: full,
        });
    }
    Ok(())
}

/// A projection that can host **forward-time additive** LoRA/LoKr inference residuals. Implemented by
/// BOTH the txt2img DiT's [`AdaptLinear`] leaves (dense or packed) and the control DiT's [`LoraLinear`]
/// leaves (sc-11720), so a single installer ([`install_additive`]) serves either Krea DiT. Both back the
/// residual with the SAME shared `candle_gen::quant` math (the LoRA two-small-matmul arm / the structured
/// Kronecker vec-trick for LoKr), so additive equals the old dense fold to f32 tolerance (~1 ULP).
pub trait AdditiveProj {
    /// `(out_features, in_features)` of the base projection — the shape guard each resolved factor is
    /// checked against before it is pushed.
    fn out_in(&self) -> (usize, usize);
    /// Push an additive LoRA residual `scale·((x·a)·b)` (`a`: `[in, rank]`, `b`: `[rank, out]`).
    fn add_lora(&mut self, a: Tensor, b: Tensor, scale: f64);
    /// Push an additive structured-LoKr residual (the allocation-free Kronecker form).
    fn add_lokr(&mut self, factors: LokrFactors);
}

impl AdditiveProj for AdaptLinear {
    fn out_in(&self) -> (usize, usize) {
        self.base_shape()
    }
    fn add_lora(&mut self, a: Tensor, b: Tensor, scale: f64) {
        self.push_lora(a, b, scale);
    }
    fn add_lokr(&mut self, factors: LokrFactors) {
        self.push_lokr_structured(factors);
    }
}

impl AdditiveProj for LoraLinear {
    fn out_in(&self) -> (usize, usize) {
        (self.out_features(), self.in_features())
    }
    fn add_lora(&mut self, a: Tensor, b: Tensor, scale: f64) {
        // The inherent inference-residual push (sc-11103) — NOT this trait method (distinct name).
        LoraLinear::push_additive_lora(self, a, b, scale);
    }
    fn add_lokr(&mut self, factors: LokrFactors) {
        LoraLinear::push_additive_lokr(self, factors);
    }
}

/// A Krea DiT that exposes its adaptable projection surface to [`install_additive`]. Both the txt2img
/// [`Krea2Transformer`] and the control [`KreaTrainDit`] implement it; the closure is invoked once per
/// leaf with its canonical dotted path (the key a PEFT/kohya adapter targets) and the projection as a
/// `&mut dyn AdditiveProj`, so one resolve+attach body drives either DiT regardless of its leaf type.
pub trait AdditiveDit {
    fn visit_additive(
        &mut self,
        f: &mut dyn FnMut(&str, &mut dyn AdditiveProj) -> Result<()>,
    ) -> Result<()>;
    /// The device the base weights live on (residual factors are moved onto it at push).
    fn adapter_device(&self) -> Device;
    /// The adaptable-surface description used in the "matched no target" error.
    fn adapter_surface_hint(&self) -> &'static str;
}

/// Install `specs` as **forward-time additive residuals** on a Krea DiT (sc-11105 / sc-11720): resolve
/// each LoRA/LoKr file into unmerged factors, then walk the DiT's [`AdditiveDit`] surface once pushing
/// residuals onto matched projections — the base is never dequantized or folded, so a packed q4/q8 tier
/// keeps its footprint and a dense bf16 base stays an evictable mmap while the user adapter applies.
/// Routing mirrors [`merge_adapters`] (a key-sniffed third-party LyCORIS LoKr is handled additively too,
/// via the structured Kronecker vec-trick). A LoKr with no allocation-free structured form (a tucker/CP
/// `lokr_t2`, or a base that does not factor as a·b × c·d) is rejected. Like [`merge_adapters`], a
/// non-empty spec set that matches **no** target errors (never renders unadapted).
pub fn install_additive<D: AdditiveDit + ?Sized>(
    dit: &mut D,
    specs: &[AdapterSpec],
) -> Result<AdditiveReport> {
    let mut report = AdditiveReport::default();

    // The kohya `flattened → dotted` table, built from the DiT's own adaptable projection paths (all
    // Linear, rank-2), so a `lora_transformer_<flat>` key resolves exactly as the dense fold's
    // `build_kohya_table` would (the additive-path analog with no base tensor map at hand).
    let mut paths: Vec<String> = Vec::new();
    dit.visit_additive(&mut |path, _proj| {
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
        // Route exactly as `merge_adapters`: an explicit/declared LoKr, or a key-sniffed third-party
        // LyCORIS LoKr (`lokr_*` without a `networkType` stamp), resolves as LoKr; else LoRA. A
        // `Lora`-declared file whose metadata says `networkType=lokr` is a loud mismatch.
        let is_lokr = spec.kind == AdapterKind::Lokr || af.declares_lokr() || has_lokr_keys(&af);
        if spec.kind == AdapterKind::Lora && af.declares_lokr() {
            return Err(CandleError::Msg(format!(
                "krea: adapter {} declared Lora but its metadata says networkType=lokr",
                spec.path.display()
            )));
        }
        if is_lokr {
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
    // factors are read on the CPU but the base weight lives on the DiT device (CUDA on a packed tier), so
    // they are moved onto it at push. A factor whose dims don't match is surfaced as a skipped key, never
    // a crashing forward (the additive analog of the fold path's shape guard).
    let device = dit.adapter_device();
    let mut matched: HashSet<String> = HashSet::new();
    let mut applied = 0usize;
    let mut skipped_keys = 0usize;
    dit.visit_additive(&mut |path, proj| {
        let (out_f, in_f) = proj.out_in();
        if let Some(list) = pending_lora.get(path) {
            matched.insert(path.to_string());
            for p in list {
                if p.a.dims()[0] != in_f || p.b.dims()[1] != out_f {
                    skipped_keys += 1;
                    continue;
                }
                proj.add_lora(p.a.to_device(&device)?, p.b.to_device(&device)?, p.scale);
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
                    None, // no tucker/CP `lokr_t2` on the peft/lycoris LoKr surface
                    p.w2_a.as_ref(),
                    p.w2_b.as_ref(),
                )? {
                    Some(factors) => {
                        proj.add_lokr(factors.to_device(&device)?);
                        applied += 1;
                    }
                    None => {
                        return Err(CandleError::Msg(format!(
                            "krea: LoKr target `{path}` is not deferrable (a tucker/CP `lokr_t2`, or a \
                             base that does not factor as a·b × c·d) — no allocation-free structured \
                             form. Use a dense bf16 snapshot."
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
    if !specs.is_empty() && report.applied == 0 {
        return Err(no_target_matched(
            "krea",
            dit.adapter_surface_hint(),
            specs.len(),
        ));
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train_dit::KREA_ATTN_TARGETS;
    use candle_gen::candle_core::safetensors::save as save_tensors;
    use candle_gen::candle_core::{DType, Device};

    /// A tiny stand-in for the base DiT tensor map: two attention Linears + one conv (4-D) weight.
    fn base_map() -> HashMap<String, Tensor> {
        let dev = Device::Cpu;
        let mut m = HashMap::new();
        m.insert(
            "transformer_blocks.0.attn.to_q.weight".into(),
            Tensor::zeros((4, 4), DType::BF16, &dev).unwrap(),
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

    fn max_abs(t: &Tensor) -> f32 {
        t.abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    /// Bare dotted (the trainer's format), prefixed PEFT, and kohya flattened all resolve to the same
    /// dotted DiT path.
    #[test]
    fn classify_lora_resolves_bare_peft_and_kohya() {
        let table = build_kohya_table(&base_map(), &[2]);
        // bare dotted (what `save_lora_peft(set, "", …)` writes for the DiT).
        let (p, _) =
            classify_lora_key("transformer_blocks.0.attn.to_q.lora_A.weight", &table).unwrap();
        assert_eq!(p, "transformer_blocks.0.attn.to_q");
        // PEFT-prefixed (community / peft.save_pretrained).
        let (p, r) = classify_lora_key(
            "transformer.transformer_blocks.0.attn.to_q.lora_B.weight",
            &table,
        )
        .unwrap();
        assert_eq!(p, "transformer_blocks.0.attn.to_q");
        assert!(matches!(r, Role::Up));
        // `.default.` infix.
        assert!(matches!(
            classify_lora_key(
                "base_model.model.transformer.transformer_blocks.0.attn.to_q.lora_B.default.weight",
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

    /// sc-8185: ostris **ai-toolkit** keys Krea-2 LoRAs to the raw checkpoint layout
    /// (`diffusion_model.blocks.N.attn.wq`, an `mlp` FFN, `txtfusion.…`). Those must resolve to the
    /// same canonical DiT paths the merge folds into — in particular `wo` → `to_out.0` and
    /// `mlp` → `ff` — and the normalizer must be a no-op on already-diffusers paths (our own LoRAs).
    #[test]
    fn classify_lora_normalizes_native_aitoolkit_naming() {
        let table = build_kohya_table(&base_map(), &[2]);
        let cases = [
            (
                "diffusion_model.blocks.0.attn.wq.lora_A.weight",
                "transformer_blocks.0.attn.to_q",
            ),
            (
                "diffusion_model.blocks.3.attn.wo.lora_B.weight",
                "transformer_blocks.3.attn.to_out.0",
            ),
            (
                "diffusion_model.blocks.5.attn.gate.lora_A.weight",
                "transformer_blocks.5.attn.to_gate",
            ),
            (
                "diffusion_model.blocks.2.mlp.down.lora_A.weight",
                "transformer_blocks.2.ff.down",
            ),
            (
                "diffusion_model.txtfusion.layerwise_blocks.0.attn.wk.lora_A.weight",
                "text_fusion.layerwise_blocks.0.attn.to_k",
            ),
            (
                "diffusion_model.txtfusion.refiner_blocks.1.mlp.up.lora_B.weight",
                "text_fusion.refiner_blocks.1.ff.up",
            ),
        ];
        for (key, want) in cases {
            let (p, _) = classify_lora_key(key, &table).unwrap();
            assert_eq!(p, want, "native key {key} must normalize to {want}");
        }
        // No-op on already-diffusers paths (our converter/trainer output).
        assert_eq!(
            normalize_native_krea_path("transformer_blocks.0.attn.to_out.0"),
            "transformer_blocks.0.attn.to_out.0"
        );
        assert_eq!(
            normalize_native_krea_path("text_fusion.refiner_blocks.1.ff.gate"),
            "text_fusion.refiner_blocks.1.ff.gate"
        );
    }

    /// Bare-dotted LoRA merges into `W += (alpha/rank)·scale·B·A`; base+delta is exact in f32.
    #[test]
    fn merge_lora_bare_folds_expected_delta() {
        let mut map = base_map();
        let down = t2(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], 2, 4); // A [rank=2, in=4]
        let up = t2(&[2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0], 4, 2); // B [out=4, rank=2]
        let af = AdapterFile {
            tensors: HashMap::from([
                (
                    "transformer_blocks.0.attn.to_q.lora_A.weight".to_string(),
                    down.clone(),
                ),
                (
                    "transformer_blocks.0.attn.to_q.lora_B.weight".to_string(),
                    up.clone(),
                ),
                (
                    "transformer_blocks.0.attn.to_q.alpha".to_string(),
                    Tensor::from_vec(vec![4.0f32], (1,), &Device::Cpu).unwrap(),
                ),
            ]),
            meta: HashMap::new(),
        };
        let table = build_kohya_table(&map, &[2]);
        let mut report = MergeReport::default();
        // scale 1.0; alpha 4, rank 2 ⇒ effective 2.0. ΔW = 2.0·(B·A).
        merge_lora_file(&mut map, &af, 1.0, &table, &mut report).unwrap();
        assert_eq!(report.merged, 1);
        let merged = map
            .get("transformer_blocks.0.attn.to_q.weight")
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let expected = reconstruct_lora_delta(&down, &up, 4.0, 2.0, 1.0).unwrap(); // base is zero
        assert!(max_abs(&(merged - expected).unwrap()) < 1e-2); // bf16 base round-trip tolerance
    }

    /// sc-5374: a diffusers-format LoRA with NO per-target `.alpha` tensor but a `lora_adapter_metadata`
    /// blob (`lora_alpha = 16`, `r = 8`) merges at the metadata-derived `(16/8)·scale = 2.0`, not the
    /// old `alpha = rank` default. Bare-dotted DiT keys; base is zero so the merged weight IS the delta.
    #[test]
    fn merge_lora_honors_lora_adapter_metadata_alpha() {
        let dev = Device::Cpu;
        let mut map = base_map();
        let path = "transformer_blocks.0.attn.to_q";
        let down = Tensor::randn(0f32, 1f32, (8, 4), &dev).unwrap(); // A [r=8, in=4]
        let up = Tensor::randn(0f32, 1f32, (4, 8), &dev).unwrap(); // B [out=4, r=8]
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
        let expected = reconstruct_lora_delta(&down, &up, 16.0, 8.0, 1.0).unwrap();
        assert!(max_abs(&(&merged - &expected).unwrap()) < 1e-4);
        // The pre-sc-5374 default (alpha = rank ⇒ scale 1.0) would diverge by a factor of 2.
        let buggy = reconstruct_lora_delta(&down, &up, 8.0, 8.0, 1.0).unwrap();
        assert!(
            max_abs(&(&merged - &buggy).unwrap()) > 1e-3,
            "metadata alpha must differ from alpha=rank"
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

    /// LoKr merges `δ = (alpha/rank)·kron(w1,w2)` into the dense weight, reading rank/alpha from meta.
    #[test]
    fn merge_lokr_folds_kron_delta() {
        let mut map = base_map();
        let w1 = t2(&[1.0, 0.0, 0.0, 1.0], 2, 2);
        let w2 = t2(&[0.5, 0.0, 0.0, 0.5], 2, 2);
        let af = AdapterFile {
            tensors: HashMap::from([
                (
                    "transformer_blocks.0.attn.to_q.lokr_w1".to_string(),
                    w1.clone(),
                ),
                (
                    "transformer_blocks.0.attn.to_q.lokr_w2".to_string(),
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
            .get("transformer_blocks.0.attn.to_q.weight")
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
        assert!(max_abs(&(merged - expected).unwrap()) < 1e-2);
    }

    /// The keystone sc-8776 round-trip: an **ostris ai-toolkit** LoKr — **no** `networkType` metadata,
    /// per-target `.alpha` tensors, full `lokr_w1`/`lokr_w2` factors, native `diffusion_model.blocks…`
    /// / `wq` / `mlp` naming across the attention (incl. `gate`) + SwiGLU FFN of a single-stream **and**
    /// a text_fusion block — is written to disk, read back through the public [`merge_adapters`] with
    /// the worker's `AdapterKind::Lora` classification, and folds in **every** target with
    /// `skipped_keys == 0`. Because both Kronecker legs are full, the lycoris scale is 1, so the huge
    /// sentinel `.alpha` (~1e10, as real ai-toolkit files ship) is correctly ignored.
    #[test]
    fn merge_adapters_sniffs_full_surface_aitoolkit_lokr() {
        let dev = Device::Cpu;

        // Base surface: one single-stream block (5 attn incl to_gate + 3 ff) + one text_fusion attn.
        let mut map: HashMap<String, Tensor> = HashMap::new();
        let base_zero = |m: &mut HashMap<String, Tensor>, k: String| {
            m.insert(k, Tensor::zeros((4, 4), DType::BF16, &dev).unwrap());
        };
        for t in ["to_q", "to_k", "to_v", "to_gate", "to_out.0"] {
            base_zero(&mut map, format!("transformer_blocks.0.attn.{t}.weight"));
        }
        for t in ["gate", "up", "down"] {
            base_zero(&mut map, format!("transformer_blocks.0.ff.{t}.weight"));
        }
        base_zero(
            &mut map,
            "text_fusion.layerwise_blocks.0.attn.to_q.weight".into(),
        );

        // ai-toolkit native LoKr: full 2×2 ⊗ 2×2 = 4×4 delta, a sentinel per-target `.alpha`.
        let w1 = t2(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let w2 = t2(&[1.0, 0.0, 0.0, 1.0], 2, 2);
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        let add = |t: &mut HashMap<String, Tensor>, module: &str| {
            t.insert(format!("{module}.lokr_w1"), w1.clone());
            t.insert(format!("{module}.lokr_w2"), w2.clone());
            t.insert(
                format!("{module}.alpha"),
                Tensor::from_vec(vec![9.999e9f32], (1,), &dev).unwrap(),
            );
        };
        for t in ["wq", "wk", "wv", "gate", "wo"] {
            add(&mut tensors, &format!("diffusion_model.blocks.0.attn.{t}"));
        }
        for t in ["gate", "up", "down"] {
            add(&mut tensors, &format!("diffusion_model.blocks.0.mlp.{t}"));
        }
        add(
            &mut tensors,
            "diffusion_model.txtfusion.layerwise_blocks.0.attn.wq",
        );

        let file = std::env::temp_dir().join(format!(
            "krea_aitoolkit_lokr_{}.safetensors",
            std::process::id()
        ));
        save_tensors(&tensors, &file).unwrap();

        // Classified `Lora` by the worker (no networkType); the sniff must route it to the LoKr merge.
        let report = merge_adapters(
            &mut map,
            &[AdapterSpec::new(file.clone(), 1.0, AdapterKind::Lora)],
        );
        std::fs::remove_file(&file).ok();
        let report = report.unwrap();

        // All 9 targets merged (5 attn + 3 ff + 1 text_fusion), nothing skipped.
        assert_eq!(
            report.merged, 9,
            "every attn/gate/ffn/text_fusion target must merge"
        );
        assert_eq!(
            report.skipped_keys, 0,
            "no ai-toolkit target should be skipped"
        );

        // Both-full ⇒ scale 1: merged to_q (zero base) is exactly kron(w1,w2), NOT scaled by ~1e10.
        let merged = map
            .get("transformer_blocks.0.attn.to_q.weight")
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
            1.0,
            1.0,
            1.0,
            (4, 4),
        )
        .unwrap();
        assert!(
            max_abs(&(&merged - &expected).unwrap()) < 1e-4,
            "both-full LoKr must fold at scale 1 (sentinel alpha ignored)"
        );
        assert!(max_abs(&merged) > 0.0, "the delta must be non-trivial");
    }

    /// sc-8776: with **no** file-level `rank`/`alpha` metadata, a **decomposed** LoKr leg
    /// (`lokr_w2_a`/`lokr_w2_b`) honors the per-target `.alpha` tensor — scale `alpha/rank` where
    /// `rank` is the decomposed inner dim — mirroring MLX `ThirdPartyLokr` / ai-toolkit
    /// (`scale = alpha / lora_dim`). Uses a rank-0 `.alpha` scalar, as real ai-toolkit files ship.
    #[test]
    fn merge_lokr_honors_per_target_alpha_when_decomposed() {
        let dev = Device::Cpu;
        let mut map = base_map(); // has transformer_blocks.0.attn.to_q [4,4]
        let path = "transformer_blocks.0.attn.to_q";
        let w1 = t2(&[1.0, 0.0, 0.0, 1.0], 2, 2); // full
        let w2a = t2(&[1.0, 0.0, 0.0, 1.0], 2, 2); // [out_k=2, rank=2]
        let w2b = t2(&[0.5, 1.0, 1.5, 2.0], 2, 2); // [rank=2, in_n=2]
        let af = AdapterFile {
            tensors: HashMap::from([
                (format!("{path}.lokr_w1"), w1.clone()),
                (format!("{path}.lokr_w2_a"), w2a.clone()),
                (format!("{path}.lokr_w2_b"), w2b.clone()),
                (
                    format!("{path}.alpha"),
                    Tensor::from_vec(vec![8.0f32], (), &dev).unwrap(), // rank-0 scalar
                ),
            ]),
            meta: HashMap::new(), // no networkType / rank / alpha
        };
        let table = build_kohya_table(&map, &[2]);
        let mut report = MergeReport::default();
        merge_lokr_file(&mut map, &af, 1.0, &table, &mut report).unwrap();
        assert_eq!(report.merged, 1);
        assert_eq!(report.skipped_keys, 0);

        let merged = map
            .get(&format!("{path}.weight"))
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        // rank = w2_a.dims[1] = 2, alpha = 8 ⇒ scale 4.
        let expected = reconstruct_lokr_delta(
            Some(&w1),
            None,
            None,
            None,
            Some(&w2a),
            Some(&w2b),
            8.0,
            2.0,
            1.0,
            (4, 4),
        )
        .unwrap();
        assert!(max_abs(&(&merged - &expected).unwrap()) < 1e-4);
        // The old alpha=rank default (scale 1) would diverge by a factor of 4.
        let buggy = reconstruct_lokr_delta(
            Some(&w1),
            None,
            None,
            None,
            Some(&w2a),
            Some(&w2b),
            2.0,
            2.0,
            1.0,
            (4, 4),
        )
        .unwrap();
        assert!(
            max_abs(&(&merged - &buggy).unwrap()) > 1e-3,
            "per-target alpha must differ from the alpha=rank default"
        );
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

    /// The keystone train→infer round-trip at the **map** level: a PEFT `.safetensors` written by the
    /// **actual trainer** path ([`candle_gen::train::lora::save_lora_peft`] with the DiT's empty prefix)
    /// is read back through the public [`merge_adapters`] entry, and the merged weight equals the
    /// trained delta.
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
        let path = "transformer_blocks.3.attn.to_v";
        let base_w = Tensor::zeros((4, 4), DType::F32, &dev).unwrap();
        let mut host = Host(LoraLinear::from_linear(
            Linear::new(base_w, None),
            4,
            4,
            path.into(),
        ));

        // rank 2, alpha 4 ⇒ effective 2.0. Force B (vars[1]) nonzero so ΔW ≠ 0 (zero-init B no-ops).
        let set = build_lora_targets(&mut host, &["to_v".to_string()], 2, 4.0, 7, &dev).unwrap();
        let up_randn = Tensor::randn(0f32, 1f32, (4, 2), &dev).unwrap();
        set.vars[1].set(&up_randn).unwrap(); // vars = [down(A), up(B)]

        let file = std::env::temp_dir().join(format!(
            "candle_krea_lora_roundtrip_{}.safetensors",
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

        assert_eq!(report.merged, 1, "the trained to_v adapter must merge");
        let expected = reconstruct_lora_delta(
            set.vars[0].as_tensor(),
            set.vars[1].as_tensor(),
            4.0,
            2.0,
            1.0,
        )
        .unwrap();
        let merged = map[&format!("{path}.weight")].to_dtype(DType::F32).unwrap();
        assert!(max_abs(&(&merged - &expected).unwrap()) < 1e-2);
        assert!(
            max_abs(&expected) > 0.0,
            "forced-nonzero B must yield a non-trivial delta"
        );
    }

    /// The end-to-end engine path under test: [`merge_into_weights`] preloads the attention surface
    /// from a real (mmaped) [`Weights`], folds a directly-built LoRA in, and installs the overlay — so
    /// `Weights::get` serves the **merged** `to_q` while every untargeted projection is untouched.
    #[test]
    fn merge_into_weights_overlays_attention_surface() {
        let dev = Device::Cpu;
        let pid = std::process::id();
        let base_file = std::env::temp_dir().join(format!("krea_adapt_base_{pid}.safetensors"));
        let adapter_file = std::env::temp_dir().join(format!("krea_adapt_lora_{pid}.safetensors"));

        // A 1-block base snapshot: the four attention projections, zero-initialized.
        let mut base = HashMap::new();
        for target in KREA_ATTN_TARGETS {
            base.insert(
                format!("transformer_blocks.0.attn.{target}.weight"),
                Tensor::zeros((4, 4), DType::BF16, &dev).unwrap(),
            );
        }
        save_tensors(&base, &base_file).unwrap();

        // A bare-dotted LoRA targeting only to_q (alpha 4, rank 2 ⇒ effective 2.0).
        let down = Tensor::randn(0f32, 1f32, (2, 4), &dev).unwrap();
        let up = Tensor::randn(0f32, 1f32, (4, 2), &dev).unwrap();
        let adapter = HashMap::from([
            (
                "transformer_blocks.0.attn.to_q.lora_A.weight".to_string(),
                down.clone(),
            ),
            (
                "transformer_blocks.0.attn.to_q.lora_B.weight".to_string(),
                up.clone(),
            ),
            (
                "transformer_blocks.0.attn.to_q.alpha".to_string(),
                Tensor::from_vec(vec![4.0f32], (1,), &dev).unwrap(),
            ),
        ]);
        save_tensors(&adapter, &adapter_file).unwrap();

        let mut cfg = Krea2Config::turbo();
        cfg.num_layers = 1;
        let mut w = Weights::from_file(&base_file, &dev, DType::BF16).unwrap();
        let report = merge_into_weights(
            &mut w,
            &cfg,
            &[AdapterSpec::new(
                adapter_file.clone(),
                1.0,
                AdapterKind::Lora,
            )],
        )
        .unwrap();

        assert_eq!(report.merged, 1, "to_q must merge");
        assert_eq!(report.skipped_keys, 0, "nothing should be skipped");

        // to_q now serves the trained delta (base was zero) ...
        let merged = w.get_f32("transformer_blocks.0.attn.to_q.weight").unwrap();
        let expected = reconstruct_lora_delta(&down, &up, 4.0, 2.0, 1.0).unwrap();
        assert!(max_abs(&(merged - expected).unwrap()) < 1e-4);
        // ... while the untargeted projections stay at their (zero) base.
        for target in ["to_k", "to_v", "to_out.0"] {
            let untouched = w
                .get_f32(&format!("transformer_blocks.0.attn.{target}.weight"))
                .unwrap();
            assert_eq!(max_abs(&untouched), 0.0, "{target} must be untouched");
        }

        std::fs::remove_file(&base_file).ok();
        std::fs::remove_file(&adapter_file).ok();
    }

    /// AC: a scale-0 adapter merge is byte-exact with the base (`δ·0 = 0`), so the overlaid weight
    /// equals the original — a LoRA at strength 0 is a no-op render.
    #[test]
    fn scale_zero_merge_is_base() {
        let dev = Device::Cpu;
        let pid = std::process::id();
        let base_file = std::env::temp_dir().join(format!("krea_adapt_base0_{pid}.safetensors"));
        let adapter_file = std::env::temp_dir().join(format!("krea_adapt_lora0_{pid}.safetensors"));

        // A nonzero base so "equals base" is a real assertion, not a trivial zero match.
        let base_q = Tensor::randn(0f32, 1f32, (4, 4), &dev).unwrap();
        let mut base = HashMap::new();
        base.insert(
            "transformer_blocks.0.attn.to_q.weight".to_string(),
            base_q.to_dtype(DType::BF16).unwrap(),
        );
        for target in ["to_k", "to_v", "to_out.0"] {
            base.insert(
                format!("transformer_blocks.0.attn.{target}.weight"),
                Tensor::zeros((4, 4), DType::BF16, &dev).unwrap(),
            );
        }
        save_tensors(&base, &base_file).unwrap();

        let adapter = HashMap::from([
            (
                "transformer_blocks.0.attn.to_q.lora_A.weight".to_string(),
                Tensor::randn(0f32, 1f32, (2, 4), &dev).unwrap(),
            ),
            (
                "transformer_blocks.0.attn.to_q.lora_B.weight".to_string(),
                Tensor::randn(0f32, 1f32, (4, 2), &dev).unwrap(),
            ),
            (
                "transformer_blocks.0.attn.to_q.alpha".to_string(),
                Tensor::from_vec(vec![4.0f32], (1,), &dev).unwrap(),
            ),
        ]);
        save_tensors(&adapter, &adapter_file).unwrap();

        let mut cfg = Krea2Config::turbo();
        cfg.num_layers = 1;
        let mut w = Weights::from_file(&base_file, &dev, DType::BF16).unwrap();
        let report = merge_into_weights(
            &mut w,
            &cfg,
            &[AdapterSpec::new(
                adapter_file.clone(),
                0.0,
                AdapterKind::Lora,
            )],
        )
        .unwrap();
        assert_eq!(report.merged, 1, "the target still 'merges' (a zero delta)");

        // Overlaid to_q (bf16 base → f32 + 0) must equal the original bf16 base, byte-for-byte.
        let merged = w.get_f32("transformer_blocks.0.attn.to_q.weight").unwrap();
        let original = base_q
            .to_dtype(DType::BF16)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        assert_eq!(
            max_abs(&(merged - original).unwrap()),
            0.0,
            "scale-0 merge must be byte-exact with the base"
        );

        std::fs::remove_file(&base_file).ok();
        std::fs::remove_file(&adapter_file).ok();
    }

    /// **Adapter merge composes with a PACKED base (sc-9411).** A packed q4 transformer component whose
    /// `to_q`/`to_k`/`to_v`/`to_out.0` are MLX-packed triples: merging a LoRA that targets only `to_q`
    /// must (a) reconstruct the dense grid from `to_q`'s packed parts, fold the delta in, and install
    /// the merged **dense** weight in the overlay (so `Weights::get` serves `dequant(grid) + δ`); and
    /// (b) leave every untargeted packed projection **out** of the overlay, so it still loads packed via
    /// `linear_detect`. This is the packed-base ⊕ adapter-overlay compose the story requires.
    #[test]
    fn merge_into_weights_composes_with_packed_base() {
        use candle_gen::candle_core::safetensors;
        use candle_gen::quant::dequant_mlx_q4_reference_gs;

        let dev = Device::Cpu;
        const G: usize = 64;
        // Build an MLX group-64 Q4 packed triple for an [out, in] weight, returning (wq, scales, biases).
        let q4 = |out_dim: usize, in_dim: usize| {
            let codes: Vec<u8> = (0..out_dim * in_dim)
                .map(|i| ((i * 5 + i / 7) % 16) as u8)
                .collect();
            let groups = out_dim * in_dim / G;
            let scales: Vec<f32> = (0..groups).map(|g| 0.05 * (g as f32 + 1.0)).collect();
            let biases: Vec<f32> = (0..groups).map(|g| -0.4 - 0.1 * g as f32).collect();
            let words: Vec<u32> = codes
                .chunks_exact(8)
                .map(|c| {
                    c.iter()
                        .enumerate()
                        .fold(0u32, |acc, (i, &q)| acc | ((q as u32 & 0xF) << (4 * i)))
                })
                .collect();
            (
                Tensor::from_vec(words, (out_dim, in_dim / 8), &dev).unwrap(),
                Tensor::from_vec(scales, (out_dim, in_dim / G), &dev).unwrap(),
                Tensor::from_vec(biases, (out_dim, in_dim / G), &dev).unwrap(),
            )
        };

        // A 1-block packed component: all four attention projections packed (128×128 ⇒ group-64).
        let (dim, pid) = (128usize, std::process::id());
        let dir = std::env::temp_dir().join(format!("krea_packed_compose_{pid}"));
        std::fs::create_dir_all(&dir).unwrap();
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        let mut grids: HashMap<String, Tensor> = HashMap::new();
        for target in ["to_q", "to_k", "to_v", "to_out.0"] {
            let (wq, s, b) = q4(dim, dim);
            let base = format!("transformer_blocks.0.attn.{target}");
            grids.insert(
                target.to_string(),
                dequant_mlx_q4_reference_gs(&wq, &s, &b, G).unwrap(),
            );
            tensors.insert(format!("{base}.weight"), wq);
            tensors.insert(format!("{base}.scales"), s);
            tensors.insert(format!("{base}.biases"), b);
        }
        safetensors::save(&tensors, dir.join("model.safetensors")).unwrap();
        std::fs::write(
            dir.join("config.json"),
            serde_json::json!({ "quantization": { "bits": 4, "group_size": G } }).to_string(),
        )
        .unwrap();

        // A bare-dotted LoRA targeting only to_q.
        let down = Tensor::randn(0f32, 1f32, (2, dim), &dev).unwrap();
        let up = Tensor::randn(0f32, 1f32, (dim, 2), &dev).unwrap();
        let adapter_file = std::env::temp_dir().join(format!("krea_packed_lora_{pid}.safetensors"));
        safetensors::save(
            &HashMap::from([
                (
                    "transformer_blocks.0.attn.to_q.lora_A.weight".to_string(),
                    down.clone(),
                ),
                (
                    "transformer_blocks.0.attn.to_q.lora_B.weight".to_string(),
                    up.clone(),
                ),
                (
                    "transformer_blocks.0.attn.to_q.alpha".to_string(),
                    Tensor::from_vec(vec![2.0f32], (1,), &dev).unwrap(),
                ),
            ]),
            &adapter_file,
        )
        .unwrap();

        let mut cfg = Krea2Config::turbo();
        cfg.num_layers = 1;
        let mut w = Weights::from_dir(&dir, &dev, DType::BF16).unwrap();
        assert!(w.packed().is_some(), "packed component");

        let report = merge_into_weights(
            &mut w,
            &cfg,
            &[AdapterSpec::new(
                adapter_file.clone(),
                1.0,
                AdapterKind::Lora,
            )],
        )
        .unwrap();
        assert_eq!(report.merged, 1, "only to_q merges");

        // (a) to_q now serves dequant(grid) + δ — a DENSE overlay weight (not the packed triple).
        let merged = w.get_f32("transformer_blocks.0.attn.to_q.weight").unwrap();
        let delta = reconstruct_lora_delta(&down, &up, 2.0, 2.0, 1.0).unwrap();
        let expected = (grids["to_q"].clone() + delta).unwrap();
        assert!(
            max_abs(&(merged - expected).unwrap()) < 1e-2,
            "packed to_q must merge into dequant(grid)+δ"
        );

        // (b) the untargeted projections load PACKED (they were left out of the overlay), so the packed
        // base stays packed — the compose invariant.
        for target in ["to_k", "to_v", "to_out.0"] {
            let lin = crate::loader::linear_detect(
                &w,
                &format!("transformer_blocks.0.attn.{target}"),
                false,
            )
            .unwrap();
            assert!(
                lin.is_packed(),
                "{target} was not adapter-targeted ⇒ must remain packed after the merge"
            );
        }
        // to_q now resolves dense (its overlay shadows the packed triple).
        let q = crate::loader::linear_detect(&w, "transformer_blocks.0.attn.to_q", false).unwrap();
        assert!(!q.is_packed(), "adapter-merged to_q resolves dense");

        std::fs::remove_dir_all(&dir).ok();
        std::fs::remove_file(&adapter_file).ok();
    }

    /// **Additive == folded parity at the resolver level (sc-11105).** The unmerged factors
    /// [`resolve_lora_file`] produces reproduce the folded `x·(W + δ)ᵀ` on a dense
    /// [`candle_gen::quant::AdaptLinear`] base — using this crate's exact [`classify_lora_key`] (incl.
    /// the ai-toolkit native rename) + effective alpha/rank — so the packed additive path and the dense
    /// fold path agree to f32 tolerance. Uses the native `blocks.0.attn.wq` spelling to also cover
    /// [`normalize_native_krea_path`].
    #[test]
    fn resolve_lora_matches_fold_on_dense() {
        use candle_gen::candle_nn::Linear;
        use candle_gen::quant::AdaptLinear;
        let dev = Device::Cpu;
        let (out_dim, in_dim, rank) = (16usize, 12usize, 3usize);
        // ai-toolkit native path → normalize_native rewrites to transformer_blocks.0.attn.to_q.
        let native = "blocks.0.attn.wq";
        let diffusers = "transformer_blocks.0.attn.to_q";
        let down = Tensor::randn(0f32, 1f32, (rank, in_dim), &dev).unwrap();
        let up = Tensor::randn(0f32, 1f32, (out_dim, rank), &dev).unwrap();
        let (alpha, scale) = (6.0f32, 0.8f32);
        let af = AdapterFile {
            tensors: HashMap::from([
                (format!("{native}.lora_A.weight"), down.clone()),
                (format!("{native}.lora_B.weight"), up.clone()),
                (
                    format!("{native}.alpha"),
                    Tensor::from_vec(vec![alpha], (1,), &dev).unwrap(),
                ),
            ]),
            meta: HashMap::new(),
        };
        let table: BTreeMap<String, String> = BTreeMap::new();
        let mut pending: BTreeMap<String, Vec<PendingLora>> = BTreeMap::new();
        let mut skipped = 0usize;
        resolve_lora_file(&af, scale, &table, &mut pending, &mut skipped).unwrap();
        assert_eq!(skipped, 0);
        let p = &pending[diffusers][0];
        assert_eq!(p.a.dims(), &[in_dim, rank], "a = downᵀ [in, rank]");
        assert_eq!(p.b.dims(), &[rank, out_dim], "b = upᵀ·ratio [rank, out]");

        let w = Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev).unwrap();
        let mut additive = AdaptLinear::from_dense(Linear::new(w.clone(), None), in_dim, out_dim);
        additive.push_lora(p.a.clone(), p.b.clone(), p.scale);
        let delta = reconstruct_lora_delta(&down, &up, alpha, rank as f32, scale).unwrap();
        let folded =
            AdaptLinear::from_dense(Linear::new((w + delta).unwrap(), None), in_dim, out_dim);
        let x = Tensor::randn(0f32, 1f32, (2usize, in_dim), &dev).unwrap();
        let d = max_abs(&(additive.forward(&x).unwrap() - folded.forward(&x).unwrap()).unwrap());
        assert!(d < 1e-4, "resolved additive != folded ({d})");
    }

    /// **Additive == folded parity for a peft LoKr (sc-11105).** The structured factors
    /// [`resolve_lokr_file`] produces (full `(alpha/rank)·scale` baked) reproduce the folded LoKr delta
    /// on a dense base — the packed additive path equals the dense fold to f32 tolerance.
    #[test]
    fn resolve_lokr_matches_fold_on_dense() {
        use candle_gen::candle_nn::Linear;
        use candle_gen::quant::{AdaptLinear, LokrFactors};
        let dev = Device::Cpu;
        let (a, b, c, d) = (2usize, 3, 4, 5);
        let (out, inp) = (a * b, c * d);
        let w1 = Tensor::randn(0f32, 1f32, (a, c), &dev).unwrap();
        let w2 = Tensor::randn(0f32, 1f32, (b, d), &dev).unwrap();
        let path = "transformer_blocks.0.attn.to_q";
        let af = AdapterFile {
            tensors: HashMap::from([
                (format!("{path}.lokr_w1"), w1.clone()),
                (format!("{path}.lokr_w2"), w2.clone()),
            ]),
            meta: HashMap::from([
                ("networkType".to_string(), "lokr".to_string()),
                ("rank".to_string(), "2".to_string()),
                ("alpha".to_string(), "4".to_string()),
            ]),
        };
        let table: BTreeMap<String, String> = BTreeMap::new();
        let mut pending: BTreeMap<String, Vec<PendingLokr>> = BTreeMap::new();
        let mut skipped = 0usize;
        resolve_lokr_file(&af, 0.5, &table, &mut pending, &mut skipped).unwrap();
        let p = &pending[path][0];
        // full = (alpha/rank)·scale = (4/2)·0.5 = 1.0.
        let factors = LokrFactors::build(
            p.scale,
            (out, inp),
            p.w1.as_ref(),
            p.w1_a.as_ref(),
            p.w1_b.as_ref(),
            p.w2.as_ref(),
            None,
            p.w2_a.as_ref(),
            p.w2_b.as_ref(),
        )
        .unwrap()
        .expect("a plain linear LoKr is deferrable");
        let base_w = Tensor::randn(0f32, 1f32, (out, inp), &dev).unwrap();
        let mut additive = AdaptLinear::from_dense(Linear::new(base_w.clone(), None), inp, out);
        additive.push_lokr_structured(factors);
        let delta = reconstruct_lokr_delta(
            Some(&w1),
            None,
            None,
            Some(&w2),
            None,
            None,
            4.0,
            2.0,
            0.5,
            (out, inp),
        )
        .unwrap();
        let folded =
            AdaptLinear::from_dense(Linear::new((base_w + delta).unwrap(), None), inp, out);
        let x = Tensor::randn(0f32, 1f32, (2usize, inp), &dev).unwrap();
        let dd = max_abs(&(additive.forward(&x).unwrap() - folded.forward(&x).unwrap()).unwrap());
        assert!(dd < 1e-4, "resolved additive LoKr != folded ({dd})");
        assert_eq!(skipped, 0);
    }

    /// **Generic wide-surface install over both leaf types (sc-11720).** [`install_additive`] drives a DiT
    /// through the [`AdditiveDit`] trait, pushing residuals onto BOTH a control-DiT [`LoraLinear`] leaf and
    /// a txt2img [`AdaptLinear`] leaf, and onto a FRONT-END path (`img_in`) outside the attention set —
    /// covering the trait unification AND the widened surface in one shot. Each adapted forward equals the
    /// folded `x·(W + δ)ᵀ` to f32 tolerance (default alpha = rank ⇒ ratio 1).
    #[test]
    fn install_additive_drives_lora_and_adapt_leaves_wide_surface() {
        use candle_gen::candle_nn::{Linear, Module};
        use candle_gen::quant::AdaptLinear;
        use candle_gen::train::lora::LoraLinear;

        let dev = Device::Cpu;
        let (out_dim, in_dim, rank) = (8usize, 6usize, 2usize);
        let wq = Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev).unwrap();
        let wi = Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev).unwrap();
        let leg = || {
            (
                Tensor::randn(0f32, 1f32, (rank, in_dim), &dev).unwrap(),
                Tensor::randn(0f32, 1f32, (out_dim, rank), &dev).unwrap(),
            )
        };
        let (dq, uq) = leg();
        let (di, ui) = leg();

        // One adapter file targeting an ATTENTION leaf (→ LoraLinear) and a FRONT-END leaf (→ AdaptLinear).
        let dir = std::env::temp_dir().join("krea_sc11720_wide");
        std::fs::create_dir_all(&dir).unwrap();
        let adapter_file = dir.join("wide.safetensors");
        save_tensors(
            &HashMap::from([
                (
                    "transformer_blocks.0.attn.to_q.lora_A.weight".to_string(),
                    dq.clone(),
                ),
                (
                    "transformer_blocks.0.attn.to_q.lora_B.weight".to_string(),
                    uq.clone(),
                ),
                ("img_in.lora_A.weight".to_string(), di.clone()),
                ("img_in.lora_B.weight".to_string(), ui.clone()),
            ]),
            &adapter_file,
        )
        .unwrap();

        struct MockDit {
            device: Device,
            to_q: LoraLinear,
            img_in: AdaptLinear,
        }
        impl AdditiveDit for MockDit {
            fn visit_additive(
                &mut self,
                f: &mut dyn FnMut(&str, &mut dyn AdditiveProj) -> Result<()>,
            ) -> Result<()> {
                f("transformer_blocks.0.attn.to_q", &mut self.to_q)?;
                f("img_in", &mut self.img_in)?;
                Ok(())
            }
            fn adapter_device(&self) -> Device {
                self.device.clone()
            }
            fn adapter_surface_hint(&self) -> &'static str {
                "mock"
            }
        }

        let mut dit = MockDit {
            device: dev.clone(),
            to_q: LoraLinear::from_linear(
                Linear::new(wq.clone(), None),
                in_dim,
                out_dim,
                "transformer_blocks.0.attn.to_q".into(),
            ),
            img_in: AdaptLinear::from_dense(Linear::new(wi.clone(), None), in_dim, out_dim),
        };
        let report = install_additive(
            &mut dit,
            &[AdapterSpec::new(adapter_file, 1.0, AdapterKind::Lora)],
        )
        .unwrap();
        assert_eq!(
            report.applied, 2,
            "both the attention LoraLinear and the front-end AdaptLinear must adapt"
        );
        assert!(report.skipped_targets.is_empty());

        let x = Tensor::randn(0f32, 1f32, (2usize, in_dim), &dev).unwrap();
        let q_delta = reconstruct_lora_delta(&dq, &uq, rank as f32, rank as f32, 1.0).unwrap();
        let q_folded =
            AdaptLinear::from_dense(Linear::new((wq + q_delta).unwrap(), None), in_dim, out_dim);
        let q_diff =
            max_abs(&(dit.to_q.forward(&x).unwrap() - q_folded.forward(&x).unwrap()).unwrap());
        assert!(
            q_diff < 1e-4,
            "LoraLinear attention leaf additive != fold ({q_diff})"
        );

        let i_delta = reconstruct_lora_delta(&di, &ui, rank as f32, rank as f32, 1.0).unwrap();
        let i_folded =
            AdaptLinear::from_dense(Linear::new((wi + i_delta).unwrap(), None), in_dim, out_dim);
        let i_diff =
            max_abs(&(dit.img_in.forward(&x).unwrap() - i_folded.forward(&x).unwrap()).unwrap());
        assert!(
            i_diff < 1e-4,
            "AdaptLinear front-end leaf additive != fold ({i_diff})"
        );

        std::fs::remove_dir_all(&dir).ok();
    }
}
