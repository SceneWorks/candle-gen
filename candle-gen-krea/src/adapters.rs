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
//! blocks, [`merge_surface_keys`]) — so an ai-toolkit LoKr that adapts gate + FFN folds in fully
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

use std::collections::{BTreeMap, HashMap};

use candle_gen::candle_core::Tensor;
use candle_gen::gen_core::{AdapterKind, AdapterSpec};
use candle_gen::train::lora::{reconstruct_lokr_delta, reconstruct_lora_delta, LoraAdapterMeta};
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
                triples.entry(path).or_default().alpha = Some(read_scalar(t)?)
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
                Some(path) => grouped.entry(path).or_default().alpha = Some(read_scalar(t)?),
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
/// base weights ([`attention_surface_keys`]) onto the CPU, fold each adapter's delta in
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
            map.insert(key.clone(), w.get_cpu(&key)?);
        }
    }
    let report = merge_adapters(&mut map, specs)?;
    w.set_overlay(map);
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
}
