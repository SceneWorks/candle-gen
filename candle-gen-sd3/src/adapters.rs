//! SD3.5 inference-side adapter merge (sc-8498, epic 7982) — load a trained / community SD3.5
//! LoRA/LoKr `.safetensors` and fold its delta into the MMDiT dense weights at the
//! safetensors-key level **before** [`crate::transformer::Sd3Transformer`] is built (and before the
//! optional Q4/Q8 quantize). The candle twin of the z-image (sc-5166) / krea (sc-7836) DiT merge and
//! the SDXL UNet merge (sc-5165): community SD3.5 LoRAs now actually apply in candle inference (the
//! C6 community-LoRA acceptance criterion that C7's `supports_lora:false` could not meet).
//!
//! **Merge, don't residual** (same rationale as SDXL / Z-Image / Krea): inference has no need to keep
//! the factors trainable, so it folds `W += δ` into the dense weight and reproduces the merged-weight
//! forward exactly. The flow-match sampler is chaos-sensitive — `(W+δ)·x` ≠ `W·x + δ·x` to ~1 ULP —
//! so a live residual would drift. The delta is reconstructed with the **same** f32 math the trainer
//! uses ([`reconstruct_lora_delta`] / [`reconstruct_lokr_delta`]).
//!
//! **Merge at the safetensors-key level, then quantize.** The DiT reads its `transformer/` keys 1:1,
//! so `{path}.weight` is the base key for every Linear an adapter targets. The merge runs over a CPU
//! tensor map of the `transformer/` snapshot; the merged map is then handed to the DiT builder via a
//! `VarBuilder::from_tensors`, and the Q4/Q8 quantize (if any) runs on the *merged* dense weight — so
//! adapters and quantization compose (the dense weight gets the delta, then quantizes).
//!
//! **The kohya / sd-scripts `lora_sd3` key surface — fused MMDiT naming, NOT diffusers.** Community
//! SD3.5 LoRAs (the test artifact `SD3.5-Turbo-Portrait.safetensors`, 1131 keys, dim32/alpha16) are
//! trained by sd-scripts against the **original (sai / sd3-impls) MMDiT** module layout:
//!   * `lora_unet_joint_blocks_<i>_x_block_attn_qkv`      — a FUSED image-stream Q·K·V projection
//!   * `lora_unet_joint_blocks_<i>_context_block_attn_qkv`— a FUSED text-stream Q·K·V projection
//!   * `..._x_block_attn_proj` / `..._context_block_attn_proj`   — the output projections
//!   * `..._{x,context}_block_mlp_fc1` / `_mlp_fc2`        — the GELU MLP
//!   * `..._{x,context}_block_adaLN_modulation_1`          — the AdaLN modulation linear
//!
//! The port reads the **diffusers** split-QKV surface (`attn.to_q/to_k/to_v`, `attn.add_q_proj/…`,
//! `attn.to_out.0`, `attn.to_add_out`, `ff.net.0.proj`/`ff.net.2`, `norm1.linear`/`norm1_context.linear`).
//! [`map_sd3_module`] translates names; the fused `attn_qkv` is the trickiest — a single LoRA
//! `(down [r,in], up [3·out,r])` must be **split into the three** `to_q/to_k/to_v` deltas. The
//! diffusers SD3 conversion splits `qkv.weight` as `chunk(3, dim=0)` in `q,k,v` order with the SAME
//! `in` projection, so the down factor is **shared** across all three and the up factor splits along
//! its **leading** (output) dim into three equal `[out, r]` blocks. The adaLN modulation maps 1:1
//! (the diffusers converter copies `adaLN_modulation.1` straight into `norm1.linear` with no row
//! reorder), so it is a plain rename.
//!
//! **Family-match policy:** no base-model gating (the SDXL / Z-Image / Krea precedent; base-model
//! gating is a `wan-video`-only worker concern). The engine merges whatever MMDiT-targeting factors
//! the file carries. Out-of-surface keys (text-encoder LoRAs — this is a DiT-only merge — or
//! unresolved module stems) are **counted and surfaced** in [`MergeReport`], never silently dropped.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use candle_gen::candle_core::{safetensors as cst, DType, Device, Tensor};
use candle_gen::gen_core::{AdapterKind, AdapterSpec};
use candle_gen::train::lora::{reconstruct_lokr_delta, reconstruct_lora_delta, LoraAdapterMeta};
use candle_gen::{CandleError, Result};

/// kohya / sd-scripts `lora_sd3` flattened-module prefix (the SD3 analog of SDXL's `lora_unet_`). The
/// `_`-flattened stem after this prefix is an sd3-impls module path; [`map_sd3_module`] translates it.
const KOHYA_PREFIX: &str = "lora_unet_";

/// PEFT key prefixes tolerated on read, longest-first. A diffusers / `peft.save_pretrained()` SD3.5
/// LoRA wraps the DiT under one of these; stripping yields the diffusers dotted module path. A key
/// matching none is taken as-is (bare dotted, the candle trainer's own format).
const PEFT_PREFIXES: [&str; 4] = [
    "base_model.model.transformer.",
    "base_model.model.",
    "diffusion_model.",
    "transformer.",
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

/// Outcome of merging the adapter specs into the base MMDiT tensor map: how many base weights were
/// updated, and how many keys fell outside the merge surface (text-encoder / unresolved / shape
/// mismatch — surfaced, not silently dropped).
#[derive(Debug, Default, PartialEq, Eq)]
pub struct MergeReport {
    pub merged: usize,
    pub skipped_keys: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Role {
    Down,
    Up,
    Alpha,
}

/// How a single LoRA module maps onto the port's diffusers base weight(s). A plain target folds into
/// one `{path}.weight`; a fused QKV target folds into three (`to_q`/`to_k`/`to_v` or `add_q`/`add_k`/
/// `add_v`) by splitting the up factor along its leading (output) dim.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Target {
    /// One diffusers dotted module path (`{path}.weight` is the base key).
    Single(String),
    /// A fused Q·K·V LoRA → the three diffusers dotted paths it splits into (q, k, v order).
    FusedQkv([String; 3]),
}

#[derive(Default)]
struct LoraTriple {
    down: Option<Tensor>, // A: [rank, in]
    up: Option<Tensor>,   // B: [out, rank]  (3·out for a fused qkv)
    alpha: Option<f32>,
}

/// A loaded adapter file: its tensors (CPU, native dtype) and the safetensors header metadata.
struct AdapterFile {
    tensors: HashMap<String, Tensor>,
    meta: HashMap<String, String>,
}

/// Read an adapter `.safetensors` once: tensors via candle's loader, metadata via the safetensors
/// header reader (candle's `load` drops the header `__metadata__`, which the alpha/rank can live in).
fn read_adapter(path: &Path) -> Result<AdapterFile> {
    let bytes = std::fs::read(path)
        .map_err(|e| CandleError::Msg(format!("read adapter {}: {e}", path.display())))?;
    let tensors = cst::load_buffer(&bytes, &Device::Cpu)?;
    let (_, md) = safetensors::SafeTensors::read_metadata(&bytes)
        .map_err(|e| CandleError::Msg(format!("read adapter metadata {}: {e}", path.display())))?;
    let meta = md.metadata().clone().unwrap_or_default();
    Ok(AdapterFile { tensors, meta })
}

/// Strip the longest matching PEFT prefix, or return the key unchanged (bare dotted path).
fn strip_peft_prefix(key: &str) -> &str {
    for p in PEFT_PREFIXES {
        if let Some(rem) = key.strip_prefix(p) {
            return rem;
        }
    }
    key
}

/// Translate an sd-scripts / kohya `lora_sd3` flattened module stem (everything after `lora_unet_`)
/// into the port's diffusers [`Target`], or `None` if it is not a recognized MMDiT module.
///
/// The sd3-impls naming the converter folds into diffusers:
///   `joint_blocks_<i>_x_block_attn_qkv`        → FusedQkv(to_q, to_k, to_v)
///   `joint_blocks_<i>_context_block_attn_qkv`  → FusedQkv(add_q_proj, add_k_proj, add_v_proj)
///   `joint_blocks_<i>_x_block_attn_proj`       → attn.to_out.0
///   `joint_blocks_<i>_context_block_attn_proj` → attn.to_add_out
///   `joint_blocks_<i>_x_block_mlp_fc1`         → ff.net.0.proj
///   `joint_blocks_<i>_x_block_mlp_fc2`         → ff.net.2
///   `joint_blocks_<i>_context_block_mlp_fc1`   → ff_context.net.0.proj
///   `joint_blocks_<i>_context_block_mlp_fc2`   → ff_context.net.2
///   `joint_blocks_<i>_x_block_adaLN_modulation_1`       → norm1.linear
///   `joint_blocks_<i>_context_block_adaLN_modulation_1` → norm1_context.linear
fn map_sd3_module(stem: &str) -> Option<Target> {
    let rest = stem.strip_prefix("joint_blocks_")?;
    // Split the leading block index off `<i>_<module>`.
    let us = rest.find('_')?;
    let idx: usize = rest[..us].parse().ok()?;
    let module = &rest[us + 1..];
    let pre = format!("transformer_blocks.{idx}");

    // Which stream + module leaf.
    let (stream, leaf) = if let Some(l) = module.strip_prefix("x_block_") {
        ("x", l)
    } else if let Some(l) = module.strip_prefix("context_block_") {
        ("context", l)
    } else {
        return None;
    };

    Some(match (stream, leaf) {
        ("x", "attn_qkv") => Target::FusedQkv([
            format!("{pre}.attn.to_q"),
            format!("{pre}.attn.to_k"),
            format!("{pre}.attn.to_v"),
        ]),
        ("context", "attn_qkv") => Target::FusedQkv([
            format!("{pre}.attn.add_q_proj"),
            format!("{pre}.attn.add_k_proj"),
            format!("{pre}.attn.add_v_proj"),
        ]),
        ("x", "attn_proj") => Target::Single(format!("{pre}.attn.to_out.0")),
        ("context", "attn_proj") => Target::Single(format!("{pre}.attn.to_add_out")),
        ("x", "mlp_fc1") => Target::Single(format!("{pre}.ff.net.0.proj")),
        ("x", "mlp_fc2") => Target::Single(format!("{pre}.ff.net.2")),
        ("context", "mlp_fc1") => Target::Single(format!("{pre}.ff_context.net.0.proj")),
        ("context", "mlp_fc2") => Target::Single(format!("{pre}.ff_context.net.2")),
        ("x", "adaLN_modulation_1") => Target::Single(format!("{pre}.norm1.linear")),
        ("context", "adaLN_modulation_1") => Target::Single(format!("{pre}.norm1_context.linear")),
        _ => return None,
    })
}

/// Resolve a *diffusers-named* dotted module path (PEFT / bare) into a [`Target`]. A diffusers
/// `attn.qkv`-style fused key does not exist in the diffusers layout (it is split at conversion), so
/// every diffusers path is a [`Target::Single`]. The candle trainer + diffusers/PEFT community LoRAs
/// take this path; only the sd-scripts kohya files take the fused [`map_sd3_module`] path.
fn diffusers_target(path: &str) -> Target {
    Target::Single(path.to_string())
}

/// Map one LoRA key to `(target, role)`, or `None` if outside the MMDiT merge surface. kohya
/// (`lora_unet_<flat>…`) translates the flattened stem via [`map_sd3_module`]; PEFT/bare resolve the
/// diffusers dotted path directly after the optional prefix strip.
fn classify_lora_key(key: &str) -> Option<(Target, Role)> {
    if let Some(rem) = key.strip_prefix(KOHYA_PREFIX) {
        for (suf, role) in [
            (".lora_down.weight", Role::Down),
            (".lora_up.weight", Role::Up),
            (".alpha", Role::Alpha),
        ] {
            if let Some(stem) = rem.strip_suffix(suf) {
                return map_sd3_module(stem).map(|t| (t, role));
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
            return Some((diffusers_target(path), role));
        }
    }
    None
}

/// Map one LoKr factor key to `(target, factor_name)`, or `None` if out of surface.
fn classify_lokr_key(key: &str) -> Option<(Target, &'static str)> {
    for suf in LOKR_SUFFIXES {
        if let Some(stem) = key.strip_suffix(suf) {
            let factor = &suf[1..]; // drop the leading '.'
            return if let Some(flat) = stem.strip_prefix(KOHYA_PREFIX) {
                map_sd3_module(flat).map(|t| (t, factor))
            } else {
                Some((diffusers_target(strip_peft_prefix(stem)), factor))
            };
        }
    }
    None
}

fn read_scalar(t: &Tensor) -> Result<f32> {
    Ok(t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?[0])
}

/// Merge `delta` (`[out, in]` f32) into the base weight at `key`, computing `W += δ` in f32 (the
/// merged map is later served by a `VarBuilder::from_tensors`, cast to the load dtype). A missing key
/// or a shape-mismatched base (e.g. an unexpected fused split) is surfaced as skipped, never a hard
/// error.
fn merge_into(
    base: &mut HashMap<String, Tensor>,
    key: &str,
    delta: &Tensor,
    report: &mut MergeReport,
) -> Result<()> {
    let merged = {
        let Some(w) = base.get(key) else {
            report.skipped_keys += 1;
            return Ok(());
        };
        if w.dims() != delta.dims() {
            report.skipped_keys += 1;
            return Ok(());
        }
        (w.to_dtype(DType::F32)? + delta)?
    };
    base.insert(key.to_string(), merged);
    report.merged += 1;
    Ok(())
}

/// Fold a complete LoRA `(down, up, alpha)` triple into `base`, handling the fused-QKV split: a
/// [`Target::FusedQkv`] splits `up [3·out, rank]` along dim 0 into three `[out, rank]` blocks (q, k, v
/// order) sharing the single `down [rank, in]`, reconstructs each delta, and merges into the
/// respective `to_q/to_k/to_v` (or `add_q/add_k/add_v`) base weight.
fn merge_lora_triple(
    base: &mut HashMap<String, Tensor>,
    target: &Target,
    down: &Tensor,
    up: &Tensor,
    alpha: f32,
    scale: f32,
    report: &mut MergeReport,
) -> Result<()> {
    if down.dims().len() != 2 || up.dims().len() != 2 {
        report.skipped_keys += 1; // conv-shaped or malformed — out of surface
        return Ok(());
    }
    let rank = down.dims()[0] as f32;
    match target {
        Target::Single(path) => {
            let delta = reconstruct_lora_delta(down, up, alpha, rank, scale)?;
            merge_into(base, &format!("{path}.weight"), &delta, report)
        }
        Target::FusedQkv(paths) => {
            // up is [3·out, rank]; split along the leading (output) dim into q,k,v.
            let total_out = up.dims()[0];
            if !total_out.is_multiple_of(3) {
                report.skipped_keys += 1;
                return Ok(());
            }
            let out = total_out / 3;
            for (i, path) in paths.iter().enumerate() {
                let up_i = up.narrow(0, i * out, out)?;
                let delta = reconstruct_lora_delta(down, &up_i, alpha, rank, scale)?;
                merge_into(base, &format!("{path}.weight"), &delta, report)?;
            }
            Ok(())
        }
    }
}

/// Merge one LoRA file into `base` at `scale`: classify every key, fold complete `(down, up)` pairs
/// (with the fused-QKV split) into the diffusers base weights. `rank` is `down`'s leading dim; `alpha`
/// is the per-target `.alpha` tensor when present, else the `lora_adapter_metadata` blob's
/// `alpha_pattern`/`lora_alpha` (diffusers / PEFT `save_lora_adapter` ships no `.alpha` tensor), else
/// `rank`. Half-pairs are surfaced as skipped.
fn merge_lora_file(
    base: &mut HashMap<String, Tensor>,
    af: &AdapterFile,
    scale: f32,
    report: &mut MergeReport,
) -> Result<()> {
    // Group by the resolved target (its debug string is a stable key).
    let mut triples: BTreeMap<String, (Target, LoraTriple)> = BTreeMap::new();
    for (key, t) in &af.tensors {
        match classify_lora_key(key) {
            Some((target, role)) => {
                let entry = triples
                    .entry(format!("{target:?}"))
                    .or_insert_with(|| (target, LoraTriple::default()));
                match role {
                    Role::Down => entry.1.down = Some(t.clone()),
                    Role::Up => entry.1.up = Some(t.clone()),
                    Role::Alpha => entry.1.alpha = Some(read_scalar(t)?),
                }
            }
            None => report.skipped_keys += 1,
        }
    }

    let cfg = LoraAdapterMeta::from_file_metadata(&af.meta);
    for (_, (target, t)) in triples {
        let (Some(down), Some(up)) = (t.down, t.up) else {
            report.skipped_keys += 1; // half-pair (partner targeted a non-routable module)
            continue;
        };
        // The metadata path uses the first target path for the per-module override lookup.
        let path = match &target {
            Target::Single(p) => p.clone(),
            Target::FusedQkv(p) => p[0].clone(),
        };
        let (cfg_alpha, cfg_rank) = cfg.as_ref().map_or((None, None), |c| c.effective(&path));
        let rank = cfg_rank.unwrap_or(down.dims()[0] as f32);
        let alpha = t.alpha.or(cfg_alpha).unwrap_or(rank);
        merge_lora_triple(base, &target, &down, &up, alpha, scale, report)?;
    }
    Ok(())
}

/// Merge one LoKr file into `base` at `scale`. LoKr is **not** the fused-QKV community surface (kohya
/// `lora_sd3` LoKr is rare and, when present, is keyed per split projection), so every LoKr factor
/// resolves to a [`Target::Single`]; a fused LoKr would map to a FusedQkv whose base weight shape
/// would mismatch the kron reconstruction and be surfaced as skipped.
fn merge_lokr_file(
    base: &mut HashMap<String, Tensor>,
    af: &AdapterFile,
    scale: f32,
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

    let mut grouped: BTreeMap<String, (Target, BTreeMap<&'static str, Tensor>)> = BTreeMap::new();
    for (key, t) in &af.tensors {
        match classify_lokr_key(key) {
            Some((target, factor)) => {
                grouped
                    .entry(format!("{target:?}"))
                    .or_insert_with(|| (target, BTreeMap::new()))
                    .1
                    .insert(factor, t.clone());
            }
            None => report.skipped_keys += 1,
        }
    }

    for (_, (target, f)) in grouped {
        // Only single-projection LoKr is in surface; a fused LoKr's kron reconstruction has no
        // well-defined per-q/k/v split, so route it through the first path and let the shape check
        // surface the mismatch as skipped.
        let path = match &target {
            Target::Single(p) => p.clone(),
            Target::FusedQkv(p) => p[0].clone(),
        };
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

/// Whether the adapter file declares LoKr in its `networkType` metadata.
fn declares_lokr(af: &AdapterFile) -> bool {
    af.meta.get("networkType").map(String::as_str) == Some("lokr")
}

/// Fold every adapter spec in `specs` into the base MMDiT tensor `map` (CPU, native dtype) at each
/// spec's `scale` — LoRA and LoKr, merged into the dense weights (`W += δ`), with the kohya `lora_sd3`
/// fused-QKV split handled. Returns the [`MergeReport`]; errors if a non-empty spec list matches
/// **no** target (a format / prefix misconfiguration — the worker should then fall back rather than
/// render an unadapted image silently).
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
        match spec.kind {
            AdapterKind::Lokr => merge_lokr_file(map, &af, spec.scale, &mut report)?,
            AdapterKind::Lora => {
                if declares_lokr(&af) {
                    return Err(CandleError::Msg(format!(
                        "sd3: adapter {} declared Lora but its metadata says networkType=lokr",
                        spec.path.display()
                    )));
                }
                merge_lora_file(map, &af, spec.scale, &mut report)?;
            }
        }
    }
    if report.merged == 0 {
        return Err(CandleError::Msg(format!(
            "sd3: no adapter target modules matched across {} file(s) — expected kohya/sd-scripts \
             `lora_unet_joint_blocks_<i>_{{x,context}}_block_<attn_qkv|attn_proj|mlp_fc1|mlp_fc2|\
             adaLN_modulation_1>.lora_down/up.weight`, or diffusers/PEFT/bare \
             `<path>.lora_A/B.weight` over the MMDiT (transformer_blocks.<i>.<attn|ff|norm1>…). \
             Text-encoder adapters are out of surface",
            specs.len()
        )));
    }
    Ok(report)
}

/// Read the snapshot `transformer/` `.safetensors` into a CPU tensor map, fold the LoRA/LoKr `specs`
/// in ([`merge_adapters`], f32 math), and return the merged map (CPU, the snapshot's native dtype, +
/// the f32 deltas already summed). The caller hands this to `VarBuilder::from_tensors` to build the
/// MMDiT (and quantizes the *merged* dense weights), so adapters + Q4/Q8 compose. A no-op (the
/// unadapted snapshot) when `specs` is empty.
pub fn merged_transformer_tensors(
    transformer_files: &[std::path::PathBuf],
    specs: &[AdapterSpec],
) -> Result<(HashMap<String, Tensor>, MergeReport)> {
    let mut map: HashMap<String, Tensor> = HashMap::new();
    for f in transformer_files {
        let part = cst::load(f, &Device::Cpu)?;
        map.extend(part);
    }
    let report = merge_adapters(&mut map, specs)?;
    Ok((map, report))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::safetensors::save as save_tensors;

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

    /// The kohya `lora_sd3` fused-QKV name resolves to the THREE diffusers split paths in q,k,v order;
    /// the context stream resolves to add_q/k/v_proj.
    #[test]
    fn map_sd3_fused_qkv_splits_to_three_diffusers_paths() {
        assert_eq!(
            map_sd3_module("joint_blocks_0_x_block_attn_qkv").unwrap(),
            Target::FusedQkv([
                "transformer_blocks.0.attn.to_q".into(),
                "transformer_blocks.0.attn.to_k".into(),
                "transformer_blocks.0.attn.to_v".into(),
            ])
        );
        assert_eq!(
            map_sd3_module("joint_blocks_12_context_block_attn_qkv").unwrap(),
            Target::FusedQkv([
                "transformer_blocks.12.attn.add_q_proj".into(),
                "transformer_blocks.12.attn.add_k_proj".into(),
                "transformer_blocks.12.attn.add_v_proj".into(),
            ])
        );
    }

    /// Every non-QKV sd3-impls module name maps to its single diffusers path.
    #[test]
    fn map_sd3_single_modules() {
        let cases = [
            (
                "joint_blocks_0_x_block_attn_proj",
                "transformer_blocks.0.attn.to_out.0",
            ),
            (
                "joint_blocks_3_context_block_attn_proj",
                "transformer_blocks.3.attn.to_add_out",
            ),
            (
                "joint_blocks_5_x_block_mlp_fc1",
                "transformer_blocks.5.ff.net.0.proj",
            ),
            (
                "joint_blocks_5_x_block_mlp_fc2",
                "transformer_blocks.5.ff.net.2",
            ),
            (
                "joint_blocks_7_context_block_mlp_fc1",
                "transformer_blocks.7.ff_context.net.0.proj",
            ),
            (
                "joint_blocks_7_context_block_mlp_fc2",
                "transformer_blocks.7.ff_context.net.2",
            ),
            (
                "joint_blocks_2_x_block_adaLN_modulation_1",
                "transformer_blocks.2.norm1.linear",
            ),
            (
                "joint_blocks_2_context_block_adaLN_modulation_1",
                "transformer_blocks.2.norm1_context.linear",
            ),
        ];
        for (stem, want) in cases {
            assert_eq!(
                map_sd3_module(stem).unwrap(),
                Target::Single(want.to_string()),
                "{stem}"
            );
        }
        // Garbage / text-encoder stems are out of surface.
        assert!(map_sd3_module("te1_text_model_encoder_layers_0").is_none());
        assert!(map_sd3_module("joint_blocks_0_x_block_unknown").is_none());
    }

    /// `classify_lora_key` resolves the full kohya `lora_unet_…` key (down/up/alpha) and the PEFT/bare
    /// diffusers keys.
    #[test]
    fn classify_lora_resolves_kohya_and_peft() {
        let (t, r) =
            classify_lora_key("lora_unet_joint_blocks_0_x_block_attn_qkv.lora_down.weight")
                .unwrap();
        assert!(matches!(t, Target::FusedQkv(_)));
        assert_eq!(r, Role::Down);
        let (t, r) =
            classify_lora_key("lora_unet_joint_blocks_0_x_block_attn_proj.lora_up.weight").unwrap();
        assert_eq!(
            t,
            Target::Single("transformer_blocks.0.attn.to_out.0".into())
        );
        assert_eq!(r, Role::Up);
        // PEFT-prefixed diffusers key.
        let (t, _) =
            classify_lora_key("transformer.transformer_blocks.0.attn.to_q.lora_A.weight").unwrap();
        assert_eq!(t, Target::Single("transformer_blocks.0.attn.to_q".into()));
        // Text-encoder kohya keys are out of surface.
        assert!(classify_lora_key(
            "lora_te1_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight"
        )
        .is_none());
    }

    /// A tiny base map: the 38-key SD3 surface for one block at a small inner dim, so the fused-QKV
    /// split + every single target can be exercised cheaply.
    fn base_map(inner: usize, ff: usize) -> HashMap<String, Tensor> {
        let dev = Device::Cpu;
        let mut m = HashMap::new();
        let mut z = |k: String, o: usize, i: usize| {
            m.insert(k, Tensor::zeros((o, i), DType::BF16, &dev).unwrap());
        };
        for leaf in [
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
            "add_q_proj",
            "add_k_proj",
            "add_v_proj",
            "to_add_out",
        ] {
            z(
                format!("transformer_blocks.0.attn.{leaf}.weight"),
                inner,
                inner,
            );
        }
        z(
            "transformer_blocks.0.ff.net.0.proj.weight".into(),
            ff,
            inner,
        );
        z("transformer_blocks.0.ff.net.2.weight".into(), inner, ff);
        z(
            "transformer_blocks.0.norm1.linear.weight".into(),
            6 * inner,
            inner,
        );
        m
    }

    /// The fused-QKV merge: a single LoRA `(down [r,in], up [3·inner,r])` folds into to_q/to_k/to_v,
    /// each the corresponding `[inner, r]` slice of `up` times the shared `down`. Base is zero so the
    /// merged weight IS the per-split delta.
    #[test]
    fn merge_fused_qkv_splits_into_three_targets() {
        let inner = 4usize;
        let mut map = base_map(inner, 8);
        let dev = Device::Cpu;
        let rank = 2usize;
        let down = Tensor::randn(0f32, 1f32, (rank, inner), &dev).unwrap();
        let up = Tensor::randn(0f32, 1f32, (3 * inner, rank), &dev).unwrap();
        let af = AdapterFile {
            tensors: HashMap::from([
                (
                    "lora_unet_joint_blocks_0_x_block_attn_qkv.lora_down.weight".to_string(),
                    down.clone(),
                ),
                (
                    "lora_unet_joint_blocks_0_x_block_attn_qkv.lora_up.weight".to_string(),
                    up.clone(),
                ),
                (
                    "lora_unet_joint_blocks_0_x_block_attn_qkv.alpha".to_string(),
                    Tensor::from_vec(vec![2.0f32], (1,), &dev).unwrap(),
                ),
            ]),
            meta: HashMap::new(),
        };
        let mut report = MergeReport::default();
        merge_lora_file(&mut map, &af, 1.0, &mut report).unwrap();
        assert_eq!(report.merged, 3, "fused qkv must fold into 3 targets");
        assert_eq!(report.skipped_keys, 0);
        for (i, leaf) in ["to_q", "to_k", "to_v"].iter().enumerate() {
            let up_i = up.narrow(0, i * inner, inner).unwrap();
            let expected = reconstruct_lora_delta(&down, &up_i, 2.0, rank as f32, 1.0).unwrap();
            let merged = map
                .get(&format!("transformer_blocks.0.attn.{leaf}.weight"))
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap();
            assert!(max_abs(&(merged - expected).unwrap()) < 1e-2, "{leaf}");
        }
    }

    /// A single (non-fused) target folds `W += (alpha/rank)·B·A` exactly.
    #[test]
    fn merge_single_target_folds_delta() {
        let inner = 4usize;
        let mut map = base_map(inner, 8);
        let down = t2(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], 2, 4);
        let up = t2(&[2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0], 4, 2);
        let af = AdapterFile {
            tensors: HashMap::from([
                (
                    "lora_unet_joint_blocks_0_x_block_attn_proj.lora_down.weight".to_string(),
                    down.clone(),
                ),
                (
                    "lora_unet_joint_blocks_0_x_block_attn_proj.lora_up.weight".to_string(),
                    up.clone(),
                ),
                (
                    "lora_unet_joint_blocks_0_x_block_attn_proj.alpha".to_string(),
                    Tensor::from_vec(vec![4.0f32], (1,), &Device::Cpu).unwrap(),
                ),
            ]),
            meta: HashMap::new(),
        };
        let mut report = MergeReport::default();
        merge_lora_file(&mut map, &af, 1.0, &mut report).unwrap();
        assert_eq!(report.merged, 1);
        let merged = map
            .get("transformer_blocks.0.attn.to_out.0.weight")
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let expected = reconstruct_lora_delta(&down, &up, 4.0, 2.0, 1.0).unwrap();
        assert!(max_abs(&(merged - expected).unwrap()) < 1e-2);
    }

    /// scale=0 ≡ base byte-exact (δ·0 = 0), including the fused-QKV split: a LoRA at strength 0 is a
    /// no-op render. A nonzero base makes "equals base" a real assertion.
    #[test]
    fn scale_zero_is_base_byte_exact() {
        let inner = 4usize;
        let dev = Device::Cpu;
        let mut map = base_map(inner, 8);
        // Give to_q/to_k/to_v a nonzero base.
        for leaf in ["to_q", "to_k", "to_v"] {
            map.insert(
                format!("transformer_blocks.0.attn.{leaf}.weight"),
                Tensor::randn(0f32, 1f32, (inner, inner), &dev)
                    .unwrap()
                    .to_dtype(DType::BF16)
                    .unwrap(),
            );
        }
        let original: HashMap<String, Tensor> = ["to_q", "to_k", "to_v"]
            .iter()
            .map(|leaf| {
                let k = format!("transformer_blocks.0.attn.{leaf}.weight");
                (k.clone(), map[&k].to_dtype(DType::F32).unwrap())
            })
            .collect();

        let af = AdapterFile {
            tensors: HashMap::from([
                (
                    "lora_unet_joint_blocks_0_x_block_attn_qkv.lora_down.weight".to_string(),
                    Tensor::randn(0f32, 1f32, (2, inner), &dev).unwrap(),
                ),
                (
                    "lora_unet_joint_blocks_0_x_block_attn_qkv.lora_up.weight".to_string(),
                    Tensor::randn(0f32, 1f32, (3 * inner, 2), &dev).unwrap(),
                ),
                (
                    "lora_unet_joint_blocks_0_x_block_attn_qkv.alpha".to_string(),
                    Tensor::from_vec(vec![16.0f32], (1,), &dev).unwrap(),
                ),
            ]),
            meta: HashMap::new(),
        };
        let mut report = MergeReport::default();
        merge_lora_file(&mut map, &af, 0.0, &mut report).unwrap();
        assert_eq!(report.merged, 3);
        for leaf in ["to_q", "to_k", "to_v"] {
            let k = format!("transformer_blocks.0.attn.{leaf}.weight");
            let merged = map[&k].to_dtype(DType::F32).unwrap();
            assert_eq!(
                max_abs(&(merged - original[&k].clone()).unwrap()),
                0.0,
                "scale-0 merge must be byte-exact with base ({leaf})"
            );
        }
    }

    /// dim32/alpha16 scaling (the test artifact's params): the effective multiplier is alpha/rank =
    /// 16/32 = 0.5, NOT 1.0. Verifies the alpha tensor drives the scale.
    #[test]
    fn dim32_alpha16_scales_by_half() {
        let inner = 4usize;
        let mut map = base_map(inner, 8);
        let dev = Device::Cpu;
        let rank = 32usize;
        let down = Tensor::randn(0f32, 1f32, (rank, inner), &dev).unwrap();
        let up = Tensor::randn(0f32, 1f32, (inner, rank), &dev).unwrap();
        let af = AdapterFile {
            tensors: HashMap::from([
                (
                    "lora_unet_joint_blocks_0_x_block_attn_proj.lora_down.weight".to_string(),
                    down.clone(),
                ),
                (
                    "lora_unet_joint_blocks_0_x_block_attn_proj.lora_up.weight".to_string(),
                    up.clone(),
                ),
                (
                    "lora_unet_joint_blocks_0_x_block_attn_proj.alpha".to_string(),
                    Tensor::from_vec(vec![16.0f32], (1,), &dev).unwrap(),
                ),
            ]),
            meta: HashMap::new(),
        };
        let mut report = MergeReport::default();
        merge_lora_file(&mut map, &af, 1.0, &mut report).unwrap();
        let merged = map["transformer_blocks.0.attn.to_out.0.weight"]
            .to_dtype(DType::F32)
            .unwrap();
        let expected = reconstruct_lora_delta(&down, &up, 16.0, 32.0, 1.0).unwrap();
        assert!(max_abs(&(&merged - &expected).unwrap()) < 1e-3);
        // The alpha=rank (scale 1.0) default would be 2× off.
        let buggy = reconstruct_lora_delta(&down, &up, 32.0, 32.0, 1.0).unwrap();
        assert!(max_abs(&(&merged - &buggy).unwrap()) > 1e-4);
    }

    /// LoKr (single projection) merges `δ = (alpha/rank)·kron(w1,w2)` into the dense weight.
    #[test]
    fn merge_lokr_single_target() {
        let inner = 4usize;
        let mut map = base_map(inner, 8);
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
        let mut report = MergeReport::default();
        merge_lokr_file(&mut map, &af, 1.0, &mut report).unwrap();
        assert_eq!(report.merged, 1);
        let merged = map["transformer_blocks.0.attn.to_q.weight"]
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

    /// A non-empty spec list that matches nothing is a loud error (never a silent unadapted render).
    #[test]
    fn empty_match_is_error() {
        let pid = std::process::id();
        let file = std::env::temp_dir().join(format!("sd3_lora_nomatch_{pid}.safetensors"));
        let adapter = HashMap::from([(
            "lora_te1_text_model_encoder_layers_0.lora_down.weight".to_string(),
            t2(&[0.0, 0.0], 1, 2),
        )]);
        save_tensors(&adapter, &file).unwrap();
        let mut map = base_map(4, 8);
        let res = merge_adapters(
            &mut map,
            &[AdapterSpec::new(file.clone(), 1.0, AdapterKind::Lora)],
        );
        std::fs::remove_file(&file).ok();
        assert!(res.is_err(), "no-match spec list must error");
    }

    /// merge → quantize composition: a fused-QKV LoRA merged into a 32-wide base, then the merged
    /// weight quantized to Q8 and dequantized, tracks the (dense) merged weight — proving adapters fold
    /// BEFORE quantization (the dense weight gets the delta, then quantizes).
    #[test]
    fn merge_then_quantize_composes() {
        use crate::quant::ggml_dtype;
        use candle_gen::candle_core::quantized::QTensor;
        use candle_gen::gen_core::Quant;

        let inner = 32usize; // one Q8_0 block per contraction row
        let dev = Device::Cpu;
        let mut map = HashMap::new();
        for leaf in ["to_q", "to_k", "to_v"] {
            map.insert(
                format!("transformer_blocks.0.attn.{leaf}.weight"),
                Tensor::randn(0f32, 1f32, (inner, inner), &dev)
                    .unwrap()
                    .to_dtype(DType::BF16)
                    .unwrap(),
            );
        }
        let af = AdapterFile {
            tensors: HashMap::from([
                (
                    "lora_unet_joint_blocks_0_x_block_attn_qkv.lora_down.weight".to_string(),
                    Tensor::randn(0f32, 1f32, (4, inner), &dev).unwrap(),
                ),
                (
                    "lora_unet_joint_blocks_0_x_block_attn_qkv.lora_up.weight".to_string(),
                    Tensor::randn(0f32, 1f32, (3 * inner, 4), &dev).unwrap(),
                ),
                (
                    "lora_unet_joint_blocks_0_x_block_attn_qkv.alpha".to_string(),
                    Tensor::from_vec(vec![4.0f32], (1,), &dev).unwrap(),
                ),
            ]),
            meta: HashMap::new(),
        };
        let mut report = MergeReport::default();
        merge_lora_file(&mut map, &af, 1.0, &mut report).unwrap();
        assert_eq!(report.merged, 3);

        // The merged dense weight (f32) is the quantize source; dequant must track it.
        let merged = map["transformer_blocks.0.attn.to_q.weight"]
            .to_dtype(DType::F32)
            .unwrap();
        let qt = QTensor::quantize_onto(&merged, ggml_dtype(Quant::Q8), &dev).unwrap();
        let recon = qt.dequantize(&dev).unwrap();
        // Relative recon error within Q8 tolerance — the delta is genuinely inside the quantized blocks.
        let num = (&merged - &recon)
            .unwrap()
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
            .sqrt();
        let den = merged
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
            .sqrt();
        assert!(
            num / den < 0.05,
            "Q8 recon of merged weight off by {}",
            num / den
        );
    }
}
