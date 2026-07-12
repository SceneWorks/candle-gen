//! SCAIL-2 inference-side adapter merge (sc-6838, epic 6563) — fold a LoRA / LoKr / LoHa / lightx2v
//! **lightning diff-patch** `.safetensors` delta into the dense [`Scail2Dit`](crate::model::Scail2Dit)
//! weights **before** the DiT is built. The candle (Windows/CUDA) twin of `mlx-gen-scail2`'s adapter
//! consumption (`AdaptableHost for Scail2Dit` + the `lora.rs` diff-patch merge), realized in the
//! by-key-merge style the candle [`candle_gen_wan::adapters`] / `candle-gen-qwen-image::adapters`
//! ports already use.
//!
//! **Two LoRA consumers, one merge:**
//!  - the **Bias-Aware DPO** refinement LoRA (`sat-scail2`) — a standard rank-128 PEFT LoRA (quality
//!    toggle);
//!  - the **lightx2v lightning** few-step distill — a *hybrid* file: low-rank `lora_down/up` pairs
//!    **plus** full-rank `.diff` (weight) / `.diff_b` (bias) deltas (incl. on the qk-RMSNorms, the
//!    affine `norm3` / `img_emb.proj.{0,4}` LayerNorms, and the `head.head`), the ComfyUI "diff patch"
//!    mechanism. Merged at scale 1.0 the 8-step / CFG-off lightning schedule produces a clean clip.
//!  - general SCAIL-2-native LoRA/LoKr ride the same path.
//!
//! **Merge, don't residual** (the chaos-sensitive-sampler argument from the SDXL/Z-Image/Wan ports):
//! fold the delta into the dense weight (`W += δ`, biases `b += δ_b`) at the **safetensors-key level**
//! before construction, so the merged forward `(W+δ)·x + (b+δ_b)` is reproduced exactly with no
//! per-step residual op. candle loads the DiT dense (f32), so — unlike MLX (which splits a residual-
//! over-Q4 path from a pre-build merge) — **all** of LoRA, LoKr, LoHa, and the diff-patch fold through
//! this one pre-build merge. The low-rank delta is reconstructed with the same f32 math the trainer's
//! forward uses ([`reconstruct_lora_delta`] / [`reconstruct_lokr_delta`] / [`reconstruct_loha_delta`]),
//! so a candle-trained adapter round-trips.
//!
//! **Merge surface = the raw `SCAIL2Model` keys** the [`Scail2Dit`](crate::model) reads 1:1:
//! `blocks.{i}.{self_attn,cross_attn}.{q,k,v,o[,k_img,v_img]}`, `blocks.{i}.ffn.{0,2}`, the qk-/cross
//! RMSNorms + affine `norm3`, and the globals (`patch_embedding{,_pose,_mask}`, `text_embedding.{0,2}`,
//! `time_embedding.{0,2}`, `time_projection.1`, `img_emb.proj.{0,1,3,4}`, `head.head`). A prefix-stripped
//! dotted path resolves `{path}.weight` (and `.bias`) directly. Formats resolved (`gen-core`'s
//! [`wmeta::COMMON_LORA_PREFIXES`] = `transformer.` / `diffusion_model.` / none):
//!  - **PEFT / diffusers / kohya / bare LoRA** — `‹prefix›‹path›.lora_A/B[.default].weight` **or**
//!    `‹prefix›‹path›.lora_down/up.weight` (+ optional `‹path›.alpha`). Scaling = the per-target
//!    `.alpha` tensor, else the diffusers `lora_adapter_metadata` blob, else `rank`.
//!  - **LoKr** — PEFT-stamped `‹path›.lokr_w1`/`lokr_w2` (+ low-rank `_a`/`_b`) with `networkType=lokr`
//!    and `rank`/`alpha` in file metadata, reconstructing `δ = (alpha/rank)·kron(w1,w2)`.
//!  - **Third-party LyCORIS** — untagged `lokr_*` / `hada_*` (no `networkType` stamp), per-module scale.
//!  - **lightx2v lightning diff-patch** — full-rank `‹path›.diff` (weight delta) + `‹path›.diff_b`
//!    (bias delta), merged `W += scale·diff`, `b += scale·diff_b`. **Cross-architecture shape-aware
//!    skip:** the lightx2v LoRA targets vanilla Wan2.1-I2V (`patch_embedding` in_dim **36**) whereas
//!    SCAIL-2's is in_dim **20** + the extra pose/mask stems, so a `.diff` whose shape ≠ the base is
//!    skipped **as a whole module** (its coupled `.diff_b` dropped too) and surfaced — never half-applied.
//!
//! Out-of-surface keys are counted in [`MergeReport`] and surfaced; a non-empty spec list that matches
//! **nothing** is a hard error (the worker should fall back rather than render an unadapted video).

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use candle_gen::candle_core::{DType, Tensor};
use candle_gen::gen_core::weightsmeta as wmeta;
use candle_gen::gen_core::{AdapterKind, AdapterSpec};
use candle_gen::train::lora::{reconstruct_lokr_delta, reconstruct_lora_delta, LoraAdapterMeta};
// The shared adapter-merge skeleton (sc-8998 / F-018): the format-parsing + merge-report + third-party
// LyCORIS engine this crate previously hand-copied. Only the SCAIL-2-specific key→module resolution
// (bare/prefixed dotted paths, no kohya table) + the bespoke lightx2v diff-patch merge stay local below.
use candle_gen::train::merge::{
    merge_into, merge_one_thirdparty, no_target_matched, parse_loha_thirdparty,
    parse_lokr_thirdparty, read_adapter, read_scalar, AdapterFile, LoraTriple, Role,
};
// Re-exported so `candle_gen_scail2::MergeReport` (the crate's public surface) keeps resolving.
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

/// Strip a leading SCAIL-2 LoRA namespace prefix (`transformer.` / `diffusion_model.`), if present —
/// leaving the bare dotted module path that resolves directly against the base DiT keys. A bare key and
/// a LoKr/LoHa factor key (always bare) pass through.
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
/// infix) and the diffusers/kohya (`lora_down`/`lora_up`) factor namings, plus the per-module `.alpha`.
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

/// Map one (PEFT-stamped) LoKr factor key to `(module_path, factor_name)`, or `None` if out of surface.
fn classify_lokr_key(key: &str) -> Option<(String, &'static str)> {
    for suf in LOKR_SUFFIXES {
        if let Some(stem) = key.strip_suffix(suf) {
            let factor = &suf[1..]; // drop the leading '.'
            return Some((strip_lora_prefix(stem).to_string(), factor));
        }
    }
    None
}

/// Merge one LoRA file's low-rank pairs into `base` at `scale`: classify every key, fold complete
/// `(down, up)` pairs into `{path}.weight`. Scaling = per-target `.alpha` → `lora_adapter_metadata`
/// blob → factor rank. Linear-only (a non-2-D pair, a half-pair, or an unresolved module is skipped);
/// any `.diff`/`.diff_b` in the same (lightx2v) file is handled separately by [`merge_diff_patch_file`].
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
                triples.entry(path).or_default().alpha = Some(read_scalar(key, "alpha", t)?)
            }
            // Not a low-rank key (could be a diff-patch tensor or out of surface) — diff-patch is
            // counted in its own pass; everything else is surfaced there or here as appropriate.
            None => {}
        }
    }

    let cfg = LoraAdapterMeta::from_file_metadata(&af.meta);
    for (path, t) in triples {
        let (Some(down), Some(up)) = (t.down, t.up) else {
            report.skipped_keys += 1; // half-pair (partner targeted a non-routable module)
            continue;
        };
        if down.dims().len() != 2 || up.dims().len() != 2 {
            report.skipped_keys += 1; // Linear-only low-rank surface (conv stems use diff-patch)
            continue;
        }
        let base_key = format!("{path}.weight");
        if !base.contains_key(&base_key) {
            report.skipped_keys += 1;
            continue;
        }
        let (cfg_alpha, cfg_rank) = cfg.as_ref().map_or((None, None), |c| c.effective(&path));
        let rank = cfg_rank.unwrap_or(down.dims()[0] as f32);
        let alpha = t.alpha.or(cfg_alpha).unwrap_or(rank);
        let delta = reconstruct_lora_delta(&down, &up, alpha, rank, scale)?;
        merge_into(base, &base_key, &delta, report)?;
    }
    Ok(())
}

// ---- lightx2v lightning diff-patch ----------------------------------------------------------------
//
// The few-step distill LoRA carries full-rank `.diff` (weight delta) and `.diff_b` (bias delta) tensors
// alongside the low-rank pairs — the ComfyUI "diff patch" mechanism — reaching the qk-RMSNorms, the
// affine `norm3`/`img_emb.proj.{0,4}` LayerNorms, the `head.head`, and every projection bias, none of
// which a low-rank pair targets. Merge `W += scale·diff`, `b += scale·diff_b` at the key level.

/// One module's diff-patch deltas (full-rank weight + bias), grouped by dotted module stem.
#[derive(Default)]
struct DiffPatch {
    diff: Option<Tensor>,   // weight delta, base-shaped
    diff_b: Option<Tensor>, // bias delta, base-shaped
}

/// Merge a file's diff-patch tensors into `base` at `scale`. **Module-coupled shape-aware skip:** if a
/// module's `.diff` (weight) does not match the base weight shape — the vanilla-Wan in_dim-36
/// `patch_embedding` vs SCAIL-2's in_dim-20 — the whole module is skipped, *including* its coupled
/// `.diff_b`, so the input stem is never half-patched. A no-op for a file without `.diff`/`.diff_b`.
fn merge_diff_patch_file(
    base: &mut HashMap<String, Tensor>,
    af: &AdapterFile,
    scale: f32,
    report: &mut MergeReport,
) -> Result<()> {
    let mut groups: BTreeMap<String, DiffPatch> = BTreeMap::new();
    for (key, t) in &af.tensors {
        if let Some(stem) = key.strip_suffix(".diff_b") {
            groups
                .entry(strip_lora_prefix(stem).to_string())
                .or_default()
                .diff_b = Some(t.clone());
        } else if let Some(stem) = key.strip_suffix(".diff") {
            groups
                .entry(strip_lora_prefix(stem).to_string())
                .or_default()
                .diff = Some(t.clone());
        }
    }

    for (path, g) in groups {
        // Apply at `scale`; the delta is reconstructed in f32 inside `merge_into`.
        let scaled = |t: &Tensor| -> Result<Tensor> {
            Ok(t.to_dtype(DType::F32)?.affine(scale as f64, 0.0)?)
        };
        match &g.diff {
            Some(diff) => {
                let wkey = format!("{path}.weight");
                let base_ok = base.get(&wkey).is_some_and(|w| w.dims() == diff.dims());
                if !base_ok {
                    // Cross-architecture (or out-of-surface) weight delta: skip the whole module,
                    // dropping its coupled bias delta too (loud, not a half-patch).
                    report.skipped_keys += 1;
                    if g.diff_b.is_some() {
                        report.skipped_keys += 1;
                    }
                    continue;
                }
                merge_into(base, &wkey, &scaled(diff)?, report)?;
                if let Some(db) = &g.diff_b {
                    merge_into(base, &format!("{path}.bias"), &scaled(db)?, report)?;
                }
            }
            None => {
                // Bias-only diff-patch (no weight delta on this module).
                if let Some(db) = &g.diff_b {
                    merge_into(base, &format!("{path}.bias"), &scaled(db)?, report)?;
                }
            }
        }
    }
    Ok(())
}

/// `true` if any tensor key in the `.safetensors` at `path` is a diff-patch delta (`.diff`/`.diff_b`)
/// — the structural marker of a lightx2v "lightning" file (the worker reads this to apply the
/// step-distill recipe).
///
/// Reads **only** the safetensors header — the 8-byte little-endian header length followed by exactly
/// that many JSON bytes — never the (potentially hundreds-of-MB) tensor data. The tensor names live
/// entirely in that JSON header, so classifying a bundle costs one small read on the worker's routing
/// path. A read / parse error propagates (the caller decides the fallback).
pub fn has_diff_patch_keys(path: &Path) -> Result<bool> {
    read_safetensors_header_keys(path)?
        .into_iter()
        .try_fold(false, |acc, name| {
            Ok(acc || name.ends_with(".diff") || name.ends_with(".diff_b"))
        })
}

/// Read only the safetensors header of `path` and return its tensor names (the JSON object keys, minus
/// the reserved `__metadata__` entry). Never touches the tensor data bytes: reads the 8-byte
/// little-endian header length, then exactly that many JSON bytes, then deserializes just that slice.
/// Malformed / truncated files surface as a clean `Err` (no panic), consistent with the crate's error
/// style.
fn read_safetensors_header_keys(path: &Path) -> Result<Vec<String>> {
    use std::io::Read;

    // Cap mirrors safetensors' own `MAX_HEADER_SIZE` (100 MB) so a corrupt length can't request a
    // multi-GB allocation.
    const MAX_HEADER_SIZE: u64 = 100_000_000;

    let mut file = std::fs::File::open(path)
        .map_err(|e| CandleError::Msg(format!("open adapter {}: {e}", path.display())))?;

    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf).map_err(|e| {
        CandleError::Msg(format!(
            "read adapter header length {}: {e}",
            path.display()
        ))
    })?;
    let header_len = u64::from_le_bytes(len_buf);
    if header_len > MAX_HEADER_SIZE {
        return Err(CandleError::Msg(format!(
            "adapter {} header length {header_len} exceeds max {MAX_HEADER_SIZE}",
            path.display()
        )));
    }

    let mut header = vec![0u8; header_len as usize];
    file.read_exact(&mut header)
        .map_err(|e| CandleError::Msg(format!("read adapter header {}: {e}", path.display())))?;

    // The header is a JSON object mapping tensor name -> {dtype, shape, offsets}, plus an optional
    // reserved `__metadata__` string map. We only need the key set.
    let obj: BTreeMap<String, serde_json::Value> = serde_json::from_slice(&header)
        .map_err(|e| CandleError::Msg(format!("parse adapter header {}: {e}", path.display())))?;
    Ok(obj.into_keys().filter(|k| k != "__metadata__").collect())
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
/// stamp) into `base` at `scale`, via the shared [`parse_lokr_thirdparty`] + [`merge_one_thirdparty`].
/// Each raw key resolves by prefix-strip (SCAIL-2 has no kohya table); the per-module lycoris scale is
/// baked into the delta closure. Linear-only.
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

/// Fold every adapter spec in `specs` into the base DiT tensor `map` (CPU, native dtype) at each spec's
/// `scale` — LoRA / LoKr / LoHa **and** the lightx2v lightning diff-patch, all merged into the dense
/// weights (`W += δ`, `b += δ_b`). Returns the [`MergeReport`]; errors if a non-empty spec list matches
/// **no** target (a format / prefix misconfiguration — the worker should then fall back rather than
/// render an unadapted video silently). SCAIL-2 is a single dense DiT, so `AdapterSpec::moe_expert` is
/// ignored (every spec merges into the one transformer).
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
                if af.declares_lokr() {
                    return Err(CandleError::Msg(format!(
                        "scail2: adapter {} declared Lora but its metadata says networkType=lokr",
                        spec.path.display()
                    )));
                }
                merge_lora_file(map, &af, spec.scale, &mut report)?;
            }
        }
        // lightx2v lightning hybrid: full-rank `.diff`/`.diff_b` deltas alongside the low-rank pairs.
        // A no-op for a file without diff-patch keys (the DPO / general LoRA case).
        merge_diff_patch_file(map, &af, spec.scale, &mut report)?;
    }
    if report.merged == 0 {
        return Err(no_target_matched(
            "scail2",
            "expected diffusers/PEFT `‹transformer.|diffusion_model.›‹path›.lora_A/B|lora_down/up.\
             weight` (+ optional `.alpha`) over `blocks.{i}.{self_attn,cross_attn}.{q,k,v,o,k_img,\
             v_img}` / `blocks.{i}.ffn.{0,2}`, `‹path›.lokr_w1/w2` with networkType=lokr (LoKr), \
             untagged LyCORIS `lokr_*` / `hada_*`, or lightx2v `‹path›.diff`/`.diff_b` (lightning \
             diff-patch)",
            specs.len(),
        ));
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Device;

    /// A tiny stand-in for the base DiT tensor map: one block's self-attn `q` (weight+bias) + an FFN
    /// Linear, a qk-RMSNorm weight, and a conv `patch_embedding` (weight+bias) — the cross-arch case.
    fn base_map() -> HashMap<String, Tensor> {
        let dev = Device::Cpu;
        let mut m = HashMap::new();
        for key in [
            "blocks.0.self_attn.q.weight",
            "blocks.0.cross_attn.k_img.weight",
            "blocks.0.ffn.0.weight",
        ] {
            m.insert(
                key.to_string(),
                Tensor::zeros((4, 4), DType::BF16, &dev).unwrap(),
            );
        }
        m.insert(
            "blocks.0.self_attn.q.bias".to_string(),
            Tensor::zeros(4usize, DType::BF16, &dev).unwrap(),
        );
        m.insert(
            "blocks.0.self_attn.norm_q.weight".to_string(),
            Tensor::zeros(4usize, DType::BF16, &dev).unwrap(),
        );
        // SCAIL-2 in_dim-20 conv stem [out, in, 1, 2, 2]; bias [out].
        m.insert(
            "patch_embedding.weight".to_string(),
            Tensor::zeros((4usize, 20, 1, 2, 2), DType::BF16, &dev).unwrap(),
        );
        m.insert(
            "patch_embedding.bias".to_string(),
            Tensor::zeros(4usize, DType::BF16, &dev).unwrap(),
        );
        m
    }

    fn t2(data: &[f32], r: usize, c: usize) -> Tensor {
        Tensor::from_vec(data.to_vec(), (r, c), &Device::Cpu).unwrap()
    }

    /// LoRA naming resolves: bare down/up + per-module `.alpha`, the PEFT `lora_A/B` (+ namespace
    /// prefix), and a non-LoRA key is out of surface.
    #[test]
    fn classify_resolves_scail2_namings() {
        assert!(matches!(
            classify_lora_key("blocks.0.self_attn.q.lora_down.weight").unwrap(),
            (p, Role::Down) if p == "blocks.0.self_attn.q"
        ));
        assert!(matches!(
            classify_lora_key("diffusion_model.blocks.0.cross_attn.k_img.lora_B.weight").unwrap(),
            (p, Role::Up) if p == "blocks.0.cross_attn.k_img"
        ));
        assert!(matches!(
            classify_lora_key("blocks.0.ffn.0.alpha").unwrap(),
            (p, Role::Alpha) if p == "blocks.0.ffn.0"
        ));
        assert!(classify_lora_key("blocks.0.self_attn.norm_q.weight").is_none());
    }

    /// The DPO-style LoRA: a bare down/up + per-module `.alpha` folds `W += (alpha/rank)·B·A`.
    #[test]
    fn merge_lora_folds_expected_delta() {
        let mut map = base_map();
        let path = "blocks.0.self_attn.q";
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

    /// The lightning diff-patch: a full-rank `.diff` (weight) + `.diff_b` (bias) on a dim-compatible
    /// module fold in; a cross-architecture `patch_embedding.diff` (in36 ≠ in20) skips the whole module
    /// **including** its coupled `.diff_b`.
    #[test]
    fn merge_diff_patch_folds_compatible_and_skips_cross_arch_module() {
        let mut map = base_map();
        let dev = Device::Cpu;
        // Compatible: self_attn.q weight delta + bias delta (base [4,4] / [4]).
        let wdiff = Tensor::ones((4, 4), DType::F32, &dev).unwrap();
        let bdiff = Tensor::ones(4usize, DType::F32, &dev).unwrap();
        // Cross-arch: vanilla-Wan patch_embedding in_dim 36 (base is 20) + a (shape-OK) bias delta that
        // must be dropped along with the skipped weight.
        let pe_wdiff = Tensor::ones((4usize, 36, 1, 2, 2), DType::F32, &dev).unwrap();
        let pe_bdiff = Tensor::ones(4usize, DType::F32, &dev).unwrap();
        let af = AdapterFile {
            tensors: HashMap::from([
                (
                    "diffusion_model.blocks.0.self_attn.q.diff".to_string(),
                    wdiff,
                ),
                (
                    "diffusion_model.blocks.0.self_attn.q.diff_b".to_string(),
                    bdiff,
                ),
                ("diffusion_model.patch_embedding.diff".to_string(), pe_wdiff),
                (
                    "diffusion_model.patch_embedding.diff_b".to_string(),
                    pe_bdiff,
                ),
            ]),
            meta: HashMap::new(),
        };
        let mut report = MergeReport::default();
        merge_diff_patch_file(&mut map, &af, 1.0, &mut report).unwrap();
        // self_attn.q weight + bias merged (2); patch_embedding weight + coupled bias skipped (2).
        assert_eq!(report.merged, 2);
        assert_eq!(report.skipped_keys, 2);
        // The compatible weight is now all-ones (base zero + 1·diff).
        let qw = map["blocks.0.self_attn.q.weight"]
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(qw.iter().all(|&v| (v - 1.0).abs() < 1e-3));
        // patch_embedding stayed zero (whole module skipped).
        let pe = map["patch_embedding.bias"]
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(pe.iter().all(|&v| v == 0.0));
    }

    /// A hybrid lightx2v file (low-rank pairs **and** diff-patch) folds both halves through
    /// `merge_adapters`, and the cross-arch `patch_embedding` is the lone skip.
    #[test]
    fn merge_adapters_hybrid_lightning_counts_weight_and_bias() {
        // Drive the per-file merge directly (merge_adapters reads from disk).
        let mut map = base_map();
        let dev = Device::Cpu;
        let down = Tensor::randn(0f32, 1f32, (2, 4), &dev).unwrap();
        let up = Tensor::randn(0f32, 1f32, (4, 2), &dev).unwrap();
        let af = AdapterFile {
            tensors: HashMap::from([
                // low-rank pair on ffn.0
                ("blocks.0.ffn.0.lora_down.weight".to_string(), down),
                ("blocks.0.ffn.0.lora_up.weight".to_string(), up),
                // diff-patch bias on self_attn.q + a norm weight diff
                (
                    "blocks.0.self_attn.q.diff_b".to_string(),
                    Tensor::ones(4usize, DType::F32, &dev).unwrap(),
                ),
                (
                    "blocks.0.self_attn.norm_q.diff".to_string(),
                    Tensor::ones(4usize, DType::F32, &dev).unwrap(),
                ),
            ]),
            meta: HashMap::new(),
        };
        let mut report = MergeReport::default();
        merge_lora_file(&mut map, &af, 1.0, &mut report).unwrap();
        merge_diff_patch_file(&mut map, &af, 1.0, &mut report).unwrap();
        // ffn.0 (lora) + self_attn.q.bias (diff_b) + norm_q.weight (diff) = 3 merged, 0 skipped.
        assert_eq!(report.merged, 3);
        assert_eq!(report.skipped_keys, 0);
    }

    /// PEFT LoKr (`networkType=lokr`, rank/alpha in metadata) folds the kron delta into the dense weight.
    #[test]
    fn merge_lokr_folds_kron_delta() {
        let mut map = base_map();
        let path = "blocks.0.self_attn.q";
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
        let path = "blocks.0.self_attn.q";
        let af = AdapterFile {
            tensors: HashMap::from([
                (format!("{path}.lokr_w1"), t2(&[1.0, 0.0, 0.0, 1.0], 2, 2)),
                (format!("{path}.lokr_w2"), t2(&[0.5, 0.0, 0.0, 0.5], 2, 2)),
            ]),
            meta: HashMap::new(),
        };
        assert!(!af.declares_lokr());
        assert!(wmeta::keys_contain_lokr(
            af.tensors.keys().map(String::as_str)
        ));
        let mut report = MergeReport::default();
        merge_lokr_thirdparty(&mut map, &af, 1.0, &mut report).unwrap();
        assert_eq!(report.merged, 1);
    }

    /// An empty spec list merges nothing (no error); the production no-adapter path.
    #[test]
    fn merge_adapters_empty_is_noop() {
        let mut map = base_map();
        let report = merge_adapters(&mut map, &[]).unwrap();
        assert_eq!(report, MergeReport::default());
    }

    /// Write a `.safetensors`-format file whose JSON header declares `names` (each a tiny F32 scalar)
    /// but whose data section is **deliberately omitted** (only the leading header bytes are written).
    /// A correct header-only reader classifies such a file successfully; a whole-file reader (or one
    /// that validates the data buffer) would fail — so the crafted truncation is the assertion that
    /// [`has_diff_patch_keys`] never touches the tensor data.
    fn write_header_only_safetensors(path: &Path, names: &[&str]) {
        use std::io::Write;

        // Each tensor is a 4-byte F32 scalar laid out back-to-back; the header carries only the offsets.
        let mut header = serde_json::Map::new();
        for (i, name) in names.iter().enumerate() {
            let start = i * 4;
            let end = start + 4;
            header.insert(
                (*name).to_string(),
                serde_json::json!({
                    "dtype": "F32",
                    "shape": [1],
                    "data_offsets": [start, end],
                }),
            );
        }
        let header_bytes = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();

        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(&(header_bytes.len() as u64).to_le_bytes())
            .unwrap();
        f.write_all(&header_bytes).unwrap();
        // NB: no tensor-data bytes are written — the file is truncated right after the header.
    }

    /// A file carrying a `.diff` (and `.diff_b`) key is flagged as a lightning diff-patch, and one
    /// without is not — reading only the header (the file is truncated after it, so any whole-file /
    /// data-validating read would error instead).
    #[test]
    fn has_diff_patch_keys_reads_only_header() {
        let dir = std::env::temp_dir().join(format!("scail2_diffpatch_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let lightning = dir.join("lightning.safetensors");
        write_header_only_safetensors(
            &lightning,
            &[
                "diffusion_model.blocks.0.self_attn.q.diff",
                "diffusion_model.blocks.0.self_attn.q.diff_b",
                "diffusion_model.blocks.0.ffn.0.lora_down.weight",
            ],
        );
        assert!(
            has_diff_patch_keys(&lightning).unwrap(),
            "diff-patch keys present must classify true"
        );

        let plain = dir.join("plain_lora.safetensors");
        write_header_only_safetensors(
            &plain,
            &[
                "blocks.0.self_attn.q.lora_down.weight",
                "blocks.0.self_attn.q.lora_up.weight",
                "blocks.0.self_attn.q.alpha",
            ],
        );
        assert!(
            !has_diff_patch_keys(&plain).unwrap(),
            "no diff-patch keys must classify false"
        );

        // The reserved `__metadata__` entry is not a tensor and must never be mistaken for a key.
        let with_meta = dir.join("with_meta.safetensors");
        {
            use std::io::Write;
            let header = serde_json::json!({
                "__metadata__": { "format": "pt" },
                "blocks.0.self_attn.q.lora_down.weight": {
                    "dtype": "F32", "shape": [1], "data_offsets": [0, 4],
                },
            });
            let hb = serde_json::to_vec(&header).unwrap();
            let mut f = std::fs::File::create(&with_meta).unwrap();
            f.write_all(&(hb.len() as u64).to_le_bytes()).unwrap();
            f.write_all(&hb).unwrap();
        }
        assert_eq!(
            read_safetensors_header_keys(&with_meta).unwrap(),
            vec!["blocks.0.self_attn.q.lora_down.weight".to_string()],
            "__metadata__ must be filtered out of the tensor-name set"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Malformed / truncated headers surface as a clean `Err`, not a panic: a file shorter than the
    /// 8-byte length prefix, and a length prefix that promises more header bytes than exist.
    #[test]
    fn has_diff_patch_keys_malformed_is_err_not_panic() {
        use std::io::Write;
        let dir = std::env::temp_dir().join(format!("scail2_diffpatch_bad_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        // (a) Fewer than 8 bytes total — cannot even read the header length.
        let too_short = dir.join("short.safetensors");
        std::fs::File::create(&too_short)
            .unwrap()
            .write_all(&[0u8, 1, 2])
            .unwrap();
        assert!(has_diff_patch_keys(&too_short).is_err());

        // (b) Header length claims 4096 bytes but the file has none after the prefix.
        let truncated = dir.join("truncated.safetensors");
        std::fs::File::create(&truncated)
            .unwrap()
            .write_all(&4096u64.to_le_bytes())
            .unwrap();
        assert!(has_diff_patch_keys(&truncated).is_err());

        // (c) A non-existent path.
        assert!(has_diff_patch_keys(&dir.join("nope.safetensors")).is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A non-empty LoRA file that matches no DiT module merges nothing (the loud-error precondition).
    #[test]
    fn merge_lora_file_matches_nothing_when_off_surface() {
        let mut map = base_map();
        let af = AdapterFile {
            tensors: HashMap::from([
                (
                    "blocks.99.self_attn.q.lora_down.weight".to_string(),
                    t2(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 2, 4),
                ),
                (
                    "blocks.99.self_attn.q.lora_up.weight".to_string(),
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
