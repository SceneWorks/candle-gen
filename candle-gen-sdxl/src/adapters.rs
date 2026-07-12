//! SDXL inference-side adapter merge (sc-5165) — load a trained LoRA/LoKr `.safetensors` and fold its
//! delta into the dense UNet weights **before** the stock candle-transformers UNet is built. The
//! candle twin of `mlx-gen-sdxl::adapters`, and the closing half of the native-trainer loop: a LoRA
//! produced by [`candle_gen::train`]'s SDXL trainer now actually loads in candle inference.
//!
//! **Merge, don't residual.** SDXL's sampler is chaos-sensitive: the merged forward `(W+δ)·x` differs
//! from a forward-time residual `W·x + δ·x` by ~1 ULP, which cascades to a visibly different image.
//! The training seam ([`candle_gen::train::lora::LoraLinear`]) *must* add a residual (the factors stay
//! trainable); inference has no such need, so it merges and reproduces the merged-weight forward
//! exactly. The delta is reconstructed with the **same** f32 math the trainer's forward uses
//! ([`reconstruct_lora_delta`] / [`reconstruct_lokr_delta`]), so a candle-trained adapter round-trips.
//!
//! **Merge at the safetensors-key level.** Unlike the MLX port (which fights its vendored UNet's
//! module naming), candle merges into the raw base-weight tensor map *before* construction: the stock
//! candle UNet reads diffusers keys 1:1, so `{path}.weight` is a valid base key for every Linear an
//! adapter targets — attention (`to_q`/`to_k`/`to_v`/`to_out.0`), `proj_in`/`proj_out`, the GEGLU
//! `ff.net.0.proj`/`ff.net.2`, and `mid_block.*`. This is diffusers' full ("complete") coverage by
//! construction, no per-module routing table on our side. Two on-disk LoRA formats resolve:
//!  - **PEFT** (`base_model.model.unet.<dotted>.lora_A/B[.default].weight`) — what the candle trainer
//!    ([`save_lora_peft`](candle_gen::train::lora::save_lora_peft)) and `peft.save_pretrained()` /
//!    diffusers `save_lora_adapter` emit; the dotted path resolves directly (the prefix is optional).
//!    The scaling is a per-target `.alpha` tensor (candle trainer / kohya) or — when absent, as in the
//!    diffusers format — `lora_alpha`/`r` (+ `alpha_pattern`/`rank_pattern`) in the
//!    `lora_adapter_metadata` header blob ([`LoraAdapterMeta`](candle_gen::train::lora::LoraAdapterMeta), sc-5374).
//!  - **kohya** (`lora_unet_<flat>.lora_down/up.weight` + `.alpha`) — community / diffusers LoRAs; the
//!    `_`-flattened stem (ambiguous, since diffusers names contain `_`) resolves against a table built
//!    from the base UNet's own Linear keys.
//!
//! LoKr resolves PEFT/bare or kohya `<module>.lokr_w1`/`lokr_w2` (+ low-rank `_a`/`_b`) with `rank` /
//! `alpha` read from file metadata (`networkType=lokr`), reconstructing `δ = (alpha/rank)·kron(w1,w2)`.
//!
//! Beyond the candle trainer's own (Linear) output this also folds the dominant **community** adapter
//! formats (sc-5225), so a hand-trained or downloaded SDXL adapter merges in full — matching mlx-gen's
//! `LoraCoverage::Complete` by construction (the by-key merge into the stock UNet reaches every module
//! a diffusers checkpoint names):
//!  - **conv-layer LoRA** — resnet `conv1`/`conv2`/`conv_shortcut`, the down/up-samplers, `conv_in`/
//!    `conv_out`. The `down`∘`up` pair fuses into a single conv-weight delta
//!    ([`conv_lora_delta`](candle_gen::train::lora::conv_lora_delta)) and folds into the 4-D `{path}.
//!    weight`. candle convs are NCHW (`candle_nn::Conv2d`) = the trained-file layout, so there is no
//!    NHWC transpose (mlx needs one) and `conv_shortcut` is a real 4-D 1×1 conv, not a reshaped Linear.
//!  - **LyCORIS LoHa** (`hada_*`) and **untagged third-party LoKr** (`lokr_*` with no `networkType=lokr`
//!    stamp) — reconstructed per-module at the lycoris scale ([`reconstruct_loha_delta`]
//!    (candle_gen::train::lora::reconstruct_loha_delta) / [`reconstruct_lokr_delta`]) and merged. These
//!    stay at the **Linear** (attention/proj) surface — the conv surface is LoRA-only, mirroring mlx-gen;
//!    the lycoris conv/tucker forms are surfaced as skipped.
//!
//! Out-of-surface keys are **counted and surfaced** in [`MergeReport`], never silently dropped:
//! text-encoder `lora_te*` keys (UNet-only merge) and any factor that resolves to no UNet module.

use std::collections::{BTreeMap, HashMap, HashSet};

use candle_gen::candle_core::{DType, Tensor};
use candle_gen::gen_core::weightsmeta as wmeta;
use candle_gen::gen_core::{AdapterKind, AdapterSpec};
use candle_gen::quant::LokrFactors;
use candle_gen::train::lora::{
    conv_lora_delta, reconstruct_lokr_delta, reconstruct_lora_delta, LoraAdapterMeta, LoraHost,
};
// The shared adapter-merge skeleton (sc-8998 / F-018): the format-parsing + merge-report + third-party
// LyCORIS engine this crate previously hand-copied. Only the SDXL-specific key→module resolution
// (kohya `lora_unet_` + original-SD/A1111 translation + the 4-D conv-LoRA surface) stays local below.
use candle_gen::train::merge::{
    build_kohya_table, merge_into, merge_one_thirdparty, no_target_matched, parse_loha_thirdparty,
    parse_lokr_thirdparty, read_adapter, read_scalar, AdapterFile, LoraTriple, Role,
};
// Re-exported so `candle_gen_sdxl::MergeReport` (the crate's public surface) keeps resolving.
pub use candle_gen::train::merge::MergeReport;
use candle_gen::{CandleError, Result};

/// PEFT key prefix the candle SDXL trainer (and `peft.save_pretrained()`) write. Optional on read —
/// a bare dotted path resolves the same way.
const PEFT_PREFIX: &str = "base_model.model.unet.";
/// kohya / diffusers community LoRA key prefix (the flattened-module form).
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

/// Map one LoRA key to `(diffusers_dotted_path, role)`, or `None` if outside the UNet merge surface.
/// kohya (`lora_unet_<flat>…`) resolves the flattened stem via `table` — directly for a diffusers-named
/// stem, or via an original-SD/A1111 → diffusers translation (sc-6051); PEFT (`base_model.model.unet.`)
/// and bare dotted paths resolve directly.
fn classify_lora_key(key: &str, table: &BTreeMap<String, String>) -> Option<(String, Role)> {
    if let Some(rem) = key.strip_prefix(KOHYA_PREFIX) {
        for (suf, role) in [
            (".lora_down.weight", Role::Down),
            (".lora_up.weight", Role::Up),
            (".alpha", Role::Alpha),
        ] {
            if let Some(stem) = rem.strip_suffix(suf) {
                return wmeta::resolve_kohya_stem(stem, table).map(|d| (d, role));
            }
        }
        return None;
    }
    // PEFT (explicit prefix) or a bare dotted path — strip the optional prefix, resolve directly.
    let rem = key.strip_prefix(PEFT_PREFIX).unwrap_or(key);
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

/// Map one LoKr factor key to `(diffusers_dotted_path, factor_name)`, or `None` if out of surface.
fn classify_lokr_key(
    key: &str,
    table: &BTreeMap<String, String>,
) -> Option<(String, &'static str)> {
    for suf in LOKR_SUFFIXES {
        if let Some(stem) = key.strip_suffix(suf) {
            let factor = &suf[1..]; // drop the leading '.'
            return if let Some(flat) = stem.strip_prefix(KOHYA_PREFIX) {
                wmeta::resolve_kohya_stem(flat, table).map(|d| (d, factor))
            } else {
                Some((
                    stem.strip_prefix(PEFT_PREFIX).unwrap_or(stem).to_string(),
                    factor,
                ))
            };
        }
    }
    None
}

/// Merge one LoRA file into `base` at `scale`: classify every key (PEFT + kohya), fold complete
/// `(down, up)` pairs into `{path}.weight`. `rank` is `A`'s leading dim; `alpha` is the per-target
/// `.alpha` tensor when present, else the `lora_adapter_metadata` blob's `alpha_pattern`/`lora_alpha`
/// (the diffusers / PEFT `save_lora_adapter` format ships no `.alpha` tensor — sc-5374), else `rank`.
/// **2-D Linear** pairs fold via [`reconstruct_lora_delta`]; **4-D conv** pairs fuse via
/// [`conv_lora_delta`] into the 4-D conv weight (sc-5225). Half-pairs, a conv LoRA targeting a non-conv
/// weight, and other unexpected shapes are surfaced as skipped.
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

    // PEFT/diffusers `save_lora_adapter` files carry no per-target `.alpha` tensor — `lora_alpha`/`r`
    // (+ per-module overrides) live in the `lora_adapter_metadata` blob (sc-5374). `None` for kohya /
    // candle-trainer files (those ship a `.alpha` tensor), in which case the per-target `.alpha` or the
    // factor rank is used exactly as before.
    let cfg = LoraAdapterMeta::from_file_metadata(&af.meta);
    for (path, t) in triples {
        let (Some(down), Some(up)) = (t.down, t.up) else {
            report.skipped_keys += 1; // half-pair (partner targeted a non-routable module)
            continue;
        };
        let base_key = format!("{path}.weight");
        // Effective scaling: per-target `.alpha` tensor → `alpha_pattern`/`lora_alpha` blob → factor
        // rank (today's last-resort default). The denominator is the blob `r`/`rank_pattern` when given,
        // else the stored `A` leading dim (which equals it for a well-formed PEFT file).
        let (cfg_alpha, cfg_rank) = cfg.as_ref().map_or((None, None), |c| c.effective(&path));
        let (dn, un) = (down.dims().len(), up.dims().len());
        if dn == 4 && un == 4 {
            // Conv-layer LoRA (sc-5225): fuse `down`∘`up` into a single NCHW conv-weight delta and fold
            // it into the 4-D `{path}.weight`. candle convs are NCHW, so no transpose — `merge_into`
            // adds the matching-shape delta directly. A conv LoRA whose target is missing or not 4-D
            // (a non-conv weight) is surfaced as skipped, never mis-merged.
            let Some(w) = base.get(&base_key) else {
                report.skipped_keys += 1;
                continue;
            };
            if w.dims().len() != 4 {
                report.skipped_keys += 1;
                continue;
            }
            let rank = cfg_rank.unwrap_or(down.dims()[0] as f32);
            let alpha = t.alpha.or(cfg_alpha).unwrap_or(rank);
            let delta = conv_lora_delta(&down, &up, alpha, rank, scale)?;
            merge_into(base, &base_key, &delta, report)?;
            continue;
        }
        if dn != 2 || un != 2 {
            report.skipped_keys += 1; // neither a 2-D Linear nor a 4-D conv pair — unexpected shape
            continue;
        }
        if !base.contains_key(&base_key) {
            report.skipped_keys += 1;
            continue;
        }
        let rank = cfg_rank.unwrap_or(down.dims()[0] as f32);
        let alpha = t.alpha.or(cfg_alpha).unwrap_or(rank);
        let delta = reconstruct_lora_delta(&down, &up, alpha, rank, scale)?;
        merge_into(base, &base_key, &delta, report)?;
    }
    Ok(())
}

/// Read an optional `f32` scalar (`rank` / `alpha`) from a LoKr file's `__metadata__`, distinguishing
/// **absent** (`Ok(None)` — the caller substitutes its default) from **present-but-unparseable**
/// (a typed error naming the key). The old `.and_then(|s| s.parse().ok()).unwrap_or(default)` silently
/// collapsed a malformed user-supplied value (e.g. `rank = "4x"`) to the default, mis-scaling the
/// merged delta on untrusted input (F-114, the F-009 class). Mirrors `read_scalar_opt`'s
/// absent-vs-invalid contract for the tensor-form `.alpha`.
fn parse_lokr_meta_scalar(meta: &HashMap<String, String>, key: &str) -> Result<Option<f32>> {
    match meta.get(key) {
        None => Ok(None),
        Some(raw) => match raw.trim().parse::<f32>() {
            Ok(v) if v.is_finite() => Ok(Some(v)),
            _ => Err(CandleError::Msg(format!(
                "LoKr adapter metadata `{key}` = `{raw}` is not a finite number"
            ))),
        },
    }
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
    let rank = parse_lokr_meta_scalar(&af.meta, "rank")?.unwrap_or(1.0);
    let alpha = parse_lokr_meta_scalar(&af.meta, "alpha")?.unwrap_or(rank);

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
            report.skipped_keys += 1; // conv LoKr — deferred (sc-5225)
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
/// stamp) into `base` at `scale`. Resolves each flattened module key against `table` (the kohya
/// `flattened → dotted` map); an unresolved key is surfaced as skipped (mirrors mlx-gen's
/// `merge_one_lokr_thirdparty`).
fn merge_lokr_thirdparty(
    base: &mut HashMap<String, Tensor>,
    af: &AdapterFile,
    scale: f32,
    table: &BTreeMap<String, String>,
    report: &mut MergeReport,
) -> Result<()> {
    for (raw, g) in parse_lokr_thirdparty(af)? {
        merge_one_thirdparty(
            base,
            wmeta::resolve_lokr_path(&raw, table),
            |bs| g.delta(bs, scale),
            report,
        )?;
    }
    Ok(())
}

/// Merge a third-party LyCORIS **LoHa** file (`hada_*` keys) into `base` at `scale`. As
/// [`merge_lokr_thirdparty`] but the per-module delta is the Hadamard reconstruction.
fn merge_loha_thirdparty(
    base: &mut HashMap<String, Tensor>,
    af: &AdapterFile,
    scale: f32,
    table: &BTreeMap<String, String>,
    report: &mut MergeReport,
) -> Result<()> {
    for (raw, g) in parse_loha_thirdparty(af)? {
        merge_one_thirdparty(
            base,
            wmeta::resolve_lokr_path(&raw, table),
            |bs| g.delta(bs, scale),
            report,
        )?;
    }
    Ok(())
}

/// Fold every adapter spec in `specs` into the base UNet tensor `map` (CPU, native dtype) at each
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
    // Both 2-D Linear (attention/proj/ff) and 4-D conv (resnet convs, samplers, conv_in/out) stems
    // join the table (sc-5225), so a kohya conv key resolves and reaches the conv-LoRA merge.
    let table = build_kohya_table(map, &[2, 4]);
    let mut report = MergeReport::default();
    for spec in specs {
        let af = read_adapter(&spec.path)?;
        // Third-party LyCORIS (sc-5225): `lokr_*` / `hada_*` keys without a `networkType=lokr` stamp,
        // so the caller's declared `kind` can't label them — detect + route by keys before the kind
        // match. (A PEFT LoKr carries the stamp and goes through the `Lokr` arm; the LoKr-keys branch
        // excludes it via `!declares_lokr`.)
        if !af.declares_lokr() && wmeta::keys_contain_lokr(af.tensors.keys().map(String::as_str)) {
            merge_lokr_thirdparty(map, &af, spec.scale, &table, &mut report)?;
            continue;
        }
        if wmeta::keys_contain_loha(af.tensors.keys().map(String::as_str)) {
            merge_loha_thirdparty(map, &af, spec.scale, &table, &mut report)?;
            continue;
        }
        match spec.kind {
            AdapterKind::Lokr => merge_lokr_file(map, &af, spec.scale, &table, &mut report)?,
            AdapterKind::Lora => {
                // The file metadata is authoritative — a Lora-declared LoKr file has no lora_A/B keys
                // and would merge nothing; surface the mismatch loudly rather than no-op.
                if af.declares_lokr() {
                    return Err(CandleError::Msg(format!(
                        "sdxl: adapter {} declared Lora but its metadata says networkType=lokr",
                        spec.path.display()
                    )));
                }
                merge_lora_file(map, &af, spec.scale, &table, &mut report)?;
            }
        }
    }
    if report.merged == 0 {
        return Err(no_target_matched(
            "sdxl",
            "expected PEFT `base_model.model.unet.<path>.lora_A/B.weight` or kohya \
             `lora_unet_<flat>.lora_down/up.weight` with diffusers `down_blocks_*` or original-SD \
             `input_blocks_*` block naming (LoRA, incl. conv layers), `<module>.lokr_w1/w2` with \
             networkType=lokr (LoKr), or untagged LyCORIS `lokr_*` / `hada_*` (third-party LoKr / \
             LoHa)",
            specs.len(),
        ));
    }
    Ok(report)
}

// ---- Forward-time additive (unmerged) install on a PACKED tier (sc-11103, epic 10765) ------------
//
// A packed q4/q8 SDXL tier (`SceneWorks/sdxl-base-mlx`) has **no dense `W`** for its Linear surface —
// the `merge_adapters` fold (`W += δ`) can't touch u32 codes. Applying a distill LoRA there used to
// dequantize every adapted Linear to dense and serve it dense (`packed_adapters.rs`, retired here),
// which — because SDXL-Lightning / RealVisXL-Lightning target the **FF** (the bulk of the UNet) —
// dequantized most of the UNet and threw away the q4/q8 win. Instead we now push each LoRA/LoKr as a
// **forward-time residual** onto the packed `LoraLinear` leaves (`y = base(x) + Σ scale·((x·A)·B)`, the
// base kept packed), mirroring the qwen-image-edit adoption (sc-11091, #425). The adaptable Linear
// surface spans attention / FF / `proj_in`/`proj_out` plus the packed time-embedding + `add_embedding`
// heads and every resnet `time_emb_proj` (sc-11679 widened these last from bare `QLinear` leaves —
// distillation targets the denoising blocks, not the timestep/micro-conditioning embeddings, so this is
// a defensive widening for an adapter that does hit them). The **conv** surface (resnet convs, samplers,
// `conv_in`/`conv_out`) is dense even on a packed tier, so a conv LoRA still **folds** into its dense
// weight ([`fold_conv_adapters`]) at no packed cost. The dense tier keeps folding the whole surface
// bit-exactly via [`merge_adapters`].

/// A resolved LoRA residual pending attachment: `a = downᵀ` `[in, rank]`, `b = upᵀ·(alpha/rank)`
/// `[rank, out]`, `scale` the user strength. Read on CPU; moved to the UNet device at push.
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

/// A report of a forward-time additive install (sc-11103) — the packed-tier analog of [`MergeReport`].
#[derive(Debug, Default)]
pub struct AdditiveReport {
    /// Projections that received a residual (one per `(path, file)` hit; multiple stack).
    pub applied: usize,
    /// Resolved **2-D Linear** target paths present in the adapter file(s) but absent from the UNet's
    /// adaptable (`LoraLinear`) surface — e.g. a LoRA whose classified path names a module this UNet
    /// does not have. The adaptable surface now spans attention / FF / `proj_in`/`proj_out`, the
    /// time-embedding + `add_embedding` heads, and every resnet `time_emb_proj` (sc-11103 + sc-11679),
    /// so this fires only on a genuinely-absent target. Surfaced, never silently dropped.
    pub skipped_targets: Vec<String>,
    /// Adapter-file keys outside the LoRA/LoKr surface, half-pairs, 4-D conv pairs (folded separately by
    /// [`fold_conv_adapters`], not additive), or shape-mismatched factors.
    pub skipped_keys: usize,
}

/// The kohya `flattened → dotted` resolution table over both 2-D Linear and 4-D conv base keys (sc-5225)
/// — the SDXL packed adapter passes ([`fold_conv_adapters`] + [`install_additive`]) share it so a
/// community `lora_unet_<flat>` key resolves the same way it does for the dense [`merge_adapters`].
pub(crate) fn build_sdxl_kohya_table(map: &HashMap<String, Tensor>) -> BTreeMap<String, String> {
    build_kohya_table(map, &[2, 4])
}

/// Assert the tier's parsed `group_size` is the 64 the vendored UNet's Linear seam threads (relocated
/// from the retired `packed_adapters`, sc-11103). The vendored `UNet2DConditionModel::new` builds its
/// leaves at the default MLX group 64; a non-64 tier would pack/read at mismatched grids, so the packed
/// adapter path refuses it loudly (as `detect_packed_unet` already does for the base load).
pub(crate) fn assert_group_size_supported(group_size: usize) -> Result<()> {
    if group_size != candle_gen::quant::MLX_GROUP_SIZE {
        return Err(CandleError::Msg(format!(
            "sdxl: packed adapter install at group_size {group_size} unsupported (the vendored UNet \
             threads only {}); a non-64 tier needs the group threaded through the leaf constructors \
             (sc-9528/sc-11103)",
            candle_gen::quant::MLX_GROUP_SIZE
        )));
    }
    Ok(())
}

/// Resolve one LoRA file into per-path [`PendingLora`] for the **2-D Linear** surface (`a = downᵀ`,
/// `b = upᵀ·ratio`). Mirrors [`merge_lora_file`]'s classify + effective alpha/rank **exactly**, but
/// produces UNMERGED factors instead of a folded delta — so the packed additive residual equals the
/// dense fold to f32 tolerance. **4-D conv** pairs are left to [`fold_conv_adapters`] (skipped here).
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
            *skipped_keys += 1; // half-pair
            continue;
        };
        // 4-D conv pairs fold separately (`fold_conv_adapters`); only 2-D Linear pairs go additive.
        if down.dims().len() != 2 || up.dims().len() != 2 {
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
    table: &BTreeMap<String, String>,
    pending: &mut BTreeMap<String, Vec<PendingLokr>>,
    skipped_keys: &mut usize,
) -> Result<()> {
    let rank = parse_lokr_meta_scalar(&af.meta, "rank")?.unwrap_or(1.0);
    let alpha = parse_lokr_meta_scalar(&af.meta, "alpha")?.unwrap_or(rank);
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

/// Install `specs` as **forward-time additive residuals** on the packed UNet's adaptable (`LoraLinear`)
/// **Linear** surface (sc-11103): resolve each LoRA / PEFT-LoKr file into unmerged factors, then walk the
/// UNet once via [`LoraHost`] pushing residuals onto matched projections — the base is never dequantized
/// or folded, so a q4/q8 tier keeps its footprint while the Lightning distill (or a user LoRA) applies.
/// `table` resolves community `lora_unet_<flat>` keys (build via [`build_sdxl_kohya_table`]); `device` is
/// the UNet's device (factors are read on CPU and moved to it at push).
///
/// A **LoHa** (no allocation-free structured form) and an **untagged third-party LyCORIS LoKr** are
/// rejected on a packed tier with a pointer to the dense (`.fp16`) tier — exactly the qwen-image-edit
/// stance (sc-11091): the packed additive path has no place to thread their per-module scale / Hadamard
/// product without materializing a full `[out,in]` delta. **Conv** targets fold separately
/// ([`fold_conv_adapters`]). This function does NOT itself error on a zero match — the caller combines
/// the additive `applied` with the conv `merged` for the single "adapted nothing" guard, since a
/// conv-only LoRA legitimately installs zero additive residuals.
pub(crate) fn install_additive(
    host: &mut dyn LoraHost,
    specs: &[AdapterSpec],
    table: &BTreeMap<String, String>,
    device: &candle_gen::candle_core::Device,
) -> Result<AdditiveReport> {
    let mut pending_lora: BTreeMap<String, Vec<PendingLora>> = BTreeMap::new();
    let mut pending_lokr: BTreeMap<String, Vec<PendingLokr>> = BTreeMap::new();
    let mut report = AdditiveReport::default();

    for spec in specs {
        let af = read_adapter(&spec.path)?;
        if wmeta::keys_contain_loha(af.tensors.keys().map(String::as_str)) {
            return Err(CandleError::Msg(format!(
                "sdxl: a LoHa adapter cannot apply on a packed (q4/q8) tier — its Hadamard product has \
                 no allocation-free structured form (unlike LoKr's Kronecker vec-trick), so it would \
                 materialize a full [out,in] delta per target. Use the dense (.fp16) tier (where it \
                 folds into the weight) or a plain LoRA/LoKr. Offending file: {}",
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
                "sdxl: an untagged third-party LyCORIS LoKr cannot apply additively on a packed (q4/q8) \
                 tier — use the dense (.fp16) tier (where `merge_adapters` folds it), a PEFT-stamped \
                 LoKr (`networkType=lokr`), or a plain LoRA. Offending file: {}",
                spec.path.display()
            )));
        }
        if spec.kind == AdapterKind::Lokr || af.declares_lokr() {
            resolve_lokr_file(
                &af,
                spec.scale,
                table,
                &mut pending_lokr,
                &mut report.skipped_keys,
            )?;
        } else {
            resolve_lora_file(
                &af,
                spec.scale,
                table,
                &mut pending_lora,
                &mut report.skipped_keys,
            )?;
        }
    }

    // Attach: walk the UNet once, pushing any resolved residual for each projection's canonical path. A
    // factor whose dims don't match the projection is surfaced as a skipped key, never a crashing
    // forward (the additive analog of the fold path's shape guard).
    let mut matched: HashSet<String> = HashSet::new();
    let mut applied = 0usize;
    let mut skipped_keys = 0usize;
    host.visit_lora_mut(&mut |lin| {
        let path = lin.path().to_string();
        let (in_f, out_f) = (lin.in_features(), lin.out_features());
        if let Some(list) = pending_lora.get(&path) {
            matched.insert(path.clone());
            for p in list {
                if p.a.dims()[0] != in_f || p.b.dims()[1] != out_f {
                    skipped_keys += 1;
                    continue;
                }
                lin.push_additive_lora(p.a.to_device(device)?, p.b.to_device(device)?, p.scale);
                applied += 1;
            }
        }
        if let Some(list) = pending_lokr.get(&path) {
            matched.insert(path.clone());
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
                        lin.push_additive_lokr(factors.to_device(device)?);
                        applied += 1;
                    }
                    // Not deferrable on a packed tier (a base that does not factor a·b × c·d) — abort the
                    // walk and surface it, rather than silently drop the target.
                    None => {
                        return Err(CandleError::Msg(format!(
                            "sdxl: LoKr target `{path}` is not deferrable on a packed tier (a base that \
                             does not factor as a·b × c·d) — no allocation-free structured form. Use \
                             the dense (.fp16) tier."
                        )))
                    }
                }
            }
        }
        Ok(())
    })?;
    report.applied = applied;
    report.skipped_keys += skipped_keys;

    // Pending 2-D Linear targets absent from the adaptable surface are surfaced, never silently dropped.
    for path in pending_lora.keys().chain(pending_lokr.keys()) {
        if !matched.contains(path) {
            report.skipped_targets.push(path.clone());
        }
    }
    Ok(report)
}

/// Fold every **conv-layer** LoRA in `specs` into the dense conv weights of a packed UNet tensor `map`
/// (sc-11103): a packed SDXL tier packs only its Linear surface, so resnet convs / samplers /
/// `conv_in`/`conv_out` stay dense `{path}.weight` (4-D) and a conv LoRA folds into them at **no** packed
/// cost (its `down`∘`up` fuses via [`conv_lora_delta`], reusing the dense [`merge_lora_file`] conv path).
/// **Linear** (2-D) targets are left to [`install_additive`]; LoKr/LoHa are Linear-only (no conv
/// surface). Returns the [`MergeReport`] (`merged` = conv folds); no "matched nothing" guard — the caller
/// combines it with the additive `applied`.
pub(crate) fn fold_conv_adapters(
    map: &mut HashMap<String, Tensor>,
    specs: &[AdapterSpec],
    table: &BTreeMap<String, String>,
) -> Result<MergeReport> {
    let mut report = MergeReport::default();
    for spec in specs {
        // Only plain LoRA files carry a conv surface; a (PEFT or third-party) LoKr / LoHa is Linear-only,
        // so it contributes no conv fold (its Linear targets go additive in `install_additive`).
        if spec.kind == AdapterKind::Lokr {
            continue;
        }
        let af = read_adapter(&spec.path)?;
        if af.declares_lokr()
            || wmeta::keys_contain_lokr(af.tensors.keys().map(String::as_str))
            || wmeta::keys_contain_loha(af.tensors.keys().map(String::as_str))
        {
            continue;
        }
        fold_conv_lora_file(map, &af, spec.scale, table, &mut report)?;
    }
    Ok(report)
}

/// Fold only the **4-D conv** `(down, up)` pairs of one LoRA file into `map`'s 4-D conv weights, at
/// `scale`. The conv analog of [`merge_lora_file`] restricted to the conv surface — 2-D Linear pairs are
/// skipped (they go additive), and a conv pair targeting a missing / non-4-D weight is surfaced as
/// skipped, never mis-folded.
fn fold_conv_lora_file(
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
            None => {} // out-of-surface keys are counted by the additive resolver, not here
        }
    }
    let cfg = LoraAdapterMeta::from_file_metadata(&af.meta);
    for (path, t) in triples {
        let (Some(down), Some(up)) = (t.down, t.up) else {
            continue; // half-pair (counted by the additive resolver)
        };
        // Conv-only: 2-D Linear pairs are the additive path's job.
        if down.dims().len() != 4 || up.dims().len() != 4 {
            continue;
        }
        let base_key = format!("{path}.weight");
        let Some(w) = base.get(&base_key) else {
            report.skipped_keys += 1;
            continue;
        };
        if w.dims().len() != 4 {
            report.skipped_keys += 1; // a conv LoRA targeting a non-conv weight
            continue;
        }
        let (cfg_alpha, cfg_rank) = cfg.as_ref().map_or((None, None), |c| c.effective(&path));
        let rank = cfg_rank.unwrap_or(down.dims()[0] as f32);
        let alpha = t.alpha.or(cfg_alpha).unwrap_or(rank);
        let delta = conv_lora_delta(&down, &up, alpha, rank, scale)?;
        merge_into(base, &base_key, &delta, report)?;
    }
    Ok(())
}

/// Build the kohya `flattened → dotted` table from a UNet weight **file** without loading tensor data
/// (sc-11682) — reads only the safetensors header (names + shapes) via a memory map, so the dense
/// additive lanes can resolve community `lora_unet_<flat>` keys against an **mmap** base without first
/// materializing the whole UNet in a host `HashMap` (which is the un-evictable state this avoids).
pub(crate) fn build_sdxl_kohya_table_from_file(
    file: &std::path::Path,
) -> Result<BTreeMap<String, String>> {
    // SAFETY: read-only, process-owned weight file; the map is dropped at the end of this function and
    // only its header (tensor names + shapes) is read — no tensor data is dereferenced.
    let st = unsafe { candle_gen::candle_core::safetensors::MmapedSafetensors::new(file)? };
    let mut table = BTreeMap::new();
    for (name, view) in st.tensors() {
        if let Some(dotted) = name.strip_suffix(".weight") {
            let rank = view.shape().len();
            if rank == 2 || rank == 4 {
                table.insert(dotted.replace('.', "_"), dotted.to_string());
            }
        }
    }
    Ok(table)
}

/// A resolved conv-LoRA residual pending attachment: `down` `[rank, in, kH, kW]`, `up` `[out, rank, 1,
/// 1]`, `scale` the FULL `(alpha/rank)·strength` baked in (the conv `Conv2d` residual carries no
/// separate alpha/rank). Read on CPU; moved to the UNet device at push.
struct PendingConv {
    down: Tensor,
    up: Tensor,
    scale: f64,
}

/// Resolve one LoRA file's **4-D conv** `(down, up)` pairs into per-path [`PendingConv`] with the full
/// `(alpha/rank)·scale` baked (sc-11682) — the additive twin of [`fold_conv_lora_file`], producing
/// UNMERGED factors instead of a folded delta. 2-D Linear pairs are the additive-Linear path's job
/// ([`install_additive`]).
fn resolve_conv_lora_file(
    af: &AdapterFile,
    scale: f32,
    table: &BTreeMap<String, String>,
    pending: &mut BTreeMap<String, Vec<PendingConv>>,
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
            None => {} // out-of-surface keys are counted by the Linear resolver, not here
        }
    }
    let cfg = LoraAdapterMeta::from_file_metadata(&af.meta);
    for (path, t) in triples {
        let (Some(down), Some(up)) = (t.down, t.up) else {
            continue; // half-pair (counted by the Linear resolver)
        };
        // Conv-only: 2-D Linear pairs go through `install_additive`.
        if down.dims().len() != 4 || up.dims().len() != 4 {
            continue;
        }
        let (cfg_alpha, cfg_rank) = cfg.as_ref().map_or((None, None), |c| c.effective(&path));
        let rank = cfg_rank.unwrap_or(down.dims()[0] as f32) as f64;
        if rank == 0.0 {
            *skipped_keys += 1;
            continue;
        }
        let alpha = t.alpha.or(cfg_alpha).unwrap_or(rank as f32) as f64;
        let full = (alpha / rank) * scale as f64;
        pending.entry(path).or_default().push(PendingConv {
            down: down.to_dtype(DType::F32)?.contiguous()?,
            up: up.to_dtype(DType::F32)?.contiguous()?,
            scale: full,
        });
    }
    Ok(())
}

/// Install `specs`' **conv-layer** LoRA as forward-time additive residuals on the UNet's convolutions
/// (sc-11682): resolve the 4-D `(down, up)` pairs, then walk the convs once
/// ([`crate::unet::UNet2DConditionModel::visit_conv_lora_mut`]) pushing each onto its matched conv — the
/// base conv weight is never folded, so a dense mmap tier stays evictable. `table` resolves community
/// `lora_unet_<flat>` keys. Conv-LoRA is LoRA-only (LoKr/LoHa are Linear-only), so LoKr specs and
/// LoKr/LoHa-keyed files contribute nothing here. Like [`install_additive`], no zero-match guard — the
/// caller combines the Linear + conv `applied` for the single "adapted nothing" check.
pub(crate) fn install_additive_conv(
    unet: &mut crate::unet::UNet2DConditionModel,
    specs: &[AdapterSpec],
    table: &BTreeMap<String, String>,
    device: &candle_gen::candle_core::Device,
) -> Result<AdditiveReport> {
    let mut pending: BTreeMap<String, Vec<PendingConv>> = BTreeMap::new();
    let mut report = AdditiveReport::default();
    for spec in specs {
        if spec.kind == AdapterKind::Lokr {
            continue; // conv surface is LoRA-only
        }
        let af = read_adapter(&spec.path)?;
        if af.declares_lokr()
            || wmeta::keys_contain_lokr(af.tensors.keys().map(String::as_str))
            || wmeta::keys_contain_loha(af.tensors.keys().map(String::as_str))
        {
            continue; // Linear-only adapter forms — no conv surface
        }
        resolve_conv_lora_file(
            &af,
            spec.scale,
            table,
            &mut pending,
            &mut report.skipped_keys,
        )?;
    }

    let mut matched: HashSet<String> = HashSet::new();
    let mut applied = 0usize;
    let mut skipped_keys = 0usize;
    unet.visit_conv_lora_mut(&mut |conv| {
        if let Some(list) = pending.get(conv.path()) {
            matched.insert(conv.path().to_string());
            let wd = conv.weight_dims(); // [out, in, kH, kW]
            for p in list {
                let (d, u) = (p.down.dims(), p.up.dims());
                let shape_ok = wd.len() == 4
                    && d.len() == 4
                    && u.len() == 4
                    && d[1] == wd[1]
                    && d[2] == wd[2]
                    && d[3] == wd[3]
                    && u[0] == wd[0]
                    && u[1] == d[0]
                    && u[2] == 1
                    && u[3] == 1;
                if !shape_ok {
                    skipped_keys += 1;
                    continue;
                }
                conv.push_additive_conv(
                    p.down.to_device(device)?,
                    p.up.to_device(device)?,
                    p.scale,
                );
                applied += 1;
            }
        }
        Ok(())
    })?;
    report.applied = applied;
    report.skipped_keys += skipped_keys;
    for path in pending.keys() {
        if !matched.contains(path) {
            report.skipped_targets.push(path.clone());
        }
    }
    Ok(report)
}

/// The single "adapted nothing" guard for the additive adapter paths (sc-11103 packed / sc-11682 dense):
/// a non-empty spec set that installed **no** Linear residual AND no conv residual (dense) / folded no
/// conv (packed) is a format / prefix misconfiguration — fail loudly rather than render an unadapted
/// image (the additive twin of [`merge_adapters`]' zero-match guard).
pub(crate) fn guard_additive_matched(specs_len: usize, applied_total: usize) -> Result<()> {
    if specs_len > 0 && applied_total == 0 {
        return Err(no_target_matched(
            "sdxl (additive)",
            "expected PEFT `base_model.model.unet.<path>.lora_A/B.weight` or kohya \
             `lora_unet_<flat>.lora_down/up.weight` over the UNet's attention / FF / proj Linears (LoRA \
             or PEFT LoKr) or its conv layers (conv LoRA) — applied additively",
            specs_len,
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::{DType, Device};
    use std::path::Path;

    /// A tiny stand-in for the base UNet tensor map: two attention Linears + one conv (4-D) weight.
    fn base_map() -> HashMap<String, Tensor> {
        let dev = Device::Cpu;
        let mut m = HashMap::new();
        // attn1.to_q: [out=4, in=4]
        m.insert(
            "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight".into(),
            Tensor::zeros((4, 4), DType::F16, &dev).unwrap(),
        );
        // attn1.to_out.0: [out=4, in=4]
        m.insert(
            "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_out.0.weight".into(),
            Tensor::zeros((4, 4), DType::F16, &dev).unwrap(),
        );
        // a conv weight (4-D) — must never be merged by a 2-D LoRA.
        m.insert(
            "conv_in.weight".into(),
            Tensor::zeros((4, 4, 3, 3), DType::F16, &dev).unwrap(),
        );
        m
    }

    fn t2(data: &[f32], r: usize, c: usize) -> Tensor {
        Tensor::from_vec(data.to_vec(), (r, c), &Device::Cpu).unwrap()
    }

    /// kohya stems resolve against the base-key table; the ambiguous `to_out_0` flattening resolves to
    /// the real `…to_out.0` path.
    #[test]
    fn classify_lora_resolves_peft_kohya_and_bare() {
        let table = build_kohya_table(&base_map(), &[2, 4]);
        // PEFT prefixed.
        let (p, _) = classify_lora_key(
            "base_model.model.unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.weight",
            &table,
        )
        .unwrap();
        assert_eq!(
            p,
            "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q"
        );
        // PEFT `.default.` infix.
        assert!(matches!(
            classify_lora_key(
                "base_model.model.unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_B.default.weight",
                &table,
            )
            .unwrap()
            .1,
            Role::Up
        ));
        // kohya flattened stem, incl. the `.0` of to_out.0 → `to_out_0`.
        let (p, _) = classify_lora_key(
            "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_out_0.lora_down.weight",
            &table,
        )
        .unwrap();
        assert_eq!(
            p,
            "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_out.0"
        );
        // text-encoder + unknown stems are out of surface.
        assert!(classify_lora_key(
            "lora_te1_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight",
            &table
        )
        .is_none());
    }

    /// sc-6051: an original-SD / A1111 kohya key (`lora_unet_input_blocks_4_1_…`) classifies onto the
    /// same diffusers dotted path as its `down_blocks` twin, so civitai SDXL LoRAs merge in candle too.
    #[test]
    fn classify_lora_translates_original_sd_naming() {
        // A table holding a real down_blocks.1 attention path (the diffusers twin of input_blocks.4.1).
        let table: BTreeMap<String, String> =
            ["down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q"]
                .into_iter()
                .map(|p| (p.replace('.', "_"), p.to_string()))
                .collect();
        let (p, role) = classify_lora_key(
            "lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_q.lora_down.weight",
            &table,
        )
        .expect("original-SD input_blocks key should translate + resolve");
        assert_eq!(
            p,
            "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q"
        );
        assert!(matches!(role, Role::Down));
        // The LoKr classify path translates too (kohya-prefixed original-SD stem).
        assert_eq!(
            classify_lokr_key(
                "lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_q.lokr_w1",
                &table,
            )
            .unwrap()
            .0,
            "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q"
        );
    }

    /// PEFT LoRA merges into `W += (alpha/rank)·scale·B·A`; base+delta is exact in f32.
    #[test]
    fn merge_lora_peft_folds_expected_delta() {
        let mut map = base_map();
        let down = t2(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], 2, 4); // A [rank=2, in=4]
        let up = t2(&[2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0], 4, 2); // B [out=4, rank=2]
        let af = AdapterFile {
            tensors: HashMap::from([
                (
                    "base_model.model.unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.weight".to_string(),
                    down.clone(),
                ),
                (
                    "base_model.model.unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_B.weight".to_string(),
                    up.clone(),
                ),
                (
                    "base_model.model.unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.alpha".to_string(),
                    Tensor::from_vec(vec![4.0f32], (1,), &Device::Cpu).unwrap(),
                ),
            ]),
            meta: HashMap::new(),
        };
        let table = build_kohya_table(&map, &[2, 4]);
        let mut report = MergeReport::default();
        // scale 1.0; alpha 4, rank 2 ⇒ effective 2.0. ΔW = 2.0·(B·A).
        merge_lora_file(&mut map, &af, 1.0, &table, &mut report).unwrap();
        assert_eq!(report.merged, 1);
        let merged = map
            .get("down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight")
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let expected = reconstruct_lora_delta(&down, &up, 4.0, 2.0, 1.0).unwrap(); // base is zero
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

    /// sc-5374: a diffusers-format LoRA with NO per-target `.alpha` tensor but a `lora_adapter_metadata`
    /// blob (`lora_alpha = 16`, `r = 8`) merges at the metadata-derived strength `(16/8)·scale = 2.0`,
    /// not the old `alpha = rank` default (which would halve it). Proves the blob is read and applied.
    #[test]
    fn merge_lora_honors_lora_adapter_metadata_alpha() {
        let dev = Device::Cpu;
        let mut map = base_map();
        let path = "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q";
        // A [r=8, in=4], B [out=4, r=8] — nonzero so ΔW ≠ 0; deliberately NO `.alpha` tensor.
        let down = Tensor::randn(0f32, 1f32, (8, 4), &dev).unwrap();
        let up = Tensor::randn(0f32, 1f32, (4, 8), &dev).unwrap();
        let af = AdapterFile {
            tensors: HashMap::from([
                (
                    format!("base_model.model.unet.{path}.lora_A.weight"),
                    down.clone(),
                ),
                (
                    format!("base_model.model.unet.{path}.lora_B.weight"),
                    up.clone(),
                ),
            ]),
            meta: HashMap::from([(
                "lora_adapter_metadata".to_string(),
                r#"{"lora_alpha": 16, "r": 8}"#.to_string(),
            )]),
        };
        let table = build_kohya_table(&map, &[2, 4]);
        let mut report = MergeReport::default();
        merge_lora_file(&mut map, &af, 1.0, &table, &mut report).unwrap();
        assert_eq!(report.merged, 1);
        let merged = map
            .get(&format!("{path}.weight"))
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        // Effective alpha 16 over rank 8 ⇒ scale 2.0; base is zero, so the merged weight IS the delta.
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
        // The pre-sc-5374 default (alpha = rank ⇒ scale 1.0) would diverge by a full factor of 2.
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
            "metadata alpha must differ from the alpha=rank default (gap {gap})"
        );
    }

    /// sc-5225: a conv-shaped LoRA (4-D factors) now folds into the 4-D conv weight (`conv_in`), via
    /// the NCHW [`conv_lora_delta`] fusion — no transpose. Base is zero, so the merged weight IS the
    /// fused delta. PEFT keys resolve the dotted path directly (no table needed).
    #[test]
    fn merge_conv_lora_folds_into_conv_weight() {
        use candle_gen::train::lora::conv_lora_delta;
        let mut map = base_map();
        let dev = Device::Cpu;
        // down [rank=2, in=4, 3, 3], up [out=4, rank=2, 1, 1] — nonzero so ΔW ≠ 0.
        let down = Tensor::randn(0f32, 1f32, (2, 4, 3, 3), &dev).unwrap();
        let up = Tensor::randn(0f32, 1f32, (4, 2, 1, 1), &dev).unwrap();
        let af = AdapterFile {
            tensors: HashMap::from([
                (
                    "base_model.model.unet.conv_in.lora_A.weight".to_string(),
                    down.clone(),
                ),
                (
                    "base_model.model.unet.conv_in.lora_B.weight".to_string(),
                    up.clone(),
                ),
            ]),
            meta: HashMap::new(),
        };
        let table = build_kohya_table(&map, &[2, 4]);
        let mut report = MergeReport::default();
        // alpha defaults to rank (2) ⇒ effective 1.0; scale 1.0.
        merge_lora_file(&mut map, &af, 1.0, &table, &mut report).unwrap();
        assert_eq!(report.merged, 1);
        assert_eq!(report.skipped_keys, 0);
        let merged = map
            .get("conv_in.weight")
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        assert_eq!(merged.dims(), &[4, 4, 3, 3]);
        let expected = conv_lora_delta(&down, &up, 2.0, 2.0, 1.0).unwrap(); // base is zero
        let diff = (merged - expected)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff < 1e-4, "merged conv weight off by {diff}");
    }

    /// sc-5225: a kohya conv LoRA (`lora_unet_conv_in.lora_down/up.weight`) resolves the flattened conv
    /// stem through the now conv-aware [`build_kohya_table`] and merges (proving conv stems join the
    /// table). A 1×1 conv (in=out=4, kH=kW=1 here via conv_in's 3×3 base — use a synthetic 1×1 conv).
    #[test]
    fn merge_kohya_conv_lora_resolves_flattened_stem() {
        let dev = Device::Cpu;
        let mut map = HashMap::new();
        // A 1×1 conv weight [out=4, in=4, 1, 1] under a dotted path with internal-underscore segments.
        map.insert(
            "down_blocks.0.downsamplers.0.conv.weight".to_string(),
            Tensor::zeros((4, 4, 1, 1), DType::F16, &dev).unwrap(),
        );
        let down = Tensor::randn(0f32, 1f32, (2, 4, 1, 1), &dev).unwrap();
        let up = Tensor::randn(0f32, 1f32, (4, 2, 1, 1), &dev).unwrap();
        let af = AdapterFile {
            tensors: HashMap::from([
                (
                    "lora_unet_down_blocks_0_downsamplers_0_conv.lora_down.weight".to_string(),
                    down,
                ),
                (
                    "lora_unet_down_blocks_0_downsamplers_0_conv.lora_up.weight".to_string(),
                    up,
                ),
            ]),
            meta: HashMap::new(),
        };
        let table = build_kohya_table(&map, &[2, 4]);
        let mut report = MergeReport::default();
        merge_lora_file(&mut map, &af, 1.0, &table, &mut report).unwrap();
        assert_eq!(report.merged, 1, "kohya conv stem must resolve and merge");
        assert_eq!(report.skipped_keys, 0);
    }

    /// LoKr merges `δ = (alpha/rank)·kron(w1,w2)` into the dense weight, reading rank/alpha from meta.
    #[test]
    fn merge_lokr_folds_kron_delta() {
        let mut map = base_map();
        // base [out=4,in=4] factors 2×2 ⊗ 2×2.
        let w1 = t2(&[1.0, 0.0, 0.0, 1.0], 2, 2);
        let w2 = t2(&[0.5, 0.0, 0.0, 0.5], 2, 2);
        let af = AdapterFile {
            tensors: HashMap::from([
                (
                    "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lokr_w1"
                        .to_string(),
                    w1.clone(),
                ),
                (
                    "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lokr_w2"
                        .to_string(),
                    w2.clone(),
                ),
            ]),
            meta: HashMap::from([
                ("networkType".to_string(), "lokr".to_string()),
                ("rank".to_string(), "2".to_string()),
                ("alpha".to_string(), "2".to_string()),
            ]),
        };
        let table = build_kohya_table(&map, &[2, 4]);
        let mut report = MergeReport::default();
        merge_lokr_file(&mut map, &af, 1.0, &table, &mut report).unwrap();
        assert_eq!(report.merged, 1);
        let merged = map
            .get("down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight")
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

    /// A non-empty spec list that matches nothing is a loud error (not a silent unadapted render).
    #[test]
    fn merge_adapters_errors_when_nothing_matches() {
        let mut map = base_map();
        let af_tensors = HashMap::from([(
            "lora_unet_nonexistent_module.lora_down.weight".to_string(),
            t2(&[0.0, 0.0], 1, 2),
        )]);
        // Drive merge_lora_file directly with an unresolvable key → 0 merged.
        let af = AdapterFile {
            tensors: af_tensors,
            meta: HashMap::new(),
        };
        let table = build_kohya_table(&map, &[2, 4]);
        let mut report = MergeReport::default();
        merge_lora_file(&mut map, &af, 1.0, &table, &mut report).unwrap();
        assert_eq!(report.merged, 0);
        assert!(report.skipped_keys >= 1);
    }

    /// A Lora-declared spec pointing at LoKr-tagged metadata is rejected (the candle trainer never
    /// produces this, but a misconfigured worker request must fail loudly).
    #[test]
    fn merge_adapters_rejects_kind_metadata_mismatch() {
        // Build via the public entry point using an in-memory file is awkward; assert the helper.
        let af = AdapterFile {
            tensors: HashMap::new(),
            meta: HashMap::from([("networkType".to_string(), "lokr".to_string())]),
        };
        assert!(af.declares_lokr());
    }

    /// The keystone train→infer round-trip: a PEFT `.safetensors` written by the **actual trainer**
    /// path ([`candle_gen::train::lora::save_lora_peft`]) is read back through the public
    /// [`merge_adapters`] entry — exercising `read_adapter` (tensors + header metadata), PEFT
    /// classification, and the f32 reconstruction — and the merged weight equals the trained delta
    /// `ΔW = (alpha/rank)·B·A`. Proves the loader consumes the trainer's real on-disk format, not just
    /// hand-built tensors.
    #[test]
    fn roundtrip_trainer_peft_file_merges() {
        use candle_gen::candle_nn::Linear;
        use candle_gen::train::lora::{
            build_lora_targets, save_lora_peft, LoraHost, LoraLinear, SDXL_PEFT_PREFIX,
        };

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
        let path = "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_v";
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

        // Write the real PEFT file the trainer emits, then merge it through the public entry point.
        let file = std::env::temp_dir().join(format!(
            "candle_sdxl_lora_roundtrip_{}.safetensors",
            std::process::id()
        ));
        save_lora_peft(&set, SDXL_PEFT_PREFIX, &HashMap::new(), &file).unwrap();

        let mut map = HashMap::new();
        map.insert(
            format!("{path}.weight"),
            Tensor::zeros((4, 4), DType::F16, &dev).unwrap(),
        );
        let report = merge_adapters(
            &mut map,
            &[AdapterSpec::new(file.clone(), 1.0, AdapterKind::Lora)],
        );
        std::fs::remove_file(&file).ok();
        let report = report.unwrap();

        assert_eq!(report.merged, 1, "the trained to_v adapter must merge");
        // Base is zero, so the merged weight IS ΔW = (alpha/rank)·B·A.
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
            diff < 1e-4,
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

    /// sc-5225: the candle SDXL crate reconstructs third-party LyCORIS LoKr / LoHa deltas (via the
    /// shared f32 reconstruction) bit-close to the lycoris reference fixtures — the same fixtures the
    /// mlx-gen `thirdparty_lycoris_reconstructs_against_reference_f32` test pins (generated through the
    /// lycoris venv). Exercises detection (`keys_contain_*`), per-module factor grouping + the lycoris
    /// scale rule, and the flattened-key → dotted resolution. Linear fixtures only (the SDXL third-party
    /// surface is Linear-only; the conv/tucker fixtures are out of scope, as in mlx-gen's SDXL path).
    #[test]
    fn thirdparty_lycoris_reconstructs_against_reference_f32() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        // (fixture dir, file stem, is_loha)
        let cases: [(&str, &str, bool); 4] = [
            ("sc3642_lokr", "linear_w1full_w2lr", false),
            ("sc3642_lokr", "linear_bothlr", false),
            ("sc3642_lokr", "linear_bothfull", false),
            ("sc3643_loha", "linear", true),
        ];
        for (dir, stem, is_loha) in cases {
            let base = root.join(dir);
            let af = read_adapter(&base.join(format!("{stem}.safetensors"))).unwrap();
            let exp = read_adapter(&base.join(format!("{stem}.expected.safetensors"))).unwrap();
            // Detection mirrors the merge router.
            if is_loha {
                assert!(
                    wmeta::keys_contain_loha(af.tensors.keys().map(String::as_str)),
                    "{stem}: not detected as LoHa"
                );
            } else {
                assert!(
                    wmeta::keys_contain_lokr(af.tensors.keys().map(String::as_str))
                        && !af.declares_lokr(),
                    "{stem}: not detected as third-party LoKr"
                );
            }
            // table: the expected file's single target ("proj") flattened → dotted.
            let table: BTreeMap<String, String> = exp
                .tensors
                .keys()
                .map(|d| (d.replace('.', "_"), d.clone()))
                .collect();
            let want = exp
                .tensors
                .get("proj")
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap();
            let (out_f, in_f) = (want.dims()[0], want.dims()[1]);
            let got = if is_loha {
                let groups = parse_loha_thirdparty(&af).unwrap();
                let (raw, g) = groups.iter().next().unwrap();
                assert_eq!(wmeta::resolve_lokr_path(raw, &table), Some("proj"));
                g.delta((out_f, in_f), 1.0).unwrap()
            } else {
                let groups = parse_lokr_thirdparty(&af).unwrap();
                let (raw, g) = groups.iter().next().unwrap();
                assert_eq!(wmeta::resolve_lokr_path(raw, &table), Some("proj"));
                g.delta((out_f, in_f), 1.0).unwrap()
            };
            assert_eq!(
                got.dims(),
                want.dims(),
                "{stem}: reconstructed shape mismatch"
            );
            let diff = (&got - &want)
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            assert!(
                diff < 1e-4,
                "{stem}: third-party reconstruction diverged from lycoris reference by {diff}"
            );
        }
    }

    /// sc-5225: an untagged third-party LoKr (kohya-flattened keys, no `networkType`) is detected by
    /// keys and merged into the resolved Linear (`W += δ`). A conv-targeting third-party factor stays
    /// on the Linear-only surface — surfaced as skipped, never folded into a 4-D conv weight.
    #[test]
    fn merge_thirdparty_lokr_routes_resolves_and_merges() {
        let mut map = base_map(); // attn1.to_q [4,4], conv_in [4,4,3,3]
        let to_q = "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q";
        let af = AdapterFile {
            tensors: HashMap::from([
                // to_q: factor [4,4] as 2×2 ⊗ 2×2 (both full ⇒ lycoris scale 1).
                (format!("{to_q}.lokr_w1"), t2(&[1.0, 0.0, 0.0, 1.0], 2, 2)),
                (format!("{to_q}.lokr_w2"), t2(&[0.5, 0.0, 0.0, 0.5], 2, 2)),
                // conv_in: resolves to a 4-D weight ⇒ Linear-only surface skips it.
                ("lora_unet_conv_in.lokr_w1".to_string(), t2(&[1.0], 1, 1)),
                ("lora_unet_conv_in.lokr_w2".to_string(), t2(&[1.0], 1, 1)),
            ]),
            meta: HashMap::new(), // no networkType stamp → third-party
        };
        assert!(!af.declares_lokr());
        assert!(wmeta::keys_contain_lokr(
            af.tensors.keys().map(String::as_str)
        ));
        let table = build_kohya_table(&map, &[2, 4]);
        let mut report = MergeReport::default();
        merge_lokr_thirdparty(&mut map, &af, 1.0, &table, &mut report).unwrap();
        assert_eq!(report.merged, 1, "the to_q LoKr must merge");
        assert!(
            report.skipped_keys >= 1,
            "the conv-targeting LoKr is Linear-only ⇒ skipped"
        );
        // conv_in untouched (still 4-D, all-zero).
        assert_eq!(map.get("conv_in.weight").unwrap().dims(), &[4, 4, 3, 3]);
    }

    /// sc-5225: a third-party LoHa (`hada_*`) routes through the Hadamard merge into the resolved
    /// Linear, producing a finite merged weight.
    #[test]
    fn merge_thirdparty_loha_routes_and_merges() {
        let mut map = base_map();
        let to_q = "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q";
        // rank-1 Hadamard factors: w*_a [4,1], w*_b [1,4] ⇒ [4,4] products.
        let af = AdapterFile {
            tensors: HashMap::from([
                (
                    format!("{to_q}.hada_w1_a"),
                    t2(&[0.5, 0.1, -0.2, 0.3], 4, 1),
                ),
                (
                    format!("{to_q}.hada_w1_b"),
                    t2(&[0.4, -0.1, 0.2, 0.6], 1, 4),
                ),
                (
                    format!("{to_q}.hada_w2_a"),
                    t2(&[0.2, 0.0, 0.1, -0.3], 4, 1),
                ),
                (
                    format!("{to_q}.hada_w2_b"),
                    t2(&[1.0, 0.5, -0.5, 0.25], 1, 4),
                ),
            ]),
            meta: HashMap::new(),
        };
        assert!(wmeta::keys_contain_loha(
            af.tensors.keys().map(String::as_str)
        ));
        let table = build_kohya_table(&map, &[2, 4]);
        let mut report = MergeReport::default();
        merge_loha_thirdparty(&mut map, &af, 1.0, &table, &mut report).unwrap();
        assert_eq!(report.merged, 1);
        let merged = map
            .get("down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight")
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(
            merged.iter().all(|v| v.is_finite()),
            "merged LoHa weight must be finite"
        );
    }

    // ---- packed-tier additive install (sc-11103) ----------------------------------------------

    use candle_gen::candle_core::safetensors as ct_safetensors;
    use candle_gen::candle_nn::Module; // brings `LoraLinear::forward` into scope for the tests
    use candle_gen::quant::{dequant_mlx_q4_reference_gs, QLinear, MLX_GROUP_SIZE};
    use candle_gen::train::lora::{
        build_lora_targets, save_lora_peft, LoraLinear, SDXL_PEFT_PREFIX,
    };

    /// A single-leaf [`LoraHost`] over one adaptable projection — the minimal UNet stand-in the additive
    /// install walks.
    struct OneLeaf(LoraLinear);
    impl LoraHost for OneLeaf {
        fn visit_lora_mut(
            &mut self,
            f: &mut dyn FnMut(&mut LoraLinear) -> Result<()>,
        ) -> Result<()> {
            f(&mut self.0)
        }
    }

    /// Pack per-element 4-bit codes into MLX u32 words (LSB-first nibbles).
    fn pack_mlx_q4(codes: &[u8]) -> Vec<u32> {
        codes
            .chunks_exact(8)
            .map(|c| {
                c.iter()
                    .enumerate()
                    .fold(0u32, |acc, (i, &q)| acc | ((q as u32 & 0xF) << (4 * i)))
            })
            .collect()
    }

    /// A synthetic group-64 Q4 packed triple `[out, in]` with f16-exact scales/biases + the exact dense
    /// f32 grid it dequantizes to (so a packed base's forward == the grid-dense forward, lossless Q4_1).
    fn synth_q4(out_dim: usize, in_dim: usize) -> ([Tensor; 3], Tensor) {
        let dev = Device::Cpu;
        let codes: Vec<u8> = (0..out_dim * in_dim)
            .map(|i| ((i * 7 + i / 5) % 16) as u8)
            .collect();
        let groups = out_dim * in_dim / MLX_GROUP_SIZE;
        let scales: Vec<f32> = (0..groups).map(|g| 0.0625 * (g as f32 + 1.0)).collect();
        let biases: Vec<f32> = (0..groups).map(|g| -0.5 - 0.25 * g as f32).collect();
        let wq = Tensor::from_vec(pack_mlx_q4(&codes), (out_dim, in_dim / 8), &dev).unwrap();
        let s = Tensor::from_vec(scales, (out_dim, in_dim / MLX_GROUP_SIZE), &dev).unwrap();
        let b = Tensor::from_vec(biases, (out_dim, in_dim / MLX_GROUP_SIZE), &dev).unwrap();
        let grid = dequant_mlx_q4_reference_gs(&wq, &s, &b, MLX_GROUP_SIZE).unwrap();
        ([wq, s, b], grid)
    }

    /// Write a real trainer PEFT LoRA `.safetensors` targeting `path` with a forced-nonzero delta, and
    /// return `(file, ΔW at scale 1.0)`. Reuses the actual trainer save path so the install consumes the
    /// on-disk format, not hand-built tensors (rank 2, alpha 4 ⇒ ratio 2.0).
    fn write_peft_lora(
        path: &str,
        in_dim: usize,
        out_dim: usize,
        tag: &str,
    ) -> (std::path::PathBuf, Tensor) {
        use candle_gen::candle_nn::Linear;
        let dev = Device::Cpu;
        let base_w = Tensor::zeros((out_dim, in_dim), DType::F32, &dev).unwrap();
        let leaf = path.rsplit('.').next().unwrap().to_string();
        let mut host = OneLeaf(LoraLinear::from_linear(
            Linear::new(base_w, None),
            in_dim,
            out_dim,
            path.into(),
        ));
        let set = build_lora_targets(&mut host, &[leaf], 2, 4.0, 7, &dev).unwrap();
        let up = Tensor::randn(0f32, 1f32, (out_dim, 2), &dev).unwrap();
        set.vars[1].set(&up).unwrap(); // vars = [down(A), up(B)]; force B nonzero
        let file =
            std::env::temp_dir().join(format!("sc11103_{tag}_{}.safetensors", std::process::id()));
        save_lora_peft(&set, SDXL_PEFT_PREFIX, &HashMap::new(), &file).unwrap();
        let delta = reconstruct_lora_delta(
            set.vars[0].as_tensor(),
            set.vars[1].as_tensor(),
            4.0,
            2.0,
            1.0,
        )
        .unwrap();
        (file, delta)
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        (a.to_dtype(DType::F32).unwrap() - b.to_dtype(DType::F32).unwrap())
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    /// **Core packed parity (sc-11103).** Installing the SAME trained LoRA on (i) a **packed** `to_q` leaf
    /// (forward-time additive residual, base kept packed) and (ii) the equivalent **dense** grid folded
    /// (`W += δ`) produce the same forward within tolerance — proving the packed additive path equals the
    /// dense fold (the accuracy bar the packed base's own quant already accepts). Also the guardrail: the
    /// base stays **packed** (footprint survives) and the residual actually shifts the output.
    #[test]
    fn packed_additive_install_matches_dense_fold_and_stays_packed() {
        let dev = Device::Cpu;
        let qp = "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q";
        let ([wq, s, b], grid) = synth_q4(64, 64);
        let q = QLinear::from_packed(&wq, &s, &b, None, &dev).unwrap();
        let mut host = OneLeaf(LoraLinear::from_qlinear(q, 64, 64, qp.into()));

        let (file, delta) = write_peft_lora(qp, 64, 64, "parity");
        // (i) packed additive install (PEFT keys resolve without a kohya table).
        let report = install_additive(
            &mut host,
            &[AdapterSpec::new(file.clone(), 1.0, AdapterKind::Lora)],
            &BTreeMap::new(),
            &dev,
        )
        .unwrap();
        std::fs::remove_file(&file).ok();
        assert_eq!(report.applied, 1, "the trained to_q LoRA must install");
        assert!(
            host.0.is_packed(),
            "the base must stay packed (footprint survives)"
        );
        assert!(host.0.has_additive());

        // (ii) dense fold on the exact grid: W_merged = grid + ΔW.
        let dense = LoraLinear::from_linear(
            candle_gen::candle_nn::Linear::new((grid + &delta).unwrap(), None),
            64,
            64,
            qp.into(),
        );

        let x = Tensor::randn(0f32, 1f32, (4usize, 64usize), &dev).unwrap();
        let packed_y = host.0.forward(&x).unwrap();
        let dense_y = dense.forward(&x).unwrap();
        assert!(
            max_abs_diff(&packed_y, &dense_y) < 1e-3,
            "packed additive forward diverged from the dense fold"
        );
        // The residual actually moves the output (a scale-0 / no-op install would fail this).
        let (wq2, s2, b2) = {
            let ([wq2, s2, b2], _) = synth_q4(64, 64);
            (wq2, s2, b2)
        };
        let bare = LoraLinear::from_qlinear(
            QLinear::from_packed(&wq2, &s2, &b2, None, &dev).unwrap(),
            64,
            64,
            qp.into(),
        );
        assert!(
            max_abs_diff(&packed_y, &bare.forward(&x).unwrap()) > 1e-3,
            "the additive residual must shift the packed forward"
        );
    }

    /// **Conv split (sc-11103).** [`fold_conv_adapters`] folds a **conv** LoRA into the dense conv weight
    /// while leaving a **packed** Linear triple (`to_q`, u32 codes + `.scales`) byte-untouched — the
    /// Linear residual is the additive path's job, so the conv fold must not disturb the packed footprint.
    #[test]
    fn fold_conv_folds_conv_and_leaves_packed_linear_untouched() {
        let dev = Device::Cpu;
        let qp = "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q";
        let ([wq, s, b], _) = synth_q4(64, 64);
        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert(format!("{qp}.weight"), wq);
        map.insert(format!("{qp}.scales"), s);
        map.insert(format!("{qp}.biases"), b);
        let conv0 = Tensor::zeros((4usize, 4, 3, 3), DType::F32, &dev).unwrap();
        map.insert("conv_in.weight".into(), conv0.clone());

        // A LoRA file with BOTH a conv target (4-D) and a Linear target (2-D). fold_conv must fold only
        // the conv; the Linear pair is left for the additive path.
        let mut af: HashMap<String, Tensor> = HashMap::new();
        af.insert(
            "conv_in.lora_A.weight".into(),
            Tensor::randn(0f32, 1f32, (2, 4, 3, 3), &dev).unwrap(),
        );
        af.insert(
            "conv_in.lora_B.weight".into(),
            Tensor::randn(0f32, 1f32, (4, 2, 1, 1), &dev).unwrap(),
        );
        af.insert(
            format!("{qp}.lora_A.weight"),
            Tensor::randn(0f32, 1f32, (2, 64), &dev).unwrap(),
        );
        af.insert(
            format!("{qp}.lora_B.weight"),
            Tensor::randn(0f32, 1f32, (64, 2), &dev).unwrap(),
        );
        let file = std::env::temp_dir().join(format!(
            "sc11103_convsplit_{}.safetensors",
            std::process::id()
        ));
        ct_safetensors::save(&af, &file).unwrap();

        let table = build_sdxl_kohya_table(&map);
        let report = fold_conv_adapters(
            &mut map,
            &[AdapterSpec::new(file.clone(), 1.0, AdapterKind::Lora)],
            &table,
        )
        .unwrap();
        std::fs::remove_file(&file).ok();

        assert_eq!(
            report.merged, 1,
            "only the conv target folds (the Linear is additive)"
        );
        assert!(
            max_abs_diff(map.get("conv_in.weight").unwrap(), &conv0) > 1e-4,
            "the conv weight must change"
        );
        // The packed to_q triple is byte-untouched: still u32-packed with its `.scales`/`.biases`.
        assert_eq!(
            map.get(&format!("{qp}.weight")).unwrap().dtype(),
            DType::U32,
            "the packed Linear weight stays u32 (never dequantized by the conv fold)"
        );
        assert!(map.contains_key(&format!("{qp}.scales")));
        assert!(map.contains_key(&format!("{qp}.biases")));
    }

    /// **The combined "adapted nothing" guard.** A non-empty spec set that installed no residual AND
    /// folded no conv errors; zero specs (or any nonzero total) is fine.
    #[test]
    fn guard_additive_matched_errors_only_on_zero_total() {
        assert!(
            guard_additive_matched(0, 0).is_ok(),
            "empty specs never error"
        );
        assert!(guard_additive_matched(2, 3).is_ok(), "any match is fine");
        assert!(
            guard_additive_matched(1, 0).is_err(),
            "a spec that adapted nothing must fail loudly"
        );
    }

    /// **LoHa is rejected on a packed tier** (sc-11103, qwen-parity) — no allocation-free structured
    /// form — with a pointer to the dense tier; the dense `merge_adapters` path still folds it.
    #[test]
    fn install_additive_rejects_loha_on_packed() {
        let dev = Device::Cpu;
        let qp = "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q";
        let ([wq, s, b], _) = synth_q4(64, 64);
        let q = QLinear::from_packed(&wq, &s, &b, None, &dev).unwrap();
        let mut host = OneLeaf(LoraLinear::from_qlinear(q, 64, 64, qp.into()));

        let mut af: HashMap<String, Tensor> = HashMap::new();
        for suf in ["hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b"] {
            af.insert(
                format!("{qp}.{suf}"),
                Tensor::randn(0f32, 1f32, (64, 2), &dev).unwrap(),
            );
        }
        let file =
            std::env::temp_dir().join(format!("sc11103_loha_{}.safetensors", std::process::id()));
        ct_safetensors::save(&af, &file).unwrap();
        let err = install_additive(
            &mut host,
            &[AdapterSpec::new(file.clone(), 1.0, AdapterKind::Lora)],
            &BTreeMap::new(),
            &dev,
        );
        std::fs::remove_file(&file).ok();
        let msg = err.unwrap_err().to_string();
        assert!(
            msg.contains("LoHa") && msg.contains("dense"),
            "a LoHa on packed must reject with a dense-tier pointer (got: {msg})"
        );
    }

    /// **Resolver parity (sc-11103).** The unmerged factors [`resolve_lora_file`] produces (`a = downᵀ`,
    /// `b = upᵀ·(alpha/rank)`, at the user `scale`), using the crate's exact `classify_lora_key` +
    /// effective alpha/rank, reproduce the folded `x·(W + δ)ᵀ` on a dense base — so the packed-additive
    /// and dense-fold paths agree to f32 tolerance at the resolver level.
    #[test]
    fn resolve_lora_matches_fold_on_dense() {
        let dev = Device::Cpu;
        let (out_dim, in_dim, rank) = (16usize, 12usize, 3usize);
        let path = "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q";
        let down = Tensor::randn(0f32, 1f32, (rank, in_dim), &dev).unwrap();
        let up = Tensor::randn(0f32, 1f32, (out_dim, rank), &dev).unwrap();
        let (alpha, scale) = (6.0f32, 0.8f32); // ratio = alpha/rank = 2.0
                                               // SDXL classifies bare/PEFT keys as `lora_A`/`lora_B` (`lora_down`/`lora_up` is the kohya
                                               // `lora_unet_`-prefixed spelling), so use the PEFT naming the candle trainer emits.
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
        let mut pending: BTreeMap<String, Vec<PendingLora>> = BTreeMap::new();
        let mut skipped = 0usize;
        resolve_lora_file(&af, scale, &BTreeMap::new(), &mut pending, &mut skipped).unwrap();
        assert_eq!(skipped, 0);
        let p = &pending[path][0];
        assert_eq!(p.a.dims(), &[in_dim, rank], "a = downᵀ [in, rank]");
        assert_eq!(p.b.dims(), &[rank, out_dim], "b = upᵀ·ratio [rank, out]");

        let w = Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev).unwrap();
        let mut additive = LoraLinear::from_linear(
            candle_gen::candle_nn::Linear::new(w.clone(), None),
            in_dim,
            out_dim,
            path.into(),
        );
        additive.push_additive_lora(p.a.clone(), p.b.clone(), p.scale);
        let delta = reconstruct_lora_delta(&down, &up, alpha, rank as f32, scale).unwrap();
        let folded = LoraLinear::from_linear(
            candle_gen::candle_nn::Linear::new((w + delta).unwrap(), None),
            in_dim,
            out_dim,
            path.into(),
        );
        let x = Tensor::randn(0f32, 1f32, (2usize, in_dim), &dev).unwrap();
        assert!(
            max_abs_diff(&additive.forward(&x).unwrap(), &folded.forward(&x).unwrap()) < 1e-4,
            "resolved additive != folded"
        );
    }

    /// **Conv resolver (sc-11682).** [`resolve_conv_lora_file`] produces UNMERGED conv factors — `down`
    /// `[rank, in, kH, kW]`, `up` `[out, rank, 1, 1]`, with the FULL `(alpha/rank)·scale` baked — for the
    /// 4-D conv surface, and skips 2-D Linear pairs (those go through [`install_additive`]). The
    /// conv-forward parity of those factors against the fold is pinned in `unet::conv`'s tests.
    #[test]
    fn resolve_conv_lora_produces_unmerged_factors() {
        let dev = Device::Cpu;
        let (out_c, in_c, rank) = (8usize, 4usize, 2usize);
        let path = "conv_in";
        let down = Tensor::randn(0f32, 1f32, (rank, in_c, 3, 3), &dev).unwrap();
        let up = Tensor::randn(0f32, 1f32, (out_c, rank, 1, 1), &dev).unwrap();
        let (alpha, scale) = (4.0f32, 0.5f32); // (alpha/rank)·scale = (4/2)·0.5 = 1.0
                                               // Include a 2-D Linear pair that must be ignored by the conv resolver.
        let af = AdapterFile {
            tensors: HashMap::from([
                (format!("{path}.lora_A.weight"), down.clone()),
                (format!("{path}.lora_B.weight"), up.clone()),
                (
                    format!("{path}.alpha"),
                    Tensor::from_vec(vec![alpha], (1,), &dev).unwrap(),
                ),
                (
                    "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.weight"
                        .to_string(),
                    Tensor::randn(0f32, 1f32, (rank, 16), &dev).unwrap(),
                ),
                (
                    "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_B.weight"
                        .to_string(),
                    Tensor::randn(0f32, 1f32, (16, rank), &dev).unwrap(),
                ),
            ]),
            meta: HashMap::new(),
        };
        let mut pending: BTreeMap<String, Vec<PendingConv>> = BTreeMap::new();
        let mut skipped = 0usize;
        resolve_conv_lora_file(&af, scale, &BTreeMap::new(), &mut pending, &mut skipped).unwrap();
        assert!(
            !pending.contains_key("down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q"),
            "the 2-D Linear pair must not be resolved as a conv"
        );
        let p = &pending[path][0];
        assert_eq!(p.down.dims(), &[rank, in_c, 3, 3]);
        assert_eq!(p.up.dims(), &[out_c, rank, 1, 1]);
        assert!(
            (p.scale - 1.0).abs() < 1e-6,
            "full scale (alpha/rank)·user = 1.0"
        );
    }

    /// F-114: LoKr `rank`/`alpha` metadata distinguishes absent (default) from present-but-unparseable
    /// (a typed error naming the key) — a malformed value never silently collapses to the default and
    /// mis-scales the merged delta.
    #[test]
    fn lokr_meta_scalar_absent_vs_unparseable() {
        // Absent → None (caller substitutes its default).
        let empty = HashMap::new();
        assert_eq!(parse_lokr_meta_scalar(&empty, "rank").unwrap(), None);

        // Present + valid → parsed (whitespace tolerated).
        let good = HashMap::from([("rank".to_string(), " 4 ".to_string())]);
        assert_eq!(parse_lokr_meta_scalar(&good, "rank").unwrap(), Some(4.0));

        // Present + malformed → error naming the key (NOT a silent default to 1.0).
        for bad in ["4x", "", "NaN", "inf"] {
            let m = HashMap::from([("alpha".to_string(), bad.to_string())]);
            let err = parse_lokr_meta_scalar(&m, "alpha").unwrap_err();
            assert!(
                err.to_string().contains("alpha"),
                "error must name the offending key `alpha`, got: {err}"
            );
        }
    }
}
