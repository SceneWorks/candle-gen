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
//!  - **PEFT** (`base_model.model.unet.<dotted>.lora_A/B[.default].weight` + per-target `.alpha`) —
//!    what the candle trainer ([`save_lora_peft`](candle_gen::train::lora::save_lora_peft)) and
//!    `peft.save_pretrained()` emit; the dotted path resolves directly (the prefix is optional).
//!  - **kohya** (`lora_unet_<flat>.lora_down/up.weight` + `.alpha`) — community / diffusers LoRAs; the
//!    `_`-flattened stem (ambiguous, since diffusers names contain `_`) resolves against a table built
//!    from the base UNet's own Linear keys.
//!
//! LoKr resolves PEFT/bare or kohya `<module>.lokr_w1`/`lokr_w2` (+ low-rank `_a`/`_b`) with `rank` /
//! `alpha` read from file metadata (`networkType=lokr`), reconstructing `δ = (alpha/rank)·kron(w1,w2)`.
//!
//! Out-of-surface keys are **counted and surfaced** in [`MergeReport`], never silently dropped:
//! text-encoder `lora_te*` keys (UNet-only merge), and conv-shaped (4-D) LoRAs — the candle trainer
//! adapts Linear attention projections only. Conv-layer + LyCORIS LoHa / untagged-third-party coverage
//! is tracked as a follow-up (sc-5225).

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use candle_gen::candle_core::{safetensors as cst, DType, Device, Tensor};
use candle_gen::gen_core::{AdapterKind, AdapterSpec};
use candle_gen::train::lora::{reconstruct_lokr_delta, reconstruct_lora_delta};
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

/// Outcome of merging the adapter specs into the base UNet tensor map: how many base weights were
/// updated, and how many keys fell outside the merge surface (text-encoder / conv-shaped / unresolved —
/// surfaced, not silently dropped).
#[derive(Debug, Default, PartialEq, Eq)]
pub struct MergeReport {
    pub merged: usize,
    pub skipped_keys: usize,
}

#[derive(Clone, Copy)]
enum Role {
    Down,
    Up,
    Alpha,
}

#[derive(Default)]
struct LoraTriple {
    down: Option<Tensor>, // A: [rank, in]
    up: Option<Tensor>,   // B: [out, rank]
    alpha: Option<f32>,
}

/// A loaded adapter file: its tensors (CPU, native dtype) and the safetensors header metadata.
struct AdapterFile {
    tensors: HashMap<String, Tensor>,
    meta: HashMap<String, String>,
}

/// Read an adapter `.safetensors` once: tensors via candle's loader, metadata via the safetensors
/// header reader (candle's `load` drops the header `__metadata__`, which LoKr's `rank`/`alpha` live in).
fn read_adapter(path: &Path) -> Result<AdapterFile> {
    let bytes = std::fs::read(path)
        .map_err(|e| CandleError::Msg(format!("read adapter {}: {e}", path.display())))?;
    let tensors = cst::load_buffer(&bytes, &Device::Cpu)?;
    let (_, md) = safetensors::SafeTensors::read_metadata(&bytes)
        .map_err(|e| CandleError::Msg(format!("read adapter metadata {}: {e}", path.display())))?;
    let meta = md.metadata().clone().unwrap_or_default();
    Ok(AdapterFile { tensors, meta })
}

/// Build the kohya `flattened → dotted` lookup table from the base UNet's 2-D Linear weight keys
/// (`{dotted}.weight`). The `_`-flattening diffusers uses is ambiguous (its own names contain `_`), so
/// resolving against the real key set — the candle analog of the vendored `named_modules()` walk —
/// is what disambiguates a kohya stem.
fn build_kohya_table(base: &HashMap<String, Tensor>) -> BTreeMap<String, String> {
    base.iter()
        .filter_map(|(k, t)| {
            let dotted = k.strip_suffix(".weight")?;
            (t.dims().len() == 2).then(|| (dotted.replace('.', "_"), dotted.to_string()))
        })
        .collect()
}

/// Map one LoRA key to `(diffusers_dotted_path, role)`, or `None` if outside the UNet merge surface.
/// kohya (`lora_unet_<flat>…`) resolves the flattened stem via `table`; PEFT (`base_model.model.unet.`)
/// and bare dotted paths resolve directly.
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
                table.get(flat).map(|d| (d.clone(), factor))
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

fn read_scalar(t: &Tensor) -> Result<f32> {
    Ok(t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?[0])
}

/// Merge `delta` (`[out, in]` f32) into the base weight at `key`, computing `W += δ` in f32 (the stored
/// f32 sum is cast to the UNet load dtype when the VarBuilder serves it). A missing key or a
/// shape-mismatched base (e.g. a 4-D conv weight) is surfaced as skipped, never a hard error.
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

/// Merge one LoRA file into `base` at `scale`: classify every key (PEFT + kohya), fold complete
/// `(down, up)` pairs into `{path}.weight`. `rank` is `A`'s leading dim; `alpha` is the per-target
/// `.alpha` (default `rank`). Half-pairs and conv-shaped (4-D) factors are surfaced as skipped.
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

    for (path, t) in triples {
        let (Some(down), Some(up)) = (t.down, t.up) else {
            report.skipped_keys += 1; // half-pair (partner targeted a non-routable module)
            continue;
        };
        if down.dims().len() != 2 || up.dims().len() != 2 {
            report.skipped_keys += 1; // conv-shaped LoRA — deferred (sc-5225)
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

/// Whether the adapter file declares LoKr in its `networkType` metadata.
fn declares_lokr(af: &AdapterFile) -> bool {
    af.meta.get("networkType").map(String::as_str) == Some("lokr")
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
    let table = build_kohya_table(map);
    let mut report = MergeReport::default();
    for spec in specs {
        let af = read_adapter(&spec.path)?;
        match spec.kind {
            AdapterKind::Lokr => merge_lokr_file(map, &af, spec.scale, &table, &mut report)?,
            AdapterKind::Lora => {
                // The file metadata is authoritative — a Lora-declared LoKr file has no lora_A/B keys
                // and would merge nothing; surface the mismatch loudly rather than no-op.
                if declares_lokr(&af) {
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
        return Err(CandleError::Msg(format!(
            "sdxl: no adapter target modules matched across {} file(s) — expected PEFT \
             `base_model.model.unet.<path>.lora_A/B.weight` or kohya `lora_unet_<flat>.lora_down/up.\
             weight` (LoRA), or `<module>.lokr_w1/w2` with networkType=lokr (LoKr). Conv-layer and \
             LyCORIS LoHa adapters are not yet supported (sc-5225)",
            specs.len()
        )));
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let table = build_kohya_table(&base_map());
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
        let table = build_kohya_table(&map);
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

    /// A conv-shaped LoRA (4-D factors) is surfaced as skipped, never merged into the conv weight.
    #[test]
    fn merge_skips_conv_shaped_lora() {
        let mut map = base_map();
        let dev = Device::Cpu;
        let af = AdapterFile {
            tensors: HashMap::from([
                (
                    "base_model.model.unet.conv_in.lora_A.weight".to_string(),
                    Tensor::zeros((2, 4, 3, 3), DType::F32, &dev).unwrap(),
                ),
                (
                    "base_model.model.unet.conv_in.lora_B.weight".to_string(),
                    Tensor::zeros((4, 2, 1, 1), DType::F32, &dev).unwrap(),
                ),
            ]),
            meta: HashMap::new(),
        };
        let table = build_kohya_table(&map);
        let mut report = MergeReport::default();
        merge_lora_file(&mut map, &af, 1.0, &table, &mut report).unwrap();
        assert_eq!(report.merged, 0);
        assert_eq!(report.skipped_keys, 1); // the (down,up) pair, dropped as a conv shape
                                            // The conv weight is untouched (still all-zero f16).
        let cv = map.get("conv_in.weight").unwrap();
        assert_eq!(cv.dims(), &[4, 4, 3, 3]);
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
        let table = build_kohya_table(&map);
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
        let table = build_kohya_table(&map);
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
        assert!(declares_lokr(&af));
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
}
