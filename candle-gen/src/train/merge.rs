//! The shared inference-side **adapter-merge skeleton** (sc-8998 / F-018).
//!
//! Every candle provider family (SDXL, Z-Image, Wan, Lens, SD3.5, Qwen-Image, Krea, SCAIL2) closes
//! its native-trainer loop the same way: load a trained LoRA/LoKr `.safetensors`, reconstruct the
//! weight delta at f32, and **fold it into the dense base weights** (`W += δ`) *before* the stock
//! model is built — a merge, not a forward-time residual, because the chaos-sensitive samplers make
//! `(W+δ)·x` and `W·x + δ·x` diverge by ~1 ULP into a visibly different image (see
//! [`reconstruct_lora_delta`](super::lora::reconstruct_lora_delta)). Eight crates hand-copied the
//! same format-parsing / merge-report skeleton around that reconstruction; this module is its single
//! home (the delta reconstruction itself already lives in [`super::lora`]).
//!
//! ## What is shared here (byte-identical across the families)
//!
//! - [`MergeReport`] — merged / skipped-key tally, and its zero-match loud-error contract (a
//!   non-empty spec list that matches *nothing* is a format/prefix misconfiguration, surfaced loudly
//!   via [`no_target_matched`] rather than silently rendering an unadapted image).
//! - [`Role`] / [`LoraTriple`] — the `(down, up, alpha)` grouping of a LoRA target's factors.
//! - [`AdapterFile`] + [`read_adapter`] — read a `.safetensors` once (tensors via candle's loader,
//!   header `__metadata__` via the safetensors reader, which candle's `load` drops but LoKr's
//!   `rank`/`alpha` live in), plus [`AdapterFile::declares_lokr`].
//! - [`merge_into`] — fold one `[out,in]` f32 delta into `{key}` (`W += δ` in f32), a missing or
//!   shape-mismatched base surfaced as skipped.
//! - [`read_scalar`] / [`read_scalar_opt`] — a per-target `.alpha` scalar (panicking vs the
//!   size-0-tolerant F-009/sc-8989 form for third-party files).
//! - [`build_kohya_table`] — the `flattened → dotted` disambiguation table from the base key set.
//! - The whole **third-party LyCORIS** engine: [`ThirdPartyLokr`] / [`ThirdPartyLoha`] (per-module
//!   lycoris-scale reconstruction) + [`parse_lokr_thirdparty`] / [`parse_loha_thirdparty`] +
//!   [`merge_one_thirdparty`]. Untagged `lokr_*` / `hada_*` files carry no `networkType` stamp and
//!   derive rank/alpha/scale *per module*; SDXL, Qwen-Image and SCAIL2 share this verbatim.
//!
//! ## What stays per-family (the load-bearing drift the finding warns about)
//!
//! The **key → base-module resolution** is genuinely family-specific and MUST stay in each crate —
//! unifying it would change merge output. Each crate keeps a thin `classify_*` / `merge_*_file` /
//! `merge_adapters` shell built on these primitives, e.g.:
//!  - SDXL: `lora_unet_` kohya + original-SD/A1111 translation + **conv-layer** LoRA fusion.
//!  - SD3.5: kohya `lora_sd3` MMDiT-native → diffusers port with **fused-QKV row-slice** targets.
//!  - Krea: ai-toolkit native (`blocks`/`wq`/`mlp`) → diffusers rename.
//!  - Wan: bare + `diffusion_model.` prefixes, explicit text-encoder guard (no sc-5374 blob).
//!  - SCAIL2: no kohya table + lightx2v `.diff`/`.diff_b` diff-patch merge.
//! This is distinct from the deliberate per-crate `compute_loss_grads` trainer duplication (sc-7787),
//! which is intentionally NOT consolidated.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use candle_core::{safetensors as cst, DType, Device, Tensor};

use crate::gen_core::weightsmeta as wmeta;
use crate::train::lora::{reconstruct_loha_delta, reconstruct_lokr_delta};
use crate::{CandleError, Result};

/// Outcome of merging adapter specs into a base tensor map: how many base weights were updated, and
/// how many keys fell outside the merge surface (text-encoder keys, a conv/tucker factor on a
/// Linear-only surface, or an unresolved module — surfaced, not silently dropped). The `merged`
/// count gates the zero-match loud error ([`no_target_matched`]); most call sites discard the
/// populated report (F-051 / sc-9035 ratified silent library-side merges — no per-merge stderr).
#[derive(Debug, Default, PartialEq, Eq)]
pub struct MergeReport {
    pub merged: usize,
    pub skipped_keys: usize,
}

/// Which factor a LoRA key carries. `to_out.0`'s `.0` is a path segment, so classification splits on
/// the `.lora_{down,up,A,B}.weight` / `.alpha` *suffix*, never a leading token.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Down,
    Up,
    Alpha,
}

/// The grouped factors of one LoRA target: `A` `[rank, in]`, `B` `[out, rank]`, and an optional
/// per-target `.alpha` scalar. A crate collects these per resolved module path, then reconstructs the
/// delta once both legs are present (a half-pair is surfaced as skipped).
#[derive(Default)]
pub struct LoraTriple {
    pub down: Option<Tensor>,
    pub up: Option<Tensor>,
    pub alpha: Option<f32>,
}

/// A loaded adapter file: its tensors (CPU, native dtype) and the safetensors header `__metadata__`.
pub struct AdapterFile {
    pub tensors: HashMap<String, Tensor>,
    pub meta: HashMap<String, String>,
}

impl AdapterFile {
    /// Whether the file declares LoKr in its `networkType` metadata (the SceneWorks/PEFT stamp the
    /// candle trainer writes). A **third-party** LyCORIS LoKr has the `lokr_*` factors but **no**
    /// stamp — detected by keys instead (see [`parse_lokr_thirdparty`]).
    pub fn declares_lokr(&self) -> bool {
        wmeta::is_lokr_network_type(self.meta.get("networkType").map(String::as_str))
    }
}

/// Read an adapter `.safetensors` once: tensors via candle's loader (CPU, native dtype), metadata via
/// the safetensors header reader (candle's `load` drops the header `__metadata__`, which LoKr's
/// `rank`/`alpha` live in).
pub fn read_adapter(path: &Path) -> Result<AdapterFile> {
    let bytes = std::fs::read(path)
        .map_err(|e| CandleError::Msg(format!("read adapter {}: {e}", path.display())))?;
    let tensors = cst::load_buffer(&bytes, &Device::Cpu)?;
    let (_, md) = safetensors::SafeTensors::read_metadata(&bytes)
        .map_err(|e| CandleError::Msg(format!("read adapter metadata {}: {e}", path.display())))?;
    let meta = md.metadata().clone().unwrap_or_default();
    Ok(AdapterFile { tensors, meta })
}

/// Read a scalar tensor (`[]` or `[1]`) as `f32` — the per-target `.alpha` read. **Panics** on an
/// empty tensor (the trainer / kohya format always writes a 1-element `.alpha`); third-party files
/// use [`read_scalar_opt`] for the size-0-tolerant read.
pub fn read_scalar(t: &Tensor) -> Result<f32> {
    Ok(t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?[0])
}

/// Read a per-module `.alpha` scalar as `f32`, returning `None` for a size-0 (malformed) tensor
/// rather than panicking (F-009 / sc-8989) — third-party adapters store `alpha` in their compute
/// dtype and may ship a degenerate tensor. The candle twin of mlx-gen's `scalar_alpha`.
pub fn read_scalar_opt(t: &Tensor) -> Result<Option<f32>> {
    if t.elem_count() == 0 {
        return Ok(None);
    }
    Ok(t.to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?
        .first()
        .copied())
}

/// Merge `delta` (`[out, in]`, expected f32) into the base weight at `key`, computing `W += δ` in f32
/// (the stored f32 sum is cast to the model load dtype when the loader serves it). A missing key or a
/// shape-mismatched base (e.g. a 4-D conv weight under a 2-D Linear delta) is surfaced as skipped,
/// never a hard error. The `delta.to_dtype(F32)` is defensive — the [`super::lora`] reconstructors
/// already return f32, so it is an idempotent no-op that keeps the merged bytes identical whether the
/// caller pre-cast or not.
pub fn merge_into(
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
        (w.to_dtype(DType::F32)? + delta.to_dtype(DType::F32)?)?
    };
    base.insert(key.to_string(), merged);
    report.merged += 1;
    Ok(())
}

/// Build the kohya `flattened → dotted` lookup table from the base model's weight keys
/// (`{dotted}.weight`), keeping only tensors whose rank is in `dims` (`&[2]` for a Linear-only DiT
/// surface, `&[2, 4]` to also admit 4-D conv stems for the SDXL conv-LoRA surface, sc-5225). The
/// `_`-flattening diffusers uses is ambiguous — its own module names contain `_` — so resolving a
/// kohya stem against the real key set is what disambiguates it (the candle analog of a vendored
/// `named_modules()` walk).
pub fn build_kohya_table(
    base: &HashMap<String, Tensor>,
    dims: &[usize],
) -> BTreeMap<String, String> {
    base.iter()
        .filter_map(|(k, t)| {
            let dotted = k.strip_suffix(".weight")?;
            dims.contains(&t.dims().len())
                .then(|| (dotted.replace('.', "_"), dotted.to_string()))
        })
        .collect()
}

/// The shared zero-match loud error: a non-empty spec list that folded **no** target is a format /
/// prefix misconfiguration, so the caller hard-errors (the worker then falls back rather than render
/// an unadapted image silently). `family` is the crate tag, `expected` the per-family key-form hint.
pub fn no_target_matched(family: &str, expected: &str, file_count: usize) -> CandleError {
    CandleError::Msg(format!(
        "{family}: no adapter target modules matched across {file_count} file(s) — {expected}"
    ))
}

// ---- Third-party LyCORIS LoKr / LoHa (sc-5225) ---------------------------------------------------
//
// kohya / ai-toolkit / lycoris-lib LoKr (`lokr_*`) and LoHa (`hada_*`) files ship the decomposition
// factors but NOT the `networkType=lokr` stamp `AdapterFile::declares_lokr` keys off, and derive
// rank/alpha/scale **per module** (vs the PEFT path's one global pair). The `wmeta` layer does the
// string/metadata logic (key detection, suffix tables, flattened→dotted resolution); this module
// owns the per-module factor grouping + the lycoris scale rule, reconstructing with the shared f32
// math in [`super::lora`]. **Linear-only** to match mlx-gen: a factor resolving to a 4-D conv weight
// — including the lycoris conv/tucker (`lokr_t2` / `hada_t1` / `hada_t2`) forms — is surfaced as
// skipped, never mis-merged.

/// One module's third-party LoKr factors (full `w1`/`w2`, low-rank `_a`/`_b`, optional per-module
/// `.alpha`). The tucker `lokr_t2` factor is conv-only and out of the Linear surface, so it is ignored.
#[derive(Default)]
pub struct ThirdPartyLokr {
    pub w1: Option<Tensor>,
    pub w1_a: Option<Tensor>,
    pub w1_b: Option<Tensor>,
    pub w2: Option<Tensor>,
    pub w2_a: Option<Tensor>,
    pub w2_b: Option<Tensor>,
    pub alpha: Option<f32>,
}

impl ThirdPartyLokr {
    /// The factorization rank (`lora_dim`): `lokr_w1_a` is `[shape0, dim]`; else the non-tucker
    /// `lokr_w2_a` is `[shape0, dim]`. `None` when **both** factors are full — lycoris then forces
    /// `alpha = lora_dim` ⇒ scale 1, so rank is unused.
    fn rank(&self) -> Option<f32> {
        if let Some(a) = &self.w1_a {
            return Some(a.dims()[1] as f32);
        }
        self.w2_a.as_ref().map(|a| a.dims()[1] as f32)
    }

    /// LyCORIS `scale = alpha / lora_dim` (alpha defaulting to `lora_dim`), EXCEPT both-full forces
    /// scale 1 (`LokrModule.__init__`: `if use_w1 and use_w2: alpha = lora_dim`).
    fn lycoris_scale(&self) -> f32 {
        match self.rank() {
            None => 1.0,
            Some(r) => self.alpha.unwrap_or(r) / r,
        }
    }

    /// Reconstruct this module's `[out, in]` delta (lycoris per-module scale × `user_scale` baked in),
    /// via the shared [`reconstruct_lokr_delta`] (lycoris scale as `alpha` over `rank = 1.0`).
    pub fn delta(&self, base_shape: (usize, usize), user_scale: f32) -> Result<Tensor> {
        reconstruct_lokr_delta(
            self.w1.as_ref(),
            self.w1_a.as_ref(),
            self.w1_b.as_ref(),
            self.w2.as_ref(),
            self.w2_a.as_ref(),
            self.w2_b.as_ref(),
            self.lycoris_scale(),
            1.0,
            user_scale,
            base_shape,
        )
    }
}

/// One module's third-party LoHa factors — two low-rank Hadamard pairs + an optional per-module
/// `.alpha`. The tucker `hada_t1`/`hada_t2` factors are conv-only and ignored (Linear surface).
#[derive(Default)]
pub struct ThirdPartyLoha {
    pub w1_a: Option<Tensor>,
    pub w1_b: Option<Tensor>,
    pub w2_a: Option<Tensor>,
    pub w2_b: Option<Tensor>,
    pub alpha: Option<f32>,
}

impl ThirdPartyLoha {
    /// rank (`lora_dim`) = `hada_w1_b.shape[0]` (lycoris stores `hada_w1_b` as `[lora_dim, …]`).
    fn rank(&self) -> Option<f32> {
        self.w1_b.as_ref().map(|b| b.dims()[0] as f32)
    }

    /// LyCORIS `scale = alpha / lora_dim` (alpha defaulting to `lora_dim`). LoHa is always decomposed
    /// (no both-full case), so — unlike LoKr — there is no forced-1 branch.
    fn lycoris_scale(&self) -> f32 {
        match self.rank() {
            None => 1.0,
            Some(r) => self.alpha.unwrap_or(r) / r,
        }
    }

    /// Reconstruct this module's `[out, in]` Hadamard delta (lycoris scale × `user_scale` baked in).
    /// Errors if a `hada_w1/w2` `a`/`b` leg is missing (a conv-tucker-only module never reaches here —
    /// it resolves to a 4-D base and skips first).
    pub fn delta(&self, base_shape: (usize, usize), user_scale: f32) -> Result<Tensor> {
        let (w1_a, w1_b, w2_a, w2_b) = match (&self.w1_a, &self.w1_b, &self.w2_a, &self.w2_b) {
            (Some(a), Some(b), Some(c), Some(d)) => (a, b, c, d),
            _ => {
                return Err(CandleError::Msg(
                    "loha: a hada_w1/w2 a/b factor is missing".into(),
                ))
            }
        };
        reconstruct_loha_delta(
            w1_a,
            w1_b,
            w2_a,
            w2_b,
            self.lycoris_scale() * user_scale,
            base_shape,
        )
    }
}

/// Group a third-party LoKr file's tensors by raw module key (the part before `.lokr_*`/`.alpha`). The
/// raw key is whatever the trainer wrote — a `<PREFIX>_<flattened.path>` (kohya/lycoris) or a dotted
/// path — resolved to a base module path by the caller before merge.
pub fn parse_lokr_thirdparty(af: &AdapterFile) -> Result<BTreeMap<String, ThirdPartyLokr>> {
    let mut groups: BTreeMap<String, ThirdPartyLokr> = BTreeMap::new();
    for (key, t) in &af.tensors {
        if let Some(raw) = key.strip_suffix(".alpha") {
            if let Some(a) = read_scalar_opt(t)? {
                groups.entry(raw.to_string()).or_default().alpha = Some(a);
            }
            continue;
        }
        if let Some((path, factor)) = wmeta::split_factor_key(key, &wmeta::LOKR_TP_SUFFIXES) {
            let g = groups.entry(path.to_string()).or_default();
            match factor {
                "lokr_w1" => g.w1 = Some(t.clone()),
                "lokr_w1_a" => g.w1_a = Some(t.clone()),
                "lokr_w1_b" => g.w1_b = Some(t.clone()),
                "lokr_w2" => g.w2 = Some(t.clone()),
                "lokr_w2_a" => g.w2_a = Some(t.clone()),
                "lokr_w2_b" => g.w2_b = Some(t.clone()),
                "lokr_t2" => {} // tucker (conv-only) — out of the Linear surface; module skips below.
                _ => {}
            }
        }
    }
    Ok(groups)
}

/// Group a third-party LoHa file's tensors by raw module key (the part before `.hada_*`/`.alpha`).
pub fn parse_loha_thirdparty(af: &AdapterFile) -> Result<BTreeMap<String, ThirdPartyLoha>> {
    let mut groups: BTreeMap<String, ThirdPartyLoha> = BTreeMap::new();
    for (key, t) in &af.tensors {
        if let Some(raw) = key.strip_suffix(".alpha") {
            if let Some(a) = read_scalar_opt(t)? {
                groups.entry(raw.to_string()).or_default().alpha = Some(a);
            }
            continue;
        }
        if let Some((path, factor)) = wmeta::split_factor_key(key, &wmeta::LOHA_TP_SUFFIXES) {
            let g = groups.entry(path.to_string()).or_default();
            match factor {
                "hada_w1_a" => g.w1_a = Some(t.clone()),
                "hada_w1_b" => g.w1_b = Some(t.clone()),
                "hada_w2_a" => g.w2_a = Some(t.clone()),
                "hada_w2_b" => g.w2_b = Some(t.clone()),
                "hada_t1" | "hada_t2" => {} // tucker (conv-only) — module skips at the shape gate.
                _ => {}
            }
        }
    }
    Ok(groups)
}

/// Merge one reconstructed `[out, in]` delta into the base at the **already-resolved** Linear module
/// `path` (`W += δ`). Shared by the third-party LoKr + LoHa paths: Linear-only shape gate →
/// reconstruct → merge. `path == None` (an unresolved key), a missing weight, or a 4-D (conv) target
/// is surfaced as skipped, never mis-merged. Each family resolves the raw key to `path` its own way
/// (kohya table + original-SD for SDXL, bare `strip_lora_prefix` for the DiT families).
pub fn merge_one_thirdparty(
    base: &mut HashMap<String, Tensor>,
    path: Option<&str>,
    delta_at: impl FnOnce((usize, usize)) -> Result<Tensor>,
    report: &mut MergeReport,
) -> Result<()> {
    let Some(path) = path else {
        report.skipped_keys += 1;
        return Ok(());
    };
    let base_key = format!("{path}.weight");
    let Some(w) = base.get(&base_key) else {
        report.skipped_keys += 1;
        return Ok(());
    };
    if w.dims().len() != 2 {
        report.skipped_keys += 1; // Linear-only surface (the conv surface is LoRA-only)
        return Ok(());
    }
    let (out_f, in_f) = (w.dims()[0], w.dims()[1]);
    let delta = delta_at((out_f, in_f))?;
    merge_into(base, &base_key, &delta, report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t2(data: &[f32], r: usize, c: usize) -> Tensor {
        Tensor::from_vec(data.to_vec(), (r, c), &Device::Cpu).unwrap()
    }

    fn base_map() -> HashMap<String, Tensor> {
        let dev = Device::Cpu;
        let mut m = HashMap::new();
        m.insert(
            "attn.to_q.weight".into(),
            Tensor::zeros((4, 4), DType::F16, &dev).unwrap(),
        );
        m.insert(
            "conv.weight".into(),
            Tensor::zeros((4, 4, 3, 3), DType::F16, &dev).unwrap(),
        );
        m
    }

    /// `merge_into` folds a shape-matched f32 delta and counts it; a missing or shape-mismatched key
    /// is surfaced as skipped, never an error. Base is zero so the merged weight IS the delta.
    #[test]
    fn merge_into_folds_and_skips() {
        let mut map = base_map();
        let delta = t2(
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            4,
            4,
        );
        let mut report = MergeReport::default();
        merge_into(&mut map, "attn.to_q.weight", &delta, &mut report).unwrap();
        assert_eq!(
            report,
            MergeReport {
                merged: 1,
                skipped_keys: 0
            }
        );
        let got = map["attn.to_q.weight"].to_dtype(DType::F32).unwrap();
        let diff = (got - &delta)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff < 1e-4);

        // missing key → skipped.
        let mut report = MergeReport::default();
        merge_into(&mut map, "nope.weight", &delta, &mut report).unwrap();
        assert_eq!(
            report,
            MergeReport {
                merged: 0,
                skipped_keys: 1
            }
        );

        // shape mismatch (4-D conv base vs 2-D delta) → skipped, conv untouched.
        let mut report = MergeReport::default();
        merge_into(&mut map, "conv.weight", &delta, &mut report).unwrap();
        assert_eq!(
            report,
            MergeReport {
                merged: 0,
                skipped_keys: 1
            }
        );
        assert_eq!(map["conv.weight"].dims(), &[4, 4, 3, 3]);
    }

    /// A scale-0 third-party merge is byte-exact with the base (`δ·0 = 0`): the merged weight equals
    /// the original. Proves the scale composes multiplicatively through `merge_one_thirdparty`.
    #[test]
    fn scale_zero_thirdparty_is_base() {
        let dev = Device::Cpu;
        let base_q = Tensor::randn(0f32, 1f32, (4, 4), &dev).unwrap();
        let mut map = HashMap::new();
        map.insert(
            "attn.to_q.weight".to_string(),
            base_q.to_dtype(DType::F16).unwrap(),
        );
        let g = ThirdPartyLokr {
            w1: Some(t2(&[1.0, 0.0, 0.0, 1.0], 2, 2)),
            w2: Some(t2(&[0.5, 0.3, -0.2, 0.4], 2, 2)),
            ..Default::default()
        };
        let mut report = MergeReport::default();
        merge_one_thirdparty(
            &mut map,
            Some("attn.to_q"),
            |bs| g.delta(bs, 0.0),
            &mut report,
        )
        .unwrap();
        assert_eq!(report.merged, 1, "the target still 'merges' a zero delta");
        let merged = map["attn.to_q.weight"].to_dtype(DType::F32).unwrap();
        let original = base_q
            .to_dtype(DType::F16)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        assert_eq!(
            (merged - original)
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap(),
            0.0,
            "scale-0 merge must be byte-exact with the base"
        );
    }

    /// An unresolved third-party key (`path == None`) and a conv (4-D) target both skip, never merge.
    #[test]
    fn merge_one_thirdparty_gates_unresolved_and_conv() {
        let mut map = base_map();
        let g = ThirdPartyLokr {
            w1: Some(t2(&[1.0], 1, 1)),
            w2: Some(t2(&[1.0], 1, 1)),
            ..Default::default()
        };
        // Unresolved.
        let mut report = MergeReport::default();
        merge_one_thirdparty(&mut map, None, |bs| g.delta(bs, 1.0), &mut report).unwrap();
        assert_eq!(
            report,
            MergeReport {
                merged: 0,
                skipped_keys: 1
            }
        );
        // Conv target (4-D) is Linear-only ⇒ skipped.
        let mut report = MergeReport::default();
        merge_one_thirdparty(&mut map, Some("conv"), |bs| g.delta(bs, 1.0), &mut report).unwrap();
        assert_eq!(
            report,
            MergeReport {
                merged: 0,
                skipped_keys: 1
            }
        );
    }

    /// `build_kohya_table` disambiguates the ambiguous `_`-flattening against the real key set, and
    /// `dims` gates which base tensors join (2-D Linear only vs +4-D conv, sc-5225).
    #[test]
    fn build_kohya_table_dims_gate() {
        let map = base_map();
        let linear_only = build_kohya_table(&map, &[2]);
        assert_eq!(
            linear_only.get("attn_to_q").map(String::as_str),
            Some("attn.to_q")
        );
        assert!(
            linear_only.get("conv").is_none(),
            "conv is 4-D, excluded from a 2-D table"
        );
        let with_conv = build_kohya_table(&map, &[2, 4]);
        assert_eq!(with_conv.get("conv").map(String::as_str), Some("conv"));
    }

    /// `read_scalar_opt` tolerates a size-0 tensor (F-009 / sc-8989) where `read_scalar` would panic.
    #[test]
    fn read_scalar_opt_tolerates_empty() {
        let one = t2(&[7.0], 1, 1);
        assert_eq!(read_scalar_opt(&one).unwrap(), Some(7.0));
        assert_eq!(read_scalar(&one).unwrap(), 7.0);
        let empty = Tensor::from_vec(Vec::<f32>::new(), (0,), &Device::Cpu).unwrap();
        assert_eq!(read_scalar_opt(&empty).unwrap(), None);
    }

    /// `declares_lokr` reads the `networkType=lokr` stamp via the wmeta helper; an untagged file is not
    /// declared (routed by keys instead).
    #[test]
    fn declares_lokr_reads_stamp() {
        let stamped = AdapterFile {
            tensors: HashMap::new(),
            meta: HashMap::from([("networkType".to_string(), "lokr".to_string())]),
        };
        assert!(stamped.declares_lokr());
        let untagged = AdapterFile {
            tensors: HashMap::new(),
            meta: HashMap::new(),
        };
        assert!(!untagged.declares_lokr());
    }

    /// The lycoris both-full LoKr forces scale 1 (rank unused); a decomposed leg yields `alpha/dim`.
    #[test]
    fn thirdparty_lokr_lycoris_scale_rule() {
        let both_full = ThirdPartyLokr {
            w1: Some(t2(&[1.0], 1, 1)),
            w2: Some(t2(&[1.0], 1, 1)),
            ..Default::default()
        };
        assert_eq!(both_full.lycoris_scale(), 1.0);
        // w1_a [out, dim=4], alpha 8 ⇒ scale 2.0.
        let decomposed = ThirdPartyLokr {
            w1_a: Some(Tensor::zeros((4, 4), DType::F32, &Device::Cpu).unwrap()),
            w1_b: Some(Tensor::zeros((4, 4), DType::F32, &Device::Cpu).unwrap()),
            w2: Some(t2(&[1.0], 1, 1)),
            alpha: Some(8.0),
            ..Default::default()
        };
        assert_eq!(decomposed.lycoris_scale(), 2.0);
    }

    /// The shared zero-match error carries the family tag, hint, and file count.
    #[test]
    fn no_target_matched_formats() {
        let e = no_target_matched("wan", "expected PEFT keys", 3).to_string();
        assert!(e.contains("wan:"));
        assert!(e.contains("across 3 file(s)"));
        assert!(e.contains("expected PEFT keys"));
    }
}
