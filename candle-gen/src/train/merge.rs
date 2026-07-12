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
//! - [`read_scalar`] / [`read_scalar_opt`] — a per-target scalar (a LoRA `.alpha`, or any other
//!   1-element meta tensor), hardened against malformed third-party files (F-009 / sc-8989): a bad
//!   tensor is a typed `Err` naming the key and a caller-supplied `field` label (F-119 / sc-11208, so
//!   a non-`.alpha` scalar like `inject_offset` is not mislabelled), never a panic; `read_scalar_opt`
//!   additionally tolerates a size-0 tensor as `None`.
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

/// The hard ceiling on an adapter `.safetensors` size, enforced before [`read_adapter`] buffers the
/// whole file into memory (F-078 / sc-9058). LoRA / LoKr / LoHa adapters are the tiny delta factors of
/// a model — real ones run MBs to low single-digit GBs even for a full-rank fold of a 32B DiT; nothing
/// legitimate approaches this. The cap turns a corrupt or malicious multi-tens-of-GB "adapter" (which
/// candle's loader would otherwise `std::fs::read` in full, driving the process OOM) into a clean,
/// descriptive `Err` naming the path, the cap, and the actual size — rejected on a cheap `fs::metadata`
/// stat, before a single byte is allocated. 8 GiB leaves generous headroom above any real adapter while
/// staying well under the memory a buffered read would demand for a hostile file. Base weights are
/// mmap'd, not buffered, so they are (correctly) unaffected by this.
pub const MAX_ADAPTER_BYTES: u64 = 8 * 1024 * 1024 * 1024;

/// Read an adapter `.safetensors` once: tensors via candle's loader (CPU, native dtype), metadata via
/// the safetensors header reader (candle's `load` drops the header `__metadata__`, which LoKr's
/// `rank`/`alpha` live in).
///
/// The file is buffered whole (unlike the mmap'd base-weight paths), so its on-disk size is checked
/// against [`MAX_ADAPTER_BYTES`] *before* the allocation (F-078 / sc-9058): an over-cap file is a clean
/// `Err` naming the path, cap, and actual size, never an unbounded `std::fs::read` that could OOM the
/// process. Normal-sized adapters are unaffected — the byte-for-byte load path below is unchanged.
pub fn read_adapter(path: &Path) -> Result<AdapterFile> {
    read_adapter_capped(path, MAX_ADAPTER_BYTES)
}

/// [`read_adapter`] with an explicit byte cap — the shared body; `read_adapter` passes
/// [`MAX_ADAPTER_BYTES`]. Factored out so the guard can be exercised with a tiny cap in tests without
/// crafting a multi-GB file (the fully-in-memory buffer read is exactly why the cap exists).
fn read_adapter_capped(path: &Path, max_bytes: u64) -> Result<AdapterFile> {
    let size = std::fs::metadata(path)
        .map_err(|e| CandleError::Msg(format!("stat adapter {}: {e}", path.display())))?
        .len();
    if size > max_bytes {
        return Err(CandleError::Msg(format!(
            "adapter {} is {size} bytes, exceeding the {max_bytes}-byte cap; \
             refusing to buffer it into memory",
            path.display(),
        )));
    }
    let bytes = std::fs::read(path)
        .map_err(|e| CandleError::Msg(format!("read adapter {}: {e}", path.display())))?;
    let tensors = cst::load_buffer(&bytes, &Device::Cpu)?;
    let (_, md) = safetensors::SafeTensors::read_metadata(&bytes)
        .map_err(|e| CandleError::Msg(format!("read adapter metadata {}: {e}", path.display())))?;
    let meta = md.metadata().clone().unwrap_or_default();
    Ok(AdapterFile { tensors, meta })
}

/// Read a scalar tensor (`[]` or `[1]`) as `f32` — a per-target scalar read (a LoRA `.alpha`, or any
/// other 1-element meta tensor). A well-formed trainer / kohya scalar is always a finite 1-element
/// tensor; a **malformed / truncated third-party** file can ship a degenerate one (F-009 / sc-8989,
/// extended to non-adapter scalars in F-119 / sc-11208). Rather than panicking on the old
/// `to_vec1()[0]` index (a library-runtime crash triggerable by an untrusted file), this returns a
/// descriptive typed error naming `key` and the caller-supplied `field` label (so an `inject_offset`
/// meta tensor is not mislabelled `.alpha`) plus what was wrong (empty, multi-element, or non-finite).
/// Any dtype is accepted — the value is read after casting to f32. Callers with a size-0-tolerant
/// path use [`read_scalar_opt`] instead.
pub fn read_scalar(key: &str, field: &str, t: &Tensor) -> Result<f32> {
    read_scalar_opt(key, field, t)?.ok_or_else(|| {
        CandleError::Msg(format!(
            "scalar tensor `{key}` (`{field}`) is empty (0 elements); expected a finite scalar"
        ))
    })
}

/// Read a per-module scalar as `f32`, returning `None` for a size-0 (malformed) tensor rather than
/// panicking (F-009 / sc-8989) — third-party adapters store `alpha` in their compute dtype and may
/// ship a degenerate tensor. A **non-empty but malformed** tensor (more than one element, or a
/// non-finite value) is a descriptive typed error naming `key` and the caller-supplied `field` label,
/// never a panic or a silently-wrong scale. The candle twin of mlx-gen's `scalar_alpha`.
pub fn read_scalar_opt(key: &str, field: &str, t: &Tensor) -> Result<Option<f32>> {
    if t.elem_count() == 0 {
        return Ok(None);
    }
    let vals = t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    if vals.len() != 1 {
        return Err(CandleError::Msg(format!(
            "scalar tensor `{key}` (`{field}`) has {} elements, expected a scalar",
            vals.len()
        )));
    }
    let v = vals[0];
    if !v.is_finite() {
        return Err(CandleError::Msg(format!(
            "scalar tensor `{key}` (`{field}`) is non-finite ({v})"
        )));
    }
    Ok(Some(v))
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
            if let Some(a) = read_scalar_opt(key, "alpha", t)? {
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
            if let Some(a) = read_scalar_opt(key, "alpha", t)? {
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
            !linear_only.contains_key("conv"),
            "conv is 4-D, excluded from a 2-D table"
        );
        let with_conv = build_kohya_table(&map, &[2, 4]);
        assert_eq!(with_conv.get("conv").map(String::as_str), Some("conv"));
    }

    /// A well-formed 1-element `.alpha` reads the exact stored value, byte-for-byte, through both the
    /// panicking-free `read_scalar` and the size-0-tolerant `read_scalar_opt` (regression guard: the
    /// F-009 / sc-8989 hardening must NOT change values for good adapters). A scalar-shaped (`[]`)
    /// tensor reads identically to a `[1]` one.
    #[test]
    fn read_scalar_well_formed_exact() {
        let one = t2(&[7.5], 1, 1);
        assert_eq!(read_scalar("m.alpha", "alpha", &one).unwrap(), 7.5);
        assert_eq!(
            read_scalar_opt("m.alpha", "alpha", &one).unwrap(),
            Some(7.5)
        );
        // A rank-0 scalar tensor (`[]`) — the other well-formed on-disk shape.
        let scalar = Tensor::new(3.25f32, &Device::Cpu).unwrap();
        assert_eq!(read_scalar("m.alpha", "alpha", &scalar).unwrap(), 3.25);
        // Non-f32 dtype (e.g. a compute-dtype alpha) casts cleanly, no panic.
        let as_i64 = Tensor::new(4i64, &Device::Cpu).unwrap();
        assert_eq!(read_scalar("m.alpha", "alpha", &as_i64).unwrap(), 4.0);
    }

    /// A malformed `.alpha` from an untrusted third-party file yields a clean, descriptive `Err`
    /// naming the offending key — never a panic (F-009 / sc-8989). Covers size-0, multi-element, and
    /// non-finite. `read_scalar_opt` still maps size-0 → `None` (its size-0-tolerant contract) but
    /// rejects the other two.
    #[test]
    fn read_scalar_malformed_errs_not_panics() {
        // Size-0: `read_scalar` errors (naming the key); `read_scalar_opt` tolerates it as None.
        let empty = Tensor::from_vec(Vec::<f32>::new(), (0,), &Device::Cpu).unwrap();
        let err = read_scalar("blk.0.alpha", "alpha", &empty)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("blk.0.alpha") && err.contains("empty"),
            "got: {err}"
        );
        assert_eq!(
            read_scalar_opt("blk.0.alpha", "alpha", &empty).unwrap(),
            None
        );

        // Multi-element: both variants error (a scalar was expected).
        let two = t2(&[1.0, 2.0], 1, 2);
        for got in [
            read_scalar("blk.0.alpha", "alpha", &two)
                .unwrap_err()
                .to_string(),
            read_scalar_opt("blk.0.alpha", "alpha", &two)
                .unwrap_err()
                .to_string(),
        ] {
            assert!(
                got.contains("blk.0.alpha") && got.contains("2 elements"),
                "got: {got}"
            );
        }

        // Non-finite (NaN / Inf): both variants error rather than merge a poisoned scale.
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let t = t2(&[bad], 1, 1);
            let got = read_scalar("blk.0.alpha", "alpha", &t)
                .unwrap_err()
                .to_string();
            assert!(
                got.contains("blk.0.alpha") && got.contains("non-finite"),
                "got: {got}"
            );
            assert!(read_scalar_opt("blk.0.alpha", "alpha", &t).is_err());
        }
    }

    /// F-119 (sc-11208): the error message names the caller-supplied `field`, so a non-`.alpha` scalar
    /// (e.g. a control-branch `meta.inject_offset` tensor) is NOT mislabelled `.alpha`. Regression guard
    /// for the hardcoded label the reviewer flagged.
    #[test]
    fn read_scalar_names_the_field_label() {
        let empty = Tensor::from_vec(Vec::<f32>::new(), (0,), &Device::Cpu).unwrap();
        let err = read_scalar("meta.inject_offset", "inject_offset", &empty)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("inject_offset") && !err.contains("alpha"),
            "message must name the field, not `.alpha`: {err}"
        );
        // The alpha callers still surface `alpha` in the label.
        let two = t2(&[1.0, 2.0], 1, 2);
        let a = read_scalar("m.beta", "alpha", &two)
            .unwrap_err()
            .to_string();
        assert!(a.contains("alpha"), "got: {a}");
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

    /// A unique per-process temp path so parallel test binaries don't collide.
    fn tmp_adapter(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "candle_gen_merge_{tag}_{}_{:?}.safetensors",
            std::process::id(),
            std::thread::current().id(),
        ))
    }

    /// Write a tiny well-formed `.safetensors` (one small tensor) to `path`.
    fn write_tiny_adapter(path: &Path) {
        let mut m: HashMap<String, Tensor> = HashMap::new();
        m.insert("m.lora_down.weight".to_string(), t2(&[1.0, 2.0], 1, 2));
        cst::save(&m, path).unwrap();
    }

    /// A normal-sized adapter loads its tensors under the size cap (F-078 / sc-9058): the guard is a
    /// no-op for legitimate files — same bytes, same tensor map.
    #[test]
    fn read_adapter_loads_normal_file_under_cap() {
        let path = tmp_adapter("normal");
        write_tiny_adapter(&path);
        // Real entry (8 GiB cap) and an explicit small-but-sufficient cap both load identically.
        let af = read_adapter(&path).unwrap();
        assert!(af.tensors.contains_key("m.lora_down.weight"));
        let capped = read_adapter_capped(&path, MAX_ADAPTER_BYTES).unwrap();
        assert_eq!(capped.tensors.len(), af.tensors.len());
        std::fs::remove_file(&path).ok();
    }

    /// An over-cap adapter is a clean, descriptive `Err` — not an OOM, not a panic — naming the path,
    /// the cap, and the actual size, BEFORE the file is buffered into memory (F-078 / sc-9058). Uses a
    /// tiny (1-byte) cap against a small real file so no GBs are allocated.
    #[test]
    fn read_adapter_over_cap_errs_cleanly() {
        let path = tmp_adapter("overcap");
        write_tiny_adapter(&path);
        let size = std::fs::metadata(&path).unwrap().len();
        assert!(size > 1, "fixture must exceed the 1-byte test cap");
        let err = match read_adapter_capped(&path, 1) {
            Ok(_) => panic!("over-cap adapter must be rejected, not loaded"),
            Err(e) => e.to_string(),
        };
        std::fs::remove_file(&path).ok();
        assert!(err.contains("adapter"), "message names the subject: {err}");
        assert!(
            err.contains(&size.to_string()),
            "message names the actual size ({size}): {err}"
        );
        assert!(err.contains("cap"), "message names the cap: {err}");
        assert!(
            err.contains(&path.display().to_string())
                || err.contains(path.file_name().unwrap().to_str().unwrap()),
            "message names the path: {err}"
        );
    }

    /// A missing adapter path is a clean stat `Err`, not a panic (the cap check stats first).
    #[test]
    fn read_adapter_missing_path_errs() {
        let path = tmp_adapter("does_not_exist_xyz");
        std::fs::remove_file(&path).ok();
        let err = match read_adapter(&path) {
            Ok(_) => panic!("missing adapter must error"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("stat adapter"), "got: {err}");
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
