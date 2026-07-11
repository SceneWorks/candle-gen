//! Candle (Windows/CUDA) FLUX.2-klein **single-file → diffusers** converter (sc-7459, epic 6564
//! story 3) — the candle twin of `mlx_gen_flux2::convert`'s `convert_and_assemble` (sc-3136).
//!
//! Community FLUX.2-klein fine-tunes such as `wikeeyang/Flux2-Klein-9B-True-V2` (the `flux2_klein_9b_true_v2`
//! variant) ship the transformer ONLY, as a single flat `.safetensors` in the original (ComfyUI/BFL)
//! key convention — no diffusers subfolders, no text-encoder / VAE. The candle `flux2_klein_9b`
//! loader ([`crate::transformer::Flux2Transformer::new`]) consumes the **diffusers** key tree, so the
//! on-disk tensors must already be in diffusers convention before a true_v2 snapshot can load. This
//! module reproduces, in candle, the exact key remap the MLX converter (and diffusers'
//! `convert_flux2_transformer_checkpoint_to_diffusers`) applies:
//!
//!   * key renames (`img_in` → `x_embedder`, `*.lin` → `*.linear`, …),
//!   * double-block fused `qkv` `[3·d, d]` row-split into `to_q`/`to_k`/`to_v` (img stream) and
//!     `add_q_proj`/`add_k_proj`/`add_v_proj` (txt stream),
//!   * single-block `linear1`/`linear2` → `to_qkv_mlp_proj`/`to_out` (1:1 — diffusers keeps the
//!     single block fused),
//!   * `final_layer.adaLN_modulation.1` → `norm_out.linear` WITH a **scale/shift swap**: BFL packs
//!     `(shift, scale)`; diffusers/this crate expect `(scale, shift)`. This one swap is load-bearing —
//!     that tensor modulates every output patch, so getting it wrong corrupts the whole image with a
//!     periodic weave (mlx sc-2220).
//!
//! then assembles a complete local diffusers model dir by borrowing the untouched VAE / text encoder /
//! tokenizer / scheduler from an already-installed base FLUX.2-klein-9B snapshot.
//!
//! **Pure structural transform.** The renames + `qkv` row-split + the adaLN half-swap are all
//! contiguous-slice memory ops (no arithmetic), so the tensors' dtype (bf16) is preserved bit-exactly
//! — candle's CPU `narrow`/`cat`/`contiguous` just reshape the buffer. No GPU, no quantization (klein
//! loads dense; the dev pre-quant path is a separate concern).
//!
//! **Borrowing on Windows.** Unlike the MLX converter (macOS, absolute symlinks), the candle box is
//! Windows: directory symlinks need privilege AND later fail to read with `ERROR_UNTRUSTED_MOUNT_POINT`
//! (the same defect the worker's `downloads.rs` documents for HF-cache symlinks). So the borrowed
//! components are **hardlinked** file-by-file (recreating the dir tree) — no privilege, no reparse-point
//! read defect, and no multi-GB duplication on the same volume — with a copy fallback for cross-volume.
//! On unix the borrow is an absolute symlink, matching MLX.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_gen::candle_core::safetensors::MmapedSafetensors;
// Use candle_core's `Error`/`Result` (not candle-gen's `CandleError`) throughout: candle_core::Error
// carries `From<std::io::Error>` (the `Io` variant) + a `Msg` constructor, so the converter's many
// filesystem `?` calls just work and a `candle_core::Error` flows up to the worker (which maps it to a
// String). candle-gen's `CandleError` has no `From<io::Error>`, so it would force a `.map_err` per fs call.
use candle_gen::candle_core::{DType, Device, Error, Result, Tensor};

/// Borrowed-from-base subdirs the candle [`crate`] loader consumes (`text_encoder` / `vae` /
/// `tokenizer`). A transformer-only fine-tune does not touch these, so taking them from the installed
/// base klein-9B is correct.
const BORROWED_SUBDIRS_REQUIRED: &[&str] = &["vae", "text_encoder", "tokenizer"];
/// Borrowed-from-base subdirs that complete the diffusers snapshot but are NOT read by the candle
/// loader (the scheduler config is applied in-code); borrowed when present, skipped when absent so a
/// base snapshot laid out slightly differently still converts.
const BORROWED_SUBDIRS_OPTIONAL: &[&str] = &["scheduler"];
/// Borrowed-from-base top-level files (copied as real files — small, and must survive the worker's
/// temp→final atomic rename).
const BORROWED_FILES: &[&str] = &["model_index.json"];

/// Top-level (non-block) direct renames: original → diffusers.
const TOP_RENAMES: &[(&str, &str)] = &[
    ("img_in.weight", "x_embedder.weight"),
    ("txt_in.weight", "context_embedder.weight"),
    (
        "time_in.in_layer.weight",
        "time_guidance_embed.timestep_embedder.linear_1.weight",
    ),
    (
        "time_in.out_layer.weight",
        "time_guidance_embed.timestep_embedder.linear_2.weight",
    ),
    (
        "double_stream_modulation_img.lin.weight",
        "double_stream_modulation_img.linear.weight",
    ),
    (
        "double_stream_modulation_txt.lin.weight",
        "double_stream_modulation_txt.linear.weight",
    ),
    (
        "single_stream_modulation.lin.weight",
        "single_stream_modulation.linear.weight",
    ),
    ("final_layer.linear.weight", "proj_out.weight"),
];

/// Handled separately (scale/shift swap): `final_layer.adaLN_modulation.1` → `norm_out.linear`.
const ADALN_SOURCE: &str = "final_layer.adaLN_modulation.1.weight";
const ADALN_TARGET: &str = "norm_out.linear.weight";

/// Per-double-block renames (original suffix → diffusers suffix), excluding the fused qkv tensors
/// which are row-split below.
const DOUBLE_RENAMES: &[(&str, &str)] = &[
    ("img_attn.norm.query_norm.weight", "attn.norm_q.weight"),
    ("img_attn.norm.key_norm.weight", "attn.norm_k.weight"),
    ("img_attn.proj.weight", "attn.to_out.0.weight"),
    ("img_mlp.0.weight", "ff.linear_in.weight"),
    ("img_mlp.2.weight", "ff.linear_out.weight"),
    (
        "txt_attn.norm.query_norm.weight",
        "attn.norm_added_q.weight",
    ),
    ("txt_attn.norm.key_norm.weight", "attn.norm_added_k.weight"),
    ("txt_attn.proj.weight", "attn.to_add_out.weight"),
    ("txt_mlp.0.weight", "ff_context.linear_in.weight"),
    ("txt_mlp.2.weight", "ff_context.linear_out.weight"),
];

/// Fused qkv suffix → `(q, k, v)` target suffixes, per stream.
const DOUBLE_QKV: &[(&str, [&str; 3])] = &[
    (
        "img_attn.qkv.weight",
        ["attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight"],
    ),
    (
        "txt_attn.qkv.weight",
        [
            "attn.add_q_proj.weight",
            "attn.add_k_proj.weight",
            "attn.add_v_proj.weight",
        ],
    ),
];

/// Per-single-block renames (1:1; diffusers keeps the fused single block).
const SINGLE_RENAMES: &[(&str, &str)] = &[
    ("linear1.weight", "attn.to_qkv_mlp_proj.weight"),
    ("linear2.weight", "attn.to_out.weight"),
    ("norm.query_norm.weight", "attn.norm_q.weight"),
    ("norm.key_norm.weight", "attn.norm_k.weight"),
];

/// The transformer weights filename written into `out/transformer/` (the diffusers convention; the
/// loader reads every `.safetensors` in the dir, so the exact name is cosmetic).
const TRANSFORMER_WEIGHTS: &str = "diffusion_pytorch_model.safetensors";

/// Count the blocks under `prefix` (`max(i)+1` over keys matching `^{prefix}.{i}.…`), the fork's
/// `_count_blocks` — derives the layer count from the checkpoint itself rather than the config.
fn count_blocks<'a>(keys: impl Iterator<Item = &'a str>, prefix: &str) -> usize {
    let pat = format!("{prefix}.");
    let mut max_idx: Option<usize> = None;
    for k in keys {
        let Some(rest) = k.strip_prefix(&pat) else {
            continue;
        };
        let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        // Require a trailing '.' after the index so `double_blocks.10.` parses as 10, not a prefix
        // collision; a key like `double_blocksX` (no match) is already filtered by `strip_prefix`.
        if digits.is_empty() || !rest[digits.len()..].starts_with('.') {
            continue;
        }
        if let Ok(i) = digits.parse::<usize>() {
            max_idx = Some(max_idx.map_or(i, |m| m.max(i)));
        }
    }
    max_idx.map_or(0, |m| m + 1)
}

/// Row-split a `[3·d, …]` tensor into three equal `[d, …]` chunks along axis 0 (the candle twin of
/// `mx.split(t, 3, 0)`). Pure memory copies — no arithmetic, dtype-preserving.
///
/// Each chunk is materialized into **owned zero-offset storage** (`force_contiguous` — NOT
/// `.contiguous()`, which is a no-op on a dim-0 narrow's contiguous strides, and NOT `.copy()`,
/// which clones the whole fused buffer layout-included): `QTensor::quantize{,_onto}` reads the raw
/// backing storage and ignores a view's start offset, so feeding narrow views into the
/// in-memory-map → quantize path (sc-10680) loaded every to_k/to_v as a copy of to_q (sc-11028).
/// Owned chunks also let the fused parent storage free instead of being pinned three times by the
/// map.
fn chunk3(t: &Tensor) -> Result<[Tensor; 3]> {
    let rows = t.dim(0)?;
    if !rows.is_multiple_of(3) {
        return Err(Error::Msg(format!(
            "fused qkv split expects a row count divisible by 3, got shape {:?}",
            t.dims()
        )));
    }
    let each = rows / 3;
    let q = t.narrow(0, 0, each)?.force_contiguous()?;
    let k = t.narrow(0, each, each)?.force_contiguous()?;
    let v = t.narrow(0, 2 * each, each)?.force_contiguous()?;
    Ok([q, k, v])
}

/// Split a `[2·d, …]` tensor at the midpoint and swap the halves: BFL `(shift, scale)` → diffusers
/// `(scale, shift)`. Load-bearing (mlx sc-2220). Contiguous slices + cat — dtype-preserving.
fn swap_halves(t: &Tensor) -> Result<Tensor> {
    let rows = t.dim(0)?;
    if !rows.is_multiple_of(2) {
        return Err(Error::Msg(format!(
            "adaLN half-swap expects an even row count, got shape {:?}",
            t.dims()
        )));
    }
    let half = rows / 2;
    let first = t.narrow(0, 0, half)?;
    let second = t.narrow(0, half, half)?;
    Tensor::cat(&[&second, &first], 0)?.contiguous()
}

/// Map an original-format FLUX.2-klein transformer tensor set onto the diffusers key set (the candle
/// twin of the fork's `build_target_state_dict`). Pure remapping — renames + qkv row-split + the adaLN
/// half-swap. The produced keys are exactly the base diffusers transformer's keys. Source tensors are
/// loaded lazily from the mmap (and dropped after each op) so only the produced map is held resident.
fn build_target_state_dict(src: &MmapedSafetensors) -> Result<HashMap<String, Tensor>> {
    let cpu = Device::Cpu;
    let names: Vec<String> = src.tensors().into_iter().map(|(name, _)| name).collect();
    let load = |name: &str| -> Result<Tensor> {
        src.load(name, &cpu)
            .map_err(|e| Error::Msg(format!("flux2 convert: source is missing `{name}`: {e}")))
    };

    let mut out: HashMap<String, Tensor> = HashMap::new();
    for (s, d) in TOP_RENAMES {
        out.insert((*d).to_string(), load(s)?);
    }
    out.insert(ADALN_TARGET.to_string(), swap_halves(&load(ADALN_SOURCE)?)?);

    let n_double = count_blocks(names.iter().map(String::as_str), "double_blocks");
    for i in 0..n_double {
        let (s, d) = (
            format!("double_blocks.{i}"),
            format!("transformer_blocks.{i}"),
        );
        for (src_suffix, [q, k, v]) in DOUBLE_QKV {
            let [tq, tk, tv] = chunk3(&load(&format!("{s}.{src_suffix}"))?)?;
            out.insert(format!("{d}.{q}"), tq);
            out.insert(format!("{d}.{k}"), tk);
            out.insert(format!("{d}.{v}"), tv);
        }
        for (src_suffix, dst_suffix) in DOUBLE_RENAMES {
            out.insert(
                format!("{d}.{dst_suffix}"),
                load(&format!("{s}.{src_suffix}"))?,
            );
        }
    }

    let n_single = count_blocks(names.iter().map(String::as_str), "single_blocks");
    for i in 0..n_single {
        let (s, d) = (
            format!("single_blocks.{i}"),
            format!("single_transformer_blocks.{i}"),
        );
        for (src_suffix, dst_suffix) in SINGLE_RENAMES {
            out.insert(
                format!("{d}.{dst_suffix}"),
                load(&format!("{s}.{src_suffix}"))?,
            );
        }
    }

    Ok(out)
}

// --- In-place ComfyUI FLUX.2-dev fp8-mixed DiT (epic 10451 Phase 2e, sc-10680) ---
//
// The measured target is `diffusion_models/flux2_dev_fp8mixed.safetensors` (555 tensors: BF16×171,
// F32×256, **F8_E4M3×128**). Same BFL-native keys + structural remap as klein (renames + qkv row-split
// + the adaLN half-swap), but two convention differences the klein tables above do NOT cover:
//
//   1. **Per-head RMSNorm weights are named `.scale`**, not `.weight` — the BFL-official spelling (the
//      klein *community* fine-tune the [`DOUBLE_RENAMES`]/[`SINGLE_RENAMES`] targets were built from
//      re-keyed those to `.weight`). So the norm SOURCE suffixes here are `.scale`.
//   2. **dev carries a guidance embedder** (`guidance_in.{in,out}_layer`) — klein is CFG-free-distilled
//      and has none; the klein [`TOP_RENAMES`] omits it.
//
// and one quant difference: the MLP Linears are **inline-scale fp8** — an `F8_E4M3` `.weight` with a
// sibling `.weight_scale` F32 scalar (`w = w_fp8·weight_scale`); the `.input_scale` sibling is the
// activation quant scale, irrelevant to a dequantized matmul, so it is dropped.

/// dev-only guidance embedder: BFL `guidance_in.{in,out}_layer` → diffusers
/// `time_guidance_embed.guidance_embedder.linear_{1,2}` (the structural twin of the `time_in` →
/// `timestep_embedder` entries in [`TOP_RENAMES`]). `Flux2Transformer::new` gates the guidance embedder
/// on the presence of `guidance_embedder.linear_1.weight`, so these keys switch the DiT into dev mode.
const COMFY_GUIDANCE_RENAMES: &[(&str, &str)] = &[
    (
        "guidance_in.in_layer.weight",
        "time_guidance_embed.guidance_embedder.linear_1.weight",
    ),
    (
        "guidance_in.out_layer.weight",
        "time_guidance_embed.guidance_embedder.linear_2.weight",
    ),
];

/// Per-double-block renames for the BFL-official checkpoint — identical to [`DOUBLE_RENAMES`] except the
/// per-head RMSNorm SOURCE suffixes are `.scale` (BFL-official) rather than `.weight` (klein community).
const COMFY_DOUBLE_RENAMES: &[(&str, &str)] = &[
    ("img_attn.norm.query_norm.scale", "attn.norm_q.weight"),
    ("img_attn.norm.key_norm.scale", "attn.norm_k.weight"),
    ("img_attn.proj.weight", "attn.to_out.0.weight"),
    ("img_mlp.0.weight", "ff.linear_in.weight"),
    ("img_mlp.2.weight", "ff.linear_out.weight"),
    ("txt_attn.norm.query_norm.scale", "attn.norm_added_q.weight"),
    ("txt_attn.norm.key_norm.scale", "attn.norm_added_k.weight"),
    ("txt_attn.proj.weight", "attn.to_add_out.weight"),
    ("txt_mlp.0.weight", "ff_context.linear_in.weight"),
    ("txt_mlp.2.weight", "ff_context.linear_out.weight"),
];

/// Per-single-block renames for the BFL-official checkpoint — as [`SINGLE_RENAMES`] but with `.scale`
/// RMSNorm source suffixes.
const COMFY_SINGLE_RENAMES: &[(&str, &str)] = &[
    ("linear1.weight", "attn.to_qkv_mlp_proj.weight"),
    ("linear2.weight", "attn.to_out.weight"),
    ("norm.query_norm.scale", "attn.norm_q.weight"),
    ("norm.key_norm.scale", "attn.norm_k.weight"),
];

/// Build the diffusers-key transformer tensor map for an **in-place ComfyUI FLUX.2-dev fp8-mixed** DiT
/// (epic 10451 Phase 2e, sc-10680) — the dequant-aware sibling of [`build_target_state_dict`], for
/// `VarBuilder::from_tensors`. Same structural remap (renames + qkv row-split + adaLN half-swap) with
/// the dev guidance embedder ([`COMFY_GUIDANCE_RENAMES`]) and the BFL-official `.scale` norm spellings
/// ([`COMFY_DOUBLE_RENAMES`]/[`COMFY_SINGLE_RENAMES`]), plus **inline-scale fp8 dequant** applied inside
/// the load closure: an `F8_E4M3` weight is `w = w_fp8·weight_scale → dtype`; every other tensor is a
/// plain cast to `dtype`. `weight_scale` is a per-tensor F32 scalar; `input_scale` (the activation
/// scale) is never referenced. `dtype` is the pipeline compute dtype (f32).
///
/// **Typed reject** (surfaced, never a silent degraded load) when an `F8_E4M3` weight has no
/// `.weight_scale` sibling — a partial/unsupported checkpoint. Source tensors are loaded lazily from the
/// mmap and dropped after each op, so only the produced map is held resident. Leaves the tested
/// [`build_target_state_dict`]/[`convert_and_assemble`] (klein, bf16, no dequant) untouched.
pub(crate) fn build_comfyui_dit_map(
    src: &MmapedSafetensors,
    dtype: DType,
) -> Result<HashMap<String, Tensor>> {
    let cpu = Device::Cpu;
    let names: Vec<String> = src.tensors().into_iter().map(|(name, _)| name).collect();
    // Dequant-aware load: an `F8_E4M3` `.weight` is multiplied by its `.weight_scale` sibling and
    // upcast to `dtype`; every other tensor is a plain cast. `chunk3`/`swap_halves` receive the
    // already-cast tensor (the qkv + adaLN are BF16, so no dequant runs through them).
    let load = |name: &str| -> Result<Tensor> {
        let t = src
            .load(name, &cpu)
            .map_err(|e| Error::Msg(format!("flux2 comfyui: source is missing `{name}`: {e}")))?;
        if t.dtype() != DType::F8E4M3 {
            return t.to_dtype(dtype);
        }
        // Inline-scale fp8: dequant against the `.weight_scale` sibling (typed-reject if absent).
        let base = name.strip_suffix(".weight").ok_or_else(|| {
            Error::Msg(format!(
                "flux2 comfyui: fp8 tensor `{name}` does not end in `.weight`"
            ))
        })?;
        let scale_name = format!("{base}.weight_scale");
        let scale = src.load(&scale_name, &cpu).map_err(|e| {
            Error::Msg(format!(
                "flux2 comfyui: fp8 weight `{name}` is missing its `{scale_name}` scale sibling \
                 (partial/unsupported checkpoint?): {e}"
            ))
        })?;
        // Scalar (shape `[]`); flatten so a stray `[1]` shape is tolerated too.
        let s = scale
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let s = *s
            .first()
            .ok_or_else(|| Error::Msg(format!("flux2 comfyui: `{scale_name}` is empty")))?;
        t.to_dtype(DType::F32)?
            .affine(s as f64, 0.0)?
            .to_dtype(dtype)
    };

    let mut out: HashMap<String, Tensor> = HashMap::new();
    for (s, d) in TOP_RENAMES {
        out.insert((*d).to_string(), load(s)?);
    }
    for (s, d) in COMFY_GUIDANCE_RENAMES {
        out.insert((*d).to_string(), load(s)?);
    }
    out.insert(ADALN_TARGET.to_string(), swap_halves(&load(ADALN_SOURCE)?)?);

    let n_double = count_blocks(names.iter().map(String::as_str), "double_blocks");
    for i in 0..n_double {
        let (s, d) = (
            format!("double_blocks.{i}"),
            format!("transformer_blocks.{i}"),
        );
        for (src_suffix, [q, k, v]) in DOUBLE_QKV {
            let [tq, tk, tv] = chunk3(&load(&format!("{s}.{src_suffix}"))?)?;
            out.insert(format!("{d}.{q}"), tq);
            out.insert(format!("{d}.{k}"), tk);
            out.insert(format!("{d}.{v}"), tv);
        }
        for (src_suffix, dst_suffix) in COMFY_DOUBLE_RENAMES {
            out.insert(
                format!("{d}.{dst_suffix}"),
                load(&format!("{s}.{src_suffix}"))?,
            );
        }
    }

    let n_single = count_blocks(names.iter().map(String::as_str), "single_blocks");
    for i in 0..n_single {
        let (s, d) = (
            format!("single_blocks.{i}"),
            format!("single_transformer_blocks.{i}"),
        );
        for (src_suffix, dst_suffix) in COMFY_SINGLE_RENAMES {
            out.insert(
                format!("{d}.{dst_suffix}"),
                load(&format!("{s}.{src_suffix}"))?,
            );
        }
    }

    Ok(out)
}

/// The `.safetensors` shards in a transformer dir (sorted).
fn safetensors_shards(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut shards: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors"))
        .filter(|p| !candle_gen::gen_core::weightsmeta::is_hidden_file(p))
        .collect();
    shards.sort();
    Ok(shards)
}

/// Hard guard: the produced key set + shapes must exactly match the base klein diffusers transformer
/// (the ground-truth layout the loader consumes). Catches a botched remap (missing / extra / wrong-shape
/// keys) at convert time rather than as garbage at generate time. Header-only read of the base shards.
fn validate_against_base(
    produced: &HashMap<String, Tensor>,
    base_transformer_dir: &Path,
) -> Result<()> {
    let shards = safetensors_shards(base_transformer_dir)?;
    if shards.is_empty() {
        return Err(Error::Msg(format!(
            "no base transformer safetensors in {}",
            base_transformer_dir.display()
        )));
    }
    // SAFETY: mmap of read-only weight files (header parse only; we never `.load` the bodies here).
    let base_st = unsafe { MmapedSafetensors::multi(&shards)? };
    let base: HashMap<String, Vec<usize>> = base_st
        .tensors()
        .into_iter()
        .map(|(name, view)| (name, view.shape().to_vec()))
        .collect();

    let mut missing: Vec<&String> = base.keys().filter(|k| !produced.contains_key(*k)).collect();
    let mut extra: Vec<&String> = produced.keys().filter(|k| !base.contains_key(*k)).collect();
    let mut bad_shape: Vec<&String> = produced
        .iter()
        .filter(|(k, v)| base.get(*k).is_some_and(|b| b.as_slice() != v.dims()))
        .map(|(k, _)| k)
        .collect();
    if missing.is_empty() && extra.is_empty() && bad_shape.is_empty() {
        return Ok(());
    }
    missing.sort();
    extra.sort();
    bad_shape.sort();
    Err(Error::Msg(format!(
        "flux2 convert validation FAILED vs base transformer: {} missing, {} extra, {} shape mismatch. \
         missing={:?} extra={:?} shape={:?}",
        missing.len(),
        extra.len(),
        bad_shape.len(),
        &missing[..missing.len().min(5)],
        &extra[..extra.len().min(5)],
        &bad_shape[..bad_shape.len().min(5)],
    )))
}

/// Remove an existing path (file, symlink, or directory) if present, so a re-convert is idempotent.
fn remove_if_exists(path: &Path) -> Result<()> {
    // `symlink_metadata` does not follow the link, so a dangling symlink is still detected.
    match std::fs::symlink_metadata(path) {
        Ok(meta) if meta.is_dir() => std::fs::remove_dir_all(path)?,
        Ok(_) => std::fs::remove_file(path)?,
        Err(ref e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(e.into()),
    }
    Ok(())
}

/// Borrow a base component directory into `dst`. unix: an absolute symlink (matches MLX). windows: a
/// hardlinked file tree (no privilege, no reparse-point read defect, no duplication on the same volume),
/// copy fallback for cross-volume.
fn borrow_dir(src: &Path, dst: &Path) -> Result<()> {
    remove_if_exists(dst)?;
    #[cfg(unix)]
    {
        let canonical = std::fs::canonicalize(src)?;
        std::os::unix::fs::symlink(&canonical, dst)?;
        Ok(())
    }
    #[cfg(windows)]
    {
        link_tree(src, dst)
    }
    #[cfg(not(any(unix, windows)))]
    {
        link_tree(src, dst)
    }
}

/// Recreate `src`'s directory tree under `dst`, hardlinking each (canonicalized — HF-cache files are
/// themselves symlinks/hardlinks to `blobs/`) file; copy fallback when a hardlink can't be made
/// (e.g. cross-volume). Used on windows (and any non-unix target).
#[cfg(not(unix))]
fn link_tree(src: &Path, dst: &Path) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            link_tree(&from, &to)?;
        } else {
            let canonical = std::fs::canonicalize(&from)?;
            if std::fs::hard_link(&canonical, &to).is_err() {
                std::fs::copy(&canonical, &to)?;
            }
        }
    }
    Ok(())
}

/// Convert `source_file` (an original single-file FLUX.2-klein transformer in BFL convention) into
/// `out_dir` as a complete diffusers model dir, borrowing the VAE / text encoder / tokenizer /
/// scheduler from `base_dir` (an installed base FLUX.2-klein-9B diffusers snapshot). Returns `out_dir`.
/// The result loads directly through the candle [`crate::config::FLUX2_KLEIN_9B_ID`] loader via the
/// worker's `modelPath` seam.
///
/// Candle twin of `mlx_gen_flux2::convert::convert_and_assemble` (sc-3136 / sc-7459). The transformer
/// weights + its `config.json` and `model_index.json` are written as real files (so they survive the
/// worker's temp→final atomic rename); the borrowed component dirs are absolute symlinks (unix) or
/// hardlink trees (windows).
pub fn convert_and_assemble(
    source_file: impl AsRef<Path>,
    base_dir: impl AsRef<Path>,
    out_dir: impl AsRef<Path>,
) -> Result<PathBuf> {
    let source = source_file.as_ref();
    let base = base_dir.as_ref();
    let out = out_dir.as_ref();
    let base_transformer = base.join("transformer");
    if !source.is_file() {
        return Err(Error::Msg(format!(
            "flux2 convert: source transformer file not found: {}",
            source.display()
        )));
    }
    if !base_transformer.is_dir() {
        return Err(Error::Msg(format!(
            "flux2 convert: base transformer dir not found: {}",
            base_transformer.display()
        )));
    }

    // SAFETY: mmap of a read-only weight file; standard candle loading path.
    let src = unsafe { MmapedSafetensors::new(source)? };
    let produced = build_target_state_dict(&src)?;
    validate_against_base(&produced, &base_transformer)?;

    let out_transformer = out.join("transformer");
    std::fs::create_dir_all(&out_transformer)?;
    candle_gen::candle_core::safetensors::save(
        &produced,
        out_transformer.join(TRANSFORMER_WEIGHTS),
    )?;
    std::fs::copy(
        base_transformer.join("config.json"),
        out_transformer.join("config.json"),
    )?;

    // Borrow the untouched components from the base klein snapshot.
    for name in BORROWED_FILES {
        let src_path = std::fs::canonicalize(base.join(name))?;
        let dst = out.join(name);
        remove_if_exists(&dst)?;
        std::fs::copy(&src_path, &dst)?;
    }
    for name in BORROWED_SUBDIRS_REQUIRED {
        let src_path = base.join(name);
        if !src_path.is_dir() {
            return Err(Error::Msg(format!(
                "flux2 convert: base component missing: {}",
                src_path.display()
            )));
        }
        borrow_dir(&src_path, &out.join(name))?;
    }
    for name in BORROWED_SUBDIRS_OPTIONAL {
        let src_path = base.join(name);
        if src_path.is_dir() {
            borrow_dir(&src_path, &out.join(name))?;
        }
    }

    Ok(out.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(rows: usize, cols: usize, start: f32) -> Tensor {
        let v: Vec<f32> = (0..rows * cols).map(|i| start + i as f32).collect();
        Tensor::from_vec(v, (rows, cols), &Device::Cpu).unwrap()
    }

    #[test]
    fn chunk3_splits_in_qkv_order() {
        // [6, 2] = three [2,2] row chunks: q=rows0-1, k=rows2-3, v=rows4-5.
        let t = ramp(6, 2, 0.0);
        let [q, k, v] = chunk3(&t).unwrap();
        assert_eq!(q.dims(), &[2, 2]);
        assert_eq!(
            q.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![0., 1., 2., 3.]
        );
        assert_eq!(
            k.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![4., 5., 6., 7.]
        );
        assert_eq!(
            v.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![8., 9., 10., 11.]
        );
        // A non-divisible row count is rejected.
        assert!(chunk3(&ramp(5, 2, 0.0)).is_err());
        // sc-11028: the chunks must be OWNED zero-offset tensors, not views into the fused
        // storage — `QTensor::quantize{,_onto}` reads the raw backing storage (ignoring a view's
        // start offset), so an offset-view k/v fed into the in-memory-map → quantize path would
        // quantize q's rows instead.
        for (name, c) in [("q", &q), ("k", &k), ("v", &v)] {
            assert_eq!(
                c.layout().start_offset(),
                0,
                "chunk3 `{name}` must be materialized (owned, zero-offset), not an offset view"
            );
        }
    }

    #[test]
    fn swap_halves_swaps_shift_and_scale() {
        // [4, 2]: BFL (shift=rows0-1, scale=rows2-3) → diffusers (scale, shift).
        let t = ramp(4, 2, 0.0);
        let s = swap_halves(&t).unwrap();
        assert_eq!(s.dims(), &[4, 2]);
        // Now scale (was rows2-3) comes first, then shift (was rows0-1).
        assert_eq!(
            s.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![4., 5., 6., 7., 0., 1., 2., 3.]
        );
        assert!(swap_halves(&ramp(3, 2, 0.0)).is_err());
    }

    /// A minimal but complete fixture: one double + one single block + every top-level key, in BFL
    /// convention, plus a base diffusers snapshot whose transformer has exactly the keys/shapes the
    /// remap should produce. Proves the full key remap + assemble + borrow, and that
    /// `validate_against_base` passes only when the produced layout matches.
    #[test]
    fn convert_and_assemble_remaps_keys_and_borrows() {
        let d = 4usize; // inner width; all 2-D weights are [out, in] over this.
        let tmp = std::env::temp_dir().join(format!("cg_flux2_convert_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let src_dir = tmp.join("src");
        let base = tmp.join("base");
        let out = tmp.join("out");
        std::fs::create_dir_all(&src_dir).unwrap();
        let base_transformer = base.join("transformer");
        std::fs::create_dir_all(&base_transformer).unwrap();

        // --- source single-file (BFL keys) ---
        let mut src: HashMap<String, Tensor> = HashMap::new();
        // Top-level: every TOP_RENAME source (square [d, d] here for simplicity).
        for (s, _) in TOP_RENAMES {
            src.insert((*s).to_string(), ramp(d, d, 1.0));
        }
        // adaLN packs (shift, scale): [2d, d].
        src.insert(ADALN_SOURCE.to_string(), ramp(2 * d, d, 100.0));
        // Double block 0: fused qkv [3d, d] per stream + the renamed leaves.
        src.insert(
            "double_blocks.0.img_attn.qkv.weight".into(),
            ramp(3 * d, d, 10.0),
        );
        src.insert(
            "double_blocks.0.txt_attn.qkv.weight".into(),
            ramp(3 * d, d, 20.0),
        );
        src.insert(
            "double_blocks.0.img_attn.norm.query_norm.weight".into(),
            ramp(1, d, 0.0),
        );
        src.insert(
            "double_blocks.0.img_attn.norm.key_norm.weight".into(),
            ramp(1, d, 0.0),
        );
        src.insert(
            "double_blocks.0.img_attn.proj.weight".into(),
            ramp(d, d, 0.0),
        );
        src.insert(
            "double_blocks.0.img_mlp.0.weight".into(),
            ramp(2 * d, d, 0.0),
        );
        src.insert(
            "double_blocks.0.img_mlp.2.weight".into(),
            ramp(d, 2 * d, 0.0),
        );
        src.insert(
            "double_blocks.0.txt_attn.norm.query_norm.weight".into(),
            ramp(1, d, 0.0),
        );
        src.insert(
            "double_blocks.0.txt_attn.norm.key_norm.weight".into(),
            ramp(1, d, 0.0),
        );
        src.insert(
            "double_blocks.0.txt_attn.proj.weight".into(),
            ramp(d, d, 0.0),
        );
        src.insert(
            "double_blocks.0.txt_mlp.0.weight".into(),
            ramp(2 * d, d, 0.0),
        );
        src.insert(
            "double_blocks.0.txt_mlp.2.weight".into(),
            ramp(d, 2 * d, 0.0),
        );
        // Single block 0.
        src.insert("single_blocks.0.linear1.weight".into(), ramp(3 * d, d, 0.0));
        src.insert("single_blocks.0.linear2.weight".into(), ramp(d, 3 * d, 0.0));
        src.insert(
            "single_blocks.0.norm.query_norm.weight".into(),
            ramp(1, d, 0.0),
        );
        src.insert(
            "single_blocks.0.norm.key_norm.weight".into(),
            ramp(1, d, 0.0),
        );
        let source_file = src_dir.join("model.safetensors");
        candle_gen::candle_core::safetensors::save(&src, &source_file).unwrap();

        // --- base diffusers transformer (the expected produced layout) ---
        let mut base_tf: HashMap<String, Tensor> = HashMap::new();
        for (_, dkey) in TOP_RENAMES {
            base_tf.insert((*dkey).to_string(), ramp(d, d, 0.0));
        }
        base_tf.insert(ADALN_TARGET.to_string(), ramp(2 * d, d, 0.0));
        for q in ["attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight"] {
            base_tf.insert(format!("transformer_blocks.0.{q}"), ramp(d, d, 0.0));
        }
        for q in [
            "attn.add_q_proj.weight",
            "attn.add_k_proj.weight",
            "attn.add_v_proj.weight",
        ] {
            base_tf.insert(format!("transformer_blocks.0.{q}"), ramp(d, d, 0.0));
        }
        base_tf.insert(
            "transformer_blocks.0.attn.norm_q.weight".into(),
            ramp(1, d, 0.0),
        );
        base_tf.insert(
            "transformer_blocks.0.attn.norm_k.weight".into(),
            ramp(1, d, 0.0),
        );
        base_tf.insert(
            "transformer_blocks.0.attn.to_out.0.weight".into(),
            ramp(d, d, 0.0),
        );
        base_tf.insert(
            "transformer_blocks.0.ff.linear_in.weight".into(),
            ramp(2 * d, d, 0.0),
        );
        base_tf.insert(
            "transformer_blocks.0.ff.linear_out.weight".into(),
            ramp(d, 2 * d, 0.0),
        );
        base_tf.insert(
            "transformer_blocks.0.attn.norm_added_q.weight".into(),
            ramp(1, d, 0.0),
        );
        base_tf.insert(
            "transformer_blocks.0.attn.norm_added_k.weight".into(),
            ramp(1, d, 0.0),
        );
        base_tf.insert(
            "transformer_blocks.0.attn.to_add_out.weight".into(),
            ramp(d, d, 0.0),
        );
        base_tf.insert(
            "transformer_blocks.0.ff_context.linear_in.weight".into(),
            ramp(2 * d, d, 0.0),
        );
        base_tf.insert(
            "transformer_blocks.0.ff_context.linear_out.weight".into(),
            ramp(d, 2 * d, 0.0),
        );
        base_tf.insert(
            "single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight".into(),
            ramp(3 * d, d, 0.0),
        );
        base_tf.insert(
            "single_transformer_blocks.0.attn.to_out.weight".into(),
            ramp(d, 3 * d, 0.0),
        );
        base_tf.insert(
            "single_transformer_blocks.0.attn.norm_q.weight".into(),
            ramp(1, d, 0.0),
        );
        base_tf.insert(
            "single_transformer_blocks.0.attn.norm_k.weight".into(),
            ramp(1, d, 0.0),
        );
        candle_gen::candle_core::safetensors::save(
            &base_tf,
            base_transformer.join("diffusion_pytorch_model.safetensors"),
        )
        .unwrap();
        std::fs::write(base_transformer.join("config.json"), b"{}").unwrap();
        // Borrowed components.
        for sub in ["vae", "text_encoder", "tokenizer", "scheduler"] {
            std::fs::create_dir_all(base.join(sub)).unwrap();
            std::fs::write(base.join(sub).join("config.json"), b"{}").unwrap();
        }
        std::fs::write(base.join("model_index.json"), b"{}").unwrap();

        // --- convert ---
        let result = convert_and_assemble(&source_file, &base, &out).unwrap();
        assert_eq!(result, out);

        // Produced transformer loads + has EXACTLY the base key set.
        let produced = candle_gen::candle_core::safetensors::load(
            out.join("transformer").join(TRANSFORMER_WEIGHTS),
            &Device::Cpu,
        )
        .unwrap();
        let mut got: Vec<&String> = produced.keys().collect();
        let mut want: Vec<&String> = base_tf.keys().collect();
        got.sort();
        want.sort();
        assert_eq!(
            got, want,
            "produced key set must equal the base diffusers transformer"
        );

        // The qkv split is in q/k/v order: to_q == first third of the source fused img qkv.
        let to_q = produced["transformer_blocks.0.attn.to_q.weight"]
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let expect_q: Vec<f32> = (0..d * d).map(|i| 10.0 + i as f32).collect();
        assert_eq!(to_q, expect_q);

        // The adaLN half-swap landed: norm_out.linear first half == source second half (scale).
        let norm_out = produced[ADALN_TARGET]
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        // source adaLN ramps from 100.0 over [2d, d]; second half starts at 100.0 + d*d.
        assert_eq!(norm_out[0], 100.0 + (d * d) as f32);

        // Borrowed components are present + readable in the converted dir.
        for sub in ["vae", "text_encoder", "tokenizer", "scheduler"] {
            assert!(
                out.join(sub).join("config.json").is_file(),
                "{sub} borrowed"
            );
        }
        assert!(out.join("model_index.json").is_file());
        assert!(out.join("transformer").join("config.json").is_file());

        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// A scalar (rank-0) F32 tensor — the shape the inline-scale `.weight_scale`/`.input_scale`
    /// siblings carry in the measured `flux2_dev_fp8mixed` header.
    fn scalar(v: f32) -> Tensor {
        Tensor::from_vec(vec![v], (), &Device::Cpu).unwrap()
    }

    /// A minimal BFL-native fp8-mixed fixture (one double + one single block + every top-level key,
    /// including the dev `guidance_in` and the `.scale`-named RMSNorms): the MLP Linears are `F8_E4M3`
    /// with `.weight_scale`/`.input_scale` scalar siblings, everything else BF16. When
    /// `include_single_linear1_scale` is false the `single_blocks.0.linear1.weight_scale` sibling is
    /// omitted (to exercise the typed reject). Returns the saved file path + the two fp8 weights that
    /// must dequant, keyed by their diffusers target, with their scale, so the caller can assert the
    /// exact `w = w_fp8·weight_scale` math independent of fp8 rounding.
    fn write_comfyui_fixture(
        dir: &Path,
        d: usize,
        include_single_linear1_scale: bool,
    ) -> (PathBuf, Vec<(String, Tensor, f32)>) {
        std::fs::create_dir_all(dir).unwrap();
        let mut src: HashMap<String, Tensor> = HashMap::new();
        // Top-level (BF16) — every TOP_RENAME + the dev guidance embedder + adaLN.
        for (s, _) in TOP_RENAMES {
            src.insert(
                (*s).to_string(),
                ramp(d, d, 1.0).to_dtype(DType::BF16).unwrap(),
            );
        }
        for (s, _) in COMFY_GUIDANCE_RENAMES {
            src.insert(
                (*s).to_string(),
                ramp(d, d, 5.0).to_dtype(DType::BF16).unwrap(),
            );
        }
        src.insert(
            ADALN_SOURCE.to_string(),
            ramp(2 * d, d, 100.0).to_dtype(DType::BF16).unwrap(),
        );
        // Double block 0: BF16 qkv/proj/norms(.scale), fp8 mlp with scale siblings.
        src.insert(
            "double_blocks.0.img_attn.qkv.weight".into(),
            ramp(3 * d, d, 10.0).to_dtype(DType::BF16).unwrap(),
        );
        src.insert(
            "double_blocks.0.txt_attn.qkv.weight".into(),
            ramp(3 * d, d, 20.0).to_dtype(DType::BF16).unwrap(),
        );
        for leaf in [
            "img_attn.norm.query_norm.scale",
            "img_attn.norm.key_norm.scale",
            "txt_attn.norm.query_norm.scale",
            "txt_attn.norm.key_norm.scale",
        ] {
            src.insert(
                format!("double_blocks.0.{leaf}"),
                ramp(1, d, 0.0).to_dtype(DType::BF16).unwrap(),
            );
        }
        src.insert(
            "double_blocks.0.img_attn.proj.weight".into(),
            ramp(d, d, 0.0).to_dtype(DType::BF16).unwrap(),
        );
        src.insert(
            "double_blocks.0.txt_attn.proj.weight".into(),
            ramp(d, d, 0.0).to_dtype(DType::BF16).unwrap(),
        );
        let mut expect: Vec<(String, Tensor, f32)> = Vec::new();
        // fp8 MLP leaves: (source key, target key, [out,in], scale).
        let fp8_leaves: &[(&str, &str, [usize; 2], f32)] = &[
            (
                "double_blocks.0.img_mlp.0.weight",
                "transformer_blocks.0.ff.linear_in.weight",
                [2 * d, d],
                2.0,
            ),
            (
                "double_blocks.0.img_mlp.2.weight",
                "transformer_blocks.0.ff.linear_out.weight",
                [d, 2 * d],
                0.5,
            ),
            (
                "double_blocks.0.txt_mlp.0.weight",
                "transformer_blocks.0.ff_context.linear_in.weight",
                [2 * d, d],
                3.0,
            ),
            (
                "double_blocks.0.txt_mlp.2.weight",
                "transformer_blocks.0.ff_context.linear_out.weight",
                [d, 2 * d],
                0.25,
            ),
            (
                "single_blocks.0.linear1.weight",
                "single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight",
                [3 * d, d],
                4.0,
            ),
            (
                "single_blocks.0.linear2.weight",
                "single_transformer_blocks.0.attn.to_out.weight",
                [d, 3 * d],
                1.5,
            ),
        ];
        for (skey, dkey, [rows, cols], scale) in fp8_leaves {
            let w_fp8 = ramp(*rows, *cols, 1.0).to_dtype(DType::F8E4M3).unwrap();
            // The exact dequant target: (fp8 → f32) · scale — independent of fp8 rounding.
            let expected = w_fp8
                .to_dtype(DType::F32)
                .unwrap()
                .affine(*scale as f64, 0.0)
                .unwrap();
            expect.push(((*dkey).to_string(), expected, *scale));
            let is_omitted_scale =
                !include_single_linear1_scale && *skey == "single_blocks.0.linear1.weight";
            let base = skey.strip_suffix(".weight").unwrap();
            src.insert((*skey).to_string(), w_fp8);
            if !is_omitted_scale {
                src.insert(format!("{base}.weight_scale"), scalar(*scale));
            }
            // The activation scale is always present but never consumed.
            src.insert(format!("{base}.input_scale"), scalar(7.0));
        }
        // Single block 0 BF16 norms.
        for leaf in ["norm.query_norm.scale", "norm.key_norm.scale"] {
            src.insert(
                format!("single_blocks.0.{leaf}"),
                ramp(1, d, 0.0).to_dtype(DType::BF16).unwrap(),
            );
        }
        let file = dir.join("flux2_dev_fp8mixed_fixture.safetensors");
        candle_gen::candle_core::safetensors::save(&src, &file).unwrap();
        (file, expect)
    }

    #[test]
    fn build_comfyui_dit_map_dequants_and_remaps() {
        let d = 4usize;
        let tmp = std::env::temp_dir().join(format!("cg_flux2_comfyui_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let (file, expect) = write_comfyui_fixture(&tmp, d, true);

        // SAFETY: read-only mmap of the fixture we just wrote.
        let mmap = unsafe { MmapedSafetensors::new(&file).unwrap() };
        let map = build_comfyui_dit_map(&mmap, DType::F32).unwrap();

        // Every fp8 MLP leaf dequants to exactly `(w_fp8 → f32)·weight_scale` under its diffusers key.
        for (dkey, expected, scale) in &expect {
            let got = map
                .get(dkey)
                .unwrap_or_else(|| panic!("missing dequanted key {dkey}"));
            assert_eq!(got.dtype(), DType::F32);
            let g = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let e = expected.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            assert_eq!(
                g, e,
                "dequant of {dkey} (scale {scale}) must be w_fp8·weight_scale"
            );
        }

        // The dev guidance embedder is present (switches the DiT into guidance-distilled dev mode).
        assert!(map.contains_key("time_guidance_embed.guidance_embedder.linear_1.weight"));
        assert!(map.contains_key("time_guidance_embed.guidance_embedder.linear_2.weight"));
        assert!(map.contains_key("time_guidance_embed.timestep_embedder.linear_1.weight"));

        // The BFL-official `.scale` RMSNorms landed under the diffusers `.weight` names.
        for k in [
            "transformer_blocks.0.attn.norm_q.weight",
            "transformer_blocks.0.attn.norm_added_k.weight",
            "single_transformer_blocks.0.attn.norm_q.weight",
        ] {
            assert!(map.contains_key(k), "expected remapped norm {k}");
        }

        // The scale companions are consumed, never emitted (no `.weight_scale`/`.input_scale` leaks).
        assert!(
            !map.keys()
                .any(|k| k.contains("weight_scale") || k.contains("input_scale")),
            "scale siblings must be dropped"
        );
        // The qkv row-split still runs (BF16, no dequant): q is the first third of the fused img qkv.
        assert!(map.contains_key("transformer_blocks.0.attn.to_q.weight"));

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn build_comfyui_dit_map_rejects_missing_weight_scale() {
        let d = 4usize;
        let tmp = std::env::temp_dir().join(format!("cg_flux2_comfyui_bad_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        // Omit `single_blocks.0.linear1.weight_scale` → the fp8 weight has no scale sibling.
        let (file, _) = write_comfyui_fixture(&tmp, d, false);
        let mmap = unsafe { MmapedSafetensors::new(&file).unwrap() };
        let err = build_comfyui_dit_map(&mmap, DType::F32).expect_err(
            "an fp8 weight with no .weight_scale must be a typed reject, not a silent load",
        );
        let msg = format!("{err}");
        assert!(
            msg.contains("weight_scale") && msg.contains("linear1"),
            "the reject must name the missing scale sibling, got: {msg}"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
