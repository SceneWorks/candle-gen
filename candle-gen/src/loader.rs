//! Shared sorted-`.safetensors` → unsafe-mmap loader (sc-8999 / F-019).
//!
//! Nearly every provider crate re-implemented the same idiom: list a snapshot component directory,
//! keep the `.safetensors` entries, sort them into a deterministic shard order, error with a crafted
//! message if none are found, then `unsafe { VarBuilder::from_mmaped_safetensors(..) }` them into a
//! [`VarBuilder`]. It was duplicated ~34 times, with error-string drift and one behavioral drift
//! (flux2's single-file control checkpoint skipped the "no .safetensors" check — a missing file
//! surfaced as a raw mmap error instead of the crafted message). Consolidating gives the workspace a
//! *single* audited `unsafe` mmap surface (also aids F-062): shard handling, error text, and the
//! SAFETY invariant now live in one place.
//!
//! Two entry shapes cover every call site:
//! - Directory of shards ([`sorted_safetensors`] / [`load_sorted_mmap`] / [`component_vb`]) — a
//!   diffusers component subdir (`transformer/`, `vae/`, `text_encoder/`, …) that always holds ≥1
//!   `.safetensors` shard.
//! - A path that is *either* a single `.safetensors` file *or* a directory of shards
//!   ([`resolve_weight_files`] / [`load_path_mmap`]) — control / IP-adapter checkpoints that ship as a
//!   lone file or a sharded dir.
//!
//! ## Shard ordering invariant (load-bearing)
//! The shard order is a lexical [`slice::sort`] of the resolved paths, reproduced *exactly* from the
//! call sites this replaced. When the same tensor key appears in more than one shard (rare, but real
//! for some re-exported checkpoints) the *last* shard in this order wins inside candle's
//! `from_mmaped_safetensors`. Do NOT change the sort key (e.g. to a numeric shard index): it would
//! silently pick a different shard's copy of a duplicated key and corrupt weight loading.

use std::path::{Path, PathBuf};

use candle_core::{DType, Device};
use candle_nn::VarBuilder;

use crate::{CandleError, Result};

/// The deterministic, sorted list of every `*.safetensors` file directly under `dir`.
///
/// `label` prefixes the error text (e.g. `"flux"`, `"z-image control"`) so callers keep their
/// crafted, provider-specific diagnostics. Errors if `dir` cannot be read or holds no
/// `.safetensors`.
///
/// The sort is a plain lexical [`slice::sort`] — see the module docs for why this ordering is
/// load-bearing and must not change.
///
/// Hidden entries are skipped ([`gen_core::weightsmeta::is_hidden_file`]): a macOS AppleDouble
/// sidecar (`._model.safetensors`) carries the `.safetensors` extension and sorts *ahead* of the
/// real shard, so an extension-only filter would hand it to the mmap first (SceneWorks#1333).
pub fn sorted_safetensors(dir: &Path, label: &str) -> Result<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| CandleError::Msg(format!("{label}: read {}: {e}", dir.display())))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .filter(|p| !gen_core::weightsmeta::is_hidden_file(p))
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(CandleError::Msg(format!(
            "{label}: no .safetensors found in {}",
            dir.display()
        )));
    }
    Ok(files)
}

/// mmap a [`VarBuilder`] at `dtype`/`device` over every `.safetensors` shard in `dir`
/// (deterministically sorted; errors if the dir is unreadable or empty). This is the shared body the
/// story calls `component_vb(dir, dtype, device, label)`.
pub fn load_sorted_mmap(
    dir: &Path,
    dtype: DType,
    device: &Device,
    label: &str,
) -> Result<VarBuilder<'static>> {
    let files = sorted_safetensors(dir, label)?;
    mmap_var_builder(&files, dtype, device)
}

/// mmap a [`VarBuilder`] over the snapshot component subdir `sub` under `root` (i.e.
/// `load_sorted_mmap(&root.join(sub), ..)`, but with a "missing component dir" check that names the
/// snapshot root — the shape the majority of providers use for `transformer/`, `vae/`,
/// `text_encoder/`, …).
pub fn component_vb(
    root: &Path,
    sub: &str,
    dtype: DType,
    device: &Device,
    label: &str,
) -> Result<VarBuilder<'static>> {
    let dir = root.join(sub);
    if !dir.is_dir() {
        return Err(CandleError::Msg(format!(
            "{label}: snapshot missing the {sub}/ component directory (at {})",
            root.display()
        )));
    }
    load_sorted_mmap(&dir, dtype, device, label)
}

/// Resolve a checkpoint `path` that is *either* a single `.safetensors` file *or* a directory of
/// sharded `.safetensors` into the sorted file list to mmap.
///
/// This is the file-or-dir shape used by control / IP-adapter checkpoints. Crucially, unlike the
/// pre-consolidation flux2 copy, a **missing** single-file path yields the crafted `label`-prefixed
/// "not found" error here (the caller's message), not a raw mmap failure downstream (fixes the F-019
/// flux2 drift).
pub fn resolve_weight_files(path: &Path, label: &str) -> Result<Vec<PathBuf>> {
    if path.is_file() {
        return Ok(vec![path.to_path_buf()]);
    }
    if path.is_dir() {
        // A dir must hold ≥1 shard; reuse the same sorted/error-if-empty semantics.
        return sorted_safetensors(path, label);
    }
    Err(CandleError::Msg(format!(
        "{label}: no .safetensors at {} (expected a .safetensors file or a dir of shards)",
        path.display()
    )))
}

/// mmap a [`VarBuilder`] over a file-or-dir checkpoint `path` at `dtype`/`device`
/// ([`resolve_weight_files`] + mmap).
pub fn load_path_mmap(
    path: &Path,
    dtype: DType,
    device: &Device,
    label: &str,
) -> Result<VarBuilder<'static>> {
    let files = resolve_weight_files(path, label)?;
    mmap_var_builder(&files, dtype, device)
}

/// mmap a single `.safetensors` `file` and materialize **only** the tensor named `name`, cast to
/// `dtype` and placed on `device`. `label` prefixes the crafted "missing tensor" error so callers keep
/// their provider-specific diagnostics.
///
/// This is the header-only counterpart to [`mmap_var_builder`] for the F-010 pattern: several call
/// sites called `candle_core::safetensors::load(file, device)` — which materializes **every** tensor of
/// a CLIP checkpoint on the device — solely to extract one head tensor (e.g. `text_projection.weight`),
/// costing ~1.4–1.7 GB of transient VRAM and a second full disk read. Safetensors keeps the tensor
/// index in the file header, so an mmap lets us read a single view and load just its bytes; the peak
/// footprint drops to that one tensor.
///
/// The returned tensor is **byte-identical** to `safetensors::load(file, device)?.get(name)?
/// .to_dtype(dtype)?`: the view is loaded at its stored dtype onto `device`, then cast to `dtype`
/// (skipping the cast when it already matches, so no redundant copy).
pub fn load_one_tensor(
    file: &Path,
    name: &str,
    dtype: DType,
    device: &Device,
    label: &str,
) -> Result<candle_core::Tensor> {
    // SAFETY: same invariant as `mmap_var_builder` — a read-only, process-owned weight file, mapped
    // only for the duration of this call and not mutated behind the mapping.
    let st = unsafe { candle_core::safetensors::MmapedSafetensors::new(file)? };
    // Craft the label-prefixed missing-key message rather than surfacing candle's raw error, matching
    // the F-019 loader style and the messages the pre-consolidation call sites produced.
    if st.get(name).is_err() {
        return Err(CandleError::Msg(format!(
            "{label}: tensor `{name}` missing from {}",
            file.display()
        )));
    }
    let t = st.load(name, device)?;
    if t.dtype() == dtype {
        Ok(t)
    } else {
        Ok(t.to_dtype(dtype)?)
    }
}

/// Shard-aware [`load_one_tensor`]: locate the single shard in `files` whose header carries `name`
/// (a header-only probe, no tensor bytes read) and materialize just that one tensor from it.
///
/// The single-file [`load_one_tensor`] is the F-010 fast path for the common snapshot that ships the
/// whole checkpoint in one file. When a snapshot is resharded across several `*.safetensors` (the
/// F-037 / sc-9021 fallback), the head tensor lives in exactly one of them; this scans the shard list
/// in order and reads it from the first shard that has it, preserving the header-only footprint.
pub fn load_one_tensor_sharded(
    files: &[PathBuf],
    name: &str,
    dtype: DType,
    device: &Device,
    label: &str,
) -> Result<candle_core::Tensor> {
    for file in files {
        // SAFETY: same invariant as `load_one_tensor` — read-only, process-owned weight file mapped
        // only to probe the header, not mutated behind the mapping.
        let st = unsafe { candle_core::safetensors::MmapedSafetensors::new(file)? };
        if st.get(name).is_ok() {
            return load_one_tensor(file, name, dtype, device, label);
        }
    }
    Err(CandleError::Msg(format!(
        "{label}: tensor `{name}` missing from all {} shard(s)",
        files.len()
    )))
}

/// The one audited `unsafe` mmap surface: build a `'static` [`VarBuilder`] over the already-resolved
/// shard `files` at `dtype`/`device`.
///
/// # SAFETY
/// `VarBuilder::from_mmaped_safetensors` memory-maps each file and hands out tensors that borrow the
/// mapping for the process lifetime. This is sound because these are the model's own weight files:
/// read-only, owned by (and trusted for) this process, and not mutated or truncated by us while the
/// mapping is live. This is the standard candle weight-loading path; concentrating it here means the
/// invariant is documented and audited once instead of at ~34 scattered call sites (F-019 / F-062).
pub fn mmap_var_builder(
    files: &[PathBuf],
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder<'static>> {
    // SAFETY: see the function-level doc — mmap of read-only, process-owned weight files, live for
    // the process lifetime and not mutated behind the mapping. Standard candle loading path.
    Ok(unsafe { VarBuilder::from_mmaped_safetensors(files, dtype, device)? })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    /// Write a single-tensor `.safetensors` file holding `name -> [value]` (f32).
    fn write_st(path: &Path, name: &str, value: f32) {
        let t = Tensor::new(&[value], &Device::Cpu).unwrap();
        let mut map = std::collections::HashMap::new();
        map.insert(name.to_string(), t);
        candle_core::safetensors::save(&map, path).unwrap();
    }

    fn tmp_dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "candle_gen_loader_test_{tag}_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn sorted_is_lexical_not_numeric() {
        // The sort must be lexical over the full path — the load-bearing shard order. With numeric
        // shard indices `2 < 10`, but lexically `"...-00010..." < "...-00002..."` is FALSE, so the
        // zero-padded diffusers names sort correctly; a *non*-padded scheme sorts lexically (10<2),
        // which is exactly the ordering `from_mmaped_safetensors` will see. Assert the raw sort order.
        let dir = tmp_dir("lexical");
        for n in [
            "model-2.safetensors",
            "model-10.safetensors",
            "model-1.safetensors",
        ] {
            write_st(&dir.join(n), "w", 0.0);
        }
        let files = sorted_safetensors(&dir, "test").unwrap();
        let names: Vec<String> = files
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        // Plain lexical order: "1" < "10" < "2".
        assert_eq!(
            names,
            vec![
                "model-1.safetensors",
                "model-10.safetensors",
                "model-2.safetensors"
            ]
        );

        // Zero-padded shard names (the diffusers convention) sort into true numeric order.
        let dir2 = tmp_dir("padded");
        for n in [
            "model-00002-of-00010.safetensors",
            "model-00010-of-00010.safetensors",
            "model-00001-of-00010.safetensors",
        ] {
            write_st(&dir2.join(n), "w", 0.0);
        }
        let padded: Vec<String> = sorted_safetensors(&dir2, "test")
            .unwrap()
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            padded,
            vec![
                "model-00001-of-00010.safetensors",
                "model-00002-of-00010.safetensors",
                "model-00010-of-00010.safetensors",
            ]
        );
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&dir2);
    }

    #[test]
    fn ignores_non_safetensors_and_errors_when_empty() {
        let dir = tmp_dir("empty");
        std::fs::write(dir.join("config.json"), b"{}").unwrap();
        std::fs::write(dir.join("model.bin"), b"x").unwrap();
        let err = sorted_safetensors(&dir, "myprov").unwrap_err();
        assert!(
            matches!(err, CandleError::Msg(m) if m.contains("myprov") && m.contains("no .safetensors"))
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn multi_shard_key_union_loads() {
        // Two shards, disjoint keys — the union must be visible through the VarBuilder.
        let dir = tmp_dir("union");
        write_st(
            &dir.join("model-00001-of-00002.safetensors"),
            "a.weight",
            1.0,
        );
        write_st(
            &dir.join("model-00002-of-00002.safetensors"),
            "b.weight",
            2.0,
        );
        let vb = load_sorted_mmap(&dir, DType::F32, &Device::Cpu, "test").unwrap();
        assert!(vb.contains_tensor("a.weight"));
        assert!(vb.contains_tensor("b.weight"));
        let a = vb.get(1, "a.weight").unwrap();
        let b = vb.get(1, "b.weight").unwrap();
        assert_eq!(a.to_vec1::<f32>().unwrap(), vec![1.0]);
        assert_eq!(b.to_vec1::<f32>().unwrap(), vec![2.0]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resolve_weight_files_single_file_and_dir_and_missing() {
        let dir = tmp_dir("fileordir");
        // Single-file path.
        let single = dir.join("checkpoint.safetensors");
        write_st(&single, "w", 3.0);
        assert_eq!(
            resolve_weight_files(&single, "ctl").unwrap(),
            vec![single.clone()]
        );

        // Dir-of-shards path.
        let sub = dir.join("shards");
        std::fs::create_dir_all(&sub).unwrap();
        write_st(&sub.join("a.safetensors"), "w", 0.0);
        write_st(&sub.join("b.safetensors"), "w", 0.0);
        assert_eq!(resolve_weight_files(&sub, "ctl").unwrap().len(), 2);

        // Missing path → crafted, label-prefixed error (the F-019 flux2 fix: not a raw mmap failure).
        let missing = dir.join("nope.safetensors");
        let err = resolve_weight_files(&missing, "ctl").unwrap_err();
        assert!(
            matches!(err, CandleError::Msg(m) if m.contains("ctl") && m.contains("no .safetensors"))
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_one_tensor_reads_named_only_and_errors_on_missing() {
        // A multi-tensor file: the helper must return exactly the named tensor (byte-identical to a
        // full load + get + to_dtype) and never depend on the other tensors being materialized.
        let dir = tmp_dir("one_tensor");
        let file = dir.join("checkpoint.safetensors");
        let mut map = std::collections::HashMap::new();
        map.insert(
            "text_projection.weight".to_string(),
            Tensor::new(&[1.0f32, 2.0, 3.0], &Device::Cpu).unwrap(),
        );
        map.insert(
            "unused.big".to_string(),
            Tensor::new(&[9.0f32, 9.0, 9.0, 9.0], &Device::Cpu).unwrap(),
        );
        candle_core::safetensors::save(&map, &file).unwrap();

        // The named tensor comes back with its value + shape intact.
        let t = load_one_tensor(
            &file,
            "text_projection.weight",
            DType::F32,
            &Device::Cpu,
            "prov",
        )
        .unwrap();
        assert_eq!(t.dims(), &[3]);
        assert_eq!(t.to_vec1::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);

        // Byte-identical to the old full-load path.
        let full = candle_core::safetensors::load(&file, &Device::Cpu).unwrap();
        let reference = full
            .get("text_projection.weight")
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        assert_eq!(
            t.to_vec1::<f32>().unwrap(),
            reference.to_vec1::<f32>().unwrap()
        );

        // A missing name yields the crafted, label-prefixed error (not candle's raw message).
        let err =
            load_one_tensor(&file, "nope.weight", DType::F32, &Device::Cpu, "prov").unwrap_err();
        assert!(
            matches!(err, CandleError::Msg(m) if m.contains("prov") && m.contains("nope.weight")),
            "expected a crafted missing-tensor error"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_one_tensor_sharded_finds_the_owning_shard_and_errors_when_absent() {
        // Two shards, each holding a different tensor: the helper probes headers and reads the named
        // tensor from whichever shard owns it — the resharded-snapshot path (F-037 × F-010).
        let dir = tmp_dir("one_tensor_sharded");
        let a = dir.join("model-00001-of-00002.safetensors");
        let b = dir.join("model-00002-of-00002.safetensors");
        write_st(&a, "text_model.embed.weight", 1.0);
        write_st(&b, "text_projection.weight", 7.0);
        let files = sorted_safetensors(&dir, "prov").unwrap();

        // Lives in the SECOND (non-first) shard — found regardless of scan position, with value intact.
        let t = load_one_tensor_sharded(
            &files,
            "text_projection.weight",
            DType::F32,
            &Device::Cpu,
            "prov",
        )
        .unwrap();
        assert_eq!(t.to_vec1::<f32>().unwrap(), vec![7.0]);

        // The requested dtype is honored on the owning-shard read: the f32 fixture is coerced to F16.
        let t16 = load_one_tensor_sharded(
            &files,
            "text_projection.weight",
            DType::F16,
            &Device::Cpu,
            "prov",
        )
        .unwrap();
        assert_eq!(t16.dtype(), DType::F16);
        assert_eq!(
            t16.to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap(),
            vec![7.0]
        );

        // Absent from every shard → crafted, label-prefixed error naming the key.
        let err = load_one_tensor_sharded(&files, "nope.weight", DType::F32, &Device::Cpu, "prov")
            .unwrap_err();
        assert!(
            matches!(err, CandleError::Msg(m) if m.contains("prov") && m.contains("nope.weight")),
            "expected a crafted missing-tensor error across shards"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn component_vb_missing_subdir_errors() {
        let dir = tmp_dir("component");
        match component_vb(&dir, "transformer", DType::F32, &Device::Cpu, "prov") {
            Err(CandleError::Msg(m)) => {
                assert!(m.contains("prov") && m.contains("transformer/"), "got: {m}")
            }
            _ => panic!("expected a crafted missing-subdir error"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// SceneWorks#1333: a macOS AppleDouble sidecar (`._model.safetensors`) has extension
    /// `safetensors` and sorts *before* the real shard, so an extension-only filter would mmap it
    /// first. It must not appear in the shard list. Sidecars ride into Linux/Windows checkouts via
    /// Mac-authored zips and copied HF caches, so this is not a macOS-only concern.
    #[test]
    fn sorted_safetensors_skips_appledouble_sidecar() {
        let dir = tmp_dir("appledouble");
        write_st(&dir.join("model.safetensors"), "blk.weight", 1.0);
        // Real AppleDouble header: magic 0x00051607, version 0x00020000.
        std::fs::write(
            dir.join("._model.safetensors"),
            [0x00, 0x05, 0x16, 0x07, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00],
        )
        .unwrap();

        let files = sorted_safetensors(&dir, "prov").unwrap();
        assert_eq!(
            files.len(),
            1,
            "sidecar leaked into the shard list: {files:?}"
        );
        assert_eq!(files[0].file_name().unwrap(), "model.safetensors");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A dir holding *only* a sidecar has no shards — the crafted "no .safetensors" error must fire
    /// rather than a raw mmap failure on the sidecar's bogus header.
    #[test]
    fn sorted_safetensors_with_only_a_sidecar_errors() {
        let dir = tmp_dir("only_sidecar");
        std::fs::write(dir.join("._model.safetensors"), [0x00, 0x05, 0x16, 0x07]).unwrap();

        match sorted_safetensors(&dir, "prov") {
            Err(CandleError::Msg(m)) => assert!(m.contains("no .safetensors"), "got: {m}"),
            _ => panic!("expected the crafted no-shards error"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
