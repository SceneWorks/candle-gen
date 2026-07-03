//! A small safetensors key→`Tensor` map for non-`VarBuilder` weight loads — the candle twin of
//! `mlx_gen::weights::Weights` (sc-5491).
//!
//! Most pipelines build the stock UNet/VAE/DiT through a `VarBuilder`, but several loads want a raw
//! key→`Tensor` map (cast to the compute dtype on load) instead: the SDXL IP-Adapter Resampler mixes
//! a learned-`latents` tensor with fused-projection Linears, the ControlNet adds per-residual
//! zero-convs, and the FLUX-family IP-Adapter / PuLID EVA-CLIP towers load the same way. Because that
//! surface is shared across unrelated provider crates (SDXL, FLUX, PuLID), it lives here in the
//! `candle-gen` core crate rather than in any one pipeline crate (F-060, sc-9044) — `candle-gen-sdxl`
//! re-exports it from its old `weights` path for source compatibility.

use std::collections::HashMap;
use std::path::Path;

use candle_core::safetensors::MmapedSafetensors;
use candle_core::{safetensors as cst, DType, Tensor};

use crate::candle_core::Device;
use crate::{CandleError, Result};

/// Coerce a loaded tensor to the compute `dtype`, but only when it is a FLOATING tensor.
///
/// Integer buffers — e.g. the CLIP image encoder's I64 `position_ids` (`h94/IP-Adapter`
/// `models/image_encoder`) — are left as-is: casting an int index buffer to f16 is meaningless, and
/// on CUDA (sm_120) the int→f16 cast kernel isn't compiled, so `to_dtype` there fails with
/// `DriverError(CUDA_ERROR_NOT_FOUND, "named symbol not found")` (sc-5488). The consumers here only
/// `require()` the float weights, so an untouched buffer is simply never read.
fn coerce_float(v: Tensor, dtype: DType) -> Result<Tensor> {
    let is_float = matches!(
        v.dtype(),
        DType::F16 | DType::BF16 | DType::F32 | DType::F64
    );
    if is_float && v.dtype() != dtype {
        Ok(v.to_dtype(dtype)?)
    } else {
        Ok(v)
    }
}

/// A loaded checkpoint weight map (every tensor coerced to the requested compute dtype on load).
pub struct Weights {
    map: HashMap<String, Tensor>,
}

impl Weights {
    /// Load every tensor from a `.safetensors` file onto `device`, casting to `dtype` (f16 in
    /// production, f32 for CPU parity), matching how `mlx-gen-sdxl` casts the IP-Adapter bundle to the
    /// UNet dtype before building.
    pub fn from_file(path: &Path, device: &Device, dtype: DType) -> Result<Self> {
        let raw = cst::load(path, device)?;
        let mut map = HashMap::with_capacity(raw.len());
        for (k, v) in raw {
            map.insert(k, coerce_float(v, dtype)?);
        }
        Ok(Self { map })
    }

    /// Load and MERGE every `.safetensors` file in `files` into one weight map, in the given order
    /// (each tensor coerced to `dtype` exactly like [`from_file`](Self::from_file)).
    ///
    /// A key that appears in more than one shard is a HARD ERROR (naming the key + offending shard) —
    /// not a silent last-file-wins overwrite. In a normal sharded checkpoint every tensor lives in
    /// exactly one shard, so a cross-shard duplicate is abnormal (a mis-sharded / double-listed
    /// checkpoint, or a stray `.safetensors` polluting the dir); letting a stray file shadow the real
    /// weights with no diagnostic is exactly the footgun F-064 (sc-9050) closed for the sibling
    /// `candle-gen-sam3` / `candle-gen-depth` loaders, so `from_files` mirrors that policy.
    ///
    /// Callers pass a deterministically sorted shard list (see
    /// [`crate::loader::sorted_safetensors`]); this is the shard-aware path for snapshots that
    /// ship the checkpoint across multiple `*.safetensors` instead of a single file (F-037, sc-9021).
    pub fn from_files(files: &[impl AsRef<Path>], device: &Device, dtype: DType) -> Result<Self> {
        let mut map = HashMap::new();
        for path in files {
            let path = path.as_ref();
            let raw = cst::load(path, device)?;
            for (k, v) in raw {
                let v = coerce_float(v, dtype)?;
                if map.insert(k.clone(), v).is_some() {
                    return Err(CandleError::Msg(format!(
                        "duplicate tensor key {k:?} while merging shard {}: a checkpoint's tensors \
                         must each live in exactly one .safetensors shard — this snapshot has {k:?} \
                         in more than one file (mis-sharded checkpoint or a stray .safetensors in \
                         the dir)",
                        path.display()
                    )));
                }
            }
        }
        Ok(Self { map })
    }

    /// Load only the tensors whose key starts with one of `prefixes`, via a header-only mmap
    /// (sc-8990 / F-010), casting floats to `dtype` exactly as [`from_file`](Self::from_file).
    ///
    /// The `openai/clip-vit-large-patch14` snapshot ships the *full* `CLIPModel` in one file, so the
    /// image embedder's old `from_file` materialized the entire checkpoint — including the unused
    /// `text_model.*` tower — on the device. Restricting to the needed prefixes (`vision_model.` +
    /// `visual_projection.`) drops that transient. Each retained tensor is byte-identical to what
    /// `from_file` would have produced for the same key.
    pub fn from_file_filtered(
        path: &Path,
        device: &Device,
        dtype: DType,
        prefixes: &[&str],
    ) -> Result<Self> {
        // SAFETY: read-only, process-owned weight file, mapped only for this load and not mutated
        // behind the mapping — the standard candle weight-loading invariant.
        let st = unsafe { MmapedSafetensors::new(path)? };
        let mut map = HashMap::new();
        for (k, _view) in st.tensors() {
            if !prefixes.iter().any(|p| k.starts_with(p)) {
                continue;
            }
            // Load just this one tensor's bytes (header-only mmap), then re-cast floats to the compute
            // dtype via the shared `coerce_float` helper — identical per-tensor handling to
            // `from_file`, so retained values are byte-equal.
            let v = coerce_float(st.load(&k, device)?, dtype)?;
            map.insert(k, v);
        }
        Ok(Self { map })
    }

    /// Shard-aware [`from_file_filtered`](Self::from_file_filtered): apply the same header-only,
    /// prefix-restricted read to EVERY `.safetensors` in `files` and merge the results, so a resharded
    /// snapshot whose prefix-matched tensors are split across shards loads their full union (F-037,
    /// sc-9021) while still never materializing the unmatched towers (sc-8990 / F-010).
    ///
    /// Duplicate-key policy matches [`from_files`](Self::from_files): a key present in more than one
    /// shard is a HARD ERROR, not a silent last-file-wins overwrite (F-064, sc-9050).
    pub fn from_files_filtered(
        files: &[impl AsRef<Path>],
        device: &Device,
        dtype: DType,
        prefixes: &[&str],
    ) -> Result<Self> {
        let mut map = HashMap::new();
        for path in files {
            let path = path.as_ref();
            // SAFETY: read-only, process-owned weight file, mapped only for this load and not mutated
            // behind the mapping — the standard candle weight-loading invariant.
            let st = unsafe { MmapedSafetensors::new(path)? };
            for (k, _view) in st.tensors() {
                if !prefixes.iter().any(|p| k.starts_with(p)) {
                    continue;
                }
                let v = coerce_float(st.load(&k, device)?, dtype)?;
                if map.insert(k.clone(), v).is_some() {
                    return Err(CandleError::Msg(format!(
                        "duplicate tensor key {k:?} while merging shard {}: a checkpoint's tensors \
                         must each live in exactly one .safetensors shard — this snapshot has {k:?} \
                         in more than one file (mis-sharded checkpoint or a stray .safetensors in \
                         the dir)",
                        path.display()
                    )));
                }
            }
        }
        Ok(Self { map })
    }

    /// Fetch a required tensor, erroring (not panicking) when a checkpoint is missing a key.
    pub fn require(&self, key: &str) -> Result<Tensor> {
        self.map
            .get(key)
            .cloned()
            .ok_or_else(|| CandleError::Msg(format!("missing tensor: {key}")))
    }

    /// Whether `key` is present (e.g. the ControlNet's optional `encoder_hid_proj`).
    pub fn contains(&self, key: &str) -> bool {
        self.map.contains_key(key)
    }

    /// Iterate the tensor keys (drives the `ip_adapter.{n}` index discovery in the SDXL
    /// `load_ip_kv_pairs` loader).
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.map.keys()
    }

    /// Build directly from an in-memory map — tests (including cross-crate ones, e.g. the FLUX
    /// IP-Adapter image-encoder fixtures, sc-5872) construct synthetic weights without a file, and a
    /// caller can assemble a checkpoint programmatically.
    pub fn from_map(map: HashMap<String, Tensor>) -> Self {
        Self { map }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn write_st(path: &Path, name: &str, value: f32) {
        let t = Tensor::new(&[value], &Device::Cpu).unwrap();
        let mut m = HashMap::new();
        m.insert(name.to_string(), t);
        candle_core::safetensors::save(&m, path).unwrap();
    }

    fn scratch_dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "candle_gen_weights_test_{tag}_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// `from_files` merges every shard into one map — the UNION of disjoint keys is visible (the
    /// shard-aware path for F-037).
    #[test]
    fn from_files_merges_disjoint_shards() {
        let dir = scratch_dir("merge");
        let a = dir.join("model-00001-of-00002.safetensors");
        let b = dir.join("model-00002-of-00002.safetensors");
        write_st(&a, "a.weight", 1.0);
        write_st(&b, "b.weight", 2.0);
        let w = Weights::from_files(&[a, b], &Device::Cpu, DType::F32).unwrap();
        assert_eq!(
            w.require("a.weight").unwrap().to_vec1::<f32>().unwrap(),
            vec![1.0]
        );
        assert_eq!(
            w.require("b.weight").unwrap().to_vec1::<f32>().unwrap(),
            vec![2.0]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A key present in more than one shard is a HARD ERROR naming the key — not a silent
    /// last-file-wins overwrite (align with the F-064 / sc-9050 duplicate-key policy that the sibling
    /// sam3/depth loaders enforce). A stray or mis-sharded file must not silently shadow real weights.
    #[test]
    fn from_files_errors_on_duplicate() {
        let dir = scratch_dir("dup");
        let first = dir.join("a.safetensors");
        let last = dir.join("b.safetensors");
        write_st(&first, "shared", 10.0);
        write_st(&last, "shared", 20.0);
        match Weights::from_files(&[first, last], &Device::Cpu, DType::F32) {
            Err(CandleError::Msg(m)) => assert!(
                m.contains("duplicate tensor key") && m.contains("shared"),
                "expected a duplicate-key error naming the key, got: {m}"
            ),
            Err(e) => panic!("expected a duplicate-key CandleError::Msg, got: {e}"),
            Ok(_) => panic!("expected a duplicate-key error, but from_files succeeded"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `from_file_filtered` materializes only the prefix-matched keys (byte-identical to `from_file`
    /// for those keys) and drops everything else — e.g. the unused CLIP text tower for the image path.
    #[test]
    fn from_file_filtered_keeps_only_matching_prefixes() {
        let dev = Device::Cpu;
        let dir = std::env::temp_dir().join(format!("sdxl_weights_filter_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("model.safetensors");

        let mut map = HashMap::new();
        map.insert(
            "vision_model.post_layernorm.weight".to_string(),
            Tensor::new(&[1.0f32, 2.0], &dev).unwrap(),
        );
        map.insert(
            "visual_projection.weight".to_string(),
            Tensor::new(&[3.0f32, 4.0], &dev).unwrap(),
        );
        map.insert(
            "text_model.embeddings.token_embedding.weight".to_string(),
            Tensor::new(&[9.0f32, 9.0], &dev).unwrap(),
        );
        cst::save(&map, &file).unwrap();

        let w = Weights::from_file_filtered(
            &file,
            &dev,
            DType::F32,
            &["vision_model.", "visual_projection."],
        )
        .unwrap();

        // Kept: the two vision-side prefixes, values intact.
        assert!(w.contains("vision_model.post_layernorm.weight"));
        assert_eq!(
            w.require("visual_projection.weight")
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![3.0, 4.0]
        );
        // Dropped: the unused text tower is never materialized.
        assert!(!w.contains("text_model.embeddings.token_embedding.weight"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `from_files_filtered` unions the prefix-matched keys across shards (F-037 × F-010) and still
    /// drops the unmatched tower; a cross-shard duplicate is the same hard error as `from_files`.
    #[test]
    fn from_files_filtered_unions_shards_and_drops_unmatched() {
        let dir = scratch_dir("filter_shards");
        let a = dir.join("model-00001-of-00002.safetensors");
        let b = dir.join("model-00002-of-00002.safetensors");
        // Vision tensors split across two shards; a text-tower tensor (co-resident in shard b) that
        // must be dropped. `write_st` writes a single-tensor file, so shard b is saved directly.
        write_st(&a, "vision_model.a.weight", 1.0);
        let mut mb = HashMap::new();
        mb.insert(
            "visual_projection.weight".to_string(),
            Tensor::new(&[2.0f32], &Device::Cpu).unwrap(),
        );
        mb.insert(
            "text_model.dead.weight".to_string(),
            Tensor::new(&[9.0f32], &Device::Cpu).unwrap(),
        );
        cst::save(&mb, &b).unwrap();

        let w = Weights::from_files_filtered(
            &[a, b],
            &Device::Cpu,
            DType::F32,
            &["vision_model.", "visual_projection."],
        )
        .unwrap();
        assert_eq!(
            w.require("vision_model.a.weight")
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![1.0]
        );
        assert!(w.contains("visual_projection.weight"));
        assert!(!w.contains("text_model.dead.weight"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A cross-shard DUPLICATE under a matched prefix is a HARD ERROR naming the key — the same F-064
    /// (sc-9050) policy `from_files` enforces, extended to the filtered/header-only path.
    #[test]
    fn from_files_filtered_errors_on_cross_shard_duplicate() {
        let dir = scratch_dir("filter_dup");
        let a = dir.join("model-00001-of-00002.safetensors");
        let b = dir.join("model-00002-of-00002.safetensors");
        // Same matched-prefix key in BOTH shards → must not silently last-file-wins.
        write_st(&a, "vision_model.shared.weight", 1.0);
        write_st(&b, "vision_model.shared.weight", 2.0);
        match Weights::from_files_filtered(&[a, b], &Device::Cpu, DType::F32, &["vision_model."]) {
            Err(CandleError::Msg(m)) => assert!(
                m.contains("duplicate tensor key") && m.contains("vision_model.shared.weight"),
                "expected a duplicate-key error naming the key, got: {m}"
            ),
            Err(e) => panic!("expected a duplicate-key CandleError::Msg, got: {e}"),
            Ok(_) => panic!("expected a duplicate-key error, but from_files_filtered succeeded"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
