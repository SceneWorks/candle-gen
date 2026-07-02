//! A small safetensors key→`Tensor` map for the IP-Adapter / ControlNet loads (sc-5491) — the candle
//! twin of `mlx_gen::weights::Weights`. The stock SDXL UNet/VAE build through a `VarBuilder`, but the
//! IP-Adapter Resampler mixes a learned-`latents` tensor with fused-projection Linears, and the
//! ControlNet adds the per-residual zero-convs, so a raw key→`Tensor` map (cast to the compute dtype on
//! load) is the natural loader for both.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{safetensors as cst, DType, Tensor};

use candle_gen::candle_core::Device;
use candle_gen::{CandleError, Result};

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
    /// [`candle_gen::loader::sorted_safetensors`]); this is the shard-aware path for snapshots that
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

    /// Iterate the tensor keys (drives the `ip_adapter.{n}` index discovery in
    /// [`load_ip_kv_pairs`](crate::ip_adapter::load_ip_kv_pairs)).
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
}
