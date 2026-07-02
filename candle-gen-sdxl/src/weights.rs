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
    /// (each tensor coerced to `dtype` exactly like [`from_file`](Self::from_file)). When a key
    /// appears in more than one file the *last* file wins — matching candle's own
    /// `from_mmaped_safetensors` shard semantics, so a `Weights` map and a `VarBuilder` over the same
    /// sorted shard list resolve identical tensors.
    ///
    /// Callers pass a deterministically sorted shard list (see
    /// [`candle_gen::loader::sorted_safetensors`]); this is the shard-aware path for snapshots that
    /// ship the checkpoint across multiple `*.safetensors` instead of a single file (F-037, sc-9021).
    pub fn from_files(files: &[impl AsRef<Path>], device: &Device, dtype: DType) -> Result<Self> {
        let mut map = HashMap::new();
        for path in files {
            let raw = cst::load(path.as_ref(), device)?;
            for (k, v) in raw {
                map.insert(k, coerce_float(v, dtype)?);
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

    /// When a key repeats across files the LAST file wins — matching candle's
    /// `from_mmaped_safetensors`, so a `Weights` map and a `VarBuilder` over the same sorted list
    /// resolve the identical tensor.
    #[test]
    fn from_files_last_shard_wins_on_duplicate_key() {
        let dir = scratch_dir("dup");
        let first = dir.join("a.safetensors");
        let last = dir.join("b.safetensors");
        write_st(&first, "shared", 10.0);
        write_st(&last, "shared", 20.0);
        let w = Weights::from_files(&[first, last], &Device::Cpu, DType::F32).unwrap();
        assert_eq!(
            w.require("shared").unwrap().to_vec1::<f32>().unwrap(),
            vec![20.0]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
