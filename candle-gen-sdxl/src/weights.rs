//! A small safetensors key→`Tensor` map for the IP-Adapter / ControlNet loads (sc-5491) — the candle
//! twin of `mlx_gen::weights::Weights`. The stock SDXL UNet/VAE build through a `VarBuilder`, but the
//! IP-Adapter Resampler mixes a learned-`latents` tensor with fused-projection Linears, and the
//! ControlNet adds the per-residual zero-convs, so a raw key→`Tensor` map (cast to the compute dtype on
//! load) is the natural loader for both.

use std::collections::HashMap;
use std::path::Path;

use candle_core::safetensors::MmapedSafetensors;
use candle_core::{safetensors as cst, DType, Tensor};

use candle_gen::candle_core::Device;
use candle_gen::{CandleError, Result};

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
            // Only re-cast FLOATING tensors to the compute dtype. Integer buffers — e.g. the CLIP
            // image encoder's I64 `position_ids` (`h94/IP-Adapter` `models/image_encoder`) — are left
            // as-is: casting an int index buffer to f16 is meaningless, and on CUDA (sm_120) the
            // int→f16 cast kernel isn't compiled, so `to_dtype` there fails with
            // `DriverError(CUDA_ERROR_NOT_FOUND, "named symbol not found")` (sc-5488). The consumers
            // here only `require()` the float weights, so the untouched buffer is simply never read.
            let is_float = matches!(
                v.dtype(),
                DType::F16 | DType::BF16 | DType::F32 | DType::F64
            );
            let v = if is_float && v.dtype() != dtype {
                v.to_dtype(dtype)?
            } else {
                v
            };
            map.insert(k, v);
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
            // dtype — identical per-tensor handling to `from_file`, so retained values are byte-equal.
            let v = st.load(&k, device)?;
            let is_float = matches!(
                v.dtype(),
                DType::F16 | DType::BF16 | DType::F32 | DType::F64
            );
            let v = if is_float && v.dtype() != dtype {
                v.to_dtype(dtype)?
            } else {
                v
            };
            map.insert(k, v);
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
}
