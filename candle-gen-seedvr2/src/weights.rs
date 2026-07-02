//! A tiny key→tensor map (the candle twin of `mlx_gen::weights::Weights`) so the SeedVR2 modules can
//! be ported 1:1 from the MLX reference — looking weights up by dotted key and reading shapes off the
//! tensors, rather than threading explicit shapes through `VarBuilder::get`. The converter
//! ([`crate::convert`]) produces one of these (DiT key-renames / VAE pass-through), then the modules
//! consume it.

use std::collections::HashMap;

use candle_gen::candle_core::{Device, Tensor};
use candle_gen::{CandleError, Result as CResult};

/// An owned `name → Tensor` table.
#[derive(Default)]
pub struct Weights {
    map: HashMap<String, Tensor>,
}

impl Weights {
    pub fn empty() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    /// Load a safetensors file onto `device` (every tensor kept at its on-disk dtype).
    pub fn from_file(path: impl AsRef<std::path::Path>, device: &Device) -> CResult<Self> {
        let map = candle_gen::candle_core::safetensors::load(path.as_ref(), device)?;
        Ok(Self { map })
    }

    pub fn insert(&mut self, key: impl Into<String>, t: Tensor) {
        self.map.insert(key.into(), t);
    }

    pub fn get(&self, key: &str) -> Option<&Tensor> {
        self.map.get(key)
    }

    /// Look up `key` or fail with a clear message (the candle twin of mlx `Weights::require`).
    pub fn require(&self, key: &str) -> CResult<&Tensor> {
        self.map
            .get(key)
            .ok_or_else(|| CandleError::Msg(format!("seedvr2: missing weight tensor `{key}`")))
    }

    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.map.keys().map(String::as_str)
    }

    /// Drain `self` into `(key, tensor)` pairs, consuming the map so each raw tensor can be dropped by
    /// the caller as it is cast (streaming convert+cast keeps peak load memory to ~1× rather than
    /// holding the whole raw fp16 set and the whole cast copy at once — sc-9042/F-058).
    pub fn into_iter_entries(self) -> impl Iterator<Item = (String, Tensor)> {
        self.map.into_iter()
    }
}
