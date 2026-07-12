//! Shared golden-fixture loader for the Bernini planner CPU parity tests. Reads the committed
//! `tests/fixtures/*.safetensors` goldens (dumped from the reference by the MLX lane's
//! `tools/dump_bernini_*_golden.py`, reused byte-for-byte here) without depending on candle's
//! supported-dtype set — the goldens carry `I32`/`I8` tensors candle can't natively load, so we parse
//! the safetensors container by hand and expose typed accessors + a f32-only `VarBuilder`.

#![allow(dead_code)]

use std::collections::HashMap;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::VarBuilder;

struct Entry {
    dtype: String,
    shape: Vec<usize>,
    start: usize,
    end: usize,
}

pub struct Golden {
    data: Vec<u8>,
    data_start: usize,
    entries: HashMap<String, Entry>,
    meta: HashMap<String, String>,
}

impl Golden {
    /// Load a fixture from `tests/fixtures/<name>.safetensors`.
    pub fn load(name: &str) -> Golden {
        let path = format!(
            "{}/tests/fixtures/{}.safetensors",
            env!("CARGO_MANIFEST_DIR"),
            name
        );
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
        let hlen = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        let header: serde_json::Value =
            serde_json::from_slice(&bytes[8..8 + hlen]).expect("parse safetensors header");
        let obj = header.as_object().expect("header object");

        let mut entries = HashMap::new();
        let mut meta = HashMap::new();
        for (k, v) in obj {
            if k == "__metadata__" {
                for (mk, mv) in v.as_object().expect("metadata object") {
                    meta.insert(mk.clone(), mv.as_str().unwrap_or_default().to_string());
                }
                continue;
            }
            let dtype = v["dtype"].as_str().unwrap().to_string();
            let shape: Vec<usize> = v["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as usize)
                .collect();
            let offs = v["data_offsets"].as_array().unwrap();
            entries.insert(
                k.clone(),
                Entry {
                    dtype,
                    shape,
                    start: offs[0].as_u64().unwrap() as usize,
                    end: offs[1].as_u64().unwrap() as usize,
                },
            );
        }
        Golden {
            data_start: 8 + hlen,
            data: bytes,
            entries,
            meta,
        }
    }

    pub fn has(&self, key: &str) -> bool {
        self.entries.contains_key(key)
    }

    pub fn meta(&self, key: &str) -> Option<&str> {
        self.meta.get(key).map(|s| s.as_str())
    }

    pub fn meta_req(&self, key: &str) -> &str {
        self.meta(key)
            .unwrap_or_else(|| panic!("missing metadata {key}"))
    }

    pub fn shape(&self, key: &str) -> Vec<usize> {
        self.entries
            .get(key)
            .unwrap_or_else(|| panic!("missing tensor {key}"))
            .shape
            .clone()
    }

    fn raw(&self, key: &str) -> &[u8] {
        let e = self
            .entries
            .get(key)
            .unwrap_or_else(|| panic!("missing tensor {key}"));
        &self.data[self.data_start + e.start..self.data_start + e.end]
    }

    fn dtype(&self, key: &str) -> &str {
        &self.entries.get(key).unwrap().dtype
    }

    pub fn f32(&self, key: &str) -> Vec<f32> {
        assert_eq!(self.dtype(key), "F32", "{key} is not F32");
        self.raw(key)
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    pub fn i32(&self, key: &str) -> Vec<i32> {
        assert_eq!(self.dtype(key), "I32", "{key} is not I32");
        self.raw(key)
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    /// i64 view of an integer tensor (accepts I32 or I64 storage).
    pub fn i64(&self, key: &str) -> Vec<i64> {
        match self.dtype(key) {
            "I32" => self.i32(key).into_iter().map(|x| x as i64).collect(),
            "I64" => self
                .raw(key)
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            d => panic!("{key} is {d}, not an integer tensor"),
        }
    }

    pub fn i8(&self, key: &str) -> Vec<i8> {
        assert_eq!(self.dtype(key), "I8", "{key} is not I8");
        self.raw(key).iter().map(|&b| b as i8).collect()
    }

    pub fn bools_from_i32(&self, key: &str) -> Vec<bool> {
        self.i32(key).into_iter().map(|x| x != 0).collect()
    }

    /// A candle F32 [shape] tensor for an F32 golden entry.
    pub fn tensor(&self, key: &str, dev: &Device) -> Tensor {
        let v = self.f32(key);
        let shape = self.shape(key);
        Tensor::from_vec(v, shape, dev).unwrap()
    }

    /// A `VarBuilder` over **all F32 tensors** in the fixture (weights + f32 io), rooted at the file
    /// top level. Navigate to the model / connector namespace with `.pp("w.model")` etc.
    pub fn var_builder(&self, dev: &Device) -> VarBuilder<'static> {
        self.var_builder_dtype(dev, DType::F32)
    }

    /// Like [`Golden::var_builder`], but every weight is cast to `dtype`. Used to exercise the
    /// production bf16 weight layout against f32 inputs (sc-11150 — the vision-tower dtype contract).
    pub fn var_builder_dtype(&self, dev: &Device, dtype: DType) -> VarBuilder<'static> {
        let mut map: HashMap<String, Tensor> = HashMap::new();
        for (k, e) in &self.entries {
            if e.dtype == "F32" {
                map.insert(k.clone(), self.tensor(k, dev).to_dtype(dtype).unwrap());
            }
        }
        VarBuilder::from_tensors(map, dtype, dev)
    }
}

/// (peak abs diff, peak-relative `max|Δ|/max|b|`).
pub fn errors(a: &[f32], b: &[f32]) -> (f32, f32) {
    assert_eq!(a.len(), b.len(), "length mismatch");
    let peak = b.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-12);
    let max_diff = a
        .iter()
        .zip(b)
        .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()));
    (max_diff, max_diff / peak)
}

pub fn flat_f32(t: &Tensor) -> Vec<f32> {
    t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
}
