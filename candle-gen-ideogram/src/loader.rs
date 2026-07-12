//! Weight loading for the Ideogram 4 DiT — a thin shape-inferring wrapper over candle's
//! [`MmapedSafetensors`], mirroring `mlx-gen-ideogram`'s `Weights`/`lin` interface so the
//! `transformer` port stays a near-1:1 translation. [`linear`] builds a [`Linear`] from the actual
//! `{base}.weight` (and optional `{base}.bias`) tensor shapes, so dims that aren't in the public
//! config (e.g. the `t_embedding` MLP hidden width) need no hardcoding.
//!
//! **Packed-tier detect (sc-9412).** The hosted `SceneWorks/ideogram-4-mlx` (q4/q8) snapshot stores
//! each quantized projection as the MLX packed triple `{base}.weight` (u32 codes) + `{base}.scales` +
//! `{base}.biases` (bf16). Unlike krea/boogu, the ideogram converter emits **no** `quantization` block
//! in `config.json`, so — exactly like the shared VarBuilder `candle_gen::quant::lin` — detection keys
//! purely on the presence of the `{base}.scales` **sibling**, and the group size defaults to the MLX
//! default ([`candle_gen::quant::MLX_GROUP_SIZE`], 64 — the value the hosted tier packs at). Should a
//! future tier ship the `quantization.group_size` block (sc-9474), [`Weights::from_dir`] reads it and
//! threads it through. [`linear_detect`] / [`embedding_detect`] build the quantized module straight
//! from the packed parts through the shared group-size-aware loaders (no dense staging — see
//! [`crate::quant`]). Absent the `.scales` sibling, the dense path is unchanged.
//!
//! **Turbo TurboTime LoRA — forward-time additive, both tiers (sc-11104).** The LoRA never mutates a
//! base weight. Each projection loads its base straight from the mmap (dense or MLX-packed), and the
//! turbo LoRA is attached *after* build as a forward-time additive residual on the shared
//! [`crate::quant::QLinear`] (`y = base(x) + Σ scale·((x·A)·B)`,
//! [`crate::adapters::install_turbo_lora_additive`]). So the base — dense bf16 or packed q4/q8 — stays a
//! clean, disk-backed mmap that the offload/eviction machinery can drop and restore cheaply, instead of
//! an in-memory-modified merged weight. This retired both the old override/fold layer (a merged dense
//! weight installed over the mmap) and the packed **dequant-fold** (reconstruct the dense grid, fold,
//! install a dense override).

use std::path::{Path, PathBuf};

use candle_gen::candle_core::safetensors::MmapedSafetensors;
use candle_gen::candle_core::{DType, Device, Error, Result, Tensor};
use candle_gen::candle_nn::{Embedding, Linear};
use candle_gen::quant::{PackedConfig, MLX_GROUP_SIZE};

use crate::quant::{QEmbedding, QLinear};

/// An mmaped component-directory of `.safetensors`, loading tensors at a fixed compute dtype. The DiT's
/// weights load straight from the mmap (no override layer): the TurboTime LoRA never mutates a base
/// weight — it rides as a forward-time additive residual pushed onto the built projection
/// ([`crate::adapters::install_turbo_lora_additive`]), so a base (dense or packed) stays a clean,
/// disk-backed mmap that the offload/eviction machinery can drop and restore cheaply (sc-11104).
pub struct Weights {
    st: MmapedSafetensors,
    device: Device,
    dtype: DType,
    /// The MLX group size the packed shapes can't disambiguate. `Some` when a `quantization` block is
    /// present in `config.json` (a future tier that emits it); `None` when absent — the ideogram
    /// converter emits no such block, so the group size defaults to [`MLX_GROUP_SIZE`] on the packed
    /// path. Packed-detect itself keys on the `.scales` sibling, not on this being `Some`.
    packed_group_size: Option<usize>,
}

impl Weights {
    /// mmap every `*.safetensors` in `dir` (sorted; later files win on name collision), reading the
    /// component `config.json`'s `quantization.group_size` (if any) for the packed-tier path.
    pub fn from_dir(dir: &Path, device: &Device, dtype: DType) -> Result<Self> {
        let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
            .map_err(|e| Error::Msg(format!("ideogram: read {}: {e}", dir.display())))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .filter(|p| !candle_gen::gen_core::weightsmeta::is_hidden_file(p))
            .collect();
        files.sort();
        if files.is_empty() {
            return Err(Error::Msg(format!(
                "ideogram: no .safetensors in {}",
                dir.display()
            )));
        }
        // SAFETY: read-only mmap of weight files; the standard candle loading path.
        let st = unsafe { MmapedSafetensors::multi(&files)? };
        Ok(Self {
            st,
            device: device.clone(),
            dtype,
            packed_group_size: read_packed_group_size(dir),
        })
    }

    /// Load `name` at the component dtype on the component device (straight from the mmap).
    pub fn get(&self, name: &str) -> Result<Tensor> {
        self.st.load(name, &self.device)?.to_dtype(self.dtype)
    }

    /// Load `name` forcing f32 — norm weights and the packed triple's `.scales`/`.biases`.
    pub fn get_f32(&self, name: &str) -> Result<Tensor> {
        self.st.load(name, &self.device)?.to_dtype(DType::F32)
    }

    /// Load `name` at its **native** stored dtype (no cast) on the component device — used for the
    /// packed triple's u32 codes (casting would reinterpret the bit-packed nibbles).
    pub fn get_native(&self, name: &str) -> Result<Tensor> {
        self.st.load(name, &self.device)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.st.get(name).is_ok()
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// The MLX group size for the packed path: the `config.json` `quantization.group_size` if present,
    /// else the MLX default [`MLX_GROUP_SIZE`] (64 — the value the hosted tier packs at; the converter
    /// emits no block).
    fn group_size(&self) -> usize {
        self.packed_group_size.unwrap_or(MLX_GROUP_SIZE)
    }
}

/// Read `{dir}/config.json`'s `quantization.group_size` — `None` when the block is absent (the
/// ideogram converter emits no such block, so the packed path defaults to [`MLX_GROUP_SIZE`]). Reuses
/// the shared [`PackedConfig`] parse so a future tier that ships the block is honoured verbatim
/// (sc-9474).
fn read_packed_group_size(dir: &Path) -> Option<usize> {
    let text = std::fs::read_to_string(dir.join("config.json")).ok()?;
    let v: serde_json::Value = serde_json::from_str(&text).ok()?;
    PackedConfig::from_config(&v).map(|c| c.group_size as usize)
}

/// Build a [`Linear`] from `{base}.weight` (+ `{base}.bias` when `bias`), inferring in/out dims from
/// the stored tensor shape (`[out, in]`, PyTorch/HF convention).
pub fn linear(w: &Weights, base: &str, bias: bool) -> Result<Linear> {
    let weight = w.get(&format!("{base}.weight"))?;
    let bias = if bias {
        Some(w.get(&format!("{base}.bias"))?)
    } else {
        None
    };
    Ok(Linear::new(weight, bias))
}

/// Wrap a loaded dense [`Linear`] as the shared [`QLinear`] (`AdaptLinear`), recovering the logical
/// `[out, in]` from the weight shape so a residual installer can shape-check without re-reading it.
fn dense_adapt(l: Linear) -> QLinear {
    let dims = l.weight().dims();
    let (out, in_) = (dims[0], dims[1]);
    QLinear::from_dense(l, in_, out)
}

/// **Packed-detecting** [`QLinear`] loader (sc-9412, additive route sc-11104). In order:
///
/// 1. **Packed** (`{base}.scales` present): build the base straight from the MLX packed triple at the
///    tier's group size (**no dense weight materialized**) — the base stays quantized and the turbo LoRA
///    rides as a forward-time additive residual pushed post-load
///    ([`crate::adapters::install_turbo_lora_additive`]).
/// 2. **Dense** (otherwise): the exact [`linear`] behavior (`{base}.weight` [+ `{base}.bias`]).
///
/// Every arm yields the shared [`QLinear`] with **no residual attached** (the turbo LoRA never mutates a
/// base weight on either tier — it attaches additively), so before any install its forward is
/// byte-identical to the bare base. `base` is the full dotted key prefix (e.g.
/// `layers.0.attention.qkv`), so the `.scales`/`.biases` siblings survive any nesting — build the base
/// string first, then detect.
pub fn linear_detect(w: &Weights, base: &str, bias: bool) -> Result<QLinear> {
    let weight_key = format!("{base}.weight");
    let scales_key = format!("{base}.scales");
    // (1) A `.scales` sibling → build the packed base straight from the MLX triple (kept quantized).
    if w.contains(&scales_key) {
        let wq = w.get_native(&weight_key)?;
        let scales = w.get_f32(&scales_key)?;
        let biases = w.get_f32(&format!("{base}.biases"))?;
        let dense_bias = if bias {
            Some(w.get(&format!("{base}.bias"))?)
        } else {
            None
        };
        // Recover the logical `[out, in]` from the scales shape `[out, in/group]` — no dense
        // materialization, and no dependence on the packed-code column count (Q4 vs Q8 differ).
        let out = scales.dim(0)?;
        let in_ = scales.dim(1)? * w.group_size();
        let packed = candle_gen::quant::QLinear::from_packed_gs(
            &wq,
            &scales,
            &biases,
            dense_bias,
            w.group_size(),
            w.device(),
        )?;
        return Ok(QLinear::from_packed(packed, in_, out));
    }
    // (3) Dense path unchanged.
    Ok(dense_adapt(linear(w, base, bias)?))
}

/// **Packed-detecting** [`QEmbedding`] loader (sc-9412): packed straight from the MLX triple when
/// `{base}.scales` is present (dequantized to the component dtype — dtype parity with the dense table),
/// else a dense [`Embedding`] from `{base}.weight` (`hidden` inferred from the stored `[vocab, hidden]`
/// shape). The DiT's `embed_image_indicator` table stays **dense** in the hosted q4/q8 tiers, so today
/// this takes the dense arm; the packed arm is the future-proof path (and guards against a silent dense
/// read of u32 codes should a tier ever pack the table).
pub fn embedding_detect(w: &Weights, base: &str) -> Result<QEmbedding> {
    let scales_key = format!("{base}.scales");
    if w.contains(&scales_key) {
        let wq = w.get_native(&format!("{base}.weight"))?;
        let scales = w.get_f32(&scales_key)?;
        let biases = w.get_f32(&format!("{base}.biases"))?;
        return QEmbedding::packed(&wq, &scales, &biases, w.dtype(), w.group_size());
    }
    let weight = w.get(&format!("{base}.weight"))?;
    let hidden = weight.dim(1)?;
    Ok(QEmbedding::dense(Embedding::new(weight, hidden)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::safetensors;
    use std::collections::HashMap;

    /// The Ideogram MLX tier's group size (64 — the MLX default; the converter emits no config block).
    const G: usize = 64;

    /// Build an MLX group-64 Q4 packed triple for an `[out, in]` weight — `(wq u32, scales, biases,
    /// affine grid)`. The affine grid is the exact dense weight the pack represents.
    fn q4_packed(out_dim: usize, in_dim: usize) -> (Tensor, Tensor, Tensor, Tensor) {
        let dev = Device::Cpu;
        let codes: Vec<u8> = (0..out_dim * in_dim)
            .map(|i| ((i * 7 + i / 13) % 16) as u8)
            .collect();
        let groups = out_dim * in_dim / G;
        let scales: Vec<f32> = (0..groups).map(|g| 0.0625 * (g as f32 + 1.0)).collect();
        let biases: Vec<f32> = (0..groups).map(|g| -0.5 - 0.25 * g as f32).collect();
        let gpr = in_dim / G;
        let grid: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| {
                let (row, col) = (i / in_dim, i % in_dim);
                let g = row * gpr + col / G;
                scales[g] * codes[i] as f32 + biases[g]
            })
            .collect();
        let words: Vec<u32> = codes
            .chunks_exact(8)
            .map(|c| {
                c.iter()
                    .enumerate()
                    .fold(0u32, |acc, (i, &q)| acc | ((q as u32 & 0xF) << (4 * i)))
            })
            .collect();
        (
            Tensor::from_vec(words, (out_dim, in_dim / 8), &dev).unwrap(),
            Tensor::from_vec(scales, (out_dim, gpr), &dev).unwrap(),
            Tensor::from_vec(biases, (out_dim, gpr), &dev).unwrap(),
            Tensor::from_vec(grid, (out_dim, in_dim), &dev).unwrap(),
        )
    }

    /// Write a component dir. `config`: `None` writes no `config.json` (the real ideogram tier —
    /// packed-detect must still fire off the `.scales` sibling); `Some(gs)` writes a
    /// `quantization.group_size` block (a hypothetical future tier).
    fn write_component(dir: &Path, tensors: HashMap<String, Tensor>, config: Option<usize>) {
        std::fs::create_dir_all(dir).unwrap();
        safetensors::save(&tensors, dir.join("model.safetensors")).unwrap();
        if let Some(gs) = config {
            let cfg = serde_json::json!({ "quantization": { "bits": 4, "group_size": gs } });
            std::fs::write(dir.join("config.json"), cfg.to_string()).unwrap();
        }
    }

    fn cosine(a: &Tensor, b: &Tensor) -> f64 {
        let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (x, y) in a.iter().zip(&b) {
            dot += (*x as f64) * (*y as f64);
            na += (*x as f64) * (*x as f64);
            nb += (*y as f64) * (*y as f64);
        }
        dot / (na.sqrt() * nb.sqrt() + 1e-12)
    }

    /// **Packed-detect fires on the ideogram key layout WITHOUT a config block (sc-9412).** The real
    /// ideogram tier ships no `quantization` block, so packed-detect must key purely on the `.scales`
    /// sibling: a packed `layers.0.attention.qkv` (group-64 triple, no config.json) must `linear_detect`
    /// to a `Packed` projection, while a dense sibling (`layers.0.attention.norm_q`, no `.scales`) stays
    /// `Dense`. The packed forward reproduces the affine grid (proving the group-64 default is applied,
    /// not a silent dense fallback over u32 codes).
    #[test]
    fn linear_detect_fires_without_config_block_and_leaves_dense_unchanged() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (128usize, 256usize);
        let (wq, s, b, grid) = q4_packed(out_dim, in_dim);
        let dense_w = Tensor::randn(0f32, 1f32, (256usize, 256usize), &dev)?;

        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert("layers.0.attention.qkv.weight".into(), wq);
        map.insert("layers.0.attention.qkv.scales".into(), s);
        map.insert("layers.0.attention.qkv.biases".into(), b);
        map.insert("layers.0.attention.norm_q.weight".into(), dense_w);

        let dir = std::env::temp_dir().join(format!("sc9412_detect_{}", std::process::id()));
        write_component(&dir, map, None); // NO config.json — real ideogram tier
        let w = Weights::from_dir(&dir, &dev, DType::F32)?;
        assert!(
            w.packed_group_size.is_none(),
            "no config block ⇒ group size defaults to the MLX default"
        );

        let packed = linear_detect(&w, "layers.0.attention.qkv", false)?;
        assert!(
            packed.is_packed(),
            "`.scales` sibling (no config block) ⇒ packed load, not a silent dense fallback"
        );
        let dense = linear_detect(&w, "layers.0.attention.norm_q", false)?;
        assert!(!dense.is_packed(), "no `.scales` ⇒ dense path unchanged");

        // The packed forward reproduces the affine grid (group-64 default + dequant-on-forward).
        let grid_lin = dense_adapt(Linear::new(grid, None));
        let x = Tensor::randn(0f32, 1f32, (4, in_dim), &dev)?;
        let cos = cosine(&packed.forward(&x)?, &grid_lin.forward(&x)?);
        assert!(cos > 0.99999, "group-64 packed vs grid cosine {cos:.6}");

        std::fs::remove_dir_all(&dir).ok();
        Ok(())
    }

    /// A **dense bf16 component** (no `.scales` anywhere) takes the dense path — every `linear_detect`
    /// stays `Dense`. The one-crate-serves-both contract.
    #[test]
    fn dense_component_takes_dense_path() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (64usize, 128usize);
        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert(
            "input_proj.weight".into(),
            Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev)?,
        );
        map.insert(
            "input_proj.bias".into(),
            Tensor::zeros((out_dim,), DType::F32, &dev)?,
        );
        let dir = std::env::temp_dir().join(format!("sc9412_dense_{}", std::process::id()));
        write_component(&dir, map, None);

        let w = Weights::from_dir(&dir, &dev, DType::F32)?;
        assert!(!linear_detect(&w, "input_proj", true)?.is_packed());
        std::fs::remove_dir_all(&dir).ok();
        Ok(())
    }

    /// A packed projection **with a dense `.bias`** (the ideogram `input_proj`/`adaln_proj`/`t_embedding`
    /// /`final_layer` biased projections) loads packed and its forward includes the bias — the packed
    /// path's own `.biases` (the affine group biases) is distinct from the projection's dense `.bias`.
    #[test]
    fn packed_linear_with_dense_bias() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (128usize, 256usize);
        let (wq, s, b, grid) = q4_packed(out_dim, in_dim);
        let dbias = Tensor::randn(0f32, 1f32, (out_dim,), &dev)?;

        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert("input_proj.weight".into(), wq);
        map.insert("input_proj.scales".into(), s);
        map.insert("input_proj.biases".into(), b);
        map.insert("input_proj.bias".into(), dbias.clone());
        let dir = std::env::temp_dir().join(format!("sc9412_bias_{}", std::process::id()));
        write_component(&dir, map, None);
        let w = Weights::from_dir(&dir, &dev, DType::F32)?;

        let packed = linear_detect(&w, "input_proj", true)?;
        assert!(packed.is_packed());
        let dense = dense_adapt(Linear::new(grid, Some(dbias)));
        let x = Tensor::randn(0f32, 1f32, (4, in_dim), &dev)?;
        let cos = cosine(&packed.forward(&x)?, &dense.forward(&x)?);
        assert!(
            cos > 0.99999,
            "packed+bias vs dense-grid+bias cosine {cos:.6}"
        );
        std::fs::remove_dir_all(&dir).ok();
        Ok(())
    }
}
