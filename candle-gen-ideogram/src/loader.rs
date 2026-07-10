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
//! **Adapter compose (sc-9412).** The DiT's `insert_override` (the turbo TurboTime LoRA merge) installs
//! dense CPU-side weights that take priority over the mmap. [`linear_detect`] checks the **override
//! first**: a projection the adapter merge targeted resolves to its merged **dense** weight (the merge
//! reconstructs the dense base from the packed parts before folding, [`crate::adapters`]), while an
//! untargeted packed projection stays packed. So the packed base and the dense adapter override compose.
//! [`dequant_packed_base`] is the reconstruction the merge uses to build a mergeable dense base off the
//! packed triple.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_gen::candle_core::safetensors::MmapedSafetensors;
use candle_gen::candle_core::{DType, Device, Error, Result, Tensor};
use candle_gen::candle_nn::{Embedding, Linear};
use candle_gen::quant::{
    dequant_mlx_q4_reference_gs, dequant_mlx_q8_gs, mlx_packed_bits_gs, PackedConfig,
    MLX_GROUP_SIZE,
};

use crate::quant::{QEmbedding, QLinear};

/// An mmaped component-directory of `.safetensors`, loading tensors at a fixed compute dtype, with an
/// optional **override layer** (`overlay`) consulted before the mmap — used to serve LoRA-merged
/// weights ([`crate::adapters`]) without re-reading the base.
pub struct Weights {
    st: MmapedSafetensors,
    overlay: HashMap<String, Tensor>,
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
            overlay: HashMap::new(),
            device: device.clone(),
            dtype,
            packed_group_size: read_packed_group_size(dir),
        })
    }

    /// Load `name` at the component dtype **on the component device** (the override layer wins over the
    /// mmap). The override is the CPU-folded adapter-merge result ([`Self::get_cpu_merge_base`] +
    /// `merge_turbo_lora` run on the CPU), so it MUST be moved to `self.device` here — otherwise a
    /// merged projection stays on the CPU while every other weight is on the GPU, and the forward matmul
    /// raises `device mismatch ... lhs: Cuda, rhs: Cpu` (sc-9654, surfaced on the dense bf16 turbo tier).
    pub fn get(&self, name: &str) -> Result<Tensor> {
        if let Some(t) = self.overlay.get(name) {
            return t.to_dtype(self.dtype)?.to_device(&self.device);
        }
        self.st.load(name, &self.device)?.to_dtype(self.dtype)
    }

    /// Load the **base** (pre-override) tensor forcing f32 — norm weights, or the original weight an
    /// adapter merge folds a delta into.
    pub fn get_f32(&self, name: &str) -> Result<Tensor> {
        self.st.load(name, &self.device)?.to_dtype(DType::F32)
    }

    /// Load `name` at its **native** stored dtype (no cast) on the component device — used for the
    /// packed triple's u32 codes (casting would reinterpret the bit-packed nibbles).
    pub fn get_native(&self, name: &str) -> Result<Tensor> {
        self.st.load(name, &self.device)
    }

    /// Install an override tensor for `name` (served by [`Self::get`] thereafter).
    pub fn insert_override(&mut self, name: impl Into<String>, tensor: Tensor) {
        self.overlay.insert(name.into(), tensor);
    }

    pub fn contains(&self, name: &str) -> bool {
        self.overlay.contains_key(name) || self.st.get(name).is_ok()
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

    /// Whether the override layer holds a (dense, adapter-merged) tensor for `name`. The packed
    /// detectors read this first so an adapter-targeted projection resolves to its merged dense weight
    /// rather than the packed triple (sc-9412 adapter compose).
    fn overlay_has(&self, name: &str) -> bool {
        self.overlay.contains_key(name)
    }

    /// The **dense** CPU base weight for an adapter merge target `weight_key` (`{base}.weight`) — the
    /// adapter-compose seam (sc-9412). On a dense tier this is the on-disk weight loaded onto the CPU.
    /// On a **packed** tier whose `{base}.scales` sibling is present, the weight is u32 codes, so the
    /// dense grid is reconstructed from the packed triple at the tier's group size
    /// ([`dequant_packed_base`], f32) — the mergeable base the LoRA delta folds into. The resulting
    /// merged weight is installed in the override, so [`linear_detect`] then loads it dense (the packed
    /// base stays packed for untargeted projections).
    pub(crate) fn get_cpu_merge_base(&self, weight_key: &str) -> Result<Tensor> {
        if let Some(base) = weight_key.strip_suffix(".weight") {
            let scales_key = format!("{base}.scales");
            if self.st.get(&scales_key).is_ok() {
                let wq = self.st.load(weight_key, &Device::Cpu)?;
                let scales = self
                    .st
                    .load(&scales_key, &Device::Cpu)?
                    .to_dtype(DType::F32)?;
                let biases = self
                    .st
                    .load(&format!("{base}.biases"), &Device::Cpu)?
                    .to_dtype(DType::F32)?;
                return dequant_packed_base(&wq, &scales, &biases, self.group_size());
            }
        }
        self.st.load(weight_key, &Device::Cpu)
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

/// Reconstruct the **dense** f32 grid a packed triple (`{base}.weight` u32 codes + `.scales` +
/// `.biases`) represents, at the tier's `group_size` — the adapter-merge base (sc-9412). The TurboTime
/// LoRA merge folds its delta into this reconstructed dense weight (CPU, f32) and installs the result in
/// the override, so the merged projection loads dense while the untargeted bulk stays packed. Bit-width
/// is inferred from the packed shapes (Q4 → the lossless affine grid; Q8 → its exact grid), mirroring
/// the shared `repack_packed_weight` dispatch.
pub fn dequant_packed_base(
    wq: &Tensor,
    scales: &Tensor,
    biases: &Tensor,
    group_size: usize,
) -> Result<Tensor> {
    let wq_cols = wq.dim(1)?;
    let s_cols = scales.dim(1)?;
    match mlx_packed_bits_gs(wq_cols, s_cols, group_size) {
        4 => dequant_mlx_q4_reference_gs(wq, scales, biases, group_size),
        8 => dequant_mlx_q8_gs(wq, scales, biases, group_size),
        b => Err(Error::Msg(format!(
            "ideogram: unsupported MLX packed bit-width {b} (wq cols {wq_cols}, scales cols {s_cols}, \
             group {group_size})"
        ))),
    }
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

/// **Packed-detecting** [`QLinear`] loader (sc-9412) with adapter-override priority. In order:
///
/// 1. **Override** (`{base}.weight` is adapter-merged): the merge already reconstructed a dense weight
///    (from the packed parts if the tier is packed, [`crate::adapters`]) and installed it, so load that
///    **dense** merged weight — a `Dense` `QLinear`. The packed base composes with the adapter.
/// 2. **Packed** (`{base}.scales` present, no override): build a `Packed` projection straight from the
///    MLX packed triple at the tier's group size — **no dense weight materialized**.
/// 3. **Dense** (otherwise): the exact [`linear`] behavior (`{base}.weight` [+ `{base}.bias`]).
///
/// `base` is the full dotted key prefix (e.g. `layers.0.attention.qkv`), so the `.scales`/`.biases`
/// siblings survive any nesting — build the base string first, then detect.
pub fn linear_detect(w: &Weights, base: &str, bias: bool) -> Result<QLinear> {
    let weight_key = format!("{base}.weight");
    let scales_key = format!("{base}.scales");
    // (1) An adapter-merged dense weight in the override wins — load it dense (adapter compose).
    if w.overlay_has(&weight_key) {
        return Ok(QLinear::dense(linear(w, base, bias)?));
    }
    // (2) A `.scales` sibling → build straight from the packed parts.
    if w.contains(&scales_key) {
        let wq = w.get_native(&weight_key)?;
        let scales = w.get_f32(&scales_key)?;
        let biases = w.get_f32(&format!("{base}.biases"))?;
        let dense_bias = if bias {
            Some(w.get(&format!("{base}.bias"))?)
        } else {
            None
        };
        return QLinear::packed(&wq, &scales, &biases, dense_bias, w.group_size());
    }
    // (3) Dense path unchanged.
    Ok(QLinear::dense(linear(w, base, bias)?))
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
    use candle_gen::candle_nn::Module;
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
        let grid_lin = QLinear::dense(Linear::new(grid, None));
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
        let dense = QLinear::dense(Linear::new(grid, Some(dbias)));
        let x = Tensor::randn(0f32, 1f32, (4, in_dim), &dev)?;
        let cos = cosine(&packed.forward(&x)?, &dense.forward(&x)?);
        assert!(
            cos > 0.99999,
            "packed+bias vs dense-grid+bias cosine {cos:.6}"
        );
        std::fs::remove_dir_all(&dir).ok();
        Ok(())
    }

    /// **Adapter override wins over the packed base (sc-9412 adapter compose).** With a packed
    /// `layers.0.attention.qkv` triple in the component AND an override-installed dense
    /// `...qkv.weight` (the TurboTime-merged weight), `linear_detect` must take the **dense** override
    /// path — not the packed triple — and its forward must reproduce the override weight exactly. This
    /// is the seam that lets the turbo LoRA merge into a packed tier: the adapted projection loads
    /// dense, the rest stays packed.
    #[test]
    fn override_shadows_packed_base_for_adapter_compose() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (128usize, 256usize);
        let (wq, s, b, _grid) = q4_packed(out_dim, in_dim);

        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert("layers.0.attention.qkv.weight".into(), wq);
        map.insert("layers.0.attention.qkv.scales".into(), s);
        map.insert("layers.0.attention.qkv.biases".into(), b);
        let dir = std::env::temp_dir().join(format!("sc9412_override_{}", std::process::id()));
        write_component(&dir, map, None);
        let mut w = Weights::from_dir(&dir, &dev, DType::F32)?;

        // Without an override, the projection loads packed.
        assert!(linear_detect(&w, "layers.0.attention.qkv", false)?.is_packed());

        // Install a distinctive dense "merged" weight; `linear_detect` must load it dense.
        let merged = Tensor::randn(3f32, 0.5f32, (out_dim, in_dim), &dev)?;
        w.insert_override("layers.0.attention.qkv.weight", merged.clone());

        let lin = linear_detect(&w, "layers.0.attention.qkv", false)?;
        assert!(
            !lin.is_packed(),
            "an overridden (adapter-merged) weight must take the dense path, shadowing the packed triple"
        );
        let x = Tensor::randn(0f32, 1f32, (4, in_dim), &dev)?;
        let want = Linear::new(merged, None).forward(&x)?;
        let dev_max = (lin.forward(&x)?.sub(&want)?)
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert_eq!(
            dev_max, 0.0,
            "override forward must equal the merged dense weight"
        );
        std::fs::remove_dir_all(&dir).ok();
        Ok(())
    }

    /// **`get_cpu_merge_base` reconstructs the dense grid from the packed triple (sc-9412).** The
    /// TurboTime LoRA merge folds its delta into this reconstructed base; on a packed tier the base must
    /// be the exact affine grid the pack represents (f32), NOT the u32 codes. A dense tier returns the
    /// on-disk weight unchanged.
    #[test]
    fn get_cpu_merge_base_dequantizes_packed_and_passes_dense() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (128usize, 256usize);
        let (wq, s, b, grid) = q4_packed(out_dim, in_dim);

        // Packed tier: base is the reconstructed dense grid.
        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert("layers.0.attention.qkv.weight".into(), wq);
        map.insert("layers.0.attention.qkv.scales".into(), s);
        map.insert("layers.0.attention.qkv.biases".into(), b);
        let dir = std::env::temp_dir().join(format!("sc9412_base_{}", std::process::id()));
        write_component(&dir, map, None);
        let w = Weights::from_dir(&dir, &dev, DType::F32)?;
        let base = w.get_cpu_merge_base("layers.0.attention.qkv.weight")?;
        assert_eq!(base.dims(), &[out_dim, in_dim], "reconstructed dense shape");
        assert!(
            cosine(&base, &grid) > 0.99999,
            "reconstructed base must equal the affine grid"
        );
        std::fs::remove_dir_all(&dir).ok();

        // Dense tier: base is the on-disk weight (identity round-trip).
        let dense_w = Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev)?;
        let mut dmap: HashMap<String, Tensor> = HashMap::new();
        dmap.insert("layers.0.attention.qkv.weight".into(), dense_w.clone());
        let ddir = std::env::temp_dir().join(format!("sc9412_base_dense_{}", std::process::id()));
        write_component(&ddir, dmap, None);
        let dw = Weights::from_dir(&ddir, &dev, DType::F32)?;
        let dbase = dw.get_cpu_merge_base("layers.0.attention.qkv.weight")?;
        let dev_max = (dbase.sub(&dense_w)?)
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert_eq!(dev_max, 0.0, "dense tier base is the on-disk weight");
        std::fs::remove_dir_all(&ddir).ok();
        Ok(())
    }
}
