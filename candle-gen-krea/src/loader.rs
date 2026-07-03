//! Weight loading for the Krea 2 DiT + Qwen3-VL-4B condition encoder — a thin shape-inferring wrapper
//! over candle's [`MmapedSafetensors`], mirroring `candle-gen-boogu`/`candle-gen-ideogram`'s `Weights`
//! interface so the port stays a near-1:1 translation of `mlx-gen-krea` (whose `Weights::from_dir`
//! loads the identity-keyed diffusers checkpoint directly). [`linear`] builds a [`Linear`] from the
//! actual `{base}.weight` (+ optional `{base}.bias`) tensor shapes.
//!
//! **Packed-tier detect (sc-9411).** When a component dir is an MLX-packed q4/q8 snapshot
//! (`SceneWorks/krea-2-turbo-mlx`, group size 64), each quantized projection is stored as the triple
//! `{base}.weight` (u32 codes) + `{base}.scales` + `{base}.biases`, and the component `config.json`
//! carries a `quantization: { bits, group_size }` block ([`candle_gen::quant::PackedConfig`]).
//! [`Weights::from_dir`] reads that block; [`linear_detect`] / [`embedding_detect`] then packed-**detect**
//! the `.scales` sibling and build the quantized module straight from the packed parts through the shared
//! group-size-aware loaders (no dense staging — see [`crate::quant`]). Absent the block / `.scales`, the
//! dense path is unchanged.
//!
//! **Adapter compose (sc-9411).** The DiT's `set_overlay` (adapter merge, sc-7836) installs dense
//! CPU-side weights that take priority over the mmap. [`linear_detect`] checks the **overlay first**: a
//! projection the adapter merge targeted resolves to its merged **dense** weight (the merge
//! reconstructs the dense base from the packed parts before folding, [`crate::adapters`]), while an
//! untargeted packed projection stays packed. So the packed base and the dense adapter overlay compose.
//! [`dequant_packed_base`] is the reconstruction the merge uses to build a mergeable dense base off the
//! packed triple.

use std::collections::HashMap;
use std::path::Path;

use candle_gen::candle_core::safetensors::MmapedSafetensors;
use candle_gen::candle_core::{DType, Device, Result, Tensor};
use candle_gen::candle_nn::{Embedding, Linear};
use candle_gen::quant::{
    dequant_mlx_q4_reference_gs, dequant_mlx_q8_gs, mlx_packed_bits_gs, PackedConfig,
};

use crate::quant::{QEmbedding, QLinear};

/// An mmaped component-directory of `.safetensors`, loading tensors at a fixed compute dtype.
///
/// An optional in-memory `overlay` (installed by [`set_overlay`](Weights::set_overlay)) takes priority
/// over the mmap for the keys it holds — the inference-side LoRA/LoKr adapter merge (sc-7836) folds its
/// deltas into the targeted dense weights on the CPU in f32, then installs them here so
/// [`crate::transformer::Krea2Transformer::load`] reads the **merged** weight without re-mmapping or
/// touching the untargeted bulk of the model. Overlay tensors are stored CPU-side (where the merge runs)
/// and moved to `device` / cast to the requested dtype on read, exactly like the mmap path.
pub struct Weights {
    st: MmapedSafetensors,
    device: Device,
    dtype: DType,
    overlay: HashMap<String, Tensor>,
    /// The component's `quantization` manifest, `Some` for a packed q4/q8 tier (carries the group size
    /// the packed shapes can't disambiguate), `None` for a dense bf16 tier.
    packed: Option<PackedConfig>,
}

impl Weights {
    /// mmap every `*.safetensors` in `dir` (sorted; later files win on name collision), reading the
    /// component `config.json`'s `quantization` block (if any) for the packed-tier path.
    pub fn from_dir(dir: &Path, device: &Device, dtype: DType) -> Result<Self> {
        let files = candle_gen::sorted_safetensors(dir, "krea")
            .map_err(|e| candle_gen::candle_core::Error::Msg(e.to_string()))?;
        // SAFETY: read-only mmap of weight files; the standard candle loading path.
        let st = unsafe { MmapedSafetensors::multi(&files)? };
        Ok(Self {
            st,
            device: device.clone(),
            dtype,
            overlay: HashMap::new(),
            packed: read_packed_config(dir)?,
        })
    }

    /// mmap a single `.safetensors` file (used by the committed parity fixtures). Dense-only (no
    /// packed config), so the packed path is never taken for a single-file fixture.
    pub fn from_file(path: &Path, device: &Device, dtype: DType) -> Result<Self> {
        // SAFETY: read-only mmap of a weight file; the standard candle loading path.
        let st = unsafe { MmapedSafetensors::new(path)? };
        Ok(Self {
            st,
            device: device.clone(),
            dtype,
            overlay: HashMap::new(),
            packed: None,
        })
    }

    /// Load `name` at the component dtype — from the [`overlay`](Weights::set_overlay) if present
    /// (adapter-merged weight), else the mmap.
    pub fn get(&self, name: &str) -> Result<Tensor> {
        if let Some(t) = self.overlay.get(name) {
            return t.to_device(&self.device)?.to_dtype(self.dtype);
        }
        self.st.load(name, &self.device)?.to_dtype(self.dtype)
    }

    /// Load `name` preserving its on-disk dtype (e.g. int `input_ids` in a parity fixture). The overlay
    /// only ever holds merged DiT weights (never raw-dtype tensors), so this stays the mmap path.
    pub fn get_raw(&self, name: &str) -> Result<Tensor> {
        self.st.load(name, &self.device)
    }

    /// Load `name` at its **native** stored dtype (no cast) on the component device — used for the
    /// packed triple's u32 codes (casting would reinterpret the bit-packed nibbles). The overlay only
    /// holds merged dense DiT weights (never u32 codes), so this stays the mmap path.
    pub fn get_native(&self, name: &str) -> Result<Tensor> {
        self.st.load(name, &self.device)
    }

    /// Load `name` forcing f32 (the `+1` norm weights and other precision-sensitive scalars) — from the
    /// overlay if present, else the mmap.
    pub fn get_f32(&self, name: &str) -> Result<Tensor> {
        if let Some(t) = self.overlay.get(name) {
            return t.to_device(&self.device)?.to_dtype(DType::F32);
        }
        self.st.load(name, &self.device)?.to_dtype(DType::F32)
    }

    /// Load `name` onto the **CPU** at its on-disk dtype. Used by the inference-side adapter merge
    /// ([`crate::adapters`]), which reconstructs LoRA/LoKr deltas on the CPU (matching the CPU-loaded
    /// adapter factors) and folds them into the base weight before installing the [`overlay`](Weights::set_overlay).
    pub(crate) fn get_cpu(&self, name: &str) -> Result<Tensor> {
        self.st.load(name, &Device::Cpu)
    }

    /// Install an in-memory `overlay` of (CPU-resident) tensors that take priority over the mmap for the
    /// keys they cover — the adapter-merged dense weights (sc-7836). Replaces any prior overlay.
    pub(crate) fn set_overlay(&mut self, overlay: HashMap<String, Tensor>) {
        self.overlay = overlay;
    }

    pub fn contains(&self, name: &str) -> bool {
        self.overlay.contains_key(name) || self.st.get(name).is_ok()
    }

    /// All tensor keys in the component (for architecture validation).
    pub fn keys(&self) -> Vec<String> {
        self.st.tensors().into_iter().map(|(k, _)| k).collect()
    }

    /// On-disk shape of `name` (for architecture validation), or `None` if absent. The overlay never
    /// changes a weight's shape, so the mmap is authoritative.
    pub fn shape(&self, name: &str) -> Option<Vec<usize>> {
        self.st.get(name).ok().map(|v| v.shape().to_vec())
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// The MLX `quantization` block when this component is a packed q4/q8 tier, else `None`.
    pub fn packed(&self) -> Option<PackedConfig> {
        self.packed
    }

    /// Whether the [`overlay`](Weights::set_overlay) holds a (dense, adapter-merged) tensor for `name`.
    /// The packed detectors read this first so an adapter-targeted projection resolves to its merged
    /// dense weight rather than the packed triple (sc-9411 adapter compose).
    fn overlay_has(&self, name: &str) -> bool {
        self.overlay.contains_key(name)
    }

    /// The **dense** CPU base weight for an adapter merge target `weight_key` (`{base}.weight`) — the
    /// adapter-compose seam (sc-9411). On a dense tier this is the on-disk weight loaded onto the CPU
    /// (exactly [`Self::get_cpu`]). On a **packed** tier whose `{base}.scales` sibling is present, the
    /// weight is u32 codes, so the dense grid is reconstructed from the packed triple at the tier's
    /// group size ([`dequant_packed_base`], f32) — the mergeable base the LoRA/LoKr delta folds into.
    /// The resulting merged weight is installed in the overlay, so [`linear_detect`] then loads it
    /// dense (the packed base stays packed for untargeted projections).
    pub(crate) fn get_cpu_merge_base(&self, weight_key: &str) -> Result<Tensor> {
        if let Some(base) = weight_key.strip_suffix(".weight") {
            let scales_key = format!("{base}.scales");
            if let (Some(cfg), true) = (self.packed, self.st.get(&scales_key).is_ok()) {
                let wq = self.st.load(weight_key, &Device::Cpu)?;
                let scales = self
                    .st
                    .load(&scales_key, &Device::Cpu)?
                    .to_dtype(DType::F32)?;
                let biases = self
                    .st
                    .load(&format!("{base}.biases"), &Device::Cpu)?
                    .to_dtype(DType::F32)?;
                return dequant_packed_base(&wq, &scales, &biases, cfg.group_size as usize);
            }
        }
        self.get_cpu(weight_key)
    }
}

/// Read `{dir}/config.json`'s `quantization` block: `Ok(Some(cfg))` for a packed tier, `Ok(None)` for
/// a dense tier (a genuinely-absent `config.json` — a single-file fixture — still loads dense).
///
/// A **present-but-corrupt** `config.json` (I/O error or malformed JSON — e.g. a partial download)
/// returns an `Err` naming the file rather than silently swallowing to the dense path, so a damaged
/// packed snapshot surfaces instead of loading the wrong (dense) tier with no diagnostic (sc-9426,
/// F-073 sibling — the `component_is_packed` twin in flux2). Mirrors boogu's `read_packed_config`
/// (sc-9410) and z-image's `component_is_packed` (sc-9408).
fn read_packed_config(dir: &Path) -> Result<Option<PackedConfig>> {
    let path = dir.join("config.json");
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        // No config.json at all → legitimate dense / single-file fixture tier.
        Err(ref e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        // Present but unreadable (permissions, partial download) → surface, don't swallow.
        Err(e) => {
            return Err(candle_gen::candle_core::Error::Msg(format!(
                "krea: read {}: {e}",
                path.display()
            )))
        }
    };
    // Present but malformed JSON → corrupt snapshot, error rather than fall to dense.
    let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| {
        candle_gen::candle_core::Error::Msg(format!(
            "krea: parse {} (corrupt snapshot?): {e}",
            path.display()
        ))
    })?;
    Ok(PackedConfig::from_config(&v))
}

/// Reconstruct the **dense** f32 grid a packed triple (`{base}.weight` u32 codes + `.scales` +
/// `.biases`) represents, at the tier's `group_size` — the adapter-merge base (sc-9411). The
/// `krea_2_raw` adapter merge folds its delta into this reconstructed dense weight (CPU, f32, matching
/// the trainer's math) and installs the result in the overlay, so the merged projection loads dense
/// while the untargeted bulk stays packed. Bit-width is inferred from the packed shapes (Q4 → the
/// lossless affine grid; Q8 → its exact grid), mirroring the shared `repack_packed_weight` dispatch.
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
        b => Err(candle_gen::candle_core::Error::Msg(format!(
            "krea: unsupported MLX packed bit-width {b} (wq cols {wq_cols}, scales cols {s_cols}, \
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

/// **Packed-detecting** [`QLinear`] loader (sc-9411) with adapter-overlay priority. In order:
///
/// 1. **Overlay** (`{base}.weight` is adapter-merged): the merge already reconstructed a dense weight
///    (from the packed parts if the tier is packed, [`crate::adapters`]) and installed it, so load
///    that **dense** merged weight — a `Dense` `QLinear`. The packed base composes with the adapter.
/// 2. **Packed** (a packed tier + `{base}.scales` present, no overlay): build a `Packed` projection
///    straight from the MLX packed triple at the tier's group size — **no dense weight materialized**.
/// 3. **Dense** (otherwise): the exact [`linear`] behavior (`{base}.weight` [+ `{base}.bias`]).
///
/// `base` is the full dotted key prefix (e.g. `attn.to_out.0`), so the `.scales`/`.biases` siblings
/// survive any `to_out.0`-style nesting — build the base string first, then detect (the key-remap trap
/// the `linear_detect_fires_on_to_out_remap` test pins on the real Krea `to_out.0` layout).
pub fn linear_detect(w: &Weights, base: &str, bias: bool) -> Result<QLinear> {
    let weight_key = format!("{base}.weight");
    let scales_key = format!("{base}.scales");
    // (1) An adapter-merged dense weight in the overlay wins — load it dense (adapter compose).
    if w.overlay_has(&weight_key) {
        return Ok(QLinear::dense(linear(w, base, bias)?));
    }
    // (2) A packed tier with a `.scales` sibling → build straight from the packed parts.
    if let (Some(cfg), true) = (w.packed(), w.contains(&scales_key)) {
        let wq = w.get_native(&weight_key)?;
        let scales = w.get_f32(&scales_key)?;
        let biases = w.get_f32(&format!("{base}.biases"))?;
        let dense_bias = if bias {
            Some(w.get(&format!("{base}.bias"))?)
        } else {
            None
        };
        return QLinear::packed(&wq, &scales, &biases, dense_bias, cfg.group_size as usize);
    }
    // (3) Dense path unchanged.
    Ok(QLinear::dense(linear(w, base, bias)?))
}

/// **Packed-detecting** [`QEmbedding`] loader (sc-9411): packed straight from the MLX triple when the
/// component is a packed tier and `{base}.scales` is present (dequantized to the component dtype — dtype
/// parity with the dense table), else a dense [`Embedding`] from `{base}.weight` (`hidden` inferred from
/// the stored `[vocab, hidden]` shape). The Krea Qwen3-VL TE keeps `embed_tokens` **dense** in the
/// hosted q4/q8 tiers, so today this takes the dense arm; the packed arm is the future-proof path (and
/// guards against a silent dense read of u32 codes should a tier ever pack the table).
pub fn embedding_detect(w: &Weights, base: &str) -> Result<QEmbedding> {
    let scales_key = format!("{base}.scales");
    if let (Some(cfg), true) = (w.packed(), w.contains(&scales_key)) {
        let wq = w.get_native(&format!("{base}.weight"))?;
        let scales = w.get_f32(&scales_key)?;
        let biases = w.get_f32(&format!("{base}.biases"))?;
        return QEmbedding::packed(&wq, &scales, &biases, w.dtype(), cfg.group_size as usize);
    }
    let weight = w.get(&format!("{base}.weight"))?;
    let hidden = weight.dim(1)?;
    Ok(QEmbedding::dense(Embedding::new(weight, hidden)))
}

/// Standard RMSNorm over the last dim with weight `w` and eps (candle's fused op). Used by the Qwen3-VL
/// text encoder (whose norm weight is applied directly, NOT the DiT's `+1` convention).
pub(crate) fn rmsnorm(x: &Tensor, w: &Tensor, eps: f64) -> Result<Tensor> {
    candle_gen::candle_nn::ops::rms_norm(&x.contiguous()?, w, eps as f32)
}

/// Load a `+1` RMSNorm weight (the reference `RMSNorm(weight = scale + 1.0)`): the on-disk `scale` is
/// centered at 0, so pre-fold the `+1` into an **f32** weight at load. Pairs with [`rms_scale`], which
/// always reduces in f32. Mirrors `mlx-gen-krea`'s `RmsScale`.
pub(crate) fn rms_scale_weight(w: &Weights, key: &str) -> Result<Tensor> {
    w.get_f32(key)? + 1.0
}

/// Apply a pre-folded `+1` RMSNorm (`weight` already = `scale + 1`, f32) over the last dim, computing
/// in f32 and casting back to `x`'s dtype — the byte-equivalent of the reference
/// `F.rms_norm(x.float(), weight).to(dtype)`.
pub(crate) fn rms_scale(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let dt = x.dtype();
    let y = candle_gen::candle_nn::ops::rms_norm(
        &x.to_dtype(DType::F32)?.contiguous()?,
        weight,
        eps as f32,
    )?;
    y.to_dtype(dt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::safetensors;
    use candle_gen::candle_nn::Module;
    use std::collections::HashMap;

    /// The Krea MLX tier's group size (64) — the one carried from `config.json`.
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

    fn write_component(dir: &Path, tensors: HashMap<String, Tensor>, quant: bool) {
        std::fs::create_dir_all(dir).unwrap();
        safetensors::save(&tensors, dir.join("model.safetensors")).unwrap();
        let cfg = if quant {
            serde_json::json!({ "quantization": { "bits": 4, "group_size": G } })
        } else {
            serde_json::json!({ "hidden_size": 6144 })
        };
        std::fs::write(dir.join("config.json"), cfg.to_string()).unwrap();
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

    /// **Packed-detect fires on the Krea key layout, incl. the `attn.to_out.0` nesting (sc-9411).** A
    /// packed q4 component (`quantization` block present) whose `attn.to_out.0` is a group-64 packed
    /// triple must `linear_detect` to a `Packed` projection — the `.scales`/`.biases` siblings surviving
    /// the `to_out.0` base — while a dense sibling (`attn.to_q`, no `.scales`) stays `Dense`. The packed
    /// forward reproduces the affine grid (proving the group-64 repack + threading is correct, not a
    /// silent dense fallback).
    #[test]
    fn linear_detect_fires_on_to_out_remap_and_leaves_dense_unchanged() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (128usize, 256usize);
        let (wq, s, b, grid) = q4_packed(out_dim, in_dim);
        let dense_w = Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev)?;

        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert("attn.to_out.0.weight".into(), wq);
        map.insert("attn.to_out.0.scales".into(), s);
        map.insert("attn.to_out.0.biases".into(), b);
        map.insert("attn.to_q.weight".into(), dense_w);

        let dir = std::env::temp_dir().join(format!("sc9411_detect_{}", std::process::id()));
        write_component(&dir, map, true);
        let w = Weights::from_dir(&dir, &dev, DType::F32)?;
        assert_eq!(w.packed().map(|c| c.group_size), Some(G as i32));

        let packed = linear_detect(&w, "attn.to_out.0", false)?;
        assert!(
            packed.is_packed(),
            "`.scales` under to_out.0 + quant config ⇒ packed load, not a silent dense fallback"
        );
        let dense = linear_detect(&w, "attn.to_q", false)?;
        assert!(!dense.is_packed(), "no `.scales` ⇒ dense path unchanged");

        // The packed forward reproduces the affine grid (group-64 repack + dequant-on-forward).
        let grid_lin = QLinear::dense(Linear::new(grid, None));
        let x = Tensor::randn(0f32, 1f32, (4, in_dim), &dev)?;
        let cos = cosine(&packed.forward(&x)?, &grid_lin.forward(&x)?);
        assert!(cos > 0.99999, "group-64 packed vs grid cosine {cos:.6}");

        std::fs::remove_dir_all(&dir).ok();
        Ok(())
    }

    /// A **dense bf16 component** (config.json has no `quantization` block) takes the dense path — the
    /// loader gates on the config, so `Weights::packed()` is `None` and every `linear_detect` stays
    /// `Dense`. The one-crate-serves-both contract.
    #[test]
    fn dense_component_takes_dense_path() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (64usize, 128usize);
        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert(
            "attn.to_q.weight".into(),
            Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev)?,
        );
        let dir = std::env::temp_dir().join(format!("sc9411_dense_{}", std::process::id()));
        write_component(&dir, map, false);

        let w = Weights::from_dir(&dir, &dev, DType::F32)?;
        assert!(w.packed().is_none(), "no quantization block ⇒ dense tier");
        assert!(!linear_detect(&w, "attn.to_q", false)?.is_packed());
        std::fs::remove_dir_all(&dir).ok();
        Ok(())
    }

    /// The packed-detecting **embedding** loader fires on a group-64 packed `embed_tokens` triple and
    /// reproduces its affine grid rows (the future-proof path — the Krea tier keeps this table dense).
    #[test]
    fn embedding_detect_group64() -> Result<()> {
        let dev = Device::Cpu;
        let (vocab, hidden) = (128usize, 128usize);
        let (wq, s, b, grid) = q4_packed(vocab, hidden);

        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert("embed_tokens.weight".into(), wq);
        map.insert("embed_tokens.scales".into(), s);
        map.insert("embed_tokens.biases".into(), b);
        let dir = std::env::temp_dir().join(format!("sc9411_emb_{}", std::process::id()));
        write_component(&dir, map, true);

        let w = Weights::from_dir(&dir, &dev, DType::F32)?;
        let emb = embedding_detect(&w, "embed_tokens")?;
        assert!(
            emb.is_packed(),
            "`.scales` + quant config ⇒ packed embedding"
        );

        let dense = QEmbedding::dense(Embedding::new(grid, hidden));
        let idx = Tensor::from_vec(vec![0u32, 5, 127, 12, 5], (5,), &dev)?;
        let dev_max = (emb.forward(&idx)?.sub(&dense.forward(&idx)?)?)
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert_eq!(dev_max, 0.0, "packed embedding deviates from the grid");
        std::fs::remove_dir_all(&dir).ok();
        Ok(())
    }

    /// **Adapter overlay wins over the packed base (sc-9411 adapter compose).** With a packed
    /// `attn.to_q` triple in the component AND an overlay-installed dense `attn.to_q.weight` (the
    /// adapter-merged weight), `linear_detect` must take the **dense** overlay path — not the packed
    /// triple — and its forward must reproduce the overlay weight exactly. This is the seam that lets a
    /// LoRA merge into a packed tier: the adapted projection loads dense, the rest stays packed.
    #[test]
    fn overlay_shadows_packed_base_for_adapter_compose() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (128usize, 256usize);
        let (wq, s, b, _grid) = q4_packed(out_dim, in_dim);

        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert("attn.to_q.weight".into(), wq);
        map.insert("attn.to_q.scales".into(), s);
        map.insert("attn.to_q.biases".into(), b);
        let dir = std::env::temp_dir().join(format!("sc9411_overlay_{}", std::process::id()));
        write_component(&dir, map, true);
        let mut w = Weights::from_dir(&dir, &dev, DType::F32)?;

        // Without an overlay, `attn.to_q` loads packed.
        assert!(linear_detect(&w, "attn.to_q", false)?.is_packed());

        // Install a distinctive dense "merged" weight in the overlay; `linear_detect` must load it dense.
        let merged = Tensor::randn(3f32, 0.5f32, (out_dim, in_dim), &dev)?;
        let mut overlay = HashMap::new();
        overlay.insert("attn.to_q.weight".to_string(), merged.clone());
        w.set_overlay(overlay);

        let lin = linear_detect(&w, "attn.to_q", false)?;
        assert!(
            !lin.is_packed(),
            "an overlaid (adapter-merged) weight must take the dense path, shadowing the packed triple"
        );
        let x = Tensor::randn(0f32, 1f32, (4, in_dim), &dev)?;
        let want = Linear::new(merged, None).forward(&x)?;
        let dev_max = (lin.forward(&x)?.sub(&want)?)
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert_eq!(
            dev_max, 0.0,
            "overlay forward must equal the merged dense weight"
        );
        std::fs::remove_dir_all(&dir).ok();
        Ok(())
    }

    /// **`get_cpu_merge_base` reconstructs the dense grid from the packed triple (sc-9411).** The
    /// adapter merge folds its delta into this reconstructed base; on a packed tier the base must be the
    /// exact affine grid the pack represents (f32), NOT the u32 codes. A dense tier returns the on-disk
    /// weight unchanged.
    #[test]
    fn get_cpu_merge_base_dequantizes_packed_and_passes_dense() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (128usize, 256usize);
        let (wq, s, b, grid) = q4_packed(out_dim, in_dim);

        // Packed tier: base is the reconstructed dense grid.
        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert("attn.to_q.weight".into(), wq);
        map.insert("attn.to_q.scales".into(), s);
        map.insert("attn.to_q.biases".into(), b);
        let dir = std::env::temp_dir().join(format!("sc9411_base_{}", std::process::id()));
        write_component(&dir, map, true);
        let w = Weights::from_dir(&dir, &dev, DType::F32)?;
        let base = w.get_cpu_merge_base("attn.to_q.weight")?;
        assert_eq!(base.dims(), &[out_dim, in_dim], "reconstructed dense shape");
        assert!(
            cosine(&base, &grid) > 0.99999,
            "reconstructed base must equal the affine grid"
        );
        std::fs::remove_dir_all(&dir).ok();

        // Dense tier: base is the on-disk weight (identity round-trip).
        let dense_w = Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev)?;
        let mut dmap: HashMap<String, Tensor> = HashMap::new();
        dmap.insert("attn.to_q.weight".into(), dense_w.clone());
        let ddir = std::env::temp_dir().join(format!("sc9411_base_dense_{}", std::process::id()));
        write_component(&ddir, dmap, false);
        let dw = Weights::from_dir(&ddir, &dev, DType::F32)?;
        let dbase = dw.get_cpu_merge_base("attn.to_q.weight")?;
        let dev_max = (dbase.sub(&dense_w)?)
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert_eq!(dev_max, 0.0, "dense tier base is the on-disk weight");
        std::fs::remove_dir_all(&ddir).ok();
        Ok(())
    }

    /// `read_packed_config` distinguishes absent-vs-corrupt (sc-9426, F-073 sibling — the flux2
    /// `component_is_packed` twin): a `quantization` block → packed `Some`, a plain config or a
    /// genuinely-absent `config.json` → dense `None` (unchanged), but a *present-but-corrupt*
    /// `config.json` (malformed JSON, e.g. a partial download) errors loudly naming the file instead
    /// of silently swallowing to the dense path.
    #[test]
    fn read_packed_config_absent_vs_corrupt() {
        let dir = std::env::temp_dir().join(format!("sc9426_krea_cfg_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        // A `quantization` block → packed tier.
        let packed = dir.join("packed");
        std::fs::create_dir_all(&packed).unwrap();
        std::fs::write(
            packed.join("config.json"),
            r#"{"quantization": {"bits": 4, "group_size": 64}}"#,
        )
        .unwrap();
        assert!(
            read_packed_config(&packed).unwrap().is_some(),
            "a `quantization` block ⇒ packed tier"
        );

        // A plain config with no `quantization` block → dense.
        let dense = dir.join("dense");
        std::fs::create_dir_all(&dense).unwrap();
        std::fs::write(dense.join("config.json"), r#"{"hidden_size": 6144}"#).unwrap();
        assert!(
            read_packed_config(&dense).unwrap().is_none(),
            "no `quantization` block ⇒ dense tier"
        );

        // No `config.json` at all → dense (single-file fixtures still load).
        let absent = dir.join("absent");
        std::fs::create_dir_all(&absent).unwrap();
        assert!(
            read_packed_config(&absent).unwrap().is_none(),
            "absent config.json ⇒ dense (unchanged)"
        );

        // A config.json that is *present but corrupt* (malformed JSON) → error naming the file, NOT a
        // silent dense fallback.
        let corrupt = dir.join("corrupt");
        std::fs::create_dir_all(&corrupt).unwrap();
        std::fs::write(corrupt.join("config.json"), b"{ not json").unwrap();
        let err = read_packed_config(&corrupt)
            .expect_err("corrupt config.json must error, not fall to dense");
        assert!(
            format!("{err}").contains("config.json"),
            "the error should name the offending file, got: {err}"
        );

        std::fs::remove_dir_all(&dir).ok();
    }
}
