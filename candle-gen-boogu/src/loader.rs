//! Weight loading for the Boogu DiT + Qwen3-VL condition encoder — a thin shape-inferring wrapper
//! over candle's [`MmapedSafetensors`], mirroring `mlx-gen-boogu`'s `Weights`/`lin` interface (and
//! `candle-gen-ideogram`'s `loader::Weights`) so the port stays a near-1:1 translation. [`linear`]
//! builds a [`Linear`] from the actual `{base}.weight` (+ optional `{base}.bias`) tensor shapes, so
//! dims that aren't in the public config (the FFN inner width, the embedder MLP hidden) need no
//! hardcoding.
//!
//! **Packed-tier detect (sc-9410).** When a component dir is an MLX-packed q4 snapshot
//! (`SceneWorks/boogu-image-mlx`, group size 32), each quantized projection / embedding is stored as
//! the triple `{base}.weight` (u32 codes) + `{base}.scales` + `{base}.biases`, and the component
//! `config.json` carries a `quantization: { bits, group_size }` block ([`candle_gen::quant::PackedConfig`]).
//! [`Weights::from_dir`] reads that block; [`linear_detect`] / [`embedding_detect`] then packed-**detect**
//! the `.scales` sibling and build the quantized module straight from the packed parts through the shared
//! group-size-aware loaders (no dense staging — see [`crate::quant`]). Absent the block / `.scales`, the
//! dense path is unchanged.
//!
//! **Vision tower is dense bf16 even in the packed tiers (sc-9410).** MLX packs the boogu `mllm/`
//! component selectively: in `edit-q4` only `model.language_model.*` (the TE, 252 `.scales`) is
//! packed; all 351 `model.visual.*` (the Qwen3-VL vision tower) stay BF16 (verified against the hosted
//! `SceneWorks/boogu-image-mlx/edit-q4/mllm/model.safetensors` header — 0 `model.visual.*.scales`).
//! The vision tower therefore loads dense via [`linear_guard_dense`], but it *shares* the `mllm/` dir
//! (and so the packed `config.json` flag) with the packed TE — so a bare dense read here would be a
//! **silent** skip if a future tier ever packed the tower. [`linear_guard_dense`] errors loudly on a
//! stray `.scales` sibling instead.

use std::path::Path;

use candle_gen::candle_core::safetensors::MmapedSafetensors;
use candle_gen::candle_core::{DType, Device, Result, Tensor, D};
use candle_gen::candle_nn::{Embedding, Linear};
use candle_gen::quant::PackedConfig;

use crate::quant::{QEmbedding, QLinear};

/// An mmaped component-directory of `.safetensors`, loading tensors at a fixed compute dtype. Carries
/// the MLX `quantization` block ([`Self::packed`]) when the component is a packed q4 tier (sc-9410),
/// so [`linear_detect`] / [`embedding_detect`] can build the quantized modules at the tier's group
/// size straight from the packed parts.
pub struct Weights {
    st: MmapedSafetensors,
    device: Device,
    dtype: DType,
    /// The component's `quantization` manifest, `Some` for a packed q4 tier (carries the group size
    /// the shapes can't disambiguate), `None` for a dense bf16 tier.
    packed: Option<PackedConfig>,
}

impl Weights {
    /// mmap every `*.safetensors` in `dir` (sorted; later files win on name collision), reading the
    /// component `config.json`'s `quantization` block (if any) for the packed-tier path.
    pub fn from_dir(dir: &Path, device: &Device, dtype: DType) -> Result<Self> {
        let files = candle_gen::sorted_safetensors(dir, "boogu")
            .map_err(|e| candle_gen::candle_core::Error::Msg(e.to_string()))?;
        // SAFETY: read-only mmap of weight files; the standard candle loading path.
        let st = unsafe { MmapedSafetensors::multi(&files)? };
        Ok(Self {
            st,
            device: device.clone(),
            dtype,
            packed: read_packed_config(dir)?,
        })
    }

    /// Load `name` at the component dtype.
    pub fn get(&self, name: &str) -> Result<Tensor> {
        self.st.load(name, &self.device)?.to_dtype(self.dtype)
    }

    /// Load `name` at its **native** stored dtype (no cast) on the component device — used for the
    /// packed triple's u32 codes (a cast would reinterpret the bit-packed nibbles).
    pub fn get_native(&self, name: &str) -> Result<Tensor> {
        self.st.load(name, &self.device)
    }

    /// Load `name` forcing f32 (norm weights and other precision-sensitive scalars).
    pub fn get_f32(&self, name: &str) -> Result<Tensor> {
        self.st.load(name, &self.device)?.to_dtype(DType::F32)
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

    /// The MLX `quantization` block when this component is a packed q4 tier, else `None`.
    pub fn packed(&self) -> Option<PackedConfig> {
        self.packed
    }
}

/// Read `{dir}/config.json`'s `quantization` block: `Some` for a packed q4 tier, `None` for a dense
/// tier. A **genuinely-absent** `config.json` (file NotFound) is a legitimate dense/single-file fixture
/// shape → `None` (a fixture with no `config.json` still loads dense). A config that **is present but
/// corrupt** (I/O error or malformed JSON — e.g. a partial download) errors loudly naming the file
/// rather than silently swallowing to the dense path (wrong tier / missing weights, no diagnostic).
/// Mirrors krea's `read_packed_config` and z-image's `component_is_packed` (sc-9426, F-073 sibling).
fn read_packed_config(dir: &Path) -> Result<Option<PackedConfig>> {
    let path = dir.join("config.json");
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        // No config.json at all → legitimate dense / single-file fixture tier.
        Err(ref e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        // Present but unreadable (permissions, partial download) → surface, don't swallow.
        Err(e) => {
            return Err(candle_gen::candle_core::Error::Msg(format!(
                "boogu: read {}: {e}",
                path.display()
            )))
        }
    };
    // Present but malformed JSON → corrupt snapshot, error rather than fall to dense.
    let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| {
        candle_gen::candle_core::Error::Msg(format!(
            "boogu: parse {} (corrupt snapshot?): {e}",
            path.display()
        ))
    })?;
    Ok(PackedConfig::from_config(&v))
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

/// A **dense** [`Linear`] like [`linear`], but with a loud guard: if a `{base}.scales` sibling is
/// present (i.e. this weight is actually MLX-packed u32 codes), it errors instead of silently reading
/// the codes as bf16 garbage (sc-9410, Issue 1 branch c).
///
/// The Qwen3-VL **vision tower** (`model.visual.*`) is dense bf16 in *every* hosted boogu tier —
/// including `edit-q4`, whose `mllm/config.json` DOES carry `quantization: { bits:4, group_size:32 }`
/// but where MLX selectively packed **only** the language model (`model.language_model.*`, 252
/// `.scales`) and left all 351 `model.visual.*` tensors BF16 (verified 2026-07-02 against the hosted
/// `SceneWorks/boogu-image-mlx/edit-q4/mllm/model.safetensors` header: 0 `model.visual.*.scales`).
/// So the dense path is correct today. But the vision tower shares the `mllm/` dir — hence the packed
/// `config.json` flag — with the packed TE, so a bare dense [`linear`] here would be a *silent* skip
/// if a future tier ever packs `model.visual.*`. This guard turns that latent silent-garbage into a
/// hard load error. (Dense components with no packed config are unaffected: `packed()` is `None`.)
pub fn linear_guard_dense(w: &Weights, base: &str, bias: bool) -> Result<Linear> {
    if w.packed().is_some() && w.contains(&format!("{base}.scales")) {
        return Err(candle_gen::candle_core::Error::Msg(format!(
            "boogu: `{base}` has a `.scales` sibling in a packed component but is loaded dense — the \
             vision tower is bf16 in the hosted tiers; a packed vision tower must route through \
             `linear_detect` (sc-9410)."
        )));
    }
    linear(w, base, bias)
}

/// **Packed-detecting** [`QLinear`] loader (sc-9410): when the component is a packed q4 tier *and*
/// `{base}.scales` is present, build a `Packed` projection straight from the MLX packed triple at the
/// tier's group size — **no dense weight is materialized**. Otherwise the dense path is taken
/// unchanged (`{base}.weight` [+ `{base}.bias`], the exact [`linear`] behavior).
///
/// `base` is the full dotted key prefix (e.g. `attn.to_out.0`), so the `.scales`/`.biases` siblings
/// survive any `to_out.0`-style nesting — the key-remap trap: build the base string first, then detect
/// (the `linear_detect_fires_on_to_out_remap` test pins this on the real boogu `to_out.0` layout).
pub fn linear_detect(w: &Weights, base: &str, bias: bool) -> Result<QLinear> {
    let scales_key = format!("{base}.scales");
    if let (Some(cfg), true) = (w.packed(), w.contains(&scales_key)) {
        let wq = w.get_native(&format!("{base}.weight"))?;
        let scales = w.get_f32(&scales_key)?;
        let biases = w.get_f32(&format!("{base}.biases"))?;
        let dense_bias = if bias {
            Some(w.get(&format!("{base}.bias"))?)
        } else {
            None
        };
        return QLinear::packed(&wq, &scales, &biases, dense_bias, cfg.group_size as usize);
    }
    Ok(QLinear::dense(linear(w, base, bias)?))
}

/// **Packed-detecting** [`QEmbedding`] loader (sc-9410): packed straight from the MLX triple when the
/// component is a packed q4 tier and `{base}.scales` is present (dequantized to the component dtype —
/// dtype parity with the dense table), else a dense [`Embedding`] from `{base}.weight` (`hidden`
/// inferred from the stored `[vocab, hidden]` shape). The Qwen3-VL TE `embed_tokens` is packed in q4.
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

/// RMSNorm over the last dim with weight `w` (candle's fused op; eps as f32). Inference-only — the
/// fused kernel has no backward, which is irrelevant here.
pub(crate) fn rmsnorm(x: &Tensor, w: &Tensor, eps: f64) -> Result<Tensor> {
    candle_gen::candle_nn::ops::rms_norm(&x.contiguous()?, w, eps as f32)
}

/// Plain LayerNorm over the last dim with **no affine** (LuminaLayerNormContinuous's inner norm,
/// eps 1e-6): `(x − mean) / sqrt(var + eps)`. Computed in f32 then cast back to `x`'s dtype.
pub(crate) fn layernorm_noaffine(x: &Tensor, eps: f64) -> Result<Tensor> {
    let dt = x.dtype();
    let x = x.to_dtype(DType::F32)?;
    let mean = x.mean_keepdim(D::Minus1)?;
    let centered = x.broadcast_sub(&mean)?;
    let var = centered.sqr()?.mean_keepdim(D::Minus1)?;
    let normed = centered.broadcast_div(&(var + eps)?.sqrt()?)?;
    normed.to_dtype(dt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::safetensors;
    use std::collections::HashMap;

    /// The boogu MLX tier's group size (32) — the one the shapes can't recover, so the loader must
    /// carry it from `config.json`.
    const G: usize = 32;

    /// Build an MLX group-32 Q4 packed triple for an `[out, in]` weight — `(wq u32, scales, biases,
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
            serde_json::json!({ "hidden_size": 3360 })
        };
        std::fs::write(dir.join("config.json"), cfg.to_string()).unwrap();
    }

    /// **Packed-detect fires on the boogu key layout, incl. the `attn.to_out.0` nesting (sc-9410).**
    /// A packed q4 component (with the `quantization` block) whose `attn.to_out.0` is a group-32 packed
    /// triple must `linear_detect` to a `Packed` projection — the `.scales`/`.biases` siblings surviving
    /// the `to_out.0` base — while a dense sibling (`attn.to_q`, no `.scales`) stays `Dense`. The packed
    /// forward reproduces the affine grid the pack represents (proving the group-32 repack + threading is
    /// correct, not a silent dense fallback).
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
        map.insert("attn.to_q.weight".into(), dense_w.clone());

        let dir = std::env::temp_dir().join(format!("sc9410_detect_{}", std::process::id()));
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

        // The packed forward reproduces the affine grid (group-32 repack + dequant-on-forward).
        let grid_lin = QLinear::dense(Linear::new(grid, None));
        let x = Tensor::randn(0f32, 1f32, (4, in_dim), &dev)?;
        let p = packed.forward(&x)?.flatten_all()?.to_vec1::<f32>()?;
        let g = grid_lin.forward(&x)?.flatten_all()?.to_vec1::<f32>()?;
        let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (a, c) in p.iter().zip(&g) {
            dot += (*a as f64) * (*c as f64);
            na += (*a as f64) * (*a as f64);
            nb += (*c as f64) * (*c as f64);
        }
        let cos = dot / (na.sqrt() * nb.sqrt() + 1e-12);
        assert!(cos > 0.99999, "group-32 packed vs grid cosine {cos:.6}");

        std::fs::remove_dir_all(&dir).ok();
        Ok(())
    }

    /// **The vision-tower dense guard errors loudly on a stray `.scales` (sc-9410, Issue 1 branch c).**
    /// The Qwen3-VL vision tower is bf16 in every hosted boogu tier, so it loads dense via
    /// `linear_guard_dense`. But it shares the packed `mllm/` config with the packed TE, so if a future
    /// tier ever packed a `model.visual.*` weight, a bare dense read would silently load u32 codes as
    /// garbage. The guard turns that into a hard load error instead. A dense sibling (no `.scales`)
    /// still loads fine.
    #[test]
    fn linear_guard_dense_errors_on_packed_vision_weight() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (128usize, 256usize);
        let (wq, s, b, _grid) = q4_packed(out_dim, in_dim);

        let mut map: HashMap<String, Tensor> = HashMap::new();
        // A hypothetically-packed vision weight (would be silent garbage under a bare dense read).
        map.insert("blocks.0.attn.qkv.weight".into(), wq);
        map.insert("blocks.0.attn.qkv.scales".into(), s);
        map.insert("blocks.0.attn.qkv.biases".into(), b);
        // A genuinely dense vision weight (the reality in the hosted tiers).
        map.insert(
            "blocks.0.attn.proj.weight".into(),
            Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev)?,
        );

        let dir = std::env::temp_dir().join(format!("sc9410_guard_{}", std::process::id()));
        write_component(&dir, map, true); // packed component config (quantization block present)
        let w = Weights::from_dir(&dir, &dev, DType::F32)?;
        assert!(w.packed().is_some(), "packed component config");

        // Packed vision weight ⇒ loud error, never a silent dense read of u32 codes.
        let err = linear_guard_dense(&w, "blocks.0.attn.qkv", false);
        assert!(
            err.is_err(),
            "a `.scales` sibling on a dense-loaded vision weight must error, not silently load garbage"
        );
        // Genuinely-dense vision weight ⇒ loads fine.
        assert!(
            linear_guard_dense(&w, "blocks.0.attn.proj", false).is_ok(),
            "a dense vision weight (no `.scales`) still loads"
        );

        std::fs::remove_dir_all(&dir).ok();
        Ok(())
    }

    /// A **dense bf16 component** (config.json has no `quantization` block) takes the dense path even if
    /// a stray `.scales` sibling were present — the loader gates on the config's group size, so
    /// `Weights::packed()` is `None` and every `linear_detect` stays `Dense`. The one-crate-serves-both
    /// contract: the same code loads a dense bf16 tier and a packed q4 tier.
    #[test]
    fn dense_component_takes_dense_path() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (64usize, 128usize);
        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert(
            "attn.to_q.weight".into(),
            Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev)?,
        );
        let dir = std::env::temp_dir().join(format!("sc9410_dense_{}", std::process::id()));
        write_component(&dir, map, false);

        let w = Weights::from_dir(&dir, &dev, DType::F32)?;
        assert!(w.packed().is_none(), "no quantization block ⇒ dense tier");
        let lin = linear_detect(&w, "attn.to_q", false)?;
        assert!(!lin.is_packed(), "dense tier ⇒ dense projection");

        std::fs::remove_dir_all(&dir).ok();
        Ok(())
    }

    /// `read_packed_config` distinguishes absent-vs-corrupt (sc-9426, F-073 sibling): a `quantization`
    /// block → packed `Some`, a plain config or a genuinely-absent `config.json` → dense `None`
    /// (unchanged), but a *present-but-corrupt* `config.json` (malformed JSON, e.g. a partial download)
    /// errors loudly naming the file instead of silently swallowing to the dense path.
    #[test]
    fn read_packed_config_absent_vs_corrupt() {
        let dir = std::env::temp_dir().join(format!("sc9426_boogu_cfg_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        // A `quantization` block → packed tier.
        let packed = dir.join("packed");
        std::fs::create_dir_all(&packed).unwrap();
        std::fs::write(
            packed.join("config.json"),
            r#"{"quantization": {"bits": 4, "group_size": 32}}"#,
        )
        .unwrap();
        assert!(
            read_packed_config(&packed).unwrap().is_some(),
            "a `quantization` block ⇒ packed tier"
        );

        // A plain config with no `quantization` block → dense.
        let dense = dir.join("dense");
        std::fs::create_dir_all(&dense).unwrap();
        std::fs::write(dense.join("config.json"), r#"{"hidden_size": 3360}"#).unwrap();
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

    /// The packed-detecting **embedding** loader fires on a group-32 packed `embed_tokens` triple and
    /// reproduces its affine grid rows, dequantized to the component dtype (dtype parity); a dense
    /// `embed_tokens` (the boogu tier actually keeps this table bf16) stays `Dense`.
    #[test]
    fn embedding_detect_group32() -> Result<()> {
        let dev = Device::Cpu;
        let (vocab, hidden) = (96usize, 128usize);
        let (wq, s, b, grid) = q4_packed(vocab, hidden);

        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert("embed_tokens.weight".into(), wq);
        map.insert("embed_tokens.scales".into(), s);
        map.insert("embed_tokens.biases".into(), b);
        let dir = std::env::temp_dir().join(format!("sc9410_emb_{}", std::process::id()));
        write_component(&dir, map, true);

        let w = Weights::from_dir(&dir, &dev, DType::F32)?;
        let emb = embedding_detect(&w, "embed_tokens")?;
        assert!(
            emb.is_packed(),
            "`.scales` + quant config ⇒ packed embedding"
        );

        let dense = QEmbedding::dense(Embedding::new(grid, hidden));
        let idx = Tensor::from_vec(vec![0u32, 5, 95, 12, 5], (5,), &dev)?;
        let p = emb.forward(&idx)?;
        let d = dense.forward(&idx)?;
        let dev_max = (p.sub(&d)?).abs()?.max_all()?.to_scalar::<f32>()?;
        assert_eq!(
            dev_max, 0.0,
            "group-32 packed embedding deviates from the grid"
        );

        std::fs::remove_dir_all(&dir).ok();
        Ok(())
    }
}
