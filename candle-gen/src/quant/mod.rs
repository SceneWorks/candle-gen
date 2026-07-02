//! The shared **packed-load** module (sc-9086, epic 9083) — the candle twin of
//! `mlx_gen::quant::{lin, embedding}` ([[the MLX Group-B template]], sc-8669). Every provider
//! crate's loader packed-**detects** with one call: a pre-quantized MLX tier (epic 8506, e.g.
//! `SceneWorks/z-image-turbo-mlx`) stores each quantized `Linear` / token embedding as the packed
//! triple `{base}.weight` (u32 codes) + `{base}.scales` + `{base}.biases`; [`lin`] / [`embedding`]
//! auto-detect it by the presence of `{base}.scales` (and/or the component `config.json`'s
//! `quantization: { bits, group_size }` block — [`PackedConfig`]) and build the quantized module
//! **directly from the packed parts** via the sc-9085 [`repack`] seam. Absent `.scales`, the dense
//! path is taken **unchanged**, so one loader serves both a dense bf16 and a pre-quantized snapshot.
//!
//! **The quantized forward dequantizes the weight into a *dense* matmul (sc-7702).** It does NOT
//! take candle's int8 `QMatMul` fast path (`fast_mmvq`/`fast_mmq`), which quantizes the *activation*
//! to per-32-element `q8_1`: a single outlier text feature (gpt-oss ±10⁴) sets a block's int8 scale
//! and zeros the co-located channels, so a Q4 denoise diverges to NaN — a solid-black render.
//! Dequantizing the weight to the activation dtype and running a plain matmul keeps the activation
//! full-precision, so uniform Q4 renders coherently. The resident footprint stays the small
//! quantized [`QTensor`]; the dequant is per-forward. This mirrors the Lens DiT quant (sc-5117) and
//! is regression-tested in [`tests`] (`q4_packed_forward_survives_outlier_activations`, CUDA-gated).
//!
//! **Idempotent [`QLinear::quantize`].** Crates call `quantize(bits)` after a dense load today; on a
//! QLinear that already loaded packed (`Quantized`) that call is a **no-op** — it does not
//! re-quantize (mirroring MLX's `AdaptableLinear::quantize` no-op-when-`Quantized`). So a loader can
//! packed-detect *and* keep an unconditional post-load `quantize` pass, and the two compose.
//!
//! **Bit-width & repack.** MLX packs group-wise **affine** (`w = scale·q + bias`) at group size 64;
//! the bit-width is inferred from the packed shapes ([`repack::mlx_packed_bits`]), so no side
//! manifest is needed. Q4 repacks **losslessly** into GGML `Q4_1` (same affine form; one MLX group =
//! two `Q4_1` blocks). Q8 has no affine GGML container, so the Q8 tier is materialized to its exact
//! MLX grid and re-quantized to symmetric `Q8_0` (0.56 % mean / 0.87 % worst relative RMS on the real
//! z-image Q8 tier — the accepted sc-9085 double-quant). See [`repack`] for the byte-level details.

pub mod repack;

pub use repack::{
    dequant_mlx_q4_reference, dequant_mlx_q8, f16_exact, mlx_packed_bits, repack_mlx_q4_to_q4_1,
    MLX_GROUP_SIZE,
};

use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Embedding, Linear, Module, VarBuilder};
use gen_core::Quant;

/// The GGUF block type a load-time [`Quant`] level maps to when quantizing a *dense* weight in place
/// — `Q4_0` / `Q8_0` (block size 32). Shared with the per-crate seams (Lens sc-5117, FLUX.2 sc-5917):
/// the single source of truth for the family's `Quant → GgmlDType` mapping. The **packed** path uses
/// `Q4_1` instead (the affine container the MLX tiers repack into losslessly — [`repack`]).
pub fn ggml_dtype(quant: Quant) -> GgmlDType {
    match quant {
        Quant::Q4 => GgmlDType::Q4_0,
        Quant::Q8 => GgmlDType::Q8_0,
    }
}

/// A component's `quantization` manifest block — the candle twin of `mlx_gen_flux2::config::Flux2Quant`
/// (generalized out of the dormant per-crate `Flux2Quant`, sc-9086). An install-time convert job
/// writes `quantization: { "bits", "group_size" }` into a packed component's `config.json`; its
/// presence is a redundant (with the `.scales` sibling) packed-detect key and carries the group size
/// for crates that would rather read it than infer it from shapes. Kept `i32`-typed to match the JSON
/// the MLX converter emits (and mlx-gen's `Flux2Quant`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackedConfig {
    pub bits: i32,
    pub group_size: i32,
}

impl PackedConfig {
    /// Parse the `quantization` block out of a component `config.json` value — `None` when the block
    /// is absent (a dense component) or malformed. Detects a packed tier without touching the
    /// safetensors: `PackedConfig::from_config(cfg).is_some()` ⇔ the loader should take the packed
    /// path.
    pub fn from_config(cfg: &serde_json::Value) -> Option<Self> {
        let q = cfg.get("quantization")?;
        Some(Self {
            bits: q.get("bits")?.as_i64()? as i32,
            group_size: q.get("group_size")?.as_i64()? as i32,
        })
    }
}

/// A `Linear` projection that is **dense** (the loaded bf16/f32 weight) or **GGUF-quantized** (a
/// `QTensor` weight + full-precision bias, dequantized to a dense matmul each forward — sc-7702).
/// Built either dense (`Self::linear*`) or straight from MLX-packed parts (`Self::from_packed*`);
/// [`Self::quantize`] folds a dense one to `Q4_0`/`Q8_0` in place and is a no-op on an already-packed
/// one. Every arm computes `x·Wᵀ + b`.
pub enum QLinear {
    Dense(Linear),
    Quantized {
        /// The GGUF-quantized weight (`Q4_1` from a packed tier, or `Q4_0`/`Q8_0` from a load-time
        /// quantize); dequantized to the activation dtype per forward.
        weight: QTensor,
        /// The bias kept full-precision (`None` for bias-less projections).
        bias: Option<Tensor>,
    },
}

impl QLinear {
    /// A biased dense `[out, in]` projection from `vb` (`{prefix}.weight` + `{prefix}.bias`).
    pub fn linear(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self::Dense(candle_nn::linear(in_dim, out_dim, vb)?))
    }

    /// A bias-less dense `[out, in]` projection from `vb` (`{prefix}.weight`).
    pub fn linear_no_bias(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self::Dense(candle_nn::linear_no_bias(in_dim, out_dim, vb)?))
    }

    /// Build a `Quantized` projection directly from an MLX packed triple (`wq` u32 codes + `scales` +
    /// `biases`) on `device` — Q4 via the lossless `Q4_1` repack, Q8 via dequant → `Q8_0` re-quant
    /// (bit-width inferred from the shapes). `bias` is the optional dense `{base}.bias`, kept
    /// full-precision. No dense weight is ever materialized on the Q4 path (the whole point: the
    /// packed footprint lands on `device` directly).
    pub fn from_packed(
        wq: &Tensor,
        scales: &Tensor,
        biases: &Tensor,
        bias: Option<Tensor>,
        device: &Device,
    ) -> Result<Self> {
        let weight = repack_packed_weight(wq, scales, biases, device)?;
        Ok(Self::Quantized { weight, bias })
    }

    /// `x·Wᵀ + b`. Both arms run a **dense** matmul: `Dense` delegates to `candle_nn::Linear`;
    /// `Quantized` dequantizes its weight (and bias) to the activation dtype and delegates likewise.
    /// Dequantizing to a dense matmul — rather than candle's int8 `QMatMul` fast path — is the
    /// sc-7702 fix (see the module docs).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(l) => l.forward(x),
            Self::Quantized { weight, bias } => {
                let in_dtype = x.dtype();
                let w = weight.dequantize(x.device())?.to_dtype(in_dtype)?;
                let bias = match bias {
                    Some(b) => Some(b.to_dtype(in_dtype)?),
                    None => None,
                };
                Linear::new(w, bias).forward(x)
            }
        }
    }

    /// Fold a **dense** projection to `Q4_0`/`Q8_0` in place — **idempotent**: a no-op when already
    /// `Quantized` (whether from a load-time quantize or a packed-tier load), so a loader can
    /// packed-detect *and* keep an unconditional post-load `quantize` pass. The weight is quantized on
    /// the CPU and placed back on its original device via `QTensor::quantize_onto`; the bias stays
    /// full-precision.
    pub fn quantize(&mut self, quant: Quant) -> Result<()> {
        let Self::Dense(l) = self else {
            return Ok(());
        };
        let device = l.weight().device().clone();
        let w_cpu = l.weight().to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        let weight = QTensor::quantize_onto(&w_cpu, ggml_dtype(quant), &device)?;
        let bias = l.bias().cloned();
        *self = Self::Quantized { weight, bias };
        Ok(())
    }

    /// Whether this projection loaded (or was folded) to a quantized weight.
    pub fn is_quantized(&self) -> bool {
        matches!(self, Self::Quantized { .. })
    }
}

/// A token embedding that is **dense** (the loaded `[vocab, hidden]` table) or **GGUF-quantized**
/// (the table stored as a `QTensor`, dequantized per forward). The TE `embed_tokens` is packed in the
/// MLX tiers, so the packed-load path needs the embedding analogue of [`QLinear`]. The forward is the
/// same index-select as `candle_nn::Embedding`.
pub enum QEmbedding {
    Dense(Embedding),
    Quantized {
        /// The GGUF-quantized `[vocab, hidden]` table; dequantized to `out_dtype` per forward, then
        /// index-selected.
        table: QTensor,
        hidden_size: usize,
        /// The dtype the dequantized table is cast to before index-select — the dense embedding
        /// table's dtype (i.e. `vb.dtype()`). Mirrors how [`QLinear::forward`] casts its dequantized
        /// weight to the activation dtype, so a packed bf16 text-encoder embedding yields bf16 rows
        /// exactly as the dense path would (dtype parity). Defaults to `F32` (the `QTensor`'s natural
        /// dequant dtype) for [`Self::from_packed`] callers that don't specify one.
        out_dtype: DType,
    },
}

impl QEmbedding {
    /// A dense `[vocab, hidden]` embedding from `vb` (`{prefix}.weight`).
    pub fn embedding(vocab: usize, hidden: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self::Dense(candle_nn::embedding(vocab, hidden, vb)?))
    }

    /// Build a `Quantized` embedding directly from an MLX packed triple on `device` (Q4 lossless
    /// repack / Q8 re-quant, as [`QLinear::from_packed`]). The `[vocab, hidden]` table's `hidden`
    /// (the last dim, the group axis) is `scales.cols · group_size`. The forward dequantizes to
    /// `F32` (the `QTensor`'s natural dtype); use [`Self::from_packed_dtype`] to match a non-f32
    /// dense-path output dtype (dtype parity).
    pub fn from_packed(
        wq: &Tensor,
        scales: &Tensor,
        biases: &Tensor,
        device: &Device,
    ) -> Result<Self> {
        Self::from_packed_dtype(wq, scales, biases, device, DType::F32)
    }

    /// As [`Self::from_packed`], but the forward dequantizes to `out_dtype` (the dense-path table
    /// dtype, `vb.dtype()`) — so a packed bf16 embedding yields bf16 rows exactly as the dense path
    /// would, mirroring [`QLinear::forward`]'s activation-dtype cast.
    pub fn from_packed_dtype(
        wq: &Tensor,
        scales: &Tensor,
        biases: &Tensor,
        device: &Device,
        out_dtype: DType,
    ) -> Result<Self> {
        let table = repack_packed_weight(wq, scales, biases, device)?;
        let hidden = table.shape().dims()[1];
        Ok(Self::Quantized {
            table,
            hidden_size: hidden,
            out_dtype,
        })
    }

    /// Index-select the embedding rows for `indexes`. Dense delegates to `candle_nn::Embedding`;
    /// quantized dequantizes the table and casts it to `out_dtype` (the dense-path table dtype) once,
    /// then index-selects — the same shape *and* dtype contract as the dense forward.
    pub fn forward(&self, indexes: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(e) => e.forward(indexes),
            Self::Quantized {
                table,
                hidden_size,
                out_dtype,
            } => {
                let w = table.dequantize(indexes.device())?.to_dtype(*out_dtype)?;
                Embedding::new(w, *hidden_size).forward(indexes)
            }
        }
    }

    pub fn is_quantized(&self) -> bool {
        matches!(self, Self::Quantized { .. })
    }
}

/// Repack an MLX packed triple into a resident [`QTensor`] on `device`: **Q4** via the lossless
/// `Q4_1` repack ([`repack_mlx_q4_to_q4_1`]); **Q8** via the exact-grid dequant + a `Q8_0` re-quant
/// (no affine 8-bit GGML container — the accepted sc-9085 double-quant, 0.56 % mean rel RMS). The
/// bit-width is inferred from the packed shapes ([`mlx_packed_bits`]).
fn repack_packed_weight(
    wq: &Tensor,
    scales: &Tensor,
    biases: &Tensor,
    device: &Device,
) -> Result<QTensor> {
    let (wq_cols, s_cols) = (wq.dims2()?.1, scales.dims2()?.1);
    match mlx_packed_bits(wq_cols, s_cols) {
        4 => repack_mlx_q4_to_q4_1(wq, scales, biases, device),
        8 => {
            let grid = dequant_mlx_q8(wq, scales, biases)?;
            // `quantize_onto` needs a CPU source; `dequant_mlx_q8` already returns on the CPU.
            QTensor::quantize_onto(&grid, GgmlDType::Q8_0, device)
        }
        b => candle_core::bail!(
            "unsupported MLX packed bit-width {b} (wq {wq_cols}, scales {s_cols})"
        ),
    }
}

/// Load `{base}` as a [`QLinear`] — **packed** when the `{base}.scales` sibling is present in `vb`
/// (a pre-quantized MLX tier: build the quantized weight straight from the packed parts), else
/// **dense** (`{base}.weight`, path unchanged). `bias` additionally loads the dense `{base}.bias`
/// (distinct from the packed path's own `{base}.biases`, which is always loaded packed). The candle
/// twin of `mlx_gen::quant::lin`: one loader serves both a dense bf16 and a packed snapshot, with no
/// `quantization` manifest to read. `vb`'s dtype is the dense-path weight dtype; the packed path
/// builds on `vb`'s device.
pub fn lin(
    vb: &VarBuilder,
    base: &str,
    in_dim: usize,
    out_dim: usize,
    bias: bool,
) -> Result<QLinear> {
    let scales_key = format!("{base}.scales");
    if vb.contains_tensor(&scales_key) {
        let device = vb.device().clone();
        // The u32 packed codes must load at their native `U32` (a cast to the vb's float dtype would
        // reinterpret the bit-packed nibbles); the scales/biases upcast bf16 → f32 exactly.
        let wq = vb.get_unchecked_dtype(&format!("{base}.weight"), DType::U32)?;
        let scales = vb.get_unchecked_dtype(&scales_key, DType::F32)?;
        let biases = vb.get_unchecked_dtype(&format!("{base}.biases"), DType::F32)?;
        let bias = if bias {
            Some(vb.get_unchecked_dtype(&format!("{base}.bias"), vb.dtype())?)
        } else {
            None
        };
        return QLinear::from_packed(&wq, &scales, &biases, bias, &device);
    }
    if bias {
        QLinear::linear(in_dim, out_dim, vb.pp(base))
    } else {
        QLinear::linear_no_bias(in_dim, out_dim, vb.pp(base))
    }
}

/// Load `{base}` as a [`QEmbedding`] — packed when `{base}.scales` is present, else dense (the
/// embedding analogue of [`lin`]; the candle twin of `mlx_gen::quant::embedding`). The TE
/// `embed_tokens` is packed in the MLX tiers, so this closes the packed-detect surface over both the
/// projections and the token embedding.
pub fn embedding(vb: &VarBuilder, base: &str, vocab: usize, hidden: usize) -> Result<QEmbedding> {
    let scales_key = format!("{base}.scales");
    if vb.contains_tensor(&scales_key) {
        let device = vb.device().clone();
        let wq = vb.get_unchecked_dtype(&format!("{base}.weight"), DType::U32)?;
        let scales = vb.get_unchecked_dtype(&scales_key, DType::F32)?;
        let biases = vb.get_unchecked_dtype(&format!("{base}.biases"), DType::F32)?;
        // Dequantize the table to the dense-path table dtype (`vb.dtype()`), so a packed bf16
        // text-encoder embedding yields bf16 rows exactly as the dense path would (dtype parity).
        return QEmbedding::from_packed_dtype(&wq, &scales, &biases, &device, vb.dtype());
    }
    QEmbedding::embedding(vocab, hidden, vb.pp(base))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::safetensors::MmapedSafetensors;

    // ---- helpers -----------------------------------------------------------------------------

    /// Cosine similarity over all elements (f64), the quant-parity metric.
    fn cosine(a: &Tensor, b: &Tensor) -> f32 {
        let a = a.to_dtype(DType::F32).unwrap().flatten_all().unwrap();
        let b = b.to_dtype(DType::F32).unwrap().flatten_all().unwrap();
        let a = a.to_vec1::<f32>().unwrap();
        let b = b.to_vec1::<f32>().unwrap();
        let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (x, y) in a.iter().zip(b.iter()) {
            dot += (*x as f64) * (*y as f64);
            na += (*x as f64) * (*x as f64);
            nb += (*y as f64) * (*y as f64);
        }
        (dot / (na.sqrt() * nb.sqrt() + 1e-12)) as f32
    }

    /// Test-side MLX packer: per-element 4-bit codes → MLX u32 words (LSB-first nibbles).
    fn pack_mlx_q4(codes: &[u8]) -> Vec<u32> {
        codes
            .chunks_exact(8)
            .map(|c| {
                c.iter()
                    .enumerate()
                    .fold(0u32, |acc, (i, &q)| acc | ((q as u32 & 0xF) << (4 * i)))
            })
            .collect()
    }

    /// Build a real MLX Q4 packed triple for an `[out, in]` weight (group 64) with f16-exact,
    /// per-group-distinct scales/biases and position-dependent codes — the packed-parts fixtures the
    /// `from_packed` / detect loaders consume, plus the exact affine grid they must reproduce.
    fn q4_fixture(out_dim: usize, in_dim: usize) -> (Tensor, Tensor, Tensor, Vec<f32>) {
        let dev = Device::Cpu;
        let codes: Vec<u8> = (0..out_dim * in_dim)
            .map(|i| ((i * 7 + i / 13) % 16) as u8)
            .collect();
        let groups = out_dim * in_dim / MLX_GROUP_SIZE;
        let scales: Vec<f32> = (0..groups).map(|g| 0.0625 * (g as f32 + 1.0)).collect();
        let biases: Vec<f32> = (0..groups).map(|g| -0.5 - 0.25 * g as f32).collect();
        let gpr = in_dim / MLX_GROUP_SIZE;
        let grid: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| {
                let (row, col) = (i / in_dim, i % in_dim);
                let g = row * gpr + col / MLX_GROUP_SIZE;
                scales[g] * codes[i] as f32 + biases[g]
            })
            .collect();
        let wq = Tensor::from_vec(pack_mlx_q4(&codes), (out_dim, in_dim / 8), &dev).unwrap();
        let s = Tensor::from_vec(scales, (out_dim, gpr), &dev).unwrap();
        let b = Tensor::from_vec(biases, (out_dim, gpr), &dev).unwrap();
        (wq, s, b, grid)
    }

    // ---- packed-vs-dense forward parity ------------------------------------------------------

    /// A packed-loaded `QLinear` forward matches a *dense* linear built from the SAME affine grid the
    /// pack represents — bit-exact (the repack is lossless, and both forwards dequant-to-dense-matmul).
    #[test]
    fn packed_qlinear_forward_matches_dense_grid() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (64, 128);
        let (wq, s, b, grid) = q4_fixture(out_dim, in_dim);

        let packed = QLinear::from_packed(&wq, &s, &b, None, &dev)?;
        assert!(packed.is_quantized());
        let dense = QLinear::Dense(Linear::new(
            Tensor::from_vec(grid, (out_dim, in_dim), &dev)?,
            None,
        ));

        let x = Tensor::randn(0f32, 1f32, (4, in_dim), &dev)?;
        let cos = cosine(&packed.forward(&x)?, &dense.forward(&x)?);
        assert!(cos > 0.99999, "packed vs dense-grid cosine {cos:.6}");
        Ok(())
    }

    /// A packed-loaded `QEmbedding` reproduces the affine grid rows exactly (index-select over the
    /// dequantized table == the dense grid table).
    #[test]
    fn packed_qembedding_matches_dense_grid() -> Result<()> {
        let dev = Device::Cpu;
        let (vocab, hidden) = (32, 128);
        let (wq, s, b, grid) = q4_fixture(vocab, hidden);

        let packed = QEmbedding::from_packed(&wq, &s, &b, &dev)?;
        assert!(packed.is_quantized());
        let dense = QEmbedding::Dense(Embedding::new(
            Tensor::from_vec(grid, (vocab, hidden), &dev)?,
            hidden,
        ));

        let idx = Tensor::from_vec(vec![0u32, 5, 31, 12, 5], (5,), &dev)?;
        let (p, d) = (packed.forward(&idx)?, dense.forward(&idx)?);
        let dev_max = (p.sub(&d)?).abs()?.max_all()?.to_scalar::<f32>()?;
        assert_eq!(dev_max, 0.0, "packed embedding deviates from dense grid");
        Ok(())
    }

    /// The packed `QEmbedding` forward output dtype matches the dense embedding path's — a bf16
    /// dense table yields bf16 rows, so the packed path (loaded with the same `vb.dtype()`) must too.
    /// The dense path here goes through the same VarBuilder-detect loader (`embedding`), so it holds
    /// the `[base].weight` in the vb's bf16 dtype exactly as a real bf16 text-encoder would.
    #[test]
    fn packed_qembedding_forward_dtype_matches_dense() -> Result<()> {
        let dev = Device::Cpu;
        let (vocab, hidden) = (32, 128);
        let (wq, s, b, grid) = q4_fixture(vocab, hidden);

        // A packed table and a dense table, both written to safetensors and loaded through the
        // `embedding` detect-loader at bf16 vb dtype — the dense table is the reference dtype path.
        let mut map: std::collections::HashMap<String, Tensor> = std::collections::HashMap::new();
        map.insert("emb.weight".into(), wq);
        map.insert("emb.scales".into(), s);
        map.insert("emb.biases".into(), b);
        map.insert(
            "dense.weight".into(),
            Tensor::from_vec(grid, (vocab, hidden), &dev)?,
        );

        let tmp = std::env::temp_dir().join(format!(
            "sc9086_emb_dtype_{}.safetensors",
            std::process::id()
        ));
        candle_core::safetensors::save(&map, &tmp)?;
        // SAFETY: we just wrote this file and nothing else touches it during the test.
        let st = unsafe { MmapedSafetensors::new(&tmp)? };
        let vb = VarBuilder::from_backend(Box::new(st), DType::BF16, dev.clone());

        let packed = embedding(&vb, "emb", vocab, hidden)?;
        assert!(
            packed.is_quantized(),
            "`.scales` present ⇒ packed embedding"
        );
        let dense = embedding(&vb, "dense", vocab, hidden)?;
        assert!(!dense.is_quantized(), "no `.scales` ⇒ dense embedding");

        let idx = Tensor::from_vec(vec![0u32, 5, 31, 12, 5], (5,), &dev)?;
        let p = packed.forward(&idx)?;
        let d = dense.forward(&idx)?;
        assert_eq!(d.dtype(), DType::BF16, "dense bf16 embedding yields bf16");
        assert_eq!(
            p.dtype(),
            d.dtype(),
            "packed embedding forward dtype must match the dense path (dtype parity)"
        );

        std::fs::remove_file(&tmp).ok();
        Ok(())
    }

    // ---- idempotent quantize ------------------------------------------------------------------

    /// `quantize` is a no-op on an already-packed `QLinear` — it must NOT re-quantize (double-quantize)
    /// a weight that loaded packed. The stored `Q4_1` weight and the forward stay unchanged.
    #[test]
    fn quantize_is_noop_on_packed() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (64, 128);
        let (wq, s, b, _grid) = q4_fixture(out_dim, in_dim);

        let mut packed = QLinear::from_packed(&wq, &s, &b, None, &dev)?;
        let x = Tensor::randn(0f32, 1f32, (4, in_dim), &dev)?;
        let before = packed.forward(&x)?;

        // The weight is Q4_1 (packed), not the Q4_0 a load-time quantize would emit.
        match &packed {
            QLinear::Quantized { weight, .. } => assert_eq!(weight.dtype(), GgmlDType::Q4_1),
            _ => panic!("expected packed Quantized"),
        }

        packed.quantize(Quant::Q4)?; // must no-op, not re-quantize to Q4_0
        match &packed {
            QLinear::Quantized { weight, .. } => assert_eq!(
                weight.dtype(),
                GgmlDType::Q4_1,
                "quantize re-quantized a packed weight (Q4_1 → {:?})",
                weight.dtype()
            ),
            _ => panic!("quantize turned a packed linear dense"),
        }
        let after = packed.forward(&x)?;
        let dev_max = (before.sub(&after)?).abs()?.max_all()?.to_scalar::<f32>()?;
        assert_eq!(dev_max, 0.0, "no-op quantize changed the forward");
        Ok(())
    }

    /// The dense→quantize path stays idempotent too (a second `quantize` is a no-op, not a panic).
    #[test]
    fn quantize_dense_then_idempotent() -> Result<()> {
        let dev = Device::Cpu;
        let w = Tensor::randn(0f32, 1f32, (64, 32), &dev)?;
        let mut lin = QLinear::Dense(Linear::new(w, None));
        lin.quantize(Quant::Q8)?;
        assert!(lin.is_quantized());
        lin.quantize(Quant::Q8)?; // no-op, must not error
        assert!(matches!(lin, QLinear::Quantized { bias: None, .. }));
        Ok(())
    }

    // ---- PackedConfig detect ------------------------------------------------------------------

    #[test]
    fn packed_config_detects_quantization_block() {
        let packed = serde_json::json!({ "quantization": { "bits": 4, "group_size": 64 } });
        assert_eq!(
            PackedConfig::from_config(&packed),
            Some(PackedConfig {
                bits: 4,
                group_size: 64
            })
        );
        let dense = serde_json::json!({ "hidden_size": 2048 });
        assert_eq!(PackedConfig::from_config(&dense), None);
    }

    // ---- packed-detect over a VarBuilder ------------------------------------------------------

    /// `lin` / `embedding` route to the packed path when the `.scales` sibling is present in the
    /// VarBuilder, and to the dense path otherwise — the one-call packed-detect contract. Uses an
    /// in-memory safetensors so no weights are needed.
    #[test]
    fn lin_and_embedding_packed_detect() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (64, 128);
        let (wq, s, b, _grid) = q4_fixture(out_dim, in_dim);
        let mut map: std::collections::HashMap<String, Tensor> = std::collections::HashMap::new();
        map.insert("proj.weight".into(), wq);
        map.insert("proj.scales".into(), s);
        map.insert("proj.biases".into(), b);
        // A dense sibling with no `.scales`.
        map.insert(
            "dense.weight".into(),
            Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev)?,
        );

        let tmp =
            std::env::temp_dir().join(format!("sc9086_detect_{}.safetensors", std::process::id()));
        candle_core::safetensors::save(&map, &tmp)?;
        // SAFETY: we just wrote this file and nothing else touches it during the test.
        let st = unsafe { MmapedSafetensors::new(&tmp)? };
        let vb = VarBuilder::from_backend(Box::new(st), DType::F32, dev.clone());

        let packed = lin(&vb, "proj", in_dim, out_dim, false)?;
        assert!(packed.is_quantized(), "`.scales` present ⇒ packed path");
        let dense = lin(&vb, "dense", in_dim, out_dim, false)?;
        assert!(!dense.is_quantized(), "no `.scales` ⇒ dense path unchanged");

        let emb = embedding(&vb, "proj", out_dim, in_dim)?;
        assert!(emb.is_quantized(), "`.scales` present ⇒ packed embedding");

        std::fs::remove_file(&tmp).ok();
        Ok(())
    }

    // ---- sc-7702 outlier / no-fast_mmq regression --------------------------------------------

    /// **sc-7702 regression (CUDA only).** The shared `QLinear` forward must stay accurate for
    /// activations with massive outliers — the gpt-oss text features (±10⁴) that made a Q4 DiT render
    /// solid black through candle's int8 `QMatMul` fast path (`fast_mmq`, batch > 8): that path
    /// quantizes the *activation* to per-32-element `q8_1`, so one outlier sets a block scale that
    /// rounds every co-located channel to **zero**. The packed-load `QLinear` must NOT take that path;
    /// it dequantizes the weight to a *dense* matmul, keeping the activation full-precision. Here each
    /// block's outlier sits on a **zero-weight** channel (carries no signal — the reference is built
    /// purely from its block-mates): the dequant path tracks the f32 reference, the raw int8 path
    /// collapses to ~0. A revert to `QMatMul::forward` inside `QLinear::forward` fails the `> 0.99`
    /// assert. This exercises the shared **dequant-on-forward** compute path in `QLinear::forward` —
    /// the byte-identical path a packed `Q4_1` load and a load-time `Q4_0` quantize both feed (the
    /// weight is dequantized to a dense matmul; the int8 activation fast path stays off), so a
    /// `Q4_0` `QLinear` built via the load-time `quantize` covers the packed forward too. Skips on
    /// CPU (the int8 MMVQ/MMQ path is CUDA-only).
    #[test]
    fn q4_packed_forward_survives_outlier_activations() -> Result<()> {
        use candle_core::quantized::QMatMul;
        let dev = match crate::default_device() {
            Ok(d) if !matches!(d, Device::Cpu) => d,
            _ => {
                eprintln!(
                    "SKIP q4_packed_forward_survives_outlier_activations: needs a CUDA device"
                );
                return Ok(());
            }
        };
        let (in_dim, out_dim, m, blk) = (256usize, 256usize, 64usize, 32usize); // m>8 ⇒ MMQ path

        // A dense weight whose per-32-block channel-0 is ZEROED (the outlier channel carries no
        // signal). Built on the CPU (the `quantize_onto` source requirement) then moved to `dev`.
        let mut w = Tensor::randn(0f32, 0.1f32, (out_dim, in_dim), &Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for o in 0..out_dim {
            for b in 0..(in_dim / blk) {
                w[o * in_dim + b * blk] = 0.0;
            }
        }
        let w_cpu = Tensor::from_vec(w, (out_dim, in_dim), &Device::Cpu)?;
        let w = w_cpu.to_device(&dev)?;

        // Activation: ~N(0,1) with a +30000 outlier on each block's channel-0.
        let mut x = Tensor::randn(0f32, 1f32, (m, in_dim), &dev)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for r in 0..m {
            for b in 0..(in_dim / blk) {
                x[r * in_dim + b * blk] = 30000.0;
            }
        }
        let x = Tensor::from_vec(x, (m, in_dim), &dev)?;
        let reference = x.matmul(&w.t()?)?; // f32 dense ground truth on `dev`

        // Dequant-on-forward path: a Q4 `QLinear` built via the load-time `quantize` — the shortest
        // way to a real QTensor, and the byte-identical `QLinear::forward` compute path a packed
        // `from_packed` load also feeds (both dequant the weight to a dense matmul).
        let qt = QTensor::quantize_onto(&w_cpu, GgmlDType::Q4_0, &dev)?;
        let mut lin = QLinear::Dense(Linear::new(w, None));
        lin.quantize(Quant::Q4)?;
        let dequant = lin.forward(&x)?;

        // Raw int8 `QMatMul` over the same Q4 weight — the path the black-render bug took.
        let int8 = QMatMul::from_qtensor(qt)?.forward(&x)?;

        let dq = cosine(&dequant, &reference);
        let q8 = cosine(&int8, &reference);
        eprintln!("sc-7702 packed outlier cosine: dequant(QLinear)={dq:.4}  int8(QMatMul)={q8:.4}");
        assert!(
            dq > 0.99,
            "packed QLinear Q4 forward must track f32 under outlier activations (cosine {dq:.4}) — \
             the int8 activation-quant path must not return inside QLinear::forward (sc-7702)"
        );
        assert!(
            dq > q8 + 0.05,
            "vacuous test: the raw int8 path ({q8:.4}) did not degrade vs dequant ({dq:.4})"
        );
        Ok(())
    }
}
