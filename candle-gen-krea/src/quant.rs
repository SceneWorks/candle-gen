//! Krea 2 packed-load seam (sc-9411, sc-9089 umbrella) — the candle twin of the flux2-dev conversion
//! (sc-9087), z-image (sc-9408), flux-schnell (sc-9407), and boogu (sc-9410), built on the shared
//! [`candle_gen::quant`] packed-load module (sc-9086).
//!
//! Krea ships a **pre-quantized** MLX tier (`SceneWorks/krea-2-turbo-mlx`, bf16/q4/q8) whose q4/q8
//! snapshots store each quantized `Linear` as the MLX packed triple `{base}.weight` (u32 codes) +
//! `{base}.scales` + `{base}.biases`. The group size is read from each component `config.json`'s
//! `quantization.group_size` ([`candle_gen::quant::PackedConfig`]) and threaded through the shared
//! group-size-aware loaders ([`candle_gen::quant::QLinear::from_packed_gs`] /
//! `QEmbedding::from_packed_dtype_gs`, Q4 → `Q4_1` lossless repack, Q8 → `Q8_0`) — **not** a hardcoded
//! 64 (the hosted tier happens to pack at 64, but the loader honours whatever the config says, exactly
//! as boogu threads its group 32). **No dense bf16 weight is ever materialized** on the packed path.
//!
//! Absent `.scales` (a dense bf16 tier, or a projection MLX left dense) the **dense** path is taken
//! **unchanged** (`candle_nn::Linear` / `Embedding` from `{base}.weight`), so one crate serves both a
//! dense bf16 and a packed q4/q8 snapshot.
//!
//! Krea's loader is a thin `MmapedSafetensors` wrapper ([`crate::loader::Weights`]) with an
//! adapter-merge **overlay** (`set_overlay`, sc-7836), not a `VarBuilder` — so this seam builds the
//! quantized module from **raw tensors** pulled through `Weights` (`from_packed_gs`) rather than the
//! VarBuilder-detecting `candle_gen::quant::lin`. The compute path is identical (the shared
//! dequant-on-forward `QLinear`/`QEmbedding`, sc-7702 — *not* candle's int8 `QMatMul` fast path, whose
//! q8_1 activation quant NaNs on outlier text features).
//!
//! **Adapter compose (sc-9411).** The `krea_2_raw` LoRA/LoKr merge folds deltas into the DiT attention
//! /FFN Linears — which is exactly the packed surface. On a packed tier the merge reconstructs the
//! dense base from the packed parts on the CPU and installs the merged **dense** weight in the overlay
//! ([`crate::adapters::merge_into_weights`]); the overlay then shadows the packed path for those keys
//! ([`crate::loader::linear_detect`] reads the overlay first). So the packed base stays packed for
//! untargeted projections, while an adapted projection resolves to its correct merged dense weight —
//! there is no packed `quantize_onto` to no-op here (that is the VarBuilder crates' seam).

use candle_gen::candle_core::{DType, Device, Result, Tensor};
use candle_gen::candle_nn::{Embedding, Linear, Module};
use candle_gen::quant::{self as shared, AdaptLinear};

/// A Linear projection that is **residual-capable** (dense or MLX-packed base + optional forward-time
/// additive LoRA/LoKr residuals — the shared [`candle_gen::quant::AdaptLinear`], sc-11105) or
/// **int8-ConvRot** (a community INT8-ConvRot checkpoint's per-output-channel int8 projection, sc-9300).
/// Built dense ([`Self::dense`]), packed ([`Self::packed`]), or int8 ([`Self::convrot_int8`]); every
/// forward computes `x·Wᵀ + b` (+ any additive residual).
///
/// The `Adapt` arm folds the former `Dense(Linear)` + `Packed(shared::QLinear)` cases into the one
/// shared `AdaptLinear` (sc-11105): a dense base (the loaded bf16 weight, possibly adapter-merged via
/// the overlay — [`crate::adapters::merge_into_weights`]) or an MLX-packed base loaded straight from the
/// packed parts (`Q4_1`/`Q8_0`, **dequantizes-on-forward** into a dense matmul — sc-7702, *not* the int8
/// `QMatMul` fast path). On a **packed** tier a user LoRA rides as a forward-time additive residual on
/// the packed base ([`crate::adapters::install_additive`]) — the packed base keeps its footprint.
pub enum QLinear {
    /// Dense or MLX-packed base + optional forward-time additive residuals (sc-11105).
    Adapt(AdaptLinear),
    /// A community **INT8-ConvRot** projection (sc-9300 loader + sc-9601 online rotation): the checkpoint
    /// ships `(N, K)` int8 codes for the **rotated** weight `RHT(W) = W·R` + a `[N]` per-output-row
    /// `weight_scale`, where `R` is the regular-Hadamard transform applied block-diagonally in groups of
    /// `group_size` (256) along `K`. The forward applies the matching **online activation rotation**
    /// `RHT(x) = x·R` ([`candle_gen::quant::convrot_rotate`]) *before* the int8 GEMM, so
    /// `RHT(x)·RHT(W)ᵀ = x·Wᵀ` (`R` orthogonal). On CUDA that GEMM is a cuBLASLt IGEMM + per-row dequant
    /// ([`candle_gen::quant::Int8Linear`]); off-CUDA (CPU tests / Metal) it dequantizes the stored rotated
    /// weight to a dense matmul.
    ///
    /// **The rotation is the sc-9300 → sc-9601 fix.** Without `RHT(x)` the GEMM reconstructs
    /// `x·(W·R)ᵀ` — noise (dequantized `blocks.0.attn.wq` has cosine ≈ 0.07 with the canonical `to_q`;
    /// the A/B render was PSNR ≈ 8 dB vs bf16). With it the render is coherent (verified cosine 0.99991
    /// vs the f32 reference linear). See [`candle_gen::quant::convrot`] for the recovered transform.
    ConvRotInt8(ConvRotInt8),
}

/// The stored parts of an INT8-ConvRot projection (sc-9300 loader + sc-9601 rotation): the `(N, K)` int8
/// codes of the **rotated** weight `W·R` (on the CPU as `I64`, staged to the resident **compute** device
/// `i8` inside `Int8Linear`), the `[N]` per-output-row dequant scale, the ConvRot `group_size` (the
/// regular-Hadamard order, 256), and the optional dense bias (ConvRot Krea projections are bias-free, but
/// the field keeps the type general). When the model's compute device is CUDA, a per-channel `Int8Linear`
/// is built **eagerly at construction** on that device (F-121 / sc-11208) — the loader materializes the
/// int8 codes on the CPU to avoid 8× VRAM, so the leg is built on the compute device the activations live
/// on, **not** the CPU-resident codes' device, and `Int8Linear::from_per_channel_parts` stages the CPU
/// codes onto it. The CPU / non-CUDA fallback dequantizes the rotated `w[o, :] = q[o, :] · scale[o]`. Both
/// paths first apply the online activation rotation `RHT(x)` (built once, cached in `rot`) so the GEMM
/// reconstructs `x·Wᵀ`.
pub struct ConvRotInt8 {
    w_i8: Tensor,
    scale: Vec<f32>,
    /// The regular-Hadamard order `R` was folded into the stored weight at (`convrot_groupsize`, 256).
    group_size: usize,
    bias: Option<Tensor>,
    /// The `[group_size, group_size]` rotation `R = H/√group_size`, built lazily on the activation's
    /// device on first forward and cached (tiny — 256² f32 — but shouldn't rebuild per projection/step).
    rot: std::sync::OnceLock<Tensor>,
    /// The cuBLASLt IGEMM leg, built eagerly at construction on the model's resident **compute** device
    /// when that device is CUDA (F-121 / sc-11208): a cublasLt init failure surfaces as the crate's typed
    /// error from [`QLinear::convrot_int8`] (where `?` is available) instead of aborting the sampler
    /// thread via `.expect()` mid-render. Built on the compute device, **not** the CPU-resident codes'
    /// device (the loader keeps the int8 codes on the CPU to save VRAM), so a normal CUDA render always
    /// takes this int8 IGEMM leg. `None` when the compute device isn't CUDA (CPU / Metal fallback in
    /// [`Self::forward`]).
    #[cfg(feature = "cuda")]
    lt: Option<std::sync::Arc<candle_gen::quant::Int8Linear>>,
}

impl ConvRotInt8 {
    /// The dense `(N, K)` f32 *rotated* weight the int8 codes + per-row scale represent
    /// (`w[o,:] = q[o,:]·s[o]` = `RHT(W)[o,:]`). Paired with the online `RHT(x)` this yields `x·Wᵀ`.
    fn dequant_dense(&self) -> Result<Tensor> {
        let n = self.w_i8.dims2()?.0;
        let scale_col = Tensor::from_vec(self.scale.clone(), (n, 1), self.w_i8.device())?;
        self.w_i8.to_dtype(DType::F32)?.broadcast_mul(&scale_col)
    }

    /// The cached regular-Hadamard rotation `R` on `x`'s device (built once).
    fn rotation(&self, x: &Tensor) -> Result<&Tensor> {
        if let Some(r) = self.rot.get() {
            return Ok(r);
        }
        let r = candle_gen::quant::regular_hadamard(self.group_size, x.device())?;
        Ok(self.rot.get_or_init(|| r))
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Online ConvRot leg (sc-9601): rotate the activation by the same regular Hadamard folded into
        // the stored weight, so `RHT(x)·RHT(W)ᵀ = x·Wᵀ`. Runs on both the CUDA and CPU paths.
        let r = self.rotation(x)?.clone();
        let xr = candle_gen::quant::convrot_rotate(x, &r)?;

        #[cfg(feature = "cuda")]
        if x.device().is_cuda() {
            // Eagerly built in `convrot_int8` (F-121 / sc-11208) on the compute device; present whenever
            // the model's compute device is CUDA — which is the case on every real CUDA render, even
            // though the int8 codes themselves are CPU-materialized. If it's absent (CPU/Metal compute),
            // fall through to the dequant-dense path.
            if let Some(lin) = &self.lt {
                return lin.forward(&xr);
            }
        }
        // CPU / non-CUDA fallback: dequant-to-dense matmul (sc-7702-style; keeps activations full-precision).
        let in_dtype = x.dtype();
        let w = self.dequant_dense()?.to_dtype(in_dtype)?;
        let bias = match &self.bias {
            Some(b) => Some(b.to_dtype(in_dtype)?),
            None => None,
        };
        Linear::new(w, bias).forward(&xr.to_dtype(in_dtype)?)
    }
}

impl QLinear {
    /// Wrap an already-loaded dense [`Linear`] (the loader built it from `{base}.weight` [+ `.bias`],
    /// or from the adapter-merged overlay) as a dense-base [`AdaptLinear`] — the logical `[out, in]` dims
    /// are recovered from the weight shape (`[out, in]`).
    pub fn dense(linear: Linear) -> Self {
        // `weight()` is `[out, in]`; read the dims before moving `linear` into the adapter.
        let (out_dim, in_dim) = {
            let d = linear.weight().dims();
            (d[0], d[1])
        };
        Self::Adapt(AdaptLinear::from_dense(linear, in_dim, out_dim))
    }

    /// Build a **packed** projection from the MLX packed triple (`wq` u32 codes + `scales` + `biases`,
    /// optional dense `bias`) at the tier's `group_size`, on `wq`'s device, via the shared group-size
    /// -aware loader, wrapped as a packed-base [`AdaptLinear`] (residual-capable — sc-11105). No dense
    /// weight is materialized. The logical `[out, in]` dims come from the `scales` shape (`[out,
    /// in/group_size]`).
    pub fn packed(
        wq: &Tensor,
        scales: &Tensor,
        biases: &Tensor,
        bias: Option<Tensor>,
        group_size: usize,
    ) -> Result<Self> {
        let device = wq.device().clone();
        let (out_dim, in_dim) = {
            let sd = scales.dims();
            (sd[0], sd[1] * group_size)
        };
        let q = shared::QLinear::from_packed_gs(wq, scales, biases, bias, group_size, &device)?;
        Ok(Self::Adapt(AdaptLinear::from_packed(q, in_dim, out_dim)))
    }

    /// Build an **INT8-ConvRot** projection (sc-9300 loader + sc-9601 rotation) straight from the
    /// checkpoint's stored parts: the `(N, K)` int8 codes `w_i8` of the **rotated** weight `W·R` (any
    /// dtype; narrowed at the int8 stage), the `[N]` per-output-row `weight_scale`, the ConvRot
    /// `group_size` (the regular-Hadamard order the export folded into the weight — `convrot_groupsize`,
    /// 256), the optional dense `bias`, and `device` — the model's resident **compute** device (where the
    /// activations live). No re-quantization. `group_size` must be a power of four and divide `K`; the
    /// forward applies the online `RHT(x)` so `RHT(x)·(W·R)ᵀ = x·Wᵀ`.
    ///
    /// **`device` is the compute device, not `w_i8.device()` (F-121 / sc-11208).** The loader
    /// materializes the int8 codes on the CPU to avoid holding them at 8× the size on the GPU
    /// ([`crate::loader::Weights::get_int8_codes`]), so `w_i8` is CPU-resident on a real CUDA render.
    /// The cuBLASLt IGEMM leg must therefore be built on the passed-in CUDA compute device — not gated on
    /// the CPU-resident codes' device — so a normal CUDA render takes the int8 IGEMM path (byte-identical
    /// to the pre-eager lazy build, which staged onto the activation's device);
    /// `Int8Linear::from_per_channel_parts` stages the CPU codes onto that device.
    pub fn convrot_int8(
        w_i8: Tensor,
        scale: Vec<f32>,
        group_size: usize,
        bias: Option<Tensor>,
        device: &Device,
    ) -> Result<Self> {
        let (n, k) = w_i8.dims2()?;
        if scale.len() != n {
            return Err(candle_gen::candle_core::Error::Msg(format!(
                "krea convrot: weight_scale len {} != weight rows {n}",
                scale.len()
            )));
        }
        if !candle_gen::quant::is_power_of_four(group_size) || !k.is_multiple_of(group_size) {
            return Err(candle_gen::candle_core::Error::Msg(format!(
                "krea convrot: group_size {group_size} must be a power of four dividing K ({k})"
            )));
        }
        // Build the cuBLASLt IGEMM leg eagerly on the model's resident COMPUTE device (F-121 /
        // sc-11208) so a cublasLt init failure is this typed error (where `?` is available), not an
        // `.expect()` panic on the first sampler forward. Gated on `device.is_cuda()`, NOT
        // `w_i8.device().is_cuda()`: the loader keeps the int8 codes on the CPU (VRAM), so gating on the
        // codes' device would leave `lt` None on every real CUDA render and drop into a cross-device
        // dequant-dense matmul. `from_per_channel_parts` stages the CPU codes onto `device`.
        #[cfg(feature = "cuda")]
        let lt = if device.is_cuda() {
            let cublas = std::sync::Arc::new(candle_gen::quant::CublasLt::new(device)?);
            Some(std::sync::Arc::new(
                candle_gen::quant::Int8Linear::from_per_channel_parts(
                    w_i8.clone(),
                    scale.clone(),
                    bias.clone(),
                    cublas,
                )?,
            ))
        } else {
            None
        };
        #[cfg(not(feature = "cuda"))]
        let _ = device;
        Ok(Self::ConvRotInt8(ConvRotInt8 {
            w_i8,
            scale,
            group_size,
            bias,
            rot: std::sync::OnceLock::new(),
            #[cfg(feature = "cuda")]
            lt,
        }))
    }

    /// `x·Wᵀ + b` (+ any additive residual). The `Adapt` arm delegates to the shared [`AdaptLinear`]
    /// (dense `candle_nn::Linear`, or the packed dequant-on-forward `QLinear` — sc-7702 — plus its
    /// residuals); int8-ConvRot to the cuBLASLt IGEMM (CUDA) or a dequant-dense matmul (CPU).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Adapt(l) => l.forward(x),
            Self::ConvRotInt8(l) => l.forward(x),
        }
    }

    /// Whether this projection loaded directly from the MLX-packed tier (its `Adapt` base is packed) —
    /// used by the loaders + tests to assert a packed tier fired the packed path (not a silent dense
    /// fallback).
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn is_packed(&self) -> bool {
        matches!(self, Self::Adapt(a) if a.is_packed())
    }

    /// Whether this projection loaded as an INT8-ConvRot int8 layer (sc-9300) — the detect-arm assertion
    /// (a ConvRot checkpoint's quantized surface fired the int8 path, not a silent dense fallback).
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn is_convrot_int8(&self) -> bool {
        matches!(self, Self::ConvRotInt8(_))
    }

    /// The inner residual-capable [`AdaptLinear`] (for the additive install to push a forward-time LoRA/
    /// LoKr residual — sc-11105), or `None` on an int8-ConvRot projection (never adaptable — the ConvRot
    /// lane rejects adapters). The `visit_adaptable_mut` walk yields this to the installer.
    pub fn as_adapt_mut(&mut self) -> Option<&mut AdaptLinear> {
        match self {
            Self::Adapt(a) => Some(a),
            Self::ConvRotInt8(_) => None,
        }
    }
}

impl Module for QLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        QLinear::forward(self, x)
    }
}

/// A token embedding that is **dense** (`candle_nn::Embedding`) or **packed** (loaded straight from the
/// MLX-packed tier's `embed_tokens` triple via the shared [`candle_gen::quant::QEmbedding`], sc-9411).
/// The Krea Qwen3-VL TE keeps `language_model.embed_tokens` **dense** in the hosted q4/q8 tiers (only
/// the layer projections are packed), so this closes the packed-detect surface for future-proofing +
/// the guard, while the hosted tier takes the dense arm.
pub enum QEmbedding {
    Dense(Embedding),
    Packed(shared::QEmbedding),
}

impl QEmbedding {
    /// Wrap an already-loaded dense [`Embedding`].
    pub fn dense(embedding: Embedding) -> Self {
        Self::Dense(embedding)
    }

    /// Build a **packed** embedding from the MLX packed triple at `group_size`, dequantized to
    /// `out_dtype` (the dense-path table dtype — dtype parity), on `wq`'s device.
    pub fn packed(
        wq: &Tensor,
        scales: &Tensor,
        biases: &Tensor,
        out_dtype: candle_gen::candle_core::DType,
        group_size: usize,
    ) -> Result<Self> {
        let device = wq.device().clone();
        Ok(Self::Packed(shared::QEmbedding::from_packed_dtype_gs(
            wq, scales, biases, &device, out_dtype, group_size,
        )?))
    }

    /// Index-select the embedding rows for `indexes`.
    pub fn forward(&self, indexes: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(e) => e.forward(indexes),
            Self::Packed(e) => e.forward(indexes),
        }
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn is_packed(&self) -> bool {
        matches!(self, Self::Packed(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::{DType, Device};

    /// The Krea tier's MLX quant group size (64, read from `config.json` at load — pinned here as the
    /// hosted value so the fixtures exercise the real tier's grouping).
    const G: usize = 64;

    /// Test-side MLX Q4 packer at group `G`: per-element 4-bit codes → MLX u32 words (LSB-first
    /// nibbles). Returns `(wq [out, in/8] u32, scales [out, in/G], biases [out, in/G], affine grid)` —
    /// the exact packed-parts fixture the loaders consume plus the affine grid they reproduce.
    fn q4_packed(out_dim: usize, in_dim: usize) -> (Tensor, Tensor, Tensor, Vec<f32>) {
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
        let wq = Tensor::from_vec(words, (out_dim, in_dim / 8), &dev).unwrap();
        let s = Tensor::from_vec(scales, (out_dim, gpr), &dev).unwrap();
        let b = Tensor::from_vec(biases, (out_dim, gpr), &dev).unwrap();
        (wq, s, b, grid)
    }

    fn cosine(a: &Tensor, b: &Tensor) -> f32 {
        let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (x, y) in a.iter().zip(&b) {
            dot += (*x as f64) * (*y as f64);
            na += (*x as f64) * (*x as f64);
            nb += (*y as f64) * (*y as f64);
        }
        (dot / (na.sqrt() * nb.sqrt() + 1e-12)) as f32
    }

    /// A group-64 packed `QLinear` forward matches a dense linear built from the SAME affine grid the
    /// pack represents — bit-exact (the Q4 → Q4_1 repack is lossless, both forwards dequant-to-dense
    /// -matmul). Uses a `to_out.0`-style output-projection shape.
    #[test]
    fn packed_qlinear_matches_dense_grid() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (128usize, 256usize);
        let (wq, s, b, grid) = q4_packed(out_dim, in_dim);

        let packed = QLinear::packed(&wq, &s, &b, None, G)?;
        assert!(packed.is_packed(), "group-64 triple ⇒ packed load");
        let dense = QLinear::dense(Linear::new(
            Tensor::from_vec(grid, (out_dim, in_dim), &dev)?,
            None,
        ));
        assert!(!dense.is_packed());

        let x = Tensor::randn(0f32, 1f32, (4, in_dim), &dev)?;
        let cos = cosine(&packed.forward(&x)?, &dense.forward(&x)?);
        assert!(
            cos > 0.99999,
            "group-64 packed vs affine-grid cosine {cos:.6}"
        );
        Ok(())
    }

    /// A group-64 packed `QEmbedding` reproduces the affine grid rows exactly, dequantized to the
    /// requested dtype (the future-proof path for a tier that ever packs `embed_tokens`).
    #[test]
    fn packed_qembedding_matches_dense_grid() -> Result<()> {
        let dev = Device::Cpu;
        let (vocab, hidden) = (64usize, 128usize);
        let (wq, s, b, grid) = q4_packed(vocab, hidden);

        let packed = QEmbedding::packed(&wq, &s, &b, DType::F32, G)?;
        assert!(packed.is_packed(), "group-64 triple ⇒ packed embedding");
        let dense = QEmbedding::dense(Embedding::new(
            Tensor::from_vec(grid, (vocab, hidden), &dev)?,
            hidden,
        ));

        let idx = Tensor::from_vec(vec![0u32, 5, 63, 12, 5], (5,), &dev)?;
        let p = packed.forward(&idx)?;
        let d = dense.forward(&idx)?;
        let dev_max = (p.sub(&d)?).abs()?.max_all()?.to_scalar::<f32>()?;
        assert_eq!(
            dev_max, 0.0,
            "group-64 packed embedding deviates from the grid"
        );
        Ok(())
    }

    /// F-121 (sc-11208): `convrot_int8` now builds the cuBLASLt IGEMM leg eagerly (where `?` is
    /// available) rather than in a lazy `get_or_init(|| … .expect())` on the sampler thread. On CPU
    /// that eager leg is skipped, so a valid build must construct + forward without panicking, and
    /// malformed parts must be typed errors from the constructor (never a `.expect()`/panic).
    #[test]
    fn convrot_int8_constructor_is_typed_error_not_panic() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (64usize, 128usize);
        let mut wv = vec![0f32; out_dim * in_dim];
        for o in 0..out_dim {
            for j in 0..in_dim {
                wv[o * in_dim + j] = ((o * 7 + j * 3) % 51) as f32 / 25.0 - 1.0;
            }
        }
        // Canonical weight → rotate → per-row int8: the on-disk ConvRot granularity.
        let w = Tensor::from_vec(wv, (out_dim, in_dim), &dev)?;
        let r = candle_gen::quant::regular_hadamard(G, &dev)?;
        let rw = candle_gen::quant::convrot_rotate(&w, &r)?;
        let pc = candle_gen::quant::quantize_weight_int8_per_channel(&rw)?;

        // Valid build + forward (CPU compute → dequant-dense leg): must not panic.
        let lin = QLinear::convrot_int8(pc.q.clone(), pc.scale.clone(), G, None, &dev)?;
        assert!(lin.is_convrot_int8());
        let x = Tensor::randn(0f32, 1f32, (4, in_dim), &dev)?;
        assert_eq!(lin.forward(&x)?.dims(), &[4, out_dim]);

        // Malformed parts ⇒ typed errors from the constructor, not panics.
        assert!(
            QLinear::convrot_int8(pc.q.clone(), vec![1.0; out_dim + 1], G, None, &dev).is_err(),
            "scale-len mismatch must be a typed error"
        );
        assert!(
            QLinear::convrot_int8(pc.q.clone(), pc.scale.clone(), G + 1, None, &dev).is_err(),
            "non-power-of-four group_size must be a typed error"
        );
        Ok(())
    }

    /// F-121 device regression (sc-11208): the production ConvRot path materializes the int8 codes on
    /// the **CPU** (the loader's `get_int8_codes`, to avoid 8× VRAM), so `convrot_int8` must build the
    /// cuBLASLt IGEMM leg on the model's resident **compute** device (the activation's CUDA device), NOT
    /// the CPU-resident codes' device. This reproduces that exact split — CPU codes + a CUDA compute
    /// device + a CUDA activation — and asserts the int8 IGEMM leg (not a cross-device dequant-dense
    /// fallback, which the buggy `w_i8.device().is_cuda()` gate would have hit) runs and matches the CPU
    /// dequant-dense reference within int8 tolerance. A CPU-only test cannot catch this — on CPU `lt` is
    /// legitimately `None`. Skips cleanly when no CUDA device is present.
    #[cfg(feature = "cuda")]
    #[test]
    fn convrot_int8_cuda_forward_with_cpu_codes_matches_dequant_dense() -> Result<()> {
        let cuda = match Device::cuda_if_available(0) {
            Ok(d @ Device::Cuda(_)) => d,
            _ => {
                eprintln!("[sc-11208] no CUDA device; skipping ConvRot int8 CUDA forward gate");
                return Ok(());
            }
        };
        let cpu = Device::Cpu;
        // cuBLASLt int8 needs K and N multiples of 16; group_size (a power of four) must divide K.
        let (out_dim, in_dim) = (64usize, 128usize);
        let mut wv = vec![0f32; out_dim * in_dim];
        for o in 0..out_dim {
            for j in 0..in_dim {
                wv[o * in_dim + j] = ((o * 7 + j * 3) % 51) as f32 / 25.0 - 1.0;
            }
        }
        // Canonical weight → rotate → per-row int8, codes left on the CPU exactly as the loader's
        // `get_int8_codes` materializes them in production.
        let w = Tensor::from_vec(wv, (out_dim, in_dim), &cpu)?;
        let r = candle_gen::quant::regular_hadamard(G, &cpu)?;
        let rw = candle_gen::quant::convrot_rotate(&w, &r)?;
        let pc = candle_gen::quant::quantize_weight_int8_per_channel(&rw)?;
        let w_i8_cpu = pc.q;
        assert!(
            !w_i8_cpu.device().is_cuda(),
            "fixture must mirror production: int8 codes materialized on the CPU"
        );

        // Compute device = CUDA (the resident / activation device), codes on CPU: the production split.
        let lin_cuda = QLinear::convrot_int8(w_i8_cpu.clone(), pc.scale.clone(), G, None, &cuda)?;
        assert!(lin_cuda.is_convrot_int8());
        let x = Tensor::randn(0f32, 1f32, (8, in_dim), &cuda)?;
        // Must take the int8 IGEMM leg (built on `cuda`), NOT fall through to a cross-device
        // dequant-dense matmul (CPU weight × CUDA activation), which is the F-121 bug this guards.
        let y_cuda = lin_cuda
            .forward(&x)?
            .to_dtype(DType::F32)?
            .to_device(&cpu)?;

        // Reference: the SAME parts on a CPU compute device (dequant-dense leg), full-precision
        // activation — the int8-tolerance yardstick for the IGEMM path.
        let lin_cpu = QLinear::convrot_int8(w_i8_cpu, pc.scale.clone(), G, None, &cpu)?;
        let y_ref = lin_cpu.forward(&x.to_device(&cpu)?)?.to_dtype(DType::F32)?;

        let cos = cosine(&y_cuda, &y_ref);
        let num = (y_cuda.sub(&y_ref)?)
            .sqr()?
            .mean_all()?
            .to_scalar::<f32>()?
            .sqrt();
        let den = y_ref
            .sqr()?
            .mean_all()?
            .to_scalar::<f32>()?
            .sqrt()
            .max(1e-6);
        let rel = num / den;
        assert!(
            cos > 0.99 && rel < 0.1,
            "CUDA int8 IGEMM vs CPU dequant-dense: cosine {cos:.5}, rel-RMS {rel:.4}"
        );
        Ok(())
    }
}
