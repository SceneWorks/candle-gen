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

use candle_gen::candle_core::{DType, Result, Tensor};
use candle_gen::candle_nn::{Embedding, Linear, Module};
use candle_gen::quant as shared;

/// A Linear projection that is **dense** (the loaded bf16 weight, possibly adapter-merged via the
/// overlay), **packed** (loaded straight from the MLX-packed tier via the shared
/// [`candle_gen::quant::QLinear`], sc-9411), or **int8-ConvRot** (a community INT8-ConvRot checkpoint's
/// per-output-channel int8 projection, sc-9300). Built dense ([`Self::dense`]), packed
/// ([`Self::packed`]), or int8 ([`Self::convrot_int8`]); every forward computes `x·Wᵀ + b`.
pub enum QLinear {
    Dense(Linear),
    /// Loaded directly from the MLX-packed tier through the shared module — the resident `Q4_1`/`Q8_0`
    /// weight **dequantizes-on-forward** into a dense matmul (sc-7702, *not* the int8 `QMatMul` fast
    /// path).
    Packed(shared::QLinear),
    /// A community **INT8-ConvRot** projection (sc-9300): the checkpoint ships `(N, K)` int8 codes with
    /// a rotation folded into the stored weight + a `[N]` per-output-row `weight_scale`. On CUDA the
    /// forward is a cuBLASLt IGEMM + per-row dequant ([`candle_gen::quant::Int8Linear`]); off-CUDA
    /// (CPU tests / Metal) it dequantizes the weight to a dense matmul.
    ///
    /// **A/B finding (sc-9300): this alone does NOT reconstruct `X·Wᵀ`.** The stored int8 weight is a
    /// *rotated* weight `R·W` (verified: dequantized `blocks.0.attn.wq` has cosine ≈ 0.07 with the
    /// canonical `to_q`), so the matching **online activation rotation** `x → x·R` must run before the
    /// IGEMM for `x·(R·W)ᵀ = x·Wᵀ` to hold. Without it the render is pure noise (the A/B render, PSNR
    /// ≈ 8 dB vs bf16). That online rotation is the arXiv 2512.03673 / ComfyUI ConvRot leg the story
    /// scoped out (GPL-3, clean-room reimplementation) — the follow-up sc-9601. This arm is the correct
    /// *loader + per-channel int8 compute*; the missing rotation is what makes the consume path coherent.
    ConvRotInt8(ConvRotInt8),
}

/// The stored parts of an INT8-ConvRot projection (sc-9300): the `(N, K)` int8 codes (on the CPU as
/// `I64`, staged to a resident device `i8` inside `Int8Linear`), the `[N]` per-output-row dequant
/// scale, and the optional dense bias (ConvRot Krea projections are bias-free, but the field keeps the
/// type general). On CUDA a per-channel `Int8Linear` is built lazily on first forward and cached; the
/// CPU fallback dequantizes `w[o, :] = q[o, :] · scale[o]`. The stored weight is rotated, so neither
/// path reconstructs `X·Wᵀ` without the online `x·R` leg (the sc-9300 A/B NO-GO).
pub struct ConvRotInt8 {
    w_i8: Tensor,
    scale: Vec<f32>,
    bias: Option<Tensor>,
    #[cfg(feature = "cuda")]
    lt: std::sync::OnceLock<std::sync::Arc<candle_gen::quant::Int8Linear>>,
}

impl ConvRotInt8 {
    /// The dense `(N, K)` f32 weight the int8 codes + per-row scale represent (`w[o,:] = q[o,:]·s[o]`).
    fn dequant_dense(&self) -> Result<Tensor> {
        let n = self.w_i8.dims2()?.0;
        let scale_col = Tensor::from_vec(self.scale.clone(), (n, 1), self.w_i8.device())?;
        self.w_i8.to_dtype(DType::F32)?.broadcast_mul(&scale_col)
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if x.device().is_cuda() {
            let lin = self.lt.get_or_init(|| {
                std::sync::Arc::new(
                    candle_gen::quant::Int8Linear::from_per_channel_parts(
                        self.w_i8.clone(),
                        self.scale.clone(),
                        self.bias.clone(),
                        std::sync::Arc::new(
                            candle_gen::quant::CublasLt::new(x.device())
                                .expect("cublasLt handle for int8 convrot"),
                        ),
                    )
                    .expect("build Int8Linear from convrot parts"),
                )
            });
            return lin.forward(x);
        }
        // CPU / non-CUDA fallback: dequant-to-dense matmul (sc-7702-style; keeps activations full-precision).
        let in_dtype = x.dtype();
        let w = self.dequant_dense()?.to_dtype(in_dtype)?;
        let bias = match &self.bias {
            Some(b) => Some(b.to_dtype(in_dtype)?),
            None => None,
        };
        Linear::new(w, bias).forward(x)
    }
}

impl QLinear {
    /// Wrap an already-loaded dense [`Linear`] (the loader built it from `{base}.weight` [+ `.bias`],
    /// or from the adapter-merged overlay).
    pub fn dense(linear: Linear) -> Self {
        Self::Dense(linear)
    }

    /// Build a **packed** projection from the MLX packed triple (`wq` u32 codes + `scales` + `biases`,
    /// optional dense `bias`) at the tier's `group_size`, on `wq`'s device, via the shared group-size
    /// -aware loader. No dense weight is materialized.
    pub fn packed(
        wq: &Tensor,
        scales: &Tensor,
        biases: &Tensor,
        bias: Option<Tensor>,
        group_size: usize,
    ) -> Result<Self> {
        let device = wq.device().clone();
        Ok(Self::Packed(shared::QLinear::from_packed_gs(
            wq, scales, biases, bias, group_size, &device,
        )?))
    }

    /// Build an **INT8-ConvRot** projection (sc-9300) straight from the checkpoint's stored parts: the
    /// `(N, K)` int8 codes `w_i8` (any dtype; narrowed at the int8 stage), the `[N]` per-output-row
    /// `weight_scale`, and the optional dense `bias`. No re-quantization. **The stored weight is
    /// rotated** (`R·W`); reconstructing `X·Wᵀ` needs the online `x·R` leg, which this arm does NOT
    /// apply (the sc-9300 A/B NO-GO — see [`Self::ConvRotInt8`]).
    pub fn convrot_int8(w_i8: Tensor, scale: Vec<f32>, bias: Option<Tensor>) -> Result<Self> {
        let n = w_i8.dims2()?.0;
        if scale.len() != n {
            return Err(candle_gen::candle_core::Error::Msg(format!(
                "krea convrot: weight_scale len {} != weight rows {n}",
                scale.len()
            )));
        }
        Ok(Self::ConvRotInt8(ConvRotInt8 {
            w_i8,
            scale,
            bias,
            #[cfg(feature = "cuda")]
            lt: std::sync::OnceLock::new(),
        }))
    }

    /// `x·Wᵀ + b`. Dense delegates to `candle_nn::Linear`; packed to the shared dequant-on-forward
    /// `QLinear` (sc-7702); int8-ConvRot to the cuBLASLt IGEMM (CUDA) or a dequant-dense matmul (CPU).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(l) => l.forward(x),
            Self::Packed(l) => l.forward(x),
            Self::ConvRotInt8(l) => l.forward(x),
        }
    }

    /// Whether this projection loaded directly from the MLX-packed tier (the packed path) — used by the
    /// loaders + tests to assert a packed tier fired the packed path (not a silent dense fallback).
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn is_packed(&self) -> bool {
        matches!(self, Self::Packed(_))
    }

    /// Whether this projection loaded as an INT8-ConvRot int8 layer (sc-9300) — the detect-arm assertion
    /// (a ConvRot checkpoint's quantized surface fired the int8 path, not a silent dense fallback).
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn is_convrot_int8(&self) -> bool {
        matches!(self, Self::ConvRotInt8(_))
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
}
