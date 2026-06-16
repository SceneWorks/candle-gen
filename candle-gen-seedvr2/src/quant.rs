//! SeedVR2 DiT load-time Q4/Q8 quantization mapping (sc-5927) — the candle twin of
//! `mlx-gen-seedvr2`'s group-wise-affine `quantize()` path (sc-5198), built on **candle-core's GGUF
//! `QMatMul`** (the same first-class quant seam [`candle-gen-lens`](../../candle-gen-lens) uses, rather
//! than a bespoke affine packer). Quantization is **Linear-only** by construction (the SeedVR2 DiT has
//! no convs; the VAE stays dense) and skips any Linear whose contraction (`in_features`) is not a
//! multiple of the 32-wide GGUF block — matching the reference predicate, which leaves `vid_in.proj`
//! (in=132) dense. [`crate::dit::Seedvr2Transformer::quantize`] folds each DiT Linear in place after
//! the (dense) weights have loaded; fp16 stays the default (sc-5928 picks the level per request).
//!
//! The MLX reference uses a group-64 affine quant; candle's `Q4_0`/`Q8_0` are 32-block legacy quants.
//! Both are near-lossless at int8 / coherent at int4 — we match the *behavior* (the bar is "near-
//! lossless vs fp16 on CUDA"), not bit-exact MLX weights (there is no pre-Rust reference to match).

use candle_gen::candle_core::quantized::GgmlDType;
use candle_gen::gen_core::Quant;

/// GGUF block size for `Q4_0`/`Q8_0` (the candle-core default legacy quants). A Linear is quantized
/// only when its `in_features` divides this; otherwise it stays dense (the reference predicate).
pub const QUANT_BLOCK: usize = 32;

/// The GGUF block type a [`Quant`] level maps to — int8 → `Q8_0` (near-lossless), int4 → `Q4_0`.
/// Shared single source of truth for the family's `Quant → GgmlDType` mapping.
pub fn ggml_dtype(quant: Quant) -> GgmlDType {
    match quant {
        Quant::Q4 => GgmlDType::Q4_0,
        Quant::Q8 => GgmlDType::Q8_0,
    }
}
