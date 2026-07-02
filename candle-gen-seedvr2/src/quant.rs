//! SeedVR2 DiT load-time Q4/Q8 quantization mapping (sc-5927) — the candle twin of
//! `mlx-gen-seedvr2`'s group-wise-affine `quantize()` path (sc-5198), built on **candle-core's GGUF
//! `QMatMul`**. Quantization is **Linear-only** by construction (the SeedVR2 DiT has no convs; the VAE
//! stays dense) and skips any Linear whose contraction (`in_features`) is not a multiple of the 32-wide
//! GGUF block — matching the reference predicate, which leaves `vid_in.proj` (in=132) dense.
//! [`crate::dit::Seedvr2Transformer::quantize`] folds each DiT Linear in place after the (dense) weights
//! have loaded; fp16 stays the default (sc-5928 picks the level per request).
//!
//! **F-025 / sc-9005:** the `Dense|Quantized` Linear seam this crate carried (`dit::Linear`) is now the
//! ONE shared [`candle_gen::quant::QLinear`] (SeedVR2 was one of four drifted copies). The
//! `Quant → GgmlDType` mapping and the GGUF block size likewise live there now; re-export them under the
//! crate-local names so `dit.rs` and any external reference resolve unchanged.

pub use candle_gen::quant::{ggml_dtype, QUANT_BLOCK};
