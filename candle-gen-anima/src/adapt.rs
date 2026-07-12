//! Anima's residual-capable linear — **hoisted into the shared core** (sc-11091, epic 10765). The
//! forward-time additive LoRA / structured-LoKr wrapper that used to live here (sc-10640) is now the
//! one shared [`candle_gen::quant::AdaptLinear`], collapsed together with `candle-gen-wan`'s copy
//! (sc-10094) and adopted by qwen-image-edit. Anima re-exports it verbatim so every DiT + conditioner
//! projection (`crate::transformer` / `crate::conditioner`) and the residual installer
//! (`crate::adapters::install_anima_residuals`) keep referencing `crate::adapt::{AdaptLinear,
//! LokrFactors}` unchanged. The mechanism, the packed-Kronecker vec-trick, and the full test suite now
//! live in `candle-gen/src/quant/adapt.rs`.

pub use candle_gen::quant::{AdaptLinear, LokrFactors};
