//! Vendored, **i32-overflow-safe** FLUX.1 VAEs (sc-11154 / F-081).
//!
//! Faithful copies of candle-transformers `flux::autoencoder` (the BFL/native `AutoEncoder`, used by
//! the dense txt2img + control/IP-Adapter encode paths) and `z_image::vae` (the diffusers
//! `AutoEncoderKL` — FLUX.1's packed/bf16 turnkey VAE shares z-image's diffusers layout) at the
//! workspace candle pin, vendored solely so the single unguarded `scaled_dot_product_attention` can
//! route through the shared [`candle_gen::sdpa_budgeted_flat`].
//!
//! The stock upstream mid-block is a single-head spatial self-attention that materializes an unchunked
//! `[B, 1, H·W, H·W]` scores tensor. At the advertised 2048² decode `H·W = 256² = 65536`, so
//! `65536² ≈ 4.3e9 > i32::MAX`; candle's CUDA kernels index scores with i32 and silently corrupt the
//! tail — a garbage decode after a numerically-correct denoise. Every other line is byte-faithful with
//! the upstream module; CPU forward-parity tests ([`native::parity_tests`], [`diffusers::parity_tests`])
//! pin both copies to the stock modules so the vendoring changed nothing numerically, and the budgeted
//! attention is itself byte-identical to the stock single pass below `ATTN_SCORES_BUDGET`.
pub mod diffusers;
pub mod native;
