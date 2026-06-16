//! # candle-gen-seedvr2
//!
//! The **SeedVR2** provider crate for [`candle-gen`](candle_gen) — the Windows/CUDA sibling of
//! `mlx-gen-seedvr2` (epic 4811 / sc-5157). A native-candle port of the ByteDance one-step
//! diffusion-transformer super-resolution upscaler:
//!
//! 1. **DiT** — a dual-stream MMDiT with adaptive **spatiotemporal window attention**
//!    (`window=(T,H,W)=(4,3,3)`, shifted on odd layers), 3D axial RoPE, QK-norm, SwiGLU, AdaLN.
//! 2. **3D causal video VAE** — `CausalConv3d` (candle has no conv3d → conv2d temporal-sum, see
//!    [`conv3d`]) encoder/decoder with `temporal_down/up_blocks=2`, GroupNorm, per-frame attention.
//! 3. **One-step Euler** + a precomputed negative-prompt embedding (bundled, no runtime text encoder).
//!
//! **This slice (sc-5157): the 3B engine + image-mode parity.** Video mode (the 5-D temporal pass)
//! is sc-5926; 7B + int8/int4 quant is sc-5927; worker wiring / gating is sc-5928.

pub mod config;
pub mod conv3d;
pub mod nn;
pub mod vae;
pub mod weights;
