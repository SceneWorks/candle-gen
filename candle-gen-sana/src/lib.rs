//! # candle-gen-sana
//!
//! SANA (NVlabs) provider crate for [`candle-gen`] — the Windows/CUDA + Linux sibling of
//! `mlx-gen-sana` (mlx-gen #612), epic 11776.
//!
//! **Gating spike sc-11777** delivers the two hard primitives whose candle/CUDA feasibility was the
//! GO/NO-GO question for a native SANA port:
//!
//!  - the **DC-AE** (deep-compression autoencoder) **f32 image decoder** — 6-stage conv decoder,
//!    `ResBlock`s, `EfficientViTBlock`s, `ConvPixelShuffle` up-sampling, trimmed-RMS norm (`trms2d`),
//!    SiLU — a faithful component port of diffusers `AutoencoderDC`
//!    (`mit-han-lab/dc-ae-f32c32-sana-1.0`, the autoencoder behind SANA-1.6B 1024px); and
//!  - the **EfficientViT GLU** ReLU-**linear**-attention block (O(N), softmax-free) — the *shared hard
//!    primitive* the SANA Linear-DiT trunk (story 2) reuses, so it is written once here
//!    ([`dc_ae::relu_linear_attention`] + the `LinearAttn` block).
//!
//! A compact symmetric **encoder** ([`dc_ae::DcAeEncoder`]) rides along only far enough for a
//! round-trip reconstruction check; the decoder is the parity deliverable. See [`dc_ae`] for the
//! block-by-block port and the port notes (NCHW-native, f32).
//!
//! The full native-SANA pipeline (Linear-DiT trunk, flow / SCM schedulers, Gemma text encoder, e2e
//! wiring, gen-core registration) lands in the sibling stories of epic 11776, mirroring
//! mlx-gen-sana's sc-8487..8490.

pub mod config;
pub mod dc_ae;

pub use config::{BlockType, DcAeConfig};
pub use dc_ae::{DcAeDecoder, DcAeEncoder};
