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
//! **sc-11778** adds the **Linear-DiT trunk** ([`transformer::SanaTransformer`]) — the ReLU
//! linear-attention DiT blocks (reusing [`dc_ae::relu_linear_attention`]), the `GLUMBConv` Mix-FFN
//! (3×3 depthwise conv, reusing [`dc_ae::glu_mbconv_core`]), NoPE, and the adaLN-single timestep /
//! caption conditioning (base SANA-1.6B + the SANA-Sprint guidance-embed / qk-norm superset). Its
//! `[B, 32, H, W]` noise prediction feeds [`dc_ae::DcAeDecoder::decode`] directly.
//!
//! **sc-11779** adds the **text conditioning** ([`text_encoder::SanaTextEncoder`]) — a thin wrapper
//! that REUSES PiD's native gemma-2-2b-it CHI caption encoder ([`candle_gen_pid::CaptionEncoder`])
//! via the shared [`candle_gen_pid::CaptionEncoder::with_chi_prompt`] seam, differing from PiD only
//! in the CHI template's quoting around `Enhanced prompt`. Prompt → `[1, 300, 2304]` gemma
//! last-hidden caption embedding feeding the trunk's `attn2` cross-attention. Mirrors mlx-gen-sana's
//! sc-8488 (mlx-gen #614).
//!
//! **sc-11780** assembles the end-to-end base txt2img [`pipeline`] (TE → trunk → DC-AE, driven by
//! candle's unified flow scheduler, static shift 3.0, true CFG) and the gen-core [`model`] adapter
//! (registered under `sana_1600m`), mirroring mlx-gen-sana's sc-8489. The CFG-free SCM/Sprint distill
//! (mlx sc-8490) is not ported on the candle side.

pub mod config;
pub mod dc_ae;
pub mod model;
pub mod pipeline;
pub mod text_encoder;
pub mod transformer;

pub use config::{BlockType, DcAeConfig, SanaTransformerConfig};
pub use dc_ae::{DcAeDecoder, DcAeEncoder};
pub use model::{descriptor, load, MODEL_ID};
pub use pipeline::{SanaGenerateRequest, SanaPipeline};
pub use text_encoder::{SanaTextEncoder, MAX_SEQUENCE_LENGTH, SANA_CHI_PROMPT};
pub use transformer::SanaTransformer;

/// Force the linker to keep this crate's `register_generators!` submission (epic 3720): a binary that
/// only resolves the model by id through the gen-core registry never names a symbol here, so without a
/// referenced item the whole object — and its `inventory::submit!` — can be dead-stripped. Call this
/// once from `main` (as the sibling candle providers do) before `gen_core::registry::load`.
pub fn force_link() {}
