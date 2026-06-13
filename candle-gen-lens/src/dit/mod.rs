//! The Lens denoising **DiT** (sc-5112) — a 48-layer dual-stream MMDiT with joint image+text
//! attention, complex axial RoPE on both streams, and SwiGLU MLPs. A candle port of
//! `mlx-gen-lens/src/dit/` (which itself ports `_vendor/lens/transformer.py::LensTransformer2DModel`),
//! architecturally a near-twin of `candle-gen-qwen-image`'s MMDiT — the RoPE, joint attention, AdaLN
//! modulation, and `AdaLayerNormContinuous` all follow that seam; the Lens-specific pieces are the
//! **multi-layer text front-end**, the **fused `img_qkv`/`txt_qkv`** projections, the **`[img, txt]`**
//! join order, the **SwiGLU GateMLP**, and the **biased `norm_out`**.
//!
//! `[batch, seq, dim]` tensors throughout. The model consumes already-patchified image latents
//! `[B, img_len, 128]` plus the 4 captured gpt-oss text-feature layers `[B, txt_len, 2880]` and
//! predicts the patch-space velocity `[B, img_len, 128]`; patch/unpatch + the sampler are sc-5114/5115.

pub mod attention;
pub mod block;
pub mod rope;
#[allow(clippy::module_inception)]
pub mod transformer;

pub use attention::LensJointAttention;
pub use block::LensTransformerBlock;
pub use rope::LensRope3d;
pub use transformer::{LensDitConfig, LensTransformer};
