//! Vendored, training-adapted SDXL UNet stack (sc-5165).
//!
//! A faithful copy of candle-transformers `stable_diffusion::{unet_2d, unet_2d_blocks, attention,
//! resnet, embeddings}` at the workspace candle pin (`65ecb58`), vendored because the native
//! LoRA/LoKr trainer needs two things the stock (opaque, all-private-fields) UNet does not permit:
//!
//!  1. a **trainable LoRA/LoKr residual** spliced into the attention projections
//!     (`to_q`/`to_k`/`to_v`/`to_out.0`) — the stock `CrossAttention` builds them from frozen
//!     `nn::Linear` with no seam; and
//!  2. ownership of the **block-by-block forward** so activations can be checkpointed (recomputed in
//!     the backward pass) — required because candle's fused attention (flash/sdpa) has no backward,
//!     forcing the materialized O(seq²) math attention whose activations must be bounded.
//!
//! The flash-attn path is dropped (non-differentiable; see [`attention`]); training uses the math /
//! sliced attention. VAE + CLIP stay **stock** (frozen at train time, no adapter, no checkpointing).
//! Inference is unchanged — it still runs the stock candle-transformers UNet; a CPU forward-parity
//! test pins this vendored copy to that stock module so the two never drift.
//!
//! As vendored this is a byte-faithful replica; the LoRA seam + checkpoint boundaries are layered on
//! in subsequent slices of this story.
mod attention;
mod controlnet;
mod conv;
mod embeddings;
mod resnet;
mod unet_2d;
mod unet_2d_blocks;
mod vae_encode;

pub use controlnet::{ControlNet, ControlNetConfig, ControlResiduals};
// The canonical SDXL UNet sub-config, shared by the InstantID UNet loader (sc-5491) and the Kolors
// IP-Adapter provider (sc-5488), which loads the SDXL-family Kolors UNet into this vendored stack.
pub use controlnet::sdxl_unet_config;
pub use unet_2d::{BlockConfig, UNet2DConditionModel, UNet2DConditionModelConfig};
pub use vae_encode::VaeMomentsEncoder;

#[cfg(test)]
mod parity_tests {
    //! Pin the vendored UNet to the stock candle-transformers UNet: built from the *same*
    //! `VarMap`-backed weights with no adapter installed, the two must produce bit-identical forward
    //! output. This is the regression guard that the vendoring (candle::→candle_core::, the `conv`
    //! shim, the flash stub, the `LoraLinear` swap) changed nothing numerically.
    use super::{BlockConfig, UNet2DConditionModel, UNet2DConditionModelConfig};
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{VarBuilder, VarMap};
    use candle_transformers::models::stable_diffusion::unet_2d as stock;

    /// A small SDXL-shaped config (one cross-attn block + one basic block + cross-attn mid) that
    /// exercises every vendored code path cheaply on CPU.
    fn blocks() -> Vec<BlockConfig> {
        vec![
            BlockConfig {
                out_channels: 32,
                use_cross_attn: Some(1),
                attention_head_dim: 8,
            },
            BlockConfig {
                out_channels: 64,
                use_cross_attn: None,
                attention_head_dim: 8,
            },
        ]
    }

    fn vendored_cfg() -> UNet2DConditionModelConfig {
        UNet2DConditionModelConfig {
            center_input_sample: false,
            flip_sin_to_cos: true,
            freq_shift: 0.,
            blocks: blocks(),
            layers_per_block: 1,
            downsample_padding: 1,
            mid_block_scale_factor: 1.,
            norm_num_groups: 32,
            norm_eps: 1e-5,
            cross_attention_dim: 64,
            use_linear_projection: false,
        }
    }

    fn stock_cfg() -> stock::UNet2DConditionModelConfig {
        stock::UNet2DConditionModelConfig {
            center_input_sample: false,
            flip_sin_to_cos: true,
            freq_shift: 0.,
            blocks: blocks()
                .into_iter()
                .map(|b| stock::BlockConfig {
                    out_channels: b.out_channels,
                    use_cross_attn: b.use_cross_attn,
                    attention_head_dim: b.attention_head_dim,
                })
                .collect(),
            layers_per_block: 1,
            downsample_padding: 1,
            mid_block_scale_factor: 1.,
            norm_num_groups: 32,
            norm_eps: 1e-5,
            cross_attention_dim: 64,
            sliced_attention_size: None,
            use_linear_projection: false,
        }
    }

    #[test]
    fn vendored_unet_matches_stock_forward() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        // The vendored UNet is built first, populating the VarMap with random weights; the stock UNet
        // then reads the SAME parameters (identical names/shapes), so any output difference is a
        // forward-logic difference, not a weight difference.
        let vendored = UNet2DConditionModel::new(vb.clone(), 4, 4, false, vendored_cfg()).unwrap();
        let stock_unet = stock::UNet2DConditionModel::new(vb, 4, 4, false, stock_cfg()).unwrap();

        let x = Tensor::randn(0f32, 1f32, (1, 4, 16, 16), &dev).unwrap();
        let ehs = Tensor::randn(0f32, 1f32, (1, 7, 64), &dev).unwrap();
        let y_v = vendored.forward(&x, 10.0, &ehs).unwrap();
        let y_s = stock_unet.forward(&x, 10.0, &ehs).unwrap();

        assert_eq!(y_v.dims(), y_s.dims());
        let diff = (y_v - y_s)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff < 1e-5, "vendored UNet diverged from stock by {diff}");
    }
}
