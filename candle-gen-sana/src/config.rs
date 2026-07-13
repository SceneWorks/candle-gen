//! DC-AE (deep-compression autoencoder) configuration — gating spike sc-11777 (epic 11776).
//!
//! Values mirror the diffusers `AutoencoderDC` config for `mit-han-lab/dc-ae-f32c32-sana-1.0` (the
//! autoencoder behind SANA-1.6B 1024px), matching the mlx-gen-sana port (mlx-gen #612) this crate is
//! the Windows/CUDA sibling of. The **decoder** is the spike's GO/NO-GO deliverable; a compact
//! symmetric **encoder** rides along only far enough for a round-trip reconstruction check.

/// Per-stage block kind. The SANA-1.0 autoencoder runs `ResBlock` in the three shallow (high-res)
/// stages and `EfficientViTBlock` (ReLU linear attention) in the three deep (low-res) stages.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BlockType {
    Res,
    EfficientVit,
}

/// DC-AE hyper-parameters. Stored stage order is shallow→deep (index 0 = 128-channel/full-res stage
/// … index 5 = 1024-channel/lowest-res stage), matching the on-disk `decoder.up_blocks.{i}` /
/// `encoder.down_blocks.{i}` numbering. Decode iterates deep→shallow; encode iterates shallow→deep.
#[derive(Clone, Debug)]
pub struct DcAeConfig {
    pub in_channels: i32,
    pub latent_channels: i32,
    pub attention_head_dim: i32,
    pub block_out_channels: Vec<i32>,
    pub layers_per_block: Vec<i32>,
    pub block_types: Vec<BlockType>,
    /// One `kernel_size` per multiscale QKV projection in the EfficientViT stages (`[5]` for SANA-1.0).
    pub qkv_multiscales: Vec<i32>,
    /// RMS-norm epsilon (`1e-5` throughout the autoencoder).
    pub norm_eps: f32,
    /// Linear-attention denominator epsilon (`1e-15`).
    pub attn_eps: f32,
    /// VAE latent scaling factor (`z_decode = z / scaling_factor`). Applied by the caller, not the
    /// decoder, mirroring diffusers `Decoder.forward` (which receives an already-scaled latent).
    pub scaling_factor: f32,
}

impl DcAeConfig {
    /// `mit-han-lab/dc-ae-f32c32-sana-1.0` config.
    pub fn sana_f32c32() -> Self {
        use BlockType::{EfficientVit as E, Res as R};
        Self {
            in_channels: 3,
            latent_channels: 32,
            attention_head_dim: 32,
            block_out_channels: vec![128, 256, 512, 512, 1024, 1024],
            layers_per_block: vec![3, 3, 3, 3, 3, 3],
            block_types: vec![R, R, R, E, E, E],
            qkv_multiscales: vec![5],
            norm_eps: 1e-5,
            attn_eps: 1e-15,
            scaling_factor: 0.41407,
        }
    }

    /// A tiny CPU-deterministic config used by the component/round-trip unit tests: three shallow-ish
    /// `Res` stages + one deep `EfficientVit` stage, small channel counts and a single layer per
    /// stage, so a random-weight forward runs fast on CPU while exercising every primitive
    /// (ResBlock, EfficientViT linear-attn, GLUMBConv, ConvPixelShuffle up/down, trms2d). Channel
    /// counts stay divisible by `attention_head_dim` (the deep stage is an attention stage).
    pub fn tiny_test() -> Self {
        use BlockType::{EfficientVit as E, Res as R};
        Self {
            in_channels: 3,
            latent_channels: 8,
            attention_head_dim: 8,
            block_out_channels: vec![16, 16, 32, 32],
            layers_per_block: vec![1, 1, 1, 1],
            block_types: vec![R, R, R, E],
            qkv_multiscales: vec![3],
            norm_eps: 1e-5,
            attn_eps: 1e-15,
            scaling_factor: 0.41407,
        }
    }

    pub fn num_stages(&self) -> usize {
        self.block_out_channels.len()
    }

    /// Total spatial compression factor (`2^(num_stages-1)`) — the deepest stage carries no
    /// up/down-sample, each of the other `num_stages-1` stages is a ×2 rung.
    pub fn spatial_compression(&self) -> i32 {
        1 << (self.num_stages() - 1)
    }
}
