//! SeedVR2 model configuration — the candle port of `mlx-gen-seedvr2/src/config.rs` (the 3B default
//! plus the 7B override set). Mirrors the mflux reference `SeedVR2Transformer` constructor defaults
//! and `ModelConfig.seedvr2_3b/7b`. The VAE config is shared across both variants.
//!
//! Dimensions are `usize` (candle's shape type); RoPE/MLP/AdaLN toggles are `bool`; `window` is the
//! `(T,H,W)` attention window.

/// Diffusion-transformer hyper-parameters.
#[derive(Clone, Copy, Debug)]
pub struct DitConfig {
    pub vid_in_channels: usize,  // 33 = noise(16) + cond latent(16) + mask(1)
    pub vid_out_channels: usize, // 16
    pub vid_dim: usize,          // 3B 2560 / 7B 3072
    pub txt_in_dim: usize,       // 5120 (precomputed neg-prompt embedding width)
    pub heads: usize,            // 3B 20 / 7B 24
    pub head_dim: usize,         // 128
    pub expand_ratio: usize,     // 4
    pub num_layers: usize,       // 3B 32 / 7B 36
    pub mm_layers: usize,        // dual-stream layers; >= this index uses shared (`.all`) weights
    pub patch_t: usize,          // 1
    pub patch_h: usize,          // 2
    pub patch_w: usize,          // 2
    pub rope_dim: usize,         // 3B 128 / 7B 64
    pub rope_on_text: bool,      // 3B true / 7B false
    pub rope_pixel: bool,        // freqs_for: 3B "lang"(false) / 7B "pixel"(true)
    pub swiglu_mlp: bool,        // 3B swiglu(true) / 7B "normal" gelu(false)
    pub use_output_ada: bool,    // 3B true / 7B false
    pub last_layer_vid_only: bool, // 3B true / 7B false
    pub norm_eps: f64,           // 1e-5
    pub window: (usize, usize, usize), // (4,3,3)
}

impl DitConfig {
    /// SeedVR2-3B (the primary variant).
    pub fn seedvr2_3b() -> Self {
        Self {
            vid_in_channels: 33,
            vid_out_channels: 16,
            vid_dim: 2560,
            txt_in_dim: 5120,
            heads: 20,
            head_dim: 128,
            expand_ratio: 4,
            num_layers: 32,
            mm_layers: 10,
            patch_t: 1,
            patch_h: 2,
            patch_w: 2,
            rope_dim: 128,
            rope_on_text: true,
            rope_pixel: false,
            swiglu_mlp: true,
            use_output_ada: true,
            last_layer_vid_only: true,
            norm_eps: 1e-5,
            window: (4, 3, 3),
        }
    }

    /// SeedVR2-7B override set (sc-5197 / sc-5927). dim 3072 / 24 heads / 36 layers, `mm_layers=36`
    /// (every layer dual-stream — no shared `.all`), `rope_dim=64` **pixel-mode** RoPE with
    /// `rope_on_text=false`, `mlp_type="normal"` (GELU), no output AdaLN / no last-layer-vid-only.
    pub fn seedvr2_7b() -> Self {
        Self {
            vid_dim: 3072,
            heads: 24,
            num_layers: 36,
            mm_layers: 36,
            rope_dim: 64,
            rope_on_text: false,
            rope_pixel: true,
            swiglu_mlp: false,
            use_output_ada: false,
            last_layer_vid_only: false,
            ..Self::seedvr2_3b()
        }
    }
}

/// 3D causal video VAE config (shared by 3B and 7B).
#[derive(Clone, Copy, Debug)]
pub struct VaeConfig {
    pub in_channels: usize,             // 3
    pub out_channels: usize,            // 3
    pub latent_channels: usize,         // 16
    pub block_out_channels: [usize; 4], // (128,256,512,512)
    pub enc_layers_per_block: usize,    // 2
    pub dec_layers_per_block: usize,    // 3
    pub temporal_down_blocks: usize,    // 2
    pub temporal_up_blocks: usize,      // 2
    pub scaling_factor: f64,            // 0.9152
    pub spatial_scale: usize,           // 8
    pub group_norm_groups: usize,       // 32
    pub group_norm_eps: f64,            // 1e-6
}

impl VaeConfig {
    pub fn seedvr2() -> Self {
        Self {
            in_channels: 3,
            out_channels: 3,
            latent_channels: 16,
            block_out_channels: [128, 256, 512, 512],
            enc_layers_per_block: 2,
            dec_layers_per_block: 3,
            temporal_down_blocks: 2,
            temporal_up_blocks: 2,
            scaling_factor: 0.9152,
            spatial_scale: 8,
            group_norm_groups: 32,
            group_norm_eps: 1e-6,
        }
    }
}

/// VAE total spatial downscale (`spatial_scale` 8) × DiT patch (2) = 16. Output dims must be ÷ this.
pub const VAE_SCALE: u32 = 16;
/// The 1-step Euler timestep (= the scheduler's `num_train_steps` default, 1000).
pub const TIMESTEP: f64 = 1000.0;
