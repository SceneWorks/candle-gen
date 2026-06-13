//! FLUX.2-klein configuration, ported from `mlx-gen-flux2`'s `config.rs` (itself lifted from the
//! frozen mflux fork). The dims are the **klein-9b** target. Kept dimension-parametric so a future
//! 4b variant is only a constructor change.

/// Registry id for FLUX.2-klein-9B txt2img.
pub const FLUX2_KLEIN_9B_ID: &str = "flux2_klein_9b";

pub const DEFAULT_WIDTH: u32 = 1024;
pub const DEFAULT_HEIGHT: u32 = 1024;
/// Distilled klein default — the fork generates in 4 steps.
pub const DEFAULT_STEPS: u32 = 4;
/// Distilled klein runs at guidance 1.0 (no CFG); >1.0 enables a classifier-free negative pass.
pub const DEFAULT_GUIDANCE: f32 = 1.0;

/// Both image dims must be multiples of 16 (VAE /8 then the DiT's 2×2 patch) for a clean pack.
pub const SIZE_MULTIPLE: u32 = 16;

/// Dimension-parametric FLUX.2 model dimensions (klein-9b values).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Flux2Config {
    // --- MMDiT transformer ---
    /// Double (joint img+txt) blocks. 9b: 8.
    pub num_double_layers: usize,
    /// Single (fused parallel attention+SwiGLU) blocks. 9b: 24.
    pub num_single_layers: usize,
    /// Attention heads. 9b: 32.
    pub num_heads: usize,
    /// Per-head dim. `inner_dim = num_heads * head_dim` (9b: 4096).
    pub head_dim: usize,
    /// Latent channels entering/leaving the transformer = `num_latent_channels * 4` (2×2 patch).
    pub in_channels: usize,
    pub out_channels: usize,
    /// Text-embedding width entering the joint blocks = `3 * te_hidden_size` (concat of 3 Qwen3
    /// hidden-state layers). 9b: 12288.
    pub joint_attention_dim: usize,
    /// Single-block SwiGLU expansion ratio (`mlp_hidden = mlp_ratio * inner_dim`). 9b: 3.0.
    pub mlp_ratio: f32,
    /// Sinusoidal timestep-embedding width feeding `time_guidance_embed.linear_1` (klein: 256).
    pub timestep_channels: usize,

    // --- 4-axis RoPE over ids (t, h, w, layer) ---
    pub axes_dim: [usize; 4],
    pub rope_theta: f32,

    // --- Qwen3 text encoder ---
    pub te_hidden_size: usize,
    pub te_intermediate_size: usize,
    pub te_n_layers: usize,
    pub te_n_heads: usize,
    pub te_n_kv_heads: usize,
    pub te_head_dim: usize,
    pub te_rope_theta: f32,
    pub te_rms_norm_eps: f64,
    /// Hidden-state indices (index 0 = embeddings, index k = output of layer k-1) concatenated into
    /// `prompt_embeds`. klein: (9, 18, 27) → 3·hidden = 12288.
    pub te_out_layers: [usize; 3],
    pub max_sequence_length: usize,

    // --- VAE / latent geometry ---
    pub num_latent_channels: usize,
    pub vae_scale_factor: usize,
}

impl Flux2Config {
    /// FLUX.2-klein-9b (the story target).
    pub fn klein_9b() -> Self {
        Self {
            num_double_layers: 8,
            num_single_layers: 24,
            num_heads: 32,
            head_dim: 128,
            in_channels: 128,
            out_channels: 128,
            joint_attention_dim: 12288,
            mlp_ratio: 3.0,
            timestep_channels: 256,
            axes_dim: [32, 32, 32, 32],
            rope_theta: 2000.0,
            te_hidden_size: 4096,
            te_intermediate_size: 12288,
            te_n_layers: 36,
            te_n_heads: 32,
            te_n_kv_heads: 8,
            te_head_dim: 128,
            te_rope_theta: 1_000_000.0,
            te_rms_norm_eps: 1e-6,
            te_out_layers: [9, 18, 27],
            max_sequence_length: 512,
            num_latent_channels: 32,
            vae_scale_factor: 8,
        }
    }

    /// `num_heads * head_dim` — the transformer inner width (9b: 4096).
    pub fn inner_dim(&self) -> usize {
        self.num_heads * self.head_dim
    }

    /// Single-block SwiGLU hidden width (`mlp_ratio * inner_dim`, 9b: 12288).
    pub fn single_mlp_hidden(&self) -> usize {
        (self.mlp_ratio * self.inner_dim() as f32) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn klein_9b_dims_match_fork() {
        let c = Flux2Config::klein_9b();
        assert_eq!(c.num_double_layers, 8);
        assert_eq!(c.num_single_layers, 24);
        assert_eq!(c.num_heads, 32);
        assert_eq!(c.inner_dim(), 4096);
        assert_eq!(c.joint_attention_dim, 3 * c.te_hidden_size);
        assert_eq!(c.in_channels, c.num_latent_channels * 4);
        assert_eq!(c.single_mlp_hidden(), 12288);
        // RoPE axes sum to the head dim; each axis emits dim/2 freqs → cos/sin width head_dim/2.
        assert_eq!(c.axes_dim.iter().sum::<usize>(), c.head_dim);
        assert_eq!(c.te_head_dim * c.te_n_heads, c.te_hidden_size);
    }
}
