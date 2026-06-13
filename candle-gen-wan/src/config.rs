//! Static configuration for the **Wan2.2 TI2V-5B** text-to-video model, read from the diffusers
//! checkpoint (`Wan-AI/Wan2.2-TI2V-5B-Diffusers`): `transformer/config.json` (`WanTransformer3DModel`),
//! `vae/config.json` (`AutoencoderKLWan`), `text_encoder/config.json` (`UMT5EncoderModel`), and
//! `scheduler/scheduler_config.json` (`UniPCMultistepScheduler`, flow-match).

/// Registry id — matches the mlx-gen-wan descriptor so a consumer resolves the same engine across
/// backends.
pub const MODEL_ID: &str = "wan2_2_ti2v_5b";

/// Default denoise steps (diffusers `sample_steps` / the UniPC default for the 5B).
pub const DEFAULT_STEPS: u32 = 40;
/// Default classifier-free guidance scale (`sample_guide_scale`).
pub const DEFAULT_GUIDANCE: f32 = 5.0;
/// Default output frame count. Must satisfy `frames % 4 == 1` (one latent frame + groups of 4).
pub const DEFAULT_FRAMES: u32 = 81;
/// Default playback / muxing cadence (`sample_fps`).
pub const DEFAULT_FPS: u32 = 24;
/// Flow-match time-shift applied to the sigma schedule (`flow_shift`).
pub const FLOW_SHIFT: f64 = 5.0;
/// Diffusion training horizon (`num_train_timesteps`).
pub const NUM_TRAIN_TIMESTEPS: usize = 1000;

/// Wan's default negative prompt (the reference anti-artifact string) used when CFG is on and the
/// request supplies none.
pub const NEGATIVE_FALLBACK: &str =
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，\
     低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，\
     毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走";

/// Spatial size must be a multiple of `vae_stride_spatial (16) × patch (2) = 32` so the latent
/// (`H/16`) is even for the DiT 2×2 spatial patch.
pub const SIZE_MULTIPLE: u32 = 32;
/// VAE spatial downsample factor (latent `H = height / 16`).
pub const VAE_STRIDE_SPATIAL: u32 = 16;
/// VAE temporal downsample factor (latent `T = (frames - 1) / 4 + 1`).
pub const VAE_STRIDE_TEMPORAL: u32 = 4;

/// `WanTransformer3DModel` dims (TI2V-5B, dense — no MoE).
#[derive(Clone, Copy, Debug)]
pub struct TransformerConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    /// `num_heads × head_dim` = 3072.
    pub dim: usize,
    pub ffn_dim: usize,
    pub freq_dim: usize,
    pub text_dim: usize,
    /// `(p_t, p_h, p_w)` patch (`(1, 2, 2)`).
    pub patch: (usize, usize, usize),
    pub eps: f64,
    pub rope_theta: f64,
    pub rope_max_seq_len: usize,
}

impl TransformerConfig {
    pub fn ti2v_5b() -> Self {
        Self {
            in_channels: 48,
            out_channels: 48,
            num_layers: 30,
            num_heads: 24,
            head_dim: 128,
            dim: 3072,
            ffn_dim: 14336,
            freq_dim: 256,
            text_dim: 4096,
            patch: (1, 2, 2),
            eps: 1e-6,
            rope_theta: 10000.0,
            rope_max_seq_len: 1024,
        }
    }
}

/// `AutoencoderKLWan` (z48, `is_residual`) decoder dims.
#[derive(Clone, Copy, Debug)]
pub struct VaeConfig {
    pub z_dim: usize,
    /// Decoder base width (`decoder_base_dim`).
    pub base_dim: usize,
    pub num_res_blocks: usize,
    /// Final spatial unpatchify factor (`patch_size`).
    pub patch_size: usize,
    /// Channels emitted by `conv_out` before unpatchify (= `out_channels × patch²` = 12).
    pub conv_out_channels: usize,
    pub out_channels: usize,
}

impl VaeConfig {
    pub fn ti2v_5b() -> Self {
        Self {
            z_dim: 48,
            base_dim: 256,
            num_res_blocks: 2,
            patch_size: 2,
            conv_out_channels: 12,
            out_channels: 3,
        }
    }
}

/// Per-channel latent de-normalization (`z = z·std + mean` before decode), from `vae/config.json`.
pub const LATENTS_MEAN: [f32; 48] = [
    -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557, -0.1382, 0.0542, 0.2813,
    0.0891, 0.157, -0.0098, 0.0375, -0.1825, -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108,
    -0.2158, 0.2502, -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.123, -0.0313,
    -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.052, 0.3748, 0.0152, 0.1957, 0.1433, -0.2944,
    0.3573, -0.0548, -0.1681, -0.0667,
];
pub const LATENTS_STD: [f32; 48] = [
    0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.499, 0.4818, 0.5013, 0.8158, 1.0344, 0.5894, 1.0901,
    0.6885, 0.6165, 0.8454, 0.4978, 0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
    0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093, 0.5709, 0.6065, 0.6415, 0.4944,
    0.5726, 1.2042, 0.5458, 1.6887, 0.3971, 1.06, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
];

/// `UMT5EncoderModel` (`google/umt5-xxl`) dims.
#[derive(Clone, Copy, Debug)]
pub struct TextEncoderConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub d_ff: usize,
    pub d_kv: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub num_buckets: usize,
    pub max_distance: usize,
    pub eps: f64,
    pub max_length: usize,
    pub pad_token_id: i32,
}

impl TextEncoderConfig {
    pub fn umt5_xxl() -> Self {
        Self {
            vocab_size: 256384,
            d_model: 4096,
            d_ff: 10240,
            d_kv: 64,
            num_heads: 64,
            num_layers: 24,
            num_buckets: 32,
            max_distance: 128,
            eps: 1e-6,
            max_length: 512,
            pad_token_id: 0,
        }
    }
}
