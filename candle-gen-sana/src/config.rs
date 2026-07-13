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

/// SANA Linear-DiT **trunk** configuration (sc-11778, epic 11776) — the candle mirror of
/// `mlx_gen_sana::SanaTransformerConfig` (mlx-gen #613, story sc-8487).
///
/// Values mirror the diffusers `SanaTransformer2DModel` config for
/// `Efficient-Large-Model/Sana_1600M_1024px_diffusers` (the 1.6B model). Only the fields the forward
/// needs are modelled. Positional embedding is **NoPE** (`interpolation_scale` is `None` in the real
/// config, so `patch_embed` carries no `pos_embed` and the conv patchify + the Mix-FFN 3×3 depthwise
/// conv supply all locality); `guidance_embeds` / `qk_norm` are off for base SANA-1.6B and on for the
/// SANA-Sprint distilled variant.
#[derive(Clone, Debug)]
pub struct SanaTransformerConfig {
    /// Latent channels in (DC-AE f32c32 → 32).
    pub in_channels: i32,
    /// Latent channels out (== `in_channels` for SANA-1.6B; matches DC-AE decode input).
    pub out_channels: i32,
    /// Self-attention heads.
    pub num_attention_heads: i32,
    /// Per-head dim of self-attention (`inner_dim = num_attention_heads * attention_head_dim`).
    pub attention_head_dim: i32,
    /// Number of `SanaTransformerBlock`s.
    pub num_layers: i32,
    /// Cross-attention heads.
    pub num_cross_attention_heads: i32,
    /// Per-head dim of cross-attention.
    pub cross_attention_head_dim: i32,
    /// Caption (cross-attn KV) embedding channels in (Gemma-2 CHI → 2304).
    pub caption_channels: i32,
    /// GLUMBConv Mix-FFN expand ratio (`hidden = int(mlp_ratio * inner_dim)`).
    pub mlp_ratio: f32,
    /// Patchify conv kernel/stride (`1` for SANA — DC-AE already did the 32× spatial compression).
    pub patch_size: i32,
    /// LayerNorm epsilon for the affine-free norms (`norm1`/`norm2`/`norm_out`, `1e-6`).
    pub norm_eps: f32,
    /// `caption_norm` RMSNorm epsilon (`1e-5`).
    pub caption_norm_eps: f32,
    /// **SANA-Sprint** `qk_norm = "rms_norm_across_heads"` RMSNorm epsilon. diffusers builds the
    /// attn1/attn2 qk-norm via `Attention.__init__` WITHOUT an explicit `eps`, so it falls back to
    /// that constructor's default `eps = 1e-5` — distinct from [`Self::norm_eps`] (`1e-6`, the
    /// affine-free LayerNorms). Only consulted on the qk-norm path when [`Self::qk_norm`] is set.
    pub attn_qk_norm_eps: f32,
    /// Linear-attention denominator epsilon (`1e-15`, matching the DC-AE primitive).
    pub attn_eps: f32,
    /// **SANA-Sprint** (sc-8490): the trunk carries an extra *guidance embedder* — a
    /// timestep-embedding-style MLP on the embedded CFG-free guidance scalar, summed into the
    /// timestep conditioning (`conditioning = timesteps_emb + guidance_emb`). `false` for base
    /// SANA-1.6B (the trunk is byte-unchanged); `true` for Sprint. When set, the `time_embed.*` keys
    /// switch to the `SanaCombinedTimestepGuidanceEmbeddings` layout (no `.emb.` nesting; adds
    /// `time_embed.guidance_embedder.*`).
    pub guidance_embeds: bool,
    /// Sprint `guidance_embeds_scale` (`0.1`) — the pipeline pre-multiplies the guidance scalar by
    /// this before the embedder. Only consulted when [`Self::guidance_embeds`] is set.
    pub guidance_embeds_scale: f32,
    /// **SANA-Sprint** `qk_norm = "rms_norm_across_heads"`: an RMSNorm over the full projected
    /// query/key (the whole `inner_dim`, before the head split + the ReLU) in BOTH `attn1` (linear
    /// self-attn) and `attn2` (cross-attn). `false` for base SANA-1.6B (`qk_norm = None`), `true` for
    /// Sprint. When set the loader requires `attn1.norm_q/k.weight` + `attn2.norm_q/k.weight`.
    pub qk_norm: bool,
}

impl SanaTransformerConfig {
    /// `inner_dim = num_attention_heads * attention_head_dim`.
    pub fn inner_dim(&self) -> i32 {
        self.num_attention_heads * self.attention_head_dim
    }

    /// `Efficient-Large-Model/Sana_1600M_1024px_diffusers` transformer config.
    pub fn sana_1600m() -> Self {
        Self {
            in_channels: 32,
            out_channels: 32,
            num_attention_heads: 70,
            attention_head_dim: 32, // inner_dim = 2240
            num_layers: 20,
            num_cross_attention_heads: 20,
            cross_attention_head_dim: 112, // cross inner = 2240
            caption_channels: 2304,
            mlp_ratio: 2.5,
            patch_size: 1,
            norm_eps: 1e-6,
            caption_norm_eps: 1e-5,
            attn_qk_norm_eps: 1e-5,
            attn_eps: 1e-15,
            // Base SANA-1.6B has no guidance embedder and no qk-norm — the trunk is byte-unchanged.
            guidance_embeds: false,
            guidance_embeds_scale: 0.1,
            qk_norm: false,
        }
    }

    /// `Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers` transformer config: the
    /// continuous-time-consistency-distilled, CFG-free few-step variant. Same Linear-DiT backbone as
    /// [`Self::sana_1600m`] plus the Sprint deltas (`guidance_embeds = true`, `qk_norm =
    /// "rms_norm_across_heads"`).
    pub fn sana_sprint_1600m() -> Self {
        Self {
            guidance_embeds: true,
            guidance_embeds_scale: 0.1,
            qk_norm: true,
            ..Self::sana_1600m()
        }
    }
}
