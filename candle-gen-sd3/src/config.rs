//! SD3.5 architecture configuration (sc-7876, epic 7982).
//!
//! All layer counts, dims, and the T5 sequence length are config-driven so the later stories (C2
//! Large+Turbo pipeline, C3 Medium MMDiT-X) can reuse the same `Sd3Config` with different presets
//! rather than re-deriving the geometry. The defaults here are the **Large** preset
//! (`stabilityai/stable-diffusion-3.5-large`), spike-confirmed against the public diffusers
//! `SD3Transformer2DModel` config.json and the SD3 paper "Scaling Rectified Flow Transformers".

/// SD3.5 model + MMDiT geometry. Constructed via the named presets ([`Sd3Config::large`]); the
/// fields are public so C2/C3 can tweak the T5 length or block count without a new constructor.
#[derive(Debug, Clone)]
pub struct Sd3Config {
    // ---- latent / patchify ----
    /// VAE latent channel count (and the DiT `in_channels`). SD3.5 = 16.
    pub in_channels: usize,
    /// Patchify patch size on each spatial axis. SD3.5 = 2.
    pub patch_size: usize,
    /// Max latent grid side the learned positional embedding table covers (the diffusers
    /// `pos_embed_max_size`). The patchified token grid is cropped from the centre of this table.
    pub pos_embed_max_size: usize,

    // ---- MMDiT core ----
    /// Joint attention hidden width (`num_attention_heads * attention_head_dim`). Large = 2432.
    pub inner_dim: usize,
    /// Attention heads. Large = 38.
    pub num_heads: usize,
    /// Per-head dim (`inner_dim / num_heads`). Large = 64.
    pub head_dim: usize,
    /// Joint/double-stream block count. Large = 38.
    pub num_layers: usize,
    /// FFN hidden expansion ratio (diffusers `mlp_ratio` = 4.0).
    pub mlp_ratio: f32,
    /// Whether per-head QK-RMSNorm is applied (SD3.5 Large/Medium = true; vanilla SD3 = false).
    pub qk_norm: bool,
    /// Whether the LAST joint block drops its context (text) stream output (`context_pre_only`):
    /// the final block only needs to update the image tokens, so its `ff_context`/`add_*_out` are
    /// absent in the checkpoint. diffusers sets this on the last block.
    pub context_pre_only_last: bool,

    // ---- conditioning aggregator ----
    /// Pooled projection width = CLIP-L pooled (768) + CLIP-bigG pooled (1280) = 2048. Added to the
    /// timestep embedding (NOT the token sequence).
    pub pooled_dim: usize,
    /// Joint attention context width the DiT's `context_embedder` consumes. SD3.5 = 4096 (the T5
    /// hidden; the concatenated CLIP context is zero-padded up to this on the hidden axis).
    pub joint_attention_dim: usize,
    /// CLIP-L penultimate hidden width (768).
    pub clip_l_dim: usize,
    /// CLIP-bigG penultimate hidden width (1280).
    pub clip_g_dim: usize,
    /// Combined CLIP context width before zero-pad to `joint_attention_dim` (clip_l_dim + clip_g_dim
    /// = 2048).
    pub clip_concat_dim: usize,
    /// CLIP token length (both encoders). SD3.5 = 77.
    pub clip_seq_len: usize,
    /// T5-XXL token length — **configurable** (SD3.5 default 256; 77/512 are also valid). The full
    /// context sequence is `clip_seq_len + t5_seq_len` (333 at the defaults).
    pub t5_seq_len: usize,
    /// T5-XXL hidden width (4096).
    pub t5_dim: usize,

    // ---- timestep embedding ----
    /// Sinusoidal timestep embedding width before the MLP (diffusers `time_embed` in dim = 256).
    pub timestep_channels: usize,

    // ---- MMDiT-X dual attention (SD3.5 Medium, sc-7878) ----
    /// The block indices that carry the SECOND image-only self-attention (`attn2`) of the
    /// **MMDiT-X** architecture — the diffusers `dual_attention_layers`. For these blocks `norm1` is
    /// a `SD35AdaLayerNormZeroX` emitting **9** AdaLN chunks (the usual 6 + `shift_msa2/scale_msa2/
    /// gate_msa2`), and the block adds `gate_msa2 · attn2(modulate(LN(img), shift2, scale2))` to the
    /// image stream alongside the joint-attn + mlp residuals. **Large = empty** (no dual blocks);
    /// **Medium = [0..=12]** (the first 13 of 24 blocks). Verified against the public HF
    /// `transformer/config.json`.
    pub dual_attention_layers: Vec<usize>,
}

impl Sd3Config {
    /// The full joint context sequence length: CLIP (77) ++ T5 (`t5_seq_len`). 333 at the SD3.5
    /// defaults (77 + 256).
    pub fn context_seq_len(&self) -> usize {
        self.clip_seq_len + self.t5_seq_len
    }

    /// The patchified-latent channel count the `proj_out`/unpatchify head produces:
    /// `patch_size^2 * in_channels`.
    pub fn patch_dim(&self) -> usize {
        self.patch_size * self.patch_size * self.in_channels
    }

    /// FFN hidden width for a joint block (`mlp_ratio * inner_dim`).
    pub fn ff_hidden(&self) -> usize {
        (self.mlp_ratio * self.inner_dim as f32) as usize
    }

    /// Whether block `i` is a **dual-attention** (MMDiT-X) block — i.e. carries the second
    /// image-only `attn2` and a 9-chunk `norm1`. `false` for every block on Large (no dual layers).
    pub fn is_dual_block(&self, i: usize) -> bool {
        self.dual_attention_layers.contains(&i)
    }

    /// **SD3.5 Large** preset (`stabilityai/stable-diffusion-3.5-large`).
    pub fn large() -> Self {
        Self {
            in_channels: 16,
            patch_size: 2,
            pos_embed_max_size: 192,
            inner_dim: 2432,
            num_heads: 38,
            head_dim: 64,
            num_layers: 38,
            mlp_ratio: 4.0,
            qk_norm: true,
            context_pre_only_last: true,
            pooled_dim: 2048,
            joint_attention_dim: 4096,
            clip_l_dim: 768,
            clip_g_dim: 1280,
            clip_concat_dim: 2048,
            clip_seq_len: 77,
            t5_seq_len: 256,
            t5_dim: 4096,
            timestep_channels: 256,
            // Large has NO dual-attention blocks (every block is the standard 6-chunk joint block).
            dual_attention_layers: Vec::new(),
        }
    }

    /// **SD3.5 Medium** preset (`stabilityai/stable-diffusion-3.5-medium`) — the **MMDiT-X** model.
    /// Geometry from the public HF `transformer/config.json`: `num_layers=24`,
    /// `num_attention_heads=24`, `attention_head_dim=64` (⇒ `inner_dim=1536`),
    /// `caption_projection_dim=1536`, `pos_embed_max_size=384` (2 MP / up to 1440²),
    /// `dual_attention_layers=[0..=12]` (the first 13 blocks carry the second image-only attention),
    /// `qk_norm="rms_norm"`. Conditioning (triple-TE 2048 pooled / 4096 joint), patch 2, 16-ch
    /// in/out are shared with Large. `context_pre_only_last` is still set (the final, 24th, block
    /// drops its text stream — same as Large/SD3).
    pub fn medium() -> Self {
        Self {
            in_channels: 16,
            patch_size: 2,
            pos_embed_max_size: 384,
            inner_dim: 1536,
            num_heads: 24,
            head_dim: 64,
            num_layers: 24,
            mlp_ratio: 4.0,
            qk_norm: true,
            context_pre_only_last: true,
            pooled_dim: 2048,
            joint_attention_dim: 4096,
            clip_l_dim: 768,
            clip_g_dim: 1280,
            clip_concat_dim: 2048,
            clip_seq_len: 77,
            t5_seq_len: 256,
            t5_dim: 4096,
            timestep_channels: 256,
            // MMDiT-X: the first 13 blocks (0..=12) carry the dual image-only attention.
            dual_attention_layers: (0..=12).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn large_preset_geometry_matches_diffusers() {
        let c = Sd3Config::large();
        // inner_dim factors cleanly into heads × head_dim.
        assert_eq!(c.inner_dim, c.num_heads * c.head_dim);
        assert_eq!(c.inner_dim, 2432);
        assert_eq!(c.num_heads, 38);
        assert_eq!(c.num_layers, 38);
        // pooled = CLIP-L (768) + bigG (1280) = 2048.
        assert_eq!(c.clip_l_dim + c.clip_g_dim, c.pooled_dim);
        assert_eq!(c.clip_concat_dim, 2048);
        // context sequence = 77 CLIP + 256 T5 = 333 at the defaults.
        assert_eq!(c.context_seq_len(), 333);
        // T5 hidden is the joint-attention width.
        assert_eq!(c.t5_dim, c.joint_attention_dim);
        // patch dim = 2*2*16 = 64.
        assert_eq!(c.patch_dim(), 64);
    }

    #[test]
    fn medium_preset_geometry_matches_diffusers() {
        let c = Sd3Config::medium();
        // inner_dim = heads × head_dim = 24 × 64 = 1536.
        assert_eq!(c.inner_dim, c.num_heads * c.head_dim);
        assert_eq!(c.inner_dim, 1536);
        assert_eq!(c.num_heads, 24);
        assert_eq!(c.head_dim, 64);
        assert_eq!(c.num_layers, 24);
        assert_eq!(c.pos_embed_max_size, 384);
        // The first 13 blocks (0..=12) are dual-attention; 13..=23 are standard joint blocks.
        assert_eq!(c.dual_attention_layers, (0..=12).collect::<Vec<_>>());
        assert_eq!(c.dual_attention_layers.len(), 13);
        for i in 0..=12 {
            assert!(c.is_dual_block(i), "block {i} must be dual");
        }
        for i in 13..24 {
            assert!(!c.is_dual_block(i), "block {i} must be standard");
        }
        // Shared conditioning widths with Large.
        assert_eq!(c.pooled_dim, 2048);
        assert_eq!(c.joint_attention_dim, 4096);
        assert_eq!(c.patch_dim(), 64);
    }

    #[test]
    fn large_has_no_dual_blocks() {
        let c = Sd3Config::large();
        assert!(c.dual_attention_layers.is_empty());
        for i in 0..c.num_layers {
            assert!(!c.is_dual_block(i), "Large block {i} must not be dual");
        }
    }

    #[test]
    fn t5_length_is_configurable() {
        let mut c = Sd3Config::large();
        assert_eq!(c.context_seq_len(), 333);
        c.t5_seq_len = 512;
        assert_eq!(c.context_seq_len(), 77 + 512);
        c.t5_seq_len = 77;
        assert_eq!(c.context_seq_len(), 154);
    }
}
