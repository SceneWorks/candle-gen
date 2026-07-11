//! Krea 2's **Qwen3-VL-4B vision tower** for image-grounded edit conditioning (epic 10871 / sc-10880).
//!
//! Krea's Qwen3-VL condition encoder is text-only for t2i (the vision tower is never assembled). The
//! image-edit path needs the encoder to *see* the source image(s), so this module builds the
//! parity-tested [`candle_gen_boogu::vision::VisionTower`] from a **Krea-4B** [`VisionConfig`] over the
//! `visual.*` weights that already ship inside the Krea `text_encoder/` checkpoint — reuse, not a
//! re-port (the ViT block code is dim-generic off `VisionConfig`; only the width/depth/projection/
//! deepstack indices differ from boogu's 8B tower).
//!
//! The 4B `vision_config` was extracted from the real `krea/Krea-2-Raw` `text_encoder/config.json`
//! (sc-10875): hidden 1024, depth 24, num_heads 16, out_hidden 2560 (= the LM hidden width, so the
//! merged vision embeds splice straight into the token stream), deepstack `[5, 11, 17]`. Patch/merge/
//! temporal geometry (16 / 2 / 2) and `num_position_embeddings` 2304 are shared with boogu's 8B.

use std::path::Path;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::Result;
use candle_gen_boogu::loader::Weights as BooguWeights;
pub use candle_gen_boogu::vision::preprocess::preprocess_image;
pub use candle_gen_boogu::vision::{VisionConfig, VisionTower};

/// Standard Qwen3-VL vision/image token ids (shared across Qwen3-VL checkpoints; confirmed for Krea in
/// sc-10875): the `<|image_pad|>` placeholder the merged vision embeds are spliced over.
pub const IMAGE_TOKEN_ID: u32 = 151655;
/// `<|vision_start|>` — opens a reference image's vision block in the edit template.
pub const VISION_START_ID: u32 = 151652;
/// `<|vision_end|>` — closes it.
pub const VISION_END_ID: u32 = 151653;

/// The vision tower runs f32 (parity-grade for this encoder, shared with the boogu port); the DiT casts
/// the resulting features to bf16.
const VISION_DTYPE: DType = DType::F32;

/// Krea 2 Raw's Qwen3-VL-**4B** `vision_config` (verbatim from the real `text_encoder/config.json`,
/// sc-10875). Distinct from boogu's [`VisionConfig::qwen3_vl`] (8B) in exactly four fields: `hidden_size`
/// 1024 (vs 1152), `depth` 24 (vs 27), `out_hidden_size` 2560 (vs 4096), `deepstack_visual_indexes`
/// `[5, 11, 17]` (vs `[8, 16, 24]`). Everything else (heads 16, patch 16, merge 2, temporal 2,
/// position-embeddings 2304, in-channels 3) matches.
pub fn krea_vision_config() -> VisionConfig {
    VisionConfig {
        hidden_size: 1024,
        num_heads: 16,
        depth: 24,
        out_hidden_size: 2560,
        patch_size: 16,
        temporal_patch_size: 2,
        spatial_merge_size: 2,
        in_channels: 3,
        num_position_embeddings: 2304,
        deepstack_visual_indexes: vec![5, 11, 17],
    }
}

/// Build the Krea Qwen3-VL-4B vision tower from a snapshot's `text_encoder/` dir (epic 10871 /
/// sc-10880). The `visual.*` weights live in the SAME `text_encoder/` checkpoint as the LM (Krea keys
/// them `visual.*`, not boogu's `model.visual.*`), so the tower loads through a boogu [`BooguWeights`]
/// over that dir at f32 with prefix `"visual"`. On a packed Krea tier the LM is packed but the vision
/// tower stays dense bf16 (loaded → f32 here), which boogu's `VisionTower::load` guards for.
pub fn load_vision_tower(root: impl AsRef<Path>, device: &Device) -> Result<VisionTower> {
    let dir = root.as_ref().join("text_encoder");
    let w = BooguWeights::from_dir(&dir, device, VISION_DTYPE)?;
    Ok(VisionTower::load(&w, krea_vision_config(), "visual")?)
}

/// Preprocess one RGB8 image + run the tower → `(image_embeds [n, 2560], deepstack [3×[n, 2560]],
/// grid_thw [1, gh, gw])`. `n` = the merged token count `(gh/2)·(gw/2)` = the number of `<|image_pad|>`
/// placeholders the tokenizer must emit for this reference. All f32 (the DiT casts to bf16 downstream).
pub fn encode_image(
    tower: &VisionTower,
    pixels_hwc: &[u8],
    height: usize,
    width: usize,
    device: &Device,
) -> Result<(Tensor, Vec<Tensor>, [i32; 3])> {
    let (pixel_values, grid) = preprocess_image(pixels_hwc, height, width, device)?;
    let (embeds, deepstack) = tower.forward(&pixel_values, &[grid])?;
    Ok((embeds, deepstack, grid))
}

/// The number of merged vision tokens (`<|image_pad|>` placeholders) a `grid_thw` produces: `t · (h/m) ·
/// (w/m)` with merge `m = spatial_merge_size`.
pub fn merged_token_count(grid: [i32; 3], merge: usize) -> usize {
    let m = merge as i32;
    (grid[0] * (grid[1] / m) * (grid[2] / m)) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn krea_4b_config_matches_extracted_values() {
        let c = krea_vision_config();
        assert_eq!(c.hidden_size, 1024);
        assert_eq!(c.depth, 24);
        assert_eq!(c.num_heads, 16);
        assert_eq!(c.out_hidden_size, 2560);
        assert_eq!(c.deepstack_visual_indexes, vec![5, 11, 17]);
        // head_dim = 1024/16 = 64.
        assert_eq!(c.head_dim(), 64);
    }

    #[test]
    fn krea_4b_differs_from_boogu_8b_in_four_fields() {
        let k = krea_vision_config();
        let b = VisionConfig::qwen3_vl();
        assert_ne!(k.hidden_size, b.hidden_size);
        assert_ne!(k.depth, b.depth);
        assert_ne!(k.out_hidden_size, b.out_hidden_size);
        assert_ne!(k.deepstack_visual_indexes, b.deepstack_visual_indexes);
        // Shared geometry.
        assert_eq!(k.num_heads, b.num_heads);
        assert_eq!(k.patch_size, b.patch_size);
        assert_eq!(k.spatial_merge_size, b.spatial_merge_size);
        assert_eq!(k.num_position_embeddings, b.num_position_embeddings);
    }

    #[test]
    fn merged_token_count_is_grid_over_merge_squared() {
        // 512² → grid [1, 32, 32] → merged (32/2)·(32/2) = 256.
        assert_eq!(merged_token_count([1, 32, 32], 2), 256);
        assert_eq!(merged_token_count([1, 4, 2], 2), 2);
    }
}
