//! EVA `PatchEmbed`: `Conv2d(in→embed, kernel=stride=patch)` over NCHW pixels. Candle port of
//! `eva_vit_model.py PatchEmbed`. The MLX-converted checkpoint stores the conv weight OHWI
//! `[embed, patch, patch, in]` (channels-last); candle's `conv2d` is OIHW, so it is transposed at load
//! (the candle face-stack convention, `common::ohwi_to_oihw`).

use candle_core::Tensor;

use candle_gen::weights::Weights;
use candle_gen::Result as GenResult;

use crate::eva_clip::join;

pub struct PatchEmbed {
    proj_w: Tensor, // [embed, in, patch, patch] (OIHW, transposed from the file's OHWI)
    proj_b: Tensor, // [embed]
    patch: usize,
    embed_dim: usize,
}

impl PatchEmbed {
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        patch: usize,
        embed_dim: usize,
    ) -> GenResult<Self> {
        let ohwi = w.require(&join(prefix, "proj.weight"))?; // [embed, patch, patch, in]
        let proj_w = ohwi.permute((0, 3, 1, 2))?.contiguous()?; // → [embed, in, patch, patch]
        Ok(Self {
            proj_w,
            proj_b: w.require(&join(prefix, "proj.bias"))?,
            patch,
            embed_dim,
        })
    }

    /// `pixel_values`: NCHW `[B, in, H, W]` (H=W=image_size) → `[B, grid², embed]` (row-major, matching
    /// torch `flatten(2).transpose(1,2)`).
    pub fn forward(&self, pixel_values: &Tensor) -> candle_core::Result<Tensor> {
        // stride = kernel = patch, no padding → [B, embed, grid, grid].
        let y = pixel_values.conv2d(&self.proj_w, 0, self.patch, 1, 1)?;
        let b = self.proj_b.reshape((1, self.embed_dim, 1, 1))?;
        let y = y.broadcast_add(&b)?;
        let (bsz, e, g, _g) = y.dims4()?;
        // flatten(2).transpose(1,2): [B,embed,g,g] → [B,embed,g²] → [B,g²,embed].
        y.reshape((bsz, e, g * g))?.transpose(1, 2)?.contiguous()
    }
}
