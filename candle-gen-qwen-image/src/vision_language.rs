//! `QwenVisionLanguageEncoder` — the Qwen-Image-Edit conditioning encoder (candle port of
//! `mlx-gen-qwen-image`'s `text_encoder/vision_language`). It:
//!
//! 1. Embeds `input_ids`, then **splices** the vision-transformer embeds into the positions of the
//!    `<|image_pad|>` (151655) tokens (consumed in order).
//! 2. Runs the 28 LM layers + final RMSNorm (the verified [`QwenTextEncoder`] stack). The fork uses
//!    **sequential** RoPE here (`position_ids = arange(seq)`), so the standard text RoPE applies.
//! 3. **Drops the first 64** template tokens (vs the T2I text path's 34).
//!
//! The vision tower depends only on the image, so the caller computes the vision embeds **once**
//! ([`encode_vision`](QwenVisionLanguageEncoder::encode_vision)) and reuses them for the positive +
//! negative prompts ([`encode_with_vision`](QwenVisionLanguageEncoder::encode_with_vision)).

use std::path::Path;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::{CandleError, Result};

use crate::config::TextEncoderConfig;
use crate::text_encoder::QwenTextEncoder;
use crate::vision::{Grid, VisionConfig, VisionTransformer};

/// The Qwen2.5-VL encoder runs in f32 (the mlx provider rounds only the final embeds to bf16).
const ENC_DTYPE: DType = DType::F32;

pub struct QwenVisionLanguageEncoder {
    lm: QwenTextEncoder,
    visual: VisionTransformer,
}

impl QwenVisionLanguageEncoder {
    /// `<|image_pad|>` token id (the placeholder replaced by vision embeds).
    pub const IMAGE_TOKEN_ID: u32 = 151655;
    /// Tokens dropped from the front of the Edit chat template.
    pub const EDIT_DROP_IDX: usize = 64;

    pub fn new(lm: QwenTextEncoder, visual: VisionTransformer) -> Self {
        Self { lm, visual }
    }

    /// Run the vision transformer over the reference patches → vision embeds `[n_vis, hidden]` (f32).
    /// Depends only on the image, so compute it **once** and reuse it across prompts.
    pub fn encode_vision(&self, pixel_values: &Tensor, grids: &[Grid]) -> Result<Tensor> {
        self.visual.forward(pixel_values, grids)
    }

    /// Splice precomputed `vision` embeds into the prompt token stream, run the LM, and drop the
    /// leading template tokens. `input_ids`: `[1, s]` (u32); `vision`: `[n_vis, hidden]`. Returns the
    /// prompt embeds `[1, s-64, hidden]` (f32).
    pub fn encode_with_vision(&self, input_ids: &Tensor, vision: &Tensor) -> Result<Tensor> {
        let embeds = self.lm.embed(input_ids)?; // [1, s, h] f32
        let spliced = self.splice(&embeds, input_ids, vision)?;
        let hidden = self.lm.forward_from_embeds(&spliced)?; // [1, s, h]
        let s = hidden.dim(1)?;
        if s <= Self::EDIT_DROP_IDX {
            return Err(CandleError::Msg(format!(
                "qwen edit: prompt sequence ({s}) is shorter than the {} template tokens to drop",
                Self::EDIT_DROP_IDX
            )));
        }
        Ok(hidden.narrow(1, Self::EDIT_DROP_IDX, s - Self::EDIT_DROP_IDX)?)
    }

    /// Convenience: [`encode_vision`](Self::encode_vision) then
    /// [`encode_with_vision`](Self::encode_with_vision) for a single prompt.
    pub fn encode(
        &self,
        input_ids: &Tensor,
        pixel_values: &Tensor,
        grids: &[Grid],
    ) -> Result<Tensor> {
        let vision = self.encode_vision(pixel_values, grids)?;
        self.encode_with_vision(input_ids, &vision)
    }

    /// Replace `<|image_pad|>` embeddings with the vision embeds (in order): build
    /// `[text_embeds ‖ vision_embeds]` and gather each output row at either its text position or the
    /// next vision row.
    fn splice(&self, embeds: &Tensor, input_ids: &Tensor, vision: &Tensor) -> Result<Tensor> {
        let (b, s, h) = embeds.dims3()?;
        let n_text = b * s;
        let n_vis = vision.dims()[0];
        let ids: Vec<u32> = input_ids.flatten_all()?.to_vec1::<u32>()?;
        let gather = image_gather_index(&ids, Self::IMAGE_TOKEN_ID, n_vis, n_text);

        let embeds_flat = embeds.reshape((n_text, h))?;
        let vision = vision.to_dtype(embeds.dtype())?;
        let src = Tensor::cat(&[&embeds_flat, &vision], 0)?; // [n_text + n_vis, h]
        let idx = Tensor::from_vec(gather, n_text, input_ids.device())?;
        Ok(src.index_select(&idx, 0)?.reshape((b, s, h))?)
    }
}

/// Gather indices into `[text_embeds(n_text) ‖ vision_embeds(n_vis)]`: image-token positions map to
/// the next vision row (`n_text + vi`), all others to their own text position. Pure — unit-tested.
pub fn image_gather_index(
    ids: &[u32],
    image_token_id: u32,
    n_vis: usize,
    n_text: usize,
) -> Vec<u32> {
    let mut out = Vec::with_capacity(n_text);
    let mut vi = 0usize;
    for (p, &id) in ids.iter().enumerate() {
        if id == image_token_id && vi < n_vis {
            out.push((n_text + vi) as u32);
            vi += 1;
        } else {
            out.push(p as u32);
        }
    }
    out
}

/// mmap a [`VarBuilder`] over every `.safetensors` in `root/sub` at `dtype`.
fn component_vb(
    root: &Path,
    sub: &str,
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder<'static>> {
    candle_gen::component_vb(root, sub, dtype, device, "qwen edit")
}

/// Load the Qwen-Image-**Edit** vision-language conditioning encoder from a `Qwen/Qwen-Image-Edit`
/// snapshot: the Qwen2.5-VL LM (`model.*`) + vision transformer (`visual.*`), both living under
/// `text_encoder/`. The validated reference snapshot is `-2511`.
pub fn load_vision_language_encoder(
    root: &Path,
    device: &Device,
) -> Result<QwenVisionLanguageEncoder> {
    let te_vb = component_vb(root, "text_encoder", ENC_DTYPE, device)?;
    let lm = QwenTextEncoder::new(&TextEncoderConfig::qwen_image(), te_vb.clone())?;
    let visual = VisionTransformer::new(te_vb, &VisionConfig::qwen_image_edit())?;
    Ok(QwenVisionLanguageEncoder::new(lm, visual))
}

#[cfg(test)]
mod tests {
    use super::image_gather_index;

    #[test]
    fn gather_replaces_image_tokens_in_order() {
        // ids: [a, PAD, PAD, b] with 4 text rows + 2 vision rows → vision at indices 4,5.
        let ids = [10u32, 151655, 151655, 11];
        let got = image_gather_index(&ids, 151655, 2, 4);
        assert_eq!(got, vec![0, 4, 5, 3]);
    }

    #[test]
    fn gather_stops_when_vision_exhausted() {
        // Only 1 vision row for 2 PADs: the second PAD keeps its text position.
        let ids = [151655u32, 151655, 7];
        let got = image_gather_index(&ids, 151655, 1, 3);
        assert_eq!(got, vec![3, 1, 2]);
    }
}
