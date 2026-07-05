//! Qwen-Image-Edit VL tokenization + reference latents — candle port of `mlx-gen-qwen-image`'s
//! `vl_tokenizer`. Turns a reference image + edit prompt into the inputs the
//! [`crate::vision_language::QwenVisionLanguageEncoder`] consumes (`input_ids`, `pixel_values`,
//! `grid`), and VAE-encodes the reference for the transformer's **dual-latent** conditioning.
//!
//! Pipeline: resize the reference to the **condition size** (~384² area, sides /32, BICUBIC) → the
//! [`QwenImageProcessor`] (smart_resize /28 + patchify) → expand the single `<|image_pad|>` in the
//! fixed edit template to `prod(grid)//merge²` copies → tokenize. Separately, the dual-latent path
//! LANCZOS-resizes the reference to the VL dims, VAE-encodes, and packs.

use candle_gen::candle_core::{Device, Tensor};
use candle_gen::gen_core::imageops::{resize_bicubic_u8, resize_lanczos_u8};
use candle_gen::gen_core::tokenizer::TextTokenizer;
use candle_gen::{CandleError, Result};

use crate::image_processor::{Grid, ImageInput, QwenImageProcessor};
use crate::pipeline::pack_latents;
use crate::vae::QwenVaeEncoder;

/// The image-only output of [`preprocess_edit_image`] — `pixel_values` + `grid` for the vision tower,
/// plus the `<|image_pad|>` count the grid expands to. Depends only on the reference image, so it is
/// computed **once** and reused for the positive + negative prompts.
pub struct EditImage {
    pub pixel_values: Tensor,
    pub grid: Grid,
    pub n_image_tokens: usize,
}

/// Target condition area (fork `CONDITION_IMAGE_SIZE`): the reference is scaled to ~384², preserving
/// aspect, with each side rounded to a multiple of 32.
const CONDITION_AREA: f64 = 384.0 * 384.0;

/// Condition-resize dims `(width, height)`: `w = round(√(area·ratio)/32)·32`, `h = round((w/ratio)
/// /32)·32`, `ratio = width/height`. Round-half-to-even (Python `round`).
pub fn condition_resize_dims(width: usize, height: usize) -> (usize, usize) {
    let ratio = width as f64 / height as f64;
    let cw = (CONDITION_AREA * ratio).sqrt();
    let ch = cw / ratio;
    let cw = (cw / 32.0).round_ties_even() * 32.0;
    let ch = (ch / 32.0).round_ties_even() * 32.0;
    (cw as usize, ch as usize)
}

/// The fork's edit chat template (`use_picture_prefix=False`) with the single `<|image_pad|>`
/// expanded to `n_image_tokens` copies and the user prompt inserted.
pub fn build_edit_text(prompt: &str, n_image_tokens: usize) -> String {
    let pads = "<|image_pad|>".repeat(n_image_tokens);
    format!(
        "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, \
         texture, objects, background), then explain how the user's text instruction should alter \
         or modify the image. Generate a new image that meets the user's requirements while \
         maintaining consistency with the original input where appropriate.<|im_end|>\n\
         <|im_start|>user\n<|vision_start|>{pads}<|vision_end|>{prompt}<|im_end|>\n\
         <|im_start|>assistant\n"
    )
}

/// Image-only half: condition-resize (BICUBIC, /32) + patchify the reference, returning the
/// `pixel_values`, `grid`, and the `<|image_pad|>` count. `image` is RGB uint8 HWC. Independent of
/// the prompt, so the Edit generator runs this **once** per generation.
pub fn preprocess_edit_image(
    processor: &QwenImageProcessor,
    image: ImageInput,
    device: &Device,
) -> Result<EditImage> {
    let (cw, ch) = condition_resize_dims(image.width, image.height);
    if cw == 0 || ch == 0 {
        return Err(CandleError::Msg(format!(
            "qwen edit: condition-resize of a {}x{} reference collapses to a zero dimension \
             ({cw}x{ch}); aspect ratio is too extreme",
            image.width, image.height
        )));
    }
    let resized: Vec<u8> = if (image.height, image.width) == (ch, cw) {
        image.data.to_vec()
    } else {
        // `resize_bicubic_u8` already returns integer-valued, [0,255]-clamped f32; the `round().clamp()`
        // is an explicit/defensive u8 quantization matching the sibling `image_processor` path
        // (byte-identical here — not a bug fix).
        resize_bicubic_u8(image.data, image.height, image.width, ch, cw)?
            .iter()
            .map(|&v| v.round().clamp(0.0, 255.0) as u8)
            .collect()
    };

    let processed = processor.preprocess(
        ImageInput {
            data: &resized,
            height: ch,
            width: cw,
        },
        device,
    )?;

    // image_pad count = prod(grid) // merge².
    let merge2 = (processor.merge_size * processor.merge_size) as i32;
    let g = processed.grid;
    let n_image_tokens = (g[0] * g[1] * g[2] / merge2) as usize;

    Ok(EditImage {
        pixel_values: processed.pixel_values,
        grid: processed.grid,
        n_image_tokens,
    })
}

/// Prompt-only half: expand the edit template's `<|image_pad|>` to `n_image_tokens` copies and
/// tokenize, returning the input-id row (u32). Run once per prompt (positive + negative).
pub fn tokenize_edit_text(
    tokenizer: &TextTokenizer,
    prompt: &str,
    n_image_tokens: usize,
) -> Result<Vec<u32>> {
    let out = tokenizer
        .tokenize_preformatted(&build_edit_text(prompt, n_image_tokens))
        .map_err(|e| CandleError::Msg(format!("qwen edit: tokenize: {e}")))?;
    Ok(out.ids.iter().map(|&i| i as u32).collect())
}

/// VAE-encode + pack a reference image for the dual-latent path. Resize to `(calc_w, calc_h)` via
/// **LANCZOS** (the fork's `scale_to_dimensions`), normalize `[0,255] → [-1,1]` as NCHW, VAE-encode,
/// and `pack_latents`. Returns `(image_latents [1, (calc_h/16)·(calc_w/16), 64], cond_grid
/// (latent_h, latent_w))`. Port of `create_image_conditioning_latents` for a single reference.
pub fn encode_reference_latents(
    vae_encoder: &QwenVaeEncoder,
    image: ImageInput,
    calc_w: u32,
    calc_h: u32,
    device: &Device,
) -> Result<(Tensor, (usize, usize))> {
    let (cw, ch) = (calc_w as usize, calc_h as usize);
    let resized: Vec<f32> = if (image.height, image.width) == (ch, cw) {
        image.data.iter().map(|&p| p as f32).collect()
    } else {
        resize_lanczos_u8(image.data, image.height, image.width, ch, cw)?
    };
    // [0,255] → [-1,1], laid out NCHW [1, 3, calc_h, calc_w].
    let plane = ch * cw;
    let mut nchw = vec![0f32; 3 * plane];
    for y in 0..ch {
        for x in 0..cw {
            for c in 0..3 {
                nchw[c * plane + y * cw + x] = resized[(y * cw + x) * 3 + c] / 255.0 * 2.0 - 1.0;
            }
        }
    }
    let img = Tensor::from_vec(nchw, (1, 3, ch, cw), device)?;
    let latent = vae_encoder.encode(&img)?; // [1, 16, ch/8, cw/8]
    let packed = pack_latents(&latent, calc_w, calc_h)?;
    Ok((packed, ((calc_h / 16) as usize, (calc_w / 16) as usize)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edit_text_expands_image_pad_and_keeps_structure() {
        let t = build_edit_text("make it night", 3);
        assert_eq!(t.matches("<|image_pad|>").count(), 3);
        assert!(t.contains(
            "<|vision_start|><|image_pad|><|image_pad|><|image_pad|><|vision_end|>make it night<|im_end|>"
        ));
        assert!(t.starts_with("<|im_start|>system\nDescribe the key features"));
        assert!(t.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn condition_resize_dims_match_reference() {
        assert_eq!(condition_resize_dims(512, 512), (384, 384)); // square
        assert_eq!(condition_resize_dims(768, 512), (480, 320)); // 3:2 landscape
        assert_eq!(condition_resize_dims(512, 768), (320, 480)); // 2:3 portrait
    }
}
