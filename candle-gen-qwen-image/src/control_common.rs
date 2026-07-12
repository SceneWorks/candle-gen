//! Shared scaffolding for the Qwen-Image control lane (sc-9011, F-074): the 2512-Fun-Controlnet-Union
//! VACE lane ([`crate::control_fun`]). (Originally shared with the InstantX strict-pose lane, retired
//! in sc-9868.)
//!
//! The bespoke provider reuses a common component loader, prompt encoder, control-image
//! preprocessor, and VAE-output converter — historically copied verbatim between the two control files,
//! differing only by an error-message `label`. This module holds that common code parameterized by the
//! `label`, so a preprocessing fix lands once. The genuinely different pieces (checkpoint resolution, the
//! control-branch type and its forward wiring, the packed control context) stay in the lane's file.
//!
//! These are byte-for-byte the previous per-lane private helpers; the only change is threading the
//! `label` through the error messages, so both lanes' outputs are preserved exactly.

use std::path::Path;

use candle_gen::candle_core::{DType, Device, IndexOp, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use candle_gen::gen_core::Image;
use candle_gen::{CandleError, Result};

use crate::config::TextEncoderConfig;
use crate::text_encoder::QwenTextEncoder;

/// mmap a [`VarBuilder`] over every `.safetensors` in `root/sub` at `dtype`. `label` prefixes the error
/// messages (`"qwen control"` / `"qwen fun-control"`).
pub(crate) fn component_vb(
    root: &Path,
    sub: &str,
    dtype: DType,
    device: &Device,
    label: &str,
) -> Result<VarBuilder<'static>> {
    // Shared sorted-`.safetensors` → mmap (sc-8999 / F-019).
    candle_gen::component_vb(root, sub, dtype, device, label)
}

/// Build the Qwen-Image tokenizer from `root/tokenizer/tokenizer.json` **once**, so callers can cache
/// it on their generator struct and reuse it across encodes (sc-8991 / F-011) rather than re-parsing
/// `tokenizer.json` per prompt/branch. Byte-identical [`TokenizerConfig`] to the old per-encode load, so
/// the cached tokenizer yields the same ids. `label` prefixes the error message.
pub(crate) fn load_tokenizer(
    root: &Path,
    te_cfg: &TextEncoderConfig,
    label: &str,
) -> Result<TextTokenizer> {
    TextTokenizer::from_file(
        root.join("tokenizer/tokenizer.json"),
        tokenizer_config(te_cfg),
    )
    .map_err(|e| CandleError::Msg(format!("{label}: load tokenizer: {e}")))
}

/// The Qwen-Image [`TokenizerConfig`] — one home for the max-length / pad / chat-template policy so no
/// lane can silently drift (F-134 / sc-11190). [`load_tokenizer`] loads it from `root/tokenizer/`; the
/// edit lane (`edit.rs`) reuses this config with its own `-2511` processor-bundle path resolution.
/// `pad_to_max_length: false` is load-bearing for the empty-uncond CFG path (sc-8646 class).
pub(crate) fn tokenizer_config(te_cfg: &TextEncoderConfig) -> TokenizerConfig {
    TokenizerConfig {
        max_length: te_cfg.max_length,
        pad_token_id: te_cfg.pad_token_id,
        chat_template: ChatTemplate::QwenImage,
        pad_to_max_length: false,
    }
}

/// Tokenize + encode `prompt` → `prompt_embeds` `[1, seq, 3584]` at `dit_dtype` (bf16). Mirrors the
/// txt2img `Pipeline::encode`. `tok` is the caller's cached tokenizer ([`load_tokenizer`]); `label`
/// prefixes the error messages.
pub(crate) fn encode(
    tok: &TextTokenizer,
    te: &QwenTextEncoder,
    device: &Device,
    dit_dtype: DType,
    prompt: &str,
    label: &str,
) -> Result<Tensor> {
    let out = tok
        .tokenize(prompt)
        .map_err(|e| CandleError::Msg(format!("{label}: tokenize: {e}")))?;
    let len = out.ids.len();
    let ids: Vec<u32> = out.ids.iter().map(|&i| i as u32).collect();
    let input_ids = Tensor::from_vec(ids, (1, len), device)?;
    Ok(te.prompt_embeds(&input_ids)?.to_dtype(dit_dtype)?)
}

/// A pre-rendered/preprocessed RGB8 control image (at the request size) → `[1, 3, H, W]` f32 in
/// `[-1, 1]` (the VAE encoder's input range). Requires `image` already at `width × height` (the worker
/// renders/preprocesses at the target size — no silent stretch). `label` prefixes the error messages.
pub(crate) fn preprocess_control_image(
    image: &Image,
    width: u32,
    height: u32,
    device: &Device,
    label: &str,
) -> Result<Tensor> {
    if image.width != width || image.height != height {
        return Err(CandleError::Msg(format!(
            "{label}: control image {}x{} must match the request {width}x{height}",
            image.width, image.height
        )));
    }
    let (w, h) = (width as usize, height as usize);
    if image.pixels.len() != w * h * 3 {
        return Err(CandleError::Msg(format!(
            "{label}: control image buffer {} != {w}x{h}x3",
            image.pixels.len()
        )));
    }
    let mut data = vec![0f32; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let v = image.pixels[(y * w + x) * 3 + c] as f32 / 127.5 - 1.0;
                data[c * h * w + y * w + x] = v;
            }
        }
    }
    Ok(Tensor::from_vec(data, (1, 3, h, w), device)?)
}

/// VAE output `[1, 3, H, W]` in `[-1, 1]` → an RGB8 [`Image`]. Identical for both lanes.
pub(crate) fn to_image(decoded: &Tensor) -> Result<Image> {
    let img = ((decoded.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?.to_dtype(DType::U8)?;
    let img = img.i(0)?.to_device(&Device::Cpu)?;
    let (c, h, w) = img.dims3()?;
    if c != 3 {
        return Err(CandleError::Msg(format!("expected 3 channels, got {c}")));
    }
    let pixels = img.permute((1, 2, 0))?.flatten_all()?.to_vec1::<u8>()?;
    Ok(Image {
        width: w as u32,
        height: h as u32,
        pixels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preprocess_shape_and_range() {
        // Both lanes' `control_preprocess_shape_and_range` assertions reduce to this shared path.
        let img = Image {
            width: 16,
            height: 8,
            pixels: vec![255u8; 16 * 8 * 3],
        };
        let t = preprocess_control_image(&img, 16, 8, &Device::Cpu, "qwen control").unwrap();
        assert_eq!(t.dims(), &[1, 3, 8, 16]);
        // 255 → 255/127.5 - 1 = 1.0
        let v = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(v.iter().all(|x| (x - 1.0).abs() < 1e-4));
        // size mismatch errors loudly.
        assert!(preprocess_control_image(&img, 32, 8, &Device::Cpu, "qwen control").is_err());
    }

    #[test]
    fn preprocess_label_threads_into_error() {
        // The only per-lane difference is the error label — verify both parameterizations surface it.
        let img = Image {
            width: 16,
            height: 8,
            pixels: vec![0u8; 16 * 8 * 3],
        };
        let e = preprocess_control_image(&img, 32, 8, &Device::Cpu, "qwen control")
            .unwrap_err()
            .to_string();
        assert!(e.starts_with("qwen control:"), "got: {e}");
        let e = preprocess_control_image(&img, 32, 8, &Device::Cpu, "qwen fun-control")
            .unwrap_err()
            .to_string();
        assert!(e.starts_with("qwen fun-control:"), "got: {e}");
    }

    #[test]
    fn to_image_roundtrips_channels_and_shape() {
        // A [1,3,2,4] tensor at the extremes maps back to a 4x2 RGB8 image (1.0 → 255, -1.0 → 0).
        let t = Tensor::from_vec(
            (0..24)
                .map(|i| if i % 2 == 0 { 1f32 } else { -1f32 })
                .collect::<Vec<_>>(),
            (1, 3, 2, 4),
            &Device::Cpu,
        )
        .unwrap();
        let out = to_image(&t).unwrap();
        assert_eq!((out.width, out.height), (4, 2));
        assert_eq!(out.pixels.len(), 4 * 2 * 3);
    }

    #[test]
    fn tokenizer_config_matches_qwen_image_policy() {
        // F-134 (sc-11190): the one `tokenizer_config()` home must reproduce the byte-identical policy
        // the txt2img / sequential-TE / edit lanes previously built inline — any drift here silently
        // changes the caption ids (the sc-8646 `pad_to_max_length: false` uncond path especially).
        let te_cfg = TextEncoderConfig::qwen_image();
        let cfg = tokenizer_config(&te_cfg);
        assert_eq!(cfg.max_length, te_cfg.max_length);
        assert_eq!(cfg.pad_token_id, te_cfg.pad_token_id);
        assert!(matches!(cfg.chat_template, ChatTemplate::QwenImage));
        assert!(!cfg.pad_to_max_length);
    }
}
