//! Boogu instruction tokenization — the Qwen3-VL chat template + tokenizer that turns a text prompt
//! into the `input_ids` the condition encoder consumes. Port of `mlx-gen-boogu`'s `tokenizer.rs`.
//!
//! The reference builds messages `[system, user]` and calls `apply_chat_template(...,
//! add_generation_prompt=False)` (no trailing assistant turn). For text-to-image the system prompt is
//! [`SYSTEM_PROMPT_T2I`]; the CFG-negative is the **empty** instruction with [`SYSTEM_PROMPT_DROP`].
//! We render the exact ChatML string ourselves and encode with `add_special_tokens=false` (the
//! `<|im_start|>`/`<|im_end|>` markers are literal special tokens already in the string).

use std::path::Path;

use candle_gen::candle_core::{Device, Tensor};
use candle_gen::gen_core::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use candle_gen::{CandleError, Result};

/// Text-to-image system prompt (reference `SYSTEM_PROMPT_4_T2I`).
pub const SYSTEM_PROMPT_T2I: &str = "You are a helpful assistant that generates high-quality images based on user instructions. The instructions are as follows.";

/// Empty-instruction (CFG negative) / unified-edit system prompt (reference `SYSTEM_PROMPT_DROP` ==
/// `SYSTEM_PROMPT_4_TI2I_UNIFIED`).
pub const SYSTEM_PROMPT_DROP: &str = "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.";

/// Qwen3-VL vision marker tokens (`mllm/tokenizer.json` added tokens). The processor expands a single
/// `<|image_pad|>` into `merged` copies; we render the expanded block directly.
const VISION_START: &str = "<|vision_start|>";
const VISION_END: &str = "<|vision_end|>";
const IMAGE_PAD: &str = "<|image_pad|>";

/// Render the ChatML string for a `(system, user)` turn pair with no generation prompt:
/// `<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n`.
fn render_chat(system: &str, user: &str) -> String {
    format!("<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n")
}

/// Render the ChatML string for an image-conditioned `(system, user)` turn: one
/// `<|vision_start|>` + `n_k`×`<|image_pad|>` + `<|vision_end|>` block per reference (in order), all
/// prepended to the user text — the Qwen3-VL chat template + processor expansion for
/// `content = [image₀, …, image_{N-1}, text]` (images first, no separators, then the instruction).
fn render_chat_with_images(system: &str, user: &str, num_image_tokens: &[usize]) -> String {
    let blocks: String = num_image_tokens
        .iter()
        .map(|&n| format!("{VISION_START}{}{VISION_END}", IMAGE_PAD.repeat(n)))
        .collect();
    format!("<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{blocks}{user}<|im_end|>\n")
}

/// The Boogu condition tokenizer: the snapshot's `mllm/tokenizer.json` wrapped so we can render the
/// Boogu chat templates and encode them. Builds `input_ids` directly on the model device.
pub struct BooguTokenizer {
    inner: TextTokenizer,
    device: Device,
    /// The RoPE-table cap the condition encoder is sized for
    /// ([`crate::pipeline::MAX_TEXT_TOKENS`]); every encode path rejects a longer sequence up front.
    max_tokens: usize,
}

impl BooguTokenizer {
    /// Load from a snapshot's `mllm/tokenizer.json`. `max_tokens` is the RoPE-table cap the condition
    /// encoder is sized for; every encode path rejects an over-length sequence up front with a clear
    /// length error rather than failing deep in `Rotary::text`'s `narrow` (sc-9047).
    pub fn from_snapshot(
        root: impl AsRef<Path>,
        device: &Device,
        max_tokens: usize,
    ) -> Result<Self> {
        let inner = TextTokenizer::from_file(
            root.as_ref().join("mllm").join("tokenizer.json"),
            TokenizerConfig {
                // We render the chat string ourselves and call `encode_ids` directly, so the config
                // template/padding are unused; keep them inert.
                max_length: max_tokens,
                pad_token_id: 151643, // Qwen <|endoftext|>; unused (no padding on this path)
                chat_template: ChatTemplate::None,
                pad_to_max_length: false,
            },
        )
        .map_err(|e| CandleError::Msg(format!("boogu: load mllm tokenizer: {e}")))?;
        Ok(Self {
            inner,
            device: device.clone(),
            max_tokens,
        })
    }

    /// Encode a rendered chat string to a `[1, L]` u32 `input_ids` tensor (`add_special_tokens=false`).
    /// An over-length sequence (incl. any spliced `<|image_pad|>` blocks) is rejected up front with a
    /// clear length error naming the cap and the actual length, rather than an opaque tensor-shape
    /// error deep in the condition encoder (sc-9047). Mirrors the sibling ideogram Qwen3-VL port.
    fn encode(&self, text: &str) -> Result<Tensor> {
        let ids = self
            .inner
            .encode_ids(text, false)
            .map_err(|e| CandleError::Msg(format!("boogu: tokenize: {e}")))?;
        check_len(ids.len(), self.max_tokens)?;
        let ids: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
        let len = ids.len();
        Ok(Tensor::from_vec(ids, (1, len), &self.device)?)
    }

    /// Encode the **positive** text-to-image instruction → `input_ids` `[1, L]`.
    pub fn encode_t2i(&self, prompt: &str) -> Result<Tensor> {
        self.encode(&render_chat(SYSTEM_PROMPT_T2I, prompt))
    }

    /// Encode the CFG **negative** (empty instruction with the drop system prompt) → `[1, L]`.
    pub fn encode_negative(&self) -> Result<Tensor> {
        self.encode(&render_chat(SYSTEM_PROMPT_DROP, ""))
    }

    /// Encode the **edit** instruction (text-only) → `input_ids` `[1, L]`. The TI2I unified system
    /// prompt ([`SYSTEM_PROMPT_DROP`]) is shared with image editing, so the CFG negative is just
    /// [`Self::encode_negative`] (empty user text, same system prompt).
    pub fn encode_edit(&self, instruction: &str) -> Result<Tensor> {
        self.encode(&render_chat(SYSTEM_PROMPT_DROP, instruction))
    }

    /// Encode the **image-conditioned edit** instruction → `input_ids` `[1, L]`, with the reference
    /// image's `num_image_tokens` (= merged vision tokens) `<|image_pad|>` placeholders spliced into
    /// the user turn. The text encoder then replaces those placeholder embeddings with the vision
    /// tower's output ([`crate::text_encoder::BooguTextEncoder::last_hidden_with_image`]).
    pub fn encode_edit_with_image(
        &self,
        instruction: &str,
        num_image_tokens: usize,
    ) -> Result<Tensor> {
        self.encode_edit_with_images(instruction, &[num_image_tokens])
    }

    /// Encode the **multi-image-conditioned edit** instruction → `input_ids` `[1, L]`, with one
    /// `<|image_pad|>` block per reference (`num_image_tokens[k]` placeholders for reference k, in
    /// order). The text encoder splices each reference's vision-tower output into its block
    /// ([`crate::text_encoder::BooguTextEncoder::last_hidden_with_images`]).
    pub fn encode_edit_with_images(
        &self,
        instruction: &str,
        num_image_tokens: &[usize],
    ) -> Result<Tensor> {
        self.encode(&render_chat_with_images(
            SYSTEM_PROMPT_DROP,
            instruction,
            num_image_tokens,
        ))
    }
}

/// Validate a rendered-chat token count against the RoPE-table cap (sc-9047): an empty sequence or one
/// longer than `max_tokens` (incl. any spliced `<|image_pad|>` blocks) returns a clear, actionable
/// length error naming the cap and the actual length — instead of an opaque `narrow` tensor-shape error
/// deep in the condition encoder. Pure so it is unit-testable without a real snapshot tokenizer.
fn check_len(len: usize, max_tokens: usize) -> Result<()> {
    if len == 0 {
        return Err(CandleError::Msg("boogu: empty token sequence".into()));
    }
    if len > max_tokens {
        return Err(CandleError::Msg(format!(
            "boogu: prompt has {len} tokens, exceeds max_text_tokens={max_tokens}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_len_rejects_over_cap_with_clear_message() {
        // An over-length prompt returns an actionable length error naming the cap and the actual
        // length — NOT an opaque tensor `narrow` error mid-generate (sc-9047).
        let err = check_len(1281, 1280).unwrap_err().to_string();
        assert!(err.contains("1281"), "names the actual length: {err}");
        assert!(err.contains("max_text_tokens=1280"), "names the cap: {err}");
        assert!(!err.contains("narrow"), "not an opaque tensor error: {err}");
    }

    #[test]
    fn check_len_accepts_at_and_below_cap() {
        // At-limit and below-limit prompts pass validation.
        assert!(check_len(1280, 1280).is_ok());
        assert!(check_len(1, 1280).is_ok());
    }

    #[test]
    fn check_len_rejects_empty() {
        assert!(check_len(0, 1280)
            .unwrap_err()
            .to_string()
            .contains("empty"));
    }

    #[test]
    fn multi_image_template_emits_one_block_per_reference() {
        let s = render_chat_with_images("sys", "edit it", &[2, 3]);
        // One vision block per reference, in order, before the instruction.
        assert_eq!(s.matches(VISION_START).count(), 2);
        assert_eq!(s.matches(VISION_END).count(), 2);
        // Total `<|image_pad|>` placeholders = sum of per-reference token counts.
        assert_eq!(s.matches(IMAGE_PAD).count(), 5);
        // Images precede the instruction text in the user turn.
        let vis = s.find(VISION_START).unwrap();
        let txt = s.find("edit it").unwrap();
        assert!(vis < txt);
        // A single-image render is the one-element case.
        let one = render_chat_with_images("sys", "x", &[4]);
        assert_eq!(one.matches(VISION_START).count(), 1);
        assert_eq!(one.matches(IMAGE_PAD).count(), 4);
    }
}
