//! Krea 2 condition tokenization (sc-7569) — the Qwen3-VL prompt template + fast `Qwen2Tokenizer`
//! that turns a text prompt into the `input_ids` the condition encoder consumes. Port of
//! `mlx-gen-krea`'s `text_encoder/tokenizer.rs`.
//!
//! The reference `Qwen3VLConditioner` wraps the user text in a fixed system-instruction template + an
//! `assistant` generation cue, tokenizes (`add_special_tokens=false`; the `<|im_start|>`/`<|im_end|>`
//! markers are added-tokens in `tokenizer.json`), runs Qwen3-VL, then drops the leading
//! [`PREFIX_TOKENS`] system-prefix tokens from the conditioning (the encoder does the drop). We render
//! the exact template string ourselves and encode it. The reference pads to `max_length`, which only
//! adds masked tokens; for the per-sample `B = 1` path the natural length is numerically equivalent
//! (the encoder runs masked and the DiT trims padding), so we emit the natural-length ids.

use std::path::Path;

use candle_gen::candle_core::{Device, Tensor};
use candle_gen::gen_core::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use candle_gen::{CandleError, Result};

/// System-instruction prefix (reference `prompt_template_encode_prefix`). Tokenizes to exactly
/// [`PREFIX_TOKENS`] tokens — the slice the encoder drops.
pub const PREFIX: &str = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n";

/// `assistant` generation cue appended after the user text (reference `prompt_template_encode_suffix`).
pub const SUFFIX: &str = "<|im_end|>\n<|im_start|>assistant\n";

/// Number of leading template-prefix tokens dropped from the conditioning (reference
/// `prompt_template_encode_start_idx`); [`PREFIX`] tokenizes to this many.
pub const PREFIX_TOKENS: usize = 34;

/// Qwen <|endoftext|> id — the pad token (unused on the natural-length path).
const PAD_TOKEN_ID: i32 = 151643;

/// Render the full template string for a user prompt: `{PREFIX}{user}{SUFFIX}`.
fn render(user: &str) -> String {
    format!("{PREFIX}{user}{SUFFIX}")
}

/// Render the **image-edit** template (epic 10871 / sc-10880): `{PREFIX}` then, for each reference, a
/// vision block `<|vision_start|><|image_pad|>×n<|vision_end|>` (`n` = that reference's merged vision
/// token count), then the user instruction, then `{SUFFIX}`. Mirrors Qwen3-VL's image-in-user-turn chat
/// template; the encoder splices the vision embeds over the `<|image_pad|>` runs
/// ([`crate::text_encoder::KreaTextEncoder::forward_with_images`]). The vision markers are added-tokens
/// in `tokenizer.json`, so each `<|image_pad|>` encodes to exactly one id.
fn render_edit(user: &str, n_per_ref: &[usize]) -> String {
    let mut s = PREFIX.to_string();
    for &n in n_per_ref {
        s.push_str("<|vision_start|>");
        for _ in 0..n {
            s.push_str("<|image_pad|>");
        }
        s.push_str("<|vision_end|>");
    }
    s.push_str(user);
    s.push_str(SUFFIX);
    s
}

/// The Krea condition tokenizer: the snapshot's `tokenizer/tokenizer.json` wrapped to render the Krea
/// template and encode it. Builds `input_ids` directly on the model device.
pub struct KreaTokenizer {
    inner: TextTokenizer,
    device: Device,
}

impl KreaTokenizer {
    /// Load from a snapshot's `tokenizer/tokenizer.json`.
    pub fn from_snapshot(root: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let inner = TextTokenizer::from_file(
            root.as_ref().join("tokenizer").join("tokenizer.json"),
            TokenizerConfig {
                // We render the template string ourselves and call `encode_ids` directly, so the
                // config template/padding are inert.
                max_length: 512,
                pad_token_id: PAD_TOKEN_ID,
                chat_template: ChatTemplate::None,
                pad_to_max_length: false,
            },
        )
        .map_err(|e| CandleError::Msg(format!("krea: load tokenizer: {e}")))?;
        Ok(Self {
            inner,
            device: device.clone(),
        })
    }

    /// Encode a rendered string to ids (`add_special_tokens=false`, matching the reference).
    fn encode(&self, text: &str) -> Result<Vec<i32>> {
        self.inner
            .encode_ids(text, false)
            .map_err(|e| CandleError::Msg(format!("krea: tokenize: {e}")))
    }

    /// Raw id vector for the templated prompt (parity testing against the reference `input_ids`).
    pub fn ids(&self, prompt: &str) -> Result<Vec<i32>> {
        self.encode(&render(prompt))
    }

    /// Raw id vector for the image-edit templated prompt (parity testing).
    pub fn edit_ids(&self, prompt: &str, n_per_ref: &[usize]) -> Result<Vec<i32>> {
        self.encode(&render_edit(prompt, n_per_ref))
    }

    /// Encode the **image-edit** templated prompt → `input_ids` `[1, L]` u32 (epic 10871 / sc-10880).
    /// `n_per_ref[k]` is reference `k`'s merged vision token count (from the vision tower's `grid_thw`);
    /// the encoder splices the vision embeds over the emitted `<|image_pad|>` runs and drops the same
    /// [`PREFIX_TOKENS`] system prefix. Over-length prompts are rejected up front (sc-9047).
    pub fn encode_with_images(
        &self,
        prompt: &str,
        n_per_ref: &[usize],
        max_tokens: usize,
    ) -> Result<Tensor> {
        let ids = self.edit_ids(prompt, n_per_ref)?;
        check_len(ids.len(), max_tokens)?;
        let ids: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
        let len = ids.len();
        Ok(Tensor::from_vec(ids, (1, len), &self.device)?)
    }

    /// Token count of the bare [`PREFIX`] (should equal [`PREFIX_TOKENS`]).
    pub fn prefix_len(&self) -> Result<usize> {
        Ok(self.encode(PREFIX)?.len())
    }

    /// Encode the templated prompt → `input_ids` `[1, L]` u32. The encoder drops the leading
    /// [`PREFIX_TOKENS`] from the resulting conditioning.
    ///
    /// `max_tokens` is the RoPE-table cap the condition encoder is sized for
    /// ([`crate::pipeline::MAX_TEXT_TOKENS`]); an over-length prompt is rejected up front with a clear
    /// length error, rather than failing deep in `Rotary::text`'s `narrow` with an opaque candle
    /// shape error mid-generate (sc-9047). Mirrors the sibling ideogram Qwen3-VL port's policy.
    pub fn encode_prompt(&self, prompt: &str, max_tokens: usize) -> Result<Tensor> {
        let ids = self.ids(prompt)?;
        check_len(ids.len(), max_tokens)?;
        let ids: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
        let len = ids.len();
        Ok(Tensor::from_vec(ids, (1, len), &self.device)?)
    }
}

/// Validate a templated-prompt token count against the RoPE-table cap (sc-9047): an empty sequence or
/// one longer than `max_tokens` returns a clear, actionable length error naming the cap and the actual
/// length — instead of an opaque `narrow` tensor-shape error deep in the condition encoder. Pure so it
/// is unit-testable without a real snapshot tokenizer.
fn check_len(len: usize, max_tokens: usize) -> Result<()> {
    if len == 0 {
        return Err(CandleError::Msg("krea: empty token sequence".into()));
    }
    if len > max_tokens {
        return Err(CandleError::Msg(format!(
            "krea: prompt has {len} tokens (incl. the {PREFIX_TOKENS}-token template prefix), \
             exceeds max_text_tokens={max_tokens}"
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
        let err = check_len(1025, 1024).unwrap_err().to_string();
        assert!(err.contains("1025"), "names the actual length: {err}");
        assert!(err.contains("max_text_tokens=1024"), "names the cap: {err}");
        assert!(!err.contains("narrow"), "not an opaque tensor error: {err}");
    }

    #[test]
    fn check_len_accepts_at_and_below_cap() {
        // At-limit and below-limit prompts pass validation.
        assert!(check_len(1024, 1024).is_ok());
        assert!(check_len(1, 1024).is_ok());
    }

    #[test]
    fn check_len_rejects_empty() {
        assert!(check_len(0, 1024)
            .unwrap_err()
            .to_string()
            .contains("empty"));
    }

    #[test]
    fn render_edit_emits_one_vision_block_per_reference() {
        // Two references (n = 3 and n = 2 merged tokens) → two vision blocks, 5 image_pad markers total,
        // in order, with the user instruction after the last vision block.
        let s = render_edit("make it autumn", &[3, 2]);
        assert!(s.starts_with(PREFIX));
        assert!(s.ends_with(SUFFIX));
        assert_eq!(s.matches("<|vision_start|>").count(), 2);
        assert_eq!(s.matches("<|vision_end|>").count(), 2);
        assert_eq!(s.matches("<|image_pad|>").count(), 5);
        // The instruction sits after the final vision_end and before the suffix.
        let after_vision = s.rsplit_once("<|vision_end|>").unwrap().1;
        assert!(after_vision.starts_with("make it autumn"));
    }

    #[test]
    fn render_edit_zero_refs_is_plain_template() {
        // No references → identical to the plain t2i template.
        assert_eq!(render_edit("a cat", &[]), render("a cat"));
    }
}
