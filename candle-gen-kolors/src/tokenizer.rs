//! Kolors prompt tokenization — the candle port of `mlx-gen-kolors`'s `tokenizer.rs`. Reproduces the
//! ChatGLM3 tokenizer the diffusers `KolorsPipeline` drives so the [`crate::chatglm3`] encoder receives
//! byte-identical `input_ids` / `attention_mask` / `position_ids`.
//!
//! ChatGLM3 ships only a **slow** SentencePiece tokenizer; the fast `tokenizer.json` is materialized
//! once into the snapshot's `tokenizer/` dir by Kolors' `tools/build_kolors_tokenizer.py` (a faithful
//! `LlamaConverter` replica). This wrapper loads it via the `tokenizers` crate for the SP **content**
//! ids and applies the ChatGLM-specific framing:
//!
//!  - **Prefix tokens** `[gMASK]` (64790) + `sop` (64792) prepended.
//!  - **Truncation** of the content to `max_length - 2` (reserving the 2 prefix tokens).
//!  - **Left padding** to `max_length` (256) with pad = unk = 0 (`padding_side="left"`), producing the
//!    matching `attention_mask` (`[0]*pad + [1]*len`) and `position_ids` (`[0]*pad + 0..len`) — the
//!    left-pad restarts real-token positions at 0, and Kolors threads these `position_ids` into the
//!    encoder RoPE.

use std::path::Path;

use candle_gen::{CandleError, Result};
use tokenizers::Tokenizer;

/// `[gMASK]` prefix token id (appended after the SP vocab).
pub const GMASK_ID: u32 = 64790;
/// `sop` (start-of-prompt) prefix token id.
pub const SOP_ID: u32 = 64792;
/// Pad token id = SentencePiece `unk_id` (0), left-padded by the ChatGLM tokenizer.
pub const PAD_ID: u32 = 0;
/// Kolors' fixed prompt length (`max_sequence_length`).
pub const MAX_LEN: usize = 256;

const PREFIX: [u32; 2] = [GMASK_ID, SOP_ID];

/// One tokenized prompt, left-padded to the configured length. `position_ids` is ChatGLM-specific
/// (Kolors threads it into the encoder RoPE); all three are length `max_len`.
pub struct KolorsTokens {
    pub input_ids: Vec<u32>,
    pub attention_mask: Vec<u32>,
    pub position_ids: Vec<i64>,
}

/// The Kolors (ChatGLM3) tokenizer.
pub struct KolorsTokenizer {
    inner: Tokenizer,
    max_len: usize,
}

impl KolorsTokenizer {
    /// Load from a snapshot `tokenizer/` dir containing the materialized `tokenizer.json`.
    pub fn from_dir(tokenizer_dir: impl AsRef<Path>) -> Result<Self> {
        Self::from_file(tokenizer_dir.as_ref().join("tokenizer.json"), MAX_LEN)
    }

    /// Load from an explicit `tokenizer.json` path with a chosen max length. `max_len` must leave room
    /// for the 2-token GMASK/SOP prefix plus at least one content token.
    pub fn from_file(tokenizer_json: impl AsRef<Path>, max_len: usize) -> Result<Self> {
        if max_len < PREFIX.len() + 1 {
            return Err(CandleError::Msg(format!(
                "kolors tokenizer: max_len must be >= {} (the {}-token GMASK/SOP prefix plus at least \
                 one content token); got {max_len}",
                PREFIX.len() + 1,
                PREFIX.len()
            )));
        }
        let mut inner = Tokenizer::from_file(tokenizer_json.as_ref())
            .map_err(|e| CandleError::Msg(format!("kolors: load tokenizer.json: {e}")))?;
        // This wrapper owns the (left-)padding and truncation; disable the tokenizer's own so `encode`
        // returns the raw SP content ids.
        inner.with_padding(None);
        let _ = inner.with_truncation(None);
        Ok(Self { inner, max_len })
    }

    /// Tokenize one prompt → left-padded `(max_len,)` `input_ids` / `attention_mask` / `position_ids`,
    /// byte-identical to `ChatGLMTokenizer(prompt, padding="max_length", max_length=max_len,
    /// truncation=True)`.
    pub fn encode(&self, prompt: &str) -> Result<KolorsTokens> {
        // SP content ids (no special tokens — the tokenizer.json has no post-processor).
        let enc = self
            .inner
            .encode(prompt, false)
            .map_err(|e| CandleError::Msg(format!("kolors: tokenize: {e}")))?;
        Ok(frame(enc.get_ids(), self.max_len))
    }
}

/// The pure ChatGLM framing of raw SP `content` ids into the left-padded `(max_len,)` tensors:
/// truncate content to `max_len - 2`, prepend the GMASK/SOP prefix, left-pad with `PAD_ID`, and build
/// the matching attention mask + position ids. Factored out of [`KolorsTokenizer::encode`] so the
/// framing is unit-testable without a loaded tokenizer. Assumes `max_len >= PREFIX.len() + 1` (the
/// `from_file` guard), so `pad` never underflows.
fn frame(content: &[u32], max_len: usize) -> KolorsTokens {
    let keep = max_len - PREFIX.len();
    let content = &content[..content.len().min(keep)];

    let mut ids: Vec<u32> = Vec::with_capacity(PREFIX.len() + content.len());
    ids.extend_from_slice(&PREFIX);
    ids.extend_from_slice(content);
    let len = ids.len();
    let pad = max_len - len;

    let mut input_ids = vec![PAD_ID; pad];
    input_ids.extend_from_slice(&ids);
    let mut attention_mask = vec![0u32; pad];
    attention_mask.resize(max_len, 1); // pad..max_len = valid (len of them)
    let mut position_ids = vec![0i64; pad];
    position_ids.extend(0..len as i64);

    KolorsTokens {
        input_ids,
        attention_mask,
        position_ids,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_file_rejects_max_len_below_prefix_plus_one() {
        for bad in [0usize, 1, 2] {
            let err = KolorsTokenizer::from_file("/nonexistent/tokenizer.json", bad)
                .err()
                .expect("should reject")
                .to_string();
            assert!(err.contains("max_len"), "max_len={bad} not rejected: {err}");
        }
    }

    #[test]
    fn framing_left_pads_with_prefix_and_position_ids() {
        // content [10, 11, 12] in a max_len-8 frame → 2 prefix + 3 content = 5 real, 3 left pad.
        let t = frame(&[10, 11, 12], 8);
        assert_eq!(t.input_ids, vec![0, 0, 0, GMASK_ID, SOP_ID, 10, 11, 12]);
        assert_eq!(t.attention_mask, vec![0, 0, 0, 1, 1, 1, 1, 1]);
        // position_ids restart at 0 for the first real token (left-pad convention).
        assert_eq!(t.position_ids, vec![0, 0, 0, 0, 1, 2, 3, 4]);
        assert_eq!(t.input_ids.len(), 8);
    }

    #[test]
    fn framing_truncates_content_to_reserve_prefix() {
        // max_len 5 reserves 2 for the prefix → keep at most 3 content tokens.
        let t = frame(&[1, 2, 3, 4, 5, 6], 5);
        assert_eq!(t.input_ids, vec![GMASK_ID, SOP_ID, 1, 2, 3]);
        assert_eq!(t.attention_mask, vec![1, 1, 1, 1, 1]); // full, no padding
        assert_eq!(t.position_ids, vec![0, 1, 2, 3, 4]);
    }
}
