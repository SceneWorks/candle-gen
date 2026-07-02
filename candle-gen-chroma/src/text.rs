//! Chroma text conditioning — the T5-XXL prompt encode.
//!
//! Chroma conditions on **T5-XXL only** (no CLIP / no pooled). The mlx provider pads the T5 sequence
//! to 512 and runs it *masked*; the candle slice instead encodes at the prompt's **natural length**
//! and runs T5 unmasked:
//! - candle's [`T5EncoderModel`] exposes no key-padding mask (only the decoder builds causal masks),
//! - padding to 512 and running unmasked would let the pad tokens pollute the real-token embeddings —
//!   *worse* parity than not padding at all,
//! - at natural length every position is a real (content / `</s>`) token, so the MMDiT attention mask
//!   the mlx provider applies would be all-ones and is unnecessary (see `transformer.rs`).
//!
//! The vendored `assets/t5_tokenizer.json` (google t5-v1.1-xxl, converted from the snapshot's
//! `spiece.model`, identical to the mlx provider's asset) is used directly with its baked-in padding
//! **disabled**, so `encode` returns the natural-length ids. The T5 encoder lives in `text_encoder/`
//! (the Chroma diffusers layout; FLUX puts T5 in `text_encoder_2/`).

use std::path::{Path, PathBuf};

use candle_gen::candle_core::{Device, Tensor};
use candle_gen::{CandleError, Result};
use candle_transformers::models::t5::{Config as T5Config, T5EncoderModel};
use tokenizers::Tokenizer;

/// The vendored T5-XXL tokenizer (converted from Chroma's `spiece.model`).
const T5_TOKENIZER_JSON: &[u8] = include_bytes!("../assets/t5_tokenizer.json");

/// Load the vendored T5 tokenizer with padding **disabled** (so `encode` returns natural-length ids;
/// the JSON ships a padding config that would otherwise auto-pad).
pub fn load_tokenizer() -> Result<Tokenizer> {
    let mut tok = Tokenizer::from_bytes(T5_TOKENIZER_JSON)
        .map_err(|e| CandleError::Msg(format!("chroma: load vendored T5 tokenizer: {e}")))?;
    tok.with_padding(None);
    Ok(tok)
}

/// Load the T5-XXL encoder from the Chroma snapshot's `text_encoder/` (config.json + sharded
/// safetensors), at f32 on `device`.
pub fn load_t5(root: &Path, device: &Device) -> Result<T5EncoderModel> {
    let dir = root.join("text_encoder");
    let cfg_str = std::fs::read_to_string(dir.join("config.json"))
        .map_err(|e| CandleError::Msg(format!("chroma: read text_encoder/config.json: {e}")))?;
    let cfg: T5Config = serde_json::from_str(&cfg_str)
        .map_err(|e| CandleError::Msg(format!("chroma: parse T5 config.json: {e}")))?;
    let files = safetensors_in(&dir)?;
    // f32: the Chroma DiT runs f32 activations and `context_embedder` requires an f32 input; loading
    // the bf16 checkpoint as f32 keeps the weight values (bf16) in f32 containers (mlx parity).
    let vb = candle_gen::mmap_var_builder(&files, candle_gen::candle_core::DType::F32, device)?;
    T5EncoderModel::load(vb, &cfg).map_err(Into::into)
}

/// Encode `prompt` → the T5 sequence embedding `[1, L, 4096]` (f32), at natural length (the `</s>`
/// eos is appended by the tokenizer; no padding). `t5` is `&mut` because its `forward` carries a
/// relative-position-bias cache.
pub fn encode_prompt(
    tokenizer: &Tokenizer,
    t5: &mut T5EncoderModel,
    prompt: &str,
    device: &Device,
) -> Result<Tensor> {
    let enc = tokenizer
        .encode(prompt, true)
        .map_err(|e| CandleError::Msg(format!("chroma: T5 tokenize: {e}")))?;
    let ids: Vec<u32> = enc.get_ids().to_vec();
    if ids.is_empty() {
        return Err(CandleError::Msg("chroma: empty T5 tokenization".into()));
    }
    let input = Tensor::new(ids.as_slice(), device)?.unsqueeze(0)?; // [1, L]
    Ok(t5.forward(&input)?)
}

/// Sorted list of every `.safetensors` in `dir` (sharded T5 checkpoints ship as
/// `model-0000n-of-0000m.safetensors`). Errors if none are found.
fn safetensors_in(dir: &Path) -> Result<Vec<PathBuf>> {
    candle_gen::sorted_safetensors(dir, "chroma")
}
