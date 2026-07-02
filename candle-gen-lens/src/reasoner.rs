//! The optional Lens **PromptReasoner** (sc-5118) — the local gpt-oss `generate` path of the vendor
//! `_vendor/lens/reasoner.py::PromptReasoner`. It rewrites the user prompt into a richer
//! text-to-image prompt *before* encoding.
//!
//! **Off by default** (the vendor pipeline's `enable_reasoner`, and not wired into the candle registry
//! / generate path — the worker always runs with it off). The OpenAI-compatible-API branch of the
//! vendor reasoner is host-agnostic and needs no model, so this is only the local-`generate` path.
//!
//! Turning the encoder-only gpt-oss into a **generating** model adds the full 24-layer stack + final
//! `norm` + `lm_head` + an incremental KV-cache greedy decode ([`LensReasonerModel`] in
//! [`crate::text_encoder`]), the harmony `reasoning_effort="low"` template
//! ([`LensTokenizer::encode_reasoner`]), and the harmony-channel output parse
//! ([`clean_reasoner_output`]). The MoE experts can be quantized (Q4/Q8, sc-5111) so the reasoner
//! loads at the same ~12 GB as the encoder.

use std::path::{Path, PathBuf};

use candle_gen::candle_core::quantized::GgmlDType;
use candle_gen::candle_core::DType;
use candle_gen::{CandleError, Result as CResult};

use crate::text::{clean_reasoner_output, LensTokenizer};
use crate::text_encoder::{Config as EncoderConfig, LensReasonerModel};

/// Generation defaults from the vendor `PromptReasoner.__init__` (`max_new_tokens=4096`). The vendor
/// default `temperature=0.7` samples; the candle reasoner is **greedy** (deterministic), matching the
/// parity reference (torch `generate(do_sample=False)`).
pub const DEFAULT_MAX_NEW_TOKENS: usize = 4096;

/// The local PromptReasoner: the generating gpt-oss model + the tokenizer (harmony reasoner template +
/// output parse). The vendor default `enable=False` is the caller's concern; this is the
/// `enable=True` local-`generate` path.
pub struct LensReasoner {
    model: LensReasonerModel,
    tokenizer: LensTokenizer,
}

impl LensReasoner {
    /// Load from a Lens snapshot dir (`text_encoder/` shards + `tokenizer/tokenizer.json`) on the
    /// default device. `dtype` is the compute dtype (bf16 in production); `quant` (Q4/Q8) transcodes
    /// the MoE experts to keep the reasoner at ~12 GB (sc-5111).
    pub fn load(
        snapshot_dir: impl AsRef<Path>,
        dtype: DType,
        quant: Option<GgmlDType>,
    ) -> CResult<Self> {
        let root = snapshot_dir.as_ref();
        let tokenizer = LensTokenizer::from_file(root.join("tokenizer").join("tokenizer.json"))?;
        let device = candle_gen::default_device()?;
        let files = safetensors_files(&root.join("text_encoder"))?;
        let vb = candle_gen::mmap_var_builder(&files, dtype, &device)?;
        let model = LensReasonerModel::new(&EncoderConfig::gpt_oss_20b(), vb, quant)?;
        Ok(Self { model, tokenizer })
    }

    /// Refine one prompt via the local gpt-oss (greedy decode). `date` fills the harmony preamble's
    /// `Current date:` line. Returns the cleaned final-channel rewrite, or the original `prompt` when
    /// the reasoner produced no usable final text (the vendor `clean_text_out or prompt`).
    pub fn refine(&self, prompt: &str, max_new_tokens: usize, date: &str) -> CResult<String> {
        let input_ids = self.tokenizer.encode_reasoner(prompt, date)?;
        if input_ids.is_empty() {
            return Err(CandleError::Msg("lens reasoner: empty tokenization".into()));
        }
        let new_tokens = self.model.generate_greedy(&input_ids, max_new_tokens)?;
        let raw = self.tokenizer.decode(&new_tokens)?;
        let cleaned = clean_reasoner_output(&raw);
        Ok(if cleaned.is_empty() {
            prompt.to_string()
        } else {
            cleaned
        })
    }
}

/// The sorted `.safetensors` files of a weights dir (errors if the dir or its weights are missing).
fn safetensors_files(dir: &Path) -> CResult<Vec<PathBuf>> {
    if !dir.is_dir() {
        return Err(CandleError::Msg(format!(
            "lens reasoner: missing weights dir {}",
            dir.display()
        )));
    }
    // Shared sorted-`.safetensors` resolver (sc-8999 / F-019); the crafted "missing dir" message
    // above stays local.
    candle_gen::sorted_safetensors(dir, "lens reasoner")
}
