//! SD3.5 triple text-encoder **aggregator** (sc-7876, epic 7982).
//!
//! SD3.5 conditions on three text encoders — CLIP-L, OpenCLIP bigG, and T5-XXL — combined into two
//! tensors fed to the MMDiT:
//!
//! - **pooled** `[B, 2048]` = `cat(CLIP-L pooled [768], CLIP-bigG pooled [1280])`. This is added to
//!   the timestep embedding (NOT to the token sequence) — it conditions the AdaLN modulation.
//! - **context** `[B, 333, 4096]` (at the SD3.5 defaults) = the token sequence the joint blocks
//!   attend over. Built in two steps, exactly as the public diffusers `StableDiffusion3Pipeline`:
//!   1. CLIP context = `cat(CLIP-L penultimate [77, 768], CLIP-bigG penultimate [77, 1280])` →
//!      `[77, 2048]`, then **zero-padded on the hidden axis** to `[77, 4096]`
//!      (`joint_attention_dim`). The pad is on the *trailing* hidden dims (diffusers
//!      `F.pad(clip, (0, t5_dim - clip_concat_dim))`).
//!   2. context = `cat([clip_padded [77, 4096], t5 [t5_len, 4096]], dim=seq)` → `[77 + t5_len, 4096]`.
//!
//! This module owns the **aggregation** — the parity-critical concat/pad/order that the spike
//! flagged. The actual CLIP/T5 forward (loading the encoders, penultimate-layer extraction, EOS
//! pooling) is wired in C2's pipeline; keeping the aggregator a pure tensor transform lets the
//! ordering be unit-tested on CPU with synthetic encoder outputs (no weights/GPU needed), the same
//! correctness bar epic 7841 used.

use std::path::Path;

use candle_gen::candle_core::IndexOp;
use candle_gen::candle_core::{DType, Device, Result, Tensor, D};
use candle_gen::candle_nn::{Linear, Module};
use candle_gen::{CandleError, Result as CandleResult};
use candle_transformers::models::stable_diffusion::{self, clip};
use candle_transformers::models::t5::{Config as T5Config, T5EncoderModel};
use tokenizers::Tokenizer;

use crate::config::Sd3Config;

/// The raw per-encoder outputs the aggregator combines. Produced by the encoders in C2; here they
/// are the inputs to the pure aggregation so the ordering is testable in isolation.
///
/// All tensors carry a leading batch axis `B`.
pub struct EncoderOutputs {
    /// CLIP-L penultimate hidden state `[B, clip_seq_len, clip_l_dim]` (768-wide).
    pub clip_l_hidden: Tensor,
    /// CLIP-bigG penultimate hidden state `[B, clip_seq_len, clip_g_dim]` (1280-wide).
    pub clip_g_hidden: Tensor,
    /// CLIP-L pooled/projected output `[B, clip_l_dim]` (768-wide).
    pub clip_l_pooled: Tensor,
    /// CLIP-bigG pooled/projected output `[B, clip_g_dim]` (1280-wide).
    pub clip_g_pooled: Tensor,
    /// T5-XXL encoder sequence `[B, t5_seq_len, t5_dim]` (4096-wide).
    pub t5_hidden: Tensor,
}

/// The two SD3.5 conditioning tensors fed to the MMDiT.
pub struct Sd3Conditioning {
    /// `[B, pooled_dim]` (2048) — added to the timestep embedding.
    pub pooled: Tensor,
    /// `[B, context_seq_len, joint_attention_dim]` (333 × 4096 at defaults) — the joint token
    /// sequence.
    pub context: Tensor,
}

/// Build the SD3.5 pooled + context conditioning from the three encoders' outputs.
///
/// Order and padding match the public diffusers `StableDiffusion3Pipeline._get_clip_prompt_embeds`
/// + `encode_prompt`:
/// - pooled = `cat([clip_l_pooled, clip_g_pooled], dim=-1)`;
/// - clip_context = `cat([clip_l_hidden, clip_g_hidden], dim=-1)` then right-pad the hidden axis to
///   `joint_attention_dim` with zeros;
/// - context = `cat([clip_context, t5_hidden], dim=seq)`.
pub fn aggregate(cfg: &Sd3Config, enc: &EncoderOutputs) -> Result<Sd3Conditioning> {
    // ---- pooled [B, 2048] ----
    let pooled = Tensor::cat(&[&enc.clip_l_pooled, &enc.clip_g_pooled], D::Minus1)?;

    // ---- CLIP context [B, 77, 2048] -> zero-pad hidden axis to [B, 77, 4096] ----
    let clip_context = Tensor::cat(&[&enc.clip_l_hidden, &enc.clip_g_hidden], D::Minus1)?;
    // The concatenated CLIP width must be the configured `clip_concat_dim` (768 + 1280 = 2048); a
    // mismatch means a mis-shaped encoder output, caught here before the pad rather than producing a
    // silently wrong context.
    let clip_w = clip_context.dim(D::Minus1)?;
    if clip_w != cfg.clip_concat_dim {
        return Err(candle_gen::candle_core::Error::Msg(format!(
            "sd3 aggregator: concatenated CLIP context width {clip_w} != configured \
             clip_concat_dim {} (clip_l_dim {} + clip_g_dim {})",
            cfg.clip_concat_dim, cfg.clip_l_dim, cfg.clip_g_dim
        )));
    }
    let clip_padded = pad_hidden_to(&clip_context, cfg.joint_attention_dim)?;

    // ---- context = cat([clip_padded, t5], seq) -> [B, 333, 4096] ----
    let context = Tensor::cat(&[&clip_padded, &enc.t5_hidden], 1)?;

    Ok(Sd3Conditioning { pooled, context })
}

/// Right-pad the LAST (hidden) axis of `x` `[..., h]` to width `target` with zeros (`F.pad(x, (0,
/// target - h))`). Errors if `x` is already wider than `target`.
fn pad_hidden_to(x: &Tensor, target: usize) -> Result<Tensor> {
    let h = x.dim(D::Minus1)?;
    if h == target {
        return Ok(x.clone());
    }
    if h > target {
        return Err(candle_gen::candle_core::Error::Msg(format!(
            "sd3 aggregator: clip context hidden {h} exceeds joint_attention_dim {target}"
        )));
    }
    let mut shape = x.dims().to_vec();
    *shape.last_mut().unwrap() = target - h;
    let pad = Tensor::zeros(shape, x.dtype(), x.device())?;
    Tensor::cat(&[x, &pad], D::Minus1)
}

/// Build zeroed encoder outputs at the config's shapes for a given batch. NOTE: this is a
/// structural-test fixture only — the real CFG negative branch is the **empty-prompt encode** through
/// the same encoders (NOT a zero tensor; see the pipeline module header), so `zeroed_outputs` is not
/// on the generation path. Its only consumer is `zeroed_outputs_aggregate_to_correct_shape`.
pub fn zeroed_outputs(
    cfg: &Sd3Config,
    batch: usize,
    dtype: DType,
    device: &Device,
) -> Result<EncoderOutputs> {
    Ok(EncoderOutputs {
        clip_l_hidden: Tensor::zeros((batch, cfg.clip_seq_len, cfg.clip_l_dim), dtype, device)?,
        clip_g_hidden: Tensor::zeros((batch, cfg.clip_seq_len, cfg.clip_g_dim), dtype, device)?,
        clip_l_pooled: Tensor::zeros((batch, cfg.clip_l_dim), dtype, device)?,
        clip_g_pooled: Tensor::zeros((batch, cfg.clip_g_dim), dtype, device)?,
        t5_hidden: Tensor::zeros((batch, cfg.t5_seq_len, cfg.t5_dim), dtype, device)?,
    })
}

/// CLIP token cap (both encoders). SD3.5 truncates/pads the prompt to 77 tokens.
const CLIP_MAX_LEN: usize = 77;

/// T5 pad token id (`<pad>`). SD3.5 pads the T5 sequence to `t5_seq_len` with this id and attends
/// every position (no T5 attention mask), so the padded length is parity-critical.
const T5_PAD_TOKEN_ID: u32 = 0;

/// Resolve a CLIP encoder's **pad token id** from its diffusers `tokenizer_config.json` `pad_token`
/// (mapped through the tokenizer vocab), falling back to `eos_id` when the config/token is absent.
///
/// This is parity-critical and **differs between the two SD3.5 CLIP encoders** (sc-9076): the HF
/// `CLIPTokenizer` pads with `padding="max_length"` to 77 using the tokenizer's configured
/// `pad_token`, and the two encoders configure it differently —
///  - **CLIP-L** (`tokenizer/`): `pad_token = "<|endoftext|>"` → pad id **49407** (== eos);
///  - **CLIP-bigG** (`tokenizer_2/`): `pad_token = "!"` → pad id **0** (NOT eos).
///
/// Hardcoding eos for both (the pre-sc-9076 behaviour) padded bigG with 49407 instead of 0, which
/// diverges the bigG penultimate hidden state on every pad position under causal attention, and hence
/// the joint CLIP context fed to the MMDiT. The pooled vector was unaffected (it is read at the EOS
/// position, which precedes the pad tail), so only a component-parity run against diffusers surfaces
/// it. Caught by `tests/component_parity.rs`.
/// The `<|endoftext|>` id for a CLIP tokenizer (49407 canonical) — the pooling anchor + pad fallback.
fn clip_eos_id(tok: &Tokenizer) -> CandleResult<u32> {
    tok.get_vocab(true)
        .get("<|endoftext|>")
        .copied()
        .ok_or_else(|| CandleError::Msg("sd3: CLIP tokenizer missing <|endoftext|>".into()))
}

fn resolve_clip_pad_id(dir: &Path, tok: &Tokenizer, eos_id: u32) -> u32 {
    // `tokenizer_config.json` -> `pad_token` (a string, e.g. "!" or "<|endoftext|>"); map to its id.
    let cfg = dir.join("tokenizer_config.json");
    let pad_str = std::fs::read_to_string(&cfg)
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        .and_then(|v| match &v["pad_token"] {
            // `pad_token` is either a bare string or an `AddedToken`-shaped object with `content`.
            serde_json::Value::String(s) => Some(s.clone()),
            serde_json::Value::Object(o) => {
                o.get("content").and_then(|c| c.as_str().map(String::from))
            }
            _ => None,
        });
    match pad_str {
        Some(s) => tok.get_vocab(true).get(&s).copied().unwrap_or(eos_id),
        None => eos_id,
    }
}

/// Right-pad / hard-truncate a CLIP token row to exactly `CLIP_MAX_LEN`. SD3.5's diffusers pipeline
/// truncates to the model max (77) with a warning; we truncate (keeping BOS + the leading content)
/// and re-append the EOS so the pooled EOS lookup still finds it. Pads with the encoder's pad id.
fn fit_clip_tokens(mut ids: Vec<u32>, pad_id: u32, eos_id: u32) -> Vec<u32> {
    if ids.len() > CLIP_MAX_LEN {
        ids.truncate(CLIP_MAX_LEN);
        // Force the last slot to EOS so `eos_position` still selects a real EOS.
        *ids.last_mut().unwrap() = eos_id;
    } else {
        ids.resize(CLIP_MAX_LEN, pad_id);
    }
    ids
}

/// The EOS position of a CLIP token row = the **first** occurrence of the EOS id (sc-8982).
///
/// HF `CLIPTextModel` pools the final hidden state at `input_ids.argmax(-1)` (EOS =
/// `<|endoftext|>` = 49407 is the highest id), and torch's `argmax` returns the FIRST maximal
/// index. CLIP-L pads with the EOS id (its `pad_token == eos`), so every pad slot ties at the max —
/// the pooled hidden must come from the first EOS, not a trailing pad (the hidden states differ under
/// causal attention). bigG pads with `!` (0, sc-9076), a unique-max EOS, but the first-occurrence
/// lookup is correct either way. Falls back to the first arg-max index if `eos_id` is absent
/// (torch-`argmax` parity for a degenerate row).
fn eos_position(ids: &[u32], eos_id: u32) -> usize {
    ids.iter().position(|&v| v == eos_id).unwrap_or_else(|| {
        let max = ids.iter().copied().max().unwrap_or(0);
        ids.iter().position(|&v| v == max).unwrap_or(0)
    })
}

/// Pool a CLIP final-norm hidden state `[1, seq, embed]` at the row's EOS position (sc-8982):
/// returns `[1, embed]` taken at the FIRST occurrence of `eos_id` in `ids` (diffusers' pooled
/// `text_embeds` lookup), NOT the last sequence slot (a trailing EOS-id pad token).
fn pool_hidden_at_eos(final_hidden: &Tensor, ids: &[u32], eos_id: u32) -> Result<Tensor> {
    let eos = eos_position(ids, eos_id);
    final_hidden.i((0, eos))?.unsqueeze(0)
}

/// The three loaded SD3.5 text encoders + their tokenizers and pooled-projection heads. Built once
/// per model (held resident by the pipeline); [`encode`](Self::encode) is called per request and
/// produces the [`EncoderOutputs`] the [`aggregate`] step combines.
///
/// - CLIP-L (`text_encoder/`, embed 768) and CLIP-bigG (`text_encoder_2/`, embed 1280): the
///   **penultimate** hidden state (`hidden_states[-2]`, pre-final-norm — diffusers
///   `output_hidden_states=True`) feeds the joint context; the **final**-norm hidden at the EOS
///   position, projected through each encoder's `text_projection`, feeds the pooled vector.
/// - T5-XXL (`text_encoder_3/`, hidden 4096): the full encoder sequence, padded to `t5_seq_len`.
pub struct Sd3TextEncoders {
    tok_l: Tokenizer,
    tok_g: Tokenizer,
    tok_t5: Tokenizer,
    clip_l: clip::ClipTextTransformer,
    clip_g: clip::ClipTextTransformer,
    /// CLIP-L `text_projection.weight` (`[768, 768]`, no bias).
    proj_l: Linear,
    /// CLIP-bigG `text_projection.weight` (`[1280, 1280]`, no bias).
    proj_g: Linear,
    /// CLIP-L pad token id — `<|endoftext|>` (49407) per its `tokenizer_config.json` (sc-9076).
    pad_l: u32,
    /// CLIP-bigG pad token id — `!` (0) per its `tokenizer_config.json` (sc-9076). Differs from L.
    pad_g: u32,
    t5: T5EncoderModel,
    t5_seq_len: usize,
    device: Device,
    dtype: DType,
}

impl Sd3TextEncoders {
    /// Load the three encoders from a `stabilityai/stable-diffusion-3.5-*` diffusers snapshot:
    /// `text_encoder/` (CLIP-L), `text_encoder_2/` (CLIP-bigG), `text_encoder_3/` (T5-XXL). The two
    /// CLIP tokenizers load from `tokenizer.json` when present and otherwise are **synthesized** from
    /// the stock `vocab.json` + `merges.txt` (sc-8500; see [`crate::clip_tokenizer`]); T5 ships its
    /// own `tokenizer.json`. `t5_seq_len` is the configured T5 length (256 default).
    pub fn load(
        root: &Path,
        t5_seq_len: usize,
        device: &Device,
        dtype: DType,
    ) -> CandleResult<Self> {
        let cfg_l = clip::Config::sdxl(); // CLIP-L (openai/clip-vit-large-patch14, embed 768)
        let cfg_g = clip::Config::sdxl2(); // OpenCLIP bigG (embed 1280)

        // CLIP-L / CLIP-bigG: load `tokenizer.json` if present, else SYNTHESIZE the CLIP
        // byte-level BPE from `vocab.json` + `merges.txt` (a stock gated diffusers SD3.5
        // download ships no `tokenizer.json` for the CLIP encoders — sc-8500).
        let tok_l = crate::clip_tokenizer::load_clip_tokenizer(&root.join("tokenizer"), "CLIP-L")?;
        let tok_g =
            crate::clip_tokenizer::load_clip_tokenizer(&root.join("tokenizer_2"), "CLIP-bigG")?;
        // Per-encoder pad token id (sc-9076): resolved from each `tokenizer_config.json` `pad_token`,
        // NOT hardcoded to eos — bigG pads with `!` (0), L pads with `<|endoftext|>` (49407). The
        // eos id (49407 in the canonical CLIP vocab) is the fallback for either.
        let eos_l = clip_eos_id(&tok_l)?;
        let eos_g = clip_eos_id(&tok_g)?;
        let pad_l = resolve_clip_pad_id(&root.join("tokenizer"), &tok_l, eos_l);
        let pad_g = resolve_clip_pad_id(&root.join("tokenizer_2"), &tok_g, eos_g);
        // T5 ships its own `tokenizer.json` in a stock snapshot (out of scope for sc-8500).
        let tok_t5 = Tokenizer::from_file(root.join("tokenizer_3/tokenizer.json"))
            .map_err(|e| CandleError::Msg(format!("sd3: load T5 tokenizer: {e}")))?;

        let l_file = single_safetensors(root, "text_encoder")?;
        let g_file = single_safetensors(root, "text_encoder_2")?;
        let clip_l = stable_diffusion::build_clip_transformer(&cfg_l, &l_file, device, dtype)?;
        let clip_g = stable_diffusion::build_clip_transformer(&cfg_g, &g_file, device, dtype)?;
        let proj_l = load_text_projection(&l_file, "text_encoder", device, dtype)?;
        let proj_g = load_text_projection(&g_file, "text_encoder_2", device, dtype)?;

        // T5-XXL (`text_encoder_3/`, sharded; config.json alongside).
        let t5_dir = root.join("text_encoder_3");
        let t5_cfg: T5Config = {
            let cfg = std::fs::read_to_string(t5_dir.join("config.json")).map_err(|e| {
                CandleError::Msg(format!("sd3: read text_encoder_3/config.json: {e}"))
            })?;
            serde_json::from_str(&cfg)
                .map_err(|e| CandleError::Msg(format!("sd3: parse T5 config.json: {e}")))?
        };
        let t5_files = safetensors_in(&t5_dir)?;
        let t5_vb = candle_gen::mmap_var_builder(&t5_files, dtype, device)?;
        let t5 = T5EncoderModel::load(t5_vb, &t5_cfg)?;

        Ok(Self {
            tok_l,
            tok_g,
            tok_t5,
            clip_l,
            clip_g,
            proj_l,
            proj_g,
            pad_l,
            pad_g,
            t5,
            t5_seq_len,
            device: device.clone(),
            dtype,
        })
    }

    /// Run one CLIP encoder for `prompt`: returns `(penultimate [1, 77, embed], pooled [1, embed])`.
    /// The penultimate hidden is the pre-final-norm `hidden_states[-2]`; the pooled is the EOS-position
    /// final-norm hidden projected through that encoder's `text_projection`. `pad_id` is the encoder's
    /// diffusers `pad_token` id — differs between L (49407) and bigG (0), see [`resolve_clip_pad_id`].
    fn encode_clip(
        &self,
        tok: &Tokenizer,
        clip: &clip::ClipTextTransformer,
        proj: &Linear,
        pad_id: u32,
        prompt: &str,
    ) -> CandleResult<(Tensor, Tensor)> {
        let eos_id = clip_eos_id(tok)?;
        // SD3.5 CLIP pads with the encoder's configured pad token (L: `<|endoftext|>`; bigG: `!` = 0),
        // then truncates/pads to 77 (sc-9076). The pooling anchor is still the FIRST eos (sc-8982).
        let ids = tok
            .encode(prompt, true)
            .map_err(|e| CandleError::Msg(format!("sd3: CLIP tokenize: {e}")))?
            .get_ids()
            .to_vec();
        let ids = fit_clip_tokens(ids, pad_id, eos_id);
        let input = Tensor::new(ids.as_slice(), &self.device)?.reshape((1, CLIP_MAX_LEN))?;
        // `forward_until_encoder_layer(.., -2)` → (final-norm hidden, penultimate hidden).
        let (final_hidden, penult) = clip.forward_until_encoder_layer(&input, usize::MAX, -2)?;
        let pooled_eos = pool_hidden_at_eos(&final_hidden, &ids, eos_id)?; // [1, embed]
        let pooled = proj.forward(&pooled_eos)?;
        Ok((penult.to_dtype(self.dtype)?, pooled.to_dtype(self.dtype)?))
    }

    /// Encode `prompt` into the per-encoder [`EncoderOutputs`] (batch 1). T5 is tokenized and padded
    /// to `t5_seq_len` with the pad id; every position is attended (no T5 mask), matching diffusers.
    pub fn encode(&mut self, prompt: &str) -> CandleResult<EncoderOutputs> {
        let (clip_l_hidden, clip_l_pooled) =
            self.encode_clip(&self.tok_l, &self.clip_l, &self.proj_l, self.pad_l, prompt)?;
        let (clip_g_hidden, clip_g_pooled) =
            self.encode_clip(&self.tok_g, &self.clip_g, &self.proj_g, self.pad_g, prompt)?;

        // T5 sequence, padded/truncated to t5_seq_len.
        let mut t5_ids: Vec<u32> = self
            .tok_t5
            .encode(prompt, true)
            .map_err(|e| CandleError::Msg(format!("sd3: T5 tokenize: {e}")))?
            .get_ids()
            .to_vec();
        t5_ids.truncate(self.t5_seq_len);
        t5_ids.resize(self.t5_seq_len, T5_PAD_TOKEN_ID);
        let t5_input = Tensor::new(t5_ids.as_slice(), &self.device)?.unsqueeze(0)?;
        let t5_hidden = self.t5.forward(&t5_input)?.to_dtype(self.dtype)?; // [1, t5_seq_len, 4096]

        Ok(EncoderOutputs {
            clip_l_hidden,
            clip_g_hidden,
            clip_l_pooled,
            clip_g_pooled,
            t5_hidden,
        })
    }

    /// A/B test hook (sc-9581): encode `prompt` exactly like [`Self::encode`] but with the CLIP-bigG
    /// pad token forced to `pad_g_override` instead of the resolved [`Self::pad_g`]. Used only to
    /// reproduce the pre-sc-9076 behavior (bigG padded with eos 49407) against the fixed behavior
    /// (bigG padded with its configured `!` = 0) so a test can measure the end-to-end conditioning
    /// difference the fix produces. Never used in production.
    #[cfg(feature = "sc9581-ab")]
    pub fn encode_with_pad_g(
        &mut self,
        prompt: &str,
        pad_g_override: u32,
    ) -> CandleResult<EncoderOutputs> {
        let (clip_l_hidden, clip_l_pooled) =
            self.encode_clip(&self.tok_l, &self.clip_l, &self.proj_l, self.pad_l, prompt)?;
        let (clip_g_hidden, clip_g_pooled) = self.encode_clip(
            &self.tok_g,
            &self.clip_g,
            &self.proj_g,
            pad_g_override,
            prompt,
        )?;

        let mut t5_ids: Vec<u32> = self
            .tok_t5
            .encode(prompt, true)
            .map_err(|e| CandleError::Msg(format!("sd3: T5 tokenize: {e}")))?
            .get_ids()
            .to_vec();
        t5_ids.truncate(self.t5_seq_len);
        t5_ids.resize(self.t5_seq_len, T5_PAD_TOKEN_ID);
        let t5_input = Tensor::new(t5_ids.as_slice(), &self.device)?.unsqueeze(0)?;
        let t5_hidden = self.t5.forward(&t5_input)?.to_dtype(self.dtype)?;

        Ok(EncoderOutputs {
            clip_l_hidden,
            clip_g_hidden,
            clip_l_pooled,
            clip_g_pooled,
            t5_hidden,
        })
    }

    /// The resolved CLIP-bigG pad token id (sc-9076/sc-9581 A/B introspection). bigG resolves to
    /// `!` = 0; CLIP-L to `<|endoftext|>` = 49407.
    #[cfg(feature = "sc9581-ab")]
    pub fn pad_g(&self) -> u32 {
        self.pad_g
    }
}

/// Resolve the single `model.safetensors` (or first sorted shard) in a snapshot component subdir.
fn single_safetensors(root: &Path, sub: &str) -> CandleResult<std::path::PathBuf> {
    let files = safetensors_in(&root.join(sub))?;
    Ok(files.into_iter().next().unwrap())
}

/// Sorted list of every `.safetensors` in `dir` (single-file or sharded), erroring if absent.
fn safetensors_in(dir: &Path) -> CandleResult<Vec<std::path::PathBuf>> {
    if !dir.is_dir() {
        return Err(CandleError::Msg(format!(
            "sd3 snapshot is missing the {} component directory",
            dir.display()
        )));
    }
    // Shared sorted-`.safetensors` resolver (sc-8999 / F-019); the crafted "missing dir" message
    // above stays local.
    candle_gen::sorted_safetensors(dir, "sd3")
}

/// Load a CLIP `text_projection.weight` (no bias) from a CLIP checkpoint into a [`Linear`]. SD3.5's
/// CLIP-L and bigG are `CLIPTextModelWithProjection`s — `build_clip_transformer` reads only the
/// `text_model.*`; the pooled head's projection lives at the top level as `text_projection.weight`.
fn load_text_projection(
    file: &Path,
    sub: &str,
    device: &Device,
    dtype: DType,
) -> CandleResult<Linear> {
    // Header-only mmap read of the single pooled-head tensor (sc-8990 / F-010): `build_clip_transformer`
    // already read this same file for `text_model.*`, so materializing the whole CLIP checkpoint on the
    // GPU again just to grab one weight cost ~1.7 GB of transient VRAM plus a second full disk read.
    let w = candle_gen::load_one_tensor(
        file,
        "text_projection.weight",
        dtype,
        device,
        &format!("sd3 conditioning ({sub}/)"),
    )?;
    Ok(Linear::new(w, None))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Device;

    fn fixture(cfg: &Sd3Config, batch: usize) -> EncoderOutputs {
        let dev = Device::Cpu;
        // Distinctive fill values per source so the concat ORDER is observable in the output.
        EncoderOutputs {
            clip_l_hidden: Tensor::full(1f32, (batch, cfg.clip_seq_len, cfg.clip_l_dim), &dev)
                .unwrap(),
            clip_g_hidden: Tensor::full(2f32, (batch, cfg.clip_seq_len, cfg.clip_g_dim), &dev)
                .unwrap(),
            clip_l_pooled: Tensor::full(3f32, (batch, cfg.clip_l_dim), &dev).unwrap(),
            clip_g_pooled: Tensor::full(4f32, (batch, cfg.clip_g_dim), &dev).unwrap(),
            t5_hidden: Tensor::full(5f32, (batch, cfg.t5_seq_len, cfg.t5_dim), &dev).unwrap(),
        }
    }

    #[test]
    fn aggregate_shapes_match_sd35_defaults() {
        let cfg = Sd3Config::large();
        let enc = fixture(&cfg, 1);
        let out = aggregate(&cfg, &enc).unwrap();
        // pooled = 768 + 1280 = 2048.
        assert_eq!(out.pooled.dims(), &[1, 2048]);
        // context = (77 + 256) x 4096 = 333 x 4096.
        assert_eq!(out.context.dims(), &[1, 333, 4096]);
    }

    #[test]
    fn pooled_concat_order_is_l_then_g() {
        let cfg = Sd3Config::large();
        let enc = fixture(&cfg, 1);
        let out = aggregate(&cfg, &enc).unwrap();
        let v = out.pooled.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // First 768 from CLIP-L (filled 3), next 1280 from bigG (filled 4).
        assert!(
            v[..768].iter().all(|&x| x == 3.0),
            "CLIP-L pooled goes first"
        );
        assert!(
            v[768..2048].iter().all(|&x| x == 4.0),
            "bigG pooled goes second"
        );
    }

    #[test]
    fn context_layout_is_clip_padded_then_t5() {
        let cfg = Sd3Config::large();
        let enc = fixture(&cfg, 1);
        let out = aggregate(&cfg, &enc).unwrap();
        // Token 0 (a CLIP token): hidden = [CLIP-L 768 = 1, bigG 1280 = 2, zero-pad 2048 = 0].
        let tok0 = out.context.i((0, 0)).unwrap().to_vec1::<f32>().unwrap();
        assert!(tok0[..768].iter().all(|&x| x == 1.0), "clip-l region");
        assert!(tok0[768..2048].iter().all(|&x| x == 2.0), "bigg region");
        assert!(
            tok0[2048..4096].iter().all(|&x| x == 0.0),
            "zero-pad region"
        );
        // Token 77 (the first T5 token): all 5 across the full 4096 width.
        let tok_t5 = out.context.i((0, 77)).unwrap().to_vec1::<f32>().unwrap();
        assert!(
            tok_t5.iter().all(|&x| x == 5.0),
            "t5 region is full-width 4096"
        );
    }

    #[test]
    fn t5_length_drives_context_seq() {
        let mut cfg = Sd3Config::large();
        cfg.t5_seq_len = 512;
        let enc = fixture(&cfg, 2);
        let out = aggregate(&cfg, &enc).unwrap();
        assert_eq!(out.context.dims(), &[2, 77 + 512, 4096]);
        assert_eq!(out.pooled.dims(), &[2, 2048]);
    }

    #[test]
    fn aggregate_rejects_misshaped_clip_width() {
        // A config whose clip_concat_dim disagrees with clip_l_dim + clip_g_dim trips the guard.
        let mut cfg = Sd3Config::large();
        cfg.clip_concat_dim = 999; // != 768 + 1280
        let enc = fixture(&cfg, 1);
        assert!(aggregate(&cfg, &enc).is_err());
    }

    /// Build a tiny WordLevel [`Tokenizer`] whose vocab maps the given `(token, id)` pairs — enough to
    /// exercise `resolve_clip_pad_id`'s string→id lookup without a real CLIP snapshot.
    fn tiny_tokenizer(dir: &std::path::Path, pairs: &[(&str, u32)]) -> tokenizers::Tokenizer {
        use tokenizers::models::wordlevel::WordLevel;
        // Build a WordLevel model from a temp `vocab.json` (avoids the builder's `ahash::AHashMap`
        // vocab type, which isn't a direct dep here). The vocab just needs the pad/eos strings.
        let vocab_json: String = format!(
            "{{{}}}",
            pairs
                .iter()
                .map(|(t, i)| format!("{}:{i}", serde_json::to_string(t).unwrap()))
                .collect::<Vec<_>>()
                .join(",")
        );
        let vocab_path = dir.join("vocab.json");
        std::fs::write(&vocab_path, vocab_json).unwrap();
        let wl = WordLevel::from_file(vocab_path.to_str().unwrap(), "<unk>".into()).unwrap();
        tokenizers::Tokenizer::new(wl)
    }

    /// `resolve_clip_pad_id` (sc-9076) reads each encoder's `tokenizer_config.json` `pad_token` and
    /// maps it through the vocab: bigG's `!` → 0, L's `<|endoftext|>` → 49407; missing config/token
    /// falls back to eos. The pre-fix hardcoded-eos behaviour padded bigG with 49407 (wrong).
    #[test]
    fn resolve_clip_pad_id_reads_per_encoder_pad_token() {
        let dir = std::env::temp_dir().join(format!("sd3_pad_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let tok = tiny_tokenizer(&dir, &[("<unk>", 1), ("!", 0), ("<|endoftext|>", 49407)]);

        // bigG-style config: pad_token = "!" (a bare string) -> id 0.
        std::fs::write(dir.join("tokenizer_config.json"), r#"{"pad_token":"!"}"#).unwrap();
        assert_eq!(
            resolve_clip_pad_id(&dir, &tok, 49407),
            0,
            "bigG pads with ! (0)"
        );

        // L-style config: pad_token = "<|endoftext|>" -> id 49407 (== eos).
        std::fs::write(
            dir.join("tokenizer_config.json"),
            r#"{"pad_token":"<|endoftext|>"}"#,
        )
        .unwrap();
        assert_eq!(
            resolve_clip_pad_id(&dir, &tok, 49407),
            49407,
            "L pads with eos"
        );

        // AddedToken-object shaped pad_token (`{"content": "!"}`) -> id 0.
        std::fs::write(
            dir.join("tokenizer_config.json"),
            r#"{"pad_token":{"content":"!","lstrip":false}}"#,
        )
        .unwrap();
        assert_eq!(
            resolve_clip_pad_id(&dir, &tok, 49407),
            0,
            "object-shaped pad_token"
        );

        // No config file -> fall back to eos.
        std::fs::remove_file(dir.join("tokenizer_config.json")).unwrap();
        assert_eq!(
            resolve_clip_pad_id(&dir, &tok, 49407),
            49407,
            "missing config -> eos fallback"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `fit_clip_tokens` pads short rows to 77 and hard-truncates long rows, keeping an EOS in the
    /// last slot so the pooled EOS lookup still lands on a real EOS (not a content token).
    #[test]
    fn fit_clip_tokens_pads_and_truncates_with_eos() {
        let eos = 49407u32;
        // Short row pads to 77 with the pad id.
        let short = fit_clip_tokens(vec![49406, 320, eos], 9, eos);
        assert_eq!(short.len(), 77);
        assert_eq!(&short[..3], &[49406, 320, eos]);
        assert!(short[3..].iter().all(|&x| x == 9));
        // Over-long row truncates to 77 and forces the last slot to EOS.
        let long: Vec<u32> = (0..100).collect();
        let fit = fit_clip_tokens(long, 9, eos);
        assert_eq!(fit.len(), 77);
        assert_eq!(*fit.last().unwrap(), eos);
    }

    /// `eos_position` finds the FIRST EOS, even when the row is padded with the EOS id itself —
    /// the production case: `encode_clip` pads with `pad_id == eos_id`, so every pad slot ties at
    /// the max token id and a last-maximal lookup (the sc-8982 bug) lands on the trailing pad.
    #[test]
    fn eos_position_is_first_eos() {
        let eos = 49407u32;
        // Row padded with a distinct pad id (< EOS): EOS is the unique max.
        assert_eq!(eos_position(&[49406, 320, eos, 9, 9], eos), 2);
        assert_eq!(eos_position(&[49406, 1, 2, 3, eos], eos), 4);
        // Row padded with the EOS id (the production path): must pick the FIRST EOS, not the last
        // pad slot. Pre-fix `max_by_key` returned 4 here.
        assert_eq!(eos_position(&[49406, 320, eos, eos, eos], eos), 2);
        // Exactly what `fit_clip_tokens(.., eos, eos)` produces for a short prompt.
        let ids = fit_clip_tokens(vec![49406, 320, eos], eos, eos);
        assert_eq!(eos_position(&ids, eos), 2);
        // Degenerate row with no EOS: torch-argmax parity (first maximal index).
        assert_eq!(eos_position(&[5, 9, 9, 3], eos), 1);
    }

    /// The pooled vector must equal the final hidden state at the FIRST EOS position, not the last
    /// sequence slot. Synthetic CPU hidden state where each position is filled with its own index,
    /// row padded with the EOS id exactly as `encode_clip` pads short prompts — the pre-fix code
    /// pooled position `seq-1` (a trailing pad) and fails this test.
    #[test]
    fn pooled_hidden_is_taken_at_first_eos_not_last_pad() {
        let eos = 49407u32;
        let (seq, embed) = (77usize, 8usize);
        // ids = short prompt [BOS, tok, EOS] padded to 77 with the EOS id (pad_id == eos_id).
        let ids = fit_clip_tokens(vec![49406, 320, eos], eos, eos);
        assert_eq!(ids.len(), seq);
        // hidden[0, p, :] = p, so the pooled row identifies the position it was taken from.
        let rows: Vec<f32> = (0..seq).flat_map(|p| vec![p as f32; embed]).collect();
        let hidden = Tensor::from_vec(rows, (1, seq, embed), &Device::Cpu).unwrap();
        let pooled = pool_hidden_at_eos(&hidden, &ids, eos).unwrap();
        assert_eq!(pooled.dims(), &[1, embed]);
        let v = pooled.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(
            v.iter().all(|&x| x == 2.0),
            "pooled must come from the first EOS (position 2), got values {v:?} (position 76 = \
             trailing pad would be the sc-8982 bug)"
        );
    }

    #[test]
    fn zeroed_outputs_aggregate_to_correct_shape() {
        let cfg = Sd3Config::large();
        let enc = zeroed_outputs(&cfg, 1, DType::F32, &Device::Cpu).unwrap();
        let out = aggregate(&cfg, &enc).unwrap();
        assert_eq!(out.context.dims(), &[1, 333, 4096]);
        assert_eq!(out.pooled.dims(), &[1, 2048]);
    }
}
