//! Crate-private shared prompt-encode + component-load routines for the Wan family (sc-9000 / F-020).
//!
//! The 5B provider ([`crate::lib`]), the 14B MoE provider ([`crate::wan14b`]), the VACE provider
//! ([`crate::model_vace`]), and the LoRA trainer ([`crate::training`]) all ran an **identical**
//! tokenize → empty-guard → UMT5-encode → zero-pad/truncate-to-512 routine (four copies, ~200 lines,
//! including the identical sc-7078 empty-prompt comment) and a bespoke `component_vb`. A future
//! tokenizer fix had to land 4× — the exact bug class sc-7078 (the empty-prompt CUDA gather) and
//! sc-3697 (the 512-pad collapse) were. This module is the single home for both.
//!
//! ## Two load-bearing invariants (do NOT regress)
//! - **512-pad (sc-3697):** the Wan DiT cross-attends over a context **zero-padded to `max_length`
//!   (512)** — the reference `WanPipeline` pads the UMT5 embeds to 512 before the transformer (the
//!   model was trained that way). Feeding only the real tokens silently breaks conditioning.
//! - **Empty-prompt guard (sc-7078):** the `gen_core` tokenizer short-circuits an empty prompt to
//!   zero ids, but a 0-length sequence builds a degenerate `(1,1)` tensor whose 0-element f32
//!   embedding gather reads out of bounds on CUDA (`CUDA_ERROR_ILLEGAL_ADDRESS`, surfacing later as a
//!   misleading `CUBLAS_STATUS_EXECUTION_FAILED`). Emit one pad token so a 0-length sequence never
//!   reaches the gather.

use std::path::Path;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use candle_gen::{CandleError, Result as CResult};

use crate::config::TextEncoderConfig;
use crate::text_encoder::Umt5Encoder;

/// mmap a snapshot component subdir `sub` (under `root`) into a [`VarBuilder`] at `dtype`/`device`.
///
/// This is the Wan flavor of the shared [`candle_gen::loader::component_vb`]: it keeps the crafted,
/// **provider-specific** "missing directory" diagnostic (which names the *expected* Wan snapshot — e.g.
/// `Wan2.2-TI2V-5B` vs `Wan2.2-T2V-A14B` vs `Wan2.1-VACE-14B`) that the generic loader does not
/// reproduce. `snapshot_desc` is that per-provider description (e.g. `"Wan2.2-TI2V-5B diffusers"`);
/// `label` prefixes the shared sorted-`.safetensors` resolver's errors (sc-8999 / F-019).
pub(crate) fn component_vb(
    root: &Path,
    sub: &str,
    dtype: DType,
    device: &Device,
    label: &str,
    snapshot_desc: &str,
) -> CResult<VarBuilder<'static>> {
    let dir = root.join(sub);
    if !dir.is_dir() {
        return Err(CandleError::Msg(format!(
            "{label} snapshot is missing the {sub}/ dir (expected a {snapshot_desc} \
             snapshot at {})",
            root.display()
        )));
    }
    // Shared sorted-`.safetensors` → mmap (sc-8999 / F-019); the crafted "missing dir" message above
    // stays local (it names the expected Wan snapshot).
    let files = candle_gen::sorted_safetensors(&dir, label)?;
    candle_gen::mmap_var_builder(&files, dtype, device)
}

/// Build the Wan UMT5 tokenizer from `root/tokenizer/tokenizer.json` **once**, so callers can cache it
/// on their `Components` and reuse it across every encode (sc-8991 / F-011) instead of re-parsing the
/// multi-MB `tokenizer.json` on each prompt/branch. The [`TokenizerConfig`] here is byte-identical to
/// the one the per-encode load used, so the cached tokenizer yields the same ids. `label` prefixes the
/// load error (`"wan"`, `"wan-14b"`, `"wan-vace"`, `"wan trainer"`).
pub(crate) fn build_umt5_tokenizer(
    root: &Path,
    te_cfg: &TextEncoderConfig,
    label: &str,
) -> CResult<TextTokenizer> {
    TextTokenizer::from_file(
        root.join("tokenizer/tokenizer.json"),
        TokenizerConfig {
            max_length: te_cfg.max_length,
            pad_token_id: te_cfg.pad_token_id,
            chat_template: ChatTemplate::None,
            pad_to_max_length: false,
        },
    )
    .map_err(|e| CandleError::Msg(format!("{label}: load tokenizer: {e}")))
}

/// Tokenize `prompt` → UMT5-encode → zero-pad/truncate to `max_length` (512) → `[1, 512, 4096]` in
/// `out_dtype`. The single home for the Wan text-encode routine (sc-9000 / F-020).
///
/// - `tok` is the cached UMT5 tokenizer ([`build_umt5_tokenizer`]); the caller loads it once and
///   reuses it across encodes (sc-8991 / F-011) rather than re-parsing `tokenizer.json` per prompt.
/// - `label` prefixes the tokenize error text (e.g. `"wan"`, `"wan-14b"`, `"wan-vace"`,
///   `"wan trainer"`).
/// - `out_dtype` is the dtype of the returned embeds **and** the zero-pad. The three inference
///   providers run the encoder at f32 and pass `ENC_DTYPE` (= f32), so the cast is a no-op and the
///   pad is f32 — byte-identical to the pre-consolidation copies. The trainer loads the encoder at
///   **bf16** and passes f32, reproducing its (previously drifted) `.to_dtype(F32)` upcast exactly.
///
/// See the module docs for the two load-bearing invariants (512-pad sc-3697, empty-prompt guard
/// sc-7078) this routine guards.
pub(crate) fn umt5_encode_padded(
    tok: &TextTokenizer,
    te_cfg: &TextEncoderConfig,
    te: &Umt5Encoder,
    prompt: &str,
    device: &Device,
    out_dtype: DType,
    label: &str,
) -> CResult<Tensor> {
    let out = tok
        .tokenize(prompt)
        .map_err(|e| CandleError::Msg(format!("{label}: tokenize: {e}")))?;
    let mut ids: Vec<u32> = out.ids.iter().map(|&i| i as u32).collect();
    if ids.is_empty() {
        // The gen_core tokenizer short-circuits an empty prompt to zero ids, but UMT5/T5 encode the
        // empty string as a single token. A 0-length sequence here would build a degenerate `(1,1)`
        // tensor (the old `.max(1)` padded the *shape* but not the data), and the f32 embedding gather
        // over zero indices is a 0-element CUDA `index_select` that reads out of bounds →
        // `CUDA_ERROR_ILLEGAL_ADDRESS` (it surfaces deferred at the next cublas call as a misleading
        // `CUBLAS_STATUS_EXECUTION_FAILED`). Emit one pad token so a 0-length sequence never reaches
        // the gather. (sc-7078)
        ids.push(te_cfg.pad_token_id as u32);
    }
    let len = ids.len();
    let input_ids = Tensor::from_vec(ids, (1, len), device)?;
    let embeds = te.encode(&input_ids)?.to_dtype(out_dtype)?; // [1, S, 4096]

    // The Wan DiT cross-attends over a context **zero-padded to `max_length` (512)** — the reference
    // `WanPipeline` pads the UMT5 embeds to 512 before the transformer (the model was trained that
    // way). Feeding only the real tokens silently breaks conditioning. (sc-3697)
    let max_len = te_cfg.max_length;
    let dim = embeds.dim(2)?;
    match len.cmp(&max_len) {
        std::cmp::Ordering::Less => {
            let pad = Tensor::zeros((1, max_len - len, dim), out_dtype, device)?;
            Ok(Tensor::cat(&[&embeds, &pad], 1)?)
        }
        std::cmp::Ordering::Greater => Ok(embeds.narrow(1, 0, max_len)?),
        std::cmp::Ordering::Equal => Ok(embeds),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Device;
    use candle_gen::candle_nn::VarBuilder;

    /// A deliberately tiny UMT5 config (1 layer, small dims) that keeps the load-bearing
    /// `max_length = 512` so the pad-to-512 path is exercised. `vocab_size` covers the tiny
    /// tokenizer's ids.
    fn tiny_cfg() -> TextEncoderConfig {
        TextEncoderConfig {
            vocab_size: 16,
            d_model: 8,
            d_ff: 16,
            d_kv: 4,
            num_heads: 2,
            num_layers: 1,
            num_buckets: 8,
            max_distance: 128,
            eps: 1e-6,
            max_length: 512,
            pad_token_id: 0,
        }
    }

    /// Build a tiny UMT5 encoder (zeros weights — enough for shape/finite/determinism assertions) plus
    /// a minimal on-disk `tokenizer/tokenizer.json` under `dir` and the matching cached tokenizer.
    /// Returns the config + encoder + tokenizer.
    fn tiny_encoder(dir: &Path) -> (TextEncoderConfig, Umt5Encoder, TextTokenizer) {
        let cfg = tiny_cfg();
        // A VarMap backend (not `VarBuilder::zeros`): the packed-detect loaders (sc-10025) probe
        // `{key}.scales` via `contains_tensor`, and the `Zeros` backend reports EVERY key present (so it
        // would spuriously fire the packed arm). A fresh VarMap reports only the keys the encoder
        // actually `get`s — no `.scales` → every leaf takes the dense arm, exactly as before. Weights are
        // init'd (finite) and reused within the built encoder, which is all these determinism/shape tests
        // need.
        let varmap = candle_gen::candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let te = Umt5Encoder::new(&cfg, vb).expect("tiny encoder");
        std::fs::create_dir_all(dir.join("tokenizer")).unwrap();
        std::fs::write(dir.join("tokenizer/tokenizer.json"), TINY_TOKENIZER_JSON).unwrap();
        let tok = build_umt5_tokenizer(dir, &cfg, "t").expect("tiny tokenizer");
        (cfg, te, tok)
    }

    // A minimal WordLevel `tokenizer.json` (a couple of words + unk), whitespace pre-tokenizer, no
    // chat template — just enough for `Tokenizer::from_file` to encode fixed prompts to small ids.
    const TINY_TOKENIZER_JSON: &str = r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": { "type": "Whitespace" },
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "vocab": { "<unk>": 0, "hello": 1, "world": 2, "<pad>": 3 },
    "unk_token": "<unk>"
  }
}"#;

    #[test]
    fn pads_to_max_length() {
        let tmp = tempdir();
        let (cfg, te, tok) = tiny_encoder(&tmp);
        let out = umt5_encode_padded(
            &tok,
            &cfg,
            &te,
            "hello world",
            &Device::Cpu,
            DType::F32,
            "t",
        )
        .expect("encode");
        // [1, 512, d_model] — the load-bearing 512 pad (sc-3697).
        assert_eq!(out.dim(0).unwrap(), 1);
        assert_eq!(out.dim(1).unwrap(), cfg.max_length);
        assert_eq!(out.dtype(), DType::F32);
    }

    #[test]
    fn empty_prompt_produces_valid_padded_uncond() {
        let tmp = tempdir();
        let (cfg, te, tok) = tiny_encoder(&tmp);
        // The empty-prompt path (sc-7078): must NOT build a 0-length sequence, must still pad to 512.
        let out = umt5_encode_padded(&tok, &cfg, &te, "", &Device::Cpu, DType::F32, "t")
            .expect("empty encode");
        assert_eq!(out.dim(1).unwrap(), cfg.max_length);
        // The uncond embedding is finite (no OOB gather garbage / NaN).
        let flat: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert!(flat.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn deterministic_for_fixed_prompt() {
        let tmp = tempdir();
        let (cfg, te, tok) = tiny_encoder(&tmp);
        let a = umt5_encode_padded(
            &tok,
            &cfg,
            &te,
            "hello world",
            &Device::Cpu,
            DType::F32,
            "t",
        )
        .unwrap();
        let b = umt5_encode_padded(
            &tok,
            &cfg,
            &te,
            "hello world",
            &Device::Cpu,
            DType::F32,
            "t",
        )
        .unwrap();
        let av: Vec<f32> = a.flatten_all().unwrap().to_vec1().unwrap();
        let bv: Vec<f32> = b.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(av, bv);
    }

    #[test]
    fn out_dtype_bf16_matches_trainer_path() {
        // The trainer loads the encoder at bf16 and upcasts to f32; here we exercise the dtype param
        // directly: an f32 encoder cast to bf16 output yields a bf16, 512-padded tensor.
        let tmp = tempdir();
        let (cfg, te, tok) = tiny_encoder(&tmp);
        let out =
            umt5_encode_padded(&tok, &cfg, &te, "hello", &Device::Cpu, DType::BF16, "t").unwrap();
        assert_eq!(out.dtype(), DType::BF16);
        assert_eq!(out.dim(1).unwrap(), cfg.max_length);
    }

    #[test]
    fn cached_tokenizer_matches_fresh_from_file() {
        // sc-8991 / F-011: the cached `build_umt5_tokenizer` must yield BYTE-IDENTICAL ids to a fresh
        // per-encode `from_file` load — caching only removes the redundant re-parse, never changes the
        // tokenization output. Cover a normal prompt AND the empty (uncond) short-circuit.
        let tmp = tempdir();
        let (cfg, _te, cached) = tiny_encoder(&tmp);
        for prompt in ["hello world", ""] {
            let fresh = build_umt5_tokenizer(&tmp, &cfg, "t").expect("fresh tokenizer");
            let cached_ids = cached.tokenize(prompt).unwrap().ids;
            let fresh_ids = fresh.tokenize(prompt).unwrap().ids;
            assert_eq!(cached_ids, fresh_ids, "prompt {prompt:?}");
        }
    }

    #[test]
    fn component_vb_missing_dir_errors_with_snapshot_desc() {
        let tmp = tempdir();
        // `VarBuilder` is not `Debug`, so match the error out rather than `unwrap_err()`.
        let msg = match component_vb(
            &tmp,
            "transformer",
            DType::F32,
            &Device::Cpu,
            "wan",
            "Wan2.2-TI2V-5B diffusers",
        ) {
            Ok(_) => panic!("expected a missing-dir error for an empty snapshot root"),
            Err(e) => format!("{e}"),
        };
        assert!(msg.contains("transformer/"), "msg: {msg}");
        assert!(msg.contains("Wan2.2-TI2V-5B diffusers"), "msg: {msg}");
    }

    fn tempdir() -> std::path::PathBuf {
        let base = std::env::temp_dir().join(format!(
            "wan-text-encode-test-{}-{}",
            std::process::id(),
            fastrand_u64()
        ));
        std::fs::create_dir_all(&base).unwrap();
        base
    }

    // Tiny non-crypto unique suffix (avoid pulling a dep just for tests).
    fn fastrand_u64() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    }
}
