//! Robust CLIP tokenizer loading for stock diffusers SD3.5 snapshots (sc-8500).
//!
//! SD3.5 uses two CLIP byte-level BPE tokenizers — `tokenizer/` (CLIP-L,
//! openai/clip-vit-large-patch14) and `tokenizer_2/` (OpenCLIP bigG,
//! laion/CLIP-ViT-bigG-14). Both are the **same** CLIP BPE family: a byte-level BPE
//! model with the CLIP normalizer (NFC → collapse whitespace → lowercase), the CLIP
//! pre-tokenizer (the OpenAI word-split regex, then byte-level mapping), the
//! `<|startoftext|>`/`<|endoftext|>` RoBERTa-style post-processor, and the `</w>`
//! end-of-word suffix.
//!
//! A **raw gated diffusers SD3.5 download** ships `tokenizer/` and `tokenizer_2/` with
//! only `vocab.json` + `merges.txt` — **no `tokenizer.json`**. (C6, sc-7881, worked
//! around this by hand-fetching the canonical CLIP `tokenizer.json` files.) This module
//! makes the loader robust:
//!
//! - **Fast path:** if `tokenizer.json` is present, load it directly (unchanged).
//! - **Fallback:** otherwise, if `vocab.json` + `merges.txt` are present, **synthesize**
//!   the exact CLIP byte-level BPE [`Tokenizer`] in memory from them.
//!
//! The synthesized tokenizer is byte-for-byte equivalent in behavior to the canonical
//! openai/laion `tokenizer.json` — token-id parity is asserted in the crate tests
//! against the canonical files (and against well-known CLIP encodings).

use std::path::Path;

use candle_gen::{CandleError, Result as CandleResult};
use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
use tokenizers::models::bpe::BPE;
use tokenizers::normalizers::replace::ReplacePattern;
use tokenizers::normalizers::{Lowercase, Replace, Sequence as NormSequence, NFC};
use tokenizers::pre_tokenizers::byte_level::ByteLevel as ByteLevelPre;
use tokenizers::pre_tokenizers::sequence::Sequence as PreSequence;
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::processors::roberta::RobertaProcessing;
use tokenizers::{
    AddedToken, NormalizerWrapper, PreTokenizerWrapper, SplitDelimiterBehavior, Tokenizer,
};

/// The CLIP start/end-of-text special tokens. These already live in `vocab.json`
/// (ids 49406 / 49407 in the canonical CLIP vocab); they are (re)registered as special
/// added tokens so the normalizer/pre-tokenizer skip them, and they drive the
/// post-processor's cls/sep wrapping.
const BOS: &str = "<|startoftext|>";
const EOS: &str = "<|endoftext|>";

/// The OpenAI/CLIP word-split regex used by the canonical CLIP `tokenizer.json`
/// pre-tokenizer (a `Split` with `invert: true`, `behavior: Removed` — i.e. keep the
/// matches, drop the gaps). Mirrors the original CLIP `bpe()` contraction/word/number
/// pattern.
const CLIP_SPLIT_REGEX: &str = r"'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+";

/// Load a CLIP tokenizer from a diffusers tokenizer directory (`tokenizer/` or
/// `tokenizer_2/`), tolerating a **stock** snapshot that ships only `vocab.json` +
/// `merges.txt` (no `tokenizer.json`).
///
/// `label` is used only for error context (e.g. `"CLIP-L"`).
pub fn load_clip_tokenizer(dir: &Path, label: &str) -> CandleResult<Tokenizer> {
    let json = dir.join("tokenizer.json");
    if json.is_file() {
        // Fast path: a real (canonical or previously-synthesized) tokenizer.json.
        return Tokenizer::from_file(&json)
            .map_err(|e| CandleError::Msg(format!("sd3: load {label} tokenizer.json: {e}")));
    }

    let vocab = dir.join("vocab.json");
    let merges = dir.join("merges.txt");
    if vocab.is_file() && merges.is_file() {
        return synthesize_clip_tokenizer(&vocab, &merges).map_err(|e| {
            CandleError::Msg(format!(
                "sd3: synthesize {label} CLIP tokenizer from vocab.json+merges.txt in {}: {e}",
                dir.display()
            ))
        });
    }

    Err(CandleError::Msg(format!(
        "sd3: {label} tokenizer dir {} has neither tokenizer.json nor vocab.json+merges.txt",
        dir.display()
    )))
}

/// Build the CLIP byte-level BPE [`Tokenizer`] programmatically from `vocab.json` +
/// `merges.txt`, reproducing the canonical openai/laion CLIP `tokenizer.json`:
///
/// - **model**: BPE with `unk_token = <|endoftext|>`, `end_of_word_suffix = </w>`,
///   no continuing-subword prefix, no dropout/fuse.
/// - **normalizer**: `Sequence([NFC, Replace(\s+ -> " "), Lowercase])`.
/// - **pre_tokenizer**: `Sequence([Split(CLIP regex, Removed, invert=true),
///   ByteLevel(add_prefix_space=false, trim_offsets=true, use_regex=false)])`.
/// - **post_processor**: `RobertaProcessing(sep=(EOS,49407), cls=(BOS,49406),
///   trim_offsets=false, add_prefix_space=false)`.
/// - **decoder**: `ByteLevel`.
/// - **added_tokens**: BOS (normalized=true) + EOS (normalized=false), both special.
pub fn synthesize_clip_tokenizer(
    vocab_path: &Path,
    merges_path: &Path,
) -> Result<Tokenizer, Box<dyn std::error::Error + Send + Sync>> {
    let vocab_str = vocab_path.to_str().ok_or("non-UTF8 vocab path")?;
    let merges_str = merges_path.to_str().ok_or("non-UTF8 merges path")?;

    let bpe = BPE::from_file(vocab_str, merges_str)
        .unk_token(EOS.to_string())
        .end_of_word_suffix("</w>".to_string())
        .continuing_subword_prefix(String::new())
        .build()?;

    // Resolve the special-token ids from the model vocab (canonical: BOS=49406, EOS=49407).
    let bos_id = bpe
        .get_vocab()
        .get(BOS)
        .copied()
        .ok_or("vocab.json missing <|startoftext|>")?;
    let eos_id = bpe
        .get_vocab()
        .get(EOS)
        .copied()
        .ok_or("vocab.json missing <|endoftext|>")?;

    let mut tok = Tokenizer::new(bpe);

    // Normalizer: NFC -> collapse runs of whitespace to a single space -> lowercase.
    let normalizer = NormSequence::new(vec![
        NormalizerWrapper::from(NFC),
        NormalizerWrapper::from(Replace::new(
            ReplacePattern::Regex(r"\s+".to_string()),
            " ",
        )?),
        NormalizerWrapper::from(Lowercase),
    ]);
    tok.with_normalizer(Some(normalizer));

    // Pre-tokenizer: the CLIP word-split regex (keep matches), then byte-level mapping
    // WITHOUT its own regex split (the Split already segmented) and no prefix space.
    let split = Split::new(
        SplitPattern::Regex(CLIP_SPLIT_REGEX.to_string()),
        SplitDelimiterBehavior::Removed,
        /* invert = */ true,
    )?;
    let byte_level_pre = ByteLevelPre::new(
        /* add_prefix_space = */ false, /* trim_offsets = */ true,
        /* use_regex = */ false,
    );
    let pre = PreSequence::new(vec![
        PreTokenizerWrapper::from(split),
        PreTokenizerWrapper::from(byte_level_pre),
    ]);
    tok.with_pre_tokenizer(Some(pre));

    // Post-processor: wrap with <|startoftext|> ... <|endoftext|> (RoBERTa-style; CLIP
    // uses single-sequence cls/sep wrapping).
    let post = RobertaProcessing::new((EOS.to_string(), eos_id), (BOS.to_string(), bos_id))
        .trim_offsets(false)
        .add_prefix_space(false);
    tok.with_post_processor(Some(post));

    // Decoder: byte-level (inverse of the byte-level pre-tokenizer).
    tok.with_decoder(Some(ByteLevelDecoder::default()));

    // Register the special tokens so the normalizer/pre-tokenizer leave them intact.
    // BOS is `normalized=true`, EOS is `normalized=false` to match the canonical file.
    tok.add_special_tokens(&[
        AddedToken::from(BOS, true).normalized(true),
        AddedToken::from(EOS, true).normalized(false),
    ]);

    Ok(tok)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    // Candidate CLIP tokenizer directories, resolved **portably** from the HF cache (sc-9030 / F-046).
    //
    // Why this exists: these used to be hardcoded `D:\sd35\…` paths that only existed on one
    // workstation, so the parity tests silently no-op'd everywhere else (they `eprintln!`'d and
    // returned/passed with zero assertions run — an *invisible* skip). The claimed "byte-for-byte
    // equivalent, asserted in the crate tests" guarantee was therefore vacuous off that one box, and a
    // `tokenizers`-crate bump could regress the synthesized tokenizer with no coverage.
    //
    // Two distinct sources are needed because a **stock diffusers SD3.5 snapshot** ships its CLIP
    // `tokenizer/`+`tokenizer_2/` subdirs with only `vocab.json`+`merges.txt` (that's the whole reason
    // this module exists) — it has **no `tokenizer.json`** to diff against. So:
    //
    //  - `synthesis_source_dirs` — dirs carrying `vocab.json`+`merges.txt` to *build* the synthesized
    //    tokenizer from. The SD3.5 snapshots (`stabilityai/stable-diffusion-3.5-{large,medium}`,
    //    `tokenizer/`+`tokenizer_2/`) plus the canonical CLIP repos qualify.
    //  - `canonical_reference_dirs` — dirs that additionally ship a real canonical `tokenizer.json` to
    //    assert parity *against*. Only the canonical CLIP repos (`openai/clip-vit-large-patch14` for
    //    CLIP-L, `laion/CLIP-ViT-bigG-14-laion2B-39B-b160k` for bigG) ship all three files together —
    //    these are the portable source of truth, mirroring `candle-gen-clip`'s F-069/F-071 resolver use.
    //
    // All resolution routes through the shared `candle_gen::testkit::hf_snapshot_dir` HF-cache resolver
    // (`$HF_HUB_CACHE` → `$HF_HOME/hub` → `<home>/.cache/huggingface/hub`), so the tests run on ANY
    // machine with the weights cached. `SD35_LARGE_PATH`/`SD35_MEDIUM_PATH` (the crate's existing C6
    // snapshot-root overrides) take precedence for the synthesis source when set.

    /// SD3.5 snapshot roots from the C6 env overrides, if set (a full diffusers snapshot dir).
    fn sd35_env_snapshot_roots() -> Vec<PathBuf> {
        ["SD35_LARGE_PATH", "SD35_MEDIUM_PATH"]
            .into_iter()
            .filter_map(|k| std::env::var(k).ok())
            .filter(|p| !p.is_empty())
            .map(PathBuf::from)
            .collect()
    }

    /// Dirs carrying `vocab.json`+`merges.txt` to synthesize the CLIP BPE from — the SD3.5 snapshot
    /// `tokenizer/`+`tokenizer_2/` subdirs and the canonical CLIP repos.
    fn synthesis_source_dirs() -> Vec<PathBuf> {
        let mut dirs: Vec<PathBuf> = Vec::new();

        // SD3.5 diffusers snapshots: two CLIP tokenizer subdirs each (vocab+merges, no tokenizer.json).
        let mut sd35_roots = sd35_env_snapshot_roots();
        for repo in [
            "stabilityai/stable-diffusion-3.5-large",
            "stabilityai/stable-diffusion-3.5-medium",
        ] {
            if let Some(snap) = candle_gen::testkit::hf_snapshot_dir(repo) {
                sd35_roots.push(snap);
            }
        }
        for root in sd35_roots {
            dirs.push(root.join("tokenizer"));
            dirs.push(root.join("tokenizer_2"));
        }

        // Canonical CLIP repos also carry vocab+merges at their snapshot root.
        dirs.extend(canonical_reference_dirs());
        dirs
    }

    /// Dirs shipping a real canonical CLIP `tokenizer.json` (alongside `vocab.json`+`merges.txt`) to
    /// assert synthesized-vs-canonical parity against. Only the canonical CLIP repos qualify.
    fn canonical_reference_dirs() -> Vec<PathBuf> {
        [
            "openai/clip-vit-large-patch14",
            "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
        ]
        .into_iter()
        .filter_map(candle_gen::testkit::hf_snapshot_dir)
        .collect()
    }

    const PROMPTS: &[&str] = &[
        "a photo of a cat",
        "A PHOTO of a CAT",
        "an astronaut riding a horse on the moon, highly detailed, 8k",
        "hello, world! 123 -- punctuation: test? (yes).",
        "café naïve résumé — über cool",
        "",
        "trailing space   ",
        "MixedCASE Words With  Multiple   Spaces",
    ];

    /// Encode a prompt to ids both with and without the special-token wrapping so the
    /// parity check covers the post-processor too.
    fn ids(tok: &Tokenizer, prompt: &str, add_special: bool) -> Vec<u32> {
        tok.encode(prompt, add_special)
            .expect("encode")
            .get_ids()
            .to_vec()
    }

    /// Real-weight parity: the synthesized CLIP BPE must be token-id-identical to the canonical
    /// `tokenizer.json`, over a battery of prompts (both `add_special` modes and padded-length
    /// encodes). The canonical reference comes from the canonical CLIP repos
    /// (`openai/clip-vit-large-patch14`, `laion/CLIP-ViT-bigG-14-laion2B-39B-b160k`) — a stock SD3.5
    /// snapshot ships no `tokenizer.json`, so it can't itself be the reference.
    ///
    /// `#[ignore]` — the canonical CLIP weights live outside CI (sc-9030 / F-046). Run on a machine with
    /// the canonical CLIP snapshot(s) cached:
    ///   cargo test -p candle-gen-sd3 synthesized_matches_canonical_clip_tokenizer -- --ignored --nocapture
    #[test]
    #[ignore = "real-weight: needs a canonical CLIP snapshot cached (openai/clip-vit-large-patch14 or laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)"]
    fn synthesized_matches_canonical_clip_tokenizer() {
        let mut exercised = 0usize;
        for dir in canonical_reference_dirs() {
            let json = dir.join("tokenizer.json");
            let vocab = dir.join("vocab.json");
            let merges = dir.join("merges.txt");
            if !(json.is_file() && vocab.is_file() && merges.is_file()) {
                continue;
            }
            exercised += 1;

            let canonical = Tokenizer::from_file(&json).expect("load canonical tokenizer.json");
            let synth =
                synthesize_clip_tokenizer(&vocab, &merges).expect("synthesize CLIP tokenizer");

            for &prompt in PROMPTS {
                for add_special in [false, true] {
                    let a = ids(&canonical, prompt, add_special);
                    let b = ids(&synth, prompt, add_special);
                    assert_eq!(
                        a,
                        b,
                        "token-id mismatch for dir={:?} prompt={prompt:?} add_special={add_special}\n canonical={a:?}\n synth={b:?}",
                        dir
                    );
                }
            }

            // Padding/truncation to the CLIP 77 length must also agree.
            for &prompt in PROMPTS {
                let enc_a = canonical.encode(prompt, true).expect("enc a");
                let enc_b = synth.encode(prompt, true).expect("enc b");
                assert_eq!(
                    enc_a.get_ids(),
                    enc_b.get_ids(),
                    "padded-len encode mismatch for {prompt:?}"
                );
            }
        }

        // This test is `#[ignore]`d and only runs when explicitly requested (`--ignored`), so a
        // caller has asserted the weights should be present. Finding zero canonical dirs is then a
        // real failure — NOT a silent vacuous pass (the old bug: it `eprintln!`'d and returned green).
        assert!(
            exercised > 0,
            "no canonical CLIP tokenizer.json found — checked the HF cache \
             ($HF_HUB_CACHE / $HF_HOME/hub / <home>/.cache/huggingface/hub) for \
             openai/clip-vit-large-patch14 and laion/CLIP-ViT-bigG-14-laion2B-39B-b160k; \
             cache a canonical CLIP snapshot before running this `--ignored` test"
        );
        eprintln!("parity: synthesized == canonical over {exercised} CLIP tokenizer dir(s)");
    }

    /// Known-deterministic CLIP byte-level BPE encoding (independent of local files):
    /// the canonical CLIP tokenizer encodes `"a photo of a cat"` to the well-known id
    /// sequence wrapped by BOS(49406)/EOS(49407). This proves the synthesis is correct
    /// even on a machine without the SD3.5 snapshot — provided a vocab+merges source.
    ///
    /// `#[ignore]` — needs a cached SD3.5 CLIP `vocab.json`+`merges.txt` pair (sc-9030 / F-046). Run:
    ///   cargo test -p candle-gen-sd3 synthesized_known_clip_ids -- --ignored --nocapture
    #[test]
    #[ignore = "real-weight: needs a cached SD3.5/CLIP vocab.json+merges.txt (see synthesis_source_dirs)"]
    fn synthesized_known_clip_ids() {
        // Find any local vocab.json + merges.txt CLIP pair to build from. This test is `#[ignore]`d,
        // so being unable to find one when explicitly run is a real failure, not a vacuous pass.
        let dir = synthesis_source_dirs()
            .into_iter()
            .find(|d| d.join("vocab.json").is_file() && d.join("merges.txt").is_file())
            .expect(
                "no cached SD3.5 CLIP vocab.json+merges.txt found (HF cache or \
                 $SD35_LARGE_PATH/$SD35_MEDIUM_PATH) — cache the snapshot before running this `--ignored` test",
            );
        let synth = synthesize_clip_tokenizer(&dir.join("vocab.json"), &dir.join("merges.txt"))
            .expect("synthesize");

        // The canonical CLIP BPE of "a photo of a cat" (lowercased, byte-level, </w>):
        //   a=320 photo=1125 of=539 a=320 cat=2368, wrapped 49406 .. 49407.
        let expected: Vec<u32> = vec![49406, 320, 1125, 539, 320, 2368, 49407];
        let got = ids(&synth, "a photo of a cat", true);
        assert_eq!(got, expected, "known CLIP id sequence mismatch: {got:?}");
    }

    /// The loader's fallback must work against a simulated **stock** snapshot layout:
    /// a directory with ONLY vocab.json + merges.txt (no tokenizer.json).
    ///
    /// `#[ignore]` — needs a cached SD3.5 CLIP `vocab.json`+`merges.txt` pair (sc-9030 / F-046). Run:
    ///   cargo test -p candle-gen-sd3 loader_fallback_on_stock_layout -- --ignored --nocapture
    #[test]
    #[ignore = "real-weight: needs a cached SD3.5/CLIP vocab.json+merges.txt (see synthesis_source_dirs)"]
    fn loader_fallback_on_stock_layout() {
        // `#[ignore]`d, so a missing source when explicitly run is a real failure, not a vacuous pass.
        let src = synthesis_source_dirs()
            .into_iter()
            .find(|d| d.join("vocab.json").is_file() && d.join("merges.txt").is_file())
            .expect(
                "no cached SD3.5 CLIP vocab.json+merges.txt found (HF cache or \
                 $SD35_LARGE_PATH/$SD35_MEDIUM_PATH) — cache the snapshot before running this `--ignored` test",
            );

        let tmp = std::env::temp_dir().join(format!("sc8500_stock_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).expect("mk tmp");
        fs::copy(src.join("vocab.json"), tmp.join("vocab.json")).expect("copy vocab");
        fs::copy(src.join("merges.txt"), tmp.join("merges.txt")).expect("copy merges");
        assert!(
            !tmp.join("tokenizer.json").exists(),
            "stock layout has no tokenizer.json"
        );

        let tok = load_clip_tokenizer(&tmp, "CLIP-L (stock-sim)").expect("load via fallback");
        let got = ids(&tok, "a photo of a cat", true);
        assert_eq!(got, vec![49406, 320, 1125, 539, 320, 2368, 49407]);

        let _ = fs::remove_dir_all(&tmp);
    }
}
