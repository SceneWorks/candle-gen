//! SANA text-conditioning reuse tests (sc-11779, the candle sibling of mlx-gen-sana sc-8488).
//!
//! SANA's text conditioning REUSES PiD's gemma-2-2b-it CHI caption encoder
//! ([`candle_gen_pid::CaptionEncoder`]); the only divergence is the CHI prompt text. These tests pin:
//!
//!  * [`sana_chi_prompt_is_pid_chi_with_single_quotes`] — DEFAULT (no weights). The hard contract on
//!    the CHI template: SANA's `SANA_CHI_PROMPT` equals PiD's `CHI_PROMPT` in every character except
//!    the quoting around `Enhanced prompt` (single- vs double-quote), and equals the joined diffusers
//!    `complex_human_instruction` list. A wrong CHI string is a discrete, O(1) failure that silently
//!    wrecks conditioning, so it is asserted exactly.
//!
//!  * [`sana_selection_index_matches_reference`] — DEFAULT (no weights). The 300-token selection math
//!    (`[0] + range(max_len − 299, max_len)`) via the shared
//!    [`candle_gen_pid::caption::select_index`] the encoder actually calls.
//!
//!  * [`caption_encoding_matches_reference`] — `#[ignore]`d (needs the un-gated
//!    `SceneWorks/gemma-2-2b-it` weights; the numeric golden vs diffusers is additionally gated on
//!    `SANA_CAPTION_GOLDEN`). Asserts the `[1, 300, 2304]` shape + the length policy, and — when the
//!    golden safetensors is present — exact token-id match (tokenizer + CHI-prompt + length-policy
//!    correctness) plus a cosine gate on the caption embedding vs the SANA reference.
//!
//! ```sh
//! cargo test -p candle-gen-sana --test text_encoder            # default (CHI + selection contract)
//! PID_GEMMA_DIR=/path/to/gemma-2-2b-it \
//!   cargo test -p candle-gen-sana --release --test text_encoder -- --ignored --nocapture
//! ```

use candle_gen_sana::{SanaTextEncoder, MAX_SEQUENCE_LENGTH, SANA_CHI_PROMPT};

const CAPTION: &str =
    "a mountain valley landscape at golden hour with a winding river and pine forest";

/// SANA's exact `complex_human_instruction` list (diffusers `pipeline_sana.py` / NVlabs Sana),
/// joined by `"\n"` — the reference definition `SANA_CHI_PROMPT` must equal.
fn sana_chi_joined() -> String {
    [
        "Given a user prompt, generate an 'Enhanced prompt' that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:",
        "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.",
        "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.",
        "Here are examples of how to transform or refine prompts:",
        "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.",
        "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.",
        "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:",
        "User Prompt: ",
    ]
    .join("\n")
}

#[test]
fn sana_chi_prompt_is_pid_chi_with_single_quotes() {
    // 1. SANA_CHI_PROMPT is exactly the joined diffusers complex_human_instruction list.
    assert_eq!(
        SANA_CHI_PROMPT,
        sana_chi_joined(),
        "SANA_CHI_PROMPT must equal `\"\\n\".join(complex_human_instruction)`"
    );

    // 2. It differs from PiD's CHI in EXACTLY the quoting around `Enhanced prompt` — the load-bearing
    //    divergence that forced parameterizing rather than reusing PiD's text. Both contain the
    //    single-quote form (resp. double-quote), are the same length, and are otherwise identical.
    let pid = candle_gen_pid::caption::CHI_PROMPT;
    assert_eq!(
        SANA_CHI_PROMPT.len(),
        pid.len(),
        "same length — divergence is only the quote glyph, not the wording"
    );
    assert!(SANA_CHI_PROMPT.contains("an 'Enhanced prompt'"));
    assert!(pid.contains("an \"Enhanced prompt\""));
    assert_ne!(
        SANA_CHI_PROMPT, pid,
        "SANA and PiD CHI prompts must differ (quote style) — do not reuse PiD's text"
    );
    // Replacing SANA's single-quotes with PiD's double-quotes recovers PiD's string exactly: proves
    // the quote glyph is the SOLE difference.
    assert_eq!(
        SANA_CHI_PROMPT.replacen("an 'Enhanced prompt'", "an \"Enhanced prompt\"", 1),
        pid,
        "the only difference between the SANA and PiD CHI prompts is the Enhanced-prompt quoting"
    );

    // Both end with the trailing "User Prompt: " the caption is appended after.
    assert!(SANA_CHI_PROMPT.ends_with("User Prompt: "));
}

#[test]
fn sana_selection_index_matches_reference() {
    // SANA (diffusers `_get_gemma_prompt_embeds`): select_index = [0] + range(-(300 - 1), 0), i.e.
    // the <bos> plus the trailing 299 tokens of the `max_len`-long last-hidden sequence → 300 tokens.
    assert_eq!(MAX_SEQUENCE_LENGTH, 300, "SANA max_sequence_length");

    let max_len = 555usize; // representative num_chi_tokens + 300 - 2
    let sel = candle_gen_pid::caption::select_index(max_len);

    assert_eq!(
        sel.len(),
        MAX_SEQUENCE_LENGTH as usize,
        "exactly 300 selected tokens"
    );
    assert_eq!(sel[0], 0, "position 0 (<bos>) is preserved");
    assert_eq!(
        *sel.last().unwrap(),
        max_len as u32 - 1,
        "selection ends at the final token"
    );

    // Byte-identical to the explicit reference `[0] + list(range(max_len - 299, max_len))`.
    let mut expect = vec![0u32];
    expect.extend((max_len as u32 - (MAX_SEQUENCE_LENGTH as u32 - 1))..max_len as u32);
    assert_eq!(sel, expect, "selection index must match the SANA reference");
}

/// Resolve the gemma-2-2b-it snapshot dir: explicit `PID_GEMMA_DIR`, else the un-gated
/// `SceneWorks/gemma-2-2b-it` HF-cache mirror.
fn gemma_snapshot() -> std::path::PathBuf {
    if let Ok(dir) = std::env::var("PID_GEMMA_DIR") {
        return std::path::PathBuf::from(dir);
    }
    candle_gen::testkit::require_hf_snapshot_dir("SceneWorks/gemma-2-2b-it")
}

#[test]
#[ignore = "needs SceneWorks/gemma-2-2b-it weights; numeric parity additionally needs SANA_CAPTION_GOLDEN"]
fn caption_encoding_matches_reference() {
    use candle_gen::candle_core::{DType, Device};

    let device = Device::Cpu;
    let snap = gemma_snapshot();
    let enc = SanaTextEncoder::from_snapshot(&snap, &device).unwrap();

    // --- length policy: padded ids are num_chi_tokens + 300 - 2 long ---
    let (ids, mask) = enc.token_ids(CAPTION).unwrap();
    let expected_len = (enc.num_chi_tokens() + MAX_SEQUENCE_LENGTH - 2) as usize;
    assert_eq!(ids.len(), expected_len, "padded length (num_chi + 298)");
    assert_eq!(mask.len(), expected_len, "mask length matches ids");
    eprintln!(
        "token ids len={} num_chi_tokens={}",
        ids.len(),
        enc.num_chi_tokens()
    );

    // --- shape: [1, 300, 2304] (gemma last-hidden, select_index-gathered) ---
    let embs = enc.encode(CAPTION).unwrap();
    assert_eq!(
        embs.dims(),
        &[1, MAX_SEQUENCE_LENGTH as usize, 2304],
        "caption_embs shape [1, 300, 2304]"
    );

    // --- numeric parity vs the SANA reference (optional golden) ---
    let Some(golden) = std::env::var_os("SANA_CAPTION_GOLDEN") else {
        eprintln!("SANA_CAPTION_GOLDEN unset — shape + length asserted; skipping numeric parity");
        return;
    };
    let golden = candle_gen::Weights::from_file(std::path::Path::new(&golden), &device, DType::F32)
        .expect("load SANA caption golden");

    // gate 1: exact token-id match (the hard correctness proof for tokenizer + CHI + length).
    let ref_ids = golden.require("input_ids").expect("golden input_ids");
    let ref_ids: Vec<i32> = ref_ids
        .to_dtype(DType::I64)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<i64>()
        .unwrap()
        .into_iter()
        .map(|v| v as i32)
        .collect();
    assert_eq!(
        ids, ref_ids,
        "token ids must match the SANA reference exactly"
    );

    // gate 2: caption_embs cosine vs golden.
    let ref_embs = golden.require("caption_embs").expect("golden caption_embs");
    assert_eq!(
        ref_embs.dims(),
        &[1, 300, 2304],
        "golden caption_embs shape"
    );
    let a = embs.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let b = ref_embs
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let cos = candle_gen::testkit::cosine(&a, &b);
    eprintln!("caption_embs cosine={cos}");
    assert!(
        cos > 0.998,
        "caption_embs cosine={cos} — forward divergence (ids matched exactly)"
    );
}
