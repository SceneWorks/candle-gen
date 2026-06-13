//! sc-5118 — Lens **PromptReasoner** (local gpt-oss generate) parity vs torch `generate`.
//!
//! Three gates, all against the golden from `scripts/dump_lens_reasoner_golden.py` (torch greedy
//! `generate(do_sample=False)`):
//!  1. **Template byte-check** — the candle harmony reasoner render + tokenize ([`encode_reasoner`])
//!     reproduces the golden's `input_ids` exactly.
//!  2. **Greedy parity** — the candle KV-cache greedy decode matches torch's leading greedy tokens
//!     (first token hard-gated; the matching prefix is reported — cross-build bf16 candle-vs-torch
//!     argmax can diverge on a late near-tie, exactly as the encoder e2e, so the gate is prefix-based).
//!  3. **Cache equivalence** (no torch) — the incremental cached decode tracks a teacher-forced full
//!     recompute of the same tokens (`next_token_argmax`), proving the KV cache + sliding-window
//!     eviction are correct.
//!
//! Heavy + machine-specific (loads the full ~40 GB bf16 gpt-oss + needs the GPU), so it is **gated**:
//!   LENS_SNAPSHOT_DIR     — the `microsoft/Lens-Turbo` snapshot root (tokenizer/ + text_encoder/)
//!   LENS_REASONER_GOLDENS — lens_reasoner_golden.safetensors (default: .scratch/lens-reasoner-goldens/…)
//! Run with the `cuda` feature:
//!   cargo test -p candle-gen-lens --features cuda --test reasoner_parity -- --nocapture

use candle_gen::candle_core::{DType, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen_lens::text::LensTokenizer;
use candle_gen_lens::text_encoder::{Config, LensReasonerModel};

type AnyErr = Box<dyn std::error::Error>;

fn utf8(t: &Tensor) -> Result<String, AnyErr> {
    Ok(String::from_utf8(t.to_dtype(DType::U8)?.to_vec1::<u8>()?)?)
}

fn ids(t: &Tensor) -> Result<Vec<u32>, AnyErr> {
    Ok(t.to_dtype(DType::U32)?.flatten_all()?.to_vec1::<u32>()?)
}

#[test]
fn lens_reasoner_matches_reference() -> Result<(), AnyErr> {
    let root = match std::env::var("LENS_SNAPSHOT_DIR") {
        Ok(d) => d,
        Err(_) => {
            eprintln!("SKIP: set LENS_SNAPSHOT_DIR to the Lens-Turbo snapshot root");
            return Ok(());
        }
    };
    let goldens_path = std::env::var("LENS_REASONER_GOLDENS").unwrap_or_else(|_| {
        ".scratch/lens-reasoner-goldens/lens_reasoner_golden.safetensors".to_string()
    });
    if !std::path::Path::new(&goldens_path).exists() {
        eprintln!(
            "SKIP: goldens not found at {goldens_path} (run scripts/dump_lens_reasoner_golden.py)"
        );
        return Ok(());
    }

    let device = candle_gen::default_device()?;
    eprintln!("device: {device:?}");
    let g = candle_gen::candle_core::safetensors::load(&goldens_path, &device)?;
    let prompt = utf8(&g["prompt_utf8"])?;
    let date = utf8(&g["date_utf8"])?;
    let input_ids = ids(&g["input_ids"])?;
    let want_new = ids(&g["new_tokens"])?;
    let max_new = want_new.len();
    eprintln!(
        "prompt={prompt:?}  date={date}  L={}  max_new={max_new}",
        input_ids.len()
    );

    // 1. Template byte-check.
    let root_path = std::path::Path::new(&root);
    let tok = LensTokenizer::from_file(root_path.join("tokenizer/tokenizer.json"))?;
    let got_ids = tok.encode_reasoner(&prompt, &date)?;
    assert_eq!(
        got_ids,
        input_ids,
        "reasoner template ids differ from the golden (len {} vs {})",
        got_ids.len(),
        input_ids.len()
    );
    eprintln!("template: {} ids byte-exact", got_ids.len());

    // Load the generating model (bf16; experts MXFP4 → bf16 inside the module).
    eprintln!("loading reasoner model (MXFP4→bf16 + lm_head)…");
    let te_dir = root_path.join("text_encoder");
    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(&te_dir)
        .expect("read text_encoder dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no .safetensors in text_encoder/");
    // SAFETY: mmap of read-only weight files (the standard candle loading path).
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, DType::BF16, &device)? };
    let model = LensReasonerModel::new(&Config::gpt_oss_20b(), vb, None)?;

    // 2. Greedy parity — the prefill + lm_head + argmax (first token) and the cached decode must
    //    reproduce torch's greedy tokens. Only the first token is hard-gated: cross-build bf16 makes the
    //    per-step logits diverge enough to flip argmax on a late near-tie (the same effect as the
    //    encoder e2e's 0.997 cosine); the deterministic correctness proof is gate #3.
    let got_new = model.generate_greedy(&input_ids, max_new)?;
    let match_len = got_new
        .iter()
        .zip(&want_new)
        .take_while(|(a, b)| a == b)
        .count();
    eprintln!(
        "greedy: matched {match_len}/{} leading tokens\n  torch {:?}\n  rust  {:?}",
        want_new.len(),
        &want_new[..want_new.len().min(8)],
        &got_new[..got_new.len().min(8)],
    );
    assert_eq!(
        got_new.first(),
        want_new.first(),
        "first greedy token differs from torch — prefill/lm_head/argmax bug"
    );

    // 3. Cache equivalence (no torch): the cached free-run decode should track a teacher-forced full
    //    recompute over the same tokens, proving the KV cache + the sliding-window eviction. bf16 picks
    //    different matmul kernels for the [1,d] decode step vs the batched [L,d] recompute, so a rare
    //    near-tie can flip — gate on a HIGH agreement fraction and report it.
    let mut forced = input_ids.clone();
    forced.extend_from_slice(&got_new[..got_new.len() - 1]);
    let pred = model.next_token_argmax(&forced)?;
    let l = input_ids.len();
    let recomputed = &pred[l - 1..]; // predictions at positions L-1, L, … → the generated tokens
    let agree = recomputed
        .iter()
        .zip(&got_new)
        .filter(|(a, b)| a == b)
        .count();
    eprintln!(
        "cache equivalence: {agree}/{} agree with the full recompute\n  recompute {:?}\n  cached    {:?}",
        got_new.len(),
        &recomputed[..recomputed.len().min(12)],
        &got_new[..got_new.len().min(12)],
    );
    assert!(
        agree * 10 >= got_new.len() * 9,
        "cached decode agrees with the recompute on only {agree}/{} — a KV-cache bug, not bf16 drift",
        got_new.len()
    );
    eprintln!("ALL PASS");
    Ok(())
}
