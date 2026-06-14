//! GPU/CPU smoke driver for the candle prompt-refine provider (sc-5500 acceptance).
//!
//! Loads a Llama-3.2-3B-Instruct snapshot and refines a sample prompt the way the worker will: the
//! prompt-rewrite rules + a model prompt-guide go in the `system` message (the caller's job — the
//! `TextLlm` contract is generic), the user's text in `prompt`. Prints the rewrite so a real run can
//! be eyeballed.
//!
//! Run: `set PROMPT_REFINE_MODEL_DIR=<snapshot dir>; cargo run -p candle-gen-prompt-refine --example refine --features cuda`

use candle_gen::gen_core::{
    registry, LoadSpec, Progress, TextLlmRequest, TextLlmSampling, WeightsSource,
};
use candle_gen_prompt_refine::prompt::PROMPT_REFINE_ID;

// A representative caller-built system prompt — the prompt-refine rules (image medium) + a short
// model prompt-guide — mirroring what the worker's `build_system_prompt` produces.
const SYSTEM: &str = "You are a prompt rewriter for a generative image model.\n\
Rewrite the user's input into a single, precise image prompt that follows the model's prompt guide below.\n\n\
Rules:\n\
- Output exactly one rewritten prompt and nothing else — no explanations, reasoning, commentary, or labels.\n\
- Preserve the user's intent: do not change the subjects, attributes, actions, or setting they described.\n\
- Follow the guide's recommended structure and phrasing.\n\n\
# Model prompt guide\n\n\
Prefer a single flowing sentence. Lead with the main subject, then key attributes, then the setting \
and lighting. Add concrete, photographic detail (lens, time of day) when it stays consistent with the user's meaning.";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::var("PROMPT_REFINE_MODEL_DIR")
        .map_err(|_| "set PROMPT_REFINE_MODEL_DIR to a Llama-3.2-3B-Instruct snapshot directory")?;
    let user = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "a cat sitting on a windowsill".to_owned());

    let spec = LoadSpec::new(WeightsSource::Dir(dir.into()));
    let refiner = registry::load_textllm(PROMPT_REFINE_ID, &spec)?;

    let req = TextLlmRequest {
        system: SYSTEM.to_owned(),
        prompt: user.clone(),
        sampling: TextLlmSampling {
            seed: Some(42),
            ..Default::default()
        },
        ..Default::default()
    };

    let mut steps = 0u32;
    let out = refiner.generate(&req, &mut |p| {
        if let Progress::Step { current, .. } = p {
            steps = current;
        }
    })?;

    println!("\n=== input ===\n{user}");
    println!(
        "\n=== refined ({} tokens, {} progress steps, finish={:?}) ===\n{}\n",
        out.generated_tokens.unwrap_or(0),
        steps,
        out.finish_reason,
        out.text
    );
    Ok(())
}
