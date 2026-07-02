//! The autoregressive text-generation runtime — the candle port of `mlx-gen-sensenova`'s
//! `runtime.rs` (the slice needed by the understanding surface: VQA + interleave).
//!
//! SenseNova-U1 is a *generating* LLM, so beyond a forward pass the understanding path needs an AR
//! runtime: a prefix is prefilled into a [`KvCache`](crate::qwen3::KvCache), then tokens are decoded
//! one at a time — each new token forwarded through the cached backbone at the next temporal
//! position to produce the logits for the token after it. This module ports the reference's pieces
//! (`modeling_neo_chat.py`):
//!
//! * [`Qwen3Backbone::decode_logits`] — one cached single-token forward → next-token logits.
//! * [`Qwen3Backbone::generate`] — greedy/sampled rollout to an EOS or token budget (the runtime
//!   under `chat` / `answer_question`).
//!
//! Positions: text tokens advance the temporal axis by one per token (`h = w = 0`), matching the
//! reference. The understanding path ([`Path::Und`]) drives text decode. The interleave rollout
//! drives its own loop directly over [`decode_logits`](Qwen3Backbone::decode_logits) (it alternates
//! text decode and gen-path image generation), so only these two primitives live here.

use candle_gen::candle_core::Result as CResult;
use candle_gen::gen_core::CancelFlag;
use candle_gen::{CandleError, Result};

use crate::qwen3::{KvCache, Path, Qwen3Backbone};

/// How the next token is chosen from a logits row.
#[derive(Clone, Copy, Debug)]
pub enum Sampler {
    /// Argmax — the reference deterministic chat path.
    Greedy,
    /// Temperature + nucleus (top-p) + top-k sampling. `top_p`/`top_k` of `1.0`/`0` disable that
    /// stage; `temperature` must be `> 0`.
    Sample {
        temperature: f32,
        top_p: f32,
        top_k: usize,
        seed: u64,
    },
}

impl Sampler {
    /// Pick a token id from a `[vocab]` logits row, advancing `rng` for the stochastic variants.
    fn pick(&self, logits: &[f32], rng: &mut SplitMix64) -> i32 {
        match *self {
            Sampler::Greedy => argmax(logits),
            Sampler::Sample {
                temperature,
                top_p,
                top_k,
                ..
            } => sample(logits, temperature, top_p, top_k, rng),
        }
    }

    fn seed(&self) -> u64 {
        match *self {
            Sampler::Greedy => 0,
            Sampler::Sample { seed, .. } => seed,
        }
    }
}

impl Qwen3Backbone {
    /// One cached single-token forward on the understanding path: embed `token`, run it at temporal
    /// position `pos_t` (`h = w = 0`), persist its K/V, and return the `[vocab]` next-token logits.
    pub fn decode_logits(&self, token: i32, pos_t: i32, cache: &mut KvCache) -> CResult<Vec<f32>> {
        let embeds = self.embed(&[token])?; // [1, 1, hidden]
        let hidden = self.forward_cached(&embeds, &[pos_t], &[0], &[0], Path::Und, cache, true)?;
        let logits = self.lm_head(&hidden)?; // [1, 1, vocab]
        let vocab = logits.dim(2)?;
        logits.reshape((vocab,))?.to_vec1::<f32>()
    }

    /// Greedy/sampled AR text rollout. `first_logits` are the prefix's last-position logits (the
    /// distribution over the first generated token); `t_idx` is the prefix's max temporal index.
    /// Decoding stops at any id in `eos` (not emitted) or after `max_new_tokens`. Returns the
    /// generated token ids.
    ///
    /// `cancel` is the cooperative cancellation handle (sc-9123, the candle sibling of mlx-gen's
    /// F-037/sc-9093 change): checked before each decoded token so a worker-consumed multi-minute
    /// VQA / understanding rollout is cancellable per token, not only at its natural end. Returns
    /// the typed [`CandleError::Canceled`] on trip so the worker can key off it.
    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        &self,
        first_logits: &[f32],
        cache: &mut KvCache,
        t_idx: i32,
        eos: &[i32],
        max_new_tokens: usize,
        sampler: Sampler,
        cancel: Option<&CancelFlag>,
    ) -> Result<Vec<i32>> {
        let mut rng = SplitMix64::new(sampler.seed());
        let mut logits = first_logits.to_vec();
        let mut out = Vec::new();
        let mut t = t_idx;
        for _ in 0..max_new_tokens {
            if cancel.is_some_and(CancelFlag::is_cancelled) {
                return Err(CandleError::Canceled);
            }
            let next = sampler.pick(&logits, &mut rng);
            if eos.contains(&next) {
                break;
            }
            out.push(next);
            t += 1;
            logits = self.decode_logits(next, t, cache)?;
        }
        Ok(out)
    }
}

/// Index of the maximum logit (ties → lowest index, matching `torch.argmax`).
pub(crate) fn argmax(logits: &[f32]) -> i32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as i32
}

/// Temperature + top-k + nucleus (top-p) sampling over a logits row.
fn sample(logits: &[f32], temperature: f32, top_p: f32, top_k: usize, rng: &mut SplitMix64) -> i32 {
    let temperature = temperature.max(1e-6);
    let mut order: Vec<usize> = (0..logits.len()).collect();
    // Total order: descending logit, ties broken by ascending index.
    let by_logit_then_index = |&a: &usize, &b: &usize| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    };

    let k = if top_k == 0 {
        order.len()
    } else {
        top_k.min(order.len())
    };
    if k < order.len() {
        order.select_nth_unstable_by(k - 1, by_logit_then_index);
        order.truncate(k);
    }
    order.sort_unstable_by(by_logit_then_index);

    // Softmax (in the truncated set) at the given temperature, numerically stabilised.
    let max_logit = logits[order[0]];
    let mut probs: Vec<f32> = order
        .iter()
        .map(|&i| ((logits[i] - max_logit) / temperature).exp())
        .collect();
    let sum: f32 = probs.iter().sum();
    for p in &mut probs {
        *p /= sum;
    }

    // top-p (nucleus): keep the smallest prefix whose cumulative prob ≥ top_p.
    if top_p < 1.0 {
        let mut cum = 0.0f32;
        let mut cutoff = probs.len();
        for (i, &p) in probs.iter().enumerate() {
            cum += p;
            if cum >= top_p {
                cutoff = i + 1;
                break;
            }
        }
        order.truncate(cutoff);
        probs.truncate(cutoff);
        let renorm: f32 = probs.iter().sum();
        for p in &mut probs {
            *p /= renorm;
        }
    }

    // Inverse-CDF sample.
    let r = rng.next_f32();
    let mut cum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cum += p;
        if r <= cum {
            return order[i] as i32;
        }
    }
    order[order.len() - 1] as i32
}

/// SplitMix64 increment (the golden-ratio odd constant).
pub(crate) const SPLITMIX64_INCREMENT: u64 = 0x9E37_79B9_7F4A_7C15;

/// SplitMix64 — a tiny deterministic PRNG for reproducible sampling.
pub(crate) struct SplitMix64(u64);

impl SplitMix64 {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(SPLITMIX64_INCREMENT);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u64 << 24) as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_breaks_ties_to_lowest_index() {
        assert_eq!(argmax(&[0.1, 0.5, 0.5, 0.2]), 1);
        assert_eq!(argmax(&[3.0, 1.0, 2.0]), 0);
    }

    #[test]
    fn top_k_one_is_argmax() {
        let logits = [0.1, 2.0, 0.5, 1.0];
        let mut rng = SplitMix64::new(123);
        for _ in 0..16 {
            assert_eq!(sample(&logits, 1.0, 1.0, 1, &mut rng), 1);
        }
    }

    #[test]
    fn sampling_is_seed_deterministic() {
        let logits = [0.2, 1.5, 0.3, 0.9, 0.1];
        let s = Sampler::Sample {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            seed: 42,
        };
        let run = || {
            let mut rng = SplitMix64::new(s.seed());
            (0..8)
                .map(|_| s.pick(&logits, &mut rng))
                .collect::<Vec<_>>()
        };
        assert_eq!(run(), run(), "same seed → identical token sequence");
    }
}
