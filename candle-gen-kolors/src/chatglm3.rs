//! ChatGLM3-6B forward — Kolors' text encoder. The candle port of `mlx-gen-kolors`'s `chatglm3.rs`,
//! a faithful reproduction of the diffusers `KolorsPipeline` reference (`ChatGLMModel`, encoder-only:
//! no LM head, no generation, no KV cache), driven by `encode_prompt` with `output_hidden_states`.
//!
//! ChatGLM3-specific pieces:
//!  - **Half-dim interleaved RoPE.** Rotary applies to the **first `rotary_dim` (64)** of the 128-wide
//!    head dim, with **adjacent-pair** interleaving `(x[2i], x[2i+1])` (NOT the HF half-split); the
//!    trailing 64 dims pass through unrotated. θ = 10000, constant across layers.
//!  - **Fused, biased `query_key_value`.** One `[4608, 4096]` Linear (with bias) →
//!    q (32·128) + k (2·128) + v (2·128); **multi-query attention, 2 KV groups** broadcast to 32 query
//!    heads. The output `dense` proj is bias-less.
//!  - **RMSNorm = plain `weight · x̂`** (eps 1e-5) — candle's `RmsNorm` (NOT Gemma's `(1 + weight)`).
//!  - **GLMBlock pre-norm residual**: `h = x + dense(attn(input_ln(x)))`;
//!    `out = h + mlp(post_attn_ln(h))`. MLP = `dense_4h_to_h(silu(g) · u)` where `dense_h_to_4h` fuses
//!    gate+up (out `2·13696`); the activation is SiLU.
//!  - **Standard `1/√head_dim` scaled-dot-product** + plain softmax (the legacy
//!    `apply_query_key_layer_scaling` does not run on the SDPA path the reference takes).
//!
//! ### Output contract (what Kolors consumes)
//! Kolors uses **`hidden_states[-2]`** (penultimate, layer-26 output) as the cross-attention context and
//! **`hidden_states[-1]` at the last sequence position** as the pooled add-embedding — neither is
//! final-normed, so the encoder's `final_layernorm` weight is never consumed and is not loaded.

use candle_gen::candle_core::Result as CandleResult;
use candle_gen::candle_core::{Device, Tensor, D};
use candle_gen::candle_nn::{self as nn, Module, RmsNorm, VarBuilder};

use crate::config::ChatGlmConfig;
use crate::tokenizer::KolorsTokens;

struct GlmBlock {
    input_ln: RmsNorm,
    post_attn_ln: RmsNorm,
    qkv: nn::Linear,     // fused query_key_value, biased
    dense: nn::Linear,   // output projection, bias-less
    h_to_4h: nn::Linear, // fused gate+up, bias-less
    h4_to_h: nn::Linear, // bias-less
}

/// The ChatGLM3-6B backbone used as the Kolors text encoder.
pub struct ChatGlmModel {
    embed: nn::Embedding,
    layers: Vec<GlmBlock>,
    cfg: ChatGlmConfig,
    device: Device,
}

impl ChatGlmModel {
    /// Build from the Kolors `text_encoder/` VarBuilder (the `embedding.word_embeddings.*` /
    /// `encoder.layers.{i}.*` diffusers layout). `encoder.final_layernorm.*` is present in the snapshot
    /// but unused by Kolors conditioning, so it is not loaded.
    pub fn new(cfg: ChatGlmConfig, vb: VarBuilder) -> CandleResult<Self> {
        let device = vb.device().clone();
        let embed = nn::embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            vb.pp("embedding.word_embeddings"),
        )?;
        let attn_out = cfg.num_heads * cfg.head_dim; // 4096
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let b = vb.pp(format!("encoder.layers.{i}"));
            layers.push(GlmBlock {
                input_ln: nn::rms_norm(cfg.hidden_size, cfg.rms_eps, b.pp("input_layernorm"))?,
                post_attn_ln: nn::rms_norm(
                    cfg.hidden_size,
                    cfg.rms_eps,
                    b.pp("post_attention_layernorm"),
                )?,
                qkv: nn::linear(
                    cfg.hidden_size,
                    cfg.qkv_out(),
                    b.pp("self_attention.query_key_value"),
                )?,
                dense: nn::linear_no_bias(attn_out, cfg.hidden_size, b.pp("self_attention.dense"))?,
                h_to_4h: nn::linear_no_bias(
                    cfg.hidden_size,
                    2 * cfg.ffn_hidden,
                    b.pp("mlp.dense_h_to_4h"),
                )?,
                h4_to_h: nn::linear_no_bias(
                    cfg.ffn_hidden,
                    cfg.hidden_size,
                    b.pp("mlp.dense_4h_to_h"),
                )?,
            });
        }
        Ok(Self {
            embed,
            layers,
            cfg,
            device,
        })
    }

    /// Extract Kolors conditioning for one prompt: `(context [1, S, hidden], pooled [1, hidden])`.
    /// `context` = the penultimate hidden state (layer-26 output); `pooled` = the final hidden state at
    /// the **last sequence position**. Threads the tokenizer's left-padded `position_ids` into RoPE.
    /// Batch size 1 (production encode is always B==1; Kolors CFG-batches the two prompts' results, not
    /// the encode itself).
    pub fn encode_prompt(&self, tokens: &KolorsTokens) -> CandleResult<(Tensor, Tensor)> {
        let s = tokens.input_ids.len();
        let hidden = self.cfg.hidden_size;
        let ids = Tensor::from_vec(tokens.input_ids.clone(), (1, s), &self.device)?;
        let mut h = self.embed.forward(&ids)?; // [1, s, hidden]
        let mask = self.causal_padding_mask(&tokens.attention_mask, s)?;
        let (cos, sin) = self.rope_tables(&tokens.position_ids)?;

        // context = output of layer `num_layers - 2` (hidden_states[-2]); pooled source = final h.
        let context_idx = self.cfg.num_layers - 2;
        let mut context: Option<Tensor> = None;
        for (i, layer) in self.layers.iter().enumerate() {
            h = self.block(layer, &h, &mask, &cos, &sin)?;
            if i == context_idx {
                context = Some(h.clone());
            }
        }
        let context = context.expect("chatglm3 has >= 2 layers");
        // pooled = h[:, s-1, :] (the last sequence position of the final hidden state).
        let pooled = h.narrow(1, s - 1, 1)?.squeeze(1)?.reshape((1, hidden))?;
        Ok((context, pooled))
    }

    /// One GLM block: pre-norm residual attention then pre-norm residual SwiGLU MLP.
    fn block(
        &self,
        layer: &GlmBlock,
        x: &Tensor,
        mask: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> CandleResult<Tensor> {
        let r = self.attn(layer, &layer.input_ln.forward(x)?, mask, cos, sin)?;
        let h = (x + r)?;
        let r = self.mlp(layer, &layer.post_attn_ln.forward(&h)?)?;
        &h + r
    }

    /// Fused-QKV GQA attention with GLM half-dim interleaved RoPE and an additive causal+padding mask.
    fn attn(
        &self,
        layer: &GlmBlock,
        x: &Tensor,
        mask: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> CandleResult<Tensor> {
        let (b, s, _) = x.dims3()?;
        let (nh, kv, d) = (
            self.cfg.num_heads,
            self.cfg.num_kv_groups,
            self.cfg.head_dim,
        );
        // Fused QKV → [b, s, (nh + 2·kv), d]; head-major: heads 0..nh = q, nh..nh+kv = k, … = v.
        let qkv = layer.qkv.forward(x)?.reshape((b, s, nh + 2 * kv, d))?;
        let q = qkv.narrow(2, 0, nh)?;
        let k = qkv.narrow(2, nh, kv)?;
        let v = qkv.narrow(2, nh + kv, kv)?;

        let q = self.apply_rope(&q, cos, sin)?;
        let k = self.apply_rope(&k, cos, sin)?;

        // [b, s, heads, d] → [b, heads, s, d].
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;
        // GQA: broadcast each KV head to `nh/kv` contiguous query heads.
        let g = nh / kv;
        let k = repeat_kv(&k, g)?;
        let v = repeat_kv(&v, g)?;

        let scale = 1.0 / (d as f64).sqrt();
        let scores = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?; // [b, nh, s, s]
        let scores = scores.broadcast_add(mask)?; // mask [b, 1, s, s] broadcasts over heads
        let probs = nn::ops::softmax_last_dim(&scores)?;
        let out = probs.matmul(&v)?; // [b, nh, s, d]
        let out = out.transpose(1, 2)?.contiguous()?.reshape((b, s, nh * d))?;
        layer.dense.forward(&out)
    }

    /// SwiGLU MLP: `dense_h_to_4h` fuses gate+up (out `2·ffn`); `silu(gate)·up → dense_4h_to_h`.
    fn mlp(&self, layer: &GlmBlock, x: &Tensor) -> CandleResult<Tensor> {
        let ffn = self.cfg.ffn_hidden;
        let gu = layer.h_to_4h.forward(x)?;
        let gate = gu.narrow(D::Minus1, 0, ffn)?;
        let up = gu.narrow(D::Minus1, ffn, ffn)?;
        let gated = (nn::ops::silu(&gate)? * up)?;
        layer.h4_to_h.forward(&gated)
    }

    /// Rotary `(cos, sin)`, each `[1, seq, 1, rotary_dim/2]`, for the given absolute `positions` (one
    /// per sequence slot — Kolors' left-padded `position_ids`). θ = 10000, computed once.
    fn rope_tables(&self, positions: &[i64]) -> CandleResult<(Tensor, Tensor)> {
        let half = self.cfg.rotary_dim / 2; // 32
        let rot = self.cfg.rotary_dim as f64; // 64
        let inv_freq: Vec<f32> = (0..half)
            .map(|j| (1.0 / self.cfg.rope_base.powf((2 * j) as f64 / rot)) as f32)
            .collect();
        let seq = positions.len();
        let mut freqs = Vec::with_capacity(seq * half);
        for &p in positions {
            for &f in &inv_freq {
                freqs.push(p as f32 * f);
            }
        }
        let freqs = Tensor::from_vec(freqs, (1, seq, 1, half), &self.device)?;
        Ok((freqs.cos()?, freqs.sin()?))
    }

    /// GLM interleaved half-dim RoPE on `x` `[b, s, heads, d]`: rotate the first `rotary_dim` dims as
    /// adjacent pairs `(x[2i], x[2i+1])` against `(cos, sin)` `[1, s, 1, rotary_dim/2]`; pass the
    /// trailing `d - rotary_dim` dims through.
    fn apply_rope(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> CandleResult<Tensor> {
        let (b, s, h, d) = x.dims4()?;
        let rot = self.cfg.rotary_dim;
        let half = rot / 2;
        let x_rot = x.narrow(3, 0, rot)?; // [b, s, h, rot]
        let x_pass = x.narrow(3, rot, d - rot)?; // [b, s, h, d-rot]
        let xr = x_rot.reshape((b, s, h, half, 2))?;
        let x0 = xr.narrow(4, 0, 1)?.squeeze(4)?; // even lane [b, s, h, half]
        let x1 = xr.narrow(4, 1, 1)?.squeeze(4)?; // odd lane
        let out0 = (x0.broadcast_mul(cos)? - x1.broadcast_mul(sin)?)?;
        let out1 = (x1.broadcast_mul(cos)? + x0.broadcast_mul(sin)?)?;
        // Re-interleave: stack on a new trailing axis then fold back to `rot`.
        let rotated = Tensor::stack(&[&out0, &out1], 4)?.reshape((b, s, h, rot))?;
        Tensor::cat(&[&rotated, &x_pass], 3)
    }

    /// Additive `[1, 1, s, s]` mask in f32 (B==1), mirroring the reference `get_masks`: a real query row
    /// `i` attends key `j` iff causal (`j ≤ i`) and the key is not padding; a padding query row attends
    /// everything (so its hidden state is deterministic). Disallowed → a large finite negative.
    fn causal_padding_mask(&self, attention_mask: &[u32], s: usize) -> CandleResult<Tensor> {
        let data = causal_padding_data(attention_mask, 1, s);
        Tensor::from_vec(data, (1, 1, s, s), &self.device)
    }
}

/// Broadcast each of `x`'s `kv` heads to `g` contiguous query heads — `[b, kv, s, d] → [b, kv·g, s, d]`,
/// head `kv_idx·g + j` served by KV head `kv_idx` (so query head `i` uses KV head `i / g`). `g == 1` is
/// a no-op (SDXL/standard MHA).
fn repeat_kv(x: &Tensor, g: usize) -> CandleResult<Tensor> {
    if g == 1 {
        return Ok(x.clone());
    }
    let (b, kv, s, d) = x.dims4()?;
    x.unsqueeze(2)?
        .broadcast_as((b, kv, g, s, d))?
        .contiguous()?
        .reshape((b, kv * g, s, d))
}

/// Flat `[b·s·s]` additive-mask data: for each batch row, query `i` is masked off key `j` (value
/// `-1e30`) unless that row's own padding allows it (causal AND non-pad key, OR a pad query). Pure (no
/// device), so the per-row mask logic is unit-testable. Assumes `m.len() == b·s`.
fn causal_padding_data(m: &[u32], b: usize, s: usize) -> Vec<f32> {
    let neg = -1e30f32;
    let mut data = vec![0f32; b * s * s];
    for bi in 0..b {
        let row = &m[bi * s..(bi + 1) * s];
        for i in 0..s {
            let pad_query = row[i] == 0;
            for j in 0..s {
                let allowed = pad_query || (j <= i && row[j] != 0);
                if !allowed {
                    data[(bi * s + i) * s + j] = neg;
                }
            }
        }
    }
    data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn causal_padding_data_uses_per_row_padding() {
        // Row 0 all-real ([1,1,1]); row 1 key 0 padded ([0,1,1]).
        let s = 3;
        let m = [1u32, 1, 1, /* row 1 */ 0, 1, 1];
        let data = causal_padding_data(&m, 2, s);
        let neg = -1e30f32;
        let at = |bi: usize, i: usize, j: usize| data[(bi * s + i) * s + j];

        // Row 0 is a plain causal mask.
        assert_eq!(at(0, 0, 1), neg); // future key masked
        assert_eq!(at(0, 1, 0), 0.0); // past real key attended

        // Row 1, query 1 (real): key 0 is padding ⇒ masked even though causal.
        assert_eq!(at(1, 1, 0), neg);
        assert_eq!(at(1, 1, 1), 0.0);
        assert_ne!(at(0, 1, 0), at(1, 1, 0));

        // Row 1, query 0 is itself padding ⇒ attends everything.
        assert_eq!(at(1, 0, 0), 0.0);
        assert_eq!(at(1, 0, 2), 0.0);
        assert_ne!(at(0, 0, 2), at(1, 0, 2));
    }
}
