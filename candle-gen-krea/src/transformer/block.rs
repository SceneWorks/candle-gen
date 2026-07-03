//! Krea 2 DiT building blocks — port of `mlx-gen-krea`'s `transformer/block.rs` (the reference
//! `mmdit.py` modules): the sigmoid-**gated** GQA attention (`Attention` + `QKNorm`), the `SwiGLU`
//! FFN, the `+1` `RMSNorm`, the un-modulated `TextFusionBlock`, the `DoubleSharedModulation`
//! single-stream block, and the `TextFusionTransformer` layer aggregator.
//!
//! Every `RMSNorm` here computes `weight = scale + 1` in f32 (the reference stores the raw `scale`,
//! centered at 0), distinct from the apply-weight-directly norms of the Qwen3-VL text encoder.
//! Attention adds a `to_gate` projection: the post-attention output is multiplied by
//! `sigmoid(to_gate(x))` before `to_out`. Block gates (`pregate`/`postgate`) are raw (no activation).

use candle_gen::candle_core::{Result, Tensor, D};
use candle_gen::candle_nn::ops::{sigmoid, softmax_last_dim};
use candle_gen::candle_nn::{Linear, Module};

use super::rope::apply_interleaved_rope;
use crate::loader::{linear, linear_detect, rms_scale, rms_scale_weight, Weights};
use crate::quant::QLinear;

/// Join a module prefix with a leaf name.
fn join(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}.{name}")
    }
}

/// Repeat each kv head `groups` times consecutively ([b,s,hkv,hd] → [b,s,hkv·groups,hd]) —
/// `repeat_interleave` over the head axis, matching the reference `enable_gqa`.
fn repeat_kv(x: &Tensor, groups: usize) -> Result<Tensor> {
    if groups == 1 {
        return Ok(x.clone());
    }
    let (b, s, hkv, hd) = x.dims4()?;
    x.unsqueeze(3)?
        .expand((b, s, hkv, groups, hd))?
        .contiguous()?
        .reshape((b, s, hkv * groups, hd))
}

/// Bidirectional, unmasked scaled-dot-product attention. `q`/`k`/`v`: `[b, h, s, hd]`.
///
/// i32-overflow guard (sc-9116): the image-token scores `[b, h, s, s]` reach `~24·16384² ≈ 6.4e9 >
/// i32::MAX` at a 2048² render, silently corrupting the tail rows on the candle CUDA kernels. The
/// shared budgeted helper chunks over the query rows (byte-identical for the common sizes); the softmax
/// closure preserves the exact fused `softmax_last_dim`.
fn sdpa(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64) -> Result<Tensor> {
    candle_gen::sdpa_budgeted_bhsd(
        q,
        k,
        v,
        scale,
        None,
        softmax_last_dim,
        candle_gen::ATTN_SCORES_BUDGET,
    )
}

// ── `+1` RMSNorm ────────────────────────────────────────────────────────────────────────────
/// Reference `RMSNorm`: `F.rms_norm(x.float(), weight = scale.float() + 1.0)` then cast back. The
/// stored param is the raw `scale` (centered at 0); the `+1` is pre-folded into an f32 weight at load.
pub struct RmsScale {
    weight: Tensor, // f32, = scale + 1
    eps: f64,
}

impl RmsScale {
    pub fn load(w: &Weights, key: &str, eps: f64) -> Result<Self> {
        Ok(Self {
            weight: rms_scale_weight(w, key)?,
            eps,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        rms_scale(x, &self.weight, self.eps)
    }
}

// ── Sigmoid-gated GQA attention (reference `Attention`) ─────────────────────────────────────
pub struct GatedAttention {
    q: QLinear,
    k: QLinear,
    v: QLinear,
    gate: QLinear,
    o: QLinear,
    norm_q: RmsScale,
    norm_k: RmsScale,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl GatedAttention {
    pub fn load(
        w: &Weights,
        prefix: &str,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        eps: f64,
    ) -> Result<Self> {
        Ok(Self {
            q: linear_detect(w, &join(prefix, "to_q"), false)?,
            k: linear_detect(w, &join(prefix, "to_k"), false)?,
            v: linear_detect(w, &join(prefix, "to_v"), false)?,
            gate: linear_detect(w, &join(prefix, "to_gate"), false)?,
            o: linear_detect(w, &join(prefix, "to_out.0"), false)?,
            norm_q: RmsScale::load(w, &join(prefix, "norm_q.weight"), eps)?,
            norm_k: RmsScale::load(w, &join(prefix, "norm_k.weight"), eps)?,
            heads,
            kv_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
        })
    }

    /// `x`: `[b, s, hidden]`. `rope`: `Some((cos, sin))` (`[1, s, head_dim/2]`) for the single-stream
    /// blocks; `None` for the text-fusion blocks (no positional encoding). Unmasked (B=1 full sequence).
    pub fn forward(&self, x: &Tensor, rope: Option<(&Tensor, &Tensor)>) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let (nh, nkv, hd) = (self.heads, self.kv_heads, self.head_dim);

        let q = self.q.forward(x)?.reshape((b, s, nh, hd))?;
        let k = self.k.forward(x)?.reshape((b, s, nkv, hd))?;
        let v = self.v.forward(x)?.reshape((b, s, nkv, hd))?;
        let gate = self.gate.forward(x)?; // [b, s, hidden]

        let q = self.norm_q.forward(&q)?;
        let k = self.norm_k.forward(&k)?;
        let (q, k) = match rope {
            Some((cos, sin)) => (
                apply_interleaved_rope(&q, cos, sin)?,
                apply_interleaved_rope(&k, cos, sin)?,
            ),
            None => (q, k),
        };

        let groups = nh / nkv;
        let k = repeat_kv(&k, groups)?;
        let v = repeat_kv(&v, groups)?;

        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;
        let o = sdpa(&q, &k, &v, self.scale)?;
        let o = o.transpose(1, 2)?.contiguous()?.reshape((b, s, nh * hd))?;

        // Sigmoid gate the attention output, then the shared output projection.
        let gated = (o * sigmoid(&gate)?)?;
        self.o.forward(&gated)
    }
}

// ── SwiGLU feed-forward (reference `SwiGLU`: `down(silu(gate(x)) * up(x))`) ──────────────────
pub struct SwiGlu {
    gate: QLinear,
    up: QLinear,
    down: QLinear,
}

impl SwiGlu {
    pub fn load(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            gate: linear_detect(w, &join(prefix, "gate"), false)?,
            up: linear_detect(w, &join(prefix, "up"), false)?,
            down: linear_detect(w, &join(prefix, "down"), false)?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gated = (self.gate.forward(x)?.silu()? * self.up.forward(x)?)?;
        self.down.forward(&gated)
    }
}

// ── Un-modulated text-fusion block (reference `TextFusionBlock`) ─────────────────────────────
/// `x = x + attn(prenorm(x)); x = x + mlp(postnorm(x))`. No modulation, no RoPE.
pub struct TextFusionBlock {
    prenorm: RmsScale,
    postnorm: RmsScale,
    attn: GatedAttention,
    mlp: SwiGlu,
}

impl TextFusionBlock {
    pub fn load(
        w: &Weights,
        prefix: &str,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        eps: f64,
    ) -> Result<Self> {
        Ok(Self {
            prenorm: RmsScale::load(w, &join(prefix, "norm1.weight"), eps)?,
            postnorm: RmsScale::load(w, &join(prefix, "norm2.weight"), eps)?,
            attn: GatedAttention::load(w, &join(prefix, "attn"), heads, kv_heads, head_dim, eps)?,
            mlp: SwiGlu::load(w, &join(prefix, "ff"))?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = (x + self.attn.forward(&self.prenorm.forward(x)?, None)?)?;
        &x + self.mlp.forward(&self.postnorm.forward(&x)?)?
    }
}

// ── DoubleSharedModulation single-stream block (reference `SingleStreamBlock`) ──────────────
/// `mod(tvec) = tvec + scale_shift_table` → 6 chunks `(prescale, preshift, pregate, postscale,
/// postshift, postgate)`; then
/// `x += pregate · attn((1+prescale)·prenorm(x) + preshift)` and
/// `x += postgate · mlp((1+postscale)·postnorm(x) + postshift)`. Gates are raw (no activation).
pub struct SingleStreamBlock {
    scale_shift_table: Tensor, // [1, 1, 6·hidden]
    prenorm: RmsScale,
    postnorm: RmsScale,
    attn: GatedAttention,
    mlp: SwiGlu,
}

impl SingleStreamBlock {
    pub fn load(
        w: &Weights,
        prefix: &str,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        hidden: usize,
        eps: f64,
    ) -> Result<Self> {
        // Stored `[6, hidden]`; flatten row-major to `[1, 1, 6·hidden]` so a single broadcast-add onto
        // `tvec` (`[b, 1, 6·hidden]`) and a 6-way split reproduce the reference's `chunk(6, -1)` order.
        let sst = w
            .get(&join(prefix, "scale_shift_table"))?
            .reshape((1, 1, 6 * hidden))?;
        Ok(Self {
            scale_shift_table: sst,
            prenorm: RmsScale::load(w, &join(prefix, "norm1.weight"), eps)?,
            postnorm: RmsScale::load(w, &join(prefix, "norm2.weight"), eps)?,
            attn: GatedAttention::load(w, &join(prefix, "attn"), heads, kv_heads, head_dim, eps)?,
            mlp: SwiGlu::load(w, &join(prefix, "ff"))?,
        })
    }

    /// `x`: `[b, s, hidden]`, `tvec`: `[b, 1, 6·hidden]` (shared `time_mod_proj` output), `cos`/`sin`:
    /// `[1, s, head_dim/2]`.
    pub fn forward(&self, x: &Tensor, tvec: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let m = tvec.broadcast_add(&self.scale_shift_table)?; // [b, 1, 6·hidden]
        let chunks = m.chunk(6, D::Minus1)?; // 6 × [b, 1, hidden]
        let (prescale, preshift, pregate) = (&chunks[0], &chunks[1], &chunks[2]);
        let (postscale, postshift, postgate) = (&chunks[3], &chunks[4], &chunks[5]);

        let pre = self
            .prenorm
            .forward(x)?
            .broadcast_mul(&(prescale + 1.0)?)?
            .broadcast_add(preshift)?;
        let attn = self.attn.forward(&pre, Some((cos, sin)))?;
        let x = (x + attn.broadcast_mul(pregate)?)?;

        let post = self
            .postnorm
            .forward(&x)?
            .broadcast_mul(&(postscale + 1.0)?)?
            .broadcast_add(postshift)?;
        let mlp = self.mlp.forward(&post)?;
        &x + mlp.broadcast_mul(postgate)?
    }
}

// ── TextFusionTransformer (reference `TextFusionTransformer`) ────────────────────────────────
/// Aggregates the `num_layers` stacked Qwen3-VL hidden states into one conditioning stream:
/// `layerwise_blocks` attend across the layer axis (per token) → `projector` collapses `num_layers→1`
/// → `refiner_blocks` attend across the token axis.
pub struct TextFusionTransformer {
    layerwise: Vec<TextFusionBlock>,
    projector: Linear, // Linear(num_layers → 1), no bias
    refiner: Vec<TextFusionBlock>,
}

impl TextFusionTransformer {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        w: &Weights,
        num_layerwise: usize,
        num_refiner: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        eps: f64,
    ) -> Result<Self> {
        let block = |i: usize, kind: &str| {
            TextFusionBlock::load(
                w,
                &format!("text_fusion.{kind}.{i}"),
                heads,
                kv_heads,
                head_dim,
                eps,
            )
        };
        Ok(Self {
            layerwise: (0..num_layerwise)
                .map(|i| block(i, "layerwise_blocks"))
                .collect::<Result<_>>()?,
            projector: linear(w, "text_fusion.projector", false)?,
            refiner: (0..num_refiner)
                .map(|i| block(i, "refiner_blocks"))
                .collect::<Result<_>>()?,
        })
    }

    /// `x`: `[b, n_tokens, num_layers, txt_dim]` (the stacked select-layer hidden states). Returns the
    /// fused conditioning `[b, n_tokens, txt_dim]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, n_tok, n_layers, d) = x.dims4()?;

        // Layerwise attention: each token's `num_layers` stack is a sequence (batch = b·n_tokens).
        let mut h = x.reshape((b * n_tok, n_layers, d))?;
        for blk in &self.layerwise {
            h = blk.forward(&h)?;
        }

        // `(b n_tok) n_layers d -> b n_tok d n_layers`, project `num_layers → 1`, drop the axis.
        let h = h
            .reshape((b, n_tok, n_layers, d))?
            .permute((0, 1, 3, 2))? // [b, n_tok, d, n_layers]
            .contiguous()?;
        let h = self
            .projector
            .forward(&h.reshape((b * n_tok * d, n_layers))?)?; // [b·n_tok·d, 1]
        let mut h = h.reshape((b, n_tok, d))?;

        // Token-axis refinement.
        for blk in &self.refiner {
            h = blk.forward(&h)?;
        }
        Ok(h)
    }
}
