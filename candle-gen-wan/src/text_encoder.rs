//! **UMT5-XXL** text encoder (`google/umt5-xxl`, `UMT5EncoderModel`) — Wan's prompt encoder, run in
//! f32. Three deviations from vanilla HF T5 the port honors:
//! 1. **Per-layer** relative-position bias (`shared_pos = False`): every block owns its own
//!    `[num_buckets, num_heads]` bucket table (`encoder.block.{i}.layer.0.SelfAttention.relative_attention_bias`).
//! 2. **Gated-GELU** FFN (`DenseReluDense.{wi_0, wi_1, wo}`): `wo(gelu(wi_0(x)) · wi_1(x))`.
//! 3. **No** `1/√d` attention scaling (T5 folds it into the weights); `T5LayerNorm` is RMS (no mean
//!    subtraction, no bias).

use candle_gen::candle_core::{DType, Device, Result, Tensor, D};
use candle_gen::candle_nn::ops::softmax_last_dim;
use candle_gen::candle_nn::VarBuilder;
use candle_gen::quant::{embedding_gs, QEmbedding, MLX_GROUP_SIZE};

use crate::config::TextEncoderConfig;
use crate::quant::QLinear;

/// T5 RMS LayerNorm (no centering, no bias), computed in f32.
fn t5_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let dt = x.dtype();
    let xf = x.to_dtype(DType::F32)?;
    let var = xf.sqr()?.mean_keepdim(D::Minus1)?;
    let normed = xf.broadcast_div(&(var + eps)?.sqrt()?)?;
    normed.to_dtype(dt)?.broadcast_mul(weight)
}

struct Attention {
    q: QLinear,
    k: QLinear,
    v: QLinear,
    o: QLinear,
    rel_bias: Tensor, // [num_buckets, num_heads]
    num_heads: usize,
    d_kv: usize,
}

impl Attention {
    fn new(cfg: &TextEncoderConfig, vb: VarBuilder) -> Result<Self> {
        let inner = cfg.num_heads * cfg.d_kv;
        Ok(Self {
            // Bias-less UMT5 projections, packed-detect (sc-10025) — dense in the hosted Wan tier.
            q: QLinear::linear_detect(cfg.d_model, inner, &vb, "q", false)?,
            k: QLinear::linear_detect(cfg.d_model, inner, &vb, "k", false)?,
            v: QLinear::linear_detect(cfg.d_model, inner, &vb, "v", false)?,
            o: QLinear::linear_detect(inner, cfg.d_model, &vb, "o", false)?,
            rel_bias: vb
                .pp("relative_attention_bias")
                .get((cfg.num_buckets, cfg.num_heads), "weight")?,
            num_heads: cfg.num_heads,
            d_kv: cfg.d_kv,
        })
    }

    /// `x`: `[B, S, d_model]`; `bucket_idx`: a host `[S*S]` of relative-position bucket ids.
    fn forward(&self, x: &Tensor, bucket_idx: &Tensor, s: usize) -> Result<Tensor> {
        let (b, _, _) = x.dims3()?;
        let to_heads = |t: &Tensor| -> Result<Tensor> {
            t.reshape((b, s, self.num_heads, self.d_kv))?
                .transpose(1, 2)?
                .contiguous()
        };
        let q = to_heads(&self.q.forward(x)?)?; // [B,H,S,d_kv]
        let k = to_heads(&self.k.forward(x)?)?;
        let v = to_heads(&self.v.forward(x)?)?;
        // No 1/sqrt(d) scaling (T5 convention).
        let scores = q.matmul(&k.transpose(D::Minus2, D::Minus1)?.contiguous()?)?; // [B,H,S,S]
                                                                                   // position bias: gather rel_bias[bucket] → [S*S,H] → [H,S,S].
        let bias = self
            .rel_bias
            .index_select(bucket_idx, 0)? // [S*S, H]
            .reshape((s, s, self.num_heads))?
            .permute((2, 0, 1))?
            .unsqueeze(0)?
            .contiguous()?; // [1,H,S,S]
        let attn = softmax_last_dim(&scores.broadcast_add(&bias)?)?;
        let ctx = attn.matmul(&v)?; // [B,H,S,d_kv]
        let ctx = ctx
            .transpose(1, 2)?
            .reshape((b, s, self.num_heads * self.d_kv))?;
        self.o.forward(&ctx)
    }
}

struct Ffn {
    wi_0: QLinear,
    wi_1: QLinear,
    wo: QLinear,
}

impl Ffn {
    fn new(cfg: &TextEncoderConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            wi_0: QLinear::linear_detect(cfg.d_model, cfg.d_ff, &vb, "wi_0", false)?,
            wi_1: QLinear::linear_detect(cfg.d_model, cfg.d_ff, &vb, "wi_1", false)?,
            wo: QLinear::linear_detect(cfg.d_ff, cfg.d_model, &vb, "wo", false)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.wi_0.forward(x)?.gelu()?; // tanh-approx GELU
        let up = self.wi_1.forward(x)?;
        self.wo.forward(&(gate * up)?)
    }
}

struct Block {
    norm1: Tensor,
    attn: Attention,
    norm2: Tensor,
    ffn: Ffn,
}

impl Block {
    fn new(cfg: &TextEncoderConfig, vb: VarBuilder) -> Result<Self> {
        let l0 = vb.pp("layer").pp("0");
        let l1 = vb.pp("layer").pp("1");
        Ok(Self {
            norm1: l0.pp("layer_norm").get(cfg.d_model, "weight")?,
            attn: Attention::new(cfg, l0.pp("SelfAttention"))?,
            norm2: l1.pp("layer_norm").get(cfg.d_model, "weight")?,
            ffn: Ffn::new(cfg, l1.pp("DenseReluDense"))?,
        })
    }

    fn forward(&self, x: &Tensor, bucket_idx: &Tensor, s: usize, eps: f64) -> Result<Tensor> {
        let h = self
            .attn
            .forward(&t5_norm(x, &self.norm1, eps)?, bucket_idx, s)?;
        let x = (x + h)?;
        let h = self.ffn.forward(&t5_norm(&x, &self.norm2, eps)?)?;
        x + h
    }
}

pub struct Umt5Encoder {
    shared: QEmbedding,
    blocks: Vec<Block>,
    final_norm: Tensor,
    cfg: TextEncoderConfig,
    device: Device,
}

impl Umt5Encoder {
    pub fn new(cfg: &TextEncoderConfig, vb: VarBuilder) -> Result<Self> {
        // Packed-detect the `shared` embedding (sc-10025) via the shared loader — dense in the hosted Wan
        // tier (the MLX build keeps the T5 dense); the packed arm future-proofs + closes the guard. Its
        // dense fallback is the shape-checked `candle_nn::embedding`, so the forward path is unchanged.
        let shared = embedding_gs(&vb, "shared", cfg.vocab_size, cfg.d_model, MLX_GROUP_SIZE)?;
        let enc = vb.pp("encoder");
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            blocks.push(Block::new(cfg, enc.pp("block").pp(i))?);
        }
        let final_norm = enc.pp("final_layer_norm").get(cfg.d_model, "weight")?;
        Ok(Self {
            shared,
            blocks,
            final_norm,
            cfg: *cfg,
            device: vb.device().clone(),
        })
    }

    /// Bidirectional T5 relative-position bucket for `key_pos - query_pos`.
    fn rp_bucket(&self, relative_position: i64) -> u32 {
        let num_buckets = (self.cfg.num_buckets / 2) as i64; // 16
        let mut ret = if relative_position > 0 {
            num_buckets
        } else {
            0
        };
        let n = relative_position.abs();
        let max_exact = num_buckets / 2; // 8
        let bucket = if n < max_exact {
            n
        } else {
            let v = max_exact
                + ((n as f64 / max_exact as f64).ln()
                    / (self.cfg.max_distance as f64 / max_exact as f64).ln()
                    * (num_buckets - max_exact) as f64) as i64;
            v.min(num_buckets - 1)
        };
        ret += bucket;
        ret as u32
    }

    fn bucket_grid(&self, s: usize) -> Result<Tensor> {
        let mut idx = Vec::with_capacity(s * s);
        for i in 0..s {
            for j in 0..s {
                idx.push(self.rp_bucket(j as i64 - i as i64));
            }
        }
        Tensor::from_vec(idx, s * s, &self.device)
    }

    /// `input_ids`: `[1, S]` (u32) → prompt embeds `[1, S, d_model]` in f32.
    pub fn encode(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (_b, s) = input_ids.dims2()?;
        let bucket_idx = self.bucket_grid(s)?;
        let mut x = self.shared.forward(input_ids)?; // [1,S,d_model] f32
        for blk in &self.blocks {
            x = blk.forward(&x, &bucket_idx, s, self.cfg.eps)?;
        }
        t5_norm(&x, &self.final_norm, self.cfg.eps)
    }
}
