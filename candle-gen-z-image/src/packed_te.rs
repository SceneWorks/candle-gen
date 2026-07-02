//! Vendored **inference** Z-Image Qwen3 text encoder with a packed-load seam (sc-9408).
//!
//! A faithful copy of the stock `candle-transformers` `z_image::text_encoder` (Qwen3 adapter) at the
//! workspace candle pin (`c1e6756`), vendored because its `Attention` / `Mlp` / `embed_tokens` build
//! frozen `candle_nn::Linear` / `Embedding` with no seam, so they cannot load the pre-quantized
//! MLX-packed TE tier (`model.embed_tokens` + every `self_attn.{q,k,v,o}_proj` / `mlp.{gate,up,down}_proj`
//! is packed as `{base}.weight` u32 + `.scales` + `.biases`). Every packed projection is a
//! [`crate::quant::QLinear`] / the embedding a [`crate::quant::QEmbedding`], both packed-**detecting**
//! the `.scales` sibling; the RMSNorms (`input_layernorm`, `post_attention_layernorm`, per-head
//! `q_norm`/`k_norm`) stay dense.
//!
//! **The dense path is byte-identical to the stock encoder** (`parity_tests` pins it: no `.scales`
//! present ⇒ every projection takes the dense arm, and the forward matches the stock forward). The
//! forward math is the stock Qwen3: per-head q/k RMSNorm, RoPE, GQA `repeat_kv`, and the Z-Image quirk
//! — return the **second-to-last** layer hidden state (`num_hidden_layers - 2`) with **no final norm**.
//! Used only when the snapshot is a packed tier ([`crate::pipeline`]); a dense snapshot keeps the stock
//! `ZImageTextEncoder`.

use std::sync::Arc;

use candle_gen::candle_core::{DType, Device, Module, Result, Tensor};
use candle_gen::candle_nn::{Activation, RmsNorm, VarBuilder};
use candle_transformers::models::z_image::text_encoder::TextEncoderConfig;

use crate::quant::{QEmbedding, QLinear};

// ==================== Rotary Embedding (copied verbatim — no weights) ====================

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, cfg: &TextEncoderConfig, dev: &Device) -> Result<Self> {
        let dim = cfg.head_dim;
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(DType::F32)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
        })
    }

    /// Apply RoPE (q, k shape: B x H x L x D).
    fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?;
        let sin = self.sin.narrow(0, offset, seq_len)?;
        let q_embed = candle_gen::candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_gen::candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

// ==================== MLP (packed seam) ====================
struct Mlp {
    gate_proj: QLinear,
    up_proj: QLinear,
    down_proj: QLinear,
    act_fn: Activation,
}

impl Mlp {
    fn new(cfg: &TextEncoderConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: QLinear::linear_detect(
                cfg.hidden_size,
                cfg.intermediate_size,
                &vb,
                "gate_proj",
                false,
            )?,
            up_proj: QLinear::linear_detect(
                cfg.hidden_size,
                cfg.intermediate_size,
                &vb,
                "up_proj",
                false,
            )?,
            down_proj: QLinear::linear_detect(
                cfg.intermediate_size,
                cfg.hidden_size,
                &vb,
                "down_proj",
                false,
            )?,
            act_fn: cfg.hidden_act,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let lhs = self.gate_proj.forward(x)?.apply(&self.act_fn)?;
        let rhs = self.up_proj.forward(x)?;
        self.down_proj.forward(&(lhs * rhs)?)
    }
}

// ==================== Attention (packed seam) ====================

fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        Ok(x)
    } else {
        let (b_sz, n_kv_head, seq_len, head_dim) = x.dims4()?;
        x.unsqueeze(2)?
            .broadcast_as((b_sz, n_kv_head, n_rep, seq_len, head_dim))?
            .reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))
    }
}
struct Attention {
    q_proj: QLinear,
    k_proj: QLinear,
    v_proj: QLinear,
    o_proj: QLinear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    rotary_emb: Arc<RotaryEmbedding>,
}

impl Attention {
    fn new(
        cfg: &TextEncoderConfig,
        rotary_emb: Arc<RotaryEmbedding>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let head_dim = cfg.head_dim;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;

        // Qwen3 attention is bias-less (`cfg.attention_bias == false`).
        let bias = cfg.attention_bias;
        let q_proj =
            QLinear::linear_detect(cfg.hidden_size, num_heads * head_dim, &vb, "q_proj", bias)?;
        let k_proj = QLinear::linear_detect(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            &vb,
            "k_proj",
            bias,
        )?;
        let v_proj = QLinear::linear_detect(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            &vb,
            "v_proj",
            bias,
        )?;
        let o_proj =
            QLinear::linear_detect(num_heads * head_dim, cfg.hidden_size, &vb, "o_proj", bias)?;

        let q_norm = candle_gen::candle_nn::rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = candle_gen::candle_nn::rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;

        let hidden_size = head_dim * cfg.num_attention_heads;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            hidden_size,
            rotary_emb,
        })
    }

    fn forward(&self, x: &Tensor, attn_mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((b, l, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Per-head RMSNorm (Qwen3).
        let q_flat = q.flatten(0, 2)?;
        let k_flat = k.flatten(0, 2)?;
        let q_flat = self.q_norm.forward(&q_flat)?;
        let k_flat = self.k_norm.forward(&k_flat)?;
        let q = q_flat.reshape((b, self.num_heads, l, self.head_dim))?;
        let k = k_flat.reshape((b, self.num_kv_heads, l, self.head_dim))?;

        let (q, k) = self.rotary_emb.apply(&q, &k, offset)?;

        let k = repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(v, self.num_kv_groups)?.contiguous()?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        if let Some(m) = attn_mask {
            scores = scores.broadcast_add(m)?;
        }
        let probs = candle_gen::candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?;

        let ctx = ctx.transpose(1, 2)?.reshape((b, l, self.hidden_size))?;
        self.o_proj.forward(&ctx)
    }
}

// ==================== Decoder Layer ====================
struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    ln1: RmsNorm,
    ln2: RmsNorm,
}

impl DecoderLayer {
    fn new(cfg: &TextEncoderConfig, rotary: Arc<RotaryEmbedding>, vb: VarBuilder) -> Result<Self> {
        let self_attn = Attention::new(cfg, rotary, vb.pp("self_attn"))?;
        let mlp = Mlp::new(cfg, vb.pp("mlp"))?;
        let ln1 = candle_gen::candle_nn::rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("input_layernorm"),
        )?;
        let ln2 = candle_gen::candle_nn::rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            ln1,
            ln2,
        })
    }

    fn forward(&self, x: &Tensor, mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let h = self.ln1.forward(x)?;
        let h = self.self_attn.forward(&h, mask, offset)?;
        let x = (x + h)?;
        let h2 = self.ln2.forward(&x)?;
        let h2 = self.mlp.forward(&h2)?;
        x + h2
    }
}

// ==================== ZImageTextEncoder (packed seam) ====================

/// Z-Image Qwen3 text encoder. Returns the second-to-last layer hidden state without a final RMSNorm.
/// Built from the *same* `model.*` keys as the stock encoder; on a dense tier (no `.scales`) the
/// forward is byte-identical to the stock encoder (`parity_tests`).
pub struct ZImageTextEncoder {
    embed_tokens: QEmbedding,
    layers: Vec<DecoderLayer>,
    num_hidden_layers: usize,
    device: Device,
    dtype: DType,
}

impl ZImageTextEncoder {
    pub fn new(cfg: &TextEncoderConfig, vb: VarBuilder) -> Result<Self> {
        // Weights live under the `model.` prefix.
        let vb_model = vb.pp("model");

        let embed_tokens =
            QEmbedding::detect(&vb_model, "embed_tokens", cfg.vocab_size, cfg.hidden_size)?;

        let rotary = Arc::new(RotaryEmbedding::new(vb.dtype(), cfg, vb.device())?);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_layers = vb_model.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(cfg, rotary.clone(), vb_layers.pp(i))?);
        }
        // The final norm (`model.norm.weight`) is intentionally NOT loaded — the encoder returns the
        // second-to-last layer output without the final norm.

        Ok(Self {
            embed_tokens,
            layers,
            num_hidden_layers: cfg.num_hidden_layers,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    fn causal_mask(&self, b: usize, tgt: usize, offset: usize) -> Result<Tensor> {
        let minf = f32::NEG_INFINITY;
        let mask: Vec<_> = (0..tgt)
            .flat_map(|i| {
                (0..(tgt + offset)).map(move |j| if j <= i + offset { 0.0 } else { minf })
            })
            .collect();
        Tensor::from_slice(&mask, (b, 1, tgt, tgt + offset), &self.device)?.to_dtype(self.dtype)
    }

    /// Encode `input_ids` `(B, seq_len)` → the layer[-2] hidden states `(B, seq_len, hidden)`, WITHOUT
    /// the final RMSNorm (the Z-Image convention the pipeline's `cap_feats` relies on).
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (b, l) = input_ids.dims2()?;
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;

        let causal = if l == 1 {
            None
        } else {
            Some(self.causal_mask(b, l, 0)?)
        };

        let target_layer = self.num_hidden_layers - 2;
        for (i, layer) in self.layers.iter().enumerate() {
            hidden_states = layer.forward(&hidden_states, causal.as_ref(), 0)?;
            if i == target_layer {
                return Ok(hidden_states);
            }
        }
        candle_gen::candle_core::bail!("z-image te: layer index out of bounds")
    }
}

#[cfg(test)]
mod parity_tests {
    //! Pin the vendored DENSE path to the stock Qwen3 encoder: same `VarMap` weights (no `.scales`),
    //! identical layer[-2] output — the guard that the packed-seam vendoring changed nothing on a dense
    //! tier.
    use super::*;
    use candle_gen::candle_core::{Device, Tensor};
    use candle_gen::candle_nn::{VarBuilder, VarMap};
    use candle_transformers::models::z_image::text_encoder::ZImageTextEncoder as StockTe;

    /// A tiny Qwen3 config: hidden 32, 4 heads / 2 kv-heads, head_dim 8, 4 layers — enough to exercise
    /// GQA repeat + per-head norm + the layer[-2] return cheaply on CPU.
    fn tiny_cfg() -> TextEncoderConfig {
        let mut cfg = TextEncoderConfig::z_image();
        cfg.vocab_size = 64;
        cfg.hidden_size = 32;
        cfg.intermediate_size = 48;
        cfg.num_hidden_layers = 4;
        cfg.num_attention_heads = 4;
        cfg.num_key_value_heads = 2;
        cfg.head_dim = 8;
        cfg
    }

    #[test]
    fn vendored_dense_te_matches_stock_forward() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        // Vendored built first (populates the VarMap); stock reads the same params — no `.scales`, so
        // every projection is Dense.
        let vendored = ZImageTextEncoder::new(&cfg, vb.clone()).unwrap();
        let stock = StockTe::new(&cfg, vb).unwrap();

        let ids = Tensor::from_vec(vec![1u32, 5, 12, 0, 3, 7], (1, 6), &dev).unwrap();
        let y_v = vendored.forward(&ids).unwrap();
        let y_s = stock.forward(&ids).unwrap();
        assert_eq!(y_v.dims(), y_s.dims());
        let diff = (y_v - y_s)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            diff < 1e-5,
            "vendored dense TE diverged from stock by {diff}"
        );
    }
}
