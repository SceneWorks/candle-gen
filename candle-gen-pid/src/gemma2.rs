//! Gemma-2-2B-IT decoder — PiD's caption text encoder. Port of HF `Gemma2Model` (the
//! `.get_decoder()` stack PiD loads: embedding → 26 norm-sandwich decoder layers → final RMSNorm →
//! last-hidden `[B, L, 2304]`; no lm_head / final-logit-softcap needed). Faithful port of
//! `mlx-gen-pid`'s `gemma2.rs` in candle idioms; runs f32.
//!
//! Gemma-2 specifics (vs the Gemma-3 LTX port): **attention logit soft-capping** `50·tanh(s/50)`
//! pre-softmax (so no fused SDPA — explicit attention), **no q/k norm**, RoPE is the standard HF
//! **rotate_half** convention (not PiD's interleaved), attention scale `query_pre_attn_scalar^-0.5`,
//! GQA (8 query / 4 KV heads, head_dim 256 — independent of hidden 2304), gelu-tanh MLP, and the
//! norm-sandwich block. RMSNorm is Gemma's `x·rsqrt(mean(x²)+eps)·(1+w)` (the `+1` pre-folded at load);
//! token embeddings are scaled by `√hidden_size`.
//!
//! PiD captions are ≤300 tokens ≪ the 4096 sliding window, so every layer is plain full-causal — a
//! single causal (+ optional padding) mask suffices.

use candle_gen::candle_core::{DType, Device, Tensor, D};
use candle_gen::candle_nn::ops::softmax_last_dim;
use candle_gen::candle_nn::{Linear, Module};
use candle_gen::{Result, Weights};

use crate::nn::rms;

/// Gemma-2 decoder configuration.
#[derive(Debug, Clone)]
pub struct Gemma2Config {
    pub hidden_size: i32,
    pub num_layers: i32,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub intermediate_size: i32,
    pub rope_theta: f32,
    pub attn_softcap: f32,
    pub query_pre_attn_scalar: f32,
    pub rms_eps: f32,
}

impl Gemma2Config {
    /// The released `gemma-2-2b-it` config.
    pub fn gemma_2_2b() -> Self {
        Self {
            hidden_size: 2304,
            num_layers: 26,
            num_heads: 8,
            num_kv_heads: 4,
            head_dim: 256,
            intermediate_size: 9216,
            rope_theta: 10000.0,
            attn_softcap: 50.0,
            query_pre_attn_scalar: 256.0,
            rms_eps: 1e-6,
        }
    }
}

/// Bias-less Linear over a raw `{key}` weight (Gemma projections carry no bias).
fn lin(w: &Weights, key: &str) -> Result<Linear> {
    Ok(Linear::new(w.require(key)?, None))
}

/// `weight + 1.0` (Gemma RMSNorm scale), pre-folded at load so [`crate::nn::rms`] applies it directly.
fn norm_alpha(w: &Weights, key: &str) -> Result<Tensor> {
    Ok((w.require(key)? + 1.0)?)
}

/// Host `(cos, sin)` `[seq, head_dim]` (f32) for HF rotate_half RoPE: `emb = cat(freqs, freqs)`.
fn rope_tables(head_dim: i32, seq: i32, theta: f32, device: &Device) -> Result<(Tensor, Tensor)> {
    let half = (head_dim / 2) as usize;
    let inv: Vec<f64> = (0..half)
        .map(|i| 1.0 / (theta as f64).powf((2 * i) as f64 / head_dim as f64))
        .collect();
    let hd = head_dim as usize;
    let s = seq as usize;
    let mut cos = vec![0f32; s * hd];
    let mut sin = vec![0f32; s * hd];
    for p in 0..s {
        for j in 0..hd {
            let f = inv[j % half]; // emb = cat(freqs, freqs) -> index wraps at half
            let a = p as f64 * f;
            cos[p * hd + j] = a.cos() as f32;
            sin[p * hd + j] = a.sin() as f32;
        }
    }
    Ok((
        Tensor::from_vec(cos, (s, hd), device)?,
        Tensor::from_vec(sin, (s, hd), device)?,
    ))
}

/// `rotate_half(x) = cat(-x[..,h:], x[..,:h])` for `[B,H,L,D]`.
fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let d = x.dim(D::Minus1)?;
    let x1 = x.narrow(D::Minus1, 0, d / 2)?;
    let x2 = x.narrow(D::Minus1, d / 2, d / 2)?;
    Ok(Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?)
}

/// `x·cos + rotate_half(x)·sin` with `cos`/`sin` `[L, D]` broadcast over `[B,H,L,D]`.
fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (l, d) = cos.dims2()?;
    let cos = cos.reshape((1, 1, l, d))?;
    let sin = sin.reshape((1, 1, l, d))?;
    let xc = x.broadcast_mul(&cos)?;
    let xs = rotate_half(x)?.broadcast_mul(&sin)?;
    Ok((xc + xs)?)
}

/// Repeat KV heads `n_rep×` along the head axis (`[B,nkv,L,D]` → `[B,nkv·n_rep,L,D]`).
fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let (b, nkv, l, d) = x.dims4()?;
    Ok(x.unsqueeze(2)?
        .broadcast_as((b, nkv, n_rep, l, d))?
        .reshape((b, nkv * n_rep, l, d))?)
}

struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f64,
    softcap: f64,
}

impl Attention {
    fn from_weights(w: &Weights, prefix: &str, cfg: &Gemma2Config) -> Result<Self> {
        Ok(Self {
            q: lin(w, &format!("{prefix}.q_proj.weight"))?,
            k: lin(w, &format!("{prefix}.k_proj.weight"))?,
            v: lin(w, &format!("{prefix}.v_proj.weight"))?,
            o: lin(w, &format!("{prefix}.o_proj.weight"))?,
            num_heads: cfg.num_heads as usize,
            num_kv_heads: cfg.num_kv_heads as usize,
            head_dim: cfg.head_dim as usize,
            scale: (cfg.query_pre_attn_scalar as f64).powf(-0.5),
            softcap: cfg.attn_softcap as f64,
        })
    }

    /// `x`: `[B,L,hidden]`; `cos`/`sin`: `[L,head_dim]`; `mask`: additive `[1,1,L,L]`.
    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;
        let hd = self.head_dim;
        let to_heads = |a: Tensor, nh: usize| -> Result<Tensor> {
            Ok(a.reshape((b, l, nh, hd))?.transpose(1, 2)?.contiguous()?)
        };
        let q = apply_rope(&to_heads(self.q.forward(x)?, self.num_heads)?, cos, sin)?;
        let k = apply_rope(&to_heads(self.k.forward(x)?, self.num_kv_heads)?, cos, sin)?;
        let v = to_heads(self.v.forward(x)?, self.num_kv_heads)?;
        let n_rep = self.num_heads / self.num_kv_heads;
        let k = repeat_kv(&k, n_rep)?;
        let v = repeat_kv(&v, n_rep)?;

        // explicit attention (logit soft-cap blocks fused SDPA): softcap·tanh(scores/softcap)
        let scores = (q.contiguous()?.matmul(&k.transpose(2, 3)?.contiguous()?)? * self.scale)?;
        let scores = ((scores * (1.0 / self.softcap))?.tanh()? * self.softcap)?;
        let scores = scores.broadcast_add(mask)?;
        let attn = softmax_last_dim(&scores)?;
        let out = attn.matmul(&v.contiguous()?)?; // [B,H,L,D]
        let out = out
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, l, self.num_heads * hd))?;
        Ok(self.o.forward(&out)?)
    }
}

struct Mlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl Mlp {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            gate: lin(w, &format!("{prefix}.gate_proj.weight"))?,
            up: lin(w, &format!("{prefix}.up_proj.weight"))?,
            down: lin(w, &format!("{prefix}.down_proj.weight"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let g = self.gate.forward(x)?.gelu()?; // gelu-tanh
        Ok(self.down.forward(&(g * self.up.forward(x)?)?)?)
    }
}

struct Layer {
    input_ln: Tensor,
    attn: Attention,
    post_attn_ln: Tensor,
    pre_ff_ln: Tensor,
    mlp: Mlp,
    post_ff_ln: Tensor,
    eps: f32,
}

impl Layer {
    fn from_weights(w: &Weights, prefix: &str, cfg: &Gemma2Config) -> Result<Self> {
        Ok(Self {
            input_ln: norm_alpha(w, &format!("{prefix}.input_layernorm.weight"))?,
            attn: Attention::from_weights(w, &format!("{prefix}.self_attn"), cfg)?,
            post_attn_ln: norm_alpha(w, &format!("{prefix}.post_attention_layernorm.weight"))?,
            pre_ff_ln: norm_alpha(w, &format!("{prefix}.pre_feedforward_layernorm.weight"))?,
            mlp: Mlp::from_weights(w, &format!("{prefix}.mlp"))?,
            post_ff_ln: norm_alpha(w, &format!("{prefix}.post_feedforward_layernorm.weight"))?,
            eps: cfg.rms_eps,
        })
    }

    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let h = self
            .attn
            .forward(&rms(x, &self.input_ln, self.eps)?, cos, sin, mask)?;
        let x = (x + rms(&h, &self.post_attn_ln, self.eps)?)?;
        let h = self.mlp.forward(&rms(&x, &self.pre_ff_ln, self.eps)?)?;
        Ok((&x + rms(&h, &self.post_ff_ln, self.eps)?)?)
    }
}

/// The Gemma-2 decoder (caption encoder).
pub struct Gemma2 {
    embed: Tensor, // [vocab, hidden]
    layers: Vec<Layer>,
    norm: Tensor,
    cfg: Gemma2Config,
    device: Device,
}

impl Gemma2 {
    /// `prefix` is `"model."` for the HF checkpoint layout.
    pub fn from_weights(w: &Weights, prefix: &str, cfg: &Gemma2Config) -> Result<Self> {
        let layers = (0..cfg.num_layers)
            .map(|i| Layer::from_weights(w, &format!("{prefix}layers.{i}"), cfg))
            .collect::<Result<Vec<_>>>()?;
        let embed = w.require(&format!("{prefix}embed_tokens.weight"))?;
        let device = embed.device().clone();
        Ok(Self {
            embed,
            layers,
            norm: norm_alpha(w, &format!("{prefix}norm.weight"))?,
            cfg: cfg.clone(),
            device,
        })
    }

    /// The device the encoder's weights live on (so callers build `ids`/`mask` tensors there).
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// `ids`: `[B, L]` (u32). `pad_mask`: optional `[B, L]` (1 = real, 0 = pad). Returns the
    /// last-hidden states `[B, L, hidden]` (f32).
    pub fn forward(&self, ids: &Tensor, pad_mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, l) = ids.dims2()?;
        let hidden = self.cfg.hidden_size as usize;

        // embed + √hidden scale
        let flat = ids.reshape((b * l,))?;
        let emb = self.embed.index_select(&flat, 0)?.reshape((b, l, hidden))?;
        let normalizer = (self.cfg.hidden_size as f64).sqrt();
        let mut x = (emb * normalizer)?;

        let (cos, sin) = rope_tables(
            self.cfg.head_dim,
            l as i32,
            self.cfg.rope_theta,
            &self.device,
        )?;
        let mask = self.causal_mask(b, l, pad_mask)?;
        for layer in &self.layers {
            x = layer.forward(&x, &cos, &sin, &mask)?;
        }
        rms(&x, &self.norm, self.cfg.rms_eps)
    }

    /// Additive `[B,1,L,L]` causal mask (0 where a query may attend, large-negative otherwise),
    /// optionally also masking padding keys (`pad_mask[b,j]==0`).
    fn causal_mask(&self, b: usize, l: usize, pad_mask: Option<&Tensor>) -> Result<Tensor> {
        let neg = -1e9f32;
        let mut m = vec![0f32; l * l];
        for i in 0..l {
            for j in 0..l {
                if j > i {
                    m[i * l + j] = neg;
                }
            }
        }
        let causal = Tensor::from_vec(m, (1, 1, l, l), &self.device)?;
        match pad_mask {
            None => Ok(causal),
            Some(pm) => {
                // pad_mask [B,L] (1 real / 0 pad) -> additive key mask [B,1,1,L]
                let pad = pm.to_dtype(DType::F32)?.reshape((b, 1, 1, l))?;
                // (1 - pad) * neg
                let key_add = ((1.0 - pad)? * neg as f64)?;
                Ok(causal.broadcast_add(&key_add)?)
            }
        }
    }
}
