//! Vendored **diffusers-layout** FLUX.1 text encoders (CLIP-L + T5-XXL) with the shared packed-load
//! seam (sc-9407) — the candle twin of the flux2-dev vendored text encoder (sc-9087) and the z-image
//! `packed_te` (sc-9408).
//!
//! **Why vendor.** The pre-quantized MLX tier (`SceneWorks/flux1-schnell-mlx`, epic 8506) packs the
//! CLIP-L text encoder (`text_model.*`) and the T5-XXL encoder (`encoder.block.*`, `shared`) as MLX
//! packed triples. The stock `candle-transformers` `ClipTextTransformer` / `T5EncoderModel` build their
//! projections through the plain `candle_nn::{linear, embedding}`, with no seam to load from packed
//! parts. So the packed path vendors minimal **encoder-only** copies here, building every `Linear` /
//! embedding through the packed-detecting [`crate::quant::QLinear`] / [`crate::quant::QEmbedding`]:
//! q4/q8 load straight from the packed parts (no dense staging), and a dense bf16 diffusers snapshot
//! (no `.scales`) loads through the same code unchanged.
//!
//! These are faithful ports of the stock forwards — CLIP: token+position embed → 12 pre-norm attention
//! layers (causal mask, QuickGELU MLP) → final LayerNorm → argmax-EOT pool; T5: `shared` embed → 24
//! encoder blocks (T5 RMSNorm, relative-position-bias self-attention on block 0, gated-GELU FF) → final
//! RMSNorm. The T5 relative-attention-bucket math is copied verbatim from the stock encoder. A CI test
//! pins a dense build of each against the stock model at 1e-4 on shared weights.

use candle_gen::candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_gen::candle_nn::{LayerNorm, Module, VarBuilder};

use crate::quant::{QEmbedding, QLinear};

// ============================================================================================
// CLIP-L text encoder (openai/clip-vit-large-patch14 layout, `text_model.` prefix).
// ============================================================================================

/// Fixed CLIP-L text config FLUX uses (identical across schnell/dev).
pub struct ClipConfig {
    pub vocab_size: usize,
    pub embed_dim: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
}

impl ClipConfig {
    pub fn flux() -> Self {
        Self {
            vocab_size: 49408,
            embed_dim: 768,
            intermediate_size: 3072,
            max_position_embeddings: 77,
            num_hidden_layers: 12,
            num_attention_heads: 12,
        }
    }
}

const CLIP_LN_EPS: f64 = 1e-5;

/// `nn::layer_norm` (affine) via explicit weight/bias load — the packed tier keeps the CLIP LayerNorms
/// dense (they carry no `.scales`), so a plain load suffices.
fn clip_layer_norm(dim: usize, vb: &VarBuilder, prefix: &str) -> Result<LayerNorm> {
    let w = vb.get(dim, &format!("{prefix}.weight"))?;
    let b = vb.get(dim, &format!("{prefix}.bias"))?;
    Ok(LayerNorm::new(w, b, CLIP_LN_EPS))
}

struct ClipEmbeddings {
    token_embedding: QEmbedding,
    position_embedding: QEmbedding,
    position_ids: Tensor,
}

impl ClipEmbeddings {
    fn new(cfg: &ClipConfig, vb: &VarBuilder) -> Result<Self> {
        let emb = vb.pp("embeddings");
        Ok(Self {
            token_embedding: QEmbedding::detect(
                &emb,
                "token_embedding",
                cfg.vocab_size,
                cfg.embed_dim,
            )?,
            position_embedding: QEmbedding::detect(
                &emb,
                "position_embedding",
                cfg.max_position_embeddings,
                cfg.embed_dim,
            )?,
            position_ids: Tensor::arange(0u32, cfg.max_position_embeddings as u32, vb.device())?
                .unsqueeze(0)?,
        })
    }

    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let seq_length = input_ids.dim(D::Minus1)?;
        let inputs_embeds = self.token_embedding.forward(input_ids)?;
        let position_ids = self.position_ids.narrow(1, 0, seq_length)?;
        let position_embedding = self.position_embedding.forward(&position_ids)?;
        inputs_embeds.broadcast_add(&position_embedding)
    }
}

struct ClipAttention {
    q_proj: QLinear,
    k_proj: QLinear,
    v_proj: QLinear,
    out_proj: QLinear,
    head_dim: usize,
    scale: f64,
    num_heads: usize,
}

impl ClipAttention {
    fn new(cfg: &ClipConfig, vb: &VarBuilder) -> Result<Self> {
        let d = cfg.embed_dim;
        let head_dim = d / cfg.num_attention_heads;
        let lin = |n: &str| QLinear::linear_detect(d, d, vb, n, true);
        Ok(Self {
            q_proj: lin("q_proj")?,
            k_proj: lin("k_proj")?,
            v_proj: lin("v_proj")?,
            out_proj: lin("out_proj")?,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
            num_heads: cfg.num_attention_heads,
        })
    }

    fn shape(&self, xs: &Tensor, seq_len: usize, bsz: usize) -> Result<Tensor> {
        xs.reshape((bsz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()
    }

    fn forward(&self, xs: &Tensor, causal_mask: &Tensor) -> Result<Tensor> {
        let in_dtype = xs.dtype();
        let (bsz, seq_len, embed_dim) = xs.dims3()?;
        let proj_shape = (bsz * self.num_heads, seq_len, self.head_dim);
        let q = (self.q_proj.forward(xs)? * self.scale)?;
        let q = self
            .shape(&q, seq_len, bsz)?
            .reshape(proj_shape)?
            .to_dtype(DType::F32)?;
        let k = self
            .shape(&self.k_proj.forward(xs)?, seq_len, bsz)?
            .reshape(proj_shape)?
            .to_dtype(DType::F32)?;
        let v = self
            .shape(&self.v_proj.forward(xs)?, seq_len, bsz)?
            .reshape(proj_shape)?
            .to_dtype(DType::F32)?;
        let attn_weights = q.matmul(&k.transpose(1, 2)?)?;
        let src_len = k.dim(1)?;
        let attn_weights = attn_weights
            .reshape((bsz, self.num_heads, seq_len, src_len))?
            .broadcast_add(causal_mask)?
            .reshape((bsz * self.num_heads, seq_len, src_len))?;
        let attn_weights = candle_gen::candle_nn::ops::softmax(&attn_weights, D::Minus1)?;
        let attn_output = attn_weights.matmul(&v)?.to_dtype(in_dtype)?;
        let attn_output = attn_output
            .reshape((bsz, self.num_heads, seq_len, self.head_dim))?
            .transpose(1, 2)?
            .reshape((bsz, seq_len, embed_dim))?;
        self.out_proj.forward(&attn_output)
    }
}

struct ClipMlp {
    fc1: QLinear,
    fc2: QLinear,
}

impl ClipMlp {
    fn new(cfg: &ClipConfig, vb: &VarBuilder) -> Result<Self> {
        Ok(Self {
            fc1: QLinear::linear_detect(cfg.embed_dim, cfg.intermediate_size, vb, "fc1", true)?,
            fc2: QLinear::linear_detect(cfg.intermediate_size, cfg.embed_dim, vb, "fc2", true)?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.fc1.forward(xs)?;
        // QuickGELU: x * sigmoid(1.702 x).
        let xs = (&xs * candle_gen::candle_nn::ops::sigmoid(&(&xs * 1.702f64)?)?)?;
        self.fc2.forward(&xs)
    }
}

struct ClipLayer {
    self_attn: ClipAttention,
    layer_norm1: LayerNorm,
    mlp: ClipMlp,
    layer_norm2: LayerNorm,
}

impl ClipLayer {
    fn new(cfg: &ClipConfig, vb: &VarBuilder) -> Result<Self> {
        Ok(Self {
            self_attn: ClipAttention::new(cfg, &vb.pp("self_attn"))?,
            layer_norm1: clip_layer_norm(cfg.embed_dim, vb, "layer_norm1")?,
            mlp: ClipMlp::new(cfg, &vb.pp("mlp"))?,
            layer_norm2: clip_layer_norm(cfg.embed_dim, vb, "layer_norm2")?,
        })
    }

    fn forward(&self, xs: &Tensor, causal_mask: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let xs = self.layer_norm1.forward(xs)?;
        let xs = self.self_attn.forward(&xs, causal_mask)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let ys = self.layer_norm2.forward(&xs)?;
        let ys = self.mlp.forward(&ys)?;
        ys + residual
    }
}

/// The packed CLIP-L text transformer — token+position embed → 12 pre-norm layers → final LayerNorm,
/// pooled at the argmax-EOT position (FLUX's `vec`/`y` conditioning).
pub struct PackedClipText {
    embeddings: ClipEmbeddings,
    layers: Vec<ClipLayer>,
    final_layer_norm: LayerNorm,
}

impl PackedClipText {
    /// Load from `vb` rooted at the CLIP `text_model.` prefix (the pipeline passes `vb.pp("text_model")`).
    pub fn new(cfg: &ClipConfig, vb: VarBuilder) -> Result<Self> {
        let embeddings = ClipEmbeddings::new(cfg, &vb)?;
        let enc = vb.pp("encoder").pp("layers");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(ClipLayer::new(cfg, &enc.pp(i))?);
        }
        let final_layer_norm = clip_layer_norm(cfg.embed_dim, &vb, "final_layer_norm")?;
        Ok(Self {
            embeddings,
            layers,
            final_layer_norm,
        })
    }

    fn causal_mask(bsz: usize, seq_len: usize, device: &Device) -> Result<Tensor> {
        let mask: Vec<f32> = (0..seq_len)
            .flat_map(|i| (0..seq_len).map(move |j| if j > i { f32::MIN } else { 0. }))
            .collect();
        Tensor::from_slice(&mask, (seq_len, seq_len), device)?
            .broadcast_as((bsz, 1, seq_len, seq_len))
    }

    /// The pooled `[B, 768]` vector — the causal-attended stack read at each row's argmax (EOT) token,
    /// exactly like the stock `ClipTextTransformer` `Module::forward`.
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (bsz, seq_len) = input_ids.dims2()?;
        let mut xs = self.embeddings.forward(input_ids)?;
        let mask = Self::causal_mask(bsz, seq_len, input_ids.device())?;
        for layer in &self.layers {
            xs = layer.forward(&xs, &mask)?;
        }
        let output = self.final_layer_norm.forward(&xs)?;
        let seq_max = input_ids.argmax(D::Minus1)?.to_dtype(DType::I64)?;
        let mut indices = Vec::new();
        for (b, &s) in seq_max.to_vec1::<i64>()?.iter().enumerate() {
            indices.push(output.i((b, s as usize))?.unsqueeze(0)?);
        }
        Tensor::cat(&indices, 0)
    }
}

// ============================================================================================
// T5-XXL encoder (google/t5-v1_1-xxl layout: `shared`, `encoder.block.*`, `encoder.final_layer_norm`).
// ============================================================================================

/// The T5-XXL config subset the FLUX encoder path needs (encoder-only, gated-GELU).
pub struct T5Config {
    pub vocab_size: usize,
    pub d_model: usize,
    pub d_kv: usize,
    pub d_ff: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub relative_attention_num_buckets: usize,
    pub relative_attention_max_distance: usize,
    pub layer_norm_epsilon: f64,
}

impl T5Config {
    pub fn xxl() -> Self {
        Self {
            vocab_size: 32128,
            d_model: 4096,
            d_kv: 64,
            d_ff: 10240,
            num_layers: 24,
            num_heads: 64,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            layer_norm_epsilon: 1e-6,
        }
    }
}

/// T5 RMSNorm (the "T5 layer norm": mean-of-squares in f32, scale by the dense `weight`).
struct T5LayerNorm {
    weight: Tensor,
    eps: f64,
}

impl T5LayerNorm {
    fn new(dim: usize, eps: f64, vb: &VarBuilder, prefix: &str) -> Result<Self> {
        Ok(Self {
            weight: vb.get(dim, &format!("{prefix}.weight"))?,
            eps,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dtype = xs.dtype();
        let xs_f32 = xs.to_dtype(DType::F32)?;
        let variance = xs_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let xs = xs_f32.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        xs.to_dtype(dtype)?
            .broadcast_mul(&self.weight.to_dtype(dtype)?)
    }
}

/// T5 gated-GELU FF: `wo( gelu(wi_0(x)) * wi_1(x) )`, all bias-less.
struct T5DenseGatedActDense {
    wi_0: QLinear,
    wi_1: QLinear,
    wo: QLinear,
}

impl T5DenseGatedActDense {
    fn new(cfg: &T5Config, vb: &VarBuilder) -> Result<Self> {
        Ok(Self {
            wi_0: QLinear::linear_detect(cfg.d_model, cfg.d_ff, vb, "wi_0", false)?,
            wi_1: QLinear::linear_detect(cfg.d_model, cfg.d_ff, vb, "wi_1", false)?,
            wo: QLinear::linear_detect(cfg.d_ff, cfg.d_model, vb, "wo", false)?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // T5 v1.1 uses the "new"/tanh GELU (NewGelu) — candle `gelu` is the tanh approximation.
        let hidden_gelu = self.wi_0.forward(xs)?.gelu()?;
        let hidden_linear = self.wi_1.forward(xs)?;
        self.wo.forward(&hidden_gelu.broadcast_mul(&hidden_linear)?)
    }
}

struct T5LayerFF {
    dense: T5DenseGatedActDense,
    layer_norm: T5LayerNorm,
}

impl T5LayerFF {
    fn new(cfg: &T5Config, vb: &VarBuilder) -> Result<Self> {
        Ok(Self {
            dense: T5DenseGatedActDense::new(cfg, &vb.pp("DenseReluDense"))?,
            layer_norm: T5LayerNorm::new(cfg.d_model, cfg.layer_norm_epsilon, vb, "layer_norm")?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let ys = self.layer_norm.forward(xs)?;
        let ys = self.dense.forward(&ys)?;
        xs + ys
    }
}

struct T5Attention {
    q: QLinear,
    k: QLinear,
    v: QLinear,
    o: QLinear,
    relative_attention_bias: Option<QEmbedding>,
    n_heads: usize,
    d_kv: usize,
    inner_dim: usize,
    num_buckets: usize,
    max_distance: usize,
}

impl T5Attention {
    fn new(has_rel_bias: bool, cfg: &T5Config, vb: &VarBuilder) -> Result<Self> {
        let inner_dim = cfg.num_heads * cfg.d_kv;
        Ok(Self {
            q: QLinear::linear_detect(cfg.d_model, inner_dim, vb, "q", false)?,
            k: QLinear::linear_detect(cfg.d_model, inner_dim, vb, "k", false)?,
            v: QLinear::linear_detect(cfg.d_model, inner_dim, vb, "v", false)?,
            o: QLinear::linear_detect(inner_dim, cfg.d_model, vb, "o", false)?,
            relative_attention_bias: if has_rel_bias {
                Some(QEmbedding::detect(
                    vb,
                    "relative_attention_bias",
                    cfg.relative_attention_num_buckets,
                    cfg.num_heads,
                )?)
            } else {
                None
            },
            n_heads: cfg.num_heads,
            d_kv: cfg.d_kv,
            inner_dim,
            num_buckets: cfg.relative_attention_num_buckets,
            max_distance: cfg.relative_attention_max_distance,
        })
    }

    /// The bidirectional relative-position-bias buckets — copied verbatim from the stock T5 encoder
    /// (`use_cache = false`, encoder path: `q_start = 0`, `q_end = kv_len`).
    fn relative_buckets(&self, q_len: usize, kv_len: usize, device: &Device) -> Result<Tensor> {
        let num_buckets = self.num_buckets as u32 / 2;
        let max_exact = num_buckets / 2;
        let rel: Vec<Vec<u32>> = (0..q_len as u32)
            .map(|i| {
                (0..kv_len as u32)
                    .map(|j| {
                        if i < j {
                            if j - i < max_exact {
                                j - i + num_buckets
                            } else {
                                let b = f32::log(
                                    (j - i) as f32 / max_exact as f32,
                                    self.max_distance as f32 / max_exact as f32,
                                ) * (num_buckets - max_exact) as f32;
                                u32::min(
                                    max_exact + num_buckets + b as u32,
                                    self.num_buckets as u32 - 1,
                                )
                            }
                        } else if i - j < max_exact {
                            i - j
                        } else {
                            let b = f32::log(
                                (i - j) as f32 / max_exact as f32,
                                self.max_distance as f32 / max_exact as f32,
                            ) * (num_buckets - max_exact) as f32;
                            u32::min(max_exact + b as u32, num_buckets - 1)
                        }
                    })
                    .collect()
            })
            .collect();
        Tensor::new(rel, device)
    }

    /// Encoder self-attention. `position_bias` is threaded from block 0 (only block 0 owns
    /// `relative_attention_bias`), matching the stock encoder.
    fn forward(&self, xs: &Tensor, position_bias: Option<&Tensor>) -> Result<(Tensor, Tensor)> {
        let (b_sz, q_len) = (xs.dim(0)?, xs.dim(1)?);
        let kv_len = q_len;
        let q = self.q.forward(xs)?;
        let k = self.k.forward(xs)?;
        let v = self.v.forward(xs)?;
        let to_heads = |t: Tensor| -> Result<Tensor> {
            t.reshape((b_sz, q_len, self.n_heads, self.d_kv))?
                .transpose(1, 2)?
                .contiguous()
        };
        let q = to_heads(q)?;
        let k = to_heads(k)?;
        let v = to_heads(v)?;
        let scores = q.matmul(&k.t()?)?;
        let position_bias = match position_bias {
            Some(pb) => pb.clone(),
            None => {
                let buckets = self.relative_buckets(q_len, kv_len, q.device())?;
                self.relative_attention_bias
                    .as_ref()
                    .expect("block 0 owns the relative_attention_bias")
                    .forward(&buckets)?
                    .permute((2, 0, 1))?
                    .unsqueeze(0)?
                    .to_dtype(scores.dtype())?
            }
        };
        let scores = scores.broadcast_add(&position_bias)?;
        let attn = candle_gen::candle_nn::ops::softmax_last_dim(&scores)?;
        let out = attn.matmul(&v)?;
        let out = out
            .transpose(1, 2)?
            .reshape((b_sz, q_len, self.inner_dim))?;
        Ok((self.o.forward(&out)?, position_bias))
    }
}

struct T5LayerSelfAttention {
    self_attention: T5Attention,
    layer_norm: T5LayerNorm,
}

impl T5LayerSelfAttention {
    fn new(has_rel_bias: bool, cfg: &T5Config, vb: &VarBuilder) -> Result<Self> {
        Ok(Self {
            self_attention: T5Attention::new(has_rel_bias, cfg, &vb.pp("SelfAttention"))?,
            layer_norm: T5LayerNorm::new(cfg.d_model, cfg.layer_norm_epsilon, vb, "layer_norm")?,
        })
    }

    fn forward(&self, xs: &Tensor, position_bias: Option<&Tensor>) -> Result<(Tensor, Tensor)> {
        let normed = self.layer_norm.forward(xs)?;
        let (ys, pb) = self.self_attention.forward(&normed, position_bias)?;
        Ok(((xs + ys)?, pb))
    }
}

struct T5Block {
    self_attn: T5LayerSelfAttention,
    ff: T5LayerFF,
}

impl T5Block {
    fn new(has_rel_bias: bool, cfg: &T5Config, vb: &VarBuilder) -> Result<Self> {
        let vb = vb.pp("layer");
        Ok(Self {
            self_attn: T5LayerSelfAttention::new(has_rel_bias, cfg, &vb.pp("0"))?,
            ff: T5LayerFF::new(cfg, &vb.pp("1"))?,
        })
    }

    fn forward(&self, xs: &Tensor, position_bias: Option<&Tensor>) -> Result<(Tensor, Tensor)> {
        let (xs, pb) = self.self_attn.forward(xs, position_bias)?;
        Ok((self.ff.forward(&xs)?, pb))
    }
}

/// The packed T5-XXL encoder — `shared` embed → 24 encoder blocks (relative bias on block 0) → final
/// RMSNorm. Returns the `[B, L, 4096]` sequence FLUX consumes as `txt`.
pub struct PackedT5Encoder {
    shared: QEmbedding,
    blocks: Vec<T5Block>,
    final_layer_norm: T5LayerNorm,
}

impl PackedT5Encoder {
    /// Load from `vb` rooted at the T5 checkpoint root (`shared`, `encoder.block.*`,
    /// `encoder.final_layer_norm`).
    pub fn new(cfg: &T5Config, vb: VarBuilder) -> Result<Self> {
        let shared = QEmbedding::detect(&vb, "shared", cfg.vocab_size, cfg.d_model)?;
        let enc = vb.pp("encoder");
        let block_vb = enc.pp("block");
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            blocks.push(T5Block::new(i == 0, cfg, &block_vb.pp(i))?);
        }
        let final_layer_norm = T5LayerNorm::new(
            cfg.d_model,
            cfg.layer_norm_epsilon,
            &enc,
            "final_layer_norm",
        )?;
        Ok(Self {
            shared,
            blocks,
            final_layer_norm,
        })
    }

    /// Encode `input_ids` `[B, L]` → `[B, L, 4096]`. `out_dtype` casts the token embedding to the
    /// compute dtype (bf16) before the encoder, as the stock `forward_dt` does.
    pub fn forward(&self, input_ids: &Tensor, out_dtype: DType) -> Result<Tensor> {
        let mut hidden = self.shared.forward(input_ids)?.to_dtype(out_dtype)?;
        let mut position_bias: Option<Tensor> = None;
        for block in &self.blocks {
            let (h, pb) = block.forward(&hidden, position_bias.as_ref())?;
            hidden = h;
            position_bias = Some(pb);
        }
        self.final_layer_norm.forward(&hidden)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::{DType, Device};
    use candle_gen::candle_nn::VarMap;

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        (a.to_dtype(DType::F32)
            .unwrap()
            .sub(&b.to_dtype(DType::F32).unwrap())
            .unwrap())
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
    }

    /// **Vendored CLIP ≡ stock candle-transformers CLIP (dense, 1e-4).** Both models load from the SAME
    /// random `VarMap` (identical `text_model.*` diffusers keys), so a byte-for-byte match of their
    /// pooled `[B, 768]` outputs proves the vendored packed-detect CLIP is a faithful dense port — the
    /// dense path is unchanged, and (since the packed forward = dequant-then-dense-matmul) a packed load
    /// of the same weights would track it too. A tiny 2-layer config keeps it GPU-free and fast.
    #[test]
    fn vendored_clip_matches_stock_dense() -> Result<()> {
        use candle_transformers::models::clip::text_model::{
            Activation, ClipTextConfig, ClipTextTransformer,
        };
        let dev = Device::Cpu;
        let cfg = ClipConfig {
            vocab_size: 64,
            embed_dim: 32,
            intermediate_size: 64,
            max_position_embeddings: 16,
            num_hidden_layers: 2,
            num_attention_heads: 4,
        };
        let stock_cfg = ClipTextConfig {
            vocab_size: cfg.vocab_size,
            embed_dim: cfg.embed_dim,
            projection_dim: cfg.embed_dim,
            activation: Activation::QuickGelu,
            intermediate_size: cfg.intermediate_size,
            max_position_embeddings: cfg.max_position_embeddings,
            pad_with: None,
            num_hidden_layers: cfg.num_hidden_layers,
            num_attention_heads: cfg.num_attention_heads,
        };

        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        // Build the vendored model (forces the `text_model.*` keys to be created in the VarMap).
        let vendored = PackedClipText::new(&cfg, vb.pp("text_model"))?;
        // Build the stock model from the SAME VarMap → identical weights.
        let vb2 = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let stock = ClipTextTransformer::new(vb2.pp("text_model"), &stock_cfg)?;

        let ids = Tensor::from_vec(vec![3u32, 8, 15, 2, 0, 0], (1, 6), &dev)?;
        let v = vendored.forward(&ids)?;
        let s = stock.forward(&ids)?;
        assert_eq!(v.dims(), s.dims());
        let d = max_abs_diff(&v, &s);
        assert!(d < 1e-4, "vendored CLIP vs stock max|Δ| = {d}");
        Ok(())
    }

    /// **Vendored T5 encoder ≡ stock candle-transformers T5 encoder (dense, 1e-4).** Same-VarMap parity
    /// on the encoder path (`shared`, `encoder.block.*`, `encoder.final_layer_norm`, gated-GELU FF,
    /// block-0 relative bias). A tiny 2-block config keeps it GPU-free.
    #[test]
    fn vendored_t5_matches_stock_dense() -> Result<()> {
        use candle_transformers::models::t5::{
            ActivationWithOptionalGating, Config as StockT5Config, T5EncoderModel,
        };
        let dev = Device::Cpu;
        let cfg = T5Config {
            vocab_size: 64,
            d_model: 32,
            d_kv: 8,
            d_ff: 64,
            num_layers: 2,
            num_heads: 4,
            relative_attention_num_buckets: 8,
            relative_attention_max_distance: 128,
            layer_norm_epsilon: 1e-6,
        };
        let stock_cfg = StockT5Config {
            vocab_size: cfg.vocab_size,
            d_model: cfg.d_model,
            d_kv: cfg.d_kv,
            d_ff: cfg.d_ff,
            num_layers: cfg.num_layers,
            num_decoder_layers: None,
            num_heads: cfg.num_heads,
            relative_attention_num_buckets: cfg.relative_attention_num_buckets,
            relative_attention_max_distance: cfg.relative_attention_max_distance,
            dropout_rate: 0.0,
            layer_norm_epsilon: cfg.layer_norm_epsilon,
            initializer_factor: 1.0,
            feed_forward_proj: ActivationWithOptionalGating {
                gated: true,
                activation: candle_gen::candle_nn::Activation::NewGelu,
            },
            tie_word_embeddings: false,
            is_decoder: false,
            is_encoder_decoder: true,
            use_cache: false,
            pad_token_id: 0,
            eos_token_id: 1,
            decoder_start_token_id: Some(0),
        };

        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let vendored = PackedT5Encoder::new(&cfg, vb)?;
        let vb2 = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let mut stock = T5EncoderModel::load(vb2, &stock_cfg)?;

        let ids = Tensor::from_vec(vec![3u32, 8, 15, 2, 7, 1], (1, 6), &dev)?;
        let v = vendored.forward(&ids, DType::F32)?;
        let s = stock.forward(&ids)?;
        assert_eq!(v.dims(), s.dims());
        let d = max_abs_diff(&v, &s);
        assert!(d < 1e-4, "vendored T5 encoder vs stock max|Δ| = {d}");
        Ok(())
    }
}
