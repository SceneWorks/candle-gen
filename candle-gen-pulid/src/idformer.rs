//! IDFormer (sc-5492) — the PuLID perceiver-resampler that fuses the ArcFace embedding + the EVA visual
//! features into the 32-token `id_embedding` the FLUX cross-attn (the [`crate::ca`] modules) injects.
//! Candle port of `mlx-gen-pulid`'s `idformer.rs` (= `pulid/encoders_transformer.py IDFormer`).
//!
//! Structure (dim=1024, depth=10, heads=16, dim_head=64, 5 id tokens, 32 queries, out 2048):
//!   * `latents` [1,32,1024] learned queries; `proj_out` [1024,2048] learned param (raw matmul).
//!   * `id_embedding_mapping`: 1280→1024→(LN,LeakyReLU)→1024→(LN,LeakyReLU)→1024×5  (the 5 id tokens).
//!   * `mapping_0..4`: 1024→1024→(LN,LeakyReLU)→1024→(LN,LeakyReLU)→1024  (projects each EVA scale).
//!   * 10 × (PerceiverAttention + FeedForward), grouped 5 scales × 2 layers.
//!
//! All LayerNorms are `nn.LayerNorm` default **ε=1e-5** (distinct from the EVA tower's 1e-6). The
//! perceiver attention uses SDPA with `scale = dim_head^-0.5` and softmax in f32.

use candle_core::{Tensor, D};
use candle_nn::ops::softmax_last_dim;
use candle_nn::{LayerNorm, Linear, Module};

use candle_gen::weights::Weights;
use candle_gen::Result as GenResult;
use candle_gen::{CandleError, Result};

/// nn.LayerNorm default epsilon (the IDFormer/PuLID modules; NOT the EVA 1e-6).
const EPS: f64 = 1e-5;
/// nn.LeakyReLU default negative slope.
const LEAKY: f64 = 0.01;

fn join(p: &str, leaf: &str) -> String {
    format!("{p}.{leaf}")
}

/// Build a biased [`LayerNorm`] (ε=1e-5) from `{prefix}.weight` / `{prefix}.bias`.
fn layer_norm(w: &Weights, prefix: &str) -> GenResult<LayerNorm> {
    Ok(LayerNorm::new(
        w.require(&format!("{prefix}.weight"))?,
        w.require(&format!("{prefix}.bias"))?,
        EPS,
    ))
}

/// `leaky_relu(x) = max(x, slope·x)`.
fn leaky_relu(x: &Tensor) -> candle_core::Result<Tensor> {
    x.maximum(&(x * LEAKY)?)
}

/// The shared `Linear→LN→LeakyReLU→Linear→LN→LeakyReLU→Linear` mapping head (Sequential indices
/// 0,1,3,4,6 in the checkpoint). All three linears are biased.
struct MappingMlp {
    l0: Linear,
    ln1: LayerNorm,
    l3: Linear,
    ln4: LayerNorm,
    l6: Linear,
}

impl MappingMlp {
    fn from_weights(w: &Weights, p: &str) -> GenResult<Self> {
        let lin = |leaf: &str| -> GenResult<Linear> {
            Ok(Linear::new(
                w.require(&join(p, &format!("{leaf}.weight")))?,
                Some(w.require(&join(p, &format!("{leaf}.bias")))?),
            ))
        };
        Ok(Self {
            l0: lin("0")?,
            ln1: layer_norm(w, &join(p, "1"))?,
            l3: lin("3")?,
            ln4: layer_norm(w, &join(p, "4"))?,
            l6: lin("6")?,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let h = self.l0.forward(x)?;
        let h = self.ln1.forward(&h)?;
        let h = leaky_relu(&h)?;
        let h = self.l3.forward(&h)?;
        let h = self.ln4.forward(&h)?;
        let h = leaky_relu(&h)?;
        self.l6.forward(&h)
    }
}

/// PerceiverAttention: q from `latents`, k/v from `cat(ctx, latents)`. bias-free linears, ε=1e-5,
/// SDPA `scale = dim_head^-0.5`. (Shares its structure with [`crate::ca::PerceiverAttentionCA`]; the only
/// behavioral difference is the k/v source — here `cat(ctx, latents)`, there `x` alone.)
struct PerceiverAttention {
    norm1: LayerNorm,
    norm2: LayerNorm,
    to_q: Linear,
    to_kv: Linear,
    to_out: Linear,
    heads: usize,
    dim_head: usize,
}

impl PerceiverAttention {
    fn from_weights(w: &Weights, p: &str, heads: usize, dim_head: usize) -> GenResult<Self> {
        let lin = |leaf: &str| -> GenResult<Linear> {
            Ok(Linear::new(
                w.require(&join(p, &format!("{leaf}.weight")))?,
                None,
            ))
        };
        Ok(Self {
            norm1: layer_norm(w, &join(p, "norm1"))?,
            norm2: layer_norm(w, &join(p, "norm2"))?,
            to_q: lin("to_q")?,
            to_kv: lin("to_kv")?,
            to_out: lin("to_out")?,
            heads,
            dim_head,
        })
    }

    /// `ctx`: `[B, n_ctx, dim]` (image/id features); `latents`: `[B, n_lat, dim]` (queries).
    fn forward(&self, ctx: &Tensor, latents: &Tensor) -> candle_core::Result<Tensor> {
        let x = self.norm1.forward(ctx)?;
        let lat = self.norm2.forward(latents)?;
        let (b, n_lat, _dim) = lat.dims3()?;
        let (h, hd) = (self.heads, self.dim_head);

        let q = self.to_q.forward(&lat)?;
        let kv = self.to_kv.forward(&Tensor::cat(&[&x, &lat], 1)?)?;
        let n_kv = kv.dim(1)?;
        let parts = kv.chunk(2, D::Minus1)?; // [k | v] along the feature axis
        let to_heads = |t: &Tensor, n: usize| -> candle_core::Result<Tensor> {
            t.reshape((b, n, h, hd))?.transpose(1, 2)?.contiguous()
        };
        let q = to_heads(&q, n_lat)?;
        let k = to_heads(&parts[0], n_kv)?;
        let v = to_heads(&parts[1], n_kv)?;

        let scale = (hd as f64).powf(-0.5);
        let scores = (q.matmul(&k.transpose(D::Minus1, D::Minus2)?.contiguous()?)? * scale)?;
        let attn = softmax_last_dim(&scores)?.matmul(&v)?;
        let out = attn
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, n_lat, h * hd))?;
        self.to_out.forward(&out)
    }
}

/// FeedForward: `LN → Linear(no bias) → GELU(exact) → Linear(no bias)` (Sequential 0,1,3).
struct FeedForward {
    ln: LayerNorm,
    l1: Linear,
    l3: Linear,
}

impl FeedForward {
    fn from_weights(w: &Weights, p: &str) -> GenResult<Self> {
        Ok(Self {
            ln: layer_norm(w, &join(p, "0"))?,
            l1: Linear::new(w.require(&join(p, "1.weight"))?, None),
            l3: Linear::new(w.require(&join(p, "3.weight"))?, None),
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let h = self.ln.forward(x)?;
        let h = self.l1.forward(&h)?;
        let h = h.gelu_erf()?;
        self.l3.forward(&h)
    }
}

#[derive(Clone, Debug)]
pub struct IdFormerConfig {
    pub dim: usize,
    pub depth: usize,
    pub heads: usize,
    pub dim_head: usize,
    pub num_id_token: usize,
    pub num_queries: usize,
    pub output_dim: usize,
}

impl Default for IdFormerConfig {
    fn default() -> Self {
        Self {
            dim: 1024,
            depth: 10,
            heads: 16,
            dim_head: 64,
            num_id_token: 5,
            num_queries: 32,
            output_dim: 2048,
        }
    }
}

pub struct IdFormer {
    latents: Tensor,  // [1, num_queries, dim]
    proj_out: Tensor, // [dim, output_dim]
    id_embedding_mapping: MappingMlp,
    mapping: Vec<MappingMlp>, // 5
    layers: Vec<(PerceiverAttention, FeedForward)>,
    per_scale: usize, // depth / 5
    cfg: IdFormerConfig,
}

impl IdFormer {
    /// `prefix` is the top-level module name (`"pulid_encoder"`).
    pub fn from_weights(w: &Weights, prefix: &str, cfg: IdFormerConfig) -> GenResult<Self> {
        let p = |leaf: &str| join(prefix, leaf);
        let mapping = (0..5)
            .map(|i| MappingMlp::from_weights(w, &p(&format!("mapping_{i}"))))
            .collect::<GenResult<Vec<_>>>()?;
        let layers = (0..cfg.depth)
            .map(|i| {
                Ok((
                    PerceiverAttention::from_weights(
                        w,
                        &p(&format!("layers.{i}.0")),
                        cfg.heads,
                        cfg.dim_head,
                    )?,
                    FeedForward::from_weights(w, &p(&format!("layers.{i}.1")))?,
                ))
            })
            .collect::<GenResult<Vec<_>>>()?;
        Ok(Self {
            latents: w.require(&p("latents"))?,
            proj_out: w.require(&p("proj_out"))?,
            id_embedding_mapping: MappingMlp::from_weights(w, &p("id_embedding_mapping"))?,
            mapping,
            per_scale: cfg.depth / 5,
            layers,
            cfg,
        })
    }

    /// `id_cond`: `[B, 1280]` (cat of ArcFace 512 + id_cond_vit 768).
    /// `id_vit_hidden`: 5 × `[B, 577, 1024]` (the EVA hidden states).
    /// Returns `id_embedding` `[B, num_queries, output_dim]` (32×2048).
    pub fn forward(&self, id_cond: &Tensor, id_vit_hidden: &[Tensor]) -> Result<Tensor> {
        if id_vit_hidden.len() != 5 {
            return Err(CandleError::Msg(format!(
                "IDFormer expects 5 EVA hidden states, got {}",
                id_vit_hidden.len()
            )));
        }
        let b = id_cond.dim(0)?;
        let dim = self.cfg.dim;

        let mut latents = self
            .latents
            .broadcast_as((b, self.cfg.num_queries, dim))?
            .contiguous()?;
        // id tokens: 1280 → 1024*5 → [B, 5, 1024]
        let x =
            self.id_embedding_mapping
                .forward(id_cond)?
                .reshape((b, self.cfg.num_id_token, dim))?;
        latents = Tensor::cat(&[&latents, &x], 1)?; // [B, 37, 1024]

        for (i, (mapping, hidden_i)) in self.mapping.iter().zip(id_vit_hidden.iter()).enumerate() {
            let vit = mapping.forward(hidden_i)?;
            let ctx = Tensor::cat(&[&x, &vit], 1)?; // [B, 5+577, 1024]
            for l in (i * self.per_scale)..((i + 1) * self.per_scale) {
                let (attn, ff) = &self.layers[l];
                latents = (&latents + attn.forward(&ctx, &latents)?)?;
                latents = (&latents + ff.forward(&latents)?)?;
            }
        }

        // Take the 32 query tokens, project to output_dim via the raw param matmul.
        let q = latents.narrow(1, 0, self.cfg.num_queries)?; // [B, 32, 1024]
        let proj = self
            .proj_out
            .unsqueeze(0)?
            .broadcast_as((b, dim, self.cfg.output_dim))?
            .contiguous()?;
        Ok(q.contiguous()?.matmul(&proj)?)
    }
}
