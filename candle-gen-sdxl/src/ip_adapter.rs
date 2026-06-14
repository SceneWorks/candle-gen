//! IP-Adapter (image-prompt / identity conditioning) for SDXL — the candle twin of
//! `mlx-gen-sdxl::ip_adapter` (epic 5480, sc-5491; reused by sc-5488 IP-Adapter-Plus). Two pieces:
//!
//! 1. [`Resampler`] — the IP-Adapter "plus" image projection (`image_proj.*`): the *original Tencent*
//!    perceiver/Resampler (fused `to_kv`, bias-free projections — NOT the diffusers refactor) mapping
//!    image/identity features `[B, Nx, embed_dim]` → `[B, num_queries, output_dim]` tokens (16×2048 for
//!    SDXL). For **InstantID** the input is a single 512-d antelopev2 ArcFace embedding (`[B, 1, 512]`,
//!    [`ResamplerConfig::instantid_face`]); for IP-Adapter-Plus it's the ViT-H penultimate `[B,257,1280]`.
//! 2. [`load_ip_kv_pairs`] — the **decoupled cross-attention** K/V projections (`ip_adapter.{n}.to_k_ip
//!    /to_v_ip.weight`, bias-free `[hidden, cross_attention_dim]`) the UNet installs into its cross-attn
//!    in the diffusers attn-walk order (70 pairs for SDXL).
//!
//! Everything here is **all-sequence math** (Linear + LayerNorm + attention over `[B, N, D]`), so —
//! unlike the conv face/UNet models — there is NO NHWC↔NCHW transpose: the Tencent weight layout ports
//! 1:1 onto `candle_nn::Linear` ( `[out, in]` weights, `x @ Wᵀ` ).

use candle_core::{DType, Tensor, D};
use candle_nn::ops::softmax;
use candle_nn::{LayerNorm, Linear, Module};

use candle_gen::{CandleError, Result};

use crate::weights::Weights;

/// LayerNorm epsilon — the Tencent Resampler's `nn.LayerNorm` default (matches mlx-gen-sdxl).
const LN_EPS: f64 = 1e-5;

/// IP-Adapter "plus" Resampler config. Defaults are `ip-adapter-plus_sdxl_vit-h`.
#[derive(Clone, Debug)]
pub struct ResamplerConfig {
    /// Working width (`dim`); also the latent/query width. 1280 for plus-vit-h.
    pub dim: usize,
    /// Number of perceiver blocks (`depth`). 4 for plus-vit-h.
    pub depth: usize,
    /// Attention heads. 20 for plus-vit-h (head_dim 64).
    pub heads: usize,
    pub dim_head: usize,
    /// Output query tokens (`num_queries`). 16 for plus-vit-h.
    pub num_queries: usize,
    /// Input feature width feeding `proj_in` (ViT-H hidden 1280; ArcFace 512; ViT-L-336 1024).
    pub embed_dim: usize,
    /// Output token width (= UNet `cross_attention_dim`). 2048 for SDXL.
    pub output_dim: usize,
}

impl ResamplerConfig {
    /// `ip-adapter-plus_sdxl_vit-h` (ViT-H penultimate `[B,257,1280]`).
    pub fn plus_sdxl_vit_h() -> Self {
        Self {
            dim: 1280,
            depth: 4,
            heads: 20,
            dim_head: 64,
            num_queries: 16,
            embed_dim: 1280,
            output_dim: 2048,
        }
    }

    /// Kolors IP-Adapter-Plus (ViT-L/14-336 penultimate `[B,?,1024]`; working width 2048). Pinned by
    /// the on-disk shapes: `proj_in [2048,1024]`, `to_q [768,2048]` (inner=heads·dim_head=768, dim_head
    /// 64 ⇒ heads 12).
    pub fn kolors_plus() -> Self {
        Self {
            dim: 2048,
            depth: 4,
            heads: 12,
            dim_head: 64,
            num_queries: 16,
            embed_dim: 1024,
            output_dim: 2048,
        }
    }

    /// InstantID's face Resampler (`image_proj.*` of `InstantX/InstantID`). The vendored InstantID
    /// `Resampler` is the SAME Tencent layout as [`plus_sdxl_vit_h`](Self::plus_sdxl_vit_h); the only
    /// delta is the input feature width — a single 512-d antelopev2 ArcFace embedding (fed `[B, 1, 512]`)
    /// instead of the ViT-H penultimate. InstantID uses `apply_pos_emb=False` +
    /// `num_latents_mean_pooled=0`, so those branches are absent — exactly this [`Resampler`].
    pub fn instantid_face() -> Self {
        Self {
            dim: 1280,
            depth: 4,
            heads: 20,
            dim_head: 64,
            num_queries: 16,
            embed_dim: 512,
            output_dim: 2048,
        }
    }
}

/// `candle_nn::LayerNorm` from `{prefix}.weight` + `{prefix}.bias`.
fn layer_norm(w: &Weights, prefix: &str) -> Result<LayerNorm> {
    Ok(LayerNorm::new(
        w.require(&format!("{prefix}.weight"))?,
        w.require(&format!("{prefix}.bias"))?,
        LN_EPS,
    ))
}

/// `candle_nn::Linear` (`[out, in]` weight) from `{prefix}.weight` (+ `{prefix}.bias` when `bias`).
fn linear(w: &Weights, prefix: &str, bias: bool) -> Result<Linear> {
    let weight = w.require(&format!("{prefix}.weight"))?;
    let b = if bias {
        Some(w.require(&format!("{prefix}.bias"))?)
    } else {
        None
    };
    Ok(Linear::new(weight, b))
}

/// Multi-head scaled-dot-product attention over `[B, heads, Nq, dim_head]` queries against
/// `[B, heads, S, dim_head]` keys/values. Scores/softmax run in f32 then cast back (the production f16
/// path is identity-directional, not bit-exact — mirrors the vendored UNet's f32 softmax).
fn sdpa(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64) -> Result<Tensor> {
    let in_dtype = q.dtype();
    let q = q.to_dtype(DType::F32)?;
    let k = k.to_dtype(DType::F32)?;
    let v = v.to_dtype(DType::F32)?;
    let scores = (q.matmul(&k.transpose(D::Minus1, D::Minus2)?.contiguous()?)? * scale)?;
    let probs = softmax(&scores, D::Minus1)?;
    let o = probs.matmul(&v.contiguous()?)?;
    Ok(o.to_dtype(in_dtype)?)
}

/// PerceiverAttention block (`layers.{i}.0`): cross-attention from the learned `latents` (queries) to
/// `cat([image_features, latents])` (keys/values), with a fused `to_kv` projection.
struct PerceiverAttention {
    norm1: LayerNorm, // on the image features (x)
    norm2: LayerNorm, // on the latents
    to_q: Linear,     // bias-free, dim → inner
    to_kv: Linear,    // bias-free, dim → 2·inner (fused)
    to_out: Linear,   // bias-free, inner → dim
    heads: usize,
    dim_head: usize,
    scale: f64,
}

impl PerceiverAttention {
    fn from_weights(w: &Weights, prefix: &str, cfg: &ResamplerConfig) -> Result<Self> {
        Ok(Self {
            norm1: layer_norm(w, &format!("{prefix}.norm1"))?,
            norm2: layer_norm(w, &format!("{prefix}.norm2"))?,
            to_q: linear(w, &format!("{prefix}.to_q"), false)?,
            to_kv: linear(w, &format!("{prefix}.to_kv"), false)?,
            to_out: linear(w, &format!("{prefix}.to_out"), false)?,
            heads: cfg.heads,
            dim_head: cfg.dim_head,
            scale: (cfg.dim_head as f64).powf(-0.5),
        })
    }

    /// `[B, N, inner]` → `[B, heads, N, dim_head]`.
    fn to_heads(&self, a: &Tensor) -> Result<Tensor> {
        let (b, n, _) = a.dims3()?;
        Ok(a.reshape((b, n, self.heads, self.dim_head))?
            .transpose(1, 2)?
            .contiguous()?)
    }

    /// `x`: image features `[B, Nx, dim]`; `latents`: `[B, Nq, dim]`. Returns the `to_out` projection
    /// `[B, Nq, dim]` (the Resampler adds the residual outside).
    fn forward(&self, x: &Tensor, latents: &Tensor) -> Result<Tensor> {
        let x = self.norm1.forward(x)?;
        let latents = self.norm2.forward(latents)?;
        let (b, nq, _) = latents.dims3()?;
        let inner = self.heads * self.dim_head;

        let q = self.to_q.forward(&latents)?;
        let kv_input = Tensor::cat(&[&x, &latents], 1)?; // [B, Nx+Nq, dim]
        let kv = self.to_kv.forward(&kv_input)?; // [B, S, 2·inner]
        let k = kv.narrow(D::Minus1, 0, inner)?;
        let v = kv.narrow(D::Minus1, inner, inner)?;

        let q = self.to_heads(&q)?;
        let k = self.to_heads(&k)?;
        let v = self.to_heads(&v)?;
        let o = sdpa(&q, &k, &v, self.scale)?; // [B, heads, Nq, dim_head]
        let o = o.transpose(1, 2)?.contiguous()?.reshape((b, nq, inner))?;
        Ok(self.to_out.forward(&o)?)
    }
}

/// FeedForward block (`layers.{i}.1`): LayerNorm(`.0`) → Linear(`.1`, dim→4·dim) → GELU(erf) →
/// Linear(`.3`, 4·dim→dim), the two Linears bias-free. The Resampler adds the residual outside.
struct ResamplerFeedForward {
    ln: LayerNorm,
    fc1: Linear,
    fc2: Linear,
}

impl ResamplerFeedForward {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            ln: layer_norm(w, &format!("{prefix}.0"))?,
            fc1: linear(w, &format!("{prefix}.1"), false)?,
            fc2: linear(w, &format!("{prefix}.3"), false)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.ln.forward(x)?;
        let y = self.fc1.forward(&y)?;
        // GELU(erf) — the exact GELU (`gelu_exact` in mlx-gen), NOT candle's tanh-approx `gelu()`.
        let y = y.gelu_erf()?;
        Ok(self.fc2.forward(&y)?)
    }
}

/// The IP-Adapter "plus" image projection (`image_proj.*`): image/identity features →
/// `[B, num_queries, output_dim]` tokens.
pub struct Resampler {
    /// `[1, num_queries, dim]` learned query latents.
    latents: Tensor,
    proj_in: Linear,  // embed_dim → dim (+bias)
    proj_out: Linear, // dim → output_dim (+bias)
    norm_out: LayerNorm,
    layers: Vec<(PerceiverAttention, ResamplerFeedForward)>,
    dim: usize,
    num_queries: usize,
    output_dim: usize,
}

impl Resampler {
    /// The compute dtype (the learned latents' dtype).
    pub fn dtype(&self) -> DType {
        self.latents.dtype()
    }

    /// The output token width (= UNet `cross_attention_dim`, 2048 for SDXL).
    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    /// The query-token count (16 for every IP-Adapter Resampler).
    pub fn num_queries(&self) -> usize {
        self.num_queries
    }

    /// Load from the `image_proj` namespace of an IP-Adapter-plus checkpoint.
    pub fn from_weights(w: &Weights, prefix: &str, cfg: &ResamplerConfig) -> Result<Self> {
        let latents = w.require(&format!("{prefix}.latents"))?;
        let layers = (0..cfg.depth)
            .map(|i| -> Result<_> {
                let attn =
                    PerceiverAttention::from_weights(w, &format!("{prefix}.layers.{i}.0"), cfg)?;
                let ff = ResamplerFeedForward::from_weights(w, &format!("{prefix}.layers.{i}.1"))?;
                Ok((attn, ff))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            latents,
            proj_in: linear(w, &format!("{prefix}.proj_in"), true)?,
            proj_out: linear(w, &format!("{prefix}.proj_out"), true)?,
            norm_out: layer_norm(w, &format!("{prefix}.norm_out"))?,
            layers,
            dim: cfg.dim,
            num_queries: cfg.num_queries,
            output_dim: cfg.output_dim,
        })
    }

    /// `image_features`: `[B, Nx, embed_dim]` → image/identity tokens `[B, num_queries, output_dim]`.
    pub fn forward(&self, image_features: &Tensor) -> Result<Tensor> {
        let b = image_features.dim(0)?;
        if self.latents.dims() != [1, self.num_queries, self.dim] {
            return Err(CandleError::Msg(format!(
                "resampler latents shape {:?} != [1, {}, {}]",
                self.latents.dims(),
                self.num_queries,
                self.dim
            )));
        }
        let mut latents = self
            .latents
            .broadcast_as((b, self.num_queries, self.dim))?
            .contiguous()?;
        let x = self.proj_in.forward(image_features)?;
        for (attn, ff) in &self.layers {
            latents = (attn.forward(&x, &latents)? + &latents)?;
            latents = (ff.forward(&latents)? + &latents)?;
        }
        let out = self.proj_out.forward(&latents)?;
        Ok(self.norm_out.forward(&out)?)
    }
}

/// Load the decoupled cross-attention **K/V projection pairs** from an IP-Adapter checkpoint
/// (`ip_adapter.{n}.to_k_ip/to_v_ip.weight`, bias-free `[hidden, cross_attention_dim]`), in the
/// diffusers `ip_adapter.{n}` **numeric order** — which is the UNet cross-attention walk order the UNet
/// installs them in. 70 pairs for SDXL.
pub fn load_ip_kv_pairs(w: &Weights) -> Result<Vec<(Tensor, Tensor)>> {
    let mut idxs: Vec<u32> = w
        .keys()
        .filter_map(|k| {
            k.strip_prefix("ip_adapter.")
                .and_then(|r| r.strip_suffix(".to_k_ip.weight"))
                .and_then(|n| n.parse::<u32>().ok())
        })
        .collect();
    idxs.sort_unstable();
    if idxs.is_empty() {
        return Err(CandleError::Msg(
            "ip_adapter: no ip_adapter.{n}.to_k_ip.weight keys found".into(),
        ));
    }
    idxs.into_iter()
        .map(|n| {
            let k = w.require(&format!("ip_adapter.{n}.to_k_ip.weight"))?;
            let v = w.require(&format!("ip_adapter.{n}.to_v_ip.weight"))?;
            Ok((k, v))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use std::collections::HashMap;

    fn randn(shape: (usize, usize, usize), dev: &Device) -> Tensor {
        Tensor::randn(0f32, 1f32, shape, dev).unwrap()
    }
    fn randn1(n: usize, dev: &Device) -> Tensor {
        Tensor::randn(0f32, 1f32, (n,), dev).unwrap()
    }
    fn randn2(r: usize, c: usize, dev: &Device) -> Tensor {
        Tensor::randn(0f32, 1f32, (r, c), dev).unwrap()
    }

    /// The Resampler forward produces `[B, num_queries, output_dim]` finite tokens for a synthetic
    /// (tiny-config) weight set — exercising proj_in, the depth perceiver+FF blocks (fused to_kv split,
    /// multi-head SDPA, the two residuals), proj_out and norm_out. Numerical parity vs the real
    /// antelopev2/ip-adapter weights is the Phase-5 GPU validation; this pins the port's structure.
    #[test]
    fn resampler_forward_shape_and_finite() {
        let dev = Device::Cpu;
        let cfg = ResamplerConfig {
            dim: 8,
            depth: 2,
            heads: 2,
            dim_head: 4,
            num_queries: 4,
            embed_dim: 6,
            output_dim: 10,
        };
        let inner = cfg.heads * cfg.dim_head; // 8
        let p = "image_proj";
        let mut m: HashMap<String, Tensor> = HashMap::new();
        m.insert(
            format!("{p}.latents"),
            randn((1, cfg.num_queries, cfg.dim), &dev),
        );
        m.insert(
            format!("{p}.proj_in.weight"),
            randn2(cfg.dim, cfg.embed_dim, &dev),
        );
        m.insert(format!("{p}.proj_in.bias"), randn1(cfg.dim, &dev));
        m.insert(
            format!("{p}.proj_out.weight"),
            randn2(cfg.output_dim, cfg.dim, &dev),
        );
        m.insert(format!("{p}.proj_out.bias"), randn1(cfg.output_dim, &dev));
        m.insert(format!("{p}.norm_out.weight"), randn1(cfg.output_dim, &dev));
        m.insert(format!("{p}.norm_out.bias"), randn1(cfg.output_dim, &dev));
        for i in 0..cfg.depth {
            let a = format!("{p}.layers.{i}.0");
            m.insert(format!("{a}.norm1.weight"), randn1(cfg.dim, &dev));
            m.insert(format!("{a}.norm1.bias"), randn1(cfg.dim, &dev));
            m.insert(format!("{a}.norm2.weight"), randn1(cfg.dim, &dev));
            m.insert(format!("{a}.norm2.bias"), randn1(cfg.dim, &dev));
            m.insert(format!("{a}.to_q.weight"), randn2(inner, cfg.dim, &dev));
            m.insert(
                format!("{a}.to_kv.weight"),
                randn2(2 * inner, cfg.dim, &dev),
            );
            m.insert(format!("{a}.to_out.weight"), randn2(cfg.dim, inner, &dev));
            let f = format!("{p}.layers.{i}.1");
            m.insert(format!("{f}.0.weight"), randn1(cfg.dim, &dev));
            m.insert(format!("{f}.0.bias"), randn1(cfg.dim, &dev));
            m.insert(format!("{f}.1.weight"), randn2(4 * cfg.dim, cfg.dim, &dev));
            m.insert(format!("{f}.3.weight"), randn2(cfg.dim, 4 * cfg.dim, &dev));
        }
        let w = Weights::from_map(m);
        let r = Resampler::from_weights(&w, p, &cfg).unwrap();
        assert_eq!(r.output_dim(), 10);
        assert_eq!(r.num_queries(), 4);

        let feats = randn((2, 3, cfg.embed_dim), &dev); // [B=2, Nx=3, embed_dim]
        let out = r.forward(&feats).unwrap();
        assert_eq!(out.dims(), &[2, cfg.num_queries, cfg.output_dim]);
        let vals = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(
            vals.iter().all(|v| v.is_finite()),
            "resampler output not finite"
        );
    }

    /// `load_ip_kv_pairs` discovers the `ip_adapter.{n}` indices and returns the pairs in ascending
    /// numeric order (the diffusers attn-walk order the UNet installs them in) — not string order
    /// (where "10" would sort before "2").
    #[test]
    fn ip_kv_pairs_sorted_by_numeric_index() {
        let dev = Device::Cpu;
        let mut m: HashMap<String, Tensor> = HashMap::new();
        for n in [10u32, 2, 0] {
            m.insert(format!("ip_adapter.{n}.to_k_ip.weight"), randn2(4, 6, &dev));
            m.insert(format!("ip_adapter.{n}.to_v_ip.weight"), randn2(4, 6, &dev));
        }
        let w = Weights::from_map(m);
        let pairs = load_ip_kv_pairs(&w).unwrap();
        assert_eq!(pairs.len(), 3);
        for (k, v) in &pairs {
            assert_eq!(k.dims(), &[4, 6]);
            assert_eq!(v.dims(), &[4, 6]);
        }
    }

    /// An empty / wrong-namespace checkpoint errors loudly rather than returning zero pairs.
    #[test]
    fn ip_kv_pairs_errors_when_absent() {
        let w = Weights::from_map(HashMap::new());
        assert!(load_ip_kv_pairs(&w).is_err());
    }
}
