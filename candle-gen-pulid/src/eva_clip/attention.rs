//! EVA sub-LN `Attention`. Candle port of `eva_vit_model.py Attention(subln=True, rope=…)`.
//!
//! subln layout: separate `q_proj`/`k_proj`/`v_proj` (Linear, bias=False) plus standalone `q_bias` and
//! `v_bias` params (k has **no** bias), an `inner_attn_ln` (LayerNorm over all-head-dim) before `proj`.
//! RoPE is applied to the **patch** tokens of q/k only (the CLS token at index 0 is left unrotated).
//! Attention: scale q by `head_dim**-0.5`, softmax in f32 (the reference's explicit path; xformers
//! absent ⇒ `xattn=False`).

use candle_core::{Tensor, D};
use candle_nn::ops::softmax_last_dim;
use candle_nn::{LayerNorm, Linear, Module};

use candle_gen::weights::Weights;
use candle_gen::Result as GenResult;

use crate::eva_clip::rope::VisionRope;
use crate::eva_clip::{join, layer_norm};

pub struct Attention {
    q_proj: Linear, // q_proj.weight + standalone q_bias
    k_proj: Linear, // k_proj.weight, NO bias
    v_proj: Linear, // v_proj.weight + standalone v_bias
    inner_ln: LayerNorm,
    proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl Attention {
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        num_heads: usize,
        head_dim: usize,
    ) -> GenResult<Self> {
        let q_proj = Linear::new(
            w.require(&join(prefix, "q_proj.weight"))?,
            Some(w.require(&join(prefix, "q_bias"))?),
        );
        let k_proj = Linear::new(w.require(&join(prefix, "k_proj.weight"))?, None);
        let v_proj = Linear::new(
            w.require(&join(prefix, "v_proj.weight"))?,
            Some(w.require(&join(prefix, "v_bias"))?),
        );
        let proj = Linear::new(
            w.require(&join(prefix, "proj.weight"))?,
            Some(w.require(&join(prefix, "proj.bias"))?),
        );
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            inner_ln: layer_norm(w, &join(prefix, "inner_attn_ln"))?,
            proj,
            num_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
        })
    }

    /// `x`: `[B, N, C]` (N = 1 CLS + grid² patches). `rope` is the shared block-invariant table.
    pub fn forward(&self, x: &Tensor, rope: &VisionRope) -> candle_core::Result<Tensor> {
        let (b, n, _c) = x.dims3()?;
        let (h, hd) = (self.num_heads, self.head_dim);

        // subln projections: q/v biased, k unbiased.
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // [B, N, C] → [B, heads, N, hd]
        let to_heads = |t: &Tensor| -> candle_core::Result<Tensor> {
            t.reshape((b, n, h, hd))?.transpose(1, 2)?.contiguous()
        };
        let q = self.rope_patch_tokens(&to_heads(&q)?, rope)?;
        let k = self.rope_patch_tokens(&to_heads(&k)?, rope)?;
        let v = to_heads(&v)?;

        // SDPA (softmax in f32; the head dim is small, the tower is f32 anyway).
        let scores = (q.matmul(&k.transpose(D::Minus1, D::Minus2)?.contiguous()?)? * self.scale)?;
        let probs = softmax_last_dim(&scores)?;
        let attn = probs.matmul(&v)?;

        let out = attn
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, n, h * hd))?;
        let out = self.inner_ln.forward(&out)?;
        self.proj.forward(&out)
    }

    /// Apply RoPE to `x[:, :, 1:, :]` (patch tokens) only; the CLS token at index 0 is untouched.
    fn rope_patch_tokens(&self, x: &Tensor, rope: &VisionRope) -> candle_core::Result<Tensor> {
        let n = x.dim(2)?;
        let cls = x.narrow(2, 0, 1)?;
        let pat = rope.apply(&x.narrow(2, 1, n - 1)?)?;
        Tensor::cat(&[&cls, &pat], 2)
    }
}
