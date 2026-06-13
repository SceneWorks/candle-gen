//! Lens joint (dual-stream) attention (`LensJointAttention`). **Fused** `img_qkv`/`txt_qkv`
//! projections (biased) split into per-stream q/k/v, per-head q/k RMSNorm, interleaved-complex RoPE on
//! both streams, then attention over the **`[img, txt]`**-concatenated sequence (image tokens first,
//! matching the Lens `_build_joint_attention_mask`), split back and projected (`to_out.0` for image,
//! `to_add_out` for text).

use candle_gen::candle_core::{Result, Tensor};
use candle_gen::candle_nn::{
    linear, ops::softmax_last_dim, rms_norm, Linear, Module, RmsNorm, VarBuilder,
};

use super::rope::apply_rope;

/// QK-RMSNorm epsilon (the Lens block builds `LensJointAttention(eps=1e-6)`).
const RMS_EPS: f64 = 1e-6;

pub struct LensJointAttention {
    img_qkv: Linear,
    txt_qkv: Linear,
    to_out: Linear,
    to_add_out: Linear,
    norm_q: RmsNorm,
    norm_k: RmsNorm,
    norm_added_q: RmsNorm,
    norm_added_k: RmsNorm,
    heads: usize,
    head_dim: usize,
}

impl LensJointAttention {
    /// `inner = heads · head_dim`. The fused QKV are `[3·inner, inner]` (biased); `to_out`/`to_add_out`
    /// are `[inner, inner]` (biased). QK-norm weights are `[head_dim]`.
    pub fn new(inner: usize, heads: usize, head_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            img_qkv: linear(inner, 3 * inner, vb.pp("img_qkv"))?,
            txt_qkv: linear(inner, 3 * inner, vb.pp("txt_qkv"))?,
            to_out: linear(inner, inner, vb.pp("to_out").pp("0"))?,
            to_add_out: linear(inner, inner, vb.pp("to_add_out"))?,
            norm_q: rms_norm(head_dim, RMS_EPS, vb.pp("norm_q"))?,
            norm_k: rms_norm(head_dim, RMS_EPS, vb.pp("norm_k"))?,
            norm_added_q: rms_norm(head_dim, RMS_EPS, vb.pp("norm_added_q"))?,
            norm_added_k: rms_norm(head_dim, RMS_EPS, vb.pp("norm_added_k"))?,
            heads,
            head_dim,
        })
    }

    /// Fused QKV → `q`/`k`/`v` each `[B, seq, heads, head_dim]`.
    fn qkv(&self, lin: &Linear, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (b, s, _) = x.dims3()?;
        let t = lin.forward(x)?.reshape((b, s, 3, self.heads, self.head_dim))?;
        let q = t.narrow(2, 0, 1)?.reshape((b, s, self.heads, self.head_dim))?;
        let k = t.narrow(2, 1, 1)?.reshape((b, s, self.heads, self.head_dim))?;
        let v = t.narrow(2, 2, 1)?.reshape((b, s, self.heads, self.head_dim))?;
        Ok((q, k, v))
    }

    /// Optional per-head q/k RMSNorm over `head_dim` (applied at `[B, seq, heads, head_dim]`), then
    /// `→ [B, heads, seq, head_dim]` contiguous for attention.
    fn to_heads(x: &Tensor, norm: Option<&RmsNorm>) -> Result<Tensor> {
        let x = match norm {
            Some(n) => n.forward(x)?,
            None => x.clone(),
        };
        x.transpose(1, 2)?.contiguous()
    }

    /// `img`/`txt`: `[B, seq, inner]`; rope tables `[seq, head_dim/2]`; `mask`: optional additive
    /// `[B, 1, 1, img+txt]`. Returns `(img_attn, txt_attn)`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        img: &Tensor,
        txt: &Tensor,
        img_cos: &Tensor,
        img_sin: &Tensor,
        txt_cos: &Tensor,
        txt_sin: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let (h, hd) = (self.heads, self.head_dim);
        let img_seq = img.dim(1)?;

        let (img_q, img_k, img_v) = self.qkv(&self.img_qkv, img)?;
        let (txt_q, txt_k, txt_v) = self.qkv(&self.txt_qkv, txt)?;

        // QK RMSNorm + per-stream interleaved RoPE; v has no norm/rope.
        let img_q = apply_rope(&Self::to_heads(&img_q, Some(&self.norm_q))?, img_cos, img_sin)?;
        let img_k = apply_rope(&Self::to_heads(&img_k, Some(&self.norm_k))?, img_cos, img_sin)?;
        let img_v = Self::to_heads(&img_v, None)?;
        let txt_q = apply_rope(
            &Self::to_heads(&txt_q, Some(&self.norm_added_q))?,
            txt_cos,
            txt_sin,
        )?;
        let txt_k = apply_rope(
            &Self::to_heads(&txt_k, Some(&self.norm_added_k))?,
            txt_cos,
            txt_sin,
        )?;
        let txt_v = Self::to_heads(&txt_v, None)?;

        // Joint [img, txt] over the sequence axis (image first), all [B, heads, seq, head_dim].
        let q = Tensor::cat(&[&img_q, &txt_q], 2)?;
        let k = Tensor::cat(&[&img_k, &txt_k], 2)?;
        let v = Tensor::cat(&[&img_v, &txt_v], 2)?;

        let scale = (hd as f64).powf(-0.5);
        let scores = (q.contiguous()?.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
        let scores = match mask {
            Some(m) => scores.broadcast_add(m)?,
            None => scores,
        };
        let probs = softmax_last_dim(&scores)?;
        let o = probs.matmul(&v.contiguous()?)?; // [B, heads, joint, head_dim]
        let (b, _, joint, _) = o.dims4()?;
        let o = o.transpose(1, 2)?.reshape((b, joint, h * hd))?;

        // Split back at the image/text boundary (image first).
        let img_o = o.narrow(1, 0, img_seq)?.contiguous()?;
        let txt_o = o.narrow(1, img_seq, joint - img_seq)?.contiguous()?;
        Ok((self.to_out.forward(&img_o)?, self.to_add_out.forward(&txt_o)?))
    }
}
