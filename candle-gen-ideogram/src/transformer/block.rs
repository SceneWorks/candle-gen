//! Ideogram 4 DiT block: attention + SwiGLU MLP with AdaLN "sandwich" norms (a pre-norm scaled by
//! `1+scale`, a post-norm gated by `tanh(gate)`), full segment-masked attention, per-head q/k
//! RMSNorm, and interleaved 3D MRoPE. Port of `Ideogram4Attention` / `Ideogram4MLP` /
//! `Ideogram4TransformerBlock`.

use candle_gen::candle_core::{Result, Tensor, D};
use candle_gen::candle_nn::ops::softmax_last_dim;

use super::rmsnorm;
use crate::loader::{linear_detect, Weights};
use crate::quant::QLinear;

/// Per-head q/k RMSNorm eps (upstream `Ideogram4Attention`, hardcoded 1e-5).
const ATTN_QK_EPS: f64 = 1e-5;

// ── Attention ────────────────────────────────────────────────────────────────────────────
pub struct Ideogram4Attention {
    qkv: QLinear,
    o: QLinear,
    norm_q: Tensor,
    norm_k: Tensor,
    num_heads: usize,
    head_dim: usize,
}

impl Ideogram4Attention {
    pub fn load(w: &Weights, prefix: &str, num_heads: usize, head_dim: usize) -> Result<Self> {
        Ok(Self {
            qkv: linear_detect(w, &format!("{prefix}.qkv"), false)?,
            o: linear_detect(w, &format!("{prefix}.o"), false)?,
            norm_q: w.get(&format!("{prefix}.norm_q.weight"))?,
            norm_k: w.get(&format!("{prefix}.norm_k.weight"))?,
            num_heads,
            head_dim,
        })
    }

    /// `x`: `[B, L, emb]`; `cos`/`sin`: `[B, L, head_dim]`; `mask`: optional additive `[B, 1, L, L]`
    /// (`None` skips the add — uniform-segment path, sc-8992).
    pub fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let (nh, hd) = (self.num_heads, self.head_dim);

        // qkv → [B, L, 3, H, hd] → q,k,v [B, L, H, hd]
        let qkv = self.qkv.forward(x)?.reshape((b, s, 3, nh, hd))?;
        let q = qkv.narrow(2, 0, 1)?.contiguous()?.reshape((b, s, nh, hd))?;
        let k = qkv.narrow(2, 1, 1)?.contiguous()?.reshape((b, s, nh, hd))?;
        let v = qkv.narrow(2, 2, 1)?.contiguous()?.reshape((b, s, nh, hd))?;

        // Per-head q/k RMSNorm over the head dim, before transpose + RoPE.
        let q = rmsnorm(&q, &self.norm_q, ATTN_QK_EPS)?;
        let k = rmsnorm(&k, &self.norm_k, ATTN_QK_EPS)?;

        // [B,L,H,hd] → [B,H,L,hd]
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let q = apply_rope(&q, cos, sin)?;
        let k = apply_rope(&k, cos, sin)?;

        let scale = (hd as f64).powf(-0.5);
        // Uniform-segment renders pass `None` (mask is all-zeros): skip the broadcast-add entirely
        // (sc-8992). `softmax(scores + 0) == softmax(scores)`, so this is byte-identical. The additive
        // mask is `[B,1,L,L]` (per-query) — cast to the scores dtype up front so the budgeted helper's
        // per-chunk narrow over dim-2 slices the matching query rows.
        // i32-overflow guard (sc-9116): the image-token scores `[B,H,L,L]` reach `~24·16384² ≈ 6.4e9 >
        // i32::MAX` at a 2048² render, so chunk over the query rows (byte-identical for common sizes).
        let mask = mask.map(|m| m.to_dtype(q.dtype())).transpose()?;
        let o = candle_gen::sdpa_budgeted_bhsd(
            &q,
            &k,
            &v,
            scale,
            mask.as_ref(),
            softmax_last_dim,
            candle_gen::ATTN_SCORES_BUDGET,
        )?; // [B,H,L,hd]
        let o = o.transpose(1, 2)?.contiguous()?.reshape((b, s, nh * hd))?;
        self.o.forward(&o)
    }

    /// Visit the two adaptable attention projections (`{prefix}.qkv`, `{prefix}.o`) with their canonical
    /// DiT dotted paths — the surface the TurboTime LoRA's `qkv`/`o` targets resolve against (sc-11104).
    pub fn visit_adaptable_mut(
        &mut self,
        prefix: &str,
        f: &mut dyn FnMut(&str, &mut QLinear) -> Result<()>,
    ) -> Result<()> {
        f(&format!("{prefix}.qkv"), &mut self.qkv)?;
        f(&format!("{prefix}.o"), &mut self.o)?;
        Ok(())
    }
}

/// HF half-split RoPE in `[B, H, L, hd]` layout: `cos`/`sin` `[B, L, hd]` → broadcast over heads.
fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let cos = cos.unsqueeze(1)?.to_dtype(x.dtype())?; // [B,1,L,hd]
    let sin = sin.unsqueeze(1)?.to_dtype(x.dtype())?;
    let chunks = x.chunk(2, D::Minus1)?;
    let x1 = chunks[0].contiguous()?;
    let x2 = chunks[1].contiguous()?;
    let rot = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;
    x.broadcast_mul(&cos)? + rot.broadcast_mul(&sin)?
}

// ── SwiGLU MLP ───────────────────────────────────────────────────────────────────────────
pub struct Ideogram4Mlp {
    w1: QLinear,
    w2: QLinear,
    w3: QLinear,
}

impl Ideogram4Mlp {
    pub fn load(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            w1: linear_detect(w, &format!("{prefix}.w1"), false)?,
            w2: linear_detect(w, &format!("{prefix}.w2"), false)?,
            w3: linear_detect(w, &format!("{prefix}.w3"), false)?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gated = (self.w1.forward(x)?.silu()? * self.w3.forward(x)?)?;
        self.w2.forward(&gated)
    }

    /// Visit the three adaptable SwiGLU projections (`{prefix}.w1/w2/w3`) with their canonical DiT
    /// dotted paths — the surface the TurboTime LoRA's feed-forward targets resolve against (sc-11104).
    pub fn visit_adaptable_mut(
        &mut self,
        prefix: &str,
        f: &mut dyn FnMut(&str, &mut QLinear) -> Result<()>,
    ) -> Result<()> {
        f(&format!("{prefix}.w1"), &mut self.w1)?;
        f(&format!("{prefix}.w2"), &mut self.w2)?;
        f(&format!("{prefix}.w3"), &mut self.w3)?;
        Ok(())
    }
}

// ── Block ────────────────────────────────────────────────────────────────────────────────
pub struct Ideogram4Block {
    attention: Ideogram4Attention,
    feed_forward: Ideogram4Mlp,
    attention_norm1: Tensor,
    attention_norm2: Tensor,
    ffn_norm1: Tensor,
    ffn_norm2: Tensor,
    adaln_modulation: QLinear,
    eps: f64,
}

impl Ideogram4Block {
    pub fn load(
        w: &Weights,
        prefix: &str,
        num_heads: usize,
        head_dim: usize,
        norm_eps: f64,
    ) -> Result<Self> {
        Ok(Self {
            attention: Ideogram4Attention::load(
                w,
                &format!("{prefix}.attention"),
                num_heads,
                head_dim,
            )?,
            feed_forward: Ideogram4Mlp::load(w, &format!("{prefix}.feed_forward"))?,
            attention_norm1: w.get(&format!("{prefix}.attention_norm1.weight"))?,
            attention_norm2: w.get(&format!("{prefix}.attention_norm2.weight"))?,
            ffn_norm1: w.get(&format!("{prefix}.ffn_norm1.weight"))?,
            ffn_norm2: w.get(&format!("{prefix}.ffn_norm2.weight"))?,
            adaln_modulation: linear_detect(w, &format!("{prefix}.adaln_modulation"), true)?,
            eps: norm_eps,
        })
    }

    /// `x`: `[B, L, emb]`; `adaln_input`: `[B, 1, adaln_dim]`; `cos`/`sin`: `[B, L, head_dim]`;
    /// `mask`: optional additive `[B, 1, L, L]` (`None` = uniform segment, no masking; sc-8992).
    pub fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: Option<&Tensor>,
        adaln_input: &Tensor,
    ) -> Result<Tensor> {
        let mod_ = self.adaln_modulation.forward(adaln_input)?; // [B,1,4*emb]
        let chunks = mod_.chunk(4, D::Minus1)?;
        let scale_msa = (chunks[0].contiguous()? + 1.0)?;
        let gate_msa = chunks[1].contiguous()?.tanh()?;
        let scale_mlp = (chunks[2].contiguous()? + 1.0)?;
        let gate_mlp = chunks[3].contiguous()?.tanh()?;

        let normed = rmsnorm(x, &self.attention_norm1, self.eps)?.broadcast_mul(&scale_msa)?;
        let attn_out = self.attention.forward(&normed, cos, sin, mask)?;
        let x =
            (x + rmsnorm(&attn_out, &self.attention_norm2, self.eps)?.broadcast_mul(&gate_msa)?)?;

        let normed2 = rmsnorm(&x, &self.ffn_norm1, self.eps)?.broadcast_mul(&scale_mlp)?;
        let ff = self.feed_forward.forward(&normed2)?;
        let x = (&x + rmsnorm(&ff, &self.ffn_norm2, self.eps)?.broadcast_mul(&gate_mlp)?)?;
        Ok(x)
    }

    /// Visit every adaptable projection in this block with its canonical DiT dotted path (sc-11104):
    /// `{prefix}.attention.{qkv,o}`, `{prefix}.feed_forward.{w1,w2,w3}`, `{prefix}.adaln_modulation`.
    pub fn visit_adaptable_mut(
        &mut self,
        prefix: &str,
        f: &mut dyn FnMut(&str, &mut QLinear) -> Result<()>,
    ) -> Result<()> {
        self.attention
            .visit_adaptable_mut(&format!("{prefix}.attention"), f)?;
        self.feed_forward
            .visit_adaptable_mut(&format!("{prefix}.feed_forward"), f)?;
        f(
            &format!("{prefix}.adaln_modulation"),
            &mut self.adaln_modulation,
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::{safetensors, DType, Device};
    use std::collections::HashMap;
    use std::path::Path;

    /// Build a `Weights` over a written dense component from `(name → [out,in]-or-[dim])` shapes.
    fn weights_from(dir: &Path, shapes: &[(&str, Vec<usize>)]) -> Weights {
        let dev = Device::Cpu;
        let mut map: HashMap<String, Tensor> = HashMap::new();
        for (name, dims) in shapes {
            map.insert(
                (*name).to_string(),
                Tensor::randn(0f32, 1f32, dims.clone(), &dev).unwrap(),
            );
        }
        std::fs::create_dir_all(dir).unwrap();
        safetensors::save(&map, dir.join("model.safetensors")).unwrap();
        Weights::from_dir(dir, &dev, DType::F32).unwrap()
    }

    /// **The block visitor emits the canonical DiT dotted paths (sc-11104).** These are exactly the keys
    /// the TurboTime LoRA's prefix-stripped modules resolve against, so a path typo here would silently
    /// drop every residual (no-match). Locks the full per-block surface, in walk order.
    #[test]
    fn block_visitor_emits_canonical_paths() {
        let (e, h, hd, hidden) = (8usize, 2usize, 4usize, 16usize);
        let dir = std::env::temp_dir().join(format!("sc11104_blk_{}", std::process::id()));
        let w = weights_from(
            &dir,
            &[
                ("layers.0.attention.qkv.weight", vec![3 * h * hd, e]),
                ("layers.0.attention.o.weight", vec![e, h * hd]),
                ("layers.0.attention.norm_q.weight", vec![hd]),
                ("layers.0.attention.norm_k.weight", vec![hd]),
                ("layers.0.feed_forward.w1.weight", vec![hidden, e]),
                ("layers.0.feed_forward.w2.weight", vec![e, hidden]),
                ("layers.0.feed_forward.w3.weight", vec![hidden, e]),
                ("layers.0.adaln_modulation.weight", vec![4 * e, e]),
                ("layers.0.adaln_modulation.bias", vec![4 * e]),
                ("layers.0.attention_norm1.weight", vec![e]),
                ("layers.0.attention_norm2.weight", vec![e]),
                ("layers.0.ffn_norm1.weight", vec![e]),
                ("layers.0.ffn_norm2.weight", vec![e]),
            ],
        );
        let mut block = Ideogram4Block::load(&w, "layers.0", h, hd, 1e-6).unwrap();
        let mut paths = Vec::new();
        block
            .visit_adaptable_mut("layers.0", &mut |p, _| {
                paths.push(p.to_string());
                Ok(())
            })
            .unwrap();
        assert_eq!(
            paths,
            vec![
                "layers.0.attention.qkv",
                "layers.0.attention.o",
                "layers.0.feed_forward.w1",
                "layers.0.feed_forward.w2",
                "layers.0.feed_forward.w3",
                "layers.0.adaln_modulation",
            ]
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}
