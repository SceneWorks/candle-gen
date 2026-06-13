//! Lens dual-stream MMDiT block (`LensTransformerBlock`). Each stream (image, text) gets two AdaLN
//! modulations from the timestep embedding — `mod1` around the joint attention, `mod2` around the
//! **SwiGLU** MLP — with gated residuals. Norms are affine **RMSNorm** (`rms_norm=True`, eps 1e-6).

use candle_gen::candle_core::{Result, Tensor, D};
use candle_gen::candle_nn::{linear, linear_no_bias, rms_norm, Linear, Module, RmsNorm, VarBuilder};

use super::attention::LensJointAttention;

const NORM_EPS: f64 = 1e-6;

/// SwiGLU MLP (`GateMLP`): `w2(silu(w1(x)) · w3(x))`, all bias-less. Hidden width `inner/3·8`.
struct GateMlp {
    w1: Linear,
    w2: Linear,
    w3: Linear,
}

impl GateMlp {
    fn new(inner: usize, hidden: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            w1: linear_no_bias(inner, hidden, vb.pp("w1"))?,
            w2: linear_no_bias(hidden, inner, vb.pp("w2"))?,
            w3: linear_no_bias(inner, hidden, vb.pp("w3"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.w1.forward(x)?.silu()?;
        let up = self.w3.forward(x)?;
        self.w2.forward(&gate.broadcast_mul(&up)?)
    }
}

/// Split a `[B, 3·dim]` modulation laid out **(shift, scale, gate)** into the three `[B, 1, dim]`
/// broadcastable parts (the reference `_modulate`).
fn chunk3(m: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
    let dim = m.dim(D::Minus1)? / 3;
    let shift = m.narrow(D::Minus1, 0, dim)?.unsqueeze(1)?;
    let scale = m.narrow(D::Minus1, dim, dim)?.unsqueeze(1)?;
    let gate = m.narrow(D::Minus1, 2 * dim, dim)?.unsqueeze(1)?;
    Ok((shift, scale, gate))
}

/// AdaLN modulate: returns `(x·(1+scale) + shift, gate)`.
fn modulate(x: &Tensor, m: &Tensor) -> Result<(Tensor, Tensor)> {
    let (shift, scale, gate) = chunk3(m)?;
    let out = x.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(&shift)?;
    Ok((out, gate))
}

/// Gated residual `x + gate·y`.
fn gated(x: &Tensor, gate: &Tensor, y: &Tensor) -> Result<Tensor> {
    x + y.broadcast_mul(gate)?
}

pub struct LensTransformerBlock {
    img_mod: Linear,
    txt_mod: Linear,
    img_norm1: RmsNorm,
    img_norm2: RmsNorm,
    txt_norm1: RmsNorm,
    txt_norm2: RmsNorm,
    attn: LensJointAttention,
    img_mlp: GateMlp,
    txt_mlp: GateMlp,
}

impl LensTransformerBlock {
    pub fn new(
        inner: usize,
        heads: usize,
        head_dim: usize,
        mlp_hidden: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            img_mod: linear(inner, 6 * inner, vb.pp("img_mod").pp("1"))?,
            txt_mod: linear(inner, 6 * inner, vb.pp("txt_mod").pp("1"))?,
            img_norm1: rms_norm(inner, NORM_EPS, vb.pp("img_norm1"))?,
            img_norm2: rms_norm(inner, NORM_EPS, vb.pp("img_norm2"))?,
            txt_norm1: rms_norm(inner, NORM_EPS, vb.pp("txt_norm1"))?,
            txt_norm2: rms_norm(inner, NORM_EPS, vb.pp("txt_norm2"))?,
            attn: LensJointAttention::new(inner, heads, head_dim, vb.pp("attn"))?,
            img_mlp: GateMlp::new(inner, mlp_hidden, vb.pp("img_mlp"))?,
            txt_mlp: GateMlp::new(inner, mlp_hidden, vb.pp("txt_mlp"))?,
        })
    }

    /// Returns `(encoder_hidden_states, hidden_states)` (text, image) — the reference block's order.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        hidden_states: &Tensor,         // image [B, img_seq, inner]
        encoder_hidden_states: &Tensor, // text  [B, txt_seq, inner]
        temb: &Tensor,                  // [B, inner]
        img_cos: &Tensor,
        img_sin: &Tensor,
        txt_cos: &Tensor,
        txt_sin: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        // SiLU'd timestep → per-stream 6·dim modulation, split into mod1 (around attn) / mod2 (MLP).
        let act = temb.silu()?;
        let img_mod = self.img_mod.forward(&act)?; // [B, 6·inner]
        let txt_mod = self.txt_mod.forward(&act)?;
        let half = img_mod.dim(D::Minus1)? / 2; // 3·inner
        let im0 = img_mod.narrow(D::Minus1, 0, half)?;
        let im1 = img_mod.narrow(D::Minus1, half, half)?;
        let tm0 = txt_mod.narrow(D::Minus1, 0, half)?;
        let tm1 = txt_mod.narrow(D::Minus1, half, half)?;

        // Attention path.
        let (img_n, img_g1) = modulate(&self.img_norm1.forward(hidden_states)?, &im0)?;
        let (txt_n, txt_g1) = modulate(&self.txt_norm1.forward(encoder_hidden_states)?, &tm0)?;
        let (img_attn, txt_attn) =
            self.attn
                .forward(&img_n, &txt_n, img_cos, img_sin, txt_cos, txt_sin, mask)?;
        let hidden_states = gated(hidden_states, &img_g1, &img_attn)?;
        let encoder_hidden_states = gated(encoder_hidden_states, &txt_g1, &txt_attn)?;

        // Feed-forward path (SwiGLU).
        let (img_n2, img_g2) = modulate(&self.img_norm2.forward(&hidden_states)?, &im1)?;
        let hidden_states = gated(&hidden_states, &img_g2, &self.img_mlp.forward(&img_n2)?)?;
        let (txt_n2, txt_g2) = modulate(&self.txt_norm2.forward(&encoder_hidden_states)?, &tm1)?;
        let encoder_hidden_states =
            gated(&encoder_hidden_states, &txt_g2, &self.txt_mlp.forward(&txt_n2)?)?;

        Ok((encoder_hidden_states, hidden_states))
    }
}
