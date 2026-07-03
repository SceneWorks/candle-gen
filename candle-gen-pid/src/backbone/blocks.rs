//! Attention + transformer blocks: `RotaryAttention` (pixel stream), `MMDiTJointAttention` +
//! `MMDiTBlockT2I` (dual-stream patch blocks), and `PiTBlock` (per-pixel block). Faithful port of the
//! corresponding classes in `pixeldit_official.py`.
//!
//! candle simplification vs the MLX port: `candle_gen::sdpa_budgeted_bhsd` materializes (query-chunked)
//! scores, so it works at any `head_dim` — the pixel stream's `head_dim = 72` needs no flash-pad hack
//! (the MLX port padded to 80 for MLX's fused kernel). Runs f32 throughout.

use candle_gen::candle_core::{Tensor, D};
use candle_gen::candle_nn::ops::softmax_last_dim;
use candle_gen::candle_nn::{Linear, Module};
use candle_gen::{Result, Weights};

use super::layers::{FeedForward, Mlp, RMS_EPS};
use super::rope::apply_rope;
use crate::nn::{linear, rms};

/// `norm · (scale + 1) + shift` (the DiT AdaLN modulation; mirrors `mlx_gen::nn::modulate`).
fn modulate(norm: &Tensor, scale: &Tensor, shift: &Tensor) -> Result<Tensor> {
    Ok(norm.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(shift)?)
}

/// `x + gate · y` (mirrors `mlx_gen::nn::gated`). `gate` broadcasts over the token axis.
fn gated(x: &Tensor, gate: &Tensor, y: &Tensor) -> Result<Tensor> {
    Ok(x.broadcast_add(&y.broadcast_mul(gate)?)?)
}

/// `[B,S,3·H·Dh]` → q,k,v each `[B,S,H,Dh]` via `reshape(B,S,3,H,Dh)` then slice the `3` axis.
fn split_qkv(qkv: &Tensor, heads: usize, head_dim: usize) -> Result<(Tensor, Tensor, Tensor)> {
    let (b, s, _) = qkv.dims3()?;
    let q5 = qkv.reshape((b, s, 3, heads, head_dim))?;
    let take = |i: usize| -> Result<Tensor> {
        Ok(q5
            .narrow(2, i, 1)?
            .contiguous()?
            .reshape((b, s, heads, head_dim))?)
    };
    Ok((take(0)?, take(1)?, take(2)?))
}

/// `[B,H,S,Dh]` → `[B,S,H·Dh]`.
fn merge_heads(x: &Tensor) -> Result<Tensor> {
    let (b, h, s, d) = x.dims4()?;
    Ok(x.transpose(1, 2)?.contiguous()?.reshape((b, s, h * d))?)
}

/// `[B,S,H,Dh]` → `[B,H,S,Dh]`.
fn to_bhsd(x: &Tensor) -> Result<Tensor> {
    Ok(x.transpose(1, 2)?.contiguous()?)
}

fn sdpa(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64) -> Result<Tensor> {
    Ok(candle_gen::sdpa_budgeted_bhsd(
        q,
        k,
        v,
        scale,
        None,
        softmax_last_dim,
        candle_gen::ATTN_SCORES_BUDGET,
    )?)
}

/// `RotaryAttention` — single-stream qk-normed rotary attention (the PiT pixel block's attention).
pub struct RotaryAttention {
    qkv: Linear,
    q_norm: Tensor,
    k_norm: Tensor,
    proj: Linear,
    heads: usize,
    head_dim: usize,
}

impl RotaryAttention {
    pub fn from_weights(w: &Weights, prefix: &str, dim: i32, heads: i32) -> Result<Self> {
        Ok(Self {
            qkv: linear(w, &format!("{prefix}.qkv"))?,
            q_norm: w.require(&format!("{prefix}.q_norm.weight"))?,
            k_norm: w.require(&format!("{prefix}.k_norm.weight"))?,
            proj: linear(w, &format!("{prefix}.proj"))?,
            heads: heads as usize,
            head_dim: (dim / heads) as usize,
        })
    }

    /// `x`: `[B, N, dim]`; `cos`/`sin`: `[N, head_dim/2]`.
    pub fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (q, k, v) = split_qkv(&self.qkv.forward(x)?, self.heads, self.head_dim)?;
        let q = rms(&q, &self.q_norm, RMS_EPS)?;
        let k = rms(&k, &self.k_norm, RMS_EPS)?;
        let (q, k, v) = (to_bhsd(&q)?, to_bhsd(&k)?, to_bhsd(&v)?);
        let (q, k) = apply_rope(&q, &k, cos, sin)?;
        let scale = (self.head_dim as f64).powf(-0.5);
        let o = sdpa(&q, &k, &v, scale)?;
        Ok(self.proj.forward(&merge_heads(&o)?)?)
    }
}

/// `MMDiTJointAttention` — separate img/txt QKV with per-stream qk-norm, RoPE (2-D on img, 1-D on
/// txt), a single joint SDPA over `[txt, img]`, then per-stream output projections.
pub struct MMDiTJointAttention {
    qkv_x: Linear,
    qkv_y: Linear,
    q_norm_x: Tensor,
    k_norm_x: Tensor,
    q_norm_y: Tensor,
    k_norm_y: Tensor,
    proj_x: Linear,
    proj_y: Linear,
    heads: usize,
    head_dim: usize,
}

impl MMDiTJointAttention {
    pub fn from_weights(w: &Weights, prefix: &str, dim: i32, heads: i32) -> Result<Self> {
        Ok(Self {
            qkv_x: linear(w, &format!("{prefix}.qkv_x"))?,
            qkv_y: linear(w, &format!("{prefix}.qkv_y"))?,
            q_norm_x: w.require(&format!("{prefix}.q_norm_x.weight"))?,
            k_norm_x: w.require(&format!("{prefix}.k_norm_x.weight"))?,
            q_norm_y: w.require(&format!("{prefix}.q_norm_y.weight"))?,
            k_norm_y: w.require(&format!("{prefix}.k_norm_y.weight"))?,
            proj_x: linear(w, &format!("{prefix}.proj_x"))?,
            proj_y: linear(w, &format!("{prefix}.proj_y"))?,
            heads: heads as usize,
            head_dim: (dim / heads) as usize,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: &Tensor,
        y: &Tensor,
        cos_img: &Tensor,
        sin_img: &Tensor,
        cos_txt: &Tensor,
        sin_txt: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let ny = y.dim(1)?;
        let nx = x.dim(1)?;

        let (qx, kx, vx) = split_qkv(&self.qkv_x.forward(x)?, self.heads, self.head_dim)?;
        let qx = rms(&qx, &self.q_norm_x, RMS_EPS)?;
        let kx = rms(&kx, &self.k_norm_x, RMS_EPS)?;
        let (qy, ky, vy) = split_qkv(&self.qkv_y.forward(y)?, self.heads, self.head_dim)?;
        let qy = rms(&qy, &self.q_norm_y, RMS_EPS)?;
        let ky = rms(&ky, &self.k_norm_y, RMS_EPS)?;

        let (qx, kx, vx) = (to_bhsd(&qx)?, to_bhsd(&kx)?, to_bhsd(&vx)?);
        let (qy, ky, vy) = (to_bhsd(&qy)?, to_bhsd(&ky)?, to_bhsd(&vy)?);
        let (qx, kx) = apply_rope(&qx, &kx, cos_img, sin_img)?;
        let (qy, ky) = apply_rope(&qy, &ky, cos_txt, sin_txt)?;

        // joint sequence [txt, img] along the token axis (axis 2 of [B,H,S,Dh])
        let q = Tensor::cat(&[&qy, &qx], 2)?;
        let k = Tensor::cat(&[&ky, &kx], 2)?;
        let v = Tensor::cat(&[&vy, &vx], 2)?;
        let scale = (self.head_dim as f64).powf(-0.5);
        let out = sdpa(&q, &k, &v, scale)?;

        let out_y = merge_heads(&out.narrow(2, 0, ny)?.contiguous()?)?;
        let out_x = merge_heads(&out.narrow(2, ny, nx)?.contiguous()?)?;
        Ok((self.proj_x.forward(&out_x)?, self.proj_y.forward(&out_y)?))
    }
}

/// `MMDiTBlockT2I` — dual-stream block: joint attention + per-stream SwiGLU FFN, each gated by an
/// AdaLN modulation of the shared (already-SiLU'd) condition.
pub struct MMDiTBlockT2I {
    norm_x1: Tensor,
    norm_y1: Tensor,
    attn: MMDiTJointAttention,
    norm_x2: Tensor,
    norm_y2: Tensor,
    mlp_x: FeedForward,
    mlp_y: FeedForward,
    adaln_img: Linear,
    adaln_txt: Linear,
}

impl MMDiTBlockT2I {
    pub fn from_weights(w: &Weights, prefix: &str, dim: i32, heads: i32) -> Result<Self> {
        Ok(Self {
            norm_x1: w.require(&format!("{prefix}.norm_x1.weight"))?,
            norm_y1: w.require(&format!("{prefix}.norm_y1.weight"))?,
            attn: MMDiTJointAttention::from_weights(w, &format!("{prefix}.attn"), dim, heads)?,
            norm_x2: w.require(&format!("{prefix}.norm_x2.weight"))?,
            norm_y2: w.require(&format!("{prefix}.norm_y2.weight"))?,
            mlp_x: FeedForward::from_weights(w, &format!("{prefix}.mlp_x"))?,
            mlp_y: FeedForward::from_weights(w, &format!("{prefix}.mlp_y"))?,
            adaln_img: linear(w, &format!("{prefix}.adaLN_modulation_img.0"))?,
            adaln_txt: linear(w, &format!("{prefix}.adaLN_modulation_txt.0"))?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: &Tensor,
        y: &Tensor,
        c: &Tensor,
        cos_img: &Tensor,
        sin_img: &Tensor,
        cos_txt: &Tensor,
        sin_txt: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let mx = self.adaln_img.forward(c)?.chunk(6, D::Minus1)?;
        let my = self.adaln_txt.forward(c)?.chunk(6, D::Minus1)?;

        let x_norm = modulate(&rms(x, &self.norm_x1, RMS_EPS)?, &mx[1], &mx[0])?;
        let y_norm = modulate(&rms(y, &self.norm_y1, RMS_EPS)?, &my[1], &my[0])?;
        let (attn_x, attn_y) = self
            .attn
            .forward(&x_norm, &y_norm, cos_img, sin_img, cos_txt, sin_txt)?;
        let x = gated(x, &mx[2], &attn_x)?;
        let y = gated(y, &my[2], &attn_y)?;

        let x_mlp = self.mlp_x.forward(&modulate(
            &rms(&x, &self.norm_x2, RMS_EPS)?,
            &mx[4],
            &mx[3],
        )?)?;
        let x = gated(&x, &mx[5], &x_mlp)?;
        let y_mlp = self.mlp_y.forward(&modulate(
            &rms(&y, &self.norm_y2, RMS_EPS)?,
            &my[4],
            &my[3],
        )?)?;
        let y = gated(&y, &my[5], &y_mlp)?;
        Ok((x, y))
    }
}

/// `PiTBlock` — per-pixel block: compress the per-patch pixels to one attention token, rotary
/// attention across patch tokens, expand back, GELU MLP; both stages AdaLN-gated per pixel.
pub struct PiTBlock {
    compress_to_attn: Linear,
    expand_from_attn: Linear,
    norm1: Tensor,
    attn: RotaryAttention,
    norm2: Tensor,
    mlp: Mlp,
    adaln: Linear,
    pixel_dim: usize,
    attn_dim: usize,
}

impl PiTBlock {
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        pixel_dim: i32,
        attn_dim: i32,
        attn_heads: i32,
    ) -> Result<Self> {
        Ok(Self {
            compress_to_attn: linear(w, &format!("{prefix}.compress_to_attn"))?,
            expand_from_attn: linear(w, &format!("{prefix}.expand_from_attn"))?,
            norm1: w.require(&format!("{prefix}.norm1.weight"))?,
            attn: RotaryAttention::from_weights(
                w,
                &format!("{prefix}.attn"),
                attn_dim,
                attn_heads,
            )?,
            norm2: w.require(&format!("{prefix}.norm2.weight"))?,
            mlp: Mlp::from_weights(w, &format!("{prefix}.mlp"))?,
            adaln: linear(w, &format!("{prefix}.adaLN_modulation.0"))?,
            pixel_dim: pixel_dim as usize,
            attn_dim: attn_dim as usize,
        })
    }

    /// `x`: `[B·L, P², pixel_dim]`; `s_cond`: `[B·L, context_dim]`; `cos`/`sin`: pixel-stream
    /// 2-D RoPE for the `(Hs, Ws)` patch grid. `b`/`l` are the batch and patch-grid token counts.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: &Tensor,
        s_cond: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        b: usize,
        l: usize,
    ) -> Result<Tensor> {
        let (bl, p2, _) = x.dims3()?;
        let cond = self
            .adaln
            .forward(s_cond)?
            .reshape((bl, p2, 6 * self.pixel_dim))?;
        let m = cond.chunk(6, D::Minus1)?; // shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp

        let x_norm = modulate(&rms(x, &self.norm1, RMS_EPS)?, &m[1], &m[0])?;
        let x_flat = x_norm.reshape((bl, p2 * self.pixel_dim))?;
        let x_comp = self
            .compress_to_attn
            .forward(&x_flat)?
            .reshape((b, l, self.attn_dim))?;
        let attn_out = self.attn.forward(&x_comp, cos, sin)?;
        let attn_exp = self
            .expand_from_attn
            .forward(&attn_out.reshape((b * l, self.attn_dim))?)?
            .reshape((bl, p2, self.pixel_dim))?;
        let x = gated(x, &m[2], &attn_exp)?;

        let mlp_out =
            self.mlp
                .forward(&modulate(&rms(&x, &self.norm2, RMS_EPS)?, &m[4], &m[3])?)?;
        gated(&x, &m[5], &mlp_out)
    }
}
