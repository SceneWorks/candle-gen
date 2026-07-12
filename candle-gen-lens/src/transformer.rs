//! The Lens denoising **DiT** (`LensTransformer2DModel`, sc-5112) — a 48-layer dual-stream MMDiT with
//! joint image+text attention, complex axial RoPE on both streams, and SwiGLU MLPs. A from-scratch
//! candle port of the vendor `LensTransformer2DModel`, architecturally a near-twin of
//! [`candle-gen-qwen-image`]'s MMDiT (the RoPE, joint attention, AdaLN modulation and
//! `AdaLayerNormContinuous` all follow that seam). The Lens-specific pieces are:
//!
//! - a **multi-layer text front-end** — the 4 captured gpt-oss layers (each `[B, txt, 2880]`) get a
//!   per-layer affine RMSNorm (eps **1e-5**) then channel-concat (`2880·4 = 11520`) → `txt_in`;
//! - **fused** per-stream `img_qkv` / `txt_qkv` projections (split into q/k/v after the matmul);
//! - **`[img, txt]`** join order (image tokens first — the reference orders image first);
//! - **SwiGLU GateMLP** (`w2(silu(w1·x) · w3·x)`, hidden `inner/3·8 = 4096`);
//! - affine **RMSNorm** block norms (`rms_norm=True`, eps 1e-6) rather than affine-free LayerNorm;
//! - a **biased** `norm_out.linear` (the checkpoint's `AdaLayerNormContinuous` uses the bias).
//!
//! `[B, seq, dim]` tensors throughout. The model consumes already-patchified image latents
//! `[B, img_len, 128]` plus the 4 captured text-feature layers and predicts the patch-space velocity
//! `[B, img_len, 128]` (= `patch²·out_channels`). Run bf16 in production / f32 for the parity gate.

use candle_gen::candle_core::{DType, Device, Result, Tensor, D};
use candle_gen::candle_nn::{
    linear, ops::softmax_last_dim, rms_norm, Linear, Module, RmsNorm, VarBuilder,
};
use candle_gen::gen_core::Quant;

use crate::quant::QLinear;
use crate::rope::{apply_rope, LensRope};

/// Block / QK-norm / norm_out epsilon (the reference builds its norms at eps 1e-6). Shared with the
/// trainable twin ([`crate::dit_train`]) so the two stay in lockstep.
pub const EPS: f64 = 1e-6;
/// The multi-layer text front-end RMSNorm epsilon (the `txt_norm` per-layer norms use eps 1e-5).
pub const TXT_NORM_EPS: f64 = 1e-5;

/// The Lens / Lens-Turbo `transformer/config.json` values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LensDitConfig {
    pub patch_size: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub inner_dim: usize,
    /// gpt-oss hidden width per captured text layer (2880).
    pub enc_hidden_dim: usize,
    /// Number of captured gpt-oss layers (`selected_layer_index = [5, 11, 17, 23]`).
    pub num_text_layers: usize,
    /// Sinusoidal timestep-embedding width (256).
    pub timestep_channels: usize,
    pub axes_dims_rope: [usize; 3],
    pub rope_theta: f32,
}

impl LensDitConfig {
    pub fn lens() -> Self {
        Self {
            patch_size: 2,
            in_channels: 128,
            out_channels: 32,
            num_layers: 48,
            num_heads: 24,
            head_dim: 64,
            inner_dim: 1536,
            enc_hidden_dim: 2880,
            num_text_layers: 4,
            timestep_channels: 256,
            axes_dims_rope: [8, 28, 28],
            rope_theta: 10_000.0,
        }
    }

    /// SwiGLU GateMLP hidden width: `inner/3·8` (= 4096).
    pub fn mlp_hidden(&self) -> usize {
        self.inner_dim / 3 * 8
    }

    /// Concatenated text front-end width: `enc_hidden_dim · num_text_layers` (= 11520).
    pub fn txt_in_dim(&self) -> usize {
        self.enc_hidden_dim * self.num_text_layers
    }
}

/// Affine-free LayerNorm over the last axis (dtype-preserving; computed in f32).
fn layer_norm(x: &Tensor) -> Result<Tensor> {
    let dt = x.dtype();
    let xf = x.to_dtype(DType::F32)?;
    let mean = xf.mean_keepdim(D::Minus1)?;
    let xc = xf.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
    xc.broadcast_div(&(var + EPS)?.sqrt()?)?.to_dtype(dt)
}

/// Split a `[B, 3·inner]` modulation chunk into `(shift, scale, gate)`, each `[B, 1, inner]` —
/// the reference `_modulate` layout is **(shift, scale, gate)**.
fn chunk3(m: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
    let inner = m.dim(D::Minus1)? / 3;
    let shift = m.narrow(D::Minus1, 0, inner)?.unsqueeze(1)?;
    let scale = m.narrow(D::Minus1, inner, inner)?.unsqueeze(1)?;
    let gate = m.narrow(D::Minus1, 2 * inner, inner)?.unsqueeze(1)?;
    Ok((shift, scale, gate))
}

/// AdaLN modulate: returns `(x·(1+scale) + shift, gate)`. Composable (no fused op), so the trainable
/// twin ([`crate::dit_train`]) reuses it verbatim.
pub fn modulate(x: &Tensor, m: &Tensor) -> Result<(Tensor, Tensor)> {
    let (shift, scale, gate) = chunk3(m)?;
    let out = x.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(&shift)?;
    Ok((out, gate))
}

/// `x + gate·y`. Composable; reused by [`crate::dit_train`].
pub fn gated(x: &Tensor, gate: &Tensor, y: &Tensor) -> Result<Tensor> {
    x.broadcast_add(&y.broadcast_mul(gate)?)
}

/// Max elements in a single attention scores tensor `[B,H,Sq,Sk]` before [`attention`] chunks over the
/// query rows. candle CUDA kernels index elements with **i32**, so a scores/probs tensor exceeding
/// `i32::MAX` (~2.147B) silently corrupts its tail — garbage attention in the trailing query rows →
/// noise, with no error (sc-5487, the FLUX.2 fix; sc-8983 ports it here). Lens advertises buckets up
/// to a 1440 base and always renders the CFG batch of 2: at the largest buckets the joint
/// `[img, txt]` sequence pushes the 24-head, B=2 scores tensor to ~3.8B elements, past the limit.
/// 1.0B keeps each chunk well under it while leaving the common sizes a single un-chunked pass, so
/// those stay byte-identical.
/// SDPA over `[B,H,S,head_dim]` q/k/v → `[B, S, H·head_dim]`. scale = `head_dim^-0.5`. `mask` is an
/// optional additive mask broadcast onto the scores; it must broadcast over the query rows (the
/// [`build_joint_mask`] shape `[B, 1, 1, Sk]`). Delegates to the shared i32-overflow-safe
/// [`candle_gen::sdpa_budgeted_bhsd`] (sc-9570), which chunks over the query rows once the `[B,H,Sq,Sk]`
/// scores tensor would exceed [`candle_gen::ATTN_SCORES_BUDGET`] (the candle CUDA i32-index limit) —
/// broadcasting the `[B,1,1,Sk]` mask identically onto every chunk. The `softmax_last_dim` closure keeps
/// the exact fused softmax; each query row's softmax is independent, so the chunked result is
/// byte-identical to the single pass. This crate does the head-merge transpose/reshape here.
fn attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    head_dim: usize,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    let (b, _h, s, _d) = q.dims4()?;
    let scale = (head_dim as f64).powf(-0.5);
    let o = candle_gen::sdpa_budgeted_bhsd(
        q,
        k,
        v,
        scale,
        mask,
        softmax_last_dim,
        candle_gen::ATTN_SCORES_BUDGET,
    )?; // [B,H,S,head_dim]
    let (_b, h, _s, d) = o.dims4()?;
    o.transpose(1, 2)?.reshape((b, s, h * d))
}

/// Sinusoidal timestep embedding `[1, dim]` from the raw sigma (diffusers `Timesteps(dim,
/// flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000)`): arg `= σ·1000·freq`, `[cos | sin]`.
fn timestep_embedding(sigma: f32, dim: usize, device: &Device) -> Result<Tensor> {
    let half = dim / 2;
    let ln = 10000f32.ln();
    let mut cos = vec![0f32; half];
    let mut sin = vec![0f32; half];
    for k in 0..half {
        let freq = (-ln * k as f32 / half as f32).exp();
        let arg = sigma * 1000.0 * freq;
        cos[k] = arg.cos();
        sin[k] = arg.sin();
    }
    let cos = Tensor::from_vec(cos, (1, half), device)?;
    let sin = Tensor::from_vec(sin, (1, half), device)?;
    Tensor::cat(&[&cos, &sin], D::Minus1)
}

/// `temb = linear_2(silu(linear_1(proj(t))))`, `[1] → [1, inner]`. Frozen + composable, so the
/// trainable twin ([`crate::dit_train`]) reuses it (the timestep embed is upstream of every adapter).
pub struct TimeEmbed {
    linear_1: Linear,
    linear_2: Linear,
    channels: usize,
}

impl TimeEmbed {
    pub fn new(cfg: &LensDitConfig, vb: VarBuilder) -> Result<Self> {
        let inner = cfg.inner_dim;
        let te = vb.pp("timestep_embedder");
        Ok(Self {
            linear_1: linear(cfg.timestep_channels, inner, te.pp("linear_1"))?,
            linear_2: linear(inner, inner, te.pp("linear_2"))?,
            channels: cfg.timestep_channels,
        })
    }

    pub fn forward(&self, sigma: f32, device: &Device, dtype: DType) -> Result<Tensor> {
        let emb = timestep_embedding(sigma, self.channels, device)?.to_dtype(dtype)?;
        let h = self.linear_1.forward(&emb)?.silu()?;
        self.linear_2.forward(&h)
    }
}

/// SwiGLU MLP (`GateMLP`): `w2(silu(w1·x) · w3·x)`, all bias-less. Hidden width `inner/3·8`. The three
/// projections are [`QLinear`] so they can be Q4/Q8-quantized (sc-5117).
struct GateMlp {
    w1: QLinear,
    w2: QLinear,
    w3: QLinear,
}

impl GateMlp {
    fn new(inner: usize, hidden: usize, vb: VarBuilder) -> Result<Self> {
        // Packed-detect each projection (sc-9413): a `SceneWorks/lens-mlx` q4/q8 tier loads straight
        // from the packed parts, a dense tier loads dense (then optionally folded by `quantize`).
        Ok(Self {
            w1: QLinear::linear_detect(inner, hidden, &vb, "w1", false)?,
            w2: QLinear::linear_detect(hidden, inner, &vb, "w2", false)?,
            w3: QLinear::linear_detect(inner, hidden, &vb, "w3", false)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.w1.forward(x)?.silu()?;
        let up = self.w3.forward(x)?;
        self.w2.forward(&gate.mul(&up)?)
    }

    fn quantize(&mut self, quant: Quant) -> Result<()> {
        self.w1.quantize(quant)?;
        self.w2.quantize(quant)?;
        self.w3.quantize(quant)?;
        Ok(())
    }

    /// Visit the three SwiGLU projections (`{prefix}.w1/w2/w3`) — part of the adaptable surface (sc-11105).
    fn visit_adaptable_mut(
        &mut self,
        prefix: &str,
        f: &mut dyn FnMut(&str, &mut QLinear) -> candle_gen::Result<()>,
    ) -> candle_gen::Result<()> {
        f(&format!("{prefix}.w1"), &mut self.w1)?;
        f(&format!("{prefix}.w2"), &mut self.w2)?;
        f(&format!("{prefix}.w3"), &mut self.w3)?;
        Ok(())
    }
}

/// Lens joint (dual-stream) attention. **Fused** `img_qkv`/`txt_qkv` (biased) split into per-stream
/// q/k/v, per-head q/k RMSNorm, interleaved-complex RoPE on both streams, then SDPA over the
/// **`[img, txt]`**-concatenated sequence (image first), split back and projected (`to_out.0` for
/// image, `to_add_out` for text).
struct JointAttention {
    img_qkv: QLinear,
    txt_qkv: QLinear,
    to_out: QLinear,
    to_add_out: QLinear,
    norm_q: RmsNorm,
    norm_k: RmsNorm,
    norm_added_q: RmsNorm,
    norm_added_k: RmsNorm,
    heads: usize,
    head_dim: usize,
}

impl JointAttention {
    fn new(cfg: &LensDitConfig, vb: VarBuilder) -> Result<Self> {
        let inner = cfg.inner_dim;
        let hd = cfg.head_dim;
        Ok(Self {
            // Packed-detect each projection (sc-9413). `to_out.0` is threaded as one base string so the
            // `.scales`/`.biases` siblings survive the `.0` nesting (never `.pp("0")` past the sibling).
            img_qkv: QLinear::linear_detect(inner, 3 * inner, &vb, "img_qkv", true)?,
            txt_qkv: QLinear::linear_detect(inner, 3 * inner, &vb, "txt_qkv", true)?,
            to_out: QLinear::linear_detect(inner, inner, &vb, "to_out.0", true)?,
            to_add_out: QLinear::linear_detect(inner, inner, &vb, "to_add_out", true)?,
            norm_q: rms_norm(hd, EPS, vb.pp("norm_q"))?,
            norm_k: rms_norm(hd, EPS, vb.pp("norm_k"))?,
            norm_added_q: rms_norm(hd, EPS, vb.pp("norm_added_q"))?,
            norm_added_k: rms_norm(hd, EPS, vb.pp("norm_added_k"))?,
            heads: cfg.num_heads,
            head_dim: hd,
        })
    }

    /// Quantize the four fused/output projections to Q4/Q8 (sc-5117). Called **after** any adapter
    /// merge (the merge folds `W += δ` into the dense weight before the DiT is built, so the quantized
    /// base already carries the adapter delta). The QK-norm weights stay full precision.
    fn quantize(&mut self, quant: Quant) -> Result<()> {
        self.img_qkv.quantize(quant)?;
        self.txt_qkv.quantize(quant)?;
        self.to_out.quantize(quant)?;
        self.to_add_out.quantize(quant)?;
        Ok(())
    }

    /// Visit the four joint-attention projections (`{prefix}.img_qkv/txt_qkv/to_out.0/to_add_out`) — the
    /// surface the Lens trainer's LoRA/LoKr adapts (sc-11105). `img_qkv`/`txt_qkv` are the FUSED q/k/v
    /// projections; a fused-QKV LoRA rides the single projection unmerged (no split needed).
    fn visit_adaptable_mut(
        &mut self,
        prefix: &str,
        f: &mut dyn FnMut(&str, &mut QLinear) -> candle_gen::Result<()>,
    ) -> candle_gen::Result<()> {
        f(&format!("{prefix}.img_qkv"), &mut self.img_qkv)?;
        f(&format!("{prefix}.txt_qkv"), &mut self.txt_qkv)?;
        f(&format!("{prefix}.to_out.0"), &mut self.to_out)?;
        f(&format!("{prefix}.to_add_out"), &mut self.to_add_out)?;
        Ok(())
    }

    /// Fused QKV → `(q, k, v)` each `[B, seq, heads, head_dim]`.
    fn qkv(&self, lin: &QLinear, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (b, s, _) = x.dims3()?;
        let (h, hd) = (self.heads, self.head_dim);
        let t = lin.forward(x)?.reshape((b, s, 3, h, hd))?;
        let q = t.narrow(2, 0, 1)?.squeeze(2)?.contiguous()?;
        let k = t.narrow(2, 1, 1)?.squeeze(2)?.contiguous()?;
        let v = t.narrow(2, 2, 1)?.squeeze(2)?.contiguous()?;
        Ok((q, k, v))
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        img: &Tensor,
        txt: &Tensor,
        img_cos: &Tensor,
        img_sin: &Tensor,
        txt_cos: &Tensor,
        txt_sin: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let img_seq = img.dim(1)?;
        let txt_seq = txt.dim(1)?;
        let hd = self.head_dim;

        let (iq, ik, iv) = self.qkv(&self.img_qkv, img)?;
        let (tq, tk, tv) = self.qkv(&self.txt_qkv, txt)?;

        // QK RMSNorm over head_dim (in `[B, seq, heads, head_dim]`).
        let iq = self.norm_q.forward(&iq)?;
        let ik = self.norm_k.forward(&ik)?;
        let tq = self.norm_added_q.forward(&tq)?;
        let tk = self.norm_added_k.forward(&tk)?;

        // To heads-first `[B, heads, seq, head_dim]`, then interleaved RoPE on q/k.
        let bhsd = |x: &Tensor| -> Result<Tensor> { x.transpose(1, 2)?.contiguous() };
        let iq = apply_rope(&bhsd(&iq)?, img_cos, img_sin)?;
        let ik = apply_rope(&bhsd(&ik)?, img_cos, img_sin)?;
        let iv = bhsd(&iv)?;
        let tq = apply_rope(&bhsd(&tq)?, txt_cos, txt_sin)?;
        let tk = apply_rope(&bhsd(&tk)?, txt_cos, txt_sin)?;
        let tv = bhsd(&tv)?;

        // Joint `[img, txt]` (image first) over the sequence axis.
        let q = Tensor::cat(&[&iq, &tq], 2)?;
        let k = Tensor::cat(&[&ik, &tk], 2)?;
        let v = Tensor::cat(&[&iv, &tv], 2)?;
        let o = attention(&q, &k, &v, hd, mask)?; // [B, joint, H·head_dim]

        // Split back at the image/text boundary (image first).
        let img_o = o.narrow(1, 0, img_seq)?.contiguous()?;
        let txt_o = o.narrow(1, img_seq, txt_seq)?.contiguous()?;
        Ok((
            self.to_out.forward(&img_o)?,
            self.to_add_out.forward(&txt_o)?,
        ))
    }
}

/// Lens dual-stream MMDiT block. Each stream (image, text) gets two AdaLN modulations from the
/// timestep embedding — `mod1` around the joint attention, `mod2` around the SwiGLU MLP — with gated
/// residuals. Norms are affine RMSNorm (eps 1e-6). Public so the parity gate (and the Q4/Q8 quant
/// path, sc-5117) can drive a single block in isolation.
pub struct LensTransformerBlock {
    img_mod: Linear,
    txt_mod: Linear,
    img_norm1: RmsNorm,
    img_norm2: RmsNorm,
    txt_norm1: RmsNorm,
    txt_norm2: RmsNorm,
    attn: JointAttention,
    img_mlp: GateMlp,
    txt_mlp: GateMlp,
}

impl LensTransformerBlock {
    pub fn new(cfg: &LensDitConfig, vb: VarBuilder) -> Result<Self> {
        let inner = cfg.inner_dim;
        let hidden = cfg.mlp_hidden();
        Ok(Self {
            img_mod: linear(inner, 6 * inner, vb.pp("img_mod").pp("1"))?,
            txt_mod: linear(inner, 6 * inner, vb.pp("txt_mod").pp("1"))?,
            img_norm1: rms_norm(inner, EPS, vb.pp("img_norm1"))?,
            img_norm2: rms_norm(inner, EPS, vb.pp("img_norm2"))?,
            txt_norm1: rms_norm(inner, EPS, vb.pp("txt_norm1"))?,
            txt_norm2: rms_norm(inner, EPS, vb.pp("txt_norm2"))?,
            attn: JointAttention::new(cfg, vb.pp("attn"))?,
            img_mlp: GateMlp::new(inner, hidden, vb.pp("img_mlp"))?,
            txt_mlp: GateMlp::new(inner, hidden, vb.pp("txt_mlp"))?,
        })
    }

    /// Quantize the block's compute-heavy linears to Q4/Q8 (sc-5117): the joint-attention projections
    /// and both SwiGLU MLPs. The AdaLN modulations (`img_mod`/`txt_mod`) and the RMSNorm weights stay
    /// full precision (small, and precision-sensitive — the modulation drives every gated residual).
    pub fn quantize(&mut self, quant: Quant) -> Result<()> {
        self.attn.quantize(quant)?;
        self.img_mlp.quantize(quant)?;
        self.txt_mlp.quantize(quant)?;
        Ok(())
    }

    /// Visit every adaptable projection in the block under `{prefix}` — the joint attention + both
    /// SwiGLU MLPs. The AdaLN modulations (`img_mod`/`txt_mod`, plain dense `Linear`) and the RMSNorms
    /// are not `QLinear` and are not part of the additive surface (a LoRA targeting them routes to the
    /// dense tier). sc-11105.
    fn visit_adaptable_mut(
        &mut self,
        prefix: &str,
        f: &mut dyn FnMut(&str, &mut QLinear) -> candle_gen::Result<()>,
    ) -> candle_gen::Result<()> {
        self.attn
            .visit_adaptable_mut(&format!("{prefix}.attn"), f)?;
        self.img_mlp
            .visit_adaptable_mut(&format!("{prefix}.img_mlp"), f)?;
        self.txt_mlp
            .visit_adaptable_mut(&format!("{prefix}.txt_mlp"), f)?;
        Ok(())
    }

    /// Returns `(encoder_hidden_states, hidden_states)` (text, image) — the reference block's order.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        hidden: &Tensor,  // image [B, img_seq, inner]
        encoder: &Tensor, // text  [B, txt_seq, inner]
        temb: &Tensor,    // [B, inner]
        img_cos: &Tensor,
        img_sin: &Tensor,
        txt_cos: &Tensor,
        txt_sin: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        // SiLU'd timestep → per-stream 6·inner modulation, split into mod1 (around attn) / mod2 (MLP).
        let act = temb.silu()?;
        let img_mod = self.img_mod.forward(&act)?;
        let txt_mod = self.txt_mod.forward(&act)?;
        let n = img_mod.dim(D::Minus1)? / 2;
        let (im0, im1) = (
            img_mod.narrow(D::Minus1, 0, n)?,
            img_mod.narrow(D::Minus1, n, n)?,
        );
        let (tm0, tm1) = (
            txt_mod.narrow(D::Minus1, 0, n)?,
            txt_mod.narrow(D::Minus1, n, n)?,
        );

        // attention path
        let (img_n, img_g1) = modulate(&self.img_norm1.forward(hidden)?, &im0)?;
        let (txt_n, txt_g1) = modulate(&self.txt_norm1.forward(encoder)?, &tm0)?;
        let (img_attn, txt_attn) = self
            .attn
            .forward(&img_n, &txt_n, img_cos, img_sin, txt_cos, txt_sin, mask)?;
        let hidden = gated(hidden, &img_g1, &img_attn)?;
        let encoder = gated(encoder, &txt_g1, &txt_attn)?;

        // feed-forward path (SwiGLU)
        let (img_n2, img_g2) = modulate(&self.img_norm2.forward(&hidden)?, &im1)?;
        let hidden = gated(&hidden, &img_g2, &self.img_mlp.forward(&img_n2)?)?;
        let (txt_n2, txt_g2) = modulate(&self.txt_norm2.forward(&encoder)?, &tm1)?;
        let encoder = gated(&encoder, &txt_g2, &self.txt_mlp.forward(&txt_n2)?)?;

        Ok((encoder, hidden))
    }
}

/// `AdaLayerNormContinuous`: affine-free LayerNorm scaled/shifted by `linear(silu(temb))`. The Lens
/// checkpoint's `norm_out.linear` carries a **bias** the reference uses. `[scale | shift]` →
/// `(1+scale)·LN(x) + shift`.
pub struct NormOut {
    linear: Linear,
}

impl NormOut {
    pub fn new(cfg: &LensDitConfig, vb: VarBuilder) -> Result<Self> {
        let inner = cfg.inner_dim;
        Ok(Self {
            linear: linear(inner, 2 * inner, vb.pp("linear"))?,
        })
    }

    pub fn forward(&self, x: &Tensor, temb: &Tensor) -> Result<Tensor> {
        let p = self.linear.forward(&temb.silu()?)?;
        let inner = p.dim(D::Minus1)? / 2;
        let scale = p.narrow(D::Minus1, 0, inner)?.unsqueeze(1)?;
        let shift = p.narrow(D::Minus1, inner, inner)?.unsqueeze(1)?;
        layer_norm(x)?
            .broadcast_mul(&(scale + 1.0)?)?
            .broadcast_add(&shift)
    }
}

/// Per-render RoPE-table cache (sc-8992 / F-012). The image/text `(cos, sin)` tables depend only on
/// the fixed latent grid `(frame, h, w)` and `txt_len` — not on σ / the current latent — so they are
/// identical across every denoise step (×2 under CFG). Cache them keyed on that geometry and rebuild
/// only when it changes; hits Arc-clone the stored handles. Byte-identical to recomputing.
struct LensRopeCache {
    frame: usize,
    h: usize,
    w: usize,
    txt_len: usize,
    img_cos: Tensor,
    img_sin: Tensor,
    txt_cos: Tensor,
    txt_sin: Tensor,
}

/// The Lens denoising DiT (`LensTransformer2DModel`).
pub struct LensTransformer {
    img_in: QLinear,
    txt_norm: Vec<RmsNorm>, // per-layer text front-end RMSNorm (eps 1e-5)
    txt_in: QLinear,
    time_embed: TimeEmbed,
    blocks: Vec<LensTransformerBlock>,
    norm_out: NormOut,
    proj_out: QLinear,
    rope: LensRope,
    cfg: LensDitConfig,
    device: Device,
    dtype: DType,
    /// `Mutex` (not `RefCell`): the DiT is shared as `Arc<LensTransformer>` and must stay `Send + Sync`.
    rope_cache: std::sync::Mutex<Option<LensRopeCache>>,
}

impl LensTransformer {
    /// Load from a diffusers `transformer/` weight set at `dtype` (bf16 production / f32 gate).
    pub fn new(cfg: &LensDitConfig, vb: VarBuilder) -> Result<Self> {
        let inner = cfg.inner_dim;
        let mut txt_norm = Vec::with_capacity(cfg.num_text_layers);
        for i in 0..cfg.num_text_layers {
            txt_norm.push(rms_norm(
                cfg.enc_hidden_dim,
                TXT_NORM_EPS,
                vb.pp("txt_norm").pp(i),
            )?);
        }
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            blocks.push(LensTransformerBlock::new(
                cfg,
                vb.pp("transformer_blocks").pp(i),
            )?);
        }
        Ok(Self {
            // Packed-detect the three top-level projections (sc-9413).
            img_in: QLinear::linear_detect(cfg.in_channels, inner, &vb, "img_in", true)?,
            txt_norm,
            txt_in: QLinear::linear_detect(cfg.txt_in_dim(), inner, &vb, "txt_in", true)?,
            time_embed: TimeEmbed::new(cfg, vb.pp("time_text_embed"))?,
            blocks,
            norm_out: NormOut::new(cfg, vb.pp("norm_out"))?,
            proj_out: QLinear::linear_detect(
                inner,
                cfg.patch_size * cfg.patch_size * cfg.out_channels,
                &vb,
                "proj_out",
                true,
            )?,
            rope: LensRope::new(cfg.rope_theta, cfg.axes_dims_rope),
            cfg: *cfg,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            rope_cache: std::sync::Mutex::new(None),
        })
    }

    /// Build (or reuse) the image + text RoPE `(cos, sin)` tables for this render's fixed geometry
    /// (sc-8992). Recomputed only when `(frame, h, w, txt_len)` changes; otherwise the Arc-backed
    /// handles are cloned. Construction is identical to computing it inline, so every step is
    /// byte-identical.
    #[allow(clippy::type_complexity)]
    fn rope_tables(
        &self,
        frame: usize,
        h: usize,
        w: usize,
        txt_len: usize,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let mut guard = candle_gen::lock_recover(&self.rope_cache);
        if let Some(c) = guard.as_ref() {
            if c.frame == frame && c.h == h && c.w == w && c.txt_len == txt_len {
                return Ok((
                    c.img_cos.clone(),
                    c.img_sin.clone(),
                    c.txt_cos.clone(),
                    c.txt_sin.clone(),
                ));
            }
        }
        let (img_cos, img_sin) = self.rope.img_cos_sin(frame, h, w, &self.device)?;
        let (txt_cos, txt_sin) = self.rope.txt_cos_sin(txt_len, h, w, &self.device)?;
        *guard = Some(LensRopeCache {
            frame,
            h,
            w,
            txt_len,
            img_cos: img_cos.clone(),
            img_sin: img_sin.clone(),
            txt_cos: txt_cos.clone(),
            txt_sin: txt_sin.clone(),
        });
        Ok((img_cos, img_sin, txt_cos, txt_sin))
    }

    /// Fold the DiT's compute-heavy linears to Q4/Q8 in place (sc-5117): `img_in`, `txt_in`,
    /// `proj_out`, and every block's attention projections + SwiGLU MLPs. The timestep embedder, the
    /// AdaLN modulations, `norm_out`, and all RMSNorm weights stay full precision (small and
    /// precision-sensitive). Call **after** any adapter merge — the merge folds `W += δ` into the dense
    /// weight before the DiT is built, so quantizing here transcodes the already-adapted base. Mirrors
    /// `mlx-gen-lens::dit::LensTransformer::quantize` (sc-3175).
    ///
    /// **No-op over a packed tier (sc-9413).** When the DiT loaded from a packed `SceneWorks/lens-mlx`
    /// tier, each projection is already `QLinear::Packed` (loaded straight from the packed parts), and
    /// the per-`QLinear` `quantize` no-ops on it — no dense staging, no re-quantize. This pass then only
    /// folds the dense-tier path (`SceneWorks/Lens` bf16 + optional adapter delta); the two compose.
    ///
    /// **Uniform** `Q4_0`/`Q8_0` across every quantized linear — including the SwiGLU MLP. Uniform Q4
    /// once rendered solid black; sc-7702 traced that to candle's int8 `QMatMul` activation-quant path
    /// (not 4-bit weight precision), and [`QLinear`] now dequantizes to a dense matmul — so uniform Q4
    /// is coherent end-to-end with the full weight-VRAM saving (no MLP-stays-Q8 carve-out needed).
    pub fn quantize(&mut self, quant: Quant) -> Result<()> {
        self.img_in.quantize(quant)?;
        self.txt_in.quantize(quant)?;
        self.proj_out.quantize(quant)?;
        for block in &mut self.blocks {
            block.quantize(quant)?;
        }
        Ok(())
    }

    /// The device the DiT weights live on — the forward-time residual factors are read on the CPU and
    /// moved here at install (else the residual matmul is a device mismatch). sc-11105.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Whether the DiT loaded from a packed MLX tier (any projection is quantized) — the gate the loader
    /// uses to route adapters to the additive install vs the dense fold (sc-11105). Probes `img_in`, a
    /// projection packed in every quantized tier.
    pub fn is_packed(&self) -> bool {
        self.img_in.is_packed()
    }

    /// Walk every adaptable projection, invoking `f(path, &mut QLinear)` once each with the projection's
    /// canonical DiT dotted path — the same paths [`crate::adapters::classify_lora_key`] resolves a LoRA
    /// key to (`img_in`, `txt_in`, `proj_out`, and each `transformer_blocks.{i}` attention + SwiGLU
    /// projection). The additive installer ([`crate::adapters::install_additive`]) pushes a resolved
    /// LoRA/LoKr residual onto each matched projection so a user adapter applies on a packed q4/q8 tier
    /// with the base kept packed (sc-11105).
    pub fn visit_adaptable_mut(
        &mut self,
        f: &mut dyn FnMut(&str, &mut QLinear) -> candle_gen::Result<()>,
    ) -> candle_gen::Result<()> {
        f("img_in", &mut self.img_in)?;
        f("txt_in", &mut self.txt_in)?;
        for (i, blk) in self.blocks.iter_mut().enumerate() {
            blk.visit_adaptable_mut(&format!("transformer_blocks.{i}"), f)?;
        }
        f("proj_out", &mut self.proj_out)?;
        Ok(())
    }

    /// Forward.
    ///
    /// - `hidden_states`: `[B, img_len, in_channels]` patchified image latents (`img_len = frame·h·w`).
    /// - `text_feats`: the `num_text_layers` captured gpt-oss layers, each `[B, txt_len, enc_hidden_dim]`.
    /// - `text_valid`: optional `[B, txt_len]` (1 = valid) → additive joint attention mask; `None` =
    ///   all text valid (no padding), the single-prompt path.
    /// - `timestep`: the scalar sigma in `[0, 1]`.
    /// - `(frame, h, w)`: the latent grid shape (`img_len = frame·h·w`).
    ///
    /// Returns `[B, img_len, patch²·out_channels]` (= 128) patch-space velocity.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        text_feats: &[Tensor],
        text_valid: Option<&Tensor>,
        timestep: f32,
        frame: usize,
        h: usize,
        w: usize,
    ) -> Result<Tensor> {
        // Public boundary: return a typed error rather than aborting the process on a
        // caller-supplied text-feature count mismatch (sc-9025 / F-041).
        if text_feats.len() != self.cfg.num_text_layers {
            return Err(candle_gen::candle_core::Error::Msg(format!(
                "lens transformer forward: expected {} text-feature layers, got {}",
                self.cfg.num_text_layers,
                text_feats.len()
            )));
        }
        let img_len = hidden_states.dim(1)?;
        let txt_len = text_feats[0].dim(1)?;

        let mut hidden = self.img_in.forward(hidden_states)?;

        // Multi-layer text front-end: per-layer RMSNorm (eps 1e-5) → channel-concat → txt_in.
        let mut normed = Vec::with_capacity(self.cfg.num_text_layers);
        for (i, feat) in text_feats.iter().enumerate() {
            normed.push(self.txt_norm[i].forward(feat)?);
        }
        let normed_refs: Vec<&Tensor> = normed.iter().collect();
        let mut encoder = self
            .txt_in
            .forward(&Tensor::cat(&normed_refs, D::Minus1)?)?;

        let temb = self
            .time_embed
            .forward(timestep, &self.device, self.dtype)?;
        // RoPE tables are step-invariant (fixed grid geometry), so cache them per render and reuse
        // across every step / CFG pass (sc-8992).
        let (img_cos, img_sin, txt_cos, txt_sin) = self.rope_tables(frame, h, w, txt_len)?;

        let mask = match text_valid {
            Some(valid) => Some(build_joint_mask(valid, img_len, self.dtype, &self.device)?),
            None => None,
        };

        for block in &self.blocks {
            let (e, hs) = block.forward(
                &hidden,
                &encoder,
                &temb,
                &img_cos,
                &img_sin,
                &txt_cos,
                &txt_sin,
                mask.as_ref(),
            )?;
            encoder = e;
            hidden = hs;
        }

        let hidden = self.norm_out.forward(&hidden, &temb)?;
        self.proj_out.forward(&hidden)
    }
}

/// Additive joint attention mask `[B, 1, 1, img_len + txt_len]`: image tokens always valid; text
/// positions follow `text_valid` (1 = valid). Padded positions get a large-negative additive term so
/// the softmax masks them out (`(valid − 1)·1e9`, valid → 0). Composable; reused by [`crate::dit_train`].
pub fn build_joint_mask(
    text_valid: &Tensor,
    img_len: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let (b, txt_len) = text_valid.dims2()?;
    let img_ones = Tensor::ones((b, img_len), DType::F32, device)?;
    let valid = Tensor::cat(&[&img_ones, &text_valid.to_dtype(DType::F32)?], 1)?;
    let additive = ((valid - 1.0)? * 1e9)?; // valid → 0, invalid → -1e9
    additive
        .reshape((b, 1, 1, img_len + txt_len))?
        .to_dtype(dtype)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_nn::VarMap;

    /// A tiny Lens-shaped DiT config (1 text layer) — enough to build a real `LensTransformer` on CPU
    /// so the public-boundary text-feature guard can be exercised without loading real weights.
    fn tiny_cfg() -> LensDitConfig {
        LensDitConfig {
            patch_size: 2,
            in_channels: 32,
            out_channels: 8,
            num_layers: 1,
            num_heads: 2,
            head_dim: 8,
            inner_dim: 16,
            enc_hidden_dim: 12,
            num_text_layers: 1,
            timestep_channels: 16,
            axes_dims_rope: [2, 2, 4],
            rope_theta: 10_000.0,
        }
    }

    /// The public `forward` must return a typed `Err` (not panic/abort) when the caller supplies the
    /// wrong number of text-feature layers (sc-9025 / F-041).
    #[test]
    fn forward_rejects_wrong_text_feature_count() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let dit = LensTransformer::new(&cfg, vb).unwrap();
        let (h, w) = (2usize, 2usize);
        let img_len = h * w;
        let hidden = Tensor::zeros((1, img_len, cfg.in_channels), DType::F32, &dev).unwrap();
        // cfg.num_text_layers == 1, but pass two layers → must error, not panic.
        let feat = Tensor::zeros((1, 3, cfg.enc_hidden_dim), DType::F32, &dev).unwrap();
        let err = dit
            .forward(&hidden, &[feat.clone(), feat], None, 0.5, 1, h, w)
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("expected 1 text-feature layers, got 2"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn chunked_attention_matches_single_pass() {
        // Per-query-row softmax is independent, so chunking over query rows (forced via a tiny budget)
        // must match the single pass bit-for-bit — the guard for the i32-overflow fix (sc-8983,
        // ported from FLUX.2's sc-5487). Retargeted onto the shared `candle_gen::sdpa_budgeted_bhsd`
        // (sc-9570) with this crate's exact `softmax_last_dim` closure, checked with and without the
        // `[B,1,1,Sk]` additive mask.
        let dev = Device::Cpu;
        let (b, h, s, d) = (2usize, 2usize, 7usize, 4usize);
        let q = Tensor::randn(0f32, 1f32, (b, h, s, d), &dev).unwrap();
        let k = Tensor::randn(0f32, 1f32, (b, h, s, d), &dev).unwrap();
        let v = Tensor::randn(0f32, 1f32, (b, h, s, d), &dev).unwrap();
        let scale = (d as f64).powf(-0.5);
        let sm = |x: &Tensor| softmax_last_dim(x);
        // Mask the last two "text" positions of the second batch item (build_joint_mask shape).
        let valid = Tensor::from_vec(vec![1f32, 1., 1., 1., 1., 1., 0., 0.], (2, 4), &dev).unwrap();
        let mask = build_joint_mask(&valid, s - 4, DType::F32, &dev).unwrap();
        for m in [None, Some(&mask)] {
            // Huge budget → single pass; tiny budget (1) → single-row chunks; a MID-SIZE budget forces
            // multi-row chunks + a remainder (block=3 over s=7 → 3,3,1) — the sc-9116 hardening ask.
            let single =
                candle_gen::sdpa_budgeted_bhsd(&q, &k, &v, scale, m, sm, usize::MAX).unwrap();
            let a = single.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            // budget = b·h·s·block = 2·2·7·3 = 84 → block = 84/(2·2·7) = 3.
            for budget in [1usize, 84] {
                let chunked =
                    candle_gen::sdpa_budgeted_bhsd(&q, &k, &v, scale, m, sm, budget).unwrap();
                assert_eq!(single.dims(), chunked.dims());
                let c = chunked.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                for (x, y) in a.iter().zip(&c) {
                    assert!(
                        (x - y).abs() < 1e-6,
                        "chunked attention diverged (mask={}, budget={budget}): {x} vs {y}",
                        m.is_some()
                    );
                }
            }
        }
    }

    /// **Additive install on the Lens DiT (sc-11105).** A bare-dotted LoRA over two real `attn`
    /// projections in block 0 installs as forward-time residuals: the report counts both, the DiT
    /// forward shifts vs the un-adapted model, and no target is left unresolved — proving the visitor's
    /// canonical paths line up with `adapters::classify_lora_key`. Exercises the packed-tier install
    /// wiring end-to-end on a dense-base fixture (the base-agnostic additive path is byte-equal on a
    /// packed base; the stays-packed property is proven at the `AdaptLinear` unit level).
    #[test]
    fn install_additive_lora_on_dit_applies_and_shifts() {
        use candle_gen::gen_core::{AdapterKind, AdapterSpec};
        use std::collections::HashMap as Map;
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let base = LensTransformer::new(&cfg, vb.clone()).unwrap();
        let mut adapted = LensTransformer::new(&cfg, vb).unwrap();

        let inner = cfg.inner_dim;
        let rank = 2usize;
        let mut map: Map<String, Tensor> = Map::new();
        for proj in ["to_out.0", "to_add_out"] {
            let path = format!("transformer_blocks.0.attn.{proj}");
            map.insert(
                format!("{path}.lora_A.weight"),
                Tensor::randn(0f32, 0.5f32, (rank, inner), &dev).unwrap(),
            );
            map.insert(
                format!("{path}.lora_B.weight"),
                Tensor::randn(0f32, 0.5f32, (inner, rank), &dev).unwrap(),
            );
        }
        let tmp =
            std::env::temp_dir().join(format!("sc11105_lens_{}.safetensors", std::process::id()));
        candle_gen::candle_core::safetensors::save(&map, &tmp).unwrap();
        let report = crate::adapters::install_additive(
            &mut adapted,
            &[AdapterSpec::new(tmp.clone(), 1.0, AdapterKind::Lora)],
        )
        .unwrap();
        std::fs::remove_file(&tmp).ok();
        assert_eq!(
            report.applied, 2,
            "both to_out.0 + to_add_out residuals installed"
        );
        assert!(report.skipped_targets.is_empty(), "no unresolved targets");

        let (h, w) = (2usize, 2usize);
        let img_len = h * w;
        let hidden = Tensor::randn(0f32, 1f32, (1, img_len, cfg.in_channels), &dev).unwrap();
        let feat = Tensor::randn(0f32, 1f32, (1, 3, cfg.enc_hidden_dim), &dev).unwrap();
        let y_base = base
            .forward(&hidden, std::slice::from_ref(&feat), None, 0.5, 1, h, w)
            .unwrap();
        let y_adapt = adapted
            .forward(&hidden, std::slice::from_ref(&feat), None, 0.5, 1, h, w)
            .unwrap();
        let shift = (y_adapt - y_base)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            shift > 1e-5,
            "additive LoRA did not shift the DiT forward ({shift})"
        );
    }

    #[test]
    fn dims_match_checkpoint() {
        let c = LensDitConfig::lens();
        assert_eq!(c.num_layers, 48);
        assert_eq!(c.inner_dim, c.num_heads * c.head_dim); // 1536 = 24·64
        assert_eq!(c.in_channels, 128);
        assert_eq!(c.out_channels, 32);
        assert_eq!(c.patch_size * c.patch_size * c.out_channels, 128); // proj_out width
        assert_eq!(c.mlp_hidden(), 4096); // inner/3·8
        assert_eq!(c.txt_in_dim(), 11520); // 2880·4
        assert_eq!(c.axes_dims_rope.iter().sum::<usize>(), c.head_dim); // 8+28+28 = 64
    }
}
