//! The FLUX.2 **MMDiT** transformer. Port of `mlx-gen-flux2`'s `transformer.rs`, run in candle f32.
//!
//! Shape anchors (klein-9b): `inner_dim = 4096`, `in/out_channels = 128`, `joint_attention_dim =
//! 12288`, `num_heads = 32`, `head_dim = 128`, 8 double (joint) blocks + 24 single (fused parallel)
//! blocks. The joint sequence order is **`[txt, img]`** in every concat / RoPE / attention. The
//! double block returns `(txt, img)`.
//!
//! Parity-load-bearing details (verified against the fork): LayerNorms are affine-free with
//! `eps = 1e-6`; the per-head q/k RMSNorm uses `eps = 1e-5`; `modulate = (1+scale)·norm + shift`
//! (strong f32 1); modulation is **global** (produced once from `temb`, shared across all blocks of
//! a stream); the RoPE is interleaved (see [`crate::pos_embed`]); the timestep fed in is the **scaled
//! sigma `σ·1000`** and the velocity is applied with a negative `dt` (no negation, in the pipeline).

use candle_gen::candle_core::{DType, Device, Result, Tensor, D};
use candle_gen::candle_nn::{ops::softmax_last_dim, rms_norm, Module, RmsNorm, VarBuilder};
use candle_gen::gen_core::Quant;

use crate::config::Flux2Config;
use crate::pos_embed::Flux2PosEmbed;
use crate::quant::{rms_norm_to, QLinear};

const LN_EPS: f64 = 1e-6;
const RMS_EPS: f64 = 1e-5;

/// Affine-free LayerNorm over the last axis (eps 1e-6), in f32.
fn layer_norm(x: &Tensor) -> Result<Tensor> {
    let x = x.to_dtype(DType::F32)?;
    let mean = x.mean_keepdim(D::Minus1)?;
    let xc = x.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
    xc.broadcast_div(&(var + LN_EPS)?.sqrt()?)
}

/// `(1 + scale)·norm + shift`, broadcasting modulation `[B,1,D]` over `[B,S,D]`.
fn modulate(norm: &Tensor, scale: &Tensor, shift: &Tensor) -> Result<Tensor> {
    let one_plus = (scale + 1.0)?;
    norm.broadcast_mul(&one_plus)?.broadcast_add(shift)
}

/// `x + gate·y`, broadcasting gate `[B,1,D]`.
fn gated(x: &Tensor, gate: &Tensor, y: &Tensor) -> Result<Tensor> {
    x + y.broadcast_mul(gate)?
}

/// SwiGLU: split the last axis in half, `silu(a)·b`.
fn swiglu(x: &Tensor) -> Result<Tensor> {
    let half = x.dim(D::Minus1)? / 2;
    let a = x.narrow(D::Minus1, 0, half)?;
    let b = x.narrow(D::Minus1, half, half)?;
    a.silu()? * b
}

/// Sinusoidal timestep embedding `[1, dim]`: `[cos(args) | sin(args)]`, `args = t · 10000^{-i/half}`
/// (diffusers `flip_sin_to_cos = True`, cos first).
fn timestep_embedding(t: f32, dim: usize, device: &Device) -> Result<Tensor> {
    let half = dim / 2;
    let mut cos = vec![0f32; half];
    let mut sin = vec![0f32; half];
    let ln10000 = 10000f32.ln();
    for i in 0..half {
        let freq = (-ln10000 * i as f32 / half as f32).exp();
        let arg = t * freq;
        cos[i] = arg.cos();
        sin[i] = arg.sin();
    }
    let cos = Tensor::from_vec(cos, (1, half), device)?;
    let sin = Tensor::from_vec(sin, (1, half), device)?;
    Tensor::cat(&[&cos, &sin], D::Minus1)
}

/// SDPA over `[B,H,S,D]` q/k/v → `[B, S, H·D]`. scale = `head_dim^-0.5`. Delegates to the shared
/// i32-overflow-safe [`candle_gen::sdpa_budgeted_bhsd`] (sc-9570), which chunks over the query rows once
/// the `[B,H,Sq,Sk]` scores tensor would exceed [`candle_gen::ATTN_SCORES_BUDGET`] (the candle CUDA
/// i32-index limit). The `softmax_last_dim` closure keeps the exact fused softmax; each query row's
/// softmax is over all keys and independent, so the chunked result is byte-identical to the single pass —
/// only the long edit/joint sequences trip it. This crate does the head-merge transpose/reshape here.
fn attention(q: &Tensor, k: &Tensor, v: &Tensor, head_dim: usize) -> Result<Tensor> {
    let (b, _h, s, _d) = q.dims4()?;
    let scale = (head_dim as f64).powf(-0.5);
    let o = candle_gen::sdpa_budgeted_bhsd(
        q,
        k,
        v,
        scale,
        None,
        softmax_last_dim,
        candle_gen::ATTN_SCORES_BUDGET,
    )?; // [B,H,S,D]
    let (_b, h, _s, d) = o.dims4()?;
    o.transpose(1, 2)?.reshape((b, s, h * d))
}

/// Reshape `[B,S,inner]` → `[B,H,S,head_dim]`, applying per-head RMSNorm (over head_dim) when `norm`
/// is given (q/k), none for v.
fn to_heads(x: &Tensor, heads: usize, head_dim: usize, norm: Option<&RmsNorm>) -> Result<Tensor> {
    let (b, s, _) = x.dims3()?;
    let x = x.reshape((b, s, heads, head_dim))?;
    let x = match norm {
        Some(n) => n.forward(&x)?,
        None => x,
    };
    x.transpose(1, 2)?.contiguous() // [B,H,S,head_dim]
}

/// A sinusoidal-scalar embedding MLP: `timestep_embedding → linear_1 → silu → linear_2` → `[1, inner]`.
/// Shared by the timestep and (dev) guidance branches of `time_guidance_embed`.
struct SinEmbed {
    linear_1: QLinear,
    linear_2: QLinear,
    channels: usize,
}

impl SinEmbed {
    fn new(cfg: &Flux2Config, vb: VarBuilder) -> Result<Self> {
        let inner = cfg.inner_dim();
        Ok(Self {
            linear_1: QLinear::linear_detect(cfg.timestep_channels, inner, &vb, "linear_1", false)?,
            linear_2: QLinear::linear_detect(inner, inner, &vb, "linear_2", false)?,
            channels: cfg.timestep_channels,
        })
    }

    fn forward(&self, scalar: f32, device: &Device) -> Result<Tensor> {
        let emb = timestep_embedding(scalar, self.channels, device)?;
        let h = self.linear_1.forward(&emb)?.silu()?;
        self.linear_2.forward(&h)
    }

    fn quantize_onto(&mut self, quant: Quant, device: &Device) -> Result<()> {
        self.linear_1.quantize_onto(quant, device)?;
        self.linear_2.quantize_onto(quant, device)
    }
}

/// FLUX.2 `time_guidance_embed`: the timestep embedder (`timestep_embedder.*`, always present) plus —
/// on the guidance-distilled **dev** checkpoint only — a guidance embedder (`guidance_embedder.*`).
/// `temb = time_emb(σ·1000) + guidance_emb(guidance·1000)` (diffusers `Flux2TimestepGuidanceEmbeddings`,
/// no pooled-CLIP term); klein has no guidance embedder, so `temb` is the timestep embedding alone.
struct TimeGuidanceEmbed {
    timestep: SinEmbed,
    guidance: Option<SinEmbed>,
}

impl TimeGuidanceEmbed {
    fn new(cfg: &Flux2Config, vb: VarBuilder) -> Result<Self> {
        let timestep = SinEmbed::new(cfg, vb.pp("timestep_embedder"))?;
        // The guidance embedder exists only on dev; gate on the weight (mirrors the mlx `w.get(...)`
        // presence check) so a klein checkpoint loads without looking for absent keys.
        let guidance = if vb.contains_tensor("guidance_embedder.linear_1.weight") {
            Some(SinEmbed::new(cfg, vb.pp("guidance_embedder"))?)
        } else {
            None
        };
        Ok(Self { timestep, guidance })
    }

    /// `timestep` is fed as σ·1000 (the caller scales it). `guidance` is the raw guidance scale (e.g.
    /// 4.0); it is scaled ×1000 here (the diffusers `guidance = guidance * 1000` step) and added only
    /// when this is a dev transformer. A `Some(guidance)` on klein (no embedder) is silently ignored.
    fn forward(&self, timestep: f32, guidance: Option<f32>, device: &Device) -> Result<Tensor> {
        let mut temb = self.timestep.forward(timestep, device)?;
        if let (Some(g), Some(gemb)) = (guidance, &self.guidance) {
            temb = (temb + gemb.forward(g * 1000.0, device)?)?;
        }
        Ok(temb)
    }

    fn quantize_onto(&mut self, quant: Quant, device: &Device) -> Result<()> {
        self.timestep.quantize_onto(quant, device)?;
        if let Some(g) = self.guidance.as_mut() {
            g.quantize_onto(quant, device)?;
        }
        Ok(())
    }
}

/// Global modulation: `silu(temb) → linear → split 3·sets` → `sets × (shift, scale, gate)` (each
/// `[B,1,inner]`).
struct Modulation {
    linear: QLinear,
    sets: usize,
}

impl Modulation {
    fn new(cfg: &Flux2Config, sets: usize, vb: VarBuilder) -> Result<Self> {
        let inner = cfg.inner_dim();
        Ok(Self {
            linear: QLinear::linear_detect(inner, 3 * sets * inner, &vb, "linear", false)?,
            sets,
        })
    }

    fn quantize_onto(&mut self, quant: Quant, device: &Device) -> Result<()> {
        self.linear.quantize_onto(quant, device)
    }

    fn forward(&self, temb: &Tensor) -> Result<Vec<(Tensor, Tensor, Tensor)>> {
        let m = self.linear.forward(&temb.silu()?)?.unsqueeze(1)?; // [B,1,3·sets·inner]
        let inner = m.dim(D::Minus1)? / (3 * self.sets);
        let mut out = Vec::with_capacity(self.sets);
        for i in 0..self.sets {
            let base = 3 * i * inner;
            let shift = m.narrow(D::Minus1, base, inner)?;
            let scale = m.narrow(D::Minus1, base + inner, inner)?;
            let gate = m.narrow(D::Minus1, base + 2 * inner, inner)?;
            out.push((shift, scale, gate));
        }
        Ok(out)
    }
}

/// Joint attention for a double block: separate img/txt q/k/v with per-head q/k RMSNorm, attention
/// over the concatenated `[txt, img]` sequence with interleaved RoPE, split back.
struct DoubleAttention {
    to_q: QLinear,
    to_k: QLinear,
    to_v: QLinear,
    to_out: QLinear,
    norm_q: RmsNorm,
    norm_k: RmsNorm,
    add_q: QLinear,
    add_k: QLinear,
    add_v: QLinear,
    to_add_out: QLinear,
    norm_added_q: RmsNorm,
    norm_added_k: RmsNorm,
    heads: usize,
    head_dim: usize,
}

impl DoubleAttention {
    fn new(cfg: &Flux2Config, vb: VarBuilder) -> Result<Self> {
        let inner = cfg.inner_dim();
        let hd = cfg.head_dim;
        Ok(Self {
            to_q: QLinear::linear_detect(inner, inner, &vb, "to_q", false)?,
            to_k: QLinear::linear_detect(inner, inner, &vb, "to_k", false)?,
            to_v: QLinear::linear_detect(inner, inner, &vb, "to_v", false)?,
            // `to_out.0`: the packed `.scales`/`.biases` siblings sit under the same dotted prefix, so
            // pass the full `to_out.0` base to `linear_detect` (never `.pp("0")` past the sibling — the
            // sc-8670 remap trap the story flags).
            to_out: QLinear::linear_detect(inner, inner, &vb, "to_out.0", false)?,
            norm_q: rms_norm(hd, RMS_EPS, vb.pp("norm_q"))?,
            norm_k: rms_norm(hd, RMS_EPS, vb.pp("norm_k"))?,
            add_q: QLinear::linear_detect(inner, inner, &vb, "add_q_proj", false)?,
            add_k: QLinear::linear_detect(inner, inner, &vb, "add_k_proj", false)?,
            add_v: QLinear::linear_detect(inner, inner, &vb, "add_v_proj", false)?,
            to_add_out: QLinear::linear_detect(inner, inner, &vb, "to_add_out", false)?,
            norm_added_q: rms_norm(hd, RMS_EPS, vb.pp("norm_added_q"))?,
            norm_added_k: rms_norm(hd, RMS_EPS, vb.pp("norm_added_k"))?,
            heads: cfg.num_heads,
            head_dim: hd,
        })
    }

    fn quantize_onto(&mut self, quant: Quant, device: &Device) -> Result<()> {
        for l in [
            &mut self.to_q,
            &mut self.to_k,
            &mut self.to_v,
            &mut self.to_out,
            &mut self.add_q,
            &mut self.add_k,
            &mut self.add_v,
            &mut self.to_add_out,
        ] {
            l.quantize_onto(quant, device)?;
        }
        self.norm_q = rms_norm_to(&self.norm_q, RMS_EPS, device)?;
        self.norm_k = rms_norm_to(&self.norm_k, RMS_EPS, device)?;
        self.norm_added_q = rms_norm_to(&self.norm_added_q, RMS_EPS, device)?;
        self.norm_added_k = rms_norm_to(&self.norm_added_k, RMS_EPS, device)?;
        Ok(())
    }

    /// `norm_img` / `norm_txt`: the modulated, normed streams. Returns `(img_out, txt_out)`.
    fn forward(
        &self,
        norm_img: &Tensor,
        norm_txt: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (h, hd) = (self.heads, self.head_dim);
        let txt_seq = norm_txt.dim(1)?;

        // img stream q/k/v
        let iq = to_heads(&self.to_q.forward(norm_img)?, h, hd, Some(&self.norm_q))?;
        let ik = to_heads(&self.to_k.forward(norm_img)?, h, hd, Some(&self.norm_k))?;
        let iv = to_heads(&self.to_v.forward(norm_img)?, h, hd, None)?;
        // txt stream q/k/v
        let tq = to_heads(
            &self.add_q.forward(norm_txt)?,
            h,
            hd,
            Some(&self.norm_added_q),
        )?;
        let tk = to_heads(
            &self.add_k.forward(norm_txt)?,
            h,
            hd,
            Some(&self.norm_added_k),
        )?;
        let tv = to_heads(&self.add_v.forward(norm_txt)?, h, hd, None)?;

        // Concat [txt, img] along the sequence axis, apply RoPE to the full q/k.
        let q = Tensor::cat(&[&tq, &iq], 2)?;
        let k = Tensor::cat(&[&tk, &ik], 2)?;
        let v = Tensor::cat(&[&tv, &iv], 2)?;
        let q = Flux2PosEmbed::apply(&q, cos, sin)?;
        let k = Flux2PosEmbed::apply(&k, cos, sin)?;

        let o = attention(&q, &k, &v, hd)?; // [B, txt_seq+img_seq, inner]
        let txt_out = o.narrow(1, 0, txt_seq)?;
        let img_out = o.narrow(1, txt_seq, o.dim(1)? - txt_seq)?;
        let txt_out = self.to_add_out.forward(&txt_out.contiguous()?)?;
        let img_out = self.to_out.forward(&img_out.contiguous()?)?;
        Ok((img_out, txt_out))
    }
}

/// SwiGLU feed-forward: `linear_in → swiglu → linear_out`.
struct FeedForward {
    linear_in: QLinear,
    linear_out: QLinear,
}

impl FeedForward {
    fn new(in_dim: usize, hidden: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            linear_in: QLinear::linear_detect(in_dim, 2 * hidden, &vb, "linear_in", false)?,
            linear_out: QLinear::linear_detect(hidden, in_dim, &vb, "linear_out", false)?,
        })
    }

    fn quantize_onto(&mut self, quant: Quant, device: &Device) -> Result<()> {
        self.linear_in.quantize_onto(quant, device)?;
        self.linear_out.quantize_onto(quant, device)
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = swiglu(&self.linear_in.forward(x)?)?;
        self.linear_out.forward(&h)
    }
}

struct DoubleBlock {
    attn: DoubleAttention,
    ff: FeedForward,
    ff_context: FeedForward,
}

impl DoubleBlock {
    fn new(cfg: &Flux2Config, vb: VarBuilder) -> Result<Self> {
        let inner = cfg.inner_dim();
        let ff_hidden = (cfg.mlp_ratio * inner as f32) as usize;
        Ok(Self {
            attn: DoubleAttention::new(cfg, vb.pp("attn"))?,
            ff: FeedForward::new(inner, ff_hidden, vb.pp("ff"))?,
            ff_context: FeedForward::new(inner, ff_hidden, vb.pp("ff_context"))?,
        })
    }

    fn quantize_onto(&mut self, quant: Quant, device: &Device) -> Result<()> {
        self.attn.quantize_onto(quant, device)?;
        self.ff.quantize_onto(quant, device)?;
        self.ff_context.quantize_onto(quant, device)
    }

    /// Returns `(txt, img)` (note order).
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        img: &Tensor,
        txt: &Tensor,
        img_mod: &[(Tensor, Tensor, Tensor)],
        txt_mod: &[(Tensor, Tensor, Tensor)],
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (shift_msa, scale_msa, gate_msa) = &img_mod[0];
        let (shift_mlp, scale_mlp, gate_mlp) = &img_mod[1];
        let (c_shift_msa, c_scale_msa, c_gate_msa) = &txt_mod[0];
        let (c_shift_mlp, c_scale_mlp, c_gate_mlp) = &txt_mod[1];

        let norm_img = modulate(&layer_norm(img)?, scale_msa, shift_msa)?;
        let norm_txt = modulate(&layer_norm(txt)?, c_scale_msa, c_shift_msa)?;
        let (img_attn, txt_attn) = self.attn.forward(&norm_img, &norm_txt, cos, sin)?;
        let mut img = gated(img, gate_msa, &img_attn)?;
        let mut txt = gated(txt, c_gate_msa, &txt_attn)?;

        let norm_img2 = modulate(&layer_norm(&img)?, scale_mlp, shift_mlp)?;
        let img_ff = self.ff.forward(&norm_img2)?;
        img = gated(&img, gate_mlp, &img_ff)?;

        let norm_txt2 = modulate(&layer_norm(&txt)?, c_scale_mlp, c_shift_mlp)?;
        let txt_ff = self.ff_context.forward(&norm_txt2)?;
        txt = gated(&txt, c_gate_mlp, &txt_ff)?;

        Ok((txt, img))
    }
}

/// Single (fused parallel attention + SwiGLU) block: one projection produces q/k/v and the MLP input.
struct SingleBlock {
    to_qkv_mlp: QLinear,
    to_out: QLinear,
    norm_q: RmsNorm,
    norm_k: RmsNorm,
    inner: usize,
    heads: usize,
    head_dim: usize,
}

impl SingleBlock {
    fn new(cfg: &Flux2Config, vb: VarBuilder) -> Result<Self> {
        let inner = cfg.inner_dim();
        let mlp_hidden = cfg.single_mlp_hidden();
        let proj_out = 3 * inner + 2 * mlp_hidden;
        // The single block's projections nest under `attn.` in the diffusers checkpoint.
        let attn = vb.pp("attn");
        Ok(Self {
            to_qkv_mlp: QLinear::linear_detect(inner, proj_out, &attn, "to_qkv_mlp_proj", false)?,
            to_out: QLinear::linear_detect(inner + mlp_hidden, inner, &attn, "to_out", false)?,
            norm_q: rms_norm(cfg.head_dim, RMS_EPS, attn.pp("norm_q"))?,
            norm_k: rms_norm(cfg.head_dim, RMS_EPS, attn.pp("norm_k"))?,
            inner,
            heads: cfg.num_heads,
            head_dim: cfg.head_dim,
        })
    }

    fn quantize_onto(&mut self, quant: Quant, device: &Device) -> Result<()> {
        self.to_qkv_mlp.quantize_onto(quant, device)?;
        self.to_out.quantize_onto(quant, device)?;
        self.norm_q = rms_norm_to(&self.norm_q, RMS_EPS, device)?;
        self.norm_k = rms_norm_to(&self.norm_k, RMS_EPS, device)?;
        Ok(())
    }

    fn forward(
        &self,
        hidden: &Tensor,
        m: &(Tensor, Tensor, Tensor),
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let (shift, scale, gate) = m;
        let norm = modulate(&layer_norm(hidden)?, scale, shift)?;
        let proj = self.to_qkv_mlp.forward(&norm)?;
        let inner = self.inner;
        let q = proj.narrow(D::Minus1, 0, inner)?;
        let k = proj.narrow(D::Minus1, inner, inner)?;
        let v = proj.narrow(D::Minus1, 2 * inner, inner)?;
        let mlp = proj.narrow(D::Minus1, 3 * inner, proj.dim(D::Minus1)? - 3 * inner)?;

        let q = to_heads(&q, self.heads, self.head_dim, Some(&self.norm_q))?;
        let k = to_heads(&k, self.heads, self.head_dim, Some(&self.norm_k))?;
        let v = to_heads(&v, self.heads, self.head_dim, None)?;
        let q = Flux2PosEmbed::apply(&q, cos, sin)?;
        let k = Flux2PosEmbed::apply(&k, cos, sin)?;
        let attn = attention(&q, &k, &v, self.head_dim)?; // [B,S,inner]

        let mlp = swiglu(&mlp)?; // [B,S,mlp_hidden]
        let cat = Tensor::cat(&[&attn, &mlp], D::Minus1)?;
        let attn_output = self.to_out.forward(&cat)?;
        gated(hidden, gate, &attn_output)
    }
}

/// AdaLayerNorm-Continuous output head: `silu(temb) → linear → (scale, shift)`, then
/// `(1+scale)·LN(x) + shift`.
struct NormOut {
    linear: QLinear,
}

impl NormOut {
    fn new(cfg: &Flux2Config, vb: VarBuilder) -> Result<Self> {
        let inner = cfg.inner_dim();
        Ok(Self {
            linear: QLinear::linear_detect(inner, 2 * inner, &vb, "linear", false)?,
        })
    }

    fn quantize_onto(&mut self, quant: Quant, device: &Device) -> Result<()> {
        self.linear.quantize_onto(quant, device)
    }

    fn forward(&self, x: &Tensor, temb: &Tensor) -> Result<Tensor> {
        let p = self.linear.forward(&temb.silu()?)?.unsqueeze(1)?; // [B,1,2·inner]
        let inner = p.dim(D::Minus1)? / 2;
        let scale = p.narrow(D::Minus1, 0, inner)?; // scale first
        let shift = p.narrow(D::Minus1, inner, inner)?;
        modulate(&layer_norm(x)?, &scale, &shift)
    }
}

/// The FLUX.2 MMDiT.
/// Per-render RoPE-table cache (sc-8992 / F-012). The `[txt, img]` `(cos, sin)` tables depend only on
/// the fixed `img_ids`/`txt_ids` geometry (not σ / the current latent), so they are identical across
/// every denoise step (×2 under CFG). Cache them keyed on the ids and rebuild only when the geometry
/// changes; the stored handles are Arc-cloned on a hit (cheap). Byte-identical to recomputing.
struct Flux2RopeCache {
    img_ids: Vec<[i64; 4]>,
    txt_ids: Vec<[i64; 4]>,
    cos: Tensor,
    sin: Tensor,
}

pub struct Flux2Transformer {
    x_embedder: QLinear,
    context_embedder: QLinear,
    time_embed: TimeGuidanceEmbed,
    mod_img: Modulation,
    mod_txt: Modulation,
    mod_single: Modulation,
    double_blocks: Vec<DoubleBlock>,
    single_blocks: Vec<SingleBlock>,
    norm_out: NormOut,
    proj_out: QLinear,
    pos_embed: Flux2PosEmbed,
    device: Device,
    /// `Mutex` (not `RefCell`) because the transformer is shared as `Arc<Flux2Transformer>` and must
    /// stay `Send + Sync`. Contended only within a single render; the lock is held only to swap handles.
    rope_cache: std::sync::Mutex<Option<Flux2RopeCache>>,
}

impl Flux2Transformer {
    pub fn new(cfg: &Flux2Config, vb: VarBuilder) -> Result<Self> {
        let inner = cfg.inner_dim();
        let mut double_blocks = Vec::with_capacity(cfg.num_double_layers);
        for i in 0..cfg.num_double_layers {
            double_blocks.push(DoubleBlock::new(cfg, vb.pp("transformer_blocks").pp(i))?);
        }
        let mut single_blocks = Vec::with_capacity(cfg.num_single_layers);
        for i in 0..cfg.num_single_layers {
            single_blocks.push(SingleBlock::new(
                cfg,
                vb.pp("single_transformer_blocks").pp(i),
            )?);
        }
        Ok(Self {
            x_embedder: QLinear::linear_detect(cfg.in_channels, inner, &vb, "x_embedder", false)?,
            context_embedder: QLinear::linear_detect(
                cfg.joint_attention_dim,
                inner,
                &vb,
                "context_embedder",
                false,
            )?,
            time_embed: TimeGuidanceEmbed::new(cfg, vb.pp("time_guidance_embed"))?,
            mod_img: Modulation::new(cfg, 2, vb.pp("double_stream_modulation_img"))?,
            mod_txt: Modulation::new(cfg, 2, vb.pp("double_stream_modulation_txt"))?,
            mod_single: Modulation::new(cfg, 1, vb.pp("single_stream_modulation"))?,
            double_blocks,
            single_blocks,
            norm_out: NormOut::new(cfg, vb.pp("norm_out"))?,
            proj_out: QLinear::linear_detect(inner, cfg.out_channels, &vb, "proj_out", false)?,
            pos_embed: Flux2PosEmbed::new(cfg),
            device: vb.device().clone(),
            rope_cache: std::sync::Mutex::new(None),
        })
    }

    /// Build (or reuse) the `[txt, img]` RoPE `(cos, sin)` tables for this render's fixed geometry
    /// (sc-8992). Recomputed only when `img_ids`/`txt_ids` change vs the cached entry; otherwise the
    /// Arc-backed handles are cloned. The construction is identical to computing it inline, so every
    /// step remains byte-identical.
    fn rope_tables(&self, img_ids: &[[i64; 4]], txt_ids: &[[i64; 4]]) -> Result<(Tensor, Tensor)> {
        let mut guard = candle_gen::lock_recover(&self.rope_cache);
        if let Some(c) = guard.as_ref() {
            if c.img_ids.as_slice() == img_ids && c.txt_ids.as_slice() == txt_ids {
                return Ok((c.cos.clone(), c.sin.clone()));
            }
        }
        let (txt_cos, txt_sin) = self.pos_embed.cos_sin(txt_ids, &self.device)?;
        let (img_cos, img_sin) = self.pos_embed.cos_sin(img_ids, &self.device)?;
        let cos = Tensor::cat(&[&txt_cos, &img_cos], 0)?;
        let sin = Tensor::cat(&[&txt_sin, &img_sin], 0)?;
        *guard = Some(Flux2RopeCache {
            img_ids: img_ids.to_vec(),
            txt_ids: txt_ids.to_vec(),
            cos: cos.clone(),
            sin: sin.clone(),
        });
        Ok((cos, sin))
    }

    /// Fold every projection to `Q4_0`/`Q8_0` **onto `device`** and carry the full-precision norms
    /// there too (CPU-staged dev quant path, sc-7457). Call after building the dense transformer on
    /// the CPU; afterwards the transformer's compute device is `device` (the GPU). Idempotent per
    /// `QLinear`. The affine-free LayerNorms hold no weights, and `pos_embed` builds its RoPE tables on
    /// `self.device` at forward time, so updating `self.device` is enough to move them.
    pub fn quantize(&mut self, quant: Quant, device: &Device) -> Result<()> {
        self.x_embedder.quantize_onto(quant, device)?;
        self.context_embedder.quantize_onto(quant, device)?;
        self.time_embed.quantize_onto(quant, device)?;
        self.mod_img.quantize_onto(quant, device)?;
        self.mod_txt.quantize_onto(quant, device)?;
        self.mod_single.quantize_onto(quant, device)?;
        for b in &mut self.double_blocks {
            b.quantize_onto(quant, device)?;
        }
        for b in &mut self.single_blocks {
            b.quantize_onto(quant, device)?;
        }
        self.norm_out.quantize_onto(quant, device)?;
        self.proj_out.quantize_onto(quant, device)?;
        self.device = device.clone();
        // The RoPE cache pins tables on the old device; drop it so the next forward rebuilds on
        // `device` (sc-8992). Empty at load anyway, but keep this correct if quantize ever runs after
        // a warmup forward.
        *candle_gen::lock_recover(&self.rope_cache) = None;
        Ok(())
    }

    /// Predict velocity. `hidden_states` `[B, seq_img, 128]`, `encoder_hidden_states`
    /// `[B, seq_txt, joint]`, `img_ids`/`txt_ids` the 4-axis position ids, `timestep` = `σ·1000`.
    /// `guidance` is the raw embedded-guidance scale for the guidance-distilled **dev** path (e.g.
    /// 4.0), or `None` for klein (distilled / true-CFG); it is ignored unless this transformer carries
    /// the dev guidance embedder.
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        img_ids: &[[i64; 4]],
        txt_ids: &[[i64; 4]],
        timestep: f32,
        guidance: Option<f32>,
    ) -> Result<Tensor> {
        self.forward_inner(
            hidden_states,
            encoder_hidden_states,
            img_ids,
            txt_ids,
            timestep,
            guidance,
            None,
        )
    }

    /// FLUX.2-dev Fun-Controlnet-Union forward (sc-7460/sc-2292): [`Self::forward`] plus a VACE control
    /// branch. `control = (branch, control_context, scale)` — the branch's per-block hints are computed
    /// once from the post-embedder image+caption streams and added to the base image stream after each
    /// base double block in `branch.places`, scaled by `scale`. At `scale = 0` the result is
    /// byte-identical to the base forward (the parity self-check).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_control(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        img_ids: &[[i64; 4]],
        txt_ids: &[[i64; 4]],
        timestep: f32,
        guidance: Option<f32>,
        control: (&Flux2ControlBranch, &Tensor, f32),
    ) -> Result<Tensor> {
        self.forward_inner(
            hidden_states,
            encoder_hidden_states,
            img_ids,
            txt_ids,
            timestep,
            guidance,
            Some(control),
        )
    }

    /// Shared body behind [`forward`](Self::forward) and [`forward_with_control`](Self::forward_with_control).
    /// `control` (when `Some`) injects the VACE control hints (dev pose, sc-7460); `None` is the plain
    /// base forward. The control hints are computed once from the post-embedder streams before the base
    /// double-block loop, then added into the base image stream after the mapped base double blocks.
    #[allow(clippy::too_many_arguments)]
    fn forward_inner(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        img_ids: &[[i64; 4]],
        txt_ids: &[[i64; 4]],
        timestep: f32,
        guidance: Option<f32>,
        control: Option<(&Flux2ControlBranch, &Tensor, f32)>,
    ) -> Result<Tensor> {
        let temb = self.time_embed.forward(timestep, guidance, &self.device)?;
        let mut img = self
            .x_embedder
            .forward(&hidden_states.to_dtype(DType::F32)?)?;
        let mut txt = self
            .context_embedder
            .forward(&encoder_hidden_states.to_dtype(DType::F32)?)?;

        // RoPE table over the [txt, img] sequence. Step-invariant (fixed ids), so cached per render
        // and reused across every step / CFG pass (sc-8992).
        let (cos, sin) = self.rope_tables(img_ids, txt_ids)?;

        let img_mod = self.mod_img.forward(&temb)?;
        let txt_mod = self.mod_txt.forward(&temb)?;

        // VACE control hints (sc-7460): computed once from the post-embedder image+caption streams,
        // before the base double-block loop (the fork's `forward_control`), then injected per block.
        let hints = match control {
            Some((branch, cc, _)) => {
                Some(branch.forward_control(&img, &txt, cc, &img_mod, &txt_mod, &cos, &sin)?)
            }
            None => None,
        };

        for (idx, block) in self.double_blocks.iter().enumerate() {
            let (t, i) = block.forward(&img, &txt, &img_mod, &txt_mod, &cos, &sin)?;
            txt = t;
            img = i;
            // Add the control hint into the base image stream (`img + hints[n]·scale`) at the mapped
            // base double blocks. `scale = 0` → `+0` → byte-identical to the base forward.
            if let (Some(hints), Some((branch, _, scale))) = (&hints, &control) {
                if let Some(n) = branch.hint_index(idx) {
                    img = (&img + (&hints[n] * (*scale as f64))?)?;
                }
            }
        }

        let txt_seq = txt.dim(1)?;
        let mut hidden = Tensor::cat(&[&txt, &img], 1)?;
        let single_mod = self.mod_single.forward(&temb)?;
        for block in &self.single_blocks {
            hidden = block.forward(&hidden, &single_mod[0], &cos, &sin)?;
        }

        let img_seq = hidden.dim(1)? - txt_seq;
        let img_out = hidden.narrow(1, txt_seq, img_seq)?;
        let img_out = self.norm_out.forward(&img_out.contiguous()?, &temb)?;
        self.proj_out.forward(&img_out)
    }
}

// ---- sc-7460: FLUX.2-dev Fun-Controlnet-Union (VACE-style strict pose) -------------------------
//
// Port of `alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union` (`videox_fun/models/flux2_transformer2d_control.py`),
// mirroring the merged mlx-gen control branch (sc-2292): a VACE-style ControlNet on the FIRST 4 of
// dev's 8 base double blocks (`control_layers = range(0, num_double_layers, 2) = [0, 2, 4, 6]`). A
// `control_img_in` patch embedder maps the packed control context (control latent 128 + zero mask 4 +
// zero inpaint latent 128 = 260) into the inner dim; N control double blocks thread an internal
// `(c_image, txt)` pair — block 0 seeds `c = before_proj(c) + img_embed`, each runs a full base
// double-block forward and emits `after_proj(c)` as its hint — and the hints are added into the base
// image stream after the matching base double blocks, scaled by `control_context_scale`. The control
// blocks reuse the base `double_stream_modulation_{img,txt}` + RoPE (passed through); the threaded
// `txt` is local to the control stack (the base caption stream is untouched).

/// In-features of `control_img_in` = the packed control-context width: control latent 128, a zero
/// inpaint mask 4, and a zero inpaint latent 128 → 260, per `pipeline_flux2_control.py`
/// (`torch.concat([control_latents, mask_condition, inpaint_latent], dim=2)`). The union ControlNet's
/// pose-only layout zeros the mask + inpaint, leaving the VAE-encoded pose skeleton in `control_latents`.
pub const CONTROL_IN_DIM: usize = 260;

/// One VACE control block: a full FLUX.2 double block (its own attn / ff / ff_context weights) plus
/// the `after_proj` hint projection (every block) and `before_proj` (block 0 only) seeding the control
/// branch from the base image embedding. Port of `Flux2ControlTransformerBlock`. All three control
/// Linears are bias-carrying.
struct Flux2ControlBlock {
    base: DoubleBlock,
    /// `before_proj(c) + img_embed` seeds block 0 (`None` for the rest).
    before_proj: Option<QLinear>,
    /// `after_proj(c)` — the per-block hint added into the base image stream.
    after_proj: QLinear,
}

impl Flux2ControlBlock {
    fn new(cfg: &Flux2Config, vb: VarBuilder, has_before_proj: bool) -> Result<Self> {
        let inner = cfg.inner_dim();
        // The control block's attn/ff/ff_context keys match a base double block 1:1 (diffusers naming,
        // `attn.to_out.0` read natively by `DoubleBlock`); load dense, quantized in place after load.
        let base = DoubleBlock::new(cfg, vb.clone())?;
        let after_proj = QLinear::linear_detect(inner, inner, &vb, "after_proj", true)?;
        let before_proj = if has_before_proj {
            Some(QLinear::linear_detect(
                inner,
                inner,
                &vb,
                "before_proj",
                true,
            )?)
        } else {
            None
        };
        Ok(Self {
            base,
            before_proj,
            after_proj,
        })
    }

    /// Quantize the block's base double block + the `after_proj`/`before_proj` projections **onto**
    /// `device` (all `% 32 == 0`). The only control Linear left dense is `control_img_in` (260
    /// in-features), handled in [`Flux2ControlBranch::quantize`].
    fn quantize_onto(&mut self, quant: Quant, device: &Device) -> Result<()> {
        self.base.quantize_onto(quant, device)?;
        self.after_proj.quantize_onto(quant, device)?;
        if let Some(bp) = &mut self.before_proj {
            bp.quantize_onto(quant, device)?;
        }
        Ok(())
    }
}

/// The FLUX.2-dev Fun-Controlnet-Union control branch (sc-7460): the `control_img_in` patch embedder
/// plus the N control blocks injecting hints into the base double blocks at `control_layers`. Built
/// from the Fun-Controlnet-Union checkpoint and driven by [`Flux2Transformer::forward_with_control`].
pub struct Flux2ControlBranch {
    /// `control_img_in`: 260 → inner. Kept **dense** (260 in-features is not a multiple of the
    /// Q4_0/Q8_0 block size 32), matching the fork's `nn.quantize` predicate. Bias-carrying.
    control_img_in: QLinear,
    blocks: Vec<Flux2ControlBlock>,
    /// Base double-block indices each control block injects into (`control_layers`); `places[n]` is
    /// the base index for hint `n` (`[0, 2, 4, 6]` for dev's 8 double blocks).
    places: Vec<usize>,
}

impl Flux2ControlBranch {
    /// Build from the Fun-Controlnet-Union checkpoint VarBuilder. Keys are un-prefixed for a real
    /// checkpoint (`control_img_in.*`, `control_transformer_blocks.{i}.*`). `control_layers =
    /// range(0, num_double_layers, 2)`.
    pub fn new(cfg: &Flux2Config, vb: VarBuilder) -> Result<Self> {
        let inner = cfg.inner_dim();
        let places = cfg.control_layer_places();
        let control_img_in =
            QLinear::linear_detect(CONTROL_IN_DIM, inner, &vb, "control_img_in", true)?;
        let mut blocks = Vec::with_capacity(places.len());
        for i in 0..places.len() {
            blocks.push(Flux2ControlBlock::new(
                cfg,
                vb.pp("control_transformer_blocks").pp(i),
                i == 0,
            )?);
        }
        Ok(Self {
            control_img_in,
            blocks,
            places,
        })
    }

    /// Quantize the control blocks (+ their `after_proj`/`before_proj`) **onto** `device`;
    /// `control_img_in` stays dense and is moved to `device`.
    pub fn quantize(&mut self, quant: Quant, device: &Device) -> Result<()> {
        self.control_img_in.to_device(device)?;
        for b in &mut self.blocks {
            b.quantize_onto(quant, device)?;
        }
        Ok(())
    }

    /// The hint index injected at base double-block `idx`, or `None`.
    fn hint_index(&self, idx: usize) -> Option<usize> {
        self.places.iter().position(|&p| p == idx)
    }

    /// Run the control stack → per-block hints (the fork's `forward_control`). `img_embed`/`txt_embed`
    /// are the post-embedder base streams; `control_context` is the packed 260-ch control context;
    /// `img_mod`/`txt_mod`/`cos`/`sin` are the shared base double-stream modulation + RoPE (the control
    /// blocks reuse the base modulation, per the fork). The threaded `txt` is local to the control
    /// stack — only the image-stream hints leave.
    #[allow(clippy::too_many_arguments)]
    fn forward_control(
        &self,
        img_embed: &Tensor,
        txt_embed: &Tensor,
        control_context: &Tensor,
        img_mod: &[(Tensor, Tensor, Tensor)],
        txt_mod: &[(Tensor, Tensor, Tensor)],
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Vec<Tensor>> {
        let mut c = self
            .control_img_in
            .forward(&control_context.to_dtype(DType::F32)?)?;
        let mut txt = txt_embed.clone();
        let mut hints = Vec::with_capacity(self.blocks.len());
        for (i, block) in self.blocks.iter().enumerate() {
            if i == 0 {
                let bp = block.before_proj.as_ref().ok_or_else(|| {
                    candle_gen::candle_core::Error::Msg(
                        "flux2 control block 0 is missing before_proj".into(),
                    )
                })?;
                c = (&bp.forward(&c)? + img_embed)?;
            }
            // The base double block returns `(txt, img)`; the control image stream is `img` (`new_c`).
            let (new_txt, new_c) = block.base.forward(&c, &txt, img_mod, txt_mod, cos, sin)?;
            hints.push(block.after_proj.forward(&new_c)?);
            c = new_c;
            txt = new_txt;
        }
        Ok(hints)
    }
}

/// The FLUX.2-dev base MMDiT + its Fun-Controlnet-Union control branch (sc-7460). Composes the
/// parity-proven [`Flux2Transformer`] with a [`Flux2ControlBranch`]; [`forward`](Self::forward)
/// threads the control context + scale, and [`quantize`](Self::quantize) packs both onto the device.
pub struct Flux2ControlTransformer {
    base: Flux2Transformer,
    branch: Flux2ControlBranch,
}

impl Flux2ControlTransformer {
    pub fn new(base: Flux2Transformer, branch: Flux2ControlBranch) -> Self {
        Self { base, branch }
    }

    /// Quantize the base DiT + the control branch **onto** `device` (the dev CPU-stage path). The base
    /// is staged dense in CPU RAM and quantized onto the GPU; the control overlay is small and may be
    /// loaded dense on the GPU and quantized in place — either way `device` is the compute device.
    pub fn quantize(&mut self, quant: Quant, device: &Device) -> Result<()> {
        self.base.quantize(quant, device)?;
        self.branch.quantize(quant, device)?;
        Ok(())
    }

    /// Control forward: latent `[B, seq, in]` + text embeds + ids + timestep (σ·1000) + embedded
    /// `guidance` + packed `control_context` (260ch, same image seq as the latent) + `scale`. Returns
    /// the velocity `[B, seq_img, out]`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        img_ids: &[[i64; 4]],
        txt_ids: &[[i64; 4]],
        timestep: f32,
        guidance: Option<f32>,
        control_context: &Tensor,
        control_context_scale: f32,
    ) -> Result<Tensor> {
        self.base.forward_with_control(
            hidden_states,
            encoder_hidden_states,
            img_ids,
            txt_ids,
            timestep,
            guidance,
            (&self.branch, control_context, control_context_scale),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestep_embedding_is_cos_then_sin_and_pos_zero_is_one_zero() {
        let emb = timestep_embedding(0.0, 256, &Device::Cpu).unwrap();
        assert_eq!(emb.dims(), &[1, 256]);
        let v = emb.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // t=0: all args 0 → cos 1 (first half), sin 0 (second half).
        for c in &v[..128] {
            assert!((c - 1.0).abs() < 1e-6);
        }
        for s in &v[128..] {
            assert!(s.abs() < 1e-6);
        }
    }

    #[test]
    fn swiglu_halves_last_dim() {
        let x = Tensor::ones((1, 2, 8), DType::F32, &Device::Cpu).unwrap();
        let y = swiglu(&x).unwrap();
        assert_eq!(y.dims(), &[1, 2, 4]);
    }

    #[test]
    fn chunked_attention_matches_single_pass() {
        // Per-query-row softmax is independent, so chunking over query rows (forced via a tiny budget)
        // must match the single pass bit-for-bit — the guard for the i32-overflow fix (sc-5487).
        // Retargeted onto the shared `candle_gen::sdpa_budgeted_bhsd` (sc-9570) with this crate's exact
        // `softmax_last_dim` closure and no mask.
        let dev = Device::Cpu;
        let (b, h, s, d) = (1usize, 2usize, 7usize, 4usize);
        let q = Tensor::randn(0f32, 1f32, (b, h, s, d), &dev).unwrap();
        let k = Tensor::randn(0f32, 1f32, (b, h, s, d), &dev).unwrap();
        let v = Tensor::randn(0f32, 1f32, (b, h, s, d), &dev).unwrap();
        let scale = (d as f64).powf(-0.5);
        let sm = |x: &Tensor| softmax_last_dim(x);
        // Huge budget → single pass; tiny budget (1) → chunked into single-row blocks; a MID-SIZE
        // budget forces multi-row chunks + a remainder (block=3 over s=7 → 3,3,1) — the sc-9116
        // test-hardening ask (a single-row chunk can hide an off-by-one the multi-row path would trip).
        let single =
            candle_gen::sdpa_budgeted_bhsd(&q, &k, &v, scale, None, sm, usize::MAX).unwrap();
        let a = single.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // budget = b·h·s·block = 1·2·7·3 = 42 → block = 42/(1·2·7) = 3.
        for budget in [1usize, 42] {
            let chunked =
                candle_gen::sdpa_budgeted_bhsd(&q, &k, &v, scale, None, sm, budget).unwrap();
            assert_eq!(single.dims(), chunked.dims());
            let c = chunked.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            for (x, y) in a.iter().zip(&c) {
                assert!(
                    (x - y).abs() < 1e-6,
                    "chunked attention diverged at budget {budget}: {x} vs {y}"
                );
            }
        }
    }

    #[test]
    fn modulate_is_one_plus_scale() {
        let norm = Tensor::ones((1, 2, 4), DType::F32, &Device::Cpu).unwrap();
        let scale = Tensor::zeros((1, 1, 4), DType::F32, &Device::Cpu).unwrap();
        let shift = Tensor::ones((1, 1, 4), DType::F32, &Device::Cpu).unwrap();
        // (1+0)*1 + 1 = 2
        let out = modulate(&norm, &scale, &shift).unwrap();
        let v = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for x in v {
            assert!((x - 2.0).abs() < 1e-6);
        }
    }
}
