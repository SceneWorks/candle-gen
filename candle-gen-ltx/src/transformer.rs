//! LTX-2.3 **AudioVideo DiT** (`AVTransformer3DModel`) — port of mlx-gen-ltx `transformer.rs`
//! (`AvDiT`). Each stream: patchify_proj (128→4096) → adaLN-single (timestep→9·dim) + prompt-adaLN
//! (→2·dim) → 48 gated blocks → affine-false LayerNorm output head + 2-row scale-shift → proj_out
//! (→128) velocity.
//!
//! Per-block (gated 9-row `scale_shift_table` + adaLN-single timestep; rows [shift,scale,gate] ×
//! {MSA 0:3, FF 3:6, text-cross-attn 6:9}): MSA self-attn (q/k RMSNorm over full inner, split 3-D
//! RoPE, **2·sigmoid** per-head gate) → prompt-modulated text cross-attn (no RoPE) → tanh-gelu FFN,
//! each adaLN-modulated (`x·(1+scale)+shift`) and gated (`x + out·gate`). Our checkpoint is dense
//! bf16; the whole forward runs bf16, with attention/norms/layernorm computed in f32 for fidelity.

use candle_gen::candle_core::{DType, Device, Result, Tensor, D};
use candle_gen::candle_nn::{ops::rms_norm, ops::sigmoid, ops::softmax_last_dim, VarBuilder};

use crate::config::AvConfig;
use crate::quant::{qlinear, QLinear};
use crate::rope::{apply_split_rope, precompute_split_freqs_nd, time_axis};

/// Packed-detecting biased Linear (sc-9417): loads the MLX-packed AvDiT projection triple when a
/// `{key}.scales` sibling is present (attn `to_{q,k,v,out}` + `ff.proj_in/out` are packed in the
/// `SceneWorks/ltx-2.3-mlx` q4/q8 tiers), else the dense bf16 weight [+ bias] unchanged. Every AvDiT
/// projection carries a bias in the checkpoint.
fn linear(vb: &VarBuilder, key: &str) -> Result<QLinear> {
    qlinear(vb, key, true)
}

/// `x·(1+scale)+shift`; scale/shift `[B,1,inner]` broadcast over the token axis.
fn modulate(x: &Tensor, scale: &Tensor, shift: &Tensor) -> Result<Tensor> {
    x.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(shift)
}

/// `x + out·gate`; gate `[B,1,inner]` broadcasts over `out [B,S,inner]`.
fn gated(x: &Tensor, out: &Tensor, gate: &Tensor) -> Result<Tensor> {
    x + out.broadcast_mul(gate)?
}

/// Weightless RMSNorm (unit weight) over the last axis, in f32.
fn rms_noweight(x: &Tensor, eps: f64) -> Result<Tensor> {
    let xf = x.to_dtype(DType::F32)?.contiguous()?;
    let inv = (xf.sqr()?.mean_keepdim(D::Minus1)? + eps)?
        .sqrt()?
        .recip()?;
    xf.broadcast_mul(&inv)?.to_dtype(x.dtype())
}

/// PixArt sinusoidal timestep embedding (flip_sin_to_cos, cos first), `[N,256]` f32. `ts` is `[N]`
/// f32 (already × timestep_scale_multiplier).
fn timestep_embedding(ts: &Tensor, device: &Device) -> Result<Tensor> {
    const TIME_PROJ_DIM: usize = 256;
    let half = TIME_PROJ_DIM / 2;
    let neg_ln = -(10000f64).ln();
    let freqs: Vec<f32> = (0..half)
        .map(|i| (neg_ln * i as f64 / half as f64).exp() as f32)
        .collect();
    let n = ts.dim(0)?;
    let freq = Tensor::from_vec(freqs, (1, half), device)?;
    let emb = ts.reshape((n, 1))?.broadcast_mul(&freq)?; // (N, half)
    Tensor::cat(&[&emb.cos()?, &emb.sin()?], 1) // (N, 256)
}

/// `table[row] + ts4[:,:,row,:]` for `row in [lo,hi)`; each result `[B,1,inner]`.
fn ada_values(table: &Tensor, ts_emb: &Tensor, lo: usize, hi: usize) -> Result<Vec<Tensor>> {
    let (num, inner) = table.dims2()?;
    let (b, s, _) = ts_emb.dims3()?;
    let ts4 = ts_emb.reshape((b, s, num, inner))?;
    let mut out = Vec::with_capacity(hi - lo);
    for row in lo..hi {
        let trow = table.narrow(0, row, 1)?.reshape((1, 1, inner))?;
        let tsrow = ts4.narrow(2, row, 1)?.squeeze(2)?; // (b,s,inner)
        out.push(trow.broadcast_add(&tsrow)?);
    }
    Ok(out)
}

struct Attention {
    to_q: QLinear,
    to_k: QLinear,
    to_v: QLinear,
    to_out: QLinear,
    q_norm: Tensor,
    k_norm: Tensor,
    gate: QLinear,
    heads: usize,
    dim_head: usize,
    eps: f64,
}

impl Attention {
    /// Load with explicit head dims — the cross-modal + audio attns run at the audio inner dim
    /// (heads 32 × head_dim 64 = 2048), and the q/kv input dims ride on the loaded weight shapes.
    fn load_with_dims(vb: VarBuilder, heads: usize, dim_head: usize, eps: f64) -> Result<Self> {
        Ok(Self {
            to_q: linear(&vb, "to_q")?,
            to_k: linear(&vb, "to_k")?,
            to_v: linear(&vb, "to_v")?,
            to_out: linear(&vb, "to_out.0")?,
            q_norm: vb.get_unchecked("q_norm.weight")?.to_dtype(DType::BF16)?,
            k_norm: vb.get_unchecked("k_norm.weight")?.to_dtype(DType::BF16)?,
            gate: linear(&vb, "to_gate_logits")?,
            heads,
            dim_head,
            eps,
        })
    }

    fn to_heads(&self, x: &Tensor) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        x.reshape((b, s, self.heads, self.dim_head))?
            .transpose(1, 2)
    }

    /// `rope` rotates the query (and the key when `k_rope` is `None`); `k_rope` rotates the key
    /// separately (cross-modal: video-positioned q, audio-positioned k, or vice-versa). `rope ==
    /// None` ⇒ no RoPE on either (text cross-attention). Self-attn when `context` is `None`.
    fn forward(
        &self,
        x: &Tensor,
        context: Option<&Tensor>,
        rope: Option<(&Tensor, &Tensor)>,
        k_rope: Option<(&Tensor, &Tensor)>,
    ) -> Result<Tensor> {
        let ctx = context.unwrap_or(x);
        // q/k RMSNorm over the full inner dim (pre-head), then head reshape.
        let q = rms_norm(
            &self.to_q.forward(x)?.contiguous()?,
            &self.q_norm,
            self.eps as f32,
        )?;
        let k = rms_norm(
            &self.to_k.forward(ctx)?.contiguous()?,
            &self.k_norm,
            self.eps as f32,
        )?;
        let v = self.to_v.forward(ctx)?;
        let mut qh = self.to_heads(&q)?;
        let mut kh = self.to_heads(&k)?;
        let vh = self.to_heads(&v)?;
        if let Some((cos, sin)) = rope {
            qh = apply_split_rope(&qh, cos, sin)?;
            let (kc, ks) = k_rope.unwrap_or((cos, sin));
            kh = apply_split_rope(&kh, kc, ks)?;
        }
        // Attention in f32. i32-overflow guard (sc-9116): the video-DiT self-attn scores `[b,h,s,s]`
        // reach `i32::MAX` at max_size 1280 / long clips (49 frames → 40·40·7 = 11200 tokens →
        // `32·11200² ≈ 4.0e9 > i32::MAX`, growing with clip length), silently corrupting the tail rows
        // on the candle CUDA kernels. The shared budgeted helper chunks over the query rows
        // (byte-identical for common sizes; cross-attn to the fixed text context is a single un-chunked
        // pass). Softmax closure preserves the exact fused `softmax_last_dim`.
        let scale = 1.0 / (self.dim_head as f64).sqrt();
        let qf = qh.to_dtype(DType::F32)?.contiguous()?;
        let kf = kh.to_dtype(DType::F32)?.contiguous()?;
        let vf = vh.to_dtype(DType::F32)?.contiguous()?;
        let out = candle_gen::sdpa_budgeted_bhsd(
            &qf,
            &kf,
            &vf,
            scale,
            None,
            softmax_last_dim,
            candle_gen::ATTN_SCORES_BUDGET,
        )?; // (b,h,s,d)
        let (b, s, _) = x.dims3()?;
        let inner = self.heads * self.dim_head;
        let mut out = out
            .transpose(1, 2)?
            .reshape((b, s, inner))?
            .to_dtype(DType::BF16)?;
        // Per-head gate: 2·sigmoid(logits) (zero-init → identity).
        let logits = self.gate.forward(x)?;
        let gates = (sigmoid(&logits)? * 2.0)?.reshape((b, s, self.heads, 1))?;
        out = out
            .reshape((b, s, self.heads, self.dim_head))?
            .broadcast_mul(&gates)?
            .reshape((b, s, inner))?;
        self.to_out.forward(&out)
    }
}

struct FeedForward {
    proj_in: QLinear,
    proj_out: QLinear,
}

impl FeedForward {
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            proj_in: linear(&vb.pp("net.0"), "proj")?,
            proj_out: linear(&vb.pp("net"), "2")?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // tanh-approx gelu.
        self.proj_out.forward(&self.proj_in.forward(x)?.gelu()?)
    }
}

struct AdaLayerNormSingle {
    ts_lin1: QLinear,
    ts_lin2: QLinear,
    linear: QLinear,
}

impl AdaLayerNormSingle {
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            ts_lin1: linear(&vb.pp("emb.timestep_embedder"), "linear_1")?,
            ts_lin2: linear(&vb.pp("emb.timestep_embedder"), "linear_2")?,
            linear: linear(&vb, "linear")?,
        })
    }

    /// `ts_flat` is `[N]` f32 (already scaled). Returns `(scale_shift [N, coeff·inner], embedded
    /// [N, inner])`, bf16.
    fn forward(&self, ts_flat: &Tensor, device: &Device) -> Result<(Tensor, Tensor)> {
        let proj = timestep_embedding(ts_flat, device)?.to_dtype(DType::BF16)?;
        let h = self.ts_lin1.forward(&proj)?.silu()?;
        let embedded = self.ts_lin2.forward(&h)?;
        let scale_shift = self.linear.forward(&embedded.silu()?)?;
        Ok((scale_shift, embedded))
    }
}

/// Affine-false LayerNorm over the last axis (computed in f32, cast back).
fn layer_norm_noaffine(x: &Tensor, eps: f64) -> Result<Tensor> {
    let xf = x.to_dtype(DType::F32)?.contiguous()?;
    let mean = xf.mean_keepdim(D::Minus1)?;
    let xc = xf.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
    let normed = xc.broadcast_div(&(var + eps)?.sqrt()?)?;
    normed.to_dtype(x.dtype())
}

// =================================================================================================
// AvDiT — the dual-modal AudioVideo DiT (sc-5495). The video stack ([`AvStream`] + gated video
// attns) + an audio stack at the audio inner dim (2048) + bidirectional cross-modal attention. Per
// block: video self+text-CA → audio self+text-CA → cross-modal (a2v updates video, v2a updates audio)
// → video FF → audio FF. Predicts `(video_velocity, audio_velocity)`. Mirrors mlx-gen-ltx `AvDiT`.
// =================================================================================================

/// Precomputed per-stream adaLN timestep tensors (`Stream::prepare`).
struct AvTs {
    ts_emb: Tensor,        // (b,1,9·inner)
    emb_ts: Tensor,        // (b,1,inner)
    prompt_ts: Tensor,     // (b,1,2·inner)
    cross_ss_ts: Tensor,   // (b,1,4·inner)
    cross_gate_ts: Tensor, // (b,1,inner)
}

/// One modality's non-block modules + dims (the video or audio half of the AV DiT).
struct AvStream {
    patchify: QLinear,
    adaln: AdaLayerNormSingle,
    prompt_adaln: AdaLayerNormSingle,
    cross_ss_adaln: AdaLayerNormSingle,
    cross_gate_adaln: AdaLayerNormSingle,
    scale_shift_table: Tensor, // (2, inner) bf16
    proj_out: QLinear,
    inner: usize,
    coeff: usize, // adaLN row count (9 gated)
    eps: f64,
}

impl AvStream {
    #[allow(clippy::too_many_arguments)]
    fn load(
        vb: &VarBuilder,
        patchify: &str,
        adaln: &str,
        prompt: &str,
        cross_ss: &str,
        cross_gate: &str,
        sst: &str,
        proj_out: &str,
        inner: usize,
        eps: f64,
    ) -> Result<Self> {
        Ok(Self {
            patchify: linear(vb, patchify)?,
            adaln: AdaLayerNormSingle::load(vb.pp(adaln))?,
            prompt_adaln: AdaLayerNormSingle::load(vb.pp(prompt))?,
            cross_ss_adaln: AdaLayerNormSingle::load(vb.pp(cross_ss))?,
            cross_gate_adaln: AdaLayerNormSingle::load(vb.pp(cross_gate))?,
            scale_shift_table: vb.get_unchecked(sst)?.to_dtype(DType::BF16)?,
            proj_out: linear(vb, proj_out)?,
            inner,
            coeff: 9,
            eps,
        })
    }

    /// adaLN-single + prompt/cross controllers for a uniform T2V timestep `sigma`.
    fn ts_embeds(&self, sigma: f64, ts_mult: f64, b: usize, device: &Device) -> Result<AvTs> {
        let inner = self.inner;
        let ts_scaled = (sigma * ts_mult) as f32;
        let ts_flat = Tensor::from_vec(vec![ts_scaled; b], (b,), device)?;
        let (ss, emb) = self.adaln.forward(&ts_flat, device)?;
        let (pss, _) = self.prompt_adaln.forward(&ts_flat, device)?;
        let (css, _) = self.cross_ss_adaln.forward(&ts_flat, device)?;
        let (cgs, _) = self.cross_gate_adaln.forward(&ts_flat, device)?;
        Ok(AvTs {
            ts_emb: ss.reshape((b, 1, self.coeff * inner))?,
            emb_ts: emb.reshape((b, 1, inner))?,
            prompt_ts: pss.reshape((b, 1, 2 * inner))?,
            cross_ss_ts: css.reshape((b, 1, 4 * inner))?,
            cross_gate_ts: cgs.reshape((b, 1, inner))?,
        })
    }

    fn output_head(&self, h: &Tensor, emb_ts: &Tensor) -> Result<Tensor> {
        let b = h.dim(0)?;
        let table = self.scale_shift_table.reshape((1, 1, 2, self.inner))?;
        let ss = table.broadcast_add(&emb_ts.reshape((b, 1, 1, self.inner))?)?;
        let shift = ss.narrow(2, 0, 1)?.squeeze(2)?;
        let scale = ss.narrow(2, 1, 1)?.squeeze(2)?;
        let normed = layer_norm_noaffine(h, self.eps)?;
        self.proj_out.forward(&modulate(&normed, &scale, &shift)?)
    }
}

/// Borrowed per-stream args threaded into an [`AvBlock`].
struct AvStreamArgs<'a> {
    ts_emb: &'a Tensor,
    prompt_ts: &'a Tensor,
    context: &'a Tensor,
    cos: &'a Tensor,
    sin: &'a Tensor,
    cross_cos: &'a Tensor,
    cross_sin: &'a Tensor,
    cross_ss_ts: &'a Tensor,
    cross_gate_ts: &'a Tensor,
}

/// `4·scale-shift + 1·gate` cross-modal adaLN values from the pre-split tables → `(scale_a2v,
/// shift_a2v, scale_v2a, shift_v2a, gate)`.
fn av_ca_ada(
    ss_table: &Tensor,
    gate_table: &Tensor,
    ss_ts: &Tensor,
    gate_ts: &Tensor,
) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
    let ss = ada_values(ss_table, ss_ts, 0, 4)?;
    let g = ada_values(gate_table, gate_ts, 0, 1)?;
    Ok((
        ss[0].clone(),
        ss[1].clone(),
        ss[2].clone(),
        ss[3].clone(),
        g[0].clone(),
    ))
}

/// One AudioVideo transformer block (`BasicAVTransformerBlock`).
struct AvBlock {
    attn1: Attention,
    attn2: Attention,
    ff: FeedForward,
    v_sst: Tensor, // (9, 4096)
    v_pst: Tensor, // (2, 4096)
    a_attn1: Attention,
    a_attn2: Attention,
    a_ff: FeedForward,
    a_sst: Tensor, // (9, 2048)
    a_pst: Tensor, // (2, 2048)
    a2v: Attention,
    v2a: Attention,
    ca_audio_ss: Tensor,   // (4, 2048)
    ca_audio_gate: Tensor, // (1, 2048)
    ca_video_ss: Tensor,   // (4, 4096)
    ca_video_gate: Tensor, // (1, 4096)
    eps: f64,
}

impl AvBlock {
    fn load(vb: VarBuilder, cfg: &AvConfig) -> Result<Self> {
        let eps = cfg.video.norm_eps;
        let (vh, vdh) = (cfg.video.num_heads, cfg.video.head_dim);
        let (ah, adh) = (cfg.audio_heads, cfg.audio_head_dim);
        let bf = |k: &str| -> Result<Tensor> { vb.get_unchecked(k)?.to_dtype(DType::BF16) };
        // Split a (5, dim) cross table → 4-row scale-shift + 1-row gate.
        let split = |key: &str| -> Result<(Tensor, Tensor)> {
            let t = bf(key)?;
            Ok((t.narrow(0, 0, 4)?, t.narrow(0, 4, 1)?))
        };
        let (ca_audio_ss, ca_audio_gate) = split("scale_shift_table_a2v_ca_audio")?;
        let (ca_video_ss, ca_video_gate) = split("scale_shift_table_a2v_ca_video")?;
        Ok(Self {
            attn1: Attention::load_with_dims(vb.pp("attn1"), vh, vdh, eps)?,
            attn2: Attention::load_with_dims(vb.pp("attn2"), vh, vdh, eps)?,
            ff: FeedForward::load(vb.pp("ff"))?,
            v_sst: bf("scale_shift_table")?,
            v_pst: bf("prompt_scale_shift_table")?,
            a_attn1: Attention::load_with_dims(vb.pp("audio_attn1"), ah, adh, eps)?,
            a_attn2: Attention::load_with_dims(vb.pp("audio_attn2"), ah, adh, eps)?,
            a_ff: FeedForward::load(vb.pp("audio_ff"))?,
            a_sst: bf("audio_scale_shift_table")?,
            a_pst: bf("audio_prompt_scale_shift_table")?,
            a2v: Attention::load_with_dims(vb.pp("audio_to_video_attn"), ah, adh, eps)?,
            v2a: Attention::load_with_dims(vb.pp("video_to_audio_attn"), ah, adh, eps)?,
            ca_audio_ss,
            ca_audio_gate,
            ca_video_ss,
            ca_video_gate,
            eps,
        })
    }

    /// Self-attn (RoPE) → prompt-modulated text cross-attention (no RoPE), for one modality.
    fn self_and_text(
        &self,
        x: &Tensor,
        attn1: &Attention,
        attn2: &Attention,
        sst: &Tensor,
        pst: &Tensor,
        a: &AvStreamArgs,
    ) -> Result<Tensor> {
        let msa = ada_values(sst, a.ts_emb, 0, 3)?;
        let norm = modulate(&rms_noweight(x, self.eps)?, &msa[1], &msa[0])?;
        let attn = attn1.forward(&norm, None, Some((a.cos, a.sin)), None)?;
        let x = gated(x, &attn, &msa[2])?;

        let p = ada_values(pst, a.prompt_ts, 0, 2)?;
        let context = modulate(a.context, &p[1], &p[0])?;

        let ca = ada_values(sst, a.ts_emb, 6, 9)?;
        let norm_ca = modulate(&rms_noweight(&x, self.eps)?, &ca[1], &ca[0])?;
        let cross = attn2.forward(&norm_ca, Some(&context), None, None)?;
        gated(&x, &cross, &ca[2])
    }

    fn feed_forward(
        &self,
        x: &Tensor,
        ff: &FeedForward,
        sst: &Tensor,
        ts_emb: &Tensor,
    ) -> Result<Tensor> {
        let mlp = ada_values(sst, ts_emb, 3, 6)?;
        let norm = modulate(&rms_noweight(x, self.eps)?, &mlp[1], &mlp[0])?;
        let ff_out = ff.forward(&norm)?;
        gated(x, &ff_out, &mlp[2])
    }

    /// Joint forward `(vx, ax)` → `(vx, ax)`.
    fn forward(
        &self,
        vx: &Tensor,
        ax: &Tensor,
        v: &AvStreamArgs,
        a: &AvStreamArgs,
    ) -> Result<(Tensor, Tensor)> {
        let mut vx =
            self.self_and_text(vx, &self.attn1, &self.attn2, &self.v_sst, &self.v_pst, v)?;
        let mut ax = self.self_and_text(
            ax,
            &self.a_attn1,
            &self.a_attn2,
            &self.a_sst,
            &self.a_pst,
            a,
        )?;

        // Cross-modal — both directions read the pre-update rms_norm snapshots.
        let vx_n3 = rms_noweight(&vx, self.eps)?;
        let ax_n3 = rms_noweight(&ax, self.eps)?;
        let (sca_a2v, sha_a2v, sca_v2a, sha_v2a, gate_v2a) = av_ca_ada(
            &self.ca_audio_ss,
            &self.ca_audio_gate,
            a.cross_ss_ts,
            a.cross_gate_ts,
        )?;
        let (scv_a2v, shv_a2v, scv_v2a, shv_v2a, gate_a2v) = av_ca_ada(
            &self.ca_video_ss,
            &self.ca_video_gate,
            v.cross_ss_ts,
            v.cross_gate_ts,
        )?;

        // Audio-to-Video: Q from video (video cross-PE), K/V from audio (audio cross-PE).
        let a2v = self.a2v.forward(
            &modulate(&vx_n3, &scv_a2v, &shv_a2v)?,
            Some(&modulate(&ax_n3, &sca_a2v, &sha_a2v)?),
            Some((v.cross_cos, v.cross_sin)),
            Some((a.cross_cos, a.cross_sin)),
        )?;
        vx = gated(&vx, &a2v, &gate_a2v)?;

        // Video-to-Audio: Q from audio (audio cross-PE), K/V from video (video cross-PE).
        let v2a = self.v2a.forward(
            &modulate(&ax_n3, &sca_v2a, &sha_v2a)?,
            Some(&modulate(&vx_n3, &scv_v2a, &shv_v2a)?),
            Some((a.cross_cos, a.cross_sin)),
            Some((v.cross_cos, v.cross_sin)),
        )?;
        ax = gated(&ax, &v2a, &gate_v2a)?;

        vx = self.feed_forward(&vx, &self.ff, &self.v_sst, v.ts_emb)?;
        ax = self.feed_forward(&ax, &self.a_ff, &self.a_sst, a.ts_emb)?;
        Ok((vx, ax))
    }
}

/// Per-render RoPE-table cache (sc-8992 / F-012). LTX builds **four** split-RoPE `(cos, sin)` tables
/// per forward (video self, video↔time cross, audio self, audio↔time cross) — ~4.7M trig evals — all
/// derived solely from the fixed `video_grid`/`audio_grid` position grids, not σ / the latents. So they
/// are identical across every denoise step. Cache them keyed on the grids' host contents (a few hundred
/// floats, negligible vs the trig) and rebuild only when the grids change. Byte-identical to recomputing.
struct AvRopeCache {
    video_grid: Vec<f32>,
    audio_grid: Vec<f32>,
    v_cos: Tensor,
    v_sin: Tensor,
    vc_cos: Tensor,
    vc_sin: Tensor,
    a_cos: Tensor,
    a_sin: Tensor,
    ac_cos: Tensor,
    ac_sin: Tensor,
}

/// The LTX-2.3 **AudioVideo** DiT. Predicts `(video_velocity, audio_velocity)` from the two latent
/// token streams + shared text conditioning.
pub struct AvDiT {
    video: AvStream,
    audio: AvStream,
    blocks: Vec<AvBlock>,
    cfg: AvConfig,
    device: Device,
    /// `Mutex` (not `RefCell`): the DiT is shared as `Arc<AvDiT>` and must stay `Send + Sync`.
    rope_cache: std::sync::Mutex<Option<AvRopeCache>>,
}

impl AvDiT {
    /// Build from a VarBuilder rooted at `model.diffusion_model.`.
    pub fn new(vb: VarBuilder, cfg: &AvConfig) -> Result<Self> {
        let device = vb.device().clone();
        let video = AvStream::load(
            &vb,
            "patchify_proj",
            "adaln_single",
            "prompt_adaln_single",
            "av_ca_video_scale_shift_adaln_single",
            "av_ca_a2v_gate_adaln_single",
            "scale_shift_table",
            "proj_out",
            cfg.video.inner_dim(),
            cfg.video.norm_eps,
        )?;
        let audio = AvStream::load(
            &vb,
            "audio_patchify_proj",
            "audio_adaln_single",
            "audio_prompt_adaln_single",
            "av_ca_audio_scale_shift_adaln_single",
            "av_ca_v2a_gate_adaln_single",
            "audio_scale_shift_table",
            "audio_proj_out",
            cfg.audio_inner(),
            cfg.video.norm_eps,
        )?;
        let mut blocks = Vec::with_capacity(cfg.video.num_layers);
        for i in 0..cfg.video.num_layers {
            blocks.push(AvBlock::load(
                vb.pp(format!("transformer_blocks.{i}")),
                cfg,
            )?);
        }
        Ok(Self {
            video,
            audio,
            blocks,
            cfg: cfg.clone(),
            device,
            rope_cache: std::sync::Mutex::new(None),
        })
    }

    /// Build (or reuse) the four split-RoPE tables for this render's fixed position grids (sc-8992).
    /// Recomputed only when `video_grid`/`audio_grid` change; otherwise the Arc-backed handles are
    /// cloned. Construction is identical to computing it inline, so every step is byte-identical.
    #[allow(clippy::type_complexity)]
    fn rope_tables(
        &self,
        video_grid: &Tensor,
        audio_grid: &Tensor,
    ) -> Result<[(Tensor, Tensor); 4]> {
        let vkey = video_grid.flatten_all()?.to_vec1::<f32>()?;
        let akey = audio_grid.flatten_all()?.to_vec1::<f32>()?;
        let mut guard = candle_gen::lock_recover(&self.rope_cache);
        if let Some(c) = guard.as_ref() {
            if c.video_grid == vkey && c.audio_grid == akey {
                return Ok([
                    (c.v_cos.clone(), c.v_sin.clone()),
                    (c.vc_cos.clone(), c.vc_sin.clone()),
                    (c.a_cos.clone(), c.a_sin.clone()),
                    (c.ac_cos.clone(), c.ac_sin.clone()),
                ]);
            }
        }
        let device = &self.device;
        let theta = self.cfg.video.rope_theta;
        // Self RoPE (video 3-axis @4096, audio 1-axis @2048) + cross RoPE (time axis @2048 both).
        let (v_cos, v_sin) = precompute_split_freqs_nd(
            video_grid,
            self.cfg.video.inner_dim(),
            theta,
            &self.cfg.video.rope_max_pos,
            self.cfg.video.num_heads,
            device,
        )?;
        let (vc_cos, vc_sin) = precompute_split_freqs_nd(
            &time_axis(video_grid)?,
            self.cfg.cross_inner,
            theta,
            &[self.cfg.cross_max_pos],
            self.cfg.video.num_heads,
            device,
        )?;
        let (a_cos, a_sin) = precompute_split_freqs_nd(
            audio_grid,
            self.cfg.audio_inner(),
            theta,
            &[self.cfg.audio_max_pos],
            self.cfg.audio_heads,
            device,
        )?;
        let (ac_cos, ac_sin) = precompute_split_freqs_nd(
            &time_axis(audio_grid)?,
            self.cfg.cross_inner,
            theta,
            &[self.cfg.cross_max_pos],
            self.cfg.audio_heads,
            device,
        )?;
        *guard = Some(AvRopeCache {
            video_grid: vkey,
            audio_grid: akey,
            v_cos: v_cos.clone(),
            v_sin: v_sin.clone(),
            vc_cos: vc_cos.clone(),
            vc_sin: vc_sin.clone(),
            a_cos: a_cos.clone(),
            a_sin: a_sin.clone(),
            ac_cos: ac_cos.clone(),
            ac_sin: ac_sin.clone(),
        });
        Ok([
            (v_cos, v_sin),
            (vc_cos, vc_sin),
            (a_cos, a_sin),
            (ac_cos, ac_sin),
        ])
    }

    /// Joint velocity forward.
    ///
    /// * `*_latent` — `[B, S, 128]` patchified tokens.
    /// * `sigma` — scalar σ (uniform T2V timestep, shared by both streams).
    /// * `*_context` — text embeddings (video 4096, audio 2048).
    /// * `*_grid` — position grids (video `[1,3,Tv,2]`, audio `[1,1,Ta,2]`).
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        video_latent: &Tensor,
        audio_latent: &Tensor,
        sigma: f64,
        video_context: &Tensor,
        audio_context: &Tensor,
        video_grid: &Tensor,
        audio_grid: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let device = &self.device;
        let b = video_latent.dim(0)?;
        let ts_mult = self.cfg.video.timestep_scale_multiplier;

        let v_ts = self.video.ts_embeds(sigma, ts_mult, b, device)?;
        let a_ts = self.audio.ts_embeds(sigma, ts_mult, b, device)?;

        // The four split-RoPE tables are step-invariant (fixed position grids), so cache them per
        // render and reuse across every step (sc-8992).
        let [(v_cos, v_sin), (vc_cos, vc_sin), (a_cos, a_sin), (ac_cos, ac_sin)] =
            self.rope_tables(video_grid, audio_grid)?;

        let mut vx = self
            .video
            .patchify
            .forward(&video_latent.to_dtype(DType::BF16)?)?;
        let mut ax = self
            .audio
            .patchify
            .forward(&audio_latent.to_dtype(DType::BF16)?)?;
        let v_ctx = video_context.to_dtype(DType::BF16)?;
        let a_ctx = audio_context.to_dtype(DType::BF16)?;

        let va = AvStreamArgs {
            ts_emb: &v_ts.ts_emb,
            prompt_ts: &v_ts.prompt_ts,
            context: &v_ctx,
            cos: &v_cos,
            sin: &v_sin,
            cross_cos: &vc_cos,
            cross_sin: &vc_sin,
            cross_ss_ts: &v_ts.cross_ss_ts,
            cross_gate_ts: &v_ts.cross_gate_ts,
        };
        let aa = AvStreamArgs {
            ts_emb: &a_ts.ts_emb,
            prompt_ts: &a_ts.prompt_ts,
            context: &a_ctx,
            cos: &a_cos,
            sin: &a_sin,
            cross_cos: &ac_cos,
            cross_sin: &ac_sin,
            cross_ss_ts: &a_ts.cross_ss_ts,
            cross_gate_ts: &a_ts.cross_gate_ts,
        };

        for block in &self.blocks {
            let (nv, na) = block.forward(&vx, &ax, &va, &aa)?;
            vx = nv;
            ax = na;
        }
        let v_vel = self.video.output_head(&vx, &v_ts.emb_ts)?;
        let a_vel = self.audio.output_head(&ax, &a_ts.emb_ts)?;
        Ok((v_vel, a_vel))
    }
}
