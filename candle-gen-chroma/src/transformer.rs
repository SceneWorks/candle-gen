//! The Chroma DiT (`ChromaTransformer2DModel`) — the candle (Windows/CUDA) port of
//! `mlx-gen-chroma`'s `transformer.rs`, run in candle-native f32.
//!
//! The FLUX MMDiT skeleton (19 dual + 38 single blocks, FluxPosEmbed RoPE, gelu-tanh FFN) with the
//! Chroma deltas:
//! - the distilled-guidance modulation generator —
//!   `ChromaCombinedTimestepTextProjEmbeddings` + `ChromaApproximator` → `pooled_temb [B,
//!   mod_index_len, inner]`. ALL per-block modulation is *sliced* from this table; there is no
//!   per-block modulation linear (**pruned adaLN**).
//! - **T5-XXL-only** conditioning (no CLIP / no pooled), QK-norm RMS eps **1e-6** (NOT FLUX's 1e-5),
//!   gelu-tanh FFN, and the pruned `norm_out` + `proj_out`.
//!
//! **v1 (sc-5484) deliberately omits the MMDiT attention mask.** The mlx provider pads the T5
//! sequence to 512 and masks it; the candle slice instead encodes the prompt at its natural length
//! (candle's `T5EncoderModel` exposes no key-padding mask, and padding-then-unmasked would be *worse*
//! parity than not padding), so every token in the sequence is real and attended — the mask would be
//! all-ones. The cross-backend f32 floor (~1e-3) absorbs the one-extra-pad-token nuance. Matching the
//! candle FLUX slice's "T5 unmasked" choice.
//!
//! LoRA/LoKr adapters and *on-the-fly* Q4/Q8 quantization of a dense tier are NOT wired in this slice
//! (rejected at load), so the `Adaptable*`/`quantize` machinery the mlx provider carries is dropped.
//! **Pre-quantized MLX-packed tiers ARE loaded, though** (sc-9409, `SceneWorks/chroma1-*-mlx` q4/q8):
//! every projection is a [`crate::quant::QLinear`] built via `linear_detect_gs`, which packed-detects
//! the MLX `.scales` sibling and loads straight from the packed parts (no dense staging), or falls back
//! to the plain dense biased load (dense tier, unchanged) — see [`crate::quant`]. The T5-XXL encoder
//! and VAE ship dense in every tier, so only the DiT threads through the packed seam.

use candle_gen::candle_core::{DType, Device, Result, Tensor, D};
use candle_gen::candle_nn::{ops::softmax_last_dim, rms_norm, Module, RmsNorm, VarBuilder};

use crate::config::ChromaTransformerConfig;
use crate::quant::QLinear;
use crate::rope::RopeTable;

/// AdaLayerNorm LayerNorm epsilon (all pruned norms + `norm_out`, `elementwise_affine=False`).
const LN_EPS: f64 = 1e-6;
/// QK-norm RMS epsilon. Chroma's `FluxAttention(eps=1e-6)` — **NOT** FLUX's 1e-5.
const QK_RMS_EPS: f64 = 1e-6;
/// RMSNorm epsilon for the Approximator norms — torch `nn.RMSNorm(hidden)` with `eps=None` resolves
/// to `torch.finfo(float32).eps` (the f32 path).
const APPROX_RMS_EPS: f64 = 1.192_092_9e-7;
/// Sinusoid frequency base (diffusers `get_timestep_embedding` `max_period`).
const MAX_PERIOD: f64 = 10000.0;

// ============================ leaf helpers ============================

/// Affine-free LayerNorm over the last axis (eps 1e-6), in f32.
fn layer_norm(x: &Tensor) -> Result<Tensor> {
    let x = x.to_dtype(DType::F32)?;
    let mean = x.mean_keepdim(D::Minus1)?;
    let xc = x.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
    xc.broadcast_div(&(var + LN_EPS)?.sqrt()?)
}

/// adaLN affine `normed·(1+scale) + shift`, broadcasting modulation `[B,1,D]` over `[B,S,D]`.
fn modulate(normed: &Tensor, scale: &Tensor, shift: &Tensor) -> Result<Tensor> {
    let one_plus = (scale + 1.0)?;
    normed.broadcast_mul(&one_plus)?.broadcast_add(shift)
}

/// `x + gate·y`, broadcasting gate `[B,1,D]` over `[B,S,D]`.
fn gated(x: &Tensor, gate: &Tensor, y: &Tensor) -> Result<Tensor> {
    x + y.broadcast_mul(gate)?
}

/// The `j`-th modulation row of a `[B,K,inner]` table, as `[B,1,inner]` (broadcastable over seq).
fn row(table: &Tensor, j: usize) -> Result<Tensor> {
    table.narrow(1, j, 1)
}

/// `len` contiguous rows of `[B,K,inner]` from `start`, as `[B,len,inner]`.
fn rows(table: &Tensor, start: usize, len: usize) -> Result<Tensor> {
    table.narrow(1, start, len)
}

/// `get_timestep_embedding(timesteps, dim, flip_sin_to_cos=True, downscale_freq_shift)` (diffusers),
/// in f32, for a vector of `timesteps` `[N]` → `[N, dim]`. `flip_sin_to_cos=True` ⇒ order `[cos, sin]`.
/// `dim` even. Frequencies are computed host-side, then the table is materialized on `device`.
fn timestep_embedding(
    timesteps: &Tensor,
    dim: usize,
    downscale_freq_shift: f64,
    device: &Device,
) -> Result<Tensor> {
    let half = dim / 2;
    let factor = -MAX_PERIOD.ln() / (half as f64 - downscale_freq_shift);
    let freqs: Vec<f32> = (0..half)
        .map(|i| (i as f64 * factor).exp() as f32)
        .collect();
    let freqs = Tensor::from_vec(freqs, (1, half), device)?; // [1, half]
    let n = timesteps.elem_count();
    let t = timesteps.to_dtype(DType::F32)?.reshape((n, 1))?; // [N, 1]
    let emb = t.broadcast_mul(&freqs)?; // [N, half]
    Tensor::cat(&[&emb.cos()?, &emb.sin()?], D::Minus1) // [N, dim]
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

/// SDPA over `[B,H,S,head_dim]` q/k/v → `[B, S, H·head_dim]`. scale = `head_dim^-0.5`. No mask (the
/// natural-length T5 encode leaves no padded positions to mask — see the module docs). Delegates to the
/// shared i32-overflow-safe [`candle_gen::sdpa_budgeted_bhsd`] (sc-9570), which chunks over the query
/// rows once the `[B,H,Sq,Sk]` scores tensor would exceed [`candle_gen::ATTN_SCORES_BUDGET`] (the candle
/// CUDA i32-index limit). The `softmax_last_dim` closure keeps the exact fused softmax; each query row's
/// softmax is over all keys and independent, so the chunked result is byte-identical to the single pass —
/// only the large renders trip it. This crate does the head-merge transpose/reshape here.
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
    )?; // [B,H,S,head_dim]
    let (_b, h, _s, d) = o.dims4()?;
    o.transpose(1, 2)?.reshape((b, s, h * d))
}

/// gelu-tanh feed-forward `lin2(gelu(lin1(x)))` (diffusers FLUX `FeedForward`, `net.0.proj` /
/// `net.2`, `mlp_ratio = 4`).
struct FeedForward {
    lin1: QLinear,
    lin2: QLinear,
}

impl FeedForward {
    fn new(inner: usize, vb: VarBuilder, gs: usize) -> Result<Self> {
        let hidden = 4 * inner;
        Ok(Self {
            lin1: QLinear::linear_detect_gs(inner, hidden, &vb, "net.0.proj", true, gs)?,
            lin2: QLinear::linear_detect_gs(hidden, inner, &vb, "net.2", true, gs)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // candle's `gelu` is the tanh approximation (matches diffusers `gelu-approximate`).
        self.lin2.forward(&self.lin1.forward(x)?.gelu()?)
    }
}

// ============================ embeddings + Approximator ============================

/// `ChromaCombinedTimestepTextProjEmbeddings` — builds the Approximator input vector (parameter-free
/// apart from the precomputed `mod_proj` index embedding).
struct TimestepTextProj {
    /// `approximator_num_channels / 4` — the per-(time|guidance) embedding width.
    num_channels: usize,
    /// `timestep_embedding(idx·1000, 2·num_channels)` for `idx ∈ 0..mod_index_len`, `[N, 2·nc]`.
    mod_proj: Tensor,
    device: Device,
}

impl TimestepTextProj {
    fn new(cfg: &ChromaTransformerConfig, device: &Device) -> Result<Self> {
        let num_channels = cfg.approximator_num_channels / 4;
        let n = cfg.mod_index_len();
        let idx: Vec<f32> = (0..n).map(|i| (i as f32) * 1000.0).collect();
        let idx = Tensor::from_vec(idx, n, device)?;
        let mod_proj = timestep_embedding(&idx, 2 * num_channels, 0.0, device)?;
        Ok(Self {
            num_channels,
            mod_proj,
            device: device.clone(),
        })
    }

    /// `timestep` already scaled (`t·1000`), shape `[B]`. Returns `input_vec [B, mod_index_len, 4·nc]`.
    fn forward(&self, timestep: &Tensor) -> Result<Tensor> {
        let b = timestep.elem_count();
        let n = self.mod_proj.dim(0)?;
        let nc2 = 2 * self.num_channels;
        let time = timestep_embedding(timestep, self.num_channels, 0.0, &self.device)?; // [B, nc]
        let zeros = Tensor::zeros(b, DType::F32, &self.device)?;
        let guid = timestep_embedding(&zeros, self.num_channels, 0.0, &self.device)?; // [B, nc]
        let tg = Tensor::cat(&[&time, &guid], D::Minus1)?.reshape((b, 1, nc2))?; // [B,1,2nc]
        let tg = tg.broadcast_as((b, n, nc2))?.contiguous()?;
        let mp = self
            .mod_proj
            .reshape((1, n, nc2))?
            .broadcast_as((b, n, nc2))?
            .contiguous()?;
        Tensor::cat(&[&tg, &mp], D::Minus1) // [B, N, 4nc]
    }
}

/// `ChromaApproximator` — `in_proj` then `n_layers` residual blocks
/// `x = x + linear_2(silu(linear_1(rms_norm(x))))`, then `out_proj`.
struct Approximator {
    in_proj: QLinear,
    layers: Vec<(QLinear, QLinear)>,
    norms: Vec<RmsNorm>,
    out_proj: QLinear,
}

impl Approximator {
    fn load(vb: VarBuilder, cfg: &ChromaTransformerConfig, gs: usize) -> Result<Self> {
        let vb = vb.pp("distilled_guidance_layer");
        let in_dim = cfg.approximator_num_channels; // 4·nc = 64
        let hidden = cfg.approximator_hidden_dim; // 5120
        let inner = cfg.inner_dim(); // 3072
                                     // The Approximator ships **dense** even in the packed tiers, so these `linear_detect_gs` calls
                                     // all take the dense arm today; routing them through packed-detect anyway (sc-9486 superset)
                                     // keeps a single, uniform loader across the DiT and makes a future packed-Approximator tier a
                                     // no-op change here.
        let mut layers = Vec::with_capacity(cfg.approximator_layers);
        let mut norms = Vec::with_capacity(cfg.approximator_layers);
        for i in 0..cfg.approximator_layers {
            let lvb = vb.pp("layers").pp(i);
            layers.push((
                QLinear::linear_detect_gs(hidden, hidden, &lvb, "linear_1", true, gs)?,
                QLinear::linear_detect_gs(hidden, hidden, &lvb, "linear_2", true, gs)?,
            ));
            norms.push(rms_norm(hidden, APPROX_RMS_EPS, vb.pp("norms").pp(i))?);
        }
        Ok(Self {
            in_proj: QLinear::linear_detect_gs(in_dim, hidden, &vb, "in_proj", true, gs)?,
            layers,
            norms,
            out_proj: QLinear::linear_detect_gs(hidden, inner, &vb, "out_proj", true, gs)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = self.in_proj.forward(x)?;
        for ((lin1, lin2), norm) in self.layers.iter().zip(self.norms.iter()) {
            let n = norm.forward(&x)?;
            let h = lin2.forward(&lin1.forward(&n)?.silu()?)?;
            x = (x + h)?;
        }
        self.out_proj.forward(&x)
    }
}

// ============================ blocks ============================

struct DoubleAttn {
    to_q: QLinear,
    to_k: QLinear,
    to_v: QLinear,
    to_out: QLinear,
    add_q: QLinear,
    add_k: QLinear,
    add_v: QLinear,
    to_add_out: QLinear,
    norm_q: RmsNorm,
    norm_k: RmsNorm,
    norm_added_q: RmsNorm,
    norm_added_k: RmsNorm,
    heads: usize,
    head_dim: usize,
}

impl DoubleAttn {
    fn load(vb: VarBuilder, cfg: &ChromaTransformerConfig, gs: usize) -> Result<Self> {
        let inner = cfg.inner_dim();
        let hd = cfg.attention_head_dim;
        // Every Chroma DiT projection is biased; `linear_detect_gs` packed-detects the MLX `.scales`
        // sibling (packed tier) or falls back to the dense biased load (dense tier), threading the
        // `config.json` group size. `to_out` nests at `to_out.0` — pass the full base so the
        // `.scales`/`.biases`/`.bias` siblings survive the key nesting (never `.pp("0")` past scales).
        let lin = |name: &str| QLinear::linear_detect_gs(inner, inner, &vb, name, true, gs);
        Ok(Self {
            to_q: lin("to_q")?,
            to_k: lin("to_k")?,
            to_v: lin("to_v")?,
            to_out: lin("to_out.0")?,
            add_q: lin("add_q_proj")?,
            add_k: lin("add_k_proj")?,
            add_v: lin("add_v_proj")?,
            to_add_out: lin("to_add_out")?,
            norm_q: rms_norm(hd, QK_RMS_EPS, vb.pp("norm_q"))?,
            norm_k: rms_norm(hd, QK_RMS_EPS, vb.pp("norm_k"))?,
            norm_added_q: rms_norm(hd, QK_RMS_EPS, vb.pp("norm_added_q"))?,
            norm_added_k: rms_norm(hd, QK_RMS_EPS, vb.pp("norm_added_k"))?,
            heads: cfg.num_attention_heads,
            head_dim: hd,
        })
    }

    /// Joint attention over `cat([text, image])`. Returns `(image_attn [B,Si,inner], text_attn
    /// [B,St,inner])`.
    fn forward(
        &self,
        hidden: &Tensor,
        encoder: &Tensor,
        rope: &RopeTable,
    ) -> Result<(Tensor, Tensor)> {
        let (h, hd) = (self.heads, self.head_dim);
        let q = to_heads(&self.to_q.forward(hidden)?, h, hd, Some(&self.norm_q))?;
        let k = to_heads(&self.to_k.forward(hidden)?, h, hd, Some(&self.norm_k))?;
        let v = to_heads(&self.to_v.forward(hidden)?, h, hd, None)?;
        let eq = to_heads(
            &self.add_q.forward(encoder)?,
            h,
            hd,
            Some(&self.norm_added_q),
        )?;
        let ek = to_heads(
            &self.add_k.forward(encoder)?,
            h,
            hd,
            Some(&self.norm_added_k),
        )?;
        let ev = to_heads(&self.add_v.forward(encoder)?, h, hd, None)?;
        // Concatenate [text, image] along the sequence axis (matches the RoPE id order).
        let q = rope.apply(&Tensor::cat(&[&eq, &q], 2)?)?;
        let k = rope.apply(&Tensor::cat(&[&ek, &k], 2)?)?;
        let v = Tensor::cat(&[&ev, &v], 2)?;
        let out = attention(&q, &k, &v, hd)?; // [B, St+Si, inner]
        let st = encoder.dim(1)?;
        let txt = out.narrow(1, 0, st)?;
        let img = out.narrow(1, st, hidden.dim(1)?)?;
        Ok((
            self.to_out.forward(&img.contiguous()?)?,
            self.to_add_out.forward(&txt.contiguous()?)?,
        ))
    }
}

struct DoubleBlock {
    attn: DoubleAttn,
    ff: FeedForward,
    ff_context: FeedForward,
}

impl DoubleBlock {
    fn load(vb: VarBuilder, cfg: &ChromaTransformerConfig, gs: usize) -> Result<Self> {
        let inner = cfg.inner_dim();
        Ok(Self {
            attn: DoubleAttn::load(vb.pp("attn"), cfg, gs)?,
            ff: FeedForward::new(inner, vb.pp("ff"), gs)?,
            ff_context: FeedForward::new(inner, vb.pp("ff_context"), gs)?,
        })
    }

    /// `temb` is the 12-row modulation slice `[B,12,inner]` (`[:6]` image, `[6:]` text). Each stream's
    /// rows are `(shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)`. Returns `(encoder,
    /// hidden)`.
    fn forward(
        &self,
        hidden: &Tensor,
        encoder: &Tensor,
        temb: &Tensor,
        rope: &RopeTable,
    ) -> Result<(Tensor, Tensor)> {
        let norm_hidden = modulate(&layer_norm(hidden)?, &row(temb, 1)?, &row(temb, 0)?)?;
        let norm_encoder = modulate(&layer_norm(encoder)?, &row(temb, 7)?, &row(temb, 6)?)?;

        let (attn_img, attn_txt) = self.attn.forward(&norm_hidden, &norm_encoder, rope)?;

        // image stream.
        let hidden = gated(hidden, &row(temb, 2)?, &attn_img)?;
        let nh = modulate(&layer_norm(&hidden)?, &row(temb, 4)?, &row(temb, 3)?)?;
        let hidden = gated(&hidden, &row(temb, 5)?, &self.ff.forward(&nh)?)?;

        // text stream.
        let encoder = gated(encoder, &row(temb, 8)?, &attn_txt)?;
        let ne = modulate(&layer_norm(&encoder)?, &row(temb, 10)?, &row(temb, 9)?)?;
        let encoder = gated(&encoder, &row(temb, 11)?, &self.ff_context.forward(&ne)?)?;

        Ok((encoder, hidden))
    }
}

struct SingleAttn {
    to_q: QLinear,
    to_k: QLinear,
    to_v: QLinear,
    norm_q: RmsNorm,
    norm_k: RmsNorm,
    heads: usize,
    head_dim: usize,
}

impl SingleAttn {
    fn load(vb: VarBuilder, cfg: &ChromaTransformerConfig, gs: usize) -> Result<Self> {
        let inner = cfg.inner_dim();
        let hd = cfg.attention_head_dim;
        let lin = |name: &str| QLinear::linear_detect_gs(inner, inner, &vb, name, true, gs);
        Ok(Self {
            to_q: lin("to_q")?,
            to_k: lin("to_k")?,
            to_v: lin("to_v")?,
            norm_q: rms_norm(hd, QK_RMS_EPS, vb.pp("norm_q"))?,
            norm_k: rms_norm(hd, QK_RMS_EPS, vb.pp("norm_k"))?,
            heads: cfg.num_attention_heads,
            head_dim: hd,
        })
    }

    fn forward(&self, x: &Tensor, rope: &RopeTable) -> Result<Tensor> {
        let (h, hd) = (self.heads, self.head_dim);
        let q = rope.apply(&to_heads(
            &self.to_q.forward(x)?,
            h,
            hd,
            Some(&self.norm_q),
        )?)?;
        let k = rope.apply(&to_heads(
            &self.to_k.forward(x)?,
            h,
            hd,
            Some(&self.norm_k),
        )?)?;
        let v = to_heads(&self.to_v.forward(x)?, h, hd, None)?;
        attention(&q, &k, &v, hd)
    }
}

struct SingleBlock {
    attn: SingleAttn,
    proj_mlp: QLinear,
    proj_out: QLinear,
}

impl SingleBlock {
    fn load(vb: VarBuilder, cfg: &ChromaTransformerConfig, gs: usize) -> Result<Self> {
        let inner = cfg.inner_dim();
        let mlp_hidden = 4 * inner;
        Ok(Self {
            attn: SingleAttn::load(vb.pp("attn"), cfg, gs)?,
            proj_mlp: QLinear::linear_detect_gs(inner, mlp_hidden, &vb, "proj_mlp", true, gs)?,
            proj_out: QLinear::linear_detect_gs(
                inner + mlp_hidden,
                inner,
                &vb,
                "proj_out",
                true,
                gs,
            )?,
        })
    }

    /// `temb` is the 3-row modulation slice `[B,3,inner]` (shift, scale, gate). `hidden` is the joint
    /// `[text|image]` stream.
    fn forward(&self, hidden: &Tensor, temb: &Tensor, rope: &RopeTable) -> Result<Tensor> {
        let norm_hidden = modulate(&layer_norm(hidden)?, &row(temb, 1)?, &row(temb, 0)?)?;
        let mlp = self.proj_mlp.forward(&norm_hidden)?.gelu()?;
        let attn = self.attn.forward(&norm_hidden, rope)?;
        let proj = self
            .proj_out
            .forward(&Tensor::cat(&[&attn, &mlp], D::Minus1)?)?;
        gated(hidden, &row(temb, 2)?, &proj)
    }
}

// ============================ the transformer ============================

pub struct ChromaTransformer {
    cfg: ChromaTransformerConfig,
    x_embedder: QLinear,
    context_embedder: QLinear,
    time_text_embed: TimestepTextProj,
    approximator: Approximator,
    double_blocks: Vec<DoubleBlock>,
    single_blocks: Vec<SingleBlock>,
    proj_out: QLinear,
}

impl ChromaTransformer {
    /// Build from a diffusers `transformer/` VarBuilder (f32) at an explicit MLX `group_size` read from
    /// the packed `transformer/config.json`'s `quantization.group_size` (sc-9409; a **dense** snapshot
    /// passes the default 64). The block counts come from the config (the VarBuilder errors loudly on a
    /// key mismatch, so a wrong checkpoint fails at load).
    ///
    /// Every packed projection is loaded via [`crate::quant::QLinear::linear_detect_gs`], which
    /// packed-detects the `.scales` sibling (packed tier) or falls back to the dense biased load
    /// unchanged (dense tier — and the still-dense `x_embedder`/`context_embedder`/`proj_out` +
    /// Approximator inside a packed tier). `x_embedder`/`context_embedder`/`proj_out` are dense in every
    /// shipped tier but route through the same detect loader for a single uniform seam.
    pub fn new_gs(cfg: ChromaTransformerConfig, vb: VarBuilder, gs: usize) -> Result<Self> {
        let inner = cfg.inner_dim();
        let device = vb.device().clone();
        let double_blocks = (0..cfg.num_layers)
            .map(|i| DoubleBlock::load(vb.pp("transformer_blocks").pp(i), &cfg, gs))
            .collect::<Result<Vec<_>>>()?;
        let single_blocks = (0..cfg.num_single_layers)
            .map(|i| SingleBlock::load(vb.pp("single_transformer_blocks").pp(i), &cfg, gs))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            x_embedder: QLinear::linear_detect_gs(
                cfg.in_channels,
                inner,
                &vb,
                "x_embedder",
                true,
                gs,
            )?,
            context_embedder: QLinear::linear_detect_gs(
                cfg.joint_attention_dim,
                inner,
                &vb,
                "context_embedder",
                true,
                gs,
            )?,
            time_text_embed: TimestepTextProj::new(&cfg, &device)?,
            approximator: Approximator::load(vb.clone(), &cfg, gs)?,
            double_blocks,
            single_blocks,
            proj_out: QLinear::linear_detect_gs(inner, cfg.in_channels, &vb, "proj_out", true, gs)?,
            cfg,
        })
    }

    /// `pooled_temb [B, mod_index_len, inner]` for a **raw** (unscaled) timestep `[B]`. Depends only
    /// on the timestep, so the denoise loop computes it once per step and shares it across both CFG
    /// branches.
    pub fn pooled_temb(&self, timestep: &Tensor) -> Result<Tensor> {
        let scaled = (timestep.to_dtype(DType::F32)? * 1000.0)?;
        self.approximator
            .forward(&self.time_text_embed.forward(&scaled)?)
    }

    /// Run the MMDiT given the pre-built step-invariant tensors. `hidden [B, Si, in_channels]` packed
    /// image latent tokens, `encoder [B, St, joint_attention_dim]` T5 embeddings, `pooled` the
    /// Approximator modulation table, `rope` the table over `cat(txt_ids, img_ids)`. Returns the
    /// predicted velocity `[B, Si, in_channels]`.
    pub fn forward_prepared(
        &self,
        hidden: &Tensor,
        encoder: &Tensor,
        pooled: &Tensor,
        rope: &RopeTable,
    ) -> Result<Tensor> {
        let mut hidden = self.x_embedder.forward(&hidden.to_dtype(DType::F32)?)?;
        let mut encoder = self
            .context_embedder
            .forward(&encoder.to_dtype(DType::F32)?)?;

        let st = encoder.dim(1)?;
        let n_single = self.cfg.num_single_layers;
        let img_offset = 3 * n_single;
        let txt_offset = img_offset + 6 * self.cfg.num_layers;

        for (i, block) in self.double_blocks.iter().enumerate() {
            let img = rows(pooled, img_offset + 6 * i, 6)?;
            let txt = rows(pooled, txt_offset + 6 * i, 6)?;
            let temb = Tensor::cat(&[&img, &txt], 1)?; // [B,12,inner]
            let (e, h) = block.forward(&hidden, &encoder, &temb, rope)?;
            encoder = e;
            hidden = h;
        }

        let mut joint = Tensor::cat(&[&encoder, &hidden], 1)?; // [B, St+Si, inner]
        for (i, block) in self.single_blocks.iter().enumerate() {
            let temb = rows(pooled, 3 * i, 3)?;
            joint = block.forward(&joint, &temb, rope)?;
        }

        // Drop the text tokens; pruned `norm_out` (shift, scale = pooled[-2:]); proj_out.
        let si = joint.dim(1)? - st;
        let hidden = joint.narrow(1, st, si)?;
        let n = self.cfg.mod_index_len();
        let no = rows(pooled, n - 2, 2)?;
        let hidden = modulate(&layer_norm(&hidden)?, &row(&no, 1)?, &row(&no, 0)?)?;
        self.proj_out.forward(&hidden)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestep_embedding_is_cos_then_sin() {
        // t=0 → all args 0 → cos 1 (first half), sin 0 (second half).
        let t = Tensor::zeros(1, DType::F32, &Device::Cpu).unwrap();
        let emb = timestep_embedding(&t, 32, 0.0, &Device::Cpu).unwrap();
        assert_eq!(emb.dims(), &[1, 32]);
        let v = emb.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for c in &v[..16] {
            assert!((c - 1.0).abs() < 1e-6);
        }
        for s in &v[16..] {
            assert!(s.abs() < 1e-6);
        }
    }

    #[test]
    fn chunked_attention_matches_single_pass() {
        // Per-query-row softmax is independent, so chunking over query rows (forced via a tiny budget)
        // must match the single pass bit-for-bit — the guard for the i32-overflow fix (sc-8983,
        // ported from FLUX.2's sc-5487). Retargeted onto the shared `candle_gen::sdpa_budgeted_bhsd`
        // (sc-9570) with this crate's exact `softmax_last_dim` closure and no mask.
        let dev = Device::Cpu;
        let (b, h, s, d) = (1usize, 2usize, 7usize, 4usize);
        let q = Tensor::randn(0f32, 1f32, (b, h, s, d), &dev).unwrap();
        let k = Tensor::randn(0f32, 1f32, (b, h, s, d), &dev).unwrap();
        let v = Tensor::randn(0f32, 1f32, (b, h, s, d), &dev).unwrap();
        let scale = (d as f64).powf(-0.5);
        let sm = |x: &Tensor| softmax_last_dim(x);
        // Huge budget → single pass; tiny budget (1) → single-row chunks; a MID-SIZE budget forces
        // multi-row chunks + a remainder (block=3 over s=7 → 3,3,1) — the sc-9116 test-hardening ask.
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
        for x in out.flatten_all().unwrap().to_vec1::<f32>().unwrap() {
            assert!((x - 2.0).abs() < 1e-6);
        }
    }

    #[test]
    fn row_and_rows_slice_the_modulation_table() {
        // [B=1, K=4, inner=2]
        let t = Tensor::arange(0f32, 8f32, &Device::Cpu)
            .unwrap()
            .reshape((1, 4, 2))
            .unwrap();
        assert_eq!(row(&t, 2).unwrap().dims(), &[1, 1, 2]);
        assert_eq!(rows(&t, 1, 3).unwrap().dims(), &[1, 3, 2]);
        let r2 = row(&t, 2)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(r2, vec![4.0, 5.0]);
    }
}
