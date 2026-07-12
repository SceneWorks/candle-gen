//! Krea 2 **pose-ControlNet control branch** — the sc-8460 spike (epic 8459).
//!
//! A ControlNet-style trainable side branch over the Krea 2 single-stream DiT: the first `N`
//! (default 7 of 28) single-stream blocks are **copied** into a fully trainable branch
//! ([`ControlBranch`]), driven by the same frozen pre-main front-end the base uses. The branch input
//! is the base's joint `[ctx; img]` sequence with the **control latent embedding** (the VAE-encoded
//! pose skeleton, patch-embedded through the *frozen* base `img_in` —
//! [`KreaTrainDit::embed_latent`]) added onto the image-token slice. After each branch block `i` a
//! **zero-initialized** output projection produces a per-block residual that is added (scaled by
//! `control_scale`) onto the frozen main stream's **image tokens** entering main block `i` — so at
//! step 0 the branch is an exact identity over the base model (the ControlNet zero-init property).
//!
//! Everything in the branch trains (attention, gate, SwiGLU, norms, modulation table, and the output
//! projections); the entire base (DiT + text encoder + VAE) stays frozen. The branch reuses the
//! composable (differentiable) ops of [`crate::train_dit`] — the fused `softmax_last_dim` /
//! `ops::rms_norm` kernels have no backward.
//!
//! **Dtype.** Branch matmul weights live in a configurable dtype (bf16 default — full f32 master
//! weights for a ~3B-param branch triple the optimizer footprint; see the trainer CLI's
//! `--branch-dtype`); the `+1`-folded RMSNorm scales stay f32 (tiny, and [`rms_scale_diff`] reduces
//! in f32 anyway). Weights are cast to the activation dtype inside the forward when they differ —
//! the cast is differentiable, so f32 master-weight training works through the same code path.
//!
//! **Checkpoint format (spike).** A flat `.safetensors` of the branch tensors keyed
//! `blocks.{i}.<leaf>` (norm scales stored **pre-folded** `scale + 1` under `*.weight_p1` — this
//! checkpoint is only read back by [`ControlBranch::from_checkpoint`], never by a diffusers loader),
//! plus `blocks.{i}.proj_out.weight` for the zero-init projections. `N` is inferred from the keys.

use std::collections::HashMap;
use std::path::Path;

use candle_gen::candle_core::backprop::GradStore;
use candle_gen::candle_core::{DType, Device, Tensor, Var};
use candle_gen::candle_nn::ops::sigmoid;
use candle_gen::train::flow_match::{self, velocity_loss};
use candle_gen::train::gradient_checkpoint::{checkpointed_backward, Segment};
use candle_gen::{CandleError, Result};

use crate::config::Krea2Config;
use crate::loader::{rms_scale_weight, Weights};
use crate::train_dit::{repeat_kv, rms_scale_diff, sdpa_diff, KreaTrainDit, MainCtx};
use crate::transformer::rope::apply_interleaved_rope;

/// Default residual RMS clamp τ. The run-3 probe showed `τ = 1.0` is destruction-permissive: the
/// block-0 projection regrew to the ceiling even at lr 2e-5 (AdamW's normalized steps reach unit
/// gain on a zero-init 6144² matrix in ~`0.0128/lr` steps under any persistent gradient), and a
/// full-stream-RMS injection at the earliest blocks overwrites rather than steers. At `0.15` even a
/// ceiling-riding block remains a perturbation.
pub const DEFAULT_RESIDUAL_CLAMP: f64 = 0.15;

/// Default injection offset: the residual from branch block `i` is added to the main stream
/// entering block `i + offset`. `1` skips main block 0 — the run-3 degeneracy's preferred site
/// (it precedes ALL base computation, so overwriting there is the cheapest way to satisfy the
/// training loss; every probe showed block 0 pinned at the clamp ceiling while blocks ≥ 2 stayed
/// healthy). Tradeoff: the control signal first lands after one base block of computation — the
/// standard "branch output feeds the NEXT block" DiT-ControlNet layout, at the cost of no direct
/// control over block 0's features.
pub const DEFAULT_INJECT_OFFSET: usize = 1;

/// Checkpoint key persisting [`DEFAULT_INJECT_OFFSET`]'s actual value (an f32 `[1]` tensor — the
/// spike checkpoint format has no metadata header). Absent in pre-offset checkpoints ⇒ `0`.
const META_INJECT_OFFSET: &str = "meta.inject_offset";

/// A trainable no-bias linear over a [`Var`] weight `[out, in]`; the weight is cast to the activation
/// dtype in the forward (differentiable — f32 master weights train through it).
struct VLin {
    w: Var,
}

impl VLin {
    fn forward(&self, x: &Tensor, frozen: bool) -> Result<Tensor> {
        let w = rd(&self.w, frozen).to_dtype(x.dtype())?;
        Ok(x.broadcast_matmul(&w.t()?)?)
    }
}

/// Read a `Var`'s current value: the live (graph-tracked) variable while training, or a **detached**
/// storage-sharing view when the branch is [frozen](ControlBranch::freeze) for inference. Without the
/// detach, every sampler step would build (and, chained through the Euler update, RETAIN across all
/// steps) a full autograd graph over the branch + the 28 main blocks — an OOM at render resolution.
fn rd(v: &Var, frozen: bool) -> Tensor {
    if frozen {
        v.as_tensor().detach()
    } else {
        v.as_tensor().clone()
    }
}

/// Registers every created [`Var`] under its checkpoint key, in a deterministic order.
#[derive(Default)]
struct VarReg {
    named: Vec<(String, Var)>,
}

impl VarReg {
    fn var(&mut self, name: String, t: &Tensor) -> Result<Var> {
        let v = Var::from_tensor(t)?;
        self.named.push((name, v.clone()));
        Ok(v)
    }

    fn lin(&mut self, name: String, t: &Tensor) -> Result<VLin> {
        Ok(VLin {
            w: self.var(name, t)?,
        })
    }
}

/// The tensor source a branch is built from: base-block weights (fresh training) or a saved
/// checkpoint (resume / inference). Returns tensors already in their final storage dtype.
type Getter<'a> = dyn Fn(&str) -> Result<Tensor> + 'a;

/// Trainable twin of the base block's sigmoid-gated GQA attention — every projection is a [`Var`].
struct ControlAttention {
    q: VLin,
    k: VLin,
    v: VLin,
    gate: VLin,
    o: VLin,
    norm_q: Var, // f32, scale + 1
    norm_k: Var, // f32, scale + 1
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    eps: f64,
    scale: f64,
}

impl ControlAttention {
    fn build(reg: &mut VarReg, get: &Getter, prefix: &str, cfg: &Krea2Config) -> Result<Self> {
        let key = |leaf: &str| format!("{prefix}.{leaf}");
        Ok(Self {
            q: reg.lin(key("to_q.weight"), &get(&key("to_q.weight"))?)?,
            k: reg.lin(key("to_k.weight"), &get(&key("to_k.weight"))?)?,
            v: reg.lin(key("to_v.weight"), &get(&key("to_v.weight"))?)?,
            gate: reg.lin(key("to_gate.weight"), &get(&key("to_gate.weight"))?)?,
            o: reg.lin(key("to_out.0.weight"), &get(&key("to_out.0.weight"))?)?,
            norm_q: reg.var(key("norm_q.weight_p1"), &get(&key("norm_q.weight_p1"))?)?,
            norm_k: reg.var(key("norm_k.weight_p1"), &get(&key("norm_k.weight_p1"))?)?,
            heads: cfg.num_attention_heads,
            kv_heads: cfg.num_kv_heads,
            head_dim: cfg.attention_head_dim,
            eps: cfg.norm_eps,
            scale: (cfg.attention_head_dim as f64).powf(-0.5),
        })
    }

    fn forward(&self, x: &Tensor, rope: (&Tensor, &Tensor), frozen: bool) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let (nh, nkv, hd) = (self.heads, self.kv_heads, self.head_dim);

        let q = self.q.forward(x, frozen)?.reshape((b, s, nh, hd))?;
        let k = self.k.forward(x, frozen)?.reshape((b, s, nkv, hd))?;
        let v = self.v.forward(x, frozen)?.reshape((b, s, nkv, hd))?;
        let gate = self.gate.forward(x, frozen)?;

        let q = rms_scale_diff(&q, &rd(&self.norm_q, frozen), self.eps)?;
        let k = rms_scale_diff(&k, &rd(&self.norm_k, frozen), self.eps)?;
        let (cos, sin) = rope;
        let q = apply_interleaved_rope(&q, cos, sin)?;
        let k = apply_interleaved_rope(&k, cos, sin)?;

        let groups = nh / nkv;
        let k = repeat_kv(&k, groups)?;
        let v = repeat_kv(&v, groups)?;

        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;
        let o = sdpa_diff(&q, &k, &v, self.scale)?;
        let o = o.transpose(1, 2)?.contiguous()?.reshape((b, s, nh * hd))?;

        let gated = (o * sigmoid(&gate)?)?;
        self.o.forward(&gated, frozen)
    }
}

/// One fully trainable branch block (a `Var`-backed copy of a base single-stream block) plus its
/// zero-initialized per-block output projection.
struct ControlBlock {
    sst: Var,      // [1, 1, 6·hidden], branch dtype
    prenorm: Var,  // f32, scale + 1
    postnorm: Var, // f32, scale + 1
    attn: ControlAttention,
    ff_gate: VLin,
    ff_up: VLin,
    ff_down: VLin,
    proj_out: VLin, // zero-init [hidden, hidden] — the ControlNet identity seam
    eps: f64,
}

impl ControlBlock {
    fn build(reg: &mut VarReg, get: &Getter, i: usize, cfg: &Krea2Config) -> Result<Self> {
        let key = |leaf: &str| format!("blocks.{i}.{leaf}");
        Ok(Self {
            sst: reg.var(key("scale_shift_table"), &get(&key("scale_shift_table"))?)?,
            prenorm: reg.var(key("norm1.weight_p1"), &get(&key("norm1.weight_p1"))?)?,
            postnorm: reg.var(key("norm2.weight_p1"), &get(&key("norm2.weight_p1"))?)?,
            attn: ControlAttention::build(reg, get, &key("attn"), cfg)?,
            ff_gate: reg.lin(key("ff.gate.weight"), &get(&key("ff.gate.weight"))?)?,
            ff_up: reg.lin(key("ff.up.weight"), &get(&key("ff.up.weight"))?)?,
            ff_down: reg.lin(key("ff.down.weight"), &get(&key("ff.down.weight"))?)?,
            proj_out: reg.lin(key("proj_out.weight"), &get(&key("proj_out.weight"))?)?,
            eps: cfg.norm_eps,
        })
    }

    /// Mirror of [`crate::train_dit`]'s `TrainBlock::forward` over the `Var`-backed weights.
    fn forward(
        &self,
        x: &Tensor,
        tvec: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        frozen: bool,
    ) -> Result<Tensor> {
        use candle_gen::candle_core::D;
        let sst = rd(&self.sst, frozen).to_dtype(tvec.dtype())?;
        let m = tvec.broadcast_add(&sst)?; // [b, 1, 6·hidden]
        let chunks = m.chunk(6, D::Minus1)?;
        let (prescale, preshift, pregate) = (&chunks[0], &chunks[1], &chunks[2]);
        let (postscale, postshift, postgate) = (&chunks[3], &chunks[4], &chunks[5]);

        let pre = rms_scale_diff(x, &rd(&self.prenorm, frozen), self.eps)?
            .broadcast_mul(&(prescale + 1.0)?)?
            .broadcast_add(preshift)?;
        let attn = self.attn.forward(&pre, (cos, sin), frozen)?;
        let x = (x + attn.broadcast_mul(pregate)?)?;

        let post = rms_scale_diff(&x, &rd(&self.postnorm, frozen), self.eps)?
            .broadcast_mul(&(postscale + 1.0)?)?
            .broadcast_add(postshift)?;
        let gated =
            (self.ff_gate.forward(&post, frozen)?.silu()? * self.ff_up.forward(&post, frozen)?)?;
        let mlp = self.ff_down.forward(&gated, frozen)?;
        Ok((&x + mlp.broadcast_mul(postgate)?)?)
    }
}

/// The trainable Krea 2 control branch: `N` `Var`-backed copies of the first `N` base single-stream
/// blocks + one zero-init output projection each. Built either fresh from the base weights
/// ([`from_base`](Self::from_base)) or from a saved spike checkpoint
/// ([`from_checkpoint`](Self::from_checkpoint)).
pub struct ControlBranch {
    blocks: Vec<ControlBlock>,
    named: Vec<(String, Var)>,
    /// Inference mode: weight reads are detached from the autograd graph (see [`rd`]). Off (training)
    /// by default; flip with [`freeze`](Self::freeze) before using the branch in a sampler loop.
    frozen: bool,
    /// Residual RMS clamp τ (sc-8460 step-500 probe): each injected residual's RMS is capped at
    /// `τ ×` the RMS of the main image-token slice it lands on. `None` = unclamped. Without it the
    /// zero-init block-0 projection is free to grow until it **overwrites** the main stream (the
    /// step-500 run reached ‖res₀‖ ≈ 25×‖main₀‖ while the single-forward training loss stayed
    /// normal — downstream RMSNorms absorb the scale — yet the 8-step sampler compounds the
    /// hypersensitive velocity into full-frame noise). The clamp factor is treated as a constant
    /// (stop-grad, like adaptive grad clipping), so training simply optimizes within the budget.
    /// Applied identically in the training, checkpointed, inference, and probe paths.
    clamp_tau: Option<f64>,
    /// Residual from branch block `i` injects into main block `i + inject_offset` (see
    /// [`DEFAULT_INJECT_OFFSET`]). Persisted in the checkpoint; must match between train and infer.
    inject_offset: usize,
}

impl ControlBranch {
    /// Copy the first `n` base single-stream blocks (from the same mmap'd `transformer/` [`Weights`]
    /// the frozen DiT loads) into a trainable branch at `dtype`, with zero-init output projections.
    /// Branch block `i`'s residual injects into main block `i + inject_offset`.
    pub fn from_base(
        w: &Weights,
        cfg: &Krea2Config,
        n: usize,
        dtype: DType,
        inject_offset: usize,
    ) -> Result<Self> {
        if n == 0 || inject_offset + n > cfg.num_layers {
            return Err(CandleError::Msg(format!(
                "control branch: need 1 <= n_blocks and inject_offset + n_blocks <= {} \
                 (got n_blocks {n}, inject_offset {inject_offset})",
                cfg.num_layers
            )));
        }
        let hidden = cfg.hidden_size;
        let device = w.device().clone();
        let get = move |key: &str| -> Result<Tensor> {
            // "blocks.{i}.<leaf>" -> the base "transformer_blocks.{i}.<leaf>" (or synthesized).
            let (i, leaf) = split_block_key(key)?;
            let base = format!("transformer_blocks.{i}");
            match leaf {
                "scale_shift_table" => Ok(w
                    .get(&format!("{base}.scale_shift_table"))?
                    .reshape((1, 1, 6 * hidden))?
                    .to_dtype(dtype)?),
                "norm1.weight_p1" => Ok(rms_scale_weight(w, &format!("{base}.norm1.weight"))?),
                "norm2.weight_p1" => Ok(rms_scale_weight(w, &format!("{base}.norm2.weight"))?),
                "attn.norm_q.weight_p1" => {
                    Ok(rms_scale_weight(w, &format!("{base}.attn.norm_q.weight"))?)
                }
                "attn.norm_k.weight_p1" => {
                    Ok(rms_scale_weight(w, &format!("{base}.attn.norm_k.weight"))?)
                }
                "proj_out.weight" => Ok(Tensor::zeros((hidden, hidden), dtype, &device)?),
                other => Ok(w.get(&format!("{base}.{other}"))?.to_dtype(dtype)?),
            }
        };
        Self::build(&get, cfg, n, inject_offset)
    }

    /// Load a branch back from a spike checkpoint (`.safetensors` written by [`save`](Self::save));
    /// `N`, dtypes, and the injection offset are read from the file (a pre-offset checkpoint has no
    /// meta key ⇒ offset 0, its training-time layout).
    pub fn from_checkpoint(path: &Path, cfg: &Krea2Config, device: &Device) -> Result<Self> {
        let tensors = candle_gen::candle_core::safetensors::load(path, device)?;
        let n = 1 + tensors
            .keys()
            .filter_map(|k| split_block_key(k).ok().map(|(i, _)| i))
            .max()
            .ok_or_else(|| {
                CandleError::Msg(format!(
                    "control branch: no blocks.{{i}}.* keys in {}",
                    path.display()
                ))
            })?;
        let inject_offset = match tensors.get(META_INJECT_OFFSET) {
            // A malformed/truncated overlay can ship a size-0 `meta.inject_offset`; route the read
            // through the hardened scalar reader (F-119 / sc-11208, F-009 class) so a corrupt
            // studio-trained checkpoint is a typed error at `Krea2Control::load`, not a worker panic.
            Some(t) => {
                candle_gen::train::merge::read_scalar(META_INJECT_OFFSET, "inject_offset", t)?
                    as usize
            }
            None => 0,
        };
        let get = |key: &str| -> Result<Tensor> {
            tensors.get(key).cloned().ok_or_else(|| {
                CandleError::Msg(format!(
                    "control branch: missing key {key} in {}",
                    path.display()
                ))
            })
        };
        Self::build(&get, cfg, n, inject_offset)
    }

    fn build(get: &Getter, cfg: &Krea2Config, n: usize, inject_offset: usize) -> Result<Self> {
        let mut reg = VarReg::default();
        let blocks = (0..n)
            .map(|i| ControlBlock::build(&mut reg, get, i, cfg))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            blocks,
            named: reg.named,
            frozen: false,
            clamp_tau: Some(DEFAULT_RESIDUAL_CLAMP),
            inject_offset,
        })
    }

    /// The main-block index branch block `i`'s residual injects into is `i + inject_offset()`.
    pub fn inject_offset(&self) -> usize {
        self.inject_offset
    }

    /// Switch the branch to inference mode: every weight read is detached, so a sampler loop builds
    /// **no** autograd graph (which the Euler update would otherwise chain and retain across all
    /// steps — an OOM at render resolution). Values are unchanged; there is no unfreeze (reload for
    /// training).
    pub fn freeze(&mut self) {
        self.frozen = true;
    }

    /// Override the residual RMS clamp τ (`None` = unclamped — the pre-fix behavior, kept for
    /// A/B-ing checkpoints trained without the clamp). See the field docs for why the default is on.
    pub fn set_residual_clamp(&mut self, tau: Option<f64>) {
        self.clamp_tau = tau;
    }

    /// Clamp an (already `control_scale`-scaled) residual so its RMS is at most
    /// `τ × RMS(main image tokens)`: `res · min(1, τ·rms(main)/rms(res))`. The factor is a detached
    /// scalar (stop-grad, like adaptive gradient clipping), so gradients flow through the scaled
    /// residual only. No-op when unclamped, when the residual is already within budget, or when it
    /// is exactly zero (the step-0 identity is untouched).
    fn apply_clamp(&self, res: &Tensor, main_img: &Tensor) -> Result<Tensor> {
        let Some(tau) = self.clamp_tau else {
            return Ok(res.clone());
        };
        let rms = |t: &Tensor| -> Result<f64> {
            Ok((t
                .detach()
                .to_dtype(DType::F32)?
                .sqr()?
                .mean_all()?
                .to_scalar::<f32>()? as f64)
                .sqrt())
        };
        let rn = rms(res)?;
        let cap = tau * rms(main_img)?;
        if rn <= cap || rn == 0.0 {
            Ok(res.clone())
        } else {
            Ok((res * (cap / rn))?)
        }
    }

    /// Number of branch blocks (`N`).
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Every trainable `Var` (blocks + projections), in deterministic construction order — the
    /// optimizer parameter list.
    pub fn vars(&self) -> Vec<Var> {
        self.named.iter().map(|(_, v)| v.clone()).collect()
    }

    /// The zero-init output-projection `Var`s only — the optimizer group that gets weight decay +
    /// a reduced lr (magnitude control on the injection gain must be structural: AdamW's normalized
    /// steps regrow a zero-init 6144² projection to unit gain in ~`0.0128/lr` steps otherwise).
    pub fn proj_vars(&self) -> Vec<Var> {
        self.named
            .iter()
            .filter(|(n, _)| n.contains("proj_out"))
            .map(|(_, v)| v.clone())
            .collect()
    }

    /// Every non-projection `Var` (the branch block bodies) — the plain optimizer group.
    pub fn body_vars(&self) -> Vec<Var> {
        self.named
            .iter()
            .filter(|(n, _)| !n.contains("proj_out"))
            .map(|(_, v)| v.clone())
            .collect()
    }

    /// Trainable parameter count.
    pub fn num_params(&self) -> usize {
        self.named
            .iter()
            .map(|(_, v)| v.as_tensor().elem_count())
            .sum()
    }

    /// Write the branch to a flat `.safetensors` (the spike checkpoint [`from_checkpoint`] reads),
    /// including the injection-offset meta tensor.
    pub fn save(&self, path: &Path) -> Result<()> {
        let mut map: HashMap<String, Tensor> = self
            .named
            .iter()
            .map(|(k, v)| (k.clone(), v.as_tensor().clone()))
            .collect();
        let dev = candle_gen::candle_core::Device::Cpu;
        map.insert(
            META_INJECT_OFFSET.to_string(),
            Tensor::from_vec(vec![self.inject_offset as f32], (1,), &dev)?,
        );
        Ok(candle_gen::candle_core::safetensors::save(&map, path)?)
    }

    /// Run the branch over the base's joint sequence and return the per-block residuals
    /// `[b, img_len, hidden]` (one per branch block, for main blocks `0..N`). The control tokens are
    /// added onto the image-token slice of the branch **input**; each residual is the zero-init
    /// projection of the branch hidden state's image tokens after that branch block.
    fn residuals(
        &self,
        combined: &Tensor,
        ctrl_tokens: &Tensor,
        ctx: &MainCtx,
    ) -> Result<Vec<Tensor>> {
        self.residuals_mode(combined, ctrl_tokens, ctx, self.frozen)
    }

    /// [`residuals`](Self::residuals) with an explicit graph mode — `frozen = true` forces detached
    /// weight reads regardless of the branch state (the always-graph-free probe path).
    fn residuals_mode(
        &self,
        combined: &Tensor,
        ctrl_tokens: &Tensor,
        ctx: &MainCtx,
        frozen: bool,
    ) -> Result<Vec<Tensor>> {
        let txt = combined.narrow(1, 0, ctx.cap_len)?;
        let img = (combined.narrow(1, ctx.cap_len, ctx.img_len)? + ctrl_tokens)?;
        let mut h = Tensor::cat(&[&txt, &img], 1)?;
        let mut out = Vec::with_capacity(self.blocks.len());
        for blk in &self.blocks {
            h = blk.forward(&h, &ctx.tvec, &ctx.rcos, &ctx.rsin, frozen)?;
            let h_img = h.narrow(1, ctx.cap_len, ctx.img_len)?;
            out.push(blk.proj_out.forward(&h_img, frozen)?);
        }
        Ok(out)
    }

    /// The residual (by branch-block index) that injects into main block `j`, honoring
    /// [`inject_offset`](Self::inject_offset): `Some(j - offset)` when `offset <= j < offset + N`.
    fn residual_index_for_main_block(&self, j: usize) -> Option<usize> {
        j.checked_sub(self.inject_offset)
            .filter(|&i| i < self.blocks.len())
    }
}

/// `"blocks.{i}.<leaf>"` → `(i, leaf)`.
fn split_block_key(key: &str) -> Result<(usize, &str)> {
    let rest = key
        .strip_prefix("blocks.")
        .ok_or_else(|| CandleError::Msg(format!("control branch: unexpected key {key}")))?;
    let (idx, leaf) = rest
        .split_once('.')
        .ok_or_else(|| CandleError::Msg(format!("control branch: unexpected key {key}")))?;
    let i: usize = idx
        .parse()
        .map_err(|_| CandleError::Msg(format!("control branch: unexpected key {key}")))?;
    Ok((i, leaf))
}

/// Velocity prediction **with** the control branch: the frozen base forward with the branch's
/// per-block residuals (scaled by `control_scale`) added onto the main stream's image tokens
/// entering main blocks `0..N`.
///
/// `control_scale == 0.0` short-circuits to the plain base forward — **byte-identical** to an
/// un-branched generation (the branch is never run), which is the spike's identity contract.
///
/// `ctrl_latent` is the VAE-encoded pose skeleton `[b, 16, H/8, W/8]` (the same normalized latent
/// space as the noisy latent); it is patch-embedded through the frozen base `img_in`.
pub fn forward_with_control(
    dit: &KreaTrainDit,
    branch: &ControlBranch,
    latent: &Tensor,
    timestep: &Tensor,
    context: &Tensor,
    ctrl_latent: &Tensor,
    control_scale: f64,
) -> Result<Tensor> {
    if control_scale == 0.0 {
        return Ok(dit.forward(latent, timestep, context)?);
    }
    let (combined, ctx) = dit.forward_pre_main(latent, timestep, context)?;
    let ctrl_tokens = dit.embed_latent(ctrl_latent)?;
    let residuals = branch.residuals(&combined, &ctrl_tokens, &ctx)?;

    let mut x = combined;
    for (j, blk) in dit.blocks().iter().enumerate() {
        if let Some(r) = branch
            .residual_index_for_main_block(j)
            .and_then(|k| residuals.get(k))
        {
            let txt = x.narrow(1, 0, ctx.cap_len)?;
            let img = x.narrow(1, ctx.cap_len, ctx.img_len)?;
            let scaled = (r * control_scale)?.to_dtype(x.dtype())?;
            let scaled = branch.apply_clamp(&scaled, &img)?;
            let img = (img + scaled)?;
            x = Tensor::cat(&[&txt, &img], 1)?;
        }
        x = blk.forward(&x, &ctx.tvec, &ctx.rcos, &ctx.rsin)?;
    }
    Ok(dit.velocity_out(&x, &ctx)?)
}

/// DIAGNOSTIC + TELEMETRY (sc-8460): the branched forward with per-injection-point norms. For each
/// branch block `i` (injecting into main block `i + offset`) reports
/// `(‖res_i‖₂ pre-clamp, ‖res_i‖₂ post-clamp, ‖main_img‖₂)` — the residual the branch WANTED to
/// add, what was actually added, and the main-stream image-token slice it lands on — plus the
/// branched velocity and the un-branched base velocity for the same inputs. Always runs with
/// **detached** weight reads (graph-free) regardless of the branch mode, so the trainer can call it
/// mid-run for live ratio telemetry without retaining a training-sized graph.
#[allow(clippy::type_complexity)]
pub fn probe_forward(
    dit: &KreaTrainDit,
    branch: &ControlBranch,
    latent: &Tensor,
    timestep: &Tensor,
    context: &Tensor,
    ctrl_latent: &Tensor,
    control_scale: f64,
) -> Result<(Vec<(f64, f64, f64)>, Tensor, Tensor)> {
    let norm = |t: &Tensor| -> Result<f64> {
        let sq = t.detach().to_dtype(DType::F32)?.sqr()?.sum_all()?;
        Ok((sq.to_scalar::<f32>()? as f64).sqrt())
    };
    let (combined, ctx) = dit.forward_pre_main(latent, timestep, context)?;
    let ctrl_tokens = dit.embed_latent(ctrl_latent)?;
    let residuals = branch.residuals_mode(&combined, &ctrl_tokens, &ctx, true)?;

    let mut report = Vec::with_capacity(residuals.len());
    let mut x = combined.clone();
    for (j, blk) in dit.blocks().iter().enumerate() {
        if let Some(r) = branch
            .residual_index_for_main_block(j)
            .and_then(|k| residuals.get(k))
        {
            let txt = x.narrow(1, 0, ctx.cap_len)?;
            let img = x.narrow(1, ctx.cap_len, ctx.img_len)?;
            let scaled = (r * control_scale)?.to_dtype(x.dtype())?;
            let clamped = branch.apply_clamp(&scaled, &img)?;
            report.push((norm(&scaled)?, norm(&clamped)?, norm(&img)?));
            let img = (img + clamped)?;
            x = Tensor::cat(&[&txt, &img], 1)?;
        }
        x = blk.forward(&x, &ctx.tvec, &ctx.rcos, &ctx.rsin)?;
    }
    let v_branched = dit.velocity_out(&x, &ctx)?;

    // Un-branched base velocity over the SAME pre-main output.
    let mut xb = combined;
    for blk in dit.blocks() {
        xb = blk.forward(&xb, &ctx.tvec, &ctx.rcos, &ctx.rsin)?;
    }
    let v_base = dit.velocity_out(&xb, &ctx)?;
    Ok((report, v_branched, v_base))
}

/// One training micro-step's forward + backward at flow-match `sigma`: build the noised latent, run
/// the control-branched velocity prediction (training residual scale **1.0** — `control_scale` is an
/// inference knob), regress the raw velocity toward `noise − x0`, and return `(loss, grads)` keyed by
/// the branch `Var`s. The training twin of the Krea LoRA trainer's `compute_loss_grads`.
///
/// `use_checkpoint` selects the **gradient-checkpointed** backward (the sc-5165 / sc-7900 segmented
/// VJP the big-DiT LoRA trainers use, [`checkpointed_backward`]) over the dense monolithic
/// `loss.backward()`. The dense backward retains every activation of the 28 frozen main blocks + the
/// `N` branch blocks at once (OOM ≥ 512² on a 96 GB card); the checkpointed chain keeps only the
/// segment-boundary states. Topology: the chain state is `[main_h, branch_h]` through main blocks
/// `0..offset+N` — segments before the injection window pass the branch input through untouched;
/// segment `j` in the window advances branch block `j − offset`, projects + clamps its residual onto
/// `main_h`'s image tokens, then runs main block `j` — dropping to `[main_h]` after the last
/// injection, with the `velocity_out` + loss regression as the final segment. The pre-main
/// front-end + control embedding are frozen (no upstream `Var`s), so the seed boundary is simply
/// detached.
#[allow(clippy::too_many_arguments)]
pub fn control_loss_grads(
    dit: &KreaTrainDit,
    branch: &ControlBranch,
    x0: &Tensor,
    ctrl_latent: &Tensor,
    cap: &Tensor,
    sigma: f32,
    noise: &Tensor,
    mae: bool,
    compute_dtype: DType,
    use_checkpoint: bool,
) -> Result<(f32, GradStore)> {
    let device = x0.device();
    let (x_t, target) = flow_match::build_batch(x0, noise, sigma as f64)?;
    let x_t = x_t.to_dtype(compute_dtype)?;
    let context = cap.unsqueeze(0)?; // (L, n, d) -> (1, L, n, d)
    let t = Tensor::from_vec(vec![sigma], (1,), device)?;

    if !use_checkpoint {
        let v = forward_with_control(dit, branch, &x_t, &t, &context, ctrl_latent, 1.0)?;
        let loss = velocity_loss(&v, &target, mae)?;
        let loss_val = loss.to_dtype(DType::F32)?.to_scalar::<f32>()?;
        let grads = loss.backward()?;
        return Ok((loss_val, grads));
    }

    // Frozen pre-main + control embedding (no Vars upstream → the detached seed boundary).
    let (combined, ctx) = dit.forward_pre_main(&x_t, &t, &context)?;
    let ctrl_tokens = dit.embed_latent(ctrl_latent)?;
    let branch_input = {
        let txt = combined.narrow(1, 0, ctx.cap_len)?;
        let img = (combined.narrow(1, ctx.cap_len, ctx.img_len)? + &ctrl_tokens)?;
        Tensor::cat(&[&txt, &img], 1)?
    };

    let n = branch.blocks.len();
    let offset = branch.inject_offset;
    let blocks = dit.blocks();
    let ctx_ref = &ctx;
    let mut segs: Vec<Segment> = Vec::with_capacity(blocks.len() + 1);
    for (j, blk) in blocks.iter().enumerate() {
        if j < offset {
            // Before the injection window: [main_h, branch_h] -> run main block j, pass the
            // (still-unconsumed) branch input through the boundary untouched.
            segs.push(Box::new(move |st: &[Tensor]| {
                let mh = blk.forward(&st[0], &ctx_ref.tvec, &ctx_ref.rcos, &ctx_ref.rsin)?;
                Ok(vec![mh, st[1].clone()])
            }));
        } else if let Some(i) = branch.residual_index_for_main_block(j) {
            // [main_h, branch_h] -> advance branch block i = j - offset, inject its residual into
            // main block j's input, run main block j.
            let cblk = &branch.blocks[i];
            let drop_branch = i + 1 == n; // last injection: the branch stream leaves the state
            segs.push(Box::new(move |st: &[Tensor]| {
                use candle_gen::candle_core::Error as CoreError;
                let (main_h, branch_h) = (&st[0], &st[1]);
                let bh = cblk
                    .forward(branch_h, &ctx_ref.tvec, &ctx_ref.rcos, &ctx_ref.rsin, false)
                    .map_err(CoreError::wrap)?;
                let bh_img = bh.narrow(1, ctx_ref.cap_len, ctx_ref.img_len)?;
                let res = cblk
                    .proj_out
                    .forward(&bh_img, false)
                    .map_err(CoreError::wrap)?;
                let txt = main_h.narrow(1, 0, ctx_ref.cap_len)?;
                let img = main_h.narrow(1, ctx_ref.cap_len, ctx_ref.img_len)?;
                let res = branch
                    .apply_clamp(&res.to_dtype(main_h.dtype())?, &img)
                    .map_err(CoreError::wrap)?;
                let img = (img + res)?;
                let inj = Tensor::cat(&[&txt, &img], 1)?;
                let mh = blk.forward(&inj, &ctx_ref.tvec, &ctx_ref.rcos, &ctx_ref.rsin)?;
                if drop_branch {
                    Ok(vec![mh])
                } else {
                    Ok(vec![mh, bh])
                }
            }));
        } else {
            // Past the injection window: [main_h] -> plain frozen main block.
            segs.push(Box::new(move |st: &[Tensor]| {
                Ok(vec![blk.forward(
                    &st[0],
                    &ctx_ref.tvec,
                    &ctx_ref.rcos,
                    &ctx_ref.rsin,
                )?])
            }));
        }
    }
    // Final segment: the output head + the flow-match regression -> [loss].
    let target_owned = target.clone();
    segs.push(Box::new(move |st: &[Tensor]| {
        let v = dit.velocity_out(&st[0], ctx_ref)?;
        Ok(vec![velocity_loss(&v, &target_owned, mae)?])
    }));

    let seed = [combined.detach(), branch_input.detach()];
    let vars = branch.vars();
    checkpointed_backward(&segs, &seed, &vars)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testfix::{
        randn_seeded, tiny_batch, tiny_batch_seeded, tiny_dit, tiny_dit_layers, tiny_dit_seeded,
    };
    use candle_gen::candle_core::Device;
    use candle_gen::train::flow_match::velocity_loss;
    use candle_gen::train::optim::{clip_grad_norm, TrainOptimizer};
    use rand::{rngs::StdRng, SeedableRng};

    fn nudge_vars(branch: &ControlBranch, dev: &Device) {
        for v in branch.vars() {
            v.set(&Tensor::randn(0f32, 0.02f32, v.as_tensor().dims(), dev).unwrap())
                .unwrap();
        }
    }

    /// The keystone zero-init gate: with untouched (zero) output projections the branched forward is
    /// **exactly** the base forward at `control_scale = 1.0` — the ControlNet step-0 identity.
    #[test]
    fn zero_init_branch_is_identity() {
        let dev = Device::Cpu;
        let (dit, c, path) = tiny_dit();
        let w = Weights::from_file(&path, &dev, DType::F32).unwrap();
        let branch = ControlBranch::from_base(&w, &c, 1, DType::F32, 0).unwrap();

        let (x0, cap, _) = tiny_batch(&c);
        let ctrl = Tensor::randn(0f32, 1f32, x0.dims(), &dev).unwrap();
        let ctxt = cap.unsqueeze(0).unwrap();
        let t = Tensor::from_vec(vec![0.5f32], (1,), &dev).unwrap();

        let base = dit.forward(&x0, &t, &ctxt).unwrap();
        let with = forward_with_control(&dit, &branch, &x0, &t, &ctxt, &ctrl, 1.0).unwrap();
        let a = base.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = with.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(a, b, "zero-init projections must be a step-0 identity");
        let _ = std::fs::remove_file(path);
    }

    /// `control_scale = 0` short-circuits to the plain base forward even with a trained (nonzero)
    /// branch — the byte-identity contract of the inference hook.
    #[test]
    fn scale_zero_is_base_forward() {
        let dev = Device::Cpu;
        let (dit, c, path) = tiny_dit();
        let w = Weights::from_file(&path, &dev, DType::F32).unwrap();
        let branch = ControlBranch::from_base(&w, &c, 1, DType::F32, 0).unwrap();
        nudge_vars(&branch, &dev);

        let (x0, cap, _) = tiny_batch(&c);
        let ctrl = Tensor::randn(0f32, 1f32, x0.dims(), &dev).unwrap();
        let ctxt = cap.unsqueeze(0).unwrap();
        let t = Tensor::from_vec(vec![0.5f32], (1,), &dev).unwrap();

        let base = dit.forward(&x0, &t, &ctxt).unwrap();
        let with = forward_with_control(&dit, &branch, &x0, &t, &ctxt, &ctrl, 0.0).unwrap();
        let a = base.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = with.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(a, b, "scale 0 must be the un-branched forward");
        let _ = std::fs::remove_file(path);
    }

    /// Backprop reaches **every** branch `Var` (through the frozen main stack into the injected
    /// residuals) with finite gradients, and a few AdamW steps lower the loss on a fixed batch.
    #[test]
    fn backward_reaches_branch_and_descends() {
        let dev = Device::Cpu;
        // Draw the ENTIRE fixture — base DiT weights, branch nudge, probe batch, control latent —
        // from one seeded `StdRng`, for the same reason as `control_train::control_trainer_descends`
        // (sc-10794): candle's CPU `randn` is unseedable (it pulls the process-global `rand::rng()`),
        // so before this every draw was nondeterministic and a marginal 6-step descent could flip
        // sign on an unlucky init (or on ubuntu's float reassociation vs macos), red-failing CI. A
        // distinct fixed seed makes the loss trajectory reproducible run-to-run and platform-to-
        // platform; the larger step budget then buys an unambiguous drop (see the relative-floor
        // assert below). Seed is 10794-adjacent but distinct from the sibling tests so they don't
        // share a trajectory.
        let mut rng = StdRng::seed_from_u64(10795);
        let (dit, c, path) = tiny_dit_seeded(&mut rng);
        let w = Weights::from_file(&path, &dev, DType::F32).unwrap();
        let branch = ControlBranch::from_base(&w, &c, 1, DType::F32, 0).unwrap();
        // Seeded twin of `nudge_vars`: nudge off the zero-init identity so there's a signal to
        // descend, but through the seeded rng so the nudge is reproducible too (leaving the shared
        // unseeded `nudge_vars` untouched for the structural/identity tests that don't need a seed).
        for v in branch.vars() {
            v.set(&randn_seeded(&mut rng, 0.0, 0.02, v.as_tensor().dims()))
                .unwrap();
        }
        let vars = branch.vars();

        let (x0, cap, noise) = tiny_batch_seeded(&c, &mut rng);
        let ctrl = randn_seeded(&mut rng, 0.0, 1.0, x0.dims());
        let ctxt = cap.unsqueeze(0).unwrap();
        let t = Tensor::from_vec(vec![0.5f32], (1,), &dev).unwrap();
        let x_t = ((&x0 * 0.5).unwrap() + (&noise * 0.5).unwrap()).unwrap();
        let target = (&noise - &x0).unwrap();

        let loss_of = |()| -> (f32, candle_gen::candle_core::backprop::GradStore) {
            let v = forward_with_control(&dit, &branch, &x_t, &t, &ctxt, &ctrl, 1.0).unwrap();
            let loss = velocity_loss(&v, &target, false).unwrap();
            let lv = loss.to_vec0::<f32>().unwrap();
            (lv, loss.backward().unwrap())
        };

        let (loss0, grads) = loss_of(());
        assert!(loss0.is_finite());
        for (i, v) in vars.iter().enumerate() {
            let g = grads
                .get(v.as_tensor())
                .unwrap_or_else(|| panic!("branch var {i} has no gradient"));
            assert!(
                g.flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()
                    .iter()
                    .all(|x| x.is_finite()),
                "branch var {i} gradient has non-finite entries"
            );
        }

        let mut opt = TrainOptimizer::from_config("adamw", vars.clone(), 1e-2, 0.0).unwrap();
        let mut grads = grads;
        for _ in 0..60 {
            clip_grad_norm(&mut grads, &vars, 1.0).unwrap();
            opt.step(&grads).unwrap();
            let (_l, g) = loss_of(());
            grads = g;
        }
        let (loss1, _) = loss_of(());
        // A correctly working branch descends the fixed-batch loss by a wide margin over these 60
        // AdamW steps. Assert a >=10% relative drop rather than a bare `loss1 < loss0`: the 0.90 bar
        // sits far clear of the real ratio, so cross-platform float reassociation (the ubuntu-vs-macos
        // delta that flaked the sibling `control_trainer_descends`) cannot lift it over the bar, while
        // a branch that stopped learning (ratio ~1.0) still fails hard. This stays a genuine descent
        // gate, not a `<= before + epsilon` no-op.
        assert!(
            loss1 < loss0 * 0.9,
            "AdamW steps on a fixed batch should lower the loss by >=10%: {loss0} -> {loss1} (ratio {})",
            loss1 / loss0
        );
        let _ = std::fs::remove_file(path);
    }

    /// The correctness gate for the gradient-checkpointing lever: [`control_loss_grads`] with
    /// `use_checkpoint = true` (the segmented `[main_h, branch_h]` VJP) must reproduce the dense
    /// monolithic `loss.backward()` — same loss, same gradient on **every** branch `Var` (mod float
    /// reassociation). The control twin of the LoRA trainer's `dense_and_checkpoint_grads_match`.
    #[test]
    fn dense_and_checkpoint_grads_match_control() {
        let dev = Device::Cpu;
        let (dit, c, path) = tiny_dit();
        let w = Weights::from_file(&path, &dev, DType::F32).unwrap();
        let branch = ControlBranch::from_base(&w, &c, 1, DType::F32, 0).unwrap();
        nudge_vars(&branch, &dev);
        let vars = branch.vars();

        let (x0, cap, noise) = tiny_batch(&c);
        let ctrl = Tensor::randn(0f32, 1f32, x0.dims(), &dev).unwrap();

        let run = |ckpt: bool| {
            control_loss_grads(
                &dit,
                &branch,
                &x0,
                &ctrl,
                &cap,
                0.5,
                &noise,
                false,
                DType::F32,
                ckpt,
            )
            .unwrap()
        };
        let (loss_d, g_d) = run(false);
        let (loss_c, g_c) = run(true);

        assert!(
            (loss_d - loss_c).abs() < 1e-4,
            "loss: dense {loss_d} vs checkpoint {loss_c}"
        );
        let grad_vec = |g: &candle_gen::candle_core::backprop::GradStore, v: &Var| {
            g.get(v.as_tensor())
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        };
        let mut saw_nonzero = false;
        for (idx, v) in vars.iter().enumerate() {
            assert!(
                g_d.get(v.as_tensor()).is_some() && g_c.get(v.as_tensor()).is_some(),
                "var {idx} missing a gradient (dense or checkpoint)"
            );
            let a = grad_vec(&g_d, v);
            let b = grad_vec(&g_c, v);
            assert_eq!(a.len(), b.len());
            for (x, y) in a.iter().zip(b.iter()) {
                assert!(
                    (x - y).abs() < 1e-4,
                    "grad mismatch for var {idx} (dense {x} vs checkpoint {y})"
                );
                if x.abs() > 1e-6 {
                    saw_nonzero = true;
                }
            }
        }
        assert!(saw_nonzero, "expected nonzero branch grads to compare");
        let _ = std::fs::remove_file(path);
    }

    /// Train/infer branch-path parity (the sc-8460 step-500 probe regression gate): with the SAME
    /// nudged weights, the training-mode forward (graph-tracked Vars, as `control_loss_grads`
    /// applies the branch) and the frozen inference-mode forward (as `krea-control-infer` applies
    /// it) must be **exactly** equal — freezing changes graph tracking, never values — and
    /// [`probe_forward`]'s branched velocity must match both.
    #[test]
    fn train_and_infer_branch_paths_match() {
        let dev = Device::Cpu;
        let (dit, c, path) = tiny_dit();
        let w = Weights::from_file(&path, &dev, DType::F32).unwrap();
        let train_b = ControlBranch::from_base(&w, &c, 1, DType::F32, 0).unwrap();
        nudge_vars(&train_b, &dev);
        let ckpt = std::env::temp_dir().join(format!(
            "krea_ctrl_parity_{}.safetensors",
            std::process::id()
        ));
        train_b.save(&ckpt).unwrap();
        let mut infer_b = ControlBranch::from_checkpoint(&ckpt, &c, &dev).unwrap();
        infer_b.freeze();

        let (x0, cap, _) = tiny_batch(&c);
        let ctrl = Tensor::randn(0f32, 1f32, x0.dims(), &dev).unwrap();
        let ctxt = cap.unsqueeze(0).unwrap();
        let t = Tensor::from_vec(vec![0.5f32], (1,), &dev).unwrap();

        let flat = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let v_train =
            flat(&forward_with_control(&dit, &train_b, &x0, &t, &ctxt, &ctrl, 1.0).unwrap());
        let v_infer =
            flat(&forward_with_control(&dit, &infer_b, &x0, &t, &ctxt, &ctrl, 1.0).unwrap());
        assert_eq!(
            v_train, v_infer,
            "train-mode and frozen forwards must be identical"
        );

        let (report, v_probe, v_base) =
            probe_forward(&dit, &infer_b, &x0, &t, &ctxt, &ctrl, 1.0).unwrap();
        assert_eq!(
            flat(&v_probe),
            v_infer,
            "probe forward must match the real forward"
        );
        assert_eq!(report.len(), 1);
        assert!(report[0].0.is_finite() && report[0].1 > 0.0);
        // Nudged (nonzero) projections => the branched velocity differs from base.
        assert_ne!(flat(&v_base), v_infer);
        let _ = std::fs::remove_file(ckpt);
        let _ = std::fs::remove_file(path);
    }

    /// The residual RMS clamp (sc-8460 step-500 fix): with projections blown up far past the
    /// stream scale, the default clamp caps every injected residual at `τ × ‖main‖` (probe-verified),
    /// gradients still flow through the clamped path, an in-budget residual is untouched
    /// (clamp(None) == clamp(τ) when small), and the step-0 zero residual stays an exact identity.
    #[test]
    fn residual_clamp_caps_swamping() {
        let dev = Device::Cpu;
        let (dit, c, path) = tiny_dit();
        let w = Weights::from_file(&path, &dev, DType::F32).unwrap();

        // Blow the projections up: residuals many times the stream norm.
        let mut big = ControlBranch::from_base(&w, &c, 1, DType::F32, 0).unwrap();
        for (name, v) in &big.named {
            if name.contains("proj_out") {
                v.set(&Tensor::randn(0f32, 5.0f32, v.as_tensor().dims(), &dev).unwrap())
                    .unwrap();
            }
        }

        let (x0, cap, noise) = tiny_batch(&c);
        let ctrl = Tensor::randn(0f32, 1f32, x0.dims(), &dev).unwrap();
        let ctxt = cap.unsqueeze(0).unwrap();
        let t = Tensor::from_vec(vec![0.5f32], (1,), &dev).unwrap();

        // Unclamped: pre == post and the ratio exceeds τ (the degenerate regime).
        big.set_residual_clamp(None);
        let (rep_off, _, _) = probe_forward(&dit, &big, &x0, &t, &ctxt, &ctrl, 1.0).unwrap();
        assert_eq!(rep_off[0].0, rep_off[0].1, "no clamp => post == pre");
        assert!(
            rep_off[0].0 / rep_off[0].2 > DEFAULT_RESIDUAL_CLAMP * 1.5,
            "test setup should swamp: ratio {}",
            rep_off[0].0 / rep_off[0].2
        );

        // Clamped: the PRE-clamp ratio still reports the raw (over-budget) magnitude while the
        // POST-clamp ratio is capped at τ (probe norms are per-tensor L2 over equal element counts,
        // so the L2 ratio equals the RMS ratio).
        big.set_residual_clamp(Some(DEFAULT_RESIDUAL_CLAMP));
        let (rep_on, _, _) = probe_forward(&dit, &big, &x0, &t, &ctxt, &ctrl, 1.0).unwrap();
        assert!(
            rep_on[0].0 / rep_on[0].2 > DEFAULT_RESIDUAL_CLAMP * 1.5,
            "pre-clamp telemetry must keep reporting the raw magnitude"
        );
        assert!(
            rep_on[0].1 / rep_on[0].2 <= DEFAULT_RESIDUAL_CLAMP * 1.01,
            "clamp must cap the ratio: {}",
            rep_on[0].1 / rep_on[0].2
        );

        // Gradients still flow through the clamped path.
        let (loss, grads) = control_loss_grads(
            &dit,
            &big,
            &x0,
            &ctrl,
            &cap,
            0.5,
            &noise,
            false,
            DType::F32,
            false,
        )
        .unwrap();
        assert!(loss.is_finite());
        let some_proj = big
            .named
            .iter()
            .find(|(n, _)| n.contains("proj_out"))
            .map(|(_, v)| v.clone())
            .unwrap();
        assert!(
            grads.get(some_proj.as_tensor()).is_some(),
            "clamped residual must still carry gradient to the projection"
        );

        // An in-budget residual is untouched by the clamp (identical velocities), and the exact
        // zero-init residual stays an identity with the clamp on.
        let mut small = ControlBranch::from_base(&w, &c, 1, DType::F32, 0).unwrap();
        let flat = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let v_id = flat(&forward_with_control(&dit, &small, &x0, &t, &ctxt, &ctrl, 1.0).unwrap());
        assert_eq!(
            v_id,
            flat(&dit.forward(&x0, &t, &ctxt).unwrap()),
            "zero residual + clamp must stay a step-0 identity"
        );
        nudge_vars(&small, &dev);
        // Force the projections tiny so the residual is far under the τ budget (the generic nudge
        // is borderline at τ = 0.15).
        for (name, v) in &small.named {
            if name.contains("proj_out") {
                v.set(&Tensor::randn(0f32, 1e-4f32, v.as_tensor().dims(), &dev).unwrap())
                    .unwrap();
            }
        }
        small.set_residual_clamp(Some(DEFAULT_RESIDUAL_CLAMP));
        let v_on = flat(&forward_with_control(&dit, &small, &x0, &t, &ctxt, &ctrl, 1.0).unwrap());
        small.set_residual_clamp(None);
        let v_off = flat(&forward_with_control(&dit, &small, &x0, &t, &ctxt, &ctrl, 1.0).unwrap());
        assert_eq!(v_on, v_off, "an in-budget residual must pass unclamped");
        let _ = std::fs::remove_file(path);
    }

    /// Save → load roundtrip: the reloaded branch (N inferred from the file) produces the same
    /// branched forward as the original.
    #[test]
    fn checkpoint_roundtrip() {
        let dev = Device::Cpu;
        let (dit, c, path) = tiny_dit();
        let w = Weights::from_file(&path, &dev, DType::F32).unwrap();
        let branch = ControlBranch::from_base(&w, &c, 1, DType::F32, 0).unwrap();
        nudge_vars(&branch, &dev);

        let ckpt =
            std::env::temp_dir().join(format!("krea_ctrl_ckpt_{}.safetensors", std::process::id()));
        branch.save(&ckpt).unwrap();
        let mut loaded = ControlBranch::from_checkpoint(&ckpt, &c, &dev).unwrap();
        // Inference mode (detached weight reads) must not change values — only graph tracking.
        loaded.freeze();
        assert_eq!(loaded.num_blocks(), 1);
        assert_eq!(loaded.num_params(), branch.num_params());

        let (x0, cap, _) = tiny_batch(&c);
        let ctrl = Tensor::randn(0f32, 1f32, x0.dims(), &dev).unwrap();
        let ctxt = cap.unsqueeze(0).unwrap();
        let t = Tensor::from_vec(vec![0.5f32], (1,), &dev).unwrap();
        let a = forward_with_control(&dit, &branch, &x0, &t, &ctxt, &ctrl, 1.0)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let b = forward_with_control(&dit, &loaded, &x0, &t, &ctxt, &ctrl, 1.0)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(a, b, "checkpoint roundtrip must reproduce the forward");
        let _ = std::fs::remove_file(ckpt);
        let _ = std::fs::remove_file(path);
    }

    /// F-119 (sc-11208): a malformed/truncated overlay can ship a degenerate size-0
    /// `meta.inject_offset`. `from_checkpoint` must turn that into a typed error (via the hardened
    /// `read_scalar`), not panic on a `[0]` index of an empty tensor on the worker thread. The
    /// well-formed roundtrip above proves the success path is preserved.
    #[test]
    fn checkpoint_empty_inject_offset_is_typed_error() {
        let dev = Device::Cpu;
        let (_dit, c, path) = tiny_dit();
        let w = Weights::from_file(&path, &dev, DType::F32).unwrap();
        let branch = ControlBranch::from_base(&w, &c, 1, DType::F32, 0).unwrap();

        let ckpt = std::env::temp_dir().join(format!(
            "krea_ctrl_ckpt_empty_{}.safetensors",
            std::process::id()
        ));
        branch.save(&ckpt).unwrap();

        // Corrupt just the scalar meta tensor to size-0, re-save, and reload.
        let mut tensors = candle_gen::candle_core::safetensors::load(&ckpt, &dev).unwrap();
        tensors.insert(
            META_INJECT_OFFSET.to_string(),
            Tensor::from_vec(Vec::<f32>::new(), (0,), &dev).unwrap(),
        );
        candle_gen::candle_core::safetensors::save(&tensors, &ckpt).unwrap();

        let loaded = ControlBranch::from_checkpoint(&ckpt, &c, &dev);
        assert!(
            loaded.is_err(),
            "size-0 inject_offset must be a typed error, not a panic"
        );
        let _ = std::fs::remove_file(ckpt);
        let _ = std::fs::remove_file(path);
    }

    /// The injection offset (run-3 fix): with a 2-block main and `offset = 1`, the zero-init branch
    /// is still an exact identity; a nudged branch's residual lands on main block 1 only — the
    /// dense/checkpointed grads still match under the offset topology; the offset survives the
    /// checkpoint roundtrip (and its forward with it); and `from_base` rejects an offset that would
    /// push the injection window past the main stack.
    #[test]
    fn inject_offset_topology() {
        let dev = Device::Cpu;
        let (dit, c, path) = tiny_dit_layers(2);
        let w = Weights::from_file(&path, &dev, DType::F32).unwrap();

        // offset + n must fit in the main stack.
        assert!(ControlBranch::from_base(&w, &c, 2, DType::F32, 1).is_err());
        let branch = ControlBranch::from_base(&w, &c, 1, DType::F32, 1).unwrap();
        assert_eq!(branch.inject_offset(), 1);

        let (x0, cap, noise) = tiny_batch(&c);
        let ctrl = Tensor::randn(0f32, 1f32, x0.dims(), &dev).unwrap();
        let ctxt = cap.unsqueeze(0).unwrap();
        let t = Tensor::from_vec(vec![0.5f32], (1,), &dev).unwrap();
        let flat = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        // Zero-init identity holds under the offset.
        let v_id = flat(&forward_with_control(&dit, &branch, &x0, &t, &ctxt, &ctrl, 1.0).unwrap());
        assert_eq!(v_id, flat(&dit.forward(&x0, &t, &ctxt).unwrap()));

        // Nudged: dense vs checkpointed grads match under the offset topology.
        nudge_vars(&branch, &dev);
        let vars = branch.vars();
        let run = |ckpt: bool| {
            control_loss_grads(
                &dit,
                &branch,
                &x0,
                &ctrl,
                &cap,
                0.5,
                &noise,
                false,
                DType::F32,
                ckpt,
            )
            .unwrap()
        };
        let (loss_d, g_d) = run(false);
        let (loss_c, g_c) = run(true);
        assert!(
            (loss_d - loss_c).abs() < 1e-4,
            "offset loss: dense {loss_d} vs checkpoint {loss_c}"
        );
        for (idx, v) in vars.iter().enumerate() {
            let a = g_d.get(v.as_tensor()).unwrap().flatten_all().unwrap();
            let b = g_c.get(v.as_tensor()).unwrap().flatten_all().unwrap();
            for (x, y) in a
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .zip(b.to_vec1::<f32>().unwrap().iter())
            {
                assert!(
                    (x - y).abs() < 1e-4,
                    "offset grad mismatch for var {idx} (dense {x} vs checkpoint {y})"
                );
            }
        }

        // The offset is persisted and the reloaded forward matches.
        let ckpt =
            std::env::temp_dir().join(format!("krea_ctrl_off_{}.safetensors", std::process::id()));
        branch.save(&ckpt).unwrap();
        let loaded = ControlBranch::from_checkpoint(&ckpt, &c, &dev).unwrap();
        assert_eq!(loaded.inject_offset(), 1);
        let a = flat(&forward_with_control(&dit, &branch, &x0, &t, &ctxt, &ctrl, 1.0).unwrap());
        let b = flat(&forward_with_control(&dit, &loaded, &x0, &t, &ctxt, &ctrl, 1.0).unwrap());
        assert_eq!(
            a, b,
            "offset checkpoint roundtrip must reproduce the forward"
        );
        let _ = std::fs::remove_file(ckpt);
        let _ = std::fs::remove_file(path);
    }
}
