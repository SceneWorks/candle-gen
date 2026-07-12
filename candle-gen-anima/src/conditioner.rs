//! `AnimaTextConditioner` — the candle transcription of `mlx-gen-anima`'s `conditioner.rs` (diffusers
//! `condition_embedder_anima.py::AnimaTextConditioner`). Maps T5 token ids (learned query tokens via
//! `nn.Embedding(32128, 1024)`) + Qwen3 hidden states → the DiT's `encoder_hidden_states`: 6 blocks of
//! `[self-attn (RoPE θ=10000) → cross-attn into Qwen3 states → GELU MLP]`, then `out_proj` + RMSNorm,
//! and **right-pad to 512** so the DiT always sees exactly 512 text tokens.
//!
//! Weight keys are the `{prefix}.llm_adapter.`-**stripped** names (`blocks.N.*`, `embed.weight`,
//! `out_proj.*`, `norm.weight`) — identical to the diffusers `AnimaTextConditioner` state dict.

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::{Embedding, Module, VarBuilder};
use candle_gen::Result;

use crate::adapt::AdaptLinear;
use crate::config::ConditionerConfig;
use crate::nn::{apply_rope_half, rms_norm, sdpa};
use crate::rope::text_rope;

/// `(cos, sin)` RoPE tables, one pair per sequence length.
type Rope<'a> = (&'a Tensor, &'a Tensor);

/// `AnimaTextConditionerAttention`: q/k/v/o projections (no bias), per-head q/k RMSNorm (eps 1e-6),
/// half-split RoPE. Query positions and key positions can differ (cross-attn: target vs source).
struct Attention {
    q_proj: AdaptLinear,
    k_proj: AdaptLinear,
    v_proj: AdaptLinear,
    o_proj: AdaptLinear,
    q_norm: Tensor,
    k_norm: Tensor,
    heads: usize,
    head_dim: usize,
    scale: f64,
    eps: f64,
}

impl Attention {
    fn new(vb: &VarBuilder, prefix: &str, cfg: &ConditionerConfig) -> Result<Self> {
        let hd = cfg.head_dim();
        Ok(Self {
            q_proj: AdaptLinear::dense(vb, &format!("{prefix}.q_proj"))?,
            k_proj: AdaptLinear::dense(vb, &format!("{prefix}.k_proj"))?,
            v_proj: AdaptLinear::dense(vb, &format!("{prefix}.v_proj"))?,
            o_proj: AdaptLinear::dense(vb, &format!("{prefix}.o_proj"))?,
            q_norm: vb.get_unchecked(&format!("{prefix}.q_norm.weight"))?,
            k_norm: vb.get_unchecked(&format!("{prefix}.k_norm.weight"))?,
            heads: cfg.num_attention_heads,
            head_dim: hd,
            scale: (hd as f64).powf(-0.5),
            eps: cfg.norm_eps,
        })
    }

    /// `hidden`: `[B, Sq, D]` (query source). `encoder`: `Some([B, Sk, Ctx])` (cross) or `None`
    /// (self → `hidden`). `q_rope`/`k_rope` apply half-split RoPE with (possibly) different positions.
    /// Batch is 1 in this pipeline, so all tokens are real ⇒ no attention mask needed.
    fn forward(
        &self,
        hidden: &Tensor,
        encoder: Option<&Tensor>,
        q_rope: Rope,
        k_rope: Rope,
    ) -> Result<Tensor> {
        let (b, sq, _) = hidden.dims3()?;
        let kv_src = encoder.unwrap_or(hidden);
        let sk = kv_src.dim(1)?;

        let q = self
            .q_proj
            .forward(hidden)?
            .reshape((b, sq, self.heads, self.head_dim))?;
        let k = self
            .k_proj
            .forward(kv_src)?
            .reshape((b, sk, self.heads, self.head_dim))?;
        let v = self
            .v_proj
            .forward(kv_src)?
            .reshape((b, sk, self.heads, self.head_dim))?;

        let q = rms_norm(&q, &self.q_norm, self.eps)?;
        let k = rms_norm(&k, &self.k_norm, self.eps)?;
        let q = apply_rope_half(&q, q_rope.0, q_rope.1)?;
        let k = apply_rope_half(&k, k_rope.0, k_rope.1)?;

        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;
        let o = sdpa(&q, &k, &v, self.scale, None)?;
        let o = o
            .transpose(1, 2)?
            .reshape((b, sq, self.heads * self.head_dim))?;
        Ok(self.o_proj.forward(&o)?)
    }

    /// Visit this attention's four adaptable projections (`{prefix}.{q,k,v,o}_proj`) for the
    /// additive-adapter walk (sc-10640) — the conditioner's LoRA surface.
    fn visit_adaptable_mut(
        &mut self,
        prefix: &str,
        f: &mut dyn FnMut(&str, &mut AdaptLinear) -> Result<()>,
    ) -> Result<()> {
        f(&format!("{prefix}.q_proj"), &mut self.q_proj)?;
        f(&format!("{prefix}.k_proj"), &mut self.k_proj)?;
        f(&format!("{prefix}.v_proj"), &mut self.v_proj)?;
        f(&format!("{prefix}.o_proj"), &mut self.o_proj)?;
        Ok(())
    }
}

/// `AnimaTextConditionerBlock`: pre-RMSNorm self-attn → cross-attn → GELU MLP (all residual).
struct Block {
    norm_self_attn: Tensor,
    self_attn: Attention,
    norm_cross_attn: Tensor,
    cross_attn: Attention,
    norm_mlp: Tensor,
    mlp_in: AdaptLinear,  // mlp.0 (bias)
    mlp_out: AdaptLinear, // mlp.2 (bias)
    eps: f64,
}

impl Block {
    fn new(vb: &VarBuilder, prefix: &str, cfg: &ConditionerConfig) -> Result<Self> {
        Ok(Self {
            norm_self_attn: vb.get_unchecked(&format!("{prefix}.norm_self_attn.weight"))?,
            self_attn: Attention::new(vb, &format!("{prefix}.self_attn"), cfg)?,
            norm_cross_attn: vb.get_unchecked(&format!("{prefix}.norm_cross_attn.weight"))?,
            cross_attn: Attention::new(vb, &format!("{prefix}.cross_attn"), cfg)?,
            norm_mlp: vb.get_unchecked(&format!("{prefix}.norm_mlp.weight"))?,
            mlp_in: AdaptLinear::dense_bias(vb, &format!("{prefix}.mlp.0"))?,
            mlp_out: AdaptLinear::dense_bias(vb, &format!("{prefix}.mlp.2"))?,
            eps: cfg.norm_eps,
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        encoder: &Tensor,
        target_rope: Rope,
        source_rope: Rope,
    ) -> Result<Tensor> {
        // self-attn: q and k both use the target (T5 query-token) positions.
        let normed = rms_norm(hidden, &self.norm_self_attn, self.eps)?;
        let h = (hidden
            + self
                .self_attn
                .forward(&normed, None, target_rope, target_rope)?)?;
        // cross-attn: q uses target positions, k uses source (Qwen3) positions.
        let normed = rms_norm(&h, &self.norm_cross_attn, self.eps)?;
        let h = (&h
            + self
                .cross_attn
                .forward(&normed, Some(encoder), target_rope, source_rope)?)?;
        // GELU MLP (gelu_exact = erf GELU).
        let normed = rms_norm(&h, &self.norm_mlp, self.eps)?;
        let mlp = self
            .mlp_out
            .forward(&self.mlp_in.forward(&normed)?.gelu_erf()?)?;
        Ok((h + mlp)?)
    }

    /// Visit this block's 10 adaptable projections (self/cross attn q/k/v/o, `mlp.0`, `mlp.2`) for the
    /// additive-adapter walk (sc-10640). `prefix` carries the `llm_adapter.` namespace so the yielded
    /// paths match a trained adapter's `diffusion_model.llm_adapter.blocks.<i>.*` keys after stripping.
    fn visit_adaptable_mut(
        &mut self,
        prefix: &str,
        f: &mut dyn FnMut(&str, &mut AdaptLinear) -> Result<()>,
    ) -> Result<()> {
        self.self_attn
            .visit_adaptable_mut(&format!("{prefix}.self_attn"), f)?;
        self.cross_attn
            .visit_adaptable_mut(&format!("{prefix}.cross_attn"), f)?;
        f(&format!("{prefix}.mlp.0"), &mut self.mlp_in)?;
        f(&format!("{prefix}.mlp.2"), &mut self.mlp_out)?;
        Ok(())
    }
}

/// The full `AnimaTextConditioner`.
pub struct AnimaTextConditioner {
    embed: Embedding,
    blocks: Vec<Block>,
    out_proj: AdaptLinear, // bias
    norm: Tensor,
    cfg: ConditionerConfig,
    device: Device,
}

impl AnimaTextConditioner {
    /// `vb` is a VarBuilder rooted at the adapter (`"{dit_prefix}.llm_adapter"`).
    pub fn new(vb: &VarBuilder, cfg: ConditionerConfig) -> Result<Self> {
        let embed = Embedding::new(vb.get_unchecked("embed.weight")?, cfg.model_dim);
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            blocks.push(Block::new(vb, &format!("blocks.{i}"), &cfg)?);
        }
        Ok(Self {
            embed,
            blocks,
            out_proj: AdaptLinear::dense_bias(vb, "out_proj")?,
            norm: vb.get_unchecked("norm.weight")?,
            cfg,
            device: vb.device().clone(),
        })
    }

    /// `source_hidden`: `[B, Ss, source_dim]` Qwen3 states (already mask-multiplied), in `dtype`.
    /// `target_ids`: `[B, St]` **U32** T5 token ids (`St <= min_sequence_length`). Returns
    /// `[B, min_sequence_length(=512), target_dim]` in `dtype`.
    pub fn forward(
        &self,
        source_hidden: &Tensor,
        target_ids: &Tensor,
        dtype: DType,
    ) -> Result<Tensor> {
        let (b, ss, _) = source_hidden.dims3()?;
        let st = target_ids.dim(1)?;

        // learned query tokens from T5 ids (in_proj is Identity: model_dim == target_dim).
        let mut hidden = self.embed.forward(target_ids)?.to_dtype(dtype)?; // [B, St, D]

        let hd = self.cfg.head_dim();
        let theta = self.cfg.rope_theta as f64;
        let (t_cos, t_sin) = text_rope(st, hd, theta, &self.device)?; // target positions
        let (s_cos, s_sin) = text_rope(ss, hd, theta, &self.device)?; // source positions
        let target_rope = (&t_cos, &t_sin);
        let source_rope = (&s_cos, &s_sin);

        for block in &self.blocks {
            hidden = block.forward(&hidden, source_hidden, target_rope, source_rope)?;
        }
        let hidden = rms_norm(
            &self.out_proj.forward(&hidden)?,
            &self.norm,
            self.cfg.norm_eps,
        )?;

        // Right-pad to min_sequence_length so the DiT always sees exactly 512 text tokens.
        let min = self.cfg.min_sequence_length;
        if st < min {
            let pad = Tensor::zeros((b, min - st, self.cfg.target_dim), dtype, &self.device)?;
            Ok(Tensor::cat(&[&hidden, &pad], 1)?)
        } else {
            Ok(hidden)
        }
    }

    pub fn config(&self) -> &ConditionerConfig {
        &self.cfg
    }

    /// Walk every adaptable conditioner projection, invoking `f(path, &mut AdaptLinear)` with the
    /// **`llm_adapter.`-prefixed** dotted path (matching a trained adapter's key after namespace strip):
    /// each block's 10 targets + `out_proj`. The residual installer (sc-10640) routes forward-time
    /// residuals through this on a packed tier, where the conditioner stays dense bf16 but its adapter
    /// still applies unmerged.
    pub fn visit_adaptable_mut(
        &mut self,
        f: &mut dyn FnMut(&str, &mut AdaptLinear) -> Result<()>,
    ) -> Result<()> {
        for (i, blk) in self.blocks.iter_mut().enumerate() {
            blk.visit_adaptable_mut(&format!("llm_adapter.blocks.{i}"), f)?;
        }
        f("llm_adapter.out_proj", &mut self.out_proj)?;
        Ok(())
    }
}
