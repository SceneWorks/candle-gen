//! Attention Based Building Blocks
use candle_core::{DType, Result, Tensor, D};
use candle_gen::quant::QLinear;
use candle_gen::train::lora::{lora_linear_detect, lora_linear_no_bias_detect, LoraLinear};
use candle_nn as nn;
use candle_nn::Module;

// sc-9416: every attention/FF/proj Linear in this vendored UNet packed-detects through the shared
// `candle_gen::quant` seam — the MLX SDXL tiers (SceneWorks/sdxl-base-mlx q4/q8) pack the whole Linear
// surface (attn `to_q/k/v/out.0`, GEGLU `ff.net.0.proj`, `ff.net.2`, and the linear `proj_in/proj_out`),
// while convolutions + norms stay dense. A dense diffusers checkpoint has no `.scales` sibling, so
// `linear_detect` takes the plain dense path unchanged (the vendored-vs-stock parity test still holds).

#[derive(Debug)]
struct GeGlu {
    proj: QLinear,
    span: tracing::Span,
}

impl GeGlu {
    fn new(vs: nn::VarBuilder, dim_in: usize, dim_out: usize, group_size: usize) -> Result<Self> {
        let proj = QLinear::linear_detect_gs(dim_in, dim_out * 2, &vs, "proj", true, group_size)?;
        let span = tracing::span!(tracing::Level::TRACE, "geglu");
        Ok(Self { proj, span })
    }
}

impl Module for GeGlu {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let hidden_states_and_gate = self.proj.forward(xs)?.chunk(2, D::Minus1)?;
        &hidden_states_and_gate[0] * hidden_states_and_gate[1].gelu()?
    }
}

/// A feed-forward layer.
#[derive(Debug)]
struct FeedForward {
    project_in: GeGlu,
    linear: QLinear,
    span: tracing::Span,
}

impl FeedForward {
    // The glu parameter in the python code is unused?
    // https://github.com/huggingface/diffusers/blob/d3d22ce5a894becb951eec03e663951b28d45135/src/diffusers/models/attention.py#L347
    /// Creates a new feed-forward layer based on some given input dimension, some
    /// output dimension, and a multiplier to be used for the intermediary layer.
    fn new(
        vs: nn::VarBuilder,
        dim: usize,
        dim_out: Option<usize>,
        mult: usize,
        group_size: usize,
    ) -> Result<Self> {
        let inner_dim = dim * mult;
        let dim_out = dim_out.unwrap_or(dim);
        let vs = vs.pp("net");
        let project_in = GeGlu::new(vs.pp("0"), dim, inner_dim, group_size)?;
        let linear = QLinear::linear_detect_gs(inner_dim, dim_out, &vs, "2", true, group_size)?;
        let span = tracing::span!(tracing::Level::TRACE, "ff");
        Ok(Self {
            project_in,
            linear,
            span,
        })
    }
}

impl Module for FeedForward {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let xs = self.project_in.forward(xs)?;
        self.linear.forward(&xs)
    }
}

// The trainable vendored UNet (sc-5165) drops flash-attention: candle's flash kernel is a
// `CustomOp3` with no `bwd`, so it cannot back-propagate and is unusable in a training forward. The
// trainer constructs every attention with `use_flash_attn = false`, so this stub is never reached;
// it stays only to keep `CrossAttention::attention` structurally identical to the stock module this
// is vendored from. The live inference path is the dense math path, which now runs the vendored
// budgeted UNet; only the (currently-unreachable) flash-attn branch would use the stock UNet.
fn flash_attn(_: &Tensor, _: &Tensor, _: &Tensor, _: f32, _: bool) -> Result<Tensor> {
    unimplemented!("flash-attn has no backward in candle; the trainable UNet uses the math path")
}

#[derive(Debug)]
pub struct CrossAttention {
    // sc-5165: the four SDXL LoRA/LoKr target projections are `LoraLinear` (frozen base + optional
    // trainable residual) rather than `nn::Linear`. With no adapter installed they are exactly the
    // stock `nn::Linear` (forward parity is pinned by a CPU test), so inference is unchanged.
    to_q: LoraLinear,
    to_k: LoraLinear,
    to_v: LoraLinear,
    to_out: LoraLinear,
    heads: usize,
    scale: f64,
    span: tracing::Span,
    span_attn: tracing::Span,
    span_softmax: tracing::Span,
    use_flash_attn: bool,
    // IP-Adapter decoupled cross-attention (sc-5491): extra bias-free K/V projections for the image /
    // identity tokens, installed only on the cross-attn (`attn2`) modules. When present AND IP tokens
    // are set, `o += ip_scale · sdpa(q, to_k_ip(ip), to_v_ip(ip))` before the output projection. Both
    // `None` on a plain (training / stock) attention, so `forward` is byte-unchanged there — the
    // vendored-vs-stock parity test still holds.
    to_k_ip: Option<nn::Linear>,
    to_v_ip: Option<nn::Linear>,
    // The IP tokens + scale, constant across the denoise (computed once from the reference embedding),
    // so set once per generation via [`set_ip`](CrossAttention::set_ip) rather than threaded per step.
    ip_tokens: Option<Tensor>,
    ip_scale: f64,
    // sc-9424 (F-054 sibling): the head-shaped image-token K/V (`[B·heads, N_ip, head_dim]` each),
    // precomputed ONCE from the constant `ip_tokens` at [`set_ip`](CrossAttention::set_ip) time and
    // reused on every denoise step. `to_k_ip(ip_tokens)`/`to_v_ip(ip_tokens)` are step-invariant — the
    // tokens and the projection weights never change across a render — so projecting them per step was
    // pure wasted recompute (the identical finding F-054 fixed for the FLUX XLabs IP-Adapter). Cached
    // here (not in a `RefCell`) because `set_ip` already runs per-render with `&mut self` and the exact
    // constant tokens: the cache is populated at set time and cleared when the tokens are cleared/changed,
    // so it is naturally scoped to a single render and can never serve stale K/V from a prior generate().
    // The reuse is bit-identical to recomputing the projections each step.
    ip_kv: Option<(Tensor, Tensor)>,
}

impl CrossAttention {
    // Defaults should be heads = 8, dim_head = 64, context_dim = None
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vs: nn::VarBuilder,
        query_dim: usize,
        context_dim: Option<usize>,
        heads: usize,
        dim_head: usize,
        use_flash_attn: bool,
        group_size: usize,
    ) -> Result<Self> {
        let inner_dim = dim_head * heads;
        let context_dim = context_dim.unwrap_or(query_dim);
        let scale = 1.0 / f64::sqrt(dim_head as f64);
        // sc-9416: the four SDXL attention projections packed-detect their frozen base — a packed MLX
        // tier loads a `QLinear` base (inference-only, no adapter); a dense checkpoint loads the plain
        // `nn::Linear` base exactly as before, so the LoRA/LoKr trainer + IP-adapter paths are unchanged.
        let to_q = lora_linear_no_bias_detect(query_dim, inner_dim, &vs, "to_q", group_size)?;
        let to_k = lora_linear_no_bias_detect(context_dim, inner_dim, &vs, "to_k", group_size)?;
        let to_v = lora_linear_no_bias_detect(context_dim, inner_dim, &vs, "to_v", group_size)?;
        let to_out = lora_linear_detect(inner_dim, query_dim, &vs, "to_out.0", group_size)?;
        let span = tracing::span!(tracing::Level::TRACE, "xa");
        let span_attn = tracing::span!(tracing::Level::TRACE, "xa-attn");
        let span_softmax = tracing::span!(tracing::Level::TRACE, "xa-softmax");
        Ok(Self {
            to_q,
            to_k,
            to_v,
            to_out,
            heads,
            scale,
            span,
            span_attn,
            span_softmax,
            use_flash_attn,
            to_k_ip: None,
            to_v_ip: None,
            ip_tokens: None,
            ip_scale: 0.0,
            ip_kv: None,
        })
    }

    fn reshape_heads_to_batch_dim(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, dim) = xs.dims3()?;
        xs.reshape((batch_size, seq_len, self.heads, dim / self.heads))?
            .transpose(1, 2)?
            .reshape((batch_size * self.heads, seq_len, dim / self.heads))
    }

    fn reshape_batch_dim_to_heads(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, dim) = xs.dims3()?;
        xs.reshape((batch_size / self.heads, self.heads, seq_len, dim))?
            .transpose(1, 2)?
            .reshape((batch_size / self.heads, seq_len, dim * self.heads))
    }

    fn attention(&self, query: &Tensor, key: &Tensor, value: &Tensor) -> Result<Tensor> {
        let _enter = self.span_attn.enter();
        let xs = if self.use_flash_attn {
            let init_dtype = query.dtype();
            let q = query
                .to_dtype(candle_core::DType::F16)?
                .unsqueeze(0)?
                .transpose(1, 2)?;
            let k = key
                .to_dtype(candle_core::DType::F16)?
                .unsqueeze(0)?
                .transpose(1, 2)?;
            let v = value
                .to_dtype(candle_core::DType::F16)?
                .unsqueeze(0)?
                .transpose(1, 2)?;
            flash_attn(&q, &k, &v, self.scale as f32, false)?
                .transpose(1, 2)?
                .squeeze(0)?
                .to_dtype(init_dtype)?
        } else {
            let in_dtype = query.dtype();
            let query = query.to_dtype(DType::F32)?;
            let key = key.to_dtype(DType::F32)?;
            let value = value.to_dtype(DType::F32)?;
            // The composable `softmax` (not the fused `softmax_last_dim`): the fused kernel is a
            // `CustomOp` with no backward, so grads would never reach `to_q`/`to_k` through the scores
            // (sc-5165). Numerically identical, so the stock forward-parity test still holds.
            //
            // i32-overflow guard (sc-9116): `query`/`key`/`value` are `[B·heads, seq, dim]` (heads folded
            // into batch). The self-attn scores `[B·heads, seq, seq]` reach `8·65536² ≈ 3.4e10 > i32::MAX`
            // at a 2048² render (256×256 top-block latent tokens), silently corrupting the tail rows on
            // the candle CUDA kernels. The shared budgeted helper chunks over the query rows
            // (byte-identical for common sizes; cross-attn to the fixed 77-token context is a single pass).
            // `self.scale` is already applied to `key` in the un-guarded path via `key.t()·scale`; the
            // helper applies it to the scores instead (`(q·kᵀ)·scale` == `q·(kᵀ·scale)`).
            let _enter = self.span_softmax.enter();
            candle_gen::sdpa_budgeted_flat(
                &query,
                &key,
                &value,
                self.scale,
                |s| nn::ops::softmax(s, D::Minus1),
                candle_gen::ATTN_SCORES_BUDGET,
            )?
            .to_dtype(in_dtype)?
        };
        self.reshape_batch_dim_to_heads(&xs)
    }

    pub fn forward(&self, xs: &Tensor, context: Option<&Tensor>) -> Result<Tensor> {
        let _enter = self.span.enter();
        let query = self.to_q.forward(xs)?;
        let context = context.unwrap_or(xs).contiguous()?;
        let key = self.to_k.forward(&context)?;
        let value = self.to_v.forward(&context)?;
        let query = self.reshape_heads_to_batch_dim(&query)?;
        let key = self.reshape_heads_to_batch_dim(&key)?;
        let value = self.reshape_heads_to_batch_dim(&value)?;
        let mut xs = self.attention(&query, &key, &value)?;
        // IP-Adapter decoupled branch (sc-5491): reuse the text query against the image/identity
        // tokens' own K/V, scaled and added before the output projection. A no-op (skipped) unless the
        // K/V cache is populated (installed AND tokens set) — so the training / stock path is unchanged.
        // sc-9424 (F-054 sibling): the image-token K/V are precomputed ONCE in `set_ip` (they are
        // step-invariant) and reused here every step — bit-identical to `to_k_ip(ip_tokens)` /
        // `to_v_ip(ip_tokens)` per step, but without the redundant per-step projection.
        if let Some((key_ip, value_ip)) = &self.ip_kv {
            let ip_out = self.attention(&query, key_ip, value_ip)?;
            xs = (xs + (ip_out * self.ip_scale)?)?;
        }
        self.to_out.forward(&xs)
    }

    /// Visit the four adaptable projections (`to_q`/`to_k`/`to_v`/`to_out.0`) so a
    /// [`LoraHost`](candle_gen::train::lora::LoraHost) can install or clear adapters (sc-5165).
    pub fn visit_lora_mut(
        &mut self,
        f: &mut dyn FnMut(&mut LoraLinear) -> candle_gen::Result<()>,
    ) -> candle_gen::Result<()> {
        f(&mut self.to_q)?;
        f(&mut self.to_k)?;
        f(&mut self.to_v)?;
        f(&mut self.to_out)?;
        Ok(())
    }

    /// Install the IP-Adapter decoupled K/V projections (sc-5491): `k_ip`/`v_ip` are the
    /// `ip_adapter.{n}.to_k_ip/to_v_ip` weights (`[inner, cross_attention_dim]`, bias-free).
    pub fn install_ip(&mut self, k_ip: Tensor, v_ip: Tensor) {
        self.to_k_ip = Some(nn::Linear::new(k_ip, None));
        self.to_v_ip = Some(nn::Linear::new(v_ip, None));
        // sc-9424: any precomputed K/V from an earlier `set_ip` was projected with the OLD weights;
        // drop it so the next `set_ip` rebuilds against these freshly-installed projections. (Install is
        // a one-time load-time op that always precedes `set_ip`, so this is defensive, not hot-path.)
        self.ip_kv = None;
    }

    /// Set (or clear, with `None`) the IP tokens + scale used by [`forward`](Self::forward)'s decoupled
    /// branch. Constant across the denoise, so set once per generation; the clone is cheap (the face
    /// tokens are `[B, 16, 2048]`).
    ///
    /// sc-9424 (F-054 sibling): because the image tokens are step-invariant, this also PRECOMPUTES the
    /// head-shaped IP K/V (`to_k_ip(tokens)` / `to_v_ip(tokens)`, reshaped to `[B·heads, N_ip, head_dim]`)
    /// ONCE here — the `forward` decoupled branch then reuses the cache every step instead of re-running
    /// the projections per step. Clearing the tokens (or passing new ones) rebuilds/clears the cache, so
    /// it is scoped to a single render and never serves stale K/V across `generate()` calls. Returns an
    /// error only if the projection reshape fails; a no-op (clears the cache) when the decoupled K/V are
    /// not installed or `tokens` is `None`.
    pub fn set_ip(&mut self, tokens: Option<&Tensor>, scale: f64) -> Result<()> {
        self.ip_tokens = tokens.cloned();
        self.ip_scale = scale;
        self.ip_kv = match (&self.to_k_ip, &self.to_v_ip, &self.ip_tokens) {
            (Some(to_k_ip), Some(to_v_ip), Some(ip_tokens)) => {
                let key_ip = self.reshape_heads_to_batch_dim(&to_k_ip.forward(ip_tokens)?)?;
                let value_ip = self.reshape_heads_to_batch_dim(&to_v_ip.forward(ip_tokens)?)?;
                Some((key_ip, value_ip))
            }
            _ => None,
        };
        Ok(())
    }

    /// Test-only: whether the four attention projections loaded packed (a pre-quantized MLX tier) —
    /// used to assert sc-9416 packed-detect fired on the SDXL `to_q/k/v/out.0` key layout.
    #[cfg(test)]
    pub(crate) fn all_projections_packed(&self) -> bool {
        self.to_q.is_packed()
            && self.to_k.is_packed()
            && self.to_v.is_packed()
            && self.to_out.is_packed()
    }
}

/// A basic Transformer block.
#[derive(Debug)]
struct BasicTransformerBlock {
    attn1: CrossAttention,
    ff: FeedForward,
    attn2: CrossAttention,
    norm1: nn::LayerNorm,
    norm2: nn::LayerNorm,
    norm3: nn::LayerNorm,
    span: tracing::Span,
}

impl BasicTransformerBlock {
    #[allow(clippy::too_many_arguments)]
    fn new(
        vs: nn::VarBuilder,
        dim: usize,
        n_heads: usize,
        d_head: usize,
        context_dim: Option<usize>,
        use_flash_attn: bool,
        group_size: usize,
    ) -> Result<Self> {
        let attn1 = CrossAttention::new(
            vs.pp("attn1"),
            dim,
            None,
            n_heads,
            d_head,
            use_flash_attn,
            group_size,
        )?;
        let ff = FeedForward::new(vs.pp("ff"), dim, None, 4, group_size)?;
        let attn2 = CrossAttention::new(
            vs.pp("attn2"),
            dim,
            context_dim,
            n_heads,
            d_head,
            use_flash_attn,
            group_size,
        )?;
        let norm1 = nn::layer_norm(dim, 1e-5, vs.pp("norm1"))?;
        let norm2 = nn::layer_norm(dim, 1e-5, vs.pp("norm2"))?;
        let norm3 = nn::layer_norm(dim, 1e-5, vs.pp("norm3"))?;
        let span = tracing::span!(tracing::Level::TRACE, "basic-transformer");
        Ok(Self {
            attn1,
            ff,
            attn2,
            norm1,
            norm2,
            norm3,
            span,
        })
    }

    fn forward(&self, xs: &Tensor, context: Option<&Tensor>) -> Result<Tensor> {
        let _enter = self.span.enter();
        let xs = (self.attn1.forward(&self.norm1.forward(xs)?, None)? + xs)?;
        let xs = (self.attn2.forward(&self.norm2.forward(&xs)?, context)? + xs)?;
        self.ff.forward(&self.norm3.forward(&xs)?)? + xs
    }

    fn visit_lora_mut(
        &mut self,
        f: &mut dyn FnMut(&mut LoraLinear) -> candle_gen::Result<()>,
    ) -> candle_gen::Result<()> {
        self.attn1.visit_lora_mut(f)?;
        self.attn2.visit_lora_mut(f)?;
        Ok(())
    }

    /// Visit this block's **cross-attention** (`attn2` only — never the self-attn `attn1`) for the
    /// IP-Adapter install / token-set walk (sc-5491).
    fn visit_cross_attn_mut(
        &mut self,
        f: &mut dyn FnMut(&mut CrossAttention) -> Result<()>,
    ) -> Result<()> {
        f(&mut self.attn2)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SpatialTransformerConfig {
    pub depth: usize,
    pub num_groups: usize,
    pub context_dim: Option<usize>,
    pub use_linear_projection: bool,
}

impl Default for SpatialTransformerConfig {
    fn default() -> Self {
        Self {
            depth: 1,
            num_groups: 32,
            context_dim: None,
            use_linear_projection: false,
        }
    }
}

#[derive(Debug)]
enum Proj {
    Conv2d(nn::Conv2d),
    // sc-9416: the linear projection variant (SDXL's `use_linear_projection = true`) packed-detects.
    Linear(QLinear),
}

// Aka Transformer2DModel
#[derive(Debug)]
pub struct SpatialTransformer {
    norm: nn::GroupNorm,
    proj_in: Proj,
    transformer_blocks: Vec<BasicTransformerBlock>,
    proj_out: Proj,
    span: tracing::Span,
    pub config: SpatialTransformerConfig,
}

impl SpatialTransformer {
    pub fn new(
        vs: nn::VarBuilder,
        in_channels: usize,
        n_heads: usize,
        d_head: usize,
        use_flash_attn: bool,
        config: SpatialTransformerConfig,
    ) -> Result<Self> {
        Self::new_gs(
            vs,
            in_channels,
            n_heads,
            d_head,
            use_flash_attn,
            config,
            candle_gen::quant::MLX_GROUP_SIZE,
        )
    }

    /// As [`new`](Self::new), but at an explicit MLX packed `group_size` (sc-9416). The linear
    /// `proj_in`/`proj_out` (SDXL's `use_linear_projection = true`) packed-detect; the conv-projection
    /// variant stays dense (SDXL doesn't use it, and MLX affine-packs only Linear/matmul weights).
    #[allow(clippy::too_many_arguments)]
    pub fn new_gs(
        vs: nn::VarBuilder,
        in_channels: usize,
        n_heads: usize,
        d_head: usize,
        use_flash_attn: bool,
        config: SpatialTransformerConfig,
        group_size: usize,
    ) -> Result<Self> {
        let inner_dim = n_heads * d_head;
        let norm = nn::group_norm(config.num_groups, in_channels, 1e-6, vs.pp("norm"))?;
        let proj_in = if config.use_linear_projection {
            Proj::Linear(QLinear::linear_detect_gs(
                in_channels,
                inner_dim,
                &vs,
                "proj_in",
                true,
                group_size,
            )?)
        } else {
            Proj::Conv2d(nn::conv2d(
                in_channels,
                inner_dim,
                1,
                Default::default(),
                vs.pp("proj_in"),
            )?)
        };
        let mut transformer_blocks = vec![];
        let vs_tb = vs.pp("transformer_blocks");
        for index in 0..config.depth {
            let tb = BasicTransformerBlock::new(
                vs_tb.pp(index.to_string()),
                inner_dim,
                n_heads,
                d_head,
                config.context_dim,
                use_flash_attn,
                group_size,
            )?;
            transformer_blocks.push(tb)
        }
        let proj_out = if config.use_linear_projection {
            Proj::Linear(QLinear::linear_detect_gs(
                in_channels,
                inner_dim,
                &vs,
                "proj_out",
                true,
                group_size,
            )?)
        } else {
            Proj::Conv2d(nn::conv2d(
                inner_dim,
                in_channels,
                1,
                Default::default(),
                vs.pp("proj_out"),
            )?)
        };
        let span = tracing::span!(tracing::Level::TRACE, "spatial-transformer");
        Ok(Self {
            norm,
            proj_in,
            transformer_blocks,
            proj_out,
            span,
            config,
        })
    }

    pub fn forward(&self, xs: &Tensor, context: Option<&Tensor>) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (batch, _channel, height, weight) = xs.dims4()?;
        let residual = xs;
        let xs = self.norm.forward(xs)?;
        let (inner_dim, xs) = match &self.proj_in {
            Proj::Conv2d(p) => {
                let xs = p.forward(&xs)?;
                let inner_dim = xs.dim(1)?;
                let xs = xs
                    .transpose(1, 2)?
                    .t()?
                    .reshape((batch, height * weight, inner_dim))?;
                (inner_dim, xs)
            }
            Proj::Linear(p) => {
                let inner_dim = xs.dim(1)?;
                let xs = xs
                    .transpose(1, 2)?
                    .t()?
                    .reshape((batch, height * weight, inner_dim))?;
                (inner_dim, p.forward(&xs)?)
            }
        };
        let mut xs = xs;
        for block in self.transformer_blocks.iter() {
            xs = block.forward(&xs, context)?
        }
        let xs = match &self.proj_out {
            Proj::Conv2d(p) => p.forward(
                &xs.reshape((batch, height, weight, inner_dim))?
                    .t()?
                    .transpose(1, 2)?,
            )?,
            Proj::Linear(p) => p
                .forward(&xs)?
                .reshape((batch, height, weight, inner_dim))?
                .t()?
                .transpose(1, 2)?,
        };
        xs + residual
    }

    /// Visit every adaptable attention projection in this transformer's blocks (sc-5165).
    pub fn visit_lora_mut(
        &mut self,
        f: &mut dyn FnMut(&mut LoraLinear) -> candle_gen::Result<()>,
    ) -> candle_gen::Result<()> {
        for block in self.transformer_blocks.iter_mut() {
            block.visit_lora_mut(f)?;
        }
        Ok(())
    }

    /// Visit every transformer block's cross-attention for the IP-Adapter install / token-set walk
    /// (sc-5491) — in block order, so an install consumes the K/V pairs in the diffusers attn order.
    pub fn visit_cross_attn_mut(
        &mut self,
        f: &mut dyn FnMut(&mut CrossAttention) -> Result<()>,
    ) -> Result<()> {
        for block in self.transformer_blocks.iter_mut() {
            block.visit_cross_attn_mut(f)?;
        }
        Ok(())
    }

    /// Test-only: whether the linear `proj_in`/`proj_out` loaded packed (sc-9416). Conv-projection
    /// variants are never packed (MLX affine-packs only Linear weights), so this is `false` for them.
    #[cfg(test)]
    pub(crate) fn linear_projs_packed(&self) -> bool {
        matches!(&self.proj_in, Proj::Linear(q) if q.is_quantized())
            && matches!(&self.proj_out, Proj::Linear(q) if q.is_quantized())
    }

    /// Test-only: whether every transformer block's self- and cross-attention projections loaded packed.
    #[cfg(test)]
    pub(crate) fn all_block_attn_packed(&self) -> bool {
        self.transformer_blocks
            .iter()
            .all(|b| b.attn1.all_projections_packed() && b.attn2.all_projections_packed())
    }
}

/// Configuration for an attention block.
#[derive(Debug, Clone, Copy)]
pub struct AttentionBlockConfig {
    pub num_head_channels: Option<usize>,
    pub num_groups: usize,
    pub rescale_output_factor: f64,
    pub eps: f64,
}

impl Default for AttentionBlockConfig {
    fn default() -> Self {
        Self {
            num_head_channels: None,
            num_groups: 32,
            rescale_output_factor: 1.,
            eps: 1e-5,
        }
    }
}

#[derive(Debug)]
pub struct AttentionBlock {
    group_norm: nn::GroupNorm,
    query: nn::Linear,
    key: nn::Linear,
    value: nn::Linear,
    proj_attn: nn::Linear,
    channels: usize,
    num_heads: usize,
    span: tracing::Span,
    config: AttentionBlockConfig,
}

// In the .safetensor weights of official Stable Diffusion 3 Medium Huggingface repo
// https://huggingface.co/stabilityai/stable-diffusion-3-medium
// Linear layer may use a different dimension for the weight in the linear, which is
// incompatible with the current implementation of the nn::linear constructor.
// This is a workaround to handle the different dimensions.
fn get_qkv_linear(channels: usize, vs: nn::VarBuilder) -> Result<nn::Linear> {
    match vs.get((channels, channels), "weight") {
        Ok(_) => nn::linear(channels, channels, vs),
        Err(_) => {
            let weight = vs
                .get((channels, channels, 1, 1), "weight")?
                .reshape((channels, channels))?;
            let bias = vs.get((channels,), "bias")?;
            Ok(nn::Linear::new(weight, Some(bias)))
        }
    }
}

impl AttentionBlock {
    pub fn new(vs: nn::VarBuilder, channels: usize, config: AttentionBlockConfig) -> Result<Self> {
        let num_head_channels = config.num_head_channels.unwrap_or(channels);
        let num_heads = channels / num_head_channels;
        let group_norm =
            nn::group_norm(config.num_groups, channels, config.eps, vs.pp("group_norm"))?;
        let (q_path, k_path, v_path, out_path) = if vs.contains_tensor("to_q.weight") {
            ("to_q", "to_k", "to_v", "to_out.0")
        } else {
            ("query", "key", "value", "proj_attn")
        };
        let query = get_qkv_linear(channels, vs.pp(q_path))?;
        let key = get_qkv_linear(channels, vs.pp(k_path))?;
        let value = get_qkv_linear(channels, vs.pp(v_path))?;
        let proj_attn = get_qkv_linear(channels, vs.pp(out_path))?;
        let span = tracing::span!(tracing::Level::TRACE, "attn-block");
        Ok(Self {
            group_norm,
            query,
            key,
            value,
            proj_attn,
            channels,
            num_heads,
            span,
            config,
        })
    }

    fn transpose_for_scores(&self, xs: Tensor) -> Result<Tensor> {
        let (batch, t, h_times_d) = xs.dims3()?;
        xs.reshape((batch, t, self.num_heads, h_times_d / self.num_heads))?
            .transpose(1, 2)
    }
}

impl Module for AttentionBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let in_dtype = xs.dtype();
        let residual = xs;
        let (batch, channel, height, width) = xs.dims4()?;
        let xs = self
            .group_norm
            .forward(xs)?
            .reshape((batch, channel, height * width))?
            .transpose(1, 2)?;

        let query_proj = self.query.forward(&xs)?;
        let key_proj = self.key.forward(&xs)?;
        let value_proj = self.value.forward(&xs)?;

        let query_states = self
            .transpose_for_scores(query_proj)?
            .to_dtype(DType::F32)?;
        let key_states = self.transpose_for_scores(key_proj)?.to_dtype(DType::F32)?;
        let value_states = self
            .transpose_for_scores(value_proj)?
            .to_dtype(DType::F32)?;

        // scale is applied twice, hence the -0.25 here rather than -0.5.
        // https://github.com/huggingface/diffusers/blob/d3d22ce5a894becb951eec03e663951b28d45135/src/diffusers/models/attention.py#L87
        let scale = f64::powf(self.channels as f64 / self.num_heads as f64, -0.25);
        let attention_scores = (query_states * scale)?.matmul(&(key_states.t()? * scale)?)?;
        let attention_probs = nn::ops::softmax(&attention_scores, D::Minus1)?;

        // TODO: revert the call to force_contiguous once the three matmul kernels have been
        // adapted to handle layout with some dims set to 1.
        let xs = attention_probs.matmul(&value_states)?;
        let xs = xs.to_dtype(in_dtype)?;
        let xs = xs.transpose(1, 2)?.contiguous()?;
        let xs = xs.flatten_from(D::Minus2)?;
        let xs = self
            .proj_attn
            .forward(&xs)?
            .t()?
            .reshape((batch, channel, height, width))?;
        (xs + residual)? / self.config.rescale_output_factor
    }
}

#[cfg(test)]
mod ip_tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::{VarBuilder, VarMap};

    /// sc-5491: the decoupled IP-Adapter branch is a no-op until installed AND tokens are set; at
    /// `ip_scale = 0` it is byte-identical to the plain cross-attention (so the stock/training path is
    /// untouched); a positive scale changes the output; clearing the tokens reverts it. This pins the
    /// `o += ip_scale·sdpa(q, k_ip, v_ip)` injection that the whole InstantID identity path rides on.
    #[test]
    fn cross_attention_ip_branch_is_gated_and_scaled() {
        let dev = Device::Cpu;
        let vb = VarBuilder::from_varmap(&VarMap::new(), DType::F32, &dev);
        let (query_dim, ctx_dim, heads, dim_head) = (16usize, 24usize, 4usize, 4usize);
        let inner = heads * dim_head; // 16
        let mut xa = CrossAttention::new(
            vb,
            query_dim,
            Some(ctx_dim),
            heads,
            dim_head,
            false,
            candle_gen::quant::MLX_GROUP_SIZE,
        )
        .unwrap();

        let xs = Tensor::randn(0f32, 1f32, (2, 5, query_dim), &dev).unwrap(); // [B, Nq, query_dim]
        let ctx = Tensor::randn(0f32, 1f32, (2, 7, ctx_dim), &dev).unwrap(); // [B, S, cross_attention_dim]
        let base = xa.forward(&xs, Some(&ctx)).unwrap();
        let maxdiff = |a: &Tensor, b: &Tensor| {
            (a - b)
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap()
        };

        // Install the decoupled K/V (`[inner, cross_attention_dim]`) and set tokens at scale 0.
        let k_ip = Tensor::randn(0f32, 1f32, (inner, ctx_dim), &dev).unwrap();
        let v_ip = Tensor::randn(0f32, 1f32, (inner, ctx_dim), &dev).unwrap();
        xa.install_ip(k_ip, v_ip);
        let ip_tokens = Tensor::randn(0f32, 1f32, (2, 3, ctx_dim), &dev).unwrap(); // [B, N_ip, cross_attention_dim]
        xa.set_ip(Some(&ip_tokens), 0.0).unwrap();
        assert!(
            maxdiff(&xa.forward(&xs, Some(&ctx)).unwrap(), &base) < 1e-6,
            "ip_scale=0 must equal the no-IP output"
        );

        // A positive scale shifts the output.
        xa.set_ip(Some(&ip_tokens), 0.8).unwrap();
        assert!(
            maxdiff(&xa.forward(&xs, Some(&ctx)).unwrap(), &base) > 1e-4,
            "ip_scale>0 must change the output"
        );

        // Clearing the tokens reverts to the plain cross-attention.
        xa.set_ip(None, 0.0).unwrap();
        assert!(
            maxdiff(&xa.forward(&xs, Some(&ctx)).unwrap(), &base) < 1e-6,
            "clearing the IP tokens reverts to base"
        );
    }

    /// sc-9424 (F-054 sibling): the IP K/V precomputed once in `set_ip` and reused across steps must be
    /// **bit-identical** to projecting `to_k_ip(ip_tokens)` / `to_v_ip(ip_tokens)` every step. We call
    /// `forward` repeatedly with DISTINCT queries (as the denoise loop does — the query changes, the
    /// image tokens do not) and compare, at each step, the cached output against a from-scratch reference
    /// that reprojects the tokens with the same weights. Also asserts `set_ip` populates the cache and
    /// `set_ip(None)` clears it, so the cache is scoped to a single render.
    #[test]
    fn ip_kv_cache_is_bit_identical_across_steps() {
        let dev = Device::Cpu;
        let vb = VarBuilder::from_varmap(&VarMap::new(), DType::F32, &dev);
        let (query_dim, ctx_dim, heads, dim_head) = (16usize, 24usize, 4usize, 4usize);
        let inner = heads * dim_head; // 16
        let mut xa = CrossAttention::new(
            vb,
            query_dim,
            Some(ctx_dim),
            heads,
            dim_head,
            false,
            candle_gen::quant::MLX_GROUP_SIZE,
        )
        .unwrap();

        let ctx = Tensor::randn(0f32, 1f32, (2, 7, ctx_dim), &dev).unwrap();
        let k_ip = Tensor::randn(0f32, 1f32, (inner, ctx_dim), &dev).unwrap();
        let v_ip = Tensor::randn(0f32, 1f32, (inner, ctx_dim), &dev).unwrap();
        xa.install_ip(k_ip.clone(), v_ip.clone());
        let ip_tokens = Tensor::randn(0f32, 1f32, (2, 3, ctx_dim), &dev).unwrap();

        // Cache is empty before `set_ip`, populated after, cleared on `set_ip(None)`.
        assert!(xa.ip_kv.is_none(), "no cache before set_ip");
        xa.set_ip(Some(&ip_tokens), 0.7).unwrap();
        assert!(xa.ip_kv.is_some(), "set_ip must populate the K/V cache");

        // The cached head-shaped K/V must equal a fresh reshape of the raw projections.
        let flat = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let want_k = xa
            .reshape_heads_to_batch_dim(&nn::Linear::new(k_ip, None).forward(&ip_tokens).unwrap())
            .unwrap();
        let want_v = xa
            .reshape_heads_to_batch_dim(&nn::Linear::new(v_ip, None).forward(&ip_tokens).unwrap())
            .unwrap();
        let (cached_k, cached_v) = xa.ip_kv.as_ref().unwrap();
        assert_eq!(
            flat(cached_k),
            flat(&want_k),
            "cached K must match recompute"
        );
        assert_eq!(
            flat(cached_v),
            flat(&want_v),
            "cached V must match recompute"
        );

        // Across multiple steps (distinct queries), the cached branch equals a per-step recompute.
        for _step in 0..3 {
            let xs = Tensor::randn(0f32, 1f32, (2, 5, query_dim), &dev).unwrap();
            let got = xa.forward(&xs, Some(&ctx)).unwrap();
            // Reference: manually run the decoupled branch reprojecting the tokens THIS step.
            let query = xa
                .reshape_heads_to_batch_dim(&xa.to_q.forward(&xs).unwrap())
                .unwrap();
            let context = ctx.contiguous().unwrap();
            let key = xa
                .reshape_heads_to_batch_dim(&xa.to_k.forward(&context).unwrap())
                .unwrap();
            let value = xa
                .reshape_heads_to_batch_dim(&xa.to_v.forward(&context).unwrap())
                .unwrap();
            let base = xa.attention(&query, &key, &value).unwrap();
            let key_ip = xa
                .reshape_heads_to_batch_dim(
                    &xa.to_k_ip.as_ref().unwrap().forward(&ip_tokens).unwrap(),
                )
                .unwrap();
            let value_ip = xa
                .reshape_heads_to_batch_dim(
                    &xa.to_v_ip.as_ref().unwrap().forward(&ip_tokens).unwrap(),
                )
                .unwrap();
            let ip_out = xa.attention(&query, &key_ip, &value_ip).unwrap();
            let want = xa
                .to_out
                .forward(&(base + (ip_out * xa.ip_scale).unwrap()).unwrap())
                .unwrap();
            assert_eq!(
                flat(&got),
                flat(&want),
                "cached IP-K/V forward must be bit-identical to a per-step recompute"
            );
        }

        xa.set_ip(None, 0.0).unwrap();
        assert!(xa.ip_kv.is_none(), "set_ip(None) must clear the cache");
    }
}

#[cfg(test)]
mod packed_tests {
    //! sc-9416: the vendored SDXL UNet's Linear surface packed-detects through `candle_gen::quant`.
    //! These build a synthetic **packed** `SpatialTransformer` checkpoint (the MLX-tier key layout:
    //! attn `to_q/k/v/out.0`, GEGLU `ff.net.0.proj`, `ff.net.2`, linear `proj_in/proj_out` — each a
    //! `{weight u32, scales, biases}` triple) and assert every Linear loaded packed while the
    //! GroupNorm/LayerNorm stay dense, then forward to a coherent shape. A dense checkpoint (no
    //! `.scales`) is covered by the vendored-vs-stock parity test.
    use super::{SpatialTransformer, SpatialTransformerConfig};
    use candle_core::safetensors::MmapedSafetensors;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;
    use std::collections::HashMap;

    const GS: usize = 64;

    /// Pack an `[out, in]` weight as an MLX Q4 triple (LSB-first nibbles, per-group affine scales/biases).
    fn pack(map: &mut HashMap<String, Tensor>, base: &str, out_f: usize, in_f: usize, bias: bool) {
        let dev = Device::Cpu;
        let codes: Vec<u8> = (0..out_f * in_f)
            .map(|i| ((i * 5 + 3) % 16) as u8)
            .collect();
        let words: Vec<u32> = codes
            .chunks_exact(8)
            .map(|c| {
                c.iter()
                    .enumerate()
                    .fold(0u32, |acc, (i, &q)| acc | ((q as u32 & 0xF) << (4 * i)))
            })
            .collect();
        let groups = out_f * in_f / GS;
        let scales: Vec<f32> = (0..groups).map(|g| 0.03125 * (g as f32 + 1.0)).collect();
        let biases: Vec<f32> = (0..groups).map(|g| -0.25 - 0.1 * g as f32).collect();
        let gpr = in_f / GS;
        map.insert(
            format!("{base}.weight"),
            Tensor::from_vec(words, (out_f, in_f / 8), &dev).unwrap(),
        );
        map.insert(
            format!("{base}.scales"),
            Tensor::from_vec(scales, (out_f, gpr), &dev).unwrap(),
        );
        map.insert(
            format!("{base}.biases"),
            Tensor::from_vec(biases, (out_f, gpr), &dev).unwrap(),
        );
        if bias {
            map.insert(
                format!("{base}.bias"),
                Tensor::zeros((out_f,), DType::F32, &dev).unwrap(),
            );
        }
    }

    fn dense(map: &mut HashMap<String, Tensor>, key: &str, shape: &[usize]) {
        map.insert(
            key.to_string(),
            Tensor::ones(shape, DType::F32, &Device::Cpu).unwrap(),
        );
    }

    /// Build a fully-packed `SpatialTransformer` (linear projection, depth 1) and assert every Linear
    /// loaded packed (attn, FF, proj_in/out) while the norms are dense, then forward `[B,C,H,W]`.
    #[test]
    fn spatial_transformer_packed_detect_and_forward() {
        let dev = Device::Cpu;
        // in_channels == inner_dim so `use_linear_projection` shapes line up; group 64 divides 128.
        let (channels, n_heads, d_head) = (128usize, 4usize, 32usize); // inner = 128
        let ctx = 64usize;
        let mut map: HashMap<String, Tensor> = HashMap::new();

        // Linear proj_in / proj_out (use_linear_projection = true, as SDXL).
        pack(&mut map, "proj_in", channels, channels, true);
        pack(&mut map, "proj_out", channels, channels, true);
        // GroupNorm (dense — MLX packs no norms).
        dense(&mut map, "norm.weight", &[channels]);
        dense(&mut map, "norm.bias", &[channels]);

        let tb = "transformer_blocks.0";
        // attn1 (self): to_q/k/v [inner,inner], to_out.0 [inner,inner] (+bias).
        for p in ["attn1.to_q", "attn1.to_k", "attn1.to_v"] {
            pack(&mut map, &format!("{tb}.{p}"), channels, channels, false);
        }
        pack(
            &mut map,
            &format!("{tb}.attn1.to_out.0"),
            channels,
            channels,
            true,
        );
        // attn2 (cross): to_q [inner,inner]; to_k/to_v [inner, ctx]; to_out.0 [inner,inner].
        pack(
            &mut map,
            &format!("{tb}.attn2.to_q"),
            channels,
            channels,
            false,
        );
        pack(&mut map, &format!("{tb}.attn2.to_k"), channels, ctx, false);
        pack(&mut map, &format!("{tb}.attn2.to_v"), channels, ctx, false);
        pack(
            &mut map,
            &format!("{tb}.attn2.to_out.0"),
            channels,
            channels,
            true,
        );
        // FF: net.0.proj GEGLU [2*inner*4, inner]; net.2 [inner, inner*4].
        pack(
            &mut map,
            &format!("{tb}.ff.net.0.proj"),
            channels * 4 * 2,
            channels,
            true,
        );
        pack(
            &mut map,
            &format!("{tb}.ff.net.2"),
            channels,
            channels * 4,
            true,
        );
        // LayerNorms (dense).
        for n in ["norm1", "norm2", "norm3"] {
            dense(&mut map, &format!("{tb}.{n}.weight"), &[channels]);
            dense(&mut map, &format!("{tb}.{n}.bias"), &[channels]);
        }

        let tmp = std::env::temp_dir().join(format!(
            "sc9416_sdxl_st_packed_{}.safetensors",
            std::process::id()
        ));
        candle_core::safetensors::save(&map, &tmp).unwrap();
        // SAFETY: we just wrote this file and nothing else touches it during the test.
        let st = unsafe { MmapedSafetensors::new(&tmp).unwrap() };
        let vb = VarBuilder::from_backend(Box::new(st), DType::F32, dev.clone());

        let cfg = SpatialTransformerConfig {
            depth: 1,
            num_groups: 32,
            context_dim: Some(ctx),
            use_linear_projection: true,
        };
        let xf = SpatialTransformer::new_gs(vb, channels, n_heads, d_head, false, cfg, GS).unwrap();

        // Packed-detect fired on the whole Linear surface.
        assert!(
            xf.linear_projs_packed(),
            "proj_in/proj_out must load packed on a packed tier"
        );
        assert!(
            xf.all_block_attn_packed(),
            "every attn projection (to_q/k/v/out.0) must load packed"
        );

        // Forward `[B, C, H, W]` with cross context — a coherent, finite output the same shape as input.
        let x = Tensor::randn(0f32, 1f32, (1, channels, 8, 8), &dev).unwrap();
        let context = Tensor::randn(0f32, 1f32, (1, 5, ctx), &dev).unwrap();
        let y = xf.forward(&x, Some(&context)).unwrap();
        assert_eq!(y.dims(), &[1, channels, 8, 8]);
        let finite = y
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|v| v.is_finite());
        assert!(
            finite,
            "packed SpatialTransformer forward produced non-finite values"
        );
        std::fs::remove_file(&tmp).ok();
    }
}
