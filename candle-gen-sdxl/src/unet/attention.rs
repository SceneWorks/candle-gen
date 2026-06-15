//! Attention Based Building Blocks
use candle_core::{DType, IndexOp, Result, Tensor, D};
use candle_gen::train::lora::{lora_linear, lora_linear_no_bias, LoraLinear};
use candle_nn as nn;
use candle_nn::Module;

#[derive(Debug)]
struct GeGlu {
    proj: nn::Linear,
    span: tracing::Span,
}

impl GeGlu {
    fn new(vs: nn::VarBuilder, dim_in: usize, dim_out: usize) -> Result<Self> {
        let proj = nn::linear(dim_in, dim_out * 2, vs.pp("proj"))?;
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
    linear: nn::Linear,
    span: tracing::Span,
}

impl FeedForward {
    // The glu parameter in the python code is unused?
    // https://github.com/huggingface/diffusers/blob/d3d22ce5a894becb951eec03e663951b28d45135/src/diffusers/models/attention.py#L347
    /// Creates a new feed-forward layer based on some given input dimension, some
    /// output dimension, and a multiplier to be used for the intermediary layer.
    fn new(vs: nn::VarBuilder, dim: usize, dim_out: Option<usize>, mult: usize) -> Result<Self> {
        let inner_dim = dim * mult;
        let dim_out = dim_out.unwrap_or(dim);
        let vs = vs.pp("net");
        let project_in = GeGlu::new(vs.pp("0"), dim, inner_dim)?;
        let linear = nn::linear(inner_dim, dim_out, vs.pp("2"))?;
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
// is vendored from. Inference still uses the stock candle-transformers UNet (with real flash-attn).
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
    slice_size: Option<usize>,
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
}

impl CrossAttention {
    // Defaults should be heads = 8, dim_head = 64, context_dim = None
    pub fn new(
        vs: nn::VarBuilder,
        query_dim: usize,
        context_dim: Option<usize>,
        heads: usize,
        dim_head: usize,
        slice_size: Option<usize>,
        use_flash_attn: bool,
    ) -> Result<Self> {
        let inner_dim = dim_head * heads;
        let context_dim = context_dim.unwrap_or(query_dim);
        let scale = 1.0 / f64::sqrt(dim_head as f64);
        let to_q = lora_linear_no_bias(query_dim, inner_dim, vs.pp("to_q"))?;
        let to_k = lora_linear_no_bias(context_dim, inner_dim, vs.pp("to_k"))?;
        let to_v = lora_linear_no_bias(context_dim, inner_dim, vs.pp("to_v"))?;
        let to_out = lora_linear(inner_dim, query_dim, vs.pp("to_out.0"))?;
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
            slice_size,
            span,
            span_attn,
            span_softmax,
            use_flash_attn,
            to_k_ip: None,
            to_v_ip: None,
            ip_tokens: None,
            ip_scale: 0.0,
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

    fn sliced_attention(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        slice_size: usize,
    ) -> Result<Tensor> {
        let batch_size_attention = query.dim(0)?;
        let mut hidden_states = Vec::with_capacity(batch_size_attention / slice_size);
        let in_dtype = query.dtype();
        let query = query.to_dtype(DType::F32)?;
        let key = key.to_dtype(DType::F32)?;
        let value = value.to_dtype(DType::F32)?;

        for i in 0..batch_size_attention / slice_size {
            let start_idx = i * slice_size;
            let end_idx = (i + 1) * slice_size;

            let xs = query
                .i(start_idx..end_idx)?
                .matmul(&(key.i(start_idx..end_idx)?.t()? * self.scale)?)?;
            let xs = nn::ops::softmax(&xs, D::Minus1)?.matmul(&value.i(start_idx..end_idx)?)?;
            hidden_states.push(xs)
        }
        let hidden_states = Tensor::stack(&hidden_states, 0)?.to_dtype(in_dtype)?;
        self.reshape_batch_dim_to_heads(&hidden_states)
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
            let xs = query.matmul(&(key.t()? * self.scale)?)?;
            let xs = {
                let _enter = self.span_softmax.enter();
                // The composable `softmax` (not the fused `softmax_last_dim`): the fused kernel is a
                // `CustomOp` with no backward, so grads would never reach `to_q`/`to_k` through the
                // scores (sc-5165). Numerically identical, so the stock forward-parity test still holds.
                nn::ops::softmax(&xs, D::Minus1)?
            };
            xs.matmul(&value)?.to_dtype(in_dtype)?
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
        let dim0 = query.dim(0)?;
        let slice_size = self.slice_size.and_then(|slice_size| {
            if dim0 < slice_size {
                None
            } else {
                Some(slice_size)
            }
        });
        let mut xs = match slice_size {
            None => self.attention(&query, &key, &value)?,
            Some(slice_size) => self.sliced_attention(&query, &key, &value, slice_size)?,
        };
        // IP-Adapter decoupled branch (sc-5491): reuse the text query against the image/identity
        // tokens' own K/V, scaled and added before the output projection. A no-op (skipped) unless the
        // K/V are installed AND tokens are set — so the training / stock path is unchanged.
        if let (Some(to_k_ip), Some(to_v_ip), Some(ip_tokens)) =
            (&self.to_k_ip, &self.to_v_ip, &self.ip_tokens)
        {
            let key_ip = self.reshape_heads_to_batch_dim(&to_k_ip.forward(ip_tokens)?)?;
            let value_ip = self.reshape_heads_to_batch_dim(&to_v_ip.forward(ip_tokens)?)?;
            let ip_out = self.attention(&query, &key_ip, &value_ip)?;
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
    }

    /// Set (or clear, with `None`) the IP tokens + scale used by [`forward`](Self::forward)'s decoupled
    /// branch. Constant across the denoise, so set once per generation; the clone is cheap (the face
    /// tokens are `[B, 16, 2048]`).
    pub fn set_ip(&mut self, tokens: Option<&Tensor>, scale: f64) {
        self.ip_tokens = tokens.cloned();
        self.ip_scale = scale;
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
    fn new(
        vs: nn::VarBuilder,
        dim: usize,
        n_heads: usize,
        d_head: usize,
        context_dim: Option<usize>,
        sliced_attention_size: Option<usize>,
        use_flash_attn: bool,
    ) -> Result<Self> {
        let attn1 = CrossAttention::new(
            vs.pp("attn1"),
            dim,
            None,
            n_heads,
            d_head,
            sliced_attention_size,
            use_flash_attn,
        )?;
        let ff = FeedForward::new(vs.pp("ff"), dim, None, 4)?;
        let attn2 = CrossAttention::new(
            vs.pp("attn2"),
            dim,
            context_dim,
            n_heads,
            d_head,
            sliced_attention_size,
            use_flash_attn,
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
    pub sliced_attention_size: Option<usize>,
    pub use_linear_projection: bool,
}

impl Default for SpatialTransformerConfig {
    fn default() -> Self {
        Self {
            depth: 1,
            num_groups: 32,
            context_dim: None,
            sliced_attention_size: None,
            use_linear_projection: false,
        }
    }
}

#[derive(Debug)]
enum Proj {
    Conv2d(nn::Conv2d),
    Linear(nn::Linear),
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
        let inner_dim = n_heads * d_head;
        let norm = nn::group_norm(config.num_groups, in_channels, 1e-6, vs.pp("norm"))?;
        let proj_in = if config.use_linear_projection {
            Proj::Linear(nn::linear(in_channels, inner_dim, vs.pp("proj_in"))?)
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
                config.sliced_attention_size,
                use_flash_attn,
            )?;
            transformer_blocks.push(tb)
        }
        let proj_out = if config.use_linear_projection {
            Proj::Linear(nn::linear(in_channels, inner_dim, vs.pp("proj_out"))?)
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
        let mut xa =
            CrossAttention::new(vb, query_dim, Some(ctx_dim), heads, dim_head, None, false)
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
        xa.set_ip(Some(&ip_tokens), 0.0);
        assert!(
            maxdiff(&xa.forward(&xs, Some(&ctx)).unwrap(), &base) < 1e-6,
            "ip_scale=0 must equal the no-IP output"
        );

        // A positive scale shifts the output.
        xa.set_ip(Some(&ip_tokens), 0.8);
        assert!(
            maxdiff(&xa.forward(&xs, Some(&ctx)).unwrap(), &base) > 1e-4,
            "ip_scale>0 must change the output"
        );

        // Clearing the tokens reverts to the plain cross-attention.
        xa.set_ip(None, 0.0);
        assert!(
            maxdiff(&xa.forward(&xs, Some(&ctx)).unwrap(), &base) < 1e-6,
            "clearing the IP tokens reverts to base"
        );
    }
}
