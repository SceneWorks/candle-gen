//! The Kolors denoising UNet — the SDXL `UNet2DConditionModel` with the two Kolors deltas the
//! checkpoint carries:
//!
//!  1. an `encoder_hid_proj` Linear (4096 → 2048) projecting the ChatGLM3 context down to the
//!     cross-attention width, applied once up front (diffusers `encoder_hid_dim_type="text_proj"`);
//!  2. the `text_time` add-embedding's `linear_1` takes **5632** = pooled(4096) + 6·256 time-ids (vs
//!     stock SDXL's 2816 = pooled 1280 + 1536).
//!
//! The down/mid/up stack is the *identical* SDXL block layout, so this composes candle-transformers'
//! public `stable_diffusion::unet_2d_blocks` (the same blocks the stock `UNet2DConditionModel` uses)
//! and `embeddings::{Timesteps, TimestepEmbedding}` — only the top-level forward is rewritten to
//! project the context and build the SDXL `text_time` micro-conditioning the stock candle UNet omits.
//!
//! Construction mirrors candle's `UNet2DConditionModel::new` (the down/up index arithmetic is
//! copied verbatim from `stable_diffusion::unet_2d`); the model runs at **f32** (the candle port
//! recipe — candle matmul needs a single dtype; = mlx's "f32 activations over bf16 weights").

use candle_gen::candle_core::{Result, Tensor, D};
use candle_gen::candle_nn::{self as nn, Conv2dConfig, Module, VarBuilder};
use candle_transformers::models::stable_diffusion::embeddings::{TimestepEmbedding, Timesteps};
use candle_transformers::models::stable_diffusion::unet_2d_blocks::{
    CrossAttnDownBlock2D, CrossAttnDownBlock2DConfig, CrossAttnUpBlock2D, CrossAttnUpBlock2DConfig,
    DownBlock2D, DownBlock2DConfig, UNetMidBlock2DCrossAttn, UNetMidBlock2DCrossAttnConfig,
    UpBlock2D, UpBlock2DConfig,
};

/// One SDXL UNet stage: output channels, optional cross-attn (with its transformer-layer count), and
/// the per-head channel count. Matches candle's `unet_2d::BlockConfig`.
#[derive(Clone, Copy)]
struct BlockConfig {
    out_channels: usize,
    use_cross_attn: Option<usize>,
    attention_head_dim: usize,
}

/// The canonical SDXL UNet shape (`stabilityai/stable-diffusion-xl-base-1.0/unet/config.json`), shared
/// verbatim by Kolors. `cross_attention_dim = 2048` is the width the ChatGLM3 context is projected to.
struct UNetShape {
    blocks: [BlockConfig; 3],
    cross_attention_dim: usize,
    layers_per_block: usize,
    norm_num_groups: usize,
    norm_eps: f64,
    downsample_padding: usize,
    use_linear_projection: bool,
}

impl UNetShape {
    fn sdxl() -> Self {
        let bc = |out_channels, use_cross_attn, attention_head_dim| BlockConfig {
            out_channels,
            use_cross_attn,
            attention_head_dim,
        };
        Self {
            blocks: [
                bc(320, None, 5),
                bc(640, Some(2), 10),
                bc(1280, Some(10), 20),
            ],
            cross_attention_dim: 2048,
            layers_per_block: 2,
            norm_num_groups: 32,
            norm_eps: 1e-5,
            downsample_padding: 1,
            use_linear_projection: true,
        }
    }
}

enum DownBlock {
    Basic(DownBlock2D),
    CrossAttn(CrossAttnDownBlock2D),
}

enum UpBlock {
    Basic(UpBlock2D),
    CrossAttn(CrossAttnUpBlock2D),
}

/// The Kolors SDXL-family UNet (eps-prediction, 4-channel latent).
pub struct KolorsUNet {
    conv_in: nn::Conv2d,
    time_proj: Timesteps,
    time_embedding: TimestepEmbedding,
    /// SDXL `text_time` micro-conditioning: a parameterless 256-ch sinusoidal over the 6 time-ids …
    add_time_proj: Timesteps,
    /// … then the 5632→1280 MLP whose output is summed into the timestep embedding.
    add_embedding: TimestepEmbedding,
    /// Kolors-only: project the ChatGLM3 context (4096) to the cross-attention width (2048).
    encoder_hid_proj: nn::Linear,
    down_blocks: Vec<DownBlock>,
    mid_block: UNetMidBlock2DCrossAttn,
    up_blocks: Vec<UpBlock>,
    conv_norm_out: nn::GroupNorm,
    conv_out: nn::Conv2d,
    n_blocks: usize,
}

impl KolorsUNet {
    /// Build from the Kolors `unet/` VarBuilder (the diffusers SDXL key layout + the Kolors
    /// `encoder_hid_proj` / 5632 `add_embedding.linear_1`). `in_channels = out_channels = 4`.
    pub fn new(vb: VarBuilder, use_flash_attn: bool) -> Result<Self> {
        let shape = UNetShape::sdxl();
        let n_blocks = shape.blocks.len();
        let b_channels = shape.blocks[0].out_channels; // 320
        let bl_channels = shape.blocks[n_blocks - 1].out_channels; // 1280
        let bl_head_dim = shape.blocks[n_blocks - 1].attention_head_dim;
        let time_embed_dim = b_channels * 4; // 1280
        let conv_cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };

        let conv_in = nn::conv2d(4, b_channels, 3, conv_cfg, vb.pp("conv_in"))?;
        let time_proj = Timesteps::new(b_channels, true, 0.0);
        let time_embedding =
            TimestepEmbedding::new(vb.pp("time_embedding"), b_channels, time_embed_dim)?;

        // SDXL `text_time` added conditioning. `addition_time_embed_dim = 256`; `linear_1` in-features
        // = pooled(4096) + 6·256 = 5632 (the Kolors value — vs SDXL's 2816).
        let add_time_proj = Timesteps::new(256, true, 0.0);
        let add_embedding = TimestepEmbedding::new(vb.pp("add_embedding"), 5632, time_embed_dim)?;

        // Kolors context projection (ChatGLM3 4096 → cross_attention_dim 2048), with bias.
        let encoder_hid_proj =
            nn::linear(4096, shape.cross_attention_dim, vb.pp("encoder_hid_proj"))?;

        // Down blocks: block i goes blocks[i-1] -> blocks[i] channels.
        let vs_db = vb.pp("down_blocks");
        let mut down_blocks = Vec::with_capacity(n_blocks);
        for i in 0..n_blocks {
            let cfg = shape.blocks[i];
            let in_channels = if i > 0 {
                shape.blocks[i - 1].out_channels
            } else {
                b_channels
            };
            let db_cfg = DownBlock2DConfig {
                num_layers: shape.layers_per_block,
                resnet_eps: shape.norm_eps,
                resnet_groups: shape.norm_num_groups,
                add_downsample: i < n_blocks - 1,
                downsample_padding: shape.downsample_padding,
                ..Default::default()
            };
            if let Some(transformer_layers_per_block) = cfg.use_cross_attn {
                let ca = CrossAttnDownBlock2DConfig {
                    downblock: db_cfg,
                    attn_num_head_channels: cfg.attention_head_dim,
                    cross_attention_dim: shape.cross_attention_dim,
                    sliced_attention_size: None,
                    use_linear_projection: shape.use_linear_projection,
                    transformer_layers_per_block,
                };
                down_blocks.push(DownBlock::CrossAttn(CrossAttnDownBlock2D::new(
                    vs_db.pp(i.to_string()),
                    in_channels,
                    cfg.out_channels,
                    Some(time_embed_dim),
                    use_flash_attn,
                    ca,
                )?));
            } else {
                down_blocks.push(DownBlock::Basic(DownBlock2D::new(
                    vs_db.pp(i.to_string()),
                    in_channels,
                    cfg.out_channels,
                    Some(time_embed_dim),
                    db_cfg,
                )?));
            }
        }

        // Mid: the last block's cross-attn count (10 for SDXL).
        let mid_transformer_layers = shape.blocks[n_blocks - 1].use_cross_attn.unwrap_or(1);
        let mid_cfg = UNetMidBlock2DCrossAttnConfig {
            resnet_eps: shape.norm_eps,
            output_scale_factor: 1.0,
            cross_attn_dim: shape.cross_attention_dim,
            attn_num_head_channels: bl_head_dim,
            resnet_groups: Some(shape.norm_num_groups),
            use_linear_projection: shape.use_linear_projection,
            transformer_layers_per_block: mid_transformer_layers,
            ..Default::default()
        };
        let mid_block = UNetMidBlock2DCrossAttn::new(
            vb.pp("mid_block"),
            bl_channels,
            Some(time_embed_dim),
            use_flash_attn,
            mid_cfg,
        )?;

        // Up blocks: checkpoint up_blocks.{i} corresponds to config index `n-1-i` (reversed order).
        let vs_ub = vb.pp("up_blocks");
        let mut up_blocks = Vec::with_capacity(n_blocks);
        for i in 0..n_blocks {
            let cfg = shape.blocks[n_blocks - 1 - i];
            let prev_out_channels = if i > 0 {
                shape.blocks[n_blocks - i].out_channels
            } else {
                bl_channels
            };
            let in_channels = {
                let index = if i == n_blocks - 1 {
                    0
                } else {
                    n_blocks - i - 2
                };
                shape.blocks[index].out_channels
            };
            let ub_cfg = UpBlock2DConfig {
                num_layers: shape.layers_per_block + 1,
                resnet_eps: shape.norm_eps,
                resnet_groups: shape.norm_num_groups,
                add_upsample: i < n_blocks - 1,
                ..Default::default()
            };
            if let Some(transformer_layers_per_block) = cfg.use_cross_attn {
                let ca = CrossAttnUpBlock2DConfig {
                    upblock: ub_cfg,
                    attn_num_head_channels: cfg.attention_head_dim,
                    cross_attention_dim: shape.cross_attention_dim,
                    sliced_attention_size: None,
                    use_linear_projection: shape.use_linear_projection,
                    transformer_layers_per_block,
                };
                up_blocks.push(UpBlock::CrossAttn(CrossAttnUpBlock2D::new(
                    vs_ub.pp(i.to_string()),
                    in_channels,
                    prev_out_channels,
                    cfg.out_channels,
                    Some(time_embed_dim),
                    use_flash_attn,
                    ca,
                )?));
            } else {
                up_blocks.push(UpBlock::Basic(UpBlock2D::new(
                    vs_ub.pp(i.to_string()),
                    in_channels,
                    prev_out_channels,
                    cfg.out_channels,
                    Some(time_embed_dim),
                    ub_cfg,
                )?));
            }
        }

        let conv_norm_out = nn::group_norm(
            shape.norm_num_groups,
            b_channels,
            shape.norm_eps,
            vb.pp("conv_norm_out"),
        )?;
        let conv_out = nn::conv2d(b_channels, 4, 3, conv_cfg, vb.pp("conv_out"))?;

        Ok(Self {
            conv_in,
            time_proj,
            time_embedding,
            add_time_proj,
            add_embedding,
            encoder_hid_proj,
            down_blocks,
            mid_block,
            up_blocks,
            conv_norm_out,
            conv_out,
            n_blocks,
        })
    }

    /// Project the raw ChatGLM3 context `[B, S, 4096]` to the cross-attention width `[B, S, 2048]`
    /// via the Kolors `encoder_hid_proj`.
    ///
    /// This projection is **step-invariant** (it depends only on the prompt embeddings, not the
    /// timestep), so callers should hoist it out of the per-step denoise loop and feed the result to
    /// [`Self::forward_projected`] — matching the pose-control / IP-Adapter providers, which project
    /// once up front (`control.rs`, `ip_provider.rs`) before the vendored `forward_instantid`.
    pub fn project_context(&self, context: &Tensor) -> Result<Tensor> {
        self.encoder_hid_proj.forward(context)
    }

    /// Predict `eps` for one denoise step from an **already-projected** context.
    /// - `xs`: latents `[B, 4, H/8, W/8]`.
    /// - `timestep`: the (leading) float timestep, broadcast to the batch.
    /// - `encoder_hidden_states`: the 2048-wide cross-attention context, i.e. the output of
    ///   [`Self::project_context`], computed ONCE before the denoise loop (NOT the raw 4096-wide
    ///   ChatGLM3 context — the `encoder_hid_proj` projection is step-invariant and hoisted out).
    /// - `pooled`: the pooled ChatGLM3 add-embedding `[B, 4096]`; `time_ids`: micro-conditioning `[B, 6]`.
    ///
    /// Splitting the projection out of the forward avoids recomputing the step-invariant
    /// `encoder_hid_proj` every denoise step, and matches the pose-control / IP-Adapter providers
    /// (`control.rs`, `ip_provider.rs`), which likewise project up front (sc-9040 / F-056).
    pub fn forward_projected(
        &self,
        xs: &Tensor,
        timestep: f64,
        encoder_hidden_states: &Tensor,
        pooled: &Tensor,
        time_ids: &Tensor,
    ) -> Result<Tensor> {
        let (bsize, _channels, height, width) = xs.dims4()?;
        let device = xs.device();
        let num_upsamplers = self.n_blocks - 1;
        let default_overall_up_factor = 2usize.pow(num_upsamplers as u32);
        let forward_upsample_size =
            height % default_overall_up_factor != 0 || width % default_overall_up_factor != 0;

        // 1. time embedding.
        let emb = (Tensor::ones(bsize, xs.dtype(), device)? * timestep)?;
        let emb = self.time_proj.forward(&emb)?;
        let emb = self.time_embedding.forward(&emb)?;

        // SDXL `text_time` added conditioning: sinusoidal time_ids (flattened) ++ pooled → add_embedding.
        let (b, n_ids) = time_ids.dims2()?;
        let time_ids_emb = self.add_time_proj.forward(&time_ids.flatten_all()?)?; // [b·6, 256]
        let time_ids_emb = time_ids_emb.reshape((b, n_ids * 256))?; // [b, 1536]
        let add_in = Tensor::cat(&[pooled, &time_ids_emb], D::Minus1)?; // [b, 5632]
        let aug_emb = self.add_embedding.forward(&add_in)?;
        let emb = (emb + aug_emb)?;

        // 2. pre-process.
        let xs = self.conv_in.forward(xs)?;

        // 3. down.
        let mut down_block_res_xs = vec![xs.clone()];
        let mut xs = xs;
        for down_block in self.down_blocks.iter() {
            let (out, res_xs) = match down_block {
                DownBlock::Basic(b) => b.forward(&xs, Some(&emb))?,
                DownBlock::CrossAttn(b) => {
                    b.forward(&xs, Some(&emb), Some(encoder_hidden_states))?
                }
            };
            down_block_res_xs.extend(res_xs);
            xs = out;
        }

        // 4. mid.
        let mut xs = self
            .mid_block
            .forward(&xs, Some(&emb), Some(encoder_hidden_states))?;

        // 5. up.
        let mut upsample_size = None;
        for (i, up_block) in self.up_blocks.iter().enumerate() {
            let n_resnets = match up_block {
                UpBlock::Basic(b) => b.resnets.len(),
                UpBlock::CrossAttn(b) => b.upblock.resnets.len(),
            };
            let res_xs = down_block_res_xs.split_off(down_block_res_xs.len() - n_resnets);
            if i < self.n_blocks - 1 && forward_upsample_size {
                let (_, _, h, w) = down_block_res_xs.last().unwrap().dims4()?;
                upsample_size = Some((h, w));
            }
            xs = match up_block {
                UpBlock::Basic(b) => b.forward(&xs, &res_xs, Some(&emb), upsample_size)?,
                UpBlock::CrossAttn(b) => b.forward(
                    &xs,
                    &res_xs,
                    Some(&emb),
                    upsample_size,
                    Some(encoder_hidden_states),
                )?,
            };
        }

        // 6. post-process.
        let xs = self.conv_norm_out.forward(&xs)?;
        let xs = nn::ops::silu(&xs)?;
        self.conv_out.forward(&xs)
    }
}

#[cfg(test)]
mod tests {
    use candle_gen::candle_core::{DType, Device, Tensor};
    use candle_gen::candle_nn::{self as nn, Module, VarBuilder};

    /// The `encoder_hid_proj` projection (ChatGLM3 4096 → cross-attention 2048) is **step-invariant**:
    /// it depends only on the prompt context, not the timestep. This is the property that lets
    /// `Pipeline::render` hoist it out of the denoise loop (compute once via `project_context`, feed
    /// `forward_projected` each step) instead of re-projecting inside `forward` every step
    /// (sc-9040 / F-056). Assert that projecting once and reusing across N steps is BIT-IDENTICAL to
    /// re-projecting per step, so the hoist cannot change any pixel.
    #[test]
    fn encoder_hid_proj_is_step_invariant_bit_identical() -> candle_gen::candle_core::Result<()> {
        let dev = Device::Cpu;
        // A tiny stand-in for the 4096→2048 projection with fixed, deterministic weights + bias,
        // built exactly as `KolorsUNet::new` builds `encoder_hid_proj` (`nn::linear`, with bias).
        let (in_dim, out_dim) = (8usize, 4usize);
        let w =
            Tensor::arange(0f32, (out_dim * in_dim) as f32, &dev)?.reshape((out_dim, in_dim))?;
        let b = Tensor::arange(0f32, out_dim as f32, &dev)?;
        let mut ts = std::collections::HashMap::new();
        ts.insert("proj.weight".to_string(), w);
        ts.insert("proj.bias".to_string(), b);
        let vb = VarBuilder::from_tensors(ts, DType::F32, &dev);
        let proj = nn::linear(in_dim, out_dim, vb.pp("proj"))?;

        // A [B=2, S=3, 4096-analog] context, like the CFG-batched ChatGLM3 hidden states.
        let context = Tensor::randn(0f32, 1f32, (2, 3, in_dim), &dev)?;

        // Hoisted: project ONCE up front.
        let hoisted = proj.forward(&context)?;

        // Per-step: re-project every "step"; each result must be byte-identical to the hoisted one.
        for _step in 0..5 {
            let per_step = proj.forward(&context)?;
            let diff = (&per_step - &hoisted)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?;
            assert_eq!(
                diff, 0f32,
                "encoder_hid_proj must be step-invariant (bit-identical)"
            );
        }
        Ok(())
    }
}
