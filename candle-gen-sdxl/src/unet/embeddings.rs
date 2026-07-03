use candle_core::{Result, Tensor, D};
use candle_gen::quant::QLinear;
use candle_nn as nn;
use candle_nn::Module;

#[derive(Debug)]
pub struct TimestepEmbedding {
    // sc-9416: `linear_1`/`linear_2` packed-detect through the shared `candle_gen::quant` seam — the
    // MLX SDXL tiers pack `time_embedding.linear_{1,2}` + `add_embedding.linear_{1,2}`. Dense
    // checkpoints have no `.scales` sibling, so `linear_detect` takes the plain dense path unchanged.
    linear_1: QLinear,
    linear_2: QLinear,
}

impl TimestepEmbedding {
    // act_fn: "silu"
    pub fn new(vs: nn::VarBuilder, channel: usize, time_embed_dim: usize) -> Result<Self> {
        Self::new_gs(
            vs,
            channel,
            time_embed_dim,
            candle_gen::quant::MLX_GROUP_SIZE,
        )
    }

    /// As [`new`](Self::new), but at an explicit MLX packed `group_size` (sc-9416) — threaded from the
    /// component `config.json`'s `quantization.group_size` on the packed txt2img UNet load.
    pub fn new_gs(
        vs: nn::VarBuilder,
        channel: usize,
        time_embed_dim: usize,
        group_size: usize,
    ) -> Result<Self> {
        let linear_1 =
            QLinear::linear_detect_gs(channel, time_embed_dim, &vs, "linear_1", true, group_size)?;
        let linear_2 = QLinear::linear_detect_gs(
            time_embed_dim,
            time_embed_dim,
            &vs,
            "linear_2",
            true,
            group_size,
        )?;
        Ok(Self { linear_1, linear_2 })
    }
}

impl Module for TimestepEmbedding {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = nn::ops::silu(&self.linear_1.forward(xs)?)?;
        self.linear_2.forward(&xs)
    }
}

#[derive(Debug)]
pub struct Timesteps {
    num_channels: usize,
    flip_sin_to_cos: bool,
    downscale_freq_shift: f64,
}

impl Timesteps {
    pub fn new(num_channels: usize, flip_sin_to_cos: bool, downscale_freq_shift: f64) -> Self {
        Self {
            num_channels,
            flip_sin_to_cos,
            downscale_freq_shift,
        }
    }
}

impl Module for Timesteps {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let half_dim = (self.num_channels / 2) as u32;
        let exponent = (Tensor::arange(0, half_dim, xs.device())?
            .to_dtype(candle_core::DType::F32)?
            * -f64::ln(10000.))?;
        let exponent = (exponent / (half_dim as f64 - self.downscale_freq_shift))?;
        let emb = exponent.exp()?.to_dtype(xs.dtype())?;
        // emb = timesteps[:, None].float() * emb[None, :]
        let emb = xs.unsqueeze(D::Minus1)?.broadcast_mul(&emb.unsqueeze(0)?)?;
        let (cos, sin) = (emb.cos()?, emb.sin()?);
        let emb = if self.flip_sin_to_cos {
            Tensor::cat(&[&cos, &sin], D::Minus1)?
        } else {
            Tensor::cat(&[&sin, &cos], D::Minus1)?
        };
        if self.num_channels % 2 == 1 {
            emb.pad_with_zeros(D::Minus2, 0, 1)
        } else {
            Ok(emb)
        }
    }
}
