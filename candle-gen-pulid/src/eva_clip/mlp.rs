//! EVA `SwiGLU` FFN with sub-LN: `w3( ffn_ln( silu(w1·x) * (w2·x) ) )`. Candle port of
//! `eva_vit_model.py SwiGLU` (naiveswiglu + subln). All three linears are biased; `ffn_ln` is a
//! LayerNorm over the hidden dim (2730).

use candle_core::Tensor;
use candle_nn::{LayerNorm, Linear, Module};

use candle_gen::weights::Weights;
use candle_gen::Result as GenResult;

use crate::eva_clip::{join, layer_norm};

pub struct SwiGlu {
    w1: Linear,
    w2: Linear,
    ffn_ln: LayerNorm,
    w3: Linear,
}

impl SwiGlu {
    pub fn from_weights(w: &Weights, prefix: &str) -> GenResult<Self> {
        let lin = |leaf: &str| -> GenResult<Linear> {
            Ok(Linear::new(
                w.require(&join(prefix, &format!("{leaf}.weight")))?,
                Some(w.require(&join(prefix, &format!("{leaf}.bias")))?),
            ))
        };
        Ok(Self {
            w1: lin("w1")?,
            w2: lin("w2")?,
            ffn_ln: layer_norm(w, &join(prefix, "ffn_ln"))?,
            w3: lin("w3")?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x1 = self.w1.forward(x)?;
        let x2 = self.w2.forward(x)?;
        let hidden = (x1.silu()? * x2)?;
        let hidden = self.ffn_ln.forward(&hidden)?;
        self.w3.forward(&hidden)
    }
}
