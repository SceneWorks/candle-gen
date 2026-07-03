//! EVA `Block`: pre-norm residual `x += attn(norm1(x)); x += mlp(norm2(x))` (no LayerScale —
//! EVA02-CLIP-L has `init_values=None`, postnorm=False). `norm1`/`norm2` are LayerNorm(weight+bias,
//! ε=1e-6). Candle port of `eva_vit_model.py Block`.

use candle_core::Tensor;
use candle_nn::{LayerNorm, Module};

use candle_gen::weights::Weights;
use candle_gen::Result as GenResult;

use crate::eva_clip::attention::Attention;
use crate::eva_clip::mlp::SwiGlu;
use crate::eva_clip::rope::VisionRope;
use crate::eva_clip::{join, layer_norm};

pub struct Block {
    norm1: LayerNorm,
    norm2: LayerNorm,
    attn: Attention,
    mlp: SwiGlu,
}

impl Block {
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        num_heads: usize,
        head_dim: usize,
    ) -> GenResult<Self> {
        Ok(Self {
            norm1: layer_norm(w, &join(prefix, "norm1"))?,
            norm2: layer_norm(w, &join(prefix, "norm2"))?,
            attn: Attention::from_weights(w, &join(prefix, "attn"), num_heads, head_dim)?,
            mlp: SwiGlu::from_weights(w, &join(prefix, "mlp"))?,
        })
    }

    pub fn forward(&self, x: &Tensor, rope: &VisionRope) -> candle_core::Result<Tensor> {
        let n1 = self.norm1.forward(x)?;
        let x = (x + self.attn.forward(&n1, rope)?)?;
        let n2 = self.norm2.forward(&x)?;
        &x + self.mlp.forward(&n2)?
    }
}
