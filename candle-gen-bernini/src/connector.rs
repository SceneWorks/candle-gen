//! The Bernini planner's `MLPConnector` (candle sibling of `mlx-gen-bernini/src/connector.rs`,
//! sc-5139). Two projection heads off the planner's penultimate hidden state (3584):
//!   - **`for_gen`** (generation branch → renderer prompt-embed width 4096):
//!     `Linear(3584→4096) → GELU → RMSNorm(4096) → Linear(4096→4096)`.
//!   - **`for_vit`** (ViT branch → clip-diff condition width 3584):
//!     `Linear(3584→3584) → GELU → Linear(3584→3584) → RMSNorm(3584) → Linear(3584→3584)`.
//!
//! GELU is exact (erf); RMSNorm is weight-only, f32, eps 1e-6. Both branch linears carry bias.

use candle_gen::candle_core::Tensor;
use candle_gen::candle_nn::{Linear, Module, VarBuilder};
use candle_gen::Result as CResult;

use crate::nn::{lin_bias, rms_norm};

const RMS_EPS: f64 = 1e-6;

/// The Bernini `MLPConnector` (gen + vit branches).
pub struct MlpConnector {
    gen0: Linear,
    gen_rms: Tensor,
    gen3: Linear,
    vit0: Linear,
    vit2: Linear,
    vit_rms: Tensor,
    vit4: Linear,
}

impl MlpConnector {
    /// Build from a `VarBuilder` rooted at the connector namespace (`proj_gen.*` / `pred_vit.*`).
    pub fn new(vb: VarBuilder) -> CResult<Self> {
        Ok(Self {
            gen0: lin_bias(&vb, "proj_gen.0")?,
            gen_rms: vb.get_unchecked("proj_gen.2.weight")?,
            gen3: lin_bias(&vb, "proj_gen.3")?,
            vit0: lin_bias(&vb, "pred_vit.0")?,
            vit2: lin_bias(&vb, "pred_vit.2")?,
            vit_rms: vb.get_unchecked("pred_vit.3.weight")?,
            vit4: lin_bias(&vb, "pred_vit.4")?,
        })
    }

    /// Generation branch → `[*, 4096]` renderer-prompt embeds.
    pub fn for_gen(&self, x: &Tensor) -> CResult<Tensor> {
        let x = self.gen0.forward(x)?.gelu_erf()?;
        let x = rms_norm(&x, &self.gen_rms, RMS_EPS)?;
        Ok(self.gen3.forward(&x)?)
    }

    /// ViT branch → `[*, 3584]` clip-diff condition.
    pub fn for_vit(&self, x: &Tensor) -> CResult<Tensor> {
        let x = self.vit0.forward(x)?.gelu_erf()?;
        let x = self.vit2.forward(&x)?;
        let x = rms_norm(&x, &self.vit_rms, RMS_EPS)?;
        Ok(self.vit4.forward(&x)?)
    }
}
