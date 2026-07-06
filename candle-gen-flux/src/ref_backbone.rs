//! A tier-detecting FLUX.1 backbone for the **reference** lanes (sc-10103, epic 9083) — the candle
//! twin of `mlx-gen-flux`'s `load_flux1`, which the MLX PuLID / IP-adapter providers delegate their
//! FLUX backbone to.
//!
//! The txt2img generator ([`crate::pipeline`]) already auto-detects the tier — a dense **BFL** snapshot
//! (`flux1-*.safetensors` + `ae.safetensors` at the root) vs a pre-quantized **diffusers-layout MLX
//! turnkey** (`SceneWorks/flux1-dev-mlx` q4/q8/bf16: `transformer/` + `text_encoder{,_2}/` + `vae/`) —
//! and builds the right CLIP / T5 / DiT / VAE for each (`Pipeline::load_components`). But that path was
//! wired only into `load_dev`/`load_schnell`; the reference providers (`candle-gen-pulid`, the FLUX
//! IP-adapter) built their backbone by hand from the single-file BFL layout and so could NOT read the
//! turnkey tiers.
//!
//! [`FluxRefBackbone`] closes that gap by **reusing the exact same** [`Pipeline::load_components`]
//! detect-and-load path the shipped txt2img generator uses, then exposing the three ops a reference lane
//! needs on top of it — [`encode_text`](FluxRefBackbone::encode_text),
//! [`forward_injected`](FluxRefBackbone::forward_injected) (the post-block [`DitImageInjector`] seam,
//! now on both the BFL [`IpFlux`](crate::ip_dit::IpFlux) and the diffusers
//! [`PackedFluxDit`](crate::packed_dit::PackedFluxDit)), and [`decode`](FluxRefBackbone::decode). So a
//! reference lane inherits whatever tier handling the base generator has (q4/q8/bf16), with one
//! packed-detect path and zero drift.
//!
//! PiD is **not** owned here: the PiD super-resolving decoder is a per-generation choice the reference
//! provider builds separately (PuLID's `with_pid`), so [`decode`](FluxRefBackbone::decode) takes the
//! decoder explicitly. The backbone loads with no PiD spec.

use std::path::Path;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::Image;
use candle_gen::Result;
use candle_gen_pid::PidDecoder;

use crate::ip_dit::DitImageInjector;
use crate::pipeline::{Components, Pipeline};
use crate::Variant;

/// A loaded, tier-detected FLUX.1 backbone (CLIP + T5 + DiT + VAE + tokenizers) for the reference
/// lanes. Holds the light [`Pipeline`] handle (variant/root/device/dtype) plus the loaded
/// [`Components`] — the SAME pair the txt2img generator caches — so every op delegates to the shared,
/// already-validated pipeline code.
pub struct FluxRefBackbone {
    pipeline: Pipeline,
    components: Components,
}

impl FluxRefBackbone {
    /// Load the FLUX backbone from a snapshot `root`, auto-detecting the tier: a dense BFL snapshot vs a
    /// pre-quantized diffusers-layout MLX turnkey tier subdir (`…/q4`, `…/q8`, `…/bf16`). `root` is the
    /// tier subdir the worker resolved (via `standard_tier_subdir`) for the packed turnkey, or the plain
    /// BFL snapshot root. `variant` is always [`Variant::Dev`] for PuLID; both variants are supported.
    /// No PiD spec is captured — the reference provider owns its PiD decoder separately (see
    /// [`decode`](Self::decode)).
    pub fn load(root: &Path, variant: Variant, device: &Device, dtype: DType) -> Result<Self> {
        let pipeline = Pipeline::load(variant, root, device, dtype, None);
        let components = pipeline.load_components()?;
        Ok(Self {
            pipeline,
            components,
        })
    }

    /// Encode `prompt` into FLUX's two conditioning tensors — the T5 sequence `(1, L, 4096)` and the
    /// CLIP pooled vector `(1, 768)` — at the compute dtype. Tier-agnostic: delegates to the shared
    /// `Pipeline::text_embeddings`, which runs the stock or packed text encoders as loaded.
    pub fn encode_text(&self, prompt: &str) -> Result<(Tensor, Tensor)> {
        self.pipeline.text_embeddings(&self.components, prompt)
    }

    /// The FLUX DiT velocity forward with the optional **post-block** image-stream residual injector —
    /// the PuLID id cross-attn seam. Dispatches to the loaded tier's DiT: the BFL
    /// [`IpFlux::forward_injected`](crate::ip_dit::IpFlux::forward_injected) (dense snapshot) or the
    /// diffusers [`PackedFluxDit::forward_injected`](crate::packed_dit::PackedFluxDit::forward_injected)
    /// (packed/dense turnkey tier). `injector = None` is the plain FLUX forward. `guidance` is the dev
    /// per-batch embedded guidance (`None` for schnell). The two DiTs take the same argument shapes and
    /// inject at the same layout-agnostic block indices, so the caller's [`DitImageInjector`] is
    /// unchanged across tiers.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_injected(
        &self,
        img: &Tensor,
        img_ids: &Tensor,
        txt: &Tensor,
        txt_ids: &Tensor,
        timesteps: &Tensor,
        y: &Tensor,
        guidance: Option<&Tensor>,
        injector: Option<&dyn DitImageInjector>,
    ) -> Result<Tensor> {
        let out = match &self.components {
            Components::Stock { transformer, .. } => transformer
                .forward_injected(img, img_ids, txt, txt_ids, timesteps, y, guidance, injector)?,
            Components::Packed { transformer, .. } => transformer
                .forward_injected(img, img_ids, txt, txt_ids, timesteps, y, guidance, injector)?,
        };
        Ok(out)
    }

    /// Decode the denoised latents `(1, h·w, 64)` to an RGB8 [`Image`], routing through the loaded tier's
    /// VAE (stock `AutoEncoder` / packed `AutoEncoderKL`) — or, when `pid` is `Some`, the caller's PiD
    /// super-resolving decoder (which consumes the same unpacked latent). `height`/`width` are the
    /// requested pixel dims.
    pub fn decode(
        &self,
        latents: &Tensor,
        height: usize,
        width: usize,
        pid: Option<&PidDecoder>,
    ) -> Result<Image> {
        self.pipeline
            .decode_ref(&self.components, latents, height, width, pid)
    }
}
