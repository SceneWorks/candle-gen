//! SDXL component loaders for the InstantID provider (sc-5491, epic 5480) — the candle twins of
//! `mlx-gen-sdxl`'s `load_unet_dtype` / `load_vae` / `load_controlnet`. The txt2img [`crate::pipeline`]
//! loads the **stock** candle-transformers UNet internally; InstantID needs the **vendored** UNet (the
//! one carrying the `add_embedding` micro-conditioning + the decoupled IP-Adapter cross-attention from
//! phase 2c), so these build that stack from an SDXL snapshot + a diffusers ControlNet checkpoint.
//!
//! The IP-Adapter K/V install is NOT done here — the caller (the `candle-gen-instantid` glue) loads the
//! converted `ip-adapter.safetensors` (`image_proj.*` Resampler + `ip_adapter.*` pairs) and calls
//! [`UNet2DConditionModel::install_ip_adapter`] on the returned UNet, mirroring `InstantId::load`.

use std::path::Path;

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use candle_transformers::models::stable_diffusion::vae::AutoEncoderKL;
use candle_transformers::models::stable_diffusion::StableDiffusionConfig;

use candle_gen::gen_core::WeightsSource;
use candle_gen::{CandleError, Result};

use crate::pipeline::{hf_get, snapshot_file, VAE_FIX_FILE, VAE_FIX_REPO};
use crate::unet::{sdxl_unet_config, ControlNet, ControlNetConfig, UNet2DConditionModel};

/// SDXL `add_embedding` dims (diffusers `unet/config.json`): `addition_time_embed_dim = 256`,
/// `projection_class_embeddings_input_dim = 2816` (pooled 1280 + 6·256). The InstantID UNet needs the
/// `add_embedding` head the plain `forward` omits.
const ADDITION_TIME_EMBED_DIM: usize = 256;
const PROJECTION_INPUT_DIM: usize = 2816;

/// Load the **vendored** SDXL UNet from `root/unet/diffusion_pytorch_model.fp16.safetensors` with the
/// `add_embedding` head loaded (so [`UNet2DConditionModel::forward_instantid`] runs). Math attention
/// (`use_flash_attn = false`) — the vendored flash path is a stub; perf tuning is later. The caller
/// installs the IP-Adapter K/V pairs.
pub fn load_instantid_unet(
    root: &Path,
    device: &Device,
    dtype: DType,
) -> Result<UNet2DConditionModel> {
    let unet_file = snapshot_file(root, "unet/diffusion_pytorch_model.fp16.safetensors")?;
    // One mmap'd VarBuilder feeds both the UNet body and the `add_embedding` head (both live in the
    // same `unet/` checkpoint). `VarBuilder` is Arc-backed, so the clone is cheap.
    let vs = unsafe { VarBuilder::from_mmaped_safetensors(&[unet_file], dtype, device)? };
    let unet = UNet2DConditionModel::new(vs.clone(), 4, 4, false, sdxl_unet_config())?
        .with_add_embedding(vs, ADDITION_TIME_EMBED_DIM, PROJECTION_INPUT_DIM)?;
    Ok(unet)
}

/// Load the f16-stable SDXL VAE (`madebyollin/sdxl-vae-fp16-fix`, resolved via `hf-hub` exactly as the
/// txt2img path) at `dtype`. Resolution-agnostic — `build_vae` reads only the autoencoder sub-config.
pub fn load_sdxl_vae(device: &Device, dtype: DType) -> Result<AutoEncoderKL> {
    let config = StableDiffusionConfig::sdxl(None, None, None);
    Ok(config.build_vae(hf_get(VAE_FIX_REPO, VAE_FIX_FILE)?, device, dtype)?)
}

/// Load a stock diffusers SDXL `ControlNetModel` (the InstantID IdentityNet, or the OpenPose CN for
/// pose mode) from a `WeightsSource`. A `Dir` resolves `diffusion_pytorch_model.safetensors` (then the
/// `.fp16` variant); a `File` is used directly. No conversion — the diffusers key layout is what
/// [`ControlNet::new`] reads.
pub fn load_sdxl_controlnet(
    source: &WeightsSource,
    device: &Device,
    dtype: DType,
) -> Result<ControlNet> {
    let file = match source {
        WeightsSource::File(f) => f.clone(),
        WeightsSource::Dir(d) => {
            let primary = d.join("diffusion_pytorch_model.safetensors");
            if primary.is_file() {
                primary
            } else {
                let fp16 = d.join("diffusion_pytorch_model.fp16.safetensors");
                if fp16.is_file() {
                    fp16
                } else {
                    return Err(CandleError::Msg(format!(
                        "sdxl controlnet: no diffusion_pytorch_model(.fp16).safetensors in {}",
                        d.display()
                    )));
                }
            }
        }
    };
    let vs = unsafe { VarBuilder::from_mmaped_safetensors(&[file], dtype, device)? };
    ControlNet::new(vs, &ControlNetConfig::sdxl())
}
