//! Shared FLUX.1 component-loading stack (sc-9003 / F-023).
//!
//! The three FLUX.1 providers — the stock txt2img [`crate::pipeline`], the XLabs IP-Adapter
//! [`crate::ip_provider`], and the Fun-Controlnet-Union [`crate::control_provider`] — each loaded the
//! **same** backbone: CLIP-L from `text_encoder/`, T5-XXL from `text_encoder_2/` (config parse + sharded
//! mmap), the DiT from the root BFL `flux1-*.safetensors`, and the AutoEncoder VAE from `ae.safetensors`.
//! Plus the identical CPU-seeded initial-noise block (sc-3673 determinism). That was copy-pasted three
//! times, with only the provider-specific error label differing — so the parity-critical constants (the
//! T5 pad/attention convention, the noise geometry, the `text_model.` CLIP prefix) could drift
//! independently across the copies.
//!
//! This module is the single home. Each loader takes a `label` (`"flux"`, `"flux ip-adapter"`,
//! `"flux1 control"`) so callers keep their crafted, provider-specific diagnostics, and reuses the
//! workspace [`candle_gen::loader`] mmap surface (F-019) for the shard listing / mmap. The genuine
//! per-provider drift — which DiT wrapper (stock [`Flux`] vs the forked [`crate::ip_dit::IpFlux`]) is
//! built, whether the VAE is also mirrored into a mean-encoder — stays with the caller: these helpers
//! hand back the loaded CLIP / T5 / a DiT `VarBuilder` / the VAE and let the provider assemble its own
//! model shape.

use std::path::{Path, PathBuf};

use crate::vae::native::AutoEncoder;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::clip::text_model::ClipTextTransformer;
use candle_transformers::models::t5::{Config as T5Config, T5EncoderModel};
use rand::{rngs::StdRng, SeedableRng};

use candle_gen::{CandleError, Result};

use crate::pipeline::{ae_config, clip_config};
use crate::Variant;

/// mmap a [`VarBuilder`] over `files` at `dtype`/`device`, erroring (with the `label` prefix) if any is
/// missing. The shared body behind the three providers' former `mmap_vb` copies — the missing-file check
/// keeps the crafted "snapshot is missing X" diagnostic rather than surfacing candle's raw mmap error.
pub(crate) fn mmap_vb(
    files: &[PathBuf],
    dtype: DType,
    device: &Device,
    label: &str,
) -> Result<VarBuilder<'static>> {
    for f in files {
        if !f.is_file() {
            return Err(CandleError::Msg(format!(
                "{label} snapshot is missing {}",
                f.display()
            )));
        }
    }
    candle_gen::mmap_var_builder(files, dtype, device)
}

/// Sorted list of every `.safetensors` in the snapshot subdir `dir` (sharded T5 ships as
/// `model-0000n-of-0000m.safetensors`), `label`-prefixed on error. Thin wrapper over the shared
/// [`candle_gen::sorted_safetensors`] (F-019).
pub(crate) fn safetensors_in(dir: &Path, label: &str) -> Result<Vec<PathBuf>> {
    candle_gen::sorted_safetensors(dir, label)
}

/// Load the dense **BFL**-layout FLUX text encoders — CLIP-L (`text_encoder/model.safetensors`, pooled
/// under the `text_model.` prefix) and T5-XXL (`text_encoder_2/`, sharded, with its `config.json`
/// alongside) — at `dtype`/`device`. The single home for the block the three providers copy-pasted; the
/// `text_model.` prefix, the fixed [`clip_config`], and the T5 config parse are parity-critical and now
/// live once. `label` prefixes the crafted config-read/parse errors.
pub(crate) fn text_encoders(
    root: &Path,
    dtype: DType,
    device: &Device,
    label: &str,
) -> Result<(ClipTextTransformer, T5EncoderModel)> {
    // CLIP-L (openai/clip-vit-large-patch14 layout) under `text_encoder/`; the candle transformer pools
    // under the `text_model.` prefix. Config is fixed for FLUX.
    let clip_vb = mmap_vb(
        &[root.join("text_encoder/model.safetensors")],
        dtype,
        device,
        label,
    )?;
    let clip = ClipTextTransformer::new(clip_vb.pp("text_model"), &clip_config())?;

    // T5-XXL under `text_encoder_2/` (sharded; config.json alongside).
    let t5_dir = root.join("text_encoder_2");
    let t5_cfg: T5Config = {
        let cfg = std::fs::read_to_string(t5_dir.join("config.json")).map_err(|e| {
            CandleError::Msg(format!("{label}: read text_encoder_2/config.json: {e}"))
        })?;
        serde_json::from_str(&cfg)
            .map_err(|e| CandleError::Msg(format!("{label}: parse T5 config.json: {e}")))?
    };
    let t5_vb = mmap_vb(&safetensors_in(&t5_dir, label)?, dtype, device, label)?;
    let t5 = T5EncoderModel::load(t5_vb, &t5_cfg)?;

    Ok((clip, t5))
}

/// mmap a [`VarBuilder`] over the root BFL DiT checkpoint (`flux1-{schnell,dev}.safetensors`) for
/// `variant` at `dtype`/`device`. The caller builds the stock [`candle_transformers::models::flux::model::Flux`]
/// or the forked [`crate::ip_dit::IpFlux`] over it — that choice is the genuine per-provider drift, so it
/// stays with the caller; this just resolves + mmaps the (single) checkpoint file.
pub(crate) fn dit_vb(
    root: &Path,
    variant: Variant,
    dtype: DType,
    device: &Device,
    label: &str,
) -> Result<VarBuilder<'static>> {
    mmap_vb(
        &[root.join(variant.transformer_file())],
        dtype,
        device,
        label,
    )
}

/// Load the FLUX **AutoEncoder** VAE (`ae.safetensors` at the snapshot root) for `variant` at
/// `dtype`/`device`. Returns the VAE plus its backing [`VarBuilder`] so a caller that needs a second view
/// of the same weights (the control provider's deterministic mean-encoder, sc-8988) can reuse the mmap
/// rather than opening the file twice.
pub(crate) fn vae(
    root: &Path,
    variant: Variant,
    dtype: DType,
    device: &Device,
    label: &str,
) -> Result<(AutoEncoder, VarBuilder<'static>)> {
    let vae_vb = mmap_vb(&[root.join("ae.safetensors")], dtype, device, label)?;
    let vae = AutoEncoder::new(&ae_config(variant), vae_vb.clone())?;
    Ok((vae, vae_vb))
}

/// The FLUX deterministic, launch-portable initial latent noise (sc-3673): `N(0,1)` in candle's
/// `get_noise` shape `(1, channels, lat_h, lat_w)`, drawn from a fixed-algorithm CPU `StdRng` seeded by
/// `seed`, built on CPU then moved to `device` at `dtype`. The flow-match Euler step injects no per-step
/// noise, so generation is a pure function of `(seed, request)` — the seed-determinism contract. One home
/// for the block the three providers copy-pasted verbatim.
pub(crate) fn seeded_noise(
    seed: u64,
    channels: usize,
    lat_h: usize,
    lat_w: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let n = channels * lat_h * lat_w;
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = candle_gen::seeded_normal_vec(&mut rng, n);
    Tensor::from_vec(noise, (1, channels, lat_h, lat_w), &Device::Cpu)?
        .to_device(device)?
        .to_dtype(dtype)
        .map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `mmap_vb` names the missing file with the caller's `label` prefix (the crafted diagnostic each
    /// provider kept), not candle's raw mmap error.
    #[test]
    fn mmap_vb_missing_file_is_label_prefixed() {
        let dir = std::env::temp_dir().join(format!("flux1_load_mmap_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let missing = dir.join("nope.safetensors");
        // `VarBuilder` (the Ok arm) is not `Debug`, so match rather than `unwrap_err`.
        match mmap_vb(
            std::slice::from_ref(&missing),
            DType::F32,
            &Device::Cpu,
            "flux ip-adapter",
        ) {
            Err(CandleError::Msg(m)) => {
                assert!(
                    m.contains("flux ip-adapter") && m.contains("is missing"),
                    "got: {m}"
                );
            }
            _ => panic!("expected a crafted missing-file error"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `seeded_noise` is a pure function of the seed (launch-portable determinism, sc-3673): same seed ⇒
    /// bit-identical noise; a different seed ⇒ different noise; the shape is candle's `get_noise` NCHW.
    #[test]
    fn seeded_noise_is_deterministic_shaped_and_seed_sensitive() {
        let dev = Device::Cpu;
        let a = seeded_noise(42, 16, 4, 6, &dev, DType::F32).unwrap();
        let b = seeded_noise(42, 16, 4, 6, &dev, DType::F32).unwrap();
        let c = seeded_noise(43, 16, 4, 6, &dev, DType::F32).unwrap();
        assert_eq!(a.dims(), &[1, 16, 4, 6]);
        let (av, bv, cv) = (
            a.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            b.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            c.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        );
        assert_eq!(av, bv, "same seed ⇒ identical noise");
        assert_ne!(av, cv, "different seed ⇒ different noise");
        assert!(av.iter().all(|v| v.is_finite()));
    }

    /// `seeded_noise` honors the requested dtype (the providers draw at bf16).
    #[test]
    fn seeded_noise_honors_dtype() {
        let dev = Device::Cpu;
        let t = seeded_noise(7, 16, 2, 2, &dev, DType::BF16).unwrap();
        assert_eq!(t.dtype(), DType::BF16);
    }
}
