//! SeedVR2 image-mode pipeline — candle port of `mlx-gen-seedvr2/src/pipeline.rs` (sc-5157).
//!
//! One-step super-resolution: preprocess the LR image (bicubic upscale to target, optional `softness`
//! pre-blur, [-1,1]) → VAE encode → conditioning latent (encoded latent + ones-mask) → concat fresh
//! noise → DiT (one step) → `latents = noise − DiT_out` → VAE decode → crop → LAB+wavelet color
//! correction → RGB8. Video mode (the 5-D temporal pass) is sc-5926.
//!
//! The negative-prompt conditioning is a precomputed embedding (bundled `data/neg_embed.safetensors`,
//! no runtime text encoder), loaded at construction.

use candle_gen::candle_core::{DType, Device, Result, Tensor};
use candle_gen::gen_core::{imageops, Image};
use candle_gen::{CandleError, Result as CResult};
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};

use crate::config::{DitConfig, TIMESTEP};
use crate::dit::Seedvr2Transformer;
use crate::vae::Seedvr2Vae;
use crate::weights::Weights;
use crate::{color, convert};

/// Post-decode color-correction luminance weight (the reference `apply_color_correction` default).
const LUMINANCE_WEIGHT: f32 = 0.8;

pub struct Seedvr2Pipeline {
    pub vae: Seedvr2Vae,
    pub transformer: Seedvr2Transformer,
    neg_embed: Tensor,
    dtype: DType,
    device: Device,
}

/// The bundled precomputed negative-prompt embedding → `(1, Lt, 5120)` at `dt`.
fn load_neg_embed(dt: DType, dev: &Device) -> CResult<Tensor> {
    const BYTES: &[u8] = include_bytes!("../data/neg_embed.safetensors");
    let map = candle_gen::candle_core::safetensors::load_buffer(BYTES, dev)?;
    let emb = map
        .get("embedding")
        .ok_or_else(|| CandleError::Msg("seedvr2 neg-embed: missing `embedding`".into()))?;
    Ok(emb.to_dtype(dt)?.unsqueeze(0)?)
}

/// Deterministic N(0,1) noise of a 5-D shape (CPU `StdRng`/ChaCha, launch-portable per seed).
fn seeded_normal5(
    seed: u64,
    shape: (usize, usize, usize, usize, usize),
    dt: DType,
    dev: &Device,
) -> CResult<Tensor> {
    let (a, b, c, d, e) = shape;
    let mut rng = StdRng::seed_from_u64(seed);
    let data: Vec<f32> = (0..a * b * c * d * e)
        .map(|_| StandardNormal.sample(&mut rng))
        .collect();
    Ok(Tensor::from_vec(data, shape, dev)?.to_dtype(dt)?)
}

/// `(1,3,H,W)` in [-1,1] → RGB8 [`Image`].
fn decoded_to_image(decoded: &Tensor) -> Result<Image> {
    let (_b, _c, h, w) = decoded.dims4()?;
    let u8s = ((decoded.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?
        .to_dtype(DType::U8)?
        .to_device(&Device::Cpu)?;
    let chw = u8s.squeeze(0)?; // (3,H,W)
    let pixels = chw.permute((1, 2, 0))?.flatten_all()?.to_vec1::<u8>()?;
    Ok(Image {
        width: w as u32,
        height: h as u32,
        pixels,
    })
}

impl Seedvr2Pipeline {
    /// Build from already-converted candle-layout VAE + DiT weights + a neg-embed (parity tests).
    pub fn from_parts(
        vae: Seedvr2Vae,
        transformer: Seedvr2Transformer,
        neg_embed: Tensor,
        dtype: DType,
        device: Device,
    ) -> Self {
        Self {
            vae,
            transformer,
            neg_embed,
            dtype,
            device,
        }
    }

    /// Load from a raw `numz/SeedVR2_comfyUI` checkpoint dir: convert in-memory (no Python), cast to
    /// `dt`, attach the bundled neg-embed. `dit_file` selects 3B/7B.
    pub fn load(
        raw_dir: impl AsRef<std::path::Path>,
        dit_file: &str,
        cfg: &DitConfig,
        dt: DType,
        device: &Device,
    ) -> CResult<Self> {
        let dir = raw_dir.as_ref();
        let vae_raw = Weights::from_file(dir.join("ema_vae_fp16.safetensors"), device)?;
        let dit_raw = Weights::from_file(dir.join(dit_file), device)?;
        let vae_w = convert::convert_vae(&vae_raw)?.cast(dt)?;
        let dit_w = convert::convert_dit(&dit_raw)?.cast(dt)?;
        let vae = Seedvr2Vae::from_weights(&vae_w)?;
        let transformer = Seedvr2Transformer::from_weights(&dit_w, cfg)?;
        let neg_embed = load_neg_embed(dt, device)?;
        Ok(Self::from_parts(
            vae,
            transformer,
            neg_embed,
            dt,
            device.clone(),
        ))
    }

    /// Build the static condition `[latent, ones-mask]` → `(B, 17, T', h, w)`.
    fn condition(latent: &Tensor) -> Result<Tensor> {
        let (b, _c, t, h, w) = latent.dims5()?;
        let mask = Tensor::ones((b, 1, t, h, w), latent.dtype(), latent.device())?;
        Tensor::cat(&[latent, &mask], 1)
    }

    /// One denoise step: `vid = [noise, condition]` → DiT → `noise − DiT_out`.
    fn denoise(&self, noise: &Tensor, condition: &Tensor) -> Result<Tensor> {
        let model_input = Tensor::cat(&[noise, condition], 1)?; // (B,33,T',h,w)
        let dit_out = self
            .transformer
            .forward(&model_input, &self.neg_embed, TIMESTEP)?;
        noise - dit_out
    }

    /// Decode latents and crop to `(true_h, true_w)` (first frame) → `(1,3,true_h,true_w)`.
    fn decode_crop(&self, latents: &Tensor, true_h: usize, true_w: usize) -> Result<Tensor> {
        let decoded = self.vae.decode(latents)?; // (1,3,T,H,W)
        decoded
            .narrow(2, 0, 1)?
            .squeeze(2)? // (1,3,H,W)
            .narrow(2, 0, true_h)?
            .narrow(3, 0, true_w)?
            .contiguous()
    }

    /// Full model path (no color correction): preprocessed image + injected noise → decoded crop.
    /// Public for the golden parity harness (the engine-level parity check).
    pub fn run_model(
        &self,
        processed: &Tensor,
        noise: &Tensor,
        true_h: usize,
        true_w: usize,
    ) -> Result<Tensor> {
        let latent = self.vae.encode(processed)?;
        let cond = Self::condition(&latent)?;
        let latents = self.denoise(noise, &cond)?;
        self.decode_crop(&latents, true_h, true_w)
    }

    /// End-to-end upscale: LR `image` → `(width, height)` super-resolved RGB8 image.
    pub fn generate(
        &self,
        image: &Image,
        width: usize,
        height: usize,
        seed: u64,
        softness: f32,
    ) -> CResult<Image> {
        let processed = self.preprocess(image, width, height, softness)?; // (1,3,H,W)
        let latent = self.vae.encode(&processed)?;
        let (_b, _c, lt, lh, lw) = latent.dims5()?;
        let noise = seeded_normal5(seed, (1, 16, lt, lh, lw), self.dtype, &self.device)?;
        let cond = Self::condition(&latent)?;
        let latents = self.denoise(&noise, &cond)?;
        let decoded = self.decode_crop(&latents, height, width)?; // (1,3,H,W)
        let corrected = color::apply_color_correction(
            &decoded.to_dtype(DType::F32)?,
            &processed.to_dtype(DType::F32)?,
            LUMINANCE_WEIGHT,
        )?;
        Ok(decoded_to_image(&corrected)?)
    }

    /// LR `Image` → `(1,3,height,width)` in [-1,1] at the model dtype. Bicubic resize to target;
    /// optional `softness` pre-blur via a smaller round-trip.
    pub fn preprocess(
        &self,
        image: &Image,
        width: usize,
        height: usize,
        softness: f32,
    ) -> CResult<Tensor> {
        let (ih, iw) = (image.height as usize, image.width as usize);
        let resized: Vec<f32> = if softness > 0.0 {
            let factor = 1.0 + softness.clamp(0.0, 1.0) * 7.0;
            let dw = ((width as f32 / factor) as usize).max(2);
            let dh = ((height as f32 / factor) as usize).max(2);
            let down = imageops::resize_bicubic_u8(&image.pixels, ih, iw, dh, dw);
            let down_u8: Vec<u8> = down
                .iter()
                .map(|&v| v.round().clamp(0.0, 255.0) as u8)
                .collect();
            imageops::resize_bicubic_u8(&down_u8, dh, dw, height, width)
        } else {
            imageops::resize_bicubic_u8(&image.pixels, ih, iw, height, width)
        };
        // HWC [0,255] → [-1,1] → (1,3,H,W)
        let arr = Tensor::from_vec(resized, (height, width, 3), &self.device)?;
        let arr = (arr.affine(2.0 / 255.0, -1.0))?;
        Ok(arr
            .permute((2, 0, 1))?
            .unsqueeze(0)?
            .to_dtype(self.dtype)?
            .contiguous()?)
    }
}
