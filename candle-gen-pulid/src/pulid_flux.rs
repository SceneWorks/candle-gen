//! PuLID-FLUX end-to-end provider (sc-5492) — the candle (Windows/CUDA) twin of `mlx-gen-pulid`'s
//! `pulid_flux.rs`. Assembles the full face-identity path on top of the candle FLUX.1-dev backbone:
//!
//!   1. **Face analysis** (native, `candle-gen-face`): the reference face → `FaceAnalysis::analyze` →
//!      largest face's ArcFace embedding (512-d) + `face_features_image` (512² aligned, bg-whitened
//!      grayscale via BiSeNet). No Python/onnx.
//!   2. **EVA-CLIP** ([`crate::eva_clip`]): `face_features_image` → resize/normalize → `id_cond_vit`
//!      (768-d, L2-normalized) + 5 hidden states.
//!   3. **IDFormer** ([`crate::idformer`]): `id_cond = cat(arcface 512, id_cond_vit 768)` + hidden →
//!      `id_embedding` `[1,32,2048]`.
//!   4. **CA injection** ([`crate::ca`]): build a [`PulidCa`] bound to the id_embedding and run the FLUX
//!      flow-match denoise through [`IpFlux::forward_injected`] → AutoEncoder decode.
//!
//! The conditioning path (EVA tower + IDFormer + the 20 CA modules) runs in **f32** for identity
//! fidelity; the candle FLUX DiT image stream is bf16, so the CA residual is cast to the image dtype at
//! injection (the `r.to_dtype(img.dtype())` in `IpFlux::forward_injected`). FLUX.1-dev is the only PuLID
//! backbone (guidance-distilled, single distilled forward per step — real-CFG / uncond-id is a later
//! slice, matching the candle `supports_true_cfg: false` stance).
//!
//! Like the candle InstantID / IP-Adapter providers, [`PulidFlux`] is a plain struct the worker drives
//! **directly** (a bespoke reference stream), NOT a gen-core-registered generator.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::clip::text_model::ClipTextTransformer;
use candle_transformers::models::flux::autoencoder::AutoEncoder;
use candle_transformers::models::flux::sampling::{get_schedule, State};
use candle_transformers::models::t5::{Config as T5Config, T5EncoderModel};
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};

use candle_gen::gen_core::runtime::CancelFlag;
use candle_gen::gen_core::{Image, Progress};
use candle_gen::{CandleError, Result};
use candle_gen_flux::{
    ae_config, clip_config, decode_latents, encode_text, flux_config, DitImageInjector, IpFlux,
    Variant,
};
use candle_gen_sdxl::weights::Weights;

use crate::ca::PulidCa;
use crate::eva_clip::{transform, EvaConfig, EvaVisionTransformer};
use crate::idformer::{IdFormer, IdFormerConfig};

/// FLUX.1-dev DiT block counts (the PuLID injection schedule is defined over these).
const NUM_DOUBLE_BLOCKS: usize = 19;
const NUM_SINGLE_BLOCKS: usize = 38;
/// FLUX runs at bf16; the conditioning path runs at f32 (identity fidelity).
const DTYPE: DType = DType::BF16;
const COND_DTYPE: DType = DType::F32;
/// FLUX latent channel count (the raw VAE latent / initial noise; the DiT packs it 2×2 to 64).
const LATENT_CHANNELS: usize = 16;
/// FLUX dev's resolution-dependent flow-match time-shift endpoints (the txt2img / IP-Adapter pipeline).
const BASE_SHIFT: f64 = 0.5;
const MAX_SHIFT: f64 = 1.15;

/// Default PuLID `id_weight` (the reference-face strength; 0–3, upstream default 1.0).
pub const DEFAULT_ID_WEIGHT: f32 = 1.0;
/// Default dev guidance for the PuLID photoreal recipe.
pub const DEFAULT_GUIDANCE: f32 = 4.0;

/// Paths to the PuLID-FLUX checkpoints.
pub struct PulidFluxPaths {
    /// The black-forest-labs `FLUX.1-dev` snapshot dir (`flux1-dev.safetensors`, `ae.safetensors`,
    /// `text_encoder/`, `text_encoder_2/`, `tokenizer_2/`).
    pub flux_base: PathBuf,
    /// `guozinan/PuLID` `pulid_flux_v0.9.1.safetensors` (holds both `pulid_encoder.*` = the IDFormer and
    /// `pulid_ca.*` = the 20 cross-attn modules).
    pub pulid_weights: PathBuf,
    /// The converted EVA02-CLIP-L-14-336 safetensors (the `convert_eva_clip.py` output; bare key names).
    pub eva_weights: PathBuf,
    /// The native face-stack dir (`scrfd_10g` / `arcface_iresnet100` / `bisenet_parsing`).
    pub face_dir: PathBuf,
}

/// One PuLID-FLUX generation request.
#[derive(Clone)]
pub struct PulidFluxRequest {
    pub prompt: String,
    pub width: u32,
    pub height: u32,
    pub steps: usize,
    /// Guidance scale — embedded by the dev DiT.
    pub guidance: f32,
    /// PuLID id_weight (reference-face strength; `0.0` ⇒ the no-id ablation = plain FLUX).
    pub id_weight: f32,
    pub seed: u64,
    /// Cooperative cancellation, checked before each denoise step (the engine contract).
    pub cancel: CancelFlag,
}

impl Default for PulidFluxRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            width: 1024,
            height: 1024,
            steps: 25,
            guidance: DEFAULT_GUIDANCE,
            id_weight: DEFAULT_ID_WEIGHT,
            seed: 0,
            cancel: CancelFlag::default(),
        }
    }
}

/// mmap a [`VarBuilder`] over `files` at `dtype`/`device`, erroring if any is missing.
fn mmap_vb(files: &[PathBuf], dtype: DType, device: &Device) -> Result<VarBuilder<'static>> {
    for f in files {
        if !f.is_file() {
            return Err(CandleError::Msg(format!(
                "pulid_flux snapshot is missing {}",
                f.display()
            )));
        }
    }
    // SAFETY: mmap of read-only weight files; the standard candle loading path.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(files, dtype, device)? };
    Ok(vb)
}

/// Sorted list of every `.safetensors` in `dir` (the sharded T5 checkpoint). Errors if none are found.
fn safetensors_in(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| CandleError::Msg(format!("pulid_flux: read {}: {e}", dir.display())))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(CandleError::Msg(format!(
            "pulid_flux: no .safetensors found in {}",
            dir.display()
        )));
    }
    Ok(files)
}

/// L2-normalize each row of `[B, D]` over the feature axis (the PuLID `id_cond_vit` normalization),
/// clamping the norm to a tiny epsilon so a degenerate zero-norm row yields a zero vector, not NaN.
fn l2_normalize_rows(x: &Tensor) -> candle_core::Result<Tensor> {
    let norm = x
        .sqr()?
        .sum_keepdim(1)?
        .sqrt()?
        .clamp(1e-12f32, f32::INFINITY)?;
    x.broadcast_div(&norm)
}

/// The loaded PuLID-FLUX model: the FLUX backbone (text encoders + forked DiT + VAE) + the EVA tower +
/// the IDFormer + the kept PuLID checkpoint (for the per-generate [`PulidCa`]) + the native face stack.
pub struct PulidFlux {
    /// The FLUX snapshot root (for the T5 tokenizer in `encode_text`).
    root: PathBuf,
    device: Device,
    dtype: DType,
    clip: ClipTextTransformer,
    /// `Mutex` because `T5EncoderModel::forward` takes `&mut self` while `generate` is `&self`; locked
    /// only for the once-per-request text encode.
    t5: Mutex<T5EncoderModel>,
    transformer: IpFlux,
    vae: AutoEncoder,
    eva: EvaVisionTransformer,
    idformer: IdFormer,
    /// The PuLID checkpoint (f32) — kept to build a per-generate [`PulidCa`] from `pulid_ca.*`
    /// (`pulid_encoder.*` is already consumed by `idformer`).
    pulid: Weights,
    face: candle_gen_face::CandleFaceAnalysis,
}

impl PulidFlux {
    /// Load the FLUX.1-dev backbone + the EVA tower + the IDFormer + the PuLID CA weights + the native
    /// face stack (with the BiSeNet parser) from the [`PulidFluxPaths`].
    pub fn load(paths: &PulidFluxPaths) -> Result<Self> {
        let device = candle_gen::default_device()?;
        let dtype = DTYPE;
        let variant = Variant::Dev;
        let root = paths.flux_base.clone();

        // CLIP-L (text) under `text_encoder/`.
        let clip_vb = mmap_vb(
            &[root.join("text_encoder/model.safetensors")],
            dtype,
            &device,
        )?;
        let clip = ClipTextTransformer::new(clip_vb.pp("text_model"), &clip_config())?;

        // T5-XXL under `text_encoder_2/` (sharded; config.json alongside).
        let t5_dir = root.join("text_encoder_2");
        let t5_cfg: T5Config = {
            let cfg = std::fs::read_to_string(t5_dir.join("config.json")).map_err(|e| {
                CandleError::Msg(format!("pulid_flux: read text_encoder_2/config.json: {e}"))
            })?;
            serde_json::from_str(&cfg)
                .map_err(|e| CandleError::Msg(format!("pulid_flux: parse T5 config.json: {e}")))?
        };
        let t5_vb = mmap_vb(&safetensors_in(&t5_dir)?, dtype, &device)?;
        let t5 = T5EncoderModel::load(t5_vb, &t5_cfg)?;

        // The forked FLUX DiT (the post-block injector seam) from the root BFL checkpoint.
        let dit_vb = mmap_vb(&[root.join(variant.transformer_file())], dtype, &device)?;
        let transformer = IpFlux::new(&flux_config(variant), dit_vb)?;

        // FLUX AutoEncoder (`ae.safetensors`).
        let vae_vb = mmap_vb(&[root.join("ae.safetensors")], dtype, &device)?;
        let vae = AutoEncoder::new(&ae_config(variant), vae_vb)?;

        // EVA-CLIP tower (f32 conditioning path).
        let eva_w = Weights::from_file(&paths.eva_weights, &device, COND_DTYPE).map_err(|e| {
            CandleError::Msg(format!(
                "pulid_flux: load EVA weights {:?}: {e}",
                paths.eva_weights
            ))
        })?;
        let eva = EvaVisionTransformer::from_weights(&eva_w, "", EvaConfig::default())?;

        // PuLID encoder (IDFormer) + CA weights (f32).
        let pulid = Weights::from_file(&paths.pulid_weights, &device, COND_DTYPE).map_err(|e| {
            CandleError::Msg(format!(
                "pulid_flux: load PuLID weights {:?}: {e}",
                paths.pulid_weights
            ))
        })?;
        let idformer = IdFormer::from_weights(&pulid, "pulid_encoder", IdFormerConfig::default())?;

        // Native face stack + BiSeNet parser (the `face_features_image` path).
        let face = candle_gen_face::load_with_parser_on(&paths.face_dir, &device)?;

        Ok(Self {
            root,
            device,
            dtype,
            clip,
            t5: Mutex::new(t5),
            transformer,
            vae,
            eva,
            idformer,
            pulid,
            face,
        })
    }

    /// Reference face (RGB [`Image`]) → `id_embedding` `[1,32,2048]` (f32). Mirrors PuLID's
    /// `get_id_embedding` (the conditional side).
    pub fn compute_id_embedding(&self, reference: &Image) -> Result<Tensor> {
        let inner = self.face.inner();
        let (h, w) = (reference.height as usize, reference.width as usize);
        let faces = inner.analyze(&reference.pixels, h, w)?;
        let face = faces.first().ok_or_else(|| {
            CandleError::Msg("pulid_flux: no face detected in the reference image".into())
        })?;
        // ArcFace 512-d (raw, un-normalized) → [1, 512] f32.
        let dim = face.embedding.len();
        let arcface = Tensor::from_vec(face.embedding.clone(), (1, dim), &self.device)?;
        // face_features_image (512² NCHW) → EVA 336² transform → tower.
        let ffi = inner.face_features_image(&reference.pixels, h, w, face)?;
        let eva_in = transform::eva_transform(&ffi, self.eva.config().image_size)?;
        let eva_out = self.eva.forward(&eva_in)?;
        let id_cond_vit = l2_normalize_rows(&eva_out.id_cond_vit)?; // [1,768]
        let id_cond = Tensor::cat(&[&arcface, &id_cond_vit], 1)?; // [1,1280]
        self.idformer.forward(&id_cond, &eva_out.hidden)
    }

    /// Reference-image identity T2I: condition the FLUX.1-dev generation on `reference`'s PuLID
    /// id_embedding at `req.id_weight` (a single distilled forward per step — no true-CFG).
    pub fn generate(
        &self,
        req: &PulidFluxRequest,
        reference: &Image,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        if req.cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }

        // Identity conditioning (computed once; constant across the denoise).
        let id_embedding = self.compute_id_embedding(reference)?;
        let pulid_ca = PulidCa::from_weights(
            &self.pulid,
            "pulid_ca",
            id_embedding,
            req.id_weight as f64,
            NUM_DOUBLE_BLOCKS,
            NUM_SINGLE_BLOCKS,
        )?;

        // Text conditioning (T5 seq + CLIP pooled).
        let (t5_emb, clip_emb) = encode_text(
            Variant::Dev,
            &self.root,
            &self.device,
            self.dtype,
            &self.clip,
            &self.t5,
            &req.prompt,
        )?;

        // candle's get_noise geometry: latent is /8 of a multiple-of-16 request.
        let lat_h = (req.height as usize).div_ceil(16) * 2;
        let lat_w = (req.width as usize).div_ceil(16) * 2;
        let n = LATENT_CHANNELS * lat_h * lat_w;
        // sc-3673 parity: deterministic, launch-portable CPU-seeded initial noise.
        let mut rng = StdRng::seed_from_u64(req.seed);
        let noise: Vec<f32> = (0..n).map(|_| StandardNormal.sample(&mut rng)).collect();
        let noise = Tensor::from_vec(noise, (1, LATENT_CHANNELS, lat_h, lat_w), &Device::Cpu)?
            .to_device(&self.device)?
            .to_dtype(self.dtype)?;

        let state = State::new(&t5_emb, &clip_emb, &noise)?;
        // FLUX.1-dev: resolution-dependent time-shifted flow-match schedule + embedded guidance.
        let timesteps = get_schedule(req.steps, Some((state.img.dim(1)?, BASE_SHIFT, MAX_SHIFT)));
        let guidance = req.guidance as f64;

        let latents = self.denoise(
            &state,
            &timesteps,
            guidance,
            &pulid_ca,
            &req.cancel,
            on_progress,
        )?;
        on_progress(Progress::Decoding);
        decode_latents(&self.vae, &latents, req.height as usize, req.width as usize)
    }

    /// The flow-match Euler denoise with the PuLID CA injector — the FLUX denoise calling
    /// [`IpFlux::forward_injected`] (`Some(injector)`). `img += pred·(t_prev − t_curr)` over the
    /// **descending** schedule.
    fn denoise(
        &self,
        state: &State,
        timesteps: &[f64],
        guidance: f64,
        injector: &PulidCa,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Tensor> {
        let b_sz = state.img.dim(0)?;
        let guidance_t = Tensor::full(guidance as f32, b_sz, &self.device)?;
        let total = timesteps.len().saturating_sub(1) as u32;
        let mut img = state.img.clone();
        for (i, window) in timesteps.windows(2).enumerate() {
            if cancel.is_cancelled() {
                return Err(CandleError::Canceled);
            }
            let (t_curr, t_prev) = (window[0], window[1]);
            let t_vec = Tensor::full(t_curr as f32, b_sz, &self.device)?;
            let pred = self.transformer.forward_injected(
                &img,
                &state.img_ids,
                &state.txt,
                &state.txt_ids,
                &t_vec,
                &state.vec,
                Some(&guidance_t),
                Some(injector as &dyn DitImageInjector),
            )?;
            img = (img + (pred * (t_prev - t_curr))?)?;
            on_progress(Progress::Step {
                current: i as u32 + 1,
                total,
            });
        }
        Ok(img)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The request defaults match the PuLID-FLUX dev knobs (1024², 25 steps, guidance 4.0, id 1.0).
    #[test]
    fn request_defaults() {
        let r = PulidFluxRequest::default();
        assert_eq!((r.width, r.height), (1024, 1024));
        assert_eq!(r.steps, 25);
        assert_eq!(r.guidance, DEFAULT_GUIDANCE);
        assert_eq!(r.id_weight, DEFAULT_ID_WEIGHT);
        assert!(!r.cancel.is_cancelled());
    }

    /// `l2_normalize_rows` returns unit-norm rows (and the FLUX block counts the schedule is built over
    /// are the canonical 19 / 38).
    #[test]
    fn l2_normalize_and_block_counts() {
        let dev = Device::Cpu;
        let x = Tensor::from_vec(vec![3f32, 4.0, 0.0, 0.0], (2, 2), &dev).unwrap();
        let n = l2_normalize_rows(&x).unwrap();
        let rows = n.to_vec2::<f32>().unwrap();
        assert!((rows[0][0] - 0.6).abs() < 1e-5 && (rows[0][1] - 0.8).abs() < 1e-5);
        // Second row is all-zero → stays zero (epsilon-clamped, no NaN).
        assert!(rows[1].iter().all(|&v| v == 0.0));
        assert_eq!((NUM_DOUBLE_BLOCKS, NUM_SINGLE_BLOCKS), (19, 38));
    }
}
