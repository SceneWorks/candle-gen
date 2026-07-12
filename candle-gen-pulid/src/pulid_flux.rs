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

use std::path::PathBuf;

use candle_core::{DType, Device, Tensor};
use candle_transformers::models::flux::sampling::{get_schedule, State};
use rand::{rngs::StdRng, SeedableRng};

use candle_gen::gen_core::runtime::CancelFlag;
use candle_gen::gen_core::sampling::TimestepConvention;
use candle_gen::gen_core::{Image, PidWeights, Progress};
use candle_gen::weights::Weights;
use candle_gen::{CandleError, Result};
use candle_gen_flux::{flow_mu, DitImageInjector, FluxRefBackbone, Variant, BASE_SHIFT, MAX_SHIFT};
use candle_gen_pid::{PidDecoder, PidEngine};

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
// FLUX dev's flow-match time-shift endpoints (`BASE_SHIFT`/`MAX_SHIFT`) and the `flow_mu` linear map
// are shared from `candle-gen-flux` (sc-11249 / F-140) — PuLID (always dev) reuses the exact
// parity-critical schedule constants rather than maintaining a third copy.

/// FLUX packs the /8 VAE latent 2×2, so both render dims must be multiples of 16 (the flux1 txt2img /
/// IP-Adapter / control size floor).
const SIZE_MULTIPLE: u32 = 16;

/// Reject a below-floor request loudly before any tensor work. Without it `get_schedule(0, …)` returns
/// `[NaN]` — zero sampler steps, so the pure seeded noise is decoded and returned as a "success",
/// burning GPU time for garbage. A fast typed error mirrors the sibling bespoke lanes (`reject_zero_steps`
/// in sdxl-IP / scail2 / instantid, sc-9016, F-032); this worker-driven PuLID path has no gen-core
/// capability floor upstream of it, and (like flux1-IP) previously had no size floor either
/// (sc-11182, F-102).
fn reject_below_floor(req: &PulidFluxRequest) -> Result<()> {
    if !req.width.is_multiple_of(SIZE_MULTIPLE) || !req.height.is_multiple_of(SIZE_MULTIPLE) {
        return Err(CandleError::Msg(format!(
            "pulid_flux: width/height must be multiples of {SIZE_MULTIPLE} (got {}x{})",
            req.width, req.height
        )));
    }
    if req.steps == 0 {
        return Err(CandleError::Msg(
            "pulid_flux: steps must be >= 1 (an explicit 0 renders undenoised noise)".into(),
        ));
    }
    Ok(())
}

/// Default PuLID `id_weight` (the reference-face strength; 0–3, upstream default 1.0).
pub const DEFAULT_ID_WEIGHT: f32 = 1.0;
/// Default dev guidance for the PuLID photoreal recipe.
pub const DEFAULT_GUIDANCE: f32 = 4.0;

/// Paths to the PuLID-FLUX checkpoints.
pub struct PulidFluxPaths {
    /// The FLUX.1-dev backbone snapshot dir — auto-detected (sc-10103): either a dense black-forest-labs
    /// `FLUX.1-dev` snapshot (`flux1-dev.safetensors`, `ae.safetensors`, `text_encoder{,_2}/`,
    /// `tokenizer_2/`) OR a packed/dense `SceneWorks/flux1-dev-mlx` turnkey **tier subdir**
    /// (`…/q4`, `…/q8`, `…/bf16`: `transformer/` + `text_encoder{,_2}/` + `vae/` + `tokenizer_2/`). The
    /// [`FluxRefBackbone`] loader picks the right layout — the worker resolves the tier subdir.
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
    /// Curated unified-sampler selection (epic 7114, sc-7297). `None` (or `flow_match` / `euler`) keeps
    /// the native flow-match Euler default; a curated [`Solver`](candle_gen::gen_core::sampling::Solver)
    /// name routes the PuLID-injected flow denoise through that integrator (the candle PuLID runs its
    /// OWN flow loop, vs the mlx PuLID which delegates to the FLUX backbone — so the knob is threaded
    /// here directly through [`candle_gen::run_flow_sampler`]).
    pub sampler: Option<String>,
    /// Curated σ-schedule selection (epic 7114). `None` (or a native alias) ⇒ FLUX's verbatim
    /// time-shifted `get_schedule`; a curated scheduler name re-strides σ over the dev time-shift `mu`.
    pub scheduler: Option<String>,
    pub seed: u64,
    /// Opt into the PiD super-resolving decoder (epic 7840, sc-8044): when `true` **and** the model was
    /// loaded with [`with_pid`](PulidFlux::with_pid), the final latent is decoded by the `flux` PiD student
    /// (4× SR → 2K/4K) instead of the native FLUX.1 VAE. PiD is a *generative* decoder, so face likeness
    /// may shift — the user judges per-generation. `false` (default) keeps the byte-exact VAE decode.
    pub use_pid: bool,
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
            sampler: None,
            scheduler: None,
            seed: 0,
            use_pid: false,
            cancel: CancelFlag::default(),
        }
    }
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

/// The loaded PuLID-FLUX model: the tier-detected FLUX backbone (text encoders, DiT, VAE, tokenizers),
/// the EVA tower, the IDFormer, the kept PuLID checkpoint (for the per-generate [`PulidCa`]), and the
/// native face stack.
pub struct PulidFlux {
    device: Device,
    dtype: DType,
    /// The FLUX.1-dev backbone — CLIP + T5 + DiT + VAE + both tokenizers, tier-detected at load
    /// (sc-10103): a dense BFL snapshot or a packed/dense `SceneWorks/flux1-dev-mlx` turnkey tier. The
    /// PuLID CA identity injection drives its post-block [`DitImageInjector`]
    /// [`forward_injected`](FluxRefBackbone::forward_injected) seam. Reuses the base FLUX txt2img pipeline
    /// load path, so PuLID and `flux_dev` never drift on tokenization / tier detection / VAE decode.
    backbone: FluxRefBackbone,
    eva: EvaVisionTransformer,
    idformer: IdFormer,
    /// The PuLID checkpoint (f32) — kept to build a per-generate [`PulidCa`] from `pulid_ca.*`
    /// (`pulid_encoder.*` is already consumed by `idformer`).
    pulid: Weights,
    face: candle_gen_face::CandleFaceAnalysis,
    /// Optional PiD super-resolving decoder (epic 7840, sc-8044), attached via [`with_pid`](Self::with_pid).
    /// PuLID composes the FLUX.1-dev VAE, so it loads the `flux` student (same tag as the base FLUX provider).
    pid: Option<PidEngine>,
}

impl PulidFlux {
    /// Load the FLUX.1-dev backbone + the EVA tower + the IDFormer + the PuLID CA weights + the native
    /// face stack (with the BiSeNet parser) from the [`PulidFluxPaths`].
    pub fn load(paths: &PulidFluxPaths) -> Result<Self> {
        let device = candle_gen::default_device()?;
        let dtype = DTYPE;

        // FLUX.1-dev backbone — tier-detected (sc-10103): a dense BFL snapshot or a packed/dense
        // `SceneWorks/flux1-dev-mlx` turnkey tier subdir. Reuses the base FLUX txt2img pipeline's
        // detect-and-load, so PuLID consumes the SAME q4/q8/bf16 tiers `flux_dev` does. The PuLID CA
        // injection runs through the backbone's post-block `forward_injected` seam (on the BFL `IpFlux`
        // or the diffusers `PackedFluxDit`, whichever tier loaded).
        let backbone = FluxRefBackbone::load(&paths.flux_base, Variant::Dev, &device, dtype)?;

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
            device,
            dtype,
            backbone,
            eva,
            idformer,
            pulid,
            face,
            pid: None,
        })
    }

    /// Attach the optional PiD super-resolving decoder (epic 7840, sc-8044). Same [`PidWeights`] load-spec
    /// as the registry FLUX.1 provider; PuLID composes the FLUX.1-dev VAE, so it loads the `flux` student.
    /// A `use_pid = true` request then decodes through it (4× SR) instead of the native VAE; without it,
    /// `use_pid` errors loudly. Face likeness may shift under the generative decode — the user's per-gen
    /// call. Call after [`load`](Self::load).
    pub fn with_pid(mut self, pid: &PidWeights) -> Result<Self> {
        self.pid = Some(PidEngine::from_spec(pid, "flux", &self.device)?);
        Ok(self)
    }

    /// Mint the per-generation PiD decoder when the request opted in (`use_pid`) and a student is loaded;
    /// `None` keeps the native VAE decode. Errors loudly if `use_pid` is set without a prior
    /// [`with_pid`](Self::with_pid). A clean-latent (σ=0) decoder bound to the prompt + seed; the request
    /// cancel threads in for a cancellable SR decode.
    fn pid_decoder_for(&self, req: &PulidFluxRequest) -> Result<Option<PidDecoder>> {
        // Route through the shared guarded seam (sc-11242 / F-091) so the SR decode is budgeted
        // (F-013 sc-9095) and spatially tiled (sc-10087). Clean-latent σ=0 decode, single image.
        candle_gen_pid::resolve_pid_decoder_for_fields(
            self.pid.as_ref(),
            req.use_pid,
            &req.prompt,
            1,
            req.width,
            req.height,
            &req.cancel,
            req.seed,
            "pulid",
            0.0,
        )
    }

    /// Reference face (RGB [`Image`]) → `id_embedding` `[1,32,2048]` (f32). Mirrors PuLID's
    /// `get_id_embedding` (the conditional side).
    pub fn compute_id_embedding(&self, reference: &Image) -> Result<Tensor> {
        let inner = self.face.inner();
        let (h, w) = (reference.height as usize, reference.width as usize);
        // Detect-then-embed-the-largest (sc-11249 / F-138), matching the `candle-gen-face` /
        // InstantID pattern: `detect` returns detections largest-first (no ArcFace forward), and
        // `embed` runs the single `[1,3,112,112]` iresnet100 forward on the largest one only. This
        // avoids `analyze`'s N−1 wasted host norm-crops + N-row batched forward for a group-photo
        // reference; iresnet100 has no cross-batch ops, so the kept embedding is bit-identical to the
        // old `analyze(..).first()`.
        let dets = inner.detect(&reference.pixels, h, w)?;
        let det = dets.first().ok_or_else(|| {
            CandleError::Msg("pulid_flux: no face detected in the reference image".into())
        })?;
        let face = inner.embed(&reference.pixels, h, w, det)?;
        // ArcFace 512-d (raw, un-normalized) → [1, 512] f32.
        let dim = face.embedding.len();
        let arcface = Tensor::from_vec(face.embedding.clone(), (1, dim), &self.device)?;
        // face_features_image (512² NCHW) → EVA 336² transform → tower.
        let ffi = inner.face_features_image(&reference.pixels, h, w, &face)?;
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
        reject_below_floor(req)?;

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

        // Text conditioning (T5 seq + CLIP pooled) — tier-agnostic via the backbone (dense or packed
        // encoders, same token ids either way).
        let (t5_emb, clip_emb) = self.backbone.encode_text(&req.prompt)?;

        // candle's get_noise geometry: latent is /8 of a multiple-of-16 request.
        let lat_h = (req.height as usize).div_ceil(16) * 2;
        let lat_w = (req.width as usize).div_ceil(16) * 2;
        let n = LATENT_CHANNELS * lat_h * lat_w;
        // sc-3673 parity: deterministic, launch-portable CPU-seeded initial noise.
        let mut rng = StdRng::seed_from_u64(req.seed);
        let noise = candle_gen::seeded_normal_vec(&mut rng, n);
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
            req.sampler.as_deref(),
            req.scheduler.as_deref(),
            req.seed,
            &req.cancel,
            on_progress,
        )?;
        on_progress(Progress::Decoding);
        // Decode the final latent: native FLUX.1 VAE by default, or the `flux` PiD student (4× SR) when
        // this generation opted in (`req.use_pid`) and `with_pid` loaded one (epic 7840, sc-8044). PiD is a
        // generative decoder, so face likeness may shift — the user's per-gen call.
        let pid_decoder = self.pid_decoder_for(req)?;
        self.backbone.decode(
            &latents,
            req.height as usize,
            req.width as usize,
            pid_decoder.as_ref(),
        )
    }

    /// The flow-match denoise with the PuLID CA injector, routed through the unified curated
    /// sampler/scheduler driver (epic 7114, sc-7297). The `scheduler` axis re-strides FLUX's native
    /// `get_schedule(..)` over the dev time-shift `mu`; the `sampler` axis picks the integrator. The
    /// forked [`IpFlux::forward_injected`] (`Some(injector)`) is the model forward, and the PuLID CA
    /// identity injection stays INSIDE the `predict` closure so a multi-eval solver (heun / dpmpp) re-runs
    /// the whole step. The DEFAULT (`sampler`/`scheduler` unset ⇒ euler over the native schedule) is the
    /// N1 path for the legacy inline flow-match Euler loop `img += pred·(σ_{i+1} − σ_i)`. FLUX feeds the
    /// raw timestep (`Sigma` convention: `t == σ`); guidance is a per-batch tensor the dev DiT embeds.
    /// Cancellation + progress are owned by the driver.
    #[allow(clippy::too_many_arguments)]
    fn denoise(
        &self,
        state: &State,
        timesteps: &[f64],
        guidance: f64,
        injector: &PulidCa,
        sampler: Option<&str>,
        scheduler: Option<&str>,
        seed: u64,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Tensor> {
        let b_sz = state.img.dim(0)?;
        let guidance_t = Tensor::full(guidance as f32, b_sz, &self.device)?;
        // Native schedule = candle's verbatim `get_schedule(..)` (f32 descending, trailing 0.0); the
        // default (scheduler unset / native alias) returns it byte-exact, so the legacy flow-match Euler
        // path is the N1 no-op for `img += pred·(σ_{i+1} − σ_i)`.
        let native: Vec<f32> = timesteps.iter().map(|&t| t as f32).collect();
        // PuLID is always dev, so the shared `flow_mu` is fed `Variant::Dev` (the schnell `mu = 0`
        // branch is inert here) — the candle-gen-flux twin, deduped in sc-11249 / F-140.
        let mu = flow_mu(Variant::Dev, state.img.dim(1)?);
        let steps = native.len().saturating_sub(1);
        let sigmas = candle_gen::resolve_flow_schedule(scheduler, mu, steps, &native);
        candle_gen::run_flow_sampler(
            sampler,
            TimestepConvention::Sigma,
            &sigmas,
            state.img.clone(),
            seed,
            cancel,
            on_progress,
            |img, t| -> Result<Tensor> {
                // The backbone dispatches to the loaded tier's DiT (BFL `IpFlux` or packed
                // `PackedFluxDit`) `forward_injected`; the PuLID CA identity injection lives inside this
                // closure so a multi-eval solver re-runs the whole step.
                let t_vec = Tensor::full(t, b_sz, &self.device)?;
                self.backbone.forward_injected(
                    img,
                    &state.img_ids,
                    &state.txt,
                    &state.txt_ids,
                    &t_vec,
                    &state.vec,
                    Some(&guidance_t),
                    Some(injector as &dyn DitImageInjector),
                )
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `steps == 0` and a non-multiple-of-16 size are fast typed errors (never a decoded pure-noise
    /// "success"); the defaults pass (sc-11182, F-102).
    #[test]
    fn reject_below_floor_floors_steps_and_size() {
        let base = PulidFluxRequest::default();

        let zero_steps = PulidFluxRequest {
            steps: 0,
            ..base.clone()
        };
        let err = reject_below_floor(&zero_steps).unwrap_err();
        assert!(err.to_string().contains("steps must be >= 1"), "{err}");

        let bad_size = PulidFluxRequest {
            height: 1000, // not a multiple of 16
            ..base.clone()
        };
        let err = reject_below_floor(&bad_size).unwrap_err();
        assert!(err.to_string().contains("multiples of 16"), "{err}");

        assert!(reject_below_floor(&base).is_ok());
    }

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

    /// The shared `candle-gen-flux` `flow_mu`/shift constants (sc-11249 / F-140) reproduce the deleted
    /// PuLID-local copy bit-for-bit: `BASE_SHIFT`/`MAX_SHIFT` are the dev endpoints and
    /// `flow_mu(Variant::Dev, ..)` equals the old inlined linear map at every seq_len.
    #[test]
    fn shared_flow_mu_matches_deleted_pulid_copy() {
        assert_eq!((BASE_SHIFT, MAX_SHIFT), (0.5, 1.15));
        // The exact formula the PuLID copy used before the dedup.
        let old = |seq_len: usize| -> f32 {
            let m = (MAX_SHIFT - BASE_SHIFT) / (4096.0 - 256.0);
            let b = BASE_SHIFT - m * 256.0;
            (m * seq_len as f64 + b) as f32
        };
        for &seq_len in &[256usize, 512, 1024, 4096, 4224] {
            assert_eq!(
                flow_mu(Variant::Dev, seq_len),
                old(seq_len),
                "seq_len={seq_len}"
            );
        }
        // Endpoints land on the shift constants (m·256+b = BASE_SHIFT, m·4096+b = MAX_SHIFT).
        assert!((flow_mu(Variant::Dev, 256) as f64 - BASE_SHIFT).abs() < 1e-6);
        assert!((flow_mu(Variant::Dev, 4096) as f64 - MAX_SHIFT).abs() < 1e-6);
    }
}
