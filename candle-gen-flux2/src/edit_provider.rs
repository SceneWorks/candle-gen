//! FLUX.2-klein **reference-image edit** provider (sc-5487, epic 5480) — Kontext-style edit / identity
//! conditioning on FLUX.2-klein-9B off-Mac (Windows/CUDA), the candle sibling of the `mlx-gen-flux2`
//! edit variant (`flux2_klein_9b_edit`) and the **provider half** that unblocks the worker wiring.
//! FLUX.2-klein has no torch path (it is diffusers/MLX-only), so this lane retires the worker's
//! `edit_image` → torch deferral for `flux2_klein_9b`.
//!
//! **How it conditions (no transformer change):** each reference image is VAE-encoded into the packed,
//! bn-normalized transformer latent ([`Flux2Vae::encode_packed`]) and packed to tokens, then
//! concatenated AFTER the noised target tokens on the sequence axis — the joint image stream
//! `[target, ref0, ref1, …]`. The reference grid ids are offset at `t = 10 + 10·i` (the mlx fork's
//! per-reference temporal coordinate) so the 4-axis RoPE keeps the references positionally distinct
//! from the `t = 0` target grid. The existing [`Flux2Transformer::forward`] already accepts arbitrary
//! `img_ids`, so it runs the full joint sequence unchanged; the provider keeps the leading `target_seq`
//! velocity tokens and steps only the target. The reference tokens are clean and constant across the
//! denoise (re-concatenated each step, never noised).
//!
//! Bespoke provider (NOT gen-core-registered), worker-invoked by name — mirroring the SDXL edit /
//! IP-Adapter / InstantID / PuLID providers. Determinism is the candle-lane contract (sc-3673): the
//! seeded CPU init noise reuses [`pipeline::create_noise`]. Distilled klein runs CFG-free (guidance
//! 1.0); guidance > 1 adds a classifier-free negative pass (the same convention as txt2img). No
//! `strength`: FLUX.2 edit conditions via reference token concat (a full denoise from noise), not an
//! img2img noise blend.

use std::path::PathBuf;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::imageops::resize_lanczos_u8;
use candle_gen::gen_core::runtime::CancelFlag;
use candle_gen::gen_core::sampling::TimestepConvention;
use candle_gen::gen_core::{Image, PidWeights, Progress, Quant};
// `LatentDecoder` brings the `PidDecoder::decode` trait method into scope (sc-8044).
use candle_gen::{CandleError, LatentDecoder, Result};
use candle_gen_pid::{PidDecoder, PidEngine};

use crate::config::{Flux2Variant, DEFAULT_GUIDANCE, DEFAULT_STEPS, SIZE_MULTIPLE};
use crate::text_encoder::Flux2PromptEncoder;
use crate::transformer::Flux2Transformer;
use crate::vae::Flux2Vae;
use crate::{pipeline, to_image, Pipeline, PID_BACKBONE};

/// Path to the FLUX.2 edit snapshot — just the diffusers snapshot dir (`text_encoder/`,
/// `transformer/`, `vae/`, `tokenizer/`), the same snapshot the txt2img path loads. klein at
/// `black-forest-labs/FLUX.2-klein-9B` ([`Flux2Edit::load`]); dev at `black-forest-labs/FLUX.2-dev`
/// ([`Flux2Edit::load_dev`], sc-7460).
pub struct Flux2EditPaths {
    /// FLUX.2 diffusers snapshot dir (klein or dev).
    pub root: PathBuf,
}

/// One FLUX.2-klein edit request.
#[derive(Clone)]
pub struct Flux2EditRequest {
    pub prompt: String,
    /// Classifier-free negative prompt — used only when `guidance > 1` (distilled klein runs CFG-free).
    pub negative: String,
    pub width: u32,
    pub height: u32,
    pub steps: usize,
    /// Guidance scale. 1.0 (klein default) = a single CFG-free forward; > 1.0 adds a negative pass.
    pub guidance: f32,
    pub seed: u64,
    /// Opt into the PiD super-resolving decoder (epic 7840, sc-8044): when `true` **and** the model was
    /// loaded with [`with_pid`](Flux2Edit::with_pid), the final latent is decoded by the `flux2` PiD
    /// student (4× SR → 2K/4K) instead of the native FLUX.2 VAE. `false` (default) keeps the VAE decode.
    pub use_pid: bool,
    /// Cooperative cancellation, checked before each denoise step (the engine contract).
    pub cancel: CancelFlag,
}

impl Default for Flux2EditRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            negative: String::new(),
            width: 1024,
            height: 1024,
            steps: DEFAULT_STEPS as usize,
            guidance: DEFAULT_GUIDANCE,
            seed: 0,
            use_pid: false,
            cancel: CancelFlag::default(),
        }
    }
}

/// Loaded FLUX.2-klein edit model: the Qwen3 text encoder + the MMDiT + the VAE **with the encoder**
/// (the reference encode), plus the txt2img [`Pipeline`] handle (snapshot mmap + prompt encode + the
/// latent geometry/dtype). `generate` takes `&self` (no per-call mutation), so one load serves many
/// edits.
pub struct Flux2Edit {
    pipe: Pipeline,
    variant: Flux2Variant,
    te: Flux2PromptEncoder,
    /// Prompt tokenizer, loaded+parsed **once** at load and reused across encodes (sc-8991 / F-011)
    /// instead of re-parsing `tokenizer.json` per prompt/branch.
    tokenizer: candle_gen::gen_core::tokenizer::TextTokenizer,
    transformer: Flux2Transformer,
    vae: Flux2Vae,
    /// Optional PiD super-resolving decoder (epic 7840, sc-8044), attached via [`with_pid`](Self::with_pid).
    /// FLUX.2 edit composes the FLUX.2 VAE, so it loads the SAME `flux2` student ([`PID_BACKBONE`]) as the
    /// registered FLUX.2 provider.
    pid: Option<PidEngine>,
}

impl Flux2Edit {
    /// Load the **klein** edit backbone (dense) with the VAE encoder enabled (the reference encode);
    /// distilled — guidance 1.0 (CFG-free), > 1 adds a negative pass.
    pub fn load(paths: &Flux2EditPaths) -> Result<Self> {
        Self::load_variant(paths, Flux2Variant::Klein9b, None)
    }

    /// Load the **dev** edit backbone (sc-7460): the 32B flagship via the CPU-stage → quantize-onto-GPU
    /// loader (`quant` Q4/Q8 required in practice — the dense 32B does not fit the GPU), guidance-
    /// distilled (embedded scalar, no negative pass), text-only Mistral prompt + reference token concat.
    pub fn load_dev(paths: &Flux2EditPaths, quant: Option<Quant>) -> Result<Self> {
        Self::load_variant(paths, Flux2Variant::Dev, quant)
    }

    /// Shared loader: the backbone for `variant` with the VAE encoder enabled (the reference encode).
    /// The dev quant path stages the TE + DiT dense in CPU RAM and quantizes each projection onto the
    /// GPU; klein (and dev on a fixture) loads dense on-device. f32 compute (parity-sensitive).
    fn load_variant(
        paths: &Flux2EditPaths,
        variant: Flux2Variant,
        quant: Option<Quant>,
    ) -> Result<Self> {
        let device = candle_gen::default_device()?;
        // PiD (super-resolving decode) is wired only through the txt2img render path (epic 7840 /
        // sc-7853); the edit provider passes `None`.
        let pipe = Pipeline::load(variant, quant, &paths.root, &device, None);
        // Packed MLX tier → build directly on the GPU from the packed parts (sc-9087, no ~105 GB dense
        // CPU staging); dense tier → the legacy CPU-stage → quantize-onto-GPU path. Shared TE+DiT loader
        // with txt2img / control (F-024, sc-9004). The VAE *with encoder* (the reference encode) is the
        // per-site addition.
        let (te, transformer) = pipe.load_te_and_dit()?;
        let vae = Flux2Vae::new_with_encoder(pipe.component_vb("vae")?)?;
        let tokenizer = pipe.build_tokenizer()?;
        Ok(Self {
            pipe,
            variant,
            te,
            tokenizer,
            transformer,
            vae,
            pid: None,
        })
    }

    /// Attach the optional PiD super-resolving decoder (epic 7840, sc-8044). Same [`PidWeights`] load-spec
    /// as the registry FLUX.2 provider; edit composes the FLUX.2 VAE so it loads the **same**
    /// [`PID_BACKBONE`] (`flux2`) student. A `use_pid = true` request then decodes through it (4× SR)
    /// instead of the native VAE; without it, `use_pid` errors loudly. Call after [`load`](Self::load).
    pub fn with_pid(mut self, pid: &PidWeights) -> Result<Self> {
        self.pid = Some(PidEngine::from_spec(pid, PID_BACKBONE, &self.pipe.device)?);
        Ok(self)
    }

    /// Mint the per-generation PiD decoder when the request opted in (`use_pid`) and a student is loaded;
    /// `None` keeps the native VAE decode. Errors loudly if `use_pid` is set without a prior
    /// [`with_pid`](Self::with_pid). A clean-latent (σ=0) decoder bound to the prompt + seed; the request
    /// cancel threads in for a cancellable SR decode.
    fn pid_decoder_for(&self, req: &Flux2EditRequest) -> Result<Option<PidDecoder>> {
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
            "flux2 edit",
            0.0,
        )
    }

    /// Generate one edited image. `references` (≥ 1) condition the denoise via reference token concat;
    /// the worker pre-fits them to the render size, but this re-resizes defensively.
    pub fn generate(
        &self,
        req: &Flux2EditRequest,
        references: &[Image],
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        if req.cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }
        if references.is_empty() {
            return Err(CandleError::Msg(
                "flux2 edit: at least one reference image is required".into(),
            ));
        }
        validate_request(req)?;

        let device = &self.pipe.device;
        let cfg = &self.pipe.cfg;
        let guidance = req.guidance;
        // dev is guidance-distilled (embedded scalar, single forward); klein is distilled / true-CFG
        // (a classifier-free negative pass only when guidance > 1).
        let embedded_guidance = self.variant.uses_embedded_guidance();
        let cfg_on = !embedded_guidance && guidance > 1.0;

        // Prompt embeds are seed-independent: encode once. Negative only under klein CFG.
        let prompt_embeds = self.pipe.encode(&self.te, &self.tokenizer, &req.prompt)?;
        let negative = if cfg_on {
            let neg = if req.negative.trim().is_empty() {
                " "
            } else {
                req.negative.as_str()
            };
            Some(self.pipe.encode(&self.te, &self.tokenizer, neg)?)
        } else {
            None
        };

        // Reference conditioning: VAE-encode each ref → packed tokens [1, seq_ref, 128] + grid ids at
        // t = 10 + 10·i, all concatenated on the sequence axis. Clean + constant across the denoise.
        let (ref_tokens, ref_ids) = self.encode_references(references, req.width, req.height)?;

        let (lat_h, lat_w) = pipeline::latent_dims(req.width, req.height);
        let target_seq = lat_h * lat_w;
        // The joint image-stream ids: the t=0 target grid followed by the reference grids.
        let mut img_ids = pipeline::prepare_grid_ids(lat_h, lat_w);
        img_ids.extend_from_slice(&ref_ids);
        let txt_ids = pipeline::prepare_text_ids(cfg.max_sequence_length);

        // Curated sampler/scheduler routing (epic 7114 P4, sc-7123) — the same driver the txt2img path
        // uses. The bespoke edit request carries no per-generation sampler/scheduler knob, so this runs
        // the default (`None`) euler over the native empirical-mu schedule: the N1 no-op that reproduces
        // the legacy `euler_step` flow-match loop within tolerance.
        let mu = pipeline::compute_mu(pipeline::image_seq_len(req.width, req.height), req.steps);
        let native = pipeline::schedule(req.steps, req.width, req.height);
        let sigmas = candle_gen::resolve_flow_schedule(None, mu, req.steps, &native);

        let latents = pipeline::create_noise(cfg, req.seed, req.width, req.height, device)?;
        // The driver does cancel + progress + the integrator step. The joint `[target, refs]` concat,
        // the transformer forward, the target-slice, and the guidance>1 CFG blend all live inside the
        // predict closure so a multi-eval solver re-runs them. FLUX.2 uses the Sigma convention but the
        // model embeds σ×1000, so feed `sigma * 1000.0` to the transformer.
        let latents = candle_gen::run_flow_sampler(
            None,
            TimestepConvention::Sigma,
            &sigmas,
            latents,
            req.seed,
            &req.cancel,
            on_progress,
            |latents, sigma| -> Result<Tensor> {
                let ts = sigma * 1000.0;
                // Joint image stream [target, refs] — references re-concatenated with the current target.
                let hidden = Tensor::cat(&[latents, &ref_tokens], 1)?;
                if embedded_guidance {
                    // dev: a single forward feeding the embedded guidance scalar to the DiT.
                    return self.velocity(
                        &hidden,
                        &prompt_embeds,
                        &img_ids,
                        &txt_ids,
                        ts,
                        Some(guidance),
                        target_seq,
                    );
                }
                // klein: distilled (CFG-free) or true-CFG via a negative pass when guidance > 1.
                let v = self.velocity(
                    &hidden,
                    &prompt_embeds,
                    &img_ids,
                    &txt_ids,
                    ts,
                    None,
                    target_seq,
                )?;
                match &negative {
                    Some(neg) => {
                        let vn =
                            self.velocity(&hidden, neg, &img_ids, &txt_ids, ts, None, target_seq)?;
                        // vn + guidance·(v − vn)
                        Ok((&vn + ((&v - &vn)? * guidance as f64)?)?)
                    }
                    None => Ok(v),
                }
            },
        )?;

        on_progress(Progress::Decoding);
        let packed = pipeline::unpack_latents(&latents, req.width, req.height)?;
        // Decode the final latent: native FLUX.2 VAE by default, or the `flux2` PiD student (4× SR) when
        // this generation opted in (`req.use_pid`) and `with_pid` loaded one (sc-8044). Both take the same
        // unpacked latent and emit `[-1, 1]` pixels (PiD at 4×); `to_image` reads the size from the tensor.
        let pid_decoder = self.pid_decoder_for(req)?;
        let decoded = match &pid_decoder {
            Some(pid) => pid.decode(&packed)?,        // [1,3,4H,4W]
            None => self.vae.decode_packed(&packed)?, // [1,3,H,W] in [-1,1]
        };
        to_image(&decoded)
    }

    /// Run the transformer on the joint `[target, refs]` image stream and keep the leading
    /// `target_seq` velocity tokens (the target image stream; `proj_out` is per-token, so the slice is
    /// exact). `guidance` is `Some(scale)` for dev (embedded guidance) and `None` for klein (distilled
    /// / true-CFG via the caller's negative pass).
    #[allow(clippy::too_many_arguments)]
    fn velocity(
        &self,
        hidden: &Tensor,
        embeds: &Tensor,
        img_ids: &[[i64; 4]],
        txt_ids: &[[i64; 4]],
        ts: f32,
        guidance: Option<f32>,
        target_seq: usize,
    ) -> Result<Tensor> {
        let out = self
            .transformer
            .forward(hidden, embeds, img_ids, txt_ids, ts, guidance)?;
        Ok(out.narrow(1, 0, target_seq)?)
    }

    /// Encode N reference images into packed transformer tokens + their grid ids. Each: Lanczos-resize
    /// to the render size → normalize to `[-1,1]` NCHW → [`Flux2Vae::encode_packed`] (the mean encode +
    /// 2×2 patchify + bn-normalize the transformer space expects) → pack to `[1, seq, 128]`, tagged
    /// with grid ids at `t = 10 + 10·i`. Returns the concatenated `([1, Σseq, 128], Σ grid ids)`.
    fn encode_references(
        &self,
        references: &[Image],
        width: u32,
        height: u32,
    ) -> Result<(Tensor, Vec<[i64; 4]>)> {
        let (lat_h, lat_w) = pipeline::latent_dims(width, height);
        let mut tokens: Vec<Tensor> = Vec::with_capacity(references.len());
        let mut ids: Vec<[i64; 4]> = Vec::with_capacity(references.len() * lat_h * lat_w);
        for (i, image) in references.iter().enumerate() {
            let nchw = preprocess_ref(image, width, height, &self.pipe.device, self.pipe.dtype)?;
            let packed = self.vae.encode_packed(&nchw)?; // [1, 128, H/16, W/16]
            tokens.push(pipeline::pack_nchw(&packed)?); // [1, seq, 128]
            ids.extend(pipeline::prepare_grid_ids_t(
                lat_h,
                lat_w,
                10 + 10 * i as i64,
            ));
        }
        Ok((Tensor::cat(&tokens, 1)?, ids))
    }
}

/// Validate the seed-independent request knobs before any tensor work. The empty-prompt guard
/// (sc-8987, the sc-8646 bug class) mirrors the registered txt2img `validate` and the flux1 control
/// provider: `gen_core::TextTokenizer::tokenize("")` short-circuits to a (1, 0) encoding BEFORE the
/// chat template runs, so an empty prompt would reach the TE as a zero-length sequence and surface
/// as a deep tensor-shape error (or degenerate conditioning) instead of a clean validation error.
fn validate_request(req: &Flux2EditRequest) -> Result<()> {
    if req.prompt.trim().is_empty() {
        return Err(CandleError::Msg("flux2 edit: prompt is required".into()));
    }
    if !req.width.is_multiple_of(SIZE_MULTIPLE) || !req.height.is_multiple_of(SIZE_MULTIPLE) {
        return Err(CandleError::Msg(format!(
            "flux2 edit: width/height must be multiples of {SIZE_MULTIPLE} (got {}x{})",
            req.width, req.height
        )));
    }
    if req.steps == 0 {
        return Err(CandleError::Msg("flux2 edit: steps must be >= 1".into()));
    }
    Ok(())
}

/// Lanczos-resize a reference [`Image`] (RGB8) to the render size, normalize `[0,255] → [-1,1]`, lay
/// out as NCHW `[1, 3, H, W]` — the input [`Flux2Vae::encode_packed`] expects. Mirrors the mlx
/// `preprocess_ref_image` (`2·x − 1`). A no-op resize when the source is already the render size.
/// `pub(crate)` so the control provider ([`crate::control_provider`]) reuses it to VAE-encode the
/// pose/union control image (sc-7460).
pub(crate) fn preprocess_ref(
    image: &Image,
    width: u32,
    height: u32,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let (iw, ih) = (image.width as usize, image.height as usize);
    if image.pixels.len() != iw * ih * 3 {
        return Err(CandleError::Msg(format!(
            "flux2 edit: reference pixel buffer {} != {iw}x{ih}x3",
            image.pixels.len()
        )));
    }
    let (rw, rh) = (width as usize, height as usize);
    let resized: Vec<f32> = if (ih, iw) == (rh, rw) {
        image.pixels.iter().map(|&v| v as f32).collect()
    } else {
        resize_lanczos_u8(&image.pixels, ih, iw, rh, rw)? // HWC f32 [0,255]
    };
    // [0,255] → [-1,1], then HWC → NCHW.
    let data: Vec<f32> = resized.iter().map(|&v| 2.0 * (v / 255.0) - 1.0).collect();
    let hwc = Tensor::from_vec(data, (rh, rw, 3), device)?;
    let nchw = hwc.permute((2, 0, 1))?.unsqueeze(0)?.contiguous()?;
    Ok(nchw.to_dtype(dtype)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The request defaults match the klein edit production knobs (1024², 4 distilled steps, CFG-free).
    #[test]
    fn request_defaults() {
        let r = Flux2EditRequest::default();
        assert_eq!((r.width, r.height), (1024, 1024));
        assert_eq!(r.steps, DEFAULT_STEPS as usize);
        assert_eq!(r.guidance, DEFAULT_GUIDANCE);
        assert!(!r.cancel.is_cancelled());
    }

    /// The empty-prompt guard (sc-8987, sc-8646 bug class): an empty or whitespace-only prompt is a
    /// clean validation error, never a zero-length TE sequence; a real prompt passes.
    #[test]
    fn validate_request_rejects_empty_prompt() {
        let empty = Flux2EditRequest::default();
        let err = validate_request(&empty).unwrap_err();
        assert!(err.to_string().contains("prompt is required"), "{err}");

        let whitespace = Flux2EditRequest {
            prompt: " \t\n".into(),
            ..Default::default()
        };
        let err = validate_request(&whitespace).unwrap_err();
        assert!(err.to_string().contains("prompt is required"), "{err}");

        let ok = Flux2EditRequest {
            prompt: "a portrait".into(),
            ..Default::default()
        };
        assert!(validate_request(&ok).is_ok());
    }

    /// The size/steps guards moved into `validate_request` still fire (no regression from the
    /// sc-8987 refactor).
    #[test]
    fn validate_request_keeps_size_and_steps_guards() {
        let odd = Flux2EditRequest {
            prompt: "a portrait".into(),
            width: 1000,
            ..Default::default()
        };
        assert!(validate_request(&odd)
            .unwrap_err()
            .to_string()
            .contains("multiples"));

        let zero_steps = Flux2EditRequest {
            prompt: "a portrait".into(),
            steps: 0,
            ..Default::default()
        };
        assert!(validate_request(&zero_steps)
            .unwrap_err()
            .to_string()
            .contains("steps"));
    }

    /// `preprocess_ref` lays a same-size RGB8 reference out as NCHW `[1,3,H,W]` in `[-1,1]`: white → 1,
    /// black → −1 (the `2·x − 1` normalization), with the channel axis moved to front.
    #[test]
    fn preprocess_ref_normalizes_and_lays_out_nchw() {
        let dev = Device::Cpu;
        // 2×2 image: top-left white, the rest black.
        let pixels = vec![
            255, 255, 255, 0, 0, 0, // row 0: white, black
            0, 0, 0, 0, 0, 0, // row 1: black, black
        ];
        let img = Image {
            width: 2,
            height: 2,
            pixels,
        };
        let t = preprocess_ref(&img, 2, 2, &dev, DType::F32).unwrap();
        assert_eq!(t.dims(), &[1, 3, 2, 2]);
        // Channel 0 (R), row-major after the HWC→NCHW move: [1, −1, −1, −1].
        let r = t
            .narrow(1, 0, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(r, vec![1.0, -1.0, -1.0, -1.0]);
    }
}
