//! The candle Chroma **txt2img** pipeline — the T5-XXL prompt encode → the Chroma DiT (true-CFG
//! flow-match Euler) → the FLUX 16-ch AutoencoderKL, driven through the backend-neutral
//! [`gen_core::Generator`] contract and parity-matched to the macOS `mlx-gen-chroma` provider.
//!
//! Parity choices (grounded in the mlx `model.rs`):
//! - **Packing**: noise is drawn in the VAE's /8 latent `[1, 16, h/8, w/8]`, then 2×2-packed to
//!   `[1, Si, 64]` exactly as candle FLUX's `State::new` (so the row-major `img_ids` line up). The
//!   denoised packed latent is `flux::sampling::unpack`ed back to `[1, 16, h/8, w/8]` for the VAE.
//! - **Sigmas**: Chroma's scheduler is `use_dynamic_shifting=false`. HD/Flash use the static-shift
//!   `linspace(1, 1/N, N)` (`σ' = shift·σ/(1+(shift-1)·σ)`); Base uses the beta-spaced schedule
//!   ([`crate::beta`]). NOT FLUX's resolution-dependent exp-shift.
//! - **True CFG**: `pred = neg + g·(pos − neg)`; at `g ≤ 1.0` the negative branch is skipped and
//!   `pred = pos` exactly (`chroma1_flash` is distilled to single-forward), a 2× per-step saving.
//! - **Deterministic seeding (sc-3673)**: initial noise from a fixed-algorithm CPU RNG (`StdRng`,
//!   ChaCha) seeded by `seed`, moved to the device — launch-portable per seed.
//! - **Step-invariants once per step/branch**: the Approximator modulation table (`pooled_temb`,
//!   timestep-only) is computed once per step and shared across both CFG branches; the RoPE table is
//!   built once per branch.
//!
//! Components are loaded at **f32** (the DiT runs f32 activations; the bf16 checkpoint loaded as f32
//! keeps the bf16 weight values — mlx parity) and cached by the generator across `generate` calls.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::{DType, Device, IndexOp, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::sampling::TimestepConvention;
use candle_gen::gen_core::{self, GenerationRequest, Image, Progress};
use candle_gen::{CandleError, Result};
use candle_transformers::models::flux::sampling::unpack;
use candle_transformers::models::t5::T5EncoderModel;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};
use tokenizers::Tokenizer;

use crate::config::{ChromaTransformerConfig, ChromaVariant};
use crate::rope;
use crate::text;
use crate::transformer::ChromaTransformer;
use crate::vae::Vae;

/// The VAE latent channel count (the DiT works on the 2×2-packed 64-ch form).
const LATENT_CHANNELS: usize = 16;

/// A light pipeline handle: the snapshot `root`, variant, and compute device. Heavy components load
/// via [`load_components`](Self::load_components) and are owned/cached by the generator.
pub(crate) struct Pipeline {
    variant: ChromaVariant,
    root: PathBuf,
    device: Device,
}

/// The loaded Chroma components, `Arc`-shared so the generator can cache them across `generate`
/// calls. The T5 encoder is behind a `Mutex` (its `forward` takes `&mut self` for the
/// relative-position-bias cache) — locked only for the once-per-request text encode.
#[derive(Clone)]
pub(crate) struct Components {
    tokenizer: Arc<Tokenizer>,
    t5: Arc<Mutex<T5EncoderModel>>,
    transformer: Arc<ChromaTransformer>,
    vae: Arc<Vae>,
    cfg: ChromaTransformerConfig,
}

impl Pipeline {
    pub(crate) fn load(variant: ChromaVariant, root: &Path, device: &Device) -> Self {
        Self {
            variant,
            root: root.to_path_buf(),
            device: device.clone(),
        }
    }

    /// Load the four heavy components from the Chroma diffusers snapshot (`tokenizer/` vendored,
    /// `text_encoder/` T5, `transformer/` DiT, `vae/` AutoencoderKL), all at f32.
    pub(crate) fn load_components(&self) -> Result<Components> {
        let cfg = ChromaTransformerConfig::default();
        let tokenizer = text::load_tokenizer()?;
        let t5 = text::load_t5(&self.root, &self.device)?;
        let transformer =
            ChromaTransformer::new(cfg, self.f32_vb(&self.root.join("transformer"))?)?;
        let vae = Vae::new(self.f32_vb(&self.root.join("vae"))?)?;
        Ok(Components {
            tokenizer: Arc::new(tokenizer),
            t5: Arc::new(Mutex::new(t5)),
            transformer: Arc::new(transformer),
            vae: Arc::new(vae),
            cfg,
        })
    }

    /// mmap an f32 [`VarBuilder`] over every `.safetensors` in `dir` (the DiT + VAE ship sharded).
    fn f32_vb(&self, dir: &Path) -> Result<VarBuilder<'static>> {
        let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
            .map_err(|e| CandleError::Msg(format!("chroma: read {}: {e}", dir.display())))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        files.sort();
        if files.is_empty() {
            return Err(CandleError::Msg(format!(
                "chroma: no .safetensors found in {} (expected a Chroma diffusers snapshot)",
                dir.display()
            )));
        }
        // SAFETY: mmap of read-only weight files; standard candle loading path.
        Ok(unsafe { VarBuilder::from_mmaped_safetensors(&files, DType::F32, &self.device)? })
    }

    /// Render `req` against pre-loaded `components`, emitting per-step progress and honoring
    /// `req.cancel`. One image per `req.count` (each at seed `base_seed + index`).
    pub(crate) fn render(
        &self,
        req: &GenerationRequest,
        components: &Components,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Vec<Image>> {
        let steps = req
            .steps
            .map(|s| s as usize)
            .unwrap_or(self.variant.default_steps() as usize);
        let guidance = req
            .true_cfg
            .unwrap_or_else(|| self.variant.default_true_cfg());
        let negative = req.negative_prompt.as_deref().unwrap_or("");
        let base_seed = req.seed.unwrap_or_else(gen_core::default_seed);

        let sigmas = self.sigmas(steps);

        // Encode the prompt(s) once for the whole batch (seed- and image-independent). The negative
        // branch is skipped entirely when guidance ≤ 1 (Flash single-forward) — bit-exact `pred = pos`.
        let pos_embeds = self.encode(components, &req.prompt)?;
        let neg = if guidance > 1.0 {
            Some(self.encode(components, negative)?)
        } else {
            None
        };

        let h2 = (req.height as usize).div_ceil(16);
        let w2 = (req.width as usize).div_ceil(16);
        let rope_pos = rope::build_for(&components.cfg, pos_embeds.dim(1)?, h2, w2, &self.device)?;
        let rope_neg = match &neg {
            Some(n) => Some(rope::build_for(
                &components.cfg,
                n.dim(1)?,
                h2,
                w2,
                &self.device,
            )?),
            None => None,
        };

        let mut images = Vec::with_capacity(req.count as usize);
        for index in 0..req.count {
            if req.cancel.is_cancelled() {
                return Err(CandleError::Canceled);
            }
            let seed = image_seed(base_seed, index);
            let latents = self.initial_packed_noise(seed, req.height, req.width)?;
            let latents = self.denoise(
                &components.transformer,
                latents,
                &pos_embeds,
                &rope_pos,
                neg.as_ref(),
                rope_neg.as_ref(),
                &sigmas,
                steps,
                guidance,
                req.sampler.as_deref(),
                req.scheduler.as_deref(),
                seed,
                &req.cancel,
                on_progress,
            )?;
            on_progress(Progress::Decoding);
            images.push(self.decode(&components.vae, &latents, req.height, req.width)?);
        }
        Ok(images)
    }

    /// Encode a prompt to its T5 sequence embedding `[1, L, 4096]` (natural length).
    fn encode(&self, components: &Components, prompt: &str) -> Result<Tensor> {
        let mut t5 = components.t5.lock().expect("chroma T5 mutex poisoned");
        text::encode_prompt(&components.tokenizer, &mut t5, prompt, &self.device)
    }

    /// Chroma's flow-match sigma schedule (length `steps + 1`, descending to a trailing `0`). HD/Flash
    /// use the static-shift `linspace(1, 1/N, N)`; Base uses the beta-spaced schedule.
    fn sigmas(&self, steps: usize) -> Vec<f32> {
        if self.variant.use_beta_sigmas() {
            crate::beta::base_sigmas(steps)
        } else {
            let shift = self.variant.sigma_shift();
            let n = steps.max(1);
            let smax = 1.0f32;
            let smin = 1.0 / n as f32;
            let mut s = Vec::with_capacity(n + 1);
            for i in 0..n {
                let lin = if n == 1 {
                    0.0
                } else {
                    i as f32 / (n - 1) as f32
                };
                let sigma = smax + (smin - smax) * lin; // linspace 1 → 1/N
                s.push(shift * sigma / (1.0 + (shift - 1.0) * sigma));
            }
            s.push(0.0);
            s
        }
    }

    /// sc-3673 deterministic, launch-portable initial noise in candle's get_noise shape, 2×2-packed to
    /// the DiT's `[1, Si, 64]`. N(0,1) from a fixed-algorithm CPU RNG seeded by `seed`.
    fn initial_packed_noise(&self, seed: u64, height: u32, width: u32) -> Result<Tensor> {
        let lat_h = (height as usize).div_ceil(16) * 2; // = h/8 for a multiple-of-16 request
        let lat_w = (width as usize).div_ceil(16) * 2;
        let n = LATENT_CHANNELS * lat_h * lat_w;
        let mut rng = StdRng::seed_from_u64(seed);
        let noise: Vec<f32> = (0..n).map(|_| StandardNormal.sample(&mut rng)).collect();
        let noise = Tensor::from_vec(noise, (1, LATENT_CHANNELS, lat_h, lat_w), &Device::Cpu)?
            .to_device(&self.device)?;
        pack(&noise)
    }

    /// The true-CFG flow-match denoise, routed through the unified curated sampler/scheduler driver
    /// (epic 7114 P4, sc-7123). The `scheduler` axis picks the σ schedule over the variant's static
    /// shift in log space (`mu = shift.ln()`; `native` = the byte-exact per-variant [`Self::sigmas`]),
    /// the `sampler` axis picks the integrator. The DEFAULT (`euler` over the native schedule) is the
    /// N1 no-op — algebraically the legacy flow-match Euler loop `latents += pred·(σ_next − σ_cur)`
    /// within the framework's `to_d` round-trip tolerance.
    ///
    /// Chroma feeds the raw sigma as the DiT timestep ([`TimestepConvention::Sigma`]; the Approximator
    /// scales `·1000` internally) and does true CFG, so the whole CFG blend `pred = neg + g·(pos − neg)`
    /// (or `pred = pos` when `guidance ≤ 1`) lives INSIDE the `predict` closure — a multi-eval solver
    /// re-runs it per eval. Cancellation + progress are driven by the framework.
    #[allow(clippy::too_many_arguments)]
    fn denoise(
        &self,
        transformer: &ChromaTransformer,
        latents: Tensor,
        pos_embeds: &Tensor,
        rope_pos: &rope::RopeTable,
        neg_embeds: Option<&Tensor>,
        rope_neg: Option<&rope::RopeTable>,
        native: &[f32],
        steps: usize,
        guidance: f32,
        sampler: Option<&str>,
        scheduler: Option<&str>,
        seed: u64,
        cancel: &gen_core::CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Tensor> {
        // The scheduler axis rides the variant's static shift in log space (HD shift=3 → ln(3);
        // Flash/Base shift=1 → 0). Base's native beta schedule is returned verbatim on the default
        // path, so `mu` only steers the alternative curated schedulers.
        let mu = self.variant.sigma_shift().ln();
        let sigmas = candle_gen::resolve_flow_schedule(scheduler, mu, steps, native);
        candle_gen::run_flow_sampler(
            sampler,
            TimestepConvention::Sigma,
            &sigmas,
            latents,
            seed,
            cancel,
            on_progress,
            |latents, sigma| -> Result<Tensor> {
                let ts = Tensor::from_vec(vec![sigma], 1, &self.device)?;
                // pooled_temb depends only on the timestep — compute once and share across both branches.
                let pooled = transformer.pooled_temb(&ts)?;
                let pos = transformer.forward_prepared(latents, pos_embeds, &pooled, rope_pos)?;
                match (neg_embeds, rope_neg) {
                    (Some(neg), Some(rope_n)) => {
                        let neg = transformer.forward_prepared(latents, neg, &pooled, rope_n)?;
                        // neg + g·(pos − neg)
                        Ok((&neg + ((&pos - &neg)? * guidance as f64)?)?)
                    }
                    _ => Ok(pos),
                }
            },
        )
    }

    /// Unpack the denoised packed latent `[1, Si, 64]` → `[1, 16, H/8, W/8]`, VAE-decode to an RGB8
    /// [`Image`] (the `[-1, 1]` output mapped to `[0, 255]`).
    fn decode(&self, vae: &Vae, latents: &Tensor, height: u32, width: u32) -> Result<Image> {
        let latents = unpack(latents, height as usize, width as usize)?;
        let decoded = vae.decode(&latents)?.to_dtype(DType::F32)?; // [1, 3, H, W] in [-1, 1]
        let img = ((decoded.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?.to_dtype(DType::U8)?;
        let img = img.i(0)?.to_device(&Device::Cpu)?; // [3, H, W]
        let (c, h, w) = img.dims3()?;
        if c != 3 {
            return Err(CandleError::Msg(format!("expected 3 channels, got {c}")));
        }
        let pixels = img.permute((1, 2, 0))?.flatten_all()?.to_vec1::<u8>()?;
        Ok(Image {
            width: w as u32,
            height: h as u32,
            pixels,
        })
    }
}

/// 2×2 pack `[1, 16, h, w] → [1, h/2·w/2, 64]` — candle FLUX's `State::new` image packing (so the
/// row-major `img_ids` in [`crate::rope`] line up with the packed token order).
fn pack(x: &Tensor) -> Result<Tensor> {
    let (b, c, h, w) = x.dims4()?;
    Ok(
        x.reshape((b, c, h / 2, 2, w / 2, 2))? // (b, c, h, ph, w, pw)
            .permute((0, 2, 4, 1, 3, 5))? // (b, h, w, c, ph, pw)
            .reshape((b, h / 2 * w / 2, c * 4))?,
    )
}

/// Per-image seed within a batch: image `index` renders at `base_seed + index` (wrapping), so the
/// *n*-th image reproduces in isolation at that derived seed (mlx `seed + i`).
pub(crate) fn image_seed(base_seed: u64, index: u32) -> u64 {
    base_seed.wrapping_add(index as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_seed_is_base_plus_index() {
        assert_eq!(image_seed(42, 0), 42);
        assert_eq!(image_seed(42, 7), 49);
        assert_eq!(image_seed(u64::MAX, 1), 0);
    }

    /// HD's static shift moves the interior sigmas but keeps a descending 1→0 schedule of length N+1.
    #[test]
    fn hd_sigmas_descend_with_shift() {
        let pipe = Pipeline::load(ChromaVariant::Hd, Path::new("/x"), &Device::Cpu);
        let s = pipe.sigmas(8);
        assert_eq!(s.len(), 9);
        assert!(
            (s[0] - 1.0).abs() < 1e-6,
            "starts at shift·1/(1+ (shift-1)) = 1: {s:?}"
        );
        assert!(s[8].abs() < 1e-9, "ends at 0: {s:?}");
        for w in s.windows(2) {
            assert!(w[0] > w[1], "must descend: {s:?}");
        }
    }

    /// Flash uses shift 1.0 → the schedule is the raw `linspace(1, 1/N, N)` + trailing 0.
    #[test]
    fn flash_sigmas_are_unshifted_linspace() {
        let pipe = Pipeline::load(ChromaVariant::Flash, Path::new("/x"), &Device::Cpu);
        let s = pipe.sigmas(4);
        // linspace(1, 1/4, 4) = [1, 0.75, 0.5, 0.25], then 0.
        let want = [1.0, 0.75, 0.5, 0.25, 0.0];
        for (g, w) in s.iter().zip(want) {
            assert!((g - w).abs() < 1e-6, "{g} vs {w} in {s:?}");
        }
    }

    /// Base routes through the beta-spaced schedule (distinct from the linspace).
    #[test]
    fn base_sigmas_use_beta_schedule() {
        let pipe = Pipeline::load(ChromaVariant::Base, Path::new("/x"), &Device::Cpu);
        let s = pipe.sigmas(4);
        assert_eq!(s, crate::beta::base_sigmas(4));
        // 0.79344 (beta) ≠ 0.75 (linspace) at index 1.
        assert!((s[1] - 0.75).abs() > 1e-3);
    }

    /// 2×2 pack folds `[1,16,4,4] → [1,4,64]` (Si = (4/2)·(4/2) = 4, 16·4 = 64).
    #[test]
    fn pack_shapes() {
        let x = Tensor::zeros((1, 16, 4, 4), DType::F32, &Device::Cpu).unwrap();
        let p = pack(&x).unwrap();
        assert_eq!(p.dims(), &[1, 4, 64]);
    }
}
