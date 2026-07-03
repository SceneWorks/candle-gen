//! The candle FLUX.1 **txt2img** pipeline (sc-3694) — the `candle-transformers` `flux` reference
//! model (dual CLIP-L + T5-XXL text encoders → FLUX DiT, flow-match Euler → FLUX AutoEncoder VAE)
//! driven through the backend-neutral [`gen_core::Generator`] contract, parity-matched to the macOS
//! `mlx-gen-flux` provider for both the `flux1_schnell` (distilled, 4-step, no guidance) and
//! `flux1_dev` (guidance-distilled, 25-step, guidance ~3.5) variants.
//!
//! What this wires, and the deliberate parity choices (grounded in the candle `flux` example and the
//! mlx provider's `config.rs`/`loader.rs`/`model.rs`):
//!
//! - **Weight layout — the clean split**: a black-forest-labs FLUX snapshot ships *both* the original
//!   single-file checkpoints at the root (`flux1-{schnell,dev}.safetensors`, `ae.safetensors`) *and*
//!   the diffusers component subdirs. candle's [`flux::model::Flux`] / [`flux::autoencoder::AutoEncoder`]
//!   are written against the **original BFL key layout**, so the DiT + VAE load directly from the root
//!   files (no diffusers→BFL key remap needed — the part mlx had to hand-write). The two text encoders
//!   come from the diffusers subdirs: CLIP-L from `text_encoder/` and T5-XXL from `text_encoder_2/`.
//! - **Dual text encoders**: candle's [`clip::text_model::ClipTextTransformer`] returns the **pooled**
//!   `(1, 768)` vector (argmax-at-EOT over a causal stack — FLUX's `vec`/`y` conditioning), and
//!   [`t5::T5EncoderModel`] returns the `(1, L, 4096)` **sequence** (FLUX's `txt`). T5 is padded to the
//!   variant's max length (**256** schnell / **512** dev, matching the diffusers FluxPipeline default)
//!   with the T5 pad id 0; every padded token is attended (FLUX applies no T5 attention mask), so the
//!   length is parity-critical.
//! - **CLIP tokenizer is vendored** (sc-2787 parity): the FLUX snapshot ships CLIP only as
//!   `vocab.json` + `merges.txt` (no `tokenizer.json`), and a byte-level BPE built from those
//!   mis-tokenizes CLIP's lowercased word-BPE — silently corrupting the pooled conditioning. So the
//!   HF-faithful `clip_tokenizer.json` is **compiled into the crate** (`assets/`, the same asset the
//!   mlx provider vendors) and never reconstructed from the snapshot. T5 ships a real
//!   `tokenizer_2/tokenizer.json`, which is used directly.
//! - **Flow-match schedule**: schnell uses the linear `get_schedule(steps, None)`; dev uses the
//!   resolution-dependent time-shifted `get_schedule(steps, Some((seq_len, 0.5, 1.15)))`. The denoise
//!   is candle's own additive Euler update `img = img + pred·(t_prev − t_curr)` over **descending**
//!   timesteps (1→0) — the FLUX sign convention is baked into the descending step, so unlike Z-Image
//!   there is **no velocity negation** and no separate `mu` scheduler gotcha (the shift lives inside
//!   `get_schedule`). Guidance is passed as a per-batch tensor and only *used* when the DiT config has
//!   `guidance_embed` (dev); schnell's DiT ignores it.
//! - **Deterministic seeding (sc-3673 parity)**: initial latent noise is drawn from a fixed-algorithm
//!   CPU RNG (`StdRng`, ChaCha) seeded by `seed` and moved to the device — NOT candle's CUDA
//!   `flux::sampling::get_noise` (`Tensor::randn`), whose seed→noise mapping is not launch-portable.
//!   The flow-match Euler step injects no per-step noise, so generation is a pure function of
//!   `(seed, request)` — what the gen-core-testkit seed-determinism check (sc-4481) requires.
//! - **Contract surface**: progress is `on_progress(Progress::Step/Decoding)`, cancellation is
//!   `req.cancel` → typed [`gen_core::Error::Canceled`], and each image is returned as a
//!   `gen_core::Image` (RGB8) — the worker owns asset writes.
//!
//! **First-slice surface (sc-3694), matching the SDXL/Z-Image slices:** txt2img only. img2img
//! (mlx's `Reference`/IP-adapter) and LoRA/LoKr are NOT wired here — they are rejected loudly (the
//! worker routes them to the Python fallback) rather than silently dropped.
//!
//! **Packed Q4/Q8 tiers (sc-9407, sc-9089 umbrella).** [`Pipeline::load_components`] auto-detects a
//! pre-quantized MLX-packed **diffusers**-layout tier (`SceneWorks/flux1-schnell-mlx` q4/q8) by the
//! `quantization` block in a component's `config.json` ([`Pipeline::component_is_packed`]) and loads the
//! CLIP + T5 + DiT **straight from the packed parts** through the shared [`candle_gen::quant`]
//! packed-detect (the vendored [`crate::packed_dit`] / [`crate::packed_te`] models) — no dense bf16
//! staging. The VAE dequantizes its 8 packed mid-block attention projections to dense and feeds a stock
//! diffusers `AutoEncoderKL`. A dense **BFL** snapshot (no `quantization` block) takes the stock path
//! unchanged. On-the-fly quantization of a dense tier is still NOT done (only the pre-packed tier is a
//! quantized path).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::{DType, Device, IndexOp, Tensor};
use candle_gen::candle_nn::{Module, VarBuilder};
use candle_gen::gen_core::sampling::TimestepConvention;
use candle_gen::gen_core::{self, GenerationRequest, Image, Progress};
// Shared per-image batch seed (`base + index`) — one home in `candle-gen` (sc-9043 / F-059).
use candle_gen::image_seed;
use candle_gen::{CandleError, Result};
use candle_transformers::models::clip::text_model::{
    Activation as ClipActivation, ClipTextConfig, ClipTextTransformer,
};
use candle_transformers::models::flux::autoencoder::{AutoEncoder, Config as AeConfig};
use candle_transformers::models::flux::model::{Config as FluxConfig, Flux};
use candle_transformers::models::flux::sampling::{get_schedule, unpack, State};
use candle_transformers::models::flux::WithForward;
use candle_transformers::models::t5::T5EncoderModel;
use candle_transformers::models::z_image::vae::{AutoEncoderKL, VaeConfig};
use tokenizers::Tokenizer;

use crate::packed_dit::PackedFluxDit;
use crate::packed_te::{ClipConfig, PackedClipText, PackedT5Encoder, T5Config as PackedT5Config};
use crate::Variant;

/// FLUX latent channel count (the VAE's `z_channels` and the DiT's pre-pack channel count). The DiT
/// works on the 2×2-packed form (16·4 = 64 channels), but the raw noise / VAE latent is 16-channel.
const LATENT_CHANNELS: usize = 16;

/// FLUX dev's resolution-dependent flow-match time-shift endpoints (`base_shift`, `max_shift`),
/// matching the candle `flux` example's `get_schedule(.., Some((seq_len, 0.5, 1.15)))` and the
/// diffusers FluxPipeline. schnell uses no shift (`None`).
const BASE_SHIFT: f64 = 0.5;
const MAX_SHIFT: f64 = 1.15;

/// T5 pad token id (`<pad>`) — FLUX pads the T5 sequence to the variant max length with this id, and
/// attends every padded position (no attention mask), so it is parity-relevant.
const T5_PAD_TOKEN_ID: u32 = 0;

/// The flow-match time-shift `mu` for the unified scheduler axis (epic 7114 P4, sc-7123). It mirrors
/// candle's `get_schedule(.., Some((seq_len, BASE_SHIFT, MAX_SHIFT)))` linear shift:
/// `mu = m·seq_len + b` with `m = (MAX_SHIFT − BASE_SHIFT)/(4096 − 256)`, `b = BASE_SHIFT − m·256`,
/// so gen-core's exponential time-shift (`time_shift(mu,1,v) = e/(e + (1/v − 1))`) lands on the SAME
/// shift the native schedule uses. schnell applies no shift (`get_schedule(.., None)`), so `mu = 0`.
/// Used ONLY to feed the curated `resolve_flow_schedule`; the native (default) schedule stays the
/// verbatim `get_schedule(..)` so the N1 default path is byte-exact.
pub(crate) fn flow_mu(variant: Variant, seq_len: usize) -> f32 {
    if !variant.is_dev() {
        return 0.0;
    }
    let m = (MAX_SHIFT - BASE_SHIFT) / (4096.0 - 256.0);
    let b = BASE_SHIFT - m * 256.0;
    (m * seq_len as f64 + b) as f32
}

/// A txt2img pipeline handle: the snapshot `root`, the variant, and the compute device/dtype (bf16).
/// Loading the heavy components is done by [`load_components`](Self::load_components) and owned/cached
/// by the generator, mirroring the SDXL/Z-Image providers' lazy split.
pub(crate) struct Pipeline {
    variant: Variant,
    root: PathBuf,
    device: Device,
    dtype: DType,
}

/// The loaded FLUX components, `Arc`-shared so the generator can cache them across `generate` calls
/// and cheaply clone them out for a render. Two shapes:
///
/// - [`Components::Stock`] — the dense **BFL**-layout black-forest-labs snapshot: the stock
///   `candle-transformers` CLIP / T5 / `Flux` DiT / `AutoEncoder` VAE, reading the original single-file
///   `flux1-*.safetensors` + `ae.safetensors` (path unchanged, sc-3694).
/// - [`Components::Packed`] — the pre-quantized **diffusers**-layout MLX tier
///   (`SceneWorks/flux1-schnell-mlx` q4/q8): the vendored packed-detect [`PackedClipText`] /
///   [`PackedT5Encoder`] / [`PackedFluxDit`] built straight from the packed parts (sc-9407, no dense
///   staging), + a stock diffusers `AutoEncoderKL` fed the dequantized-to-dense VAE weights.
///
/// The stock T5 encoder is behind a `Mutex` because its `forward` takes `&mut self` (position-bias
/// cache) while `Generator::generate` is `&self`; the packed T5 forward is `&self`, so no lock is
/// needed there. Cloning an enum arm clones the inner `Arc`s (cheap).
#[derive(Clone)]
pub(crate) enum Components {
    Stock {
        clip: Arc<ClipTextTransformer>,
        t5: Arc<Mutex<T5EncoderModel>>,
        transformer: Arc<Flux>,
        vae: Arc<AutoEncoder>,
        /// T5 + CLIP tokenizers, loaded+parsed **once** at component load and reused across encodes
        /// (sc-8991 / F-011) instead of re-parsing per prompt/branch.
        toks: Arc<FluxTokenizers>,
    },
    Packed {
        clip: Arc<PackedClipText>,
        t5: Arc<PackedT5Encoder>,
        transformer: Arc<PackedFluxDit>,
        vae: Arc<AutoEncoderKL>,
        /// T5 + CLIP tokenizers, loaded+parsed **once** at component load (sc-8991 / F-011).
        toks: Arc<FluxTokenizers>,
    },
}

impl Pipeline {
    /// Build the (light) pipeline handle for the FLUX snapshot `root` at the given device/dtype. Does
    /// **no** weight I/O — components load lazily via [`load_components`](Self::load_components).
    pub(crate) fn load(variant: Variant, root: &Path, device: &Device, dtype: DType) -> Self {
        Self {
            variant,
            root: root.to_path_buf(),
            device: device.clone(),
            dtype,
        }
    }

    /// Load the four heavy components from the snapshot, auto-detecting the tier. A pre-quantized
    /// **MLX-packed** diffusers tier (`SceneWorks/flux1-schnell-mlx` q4/q8) carries a `quantization`
    /// block in each component's `config.json` ([`Self::component_is_packed`]) — on detection the CLIP /
    /// T5 / DiT / VAE load **straight from the packed parts** (sc-9407, no dense bf16 staging). A dense
    /// **BFL** snapshot (black-forest-labs `FLUX.1-schnell`, no `quantization` block) takes the stock
    /// path unchanged (sc-3694).
    pub(crate) fn load_components(&self) -> Result<Components> {
        if self.component_is_packed("transformer")? {
            return self.load_packed_components();
        }
        self.load_stock_components()
    }

    /// The dense BFL-layout path (sc-3694, unchanged): CLIP-L from `text_encoder/`, T5-XXL from
    /// `text_encoder_2/`, the DiT from the root `flux1-*.safetensors` and the VAE from `ae.safetensors`.
    /// The text-encoder / DiT-mmap / VAE loads now come from the shared FLUX.1 backbone loader
    /// (sc-9003 / F-023) — the CLIP `text_model.` prefix, the T5 config parse, and the noise geometry no
    /// longer drift across the three FLUX.1 providers. This path builds the **stock**
    /// `candle-transformers` `Flux` DiT (the per-provider drift: the providers build the forked `IpFlux`).
    fn load_stock_components(&self) -> Result<Components> {
        // CLIP-L + T5-XXL text encoders (shared FLUX.1 backbone load).
        let (clip, t5) =
            crate::flux1_load::text_encoders(&self.root, self.dtype, &self.device, "flux")?;

        // FLUX DiT (original BFL checkpoint) at the snapshot root; config differs only by the
        // guidance embedding (dev embeds the guidance scale, schnell does not). The stock `Flux` over the
        // shared DiT mmap.
        let dit_vb =
            crate::flux1_load::dit_vb(&self.root, self.variant, self.dtype, &self.device, "flux")?;
        let transformer = Flux::new(&flux_config(self.variant), dit_vb)?;

        // FLUX AutoEncoder (`ae.safetensors`) at the root.
        let (vae, _vae_vb) =
            crate::flux1_load::vae(&self.root, self.variant, self.dtype, &self.device, "flux")?;

        Ok(Components::Stock {
            clip: Arc::new(clip),
            t5: Arc::new(Mutex::new(t5)),
            transformer: Arc::new(transformer),
            vae: Arc::new(vae),
            toks: Arc::new(FluxTokenizers::load(&self.root)?),
        })
    }

    /// The pre-quantized MLX-packed diffusers-layout path (sc-9407): the vendored packed-detect
    /// [`PackedClipText`] / [`PackedT5Encoder`] / [`PackedFluxDit`] load straight from the packed parts
    /// (q4 → `Q4_1` lossless, q8 → `Q8_0` requant — no dense staging); the diffusers `AutoEncoderKL` VAE
    /// is fed the 8 dequantized-to-dense attention projections (the rest of the VAE is already dense).
    fn load_packed_components(&self) -> Result<Components> {
        // CLIP-L (diffusers `text_encoder/model.safetensors`, `text_model.` prefix). Every projection +
        // the token/position embeddings are packed; the LayerNorms stay dense.
        let clip_vb = self.component_vb("text_encoder")?;
        let clip = PackedClipText::new(&ClipConfig::flux(), clip_vb.pp("text_model"))?;

        // T5-XXL encoder (diffusers `text_encoder_2/`, single-file in the packed tier). `shared` + every
        // block projection + block 0's `relative_attention_bias` are packed.
        let t5_vb = self.component_vb("text_encoder_2")?;
        let t5 = PackedT5Encoder::new(&PackedT5Config::xxl(), t5_vb)?;

        // FLUX diffusers DiT (`FluxTransformer2DModel`): 19 double + 38 single blocks, every Linear
        // packed. The block counts come from the component `config.json` (defaulting to FLUX.1's 19/38).
        let (num_double, num_single) = self.dit_block_counts()?;
        let dit_vb = self.component_vb("transformer")?;
        let transformer =
            PackedFluxDit::new(&flux_config(self.variant), num_double, num_single, dit_vb)?;

        // Diffusers `AutoEncoderKL` (identical config to z-image's VAE — 16 latent ch, [128,256,512,512],
        // scaling 0.3611 / shift 0.1159). The 8 packed mid-block attention projections dequantize to
        // dense; everything else is already dense bf16.
        let vae = AutoEncoderKL::new(&flux_vae_config(), self.vae_vb_dequantized()?)?;

        Ok(Components::Packed {
            clip: Arc::new(clip),
            t5: Arc::new(t5),
            transformer: Arc::new(transformer),
            vae: Arc::new(vae),
            toks: Arc::new(FluxTokenizers::load(&self.root)?),
        })
    }

    /// Whether the snapshot component `sub/` is a **pre-quantized MLX-packed tier** — its `config.json`
    /// carries a `quantization` block ([`candle_gen::quant::PackedConfig`]) that the install-time convert
    /// job writes. Mirrors z-image/flux2's `component_is_packed`.
    ///
    /// A **genuinely-absent** `config.json` (file NotFound) is a legitimate dense BFL snapshot shape →
    /// `Ok(false)` (the dense path), so a BFL snapshot (which has no `transformer/config.json`) loads
    /// stock. A config that **is present but corrupt** (I/O error or malformed JSON — e.g. a partial
    /// download) errors loudly naming the file rather than silently downgrading a packed component to the
    /// dense path (wrong tier / missing weights, no diagnostic). A well-formed config with no
    /// `quantization` block is a dense tier → `Ok(false)` (sc-9426, F-073 sibling).
    pub(crate) fn component_is_packed(&self, sub: &str) -> Result<bool> {
        let path = self.root.join(sub).join("config.json");
        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            // No config.json at all → legitimate dense BFL / fixture snapshot, not packed.
            Err(ref e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(false),
            // Present but unreadable (permissions, partial download) → surface, don't swallow.
            Err(e) => {
                return Err(CandleError::Msg(format!(
                    "flux: read {}: {e}",
                    path.display()
                )))
            }
        };
        // Present but malformed JSON → corrupt snapshot, error rather than fall to dense.
        let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| {
            CandleError::Msg(format!(
                "flux: parse {} (corrupt snapshot?): {e}",
                path.display()
            ))
        })?;
        Ok(candle_gen::quant::PackedConfig::from_config(&v).is_some())
    }

    /// The DiT double / single block counts from the packed `transformer/config.json`
    /// (`num_layers` / `num_single_layers`), defaulting to FLUX.1's 19 / 38 when absent.
    fn dit_block_counts(&self) -> Result<(usize, usize)> {
        let path = self.root.join("transformer").join("config.json");
        let (mut num_double, mut num_single) = (19usize, 38usize);
        if let Ok(s) = std::fs::read_to_string(&path) {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) {
                if let Some(n) = v.get("num_layers").and_then(|x| x.as_u64()) {
                    num_double = n as usize;
                }
                if let Some(n) = v.get("num_single_layers").and_then(|x| x.as_u64()) {
                    num_single = n as usize;
                }
            }
        }
        Ok((num_double, num_single))
    }

    /// Sorted `.safetensors` in the snapshot component subdir `sub` (single-file or sharded).
    fn component_files(&self, sub: &str) -> Result<Vec<PathBuf>> {
        let dir = self.root.join(sub);
        self.safetensors_in(&dir)
    }

    /// mmap a [`VarBuilder`] over every `.safetensors` in the snapshot component subdir `sub`.
    fn component_vb(&self, sub: &str) -> Result<VarBuilder<'static>> {
        let files = self.component_files(sub)?;
        candle_gen::mmap_var_builder(&files, self.dtype, &self.device)
    }

    /// Build a VAE [`VarBuilder`] for a packed tier by dequantizing the 8 packed mid-block attention
    /// projections (`{encoder,decoder}.mid_block.attentions.0.{to_q,to_k,to_v,to_out.0}`) to dense and
    /// passing every other (already-dense) tensor through unchanged — so the stock diffusers
    /// `AutoEncoderKL` never sees a `.weight` u32/`.scales`/`.biases` triple it can't read (sc-9407, the
    /// z-image VAE path).
    fn vae_vb_dequantized(&self) -> Result<VarBuilder<'static>> {
        use candle_gen::candle_core::safetensors::MmapedSafetensors;
        let files = self.component_files("vae")?;
        // SAFETY: mmap of read-only weight files; standard candle loading path.
        let st = unsafe { MmapedSafetensors::multi(&files)? };
        let src = VarBuilder::from_backend(Box::new(st), self.dtype, self.device.clone());

        // SAFETY: same file set; a second mapping to enumerate keys + load the dense tensors.
        let st2 = unsafe { MmapedSafetensors::multi(&files)? };
        let packed_bases: std::collections::HashSet<String> = st2
            .tensors()
            .iter()
            .filter_map(|(k, _)| k.strip_suffix(".scales").map(|b| b.to_string()))
            .collect();
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        for (key, _) in st2.tensors() {
            if key.ends_with(".scales") || key.ends_with(".biases") {
                continue; // folded into the dequantized dense `.weight`
            }
            if let Some(base) = key.strip_suffix(".weight") {
                if packed_bases.contains(base) {
                    let dense = crate::quant::dequant_packed_to_dense(
                        &src,
                        base,
                        &self.device,
                        self.dtype,
                    )?;
                    tensors.insert(key.clone(), dense);
                    continue;
                }
            }
            let t = st2.load(&key, &self.device)?;
            tensors.insert(key.clone(), t.to_dtype(self.dtype)?);
        }
        Ok(VarBuilder::from_tensors(tensors, self.dtype, &self.device))
    }

    /// Sorted list of every `.safetensors` in `dir` (sharded T5 checkpoints ship as
    /// `model-0000n-of-0000m.safetensors`). Errors if none are found.
    fn safetensors_in(&self, dir: &Path) -> Result<Vec<PathBuf>> {
        candle_gen::sorted_safetensors(dir, "flux")
    }

    /// Encode `prompt` into FLUX's two conditioning tensors: the T5 sequence `(1, L, 4096)` and the
    /// CLIP pooled vector `(1, 768)`, both at the compute dtype. T5 is tokenized with the snapshot's
    /// `tokenizer_2/tokenizer.json` (padded to the variant max length with id 0); CLIP with the
    /// vendored `clip_tokenizer.json` (natural length — the pooled vector is the EOT hidden state, so
    /// trailing pad would not change it under CLIP's causal attention, and is omitted to match the
    /// candle reference exactly).
    pub(crate) fn text_embeddings(
        &self,
        comps: &Components,
        prompt: &str,
    ) -> Result<(Tensor, Tensor)> {
        match comps {
            Components::Stock { clip, t5, toks, .. } => encode_text(
                self.variant,
                toks,
                &self.device,
                self.dtype,
                clip,
                t5,
                prompt,
            ),
            Components::Packed { clip, t5, toks, .. } => {
                self.encode_text_packed(clip, t5, toks, prompt)
            }
        }
    }

    /// Encode `prompt` for the packed tier: the vendored [`PackedT5Encoder`] sequence + the
    /// [`PackedClipText`] pooled vector. The tokenizers are the same two the stock path uses (T5 from
    /// the snapshot `tokenizer_2/`, CLIP vendored), padded identically, so the only difference is which
    /// model runs the ids — parity with `encode_text`. `toks` is the cached [`FluxTokenizers`] (sc-8991
    /// / F-011).
    fn encode_text_packed(
        &self,
        clip: &PackedClipText,
        t5: &PackedT5Encoder,
        toks: &FluxTokenizers,
        prompt: &str,
    ) -> Result<(Tensor, Tensor)> {
        // T5 sequence — same tokenizer + padding as `encode_text`.
        let mut t5_ids: Vec<u32> = toks
            .t5
            .encode(prompt, true)
            .map_err(|e| CandleError::Msg(format!("flux: T5 tokenize: {e}")))?
            .get_ids()
            .to_vec();
        t5_ids.resize(self.variant.t5_max_len(), T5_PAD_TOKEN_ID);
        let t5_input = Tensor::new(t5_ids.as_slice(), &self.device)?.unsqueeze(0)?;
        let t5_emb = t5.forward(&t5_input, self.dtype)?;

        // CLIP pooled vector — vendored tokenizer, natural length (EOT pool).
        let clip_ids: Vec<u32> = toks
            .clip
            .encode(prompt, true)
            .map_err(|e| CandleError::Msg(format!("flux: CLIP tokenize: {e}")))?
            .get_ids()
            .to_vec();
        if clip_ids.is_empty() {
            return Err(CandleError::Msg("flux: empty CLIP tokenization".into()));
        }
        let clip_input = Tensor::new(clip_ids.as_slice(), &self.device)?.unsqueeze(0)?;
        let clip_emb = clip.forward(&clip_input)?.to_dtype(self.dtype)?;
        Ok((t5_emb, clip_emb))
    }

    /// Render `req` against pre-loaded `components`, emitting per-step progress and honoring
    /// `req.cancel`. Returns one `gen_core::Image` per `req.count` (each with seed `base_seed + index`).
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
        let base_seed = req.seed.unwrap_or_else(gen_core::default_seed);
        // Guidance is only consumed by the dev DiT (`guidance_embed`); schnell's DiT ignores the
        // tensor, so 0.0 there is inert. Validation rejects a guidance request on schnell already.
        let guidance: f64 = if self.variant.supports_guidance() {
            req.guidance.unwrap_or(self.variant.default_guidance()) as f64
        } else {
            0.0
        };

        // candle's get_noise geometry: the latent is padded to `div_ceil(16)*2` per side (== /8 for a
        // multiple-of-16 request) — i.e. the VAE's /8 latent. We enforce the /16 alignment in `validate`.
        let lat_h = (req.height as usize).div_ceil(16) * 2;
        let lat_w = (req.width as usize).div_ceil(16) * 2;

        // Text embeddings are seed- and image-independent: encode once for the whole batch.
        let (t5_emb, clip_emb) = self.text_embeddings(components, &req.prompt)?;

        let mut images = Vec::with_capacity(req.count as usize);
        for index in 0..req.count {
            let seed = image_seed(base_seed, index);

            // sc-3673 parity — deterministic, launch-portable initial noise in candle's get_noise
            // shape (1, 16, h/8, w/8): N(0,1) from a fixed-algorithm CPU RNG seeded by `seed` (shared
            // FLUX.1 helper, sc-9003).
            let noise = crate::flux1_load::seeded_noise(
                seed,
                LATENT_CHANNELS,
                lat_h,
                lat_w,
                &self.device,
                self.dtype,
            )?;

            // Pack noise + build the conditioning state (img/img_ids/txt/txt_ids/vec) exactly as the
            // candle reference — shared by both tiers. The packed token count drives dev's
            // resolution-dependent time-shift.
            let state = State::new(&t5_emb, &clip_emb, &noise)?;
            let timesteps = if self.variant.is_dev() {
                get_schedule(steps, Some((state.img.dim(1)?, BASE_SHIFT, MAX_SHIFT)))
            } else {
                get_schedule(steps, None)
            };

            let latents = self.denoise(
                components,
                &state,
                &timesteps,
                guidance,
                seed,
                req,
                on_progress,
            )?;

            on_progress(Progress::Decoding);
            let image = match components {
                Components::Stock { vae, .. } => {
                    self.decode(vae, &latents, req.height as usize, req.width as usize)?
                }
                Components::Packed { vae, .. } => {
                    self.decode_packed(vae, &latents, req.height as usize, req.width as usize)?
                }
            };
            images.push(image);
        }
        Ok(images)
    }

    /// The flow-match denoise, routed through the unified curated sampler/scheduler driver (epic 7114
    /// P4, sc-7123). The `scheduler` axis (`req.scheduler`) picks where the σ steps land over FLUX's
    /// time-shift `mu` (`native` = the verbatim `get_schedule(..)` schedule); the `sampler` axis
    /// (`req.sampler`) picks the integrator. The DEFAULT (`sampler`/`scheduler` = `None`) is the N1
    /// no-op: `euler` over the native schedule is algebraically the legacy inline flow-match Euler loop
    /// `img += pred·(σ_{i+1} − σ_i)` within the driver's `to_d` round-trip tolerance, so default output
    /// stays parity-matched to the candle reference. FLUX feeds the raw timestep (`Sigma` convention:
    /// the model sees `t == σ` directly, NOT `t·1000`); guidance is a per-batch tensor only embedded by
    /// the dev DiT. Cancellation + progress are owned by the driver; the per-step DiT forward (and the
    /// guidance embed) live inside the `predict` closure, so a multi-eval solver re-runs the whole step.
    #[allow(clippy::too_many_arguments)]
    fn denoise(
        &self,
        components: &Components,
        state: &State,
        timesteps: &[f64],
        guidance: f64,
        seed: u64,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Tensor> {
        let b_sz = state.img.dim(0)?;
        let dev = &self.device;
        let guidance_t = Tensor::full(guidance as f32, b_sz, dev)?;
        // The native schedule is candle's verbatim `get_schedule(..)` (the byte-exact N1 default), in
        // f32 descending with a trailing 0.0; the curated `scheduler` axis re-strides it over `mu`.
        let native: Vec<f32> = timesteps.iter().map(|&t| t as f32).collect();
        let mu = flow_mu(self.variant, state.img.dim(1)?);
        let steps = native.len().saturating_sub(1);
        let sigmas =
            candle_gen::resolve_flow_schedule(req.scheduler.as_deref(), mu, steps, &native);
        // Guidance is only consumed by the dev DiT; schnell's DiT ignores the tensor. The packed DiT
        // takes the same shape (`Option<&Tensor>` — `None` for schnell, since `guidance_embed` is off).
        let packed_guidance = if self.variant.supports_guidance() {
            Some(&guidance_t)
        } else {
            None
        };
        candle_gen::run_flow_sampler(
            req.sampler.as_deref(),
            TimestepConvention::Sigma,
            &sigmas,
            state.img.clone(),
            seed,
            &req.cancel,
            on_progress,
            |img, t| -> Result<Tensor> {
                // The model is fed the raw timestep (`t == σ`) as a per-batch tensor. The forward
                // returns a `candle_core::Result`; `?` bridges it into the driver's `CandleError`.
                let t_vec = Tensor::full(t, b_sz, dev)?;
                let out = match components {
                    Components::Stock { transformer, .. } => transformer.forward(
                        img,
                        &state.img_ids,
                        &state.txt,
                        &state.txt_ids,
                        &t_vec,
                        &state.vec,
                        Some(&guidance_t),
                    )?,
                    Components::Packed { transformer, .. } => transformer.forward(
                        img,
                        &state.img_ids,
                        &state.txt,
                        &state.txt_ids,
                        &t_vec,
                        &state.vec,
                        packed_guidance,
                    )?,
                };
                Ok(out)
            },
        )
    }

    /// Unpack the denoised latents `(1, h·w, 64)` back to `(1, 16, H/8, W/8)`, VAE-decode to an RGB8
    /// [`Image`]. The AutoEncoder applies its own `(z / scale) + shift` un-scale inside `decode`; the
    /// `[-1, 1]` output is mapped to `[0, 255]` u8.
    fn decode(
        &self,
        vae: &AutoEncoder,
        latents: &Tensor,
        height: usize,
        width: usize,
    ) -> Result<Image> {
        decode_latents(vae, latents, height, width)
    }

    /// Decode the packed tier's denoised latents through the diffusers `AutoEncoderKL` (which applies
    /// `(z / scaling) + shift` inside `decode`). The latents are first unpacked from the DiT token form
    /// `(1, h·w, 64)` back to the NCHW latent `(1, 16, H/8, W/8)` the VAE expects (the same `unpack` the
    /// BFL path uses), then decoded to `[-1, 1]` and mapped to RGB8.
    fn decode_packed(
        &self,
        vae: &AutoEncoderKL,
        latents: &Tensor,
        height: usize,
        width: usize,
    ) -> Result<Image> {
        let latents = unpack(latents, height, width)?;
        let decoded = vae.decode(&latents)?.to_dtype(DType::F32)?; // (1, 3, H, W) in [-1, 1]
        let img = ((decoded.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?.to_dtype(DType::U8)?;
        let img = img.i(0)?.to_device(&Device::Cpu)?;
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

/// The diffusers `AutoEncoderKL` config for the FLUX packed VAE — identical to z-image's (16 latent
/// channels, `[128, 256, 512, 512]`, layers 2, scaling 0.3611 / shift 0.1159, norm groups 32), so the
/// shared `candle_transformers::models::z_image::vae` decoder loads the FLUX packed VAE directly.
fn flux_vae_config() -> VaeConfig {
    VaeConfig::z_image()
}

/// The vendored CLIP tokenizer JSON (bundled at compile time). Shared by the stock and packed encode
/// paths — parsing it once (in [`FluxTokenizers::load`]) instead of per-encode is part of the sc-8991 /
/// F-011 fix.
const CLIP_TOKENIZER_JSON: &[u8] = include_bytes!("../assets/clip_tokenizer.json");

/// FLUX's two prompt tokenizers — the disk-loaded T5 (`tokenizer_2/tokenizer.json`) and the vendored
/// CLIP — loaded+parsed **once** and cached on the caller's `Components` / provider struct, reused
/// across every prompt/branch encode (sc-8991 / F-011) rather than re-parsing per request. Same files +
/// same parse as the old per-encode load, so the token ids are byte-identical.
pub struct FluxTokenizers {
    t5: Tokenizer,
    clip: Tokenizer,
}

impl FluxTokenizers {
    /// Load both tokenizers from the snapshot `root` (T5 from `tokenizer_2/`, CLIP from the vendored
    /// bytes). Call once at component load.
    pub fn load(root: &Path) -> Result<Self> {
        let t5 = Tokenizer::from_file(root.join("tokenizer_2/tokenizer.json"))
            .map_err(|e| CandleError::Msg(format!("flux: load T5 tokenizer: {e}")))?;
        let clip = Tokenizer::from_bytes(CLIP_TOKENIZER_JSON)
            .map_err(|e| CandleError::Msg(format!("flux: load vendored CLIP tokenizer: {e}")))?;
        Ok(Self { t5, clip })
    }
}

/// Encode `prompt` into FLUX's two conditioning tensors for `variant`: the T5 sequence `(1, L, 4096)`
/// and the CLIP pooled vector `(1, 768)`, both at `dtype`. Shared by the txt2img
/// [`Pipeline::text_embeddings`] and the IP-Adapter provider ([`crate::ip_provider`]) so the two never
/// drift on the parity-critical tokenization (T5 padded to the variant length; the vendored CLIP
/// tokenizer). `toks` is the cached [`FluxTokenizers`] (sc-8991 / F-011). `t5` is locked only for the
/// once-per-request encode.
pub fn encode_text(
    variant: Variant,
    toks: &FluxTokenizers,
    device: &Device,
    dtype: DType,
    clip: &ClipTextTransformer,
    t5: &Mutex<T5EncoderModel>,
    prompt: &str,
) -> Result<(Tensor, Tensor)> {
    // T5 sequence.
    let mut t5_ids: Vec<u32> = toks
        .t5
        .encode(prompt, true)
        .map_err(|e| CandleError::Msg(format!("flux: T5 tokenize: {e}")))?
        .get_ids()
        .to_vec();
    // Pad/truncate to the variant's fixed T5 length (256 schnell / 512 dev). FLUX attends every
    // position (no T5 mask), so the padded length is parity-critical, not a perf knob.
    t5_ids.resize(variant.t5_max_len(), T5_PAD_TOKEN_ID);
    let t5_input = Tensor::new(t5_ids.as_slice(), device)?.unsqueeze(0)?;
    let t5_emb = {
        let mut t5 = t5.lock().expect("flux T5 mutex poisoned");
        t5.forward(&t5_input)?
    }
    .to_dtype(dtype)?;

    // CLIP pooled vector.
    let clip_ids: Vec<u32> = toks
        .clip
        .encode(prompt, true)
        .map_err(|e| CandleError::Msg(format!("flux: CLIP tokenize: {e}")))?
        .get_ids()
        .to_vec();
    if clip_ids.is_empty() {
        return Err(CandleError::Msg("flux: empty CLIP tokenization".into()));
    }
    let clip_input = Tensor::new(clip_ids.as_slice(), device)?.unsqueeze(0)?;
    let clip_emb = clip.forward(&clip_input)?.to_dtype(dtype)?;

    Ok((t5_emb, clip_emb))
}

/// Unpack the denoised latents `(1, h·w, 64)` back to `(1, 16, H/8, W/8)`, VAE-decode to an RGB8
/// [`Image`]. Shared by the txt2img [`Pipeline::decode`] and the IP-Adapter provider. The AutoEncoder
/// applies its own `(z / scale) + shift` un-scale inside `decode`; the `[-1, 1]` output is mapped to
/// `[0, 255]` u8.
pub fn decode_latents(
    vae: &AutoEncoder,
    latents: &Tensor,
    height: usize,
    width: usize,
) -> Result<Image> {
    let latents = unpack(latents, height, width)?;
    let decoded = vae.decode(&latents)?.to_dtype(DType::F32)?; // (1, 3, H, W) in [-1, 1]
    let img = ((decoded.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?.to_dtype(DType::U8)?;
    let img = img.i(0)?.to_device(&Device::Cpu)?;
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

/// The fixed CLIP-L (openai/clip-vit-large-patch14) text config FLUX uses — identical across
/// schnell/dev. Mirrors the candle `flux` example's hardcoded `ClipTextConfig`.
pub fn clip_config() -> ClipTextConfig {
    ClipTextConfig {
        vocab_size: 49408,
        projection_dim: 768,
        activation: ClipActivation::QuickGelu,
        intermediate_size: 3072,
        embed_dim: 768,
        max_position_embeddings: 77,
        pad_with: None,
        num_hidden_layers: 12,
        num_attention_heads: 12,
    }
}

/// The FLUX DiT config for `variant` — schnell and dev differ only in `guidance_embed`.
pub fn flux_config(variant: Variant) -> FluxConfig {
    if variant.is_dev() {
        FluxConfig::dev()
    } else {
        FluxConfig::schnell()
    }
}

/// The FLUX AutoEncoder config for `variant` (the scale/shift factors are identical across variants;
/// the variant arm mirrors the candle example's per-model selection).
pub fn ae_config(variant: Variant) -> AeConfig {
    if variant.is_dev() {
        AeConfig::dev()
    } else {
        AeConfig::schnell()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `component_is_packed` detects the `quantization` block a packed MLX tier writes into a component
    /// `config.json` (a diffusers packed tier) but not a dense one — this is the toggle that routes
    /// `load_components` to the packed vs stock path. A *present-but-corrupt* `config.json` (malformed
    /// JSON, e.g. a partial download) errors loudly naming the file rather than silently falling to the
    /// dense path (sc-9426, F-073 sibling). GPU-free (writes/reads a small JSON file).
    #[test]
    fn component_is_packed_detects_quantization_block() -> Result<()> {
        let tmp = std::env::temp_dir().join(format!("sc9407_pkg_{}", std::process::id()));
        let packed_dir = tmp.join("transformer");
        let dense_dir = tmp.join("vae");
        std::fs::create_dir_all(&packed_dir).ok();
        std::fs::create_dir_all(&dense_dir).ok();
        std::fs::write(
            packed_dir.join("config.json"),
            r#"{ "num_layers": 19, "quantization": { "bits": 4, "group_size": 64 } }"#,
        )
        .map_err(|e| CandleError::Msg(e.to_string()))?;
        std::fs::write(
            dense_dir.join("config.json"),
            r#"{ "latent_channels": 16 }"#,
        )
        .map_err(|e| CandleError::Msg(e.to_string()))?;

        let pipe = Pipeline::load(Variant::Schnell, &tmp, &Device::Cpu, DType::F32);
        assert!(
            pipe.component_is_packed("transformer")?,
            "`quantization` block ⇒ packed"
        );
        assert!(
            !pipe.component_is_packed("vae")?,
            "no `quantization` block ⇒ dense"
        );
        assert!(
            !pipe.component_is_packed("missing")?,
            "absent component ⇒ dense (no panic)"
        );

        // A config.json that is *present but corrupt* (malformed JSON) must error naming the file, NOT
        // silently downgrade the packed component to the dense path (sc-9426 / F-073 sibling).
        let corrupt_dir = tmp.join("transformer_bad");
        std::fs::create_dir_all(&corrupt_dir).ok();
        std::fs::write(corrupt_dir.join("config.json"), b"{ not json")
            .map_err(|e| CandleError::Msg(e.to_string()))?;
        let err = pipe
            .component_is_packed("transformer_bad")
            .expect_err("corrupt config.json must error, not fall to dense");
        assert!(
            format!("{err}").contains("config.json"),
            "the error should name the offending file, got: {err}"
        );

        std::fs::remove_dir_all(&tmp).ok();
        Ok(())
    }

    /// Parity anchors against `mlx-gen-flux`: distilled step defaults (4 schnell / 25 dev), guidance
    /// support (dev only) + the 3.5 dev default, and the T5 max lengths (256 / 512). GPU-free.
    #[test]
    fn variant_defaults_match_mlx_provider() {
        assert_eq!(Variant::Schnell.default_steps(), 4);
        assert_eq!(Variant::Dev.default_steps(), 25);
        assert!(!Variant::Schnell.supports_guidance());
        assert!(Variant::Dev.supports_guidance());
        assert_eq!(Variant::Dev.default_guidance(), 3.5);
        assert_eq!(Variant::Schnell.t5_max_len(), 256);
        assert_eq!(Variant::Dev.t5_max_len(), 512);
        assert_eq!(LATENT_CHANNELS, 16);
    }

    /// The DiT config tracks the variant only through `guidance_embed`: dev embeds the guidance scale,
    /// schnell does not. The rest of the FLUX config is shared. GPU-free.
    #[test]
    fn flux_config_guidance_embed_tracks_variant() {
        assert!(flux_config(Variant::Dev).guidance_embed);
        assert!(!flux_config(Variant::Schnell).guidance_embed);
    }

    /// schnell uses an unshifted linear schedule; dev applies the resolution-dependent time-shift.
    /// Both produce `num_steps + 1` timesteps descending from 1 to 0 (the flow-match prior). The
    /// descending order is what makes the additive Euler update walk noise→data without a negation.
    #[test]
    fn schedule_is_descending_and_shift_tracks_variant() {
        let schnell = get_schedule(4, None);
        assert_eq!(schnell.len(), 5);
        assert!((schnell[0] - 1.0).abs() < 1e-9, "starts at 1: {schnell:?}");
        assert!(schnell[4].abs() < 1e-9, "ends at 0: {schnell:?}");
        for w in schnell.windows(2) {
            assert!(w[0] > w[1], "must descend: {schnell:?}");
        }
        // dev's time-shift moves the interior timesteps but keeps the 1→0 endpoints and monotonicity.
        let dev = get_schedule(25, Some((4096, BASE_SHIFT, MAX_SHIFT)));
        assert_eq!(dev.len(), 26);
        assert!((dev[0] - 1.0).abs() < 1e-9);
        assert!(dev[25].abs() < 1e-9);
        for w in dev.windows(2) {
            assert!(w[0] > w[1], "dev schedule must descend: {dev:?}");
        }
        // The shift actually changes the schedule (interior points differ from linear).
        let dev_linear = get_schedule(25, None);
        assert!(
            dev.iter()
                .zip(&dev_linear)
                .any(|(a, b)| (a - b).abs() > 1e-6),
            "dev time-shift should differ from the linear schedule"
        );
    }
}
