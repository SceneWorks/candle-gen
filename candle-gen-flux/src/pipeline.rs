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
//! **Diffusers-layout tiers — packed Q4/Q8 (sc-9407, sc-9089 umbrella) + dense bf16 (sc-10888).**
//! [`Pipeline::load_components`] auto-detects the snapshot layout ([`Pipeline::uses_diffusers_layout`]).
//! Every **diffusers**-layout tier off the `SceneWorks/flux1-{dev,schnell}-mlx` turnkey — the packed
//! q4/q8 tiers (a `quantization` block in a component's `config.json`, [`Pipeline::component_is_packed`])
//! **and** the dense bf16 tier (a `transformer/` diffusers subdir with no `quantization` block and no
//! root single-file checkpoint) — loads the CLIP + T5 + DiT through the vendored [`crate::packed_dit`] /
//! [`crate::packed_te`] models over the shared [`candle_gen::quant`] packed-**detect**: on q4/q8 each
//! weight loads **straight from the packed parts** (no dense bf16 staging); on bf16 — where no tensor has
//! a `.scales` sibling — the SAME builders load it **dense**, so bf16 is q4/q8 minus the dequant (no
//! separate loader). The VAE dequantizes any packed mid-block attention projection to dense (a no-op on
//! the fully-dense bf16 VAE) and feeds a stock diffusers `AutoEncoderKL`. A dense **BFL single-file**
//! snapshot (root `flux1-*.safetensors`) takes the stock candle-transformers path unchanged. On-the-fly
//! quantization of a dense tier is still NOT done (only the pre-packed tier is a quantized path).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::{DType, Device, IndexOp, Tensor};
use candle_gen::candle_nn::{Module, VarBuilder};
use candle_gen::gen_core::sampling::TimestepConvention;
use candle_gen::gen_core::{self, GenerationRequest, Image, PidWeights, Progress};
// Shared per-image batch seed (`base + index`) — one home in `candle-gen` (sc-9043 / F-059).
use candle_gen::{CandleError, LatentDecoder, Result};
use candle_gen_pid::{PidDecoder, PidEngine};

/// The PiD backbone (latent-space) tag for FLUX (epic 7840 / sc-7853): the `flux` 16-ch latent-space
/// student (4× SR). Shared by Boogu / Chroma / Z-Image, which reuse this FLUX.1 VAE latent space.
const PID_BACKBONE: &str = "flux";
use crate::vae::diffusers::{AutoEncoderKL, VaeConfig};
use crate::vae::native::{AutoEncoder, Config as AeConfig};
use candle_transformers::models::clip::text_model::{
    Activation as ClipActivation, ClipTextConfig, ClipTextTransformer,
};
use candle_transformers::models::flux::model::Config as FluxConfig;
use candle_transformers::models::flux::sampling::{get_schedule, unpack, State};
use candle_transformers::models::t5::T5EncoderModel;
use tokenizers::Tokenizer;

use crate::ip_dit::IpFlux;
use crate::packed_dit::PackedFluxDit;
use crate::packed_te::{ClipConfig, PackedClipText, PackedT5Encoder, T5Config as PackedT5Config};
use crate::Variant;

/// FLUX latent channel count (the VAE's `z_channels` and the DiT's pre-pack channel count). The DiT
/// works on the 2×2-packed form (16·4 = 64 channels), but the raw noise / VAE latent is 16-channel.
const LATENT_CHANNELS: usize = 16;

/// FLUX dev's resolution-dependent flow-match time-shift endpoints (`base_shift`, `max_shift`),
/// matching the candle `flux` example's `get_schedule(.., Some((seq_len, 0.5, 1.15)))` and the
/// diffusers FluxPipeline. schnell uses no shift (`None`). One home for all FLUX lanes (sc-11249 /
/// F-140): the IP-Adapter (`ip_provider`) and PuLID (`candle-gen-pulid`) reference streams share these
/// exact parity-critical endpoints rather than re-declaring them.
pub const BASE_SHIFT: f64 = 0.5;
pub const MAX_SHIFT: f64 = 1.15;

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
///
/// Shared across every FLUX lane (sc-11249 / F-140): the IP-Adapter (`ip_provider`) reaches it as
/// `pub(crate)`; the separate PuLID crate (`candle-gen-pulid`) reaches it re-exported as `pub`. PuLID
/// is always dev, so it calls `flow_mu(Variant::Dev, seq_len)` — the schnell `mu = 0` branch is inert.
pub fn flow_mu(variant: Variant, seq_len: usize) -> f32 {
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
    /// The `LoadSpec::pid` component captured at load (epic 7840 / sc-7853), built into the cached
    /// [`Components`] so the PiD engine loads once alongside the base model. `None` ⇒ native VAE decode.
    pid_spec: Option<PidWeights>,
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
        transformer: Arc<IpFlux>,
        vae: Arc<AutoEncoder>,
        /// T5 + CLIP tokenizers, loaded+parsed **once** at component load and reused across encodes
        /// (sc-8991 / F-011) instead of re-parsing per prompt/branch.
        toks: Arc<FluxTokenizers>,
        /// Optional NVIDIA PiD super-resolving decoder (epic 7840 / sc-7853); None ⇒ native VAE decode.
        pid: Option<Arc<PidEngine>>,
    },
    Packed {
        clip: Arc<PackedClipText>,
        t5: Arc<PackedT5Encoder>,
        transformer: Arc<PackedFluxDit>,
        vae: Arc<AutoEncoderKL>,
        /// T5 + CLIP tokenizers, loaded+parsed **once** at component load (sc-8991 / F-011).
        toks: Arc<FluxTokenizers>,
        /// Optional NVIDIA PiD super-resolving decoder (epic 7840 / sc-7853); None ⇒ native VAE decode.
        pid: Option<Arc<PidEngine>>,
    },
}

/// A borrowed reference to just the DiT of either tier — the only component the denoise loop touches.
/// Lets [`Pipeline::denoise`] be shared by the resident [`render`](Pipeline::render) (which borrows the
/// DiT out of the cached [`Components`]) and the sequential [`render_sequential`](Pipeline::render_sequential)
/// (which owns a just-loaded DiT after the text encoders were dropped). `Copy` — it is two thin refs.
#[derive(Clone, Copy)]
pub(crate) enum DitRef<'a> {
    Stock(&'a IpFlux),
    Packed(&'a PackedFluxDit),
}

/// A just-loaded DiT owned by the sequential path (epic 10765 Phase 1, sc-10769) — not `Arc`-cached,
/// because sequential residency deliberately drops each component after its phase rather than keeping
/// the cross-request cache.
pub(crate) enum LoadedDit {
    Stock(IpFlux),
    Packed(PackedFluxDit),
}

impl LoadedDit {
    fn as_ref(&self) -> DitRef<'_> {
        match self {
            LoadedDit::Stock(dit) => DitRef::Stock(dit),
            LoadedDit::Packed(dit) => DitRef::Packed(dit),
        }
    }
}

/// A just-loaded VAE owned by the sequential path (sc-10769). Same tier split as [`Components`]'s `vae`.
/// Both arms are boxed so the enum isn't dominated by the larger [`AutoEncoder`] (clippy
/// `large_enum_variant`); it holds exactly one VAE for a render, so the extra indirection is free.
pub(crate) enum LoadedVae {
    Stock(Box<AutoEncoder>),
    Packed(Box<AutoEncoderKL>),
}

/// The just-loaded text encoders owned by the sequential path (sc-10769). Held only across the encode
/// phase, then dropped so the ~9 GB T5-XXL frees before the DiT loads (the FLUX sequential-residency
/// win). Mirrors the tier split of [`Components`]; encoding delegates to the SAME shared encode
/// functions the resident path uses (`encode_text` / [`Pipeline::encode_text_packed`]), so tokenization
/// and outputs are byte-identical.
pub(crate) enum SeqTextEncoders {
    Stock {
        clip: ClipTextTransformer,
        t5: Mutex<T5EncoderModel>,
        toks: FluxTokenizers,
    },
    Packed {
        clip: PackedClipText,
        t5: PackedT5Encoder,
        toks: FluxTokenizers,
    },
}

impl Pipeline {
    /// Build the (light) pipeline handle for the FLUX snapshot `root` at the given device/dtype. Does
    /// **no** weight I/O — components load lazily via [`load_components`](Self::load_components).
    pub(crate) fn load(
        variant: Variant,
        root: &Path,
        device: &Device,
        dtype: DType,
        pid_spec: Option<PidWeights>,
    ) -> Self {
        Self {
            variant,
            root: root.to_path_buf(),
            device: device.clone(),
            dtype,
            pid_spec,
        }
    }

    /// Load the optional PiD super-resolving decoder once (epic 7840 / sc-7853) when the caller opted in
    /// via `LoadSpec::pid`; FLUX's own `flux` latent-space student. `None` ⇒ native VAE. Shared by both
    /// the stock and packed component-build paths.
    fn load_pid(&self) -> Result<Option<Arc<PidEngine>>> {
        Ok(match self.pid_spec.as_ref() {
            Some(spec) => Some(Arc::new(PidEngine::from_spec(
                spec,
                PID_BACKBONE,
                &self.device,
            )?)),
            None => None,
        })
    }

    /// Load the four heavy components from the snapshot, auto-detecting the layout
    /// ([`Self::uses_diffusers_layout`]). Every **diffusers-layout** tier off the
    /// `SceneWorks/flux1-{dev,schnell}-mlx` turnkey — packed q4/q8 **and** dense bf16 (sc-10888) — loads
    /// through the vendored diffusers component builders ([`Self::load_diffusers_components`]); a dense
    /// **BFL** single-file snapshot (black-forest-labs `FLUX.1-{dev,schnell}`) takes the stock
    /// `candle-transformers` path unchanged (sc-3694).
    pub(crate) fn load_components(&self) -> Result<Components> {
        if self.uses_diffusers_layout()? {
            return self.load_diffusers_components();
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
        // guidance embedding (dev embeds the guidance scale, schnell does not). Built as the vendored
        // `IpFlux` (with `ip = None`, byte-identical to the stock `candle-transformers` `Flux::forward`)
        // so the txt2img path picks up the sc-9116 i32-overflow-safe budgeted attention — the stock
        // upstream `Flux` materializes an unguarded `[…,S,S]` scores tensor that overflows i32 at 2048²
        // (S ≈ 16.9k joint tokens → `24·16.9k² ≈ 6.8e9 > i32::MAX`). The checkpoint layout is identical.
        let dit_vb =
            crate::flux1_load::dit_vb(&self.root, self.variant, self.dtype, &self.device, "flux")?;
        let transformer = IpFlux::new(&flux_config(self.variant), dit_vb)?;

        // FLUX AutoEncoder (`ae.safetensors`) at the root.
        let (vae, _vae_vb) =
            crate::flux1_load::vae(&self.root, self.variant, self.dtype, &self.device, "flux")?;

        Ok(Components::Stock {
            clip: Arc::new(clip),
            t5: Arc::new(Mutex::new(t5)),
            transformer: Arc::new(transformer),
            vae: Arc::new(vae),
            toks: Arc::new(FluxTokenizers::load(&self.root)?),
            pid: self.load_pid()?,
        })
    }

    /// The **diffusers-layout** component path — packed q4/q8 (sc-9407) **and** dense bf16 (sc-10888)
    /// off the same `SceneWorks/flux1-{dev,schnell}-mlx` turnkey. The vendored packed-**detect**
    /// [`PackedClipText`] / [`PackedT5Encoder`] / [`PackedFluxDit`] build every projection through the
    /// shared [`QLinear::linear_detect`](candle_gen::quant::QLinear::linear_detect) /
    /// [`QEmbedding::detect`](crate::quant::QEmbedding): on a packed tier they load straight from the
    /// packed parts (q4 → `Q4_1` lossless, q8 → `Q8_0` requant — no dense staging); on the dense bf16
    /// tier — where no tensor carries a `.scales` sibling — the SAME builders load each weight **dense**,
    /// so bf16 is q4/q8 minus the dequant (no separate loader). The diffusers `AutoEncoderKL` VAE is fed
    /// [`Self::vae_vb_dequantized`], which dequantizes any packed mid-block attention projection to dense
    /// and passes every already-dense tensor through (a no-op on the fully-dense bf16 VAE).
    fn load_diffusers_components(&self) -> Result<Components> {
        // CLIP-L (diffusers `text_encoder/model.safetensors`, `text_model.` prefix). On the packed tiers
        // every projection + the token/position embeddings are packed; on bf16 they load dense. The
        // LayerNorms stay dense in every tier.
        let clip_vb = self.component_vb("text_encoder")?;
        let clip = PackedClipText::new(&ClipConfig::flux(), clip_vb.pp("text_model"))?;

        // T5-XXL encoder (diffusers `text_encoder_2/`; single-file on the packed tiers, sharded on bf16).
        // `shared` + every block projection + block 0's `relative_attention_bias` are packed on q4/q8,
        // dense on bf16.
        let t5_vb = self.component_vb("text_encoder_2")?;
        let t5 = PackedT5Encoder::new(&PackedT5Config::xxl(), t5_vb)?;

        // FLUX diffusers DiT (`FluxTransformer2DModel`): 19 double + 38 single blocks, every Linear packed
        // (q4/q8) or dense (bf16). The block counts come from the component `config.json` (defaulting to
        // FLUX.1's 19/38).
        let (num_double, num_single) = self.dit_block_counts()?;
        let dit_vb = self.component_vb("transformer")?;
        let transformer =
            PackedFluxDit::new(&flux_config(self.variant), num_double, num_single, dit_vb)?;

        // Diffusers `AutoEncoderKL` (identical config to z-image's VAE — 16 latent ch, [128,256,512,512],
        // scaling 0.3611 / shift 0.1159). On q4/q8 the 8 packed mid-block attention projections dequantize
        // to dense; on bf16 nothing is packed, so every tensor passes through dense.
        let vae = AutoEncoderKL::new(&flux_vae_config(), self.vae_vb_dequantized()?)?;

        Ok(Components::Packed {
            clip: Arc::new(clip),
            t5: Arc::new(t5),
            transformer: Arc::new(transformer),
            vae: Arc::new(vae),
            toks: Arc::new(FluxTokenizers::load(&self.root)?),
            pid: self.load_pid()?,
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

    /// Whether this snapshot uses the **diffusers component layout** (`transformer/`, `text_encoder*/`,
    /// `vae/` subdirs) rather than the black-forest-labs single-file layout — i.e. whether to build the
    /// vendored diffusers components ([`load_diffusers_components`](Self::load_diffusers_components)) or
    /// the stock BFL [`IpFlux`] / `AutoEncoder` ([`load_stock_components`](Self::load_stock_components)).
    ///
    /// Three shapes off the `SceneWorks/flux1-{dev,schnell}-mlx` turnkey resolve here:
    /// - **packed q4/q8** — a `quantization` block in `transformer/config.json`
    ///   ([`Self::component_is_packed`]) ⇒ diffusers layout (`true`); the builders load straight from the
    ///   packed parts.
    /// - **dense bf16** (sc-10888) — no `quantization` block and **no** root BFL single-file DiT
    ///   checkpoint, but a `transformer/` diffusers subdir (sharded `diffusion_pytorch_model-*`) ⇒
    ///   diffusers layout (`true`). The SAME builders load it dense: the shared packed-detect reads each
    ///   tensor dense when its `.scales` sibling is absent, so bf16 is q4/q8 minus the dequant.
    /// - dense **BFL single-file** (root `flux1-{dev,schnell}.safetensors`) ⇒ NOT diffusers layout
    ///   (`false`); the stock candle-transformers path (byte-exact, unchanged; sc-3694 / sc-10769).
    ///
    /// The BFL root checkpoint is checked FIRST for the non-packed case so a **full** black-forest-labs
    /// snapshot — which ships BOTH the root single-file AND the diffusers subdirs — keeps the proven
    /// stock path rather than switching layouts.
    pub(crate) fn uses_diffusers_layout(&self) -> Result<bool> {
        if self.component_is_packed("transformer")? {
            return Ok(true);
        }
        // Dense diffusers tier (bf16): the root BFL single-file DiT is absent but the diffusers
        // `transformer/` subdir is present. (A BFL snapshot whose root checkpoint is present stays stock.)
        Ok(!self.has_bfl_dit_checkpoint() && self.root.join("transformer").is_dir())
    }

    /// Whether the root black-forest-labs single-file DiT checkpoint
    /// (`flux1-{dev,schnell}.safetensors`) is present — the discriminator between a dense BFL snapshot
    /// (stock path) and a dense diffusers snapshot (the sc-10888 bf16 tier).
    fn has_bfl_dit_checkpoint(&self) -> bool {
        self.root.join(self.variant.transformer_file()).is_file()
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

        // Resolve the decode seam once for the whole batch (epic 7840 / sc-7853): a per-generation PiD
        // decoder bound to this prompt when `req.use_pid` is set (errors if requested but not loaded),
        // else `None` → the native FLUX VAE decode. Shared across `count` images (same prompt); the PiD
        // engine lives in whichever `Components` arm loaded.
        let pid_engine = match components {
            Components::Stock { pid, .. } => pid.as_deref(),
            Components::Packed { pid, .. } => pid.as_deref(),
        };
        let pid_decoder = candle_gen_pid::resolve_pid_decoder(
            pid_engine,
            req,
            base_seed,
            self.variant.model_id(),
        )?;

        candle_gen::for_each_image_seed(base_seed, req.count, |seed| {
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

            // Borrow just the DiT out of the cached components for the shared denoise loop.
            let dit = match components {
                Components::Stock { transformer, .. } => DitRef::Stock(transformer),
                Components::Packed { transformer, .. } => DitRef::Packed(transformer),
            };
            let latents =
                self.denoise(dit, &state, &timesteps, guidance, seed, req, on_progress)?;

            on_progress(Progress::Decoding);
            match components {
                Components::Stock { vae, .. } => self.decode(
                    vae,
                    pid_decoder.as_ref(),
                    &latents,
                    req.height as usize,
                    req.width as usize,
                ),
                Components::Packed { vae, .. } => self.decode_packed(
                    vae,
                    pid_decoder.as_ref(),
                    &latents,
                    req.height as usize,
                    req.width as usize,
                ),
            }
        })
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
        dit: DitRef,
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
                let out = match dit {
                    // `IpFlux::forward` with `ip = None` is byte-identical to the stock
                    // `candle-transformers` `Flux::forward` (sc-9116) — plus the budgeted attention guard.
                    DitRef::Stock(transformer) => transformer.forward(
                        img,
                        &state.img_ids,
                        &state.txt,
                        &state.txt_ids,
                        &t_vec,
                        &state.vec,
                        Some(&guidance_t),
                        None,
                    )?,
                    DitRef::Packed(transformer) => transformer.forward(
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
        pid: Option<&PidDecoder>,
        latents: &Tensor,
        height: usize,
        width: usize,
    ) -> Result<Image> {
        decode_latents(vae, pid, latents, height, width)
    }

    /// Decode the packed tier's denoised latents. Unpack from the DiT token form `(1, h·w, 64)` back to
    /// the NCHW latent `(1, 16, H/8, W/8)`, then either the diffusers `AutoEncoderKL` (which applies
    /// `(z / scaling) + shift` inside `decode`) or — when a PiD decoder resolved (epic 7840 / sc-7853) —
    /// the super-resolving `flux`-student, which consumes the SAME unpacked latent the VAE receives (a
    /// zero-transform seam) and emits a larger `[1,3,4H,4W]` tensor. Both yield `[-1, 1]` pixels;
    /// [`to_image`] reads the size from the tensor (never `latent*8`).
    fn decode_packed(
        &self,
        vae: &AutoEncoderKL,
        pid: Option<&PidDecoder>,
        latents: &Tensor,
        height: usize,
        width: usize,
    ) -> Result<Image> {
        let latents = unpack(latents, height, width)?;
        let decoded = match pid {
            Some(pid) => pid.decode(&latents)?,
            None => vae.decode(&latents)?.to_dtype(DType::F32)?, // (1, 3, H, W) in [-1, 1]
        };
        to_image(&decoded)
    }

    /// Tier-dispatching decode for a caller that owns its OWN [`PidDecoder`] (the reference lanes via
    /// [`crate::ref_backbone::FluxRefBackbone`]) rather than the components-embedded PiD engine: routes
    /// the denoised latents through the stock [`AutoEncoder`] or the packed [`AutoEncoderKL`] exactly as
    /// [`render`](Self::render), but takes the decoder explicitly (PuLID / the IP-adapter build their PiD
    /// decoder separately from the backbone). `pid = None` ⇒ the native VAE decode.
    pub(crate) fn decode_ref(
        &self,
        components: &Components,
        latents: &Tensor,
        height: usize,
        width: usize,
        pid: Option<&PidDecoder>,
    ) -> Result<Image> {
        match components {
            Components::Stock { vae, .. } => self.decode(vae, pid, latents, height, width),
            Components::Packed { vae, .. } => self.decode_packed(vae, pid, latents, height, width),
        }
    }

    /// Load ONLY the text encoders for the sequential-residency path (epic 10765 Phase 1, sc-10769) —
    /// dropped right after the encode so the ~9 GB T5-XXL frees before the DiT loads. Same per-layout
    /// loads as [`load_stock_components`](Self::load_stock_components) /
    /// [`load_diffusers_components`](Self::load_diffusers_components), minus the DiT/VAE/PiD. `diffusers`
    /// selects the vendored diffusers builders (packed q4/q8 or dense bf16) over the stock BFL encoders.
    fn load_text_encoders_seq(&self, diffusers: bool) -> Result<SeqTextEncoders> {
        if diffusers {
            let clip_vb = self.component_vb("text_encoder")?;
            let clip = PackedClipText::new(&ClipConfig::flux(), clip_vb.pp("text_model"))?;
            let t5_vb = self.component_vb("text_encoder_2")?;
            let t5 = PackedT5Encoder::new(&PackedT5Config::xxl(), t5_vb)?;
            Ok(SeqTextEncoders::Packed {
                clip,
                t5,
                toks: FluxTokenizers::load(&self.root)?,
            })
        } else {
            let (clip, t5) =
                crate::flux1_load::text_encoders(&self.root, self.dtype, &self.device, "flux")?;
            Ok(SeqTextEncoders::Stock {
                clip,
                t5: Mutex::new(t5),
                toks: FluxTokenizers::load(&self.root)?,
            })
        }
    }

    /// Encode `prompt` through the sequential text encoders, delegating to the SAME shared encode path
    /// as the resident tier ([`encode_text`] / [`encode_text_packed`](Self::encode_text_packed)) so the
    /// tokenization + conditioning tensors are byte-identical to [`render`](Self::render).
    fn encode_seq(&self, tes: &SeqTextEncoders, prompt: &str) -> Result<(Tensor, Tensor)> {
        match tes {
            SeqTextEncoders::Stock { clip, t5, toks } => encode_text(
                self.variant,
                toks,
                &self.device,
                self.dtype,
                clip,
                t5,
                prompt,
            ),
            SeqTextEncoders::Packed { clip, t5, toks } => {
                self.encode_text_packed(clip, t5, toks, prompt)
            }
        }
    }

    /// Load ONLY the DiT for the sequential path (sc-10769) — loaded after the text encoders were
    /// dropped, so it reuses their freed allocator pool (capping peak at DiT+VAE, not TE+DiT+VAE).
    fn load_transformer_seq(&self, diffusers: bool) -> Result<LoadedDit> {
        if diffusers {
            let (num_double, num_single) = self.dit_block_counts()?;
            let dit_vb = self.component_vb("transformer")?;
            Ok(LoadedDit::Packed(PackedFluxDit::new(
                &flux_config(self.variant),
                num_double,
                num_single,
                dit_vb,
            )?))
        } else {
            let dit_vb = crate::flux1_load::dit_vb(
                &self.root,
                self.variant,
                self.dtype,
                &self.device,
                "flux",
            )?;
            Ok(LoadedDit::Stock(IpFlux::new(
                &flux_config(self.variant),
                dit_vb,
            )?))
        }
    }

    /// Load ONLY the VAE for the sequential path (sc-10769). Small relative to the DiT, so it stays
    /// co-resident with the DiT through decode (splitting them further buys ~nothing on FLUX).
    fn load_vae_seq(&self, diffusers: bool) -> Result<LoadedVae> {
        if diffusers {
            Ok(LoadedVae::Packed(Box::new(AutoEncoderKL::new(
                &flux_vae_config(),
                self.vae_vb_dequantized()?,
            )?)))
        } else {
            let (vae, _vae_vb) =
                crate::flux1_load::vae(&self.root, self.variant, self.dtype, &self.device, "flux")?;
            Ok(LoadedVae::Stock(Box::new(vae)))
        }
    }

    /// Sequential-residency render (epic 10765 Phase 1, sc-10769): load the text encoders → encode →
    /// DROP them → load the DiT + VAE → denoise/decode. Peak VRAM is bounded to the DiT+VAE working set
    /// instead of TE+DiT+VAE (reclaiming the ~9 GB T5-XXL on FLUX), so a card that OOMs the resident
    /// path can still render. Output is **bit-identical** to [`render`](Self::render) — the SAME encode,
    /// denoise, and decode code runs (`encode_seq` → [`encode_text`], the shared [`denoise`](Self::denoise)
    /// over a [`DitRef`], and [`decode`](Self::decode)/[`decode_packed`](Self::decode_packed)); only the
    /// load/free schedule differs.
    ///
    /// Selected by the generator when [`sequential_offload_enabled`] (`CANDLE_GEN_OFFLOAD=sequential`).
    /// Because it drops components, it does NOT populate the generator's `Components` cache — repeat
    /// requests reload from the (page-cached) snapshot; that reload cost is the deliberate trade for the
    /// lower peak, which is why it is opt-in per the fit-gate rather than the default.
    pub(crate) fn render_sequential(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Vec<Image>> {
        let diffusers = self.uses_diffusers_layout()?;
        let steps = req
            .steps
            .map(|s| s as usize)
            .unwrap_or(self.variant.default_steps() as usize);
        let base_seed = req.seed.unwrap_or_else(gen_core::default_seed);
        let guidance: f64 = if self.variant.supports_guidance() {
            req.guidance.unwrap_or(self.variant.default_guidance()) as f64
        } else {
            0.0
        };
        let lat_h = (req.height as usize).div_ceil(16) * 2;
        let lat_w = (req.width as usize).div_ceil(16) * 2;

        // Phase 1 — text encode, then DROP the encoders (scoped) so T5-XXL frees before the DiT loads.
        let (t5_emb, clip_emb) = {
            let tes = self.load_text_encoders_seq(diffusers)?;
            self.encode_seq(&tes, &req.prompt)?
        };

        // Phase 2 — load the DiT (reusing the encoders' freed pool) + the VAE + the optional PiD decoder.
        let dit = self.load_transformer_seq(diffusers)?;
        let vae = self.load_vae_seq(diffusers)?;
        let pid_engine = self.load_pid()?;
        let pid_decoder = candle_gen_pid::resolve_pid_decoder(
            pid_engine.as_deref(),
            req,
            base_seed,
            self.variant.model_id(),
        )?;

        // Phase 3 — per-image denoise + decode, identical to `render`'s loop.
        candle_gen::for_each_image_seed(base_seed, req.count, |seed| {
            let noise = crate::flux1_load::seeded_noise(
                seed,
                LATENT_CHANNELS,
                lat_h,
                lat_w,
                &self.device,
                self.dtype,
            )?;
            let state = State::new(&t5_emb, &clip_emb, &noise)?;
            let timesteps = if self.variant.is_dev() {
                get_schedule(steps, Some((state.img.dim(1)?, BASE_SHIFT, MAX_SHIFT)))
            } else {
                get_schedule(steps, None)
            };
            let latents = self.denoise(
                dit.as_ref(),
                &state,
                &timesteps,
                guidance,
                seed,
                req,
                on_progress,
            )?;
            on_progress(Progress::Decoding);
            match &vae {
                LoadedVae::Stock(vae) => self.decode(
                    vae,
                    pid_decoder.as_ref(),
                    &latents,
                    req.height as usize,
                    req.width as usize,
                ),
                LoadedVae::Packed(vae) => self.decode_packed(
                    vae,
                    pid_decoder.as_ref(),
                    &latents,
                    req.height as usize,
                    req.width as usize,
                ),
            }
        })
    }
}

/// Whether the sequential-residency offload path is enabled (epic 10765 Phase 1, sc-10769). Reads
/// `CANDLE_GEN_OFFLOAD`: `sequential` (case-insensitive) selects the phased load/free path; unset or any
/// other value keeps the resident, cross-request-cached default. The worker's fit-gate sets this when it
/// predicts the resident TE+DiT+VAE sum won't fit but the DiT+VAE working set will (Slice 1b wires a
/// per-load `LoadSpec::offload_policy`; this env toggle is the first-slice mechanism + the test seam).
pub(crate) fn sequential_offload_enabled() -> bool {
    std::env::var("CANDLE_GEN_OFFLOAD")
        .map(|value| value.trim().eq_ignore_ascii_case("sequential"))
        .unwrap_or(false)
}

/// Convert a decoded pixel tensor `(1, 3, H, W)` in `[-1, 1]` (f32) → RGB8 [`Image`] (`(x+1)·127.5`).
/// Shared by the native VAE decode and the PiD super-resolving decode; the output size is read from the
/// tensor, never assumed (PiD may be 4× the VAE-native size).
fn to_image(decoded: &Tensor) -> Result<Image> {
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

/// The diffusers `AutoEncoderKL` config for the FLUX packed VAE — identical to z-image's (16 latent
/// channels, `[128, 256, 512, 512]`, layers 2, scaling 0.3611 / shift 0.1159, norm groups 32). The
/// live path now decodes the FLUX packed VAE through the vendored `crate::vae::diffusers::AutoEncoderKL`.
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
        let mut t5 = candle_gen::lock_recover(t5);
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

/// Unpack the denoised latents `(1, h·w, 64)` back to `(1, 16, H/8, W/8)` and decode to an RGB8
/// [`Image`]. Shared by the txt2img [`Pipeline::decode`] and the IP-Adapter provider (which passes
/// `pid = None`). The native path uses the FLUX `AutoEncoder` (its own `(z / scale) + shift` un-scale is
/// applied inside `decode`); when a PiD decoder resolved (epic 7840 / sc-7853) the super-resolving
/// `flux`-student consumes the SAME unpacked latent the VAE receives (a zero-transform seam) and emits a
/// larger `[1,3,4H,4W]` tensor. Both yield `[-1, 1]` pixels; [`to_image`] reads the size from the tensor.
pub fn decode_latents(
    vae: &AutoEncoder,
    pid: Option<&PidDecoder>,
    latents: &Tensor,
    height: usize,
    width: usize,
) -> Result<Image> {
    let latents = unpack(latents, height, width)?;
    let decoded = match pid {
        Some(pid) => pid.decode(&latents)?,
        None => vae.decode(&latents)?.to_dtype(DType::F32)?, // (1, 3, H, W) in [-1, 1]
    };
    to_image(&decoded)
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

        let pipe = Pipeline::load(Variant::Schnell, &tmp, &Device::Cpu, DType::F32, None);
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

    /// `uses_diffusers_layout` routes ALL diffusers-layout tiers (packed q4/q8 AND dense bf16) to the
    /// vendored diffusers builders, and only a BFL single-file snapshot to the stock path (sc-10888). The
    /// dense bf16 tier (`transformer/config.json` with no `quantization` block + no root
    /// `flux1-dev.safetensors`) is the regression this guards: it used to fall to the stock BFL branch and
    /// die on a missing `flux1-dev.safetensors`. A **full** BFL snapshot (root single-file AND diffusers
    /// subdirs) must still choose stock — the root checkpoint is checked first. GPU-free (small JSON/empty
    /// files).
    #[test]
    fn uses_diffusers_layout_distinguishes_all_three_tiers() -> Result<()> {
        let base = std::env::temp_dir().join(format!("sc10888_layout_{}", std::process::id()));
        let mk_transformer_config = |dir: &Path, body: &str| -> Result<()> {
            std::fs::create_dir_all(dir.join("transformer")).ok();
            std::fs::write(dir.join("transformer").join("config.json"), body)
                .map_err(|e| CandleError::Msg(e.to_string()))
        };

        // (1) Packed q4/q8 diffusers tier — `quantization` block ⇒ diffusers layout.
        let packed = base.join("packed");
        mk_transformer_config(
            &packed,
            r#"{ "num_layers": 19, "quantization": { "bits": 4, "group_size": 64 } }"#,
        )?;
        let pipe = Pipeline::load(Variant::Dev, &packed, &Device::Cpu, DType::F32, None);
        assert!(
            pipe.uses_diffusers_layout()?,
            "packed q4/q8 must use the diffusers builders"
        );

        // (2) Dense bf16 diffusers tier — no `quantization` block, no root BFL checkpoint, but a
        // `transformer/` subdir ⇒ diffusers layout (the sc-10888 fix). This shape previously fell to the
        // stock path and failed on a missing `flux1-dev.safetensors`.
        let bf16 = base.join("bf16");
        mk_transformer_config(&bf16, r#"{ "num_layers": 19, "num_single_layers": 38 }"#)?;
        let pipe = Pipeline::load(Variant::Dev, &bf16, &Device::Cpu, DType::F32, None);
        assert!(
            !pipe.has_bfl_dit_checkpoint(),
            "the bf16 diffusers tier has no root single-file checkpoint"
        );
        assert!(
            pipe.uses_diffusers_layout()?,
            "dense bf16 diffusers tier must use the diffusers builders (sc-10888)"
        );

        // (3) Dense BFL single-file snapshot — root `flux1-dev.safetensors` present, no diffusers
        // `transformer/` ⇒ NOT diffusers layout (stock path).
        let bfl = base.join("bfl");
        std::fs::create_dir_all(&bfl).ok();
        std::fs::write(bfl.join(Variant::Dev.transformer_file()), b"")
            .map_err(|e| CandleError::Msg(e.to_string()))?;
        let pipe = Pipeline::load(Variant::Dev, &bfl, &Device::Cpu, DType::F32, None);
        assert!(
            !pipe.uses_diffusers_layout()?,
            "a BFL single-file snapshot must take the stock path"
        );

        // (4) FULL BFL snapshot — ships BOTH the root single-file AND the diffusers subdirs. The
        // root-checkpoint-first check must keep it on the proven stock path, not switch layouts.
        let full = base.join("full_bfl");
        mk_transformer_config(&full, r#"{ "num_layers": 19, "num_single_layers": 38 }"#)?;
        std::fs::write(full.join(Variant::Dev.transformer_file()), b"")
            .map_err(|e| CandleError::Msg(e.to_string()))?;
        let pipe = Pipeline::load(Variant::Dev, &full, &Device::Cpu, DType::F32, None);
        assert!(
            !pipe.uses_diffusers_layout()?,
            "a full BFL snapshot (root checkpoint present) must stay stock even with diffusers subdirs"
        );

        std::fs::remove_dir_all(&base).ok();
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
