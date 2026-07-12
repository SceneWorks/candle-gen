//! The candle Z-Image **txt2img** pipeline (sc-3693) — the `candle-transformers` `z_image`
//! reference model (Qwen3 text encoder → DiT transformer → AutoencoderKL VAE, flow-match Euler)
//! driven through the backend-neutral [`gen_core::Generator`] contract, parity-matched to the
//! macOS `mlx-gen-z-image` provider.
//!
//! What this wires, and the deliberate parity choices (all grounded in the mlx provider's
//! `model.rs`/`pipeline.rs` and Z-Image's `scheduler_config.json`):
//!
//! - **Components**: the three `candle-transformers::models::z_image` modules — `ZImageTextEncoder`
//!   (Qwen3, hidden 2560, 36 layers; returns the second-to-last hidden state, no final norm),
//!   `ZImageTransformer2DModel` (the DiT, 16-channel latent, patch 2), and `AutoEncoderKL`
//!   (diffusers VAE, /8 spatial, scaling 0.3611 / shift 0.1159 applied **inside** `decode`). Loaded
//!   at **bf16** — Z-Image is a bf16 model (unlike the fp16 SDXL family), and candle's CUDA backend
//!   runs bf16 natively.
//! - **Prompt → cap_feats**: the Qwen chat-template wrapping + host-vec tokenization come from
//!   gen-core's [`TextTokenizer`] with [`ChatTemplate::QwenInstruct`] — the *exact* template the mlx
//!   provider uses ([`crate`] docs). This is the epic-3692 "carries over via gen-core" reuse: the
//!   parity-critical template is written once in gen-core, not re-derived here. The encoder output is
//!   padded to the DiT's `SEQ_MULTI_OF` with an attention mask by the reference `prepare_inputs`.
//! - **Distilled schedule (no CFG)**: Z-Image-Turbo is guidance-distilled — a fixed **4-step**
//!   flow-match Euler schedule, no classifier-free guidance and no negative prompt. The DiT is fed
//!   the **1−σ** timestep convention and its predicted velocity is **negated** before the Euler step
//!   (Z-Image sign convention). The scheduler is driven exactly as candle's own `z_image` example —
//!   `set_timesteps(steps, Some(mu))` — which under the `z_image_turbo` config keeps the σ schedule
//!   consistent with the DiT timestep (the `None`/static-shift path desyncs them and speckles; see
//!   [`Pipeline::render`]).
//! - **Deterministic seeding (sc-3673 parity)**: initial latent noise is drawn from a
//!   fixed-algorithm CPU RNG (`StdRng`, ChaCha) seeded by `seed` and moved to the device — NOT
//!   candle's CUDA `Tensor::randn`, whose seed→noise mapping is not launch-portable. The flow-match
//!   Euler step is non-stochastic, so the whole generation is a pure function of `(seed, request)` —
//!   which is what the gen-core-testkit seed-determinism check (sc-4481) requires.
//! - **CLI/PNG/sidecar removed**: progress is `on_progress(Progress::Step/Decoding)`, cancellation is
//!   `req.cancel` → typed [`gen_core::Error::Canceled`], and each image is returned as a
//!   `gen_core::Image` (RGB8) — the worker owns asset writes.
//!
//! **First-slice surface (sc-3693), matching the SDXL slice (sc-3675):** txt2img only. img2img
//! (the mlx provider's `Reference` conditioning) is NOT wired on the registered path and is rejected
//! loudly (the worker routes it to the Python fallback). LoRA/LoKr merge into the dense DiT at load
//! (sc-5166).
//!
//! **Packed Q4/Q8 tiers (sc-9408, sc-9089 umbrella).** [`Pipeline::load_components`] auto-detects a
//! pre-quantized MLX-packed tier (`SceneWorks/z-image-turbo-mlx`) by the `quantization` block in a
//! component's `config.json` ([`Pipeline::component_is_packed`]) and loads the TE + DiT + VAE **straight
//! from the packed parts** through the shared [`candle_gen::quant`] packed-detect (the vendored
//! [`crate::packed_dit`] / [`crate::packed_te`] models) — no dense bf16 CPU staging. A dense
//! `Tongyi-MAI/Z-Image-Turbo` snapshot takes the stock `candle-transformers` path unchanged (byte-exact
//! parity, pinned by the vendored models' `parity_tests`). On-the-fly quant of a *dense* tier is still
//! not done (only the pre-packed tier is a quantized path). Component caching across calls is handled by
//! the generator's `components` cache.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::sampling::TimestepConvention;
use candle_gen::gen_core::{
    self, AdapterSpec, Conditioning, GenerationRequest, Image, PidWeights, Progress,
};
// Shared per-image batch seed (`base + index`) — one home in `candle-gen` (sc-9043 / F-059).
use candle_gen::{CandleError, Result};
use candle_gen_pid::{PidDecoder, PidEngine};

/// The PiD backbone (latent-space) tag for Z-Image (epic 7840 / sc-7853). Z-Image ships the FLUX.1
/// 16-ch VAE, so it aliases the `flux` latent space — resolved via the `zimage-turbo` tag (which the
/// PiD registry maps onto the shared `flux` student; no dedicated zimage checkpoint exists).
const PID_BACKBONE: &str = "zimage-turbo";
use candle_transformers::models::z_image::preprocess::prepare_inputs;
use candle_transformers::models::z_image::scheduler::{
    calculate_shift, FlowMatchEulerDiscreteScheduler, SchedulerConfig, BASE_IMAGE_SEQ_LEN,
    BASE_SHIFT, MAX_IMAGE_SEQ_LEN, MAX_SHIFT,
};
use candle_transformers::models::z_image::text_encoder::{TextEncoderConfig, ZImageTextEncoder};
use candle_transformers::models::z_image::transformer::{
    Config as DitConfig, ZImageTransformer2DModel,
};
use candle_transformers::models::z_image::vae::{AutoEncoderKL, Encoder as VaeEncoder, VaeConfig};

use candle_gen::gen_core::tokenizer::TextTokenizer;

use crate::common::{self, ResizePolicy};
use crate::packed_dit::ZImageTransformer2DModel as PackedDit;
use crate::packed_te::ZImageTextEncoder as PackedTe;

/// The DiT, loaded **dense** (the stock `candle-transformers` model — a dense bf16 tier or the
/// adapter-merged path) or **packed** (the vendored [`PackedDit`] built straight from an MLX-packed tier
/// — sc-9408). Both expose the same `forward(x, t, cap_feats, cap_mask)` → raw velocity, so the render
/// loops are unchanged.
pub(crate) enum DiT {
    // Boxed to keep the two arms comparably sized (`large_enum_variant`) — both models are heavy and
    // the enum lives behind an `Arc` regardless.
    Dense(Box<ZImageTransformer2DModel>),
    Packed(Box<PackedDit>),
}

impl DiT {
    pub(crate) fn forward(
        &self,
        x: &Tensor,
        t: &Tensor,
        cap_feats: &Tensor,
        cap_mask: &Tensor,
    ) -> candle_gen::candle_core::Result<Tensor> {
        match self {
            Self::Dense(m) => m.forward(x, t, cap_feats, cap_mask),
            Self::Packed(m) => m.forward(x, t, cap_feats, cap_mask),
        }
    }
}

/// The Qwen3 text encoder, dense (stock) or packed (vendored [`PackedTe`], sc-9408). Same
/// `forward(input_ids)` → layer[-2] hidden states contract.
pub(crate) enum TextEnc {
    Dense(Box<ZImageTextEncoder>),
    Packed(Box<PackedTe>),
}

impl TextEnc {
    pub(crate) fn forward(&self, input_ids: &Tensor) -> candle_gen::candle_core::Result<Tensor> {
        match self {
            Self::Dense(m) => m.forward(input_ids),
            Self::Packed(m) => m.forward(input_ids),
        }
    }
}

/// Z-Image-Turbo is guidance-distilled to a fixed 4-step schedule; used when a request omits
/// `steps`. Matches `mlx-gen-z-image`'s `DEFAULT_STEPS`.
pub(crate) const DEFAULT_STEPS: usize = 4;

/// Base (non-Turbo) Z-Image default steps — undistilled foundation model. The card recommends 28–50;
/// 50 matches the reference `ZImagePipeline` example (`num_inference_steps=50`) and the mlx base
/// provider (`mlx-gen-z-image::model_base::DEFAULT_STEPS`, sc-8320). Used when a base request omits
/// `steps`.
pub(crate) const BASE_DEFAULT_STEPS: usize = 50;

/// Flow-match time-shift for the **base** Z-Image: `scheduler/scheduler_config.json`
/// (`FlowMatchEulerDiscreteScheduler`, `shift=6.0`, `use_dynamic_shifting=false`) — static,
/// resolution-independent. **This is the sole scheduler delta vs Turbo (3.0).** Mirrors
/// `mlx-gen-z-image::model_base::SCHEDULE_SHIFT`.
pub(crate) const BASE_SCHEDULE_SHIFT: f64 = 6.0;

/// Default CFG scale for the base — the card recommends 3.0–5.0; 4.0 matches the reference
/// `ZImagePipeline` example (`guidance_scale=4`) and the mlx base provider. Used when a base request
/// omits `guidance`.
pub(crate) const BASE_DEFAULT_GUIDANCE: f32 = 4.0;

// The shared Z-Image geometry/tokenizer constants now live in [`crate::common`] (sc-9002 / F-022) —
// re-exported here at their historical `crate::pipeline::…` paths so the trainer's preview-sample path
// (sc-8650) keeps importing them from one place. Single source of truth: [`crate::common`].
pub(crate) use crate::common::{ENC_DTYPE, LATENT_CHANNELS, PATCH_SIZE, SPATIAL_SCALE};

/// img2img start step — the Z-Image "structure-preservation" convention (the fork's `init_time_step`,
/// mirrored from `mlx-gen`'s shared `img2img::init_time_step`): for a reference with `strength` in
/// `(0, 1]`, `max(1, floor(num_steps · strength))`; otherwise `0` (pure txt2img, no reference blend).
/// **Higher strength → later start → fewer denoise steps → output stays CLOSER to the reference** — the
/// inverse of the SDXL knob, matched here so the strength knob behaves identically on the Mac (MLX) and
/// Windows (candle) base lanes. `floor` because Python `int(steps · strength)` truncates toward zero for
/// `s ≥ 0`. Pure function so the cross-backend-parity law is unit-testable without a GPU.
pub(crate) fn init_time_step(num_steps: usize, strength: Option<f32>) -> usize {
    match strength {
        Some(s) if s > 0.0 => {
            let s = s.clamp(0.0, 1.0);
            ((num_steps as f32 * s) as usize).max(1)
        }
        _ => 0,
    }
}

/// Resolve the single img2img init image + its effective strength from the request's conditioning
/// (sc-8646), mirroring `mlx-gen-z-image::pipeline::resolve_reference`. A per-reference `strength`
/// overrides `req.strength`. The base Z-Image conditions on exactly one init image, so more than one
/// [`Conditioning::Reference`] is an error (multi-image would be `MultiReference`, unadvertised here);
/// non-`Reference` conditioning kinds are already rejected by the capability floor in `validate`.
pub(crate) fn resolve_reference(req: &GenerationRequest) -> Result<Option<(&Image, Option<f32>)>> {
    let mut reference = None;
    for c in &req.conditioning {
        if let Conditioning::Reference { image, strength } = c {
            if reference.is_some() {
                return Err(CandleError::Msg(
                    "z_image: multiple reference images are not supported (single img2img init only)"
                        .into(),
                ));
            }
            reference = Some((image, strength.or(req.strength)));
        }
    }
    Ok(reference)
}

/// The **base** (non-Turbo) Z-Image flow-match scheduler config: `shift = 6.0`,
/// `use_dynamic_shifting = false` (the base model's `scheduler/scheduler_config.json`). Distinct from
/// `SchedulerConfig::z_image_turbo()` (shift 3.0) — the sole scheduler delta the base introduces
/// (sc-8414). Built explicitly because candle-transformers only ships a `z_image_turbo()` constructor.
pub(crate) fn base_scheduler_config() -> SchedulerConfig {
    SchedulerConfig {
        num_train_timesteps: 1000,
        shift: BASE_SCHEDULE_SHIFT,
        use_dynamic_shifting: false,
    }
}

/// A txt2img pipeline handle: the snapshot `root` + the compute device/dtype (bf16) + any LoRA/LoKr
/// adapters to merge into the DiT at component-load time (sc-5166). Loading the heavy components is
/// done by [`load_components`](Self::load_components) and owned/cached by the generator, mirroring
/// the SDXL provider's lazy split.
pub(crate) struct Pipeline {
    root: PathBuf,
    device: Device,
    dtype: DType,
    /// Adapters merged into the DiT weights at load. Empty ⇒ the stock mmap build (zero regression).
    adapters: Vec<AdapterSpec>,
    /// The `LoadSpec::pid` component captured at load (epic 7840 / sc-7853), built into the cached
    /// [`Components`] so the PiD engine loads once alongside the base model. `None` ⇒ native VAE decode.
    pid_spec: Option<PidWeights>,
    /// External ComfyUI component sources (epic 10451 Phase 2, sc-10668). `Some` ⇒ `load_components`
    /// builds the DiT/TE/VAE from the in-place ComfyUI files (DiT + VAE key-remapped in memory)
    /// instead of a diffusers snapshot dir. Dense-only: no packed tier, no adapters, no PiD.
    comfyui: Option<std::sync::Arc<crate::comfyui::ComfyuiSources>>,
}

/// The loaded Z-Image components, `Arc`-shared so the generator can cache them across `generate`
/// calls and cheaply clone them out for a render. All three are resolution-agnostic (the DiT/VAE
/// read fixed configs; latent dims come from the request), so one set serves every request size.
#[derive(Clone)]
pub(crate) struct Components {
    text_encoder: Arc<TextEnc>,
    transformer: Arc<DiT>,
    vae: Arc<AutoEncoderKL>,
    /// Qwen tokenizer, loaded+parsed **once** at component load and reused across every prompt/uncond
    /// encode (sc-8991 / F-011) instead of re-parsing `tokenizer.json` per branch.
    tokenizer: Arc<TextTokenizer>,
    /// Optional NVIDIA PiD super-resolving decoder (epic 7840 / sc-7853); None ⇒ native VAE decode.
    pid: Option<Arc<PidEngine>>,
}

impl Pipeline {
    /// Build the (light) pipeline handle for the Z-Image snapshot `root` at the given device/dtype,
    /// with `adapters` to merge into the DiT. Does **no** weight I/O — components load lazily via
    /// [`load_components`](Self::load_components).
    pub(crate) fn load(
        root: &Path,
        device: &Device,
        dtype: DType,
        adapters: &[AdapterSpec],
        pid_spec: Option<PidWeights>,
    ) -> Self {
        Self {
            root: root.to_path_buf(),
            device: device.clone(),
            dtype,
            adapters: adapters.to_vec(),
            pid_spec,
            comfyui: None,
        }
    }

    /// Build the (light) pipeline handle for an in-place **ComfyUI** Z-Image load (sc-10668): the DiT
    /// and VAE are key-remapped from the ComfyUI single-file components in memory and the Qwen3 encoder
    /// loads verbatim, all at first [`load_components`](Self::load_components). `root` is set to the
    /// sources' `tokenizer_dir` so [`common::build_tokenizer`] finds `tokenizer/tokenizer.json`. Does no
    /// weight I/O here. Dense-only (no packed tier / adapters / PiD).
    pub(crate) fn load_comfyui(
        sources: std::sync::Arc<crate::comfyui::ComfyuiSources>,
        device: &Device,
        dtype: DType,
    ) -> Self {
        Self {
            root: sources.tokenizer_dir.clone(),
            device: device.clone(),
            dtype,
            adapters: Vec::new(),
            pid_spec: None,
            comfyui: Some(sources),
        }
    }

    /// Load the three heavy components from the snapshot's diffusers component subdirs
    /// (`text_encoder/`, `transformer/`, `vae/`). `use_accelerated_attn` enables the DiT's fused
    /// attention dispatch (CUDA flash-attn / Metal SDPA); on a build without those features the
    /// reference falls back to the backend-agnostic manual path, so this is inert there.
    pub(crate) fn load_components(&self, use_accelerated_attn: bool) -> Result<Components> {
        // sc-10668: an in-place ComfyUI load remaps the DiT/VAE and loads the Qwen3 TE verbatim from
        // the three ComfyUI single-files — no snapshot dir, no packed detection, no adapters/PiD.
        if let Some(sources) = self.comfyui.clone() {
            return self.load_comfyui_components(&sources, use_accelerated_attn);
        }
        // A pre-quantized MLX-packed tier (`SceneWorks/z-image-turbo-mlx` q4/q8) carries a
        // `quantization` block in each component's `config.json`; on detection the TE + DiT + VAE load
        // **straight from the packed parts** (sc-9408, no dense bf16 staging). A dense snapshot
        // (`Tongyi-MAI/Z-Image-Turbo`, no `quantization` block) takes the stock path unchanged. Adapters
        // (LoRA/LoKr) fold into *dense* DiT weights on the dense tier; on a **packed** tier they ride as
        // **forward-time additive residuals** on the packed base (sc-11105), so a user LoRA no longer
        // forces a full dense build.
        let packed = self.component_is_packed("transformer")?;

        let text_encoder = if self.component_is_packed("text_encoder")? {
            let vb = self.component_vb("text_encoder")?;
            TextEnc::Packed(Box::new(PackedTe::new(&TextEncoderConfig::z_image(), vb)?))
        } else {
            let vb = self.component_vb("text_encoder")?;
            TextEnc::Dense(Box::new(ZImageTextEncoder::new(
                &TextEncoderConfig::z_image(),
                vb,
            )?))
        };

        let mut dit_cfg = DitConfig::z_image_turbo();
        dit_cfg.set_use_accelerated_attn(use_accelerated_attn);
        let transformer = if packed || !self.adapters.is_empty() {
            // A packed tier OR any adapter load builds the **vendored** packed-detect DiT: a packed tier
            // loads straight from the packed parts; a dense tier loads bf16 unchanged (the vendored dense
            // forward is byte-identical to the stock model — `packed_dit::parity_tests`). Any adapters
            // install as **forward-time additive residuals** on that base (sc-11105, additive-everywhere
            // for epic 10765): the base is never folded — a packed base stays packed, a dense base stays
            // an unmutated mmap — so a user LoRA never pins an un-evictable folded weight (the fold path,
            // `merge_adapters`, is retained as a public utility but no longer on the load path). Additive
            // equals the old dense fold to f32 tolerance (~1 ULP). The stock model can't carry residuals,
            // so an adapter load on a dense tier uses the vendored DiT here.
            let vb = self.component_vb("transformer")?;
            let mut dit = PackedDit::new(&dit_cfg, vb)?;
            if !self.adapters.is_empty() {
                crate::adapters::install_additive(&mut dit, &self.adapters)?;
            }
            DiT::Packed(Box::new(dit))
        } else {
            // Dense tier, no adapters: the stock candle-transformers model (byte-identical fast path).
            DiT::Dense(Box::new(ZImageTransformer2DModel::new(
                &dit_cfg,
                self.component_vb("transformer")?,
            )?))
        };

        // The VAE packs only its 8 tiny (512×64) mid-block attention projections; rather than vendor the
        // whole 685-line `AutoEncoderKL` for negligible weights, a packed VAE dequantizes those 8 to
        // dense (from the packed parts — no dense tier downloaded) and feeds the STOCK VAE. A dense VAE
        // mmaps as before.
        let vae = if self.component_is_packed("vae")? {
            AutoEncoderKL::new(&VaeConfig::z_image(), self.vae_vb_dequantized()?)?
        } else {
            AutoEncoderKL::new(&VaeConfig::z_image(), self.component_vb("vae")?)?
        };

        let tokenizer = common::build_tokenizer(&self.root, "z-image")?;
        // Load the optional PiD super-resolving decoder once (epic 7840 / sc-7853) when the caller
        // opted in via `LoadSpec::pid`; Z-Image aliases the FLUX.1 latent space (`zimage-turbo` → the
        // shared `flux` student).
        let pid = match self.pid_spec.as_ref() {
            Some(spec) => Some(Arc::new(PidEngine::from_spec(
                spec,
                PID_BACKBONE,
                &self.device,
            )?)),
            None => None,
        };
        Ok(Components {
            text_encoder: Arc::new(text_encoder),
            transformer: Arc::new(transformer),
            vae: Arc::new(vae),
            tokenizer: Arc::new(tokenizer),
            pid,
        })
    }

    /// Load the three heavy components from an in-place ComfyUI install (sc-10668): the DiT and VAE are
    /// key-remapped in memory ([`crate::comfyui`]) then built via `VarBuilder::from_tensors`; the Qwen3
    /// text encoder loads verbatim from its ComfyUI file (standard HF layout); the
    /// tokenizer comes from our shipped snapshot. Dense bf16 only — the fp8/scaled-fp8/GGUF quant slices
    /// (sc-10670/10671/10672/10680/10681) add a dequant step ahead of the same key remaps.
    fn load_comfyui_components(
        &self,
        sources: &crate::comfyui::ComfyuiSources,
        use_accelerated_attn: bool,
    ) -> Result<Components> {
        use candle_gen::candle_core::safetensors;

        // DiT: ComfyUI-native keys → diffusers/candle keys (fused-qkv split + renames), then build.
        let dit_map = safetensors::load(&sources.transformer_file, &Device::Cpu)?;
        let dit_map = crate::comfyui::remap_dit_comfyui_to_diffusers(dit_map)?;
        let mut dit_cfg = DitConfig::z_image_turbo();
        dit_cfg.set_use_accelerated_attn(use_accelerated_attn);
        let dit_vb = VarBuilder::from_tensors(dit_map, self.dtype, &self.device);
        let transformer = DiT::Dense(Box::new(ZImageTransformer2DModel::new(&dit_cfg, dit_vb)?));

        // Text encoder: standard HF Qwen3 — loaded verbatim (only a bf16 cast via mmap).
        let te_vb = candle_gen::mmap_var_builder(
            std::slice::from_ref(&sources.text_encoder_file),
            self.dtype,
            &self.device,
        )?;
        let text_encoder = TextEnc::Dense(Box::new(ZImageTextEncoder::new(
            &TextEncoderConfig::z_image(),
            te_vb,
        )?));

        // VAE: BFL/ldm keys → diffusers keys (incl. the up-block reversal + 1×1-conv→Linear squeeze).
        let vae_map = safetensors::load(&sources.vae_file, &Device::Cpu)?;
        let vae_map = crate::comfyui::remap_vae_ldm_to_diffusers(vae_map)?;
        let vae_vb = VarBuilder::from_tensors(vae_map, self.dtype, &self.device);
        let vae = AutoEncoderKL::new(&VaeConfig::z_image(), vae_vb)?;

        let tokenizer = common::build_tokenizer(&self.root, "z-image comfyui")?;
        Ok(Components {
            text_encoder: Arc::new(text_encoder),
            transformer: Arc::new(transformer),
            vae: Arc::new(vae),
            tokenizer: Arc::new(tokenizer),
            pid: None,
        })
    }

    /// Whether the snapshot component `sub/` is a **pre-quantized MLX-packed tier** — its `config.json`
    /// carries a `quantization` block ([`candle_gen::quant::PackedConfig`]) that the install-time convert
    /// job writes for a packed component. Mirrors flux2's `component_is_packed` (sc-9087).
    ///
    /// A **genuinely-absent** `config.json` (file NotFound) is a legitimate dense/fixture snapshot shape
    /// → `Ok(false)` (dense path), so a fixture with no `config.json` still loads. A config that **is
    /// present but corrupt** (I/O error or malformed JSON — e.g. a partial download) errors loudly naming
    /// the file rather than silently downgrading a packed component to the dense path (wrong tier /
    /// missing weights, no diagnostic). A well-formed config with no `quantization` block is a dense tier
    /// → `Ok(false)` (sc-9426, F-073 sibling).
    pub(crate) fn component_is_packed(&self, sub: &str) -> Result<bool> {
        let path = self.root.join(sub).join("config.json");
        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            // No config.json at all → legitimate dense / fixture snapshot, not packed.
            Err(ref e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(false),
            // Present but unreadable (permissions, partial download) → surface, don't swallow.
            Err(e) => {
                return Err(CandleError::Msg(format!(
                    "z-image: read {}: {e}",
                    path.display()
                )))
            }
        };
        // Present but malformed JSON → corrupt snapshot, error rather than fall to dense.
        let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| {
            CandleError::Msg(format!(
                "z-image: parse {} (corrupt snapshot?): {e}",
                path.display()
            ))
        })?;
        Ok(candle_gen::quant::PackedConfig::from_config(&v).is_some())
    }

    /// Build a VAE [`VarBuilder`] for a **packed** tier by dequantizing the 8 packed mid-block attention
    /// projections (`{encoder,decoder}.mid_block.attentions.0.{to_q,to_k,to_v,to_out.0}`) to dense and
    /// passing every other (already-dense) tensor through unchanged — so the stock `AutoEncoderKL` loads
    /// without seeing a `.weight` u32/`.scales`/`.biases` triple it can't read. The dequant is ~2 MB of
    /// one-time work (the sc-9408 pragmatic VAE path — see [`crate::quant::dequant_packed_to_dense`]).
    fn vae_vb_dequantized(&self) -> Result<VarBuilder<'static>> {
        use candle_gen::candle_core::safetensors::MmapedSafetensors;
        let files = self.component_files("vae")?;
        // SAFETY: mmap of read-only weight files; standard candle loading path.
        let st = unsafe { MmapedSafetensors::multi(&files)? };
        let src = VarBuilder::from_backend(Box::new(st), self.dtype, self.device.clone());

        // Collect every tensor, dequantizing the packed attention triples and dropping their
        // `.scales`/`.biases` siblings; pass all other tensors through at their native dtype.
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        // SAFETY: same file set; read the raw header to enumerate keys.
        let st2 = unsafe { MmapedSafetensors::multi(&files)? };
        let packed_bases: std::collections::HashSet<String> = st2
            .tensors()
            .iter()
            .filter_map(|(k, _)| k.strip_suffix(".scales").map(|b| b.to_string()))
            .collect();
        for (key, _) in st2.tensors() {
            // Skip the packed-triple siblings — they're folded into the dequantized dense `.weight`.
            if key.ends_with(".scales") || key.ends_with(".biases") {
                continue;
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
            // Dense tensor — load it through at its stored dtype/device.
            let t = st2.load(&key, &self.device)?;
            tensors.insert(key.clone(), t.to_dtype(self.dtype)?);
        }
        Ok(VarBuilder::from_tensors(tensors, self.dtype, &self.device))
    }

    /// Resolve the sorted list of `.safetensors` files in the snapshot component subdir `sub`
    /// (single-file or sharded — diffusers ships both layouts), erroring if the dir or files are
    /// missing.
    fn component_files(&self, sub: &str) -> Result<Vec<PathBuf>> {
        let dir = self.root.join(sub);
        if !dir.is_dir() {
            return Err(CandleError::Msg(format!(
                "z-image snapshot is missing the {sub}/ component directory (expected a diffusers \
                 multi-component snapshot at {})",
                self.root.display()
            )));
        }
        // Shared sorted-`.safetensors` resolver (sc-8999 / F-019); the crafted "missing dir" message
        // above stays local (it names the expected diffusers snapshot).
        candle_gen::sorted_safetensors(&dir, "z-image")
    }

    /// Build a [`VarBuilder`] over every `.safetensors` in the snapshot component subdir `sub`, at
    /// this pipeline's dtype/device (the stock mmap path; no adapters).
    fn component_vb(&self, sub: &str) -> Result<VarBuilder<'static>> {
        let files = self.component_files(sub)?;
        candle_gen::mmap_var_builder(&files, self.dtype, &self.device)
    }

    /// Build the standalone f32 VAE **encoder** for the base img2img / `Reference` path (sc-8646). The
    /// decode `AutoEncoderKL` holds an encoder too, but (a) it is private and (b) its `encode` samples
    /// the diagonal-gaussian via the *device* RNG (not launch-portable — breaks sc-3673), so — exactly
    /// like [`crate::edit`] and [`crate::control`] — the raw `Encoder` is run here to take the
    /// distribution **mean** deterministically. Only built on the first img2img request (cached by the
    /// generator), so the txt2img / Turbo path never pays for it.
    pub(crate) fn load_vae_encoder(&self) -> Result<VaeEncoder> {
        let files = self.component_files("vae")?;
        let vb = candle_gen::mmap_var_builder(&files, ENC_DTYPE, &self.device)?;
        Ok(VaeEncoder::new(&VaeConfig::z_image(), vb.pp("encoder"))?)
    }

    /// VAE-encode `source` (LANCZOS-resized to the render size, normalized to `[-1, 1]` NCHW) to the
    /// deterministic clean init latent `(1, 16, H/8, W/8)` at the compute dtype (bf16): the distribution
    /// **mean** (not a sampled draw), mapped to latent space as `(mean − shift) · scale` — the same
    /// deterministic encode [`crate::edit::ZImageEdit::encode_source`] uses. `encoder` is the f32 encoder
    /// from [`load_vae_encoder`](Self::load_vae_encoder).
    pub(crate) fn encode_reference(
        &self,
        encoder: &VaeEncoder,
        source: &Image,
        width: u32,
        height: u32,
    ) -> Result<Tensor> {
        let vae_cfg = VaeConfig::z_image();
        // `ResizeIfNeeded`: no-op when the reference is already at the render size (the base img2img
        // resize policy — see [`ResizePolicy`]).
        let img = common::preprocess_image(
            source,
            width,
            height,
            ResizePolicy::ResizeIfNeeded,
            &self.device,
            "z_image img2img",
        )?; // f32 (1,3,H,W) [-1,1]
        common::encode_mean(
            encoder,
            &img,
            vae_cfg.shift_factor,
            vae_cfg.scaling_factor,
            self.dtype,
        )
    }

    /// Token `ids` → `cap_feats` `(seq, 2560)` at the compute dtype: run the dense-or-packed Qwen3
    /// encoder and squeeze the batch axis via the shared [`common::encode_ids`]. The reference
    /// `prepare_inputs` does the SEQ_MULTI_OF padding + attention mask downstream, so every id here is a
    /// valid token.
    fn encode_cap(&self, te: &TextEnc, ids: &[i32]) -> Result<Tensor> {
        common::encode_ids(ids, &self.device, self.dtype, |input_ids| {
            te.forward(input_ids)
        })
    }

    /// Prompt → `cap_feats` `(seq, 2560)`. Tokenizes with the shared Qwen chat template
    /// ([`common::prompt_ids`]) and runs the Qwen3 encoder.
    pub(crate) fn text_embeddings(
        &self,
        te: &TextEnc,
        tok: &TextTokenizer,
        prompt: &str,
    ) -> Result<Tensor> {
        let ids = common::prompt_ids(tok, prompt, "z-image")?;
        self.encode_cap(te, &ids)
    }

    /// Negative prompt → `cap_feats` for the **unconditional** CFG branch of the base path (sc-8414).
    /// Delegates to the shared [`common::uncond_ids`], which routes the **empty string** through the
    /// QwenInstruct chat-template scaffolding rather than the empty-short-circuiting `tokenize` (the
    /// sc-8646 fix — a plain `tokenize("")` yields `(1, 0)` before the template and breaks the empty
    /// negative prompt). A non-empty negative prompt takes the ordinary conditional path.
    pub(crate) fn uncond_embeddings(
        &self,
        te: &TextEnc,
        tok: &TextTokenizer,
        negative_prompt: &str,
    ) -> Result<Tensor> {
        let ids = common::uncond_ids(tok, negative_prompt, "z-image")?;
        self.encode_cap(te, &ids)
    }

    /// Render `req` against pre-loaded `components`, emitting per-step progress and honoring
    /// `req.cancel`. Returns one `gen_core::Image` per `req.count` (each with seed `base_seed + index`).
    pub(crate) fn render(
        &self,
        req: &GenerationRequest,
        components: &Components,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Vec<Image>> {
        let steps = req.steps.map(|s| s as usize).unwrap_or(DEFAULT_STEPS);
        let base_seed = req.seed.unwrap_or_else(gen_core::default_seed);
        let lat_h = (req.height / SPATIAL_SCALE) as usize;
        let lat_w = (req.width / SPATIAL_SCALE) as usize;

        // Text embeddings are seed- and image-independent: encode once for the whole batch.
        let cap =
            self.text_embeddings(&components.text_encoder, &components.tokenizer, &req.prompt)?;

        // Resolve the decode seam once for the whole batch (epic 7840 / sc-7853): a per-generation PiD
        // decoder bound to this prompt when `req.use_pid` is set (errors if requested but not loaded),
        // else `None` → the native VAE decode. Shared across `count` images (same prompt).
        let pid_decoder = candle_gen_pid::resolve_pid_decoder(
            components.pid.as_deref(),
            req,
            base_seed,
            crate::MODEL_ID,
        )?;

        candle_gen::for_each_image_seed(base_seed, req.count, |seed| {
            // sc-3673 parity — deterministic, launch-portable initial noise (shared [`common::seed_noise`]).
            let noise = common::seed_noise(seed, lat_h, lat_w, &self.device, self.dtype)?;

            // Flow-match Euler schedule. Match the candle `z_image` reference: pass `Some(mu)` (the
            // resolution-dependent shift parameter from `calculate_shift`). Under
            // `use_dynamic_shifting=false` (the `z_image_turbo` config) the `Some(mu)` arm applies NO
            // sigma shift, so the sigmas stay LINEAR and consistent with `current_timestep_normalized`
            // (which is derived from the un-shifted `timesteps`). This is correctness-critical, NOT a
            // style knob: passing `None` takes the scheduler's static-shift branch, which shifts
            // `sigmas` WITHOUT updating `timesteps` — desyncing the t fed to the DiT from the σ used in
            // the Euler step, which leaves residual high-frequency noise (visible speckle) in the
            // decode. The unit-normal noise is the flow-match txt2img prior as-is (max σ = 1.0).
            let image_seq_len =
                ((lat_h as u32 / PATCH_SIZE) * (lat_w as u32 / PATCH_SIZE)) as usize;
            let mu = calculate_shift(
                image_seq_len,
                BASE_IMAGE_SEQ_LEN,
                MAX_IMAGE_SEQ_LEN,
                BASE_SHIFT,
                MAX_SHIFT,
            );
            let mut scheduler =
                FlowMatchEulerDiscreteScheduler::new(SchedulerConfig::z_image_turbo());
            scheduler.set_timesteps(steps, Some(mu));

            // Unified curated sampler/scheduler routing (epic 7114 P4, sc-7123). The NATIVE schedule is
            // the scheduler's σ table verbatim (linear / un-shifted for the turbo config — see the
            // comment above), so `resolve_flow_schedule(None, …)` returns it byte-for-byte and the
            // default `euler` is the N1 no-op = the legacy `scheduler.step` loop
            // `x + v·(σ_{i+1} − σ_i)`. The schedule is unshifted (`mu = 0.0` for the curated axis).
            // Z-Image feeds the DiT the 1−σ conditioning (`OneMinusSigma`) and the predicted velocity
            // is NEGATED before the step — both Z-Image-specific quirks live inside the `predict`
            // closure, so a multi-eval solver re-applies them each eval.
            let native: Vec<f32> = scheduler.sigmas.iter().map(|&s| s as f32).collect();
            let sigmas =
                candle_gen::resolve_flow_schedule(req.scheduler.as_deref(), 0.0, steps, &native);

            // `prepare_inputs` pads cap_feats to SEQ_MULTI_OF (+ attention mask) and adds the
            // singleton frame axis to the latents → (1, 16, 1, lat_h, lat_w).
            let prepared = prepare_inputs(&noise, std::slice::from_ref(&cap), &self.device)?;
            let cap_feats = prepared.cap_feats;
            let cap_mask = prepared.cap_mask;

            let latents = candle_gen::run_flow_sampler(
                req.sampler.as_deref(),
                TimestepConvention::OneMinusSigma,
                &sigmas,
                prepared.latents,
                seed,
                &req.cancel,
                on_progress,
                |latents, t| -> Result<Tensor> {
                    // `t` is the 1−σ conditioning (OneMinusSigma) the DiT embeds — the same value the
                    // reference scheduler's `current_timestep_normalized` returns. The embedder upcasts
                    // to f32 internally, so f32 here is correct regardless of the model dtype.
                    let t_tensor = Tensor::from_vec(vec![t], (1,), &self.device)?;
                    let velocity = components
                        .transformer
                        .forward(latents, &t_tensor, &cap_feats, &cap_mask)?
                        .neg()?;
                    Ok(velocity)
                },
            )?;

            on_progress(Progress::Decoding);
            self.decode(&components.vae, pid_decoder.as_ref(), &latents)
        })
    }

    /// Render `req` against pre-loaded `components` on the **base** (non-Turbo) path: real
    /// classifier-free guidance over the static **shift=6.0** flow-match schedule (sc-8414, the candle
    /// sibling of `mlx-gen-z-image::model_base`). Emits per-step progress and honors `req.cancel`.
    ///
    /// Differences from [`render`](Self::render) (the Turbo path), all from the base model card /
    /// `scheduler/scheduler_config.json`:
    ///
    /// - **Static shift = 6.0** (Turbo's effective inference schedule is linear/un-shifted because its
    ///   `set_timesteps(steps, Some(mu))` call no-ops under `use_dynamic_shifting=false`). The base
    ///   builds its σ table with `set_timesteps(steps, None)` against a `shift=6.0` config, so the
    ///   static-shift branch actually fires. We feed that σ table to [`run_flow_sampler`] with
    ///   [`TimestepConvention::OneMinusSigma`], which derives the DiT timestep `t = 1−σ` from the σ
    ///   schedule **itself** — so the Turbo `None`-path "timesteps desync" speckle bug is structurally
    ///   absent here (we never read the scheduler's `timesteps`/`current_timestep_normalized`).
    /// - **Real CFG**: each step runs the DiT twice (cond + uncond) and combines
    ///   `v = v_uncond + guidance·(v_cond − v_uncond)`. `guidance == 1.0` collapses to a single cond
    ///   forward (Turbo-equivalent cost). The uncond branch encodes the negative prompt (empty string
    ///   when unset — the unconditional embedding).
    /// - **Default 50 steps** when `req.steps` is unset ([`BASE_DEFAULT_STEPS`]).
    ///
    /// **img2img / `Reference` (sc-8646).** When `clean` is `Some` (the caller VAE-encoded a reference
    /// image via [`encode_reference`](Self::encode_reference)) and `start_step > 0`, each image blends
    /// the pre-encoded clean latent with the seeded noise at `σ_start` (the flow-match interpolation
    /// `x_t = (1 − σ)·clean + σ·noise`) and denoises the **reduced** `start_step..` tail of the σ
    /// schedule — real CFG applies to the img2img tail exactly as to txt2img. `start_step == 0` (`clean`
    /// is `None`) is pure txt2img: `x_t = noise`, full schedule — byte-identical to the pre-sc-8646 path.
    /// Mirrors `mlx-gen-z-image::model_base` + `pipeline::render_batch`.
    pub(crate) fn render_base(
        &self,
        req: &GenerationRequest,
        components: &Components,
        clean: Option<&Tensor>,
        start_step: usize,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Vec<Image>> {
        let steps = req.steps.map(|s| s as usize).unwrap_or(BASE_DEFAULT_STEPS);
        let base_seed = req.seed.unwrap_or_else(gen_core::default_seed);
        let lat_h = (req.height / SPATIAL_SCALE) as usize;
        let lat_w = (req.width / SPATIAL_SCALE) as usize;

        // Real CFG: `req.guidance` is the classifier-free guidance scale (default 4.0). A value of 1.0
        // turns CFG off (single cond forward, Turbo-equivalent cost).
        let guidance = req.guidance.unwrap_or(BASE_DEFAULT_GUIDANCE);
        let cfg_on = guidance != 1.0;

        // Text embeddings are seed- and image-independent: encode once for the whole batch. The uncond
        // branch (negative prompt, empty when unset) is only encoded when CFG is active.
        let cap =
            self.text_embeddings(&components.text_encoder, &components.tokenizer, &req.prompt)?;
        let neg_cap = if cfg_on {
            let neg = req.negative_prompt.as_deref().unwrap_or("");
            Some(self.uncond_embeddings(&components.text_encoder, &components.tokenizer, neg)?)
        } else {
            None
        };

        // Resolve the decode seam once for the whole batch (epic 7840 / sc-7853): a per-generation PiD
        // decoder bound to this prompt when `req.use_pid` is set (errors if requested but not loaded),
        // else `None` → the native VAE decode. Shared across `count` images (same prompt).
        let pid_decoder = candle_gen_pid::resolve_pid_decoder(
            components.pid.as_deref(),
            req,
            base_seed,
            crate::MODEL_ID,
        )?;

        candle_gen::for_each_image_seed(base_seed, req.count, |seed| {
            // sc-3673 parity — deterministic, launch-portable initial noise (shared [`common::seed_noise`]).
            let noise = common::seed_noise(seed, lat_h, lat_w, &self.device, self.dtype)?;

            // Static shift=6.0 schedule (the base model's scheduler_config.json). Unlike the Turbo
            // path's `Some(mu)` no-op, the base passes `None` so the static-shift branch actually
            // shifts the σ table; `run_flow_sampler`'s `OneMinusSigma` derives the DiT timestep from
            // these σ directly, so there is no timesteps desync to guard against.
            let mut scheduler = FlowMatchEulerDiscreteScheduler::new(base_scheduler_config());
            scheduler.set_timesteps(steps, None);
            let native: Vec<f32> = scheduler.sigmas.iter().map(|&s| s as f32).collect();

            // Curated scheduler axis (epic 7114): an unset `req.scheduler` returns `native` verbatim
            // (the byte-exact shift=6.0 default); a curated name re-shapes σ over the same shift
            // (`mu = ln(shift)`), exactly as `mlx-gen-z-image::model_base`.
            let sigmas = candle_gen::resolve_flow_schedule(
                req.scheduler.as_deref(),
                (BASE_SCHEDULE_SHIFT as f32).ln(),
                steps,
                &native,
            );

            // img2img / `Reference` (sc-8646): blend the pre-encoded clean latent with the seeded noise
            // at `σ_start = sigmas[start]` and denoise the reduced `start..` schedule tail. `start` is
            // clamped to the schedule because a curated scheduler may return a length ≠ `steps + 1`.
            // For txt2img (`clean` is `None`, `start_step == 0`) this is `x_t = noise` over the full
            // schedule — byte-identical to the pre-sc-8646 path. Mirrors `render_batch`'s
            // `add_noise_by_interpolation` (`x_t = (1 − σ)·clean + σ·noise`).
            let start = start_step.min(sigmas.len().saturating_sub(1));
            let x_t = match clean {
                Some(clean) => {
                    let sigma_start = sigmas[start] as f64;
                    (clean.affine(1.0 - sigma_start, 0.0)? + noise.affine(sigma_start, 0.0)?)?
                }
                None => noise,
            };
            let run_sigmas = &sigmas[start..];

            // `prepare_inputs` pads cap_feats to SEQ_MULTI_OF (+ attention mask) for both the cond and
            // (when CFG is on) the uncond branch, and adds the singleton frame axis to the latents. The
            // uncond branch only uses cap_feats/cap_mask (its `latents` are discarded), so passing `x_t`
            // there is fine.
            let prepared = prepare_inputs(&x_t, std::slice::from_ref(&cap), &self.device)?;
            let cap_feats = prepared.cap_feats;
            let cap_mask = prepared.cap_mask;
            let uncond = match neg_cap.as_ref() {
                Some(neg) => {
                    let p = prepare_inputs(&x_t, std::slice::from_ref(neg), &self.device)?;
                    Some((p.cap_feats, p.cap_mask))
                }
                None => None,
            };

            let latents = candle_gen::run_flow_sampler(
                req.sampler.as_deref(),
                TimestepConvention::OneMinusSigma,
                run_sigmas,
                prepared.latents,
                seed,
                &req.cancel,
                on_progress,
                |latents, t| -> Result<Tensor> {
                    let t_tensor = Tensor::from_vec(vec![t], (1,), &self.device)?;
                    // Conditional velocity (Z-Image sign convention: the DiT output is negated before
                    // the flow-match step). The CFG combine is done on the negated velocities, which is
                    // linear so the result is identical to combining-then-negating.
                    let v_cond = components
                        .transformer
                        .forward(latents, &t_tensor, &cap_feats, &cap_mask)?
                        .neg()?;
                    let velocity = match uncond.as_ref() {
                        Some((neg_feats, neg_mask)) => {
                            let v_uncond = components
                                .transformer
                                .forward(latents, &t_tensor, neg_feats, neg_mask)?
                                .neg()?;
                            // v = v_uncond + guidance·(v_cond − v_uncond)
                            let delta = (&v_cond - &v_uncond)?;
                            (v_uncond + (delta * guidance as f64)?)?
                        }
                        None => v_cond,
                    };
                    Ok(velocity)
                },
            )?;

            on_progress(Progress::Decoding);
            self.decode(&components.vae, pid_decoder.as_ref(), &latents)
        })
    }

    /// Decode the final latents `(1, 16, 1, h, w)` to an RGB8 [`Image`] via the shared [`common::decode`].
    /// The native VAE applies its own `/scaling_factor + shift_factor` un-scale inside `decode`; when a
    /// PiD decoder resolved (epic 7840 / sc-7853) it super-resolves the same squeezed NCHW latent
    /// instead. `postprocess_image` maps the `[-1, 1]` output to `[0, 255]` u8, reading the size from the
    /// tensor.
    fn decode(
        &self,
        vae: &AutoEncoderKL,
        pid: Option<&PidDecoder>,
        latents: &Tensor,
    ) -> Result<Image> {
        common::decode(vae, pid, latents)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `component_is_packed` detects the `quantization` block a packed MLX tier writes into a component
    /// `config.json` (sc-9408) and returns false for a dense config or a missing file — the seam that
    /// routes `load_components` to the packed vs stock models. A *present-but-corrupt* `config.json`
    /// (malformed JSON, e.g. a partial download) errors loudly naming the file rather than silently
    /// falling to the dense path (sc-9426, F-073 sibling). GPU-free (only reads a small JSON file).
    #[test]
    fn component_is_packed_detects_quantization_block() {
        let dir = std::env::temp_dir().join(format!("sc9408_pipe_{}", std::process::id()));
        let packed = dir.join("transformer");
        let dense = dir.join("vae");
        std::fs::create_dir_all(&packed).unwrap();
        std::fs::create_dir_all(&dense).unwrap();
        std::fs::write(
            packed.join("config.json"),
            r#"{"dim": 3840, "quantization": {"bits": 4, "group_size": 64}}"#,
        )
        .unwrap();
        std::fs::write(dense.join("config.json"), r#"{"latent_channels": 16}"#).unwrap();

        let pipe = Pipeline::load(&dir, &Device::Cpu, DType::F32, &[], None);
        assert!(
            pipe.component_is_packed("transformer").unwrap(),
            "a `quantization` block ⇒ packed tier"
        );
        assert!(
            !pipe.component_is_packed("vae").unwrap(),
            "no `quantization` block ⇒ dense"
        );
        assert!(
            !pipe.component_is_packed("text_encoder").unwrap(),
            "missing config.json ⇒ dense (fixture still loads)"
        );

        // A config.json that is *present but corrupt* (malformed JSON) must error naming the file, NOT
        // silently downgrade the packed component to the dense path (sc-9426 / F-073 sibling).
        let corrupt = dir.join("transformer_bad");
        std::fs::create_dir_all(&corrupt).unwrap();
        std::fs::write(corrupt.join("config.json"), b"{ not json").unwrap();
        let bad_pipe = Pipeline::load(&dir, &Device::Cpu, DType::F32, &[], None);
        let err = bad_pipe
            .component_is_packed("transformer_bad")
            .expect_err("corrupt config.json must error, not fall to dense");
        assert!(
            format!("{err}").contains("config.json"),
            "the error should name the offending file, got: {err}"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Parity anchors against `mlx-gen-z-image`: the distilled 4-step default and the /8 16-channel
    /// latent geometry. GPU-free (asserts constants directly).
    #[test]
    fn parity_defaults_match_mlx_provider() {
        assert_eq!(DEFAULT_STEPS, 4);
        assert_eq!(SPATIAL_SCALE, 8);
        assert_eq!(LATENT_CHANNELS, 16);
        assert_eq!(PATCH_SIZE, 2);
    }

    /// Base (non-Turbo) parity constants vs Turbo + the mlx base provider (sc-8414 / mlx sc-8320):
    /// shift 6.0 (Turbo's config is 3.0), default 50 steps (Turbo 4), default CFG 4.0. These are the
    /// load-bearing port values from the base `scheduler_config.json` + the model card. GPU-free.
    #[test]
    fn base_constants_match_the_model_card() {
        assert_eq!(BASE_SCHEDULE_SHIFT, 6.0);
        assert_eq!(BASE_DEFAULT_STEPS, 50);
        assert_eq!(BASE_DEFAULT_GUIDANCE, 4.0);
        // The base scheduler config differs from Turbo only in the static shift.
        let base = base_scheduler_config();
        let turbo = SchedulerConfig::z_image_turbo();
        assert_eq!(base.shift, 6.0);
        assert_eq!(turbo.shift, 3.0);
        assert!(!base.use_dynamic_shifting && !turbo.use_dynamic_shifting);
        assert_eq!(base.num_train_timesteps, turbo.num_train_timesteps);
    }

    /// sc-8646 root-cause guard at the tokenizer seam (no GPU / no model weights — only the snapshot's
    /// `tokenizer/tokenizer.json`): base CFG with an **unset** negative prompt must be able to build an
    /// unconditional embedding. gen-core's [`TextTokenizer::tokenize`] short-circuits an empty prompt to
    /// a `(1, 0)` sequence **before** the chat template is applied (`pad_to_max_length = false`) — which
    /// is why routing the empty uncond through `text_embeddings` errored `z-image: empty prompt`. The
    /// fix ([`Pipeline::uncond_embeddings`]) encodes it via `encode_chat_ids("", true)`, which renders
    /// the QwenInstruct scaffolding around `""` and yields a **non-empty** role-marker token sequence.
    /// Set `Z_IMAGE_SNAPSHOT` or `Z_IMAGE_BASE_SNAPSHOT` (both ship the same Qwen tokenizer).
    #[test]
    #[ignore = "needs Z_IMAGE_SNAPSHOT/Z_IMAGE_BASE_SNAPSHOT for tokenizer.json (no GPU); run with --ignored"]
    fn empty_uncond_tokenizes_via_chat_template() {
        let snap = std::env::var("Z_IMAGE_BASE_SNAPSHOT")
            .or_else(|_| std::env::var("Z_IMAGE_SNAPSHOT"))
            .expect("set Z_IMAGE_SNAPSHOT or Z_IMAGE_BASE_SNAPSHOT to a Z-Image snapshot dir");
        let root = std::path::Path::new(&snap);
        // The shared tokenizer (`common::build_tokenizer`) with the shared config — the same seam the
        // three entry points now use (sc-9002).
        let tok = common::build_tokenizer(root, "z-image").expect("load tokenizer.json");

        // The trap: an empty prompt short-circuits to (1, 0) BEFORE the chat template is applied.
        assert!(
            tok.tokenize("").unwrap().ids.is_empty(),
            "empty prompt must short-circuit before the chat template (the sc-8646 trap)"
        );
        // The fix now lives in the shared `common::uncond_ids`: an empty negative prompt tokenizes via
        // the QwenInstruct chat template to a non-empty sequence, distinct from a real prompt — while
        // `common::prompt_ids("")` still errors on the empty short-circuit (the trap is real).
        let uncond_ids = common::uncond_ids(&tok, "", "z-image").expect("encode empty uncond");
        assert!(
            !uncond_ids.is_empty(),
            "empty uncond must tokenize via the chat template to a non-empty sequence"
        );
        assert!(
            common::prompt_ids(&tok, "", "z-image").is_err(),
            "an empty prompt through the conditional path must still error"
        );
        let real_ids =
            common::uncond_ids(&tok, "a red fox", "z-image").expect("encode real prompt");
        assert_ne!(uncond_ids, real_ids, "uncond scaffolding != a real prompt");

        // sc-8991 / F-011: the cached tokenizer must yield the SAME ids as a fresh `from_file` load —
        // caching removes the re-parse, never changes tokenization. Cover a real prompt + empty uncond.
        let fresh = common::build_tokenizer(root, "z-image").expect("fresh tokenizer");
        assert_eq!(
            common::uncond_ids(&tok, "", "z-image").unwrap(),
            common::uncond_ids(&fresh, "", "z-image").unwrap(),
        );
        assert_eq!(
            common::prompt_ids(&tok, "a red fox", "z-image").unwrap(),
            common::prompt_ids(&fresh, "a red fox", "z-image").unwrap(),
        );
    }

    /// The base static **shift=6.0** schedule (built `set_timesteps(steps, None)`) must: have
    /// `num_steps + 1` sigmas, start at max-σ **1.0**, strictly decrease, terminate at 0 — and, the
    /// load-bearing delta vs Turbo, actually apply the shift so its σ table is NOT the linear ramp.
    /// The shift biases the schedule toward high-noise steps (σ at a given fraction is ≥ the linear
    /// value), which is what an undistilled CFG model needs. GPU-free.
    #[test]
    fn base_schedule_applies_shift_six() {
        let steps = 50usize;
        let mut s = FlowMatchEulerDiscreteScheduler::new(base_scheduler_config());
        s.set_timesteps(steps, None);
        assert_eq!(s.sigmas.len(), steps + 1);
        assert!(
            (s.sigmas[0] - 1.0).abs() < 1e-9,
            "max sigma: {}",
            s.sigmas[0]
        );
        assert!(s.sigmas[steps].abs() < 1e-9, "terminal sigma must be 0");
        for w in s.sigmas.windows(2) {
            assert!(w[0] > w[1], "sigmas must strictly decrease: {:?}", s.sigmas);
        }
        // Shift actually applied: shift*x/(1+(shift-1)*x) > x for x in (0,1), so the shifted σ table
        // is strictly above the linear ramp at every interior node (and differs from Turbo's table).
        let mut turbo = FlowMatchEulerDiscreteScheduler::new(SchedulerConfig::z_image_turbo());
        turbo.set_timesteps(steps, None);
        for i in 1..steps {
            let linear = 1.0 - (i as f64) / (steps as f64);
            assert!(
                s.sigmas[i] > linear + 1e-9,
                "shift=6.0 must lift σ[{i}]={} above the linear ramp {linear}",
                s.sigmas[i]
            );
            assert!(
                s.sigmas[i] > turbo.sigmas[i] + 1e-9,
                "shift=6.0 σ[{i}]={} must exceed Turbo shift=3.0 σ={}",
                s.sigmas[i],
                turbo.sigmas[i]
            );
        }
        // The DiT timestep the base render feeds (1 − σ, OneMinusSigma) is derived from THIS σ table,
        // so it is consistent by construction — no `timesteps` desync (the Turbo `None`-path speckle
        // bug cannot occur on the base path).
    }

    /// The flow-match Euler schedule the pipeline drives (`set_timesteps(steps, Some(mu))`) must, for
    /// the distilled 4-step config: have `num_steps + 1` sigmas, start at max-σ **1.0**, be strictly
    /// decreasing, and terminate at 0.
    ///
    /// **Regression guard for the speckle bug:** at every step the timestep fed to the DiT
    /// (`(1000 − timesteps[i]) / 1000`, i.e. `current_timestep_normalized`) must equal `1 − σᵢ` (the σ
    /// the Euler step actually uses). The `Some(mu)` call keeps `timesteps` and `sigmas` consistent;
    /// the `None` call would shift `sigmas` without updating `timesteps`, breaking this identity and
    /// leaving residual high-frequency noise in the decode. GPU-free.
    #[test]
    fn flow_match_schedule_keeps_timestep_and_sigma_consistent() {
        // mu for a representative 1024² render: latent 128² → seq (128/2)² = 4096.
        let mu = calculate_shift(
            4096,
            BASE_IMAGE_SEQ_LEN,
            MAX_IMAGE_SEQ_LEN,
            BASE_SHIFT,
            MAX_SHIFT,
        );
        let mut s = FlowMatchEulerDiscreteScheduler::new(SchedulerConfig::z_image_turbo());
        s.set_timesteps(DEFAULT_STEPS, Some(mu));
        assert_eq!(s.sigmas.len(), DEFAULT_STEPS + 1);
        assert!(
            (s.sigmas[0] - 1.0).abs() < 1e-6,
            "max sigma: {}",
            s.sigmas[0]
        );
        assert!(
            (s.sigmas[DEFAULT_STEPS]).abs() < 1e-6,
            "terminal sigma must be 0"
        );
        for w in s.sigmas.windows(2) {
            assert!(w[0] > w[1], "sigmas must strictly decrease: {:?}", s.sigmas);
        }
        // The correctness-critical identity: t fed to the DiT == 1 − σ at every step.
        for i in 0..DEFAULT_STEPS {
            let t = (1000.0 - s.timesteps[i]) / 1000.0;
            assert!(
                (t - (1.0 - s.sigmas[i])).abs() < 1e-9,
                "t/σ desync at step {i}: t={t}, 1-σ={}",
                1.0 - s.sigmas[i]
            );
        }
    }

    /// The base img2img start-step law (sc-8646, the fork's `init_time_step` over `Option<f32>`):
    /// `max(1, floor(steps·strength))` for a strength in `(0, 1]`, else `0` (pure txt2img). Higher
    /// strength → later start → fewer denoise steps (Z-Image structure preservation). Pure, no GPU —
    /// the cross-backend-parity contract with `mlx-gen`'s shared `img2img::init_time_step`.
    #[test]
    fn init_time_step_is_the_fork_convention() {
        // None / non-positive strength ⇒ pure txt2img (start 0, reference ignored).
        assert_eq!(init_time_step(50, None), 0);
        assert_eq!(init_time_step(50, Some(0.0)), 0);
        assert_eq!(init_time_step(50, Some(-1.0)), 0);
        // floor(steps·strength), min 1.
        assert_eq!(init_time_step(50, Some(0.6)), 30); // floor(30.0)
        assert_eq!(init_time_step(50, Some(0.01)), 1); // floor(0.5)=0 → max(1,0)=1
        assert_eq!(init_time_step(4, Some(0.6)), 2); // floor(2.4)
        assert_eq!(init_time_step(50, Some(1.0)), 50); // == steps ⇒ empty loop, source round-trip
        assert_eq!(init_time_step(50, Some(2.0)), 50); // clamped above 1
                                                       // Monotone: higher strength ⇒ later (or equal) start.
        let starts: Vec<usize> = [0.1, 0.3, 0.5, 0.7, 0.9]
            .iter()
            .map(|&s| init_time_step(50, Some(s)))
            .collect();
        assert!(starts.windows(2).all(|w| w[0] <= w[1]), "{starts:?}");
    }

    /// `resolve_reference` pulls the single img2img init image + its effective strength from the
    /// request's conditioning (sc-8646): the per-reference strength wins over `req.strength`, a bare
    /// `Reference` falls back to `req.strength`, no `Reference` is `None`, and >1 `Reference` errors.
    /// Pure, no GPU.
    #[test]
    fn resolve_reference_picks_single_ref_and_strength() {
        use candle_gen::gen_core::{Conditioning, Image};
        let img = || Image {
            width: 8,
            height: 8,
            pixels: vec![0u8; 8 * 8 * 3],
        };

        // No conditioning ⇒ txt2img.
        let none = GenerationRequest::default();
        assert!(resolve_reference(&none).unwrap().is_none());

        // Per-reference strength wins over req.strength.
        let per_ref = GenerationRequest {
            strength: Some(0.2),
            conditioning: vec![Conditioning::Reference {
                image: img(),
                strength: Some(0.75),
            }],
            ..Default::default()
        };
        assert_eq!(resolve_reference(&per_ref).unwrap().unwrap().1, Some(0.75));

        // A bare Reference falls back to req.strength.
        let fallback = GenerationRequest {
            strength: Some(0.3),
            conditioning: vec![Conditioning::Reference {
                image: img(),
                strength: None,
            }],
            ..Default::default()
        };
        assert_eq!(resolve_reference(&fallback).unwrap().unwrap().1, Some(0.3));

        // More than one Reference is an error (single img2img init only).
        let two = GenerationRequest {
            conditioning: vec![
                Conditioning::Reference {
                    image: img(),
                    strength: None,
                },
                Conditioning::Reference {
                    image: img(),
                    strength: None,
                },
            ],
            ..Default::default()
        };
        assert!(resolve_reference(&two).is_err());
    }

    /// The img2img blend + reduced-schedule indices the base render loop reads (sc-8646), asserted on
    /// the static shift=6.0 σ table. At start `k`: the loop runs the `sigmas[k..]` tail (so
    /// `steps − k + 1` σ nodes / `steps − k` steps), σ_start = sigmas[k] ∈ (0,1) for interior `k`, and
    /// the flow-match interpolation `x_t = (1−σ)·clean + σ·noise` seeds the loop. Max strength (k=steps)
    /// ⇒ σ_start = 0 ⇒ x_t = clean and a single-node (0-step) tail: the source VAE round-trip. GPU-free.
    #[test]
    fn img2img_reduced_schedule_indices() {
        let steps = 50usize;
        let mut s = FlowMatchEulerDiscreteScheduler::new(base_scheduler_config());
        s.set_timesteps(steps, None);
        let sigmas: Vec<f32> = s.sigmas.iter().map(|&x| x as f32).collect();
        assert_eq!(sigmas.len(), steps + 1);

        // Default strength 0.6 → start 30; the tail runs sigmas[30..] (21 nodes, 20 steps).
        let start = init_time_step(steps, Some(0.6));
        assert_eq!(start, 30);
        let tail = &sigmas[start..];
        assert_eq!(tail.len(), steps - start + 1);
        assert!(
            tail[0] > 0.0 && tail[0] < 1.0,
            "σ_start in (0,1): {}",
            tail[0]
        );
        assert!(tail[tail.len() - 1].abs() < 1e-6, "tail ends at 0");

        // Max strength → start == steps → single-node tail (0 steps), σ_start == 0 ⇒ x_t == clean.
        let full = init_time_step(steps, Some(1.0));
        assert_eq!(full, steps);
        assert_eq!(sigmas[full..].len(), 1);
        assert!(sigmas[full].abs() < 1e-6);
    }
}
