//! # candle-gen-flux2
//!
//! The **FLUX.2** provider crate for [`candle-gen`](candle_gen) — the candle (Windows/CUDA) sibling of
//! `mlx-gen-flux2`. Unlike FLUX.1 (sc-3694), FLUX.2 has **no** `candle-transformers` reference: the
//! MMDiT transformer ([`transformer`]), the 32-channel 2×2-patchify VAE ([`vae`]), the decoder-LM
//! prompt-embeds text path ([`text_encoder`]), the 4-axis RoPE ([`pos_embed`]) and the flow-match
//! geometry ([`pipeline`]) are all ported here from the macOS provider.
//!
//! **Two txt2img variants** are registered, selected by [`config::Flux2Variant`]:
//! - **`flux2_klein_9b`** (sc-3695): Qwen3 TE (hidden states 9/18/27 → 12288-wide `prompt_embeds`) →
//!   the MMDiT (8 joint + 24 fused-single blocks) → the AutoencoderKL-Flux2 decoder. Distilled
//!   **4-step** flow-match Euler, CFG-free at guidance 1.0 (>1 runs a classifier-free negative pass).
//! - **`flux2_dev`** (epic 6564 sc-7457): the 32B flagship. **Mistral** TE (layers 10/20/30 →
//!   15360-wide `prompt_embeds`) → a wider/deeper MMDiT (8 joint + **48** single blocks, **48** heads,
//!   joint 15360). Guidance-**distilled** (embedded scalar, the FLUX.1-dev pattern): ~28 steps at
//!   guidance ~4 via a single forward feeding the DiT's guidance embedder — **not** true CFG.
//!
//! Same deterministic CPU-seeded-noise contract (sc-3673). Tokenization reuses gen-core's
//! [`TextTokenizer`]: klein with [`ChatTemplate::QwenInstructNoThink`], dev with
//! [`ChatTemplate::Flux2DevMistral`].
//!
//! **Sampling (epic 7114 P4, sc-7123):** both denoise loops (txt2img [`Pipeline::render`] and the edit
//! path [`Flux2Edit`]) route through the unified curated sampler/scheduler driver
//! (`candle_gen::run_flow_sampler` / `resolve_flow_schedule`). FLUX.2 is a rectified-flow engine using
//! the `Sigma` convention but embeds σ×1000, so the predict closure feeds `sigma * 1000.0` to the
//! transformer; the klein guidance>1 CFG blend / the dev embedded-guidance scalar (and, on the edit
//! path, the joint `[target, refs]` concat) live inside that closure. The descriptor advertises the
//! curated sampler/scheduler menus; the default (unset sampler/scheduler) path is the N1 no-op — euler
//! over the native empirical-mu flow-match schedule.
//!
//! **Surface:** txt2img for both variants (gen-core-registered). Conditioned dev surfaces are bespoke,
//! worker-invoked-by-name providers (the candle pattern, NOT registry entries): klein reference edit
//! [`Flux2Edit`] (sc-5487) — extended to **dev** multi-reference edit (sc-7460) via the DiT token
//! concat with the embedded-guidance forward — and dev strict-pose ControlNet [`Flux2Control`]
//! (sc-7460), the `FLUX.2-dev-Fun-Controlnet-Union` VACE branch. The dev conditioned paths run the
//! CPU-stage → quantize-onto-GPU loader ([`quant`]) so the 32B fits the memory ceiling. Still not
//! wired: the klein weight-variant edits (`flux2_klein_9b_kv_edit`) and LoRA/LoKr. `backend =
//! "candle"`, `mac_only = false`.

pub mod config;
pub mod control_provider;
pub mod convert;
pub mod edit_provider;
pub mod pipeline;
pub mod pos_embed;
pub mod quant;
pub mod text_encoder;
pub mod transformer;
pub mod vae;

pub use control_provider::{Flux2Control, Flux2ControlPaths, Flux2ControlRequest};
pub use convert::convert_and_assemble;
pub use edit_provider::{Flux2Edit, Flux2EditPaths, Flux2EditRequest};
pub use transformer::{
    Flux2ControlBranch, Flux2ControlTransformer, Flux2Transformer, CONTROL_IN_DIM,
};

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::{DType, Device, IndexOp, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::sampling::TimestepConvention;
use candle_gen::gen_core::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use candle_gen::gen_core::{
    self, Capabilities, GenerationOutput, GenerationRequest, Generator, Image, LoadSpec, Modality,
    ModelDescriptor, OffloadPolicy, PidWeights, Progress, Quant, WeightsSource,
};
use candle_gen::{CandleError, LatentDecoder, Result as CResult};
use candle_gen_pid::{PidDecoder, PidEngine};

use config::{Flux2Config, Flux2Variant, SIZE_MULTIPLE};
use text_encoder::Flux2PromptEncoder;
use vae::Flux2Vae;

/// The PiD backbone (latent-space) tag for FLUX.2 (epic 7840 / sc-7853): the `flux2` student consumes
/// the packed 128-channel BN-normalized latent at H/16 directly (the same tensor `decode_packed`
/// BN-de-normalizes). Lens reuses this same latent space (it shares the FLUX.2 VAE).
const PID_BACKBONE: &str = "flux2";

/// Qwen3 `<|endoftext|>` pad token id (klein FLUX.2 text encoder).
const QWEN_PAD_TOKEN_ID: i32 = 151643;
/// Mistral `<pad>` pad token id (dev FLUX.2 text encoder).
const MISTRAL_PAD_TOKEN_ID: i32 = 11;

/// The loaded FLUX.2 components, `Arc`-shared so the generator caches them across `generate` calls.
#[derive(Clone)]
struct Components {
    te: Arc<Flux2PromptEncoder>,
    transformer: Arc<Flux2Transformer>,
    vae: Arc<Flux2Vae>,
    /// Tokenizer (variant-specific pad token + chat template), loaded+parsed **once** at component load
    /// and reused across every prompt/branch encode (sc-8991 / F-011) instead of re-parsing
    /// `tokenizer.json` per request.
    tokenizer: Arc<TextTokenizer>,
    /// Optional NVIDIA PiD super-resolving decoder (epic 7840 / sc-7853), loaded once when the model
    /// was loaded with `LoadSpec::pid`. `None` ⇒ the native `Flux2Vae::decode_packed` (the default path).
    pid: Option<Arc<PidEngine>>,
}

/// A txt2img pipeline handle: snapshot root + device + the f32 compute dtype. `pub(crate)` so the
/// edit provider ([`edit_provider`]) reuses the snapshot mmap + prompt-encode scaffolding.
pub(crate) struct Pipeline {
    pub(crate) variant: Flux2Variant,
    pub(crate) cfg: Flux2Config,
    pub(crate) root: PathBuf,
    pub(crate) device: Device,
    pub(crate) dtype: DType,
    /// When `Some`, the quantizable components (TE + DiT) are staged dense in CPU RAM and quantized
    /// onto `device` (the dev 32B path; klein leaves this `None`).
    pub(crate) quant: Option<Quant>,
    /// The `LoadSpec::pid` component captured at load (epic 7840 / sc-7853), threaded into the lazy
    /// component build so the PiD engine loads once alongside the base model. `None` when not opted in.
    pub(crate) pid_spec: Option<PidWeights>,
    /// An in-place ComfyUI FLUX.2-dev fp8-mixed DiT single-file (epic 10451 Phase 2e, sc-10680). When
    /// `Some`, the transformer is built from this file (inline-scale fp8 dequant + BFL→diffusers remap,
    /// see [`convert::build_comfyui_dit_map`]) instead of the snapshot's `transformer/` dir; the text
    /// encoder / VAE / tokenizer still come from the resident snapshot `root`. `None` on every other
    /// path (registry txt2img, edit, control).
    pub(crate) comfyui_dit: Option<PathBuf>,
}

impl Pipeline {
    pub(crate) fn load(
        variant: Flux2Variant,
        quant: Option<Quant>,
        root: &Path,
        device: &Device,
        pid_spec: Option<PidWeights>,
    ) -> Self {
        Self {
            variant,
            cfg: variant.config(),
            root: root.to_path_buf(),
            device: device.clone(),
            // FLUX.2 runs the reference math in f32 (the TE + the MMDiT). The weights are large but
            // the math is parity-sensitive; a bf16 pass is a follow-up optimization.
            dtype: DType::F32,
            quant,
            pid_spec,
            comfyui_dit: None,
        }
    }

    /// Same as [`load`](Self::load) but sourcing the DiT from an in-place ComfyUI FLUX.2-dev fp8-mixed
    /// single-file (sc-10680). `root` is the resident FLUX.2-dev diffusers snapshot supplying the Mistral
    /// text encoder / VAE / tokenizer (the single DiT file carries none of those). `quant` (Q4/Q8) is
    /// honored for the DiT the same way the resident dev path is — the 32B does not fit the GPU dense
    /// after the fp8→f32 dequant, so each projection is folded onto the GPU. PiD is not wired here.
    pub(crate) fn load_comfyui(
        quant: Option<Quant>,
        root: &Path,
        device: &Device,
        comfyui_dit: PathBuf,
    ) -> Self {
        Self {
            variant: Flux2Variant::Dev,
            cfg: Flux2Variant::Dev.config(),
            root: root.to_path_buf(),
            device: device.clone(),
            dtype: DType::F32,
            quant,
            pid_spec: None,
            comfyui_dit: Some(comfyui_dit),
        }
    }

    /// mmap a VarBuilder over every `.safetensors` in the snapshot subdir `sub`, on `self.device`.
    pub(crate) fn component_vb(&self, sub: &str) -> CResult<VarBuilder<'static>> {
        self.component_vb_on(sub, &self.device)
    }

    /// [`Self::component_vb`] but on an explicit `device` — the quant path stages the TE + DiT on the
    /// CPU (system RAM) before quantizing onto the GPU, so the dense 32B never lands on the GPU.
    pub(crate) fn component_vb_on(
        &self,
        sub: &str,
        device: &Device,
    ) -> CResult<VarBuilder<'static>> {
        candle_gen::component_vb(&self.root, sub, self.dtype, device, "flux2")
    }

    /// Whether the snapshot component `sub/` is a **pre-quantized MLX-packed tier** — its `config.json`
    /// carries a `quantization` block (`candle_gen::quant::PackedConfig`), which an install-time convert
    /// job writes for a packed component. On a packed tier the loader builds each Linear/embedding
    /// **directly from the packed parts** on the GPU (sc-9087, no dense CPU staging); on a dense tier it
    /// falls back to the CPU-stage → quantize-onto-GPU path.
    ///
    /// A **genuinely-absent** `config.json` (file NotFound) is a legitimate dense/fixture snapshot shape
    /// → `Ok(false)` (dense path), so a single-file fixture with no `config.json` still loads. A config
    /// that **is present but corrupt** (I/O error, malformed JSON) is a damaged/partial download and
    /// errors loudly naming the file, rather than silently downgrading a packed component to the dense
    /// path (wrong tier / missing weights, no diagnostic). A well-formed config with no `quantization`
    /// block is simply a dense tier → `Ok(false)`. Mirrors the F-073 fix (sc-9010) in qwen-edit / krea.
    pub(crate) fn component_is_packed(&self, sub: &str) -> CResult<bool> {
        let path = self.root.join(sub).join("config.json");
        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            // No config.json at all → legitimate dense/fixture snapshot, not packed.
            Err(ref e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(false),
            // Present but unreadable (permissions, partial download) → surface, don't swallow.
            Err(e) => {
                return Err(CandleError::Msg(format!(
                    "flux2: read {}: {e}",
                    path.display()
                )))
            }
        };
        // Present but malformed JSON → corrupt snapshot, error rather than fall to dense.
        let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| {
            CandleError::Msg(format!(
                "flux2: parse {} (corrupt snapshot?): {e}",
                path.display()
            ))
        })?;
        Ok(candle_gen::quant::PackedConfig::from_config(&v).is_some())
    }

    /// Load the base **Mistral/Qwen3 TE + `Flux2Transformer` DiT** pair — the exact quantizable stack
    /// shared by every entry point (txt2img [`Self::load_components`], `Flux2Edit::load_variant`,
    /// `Flux2Control::load`). This is the single home for the "which builders + which tier/staging
    /// strategy" decision (F-024, sc-9004): it fixes the default builders (`Flux2PromptEncoder::new` /
    /// `Flux2Transformer::new`) and delegates the packed-vs-dense-vs-quant routing to
    /// [`Self::load_quantizable`]. Callers layer their extra components on top (the edit/control VAE
    /// *with encoder*, the control-branch overlay) — those are the genuine per-site differences and stay
    /// at the call site; only the copy-pasted TE+DiT loader moves here.
    ///
    /// A staging-strategy change (e.g. pre-quantized snapshot consumption) now lives in one place. Use
    /// [`Self::load_quantizable`] directly only if a future caller needs non-default module builders.
    pub(crate) fn load_te_and_dit(&self) -> CResult<(Flux2PromptEncoder, Flux2Transformer)> {
        self.load_quantizable(
            |cfg, vb| Ok(Flux2PromptEncoder::new(cfg, vb)?),
            |cfg, vb| Ok(Flux2Transformer::new(cfg, vb)?),
        )
    }

    /// Load ONLY the text encoder for the sequential-residency path (epic 10765 Phase 1c, sc-10868) —
    /// dropped right after the prompt encode so the decoder-LM TE (Mistral 24B on dev, Qwen3 on klein)
    /// frees before the DiT loads. Same per-tier routing (packed / dense+quant / dense) as the paired
    /// [`load_te_and_dit`](Self::load_te_and_dit) TE half; the DiT half is loaded separately (and later)
    /// via [`load_dit_seq`](Self::load_dit_seq).
    pub(crate) fn load_te_seq(&self) -> CResult<Flux2PromptEncoder> {
        self.load_one_quantizable(
            "text_encoder",
            |vb| Ok(Flux2PromptEncoder::new(&self.cfg, vb)?),
            |m, q, d| Ok(m.quantize(q, d)?),
        )
    }

    /// Load ONLY the DiT for the sequential path (sc-10868) — loaded after the text encoder was dropped,
    /// so it reuses the TE's freed allocator pool (capping peak at DiT+VAE, not TE+DiT+VAE). Same per-tier
    /// routing as the paired [`load_te_and_dit`](Self::load_te_and_dit) DiT half.
    pub(crate) fn load_dit_seq(&self) -> CResult<Flux2Transformer> {
        self.load_one_quantizable(
            "transformer",
            |vb| Ok(Flux2Transformer::new(&self.cfg, vb)?),
            |m, q, d| Ok(m.quantize(q, d)?),
        )
    }

    /// Load the TE + DiT, routing each through the **packed** path (build straight from an MLX-packed
    /// tier on the GPU — sc-9087, no ~105 GB dense CPU staging) or the legacy **dense** path (stage
    /// dense in system RAM, then quantize each projection onto the GPU) per [`Self::component_is_packed`]
    /// and `self.quant`. Shared by txt2img, `Flux2Edit::load_dev`, and `Flux2Control` (they load the same
    /// quantizable pair; the callers add the VAE / control overlay) via [`Self::load_te_and_dit`], which
    /// fixes the default builders. `mk_te` / `mk_dit` build the module from a component VarBuilder
    /// (`Flux2PromptEncoder::new` / `Flux2Transformer::new`).
    pub(crate) fn load_quantizable(
        &self,
        mk_te: impl Fn(&Flux2Config, VarBuilder) -> CResult<Flux2PromptEncoder>,
        mk_dit: impl Fn(&Flux2Config, VarBuilder) -> CResult<Flux2Transformer>,
    ) -> CResult<(Flux2PromptEncoder, Flux2Transformer)> {
        let te = self.load_one_quantizable(
            "text_encoder",
            |vb| mk_te(&self.cfg, vb),
            |m, q, d| Ok(m.quantize(q, d)?),
        )?;
        let dit = self.load_one_quantizable(
            "transformer",
            |vb| mk_dit(&self.cfg, vb),
            |m, q, d| Ok(m.quantize(q, d)?),
        )?;
        Ok((te, dit))
    }

    /// Load one quantizable component (`sub`). Three regimes:
    /// - **packed tier + quant**: build directly on the GPU from the packed parts (`.scales` detected
    ///   inside each `linear_detect`); no dense weight is ever materialized (sc-9087). The post-load
    ///   `quantize` pass is still called — it is a no-op on the already-packed projections and only
    ///   carries the dense leaves (RMSNorms, a dense token embedding) to the GPU.
    /// - **dense tier + quant**: stage dense in CPU RAM, then `quantize` folds each projection onto the
    ///   GPU (the legacy ~105 GB path, retained for dense tiers / large fixtures).
    /// - **no quant**: load dense on-device (klein, small dev fixtures).
    fn load_one_quantizable<M>(
        &self,
        sub: &str,
        build: impl FnOnce(VarBuilder) -> CResult<M>,
        quantize: impl FnOnce(&mut M, Quant, &Device) -> CResult<()>,
    ) -> CResult<M> {
        match self.quant {
            Some(q) if self.component_is_packed(sub)? => {
                // Build straight on the GPU from the packed tier — the packed footprint (≈ Q4: ¼ bf16)
                // lands directly; no dense staging.
                let mut m = build(self.component_vb_on(sub, &self.device)?)?;
                // No-op on the packed projections; moves the dense leaves to the GPU.
                quantize(&mut m, q, &self.device)?;
                Ok(m)
            }
            Some(q) => {
                // Dense tier: stage dense in CPU RAM, then quantize each projection onto the GPU.
                let mut m = build(self.component_vb_on(sub, &Device::Cpu)?)?;
                quantize(&mut m, q, &self.device)?;
                Ok(m)
            }
            None => build(self.component_vb(sub)?),
        }
    }

    /// Build the DiT from an in-place ComfyUI FLUX.2-dev fp8-mixed single-file (sc-10680): dequant the
    /// inline-scale fp8 MLPs + remap the BFL keys into an in-memory map ([`convert::build_comfyui_dit_map`])
    /// at the compute dtype (f32), then route by `self.quant` exactly as [`load_one_quantizable`]'s
    /// dense-tier regime does — the snapshot `transformer/` dir simply replaced by the single file:
    /// - **quant** (the 32B dev path): stage the dense f32 DiT in CPU RAM, then fold each projection onto
    ///   the GPU (`quantize`); the dense f32 32B never lands on the GPU (it would not fit).
    /// - **no quant** (small fixtures only): build dense on-device.
    fn load_comfyui_dit(&self, dit_file: &Path) -> CResult<Flux2Transformer> {
        // SAFETY: read-only mmap of a weight file; the standard candle loading path.
        let mmap =
            unsafe { candle_gen::candle_core::safetensors::MmapedSafetensors::new(dit_file) }
                .map_err(|e| {
                    CandleError::Msg(format!("flux2 comfyui: mmap {}: {e}", dit_file.display()))
                })?;
        let map = convert::build_comfyui_dit_map(&mmap, self.dtype)?;
        match self.quant {
            Some(q) => {
                let vb = VarBuilder::from_tensors(map, self.dtype, &Device::Cpu);
                let mut dit = Flux2Transformer::new(&self.cfg, vb)?;
                dit.quantize(q, &self.device)?;
                Ok(dit)
            }
            None => {
                let vb = VarBuilder::from_tensors(map, self.dtype, &self.device);
                Ok(Flux2Transformer::new(&self.cfg, vb)?)
            }
        }
    }

    fn load_components(&self) -> CResult<Components> {
        let (te, transformer) = match &self.comfyui_dit {
            // In-place ComfyUI DiT (sc-10680): the Mistral TE is NOT in the single DiT file, so it comes
            // from the snapshot through the same per-tier quant path (`load_te_seq` is the TE-only
            // quantizable loader); the DiT is dequanted + quantized from the in-place file.
            Some(dit_file) => (self.load_te_seq()?, self.load_comfyui_dit(dit_file)?),
            None => self.load_te_and_dit()?,
        };
        let vae = Flux2Vae::new(self.component_vb("vae")?)?;
        let tokenizer = self.build_tokenizer()?;
        // Load the optional PiD super-resolving decoder once (epic 7840 / sc-7853) when the caller
        // opted in via `LoadSpec::pid`; otherwise `None` and the render path uses the native Flux2Vae.
        let pid = match self.pid_spec.as_ref() {
            Some(spec) => Some(Arc::new(PidEngine::from_spec(
                spec,
                PID_BACKBONE,
                &self.device,
            )?)),
            None => None,
        };
        Ok(Components {
            te: Arc::new(te),
            transformer: Arc::new(transformer),
            vae: Arc::new(vae),
            tokenizer: Arc::new(tokenizer),
            pid,
        })
    }

    /// Build the prompt tokenizer **once** (sc-8991 / F-011). The config (pad token + chat template) is
    /// variant-specific: klein uses the Qwen2 `<|endoftext|>` pad + the Qwen no-think chat template; dev
    /// uses the Mistral `<pad>` + the `[INST]…[/INST]` template. Callers cache the result on their
    /// `Components` / provider struct and reuse it across encodes rather than re-parsing per prompt.
    pub(crate) fn build_tokenizer(&self) -> CResult<TextTokenizer> {
        let (pad_token_id, chat_template) = if self.variant.is_dev() {
            (MISTRAL_PAD_TOKEN_ID, ChatTemplate::Flux2DevMistral)
        } else {
            (QWEN_PAD_TOKEN_ID, ChatTemplate::QwenInstructNoThink)
        };
        TextTokenizer::from_file(
            self.root.join("tokenizer/tokenizer.json"),
            TokenizerConfig {
                max_length: self.cfg.max_sequence_length,
                pad_token_id,
                chat_template,
                pad_to_max_length: true,
            },
        )
        .map_err(|e| CandleError::Msg(format!("flux2: load tokenizer: {e}")))
    }

    /// Tokenize + encode the prompt to `prompt_embeds` `[1, 512, 3·hidden]` (f32). `tok` is the cached
    /// tokenizer ([`Self::build_tokenizer`]) — parsed once, reused across encodes (sc-8991 / F-011).
    pub(crate) fn encode(
        &self,
        te: &Flux2PromptEncoder,
        tok: &TextTokenizer,
        prompt: &str,
    ) -> CResult<Tensor> {
        let out = tok
            .tokenize(prompt)
            .map_err(|e| CandleError::Msg(format!("flux2: tokenize: {e}")))?;
        let len = out.ids.len();
        let ids: Vec<u32> = out.ids.iter().map(|&i| i as u32).collect();
        let mask: Vec<i64> = out.mask.iter().map(|&m| m as i64).collect();
        let input_ids = Tensor::from_vec(ids, (1, len), &self.device)?;
        let attn_mask = Tensor::from_vec(mask, (1, len), &self.device)?;
        Ok(te.prompt_embeds(&input_ids, &attn_mask)?)
    }

    fn render(
        &self,
        req: &GenerationRequest,
        comps: &Components,
        on_progress: &mut dyn FnMut(Progress),
    ) -> CResult<Vec<Image>> {
        let steps = req
            .steps
            .map(|s| s as usize)
            .unwrap_or(self.variant.default_steps() as usize);
        let base_seed = req.seed.unwrap_or_else(gen_core::default_seed);
        let guidance = req.guidance.unwrap_or(self.variant.default_guidance());

        // Prompt embeds are seed-independent: encode once.
        let prompt_embeds = self.encode(&comps.te, &comps.tokenizer, &req.prompt)?;
        // Two guidance regimes. dev is guidance-distilled: a single forward feeds the guidance scalar
        // to the DiT's embedded-guidance embedder (no negative pass). klein is distilled / true-CFG:
        // a classifier-free negative pass only when guidance > 1 (it runs CFG-free at 1.0).
        let negative = self.encode_negative(&comps.te, &comps.tokenizer, req, guidance)?;

        // Resolve the decode seam once for the whole batch (epic 7840 / sc-7853): a per-generation PiD
        // decoder bound to this prompt when `req.use_pid` is set (errors if requested but not loaded),
        // else `None` → the native Flux2Vae decode. Shared across `count` images (same prompt).
        let pid_decoder = candle_gen_pid::resolve_pid_decoder(
            comps.pid.as_deref(),
            req,
            base_seed,
            self.variant.id(),
        )?;

        self.sample(
            req,
            &comps.transformer,
            &comps.vae,
            &prompt_embeds,
            negative.as_ref(),
            pid_decoder.as_ref(),
            guidance,
            steps,
            base_seed,
            on_progress,
        )
    }

    /// Encode the optional classifier-free **negative** prompt for the klein CFG blend: `Some` only on a
    /// non-embedded-guidance variant with `guidance > 1` (klein runs CFG-free at 1.0; dev is embedded-
    /// guidance, single-forward, so always `None`). Takes the TE + tokenizer directly so both the
    /// resident [`render`] (cached components) and the sequential [`render_sequential`] (a just-loaded,
    /// about-to-be-dropped TE) share the exact CFG condition — the conditioning stays byte-identical.
    fn encode_negative(
        &self,
        te: &Flux2PromptEncoder,
        tok: &TextTokenizer,
        req: &GenerationRequest,
        guidance: f32,
    ) -> CResult<Option<Tensor>> {
        if !self.variant.uses_embedded_guidance() && guidance > 1.0 {
            let neg = req.negative_prompt.as_deref().unwrap_or(" ");
            Ok(Some(self.encode(te, tok, neg)?))
        } else {
            Ok(None)
        }
    }

    /// The per-image denoise + decode loop shared by the resident [`render`] and the sequential
    /// [`render_sequential`] (epic 10765 Phase 1c, sc-10868). Given the already-encoded `prompt_embeds`
    /// (+ optional klein CFG `negative`), a borrowed DiT + VAE, and the resolved PiD seam, the sampled
    /// output is **byte-identical** across both residency modes — only the load/free schedule of the
    /// components handed in differs.
    #[allow(clippy::too_many_arguments)]
    fn sample(
        &self,
        req: &GenerationRequest,
        transformer: &Flux2Transformer,
        vae: &Flux2Vae,
        prompt_embeds: &Tensor,
        negative: Option<&Tensor>,
        pid_decoder: Option<&PidDecoder>,
        guidance: f32,
        steps: usize,
        base_seed: u64,
        on_progress: &mut dyn FnMut(Progress),
    ) -> CResult<Vec<Image>> {
        let embedded_guidance = self.variant.uses_embedded_guidance();
        let (lat_h, lat_w) = pipeline::latent_dims(req.width, req.height);
        let img_ids = pipeline::prepare_grid_ids(lat_h, lat_w);
        let txt_ids = pipeline::prepare_text_ids(self.cfg.max_sequence_length);

        // Curated sampler/scheduler routing (epic 7114 P4, sc-7123). The NATIVE schedule is the legacy
        // empirical-mu flow-match sigmas (descending, trailing 0.0); the same `mu` feeds the curated
        // scheduler axis so `normal`/`karras`/etc. honor the resolution-dependent shift. The default path
        // (sampler/scheduler unset) is the N1 no-op — euler over the native schedule reproduces the legacy
        // `euler_step` flow-match loop within tolerance.
        let mu = pipeline::compute_mu(pipeline::image_seq_len(req.width, req.height), steps);
        let native = pipeline::schedule(steps, req.width, req.height);
        let sigmas =
            candle_gen::resolve_flow_schedule(req.scheduler.as_deref(), mu, steps, &native);

        candle_gen::for_each_image_seed(base_seed, req.count, |seed| {
            let latents =
                pipeline::create_noise(&self.cfg, seed, req.width, req.height, &self.device)?;

            // The driver does cancel + progress + the euler/curated integrator step. The forward (and the
            // guidance>1 CFG blend) lives inside `predict` so a multi-eval solver re-runs it. FLUX.2 uses
            // the Sigma convention but the model embeds σ×1000, so feed `sigma * 1000.0` to the transformer.
            let latents = candle_gen::run_flow_sampler(
                req.sampler.as_deref(),
                TimestepConvention::Sigma,
                &sigmas,
                latents,
                seed,
                &req.cancel,
                on_progress,
                |latents, sigma| -> CResult<Tensor> {
                    let ts = sigma * 1000.0;
                    let out = if embedded_guidance {
                        // dev: single forward feeding the embedded guidance scalar to the DiT.
                        transformer.forward(
                            latents,
                            prompt_embeds,
                            &img_ids,
                            &txt_ids,
                            ts,
                            Some(guidance),
                        )?
                    } else {
                        let v = transformer.forward(
                            latents,
                            prompt_embeds,
                            &img_ids,
                            &txt_ids,
                            ts,
                            None,
                        )?;
                        match negative {
                            Some(neg) => {
                                let vn = transformer
                                    .forward(latents, neg, &img_ids, &txt_ids, ts, None)?;
                                // vn + guidance·(v − vn)
                                (&vn + ((&v - &vn)? * guidance as f64)?)?
                            }
                            None => v,
                        }
                    };
                    Ok(out)
                },
            )?;

            on_progress(Progress::Decoding);
            let packed = pipeline::unpack_latents(&latents, req.width, req.height)?;
            let decoded = match pid_decoder {
                // PiD consumes the packed BN-normalized [1,128,H/16,W/16] latent directly (the same
                // tensor decode_packed BN-de-normalizes); returns [1,3,4H,4W].
                Some(pid) => pid.decode(&packed)?,
                None => vae.decode_packed(&packed)?, // [1,3,H,W] in [-1,1]
            };
            to_image(&decoded)
        })
    }

    /// Sequential-residency render (epic 10765 Phase 1c, sc-10868): load the text encoder → encode the
    /// prompt(s) → DROP it → load the DiT + VAE → denoise/decode. Peak VRAM is bounded to the DiT+VAE
    /// working set instead of TE+DiT+VAE (reclaiming the decoder-LM TE — the largest such win off-Mac on
    /// the 32B **dev**, where the Mistral TE is multiple GB), so a card that OOMs the resident path can
    /// still render. Output is **bit-identical** to [`render`](Self::render) — the SAME encode ([`encode`]
    /// / [`encode_negative`]), the shared [`sample`](Self::sample) denoise+decode loop; only the load/free
    /// schedule differs.
    ///
    /// Selected by the generator when [`sequential_offload_enabled`] (`CANDLE_GEN_OFFLOAD=sequential`) or
    /// `LoadSpec::offload_policy == Sequential` (the worker fit-gate sets it). Because it drops components,
    /// it does NOT populate the generator's `Components` cache — repeat requests reload from the (page-
    /// cached) snapshot; that reload cost is the deliberate trade for the lower peak, which is why it is
    /// opt-in per the fit-gate rather than the default.
    fn render_sequential(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> CResult<Vec<Image>> {
        let steps = req
            .steps
            .map(|s| s as usize)
            .unwrap_or(self.variant.default_steps() as usize);
        let base_seed = req.seed.unwrap_or_else(gen_core::default_seed);
        let guidance = req.guidance.unwrap_or(self.variant.default_guidance());

        // Phase 1 — load ONLY the text encoder (+ its cheap tokenizer), encode the prompt (+ the klein
        // CFG negative), then DROP both (scoped) so the decoder-LM TE frees before the DiT loads. The
        // encode delegates to the SAME `encode`/`encode_negative` the resident path uses, so the
        // conditioning tensors are byte-identical to `render`.
        let (prompt_embeds, negative) = {
            let te = self.load_te_seq()?;
            let tokenizer = self.build_tokenizer()?;
            let prompt_embeds = self.encode(&te, &tokenizer, &req.prompt)?;
            let negative = self.encode_negative(&te, &tokenizer, req, guidance)?;
            (prompt_embeds, negative)
        };

        // Phase 2 — load the DiT (reusing the TE's freed pool) + the VAE + the optional PiD decoder.
        let transformer = self.load_dit_seq()?;
        let vae = Flux2Vae::new(self.component_vb("vae")?)?;
        let pid = match self.pid_spec.as_ref() {
            Some(spec) => Some(PidEngine::from_spec(spec, PID_BACKBONE, &self.device)?),
            None => None,
        };
        let pid_decoder =
            candle_gen_pid::resolve_pid_decoder(pid.as_ref(), req, base_seed, self.variant.id())?;

        // Phase 3 — per-image denoise + decode, identical to `render`'s loop.
        self.sample(
            req,
            &transformer,
            &vae,
            &prompt_embeds,
            negative.as_ref(),
            pid_decoder.as_ref(),
            guidance,
            steps,
            base_seed,
            on_progress,
        )
    }
}

/// Whether the sequential-residency offload path is enabled (epic 10765 Phase 1c, sc-10868). Reads the
/// process-wide `CANDLE_GEN_OFFLOAD` env (shared with the candle FLUX.1 lane, sc-10769): `sequential`
/// (case-insensitive) selects the phased load/free path; unset or any other value keeps the resident,
/// cross-request-cached default. The worker's fit-gate drives the same choice per-load via
/// `LoadSpec::offload_policy`; this env toggle is the GPU A/B harness seam.
pub(crate) fn sequential_offload_enabled() -> bool {
    std::env::var("CANDLE_GEN_OFFLOAD")
        .map(|value| value.trim().eq_ignore_ascii_case("sequential"))
        .unwrap_or(false)
}

/// Map a decoded `[1, 3, H, W]` tensor in `[-1, 1]` to an RGB8 [`Image`].
pub(crate) fn to_image(decoded: &Tensor) -> CResult<Image> {
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

/// A loaded candle FLUX.2 generator. Loading is lazy; components build on the first `generate` and
/// are cached. `variant` selects klein vs dev (config, text encoder, tokenizer, guidance regime).
pub struct Flux2Generator {
    variant: Flux2Variant,
    descriptor: ModelDescriptor,
    root: PathBuf,
    device: Device,
    /// `Some` ⇒ CPU-stage → quantize-onto-GPU at load (dev Q4/Q8); `None` ⇒ dense.
    quant: Option<Quant>,
    /// The `LoadSpec::pid` component captured at load (epic 7840 / sc-7853), threaded into the lazy
    /// component build so the PiD engine loads once alongside the base model. `None` when not opted in.
    pid_spec: Option<PidWeights>,
    /// Component-residency policy captured from `LoadSpec::offload_policy` (epic 10765 Phase 1c,
    /// sc-10868). `Sequential` routes `generate` through [`Pipeline::render_sequential`] (load→encode→drop
    /// the text encoder before the DiT), capping peak VRAM at the cost of the components cache; `Resident`
    /// (default) keeps the cached path. The worker's fit-gate sets this when it predicts the resident sum
    /// won't fit but the DiT+VAE working set will.
    offload_policy: OffloadPolicy,
    /// An in-place ComfyUI FLUX.2-dev fp8-mixed DiT single-file (epic 10451 Phase 2e, sc-10680), set
    /// only by [`load_from_comfyui_dit`]. When `Some`, the lazy component build sources the transformer
    /// from this file (inline-scale fp8 dequant + BFL→diffusers remap) and the TE / VAE / tokenizer from
    /// [`Self::root`] (a resident FLUX.2-dev snapshot); `None` on the registry path.
    comfyui_dit: Option<PathBuf>,
    components: Mutex<Option<Components>>,
}

impl Flux2Generator {
    fn components(&self, pipe: &Pipeline) -> gen_core::Result<Components> {
        // `?` bridges the candle-side `load_components` error into `gen_core::Error`.
        Ok(candle_gen::cached(&self.components, || {
            pipe.load_components()
        })?)
    }
}

impl Generator for Flux2Generator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        let id = self.descriptor.id;
        self.descriptor.capabilities.validate_request(id, req)?;
        if req.prompt.is_empty() {
            return Err(gen_core::Error::Msg(format!(
                "{id}: prompt must not be empty"
            )));
        }
        if req.steps == Some(0) {
            return Err(gen_core::Error::Msg(format!(
                "{id}: steps must be >= 1 (an explicit 0 renders undenoised noise)"
            )));
        }
        if !req.width.is_multiple_of(SIZE_MULTIPLE) || !req.height.is_multiple_of(SIZE_MULTIPLE) {
            return Err(gen_core::Error::Msg(format!(
                "{id}: width/height must be multiples of {SIZE_MULTIPLE} (got {}x{})",
                req.width, req.height
            )));
        }
        Ok(())
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.validate(req)?;
        let pipe = match &self.comfyui_dit {
            Some(dit_file) => {
                Pipeline::load_comfyui(self.quant, &self.root, &self.device, dit_file.clone())
            }
            None => Pipeline::load(
                self.variant,
                self.quant,
                &self.root,
                &self.device,
                self.pid_spec.clone(),
            ),
        };
        // Sequential-residency offload (epic 10765, sc-10868): when selected, load→encode→drop the text
        // encoder before loading the DiT so peak VRAM is DiT+VAE, not TE+DiT+VAE — letting a card that
        // OOMs the resident path render. Output is bit-identical; it bypasses the components cache (it
        // drops what it loads). Driven by `LoadSpec::offload_policy` (the worker fit-gate sets
        // `Sequential`); `CANDLE_GEN_OFFLOAD=sequential` is an env override kept for the GPU A/B harness.
        // Never taken on the in-place ComfyUI DiT lane — that transformer is a single file, not
        // `root/transformer/`, so the sequential per-phase loaders can't source it (falls back to
        // resident). The default stays the resident, cross-request-cached path.
        let sequential = self.comfyui_dit.is_none()
            && (self.offload_policy == OffloadPolicy::Sequential || sequential_offload_enabled());
        let images = if sequential {
            pipe.render_sequential(req, on_progress)?
        } else {
            let components = self.components(&pipe)?;
            pipe.render(req, &components, on_progress)?
        };
        Ok(GenerationOutput::Images(images))
    }
}

/// The txt2img descriptor for `variant`. **klein**: guidance advertised (defaults to 1.0 / CFG-free,
/// but >1.0 runs a classifier-free negative pass), so `supports_negative_prompt`. **dev**: guidance is
/// the embedded scalar (single forward, no negative pass), so `supports_negative_prompt = false`.
/// Both: txt2img only (edit/Reference deferred to epic 6564 story 4), no LoRA, no on-the-fly quant.
fn descriptor(variant: Flux2Variant) -> ModelDescriptor {
    ModelDescriptor {
        id: variant.id(),
        family: "flux2",
        backend: "candle",
        modality: Modality::Image,
        capabilities: Capabilities {
            // dev is guidance-distilled (embedded scalar, no negative pass); klein runs a
            // classifier-free negative pass when guidance > 1.
            supports_negative_prompt: !variant.uses_embedded_guidance(),
            supports_guidance: true,
            supports_true_cfg: false,
            // txt2img only in this slice — the mlx edit/Reference surface is deferred.
            conditioning: vec![],
            supports_lora: false,
            supports_lokr: false,
            // Curated sampler/scheduler menu (epic 7114 P4, sc-7123). The legacy `flow_match_euler`
            // scheduler alias is retained and falls back to the native schedule via the N3 path.
            samplers: candle_gen::curated_sampler_names(),
            schedulers: candle_gen::menu_with_aliases(
                candle_gen::curated_scheduler_names(),
                &["flow_match_euler"],
            ),
            supported_guidance_methods: vec![],
            min_size: 256,
            max_size: 2048,
            max_count: 8,
            mac_only: false,
            // dev quantizes (CPU-stage → quantize-onto-GPU) to fit the 32B under the memory ceiling;
            // klein is small and runs dense.
            supported_quants: if variant.is_dev() {
                &[Quant::Q4, Quant::Q8]
            } else {
                &[] as &[Quant]
            },
            supports_kv_cache: false,
            // FLUX.2 uses the empirical-mu shifted flow-match schedule.
            requires_sigma_shift: true,
        },
    }
}

/// FLUX.2-klein-9b txt2img descriptor (the surface sc-3695 wired).
pub fn descriptor_klein() -> ModelDescriptor {
    descriptor(Flux2Variant::Klein9b)
}

/// FLUX.2-dev txt2img descriptor (epic 6564 story 1): the guidance-distilled 32B flagship.
pub fn descriptor_dev() -> ModelDescriptor {
    descriptor(Flux2Variant::Dev)
}

/// Construct a lazy candle FLUX.2 generator for `variant`. `spec.weights` must be a
/// [`WeightsSource::Dir`] pointing at a diffusers snapshot (`text_encoder/`, `transformer/`, `vae/`,
/// `tokenizer/`) — klein at `black-forest-labs/FLUX.2-klein-9B`, dev at `black-forest-labs/FLUX.2-dev`
/// (whose `text_encoder/` is the Mistral3 checkpoint). Adapters / control overlays are rejected (not
/// wired). `spec.quantize` (Q4/Q8) is honored for **dev** — the 32B is staged dense in CPU RAM and
/// quantized onto the GPU at load (it does not fit the GPU dense); for **klein** quantization is not
/// wired and is rejected. dev without quant loads dense (fixture-only — the full 32B needs the quant).
fn load_variant(variant: Flux2Variant, spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    let id = variant.id();
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(format!(
                "{id} expects a snapshot directory (text_encoder/ transformer/ vae/ tokenizer/), \
                 not a single .safetensors file"
            )));
        }
    };
    if !spec.adapters.is_empty() {
        return Err(gen_core::Error::Unsupported(format!(
            "candle {id} does not support LoRA/LoKr yet"
        )));
    }
    // dev honors Q4/Q8 (CPU-stage → quantize-onto-GPU); klein has no candle quant path yet.
    let quant = if variant.is_dev() {
        spec.quantize
    } else if spec.quantize.is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "candle {id} does not support on-the-fly Q4/Q8 quantization yet"
        )));
    } else {
        None
    };
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "candle {id} does not support control / IP-adapter / edit yet (txt2img only)"
        )));
    }
    let device = candle_gen::default_device()?;
    Ok(Box::new(Flux2Generator {
        variant,
        descriptor: descriptor(variant),
        root,
        device,
        quant,
        // PiD is an optional aux decoder (epic 7840 / sc-7853): capture the load-spec component (if
        // any) so the lazy component build loads the engine once. Unlike adapters/control above, it is
        // not rejected — `None` simply keeps the byte-exact native-VAE path.
        pid_spec: spec.pid.clone(),
        // Component-residency policy (epic 10765 Phase 1c, sc-10868) — `Sequential` routes generate
        // through `render_sequential`. Captured at load; the resident default is unchanged.
        offload_policy: spec.offload_policy,
        comfyui_dit: None,
        components: Mutex::new(None),
    }))
}

/// Construct a lazy candle FLUX.2-**dev** generator that reads its **DiT** in place from an existing
/// ComfyUI fp8-mixed single-file (epic 10451 Phase 2e, sc-10680) — no copy, no re-download.
/// `transformer_file` is the user's `diffusion_models/flux2_dev_fp8mixed.safetensors` (BFL-native keys,
/// inline-scale fp8 MLPs); its keys are remapped + its fp8 weights dequanted (`w = w_fp8·weight_scale`)
/// in memory at component build ([`convert::build_comfyui_dit_map`]). `snapshot_dir` is a resident
/// FLUX.2-dev diffusers snapshot supplying the Mistral text encoder, VAE, and tokenizer (none of which
/// are in the single DiT file). `quant` (Q4/Q8) folds the dequanted DiT + the Mistral TE onto the GPU —
/// the 32B dev does not fit dense — matching the resident dev path; `None` is fixture-only. txt2img
/// only; no adapters / control / edit / PiD.
pub fn load_from_comfyui_dit(
    transformer_file: impl Into<PathBuf>,
    snapshot_dir: impl Into<PathBuf>,
    quant: Option<Quant>,
) -> gen_core::Result<Box<dyn Generator>> {
    let device = candle_gen::default_device()?;
    Ok(Box::new(Flux2Generator {
        variant: Flux2Variant::Dev,
        descriptor: descriptor(Flux2Variant::Dev),
        root: snapshot_dir.into(),
        device,
        quant,
        pid_spec: None,
        // The in-place ComfyUI DiT lane keeps everything resident (its transformer is a single file, not
        // `root/transformer/`, so the sequential per-phase loaders don't apply).
        offload_policy: OffloadPolicy::Resident,
        comfyui_dit: Some(transformer_file.into()),
        components: Mutex::new(None),
    }))
}

/// Registry load hook for `flux2_klein_9b`.
pub fn load_klein(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_variant(Flux2Variant::Klein9b, spec)
}

/// Registry load hook for `flux2_dev`.
pub fn load_dev(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_variant(Flux2Variant::Dev, spec)
}

// Link-time self-registration into gen-core's model registry — one per txt2img variant.
candle_gen::register_generators! {
    descriptor_klein => load_klein,
    descriptor_dev => load_dev,
}

/// Force-link hook (keeps the `inventory::submit!` registration from being dead-stripped when the
/// crate is reached only through the registry). Same pattern as the other providers.
pub fn force_link() {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{FLUX2_DEV_ID, FLUX2_KLEIN_9B_ID};
    use candle_gen::gen_core::registry;
    use candle_gen::gen_core::ConditioningKind;

    #[test]
    fn registers_and_resolves_as_candle() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry::load(FLUX2_KLEIN_9B_ID, &spec).expect("flux2 is registered");
        assert_eq!(g.descriptor().id, FLUX2_KLEIN_9B_ID);
        assert_eq!(g.descriptor().family, "flux2");
        assert_eq!(g.descriptor().backend, "candle");
        assert_eq!(g.descriptor().modality, Modality::Image);
    }

    #[test]
    fn klein_descriptor_advertises_only_wired_txt2img_surface() {
        let d = descriptor_klein();
        assert_eq!(d.id, FLUX2_KLEIN_9B_ID);
        assert!(d.capabilities.supports_guidance);
        // klein runs a classifier-free negative pass when guidance > 1.
        assert!(d.capabilities.supports_negative_prompt);
        assert!(d.capabilities.requires_sigma_shift);
        assert!(!d.capabilities.mac_only);
        assert!(d.capabilities.conditioning.is_empty());
        assert!(!d.capabilities.supports_lora);
        assert!(!d.capabilities.supports_kv_cache);
        assert!(d.capabilities.supported_quants.is_empty());
        assert!(!d.capabilities.accepts(ConditioningKind::Reference));
    }

    #[test]
    fn dev_registers_and_advertises_embedded_guidance_surface() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry::load(FLUX2_DEV_ID, &spec).expect("flux2_dev is registered");
        assert_eq!(g.descriptor().id, FLUX2_DEV_ID);
        assert_eq!(g.descriptor().family, "flux2");
        assert_eq!(g.descriptor().backend, "candle");
        assert_eq!(g.descriptor().modality, Modality::Image);
        let d = descriptor_dev();
        assert!(d.capabilities.supports_guidance);
        // dev is guidance-distilled (embedded scalar) — no negative pass, no true-CFG, not mac-only.
        assert!(!d.capabilities.supports_negative_prompt);
        assert!(!d.capabilities.supports_true_cfg);
        assert!(!d.capabilities.mac_only);
        assert!(d.capabilities.conditioning.is_empty());
        assert!(d.capabilities.requires_sigma_shift);
        // dev advertises Q4/Q8 (CPU-stage → quantize-onto-GPU); klein advertises none.
        assert_eq!(d.capabilities.supported_quants, &[Quant::Q4, Quant::Q8]);
        assert!(descriptor_klein().capabilities.supported_quants.is_empty());
    }

    #[test]
    fn validate_accepts_txt2img_and_rejects_unsupported() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry::load(FLUX2_KLEIN_9B_ID, &spec).unwrap();
        let ok = GenerationRequest {
            prompt: "a rusty robot holding a lit candle".into(),
            ..Default::default()
        };
        assert!(g.validate(&ok).is_ok());
        for bad in [
            GenerationRequest::default(),
            GenerationRequest {
                prompt: "x".into(),
                width: 1000,
                ..Default::default()
            },
            GenerationRequest {
                prompt: "x".into(),
                steps: Some(0),
                ..Default::default()
            },
        ] {
            assert!(g.validate(&bad).is_err(), "should reject: {bad:?}");
        }
    }

    #[test]
    fn load_rejects_unwired_surfaces() {
        use candle_gen::gen_core::{AdapterKind, AdapterSpec};
        let lora = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_adapters(vec![
            AdapterSpec::new("/lora.safetensors".into(), 1.0, AdapterKind::Lora),
        ]);
        assert!(matches!(
            load_klein(&lora).err().expect("err"),
            gen_core::Error::Unsupported(_)
        ));
        let quant = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_quant(Quant::Q8);
        assert!(matches!(
            load_klein(&quant).err().expect("err"),
            gen_core::Error::Unsupported(_)
        ));
        // klein has no candle quant path — on-the-fly quant is rejected.
        let klein_quant = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_quant(Quant::Q4);
        assert!(matches!(
            load_klein(&klein_quant).err().expect("err"),
            gen_core::Error::Unsupported(_)
        ));
        // dev DOES accept Q4/Q8 (CPU-stage → quantize-onto-GPU); the generator builds lazily, so this
        // succeeds without touching the (nonexistent) weights.
        let dev_quant = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_quant(Quant::Q4);
        assert!(load_dev(&dev_quant).is_ok());
    }

    /// The loader's packed/dense routing decision (sc-9087): a component whose `config.json` carries a
    /// `quantization` block is a packed MLX tier (build directly on the GPU, no dense CPU staging); a
    /// component with a plain config, or none, is dense (the CPU-stage → quantize-onto path). Drives
    /// `Pipeline::load_one_quantizable`'s device choice.
    #[test]
    fn component_is_packed_reads_quantization_block() {
        let dir = std::env::temp_dir().join(format!("sc9087_pkg_{}", std::process::id()));
        let pipe = Pipeline::load(Flux2Variant::Dev, Some(Quant::Q4), &dir, &Device::Cpu, None);

        let packed = dir.join("transformer");
        std::fs::create_dir_all(&packed).unwrap();
        std::fs::write(
            packed.join("config.json"),
            r#"{"quantization": {"bits": 4, "group_size": 64}}"#,
        )
        .unwrap();
        assert!(
            pipe.component_is_packed("transformer").unwrap(),
            "a `quantization` block ⇒ packed tier"
        );

        let dense = dir.join("text_encoder");
        std::fs::create_dir_all(&dense).unwrap();
        std::fs::write(dense.join("config.json"), r#"{"hidden_size": 5120}"#).unwrap();
        assert!(
            !pipe.component_is_packed("text_encoder").unwrap(),
            "no `quantization` block ⇒ dense tier"
        );
        // A component with no config.json at all → dense (fixtures still load).
        assert!(!pipe.component_is_packed("vae").unwrap());

        // A config.json that is *present but corrupt* (malformed JSON, e.g. a partial download) must
        // error loudly naming the file — NOT silently fall to the dense path (sc-9426 / F-073 sibling).
        let corrupt = dir.join("vae_bad");
        std::fs::create_dir_all(&corrupt).unwrap();
        std::fs::write(corrupt.join("config.json"), b"{ not json").unwrap();
        let err = pipe
            .component_is_packed("vae_bad")
            .expect_err("corrupt config.json must error, not fall to dense");
        assert!(
            format!("{err}").contains("config.json"),
            "the error should name the offending file, got: {err}"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// The shared quantizable-loader's three device/dtype-selection regimes (the F-024 de-dup home,
    /// sc-9004). `load_one_quantizable` is the single body behind `load_te_and_dit` (and thus behind
    /// txt2img, edit, and control): the same routing decision every entry point makes. Exercised on CPU
    /// with a stub module that records the device its VarBuilder was built on:
    /// - **no quant** → build on the configured device (`self.device`), no staging.
    /// - **dense tier + quant** → stage dense on the CPU, then quantize onto `self.device`.
    /// - **packed tier + quant** → build directly on `self.device` (no dense CPU staging, sc-9087).
    ///
    /// The dtype passed to the builder is always `self.dtype` (f32) regardless of regime — the loaded
    /// weights + dtype/device stay byte-identical per site (the invariant the de-dup must preserve).
    #[test]
    fn load_one_quantizable_selects_device_per_tier() {
        use candle_gen::candle_core::safetensors;
        use std::collections::HashMap;

        /// Records the device + dtype its VarBuilder was constructed on, and whether the post-build
        /// `quantize` hook ran (the CPU-stage → quantize-onto-GPU / packed handoff).
        struct Probe {
            device: Device,
            dtype: DType,
            quantized: std::cell::Cell<bool>,
        }

        let dir = std::env::temp_dir().join(format!("sc9004_loader_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // A one-tensor safetensors shard so `component_vb_on` mmaps successfully for either component.
        let write_shard = |sub: &str, packed: bool| {
            let comp = dir.join(sub);
            std::fs::create_dir_all(&comp).unwrap();
            let mut map = HashMap::new();
            map.insert(
                "w".to_string(),
                Tensor::zeros((2, 2), DType::F32, &Device::Cpu).unwrap(),
            );
            safetensors::save(&map, comp.join("model.safetensors")).unwrap();
            if packed {
                std::fs::write(
                    comp.join("config.json"),
                    r#"{"quantization": {"bits": 4, "group_size": 64}}"#,
                )
                .unwrap();
            }
        };

        // The build closure just records what the loader handed it; the quantize closure records that it
        // ran and on which device (the CPU-stage → quantize-onto-GPU handoff).
        let build = |vb: VarBuilder| -> CResult<Probe> {
            Ok(Probe {
                device: vb.device().clone(),
                dtype: vb.dtype(),
                quantized: std::cell::Cell::new(false),
            })
        };
        let quantize = |m: &mut Probe, _q: Quant, _d: &Device| -> CResult<()> {
            m.quantized.set(true);
            Ok(())
        };

        // no quant → configured device, no staging call.
        write_shard("text_encoder", false);
        let pipe = Pipeline::load(Flux2Variant::Klein9b, None, &dir, &Device::Cpu, None);
        let p = pipe
            .load_one_quantizable("text_encoder", build, quantize)
            .unwrap();
        assert!(matches!(p.device, Device::Cpu));
        assert_eq!(p.dtype, DType::F32);
        assert!(!p.quantized.get(), "no-quant path must not quantize");

        // dense tier + quant → the builder sees the CPU (staging), then quantize runs onto the device.
        let dense = Pipeline::load(Flux2Variant::Dev, Some(Quant::Q4), &dir, &Device::Cpu, None);
        let p = dense
            .load_one_quantizable("text_encoder", build, quantize)
            .unwrap();
        assert!(
            matches!(p.device, Device::Cpu),
            "dense-tier build stages on CPU"
        );
        assert!(
            p.quantized.get(),
            "dense-tier + quant must quantize onto the device"
        );

        // packed tier + quant → the builder sees the configured device directly (no dense staging).
        write_shard("transformer", true);
        let packed = Pipeline::load(Flux2Variant::Dev, Some(Quant::Q4), &dir, &Device::Cpu, None);
        let p = packed
            .load_one_quantizable("transformer", build, quantize)
            .unwrap();
        assert!(
            matches!(p.device, Device::Cpu),
            "packed-tier build lands on the configured device (no CPU staging step)"
        );
        assert!(
            p.quantized.get(),
            "packed-tier still runs the (no-op on projections) quantize to carry dense leaves"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// `load_te_and_dit` is a thin delegation to `load_quantizable` with the default TE+DiT builders —
    /// the single home the three entry points (txt2img/edit/control) now share (F-024, sc-9004). It
    /// surfaces the underlying loader error (here: a snapshot missing the `text_encoder/` component)
    /// unchanged, confirming the delegation is wired without needing real 32B weights.
    #[test]
    fn load_te_and_dit_surfaces_missing_component() {
        let dir = std::env::temp_dir().join(format!("sc9004_missing_{}", std::process::id()));
        // No component dirs written → the shared loader must error on the missing text_encoder/.
        let pipe = Pipeline::load(Flux2Variant::Klein9b, None, &dir, &Device::Cpu, None);
        let err = pipe
            .load_te_and_dit()
            .err()
            .expect("missing components")
            .to_string();
        assert!(
            err.contains("text_encoder"),
            "delegation surfaces the loader's missing-component error, got: {err}"
        );
    }

    /// The in-place ComfyUI DiT entry point (epic 10451 Phase 2e, sc-10680) builds a lazy dev generator
    /// without touching weights: it stamps the dev descriptor + carries the DiT file, and the resident
    /// snapshot dir is the root supplying the TE/VAE/tokenizer. Loading is lazy, so this asserts the
    /// plumbing on CPU with no weights (the render itself is GPU-validated separately).
    #[test]
    fn load_from_comfyui_dit_builds_lazy_dev_generator() {
        let g = load_from_comfyui_dit(
            "/tree/diffusion_models/flux2_dev_fp8mixed.safetensors",
            "/snap/flux2-dev",
            Some(Quant::Q8),
        )
        .expect("comfyui dev generator builds lazily");
        assert_eq!(g.descriptor().id, FLUX2_DEV_ID);
        assert_eq!(g.descriptor().family, "flux2");
        assert_eq!(g.descriptor().backend, "candle");
        assert_eq!(g.descriptor().modality, Modality::Image);
    }

    #[test]
    fn load_rejects_single_file_source() {
        let spec = LoadSpec::new(WeightsSource::File("/tmp/flux2.safetensors".into()));
        let err = load_klein(&spec)
            .err()
            .expect("expected an error")
            .to_string();
        assert!(err.contains("snapshot directory"), "got: {err}");
    }

    /// The sequential-residency offload contract (epic 10765 Phase 1c, sc-10868): `with_offload_policy`
    /// is captured at load (not rejected), and the env override + spec policy select the phased path.
    /// Loading stays lazy, so this asserts the plumbing on CPU without any weights or a GPU: a
    /// `Sequential` spec builds a generator (the `render_sequential` route is selected inside `generate`,
    /// exercised end-to-end by the cuda A/B below), and the default spec stays `Resident`.
    #[test]
    fn offload_policy_is_captured_not_rejected() {
        // Default (no policy set) → Resident: the generator builds, the cached `render` path is default.
        let spec = LoadSpec::new(WeightsSource::Dir("/snap".into()));
        assert_eq!(spec.offload_policy, OffloadPolicy::Resident);
        assert!(load_dev(&spec).is_ok());
        assert!(load_klein(&spec).is_ok());

        // `Sequential` is honored, not rejected — for both variants (dev's Mistral TE is the big win; the
        // klein path is wired identically). The weights are never touched (lazy build).
        let seq = LoadSpec::new(WeightsSource::Dir("/snap".into()))
            .with_offload_policy(OffloadPolicy::Sequential);
        assert_eq!(seq.offload_policy, OffloadPolicy::Sequential);
        assert!(load_dev(&seq).is_ok());
        assert!(load_klein(&seq).is_ok());

        // The env override is read independently of the spec (the GPU A/B harness seam).
        std::env::set_var("CANDLE_GEN_OFFLOAD", "SeQuEnTiAl");
        assert!(super::sequential_offload_enabled());
        std::env::set_var("CANDLE_GEN_OFFLOAD", "resident");
        assert!(!super::sequential_offload_enabled());
        std::env::remove_var("CANDLE_GEN_OFFLOAD");
        assert!(!super::sequential_offload_enabled());
    }

    /// Shared body for the FLUX.2 offload A/B harnesses (epic 10765 Phase 1c, sc-10868 dev / sc-11008
    /// klein). Loads `label`'s snapshot from the `dir_env` env var and runs ONE probed 1024²
    /// generation whose residency mode is chosen by the same two seams `generate` reads —
    /// `CANDLE_GEN_OFFLOAD=sequential` (the env override, sc-10769) or `FLUX2_OFFLOAD_MODE=spec-sequential`
    /// → `LoadSpec::offload_policy` (the worker-facing contract, sc-10821, with `CANDLE_GEN_OFFLOAD`
    /// UNSET) — then prints the device peak VRAM (`SEQ_AB` line) and writes the raw RGB pixels to
    /// `FLUX2_OUT`. Run it TWICE in SEPARATE processes (resident vs sequential) and compare: the pixel
    /// files must be byte-identical (parity) and the sequential peak materially lower (the dense text
    /// encoder dropped before the DiT loads). Two processes are REQUIRED — candle's cudarc caching
    /// allocator never returns pages to the driver, so a second in-process run reuses the first run's
    /// pool and reads the same peak. `honor_quant` reads `FLUX2_QUANT` (q4/q8) for the dev 32B (which
    /// fits only quantized); klein has no candle quant path, so it always loads dense. Needs a real-file
    /// (hardlink-staged, not raw-HF-symlink) snapshot in `dir_env` and a CUDA device.
    #[cfg(feature = "cuda")]
    fn run_probed_offload_ab(
        label: &str,
        dir_env: &str,
        load: fn(&LoadSpec) -> gen_core::Result<Box<dyn Generator>>,
        honor_quant: bool,
        steps: u32,
    ) {
        let dir = std::env::var(dir_env).unwrap_or_else(|_| {
            panic!("set {dir_env} to a real-file (hardlink-staged) {label} snapshot")
        });
        let out = std::env::var("FLUX2_OUT").expect("set FLUX2_OUT to the pixel-dump path");
        let mut spec = LoadSpec::new(WeightsSource::Dir(dir.into()));
        // The 32B dev fits only quantized; honor FLUX2_QUANT (q4/q8), else load dense (fixture-only).
        // klein has no candle quant path (only bf16 dense is candle-loadable) → honor_quant is false.
        if honor_quant {
            spec = match std::env::var("FLUX2_QUANT")
                .unwrap_or_default()
                .to_lowercase()
                .as_str()
            {
                "q4" => spec.with_quant(Quant::Q4),
                "q8" => spec.with_quant(Quant::Q8),
                _ => spec,
            };
        }
        let spec_mode = std::env::var("FLUX2_OFFLOAD_MODE").unwrap_or_default();
        if spec_mode == "spec-sequential" {
            spec = spec.with_offload_policy(OffloadPolicy::Sequential);
        }
        let req = GenerationRequest {
            prompt: "a rusty robot holding a lit candle, studio lighting".into(),
            width: 1024,
            height: 1024,
            steps: Some(steps),
            seed: Some(42),
            count: 1,
            ..Default::default()
        };
        let sampler = candle_gen::testkit::PeakSampler::start(0);
        let g = load(&spec).unwrap_or_else(|e| panic!("load {label}: {e}"));
        let output = g.generate(&req, &mut |_| {}).expect("generate");
        let peak_mib = sampler.stop();
        let img = match output {
            GenerationOutput::Images(mut v) => v.remove(0),
            other => panic!("expected images, got {other:?}"),
        };
        std::fs::write(&out, &img.pixels).expect("write pixels");
        let env_mode = std::env::var("CANDLE_GEN_OFFLOAD").unwrap_or_default();
        let mode = if spec_mode == "spec-sequential" {
            "spec-sequential"
        } else if env_mode.eq_ignore_ascii_case("sequential") {
            "env-sequential"
        } else {
            "resident"
        };
        eprintln!(
            "SEQ_AB model={label} mode={mode} peak_mib={peak_mib} bytes={} {}x{} out={out}",
            img.pixels.len(),
            img.width,
            img.height
        );
    }

    /// Sequential-residency GPU validation (epic 10765 Phase 1c, sc-10868) for FLUX.2-**dev** (Mistral
    /// TE, guidance-distilled 32B). See [`run_probed_offload_ab`] for the A/B protocol. Ignored by
    /// default; needs a hardlink-staged FLUX.2-dev snapshot in `FLUX2_DEV_DIR`, a `FLUX2_QUANT` of
    /// `q4`/`q8` (the 32B needs it — omit only for a dense fixture), and a CUDA device.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore]
    fn flux2_dev_probed_generate_for_offload_ab() {
        run_probed_offload_ab("flux2_dev", "FLUX2_DEV_DIR", load_dev, true, 8);
    }

    /// klein sibling of [`flux2_dev_probed_generate_for_offload_ab`] (epic 10765 Phase 1c follow-up,
    /// sc-11008). Same A/B protocol, but loads FLUX.2-**klein**-9B (Qwen3 TE + 9B DiT) DENSE bf16 from
    /// `FLUX2_KLEIN_DIR` at 4 steps CFG-free. klein has NO candle quant path — the `SceneWorks/
    /// flux2-klein-9b-mlx` q4/q8 turnkey tiers are MLX-format, not candle-loadable — so only the bf16
    /// tier is candle-measurable here (`honor_quant = false`). Directly captures the klein resident-vs-
    /// sequential peak that sc-10920 could only arch-scale, validating that dropping the DENSE ~16 GB
    /// Qwen3 TE before the 9B DiT loads is the sequential floor. Ignored by default; needs a hardlink-
    /// staged klein bf16 diffusers snapshot (`text_encoder/` = Qwen3, `transformer/`, `vae/`,
    /// `tokenizer/`) in `FLUX2_KLEIN_DIR` and a CUDA device.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore]
    fn flux2_klein_probed_generate_for_offload_ab() {
        run_probed_offload_ab("flux2_klein_9b", "FLUX2_KLEIN_DIR", load_klein, false, 4);
    }
}
