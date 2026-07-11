//! The Wan2.2 **A14B dual-expert MoE** video providers (sc-5174) — the candle (Windows/CUDA) siblings
//! of `mlx-gen-wan`'s `wan2_2_t2v_14b` / `wan2_2_i2v_14b`. Both register as `backend = "candle"`,
//! [`Modality::Video`].
//!
//! Wan2.2's "MoE" is **two complete `WanTransformer3DModel` checkpoints**, not token routing: a
//! **high-noise** expert (`transformer/`) and a **low-noise** expert (`transformer_2/`). A single
//! flow-match scheduler drives the denoise; each step picks the high expert while the integer timestep
//! is `≥ boundary·1000` (T2V `0.875`, I2V `0.900`) and the low expert below it, switching the
//! transformer, its (per-expert) text context, and its guidance scale together (T2V 3.0/4.0, I2V
//! 3.5/3.5). The experts share the dimension-parametric [`WanTransformer`] (loaded with
//! [`TransformerConfig::t2v_14b`]/[`i2v_14b`](TransformerConfig::i2v_14b)) and the [`crate::vae16`] z16
//! VAE — *not* the 5B's z48 VAE (the 14B emits 16-channel latents).
//!
//! **T2V** (`wan2_2_t2v_14b`): pure text→video. **I2V** (`wan2_2_i2v_14b`): channel-concat conditioning
//! — the reference image's first-frame z16 latent + a temporal mask form a 20-channel `y` appended to
//! the 16-channel noise latent (in_dim 36) every forward (the image enters via the channels, not noise).
//!
//! **Dtypes:** UMT5 + VAE run **f32**; the experts run **bf16** (norms/modulation upcast to f32),
//! mirroring the 5B. The VAE decode **streams one latent frame at a time** (sc-5176) to bound the
//! decode-stage peak — the heavier-than-5B fix the story (sc-5174) requires.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::{safetensors as cst, DType, Device, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::{
    self, AdapterSpec, Capabilities, Conditioning, ConditioningKind, GenerationOutput,
    GenerationRequest, Generator, Image, LoadSpec, Modality, ModelDescriptor, MoeExpert, Progress,
    Quant, WeightsSource,
};
use candle_gen::{CandleError, Result as CResult};

use crate::config::{
    TextEncoderConfig, TransformerConfig, Vae16Config, DEFAULT_FPS_14B, DEFAULT_FRAMES_14B,
    DEFAULT_STEPS_14B, I2V_14B_BOUNDARY, I2V_14B_FLOW_SHIFT, I2V_14B_GUIDANCE_HIGH,
    I2V_14B_GUIDANCE_LOW, MAX_AREA_14B, MODEL_ID_I2V_14B, MODEL_ID_T2V_14B, NEGATIVE_FALLBACK,
    NUM_TRAIN_TIMESTEPS, SIZE_MULTIPLE_14B, T2V_14B_BOUNDARY, T2V_14B_FLOW_SHIFT,
    T2V_14B_GUIDANCE_HIGH, T2V_14B_GUIDANCE_LOW, VAE16_STRIDE_SPATIAL, VAE16_STRIDE_TEMPORAL,
};
use crate::pipeline::{cfg, create_noise, frames_to_images};
use crate::rope::WanRope;
use crate::scheduler::{FlowScheduler, Sampler};
use crate::text_encoder::Umt5Encoder;
use crate::transformer::WanTransformer;
use crate::vae16::WanVae16;

/// The experts run bf16 (the diffusers fp32 weights load as bf16, the 5B regime); UMT5 + VAE run f32.
const DIT_DTYPE: DType = DType::BF16;
const ENC_DTYPE: DType = DType::F32;
const VAE_DTYPE: DType = DType::F32;
const Z_DIM: usize = 16;

/// Which A14B model this generator serves — selects in_dim (16 vs 36), the MoE knobs, and whether the
/// VAE carries an encoder (I2V conditioning).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Variant {
    T2v,
    I2v,
}

impl Variant {
    fn id(self) -> &'static str {
        match self {
            Variant::T2v => MODEL_ID_T2V_14B,
            Variant::I2v => MODEL_ID_I2V_14B,
        }
    }

    fn dit_cfg(self) -> TransformerConfig {
        match self {
            Variant::T2v => TransformerConfig::t2v_14b(),
            Variant::I2v => TransformerConfig::i2v_14b(),
        }
    }

    /// `(boundary, default flow-shift, guidance_low, guidance_high)`.
    fn moe_knobs(self) -> (f64, f64, f32, f32) {
        match self {
            Variant::T2v => (
                T2V_14B_BOUNDARY,
                T2V_14B_FLOW_SHIFT,
                T2V_14B_GUIDANCE_LOW,
                T2V_14B_GUIDANCE_HIGH,
            ),
            Variant::I2v => (
                I2V_14B_BOUNDARY,
                I2V_14B_FLOW_SHIFT,
                I2V_14B_GUIDANCE_LOW,
                I2V_14B_GUIDANCE_HIGH,
            ),
        }
    }
}

/// True when classifier-free guidance is actually active: the negative/uncond branch only changes the
/// output at `guidance > 1.0`. At `guidance <= 1.0` the CFG combine `neg + g·(pos − neg)` reduces to
/// `pos` (exactly `pos` at 1.0), so the negative UMT5 encode + per-expert projection + per-step forward
/// are pure waste and are skipped (sc-8993). Kept as one predicate so the encode-time gate and the
/// per-step gate can never diverge.
fn cfg_active(guidance: f64) -> bool {
    guidance > 1.0
}

#[derive(Clone)]
struct Components {
    te: Arc<Umt5Encoder>,
    /// `transformer/` — the **high-noise** expert (timestep ≥ boundary).
    high: Arc<WanTransformer>,
    /// `transformer_2/` — the **low-noise** expert (timestep < boundary).
    low: Arc<WanTransformer>,
    vae: Arc<WanVae16>,
    /// UMT5 tokenizer, loaded+parsed **once** at component load and reused across encodes (sc-8991 /
    /// F-011) instead of re-parsing `tokenizer.json` per prompt/branch.
    tok: Arc<candle_gen::gen_core::tokenizer::TextTokenizer>,
}

struct Pipeline {
    te_cfg: TextEncoderConfig,
    dit_cfg: TransformerConfig,
    vae_cfg: Vae16Config,
    variant: Variant,
    root: PathBuf,
    device: Device,
    /// Trained LoRA/LoKr adapters to merge into the experts at load (sc-5167). Each is routed to the
    /// high and/or low expert by its [`AdapterSpec::moe_expert`].
    adapters: Vec<AdapterSpec>,
    /// In-place ComfyUI expert files (epic 10451 Phase 2c, sc-10671). When set, the two experts are
    /// built from these files (scaled-fp8 dequant + key remap, see [`crate::comfyui`]) instead of the
    /// snapshot's `transformer/` + `transformer_2/`. The UMT5 TE + VAE are read in place too when the
    /// spec carries their files (sc-10909), else from `root`; the tiny tokenizer always comes from
    /// `root`. `None` on the registry path.
    comfyui: Option<std::sync::Arc<crate::comfyui::ComfyuiExperts>>,
}

impl Pipeline {
    fn load(root: &Path, device: &Device, variant: Variant, adapters: Vec<AdapterSpec>) -> Self {
        Self {
            te_cfg: TextEncoderConfig::umt5_xxl(),
            dit_cfg: variant.dit_cfg(),
            vae_cfg: Vae16Config::wan21(),
            variant,
            root: root.to_path_buf(),
            device: device.clone(),
            adapters,
            comfyui: None,
        }
    }

    /// Same as [`load`](Self::load) but with the two experts sourced from an in-place ComfyUI install
    /// (sc-10671). `root` is the resident Wan snapshot tier supplying the UMT5 TE / VAE / tokenizer;
    /// `comfyui` names the user's two ComfyUI expert files, read in place.
    fn load_comfyui(
        root: &Path,
        device: &Device,
        variant: Variant,
        comfyui: std::sync::Arc<crate::comfyui::ComfyuiExperts>,
    ) -> Self {
        Self {
            te_cfg: TextEncoderConfig::umt5_xxl(),
            dit_cfg: variant.dit_cfg(),
            vae_cfg: Vae16Config::wan21(),
            variant,
            root: root.to_path_buf(),
            device: device.clone(),
            adapters: Vec::new(),
            comfyui: Some(comfyui),
        }
    }

    /// Build one expert from an in-place ComfyUI file (sc-10671): load its native tensor map on CPU,
    /// remap keys + dequant the scaled-fp8 weights ([`crate::comfyui`]) into the diffusers schema, then
    /// build via `VarBuilder::from_tensors` (the in-memory path the packed/adapter loads already use).
    fn build_expert_comfyui(&self, file: &Path) -> CResult<WanTransformer> {
        let map = cst::load(file, &Device::Cpu)?;
        let map = crate::comfyui::remap_and_dequant_comfyui_expert(map, DIT_DTYPE)?;
        let vb = VarBuilder::from_tensors(map, DIT_DTYPE, &self.device);
        Ok(WanTransformer::new(&self.dit_cfg, vb)?)
    }

    /// Build the UMT5 text encoder from an in-place ComfyUI file (sc-10909): load its native tensor map
    /// on CPU, dequant the companion scaled-fp8 weights ([`crate::comfyui::dequant_comfyui_umt5`], no
    /// key remap — the ComfyUI file already carries the HF `UMT5EncoderModel` keys), then build via
    /// `VarBuilder::from_tensors` at [`ENC_DTYPE`] (f32, matching the snapshot TE).
    fn build_te_comfyui(&self, file: &Path) -> CResult<Umt5Encoder> {
        let map = cst::load(file, &Device::Cpu)?;
        let map = crate::comfyui::dequant_comfyui_umt5(map, ENC_DTYPE)?;
        let vb = VarBuilder::from_tensors(map, ENC_DTYPE, &self.device);
        Ok(Umt5Encoder::new(&self.te_cfg, vb)?)
    }

    /// Build the z16 VAE from an in-place ComfyUI file (sc-10909): load its native tensor map on CPU,
    /// remap the native WAN-VAE keys to the diffusers schema
    /// ([`crate::comfyui::remap_vae_wan_to_diffusers`], values pass through as bf16), then build via
    /// `VarBuilder::from_tensors` at [`VAE_DTYPE`] (f32 upcast on `get`, byte-matching the snapshot VAE
    /// mmap). I2V builds the encoder too (the conditioning image's first-frame latent).
    fn build_vae_comfyui(&self, file: &Path) -> CResult<WanVae16> {
        let map = cst::load(file, &Device::Cpu)?;
        let map = crate::comfyui::remap_vae_wan_to_diffusers(map)?;
        let vb = VarBuilder::from_tensors(map, VAE_DTYPE, &self.device);
        match self.variant {
            Variant::I2v => Ok(WanVae16::new_with_encoder(&self.vae_cfg, vb)?),
            Variant::T2v => Ok(WanVae16::new(&self.vae_cfg, vb)?),
        }
    }

    fn component_vb(&self, sub: &str, dtype: DType) -> CResult<VarBuilder<'static>> {
        // Shared Wan component loader (sc-9000 / F-020); the crafted snapshot description (which names
        // the expected A14B variant) stays local.
        let desc = match self.variant {
            Variant::T2v => "Wan2.2-T2V-A14B diffusers",
            Variant::I2v => "Wan2.2-I2V-A14B diffusers",
        };
        crate::text_encode::component_vb(&self.root, sub, dtype, &self.device, "wan-14b", desc)
    }

    /// Build one expert from its `sub` dir, applying any adapter whose [`AdapterSpec::moe_expert`]
    /// targets it (`Some(expert)` or `None` = shared). The adapter application differs by tier:
    ///
    /// - **Dense** tier: the delta is FOLDED into the dense weights before build
    ///   ([`crate::adapters::merge_adapters`], f32 math, `VarBuilder::from_tensors`) — the
    ///   merge-not-residual pattern the SDXL/Z-Image ports established, byte-identical to before.
    ///   `merge_adapters` hard-errors on its own zero-match (the report is otherwise discarded, F-051).
    /// - **Packed** q4/q8 tier (sc-10095): a packed tier has **no dense `W`** to fold into, so the
    ///   adapters attach as forward-time **additive** residuals on the packed `QLinear`
    ///   ([`crate::adapters::install_additive`], sc-10094) — the base weight stays q4/q8. LoKr/LoHa on a
    ///   packed tier is rejected there (deferred to sc-10050/10051).
    ///
    /// Returns the expert plus `Some(applied)` on the packed path (the count of attached residuals, for
    /// the caller's cross-expert zero-match guard) or `None` on the dense/no-adapter paths (which
    /// self-guard). With no adapter for this expert the fast mmap build is used unchanged.
    fn build_expert(
        &self,
        sub: &str,
        expert: MoeExpert,
    ) -> CResult<(WanTransformer, Option<usize>)> {
        let specs: Vec<AdapterSpec> = self
            .adapters
            .iter()
            .filter(|s| s.moe_expert.is_none_or(|e| e == expert))
            .cloned()
            .collect();
        let vb = self.component_vb(sub, DIT_DTYPE)?;
        // Packed-tier marker: the sc-10025 seam packs every DiT Linear (incl. `proj_out`), so a
        // `proj_out.scales` sibling is present iff this is a pre-quantized q4/q8 tier.
        let packed = vb.contains_tensor("proj_out.scales");
        if packed {
            let mut dit = WanTransformer::new(&self.dit_cfg, vb)?;
            if specs.is_empty() {
                return Ok((dit, Some(0)));
            }
            // Additive install on the packed base — no dense weight materialized (sc-10094/10095).
            let report = crate::adapters::install_additive(&mut dit, &specs, expert)?;
            return Ok((dit, Some(report.applied)));
        }
        if specs.is_empty() {
            return Ok((WanTransformer::new(&self.dit_cfg, vb)?, None));
        }
        // Dense tier + adapters: fold the delta into the dense weights before building (the legacy
        // merge-not-residual fast path). `merge_adapters` hard-errors on its own zero-match.
        drop(vb);
        let mut map = self.load_component_map(sub)?;
        crate::adapters::merge_adapters(&mut map, &specs)?;
        let vb = VarBuilder::from_tensors(map, DIT_DTYPE, &self.device);
        Ok((WanTransformer::new(&self.dit_cfg, vb)?, None))
    }

    /// Load every `.safetensors` in the component subdir `sub` into one CPU tensor map (native dtype) —
    /// the merge-ready form the adapter fold needs (vs the mmap `component_vb` fast path).
    fn load_component_map(&self, sub: &str) -> CResult<HashMap<String, Tensor>> {
        let dir = self.root.join(sub);
        // Shared sorted-`.safetensors` resolver (sc-8999 / F-019); this path then loads the shards
        // into a CPU map for adapter merging (not the mmap fast path), so it keeps its own loop.
        let files = candle_gen::sorted_safetensors(&dir, "wan-14b")?;
        let mut map: HashMap<String, Tensor> = HashMap::new();
        for f in &files {
            map.extend(cst::load(f, &Device::Cpu)?);
        }
        Ok(map)
    }

    fn load_components(&self) -> CResult<Components> {
        // In-place ComfyUI mode (sc-10909): read the UMT5 TE from the tree when the caller passed it
        // (scaled-fp8 dequant), else the snapshot `text_encoder/`.
        let te = match self.comfyui.as_ref().and_then(|c| c.te_file.as_deref()) {
            Some(te_file) => self.build_te_comfyui(te_file)?,
            None => Umt5Encoder::new(&self.te_cfg, self.component_vb("text_encoder", ENC_DTYPE)?)?,
        };
        // transformer/ = high-noise expert, transformer_2/ = low-noise expert (diffusers WanPipeline).
        // In-place ComfyUI mode (sc-10671): build both experts from the user's files (remap + dequant);
        // no adapters (the ComfyUI base lane does not fold LoRAs), so the zero-match guard below is inert.
        let (high, high_applied, low, low_applied) = match &self.comfyui {
            Some(experts) => (
                self.build_expert_comfyui(&experts.high_file)?,
                None,
                self.build_expert_comfyui(&experts.low_file)?,
                None,
            ),
            None => {
                let (high, high_applied) = self.build_expert("transformer", MoeExpert::High)?;
                let (low, low_applied) = self.build_expert("transformer_2", MoeExpert::Low)?;
                (high, high_applied, low, low_applied)
            }
        };
        // Packed-tier zero-match guard (sc-10095): on the additive path a non-empty adapter set that
        // attached NO residual across either expert is a format/prefix misconfiguration (the dense fold
        // path self-guards inside `merge_adapters`, so it returns `None` here and is exempt). Both experts
        // share a tier, so this is `Some` iff packed.
        if !self.adapters.is_empty() {
            if let (Some(h), Some(l)) = (high_applied, low_applied) {
                if h + l == 0 {
                    return Err(CandleError::Msg(format!(
                        "{}: {} LoRA adapter file(s) matched no projection on either packed expert — \
                         check the key format (expected PEFT `<path>.lora_A/B.weight` or kohya \
                         `lora_unet_<flat>` targeting the DiT attention/FFN Linears)",
                        self.variant.id(),
                        self.adapters.len()
                    )));
                }
            }
        }
        // In-place ComfyUI mode (sc-10909): read the Wan VAE from the tree when the caller passed it
        // (native→diffusers key remap), else the snapshot `vae/`. I2V needs the VAE encoder either way.
        let vae = match self.comfyui.as_ref().and_then(|c| c.vae_file.as_deref()) {
            Some(vae_file) => self.build_vae_comfyui(vae_file)?,
            None => {
                let vae_vb = self.component_vb("vae", VAE_DTYPE)?;
                match self.variant {
                    // I2V needs the VAE encoder (the conditioning image's first-frame latent).
                    Variant::I2v => WanVae16::new_with_encoder(&self.vae_cfg, vae_vb)?,
                    Variant::T2v => WanVae16::new(&self.vae_cfg, vae_vb)?,
                }
            }
        };
        let tok = crate::text_encode::build_umt5_tokenizer(&self.root, &self.te_cfg, "wan-14b")?;
        Ok(Components {
            te: Arc::new(te),
            high: Arc::new(high),
            low: Arc::new(low),
            vae: Arc::new(vae),
            tok: Arc::new(tok),
        })
    }

    /// Tokenize + UMT5-encode `prompt` → `[1, 512, 4096]` (f32), zero-padded to `max_length` (the DiT
    /// cross-attends over the 512-padded context — the same rule as the 5B, sc-3697). Shared Wan
    /// text-encode routine (sc-9000 / F-020).
    fn encode(&self, comps: &Components, prompt: &str) -> CResult<Tensor> {
        crate::text_encode::umt5_encode_padded(
            &comps.tok,
            &self.te_cfg,
            &comps.te,
            prompt,
            &self.device,
            ENC_DTYPE,
            "wan-14b",
        )
    }

    /// Build the I2V channel-concat conditioning `y` `[1, 20, t_lat, h_lat, w_lat]` =
    /// `[mask(4), z_video(16)]`: a conditioning video (frame 0 = the preprocessed image, the rest zero)
    /// is z16-VAE-encoded, and a temporal mask (1.0 at latent frame 0, else 0.0) is prepended. Mirrors
    /// `generate_wan.py`'s `is_i2v_channel_concat` setup. Constant across denoise steps + both experts.
    fn build_i2v_y(
        &self,
        vae: &WanVae16,
        image: &Image,
        frames: u32,
        width: u32,
        height: u32,
    ) -> CResult<Tensor> {
        // Conditioning video [1, 3, F, H, W]: frame 0 = image (in [-1,1]), rest zeros.
        let first = preprocess_i2v_image(image, width, height, &self.device)?; // [1,3,1,H,W]
        let video = if frames > 1 {
            let rest = Tensor::zeros(
                (1, 3, frames as usize - 1, height as usize, width as usize),
                DType::F32,
                &self.device,
            )?;
            Tensor::cat(&[&first, &rest], 2)?
        } else {
            first
        };
        let z_video = vae.encode(&video)?; // [1, 16, t_lat, h_lat, w_lat]

        // Mask dims follow the encoder's actual output, so they always match `z_video`.
        let (_, _, t_lat, h_lat, w_lat) = z_video.dims5()?;
        // 4-channel temporal mask: 1.0 at latent frame 0 (all channels/spatial), 0.0 elsewhere.
        let plane = h_lat * w_lat;
        let mut mask = vec![0f32; 4 * t_lat * plane];
        for c in 0..4 {
            let base = c * t_lat * plane; // temporal index 0 of channel c
            for v in mask.iter_mut().skip(base).take(plane) {
                *v = 1.0;
            }
        }
        let mask = Tensor::from_vec(mask, (1, 4, t_lat, h_lat, w_lat), &self.device)?;
        Ok(Tensor::cat(&[&mask, &z_video], 1)?) // [1, 20, t_lat, h_lat, w_lat]
    }

    fn render(
        &self,
        req: &GenerationRequest,
        comps: &Components,
        on_progress: &mut dyn FnMut(Progress),
    ) -> CResult<(Vec<Image>, u32)> {
        let steps = req
            .steps
            .map(|s| s as usize)
            .unwrap_or(DEFAULT_STEPS_14B as usize);
        let frames = req.frames.unwrap_or(DEFAULT_FRAMES_14B);
        let fps = req.fps.unwrap_or(DEFAULT_FPS_14B);
        let seed = req.seed.unwrap_or_else(gen_core::default_seed);
        let sampler = Sampler::parse(req.sampler.as_deref());
        let (boundary, default_shift, gl, gh) = self.variant.moe_knobs();
        let shift = req
            .scheduler_shift
            .map(|s| s as f64)
            .unwrap_or(default_shift);
        // A scalar request guidance overrides both experts; else the per-expert (low, high) defaults.
        let (g_low, g_high) = match req.guidance {
            Some(g) => (g as f64, g as f64),
            None => (gl as f64, gh as f64),
        };

        // Text encode (pos always) once; project to each expert's context (per-expert text_embedder).
        // The negative branch is only used at guidance > 1.0, and the two experts can have distinct
        // guidance — so UMT5-encode + project the negative for an expert only when its own guidance
        // enables CFG. At guidance <= 1.0 the denoise loop never touches `*_neg`, so the 24-layer UMT5
        // forward over the negative and its projection are pure waste (sc-8993).
        let pos = self.encode(comps, &req.prompt)?;
        let high_pos = comps.high.embed_text(&pos)?;
        let low_pos = comps.low.embed_text(&pos)?;
        // Shared UMT5 negative encode, computed once if either expert has CFG active.
        let neg = if cfg_active(g_high) || cfg_active(g_low) {
            let neg_prompt = req.negative_prompt.as_deref().unwrap_or(NEGATIVE_FALLBACK);
            Some(self.encode(comps, neg_prompt)?)
        } else {
            None
        };
        let high_neg = match &neg {
            Some(neg) if cfg_active(g_high) => Some(comps.high.embed_text(neg)?),
            _ => None,
        };
        let low_neg = match &neg {
            Some(neg) if cfg_active(g_low) => Some(comps.low.embed_text(neg)?),
            _ => None,
        };

        // Latent geometry (z16 strides) + RoPE for the shared token grid.
        let t_lat = ((frames - 1) / VAE16_STRIDE_TEMPORAL + 1) as usize;
        let h_lat = (req.height / VAE16_STRIDE_SPATIAL) as usize;
        let w_lat = (req.width / VAE16_STRIDE_SPATIAL) as usize;
        let (pt, ph, pw) = self.dit_cfg.patch;
        let (ppf, pph, ppw) = (t_lat / pt, h_lat / ph, w_lat / pw);
        let (cos, sin) = WanRope::new(&self.dit_cfg).cos_sin(ppf, pph, ppw, &self.device)?;

        // I2V: build the constant channel-concat conditioning `y` (needs the VAE encoder).
        let y = match self.variant {
            Variant::I2v => {
                let image = i2v_reference(req).ok_or_else(|| {
                    CandleError::Msg(format!(
                        "{}: image-to-video requires a Reference conditioning image",
                        self.variant.id()
                    ))
                })?;
                Some(self.build_i2v_y(&comps.vae, image, frames, req.width, req.height)?)
            }
            Variant::T2v => None,
        };

        let mut latents = create_noise(seed, Z_DIM, t_lat, h_lat, w_lat, &self.device)?;
        let mut sched = FlowScheduler::new(sampler, steps, shift);
        let boundary_ts = boundary * NUM_TRAIN_TIMESTEPS as f64;
        let total = steps as u32;

        for i in 0..steps {
            if req.cancel.is_cancelled() {
                return Err(CandleError::Canceled);
            }
            let t = sched.timestep(i);
            // MoE: high-noise expert at/above the boundary timestep, low-noise below — switching the
            // transformer, its context, and its guidance together.
            let (expert, ctx_pos, ctx_neg, guidance) = if t >= boundary_ts {
                (&comps.high, &high_pos, high_neg.as_ref(), g_high)
            } else {
                (&comps.low, &low_pos, low_neg.as_ref(), g_low)
            };
            // I2V: concat the conditioning `y` onto the noise latent (→ in_dim 36) before the forward.
            let x = match &y {
                Some(y) => Tensor::cat(&[&latents, y], 1)?,
                None => latents.clone(),
            };
            let v_pos = expert.forward(&x, ctx_pos, t, &cos, &sin)?;
            // Negative branch (and CFG combine) only when this expert's guidance enables it; `ctx_neg`
            // is `Some` iff that guidance > 1.0 (sc-8993).
            let v = match ctx_neg {
                Some(ctx_neg) if cfg_active(guidance) => {
                    let v_neg = expert.forward(&x, ctx_neg, t, &cos, &sin)?;
                    cfg(&v_pos, &v_neg, guidance)?
                }
                _ => v_pos,
            };
            latents = sched.step(&v, &latents)?; // 16-channel latent (out_dim 16)
            on_progress(Progress::Step {
                current: i as u32 + 1,
                total,
            });
        }

        on_progress(Progress::Decoding);
        let decoded = comps.vae.decode(&latents)?;
        let images = frames_to_images(&decoded)?;
        Ok((images, fps))
    }
}

/// The single conditioning reference image for I2V (the first video frame), if present.
fn i2v_reference(req: &GenerationRequest) -> Option<&Image> {
    req.conditioning.iter().find_map(|c| match c {
        Conditioning::Reference { image, .. } => Some(image),
        _ => None,
    })
}

/// Preprocess an I2V conditioning [`Image`] to `[1, 3, 1, height, width]` f32 in `[-1, 1]`: a cover-fit
/// resize (`scale = max(W/iw, H/ih)`) + center-crop to the target, then `px/255·2 − 1`. Uses **bilinear**
/// resampling (the reference's PIL-exact LANCZOS, for bit-exact MLX parity, is a follow-up — sc-5174).
pub fn preprocess_i2v_image(
    image: &Image,
    width: u32,
    height: u32,
    device: &Device,
) -> CResult<Tensor> {
    let (iw, ih) = (image.width as usize, image.height as usize);
    let (tw, th) = (width as usize, height as usize);
    if image.pixels.len() != iw * ih * 3 {
        return Err(CandleError::Msg(format!(
            "wan-14b i2v image buffer {} != {iw}x{ih}x3",
            image.pixels.len()
        )));
    }
    // Cover-fit scale + integer resize dims (≥ target so the center-crop is fully covered).
    let scale = (tw as f64 / iw as f64).max(th as f64 / ih as f64);
    let nw = ((iw as f64 * scale).round() as usize).max(tw);
    let nh = ((ih as f64 * scale).round() as usize).max(th);
    let resized = bilinear_rgb(&image.pixels, iw, ih, nw, nh);
    // Center-crop to (th, tw), normalize → CHW [-1,1].
    let (x1, y1) = ((nw - tw) / 2, (nh - th) / 2);
    let plane = th * tw;
    let mut chw = vec![0f32; 3 * plane];
    for yy in 0..th {
        for xx in 0..tw {
            let src = ((y1 + yy) * nw + (x1 + xx)) * 3;
            for c in 0..3 {
                chw[c * plane + yy * tw + xx] = 2.0 * (resized[src + c] / 255.0) - 1.0;
            }
        }
    }
    Ok(Tensor::from_vec(chw, (1, 3, 1, th, tw), device)?)
}

/// Bilinear resize of an `iw×ih` RGB8 (HWC) buffer to `nw×nh`, returning HWC f32 pixel values in
/// `[0, 255]` (not normalized).
fn bilinear_rgb(px: &[u8], iw: usize, ih: usize, nw: usize, nh: usize) -> Vec<f32> {
    let mut out = vec![0f32; nw * nh * 3];
    let sx = iw as f64 / nw as f64;
    let sy = ih as f64 / nh as f64;
    for oy in 0..nh {
        // Pixel-center mapping (align_corners=False), clamped to the source extent.
        let fy = ((oy as f64 + 0.5) * sy - 0.5).clamp(0.0, (ih - 1) as f64);
        let y0 = fy.floor() as usize;
        let y1 = (y0 + 1).min(ih - 1);
        let wy = fy - y0 as f64;
        for ox in 0..nw {
            let fx = ((ox as f64 + 0.5) * sx - 0.5).clamp(0.0, (iw - 1) as f64);
            let x0 = fx.floor() as usize;
            let x1 = (x0 + 1).min(iw - 1);
            let wx = fx - x0 as f64;
            for c in 0..3 {
                let p = |y: usize, x: usize| px[(y * iw + x) * 3 + c] as f64;
                let top = p(y0, x0) * (1.0 - wx) + p(y0, x1) * wx;
                let bot = p(y1, x0) * (1.0 - wx) + p(y1, x1) * wx;
                out[(oy * nw + ox) * 3 + c] = (top * (1.0 - wy) + bot * wy) as f32;
            }
        }
    }
    out
}

/// A loaded Wan2.2 A14B generator (T2V or I2V). Heavy components (UMT5, the two 14B experts, the z16
/// VAE) are loaded lazily on the first `generate` and cached.
pub struct Wan14bGenerator {
    descriptor: ModelDescriptor,
    variant: Variant,
    root: PathBuf,
    device: Device,
    adapters: Vec<AdapterSpec>,
    /// In-place ComfyUI experts (epic 10451 Phase 2c, sc-10671), set only by
    /// [`load_from_comfyui_experts`]. When present, the lazy component build sources both experts from
    /// these files, the UMT5 TE + VAE in place when their files are set (sc-10909) else from
    /// [`Self::root`], and the tiny tokenizer always from [`Self::root`]; `None` on the registry path.
    comfyui: Option<std::sync::Arc<crate::comfyui::ComfyuiExperts>>,
    components: Mutex<Option<Components>>,
}

impl Wan14bGenerator {
    fn components(&self, pipe: &Pipeline) -> gen_core::Result<Components> {
        // `cached` recovers a poisoned lock (sc-9015) internally; `?` bridges the candle-side
        // `load_components` error into `gen_core::Error`.
        Ok(candle_gen::cached(&self.components, || {
            pipe.load_components()
        })?)
    }
}

impl Generator for Wan14bGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        let id = self.variant.id();
        self.descriptor.capabilities.validate_request(id, req)?;
        if req.prompt.is_empty() {
            return Err(gen_core::Error::Msg(format!(
                "{id}: prompt must not be empty"
            )));
        }
        if req.steps == Some(0) {
            return Err(gen_core::Error::Msg(format!("{id}: steps must be >= 1")));
        }
        if !req.width.is_multiple_of(SIZE_MULTIPLE_14B)
            || !req.height.is_multiple_of(SIZE_MULTIPLE_14B)
        {
            return Err(gen_core::Error::Msg(format!(
                "{id}: width/height must be multiples of {SIZE_MULTIPLE_14B} (got {}x{})",
                req.width, req.height
            )));
        }
        // The A14B MoE keeps two resident 14B experts; an over-area request is a far-over-envelope run
        // that fails opaquely (OOM). Reject past the documented cap with an actionable message (sc-9028).
        let area = req.width as usize * req.height as usize;
        if area > MAX_AREA_14B {
            return Err(gen_core::Error::Msg(format!(
                "{id}: width×height ({}×{} = {area} px) exceeds the max area {MAX_AREA_14B} px \
                 (704×1280); reduce the resolution",
                req.width, req.height
            )));
        }
        if let Some(f) = req.frames {
            if f == 0 || f % 4 != 1 {
                return Err(gen_core::Error::Msg(format!(
                    "{id}: frames must satisfy frames % 4 == 1 (got {f})"
                )));
            }
        }
        if self.variant == Variant::I2v && i2v_reference(req).is_none() {
            return Err(gen_core::Error::Msg(format!(
                "{id}: image-to-video requires a Reference conditioning image"
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
        let pipe = match &self.comfyui {
            Some(experts) => {
                Pipeline::load_comfyui(&self.root, &self.device, self.variant, experts.clone())
            }
            None => Pipeline::load(
                &self.root,
                &self.device,
                self.variant,
                self.adapters.clone(),
            ),
        };
        let components = self.components(&pipe)?;
        let (frames, fps) = pipe.render(req, &components, on_progress)?;
        Ok(GenerationOutput::Video {
            frames,
            fps,
            audio: None,
        })
    }
}

/// Shared descriptor surface for both A14B variants — CFG (per-expert guidance) + negative prompt,
/// UniPC/Euler samplers; H/W multiple of 16; **LoRA/LoKr supported** (sc-5167 — merged per-expert at
/// load; quant still deferred). `conditioning` differs per variant.
fn descriptor_for(variant: Variant) -> ModelDescriptor {
    ModelDescriptor {
        id: variant.id(),
        family: "wan",
        backend: "candle",
        modality: Modality::Video,
        capabilities: Capabilities {
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: false,
            conditioning: match variant {
                Variant::T2v => vec![],
                Variant::I2v => vec![ConditioningKind::Reference],
            },
            supports_lora: true,
            supports_lokr: true,
            // Curated `uni_pc` (sc-7296) → Wan's native UniPC; `euler` flow Euler. Legacy `unipc` alias.
            samplers: vec!["uni_pc", "euler", "unipc"],
            schedulers: vec![],
            supported_guidance_methods: vec![],
            min_size: 16,
            max_size: 1280,
            max_count: 1,
            mac_only: false,
            // Q4/Q8 packed MLX tiers (sc-10025): both dual-expert `WanTransformer` backbones load packed
            // via the shared packed-detect loaders; the tiers are pre-quantized (no on-the-fly quant).
            // Tier ingestion (MLX layout + key remap) is sc-10026.
            supported_quants: &[Quant::Q4, Quant::Q8],
            supports_kv_cache: false,
            requires_sigma_shift: false,
        },
    }
}

/// Wan2.2 T2V-A14B dual-expert MoE text→video descriptor.
pub fn descriptor_t2v_14b() -> ModelDescriptor {
    descriptor_for(Variant::T2v)
}

/// Wan2.2 I2V-A14B dual-expert MoE channel-concat image→video descriptor.
pub fn descriptor_i2v_14b() -> ModelDescriptor {
    descriptor_for(Variant::I2v)
}

fn load_variant(spec: &LoadSpec, variant: Variant) -> gen_core::Result<Box<dyn Generator>> {
    let id = variant.id();
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(format!(
                "{id} expects a snapshot directory (text_encoder/ transformer/ transformer_2/ vae/ \
                 tokenizer/), not a single .safetensors file"
            )));
        }
    };
    // No `spec.quantize` reject (sc-10025): the A14B quant matrix is packed-tier, not on-the-fly — a
    // q4/q8 tier is pre-quantized (the packed-detect loaders read its `.scales`), a dense tier loads
    // dense, so `spec.quantize` is a no-op tier-select marker resolved worker-side (mirrors ltx sc-9417).
    // I2V's conditioning image arrives per-request (`Conditioning::Reference`), not via `spec.control`;
    // the diffusers control/VACE overlays are not wired here.
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "{id} does not support control / VACE / IP-adapter overlays"
        )));
    }
    let device = candle_gen::default_device()?;
    Ok(Box::new(Wan14bGenerator {
        descriptor: descriptor_for(variant),
        variant,
        root,
        device,
        adapters: spec.adapters.clone(),
        comfyui: None,
        components: Mutex::new(None),
    }))
}

/// Construct a lazy candle Wan2.2 A14B generator that reads its **two DiT experts in place** from an
/// existing ComfyUI install (epic 10451 Phase 2c, sc-10671) — no copy, no re-download. `high_file` /
/// `low_file` are the user's ComfyUI high/low-noise expert files (native-Wan keys, companion scaled-fp8);
/// each is remapped + dequant'd to bf16 in memory ([`crate::comfyui`]) at component build.
///
/// `te_file` / `vae_file` optionally read the UMT5 text encoder + Wan VAE in place too (sc-10909): the
/// UMT5 (`umt5_xxl_fp8_e4m3fn_scaled`) is the same scaled-fp8 convention (dequant, no key remap), and
/// the VAE (`wan_2.1_vae.safetensors`) is native-WAN-VAE keys remapped to diffusers. When either is
/// `None` that component falls back to the `snapshot_dir` tier. `snapshot_dir` is a resident Wan2.2
/// A14B snapshot tier that always supplies at least the tiny UMT5 tokenizer (and the TE/VAE when their
/// files are absent). `variant` selects the T2V or I2V config (`patch_embedding` in-channels differ).
/// No adapters / control on this lane.
pub fn load_from_comfyui_experts(
    high_file: impl Into<PathBuf>,
    low_file: impl Into<PathBuf>,
    te_file: Option<PathBuf>,
    vae_file: Option<PathBuf>,
    snapshot_dir: impl Into<PathBuf>,
    i2v: bool,
) -> gen_core::Result<Box<dyn Generator>> {
    let variant = if i2v { Variant::I2v } else { Variant::T2v };
    let device = candle_gen::default_device()?;
    Ok(Box::new(Wan14bGenerator {
        descriptor: descriptor_for(variant),
        variant,
        root: snapshot_dir.into(),
        device,
        adapters: Vec::new(),
        comfyui: Some(std::sync::Arc::new(crate::comfyui::ComfyuiExperts {
            high_file: high_file.into(),
            low_file: low_file.into(),
            te_file,
            vae_file,
        })),
        components: Mutex::new(None),
    }))
}

/// Construct a lazy candle Wan2.2 T2V-A14B generator. `spec.weights` must be a [`WeightsSource::Dir`]
/// pointing at a `Wan-AI/Wan2.2-T2V-A14B-Diffusers` snapshot (`text_encoder/`, `transformer/`,
/// `transformer_2/`, `vae/`, `tokenizer/`).
pub fn load_t2v_14b(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_variant(spec, Variant::T2v)
}

/// Construct a lazy candle Wan2.2 I2V-A14B generator (channel-concat image→video). Same snapshot layout
/// as the T2V variant; the conditioning image arrives per-request as a `Conditioning::Reference`.
pub fn load_i2v_14b(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_variant(spec, Variant::I2v)
}

candle_gen::register_generators! {
    descriptor_t2v_14b => load_t2v_14b,
    descriptor_i2v_14b => load_i2v_14b,
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::registry;

    #[test]
    fn registers_both_as_candle_video() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        for (id, conditioning_len) in [(MODEL_ID_T2V_14B, 0usize), (MODEL_ID_I2V_14B, 1)] {
            let g = registry::load(id, &spec).expect("14b model is registered");
            assert_eq!(g.descriptor().id, id);
            assert_eq!(g.descriptor().family, "wan");
            assert_eq!(g.descriptor().backend, "candle");
            assert_eq!(g.descriptor().modality, Modality::Video);
            assert!(!g.descriptor().capabilities.mac_only);
            assert_eq!(
                g.descriptor().capabilities.conditioning.len(),
                conditioning_len
            );
        }
    }

    /// sc-8993: `cfg_active` is the single predicate gating the negative branch. CFG only affects the
    /// output at guidance > 1.0; at 1.0 the combine reduces to `pos` exactly and below 1.0 it's off, so
    /// both the encode-time and per-step negative work must be skipped. Defaults (3.0–4.0) keep it on.
    #[test]
    fn cfg_active_gates_negative_branch() {
        assert!(
            !cfg_active(1.0),
            "guidance 1.0 disables CFG (combine == pos)"
        );
        assert!(!cfg_active(0.0));
        assert!(!cfg_active(0.9));
        assert!(cfg_active(1.0001));
        assert!(cfg_active(3.0), "T2V low default keeps CFG on");
        assert!(cfg_active(4.0), "T2V high default keeps CFG on");
        // Per-expert independence: a mixed (low off / high on) request encodes+projects only the high
        // expert's negative, and vice-versa — mirroring the render's per-expert gating.
        let (g_low, g_high) = (1.0_f64, 4.0_f64);
        let neg_needed = cfg_active(g_low) || cfg_active(g_high);
        assert!(
            neg_needed,
            "shared UMT5 encode runs when either expert needs it"
        );
        assert!(
            !cfg_active(g_low),
            "low expert skips its negative projection"
        );
        assert!(
            cfg_active(g_high),
            "high expert keeps its negative projection"
        );
        // Both off: no negative work at all.
        assert!(!(cfg_active(1.0) || cfg_active(0.5)));
    }

    #[test]
    fn descriptor_surface() {
        let t2v = descriptor_t2v_14b();
        assert!(t2v.capabilities.supports_guidance);
        assert!(t2v.capabilities.supports_negative_prompt);
        assert!(!t2v.capabilities.supports_true_cfg);
        assert!(t2v.capabilities.conditioning.is_empty());
        assert!(t2v.capabilities.samplers.contains(&"uni_pc")); // curated spelling (sc-7296)
        assert!(t2v.capabilities.samplers.contains(&"unipc")); // legacy alias retained

        let i2v = descriptor_i2v_14b();
        assert!(i2v.capabilities.accepts(ConditioningKind::Reference));
    }

    #[test]
    fn validate_enforces_surface() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let t2v = registry::load(MODEL_ID_T2V_14B, &spec).unwrap();
        let ok = GenerationRequest {
            prompt: "a cat walking across a sunny garden".into(),
            width: 256,
            height: 256,
            guidance: Some(4.0),
            frames: Some(17),
            sampler: Some("uni_pc".into()),
            ..Default::default()
        };
        assert!(t2v.validate(&ok).is_ok());
        // Legacy `unipc` spelling stays accepted (sc-7296 alias).
        assert!(t2v
            .validate(&GenerationRequest {
                sampler: Some("unipc".into()),
                ..ok.clone()
            })
            .is_ok());
        for bad in [
            // empty prompt
            GenerationRequest::default(),
            // frames not ≡ 1 (mod 4)
            GenerationRequest {
                prompt: "x".into(),
                frames: Some(16),
                ..Default::default()
            },
            // size not a multiple of 16
            GenerationRequest {
                prompt: "x".into(),
                width: 300,
                ..Default::default()
            },
            // unadvertised sampler
            GenerationRequest {
                prompt: "x".into(),
                sampler: Some("dpmpp2m".into()),
                ..Default::default()
            },
            // over the MAX_AREA_14B envelope — 1280×1280 (both grid-aligned) is 2.2× the cap (sc-9028)
            GenerationRequest {
                prompt: "x".into(),
                width: 1280,
                height: 1280,
                frames: Some(17),
                sampler: Some("uni_pc".into()),
                ..Default::default()
            },
        ] {
            assert!(t2v.validate(&bad).is_err(), "should reject: {bad:?}");
        }

        // I2V rejects a request with no Reference image.
        let i2v = registry::load(MODEL_ID_I2V_14B, &spec).unwrap();
        assert!(i2v.validate(&ok).is_err(), "i2v needs a reference image");
    }

    /// The documented `MAX_AREA_14B` cap is actually enforced: an at-cap request passes and a
    /// grid-aligned over-cap request is rejected with an actionable message (sc-9028 / F-044).
    #[test]
    fn validate_enforces_max_area() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let t2v = registry::load(MODEL_ID_T2V_14B, &spec).unwrap();
        let base = GenerationRequest {
            prompt: "a cat walking across a sunny garden".into(),
            frames: Some(17),
            sampler: Some("uni_pc".into()),
            ..Default::default()
        };

        // Exactly at the cap (704×1280 = 901 120 px, both multiples of 16) is accepted.
        assert_eq!(704 * 1280, MAX_AREA_14B);
        assert!(t2v
            .validate(&GenerationRequest {
                width: 1280,
                height: 704,
                ..base.clone()
            })
            .is_ok());

        // Over the cap while both edges stay within the per-edge range (1280×1024 = 1 310 720 px,
        // both grid-aligned and ≤ 1280) is rejected specifically by the area check, with an
        // actionable message that names the cap.
        let err = t2v
            .validate(&GenerationRequest {
                width: 1280,
                height: 1024,
                ..base.clone()
            })
            .expect_err("over-area request must be rejected");
        let msg = err.to_string();
        assert!(msg.contains("max area"), "actionable message: {msg}");

        // The same cap applies to the I2V variant (both keep two resident 14B experts).
        let i2v = registry::load(MODEL_ID_I2V_14B, &spec).unwrap();
        assert!(
            i2v.validate(&GenerationRequest {
                width: 1280,
                height: 1024,
                ..base
            })
            .is_err(),
            "i2v enforces the same max-area cap"
        );
    }

    #[test]
    fn load_accepts_adapters_and_quant() {
        use candle_gen::gen_core::{AdapterKind, AdapterSpec};
        // LoRA/LoKr are supported (sc-5167) — load is lazy, so attaching adapters resolves OK
        // (the merge happens at the first `generate`).
        let lora = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_adapters(vec![
            AdapterSpec::new("/lora.safetensors".into(), 1.0, AdapterKind::Lora),
        ]);
        assert!(load_t2v_14b(&lora).is_ok());
        // Quant is now a no-op tier-select marker (packed-detect load, sc-10025) — a q4/q8 A14B tier is
        // pre-quantized, so `spec.quantize` no longer rejects; both experts load packed at ingestion.
        let quant = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_quant(Quant::Q8);
        assert!(load_i2v_14b(&quant).is_ok());
    }
}
