//! The Qwen-Image-**Edit** provider (sc-5487, epic 5480) — the candle (Windows/CUDA) sibling of
//! `mlx-gen-qwen-image`'s `QwenImageEdit`. Reference-conditioned image editing on `qwen_image_edit`:
//!
//! 1. **VL conditioning** — the reference + edit prompt go through the [`QwenVisionLanguageEncoder`]
//!    (vision tower + LM splice, Slice A) to `[1, S−64, 3584]` prompt embeds (the vision tower runs
//!    once, reused across the positive/negative prompts).
//! 2. **Dual-latent** — each reference is VAE-encoded + packed and concatenated **after** the noise
//!    over the sequence axis; the transformer's 3-axis RoPE spans `[noise] + references`
//!    ([`QwenTransformer::forward_edit`]). `zero_cond_t` (Edit-2511) modulates the conditioning
//!    tokens as clean; the original Edit / 2509 runs a single timestep (auto-detected from the
//!    transformer config).
//! 3. flow-match Euler denoise (true CFG with norm-rescale) → slice the noise prefix → VAE decode.
//!
//! A bespoke provider driven **directly** by the worker (like [`crate::control::QwenControl`] and
//! `candle_gen_sdxl::SdxlEdit`) — the registered `qwen_image` descriptor stays txt2img-only.
//!
//! NB: candle's CUDA attention indexes scores with i32, so a joint sequence whose scores tensor
//! exceeds `i32::MAX` elements (~2.1B) would silently corrupt — the shared `JointAttention` guards
//! this by chunking over query rows once the scores exceed `ATTN_SCORES_BUDGET` (sc-6217), and the
//! `edit_validate` high-res run confirms a coherent 1536² edit through that chunked path.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_gen::candle_core::{DType, Device, IndexOp, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::runtime::CancelFlag;
use candle_gen::gen_core::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use candle_gen::gen_core::{AdapterSpec, Image, Progress};
use candle_gen::{CandleError, Result};

use crate::config::{TextEncoderConfig, TransformerConfig, NEGATIVE_FALLBACK};
use crate::image_processor::{ImageInput, QwenImageProcessor};
use crate::pipeline;
use crate::transformer::QwenTransformer;
use crate::vae::{QwenVae, QwenVaeEncoder};
use crate::vision_language::{load_vision_language_encoder, QwenVisionLanguageEncoder};
use crate::vl_tokenizer::{
    condition_resize_dims, encode_reference_latents, preprocess_edit_image, tokenize_edit_text,
};

/// The transformer runs bf16 (native dtype); the VL encoder + VAE run f32.
const DIT_DTYPE: DType = DType::BF16;
const ENC_DTYPE: DType = DType::F32;

/// Paths to the Qwen-Image-Edit checkpoint.
pub struct QwenEditPaths {
    /// The `Qwen/Qwen-Image-Edit` diffusers snapshot dir (`text_encoder/` [LM + vision], `transformer/`,
    /// `vae/`, `tokenizer/`). The validated reference is `-2511`.
    pub root: PathBuf,
    /// LoRA/LoKr adapters folded into the MMDiT at load (sc-6220) — e.g. the Qwen-Image-Edit-2511
    /// Lightning distill, stacked ahead of any user adapters. **Empty** = the production (non-distilled)
    /// edit path: the transformer loads via the mmap fast path, byte-identical to before.
    pub adapters: Vec<AdapterSpec>,
}

/// One Qwen-Image-Edit generation request.
#[derive(Clone)]
pub struct QwenEditRequest {
    pub prompt: String,
    pub negative: String,
    pub width: u32,
    pub height: u32,
    pub steps: usize,
    /// True-CFG guidance scale. Ignored (CFG forced off) on the [`lightning`](Self::lightning) path.
    pub guidance: f32,
    pub seed: u64,
    /// The Qwen-Image-Edit-2511-Lightning few-step distill path (sc-6220): use the static-shift
    /// [`pipeline::lightning_sigmas`] schedule and run **CFG-off** (a single forward per step, no
    /// negative branch — the distill LoRA is CFG-distilled). The matching distill LoRA must be supplied
    /// via [`QwenEditPaths::adapters`]. `false` = the production multi-step true-CFG path.
    pub lightning: bool,
    pub cancel: CancelFlag,
}

impl Default for QwenEditRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            negative: String::new(),
            width: 1024,
            height: 1024,
            steps: 30,
            guidance: 4.0,
            seed: 0,
            lightning: false,
            cancel: CancelFlag::default(),
        }
    }
}

/// mmap a [`VarBuilder`] over every `.safetensors` in `root/sub` at `dtype`.
fn component_vb(
    root: &Path,
    sub: &str,
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder<'static>> {
    candle_gen::component_vb(root, sub, dtype, device, "qwen edit")
}

/// Load every `.safetensors` in `root/transformer` into one CPU tensor map (native dtype). The eager
/// load (vs the mmap [`component_vb`] fast path) is what lets the adapter deltas fold into the dense
/// weights before the MMDiT is built (sc-6220).
fn load_transformer_tensors(root: &Path) -> Result<HashMap<String, Tensor>> {
    let dir = root.join("transformer");
    if !dir.is_dir() {
        return Err(CandleError::Msg(format!(
            "qwen edit: snapshot is missing the transformer/ dir (at {})",
            root.display()
        )));
    }
    // Shared sorted-`.safetensors` resolver (sc-8999 / F-019); this path loads into a CPU map for
    // adapter merging (not the mmap fast path), so it keeps its own loop.
    let files = candle_gen::sorted_safetensors(&dir, "qwen edit")?;
    let mut map = HashMap::new();
    for f in &files {
        let part = candle_gen::candle_core::safetensors::load(f, &Device::Cpu)?;
        map.extend(part);
    }
    Ok(map)
}

/// Build the MMDiT, optionally folding LoRA/LoKr `adapters` into the dense weights at load (sc-6220).
/// With **no** adapters this is the mmap fast path (byte-identical to before). With adapters: load the
/// `transformer/` tensors onto CPU at native dtype, fold the deltas in f32
/// ([`crate::adapters::merge_adapters`]), then build the MMDiT from the merged map — each tensor cast to
/// `dtype` + moved to `device` as the VarBuilder serves it, so peak GPU is unchanged vs the mmap path
/// (mirrors `candle-gen-sdxl`'s `load_instantid_unet_with_adapters`). A non-empty `adapters` slice that
/// matches no MMDiT module errors (it never renders an unadapted image silently).
fn load_transformer(
    root: &Path,
    adapters: &[AdapterSpec],
    dtype: DType,
    device: &Device,
) -> Result<QwenTransformer> {
    let cfg = TransformerConfig::qwen_image();
    // The DiT packed-detects each `Linear`: an MLX-packed edit tier (`SceneWorks/qwen-image-edit-2511
    // -mlx` q4/q8) loads straight from the packed parts at the `group_size` read from
    // `transformer/config.json` (64); a dense Edit snapshot loads unchanged (the group size is inert on
    // the dense path). See `crate::transformer_group_size`.
    let gs = crate::transformer_group_size(&root.join("transformer"));
    if adapters.is_empty() {
        return Ok(QwenTransformer::new_gs(
            &cfg,
            component_vb(root, "transformer", dtype, device)?,
            gs,
        )?);
    }
    // Adapter-merge path: a LoRA/LoKr overlay implies a **dense** base checkpoint (the deltas are f32
    // and can't fold into u32 packed codes), so the merged VarBuilder is dense and packed-detect takes
    // the dense path for every projection. Group size stays the default (inert here).
    let mut tensors = load_transformer_tensors(root)?;
    // Discard the merge report — the silent twin (`candle-gen-z-image`'s
    // `transformer_vb_with_adapters`) does the same; a non-matching adapter surface already errors
    // inside `merge_adapters`, so library code stays quiet on stderr (sc-9035 / F-051).
    crate::adapters::merge_adapters(&mut tensors, adapters)?;
    let vb = VarBuilder::from_tensors(tensors, dtype, device);
    Ok(QwenTransformer::new_gs(&cfg, vb, gs)?)
}

/// `transformer/config.json` `zero_cond_t` (Edit-2511 = true; the original Edit / 2509 omit it).
///
/// A genuinely-absent `transformer/config.json` (the original Edit / 2509 snapshots don't gate on it)
/// or an absent `zero_cond_t` key defaults to `false`. But a *present-but-corrupt* config — I/O error,
/// malformed JSON, or a `zero_cond_t` of the wrong type — errors loudly rather than silently switching
/// an Edit-2511 render to the 2509 single-timestep modulation on a damaged snapshot (sc-9010 / F-073).
fn read_zero_cond_t(root: &Path) -> Result<bool> {
    let path = root.join("transformer/config.json");
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        // Absent config ⇒ documented default (2509 / original Edit).
        Err(ref e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(e) => {
            return Err(CandleError::Msg(format!(
                "qwen edit: read {}: {e}",
                path.display()
            )))
        }
    };
    let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| {
        CandleError::Msg(format!(
            "qwen edit: parse {} (corrupt snapshot?): {e}",
            path.display()
        ))
    })?;
    match v.get("zero_cond_t") {
        // Key absent ⇒ documented default.
        None | Some(serde_json::Value::Null) => Ok(false),
        Some(b) => b.as_bool().ok_or_else(|| {
            CandleError::Msg(format!(
                "qwen edit: `zero_cond_t` in {} must be a bool, got {b}",
                path.display()
            ))
        }),
    }
}

/// Locate the assembled HF `tokenizer.json` (sc-6294). The original `Qwen-Image-Edit` ships it under
/// `tokenizer/`, but `Qwen-Image-Edit-2511` ships the assembled file only inside the Qwen2.5-VL
/// processor bundle (`processor/tokenizer.json`) — the `tokenizer/` dir there carries just the BPE
/// source (`merges.txt`/`vocab.json`). The two locations are byte-identical (same SHA256), so prefer
/// `tokenizer/`, then fall back to `processor/`, so a whole-repo -2511 download loads without a
/// hand-staged tokenizer.json.
fn tokenizer_json_path(root: &Path) -> Result<PathBuf> {
    for rel in ["tokenizer/tokenizer.json", "processor/tokenizer.json"] {
        let p = root.join(rel);
        if p.is_file() {
            return Ok(p);
        }
    }
    Err(CandleError::Msg(format!(
        "qwen edit: no tokenizer.json under tokenizer/ or processor/ (at {})",
        root.display()
    )))
}

/// The loaded Qwen-Image-Edit model: the VL conditioning encoder, the MMDiT, the VAE (decode) + VAE
/// encoder (reference dual-latent), the image processor + tokenizer.
pub struct QwenEdit {
    device: Device,
    te_cfg: TextEncoderConfig,
    vl_encoder: QwenVisionLanguageEncoder,
    transformer: QwenTransformer,
    vae: QwenVae,
    vae_encoder: QwenVaeEncoder,
    processor: QwenImageProcessor,
    tokenizer: TextTokenizer,
    zero_cond_t: bool,
}

impl QwenEdit {
    /// Load the Qwen-Image-Edit components from a snapshot dir.
    pub fn load(paths: &QwenEditPaths) -> Result<Self> {
        let device = candle_gen::default_device()?;
        let root = &paths.root;
        let te_cfg = TextEncoderConfig::qwen_image();

        let vl_encoder = load_vision_language_encoder(root, &device)?;
        let transformer = load_transformer(root, &paths.adapters, DIT_DTYPE, &device)?;
        let vae = QwenVae::new(component_vb(root, "vae", ENC_DTYPE, &device)?)?;
        let vae_encoder = QwenVaeEncoder::new(component_vb(root, "vae", ENC_DTYPE, &device)?)?;
        let tokenizer = TextTokenizer::from_file(
            tokenizer_json_path(root)?,
            TokenizerConfig {
                max_length: te_cfg.max_length,
                pad_token_id: te_cfg.pad_token_id,
                chat_template: ChatTemplate::QwenImage,
                pad_to_max_length: false,
            },
        )
        .map_err(|e| CandleError::Msg(format!("qwen edit: load tokenizer: {e}")))?;

        Ok(Self {
            zero_cond_t: read_zero_cond_t(root)?,
            device,
            te_cfg,
            vl_encoder,
            transformer,
            vae,
            vae_encoder,
            processor: QwenImageProcessor::default(),
            tokenizer,
        })
    }

    /// VL-encode one prompt against the precomputed `vision` embeds → `[1, S−64, 3584]` at the DiT
    /// dtype. `n_image_tokens` is the shared `<|image_pad|>` run length (from the image preprocess).
    fn encode_prompt(
        &self,
        prompt: &str,
        n_image_tokens: usize,
        vision: &Tensor,
    ) -> Result<Tensor> {
        let ids = tokenize_edit_text(&self.tokenizer, prompt, n_image_tokens)?;
        let len = ids.len();
        let input_ids = Tensor::from_vec(ids, (1, len), &self.device)?;
        let embeds = self.vl_encoder.encode_with_vision(&input_ids, vision)?;
        Ok(embeds.to_dtype(DIT_DTYPE)?)
    }

    /// Reference-conditioned edit. `references` is the (validated non-empty) reference image set: the
    /// **first** drives the VL prompt embeds, **all** are VAE-encoded into the dual-latent sequence,
    /// and the **last** sets the condition resolution (the fork's `_compute_dimensions`).
    pub fn generate(
        &self,
        req: &QwenEditRequest,
        references: &[Image],
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        if req.cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }
        let first = references.first().ok_or_else(|| {
            CandleError::Msg("qwen edit: at least one reference image is required".into())
        })?;
        let last = references.last().expect("non-empty checked");
        let (lat_h, lat_w) = pipeline::latent_dims(req.width, req.height);

        // VL conditioning: preprocess the first reference once (image-only), run the vision tower once,
        // then encode the positive (+ negative for CFG) prompts reusing the vision embeds.
        let edit_img = preprocess_edit_image(&self.processor, image_input(first), &self.device)?;
        let vision = self
            .vl_encoder
            .encode_vision(&edit_img.pixel_values, &[edit_img.grid])?;
        let pos = self.encode_prompt(&req.prompt, edit_img.n_image_tokens, &vision)?;
        // CFG-off on the lightning path: the distill LoRA is CFG-distilled, so a single forward per
        // step (no negative branch) — matching the MLX lightning recipe (sc-6220).
        let neg = if req.guidance > 1.0 && !req.lightning {
            let n = if req.negative.trim().is_empty() {
                NEGATIVE_FALLBACK
            } else {
                req.negative.as_str()
            };
            Some(self.encode_prompt(n, edit_img.n_image_tokens, &vision)?)
        } else {
            None
        };

        // Dual-latent references (static across steps): VAE-encode each reference at the VL condition
        // resolution (from the last reference's aspect), pack, and concatenate over the sequence axis.
        let (vl_w, vl_h) = condition_resize_dims(last.width as usize, last.height as usize);
        let mut packed = Vec::with_capacity(references.len());
        let mut cond_grids = Vec::with_capacity(references.len());
        for im in references {
            let (latents, grid) = encode_reference_latents(
                &self.vae_encoder,
                image_input(im),
                vl_w as u32,
                vl_h as u32,
                &self.device,
            )?;
            packed.push(latents.to_dtype(DIT_DTYPE)?);
            cond_grids.push(grid);
        }
        let static_latents = if packed.len() == 1 {
            packed.pop().expect("len checked")
        } else {
            Tensor::cat(&packed.iter().collect::<Vec<_>>(), 1)?
        };
        let noise_seq = lat_h * lat_w;

        // Lightning uses the static-shift schedule (resolution-independent); production uses the
        // dynamic-μ schedule (sc-6220). Routed through the unified curated sampler/scheduler framework
        // (epic 7114 P4, sc-7123): `native` is whichever legacy schedule the path uses (returned
        // verbatim — N1 byte-exact for the default), and `mu` steers the (non-default) curated scheduler
        // axis by the path's own shift. The bespoke edit provider has no `req.sampler`/`req.scheduler`
        // surface yet, so both stay `None` (the N1 default: `euler` over the native schedule —
        // algebraically the legacy `euler_step` loop). The model is fed the raw sigma (`Sigma`
        // convention); Qwen-Image-Edit is **true CFG**, and the dual-latent concat/slice (concatenate the
        // updating noise with the static reference latents over the sequence axis, then slice the noise
        // prefix post-forward) lives — together with the pos/neg/blend — inside the `predict` closure.
        let (native, mu) = if req.lightning {
            (
                pipeline::lightning_sigmas(req.steps),
                pipeline::lightning_mu(),
            )
        } else {
            (
                pipeline::qwen_sigmas(req.steps, req.width, req.height),
                pipeline::qwen_mu(req.width, req.height),
            )
        };
        let sigmas = candle_gen::resolve_flow_schedule(None, mu, req.steps, &native);
        let latents = pipeline::create_noise(req.seed, req.width, req.height, &self.device)?
            .to_dtype(DIT_DTYPE)?;

        let latents = candle_gen::run_flow_sampler(
            None,
            candle_gen::gen_core::sampling::TimestepConvention::Sigma,
            &sigmas,
            latents,
            req.seed,
            &req.cancel,
            on_progress,
            |latents, sigma| -> Result<Tensor> {
                // Concatenate the (updating) noise with the (static) reference latents over the sequence.
                let joint = Tensor::cat(&[latents, &static_latents], 1)?;
                let pos_v = self
                    .transformer
                    .forward_edit(
                        &joint,
                        &pos,
                        sigma,
                        lat_h,
                        lat_w,
                        &cond_grids,
                        self.zero_cond_t,
                    )?
                    .narrow(1, 0, noise_seq)?;
                match &neg {
                    Some(neg) => {
                        let neg_v = self
                            .transformer
                            .forward_edit(
                                &joint,
                                neg,
                                sigma,
                                lat_h,
                                lat_w,
                                &cond_grids,
                                self.zero_cond_t,
                            )?
                            .narrow(1, 0, noise_seq)?;
                        Ok(pipeline::compute_guided_noise(
                            &pos_v,
                            &neg_v,
                            req.guidance,
                        )?)
                    }
                    None => Ok(pos_v),
                }
            },
        )?;

        let _ = &self.te_cfg; // kept for symmetry with the other providers' config plumbing
        on_progress(Progress::Decoding);
        let lat = pipeline::unpack_latents(&latents, req.width, req.height)?;
        let decoded = self.vae.decode(&lat)?;
        to_image(&decoded)
    }
}

/// Borrow an [`Image`] as an [`ImageInput`] (RGB uint8 HWC).
fn image_input(im: &Image) -> ImageInput<'_> {
    ImageInput {
        data: &im.pixels,
        height: im.height as usize,
        width: im.width as usize,
    }
}

/// VAE output `[1, 3, H, W]` in `[-1, 1]` → an RGB8 [`Image`].
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_defaults() {
        let r = QwenEditRequest::default();
        assert_eq!((r.width, r.height), (1024, 1024));
        assert_eq!(r.steps, 30);
        assert!(!r.cancel.is_cancelled());
    }

    fn zero_cond_t_tmp(name: &str) -> PathBuf {
        let tmp = std::env::temp_dir().join(format!(
            "qwen_edit_zct_{name}_{}_{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(tmp.join("transformer")).unwrap();
        tmp
    }

    #[test]
    fn zero_cond_t_defaults_false_when_config_absent() {
        // A nonexistent config (dir/file) → false, the original Qwen-Image-Edit / 2509 path.
        assert!(!read_zero_cond_t(Path::new("/nonexistent")).unwrap());
    }

    #[test]
    fn zero_cond_t_defaults_false_when_key_absent() {
        // Config present but the key genuinely absent (a valid 2509 config.json) → documented default.
        let tmp = zero_cond_t_tmp("keyabsent");
        std::fs::write(
            tmp.join("transformer/config.json"),
            br#"{"num_layers": 60}"#,
        )
        .unwrap();
        assert!(!read_zero_cond_t(&tmp).unwrap());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn zero_cond_t_reads_present_value() {
        // Edit-2511 config with the key set true → true.
        let tmp = zero_cond_t_tmp("present");
        std::fs::write(
            tmp.join("transformer/config.json"),
            br#"{"zero_cond_t": true}"#,
        )
        .unwrap();
        assert!(read_zero_cond_t(&tmp).unwrap());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn zero_cond_t_errors_on_corrupt_json() {
        // A present-but-malformed config (partial download) must error, NOT silently downgrade to 2509.
        let tmp = zero_cond_t_tmp("corrupt");
        std::fs::write(tmp.join("transformer/config.json"), b"{ this is not json").unwrap();
        assert!(read_zero_cond_t(&tmp).is_err());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn zero_cond_t_errors_on_wrong_type() {
        // `zero_cond_t` present but the wrong type → error naming the field, not a silent false.
        let tmp = zero_cond_t_tmp("wrongtype");
        std::fs::write(
            tmp.join("transformer/config.json"),
            br#"{"zero_cond_t": "yes"}"#,
        )
        .unwrap();
        let err = read_zero_cond_t(&tmp).unwrap_err().to_string();
        assert!(
            err.contains("zero_cond_t"),
            "error should name the field: {err}"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn tokenizer_json_path_prefers_tokenizer_then_processor() {
        // -2511 ships the assembled tokenizer.json only under processor/ (sc-6294).
        let tmp = std::env::temp_dir().join(format!("qwen_edit_tok_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(tmp.join("processor")).unwrap();
        std::fs::write(tmp.join("processor/tokenizer.json"), b"{}").unwrap();
        assert!(tokenizer_json_path(&tmp)
            .unwrap()
            .ends_with("processor/tokenizer.json"));

        // When tokenizer/ also has it (the original Edit), that location wins.
        std::fs::create_dir_all(tmp.join("tokenizer")).unwrap();
        std::fs::write(tmp.join("tokenizer/tokenizer.json"), b"{}").unwrap();
        assert!(tokenizer_json_path(&tmp)
            .unwrap()
            .ends_with("tokenizer/tokenizer.json"));

        // Neither present → a descriptive error rather than a silent panic.
        let _ = std::fs::remove_dir_all(&tmp);
        assert!(tokenizer_json_path(&tmp).is_err());
    }
}
