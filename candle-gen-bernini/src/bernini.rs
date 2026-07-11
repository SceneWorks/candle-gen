//! sc-10995 (capstone): the **full Bernini** Generator (`gen_core::load("bernini")`) — the registered
//! pipeline that strings the whole planner → renderer stack together, the candle sibling of
//! `mlx-gen-bernini/src/bernini.rs` (sc-5145). Mirrors `BerniniPipeline.__call__`:
//!
//! ```text
//!   preprocess (ViT tower on the sources + gen-target grid; VAE on the sources)
//!     → 3 planner streams (cond / uncond / imgcond)            [build_stream]
//!     → MAR semantic-planning loop                             [crate::mar::sample_vit_embed]
//!     → 4 renderer prompt streams + UMT5 concat_with_zero_init [crate::assembly]
//!     → ViT-conditioned dual-expert APG denoise                [denoise_bernini_wvitcfg]
//!     → z16 VAE decode → image (1 frame) / video
//! ```
//!
//! The planner ([`BerniniPlanner`]) is Qwen2.5-VL-7B (penultimate extractor) + [`MlpConnector`] +
//! [`DiffLossFm`] clip-diff head + the MAR mask token; the renderer is the existing dual-expert
//! [`WanTransformer`] MoE + UMT5 + z16 VAE. This assembles the `*_wapg` / `v2v_apg` ViT-conditioned
//! guidance modes ([`crate::forward::vit_one_step`]) that sc-11004 flagged as pending — the planner's
//! ViT-guidance feeding [`denoise_bernini_wvitcfg`].
//!
//! **Weights.** The renderer tier (`transformer/` `transformer_2/` `text_encoder/` `vae/` `tokenizer/`)
//! is produced by [`crate::convert::build_bernini_candle_tier`]; the planner components (`mllm/`
//! `connector/` `vit_decoder/` `mask_tokens.safetensors` + `qwen2_5_vl_config.json`) are the additional
//! candle turnkey layout this loader reads (see [`BerniniPlanner::load`]). The full 168 GB
//! `ByteDance/Bernini-Diffusers` weights are NOT downloaded here; real-weight semantic GPU-val is sc-11003.
//!
//! **Validation without real weights.** The registered generator loads lazily, so registration, the
//! descriptor, and `validate` are all exercised weightlessly. The genuinely new capstone compute (the
//! ViT-conditioned renderer denoise, [`vit_one_step`] / [`denoise_bernini_wvitcfg`]) is CPU-golden and
//! shape-tested on tiny synthetic experts. The planner seams (backbone, vision, connector, clip_diff,
//! MAR, assembly, template, process) each carry their own merged CPU parity goldens; this module wires
//! them.

use std::path::{Path, PathBuf};

use image::RgbImage;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use candle_gen::gen_core::{
    self, CancelFlag, Capabilities, Conditioning, ConditioningKind, GenerationOutput,
    GenerationRequest, Generator, Image, LoadSpec, Modality, ModelDescriptor, Progress, Quant,
    WeightsSource,
};
use candle_gen::{CandleError, Result as CResult};

use candle_gen_wan::config::{
    TextEncoderConfig, TransformerConfig, Vae16Config, DEFAULT_FRAMES_14B, NUM_TRAIN_TIMESTEPS,
    VAE16_STRIDE_SPATIAL, VAE16_STRIDE_TEMPORAL,
};
use candle_gen_wan::pipeline::{create_noise, frames_to_images};
use candle_gen_wan::rope::assign_source_ids;
use candle_gen_wan::scheduler::{flow_sigmas, FlowScheduler, Sampler};
use candle_gen_wan::text_encoder::Umt5Encoder;
use candle_gen_wan::transformer::WanTransformer;
use candle_gen_wan::vae16::WanVae16;

use crate::assembly::{concat_with_zero_init, format_mllm_inputs_embeds};
use crate::clip_diff::DiffLossFm;
use crate::config::BerniniKnobs;
use crate::connector::MlpConnector;
use crate::forward::{vit_one_step, PackedForward, VitGuidanceParams, VitMode, VitStreams};
use crate::mar::{mar_schedule, post_process_input_embeds, sample_vit_embed, StreamState, VitCfg};
use crate::preprocess::{encode_image, encode_videoclip};
use crate::process::{
    build_attention_mask_4d, generate_unified_inputs, mrope_position_ids, MRopeConfig,
};
use crate::qwen2_5_vl::{Qwen25VlText, QwenVlTextConfig};
use crate::template::BerniniTemplate;
use crate::vision::{VisionConfig, VisionTower};
use crate::vit_preprocess::{
    normalized_frame, pack_patches, preprocess_image, smart_resize, smart_video_nframes, FACTOR,
    IMAGE_MEAN, IMAGE_STD, MERGE_SIZE, PATCH_SIZE, TEMPORAL_PATCH_SIZE,
};

/// SceneWorks engine id — matches `mlx-gen-bernini`'s full pipeline so a consumer resolves the same
/// engine across backends.
pub const MODEL_ID: &str = "bernini";

/// The A14B DiT emits 16-channel latents (z16 VAE).
const Z_DIM: usize = 16;
/// The planner + renderer contexts run bf16 on the DiT side; UMT5 + z16 VAE run f32.
const PLANNER_DTYPE: DType = DType::BF16;
const ENC_DTYPE: DType = DType::F32;
const VAE_DTYPE: DType = DType::F32;
const DIT_DTYPE: DType = DType::BF16;

/// Full-pipeline CLI defaults (`bernini/cli.py` for the `BerniniPipeline` path). A request's `guidance`
/// overrides `omega_txt`; the rest are fixed until the worker surfaces them.
struct FullDefaults;
impl FullDefaults {
    const STEPS: usize = 40;
    const OMEGA_VID: f32 = 1.25;
    const OMEGA_IMG: f32 = 4.5;
    const OMEGA_TXT: f32 = 4.0;
    const OMEGA_TGT: f32 = 0.5;
    const OMEGA_SCALE: f32 = 0.8;
    const PLANNING_STEP: usize = 25;
    const VIT_TXT_CFG: f32 = 1.2;
    const VIT_IMG_CFG: f32 = 1.0;
    const VIT_DENOISING_STEP: usize = 5;
    const FLOW_SHIFT: f64 = 5.0;
    const ETA: f32 = 1.0;
    const NORM_THRESHOLD: f32 = 50.0;
    /// Source-media ViT pixel budget (`preprocess_inputs` `vit_min/max_pixels`).
    const VIT_MIN_PIXELS: i64 = 3136;
    const VIT_MAX_PIXELS: i64 = 50176;
    const FPS: u32 = 16;
}

/// Planner knobs read from the `bernini_planner.json` sidecar (else the package-config defaults).
struct PlannerKnobs {
    max_sequence_length: usize,
    num_mask_token: i32,
    clip_diff_depth: usize,
    clip_diff_in_channels: usize,
    clip_diff_shift: f32,
}

impl PlannerKnobs {
    fn from_dir(root: &Path) -> Self {
        let v: serde_json::Value = std::fs::read(root.join("bernini_planner.json"))
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or(serde_json::Value::Null);
        let i = |k: &str, d: i64| v.get(k).and_then(serde_json::Value::as_i64).unwrap_or(d);
        let cd = v
            .get("clip_diff_cfg")
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        let cdi = |k: &str, d: i64| cd.get(k).and_then(serde_json::Value::as_i64).unwrap_or(d);
        let cdf = |k: &str, d: f64| cd.get(k).and_then(serde_json::Value::as_f64).unwrap_or(d);
        Self {
            max_sequence_length: i("max_sequence_length", 512).max(1) as usize,
            num_mask_token: i("num_mask_token", 4096) as i32,
            // `vit_decoder` (`SimpleMLPAdaLN`) depth is fixed at 16 in the released checkpoint; the
            // sidecar carries z_channels / shift verbatim.
            clip_diff_depth: 16,
            clip_diff_in_channels: cdi("z_channels", 3584) as usize,
            clip_diff_shift: cdf("shift", 2.0) as f32,
        }
    }
}

/// Read the planner's MRoPE / token-id config from `qwen2_5_vl_config.json`.
fn read_mrope_config(path: &Path) -> MRopeConfig {
    let d = MRopeConfig::default();
    let v: serde_json::Value = std::fs::read(path)
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or(serde_json::Value::Null);
    let vc = v.get("vision_config").unwrap_or(&v);
    let i = |o: &serde_json::Value, k: &str, dv: i64| {
        o.get(k).and_then(serde_json::Value::as_i64).unwrap_or(dv)
    };
    let f = |o: &serde_json::Value, k: &str, dv: f64| {
        o.get(k).and_then(serde_json::Value::as_f64).unwrap_or(dv)
    };
    MRopeConfig {
        spatial_merge_size: i(vc, "spatial_merge_size", d.spatial_merge_size),
        tokens_per_second: f(vc, "tokens_per_second", d.tokens_per_second),
        image_token_id: i(&v, "image_token_id", d.image_token_id),
        video_token_id: i(&v, "video_token_id", d.video_token_id),
        vision_start_token_id: i(&v, "vision_start_token_id", d.vision_start_token_id),
    }
}

/// The loaded Bernini semantic planner: Qwen2.5-VL backbone + vision tower + connector + clip-diff head
/// + MAR mask token + the host-side templating / MRoPE config.
struct BerniniPlanner {
    backbone: Qwen25VlText,
    vision: VisionTower,
    connector: MlpConnector,
    clip_diff: DiffLossFm,
    /// A single MAR mask token `[1, 1, H]` (`self.mask_tokens[:, :1]`, broadcast over the target).
    mask_token: Tensor,
    mrope: MRopeConfig,
    template: BerniniTemplate,
    knobs: PlannerKnobs,
}

impl BerniniPlanner {
    /// Load the planner components from the candle turnkey layout under `root`:
    ///   - `mllm/` — the Qwen2.5-VL backbone (`model.*`) + vision tower (`visual.*`) + `tokenizer.json`.
    ///   - `connector/` — the `proj_gen.*` / `pred_vit.*` MLP connector.
    ///   - `vit_decoder/` — the clip-diff flow head (`net.*`).
    ///   - `mask_tokens.safetensors` — the MAR mask tokens (`mask_tokens`).
    ///   - `qwen2_5_vl_config.json` — the backbone / vision / MRoPE config.
    ///
    /// Loaded dense bf16 (planner quantization is a follow-up; the renderer experts carry the dominant
    /// footprint and quantize separately via the packed-detect tier).
    fn load(root: &Path, device: &Device) -> CResult<Self> {
        let cfg_path = root.join("qwen2_5_vl_config.json");
        let qcfg = QwenVlTextConfig::from_config_json(&cfg_path)?;
        let vcfg = VisionConfig::from_config_json(&cfg_path)?;

        let mllm_vb = candle_gen::component_vb(root, "mllm", PLANNER_DTYPE, device, MODEL_ID)?;
        let backbone = Qwen25VlText::new(qcfg, mllm_vb.pp("model"))?;
        let vision = VisionTower::new(vcfg, mllm_vb.pp("visual"))?;

        let conn_vb = candle_gen::component_vb(root, "connector", PLANNER_DTYPE, device, MODEL_ID)?;
        let connector = MlpConnector::new(conn_vb)?;

        let knobs = PlannerKnobs::from_dir(root);
        let vd_vb = candle_gen::component_vb(root, "vit_decoder", PLANNER_DTYPE, device, MODEL_ID)?;
        let clip_diff = DiffLossFm::new(
            vd_vb.pp("net"),
            knobs.clip_diff_depth,
            knobs.clip_diff_in_channels,
            knobs.clip_diff_shift,
        )?;

        // `self.mask_tokens[:, :1]` — a single mask token, broadcast over the n_query target slots.
        let mask_map = candle_gen::candle_core::safetensors::load(
            root.join("mask_tokens.safetensors"),
            device,
        )?;
        let mask_token = mask_map
            .get("mask_tokens")
            .ok_or_else(|| {
                CandleError::Msg("bernini: mask_tokens.safetensors missing `mask_tokens`".into())
            })?
            .narrow(1, 0, 1)?
            .to_dtype(PLANNER_DTYPE)?;

        let template =
            BerniniTemplate::from_tokenizer_file(root.join("mllm").join("tokenizer.json"))?;
        Ok(Self {
            backbone,
            vision,
            connector,
            clip_diff,
            mask_token,
            mrope: read_mrope_config(&cfg_path),
            template,
            knobs,
        })
    }
}

/// One preprocessed source visual: its ViT features (planner conditioning) + grid + VAE latent (renderer
/// conditioning).
struct SourceVisual {
    /// `[merged, H]` planner ViT features.
    vit_feat: Tensor,
    /// `[t, h, w]` ViT grid (drives the token count + MRoPE).
    vit_grid: [i32; 3],
    /// `[1, 16, T, H8, W8]` normalized VAE latent (the renderer source-conditioning latent).
    vae_latent: Tensor,
    /// Original source `(height, width)` (for the conversation's `image` message).
    hw: (i64, i64),
}

/// Convert a public RGB8 [`Image`] to an `image::RgbImage`.
fn to_rgb(img: &Image) -> CResult<RgbImage> {
    RgbImage::from_raw(img.width, img.height, img.pixels.clone())
        .ok_or_else(|| CandleError::Msg("bernini: malformed RGB8 conditioning image".into()))
}

/// ViT-encode one image → `[merged, H]` + grid.
fn vit_encode_image(
    planner: &BerniniPlanner,
    rgb: &RgbImage,
    dev: &Device,
) -> CResult<(Tensor, [i32; 3])> {
    let (pixels, grid) = preprocess_image(
        rgb,
        FullDefaults::VIT_MIN_PIXELS,
        FullDefaults::VIT_MAX_PIXELS,
        IMAGE_MEAN,
        IMAGE_STD,
        dev,
    )?;
    let g = [grid[0] as usize, grid[1] as usize, grid[2] as usize];
    let feat = planner.vision.forward(&pixels, &[g])?;
    Ok((feat, grid))
}

/// ViT-encode (already ViT-sampled) video frames → `[merged, H]` + grid. All frames are `smart_resize`d
/// to a common size, normalized, stacked `[F, 3, H, W]`, then `pack_patches` (temporal 2).
fn vit_encode_video(
    planner: &BerniniPlanner,
    frames: &[RgbImage],
    dev: &Device,
) -> CResult<(Tensor, [i32; 3])> {
    let (h0, w0) = (frames[0].height() as i64, frames[0].width() as i64);
    let (rh, rw) = smart_resize(
        h0,
        w0,
        FACTOR,
        FullDefaults::VIT_MIN_PIXELS,
        FullDefaults::VIT_MAX_PIXELS,
    );
    let mut chw_t = Vec::with_capacity(frames.len());
    for f in frames {
        let resized = image::imageops::resize(
            f,
            rw as u32,
            rh as u32,
            image::imageops::FilterType::CatmullRom,
        );
        chw_t.push(normalized_frame(
            resized.as_raw(),
            rh,
            rw,
            IMAGE_MEAN,
            IMAGE_STD,
            dev,
        )?);
    }
    let refs: Vec<&Tensor> = chw_t.iter().collect();
    let frames_t = Tensor::cat(&refs, 0)?; // [F, 3, H, W]
    let (pixels, grid) = pack_patches(&frames_t, PATCH_SIZE, TEMPORAL_PATCH_SIZE, MERGE_SIZE)?;
    let g = [grid[0] as usize, grid[1] as usize, grid[2] as usize];
    let feat = planner.vision.forward(&pixels, &[g])?;
    Ok((feat, grid))
}

/// The gen-target ViT grid `[t, h, w]` (sizes `n_query`, the MAR token count). Image target (`frames ==
/// 1`) ⇒ `t = 1`; video target samples `vit_fps` (= fps/8) frames from the `num_frames` clip, `t =
/// vit_frames / temporal`. The spatial grid is `smart_resize` of the output H/W under the ViT budget.
fn gen_target_grid(height: i64, width: i64, frames: usize, fps: u32) -> [i32; 3] {
    let (rh, rw) = smart_resize(
        height,
        width,
        FACTOR,
        FullDefaults::VIT_MIN_PIXELS,
        FullDefaults::VIT_MAX_PIXELS,
    );
    let gh = (rh / PATCH_SIZE) as i32;
    let gw = (rw / PATCH_SIZE) as i32;
    let t = if frames <= 1 {
        1
    } else {
        let vit_fps = (fps / 8).max(1) as f64;
        let vit_frames = smart_video_nframes(
            frames as i64,
            fps as f64,
            vit_fps,
            Some(TEMPORAL_PATCH_SIZE),
            None,
            Some(frames as i64),
            false,
        )
        .len() as i64;
        (vit_frames / TEMPORAL_PATCH_SIZE).max(1) as i32
    };
    [t, gh, gw]
}

/// Merged-token count of a ViT grid (`t·h·w / merge²`).
fn grid_tokens(grid: [i32; 3]) -> i64 {
    let m2 = (MERGE_SIZE * MERGE_SIZE) as i32;
    (grid[0] * grid[1] * grid[2] / m2) as i64
}

/// Build one planner stream's [`StreamState`] (cond / uncond / imgcond). `images`/`videos` are the
/// **present** input source visuals (empty for uncond/imgcond); `prompt` is the stream's text (raw for
/// cond/imgcond, negative for uncond). `gen_grid` is the gen-target ViT grid; `gen_is_video` selects the
/// gen slot kind. The gen-target ViT features are zeros (masked by `post_process_input_embeds`).
#[allow(clippy::too_many_arguments)]
fn build_stream(
    planner: &BerniniPlanner,
    task: &str,
    prompt: &str,
    images: &[SourceVisual],
    videos: &[SourceVisual],
    gen_grid: [i32; 3],
    gen_is_video: bool,
    out_h: i64,
    out_w: i64,
    dev: &Device,
) -> CResult<(StreamState, i32)> {
    let image_hw: Vec<(i64, i64)> = images.iter().map(|s| s.hw).collect();
    let output_t = if gen_is_video { 2 } else { 1 };
    let conv = generate_unified_inputs(prompt, &image_hw, videos.len(), output_t, out_h, out_w);

    let mut image_grids: Vec<[i32; 3]> = images.iter().map(|s| s.vit_grid).collect();
    let mut video_grids: Vec<[i32; 3]> = videos.iter().map(|s| s.vit_grid).collect();
    if gen_is_video {
        video_grids.push(gen_grid);
    } else {
        image_grids.push(gen_grid);
    }
    let image_token_nums: Vec<i64> = image_grids.iter().map(|&g| grid_tokens(g)).collect();
    let video_token_nums: Vec<i64> = video_grids.iter().map(|&g| grid_tokens(g)).collect();

    let tout =
        planner
            .template
            .encode_messages(&conv, &image_token_nums, &video_token_nums, task)?;
    let l = tout.input_ids.len();

    // visual_embeds: conversation order = [video feats, image feats, gen-target zeros].
    let h_vit = planner.knobs.clip_diff_in_channels;
    let gen_tokens = grid_tokens(gen_grid) as usize;
    let mut feats: Vec<Tensor> = Vec::new();
    for v in videos {
        feats.push(v.vit_feat.clone());
    }
    for im in images {
        feats.push(im.vit_feat.clone());
    }
    feats.push(Tensor::zeros((gen_tokens, h_vit), PLANNER_DTYPE, dev)?);
    let feat_refs: Vec<&Tensor> = feats.iter().collect();
    let visual_embeds = Tensor::cat(&feat_refs, 0)?.to_dtype(PLANNER_DTYPE)?;

    let to_i64 = |g: &[[i32; 3]]| -> Vec<[i64; 3]> {
        g.iter()
            .map(|&[a, b, c]| [a as i64, b as i64, c as i64])
            .collect()
    };
    let pos = mrope_position_ids(
        &tout.input_ids,
        &to_i64(&image_grids),
        &to_i64(&video_grids),
        &planner.mrope,
    )?;
    let mask = build_attention_mask_4d(&tout.token_type, &tout.token_segment_ids)?;

    let vin: Vec<bool> = tout.token_type.iter().map(|&t| t == 2).collect();
    let vout: Vec<bool> = tout.token_type.iter().map(|&t| t == 3).collect();

    let embeds = format_mllm_inputs_embeds(
        &planner.backbone,
        &tout.input_ids,
        Some(&visual_embeds),
        &vin,
        &vout,
    )?;
    let embeds = post_process_input_embeds(&embeds, &vout, &planner.mask_token)?;
    // Keep the additive mask in the backbone's activation dtype (its attention adds it to bf16 scores).
    let mask = mask.to_dtype(embeds.dtype())?;

    let gen_idx: Vec<u32> = (0..l).filter(|&i| vout[i]).map(|i| i as u32).collect();
    let n_query = gen_idx.len() as i32;
    Ok((
        StreamState {
            input_embeds: embeds,
            position_ids: pos,
            mask,
            gen_idx,
        },
        n_query,
    ))
}

/// Resolve the full-Bernini [`VitMode`] from the request's `video_mode` (a guidance-mode name preferred)
/// plus the conditioning + output kind. Defaults: video source ⇒ `v2v_apg`; video+image or image-refs→
/// video ⇒ `rv2v_wapg`; otherwise (t2i/i2i/t2v) ⇒ `vae_txt_vit_wapg`.
fn resolve_vit_mode(
    video_mode: Option<&str>,
    has_video: bool,
    has_image: bool,
    out_video: bool,
) -> VitMode {
    if let Some(s) = video_mode {
        if let Some(m) = VitMode::from_name(s) {
            return m;
        }
        if let Some(m) = task_to_vit_mode(s) {
            return m;
        }
    }
    match (has_video, has_image, out_video) {
        (true, _, _) => {
            if has_image {
                VitMode::Rv2vWapg
            } else {
                VitMode::V2vApg
            }
        }
        (false, true, true) => VitMode::Rv2vWapg, // r2v: image refs → video
        _ => VitMode::VaeTxtVitWapg,              // t2i / i2i / t2v
    }
}

/// Upstream task_type → full-pipeline guidance mode (fallback when `video_mode` is a task name).
fn task_to_vit_mode(task: &str) -> Option<VitMode> {
    Some(match task {
        "t2i" | "t2v" | "i2i" => VitMode::VaeTxtVitWapg,
        "v2v" | "mv2v" | "ads2v" => VitMode::V2vApg,
        "r2v" | "rv2v" => VitMode::Rv2vWapg,
        _ => return None,
    })
}

/// Stable identity + advertised capabilities for the full Bernini pipeline.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID,
        family: "bernini",
        backend: "candle",
        modality: Modality::Video,
        capabilities: Capabilities {
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: false,
            conditioning: vec![
                ConditioningKind::Reference,
                ConditioningKind::MultiReference,
                ConditioningKind::VideoClip,
            ],
            supports_lora: false,
            supports_lokr: false,
            samplers: vec!["uni_pc", "unipc"],
            schedulers: Vec::new(),
            supported_guidance_methods: vec![],
            min_size: 16,
            max_size: 1280,
            max_count: 1,
            mac_only: false,
            supported_quants: &[Quant::Q4, Quant::Q8],
            supports_kv_cache: false,
            requires_sigma_shift: false,
        },
    }
}

/// The loaded full Bernini pipeline: the snapshot dir + the resolved renderer knobs.
pub struct Bernini {
    descriptor: ModelDescriptor,
    knobs: BerniniKnobs,
    root: PathBuf,
    device: Device,
}

/// Load the full Bernini pipeline from a combined snapshot dir (the [`crate::convert`] renderer tier +
/// the planner components). Lazy: the heavy weights load on the first `generate`, so registration +
/// descriptor + `validate` resolve for a missing dir.
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(
                "bernini: expected a model directory (converted full-Bernini snapshot: transformer/ \
                 transformer_2/ text_encoder/ vae/ tokenizer/ mllm/ connector/ vit_decoder/), not a \
                 single .safetensors file"
                    .into(),
            ))
        }
    };
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(gen_core::Error::Unsupported(
            "bernini does not support control / VACE / IP-adapter overlays".into(),
        ));
    }
    let knobs = BerniniKnobs::from_dir(&root);
    let device = candle_gen::default_device()?;
    Ok(Box::new(Bernini {
        descriptor: descriptor(),
        knobs,
        root,
        device,
    }))
}

// Link-time self-registration into candle-gen's model registry (epic 3720).
candle_gen::register_generators! { descriptor => load }

/// Force-link hook (keeps the `inventory::submit!` registration from being dead-stripped).
pub fn force_link() {}

impl Generator for Bernini {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        self.descriptor
            .capabilities
            .validate_request(self.descriptor.id, req)?;
        if req.prompt.is_empty() {
            return Err(gen_core::Error::Msg(
                "bernini: prompt must not be empty".into(),
            ));
        }
        if let Some(frames) = req.frames {
            if frames == 0 || frames % 4 != 1 {
                return Err(gen_core::Error::Msg(format!(
                    "bernini: num_frames must be 1 + 4·k (got {frames})"
                )));
            }
        }
        validate_conditioning_video_clips(req).map_err(|e| gen_core::Error::Msg(e.to_string()))?;
        Ok(())
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.validate(req)?;
        Ok(self.generate_impl(req, on_progress)?)
    }
}

impl Bernini {
    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> CResult<GenerationOutput> {
        let dev = &self.device;
        let task = req.video_mode.as_deref().unwrap_or("");

        // --- Geometry + knobs ---
        let frames = req.frames.unwrap_or(DEFAULT_FRAMES_14B).max(1) as usize;
        let out_video = frames > 1;
        let width = req.width;
        let height = req.height;
        let steps = req
            .steps
            .map(|s| s as usize)
            .unwrap_or(FullDefaults::STEPS)
            .max(1);
        let seed = req.seed.unwrap_or_else(gen_core::default_seed);
        let neg = req.negative_prompt.clone().unwrap_or_default();

        let has_video = req
            .conditioning
            .iter()
            .any(|c| matches!(c, Conditioning::VideoClip { .. }));
        let has_image = req.conditioning.iter().any(|c| {
            matches!(
                c,
                Conditioning::Reference { .. } | Conditioning::MultiReference { .. }
            )
        });
        let mode = resolve_vit_mode(req.video_mode.as_deref(), has_video, has_image, out_video);

        on_progress(Progress::Step {
            current: 0,
            total: steps as u32,
        });

        // The z16 VAE (with encoder) is resident throughout: it VAE-encodes the source media into z16
        // conditioning latents up front and decodes the final latent at the end.
        let vae = {
            let vb = candle_gen::component_vb(&self.root, "vae", VAE_DTYPE, dev, MODEL_ID)?;
            WanVae16::new_with_encoder(&Vae16Config::wan21(), vb)?
        };

        // --- Stage 1: planner (loaded → 3 streams + MAR loop → freed) ---
        // Source latents (VAE) are computed here and carried past the planner drop; the ViT features are
        // consumed inside the planner block.
        let (
            max_seq,
            src_videos,
            src_images,
            s_wtxt_wvit,
            s_wtxt_wovit,
            s_wotxt_wvit,
            s_wotxt_wovit,
        ) = {
            let mut planner = BerniniPlanner::load(&self.root, dev)?;

            // Preprocess the conditioning: videos first, then images (conversation / source_id order).
            let mut videos: Vec<SourceVisual> = Vec::new();
            let mut images: Vec<SourceVisual> = Vec::new();
            for c in &req.conditioning {
                match c {
                    Conditioning::VideoClip { frames: clip, .. } => {
                        let rgb: Vec<RgbImage> = clip.iter().map(to_rgb).collect::<CResult<_>>()?;
                        let vit_frames = sample_vit_frames(&rgb);
                        let (vit_feat, vit_grid) = vit_encode_video(&planner, &vit_frames, dev)?;
                        let vae_latent = encode_videoclip(&vae, clip, width, height, dev)?;
                        let hw = (rgb[0].height() as i64, rgb[0].width() as i64);
                        videos.push(SourceVisual {
                            vit_feat,
                            vit_grid,
                            vae_latent,
                            hw,
                        });
                    }
                    Conditioning::Reference { image, .. } => {
                        let rgb = to_rgb(image)?;
                        let (vit_feat, vit_grid) = vit_encode_image(&planner, &rgb, dev)?;
                        let vae_latent = encode_image(&vae, image, width, height, dev)?;
                        images.push(SourceVisual {
                            vit_feat,
                            vit_grid,
                            vae_latent,
                            hw: (image.height as i64, image.width as i64),
                        });
                    }
                    Conditioning::MultiReference { images: imgs } => {
                        for image in imgs {
                            let rgb = to_rgb(image)?;
                            let (vit_feat, vit_grid) = vit_encode_image(&planner, &rgb, dev)?;
                            let vae_latent = encode_image(&vae, image, width, height, dev)?;
                            images.push(SourceVisual {
                                vit_feat,
                                vit_grid,
                                vae_latent,
                                hw: (image.height as i64, image.width as i64),
                            });
                        }
                    }
                    _ => {}
                }
            }

            let gen_grid = gen_target_grid(
                height as i64,
                width as i64,
                frames,
                req.fps.unwrap_or(FullDefaults::FPS),
            );

            // Three streams: cond (full), imgcond (text, no visuals), uncond (neg text, no visuals).
            let (cond, n_query) = build_stream(
                &planner,
                task,
                &req.prompt,
                &images,
                &videos,
                gen_grid,
                out_video,
                height as i64,
                width as i64,
                dev,
            )?;
            let (imgcond, _) = build_stream(
                &planner,
                task,
                &req.prompt,
                &[],
                &[],
                gen_grid,
                out_video,
                height as i64,
                width as i64,
                dev,
            )?;
            let (uncond, _) = build_stream(
                &planner,
                task,
                &neg,
                &[],
                &[],
                gen_grid,
                out_video,
                height as i64,
                width as i64,
                dev,
            )?;

            if n_query > planner.knobs.num_mask_token {
                return Err(CandleError::Msg(format!(
                    "bernini: gen-target needs {n_query} ViT tokens but the planner has only {} mask \
                     tokens — lower the resolution/frames",
                    planner.knobs.num_mask_token
                )));
            }

            let vit_cfg = VitCfg {
                planning_step: FullDefaults::PLANNING_STEP,
                vit_denoising_step: FullDefaults::VIT_DENOISING_STEP,
                vit_txt_cfg: FullDefaults::VIT_TXT_CFG,
                vit_img_cfg: FullDefaults::VIT_IMG_CFG,
            };
            let order = seeded_permutation(n_query, seed);
            let step_noise = seeded_step_noise(
                n_query,
                vit_cfg.planning_step,
                &order,
                planner.knobs.clip_diff_in_channels,
                seed,
                dev,
            )?;

            // MAR planning loop (disjoint field borrows: backbone/connector shared, clip_diff mutable).
            let streams = sample_vit_embed(
                &planner.backbone,
                &planner.connector,
                &mut planner.clip_diff,
                &cond,
                &uncond,
                &imgcond,
                &vit_cfg,
                &order,
                &step_noise,
                &req.cancel,
                &planner.mask_token,
            )?;

            // Cast the planner streams to f32 for the UMT5 concat (UMT5 runs f32). The renderer's
            // `embed_text` consumes the f32 context (as the renderer path does).
            let f32c = |a: &Tensor| a.to_dtype(ENC_DTYPE);
            let out = (
                planner.knobs.max_sequence_length.max(1),
                videos
                    .iter()
                    .map(|s| s.vae_latent.clone())
                    .collect::<Vec<_>>(),
                images
                    .iter()
                    .map(|s| s.vae_latent.clone())
                    .collect::<Vec<_>>(),
                f32c(&streams.wtxt_wvit)?,
                f32c(&streams.wtxt_wovit)?,
                f32c(&streams.wotxt_wvit)?,
                f32c(&streams.wotxt_wovit)?,
            );
            out
            // planner dropped here (freed before the renderer experts load).
        };

        // --- Stage 2: UMT5 text encode + concat_with_zero_init for the 4 renderer streams ---
        let (t5_pos, t5_neg) = {
            let tok = build_tokenizer(&self.root)?;
            let vb =
                candle_gen::component_vb(&self.root, "text_encoder", ENC_DTYPE, dev, MODEL_ID)?;
            let enc = Umt5Encoder::new(&TextEncoderConfig::umt5_xxl(), vb)?;
            let pos = umt5_encode(&enc, &tok, &req.prompt, dev)?;
            let neg = umt5_encode(&enc, &tok, &neg, dev)?;
            (pos, neg)
        };
        let pe_wtxt_wvit = concat_with_zero_init(&t5_pos, &s_wtxt_wvit, max_seq)?;
        let pe_wtxt_wovit = concat_with_zero_init(&t5_pos, &s_wtxt_wovit, max_seq)?;
        let pe_wotxt_wvit = concat_with_zero_init(&t5_neg, &s_wotxt_wvit, max_seq)?;
        let pe_wotxt_wovit = concat_with_zero_init(&t5_neg, &s_wotxt_wovit, max_seq)?;

        // --- Stage 3: load both experts, ViT-conditioned APG denoise ---
        let t_lat = ((frames as u32 - 1) / VAE16_STRIDE_TEMPORAL + 1) as usize;
        let h_lat = (height / VAE16_STRIDE_SPATIAL) as usize;
        let w_lat = (width / VAE16_STRIDE_SPATIAL) as usize;
        let init_noise = create_noise(seed, Z_DIM, t_lat, h_lat, w_lat, dev)?;

        // Source ids (videos first, then images — mirrors the packing order).
        let (nv, ni) = (src_videos.len(), src_images.len());
        let sids = assign_source_ids(
            nv + ni,
            self.knobs.max_trained_src_id,
            self.knobs.interpolate_src_id,
        );
        let video_srcs: Vec<(Tensor, f64)> = src_videos
            .iter()
            .enumerate()
            .map(|(k, v)| (v.clone(), sids[k]))
            .collect();
        let image_srcs: Vec<(Tensor, f64)> = src_images
            .iter()
            .enumerate()
            .map(|(j, im)| (im.clone(), sids[nv + j]))
            .collect();

        let base_g = VitGuidanceParams {
            omega_txt: req.guidance.unwrap_or(FullDefaults::OMEGA_TXT),
            omega_img: FullDefaults::OMEGA_IMG,
            omega_vid: FullDefaults::OMEGA_VID,
            omega_tgt: FullDefaults::OMEGA_TGT,
            eta: FullDefaults::ETA,
            norm_threshold: FullDefaults::NORM_THRESHOLD,
        };

        let dit_cfg = TransformerConfig::t2v_14b();
        let load_expert = |sub: &str| -> CResult<WanTransformer> {
            let vb = candle_gen::component_vb(&self.root, sub, DIT_DTYPE, dev, MODEL_ID)?;
            Ok(WanTransformer::new(&dit_cfg, vb)?)
        };
        let latents = {
            let high_dit = load_expert("transformer")?;
            let low_dit = load_expert("transformer_2")?;
            let streams4 = [
                &pe_wtxt_wvit,
                &pe_wtxt_wovit,
                &pe_wotxt_wvit,
                &pe_wotxt_wovit,
            ];
            let high = BVitExpert::build(&high_dit, streams4)?;
            let low = BVitExpert::build(&low_dit, streams4)?;
            let pf = PackedForward::new(
                dit_cfg,
                self.knobs.max_trained_src_id,
                self.knobs.interpolate_src_id,
            );
            let boundary_ts = self.knobs.switch_dit_boundary as f64 * NUM_TRAIN_TIMESTEPS as f64;
            let shift = req
                .scheduler_shift
                .map(|s| s as f64)
                .unwrap_or(FullDefaults::FLOW_SHIFT);
            let sampler = Sampler::parse(req.sampler.as_deref());
            let total = steps as u32;
            let mut on_step = |i: usize| {
                on_progress(Progress::Step {
                    current: i as u32 + 1,
                    total,
                })
            };
            denoise_bernini_wvitcfg(
                &pf,
                mode,
                &low,
                &high,
                boundary_ts,
                sampler,
                steps,
                shift,
                &init_noise,
                &image_srcs,
                &video_srcs,
                &base_g,
                FullDefaults::OMEGA_SCALE,
                &req.cancel,
                &mut on_step,
            )?
        };

        // --- Stage 4: z16 VAE decode → image / video ---
        on_progress(Progress::Decoding);
        let decoded = vae.decode(&latents)?;
        let images_out = frames_to_images(&decoded)?;

        if frames == 1 {
            let first = images_out
                .into_iter()
                .next()
                .ok_or_else(|| CandleError::Msg("bernini: VAE decode produced no frames".into()))?;
            Ok(GenerationOutput::Images(vec![first]))
        } else {
            let fps = req.fps.unwrap_or(FullDefaults::FPS);
            Ok(GenerationOutput::Video {
                frames: images_out,
                fps,
                audio: None,
            })
        }
    }
}

/// One expert (high or low) with its 4 prompt-embed streams already `embed_text`-projected into the
/// expert's context space (the full-Bernini ViT-conditioned path). Each stream is
/// `concat_with_zero_init(UMT5(prompt), planner ViT-context)` in renderer `text_dim` space, so it goes
/// through the same `embed_text` as the renderer's text context.
pub struct BVitExpert<'a> {
    transformer: &'a WanTransformer,
    wtxt_wvit: Tensor,
    wtxt_wovit: Tensor,
    wotxt_wvit: Tensor,
    wotxt_wovit: Tensor,
}

impl<'a> BVitExpert<'a> {
    /// `streams` = `[wtxt_wvit, wtxt_wovit, wotxt_wvit, wotxt_wovit]` prompt-embed contexts (each
    /// `[1, S, text_dim]`).
    pub fn build(dit: &'a WanTransformer, streams: [&Tensor; 4]) -> CResult<Self> {
        Ok(Self {
            transformer: dit,
            wtxt_wvit: dit.embed_text(streams[0])?,
            wtxt_wovit: dit.embed_text(streams[1])?,
            wotxt_wvit: dit.embed_text(streams[2])?,
            wotxt_wovit: dit.embed_text(streams[3])?,
        })
    }

    fn streams(&self) -> VitStreams<'_> {
        VitStreams {
            wtxt_wvit: &self.wtxt_wvit,
            wtxt_wovit: &self.wtxt_wovit,
            wotxt_wvit: &self.wotxt_wvit,
            wotxt_wovit: &self.wotxt_wovit,
        }
    }
}

/// The full-Bernini ViT-conditioned denoise loop (`sample_bernini_wvitcfg`) — the renderer-side compute
/// that consumes the planner's 4 prompt streams. The boundary-switched, [`vit_one_step`]-guided analog of
/// the renderer's plain denoise: each step picks the expert by `switch_dit_boundary`, multiplies **all
/// four** omegas (incl. `omega_tgt`) by `omega_scale` once on the first low-noise step, and applies the
/// flow step. Runs in spatial latent space `[1, 16, T, H8, W8]`.
#[allow(clippy::too_many_arguments)]
pub fn denoise_bernini_wvitcfg(
    pf: &PackedForward,
    mode: VitMode,
    low: &BVitExpert,
    high: &BVitExpert,
    boundary_ts: f64,
    sampler: Sampler,
    steps: usize,
    shift: f64,
    init_noise: &Tensor,
    images: &[(Tensor, f64)],
    videos: &[(Tensor, f64)],
    base_g: &VitGuidanceParams,
    omega_scale: f32,
    cancel: &CancelFlag,
    on_step: &mut dyn FnMut(usize),
) -> CResult<Tensor> {
    let mut sched = FlowScheduler::new(sampler, steps, shift);
    let sigmas = flow_sigmas(steps, shift);
    let mut latent = init_noise.clone();
    let mut switched = false;
    let mut g = base_g.clone();

    #[allow(clippy::needless_range_loop)]
    for i in 0..steps {
        if cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }
        on_step(i);
        let t = sched.timestep(i);
        let expert = if t >= boundary_ts {
            high
        } else {
            if !switched {
                switched = true;
                g.omega_txt *= omega_scale;
                g.omega_img *= omega_scale;
                g.omega_vid *= omega_scale;
                g.omega_tgt *= omega_scale;
            }
            low
        };
        let v = vit_one_step(
            pf,
            expert.transformer,
            mode,
            &latent,
            images,
            videos,
            t,
            sigmas[i],
            &expert.streams(),
            &g,
        )?;
        latent = sched.step(&v, &latent)?;
    }
    Ok(latent)
}

/// Build the UMT5 tokenizer from `root/tokenizer/tokenizer.json` (byte-identical config to the renderer).
fn build_tokenizer(root: &Path) -> CResult<TextTokenizer> {
    let te_cfg = TextEncoderConfig::umt5_xxl();
    TextTokenizer::from_file(
        root.join("tokenizer/tokenizer.json"),
        TokenizerConfig {
            max_length: te_cfg.max_length,
            pad_token_id: te_cfg.pad_token_id,
            chat_template: ChatTemplate::None,
            pad_to_max_length: false,
        },
    )
    .map_err(|e| CandleError::Msg(format!("bernini: load tokenizer: {e}")))
}

/// UMT5-encode `prompt` → **unpadded** `[1, S, 4096]` f32 (the `concat_with_zero_init` prepend expects
/// the raw prompt embeds, then pads/truncates the T5+planner concat to `max_sequence_length`). The
/// empty-prompt guard emits one pad token so a 0-length sequence never reaches the embedding gather.
fn umt5_encode(
    enc: &Umt5Encoder,
    tok: &TextTokenizer,
    prompt: &str,
    dev: &Device,
) -> CResult<Tensor> {
    let te_cfg = TextEncoderConfig::umt5_xxl();
    let out = tok
        .tokenize(prompt)
        .map_err(|e| CandleError::Msg(format!("bernini: tokenize: {e}")))?;
    let mut ids: Vec<u32> = out.ids.iter().map(|&i| i as u32).collect();
    if ids.is_empty() {
        ids.push(te_cfg.pad_token_id as u32);
    }
    let len = ids.len();
    let input_ids = Tensor::from_vec(ids, (1, len), dev)?;
    Ok(enc.encode(&input_ids)?.to_dtype(ENC_DTYPE)?)
}

/// Reject empty / non-`1+4k` conditioning video clips before the pipeline dereferences `frames[0]`
/// (mirrors the renderer's `encode_videoclip` guard). Free fn so it is unit-testable without weights.
fn validate_conditioning_video_clips(req: &GenerationRequest) -> CResult<()> {
    for c in &req.conditioning {
        if let Conditioning::VideoClip { frames, .. } = c {
            if frames.is_empty() {
                return Err(CandleError::Msg(
                    "bernini: empty conditioning video clip".into(),
                ));
            }
            if frames.len() % 4 != 1 {
                return Err(CandleError::Msg(format!(
                    "bernini: conditioning video-clip frame count must be 1 + 4·k (got {})",
                    frames.len()
                )));
            }
        }
    }
    Ok(())
}

/// Sub-sample a decoded clip to the ViT frame set (`smart_video_nframes`, assuming `target_fps`).
fn sample_vit_frames(frames: &[RgbImage]) -> Vec<RgbImage> {
    let fps = FullDefaults::FPS as f64;
    let vit_fps = (FullDefaults::FPS / 8).max(1) as f64;
    let idx = smart_video_nframes(
        frames.len() as i64,
        fps,
        vit_fps,
        Some(TEMPORAL_PATCH_SIZE),
        None,
        Some(frames.len() as i64),
        false,
    );
    idx.iter()
        .map(|&i| frames[(i as usize).min(frames.len() - 1)].clone())
        .collect()
}

// --- Deterministic host RNG (splitmix64 → Box-Muller) for the MAR reveal order + per-step FM noise. ---
// The MAR trajectory's torch bit-parity needs the reference CPU draw injected (a follow-up); the
// coherence bar uses this deterministic host RNG (seed-stable + injectable in tests).

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn uniform(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

fn gaussian(state: &mut u64) -> f32 {
    let u1 = uniform(state).max(1e-12);
    let u2 = uniform(state);
    ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
}

/// A deterministic reveal permutation of `[0, n)` from the seed (argsort of seeded normal noise — a
/// stable, injectable order).
fn seeded_permutation(n: i32, seed: u64) -> Vec<i32> {
    let mut state = seed ^ 0x4d_a4;
    let vals: Vec<f32> = (0..n).map(|_| gaussian(&mut state)).collect();
    let mut idx: Vec<i32> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        vals[a as usize]
            .partial_cmp(&vals[b as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx
}

/// Per-step base FM noise for the MAR loop — one `[revealed, in]` tensor per planning step (the
/// reference's `torch.randn(n_revealed, in)`, tiled ×3 inside `DiffLossFm::sample`).
fn seeded_step_noise(
    n_query: i32,
    planning_step: usize,
    order: &[i32],
    in_channels: usize,
    seed: u64,
    dev: &Device,
) -> CResult<Vec<Tensor>> {
    let schedule = mar_schedule(n_query, planning_step, order);
    let mut out = Vec::with_capacity(planning_step);
    for (s, revealed) in schedule.iter().enumerate() {
        let np = revealed.len().max(1);
        let mut state = seed ^ 0x9e37 ^ ((s as u64).wrapping_mul(0x100_0001));
        let data: Vec<f32> = (0..np * in_channels)
            .map(|_| gaussian(&mut state))
            .collect();
        out.push(Tensor::from_vec(data, (np, in_channels), dev)?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::DType;
    use candle_gen::candle_nn::VarBuilder;
    use candle_gen::gen_core::registry;
    use std::collections::HashMap;

    /// The full `bernini` engine registers + resolves via the gen_core registry (lazy load succeeds for
    /// a missing dir; the descriptor identity is what we pin).
    #[test]
    fn registers_and_resolves() {
        force_link();
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry::load(MODEL_ID, &spec).expect("bernini is registered");
        assert_eq!(g.descriptor().id, "bernini");
        assert_eq!(g.descriptor().family, "bernini");
        assert_eq!(g.descriptor().backend, "candle");
        assert_eq!(g.descriptor().modality, Modality::Video);
        assert!(!g.descriptor().capabilities.mac_only);
    }

    #[test]
    fn descriptor_surface() {
        let d = descriptor();
        assert_eq!(d.id, "bernini");
        assert!(d.capabilities.supports_guidance);
        assert!(d.capabilities.supports_negative_prompt);
        assert!(d.capabilities.accepts(ConditioningKind::Reference));
        assert!(d.capabilities.accepts(ConditioningKind::MultiReference));
        assert!(d.capabilities.accepts(ConditioningKind::VideoClip));
        assert!(d.capabilities.supported_quants.contains(&Quant::Q4));
        assert!(d.capabilities.supported_quants.contains(&Quant::Q8));
    }

    #[test]
    fn load_rejects_single_file_and_overlays() {
        let f = LoadSpec::new(WeightsSource::File("/tmp/w.safetensors".into()));
        assert!(load(&f).is_err());
        // A directory source is lazy-loaded, so it resolves past the marker.
        let d = LoadSpec::new(WeightsSource::Dir("/snap".into()));
        assert!(load(&d).is_ok());
    }

    /// Guidance-mode resolution: an explicit guidance-mode name wins, then a task-type name, then the
    /// conditioning/output defaults.
    #[test]
    fn vit_mode_resolution() {
        assert_eq!(
            resolve_vit_mode(Some("vae_txt_vit_wapg"), false, false, false),
            VitMode::VaeTxtVitWapg
        );
        assert_eq!(
            resolve_vit_mode(Some("rv2v_wapg"), true, true, true),
            VitMode::Rv2vWapg
        );
        // task-name fallback
        assert_eq!(
            resolve_vit_mode(Some("t2i"), false, false, false),
            VitMode::VaeTxtVitWapg
        );
        assert_eq!(
            resolve_vit_mode(Some("v2v"), true, false, true),
            VitMode::V2vApg
        );
        assert_eq!(
            resolve_vit_mode(Some("r2v"), false, true, true),
            VitMode::Rv2vWapg
        );
        // conditioning/output-driven defaults
        assert_eq!(
            resolve_vit_mode(None, false, false, false),
            VitMode::VaeTxtVitWapg
        ); // t2i
        assert_eq!(resolve_vit_mode(None, true, false, true), VitMode::V2vApg); // v2v
        assert_eq!(resolve_vit_mode(None, true, true, true), VitMode::Rv2vWapg); // rv2v
        assert_eq!(resolve_vit_mode(None, false, true, true), VitMode::Rv2vWapg); // r2v
        assert_eq!(
            resolve_vit_mode(None, false, true, false),
            VitMode::VaeTxtVitWapg
        ); // i2i
    }

    #[test]
    fn grid_token_count() {
        assert_eq!(grid_tokens([1, 12, 20]), 60);
        assert_eq!(grid_tokens([5, 12, 20]), 300);
    }

    /// Image targets are single-frame (`t = 1`); video targets sample `vit_fps` frames so `t > 1`; the
    /// spatial grid is the `smart_resize` of the output H/W.
    #[test]
    fn gen_target_grid_image_vs_video() {
        let img = gen_target_grid(480, 832, 1, 16);
        assert_eq!(img[0], 1, "image target is single-frame");
        assert_eq!([img[1], img[2]], [12, 20], "480x832 → 12x20 grid");
        let vid = gen_target_grid(480, 832, 81, 16);
        assert!(vid[0] > 1, "video target spans multiple temporal patches");
        assert_eq!([vid[1], vid[2]], [12, 20], "same spatial grid");
    }

    #[test]
    fn seeded_permutation_is_a_permutation() {
        let n = 60;
        let a = seeded_permutation(n, 42);
        let b = seeded_permutation(n, 42);
        assert_eq!(a, b, "deterministic for a fixed seed");
        let c = seeded_permutation(n, 7);
        assert_ne!(a, c, "different seeds → different order");
        let mut sorted = a.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..n).collect::<Vec<_>>(), "covers [0, n) once");
    }

    #[test]
    fn step_noise_shapes_match_schedule() {
        let n = 60;
        let steps = 25;
        let order = seeded_permutation(n, 42);
        let noise = seeded_step_noise(n, steps, &order, 3584, 42, &Device::Cpu).unwrap();
        assert_eq!(noise.len(), steps);
        let schedule = mar_schedule(n, steps, &order);
        for (s, arr) in noise.iter().enumerate() {
            let np = schedule[s].len().max(1);
            assert_eq!(arr.dims(), &[np, 3584], "step {s} noise shape");
        }
    }

    #[test]
    fn rejects_empty_and_miscounted_conditioning_video_clips() {
        let clip = |n: usize| Conditioning::VideoClip {
            frames: vec![
                Image {
                    width: 2,
                    height: 2,
                    pixels: vec![0u8; 2 * 2 * 3],
                };
                n
            ],
            frame_idx: 0,
            strength: 1.0,
        };
        let req = |conds: Vec<Conditioning>| GenerationRequest {
            conditioning: conds,
            ..Default::default()
        };
        assert!(validate_conditioning_video_clips(&req(vec![clip(0)])).is_err());
        assert!(validate_conditioning_video_clips(&req(vec![clip(3)])).is_err());
        assert!(validate_conditioning_video_clips(&req(vec![clip(1)])).is_ok());
        assert!(validate_conditioning_video_clips(&req(vec![clip(5)])).is_ok());
        assert!(validate_conditioning_video_clips(&req(vec![])).is_ok());
        assert!(validate_conditioning_video_clips(&req(vec![clip(5), clip(0)])).is_err());
    }

    // --- Synthetic tiny DiT for the ViT-conditioned denoise loop (no real weights). ---

    fn tiny_cfg() -> TransformerConfig {
        TransformerConfig {
            in_channels: 16,
            out_channels: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            dim: 16,
            ffn_dim: 32,
            freq_dim: 16,
            text_dim: 16,
            patch: (1, 2, 2),
            eps: 1e-6,
            rope_theta: 10000.0,
            rope_max_seq_len: 64,
        }
    }

    fn tiny_dit(cfg: &TransformerConfig, dev: &Device) -> WanTransformer {
        let mut m: HashMap<String, Tensor> = HashMap::new();
        let mut put = |k: &str, shape: &[usize]| {
            m.insert(
                k.to_string(),
                Tensor::randn(0f32, 0.2f32, shape, dev).unwrap(),
            );
        };
        let (pt, ph, pw) = cfg.patch;
        let d = cfg.dim;
        put("patch_embedding.weight", &[d, cfg.in_channels, pt, ph, pw]);
        put("patch_embedding.bias", &[d]);
        put(
            "condition_embedder.text_embedder.linear_1.weight",
            &[d, cfg.text_dim],
        );
        put("condition_embedder.text_embedder.linear_1.bias", &[d]);
        put("condition_embedder.text_embedder.linear_2.weight", &[d, d]);
        put("condition_embedder.text_embedder.linear_2.bias", &[d]);
        put(
            "condition_embedder.time_embedder.linear_1.weight",
            &[d, cfg.freq_dim],
        );
        put("condition_embedder.time_embedder.linear_1.bias", &[d]);
        put("condition_embedder.time_embedder.linear_2.weight", &[d, d]);
        put("condition_embedder.time_embedder.linear_2.bias", &[d]);
        put("condition_embedder.time_proj.weight", &[6 * d, d]);
        put("condition_embedder.time_proj.bias", &[6 * d]);
        for i in 0..cfg.num_layers {
            let b = format!("blocks.{i}");
            put(&format!("{b}.scale_shift_table"), &[1, 6, d]);
            for attn in ["attn1", "attn2"] {
                for leaf in ["to_q", "to_k", "to_v", "to_out.0"] {
                    put(&format!("{b}.{attn}.{leaf}.weight"), &[d, d]);
                    put(&format!("{b}.{attn}.{leaf}.bias"), &[d]);
                }
                put(&format!("{b}.{attn}.norm_q.weight"), &[d]);
                put(&format!("{b}.{attn}.norm_k.weight"), &[d]);
            }
            put(&format!("{b}.norm2.weight"), &[d]);
            put(&format!("{b}.norm2.bias"), &[d]);
            put(&format!("{b}.ffn.net.0.proj.weight"), &[cfg.ffn_dim, d]);
            put(&format!("{b}.ffn.net.0.proj.bias"), &[cfg.ffn_dim]);
            put(&format!("{b}.ffn.net.2.weight"), &[d, cfg.ffn_dim]);
            put(&format!("{b}.ffn.net.2.bias"), &[d]);
        }
        put("proj_out.weight", &[cfg.out_channels * pt * ph * pw, d]);
        put("proj_out.bias", &[cfg.out_channels * pt * ph * pw]);
        put("scale_shift_table", &[1, 2, d]);
        let vb = VarBuilder::from_tensors(m, DType::F32, dev);
        WanTransformer::new(cfg, vb).unwrap()
    }

    /// The full-Bernini ViT-conditioned denoise loop runs end-to-end over a tiny dual-expert (crossing
    /// the boundary so the `omega_scale` switch fires) from the planner's 4 synthetic streams and
    /// preserves the spatial latent shape. Pins the capstone loop plumbing (scheduler / expert switch /
    /// per-step `vit_one_step` / flow step) without real weights.
    #[test]
    fn denoise_wvitcfg_runs_and_keeps_shape() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let dit = tiny_dit(&cfg, &dev);
        let pf = PackedForward::new(cfg, 5.0, true);
        let noisy = Tensor::randn(0f32, 1f32, (1, 16, 3, 4, 4), &dev).unwrap();
        // 4 distinct prompt streams `[1, S, text_dim]`.
        let mk = |s: f32| {
            Tensor::randn(0f32, 1f32, (1, 5, 16), &dev)
                .unwrap()
                .affine(s as f64, 0.0)
                .unwrap()
        };
        let (s0, s1, s2, s3) = (mk(1.0), mk(0.7), mk(0.4), mk(0.2));
        let streams = [&s0, &s1, &s2, &s3];
        let low = BVitExpert::build(&dit, streams).unwrap();
        let high = BVitExpert::build(&dit, streams).unwrap();
        let g = VitGuidanceParams {
            omega_txt: 4.0,
            omega_img: 4.5,
            omega_vid: 1.25,
            omega_tgt: 0.5,
            eta: 1.0,
            norm_threshold: 50.0,
        };
        let mut on_step = |_i: usize| {};
        let out = denoise_bernini_wvitcfg(
            &pf,
            VitMode::VaeTxtVitWapg,
            &low,
            &high,
            0.875 * NUM_TRAIN_TIMESTEPS as f64, // boundary 875 → crossed within 4 steps
            Sampler::UniPC,
            4,
            5.0,
            &noisy,
            &[],
            &[],
            &g,
            0.8,
            &CancelFlag::default(),
            &mut on_step,
        )
        .expect("denoise");
        assert_eq!(
            out.dims(),
            noisy.dims(),
            "loop preserves spatial latent shape"
        );
    }
}
