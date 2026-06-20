//! The live SCAIL-2 generation pipeline — the runnable end-to-end denoise loop.
//!
//! Ports `wan/scail.py::SCAIL2Pipeline.generate`: preprocess the reference character (+ optional
//! extra characters) and the driving video into conditioning latents, then run a plain-CFG
//! flow-matching denoise (UniPC) over one or more 81-frame **segments** with clean-history continuity,
//! and VAE-decode each segment back to pixels.
//!
//! Reuse map — the heavy components are `candle-gen-wan`'s (SCAIL-2 *is* Wan2.1-14B I2V): the z16
//! [`WanVae16`] (encode/decode; its decode already streams one latent frame at a time = the
//! temporal-tiled decode the high-res fix needs), the [`Umt5Encoder`] text encoder, and the
//! flow-matching [`FlowScheduler`] (UniPC). SCAIL-2's own pieces are the [`Scail2Dit`] forward, the
//! open-CLIP [`ScailClip`] image encode, the 28-channel [`extract_and_compress_mask_to_latent`] mask
//! build, and the [`interpolate`]/[`downsample_half`] resizes.

use std::path::Path;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use candle_gen::gen_core::{GenerationOutput, Image, Progress};
use candle_gen::{CandleError, Result as CResult};
use candle_gen_wan::config::TextEncoderConfig;
use candle_gen_wan::pipeline::frames_to_images;
use candle_gen_wan::scheduler::{FlowScheduler, Sampler};
use candle_gen_wan::text_encoder::Umt5Encoder;
use candle_gen_wan::vae16::WanVae16;

use crate::clip::ScailClip;
use crate::model::{Scail2Dit, Scail2Inputs};
use crate::preprocess::{extract_and_compress_mask_to_latent, TEMPORAL_STRIDE};
use crate::resize::{clip_preprocess, downsample_half, interpolate, Interp};

/// Inputs must be divisible by 32: the pose path halves spatially (→ ÷16) before the ÷8 VAE stride, and
/// the 28-channel mask pools 8×, so both the full and half grids stay integer + even.
const DIM_ALIGN: u32 = 32;

/// The loaded SCAIL-2 components (resident in the [`crate::pipeline::Scail2`] generator's cache). All
/// run f32 (the DiT's high-token-length NaN avoidance, and z16 VAE / UMT5 / CLIP are f32 anyway).
pub struct Components {
    pub te: Umt5Encoder,
    pub dit: Scail2Dit,
    pub vae: WanVae16,
    pub clip: ScailClip,
}

/// One masked character reference (the primary subject or an extra character): an RGB image paired with
/// its color-coded segmentation mask.
pub struct CharacterRef<'a> {
    pub image: &'a Image,
    pub mask: &'a Image,
}

/// A fully-specified SCAIL-2 generation job (the engine-internal form the worker maps a
/// `GenerationRequest` onto). All images are decoded + resized to `(width, height)` here.
pub struct Scail2Job<'a> {
    pub prompt: &'a str,
    pub negative_prompt: &'a str,
    pub width: u32,
    pub height: u32,
    pub reference: CharacterRef<'a>,
    pub additional: Vec<CharacterRef<'a>>,
    pub driving_frames: &'a [Image],
    pub driving_masks: &'a [Image],
    /// `true` = cross-identity replacement, `false` = animation.
    pub replace_flag: bool,
    pub seed: u64,
    pub steps: usize,
    pub shift: f64,
    pub guidance: f64,
    pub sampler: Sampler,
    pub fps: u32,
    pub segment_len: usize,
    pub segment_overlap: usize,
}

/// Round a requested dim down to a multiple of [`DIM_ALIGN`] (min one tile).
fn align(value: u32) -> usize {
    (value / DIM_ALIGN).max(1) as usize * DIM_ALIGN as usize
}

/// Decode an `Image` (RGB24 `u8`) → `[3, th, tw]` f32 in `[-1, 1]`, resizing if its native size differs.
/// `mode` is `Bicubic` for photographic images, `Bilinear` for color masks (bounded — avoids the bicubic
/// overshoot that would invent out-of-gamut colors at mask edges).
fn image_to_chw(img: &Image, tw: usize, th: usize, mode: Interp, dev: &Device) -> CResult<Tensor> {
    let (iw, ih) = (img.width as usize, img.height as usize);
    if img.pixels.len() != iw * ih * 3 {
        return Err(CandleError::Msg(format!(
            "scail2: image pixel buffer {} != {iw}x{ih}x3",
            img.pixels.len()
        )));
    }
    let px: Vec<f32> = img.pixels.iter().map(|&p| p as f32 / 127.5 - 1.0).collect();
    let chw = Tensor::from_vec(px, (ih, iw, 3), dev)?.permute((2, 0, 1))?; // [3,H,W]
    let nchw = chw.reshape((1, 3, ih, iw))?;
    let out = if (ih, iw) != (th, tw) {
        interpolate(&nchw, th, tw, mode)?
    } else {
        nchw
    };
    Ok(out.reshape((3, th, tw))?)
}

/// Stack driving frames → `[T, 3, H, W]`.
fn stack_frames(frames: &[Image], tw: usize, th: usize, dev: &Device) -> CResult<Tensor> {
    let chw: Vec<Tensor> = frames
        .iter()
        .map(|f| -> CResult<Tensor> {
            Ok(image_to_chw(f, tw, th, Interp::Bicubic, dev)?.reshape((1, 3, th, tw))?)
        })
        .collect::<CResult<_>>()?;
    let refs: Vec<&Tensor> = chw.iter().collect();
    Ok(Tensor::cat(&refs, 0)?) // [T,3,H,W]
}

/// Stack per-frame masks → `[3, T, H, W]` (the `extract_and_compress` input layout).
fn stack_masks(masks: &[Image], tw: usize, th: usize, dev: &Device) -> CResult<Tensor> {
    let chw: Vec<Tensor> = masks
        .iter()
        .map(|m| -> CResult<Tensor> {
            Ok(image_to_chw(m, tw, th, Interp::Bilinear, dev)?.reshape((3, 1, th, tw))?)
        })
        .collect::<CResult<_>>()?;
    let refs: Vec<&Tensor> = chw.iter().collect();
    Ok(Tensor::cat(&refs, 1)?) // [3,T,H,W]
}

/// VAE-encode a `[3, T, H, W]` pixel clip (`[-1,1]`) → `[16, T_lat, H/8, W/8]` (drops the batch dim).
fn vae_encode_cthw(vae: &WanVae16, cthw: &Tensor) -> CResult<Tensor> {
    let (c, t, h, w) = cthw.dims4()?;
    let z = vae.encode(&cthw.reshape((1, c, t, h, w))?)?; // [1,16,T_lat,h,w]
    let (_, zc, zt, zh, zw) = z.dims5()?;
    Ok(z.reshape((zc, zt, zh, zw))?)
}

/// `uncond + guidance·(cond − uncond)`.
fn cfg_combine(uncond: &Tensor, cond: &Tensor, guidance: f64) -> CResult<Tensor> {
    Ok((uncond + (cond - uncond)?.affine(guidance, 0.0)?)?)
}

/// Overwrite the leading `min(history_t, T)` latent frames of `latent [16,T,h,w]` with the clean
/// history (upstream `apply_clean_history`). No-op without history.
fn apply_clean_history(latent: &Tensor, history: Option<&Tensor>) -> CResult<Tensor> {
    let Some(h) = history else {
        return Ok(latent.clone());
    };
    let lt = latent.dim(1)?;
    let ht = h.dim(1)?.min(lt);
    if ht == 0 {
        return Ok(latent.clone());
    }
    let head = h.narrow(1, 0, ht)?;
    if ht == lt {
        return Ok(head);
    }
    let tail = latent.narrow(1, ht, lt - ht)?;
    Ok(Tensor::cat(&[&head, &tail], 1)?)
}

/// Segment plan over `total` driving frames (upstream `build_segments`): a single VAE-aligned segment
/// when the clip fits, else overlapping `segment_len` windows striding by `len − overlap`.
fn build_segments(total: usize, len: usize, overlap: usize) -> Vec<(usize, usize)> {
    if total <= len {
        let keep = ((total - 1) / TEMPORAL_STRIDE) * TEMPORAL_STRIDE + 1;
        return vec![(0, keep)];
    }
    let mut segs = Vec::new();
    let stride = len - overlap;
    let mut start = 0;
    while start < total {
        let end = start + len;
        if end > total {
            break;
        }
        segs.push((start, end));
        start += stride;
    }
    segs
}

/// Tokenize + UMT5-encode `prompt` → `[L, 4096]` (f32, un-padded; the DiT's `embed_text` pads to
/// `text_len`).
fn encode_text(
    te: &Umt5Encoder,
    tok: &TextTokenizer,
    prompt: &str,
    dev: &Device,
) -> CResult<Tensor> {
    let out = tok
        .tokenize(prompt)
        .map_err(|e| CandleError::Msg(format!("scail2: tokenize: {e}")))?;
    let len = out.ids.len().max(1);
    let ids: Vec<u32> = out.ids.iter().map(|&i| i as u32).collect();
    let input_ids = Tensor::from_vec(ids, (1, len), dev)?;
    let embeds = te.encode(&input_ids)?; // [1, L, 4096]
    let (_, l, d) = embeds.dims3()?;
    Ok(embeds.reshape((l, d))?)
}

/// Run the full SCAIL-2 generation for `job` against the resident `comps`. `root` is the snapshot dir
/// (for `tokenizer/tokenizer.json`); `cancel` is polled each denoise step.
#[allow(clippy::too_many_arguments)]
pub fn generate(
    root: &Path,
    comps: &Components,
    te_cfg: &TextEncoderConfig,
    job: &Scail2Job,
    cancel: &dyn Fn() -> bool,
    on_progress: &mut dyn FnMut(Progress),
) -> CResult<GenerationOutput> {
    if job.driving_frames.is_empty() {
        return Err(CandleError::Msg(
            "scail2: a driving video is required".into(),
        ));
    }
    if job.driving_masks.len() != job.driving_frames.len() {
        return Err(CandleError::Msg(format!(
            "scail2: driving_masks ({}) must match driving_frames ({})",
            job.driving_masks.len(),
            job.driving_frames.len()
        )));
    }
    let dev = comps.dit.device();
    let (tw, th) = (align(job.width), align(job.height));
    let cfg_disabled = job.guidance <= 1.0;

    // --- decode + resize all pixel inputs to (tw, th) ---
    let ref_chw = image_to_chw(job.reference.image, tw, th, Interp::Bicubic, dev)?; // [3,H,W]
    let ref_mask_chw = image_to_chw(job.reference.mask, tw, th, Interp::Bilinear, dev)?; // [3,H,W]
    let driving = stack_frames(job.driving_frames, tw, th, dev)?; // [T,3,H,W]
    let driving_mask = stack_masks(job.driving_masks, tw, th, dev)?; // [3,T,H,W]

    // Reference char latent + its 28-ch mask (1 latent frame).
    let ref_latent = vae_encode_cthw(&comps.vae, &ref_chw.reshape((3, 1, th, tw))?)?;
    let ref_mask_28 = extract_and_compress_mask_to_latent(
        &ref_mask_chw.reshape((3, 1, th, tw))?,
        TEMPORAL_STRIDE,
    )?;
    let lat_h = ref_latent.dim(2)?;
    let lat_w = ref_latent.dim(3)?;

    // Extra characters (multi-reference): cat latents + masks on the frame axis.
    let (additional_ref_latent, additional_ref_masks) = if job.additional.is_empty() {
        (None, None)
    } else {
        let mut lats = Vec::new();
        let mut masks = Vec::new();
        for c in &job.additional {
            let img =
                image_to_chw(c.image, tw, th, Interp::Bicubic, dev)?.reshape((3, 1, th, tw))?;
            lats.push(vae_encode_cthw(&comps.vae, &img)?);
            let mk =
                image_to_chw(c.mask, tw, th, Interp::Bilinear, dev)?.reshape((3, 1, th, tw))?;
            masks.push(extract_and_compress_mask_to_latent(&mk, TEMPORAL_STRIDE)?);
        }
        let lr: Vec<&Tensor> = lats.iter().collect();
        let mr: Vec<&Tensor> = masks.iter().collect();
        (Some(Tensor::cat(&lr, 1)?), Some(Tensor::cat(&mr, 1)?))
    };

    // --- UMT5 text encode + CLIP reference-image features (once) ---
    let tok = TextTokenizer::from_file(
        root.join("tokenizer/tokenizer.json"),
        TokenizerConfig {
            max_length: te_cfg.max_length,
            pad_token_id: te_cfg.pad_token_id,
            chat_template: ChatTemplate::None,
            pad_to_max_length: false,
        },
    )
    .map_err(|e| CandleError::Msg(format!("scail2: load tokenizer: {e}")))?;
    let context = encode_text(&comps.te, &tok, job.prompt, dev)?;
    let context_null = if cfg_disabled {
        context.clone()
    } else {
        encode_text(&comps.te, &tok, job.negative_prompt, dev)?
    };
    let clip_fea = {
        let pixel = clip_preprocess(&ref_chw.reshape((1, 3, th, tw))?, 224)?;
        comps.clip.encode(&pixel)?
    };

    let segments = build_segments(
        job.driving_frames.len(),
        job.segment_len,
        job.segment_overlap,
    );
    let mut out_pieces: Vec<Tensor> = Vec::new();
    let mut prev_history_pixel: Option<Tensor> = None;

    for (seg_idx, &(seg_start, seg_end)) in segments.iter().enumerate() {
        // Pose latent (half spatial res) + driving mask for this segment.
        let pose_seg = driving.narrow(0, seg_start, seg_end - seg_start)?; // [T,3,H,W]
        let pose_half = downsample_half(&pose_seg)?; // [T,3,H/2,W/2]
        let pose_cthw = pose_half.permute((1, 0, 2, 3))?.contiguous()?; // [3,T,H/2,W/2]
        let pose_latent = vae_encode_cthw(&comps.vae, &pose_cthw)?; // [16,T_lat,h/2,w/2]
        let lat_t = pose_latent.dim(1)?;

        let dmask_seg = driving_mask.narrow(1, seg_start, seg_end - seg_start)?; // [3,T,H,W]
        let dmask_half = downsample_half(&dmask_seg)?; // [3,T,H/2,W/2]
        let driving_masks = extract_and_compress_mask_to_latent(&dmask_half, TEMPORAL_STRIDE)?;

        // ref_masks = ref_mask_28 ++ zero null-noisy-mask over the latent length.
        let null_noisy = Tensor::zeros((28, lat_t, lat_h, lat_w), DType::F32, dev)?;
        let ref_masks = Tensor::cat(&[&ref_mask_28, &null_noisy], 1)?; // [28,1+T,h,w]

        // Clean-history latent + i2v mask for segments after the first.
        let (history_latent, history_mask) = match &prev_history_pixel {
            Some(hp) if seg_idx > 0 => {
                let hl = vae_encode_cthw(&comps.vae, hp)?; // [16,h_t,h,w]
                let h_t = hl.dim(1)?.min(lat_t);
                let ones = Tensor::ones((4, h_t, lat_h, lat_w), DType::F32, dev)?;
                let hm = if h_t < lat_t {
                    let z = Tensor::zeros((4, lat_t - h_t, lat_h, lat_w), DType::F32, dev)?;
                    Tensor::cat(&[&ones, &z], 1)?
                } else {
                    ones
                };
                (Some(hl), Some(hm))
            }
            _ => (None, None),
        };

        // Seeded init noise (per-segment).
        let noise = candle_gen_wan::pipeline::create_noise(
            job.seed.wrapping_add(seg_idx as u64),
            16,
            lat_t,
            lat_h,
            lat_w,
            dev,
        )?
        .reshape((16, lat_t, lat_h, lat_w))?;

        // --- denoise (plain CFG, clean-history pinned) ---
        let mut sched = FlowScheduler::new(job.sampler, job.steps, job.shift);
        let total = sched.num_steps() as u32;
        let mut latent = apply_clean_history(&noise, history_latent.as_ref())?;
        for i in 0..job.steps {
            if cancel() {
                return Err(CandleError::Canceled);
            }
            let t = sched.timestep(i);
            let x = apply_clean_history(&latent, history_latent.as_ref())?;
            let mut inp = Scail2Inputs {
                x: &x,
                ref_latent: &ref_latent,
                ref_masks: &ref_masks,
                pose_latent: &pose_latent,
                driving_masks: &driving_masks,
                history_mask: history_mask.as_ref(),
                additional_ref_latent: additional_ref_latent.as_ref(),
                additional_ref_masks: additional_ref_masks.as_ref(),
                clip_fea: &clip_fea,
                context: &context,
                t,
                replace_flag: job.replace_flag,
            };
            let pred_cond = comps.dit.forward(&inp)?;
            let pred = if cfg_disabled {
                pred_cond
            } else {
                inp.context = &context_null;
                let pred_uncond = comps.dit.forward(&inp)?;
                cfg_combine(&pred_uncond, &pred_cond, job.guidance)?
            };
            latent = sched.step(&pred, &latent)?;
            latent = apply_clean_history(&latent, history_latent.as_ref())?;
            on_progress(Progress::Step {
                current: (i + 1) as u32,
                total,
            });
        }

        // --- decode this segment → pixels; stitch + carry history ---
        on_progress(Progress::Decoding);
        let video = comps
            .vae
            .decode(&latent.reshape((1, 16, lat_t, lat_h, lat_w))?)?; // [1,3,T_out,H,W]
        let (_, vc, vt, vh, vw) = video.dims5()?;
        let seg_video = video.reshape((vc, vt, vh, vw))?; // [3,T_out,H,W]
        let t_out = seg_video.dim(1)?;

        let keep_from = if seg_idx == 0 { 0 } else { job.segment_overlap };
        let piece = seg_video.narrow(1, keep_from, t_out - keep_from)?;
        out_pieces.push(piece);

        if seg_idx + 1 < segments.len() {
            let ov = job.segment_overlap;
            prev_history_pixel = Some(seg_video.narrow(1, t_out - ov, ov)?.contiguous()?);
        }
    }

    let piece_refs: Vec<&Tensor> = out_pieces.iter().collect();
    let full = Tensor::cat(&piece_refs, 1)?; // [3,T_total,H,W]
    let (fc, ft, fh, fw) = full.dims4()?;
    let frames = frames_to_images(&full.reshape((1, fc, ft, fh, fw))?)?;
    Ok(GenerationOutput::Video {
        frames,
        fps: job.fps,
        audio: None,
    })
}
