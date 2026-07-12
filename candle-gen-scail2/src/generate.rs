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
    /// UMT5 tokenizer, loaded+parsed **once** at component load and reused across the pos/neg encodes
    /// (sc-8991 / F-011) instead of re-parsing `tokenizer.json` per generate call.
    pub tok: TextTokenizer,
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

/// Round `n` down to the VAE temporal grid the z16 encoder needs: a `4k + 1` frame count (one latent
/// frame per [`TEMPORAL_STRIDE`] pixel frames, plus the leading key frame). `n == 0` maps to `0`.
fn vae_align(n: usize) -> usize {
    if n == 0 {
        0
    } else {
        ((n - 1) / TEMPORAL_STRIDE) * TEMPORAL_STRIDE + 1
    }
}

/// Segment plan over `total` driving frames (upstream `build_segments`): a single VAE-aligned segment
/// when the clip fits, else overlapping `segment_len` windows striding by `len − overlap`, **including a
/// final shortened window that reaches the end of the clip** so no trailing driving frames are dropped.
///
/// Callers must guarantee `overlap < len` (validated in [`generate`]); a `len == 0` clip yields no
/// segments. The tail window keeps the same `overlap` join with its predecessor as every interior
/// window, so the fixed `keep_from = overlap` stitch in [`generate`] stays correct.
fn build_segments(total: usize, len: usize, overlap: usize) -> Vec<(usize, usize)> {
    if total == 0 || len == 0 || overlap >= len {
        // Defensive: `generate` rejects these, but never underflow `len - overlap` for a direct
        // engine consumer who built a pathological `Scail2Job` (all fields are `pub`).
        return Vec::new();
    }
    if total <= len {
        return vec![(0, vae_align(total))];
    }
    let stride = len - overlap;
    let mut segs = Vec::new();
    let mut start = 0;
    while start + len <= total {
        segs.push((start, start + len));
        start += stride;
    }
    // Trailing partial: `start` now points just past the last full window's stride step, so frames
    // `[start, total)` are uncovered. Emit one more window that overlaps the previous by `overlap`
    // (i.e. begins at the same `start`) and extends to the clip end, VAE-aligned. Skip it only when
    // the remainder is shorter than a single latent frame (nothing decodable is lost).
    if start < total {
        let tail_len = vae_align(total - start);
        if tail_len > overlap {
            segs.push((start, start + tail_len));
        }
    }
    segs
}

/// Global denoise progress for step `i` (0-based) of segment `seg_idx` (0-based), given `steps` denoise
/// steps per segment over `num_segments` segments (F-125 / sc-11225).
///
/// A multi-segment SCAIL-2 job runs the same `steps` denoise once per segment. Reporting `current = i+1`
/// / `total = steps` per segment makes percent-complete jump backwards at every segment boundary — on
/// exactly the long clips where progress matters most. Instead this accumulates prior segments' steps so
/// the whole job is one monotonic `1 → steps·num_segments` sweep, matching the single-segment worker-UI
/// contract the testkit monotonicity check encodes.
fn segment_step_progress(
    seg_idx: usize,
    i: usize,
    steps: usize,
    num_segments: usize,
) -> (u32, u32) {
    let current = (seg_idx * steps + i + 1) as u32;
    let total = (steps * num_segments) as u32;
    (current, total)
}

/// Tokenize + UMT5-encode `prompt` → `[L, 4096]` (f32, un-padded; the DiT's `embed_text` pads to
/// `text_len`).
fn encode_text(
    te: &Umt5Encoder,
    tok: &TextTokenizer,
    prompt: &str,
    pad_token_id: u32,
    dev: &Device,
) -> CResult<Tensor> {
    let out = tok
        .tokenize(prompt)
        .map_err(|e| CandleError::Msg(format!("scail2: tokenize: {e}")))?;
    let mut ids: Vec<u32> = out.ids.iter().map(|&i| i as u32).collect();
    if ids.is_empty() {
        // The gen_core tokenizer short-circuits an empty prompt to zero ids, but UMT5/T5 encode the
        // empty string as a single token. A 0-length sequence here would build a degenerate `(1,1)`
        // tensor (the old `.max(1)` padded the *shape* but not the data), and the f32 embedding gather
        // over zero indices is a 0-element CUDA `index_select` that reads out of bounds →
        // `CUDA_ERROR_ILLEGAL_ADDRESS` (it surfaced deferred at the next cublas call). Emit one pad
        // token so the unconditional (empty negative-prompt) branch encodes a valid 1-token context.
        ids.push(pad_token_id);
    }
    let len = ids.len();
    let input_ids = Tensor::from_vec(ids, (1, len), dev)?;
    let embeds = te.encode(&input_ids)?; // [1, L, 4096]
    let (_, l, d) = embeds.dims3()?;
    Ok(embeds.reshape((l, d))?)
}

/// Build the SCAIL-2 UMT5 tokenizer from `root/tokenizer/tokenizer.json` **once** (sc-8991 / F-011), so
/// the generator caches it on its `Components` and reuses it across generate calls rather than
/// re-parsing per request. Byte-identical [`TokenizerConfig`] to the old per-generate load.
pub fn build_tokenizer(root: &Path, te_cfg: &TextEncoderConfig) -> CResult<TextTokenizer> {
    TextTokenizer::from_file(
        root.join("tokenizer/tokenizer.json"),
        TokenizerConfig {
            max_length: te_cfg.max_length,
            pad_token_id: te_cfg.pad_token_id,
            chat_template: ChatTemplate::None,
            pad_to_max_length: false,
        },
    )
    .map_err(|e| CandleError::Msg(format!("scail2: load tokenizer: {e}")))
}

/// Run the full SCAIL-2 generation for `job` against the resident `comps`. `cancel` is polled each
/// denoise step.
#[allow(clippy::too_many_arguments)]
pub fn generate(
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
    if job.segment_len == 0 {
        return Err(CandleError::Msg("scail2: segment_len must be > 0".into()));
    }
    if job.segment_overlap >= job.segment_len {
        // Guards the `stride = segment_len − segment_overlap` subtraction (a direct engine consumer can
        // build a `Scail2Job` with any `pub` field values) and keeps clean-history overlap well-defined.
        return Err(CandleError::Msg(format!(
            "scail2: segment_overlap ({}) must be < segment_len ({})",
            job.segment_overlap, job.segment_len
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
    // The tokenizer is loaded+parsed once at component load (sc-8991 / F-011) — reuse the cached one.
    let tok = &comps.tok;
    let pad_id = te_cfg.pad_token_id.max(0) as u32;
    let context = encode_text(&comps.te, tok, job.prompt, pad_id, dev)?;
    let context_null = if cfg_disabled {
        context.clone()
    } else {
        encode_text(&comps.te, tok, job.negative_prompt, pad_id, dev)?
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

    // Progress is reported **globally** across all segments (F-125 / sc-11225) — see
    // `segment_step_progress`. `Decoding` is likewise emitted once (on the final segment) rather than
    // mid-sweep, so the step count reaches 100% before the single decode signal.
    let seg_count = segments.len();

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
            let (current, total) = segment_step_progress(seg_idx, i, job.steps, seg_count);
            on_progress(Progress::Step { current, total });
        }

        // --- decode this segment → pixels; stitch + carry history ---
        // Emit `Decoding` once, after the final segment's denoise completes, so the global step sweep
        // reaches 100% before the single decode signal (rather than firing mid-sweep per segment).
        if seg_idx + 1 == seg_count {
            on_progress(Progress::Decoding);
        }
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

#[cfg(test)]
mod tests {
    use super::{build_segments, segment_step_progress, vae_align, TEMPORAL_STRIDE};

    // Shipped defaults (see `pipeline.rs`).
    const LEN: usize = 81;
    const OVERLAP: usize = 5;

    /// Every emitted window is VAE-temporal-aligned (`4k + 1`) so the z16 encoder accepts it, and non
    /// empty.
    fn assert_windows_valid(segs: &[(usize, usize)]) {
        for &(s, e) in segs {
            assert!(e > s, "empty/backwards window ({s}, {e})");
            let n = e - s;
            assert_eq!(
                n % TEMPORAL_STRIDE,
                1 % TEMPORAL_STRIDE,
                "window len {n} is not 4k+1"
            );
        }
    }

    #[test]
    fn vae_align_rounds_down_to_4k_plus_1() {
        assert_eq!(vae_align(0), 0);
        assert_eq!(vae_align(1), 1);
        assert_eq!(vae_align(4), 1);
        assert_eq!(vae_align(5), 5);
        assert_eq!(vae_align(8), 5);
        assert_eq!(vae_align(9), 9);
        assert_eq!(vae_align(81), 81); // a full window is already aligned
        assert_eq!(vae_align(200), 197);
    }

    #[test]
    fn single_segment_when_clip_fits() {
        // total <= len: one VAE-aligned window.
        let segs = build_segments(50, LEN, OVERLAP);
        assert_eq!(segs, vec![(0, 49)]); // vae_align(50) = 49
        assert_windows_valid(&segs);
    }

    #[test]
    fn single_frame_and_tiny_clips_do_not_panic() {
        assert_eq!(build_segments(1, LEN, OVERLAP), vec![(0, 1)]);
        assert_eq!(build_segments(2, LEN, OVERLAP), vec![(0, 1)]); // vae_align(2) = 1
        assert_eq!(build_segments(5, LEN, OVERLAP), vec![(0, 5)]);
    }

    #[test]
    fn empty_clip_yields_no_segments() {
        assert!(build_segments(0, LEN, OVERLAP).is_empty());
    }

    /// The core F-006 regression: a clip a bit longer than one window must NOT silently drop most of its
    /// tail. With len=81, overlap=5, stride=76 the old code emitted only `[0,81)` and dropped frames
    /// 81..120 (39 frames). The tail may still lose up to `TEMPORAL_STRIDE-1` frames to VAE alignment —
    /// the same unavoidable rounding the single-segment path applies — but no more.
    #[test]
    fn trailing_frames_are_not_dropped_just_over_one_window() {
        let total = 120;
        let segs = build_segments(total, LEN, OVERLAP);
        assert_windows_valid(&segs);
        // Full window then a shortened tail.
        assert_eq!(segs.first().copied(), Some((0, 81)));
        let (_, last_end) = *segs.last().unwrap();
        // Covers to the clip end, modulo VAE alignment (≤ 3 frames lost, not a whole window).
        assert_eq!(last_end, vae_align(total));
        assert!(total - last_end < TEMPORAL_STRIDE);
    }

    /// A long clip (the review's motivating case): frames must be covered essentially end-to-end (only
    /// the ≤3-frame VAE-alignment tail is dropped) and every interior join must be exactly `overlap`
    /// frames wide so the fixed `keep_from = overlap` stitch is correct.
    #[test]
    fn long_clip_covers_all_frames_with_uniform_overlap() {
        let total = 200;
        let segs = build_segments(total, LEN, OVERLAP);
        assert_windows_valid(&segs);
        assert!(segs.len() >= 3, "expected multiple windows, got {segs:?}");

        // Coverage: windows start at 0 and each next window starts before the previous ends, with the
        // join exactly `overlap` frames wide.
        assert_eq!(segs[0].0, 0);
        for pair in segs.windows(2) {
            let (prev_start, prev_end) = pair[0];
            let (next_start, _) = pair[1];
            assert!(next_start < prev_end, "gap between windows (no overlap)");
            assert!(next_start > prev_start);
            let overlap = prev_end - next_start;
            assert_eq!(overlap, OVERLAP, "join overlap must equal segment_overlap");
        }
        // The last window reaches the VAE-aligned clip end: at most TEMPORAL_STRIDE-1 frames lost.
        let (_, last_end) = *segs.last().unwrap();
        assert_eq!(last_end, vae_align(total));
        assert!(total - last_end < TEMPORAL_STRIDE);
    }

    /// Exact tiling: when the windows land flush on the clip end there must be no spurious zero/duplicate
    /// tail window.
    #[test]
    fn exact_multiple_boundary_has_no_spurious_tail() {
        // stride = 76. Two windows: [0,81) and [76,157). total = 157 → second window ends exactly at
        // total, remainder is 0, no tail appended.
        let total = 157;
        let segs = build_segments(total, LEN, OVERLAP);
        assert_windows_valid(&segs);
        assert_eq!(segs, vec![(0, 81), (76, 157)]);
        assert_eq!(segs.last().unwrap().1, total);
    }

    /// Remainder smaller than the overlap contributes nothing new (it lives entirely inside the previous
    /// window's tail), so no degenerate tail window is emitted.
    #[test]
    fn tiny_remainder_within_overlap_is_not_appended() {
        // total = 159: after [0,81),[76,157) the remainder is frames 157..159 (2 frames), vae_align = 1
        // which is < overlap(5) → skipped.
        let total = 159;
        let segs = build_segments(total, LEN, OVERLAP);
        assert_windows_valid(&segs);
        assert_eq!(segs, vec![(0, 81), (76, 157)]);
    }

    /// Underflow guard: a pathological direct-consumer job with `overlap >= len` must not underflow
    /// `len - overlap` (panic in debug / wrap in release) — it returns no segments and `generate`
    /// rejects it up front.
    #[test]
    fn overlap_ge_len_does_not_underflow() {
        assert!(build_segments(200, 81, 81).is_empty());
        assert!(build_segments(200, 81, 200).is_empty());
        assert!(build_segments(200, 0, 0).is_empty());
    }

    /// Replays the exact `(current, total)` sequence `generate`'s denoise loop emits over a multi-segment
    /// job and asserts the worker-UI progress contract (F-125 / sc-11225): one strictly monotonic sweep
    /// that never resets per segment and lands exactly on `total` at the final step.
    fn assert_global_progress_monotonic(steps: usize, num_segments: usize) {
        let expected_total = (steps * num_segments) as u32;
        let mut sweep = Vec::new();
        for seg_idx in 0..num_segments {
            for i in 0..steps {
                let (current, total) = segment_step_progress(seg_idx, i, steps, num_segments);
                assert_eq!(total, expected_total, "total must be the GLOBAL step count");
                sweep.push(current);
            }
        }
        // First reported step is 1, last is exactly `total`.
        assert_eq!(sweep.first().copied(), Some(1));
        assert_eq!(sweep.last().copied(), Some(expected_total));
        // Strictly increasing by 1 — no per-segment reset to 1, no backwards jump at a segment boundary.
        for pair in sweep.windows(2) {
            assert_eq!(
                pair[1],
                pair[0] + 1,
                "progress must advance by exactly 1 and never restart per segment"
            );
        }
        assert_eq!(sweep.len(), steps * num_segments);
    }

    /// The core F-125 regression: with >1 segment the old per-segment counting reset `current` to 1 at
    /// every boundary (percent-complete jumped backwards). Progress must instead be one global 1→total
    /// sweep. A driving clip of 200 frames tiles into multiple 81-frame segments (see
    /// `long_clip_covers_all_frames_with_uniform_overlap`).
    #[test]
    fn multi_segment_progress_is_global_and_monotonic() {
        let num_segments = build_segments(200, LEN, OVERLAP).len();
        assert!(
            num_segments >= 3,
            "expected a multi-segment tiling for the regression"
        );
        assert_global_progress_monotonic(30, num_segments);
        // A couple of other shapes for good measure.
        assert_global_progress_monotonic(1, 4);
        assert_global_progress_monotonic(50, 2);
    }

    /// A single-segment clip (the common short case) is unchanged: a plain 1→steps sweep with
    /// `total == steps`.
    #[test]
    fn single_segment_progress_matches_step_count() {
        let steps = 20;
        for i in 0..steps {
            let (current, total) = segment_step_progress(0, i, steps, 1);
            assert_eq!(current, (i + 1) as u32);
            assert_eq!(total, steps as u32);
        }
    }
}
