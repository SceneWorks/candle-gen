//! SAM3 multi-object video PCS pipeline (`Sam3VideoModel`) — candle port of `mlx-gen-sam3`'s
//! `video.rs` (epic 5482, sc-6245 under sc-5062).
//!
//! Pure-host orchestration over the (parity-green) tracker neural primitives in [`crate::tracker`]:
//! per frame, detect concept instances ([`crate::Sam3ImageSegmenter`]), propagate existing identities
//! through the per-object memory bank ([`Sam3Tracker::decode_tracked_frame`]), associate detections to
//! tracklets, seed new identities from unmatched detections
//! ([`Sam3Tracker::decode_mask_conditioning_frame`]), and encode each frame's masks into memory.
//!
//! Mirrors `transformers` `sam3_video/modeling_sam3_video.py` `_det_track_one_frame`, matching the
//! reference's **no-`kernels`** configuration: NMS (`det_nms_thresh`) and hole-filling are no-ops.
//! Masks flow as raw 288² logits (the processor sigmoids for display).

use std::collections::BTreeMap;
use std::sync::Arc;

use candle_gen::candle_core::{DType, Tensor};
use candle_gen::candle_nn::ops::sigmoid;
use candle_gen::gen_core::{CancelFlag, Quant};
use candle_gen::{CandleError, Result};

use crate::config::Sam3VisionConfig;
use crate::tracker::TrackerFrameOutput;
use crate::vision::Backbone;
use crate::{Sam3ImageSegmenter, Sam3Tracker};

// --- config (Sam3VideoConfig defaults) -----------------------------------------------------------
const LOW_RES: usize = 288; // low_res_mask_size
const SCORE_THRESH_DET: f32 = 0.5; // score_threshold_detection
const NEW_DET_THRESH: f32 = 0.7;
const ASSOC_IOU_THRESH: f32 = 0.1;
const TRK_ASSOC_IOU_THRESH: f32 = 0.5;
const HIGH_CONF_THRESH: f32 = 0.8;
const HIGH_IOU_THRESH: f32 = 0.8;
const NUM_MASKMEM: i32 = 7;
const MAX_COND_FRAME_NUM: i32 = 4;
const MAX_OBJ_PTRS: i32 = 16; // max_object_pointers_in_encoder
const RECONDITION_EVERY: i32 = 16;
const HOTSTART_DELAY: i32 = 15;
const HOTSTART_UNMATCH: usize = 8;
const HOTSTART_DUP: usize = 8;
const SUPPRESS_OCC_THRESH: f32 = 0.7; // suppress_overlapping_based_on_recent_occlusion_threshold
const NEVER_OCCLUDED: i32 = -1;
const ALWAYS_OCCLUDED: i32 = 100_000;
const NO_OBJ_LOGIT: f32 = -10.0;

/// Gathered spatial memory: `(relative_temporal_offset, maskmem_features, maskmem_pos_enc)` per frame.
type SpatialMem = Vec<(i32, Tensor, Tensor)>;
/// Gathered object pointers: `(temporal_offset, pointer [1,256])`.
type ObjPointers = Vec<(i32, Tensor)>;

/// A detection on a frame: raw 288² mask logits + score + box, plus the prompt that produced it.
struct Detection {
    mask: Vec<f32>, // [288·288] logits
    score: f32,
    prompt_id: i32,
}

/// One stored per-frame output for an object (the memory-bank entry).
#[derive(Clone)]
struct FrameMem {
    maskmem_features: Option<Tensor>, // [5184,1,64] seq-first (bf16-cast); None until memory-encoded
    maskmem_pos_enc: Option<Tensor>,  // [5184,1,64]
    object_pointer: Tensor,           // [1,256]
}

/// Per-object memory bank: conditioning-frame outputs (user/detection-seeded) + tracked-frame outputs.
#[derive(Default, Clone)]
struct ObjectBank {
    cond: BTreeMap<i32, FrameMem>,
    non_cond: BTreeMap<i32, FrameMem>,
}

/// The per-frame segmentation result: object id → 288² mask logits, in id order.
pub struct VideoFrameOutput {
    pub obj_ids: Vec<i32>,
    pub masks: Vec<Vec<f32>>, // each [288·288] logits, parallel to obj_ids
}

/// All per-video (per-`propagate`-call) mutable bookkeeping, factored out of [`Sam3VideoModel`] so a
/// clip's tracking state is a single value that can be reset wholesale between clips. F-093: because
/// `propagate` accumulates into these fields and only construction ever initialized them, a cached
/// model's second clip would otherwise inherit the first's banks/hotstart maps/removed-id bans and
/// silently corrupt tracking. Keeping them in one struct means resetting is `= SessionState::default()`
/// — a new field added here is cleared automatically, so the leak cannot silently reappear.
struct SessionState {
    obj_ids: Vec<i32>,      // ordered; index = obj_idx
    banks: Vec<ObjectBank>, // parallel to obj_ids
    obj_prompt: Vec<i32>,   // prompt id per obj_idx
    max_obj_id: i32,
    num_frames: i32,
    // hotstart metadata (keyed by obj_id)
    first_frame: BTreeMap<i32, i32>,
    unmatched_frames: BTreeMap<i32, Vec<i32>>,
    overlap_pairs: BTreeMap<(i32, i32), Vec<i32>>,
    removed: std::collections::BTreeSet<i32>,
    last_occluded: BTreeMap<i32, i32>,
}

impl Default for SessionState {
    fn default() -> Self {
        Self {
            obj_ids: Vec::new(),
            banks: Vec::new(),
            obj_prompt: Vec::new(),
            max_obj_id: -1, // no objects yet; first assigned id is max_obj_id + 1 = 0
            num_frames: 0,
            first_frame: BTreeMap::new(),
            unmatched_frames: BTreeMap::new(),
            overlap_pairs: BTreeMap::new(),
            removed: std::collections::BTreeSet::new(),
            last_occluded: BTreeMap::new(),
        }
    }
}

/// `Sam3VideoModel`: the detector + the tracker, driving the multi-object PCS pipeline.
pub struct Sam3VideoModel {
    segmenter: Sam3ImageSegmenter,
    tracker: Sam3Tracker,
    /// Per-video state (reset at the top of every [`Sam3VideoModel::propagate`] — F-093).
    session: SessionState,
}

impl Sam3VideoModel {
    pub fn from_weights(w: &crate::Weights) -> Result<Self> {
        // One PE backbone, shared between the detector segmenter and the tracker. Both load it from
        // the same `detector_model.vision_encoder.backbone` keys, so loading it twice would carry two
        // identical ~445M-param copies resident at video time (F-028).
        let cfg = Sam3VisionConfig::sam3();
        let backbone = Arc::new(Backbone::from_weights(
            w,
            "detector_model.vision_encoder.backbone",
            &cfg,
        )?);
        Ok(Self {
            segmenter: Sam3ImageSegmenter::from_weights_with_backbone(w, backbone.clone())?,
            tracker: Sam3Tracker::from_weights_with_backbone(w, backbone)?,
            session: SessionState::default(),
        })
    }

    /// Affine-quantize the whole video model to Q4/Q8. The PE backbone is shared (one `Arc`) between
    /// the detector segmenter and the tracker (F-028), so it is quantized **once** — via the
    /// segmenter, whose `Arc::make_mut` clones the shared backbone, quantizes the copy, and leaves the
    /// segmenter holding the quantized one — then reinstalled into the tracker, which only quantizes
    /// its own heads. Quantizing both independently would re-duplicate the weights we deduplicated.
    pub fn quantize(&mut self, quant: Quant) -> Result<()> {
        self.segmenter.quantize(quant)?; // backbone (clone-on-write) + segmenter heads
        self.tracker
            .set_backbone(self.segmenter.vision_backbone_arc());
        self.tracker.quantize_heads(quant)
    }

    /// The detector segmenter (read-only) — for the F-028 shared-backbone parity check.
    #[cfg(test)]
    pub(crate) fn segmenter(&self) -> &Sam3ImageSegmenter {
        &self.segmenter
    }

    /// The tracker (read-only) — for the F-028 shared-backbone parity check.
    #[cfg(test)]
    pub(crate) fn tracker(&self) -> &Sam3Tracker {
        &self.tracker
    }

    /// F-093: clear all per-video session bookkeeping to its just-constructed state, keeping the
    /// loaded weights. `propagate` accumulates per-clip state (`obj_ids`/`banks`/`obj_prompt`, the
    /// hotstart maps, `removed`, `last_occluded`, the id/frame counters) but only `from_weights`
    /// initialized it, so a second `propagate` on a cached instance would inherit the previous clip's
    /// banks and bans and silently corrupt tracking. `propagate` calls this at entry so every clip
    /// starts clean; it is also `pub` so a caller can reset explicitly between clips if desired.
    pub fn reset_session(&mut self) {
        self.session = SessionState::default();
    }

    /// Process a whole video (forward, non-streaming): `frames[f]` = NCHW `[1,3,1008,1008]`; one text
    /// prompt (`input_ids[1,32]` + `text_mask`). Returns per-frame `obj_id → 288² mask logits`.
    ///
    /// `cancel` is the caller's cooperative cancel flag (the gen-core video per-step cancel
    /// contract, sc-8972; mirrors `mlx-gen-sam3`): checked before each (seconds-to-minutes) frame,
    /// surfacing the typed [`CandleError::Canceled`] on trip. `progress` is invoked
    /// `(frame_index, total_frames)` after each propagated frame.
    pub fn propagate(
        &mut self,
        frames: &[Tensor],
        input_ids: &Tensor,
        text_mask: &[i32],
        cancel: Option<&CancelFlag>,
        mut progress: Option<&mut dyn FnMut(usize, usize)>,
    ) -> Result<Vec<VideoFrameOutput>> {
        // F-093: start every clip from clean session state so a cached instance's second `propagate`
        // never inherits the previous clip's banks/hotstart bookkeeping/removed-id bans.
        self.reset_session();
        self.session.num_frames = frames.len() as i32;
        // The concept prompt is fixed for the whole video; encode it through the 24-layer CLIP text
        // tower once here and reuse the features on every frame (F-016).
        let text = self.segmenter.encode_text(input_ids, text_mask)?; // [1, N, 256]
        let total = frames.len();
        let mut outputs = Vec::new();
        for (f, px) in frames.iter().enumerate() {
            // Honor the engine cancellation contract — check before each (seconds-to-minutes) frame.
            check_canceled(cancel)?;
            outputs.push(self.process_frame(f as i32, px, &text, text_mask)?);
            if let Some(cb) = progress.as_deref_mut() {
                cb(f, total);
            }
        }
        Ok(outputs)
    }

    fn process_frame(
        &mut self,
        frame_idx: i32,
        pixels: &Tensor,
        text: &Tensor,
        text_mask: &[i32],
    ) -> Result<VideoFrameOutput> {
        // --- Step 1: vision + detection (one shared PE backbone pass feeds both necks) ---
        let features = self.segmenter.backbone_features(pixels)?; // [1,72,72,C], 32-layer ViT once
        let (img_emb, high_res) = self.tracker.encode_frame_from_features(&features)?; // [1,72,72,256], [s0,s1]
        let g = img_emb.dim(1)?;
        let cvf = img_emb.reshape((g * g, 1, 256))?;
        let cvp = self.tracker.frame_position_encoding(g)?;
        let det = self.run_detection(&features, text, text_mask)?;

        // --- Step 2: propagate existing identities (run_mem_encoder = false) ---
        let num_existing = self.session.obj_ids.len();
        let mut trk_masks: Vec<Vec<f32>> = Vec::with_capacity(num_existing); // [288²] logits per obj
        let mut trk_scores: Vec<f32> = Vec::with_capacity(num_existing);
        for obj_idx in 0..num_existing {
            let (spatial, pointers, max_optr) = self.gather_memory(obj_idx, frame_idx);
            let conditioned = self
                .tracker
                .prepare_memory_conditioned_features(&cvf, &cvp, &spatial, &pointers, max_optr)?;
            let out = self.tracker.decode_tracked_frame(&conditioned, &high_res)?;
            let low = to_vec(&out.low_res)?;
            self.session.banks[obj_idx].non_cond.insert(
                frame_idx,
                FrameMem {
                    maskmem_features: None,
                    maskmem_pos_enc: None,
                    object_pointer: out.object_pointer.clone(),
                },
            );
            trk_scores.push(out.object_score);
            trk_masks.push(low);
        }

        // --- Step 3: associate + new-object ids + hotstart ---
        let assoc = self.associate(&det, &trk_masks);
        let new_obj_ids: Vec<i32> = (0..assoc.new_det_inds.len() as i32)
            .map(|i| self.session.max_obj_id + 1 + i)
            .collect();
        let removed_now = self.process_hotstart(frame_idx, &assoc, &new_obj_ids);

        // recondition every Nth frame: confidently re-detected tracks become conditioning frames.
        let mut reconditioned_obj_ids: Vec<i32> = Vec::new();
        if RECONDITION_EVERY > 0
            && frame_idx % RECONDITION_EVERY == 0
            && !assoc.trk_id_to_max_iou_high_conf_det.is_empty()
        {
            for &trk_oid in assoc.trk_id_to_max_iou_high_conf_det.keys() {
                if let Some(obj_idx) = self.session.obj_ids.iter().position(|&o| o == trk_oid) {
                    if trk_scores.get(obj_idx).copied().unwrap_or(f32::MIN) > HIGH_CONF_THRESH {
                        reconditioned_obj_ids.push(trk_oid);
                    }
                }
            }
        }

        // --- Step 4 (planning tail): suppress overlaps + encode memory for existing objects ---
        if num_existing > 0 {
            self.suppress_overlapping_recent_occlusion(frame_idx, &mut trk_masks, &removed_now);
            self.tracker_update_memories(frame_idx, &img_emb, &trk_masks)?;
            // move reconditioned frames from non_cond → cond so they seed future memory selection.
            for &oid in &reconditioned_obj_ids {
                if let Some(obj_idx) = self.session.obj_ids.iter().position(|&o| o == oid) {
                    if let Some(fm) = self.session.banks[obj_idx].non_cond.remove(&frame_idx) {
                        self.session.banks[obj_idx].cond.insert(frame_idx, fm);
                    }
                }
            }
        }

        // --- Step 5 (execution): add new objects from unmatched detections ---
        for (&oid, &di) in new_obj_ids.iter().zip(&assoc.new_det_inds) {
            self.add_object(oid, det.dets[di].prompt_id);
            let obj_idx = self.session.obj_ids.len() - 1;
            // binarize the detection logits at 0.5 (reference: det_mask >= 0.5) → mask prompt.
            let mask_bin: Vec<f32> = det.dets[di]
                .mask
                .iter()
                .map(|&v| if v >= 0.5 { 1.0 } else { 0.0 })
                .collect();
            let mask_nhwc = Tensor::from_vec(mask_bin, (1, LOW_RES, LOW_RES, 1), img_emb.device())?;
            let out: TrackerFrameOutput = self
                .tracker
                .decode_mask_conditioning_frame(&img_emb, &high_res, &mask_nhwc)?;
            let mem =
                self.tracker
                    .encode_new_memory(&img_emb, &out.high_res, out.object_score, true)?;
            self.session.banks[obj_idx].cond.insert(
                frame_idx,
                FrameMem {
                    maskmem_features: Some(seq_first(&mem.features, true)?),
                    maskmem_pos_enc: Some(seq_first(&mem.pos, false)?),
                    object_pointer: out.object_pointer,
                },
            );
        }
        // remove objects flagged by hotstart
        for oid in &removed_now {
            self.remove_object(*oid);
        }

        // F-015: bound the per-object memory bank. Every frame appends a `non_cond` `FrameMem` (~2.65 MB
        // of device tensors per live object) but `gather_memory` never reads entries older than its
        // `NUM_MASKMEM` / `MAX_OBJ_PTRS` windows, so a long clip's resident VRAM climbs without a ceiling
        // (~2.4 GB for 300 frames × 3 objects). Evict everything those windows have slid past — after all
        // this frame's writes (the `non_cond` inserts/fills + the recondition `non_cond`→`cond` move) so
        // it never races the current frame's state. Mirrors the mlx twin's F-024 eviction.
        self.evict_stale_memory(frame_idx);

        // --- build outputs ---
        self.build_outputs(
            &det,
            &assoc,
            &new_obj_ids,
            &trk_masks,
            &reconditioned_obj_ids,
        )
    }

    // ----- detection (run_detection, single prompt, NMS off) -----
    fn run_detection(
        &self,
        features: &Tensor,
        text: &Tensor,
        text_mask: &[i32],
    ) -> Result<DetFrame> {
        let seg = self
            .segmenter
            .forward_from_backbone_with_text(features, text, text_mask)?;
        let presence = sigmoid(&seg.presence_logits)?
            .flatten_all()?
            .to_vec1::<f32>()?[0];
        let probs: Vec<f32> = sigmoid(&seg.pred_logits)?
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .map(|&s| s * presence)
            .collect();
        // F-014: read back only the mask rows we actually keep. `pred_masks` is `[1,200,288,288]`
        // (~16.6M f32 ≈ 66 MB) but only the handful of queries scoring above `SCORE_THRESH_DET`
        // survive, so `index_select` the kept rows on-device first and read back just those — a
        // synchronous PCIe transfer that now scales with detections, not `num_queries`.
        let dets = select_detections(&seg.pred_masks, &probs)?;
        Ok(DetFrame { dets })
    }

    // ----- memory bank gather (F2.4 selection logic) -----
    fn gather_memory(&self, obj_idx: usize, frame_idx: i32) -> (SpatialMem, ObjPointers, i32) {
        let bank = &self.session.banks[obj_idx];
        // spatial memory: closest cond frames (offset 0) + non-cond at offsets [num_maskmem-1..1].
        let (selected_cond, unselected_cond) =
            select_closest_cond_frames(frame_idx, &bank.cond, MAX_COND_FRAME_NUM);
        let mut spatial: Vec<(i32, Tensor, Tensor)> = Vec::new();
        for f in &selected_cond {
            if let Some(m) = bank.cond.get(f) {
                if let (Some(feat), Some(pos)) = (&m.maskmem_features, &m.maskmem_pos_enc) {
                    spatial.push((0, feat.clone(), pos.clone()));
                }
            }
        }
        for rel in (1..NUM_MASKMEM).rev() {
            let prev = frame_idx - rel;
            let out = bank.non_cond.get(&prev).or_else(|| {
                if unselected_cond.contains(&prev) {
                    bank.cond.get(&prev)
                } else {
                    None
                }
            });
            if let Some(m) = out {
                if let (Some(feat), Some(pos)) = (&m.maskmem_features, &m.maskmem_pos_enc) {
                    spatial.push((rel, feat.clone(), pos.clone()));
                }
            }
        }
        // object pointers: eligible cond frames (t <= frame_idx) + non-cond up to max_optr-1.
        let max_optr = self.session.num_frames.min(MAX_OBJ_PTRS);
        let mut pointers: Vec<(i32, Tensor)> = Vec::new();
        for (&t, m) in &bank.cond {
            if t <= frame_idx {
                pointers.push((frame_idx - t, m.object_pointer.clone()));
            }
        }
        for t_diff in 1..max_optr {
            let r = frame_idx - t_diff;
            if r < 0 || r >= self.session.num_frames {
                break;
            }
            if let Some(m) = bank.non_cond.get(&r) {
                pointers.push((t_diff, m.object_pointer.clone()));
            }
        }
        (spatial, pointers, max_optr)
    }

    /// F-015: evict `non_cond` bank entries that no future `gather_memory` can read, derived from the
    /// same `NUM_MASKMEM` / `MAX_OBJ_PTRS` windows `gather_memory` uses (single source of truth). After
    /// processing `frame_idx` the next gather is at frame ≥ `frame_idx + 1`; heavy tensors are read back
    /// to `(frame_idx+1) − (NUM_MASKMEM−1)` (the spatial fallback window) and object pointers back to
    /// `(frame_idx+1) − (MAX_OBJ_PTRS−1)` (the pointer window), and both windows only slide forward — so
    /// any entry older than that is dead for the rest of the session. `cond` **entries** are left intact
    /// (the pointer loop reads their lightweight `object_pointer` at arbitrary keys), but their heavy
    /// `maskmem_*` tensors are nulled by [`evict_stale_cond_heavy`] once they fall out of every future
    /// spatial-read window, so a long clip's resident `cond` memory also stops climbing. For a clip
    /// shorter than the windows every entry stays readable, so eviction is a no-op and output is
    /// unchanged.
    fn evict_stale_memory(&mut self, frame_idx: i32) {
        let heavy_keep = frame_idx + 1 - (NUM_MASKMEM - 1);
        let ptr_keep = frame_idx + 1 - (MAX_OBJ_PTRS - 1);
        for bank in &mut self.session.banks {
            evict_stale_bank(bank, heavy_keep, ptr_keep);
            evict_stale_cond_heavy(bank, heavy_keep);
        }
    }

    // ----- association (_associate_det_trk; mask-IoU, no Hungarian) -----
    #[allow(clippy::needless_range_loop)] // parallel-array indexing (iou / obj_ids / dets)
    fn associate(&self, det: &DetFrame, trk_masks: &[Vec<f32>]) -> Assoc {
        let n = det.dets.len();
        let m = trk_masks.len();
        let mut a = Assoc::default();
        if m == 0 {
            a.new_det_inds = (0..n).collect();
            return a;
        }
        let det_bin: Vec<Vec<bool>> = det.dets.iter().map(|d| binarize(&d.mask)).collect();
        let trk_bin: Vec<Vec<bool>> = trk_masks.iter().map(|t| binarize(t)).collect();
        let trk_nonempty: Vec<bool> = trk_bin.iter().map(|t| t.iter().any(|&x| x)).collect();
        // IoU[n][m], zeroed across prompt groups.
        let mut iou = vec![vec![0f32; m]; n];
        for (i, db) in det_bin.iter().enumerate() {
            for (j, tb) in trk_bin.iter().enumerate() {
                if det.dets[i].prompt_id == self.session.obj_prompt[j] {
                    iou[i][j] = mask_iou(db, tb);
                }
            }
        }
        // tracks: unmatched if non-empty and no det IoU >= trk_assoc (zero-area tracks are neither
        // matched nor unmatched — the reference `_associate_det_trk` drops them from both lists).
        for j in 0..m {
            let matched = (0..n).any(|i| iou[i][j] >= TRK_ASSOC_IOU_THRESH);
            if trk_nonempty[j] && !matched {
                a.unmatched_trk.push(self.session.obj_ids[j]);
            }
        }
        // detections: new if score >= new_det_thresh and no track IoU >= assoc_iou.
        for i in 0..n {
            let matches_any = (0..m).any(|j| iou[i][j] >= ASSOC_IOU_THRESH);
            // "New detection": high enough score and matched to no existing track. Computed once and
            // reused below for the high-conf-IoU bookkeeping.
            let is_new = det.dets[i].score >= NEW_DET_THRESH && !matches_any;
            if is_new {
                a.new_det_inds.push(i);
            }
            let matched: Vec<i32> = (0..m)
                .filter(|&j| iou[i][j] >= ASSOC_IOU_THRESH)
                .map(|j| self.session.obj_ids[j])
                .collect();
            let (best_j, best_iou) = (0..m).fold((0usize, -1f32), |(bj, bi), j| {
                if iou[i][j] > bi {
                    (j, iou[i][j])
                } else {
                    (bj, bi)
                }
            });
            if det.dets[i].score >= HIGH_CONF_THRESH
                && !is_new
                && best_iou >= HIGH_IOU_THRESH
                && m > 0
            {
                a.trk_id_to_max_iou_high_conf_det
                    .insert(self.session.obj_ids[best_j], i);
            }
            a.det_to_matched_trk.push(matched);
        }
        a
    }

    // ----- hotstart (_process_hotstart) -----
    // F-129 (sc-11235): the reference's per-track `trk_keep_alive` counter feeds ONLY the
    // `suppressed_obj_ids` per-frame suppression branch (`modeling_sam3_video.py::_process_hotstart`),
    // which is gated on `not suppress_unmatched_only_within_hotstart`. facebook/sam3 ships that config
    // `True` (default), so the counter has no reader in the reference and never affects tracking or
    // removal — removal is driven solely by `unmatched_frames` / `overlap_pairs`, exactly as here. The
    // write-only counter (and its INIT/MAX/MIN constants) was therefore deleted rather than wired, and
    // the mlx twin carries the same dead state. E2E video parity against facebook/sam3 is unchanged.
    fn process_hotstart(&mut self, frame_idx: i32, a: &Assoc, new_obj_ids: &[i32]) -> Vec<i32> {
        let mut newly_removed = Vec::new();
        let hotstart_diff = frame_idx - HOTSTART_DELAY;
        for &oid in new_obj_ids {
            self.session.first_frame.entry(oid).or_insert(frame_idx);
        }
        for &oid in &a.unmatched_trk {
            self.session
                .unmatched_frames
                .entry(oid)
                .or_default()
                .push(frame_idx);
        }
        let unmatched_snapshot: Vec<(i32, usize, i32)> = self
            .session
            .unmatched_frames
            .iter()
            .map(|(&oid, fs)| {
                (
                    oid,
                    fs.len(),
                    *self.session.first_frame.get(&oid).unwrap_or(&0),
                )
            })
            .collect();
        for (oid, count, first) in unmatched_snapshot {
            if self.session.removed.contains(&oid) || newly_removed.contains(&oid) {
                continue;
            }
            if count >= HOTSTART_UNMATCH && first > hotstart_diff {
                newly_removed.push(oid);
            }
        }
        for trks in &a.det_to_matched_trk {
            if trks.len() < 2 {
                continue;
            }
            let first_appear = *trks
                .iter()
                .min_by_key(|&&o| *self.session.first_frame.get(&o).unwrap_or(&0))
                .unwrap();
            for &oid in trks {
                if oid != first_appear {
                    self.session
                        .overlap_pairs
                        .entry((first_appear, oid))
                        .or_default()
                        .push(frame_idx);
                }
            }
        }
        let overlap_snapshot: Vec<(i32, usize, i32)> = self
            .session
            .overlap_pairs
            .iter()
            .map(|(&(_f, oid), fs)| {
                (
                    oid,
                    fs.len(),
                    *self.session.first_frame.get(&oid).unwrap_or(&0),
                )
            })
            .collect();
        for (oid, count, first) in overlap_snapshot {
            if self.session.removed.contains(&oid) || newly_removed.contains(&oid) {
                continue;
            }
            if first > hotstart_diff && count >= HOTSTART_DUP {
                newly_removed.push(oid);
            }
        }
        for &oid in &newly_removed {
            self.session.removed.insert(oid);
        }
        newly_removed
    }

    // ----- occlusion-based overlap suppression -----
    #[allow(clippy::needless_range_loop)] // parallel-array indexing (masks / obj_ids / last_occ)
    fn suppress_overlapping_recent_occlusion(
        &mut self,
        frame_idx: i32,
        trk_masks: &mut [Vec<f32>],
        removed_now: &[i32],
    ) {
        let n = trk_masks.len();
        if n == 0 {
            return;
        }
        let bin: Vec<Vec<bool>> = trk_masks.iter().map(|t| binarize(t)).collect();
        let last_occ: Vec<i32> = (0..n)
            .map(|j| {
                let oid = self.session.obj_ids[j];
                self.session.last_occluded.get(&oid).copied().unwrap_or(
                    if removed_now.contains(&oid) {
                        ALWAYS_OCCLUDED
                    } else {
                        NEVER_OCCLUDED
                    },
                )
            })
            .collect();
        let mut to_suppress = vec![false; n];
        for pg in unique(&self.session.obj_prompt[0..n]) {
            let idxs: Vec<usize> = (0..n)
                .filter(|&j| self.session.obj_prompt[j] == pg)
                .collect();
            if idxs.len() <= 1 {
                continue;
            }
            for ai in 0..idxs.len() {
                for bj in (ai + 1)..idxs.len() {
                    let (i, j) = (idxs[ai], idxs[bj]);
                    if mask_iou(&bin[i], &bin[j]) < SUPPRESS_OCC_THRESH {
                        continue;
                    }
                    if last_occ[i] > last_occ[j] && last_occ[j] > NEVER_OCCLUDED {
                        to_suppress[i] = true;
                    }
                    if last_occ[j] > last_occ[i] && last_occ[i] > NEVER_OCCLUDED {
                        to_suppress[j] = true;
                    }
                }
            }
        }
        for j in 0..n {
            let occluded = !bin[j].iter().any(|&x| x);
            let oid = self.session.obj_ids[j];
            let new_lo = if occluded || to_suppress[j] {
                frame_idx
            } else {
                last_occ[j]
            };
            self.session.last_occluded.insert(oid, new_lo);
            if to_suppress[j] {
                for v in trk_masks[j].iter_mut() {
                    *v = NO_OBJ_LOGIT;
                }
            }
        }
    }

    // ----- memory encode for existing objects (_tracker_update_memories) -----
    #[allow(clippy::needless_range_loop)] // index into parallel banks / constrained masks
    fn tracker_update_memories(
        &mut self,
        frame_idx: i32,
        img_emb: &Tensor,
        trk_masks: &[Vec<f32>],
    ) -> Result<()> {
        let n = trk_masks.len();
        if n == 0 {
            return Ok(());
        }
        let constrained = suppress_pw_area_shrinkage(trk_masks, &self.session.obj_prompt[0..n]);
        for obj_idx in 0..n {
            let mask = &constrained[obj_idx];
            let appearing = mask.iter().any(|&v| v > 0.0);
            let object_score = if appearing { 10.0 } else { -10.0 };
            let mask_arr =
                Tensor::from_vec(mask.clone(), (1, 1, LOW_RES, LOW_RES), img_emb.device())?;
            let mem = self
                .tracker
                .encode_new_memory(img_emb, &mask_arr, object_score, false)?;
            let feat = seq_first(&mem.features, true)?;
            let pos = seq_first(&mem.pos, false)?;
            if let Some(fm) = self.session.banks[obj_idx].cond.get_mut(&frame_idx) {
                fm.maskmem_features = Some(feat);
                fm.maskmem_pos_enc = Some(pos);
            } else if let Some(fm) = self.session.banks[obj_idx].non_cond.get_mut(&frame_idx) {
                fm.maskmem_features = Some(feat);
                fm.maskmem_pos_enc = Some(pos);
            }
        }
        Ok(())
    }

    // ----- build outputs -----
    #[allow(clippy::needless_range_loop)] // parallel-array indexing (trk_masks / obj_ids)
    fn build_outputs(
        &self,
        det: &DetFrame,
        a: &Assoc,
        new_obj_ids: &[i32],
        trk_masks: &[Vec<f32>],
        reconditioned_obj_ids: &[i32],
    ) -> Result<VideoFrameOutput> {
        let mut obj_ids = Vec::new();
        let mut masks = Vec::new();
        let num_existing = trk_masks.len();
        for j in 0..num_existing {
            let oid = self.session.obj_ids[j];
            obj_ids.push(oid);
            if reconditioned_obj_ids.contains(&oid) {
                if let Some(&di) = a.trk_id_to_max_iou_high_conf_det.get(&oid) {
                    masks.push(det.dets[di].mask.clone());
                    continue;
                }
            }
            masks.push(trk_masks[j].clone());
        }
        for (&oid, &di) in new_obj_ids.iter().zip(&a.new_det_inds) {
            obj_ids.push(oid);
            masks.push(det.dets[di].mask.clone());
        }
        Ok(VideoFrameOutput { obj_ids, masks })
    }

    fn add_object(&mut self, obj_id: i32, prompt_id: i32) {
        self.session.obj_ids.push(obj_id);
        self.session.banks.push(ObjectBank::default());
        self.session.obj_prompt.push(prompt_id);
        self.session.max_obj_id = self.session.max_obj_id.max(obj_id);
    }

    fn remove_object(&mut self, obj_id: i32) {
        if let Some(idx) = self.session.obj_ids.iter().position(|&o| o == obj_id) {
            self.session.obj_ids.remove(idx);
            self.session.banks.remove(idx);
            self.session.obj_prompt.remove(idx);
        }
    }
}

// --- helpers -------------------------------------------------------------------------------------

struct DetFrame {
    dets: Vec<Detection>,
}

#[derive(Default)]
struct Assoc {
    new_det_inds: Vec<usize>,
    unmatched_trk: Vec<i32>,
    det_to_matched_trk: Vec<Vec<i32>>,
    trk_id_to_max_iou_high_conf_det: BTreeMap<i32, usize>,
}

/// `_select_closest_cond_frames`: ≤ `max` cond frames closest to `frame_idx`. Returns
/// (selected frame indices, unselected frame indices).
fn select_closest_cond_frames(
    frame_idx: i32,
    cond: &BTreeMap<i32, FrameMem>,
    max: i32,
) -> (Vec<i32>, std::collections::BTreeSet<i32>) {
    let keys: Vec<i32> = cond.keys().copied().collect();
    if max == -1 || keys.len() as i32 <= max {
        return (keys, std::collections::BTreeSet::new());
    }
    let mut selected: std::collections::BTreeSet<i32> = std::collections::BTreeSet::new();
    if let Some(&before) = keys.iter().filter(|&&t| t < frame_idx).max() {
        selected.insert(before);
    }
    if let Some(&after) = keys.iter().filter(|&&t| t >= frame_idx).min() {
        selected.insert(after);
    }
    let mut remaining: Vec<i32> = keys
        .iter()
        .copied()
        .filter(|t| !selected.contains(t))
        .collect();
    remaining.sort_by_key(|&t| (t - frame_idx).abs());
    for t in remaining
        .into_iter()
        .take((max - selected.len() as i32).max(0) as usize)
    {
        selected.insert(t);
    }
    let unselected: std::collections::BTreeSet<i32> = keys
        .iter()
        .copied()
        .filter(|t| !selected.contains(t))
        .collect();
    (selected.into_iter().collect(), unselected)
}

/// Prune one object's `non_cond` bank to the future-readable window (F-015 eviction core, factored out
/// as a free fn so the bound is unit-testable without a loaded model): entries older than `heavy_keep`
/// have their heavy tensors nulled (gather short-circuits on `None`), and entries older than `ptr_keep`
/// are dropped entirely (the object-pointer window has passed). `heavy_keep`/`ptr_keep` are the next
/// gather's read floors; callers derive them from `NUM_MASKMEM`/`MAX_OBJ_PTRS`.
fn evict_stale_bank(bank: &mut ObjectBank, heavy_keep: i32, ptr_keep: i32) {
    for (&k, m) in bank.non_cond.iter_mut() {
        if k < heavy_keep {
            m.maskmem_features = None;
            m.maskmem_pos_enc = None;
        }
    }
    bank.non_cond.retain(|&k, _| k >= ptr_keep);
}

/// Bound the per-object **`cond`** bank's resident memory (the F-015 sibling): null the heavy
/// `maskmem_*` tensors of conditioning frames that no future `gather_memory` can read for spatial
/// memory, keeping the lightweight `object_pointer` (still read by the pointer loop) and the entry
/// itself (so [`select_closest_cond_frames`] still sees the key — it just contributes nothing, exactly
/// as for an unselected frame).
///
/// A cond frame at key `k` can be read for spatial memory by a future gather (frame_idx' ≥ `frame_idx`
/// + 1) iff EITHER:
///  - it is still *selectable* — fewer than [`MAX_COND_FRAME_NUM`] cond frames have a key `> k`. New
///    cond frames only accrue (entries are never removed, only heavy-nulled), so once
///    `MAX_COND_FRAME_NUM` newer keys exist `k` is never among the closest again (for any frame_idx' >
///    all current keys the closest are the largest keys); OR
///  - it is inside the spatial *fallback* window, `k >= heavy_keep` (`= frame_idx + 1 −
///    (NUM_MASKMEM − 1)`), which only slides forward.
///
/// So the heavy tensors are dead exactly when BOTH fail: `k < heavy_keep` AND ≥ `MAX_COND_FRAME_NUM`
/// cond keys exceed `k`. The newest `MAX_COND_FRAME_NUM` keys are therefore always protected.
fn evict_stale_cond_heavy(bank: &mut ObjectBank, heavy_keep: i32) {
    let n = bank.cond.len() as i32;
    if n <= MAX_COND_FRAME_NUM {
        return; // every cond frame is always selectable → nothing droppable
    }
    // BTreeMap iterates ascending, so index `i` has `n - 1 - i` newer keys; "at least
    // MAX_COND_FRAME_NUM newer" is `i < n - MAX_COND_FRAME_NUM`. The newest MAX_COND_FRAME_NUM keys
    // stay selectable and are never nulled.
    let droppable_below = n - MAX_COND_FRAME_NUM;
    for (i, (&k, m)) in bank.cond.iter_mut().enumerate() {
        if (i as i32) < droppable_below && k < heavy_keep {
            m.maskmem_features = None;
            m.maskmem_pos_enc = None;
        }
    }
}

/// `_apply_non_overlapping_constraints` + `_suppress_shrinked_masks` per prompt group.
#[allow(clippy::needless_range_loop)] // pixel-wise argmax over parallel grouped masks
fn suppress_pw_area_shrinkage(masks: &[Vec<f32>], prompts: &[i32]) -> Vec<Vec<f32>> {
    let n = masks.len();
    let mut out = masks.to_vec();
    for pg in unique(prompts) {
        let idxs: Vec<usize> = (0..n).filter(|&j| prompts[j] == pg).collect();
        if idxs.len() <= 1 {
            continue;
        }
        let len = masks[0].len();
        let mut constrained: Vec<Vec<f32>> = idxs.iter().map(|&j| masks[j].clone()).collect();
        for p in 0..len {
            let (mut best, mut bv) = (0usize, f32::NEG_INFINITY);
            for (gi, &j) in idxs.iter().enumerate() {
                if masks[j][p] > bv {
                    bv = masks[j][p];
                    best = gi;
                }
            }
            for gi in 0..idxs.len() {
                if gi != best && constrained[gi][p] > NO_OBJ_LOGIT {
                    constrained[gi][p] = NO_OBJ_LOGIT;
                }
            }
        }
        for (gi, &j) in idxs.iter().enumerate() {
            let before = masks[j].iter().filter(|&&v| v > 0.0).count().max(1) as f32;
            let after = constrained[gi].iter().filter(|&&v| v > 0.0).count() as f32;
            if after / before >= 0.3 {
                out[j] = constrained[gi].clone();
            } else {
                out[j] = masks[j].iter().map(|&v| v.min(NO_OBJ_LOGIT)).collect();
            }
        }
    }
    out
}

fn unique(v: &[i32]) -> Vec<i32> {
    let mut s: Vec<i32> = v.to_vec();
    s.sort_unstable();
    s.dedup();
    s
}

/// Threshold a mask (logits or probabilities) at 0 → per-pixel bool.
fn binarize(m: &[f32]) -> Vec<bool> {
    m.iter().map(|&v| v > 0.0).collect()
}

fn mask_iou(a: &[bool], b: &[bool]) -> f32 {
    let mut inter = 0u32;
    let mut uni = 0u32;
    for (&x, &y) in a.iter().zip(b) {
        if x && y {
            inter += 1;
        }
        if x || y {
            uni += 1;
        }
    }
    inter as f32 / (uni.max(1) as f32)
}

fn to_vec(a: &Tensor) -> Result<Vec<f32>> {
    Ok(a.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?)
}

/// F-014: select the detections whose score exceeds `SCORE_THRESH_DET` from a `[1, Q, 288, 288]`
/// mask tensor, reading back **only** the kept rows.
///
/// `pred_masks` carries all `Q` (=200) query masks (~66 MB) but only the handful passing the
/// threshold are ever used, so we `index_select` the kept rows on-device and copy just those to
/// the host — the transfer scales with detections, not `num_queries`. The kept queries are
/// emitted in ascending query order, so the result is identical to reading the whole tensor back
/// and filtering on the host.
fn select_detections(pred_masks: &Tensor, probs: &[f32]) -> Result<Vec<Detection>> {
    let kept: Vec<u32> = probs
        .iter()
        .enumerate()
        .filter(|&(_, &p)| p > SCORE_THRESH_DET)
        .map(|(qi, _)| qi as u32)
        .collect();
    if kept.is_empty() {
        return Ok(Vec::new());
    }
    let per = LOW_RES * LOW_RES;
    let q = pred_masks.dim(1)?;
    let idx = Tensor::from_vec(kept.clone(), kept.len(), pred_masks.device())?;
    // [1,Q,288,288] → [Q,288²]; select kept rows → [kept,288²]; single host readback.
    let masks_v = pred_masks
        .reshape((q, per))?
        .index_select(&idx, 0)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let mut dets = Vec::with_capacity(kept.len());
    for (row, &qi) in kept.iter().enumerate() {
        dets.push(Detection {
            mask: masks_v[row * per..(row + 1) * per].to_vec(),
            score: probs[qi as usize],
            prompt_id: 0,
        });
    }
    Ok(dets)
}

/// Bail with the typed [`CandleError::Canceled`] when the caller's cooperative cancel flag has
/// tripped — [`Sam3VideoModel::propagate`] checks it before each frame (the gen-core video
/// per-step cancel contract, sc-8972; mirrors `mlx-gen-sam3`'s `video.rs`).
fn check_canceled(cancel: Option<&CancelFlag>) -> Result<()> {
    if cancel.is_some_and(CancelFlag::is_cancelled) {
        return Err(CandleError::Canceled);
    }
    Ok(())
}

/// Flatten the memory encoder's NHWC `[1,72,72,64]` output to seq-first `[5184,1,64]`. The reference
/// stores `maskmem_features` as **bfloat16** (`bf16 = true`) but `maskmem_pos_enc` stays f32, so the
/// two round-trip differently.
fn seq_first(a: &Tensor, bf16: bool) -> Result<Tensor> {
    let (_, g, _, c) = a.dims4()?;
    let flat = a.reshape((g * g, 1, c))?;
    if bf16 {
        Ok(flat.to_dtype(DType::BF16)?.to_dtype(DType::F32)?)
    } else {
        Ok(flat)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Device;
    use std::path::Path;

    fn dummy_fm() -> FrameMem {
        let t = |v: f32| Tensor::from_vec(vec![v], (1, 1), &Device::Cpu).unwrap();
        FrameMem {
            maskmem_features: Some(t(0.0)),
            maskmem_pos_enc: Some(t(0.0)),
            object_pointer: t(0.0),
        }
    }

    /// sc-8972: the per-frame propagate cancel check surfaces the **typed**
    /// [`CandleError::Canceled`] (never a stringified `Msg` — the worker keys off the variant to
    /// map a user cancel), and an absent/untripped flag is a no-op.
    #[test]
    fn check_canceled_surfaces_typed_canceled_only_on_trip() {
        assert!(check_canceled(None).is_ok(), "no flag → no-op");
        let flag = CancelFlag::new();
        assert!(check_canceled(Some(&flag)).is_ok(), "untripped → no-op");
        flag.cancel();
        assert!(
            matches!(check_canceled(Some(&flag)), Err(CandleError::Canceled)),
            "tripped → typed Canceled"
        );
    }

    /// F-015: `evict_stale_bank` drops exactly the `non_cond` entries no future `gather_memory` can
    /// read (older than the pointer window) and nulls the heavy tensors of those past the (tighter)
    /// spatial window, while keeping the rest and never touching `cond`. Derived from frame_idx=20.
    #[test]
    fn evict_stale_bank_prunes_only_unreadable_entries() {
        let mut bank = ObjectBank::default();
        for k in 0..=20 {
            bank.non_cond.insert(k, dummy_fm());
        }
        bank.cond.insert(3, dummy_fm()); // cond must be left intact

        let frame_idx = 20;
        let heavy_keep = frame_idx + 1 - (NUM_MASKMEM - 1);
        let ptr_keep = frame_idx + 1 - (MAX_OBJ_PTRS - 1);
        assert_eq!(
            (heavy_keep, ptr_keep),
            (15, 6),
            "window floors for frame 20"
        );
        evict_stale_bank(&mut bank, heavy_keep, ptr_keep);

        // keys < ptr_keep (0..=5): pointer window passed → entry dropped entirely.
        for k in 0..ptr_keep {
            assert!(!bank.non_cond.contains_key(&k), "key {k} must be evicted");
        }
        // keys in [ptr_keep, heavy_keep) (6..=14): kept, but heavy tensors nulled (object_pointer stays).
        for k in ptr_keep..heavy_keep {
            let m = bank
                .non_cond
                .get(&k)
                .unwrap_or_else(|| panic!("key {k} must be kept"));
            assert!(
                m.maskmem_features.is_none() && m.maskmem_pos_enc.is_none(),
                "heavy tensors must be nulled at key {k}"
            );
        }
        // keys >= heavy_keep (15..=20): still spatially readable → heavy tensors retained.
        for k in heavy_keep..=20 {
            let m = bank.non_cond.get(&k).unwrap();
            assert!(
                m.maskmem_features.is_some() && m.maskmem_pos_enc.is_some(),
                "heavy tensors must be kept at key {k}"
            );
        }
        // cond is never touched by `evict_stale_bank` (its own heavy bound is `evict_stale_cond_heavy`).
        assert!(bank.cond.contains_key(&3), "cond must be left intact");
        let c = bank.cond.get(&3).unwrap();
        assert!(
            c.maskmem_features.is_some() && c.maskmem_pos_enc.is_some(),
            "evict_stale_bank must not touch cond heavy tensors"
        );
    }

    /// F-015: the fix must not change tracking output — it may only drop bank state that no future
    /// `gather_memory` can read. This is the "short-video unchanged" guarantee: for a clip shorter than
    /// the pointer window (`MAX_OBJ_PTRS`) no `non_cond` **entry** is ever dropped (so the pointer loop,
    /// which reads `object_pointer` across the whole window, is byte-identical), and at every frame every
    /// bank entry a future gather could still read keeps exactly the fields that read needs. Heavy-tensor
    /// nulling outside the tighter spatial window (`NUM_MASKMEM`) is invisible to `gather_memory`, so it
    /// is not a behavior change even though those tensors are freed. We assert against the exact read-set
    /// `gather_memory` computes, so any regression that touched a live entry would trip.
    #[test]
    fn short_video_eviction_preserves_every_readable_entry() {
        let num_frames = 12i32; // < MAX_OBJ_PTRS (16): a "short" clip
        let mut bank = ObjectBank::default();
        bank.cond.insert(0, dummy_fm()); // an initial seed conditioning frame
        for frame_idx in 0..num_frames {
            bank.non_cond.insert(frame_idx, dummy_fm());

            // Eviction runs AFTER this frame's reads, so it may only affect frame_idx+1 onward. Compute
            // exactly what the NEXT gather (frame_idx+1) reads from `non_cond` and assert eviction leaves
            // all of it intact:
            //  - spatial: heavy tensors at offsets [1, NUM_MASKMEM-1] → keys (frame_idx+1)-1 .. -6.
            //  - pointers: object_pointer at offsets [1, max_optr-1] within the clip.
            let next = frame_idx + 1;
            let max_optr = num_frames.min(MAX_OBJ_PTRS);
            let mut spatial_keys: std::collections::BTreeSet<i32> =
                std::collections::BTreeSet::new();
            for rel in 1..NUM_MASKMEM {
                let k = next - rel;
                if (0..num_frames).contains(&k) {
                    spatial_keys.insert(k);
                }
            }
            let mut ptr_keys: std::collections::BTreeSet<i32> = std::collections::BTreeSet::new();
            for t_diff in 1..max_optr {
                let r = next - t_diff;
                if (0..num_frames).contains(&r) {
                    ptr_keys.insert(r);
                }
            }

            let heavy_keep = frame_idx + 1 - (NUM_MASKMEM - 1);
            let ptr_keep = frame_idx + 1 - (MAX_OBJ_PTRS - 1);
            evict_stale_bank(&mut bank, heavy_keep, ptr_keep);
            evict_stale_cond_heavy(&mut bank, heavy_keep);

            for k in &spatial_keys {
                let m = bank.non_cond.get(k).unwrap_or_else(|| {
                    panic!("frame {frame_idx}: next spatial read of non_cond key {k} evicted")
                });
                assert!(
                    m.maskmem_features.is_some() && m.maskmem_pos_enc.is_some(),
                    "frame {frame_idx}: next spatial read of non_cond key {k} lost heavy tensors"
                );
            }
            for k in &ptr_keys {
                assert!(
                    bank.non_cond.contains_key(k),
                    "frame {frame_idx}: next pointer read of non_cond key {k} was evicted"
                );
            }
        }
        // For a clip shorter than the pointer window every seeded entry survives to the end — the
        // pointer loop is byte-identical to the pre-fix behaviour.
        assert_eq!(
            bank.non_cond.len(),
            num_frames as usize,
            "no non_cond entry may be dropped for a clip shorter than MAX_OBJ_PTRS"
        );
        // The cond seed's key (and its object_pointer) is retained throughout.
        assert!(
            bank.cond.contains_key(&0),
            "cond seed entry must be retained"
        );
    }

    /// F-015: `evict_stale_cond_heavy` is a **strict no-op** for spatial memory — it only nulls cond
    /// heavy tensors that no future `gather_memory` can read. Simulate a long clip (cond seeds +
    /// reconditioning cadence), and at every frame assert the spatial read-set (the real
    /// `select_closest_cond_frames` selection ++ the unselected-cond fallback window, exactly as
    /// `gather_memory` reads) still has its heavy tensors present after all prior-frame evictions. Also
    /// proves the bound actually bites (nulls something) so the bank stops growing without a ceiling.
    #[test]
    fn evict_stale_cond_heavy_never_nulls_a_readable_frame() {
        let mut bank = ObjectBank::default();
        // cond frames: an initial seed at 0, a second seed at 5, then reconditioning every
        // RECONDITION_EVERY up to 160 — the long-clip pattern the cond leak comes from.
        let mut cond_frames: Vec<i32> = vec![0, 5];
        let mut f = RECONDITION_EVERY;
        while f <= 160 {
            cond_frames.push(f);
            f += RECONDITION_EVERY;
        }

        let n_frames = 170;
        let mut nulled_total = 0usize;
        for frame_idx in 0..n_frames {
            if cond_frames.contains(&frame_idx) {
                bank.cond.insert(frame_idx, dummy_fm());
            }
            // Read-set exactly as `gather_memory`: selected cond (offset 0) ++ unselected cond in the
            // `[frame_idx-(NUM_MASKMEM-1), frame_idx-1]` fallback window (no non_cond in this fixture).
            let (selected, unselected) =
                select_closest_cond_frames(frame_idx, &bank.cond, MAX_COND_FRAME_NUM);
            let mut read_keys: std::collections::BTreeSet<i32> = selected.into_iter().collect();
            for rel in 1..NUM_MASKMEM {
                let prev = frame_idx - rel;
                if unselected.contains(&prev) {
                    read_keys.insert(prev);
                }
            }
            for k in &read_keys {
                let m = bank.cond.get(k).unwrap();
                assert!(
                    m.maskmem_features.is_some() && m.maskmem_pos_enc.is_some(),
                    "frame {frame_idx}: cond key {k} is in the spatial read-set but its heavy \
                     tensors were nulled by a prior eviction"
                );
            }
            // Evict after the frame's reads, mirroring `evict_stale_memory`.
            let heavy_keep = frame_idx + 1 - (NUM_MASKMEM - 1);
            evict_stale_cond_heavy(&mut bank, heavy_keep);
            nulled_total = bank
                .cond
                .values()
                .filter(|m| m.maskmem_features.is_none())
                .count();
        }
        // The eviction must actually bite (not a vacuous pass): old cond frames' heavy tensors are gone
        // while every entry (and its object_pointer) is retained.
        assert!(
            nulled_total > 0,
            "expected some cond heavy tensors to be nulled over a 170-frame clip"
        );
        assert_eq!(
            bank.cond.len(),
            cond_frames.len(),
            "cond entries (object_pointer) must be retained, only heavy tensors nulled"
        );
        // The newest MAX_COND_FRAME_NUM cond frames are always protected.
        for &k in cond_frames.iter().rev().take(MAX_COND_FRAME_NUM as usize) {
            let m = bank.cond.get(&k).unwrap();
            assert!(
                m.maskmem_features.is_some(),
                "newest cond frame {k} must keep its heavy tensors (always selectable)"
            );
        }
    }

    /// Naive reference: read the WHOLE `[1,Q,288,288]` mask tensor to host, then filter — exactly
    /// the pre-F-014 behaviour. `select_detections` must match this bit-for-bit.
    fn select_detections_full_readback(pred_masks: &Tensor, probs: &[f32]) -> Vec<Detection> {
        let per = LOW_RES * LOW_RES;
        let q = pred_masks.dim(1).unwrap();
        let masks_v = pred_masks
            .reshape((q, per))
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let mut dets = Vec::new();
        for (qi, &p) in probs.iter().enumerate() {
            if p <= SCORE_THRESH_DET {
                continue;
            }
            dets.push(Detection {
                mask: masks_v[qi * per..(qi + 1) * per].to_vec(),
                score: p,
                prompt_id: 0,
            });
        }
        dets
    }

    /// F-014: the on-device `index_select` selection reads back only the kept rows, yet yields the
    /// exact same detections (masks + scores + order) as the full-readback-then-filter path.
    #[test]
    fn on_device_selection_matches_full_readback() {
        let per = LOW_RES * LOW_RES;
        // A handful of queries: some below, some above `SCORE_THRESH_DET`, boundary at exactly the
        // threshold (kept iff strictly greater), interleaved so ordering is exercised.
        let probs = [0.10f32, 0.90, SCORE_THRESH_DET, 0.51, 0.49, 0.999, 0.5001];
        let q = probs.len();
        // Distinct per-row mask values so a wrong row would be caught (row r ↦ value r + 0.5·col%7).
        let mut data = Vec::with_capacity(q * per);
        for r in 0..q {
            for c in 0..per {
                data.push(r as f32 + 0.5 * ((c % 7) as f32));
            }
        }
        let pred_masks = Tensor::from_vec(data, (1, q, LOW_RES, LOW_RES), &Device::Cpu).unwrap();

        let got = select_detections(&pred_masks, &probs).unwrap();
        let want = select_detections_full_readback(&pred_masks, &probs);

        assert_eq!(got.len(), want.len(), "kept count differs");
        // exactly the strictly-greater-than-threshold queries: 0.90, 0.51, 0.999, 0.5001 → 4.
        assert_eq!(got.len(), 4);
        for (g, w) in got.iter().zip(&want) {
            assert_eq!(g.score, w.score, "score mismatch");
            assert_eq!(g.prompt_id, w.prompt_id);
            assert_eq!(g.mask, w.mask, "mask bytes differ for score {}", g.score);
        }
    }

    /// No query passes the threshold → no host readback of masks, empty result.
    #[test]
    fn on_device_selection_empty_when_none_pass() {
        let per = LOW_RES * LOW_RES;
        let probs = [0.1f32, 0.2, SCORE_THRESH_DET, 0.0];
        let q = probs.len();
        let pred_masks = Tensor::from_vec(
            vec![1.0f32; q * per],
            (1, q, LOW_RES, LOW_RES),
            &Device::Cpu,
        )
        .unwrap();
        assert!(select_detections(&pred_masks, &probs).unwrap().is_empty());
    }

    /// F-028: the detector segmenter and the tracker must share **one** PE backbone instance (rather
    /// than each holding its own ~445M-param copy). Checks `Arc` pointer-identity of the two
    /// backbones — the cheapest, most direct proof the weights are not duplicated. Weights-gated (no
    /// torch fixture needed — only the real `facebook/sam3` weights); `#[ignore]` until staged (sc-6248).
    #[test]
    #[ignore = "needs staged facebook/sam3 weights (SAM3_WEIGHTS) — sc-6248"]
    fn backbone_is_shared_not_duplicated() {
        use candle_gen::default_device;

        let weights_path =
            std::env::var("SAM3_WEIGHTS").expect("set SAM3_WEIGHTS to the facebook/sam3 snapshot");
        let device = default_device().expect("default device");
        let wp = Path::new(&weights_path);
        let w = if wp.is_dir() {
            crate::Weights::from_dir(wp, &device)
        } else {
            crate::Weights::from_file(wp, &device)
        }
        .expect("load sam3 weights");

        let model = Sam3VideoModel::from_weights(&w).expect("build video model");
        assert!(
            Arc::ptr_eq(
                &model.segmenter().vision_backbone_arc(),
                &model.tracker().backbone_arc(),
            ),
            "segmenter and tracker must point at one shared PE backbone",
        );
    }

    /// F-093: a freshly-constructed session is empty and, crucially, `max_obj_id == -1` (so the first
    /// assigned object id is `max_obj_id + 1 == 0`, matching `from_weights`). Guards the hand-written
    /// `Default` against a silent `#[derive(Default)]` regression that would default `max_obj_id` to 0.
    #[test]
    fn session_default_is_empty_and_max_obj_id_is_minus_one() {
        let s = SessionState::default();
        assert_eq!(s.max_obj_id, -1, "first id must be max_obj_id+1 == 0");
        assert_eq!(s.num_frames, 0);
        assert!(s.obj_ids.is_empty());
        assert!(s.banks.is_empty());
        assert!(s.obj_prompt.is_empty());
        assert!(s.first_frame.is_empty());
        assert!(s.unmatched_frames.is_empty());
        assert!(s.overlap_pairs.is_empty());
        assert!(s.removed.is_empty());
        assert!(s.last_occluded.is_empty());
    }

    /// F-093 (unit): `reset_session` restores **every** per-video field, so no clip inherits the
    /// previous clip's banks, hotstart bookkeeping, removed-id bans, or id/frame counters. Populating a
    /// `SessionState` with dirty values from a hypothetical prior clip and then applying the exact reset
    /// (`= SessionState::default()`) must reproduce a pristine session — the whole leak fix in one
    /// value assignment, verified without a loaded model. (The full two-`propagate`-calls end-to-end
    /// proof is `propagate_twice_matches_fresh_model`, weights-gated below.)
    #[test]
    fn reset_session_clears_all_prior_clip_state() {
        // A session dirtied as if by a prior clip: live objects, banks, hotstart maps, bans, counters.
        let mut s = SessionState::default();
        s.obj_ids.push(7);
        s.banks.push(ObjectBank::default());
        s.obj_prompt.push(3);
        s.max_obj_id = 42;
        s.num_frames = 99;
        s.first_frame.insert(7, 2);
        s.unmatched_frames.insert(7, vec![1, 2]);
        s.overlap_pairs.insert((1, 7), vec![3]);
        s.removed.insert(9);
        s.last_occluded.insert(7, 4);
        // Sanity: it really is dirty before the reset.
        assert!(!s.obj_ids.is_empty() && s.max_obj_id != -1 && !s.removed.is_empty());

        // This is exactly what `reset_session` does.
        s = SessionState::default();

        assert_eq!(
            s.max_obj_id, -1,
            "max_obj_id must reset (stale id would ban 0..=42)"
        );
        assert_eq!(s.num_frames, 0);
        assert!(s.obj_ids.is_empty(), "objects must not carry over");
        assert!(
            s.banks.is_empty(),
            "banks (stale cond entries) must not carry over"
        );
        assert!(s.obj_prompt.is_empty());
        assert!(s.first_frame.is_empty());
        assert!(s.unmatched_frames.is_empty());
        assert!(s.overlap_pairs.is_empty());
        assert!(s.removed.is_empty(), "removed-id bans must not carry over");
        assert!(s.last_occluded.is_empty());
    }

    /// F-093 (end-to-end): running clip A then clip B on ONE cached model must yield the identical B
    /// result to a freshly-constructed model that only ran B — i.e. `propagate` is independent of any
    /// prior call. Before the fix the reused instance carried clip A's banks/hotstart/removed-id state
    /// into B and silently corrupted tracking. Weights-gated (needs `facebook/sam3`); `#[ignore]` until
    /// staged (sc-6248). Frame content is arbitrary — the invariant is *first-clip independence*.
    #[test]
    #[ignore = "needs staged facebook/sam3 weights (SAM3_WEIGHTS) — sc-6248"]
    fn propagate_twice_matches_fresh_model() {
        use candle_gen::candle_core::DType;
        use candle_gen::default_device;

        let weights_path =
            std::env::var("SAM3_WEIGHTS").expect("set SAM3_WEIGHTS to the facebook/sam3 snapshot");
        let device = default_device().expect("default device");
        let wp = Path::new(&weights_path);
        let w = if wp.is_dir() {
            crate::Weights::from_dir(wp, &device)
        } else {
            crate::Weights::from_file(wp, &device)
        }
        .expect("load sam3 weights");

        // Two distinct short clips (different length + content) so A meaningfully dirties the session.
        let frame = |seed: f32, n: usize| -> Vec<Tensor> {
            (0..n)
                .map(|f| {
                    Tensor::full(seed + f as f32, (1, 3, 1008, 1008), &device)
                        .unwrap()
                        .to_dtype(DType::F32)
                        .unwrap()
                })
                .collect()
        };
        let clip_a = frame(0.10, 5);
        let clip_b = frame(0.70, 3);
        let input_ids = Tensor::zeros((1, 32), DType::U32, &device).unwrap();
        let text_mask = vec![1i32; 32];

        // Reused model: A first (dirties the session), then B.
        let mut reused = Sam3VideoModel::from_weights(&w).expect("build reused model");
        let _ = reused
            .propagate(&clip_a, &input_ids, &text_mask, None, None)
            .expect("clip A");
        let b_after_a = reused
            .propagate(&clip_b, &input_ids, &text_mask, None, None)
            .expect("clip B on reused model");

        // Fresh model that only ever saw B.
        let mut fresh = Sam3VideoModel::from_weights(&w).expect("build fresh model");
        let b_fresh = fresh
            .propagate(&clip_b, &input_ids, &text_mask, None, None)
            .expect("clip B on fresh model");

        assert_eq!(b_after_a.len(), b_fresh.len(), "frame count must match");
        for (i, (r, f)) in b_after_a.iter().zip(&b_fresh).enumerate() {
            assert_eq!(
                r.obj_ids, f.obj_ids,
                "frame {i}: obj ids differ — clip A leaked into clip B",
            );
            assert_eq!(
                r.masks, f.masks,
                "frame {i}: masks differ — clip A leaked into clip B",
            );
        }
    }
}
