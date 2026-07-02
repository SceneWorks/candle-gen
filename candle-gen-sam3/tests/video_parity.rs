//! SAM3 end-to-end multi-object video PCS parity (sc-6245): run the candle `Sam3VideoModel::propagate`
//! on the captured clip and check per-frame `obj_id` sets + per-object 288² mask logits against the
//! SAME torch oracle the MLX twin uses (`mlx-gen/scripts/spikes/sam3_oracle/dump_video_fixture.py`,
//! a full `Sam3VideoModel` run). `#[ignore]` until weights + fixture are staged (sc-6248). Run:
//!   SAM3_WEIGHTS=<snapshot> SAM3_VIDEO_FIXTURE=<video_fixture.safetensors> \
//!     cargo test -p candle-gen-sam3 --release --features cuda --test video_parity -- --ignored --nocapture

use std::path::Path;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::ops::sigmoid;
use candle_gen::default_device;
use candle_gen_sam3::{Sam3ImageSegmenter, Sam3VideoModel, Weights};

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(&x, &y)| x * y).sum();
    let na: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

fn fx_i32(fx: &Weights, key: &str) -> Vec<i32> {
    fx.require(key)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .iter()
        .map(|&v| v as i32)
        .collect()
}

fn flat_f32(t: &Tensor) -> Vec<f32> {
    t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

fn load_weights(path: &str, device: &Device) -> Weights {
    let wp = Path::new(path);
    if wp.is_dir() {
        Weights::from_dir(wp, device)
    } else {
        Weights::from_file(wp, device)
    }
    .expect("load sam3 weights")
}

#[test]
#[ignore = "needs staged facebook/sam3 weights (SAM3_WEIGHTS) + SAM3_VIDEO_FIXTURE — sc-6248"]
fn video_pcs_matches_oracle() {
    let weights_path =
        std::env::var("SAM3_WEIGHTS").expect("set SAM3_WEIGHTS to the facebook/sam3 snapshot");
    let fixture_path =
        std::env::var("SAM3_VIDEO_FIXTURE").expect("set SAM3_VIDEO_FIXTURE to the oracle dump");

    let device = default_device().expect("default device");
    let w = load_weights(&weights_path, &device);
    let fx = Weights::from_file(&fixture_path, &device).expect("load video fixture");
    let mut model = Sam3VideoModel::from_weights(&w).expect("build video model");

    let num_frames = fx_i32(&fx, "num_frames")[0];
    let frames: Vec<Tensor> = (0..num_frames)
        .map(|f| fx.require(&format!("frame_{f}")).unwrap())
        .collect();
    let input_ids = fx.require("input_ids").unwrap();
    let text_mask = fx_i32(&fx, "attention_mask");

    let outputs = model
        .propagate(&frames, &input_ids, &text_mask, None, None)
        .expect("propagate");

    let per = 288 * 288;
    let mut worst = 1.0f32;
    for f in 0..num_frames {
        let want_ids = fx_i32(&fx, &format!("obj_ids_{f}"));
        let want_flat = flat_f32(&fx.require(&format!("masks_{f}")).unwrap()); // [num_obj,288,288]
        let out = &outputs[f as usize];
        println!("frame {f}: got obj_ids={:?} want={want_ids:?}", out.obj_ids);
        assert_eq!(out.obj_ids, want_ids, "frame {f} obj_id mismatch");
        let mut frame_worst = 1.0f32;
        for (oi, mask) in out.masks.iter().enumerate() {
            let want = &want_flat[oi * per..(oi + 1) * per];
            let c = cosine(mask, want);
            frame_worst = frame_worst.min(c);
            worst = worst.min(c);
        }
        println!("frame {f}: worst obj cosine={frame_worst:.7}");
    }
    println!("worst per-object mask cosine = {worst:.7}");
    assert!(worst > 0.999, "video PCS mask cosine {worst}");
}

// --- F-014 (sc-8994): on-device mask selection ---------------------------------------------------
//
// `run_detection` no longer reads back the whole `[1,200,288,288]` `pred_masks` (~66 MB) every
// frame; it `index_select`s the kept query rows on-device and reads back only those. This test runs
// the REAL detector over real weights on the GPU to produce an actual `pred_masks` tensor, then
// proves the on-device selection returns bit-identical mask logits + query order to the pre-fix
// full-readback-then-filter path — over a sweep of thresholds so a non-trivial (and varying)
// kept-set is always exercised. Weights-only (no video fixture needed):
//   SAM3_WEIGHTS=<snapshot> \
//     cargo test -p candle-gen-sam3 --release --features cuda --test video_parity \
//       on_device_mask_selection_matches_full_readback_gpu -- --ignored --nocapture

const LOW_RES: usize = 288;

/// Pre-fix path: read the WHOLE `[1,Q,288,288]` tensor to host, then keep `probs > thresh`.
fn select_full_readback(pred_masks: &Tensor, probs: &[f32], thresh: f32) -> Vec<(usize, Vec<f32>)> {
    let per = LOW_RES * LOW_RES;
    let q = pred_masks.dim(1).unwrap();
    let all = pred_masks
        .reshape((q, per))
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    probs
        .iter()
        .enumerate()
        .filter(|&(_, &p)| p > thresh)
        .map(|(qi, _)| (qi, all[qi * per..(qi + 1) * per].to_vec()))
        .collect()
}

/// Post-fix path (mirrors `video::select_detections`): `index_select` the kept rows on-device,
/// read back only those.
fn select_on_device(pred_masks: &Tensor, probs: &[f32], thresh: f32) -> Vec<(usize, Vec<f32>)> {
    let per = LOW_RES * LOW_RES;
    let q = pred_masks.dim(1).unwrap();
    let kept: Vec<u32> = probs
        .iter()
        .enumerate()
        .filter(|&(_, &p)| p > thresh)
        .map(|(qi, _)| qi as u32)
        .collect();
    if kept.is_empty() {
        return Vec::new();
    }
    let idx = Tensor::from_vec(kept.clone(), kept.len(), pred_masks.device()).unwrap();
    let rows = pred_masks
        .reshape((q, per))
        .unwrap()
        .index_select(&idx, 0)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    kept.iter()
        .enumerate()
        .map(|(row, &qi)| (qi as usize, rows[row * per..(row + 1) * per].to_vec()))
        .collect()
}

#[test]
#[ignore = "needs staged facebook/sam3 weights (SAM3_WEIGHTS) — sc-8994"]
fn on_device_mask_selection_matches_full_readback_gpu() {
    let weights_path =
        std::env::var("SAM3_WEIGHTS").expect("set SAM3_WEIGHTS to the facebook/sam3 snapshot");
    let device = default_device().expect("default device");
    let w = load_weights(&weights_path, &device);
    let seg = Sam3ImageSegmenter::from_weights(&w).expect("build segmenter");

    // Real concept prompt ("person") from the video oracle manifest; attention mask keeps the 3
    // real tokens (BOS, "person", EOS) and pads the rest.
    let ids: Vec<i64> = {
        let mut v = vec![49406i64, 2533, 49407];
        v.resize(32, 49407);
        v
    };
    let input_ids = Tensor::from_vec(ids, (1, 32), &device).unwrap();
    let mut text_mask = vec![0i32; 32];
    text_mask[0] = 1;
    text_mask[1] = 1;
    text_mask[2] = 1;

    // A deterministic pixel tensor (a smooth spatial gradient per channel) — produces real, varied
    // backbone activations and therefore a real `[1,200,288,288]` `pred_masks` on the device. The
    // equivalence we assert (on-device select == full-readback select) is independent of whether
    // the detections match any oracle.
    let (h, w_px) = (1008usize, 1008usize);
    let mut px = Vec::with_capacity(3 * h * w_px);
    for c in 0..3usize {
        for y in 0..h {
            for x in 0..w_px {
                let v = ((x as f32 / w_px as f32) - 0.5) * 2.0
                    + ((y as f32 / h as f32) - 0.5) * 2.0
                    + (c as f32) * 0.1;
                px.push(v);
            }
        }
    }
    let pixels = Tensor::from_vec(px, (1, 3, h, w_px), &device).unwrap();

    let out = seg
        .forward(&pixels, &input_ids, &text_mask)
        .expect("segmenter forward");
    let q = out.pred_masks.dim(1).unwrap();
    println!("pred_masks shape = {:?} (Q={q})", out.pred_masks.dims());

    // Reproduce `run_detection`'s probs exactly: σ(pred_logits) · σ(presence).
    let presence = sigmoid(&out.presence_logits)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()[0];
    let probs: Vec<f32> = sigmoid(&out.pred_logits)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .iter()
        .map(|&s| s * presence)
        .collect();
    let (pmin, pmax) = probs
        .iter()
        .fold((f32::MAX, f32::MIN), |(lo, hi), &p| (lo.min(p), hi.max(p)));
    println!("probs: Q={} range=[{pmin:.4},{pmax:.4}]", probs.len());

    // Sweep thresholds so the kept-set is non-trivial and varies (including the real 0.5 gate).
    let mut any_nonempty = false;
    for &thresh in &[0.0f32, 0.5, (pmin + pmax) * 0.5, pmax - 1e-4] {
        let want = select_full_readback(&out.pred_masks, &probs, thresh);
        let got = select_on_device(&out.pred_masks, &probs, thresh);
        println!(
            "thresh={thresh:.4}: kept={} (want={})",
            got.len(),
            want.len()
        );
        assert_eq!(
            got.len(),
            want.len(),
            "kept count differs at thresh {thresh}"
        );
        for (g, wv) in got.iter().zip(&want) {
            assert_eq!(g.0, wv.0, "query index/order differs at thresh {thresh}");
            assert_eq!(
                g.1, wv.1,
                "mask logits differ for query {} at thresh {thresh}",
                g.0
            );
        }
        if !got.is_empty() {
            any_nonempty = true;
        }
    }
    assert!(
        any_nonempty,
        "no threshold produced a non-empty kept-set — index_select readback path not exercised"
    );
    println!("F-014 on-device selection is bit-identical to full-readback across all thresholds");
}
