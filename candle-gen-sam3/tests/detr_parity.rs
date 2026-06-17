//! SAM3 DETR-detector parity (sc-6242): load the real `facebook/sam3` weights, run the full DETR
//! detector (encoder, decoder, presence, scoring) on the reference's 72² FPN feature plus the text
//! features, and check pred_logits / pred_boxes / presence against the SAME torch oracle fixture the
//! MLX port uses (`mlx-gen/scripts/spikes/sam3_oracle/dump_detr_fixture.py`). The candle output is
//! the reimplementation-of-record against `mlx-gen-sam3` (logits/boxes cosine > 0.999, the MLX bar).
//! Stays `#[ignore]` until `facebook/sam3` (gated) + the fixture are staged on the box (sc-6248).
//!
//! Run (CUDA build on the Blackwell box):
//!   SAM3_WEIGHTS=<facebook/sam3 snapshot dir OR model.safetensors> \
//!   SAM3_DETR_FIXTURE=<.../sam3_oracle/detr_fixture.safetensors> \
//!     cargo test -p candle-gen-sam3 --release --features cuda --test detr_parity -- --ignored --nocapture

use std::path::Path;

use candle_gen::candle_core::Tensor;
use candle_gen::candle_nn::ops::sigmoid;
use candle_gen::default_device;
use candle_gen_sam3::{Sam3Detector, Sam3DetrConfig, Weights};

fn cosine(a: &Tensor, b: &Tensor) -> f32 {
    let a = a.flatten_all().unwrap();
    let b = b.flatten_all().unwrap();
    let dot = (&a * &b)
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    let na = (&a * &a)
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
        .sqrt();
    let nb = (&b * &b)
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
        .sqrt();
    dot / (na * nb)
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
    let d = (a.flatten_all().unwrap() - b.flatten_all().unwrap()).unwrap();
    d.abs().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap()
}

/// instances = #{ sigmoid(logits)·sigmoid(presence) > 0.5 }.
fn instances(logits: &Tensor, presence: &Tensor) -> usize {
    let s = sigmoid(logits)
        .unwrap()
        .broadcast_mul(&sigmoid(presence).unwrap())
        .unwrap();
    s.flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .iter()
        .filter(|&&x| x > 0.5)
        .count()
}

#[test]
#[ignore = "needs staged facebook/sam3 weights (SAM3_WEIGHTS) + SAM3_DETR_FIXTURE — sc-6248"]
fn detr_detector_matches_oracle() {
    let weights_path =
        std::env::var("SAM3_WEIGHTS").expect("set SAM3_WEIGHTS to the facebook/sam3 snapshot");
    let fixture_path =
        std::env::var("SAM3_DETR_FIXTURE").expect("set SAM3_DETR_FIXTURE to the oracle dump");

    let device = default_device().expect("default device");

    let wp = Path::new(&weights_path);
    let w = if wp.is_dir() {
        Weights::from_dir(wp, &device)
    } else {
        Weights::from_file(wp, &device)
    }
    .expect("load sam3 weights");
    let fx = Weights::from_file(&fixture_path, &device).expect("load detr fixture");

    let det = Sam3Detector::from_weights(&w, "detector_model", &Sam3DetrConfig::sam3())
        .expect("build detector");

    // fpn_72 NCHW [1,256,72,72] → NHWC [1,72,72,256]
    let vision = fx
        .require("fpn_72")
        .unwrap()
        .permute([0, 2, 3, 1])
        .unwrap()
        .contiguous()
        .unwrap();
    let text = fx.require("text_features").unwrap();
    let mask: Vec<i32> = fx
        .require("attention_mask")
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .iter()
        .map(|&m| m as i32)
        .collect();

    let out = det
        .forward(&vision, &text, &mask)
        .expect("detector forward");

    let want_logits = fx.require("pred_logits").unwrap();
    let want_boxes = fx.require("pred_boxes").unwrap();
    let want_presence = fx.require("presence_logits").unwrap();

    let logits_cos = cosine(&out.pred_logits, &want_logits);
    let boxes_cos = cosine(&out.pred_boxes, &want_boxes);
    let presence_diff = max_abs_diff(&out.presence_logits, &want_presence);
    let logits_maxabs = max_abs_diff(&out.pred_logits, &want_logits);

    let got_n = instances(&out.pred_logits, &out.presence_logits);
    let want_n = instances(&want_logits, &want_presence);

    println!(
        "pred_logits: cosine={logits_cos:.6} max_abs={logits_maxabs:.4} | pred_boxes: cosine={boxes_cos:.6} | presence: |Δ|={presence_diff:.4} | instances got={got_n} want={want_n}"
    );

    assert!(
        logits_cos > 0.999,
        "pred_logits cosine {logits_cos:.6} < 0.999"
    );
    assert!(
        boxes_cos > 0.999,
        "pred_boxes cosine {boxes_cos:.6} < 0.999"
    );
    // presence is a single clamped (±10) logit through 6 decoder layers + the presence MLP; a ~1%
    // f32-matmul floor is expected (after sigmoid it's invisible).
    assert!(
        presence_diff < 0.15,
        "presence |Δ| {presence_diff:.4} too large"
    );
    assert_eq!(got_n, want_n, "instance count mismatch");
}
