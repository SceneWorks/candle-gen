//! SAM3 geometry/exemplar-encoder (box-prompt PVS path) parity (sc-6244). Two #[ignore] env-gated
//! gates against the SAME torch oracle fixtures the MLX port uses
//! (`mlx-gen/scripts/spikes/sam3_oracle/dump_geometry_fixture.py`):
//!
//! `geometry_encoder_matches_oracle` feeds the geometry encoder the exact inputs the reference module
//! received (boxes, labels, the 72² FPN feature + its sine pos embed) and checks its output prompt
//! tokens against the torch oracle — isolating the new `roi_align` + encoder (cosine > 0.9999).
//! `pvs_box_prompt_matches_oracle` runs the full box-prompted segmenter end-to-end and checks the
//! post-processed instance masks against the oracle (IoU > 0.95). Both stay `#[ignore]` until
//! `facebook/sam3` (gated) + the fixtures are staged on the box (sc-6248).
//!
//! Run (CUDA build on the Blackwell box):
//!   SAM3_WEIGHTS=<facebook/sam3 snapshot> \
//!   SAM3_GEOMETRY_FIXTURE=<.../geometry_fixture.safetensors> \
//!   SAM3_GEOMETRY_E2E_FIXTURE=<.../geometry_e2e_fixture.safetensors> \
//!     cargo test -p candle-gen-sam3 --release --features cuda --test geometry_parity -- --ignored --nocapture

use std::path::Path;

use candle_gen::candle_core::{DType, Tensor};
use candle_gen::default_device;
use candle_gen_sam3::{Sam3GeometryConfig, Sam3GeometryEncoder, Sam3ImageSegmenter, Weights};

fn sum_scalar(t: Tensor) -> f32 {
    t.sum_all().unwrap().to_scalar::<f32>().unwrap()
}

fn cosine(a: &Tensor, b: &Tensor) -> f32 {
    let a = a.flatten_all().unwrap();
    let b = b.flatten_all().unwrap();
    let dot = sum_scalar((&a * &b).unwrap());
    let na = sum_scalar((&a * &a).unwrap()).sqrt();
    let nb = sum_scalar((&b * &b).unwrap()).sqrt();
    dot / (na * nb)
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
    let d = (a.flatten_all().unwrap() - b.flatten_all().unwrap()).unwrap();
    d.abs().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap()
}

/// IoU of two binary `[h, w]` masks.
fn iou(a: &Tensor, b: &Tensor) -> f32 {
    let af = a.to_dtype(DType::F32).unwrap();
    let bf = b.to_dtype(DType::F32).unwrap();
    let inter = sum_scalar((&af * &bf).unwrap());
    let sa = sum_scalar(af);
    let sb = sum_scalar(bf);
    let union = sa + sb - inter;
    if union <= 0.0 {
        1.0
    } else {
        inter / union
    }
}

fn load_weights(path: &str, device: &candle_gen::candle_core::Device) -> Weights {
    let wp = Path::new(path);
    if wp.is_dir() {
        Weights::from_dir(wp, device)
    } else {
        Weights::from_file(wp, device)
    }
    .expect("load sam3 weights")
}

fn mask_i32(fx: &Weights, key: &str) -> Vec<i32> {
    fx.require(key)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .iter()
        .map(|&m| m as i32)
        .collect()
}

#[test]
#[ignore = "needs staged facebook/sam3 weights (SAM3_WEIGHTS) + SAM3_GEOMETRY_FIXTURE — sc-6248"]
fn geometry_encoder_matches_oracle() {
    let weights_path =
        std::env::var("SAM3_WEIGHTS").expect("set SAM3_WEIGHTS to the facebook/sam3 snapshot");
    let fixture_path = std::env::var("SAM3_GEOMETRY_FIXTURE")
        .expect("set SAM3_GEOMETRY_FIXTURE to the oracle dump");

    let device = default_device().expect("default device");
    let w = load_weights(&weights_path, &device);
    let fx = Weights::from_file(&fixture_path, &device).expect("load geometry fixture");

    let geo = Sam3GeometryEncoder::from_weights(
        &w,
        "detector_model.geometry_encoder",
        &Sam3GeometryConfig::sam3(),
    )
    .expect("build geometry encoder");

    let boxes = fx.require("box_embeddings").unwrap(); // [1,N,4] cxcywh
    let labels = mask_i32(&fx, "box_labels");
    // fpn_72 NCHW [1,256,72,72] → NHWC [1,72,72,256]
    let vision = fx
        .require("fpn_72")
        .unwrap()
        .permute([0, 2, 3, 1])
        .unwrap()
        .contiguous()
        .unwrap();
    // vision_pos NCHW [1,256,72,72] → flattened [1,H*W,256]
    let vision_pos = fx
        .require("vision_pos_72")
        .unwrap()
        .permute([0, 2, 3, 1])
        .unwrap()
        .reshape((1, 72 * 72, 256))
        .unwrap();

    let out = geo
        .forward(&boxes, &labels, &vision, &vision_pos)
        .expect("geometry forward");

    let want = fx.require("geo_output").unwrap();
    let cos = cosine(&out, &want);
    let maxabs = max_abs_diff(&out, &want);
    println!(
        "geometry prompt tokens: cosine={cos:.7} max_abs={maxabs:.5} shape={:?}",
        out.dims()
    );

    assert_eq!(out.dims(), want.dims(), "geometry output shape mismatch");
    assert!(cos > 0.9999, "geometry cosine {cos:.7} below 0.9999");
    assert!(maxabs < 1e-2, "geometry max_abs {maxabs:.5} above 1e-2");
}

#[test]
#[ignore = "needs staged facebook/sam3 weights (SAM3_WEIGHTS) + SAM3_GEOMETRY_E2E_FIXTURE — sc-6248"]
fn pvs_box_prompt_matches_oracle() {
    let weights_path =
        std::env::var("SAM3_WEIGHTS").expect("set SAM3_WEIGHTS to the facebook/sam3 snapshot");
    let fixture_path = std::env::var("SAM3_GEOMETRY_E2E_FIXTURE")
        .expect("set SAM3_GEOMETRY_E2E_FIXTURE to the oracle dump");

    let device = default_device().expect("default device");
    let w = load_weights(&weights_path, &device);
    let fx = Weights::from_file(&fixture_path, &device).expect("load geometry e2e fixture");

    let seg = Sam3ImageSegmenter::from_weights(&w).expect("build segmenter");

    let pixel_values = fx.require("pixel_values").unwrap();
    let input_ids = fx.require("input_ids").unwrap();
    let mask = mask_i32(&fx, "attention_mask");
    let boxes = fx.require("input_boxes").unwrap();
    let labels = mask_i32(&fx, "input_boxes_labels");

    let got = seg
        .segment_with_boxes(
            &pixel_values,
            &input_ids,
            &mask,
            &boxes,
            &labels,
            (1.0, 1.0),
            0.5,
            0.5,
        )
        .expect("segment_with_boxes");

    let want_masks = fx.require("instance_masks").unwrap(); // [m,288,288]
    let want_n = want_masks.dim(0).unwrap();
    println!("PVS instances: got {} want {}", got.len(), want_n);
    assert_eq!(got.len(), want_n, "PVS instance count mismatch");

    let mut worst_iou = 1.0f32;
    for (i, inst) in got.iter().enumerate() {
        let want = want_masks
            .narrow(0, i, 1)
            .unwrap()
            .reshape((288, 288))
            .unwrap();
        let m = iou(&inst.mask, &want);
        worst_iou = worst_iou.min(m);
        println!("  instance {i}: score={:.3} mask IoU={:.4}", inst.score, m);
    }
    assert!(
        worst_iou > 0.95,
        "worst PVS instance mask IoU {worst_iou:.4} below 0.95"
    );
}
