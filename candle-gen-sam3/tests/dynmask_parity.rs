//! SAM3 dynamic-multimask-via-stability parity (sc-6245): run the candle `Sam3Tracker` single-frame
//! decode with `multimask=false` (the no-prompt video-frame policy → `dynamic_multimask_via_stability`)
//! against the SAME torch oracle the MLX twin uses
//! (`mlx-gen/scripts/spikes/sam3_oracle/dump_dynmask_fixture.py`). `#[ignore]` until weights + fixture
//! are staged (sc-6248). Run:
//!   SAM3_WEIGHTS=<snapshot> SAM3_DYNMASK_FIXTURE=<dynmask_fixture.safetensors> \
//!     cargo test -p candle-gen-sam3 --release --features cuda --test dynmask_parity -- --ignored --nocapture

use std::path::Path;

use candle_gen::candle_core::{Device, Tensor};
use candle_gen::default_device;
use candle_gen_sam3::{Sam3Tracker, Weights};

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

fn scalar(t: &Tensor) -> f32 {
    t.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0]
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
#[ignore = "needs staged facebook/sam3 weights (SAM3_WEIGHTS) + SAM3_DYNMASK_FIXTURE — sc-6248"]
fn dynamic_multimask_matches_oracle() {
    let weights_path =
        std::env::var("SAM3_WEIGHTS").expect("set SAM3_WEIGHTS to the facebook/sam3 snapshot");
    let fixture_path =
        std::env::var("SAM3_DYNMASK_FIXTURE").expect("set SAM3_DYNMASK_FIXTURE to the oracle dump");

    let device = default_device().expect("default device");
    let w = load_weights(&weights_path, &device);
    let fx = Weights::from_file(&fixture_path, &device).expect("load dynmask fixture");
    let tracker = Sam3Tracker::from_weights(&w).expect("build tracker");

    let pixel_values = fx.require("pixel_values").unwrap();
    let box_v = fx
        .require("box_1008")
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let box_xyxy = [box_v[0], box_v[1], box_v[2], box_v[3]];

    let (emb, high_res) = tracker.encode_frame(&pixel_values).expect("encode_frame");
    let out = tracker
        .segment_encoded_multimask(&emb, &high_res, box_xyxy, false)
        .expect("segment multimask=false");

    let want_mask = fx.require("dyn_mask").unwrap(); // [mg, mg]
    let want_iou = scalar(&fx.require("dyn_iou").unwrap());
    assert_eq!(out.low_res.dims(), want_mask.dims(), "dyn mask shape");
    let c = cosine(&out.low_res, &want_mask);
    println!(
        "dyn_mask: cosine={c:.7}  iou ours={:.4} oracle={want_iou:.4}",
        out.iou
    );
    // The dynamic-stability branch must pick the SAME candidate as the reference (else cosine tanks).
    assert!(c > 0.999, "dyn mask cosine {c}");
    assert!(
        (out.iou - want_iou).abs() / want_iou.abs().max(1.0) < 0.01,
        "dyn iou ours {} vs {want_iou}",
        out.iou
    );
}
