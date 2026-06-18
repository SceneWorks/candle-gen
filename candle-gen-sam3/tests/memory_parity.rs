//! SAM3 tracker memory-encoder parity (sc-6245): run the candle `Sam3Tracker` memory encoder
//! (`encode_new_memory` + `prepare_mask_for_mem`) against the SAME torch oracle the MLX twin uses
//! (`mlx-gen/scripts/spikes/sam3_oracle/dump_memory_fixture.py`). `#[ignore]` until weights + fixture
//! are staged (sc-6248). Run:
//!   SAM3_WEIGHTS=<snapshot> SAM3_MEMORY_FIXTURE=<memory_fixture.safetensors> \
//!     cargo test -p candle-gen-sam3 --release --features cuda --test memory_parity -- --ignored --nocapture

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

/// NCHW `[1,C,H,W]` fixture → NHWC `[1,H,W,C]` to match our layout.
fn to_nhwc(a: &Tensor) -> Tensor {
    a.permute([0, 2, 3, 1]).unwrap().contiguous().unwrap()
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
#[ignore = "needs staged facebook/sam3 weights (SAM3_WEIGHTS) + SAM3_MEMORY_FIXTURE — sc-6248"]
fn memory_encoder_matches_oracle() {
    let weights_path =
        std::env::var("SAM3_WEIGHTS").expect("set SAM3_WEIGHTS to the facebook/sam3 snapshot");
    let fixture_path =
        std::env::var("SAM3_MEMORY_FIXTURE").expect("set SAM3_MEMORY_FIXTURE to the oracle dump");

    let device = default_device().expect("default device");
    let w = load_weights(&weights_path, &device);
    let fx = Weights::from_file(&fixture_path, &device).expect("load memory fixture");
    let tracker = Sam3Tracker::from_weights(&w).expect("build tracker");

    let pix_feat = to_nhwc(&fx.require("pix_feat").unwrap()); // [1,72,72,256]
    let pred_high_res = fx.require("pred_masks_high_res").unwrap(); // [1,1,1008,1008]
    let obj_score = scalar(&fx.require("object_score_logits").unwrap());

    // --- Stage 1: mask prep (bilinear resize 1008→1152 + sigmoid/binarize + scale·20−10).
    for (key, is_pts) in [("mask_for_mem", false), ("mask_for_mem_bin", true)] {
        let got = tracker
            .prepare_mask_for_mem(&pred_high_res, is_pts)
            .unwrap(); // [1,1152,1152,1]
        let want = to_nhwc(&fx.require(key).unwrap());
        assert_eq!(got.dims(), want.dims(), "{key} shape");
        let c = cosine(&got, &want);
        println!("{key}: cosine={c:.7}");
        assert!(c > 0.9999, "{key} cosine {c}");
    }

    // --- Stage 2: full encoder + sine position encoding. obj_score>0 here so occlusion is inactive.
    let out = tracker
        .encode_new_memory(&pix_feat, &pred_high_res, obj_score, false)
        .expect("encode_new_memory");
    let want_feat = to_nhwc(&fx.require("maskmem_features_final").unwrap()); // [1,72,72,64]
    let want_pos = to_nhwc(&fx.require("maskmem_pos_enc").unwrap());
    assert_eq!(out.features.dims(), want_feat.dims(), "features shape");
    assert_eq!(out.pos.dims(), want_pos.dims(), "pos shape");
    let cf = cosine(&out.features, &want_feat);
    let cp = cosine(&out.pos, &want_pos);
    println!("maskmem_features: cosine={cf:.7}\nmaskmem_pos_enc: cosine={cp:.7}");
    assert!(cf > 0.999, "features cosine {cf}");
    assert!(cp > 0.99999, "pos_enc cosine {cp}");

    // --- Stage 3: occlusion add. Force object absent (score ≤ 0): features gain the occlusion
    // spatial embedding over the whole grid. Expected = oracle raw + occ (NHWC broadcast).
    let occ = w
        .require("tracker_model.occlusion_spatial_embedding_parameter")
        .unwrap()
        .reshape((1, 1, 1, 64))
        .unwrap();
    let raw = to_nhwc(&fx.require("maskmem_features_raw").unwrap());
    let expected_occ = raw.broadcast_add(&occ).unwrap();
    let out_occ = tracker
        .encode_new_memory(&pix_feat, &pred_high_res, -1.0, false)
        .expect("encode_new_memory occluded");
    let co = cosine(&out_occ.features, &expected_occ);
    println!("occlusion-add: cosine={co:.7}");
    assert!(co > 0.999, "occlusion-add cosine {co}");
}
