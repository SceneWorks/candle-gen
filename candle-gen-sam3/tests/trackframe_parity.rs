//! SAM3 no-prompt tracking-frame decode parity (sc-6245): feed the captured memory-conditioned
//! features + high-res features into the candle `Sam3Tracker::decode_tracked_frame` and check the
//! low-res/high-res masks + object pointer + score against the SAME torch oracle the MLX twin uses
//! (`mlx-gen/scripts/spikes/sam3_oracle/dump_trackframe_fixture.py`). `#[ignore]` until weights +
//! fixture are staged (sc-6248). Run:
//!   SAM3_WEIGHTS=<snapshot> SAM3_TRACKFRAME_FIXTURE=<trackframe_fixture.safetensors> \
//!     cargo test -p candle-gen-sam3 --release --features cuda --test trackframe_parity -- --ignored --nocapture

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

/// NCHW `[1,C,H,W]` → NHWC `[1,H,W,C]`.
fn nhwc(a: &Tensor) -> Tensor {
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
#[ignore = "needs staged facebook/sam3 weights (SAM3_WEIGHTS) + SAM3_TRACKFRAME_FIXTURE — sc-6248"]
fn tracked_frame_decode_matches_oracle() {
    let weights_path =
        std::env::var("SAM3_WEIGHTS").expect("set SAM3_WEIGHTS to the facebook/sam3 snapshot");
    let fixture_path = std::env::var("SAM3_TRACKFRAME_FIXTURE")
        .expect("set SAM3_TRACKFRAME_FIXTURE to the oracle dump");

    let device = default_device().expect("default device");
    let w = load_weights(&weights_path, &device);
    let fx = Weights::from_file(&fixture_path, &device).expect("load trackframe fixture");
    let tracker = Sam3Tracker::from_weights(&w).expect("build tracker");

    let pix_feat = nhwc(&fx.require("pix_feat").unwrap()); // [1,72,72,256]
    let feat_s0 = nhwc(&fx.require("feat_s0").unwrap()); // [1,288,288,32]
    let feat_s1 = nhwc(&fx.require("feat_s1").unwrap()); // [1,144,144,64]

    let out = tracker
        .decode_tracked_frame(&pix_feat, &[feat_s0, feat_s1])
        .expect("decode_tracked_frame");

    let pred_want = fx.require("pred_masks").unwrap(); // [1,1,288,288]
    let high_want = fx.require("high_res_masks").unwrap(); // [1,1,1008,1008]
    let ptr_want = fx.require("object_pointer").unwrap(); // [1,1,256]
    let score_want = scalar(&fx.require("object_score_logits").unwrap());

    let c_low = cosine(&out.low_res, &pred_want);
    let c_high = cosine(&out.high_res, &high_want);
    let c_ptr = cosine(&out.object_pointer, &ptr_want);
    println!(
        "low_res cosine={c_low:.7}  high_res cosine={c_high:.7}  object_pointer cosine={c_ptr:.7}  \
         object_score got={:.4} want={score_want:.4}",
        out.object_score
    );
    assert!(c_low > 0.9999, "low_res mask cosine {c_low}");
    assert!(c_high > 0.9999, "high_res mask cosine {c_high}");
    assert!(c_ptr > 0.9999, "object_pointer cosine {c_ptr}");
    assert!(
        (out.object_score - score_want).abs() < 0.05,
        "object_score |Δ| too large"
    );
}
