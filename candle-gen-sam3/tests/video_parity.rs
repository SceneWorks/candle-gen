//! SAM3 end-to-end multi-object video PCS parity (sc-6245): run the candle `Sam3VideoModel::propagate`
//! on the captured clip and check per-frame `obj_id` sets + per-object 288² mask logits against the
//! SAME torch oracle the MLX twin uses (`mlx-gen/scripts/spikes/sam3_oracle/dump_video_fixture.py`,
//! a full `Sam3VideoModel` run). `#[ignore]` until weights + fixture are staged (sc-6248). Run:
//!   SAM3_WEIGHTS=<snapshot> SAM3_VIDEO_FIXTURE=<video_fixture.safetensors> \
//!     cargo test -p candle-gen-sam3 --release --features cuda --test video_parity -- --ignored --nocapture

use std::path::Path;

use candle_gen::candle_core::{Device, Tensor};
use candle_gen::default_device;
use candle_gen_sam3::{Sam3VideoModel, Weights};

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
        .propagate(&frames, &input_ids, &text_mask)
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
