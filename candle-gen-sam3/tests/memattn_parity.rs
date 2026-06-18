//! SAM3 tracker memory-attention parity (sc-6245): run the candle `Sam3Tracker` memory attention
//! (`condition_with_memory` + RoPE tables) against the SAME torch oracle the MLX twin uses
//! (`mlx-gen/scripts/spikes/sam3_oracle/dump_memattn_fixture.py`, captured via a forward hook on a
//! real 2-frame `Sam3VideoModel` PCS run). `#[ignore]` until weights + fixture are staged (sc-6248).
//! Run:
//!   SAM3_WEIGHTS=<snapshot> SAM3_MEMATTN_FIXTURE=<memattn_fixture.safetensors> \
//!     cargo test -p candle-gen-sam3 --release --features cuda --test memattn_parity -- --ignored --nocapture

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
#[ignore = "needs staged facebook/sam3 weights (SAM3_WEIGHTS) + SAM3_MEMATTN_FIXTURE — sc-6248"]
fn memory_attention_matches_oracle() {
    let weights_path =
        std::env::var("SAM3_WEIGHTS").expect("set SAM3_WEIGHTS to the facebook/sam3 snapshot");
    let fixture_path =
        std::env::var("SAM3_MEMATTN_FIXTURE").expect("set SAM3_MEMATTN_FIXTURE to the oracle dump");

    let device = default_device().expect("default device");
    let w = load_weights(&weights_path, &device);
    let fx = Weights::from_file(&fixture_path, &device).expect("load memattn fixture");
    let tracker = Sam3Tracker::from_weights(&w).expect("build tracker");

    // --- Stage 1: RoPE tables (host-built create_inv_freq) vs the reference rotary_emb().
    let (cos, sin) = tracker.memory_attention_rope_tables();
    for (key, got) in [("rope_cos", &cos), ("rope_sin", &sin)] {
        let want = fx.require(key).unwrap();
        assert_eq!(got.dims(), want.dims(), "{key} shape");
        let c = cosine(got, &want);
        println!("{key}: cosine={c:.7}");
        assert!(c > 0.999999, "{key} cosine {c}");
    }

    // --- Stage 2: full memory attention (self-attn + cross-attn over the bank + FFN, 4 layers).
    let cvf = fx.require("current_vision_features").unwrap(); // [5184,1,256]
    let cvp = fx.require("current_vision_position_embeddings").unwrap();
    let mem = fx.require("memory").unwrap(); // [seq_k,1,64]
    let mem_pos = fx.require("memory_pos").unwrap();
    let n_optr = fx_i32(&fx, "num_object_pointer_tokens")[0] as usize;

    let got = tracker
        .condition_with_memory(&cvf, &cvp, &mem, &mem_pos, n_optr)
        .expect("condition_with_memory");
    let want = fx.require("output").unwrap(); // [1,1,5184,256] — flatten-compatible with [1,5184,256]
    assert_eq!(
        got.flatten_all().unwrap().dims(),
        want.flatten_all().unwrap().dims(),
        "output numel"
    );
    let c = cosine(&got, &want);
    println!("memory_attention output (num_obj_ptr={n_optr}): cosine={c:.7}");
    assert!(c > 0.999, "memory_attention cosine {c}");
}
