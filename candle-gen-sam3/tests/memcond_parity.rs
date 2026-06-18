//! SAM3 per-object memory-conditioning parity (sc-6245): assemble the per-object memory bank + object
//! pointers and run the candle `Sam3Tracker::prepare_memory_conditioned_features`, then check the
//! conditioned feature map against the SAME torch oracle the MLX twin uses
//! (`mlx-gen/scripts/spikes/sam3_oracle/dump_memcond_fixture.py`). Validates the bank-assembly math
//! (temporal-pos add + object-pointer sine-PE/project/4×64-split) end-to-end through memory attention.
//! `#[ignore]` until weights + fixture are staged (sc-6248). Run:
//!   SAM3_WEIGHTS=<snapshot> SAM3_MEMCOND_FIXTURE=<memcond_fixture.safetensors> \
//!     cargo test -p candle-gen-sam3 --release --features cuda --test memcond_parity -- --ignored --nocapture

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
#[ignore = "needs staged facebook/sam3 weights (SAM3_WEIGHTS) + SAM3_MEMCOND_FIXTURE — sc-6248"]
fn memory_conditioning_matches_oracle() {
    let weights_path =
        std::env::var("SAM3_WEIGHTS").expect("set SAM3_WEIGHTS to the facebook/sam3 snapshot");
    let fixture_path =
        std::env::var("SAM3_MEMCOND_FIXTURE").expect("set SAM3_MEMCOND_FIXTURE to the oracle dump");

    let device = default_device().expect("default device");
    let w = load_weights(&weights_path, &device);
    let fx = Weights::from_file(&fixture_path, &device).expect("load memcond fixture");
    let tracker = Sam3Tracker::from_weights(&w).expect("build tracker");

    let cvf = fx.require("current_vision_features").unwrap(); // [5184,1,256]
    let cvp = fx.require("current_vision_position_embeddings").unwrap();

    // The neck's sine position encoding (current_vision_pos) is weight-free — verify the host
    // computation matches the captured fixture, so the pipeline can recompute it instead of caching it.
    let g = (cvf.dim(0).unwrap() as f64).sqrt() as usize;
    let pos_computed = tracker.frame_position_encoding(g).unwrap();
    let c_pos = cosine(&pos_computed, &cvp);
    println!("frame_position_encoding: cosine={c_pos:.7}");
    assert!(c_pos > 0.999999, "current_vision_pos cosine {c_pos}");

    // Spatial memory frames: [M,5184,1,64] features/pos + [M] offsets.
    let mem_feats = fx.require("memory_features").unwrap();
    let mem_pos = fx.require("memory_pos_enc").unwrap();
    let mem_offsets = fx_i32(&fx, "memory_offsets");
    let m = mem_feats.dim(0).unwrap();
    let seq = mem_feats.dim(1).unwrap();
    let mut spatial: Vec<(i32, Tensor, Tensor)> = Vec::new();
    for (i, &off) in mem_offsets.iter().enumerate() {
        let feat = mem_feats
            .narrow(0, i, 1)
            .unwrap()
            .reshape((seq, 1, 64))
            .unwrap();
        let pos = mem_pos
            .narrow(0, i, 1)
            .unwrap()
            .reshape((seq, 1, 64))
            .unwrap();
        spatial.push((off, feat, pos));
    }

    // Object pointers: [P,1,1,256] + [P] offsets.
    let ptrs = fx.require("object_pointers").unwrap();
    let ptr_offsets = fx_i32(&fx, "object_pointer_offsets");
    let max_optr = fx_i32(&fx, "max_object_pointers_to_use")[0];
    let n_optr_want = fx_i32(&fx, "num_object_pointer_tokens")[0];
    let p = ptrs.dim(0).unwrap();
    let mut object_pointers: Vec<(i32, Tensor)> = Vec::new();
    for (j, &off) in ptr_offsets.iter().enumerate() {
        let t = ptrs.narrow(0, j, 1).unwrap().reshape((1, 256)).unwrap();
        object_pointers.push((off, t));
    }
    println!(
        "M={m} mem_offsets={mem_offsets:?} P={p} ptr_offsets={ptr_offsets:?} max_optr={max_optr} \
         num_obj_ptr_tokens(want)={n_optr_want}"
    );

    let got = tracker
        .prepare_memory_conditioned_features(&cvf, &cvp, &spatial, &object_pointers, max_optr)
        .expect("prepare_memory_conditioned_features"); // NHWC [1,72,72,256]

    // Reference output is NCHW [1,256,72,72] → permute to NHWC for an aligned cosine.
    let want = fx
        .require("output")
        .unwrap()
        .permute([0, 2, 3, 1])
        .unwrap()
        .contiguous()
        .unwrap(); // [1,72,72,256]
    assert_eq!(got.dims(), want.dims(), "conditioned feature map shape");
    let c = cosine(&got, &want);
    println!("conditioned feature map: cosine={c:.7}");
    assert!(c > 0.999, "memory-conditioned feature map cosine {c}");
}
