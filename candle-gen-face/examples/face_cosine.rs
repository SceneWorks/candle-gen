//! ArcFace identity cosine between two images (epic 10871 P4.2 identity-preservation scoring, sc-10886).
//! The **candle** twin of `mlx-gen-face`'s `face_cosine` (mlx-gen #702): detects the largest face in
//! each image (SCRFD-10g), aligns + embeds it (ArcFace iresnet100/glintr100), L2-normalizes, and prints
//! the cosine similarity — the same detect→align→embed stack the InstantID/likeness path uses. It scores
//! any PNG/JPG, so a candle edit output can be compared against the same person reference the MLX side
//! used, closing the cross-backend A/B. Weights default to the cached `instantid-mlx` bundle (the exact
//! `scrfd_10g.safetensors` + `arcface_iresnet100.safetensors` the MLX scorer loaded).
//!
//! Runs f32 on the build's default device; CPU is fine (a single detect + one recognition forward per
//! image), so it needs no `--features cuda`.
//!
//! Run: `cargo run --release -p candle-gen-face --example face_cosine -- <imgA> <imgB>`
//! Env: `FACE_DIR` (dir holding scrfd_10g.safetensors + arcface_iresnet100.safetensors).

use std::path::PathBuf;

use candle_gen_face::load;

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

/// Load an image as interleaved RGB8, returning `(pixels, height, width)` — the shape
/// `FaceAnalysis::analyze` consumes.
fn load_rgb(path: &str) -> (Vec<u8>, usize, usize) {
    let img = image::open(path)
        .unwrap_or_else(|e| panic!("open {path}: {e}"))
        .to_rgb8();
    let (w, h) = img.dimensions();
    (img.into_raw(), h as usize, w as usize)
}

/// L2-normalize an embedding (guard the zero vector).
fn l2(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    v.iter().map(|x| x / n).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let a_path = args
        .get(1)
        .expect("usage: face_cosine <imgA> <imgB>")
        .clone();
    let b_path = args
        .get(2)
        .expect("usage: face_cosine <imgA> <imgB>")
        .clone();
    let face_dir = env_or(
        "FACE_DIR",
        "E:/huggingface/hub/models--SceneWorks--instantid-mlx/snapshots/\
         bca0cacf8e5e04529bb2b326a521361b02be84fd",
    );

    let fa = load(&PathBuf::from(&face_dir)).expect("load SCRFD + ArcFace stack from FACE_DIR");
    let inner = fa.inner();

    let (a, ah, aw) = load_rgb(&a_path);
    let (b, bh, bw) = load_rgb(&b_path);
    let faces_a = inner.analyze(&a, ah, aw).expect("analyze A");
    let faces_b = inner.analyze(&b, bh, bw).expect("analyze B");
    eprintln!(
        "[cosine] A={a_path}: {} face(s); B={b_path}: {} face(s)",
        faces_a.len(),
        faces_b.len()
    );
    if faces_a.is_empty() || faces_b.is_empty() {
        eprintln!("[cosine] NO FACE detected in one image — cannot score identity");
        println!(
            "cosine NA (missing face: A={} B={})",
            faces_a.len(),
            faces_b.len()
        );
        return;
    }
    // `analyze` returns detections largest-first, so index 0 is the dominant face in each image.
    let fa0 = &faces_a[0];
    let fb0 = &faces_b[0];
    eprintln!(
        "[cosine] A top face det={:.3} bbox={:?}; B top face det={:.3} bbox={:?}",
        fa0.det_score, fa0.bbox, fb0.det_score, fb0.bbox
    );
    let ea = l2(&fa0.embedding);
    let eb = l2(&fb0.embedding);
    let cos: f32 = ea.iter().zip(&eb).map(|(x, y)| x * y).sum();
    println!("cosine {cos:.4}");
}
