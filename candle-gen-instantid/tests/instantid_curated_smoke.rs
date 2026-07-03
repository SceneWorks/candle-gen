//! sc-7297 / epic 7114 — curated-sampler **identity** smoke for candle InstantID (the Windows/CUDA
//! twin of `mlx-gen-instantid/tests/instantid_curated_smoke.rs`).
//!
//! Validates that routing InstantID's dual-conditioning denoise through `denoise_curated` (when a
//! curated sampler is selected) preserves identity adherence as well as the bespoke ancestral default.
//! Env-driven so it runs against the app's REAL layout (RealVisXL backbone + the InstantID cache),
//! needing no torch and no golden artifacts. Run (PowerShell, MSVC vcvars + `CUDA_COMPUTE_CAP=120`):
//!
//! ```text
//! $env:IID_SDXL_BASE   = "<RealVisXL snapshot dir>"
//! $env:IID_IDENTITYNET = "<InstantX/InstantID ControlNetModel dir>"
//! $env:IID_IP_ADAPTER  = "<ip-adapter.safetensors>"
//! $env:IID_FACE_DIR    = "<scrfd_10g + arcface_iresnet100 dir>"
//! $env:IID_REF         = "<reference face .ppm (P6) — a CLEAN, single, front-facing portrait>"
//! cargo test -p candle-gen-instantid --features cuda --release --test instantid_curated_smoke -- --ignored --nocapture
//! ```
//!
//! `IID_REF` MUST be a clean single-face portrait: SCRFD picks the largest face, so a montage / group
//! photo yields a tiny low-res crop and a meaningless ArcFace embedding (every cosine then sits near the
//! different-person floor — a false failure). Set `IID_OUT=<dir>` to dump each sampler's render as a P6
//! PPM for eyeballing.
//!
//! Gate (directional, per epic 3109): every sampler — the ancestral default AND each curated solver —
//! must keep the ArcFace cosine well above 0 (identity preserved, gate >0.5). A curated solver that
//! destabilizes InstantID's strong conditioning collapses the face toward 0 / undetectable. The mlx
//! baseline lands euler/heun ~0.79, dpmpp_2m ~0.63.

use std::path::Path;

use candle_gen::gen_core::{Image, WeightsSource};
use candle_gen::testkit::{cosine, env_path, read_ppm};
use candle_gen_instantid::{letterbox, InstantId, InstantIdPaths, InstantIdRequest};

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Optional P6 PPM dump (set `IID_OUT` to a dir) so a failing run can be eyeballed — the same escape
/// hatch `src/validate.rs` uses (the codec-less `image` dep can't write PNG).
fn dump_ppm(img: &Image, name: &str) {
    let Ok(dir) = std::env::var("IID_OUT") else {
        return;
    };
    let _ = std::fs::create_dir_all(&dir);
    let path = Path::new(&dir).join(format!("{name}.ppm"));
    let mut out = format!("P6\n{} {}\n255\n", img.width, img.height).into_bytes();
    out.extend_from_slice(&img.pixels);
    let _ = std::fs::write(path, out);
}

#[test]
#[ignore = "needs RealVisXL + InstantID ControlNet + ip-adapter/scrfd/arcface + a reference face (env-driven, CUDA)"]
fn curated_samplers_preserve_identity() {
    let size = env_usize("IID_SIZE", 768) as u32;
    let steps = env_usize("IID_STEPS", 20);

    let paths = InstantIdPaths {
        sdxl_base: env_path("IID_SDXL_BASE"),
        identitynet: WeightsSource::Dir(env_path("IID_IDENTITYNET")),
        ip_adapter: env_path("IID_IP_ADAPTER"),
        adapters: Vec::new(),
    };
    let mut model = InstantId::load(&paths)
        .expect("load InstantID")
        .with_face(&env_path("IID_FACE_DIR"))
        .expect("attach face stack");

    // Reference identity (its ArcFace embedding drives the IP path).
    let ref_img = read_ppm(&env_path("IID_REF"));
    let canvas = letterbox(&ref_img, size, size);
    let ref_face = model.largest_face(&canvas).expect("detect reference face");
    let kps: Vec<(f32, f32)> = ref_face.kps.iter().map(|p| (p[0], p[1])).collect();
    println!(
        "[smoke] {size}x{size} steps={steps} | ref face det_score={:.3}",
        ref_face.det_score
    );

    // The ancestral default + a spread of curated solvers over the SAME dual conditioning. The default
    // (None ⇒ euler_ancestral) is the byte-exact baseline; the curated names exercise the new
    // `denoise_curated` route in `run_identity_denoise`.
    let samplers: [Option<&str>; 4] = [None, Some("euler"), Some("heun"), Some("dpmpp_2m")];
    let mut results: Vec<(String, f32)> = Vec::new();
    for s in samplers {
        let req = InstantIdRequest {
            prompt: "film still, a portrait photo of a person, cinematic lighting, sharp focus, \
                     high detail, looking at the camera"
                .into(),
            negative: "lowres, blurry, deformed, disfigured, cartoon, painting".into(),
            width: size,
            height: size,
            steps,
            guidance: 5.0,
            seed: 0,
            sampler: s.map(str::to_owned),
            ..Default::default()
        };
        let out = model
            .generate_with(&req, &ref_face.embedding, &kps, &mut |_| {})
            .expect("generate");
        dump_ppm(&out, s.unwrap_or("default"));
        // A destabilizing sampler can yield an image with no detectable face — treat that as
        // identity-lost (cosine 0) rather than a hard panic, so the gate reports it cleanly.
        let cos = match model.largest_face(&out) {
            Ok(f) => cosine(&ref_face.embedding, &f.embedding),
            Err(e) => {
                println!("[smoke]   (no face detected in output: {e})");
                0.0
            }
        };
        let name = s.unwrap_or("euler_ancestral(default)");
        println!("[smoke] sampler={name:<26} ArcFace-cosine(ref,gen) = {cos:.4}");
        results.push((name.to_string(), cos));
    }

    // Identity must be preserved on every path (well above the ~0 broken-pipeline floor).
    for (name, cos) in &results {
        assert!(
            *cos > 0.5,
            "sampler {name}: identity not preserved (cosine {cos:.4}); the curated route may \
             destabilize InstantID's conditioning"
        );
    }
}
