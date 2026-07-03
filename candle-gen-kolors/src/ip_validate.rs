//! Kolors IP-Adapter-Plus real-weight GPU validation (sc-5488, epic 5480) — an env-driven, `#[ignore]`d
//! integration test that drives the REAL [`IpAdapterKolors`] stack on the deployed hardware
//! (Kolors-diffusers + `Kolors-IP-Adapter-Plus` + the CLIP ViT-L/14-336 image encoder + a reference
//! image). The Kolors sibling of the SDXL IP-Adapter Phase-5 harness.
//!
//! **Quantitative gate (no extra deps).** IP-Adapter conditions on CLIP image features, so the metric is
//! the CLIP-feature cosine between the reference and the generated output — using the SAME ViT-L/14-336
//! tower the provider uses. We generate twice at one seed: **with** IP (`ip_adapter_scale > 0`) and
//! **without** (`ip_adapter_scale = 0`, the branch gated off → plain Kolors), and assert the IP run's
//! reference-cosine is meaningfully higher — the IP path actually pulls the output toward the reference.
//! Plus the cancel contract (pre + mid-denoise). Outputs are written as PPM for eyeballing.
//!
//! Run (after deploying weights into the HF cache / a local dir):
//! ```text
//! set IP_KOLORS_BASE=...\Kolors-diffusers           # tokenizer/ text_encoder/ unet/ vae/
//! set IP_KOLORS_IPADAPTER=...\Kolors-IP-Adapter-Plus # image_encoder/ + ip_adapter_plus_general.safetensors
//! set IP_KOLORS_REF=...\ref.ppm                      # a reference image (P6 PPM)
//! set IP_KOLORS_OUT=...\out                           # output dir
//! cargo test -p candle-gen-kolors --features cuda --release ip_validate::real_weight -- --ignored --nocapture
//! ```

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use candle_gen::candle_core::{DType, Device};
use candle_gen::gen_core::runtime::CancelFlag;
use candle_gen::gen_core::{Image, Progress};
use candle_gen::testkit::{cosine_dot as cosine, env_path, read_ppm, write_ppm};

use candle_gen_sdxl::ip_adapter::preprocess_clip_image_sized;
use candle_gen_sdxl::vision_encoder::{ClipVisionEncoder, VisionConfig};
use candle_gen_sdxl::weights::Weights;

use crate::ip_provider::{IpAdapterKolors, IpAdapterKolorsPaths, IpAdapterKolorsRequest};

/// A standalone CLIP ViT-L/14-336 feature extractor for the cosine metric (independent of the model's
/// private encoder): preprocess → penultimate → mean-pool over tokens → L2-normalize. Returns a 1024-vec.
struct ClipMetric {
    encoder: ClipVisionEncoder,
    size: usize,
    device: Device,
}

impl ClipMetric {
    fn load(image_encoder_dir: &Path) -> Self {
        let device = candle_gen::default_device().unwrap();
        let cfg = VisionConfig::vit_l_14_336();
        let path = ["model.safetensors", "model.fp16.safetensors"]
            .iter()
            .map(|n| image_encoder_dir.join(n))
            .find(|p| p.is_file())
            .unwrap_or_else(|| {
                panic!("no model.safetensors under {}", image_encoder_dir.display())
            });
        let w = Weights::from_file(&path, &device, DType::F32).unwrap();
        let encoder = ClipVisionEncoder::from_weights(&w, &cfg).unwrap();
        Self {
            encoder,
            size: cfg.image_size,
            device,
        }
    }

    fn feature(&self, img: &Image) -> Vec<f32> {
        let px = preprocess_clip_image_sized(img, self.size, &self.device).unwrap();
        let penult = self.encoder.penultimate(&px).unwrap(); // [1, 577, 1024]
        let pooled = penult.mean(1).unwrap().flatten_all().unwrap(); // [1024]
        let v = pooled
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        v.iter().map(|x| x / norm).collect()
    }
}

/// Drive the real Kolors IP-Adapter stack: a with-IP vs no-IP ablation (the IP run must score a higher
/// reference cosine) + the cancel contract. Visually-inspectable PPMs land in `IP_KOLORS_OUT`.
#[test]
#[ignore = "real-weight GPU validation; set IP_KOLORS_BASE/IP_KOLORS_IPADAPTER/IP_KOLORS_REF/IP_KOLORS_OUT"]
fn real_weight_ip_adapter() {
    let out_dir = env_path("IP_KOLORS_OUT");
    std::fs::create_dir_all(&out_dir).ok();
    let ip_adapter = env_path("IP_KOLORS_IPADAPTER");

    let paths = IpAdapterKolorsPaths {
        kolors_base: env_path("IP_KOLORS_BASE"),
        ip_adapter: ip_adapter.clone(),
    };
    let reference = read_ppm(&env_path("IP_KOLORS_REF"));
    println!(
        "reference {}x{}; loading IpAdapterKolors …",
        reference.width, reference.height
    );

    let t0 = std::time::Instant::now();
    let mut model = IpAdapterKolors::load(&paths).expect("load IpAdapterKolors");
    println!("loaded in {:?}", t0.elapsed());

    let base = IpAdapterKolorsRequest {
        prompt: "a cinematic portrait photo, soft natural light, photorealistic, sharp focus"
            .into(),
        negative: "blurry, lowres, deformed, watermark, text".into(),
        width: 1024,
        height: 1024,
        steps: 50,
        guidance: 5.0,
        ip_adapter_scale: 0.6,
        sampler: None,
        scheduler: None,
        seed: 12345,
        cancel: CancelFlag::new(),
    };

    let mut noop = |_p: Progress| {};

    // With IP.
    let t = std::time::Instant::now();
    let out_ip = model
        .generate(&base, &reference, &mut noop)
        .expect("generate (ip)");
    println!("[ip] {:?}", t.elapsed());
    write_ppm(&out_dir.join("kolors_ip.ppm"), &out_ip);

    // Without IP (scale 0 → branch gated off → plain Kolors at the same seed/prompt).
    let plain_req = IpAdapterKolorsRequest {
        ip_adapter_scale: 0.0,
        ..base.clone()
    };
    let t = std::time::Instant::now();
    let out_plain = model
        .generate(&plain_req, &reference, &mut noop)
        .expect("generate (no ip)");
    println!("[no-ip] {:?}", t.elapsed());
    write_ppm(&out_dir.join("kolors_no_ip.ppm"), &out_plain);

    // CLIP-feature cosine to the reference: the IP run must pull meaningfully closer.
    let metric = ClipMetric::load(&ip_adapter.join("image_encoder"));
    let ref_feat = metric.feature(&reference);
    let cos_ip = cosine(&ref_feat, &metric.feature(&out_ip));
    let cos_plain = cosine(&ref_feat, &metric.feature(&out_plain));
    println!("=== Kolors IP-Adapter validation ===");
    println!("  clip cosine (ip)    : {cos_ip:.4}");
    println!("  clip cosine (no-ip) : {cos_plain:.4}");
    println!("  delta               : {:.4}", cos_ip - cos_plain);
    println!("  outputs: {}", out_dir.display());

    // Pre-cancel: a flag set before the first step short-circuits.
    let cancelled = IpAdapterKolorsRequest {
        cancel: {
            let c = CancelFlag::new();
            c.cancel();
            c
        },
        ..base.clone()
    };
    let pre = model.generate(&cancelled, &reference, &mut noop);
    assert!(
        matches!(pre, Err(candle_gen::CandleError::Canceled)),
        "pre-cancel must return Canceled"
    );
    println!("[cancel:pre] Err(Canceled) ✓");

    // Mid-denoise cancel: flip the flag from the progress callback on the 3rd step; the next step's
    // start-of-loop check must short-circuit.
    let mid = CancelFlag::new();
    let mid_req = IpAdapterKolorsRequest {
        cancel: mid.clone(),
        ..base.clone()
    };
    let seen = Arc::new(AtomicUsize::new(0));
    let seen_cb = seen.clone();
    let mut cancel_at_3 = move |p: Progress| {
        if let Progress::Step { current, .. } = p {
            seen_cb.store(current as usize, Ordering::SeqCst);
            if current >= 3 {
                mid.cancel();
            }
        }
    };
    let res = model.generate(&mid_req, &reference, &mut cancel_at_3);
    assert!(
        matches!(res, Err(candle_gen::CandleError::Canceled)),
        "mid-cancel must return Canceled"
    );
    let steps_seen = seen.load(Ordering::SeqCst);
    assert!(
        (3..=4).contains(&steps_seen),
        "mid-cancel should stop right after step 3 (saw {steps_seen})"
    );
    println!("[cancel:mid] Err(Canceled) after {steps_seen} steps ✓");

    // The gate: IP conditioning pulls the output toward the reference in CLIP space.
    assert!(
        cos_ip > cos_plain + 0.02,
        "IP run cosine {cos_ip:.4} not meaningfully above no-IP {cos_plain:.4}"
    );
    assert!(out_ip.width == 1024 && out_ip.height == 1024);
    println!("Kolors IP-Adapter validation PASS ✅");
}
