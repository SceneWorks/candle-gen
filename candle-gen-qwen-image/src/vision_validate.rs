//! Qwen2.5-VL vision-language encoder real-weight GPU validation (sc-5487, epic 5480) — an env-driven,
//! `#[ignore]`d integration test that drives the REAL [`QwenVisionLanguageEncoder`] on the deployed
//! hardware from a `Qwen/Qwen-Image-Edit` snapshot (the validated reference is `-2511`).
//!
//! **Gate.** The vision tower must produce sane (finite, non-degenerate) embeds whose count equals the
//! `<|image_pad|>` run, and that vision content must actually flow into the spliced LM conditioning —
//! so the metric is an ablation: encode the prompt with the real vision embeds vs. with **zeroed**
//! vision embeds and assert the resulting prompt embeds differ meaningfully. This isolates Slice A
//! (the conditioning encoder) without needing the transformer / VAE (Slice B).
//!
//! Run (after deploying a Qwen-Image-Edit snapshot + a reference PPM):
//! ```text
//! set QWEN_EDIT_BASE=...\Qwen-Image-Edit-2511   # diffusers snapshot (text_encoder/ tokenizer/ …)
//! set QWEN_EDIT_REF=...\reference.ppm           # an RGB P6 PPM
//! cargo test -p candle-gen-qwen-image --features cuda --release vision_validate::real_weight -- --ignored --nocapture
//! ```

use candle_gen::candle_core::Tensor;
use candle_gen::gen_core::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use candle_gen::testkit::{env_path, read_ppm};

use crate::config::TextEncoderConfig;
use crate::image_processor::{ImageInput, QwenImageProcessor};
use crate::vision_language::load_vision_language_encoder;
use crate::vl_tokenizer::{preprocess_edit_image, tokenize_edit_text};

/// `(mean, std)` of a tensor's f32 values.
fn stats(t: &Tensor) -> (f32, f32) {
    let v = t
        .to_dtype(candle_gen::candle_core::DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let n = v.len() as f32;
    let mean = v.iter().sum::<f32>() / n;
    let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
    (mean, var.sqrt())
}

/// Mean absolute element difference between two same-shape tensors.
fn mean_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
    let av = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let bv = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let sum: f32 = av.iter().zip(&bv).map(|(x, y)| (x - y).abs()).sum();
    sum / av.len() as f32
}

#[test]
#[ignore = "real-weight GPU validation; set QWEN_EDIT_BASE/QWEN_EDIT_REF"]
fn real_weight_vision_language() {
    let base = env_path("QWEN_EDIT_BASE");
    let reference = read_ppm(&env_path("QWEN_EDIT_REF"));
    let device = candle_gen::default_device().expect("device");
    println!(
        "reference {}x{}; loading Qwen2.5-VL vision-language encoder …",
        reference.width, reference.height
    );

    let t0 = std::time::Instant::now();
    let vl = load_vision_language_encoder(&base, &device).expect("load VL encoder");
    println!("loaded in {:?}", t0.elapsed());

    // Image-only preprocess (condition-resize + patchify), then the vision tower.
    let processor = QwenImageProcessor::default();
    let edit_img = preprocess_edit_image(
        &processor,
        ImageInput {
            data: &reference.pixels,
            height: reference.height as usize,
            width: reference.width as usize,
        },
        &device,
    )
    .expect("preprocess_edit_image");
    println!(
        "grid {:?}, pixel_values {:?}, n_image_tokens {}",
        edit_img.grid,
        edit_img.pixel_values.dims(),
        edit_img.n_image_tokens
    );

    let t = std::time::Instant::now();
    let vision = vl
        .encode_vision(&edit_img.pixel_values, &[edit_img.grid])
        .expect("encode_vision");
    println!("[vision tower] {:?}", t.elapsed());
    let (n_vis, h) = (vision.dims()[0], vision.dims()[1]);
    let (vm, vs) = stats(&vision);
    println!("vision embeds [{n_vis}, {h}] mean {vm:.4} std {vs:.4}");
    assert_eq!(
        n_vis, edit_img.n_image_tokens,
        "vision rows must equal the image_pad count"
    );
    assert_eq!(h, 3584, "vision embeds must be the 3584-wide out_hidden");
    assert!(vm.is_finite() && vs.is_finite(), "non-finite vision embeds");
    assert!(vs > 1e-3, "vision embeds are degenerate (std≈0)");

    // Tokenize an edit prompt and run the splice + LM.
    let cfg = TextEncoderConfig::qwen_image();
    let tok = TextTokenizer::from_file(
        base.join("tokenizer/tokenizer.json"),
        TokenizerConfig {
            max_length: cfg.max_length,
            pad_token_id: cfg.pad_token_id,
            chat_template: ChatTemplate::QwenImage,
            pad_to_max_length: false,
        },
    )
    .expect("load tokenizer");
    let ids = tokenize_edit_text(
        &tok,
        "make the sky a vivid orange sunset",
        edit_img.n_image_tokens,
    )
    .expect("tokenize_edit_text");
    let len = ids.len();
    println!(
        "edit prompt sequence length {len} (image_pad run {})",
        edit_img.n_image_tokens
    );
    let input_ids = Tensor::from_vec(ids, (1, len), &device).expect("input_ids tensor");

    let emb_real = vl
        .encode_with_vision(&input_ids, &vision)
        .expect("encode_with_vision");
    let (em, es) = stats(&emb_real);
    println!(
        "prompt embeds {:?} mean {em:.4} std {es:.4}",
        emb_real.dims()
    );
    assert_eq!(
        emb_real.dims(),
        &[1, len - 64, 3584],
        "prompt embeds must drop the 64 template tokens"
    );
    assert!(em.is_finite() && es.is_finite(), "non-finite prompt embeds");

    // Ablation: zeroed vision embeds → the spliced conditioning must change meaningfully.
    let vision_zero = vision.zeros_like().expect("zeros_like");
    let emb_zero = vl
        .encode_with_vision(&input_ids, &vision_zero)
        .expect("encode_with_vision (zeroed vision)");
    let diff = mean_abs_diff(&emb_real, &emb_zero);
    println!("=== Qwen2.5-VL vision-language validation ===");
    println!("  prompt-embed mean|Δ| (real vs zeroed vision): {diff:.4}");
    assert!(
        diff > 1e-3,
        "vision content does not propagate into the conditioning (Δ={diff:.6}) — splice may be broken"
    );
    println!("Qwen2.5-VL vision-language validation PASS ✅");
}
