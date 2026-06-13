//! sc-5112 — Lens DiT parity vs the vendor `LensTransformer2DModel`.
//!
//! Loads the real `transformer/` weights (the cached `microsoft/Lens-Turbo` snapshot) **as f32** and
//! checks, against `scripts/dump_lens_dit_goldens.py`:
//!   1. **per-block** — block 0 reproduces the reference block output given the golden's block-0
//!      inputs (`img_in_out`, `txt_in_out`, `temb`), with the Rust-built RoPE tables;
//!   2. **full forward** — the whole 48-block DiT reproduces the reference output for the same
//!      synthetic inputs.
//!
//! f32 on both sides makes this a tight correctness gate — bf16 cross-backend accumulation over 48
//! residual blocks would obscure subtle bugs (wrong RoPE axis, transposed weight, mis-ordered
//! modulation). The golden + the ~16 GB f32 weight load keep this **env-gated**; it skips cleanly
//! when the inputs are absent (CPU CI has neither weights nor a golden).
//!
//!   LENS_TRANSFORMER_DIR — the Lens-Turbo `transformer` snapshot dir (config.json + *.safetensors)
//!   LENS_DIT_GOLDEN      — lens_dit_golden.safetensors (default: .scratch/lens-dit-golden/…)
//!
//! Run with the `cuda` feature (f32, on the GPU):
//!   cargo test -p candle-gen-lens --features cuda --test dit_parity -- --nocapture

use candle_gen::candle_core::{DType, Result, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen_lens::dit::{LensDitConfig, LensRope3d, LensTransformer, LensTransformerBlock};

/// Cosine similarity over all elements (flattened), computed in f64 on CPU.
fn cosine(a: &Tensor, b: &Tensor) -> Result<f32> {
    let a = a.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let b = b.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(a.len(), b.len(), "shape mismatch in cosine");
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b.iter()) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
    }
    Ok((dot / (na.sqrt() * nb.sqrt() + 1e-12)) as f32)
}

/// Peak relative error `max|a-b| / max|b|`, computed in f64 on CPU.
fn peak_rel(a: &Tensor, b: &Tensor) -> Result<f32> {
    let a = a.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let b = b.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let mut max_diff = 0f64;
    let mut max_ref = 0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        max_diff = max_diff.max((*x - *y).abs() as f64);
        max_ref = max_ref.max((*y).abs() as f64);
    }
    Ok((max_diff / max_ref.max(1e-12)) as f32)
}

#[test]
fn lens_dit_matches_reference() -> Result<()> {
    let tdir = match std::env::var("LENS_TRANSFORMER_DIR") {
        Ok(d) => d,
        Err(_) => {
            eprintln!("SKIP: set LENS_TRANSFORMER_DIR to the Lens-Turbo transformer snapshot dir");
            return Ok(());
        }
    };
    let golden_path = std::env::var("LENS_DIT_GOLDEN")
        .unwrap_or_else(|_| ".scratch/lens-dit-golden/lens_dit_golden.safetensors".to_string());
    if !std::path::Path::new(&golden_path).exists() {
        eprintln!("SKIP: golden not found at {golden_path} (run scripts/dump_lens_dit_goldens.py)");
        return Ok(());
    }

    let device = candle_gen::default_device()
        .map_err(|e| candle_gen::candle_core::Error::Msg(e.to_string()))?;
    eprintln!("device: {device:?}");

    let g = candle_gen::candle_core::safetensors::load(&golden_path, &device)?;
    let dims = g["dims"].to_dtype(DType::I64)?.to_vec1::<i64>()?;
    let (frame, h, w, txt_len) = (
        dims[0] as usize,
        dims[1] as usize,
        dims[2] as usize,
        dims[3] as usize,
    );
    eprintln!("dims: frame={frame} h={h} w={w} txt_len={txt_len}");

    let cfg = LensDitConfig::lens();
    let mlp_hidden = cfg.inner_dim / 3 * 8;

    // f32 mmap of the transformer shards.
    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(&tdir)
        .expect("read transformer dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no .safetensors in {tdir}");
    eprintln!("loading {} transformer shard(s) as f32…", files.len());
    // SAFETY: mmap of read-only weight files (the standard candle loading path).
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, DType::F32, &device)? };

    // --- 1. per-block: block 0 fed the reference's exact block-0 inputs ---
    let block0 = LensTransformerBlock::new(
        cfg.inner_dim,
        cfg.num_heads,
        cfg.head_dim,
        mlp_hidden,
        vb.pp("transformer_blocks").pp(0),
    )?;
    let (img_cos, img_sin, txt_cos, txt_sin) =
        LensRope3d::lens().forward(frame, h, w, txt_len, &device)?;
    let (enc0, hid0) = block0.forward(
        &g["img_in_out"],
        &g["txt_in_out"],
        &g["temb"],
        &img_cos,
        &img_sin,
        &txt_cos,
        &txt_sin,
        None,
    )?;
    let blk_enc_pr = peak_rel(&enc0, &g["block0_enc"])?;
    let blk_hid_pr = peak_rel(&hid0, &g["block0_hidden"])?;
    eprintln!(
        "block0: enc peak_rel={blk_enc_pr:.3e} cosine={:.7} | hidden peak_rel={blk_hid_pr:.3e} cosine={:.7}",
        cosine(&enc0, &g["block0_enc"])?,
        cosine(&hid0, &g["block0_hidden"])?,
    );

    // --- 2. full forward ---
    let dit = LensTransformer::new(&cfg, vb)?;
    let feats: Vec<Tensor> = (0..cfg.num_text_layers)
        .map(|i| g[&format!("feat_{i}")].clone())
        .collect();
    let out = dit.forward(&g["hidden_states"], &feats, None, &g["timestep"], frame, h, w)?;
    let out_pr = peak_rel(&out, &g["out"])?;
    let out_cos = cosine(&out, &g["out"])?;
    eprintln!("full forward: peak_rel={out_pr:.3e} cosine={out_cos:.7}");

    // Per-block is the tight correctness gate: fed the exact reference block-0 inputs, every sub-op
    // (fused QKV, QK-norm, complex RoPE, AdaLN modulation, SwiGLU GateMLP, gated residuals) is
    // exercised in isolation. The full forward then accumulates the cross-backend f32-matmul floor
    // over 48 residual blocks; cosine staying at 4+ nines is the real-bug tripwire.
    assert!(blk_enc_pr < 5e-3, "block0 enc peak_rel {blk_enc_pr:.3e} ≥ 5e-3");
    assert!(blk_hid_pr < 5e-3, "block0 hidden peak_rel {blk_hid_pr:.3e} ≥ 5e-3");
    assert!(out_pr < 2e-2, "full forward peak_rel {out_pr:.3e} ≥ 2e-2");
    assert!(out_cos > 0.9999, "full forward cosine {out_cos:.7} ≤ 0.9999");
    eprintln!("ALL PASS");
    Ok(())
}
