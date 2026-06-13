//! sc-5116 acceptance gate: the candle Lens DiT adapter merge must match the torch-PEFT reference.
//!
//! Loads the real `transformer/` weights (the cached `microsoft/Lens-Turbo` snapshot) **as f32**,
//! merges the **same** adapter files the trainer ships (`scripts/dump_lens_adapter_golden.py` →
//! diffusers `save_lora_adapter` for LoRA; `get_peft_model_state_dict` + `networkType=lokr` metadata
//! for LoKr), and asserts the merged DiT forward matches the torch-PEFT outputs. Mirrors
//! `mlx-gen-lens/tests/adapter_parity.rs`:
//!   1. **base sanity** — the un-adapted DiT reproduces the reference base forward;
//!   2. **scale-0 no-op** — a scale-0 LoRA merge is **bit-exact** equal to the base forward;
//!   3. **LoRA @ 1** — the fused-QKV LoRA merge matches `lora_out` (tight: a linear-merge delta);
//!   4. **LoKr @ 1** — the LoKr merge matches `lokr_out`.
//!
//! f32 on both sides makes this a tight correctness gate (the same 48-block f32-matmul floor as the
//! dense `dit_parity` gate). Heavy + machine-specific, so it is **gated** on env vars and skips
//! cleanly when they are unset (CPU CI has neither weights nor a GPU):
//!   LENS_DIT_DIR     — the Lens-Turbo `transformer` snapshot dir (config.json + model-*.safetensors)
//!   LENS_ADAPTER_DIR — dir holding the 3 goldens (default: .scratch/lens-adapter-goldens/), i.e.
//!                      lens_adapter_golden.safetensors + lens_lora_adapter.safetensors +
//!                      lens_lokr_adapter.safetensors (run scripts/dump_lens_adapter_golden.py)
//! Run with the `cuda` feature (absolute paths — cargo test cwd is the crate dir):
//!   cargo test -p candle-gen-lens --features cuda --test adapter_parity -- --nocapture

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_gen::candle_core::{safetensors, DType, Device, Result, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::{AdapterKind, AdapterSpec};
use candle_gen_lens::merge_adapters;
use candle_gen_lens::transformer::{LensDitConfig, LensTransformer};

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

/// Peak relative error `max|a-b| / max|b|`, in f64 on CPU.
fn peak_rel(a: &Tensor, b: &Tensor) -> Result<f32> {
    let a = a.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let b = b.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let mut max_diff = 0f64;
    let mut max_b = 0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        max_diff = max_diff.max((*x - *y).abs() as f64);
        max_b = max_b.max((*y).abs() as f64);
    }
    Ok((max_diff / max_b.max(1e-12)) as f32)
}

/// Whether two tensors are bit-for-bit equal (the scale-0 no-op check).
fn bit_exact(a: &Tensor, b: &Tensor) -> Result<bool> {
    let a = a.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let b = b.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    Ok(a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|(x, y)| x.to_bits() == y.to_bits()))
}

/// Load the `transformer/` shards into a CPU tensor map (native dtype). Cloning the map is cheap
/// (Arc-shared storage); the adapter merge replaces only the ~192 target keys with fresh f32 tensors.
fn load_map(dir: &str) -> Result<HashMap<String, Tensor>> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .expect("read transformer dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no .safetensors in {dir}");
    let mut map = HashMap::new();
    for f in &files {
        map.extend(safetensors::load(f, &Device::Cpu)?);
    }
    Ok(map)
}

/// Build an **f32** DiT from a (possibly adapter-merged) CPU tensor map, on `device`.
fn build_dit(map: HashMap<String, Tensor>, device: &Device) -> Result<LensTransformer> {
    let vb = VarBuilder::from_tensors(map, DType::F32, device);
    LensTransformer::new(&LensDitConfig::lens(), vb)
}

#[test]
fn lens_adapters_match_reference() -> Result<()> {
    let dit_dir = match std::env::var("LENS_DIT_DIR") {
        Ok(d) => d,
        Err(_) => {
            eprintln!("SKIP: set LENS_DIT_DIR to the Lens-Turbo transformer snapshot dir");
            return Ok(());
        }
    };
    let adapter_dir = std::env::var("LENS_ADAPTER_DIR")
        .unwrap_or_else(|_| ".scratch/lens-adapter-goldens".into());
    let golden = Path::new(&adapter_dir).join("lens_adapter_golden.safetensors");
    let lora = Path::new(&adapter_dir).join("lens_lora_adapter.safetensors");
    let lokr = Path::new(&adapter_dir).join("lens_lokr_adapter.safetensors");
    for p in [&golden, &lora, &lokr] {
        if !p.exists() {
            eprintln!(
                "SKIP: {} not found (run scripts/dump_lens_adapter_golden.py)",
                p.display()
            );
            return Ok(());
        }
    }

    let device = candle_gen::default_device()
        .map_err(|e| candle_gen::candle_core::Error::Msg(e.to_string()))?;
    eprintln!("device: {device:?}");

    // Goldens (f32) + the exact synthetic inputs the reference used.
    let g = safetensors::load(&golden, &device)?;
    let f32g = |k: &str| -> Result<Tensor> { g[k].to_dtype(DType::F32) };
    let grid = g["grid_fhw"].to_dtype(DType::U32)?.to_vec1::<u32>()?;
    let (frame, h, w) = (grid[0] as usize, grid[1] as usize, grid[2] as usize);
    let timestep = g["timestep"].to_dtype(DType::F32)?.to_vec1::<f32>()?[0];
    let cfg = LensDitConfig::lens();
    let feats: Vec<Tensor> = (0..cfg.num_text_layers)
        .map(|i| f32g(&format!("feat_{i}")))
        .collect::<Result<_>>()?;
    let hidden = f32g("hidden_states")?;
    let txt_len = feats[0].dim(1)?;
    eprintln!("grid=({frame},{h},{w}) timestep={timestep:.5} txt_len={txt_len}");

    // All-valid text → `None` mask (an all-ones mask is the same as no mask), matching the dumper's
    // `torch.ones(...)` and the dense `dit_parity` gate.
    let run = |dit: &LensTransformer| -> Result<Tensor> {
        dit.forward(&hidden, &feats, None, timestep, frame, h, w)
    };

    let map = load_map(&dit_dir)?;

    // --- 1. base (sanity) ---
    let base = run(&build_dit(map.clone(), &device)?)?;
    let base_pr = peak_rel(&base, &f32g("base_out")?)?;
    let base_cos = cosine(&base, &f32g("base_out")?)?;
    eprintln!("base:  peak_rel={base_pr:.3e} cosine={base_cos:.7}");
    assert!(
        base_pr < 2e-2,
        "base peak_rel {base_pr:.3e} — DiT load drift"
    );

    // --- 2. scale-0 LoRA is a bit-exact no-op (W + 0·δ == W) ---
    let mut m0 = map.clone();
    merge_adapters(
        &mut m0,
        &[AdapterSpec::new(lora.clone(), 0.0, AdapterKind::Lora)],
    )
    .map_err(|e| candle_gen::candle_core::Error::Msg(e.to_string()))?;
    let zero = run(&build_dit(m0, &device)?)?;
    assert!(
        bit_exact(&zero, &base)?,
        "scale-0 LoRA is not a bit-exact no-op"
    );
    eprintln!("scale-0 LoRA: bit-exact no-op ✓");

    // --- 3. LoRA @ scale 1 vs torch-PEFT ---
    let mut m1 = map.clone();
    let rep = merge_adapters(
        &mut m1,
        &[AdapterSpec::new(lora.clone(), 1.0, AdapterKind::Lora)],
    )
    .map_err(|e| candle_gen::candle_core::Error::Msg(e.to_string()))?;
    eprintln!(
        "lora merged: {} module(s), {} skipped",
        rep.merged, rep.skipped_keys
    );
    assert!(rep.merged > 0, "no LoRA targets matched");
    let lora_out = run(&build_dit(m1, &device)?)?;
    let lora_pr = peak_rel(&lora_out, &f32g("lora_out")?)?;
    let lora_cos = cosine(&lora_out, &f32g("lora_out")?)?;
    eprintln!("lora:  peak_rel={lora_pr:.3e} cosine={lora_cos:.7}");
    assert!(
        lora_pr < 2e-2 && lora_cos > 0.9999,
        "LoRA diverged from torch-PEFT (peak_rel {lora_pr:.3e} cosine {lora_cos:.7})"
    );
    // The adapter must actually move the output (else the gate would pass on a no-op merge bug).
    assert!(
        !bit_exact(&lora_out, &base)?,
        "LoRA merge did not change the forward — delta not applied"
    );

    // --- 4. LoKr @ scale 1 vs torch-PEFT ---
    let mut m2 = map.clone();
    let rep = merge_adapters(
        &mut m2,
        &[AdapterSpec::new(lokr.clone(), 1.0, AdapterKind::Lokr)],
    )
    .map_err(|e| candle_gen::candle_core::Error::Msg(e.to_string()))?;
    eprintln!(
        "lokr merged: {} module(s), {} skipped",
        rep.merged, rep.skipped_keys
    );
    assert!(rep.merged > 0, "no LoKr targets matched");
    let lokr_out = run(&build_dit(m2, &device)?)?;
    let lokr_pr = peak_rel(&lokr_out, &f32g("lokr_out")?)?;
    let lokr_cos = cosine(&lokr_out, &f32g("lokr_out")?)?;
    eprintln!("lokr:  peak_rel={lokr_pr:.3e} cosine={lokr_cos:.7}");
    assert!(
        lokr_pr < 2e-2 && lokr_cos > 0.9999,
        "LoKr diverged from torch-PEFT (peak_rel {lokr_pr:.3e} cosine {lokr_cos:.7})"
    );

    eprintln!("ALL PASS");
    Ok(())
}
