//! sc-9085 / sc-9086 render harness: materialize a DENSE bf16 diffusers turnkey from a hosted
//! MLX-packed quant tier (epic 8506) by routing every packed triple through the **shared packed-load
//! module** `candle_gen::quant` — each packed base is built into a resident [`QLinear`]
//! (`QLinear::from_packed`: Q4 → lossless `Q4_1` repack, Q8 → exact-grid dequant + `Q8_0` re-quant)
//! and then dequantized to the dense weight, exactly the reconstruction the production per-forward
//! path performs. The output tree feeds the standard per-model txt2img examples, so a coherent render
//! proves the shared module reconstructs every component's quantized values end-to-end through real
//! model math (the sc-9086 real-crate render check). Production keeps the QTensor RESIDENT and
//! dequantizes per forward; this materializer just snapshots the same values to disk so an unmodified
//! z-image loader can render them (no per-crate loader conversion needed — that's the umbrella story).
//!
//! ```text
//! cargo run --release --example materialize_mlx_tier -- \
//!   --tier <snapshot>/q4 --out D:\scratch\z-image-turbo-q4-dense
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_gen::candle_core::safetensors::MmapedSafetensors;
use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::quant::QLinear;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Convert one component's `model.safetensors`: packed triples → dense bf16 `{base}.weight` via the
/// shared `QLinear::from_packed` reconstruction, everything else passed through untouched.
fn convert_file(src: &Path, dst: &Path) -> Result<(usize, usize)> {
    // SAFETY: HF-cache blobs are immutable while we read them.
    let st = unsafe { MmapedSafetensors::new(src)? };
    let names: Vec<String> = st.tensors().iter().map(|(k, _)| k.clone()).collect();
    let cpu = Device::Cpu;

    let mut out: HashMap<String, Tensor> = HashMap::new();
    let (mut packed, mut dense) = (0usize, 0usize);
    for name in &names {
        if name.ends_with(".scales") || name.ends_with(".biases") {
            continue; // consumed with their base's `.weight`
        }
        let base = name.strip_suffix(".weight").unwrap_or(name);
        let scales_key = format!("{base}.scales");
        if name.ends_with(".weight") && names.iter().any(|n| n == &scales_key) {
            let wq = st.load(name, &cpu)?;
            let scales = st.load(&scales_key, &cpu)?;
            let biases = st.load(&format!("{base}.biases"), &cpu)?;
            // Build the same resident QLinear the production packed-load path builds, then dequantize
            // it to the dense weight (Q4 lossless, Q8 the accepted re-quant) — the shared module owns
            // the bit-width dispatch, so this example carries no repack logic of its own.
            let grid = match QLinear::from_packed(&wq, &scales, &biases, None, &cpu)? {
                QLinear::Quantized {
                    weight: candle_gen::quant::QuantWeight::Dequant(weight),
                    ..
                } => weight.dequantize(&cpu)?,
                // A packed tier is always the dequant-dense (sc-7702-safe) arm; the int8-fast/Dense arms
                // never come out of `from_packed`.
                _ => unreachable!("from_packed always yields a dequant-dense Quantized"),
            };
            out.insert(name.clone(), grid.to_dtype(DType::BF16)?);
            packed += 1;
        } else {
            out.insert(name.clone(), st.load(name, &cpu)?);
            dense += 1;
        }
    }
    std::fs::create_dir_all(dst.parent().ok_or("dst has no parent")?)?;
    candle_gen::candle_core::safetensors::save(&out, dst)?;
    Ok((packed, dense))
}

/// Copy a sidecar file, stripping the `quantization` block a packed component's `config.json`
/// carries (the dense materialization no longer matches it).
fn copy_sidecar(src: &Path, dst: &Path) -> Result<()> {
    std::fs::create_dir_all(dst.parent().ok_or("dst has no parent")?)?;
    if src.file_name().and_then(|s| s.to_str()) == Some("config.json") {
        let mut v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(src)?)?;
        if let Some(obj) = v.as_object_mut() {
            obj.remove("quantization");
        }
        std::fs::write(dst, serde_json::to_string_pretty(&v)?)?;
    } else {
        std::fs::copy(src, dst)?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let get = |key: &str| {
        args.iter()
            .position(|a| a == key)
            .and_then(|i| args.get(i + 1).cloned())
    };
    let tier = PathBuf::from(get("--tier").ok_or("pass --tier <packed tier dir>")?);
    let out = PathBuf::from(get("--out").ok_or("pass --out <dense output dir>")?);

    let mut stack = vec![tier.clone()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir)? {
            let path = entry?.path();
            let rel = path.strip_prefix(&tier)?.to_path_buf();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
                let t = std::time::Instant::now();
                let (packed, dense) = convert_file(&path, &out.join(&rel))?;
                println!(
                    "[materialize] {} — {packed} packed triples dequantized, {dense} dense passed \
                     through ({:.1}s)",
                    rel.display(),
                    t.elapsed().as_secs_f32()
                );
            } else {
                copy_sidecar(&path, &out.join(&rel))?;
            }
        }
    }
    println!("[materialize] dense turnkey at {}", out.display());
    Ok(())
}
