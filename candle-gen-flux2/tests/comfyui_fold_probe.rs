//! sc-11028 investigation probe (GPU, real weights, no render): isolate WHERE the flux2-dev
//! dense→quantize load path diverges from the pre-packed tier path.
//!
//! Per representative DiT layer (each remap class: plain bf16, qkv row-split, inline-scale fp8 MLP,
//! adaLN half-swap), this compares against a dense-f32 CUDA reference matmul:
//! * `quantize_onto` — the merged comfyui-lane fold (`MatmulStrategy::Int8Fast`, candle
//!   `QMatMul`: `fast_mmvq` at m≤8, `fast_mmq` at m>8),
//! * `quantize_dequant_onto` — the sc-7702-safe fold (`MatmulStrategy::DequantDense`),
//! * `from_packed_gs` — the coherent packed-tier arm (Q8 grid → `Q8_0` requant),
//!
//! at both batch regimes (m=8 vec path, m=4608 the real render's MMQ path), Q8 and Q4. It also
//! cross-checks the comfyui-dequanted weight against the packed tier's dequantized grid
//! (re-verifying the sc-10680 offline weight-parity claim with this exact code).
//!
//! This is what root-caused sc-11028: pre-fix, ONLY the qkv row-chunk layers at nonzero offsets
//! (to_k/to_v) diverged (cos≈0) — through BOTH fold arms, both quants, both batch regimes — while
//! the packed arm and the per-layer weight parity stayed ~1.0: `QTensor::quantize{,_onto}` reads
//! the raw backing storage of a strides-contiguous narrow view, ignoring its start offset, so
//! every double block's to_k/to_v quantized to a copy of to_q. The shared `QLinear` fold now
//! materializes such views (`force_contiguous`) before quantizing.
//!
//! Weightful + CUDA-only → `#[ignore]`; skips gracefully when the local files are absent. Run:
//! ```text
//! cargo test -p candle-gen-flux2 --test comfyui_fold_probe --features cuda --release -- --ignored --nocapture
//! ```

use candle_gen::candle_core::safetensors::MmapedSafetensors;
use candle_gen::candle_core::{DType, Device, Result, Tensor};
use candle_gen::candle_nn::Linear;
use candle_gen::gen_core::Quant;
use candle_gen::quant::{dequant_mlx_q8_gs, DenseLinear, QLinear};

const COMFYUI_FILE_ENV: &str = "FLUX2_COMFYUI_FILE";
const TIER_FILE_ENV: &str = "FLUX2_Q8_TIER_FILE";
const COMFYUI_FILE_DEFAULT: &str =
    r"C:\Users\Michael\ComfyUI-Shared\models\diffusion_models\flux2_dev_fp8mixed.safetensors";
const TIER_FILE_DEFAULT: &str = r"E:\huggingface\hub\models--SceneWorks--flux2-dev-mlx\snapshots\0c9b86f4d91eeaec3db11bcc9cc0e4c006faed74\q8\transformer\diffusion_pytorch_model.safetensors";
/// The flux2-dev-mlx tiers pack at the MLX default group size.
const GROUP_SIZE: usize = 64;

fn cosine(a: &Tensor, b: &Tensor) -> Result<f64> {
    let a = a
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let b = b
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b.iter()) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
    }
    Ok(dot / (na.sqrt() * nb.sqrt() + 1e-12))
}

/// Deterministic pseudo-random f32 in [-1, 1) (splitmix64 of the index) — launch-portable, no RNG.
fn pseudo_random(n: usize, salt: u64) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let mut z = (i as u64)
                .wrapping_add(salt)
                .wrapping_add(0x9E37_79B9_7F4A_7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            ((z >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// How the comfyui (BFL-native) source tensor maps onto the probed diffusers projection.
enum SourceOp {
    /// Plain load (bf16 → f32, or fp8 · `weight_scale` → f32).
    Plain,
    /// Row-chunk `i` of 3 (the fused-qkv split).
    Chunk3(usize),
    /// The adaLN `(shift, scale) → (scale, shift)` half-swap.
    SwapHalves,
}

struct LayerSpec {
    /// Comfyui/BFL source key.
    src: &'static str,
    /// Packed-tier diffusers base key (the `.weight`/`.scales`/`.biases` triple).
    tier_base: &'static str,
    op: SourceOp,
}

/// Load + dequant a comfyui source tensor to f32 CPU — the same math as
/// `convert::build_comfyui_dit_map`'s load closure (fp8: `w = w_fp8 · weight_scale`).
fn load_comfyui(src: &MmapedSafetensors, name: &str) -> Result<Tensor> {
    let cpu = Device::Cpu;
    let t = src.load(name, &cpu)?;
    if t.dtype() != DType::F8E4M3 {
        return t.to_dtype(DType::F32);
    }
    let base = name.strip_suffix(".weight").expect("fp8 key ends .weight");
    let scale = src
        .load(&format!("{base}.weight_scale"), &cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    t.to_dtype(DType::F32)?.affine(scale as f64, 0.0)
}

#[test]
#[ignore = "weightful GPU probe (sc-11028): needs the local comfyui fp8 file + the flux2-dev-mlx q8 tier"]
fn comfyui_fold_probe() -> Result<()> {
    let device = match candle_gen::default_device() {
        Ok(d) if !matches!(d, Device::Cpu) => d,
        _ => {
            eprintln!("SKIP comfyui_fold_probe: needs a CUDA device");
            return Ok(());
        }
    };
    let comfyui_path =
        std::env::var(COMFYUI_FILE_ENV).unwrap_or_else(|_| COMFYUI_FILE_DEFAULT.into());
    let tier_path = std::env::var(TIER_FILE_ENV).unwrap_or_else(|_| TIER_FILE_DEFAULT.into());
    if !std::path::Path::new(&comfyui_path).exists() || !std::path::Path::new(&tier_path).exists() {
        eprintln!("SKIP comfyui_fold_probe: weights not found ({comfyui_path} / {tier_path})");
        return Ok(());
    }
    // SAFETY: read-only mmaps of local weight files.
    let comfyui = unsafe { MmapedSafetensors::new(&comfyui_path)? };
    let tier = unsafe { MmapedSafetensors::new(&tier_path)? };
    let cpu = Device::Cpu;

    let layers = [
        LayerSpec {
            src: "img_in.weight",
            tier_base: "x_embedder",
            op: SourceOp::Plain,
        },
        LayerSpec {
            src: "double_blocks.0.img_attn.qkv.weight",
            tier_base: "transformer_blocks.0.attn.to_q",
            op: SourceOp::Chunk3(0),
        },
        LayerSpec {
            src: "double_blocks.0.img_attn.qkv.weight",
            tier_base: "transformer_blocks.0.attn.to_v",
            op: SourceOp::Chunk3(2),
        },
        LayerSpec {
            src: "double_blocks.0.img_mlp.0.weight",
            tier_base: "transformer_blocks.0.ff.linear_in",
            op: SourceOp::Plain,
        },
        LayerSpec {
            src: "single_blocks.0.linear1.weight",
            tier_base: "single_transformer_blocks.0.attn.to_qkv_mlp_proj",
            op: SourceOp::Plain,
        },
        LayerSpec {
            src: "single_blocks.0.linear2.weight",
            tier_base: "single_transformer_blocks.0.attn.to_out",
            op: SourceOp::Plain,
        },
        LayerSpec {
            src: "final_layer.adaLN_modulation.1.weight",
            tier_base: "norm_out.linear",
            op: SourceOp::SwapHalves,
        },
    ];

    let mut failures: Vec<String> = Vec::new();
    for spec in &layers {
        // ---- comfyui-side dense weight (the map's value for this key) --------------------------
        let raw = load_comfyui(&comfyui, spec.src)?;
        let w_c = match spec.op {
            SourceOp::Plain => raw,
            SourceOp::Chunk3(i) => {
                let each = raw.dim(0)? / 3;
                raw.narrow(0, i * each, each)?.contiguous()?
            }
            SourceOp::SwapHalves => {
                let half = raw.dim(0)? / 2;
                let first = raw.narrow(0, 0, half)?;
                let second = raw.narrow(0, half, half)?;
                Tensor::cat(&[&second, &first], 0)?.contiguous()?
            }
        };
        let (out_dim, in_dim) = w_c.dims2()?;

        // ---- packed-tier side: the dequantized MLX Q8 grid (f32 CPU) ---------------------------
        let wq = tier.load(&format!("{}.weight", spec.tier_base), &cpu)?;
        let scales = tier
            .load(&format!("{}.scales", spec.tier_base), &cpu)?
            .to_dtype(DType::F32)?;
        let biases = tier
            .load(&format!("{}.biases", spec.tier_base), &cpu)?
            .to_dtype(DType::F32)?;
        let grid = dequant_mlx_q8_gs(&wq, &scales, &biases, GROUP_SIZE)?;
        let w_cos = cosine(&w_c, &grid)?;
        eprintln!(
            "[probe] {:<55} [{out_dim}x{in_dim}] weight cos(comfyui, tier-grid) = {w_cos:.6}",
            spec.tier_base
        );
        if w_cos < 0.99 {
            failures.push(format!("{}: weight parity {w_cos:.6}", spec.tier_base));
        }

        // ---- forwards vs the dense-f32 CUDA reference, both batch regimes ----------------------
        let w_dev = w_c.to_device(&device)?;
        for m in [8usize, 4608] {
            let x = Tensor::from_vec(pseudo_random(m * in_dim, out_dim as u64), (m, in_dim), &cpu)?
                .to_device(&device)?;
            let reference = x.matmul(&w_dev.t()?)?;

            let mut arms: Vec<(String, QLinear)> = Vec::new();
            for quant in [Quant::Q8, Quant::Q4] {
                let mut int8 =
                    QLinear::from_dense(DenseLinear::Linear(Linear::new(w_c.clone(), None)));
                int8.quantize_onto(quant, &device)?;
                arms.push((format!("int8fast-{quant:?}"), int8));

                let mut dq =
                    QLinear::from_dense(DenseLinear::Linear(Linear::new(w_c.clone(), None)));
                dq.quantize_dequant_onto(quant, &device)?;
                arms.push((format!("dequant-{quant:?}"), dq));
            }
            arms.push((
                "packed-q8".into(),
                QLinear::from_packed_gs(&wq, &scales, &biases, None, GROUP_SIZE, &device)?,
            ));

            for (name, lin) in &arms {
                let y = lin.forward(&x)?;
                let cos = cosine(&y, &reference)?;
                let flag = if cos < 0.98 { "  <-- DIVERGED" } else { "" };
                eprintln!(
                    "[probe] {:<55} m={m:<5} {name:<14} cos={cos:.6}{flag}",
                    spec.tier_base
                );
                // Q4 quant noise is wider but still >0.98 on these well-scaled DiT weights; the
                // failure mode under investigation is catastrophic (garbage), not marginal.
                if cos < 0.98 {
                    failures.push(format!("{} m={m} {name}: cos={cos:.6}", spec.tier_base));
                }
            }
        }
    }

    assert!(
        failures.is_empty(),
        "sc-11028 probe divergences:\n{}",
        failures.join("\n")
    );
    Ok(())
}
