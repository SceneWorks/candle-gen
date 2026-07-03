//! sc-7148 — **LTX-2.3 video VAE decode CUDA peak-VRAM sweep.** The candle/CUDA analog of mlx-gen-ltx's
//! `vae_decode_sweep.rs` (sc-6894): measures the *real* concurrent GPU memory peak of
//! [`LtxVideoVae::decode`] / [`LtxVideoVae::decode_tiled`] / [`LtxVideoVae::decode_budgeted`] across a
//! grid of output sizes × tile sizes on real LTX-2.3 weights, so the budgeted decode cost model
//! (`estimated_ltx_decode_peak_gib`, the `LTX_VAE_FIXED/ACCUM/TILE_BYTES` constants) can be **fit from
//! CUDA measurements** instead of the mlx-Metal placeholder anchors it shipped with.
//!
//! Decode-only: loads just the `vae.*` keys of a dense LTX-2.3 checkpoint (the DiT is mmapped but its
//! tensors are never touched on the VAE path). One config **per process** (env-driven) so an OOM on the
//! largest configs kills only this process and the driving shell loop keeps the earlier anchors.
//! Synthetic (random) latents — we measure cost/scaling, not parity.
//!
//! Peak is sampled device-wide via `nvidia-smi --query-gpu=memory.used` in a background thread (Windows
//! WDDM reports per-process `used_memory` as `[N/A]`, so device-level used is the honest "will it fit"
//! quantity — and it matches the budget semantics, where the safe ceiling is *total* VRAM × 0.85). Run
//! on an otherwise-idle GPU; the printed `baseline` line lets you confirm that.
//!
//! ```text
//! # single-pass anchor (the FIXED + ACCUM floor):
//! set CUDA_VISIBLE_DEVICES=0
//! set LTX_CKPT=C:\Users\…\models--Lightricks--LTX-2.3\snapshots\<h>\ltx-2.3-22b-distilled.safetensors
//! set LTX_W=768& set LTX_H=768& set LTX_FRAMES=25
//!   cargo test -p candle-gen-ltx --features cuda --release --test vae_decode_sweep -- --ignored --nocapture
//! # add LTX_TILE_PX=512 [LTX_OVERLAP_PX=64 LTX_TILE_FRAMES=.. LTX_OVERLAP_FRAMES=..] for a fixed-tile run
//! # add LTX_BUDGETED=1 [LTX_VAE_BUDGET_GIB=48] to exercise the production budgeted selector
//! # add LTX_GPU=0 to pick which GPU ordinal nvidia-smi samples (default 0; pair with CUDA_VISIBLE_DEVICES)
//! ```
#![cfg(feature = "cuda")]

use std::path::PathBuf;
use std::time::Instant;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::tiling::{SpatialTiling, TemporalTiling, TilingConfig};
use candle_gen::testkit::{used_mib, PeakSampler};
use candle_gen_ltx::config::LATENT_CHANNELS;
use candle_gen_ltx::vae::{auto_tiling_budgeted_ltx, LtxVideoVae};

fn env_usize(var: &str, default: usize) -> usize {
    std::env::var(var)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn gib(bytes: f64) -> f64 {
    bytes / (1024.0 * 1024.0 * 1024.0)
}

/// Locate a dense LTX-2.3 checkpoint: `LTX_CKPT` (explicit file) wins; else search `LTX_SNAPSHOT` for a
/// `*distilled*.safetensors`/`*dev*.safetensors`/`*bf16*.safetensors` (the VAE lives under `vae.` in the
/// single-file dense checkpoint — see `candle_gen_ltx::lib`).
fn locate_ckpt() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("LTX_CKPT") {
        return Some(PathBuf::from(p));
    }
    let dir = std::env::var_os("LTX_SNAPSHOT").map(PathBuf::from)?;
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    // Prefer distilled, then dev, then a bf16 full-model fine-tune; never an LoRA/upscaler shard.
    files.sort_by_key(|p| {
        let n = p.file_name().unwrap().to_string_lossy().to_lowercase();
        let bad = n.contains("lora") || n.contains("upscaler");
        let rank = if bad {
            9
        } else if n.contains("distilled") {
            0
        } else if n.contains("dev") {
            1
        } else if n.contains("bf16") {
            2
        } else {
            3
        };
        (rank, n)
    });
    files.into_iter().next()
}

#[test]
#[ignore = "needs a dense LTX-2.3 .safetensors (LTX_CKPT / LTX_SNAPSHOT) + CUDA; GPU-heavy"]
fn ltx_vae_decode_sweep() {
    let Some(ckpt) = locate_ckpt() else {
        eprintln!(
            "skip: set LTX_CKPT to a dense LTX-2.3 .safetensors (or LTX_SNAPSHOT to its dir)"
        );
        return;
    };
    let gpu = env_usize("LTX_GPU", 0);
    let w_out = env_usize("LTX_W", 768) as i32;
    let h_out = env_usize("LTX_H", 768) as i32;
    let frames = env_usize("LTX_FRAMES", 25) as i32;
    // LTX VAE: spatial ×32, temporal ×8 **causal** (out_f = 1 + (T_lat−1)·8 ⇒ frames = 1 + 8·k).
    assert_eq!(
        (frames - 1) % 8,
        0,
        "LTX_FRAMES must be 1 + 8·k (got {frames})"
    );
    assert_eq!(h_out % 32, 0, "LTX_H must be a multiple of 32");
    assert_eq!(w_out % 32, 0, "LTX_W must be a multiple of 32");
    let (t_lat, h_lat, w_lat) = ((frames - 1) / 8 + 1, h_out / 32, w_out / 32);

    let dev = Device::new_cuda(gpu).expect("cuda device");
    // mmap the dense checkpoint as f32 (VAE_DTYPE); only the `vae.*` keys are realized below.
    // SAFETY: mmap of a read-only, process-owned weight file resolved from our snapshot; not mutated
    // behind the mapping — the standard candle loading path.
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[&ckpt], DType::F32, &dev)
            .expect("mmap LTX checkpoint")
    };
    let vae = LtxVideoVae::new(vb.pp("vae"), LATENT_CHANNELS, 4).expect("LtxVideoVae::new");

    // Warm the CUDA context / cuBLAS handles with a tiny decode so the measured peak reflects the real
    // decode's working set, not one-time context creation.
    let warm = Tensor::randn(0f32, 1f32, (1, LATENT_CHANNELS, 1, 2, 2), &dev).unwrap();
    let _ = vae.decode(&warm).unwrap();
    dev.synchronize().unwrap();

    // Synthetic latent [B=1, 128, T_lat, H_lat, W_lat] (random — cost only, not parity).
    let latent = Tensor::randn(
        0f32,
        1f32,
        (
            1,
            LATENT_CHANNELS,
            t_lat as usize,
            h_lat as usize,
            w_lat as usize,
        ),
        &dev,
    )
    .unwrap();
    let (out_f, out_h, out_w) = (1 + (t_lat - 1) * 8, h_lat * 32, w_lat * 32);

    // Tile selection: LTX_BUDGETED=1 exercises the production `auto_tiling_budgeted_ltx` selector
    // (honors LTX_VAE_BUDGET_GIB); else LTX_TILE_PX sets a fixed tile (for anchor fitting); else
    // single-pass.
    let tile_px = env_usize("LTX_TILE_PX", 0) as i32;
    let cfg: Option<TilingConfig> = if env_usize("LTX_BUDGETED", 0) == 1 {
        auto_tiling_budgeted_ltx(out_h, out_w, out_f)
            .expect("ltx decode fits the budget (catchable error if not)")
    } else if tile_px > 0 {
        let overlap_px = env_usize("LTX_OVERLAP_PX", 64) as i32;
        let tf = env_usize("LTX_TILE_FRAMES", 0) as i32;
        Some(TilingConfig {
            spatial: Some(SpatialTiling {
                tile_px,
                overlap_px,
            }),
            temporal: (tf > 0).then(|| TemporalTiling {
                tile_frames: tf,
                overlap_frames: env_usize("LTX_OVERLAP_FRAMES", (tf / 2).max(1) as usize) as i32,
            }),
        })
    } else {
        None
    };

    // The largest-tile output extents in the cost model's convention (min(tile, out_dim) per axis); a
    // None axis means "not tiled" = full extent. Matches `estimated_ltx_decode_peak_gib`'s arguments.
    let out_vox = (out_f as i64) * (out_h as i64) * (out_w as i64);
    let (tile_f, tile_h, tile_w) = match &cfg {
        Some(c) => (
            c.temporal
                .map(|t| (t.tile_frames as i64).min(out_f as i64))
                .unwrap_or(out_f as i64),
            c.spatial
                .map(|s| (s.tile_px as i64).min(out_h as i64))
                .unwrap_or(out_h as i64),
            c.spatial
                .map(|s| (s.tile_px as i64).min(out_w as i64))
                .unwrap_or(out_w as i64),
        ),
        None => (out_f as i64, out_h as i64, out_w as i64),
    };
    let tile_vox = tile_f * tile_h * tile_w;

    let baseline_mib = used_mib(gpu).unwrap_or(0);
    println!(
        "\n=== ltx sweep [gpu {gpu}]: out {out_w}x{out_h}x{out_f}  latent[z{LATENT_CHANNELS},T{t_lat},{h_lat},{w_lat}]  \
         tiled={}  cfg={cfg:?}  baseline={baseline_mib} MiB ===",
        cfg.is_some(),
    );

    let sampler = PeakSampler::start(gpu);
    let t = Instant::now();
    let video = match &cfg {
        Some(c) => vae.decode_tiled(&latent, c).unwrap(),
        None => vae.decode(&latent).unwrap(),
    };
    dev.synchronize().unwrap();
    let secs = t.elapsed().as_secs_f64();
    let peak_mib = sampler.stop();

    // Finiteness / range / shape (cheap sanity — the parity bound lives in vae_tiling_cuda-style tests).
    let v = video.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!(
        v.iter().all(|x| x.is_finite()),
        "decode produced non-finite values"
    );
    assert_eq!(
        video.dims(),
        &[1, 3, out_f as usize, out_h as usize, out_w as usize],
        "unexpected output shape"
    );

    let peak_bytes = (peak_mib as f64) * 1024.0 * 1024.0;
    let peak_gib = gib(peak_bytes);
    println!(
        "[LTX decode] -> {:?}  {secs:.1}s  peak={peak_gib:.2} GiB ({peak_mib} MiB)",
        video.dims()
    );
    // Parse-friendly anchor line (grep `^ANCHOR`): peak vs output/tile voxels → the cost coefficients.
    println!(
        "ANCHOR ltx out_vox={out_vox} tile_vox={tile_vox} peak_gib={peak_gib:.4} peak_mib={peak_mib} \
         baseline_mib={baseline_mib} bytes_per_out_vox={:.2} bytes_per_tile_vox={:.2}",
        peak_bytes / out_vox as f64,
        peak_bytes / tile_vox as f64,
    );
}
