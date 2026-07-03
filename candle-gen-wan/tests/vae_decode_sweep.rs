//! sc-7148 — **Wan2.2 z48 `vae22` video VAE decode CUDA peak-VRAM sweep.** The candle/CUDA analog of
//! mlx-gen-wan's `wedge_sweep.rs` / `vae16_decode_sweep.rs` (sc-4998 / sc-6894): measures the *real*
//! concurrent GPU memory peak of [`WanVae::decode`] (streaming) / [`WanVae::decode_tiled`] /
//! [`WanVae::decode_budgeted`] across a grid of output sizes × **spatial** tile sizes on real
//! Wan2.2-TI2V-5B z48 weights, so the budgeted decode cost model (`estimated_wan22_decode_peak_gib`,
//! the `WAN22_VAE_ACCUM_BYTES_PER_VOXEL` / `WAN22_VAE_FRAME_BYTES_PER_OUT_PX` constants) can be **fit
//! from CUDA measurements** instead of the streaming-aware placeholder it shipped with.
//!
//! Loads only the VAE component of a Wan2.2-TI2V-5B snapshot (not the 33 GB DiT), so it is cheap.
//! One config **per process** (env-driven) so an OOM on the largest configs kills only this process and
//! the driving shell loop keeps the earlier anchors. Synthetic (random) latents — cost/scaling, not
//! parity (parity lives in `vae_tiling_cuda.rs`).
//!
//! **Streaming model note:** candle's `WanVae::decode` decodes one latent frame at a time, so the
//! temporal axis is already memory-bounded — the per-tile activation spike scales with one output
//! *frame's* area (`tile_h·tile_w`), not the whole tile's volume. The sweep therefore tiles **only the
//! spatial axes** and reports `bytes_per_frame_px` (fits `FRAME`) alongside `bytes_per_out_vox`
//! (bounds `ACCUM`). Peak is sampled device-wide via `nvidia-smi` (see `candle_gen::testkit`).
//!
//! ```text
//! set CUDA_VISIBLE_DEVICES=0
//! set WAN_SNAPSHOT=C:\Users\…\models--Wan-AI--Wan2.2-TI2V-5B-Diffusers\snapshots\<h>
//! set WAN_W=1280& set WAN_H=1280& set WAN_FRAMES=13
//!   cargo test -p candle-gen-wan --features cuda --release --test vae_decode_sweep -- --ignored --nocapture
//! # add WAN_TILE_PX=512 [WAN_OVERLAP_PX=64] for a fixed spatial-tile run
//! # add WAN_BUDGETED=1 [WAN_VAE_BUDGET_GIB=48] to exercise the production budgeted selector
//! # add WAN_GPU=0 to pick which GPU ordinal nvidia-smi samples (default 0; pair with CUDA_VISIBLE_DEVICES)
//! ```
#![cfg(feature = "cuda")]

use std::path::PathBuf;
use std::time::Instant;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::tiling::TilingConfig;
use candle_gen::testkit::{used_mib, PeakSampler};
use candle_gen_wan::config::VaeConfig;
use candle_gen_wan::vae::{auto_tiling_budgeted_wan22, WanVae};

fn env_usize(var: &str, default: usize) -> usize {
    std::env::var(var)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn gib(bytes: f64) -> f64 {
    bytes / (1024.0 * 1024.0 * 1024.0)
}

/// Load just the VAE from `$WAN_SNAPSHOT/vae/diffusion_pytorch_model.safetensors` onto cuda:`gpu`.
fn load_vae(gpu: usize) -> Option<(WanVae, Device)> {
    let snap = std::env::var("WAN_SNAPSHOT").ok()?;
    let dev = Device::new_cuda(gpu).expect("cuda device");
    let f = PathBuf::from(snap)
        .join("vae")
        .join("diffusion_pytorch_model.safetensors");
    // SAFETY: mmap of a read-only, process-owned weight file resolved from `$WAN_SNAPSHOT`; not
    // mutated behind the mapping — the standard candle loading path.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[f], DType::F32, &dev).unwrap() };
    let vae = WanVae::new(&VaeConfig::ti2v_5b(), vb).unwrap();
    Some((vae, dev))
}

#[test]
#[ignore = "needs a Wan2.2-TI2V-5B snapshot dir (WAN_SNAPSHOT) + CUDA; GPU-heavy"]
fn wan_z48_vae_decode_sweep() {
    let gpu = env_usize("WAN_GPU", 0);
    let Some((vae, dev)) = load_vae(gpu) else {
        eprintln!("skip: set WAN_SNAPSHOT to a Wan2.2-TI2V-5B snapshot dir");
        return;
    };
    let w_out = env_usize("WAN_W", 1280) as i32;
    let h_out = env_usize("WAN_H", 1280) as i32;
    let frames = env_usize("WAN_FRAMES", 13) as i32;
    // z48 vae22: spatial ×16, temporal ×4 **causal** (out_f = 1 + (T_lat−1)·4 ⇒ frames = 1 + 4·k).
    assert_eq!(
        (frames - 1) % 4,
        0,
        "WAN_FRAMES must be 1 + 4·k (got {frames})"
    );
    assert_eq!(h_out % 16, 0, "WAN_H must be a multiple of 16");
    assert_eq!(w_out % 16, 0, "WAN_W must be a multiple of 16");
    let (t_lat, h_lat, w_lat) = ((frames - 1) / 4 + 1, h_out / 16, w_out / 16);

    // Warm the CUDA context / cuBLAS handles with a tiny streaming decode so the measured peak reflects
    // the real decode's working set, not one-time context creation.
    let warm = Tensor::randn(0f32, 1f32, (1, 48, 1, 4, 4), &dev).unwrap();
    let _ = vae.decode(&warm).unwrap();
    dev.synchronize().unwrap();

    // Synthetic latent [B=1, 48, T_lat, H_lat, W_lat] (random — cost only, not parity).
    let latent = Tensor::randn(
        0f32,
        1f32,
        (1, 48, t_lat as usize, h_lat as usize, w_lat as usize),
        &dev,
    )
    .unwrap();
    let (out_f, out_h, out_w) = (1 + (t_lat - 1) * 4, h_lat * 16, w_lat * 16);

    // Tile selection: WAN_BUDGETED=1 exercises the production `auto_tiling_budgeted_wan22` selector
    // (honors WAN_VAE_BUDGET_GIB); else WAN_TILE_PX sets a fixed spatial tile (for anchor fitting); else
    // single-pass. The candle decode streams temporally, so there is no temporal tiling here.
    let tile_px = env_usize("WAN_TILE_PX", 0) as i32;
    let cfg: Option<TilingConfig> = if env_usize("WAN_BUDGETED", 0) == 1 {
        auto_tiling_budgeted_wan22(out_h, out_w, out_f)
            .expect("wan z48 decode fits the budget (catchable error if not)")
    } else if tile_px > 0 {
        Some(TilingConfig::spatial_only(
            tile_px,
            env_usize("WAN_OVERLAP_PX", 64) as i32,
        ))
    } else {
        None
    };

    // Largest-tile extents in the cost model's convention (min(tile, out_dim) per axis); a None spatial
    // axis means full extent. `tile_f` is unused by the streaming model but kept for the anchor line.
    let out_vox = (out_f as i64) * (out_h as i64) * (out_w as i64);
    let (tile_h, tile_w) = match &cfg {
        Some(c) => (
            c.spatial
                .map(|s| (s.tile_px as i64).min(out_h as i64))
                .unwrap_or(out_h as i64),
            c.spatial
                .map(|s| (s.tile_px as i64).min(out_w as i64))
                .unwrap_or(out_w as i64),
        ),
        None => (out_h as i64, out_w as i64),
    };
    let frame_px = tile_h * tile_w;

    let baseline_mib = used_mib(gpu).unwrap_or(0);
    println!(
        "\n=== wan z48 sweep [gpu {gpu}]: out {out_w}x{out_h}x{out_f}  latent[z48,T{t_lat},{h_lat},{w_lat}]  \
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

    // Finiteness / range / shape (cheap sanity — the parity bound lives in vae_tiling_cuda.rs).
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
        "[WAN z48 decode] -> {:?}  {secs:.1}s  peak={peak_gib:.2} GiB ({peak_mib} MiB)",
        video.dims()
    );
    // Parse-friendly anchor line (grep `^ANCHOR`): peak vs out voxels (ACCUM floor) and per-frame px
    // (the streaming per-tile FRAME term).
    println!(
        "ANCHOR wan out_vox={out_vox} frame_px={frame_px} peak_gib={peak_gib:.4} peak_mib={peak_mib} \
         baseline_mib={baseline_mib} bytes_per_out_vox={:.2} bytes_per_frame_px={:.2}",
        peak_bytes / out_vox as f64,
        peak_bytes / frame_px as f64,
    );
}
