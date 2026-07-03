//! FLUX.2 latent geometry and the flow-match-Euler schedule. Port of `mlx-gen-flux2`'s `pipeline.rs`
//! plus the core `scheduler`'s empirical-mu sigma-shift. All functions here are weight-free and pure,
//! so they are unit-tested on CPU without a GPU or a checkpoint.
//!
//! Geometry: an `W×H` image maps to a VAE latent `[1, 32, H/8, W/8]`, a 2×2 patchify folds that into
//! the `[1, 128, H/16, W/16]` transformer space, and a pack flattens the spatial grid into a token
//! sequence `[1, (H/16)·(W/16), 128]`. txt2img samples noise directly in the packed 128-ch space.

use candle_gen::candle_core::{Device, Result, Tensor};
use rand::{rngs::StdRng, SeedableRng};

use crate::config::Flux2Config;

/// Packed-latent spatial dims `(lat_h, lat_w) = (H/16, W/16)` — the transformer token grid.
pub fn latent_dims(width: u32, height: u32) -> (usize, usize) {
    ((height / 16) as usize, (width / 16) as usize)
}

/// Image (= packed token) sequence length `(H/16)·(W/16)` (1024² → 4096).
pub fn image_seq_len(width: u32, height: u32) -> usize {
    let (h, w) = latent_dims(width, height);
    h * w
}

/// Deterministic, launch-portable packed initial noise `[1, seq, 128]` (sc-3673 parity): N(0,1) from
/// a fixed-algorithm CPU RNG seeded by `seed`, sampled in `[1, 128, lat_h, lat_w]` order then packed
/// (`reshape [1,128,h·w] → transpose [1,h·w,128]`) and moved to `device`. NOT candle's CUDA `randn`.
pub fn create_noise(
    cfg: &Flux2Config,
    seed: u64,
    width: u32,
    height: u32,
    device: &Device,
) -> Result<Tensor> {
    let (lat_h, lat_w) = latent_dims(width, height);
    let c = cfg.in_channels;
    let n = c * lat_h * lat_w;
    let mut rng = StdRng::seed_from_u64(seed);
    let data = candle_gen::seeded_normal_vec(&mut rng, n);
    let chw = Tensor::from_vec(data, (1, c, lat_h, lat_w), &Device::Cpu)?;
    // pack: [1, C, H, W] -> [1, C, H*W] -> [1, H*W, C]
    let packed = chw.reshape((1, c, lat_h * lat_w))?.transpose(1, 2)?;
    packed.contiguous()?.to_device(device)
}

/// Pack a spatial latent `[1, C, h, w]` (NCHW) into the transformer token sequence `[1, h·w, C]` —
/// the same `reshape → transpose` fold [`create_noise`] applies inline, exposed for the edit path's
/// VAE-encoded reference latents.
pub fn pack_nchw(x: &Tensor) -> Result<Tensor> {
    let (b, c, h, w) = x.dims4()?;
    x.reshape((b, c, h * w))?.transpose(1, 2)?.contiguous()
}

/// Unpack packed latents `[1, seq, C]` back to `[1, C, lat_h, lat_w]` (NCHW) for the VAE.
pub fn unpack_latents(packed: &Tensor, width: u32, height: u32) -> Result<Tensor> {
    let (lat_h, lat_w) = latent_dims(width, height);
    let (b, _seq, c) = packed.dims3()?;
    packed
        .reshape((b, lat_h, lat_w, c))?
        .permute((0, 3, 1, 2))?
        .contiguous()
}

/// Image position ids `[t, h, w, layer]`, row-major over the `(lat_h, lat_w)` grid at temporal
/// coordinate `t_coord` (`layer = 0`). The edit path offsets each reference at `t = 10 + 10·i` so the
/// RoPE separates the reference tokens from the `t = 0` target grid. Returned host-side (the RoPE
/// table is built from these).
pub fn prepare_grid_ids_t(lat_h: usize, lat_w: usize, t_coord: i64) -> Vec<[i64; 4]> {
    let mut ids = Vec::with_capacity(lat_h * lat_w);
    for h in 0..lat_h {
        for w in 0..lat_w {
            ids.push([t_coord, h as i64, w as i64, 0]);
        }
    }
    ids
}

/// Image position ids `[t, h, w, layer]`, row-major over the `(lat_h, lat_w)` grid. txt2img uses
/// `t = 0` and `layer = 0`. Returned host-side (the RoPE table is built from these).
pub fn prepare_grid_ids(lat_h: usize, lat_w: usize) -> Vec<[i64; 4]> {
    prepare_grid_ids_t(lat_h, lat_w, 0)
}

/// Text position ids `[0, 0, 0, token_index]` for a `seq`-length prompt.
pub fn prepare_text_ids(seq: usize) -> Vec<[i64; 4]> {
    (0..seq).map(|t| [0, 0, 0, t as i64]).collect()
}

// --- Flow-match Euler schedule (empirical-mu sigma shift) ---

/// The FLUX.2 empirical-mu shift for `image_seq_len` and `num_steps`, ported from gen-core's
/// `compute_mu`. Piecewise-linear in `num_steps` below seq 4300, linear-in-seq above.
pub fn compute_mu(seq_len: usize, num_steps: usize) -> f32 {
    // Empirical-mu constants (from gen-core's `compute_mu`); computed in f64 to carry their full
    // precision, cast to f32 at the end.
    let (a1, b1) = (8.738_095_24e-5_f64, 1.898_333_33_f64);
    let (a2, b2) = (0.000_169_27_f64, 0.456_666_66_f64);
    let seq = seq_len as f64;
    let mu = if seq_len > 4300 {
        a2 * seq + b2
    } else {
        let m_200 = a2 * seq + b2;
        let m_10 = a1 * seq + b1;
        let a = (m_200 - m_10) / 190.0;
        let b = m_200 - 200.0 * a;
        a * num_steps as f64 + b
    };
    mu as f32
}

/// The exponential time-shift `e / (e + (1/σ − 1))` with `e = exp(mu)` — the FLUX.2 sigma warp.
fn time_shift_exponential(mu: f32, sigma: f32) -> f32 {
    let e = mu.exp();
    e / (e + (1.0 / sigma - 1.0))
}

/// Build the descending sigma schedule of length `num_steps + 1`: `num_steps` linspace points from
/// `1.0` to `1/num_steps` warped by [`time_shift_exponential`], then a trailing `0.0`.
pub fn build_sigmas(num_steps: usize, mu: f32) -> Vec<f32> {
    let n = num_steps as f32;
    let (start, end) = (1.0f32, 1.0f32 / n);
    let mut sigmas = Vec::with_capacity(num_steps + 1);
    for i in 0..num_steps {
        let t = if num_steps == 1 {
            start
        } else {
            start + (end - start) * (i as f32) / (n - 1.0)
        };
        sigmas.push(time_shift_exponential(mu, t));
    }
    sigmas.push(0.0);
    sigmas
}

/// The native flow-match sigma schedule for a render (length `steps+1`, descending, trailing `0.0`).
/// The empirical-mu shift is fit from the image sequence length via [`compute_mu`]. This is the N1
/// native schedule the unified sampler (`candle_gen::run_flow_sampler`) integrates over.
pub fn schedule(steps: usize, width: u32, height: u32) -> Vec<f32> {
    let mu = compute_mu(image_seq_len(width, height), steps);
    build_sigmas(steps, mu)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_matches_fork() {
        assert_eq!(latent_dims(1024, 1024), (64, 64));
        assert_eq!(image_seq_len(1024, 1024), 4096);
        assert_eq!(image_seq_len(512, 768), (768 / 16) * (512 / 16));
    }

    #[test]
    fn grid_ids_are_row_major_with_zero_t_and_layer() {
        let ids = prepare_grid_ids(2, 3);
        assert_eq!(ids.len(), 6);
        assert_eq!(ids[0], [0, 0, 0, 0]);
        assert_eq!(ids[1], [0, 0, 1, 0]);
        assert_eq!(ids[3], [0, 1, 0, 0]);
        assert_eq!(ids[5], [0, 1, 2, 0]);
    }

    #[test]
    fn grid_ids_t_offsets_the_temporal_axis() {
        // The edit path tags reference grids at t = 10 + 10·i; the t=0 wrapper is unchanged.
        let ids = prepare_grid_ids_t(2, 2, 10);
        assert_eq!(ids.len(), 4);
        assert_eq!(ids[0], [10, 0, 0, 0]);
        assert_eq!(ids[1], [10, 0, 1, 0]);
        assert_eq!(ids[2], [10, 1, 0, 0]);
        assert_eq!(ids[3], [10, 1, 1, 0]);
        assert_eq!(prepare_grid_ids(2, 2), prepare_grid_ids_t(2, 2, 0));
    }

    #[test]
    fn pack_nchw_folds_spatial_into_sequence() {
        // [1, C=2, h=2, w=2] -> [1, h·w=4, C=2]: token s (row-major over h,w) carries both channels.
        let data: Vec<f32> = vec![
            0., 1., 2., 3., // channel 0 over the 2×2 grid
            10., 11., 12., 13., // channel 1
        ];
        let x = Tensor::from_vec(data, (1, 2, 2, 2), &Device::Cpu).unwrap();
        let packed = pack_nchw(&x).unwrap();
        assert_eq!(packed.dims(), &[1, 4, 2]);
        let v = packed.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // token0 = [c0@(0,0), c1@(0,0)] = [0,10]; then [1,11], [2,12], [3,13].
        assert_eq!(v, vec![0., 10., 1., 11., 2., 12., 3., 13.]);
    }

    #[test]
    fn text_ids_index_the_layer_axis() {
        let ids = prepare_text_ids(4);
        assert_eq!(
            ids,
            vec![[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 3]]
        );
    }

    #[test]
    fn noise_is_deterministic_and_packed() {
        let cfg = Flux2Config::klein_9b();
        let a = create_noise(&cfg, 42, 256, 256, &Device::Cpu).unwrap();
        let b = create_noise(&cfg, 42, 256, 256, &Device::Cpu).unwrap();
        // [1, (256/16)^2 = 256, 128]
        assert_eq!(a.dims(), &[1, 256, 128]);
        let av = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let bv = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(av, bv, "same seed → identical noise");
        let c = create_noise(&cfg, 43, 256, 256, &Device::Cpu).unwrap();
        let cv = c.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_ne!(av, cv, "different seed → different noise");
    }

    #[test]
    fn unpack_roundtrips_noise_shape() {
        let cfg = Flux2Config::klein_9b();
        let packed = create_noise(&cfg, 1, 256, 256, &Device::Cpu).unwrap();
        let un = unpack_latents(&packed, 256, 256).unwrap();
        assert_eq!(un.dims(), &[1, 128, 16, 16]);
    }

    #[test]
    fn sigmas_descend_from_below_one_to_zero() {
        let sigmas = schedule(4, 1024, 1024);
        assert_eq!(sigmas.len(), 5);
        assert!(sigmas[0] > 0.0 && sigmas[0] <= 1.0, "start {}", sigmas[0]);
        assert!(sigmas[4].abs() < 1e-9, "terminal sigma is 0");
        for w in sigmas.windows(2) {
            assert!(w[0] > w[1], "sigmas must strictly descend: {sigmas:?}");
        }
    }

    #[test]
    fn compute_mu_branches() {
        // Below the 4300 cutoff: piecewise-linear in num_steps.
        let mu_small = compute_mu(4096, 4);
        assert!(mu_small.is_finite());
        // Above the cutoff: pure linear-in-seq (independent of num_steps).
        let a = compute_mu(5000, 4);
        let b = compute_mu(5000, 50);
        assert!((a - b).abs() < 1e-6, "large-seq mu is step-independent");
    }
}
