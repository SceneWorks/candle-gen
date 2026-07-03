//! SCAIL-2 conditioning preprocessing — the **28-channel color-coded mask** build
//! (`extract_and_compress_mask_to_latent`, upstream `wan/utils/scail_utils.py`). The VAE-encode of the
//! reference / pose latents reuses [`candle_gen_wan::vae16::WanVae16`], and the CLIP image encode is
//! [`crate::clip::ScailClip`].

use candle_gen::candle_core::{DType, Result, Tensor};

/// A normalized pixel is "on" when the original `[0,255]` value is ≥ 225, i.e. `(225-127.5)/127.5` in
/// the `[-1,1]` mask space (upstream `_ON_THRESH`).
const ON_THRESH: f64 = (225.0 - 127.5) / 127.5;

/// Default temporal-compression stride (the z16 VAE temporal stride): 4 frames → 1 latent frame,
/// packed into the channel axis (×7 colors = 28).
pub const TEMPORAL_STRIDE: usize = 4;

/// `1 - x`.
fn one_minus(x: &Tensor) -> Result<Tensor> {
    x.affine(-1.0, 1.0)
}

/// `a · b · c` (same-shape elementwise).
fn mul3(a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Tensor> {
    a.broadcast_mul(b)?.broadcast_mul(c)
}

/// Convert a 3-channel RGB color-coded segmentation mask `(3, T, H, W)` in `[-1, 1]` into the
/// 28-channel binary mask latent `(28, T_latent, H/8, W/8)` the DiT's `patch_embedding_mask` consumes
/// — **no VAE**, matching upstream `extract_and_compress_mask_to_latent(additional_spatial_downsample=1)`.
///
/// Pipeline: threshold each channel at [`ON_THRESH`] → the **7 exclusive color classes**
/// (white/red/green/blue/yellow/magenta/cyan as R/G/B AND-products) → **8× area downsample** (exact
/// 8×8 average pool; `H` and `W` must be divisible by 8) → **temporal pack** by `temporal_stride`
/// (frame 0 repeated `stride` times for the lead latent frame; the `stride` frames of each latent step
/// stacked into the channel axis, 7·stride = 28).
pub fn extract_and_compress_mask_to_latent(
    mask: &Tensor,
    temporal_stride: usize,
) -> Result<Tensor> {
    let (_c, t, h, w) = mask.dims4()?;
    // Public boundary: return typed errors on caller-supplied shape/stride mismatches rather than
    // aborting the process (assert) or surfacing an opaque candle reshape failure (sc-9025 / F-041).
    if h % 8 != 0 || w % 8 != 0 {
        return Err(candle_gen::candle_core::Error::Msg(format!(
            "scail2 mask: H,W must be divisible by 8 (got {h}x{w})"
        )));
    }
    if temporal_stride == 0 {
        return Err(candle_gen::candle_core::Error::Msg(
            "scail2 mask: temporal_stride must be >= 1 (got 0)".to_string(),
        ));
    }
    if t == 0 {
        return Err(candle_gen::candle_core::Error::Msg(
            "scail2 mask: temporal dimension T must be >= 1 (got 0)".to_string(),
        ));
    }
    // The temporal pack stacks `stride` frames into the channel axis, so the padded frame count
    // (`stride + (t-1)`) must be a whole multiple of `stride`, i.e. `(t-1) % stride == 0`. Otherwise
    // the reshape below would fail with an opaque candle shape error.
    if (t - 1) % temporal_stride != 0 {
        return Err(candle_gen::candle_core::Error::Msg(format!(
            "scail2 mask: (T-1) must be divisible by temporal_stride \
             (got T={t}, temporal_stride={temporal_stride})"
        )));
    }

    // (3, T, H, W) → (T, 3, H, W), threshold each channel to {0,1}.
    let m = mask
        .permute((1, 0, 2, 3))?
        .to_dtype(DType::F32)?
        .contiguous()?;
    let chans = m.chunk(3, 1)?; // 3 × (T, 1, H, W)
    let r = chans[0].gt(ON_THRESH)?.to_dtype(DType::F32)?;
    let g = chans[1].gt(ON_THRESH)?.to_dtype(DType::F32)?;
    let b = chans[2].gt(ON_THRESH)?.to_dtype(DType::F32)?;
    let (nr, ng, nb) = (one_minus(&r)?, one_minus(&g)?, one_minus(&b)?);

    // 7 exclusive color classes (T, 7, H, W).
    let white = mul3(&r, &g, &b)?;
    let red = mul3(&r, &ng, &nb)?;
    let green = mul3(&nr, &g, &nb)?;
    let blue = mul3(&nr, &ng, &b)?;
    let yellow = mul3(&r, &g, &nb)?;
    let magenta = mul3(&r, &ng, &b)?;
    let cyan = mul3(&nr, &g, &b)?;
    let binary7 = Tensor::cat(&[&white, &red, &green, &blue, &yellow, &magenta, &cyan], 1)?;

    // 8× area downsample = exact 8×8 average pool: (T,7,H,W) → (T,7,H/8,8,W/8,8) → mean over the blocks.
    let (hl, wl) = (h / 8, w / 8);
    let pooled = binary7.reshape((t, 7, hl, 8, wl, 8))?.mean(5)?.mean(3)?; // (T, 7, hl, wl)

    // Temporal pack: lead latent frame repeats frame 0 `stride` times; T_latent groups of `stride`
    // frames stack into the channel axis → 7·stride channels.
    let stride = temporal_stride;
    let t_lat = (t - 1) / stride + 1;
    let frame0 = pooled.narrow(0, 0, 1)?; // (1, 7, hl, wl)
    let lead_refs: Vec<&Tensor> = (0..stride).map(|_| &frame0).collect();
    let lead = Tensor::cat(&lead_refs, 0)?; // (stride, 7, hl, wl)
    let rest = pooled.narrow(0, 1, t - 1)?; // (T-1, 7, hl, wl)
    let padded = Tensor::cat(&[&lead, &rest], 0)?; // (T_latent·stride, 7, hl, wl)

    padded
        .reshape((t_lat, stride * 7, hl, wl))?
        .permute((1, 0, 2, 3))? // (28, T_latent, hl, wl)
        .contiguous()
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Device;

    #[test]
    fn shapes_and_channel_count() {
        // A single 8×8 frame of pure red → 28 channels, 1 latent frame, 1×1 spatial.
        let dev = Device::Cpu;
        // RGB "red" in [-1,1] mask space: R = +1 (on), G = B = -1 (off).
        let mut px = vec![0f32; 3 * 8 * 8];
        for i in 0..(8 * 8) {
            px[i] = 1.0; // R plane on
            px[64 + i] = -1.0; // G off
            px[128 + i] = -1.0; // B off
        }
        let mask = Tensor::from_vec(px, (3, 1, 8, 8), &dev).unwrap();
        let out = extract_and_compress_mask_to_latent(&mask, TEMPORAL_STRIDE).unwrap();
        assert_eq!(out.dims(), &[28, 1, 1, 1]);
        // Class order is [white,red,green,blue,yellow,magenta,cyan]; the lead frame repeats `stride`
        // times into the channel axis, so channel `1` (red of the first packed frame) must be 1.0.
        let v: Vec<f32> = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(
            (v[1] - 1.0).abs() < 1e-5,
            "red class should be on: {:?}",
            &v[..7]
        );
        assert!(v[0].abs() < 1e-5, "white class should be off");
    }

    /// The public boundary must return a typed `Err` (not panic) on a mask whose spatial dims are
    /// not divisible by 8 (sc-9025 / F-041).
    #[test]
    fn rejects_non_divisible_spatial_dims() {
        let dev = Device::Cpu;
        let mask = Tensor::zeros((3, 1, 7, 8), DType::F32, &dev).unwrap(); // H=7 not %8
        let err = extract_and_compress_mask_to_latent(&mask, TEMPORAL_STRIDE).unwrap_err();
        assert!(
            err.to_string().contains("divisible by 8"),
            "unexpected error: {err}"
        );
    }

    /// The public boundary must return a typed `Err` (not an opaque candle reshape error) on a
    /// temporal length that does not align to the stride (sc-9025 / F-041).
    #[test]
    fn rejects_misaligned_temporal_length() {
        let dev = Device::Cpu;
        // T=3, stride=4 → (T-1) % stride == 2 != 0, so the temporal pack cannot tile cleanly.
        let mask = Tensor::zeros((3, 3, 8, 8), DType::F32, &dev).unwrap();
        let err = extract_and_compress_mask_to_latent(&mask, TEMPORAL_STRIDE).unwrap_err();
        assert!(
            err.to_string().contains("divisible by temporal_stride"),
            "unexpected error: {err}"
        );
    }
}
