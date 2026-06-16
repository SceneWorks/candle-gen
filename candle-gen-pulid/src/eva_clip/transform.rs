//! EVA-CLIP input transform: the `face_features_image` (512² aligned, background-whitened grayscale,
//! NCHW f32 in [0,1] from `candle-gen-face`) is resized to 336² and normalized with the OpenAI/EVA
//! mean/std before the ViT.
//!
//! The reference is torchvision `resize(t, 336, BICUBIC)` on a **float** tensor — antialiased
//! (downscale) Keys-cubic (a=-0.5), computed in float (NO u8 quantization, NO clamp). This is a distinct
//! path from the core u8 PIL bicubic, so the float resize is hand-rolled here (verbatim from the MLX
//! sibling). The downstream gate is ArcFace-cosine (cross-encoder), so a faithful float bicubic — not
//! byte-parity — is what's required.

use candle_core::{Device, Tensor};

use candle_gen::{CandleError, Result};

/// OpenAI/EVA normalization constants (`eva_clip/constants.py`).
pub const EVA_MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
pub const EVA_STD: [f32; 3] = [0.268_629_54, 0.261_302_6, 0.275_777_1];

/// Keys cubic (a = -0.5), support 2.0 — the bicubic filter (matches PIL/torchvision).
fn cubic(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

/// torchvision/PIL `precompute_coeffs`: antialias by scaling the filter support when downscaling, clamp
/// the window, renormalize. Returns `(window_start, weights)` per output pixel (f64).
fn coeffs(in_size: usize, out_size: usize, antialias: bool) -> Vec<(usize, Vec<f64>)> {
    let scale = in_size as f64 / out_size as f64;
    let filterscale = if antialias { scale.max(1.0) } else { 1.0 };
    let support = 2.0 * filterscale;
    let mut out = Vec::with_capacity(out_size);
    for xx in 0..out_size {
        let center = (xx as f64 + 0.5) * scale;
        let xmin = ((center - support + 0.5).floor() as i64).max(0) as usize;
        let xmax = ((center + support + 0.5).floor() as i64).min(in_size as i64) as usize;
        let mut weights = Vec::with_capacity(xmax - xmin);
        let mut total = 0.0;
        for x in xmin..xmax {
            let w = cubic((x as f64 - center + 0.5) / filterscale);
            weights.push(w);
            total += w;
        }
        if total != 0.0 {
            for w in &mut weights {
                *w /= total;
            }
        }
        out.push((xmin, weights));
    }
    out
}

/// Separable float bicubic resize of an HWC f32 image (3 channels), accumulated in f64. No
/// quantization or clamp — torchvision's float `antialias=True` bicubic.
pub fn resize_bicubic_f32(
    src: &[f32],
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
) -> Vec<f32> {
    let c = 3usize;
    // Horizontal: (in_h, in_w) → (in_h, out_w)
    let hc = coeffs(in_w, out_w, true);
    let mut horiz = vec![0f32; in_h * out_w * c];
    for y in 0..in_h {
        for (xx, (xmin, w)) in hc.iter().enumerate() {
            for ch in 0..c {
                let mut acc = 0.0f64;
                for (k, &wk) in w.iter().enumerate() {
                    acc += src[(y * in_w + xmin + k) * c + ch] as f64 * wk;
                }
                horiz[(y * out_w + xx) * c + ch] = acc as f32;
            }
        }
    }
    // Vertical: (in_h, out_w) → (out_h, out_w)
    let vc = coeffs(in_h, out_h, true);
    let mut out = vec![0f32; out_h * out_w * c];
    for (yy, (ymin, w)) in vc.iter().enumerate() {
        for x in 0..out_w {
            for ch in 0..c {
                let mut acc = 0.0f64;
                for (k, &wk) in w.iter().enumerate() {
                    acc += horiz[((ymin + k) * out_w + x) * c + ch] as f64 * wk;
                }
                out[(yy * out_w + x) * c + ch] = acc as f32;
            }
        }
    }
    out
}

/// Full EVA transform: NCHW `[1, 3, H, W]` f32 in [0,1] → resized to `size²` (float bicubic) and
/// normalized `(x - mean) / std` per channel. Returns NCHW `[1, 3, size, size]` on the input's device.
pub fn eva_transform(ffi_nchw: &Tensor, size: usize) -> Result<Tensor> {
    let (b, _c, in_h, in_w) = ffi_nchw.dims4()?;
    if b != 1 {
        return Err(CandleError::Msg(format!(
            "eva_transform handles a single image, got batch {b}"
        )));
    }
    // Read the NCHW image back as host HWC-interleaved f32.
    let src: Vec<f32> = ffi_nchw
        .to_device(&Device::Cpu)?
        .to_dtype(candle_core::DType::F32)?
        .permute((0, 2, 3, 1))? // [1,H,W,3]
        .contiguous()?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let mut resized = resize_bicubic_f32(&src, in_h, in_w, size, size);
    // Per-channel (x - mean) / std, in place on the HWC buffer.
    for px in resized.chunks_exact_mut(3) {
        for ch in 0..3 {
            px[ch] = (px[ch] - EVA_MEAN[ch]) / EVA_STD[ch];
        }
    }
    let hwc = Tensor::from_vec(resized, (1, size, size, 3), &Device::Cpu)?;
    Ok(hwc
        .permute((0, 3, 1, 2))?
        .contiguous()?
        .to_device(ffi_nchw.device())?)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A non-single-image batch is reported via `Result`; a single image resizes/normalizes to
    /// `[1, 3, size, size]`.
    #[test]
    fn eva_transform_rejects_batch_and_accepts_single() {
        let dev = Device::Cpu;
        let single = Tensor::full(0.5f32, (1, 3, 4, 4), &dev).unwrap();
        let out = eva_transform(&single, 8).unwrap();
        assert_eq!(out.dims(), &[1, 3, 8, 8]);
        // A constant 0.5 input resizes to a constant and normalizes to (0.5-mean)/std per channel.
        let v = out.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0]; // channel 0, pixel 0
        assert!((v - (0.5 - EVA_MEAN[0]) / EVA_STD[0]).abs() < 1e-4);

        let batched = Tensor::full(0.5f32, (2, 3, 4, 4), &dev).unwrap();
        let err = eva_transform(&batched, 8).unwrap_err().to_string();
        assert!(err.contains("single image"), "got: {err}");
    }
}
