//! PIL-LANCZOS-analog resize for the control-image [`letterbox`](crate::kps::letterbox) (sc-3111).
//!
//! The InstantID / OpenPose control images are a **directional** gate (the IdentityNet aspect rule,
//! sc-2009 — not bit-parity vs PIL), so the `image` crate's Lanczos3 stands in for the MLX
//! `mlx_gen::image::resize_lanczos_u8`. Same signature/contract — RGB8 HWC in → integer-valued f32 HWC
//! out (`[0,255]`) — so `letterbox`'s body is the verbatim MLX port.

use image::{imageops::FilterType, ImageBuffer, Rgb};

/// Lanczos-resample an `src_h × src_w` RGB8 image (`src`, HWC, `len == src_h·src_w·3`) to
/// `dst_h × dst_w`, returning HWC bytes as f32 (integer-valued `[0,255]`) — the candle twin of
/// `mlx_gen::image::resize_lanczos_u8`. Panics on a too-small `src` buffer (caller guarantees the size,
/// matching the MLX helper's contract).
pub(crate) fn resize_lanczos_u8(
    src: &[u8],
    src_h: usize,
    src_w: usize,
    dst_h: usize,
    dst_w: usize,
) -> Vec<f32> {
    let buf: ImageBuffer<Rgb<u8>, Vec<u8>> =
        ImageBuffer::from_raw(src_w as u32, src_h as u32, src.to_vec()).unwrap_or_else(|| {
            panic!(
                "resize_lanczos_u8: src buffer of {} bytes too small for {src_h}×{src_w}×3",
                src.len()
            )
        });
    let resized = image::imageops::resize(&buf, dst_w as u32, dst_h as u32, FilterType::Lanczos3);
    resized.into_raw().into_iter().map(|b| b as f32).collect()
}
