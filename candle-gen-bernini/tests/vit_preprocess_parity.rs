//! sc-10995: the Bernini planner's ViT (Qwen2.5-VL) image preprocessing matches the reference — candle
//! port of the mlx lane's `vit_preprocess_parity`. Golden (`tools/dump_bernini_vit_preprocess_golden.py`):
//!   - `smart_resize` — the target `(h_bar, w_bar)` (factor 28, area clamp, aspect kept, banker's round)
//!     is **bit-exact**.
//!   - patch packing + rescale + normalize — the `do_resize=False` path (dims multiples of 28, so the
//!     `image`-crate resize is identity) is **bit-exact** on a fixed uint8 image.
//!   - `smart_video_nframes` — the frame-index sampling for the ViT/VAE cases.
//!
//! CPU, no cuda/weights.

mod common;

use common::{errors, flat_f32, Golden};

use candle_gen::candle_core::Device;
use candle_gen_bernini::vit_preprocess::{
    preprocess_image, smart_resize, smart_video_nframes, IMAGE_MEAN, IMAGE_STD,
};
use image::RgbImage;

#[test]
fn smart_resize_matches_reference() {
    let g = Golden::load("vit_preprocess_golden");
    let min_pixels: i64 = g.meta_req("min_pixels").parse().unwrap();
    let max_pixels: i64 = g.meta_req("max_pixels").parse().unwrap();
    let factor: i64 = g.meta_req("factor").parse().unwrap();

    let inp = g.i32("smart_resize.in"); // [6, 2]
    let want = g.i32("smart_resize.out");
    for (i, hw) in inp.chunks_exact(2).enumerate() {
        let (rh, rw) = smart_resize(hw[0] as i64, hw[1] as i64, factor, min_pixels, max_pixels);
        assert_eq!(
            (rh as i32, rw as i32),
            (want[i * 2], want[i * 2 + 1]),
            "smart_resize case {i}: in ({},{})",
            hw[0],
            hw[1]
        );
    }
}

#[test]
fn pack_patches_matches_reference() {
    let dev = Device::Cpu;
    let g = Golden::load("vit_preprocess_golden");
    let h: u32 = g.meta_req("pack_h").parse().unwrap();
    let w: u32 = g.meta_req("pack_w").parse().unwrap();
    let min_pixels: i64 = g.meta_req("min_pixels").parse().unwrap();
    let max_pixels: i64 = g.meta_req("max_pixels").parse().unwrap();

    // image_hwc_u8 is I32 [H, W, 3]; rebuild the RgbImage (row-major HWC == RgbImage raw layout).
    let raw: Vec<u8> = g
        .i32("pack.image_hwc_u8")
        .into_iter()
        .map(|x| x as u8)
        .collect();
    let img = RgbImage::from_raw(w, h, raw).expect("rgb image");
    // Dims are multiples of 28, so smart_resize is identity → the `do_resize=False` golden path.
    let (pixel_values, grid) =
        preprocess_image(&img, min_pixels, max_pixels, IMAGE_MEAN, IMAGE_STD, &dev).expect("pack");

    let want_grid = g.i32("pack.grid_thw"); // [1, 3]
    assert_eq!(grid.to_vec(), want_grid, "grid_thw");
    let (abs, rel) = errors(&flat_f32(&pixel_values), &g.f32("pack.pixel_values"));
    println!("pack pixel_values: peak|Δ|={abs:.3e} rel={rel:.3e}");
    assert!(
        abs < 1e-5,
        "pack pixel_values bit-exact (peak|Δ|={abs:.3e})"
    );
}

#[test]
fn smart_video_nframes_matches_reference() {
    let g = Golden::load("vit_preprocess_golden");
    // meta cases: "total,video_fps,fps,frame_factor,max_frames,add_one" (min_frames = None).
    for (i, case) in g.meta_req("nframes_cases").split(';').enumerate() {
        let f: Vec<f64> = case.split(',').map(|x| x.parse::<f64>().unwrap()).collect();
        let idx = smart_video_nframes(
            f[0] as i64,
            f[1],
            f[2],
            Some(f[3] as i64),
            None,
            Some(f[4] as i64),
            f[5] != 0.0,
        );
        let got: Vec<i32> = idx.iter().map(|&x| x as i32).collect();
        assert_eq!(got, g.i32(&format!("nframes.{i}")), "nframes case {i}");
    }
}
