//! MRoPE position ids + 4-D flex attention mask match the reference, bit-exact (sc-10995, candle port
//! of the mlx lane's `process_parity`). Reads the shared `process_golden.safetensors` — integer /
//! boolean outputs, so the match is **exact**. CPU, weight-free.

mod common;

use common::Golden;

use candle_gen_bernini::process::{build_attention_mask_4d, mrope_position_ids, MRopeConfig};

fn grids(g: &Golden, key: &str) -> Vec<[i64; 3]> {
    if !g.has(key) {
        return Vec::new();
    }
    let s = g.i64(key);
    let n = g.shape(key)[0];
    (0..n)
        .map(|i| [s[i * 3], s[i * 3 + 1], s[i * 3 + 2]])
        .collect()
}

#[test]
fn process_matches_reference() {
    let g = Golden::load("process_golden");
    let cfg = MRopeConfig {
        spatial_merge_size: g.meta_req("spatial_merge_size").parse().unwrap(),
        tokens_per_second: g.meta_req("tokens_per_second").parse().unwrap(),
        image_token_id: g.meta_req("image_token_id").parse().unwrap(),
        video_token_id: g.meta_req("video_token_id").parse().unwrap(),
        vision_start_token_id: g.meta_req("vision_start_token_id").parse().unwrap(),
    };
    let tasks: Vec<&str> = g.meta_req("tasks").split(',').collect();
    assert!(!tasks.is_empty(), "fixture lists tasks");

    for task in tasks {
        let input_ids = g.i64(&format!("{task}.input_ids"));
        let image_grid = grids(&g, &format!("{task}.image_grid_thw"));
        let video_grid = grids(&g, &format!("{task}.video_grid_thw"));
        let l = input_ids.len();

        // --- position ids (exact) ---
        let pos = mrope_position_ids(&input_ids, &image_grid, &video_grid, &cfg).unwrap();
        assert_eq!(pos.dims(), &[3, l], "{task} position shape");
        let got: Vec<i64> = pos.flatten_all().unwrap().to_vec1::<i64>().unwrap();
        let want = g.i64(&format!("{task}.position_ids"));
        let pos_mismatch = got.iter().zip(&want).filter(|(a, b)| a != b).count();
        assert_eq!(
            pos_mismatch, 0,
            "{task} position_ids: {pos_mismatch} mismatched"
        );

        // --- 4-D flex mask (compare visibility, exact) ---
        let token_type = g.i32(&format!("{task}.token_type"));
        let token_seg = g.i32(&format!("{task}.token_segment_ids"));
        let mask = build_attention_mask_4d(&token_type, &token_seg).unwrap();
        assert_eq!(mask.dims(), &[1, l, l], "{task} mask shape");
        let mvis: Vec<f32> = mask.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let want_vis = g.i8(&format!("{task}.mask_vis"));
        let mask_mismatch = (0..l * l)
            .filter(|&i| (mvis[i].is_finite() as i8) != want_vis[i])
            .count();
        assert_eq!(
            mask_mismatch, 0,
            "{task} mask: {mask_mismatch} mismatched cells"
        );

        println!("{task}: L={l} position_ids + {l}x{l} mask exact");
    }
}
