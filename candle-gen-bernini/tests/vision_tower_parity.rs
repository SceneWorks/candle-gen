//! sc-10995: the native Qwen2.5-VL vision tower matches the reference forward (near-bit, f32) — candle
//! port of the mlx lane's `vision_tower_parity`. A tiny structurally-faithful ViT (4 blocks, realistic
//! grid geometry `[[1,6,6],[1,4,4]]` so the window-partition / full-vs-windowed mask logic is exercised)
//! with random weights, dumped from the reference by `tools/dump_bernini_vision_tower_golden.py`. Loads
//! `visual.*` weights + `io.pixel_values`/`io.grid_thw` and asserts the tower output against
//! `out.tokens`. CPU, f32 throughout — no cuda/weights.

mod common;

use common::{errors, flat_f32, Golden};

use candle_gen::candle_core::Device;
use candle_gen_bernini::vision::{VisionConfig, VisionTower};

fn config_from_meta(g: &Golden) -> VisionConfig {
    let fullatt: Vec<usize> = g
        .meta_req("fullatt")
        .split(',')
        .map(|s| s.parse::<usize>().unwrap())
        .collect();
    let u = |k: &str| g.meta_req(k).parse::<usize>().unwrap();
    VisionConfig {
        hidden_size: u("hidden"),
        num_heads: u("heads"),
        intermediate_size: u("intermediate"),
        depth: u("depth"),
        fullatt_block_indexes: fullatt,
        spatial_merge_size: u("spatial_merge"),
        window_size: u("window"),
        patch_size: u("patch"),
        temporal_patch_size: u("temporal_patch"),
        in_channels: u("in_chans"),
        out_hidden_size: u("out_hidden"),
    }
}

#[test]
fn vision_tower_matches_reference_f32() {
    let dev = Device::Cpu;
    let g = Golden::load("vision_tower_golden");
    let cfg = config_from_meta(&g);

    let vb = g.var_builder(&dev);
    let tower = VisionTower::new(cfg, vb.pp("visual")).expect("tower");

    let pixel_values = g.tensor("io.pixel_values", &dev);
    // grid_thw is I32 [num_items, 3].
    let grid_flat = g.i32("io.grid_thw");
    let grid: Vec<[usize; 3]> = grid_flat
        .chunks_exact(3)
        .map(|c| [c[0] as usize, c[1] as usize, c[2] as usize])
        .collect();

    let tokens = tower.forward(&pixel_values, &grid).expect("forward");
    let (abs, rel) = errors(&flat_f32(&tokens), &g.f32("out.tokens"));
    println!("vision tower: peak|Δ|={abs:.3e}  peak-rel={rel:.3e}");
    assert!(
        rel < 5e-3,
        "vision tower within the f32 matmul floor (rel {rel:.3e})"
    );
}
