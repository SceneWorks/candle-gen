//! sc-10995: the native Qwen2.5-VL vision tower matches the reference forward (near-bit, f32) — candle
//! port of the mlx lane's `vision_tower_parity`. A tiny structurally-faithful ViT (4 blocks, realistic
//! grid geometry `[[1,6,6],[1,4,4]]` so the window-partition / full-vs-windowed mask logic is exercised)
//! with random weights, dumped from the reference by `tools/dump_bernini_vision_tower_golden.py`. Loads
//! `visual.*` weights + `io.pixel_values`/`io.grid_thw` and asserts the tower output against
//! `out.tokens`. The f32 leg is CPU-only; sc-11150 (F-080) adds an f64-weight CPU leg and a bf16-weight
//! CUDA leg that both feed f32 inputs to prove the tower casts inputs to its weight dtype.

mod common;

use common::{errors, flat_f32, Golden};

use candle_gen::candle_core::{DType, Device};
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
    let grid = grid_from(&g);

    let tokens = tower.forward(&pixel_values, &grid).expect("forward");
    let (abs, rel) = errors(&flat_f32(&tokens), &g.f32("out.tokens"));
    println!("vision tower: peak|Δ|={abs:.3e}  peak-rel={rel:.3e}");
    assert!(
        rel < 5e-3,
        "vision tower within the f32 matmul floor (rel {rel:.3e})"
    );
}

fn grid_from(g: &Golden) -> Vec<[usize; 3]> {
    g.i32("io.grid_thw")
        .chunks_exact(3)
        .map(|c| [c[0] as usize, c[1] as usize, c[2] as usize])
        .collect()
}

/// sc-11150 (F-080): the production planner loads the tower from a `PLANNER_DTYPE` (bf16) `VarBuilder`
/// but feeds it **f32** pixels/RoPE/masks. Before the dtype-contract fix, `patch_embed.forward(f32
/// pixels)` against non-f32 weights hard-errored at the first ViT encode (candle matmul rejects mixed
/// dtypes), so every conditioned request failed.
///
/// CPU has no bf16 matmul kernel, so this leg exercises the *same* "input dtype ≠ weight dtype → the
/// tower casts inputs to the weight dtype" contract with an **f64** tower fed the **f32** fixture
/// inputs unchanged. It asserts the forward both *runs* (no dtype fault) and returns the weight dtype.
/// The bf16 production dtype itself is covered by the CUDA leg below.
#[test]
fn vision_tower_casts_inputs_to_weight_dtype_f64_on_cpu() {
    let dev = Device::Cpu;
    let g = Golden::load("vision_tower_golden");
    let cfg = config_from_meta(&g);

    let vb = g.var_builder_dtype(&dev, DType::F64);
    let tower = VisionTower::new(cfg, vb.pp("visual")).expect("tower");

    // f32 pixels, exactly like `vit_preprocess` hands the planner.
    let pixel_values = g.tensor("io.pixel_values", &dev);
    assert_eq!(pixel_values.dtype(), DType::F32, "inputs stay f32");
    let grid = grid_from(&g);

    // Must not fault on the mixed f32-input / f64-weight contract, and the tokens take the weight dtype.
    let tokens = tower
        .forward(&pixel_values, &grid)
        .expect("f64 tower forward on f32 inputs must not fault");
    assert_eq!(
        tokens.dtype(),
        DType::F64,
        "tower output takes weight dtype"
    );

    let out: Vec<f32> = flat_f32(&tokens.to_dtype(DType::F32).unwrap());
    let (abs, rel) = errors(&out, &g.f32("out.tokens"));
    println!("vision tower f64-contract: peak|Δ|={abs:.3e}  peak-rel={rel:.3e}");
    assert!(
        rel < 5e-3,
        "f64 vision tower within the f32 matmul floor (rel {rel:.3e})"
    );
}

/// sc-11150 (F-080), CUDA leg: the true production dtype. Build the tower from **bf16** weights (as
/// `BerniniPlanner::load` does via `PLANNER_DTYPE`) and feed the **f32** fixture inputs — the exact
/// dtype pair a conditioned `bernini` request uses. Asserts the mixed-dtype forward runs on CUDA
/// (where bf16 matmul is supported) and returns bf16 within the bf16 rounding floor.
#[cfg(feature = "cuda")]
#[test]
fn vision_tower_bf16_weights_f32_inputs_on_cuda() {
    let dev = Device::new_cuda(0).expect("cuda device 0");
    let g = Golden::load("vision_tower_golden");
    let cfg = config_from_meta(&g);

    let vb = g.var_builder_dtype(&dev, DType::BF16);
    let tower = VisionTower::new(cfg, vb.pp("visual")).expect("tower");

    let pixel_values = g.tensor("io.pixel_values", &dev);
    assert_eq!(pixel_values.dtype(), DType::F32, "inputs stay f32");
    let grid = grid_from(&g);

    let tokens = tower
        .forward(&pixel_values, &grid)
        .expect("bf16 tower forward on f32 inputs must not fault on cuda");
    assert_eq!(
        tokens.dtype(),
        DType::BF16,
        "tower output takes weight dtype"
    );

    let out: Vec<f32> = flat_f32(&tokens.to_dtype(DType::F32).unwrap());
    let (abs, rel) = errors(&out, &g.f32("out.tokens"));
    println!("vision tower bf16: peak|Δ|={abs:.3e}  peak-rel={rel:.3e}");
    assert!(
        rel < 5e-2,
        "bf16 vision tower within the bf16 rounding floor (rel {rel:.3e})"
    );
}
