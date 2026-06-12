# candle-gen

Rust-native generative image (and, later, video) model inference on
[candle](https://github.com/huggingface/candle) — the **Windows/CUDA sibling** of
[`mlx-gen`](https://github.com/michaeltrefry/mlx-gen) (Apple MLX). Both crates implement the **same**
backend-neutral [`gen_core`](https://github.com/michaeltrefry/mlx-gen/tree/main/gen-core) contract
(SceneWorks epic 3720), so a consumer pins one backend by git SHA, links its provider crates, and
calls the identical `Generator` / registry API regardless of which tensor backend is underneath.

> **Status: SDXL txt2img implemented on the Candle/CUDA lane.** `SdxlGenerator::generate` runs the
> full pipeline — dual CLIP → UNet (real CFG) → f16 VAE — for both `sdxl` and `realvisxl`
> (sc-3675, RealVisXL + parity tests sc-3677). Output is deterministic and launch-portable per seed
> (CPU-seeded noise + non-ancestral DDIM, sc-3673). Perf/VRAM work has landed: f16 CLIP + optional
> flash-attention (sc-3674), VAE tiling + staged CLIP free for torch-parity peak VRAM at 1024²
> (sc-4987), and UNet/VAE component caching across `generate` calls (sc-5037). The provider still
> self-registers into the shared `gen_core` inventory registry, with the
> `CandleError → gen_core::Error` bridge + device plumbing wired (scaffold sc-4946).

## Layout

```
candle-gen/                 # workspace root
  candle-gen/               # core crate: re-exports gen_core + candle; device/dtype helpers;
                            #   CandleError -> gen_core::Error bridge
  candle-gen-sdxl/          # SDXL provider crate: Generator impl + descriptor + inventory::submit!
  scripts/
    check-gen-core-skew.sh  # version-skew gate: fails if >1 sceneworks-gen-core resolves
  .github/workflows/ci.yml  # macOS/Linux fmt+clippy+check+test + skew self-test (CUDA lane TODO)
```

A provider crate self-registers just by being linked (`inventory::submit!`), so adding a model is
purely additive — there is no central match statement to edit. `candle-gen-sdxl` registers a single
descriptor under the id `"sdxl"` (the SceneWorks worker maps both `sdxl` and `realvisxl` to engine
id `"sdxl"`), with `backend: "candle"`.

## Backends / features

The default build is **CPU** (`candle-core`'s default) and works on macOS with no extra features.

| feature      | backend                | platform        | in `default`? |
|--------------|------------------------|-----------------|---------------|
| *(none)*     | CPU                    | all (Mac dev)   | yes           |
| `metal`      | Apple Metal GPU        | macOS           | no            |
| `cuda`       | NVIDIA CUDA            | Windows/Linux   | no            |
| `flash-attn` | implies `cuda` (TODO)  | Windows/CUDA    | no            |

`cuda` / `flash-attn` need the CUDA toolkit and **do not build on Mac**; all CUDA-only code is gated
behind `#[cfg(feature = "cuda")]`. `flash-attn` currently just implies `cuda` — the fused kernels
need the separate `candle-flash-attn` crate, wired in a later slice on the Windows box.

## gen-core pinning (read before bumping)

`sceneworks-gen-core` is pinned by **git SHA** in the root `Cargo.toml`
(`[workspace.dependencies]`) to the **same rev the SceneWorks worker pins**. Everything is
SHA-pinned: if candle-gen resolves gen-core at rev A while the worker resolves rev B, cargo silently
builds **both**, the provider crate registers into one `inventory` registry while the worker queries
the other, and the symptom is **"engine not found" at runtime** (not a compile error). Run the gate:

```bash
bash scripts/check-gen-core-skew.sh            # checks candle-gen's build graph
bash scripts/check-gen-core-skew.sh --self-test  # proves the gate fires on canned skew
```

When bumping the gen-core pin, bump it in lockstep with the worker's `mlx-gen` + `sceneworks-gen-core`
pins.

## Develop

```bash
cargo fmt --all
cargo check --workspace                 # CPU (Mac default)
cargo check --workspace --features metal  # Metal backend builds
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace                  # registry-resolution + bridge tests
```

The candle version this scaffold settled on is recorded in `[workspace.dependencies]`
(`candle-core` / `candle-nn`).
