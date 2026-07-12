# candle-gen

Rust-native generative image (and, later, video) model inference on
[candle](https://github.com/huggingface/candle) — the **Windows/CUDA sibling** of
[`mlx-gen`](https://github.com/michaeltrefry/mlx-gen) (Apple MLX). Both crates implement the **same**
backend-neutral [`gen_core`](https://github.com/michaeltrefry/mlx-gen/tree/main/gen-core) contract
(SceneWorks epic 3720), so a consumer pins one backend by git SHA, links its provider crates, and
calls the identical `Generator` / registry API regardless of which tensor backend is underneath.

> **Status: a maturing multi-family engine on the Candle/CUDA lane.** candle-gen now hosts **20+
> registered model families** across image and video, plus a captioner, text/image embedders, and
> host-side conditioning utilities — every provider self-registers into the shared `gen_core` inventory
> registry (`candle_gen::register_*!`), keeps the deterministic CPU-seeded-noise contract
> (launch-portable per seed), and rides the `CandleError → gen_core::Error` bridge. The families below
> are GPU-validated on an RTX PRO 6000 (Blackwell **sm_120**). The core `candle-gen` crate supplies the
> shared device/dtype seam, weight loaders, seeded noise, the unified sampler/scheduler framework (epic
> 7114), the LoRA/LoKr training + inference-merge harness, and the Q4/Q8 + MLX-packed quantization seam.
>
> **Image generators**
>
> | family | registered engine id(s) | architecture / notes |
> |--------|-------------------------|----------------------|
> | SDXL | `sdxl` | dual-CLIP UNet (`sdxl` + RealVisXL); img2img, LoRA train+merge, quant |
> | Z-Image | `z_image`, `z_image_turbo` | Qwen3 TE → DiT flow-match; base CFG + distilled Turbo; LoRA trainer |
> | FLUX.1 | `flux1_schnell`, `flux1_dev` | CLIP-L + T5-XXL → FLUX DiT flow-match |
> | FLUX.2 | `flux2_klein_9b`, `flux2_dev`, `flux2_dev_edit`, `flux2_dev_control` | from-scratch MMDiT; txt2img + Edit + Fun-Controlnet-Union |
> | Qwen-Image | `qwen_image` | 60-layer dual-stream MMDiT; txt2img + Edit + Fun-Controlnet-Union (VACE) + Lightning + LoRA/LoKr + MLX-packed quant |
> | Anima | `anima_base`, `anima_aesthetic`, `anima_turbo` | Cosmos-Predict2 DiT anime T2I (ER-SDE-3 sampler) |
> | Chroma | `chroma1_hd`, `chroma1_base`, `chroma1_flash` | FLUX-schnell-derived MMDiT |
> | Kolors | `kolors` | SDXL UNet + ChatGLM3-6B encoder |
> | SenseNova-U1 | `sensenova_u1_8b` | NEO-Unify dual-path Qwen3 MoT + flow head; + distilled fast tier |
> | Ideogram 4 | `ideogram_4` | + Turbo + edit |
> | Boogu-Image | `boogu_image`, `boogu_image_turbo`, `boogu_image_edit` | Lumina2/OmniGen2 MMDiT; Base/Turbo/Edit |
> | Krea 2 | `krea_2_turbo`, `krea_2_raw`, `krea_2_edit`, `krea_2_control`, `krea_2_turbo_edit` | 12B DiT; Turbo/Raw/Edit/Control + CFG-free Turbo-edit; LoRA/LoKr trainer |
> | SD3.5 | `sd3_5_large`, `sd3_5_large_turbo`, `sd3_5_medium` | MMDiT + triple text-encoder aggregator + 16-ch VAE |
> | Lens | `lens`, `lens_turbo` | gpt-oss-20B MoE encoder + dual-stream MMDiT + FLUX.2 VAE |
> | Bernini | `bernini`, `bernini_renderer` | ByteDance Wan2.2-T2V-A14B dual-expert renderer (planner + renderer) |
>
> Identity-preserving stacks compose the above rather than registering standalone engines:
> **InstantID** (`candle-gen-instantid`) layers the IP-Adapter/ControlNet + kps/openpose control render +
> `gen_core::FaceEmbedder` onto candle SDXL, and **PuLID-FLUX** (`candle-gen-pulid`) injects an
> EVA02-CLIP + IDFormer perceiver into FLUX.1-dev.
>
> **Video generators**
>
> | family | registered engine id(s) | architecture / notes |
> |--------|-------------------------|----------------------|
> | Wan2.2 | `wan2_2_ti2v_5b`, `wan2_2_t2v_14b`, `wan2_2_i2v_14b`, `wan_vace` | UMT5-XXL + WanTransformer3D (from-scratch conv3d); TI2V/I2V, dual-expert 14B, VACE, LoRA, quant, tiling |
> | LTX-2.3 | `ltx_2_3_distilled` | distilled 22B; Gemma-3-12B TE + connector + AVTransformer3D + CausalVideoAutoencoder |
> | SVD | `svd_xt` | Stable Video Diffusion img2vid-xt |
> | SCAIL-2 | `scail2_14b` | Wan2.1-14B I2V character animation / cross-identity replacement |
> | SeedVR2 | `seedvr2` | one-step DiT + 3D causal video VAE super-resolution upscaler (image **and** video) |
>
> **Captioner, embedders & conditioning utilities**
>
> - **JoyCaption** (`candle-gen-joycaption`) — the first `Captioner`: a from-scratch LLaVA
>   (SigLIP-so400m tower + gelu-MLP projector + Llama-3.1-8B decoder) turning an image into a caption.
> - **CLIP ViT-L/14** (`candle-gen-clip`) — `clip_vit_l14` image embedder + `clip_vit_l14_text` text
>   embedder providers (Dataset Doctor).
> - **Face** (`candle-gen-face`) — SCRFD detector + ArcFace embedder implementing `gen_core::FaceEmbedder`
>   for the InstantID / PuLID ports.
> - **SAM3** (`candle-gen-sam3`) — a Segment-Anything-3 concept segmenter (off-Mac person-track PCS), a
>   plain utility rather than a registered generator.
> - **Depth Anything V2** (`candle-gen-depth`) — monocular depth estimator used as a host-side ControlNet
>   depth preprocessor.
> - **PiD** (`candle-gen-pid`) — NVIDIA Pixel-Diffusion latent→pixel super-resolving decoder seam.
>
> **candle pinned to git main (post-0.10.2)** — REQUIRED for Blackwell sm_120. The crates.io 0.10.2
> release throws `CUDA_ERROR_INVALID_PTX` at the first candle-kernels kernel whenever
> candle-transformers is linked (plain candle-core works). The git rev clears it and is
> source-compatible. See `[workspace.dependencies]`.

## Layout

```
candle-gen/                 # workspace root
  candle-gen/               # core crate: gen_core + candle re-exports; device/dtype seam, weight
                            #   loaders, seeded noise, unified sampler/scheduler, LoRA/LoKr train +
                            #   inference-merge harness, Q4/Q8 + MLX-packed quant; CandleError bridge
  # --- image generators ---
  candle-gen-sdxl/          # SDXL / RealVisXL (dual-CLIP UNet) — txt2img, img2img, LoRA, quant
  candle-gen-z-image/       # Z-Image / Z-Image-Turbo (Qwen3 TE → DiT flow-match) + LoRA trainer
  candle-gen-flux/          # FLUX.1 [schnell]+[dev] (CLIP-L + T5-XXL → FLUX DiT)
  candle-gen-flux2/         # FLUX.2 klein-9B + dev + dev-edit + dev-control (from-scratch MMDiT + Qwen3)
  candle-gen-qwen-image/    # Qwen-Image: txt2img + Edit + Fun-Controlnet-Union + Lightning + LoRA + packed quant
  candle-gen-anima/         # Anima (Cosmos-Predict2 DiT) anime T2I — base/aesthetic/turbo
  candle-gen-chroma/        # Chroma (chroma1_hd/base/flash) FLUX-schnell-derived MMDiT
  candle-gen-kolors/        # Kolors (SDXL UNet + ChatGLM3-6B encoder)
  candle-gen-sensenova/     # SenseNova-U1 (NEO-Unify Qwen3 MoT + flow head) + distilled fast tier
  candle-gen-ideogram/      # Ideogram 4 (+ Turbo, edit)
  candle-gen-boogu/         # Boogu-Image-0.1 (Lumina2/OmniGen2 MMDiT) — base/turbo/edit
  candle-gen-krea/          # Krea 2 — turbo/raw/edit/control/turbo-edit + LoRA/LoKr trainer
  candle-gen-sd3/           # Stable Diffusion 3.5 Large/Large-Turbo/Medium (MMDiT + triple TE + 16-ch VAE)
  candle-gen-lens/          # Lens / Lens-Turbo (gpt-oss-20B MoE encoder + dual-stream MMDiT + FLUX.2 VAE)
  candle-gen-bernini/       # Bernini (ByteDance Wan2.2-T2V-A14B dual-expert renderer)
  candle-gen-instantid/     # InstantID identity-preserving SDXL (IP-Adapter/ControlNet + FaceEmbedder)
  candle-gen-pulid/         # PuLID-FLUX identity injection (EVA02-CLIP + IDFormer into FLUX.1-dev)
  # --- video generators ---
  candle-gen-wan/           # Wan2.2 TI2V-5B + T2V/I2V-14B + VACE (from-scratch conv3d); LoRA, quant, tiling
  candle-gen-ltx/           # LTX-2.3 (distilled 22B) txt2video (Gemma-3-12B TE + connector + AVTransformer3D)
  candle-gen-svd/           # Stable Video Diffusion (img2vid-xt)
  candle-gen-scail2/        # SCAIL-2 (Wan2.1-14B I2V character animation / identity replacement)
  candle-gen-seedvr2/       # SeedVR2 one-step DiT super-resolution upscaler (image + video)
  # --- captioner / embedders / conditioning utilities ---
  candle-gen-joycaption/    # JoyCaption (LLaVA: SigLIP + Llama-3.1-8B) image→caption Captioner
  candle-gen-clip/          # CLIP ViT-L/14 image + text embedder providers
  candle-gen-face/          # SCRFD detector + ArcFace embedder (gen_core FaceEmbedder)
  candle-gen-sam3/          # SAM3 concept segmenter (person-track PCS utility)
  candle-gen-depth/         # Depth Anything V2 monocular depth (ControlNet preprocessor)
  candle-gen-pid/           # NVIDIA PiD pixel-diffusion latent→pixel decoder seam
  vendor/
    candle-kernels/         # local fork: multi-arch fatbin for the CUDA quant path (sc-7544; see Packaging)
  scripts/
    check-gen-core-skew.sh  # version-skew gate: fails if >1 sceneworks-gen-core resolves
    check-cuda.ps1          # local cuda gate: vcvars + cargo build/test --features cuda (run pre-push)
    package-cuda.ps1        # bundle a CUDA build + redist DLLs into dist/ (sc-3676; see Packaging)
  .github/workflows/ci.yml  # macOS/Linux fmt+clippy+check+test + skew self-test; manual Windows/CUDA lane
```

A provider crate self-registers just by being linked (`candle_gen::register_*!` → `inventory::submit!`),
so adding a model is purely additive — there is no central match statement to edit. Each crate submits
one or more `ModelDescriptor`s under a stable engine id (e.g. `candle-gen-sdxl` registers `"sdxl"`, which
the SceneWorks worker maps both `sdxl` and `realvisxl` onto), all with `backend: "candle"`.

## Backends / features

The default build is **CPU** (`candle-core`'s default) and works on macOS with no extra features.

| feature      | backend                | platform        | in `default`? |
|--------------|------------------------|-----------------|---------------|
| *(none)*     | CPU                    | all (Mac dev)   | yes           |
| `metal`      | Apple Metal GPU        | macOS           | no            |
| `cuda`       | NVIDIA CUDA            | Windows/Linux   | no            |

`cuda` needs the CUDA toolkit and **does not build on Mac**; all CUDA-only code is gated behind
`#[cfg(feature = "cuda")]`. A `flash-attn` feature used to exist as a no-op alias (`= ["cuda"]`,
forwarded crate-to-crate) that wired no fused kernels; it was removed in sc-9032. When the
`candle-flash-attn` slice is scheduled, reintroduce it behind real gated code, not a bare alias.

## Packaging (Windows / CUDA) — sc-3676

The goal is **one distributable CUDA worker that runs on every NVIDIA GPU we support, not just the
build box's Blackwell** — the "central fat binary, like torch" model.

### How portability works: PTX-JIT for dense kernels, a multi-arch fatbin for quantized kernels

candle-kernels has **two** compile paths, and they need different portability treatments (verified
against the vendored sources):

- **Dense kernels** build via cudaforge `.build_ptx()` → `nvcc --ptx`, emitting one **`compute_80`
  PTX** (virtual ISA) per kernel. The driver JIT-compiles that PTX to the runtime GPU's native SASS
  at first load, so it runs on **every NVIDIA arch ≥ sm_80** — Ampere (sm_80/86) → Ada (sm_89) →
  Hopper (sm_90) → Blackwell (sm_120) — from a single embedded PTX. (Tradeoff: it does not use
  sm_90a/sm_120a arch-accelerated tensor features, and the first run is slower while the driver JITs;
  the result caches per-GPU under `%APPDATA%\NVIDIA\ComputeCache`.)
- **Quantized + MoE kernels** (`mmq_gguf/*`, `moe/*`, `mmvq_gguf` — the GGUF `QMatMul`) build via
  cudaforge `.build_lib()` → `nvcc -c`: a **static `libmoe.a` of SASS, _not_ PTX**. cudaforge emits
  one `-gencode` from `CUDA_COMPUTE_CAP` (`GpuArch::parse` runs `parse::<usize>()` on the whole
  string, so a `;`-list does **not** parse — there is no multi-cap support). At the `=80` baseline the
  archive held only an **sm_80 cubin**; SASS is not forward-compatible across major arches and there
  is no PTX to JIT, so on **Blackwell sm_120 every quant matmul silently returned zeros** — dense
  models rendered but quantized models came out black (**sc-7544**; the dense PTX path masked it).

**The fix (sc-7544): a multi-arch fatbin for the quant path.** cudaforge can't emit a cap list and the
candle pin is upstream (not a fork), so candle-kernels is **locally forked** in `vendor/candle-kernels`
(identical to the pinned rev except three `-gencode` lines in `build.rs`) and patched in via the
workspace `[patch]`. nvcc accumulates `-gencode` flags, so `libmoe.a` becomes a real fatbin embedding
**sm_80 + sm_90 + sm_120 SASS + `compute_120` PTX** — one binary that runs natively Ampere → Ada →
Hopper → Blackwell and JITs forward to newer archs. Keep `CUDA_COMPUTE_CAP=80` in the recipes (it
seeds the sm_80 baseline for both paths). Verified on RTX PRO 6000 (sm_120): `cuobjdump --list-elf`
shows sm_80/sm_90/sm_120 cubin per kernel, and `candle-gen`'s `cuda_quant_smoke` test has the Q4/Q8
`QMatMul` matching the CPU reference (cos ≈ 1.0, vs cos ≈ 0 / all-zeros before). That smoke runs in
the CUDA gate so the regression can't return silently. **Re-vendor on every candle pin bump** — see
`vendor/candle-kernels/VENDORED.md`.

### Build

Build-time needs the **CUDA 12.9 toolkit (nvcc)** + **VS 2022 v143 (MSVC 14.4x) Build Tools**; the
build is driven through `vcvars64.bat`. From a `cmd` shell that has sourced vcvars:

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set CUDA_COMPUTE_CAP=80
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
cargo build --release -p candle-gen-sdxl --example sdxl-txt2img --features cuda
```

The scripted, reproducible form of this — sources vcvars, sets the env, runs `cargo build/test
--workspace --features cuda` — is `scripts/check-cuda.ps1`. **Run it before pushing CUDA-touching
changes**: the CPU/Metal CI lanes are blind to `#[cfg(feature = "cuda")]` code, so this is the real
cuda gate.

```powershell
pwsh scripts/check-cuda.ps1            # build + test
pwsh scripts/check-cuda.ps1 -SkipTests # build-only smoke check
```

The `windows-cuda` lane in `.github/workflows/ci.yml` runs the same recipe but is **manual-only**
(`workflow_dispatch`) — it needs no standing runner. To run it in CI you must first register a
self-hosted `[self-hosted, windows, cuda]` runner, then dispatch the workflow by hand. (GitHub's
hosted GPU larger-runners are Tesla T4 / sm_75, below our sm_80 baseline, so they can't run it.)

### Bundle the runtime DLLs

The target machine needs the CUDA **runtime** libraries but should **not** require a CUDA Toolkit
install. `scripts/package-cuda.ps1` copies the binary plus the redistributable DLLs (which cudarc
dynamic-links, resolved from the exe's own directory) into `dist/`:

```powershell
pwsh scripts/package-cuda.ps1 -BinaryPath target\release\examples\sdxl-txt2img.exe
```

Bundled redist DLLs (CUDA 12.9; the script globs the version suffixes):

| DLL                          | provides            |
|------------------------------|---------------------|
| `cudart64_12.dll`            | CUDA runtime        |
| `cublas64_12.dll`            | cuBLAS              |
| `cublasLt64_12.dll`          | cuBLAS-Lt           |
| `curand64_10.dll`            | cuRAND              |
| `nvrtc64_120_0.dll`          | NVRTC               |
| `nvrtc-builtins64_129.dll`   | NVRTC builtins      |

The script also writes a `RUNTIME.txt` manifest into the bundle. Verified: with the bundle's DLLs
present and the **CUDA toolkit removed from `PATH`**, the binary runs end-to-end (DLLs resolve from
the exe's directory).

### Minimum driver

The **NVIDIA driver is not bundled** (it is not redistributable) and is what JIT-compiles the PTX +
provides `libcuda`. For the bundled **CUDA 12.9** runtime the floor is:

- **Windows: driver ≥ 576.02** (CUDA 12.9 GA).
- GPU compute capability **≥ 8.0** (Ampere / RTX 30-series or newer).

Older drivers should be updated from nvidia.com; the CUDA runtime DLLs in the bundle do **not** lift
the driver requirement.

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

The candle version this workspace settled on is recorded in `[workspace.dependencies]`
(`candle-core` / `candle-nn`).
