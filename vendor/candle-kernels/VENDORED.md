# Vendored `candle-kernels` — multi-arch fatbin fork (sc-7544)

This is a **local fork** of `candle-kernels` `0.10.2`, copied verbatim from the pinned candle revision

    https://github.com/huggingface/candle @ c1e6756a89faefa888ea57b056394a0619925b87

(the same rev the workspace pins `candle-core` / `candle-nn` / `candle-transformers` to). The copy
was originally taken at rev `65ecb58c11d2244a7e60c71bdcdb19b15b0a4343`; the 2026-07-02 pin bump to
`c1e6756a89` (upstream UAF fix #3493) did NOT require a re-copy — `candle-kernels/` is byte-identical
across `65ecb58c..c1e6756a89` (verified via the GitHub compare: no candle-kernels/ files in the diff). It is wired
into the build via a `[patch]` in the workspace `Cargo.toml`:

```toml
[patch."https://github.com/huggingface/candle"]
candle-kernels = { path = "vendor/candle-kernels" }
```

## The only change vs upstream

There are **two** changes vs upstream:

1. `build.rs` adds three `-gencode` flags to the **statically-linked quant/moe kernel** build
   (`build_lib()` → `libmoe.a`), turning its single-arch SASS object into a true **multi-arch fatbin**
   (the sc-7544 Blackwell fix, detailed below).
2. `src/cast.cu` adds the **`int32_t` source casts** (`cast_i32_f32` et al., search for `sc-9601`).
   Upstream omits I32 source casts, but candle-core's `to_dtype` still *dispatches* `cast_i32_f32` for an
   int32 tensor (`cuda_backend`), so without the symbol an on-device `i32 → f32` cast fails with "named
   symbol not found". The INT8-ConvRot int8 IGEMM accumulates in int32; this cast lets its per-row
   dequant fold stay on-device instead of a per-forward int32→host round-trip. `PTX`-embedded (goes
   through `build_ptx()`), so no `libmoe.a` change. Upstreamable (arguably an upstream gap).

Everything else — every other `.cu`/`.cuh` source, `lib.rs`, `ffi.rs`, `Cargo.toml` — is byte-for-byte
upstream, so candle-core links an otherwise-identical Rust/symbol surface (a fatter `libmoe.a` + the new
`cast_i32_*` PTX symbols). Diff against the upstream rev to confirm `build.rs` + the `cast.cu` i32 block
are the sole deltas.

### Why

`candle-kernels` compiles its GGUF `QMatMul` kernels (`mmq_gguf/*`, `moe/*`, `mmvq_gguf`) with
`nvcc -c` (a SASS **object**, no PTX) — unlike the dense kernels, which go through `build_ptx()` and
embed forward-JIT-able `compute_80` PTX. cudaforge emits one `-gencode` from `CUDA_COMPUTE_CAP`; at
the `=80` packaging baseline that is `code=sm_80` (an Ampere-only cubin). On a Blackwell **sm_120**
GPU there is no compatible cubin and no PTX to JIT, so the quant matmul **silently returns zeros**
(dense models work, quantized models render black/NaN). See the story sc-7544 and the
`candle-cuda-quant-needs-native-sm120` project memory. The fatbin embeds native `sm_80` + `sm_90` +
`sm_120` SASS plus `compute_120` PTX, so one binary runs natively Ampere → Ada → Hopper → Blackwell
and JITs forward to newer archs.

The regression is guarded by `candle-gen/tests/cuda_quant_smoke.rs` (runs in `scripts/check-cuda.ps1`).

## MAINTENANCE — re-vendor on every candle pin bump

The `[patch]` forces **these** kernel sources onto whatever `candle-core` rev the workspace pins. They
match only as long as this copy is from the **same** candle rev. **When the candle pin bumps**
(`candle-core`/`candle-nn`/`candle-transformers` rev in the workspace `Cargo.toml`):

1. Re-copy `candle-kernels/` from the new rev's checkout over `vendor/candle-kernels/`.
2. Re-apply the `build.rs` `-gencode` block above (search for `sc-7544`).
3. Re-apply the `src/cast.cu` `int32_t` cast block (search for `sc-9601`) unless upstream has added it.
4. Re-run the CUDA gate (`pwsh scripts/check-cuda.ps1`) — `cuda_quant_smoke` must pass on Blackwell.

If a bump ever lands without re-vendoring, candle-core may get **stale kernels** (subtle breakage or
link errors). If candle/cudaforge ever gains native multi-target fatbin support, drop this vendor and
the `[patch]` and configure the cap list directly.
