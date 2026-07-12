//! **cuBLASLt 8-bit GEMM wrapper** (sc-9299 spike, epic 9083) — the compute foundation for a
//! future fp8 fast tier AND the community INT8-ConvRot consume path (sc-9300). This is the piece the
//! sc-8523 NO-GO said was missing: candle's own int8 MMQ kernels (`fast_mmq`) run *slower* than plain
//! bf16 cuBLAS at DiT shapes on this box, and candle has no fp8 GEMM at all. cuBLASLt runs both at the
//! near-peak (≈2×-bf16 on Ada/Blackwell for fp8) the whole 8-bit track always assumed.
//!
//! # Why a candle-gen wrapper and not a candle fork
//! At our pin (`c1e6756a89`) candle-core already exposes [`F8E4M3`](candle_core::DType::F8E4M3) as a
//! first-class CUDA dtype (storage, casts, const-set — only the *matmul dispatch* is missing) and
//! re-exports `cudarc`, whose `cublaslt` module ships the raw sys bindings + a thin `result` layer
//! (`create_matmul_desc` / `create_matrix_layout` / `matmul` / heuristic search). The safe
//! `CudaBlasLT::matmul` only wires f32/f16/bf16 (scale-type hard-coded `CUDA_R_32F`, compute-type from
//! a per-`T` trait) — no fp8, no int8. So we drop to that `result` layer and drive `cublasLtMatmul`
//! ourselves for the two 8-bit configs. No fork; if we ever upstream, the dtype groundwork suggests a
//! PR would be welcome.
//!
//! # The two configs (locked-decision-7: sm_89 floor, modern CUDA 12.x layouts only — no col32/IMMA)
//! * **fp8 E4M3** — A/B both `CUDA_R_8F_E4M3`, `CUBLAS_COMPUTE_32F` accumulate, per-tensor `scaleA` /
//!   `scaleB` passed as **device pointers** (`A_SCALE_POINTER` / `B_SCALE_POINTER`), output bf16.
//!   cuBLASLt fp8 requires **TN** (A transposed, B non-transposed, both stored row-major) and
//!   16-byte-aligned leading dims. `FAST_ACCUM` is left OFF (the numerically-safe default; it trades
//!   accuracy for speed and the epic wants the parity proof first).
//! * **int8 IGEMM** — A/B `CUDA_R_8I`, `CUBLAS_COMPUTE_32I` → **int32** accumulate, output int32. On
//!   CUDA 12.x the relaxed **TN row-major** path removes the legacy `COL32` re-layout; scales are
//!   applied *after* the kernel (we return the raw int32 and let the caller fold `scaleA·scaleB`),
//!   which keeps the accumulate exact and matches how a ConvRot checkpoint wants to re-scale.
//!
//! # Layout contract (the fiddly part)
//! candle stores a `Linear` weight row-major as `(N, K)` (out, in) and the activation as `(M, K)`.
//! For `D = X · Wᵀ` (`(M,N)`), the TN cuBLASLt call is, in cuBLAS's **column-major** worldview:
//! `A = W` declared `(K, N)` col-major (== our row-major `(N,K)`, i.e. `op(A)=Aᵀ`, `transa=T`),
//! `B = X` declared `(K, M)` col-major (== our row-major `(M,K)`, `transb=N`), `D` declared `(N, M)`
//! col-major == our row-major `(M, N)`. Leading dims are all `K` for A/B and `N` for D. This is the
//! identity the tests pin against an f32 reference.
//!
//! Everything on the cuBLASLt handle is `#[cfg(feature = "cuda")]`; the CPU/Metal builds see only the
//! small dtype helpers ([`quantize_activation_fp8`] / [`quantize_activation_int8`], pure candle ops).

use candle_core::{DType, Device, Result, Tensor};

/// Result of a per-tensor dynamic activation quantization: the packed 8-bit tensor plus the scalar
/// `scale` such that `dequant = q * scale` (fp8: absmax/448; int8: absmax/127).
pub struct QuantizedActivation {
    /// The 8-bit tensor (`F8E4M3` for fp8; rounded int codes in `F32` for int8 — narrowed to device
    /// `i8` at the cuBLASLt seam so the pure-candle helper stays dtype-portable).
    pub q: Tensor,
    /// Per-tensor scale, `absmax / dtype_max`.
    pub scale: f32,
}

/// fp8 E4M3 finite absmax (values above this saturate to NaN/inf). Used as the dynamic-range divisor.
pub const F8E4M3_MAX: f32 = 448.0;
/// int8 symmetric range.
pub const I8_MAX: f32 = 127.0;

/// Dynamic per-tensor **fp8 E4M3** activation quant (v1, pure candle ops — a fused amax→scale→cast
/// kernel is a later optimization if the bench shows it dominates). `scale = absmax / 448`, then
/// `q = cast_f8e4m3(x / scale)`.
///
/// **The final `f32 → F8E4M3` cast is done on the CPU** and the result moved back to `x`'s device.
/// candle's CUDA fp8-cast kernel (`cast_f32_f8_e4m3`) is `#if __CUDA_ARCH__ >= 800`-gated in
/// candle-kernels and is **not present** in the repo's cap=80 fatbin on this sm_120 box (the cast
/// dispatch fails with `CUDA_ERROR_NOT_FOUND "named symbol not found"` — sc-7544 territory, but for a
/// dense-PTX cast rather than a quant matmul). The CPU cast is byte-identical (the `float8` crate does
/// the same E4M3 rounding on host) so it is a correctness-neutral workaround; wiring the device cast
/// (re-vendor candle-kernels with the fp8 casts in the multi-arch fatbin) is a perf follow-up (sc-9299
/// note → sc-9300). The divide/amax stay on-device; only the tiny elementwise cast round-trips.
pub fn quantize_activation_fp8(x: &Tensor) -> Result<QuantizedActivation> {
    let absmax = x.abs()?.flatten_all()?.max(0)?.to_dtype(DType::F32)?;
    let scale = (absmax.to_scalar::<f32>()? / F8E4M3_MAX).max(f32::MIN_POSITIVE);
    let scaled = (x.to_dtype(DType::F32)? / scale as f64)?;
    let q = scaled
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F8E4M3)?
        .to_device(x.device())?;
    Ok(QuantizedActivation { q, scale })
}

/// Dynamic per-tensor **int8** activation quant (v1). `scale = absmax / 127`, `q = round(x/scale)`
/// clamped to `[-127, 127]`, returned in `F32` (the rounded codes).
pub fn quantize_activation_int8(x: &Tensor) -> Result<QuantizedActivation> {
    let absmax = x.abs()?.flatten_all()?.max(0)?.to_dtype(DType::F32)?;
    let scale = (absmax.to_scalar::<f32>()? / I8_MAX).max(f32::MIN_POSITIVE);
    let scaled = (x.to_dtype(DType::F32)? / scale as f64)?;
    let q = scaled.round()?.clamp(-I8_MAX, I8_MAX)?;
    Ok(QuantizedActivation { q, scale })
}

/// Quantize a weight `(N, K)` to fp8 E4M3 once (per-tensor), returning the packed tensor + scale.
/// Weight quant is static (done at load), so this is the load-time twin of [`quantize_activation_fp8`].
pub fn quantize_weight_fp8(w: &Tensor) -> Result<QuantizedActivation> {
    quantize_activation_fp8(w)
}

/// Quantize a weight `(N, K)` to int8 (per-tensor), returning rounded codes (`F32`) + scale.
pub fn quantize_weight_int8(w: &Tensor) -> Result<QuantizedActivation> {
    quantize_activation_int8(w)
}

/// Result of a **per-output-channel** int8 weight quantization: the int8 codes (`F32`) plus a `[out]`
/// scale vector such that `dequant[o, :] = q[o, :] * scale[o]`. This is the granularity a community
/// INT8-ConvRot checkpoint stores (`{base}.weight_scale`, `[out, 1]` f32), and the strict superset of
/// the per-tensor path (a per-tensor scale is a per-channel vector with one distinct value).
pub struct PerChannelInt8Weight {
    /// `(N, K)` int8 codes carried in `F32` (rounded, clamped to `[-127, 127]`).
    pub q: Tensor,
    /// `[N]` (per-output-row) dequant scale, `absmax_row / 127`.
    pub scale: Vec<f32>,
}

/// Dynamic **per-output-channel** int8 weight quant — `scale[o] = absmax(row o) / 127`,
/// `q[o, :] = round(w[o, :] / scale[o])` clamped to `[-127, 127]`. The candle-side dequant fold in
/// [`CublasLt::matmul_int8_per_channel`] applies this `[out]` vector to the int32 accumulator.
///
/// This is the load-time twin of a ConvRot checkpoint's stored per-row weight scale; a loader that
/// already has the on-disk `{base}.weight_scale` and int8 `{base}.weight` skips this and builds the
/// [`PerChannelInt8Weight`] directly from the parts (no re-quantization). Present so an int8 tier can
/// be produced from a dense weight for tests / a from-dense fold.
pub fn quantize_weight_int8_per_channel(w: &Tensor) -> Result<PerChannelInt8Weight> {
    let (n, _k) = w.dims2()?;
    let absmax = w.abs()?.max(1)?.to_dtype(DType::F32)?; // [N]
    let absmax_v = absmax.to_vec1::<f32>()?;
    let scale: Vec<f32> = absmax_v
        .iter()
        .map(|&a| (a / I8_MAX).max(f32::MIN_POSITIVE))
        .collect();
    let scale_col = Tensor::from_vec(scale.clone(), (n, 1), w.device())?;
    let q = w
        .to_dtype(DType::F32)?
        .broadcast_div(&scale_col)?
        .round()?
        .clamp(-I8_MAX, I8_MAX)?;
    Ok(PerChannelInt8Weight { q, scale })
}

#[cfg(feature = "cuda")]
mod cuda_impl {
    use super::*;
    use candle_core::cuda_backend::cudarc;
    use candle_core::cuda_backend::CudaStorageSlice;
    use candle_core::op::BackpropOp;
    use candle_core::{CudaStorage, Storage};
    use cudarc::cublaslt::{result as lt, sys};
    use cudarc::driver::{CudaStream, DevicePtr, DevicePtrMut};
    use std::ffi::c_void;
    use std::mem::size_of;
    use std::sync::Arc;

    /// A cuBLASLt handle bound to a candle CUDA device, plus a reusable workspace. Cheap to build
    /// per matmul in a spike/bench; a real integration caches one per device.
    pub struct CublasLt {
        handle: sys::cublasLtHandle_t,
        stream: Arc<CudaStream>,
        device: candle_core::CudaDevice,
        workspace: cudarc::driver::CudaSlice<u8>,
        workspace_size: usize,
    }

    // The Lt handle is an opaque device-side object; guarded by the owning stream/device.
    unsafe impl Send for CublasLt {}
    unsafe impl Sync for CublasLt {}

    impl Drop for CublasLt {
        fn drop(&mut self) {
            unsafe {
                let _ = lt::destroy_handle(self.handle);
            }
        }
    }

    impl CublasLt {
        /// 32 MiB workspace — enough for the Split-K / stream-K algos cuBLASLt picks at DiT shapes.
        const WORKSPACE: usize = 32 * 1024 * 1024;

        pub fn new(dev: &Device) -> Result<Self> {
            let device = match dev {
                Device::Cuda(c) => c.clone(),
                _ => candle_core::bail!("CublasLt::new requires a CUDA device"),
            };
            let stream = device.cuda_stream();
            let handle = lt::create_handle().map_err(cublas_err)?;
            let workspace = stream.alloc_zeros::<u8>(Self::WORKSPACE).map_err(drv_err)?;
            Ok(Self {
                handle,
                stream,
                device,
                workspace,
                workspace_size: Self::WORKSPACE,
            })
        }

        /// Compute capability of the bound device as `(major, minor)` — the sm_89 eligibility gate
        /// (locked-decision-7) reads this. For the worker, this is queryable straight off the candle
        /// device via `CudaDevice::cuda_stream().context().attribute(..)` (see module writeup).
        pub fn compute_cap(&self) -> Result<(i32, i32)> {
            let ctx = self.stream.context();
            let major = ctx
                .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
                .map_err(drv_err)?;
            let minor = ctx
                .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
                .map_err(drv_err)?;
            Ok((major, minor))
        }

        /// True iff the device meets the sc-9299 sm_89 floor (fp8 needs 8.9; int8 IGEMM 8.0, but the
        /// whole 8-bit track is floored at 8.9 per locked-decision-7).
        pub fn meets_fp8_floor(&self) -> Result<bool> {
            let (maj, min) = self.compute_cap()?;
            Ok(maj > 8 || (maj == 8 && min >= 9))
        }

        /// **fp8 E4M3 GEMM**: `D = (scale_w·scale_x) · (X · Wᵀ)` → bf16 `(M, N)`.
        /// `w_fp8` is `(N, K)` `F8E4M3`, `x_fp8` is `(M, K)` `F8E4M3`, both contiguous. Stages both
        /// operands then delegates to [`Self::matmul_fp8_staged`].
        pub fn matmul_fp8(
            &self,
            w_fp8: &Tensor,
            scale_w: f32,
            x_fp8: &Tensor,
            scale_x: f32,
        ) -> Result<Tensor> {
            let a = DevFp8::stage(self, w_fp8)?;
            let b = DevFp8::stage(self, x_fp8)?;
            self.matmul_fp8_staged(&a, scale_w, &b, scale_x)
        }

        /// **int8 IGEMM**: raw `int32 = X_i8 · W_i8ᵀ` (no scale folded — caller multiplies by
        /// `scale_w·scale_x`). `w_i8`/`x_i8` carry the rounded int8 codes in `F32`; narrowed to
        /// device `i8` here. Returns `(M, N)` on-device `I32`.
        pub fn matmul_int8_raw(&self, w_i8: &Tensor, x_i8: &Tensor) -> Result<Tensor> {
            let a = DevInt8::stage(self, w_i8)?; // A = W (N,K)
            let b = DevInt8::stage(self, x_i8)?; // B = X (M,K)
            self.matmul_int8_staged(&a, &b)
        }

        /// int8 IGEMM over **pre-staged** device operands (the honest compute path — no per-call host
        /// narrowing). Returns `(M, N)` on-device `I32`. This is what the bench times.
        pub fn matmul_int8_staged(&self, w: &DevInt8, x: &DevInt8) -> Result<Tensor> {
            let (m, k, n) = (x.rows, x.cols, w.rows);
            if x.cols != w.cols {
                candle_core::bail!(
                    "int8 staged shape mismatch: x=(.,{}) w=({},{})",
                    x.cols,
                    w.rows,
                    w.cols
                );
            }
            check_alignment(k, n)?;
            let mut out = self.stream.alloc_zeros::<i32>(m * n).map_err(drv_err)?;
            {
                let (a_ptr, _ga) = w.buf.device_ptr(&self.stream);
                let (b_ptr, _gb) = x.buf.device_ptr(&self.stream);
                let (d_ptr, _gd) = out.device_ptr_mut(&self.stream);
                unsafe {
                    self.run(
                        sys::cudaDataType_t::CUDA_R_8I,
                        sys::cublasComputeType_t::CUBLAS_COMPUTE_32I,
                        sys::cudaDataType_t::CUDA_R_32I,
                        sys::cudaDataType_t::CUDA_R_32I,
                        m,
                        k,
                        n,
                        a_ptr,
                        b_ptr,
                        d_ptr,
                        None,
                        Alpha::I32(1),
                    )?;
                }
            }
            // sc-11260 (F-100): no per-call `stream.synchronize()` here. The GEMM enqueues on the
            // device stream and stream ordering already sequences every downstream consumer.
            // Correctness rests on two LOAD-BEARING invariants:
            //   (1) Same-stream ordering — the GEMM, the `out` allocation, and the host read-back
            //       copy all run on `self.stream` (= `device.cuda_stream()`), so the GEMM is
            //       ordered strictly before the read.
            //   (2) Blocking read-back — the host-fold paths
            //       (`matmul_int8[_per_channel[_staged]]`) read the accumulate back via `clone_dtoh`
            //       into a PAGEABLE (non-pinned) `Vec`. A device→pageable-host `cudaMemcpyAsync`
            //       blocks the host until the copy — hence the GEMM — completes, so no explicit sync
            //       is needed. The on-device fold (`matmul_int8_per_channel_staged_ondevice`) chains
            //       stream-ordered candle ops, and the fp8 twin (`matmul_fp8_staged`) is sync-free
            //       for the same reasons.
            // CAVEAT: this holds ONLY while read-backs land in pageable host memory. If a caller
            // ever switches the read-back to a PINNED host buffer, the blocking guarantee is lost
            // and an explicit stream sync (or event) would be required again — do not reintroduce a
            // race by that route. Draining the pipeline once per int8 projection defeated async
            // enqueue-ahead on the ConvRot resident forward.
            let storage = CudaStorage {
                slice: CudaStorageSlice::I32(out),
                device: self.device.clone(),
            };
            Ok(Tensor::from_storage(
                Storage::Cuda(storage),
                (m, n),
                BackpropOp::none(),
                false,
            ))
        }

        /// **On-device** per-output-channel int8 linear (sc-9601 perf) — the resident-weight ConvRot
        /// forward with the dequant fold kept entirely on the GPU. Runs the exact int32 IGEMM
        /// ([`Self::matmul_int8_staged`], `CudaStorageSlice::I32` on device), casts the accumulate `i32 →
        /// f32` **on-device** (`Tensor::to_dtype`, via the sc-9601 vendored `cast_i32_f32` kernel), then
        /// folds the per-row weight scale × per-tensor activation scale with a candle broadcast multiply
        /// and casts to bf16 — **no int32→host copy, no CPU fold**. This is the fast twin of
        /// [`Self::matmul_int8_per_channel_staged`] (which reads the int32 accumulate back to host because
        /// stock candle-kernels ships no CUDA `i32 → f32` cast; the vendored kernel closes that gap). The
        /// i32→f32 conversion is an exact hardware I2F (round-to-nearest, rel ≤ 6e-8 for the few outputs
        /// beyond 2²⁴ — negligible beside int8's ~1 %). `D[m, o] = (scale_w[o]·scale_x) · acc[m, o]` → bf16.
        pub fn matmul_int8_per_channel_staged_ondevice(
            &self,
            w: &DevInt8,
            scale_w: &[f32],
            x_i8: &Tensor,
            scale_x: f32,
        ) -> Result<Tensor> {
            let x = DevInt8::stage(self, x_i8)?;
            let n = w.rows;
            if scale_w.len() != n {
                candle_core::bail!(
                    "matmul_int8_per_channel_staged_ondevice: scale_w len {} != N (out) {n}",
                    scale_w.len()
                );
            }
            let acc = self.matmul_int8_staged(w, &x)?.to_dtype(DType::F32)?; // (M, N), on device
                                                                             // Fold the dequant scale on-device: per-row weight scale (a [1, N] row broadcast over M) times
                                                                             // the scalar activation scale. `scale_x` is folded into the row vector so it's one multiply.
            let row: Vec<f32> = scale_w.iter().map(|&s| s * scale_x).collect();
            let row = Tensor::from_vec(row, (1, n), acc.device())?;
            acc.broadcast_mul(&row)?.to_dtype(DType::BF16)
        }

        /// fp8 GEMM over a **pre-staged** device weight + activation (no per-call operand clone). The
        /// bench's honest fp8 compute path.
        pub fn matmul_fp8_staged(
            &self,
            w: &DevFp8,
            scale_w: f32,
            x: &DevFp8,
            scale_x: f32,
        ) -> Result<Tensor> {
            let (m, k, n) = (x.rows, x.cols, w.rows);
            if x.cols != w.cols {
                candle_core::bail!(
                    "fp8 staged shape mismatch: x=(.,{}) w=({},{})",
                    x.cols,
                    w.rows,
                    w.cols
                );
            }
            check_alignment(k, n)?;
            let mut out = unsafe { self.stream.alloc::<half::bf16>(m * n) }.map_err(drv_err)?;
            let sa = self.upload_scalar(scale_w)?;
            let sb = self.upload_scalar(scale_x)?;
            {
                let (a_ptr, _ga) = w.buf.device_ptr(&self.stream);
                let (b_ptr, _gb) = x.buf.device_ptr(&self.stream);
                let (sa_ptr, _gsa) = sa.device_ptr(&self.stream);
                let (sb_ptr, _gsb) = sb.device_ptr(&self.stream);
                let (d_ptr, _gd) = out.device_ptr_mut(&self.stream);
                unsafe {
                    self.run(
                        sys::cudaDataType_t::CUDA_R_8F_E4M3,
                        sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                        sys::cudaDataType_t::CUDA_R_32F,
                        sys::cudaDataType_t::CUDA_R_16BF,
                        m,
                        k,
                        n,
                        a_ptr,
                        b_ptr,
                        d_ptr,
                        Some((sa_ptr, sb_ptr)),
                        Alpha::F32(1.0),
                    )?;
                }
            }
            let storage = CudaStorage {
                slice: CudaStorageSlice::BF16(out),
                device: self.device.clone(),
            };
            Ok(Tensor::from_storage(
                Storage::Cuda(storage),
                (m, n),
                BackpropOp::none(),
                false,
            ))
        }

        /// Full int8 linear `D = (scale_w·scale_x)·(X·Wᵀ)` → bf16, folding the dequant scale on the
        /// candle side after the exact int32 accumulate.
        ///
        /// The int32 accumulate is read back to the host and the `scale_w·scale_x` fold + bf16 cast
        /// happen there: candle-kernels ships **no** CUDA `i32 → f32` cast (only `u32`/`i64` have full
        /// cast coverage), so an on-device `I32` tensor cannot be `to_dtype`'d. The staged
        /// [`Self::matmul_int8_staged`] keeps the accumulate on-device for the bench (which never
        /// casts it). The **on-device** fold that avoids this host round-trip entirely — via the
        /// f32-output IGEMM epilogue — is [`Self::matmul_int8_per_channel_staged_ondevice`] (sc-9601);
        /// this per-tensor convenience path keeps the host fold (it is not on a hot resident path).
        pub fn matmul_int8(
            &self,
            w_i8: &Tensor,
            scale_w: f32,
            x_i8: &Tensor,
            scale_x: f32,
        ) -> Result<Tensor> {
            let a = DevInt8::stage(self, w_i8)?;
            let b = DevInt8::stage(self, x_i8)?;
            let (m, n) = (b.rows, a.rows);
            let acc = self.matmul_int8_staged(&a, &b)?;
            // Read the on-device I32 accumulate back to host (no CUDA i32 cast exists), fold the
            // dequant scale, upload as bf16.
            let (storage, _l) = acc.storage_and_layout();
            let host_i32: Vec<i32> = match &*storage {
                Storage::Cuda(cs) => match &cs.slice {
                    CudaStorageSlice::I32(s) => self.stream.clone_dtoh(s).map_err(drv_err)?,
                    _ => candle_core::bail!("matmul_int8: expected I32 accumulate"),
                },
                _ => candle_core::bail!("matmul_int8: accumulate not on CUDA"),
            };
            let s = scale_w * scale_x;
            let host_f32: Vec<f32> = host_i32.iter().map(|&v| v as f32 * s).collect();
            Tensor::from_vec(host_f32, (m, n), acc.device())?.to_dtype(DType::BF16)
        }

        /// Full int8 linear with a **per-output-channel** weight scale — the community INT8-ConvRot
        /// dequant (sc-9300): `D[m, o] = (scale_w[o] · scale_x) · (X_i8 · W_i8ᵀ)[m, o]` → bf16. The
        /// `[N]` `scale_w` is the checkpoint's stored per-row `{base}.weight_scale`; `scale_x` is the
        /// dynamic per-tensor activation scale. A strict superset of [`Self::matmul_int8`] (pass an
        /// all-equal `scale_w` to recover the per-tensor fold). Like [`Self::matmul_int8`] the int32
        /// accumulate is read back to host for the fold + bf16 cast (candle-kernels ships no CUDA
        /// `i32 → f32` cast), so the exact accumulate is preserved and only the scale application
        /// differs.
        ///
        /// This is the exact `X·Wᵀ` compute for a per-channel-quantized int8 weight. For a ConvRot
        /// checkpoint the stored `W_i8` is the *rotated* weight `W·R`, so the consume path applies the
        /// matching online activation rotation `RHT(x)` ([`super::convrot`], sc-9601) before this call,
        /// making `RHT(x)·(W·R)ᵀ = x·Wᵀ`. The compute here is rotation-agnostic and correct either way.
        pub fn matmul_int8_per_channel(
            &self,
            w_i8: &Tensor,
            scale_w: &[f32],
            x_i8: &Tensor,
            scale_x: f32,
        ) -> Result<Tensor> {
            let a = DevInt8::stage(self, w_i8)?;
            let b = DevInt8::stage(self, x_i8)?;
            let (m, n) = (b.rows, a.rows);
            if scale_w.len() != n {
                candle_core::bail!(
                    "matmul_int8_per_channel: scale_w len {} != N (out) {n}",
                    scale_w.len()
                );
            }
            let acc = self.matmul_int8_staged(&a, &b)?;
            let (storage, _l) = acc.storage_and_layout();
            let host_i32: Vec<i32> = match &*storage {
                Storage::Cuda(cs) => match &cs.slice {
                    CudaStorageSlice::I32(s) => self.stream.clone_dtoh(s).map_err(drv_err)?,
                    _ => candle_core::bail!("matmul_int8_per_channel: expected I32 accumulate"),
                },
                _ => candle_core::bail!("matmul_int8_per_channel: accumulate not on CUDA"),
            };
            // Row-major `(M, N)`: element `(row, col)` dequants by `scale_w[col] · scale_x`.
            let host_f32: Vec<f32> = host_i32
                .iter()
                .enumerate()
                .map(|(i, &v)| v as f32 * scale_w[i % n] * scale_x)
                .collect();
            Tensor::from_vec(host_f32, (m, n), acc.device())?.to_dtype(DType::BF16)
        }

        /// Per-output-channel int8 over a **pre-staged** device weight (sc-9300) — the resident-weight
        /// form the ConvRot consume path uses so the `(N, K)` int8 codes live on-device as native `i8`
        /// (1 byte/elem) rather than as an 8×-larger I64 tensor. `w` is the staged weight, `scale_w` its
        /// `[N]` per-row dequant scale, `x_i8`/`scale_x` the dynamically-quantized activation. Same fold
        /// as [`Self::matmul_int8_per_channel`], only the weight is not re-staged per call. This is the
        /// **exact host fold** (int32→host); the on-device fast twin
        /// [`Self::matmul_int8_per_channel_staged_ondevice`] (sc-9601) folds on-device via the f32-output
        /// IGEMM and is the path the ConvRot resident forward uses when the device supports it.
        pub fn matmul_int8_per_channel_staged(
            &self,
            w: &DevInt8,
            scale_w: &[f32],
            x_i8: &Tensor,
            scale_x: f32,
        ) -> Result<Tensor> {
            let x = DevInt8::stage(self, x_i8)?;
            let (m, n) = (x.rows, w.rows);
            if scale_w.len() != n {
                candle_core::bail!(
                    "matmul_int8_per_channel_staged: scale_w len {} != N (out) {n}",
                    scale_w.len()
                );
            }
            let acc = self.matmul_int8_staged(w, &x)?;
            let (storage, _l) = acc.storage_and_layout();
            let host_i32: Vec<i32> = match &*storage {
                Storage::Cuda(cs) => match &cs.slice {
                    CudaStorageSlice::I32(s) => self.stream.clone_dtoh(s).map_err(drv_err)?,
                    _ => candle_core::bail!(
                        "matmul_int8_per_channel_staged: expected I32 accumulate"
                    ),
                },
                _ => candle_core::bail!("matmul_int8_per_channel_staged: accumulate not on CUDA"),
            };
            let host_f32: Vec<f32> = host_i32
                .iter()
                .enumerate()
                .map(|(i, &v)| v as f32 * scale_w[i % n] * scale_x)
                .collect();
            Tensor::from_vec(host_f32, (m, n), acc.device())?.to_dtype(DType::BF16)
        }

        /// Probe (once) whether this device can cast an **int32 tensor to f32 on-device** — the capability
        /// the on-device dequant fast path (sc-9601) needs, provided by the vendored `cast_i32_f32` kernel.
        /// Stock candle-kernels omits the I32 source casts, so on a build without the vendored fork this
        /// returns `false` and the caller keeps the exact int32→host fold. Runs a tiny `i32 → f32`
        /// `to_dtype` and reports success; swallows errors deliberately (a probe, not a compute).
        pub fn supports_ondevice_int8_dequant(&self) -> bool {
            let dev = Device::Cuda(self.device.clone());
            let probe = || -> Result<()> {
                let dummy = Tensor::zeros((16, 16), DType::F32, &dev)?;
                let (a, b) = (DevInt8::stage(self, &dummy)?, DevInt8::stage(self, &dummy)?);
                let acc = self.matmul_int8_staged(&a, &b)?; // I32 on device
                acc.to_dtype(DType::F32)?; // the cast_i32_f32 kernel (fails if absent)
                Ok(())
            };
            probe().is_ok()
        }

        /// Stage an fp8 weight/activation tensor (`F8E4M3`) as an owned device buffer + its shape.
        pub fn stage_fp8(&self, t: &Tensor) -> Result<DevFp8> {
            DevFp8::stage(self, t)
        }
        /// Stage an int8-code tensor (`F32`/`I64`) as an owned device `i8` buffer + its shape.
        pub fn stage_int8(&self, t: &Tensor) -> Result<DevInt8> {
            DevInt8::stage(self, t)
        }

        // --- internals -------------------------------------------------------------------------

        #[allow(clippy::too_many_arguments)]
        unsafe fn run(
            &self,
            io_dtype: sys::cudaDataType_t,
            compute: sys::cublasComputeType_t,
            scale_dtype: sys::cudaDataType_t,
            out_dtype: sys::cudaDataType_t,
            m: usize,
            k: usize,
            n: usize,
            a_ptr: cudarc::driver::sys::CUdeviceptr,
            b_ptr: cudarc::driver::sys::CUdeviceptr,
            d_ptr: cudarc::driver::sys::CUdeviceptr,
            scales: Option<(
                cudarc::driver::sys::CUdeviceptr,
                cudarc::driver::sys::CUdeviceptr,
            )>,
            alpha: Alpha,
        ) -> Result<()> {
            let desc = lt::create_matmul_desc(compute, scale_dtype).map_err(cublas_err)?;
            // TN: op(A)=Aᵀ (transa=T), op(B)=B (transb=N) — the only fp8 layout cuBLASLt accepts and
            // the modern relaxed-TN int8 row-major path (no COL32). The transpose op is a raw i32
            // (`1 == T, 0 == N`) — `cublasOperation_t` lives in `cublas::sys`, not `cublaslt::sys`,
            // and cuBLASLt reads TRANSA/TRANSB as a 32-bit int (mirrors cudarc's own `set_transpose`).
            let transa: i32 = 1; // CUBLAS_OP_T
            let transb: i32 = 0; // CUBLAS_OP_N
            set_attr(
                desc,
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
                &transa,
            )?;
            set_attr(
                desc,
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
                &transb,
            )?;

            if let Some((sa, sb)) = scales {
                set_attr(
                    desc,
                    sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                    &sa,
                )?;
                set_attr(
                    desc,
                    sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                    &sb,
                )?;
            }

            // Layouts (see module doc): A=(K,N) col-major ld=K, B=(K,M) col-major ld=K,
            // D=(N,M) col-major ld=N.
            let a_layout = lt::create_matrix_layout(io_dtype, k as u64, n as u64, k as i64)
                .map_err(cublas_err)?;
            let b_layout = lt::create_matrix_layout(io_dtype, k as u64, m as u64, k as i64)
                .map_err(cublas_err)?;
            let d_layout = lt::create_matrix_layout(out_dtype, n as u64, m as u64, n as i64)
                .map_err(cublas_err)?;

            let pref = lt::create_matmul_pref().map_err(cublas_err)?;
            lt::set_matmul_pref_attribute(
                pref,
                sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &self.workspace_size as *const _ as *const c_void,
                size_of::<usize>(),
            )
            .map_err(cublas_err)?;

            let heuristic = lt::get_matmul_algo_heuristic(
                self.handle,
                desc,
                a_layout,
                b_layout,
                d_layout,
                d_layout,
                pref,
            )
            .map_err(cublas_err)?;

            let (alpha_ptr, beta_ptr): (*const c_void, *const c_void) = match &alpha {
                Alpha::F32(a) => (
                    a as *const f32 as *const c_void,
                    &F32_ZERO as *const f32 as *const c_void,
                ),
                Alpha::I32(a) => (
                    a as *const i32 as *const c_void,
                    &I32_ZERO as *const i32 as *const c_void,
                ),
            };

            let (ws_ptr, _gw) = self.workspace.device_ptr(&self.stream);

            let res = lt::matmul(
                self.handle,
                desc,
                alpha_ptr,
                beta_ptr,
                a_ptr as *const c_void,
                a_layout,
                b_ptr as *const c_void,
                b_layout,
                d_ptr as *const c_void,
                d_layout,
                d_ptr as *mut c_void,
                d_layout,
                &heuristic.algo as *const _,
                ws_ptr as *mut c_void,
                self.workspace_size,
                // driver::sys::CUstream and cublaslt::sys::cudaStream_t are both `*mut CUstream_st`
                // but distinct newtype aliases across the two sys modules — cast the raw pointer.
                self.stream.cu_stream() as sys::cudaStream_t,
            );

            let _ = lt::destroy_matrix_layout(a_layout);
            let _ = lt::destroy_matrix_layout(b_layout);
            let _ = lt::destroy_matrix_layout(d_layout);
            let _ = lt::destroy_matmul_pref(pref);
            let _ = lt::destroy_matmul_desc(desc);
            res.map_err(cublas_err)
        }

        fn upload_scalar(&self, v: f32) -> Result<cudarc::driver::CudaSlice<f32>> {
            self.stream.clone_htod(&[v]).map_err(drv_err)
        }
    }

    /// A `(rows, cols)` fp8 weight/activation staged as an owned, contiguous, offset-0 device
    /// `F8E4M3` buffer. Staged once (at load for weights; per-forward for activations) so a repeated
    /// GEMM does not re-clone the operand — the difference between the bench's honest compute number
    /// and the naive per-call one.
    pub struct DevFp8 {
        buf: cudarc::driver::CudaSlice<float8::F8E4M3>,
        rows: usize,
        cols: usize,
    }

    impl DevFp8 {
        fn stage(lt: &CublasLt, t: &Tensor) -> Result<Self> {
            expect_dtype(t, DType::F8E4M3)?;
            let (rows, cols) = t.dims2()?;
            let t = t.contiguous()?;
            let (storage, _l) = t.storage_and_layout();
            let buf = match &*storage {
                Storage::Cuda(cs) => match &cs.slice {
                    CudaStorageSlice::F8E4M3(s) => s.try_clone().map_err(drv_err)?,
                    _ => candle_core::bail!("DevFp8::stage: expected F8E4M3 storage"),
                },
                _ => candle_core::bail!("DevFp8::stage: not a CUDA tensor"),
            };
            let _ = lt;
            Ok(Self { buf, rows, cols })
        }
    }

    /// A `(rows, cols)` int8-code operand as an owned device **byte** buffer. candle has no `i8` dtype,
    /// so the int8 codes are carried as their two's-complement `u8` byte pattern (`c as i8 as u8`);
    /// cuBLASLt reads the buffer as `CUDA_R_8I`, so byte `b` reads back as the signed code `b as i8`.
    /// Holding `u8` (candle's native byte dtype) is what lets a **CUDA** operand stage entirely
    /// on-device (sc-9601 perf): the per-forward activation narrow becomes a handful of candle kernels
    /// instead of a DtoH copy + CPU round loop + HtoD copy (the dominant cost once the dequant fold
    /// moved on-device). A CPU operand (weights loaded to host, tests) still narrows on the host.
    pub struct DevInt8 {
        buf: cudarc::driver::CudaSlice<u8>,
        rows: usize,
        cols: usize,
    }

    impl DevInt8 {
        fn stage(lt: &CublasLt, t: &Tensor) -> Result<Self> {
            let (rows, cols) = t.dims2()?;
            if t.device().is_cuda() {
                // On-device narrow to the int8 byte pattern (sc-9601): round to [-127, 127], wrap the
                // negatives into [128, 255] (two's complement: `c + 256`), then cast f32→u8. The values
                // are exact integers so the f32→u8 cast is lossless. cuBLASLt reads these bytes as i8.
                let codes = t.to_dtype(DType::F32)?.round()?.clamp(-I8_MAX, I8_MAX)?;
                let neg256 = codes.lt(0f32)?.to_dtype(DType::F32)?.affine(256.0, 0.0)?;
                let u8t = codes
                    .add(&neg256)?
                    .to_dtype(DType::U8)?
                    .flatten_all()?
                    .contiguous()?;
                let (storage, _l) = u8t.storage_and_layout();
                let buf = match &*storage {
                    Storage::Cuda(cs) => match &cs.slice {
                        CudaStorageSlice::U8(s) => s.try_clone().map_err(drv_err)?,
                        _ => candle_core::bail!("DevInt8::stage: expected U8 storage"),
                    },
                    _ => candle_core::bail!("DevInt8::stage: not a CUDA tensor"),
                };
                return Ok(Self { buf, rows, cols });
            }
            // CPU operand (weights staged from host, tests): narrow on the host to the same byte pattern.
            let host = t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            let bytes: Vec<u8> = host.iter().map(|&v| v.round() as i8 as u8).collect();
            let buf = lt.stream.clone_htod(&bytes).map_err(drv_err)?;
            Ok(Self { buf, rows, cols })
        }
    }

    /// alpha carrier so int32 vs f32 compute get correctly-typed scalars.
    enum Alpha {
        F32(f32),
        I32(i32),
    }
    static F32_ZERO: f32 = 0.0;
    static I32_ZERO: i32 = 0;

    unsafe fn set_attr<T>(
        desc: sys::cublasLtMatmulDesc_t,
        attr: sys::cublasLtMatmulDescAttributes_t,
        val: &T,
    ) -> Result<()> {
        lt::set_matmul_desc_attribute(desc, attr, val as *const T as *const c_void, size_of::<T>())
            .map_err(cublas_err)
    }

    fn check_alignment(k: usize, n: usize) -> Result<()> {
        if !k.is_multiple_of(16) || !n.is_multiple_of(16) {
            candle_core::bail!("cuBLASLt 8-bit: K ({k}) and N ({n}) must be multiples of 16");
        }
        Ok(())
    }

    fn expect_dtype(t: &Tensor, want: DType) -> Result<()> {
        if t.dtype() != want {
            candle_core::bail!("expected {want:?}, got {:?}", t.dtype());
        }
        Ok(())
    }

    fn cublas_err(e: cudarc::cublaslt::result::CublasError) -> candle_core::Error {
        candle_core::Error::Cuda(format!("cublasLt: {e:?}").into())
    }
    fn drv_err(e: cudarc::driver::DriverError) -> candle_core::Error {
        candle_core::Error::Cuda(format!("cuda driver: {e:?}").into())
    }
}

#[cfg(feature = "cuda")]
pub use cuda_impl::{CublasLt, DevFp8, DevInt8};
