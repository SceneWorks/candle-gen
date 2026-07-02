//! `Fp8Linear` / `Int8Linear` (sc-9299) — the two 8-bit linear layers over the [`CublasLt`] compute
//! leg. Each holds a *statically* quantized weight (per-tensor scale, done once at construction) and
//! quantizes the activation **dynamically** per forward (v1: amax→scale→cast in pure candle ops, a
//! fused kernel is a later optimization). This is the layer a provider crate would swap in for an
//! fp8 fast tier or an INT8-ConvRot checkpoint (the rotation itself lives in sc-9300; this is just
//! the GEMM).
//!
//! Both are `#[cfg(feature = "cuda")]` — they own a `CublasLt` handle. The weight-quant / act-quant
//! helpers they build on are pure candle ops (see [`super::cublaslt`]) and compile everywhere.

use super::cublaslt::{
    quantize_activation_fp8, quantize_activation_int8, quantize_weight_fp8, quantize_weight_int8,
    CublasLt,
};
use candle_core::{Device, Result, Tensor};
use std::sync::Arc;

/// An fp8 E4M3 linear: `y = (X · Wᵀ) · scale_w · scale_x` with a per-tensor-quantized weight and
/// dynamic per-tensor activation quant. Optional bias added back in the output dtype.
pub struct Fp8Linear {
    w_fp8: Tensor, // (N, K) F8E4M3
    scale_w: f32,
    bias: Option<Tensor>,
    lt: Arc<CublasLt>,
}

impl Fp8Linear {
    /// Quantize a dense `(N, K)` weight to fp8 E4M3 once and bind it to a cuBLASLt handle.
    pub fn new(weight: &Tensor, bias: Option<Tensor>, lt: Arc<CublasLt>) -> Result<Self> {
        let qw = quantize_weight_fp8(weight)?;
        Ok(Self {
            w_fp8: qw.q,
            scale_w: qw.scale,
            bias,
            lt,
        })
    }

    /// Build sharing an existing handle for the device (constructs a fresh handle when `None`).
    pub fn from_device(weight: &Tensor, bias: Option<Tensor>, dev: &Device) -> Result<Self> {
        Self::new(weight, bias, Arc::new(CublasLt::new(dev)?))
    }

    /// `x`: `(..., K)`; flattened to `(M, K)` for the GEMM, then reshaped back. Output bf16.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (flat, restore) = flatten_tokens(x)?;
        let qx = quantize_activation_fp8(&flat)?;
        let y = self
            .lt
            .matmul_fp8(&self.w_fp8, self.scale_w, &qx.q, qx.scale)?;
        let y = restore(y)?;
        match &self.bias {
            Some(b) => y.broadcast_add(&b.to_dtype(y.dtype())?),
            None => Ok(y),
        }
    }
}

/// An int8 IGEMM linear: exact int32 accumulate, dequant scale folded on the candle side. Same
/// dynamic-activation-quant contract as [`Fp8Linear`].
pub struct Int8Linear {
    w_i8: Tensor, // (N, K) int codes carried in F32
    scale_w: f32,
    bias: Option<Tensor>,
    lt: Arc<CublasLt>,
}

impl Int8Linear {
    pub fn new(weight: &Tensor, bias: Option<Tensor>, lt: Arc<CublasLt>) -> Result<Self> {
        let qw = quantize_weight_int8(weight)?;
        Ok(Self {
            w_i8: qw.q,
            scale_w: qw.scale,
            bias,
            lt,
        })
    }

    pub fn from_device(weight: &Tensor, bias: Option<Tensor>, dev: &Device) -> Result<Self> {
        Self::new(weight, bias, Arc::new(CublasLt::new(dev)?))
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (flat, restore) = flatten_tokens(x)?;
        let qx = quantize_activation_int8(&flat)?;
        let y = self
            .lt
            .matmul_int8(&self.w_i8, self.scale_w, &qx.q, qx.scale)?;
        let y = restore(y)?;
        match &self.bias {
            Some(b) => y.broadcast_add(&b.to_dtype(y.dtype())?),
            None => Ok(y),
        }
    }
}

/// Collapse leading dims to a `(M, K)` matrix and return a closure that restores the original
/// leading shape on the `(M, N)` output.
fn flatten_tokens(x: &Tensor) -> Result<(Tensor, impl Fn(Tensor) -> Result<Tensor>)> {
    let dims = x.dims().to_vec();
    let k = *dims.last().expect("linear input has a last dim");
    let m: usize = dims[..dims.len() - 1].iter().product();
    let flat = x.reshape((m, k))?;
    let lead = dims[..dims.len() - 1].to_vec();
    let restore = move |y: Tensor| -> Result<Tensor> {
        let n = y.dim(1)?;
        let mut out_shape = lead.clone();
        out_shape.push(n);
        y.reshape(out_shape)
    };
    Ok((flat, restore))
}
