//! The only `candle-transformers::models::with_tracing` items the vendored SDXL UNet stack uses are
//! `Conv2d` + `conv2d` (sc-5165). They are a thin tracing wrapper over `candle_nn::Conv2d`; we vendor
//! a tracing-free equivalent so this crate need not depend on `candle-transformers`' internals. The
//! computation is identical to `candle_nn::Conv2d` (tracing spans are observation-only).
use candle_core::{Result, Tensor};
use candle_nn::Module;

#[derive(Debug, Clone)]
pub struct Conv2d {
    inner: candle_nn::Conv2d,
}

impl Module for Conv2d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.inner.forward(x)
    }
}

pub fn conv2d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: candle_nn::Conv2dConfig,
    vs: candle_nn::VarBuilder,
) -> Result<Conv2d> {
    let inner = candle_nn::conv2d(in_channels, out_channels, kernel_size, cfg, vs)?;
    Ok(Conv2d { inner })
}
