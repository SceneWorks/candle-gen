//! LR-schedule helpers (sc-1551) — re-exported from gen-core so candle and MLX share **one**
//! implementation of the constant/linear/cosine + warmup policy. The trainer converts micro-steps to
//! optimizer-update counts with [`schedule_updates`], then scales the base LR each update by
//! [`lr_multiplier`].
pub use crate::gen_core::train::schedule::{lr_multiplier, schedule_updates};
pub use crate::gen_core::train::LrSchedule;
