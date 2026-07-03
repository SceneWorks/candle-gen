//! Compatibility re-export of the shared [`Weights`](candle_gen::weights::Weights) key→`Tensor` map.
//!
//! The loader itself was hoisted into the `candle-gen` core crate (F-060, sc-9044) so unrelated
//! provider crates (FLUX IP-Adapter, PuLID) no longer pull in the whole ~12k-LOC SDXL crate just for
//! it. This crate keeps the old `candle_gen_sdxl::weights::Weights` path working by re-exporting the
//! core type verbatim — SDXL's own IP-Adapter / ControlNet / vision-encoder loaders reference it here.
pub use candle_gen::weights::Weights;
