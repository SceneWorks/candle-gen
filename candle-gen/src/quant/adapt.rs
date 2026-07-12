//! Shared residual-capable linear for the **forward-time additive (unmerged) LoRA / LoKr** path
//! (sc-11091, epic 10765) — the one core [`AdaptLinear`] that the per-crate copies in
//! `candle-gen-wan/src/quant.rs` (sc-10094) and `candle-gen-anima/src/adapt.rs` (sc-10640) collapse
//! into, and the first-class seam a new consumer (qwen-image-edit Lightning on a packed q4/q8 tier)
//! adopts instead of hand-copying a third time. The candle twin of mlx-gen's `AdaptableLinear`.
//!
//! A frozen **base** — dense (`candle_nn::Linear`) or an MLX-**packed** [`super::QLinear`] that
//! dequantizes-on-forward (sc-7702) — plus zero or more **forward-time additive residuals**
//! `y = base(x) + Σ scale·((x·A)·B)`. Two residual forms, both **memory-free on a packed tier**:
//!   * `Lora` — `scale·((x·a)·b)`, two small matmuls (`a = downᵀ [in,rank]`, `b = upᵀ·(alpha/rank)
//!     [rank,out]`); never the `[out,in]` product, so a q4 base keeps its q4 footprint.
//!   * `LokrStructured` — the Kronecker vec-trick `vec(w1·reshape(x)·w2ᵀ)` (the candle port of
//!     mlx-gen's `Adapter::LokrStructured`), which applies a LoKr WITHOUT ever forming the `[out,in]`
//!     delta — the one path Wan's old copy lacked (it fell back to a dense `[out,in]` delta, packed-
//!     rejected).
//!
//! **The base weight is NEVER mutated.** On a packed q4/q8 DiT that is the whole point: the packed
//! codes survive load (`is_packed()` stays true, no dense `[out,in]` weight is materialized), and each
//! adapter rides *unmerged* as small matmuls per forward. On a **dense** base the fold path
//! ([`crate::train::merge`]) is still preferred for real runs, because a merge into a real `.weight`
//! is byte-for-byte what the chaos-sensitive samplers' goldens expect (`(W+δ)·x ≠ W·x + δ·x` to ~1
//! ULP — see [`crate::train::lora::reconstruct_lora_delta`]); the additive branch equals the fold to
//! f32 tolerance and is the *only* viable path where a merge is impossible (a quant-resident weight).
//! With **no** adapter attached the forward is byte-identical to the bare base, so swapping this in for
//! a projection leaves the plain-model / dense-fold paths unchanged.

use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

use super::{lin_gs, QLinear, MLX_GROUP_SIZE};
use crate::{CandleError, Result};

/// The frozen base weight — **dense** (`candle_nn::Linear`) or **MLX-packed** ([`super::QLinear`],
/// dequant-on-forward). Both compute `x·Wᵀ (+ b)`; neither is ever mutated by an adapter.
enum Base {
    Dense(Linear),
    Packed(QLinear),
}

impl Base {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Base::Dense(l) => l.forward(x),
            Base::Packed(q) => q.forward(x),
        }
    }

    fn is_packed(&self) -> bool {
        matches!(self, Base::Packed(_))
    }
}

/// A forward-time additive residual attached to an [`AdaptLinear`] — it never touches the frozen base
/// weight, so it is memory-free on a packed q4/q8 tier. Factors are held **f32** (the merge/train
/// dtype) and cast to the activation dtype per forward (they are tiny, so the cast is cheap).
enum Adapter {
    /// LoRA residual `scale·((x·a)·b)`: `a` `[in, rank]` (= `downᵀ`), `b` `[rank, out]` (= `upᵀ` with
    /// the `alpha/rank` ratio folded in at resolution). The **deferred two-small-matmul** form — never
    /// the `[out,in]` product — so it stays memory-free on any quant.
    Lora { a: Tensor, b: Tensor, scale: f64 },
    /// Structured LoKr residual via the Kronecker vec-trick — the FULL `(alpha/rank)·strength` scale is
    /// baked into [`LokrFactors::w2`], so a LoKr applies WITHOUT ever forming the `[out,in]` delta (the
    /// packed-capable path the whole hoist adds over Wan's old dense-only delta).
    LokrStructured { factors: LokrFactors },
}

impl Adapter {
    /// The residual this adapter contributes, in the activation dtype of `x`.
    fn residual(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Adapter::Lora { a, b, scale } => {
                let xd = x.dtype();
                let r = x
                    .broadcast_matmul(&a.to_dtype(xd)?)?
                    .broadcast_matmul(&b.to_dtype(xd)?)?;
                r * *scale
            }
            // The `scale` is already baked into `factors.w2`, so the vec-trick returns directly.
            Adapter::LokrStructured { factors } => factors.residual(x),
        }
    }
}

/// The two small Kronecker factors of a LoKr delta, kept **unmaterialized** for a deferred structured
/// forward (the candle port of mlx-gen's `LokrFactors`, sc-10713 / epic 10043). `ΔW = scale·kron(w1,
/// w2)` reshapes to the base's `[out, in]`, but the Kronecker–vector identity lets us apply it WITHOUT
/// ever forming that `[out, in]` tensor: with `w1` `[a, c]` and `w2` `[b, d]` (so `out = a·b`,
/// `in = c·d`), the residual `y = x·ΔWᵀ` is
///   `Y = w1 · X · w2ᵀ`  (then flatten row-major `[.., a, b] → [.., out]`),
/// where `X = reshape(x, [.., c, d])`. Two small matmuls (`[a,c]·[..,c,d]` then `·[d,b]`) touch only
/// the factor shapes — never `[out, in]` — so a LoKr applies on a packed q4/q8 base at the same memory
/// profile as a plain LoRA. The row-major kron ordering here matches [`crate::train::lora`]'s `kron2d`
/// (`out[i·b+k, j·d+l] = w1[i,j]·w2[k,l]`), so the structured residual equals the folded delta. The
/// full `(alpha/rank)·strength` scale is baked into [`w2`](Self::w2) at build time.
pub struct LokrFactors {
    /// `[a, c]` — the left Kronecker factor (`out = a·b`, `in = c·d`).
    w1: Tensor,
    /// `[b, d]` — the right Kronecker factor, with the full scale baked in.
    w2: Tensor,
    /// `a` — row count of `w1`; the flattened output index is `p·b + q`.
    a: usize,
    /// `b` — row count of `w2`.
    b: usize,
    /// `c` — col count of `w1`; the flattened input index is `r·d + s`.
    c: usize,
    /// `d` — col count of `w2`.
    d: usize,
}

impl LokrFactors {
    /// Build the small `[a,c]`/`[b,d]` Kronecker factors from a LoKr module's factors (full `w1`/`w2`
    /// or a low-rank `w_a·w_b` product — that product is bounded by the factor dims, NEVER `out×in`),
    /// baking the FULL `scale` into `w2`. The allocation-free counterpart to
    /// [`crate::train::lora::reconstruct_lokr_delta`] (which materializes the full `[out,in]` delta).
    ///
    /// **Scale differs from the fold path by design** (the two-conventions trap, sc-10578): the
    /// materialized `reconstruct_lokr_delta` bakes only `alpha/rank` and rides the user `strength` in
    /// the merge `scale`, whereas the structured residual carries no separate scale field, so the FULL
    /// `(alpha/rank)·strength` must be baked here (the caller derives it). Mismatching these two is a
    /// silent mis-scale, not a crash.
    ///
    /// Returns `Ok(None)` when the module has **no 2-D matrix form** deferrable via the vec-trick — a
    /// tucker/CP `lokr_t2` (conv-only), a factor that is not 2-D, or a base that does not factor as
    /// `a·b × c·d`. The caller then rejects (a packed base cannot materialize) or folds (a dense base).
    /// A missing `w1`/`w2` leg is a typed error (a malformed file), never a panic.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        scale: f64,
        base_shape: (usize, usize),
        w1: Option<&Tensor>,
        w1_a: Option<&Tensor>,
        w1_b: Option<&Tensor>,
        w2: Option<&Tensor>,
        w2_t2: Option<&Tensor>,
        w2_a: Option<&Tensor>,
        w2_b: Option<&Tensor>,
    ) -> Result<Option<Self>> {
        // A tucker/CP `w2` (a 4-D conv factor, lycoris `lokr_t2`) has no 2-D matrix form — not
        // deferrable via the vec-trick. The peft LoKr format never carries it; guard anyway so a conv
        // LoKr falls back to reject/fold rather than silently mis-applying.
        if w2_t2.is_some() {
            return Ok(None);
        }
        let f32d = |t: &Tensor| t.to_dtype(DType::F32);
        // The small Kronecker factors — full, or the low-rank inner product (bounded by the factor
        // dims, NEVER `out×in`): `w1_a @ w1_b` yields the small `[a, c]`, not the packed delta.
        let factor1 = match (w1, w1_a, w1_b) {
            (Some(w), _, _) => f32d(w)?,
            (_, Some(a), Some(b)) => f32d(a)?.matmul(&f32d(b)?)?,
            _ => {
                return Err(CandleError::Msg(
                    "lokr: w1 missing (need full lokr_w1 or lokr_w1_a·lokr_w1_b)".into(),
                ))
            }
        };
        let factor2 = match (w2, w2_a, w2_b) {
            (Some(w), _, _) => f32d(w)?,
            (_, Some(a), Some(b)) => f32d(a)?.matmul(&f32d(b)?)?,
            _ => {
                return Err(CandleError::Msg(
                    "lokr: w2 missing (need full lokr_w2 or lokr_w2_a·lokr_w2_b)".into(),
                ))
            }
        };
        // A conv-shaped (>2-D) factor is not a plain matrix — defer to reject/fold.
        if factor1.dims().len() != 2 || factor2.dims().len() != 2 {
            return Ok(None);
        }
        let (a, c) = (factor1.dims()[0], factor1.dims()[1]);
        let (b, d) = (factor2.dims()[0], factor2.dims()[1]);
        let (out_f, in_f) = base_shape;
        // The base must factor as `out = a·b`, `in = c·d` (a plain 2-D linear); anything else (a conv
        // weight with kernel dims, or a factor/base mismatch) is not this linear vec-trick.
        if a * b != out_f || c * d != in_f {
            return Ok(None);
        }
        // Bake the full scale into `w2` (keeps `w1` a clean copy); hold f32, contiguous for the matmuls.
        let w2 = (factor2 * scale)?.contiguous()?;
        let w1 = factor1.contiguous()?;
        Ok(Some(Self { w1, w2, a, b, c, d }))
    }

    /// Move the (CPU-read) factors onto `device` — the base lives on the DiT's device, so the residual
    /// matmul would be a device mismatch otherwise.
    pub fn to_device(&self, device: &Device) -> Result<Self> {
        Ok(Self {
            w1: self.w1.to_device(device)?,
            w2: self.w2.to_device(device)?,
            a: self.a,
            b: self.b,
            c: self.c,
            d: self.d,
        })
    }

    /// The deferred, allocation-free LoKr residual via the Kronecker–vector identity (`scale` already
    /// baked into [`w2`](Self::w2)). For an activation `x` of shape `[.., in]` (`in = c·d`): reshape to
    /// `[.., c, d]`, compute `Y = w1 · X · w2ᵀ` (`[.., a, b]`), and flatten row-major to `[.., out]`
    /// (`out = a·b`). The two matmuls touch only the small factor shapes — the full `[out, in]` delta
    /// is NEVER materialized, so this holds the same memory profile on a packed q4/q8 base as a plain
    /// LoRA. Factors are cast to the activation dtype so a bf16 stream runs bf16.
    fn residual(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let xd = x.dtype();
        let w1 = self.w1.to_dtype(xd)?;
        let w2t = self.w2.to_dtype(xd)?.t()?.contiguous()?; // [d, b]
        let dims = x.dims();
        let lead = &dims[..dims.len() - 1];
        let n: usize = lead.iter().product::<usize>().max(1);
        // Collapse every leading dim into ONE batch axis, then reshape `[N, in] → [N, c, d]` and batch-
        // matmul against explicit `[1, …]` factors that broadcast over `N`. Mirrors the mlx-gen
        // vec-trick and sidesteps any matmul-rank ambiguity when a leading dim happens to equal `a`/`b`.
        let xr = x.contiguous()?.reshape((n, self.c, self.d))?;
        let w1b = w1.reshape((1, self.a, self.c))?;
        let w2tb = w2t.reshape((1, self.d, self.b))?;
        // Y = w1 · X · w2ᵀ  → [N, a, b].
        let y = w1b.broadcast_matmul(&xr)?.broadcast_matmul(&w2tb)?;
        // [N, a, b] → [.., out] (out = a·b), restoring the original leading dims (row-major flatten).
        let mut ys = lead.to_vec();
        ys.push(self.a * self.b);
        y.contiguous()?.reshape(ys)
    }
}

/// A projection with a frozen base (dense or MLX-packed) and stacked forward-time LoRA/LoKr residuals.
/// Built dense ([`Self::linear`] / [`Self::linear_no_bias`] / [`Self::from_dense`] / [`Self::dense`] /
/// [`Self::dense_bias`]) or packed-detecting ([`Self::linear_detect`] / [`Self::linear_detect_gs`] /
/// [`Self::detect`]). `forward` = `base(x)` plus every residual, in push order; with no adapter it is
/// byte-identical to the bare base.
pub struct AdaptLinear {
    base: Base,
    /// The projection's logical `(out_features, in_features)` — captured at construction (recoverable
    /// even from a packed base, where the dense weight is never materialized) so the residual installer
    /// can shape-check a factor without reading the quantized weight back.
    out_features: usize,
    in_features: usize,
    /// Forward-time additive residuals, applied in push order (adapters stack). Empty on the plain /
    /// dense-fold path ⇒ forward is byte-identical to the bare base.
    adapters: Vec<Adapter>,
}

impl AdaptLinear {
    /// Wrap an already-built dense `candle_nn::Linear` with its logical `[out, in]` dims — the seam a
    /// caller uses when it holds a `Linear` directly (e.g. a folded-delta reference, or a test grid).
    pub fn from_dense(l: Linear, in_dim: usize, out_dim: usize) -> Self {
        Self {
            base: Base::Dense(l),
            out_features: out_dim,
            in_features: in_dim,
            adapters: Vec::new(),
        }
    }

    /// A biased dense `[out, in]` projection from `vb` (`{prefix}.weight` + `{prefix}.bias`), shape-
    /// checked exactly like `candle_nn::linear` — so it loads unchanged on `VarMap`-backed test
    /// fixtures.
    pub fn linear(in_dim: usize, out_dim: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self::from_dense(
            candle_nn::linear(in_dim, out_dim, vb)?,
            in_dim,
            out_dim,
        ))
    }

    /// A bias-less dense `[out, in]` projection from `vb` (`{prefix}.weight`).
    pub fn linear_no_bias(
        in_dim: usize,
        out_dim: usize,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        Ok(Self::from_dense(
            candle_nn::linear_no_bias(in_dim, out_dim, vb)?,
            in_dim,
            out_dim,
        ))
    }

    /// **Packed-detecting** `[out, in]` loader at an explicit MLX `group_size`: if `{base}.scales` is
    /// present under `vb` (a pre-quantized MLX tier), build a [`Base::Packed`] straight from the packed
    /// parts on `vb`'s device via the shared [`super::lin_gs`] — **no dense weight is materialized**.
    /// Otherwise the **dense** path is taken unchanged (`{base}.weight` [+ `{base}.bias`], shape-
    /// checked).
    ///
    /// `base` is the full dotted key prefix relative to `vb` (e.g. `to_out.0`), so the
    /// `.scales`/`.biases`/`.bias` siblings survive any `to_out.0`-style nesting: build the base string
    /// first, then detect — never `.pp()` past the scales sibling (the key-remap trap the shared loader
    /// guards).
    pub fn linear_detect_gs(
        in_dim: usize,
        out_dim: usize,
        vb: &VarBuilder,
        base: &str,
        bias: bool,
        group_size: usize,
    ) -> candle_core::Result<Self> {
        if vb.contains_tensor(&format!("{base}.scales")) {
            return Ok(Self {
                base: Base::Packed(lin_gs(vb, base, in_dim, out_dim, bias, group_size)?),
                out_features: out_dim,
                in_features: in_dim,
                adapters: Vec::new(),
            });
        }
        let sub = vb.pp(base);
        if bias {
            Self::linear(in_dim, out_dim, sub)
        } else {
            Self::linear_no_bias(in_dim, out_dim, sub)
        }
    }

    /// **Packed-detecting** `[out, in]` loader at the default MLX [`MLX_GROUP_SIZE`] (64) — thin wrapper
    /// over [`Self::linear_detect_gs`] for the callers whose hosted tiers pack at 64.
    pub fn linear_detect(
        in_dim: usize,
        out_dim: usize,
        vb: &VarBuilder,
        base: &str,
        bias: bool,
    ) -> candle_core::Result<Self> {
        Self::linear_detect_gs(in_dim, out_dim, vb, base, bias, MLX_GROUP_SIZE)
    }

    /// Bias-less, **packed-detecting** `[out, in]` projection from `{name}` on `vb` — the variant that
    /// **recovers the logical dims** from the packed `scales` shape (`[out, in/group]`) instead of
    /// taking them as arguments (the Anima DiT loader shape). If `{name}.scales` is present, load the
    /// MLX-packed triple at their native dtypes (u32 codes must NOT be cast through the vb's float
    /// dtype) at group 64; otherwise read the dense `{name}.weight` unchanged.
    pub fn detect(vb: &VarBuilder, name: &str) -> candle_core::Result<Self> {
        let scales_key = format!("{name}.scales");
        if vb.contains_tensor(&scales_key) {
            let device = vb.device().clone();
            let wq = vb.get_unchecked_dtype(&format!("{name}.weight"), DType::U32)?;
            let scales = vb.get_unchecked_dtype(&scales_key, DType::F32)?;
            let biases = vb.get_unchecked_dtype(&format!("{name}.biases"), DType::F32)?;
            // scales is [out, in/group] — recover the logical dims without touching the packed codes.
            let sdims = scales.dims();
            let out_features = sdims[0];
            let in_features = sdims[1] * MLX_GROUP_SIZE;
            let q = QLinear::from_packed_gs(&wq, &scales, &biases, None, MLX_GROUP_SIZE, &device)?;
            Ok(Self {
                base: Base::Packed(q),
                out_features,
                in_features,
                adapters: Vec::new(),
            })
        } else {
            let w = vb.get_unchecked(&format!("{name}.weight"))?;
            let (out_features, in_features) = (w.dims()[0], w.dims()[1]);
            Ok(Self {
                base: Base::Dense(Linear::new(w, None)),
                out_features,
                in_features,
                adapters: Vec::new(),
            })
        }
    }

    /// Bias-less **dense** `[out, in]` projection from `{name}.weight` (a component that never packs,
    /// e.g. Anima's conditioner q/k/v/o — dense bf16 in every tier).
    pub fn dense(vb: &VarBuilder, name: &str) -> candle_core::Result<Self> {
        let w = vb.get_unchecked(&format!("{name}.weight"))?;
        let (out_features, in_features) = (w.dims()[0], w.dims()[1]);
        Ok(Self {
            base: Base::Dense(Linear::new(w, None)),
            out_features,
            in_features,
            adapters: Vec::new(),
        })
    }

    /// **Dense** `[out, in]` projection with bias from `{name}.weight` + `{name}.bias`.
    pub fn dense_bias(vb: &VarBuilder, name: &str) -> candle_core::Result<Self> {
        let w = vb.get_unchecked(&format!("{name}.weight"))?;
        let (out_features, in_features) = (w.dims()[0], w.dims()[1]);
        let b = vb.get_unchecked(&format!("{name}.bias"))?;
        Ok(Self {
            base: Base::Dense(Linear::new(w, Some(b))),
            out_features,
            in_features,
            adapters: Vec::new(),
        })
    }

    /// `x·Wᵀ (+ b)` plus every attached additive residual, in push order. With no adapter this is
    /// byte-identical to the bare base forward.
    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let mut y = self.base.forward(x)?;
        for ad in &self.adapters {
            y = (y + ad.residual(x)?)?;
        }
        Ok(y)
    }

    /// Whether the base loaded from an MLX-packed tier (its codes are quantized) — used to gate the
    /// residual-vs-fold route and asserted by the tests (packed survives load, no dense weight
    /// materialized).
    pub fn is_packed(&self) -> bool {
        self.base.is_packed()
    }

    /// The base projection's `(out_features, in_features)` — the shape a resolved LoRA factor / LoKr
    /// delta is checked against, recoverable even from a packed base.
    pub fn base_shape(&self) -> (usize, usize) {
        (self.out_features, self.in_features)
    }

    /// The projection's contraction (`in_features`) — the last-dim an `[in, rank]` LoRA `a` factor
    /// contracts against.
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// The projection's output width (`out_features`).
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Whether any additive residual is attached.
    pub fn is_adapted(&self) -> bool {
        !self.adapters.is_empty()
    }

    /// Attach a forward-time **LoRA** residual `scale·((x·a)·b)`: `a` `[in, rank]` (= `downᵀ`), `b`
    /// `[rank, out]` (= `upᵀ` with `alpha/rank` folded in), `scale` the caller's per-adapter strength.
    /// Multiple pushes stack. Valid on **any** base — the base weight is untouched, so a packed q4/q8
    /// tier keeps its footprint.
    pub fn push_lora(&mut self, a: Tensor, b: Tensor, scale: f64) {
        self.adapters.push(Adapter::Lora { a, b, scale });
    }

    /// Attach a forward-time **structured LoKr** residual via the Kronecker vec-trick: the full
    /// `(alpha/rank)·strength` scale is already baked into `factors.w2`, so `[out,in]` is never
    /// materialized. Valid on **any** base — the base weight is untouched, so a packed q4/q8 tier keeps
    /// its footprint. Multiple pushes stack, and it mixes freely with LoRA residuals (push order).
    pub fn push_lokr_structured(&mut self, factors: LokrFactors) {
        self.adapters.push(Adapter::LokrStructured { factors });
    }
}

impl Module for AdaptLinear {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        AdaptLinear::forward(self, x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::lora::reconstruct_lokr_delta;
    use candle_core::safetensors::MmapedSafetensors;
    use candle_core::{Device, Tensor};
    use std::collections::HashMap;

    /// Test-side MLX Q4 packer (group 64): per-element 4-bit codes → u32 words (LSB-first nibbles).
    /// Returns `(wq [out, in/8] u32, scales [out, in/g], biases [out, in/g], affine grid [out, in])`.
    fn q4_packed(out_dim: usize, in_dim: usize) -> (Tensor, Tensor, Tensor, Vec<f32>) {
        let dev = Device::Cpu;
        let g = MLX_GROUP_SIZE;
        let codes: Vec<u8> = (0..out_dim * in_dim)
            .map(|i| ((i * 7 + i / 13) % 16) as u8)
            .collect();
        let gpr = in_dim / g;
        let groups = out_dim * gpr;
        // Small, BOUNDED scales/biases so the dequantized grid stays ~O(1). A large-magnitude grid
        // makes `base.forward` huge, and the residual-isolation test (adapted − base) then recovers a
        // tiny residual from a catastrophic f32 cancellation. Bounded via `% k` so it holds at any
        // group count.
        let scales: Vec<f32> = (0..groups)
            .map(|gi| 0.01 * ((gi % 5) as f32 + 1.0))
            .collect();
        let biases: Vec<f32> = (0..groups).map(|gi| -0.03 * (gi % 7) as f32).collect();
        let grid: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| {
                let (row, col) = (i / in_dim, i % in_dim);
                let gi = row * gpr + col / g;
                scales[gi] * codes[i] as f32 + biases[gi]
            })
            .collect();
        let words: Vec<u32> = codes
            .chunks_exact(8)
            .map(|c| {
                c.iter()
                    .enumerate()
                    .fold(0u32, |acc, (i, &q)| acc | ((q as u32 & 0xF) << (4 * i)))
            })
            .collect();
        let wq = Tensor::from_vec(words, (out_dim, in_dim / 8), &dev).unwrap();
        let s = Tensor::from_vec(scales, (out_dim, gpr), &dev).unwrap();
        let b = Tensor::from_vec(biases, (out_dim, gpr), &dev).unwrap();
        (wq, s, b, grid)
    }

    /// A per-call unique suffix so parallel tests never share a temp file (`cargo test` runs threads in
    /// ONE process, so a pid-only name would collide — one test truncating/deleting the file another is
    /// mid-mmap on → corrupt reads / flaky failures).
    static TMP_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    /// Build a packed [`AdaptLinear`] via [`AdaptLinear::detect`] on a written `.weight`/`.scales`/
    /// `.biases` triple (the round-trip the DiT loader takes) plus the affine grid it represents.
    fn packed_adapt(out_dim: usize, in_dim: usize) -> (AdaptLinear, Tensor) {
        let dev = Device::Cpu;
        let (wq, s, b, grid) = q4_packed(out_dim, in_dim);
        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert("p.weight".into(), wq);
        map.insert("p.scales".into(), s);
        map.insert("p.biases".into(), b);
        let uniq = TMP_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let tmp = std::env::temp_dir().join(format!(
            "adapt_core_{}_{}.safetensors",
            std::process::id(),
            uniq
        ));
        candle_core::safetensors::save(&map, &tmp).unwrap();
        // SAFETY: freshly written, single-reader for the test.
        let st = unsafe { MmapedSafetensors::new(&tmp).unwrap() };
        let vb = VarBuilder::from_backend(Box::new(st), DType::F32, dev.clone());
        let lin = AdaptLinear::detect(&vb, "p").unwrap();
        std::fs::remove_file(&tmp).ok();
        (
            lin,
            Tensor::from_vec(grid, (out_dim, in_dim), &dev).unwrap(),
        )
    }

    /// `detect` recovers the logical dims from the packed `scales` shape and keeps the base packed.
    #[test]
    fn detect_recovers_dims_and_stays_packed() {
        let (out_dim, in_dim) = (64usize, 128usize); // in divisible by group 64
        let (lin, _grid) = packed_adapt(out_dim, in_dim);
        assert!(lin.is_packed(), "`.scales` ⇒ packed base");
        assert_eq!(
            lin.base_shape(),
            (out_dim, in_dim),
            "dims from scales shape"
        );
        assert_eq!(lin.in_features(), in_dim);
        assert_eq!(lin.out_features(), out_dim);
        assert!(!lin.is_adapted());
    }

    /// The dense arm of `linear_detect` is byte-identical to the legacy `candle_nn::linear` read — a
    /// dense checkpoint (no `.scales`) loads dense, unchanged; a `.scales` sibling fires the packed arm.
    #[test]
    fn linear_detect_dense_and_packed_arms() {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (32usize, 64usize);
        // Dense arm: byte-identical to `Linear::new`.
        let w = Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev).unwrap();
        let bias = Tensor::randn(0f32, 1f32, (out_dim,), &dev).unwrap();
        let (wq, s, b, _grid) = q4_packed(out_dim, in_dim);
        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert("d.weight".into(), w.clone());
        map.insert("d.bias".into(), bias.clone());
        map.insert("p.weight".into(), wq);
        map.insert("p.scales".into(), s);
        map.insert("p.biases".into(), b);
        let uniq = TMP_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let tmp = std::env::temp_dir().join(format!(
            "adapt_detect_{}_{}.safetensors",
            std::process::id(),
            uniq
        ));
        candle_core::safetensors::save(&map, &tmp).unwrap();
        // SAFETY: freshly written, single-reader.
        let st = unsafe { MmapedSafetensors::new(&tmp).unwrap() };
        let vb = VarBuilder::from_backend(Box::new(st), DType::F32, dev.clone());

        let dense = AdaptLinear::linear_detect(in_dim, out_dim, &vb, "d", true).unwrap();
        assert!(!dense.is_packed(), "no `.scales` ⇒ dense");
        let x = Tensor::randn(0f32, 1f32, (4usize, in_dim), &dev).unwrap();
        let want = Linear::new(w, Some(bias));
        let dev_max = (dense.forward(&x).unwrap() - want.forward(&x).unwrap())
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(
            dev_max, 0.0,
            "dense arm deviates from the legacy linear read"
        );

        let packed = AdaptLinear::linear_detect(in_dim, out_dim, &vb, "p", false).unwrap();
        assert!(
            packed.is_packed(),
            "`.scales` ⇒ packed load, not a silent dense fallback"
        );
        std::fs::remove_file(&tmp).ok();
    }

    /// The additive LoRA residual `scale·((x·a)·b)` reproduces the **folded** `x·(W + δ)ᵀ` with
    /// `δ = (alpha/rank)·scale·(up·down)` on a dense f32 base — the additive==folded identity (tight in
    /// f32), the core weight-level property (no GPU needed).
    #[test]
    fn additive_lora_matches_folded_dense() {
        let dev = Device::Cpu;
        let (out_dim, in_dim, rank) = (48usize, 64usize, 4usize);
        let w = Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev).unwrap();
        let down = Tensor::randn(0f32, 1f32, (rank, in_dim), &dev).unwrap(); // A [rank, in]
        let up = Tensor::randn(0f32, 1f32, (out_dim, rank), &dev).unwrap(); // B [out, rank]
        let (alpha, user_scale) = (8.0f64, 0.7f64);
        let ratio = alpha / rank as f64;

        // a = downᵀ [in, rank]; b = (upᵀ·ratio) [rank, out].
        let a = down.t().unwrap().contiguous().unwrap();
        let b = (up.t().unwrap().contiguous().unwrap() * ratio).unwrap();
        let mut lin = AdaptLinear::from_dense(Linear::new(w.clone(), None), in_dim, out_dim);
        lin.push_lora(a, b, user_scale);
        assert!(lin.is_adapted());

        // Folded reference: δ = ratio·user_scale·(up·down); W_merged = W + δ.
        let delta = ((up.matmul(&down).unwrap()) * (ratio * user_scale)).unwrap();
        let folded = Linear::new((w + delta).unwrap(), None);

        let x = Tensor::randn(0f32, 1f32, (3usize, in_dim), &dev).unwrap();
        let dev_max = (lin.forward(&x).unwrap() - folded.forward(&x).unwrap())
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(dev_max < 1e-4, "additive vs folded deviates by {dev_max}");
    }

    /// A LoRA applied additively onto a **packed** base shifts the output, keeps the base **packed** (no
    /// dense weight materialized), and stays finite — the core acceptance on a quantized tier. A scale-0
    /// residual is an exact no-op (the mutation anchor: break the scale and this equality breaks).
    #[test]
    fn additive_lora_on_packed_shifts_stays_packed_and_scale0_is_noop() {
        let dev = Device::Cpu;
        let (out_dim, in_dim, rank) = (64usize, 128usize, 8usize);
        let (packed_base, _grid) = packed_adapt(out_dim, in_dim);
        assert!(packed_base.is_packed());

        let (mut adapted, _) = packed_adapt(out_dim, in_dim);
        let a = (Tensor::randn(0f32, 1f32, (in_dim, rank), &dev).unwrap() * 0.1).unwrap();
        let b = (Tensor::randn(0f32, 1f32, (rank, out_dim), &dev).unwrap() * 0.1).unwrap();
        adapted.push_lora(a.clone(), b.clone(), 1.0);
        assert!(adapted.is_packed(), "adapter must not un-pack the base");

        let x = Tensor::randn(0f32, 1f32, (4usize, in_dim), &dev).unwrap();
        let base_y = packed_base.forward(&x).unwrap();
        let adapted_y = adapted.forward(&x).unwrap();
        let shift = (adapted_y.sub(&base_y).unwrap())
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            shift > 1e-4,
            "additive LoRA on packed did not shift ({shift})"
        );
        assert!(
            adapted_y
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap()
                .is_finite(),
            "packed additive output non-finite"
        );

        // scale 0 ⇒ exact no-op vs the un-adapted packed base.
        let (mut zero, _) = packed_adapt(out_dim, in_dim);
        zero.push_lora(a, b, 0.0);
        let zero_dev = (zero.forward(&x).unwrap().sub(&base_y).unwrap())
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(
            zero_dev, 0.0,
            "scale-0 residual must be an exact no-op on packed"
        );
    }

    /// The **acceptance parity** on a packed base, isolated from the base's own quant error: the
    /// residual the adapter contributes equals `scale·((x·a)·b)` **exactly** (f32). `adapted.forward −
    /// base.forward` cancels the (bit-identical) packed base — the dequant-repack quant error included —
    /// leaving only the residual. This proves the LoRA is added **additively over the packed weight**
    /// (never folded into a re-quantized dense weight).
    #[test]
    fn additive_on_packed_adds_exact_residual_over_base() {
        let dev = Device::Cpu;
        let (out_dim, in_dim, rank) = (64usize, 128usize, 4usize);
        let (base, _grid) = packed_adapt(out_dim, in_dim);
        let (mut adapted, _grid) = packed_adapt(out_dim, in_dim); // bit-identical packed base
        let a = (Tensor::randn(0f32, 1f32, (in_dim, rank), &dev).unwrap() * 0.1).unwrap();
        let b = (Tensor::randn(0f32, 1f32, (rank, out_dim), &dev).unwrap() * 0.1).unwrap();
        let scale = 0.7f64;
        adapted.push_lora(a.clone(), b.clone(), scale);
        assert!(adapted.is_packed(), "base stays packed under the residual");

        let x = Tensor::randn(0f32, 1f32, (4usize, in_dim), &dev).unwrap();
        // The residual the adapter contributes = adapted − base (identical packed bases cancel exactly).
        let residual_actual = (adapted.forward(&x).unwrap() - base.forward(&x).unwrap()).unwrap();
        // Expected: scale·((x·a)·b).
        let residual_expected = ((x.matmul(&a).unwrap().matmul(&b).unwrap()) * scale).unwrap();
        let dev_max = (residual_actual - residual_expected)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            dev_max < 1e-5,
            "packed residual != scale·(x·a)·b (max diff {dev_max})"
        );
    }

    // ---- Structured (deferred) LoKr — the Kronecker vec-trick ---------------------------------------

    /// `x·ΔWᵀ` — the materialized-delta residual, the reference the vec-trick must reproduce.
    fn delta_residual(x: &Tensor, delta: &Tensor) -> Tensor {
        x.matmul(&delta.t().unwrap().contiguous().unwrap()).unwrap()
    }

    fn max_abs(t: &Tensor) -> f32 {
        t.abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    /// **Full-factor parity.** The structured vec-trick residual `vec(w1·X·w2ᵀ)` equals the
    /// materialized-delta residual `x·(kron(w1,w2))ᵀ` for a full `w1⊗w2` LoKr — proving the row-major
    /// Kronecker identity the whole port rests on, against candle's own `reconstruct_lokr_delta` fold.
    /// The built factors are the SMALL `[a,c]`/`[b,d]` matrices — the `[out,in]` delta is NEVER formed.
    #[test]
    fn structured_lokr_full_matches_reconstruct_delta() {
        let dev = Device::Cpu;
        let (a, b, c, d) = (2usize, 3, 4, 5);
        let (out, inp) = (a * b, c * d);
        let w1 = Tensor::from_vec(
            (0..(a * c))
                .map(|i| (i as f32 * 0.11).sin())
                .collect::<Vec<_>>(),
            (a, c),
            &dev,
        )
        .unwrap();
        let w2 = Tensor::from_vec(
            (0..(b * d))
                .map(|i| (i as f32 * 0.07).cos())
                .collect::<Vec<_>>(),
            (b, d),
            &dev,
        )
        .unwrap();
        let scale = 0.9f64;
        let delta = reconstruct_lokr_delta(
            Some(&w1),
            None,
            None,
            Some(&w2),
            None,
            None,
            1.0,
            1.0,
            scale as f32,
            (out, inp),
        )
        .unwrap();
        let factors = LokrFactors::build(
            scale,
            (out, inp),
            Some(&w1),
            None,
            None,
            Some(&w2),
            None,
            None,
            None,
        )
        .unwrap()
        .expect("a plain linear LoKr is deferrable");
        assert_eq!(factors.w1.dims(), &[a, c], "left factor stays [a,c]");
        assert_eq!(factors.w2.dims(), &[b, d], "right factor stays [b,d]");
        assert!(
            factors.w1.elem_count() + factors.w2.elem_count() < out * inp,
            "structured factors must be « the [out,in] delta"
        );

        let x = Tensor::from_vec(
            (0..(2 * inp))
                .map(|i| (i as f32 * 0.013 - 0.5).sin())
                .collect::<Vec<_>>(),
            (2, inp),
            &dev,
        )
        .unwrap();
        let want = delta_residual(&x, &delta);
        let got = factors.residual(&x).unwrap();
        assert_eq!(got.dims(), &[2, out]);
        let dev_max = max_abs(&(got.clone() - &want).unwrap());
        assert!(
            dev_max < 1e-4,
            "structured LoKr residual != materialized-delta residual ({dev_max})"
        );
        assert!(
            max_abs(&got) > 1e-2,
            "the LoKr residual must be materially non-zero"
        );
    }

    /// **Decomposed-factor parity.** Same identity for a low-rank LoKr (`w1_a·w1_b`, `w2_a·w2_b`): the
    /// inner products are materialized only as the SMALL `[a,c]`/`[b,d]` factors, never `[out,in]`.
    #[test]
    fn structured_lokr_decomposed_matches_reconstruct_delta() {
        let dev = Device::Cpu;
        let (a, b, c, d, r) = (3usize, 2, 5, 4, 2);
        let (out, inp) = (a * b, c * d);
        let mk = |rows: usize, cols: usize, seed: f32| {
            Tensor::from_vec(
                (0..(rows * cols))
                    .map(|i| (i as f32 * 0.09 + seed).sin() * 0.3)
                    .collect::<Vec<_>>(),
                (rows, cols),
                &dev,
            )
            .unwrap()
        };
        let (w1a, w1b) = (mk(a, r, 0.1), mk(r, c, 0.2)); // w1 = [a,c]
        let (w2a, w2b) = (mk(b, r, 0.3), mk(r, d, 0.4)); // w2 = [b,d]
        let scale = 1.3f64;
        let delta = reconstruct_lokr_delta(
            None,
            Some(&w1a),
            Some(&w1b),
            None,
            Some(&w2a),
            Some(&w2b),
            1.0,
            1.0,
            scale as f32,
            (out, inp),
        )
        .unwrap();
        let factors = LokrFactors::build(
            scale,
            (out, inp),
            None,
            Some(&w1a),
            Some(&w1b),
            None,
            None,
            Some(&w2a),
            Some(&w2b),
        )
        .unwrap()
        .expect("a decomposed linear LoKr is deferrable");
        assert_eq!(factors.w1.dims(), &[a, c]);
        assert_eq!(factors.w2.dims(), &[b, d]);

        let x = Tensor::from_vec(
            (0..inp)
                .map(|i| (i as f32 * 0.02).cos())
                .collect::<Vec<_>>(),
            (1, inp),
            &dev,
        )
        .unwrap();
        let dev_max =
            max_abs(&(factors.residual(&x).unwrap() - delta_residual(&x, &delta)).unwrap());
        assert!(
            dev_max < 1e-4,
            "decomposed structured LoKr != materialized delta ({dev_max})"
        );
    }

    /// **Acceptance parity on a PACKED base.** The structured LoKr installs on a packed q4 base, the
    /// base stays **packed** (no `[out,in]` weight materialized), and `packed_forward + residual`
    /// reproduces `packed_forward + folded_delta` within quant tolerance. Also the mutation anchor: a
    /// scale-0 LoKr is an exact no-op.
    #[test]
    fn structured_lokr_on_packed_matches_folded_and_stays_packed() {
        let dev = Device::Cpu;
        // in = c·d = 8·16 = 128 (divisible by group 64); out = a·b = 4·16 = 64.
        let (a, b, c, d) = (4usize, 16, 8, 16);
        let (out_dim, in_dim) = (a * b, c * d);
        let (base, _grid) = packed_adapt(out_dim, in_dim);
        let (mut adapted, _grid) = packed_adapt(out_dim, in_dim); // bit-identical packed base
        assert!(base.is_packed() && adapted.is_packed());

        let w1 = (Tensor::randn(0f32, 1f32, (a, c), &dev).unwrap() * 0.1).unwrap();
        let w2 = (Tensor::randn(0f32, 1f32, (b, d), &dev).unwrap() * 0.1).unwrap();
        let scale = 0.7f64;
        let factors = LokrFactors::build(
            scale,
            (out_dim, in_dim),
            Some(&w1),
            None,
            None,
            Some(&w2),
            None,
            None,
            None,
        )
        .unwrap()
        .expect("deferrable");
        assert!(factors.w1.elem_count() + factors.w2.elem_count() < out_dim * in_dim);
        adapted.push_lokr_structured(factors);
        assert!(
            adapted.is_packed(),
            "structured LoKr must not un-pack the base"
        );

        let x = Tensor::randn(0f32, 1f32, (4usize, in_dim), &dev).unwrap();
        let residual_actual = (adapted.forward(&x).unwrap() - base.forward(&x).unwrap()).unwrap();
        let delta = reconstruct_lokr_delta(
            Some(&w1),
            None,
            None,
            Some(&w2),
            None,
            None,
            1.0,
            1.0,
            scale as f32,
            (out_dim, in_dim),
        )
        .unwrap();
        let dev_max = max_abs(&(residual_actual.clone() - delta_residual(&x, &delta)).unwrap());
        assert!(
            dev_max < 1e-4,
            "packed structured LoKr residual != folded delta ({dev_max})"
        );
        assert!(
            max_abs(&residual_actual) > 1e-3,
            "the LoKr must shift the packed forward"
        );

        // Mutation: a scale-0 structured LoKr is an exact no-op over the packed base.
        let (mut zero, _) = packed_adapt(out_dim, in_dim);
        let f0 = LokrFactors::build(
            0.0,
            (out_dim, in_dim),
            Some(&w1),
            None,
            None,
            Some(&w2),
            None,
            None,
            None,
        )
        .unwrap()
        .unwrap();
        zero.push_lokr_structured(f0);
        let zero_dev = max_abs(&(zero.forward(&x).unwrap() - base.forward(&x).unwrap()).unwrap());
        assert_eq!(
            zero_dev, 0.0,
            "scale-0 structured LoKr must be an exact no-op"
        );
    }

    /// A tucker/CP `w2` (`lokr_t2`, conv-only) and a base that does not factor as `a·b × c·d` are both
    /// NOT deferrable via the 2-D vec-trick → `Ok(None)`, so the installer rejects them on a packed tier
    /// rather than materializing. A missing `w1`/`w2` leg is a typed error, never a panic.
    #[test]
    fn structured_lokr_non_deferrable_and_missing_legs() {
        let dev = Device::Cpu;
        let w1 = Tensor::zeros((3usize, 4), DType::F32, &dev).unwrap();
        let w2a = Tensor::zeros((2usize, 4), DType::F32, &dev).unwrap();
        let w2b = Tensor::zeros((2usize, 5), DType::F32, &dev).unwrap();
        let t2 = Tensor::zeros((2usize, 2, 3, 3), DType::F32, &dev).unwrap();
        // Tucker `w2_t2` present → None (the guard fires before any shape check).
        let got = LokrFactors::build(
            1.0,
            (24, 180),
            Some(&w1),
            None,
            None,
            None,
            Some(&t2),
            Some(&w2a),
            Some(&w2b),
        )
        .unwrap();
        assert!(got.is_none(), "tucker/CP LoKr must be non-deferrable");

        // A base that does not factor as a·b × c·d (here a·b = 3·2 = 6 ≠ out = 7) → None.
        let w2 = Tensor::zeros((2usize, 5), DType::F32, &dev).unwrap();
        let mism = LokrFactors::build(
            1.0,
            (7, 20),
            Some(&w1),
            None,
            None,
            Some(&w2),
            None,
            None,
            None,
        )
        .unwrap();
        assert!(
            mism.is_none(),
            "a base that doesn't factor a·b×c·d is non-deferrable"
        );

        // A missing w2 leg (no full, no a/b) is a typed error, not a panic.
        let err = LokrFactors::build(1.0, (6, 20), Some(&w1), None, None, None, None, None, None);
        assert!(err.is_err(), "missing w2 must be a typed error");
    }
}
