//! The trainable LoRA/LoKr adapter seam (sc-5165).
//!
//! [`LoraLinear`] is the candle analog of a PEFT-wrapped `nn::Linear`: a **frozen** base `Linear`
//! plus an optional low-rank (LoRA) or Kronecker (LoKr) residual added *in the forward*. The residual
//! factors are held as `Var`-backed [`Tensor`]s — storage-sharing clones of the trainer's `Var`s — so:
//!
//!  * the optimizer's `Var::set` mutates that storage in place, and the **next forward reads the new
//!    values** with no re-install and no model rebuild (candle is eager: each forward re-reads the
//!    factor storage at matmul time); and
//!  * the clones keep the `Var`'s tensor-id and variable flag, so `loss.backward()` records them as
//!    leaves and `GradStore::get(var)` returns the factor gradient.
//!
//! Factors are **f32** regardless of the train dtype (master-weights pattern, per the gen-core
//! `TrainingConfig` contract); the forward casts them to the activation dtype for the matmul (a
//! differentiable cast, so grads flow back to the f32 `Var`s). The LoKr residual is reconstructed the
//! same way the inference loader does — `ΔW = (alpha/rank)·kron(w1, w2)` at f32 — so a trained adapter
//! round-trips exactly (mirrors mlx-gen's `reconstruct_lokr_delta`, SDXL f32 path).
//!
//! A model exposes its adaptable projections by implementing [`LoraHost`]; [`build_lora_targets`] /
//! [`build_lokr_targets`] then walk the host, size + initialize the factors per target, install them,
//! and return a [`LoraSet`] (the flat `Var` list for the optimizer + the per-target metadata for
//! checkpoint save). This keeps the harness model-agnostic — SDXL, Z-Image, and Wan reuse it.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, Tensor, Var};
use candle_nn::{Linear, Module, VarBuilder};
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};

use crate::{CandleError, Result};

/// PEFT gaussian-init standard deviation for the LoRA `A` / LoKr `w1` factor (diffusers/PEFT
/// `init_lora_weights="gaussian"` and LyCORIS `init_weights` both use 0.02). The second leg (`B`, or
/// the LoKr `w2`/`w2_b`) starts at zero, so the adapter is the identity at step 0.
const INIT_STD: f32 = 0.02;

/// The SDXL default LoRA target suffixes — the attention projections (matches the torch
/// `DEFAULT_LORA_TARGET_MODULES` and the MLX trainer). `to_out.0` is the first element of diffusers'
/// `to_out` `ModuleList`, so its path segment literally contains the `.0`.
pub const SDXL_ATTN_TARGETS: [&str; 4] = ["to_q", "to_k", "to_v", "to_out.0"];

/// The PEFT key prefix SDXL writes (what `peft.save_pretrained()` emits and the SDXL loader's PEFT
/// classifier expects). The DiT families use `""` (bare dotted paths).
pub const SDXL_PEFT_PREFIX: &str = "base_model.model.unet.";

/// Which adapter parameterization a [`LoraSet`] carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdapterKind {
    /// Standard low-rank `B·A` residual.
    Lora,
    /// LyCORIS Kronecker-product residual.
    Lokr,
}

impl AdapterKind {
    /// The `networkType` metadata string written into the adapter `.safetensors`.
    pub fn network_type(self) -> &'static str {
        match self {
            AdapterKind::Lora => "lora",
            AdapterKind::Lokr => "lokr",
        }
    }
}

/// The trainable second Kronecker leg of a LoKr residual: either a full `[out_b, in_b]` matrix or a
/// low-rank product `w2_a[out_b, rank] · w2_b[rank, in_b]`.
#[derive(Debug, Clone)]
enum LokrW2 {
    Full(Tensor),
    LowRank { a: Tensor, b: Tensor },
}

/// The trainable residual spliced into a [`LoraLinear`]'s forward. Holds storage-sharing clones of
/// the trainer's `Var`s (see the module docs) — never owned weight copies.
#[derive(Debug, Clone)]
enum Adapter {
    /// `down`: `A` `[rank, in]`; `up`: `B` `[out, rank]`; residual = `scale · (x·Aᵀ)·Bᵀ`.
    Lora {
        down: Tensor,
        up: Tensor,
        scale: f64,
    },
    /// `w1` `[out_a, in_a]`, `w2` (full/low-rank) reconstructing `[out_b, in_b]`; residual =
    /// `x · ΔWᵀ` with `ΔW = scale · kron(w1, w2)` reshaped to `[out, in]`.
    Lokr {
        w1: Tensor,
        w2: LokrW2,
        out_f: usize,
        in_f: usize,
        scale: f64,
    },
}

/// 2-D Kronecker product `kron(a[m,n], b[p,q]) = [m·p, n·q]` via broadcast — differentiable, so grads
/// flow to `a`/`b`. `out[i·p+k, j·q+l] = a[i,j]·b[k,l]`.
fn kron2d(a: &Tensor, b: &Tensor) -> candle_core::Result<Tensor> {
    let (m, n) = a.dims2()?;
    let (p, q) = b.dims2()?;
    let a4 = a.reshape((m, 1, n, 1))?;
    let b4 = b.reshape((1, p, 1, q))?;
    a4.broadcast_mul(&b4)?.reshape((m * p, n * q))
}

/// A frozen base `Linear` with an optional trainable LoRA/LoKr residual. Implements
/// [`Module`](candle_nn::Module) so it drops into a vendored model exactly where an `nn::Linear` was,
/// and carries its own PEFT-style `path` (captured from the `VarBuilder` prefix at construction) so a
/// [`LoraHost`] visitor can route adapters without threading prefixes through the module tree.
#[derive(Debug, Clone)]
pub struct LoraLinear {
    base: Linear,
    in_features: usize,
    out_features: usize,
    path: String,
    adapter: Option<Adapter>,
}

impl LoraLinear {
    /// Wrap an already-built frozen base `Linear` known to map `in_features -> out_features`, at the
    /// given PEFT module `path`.
    pub fn from_linear(
        base: Linear,
        in_features: usize,
        out_features: usize,
        path: String,
    ) -> Self {
        Self {
            base,
            in_features,
            out_features,
            path,
            adapter: None,
        }
    }

    /// The PEFT module path (e.g. `down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q`),
    /// captured from the `VarBuilder` prefix at construction. Drives target matching + save keys.
    pub fn path(&self) -> &str {
        &self.path
    }

    pub fn in_features(&self) -> usize {
        self.in_features
    }

    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Whether a trainable residual is currently installed.
    pub fn is_adapted(&self) -> bool {
        self.adapter.is_some()
    }

    /// Install a LoRA residual. `down`/`up` are expected to be `Var`-backed (storage-sharing) f32
    /// tensors of shape `[rank, in]` / `[out, rank]`; `scale = alpha / rank`.
    pub fn install_lora(&mut self, down: Tensor, up: Tensor, scale: f64) {
        self.adapter = Some(Adapter::Lora { down, up, scale });
    }

    /// Drop any installed residual (back to the frozen base — the inference path with no adapter).
    pub fn clear(&mut self) {
        self.adapter = None;
    }

    fn install_lokr(&mut self, w1: Tensor, w2: LokrW2, scale: f64) {
        self.adapter = Some(Adapter::Lokr {
            w1,
            w2,
            out_f: self.out_features,
            in_f: self.in_features,
            scale,
        });
    }
}

impl Module for LoraLinear {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let y = self.base.forward(x)?;
        match &self.adapter {
            None => Ok(y),
            // Factors are f32; cast to the activation dtype for the matmul. The cast is
            // differentiable, so grads flow back to the f32 `Var`s (master-weights). The factor
            // tensors share storage with those `Var`s, so this reads the current optimizer-updated value.
            Some(Adapter::Lora { down, up, scale }) => {
                let xd = x.dtype();
                let down = down.to_dtype(xd)?;
                let up = up.to_dtype(xd)?;
                let lora = x.broadcast_matmul(&down.t()?)?.broadcast_matmul(&up.t()?)?;
                y + (lora * *scale)?
            }
            Some(Adapter::Lokr {
                w1,
                w2,
                out_f,
                in_f,
                scale,
            }) => {
                let factor2 = match w2 {
                    LokrW2::Full(w) => w.clone(),
                    LokrW2::LowRank { a, b } => a.matmul(b)?, // [out_b, rank] · [rank, in_b]
                };
                // ΔW = scale · kron(w1, w2) at f32, reshaped to [out, in] (kron already yields that
                // shape for the linear case; the reshape is a safety no-op). Cast to the activation
                // dtype for the residual matmul x·ΔWᵀ.
                let delta = kron2d(w1, &factor2)?.reshape((*out_f, *in_f))?;
                let delta = (delta * *scale)?.to_dtype(x.dtype())?;
                y + x.broadcast_matmul(&delta.t()?)?
            }
        }
    }
}

/// Build a frozen base `Linear` (no bias) and wrap it as a [`LoraLinear`], recording the
/// `VarBuilder`'s current prefix as the PEFT path. Drop-in replacement for `candle_nn::linear_no_bias`
/// inside a vendored, trainable model.
pub fn lora_linear_no_bias(
    in_f: usize,
    out_f: usize,
    vs: VarBuilder,
) -> candle_core::Result<LoraLinear> {
    let path = vs.prefix();
    let base = candle_nn::linear_no_bias(in_f, out_f, vs)?;
    Ok(LoraLinear::from_linear(base, in_f, out_f, path))
}

/// Build a frozen base `Linear` (with bias) and wrap it as a [`LoraLinear`]. Drop-in replacement for
/// `candle_nn::linear`. The adapter residual adapts only the weight; the base bias is frozen.
pub fn lora_linear(in_f: usize, out_f: usize, vs: VarBuilder) -> candle_core::Result<LoraLinear> {
    let path = vs.prefix();
    let base = candle_nn::linear(in_f, out_f, vs)?;
    Ok(LoraLinear::from_linear(base, in_f, out_f, path))
}

/// A model that exposes its adaptable [`LoraLinear`]s for the harness to install adapters into. The
/// candle analog of the MLX `AdaptableHost`. Implementors recurse their module tree, invoking `f` once
/// per adaptable projection; each `LoraLinear` already carries its PEFT `path`, so no prefix threading
/// is needed.
pub trait LoraHost {
    fn visit_lora_mut(&mut self, f: &mut dyn FnMut(&mut LoraLinear) -> Result<()>) -> Result<()>;
}

/// One installed target: its PEFT path plus the trainer-owned factor `Var`s keyed by their save-key
/// suffix (e.g. `("lora_A.weight", a)` / `("lokr_w1", w1)`). The same `Var`s are flattened into
/// [`LoraSet::vars`] for the optimizer; here they carry the suffix the checkpoint writer needs.
#[derive(Debug, Clone)]
pub struct AdapterTarget {
    pub path: String,
    factors: Vec<(&'static str, Var)>,
}

/// The result of installing adapters onto a host: the flat `Var` list to optimize, the per-target
/// metadata to save, and the network descriptors echoed into the adapter metadata.
#[derive(Debug, Clone)]
pub struct LoraSet {
    pub kind: AdapterKind,
    pub rank: u32,
    pub alpha: f32,
    /// LoKr block-split factor (`-1` = auto); unused for plain LoRA.
    pub decompose_factor: i32,
    /// Every trainable factor, for the optimizer.
    pub vars: Vec<Var>,
    targets: Vec<AdapterTarget>,
}

impl LoraSet {
    /// `scale = alpha / rank` (the residual multiplier).
    pub fn scale(&self) -> f64 {
        self.alpha as f64 / self.rank.max(1) as f64
    }

    /// Number of adapted projections.
    pub fn len(&self) -> usize {
        self.targets.len()
    }

    pub fn is_empty(&self) -> bool {
        self.targets.is_empty()
    }
}

/// PEFT suffix match: `suffix` matches a module `path` iff the path equals it or ends with `.{suffix}`
/// (so `to_q` matches `…attn1.to_q` but not `…attn1.to_qx`, and `to_out.0` matches `…attn1.to_out.0`).
fn path_matches(path: &str, suffix: &str) -> bool {
    path == suffix || path.ends_with(&format!(".{suffix}"))
}

/// LyCORIS dimension factorization: split `dimension` into `(a, b)` with `a·b == dimension` and
/// `a ≤ b`. `factor > 0` requests a block size (the pair containing `factor`, smaller-first); `-1`
/// (auto) picks the most balanced divisor pair. Faithful port of the MLX/LyCORIS `factorization`.
pub fn factorization(dimension: usize, factor: i32) -> (usize, usize) {
    if factor > 0 {
        let f = factor as usize;
        if dimension % f == 0 {
            let n = dimension / f;
            return if f > n { (n, f) } else { (f, n) };
        }
    }
    // auto (or a `factor` that doesn't divide): climb to the most balanced divisor pair, bounded by
    // `factor` (= dimension when auto).
    let cap = if factor < 0 {
        dimension
    } else {
        factor as usize
    };
    let (mut m, mut n) = (1usize, dimension);
    let mut length = m + n;
    while m < n {
        let mut new_m = m + 1;
        while dimension % new_m != 0 {
            new_m += 1;
        }
        let new_n = dimension / new_m;
        if new_m + new_n > length || new_m > cap {
            break;
        }
        m = new_m;
        n = new_n;
        length = m + n;
    }
    if m > n {
        (n, m)
    } else {
        (m, n)
    }
}

/// Deterministic, launch-portable factor init: draw `rows·cols` `N(0, std²)` values from a seeded CPU
/// `StdRng` (NOT candle's device RNG — same reasoning as the sc-3673 initial-noise path), build the
/// tensor on CPU, and move it to `device`. Returned as a trainable f32 `Var`.
fn gaussian_var(
    rows: usize,
    cols: usize,
    std: f32,
    rng: &mut StdRng,
    device: &Device,
) -> Result<Var> {
    let data: Vec<f32> = (0..rows * cols)
        .map(|_| {
            let z: f32 = StandardNormal.sample(rng);
            std * z
        })
        .collect();
    let t = Tensor::from_vec(data, (rows, cols), &Device::Cpu)?.to_device(device)?;
    Ok(Var::from_tensor(&t)?)
}

fn zero_var(rows: usize, cols: usize, device: &Device) -> Result<Var> {
    Ok(Var::from_tensor(&Tensor::zeros(
        (rows, cols),
        DType::F32,
        device,
    )?)?)
}

/// Install LoRA adapters on `host` for every adaptable projection whose path matches one of
/// `target_suffixes`. `A ~ N(0, 0.02²)` `[rank, in]`, `B = 0` `[out, rank]` (identity at step 0).
/// Factors are f32 on `device`; init is seeded by `seed` for reproducibility.
pub fn build_lora_targets(
    host: &mut dyn LoraHost,
    target_suffixes: &[String],
    rank: u32,
    alpha: f32,
    seed: u64,
    device: &Device,
) -> Result<LoraSet> {
    if rank == 0 {
        return Err(CandleError::Msg("lora rank must be >= 1".into()));
    }
    let r = rank as usize;
    let scale = alpha as f64 / r as f64;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut vars: Vec<Var> = Vec::new();
    let mut targets: Vec<AdapterTarget> = Vec::new();

    host.visit_lora_mut(&mut |lin: &mut LoraLinear| {
        if !target_suffixes.iter().any(|s| path_matches(lin.path(), s)) {
            return Ok(());
        }
        let (in_f, out_f) = (lin.in_features(), lin.out_features());
        let down = gaussian_var(r, in_f, INIT_STD, &mut rng, device)?; // A [rank, in]
        let up = zero_var(out_f, r, device)?; // B [out, rank]
        lin.install_lora(down.as_tensor().clone(), up.as_tensor().clone(), scale);
        vars.push(down.clone());
        vars.push(up.clone());
        targets.push(AdapterTarget {
            path: lin.path().to_string(),
            factors: vec![("lora_A.weight", down), ("lora_B.weight", up)],
        });
        Ok(())
    })?;

    if targets.is_empty() {
        return Err(CandleError::Msg(format!(
            "no LoRA targets matched suffixes {target_suffixes:?} on the host"
        )));
    }
    Ok(LoraSet {
        kind: AdapterKind::Lora,
        rank,
        alpha,
        decompose_factor: -1,
        vars,
        targets,
    })
}

/// Install LoKr adapters on `host` for every matching projection. The weight `[out,in]` factors as
/// `kron(w1[out_a,in_a], w2[out_b,in_b])`; `w2` is low-ranked to `rank` when `rank < min(out_b,in_b)`.
/// `w1 ~ N(0,0.02)`; the second leg is zero-init (`w2` full, or `w2_b` low-rank) so the initial delta
/// is exactly 0. `decompose_factor` (`-1` = auto) is the block-split knob. Mirrors the MLX
/// `build_lokr_targets` (init + key layout); the residual is reconstructed at f32 (SDXL path).
pub fn build_lokr_targets(
    host: &mut dyn LoraHost,
    target_suffixes: &[String],
    rank: u32,
    alpha: f32,
    decompose_factor: i32,
    seed: u64,
    device: &Device,
) -> Result<LoraSet> {
    if rank == 0 {
        return Err(CandleError::Msg("lokr rank must be >= 1".into()));
    }
    let r = rank as usize;
    let scale = alpha as f64 / r as f64;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut vars: Vec<Var> = Vec::new();
    let mut targets: Vec<AdapterTarget> = Vec::new();

    host.visit_lora_mut(&mut |lin: &mut LoraLinear| {
        if !target_suffixes.iter().any(|s| path_matches(lin.path(), s)) {
            return Ok(());
        }
        let (in_f, out_f) = (lin.in_features(), lin.out_features());
        let (out_a, out_b) = factorization(out_f, decompose_factor);
        let (in_a, in_b) = factorization(in_f, decompose_factor);

        let w1 = gaussian_var(out_a, in_a, INIT_STD, &mut rng, device)?;
        vars.push(w1.clone());
        let mut factors: Vec<(&'static str, Var)> = vec![("lokr_w1", w1.clone())];

        let runtime_w2;
        if r < out_b.min(in_b) {
            // Low-rank w2 = w2_a @ w2_b; w2_b zero-init ⇒ delta starts at 0.
            let w2a = gaussian_var(out_b, r, INIT_STD, &mut rng, device)?;
            let w2b = zero_var(r, in_b, device)?;
            vars.push(w2a.clone());
            vars.push(w2b.clone());
            factors.push(("lokr_w2_a", w2a.clone()));
            factors.push(("lokr_w2_b", w2b.clone()));
            runtime_w2 = LokrW2::LowRank {
                a: w2a.as_tensor().clone(),
                b: w2b.as_tensor().clone(),
            };
        } else {
            // Full w2, zero-init.
            let w2 = zero_var(out_b, in_b, device)?;
            vars.push(w2.clone());
            factors.push(("lokr_w2", w2.clone()));
            runtime_w2 = LokrW2::Full(w2.as_tensor().clone());
        }
        lin.install_lokr(w1.as_tensor().clone(), runtime_w2, scale);
        targets.push(AdapterTarget {
            path: lin.path().to_string(),
            factors,
        });
        Ok(())
    })?;

    if targets.is_empty() {
        return Err(CandleError::Msg(format!(
            "no LoKr targets matched suffixes {target_suffixes:?} on the host"
        )));
    }
    Ok(LoraSet {
        kind: AdapterKind::Lokr,
        rank,
        alpha,
        decompose_factor,
        vars,
        targets,
    })
}

/// Collect a target's factor tensors as CPU/f32 `(key, tensor)` save entries under `prefix`.
fn factor_entries(set: &LoraSet, prefix: &str) -> Result<Vec<(String, Tensor)>> {
    let mut out = Vec::with_capacity(set.targets.len() * 3);
    for t in &set.targets {
        for (suffix, var) in &t.factors {
            let v = var
                .as_tensor()
                .to_device(&Device::Cpu)?
                .to_dtype(DType::F32)?
                .contiguous()?;
            out.push((format!("{prefix}{}.{suffix}", t.path), v));
        }
    }
    Ok(out)
}

fn write_safetensors(
    tensors: Vec<(String, Tensor)>,
    meta: HashMap<String, String>,
    path: &Path,
) -> Result<()> {
    safetensors::serialize_to_file(tensors, Some(meta), path)
        .map_err(|e| CandleError::Msg(format!("save adapter {}: {e}", path.display())))?;
    Ok(())
}

/// Write a LoRA [`LoraSet`] as a PEFT-format `.safetensors`: keys `{prefix}{path}.lora_A.weight`
/// (`[rank, in]`), `{prefix}{path}.lora_B.weight` (`[out, rank]`), and a per-target scalar
/// `{prefix}{path}.alpha`, plus `networkType`/`rank`/`alpha` metadata (the candle save path that
/// candle-core's own `save` cannot produce — it passes `None` for metadata). `prefix` is
/// [`SDXL_PEFT_PREFIX`] for SDXL, `""` for the DiT families. Matches the MLX `save_lora_peft`.
pub fn save_lora_peft(
    set: &LoraSet,
    prefix: &str,
    extra_meta: &HashMap<String, String>,
    path: &Path,
) -> Result<()> {
    if set.kind != AdapterKind::Lora {
        return Err(CandleError::Msg(
            "save_lora_peft called on a non-LoRA set".into(),
        ));
    }
    let mut tensors = factor_entries(set, prefix)?;
    // Per-target scalar `.alpha` (PEFT reload contract).
    for t in &set.targets {
        tensors.push((
            format!("{prefix}{}.alpha", t.path),
            Tensor::from_vec(vec![set.alpha], (1,), &Device::Cpu)?,
        ));
    }
    let mut meta: HashMap<String, String> = HashMap::new();
    meta.insert("networkType".into(), set.kind.network_type().into());
    meta.insert("rank".into(), set.rank.to_string());
    meta.insert("alpha".into(), set.alpha.to_string());
    for (k, v) in extra_meta {
        meta.entry(k.clone()).or_insert_with(|| v.clone());
    }
    write_safetensors(tensors, meta, path)
}

/// Write a LoKr [`LoraSet`] as `.safetensors`: bare keys `{path}.lokr_w1` + (`lokr_w2` |
/// `lokr_w2_a`/`lokr_w2_b`), with `networkType`/`rank`/`alpha`/`decomposeFactor` metadata. No key
/// prefix (the SDXL LoKr loader accepts a `base_model.model.unet.` prefix but bare keys resolve for
/// every family). Matches the MLX `save_lokr` (integer rank/alpha rendering).
pub fn save_lokr(set: &LoraSet, extra_meta: &HashMap<String, String>, path: &Path) -> Result<()> {
    if set.kind != AdapterKind::Lokr {
        return Err(CandleError::Msg(
            "save_lokr called on a non-LoKr set".into(),
        ));
    }
    let tensors = factor_entries(set, "")?;
    let mut meta: HashMap<String, String> = HashMap::new();
    meta.insert("networkType".into(), set.kind.network_type().into());
    meta.insert("rank".into(), (set.rank as i64).to_string());
    meta.insert("alpha".into(), (set.alpha as i64).to_string());
    meta.insert("decomposeFactor".into(), set.decompose_factor.to_string());
    for (k, v) in extra_meta {
        meta.entry(k.clone()).or_insert_with(|| v.clone());
    }
    write_safetensors(tensors, meta, path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::IndexOp;

    fn fixed_linear(weight: &[f32], out_f: usize, in_f: usize) -> LoraLinear {
        let w = Tensor::from_vec(weight.to_vec(), (out_f, in_f), &Device::Cpu).unwrap();
        LoraLinear::from_linear(Linear::new(w, None), in_f, out_f, "test.to_q".into())
    }

    #[test]
    fn no_adapter_is_base_linear() {
        let lin = fixed_linear(&[1.0, 0.0, 0.0, 1.0], 2, 2);
        let x = Tensor::from_vec(vec![3.0f32, 5.0], (1, 2), &Device::Cpu).unwrap();
        let y = lin
            .forward(&x)
            .unwrap()
            .i(0)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(y, vec![3.0, 5.0]);
    }

    #[test]
    fn zero_b_residual_is_identity() {
        let mut lin = fixed_linear(&[1.0, 0.0, 0.0, 1.0], 2, 2);
        let down = Tensor::from_vec(vec![0.5f32, -0.3, 0.1, 0.2], (2, 2), &Device::Cpu).unwrap();
        let up = Tensor::zeros((2, 2), DType::F32, &Device::Cpu).unwrap();
        lin.install_lora(down, up, 1.0);
        let x = Tensor::from_vec(vec![3.0f32, 5.0], (1, 2), &Device::Cpu).unwrap();
        let y = lin
            .forward(&x)
            .unwrap()
            .i(0)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(y, vec![3.0, 5.0]);
    }

    #[test]
    fn lora_residual_math() {
        let mut lin = fixed_linear(&[0.0, 0.0, 0.0, 0.0], 2, 2);
        let down = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], (2, 2), &Device::Cpu).unwrap();
        let up = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], (2, 2), &Device::Cpu).unwrap();
        lin.install_lora(down, up, 2.0);
        let x = Tensor::from_vec(vec![3.0f32, 5.0], (1, 2), &Device::Cpu).unwrap();
        let y = lin
            .forward(&x)
            .unwrap()
            .i(0)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(y, vec![6.0, 10.0]);
    }

    #[test]
    fn backward_reaches_factors() {
        let w = Tensor::zeros((2, 2), DType::F32, &Device::Cpu).unwrap();
        let mut lin = LoraLinear::from_linear(Linear::new(w, None), 2, 2, "t".into());
        let down = Var::from_tensor(
            &Tensor::from_vec(vec![0.1f32, 0.2, 0.3, 0.4], (2, 2), &Device::Cpu).unwrap(),
        )
        .unwrap();
        let up = Var::from_tensor(
            &Tensor::from_vec(vec![0.5f32, 0.6, 0.7, 0.8], (2, 2), &Device::Cpu).unwrap(),
        )
        .unwrap();
        lin.install_lora(down.as_tensor().clone(), up.as_tensor().clone(), 1.0);
        let x = Tensor::from_vec(vec![1.0f32, 2.0], (1, 2), &Device::Cpu).unwrap();
        let loss = lin.forward(&x).unwrap().sqr().unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();
        assert!(
            grads.get(down.as_tensor()).is_some(),
            "A factor must receive a gradient"
        );
        assert!(
            grads.get(up.as_tensor()).is_some(),
            "B factor must receive a gradient"
        );
    }

    #[test]
    fn optimizer_update_seen_without_reinstall() {
        let w = Tensor::zeros((1, 1), DType::F32, &Device::Cpu).unwrap();
        let mut lin = LoraLinear::from_linear(Linear::new(w, None), 1, 1, "t".into());
        let down = Var::from_tensor(&Tensor::from_vec(vec![1.0f32], (1, 1), &Device::Cpu).unwrap())
            .unwrap();
        let up = Var::from_tensor(&Tensor::from_vec(vec![1.0f32], (1, 1), &Device::Cpu).unwrap())
            .unwrap();
        lin.install_lora(down.as_tensor().clone(), up.as_tensor().clone(), 1.0);
        let x = Tensor::from_vec(vec![2.0f32], (1, 1), &Device::Cpu).unwrap();
        let y0 = lin
            .forward(&x)
            .unwrap()
            .i(0)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()[0];
        assert_eq!(y0, 2.0);
        up.set(&Tensor::from_vec(vec![3.0f32], (1, 1), &Device::Cpu).unwrap())
            .unwrap();
        let y1 = lin
            .forward(&x)
            .unwrap()
            .i(0)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()[0];
        assert_eq!(y1, 6.0);
    }

    #[test]
    fn kron2d_matches_reference() {
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &Device::Cpu).unwrap();
        let b = Tensor::from_vec(vec![0.0f32, 5.0, 6.0, 7.0], (2, 2), &Device::Cpu).unwrap();
        let k = kron2d(&a, &b).unwrap().to_vec2::<f32>().unwrap();
        assert_eq!(
            k,
            vec![
                vec![0.0, 5.0, 0.0, 10.0],
                vec![6.0, 7.0, 12.0, 14.0],
                vec![0.0, 15.0, 0.0, 20.0],
                vec![18.0, 21.0, 24.0, 28.0],
            ]
        );
    }

    /// A zero second Kronecker leg ⇒ zero delta ⇒ the LoKr adapter is the identity over the base at
    /// init (the property training relies on), same as LoRA's `B = 0`.
    #[test]
    fn lokr_zero_init_is_identity() {
        let mut lin = fixed_linear(&[1.0, 0.0, 0.0, 1.0], 2, 2);
        let w1 = Tensor::from_vec(vec![0.3f32, -0.1, 0.2, 0.4], (2, 2), &Device::Cpu).unwrap();
        let w2 = Tensor::zeros((1, 1), DType::F32, &Device::Cpu).unwrap();
        lin.install_lokr(w1, LokrW2::Full(w2), 1.0);
        let x = Tensor::from_vec(vec![3.0f32, 5.0], (1, 2), &Device::Cpu).unwrap();
        let y = lin
            .forward(&x)
            .unwrap()
            .i(0)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(y, vec![3.0, 5.0]);
    }

    /// LoKr residual on a 1×1 base: out=in=1 factor as (1,1)⊗(1,1); ΔW = scale·(w1·w2). base 0,
    /// w1=2, w2=3, scale=1, x=5 ⇒ y = 5·(2·3) = 30.
    #[test]
    fn lokr_residual_math() {
        let mut lin = fixed_linear(&[0.0], 1, 1);
        let w1 = Tensor::from_vec(vec![2.0f32], (1, 1), &Device::Cpu).unwrap();
        let w2 = Tensor::from_vec(vec![3.0f32], (1, 1), &Device::Cpu).unwrap();
        lin.install_lokr(w1, LokrW2::Full(w2), 1.0);
        let x = Tensor::from_vec(vec![5.0f32], (1, 1), &Device::Cpu).unwrap();
        let y = lin
            .forward(&x)
            .unwrap()
            .i(0)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()[0];
        assert_eq!(y, 30.0);
    }

    #[test]
    fn factorization_matches_lycoris() {
        assert_eq!(factorization(320, -1), (16, 20));
        assert_eq!(factorization(64, -1), (8, 8));
        // factor>0: the pair containing `factor`, smaller-first (MLX convention).
        assert_eq!(factorization(320, 4), (4, 80));
        assert_eq!(factorization(320, 80), (4, 80));
    }

    #[test]
    fn path_match_rules() {
        assert!(path_matches("a.b.attn1.to_q", "to_q"));
        assert!(path_matches("a.b.attn1.to_out.0", "to_out.0"));
        assert!(!path_matches("a.b.attn1.to_qx", "to_q"));
        assert!(path_matches("to_q", "to_q"));
    }
}
