//! TurboTime LoRA application for the Ideogram 4 **Turbo** path. The few-step ostris "continuous turbo"
//! LoRA is bundled in a turbo snapshot ([`crate::config::TURBO_LORA_FILE`]); it is applied at load —
//! the candle analogue of `mlx-gen-ideogram`'s `apply_ideogram_adapters`.
//!
//! **Forward-time additive, both tiers (sc-11104).** [`install_turbo_lora_additive`] attaches each LoRA
//! as an unmerged **forward-time residual** on the shared [`crate::quant::QLinear`]
//! (`y = base(x) + Σ scale·((x·A)·B)`) — never folding it into a base weight. So the base — dense bf16
//! **or** packed q4/q8 — is never mutated: a packed tier keeps its footprint (no dequant, no dense
//! reload), and *every* base stays a clean, disk-backed mmap the offload/eviction machinery can drop and
//! restore cheaply (a folded weight would be an in-memory-modified tensor, un-mmap-restorable — the
//! reason the fold path was retired). The additive residual equals the fold `(W+δ)·x` to f32 tolerance.
//!
//! Key forms handled: `{ns}{module}.lora_{down,up}.weight` / `.lora_{A,B}.weight` (and the `.weight`-less
//! variants), namespace `ns` ∈ {`diffusion_model.`, `transformer.`, `model.`, none} (sd-scripts /
//! ai-toolkit exports). The `module` path (e.g. `layers.0.attention.qkv`) matches the DiT's safetensors
//! keys directly. An optional `{module}.alpha` applies `alpha/rank` scaling: the resolved factors are
//! `a = downᵀ`, `b = upᵀ`, `scale = eff = user·(alpha/rank)`.

use std::collections::{BTreeMap, HashSet};
use std::path::Path;

use candle_gen::candle_core::safetensors::MmapedSafetensors;
use candle_gen::candle_core::{DType, Device, Error, Result};

use crate::transformer::Ideogram4Transformer;

/// Recognized `(down, up)` suffix pairs, most-specific first.
const PAIRS: &[(&str, &str)] = &[
    (".lora_down.weight", ".lora_up.weight"),
    (".lora_A.weight", ".lora_B.weight"),
    (".lora_down", ".lora_up"),
    (".lora_A", ".lora_B"),
];

/// Namespace prefixes stripped to recover the DiT module path.
const PREFIXES: &[&str] = &["diffusion_model.", "transformer.", "model."];

/// A resolved LoRA residual pending attachment to a projection: `a = downᵀ` `[in, rank]`,
/// `b = upᵀ` `[rank, out]`, `scale = eff` (`= user_scale · (alpha/rank)` — the same effective factor the
/// dense fold bakes into its delta). Read on the CPU; moved to the DiT device at push.
struct PendingLora {
    a: candle_gen::candle_core::Tensor,
    b: candle_gen::candle_core::Tensor,
    scale: f64,
}

/// Install the TurboTime LoRA at `lora_path` onto `dit` as **forward-time additive residuals** — the
/// sole apply route, both tiers (sc-11104). Resolves every `(down, up[, alpha])` pair into unmerged
/// factors (`a = downᵀ`, `b = upᵀ`, `scale = eff = user·(alpha/rank)`), then walks the DiT once
/// ([`Ideogram4Transformer::visit_adaptable_mut`]) pushing a residual onto each matched projection. The
/// base — dense or packed — is never mutated, so it stays a clean disk-backed mmap (evictable) and a
/// q4/q8 tier keeps its footprint; the residual equals a fold `(W+δ)·x` to f32 tolerance. Returns the
/// number of adapted projections. Errors if the file is missing, a factor is not a 2-D Linear adapter,
/// or **no** target matched the DiT (a wrong key format / prefix — never renders unadapted). A resolved
/// target absent from the DiT surface, or a factor whose shape mismatches its projection, is surfaced
/// (never a crashing forward), not merged.
pub fn install_turbo_lora_additive(
    dit: &mut Ideogram4Transformer,
    lora_path: &Path,
    scale: f32,
) -> Result<usize> {
    if !lora_path.exists() {
        return Err(Error::Msg(format!(
            "ideogram turbo: TurboTime LoRA not found at {} (a turbo snapshot must ship it alongside transformer/)",
            lora_path.display()
        )));
    }
    // SAFETY: read-only mmap of the adapter file.
    let lora = unsafe { MmapedSafetensors::new(lora_path)? };
    let names: Vec<String> = lora.tensors().into_iter().map(|(n, _)| n).collect();
    let present: HashSet<&str> = names.iter().map(String::as_str).collect();

    // Resolve every down/up pair into a pending residual keyed by the DiT module path (mirrors the
    // fold's per-key math, unmerged: `a = downᵀ`, `b = upᵀ`, scale = eff).
    let mut pending: BTreeMap<String, PendingLora> = BTreeMap::new();
    for name in &names {
        let Some((base_full, up_name)) = down_pair(name, &present) else {
            continue;
        };
        let module = strip_prefix(&base_full).to_string();
        let down = lora.load(name, &Device::Cpu)?.to_dtype(DType::F32)?; // [r, in]
        let up = lora.load(&up_name, &Device::Cpu)?.to_dtype(DType::F32)?; // [out, r]
        if down.rank() != 2 || up.rank() != 2 {
            return Err(Error::Msg(format!(
                "ideogram turbo: LoRA {name} is not a 2D Linear adapter (rank {}/{})",
                up.rank(),
                down.rank()
            )));
        }
        let rank = down.dim(0)?;
        let eff = scale as f64
            * alpha_for(&lora, &base_full)
                .map(|a| a as f64 / rank as f64)
                .unwrap_or(1.0);
        // a = downᵀ [in, rank]; b = upᵀ [rank, out]. Contiguous for the residual matmuls.
        let a = down.t()?.contiguous()?;
        let b = up.t()?.contiguous()?;
        pending.insert(module, PendingLora { a, b, scale: eff });
    }

    // Attach: walk the DiT once, pushing a resolved residual onto each matched projection. The factors
    // are read on the CPU but the base lives on the DiT device (CUDA on a packed tier), so move them at
    // push. A factor whose dims don't match the projection is surfaced as skipped, never a crashing
    // forward (the additive twin of the fold path's shape guard).
    let device = dit.device();
    let mut matched: HashSet<String> = HashSet::new();
    let mut applied = 0usize;
    let mut skipped = 0usize;
    dit.visit_adaptable_mut(&mut |path, lin| {
        if let Some(p) = pending.get(path) {
            matched.insert(path.to_string());
            let (out_f, in_f) = lin.base_shape();
            if p.a.dims()[0] != in_f || p.b.dims()[1] != out_f {
                skipped += 1;
                return Ok(());
            }
            lin.push_lora(p.a.to_device(&device)?, p.b.to_device(&device)?, p.scale);
            applied += 1;
        }
        Ok(())
    })?;

    if applied == 0 {
        return Err(Error::Msg(format!(
            "ideogram turbo: no TurboTime LoRA targets matched the DiT additively (checked {} adapter \
             tensors — wrong key format/prefix?)",
            names.len()
        )));
    }
    let absent = pending.len() - matched.len();
    if absent > 0 || skipped > 0 {
        eprintln!(
            "ideogram turbo: applied {applied} additive residual(s); {absent} resolved target(s) absent \
             from the DiT, {skipped} shape-mismatched"
        );
    }
    Ok(applied)
}

/// If `name` is a recognized "down"/"A" key whose paired "up"/"B" is also present, return
/// `(module_base_with_namespace, up_key)`.
fn down_pair(name: &str, present: &HashSet<&str>) -> Option<(String, String)> {
    for (down_suf, up_suf) in PAIRS {
        if let Some(base) = name.strip_suffix(down_suf) {
            let up = format!("{base}{up_suf}");
            if present.contains(up.as_str()) {
                return Some((base.to_string(), up));
            }
        }
    }
    None
}

/// Strip a known namespace prefix to recover the DiT module path.
fn strip_prefix(base: &str) -> &str {
    for p in PREFIXES {
        if let Some(rest) = base.strip_prefix(p) {
            return rest;
        }
    }
    base
}

/// Read an optional `{base}.alpha` scalar.
fn alpha_for(lora: &MmapedSafetensors, base_full: &str) -> Option<f32> {
    let t = lora
        .load(&format!("{base_full}.alpha"), &Device::Cpu)
        .ok()?;
    t.to_dtype(DType::F32)
        .ok()?
        .flatten_all()
        .ok()?
        .to_vec1::<f32>()
        .ok()?
        .first()
        .copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_prefix_recovers_module_path() {
        assert_eq!(
            strip_prefix("diffusion_model.layers.0.attention.qkv"),
            "layers.0.attention.qkv"
        );
        assert_eq!(strip_prefix("transformer.input_proj"), "input_proj");
        assert_eq!(
            strip_prefix("layers.3.feed_forward.w1"),
            "layers.3.feed_forward.w1"
        );
    }

    #[test]
    fn down_pair_matches_known_suffixes() {
        let names = [
            "m.lora_down.weight".to_string(),
            "m.lora_up.weight".to_string(),
        ];
        let present: HashSet<&str> = names.iter().map(String::as_str).collect();
        assert_eq!(
            down_pair("m.lora_down.weight", &present),
            Some(("m".to_string(), "m.lora_up.weight".to_string()))
        );
        // The up half alone is not a "down" key.
        assert_eq!(down_pair("m.lora_up.weight", &present), None);
    }

    /// **The additive residual equals a fold to f32 tolerance (sc-11104 guardrail).** This is the parity
    /// that lets the turbo LoRA ride additively on both tiers instead of folding: build a q4 packed base
    /// (`AdaptLinear::from_packed`) and push the LoRA as an unmerged residual with the exact resolution
    /// [`install_turbo_lora_additive`] uses (`a = downᵀ`, `b = upᵀ`, `scale = eff = user·alpha/rank`);
    /// fold the *same* delta into the dense affine grid the pack represents and forward it densely. The
    /// two forwards must agree — proving the residual reproduces the fold on a kept-quantized base.
    #[test]
    fn additive_residual_matches_dense_fold_on_packed_base() -> Result<()> {
        use candle_gen::candle_core::{Device, Tensor};
        use candle_gen::candle_nn::{Linear, Module};
        use candle_gen::quant::{AdaptLinear, QLinear as SharedQLinear, MLX_GROUP_SIZE};

        let dev = Device::Cpu;
        let g = MLX_GROUP_SIZE;
        let (out_dim, in_dim, rank) = (64usize, 128usize, 4usize);

        // A group-64 Q4 pack + the exact affine grid it represents (the dense base for the fold ref).
        let codes: Vec<u8> = (0..out_dim * in_dim)
            .map(|i| ((i * 5 + i / 11) % 16) as u8)
            .collect();
        let gpr = in_dim / g;
        let groups = out_dim * gpr;
        let scales: Vec<f32> = (0..groups)
            .map(|gi| 0.02 * ((gi % 5) as f32 + 1.0))
            .collect();
        let biases: Vec<f32> = (0..groups).map(|gi| -0.05 * (gi % 7) as f32).collect();
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
        let wq = Tensor::from_vec(words, (out_dim, in_dim / 8), &dev)?;
        let s = Tensor::from_vec(scales, (out_dim, gpr), &dev)?;
        let b = Tensor::from_vec(biases, (out_dim, gpr), &dev)?;
        let grid = Tensor::from_vec(grid, (out_dim, in_dim), &dev)?;

        // A LoRA: down [rank, in], up [out, rank], alpha 8 (⇒ ratio alpha/rank = 2), user scale 1.0.
        let down = Tensor::randn(0f32, 0.1f32, (rank, in_dim), &dev)?;
        let up = Tensor::randn(0f32, 0.1f32, (out_dim, rank), &dev)?;
        let (alpha, user_scale) = (8f64, 1.0f64);
        let eff = user_scale * (alpha / rank as f64);

        // The install-time resolved residual (a = downᵀ, b = upᵀ, scale = eff).
        let (a, b_fac) = (down.t()?.contiguous()?, up.t()?.contiguous()?);
        let x = Tensor::randn(0f32, 1f32, (3, in_dim), &dev)?;

        // (A) Residual identity — on a DENSE grid base, the additive forward equals the folded forward to
        // f32 tolerance (the only gap is op order: `eff·(x·downᵀ)·upᵀ` vs `x·(grid + eff·up@down)ᵀ`).
        // This isolates the residual math from the quant error, so the bar is tight.
        let delta = up.matmul(&down)?; // [out, in]
        let mut dense_additive =
            AdaptLinear::from_dense(Linear::new(grid.clone(), None), in_dim, out_dim);
        dense_additive.push_lora(a.clone(), b_fac.clone(), eff);
        let folded = Linear::new((grid.clone() + (delta * eff)?)?, None);
        let resid_diff = (dense_additive.forward(&x)? - folded.forward(&x)?)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(resid_diff < 1e-3, "residual vs fold max diff {resid_diff}");

        // (B) The packed base carries the same residual: the packed-additive forward tracks the
        // dense-grid-additive forward within the Q4→Q4_1 repack tolerance (the same bar the packed-load
        // parity test uses) — so `packed additive == dense fold` end to end.
        let packed = SharedQLinear::from_packed_gs(&wq, &s, &b, None, g, &dev)?;
        let mut packed_additive = AdaptLinear::from_packed(packed, in_dim, out_dim);
        packed_additive.push_lora(a, b_fac, eff);
        let (pa, da) = (packed_additive.forward(&x)?, dense_additive.forward(&x)?);
        let pa = pa.flatten_all()?.to_vec1::<f32>()?;
        let da = da.flatten_all()?.to_vec1::<f32>()?;
        let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (p, d) in pa.iter().zip(&da) {
            dot += (*p as f64) * (*d as f64);
            na += (*p as f64) * (*p as f64);
            nb += (*d as f64) * (*d as f64);
        }
        let cos = dot / (na.sqrt() * nb.sqrt() + 1e-12);
        assert!(
            cos > 0.99999,
            "packed-additive vs dense-additive cosine {cos:.6}"
        );
        Ok(())
    }
}
