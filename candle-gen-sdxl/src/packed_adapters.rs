//! Fold a trained SDXL LoRA/LoKr adapter into a **packed** (pre-quantized MLX tier) UNet (sc-9528,
//! the sc-9089j follow-up deferred from sc-9416).
//!
//! **Why this is separate from the dense path.** The dense SDXL adapter merge
//! ([`crate::adapters::merge_adapters`] + [`crate::pipeline`]'s `build_unet_with_adapters`) folds the
//! delta into the dense f32 `{path}.weight` tensors on CPU, then builds the stock UNet. A packed tier
//! (`SceneWorks/sdxl-base-mlx` q4/q8) ships its UNet Linears as `{path}.weight` (u32 codes) +
//! `{path}.scales` + `{path}.biases` triples that load straight into a `QLinear::Quantized` — there is
//! no dense `{path}.weight` to add a delta to. sc-9416 guarded this combination with a loud error;
//! this module replaces that guard with a real fold.
//!
//! **Option (a): dequant → fold → keep-dense (NOT re-quantize, NOT forward-time residual).**
//! SDXL deliberately *merges* LoRA/LoKr into dense weights rather than adding a forward-time residual,
//! because its samplers are chaos-sensitive and `(W+δ)·x` ≠ `W·x + δ·x` to ~1 ULP (see
//! [`crate::adapters`]). So a residual on the packed `QLinear` is off the table. For a packed tier we:
//!   1. **Dequantize** every packed Linear triple to a dense f32 `{path}.weight` (the exact grid the
//!      packed forward would itself dequantize to — [`candle_gen::quant`]'s reference dequant).
//!   2. **Fold** the delta by reusing [`crate::adapters::merge_adapters`] **verbatim** — no LoRA/LoKr
//!      math is reimplemented here; the same f32 reconstruction the dense path (and the trainer) use.
//!   3. **Keep the adapted layers dense, leave every unadapted layer packed.** The vendored UNet's
//!      Linear seam ([`candle_gen::quant::QLinear::linear_detect_gs`]) packed-detects per layer by the
//!      presence of a `{base}.scales` sibling, so serving a merged layer as a bare dense `{base}.weight`
//!      (its packed `.scales`/`.biases`/u32 `.weight` dropped) routes it through the dense arm while
//!      every untouched layer keeps its packed triple and its packed footprint.
//!
//! **Why keep-dense and not re-quantize.** The MLX affine format has no lossless dense→packed
//! round-trip (its edge-snapping quantizer is not vendored here, and Q4 has only ~4 bits of headroom),
//! so re-quantizing a merged weight would stack a *second* quantization error on top of the fold and
//! make the packed+adapter result strictly worse than the dense+adapter result. Keeping the handful of
//! adapted layers dense makes the fold **byte-identical to the dense adapter path** for those layers,
//! so the only deviation from a dense-base render is the *base* dequant error on the adapted layers —
//! exactly the quant bar sc-9416 already accepts. SDXL adapters touch only attention/proj/conv (a small
//! minority of the UNet's ~2 500 tensors), so the packed memory benefit is preserved for the
//! overwhelming majority of layers that stay packed.

use std::collections::{HashMap, HashSet};

use candle_gen::candle_core::{DType, Tensor};
use candle_gen::gen_core::AdapterSpec;
use candle_gen::quant::{dequant_mlx_q4_reference_gs, dequant_mlx_q8_gs, mlx_packed_bits_gs};
use candle_gen::{CandleError, Result};

/// The result of splitting a raw packed-UNet map: `(dense base map for merging, the set-aside packed
/// triples keyed by base, the set of packed base keys)`.
type SplitPacked = (
    HashMap<String, Tensor>,
    HashMap<String, [Tensor; 3]>,
    HashSet<String>,
);

/// Split a raw packed-UNet tensor map into (dense base map for merging, the packed sibling parts we set
/// aside). Every packed Linear `{base}` (identified by a `{base}.scales` sibling) is **dequantized** to
/// a dense f32 `{base}.weight` in `dense`, and its three packed parts (`{base}.weight` u32 /
/// `{base}.scales` / `{base}.biases`) are moved into `packed_parts` keyed by `base`. Non-packed tensors
/// (dense conv/norm weights, biases) pass straight into `dense`. Returns the set of packed base keys.
///
/// `group_size` is the tier's parsed MLX group size (never a hardcoded 64 — the caller threads the
/// `unet/config.json` value); the Q4/Q8 bit-width is inferred per layer from the packed shapes.
fn dequantize_packed_map(raw: HashMap<String, Tensor>, group_size: usize) -> Result<SplitPacked> {
    // Discover packed bases first: any key ending in `.scales`. The matching `.weight`/`.biases`
    // siblings are then guaranteed present (an MLX tier always ships the full triple).
    let bases: Vec<String> = raw
        .keys()
        .filter_map(|k| k.strip_suffix(".scales").map(str::to_string))
        .collect();

    let mut dense: HashMap<String, Tensor> = HashMap::new();
    let mut packed_parts: HashMap<String, [Tensor; 3]> = HashMap::new();
    let mut packed_bases: HashSet<String> = HashSet::new();

    let mut consumed: HashSet<String> = HashSet::new();
    for base in &bases {
        let wq_key = format!("{base}.weight");
        let scales_key = format!("{base}.scales");
        let biases_key = format!("{base}.biases");
        let (Some(wq), Some(scales), Some(biases)) =
            (raw.get(&wq_key), raw.get(&scales_key), raw.get(&biases_key))
        else {
            // A `.scales` with no matching `.weight`/`.biases` is not a well-formed MLX triple; leave
            // the stray tensor(s) to pass through as dense below rather than guess.
            continue;
        };
        // Upcast the packed parts to the dtypes the dequant helpers expect (u32 codes stay u32; the
        // bf16/f16 scales & biases upcast to f32 exactly), then dequantize to the exact affine grid.
        let wq = wq.to_dtype(DType::U32)?;
        let scales = scales.to_dtype(DType::F32)?;
        let biases = biases.to_dtype(DType::F32)?;
        let wq_cols = wq.dims2()?.1;
        let s_cols = scales.dims2()?.1;
        let grid = match mlx_packed_bits_gs(wq_cols, s_cols, group_size) {
            4 => dequant_mlx_q4_reference_gs(&wq, &scales, &biases, group_size)?,
            8 => dequant_mlx_q8_gs(&wq, &scales, &biases, group_size)?,
            b => {
                return Err(CandleError::Msg(format!(
                    "sdxl: packed UNet Linear {base} has unsupported bit-width {b} \
                     (wq cols {wq_cols}, scales cols {s_cols}, group {group_size})"
                )))
            }
        };
        dense.insert(wq_key.clone(), grid);
        packed_parts.insert(base.clone(), [wq, scales, biases]);
        packed_bases.insert(base.clone());
        consumed.insert(wq_key);
        consumed.insert(scales_key);
        consumed.insert(biases_key);
    }

    // Everything that was not part of a packed triple passes through untouched (dense conv/norm
    // weights, dense `.bias` vectors, any stray key).
    for (k, v) in raw {
        if !consumed.contains(&k) {
            dense.insert(k, v);
        }
    }
    Ok((dense, packed_parts, packed_bases))
}

/// Fold `specs` into a raw **packed** UNet tensor map and return a map ready to feed the vendored
/// packed-detecting UNet via `VarBuilder::from_tensors` (sc-9528). Reuses
/// [`crate::adapters::merge_adapters`] for all delta math (dequant → fold → keep-dense; see the module
/// docs). `group_size` is the tier's parsed MLX group size.
///
/// The returned map serves:
/// - each **adapted** packed Linear as a bare dense f32 `{base}.weight` (its `.scales`/`.biases` and
///   u32 `.weight` dropped) — so the vendored UNet's `linear_detect_gs` takes its dense arm and the
///   folded weight is used exactly, byte-for-byte the dense adapter path,
/// - each **unadapted** packed Linear as its original `{base}.weight` (u32) + `.scales` + `.biases`
///   triple — so it stays packed and keeps its packed footprint,
/// - every dense conv/norm weight and bias unchanged (some of which the conv-LoRA surface may also have
///   folded a delta into).
pub(crate) fn fold_adapters_into_packed_map(
    raw: HashMap<String, Tensor>,
    specs: &[AdapterSpec],
    group_size: usize,
) -> Result<HashMap<String, Tensor>> {
    let (mut dense, packed_parts, packed_bases) = dequantize_packed_map(raw, group_size)?;

    // Snapshot the pre-merge dequantized grid of every packed Linear so we can tell which layers the
    // adapter actually touched: `merge_into` replaces `{base}.weight` with `W += δ` only for a folded
    // layer, so a bit-for-bit-unchanged grid ⇒ unadapted ⇒ restore its packed triple.
    let pre: HashMap<String, Tensor> = packed_bases
        .iter()
        .filter_map(|b| {
            dense
                .get(&format!("{b}.weight"))
                .map(|t| (b.clone(), t.clone()))
        })
        .collect();

    // Reuse the dense adapter merge verbatim — no LoRA/LoKr/conv delta math is reimplemented here.
    crate::adapters::merge_adapters(&mut dense, specs)?;

    // Repartition: restore the packed triple for every packed Linear the fold left untouched; adapted
    // packed Linears keep their now-dense merged `{base}.weight` (their packed siblings were never
    // re-inserted, so `linear_detect_gs` finds no `.scales` and takes the dense arm).
    for base in &packed_bases {
        let wq_key = format!("{base}.weight");
        let adapted = match (pre.get(base), dense.get(&wq_key)) {
            (Some(before), Some(after)) => tensors_differ(before, after)?,
            // A packed base whose dense `.weight` vanished should not happen (merge only replaces),
            // but treat a missing post-merge weight as "adapted" so we never restore a stale triple.
            _ => true,
        };
        if !adapted {
            let [wq, scales, biases] = &packed_parts[base];
            dense.insert(wq_key, wq.clone());
            dense.insert(format!("{base}.scales"), scales.clone());
            dense.insert(format!("{base}.biases"), biases.clone());
        }
    }
    Ok(dense)
}

/// Whether two same-shaped tensors differ anywhere (exact, in f32). Used to detect which packed Linears
/// the adapter fold touched: an untouched dequant grid is bit-identical to its pre-merge snapshot.
fn tensors_differ(a: &Tensor, b: &Tensor) -> Result<bool> {
    if a.dims() != b.dims() {
        return Ok(true);
    }
    let diff = (a.to_dtype(DType::F32)? - b.to_dtype(DType::F32)?)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    Ok(diff != 0.0)
}

/// Assert the tier's parsed `group_size` is the one the vendored UNet's Linear seam threads (64).
///
/// The vendored `UNet2DConditionModel::new` builds its leaf Linears at the default MLX group 64 (the
/// leaf `*_gs` constructors exist, but the top-level `new` → blocks → leaves chain does not yet thread a
/// non-64 group — the same nested-constructor infeasibility lens/sd3 hit in sc-9474). So the packed
/// adapter fold (like [`crate::pipeline`]'s `detect_packed_unet`) must refuse a non-64 tier loudly
/// rather than dequantize/repartition at 64 while the UNet reads at 64 for a tier packed at another
/// grid. The SDXL MLX tiers all pack at 64, so this never fires on a real tier.
pub(crate) fn assert_group_size_supported(group_size: usize) -> Result<()> {
    if group_size != candle_gen::quant::MLX_GROUP_SIZE {
        return Err(CandleError::Msg(format!(
            "sdxl: packed adapter fold at group_size {group_size} unsupported (the vendored UNet \
             threads only {}); a non-64 tier needs the group threaded new_gs → blocks → \
             ResnetBlock2D/SpatialTransformer/TimestepEmbedding *_gs (sc-9528)",
            candle_gen::quant::MLX_GROUP_SIZE
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Device;
    use candle_gen::gen_core::AdapterKind;
    use candle_gen::quant::repack::{f16_exact, MLX_GROUP_SIZE};
    use candle_gen::train::lora::{reconstruct_lora_delta, save_lora_peft, SDXL_PEFT_PREFIX};

    /// Pack per-element 4-bit codes into MLX u32 words (LSB-first nibbles) — mirrors the repack tests'
    /// `pack_mlx_q4`, so we can synthesize a real packed triple with a known dequant grid.
    fn pack_mlx_q4(codes: &[u8]) -> Vec<u32> {
        codes
            .chunks_exact(8)
            .map(|c| {
                c.iter()
                    .enumerate()
                    .fold(0u32, |acc, (i, &q)| acc | ((q as u32 & 0xF) << (4 * i)))
            })
            .collect()
    }

    /// Build a synthetic group-64 Q4 packed triple `[out, in]` with f16-exact scales/biases, returning
    /// the three packed tensors AND the exact dense f32 grid they dequantize to.
    fn synth_q4(out_dim: usize, in_dim: usize) -> ([Tensor; 3], Tensor) {
        let dev = Device::Cpu;
        let codes: Vec<u8> = (0..out_dim * in_dim)
            .map(|i| ((i * 7 + i / 5) % 16) as u8)
            .collect();
        let groups = out_dim * in_dim / MLX_GROUP_SIZE;
        let scales: Vec<f32> = (0..groups).map(|g| 0.0625 * (g as f32 + 1.0)).collect();
        let biases: Vec<f32> = (0..groups).map(|g| -0.5 - 0.25 * g as f32).collect();
        assert!(scales.iter().chain(biases.iter()).all(|&x| f16_exact(x)));
        let wq = Tensor::from_vec(pack_mlx_q4(&codes), (out_dim, in_dim / 8), &dev).unwrap();
        let s = Tensor::from_vec(scales.clone(), (out_dim, in_dim / MLX_GROUP_SIZE), &dev).unwrap();
        let b = Tensor::from_vec(biases.clone(), (out_dim, in_dim / MLX_GROUP_SIZE), &dev).unwrap();
        let grid = dequant_mlx_q4_reference_gs(&wq, &s, &b, MLX_GROUP_SIZE).unwrap();
        ([wq, s, b], grid)
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        (a.to_dtype(DType::F32).unwrap() - b.to_dtype(DType::F32).unwrap())
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    /// A tiny packed UNet map: one adapted attention Linear (`to_q`, packed) + one unadapted packed
    /// Linear (`to_k`) + one dense conv weight. `in`/`out` are multiples of 64 so the group-64 pack is
    /// well-formed.
    fn packed_map() -> (HashMap<String, Tensor>, Tensor, Tensor) {
        let dev = Device::Cpu;
        let (to_q, to_q_grid) = synth_q4(64, 64);
        let (to_k, to_k_grid) = synth_q4(64, 64);
        let mut m = HashMap::new();
        let qp = "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q";
        let kp = "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_k";
        let [wq, s, b] = to_q;
        m.insert(format!("{qp}.weight"), wq);
        m.insert(format!("{qp}.scales"), s);
        m.insert(format!("{qp}.biases"), b);
        let [wq, s, b] = to_k;
        m.insert(format!("{kp}.weight"), wq);
        m.insert(format!("{kp}.scales"), s);
        m.insert(format!("{kp}.biases"), b);
        // A dense conv weight (not packed) — must pass through untouched.
        m.insert(
            "conv_in.weight".to_string(),
            Tensor::zeros((4, 4, 3, 3), DType::F32, &dev).unwrap(),
        );
        (m, to_q_grid, to_k_grid)
    }

    /// Write a real trainer PEFT LoRA `.safetensors` targeting `path` with a forced-nonzero delta, and
    /// return (file, expected ΔW) at scale 1.0. Reuses the actual trainer save path so the fold consumes
    /// the on-disk format, not hand-built tensors.
    fn write_lora(
        path: &str,
        in_dim: usize,
        out_dim: usize,
        tag: &str,
    ) -> (std::path::PathBuf, Tensor) {
        use candle_gen::candle_nn::Linear;
        use candle_gen::train::lora::{build_lora_targets, LoraHost, LoraLinear};
        struct Host(LoraLinear);
        impl LoraHost for Host {
            fn visit_lora_mut(
                &mut self,
                f: &mut dyn FnMut(&mut LoraLinear) -> Result<()>,
            ) -> Result<()> {
                f(&mut self.0)
            }
        }
        let dev = Device::Cpu;
        let base_w = Tensor::zeros((out_dim, in_dim), DType::F32, &dev).unwrap();
        // The trainer targets the leaf name (`to_q`); the host carries the full dotted path.
        let leaf = path.rsplit('.').next().unwrap().to_string();
        let mut host = Host(LoraLinear::from_linear(
            Linear::new(base_w, None),
            in_dim,
            out_dim,
            path.into(),
        ));
        let set = build_lora_targets(&mut host, &[leaf], 2, 4.0, 7, &dev).unwrap();
        let up = Tensor::randn(0f32, 1f32, (out_dim, 2), &dev).unwrap();
        set.vars[1].set(&up).unwrap();
        let file =
            std::env::temp_dir().join(format!("sc9528_{tag}_{}.safetensors", std::process::id()));
        save_lora_peft(&set, SDXL_PEFT_PREFIX, &HashMap::new(), &file).unwrap();
        // rank 2, alpha 4 ⇒ effective 2.0; base zero ⇒ ΔW = 2.0·B·A.
        let delta = reconstruct_lora_delta(
            set.vars[0].as_tensor(),
            set.vars[1].as_tensor(),
            4.0,
            2.0,
            1.0,
        )
        .unwrap();
        (file, delta)
    }

    /// **Core parity (5a): packed+adapter == dense+adapter within the quant bar.** Fold the SAME trained
    /// LoRA onto (i) a packed base and (ii) the equivalent DENSE base (the packed grid); assert the two
    /// folded `to_q` weights match to the packed tier's tolerance — proving dequant→fold→keep-dense is
    /// numerically faithful (the only deviation is the base dequant, which is exact here since the grid
    /// IS the dense base).
    #[test]
    fn packed_adapter_matches_dense_adapter_parity() {
        let qp = "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q";
        let (packed, to_q_grid, _) = packed_map();
        let (file, expected_delta) = write_lora(qp, 64, 64, "parity");

        // (i) packed fold.
        let folded = fold_adapters_into_packed_map(
            packed,
            &[AdapterSpec::new(file.clone(), 1.0, AdapterKind::Lora)],
            MLX_GROUP_SIZE,
        )
        .unwrap();

        // (ii) dense fold onto the same grid via the dense entry point directly.
        let mut dense_base: HashMap<String, Tensor> = HashMap::new();
        dense_base.insert(format!("{qp}.weight"), to_q_grid.clone());
        crate::adapters::merge_adapters(
            &mut dense_base,
            &[AdapterSpec::new(file.clone(), 1.0, AdapterKind::Lora)],
        )
        .unwrap();
        std::fs::remove_file(&file).ok();

        let packed_wq = folded.get(&format!("{qp}.weight")).unwrap();
        let dense_wq = dense_base.get(&format!("{qp}.weight")).unwrap();
        // The adapted layer is served dense (no `.scales` sibling) — the packing benefit is dropped
        // only for adapted layers.
        assert!(
            !folded.contains_key(&format!("{qp}.scales")),
            "an adapted packed Linear must be served dense"
        );
        let diff = max_abs_diff(packed_wq, dense_wq);
        assert!(
            diff < 1e-4,
            "packed+adapter diverged from dense+adapter by {diff}"
        );
        // And it equals base grid + ΔW exactly (base grid is the exact dequant here).
        let expected = (&to_q_grid + &expected_delta).unwrap();
        assert!(max_abs_diff(packed_wq, &expected) < 1e-4);
    }

    /// **Guard-gone (5b): a packed map WITH an adapter folds and the weights actually change.** The
    /// adapted `to_q` differs from its base grid (adapted, dense), while the unadapted `to_k` is
    /// restored to its packed triple (still `.scales`-backed) — proving the fold is real and the
    /// unadapted layers keep their packing.
    #[test]
    fn packed_fold_adapts_and_keeps_unadapted_packed() {
        let qp = "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q";
        let kp = "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_k";
        let (packed, to_q_grid, _to_k_grid) = packed_map();
        let (file, _) = write_lora(qp, 64, 64, "adapts");
        let folded = fold_adapters_into_packed_map(
            packed,
            &[AdapterSpec::new(file.clone(), 1.0, AdapterKind::Lora)],
            MLX_GROUP_SIZE,
        )
        .unwrap();
        std::fs::remove_file(&file).ok();

        // to_q: adapted ⇒ dense (no scales) and changed vs the base grid.
        assert!(!folded.contains_key(&format!("{qp}.scales")));
        let adapted = folded.get(&format!("{qp}.weight")).unwrap();
        assert!(
            max_abs_diff(adapted, &to_q_grid) > 1e-3,
            "the adapter must actually change to_q"
        );
        // to_k: unadapted ⇒ its packed triple restored (still scales-backed, u32 weight).
        assert!(
            folded.contains_key(&format!("{kp}.scales"))
                && folded.contains_key(&format!("{kp}.biases")),
            "an unadapted packed Linear must keep its packed triple"
        );
        assert_eq!(
            folded.get(&format!("{kp}.weight")).unwrap().dtype(),
            DType::U32,
            "the unadapted packed weight stays u32-packed"
        );
        // conv passes through.
        assert!(folded.contains_key("conv_in.weight"));
    }

    /// **LoKr coverage on a packed tier.** A LoKr `.safetensors` (per-module `lokr_w1`/`lokr_w2` keys)
    /// folds through the SAME dequant→fold→keep-dense path — `merge_adapters` routes it through the LoKr
    /// reconstruction (here the untagged-third-party route, since `safetensors::save` writes no
    /// `networkType` header). The adapted Linear is served dense and its weight changes, proving both
    /// LoRA and LoKr are covered on the packed path.
    #[test]
    fn packed_fold_covers_lokr() {
        let qp = "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q";
        // A real LyCORIS LoKr file names modules with the kohya-flattened stem (`lora_unet_<flat>`),
        // resolved back to the dotted path via the base-key table — that is the third-party LoKr route.
        let flat = format!("lora_unet_{}", qp.replace('.', "_"));
        let (packed, to_q_grid, _) = packed_map();
        // [64,64] LoKr delta as an 8×8 ⊗ 8×8 kron with distinct nonzero factors.
        let dev = Device::Cpu;
        let w1 = Tensor::randn(0f32, 1f32, (8, 8), &dev).unwrap();
        let w2 = Tensor::randn(0f32, 1f32, (8, 8), &dev).unwrap();
        let file =
            std::env::temp_dir().join(format!("sc9528_lokr_{}.safetensors", std::process::id()));
        let mut save: HashMap<String, Tensor> = HashMap::new();
        save.insert(format!("{flat}.lokr_w1"), w1);
        save.insert(format!("{flat}.lokr_w2"), w2);
        candle_gen::candle_core::safetensors::save(&save, &file).unwrap();
        let folded = fold_adapters_into_packed_map(
            packed,
            &[AdapterSpec::new(file.clone(), 1.0, AdapterKind::Lokr)],
            MLX_GROUP_SIZE,
        )
        .unwrap();
        std::fs::remove_file(&file).ok();
        assert!(
            !folded.contains_key(&format!("{qp}.scales")),
            "adapted ⇒ dense"
        );
        let adapted = folded.get(&format!("{qp}.weight")).unwrap();
        assert!(
            max_abs_diff(adapted, &to_q_grid) > 1e-4,
            "the LoKr adapter must change to_q"
        );
    }

    /// **group_size (5c): the parsed group is used, not silently 64.** The support assertion rejects a
    /// non-64 group loudly, and the fold uses the group it is handed to dequantize (a wrong group would
    /// mis-shape the dequant and error, not silently mis-fold).
    #[test]
    fn group_size_is_threaded_not_assumed_64() {
        assert!(assert_group_size_supported(MLX_GROUP_SIZE).is_ok());
        assert!(
            assert_group_size_supported(32).is_err(),
            "a non-64 group must be rejected loudly"
        );
        // Passing the wrong group to the fold makes the packed dequant shape-check fail — proving the
        // group is actually consumed (not ignored in favor of a hardcoded 64).
        let qp = "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q";
        let (packed, ..) = packed_map();
        let (file, _) = write_lora(qp, 64, 64, "gs");
        let err = fold_adapters_into_packed_map(
            packed,
            &[AdapterSpec::new(file.clone(), 1.0, AdapterKind::Lora)],
            128, // wrong group for a group-64 pack
        );
        std::fs::remove_file(&file).ok();
        assert!(err.is_err(), "the fold must consume the group it is handed");
    }
}
