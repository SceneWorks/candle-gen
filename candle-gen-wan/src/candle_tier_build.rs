//! Offline builder for the **candle-native** packed Wan quant tiers (sc-10026, epic 9083) — hosted at
//! `SceneWorks/wan2.2-{ti2v-5b,t2v-a14b,i2v-a14b}-candle`.
//!
//! Option B of the sc-10026 design: rather than reverse-remap the native-Wan-keyed `SceneWorks/*-mlx`
//! tiers, candle hosts its OWN packed tiers on the **diffusers** keys it already reads. This builder
//! loads a `Wan-AI/Wan2.2-*-Diffusers` snapshot and rewrites each transformer component so every DiT
//! Linear weight is MLX-affine-packed ([`candle_gen::quant::pack_mlx_affine`], group 64) as the
//! `{key}.weight` (u32) + `{key}.scales` + `{key}.biases` triple **on the same diffusers key** — the
//! exact shape the sc-10025 packed-detect seam ([`crate::quant::QLinear::linear_detect`]) consumes. The
//! result keeps the diffusers component layout (`transformer/` [+ `transformer_2/`] + `text_encoder/` +
//! `vae/` + `tokenizer/`), so the worker load path resolves and loads it with **zero change** — the seam
//! fires on the `.scales` siblings.
//!
//! **What packs, what stays dense.** Only rank-2 `.weight` tensors in the transformer component(s) pack:
//! those are exactly the seam's Linear surface (attn `to_q/k/v/to_out.0`, ffn `net.0.proj/net.2`, the
//! condition-embedder Linears, `proj_out`). Everything else stays dense — the 1-D norms, the 3-D
//! `scale_shift_table`, the `patch_embedding` conv (rank ≥ 4), every `.bias`, and the whole T5 encoder
//! plus the z16 VAE (the MLX build keeps those dense too). A component's `quantize_config.json` records
//! `{ "quantization": { "group_size": 64 }, "bits": N }` so a group-size-aware loader can thread it
//! (the seam currently detects at the group-64 default).
//!
//! The [`build_candle_wan_tier`] entry point is an `#[ignore]`d test — it needs the multi-GB diffusers
//! snapshot on disk. Run it on-device per model/tier, then `hf upload` the output to the `*-candle` repo.
//! The pure [`pack_transformer_component`] core is unit-tested in CI (no weights needed).

use std::collections::HashMap;
use std::path::Path;

use candle_gen::candle_core::{DType, Device, Result, Tensor};
use candle_gen::quant::pack_mlx_affine;

/// The quant group size the candle Wan tiers pack at (matches the MLX tiers + the seam's default).
pub const TIER_GROUP_SIZE: usize = 64;

/// Rewrite a transformer component's CPU tensor `map`, MLX-affine-packing every **rank-2 `.weight`**
/// (the seam's Linear set) at `bits` (4 or 8) / [`TIER_GROUP_SIZE`] into the `{base}.weight` (u32) +
/// `{base}.scales` + `{base}.biases` triple, and leaving every other tensor untouched (dense norms /
/// conv / `scale_shift_table` / `.bias`). Returns the rewritten map + the number of Linears packed.
///
/// Rank-2 is the exact seam predicate here: in the Wan diffusers transformer the only rank-2 weights are
/// the Linears the seam routes through `linear_detect`; the `patch_embedding` conv is rank ≥ 4, the norms
/// are rank 1, `scale_shift_table` is rank 3. Packing a weight the seam would load densely would make it
/// load u32 codes as bf16 garbage — so this predicate must stay aligned with the seam (both key off "the
/// Linears"). Pure + deterministic (no I/O) so CI unit-tests it without weights.
pub fn pack_transformer_component(
    map: HashMap<String, Tensor>,
    bits: usize,
) -> Result<(HashMap<String, Tensor>, usize)> {
    let mut out: HashMap<String, Tensor> = HashMap::with_capacity(map.len());
    let mut packed = 0usize;
    for (key, value) in map {
        let is_linear_weight = key.ends_with(".weight") && value.rank() == 2;
        if is_linear_weight {
            let base = key.strip_suffix(".weight").unwrap();
            let (wq, scales, biases) =
                pack_mlx_affine(&value.to_dtype(DType::F32)?, bits, TIER_GROUP_SIZE)?;
            out.insert(format!("{base}.weight"), wq);
            out.insert(format!("{base}.scales"), scales);
            out.insert(format!("{base}.biases"), biases);
            packed += 1;
        } else {
            // Dense passthrough: norms (rank 1), scale_shift_table (rank 3), patch_embedding conv
            // (rank ≥ 4), every `.bias`, and any non-weight tensor. Kept at its on-disk dtype.
            out.insert(key, value);
        }
    }
    Ok((out, packed))
}

/// Load a transformer component dir's `.safetensors` shards, [`pack_transformer_component`] them at
/// `bits`, and write the packed component to `dst` as a single `model.safetensors` + a
/// `quantize_config.json` (group [`TIER_GROUP_SIZE`]). Any non-`.safetensors` sidecar (e.g. the
/// diffusers `config.json`) is copied verbatim.
fn pack_component_dir(src: &Path, dst: &Path, bits: usize) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    // `sorted_safetensors` returns candle-gen's error type; bridge it into `candle_core::Result`.
    let files = candle_gen::sorted_safetensors(src, "wan-candle-tier")
        .map_err(|e| candle_gen::candle_core::Error::Msg(e.to_string()))?;
    let mut map: HashMap<String, Tensor> = HashMap::new();
    for f in &files {
        map.extend(candle_gen::candle_core::safetensors::load(f, &Device::Cpu)?);
    }
    let (packed, n) = pack_transformer_component(map, bits)?;
    candle_gen::candle_core::safetensors::save(&packed, dst.join("model.safetensors"))?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        // Copy sidecars (e.g. the diffusers `config.json`), but NOT any `.safetensors` shard or its
        // `.safetensors.index.json` — we collapsed the shards into one `model.safetensors`, so a copied
        // shard index would dangle and mislead a sharded loader.
        if entry.path().is_file() && !name.contains(".safetensors") {
            std::fs::copy(entry.path(), dst.join(&name))?;
        }
    }
    std::fs::write(
        dst.join("quantize_config.json"),
        format!("{{\n  \"bits\": {bits},\n  \"quantization\": {{ \"group_size\": {TIER_GROUP_SIZE} }}\n}}\n"),
    )?;
    eprintln!(
        "[[CANDLE-TIER]] packed {n} Linears: {} -> {}",
        src.display(),
        dst.display()
    );
    Ok(())
}

/// Copy a dense component dir (e.g. `text_encoder/`, `vae/`, `tokenizer/`) verbatim, recursively.
fn copy_dir_recursive(src: &Path, dst: &Path) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if from.is_dir() {
            copy_dir_recursive(&from, &to)?;
        } else {
            std::fs::copy(&from, &to)?;
        }
    }
    Ok(())
}

/// Build one candle packed tier from a `Wan-AI/Wan2.2-*-Diffusers` snapshot at `diffusers_dir` into
/// `out_dir` at `bits` (4 or 8). Packs each transformer component (`transformer/`, plus `transformer_2/`
/// for the A14B dual-expert MoE) via [`pack_component_dir`]; copies every other component/file
/// (`text_encoder/`, `vae/`, `tokenizer/`, `scheduler/`, `model_index.json`, …) verbatim so the tier is
/// a complete, self-contained diffusers-layout snapshot the sc-10025 seam loads unchanged. Host the
/// result at `SceneWorks/wan2.2-*-candle`.
pub fn build_candle_wan_tier(diffusers_dir: &Path, out_dir: &Path, bits: usize) -> Result<()> {
    std::fs::create_dir_all(out_dir)?;
    for entry in std::fs::read_dir(diffusers_dir)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        let src = entry.path();
        let dst = out_dir.join(&name);
        if src.is_dir() && (name == "transformer" || name == "transformer_2") {
            pack_component_dir(&src, &dst, bits)?;
        } else if src.is_dir() {
            copy_dir_recursive(&src, &dst)?;
        } else {
            std::fs::copy(&src, &dst)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::safetensors::MmapedSafetensors;
    use candle_gen::candle_core::Device;
    use candle_gen::candle_nn::{Linear, Module, VarBuilder};

    fn dense(out_dim: usize, in_dim: usize, seed: f32) -> Tensor {
        let data: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| ((i as f32 + seed) * 0.013).sin() * 1.3)
            .collect();
        Tensor::from_vec(data, (out_dim, in_dim), &Device::Cpu).unwrap()
    }

    /// `pack_transformer_component` packs exactly the rank-2 `.weight`s (the Linear surface) and leaves
    /// the norm (rank 1), `scale_shift_table` (rank 3), `patch_embedding` conv (rank 5) and every `.bias`
    /// dense — the on-disk shape the seam loads.
    #[test]
    fn packs_only_rank2_weights() -> Result<()> {
        let dev = Device::Cpu;
        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert("blocks.0.attn1.to_q.weight".into(), dense(64, 128, 1.0));
        map.insert(
            "blocks.0.attn1.to_q.bias".into(),
            Tensor::zeros(64, DType::F32, &dev)?,
        );
        map.insert("blocks.0.ffn.net.0.proj.weight".into(), dense(256, 64, 2.0));
        map.insert("proj_out.weight".into(), dense(64, 64, 3.0));
        // Dense-only leaves.
        map.insert(
            "blocks.0.norm_q.weight".into(),
            Tensor::ones(64, DType::F32, &dev)?,
        ); // rank 1
        map.insert(
            "scale_shift_table".into(),
            Tensor::zeros((1, 6, 64), DType::F32, &dev)?, // rank 3
        );
        map.insert(
            "patch_embedding.weight".into(),
            Tensor::zeros((64, 48, 1, 2, 2), DType::F32, &dev)?, // rank 5 conv
        );

        let (out, packed) = pack_transformer_component(map, 4)?;
        assert_eq!(packed, 3, "the three rank-2 weights pack");
        // Packed Linears gained the triple; the u32 codes replace the dense weight.
        for base in ["blocks.0.attn1.to_q", "blocks.0.ffn.net.0.proj", "proj_out"] {
            assert_eq!(out[&format!("{base}.weight")].dtype(), DType::U32);
            assert!(out.contains_key(&format!("{base}.scales")));
            assert!(out.contains_key(&format!("{base}.biases")));
        }
        // Dense leaves untouched (dtype + rank).
        assert_eq!(out["blocks.0.attn1.to_q.bias"].rank(), 1);
        assert_eq!(out["blocks.0.norm_q.weight"].dtype(), DType::F32);
        assert_eq!(out["scale_shift_table"].rank(), 3);
        assert_eq!(out["patch_embedding.weight"].rank(), 5);
        assert!(!out.contains_key("scale_shift_table.scales"));
        Ok(())
    }

    /// End-to-end: a packed component written by `pack_transformer_component` is loaded back through the
    /// **seam** (`QLinear::linear_detect`) — it fires the packed path and the dequantized forward matches
    /// the original dense Linear (Q4 cosine parity). Proves the producer ↔ consumer round-trip that lets
    /// candle host its own tiers.
    #[test]
    fn packed_component_loads_through_the_seam() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (128, 256);
        let w = dense(out_dim, in_dim, 7.0);
        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert("blocks.0.attn1.to_q.weight".into(), w.clone());

        let (packed_map, packed) = pack_transformer_component(map, 4)?;
        assert_eq!(packed, 1);

        let tmp = std::env::temp_dir().join(format!(
            "sc10026_component_{}.safetensors",
            std::process::id()
        ));
        candle_gen::candle_core::safetensors::save(&packed_map, &tmp)?;
        // SAFETY: freshly written, single-reader for the test.
        let st = unsafe { MmapedSafetensors::new(&tmp)? };
        let vb = VarBuilder::from_backend(Box::new(st), DType::F32, dev.clone());

        // The crate seam loads it packed.
        let loaded = crate::quant::QLinear::linear_detect(
            in_dim,
            out_dim,
            &vb.pp("blocks.0.attn1"),
            "to_q",
            false,
        )?;
        assert!(
            loaded.is_packed(),
            "written tier must load through the packed seam path"
        );

        // Dequantized forward ≈ the original dense Linear (Q4 cosine parity). Q4 is genuinely lossy, so
        // this is a cosine tolerance (> 0.997), not bit-equality; the input is **deterministic** (not an
        // unseeded `randn`) so the parity is portable — an unseeded input made the cosine flap around the
        // Q4 loss floor and fail on CI's FP under a too-tight threshold ([[fp-byte-equality tests are
        // non-portable]]).
        let dense_lin = Linear::new(w, None);
        let x = dense(4, in_dim, 11.0);
        let a = loaded.forward(&x)?.flatten_all()?.to_vec1::<f32>()?;
        let b = dense_lin.forward(&x)?.flatten_all()?.to_vec1::<f32>()?;
        let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (x, y) in a.iter().zip(&b) {
            dot += *x as f64 * *y as f64;
            na += *x as f64 * *x as f64;
            nb += *y as f64 * *y as f64;
        }
        let cos = dot / (na.sqrt() * nb.sqrt() + 1e-12);
        assert!(
            cos > 0.997,
            "packed-load forward cosine {cos:.6} vs dense too low"
        );

        std::fs::remove_file(&tmp).ok();
        Ok(())
    }

    /// On-device tier build (`#[ignore]`d — needs a `Wan-AI/Wan2.2-*-Diffusers` snapshot). Run per
    /// model/tier, then `hf upload` the output dir to `SceneWorks/wan2.2-<model>-candle`:
    ///
    /// ```sh
    /// export SCENEWORKS_CANDLE_WAN_DIFFUSERS_DIR=<Wan-AI/Wan2.2-T2V-A14B-Diffusers snapshot>
    /// export SCENEWORKS_CANDLE_WAN_TIER_OUT=<out-dir>          # the packed candle tier is written here
    /// export SCENEWORKS_CANDLE_WAN_BITS=4                      # 4 (q4) or 8 (q8)
    /// cargo test -p candle-gen-wan --release build_candle_wan_tier_from_env -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "on-device tier build: needs a Wan-AI/Wan2.2-*-Diffusers snapshot on disk"]
    fn build_candle_wan_tier_from_env() {
        let diffusers = std::env::var("SCENEWORKS_CANDLE_WAN_DIFFUSERS_DIR")
            .expect("set SCENEWORKS_CANDLE_WAN_DIFFUSERS_DIR to the diffusers snapshot dir");
        let out = std::env::var("SCENEWORKS_CANDLE_WAN_TIER_OUT")
            .expect("set SCENEWORKS_CANDLE_WAN_TIER_OUT to the output tier dir");
        let bits: usize = std::env::var("SCENEWORKS_CANDLE_WAN_BITS")
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(4);
        assert!(
            bits == 4 || bits == 8,
            "SCENEWORKS_CANDLE_WAN_BITS must be 4 or 8"
        );
        super::build_candle_wan_tier(Path::new(&diffusers), Path::new(&out), bits)
            .expect("build candle wan tier");
        eprintln!("[[CANDLE-TIER]] done: q{bits} tier at {out}");
    }
}
