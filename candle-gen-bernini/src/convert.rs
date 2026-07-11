//! Offline builder for the **candle-native** Bernini renderer tier (the candle sibling of
//! `mlx_gen_wan::convert::assemble_bernini_renderer_snapshot`, sc-4705). Hostable at
//! `SceneWorks/bernini-candle`.
//!
//! The Bernini renderer IS Wan2.2-T2V-A14B, finetuned; the only Bernini-specific weights are the two
//! dual-expert DiTs, which the full `ByteDance/Bernini-Diffusers` package bundles into one combined
//! `bernini/` safetensors index (F32 shards) under the `diff_dec.transformer.` (high-noise) and
//! `diff_dec_low.transformer_2.` (low-noise) prefixes. **Crucially for candle**, stripping those
//! prefixes leaves the keys in the diffusers `WanTransformer3DModel` schema the candle
//! [`WanTransformer`](candle_gen_wan::transformer::WanTransformer) reads directly — so, unlike the MLX
//! converter (which then remaps to MLX-internal keys), candle needs **no key remap**: strip prefix →
//! pack → write.
//!
//! Each expert is written to a diffusers-layout component dir (`transformer/` + `transformer_2/`) with
//! every rank-2 `.weight` MLX-affine-packed at `bits` / group 64 via
//! [`pack_transformer_component`](candle_gen_wan::candle_tier_build::pack_transformer_component) (the
//! exact shape the sc-10025 packed-detect seam consumes) — or dense bf16 when `bits == 0`. The stock
//! Wan2.2 UMT5 (`text_encoder/`), z16 VAE (`vae/`), tokenizer (`tokenizer/`), scheduler, and
//! `model_index.json` are copied verbatim from a base Wan2.2-T2V-A14B diffusers snapshot (the reference
//! `BerniniRendererModel` itself loads T5/VAE from its `wan22_base`). A `bernini_renderer.json` sidecar
//! preserves the Bernini knobs (switch boundary / flow shift / src-id rotary).
//!
//! The [`build_bernini_candle_tier`] entry point is an `#[ignore]`d test (it needs the multi-GB package
//! on disk); the pure [`route_bernini_expert_key`] routing core is unit-tested in CI (no weights).

use std::collections::HashMap;
use std::path::Path;

use candle_gen::candle_core::{safetensors as cst, DType, Device, Result, Tensor};
use candle_gen_wan::candle_tier_build::{pack_transformer_component, TIER_GROUP_SIZE};

/// The two renderer-expert prefixes in the combined `bernini/` index → the diffusers component dir the
/// candle loader reads. `diff_dec.transformer.` = high-noise (`transformer/`), `diff_dec_low.transformer_2.`
/// = low-noise (`transformer_2/`).
const EXPERT_PREFIXES: [(&str, &str); 2] = [
    ("diff_dec_low.transformer_2.", "transformer_2"),
    ("diff_dec.transformer.", "transformer"),
];

/// Route a combined-index key to `(component dir, stripped diffusers key)`, or `None` if it is not a
/// renderer-expert tensor (the planner MLLM / connector / vit_decoder / mask_tokens / the redundant T5
/// copy are all skipped). `diff_dec_low.` is checked first, but the prefixes are disjoint anyway
/// (`diff_dec.` requires a literal `.` after `diff_dec`, which `diff_dec_low` does not have).
pub fn route_bernini_expert_key(k: &str) -> Option<(&'static str, String)> {
    for (prefix, out) in EXPERT_PREFIXES {
        if let Some(rest) = k.strip_prefix(prefix) {
            return Some((out, rest.to_string()));
        }
    }
    None
}

/// One sweep over the `bernini/` shards, routing each renderer-expert tensor to its component map with
/// the prefix stripped (diffusers keys). Non-expert tensors are never retained. Returns
/// `{"transformer" -> map, "transformer_2" -> map}`.
fn extract_experts(bernini_dir: &Path) -> Result<HashMap<&'static str, HashMap<String, Tensor>>> {
    let mut shards: Vec<std::path::PathBuf> = std::fs::read_dir(bernini_dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors"))
        .collect();
    shards.sort();
    if shards.is_empty() {
        return Err(candle_gen::candle_core::Error::Msg(format!(
            "build_bernini_candle_tier: no .safetensors shards under {} (point at the `bernini/` dir \
             of a ByteDance/Bernini-Diffusers snapshot)",
            bernini_dir.display()
        )));
    }
    let mut groups: HashMap<&'static str, HashMap<String, Tensor>> = HashMap::new();
    groups.insert("transformer", HashMap::new());
    groups.insert("transformer_2", HashMap::new());
    for shard in &shards {
        let map = cst::load(shard, &Device::Cpu)?;
        for (k, v) in map {
            if let Some((out, key)) = route_bernini_expert_key(&k) {
                groups
                    .get_mut(out)
                    .expect("component group present")
                    .insert(key, v);
            }
        }
    }
    for (name, g) in &groups {
        if g.is_empty() {
            return Err(candle_gen::candle_core::Error::Msg(format!(
                "build_bernini_candle_tier: no tensors routed to {name}/ (expected the \
                 diff_dec/diff_dec_low prefixes of a ByteDance/Bernini-Diffusers `bernini/` index)"
            )));
        }
    }
    Ok(groups)
}

/// Write one expert component `map` (diffusers keys) to `dst` as a single `model.safetensors`. When
/// `bits ∈ {4,8}` every rank-2 `.weight` is MLX-affine-packed via [`pack_transformer_component`]
/// (u32 codes + `.scales`/`.biases`, group [`TIER_GROUP_SIZE`]) and a `quantize_config.json` is written;
/// when `bits == 0` the whole component is cast to bf16 dense. Returns the number of Linears packed.
fn write_expert(map: HashMap<String, Tensor>, dst: &Path, bits: usize) -> Result<usize> {
    std::fs::create_dir_all(dst)?;
    // Cast the (F32 package) tensors to bf16 first: the dense passthrough leaves (norms,
    // patch_embedding conv, scale_shift_table, biases) then land bf16 (the DiT dtype), and the packer
    // re-casts the rank-2 weights it packs to f32 internally — so this only affects the dense leaves.
    let bf16: HashMap<String, Tensor> = map
        .into_iter()
        .map(|(k, v)| Ok((k, v.to_dtype(DType::BF16)?)))
        .collect::<Result<_>>()?;
    let (out, packed) = if bits == 0 {
        (bf16, 0)
    } else {
        pack_transformer_component(bf16, bits)?
    };
    cst::save(&out, dst.join("model.safetensors"))?;
    if bits != 0 {
        std::fs::write(
            dst.join("quantize_config.json"),
            format!(
                "{{\n  \"bits\": {bits},\n  \"quantization\": {{ \"group_size\": {TIER_GROUP_SIZE} }}\n}}\n"
            ),
        )?;
    }
    Ok(packed)
}

/// Copy a base-snapshot component dir/file into the tier verbatim, recursively (following symlinks so
/// the tier is self-contained — HF cache entries are file symlinks into `../../blobs`).
fn copy_recursive(src: &Path, dst: &Path) -> Result<()> {
    if std::fs::symlink_metadata(src)?.is_dir() {
        std::fs::create_dir_all(dst)?;
        for entry in std::fs::read_dir(src)? {
            let entry = entry?;
            copy_recursive(&entry.path(), &dst.join(entry.file_name()))?;
        }
    } else {
        if let Some(parent) = dst.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::copy(src, dst)?;
    }
    Ok(())
}

/// The Bernini renderer knobs distilled from the package `config.json` (switch boundary / flow shift /
/// src-id rotary), with the upstream defaults where a field is absent. Written as `bernini_renderer.json`.
fn bernini_renderer_knobs(pkg: &Path) -> serde_json::Value {
    use serde_json::json;
    let cfg: serde_json::Value = std::fs::read(pkg.join("config.json"))
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_else(|| json!({}));
    let f = |k: &str, d: f64| cfg.get(k).and_then(serde_json::Value::as_f64).unwrap_or(d);
    let b = |k: &str, d: bool| cfg.get(k).and_then(serde_json::Value::as_bool).unwrap_or(d);
    let i = |k: &str, d: i64| cfg.get(k).and_then(serde_json::Value::as_i64).unwrap_or(d);
    json!({
        "switch_dit_boundary": f("switch_dit_boundary", 0.875),
        "shift": f("shift", 3.0),
        "use_src_id_rotary_emb": b("use_src_id_rotary_emb", true),
        "interpolate_src_id": b("interpolate_src_id", true),
        "max_trained_src_id": i("max_trained_src_id", 5),
        "max_sequence_length": i("max_sequence_length", 512),
    })
}

/// Build one candle Bernini renderer tier from a `ByteDance/Bernini-Diffusers` package at
/// `bernini_diffusers_dir` (must contain a `bernini/` combined index) + a base Wan2.2-T2V-A14B diffusers
/// snapshot at `base_wan_snapshot` (supplying the stock UMT5 `text_encoder/`, z16 `vae/`, `tokenizer/`,
/// `scheduler/`, `model_index.json`), into `out_dir` at `bits` (4 = q4, 8 = q8, 0 = dense bf16). Host the
/// result at `SceneWorks/bernini-candle`.
pub fn build_bernini_candle_tier(
    bernini_diffusers_dir: &Path,
    base_wan_snapshot: &Path,
    out_dir: &Path,
    bits: usize,
) -> Result<()> {
    let bernini_dir = bernini_diffusers_dir.join("bernini");
    if !bernini_dir.is_dir() {
        return Err(candle_gen::candle_core::Error::Msg(format!(
            "build_bernini_candle_tier: no `bernini/` dir under {} (point at a ByteDance/Bernini-Diffusers \
             snapshot root)",
            bernini_diffusers_dir.display()
        )));
    }
    std::fs::create_dir_all(out_dir)?;

    // 1. The two renderer experts: strip prefix → diffusers keys → pack (or dense bf16).
    let mut groups = extract_experts(&bernini_dir)?;
    for (name, dir) in [
        ("transformer", "transformer"),
        ("transformer_2", "transformer_2"),
    ] {
        let map = groups.remove(name).expect("expert group present");
        let n = write_expert(map, &out_dir.join(dir), bits)?;
        eprintln!("[[BERNINI-CANDLE-TIER]] {name}: packed {n} Linears (bits={bits})");
    }

    // 2. Stock Wan2.2 components copied verbatim from the base snapshot.
    for name in [
        "text_encoder",
        "vae",
        "tokenizer",
        "scheduler",
        "model_index.json",
    ] {
        let src = base_wan_snapshot.join(name);
        if src.exists() {
            copy_recursive(&src, &out_dir.join(name))?;
        }
    }

    // 3. Bernini knobs sidecar.
    std::fs::write(
        out_dir.join("bernini_renderer.json"),
        serde_json::to_string_pretty(&bernini_renderer_knobs(bernini_diffusers_dir))
            .map_err(|e| candle_gen::candle_core::Error::Msg(e.to_string()))?,
    )?;
    eprintln!(
        "[[BERNINI-CANDLE-TIER]] done: tier at {}",
        out_dir.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The renderer-expert routing is a faithful prefix strip to diffusers keys: high/low route to
    /// `transformer`/`transformer_2`, the stripped key stays in the diffusers `WanTransformer3DModel`
    /// schema the candle loader reads, and every non-expert tensor is skipped.
    #[test]
    fn expert_key_routing() {
        let cases: &[(&str, Option<(&str, &str)>)] = &[
            (
                "diff_dec.transformer.blocks.0.attn1.to_q.weight",
                Some(("transformer", "blocks.0.attn1.to_q.weight")),
            ),
            (
                "diff_dec.transformer.patch_embedding.weight",
                Some(("transformer", "patch_embedding.weight")),
            ),
            (
                "diff_dec.transformer.condition_embedder.text_embedder.linear_1.weight",
                Some((
                    "transformer",
                    "condition_embedder.text_embedder.linear_1.weight",
                )),
            ),
            (
                "diff_dec.transformer.proj_out.weight",
                Some(("transformer", "proj_out.weight")),
            ),
            (
                "diff_dec.transformer.scale_shift_table",
                Some(("transformer", "scale_shift_table")),
            ),
            (
                "diff_dec_low.transformer_2.blocks.5.ffn.net.0.proj.weight",
                Some(("transformer_2", "blocks.5.ffn.net.0.proj.weight")),
            ),
            (
                "diff_dec_low.transformer_2.patch_embedding.bias",
                Some(("transformer_2", "patch_embedding.bias")),
            ),
            // Non-expert tensors: planner + redundant T5 are skipped.
            ("mllm.model.layers.0.self_attn.q_proj.weight", None),
            ("connector.proj_gen.0.weight", None),
            ("vit_decoder.net.cond_embed.weight", None),
            ("mask_tokens", None),
            ("t5_text_encoder.shared.weight", None),
        ];
        for (k, want) in cases {
            let got = route_bernini_expert_key(k);
            let got_ref = got.as_ref().map(|(o, s)| (*o, s.as_str()));
            assert_eq!(got_ref, *want, "routing {k}");
        }
    }

    /// The two expert prefixes are disjoint (neither shadows the other in first-match routing) —
    /// `diff_dec.` requires a literal `.` after `diff_dec`, which `diff_dec_low` lacks.
    #[test]
    fn expert_prefixes_disjoint() {
        // A `diff_dec_low` key must NEVER route to `transformer` (the high-noise expert).
        let (dir, _) =
            route_bernini_expert_key("diff_dec_low.transformer_2.blocks.0.attn1.to_q.weight")
                .unwrap();
        assert_eq!(dir, "transformer_2");
        // And a high-noise key must route to `transformer`.
        let (dir, _) =
            route_bernini_expert_key("diff_dec.transformer.blocks.0.attn1.to_q.weight").unwrap();
        assert_eq!(dir, "transformer");
    }

    /// On-device tier build (`#[ignore]`d — needs the ByteDance/Bernini-Diffusers package + a base
    /// Wan2.2-T2V-A14B diffusers snapshot on disk). Run per tier, then `hf upload` the output dir to
    /// `SceneWorks/bernini-candle`:
    ///
    /// ```sh
    /// export SCENEWORKS_BERNINI_DIFFUSERS_DIR=<ByteDance/Bernini-Diffusers snapshot root>
    /// export SCENEWORKS_BERNINI_BASE_WAN_DIR=<Wan-AI/Wan2.2-T2V-A14B-Diffusers snapshot>
    /// export SCENEWORKS_BERNINI_TIER_OUT=<out-dir>
    /// export SCENEWORKS_BERNINI_BITS=4                # 4 (q4) / 8 (q8) / 0 (dense bf16)
    /// cargo test -p candle-gen-bernini --release build_bernini_candle_tier_from_env -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "on-device tier build: needs the ByteDance/Bernini-Diffusers package + a base Wan2.2-T2V-A14B snapshot"]
    fn build_bernini_candle_tier_from_env() {
        let pkg = std::env::var("SCENEWORKS_BERNINI_DIFFUSERS_DIR")
            .expect("set SCENEWORKS_BERNINI_DIFFUSERS_DIR to the ByteDance/Bernini-Diffusers root");
        let base = std::env::var("SCENEWORKS_BERNINI_BASE_WAN_DIR")
            .expect("set SCENEWORKS_BERNINI_BASE_WAN_DIR to a Wan2.2-T2V-A14B diffusers snapshot");
        let out = std::env::var("SCENEWORKS_BERNINI_TIER_OUT")
            .expect("set SCENEWORKS_BERNINI_TIER_OUT to the output tier dir");
        let bits: usize = std::env::var("SCENEWORKS_BERNINI_BITS")
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(4);
        assert!(
            bits == 0 || bits == 4 || bits == 8,
            "BITS must be 0, 4, or 8"
        );
        build_bernini_candle_tier(Path::new(&pkg), Path::new(&base), Path::new(&out), bits)
            .expect("build bernini candle tier");
    }
}
