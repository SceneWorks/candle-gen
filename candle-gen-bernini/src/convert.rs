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
//! In addition to the two renderer experts, this converter also extracts the **semantic-planner**
//! components — the candle sibling of `mlx-gen-bernini/src/convert.rs` (sc-5144) — into the exact
//! candle turnkey layout [`crate::bernini::BerniniPlanner::load`] reads (sc-11061), so a single
//! converted directory is a complete full-`bernini` snapshot the whole planner→renderer pipeline can
//! load real weights from:
//!
//! | source prefix (combined `bernini/` index) | candle layout destination                        |
//! |-------------------------------------------|--------------------------------------------------|
//! | `mllm.model.*`  / `mllm.visual.*`         | `mllm/` component (backbone `.pp("model")`, vision `.pp("visual")`) |
//! | `mllm.lm_head.weight`                     | *dropped* (stateless extractor — no token head)  |
//! | `connector.proj_gen.*` / `connector.pred_vit.*` | `connector/` component (`MlpConnector`)    |
//! | `vit_decoder.net.*`                       | `vit_decoder/` component (`DiffLossFm` `.pp("net")`) |
//! | `mask_tokens`                             | `mask_tokens.safetensors` (key `mask_tokens`)    |
//!
//! plus the copied Qwen2.5-VL `tokenizer.json` (into `mllm/`), a `qwen2_5_vl_config.json` (copy of the
//! package's `mllm/config.json`, read by the backbone/vision/MRoPE configs), and a `bernini_planner.json`
//! sidecar (`num_mask_token`, `max_sequence_length`, `clip_diff_cfg` z_channels + shift).
//!
//! **Planner quant (sc-11062).** When `bits ∈ {4, 8}`, the `mllm/` component's Qwen2.5-VL **LLM text
//! linears** (attention `q/k/v/o_proj` + MLP `gate/up/down_proj`, per decoder layer) are MLX-affine-packed
//! at group [`TIER_GROUP_SIZE`] into the `{base}.weight` (u32) + `.scales` + `.biases` triple the loader's
//! packed-detect [`crate::qwen2_5_vl`] `Attn`/`Mlp` seam reads, with a `mllm/quantize_config.json` emitted
//! alongside. The token embedding, RMS norms, biases, and the **entire vision tower** (`visual.*`,
//! group-64-misaligned) stay dense bf16, as do the `connector/` and `vit_decoder/` components — mirroring
//! the MLX lane's conservative planner quant policy (sc-5146: only the ~7B LLM text linears quantize).
//! `bits == 0` writes the whole planner dense bf16 (unchanged).
//!
//! The [`build_bernini_candle_tier`] entry point is an `#[ignore]`d test (it needs the multi-GB package
//! on disk); the pure [`route_bernini_expert_key`] / [`route_bernini_planner_key`] routing cores are
//! unit-tested in CI (no weights), as is the on-disk layout contract vs. the loader.

use std::collections::HashMap;
use std::path::Path;

use candle_gen::candle_core::{safetensors as cst, DType, Device, Result, Tensor};
use candle_gen::quant::pack_mlx_affine;
use candle_gen_wan::candle_tier_build::{pack_transformer_component, TIER_GROUP_SIZE};

/// The two renderer-expert prefixes in the combined `bernini/` index → the diffusers component dir the
/// candle loader reads. `diff_dec.transformer.` = high-noise (`transformer/`), `diff_dec_low.transformer_2.`
/// = low-noise (`transformer_2/`).
const EXPERT_PREFIXES: [(&str, &str); 2] = [
    ("diff_dec_low.transformer_2.", "transformer_2"),
    ("diff_dec.transformer.", "transformer"),
];

// --- Planner on-disk layout contract (shared with the loader) ---------------------------------------
//
// These consts are the SINGLE SOURCE OF TRUTH for the candle full-`bernini` planner layout: the
// converter writes them and [`crate::bernini::BerniniPlanner::load`] reads them, so the two can never
// drift (a rename breaks the compile of both). The `*_PP` consts are the intra-component weight-key
// namespaces the loader `.pp(..)`s into (and that routing strips to).

/// `mllm/` component dir — holds the Qwen2.5-VL backbone (`model.*`) + vision tower (`visual.*`) weights
/// **and** the copied `tokenizer.json`.
pub const PLANNER_MLLM_DIR: &str = "mllm";
/// Backbone weight-key namespace inside `mllm/` (loader `.pp("model")`).
pub const PLANNER_MLLM_BACKBONE_PP: &str = "model";
/// Vision-tower weight-key namespace inside `mllm/` (loader `.pp("visual")`).
pub const PLANNER_MLLM_VISION_PP: &str = "visual";
/// `connector/` component dir — the `proj_gen.*` / `pred_vit.*` MLP connector.
pub const PLANNER_CONNECTOR_DIR: &str = "connector";
/// `vit_decoder/` component dir — the clip-diff flow head, keyed under `net.*`.
pub const PLANNER_VIT_DECODER_DIR: &str = "vit_decoder";
/// Clip-diff weight-key namespace inside `vit_decoder/` (loader `.pp("net")`).
pub const PLANNER_VIT_DECODER_PP: &str = "net";
/// The MAR mask-token file at the snapshot root (single tensor keyed [`PLANNER_MASK_TOKENS_KEY`]).
pub const PLANNER_MASK_TOKENS_FILE: &str = "mask_tokens.safetensors";
/// The tensor key inside [`PLANNER_MASK_TOKENS_FILE`].
pub const PLANNER_MASK_TOKENS_KEY: &str = "mask_tokens";
/// The Qwen2.5-VL config at the snapshot root (verbatim copy of the package `mllm/config.json`).
pub const PLANNER_QWEN_CONFIG_FILE: &str = "qwen2_5_vl_config.json";
/// The tokenizer copied into `mllm/` (loader reads `mllm/tokenizer.json`).
pub const PLANNER_TOKENIZER_FILE: &str = "tokenizer.json";
/// The distilled planner-knobs sidecar at the snapshot root.
pub const PLANNER_SIDECAR_FILE: &str = "bernini_planner.json";

/// The three prefix-routed planner component groups (dir-based). `mllm.lm_head` and the bare
/// `mask_tokens` parameter are handled outside this table (see [`route_bernini_planner_key`]). No prefix
/// here is a prefix of another, so first-match routing is unambiguous.
const PLANNER_PREFIXES: [(&str, &str); 3] = [
    ("mllm.", PLANNER_MLLM_DIR),
    ("connector.", PLANNER_CONNECTOR_DIR),
    ("vit_decoder.", PLANNER_VIT_DECODER_DIR),
];

/// Authoritative exact tensor counts for the three dir-based planner components, mirroring the
/// mlx-gen-bernini converter's `Component::expect` asserts (`mlx-gen-bernini/src/convert.rs`):
///   - `mllm` **728**: Qwen2.5-VL-7B `visual.*` (390) + `model.*` (338), after dropping `mllm.lm_head.weight`.
///   - `connector` **12**: `MLPConnector` (`proj_gen` 5 + `pred_vit` 7).
///   - `vit_decoder` **140**: `DiffLoss_FM` net (time/cond embed + input proj + 16 res blocks + final layer).
///
/// The single `mask_tokens` parameter (mlx `expect: 1`) is guarded separately by its `Option` presence
/// check ([`plan_components`]). Do NOT loosen these — a mismatch means the package layout changed.
const PLANNER_EXPECTED_COUNTS: [(&str, usize); 3] = [
    (PLANNER_MLLM_DIR, 728),
    (PLANNER_CONNECTOR_DIR, 12),
    (PLANNER_VIT_DECODER_DIR, 140),
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

/// Route a combined-index key to `(planner destination, stripped key)`, or `None` if it is not a planner
/// tensor (a renderer DiT, the redundant `t5_text_encoder.*` copy, or the dropped `mllm.lm_head`).
///
/// The destination is either a component dir ([`PLANNER_MLLM_DIR`] / [`PLANNER_CONNECTOR_DIR`] /
/// [`PLANNER_VIT_DECODER_DIR`] — the stripped key stays in the diffusers key schema the corresponding
/// loader reads: `model.*`/`visual.*`, `proj_gen.*`/`pred_vit.*`, `net.*`) or the special
/// [`PLANNER_MASK_TOKENS_KEY`] sentinel for the bare `mask_tokens` parameter (no trailing segment, so its
/// key is unchanged), which the writer routes to the root-level [`PLANNER_MASK_TOKENS_FILE`].
pub fn route_bernini_planner_key(k: &str) -> Option<(&'static str, String)> {
    // The planner is a stateless feature extractor — the Qwen LM head is never used.
    if k == "mllm.lm_head.weight" {
        return None;
    }
    // The bare MAR mask-token parameter has no trailing segment; its key is kept verbatim.
    if k == PLANNER_MASK_TOKENS_KEY {
        return Some((PLANNER_MASK_TOKENS_KEY, PLANNER_MASK_TOKENS_KEY.to_string()));
    }
    for (prefix, out) in PLANNER_PREFIXES {
        if let Some(rest) = k.strip_prefix(prefix) {
            return Some((out, rest.to_string()));
        }
    }
    None
}

/// The full routing plan over the combined `bernini/` index, built from a **header-only** scan of the
/// mmap'd shards — no tensor data is materialized here. Each output component maps to the list of
/// `(source key, stripped destination key)` pairs that feed it, plus the source key of the bare
/// `mask_tokens` parameter. The [`cst::MmapedSafetensors`] keeps every shard memory-mapped so the builder
/// can then load, convert, and write **one component at a time** (`st.load(src_key)` → write → drop),
/// bounding peak RSS to a single component rather than materializing the whole ~150 GB index at once
/// (sc-11169 / F-099). The old one-shot extractor accumulated both F32 experts (~114 GB) plus the planner
/// (~33 GB) in host RAM simultaneously; this streams instead.
struct RoutingPlan {
    /// All `bernini/` shards, memory-mapped (headers parsed; tensor data lazily loaded on demand).
    st: cst::MmapedSafetensors,
    /// `{"transformer" -> [(src, stripped)], "transformer_2" -> ...}` (diffusers keys).
    experts: HashMap<&'static str, Vec<(String, String)>>,
    /// `{"mllm" -> [(src, stripped)], "connector" -> ..., "vit_decoder" -> ...}` (stripped planner keys).
    planner: HashMap<&'static str, Vec<(String, String)>>,
    /// Source key of the bare `mask_tokens` parameter (always `"mask_tokens"` when present).
    mask_tokens_key: Option<String>,
}

/// Header-only scan of the `bernini/` shards: memory-map every shard via [`cst::MmapedSafetensors::multi`]
/// (headers parsed, no tensor data read) and route each tensor **key** to its renderer-expert **or**
/// planner group with the destination prefix stripped. Every non-Bernini key (the redundant T5 copy, the
/// dropped `mllm.lm_head`) is skipped. Because only keys are inspected, this holds no weight data — the
/// returned [`RoutingPlan`] lets the builder load and write one component at a time. The same invariants
/// the old one-shot extractor enforced (experts non-empty, exact planner counts, `mask_tokens` present)
/// are validated here up front, from the headers, so the expensive on-device build still fails loud and
/// early. Keys are deduplicated (shards are disjoint in practice; a key present in two shards resolves to
/// the last-mapped shard, matching [`cst::MmapedSafetensors`]'s own last-wins `load`).
fn plan_components(bernini_dir: &Path) -> Result<RoutingPlan> {
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
    // SAFETY: inherited from memmap2 — the shard files are only read (never mutated) for the lifetime of
    // the returned plan, and we never expose the mapped bytes outside `st.load`.
    let st = unsafe { cst::MmapedSafetensors::multi(&shards)? };

    let mut experts: HashMap<&'static str, Vec<(String, String)>> = HashMap::new();
    experts.insert("transformer", Vec::new());
    experts.insert("transformer_2", Vec::new());
    let mut planner: HashMap<&'static str, Vec<(String, String)>> = HashMap::new();
    planner.insert(PLANNER_MLLM_DIR, Vec::new());
    planner.insert(PLANNER_CONNECTOR_DIR, Vec::new());
    planner.insert(PLANNER_VIT_DECODER_DIR, Vec::new());
    let mut mask_tokens_key: Option<String> = None;

    // Dedup keys defensively (disjoint shards in practice) so a duplicate is routed once.
    let mut seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for (name, _view) in st.tensors() {
        if !seen.insert(name.clone()) {
            continue;
        }
        if let Some((out, key)) = route_bernini_expert_key(&name) {
            experts
                .get_mut(out)
                .expect("expert group present")
                .push((name, key));
        } else if let Some((out, key)) = route_bernini_planner_key(&name) {
            if out == PLANNER_MASK_TOKENS_KEY {
                mask_tokens_key = Some(name);
            } else {
                planner
                    .get_mut(out)
                    .expect("planner group present")
                    .push((name, key));
            }
        }
    }

    // Renderer experts have no fixed count guard here (the diffusers-key strip is a pass-through; the
    // packer/loader validate their schema) — only assert they are non-empty.
    for (name, keys) in experts.iter() {
        if keys.is_empty() {
            return Err(candle_gen::candle_core::Error::Msg(format!(
                "build_bernini_candle_tier: no tensors routed to {name}/ (expected the \
                 diff_dec/diff_dec_low prefixes of a ByteDance/Bernini-Diffusers `bernini/` index)"
            )));
        }
    }
    // Planner components: HARD exact per-component tensor-count guard, mirroring the mlx-gen-bernini
    // converter's `Component::expect` asserts.
    validate_planner_counts(&planner)?;
    if mask_tokens_key.is_none() {
        return Err(candle_gen::candle_core::Error::Msg(
            "build_bernini_candle_tier: no `mask_tokens` parameter found in the `bernini/` index"
                .into(),
        ));
    }
    Ok(RoutingPlan {
        st,
        experts,
        planner,
        mask_tokens_key,
    })
}

/// HARD exact per-component tensor-count guard for the three dir-based planner groups, mirroring the
/// mlx-gen-bernini converter's `Component::expect` asserts ([`PLANNER_EXPECTED_COUNTS`]). Returns a
/// clear `Err` naming the component + expected-vs-actual on any mismatch (a re-layout in a future
/// package revision), so the expensive on-device build fails loud and early instead of at GPU-val load.
/// Do NOT loosen these counts.
fn validate_planner_counts<V>(planner: &HashMap<&'static str, Vec<V>>) -> Result<()> {
    for (dir, expect) in PLANNER_EXPECTED_COUNTS {
        let got = planner.get(dir).map(Vec::len).unwrap_or(0);
        if got != expect {
            return Err(candle_gen::candle_core::Error::Msg(format!(
                "build_bernini_candle_tier: planner component {dir}/ expected {expect} tensors, got \
                 {got} — the ByteDance/Bernini-Diffusers planner layout may have changed"
            )));
        }
    }
    Ok(())
}

/// The base Wan2.2 snapshot components the candle renderer loader **must** read — copying any of
/// these is not optional: [`crate::bernini::BerniniRenderer`] / [`crate::pipeline`] load the UMT5
/// `text_encoder/`, the z16 `vae/`, and the UMT5 `tokenizer/` (`tokenizer/tokenizer.json`) at
/// component-load time, so a base snapshot missing any of them yields a broken, unloadable tier.
const REQUIRED_BASE_WAN_COMPONENTS: [&str; 3] = ["text_encoder", "vae", "tokenizer"];

/// The base Wan2.2 snapshot components copied for diffusers-layout completeness but **not** read by the
/// candle loader (the sampler is a `FlowScheduler` built in-code from the Bernini knobs, and
/// `model_index.json` is diffusers metadata). Copied best-effort — a missing one is not fatal.
const OPTIONAL_BASE_WAN_COMPONENTS: [&str; 2] = ["scheduler", "model_index.json"];

/// Assert every [`REQUIRED_BASE_WAN_COMPONENTS`] entry exists under the base Wan snapshot, returning a
/// clear `Err` naming the first missing component + its expected source path. Mirrors
/// [`require_planner_sources`] (and the mlx-gen-bernini converter's unconditional `place()`): a
/// base-Wan snapshot lacking the UMT5 `text_encoder/` / z16 `vae/` / `tokenizer/` the renderer loads
/// must fail LOUD at build time rather than silently ship a VAE-/encoder-less tier that only surfaces
/// as a broken load later (sc-11631; found in sc-11003 where a base snapshot lacked `vae/`).
fn require_base_wan_sources(base_wan_snapshot: &Path) -> Result<()> {
    for name in REQUIRED_BASE_WAN_COMPONENTS {
        let src = base_wan_snapshot.join(name);
        if !src.exists() {
            return Err(candle_gen::candle_core::Error::Msg(format!(
                "build_bernini_candle_tier: missing required base-Wan renderer component {} (the \
                 base Wan2.2-T2V-A14B snapshot must contain `{name}/` — the candle Bernini renderer \
                 loads the UMT5 text_encoder, z16 vae, and tokenizer from it)",
                src.display()
            )));
        }
    }
    Ok(())
}

/// Assert both planner source files the loader requires exist in the package `mllm/` dir, returning
/// their paths. The mlx-gen-bernini converter `place()`s `config.json` + `tokenizer.json`
/// unconditionally (erroring if absent); this mirrors that — a missing source is a loud build-time
/// `Err`, not a silently-incomplete snapshot the loader later chokes on.
fn require_planner_sources(pkg_mllm: &Path) -> Result<(std::path::PathBuf, std::path::PathBuf)> {
    let cfg_src = pkg_mllm.join("config.json");
    if !cfg_src.exists() {
        return Err(candle_gen::candle_core::Error::Msg(format!(
            "build_bernini_candle_tier: missing required planner config {} (the \
             ByteDance/Bernini-Diffusers package must contain `mllm/config.json`)",
            cfg_src.display()
        )));
    }
    let tok_src = pkg_mllm.join(PLANNER_TOKENIZER_FILE);
    if !tok_src.exists() {
        return Err(candle_gen::candle_core::Error::Msg(format!(
            "build_bernini_candle_tier: missing required planner tokenizer {} (the \
             ByteDance/Bernini-Diffusers package must contain `mllm/{PLANNER_TOKENIZER_FILE}`)",
            tok_src.display()
        )));
    }
    Ok((cfg_src, tok_src))
}

/// Write one expert component to `dst` as a single `model.safetensors`, **streaming** the source tensors
/// one at a time from the mmap'd shards (`keys` = the `(source key, stripped diffusers key)` pairs the
/// [`RoutingPlan`] routed to this expert). When `bits ∈ {4,8}` every rank-2 `.weight` is MLX-affine-packed
/// via [`pack_transformer_component`] (u32 codes + `.scales`/`.biases`, group [`TIER_GROUP_SIZE`]) and a
/// `quantize_config.json` is written; when `bits == 0` the whole component is dense bf16. Returns the
/// number of Linears packed.
///
/// Byte-identity with the pre-streaming (whole-map) path is preserved exactly: each source F32 tensor is
/// cast to bf16 **first** — even the rank-2 weights the packer then re-casts bf16→f32 internally — because
/// the old path cast the entire map to bf16 before packing, and that intermediate rounding is
/// load-bearing. safetensors serializes in a canonical (dtype, name) order, so the streamed insertion
/// order does not affect the output bytes. Each loaded F32 tensor is dropped at the end of its iteration,
/// so peak RSS is one source tensor plus the (packed/bf16) output map — never the whole ~114 GB expert
/// in F32.
fn write_expert_streamed(
    st: &cst::MmapedSafetensors,
    keys: &[(String, String)],
    dst: &Path,
    bits: usize,
) -> Result<usize> {
    std::fs::create_dir_all(dst)?;
    let mut out: HashMap<String, Tensor> = HashMap::with_capacity(keys.len() * 3);
    let mut packed = 0usize;
    for (src, stripped) in keys {
        // Load ONE tensor from the mmap (F32 package dtype) and cast to bf16, exactly as the old
        // whole-map pass did before packing.
        let value = st.load(src, &Device::Cpu)?.to_dtype(DType::BF16)?;
        if bits == 0 {
            out.insert(stripped.clone(), value);
        } else {
            // Feed a single-entry map through the shared packer so the per-tensor pack/dense decision is
            // identical to the whole-map path (rank-2 `.weight` → u32+scales+biases; everything else
            // dense bf16 passthrough).
            let one: HashMap<String, Tensor> = std::iter::once((stripped.clone(), value)).collect();
            let (packed_one, n) = pack_transformer_component(one, bits)?;
            out.extend(packed_one);
            packed += n;
        }
        // `value` (or the single-entry map) is dropped here, bounding peak RSS.
    }
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

/// Write one **dense bf16** planner component (`connector/` / `vit_decoder/`) to `dst` as a single
/// `model.safetensors`, **streaming** the source tensors one at a time from the mmap'd shards (`keys` =
/// the `(source key, stripped key)` pairs routed to this component). These two components are never packed
/// (small, and the clip_diff head runs ~75× through the MAR loop with triple-CFG where 4-bit error would
/// compound — sc-5146); the `mllm/` LLM linears DO pack (see [`write_planner_mllm_streamed`], sc-11062).
/// The loader reads these dense (bf16, the Qwen2.5-VL native dtype). Byte-identical to the pre-streaming
/// path (same per-tensor bf16 cast; canonical safetensors ordering), but peak RSS is one source tensor
/// plus the bf16 output map instead of the whole component in F32.
fn write_planner_component_streamed(
    st: &cst::MmapedSafetensors,
    keys: &[(String, String)],
    dst: &Path,
) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    let mut out: HashMap<String, Tensor> = HashMap::with_capacity(keys.len());
    for (src, stripped) in keys {
        let value = st.load(src, &Device::Cpu)?.to_dtype(DType::BF16)?;
        out.insert(stripped.clone(), value);
    }
    cst::save(&out, dst.join("model.safetensors"))?;
    Ok(())
}

/// Whether a **stripped** `mllm/` key (`model.*` / `visual.*`) is one of the Qwen2.5-VL **LLM text
/// linears** the tier packs at q4/q8: attention `q/k/v/o_proj` + MLP `gate/up/down_proj`, per decoder
/// layer. Everything else in the `mllm/` component stays dense — the token embedding
/// (`model.embed_tokens.weight`), the RMS norms (`*.layernorm.weight`, `model.norm.weight`), all biases,
/// and the ENTIRE vision tower (`visual.*`, whose linears are group-64-misaligned). This is EXACTLY the
/// Linear set the loader's packed-detect seam routes through `linear_detect` ([`crate::qwen2_5_vl`]'s
/// `Attn`/`Mlp`), and the same set the MLX lane quantizes (sc-5146) — the two must stay aligned (packing a
/// weight the loader reads densely would feed u32 codes into a bf16 read as garbage).
fn is_planner_llm_linear(key: &str) -> bool {
    let Some(rest) = key.strip_prefix("model.layers.") else {
        return false;
    };
    let Some((idx, tail)) = rest.split_once('.') else {
        return false;
    };
    if idx.is_empty() || !idx.bytes().all(|b| b.is_ascii_digit()) {
        return false;
    }
    matches!(
        tail,
        "self_attn.q_proj.weight"
            | "self_attn.k_proj.weight"
            | "self_attn.v_proj.weight"
            | "self_attn.o_proj.weight"
            | "mlp.gate_proj.weight"
            | "mlp.up_proj.weight"
            | "mlp.down_proj.weight"
    )
}

/// Write the `mllm/` planner component to `dst`, MLX-affine-packing **only the Qwen2.5-VL LLM text
/// linears** ([`is_planner_llm_linear`]) at `bits ∈ {4, 8}` / group [`TIER_GROUP_SIZE`] into the
/// `{base}.weight` (u32) + `.scales` + `.biases` triple the loader's packed-detect [`crate::qwen2_5_vl`]
/// seam consumes (sc-11062), and emitting a `quantize_config.json`. The token embedding, RMS norms,
/// biases, and the entire vision tower (`visual.*`) stay dense bf16. `bits == 0` writes the whole
/// component dense (identical to [`write_planner_component_streamed`]). Streams one source tensor at a
/// time from the mmap'd shards (peak RSS = one source tensor plus the output map). Each rank-2 weight is
/// cast to bf16 **first** (mirroring [`write_expert_streamed`] — the MLX numerics quantize from bf16),
/// then packed (the packer re-casts bf16→f32 internally). Returns the number of LLM Linears packed.
fn write_planner_mllm_streamed(
    st: &cst::MmapedSafetensors,
    keys: &[(String, String)],
    dst: &Path,
    bits: usize,
) -> Result<usize> {
    std::fs::create_dir_all(dst)?;
    let mut out: HashMap<String, Tensor> = HashMap::with_capacity(keys.len());
    let mut packed = 0usize;
    for (src, stripped) in keys {
        let value = st.load(src, &Device::Cpu)?.to_dtype(DType::BF16)?;
        if bits != 0 && is_planner_llm_linear(stripped) {
            let base = stripped
                .strip_suffix(".weight")
                .expect("an LLM linear key ends with .weight");
            let (wq, scales, biases) =
                pack_mlx_affine(&value.to_dtype(DType::F32)?, bits, TIER_GROUP_SIZE)?;
            out.insert(format!("{base}.weight"), wq);
            out.insert(format!("{base}.scales"), scales);
            out.insert(format!("{base}.biases"), biases);
            packed += 1;
        } else {
            // Dense passthrough: token embedding, RMS norms, biases, the whole vision tower.
            out.insert(stripped.clone(), value);
        }
    }
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

/// Write the single MAR `mask_tokens` parameter (dense bf16) to the root-level
/// [`PLANNER_MASK_TOKENS_FILE`] under key [`PLANNER_MASK_TOKENS_KEY`] (the loader's
/// `safetensors::load(..).get("mask_tokens")`).
fn write_mask_tokens(mask: Tensor, out_dir: &Path) -> Result<()> {
    let mut m: HashMap<String, Tensor> = HashMap::new();
    m.insert(
        PLANNER_MASK_TOKENS_KEY.to_string(),
        mask.to_dtype(DType::BF16)?,
    );
    cst::save(&m, out_dir.join(PLANNER_MASK_TOKENS_FILE))?;
    Ok(())
}

/// The planner knobs the [`crate::bernini::BerniniPlanner`] loader reads from `bernini_planner.json`:
/// `num_mask_token`, `max_sequence_length`, and the `clip_diff_cfg` object (the loader pulls
/// `z_channels` + `shift` from it). Distilled from the package `config.json`, with the upstream defaults
/// where a field is absent (and a synthesized `clip_diff_cfg` so `z_channels`/`shift` always resolve).
fn bernini_planner_knobs(pkg: &Path) -> serde_json::Value {
    use serde_json::json;
    let cfg: serde_json::Value = std::fs::read(pkg.join("config.json"))
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_else(|| json!({}));
    let i = |k: &str, d: i64| cfg.get(k).and_then(serde_json::Value::as_i64).unwrap_or(d);
    let clip_diff_cfg = cfg
        .get("clip_diff_cfg")
        .cloned()
        .unwrap_or_else(|| json!({ "z_channels": 3584, "shift": 2.0 }));
    let connector_cfg = cfg
        .get("connector_cfg")
        .cloned()
        .unwrap_or_else(|| json!({}));
    json!({
        "num_mask_token": i("num_mask_token", 4096),
        "max_sequence_length": i("max_sequence_length", 512),
        "clip_diff_cfg": clip_diff_cfg,
        "connector_cfg": connector_cfg,
    })
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

    // 1. Header-only routing plan over the mmap'd combined index (no weight data materialized): the two
    // renderer experts + the planner components. Each component is then loaded, converted, and written one
    // at a time below, bounding peak RSS to a single component (sc-11169 / F-099).
    let plan = plan_components(&bernini_dir)?;

    // 1a. Renderer experts: stream each source tensor → strip prefix → diffusers keys → pack (or dense
    // bf16). `write_expert_streamed` loads one tensor at a time from the mmap and drops it after
    // converting, so the whole F32 expert never lands in host RAM at once.
    for name in ["transformer", "transformer_2"] {
        let keys = plan.experts.get(name).expect("expert group present");
        let n = write_expert_streamed(&plan.st, keys, &out_dir.join(name), bits)?;
        eprintln!("[[BERNINI-CANDLE-TIER]] {name}: packed {n} Linears (bits={bits})");
    }

    // 1b. Planner `mllm/` component: the Qwen2.5-VL LLM text linears pack at q4/q8 (group 64), while the
    // token embedding / norms / biases / vision tower stay dense (sc-11062). `bits == 0` writes it fully
    // dense. Streamed one source tensor at a time.
    let mllm_keys = plan
        .planner
        .get(PLANNER_MLLM_DIR)
        .expect("planner group present");
    let mllm_n = mllm_keys.len();
    let packed =
        write_planner_mllm_streamed(&plan.st, mllm_keys, &out_dir.join(PLANNER_MLLM_DIR), bits)?;
    eprintln!(
        "[[BERNINI-CANDLE-TIER]] {PLANNER_MLLM_DIR}: {mllm_n} tensors, {packed} LLM Linears packed (bits={bits})"
    );

    // 1c. `connector/` + `vit_decoder/`: always dense bf16 (small; clip_diff runs ~75× through the MAR
    // loop with triple-CFG where 4-bit error would compound — sc-5146). Streamed likewise.
    for dir in [PLANNER_CONNECTOR_DIR, PLANNER_VIT_DECODER_DIR] {
        let keys = plan.planner.get(dir).expect("planner group present");
        let n = keys.len();
        write_planner_component_streamed(&plan.st, keys, &out_dir.join(dir))?;
        eprintln!("[[BERNINI-CANDLE-TIER]] {dir}: {n} dense bf16 tensors");
    }
    let mask_key = plan
        .mask_tokens_key
        .as_deref()
        .expect("mask_tokens present (plan_components guarantees it)");
    let mask_tokens = plan.st.load(mask_key, &Device::Cpu)?;
    write_mask_tokens(mask_tokens, out_dir)?;

    // 2. Stock Wan2.2 components copied verbatim from the base snapshot. The renderer loader REQUIRES
    // the UMT5 `text_encoder/`, z16 `vae/`, and `tokenizer/`; a base snapshot missing any of them is a
    // loud build-time `Err` (sc-11631) — never a silently-emitted, unloadable tier. `scheduler/` +
    // `model_index.json` are diffusers-layout completeness only (the sampler is built in-code), so they
    // stay best-effort copies.
    require_base_wan_sources(base_wan_snapshot)?;
    for name in REQUIRED_BASE_WAN_COMPONENTS {
        copy_recursive(&base_wan_snapshot.join(name), &out_dir.join(name))?;
    }
    for name in OPTIONAL_BASE_WAN_COMPONENTS {
        let src = base_wan_snapshot.join(name);
        if src.exists() {
            copy_recursive(&src, &out_dir.join(name))?;
        }
    }

    // 3a. Planner configs: the Qwen2.5-VL config (verbatim `mllm/config.json`) + the tokenizer copied
    // into `mllm/` (the loader reads `mllm/tokenizer.json` for the chat template) + the knobs sidecar.
    // Both the Qwen2.5-VL `config.json` and `tokenizer.json` are REQUIRED by the planner loader — fail
    // loud at build time (mirroring mlx-gen-bernini's unconditional `place()`) rather than silently
    // emitting a snapshot the loader can't read.
    let pkg_mllm = bernini_diffusers_dir.join(PLANNER_MLLM_DIR);
    let (cfg_src, tok_src) = require_planner_sources(&pkg_mllm)?;
    copy_recursive(&cfg_src, &out_dir.join(PLANNER_QWEN_CONFIG_FILE))?;
    copy_recursive(
        &tok_src,
        &out_dir.join(PLANNER_MLLM_DIR).join(PLANNER_TOKENIZER_FILE),
    )?;
    std::fs::write(
        out_dir.join(PLANNER_SIDECAR_FILE),
        serde_json::to_string_pretty(&bernini_planner_knobs(bernini_diffusers_dir))
            .map_err(|e| candle_gen::candle_core::Error::Msg(e.to_string()))?,
    )?;

    // 3b. Bernini renderer knobs sidecar.
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

    /// sc-11061: the planner routing is a faithful prefix strip into the candle turnkey layout the
    /// [`crate::bernini::BerniniPlanner`] loader reads — `mllm.model.*`/`mllm.visual.*` → `mllm/`
    /// (keys stay `model.*`/`visual.*` for the loader's `.pp("model")`/`.pp("visual")`),
    /// `connector.{proj_gen,pred_vit}.*` → `connector/`, `vit_decoder.net.*` → `vit_decoder/` (keeps
    /// `net.*` for `.pp("net")`), the bare `mask_tokens` → the mask-tokens sentinel; the renderer DiTs,
    /// the redundant T5 copy, and the dropped `mllm.lm_head` are all skipped.
    #[test]
    fn planner_key_routing() {
        let cases: &[(&str, Option<(&str, &str)>)] = &[
            // Qwen2.5-VL backbone + vision → `mllm/`, prefix stripped, `.pp` namespace preserved.
            (
                "mllm.model.layers.0.self_attn.q_proj.weight",
                Some((PLANNER_MLLM_DIR, "model.layers.0.self_attn.q_proj.weight")),
            ),
            (
                "mllm.model.embed_tokens.weight",
                Some((PLANNER_MLLM_DIR, "model.embed_tokens.weight")),
            ),
            (
                "mllm.model.norm.weight",
                Some((PLANNER_MLLM_DIR, "model.norm.weight")),
            ),
            (
                "mllm.visual.blocks.0.attn.qkv.weight",
                Some((PLANNER_MLLM_DIR, "visual.blocks.0.attn.qkv.weight")),
            ),
            (
                "mllm.visual.patch_embed.proj.weight",
                Some((PLANNER_MLLM_DIR, "visual.patch_embed.proj.weight")),
            ),
            // Connector branches → `connector/`.
            (
                "connector.proj_gen.0.weight",
                Some((PLANNER_CONNECTOR_DIR, "proj_gen.0.weight")),
            ),
            (
                "connector.pred_vit.3.weight",
                Some((PLANNER_CONNECTOR_DIR, "pred_vit.3.weight")),
            ),
            // ViT decoder (DiffLoss_FM) → `vit_decoder/`, keeps the `net.` substructure.
            (
                "vit_decoder.net.cond_embed.weight",
                Some((PLANNER_VIT_DECODER_DIR, "net.cond_embed.weight")),
            ),
            (
                "vit_decoder.net.res_blocks.7.adaLN_modulation.1.weight",
                Some((
                    PLANNER_VIT_DECODER_DIR,
                    "net.res_blocks.7.adaLN_modulation.1.weight",
                )),
            ),
            (
                "vit_decoder.net.final_layer.linear.weight",
                Some((PLANNER_VIT_DECODER_DIR, "net.final_layer.linear.weight")),
            ),
            // The bare MAR mask-token parameter → the mask-tokens sentinel, key unchanged.
            (
                "mask_tokens",
                Some((PLANNER_MASK_TOKENS_KEY, PLANNER_MASK_TOKENS_KEY)),
            ),
            // Dropped / skipped.
            ("mllm.lm_head.weight", None),
            ("diff_dec.transformer.blocks.0.attn1.to_q.weight", None),
            ("diff_dec_low.transformer_2.patch_embedding.weight", None),
            ("t5_text_encoder.shared.weight", None),
        ];
        for (k, want) in cases {
            let got = route_bernini_planner_key(k);
            let got_ref = got.as_ref().map(|(o, s)| (*o, s.as_str()));
            assert_eq!(got_ref, *want, "routing {k}");
        }
    }

    /// The renderer-expert and planner routers partition the index: no key routes to both, and every
    /// planner prefix is disjoint from the others (no first-match shadowing). The `mllm.` prefix does not
    /// swallow the dropped `lm_head`, and `mask_tokens` (no trailing dot) does not shadow any prefix.
    #[test]
    fn expert_and_planner_routers_are_disjoint() {
        let expert_keys = [
            "diff_dec.transformer.blocks.0.attn1.to_q.weight",
            "diff_dec_low.transformer_2.patch_embedding.weight",
        ];
        for k in expert_keys {
            assert!(route_bernini_expert_key(k).is_some(), "{k} is an expert");
            assert!(
                route_bernini_planner_key(k).is_none(),
                "{k} must NOT route to a planner group"
            );
        }
        let planner_keys = [
            "mllm.model.norm.weight",
            "mllm.visual.patch_embed.proj.weight",
            "connector.proj_gen.0.weight",
            "vit_decoder.net.cond_embed.weight",
            "mask_tokens",
        ];
        for k in planner_keys {
            assert!(
                route_bernini_planner_key(k).is_some(),
                "{k} is a planner tensor"
            );
            assert!(
                route_bernini_expert_key(k).is_none(),
                "{k} must NOT route to a renderer expert"
            );
        }
        // No planner prefix is a prefix of another (first-match routing is unambiguous).
        for (i, (a, _)) in PLANNER_PREFIXES.iter().enumerate() {
            for (j, (b, _)) in PLANNER_PREFIXES.iter().enumerate() {
                if i != j {
                    assert!(!a.starts_with(b), "planner prefix '{b}' shadows '{a}'");
                }
            }
        }
    }

    /// sc-11061 layout contract: the on-disk layout the converter emits is EXACTLY what
    /// [`crate::bernini::BerniniPlanner::load`] reads. This pins the shared path/namespace consts against
    /// the literal strings the loader uses — a rename in either place must update the shared const, and
    /// this test (plus the loader's own use of the same consts) is the tripwire. See the sibling
    /// `planner_layout_consts_match_loader` test in `bernini.rs` which asserts the loader-side paths are
    /// built from these very consts.
    #[test]
    fn planner_layout_consts_are_stable() {
        assert_eq!(PLANNER_MLLM_DIR, "mllm");
        assert_eq!(PLANNER_MLLM_BACKBONE_PP, "model");
        assert_eq!(PLANNER_MLLM_VISION_PP, "visual");
        assert_eq!(PLANNER_CONNECTOR_DIR, "connector");
        assert_eq!(PLANNER_VIT_DECODER_DIR, "vit_decoder");
        assert_eq!(PLANNER_VIT_DECODER_PP, "net");
        assert_eq!(PLANNER_MASK_TOKENS_FILE, "mask_tokens.safetensors");
        assert_eq!(PLANNER_MASK_TOKENS_KEY, "mask_tokens");
        assert_eq!(PLANNER_QWEN_CONFIG_FILE, "qwen2_5_vl_config.json");
        assert_eq!(PLANNER_TOKENIZER_FILE, "tokenizer.json");
        assert_eq!(PLANNER_SIDECAR_FILE, "bernini_planner.json");
        // The routing strips to keys under the loader's `.pp(..)` namespaces.
        let (_, mk) = route_bernini_planner_key("mllm.model.norm.weight").unwrap();
        assert!(mk.starts_with(&format!("{PLANNER_MLLM_BACKBONE_PP}.")));
        let (_, vk) = route_bernini_planner_key("mllm.visual.merger.mlp.0.weight").unwrap();
        assert!(vk.starts_with(&format!("{PLANNER_MLLM_VISION_PP}.")));
        let (_, nk) = route_bernini_planner_key("vit_decoder.net.cond_embed.weight").unwrap();
        assert!(nk.starts_with(&format!("{PLANNER_VIT_DECODER_PP}.")));
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

    /// The exact planner counts match the mlx-gen-bernini `Component::expect` asserts.
    #[test]
    fn planner_expected_counts_match_mlx() {
        assert_eq!(
            PLANNER_EXPECTED_COUNTS,
            [("mllm", 728), ("connector", 12), ("vit_decoder", 140)]
        );
    }

    /// Build a synthetic planner routing group with `count` dummy `(src, stripped)` key pairs under `dir`
    /// (matching the [`RoutingPlan`] shape the count guard now inspects — keys only, no tensor data).
    fn synth_group(
        dir: &'static str,
        count: usize,
    ) -> HashMap<&'static str, Vec<(String, String)>> {
        let mut inner: Vec<(String, String)> = Vec::new();
        for i in 0..count {
            inner.push((format!("w{i}"), format!("w{i}")));
        }
        let mut m: HashMap<&'static str, Vec<(String, String)>> = HashMap::new();
        m.insert(dir, inner);
        m
    }

    /// sc-11061 count guard: the exact-count guard REJECTS a synthetically short component map and
    /// ACCEPTS a full one — naming the offending component in the error.
    #[test]
    fn planner_count_guard_rejects_short_map() {
        // A full planner map (all three at their expected counts) passes.
        let mut full: HashMap<&'static str, Vec<(String, String)>> = HashMap::new();
        for (dir, expect) in PLANNER_EXPECTED_COUNTS {
            full.extend(synth_group(dir, expect));
        }
        assert!(validate_planner_counts(&full).is_ok(), "full map must pass");

        // Drop one tensor from `mllm` → short by one → Err naming `mllm`.
        let mut short = HashMap::new();
        short.extend(synth_group(PLANNER_MLLM_DIR, 727));
        short.extend(synth_group(PLANNER_CONNECTOR_DIR, 12));
        short.extend(synth_group(PLANNER_VIT_DECODER_DIR, 140));
        let err = validate_planner_counts(&short).unwrap_err().to_string();
        assert!(err.contains("mllm/"), "error names the component: {err}");
        assert!(err.contains("727"), "error reports actual count: {err}");

        // A missing component group (count 0) is also rejected.
        let mut missing = HashMap::new();
        missing.extend(synth_group(PLANNER_MLLM_DIR, 728));
        missing.extend(synth_group(PLANNER_CONNECTOR_DIR, 12));
        // no vit_decoder
        assert!(validate_planner_counts(&missing).is_err());
    }

    /// sc-11061 required-sources guard: a missing `mllm/config.json` or `mllm/tokenizer.json` yields a
    /// clear `Err`; both present yields `Ok` with the two source paths.
    #[test]
    fn require_planner_sources_errs_on_missing() {
        let tmp = std::env::temp_dir().join(format!(
            "bernini_req_src_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mllm = tmp.join("mllm");
        std::fs::create_dir_all(&mllm).unwrap();

        // Empty dir → missing config.
        let err = require_planner_sources(&mllm).unwrap_err().to_string();
        assert!(err.contains("config.json"), "names missing config: {err}");

        // config present, tokenizer still missing.
        std::fs::write(mllm.join("config.json"), b"{}").unwrap();
        let err = require_planner_sources(&mllm).unwrap_err().to_string();
        assert!(
            err.contains(PLANNER_TOKENIZER_FILE),
            "names missing tokenizer: {err}"
        );

        // Both present → Ok.
        std::fs::write(mllm.join(PLANNER_TOKENIZER_FILE), b"{}").unwrap();
        let (cfg, tok) = require_planner_sources(&mllm).unwrap();
        assert!(cfg.ends_with("config.json"));
        assert!(tok.ends_with(PLANNER_TOKENIZER_FILE));

        std::fs::remove_dir_all(&tmp).ok();
    }

    /// sc-11631 required base-Wan-sources guard: a base snapshot missing any REQUIRED renderer
    /// component (`text_encoder/` / `vae/` / `tokenizer/`) yields a clear `Err` naming it; a complete
    /// layout passes. Optional diffusers artifacts (`scheduler/` / `model_index.json`) are NOT required.
    #[test]
    fn require_base_wan_sources_errs_on_missing_vae() {
        let tmp = std::env::temp_dir().join(format!(
            "bernini_base_wan_src_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let base = tmp.join("base_wan");
        std::fs::create_dir_all(&base).unwrap();

        // Empty base → missing `text_encoder/` (the first required component).
        let err = require_base_wan_sources(&base).unwrap_err().to_string();
        assert!(
            err.contains("text_encoder"),
            "names the first missing required component: {err}"
        );

        // text_encoder present, `vae/` still missing → Err naming `vae` (the sc-11003 case).
        std::fs::create_dir_all(base.join("text_encoder")).unwrap();
        let err = require_base_wan_sources(&base).unwrap_err().to_string();
        assert!(err.contains("vae"), "names the missing vae: {err}");

        // vae present, `tokenizer/` still missing → Err naming `tokenizer`.
        std::fs::create_dir_all(base.join("vae")).unwrap();
        let err = require_base_wan_sources(&base).unwrap_err().to_string();
        assert!(
            err.contains("tokenizer"),
            "names the missing tokenizer: {err}"
        );

        // All three required components present → Ok, even without the optional diffusers artifacts.
        std::fs::create_dir_all(base.join("tokenizer")).unwrap();
        assert!(
            require_base_wan_sources(&base).is_ok(),
            "complete required layout must pass without scheduler/model_index.json"
        );

        std::fs::remove_dir_all(&tmp).ok();
    }

    // --- sc-11169 / F-099: streaming-converter byte-identity tests ---------------------------------
    //
    // These prove the streamed writers (`write_expert_streamed` / `write_planner_component_streamed`),
    // which load one source tensor at a time from mmap'd shards, produce byte-for-byte the SAME
    // `model.safetensors` as the pre-streaming path that materialized the whole component map first. The
    // reference closures below replicate the OLD whole-map logic exactly (cast the entire map to bf16,
    // then pack — or, for the planner, just bf16). Because safetensors serializes in a canonical
    // (dtype, name) order, insertion order does not matter, so equal tensor sets ⇒ equal bytes.

    /// Unique temp dir for a streaming test.
    fn stream_tmp(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!(
            "bernini_stream_{tag}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn f32_tensor(shape: &[usize], seed: f32) -> Tensor {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n)
            .map(|i| ((i as f32 + seed) * 0.017).sin() * 1.7)
            .collect();
        Tensor::from_vec(data, shape, &Device::Cpu).unwrap()
    }

    /// Reference for the OLD `write_expert`: cast the whole map to bf16, then pack (or dense bf16), save.
    fn ref_write_expert(map: HashMap<String, Tensor>, dst: &Path, bits: usize) -> Result<()> {
        std::fs::create_dir_all(dst)?;
        let bf16: HashMap<String, Tensor> = map
            .into_iter()
            .map(|(k, v)| Ok((k, v.to_dtype(DType::BF16)?)))
            .collect::<Result<_>>()?;
        let (out, _n) = if bits == 0 {
            (bf16, 0)
        } else {
            pack_transformer_component(bf16, bits)?
        };
        cst::save(&out, dst.join("model.safetensors"))?;
        Ok(())
    }

    /// The streamed expert writer is byte-identical to the old whole-map path across bits ∈ {0,4,8},
    /// with the component's tensors split across two shards (proving cross-shard streaming).
    #[test]
    fn streamed_expert_write_is_byte_identical() {
        // A representative mix: a packable rank-2 `.weight`, a rank-1 norm `.weight` (dense — rank ≠ 2),
        // a `.bias`, a rank-2 non-`.weight` (`scale_shift_table`, dense), and a rank-4 conv `.weight`
        // (dense — rank ≠ 2). in_dim 128 is a multiple of the group size so the packer accepts it.
        let tensors: Vec<(&str, Tensor)> = vec![
            ("blocks.0.attn1.to_q.weight", f32_tensor(&[8, 128], 1.0)),
            ("blocks.0.norm1.weight", f32_tensor(&[8], 2.0)),
            ("blocks.0.attn1.to_q.bias", f32_tensor(&[8], 3.0)),
            ("scale_shift_table", f32_tensor(&[2, 128], 4.0)),
            ("patch_embedding.weight", f32_tensor(&[2, 3, 2, 2], 5.0)),
        ];
        for bits in [0usize, 4, 8] {
            let tmp = stream_tmp(&format!("expert_{bits}"));
            // Split the tensors across two shards.
            let mut s0: HashMap<String, Tensor> = HashMap::new();
            let mut s1: HashMap<String, Tensor> = HashMap::new();
            for (i, (k, v)) in tensors.iter().enumerate() {
                if i % 2 == 0 {
                    s0.insert(k.to_string(), v.clone());
                } else {
                    s1.insert(k.to_string(), v.clone());
                }
            }
            let p0 = tmp.join("shard0.safetensors");
            let p1 = tmp.join("shard1.safetensors");
            cst::save(&s0, &p0).unwrap();
            cst::save(&s1, &p1).unwrap();

            // Streamed path: mmap both shards, stream one tensor at a time (src == stripped here).
            let st = unsafe { cst::MmapedSafetensors::multi(&[&p0, &p1]).unwrap() };
            let keys: Vec<(String, String)> = tensors
                .iter()
                .map(|(k, _)| (k.to_string(), k.to_string()))
                .collect();
            let dst_stream = tmp.join("stream");
            write_expert_streamed(&st, &keys, &dst_stream, bits).unwrap();

            // Reference path: load the whole component into one map, then old whole-map write.
            let mut whole: HashMap<String, Tensor> = HashMap::new();
            whole.extend(cst::load(&p0, &Device::Cpu).unwrap());
            whole.extend(cst::load(&p1, &Device::Cpu).unwrap());
            let dst_ref = tmp.join("ref");
            ref_write_expert(whole, &dst_ref, bits).unwrap();

            let a = std::fs::read(dst_stream.join("model.safetensors")).unwrap();
            let b = std::fs::read(dst_ref.join("model.safetensors")).unwrap();
            assert_eq!(
                a, b,
                "streamed expert output must be byte-identical to whole-map (bits={bits})"
            );
            std::fs::remove_dir_all(&tmp).ok();
        }
    }

    /// The streamed planner writer is byte-identical to the old whole-map (dense bf16) path, with the
    /// component split across two shards.
    #[test]
    fn streamed_planner_write_is_byte_identical() {
        let tensors: Vec<(&str, Tensor)> = vec![
            (
                "model.layers.0.self_attn.q_proj.weight",
                f32_tensor(&[4, 4], 1.0),
            ),
            ("model.norm.weight", f32_tensor(&[4], 2.0)),
            ("visual.patch_embed.proj.weight", f32_tensor(&[6, 3], 3.0)),
        ];
        let tmp = stream_tmp("planner");
        let mut s0: HashMap<String, Tensor> = HashMap::new();
        let mut s1: HashMap<String, Tensor> = HashMap::new();
        for (i, (k, v)) in tensors.iter().enumerate() {
            if i % 2 == 0 {
                s0.insert(k.to_string(), v.clone());
            } else {
                s1.insert(k.to_string(), v.clone());
            }
        }
        let p0 = tmp.join("shard0.safetensors");
        let p1 = tmp.join("shard1.safetensors");
        cst::save(&s0, &p0).unwrap();
        cst::save(&s1, &p1).unwrap();

        let st = unsafe { cst::MmapedSafetensors::multi(&[&p0, &p1]).unwrap() };
        let keys: Vec<(String, String)> = tensors
            .iter()
            .map(|(k, _)| (k.to_string(), k.to_string()))
            .collect();
        let dst_stream = tmp.join("stream");
        write_planner_component_streamed(&st, &keys, &dst_stream).unwrap();

        // Reference: whole map cast to bf16 then saved (the old `write_planner_component`).
        let mut whole: HashMap<String, Tensor> = HashMap::new();
        whole.extend(cst::load(&p0, &Device::Cpu).unwrap());
        whole.extend(cst::load(&p1, &Device::Cpu).unwrap());
        let bf16: HashMap<String, Tensor> = whole
            .into_iter()
            .map(|(k, v)| (k, v.to_dtype(DType::BF16).unwrap()))
            .collect();
        let dst_ref = tmp.join("ref");
        std::fs::create_dir_all(&dst_ref).unwrap();
        cst::save(&bf16, dst_ref.join("model.safetensors")).unwrap();

        let a = std::fs::read(dst_stream.join("model.safetensors")).unwrap();
        let b = std::fs::read(dst_ref.join("model.safetensors")).unwrap();
        assert_eq!(
            a, b,
            "streamed planner output must be byte-identical to whole-map"
        );
        std::fs::remove_dir_all(&tmp).ok();
    }

    // --- sc-11062: packed planner (mllm LLM linears) --------------------------------------------

    /// The pack predicate selects EXACTLY the Qwen2.5-VL LLM text linears (attention q/k/v/o + MLP
    /// gate/up/down, any decoder-layer index) and rejects the token embedding, norms, biases, and the
    /// whole vision tower — the same set the loader's packed-detect seam reads.
    #[test]
    fn planner_llm_linear_predicate() {
        for k in [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.27.mlp.gate_proj.weight",
            "model.layers.5.mlp.up_proj.weight",
            "model.layers.13.mlp.down_proj.weight",
        ] {
            assert!(is_planner_llm_linear(k), "{k} must pack");
        }
        for k in [
            "model.embed_tokens.weight",
            "model.norm.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.0.self_attn.q_proj.bias", // bias stays dense
            "visual.blocks.0.attn.qkv.weight",      // vision tower stays dense
            "visual.patch_embed.proj.weight",
            "visual.merger.mlp.0.weight",
            "model.layers.x.self_attn.q_proj.weight", // non-numeric layer index
            "model.layers.0.self_attn.q_proj",        // no .weight suffix
        ] {
            assert!(!is_planner_llm_linear(k), "{k} must stay dense");
        }
    }

    /// The `mllm/` writer packs ONLY the LLM text linears at q4 (u32 codes + `.scales`/`.biases`, group
    /// 64), keeps the dense `.bias` alongside, and leaves the token embedding / norms / vision tower dense
    /// bf16 — emitting a `quantize_config.json`. `bits == 0` packs nothing and writes no config.
    #[test]
    fn planner_mllm_packs_only_llm_linears() {
        // in-dims are group-64 multiples (64 / 128) so the packer accepts the LLM linears.
        let tensors: Vec<(&str, Tensor)> = vec![
            ("model.embed_tokens.weight", f32_tensor(&[64, 64], 1.0)),
            ("model.norm.weight", f32_tensor(&[64], 2.0)),
            (
                "model.layers.0.input_layernorm.weight",
                f32_tensor(&[64], 3.0),
            ),
            (
                "model.layers.0.self_attn.q_proj.weight",
                f32_tensor(&[64, 64], 4.0),
            ),
            (
                "model.layers.0.self_attn.q_proj.bias",
                f32_tensor(&[64], 5.0),
            ),
            (
                "model.layers.0.mlp.gate_proj.weight",
                f32_tensor(&[128, 64], 6.0),
            ),
            (
                "model.layers.0.mlp.down_proj.weight",
                f32_tensor(&[64, 128], 7.0),
            ),
            (
                "visual.blocks.0.attn.qkv.weight",
                f32_tensor(&[192, 64], 8.0),
            ),
        ];
        let tmp = stream_tmp("mllm_pack");
        let p0 = tmp.join("shard0.safetensors");
        let mut s0: HashMap<String, Tensor> = HashMap::new();
        for (k, v) in &tensors {
            s0.insert(k.to_string(), v.clone());
        }
        cst::save(&s0, &p0).unwrap();
        let st = unsafe { cst::MmapedSafetensors::multi(&[&p0]).unwrap() };
        let keys: Vec<(String, String)> = tensors
            .iter()
            .map(|(k, _)| (k.to_string(), k.to_string()))
            .collect();

        // bits = 4: LLM linears pack, the rest stays dense.
        let dst = tmp.join("out");
        let packed = write_planner_mllm_streamed(&st, &keys, &dst, 4).unwrap();
        assert_eq!(packed, 3, "q_proj + gate_proj + down_proj pack");
        let loaded = cst::load(dst.join("model.safetensors"), &Device::Cpu).unwrap();
        for base in [
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.mlp.gate_proj",
            "model.layers.0.mlp.down_proj",
        ] {
            assert!(
                loaded.contains_key(&format!("{base}.scales")),
                "{base}.scales present"
            );
            assert!(
                loaded.contains_key(&format!("{base}.biases")),
                "{base}.biases present"
            );
            assert_eq!(
                loaded[&format!("{base}.weight")].dtype(),
                DType::U32,
                "{base}.weight is u32 codes"
            );
        }
        assert_eq!(
            loaded["model.layers.0.self_attn.q_proj.bias"].dtype(),
            DType::BF16,
            "packed linear's bias stays dense bf16"
        );
        for k in [
            "model.embed_tokens.weight",
            "model.norm.weight",
            "model.layers.0.input_layernorm.weight",
            "visual.blocks.0.attn.qkv.weight",
        ] {
            assert!(
                !loaded.contains_key(&k.replace(".weight", ".scales")),
                "{k} must stay dense (no .scales)"
            );
            assert_eq!(loaded[k].dtype(), DType::BF16, "{k} dense bf16");
        }
        assert!(
            dst.join("quantize_config.json").exists(),
            "quantize_config.json emitted for the packed mllm component"
        );

        // bits = 0: fully dense, no config, no packed triple anywhere.
        let dst0 = tmp.join("out0");
        let packed0 = write_planner_mllm_streamed(&st, &keys, &dst0, 0).unwrap();
        assert_eq!(packed0, 0);
        let loaded0 = cst::load(dst0.join("model.safetensors"), &Device::Cpu).unwrap();
        assert!(
            !loaded0.keys().any(|k| k.ends_with(".scales")),
            "bits=0 packs nothing"
        );
        assert!(
            !dst0.join("quantize_config.json").exists(),
            "bits=0 writes no quantize_config"
        );

        std::fs::remove_dir_all(&tmp).ok();
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
