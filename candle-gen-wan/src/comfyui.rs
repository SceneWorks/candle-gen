//! ComfyUI single-file Wan2.2 expert → candle in-memory remap+dequant seam (epic 10451 Phase 2c,
//! sc-10671).
//!
//! SceneWorks lets a user point at an existing ComfyUI `models/` tree and generate from the weights in
//! place — no copy, no re-download (Phase 1 did LoRAs; sc-10668 bf16 Z-Image; sc-10670 plain-fp8
//! Qwen-Image). This is the Wan2.2 A14B slice. Wan2.2 ships as a **dual-expert MoE**: two separate DiT
//! files, the **high-noise** and **low-noise** experts (`unet/wan2.2_{t2v,i2v}_{high,low}_noise_14B_*`).
//!
//! Two transforms make an in-place expert loadable via [`candle_nn::VarBuilder::from_tensors`]:
//!
//! 1. **Scaled-fp8 dequant.** The ComfyUI file is the *companion* scaled-fp8 convention: each quantized
//!    Linear weight is `F8_E4M3` with a per-tensor scalar `.scale_weight` sibling (and, on the stock
//!    ComfyUI export, a `.scale_input` sibling + a top-level `scaled_fp8` marker tensor). The real
//!    weight is `w = w_fp8·scale_weight`; `scale_input` is the *activation* quant scale, only used by
//!    ComfyUI's fp8×fp8 matmul — irrelevant when we dequant to bf16 and run a normal bf16 matmul, so it
//!    is dropped (same posture as candle-gen-ideogram's fp8 convert). The Kijai variant
//!    (`Wan2_2-*_fp8_e4m3fn_scaled_KJ`) carries `.scale_weight` only (no `.scale_input`); the same code
//!    handles it. Non-quantized tensors (norms, biases, `modulation`, `patch_embedding`) stay dense and
//!    are cast to the compute dtype.
//! 2. **Native-Wan → diffusers key remap.** The ComfyUI file uses the **native Wan** tensor names
//!    (`blocks.N.self_attn.q`, `cross_attn`, `ffn.0/2`, `modulation`, `norm3`, `head.head`,
//!    `text_embedding.0/2`, `time_projection.1`); candle's [`crate::transformer::WanTransformer`] reads
//!    the **diffusers** schema (`blocks.N.attn1/attn2`, `ffn.net.0.proj/net.2`, `scale_shift_table`,
//!    `norm2`, `proj_out`, `condition_embedder.*`). This is the same native→diffusers mapping diffusers'
//!    own `convert_wan_to_diffusers.py` applies. Wan2.2 A14B is channel-concat (no I2V image
//!    cross-attention), so T2V and I2V experts share the key schema — only `patch_embedding` in-channels
//!    (16 vs 36) differ, which the config carries, not the remap.
//!
//! The UMT5 text encoder and the Wan VAE are ALSO read from the ComfyUI tree when present (epic 10451
//! Phase 2c follow-up, sc-10909), leaving only the tiny tokenizer snapshot-sourced:
//!
//! * **UMT5-XXL TE** (`text_encoders/umt5_xxl_fp8_e4m3fn_scaled`) is itself the *same* companion
//!   scaled-fp8 convention as the DiT experts, so it reuses this module's dequant
//!   ([`dequant_comfyui_umt5`]) — but with **no** key remap: the ComfyUI file already carries the HF
//!   `UMT5EncoderModel` keys candle's [`crate::text_encoder::Umt5Encoder`] reads.
//! * **Wan VAE** (`vae/wan_2.1_vae.safetensors`) is the *same physical* Wan2.1 16-channel VAE the
//!   Qwen-Image lane reads (sc-10830); its native-key layout is remapped to the diffusers schema
//!   [`crate::vae16::WanVae16`] reads by the shared core-crate [`remap_vae_wan_to_diffusers`].
//!
//! When a TE or VAE file is absent the component falls back to a resident `SceneWorks/wan2.2-*-candle`
//! snapshot tier (the sc-10671 posture); with both present a Wan generation is sourced entirely from
//! the user's ComfyUI tree plus our tiny tokenizer. The two big DiT experts are always read in place.
//!
//! Header-only classification (which file is a Wan DiT, which quant) is done upstream by SceneWorks
//! (`sceneworks-core::base_weights`, sc-10662); this module is handed files already identified as Wan
//! scaled-fp8 experts.

use std::collections::HashMap;
use std::path::PathBuf;

use candle_gen::candle_core::{DType, Tensor};
use candle_gen::{CandleError, Result};

/// The in-place ComfyUI files for a Wan2.2 A14B load. The two DiT experts are always read in place
/// (sc-10671); the UMT5 text encoder and the Wan VAE are read in place too when present (sc-10909),
/// else they fall back to the resident snapshot tier. Read in place, never copied; each is remapped +
/// dequant'd/upcast in memory at component build ([`remap_and_dequant_comfyui_expert`],
/// [`dequant_comfyui_umt5`], [`remap_vae_wan_to_diffusers`]).
#[derive(Clone, Debug)]
pub(crate) struct ComfyuiExperts {
    /// The **high-noise** expert (ComfyUI `*_high_noise_*` file) → candle `transformer/`.
    pub high_file: PathBuf,
    /// The **low-noise** expert (ComfyUI `*_low_noise_*` file) → candle `transformer_2/`.
    pub low_file: PathBuf,
    /// The UMT5-XXL text encoder (`text_encoders/umt5_xxl_fp8_e4m3fn_scaled`, companion scaled-fp8).
    /// `Some` ⇒ read in place ([`dequant_comfyui_umt5`]); `None` ⇒ the snapshot `text_encoder/` dir.
    pub te_file: Option<PathBuf>,
    /// The Wan VAE (`vae/wan_2.1_vae.safetensors`, native WAN-VAE keys, bf16). `Some` ⇒ read in place
    /// ([`remap_vae_wan_to_diffusers`]); `None` ⇒ the snapshot `vae/` dir.
    pub vae_file: Option<PathBuf>,
}

/// Native WAN-VAE → diffusers key remap for the in-place VAE (sc-10909). The ComfyUI
/// `vae/wan_2.1_vae.safetensors` is the *same physical* Wan2.1 16-channel VAE the Qwen-Image lane
/// reads (sc-10830) — its native-key layout is remapped to the diffusers schema
/// [`crate::vae16::WanVae16`] reads by the shared core-crate helper (a pure key rename; values pass
/// through as bf16, upcast to the VAE compute dtype at `VarBuilder` build). Re-exported so this
/// module's in-place seam names it locally alongside the DiT + UMT5 transforms.
pub use candle_gen::comfyui_vae::remap_vae_wan_to_diffusers;

/// The top-level empty marker tensor ComfyUI writes to flag a scaled-fp8 checkpoint; dropped.
const SCALED_FP8_MARKER: &str = "scaled_fp8";

/// Remap + dequant a ComfyUI Wan2.2 expert tensor map into the schema
/// [`crate::transformer::WanTransformer`] reads, ready for `VarBuilder::from_tensors`. Native-Wan DiT
/// keys → the diffusers keys candle reads ([`remap_wan_key`]); see [`dequant_scaled_fp8_map`] for the
/// dequant.
///
/// Errors when **no** `.scale_weight` companion is present — a file that is not a ComfyUI scaled-fp8
/// Wan expert (wrong file/family/quant), surfaced rather than loaded as garbage (no silent fallback).
pub fn remap_and_dequant_comfyui_expert(
    src: HashMap<String, Tensor>,
    dtype: DType,
) -> Result<HashMap<String, Tensor>> {
    dequant_scaled_fp8_map(src, dtype, "wan ComfyUI expert", remap_wan_key)
}

/// Dequant a ComfyUI UMT5-XXL text encoder tensor map (`umt5_xxl_fp8_e4m3fn_scaled`, the SAME companion
/// scaled-fp8 convention as the DiT experts, sc-10909) to the schema [`crate::text_encoder::Umt5Encoder`]
/// reads, ready for `VarBuilder::from_tensors`. The ComfyUI file already carries the HF
/// `UMT5EncoderModel` keys candle reads (`shared.weight`, `encoder.block.N.layer.{0,1}.…`,
/// `encoder.final_layer_norm.weight`) — so, unlike the DiT, there is **no** key remap, only the
/// scaled-fp8 dequant (the embedding / norms / `relative_attention_bias` are dense and cast; the fp8
/// Linears are `w = w_fp8·scale_weight`). `dtype` is the encoder compute dtype (f32).
///
/// Errors when **no** `.scale_weight` companion is present (not a scaled-fp8 UMT5 checkpoint).
pub fn dequant_comfyui_umt5(
    src: HashMap<String, Tensor>,
    dtype: DType,
) -> Result<HashMap<String, Tensor>> {
    // UMT5 keys already match candle's schema — identity remap, dequant only.
    dequant_scaled_fp8_map(src, dtype, "wan ComfyUI UMT5", |k| k.to_owned())
}

/// Shared companion-scaled-fp8 dequant over a ComfyUI tensor map, applying `remap` to each surviving
/// key. `what` labels error messages (`"wan ComfyUI expert"` / `"wan ComfyUI UMT5"`).
///
/// - **Dequant**: an `F8_E4M3` `{module}.weight` with a scalar `{module}.scale_weight` sibling becomes
///   `w = (w_fp8·scale_weight)` cast to `dtype`. `.scale_input` companions and the `scaled_fp8` marker
///   are consumed/dropped. Non-quantized tensors are cast to `dtype` (integer buffers pass through).
/// - **Remap**: `remap(key)` — the DiT native→diffusers rename, or identity for the already-diffusers
///   UMT5 keys.
///
/// Errors when **no** `.scale_weight` companion is present — a file that is not a ComfyUI scaled-fp8
/// checkpoint (wrong file/family/quant), surfaced rather than loaded as garbage (no silent fallback).
fn dequant_scaled_fp8_map(
    src: HashMap<String, Tensor>,
    dtype: DType,
    what: &str,
    remap: impl Fn(&str) -> String,
) -> Result<HashMap<String, Tensor>> {
    // Pass 1: index the per-module `.scale_weight` scalars (base = key without the `.scale_weight`
    // suffix, e.g. `blocks.0.self_attn.q`). A weight `{base}.weight` dequants against `scales[base]`.
    // The scalars (shape `[]`) are cloned out — a cheap Arc bump on tiny storage — so pass 2 can *drain*
    // `src` without an outstanding borrow, letting each fp8 source drop as its bf16 twin is inserted.
    let mut scales: HashMap<String, Tensor> = HashMap::new();
    for (key, tensor) in &src {
        if let Some(base) = key.strip_suffix(".scale_weight") {
            scales.insert(base.to_owned(), tensor.clone());
        }
    }
    if scales.is_empty() {
        return Err(CandleError::Msg(format!(
            "{what}: no `.scale_weight` companions found — not a ComfyUI scaled-fp8 checkpoint \
             (wrong file/family/quant?)"
        )));
    }

    // Pass 2: consume `src` by value (`into_iter`) so each source tensor drops at the end of its
    // iteration rather than staying resident for the whole map — the fp8 source (~14 GB per expert)
    // never sits alongside the *entire* bf16 output (~28 GB), only the single tensor being converted.
    let mut out = HashMap::with_capacity(src.len());
    for (key, tensor) in src {
        // Drop the marker, the rope `freqs` buffer (candle recomputes it; absent on UMT5), and both
        // scale companions.
        if key == SCALED_FP8_MARKER || key == "freqs" || key.ends_with(".freqs") {
            continue;
        }
        if key.ends_with(".scale_weight") || key.ends_with(".scale_input") {
            continue;
        }
        // Dequant an fp8 Linear weight that has a scale companion; otherwise a plain cast.
        let value = match key
            .strip_suffix(".weight")
            .and_then(|base| scales.get(base))
        {
            Some(scale) => dequant_scaled_fp8(what, &key, &tensor, scale, dtype)?,
            None => cast_dense(what, &key, &tensor, dtype)?,
        };
        // Free the fp8 source before the (possibly reallocating) insert, so peak host RAM tracks the
        // shrinking source + growing output rather than both maps in full.
        drop(tensor);
        out.insert(remap(&key), value);
    }
    Ok(out)
}

/// `w = (w_fp8 → f32 · scale_weight) → dtype`. `scale_weight` is a per-tensor **scalar** (shape `[]`),
/// applied via `affine` (multiply-by-f64) — reconstructing the real weight ComfyUI stored as fp8. The
/// f32 intermediate is per-tensor (immediately downcast), so it never accumulates: the only f32 buffer
/// resident is the one weight mid-conversion. (Map-level peak — the shrinking fp8 source vs. the growing
/// bf16 output — is bounded by [`dequant_scaled_fp8_map`]'s drain, not by this f32 step.)
fn dequant_scaled_fp8(
    what: &str,
    key: &str,
    w: &Tensor,
    scale: &Tensor,
    dtype: DType,
) -> Result<Tensor> {
    let s = scale
        .to_dtype(DType::F32)?
        .to_scalar::<f32>()
        .map_err(|e| {
            CandleError::Msg(format!(
                "{what}: scale_weight for {key:?} is not a scalar: {e}"
            ))
        })?;
    w.to_dtype(DType::F32)?
        .affine(s as f64, 0.0)?
        .to_dtype(dtype)
        .map_err(|e| CandleError::Msg(format!("{what}: dequant {key:?} failed: {e}")))
}

/// Cast a non-quantized tensor to the compute `dtype`; integer buffers (indices/caches) pass through.
fn cast_dense(what: &str, key: &str, t: &Tensor, dtype: DType) -> Result<Tensor> {
    if t.dtype().is_int() {
        return Ok(t.clone());
    }
    t.to_dtype(dtype)
        .map_err(|e| CandleError::Msg(format!("{what}: cast {key:?} failed: {e}")))
}

/// Native-Wan → diffusers key rename ([`crate::transformer::WanTransformer`]'s schema). Every rule is
/// exact/prefix/segment-scoped so no two collide; a key matching no rule (e.g. `patch_embedding.*`)
/// passes through unchanged.
fn remap_wan_key(key: &str) -> String {
    // --- top-level head ---
    if let Some(rest) = key.strip_prefix("head.head.") {
        return format!("proj_out.{rest}"); // native head Linear → diffusers proj_out
    }
    if key == "head.modulation" {
        return "scale_shift_table".to_owned(); // top-level [1,2,dim] head modulation
    }
    // --- top-level condition embedders ---
    const TOP: &[(&str, &str)] = &[
        (
            "text_embedding.0.",
            "condition_embedder.text_embedder.linear_1.",
        ),
        (
            "text_embedding.2.",
            "condition_embedder.text_embedder.linear_2.",
        ),
        (
            "time_embedding.0.",
            "condition_embedder.time_embedder.linear_1.",
        ),
        (
            "time_embedding.2.",
            "condition_embedder.time_embedder.linear_2.",
        ),
        ("time_projection.1.", "condition_embedder.time_proj."),
    ];
    for (from, to) in TOP {
        if let Some(rest) = key.strip_prefix(from) {
            return format!("{to}{rest}");
        }
    }
    // --- per-block ---
    if key.starts_with("blocks.") {
        // Block modulation `[1,6,dim]` → per-block scale_shift_table (exact suffix).
        if let Some(prefix) = key.strip_suffix(".modulation") {
            return format!("{prefix}.scale_shift_table");
        }
        // Segment renames inside a block. Each block key matches at most one (all mutually exclusive),
        // so a single pass suffices; norm_q/norm_k precede q/k so the shorter pattern can't shadow them.
        const BLOCK: &[(&str, &str)] = &[
            (".self_attn.norm_q.", ".attn1.norm_q."),
            (".self_attn.norm_k.", ".attn1.norm_k."),
            (".self_attn.q.", ".attn1.to_q."),
            (".self_attn.k.", ".attn1.to_k."),
            (".self_attn.v.", ".attn1.to_v."),
            (".self_attn.o.", ".attn1.to_out.0."),
            (".cross_attn.norm_q.", ".attn2.norm_q."),
            (".cross_attn.norm_k.", ".attn2.norm_k."),
            (".cross_attn.q.", ".attn2.to_q."),
            (".cross_attn.k.", ".attn2.to_k."),
            (".cross_attn.v.", ".attn2.to_v."),
            (".cross_attn.o.", ".attn2.to_out.0."),
            (".ffn.0.", ".ffn.net.0.proj."),
            (".ffn.2.", ".ffn.net.2."),
            (".norm3.", ".norm2."),
        ];
        for (from, to) in BLOCK {
            if let Some(idx) = key.find(from) {
                return format!("{}{to}{}", &key[..idx], &key[idx + from.len()..]);
            }
        }
    }
    // patch_embedding.* and anything else already match candle's schema.
    key.to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::{Device, Tensor};

    fn scalar(v: f32) -> Tensor {
        Tensor::new(v, &Device::Cpu).unwrap()
    }
    fn w(dims: &[usize], dtype: DType) -> Tensor {
        Tensor::ones(dims, DType::F32, &Device::Cpu)
            .unwrap()
            .to_dtype(dtype)
            .unwrap()
    }

    // --- key remap ---

    #[test]
    fn remaps_self_and_cross_attn() {
        assert_eq!(
            remap_wan_key("blocks.3.self_attn.q.weight"),
            "blocks.3.attn1.to_q.weight"
        );
        assert_eq!(
            remap_wan_key("blocks.3.self_attn.o.bias"),
            "blocks.3.attn1.to_out.0.bias"
        );
        assert_eq!(
            remap_wan_key("blocks.3.self_attn.norm_q.weight"),
            "blocks.3.attn1.norm_q.weight"
        );
        assert_eq!(
            remap_wan_key("blocks.3.cross_attn.k.weight"),
            "blocks.3.attn2.to_k.weight"
        );
        assert_eq!(
            remap_wan_key("blocks.3.cross_attn.norm_k.weight"),
            "blocks.3.attn2.norm_k.weight"
        );
    }

    #[test]
    fn remaps_ffn_norm_and_modulation() {
        assert_eq!(
            remap_wan_key("blocks.0.ffn.0.weight"),
            "blocks.0.ffn.net.0.proj.weight"
        );
        assert_eq!(
            remap_wan_key("blocks.0.ffn.2.bias"),
            "blocks.0.ffn.net.2.bias"
        );
        assert_eq!(
            remap_wan_key("blocks.0.norm3.weight"),
            "blocks.0.norm2.weight"
        );
        assert_eq!(
            remap_wan_key("blocks.0.modulation"),
            "blocks.0.scale_shift_table"
        );
    }

    #[test]
    fn remaps_top_level_embedders_and_head() {
        assert_eq!(
            remap_wan_key("text_embedding.0.weight"),
            "condition_embedder.text_embedder.linear_1.weight"
        );
        assert_eq!(
            remap_wan_key("time_embedding.2.bias"),
            "condition_embedder.time_embedder.linear_2.bias"
        );
        assert_eq!(
            remap_wan_key("time_projection.1.weight"),
            "condition_embedder.time_proj.weight"
        );
        assert_eq!(remap_wan_key("head.head.weight"), "proj_out.weight");
        assert_eq!(remap_wan_key("head.modulation"), "scale_shift_table");
    }

    #[test]
    fn passes_patch_embedding_unchanged() {
        assert_eq!(
            remap_wan_key("patch_embedding.weight"),
            "patch_embedding.weight"
        );
    }

    // --- dequant ---

    #[test]
    fn dequants_fp8_weight_by_scale_and_remaps_key() {
        let mut src = HashMap::new();
        // A fp8 weight of all-ones with scale 3.0 → dequant to 3.0, remapped to attn1.to_q.
        src.insert(
            "blocks.0.self_attn.q.weight".to_string(),
            w(&[4, 4], DType::F8E4M3),
        );
        src.insert("blocks.0.self_attn.q.scale_weight".to_string(), scalar(3.0));
        src.insert("blocks.0.self_attn.q.scale_input".to_string(), scalar(0.5));
        let out = remap_and_dequant_comfyui_expert(src, DType::BF16).unwrap();
        let t = out
            .get("blocks.0.attn1.to_q.weight")
            .expect("remapped weight");
        assert_eq!(t.dtype(), DType::BF16);
        let v = t
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(
            (v[0] - 3.0).abs() < 0.05,
            "1.0 * scale 3.0 = 3.0, got {}",
            v[0]
        );
        // Companions + were consumed, not emitted.
        assert!(!out.contains_key("blocks.0.attn1.to_q.scale_weight"));
        assert!(!out.contains_key("blocks.0.self_attn.q.scale_input"));
    }

    #[test]
    fn drops_marker_and_casts_dense_tensors() {
        let mut src = HashMap::new();
        src.insert(
            "blocks.0.self_attn.q.weight".to_string(),
            w(&[2, 2], DType::F8E4M3),
        );
        src.insert("blocks.0.self_attn.q.scale_weight".to_string(), scalar(1.0));
        src.insert(SCALED_FP8_MARKER.to_string(), w(&[0], DType::F8E4M3));
        src.insert("blocks.0.norm3.weight".to_string(), w(&[2], DType::F16)); // dense, no scale
        let out = remap_and_dequant_comfyui_expert(src, DType::BF16).unwrap();
        assert!(!out.contains_key(SCALED_FP8_MARKER));
        // Dense F16 norm cast to bf16 + remapped norm3 → norm2.
        assert_eq!(
            out.get("blocks.0.norm2.weight").unwrap().dtype(),
            DType::BF16
        );
    }

    #[test]
    fn drain_preserves_all_entries_across_many_weights() {
        // Guards the sc-11222 F-122 drain: consuming `src` by value (so each fp8 source drops as its
        // bf16 twin is inserted) must still emit every dequanted weight with its own scale — no entry
        // lost, each key dequanted against its *own* companion.
        let mut src = HashMap::new();
        for (i, s) in [(0usize, 2.0f32), (1, 3.0), (2, 4.0)] {
            src.insert(
                format!("blocks.{i}.self_attn.q.weight"),
                w(&[2, 2], DType::F8E4M3),
            );
            src.insert(format!("blocks.{i}.self_attn.q.scale_weight"), scalar(s));
        }
        let out = remap_and_dequant_comfyui_expert(src, DType::BF16).unwrap();
        for (i, s) in [(0usize, 2.0f32), (1, 3.0), (2, 4.0)] {
            let t = out
                .get(&format!("blocks.{i}.attn1.to_q.weight"))
                .unwrap_or_else(|| panic!("weight {i} survived the drain"));
            let v = t
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert!(
                (v[0] - s).abs() < 0.05,
                "block {i}: 1.0 * scale {s} = {s}, got {}",
                v[0]
            );
        }
    }

    #[test]
    fn kijai_variant_scale_weight_only() {
        // Kijai carries scale_weight but NO scale_input — must still dequant.
        let mut src = HashMap::new();
        src.insert("head.head.weight".to_string(), w(&[4, 4], DType::F8E4M3));
        src.insert("head.head.scale_weight".to_string(), scalar(2.0));
        let out = remap_and_dequant_comfyui_expert(src, DType::BF16).unwrap();
        assert!(out.contains_key("proj_out.weight"));
    }

    #[test]
    fn rejects_a_map_with_no_scale_companions() {
        let mut src = HashMap::new();
        src.insert(
            "blocks.0.self_attn.q.weight".to_string(),
            w(&[2, 2], DType::BF16),
        );
        assert!(remap_and_dequant_comfyui_expert(src, DType::BF16).is_err());
    }

    // --- UMT5 dequant (sc-10909): same scaled-fp8, but identity key remap ---

    #[test]
    fn umt5_dequants_fp8_linear_without_remapping_keys() {
        let mut src = HashMap::new();
        // A UMT5 SelfAttention Linear: fp8 ones × scale 2.0 → 2.0, key unchanged (no DiT remap).
        src.insert(
            "encoder.block.0.layer.0.SelfAttention.q.weight".to_string(),
            w(&[4, 4], DType::F8E4M3),
        );
        src.insert(
            "encoder.block.0.layer.0.SelfAttention.q.scale_weight".to_string(),
            scalar(2.0),
        );
        src.insert(
            "encoder.block.0.layer.0.SelfAttention.q.scale_input".to_string(),
            scalar(0.5),
        );
        // Dense tensors (embedding, norm, relative bias) cast to the compute dtype, keys unchanged.
        src.insert("shared.weight".to_string(), w(&[8, 4], DType::F16));
        src.insert(
            "encoder.block.0.layer.0.layer_norm.weight".to_string(),
            w(&[4], DType::F16),
        );
        src.insert(SCALED_FP8_MARKER.to_string(), w(&[0], DType::F8E4M3));
        let out = dequant_comfyui_umt5(src, DType::F32).unwrap();

        // The T5 key passes through verbatim (a DiT-style `self_attn.q`→`attn1.to_q` remap would break
        // Umt5Encoder's VarBuilder).
        let t = out
            .get("encoder.block.0.layer.0.SelfAttention.q.weight")
            .expect("umt5 Linear key unchanged");
        assert_eq!(t.dtype(), DType::F32);
        let v = t
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(
            (v[0] - 2.0).abs() < 0.05,
            "1.0 * scale 2.0 = 2.0, got {}",
            v[0]
        );
        // Dense tensors kept + cast; marker + scale companions dropped.
        assert_eq!(out.get("shared.weight").unwrap().dtype(), DType::F32);
        assert!(out.contains_key("encoder.block.0.layer.0.layer_norm.weight"));
        assert!(!out.contains_key(SCALED_FP8_MARKER));
        assert!(!out.contains_key("encoder.block.0.layer.0.SelfAttention.q.scale_weight"));
        assert!(!out.contains_key("encoder.block.0.layer.0.SelfAttention.q.scale_input"));
    }

    #[test]
    fn umt5_rejects_a_map_with_no_scale_companions() {
        let mut src = HashMap::new();
        src.insert("shared.weight".to_string(), w(&[8, 4], DType::BF16));
        assert!(dequant_comfyui_umt5(src, DType::F32).is_err());
    }
}
