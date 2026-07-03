//! Krea 2 transformer **architecture validation** — prove the on-disk tensor set exactly matches the
//! architecture implied by [`Krea2Config`] before the DiT forward trusts it. Port of `mlx-gen-krea`'s
//! `convert.rs` validation half (the Q4/Q8 turnkey assembly is the worker-wiring story sc-7581).
//!
//! The published `krea/Krea-2-Turbo` diffusers checkpoint uses dotted keys that map 1:1 onto the
//! `Krea2Transformer2DModel` module tree, so [`crate::loader::Weights::from_dir`] loads them directly
//! — there is no key remap. [`validate_transformer`] catches a wrong variant / truncated download /
//! config-weight mismatch loudly at load instead of as garbage latents.

use std::collections::BTreeSet;

use candle_gen::candle_core::Result;

use crate::config::Krea2Config;
use crate::loader::Weights;

/// GQA / full attention Linear weights: `to_q/to_k/to_v/to_gate/to_out.0` + per-head `norm_q/norm_k`.
fn attn_keys(prefix: &str) -> Vec<String> {
    [
        "norm_q", "norm_k", "to_q", "to_k", "to_v", "to_gate", "to_out.0",
    ]
    .iter()
    .map(|p| format!("{prefix}.{p}.weight"))
    .collect()
}

/// SwiGLU feed-forward (`gate`/`up` in, `down` out), all bias-free.
fn ff_keys(prefix: &str) -> Vec<String> {
    ["gate", "up", "down"]
        .iter()
        .map(|p| format!("{prefix}.{p}.weight"))
        .collect()
}

/// A text-fusion block (`layerwise_blocks` / `refiner_blocks`): RMSNorm-attn(+gate)-RMSNorm-SwiGLU,
/// no per-block modulation table.
fn text_block_keys(prefix: &str) -> Vec<String> {
    let mut k = attn_keys(&format!("{prefix}.attn"));
    k.extend(ff_keys(&format!("{prefix}.ff")));
    k.push(format!("{prefix}.norm1.weight"));
    k.push(format!("{prefix}.norm2.weight"));
    k
}

/// A single-stream `transformer_block`: a text-fusion-shaped block plus the per-block 6-factor
/// `scale_shift_table` (the `DoubleSharedModulation` offset added to the shared `time_mod_proj`).
fn single_block_keys(prefix: &str) -> Vec<String> {
    let mut k = text_block_keys(prefix);
    k.push(format!("{prefix}.scale_shift_table"));
    k
}

/// The complete set of transformer tensor keys implied by `cfg` (= the published 430 for Turbo/Raw).
pub fn expected_transformer_keys(cfg: &Krea2Config) -> Vec<String> {
    let mut keys = Vec::new();

    // Image patch embed.
    keys.push("img_in.weight".into());
    keys.push("img_in.bias".into());

    // Text input projection: RMSNorm(text) → Linear(text→hidden) → Linear(hidden→hidden).
    keys.push("txt_in.norm.weight".into());
    for n in ["linear_1", "linear_2"] {
        keys.push(format!("txt_in.{n}.weight"));
        keys.push(format!("txt_in.{n}.bias"));
    }

    // Timestep embed + the shared 6-factor modulation projection.
    for n in ["linear_1", "linear_2"] {
        keys.push(format!("time_embed.{n}.weight"));
        keys.push(format!("time_embed.{n}.bias"));
    }
    keys.push("time_mod_proj.weight".into());
    keys.push("time_mod_proj.bias".into());

    // text_fusion: layerwise (cross-layer-axis aggregator) → projector(12→1) → refiner (token-axis).
    for i in 0..cfg.num_layerwise_text_blocks {
        keys.extend(text_block_keys(&format!(
            "text_fusion.layerwise_blocks.{i}"
        )));
    }
    keys.push("text_fusion.projector.weight".into());
    for i in 0..cfg.num_refiner_text_blocks {
        keys.extend(text_block_keys(&format!("text_fusion.refiner_blocks.{i}")));
    }

    // The single-stream stack.
    for i in 0..cfg.num_layers {
        keys.extend(single_block_keys(&format!("transformer_blocks.{i}")));
    }

    // Continuous-AdaLN output (2-factor scale/shift table).
    keys.push("final_layer.linear.weight".into());
    keys.push("final_layer.linear.bias".into());
    keys.push("final_layer.norm.weight".into());
    keys.push("final_layer.scale_shift_table".into());

    keys
}

/// Validate a loaded transformer against `cfg`: exact key coverage (no missing, no extra) and the
/// shapes of the dimension-bearing entry points.
///
/// **INT8-ConvRot (sc-9300):** a ConvRot checkpoint is native-mmdit-keyed, so the exact diffusers
/// key-set diff would spuriously report every key missing + every native key extra. Instead validate
/// that each expected diffusers key **resolves** to a present native tensor (via the loader's
/// diffusers→native remap, which `w.contains` applies), then run the same shape checks (which also
/// resolve). This proves the ConvRot file covers the full `Krea2Transformer2DModel` surface without
/// asserting a 1:1 native key match.
pub fn validate_transformer(w: &Weights, cfg: &Krea2Config) -> Result<()> {
    if w.is_convrot() {
        let missing: Vec<String> = expected_transformer_keys(cfg)
            .into_iter()
            .filter(|k| !w.contains(k))
            .collect();
        if !missing.is_empty() {
            let head = missing
                .iter()
                .take(8)
                .map(String::as_str)
                .collect::<Vec<_>>()
                .join(", ");
            return Err(candle_gen::candle_core::Error::Msg(format!(
                "krea INT8-ConvRot: {} expected key(s) do not resolve to a native tensor [{head}]",
                missing.len(),
            )));
        }
        return validate_shapes(w, cfg);
    }
    let expected: BTreeSet<String> = expected_transformer_keys(cfg).into_iter().collect();
    // A pre-quantized snapshot replaces each Linear `{base}.weight` with the packed triple
    // `{base}.weight` + `{base}.scales` + `{base}.biases`; drop the two quant-only artifacts before
    // the coverage diff (the `{base}.weight` key is still present).
    let actual: BTreeSet<String> = w
        .keys()
        .into_iter()
        .filter(|k| !k.ends_with(".scales") && !k.ends_with(".biases"))
        .collect();

    let missing: Vec<&String> = expected.difference(&actual).collect();
    let extra: Vec<&String> = actual.difference(&expected).collect();
    if !missing.is_empty() || !extra.is_empty() {
        let head = |v: &[&String]| {
            v.iter()
                .take(8)
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        };
        return Err(candle_gen::candle_core::Error::Msg(format!(
            "krea transformer key mismatch vs config: {} missing [{}], {} extra [{}]",
            missing.len(),
            head(&missing),
            extra.len(),
            head(&extra),
        )));
    }

    validate_shapes(w, cfg)
}

/// Shape checks on the dimension-bearing entry points (Linear weight = `[out, in]`). Shared by the
/// dense/packed path and the INT8-ConvRot path (`check_shape` resolves the diffusers key to the native
/// key and skips a quantized weight whose on-disk shape differs — packed u32 codes or int8 codes).
fn validate_shapes(w: &Weights, cfg: &Krea2Config) -> Result<()> {
    let h = cfg.hidden_size;
    check_shape(w, "img_in.weight", &[h, cfg.in_channels])?;
    check_shape(w, "final_layer.linear.weight", &[cfg.in_channels, h])?;
    check_shape(w, "final_layer.scale_shift_table", &[2, h])?;
    check_shape(w, "txt_in.linear_1.weight", &[h, cfg.text_hidden_dim])?;
    check_shape(w, "txt_in.linear_2.weight", &[h, h])?;
    check_shape(
        w,
        "time_embed.linear_1.weight",
        &[h, cfg.timestep_embed_dim],
    )?;
    check_shape(w, "time_mod_proj.weight", &[cfg.time_mod_dim(), h])?;
    check_shape(w, "text_fusion.projector.weight", &[1, cfg.num_text_layers])?;
    // A representative text-fusion block (full attention, text width).
    let th = cfg.text_hidden_dim;
    check_shape(
        w,
        "text_fusion.layerwise_blocks.0.attn.to_q.weight",
        &[th, th],
    )?;
    check_shape(
        w,
        "text_fusion.layerwise_blocks.0.ff.gate.weight",
        &[cfg.text_intermediate_size, th],
    )?;
    // A representative single-stream block: GQA + the SwiGLU FFN + the 6-factor modulation table.
    check_shape(
        w,
        "transformer_blocks.0.attn.to_q.weight",
        &[cfg.q_dim(), h],
    )?;
    check_shape(
        w,
        "transformer_blocks.0.attn.to_k.weight",
        &[cfg.kv_dim(), h],
    )?;
    check_shape(
        w,
        "transformer_blocks.0.attn.to_gate.weight",
        &[cfg.q_dim(), h],
    )?;
    check_shape(
        w,
        "transformer_blocks.0.ff.gate.weight",
        &[cfg.intermediate_size, h],
    )?;
    check_shape(
        w,
        "transformer_blocks.0.scale_shift_table",
        &[Krea2Config::MOD_FACTORS, h],
    )?;
    Ok(())
}

fn check_shape(w: &Weights, key: &str, expected: &[usize]) -> Result<()> {
    // A packed (quantized) `{base}.weight` is u32-codes with a different on-disk shape; skip the
    // dense-shape check when a sibling `{base}.scales` marks it pre-quantized.
    if let Some(base) = key.strip_suffix(".weight") {
        if w.contains(&format!("{base}.scales")) {
            return Ok(());
        }
    }
    // ConvRot's per-block `scale_shift_table` is stored 1-D (`mod.lin` `[6·h]`) rather than `[6, h]`;
    // the DiT reshapes it identically row-major, so the flat form is correct. `w.shape` resolves to
    // the native key, so compare against the flattened `[6·h]` under ConvRot.
    if w.is_convrot() && key.ends_with(".scale_shift_table") {
        if let Some(shape) = w.shape(key) {
            let flat: usize = expected.iter().product();
            if shape == expected || shape == [flat] {
                return Ok(());
            }
            return Err(candle_gen::candle_core::Error::Msg(format!(
                "krea INT8-ConvRot: {key} shape {shape:?}, expected {expected:?} or [{flat}]"
            )));
        }
    }
    match w.shape(key) {
        Some(shape) if shape == expected => Ok(()),
        Some(shape) => Err(candle_gen::candle_core::Error::Msg(format!(
            "krea: {key} shape {shape:?}, expected {expected:?}"
        ))),
        None => Err(candle_gen::candle_core::Error::Msg(format!(
            "krea: missing tensor {key}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expected_key_count_matches_published_turbo() {
        let cfg = Krea2Config::turbo();
        let keys = expected_transformer_keys(&cfg);
        let unique: BTreeSet<_> = keys.iter().collect();
        assert_eq!(keys.len(), unique.len(), "no duplicate expected keys");
        // 17 top-level + 49 text_fusion (2×12 layerwise + 1 projector + 2×12 refiner) + 364 blocks
        // (28×13) = 430, matching the published safetensors index exactly.
        assert_eq!(keys.len(), 430);
    }
}
