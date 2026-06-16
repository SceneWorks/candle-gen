//! Native weight converter — candle port of `mlx-gen-seedvr2/src/convert.rs`. Maps the raw
//! `numz/SeedVR2_comfyUI` checkpoints to the key/layout the candle modules load. No Python.
//!
//! - **VAE:** pure pass-through. Unlike the MLX port (which transposes conv weights to channels-last
//!   NDHWC), candle keeps the torch `[O,I,kT,kH,kW]` layout — [`crate::conv3d::CausalConv3d`] slices
//!   the temporal axis to form each conv2d kernel `[O,I,kH,kW]` directly.
//! - **DiT:** rename the dotted attention/ada submodules (`attn.proj_qkv.vid` → `attn.proj_qkv_vid`,
//!   `ada.vid` → `ada.params_vid`, `attn.rope.rope.freqs` → `attn.rope.freqs`, `vid_out_ada.` strip,
//!   …). Shared layers store the attention projections once under `.all`; they are duplicated into
//!   both `_vid` and `_txt`. MLP keys pass through.

use crate::weights::Weights;

type CResult<T> = candle_gen::Result<T>;

/// VAE: pass-through (candle keeps torch conv layout).
pub fn convert_vae(raw: &Weights) -> CResult<Weights> {
    let mut out = Weights::empty();
    for k in raw.keys().map(String::from).collect::<Vec<_>>() {
        out.insert(k.clone(), raw.require(&k)?.clone());
    }
    Ok(out)
}

const ATTN_SUBS: [&str; 4] = ["proj_qkv", "proj_out", "norm_q", "norm_k"];

/// Map a raw DiT key to its converted target name(s) (two for a shared-layer attention `.all`).
fn dit_targets(k: &str) -> Vec<String> {
    // output AdaLN scale/shift live under `vid_out_ada.` in the checkpoint, flat in the model.
    if let Some(rest) = k.strip_prefix("vid_out_ada.") {
        return vec![rest.to_string()];
    }
    if k.contains(".attn.rope.rope.freqs") {
        return vec![k.replace(".attn.rope.rope.", ".attn.rope.")];
    }
    for sub in ATTN_SUBS {
        let all = format!(".attn.{sub}.all.");
        if k.contains(&all) {
            return vec![
                k.replace(&all, &format!(".attn.{sub}_vid.")),
                k.replace(&all, &format!(".attn.{sub}_txt.")),
            ];
        }
        for stream in ["vid", "txt"] {
            let from = format!(".attn.{sub}.{stream}.");
            if k.contains(&from) {
                return vec![k.replace(&from, &format!(".attn.{sub}_{stream}."))];
            }
        }
    }
    for stream in ["vid", "txt", "all"] {
        let from = format!(".ada.{stream}.");
        if k.contains(&from) {
            return vec![k.replace(&from, &format!(".ada.params_{stream}."))];
        }
    }
    vec![k.to_string()] // mlp.* and top-level pass through
}

/// DiT: key renames (no transposes — all weights are 2-D). Shared-layer `.all` attention projections
/// are duplicated into `_vid` + `_txt`.
pub fn convert_dit(raw: &Weights) -> CResult<Weights> {
    let mut out = Weights::empty();
    for k in raw.keys().map(String::from).collect::<Vec<_>>() {
        let v = raw.require(&k)?;
        for target in dit_targets(&k) {
            out.insert(target, v.clone());
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::dit_targets;

    #[test]
    fn renames_match_reference() {
        assert_eq!(
            dit_targets("blocks.0.attn.proj_qkv.vid.weight"),
            vec!["blocks.0.attn.proj_qkv_vid.weight"]
        );
        assert_eq!(
            dit_targets("blocks.5.attn.proj_qkv.all.weight"),
            vec![
                "blocks.5.attn.proj_qkv_vid.weight",
                "blocks.5.attn.proj_qkv_txt.weight"
            ]
        );
        assert_eq!(
            dit_targets("blocks.0.ada.vid.attn_shift"),
            vec!["blocks.0.ada.params_vid.attn_shift"]
        );
        assert_eq!(
            dit_targets("blocks.0.attn.rope.rope.freqs"),
            vec!["blocks.0.attn.rope.freqs"]
        );
        assert_eq!(dit_targets("vid_out_ada.out_scale"), vec!["out_scale"]);
        assert_eq!(
            dit_targets("blocks.0.mlp.vid.proj_in.weight"),
            vec!["blocks.0.mlp.vid.proj_in.weight"]
        );
    }
}
