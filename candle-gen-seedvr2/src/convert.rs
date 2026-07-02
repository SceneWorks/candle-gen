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

use candle_gen::candle_core::DType;

use crate::weights::Weights;

type CResult<T> = candle_gen::Result<T>;

/// VAE: pass-through (candle keeps torch conv layout), cast to `dt` tensor-by-tensor.
///
/// Consumes `raw` and drains it so each raw fp16 tensor is dropped as soon as its `dt` cast copy is
/// produced — peak load memory stays ~1× the checkpoint instead of holding the whole raw set and the
/// whole cast set at once (sc-9042/F-058). The loaded values are identical to the old
/// `convert_vae(&raw)?.cast(dt)?`.
pub fn convert_vae(raw: Weights, dt: DType) -> CResult<Weights> {
    let mut out = Weights::empty();
    for (k, v) in raw.into_iter_entries() {
        out.insert(k, v.to_dtype(dt)?);
        // `v` (the raw fp16 tensor) drops here before the next one is read.
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
///
/// Consumes `raw`, casting to `dt` and draining tensor-by-tensor so each raw fp16 tensor drops as its
/// cast copy is produced (peak load memory ~1×, not the raw set plus the cast set at once —
/// sc-9042/F-058). A `.all` key casts once and clones the cast (a cheap `Arc` bump) into both
/// `_vid`/`_txt`, so at most one raw tensor is resident at a time. Values match the old
/// `convert_dit(&raw)?.cast(dt)?` exactly.
pub fn convert_dit(raw: Weights, dt: DType) -> CResult<Weights> {
    let mut out = Weights::empty();
    for (k, v) in raw.into_iter_entries() {
        let cast = v.to_dtype(dt)?;
        // `v` (raw fp16) drops at the end of this iteration.
        let mut targets = dit_targets(&k).into_iter();
        // Move the cast into the first target; clone (Arc bump) for any additional `.all` targets.
        if let Some(first) = targets.next() {
            for extra in targets {
                out.insert(extra, cast.clone());
            }
            out.insert(first, cast);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::{convert_dit, convert_vae, dit_targets};
    use crate::weights::Weights;
    use candle_gen::candle_core::{DType, Device, Tensor};

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

    // Build a small raw `Weights` (fp16 on CPU) with representative keys.
    fn raw_fixture(dev: &Device) -> (Weights, Vec<f32>, Vec<f32>) {
        let mut w = Weights::empty();
        let vae_vals = vec![0.5f32, -1.25, 2.0, -0.75];
        let dit_all_vals = vec![1.0f32, -2.0, 3.5, -4.25, 0.125, -0.0625];
        let vae = Tensor::from_vec(vae_vals.clone(), (2, 2), dev)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        let dit_all = Tensor::from_vec(dit_all_vals.clone(), (2, 3), dev)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        w.insert("encoder.conv_in.weight", vae);
        w.insert("blocks.0.attn.proj_qkv.all.weight", dit_all);
        (w, vae_vals, dit_all_vals)
    }

    // Reference: the pre-sc-9042 two-phase behavior — clone/rename at fp16, then cast the whole set.
    // The streaming `convert_*(raw, dt)` must produce byte-identical results.
    fn expected_cast(vals: &[f32], dt: DType, dev: &Device, shape: (usize, usize)) -> Vec<f32> {
        Tensor::from_vec(vals.to_vec(), shape, dev)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap()
            .to_dtype(dt)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }

    fn tensor_f32(w: &Weights, key: &str) -> Vec<f32> {
        w.require(key)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }

    #[test]
    fn streaming_convert_preserves_values_and_dtype_bf16() {
        let dev = Device::Cpu;
        let (raw, vae_vals, dit_vals) = raw_fixture(&dev);
        let dt = DType::BF16;
        let vae_w = convert_vae(
            {
                let mut v = Weights::empty();
                v.insert(
                    "encoder.conv_in.weight",
                    raw.require("encoder.conv_in.weight").unwrap().clone(),
                );
                v
            },
            dt,
        )
        .unwrap();
        let dit_w = convert_dit(raw, dt).unwrap();

        // dtype is the requested target on every converted tensor.
        assert_eq!(vae_w.require("encoder.conv_in.weight").unwrap().dtype(), dt);
        assert_eq!(
            dit_w
                .require("blocks.0.attn.proj_qkv_vid.weight")
                .unwrap()
                .dtype(),
            dt
        );

        // values are byte-identical to the old fp16→cast path.
        assert_eq!(
            tensor_f32(&vae_w, "encoder.conv_in.weight"),
            expected_cast(&vae_vals, dt, &dev, (2, 2))
        );
        let want = expected_cast(&dit_vals, dt, &dev, (2, 3));
        // `.all` is duplicated into BOTH streams, each an exact copy.
        assert_eq!(
            tensor_f32(&dit_w, "blocks.0.attn.proj_qkv_vid.weight"),
            want
        );
        assert_eq!(
            tensor_f32(&dit_w, "blocks.0.attn.proj_qkv_txt.weight"),
            want
        );
    }

    #[test]
    fn streaming_convert_preserves_values_and_dtype_f32() {
        let dev = Device::Cpu;
        let (raw, _vae_vals, dit_vals) = raw_fixture(&dev);
        let dt = DType::F32;
        let dit_w = convert_dit(raw, dt).unwrap();
        assert_eq!(
            dit_w
                .require("blocks.0.attn.proj_qkv_txt.weight")
                .unwrap()
                .dtype(),
            dt
        );
        let want = expected_cast(&dit_vals, dt, &dev, (2, 3));
        assert_eq!(
            tensor_f32(&dit_w, "blocks.0.attn.proj_qkv_vid.weight"),
            want
        );
        assert_eq!(
            tensor_f32(&dit_w, "blocks.0.attn.proj_qkv_txt.weight"),
            want
        );
    }
}
