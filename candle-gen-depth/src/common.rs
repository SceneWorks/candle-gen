//! Shared leaf helpers for the candle Depth Anything V2 port: a safetensors weight map
//! ([`Weights`]), a per-last-dim [`Linear`], NHWC↔NCHW conv / transposed-conv wrappers, a small
//! no-mask [`sdpa`] / [`layer_norm`], and a separable NHWC [`bilinear_resize`] (both `align_corners`
//! conventions the DPT neck/head need). The candle twin of `mlx-gen-depth`'s `util.rs` (which mirrors
//! `mlx-gen-sam3`'s `util`).
//!
//! Layout: this port loads the RAW `depth-anything/Depth-Anything-V2-Small-hf` torch checkpoint,
//! whose conv kernels are OIHW (`conv2d`) / IOHW (`conv_transpose2d`) — already candle-native — so we
//! DON'T permute kernels (the MLX side does, because MLX convs are OHWI). We only transpose
//! *activations* NHWC↔NCHW around each conv so the transformer body stays channels-last and mirrors
//! the MLX modules line-by-line.

use std::collections::HashMap;
use std::path::Path;

use candle_gen::candle_core::{safetensors, DType, Device, Tensor, D};
use candle_gen::candle_nn::ops::softmax;
use candle_gen::candle_nn::{LayerNorm, Module};
use candle_gen::{CandleError, Result};

/// A loaded Depth Anything V2 weight map. Tensors are coerced to f32 on load — the parity oracle is
/// f32 and the Small (ViT-S/14) checkpoint fits comfortably in f32 on the target box.
pub struct Weights {
    map: HashMap<String, Tensor>,
}

impl Weights {
    /// Construct from an explicit tensor map (the synthetic-checkpoint test path; the candle twin of
    /// `mlx_gen::weights::Weights::empty()` + `insert`).
    pub fn from_map(map: HashMap<String, Tensor>) -> Self {
        Self { map }
    }

    /// Load every tensor from one `.safetensors` file onto `device`, coercing to f32.
    pub fn from_file(path: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let mut map = HashMap::new();
        Self::extend_from(&mut map, path.as_ref(), device)?;
        Ok(Self { map })
    }

    /// Load + merge every `*.safetensors` shard in `dir` (the published checkpoint ships a single
    /// `model.safetensors`, but be robust to a sharded snapshot).
    pub fn from_dir(dir: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let dir = dir.as_ref();
        let mut shards: Vec<_> = std::fs::read_dir(dir)
            .map_err(|e| CandleError::Msg(format!("read_dir {}: {e}", dir.display())))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().map(|x| x == "safetensors").unwrap_or(false))
            .filter(|p| !candle_gen::gen_core::weightsmeta::is_hidden_file(p))
            .collect();
        shards.sort();
        if shards.is_empty() {
            return Err(CandleError::Msg(format!(
                "no .safetensors files in {}",
                dir.display()
            )));
        }
        let mut map = HashMap::new();
        for shard in &shards {
            Self::extend_from(&mut map, shard, device)?;
        }
        Ok(Self { map })
    }

    fn extend_from(map: &mut HashMap<String, Tensor>, path: &Path, device: &Device) -> Result<()> {
        let raw = safetensors::load(path, device)?;
        for (k, v) in raw {
            let v = match v.dtype() {
                DType::F32 => v,
                DType::F16 | DType::BF16 | DType::F64 => v.to_dtype(DType::F32)?,
                // Integer tensors route through the CPU (a missing int->f32 CUDA cast on this candle
                // build); DA-V2 ships only float weights, but stay robust.
                _ => v
                    .to_device(&Device::Cpu)?
                    .to_dtype(DType::F32)?
                    .to_device(device)?,
            };
            // A key that already exists means TWO shards define the same tensor. In a normal sharded
            // safetensors checkpoint every key lives in exactly one shard, so a cross-shard duplicate is
            // abnormal — a mis-sharded or double-listed checkpoint, or a stray `.safetensors` polluting
            // the snapshot dir. Silently overwriting (the old `insert`) would let a stray file shadow the
            // real weights with no diagnostic, so we hard-error naming the key and the offending shard
            // rather than emit a bare library warning (F-064 / sc-9050; cf. the F-051 no-stderr policy —
            // this is a genuine load fault, surfaced through the crate's normal `Result` channel).
            if map.insert(k.clone(), v).is_some() {
                return Err(CandleError::Msg(format!(
                    "duplicate tensor key {k:?} while merging shard {}: a checkpoint's tensors must \
                     each live in exactly one .safetensors shard — this snapshot has {k:?} in more \
                     than one file (mis-sharded checkpoint or a stray .safetensors in the dir)",
                    path.display()
                )));
            }
        }
        Ok(())
    }

    /// Fetch a required tensor, erroring (not panicking) when a checkpoint is missing a key.
    pub fn require(&self, key: &str) -> Result<Tensor> {
        self.map
            .get(key)
            .cloned()
            .ok_or_else(|| CandleError::Msg(format!("missing tensor: {key}")))
    }
}

/// `"{prefix}.{leaf}"` (or just `leaf` when `prefix` is empty) — the empty-prefix-aware key join
/// (mirrors `mlx-gen-depth`'s `util::join`).
pub(crate) fn join(prefix: &str, leaf: &str) -> String {
    if prefix.is_empty() {
        leaf.to_string()
    } else {
        format!("{prefix}.{leaf}")
    }
}

/// A dense linear over the LAST dim (the loaded `[out, in]` weight + optional bias). Applies to any
/// `[.., in]` tensor by flattening the leading dims (robust for both the `[b, n, c]` token
/// projections and the `[b, nh, seq, hd]` head tensors).
#[derive(Clone)]
pub(crate) struct Linear {
    weight_t: Tensor, // pre-transposed [in, out], contiguous
    bias: Tensor,     // [out]
    out_features: usize,
}

impl Linear {
    /// Load `{name}.weight` + `{name}.bias` (DINOv2's Q/K/V/dense/fc projections all carry a bias).
    pub fn load(w: &Weights, name: &str) -> Result<Self> {
        let weight = w.require(&format!("{name}.weight"))?; // [out, in]
        let out_features = weight.dim(0)?;
        Ok(Self {
            weight_t: weight.t()?.contiguous()?,
            bias: w.require(&format!("{name}.bias"))?,
            out_features,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        let in_features = *dims.last().expect("linear input has rank >= 1");
        let lead: usize = dims[..dims.len() - 1].iter().product();
        let x2 = x.reshape((lead, in_features))?;
        let y = x2.matmul(&self.weight_t)?.broadcast_add(&self.bias)?;
        let mut out_shape = dims;
        *out_shape.last_mut().unwrap() = self.out_features;
        Ok(y.reshape(out_shape)?)
    }
}

/// LayerNorm over the last dim with explicit weight/bias.
pub(crate) fn layer_norm(x: &Tensor, w: &Tensor, b: &Tensor, eps: f64) -> Result<Tensor> {
    let ln = LayerNorm::new(w.clone(), b.clone(), eps);
    Ok(ln.forward(x)?)
}

/// Scaled-dot-product attention, no mask. `q`/`k`/`v`: `[b, nh, seq, hd]` → `[b, nh, seq, hd]`.
pub(crate) fn sdpa(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64) -> Result<Tensor> {
    let kt = k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
    let attn = (q.contiguous()?.matmul(&kt)? * scale)?; // [b, nh, seq_q, seq_k]
    let attn = softmax(&attn, D::Minus1)?;
    Ok(attn.matmul(&v.contiguous()?)?)
}

/// ReLU helper (the head/fusion convs are ReLU-gated).
pub(crate) fn relu(x: &Tensor) -> Result<Tensor> {
    Ok(x.relu()?)
}

/// `conv2d` on an NHWC activation with a torch-native OIHW kernel (loaded as-is). Transposes
/// NHWC→NCHW, runs candle `conv2d`, adds the optional `[O]` bias, transposes back to NHWC.
pub(crate) fn conv2d_nhwc(
    x: &Tensor,
    w: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
    padding: usize,
) -> Result<Tensor> {
    let xc = x.permute([0, 3, 1, 2])?.contiguous()?; // NHWC → NCHW
    let mut y = xc.conv2d(w, padding, stride, 1, 1)?; // [N, O, H', W']
    if let Some(b) = bias {
        y = y.broadcast_add(&b.reshape((1, b.elem_count(), 1, 1))?)?;
    }
    Ok(y.permute([0, 2, 3, 1])?.contiguous()?) // NCHW → NHWC
}

/// `conv_transpose2d` on an NHWC activation with a torch-native IOHW kernel (loaded as-is), pad 0 /
/// output_pad 0, plus the `[O]` bias.
pub(crate) fn conv_transpose2d_nhwc(
    x: &Tensor,
    w: &Tensor,
    bias: &Tensor,
    stride: usize,
) -> Result<Tensor> {
    let xc = x.permute([0, 3, 1, 2])?.contiguous()?;
    let y = xc.conv_transpose2d(w, 0, 0, stride, 1)?; // padding, output_padding, stride, dilation
    let y = y.broadcast_add(&bias.reshape((1, bias.elem_count(), 1, 1))?)?;
    Ok(y.permute([0, 2, 3, 1])?.contiguous()?)
}

/// Build a 1-D bilinear resample matrix `[out_n, in_n]` (one row per output position, two nonzero
/// entries that blend the two bracketing source samples). Following torch `interpolate(mode=
/// "bilinear")`: `align_corners=true` maps output `i` to source `i·(in-1)/(out-1)`; `false` maps to
/// the pixel-center convention `(i+0.5)·in/out - 0.5` (clamped to `[0, in-1]`).
fn interp_matrix(in_n: usize, out_n: usize, align_corners: bool, dev: &Device) -> Result<Tensor> {
    let mut m = vec![0f32; out_n * in_n];
    let last = in_n - 1;
    for i in 0..out_n {
        let src = if align_corners {
            if out_n == 1 {
                0.0
            } else {
                i as f32 * (in_n - 1) as f32 / (out_n - 1) as f32
            }
        } else {
            ((i as f32 + 0.5) * in_n as f32 / out_n as f32 - 0.5).max(0.0)
        };
        let x0 = (src.floor() as usize).min(last);
        let x1 = (x0 + 1).min(last);
        let f = (src - x0 as f32).clamp(0.0, 1.0);
        m[i * in_n + x0] += 1.0 - f;
        m[i * in_n + x1] += f; // += so x0 == x1 (clamped edge) sums to 1
    }
    Ok(Tensor::from_vec(m, (out_n, in_n), dev)?)
}

/// NHWC bilinear resize `[B, H, W, C]` → `[B, out_h, out_w, C]` (torch `interpolate(mode="bilinear")`),
/// applied as two separable matmuls (rows then cols). `align_corners` matches the torch flag at the
/// call site (the DPT fusion ×2 upsample / head upsample use `true`; the fusion residual-match resize
/// uses `false`).
pub(crate) fn bilinear_resize(
    x: &Tensor,
    out_h: usize,
    out_w: usize,
    align_corners: bool,
) -> Result<Tensor> {
    let (b, h, w, c) = x.dims4()?;
    if h == out_h && w == out_w {
        return Ok(x.clone());
    }
    let dev = x.device();
    // Work in NCHW-flattened [b*c, h, w] so the separable matmuls hit the spatial dims.
    let xc = x
        .permute([0, 3, 1, 2])?
        .contiguous()?
        .reshape((b * c, h, w))?;
    let wy = interp_matrix(h, out_h, align_corners, dev)?; // [out_h, h]
    let mid = wy
        .unsqueeze(0)?
        .broadcast_as((b * c, out_h, h))?
        .contiguous()?
        .matmul(&xc)?; // [b*c, out_h, w]
    let wx = interp_matrix(w, out_w, align_corners, dev)?; // [out_w, w]
    let out = mid.matmul(
        &wx.t()?
            .unsqueeze(0)?
            .broadcast_as((b * c, w, out_w))?
            .contiguous()?,
    )?; // [b*c, out_h, out_w]
    Ok(out
        .reshape((b, c, out_h, out_w))?
        .permute([0, 2, 3, 1])?
        .contiguous()?) // → NHWC
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interp_matrix_rows_sum_to_one() {
        let dev = Device::Cpu;
        for ac in [true, false] {
            let m = interp_matrix(4, 8, ac, &dev).unwrap();
            for row in m.to_vec2::<f32>().unwrap() {
                let s: f32 = row.iter().sum();
                assert!((s - 1.0).abs() < 1e-5, "align_corners={ac} row sums to {s}");
            }
        }
    }

    #[test]
    fn bilinear_identity_when_same_dims() {
        let dev = Device::Cpu;
        let x = Tensor::randn(0f32, 1f32, (1, 3, 4, 2), &dev).unwrap();
        let y = bilinear_resize(&x, 3, 4, true).unwrap();
        let a = x.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(a, b);
    }

    /// Write a single-tensor `.safetensors` file (`name -> [value]`, f32) at `path`.
    fn write_shard(path: &Path, name: &str, value: f32) {
        let t = Tensor::new(&[value], &Device::Cpu).unwrap();
        let mut m = HashMap::new();
        m.insert(name.to_string(), t);
        safetensors::save(&m, path).unwrap();
    }

    fn scratch_dir(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!(
            "candle_gen_depth_common_{tag}_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// Two shards with DISJOINT keys merge cleanly into the full key union (no false positive) — the
    /// normal sharded-checkpoint path is preserved byte-for-byte (F-064 / sc-9050).
    #[test]
    fn from_dir_merges_disjoint_shards_into_key_union() {
        let dir = scratch_dir("union");
        write_shard(
            &dir.join("model-00001-of-00002.safetensors"),
            "a.weight",
            1.0,
        );
        write_shard(
            &dir.join("model-00002-of-00002.safetensors"),
            "b.weight",
            2.0,
        );

        let w = Weights::from_dir(&dir, &Device::Cpu).unwrap();
        assert_eq!(
            w.require("a.weight").unwrap().to_vec1::<f32>().unwrap(),
            vec![1.0]
        );
        assert_eq!(
            w.require("b.weight").unwrap().to_vec1::<f32>().unwrap(),
            vec![2.0]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Two shards that BOTH define the same key are detected and surfaced as a descriptive error naming
    /// the offending key and shard file, instead of silently overwriting (F-064 / sc-9050).
    #[test]
    fn from_dir_errors_on_cross_shard_duplicate_key() {
        let dir = scratch_dir("dup");
        write_shard(
            &dir.join("model-00001-of-00002.safetensors"),
            "dup.weight",
            1.0,
        );
        // A stray / mis-sharded second file redefines the same key.
        write_shard(
            &dir.join("model-00002-of-00002.safetensors"),
            "dup.weight",
            2.0,
        );

        match Weights::from_dir(&dir, &Device::Cpu) {
            Err(CandleError::Msg(m)) => {
                assert!(m.contains("duplicate tensor key"), "got: {m}");
                assert!(m.contains("dup.weight"), "must name the key, got: {m}");
                assert!(
                    m.contains("model-00002-of-00002.safetensors"),
                    "must name the shard, got: {m}"
                );
            }
            Err(other) => panic!("expected a crafted duplicate-key Msg error, got: {other:?}"),
            Ok(_) => panic!("expected an error on a cross-shard duplicate key, got Ok"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
