//! Shared **Qwen3-VL text-encoder** grounding machinery (sc-11205 / F-118): the MRoPE / vision-splice
//! helpers that the Boogu (Qwen3-VL-8B) and Krea 2 (Qwen3-VL-4B) condition encoders both need on their
//! image-grounded edit paths.
//!
//! Both encoders are the same decoder-only Qwen3-VL text tower (GQA, per-head q/k RMSNorm, HF
//! half-split RoPE θ = 5e6, head_dim 128, MRoPE section `[24, 20, 20]`, spatial-merge 2) — only the
//! width / layer-selection policy differs. Before this module the RoPE table, the GQA kv-repeat, the
//! `<|image_pad|>` block scan, the vision-embed splice, the 3-D interleaved-MRoPE position + cos/sin
//! build, and the additive causal mask were **byte-identical copies** in
//! `candle-gen-boogu/src/text_encoder.rs` and `candle-gen-krea/src/text_encoder.rs` — ~250 lines of
//! parity-critical grounding logic (the interleaved-MRoPE axis selection is a known drift magnet) that
//! had to be fixed twice. Krea already depends on Boogu for the vision tower, but the grounding lives at
//! the shared commons layer here so **neither** provider owns the other's copy and both draw from one
//! audited source.
//!
//! Every helper is a pure re-hosting of the pre-existing per-crate code — no numeric change — so the
//! grounded edit encode stays byte-for-byte what it was. The per-crate encoders keep their own
//! architecture-specific `MROPE_SECTION` / `mrope_section` value and pass it in.

use candle_core::{DType, Device, Result, Tensor};

/// HF half-split RoPE table (θ over `head_dim`), built once for the max sequence length (f32). The
/// plain 1-D RoPE the **text** path uses (Qwen3-VL's MRoPE sections all index the same sequential text
/// position when there are no image tokens).
pub struct Rotary {
    cos: Tensor,
    sin: Tensor,
}

impl Rotary {
    /// Build the `cos`/`sin` tables `[max_seq, head_dim/2]` (f32) for sequential positions `0..max_seq`.
    pub fn new(head_dim: usize, theta: f32, max_seq: usize, device: &Device) -> Result<Self> {
        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / theta.powf(i as f32 / head_dim as f32))
            .collect();
        let n = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, n), device)?;
        let t = Tensor::arange(0u32, max_seq as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_seq, 1))?;
        let freqs = t.matmul(&inv_freq)?; // (max_seq, head_dim/2)
        Ok(Self {
            cos: freqs.cos()?,
            sin: freqs.sin()?,
        })
    }

    /// The plain 1-D RoPE `(cos, sin)` `[seq, head_dim/2]` for the text path (sequential positions).
    pub fn text(&self, seq: usize) -> Result<(Tensor, Tensor)> {
        Ok((self.cos.narrow(0, 0, seq)?, self.sin.narrow(0, 0, seq)?))
    }
}

/// Repeat each kv head `groups` times along the head axis ([B,nkv,S,D] → [B,nkv·groups,S,D]) — the GQA
/// key/value broadcast the Qwen3-VL text tower's attention needs (32 query / 8 kv heads).
pub fn repeat_kv(x: &Tensor, groups: usize) -> Result<Tensor> {
    if groups == 1 {
        return Ok(x.clone());
    }
    let (b, nkv, s, d) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((b, nkv, groups, s, d))?
        .contiguous()?
        .reshape((b, nkv * groups, s, d))
}

/// Slice `[1, s, d]` along the sequence axis (axis 1) to `[start, end)`.
pub fn slice_seq(x: &Tensor, start: usize, end: usize) -> Result<Tensor> {
    x.narrow(1, start, end - start)
}

/// Replace `x[:, start:end, :]` with `repl` (`[1, end-start, d]`) via concat of the surrounding slices.
pub fn replace_seq(
    x: &Tensor,
    repl: &Tensor,
    start: usize,
    end: usize,
    s: usize,
) -> Result<Tensor> {
    let before = x.narrow(1, 0, start)?;
    let after = x.narrow(1, end, s - end)?;
    Tensor::cat(&[&before, repl, &after], 1)
}

/// Contiguous runs of `image_token_id` in `ids`, returned as `(start, len)` in input-id order. Each run
/// is one reference image's `<|image_pad|>` block.
pub fn image_blocks(ids: &[u32], image_token_id: u32) -> Vec<(usize, usize)> {
    let mut blocks = Vec::new();
    let mut i = 0usize;
    while i < ids.len() {
        if ids[i] == image_token_id {
            let start = i;
            while i < ids.len() && ids[i] == image_token_id {
                i += 1;
            }
            blocks.push((start, i - start));
        } else {
            i += 1;
        }
    }
    blocks
}

/// Vision spatial merge — the LM sees one token per `merge²` patches (Qwen3-VL `spatial_merge_size`).
const SPATIAL_MERGE: i64 = 2;

/// 3-D MRoPE positions per token (mirrors `get_rope_index` + `get_vision_position_ids`): text tokens
/// advance `(i, i, i)`; the k-th image block (at running offset `cur`) takes `grids[k] = [t, h, w]`,
/// gets `t = cur`, `h = cur + row`, `w = cur + col` over its `(h/merge)×(w/merge)` merged grid, then
/// advances `cur += max(h, w) / merge`. Multiple image blocks consume `grids` in order.
pub fn mrope_positions(
    ids: &[u32],
    image_token_id: u32,
    grids: &[[i32; 3]],
) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
    let (mut pt, mut ph, mut pw) = (Vec::new(), Vec::new(), Vec::new());
    let mut cur = 0i64;
    let mut i = 0usize;
    let mut img_k = 0usize;
    while i < ids.len() {
        if ids[i] == image_token_id {
            let g = grids[img_k];
            let (llm_h, llm_w) = (g[1] as i64 / SPATIAL_MERGE, g[2] as i64 / SPATIAL_MERGE);
            let step = (g[1].max(g[2]) as i64) / SPATIAL_MERGE;
            for idx in 0..(llm_h * llm_w) {
                pt.push(cur);
                ph.push(cur + idx / llm_w);
                pw.push(cur + idx % llm_w);
            }
            cur += step;
            i += (llm_h * llm_w) as usize;
            img_k += 1;
        } else {
            pt.push(cur);
            ph.push(cur);
            pw.push(cur);
            cur += 1;
            i += 1;
        }
    }
    (pt, ph, pw)
}

/// Build the interleaved-MRoPE `cos`/`sin` `[s, head_dim/2]` (f32) for the image-grounded path. Each of
/// the `head_dim/2` frequencies takes its position from the T/H/W axis per the Qwen3-VL interleave:
/// within the first `mrope_section[1]·3` indices, `j%3==1 → H`, `j%3==2 → W`, else `T` (the tail stays
/// `T`). `mrope_section` is the per-axis (T/H/W) frequency counts over `head_dim/2` (`[24, 20, 20]` for
/// both Boogu and Krea); each provider threads its own config value so a future divergence stays local.
pub fn mrope_cos_sin(
    head_dim: usize,
    mrope_section: [usize; 3],
    rope_theta: f32,
    pt: &[i64],
    ph: &[i64],
    pw: &[i64],
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let s = pt.len();
    let hd = head_dim;
    let half = hd / 2;
    let sec_h = mrope_section[1] * 3;
    let sec_w = mrope_section[2] * 3;
    let inv: Vec<f32> = (0..half)
        .map(|j| (rope_theta as f64).powf(-(2.0 * j as f64) / hd as f64) as f32)
        .collect();

    let mut freqs = vec![0f32; s * half];
    for (i, ((&t, &h), &w)) in pt.iter().zip(ph).zip(pw).enumerate() {
        for j in 0..half {
            let pos = if j < sec_h && j % 3 == 1 {
                h
            } else if j < sec_w && j % 3 == 2 {
                w
            } else {
                t
            };
            freqs[i * half + j] = pos as f32 * inv[j];
        }
    }
    let freqs = Tensor::from_vec(freqs, (s, half), device)?;
    Ok((freqs.cos()?, freqs.sin()?))
}

/// Additive causal mask `[B, 1, S, S]` (f32): `0` where query `i` may attend key `j` (`j ≤ i`),
/// `-inf` otherwise. No padding term (the candle tokenizer emits no padding).
pub fn causal_mask(b: usize, s: usize, device: &Device) -> Result<Tensor> {
    let mut data = vec![0f32; b * s * s];
    for bi in 0..b {
        for i in 0..s {
            for j in (i + 1)..s {
                data[(bi * s + i) * s + j] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_vec(data, (b, 1, s, s), device)
}

#[cfg(test)]
mod tests {
    use super::*;

    const IMG: u32 = 151655;
    /// The Qwen3-VL MRoPE section shared by both Boogu and Krea (`text_config.rope_parameters`).
    const MROPE_SECTION: [usize; 3] = [24, 20, 20];

    #[test]
    fn image_blocks_finds_runs_in_order() {
        // text, text, [4 image], text, [2 image], text.
        let ids = [9u32, 9, IMG, IMG, IMG, IMG, 9, IMG, IMG, 9];
        assert_eq!(image_blocks(&ids, IMG), vec![(2, 4), (7, 2)]);
    }

    #[test]
    fn mrope_positions_advance_across_two_images() {
        // Block 0 ↔ grid [1,4,4] (merged 2×2 = 4 tokens, t-step max(4,4)/2 = 2);
        // block 1 ↔ grid [1,4,2] (merged 2×1 = 2 tokens, t-step max(4,2)/2 = 2).
        let ids = [9u32, 9, IMG, IMG, IMG, IMG, 9, IMG, IMG, 9];
        let grids = [[1, 4, 4], [1, 4, 2]];
        let (pt, ph, pw) = mrope_positions(&ids, IMG, &grids);
        assert_eq!(pt.len(), ids.len());

        // Leading text advances 0,1.
        assert_eq!((pt[0], pt[1]), (0, 1));
        // Image 0 sits at t-axis = 2 (the running offset); spatial in h/w.
        assert_eq!(&pt[2..6], &[2, 2, 2, 2]);
        assert_eq!(&ph[2..6], &[2, 2, 3, 3]); // rows 0,0,1,1 + offset 2
        assert_eq!(&pw[2..6], &[2, 3, 2, 3]); // cols 0,1,0,1 + offset 2
                                              // Text after image 0: offset advanced by max(4,4)/2 = 2 → 4.
        assert_eq!(pt[6], 4);
        // Image 1 sits at t-axis = 5 (one past the text), 2 tokens (2×1 grid).
        assert_eq!(&pt[7..9], &[5, 5]);
        assert_eq!(&ph[7..9], &[5, 6]); // rows 0,1 + offset 5
        assert_eq!(&pw[7..9], &[5, 5]); // single column
                                        // Trailing text: offset advanced by max(4,2)/2 = 2 → 7.
        assert_eq!(pt[9], 7);
    }

    #[test]
    fn replace_and_slice_seq_roundtrip() {
        let dev = Device::Cpu;
        // [1, 4, 2] sequence; replace the middle two positions with a marker block, then slice it back.
        let base = Tensor::arange(0f32, 8f32, &dev)
            .unwrap()
            .reshape((1, 4, 2))
            .unwrap();
        let repl = Tensor::from_vec(vec![100f32, 101.0, 102.0, 103.0], (1, 2, 2), &dev).unwrap();
        let out = replace_seq(&base, &repl, 1, 3, 4).unwrap();
        assert_eq!(out.dims(), &[1, 4, 2]);
        let mid = slice_seq(&out, 1, 3).unwrap();
        assert_eq!(
            mid.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![100.0, 101.0, 102.0, 103.0]
        );
        // The untouched head/tail rows are preserved verbatim.
        let head = slice_seq(&out, 0, 1).unwrap();
        assert_eq!(
            head.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![0.0, 1.0]
        );
    }

    #[test]
    fn causal_mask_is_lower_triangular() {
        let m = causal_mask(1, 3, &Device::Cpu).unwrap();
        assert_eq!(m.dims(), &[1, 1, 3, 3]);
        let v = m.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // Row i attends keys j ≤ i (0.0) and is masked (-inf) for j > i.
        let neg = f32::NEG_INFINITY;
        assert_eq!(v, vec![0.0, neg, neg, 0.0, 0.0, neg, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn repeat_kv_broadcasts_groups() {
        let dev = Device::Cpu;
        // [B=1, nkv=2, S=1, D=2]; repeat each kv head 3× → [1, 6, 1, 2].
        let kv = Tensor::from_vec(vec![1f32, 2.0, 3.0, 4.0], (1, 2, 1, 2), &dev).unwrap();
        let out = repeat_kv(&kv, 3).unwrap();
        assert_eq!(out.dims(), &[1, 6, 1, 2]);
        let v = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // Head 0 (1,2) repeated 3×, then head 1 (3,4) repeated 3×.
        assert_eq!(
            v,
            vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0]
        );
        // groups == 1 is the identity fast path.
        assert_eq!(
            repeat_kv(&kv, 1)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            kv.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
    }

    #[test]
    fn mrope_cos_sin_matches_text_rope_when_all_axes_equal() {
        // When every token's (t,h,w) are equal (a pure-text sequence), the interleaved MRoPE table must
        // reproduce the plain 1-D `Rotary` table position-for-position (the encoders rely on this — the
        // text path uses `Rotary`, the grounded path uses `mrope_cos_sin`, and a pure-text prompt fed to
        // either must agree). Head_dim 128, θ = 5e6, section [24,20,20] (the Boogu/Krea shared config).
        let dev = Device::Cpu;
        let (head_dim, theta) = (128usize, 5_000_000f32);
        let s = 5usize;
        let pos: Vec<i64> = (0..s as i64).collect();
        let (cos_m, sin_m) =
            mrope_cos_sin(head_dim, MROPE_SECTION, theta, &pos, &pos, &pos, &dev).unwrap();

        let rot = Rotary::new(head_dim, theta, s, &dev).unwrap();
        let (cos_t, sin_t) = rot.text(s).unwrap();

        assert_eq!(cos_m.dims(), cos_t.dims());
        let dc = (&cos_m - &cos_t)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let ds = (&sin_m - &sin_t)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            dc < 1e-6 && ds < 1e-6,
            "MRoPE with equal axes must equal the 1-D text RoPE (dc={dc}, ds={ds})"
        );
    }
}
