//! Native Qwen2.5-VL **vision tower** — the planner's image/video ViT encoder. Candle sibling of
//! `mlx-gen-bernini/src/vision.rs` (sc-5134).
//!
//! Port of `Qwen2_5_VisionTransformerPretrainedModel`. Produces `out_hidden`-d (3584) ViT tokens from
//! packed patch pixels + a `grid_thw` geometry. Weights live under `visual.*` in the planner snapshot.
//!
//! Structure mirrored faithfully:
//!   - **Patch embed** — a bias-free `Conv3d` with kernel == stride == `[temporal 2, 14, 14]`. Since
//!     the kernel spans the whole patch, the conv is exactly a per-patch matmul, so the 5-D
//!     `[embed, in, t, h, w]` weight is folded to `[embed, in·t·h·w]` and run as a bias-less `Linear`.
//!   - **`depth` blocks** — pre-norm (`Qwen2RMSNorm`, eps 1e-6) → fused-QKV (bias) attention with a
//!     **2-D rotary** (head_dim/2, θ 10000, NeoX `rotate_half`, f32) → `proj`; pre-norm → **SwiGLU**
//!     MLP (`gate`/`up`/`down`, **bias**, SiLU). Attention is windowed (`window_size 112`) on every
//!     block except `fullatt_block_indexes [7,15,23,31]` (full). The window reorder permutes
//!     merge-units; `cu_seqlens` (full, per frame) vs `cu_window_seqlens` (windowed) give the
//!     block-diagonal additive mask; softmax accumulates in f32.
//!   - **Patch merger** — `ln_q` RMSNorm → concat each `spatial_merge_size²`(=4) group → `5120` →
//!     `Linear → GELU → Linear` → `out_hidden`; then the window permutation is undone (`argsort`).
//!
//! All the integer index gymnastics depend only on `grid_thw`, so they are computed host-side in
//! [`VisionTower::build_plan`] — mirroring the reference — and the resulting permutation / rope table /
//! block masks are handed to the candle graph. Validated near-bit (f32) against the same synthetic
//! `tests/fixtures/vision_tower_golden.safetensors` the MLX lane asserts.

use candle_gen::candle_core::{DType, Tensor, D};
use candle_gen::candle_nn::{ops::softmax_last_dim, Linear, Module, VarBuilder};
use candle_gen::{CandleError, Result as CResult};

use crate::nn::{lin_bias, rms_norm};

const RMS_EPS: f64 = 1e-6;
const ROPE_THETA: f64 = 10000.0;

/// Qwen2.5-VL vision-tower config (the `vision_config` block of `mllm/config.json`).
#[derive(Clone, Debug)]
pub struct VisionConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub intermediate_size: usize,
    pub depth: usize,
    pub fullatt_block_indexes: Vec<usize>,
    pub spatial_merge_size: usize,
    pub window_size: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub in_channels: usize,
    pub out_hidden_size: usize,
}

impl Default for VisionConfig {
    /// Qwen2.5-VL-7B vision tower (the Bernini planner base).
    fn default() -> Self {
        Self {
            hidden_size: 1280,
            num_heads: 16,
            intermediate_size: 3420,
            depth: 32,
            fullatt_block_indexes: vec![7, 15, 23, 31],
            spatial_merge_size: 2,
            window_size: 112,
            patch_size: 14,
            temporal_patch_size: 2,
            in_channels: 3,
            out_hidden_size: 3584,
        }
    }
}

impl VisionConfig {
    /// Read the `vision_config` sub-object of a `qwen2_5_vl_config.json` (the snapshot copy of
    /// `mllm/config.json`).
    pub fn from_config_json(path: &std::path::Path) -> CResult<Self> {
        let v: serde_json::Value = serde_json::from_slice(
            &std::fs::read(path).map_err(|e| CandleError::Msg(format!("read config: {e}")))?,
        )
        .map_err(|e| CandleError::Msg(format!("parse {}: {e}", path.display())))?;
        let vc = v.get("vision_config").unwrap_or(&v);
        let d = Self::default();
        let i = |k: &str, dv: usize| {
            vc.get(k)
                .and_then(serde_json::Value::as_u64)
                .map(|x| x as usize)
                .unwrap_or(dv)
        };
        let fullatt = vc
            .get("fullatt_block_indexes")
            .and_then(serde_json::Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_u64().map(|n| n as usize))
                    .collect::<Vec<_>>()
            })
            .unwrap_or(d.fullatt_block_indexes);
        Ok(Self {
            hidden_size: i("hidden_size", d.hidden_size),
            num_heads: i("num_heads", d.num_heads),
            intermediate_size: i("intermediate_size", d.intermediate_size),
            depth: i("depth", d.depth),
            fullatt_block_indexes: fullatt,
            spatial_merge_size: i("spatial_merge_size", d.spatial_merge_size),
            window_size: i("window_size", d.window_size),
            patch_size: i("patch_size", d.patch_size),
            temporal_patch_size: i("temporal_patch_size", d.temporal_patch_size),
            // the package config spells this `in_chans`.
            in_channels: vc
                .get("in_chans")
                .or_else(|| vc.get("in_channels"))
                .and_then(serde_json::Value::as_u64)
                .map(|x| x as usize)
                .unwrap_or(d.in_channels),
            out_hidden_size: i("out_hidden_size", d.out_hidden_size),
        })
    }

    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    /// `spatial_merge_size²` — patches per merged token.
    fn merge_unit(&self) -> usize {
        self.spatial_merge_size * self.spatial_merge_size
    }

    /// Window edge in **merged-token** units: `window // merge // patch`.
    fn vit_merger_window_size(&self) -> usize {
        self.window_size / self.spatial_merge_size / self.patch_size
    }
}

/// HF half-split rotary `rotate_half`: `cat(-x[d/2:], x[:d/2])` on the last axis.
fn rotate_half(x: &Tensor) -> CResult<Tensor> {
    let d = x.dim(D::Minus1)?;
    let half = d / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, d - half)?;
    Ok(Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?)
}

/// One vision block: pre-norm windowed/full attention + pre-norm SwiGLU MLP, both residual.
struct Block {
    norm1: Tensor,
    norm2: Tensor,
    qkv: Linear,
    proj: Linear,
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl Block {
    fn new(vb: &VarBuilder) -> CResult<Self> {
        Ok(Self {
            norm1: vb.get_unchecked("norm1.weight")?,
            norm2: vb.get_unchecked("norm2.weight")?,
            qkv: lin_bias(&vb.pp("attn"), "qkv")?,
            proj: lin_bias(&vb.pp("attn"), "proj")?,
            gate: lin_bias(&vb.pp("mlp"), "gate_proj")?,
            up: lin_bias(&vb.pp("mlp"), "up_proj")?,
            down: lin_bias(&vb.pp("mlp"), "down_proj")?,
        })
    }

    /// Eager attention over `x` `[seq, dim]` with the precomputed `cos`/`sin` `[seq, head_dim]` (f32)
    /// and an additive `mask` `[1, seq, seq]`. q/k/v project → split heads → 2-D RoPE → `softmax(q·kᵀ/√d
    /// + mask)·v` → proj.
    fn attention(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: &Tensor,
        nh: usize,
    ) -> CResult<Tensor> {
        let (seq, dim) = x.dims2()?;
        let hd = dim / nh;

        let qkv = self.qkv.forward(x)?.reshape((seq, 3, nh, hd))?;
        let q = qkv.narrow(1, 0, 1)?.reshape((seq, nh, hd))?;
        let k = qkv.narrow(1, 1, 1)?.reshape((seq, nh, hd))?;
        let v = qkv.narrow(1, 2, 1)?.reshape((seq, nh, hd))?;

        // 2-D RoPE, cos/sin broadcast over the head axis ([seq,1,head_dim]).
        let cos = cos.reshape((seq, 1, hd))?;
        let sin = sin.reshape((seq, 1, hd))?;
        let rope = |t: &Tensor| -> CResult<Tensor> {
            Ok((t.broadcast_mul(&cos)? + rotate_half(t)?.broadcast_mul(&sin)?)?)
        };
        let q = rope(&q)?.permute((1, 0, 2))?.contiguous()?; // [nh, seq, hd]
        let k = rope(&k)?.permute((1, 0, 2))?.contiguous()?;
        let v = v.permute((1, 0, 2))?.contiguous()?;

        let scale = (hd as f64).powf(-0.5);
        let scores = (q.matmul(&k.transpose(1, 2)?.contiguous()?)? * scale)?;
        let scores = scores.broadcast_add(mask)?; // [nh, seq, seq] + [1, seq, seq]
        let weights = softmax_last_dim(&scores)?;
        let out = weights
            .matmul(&v)? // [nh, seq, hd]
            .permute((1, 0, 2))?
            .contiguous()?
            .reshape((seq, dim))?;
        Ok(self.proj.forward(&out)?)
    }

    fn mlp(&self, x: &Tensor) -> CResult<Tensor> {
        let gated = (self.gate.forward(x)?.silu()? * self.up.forward(x)?)?;
        Ok(self.down.forward(&gated)?)
    }

    fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: &Tensor,
        nh: usize,
    ) -> CResult<Tensor> {
        let a = self.attention(&rms_norm(x, &self.norm1, RMS_EPS)?, cos, sin, mask, nh)?;
        let x = (x + a)?;
        let m = self.mlp(&rms_norm(&x, &self.norm2, RMS_EPS)?)?;
        Ok((&x + m)?)
    }
}

/// Patch merger: `ln_q` RMSNorm → concat merge-unit groups → `Linear → GELU → Linear`.
struct Merger {
    ln_q: Tensor,
    mlp0: Linear,
    mlp2: Linear,
}

impl Merger {
    fn new(vb: &VarBuilder) -> CResult<Self> {
        Ok(Self {
            ln_q: vb.get_unchecked("ln_q.weight")?,
            mlp0: lin_bias(&vb.pp("mlp"), "0")?,
            mlp2: lin_bias(&vb.pp("mlp"), "2")?,
        })
    }

    /// `x` `[seq, context_dim]` → `[merged, out_hidden]` (`merged = seq / merge_unit`).
    fn forward(&self, x: &Tensor, merged: usize, merge_dim: usize) -> CResult<Tensor> {
        let x = rms_norm(x, &self.ln_q, RMS_EPS)?.reshape((merged, merge_dim))?;
        let x = self.mlp0.forward(&x)?.gelu_erf()?;
        Ok(self.mlp2.forward(&x)?)
    }
}

/// Host-side `grid_thw`-derived plan: the window permutation, its inverse, the f32 rope table (original
/// merge-unit order), and the per-token block ids for the full + windowed additive masks.
struct Plan {
    seq: usize,
    merged: usize,
    window_index: Tensor,  // u32 [merged]
    reverse_index: Tensor, // u32 [merged]
    rope: Tensor,          // f32 [seq, head_dim/2] (original order)
    full_bid: Vec<i32>,    // [seq]
    win_bid: Vec<i32>,     // [seq]
}

/// Remove consecutive duplicates (`torch.unique_consecutive`).
fn dedup_consecutive(v: &[i32]) -> Vec<i32> {
    let mut out = Vec::with_capacity(v.len());
    for &x in v {
        if out.last() != Some(&x) {
            out.push(x);
        }
    }
    out
}

/// Per-token block id from cumulative boundaries `cu` (`[0, b1, …, seq]`): token `p` ∈ `[cu[k],cu[k+1])`
/// → `k`. Mirrors the reference's diagonal-block mask construction.
fn block_ids(cu: &[i32], seq: usize) -> Vec<i32> {
    let mut bid = vec![0i32; seq];
    for k in 0..cu.len().saturating_sub(1) {
        for p in cu[k]..cu[k + 1] {
            bid[p as usize] = k as i32;
        }
    }
    bid
}

/// Build an additive attention mask `[1, seq, seq]` (f32): `0` within a block, `-inf` across blocks.
fn additive_mask(
    bid: &[i32],
    seq: usize,
    dev: &candle_gen::candle_core::Device,
) -> CResult<Tensor> {
    let mut data = vec![0f32; seq * seq];
    for i in 0..seq {
        for j in 0..seq {
            if bid[i] != bid[j] {
                data[i * seq + j] = f32::NEG_INFINITY;
            }
        }
    }
    Ok(Tensor::from_vec(data, (1, seq, seq), dev)?)
}

/// The native Qwen2.5-VL vision tower.
pub struct VisionTower {
    patch_embed: Linear,
    blocks: Vec<Block>,
    merger: Merger,
    cfg: VisionConfig,
    /// The tower's weight/activation dtype (bf16 in production via `PLANNER_DTYPE`, f32 in the parity
    /// fixture). Every runtime input — pixels, RoPE `cos`/`sin`, the additive masks — is cast to this
    /// so candle's dtype-strict matmul/binary ops don't hard-error on a mixed-dtype forward (sc-11150).
    dtype: DType,
}

impl VisionTower {
    /// Build from a `VarBuilder` rooted at the vision namespace (`visual` for the snapshot layout).
    pub fn new(cfg: VisionConfig, vb: VarBuilder) -> CResult<Self> {
        // Fold the bias-free Conv3d weight `[embed, in, t, ph, pw]` → `[embed, in·t·ph·pw]` so the
        // full-kernel conv runs as a per-patch matmul.
        let conv = vb.get_unchecked("patch_embed.proj.weight")?;
        let dtype = conv.dtype();
        let embed = conv.dim(0)?;
        let in_dim: usize = conv.dims().iter().skip(1).product();
        let patch_embed = Linear::new(conv.reshape((embed, in_dim))?, None);

        let bvb = vb.pp("blocks");
        let blocks = (0..cfg.depth)
            .map(|i| Block::new(&bvb.pp(i)))
            .collect::<CResult<Vec<_>>>()?;
        let merger = Merger::new(&vb.pp("merger"))?;
        Ok(Self {
            patch_embed,
            blocks,
            merger,
            cfg,
            dtype,
        })
    }

    pub fn config(&self) -> &VisionConfig {
        &self.cfg
    }

    /// Compute the `grid_thw`-derived plan host-side (faithful to `rot_pos_emb` + `get_window_index` +
    /// the `forward` cu-seqlen logic). `grid_thw` rows are `[t, h, w]` in patches.
    fn build_plan(
        &self,
        grid: &[[usize; 3]],
        dev: &candle_gen::candle_core::Device,
    ) -> CResult<Plan> {
        let c = &self.cfg;
        let sms = c.spatial_merge_size;
        let mu = c.merge_unit();
        let vmws = c.vit_merger_window_size();
        let rd = c.head_dim() / 2; // rope width per token
        let half = rd / 2; // inv_freq length = head_dim/4
        let inv: Vec<f64> = (0..half)
            .map(|j| 1.0 / ROPE_THETA.powf((2 * j) as f64 / rd as f64))
            .collect();

        let mut rope_rows: Vec<f32> = Vec::new(); // [seq, rd], original merge-unit order
        let mut window_index: Vec<i64> = Vec::new();
        let mut cu_window: Vec<i32> = vec![0];
        let mut cu_seqlens: Vec<i32> = vec![0];
        let mut window_index_id: i64 = 0;

        for g in grid {
            let (t, h, w) = (g[0], g[1], g[2]);
            let (llm_h, llm_w) = (h / sms, w / sms);

            // rope/pos in merge-grouped order (`rot_pos_emb`), repeated over t frames.
            for _f in 0..t {
                for br in 0..llm_h {
                    for bc in 0..llm_w {
                        for ir in 0..sms {
                            for ic in 0..sms {
                                let hpos = (br * sms + ir) as f64;
                                let wpos = (bc * sms + ic) as f64;
                                for &f in &inv {
                                    rope_rows.push((hpos * f) as f32);
                                }
                                for &f in &inv {
                                    rope_rows.push((wpos * f) as f32);
                                }
                            }
                        }
                    }
                }
            }

            // cu_seqlens (full): h*w patches per frame.
            for _f in 0..t {
                let last = *cu_seqlens.last().unwrap();
                cu_seqlens.push(last + (h * w) as i32);
            }

            // get_window_index: window-partitioned valid merge-unit indices + cu_window boundaries.
            let pad_h = vmws - llm_h % vmws; // can equal vmws when divisible (harmless 0-count window)
            let pad_w = vmws - llm_w % vmws;
            let nwh = (llm_h + pad_h) / vmws;
            let nww = (llm_w + pad_w) / vmws;
            let mut cu_prev = *cu_window.last().unwrap();
            for f in 0..t {
                for wh in 0..nwh {
                    for ww in 0..nww {
                        let mut count = 0i32;
                        for ir in 0..vmws {
                            for ic in 0..vmws {
                                let r = wh * vmws + ir;
                                let cc = ww * vmws + ic;
                                if r < llm_h && cc < llm_w {
                                    let val = (f * llm_h * llm_w + r * llm_w + cc) as i64;
                                    window_index.push(val + window_index_id);
                                    count += 1;
                                }
                            }
                        }
                        cu_prev += count * mu as i32; // cumsum(seqlens)*merge_unit + offset
                        cu_window.push(cu_prev);
                    }
                }
            }
            window_index_id += (t * llm_h * llm_w) as i64;
        }

        let merged = window_index.len();
        let seq = merged * mu;
        let cu_window = dedup_consecutive(&cu_window);

        // inverse permutation (`argsort(window_index)` for a permutation). Validate it is a genuine
        // permutation of `0..merged` and error on a malformed grid instead of an OOB panic (F-024).
        let mut reverse = vec![u32::MAX; merged];
        for (i, &wi) in window_index.iter().enumerate() {
            let wi = wi as usize;
            if wi >= reverse.len() || reverse[wi] != u32::MAX {
                return Err(CandleError::Msg(format!(
                    "bernini vision: window_index is not a valid permutation (index {wi} at \
                     position {i}, merged={merged})"
                )));
            }
            reverse[wi] = i as u32;
        }
        let window_u32: Vec<u32> = window_index.iter().map(|&x| x as u32).collect();

        Ok(Plan {
            seq,
            merged,
            window_index: Tensor::from_vec(window_u32, (merged,), dev)?,
            reverse_index: Tensor::from_vec(reverse, (merged,), dev)?,
            rope: Tensor::from_vec(rope_rows, (seq, rd), dev)?,
            full_bid: block_ids(&cu_seqlens, seq),
            win_bid: block_ids(&cu_window, seq),
        })
    }

    /// Encode packed patches → ViT tokens. `pixel_values` is `[sum_patches, in·t·ph·pw]`; `grid_thw`
    /// rows are `[t, h, w]` (patches). Returns `[sum_merged, out_hidden]` in the original (un-windowed)
    /// merge-unit order, where `sum_merged = Σ t·(h/merge)·(w/merge)`.
    pub fn forward(&self, pixel_values: &Tensor, grid_thw: &[[usize; 3]]) -> CResult<Tensor> {
        let c = &self.cfg;
        let mu = c.merge_unit();
        let dim = c.hidden_size;
        let nh = c.num_heads;
        let rd = c.head_dim() / 2;
        let dev = pixel_values.device();

        let plan = self.build_plan(grid_thw, dev)?;
        let (seq, merged) = (plan.seq, plan.merged);

        // Cast the f32 pixels to the tower's weight dtype (bf16 in production) — candle's matmul
        // hard-errors on mixed dtypes, so `patch_embed.forward` against bf16 weights needs bf16
        // activations (sc-11150).
        let pixel_values = pixel_values.to_dtype(self.dtype)?;

        // Patch embed, then reorder hidden + rope by the window permutation (merge-unit granularity).
        let h = self.patch_embed.forward(&pixel_values)?; // [seq, dim]
        let h = h
            .reshape((merged, mu, dim))?
            .index_select(&plan.window_index, 0)?
            .reshape((seq, dim))?;
        let rope = plan
            .rope
            .reshape((merged, mu, rd))?
            .index_select(&plan.window_index, 0)?
            .reshape((seq, rd))?;
        // Build the rotary tables in f32 for precision, then cast `cos`/`sin` to the activation dtype
        // so the per-block `broadcast_mul` against bf16 q/k does not fault (sc-11150).
        let emb = Tensor::cat(&[&rope, &rope], 1)?; // [seq, head_dim] f32
        let cos = emb.cos()?.to_dtype(self.dtype)?;
        let sin = emb.sin()?.to_dtype(self.dtype)?;

        // The additive masks are built f32 (0 / -inf); cast to the activation dtype so `broadcast_add`
        // onto the bf16 attention scores matches (sc-11150). f32 -inf casts to a bf16 -inf.
        let full_mask = additive_mask(&plan.full_bid, seq, dev)?.to_dtype(self.dtype)?;
        let win_mask = additive_mask(&plan.win_bid, seq, dev)?.to_dtype(self.dtype)?;

        let mut h = h;
        for (i, blk) in self.blocks.iter().enumerate() {
            let mask = if c.fullatt_block_indexes.contains(&i) {
                &full_mask
            } else {
                &win_mask
            };
            h = blk.forward(&h, &cos, &sin, mask, nh)?;
        }

        // Merge + undo the window permutation.
        let h = self.merger.forward(&h, merged, dim * mu)?;
        Ok(h.index_select(&plan.reverse_index, 0)?)
    }
}

/// `get_vit_features`: split the concatenated tower output `[Σ merged, out_hidden]` back into one
/// `[merged, out_hidden]` chunk per grid, by `t·h·w / merge²` (the reference's
/// `torch.split(image_embeds, grid.prod(-1) // merge²)`).
pub fn split_vit_features(
    embeds: &Tensor,
    grids: &[[usize; 3]],
    merge: usize,
) -> CResult<Vec<Tensor>> {
    let m2 = merge * merge;
    let sizes: Vec<usize> = grids.iter().map(|g| g[0] * g[1] * g[2] / m2).collect();
    if sizes.len() <= 1 {
        return Ok(vec![embeds.clone()]);
    }
    let mut out = Vec::with_capacity(sizes.len());
    let mut off = 0usize;
    for s in &sizes {
        out.push(embeds.narrow(0, off, *s)?);
        off += s;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::{DType, Device};

    /// split_vit_features chunks by per-grid merged-token count.
    #[test]
    fn vit_feature_split() {
        // two grids: (1,4,6)->6 merged, (1,4,4)->4 merged; total 10 rows.
        let embeds = Tensor::zeros((10, 8), DType::F32, &Device::Cpu).unwrap();
        let chunks = split_vit_features(&embeds, &[[1, 4, 6], [1, 4, 4]], 2).unwrap();
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].dims(), &[6, 8]);
        assert_eq!(chunks[1].dims(), &[4, 8]);
    }

    /// `vit_merger_window_size = window // merge // patch` and `head_dim` / `merge_unit` derivations
    /// match Qwen2.5-VL-7B.
    #[test]
    fn config_derivations() {
        let c = VisionConfig::default();
        assert_eq!(c.head_dim(), 80);
        assert_eq!(c.merge_unit(), 4);
        assert_eq!(c.vit_merger_window_size(), 4); // 112 / 2 / 14
    }

    /// rotate_half is the NeoX half-split: `[a,b,c,d] → [-c,-d,a,b]`.
    #[test]
    fn rotate_half_neox() {
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 4), &Device::Cpu).unwrap();
        let r = rotate_half(&x).unwrap();
        let got: Vec<f32> = r.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(got, vec![-3.0, -4.0, 1.0, 2.0]);
    }

    /// dedup_consecutive collapses runs (mirrors `torch.unique_consecutive`).
    #[test]
    fn dedup_runs() {
        assert_eq!(
            dedup_consecutive(&[0, 16, 24, 32, 36, 36, 36]),
            vec![0, 16, 24, 32, 36]
        );
        assert_eq!(dedup_consecutive(&[0, 0, 5]), vec![0, 5]);
    }

    /// block_ids partitions positions into the diagonal blocks named by `cu`.
    #[test]
    fn block_ids_partition() {
        // two images: [0,4) and [4,7).
        assert_eq!(block_ids(&[0, 4, 7], 7), vec![0, 0, 0, 0, 1, 1, 1]);
    }
}
