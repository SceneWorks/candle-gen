//! Qwen2.5-VL **vision transformer** — candle (Windows/CUDA) port of `mlx-gen-qwen-image`'s
//! `text_encoder/vision/`. The image branch of the VL encoder used by Qwen-Image-Edit: a patch-embed
//! (full-window Conv3d, here folded to a matmul) → 32 pre-norm blocks (windowed attention, with full
//! attention at blocks `[7,15,23,31]`) → patch merger, producing vision embeds that get spliced into
//! the text stream ([`crate::vision_language`]).
//!
//! Runs in **f32** (like the rest of the candle Qwen encoder) — the mlx provider keeps it bf16, but
//! the candle lane validates functionally, not bit-exact-vs-mlx, and the tower is small (32 blocks,
//! 1280-wide, ~750 patches) so f32 is cheap and more accurate.
//!
//! The RoPE here is the **non-interleaved** `rotate_half` form, which is exactly
//! [`candle_nn::rotary_emb::rope`] (NeoX half-split) — distinct from the MMDiT's interleaved RoPE.

use candle_gen::candle_core::{Device, Tensor};
use candle_gen::candle_nn::ops::softmax_last_dim;
use candle_gen::candle_nn::rotary_emb::rope;
use candle_gen::candle_nn::{linear, rms_norm, Linear, Module, RmsNorm, VarBuilder};
use candle_gen::Result;

/// Qwen2.5-VL vision RMSNorm epsilon (fork default).
const EPS: f64 = 1e-6;

/// `(grid_t, grid_h, grid_w)` for one image, in **patch** units.
pub type Grid = [i32; 3];

// ---------------------------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------------------------

/// Vision-transformer config. `mlp_hidden` and `out_hidden_size` (the merger output) are
/// pre-resolved. Mirrors the mlx `VisionConfig`.
#[derive(Clone, Debug)]
pub struct VisionConfig {
    pub patch_size: i32,
    pub temporal_patch_size: i32,
    pub in_channels: i32,
    pub embed_dim: i32,
    pub depth: i32,
    pub num_heads: i32,
    pub mlp_hidden: i32,
    pub out_hidden_size: i32,
    pub spatial_merge_size: i32,
    pub window_size: i32,
    pub fullatt_block_indexes: Vec<i32>,
    pub rope_theta: f32,
}

impl VisionConfig {
    /// The Qwen-Image-Edit-2511 `vision_config` (depth 32, embed 1280, 16 heads × 80,
    /// mlp_ratio 2.671875 → 3420, out 3584, window 112, full-attn at `[7,15,23,31]`).
    pub fn qwen_image_edit() -> Self {
        Self {
            patch_size: 14,
            temporal_patch_size: 2,
            in_channels: 3,
            embed_dim: 1280,
            depth: 32,
            num_heads: 16,
            mlp_hidden: 3420,
            out_hidden_size: 3584,
            spatial_merge_size: 2,
            window_size: 112,
            fullatt_block_indexes: vec![7, 15, 23, 31],
            rope_theta: 10000.0,
        }
    }

    pub fn head_dim(&self) -> i32 {
        self.embed_dim / self.num_heads
    }

    /// Vision RoPE dim = `head_dim / 2`.
    fn rope_dim(&self) -> i32 {
        self.head_dim() / 2
    }
}

// ---------------------------------------------------------------------------------------------
// Grid math (pure functions of `grid_thw` + config — no weights). Port of the mlx `grid` module.
// ---------------------------------------------------------------------------------------------

/// Port of `VisionTransformer.get_window_index` + the consecutive-dedup the fork applies in
/// `__call__`. Returns `(window_index, cu_window_seqlens)`:
///
/// - `window_index`: a permutation over the merge-groups that gathers each `merger_window²` block
///   contiguously (padding entries dropped).
/// - `cu_window_seqlens`: cumulative **patch** counts at window boundaries (deduped).
pub fn window_index(
    grids: &[Grid],
    merge: i32,
    window_size: i32,
    patch_size: i32,
) -> (Vec<i32>, Vec<i32>) {
    let merge_unit = merge * merge;
    let vmw = window_size / patch_size / merge; // merger-space window edge (4)

    let mut window_index: Vec<i32> = Vec::new();
    let mut cu_window: Vec<i32> = vec![0];
    let mut window_index_id = 0i32;

    for &[t, gh, gw] in grids {
        let llm_h = gh / merge;
        let llm_w = gw / merge;
        let pad_h = vmw - llm_h % vmw;
        let pad_w = vmw - llm_w % vmw;
        let nwh = (llm_h + pad_h) / vmw;
        let nww = (llm_w + pad_w) / vmw;

        for ti in 0..t {
            let plane = ti * llm_h * llm_w;
            for wh in 0..nwh {
                for ww in 0..nww {
                    let mut seqlen = 0i32;
                    for r in 0..vmw {
                        for c in 0..vmw {
                            let i = wh * vmw + r;
                            let j = ww * vmw + c;
                            if i < llm_h && j < llm_w {
                                window_index.push(window_index_id + plane + i * llm_w + j);
                                seqlen += 1;
                            }
                        }
                    }
                    let last = *cu_window.last().unwrap();
                    cu_window.push(last + seqlen * merge_unit);
                }
            }
        }
        window_index_id += t * llm_h * llm_w;
    }

    // Dedup consecutive-equal (drops all-padding windows' zero-length contributions).
    let mut cu_dedup = vec![cu_window[0]];
    for &v in &cu_window[1..] {
        if v != *cu_dedup.last().unwrap() {
            cu_dedup.push(v);
        }
    }
    (window_index, cu_dedup)
}

/// Full-attention cumulative seqlens: `[0, cumulative t·h·w per image]` (patch units).
pub fn cu_seqlens(grids: &[Grid]) -> Vec<i32> {
    let mut out = vec![0];
    let mut offset = 0;
    for &[t, h, w] in grids {
        offset += t * h * w;
        out.push(offset);
    }
    out
}

/// Port of `VisionTransformer.rot_pos_emb`: the 2-D vision RoPE position table `[seq, rope_dim]`
/// (each row `[h_freqs(rope_dim/2) ‖ w_freqs(rope_dim/2)]`) in the spatial-merge layout, **before**
/// the window reorder. Exact integer-driven math, built in f32 in plain Rust.
fn rot_pos_emb_table(grids: &[Grid], merge: i32, rope_dim: i32, theta: f32) -> Vec<f32> {
    let half = (rope_dim / 2) as usize; // 20
    let inv_freq: Vec<f32> = (0..half)
        .map(|k| 1.0 / theta.powf((2 * k) as f32 / rope_dim as f32))
        .collect();

    let mut data: Vec<f32> = Vec::new();
    for &[t, h, w] in grids {
        let merge_h = h / merge;
        let merge_w = w / merge;
        for _ti in 0..t {
            for a in 0..merge_h {
                for c in 0..merge_w {
                    for b in 0..merge {
                        for d in 0..merge {
                            let hpos = (a * merge + b) as f32;
                            let wpos = (c * merge + d) as f32;
                            for &f in &inv_freq {
                                data.push(hpos * f);
                            }
                            for &f in &inv_freq {
                                data.push(wpos * f);
                            }
                        }
                    }
                }
            }
        }
    }
    data
}

// ---------------------------------------------------------------------------------------------
// Weight-bearing modules
// ---------------------------------------------------------------------------------------------

/// `VisionPatchEmbed`: a bias-less full-window Conv3d folded to a matmul. The PyTorch conv weight
/// `[embed, in, kD, kH, kW]` flattens (in the on-disk `[in, kD, kH, kW]` order) to `[embed, 1176]`;
/// the `pixel_values` last dim is in the same `(channel, temporal, patch_y, patch_x)` order
/// ([`crate::image_processor`]), so `pixel_values · weightᵀ` equals the kernel==stride==full-window
/// conv exactly.
struct VisionPatchEmbed {
    /// `[1176, embed]` (the transposed flattened conv weight).
    proj_t: Tensor,
}

impl VisionPatchEmbed {
    fn new(vb: VarBuilder, cfg: &VisionConfig) -> Result<Self> {
        let (e, ic, t, p) = (
            cfg.embed_dim as usize,
            cfg.in_channels as usize,
            cfg.temporal_patch_size as usize,
            cfg.patch_size as usize,
        );
        let flat = ic * t * p * p;
        let w = vb
            .get((e, ic, t, p, p), "proj.weight")?
            .reshape((e, flat))?;
        Ok(Self {
            proj_t: w.t()?.contiguous()?,
        })
    }

    /// `pixel_values`: `[n, 1176]` → `[n, embed]`.
    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        Ok(pixel_values.matmul(&self.proj_t)?)
    }
}

/// `VisionMlp`: a biased SwiGLU (`down(silu(gate(x)) · up(x))`).
struct VisionMlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl VisionMlp {
    fn new(vb: VarBuilder, embed: usize, hidden: usize) -> Result<Self> {
        Ok(Self {
            gate: linear(embed, hidden, vb.pp("gate_proj"))?,
            up: linear(embed, hidden, vb.pp("up_proj"))?,
            down: linear(hidden, embed, vb.pp("down_proj"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate.forward(x)?.silu()?;
        let up = self.up.forward(x)?;
        Ok(self.down.forward(&(gate * up)?)?)
    }
}

/// `VisionAttention`: biased fused `qkv` → per-head split → `rotate_half` 2-D RoPE → block-diagonal
/// SDPA (one chunk per window/image, via `cu`) → biased `proj`.
struct VisionAttention {
    qkv: Linear,
    proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl VisionAttention {
    fn new(vb: VarBuilder, embed: usize, num_heads: usize) -> Result<Self> {
        let head_dim = embed / num_heads;
        Ok(Self {
            qkv: linear(embed, 3 * embed, vb.pp("qkv"))?,
            proj: linear(embed, embed, vb.pp("proj"))?,
            num_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
        })
    }

    /// `x`: `[seq, embed]`; `cos`/`sin`: `[seq, head_dim/2]`; `cu`: cumulative seqlens for this
    /// block's attention windows (`[0, …, seq]`). `cu.len() > 2` ⇒ block-diagonal (windowed).
    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, cu: &[i32]) -> Result<Tensor> {
        let seq = x.dims()[0];
        let (h, hd) = (self.num_heads, self.head_dim);
        let qkv = self.qkv.forward(x)?.reshape((seq, 3, h, hd))?;
        // each → [1, h, seq, hd]
        let to_heads = |idx: usize| -> Result<Tensor> {
            Ok(qkv
                .narrow(1, idx, 1)?
                .squeeze(1)? // [seq, h, hd]
                .transpose(0, 1)? // [h, seq, hd]
                .unsqueeze(0)? // [1, h, seq, hd]
                .contiguous()?)
        };
        let q = rope(&to_heads(0)?, cos, sin)?;
        let k = rope(&to_heads(1)?, cos, sin)?;
        let v = to_heads(2)?;

        // Block-diagonal SDPA: one attention per window/image, concatenated over the seq axis.
        let mut outs: Vec<Tensor> = Vec::with_capacity(cu.len().saturating_sub(1));
        for w in cu.windows(2) {
            let (off, len) = (w[0] as usize, (w[1] - w[0]) as usize);
            if len == 0 {
                continue;
            }
            let qc = q.narrow(2, off, len)?.contiguous()?;
            let kc = k.narrow(2, off, len)?.contiguous()?;
            let vc = v.narrow(2, off, len)?.contiguous()?;
            let scores = (qc.matmul(&kc.transpose(2, 3)?.contiguous()?)? * self.scale)?;
            let probs = softmax_last_dim(&scores)?;
            outs.push(probs.matmul(&vc)?); // [1, h, len, hd]
        }
        let attn = Tensor::cat(&outs, 2)?; // [1, h, seq, hd]
        let out = attn
            .squeeze(0)? // [h, seq, hd]
            .transpose(0, 1)? // [seq, h, hd]
            .contiguous()?
            .reshape((seq, h * hd))?;
        Ok(self.proj.forward(&out)?)
    }
}

/// `VisionBlock`: pre-norm residual — `x += attn(rms(x)); x += mlp(rms(x))`, RMSNorm(ε=1e-6).
struct VisionBlock {
    norm1: RmsNorm,
    norm2: RmsNorm,
    attn: VisionAttention,
    mlp: VisionMlp,
}

impl VisionBlock {
    fn new(vb: VarBuilder, cfg: &VisionConfig) -> Result<Self> {
        let embed = cfg.embed_dim as usize;
        Ok(Self {
            norm1: rms_norm(embed, EPS, vb.pp("norm1"))?,
            norm2: rms_norm(embed, EPS, vb.pp("norm2"))?,
            attn: VisionAttention::new(vb.pp("attn"), embed, cfg.num_heads as usize)?,
            mlp: VisionMlp::new(vb.pp("mlp"), embed, cfg.mlp_hidden as usize)?,
        })
    }

    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, cu: &[i32]) -> Result<Tensor> {
        let x = (x + self.attn.forward(&self.norm1.forward(x)?, cos, sin, cu)?)?;
        let out = (&x + self.mlp.forward(&self.norm2.forward(&x)?)?)?;
        Ok(out)
    }
}

/// `PatchMerger`: RMSNorm(`ln_q`) → group each `merge²` patches into one row → `mlp_0` → GELU →
/// `mlp_1`, mapping `embed → out_hidden`. The windowed hidden states are already `merge²`-grouped
/// (image-contiguous), so a single global `[-1, embed·merge²]` reshape matches the fork.
struct PatchMerger {
    ln_q: RmsNorm,
    mlp0: Linear,
    mlp1: Linear,
    hidden_merged: usize,
}

impl PatchMerger {
    fn new(vb: VarBuilder, cfg: &VisionConfig) -> Result<Self> {
        let embed = cfg.embed_dim as usize;
        let merge2 = (cfg.spatial_merge_size * cfg.spatial_merge_size) as usize;
        let hidden_merged = embed * merge2;
        let out = cfg.out_hidden_size as usize;
        Ok(Self {
            ln_q: rms_norm(embed, EPS, vb.pp("ln_q"))?,
            // The diffusers `Sequential` is `mlp.0` (Linear) / `mlp.1` (GELU) / `mlp.2` (Linear).
            mlp0: linear(hidden_merged, hidden_merged, vb.pp("mlp").pp("0"))?,
            mlp1: linear(hidden_merged, out, vb.pp("mlp").pp("2"))?,
            hidden_merged,
        })
    }

    /// `x`: `[seq, embed]` (window-reordered) → `[seq/merge², out_hidden]`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let rows = x.elem_count() / self.hidden_merged;
        let x = self.ln_q.forward(x)?.reshape((rows, self.hidden_merged))?;
        let x = self.mlp0.forward(&x)?.gelu_erf()?;
        Ok(self.mlp1.forward(&x)?)
    }
}

// ---------------------------------------------------------------------------------------------
// Assembly
// ---------------------------------------------------------------------------------------------

pub struct VisionTransformer {
    patch_embed: VisionPatchEmbed,
    blocks: Vec<VisionBlock>,
    merger: PatchMerger,
    cfg: VisionConfig,
    device: Device,
}

impl VisionTransformer {
    /// Build under the `visual.*` prefix (the on-disk diffusers names).
    pub fn new(vb: VarBuilder, cfg: &VisionConfig) -> Result<Self> {
        let device = vb.device().clone();
        let visual = vb.pp("visual");
        // Multi-modal guard (sc-9415): the Qwen-Image MLX tiers keep the vision tower **dense bf16**
        // (the convert job packs only the DiT; the q4/q8 TE index carries 0 `.scales` across all 390
        // `visual.*` keys). This loader therefore reads `visual.*.weight` as f32. If a future tier ever
        // packed the vision tower, its u32 codes would be silently read as bf16 garbage — so error
        // loudly on an unexpected `.scales` sibling (a representative attn projection) rather than
        // render noise. A packed vision tower must add a real packed path, not fall through.
        crate::quant::guard_dense(&visual, "blocks.0.attn.qkv")?;
        let patch_embed = VisionPatchEmbed::new(visual.pp("patch_embed"), cfg)?;
        let blocks_vb = visual.pp("blocks");
        let blocks = (0..cfg.depth)
            .map(|i| VisionBlock::new(blocks_vb.pp(i as usize), cfg))
            .collect::<Result<Vec<_>>>()?;
        let merger = PatchMerger::new(visual.pp("merger"), cfg)?;
        Ok(Self {
            patch_embed,
            blocks,
            merger,
            cfg: cfg.clone(),
            device,
        })
    }

    /// `pixel_values`: `[seq_patches, 1176]` (f32, on the device); `grids`: one `(t, gh, gw)` per
    /// image (patch units). Returns vision embeds `[seq_patches/merge², out_hidden]` (f32).
    pub fn forward(&self, pixel_values: &Tensor, grids: &[Grid]) -> Result<Tensor> {
        let cfg = &self.cfg;
        let merge = cfg.spatial_merge_size;
        let merge_unit = (merge * merge) as usize;
        let embed = cfg.embed_dim as usize;
        let rope_dim = cfg.rope_dim();

        let hidden = self.patch_embed.forward(pixel_values)?; // [seq, embed], f32
        let seq = hidden.dims()[0];
        let num_groups = seq / merge_unit;

        let (wi, cu_window) = window_index(grids, merge, cfg.window_size, cfg.patch_size);
        let cu_full = cu_seqlens(grids);

        // Window-reorder hidden at the merge-group level.
        let wi_u32: Vec<u32> = wi.iter().map(|&g| g as u32).collect();
        let wi_t = Tensor::from_vec(wi_u32, num_groups, &self.device)?;
        let hidden = hidden
            .reshape((num_groups, merge_unit, embed))?
            .index_select(&wi_t, 0)?
            .reshape((seq, embed))?;

        // Build the 2-D RoPE table, window-reorder it (group level), → cos/sin `[seq, rope_dim/... ]`.
        let table = rot_pos_emb_table(grids, merge, rope_dim, cfg.rope_theta); // [seq, rope_dim]
        let rd = rope_dim as usize;
        let mut reordered = vec![0f32; seq * rd];
        for (k, &g) in wi.iter().enumerate() {
            for j in 0..merge_unit {
                let src = ((g as usize) * merge_unit + j) * rd;
                let dst = (k * merge_unit + j) * rd;
                reordered[dst..dst + rd].copy_from_slice(&table[src..src + rd]);
            }
        }
        let rope_t = Tensor::from_vec(reordered, (seq, rd), &self.device)?;
        let cos = rope_t.cos()?;
        let sin = rope_t.sin()?;

        let mut h = hidden;
        for (i, block) in self.blocks.iter().enumerate() {
            let cu = if cfg.fullatt_block_indexes.contains(&(i as i32)) {
                &cu_full
            } else {
                &cu_window
            };
            h = block.forward(&h, &cos, &sin, cu)?;
        }

        let h = self.merger.forward(&h)?; // [num_groups, out_hidden]

        // Reverse the window reorder (inverse permutation).
        let mut reverse = vec![0u32; num_groups];
        for (k, &g) in wi.iter().enumerate() {
            reverse[g as usize] = k as u32;
        }
        let rev_t = Tensor::from_vec(reverse, num_groups, &self.device)?;
        Ok(h.index_select(&rev_t, 0)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cu_seqlens_accumulates() {
        assert_eq!(cu_seqlens(&[[1, 4, 6]]), vec![0, 24]);
        assert_eq!(cu_seqlens(&[[1, 4, 6], [1, 2, 2]]), vec![0, 24, 28]);
    }

    #[test]
    fn window_index_is_a_permutation_and_groups_windows() {
        // A 4x6 patch grid, merge=2 → llm grid 2x3 = 6 merge-groups. window edge vmw=4 covers the
        // whole llm grid → a single window → window_index is the identity permutation [0..6).
        let (wi, cu) = window_index(&[[1, 4, 6]], 2, 112, 14);
        let mut sorted = wi.clone();
        sorted.sort_unstable();
        assert_eq!(
            sorted,
            (0..6).collect::<Vec<_>>(),
            "must be a permutation of the groups"
        );
        // cu_window is in **patch** units: 6 groups × merge²(4) = 24 patches in the single window.
        assert_eq!(*cu.last().unwrap(), 24);
    }

    #[test]
    fn rope_table_shape_and_values() {
        // 2x2 patch grid, merge=2 → 1 merge-group, rope_dim=4 (half=2). One row of [h_freqs, w_freqs].
        let t = rot_pos_emb_table(&[[1, 2, 2]], 2, 4, 10000.0);
        assert_eq!(t.len(), 4 * 4); // seq=4 patches × rope_dim=4
                                    // First patch (a=c=b=d=0) → hpos=wpos=0 → all-zero row.
        assert!(t[..4].iter().all(|&x| x == 0.0));
    }
}
