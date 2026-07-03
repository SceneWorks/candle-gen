//! EVA02-CLIP-L-14-336 **visual** tower (sc-5492) — the candle (Windows/CUDA) twin of
//! `mlx-gen-pulid`'s `eva_clip`, a port of `eva_clip/eva_vit_model.py EVAVisionTransformer` (the
//! `.visual` submodule only; no text tower). PuLID-FLUX feeds the background-removed aligned face crop
//! through this to get the identity feature + the IDFormer's per-scale hidden states.
//!
//! Pipeline: `Conv2d` patch-embed → prepend CLS + add learned abs `pos_embed` → 24 sub-LN blocks
//! (interleaved 2-D RoPE on patch tokens, full SDPA, SwiGLU) → final LayerNorm → take CLS token →
//! `head` projection. Returns `id_cond_vit` (768-d) plus the 5 hidden states captured at the **input**
//! of blocks {4,8,12,16,20} (1024-d each) that the IDFormer consumes.
//!
//! Weights are the MLX-converted `convert_eva_clip.py` safetensors (bare names, no prefix) — shared
//! with the MLX sibling. The only layout fix vs MLX: the patch-embed conv is stored OHWI and candle's
//! `conv2d` is OIHW, so [`patch_embed`] transposes it at load (the candle face-stack convention). The
//! Linear/LayerNorm weights are `[out,in]`/`[C]` in both, so they load unchanged. This checkpoint has
//! `use_mean_pooling=False` (`visual.norm.*`, no `visual.fc_norm.*`) ⇒ the pooled feature is
//! `norm(x)[:, 0]` (CLS).

mod attention;
mod block;
mod mlp;
mod patch_embed;
pub mod rope;
pub mod transform;

use candle_core::Tensor;
use candle_nn::{LayerNorm, Linear, Module};

use candle_gen::weights::Weights;
use candle_gen::Result as GenResult;

use block::Block;
use patch_embed::PatchEmbed;
use rope::VisionRope;

/// EVA LayerNorm epsilon (`model.py` `partial(LayerNorm, eps=1e-6)`).
pub(crate) const EPS: f64 = 1e-6;

/// Join a dotted weight-key prefix with a leaf (`"" + leaf` ⇒ `leaf`).
pub(crate) fn join(prefix: &str, leaf: &str) -> String {
    if prefix.is_empty() {
        leaf.to_string()
    } else {
        format!("{prefix}.{leaf}")
    }
}

/// Build a [`LayerNorm`] (weight + bias, ε=1e-6) from `{prefix}.weight` / `{prefix}.bias`.
pub(crate) fn layer_norm(w: &Weights, prefix: &str) -> GenResult<LayerNorm> {
    Ok(LayerNorm::new(
        w.require(&format!("{prefix}.weight"))?,
        w.require(&format!("{prefix}.bias"))?,
        EPS,
    ))
}

/// EVA02-CLIP-L-14-336 visual-tower config.
#[derive(Clone, Debug)]
pub struct EvaConfig {
    pub image_size: usize,
    pub patch: usize,
    pub embed_dim: usize, // width = 1024
    pub depth: usize,
    pub num_heads: usize,
    pub proj_dim: usize, // head out (num_classes) = 768
    pub pt_seq_len: usize,
    pub rope_theta: f64,
    pub hidden_capture: Vec<usize>,
}

impl Default for EvaConfig {
    fn default() -> Self {
        Self {
            image_size: 336,
            patch: 14,
            embed_dim: 1024,
            depth: 24,
            num_heads: 16,
            proj_dim: 768,
            pt_seq_len: 16,
            rope_theta: 10000.0,
            hidden_capture: vec![4, 8, 12, 16, 20],
        }
    }
}

impl EvaConfig {
    pub fn head_dim(&self) -> usize {
        self.embed_dim / self.num_heads
    }
    pub fn grid(&self) -> usize {
        self.image_size / self.patch
    }
}

/// EVA visual tower output: the projected id feature + the captured intermediate hidden states.
pub struct EvaOutput {
    /// `[B, proj_dim]` (768) — the `head`-projected pooled feature (pre-L2-norm).
    pub id_cond_vit: Tensor,
    /// 5 × `[B, grid²+1, embed_dim]` (577×1024) — inputs of blocks {4,8,12,16,20}.
    pub hidden: Vec<Tensor>,
}

pub struct EvaVisionTransformer {
    patch_embed: PatchEmbed,
    cls_token: Tensor,
    pos_embed: Tensor,
    blocks: Vec<Block>,
    norm: LayerNorm,
    head: Linear,
    rope: VisionRope,
    cfg: EvaConfig,
}

impl EvaVisionTransformer {
    pub fn from_weights(w: &Weights, prefix: &str, cfg: EvaConfig) -> GenResult<Self> {
        let p = |leaf: &str| join(prefix, leaf);
        let (nh, hd) = (cfg.num_heads, cfg.head_dim());
        let cls_token = w.require(&p("cls_token"))?;
        let device = cls_token.device().clone();
        let blocks = (0..cfg.depth)
            .map(|i| Block::from_weights(w, &p(&format!("blocks.{i}")), nh, hd))
            .collect::<GenResult<Vec<_>>>()?;
        let rope = VisionRope::build(hd, cfg.grid(), cfg.pt_seq_len, cfg.rope_theta, &device)?;
        let head = Linear::new(
            w.require(&p("head.weight"))?,
            Some(w.require(&p("head.bias"))?),
        );
        Ok(Self {
            patch_embed: PatchEmbed::from_weights(w, &p("patch_embed"), cfg.patch, cfg.embed_dim)?,
            cls_token,
            pos_embed: w.require(&p("pos_embed"))?,
            blocks,
            norm: layer_norm(w, &p("norm"))?,
            head,
            rope,
            cfg,
        })
    }

    /// The tower's loaded [`EvaConfig`] (PuLID's uncond-embedding builder derives the token geometry —
    /// `grid²+1` sequence length, `embed_dim`, hidden-capture count — from this).
    pub fn config(&self) -> &EvaConfig {
        &self.cfg
    }

    /// `pixel_values`: **NCHW** `[B, 3, image_size, image_size]`, EVA-normalized.
    pub fn forward(&self, pixel_values: &Tensor) -> candle_core::Result<EvaOutput> {
        let mut x = self.patch_embed.forward(pixel_values)?; // [B, grid², embed]
        let b = x.dim(0)?;
        let cls = self.cls_token.broadcast_as((b, 1, self.cfg.embed_dim))?;
        x = Tensor::cat(&[&cls, &x], 1)?; // [B, grid²+1, embed]
        x = x.broadcast_add(&self.pos_embed)?;

        let mut hidden = Vec::with_capacity(self.cfg.hidden_capture.len());
        for (idx, blk) in self.blocks.iter().enumerate() {
            if self.cfg.hidden_capture.contains(&idx) {
                hidden.push(x.clone());
            }
            x = blk.forward(&x, &self.rope)?;
        }

        x = self.norm.forward(&x)?;
        // CLS token → head projection.
        let cls_tok = x.narrow(1, 0, 1)?.reshape((b, self.cfg.embed_dim))?;
        let id_cond_vit = self.head.forward(&cls_tok)?;
        Ok(EvaOutput {
            id_cond_vit,
            hidden,
        })
    }
}
