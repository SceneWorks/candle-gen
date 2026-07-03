//! `PixDiT_T2I` backbone forward — the base text-to-image PixelDiT that `PidNet` (the LQ
//! super-resolution variant) inherits. Dual-stream MMDiT patch blocks + per-pixel PiT blocks, 2-D NTK
//! image RoPE + 1-D text RoPE, sinusoidal timestep conditioning, unfold/fold patchify. Faithful port
//! of `PixDiT_T2I.forward` (the released SR students set `enable_ed=False`, so the inference forward is
//! this clean no-encoder-decoder path). Runs f32 throughout (the parity target).

mod blocks;
mod layers;
mod rope;

use candle_gen::candle_core::Tensor;
use candle_gen::{Result, Weights};

use crate::config::PidConfig;
use blocks::{MMDiTBlockT2I, PiTBlock};
use layers::{
    fold_patches, unfold_patches, FinalLayer, PatchTokenEmbedder, PixelTokenEmbedder,
    TimestepConditioner,
};
use rope::{rope_1d_text, rope_2d_ntk};

// The pure host-side positional math is exposed so parity tests can gate it directly.
pub use layers::sincos_2d_pos;
pub use rope::{rope_1d_text as text_rope_table, rope_2d_ntk as image_rope_table};

const ROPE_THETA: f32 = 10000.0;
const ROPE_SCALE: f32 = 16.0;

/// A hook called before each patch block with `(block_idx, s_main)`, returning the (possibly gated)
/// `s_main`. `PidNet`'s sigma-aware LQ adapter implements this to inject the controlnet-style gate
/// between patch blocks; the base T2I forward passes none.
pub trait PatchInjector {
    fn inject(&self, block_idx: i32, s_main: &Tensor) -> Result<Tensor>;
}

/// The `PixDiT_T2I` backbone.
pub struct PixDiT {
    pixel_embedder: PixelTokenEmbedder,
    s_embedder: PatchTokenEmbedder,
    t_embedder: TimestepConditioner,
    y_embedder: PatchTokenEmbedder,
    y_pos_embedding: Tensor,
    patch_blocks: Vec<MMDiTBlockT2I>,
    pixel_blocks: Vec<PiTBlock>,
    final_layer: FinalLayer,
    cfg: PidConfig,
}

/// Slice the `[B, S, …]` axis-1 prefix `[:, :n]` (no-op when `S == n`).
fn prefix_axis1(a: &Tensor, n: usize) -> Result<Tensor> {
    if a.dim(1)? == n {
        return Ok(a.clone());
    }
    Ok(a.narrow(1, 0, n)?)
}

impl PixDiT {
    /// `prefix` is `""` for a bare-key fixture or `"net."` for the released checkpoint's nesting.
    pub fn from_weights(w: &Weights, prefix: &str, cfg: &PidConfig) -> Result<Self> {
        let patch_blocks = (0..cfg.patch_depth)
            .map(|i| {
                MMDiTBlockT2I::from_weights(
                    w,
                    &format!("{prefix}patch_blocks.{i}"),
                    cfg.hidden_size,
                    cfg.num_groups,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let pixel_blocks = (0..cfg.pixel_depth)
            .map(|i| {
                PiTBlock::from_weights(
                    w,
                    &format!("{prefix}pixel_blocks.{i}"),
                    cfg.pixel_hidden_size,
                    cfg.pixel_attn_hidden_size,
                    cfg.pixel_num_groups,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            pixel_embedder: PixelTokenEmbedder::from_weights(
                w,
                &format!("{prefix}pixel_embedder"),
                cfg.pixel_hidden_size,
            )?,
            s_embedder: PatchTokenEmbedder::from_weights(w, &format!("{prefix}s_embedder"))?,
            t_embedder: TimestepConditioner::from_weights(w, &format!("{prefix}t_embedder"))?,
            y_embedder: PatchTokenEmbedder::from_weights(w, &format!("{prefix}y_embedder"))?,
            y_pos_embedding: w.require(&format!("{prefix}y_pos_embedding"))?,
            patch_blocks,
            pixel_blocks,
            final_layer: FinalLayer::from_weights(w, &format!("{prefix}final_layer"))?,
            cfg: cfg.clone(),
        })
    }

    /// `x`: `[B, 3, H, W]`; `t`: `[B]`; `y`: `[B, Ltxt, txt_embed_dim]` (caption embeddings).
    /// Returns the predicted pixel tensor `[B, 3, H, W]`.
    pub fn forward(&self, x: &Tensor, t: &Tensor, y: &Tensor) -> Result<Tensor> {
        self.forward_inner(x, t, y, None)
    }

    /// Like [`Self::forward`] but with a per-patch-block injection hook — `PidNet` passes its
    /// sigma-aware LQ adapter here to gate `s_main` between blocks.
    pub fn forward_with(
        &self,
        x: &Tensor,
        t: &Tensor,
        y: &Tensor,
        injector: &dyn PatchInjector,
    ) -> Result<Tensor> {
        self.forward_inner(x, t, y, Some(injector))
    }

    fn forward_inner(
        &self,
        x: &Tensor,
        t: &Tensor,
        y: &Tensor,
        injector: Option<&dyn PatchInjector>,
    ) -> Result<Tensor> {
        let cfg = &self.cfg;
        let patch = cfg.patch_size;
        let (b, _, h, w) = x.dims4()?;
        let (hs, ws) = (h as i32 / patch, w as i32 / patch);
        let l = (hs * ws) as usize;
        let device = x.device();

        let x_patches = unfold_patches(x, patch)?;
        let t_emb = self
            .t_embedder
            .forward(t)?
            .reshape((b, 1, cfg.hidden_size as usize))?;

        let ltxt = y.dim(1)?.min(cfg.txt_max_length as usize);
        let y = prefix_axis1(y, ltxt)?;
        let y_emb = self.y_embedder.forward(&y)?;
        let y_pos = prefix_axis1(&self.y_pos_embedding, ltxt)?;
        let mut y_emb = y_emb.broadcast_add(&y_pos)?;

        let condition = t_emb.silu()?;
        let (cos_img, sin_img) = rope_2d_ntk(
            cfg.head_dim(),
            hs,
            ws,
            cfg.rope_ref_grid_h(),
            cfg.rope_ref_grid_w(),
            ROPE_THETA,
            ROPE_SCALE,
            device,
        )?;
        let (cos_txt, sin_txt) =
            rope_1d_text(cfg.head_dim(), ltxt as i32, cfg.text_rope_theta, device)?;

        let mut s_main = self.s_embedder.forward(&x_patches)?;
        for (i, blk) in self.patch_blocks.iter().enumerate() {
            if let Some(inj) = injector {
                s_main = inj.inject(i as i32, &s_main)?;
            }
            let (sx, sy) = blk.forward(
                &s_main, &y_emb, &condition, &cos_img, &sin_img, &cos_txt, &sin_txt,
            )?;
            s_main = sx;
            y_emb = sy;
        }
        let s = s_main.broadcast_add(&t_emb)?.silu()?;
        let s_cond = s.reshape((b * l, cfg.hidden_size as usize))?;

        let mut x_pixels = self.pixel_embedder.forward(x, h as i32, w as i32, patch)?;
        let (cos_pix, sin_pix) = rope_2d_ntk(
            cfg.pixel_head_dim(),
            hs,
            ws,
            cfg.rope_ref_grid_h(),
            cfg.rope_ref_grid_w(),
            ROPE_THETA,
            ROPE_SCALE,
            device,
        )?;
        for blk in &self.pixel_blocks {
            x_pixels = blk.forward(&x_pixels, &s_cond, &cos_pix, &sin_pix, b, l)?;
        }
        let x_pixels = self.final_layer.forward(&x_pixels)?;

        // [B*L, P2, C_out] -> [B, L, P2, C_out] -> [B, C_out, P2, L] -> fold -> [B, C_out, H, W]
        let c_out = cfg.in_channels;
        let p2 = (patch * patch) as usize;
        let xp = x_pixels
            .reshape((b, l, p2, c_out as usize))?
            .permute((0, 3, 2, 1))?
            .contiguous()?;
        fold_patches(&xp, c_out, hs, ws, patch)
    }
}
