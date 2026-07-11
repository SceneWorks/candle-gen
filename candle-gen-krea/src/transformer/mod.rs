//! The Krea 2 dense single-stream DiT (`Krea2Transformer2DModel` / reference `mmdit.py`
//! `SingleStreamDiT`) forward. Port of `mlx-gen-krea`'s `transformer/mod.rs`.
//!
//! ```text
//!   img_in:        img tokens = Linear(patchify(latent, p=2))          [b, img_len, 6144]
//!   time_embed:    t   = Linear(GELU(Linear(sinusoid(timestep))))      [b, 1, 6144]
//!   time_mod_proj: tvec = Linear(GELU(t))                              [b, 1, 6·6144]   (shared modulation)
//!   text_fusion:   ctx = aggregate(stacked 12 Qwen3-VL layers)         [b, cap, 2560]
//!   txt_in:        ctx = Linear(GELU(Linear(RMSNorm(ctx))))            [b, cap, 6144]
//!   combined = [ctx ; img]                                            [b, cap+img_len, 6144]
//!   28× transformer_blocks (gated single-stream, DoubleSharedModulation, 3-axis RoPE)
//!   final_layer:   (1+scale)·RMSNorm(x) + shift → Linear(6144→64)      [b, cap+img_len, 64]
//!   slice image tokens → unpatchify                                   → velocity [b, 16, H, W]
//! ```
//!
//! Per-sample `B = 1`: the text stream arrives already trimmed to its valid length (the candle
//! tokenizer emits no padding) and the whole sequence runs **unmasked**.

pub mod block;
pub mod rope;

use candle_gen::candle_core::{DType, Device, Result, Tensor, D};

use crate::config::Krea2Config;
use crate::loader::{linear_detect, Weights};
use crate::quant::QLinear;
use block::{RmsScale, SingleStreamBlock, TextFusionTransformer};
use rope::RopeTables;

/// The Krea 2 single-stream DiT.
pub struct Krea2Transformer {
    cfg: Krea2Config,
    device: Device,
    dtype: DType,
    img_in: QLinear,
    time_embed_l1: QLinear,
    time_embed_l2: QLinear,
    time_mod_proj: QLinear,
    txt_in_norm: RmsScale,
    txt_in_l1: QLinear,
    txt_in_l2: QLinear,
    text_fusion: TextFusionTransformer,
    blocks: Vec<SingleStreamBlock>,
    final_norm: RmsScale,
    final_linear: QLinear,
    final_sstable: Tensor, // [1, 2, hidden]
    /// Per-render RoPE-table cache (sc-8992 / F-012). The joint `(cos, sin)` table depends only on the
    /// fixed geometry `(cap_len, ht, wt)` — not on the flow time / the current latent — so it is
    /// identical across every denoise step. Cache it keyed on that geometry; hits Arc-clone the stored
    /// handles. `Mutex` (not `RefCell`): the DiT is shared as `Arc<Krea2Transformer>` (`Send + Sync`).
    rope_cache: std::sync::Mutex<Option<KreaRopeCache>>,
}

struct KreaRopeCache {
    cap_len: usize,
    ht: usize,
    wt: usize,
    /// Reference-image count baked into the table (`0` = the plain t2i `[text, image]` geometry; `>0`
    /// = the edit `[text, refs…, target]` geometry, sc-10877).
    n_refs: usize,
    cos: Tensor,
    sin: Tensor,
}

impl Krea2Transformer {
    /// Build from a loaded `transformer/` weight set.
    pub fn load(w: &Weights, cfg: &Krea2Config) -> Result<Self> {
        let (heads, kv, hd, eps) = (
            cfg.num_attention_heads,
            cfg.num_kv_heads,
            cfg.attention_head_dim,
            cfg.norm_eps,
        );
        let (theads, tkv) = (cfg.text_num_attention_heads, cfg.text_num_kv_heads);
        let hidden = cfg.hidden_size;

        let final_sstable = w
            .get("final_layer.scale_shift_table")?
            .reshape((1, 2, hidden))?;

        Ok(Self {
            cfg: cfg.clone(),
            device: w.device().clone(),
            dtype: w.dtype(),
            img_in: linear_detect(w, "img_in", true)?,
            time_embed_l1: linear_detect(w, "time_embed.linear_1", true)?,
            time_embed_l2: linear_detect(w, "time_embed.linear_2", true)?,
            time_mod_proj: linear_detect(w, "time_mod_proj", true)?,
            txt_in_norm: RmsScale::load(w, "txt_in.norm.weight", eps)?,
            txt_in_l1: linear_detect(w, "txt_in.linear_1", true)?,
            txt_in_l2: linear_detect(w, "txt_in.linear_2", true)?,
            text_fusion: TextFusionTransformer::load(
                w,
                cfg.num_layerwise_text_blocks,
                cfg.num_refiner_text_blocks,
                theads,
                tkv,
                hd,
                eps,
            )?,
            blocks: (0..cfg.num_layers)
                .map(|i| {
                    SingleStreamBlock::load(
                        w,
                        &format!("transformer_blocks.{i}"),
                        heads,
                        kv,
                        hd,
                        hidden,
                        eps,
                    )
                })
                .collect::<Result<_>>()?,
            final_norm: RmsScale::load(w, "final_layer.norm.weight", eps)?,
            final_linear: linear_detect(w, "final_layer.linear", true)?,
            final_sstable,
            rope_cache: std::sync::Mutex::new(None),
        })
    }

    /// Build (or reuse) the joint RoPE `(cos, sin)` table for this render's fixed geometry (sc-8992).
    /// Recomputed only when `(cap_len, ht, wt, n_refs)` changes; otherwise the Arc-backed handles are
    /// cloned. `n_refs == 0` builds the plain t2i `[text, image]` table (byte-identical to building it
    /// inline); `n_refs > 0` builds the edit `[text, refs…, target]` table (sc-10877). Since
    /// `build_edit(n_refs = 0) ≡ build_t2i`, the t2i call stays on the exact `build_t2i` path.
    fn rope_tables(
        &self,
        cap_len: usize,
        ht: usize,
        wt: usize,
        n_refs: usize,
    ) -> Result<(Tensor, Tensor)> {
        let mut guard = self.rope_cache.lock().unwrap();
        if let Some(c) = guard.as_ref() {
            if c.cap_len == cap_len && c.ht == ht && c.wt == wt && c.n_refs == n_refs {
                return Ok((c.cos.clone(), c.sin.clone()));
            }
        }
        let (axes, theta) = (self.cfg.axes_dims_rope, self.cfg.rope_theta as f64);
        let rope = if n_refs == 0 {
            RopeTables::build_t2i(cap_len, ht, wt, axes, theta, &self.device)?
        } else {
            RopeTables::build_edit(cap_len, ht, wt, n_refs, axes, theta, &self.device)?
        };
        let (cos, sin) = rope.joint();
        *guard = Some(KreaRopeCache {
            cap_len,
            ht,
            wt,
            n_refs,
            cos: cos.clone(),
            sin: sin.clone(),
        });
        Ok((cos, sin))
    }

    /// Patch-embed a normalized `[b, 16, H, W]` latent through the (adapter-carrying) `img_in` →
    /// `[b, img_len, hidden]`. Shared by the target-latent embed and the edit reference-latent embeds
    /// (sc-10877) so every image token — noise or reference — goes through the identical projection.
    fn embed_image(&self, latent: &Tensor) -> Result<Tensor> {
        self.img_in.forward(&patchify(
            &latent.to_dtype(self.dtype)?,
            self.cfg.patch_size,
        )?)
    }

    /// The shared timestep + text front-end (sc-10877): the sinusoidal timestep embed `t`, the shared
    /// modulation `tvec = time_mod_proj(GELU(t))`, and the projected text context `ctx`
    /// `[b, cap, hidden]`. Both [`forward`](Self::forward) (t2i) and [`forward_edit`](Self::forward_edit)
    /// call this, so the t2i front-end stays byte-identical.
    fn front_end(&self, timestep: &Tensor, context: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let cfg = &self.cfg;
        let dt = self.dtype;
        let context = context.to_dtype(dt)?;

        // Timestep embed → `t`; shared modulation `tvec = time_mod_proj(GELU(t))`.
        let t_sin = temb(timestep, cfg.timestep_embed_dim, &self.device)?.to_dtype(dt)?; // [b, 1, tdim]
        let t = self
            .time_embed_l2
            .forward(&self.time_embed_l1.forward(&t_sin)?.gelu()?)?; // [b, 1, hidden]
        let tvec = self.time_mod_proj.forward(&t.gelu()?)?; // [b, 1, 6·hidden]

        // Text fusion (12 layers → 1) then the text input projection.
        let ctx = self.text_fusion.forward(&context)?; // [b, cap, text_hidden]
        let ctx = self.txt_in_norm.forward(&ctx)?;
        let ctx = self
            .txt_in_l2
            .forward(&self.txt_in_l1.forward(&ctx)?.gelu()?)?; // [b, cap, hidden]
        Ok((t, tvec, ctx))
    }

    /// Velocity prediction.
    ///
    /// - `latent`: `[b, 16, H, W]` (H, W multiples of `patch_size`),
    /// - `timestep`: `[b]` f32 (raw flow time in `[0, 1]`),
    /// - `context`: `[b, n_tokens, num_text_layers, text_hidden]` — the stacked Qwen3-VL select-layer
    ///   hidden states (sc-7569), already trimmed to the valid token count (no padding).
    ///
    /// Returns the velocity `[b, 16, H, W]`.
    pub fn forward(&self, latent: &Tensor, timestep: &Tensor, context: &Tensor) -> Result<Tensor> {
        let cfg = &self.cfg;
        let p = cfg.patch_size;
        let (_, _, h, w) = latent.dims4()?;
        let (ht, wt) = (h / p, w / p);
        let img_len = ht * wt;
        let latent_ch = cfg.in_channels / (p * p);
        let cap_len = context.dim(1)?;

        // Image patch embed + shared timestep/text front-end.
        let img = self.embed_image(latent)?; // [b, img_len, hidden]
        let (t, tvec, ctx) = self.front_end(timestep, context)?;

        // Fuse to the joint sequence and run the single-stream stack under the joint RoPE.
        let mut combined = Tensor::cat(&[&ctx, &img], 1)?; // [b, cap+img_len, hidden]

        // The joint RoPE table is step-invariant (fixed geometry), so cache it per render (sc-8992).
        let (rcos, rsin) = self.rope_tables(cap_len, ht, wt, 0)?;
        for blk in &self.blocks {
            combined = blk.forward(&combined, &tvec, &rcos, &rsin)?;
        }

        // Continuous-AdaLN output (SimpleModulation on `t`), then slice the image tokens + unpatchify.
        let out = self.final_layer(&combined, &t)?; // [b, cap+img_len, in_channels]
        let img_out = out.narrow(1, cap_len, img_len)?;
        unpatchify(&img_out, ht, wt, p, latent_ch)
    }

    /// **Kontext-style edit velocity prediction** (epic 10871 / sc-10877). Identical to
    /// [`forward`](Self::forward) but with one or more in-context reference latents prepended between the
    /// text and the noise: the joint sequence is `[text, refs…, target]`, each reference embedded through
    /// the same frozen `img_in` and positioned at its own RoPE frame ([`RopeTables::build_edit`]). The
    /// `in_channels` is unchanged (64) — this is **sequence** concat, not channel concat.
    ///
    /// - `latent`: the noise target `[b, 16, H, W]`.
    /// - `refs`: the VAE-encoded reference latents, **each at the target resolution** `[b, 16, H, W]`
    ///   (they share the target patch grid); fixed order (scene, then person — sc-10878). Empty ⇒
    ///   byte-identical to [`forward`](Self::forward) (the rope table falls back to `build_t2i`).
    /// - `context`: the (optionally image-grounded, P2) `[b, n_tokens, num_text_layers, text_hidden]`.
    ///
    /// Returns the velocity for the **target** tokens only `[b, 16, H, W]` (the reference tokens are
    /// conditioning; their output slice is discarded).
    pub fn forward_edit(
        &self,
        latent: &Tensor,
        timestep: &Tensor,
        context: &Tensor,
        refs: &[Tensor],
    ) -> Result<Tensor> {
        let cfg = &self.cfg;
        let p = cfg.patch_size;
        let (_, _, h, w) = latent.dims4()?;
        let (ht, wt) = (h / p, w / p);
        let img_len = ht * wt;
        let latent_ch = cfg.in_channels / (p * p);
        let n_refs = refs.len();
        let cap_len = context.dim(1)?;

        // Target + reference image tokens (references must share the target grid — VAE-encoded at the
        // target resolution). All go through the identical `img_in` projection.
        let img = self.embed_image(latent)?;
        let mut ref_toks = Vec::with_capacity(n_refs);
        for (i, r) in refs.iter().enumerate() {
            let (_, _, rh, rw) = r.dims4()?;
            if rh != h || rw != w {
                return Err(candle_gen::candle_core::Error::Msg(format!(
                    "krea edit: reference {i} is {rh}x{rw} but the target latent is {h}x{w}; \
                     references must be VAE-encoded at the target resolution"
                )));
            }
            ref_toks.push(self.embed_image(r)?);
        }

        let (t, tvec, ctx) = self.front_end(timestep, context)?;

        // Joint sequence `[ctx, refs…, target]` (references BEFORE the noise — the Krea2Edit contract).
        let mut parts: Vec<&Tensor> = Vec::with_capacity(2 + n_refs);
        parts.push(&ctx);
        parts.extend(ref_toks.iter());
        parts.push(&img);
        let mut combined = Tensor::cat(&parts, 1)?;

        let (rcos, rsin) = self.rope_tables(cap_len, ht, wt, n_refs)?;
        for blk in &self.blocks {
            combined = blk.forward(&combined, &tvec, &rcos, &rsin)?;
        }

        // Slice the TARGET tokens (they sit last, after the text + all reference blocks) + unpatchify.
        let out = self.final_layer(&combined, &t)?;
        let target_offset = cap_len + n_refs * img_len;
        let img_out = out.narrow(1, target_offset, img_len)?;
        unpatchify(&img_out, ht, wt, p, latent_ch)
    }

    /// Reference `LastLayer`: `SimpleModulation(t) = t + scale_shift_table` → `(scale, shift)`;
    /// `Linear((1+scale)·RMSNorm(x) + shift)`.
    fn final_layer(&self, x: &Tensor, t: &Tensor) -> Result<Tensor> {
        let m = t.broadcast_add(&self.final_sstable)?; // [b, 2, hidden] (t broadcasts 1→2)
        let scale = m.narrow(1, 0, 1)?; // [b, 1, hidden]
        let shift = m.narrow(1, 1, 1)?;
        let normed = self
            .final_norm
            .forward(x)?
            .broadcast_mul(&(scale + 1.0)?)?
            .broadcast_add(&shift)?;
        self.final_linear.forward(&normed)
    }
}

/// Reference `temb`: `freqs = exp(−ln(1e4)·arange(half)/half)`, `args = (timestep·1e3)·freqs`,
/// `concat([cos, sin], −1)` (cos-first). `timestep`: `[b]` → `[b, 1, dim]` (a per-sample vector that
/// broadcasts over the sequence). Built in f32 (the reference upcasts).
///
/// `pub(crate)` so the trainable DiT ([`crate::train_dit`]) shares the exact embedding (parity).
pub(crate) fn temb(timestep: &Tensor, dim: usize, device: &Device) -> Result<Tensor> {
    let half = dim / 2;
    let neg_ln = -(10000f64.ln()) as f32;
    let freqs: Vec<f32> = (0..half)
        .map(|i| (neg_ln * i as f32 / half as f32).exp())
        .collect();
    let freqs = Tensor::from_vec(freqs, (1, half), device)?; // [1, half]
    let b = timestep.dim(0)?;
    let scaled = (timestep.to_dtype(DType::F32)?.reshape((b, 1, 1))? * 1000.0)?; // [b, 1, 1]
    let args = scaled.broadcast_mul(&freqs.reshape((1, 1, half))?)?; // [b, 1, half]
    Tensor::cat(&[args.cos()?, args.sin()?], D::Minus1) // [b, 1, dim]
}

/// Reference `rearrange("b c (h ph) (w pw) -> b (h w) (c ph pw)")`: `[b, C, H, W] →
/// [b, (H/p)(W/p), C·p·p]` with **channel-major** patch flattening (NOT boogu's `(ph pw c)`).
///
/// `pub(crate)` so the trainable DiT ([`crate::train_dit`]) patchifies identically.
pub(crate) fn patchify(latent: &Tensor, p: usize) -> Result<Tensor> {
    let (b, c, h, w) = latent.dims4()?;
    let (ht, wt) = (h / p, w / p);
    let x = latent.reshape((b, c, ht, p, wt, p))?; // b, c, ht, ph, wt, pw
    let x = x.permute((0, 2, 4, 1, 3, 5))?; // b, ht, wt, c, ph, pw
    x.contiguous()?.reshape((b, ht * wt, c * p * p))
}

/// Inverse of [`patchify`] (`"b (h w) (c ph pw) -> b c (h ph) (w pw)"`): `[b, (h)(w), C·p·p] →
/// [b, C, h·p, w·p]`.
///
/// `pub(crate)` so the trainable DiT ([`crate::train_dit`]) unpatchifies identically.
pub(crate) fn unpatchify(
    tokens: &Tensor,
    ht: usize,
    wt: usize,
    p: usize,
    c: usize,
) -> Result<Tensor> {
    let b = tokens.dim(0)?;
    let x = tokens.contiguous()?.reshape((b, ht, wt, c, p, p))?; // b, ht, wt, c, ph, pw
    let x = x.permute((0, 3, 1, 4, 2, 5))?; // b, c, ht, ph, wt, pw
    x.contiguous()?.reshape((b, c, ht * p, wt * p))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Device;

    #[test]
    fn patchify_roundtrips_channel_major() {
        let dev = Device::Cpu;
        // [1, 4, 4, 6] with p=2 → 2×3 grid, 4·2·2 = 16 packed channels.
        let x = Tensor::arange(0f32, (4 * 4 * 6) as f32, &dev)
            .unwrap()
            .reshape((1, 4, 4, 6))
            .unwrap();
        let packed = patchify(&x, 2).unwrap();
        assert_eq!(packed.dims(), &[1, 2 * 3, 4 * 2 * 2]);
        let back = unpatchify(&packed, 2, 3, 2, 4).unwrap();
        assert_eq!(back.dims(), x.dims());
        let a = x.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = back.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(a, b, "patchify∘unpatchify must be the identity");
    }

    #[test]
    fn temb_is_cos_first_and_scaled() {
        let dev = Device::Cpu;
        // t = 0 → all angles 0 → cos half = 1, sin half = 0.
        let t = Tensor::from_vec(vec![0f32], (1,), &dev).unwrap();
        let e = temb(&t, 8, &dev).unwrap();
        assert_eq!(e.dims(), &[1, 1, 8]);
        let v = e.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(
            v[..4].iter().all(|&x| (x - 1.0).abs() < 1e-6),
            "cos-first half = 1 at t=0"
        );
        assert!(
            v[4..].iter().all(|&x| x.abs() < 1e-6),
            "sin half = 0 at t=0"
        );
    }
}
