//! The Boogu mixed single/double-stream DiT (`BooguImageTransformer2DModel`) forward. Port of
//! `mlx-gen-boogu`'s `transformer/mod.rs`.
//!
//! Two entry points share one inner path: [`BooguTransformer::forward`] (text-to-image) and
//! [`BooguTransformer::forward_edit`] (text+image-to-image with one or more reference images).
//!
//! Text-to-image flow (the reference-image blocks stay dormant):
//! ```text
//!   time_caption_embed:  temb = TimestepEmbedder(sinusoid(t·scale));  caption = Linear(RMSNorm(instr))
//!   patchify(p=2, 16→64) → x_embedder                                 → img tokens  [1, Li, 3360]
//!   context_refiner ×2  (no modulation)        on instruct tokens     [1, Lt, 3360]
//!   noise_refiner   ×2  (modulated)            on img tokens
//!   double_stream   ×8  (joint instruct↔img attn + img self-attn)
//!   fuse → [instruct; img]                                            [1, Lt+Li, 3360]
//!   single_stream   ×32 (modulated)            on the joint sequence
//!   norm_out (LuminaLayerNormContinuous + temb) → Linear(3360→64)
//!   unpatchify(img tokens)                                            → velocity [1, 16, H, W]
//! ```
//!
//! Per-sample `B = 1`: true-CFG runs this twice (cond/uncond) rather than padding a batch, so every
//! attention is full/unmasked. The instruction features arrive already trimmed to the valid token
//! count (the candle tokenizer emits no padding), so `cap_len = instruction_hidden.dim(1)`.

pub mod block;
pub mod rope;

use candle_gen::candle_core::{DType, Device, Result, Tensor, D};

use crate::config::BooguConfig;
use crate::loader::{layernorm_noaffine, linear_detect, rmsnorm, Weights};
use crate::quant::QLinear;
use block::{DoubleBlock, ModBlock, PlainBlock};
use rope::RopeTables;

/// The Boogu DiT. Carries the text-to-image modules plus the reference-image conditioning path
/// (`ref_image_patch_embedder` + `ref_image_refiner` + `image_index_embedding`) the Edit forward
/// exercises; the T2I forward simply leaves those dormant.
pub struct BooguTransformer {
    cfg: BooguConfig,
    device: Device,
    dtype: DType,
    x_embedder: QLinear,
    ref_image_patch_embedder: QLinear,
    image_index_embedding: Tensor,
    caption_norm: Tensor,
    caption_linear: QLinear,
    time_lin1: QLinear,
    time_lin2: QLinear,
    context_refiner: Vec<PlainBlock>,
    noise_refiner: Vec<ModBlock>,
    ref_image_refiner: Vec<ModBlock>,
    double_stream: Vec<DoubleBlock>,
    single_stream: Vec<ModBlock>,
    norm_out_lin1: QLinear,
    norm_out_lin2: QLinear,
    /// Per-render RoPE-table cache (sc-8992 / F-012, made multi-entry in sc-11201 / F-089). The
    /// `RopeTables` (all its cos/sin slices) depend only on the fixed geometry
    /// `(cap_len, ht, wt, ref_grids)` — not on the flow time / the current latent — so they are
    /// identical across every denoise step. Keyed on that geometry; hits clone the (Arc-backed) tables.
    /// Bounded to a few entries so the two true-CFG legs (cond + uncond, which usually differ in
    /// `cap_len`) stay resident and don't evict each other on every forward.
    rope_cache: RopeCache<RopeGeom, RopeTables>,
}

/// The geometry key the RoPE cache is keyed on: `(cap_len, ht, wt, ref_grids)`.
type RopeGeom = (usize, usize, usize, Vec<(usize, usize)>);

/// How many distinct RoPE geometries to keep cached per render. Under true CFG each denoise step runs
/// two forwards (cond + uncond) whose contexts usually differ in `cap_len`, i.e. two geometries
/// alternate across the render. A single-entry cache thrashed — evicting + rebuilding the host trig
/// tables (plus an H2D upload) on every forward (sc-11201 / F-089). Holding a few entries keeps both
/// legs resident; the small headroom absorbs any incidental extra geometry.
const ROPE_CACHE_CAP: usize = 4;

/// A tiny bounded, geometry-keyed cache for the per-render RoPE tables (sc-8992 / F-012, sc-11201 /
/// F-089). Holds up to [`ROPE_CACHE_CAP`] distinct geometries so both true-CFG legs stay resident
/// across every denoise step instead of evicting each other on every forward. On a miss the value is
/// built and inserted; on overflow the oldest entry is evicted (FIFO — with two alternating keys and
/// `cap ≥ 2` no eviction ever happens once both are cached). Hits clone the (Arc-backed) `V`.
/// `Mutex` (not `RefCell`): the DiT is shared as `Arc<…>` (`Send + Sync`).
struct RopeCache<K, V> {
    entries: std::sync::Mutex<Vec<(K, V)>>,
    cap: usize,
}

impl<K: PartialEq, V: Clone> RopeCache<K, V> {
    fn new(cap: usize) -> Self {
        Self {
            entries: std::sync::Mutex::new(Vec::new()),
            cap,
        }
    }

    /// Return the cached value for `key`, building + inserting it on a miss.
    fn get_or_build(&self, key: K, build: impl FnOnce() -> Result<V>) -> Result<V> {
        let mut guard = self.entries.lock().unwrap();
        if let Some((_, v)) = guard.iter().find(|(k, _)| *k == key) {
            return Ok(v.clone());
        }
        let v = build()?;
        if guard.len() >= self.cap {
            guard.remove(0);
        }
        guard.push((key, v.clone()));
        Ok(v)
    }
}

impl BooguTransformer {
    /// Build from a loaded `transformer/` weight set.
    pub fn load(w: &Weights, cfg: &BooguConfig) -> Result<Self> {
        let (heads, kv, hd) = (cfg.num_attention_heads, cfg.num_kv_heads, cfg.head_dim());
        let eps = cfg.norm_eps;

        let plain = |name: String| PlainBlock::load(w, &name, heads, kv, hd, eps);
        let mod_ = |name: String| ModBlock::load(w, &name, heads, kv, hd, eps);
        let dbl = |name: String| DoubleBlock::load(w, &name, heads, kv, hd, eps);

        Ok(Self {
            cfg: cfg.clone(),
            device: w.device().clone(),
            dtype: w.dtype(),
            x_embedder: linear_detect(w, "x_embedder", true)?,
            ref_image_patch_embedder: linear_detect(w, "ref_image_patch_embedder", true)?,
            image_index_embedding: w.get("image_index_embedding")?,
            caption_norm: w.get("time_caption_embed.caption_embedder.0.weight")?,
            caption_linear: linear_detect(w, "time_caption_embed.caption_embedder.1", true)?,
            time_lin1: linear_detect(w, "time_caption_embed.timestep_embedder.linear_1", true)?,
            time_lin2: linear_detect(w, "time_caption_embed.timestep_embedder.linear_2", true)?,
            context_refiner: (0..cfg.num_refiner_layers)
                .map(|i| plain(format!("context_refiner.{i}")))
                .collect::<Result<_>>()?,
            noise_refiner: (0..cfg.num_refiner_layers)
                .map(|i| mod_(format!("noise_refiner.{i}")))
                .collect::<Result<_>>()?,
            ref_image_refiner: (0..cfg.num_refiner_layers)
                .map(|i| mod_(format!("ref_image_refiner.{i}")))
                .collect::<Result<_>>()?,
            double_stream: (0..cfg.num_double_stream_layers)
                .map(|i| dbl(format!("double_stream_layers.{i}")))
                .collect::<Result<_>>()?,
            single_stream: (0..cfg.num_single_stream_layers())
                .map(|i| mod_(format!("single_stream_layers.{i}")))
                .collect::<Result<_>>()?,
            norm_out_lin1: linear_detect(w, "norm_out.linear_1", true)?,
            norm_out_lin2: linear_detect(w, "norm_out.linear_2", true)?,
            rope_cache: RopeCache::new(ROPE_CACHE_CAP),
        })
    }

    /// Build (or reuse) the `RopeTables` for this render's fixed geometry (sc-8992). Recomputed only
    /// when `(cap_len, ht, wt, ref_grids)` changes; otherwise the cached tables (Arc-backed) are
    /// cloned. Construction is identical to building it inline, so every step is byte-identical.
    fn rope_tables(
        &self,
        cap_len: usize,
        ht: usize,
        wt: usize,
        ref_grids: &[(usize, usize)],
        axes_dim: usize,
        theta: f32,
    ) -> Result<RopeTables> {
        self.rope_cache
            .get_or_build((cap_len, ht, wt, ref_grids.to_vec()), || {
                if ref_grids.is_empty() {
                    RopeTables::build_t2i(cap_len, ht, wt, axes_dim, theta, &self.device)
                } else {
                    RopeTables::build_edit(
                        cap_len,
                        ref_grids,
                        ht,
                        wt,
                        axes_dim,
                        theta,
                        &self.device,
                    )
                }
            })
    }

    /// Text-to-image velocity prediction.
    ///
    /// - `latent`: `[1, 16, H, W]` (H, W multiples of `patch_size`),
    /// - `timestep`: `[1]` f32 (raw, pre-scale),
    /// - `instruction_hidden`: `[1, L, 4096]` raw Qwen3-VL `last_hidden_state` (already trimmed).
    ///
    /// Returns the velocity `[1, 16, H, W]`.
    pub fn forward(
        &self,
        latent: &Tensor,
        timestep: &Tensor,
        instruction_hidden: &Tensor,
    ) -> Result<Tensor> {
        self.forward_inner(latent, &[], timestep, instruction_hidden)
    }

    /// Edit (text+image-to-image) velocity prediction with **one or more** reference images. Identical
    /// to [`Self::forward`] but with `ref_latents` (each `[1, 16, rH, rW]`, a VAE-encoded reference)
    /// packed — each through `ref_image_patch_embedder` + its own `image_index_embedding[i]` row +
    /// `ref_image_refiner` — *before* the noise tokens in the combined image sequence
    /// (`[ref₀; …; ref_{N-1}; noise]`). An empty slice is exactly [`Self::forward`] (text-to-image).
    /// The Boogu DiT supports up to 5 references (the `image_index_embedding` row count).
    pub fn forward_edit(
        &self,
        latent: &Tensor,
        ref_latents: &[Tensor],
        timestep: &Tensor,
        instruction_hidden: &Tensor,
    ) -> Result<Tensor> {
        self.forward_inner(latent, ref_latents, timestep, instruction_hidden)
    }

    fn forward_inner(
        &self,
        latent: &Tensor,
        ref_latents: &[Tensor],
        timestep: &Tensor,
        instruction_hidden: &Tensor,
    ) -> Result<Tensor> {
        let p = self.cfg.patch_size;
        let (_, _, h, w) = latent.dims4()?;
        let (ht, wt) = (h / p, w / p);
        let img_len = ht * wt;
        let dt = self.dtype;
        let axes_dim = self.cfg.axes_dim_rope[0];
        let theta = self.cfg.rope_theta;

        let latent = latent.to_dtype(dt)?;
        // The candle tokenizer emits no padding, so every instruction token is valid.
        let instruct = instruction_hidden.to_dtype(dt)?;
        let cap_len = instruct.dim(1)?;

        // Timestep + caption embedding.
        let temb = self.timestep_embed(timestep)?; // [1, 1, 1024]
        let caption = self.caption_linear.forward(&rmsnorm(
            &instruct,
            &self.caption_norm,
            self.cfg.norm_eps,
        )?)?; // [1, cap, 3360]

        // Patchify the noise latent → target image tokens.
        let img = self.x_embedder.forward(&patchify(&latent, p)?)?; // [1, img_len, 3360]

        // Reference images (Edit): patch-embed each + add its per-image index embedding row. The j-th
        // reference's tokens get `image_index_embedding[j]` (OmniGen2 lineage; max 5 references). The
        // patch grids drive the multi-image RoPE; an empty `ref_latents` is the text-to-image path.
        let mut ref_tokens: Vec<(Tensor, usize)> = Vec::with_capacity(ref_latents.len());
        let mut ref_grids: Vec<(usize, usize)> = Vec::with_capacity(ref_latents.len());
        for (j, rl) in ref_latents.iter().enumerate() {
            let rl = rl.to_dtype(dt)?;
            let (_, _, rh, rw) = rl.dims4()?;
            let (rht, rwt) = (rh / p, rw / p);
            let ref_t = self.ref_image_patch_embedder.forward(&patchify(&rl, p)?)?;
            let idx = self
                .image_index_embedding
                .narrow(0, j, 1)?
                .reshape((1, 1, self.cfg.hidden_size))?
                .to_dtype(dt)?;
            let ref_t = ref_t.broadcast_add(&idx)?;
            ref_tokens.push((ref_t, rht * rwt));
            ref_grids.push((rht, rwt));
        }

        // The RoPE tables are step-invariant (fixed geometry), so cache them per render (sc-8992).
        let rope = self.rope_tables(cap_len, ht, wt, &ref_grids, axes_dim, theta)?;

        let (text_cos, text_sin) = rope.text()?;
        let (noise_cos, noise_sin) = rope.image()?;
        let (comb_cos, comb_sin) = rope.combined_image()?;
        let (joint_cos, joint_sin) = rope.joint();

        // Context refinement (instruction stream).
        let mut instruct_h = caption;
        for blk in &self.context_refiner {
            instruct_h = blk.forward(&instruct_h, &text_cos, &text_sin)?;
        }

        // Noise refinement (target image stream).
        let mut img = img;
        for blk in &self.noise_refiner {
            img = blk.forward(&img, &noise_cos, &noise_sin, &temb)?;
        }

        // Reference refinement: refine EACH reference independently — its own RoPE sub-slice, no
        // cross-image attention (the OmniGen2 batched `ref_image_refiner` masks each reference to
        // itself). Then prepend the refined references to the noise tokens to form the combined image
        // sequence `[ref₀; …; ref_{N-1}; noise]` (Edit). T2I leaves the sequence as the noise tokens.
        let mut img = if ref_tokens.is_empty() {
            img
        } else {
            let mut combined: Vec<Tensor> = Vec::with_capacity(ref_tokens.len() + 1);
            let mut local = 0usize;
            for (mut ref_t, ref_len) in ref_tokens {
                let (ref_cos, ref_sin) = rope.ref_image_slice(local, ref_len)?;
                for blk in &self.ref_image_refiner {
                    ref_t = blk.forward(&ref_t, &ref_cos, &ref_sin, &temb)?;
                }
                combined.push(ref_t);
                local += ref_len;
            }
            combined.push(img);
            let refs: Vec<&Tensor> = combined.iter().collect();
            Tensor::cat(&refs, 1)?
        };

        // Dual-stream blocks (joint instruct↔combined-image attn + combined-image self-attn).
        for blk in &self.double_stream {
            let (ni, nt) = blk.forward(
                &img,
                &instruct_h,
                &joint_cos,
                &joint_sin,
                &comb_cos,
                &comb_sin,
                &temb,
            )?;
            img = ni;
            instruct_h = nt;
        }

        // Fuse to the joint sequence, then single-stream blocks.
        let mut joint = Tensor::cat(&[&instruct_h, &img], 1)?; // [1, cap+ref+img, 3360]
        for blk in &self.single_stream {
            joint = blk.forward(&joint, &joint_cos, &joint_sin, &temb)?;
        }

        // Continuous-AdaLN output projection (LuminaLayerNormContinuous, eps 1e-6, no affine).
        let scale = self.norm_out_lin1.forward(&temb.silu()?)?; // [1, 1, 3360]
        let normed = layernorm_noaffine(&joint, 1e-6)?;
        let normed = normed.broadcast_mul(&(scale + 1.0)?)?;
        let out = self.norm_out_lin2.forward(&normed)?; // [1, cap+ref+img, 64]

        // Unpatchify the trailing target-image tokens into the velocity (reference tokens, when
        // present, are dropped — only the noise/target slice is the prediction).
        let total = out.dim(1)?;
        let img_tokens = out.narrow(1, total - img_len, img_len)?;
        unpatchify(&img_tokens, ht, wt, p, self.cfg.out_channels)
    }

    /// `Lumina2CombinedTimestepCaptionEmbedding` timestep branch:
    /// `sinusoid(timestep · timestep_scale, 256) → Linear → SiLU → Linear` → `[1, 1, 1024]`.
    fn timestep_embed(&self, timestep: &Tensor) -> Result<Tensor> {
        let scaled = (timestep.to_dtype(DType::F32)? * self.cfg.timestep_scale as f64)?;
        let proj = sinusoidal_timestep(&scaled, 256, &self.device)?.to_dtype(self.dtype)?; // [1, 256]
        let t = self.time_lin1.forward(&proj)?;
        let t = t.silu()?;
        let t = self.time_lin2.forward(&t)?; // [1, 1024]
        t.unsqueeze(1) // [1, 1, 1024]
    }
}

/// diffusers `get_timestep_embedding(x, dim, flip_sin_to_cos=True, downscale_freq_shift=0,
/// max_period=10000)`: `freq_i = 10000^(−i/half)`, `emb = x·freq`, `concat([cos, sin], -1)` (cos
/// first). `x`: `[N]` → `[N, dim]`. Built in f32.
fn sinusoidal_timestep(x: &Tensor, dim: usize, device: &Device) -> Result<Tensor> {
    let half = dim / 2;
    let neg_ln = -(10000f64.ln()) as f32;
    let freqs: Vec<f32> = (0..half)
        .map(|i| (neg_ln * i as f32 / half as f32).exp())
        .collect();
    let freqs = Tensor::from_vec(freqs, (1, half), device)?; // [1, half]
    let n = x.dim(0)?;
    let emb = x.reshape((n, 1))?.broadcast_mul(&freqs)?; // [N, half]
    Tensor::cat(&[emb.cos()?, emb.sin()?], D::Minus1) // [N, dim]
}

/// `c (h p1) (w p2) -> (h w) (p1 p2 c)` with batch: `[1, C, H, W] → [1, (H/p)(W/p), p·p·C]`.
fn patchify(latent: &Tensor, p: usize) -> Result<Tensor> {
    let (b, c, h, w) = latent.dims4()?;
    let (ht, wt) = (h / p, w / p);
    let x = latent.reshape((b, c, ht, p, wt, p))?; // B, C, h, p1, w, p2
    let x = x.permute((0, 2, 4, 3, 5, 1))?; // B, h, w, p1, p2, C
    x.contiguous()?.reshape((b, ht * wt, p * p * c))
}

/// `(h w) (p1 p2 c) -> c (h p1) (w p2)` with batch: `[1, (h)(w), p·p·C] → [1, C, h·p, w·p]`.
fn unpatchify(tokens: &Tensor, ht: usize, wt: usize, p: usize, c: usize) -> Result<Tensor> {
    let b = tokens.dim(0)?;
    // `tokens` is a `narrow`ed slice of the output sequence; contiguate before reshape.
    let x = tokens.contiguous()?.reshape((b, ht, wt, p, p, c))?; // B, h, w, p1, p2, C
    let x = x.permute((0, 5, 1, 3, 2, 4))?; // B, C, h, p1, w, p2
    x.contiguous()?.reshape((b, c, ht * p, wt * p))
}

#[cfg(test)]
mod tests {
    use super::*;

    // sc-11201 / F-089: the RoPE cache must hold both true-CFG geometries at once so alternating
    // cond/uncond forwards don't thrash it (rebuild every step). We exercise the bounded cache
    // directly (it is model-independent) rather than standing up a full DiT.
    #[test]
    fn rope_cache_keeps_both_cfg_legs_resident() {
        let cache: RopeCache<usize, i32> = RopeCache::new(ROPE_CACHE_CAP);
        // Two alternating geometries stand in for the CFG cond (cap_len=64) and uncond (cap_len=8)
        // token counts. `builds` counts how many times the (expensive) table build actually runs.
        let mut builds = 0usize;
        let (cond, uncond) = (64usize, 8usize);
        for _step in 0..40 {
            for &cap in &[cond, uncond] {
                let v = cache
                    .get_or_build(cap, || {
                        builds += 1;
                        Ok(cap as i32 * 10)
                    })
                    .unwrap();
                // Every hit returns exactly what a fresh build would (value keyed on geometry).
                assert_eq!(v, cap as i32 * 10, "cached value must equal a fresh build");
            }
        }
        // 40 steps × 2 legs = 80 forwards, but only the first touch of each geometry builds: the
        // single-entry cache would have rebuilt on all 80. This proves no thrash.
        assert_eq!(
            builds, 2,
            "each CFG geometry built exactly once across all steps"
        );
    }

    #[test]
    fn rope_cache_evicts_oldest_beyond_capacity() {
        // With more distinct geometries than capacity, the oldest is evicted (FIFO) and a re-touch
        // rebuilds — but the two-geometry CFG case (cap ≥ 2) never reaches this.
        let cache: RopeCache<usize, i32> = RopeCache::new(2);
        let builds = std::cell::Cell::new(0usize);
        let build = |k: usize| {
            cache
                .get_or_build(k, || {
                    builds.set(builds.get() + 1);
                    Ok(k as i32)
                })
                .unwrap()
        };
        build(1);
        build(2);
        assert_eq!(builds.get(), 2);
        build(1); // still cached
        assert_eq!(builds.get(), 2);
        build(3); // evicts key 1 (oldest)
        assert_eq!(builds.get(), 3);
        build(1); // 1 was evicted → rebuild
        assert_eq!(builds.get(), 4);
    }
}
