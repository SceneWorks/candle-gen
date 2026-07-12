//! Token-axis packed conditioning + the 7-mode guided-velocity dispatch — the candle sibling of
//! `mlx-gen-bernini/src/forward.rs`'s `PackedForward` / `guided_velocity` / `Mode` dispatch (sc-11004).
//!
//! Each conditioning source and the noisy target are patch-embedded separately (each with its own
//! source-id RoPE, [`candle_gen_wan::rope::apply_source_id`]) and concatenated on the token axis with
//! the **target last**; at batch 1 the reference's varlen attention is one `cu_seqlens` segment, i.e.
//! plain full self-attention, so the whole packed sequence runs through
//! [`WanTransformer::forward_packed`] and the target tokens are sliced back out and unpatchified to a
//! `[1, 16, T, H/8, W/8]` velocity.
//!
//! [`guided_velocity`] runs the per-mode forward passes over the right conditioning combos and combines
//! them — either a plain weighted velocity sum (`t2v`, `v2v`, `v2v_chain`, `rv2v`) or APG in x-space
//! (`t2v_apg`, `v2v_apg`, `r2v_apg`; see [`crate::guidance`]).
//!
//! **Candle vs mlx seam.** The mlx `forward_packed` consumes a prepared cross-attention K/V cache; the
//! candle [`WanTransformer`] re-projects the (already `embed_text`-projected) `context [1, S, dim]`
//! inside each block's cross-attention, so [`PackedForward::velocity`] carries the projected context
//! tensor instead of a `(K, V)` list. The math is identical; only the cross-KV caching differs.

use std::collections::HashMap;
use std::sync::Mutex;

use candle_gen::candle_core::{Tensor, TensorId};
use candle_gen::Result as CResult;

use candle_gen_wan::config::TransformerConfig;
use candle_gen_wan::rope::{apply_source_id, assign_source_ids, WanRope};
use candle_gen_wan::transformer::WanTransformer;

use crate::config::Mode;
use crate::guidance::{normalized_guidance, normalized_guidance_chain, MomentumBuffer};
use crate::vit_guidance::{rv2v_chain, vae_txt_vit};

/// Cache key for the step-invariant RoPE `(cos, sin)` table: the token grid `(f, h, w)` plus the
/// source-id phase (its raw f64 bits — the id is either an integer or a `linspace` sample, both exact).
type RopeKey = (usize, usize, usize, u64);
/// Cache key for a conditioning source's patch-embedded tokens: `(expert identity, latent identity)`.
/// The patch-embed depends only on the DiT weights + the source latent (NOT the source-id), so the same
/// latent re-embedded under different source-ids (the renderer combos) hits the same entry.
type PatchKey = (usize, TensorId);
/// Cached patch-embed value: the source tokens `[1, L, dim]` + their `(f, h, w)` token grid.
type PatchEntry = (Tensor, (usize, usize, usize));

/// The packed-forward engine: holds the transformer geometry (RoPE builder + patch grid) so it can
/// patch-embed the target and each conditioning source with their source-id RoPE and run one packed
/// forward. Cheap to build; build once per render.
///
/// F-098: `velocity` is called 2–4× per denoise step (the guidance chain) over ~40 steps, and each call
/// rebuilt the target's + every source's RoPE `(cos, sin)` host trig tables and re-patch-embedded every
/// (step-invariant) conditioning source. Those are geometry-only, so they are memoized here behind
/// interior-mutability caches keyed by geometry alone — the memoized tensors are bit-identical to a
/// fresh build, so the guided velocity is unchanged. The `PackedForward` is built per render (never
/// shared across renders), so the caches live exactly as long as one render.
pub struct PackedForward {
    cfg: TransformerConfig,
    max_trained_src_id: f64,
    interpolate_src_id: bool,
    /// Memoized `(cos, sin)` per `(grid, source_id)` — reused by the target (id 0) across all steps and
    /// by every conditioning source across the guidance chain.
    rope_cache: Mutex<HashMap<RopeKey, (Tensor, Tensor)>>,
    /// Memoized patch-embedded source tokens per `(expert, latent)` — the conditioning sources are
    /// step-invariant, so their patch-embed is computed once per expert instead of per `velocity` call.
    patch_cache: Mutex<HashMap<PatchKey, PatchEntry>>,
}

/// The four conditioning combos (each a list of `(latent, source_id)`); the target is added per forward
/// with source_id 0.
pub struct Combos {
    /// Target only (no conditioning source).
    pub none: Vec<(Tensor, f64)>,
    /// Every conditioning **video** (source-ids mirror the video portion of [`Combos::vi`]).
    pub v: Vec<(Tensor, f64)>,
    /// Every conditioning **image**.
    pub i: Vec<(Tensor, f64)>,
    /// Videos then images (the full conditioning set).
    pub vi: Vec<(Tensor, f64)>,
}

impl PackedForward {
    pub fn new(cfg: TransformerConfig, max_trained_src_id: f64, interpolate_src_id: bool) -> Self {
        Self {
            cfg,
            max_trained_src_id,
            interpolate_src_id,
            rope_cache: Mutex::new(HashMap::new()),
            patch_cache: Mutex::new(HashMap::new()),
        }
    }

    /// The step-invariant source-id RoPE `(cos, sin)` `[L, head_dim/2]` (f32) for a token grid, memoized
    /// per `(grid, source_id)` (F-098). The build is deterministic host trig (`WanRope::cos_sin` +
    /// `apply_source_id`), so a cache hit is bit-identical to a fresh build.
    fn rope_for(
        &self,
        grid: (usize, usize, usize),
        source_id: f64,
        latent: &Tensor,
    ) -> CResult<(Tensor, Tensor)> {
        let key: RopeKey = (grid.0, grid.1, grid.2, source_id.to_bits());
        if let Some((cos, sin)) = self.rope_cache.lock().unwrap().get(&key) {
            return Ok((cos.clone(), sin.clone()));
        }
        let (cos, sin) =
            WanRope::new(&self.cfg).cos_sin(grid.0, grid.1, grid.2, latent.device())?;
        let (cos, sin) = apply_source_id(&cos, &sin, source_id, self.cfg.head_dim)?;
        self.rope_cache
            .lock()
            .unwrap()
            .insert(key, (cos.clone(), sin.clone()));
        Ok((cos, sin))
    }

    /// Patch-embed a **conditioning source** latent `[1, 16, T, H8, W8]` to `(tokens [1,L,dim], cos,
    /// sin, grid)` with the source-id RoPE folded in — the source's tokens (per expert) and its RoPE are
    /// both step-invariant, so both are memoized (F-098). The noisy **target** is embedded separately in
    /// [`Self::velocity`] (its tokens change every step; only its RoPE is cached).
    #[allow(clippy::type_complexity)]
    fn embed_segment(
        &self,
        dit: &WanTransformer,
        latent: &Tensor,
        source_id: f64,
    ) -> CResult<(Tensor, Tensor, Tensor, (usize, usize, usize))> {
        let key: PatchKey = (dit as *const WanTransformer as usize, latent.id());
        // Scope the read guard so it is dropped before `patch_embed_tokens` + the re-lock on a miss —
        // `std::sync::Mutex` is not reentrant, so holding the guard across the insert would deadlock.
        let hit = self
            .patch_cache
            .lock()
            .unwrap()
            .get(&key)
            .map(|(tokens, grid)| (tokens.clone(), *grid));
        let (tokens, grid) = match hit {
            Some(entry) => entry,
            None => {
                let (tokens, grid) = dit.patch_embed_tokens(latent)?;
                self.patch_cache
                    .lock()
                    .unwrap()
                    .insert(key, (tokens.clone(), grid));
                (tokens, grid)
            }
        };
        let (cos, sin) = self.rope_for(grid, source_id, latent)?;
        Ok((tokens, cos, sin, grid))
    }

    /// One packed forward: conditioning `sources` (each `(latent, source_id)`) + the noisy `target`
    /// (source_id 0), returning the **target** velocity `[1, 16, T, H8, W8]` (the reference's
    /// `pred[:, target_mask, :]` then unpatchify). The target is concatenated last. `context` is this
    /// expert's projected UMT5 context `[1, S, dim]` (cond or uncond).
    pub fn velocity(
        &self,
        dit: &WanTransformer,
        target: &Tensor,
        sources: &[(Tensor, f64)],
        t: f64,
        context: &Tensor,
    ) -> CResult<Tensor> {
        let mut toks = Vec::with_capacity(sources.len() + 1);
        let mut coss = Vec::with_capacity(sources.len() + 1);
        let mut sins = Vec::with_capacity(sources.len() + 1);
        for (lat, sid) in sources {
            let (tk, c, s, _) = self.embed_segment(dit, lat, *sid)?;
            toks.push(tk);
            coss.push(c);
            sins.push(s);
        }
        // The target is patch-embedded fresh every call (the noisy latent changes each step); only its
        // step-invariant source-id-0 RoPE is memoized (F-098).
        let (tk_t, grid_t) = dit.patch_embed_tokens(target)?;
        let (c_t, s_t) = self.rope_for(grid_t, 0.0, target)?;
        let l_t = grid_t.0 * grid_t.1 * grid_t.2;
        toks.push(tk_t);
        coss.push(c_t);
        sins.push(s_t);

        let tok_refs: Vec<&Tensor> = toks.iter().collect();
        let cos_refs: Vec<&Tensor> = coss.iter().collect();
        let sin_refs: Vec<&Tensor> = sins.iter().collect();
        let tokens = Tensor::cat(&tok_refs, 1)?;
        let cos = Tensor::cat(&cos_refs, 0)?;
        let sin = Tensor::cat(&sin_refs, 0)?;

        let out = dit.forward_packed(&tokens, t, context, &cos, &sin)?; // [1, total, oc·∏patch]
        let total = out.dim(1)?;
        // Slice the target tokens (last l_t) and unpatchify to [1, 16, T, H8, W8].
        let target_tokens = out.narrow(1, total - l_t, l_t)?;
        Ok(dit.unpatchify_tokens(&target_tokens, grid_t)?)
    }

    /// Assemble the four conditioning combos from the VAE-encoded `videos` / `images` source latents
    /// (each `[1, 16, T, H8, W8]`). The video-only combo carries **every** conditioning video (not just
    /// `videos[0]`), and a video keeps the same source-id it gets in the `vi` combo (F-021, ported from
    /// mlx). Videos are assigned source-ids before images (videos-first ordering).
    pub fn build_combos(&self, videos: &[Tensor], images: &[Tensor]) -> Combos {
        let (nv, ni) = (videos.len(), images.len());
        let vi_sids = assign_source_ids(nv + ni, self.max_trained_src_id, self.interpolate_src_id);
        let i_sids = assign_source_ids(ni, self.max_trained_src_id, self.interpolate_src_id);
        let v = videos
            .iter()
            .enumerate()
            .map(|(k, vid)| (vid.clone(), vi_sids[k]))
            .collect();
        let i = images
            .iter()
            .enumerate()
            .map(|(j, im)| (im.clone(), i_sids[j]))
            .collect();
        let mut vi = Vec::with_capacity(nv + ni);
        for (k, v) in videos.iter().enumerate() {
            vi.push((v.clone(), vi_sids[k]));
        }
        for (j, im) in images.iter().enumerate() {
            vi.push((im.clone(), vi_sids[nv + j]));
        }
        Combos {
            none: vec![],
            v,
            i,
            vi,
        }
    }
}

/// All the per-step guidance knobs (the omegas are already `omega_scale`-rescaled when the low-noise
/// expert is active — done by the caller).
#[derive(Clone)]
pub struct GuidanceParams {
    pub omega_vid: f32,
    pub omega_img: f32,
    pub omega_txt: f32,
    pub eta: f32,
    /// Per-term norm thresholds (`r2v_apg` uses two; the single-cond modes use index 0).
    pub norm_threshold: [f32; 2],
}

/// `x = noisy − σ·v` (velocity → x-space). APG operates in x-space.
fn to_x(noisy: &Tensor, sigma: f32, v: &Tensor) -> CResult<Tensor> {
    Ok((noisy - v.affine(sigma as f64, 0.0)?)?)
}
/// `v = (noisy − x)/σ` (x-space → velocity).
fn from_x(noisy: &Tensor, sigma: f32, x: &Tensor) -> CResult<Tensor> {
    Ok((noisy - x)?.affine(1.0 / sigma as f64, 0.0)?)
}

/// Number of APG momentum buffers a mode needs (0 for the plain modes, 1 for the single-cond `*_apg`
/// modes, 2 for the chained `r2v_apg`).
pub fn num_momentum_buffers(mode: Mode) -> usize {
    match mode {
        Mode::T2vApg | Mode::V2vApg => 1,
        Mode::R2vApg => 2,
        _ => 0,
    }
}

/// Compute the guided velocity `[1, 16, T, H8, W8]` for one denoise step (the renderer's per-mode body).
/// `ctx_cond`/`ctx_uncond` are this expert's projected UMT5 context (cond / empty-neg); `videos`/`images`
/// are the VAE-encoded source latents; `mbufs` are the APG momentum buffers (persisting across steps —
/// one for the single-cond `*_apg` modes, two for `r2v_apg`). `sigma` is this step's flow sigma (for the
/// x-space conversion).
#[allow(clippy::too_many_arguments)]
pub fn guided_velocity(
    pf: &PackedForward,
    mode: Mode,
    dit: &WanTransformer,
    noisy: &Tensor,
    videos: &[Tensor],
    images: &[Tensor],
    t: f64,
    sigma: f32,
    ctx_cond: &Tensor,
    ctx_uncond: &Tensor,
    g: &GuidanceParams,
    mbufs: &mut [MomentumBuffer],
) -> CResult<Tensor> {
    let c = pf.build_combos(videos, images);
    let v = |sources: &[(Tensor, f64)], cond: bool| -> CResult<Tensor> {
        let ctx = if cond { ctx_cond } else { ctx_uncond };
        pf.velocity(dit, noisy, sources, t, ctx)
    };
    // Weighted velocity sum for a list of (vel, weight) deltas: base + Σ w·(cur − prev).
    let chain = |terms: &[(&Tensor, f32)]| -> CResult<Tensor> {
        // terms[0] is the base (weight ignored); each subsequent is (cur, weight) diffing the prev.
        let mut acc = terms[0].0.clone();
        for w in 1..terms.len() {
            let delta = (terms[w].0 - terms[w - 1].0)?;
            acc = (acc + (delta * terms[w].1 as f64)?)?;
        }
        Ok(acc)
    };

    match mode {
        Mode::T2v => {
            let e0 = v(&c.none, false)?;
            let et = v(&c.none, true)?;
            chain(&[(&e0, 0.0), (&et, g.omega_txt)])
        }
        Mode::V2v => {
            let e_vi = v(&c.vi, false)?;
            let e_vti = v(&c.vi, true)?;
            chain(&[(&e_vi, 0.0), (&e_vti, g.omega_txt)])
        }
        Mode::V2vChain => {
            let e0 = v(&c.none, false)?;
            let ev = v(&c.v, false)?;
            let e_vti = v(&c.vi, true)?;
            chain(&[(&e0, 0.0), (&ev, g.omega_vid), (&e_vti, g.omega_txt)])
        }
        Mode::Rv2v => {
            let e0 = v(&c.none, false)?;
            let ev = v(&c.v, false)?;
            let e_vi = v(&c.vi, false)?;
            let e_vti = v(&c.vi, true)?;
            chain(&[
                (&e0, 0.0),
                (&ev, g.omega_vid),
                (&e_vi, g.omega_img),
                (&e_vti, g.omega_txt),
            ])
        }
        Mode::T2vApg => {
            let e0 = v(&c.none, false)?;
            let et = v(&c.none, true)?;
            let x0 = to_x(noisy, sigma, &e0)?;
            let xt = to_x(noisy, sigma, &et)?;
            let xg = normalized_guidance(
                &xt,
                &x0,
                g.omega_txt,
                Some(&mut mbufs[0]),
                g.eta,
                g.norm_threshold[0],
            )?;
            from_x(noisy, sigma, &xg)
        }
        Mode::V2vApg => {
            let e0 = v(&c.vi, false)?;
            let e_vti = v(&c.vi, true)?;
            let x0 = to_x(noisy, sigma, &e0)?;
            let xvti = to_x(noisy, sigma, &e_vti)?;
            let xg = normalized_guidance(
                &xvti,
                &x0,
                g.omega_txt,
                Some(&mut mbufs[0]),
                g.eta,
                g.norm_threshold[0],
            )?;
            from_x(noisy, sigma, &xg)
        }
        Mode::R2vApg => {
            let e0 = v(&c.none, false)?;
            let ei = v(&c.i, false)?;
            let eti = v(&c.i, true)?;
            let x0 = to_x(noisy, sigma, &e0)?;
            let xi = to_x(noisy, sigma, &ei)?;
            let xti = to_x(noisy, sigma, &eti)?;
            let xg = normalized_guidance_chain(
                &x0,
                &[xi, xti],
                &[g.omega_img, g.omega_txt],
                mbufs,
                g.eta,
                &g.norm_threshold,
            )?;
            from_x(noisy, sigma, &xg)
        }
    }
}

// ---------------------------------------------------------------------------
// sc-10995: the full-Bernini ViT-conditioned per-step velocity (`sample_one_step`), the candle sibling
// of `mlx-gen-bernini/src/forward.rs`'s `vit_one_step` / `VitMode`. These are the `*_wapg` guidance
// modes noted as pending in sc-11004 — the renderer-side compute that consumes the planner's 4
// prompt-embed streams. Distinct from the renderer-only [`Mode`] dispatch above.
// ---------------------------------------------------------------------------

/// One full-Bernini ViT-conditioned guidance mode (`BerniniPipeline`'s `sample_one_step` modes; the
/// renderer-only [`Mode`] variants are disjoint from these).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VitMode {
    /// `vae_txt_vit` — plain 4-prediction combine.
    VaeTxtVit,
    /// `vae_txt_vit_wapg` — the same combine with `apg_delta` on each delta (the primary t2i/edit).
    VaeTxtVitWapg,
    /// `rv2v_wapg` — the 5-prediction vid/img/txt/ViT chain, plain deltas.
    Rv2vWapg,
    /// `r2v_wapg` — the same chain with `apg_delta` deltas.
    R2vWapg,
    /// `v2v_apg` — x-space [`normalized_guidance`] over (uncond, wtxt_wvit) (the renderer family).
    V2vApg,
}

impl VitMode {
    pub fn from_name(s: &str) -> Option<VitMode> {
        Some(match s {
            "vae_txt_vit" => VitMode::VaeTxtVit,
            "vae_txt_vit_wapg" => VitMode::VaeTxtVitWapg,
            "rv2v_wapg" => VitMode::Rv2vWapg,
            "r2v_wapg" => VitMode::R2vWapg,
            "v2v_apg" => VitMode::V2vApg,
            _ => return None,
        })
    }

    fn apg(self) -> bool {
        matches!(self, VitMode::VaeTxtVitWapg | VitMode::R2vWapg)
    }

    /// Whether this full-Bernini mode **requires** source conditioning (a reference image / video clip):
    /// the reference-video (`rv2v_wapg` / `r2v_wapg`) and video-edit (`v2v_apg`) modes consume packed
    /// source latents, so with no source present they would silently render text-only. The
    /// `vae_txt_vit`/`vae_txt_vit_wapg` modes cover t2i **and** i2i — a source is optional there — so they
    /// do not require conditioning (F-096, the full-pipeline mirror of [`crate::config::Mode`]).
    pub fn needs_conditioning(self) -> bool {
        matches!(self, VitMode::Rv2vWapg | VitMode::R2vWapg | VitMode::V2vApg)
    }
}

/// Number of APG momentum buffers a full-Bernini ViT mode needs across the denoise loop (1 for the
/// x-space `v2v_apg` stream, 0 for the v-space combine modes). The buffer is allocated **once** before
/// the loop and threaded per step so a nonzero momentum default would actually carry across steps rather
/// than resetting each call (F-161).
pub fn num_vit_momentum_buffers(mode: VitMode) -> usize {
    match mode {
        VitMode::V2vApg => 1,
        _ => 0,
    }
}

/// The planner's 4 prepared prompt-embed streams, each already `embed_text`-projected to this expert's
/// context space `[1, S, dim]` (cond/uncond × wvit/wovit). The candle [`WanTransformer`] re-projects the
/// context inside each block's cross-attention, so a stream is carried as its projected context tensor
/// (not an mlx-style prepared `(K, V)` list) — the same seam as [`PackedForward::velocity`]'s `context`.
pub struct VitStreams<'a> {
    pub wtxt_wvit: &'a Tensor,
    pub wtxt_wovit: &'a Tensor,
    pub wotxt_wvit: &'a Tensor,
    pub wotxt_wovit: &'a Tensor,
}

/// Per-step ViT-conditioned guidance knobs (already `omega_scale`-rescaled by the caller when the
/// low-noise expert is active).
#[derive(Clone)]
pub struct VitGuidanceParams {
    pub omega_txt: f32,
    pub omega_img: f32,
    pub omega_vid: f32,
    pub omega_tgt: f32,
    /// x-space APG knobs for `v2v_apg` (eta, norm_threshold); unused by the v-space modes.
    pub eta: f32,
    pub norm_threshold: f32,
}

/// `sample_one_step` (the full-Bernini per-step velocity over the 4 prompt streams × the packed-latent
/// variants). Each prediction is one [`PackedForward::velocity`] (≡ the reference `shared_step`) over a
/// source subset + a chosen prompt stream:
///   - `wvae` = image ⧺ video sources; `wvidvae` = video; `wovae` = target only.
///   - `images`/`videos` are `(latent, source_id)` source lists (the target is added with id 0 inside
///     `velocity`).
///
/// The combine is the slice-A math ([`vae_txt_vit`] / [`rv2v_chain`]) directly on the target-sliced
/// spatial velocities — which are already `[1, 16, T, H8, W8]` (batch-first) on candle, so no extra
/// batch juggling is needed. `v2v_apg` routes through the x-space [`normalized_guidance`], carrying its
/// momentum through the caller-owned `mbufs` (allocated once before the denoise loop via
/// [`num_vit_momentum_buffers`]) so a nonzero momentum default actually persists across steps instead of
/// resetting each call (F-161); an empty `mbufs` degrades to no momentum. Returns `[1, 16, T, H8, W8]`.
#[allow(clippy::too_many_arguments)]
pub fn vit_one_step(
    pf: &PackedForward,
    dit: &WanTransformer,
    mode: VitMode,
    noisy: &Tensor,
    images: &[(Tensor, f64)],
    videos: &[(Tensor, f64)],
    t: f64,
    sigma: f32,
    streams: &VitStreams,
    g: &VitGuidanceParams,
    mbufs: &mut [MomentumBuffer],
) -> CResult<Tensor> {
    let wvae: Vec<(Tensor, f64)> = images.iter().chain(videos).cloned().collect();
    let v = |sources: &[(Tensor, f64)], ctx: &Tensor| pf.velocity(dit, noisy, sources, t, ctx);

    match mode {
        VitMode::VaeTxtVit | VitMode::VaeTxtVitWapg => {
            let base = v(&[], streams.wotxt_wovit)?; // wovae · wotxt_wovit
            let img = v(&wvae, streams.wotxt_wovit)?; // wvae  · wotxt_wovit
            let txt = v(&wvae, streams.wtxt_wovit)?; // wvae  · wtxt_wovit
            let vit = v(&wvae, streams.wtxt_wvit)?; // wvae  · wtxt_wvit
            vae_txt_vit(
                &base,
                &img,
                &txt,
                &vit,
                g.omega_img,
                g.omega_txt,
                g.omega_tgt,
                mode.apg(),
            )
        }
        VitMode::Rv2vWapg | VitMode::R2vWapg => {
            let base = v(&[], streams.wotxt_wovit)?;
            // `if cur_omega_X > 0` short-circuits (reuse the previous prediction, no extra forward).
            let eps_v = if g.omega_vid > 0.0 {
                v(videos, streams.wotxt_wovit)? // wvidvae · wotxt_wovit
            } else {
                base.clone()
            };
            let eps_vi = if g.omega_img > 0.0 {
                v(&wvae, streams.wotxt_wovit)? // wvae · wotxt_wovit
            } else {
                eps_v.clone()
            };
            let eps_vti = if g.omega_txt > 0.0 {
                v(&wvae, streams.wtxt_wovit)? // wvae · wtxt_wovit
            } else {
                eps_vi.clone()
            };
            let eps_vtic = if g.omega_tgt > 0.0 {
                v(&wvae, streams.wtxt_wvit)? // wvae · wtxt_wvit
            } else {
                eps_vti.clone()
            };
            rv2v_chain(
                &base,
                &eps_v,
                &eps_vi,
                &eps_vti,
                &eps_vtic,
                g.omega_vid,
                g.omega_img,
                g.omega_txt,
                g.omega_tgt,
                mode.apg(),
            )
        }
        VitMode::V2vApg => {
            let eps_uncond = v(&[], streams.wotxt_wovit)?; // wovae · wotxt_wovit
            let eps_t = v(&wvae, streams.wtxt_wvit)?; // wvae · wtxt_wvit
            let x0 = to_x(noisy, sigma, &eps_uncond)?;
            let xt = to_x(noisy, sigma, &eps_t)?;
            // Persistent momentum buffer threaded from the caller (F-161): reuse `mbufs[0]` so the
            // running average carries across denoise steps; degrade to no momentum if none was provided.
            let xg = normalized_guidance(
                &xt,
                &x0,
                g.omega_txt,
                mbufs.first_mut(),
                g.eta,
                g.norm_threshold,
            )?;
            from_x(noisy, sigma, &xg)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::{DType, Device, Tensor};
    use candle_gen::candle_nn::VarBuilder;
    use std::collections::HashMap;

    /// A tiny dense DiT config filled by the synthetic weights below (dim 16 = 2 heads × head_dim 8, z16
    /// in/out, patch (1,2,2)) — the packed geometry without real weights, runnable on CPU.
    fn tiny_cfg() -> TransformerConfig {
        TransformerConfig {
            in_channels: 16,
            out_channels: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            dim: 16,
            ffn_dim: 32,
            freq_dim: 16,
            text_dim: 16,
            patch: (1, 2, 2),
            eps: 1e-6,
            rope_theta: 10000.0,
            rope_max_seq_len: 64,
        }
    }

    fn tiny_dit(cfg: &TransformerConfig, dev: &Device) -> WanTransformer {
        let mut m: HashMap<String, Tensor> = HashMap::new();
        let mut put = |k: &str, shape: &[usize]| {
            m.insert(
                k.to_string(),
                Tensor::randn(0f32, 0.2f32, shape, dev).unwrap(),
            );
        };
        let (pt, ph, pw) = cfg.patch;
        let d = cfg.dim;
        put("patch_embedding.weight", &[d, cfg.in_channels, pt, ph, pw]);
        put("patch_embedding.bias", &[d]);
        put(
            "condition_embedder.text_embedder.linear_1.weight",
            &[d, cfg.text_dim],
        );
        put("condition_embedder.text_embedder.linear_1.bias", &[d]);
        put("condition_embedder.text_embedder.linear_2.weight", &[d, d]);
        put("condition_embedder.text_embedder.linear_2.bias", &[d]);
        put(
            "condition_embedder.time_embedder.linear_1.weight",
            &[d, cfg.freq_dim],
        );
        put("condition_embedder.time_embedder.linear_1.bias", &[d]);
        put("condition_embedder.time_embedder.linear_2.weight", &[d, d]);
        put("condition_embedder.time_embedder.linear_2.bias", &[d]);
        put("condition_embedder.time_proj.weight", &[6 * d, d]);
        put("condition_embedder.time_proj.bias", &[6 * d]);
        for i in 0..cfg.num_layers {
            let b = format!("blocks.{i}");
            put(&format!("{b}.scale_shift_table"), &[1, 6, d]);
            for attn in ["attn1", "attn2"] {
                for leaf in ["to_q", "to_k", "to_v", "to_out.0"] {
                    put(&format!("{b}.{attn}.{leaf}.weight"), &[d, d]);
                    put(&format!("{b}.{attn}.{leaf}.bias"), &[d]);
                }
                put(&format!("{b}.{attn}.norm_q.weight"), &[d]);
                put(&format!("{b}.{attn}.norm_k.weight"), &[d]);
            }
            put(&format!("{b}.norm2.weight"), &[d]);
            put(&format!("{b}.norm2.bias"), &[d]);
            put(&format!("{b}.ffn.net.0.proj.weight"), &[cfg.ffn_dim, d]);
            put(&format!("{b}.ffn.net.0.proj.bias"), &[cfg.ffn_dim]);
            put(&format!("{b}.ffn.net.2.weight"), &[d, cfg.ffn_dim]);
            put(&format!("{b}.ffn.net.2.bias"), &[d]);
        }
        put("proj_out.weight", &[cfg.out_channels * pt * ph * pw, d]);
        put("proj_out.bias", &[cfg.out_channels * pt * ph * pw]);
        put("scale_shift_table", &[1, 2, d]);
        let vb = VarBuilder::from_tensors(m, DType::F32, dev);
        WanTransformer::new(cfg, vb).unwrap()
    }

    fn max_abs(a: &Tensor, b: &Tensor) -> f32 {
        (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    fn params() -> GuidanceParams {
        GuidanceParams {
            omega_vid: 1.0,
            omega_img: 1.0,
            omega_txt: 4.0,
            eta: 0.5,
            norm_threshold: [50.0, 50.0],
        }
    }

    /// `t2v` (plain CFG over the target-only combo) computed by [`guided_velocity`] must equal the
    /// hand-written `uncond + ω·(cond − uncond)` over two [`PackedForward::velocity`] forwards — pins the
    /// mode-dispatch plumbing to the packed-forward seam (ported from mlx `t2v_mode_matches_manual_cfg`;
    /// an internal-consistency test, no real weights / no conditioning).
    #[test]
    fn t2v_mode_matches_manual_cfg() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let dit = tiny_dit(&cfg, &dev);
        let pf = PackedForward::new(cfg, 5.0, true);
        let noisy = Tensor::randn(0f32, 1f32, (1, 16, 2, 4, 4), &dev).unwrap();
        let ctx_c = Tensor::randn(0f32, 1f32, (1, 3, 16), &dev).unwrap();
        let ctx_u = Tensor::randn(0f32, 1f32, (1, 3, 16), &dev).unwrap();
        let (t, omega) = (833.0, 4.0f32);
        let g = params();
        let mut mbufs: Vec<MomentumBuffer> = Vec::new();
        let got = guided_velocity(
            &pf,
            Mode::T2v,
            &dit,
            &noisy,
            &[],
            &[],
            t,
            1.0,
            &ctx_c,
            &ctx_u,
            &g,
            &mut mbufs,
        )
        .unwrap();
        // Manual: uncond + ω·(cond − uncond) over the target-only packed forward.
        let e_u = pf.velocity(&dit, &noisy, &[], t, &ctx_u).unwrap();
        let e_c = pf.velocity(&dit, &noisy, &[], t, &ctx_c).unwrap();
        let want = (&e_u + ((&e_c - &e_u).unwrap() * omega as f64).unwrap()).unwrap();
        assert_eq!(got.dims(), noisy.dims());
        assert_eq!(max_abs(&got, &want), 0.0, "t2v must equal manual CFG");
    }

    /// The video-only guidance combo (`Combos::v`) must carry EVERY conditioning video, mirroring the
    /// `vi` combo — otherwise clips 2..n are silently dropped from the V2vChain/Rv2v video-only delta
    /// (F-021, ported from mlx). Pure host-side combo assembly: no DiT / weights.
    #[test]
    fn build_combos_includes_every_conditioning_video() {
        let dev = Device::Cpu;
        let pf = PackedForward::new(tiny_cfg(), 5.0, true);
        let v0 = Tensor::zeros((1, 16, 1, 4, 4), DType::F32, &dev).unwrap();
        let v1 = Tensor::ones((1, 16, 1, 4, 4), DType::F32, &dev).unwrap();
        let c = pf.build_combos(&[v0.clone(), v1.clone()], &[]);
        assert_eq!(c.v.len(), 2, "video-only combo must carry both videos");
        assert!(max_abs(&c.v[1].0, &v0) > 0.0, "v[1] must be the 2nd clip");
        // source-ids mirror the `vi` combo's video portion (videos-first ordering).
        assert_eq!(c.v[0].1, c.vi[0].1);
        assert_eq!(c.v[1].1, c.vi[1].1);
        // nv == 1 stays equivalent to a single-video build; nv == 0 → empty.
        let one = pf.build_combos(std::slice::from_ref(&v0), &[]);
        assert_eq!(one.v.len(), 1);
        assert_eq!(one.v[0].1, one.vi[0].1);
        assert!(pf.build_combos(&[], &[]).v.is_empty());
    }

    /// F-098: memoizing the step-invariant RoPE tables + source patch-embeds must not change the
    /// velocity. A cached `PackedForward` (whose caches are warm after the first call, and which packs
    /// the same source under two different source-ids across combos) must produce bit-identical
    /// velocities to a fresh, cache-cold `PackedForward` on every call.
    #[test]
    fn cached_velocity_matches_uncached() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let dit = tiny_dit(&cfg, &dev);
        let noisy = Tensor::randn(0f32, 1f32, (1, 16, 2, 4, 4), &dev).unwrap();
        let ctx = Tensor::randn(0f32, 1f32, (1, 3, 16), &dev).unwrap();
        let img = Tensor::randn(0f32, 1f32, (1, 16, 1, 4, 4), &dev).unwrap();

        let pf = PackedForward::new(cfg, 5.0, true);
        // Warm the caches (target RoPE for this grid, source tokens + RoPE at id 1.0).
        let warm = pf
            .velocity(&dit, &noisy, &[(img.clone(), 1.0)], 833.0, &ctx)
            .unwrap();
        // Second call on the warm caches must be bit-identical.
        let hot = pf
            .velocity(&dit, &noisy, &[(img.clone(), 1.0)], 833.0, &ctx)
            .unwrap();
        assert_eq!(
            max_abs(&warm, &hot),
            0.0,
            "warm-cache velocity must be stable"
        );
        // A fresh, cache-cold engine must match the cached one.
        let cold = PackedForward::new(cfg, 5.0, true);
        let fresh = cold
            .velocity(&dit, &noisy, &[(img.clone(), 1.0)], 833.0, &ctx)
            .unwrap();
        assert_eq!(
            max_abs(&warm, &fresh),
            0.0,
            "cache must equal a fresh build"
        );
        // The SAME source latent re-embedded under a different source-id must give a different RoPE
        // (so the patch-embed cache — keyed on latent, not source-id — never leaks a stale table).
        let id2 = pf
            .velocity(&dit, &noisy, &[(img.clone(), 2.0)], 833.0, &ctx)
            .unwrap();
        let id2_cold = cold
            .velocity(&dit, &noisy, &[(img, 2.0)], 833.0, &ctx)
            .unwrap();
        assert_eq!(max_abs(&id2, &id2_cold), 0.0, "id-2 cached == fresh");
        assert!(
            max_abs(&warm, &id2) > 0.0,
            "distinct source-ids must differ"
        );
    }

    /// A conditioning image extends the packed sequence but the sliced target velocity keeps the target's
    /// shape (ported from mlx `conditioning_source_preserves_target_shape`).
    #[test]
    fn conditioning_source_preserves_target_shape() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let dit = tiny_dit(&cfg, &dev);
        let pf = PackedForward::new(cfg, 5.0, true);
        let noisy = Tensor::randn(0f32, 1f32, (1, 16, 2, 4, 4), &dev).unwrap();
        let ctx = Tensor::randn(0f32, 1f32, (1, 3, 16), &dev).unwrap();
        let img = Tensor::zeros((1, 16, 1, 4, 4), DType::F32, &dev).unwrap();
        let vel = pf
            .velocity(&dit, &noisy, &[(img, 1.0)], 833.0, &ctx)
            .unwrap();
        assert_eq!(
            vel.dims(),
            noisy.dims(),
            "target velocity keeps target shape"
        );
    }

    /// Every advertised mode actually runs end-to-end over the packed forward with the conditioning it
    /// consumes (no stubbed mode) and yields a finite, target-shaped velocity — the "a mode you advertise
    /// MUST run" bar. Exercises all 7 renderer modes.
    #[test]
    fn all_seven_modes_run_and_keep_shape() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let dit = tiny_dit(&cfg, &dev);
        let pf = PackedForward::new(cfg, 5.0, true);
        let noisy = Tensor::randn(0f32, 1f32, (1, 16, 5, 4, 4), &dev).unwrap();
        let ctx_c = Tensor::randn(0f32, 1f32, (1, 3, 16), &dev).unwrap();
        let ctx_u = Tensor::randn(0f32, 1f32, (1, 3, 16), &dev).unwrap();
        let g = params();
        // One video clip (5 frames → t_lat 2) + one reference image.
        let vid = Tensor::randn(0f32, 1f32, (1, 16, 2, 4, 4), &dev).unwrap();
        let im = Tensor::randn(0f32, 1f32, (1, 16, 1, 4, 4), &dev).unwrap();
        for mode in [
            Mode::T2v,
            Mode::T2vApg,
            Mode::V2v,
            Mode::V2vChain,
            Mode::V2vApg,
            Mode::R2vApg,
            Mode::Rv2v,
        ] {
            let mut mbufs: Vec<MomentumBuffer> = (0..num_momentum_buffers(mode))
                .map(|_| MomentumBuffer::new(0.0))
                .collect();
            let (videos, images): (Vec<Tensor>, Vec<Tensor>) = match mode {
                Mode::T2v | Mode::T2vApg => (vec![], vec![]),
                Mode::R2vApg => (vec![], vec![im.clone()]),
                Mode::V2v => (vec![], vec![im.clone()]),
                _ => (vec![vid.clone()], vec![im.clone()]),
            };
            let out = guided_velocity(
                &pf, mode, &dit, &noisy, &videos, &images, 700.0, 0.9, &ctx_c, &ctx_u, &g,
                &mut mbufs,
            )
            .unwrap_or_else(|e| panic!("mode {mode:?} failed: {e}"));
            assert_eq!(out.dims(), noisy.dims(), "{mode:?} keeps target shape");
            let finite = out
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .all(|x| x.is_finite());
            assert!(finite, "{mode:?} produced non-finite velocity");
        }
    }

    /// Every full-Bernini ViT-conditioned mode runs end-to-end over the packed forward with the
    /// planner's 4 prompt streams + its conditioning, and yields a finite, target-shaped velocity — the
    /// "a mode you advertise MUST run" bar for the `*_wapg` / `v2v_apg` planner path (sc-10995).
    #[test]
    fn all_five_vit_modes_run_and_keep_shape() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let dit = tiny_dit(&cfg, &dev);
        let pf = PackedForward::new(cfg, 5.0, true);
        let noisy = Tensor::randn(0f32, 1f32, (1, 16, 5, 4, 4), &dev).unwrap();
        // 4 distinct prompt streams (each already `embed_text`-projected to the expert's context space).
        let mk = |s: f32| {
            let r = Tensor::randn(0f32, 1f32, (1, 3, 16), &dev).unwrap();
            dit.embed_text(&r.affine(s as f64, 0.0).unwrap()).unwrap()
        };
        let (s0, s1, s2, s3) = (mk(1.0), mk(0.7), mk(0.4), mk(0.2));
        let streams = VitStreams {
            wtxt_wvit: &s0,
            wtxt_wovit: &s1,
            wotxt_wvit: &s2,
            wotxt_wovit: &s3,
        };
        let vid = Tensor::randn(0f32, 1f32, (1, 16, 2, 4, 4), &dev).unwrap();
        let im = Tensor::randn(0f32, 1f32, (1, 16, 1, 4, 4), &dev).unwrap();
        let g = VitGuidanceParams {
            omega_txt: 4.0,
            omega_img: 4.5,
            omega_vid: 1.25,
            omega_tgt: 0.5,
            eta: 1.0,
            norm_threshold: 50.0,
        };
        for mode in [
            VitMode::VaeTxtVit,
            VitMode::VaeTxtVitWapg,
            VitMode::Rv2vWapg,
            VitMode::R2vWapg,
            VitMode::V2vApg,
        ] {
            #[allow(clippy::type_complexity)]
            let (images, videos): (Vec<(Tensor, f64)>, Vec<(Tensor, f64)>) = match mode {
                VitMode::VaeTxtVit | VitMode::VaeTxtVitWapg => (vec![(im.clone(), 1.0)], vec![]),
                _ => (vec![(im.clone(), 1.0)], vec![(vid.clone(), 2.0)]),
            };
            let mut mbufs: Vec<MomentumBuffer> = (0..num_vit_momentum_buffers(mode))
                .map(|_| MomentumBuffer::new(0.0))
                .collect();
            let out = vit_one_step(
                &pf, &dit, mode, &noisy, &images, &videos, 700.0, 0.9, &streams, &g, &mut mbufs,
            )
            .unwrap_or_else(|e| panic!("vit mode {mode:?} failed: {e}"));
            assert_eq!(out.dims(), noisy.dims(), "{mode:?} keeps target shape");
            let finite = out
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .all(|x| x.is_finite());
            assert!(finite, "{mode:?} produced non-finite velocity");
        }
    }

    /// F-096/F-161: the ViT-mode conditioning requirement + momentum-buffer count mappings.
    #[test]
    fn vit_mode_conditioning_and_momentum_maps() {
        // t2i/i2i modes accept an optional source → no requirement.
        assert!(!VitMode::VaeTxtVit.needs_conditioning());
        assert!(!VitMode::VaeTxtVitWapg.needs_conditioning());
        // reference-video / video-edit modes require a source.
        assert!(VitMode::Rv2vWapg.needs_conditioning());
        assert!(VitMode::R2vWapg.needs_conditioning());
        assert!(VitMode::V2vApg.needs_conditioning());
        // Only the x-space v2v_apg stream carries a momentum buffer.
        assert_eq!(num_vit_momentum_buffers(VitMode::V2vApg), 1);
        for m in [
            VitMode::VaeTxtVit,
            VitMode::VaeTxtVitWapg,
            VitMode::Rv2vWapg,
            VitMode::R2vWapg,
        ] {
            assert_eq!(num_vit_momentum_buffers(m), 0);
        }
    }

    /// F-161: the `v2v_apg` momentum buffer is threaded from the caller, so a persistent (nonzero-
    /// momentum) buffer carries its running average across steps — a fresh buffer each step would not.
    /// The second step sharing a buffer with the first must differ from that same step run against a
    /// fresh buffer.
    #[test]
    fn v2v_apg_momentum_persists_across_steps() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let dit = tiny_dit(&cfg, &dev);
        let pf = PackedForward::new(cfg, 5.0, true);
        let noisy = Tensor::randn(0f32, 1f32, (1, 16, 2, 4, 4), &dev).unwrap();
        let mk = |s: f32| {
            let r = Tensor::randn(0f32, 1f32, (1, 3, 16), &dev).unwrap();
            dit.embed_text(&r.affine(s as f64, 0.0).unwrap()).unwrap()
        };
        let (s0, s1, s2, s3) = (mk(1.0), mk(0.7), mk(0.4), mk(0.2));
        let streams = VitStreams {
            wtxt_wvit: &s0,
            wtxt_wovit: &s1,
            wotxt_wvit: &s2,
            wotxt_wovit: &s3,
        };
        let vid = Tensor::randn(0f32, 1f32, (1, 16, 2, 4, 4), &dev).unwrap();
        let videos = [(vid, 2.0)];
        let g = VitGuidanceParams {
            omega_txt: 4.0,
            omega_img: 4.5,
            omega_vid: 1.25,
            omega_tgt: 0.5,
            eta: 1.0,
            norm_threshold: 50.0,
        };
        let run = |mbufs: &mut [MomentumBuffer], t: f64| {
            vit_one_step(
                &pf,
                &dit,
                VitMode::V2vApg,
                &noisy,
                &[],
                &videos,
                t,
                0.9,
                &streams,
                &g,
                mbufs,
            )
            .unwrap()
        };
        // Shared buffer across two steps (momentum 0.5): step 2 sees step 1's running average.
        let mut shared: Vec<MomentumBuffer> = vec![MomentumBuffer::new(0.5)];
        let _ = run(&mut shared, 800.0);
        let carried = run(&mut shared, 700.0);
        // Same second step but with a fresh buffer (no carried history — the old per-call behavior).
        let mut fresh: Vec<MomentumBuffer> = vec![MomentumBuffer::new(0.5)];
        let fresh_second = run(&mut fresh, 700.0);
        assert!(
            max_abs(&carried, &fresh_second) > 0.0,
            "persistent momentum must change the second step"
        );
    }

    /// `vae_txt_vit` dispatch: [`vit_one_step`] must equal the manual 4-forward combine via
    /// [`crate::vit_guidance::vae_txt_vit`] over the right (source-subset × prompt-stream) pairs — pins
    /// the `sample_one_step` stream/source routing to the packed-forward seam. Distinct streams make a
    /// wrong routing show up.
    #[test]
    fn vae_txt_vit_dispatch_matches_manual() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let dit = tiny_dit(&cfg, &dev);
        let pf = PackedForward::new(cfg, 5.0, true);
        let noisy = Tensor::randn(0f32, 1f32, (1, 16, 2, 4, 4), &dev).unwrap();
        let mk = |s: f32| {
            let r = Tensor::randn(0f32, 1f32, (1, 3, 16), &dev).unwrap();
            dit.embed_text(&r.affine(s as f64, 0.0).unwrap()).unwrap()
        };
        let (s0, s1, s2, s3) = (mk(1.0), mk(0.7), mk(0.4), mk(0.2));
        let streams = VitStreams {
            wtxt_wvit: &s0,
            wtxt_wovit: &s1,
            wotxt_wvit: &s2,
            wotxt_wovit: &s3,
        };
        let im = Tensor::zeros((1, 16, 1, 4, 4), DType::F32, &dev).unwrap();
        let images = [(im, 1.0)];
        let g = VitGuidanceParams {
            omega_txt: 4.0,
            omega_img: 4.5,
            omega_vid: 1.25,
            omega_tgt: 3.0,
            eta: 1.0,
            norm_threshold: 50.0,
        };
        let mut mbufs: Vec<MomentumBuffer> = Vec::new();
        let got = vit_one_step(
            &pf,
            &dit,
            VitMode::VaeTxtVit,
            &noisy,
            &images,
            &[],
            833.0,
            1.0,
            &streams,
            &g,
            &mut mbufs,
        )
        .unwrap();
        // Manual: the four shared_step forwards, combined by the (separately validated) combine math.
        let base = pf.velocity(&dit, &noisy, &[], 833.0, &s3).unwrap();
        let img = pf.velocity(&dit, &noisy, &images, 833.0, &s3).unwrap();
        let tx = pf.velocity(&dit, &noisy, &images, 833.0, &s1).unwrap();
        let vi = pf.velocity(&dit, &noisy, &images, 833.0, &s0).unwrap();
        let want = vae_txt_vit(
            &base,
            &img,
            &tx,
            &vi,
            g.omega_img,
            g.omega_txt,
            g.omega_tgt,
            false,
        )
        .unwrap();
        assert_eq!(got.dims(), noisy.dims());
        assert_eq!(
            max_abs(&got, &want),
            0.0,
            "vae_txt_vit dispatch must equal manual combine"
        );
    }
}
