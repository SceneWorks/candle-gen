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

use candle_gen::candle_core::Tensor;
use candle_gen::Result as CResult;

use candle_gen_wan::config::TransformerConfig;
use candle_gen_wan::rope::{apply_source_id, assign_source_ids, WanRope};
use candle_gen_wan::transformer::WanTransformer;

use crate::config::Mode;
use crate::guidance::{normalized_guidance, normalized_guidance_chain, MomentumBuffer};

/// The packed-forward engine: holds the transformer geometry (RoPE builder + patch grid) so it can
/// patch-embed the target and each conditioning source with their source-id RoPE and run one packed
/// forward. Cheap + immutable; build once per render.
pub struct PackedForward {
    cfg: TransformerConfig,
    max_trained_src_id: f64,
    interpolate_src_id: bool,
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
        }
    }

    /// Patch-embed one latent `[1, 16, T, H8, W8]` to `(tokens [1,L,dim], cos, sin, grid)` with the
    /// source-id RoPE folded in. `cos`/`sin` are `[L, head_dim/2]` (f32); they are concatenated on the
    /// token axis before the forward.
    #[allow(clippy::type_complexity)]
    fn embed_segment(
        &self,
        dit: &WanTransformer,
        latent: &Tensor,
        source_id: f64,
    ) -> CResult<(Tensor, Tensor, Tensor, (usize, usize, usize))> {
        let (tokens, grid) = dit.patch_embed_tokens(latent)?;
        let (cos, sin) =
            WanRope::new(&self.cfg).cos_sin(grid.0, grid.1, grid.2, latent.device())?;
        let (cos, sin) = apply_source_id(&cos, &sin, source_id, self.cfg.head_dim)?;
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
        let (tk_t, c_t, s_t, grid_t) = self.embed_segment(dit, target, 0.0)?;
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
}
