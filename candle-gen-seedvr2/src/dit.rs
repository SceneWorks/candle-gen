//! SeedVR2 dual-stream MMDiT — candle port of `mlx-gen-seedvr2/src/dit.rs`, image-mode (B=1) parity.
//!
//! `txt_in` (proj precomputed neg-prompt embed) + `vid_in` (3-D patchify) + `emb_in` (sinusoidal
//! timestep → AdaLN params) → N dual-stream blocks → `vid_out_norm`+out-AdaLN → `vid_out` (unpatchify).
//! Each block: RMSNorm → AdaLN-in → windowed joint attention (QK-norm + 3-D axial RoPE) → AdaLN-out →
//! residual → RMSNorm → AdaLN-in → SwiGLU/GELU → AdaLN-out → residual. The window partition is
//! data-independent (grid+window+shift) so the forward/reverse permutations + per-window RoPE freqs
//! are built once per shift parity and shared across blocks.
//!
//! Each `Linear` is dense (the loaded bf16/f32 weight) or GGUF-quantized (`Q4_0`/`Q8_0`); the
//! `quantize` cascade ([`crate::quant`], sc-5927) folds every DiT Linear in place after load.

use candle_gen::candle_core::{DType, Device, Result, Tensor};
use candle_gen::gen_core::Quant;
use candle_gen::quant::{DenseLinear, QLinear};

use crate::config::DitConfig;
use crate::nn;
use crate::weights::Weights;

type CResult<T> = candle_gen::Result<T>;

// ---------------------------------------------------------------------------
// small leaves
// ---------------------------------------------------------------------------

/// A `[out,in]` weight linear (`y = x·Wᵀ (+ b)`) that is **dense** (the loaded bf16/f32 weight) or
/// **GGUF-quantized** (`Q4_0`/`Q8_0` blocks + an f32 bias; sc-5927). Built dense by [`Self::load`];
/// [`Self::quantize`] folds it in place. The quantized forward runs the GGUF `QMatMul` in f32 — the
/// CPU and CUDA dmmv paths both need an f32 activation — and casts back to the input dtype, so the
/// surrounding DiT keeps flowing bf16 exactly as the dense path does.
///
/// **Thin newtype over the shared [`candle_gen::quant::QLinear`] seam (F-025 / sc-9005).** SeedVR2 was
/// one of four drifted copies of the `Dense|Quantized` Linear; the seam now lives once in `candle-gen`.
/// SeedVR2's load-bearing behaviors are preserved as explicit knobs: the **pre-transposed** dense
/// layout (sc-8997/F-017), candle's **int8 `QMatMul`** forward ([`MatmulStrategy::Int8Fast`]), the
/// **`in_features % 32` skip** predicate (leaves `vid_in.proj` in=132 dense), the **leading-dim
/// flatten**, and the **cast-back** to the input dtype (the DiT flows bf16).
struct Linear(QLinear);
impl Linear {
    fn load(w: &Weights, prefix: &str, bias: bool) -> CResult<Self> {
        let weight = w.require(&format!("{prefix}.weight"))?; // [out, in]
        Ok(Self(QLinear::from_dense(DenseLinear::Transposed {
            weight_t: nn::transpose_weight(weight)?, // [in, out], once at load (sc-8997/F-017)
            bias: if bias {
                Some(w.require(&format!("{prefix}.bias"))?.clone())
            } else {
                None
            },
        })))
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.0.forward(x)
    }

    /// Fold a dense `[out,in]` weight to `Q4_0`/`Q8_0` in place (sc-5927). No-op when already
    /// quantized or when `in_features` is not a multiple of the GGUF block (`vid_in.proj`, in=132,
    /// stays dense — the reference predicate). Uses candle's int8 `QMatMul` forward, flattens the
    /// leading dims, and casts back to the input dtype (the shared seam's int8-fast fold with SeedVR2's
    /// exact knobs). The weight is quantized on the CPU and placed back on its original device; the bias
    /// is promoted to f32 for the post-matmul add.
    fn quantize(&mut self, quant: Quant) -> CResult<()> {
        // skip_indivisible = true (the `% 32` predicate), flatten_leading = true, cast_back = true.
        self.0.quantize_int8_fast(quant, true, true, true)?;
        Ok(())
    }
}

fn arange_f32(n: usize, dev: &Device) -> Result<Tensor> {
    Tensor::from_vec((0..n).map(|i| i as f32).collect::<Vec<_>>(), n, dev)
}

/// fast RMSNorm with a unit (`ones`) weight — the block pre-norms have no learnable scale. The
/// `ones` weight is built once per block at load (`Block::ones`) and passed in, rather than
/// re-allocated on every call inside the per-step hot loop (sc-9039).
fn rms_plain(x: &Tensor, ones: &Tensor, eps: f64) -> Result<Tensor> {
    nn::rms_norm(x, ones, eps)
}

// ---------------------------------------------------------------------------
// time embedding
// ---------------------------------------------------------------------------

struct TimeEmbedding {
    proj_in: Linear,
    proj_hid: Linear,
    proj_out: Linear,
    sinusoidal_dim: usize,
    /// The model dtype (captured from the dense weight at load) — the sinusoid is built in f32 and
    /// cast to this before the projections, so it survives `proj_in` becoming a quantized `QMatMul`.
    dtype: DType,
}
impl TimeEmbedding {
    fn load(w: &Weights, prefix: &str) -> CResult<Self> {
        let dtype = w.require(&format!("{prefix}.proj_in.weight"))?.dtype();
        Ok(Self {
            proj_in: Linear::load(w, &format!("{prefix}.proj_in"), true)?,
            proj_hid: Linear::load(w, &format!("{prefix}.proj_hid"), true)?,
            proj_out: Linear::load(w, &format!("{prefix}.proj_out"), true)?,
            sinusoidal_dim: 256,
            dtype,
        })
    }
    fn quantize(&mut self, quant: Quant) -> CResult<()> {
        self.proj_in.quantize(quant)?;
        self.proj_hid.quantize(quant)?;
        self.proj_out.quantize(quant)
    }
    /// Scalar `timestep` (B=1) → AdaLN feature `(1, emb_dim)`.
    fn forward(&self, timestep: f64, dev: &Device) -> Result<Tensor> {
        let half = self.sinusoidal_dim / 2;
        let scale = -(10000f64.ln()) / half as f64;
        let freqs: Vec<f32> = (0..half).map(|i| (i as f64 * scale).exp() as f32).collect();
        let freqs = Tensor::from_vec(freqs, (1, half), dev)?;
        let args = (freqs * timestep)?; // (1, half)
                                        // Sinusoids are built in f32 for precision; cast to the model dtype before the proj Linears
                                        // so the GEMM dtypes match (bf16 weights would otherwise reject the f32 activation).
        let emb = Tensor::cat(&[&args.sin()?, &args.cos()?], 1)?.to_dtype(self.dtype)?; // (1, 256)
        let emb = nn::silu(&self.proj_in.forward(&emb)?)?;
        let emb = nn::silu(&self.proj_hid.forward(&emb)?)?;
        self.proj_out.forward(&emb)
    }
}

// ---------------------------------------------------------------------------
// patch in / out
// ---------------------------------------------------------------------------

struct PatchIn {
    proj: Linear,
    pt: usize,
    ph: usize,
    pw: usize,
}
impl PatchIn {
    fn load(w: &Weights, prefix: &str, cfg: &DitConfig) -> CResult<Self> {
        Ok(Self {
            proj: Linear::load(w, &format!("{prefix}.proj"), true)?,
            pt: cfg.patch_t,
            ph: cfg.patch_h,
            pw: cfg.patch_w,
        })
    }
    /// `(B,C,T,H,W)` → tokens `(B, Tp*Hp*Wp, dim)` + `(Tp,Hp,Wp)`.
    fn forward(&self, vid: &Tensor) -> Result<(Tensor, (usize, usize, usize))> {
        let (b, c, t, h, wd) = vid.dims5()?;
        let (tp, hp, wp) = (t / self.pt, h / self.ph, wd / self.pw);
        let x = vid
            .reshape(&[b, c, tp, self.pt, hp, self.ph, wp, self.pw][..])?
            .permute([0usize, 2, 4, 6, 3, 5, 7, 1])?
            .contiguous()?
            .reshape((b, tp, hp, wp, self.pt * self.ph * self.pw * c))?;
        let x = self.proj.forward(&x)?;
        let dim = *x.dims().last().unwrap();
        Ok((x.reshape((b, tp * hp * wp, dim))?, (tp, hp, wp)))
    }
    fn quantize(&mut self, quant: Quant) -> CResult<()> {
        self.proj.quantize(quant) // in = patch·channels = 132 → left dense by the predicate
    }
}

struct PatchOut {
    proj: Linear,
    pt: usize,
    ph: usize,
    pw: usize,
}
impl PatchOut {
    fn load(w: &Weights, prefix: &str, cfg: &DitConfig) -> CResult<Self> {
        Ok(Self {
            proj: Linear::load(w, &format!("{prefix}.proj"), true)?,
            pt: cfg.patch_t,
            ph: cfg.patch_h,
            pw: cfg.patch_w,
        })
    }
    fn forward(&self, vid: &Tensor, shape: (usize, usize, usize)) -> Result<Tensor> {
        let (tp, hp, wp) = shape;
        let x = self.proj.forward(vid)?;
        let b = x.dim(0)?;
        let cc = *x.dims().last().unwrap() / (self.pt * self.ph * self.pw);
        x.reshape(&[b, tp, hp, wp, self.pt, self.ph, self.pw, cc][..])?
            .permute([0usize, 7, 1, 4, 2, 5, 3, 6])?
            .contiguous()?
            .reshape((b, cc, tp * self.pt, hp * self.ph, wp * self.pw))
    }
    fn quantize(&mut self, quant: Quant) -> CResult<()> {
        self.proj.quantize(quant)
    }
}

// ---------------------------------------------------------------------------
// window partition (host-side, data-independent)
// ---------------------------------------------------------------------------

/// python `round` (round-half-to-even).
fn py_round(x: f64) -> i64 {
    let f = x.floor();
    let diff = x - f;
    if (diff - 0.5).abs() < 1e-9 {
        let fi = f as i64;
        if fi % 2 == 0 {
            fi
        } else {
            fi + 1
        }
    } else {
        x.round() as i64
    }
}
fn ceil_div_f(a: f64, b: f64) -> i64 {
    (a / b).ceil() as i64
}

/// Replicates `WindowPartitioner._make_windows`. Returns each window's `(t0,t1,h0,h1,w0,w1)`.
fn make_windows(
    t: i64,
    h: i64,
    w: i64,
    window: (i64, i64, i64),
    shift: bool,
) -> Vec<(i64, i64, i64, i64, i64, i64)> {
    let (nt_w, nh_w, nw_w) = window;
    let scale = ((45.0 * 80.0) / (h as f64 * w as f64)).sqrt();
    let resized_h = py_round(h as f64 * scale) as f64;
    let resized_w = py_round(w as f64 * scale) as f64;
    let wh = ceil_div_f(resized_h, nh_w as f64);
    let ww = ceil_div_f(resized_w, nw_w as f64);
    let wt = ceil_div_f(t.min(30) as f64, nt_w as f64);

    let (st, sh_, sw_, nt, nh, nw);
    if shift {
        st = if wt < t { 0.5 } else { 0.0 };
        sh_ = if wh < h { 0.5 } else { 0.0 };
        sw_ = if ww < w { 0.5 } else { 0.0 };
        nt = if st > 0.0 {
            ceil_div_f(t as f64 - st, wt as f64) + 1
        } else {
            1
        };
        nh = if sh_ > 0.0 {
            ceil_div_f(h as f64 - sh_, wh as f64) + 1
        } else {
            1
        };
        nw = if sw_ > 0.0 {
            ceil_div_f(w as f64 - sw_, ww as f64) + 1
        } else {
            1
        };
    } else {
        st = 0.0;
        sh_ = 0.0;
        sw_ = 0.0;
        nt = ceil_div_f(t as f64, wt as f64);
        nh = ceil_div_f(h as f64, wh as f64);
        nw = ceil_div_f(w as f64, ww as f64);
    }

    let mut out = Vec::new();
    for iw in 0..nw {
        let w0 = (((iw as f64 - sw_) * ww as f64) as i64).max(0);
        let w1 = (((iw as f64 - sw_ + 1.0) * ww as f64) as i64).min(w);
        if w1 <= w0 {
            continue;
        }
        for ih in 0..nh {
            let h0 = (((ih as f64 - sh_) * wh as f64) as i64).max(0);
            let h1 = (((ih as f64 - sh_ + 1.0) * wh as f64) as i64).min(h);
            if h1 <= h0 {
                continue;
            }
            for it in 0..nt {
                let t0 = (((it as f64 - st) * wt as f64) as i64).max(0);
                let t1 = (((it as f64 - st + 1.0) * wt as f64) as i64).min(t);
                if t1 <= t0 {
                    continue;
                }
                out.push((t0, t1, h0, h1, w0, w1));
            }
        }
    }
    out
}

struct WindowPlan {
    forward_idx: Vec<u32>,
    reverse_idx: Vec<u32>,
    window_shapes: Vec<(usize, usize, usize)>, // (f,h,w) per window
}
fn window_plan(
    tp: usize,
    hp: usize,
    wp: usize,
    window: (usize, usize, usize),
    shift: bool,
) -> WindowPlan {
    let wins = make_windows(
        tp as i64,
        hp as i64,
        wp as i64,
        (window.0 as i64, window.1 as i64, window.2 as i64),
        shift,
    );
    let mut forward_idx = Vec::new();
    let mut window_shapes = Vec::new();
    for (t0, t1, h0, h1, w0, w1) in &wins {
        window_shapes.push(((t1 - t0) as usize, (h1 - h0) as usize, (w1 - w0) as usize));
        for t in *t0..*t1 {
            for h in *h0..*h1 {
                for w in *w0..*w1 {
                    forward_idx.push(((t * hp as i64 + h) * wp as i64 + w) as u32);
                }
            }
        }
    }
    let mut reverse_idx = vec![0u32; forward_idx.len()];
    for (i, &orig) in forward_idx.iter().enumerate() {
        reverse_idx[orig as usize] = i as u32;
    }
    WindowPlan {
        forward_idx,
        reverse_idx,
        window_shapes,
    }
}

// ---------------------------------------------------------------------------
// 3-D axial RoPE
// ---------------------------------------------------------------------------

/// `axial_1d(freqs, pos)`: positions `pos` (len,), base freqs `freqs` (nf,) → `(len, 2*nf)` =
/// `[p·f0, p·f0, p·f1, p·f1, …]` (each base freq duplicated). The frequency table is built in f32
/// (the `freqs` weight may be bf16, but the positions are f32 and `apply_rope` consumes this in f32) —
/// so cast `freqs` to f32 to match `pos` and keep the RoPE angles full-precision.
fn axial_1d(freqs: &Tensor, pos: &Tensor) -> Result<Tensor> {
    let len = pos.dim(0)?;
    let nf = freqs.dim(0)?;
    let freqs = freqs.to_dtype(DType::F32)?;
    let outer = pos
        .reshape((len, 1))?
        .broadcast_mul(&freqs.reshape((1, nf))?)?; // (len, nf)
    outer
        .reshape((len, nf, 1))?
        .broadcast_as((len, nf, 2))?
        .reshape((len, nf * 2))
}

/// 1-D axis positions for 3-D RoPE. lang mode (3B): `arange(n) + offset`; pixel mode (7B):
/// `linspace(-1,1,n)` (`linspace(-1,1,1) = [-1]`).
fn axis_pos(n: usize, pixel: bool, offset: i64, dev: &Device) -> Result<Tensor> {
    if pixel {
        let data: Vec<f32> = if n <= 1 {
            vec![-1.0]
        } else {
            let step = 2.0 / (n - 1) as f32;
            (0..n).map(|i| -1.0 + step * i as f32).collect()
        };
        Tensor::from_vec(data, n, dev)
    } else if offset == 0 {
        arange_f32(n, dev)
    } else {
        arange_f32(n, dev)?.affine(1.0, offset as f64)
    }
}

/// Per-window video freqs `(f*h*w, nf2*3)`; temporal positions offset by `txt_off` (lang mode).
fn vid_freq_block(
    freqs: &Tensor,
    f: usize,
    h: usize,
    w: usize,
    txt_off: i64,
    pixel: bool,
    dev: &Device,
) -> Result<Tensor> {
    let nf2 = freqs.dim(0)? * 2;
    let axt = axial_1d(freqs, &axis_pos(f, pixel, txt_off, dev)?)?
        .reshape((f, 1, 1, nf2))?
        .broadcast_as((f, h, w, nf2))?;
    let axh = axial_1d(freqs, &axis_pos(h, pixel, 0, dev)?)?
        .reshape((1, h, 1, nf2))?
        .broadcast_as((f, h, w, nf2))?;
    let axw = axial_1d(freqs, &axis_pos(w, pixel, 0, dev)?)?
        .reshape((1, 1, w, nf2))?
        .broadcast_as((f, h, w, nf2))?;
    Tensor::cat(&[&axt, &axh, &axw], 3)?.reshape((f * h * w, nf2 * 3))
}

/// Text freqs `(txt_len, nf2*3)` = the 1-D axial freqs tiled across the 3 axis slots.
fn txt_freq_block(freqs: &Tensor, txt_len: usize, dev: &Device) -> Result<Tensor> {
    let a = axial_1d(freqs, &arange_f32(txt_len, dev)?)?; // (txt_len, nf2)
    Tensor::cat(&[&a, &a, &a], 1)
}

/// rotate_half on `(..., 2k)`: pairs `(x0,x1) -> (-x1, x0)`.
fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let sh = x.dims().to_vec();
    let l = *sh.last().unwrap();
    let mut head = sh[..sh.len() - 1].to_vec();
    head.push(l / 2);
    head.push(2);
    let xr = x.reshape(head)?; // (..., k, 2)
    let nd = xr.rank() - 1;
    let x1 = xr.narrow(nd, 0, 1)?; // (..., k, 1)
    let x2 = xr.narrow(nd, 1, 1)?;
    let neg = x2.neg()?;
    Tensor::cat(&[&neg, &x1], nd)?.reshape(sh)
}

/// apply RoPE to `t` `(N, heads, head_dim)` with `freqs` `(N, rot)` (rot ≤ head_dim).
fn apply_rope(t: &Tensor, freqs: &Tensor) -> Result<Tensor> {
    let (n, _heads, hd) = t.dims3()?;
    let rot = freqs.dim(1)?;
    let t_mid = t.narrow(2, 0, rot)?; // (N,heads,rot)
    let cosf = freqs.to_dtype(DType::F32)?.cos()?.reshape((n, 1, rot))?;
    let sinf = freqs.to_dtype(DType::F32)?.sin()?.reshape((n, 1, rot))?;
    let mid_f = t_mid.to_dtype(DType::F32)?;
    let rotated = (mid_f.broadcast_mul(&cosf)? + rotate_half(&mid_f)?.broadcast_mul(&sinf)?)?
        .to_dtype(t.dtype())?;
    if rot < hd {
        let right = t.narrow(2, rot, hd - rot)?;
        Tensor::cat(&[&rotated, &right], 2)
    } else {
        Ok(rotated)
    }
}

// ---------------------------------------------------------------------------
// AdaLN modulation
// ---------------------------------------------------------------------------

struct AdaParams {
    attn_shift: Tensor,
    attn_scale: Tensor,
    attn_gate: Tensor,
    mlp_shift: Tensor,
    mlp_scale: Tensor,
    mlp_gate: Tensor,
}
impl AdaParams {
    fn load(w: &Weights, prefix: &str) -> CResult<Self> {
        let g = |n: &str| w.require(&format!("{prefix}.{n}")).cloned();
        Ok(Self {
            attn_shift: g("attn_shift")?,
            attn_scale: g("attn_scale")?,
            attn_gate: g("attn_gate")?,
            mlp_shift: g("mlp_shift")?,
            mlp_scale: g("mlp_scale")?,
            mlp_gate: g("mlp_gate")?,
        })
    }
}

/// emb (B, vid_dim, 2, 3); layer_idx 0=attn,1=mlp; comp 0=shift,1=scale,2=gate -> (B,1,vid_dim).
fn emb_param(emb: &Tensor, layer_idx: usize, comp: usize) -> Result<Tensor> {
    let m = emb.narrow(2, layer_idx, 1)?.squeeze(2)?; // (B,vid_dim,3)
    let c = m.narrow(2, comp, 1)?.squeeze(2)?; // (B,vid_dim)
    c.unsqueeze(1) // (B,1,vid_dim)
}

fn modulate_in(
    hidden: &Tensor,
    emb: &Tensor,
    layer_idx: usize,
    p_shift: &Tensor,
    p_scale: &Tensor,
) -> Result<Tensor> {
    let shift = emb_param(emb, layer_idx, 0)?.broadcast_add(p_shift)?;
    let scale = emb_param(emb, layer_idx, 1)?.broadcast_add(p_scale)?;
    hidden.broadcast_mul(&scale)?.broadcast_add(&shift)
}
fn modulate_out(
    hidden: &Tensor,
    emb: &Tensor,
    layer_idx: usize,
    p_gate: &Tensor,
) -> Result<Tensor> {
    let gate = emb_param(emb, layer_idx, 2)?.broadcast_add(p_gate)?;
    hidden.broadcast_mul(&gate)
}

// ---------------------------------------------------------------------------
// MLP
// ---------------------------------------------------------------------------

enum Mlp {
    SwiGlu {
        proj_in: Linear,
        gate: Linear,
        proj_out: Linear,
    },
    Gelu {
        proj_in: Linear,
        proj_out: Linear,
    },
}
impl Mlp {
    fn load(w: &Weights, prefix: &str, swiglu: bool) -> CResult<Self> {
        if swiglu {
            Ok(Mlp::SwiGlu {
                proj_in: Linear::load(w, &format!("{prefix}.proj_in"), false)?,
                gate: Linear::load(w, &format!("{prefix}.proj_in_gate"), false)?,
                proj_out: Linear::load(w, &format!("{prefix}.proj_out"), false)?,
            })
        } else {
            Ok(Mlp::Gelu {
                proj_in: Linear::load(w, &format!("{prefix}.proj_in"), true)?,
                proj_out: Linear::load(w, &format!("{prefix}.proj_out"), true)?,
            })
        }
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Mlp::SwiGlu {
                proj_in,
                gate,
                proj_out,
            } => {
                let g = nn::silu(&gate.forward(x)?)?;
                proj_out.forward(&g.mul(&proj_in.forward(x)?)?)
            }
            Mlp::Gelu { proj_in, proj_out } => {
                proj_out.forward(&nn::gelu_tanh(&proj_in.forward(x)?)?)
            }
        }
    }
    fn quantize(&mut self, quant: Quant) -> CResult<()> {
        match self {
            Mlp::SwiGlu {
                proj_in,
                gate,
                proj_out,
            } => {
                proj_in.quantize(quant)?;
                gate.quantize(quant)?;
                proj_out.quantize(quant)?;
            }
            Mlp::Gelu { proj_in, proj_out } => {
                proj_in.quantize(quant)?;
                proj_out.quantize(quant)?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// per-forward window/RoPE cache (shared across same-shift-parity blocks)
// ---------------------------------------------------------------------------

struct WindowCache {
    fwd: Tensor, // (L,) u32 windowed-order permutation
    rev: Tensor, // (L,) u32 inverse permutation
    window_shapes: Vec<(usize, usize, usize)>,
    vid_freqs: Tensor,         // (L, nf2*3)
    txt_freqs: Option<Tensor>, // (Lt, nf2*3) — lang mode only
}

#[allow(clippy::too_many_arguments)]
fn build_window_cache(
    freqs: &Tensor,
    vid_shape: (usize, usize, usize),
    window: (usize, usize, usize),
    shift: bool,
    pixel: bool,
    rope_on_text: bool,
    lt: usize,
    dev: &Device,
) -> Result<WindowCache> {
    let plan = window_plan(vid_shape.0, vid_shape.1, vid_shape.2, window, shift);
    let l = plan.forward_idx.len();
    let fwd = Tensor::from_vec(plan.forward_idx.clone(), l, dev)?;
    let rev = Tensor::from_vec(plan.reverse_idx.clone(), l, dev)?;
    let txt_off = if rope_on_text { lt as i64 } else { 0 };
    let mut blocks = Vec::with_capacity(plan.window_shapes.len());
    for (f, wh, ww) in &plan.window_shapes {
        blocks.push(vid_freq_block(freqs, *f, *wh, *ww, txt_off, pixel, dev)?);
    }
    let refs: Vec<&Tensor> = blocks.iter().collect();
    let vid_freqs = Tensor::cat(&refs, 0)?;
    let txt_freqs = if rope_on_text {
        Some(txt_freq_block(freqs, lt, dev)?)
    } else {
        None
    };
    Ok(WindowCache {
        fwd,
        rev,
        window_shapes: plan.window_shapes,
        vid_freqs,
        txt_freqs,
    })
}

// ---------------------------------------------------------------------------
// attention
// ---------------------------------------------------------------------------

struct MMAttention {
    qkv_vid: Linear,
    out_vid: Linear,
    nq_vid: Tensor,
    nk_vid: Tensor,
    qkv_txt: Linear,
    out_txt: Linear,
    nq_txt: Tensor,
    nk_txt: Tensor,
    freqs: Tensor,
    heads: usize,
    head_dim: usize,
    scale: f64,
    eps: f64,
    window: (usize, usize, usize),
    rope_on_text: bool,
    rope_pixel: bool,
}
impl MMAttention {
    fn load(w: &Weights, prefix: &str, cfg: &DitConfig) -> CResult<Self> {
        Ok(Self {
            qkv_vid: Linear::load(w, &format!("{prefix}.proj_qkv_vid"), false)?,
            out_vid: Linear::load(w, &format!("{prefix}.proj_out_vid"), true)?,
            nq_vid: w.require(&format!("{prefix}.norm_q_vid.weight"))?.clone(),
            nk_vid: w.require(&format!("{prefix}.norm_k_vid.weight"))?.clone(),
            qkv_txt: Linear::load(w, &format!("{prefix}.proj_qkv_txt"), false)?,
            out_txt: Linear::load(w, &format!("{prefix}.proj_out_txt"), true)?,
            nq_txt: w.require(&format!("{prefix}.norm_q_txt.weight"))?.clone(),
            nk_txt: w.require(&format!("{prefix}.norm_k_txt.weight"))?.clone(),
            freqs: w.require(&format!("{prefix}.rope.freqs"))?.clone(),
            heads: cfg.heads,
            head_dim: cfg.head_dim,
            scale: (cfg.head_dim as f64).powf(-0.5),
            eps: cfg.norm_eps,
            window: cfg.window,
            rope_on_text: cfg.rope_on_text,
            rope_pixel: cfg.rope_pixel,
        })
    }

    /// vid (1,L,vid_dim), txt (1,Lt,vid_dim) → (vid_out (1,L,vid_dim), txt_out (1,Lt,vid_dim)). B=1.
    fn forward(&self, vid: &Tensor, txt: &Tensor, cache: &WindowCache) -> Result<(Tensor, Tensor)> {
        let (h, hd) = (self.heads, self.head_dim);
        let l = vid.dim(1)?;
        let lt = txt.dim(1)?;
        let vid_dim = *vid.dims().last().unwrap();
        let txt_dim = *txt.dims().last().unwrap();

        let qkv_v = self
            .qkv_vid
            .forward(&vid.reshape((l, vid_dim))?)?
            .reshape((l, 3, h, hd))?;
        let qkv_t = self
            .qkv_txt
            .forward(&txt.reshape((lt, txt_dim))?)?
            .reshape((lt, 3, h, hd))?;

        let qkv_v = qkv_v.index_select(&cache.fwd, 0)?; // windowed order

        let pick = |q: &Tensor, i: usize| -> Result<Tensor> { q.narrow(1, i, 1)?.squeeze(1) };
        let q_v = nn::rms_norm(&pick(&qkv_v, 0)?, &self.nq_vid, self.eps)?; // (L,h,hd)
        let k_v = nn::rms_norm(&pick(&qkv_v, 1)?, &self.nk_vid, self.eps)?;
        let v_v = pick(&qkv_v, 2)?;
        let q_t = nn::rms_norm(&pick(&qkv_t, 0)?, &self.nq_txt, self.eps)?; // (Lt,h,hd)
        let k_t = nn::rms_norm(&pick(&qkv_t, 1)?, &self.nk_txt, self.eps)?;
        let v_t = pick(&qkv_t, 2)?;

        let q_v = apply_rope(&q_v, &cache.vid_freqs)?;
        let k_v = apply_rope(&k_v, &cache.vid_freqs)?;
        let (q_t, k_t) = match &cache.txt_freqs {
            Some(tf) => (apply_rope(&q_t, tf)?, apply_rope(&k_t, tf)?),
            None => (q_t, k_t),
        };

        let nwin = cache.window_shapes.len();
        let mut vid_out_parts: Vec<Tensor> = Vec::with_capacity(nwin);
        let mut txt_acc: Option<Tensor> = None;
        let mut start = 0usize;
        for (f, wh, ww) in &cache.window_shapes {
            let vlen = f * wh * ww;
            let qv = q_v.narrow(0, start, vlen)?;
            let kv = k_v.narrow(0, start, vlen)?;
            let vv = v_v.narrow(0, start, vlen)?;
            let q = Tensor::cat(&[&qv, &q_t], 0)?; // (vlen+Lt, h, hd)
            let k = Tensor::cat(&[&kv, &k_t], 0)?;
            let v = Tensor::cat(&[&vv, &v_t], 0)?;
            let s = vlen + lt;
            // (S,h,hd) -> (1,h,S,hd)
            let to_bhsd =
                |x: &Tensor| -> Result<Tensor> { x.reshape((1, s, h, hd))?.transpose(1, 2) };
            let o = nn::sdpa(&to_bhsd(&q)?, &to_bhsd(&k)?, &to_bhsd(&v)?, self.scale)?;
            let o = o.transpose(1, 2)?.contiguous()?.reshape((s, h, hd))?;
            let v_part = o.narrow(0, 0, vlen)?;
            let t_part = o.narrow(0, vlen, lt)?;
            vid_out_parts.push(v_part);
            txt_acc = Some(match txt_acc {
                Some(a) => (a + t_part)?,
                None => t_part,
            });
            start += vlen;
        }

        let vid_refs: Vec<&Tensor> = vid_out_parts.iter().collect();
        let vid_cat = Tensor::cat(&vid_refs, 0)?.reshape((l, h * hd))?;
        let vid_unwin = vid_cat.index_select(&cache.rev, 0)?;
        let vid_out = self.out_vid.forward(&vid_unwin)?.reshape((1, l, vid_dim))?;

        let txt_acc = txt_acc.expect("≥1 window");
        let txt_mean = (txt_acc * (1.0 / nwin as f64))?.reshape((lt, h * hd))?;
        let txt_out = self.out_txt.forward(&txt_mean)?.reshape((1, lt, txt_dim))?;

        let _ = self.window; // window used via cache; kept for parity with the reference config
        Ok((vid_out, txt_out))
    }

    fn quantize(&mut self, quant: Quant) -> CResult<()> {
        self.qkv_vid.quantize(quant)?;
        self.out_vid.quantize(quant)?;
        self.qkv_txt.quantize(quant)?;
        self.out_txt.quantize(quant) // norm_q/k are RMSNorm weights, not Linear
    }
}

// ---------------------------------------------------------------------------
// block
// ---------------------------------------------------------------------------

struct Block {
    attn: MMAttention,
    mlp_vid: Option<Mlp>,
    mlp_txt: Option<Mlp>,
    mlp_all: Option<Mlp>,
    ada_vid: Option<AdaParams>,
    ada_txt: Option<AdaParams>,
    ada_all: Option<AdaParams>,
    shared: bool,
    is_last: bool,
    eps: f64,
    /// Unit (`ones`) weight for the parameter-free pre-norms, built once at load and reused across
    /// steps rather than re-allocated on every `rms_plain` call in the hot loop (sc-9039). Both the
    /// vid and txt streams share `vid_dim`, so a single weight serves all pre-norms in the block.
    ones: Tensor,
}
impl Block {
    fn load(w: &Weights, idx: usize, cfg: &DitConfig) -> CResult<Self> {
        let prefix = format!("blocks.{idx}");
        let shared = idx >= cfg.mm_layers;
        let is_last = cfg.last_layer_vid_only && idx == cfg.num_layers - 1;
        let device = w
            .require(&format!("{prefix}.attn.norm_q_vid.weight"))?
            .device();
        // Unit weight for the parameter-free pre-norms (sc-9039); vid/txt both live in `vid_dim`.
        let ones = Tensor::ones(cfg.vid_dim, DType::F32, device)?;
        let attn = MMAttention::load(w, &format!("{prefix}.attn"), cfg)?;
        let (mlp_vid, mlp_txt, mlp_all) = if shared {
            (
                None,
                None,
                Some(Mlp::load(w, &format!("{prefix}.mlp.all"), cfg.swiglu_mlp)?),
            )
        } else {
            let txt = if is_last {
                None
            } else {
                Some(Mlp::load(w, &format!("{prefix}.mlp.txt"), cfg.swiglu_mlp)?)
            };
            (
                Some(Mlp::load(w, &format!("{prefix}.mlp.vid"), cfg.swiglu_mlp)?),
                txt,
                None,
            )
        };
        let (ada_vid, ada_txt, ada_all) = if shared {
            (
                None,
                None,
                Some(AdaParams::load(w, &format!("{prefix}.ada.params_all"))?),
            )
        } else {
            let txt = if is_last {
                None
            } else {
                Some(AdaParams::load(w, &format!("{prefix}.ada.params_txt"))?)
            };
            (
                Some(AdaParams::load(w, &format!("{prefix}.ada.params_vid"))?),
                txt,
                None,
            )
        };
        Ok(Self {
            attn,
            mlp_vid,
            mlp_txt,
            mlp_all,
            ada_vid,
            ada_txt,
            ada_all,
            shared,
            is_last,
            eps: cfg.norm_eps,
            ones,
        })
    }

    fn ada_v(&self) -> &AdaParams {
        if self.shared {
            self.ada_all.as_ref()
        } else {
            self.ada_vid.as_ref()
        }
        .expect("ada_v present")
    }
    fn ada_t(&self) -> &AdaParams {
        if self.shared {
            self.ada_all.as_ref()
        } else {
            self.ada_txt.as_ref()
        }
        .expect("ada_t present")
    }
    fn mlp_v(&self) -> &Mlp {
        if self.shared {
            self.mlp_all.as_ref()
        } else {
            self.mlp_vid.as_ref()
        }
        .expect("mlp_v present")
    }
    fn mlp_t(&self) -> &Mlp {
        if self.shared {
            self.mlp_all.as_ref()
        } else {
            self.mlp_txt.as_ref()
        }
        .expect("mlp_t present")
    }

    fn forward(
        &self,
        vid: &Tensor,
        txt: &Tensor,
        emb: &Tensor,
        cache: &WindowCache,
    ) -> Result<(Tensor, Tensor)> {
        let av = self.ada_v();
        let vid_attn_in = modulate_in(
            &rms_plain(vid, &self.ones, self.eps)?,
            emb,
            0,
            &av.attn_shift,
            &av.attn_scale,
        )?;
        let txt_attn_in = if self.is_last {
            rms_plain(txt, &self.ones, self.eps)?
        } else {
            let at = self.ada_t();
            modulate_in(
                &rms_plain(txt, &self.ones, self.eps)?,
                emb,
                0,
                &at.attn_shift,
                &at.attn_scale,
            )?
        };
        let (va, ta) = self.attn.forward(&vid_attn_in, &txt_attn_in, cache)?;
        let vid = (vid + modulate_out(&va, emb, 0, &av.attn_gate)?)?;
        let txt = if self.is_last {
            txt.clone()
        } else {
            (txt + modulate_out(&ta, emb, 0, &self.ada_t().attn_gate)?)?
        };

        let vid_mlp_in = modulate_in(
            &rms_plain(&vid, &self.ones, self.eps)?,
            emb,
            1,
            &av.mlp_shift,
            &av.mlp_scale,
        )?;
        let vid_mlp = modulate_out(&self.mlp_v().forward(&vid_mlp_in)?, emb, 1, &av.mlp_gate)?;
        let vid = (vid + vid_mlp)?;
        let txt = if self.is_last {
            txt
        } else {
            let at = self.ada_t();
            let txt_mlp_in = modulate_in(
                &rms_plain(&txt, &self.ones, self.eps)?,
                emb,
                1,
                &at.mlp_shift,
                &at.mlp_scale,
            )?;
            let tm = modulate_out(&self.mlp_t().forward(&txt_mlp_in)?, emb, 1, &at.mlp_gate)?;
            (txt + tm)?
        };
        Ok((vid, txt))
    }

    fn quantize(&mut self, quant: Quant) -> CResult<()> {
        self.attn.quantize(quant)?;
        if let Some(m) = &mut self.mlp_vid {
            m.quantize(quant)?;
        }
        if let Some(m) = &mut self.mlp_txt {
            m.quantize(quant)?;
        }
        if let Some(m) = &mut self.mlp_all {
            m.quantize(quant)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// transformer
// ---------------------------------------------------------------------------

pub struct Seedvr2Transformer {
    vid_in: PatchIn,
    txt_in: Linear,
    emb_in: TimeEmbedding,
    blocks: Vec<Block>,
    vid_out_norm: Option<Tensor>,
    out_shift: Option<Tensor>,
    out_scale: Option<Tensor>,
    vid_out: PatchOut,
    vid_dim: usize,
    eps: f64,
    use_output_ada: bool,
}

impl Seedvr2Transformer {
    pub fn from_weights(w: &Weights, cfg: &DitConfig) -> CResult<Self> {
        let blocks = (0..cfg.num_layers)
            .map(|i| Block::load(w, i, cfg))
            .collect::<CResult<Vec<_>>>()?;
        let (vid_out_norm, out_shift, out_scale) = if cfg.use_output_ada {
            (
                Some(w.require("vid_out_norm.weight")?.clone()),
                Some(w.require("out_shift")?.clone()),
                Some(w.require("out_scale")?.clone()),
            )
        } else {
            (None, None, None)
        };
        Ok(Self {
            vid_in: PatchIn::load(w, "vid_in", cfg)?,
            txt_in: Linear::load(w, "txt_in", true)?,
            emb_in: TimeEmbedding::load(w, "emb_in")?,
            blocks,
            vid_out_norm,
            out_shift,
            out_scale,
            vid_out: PatchOut::load(w, "vid_out", cfg)?,
            vid_dim: cfg.vid_dim,
            eps: cfg.norm_eps,
            use_output_ada: cfg.use_output_ada,
        })
    }

    /// vid `(1,33,T,H,W)`, txt `(1,Lt,5120)`, scalar `timestep` → `(1,16,T,H,W)`.
    pub fn forward(&self, vid: &Tensor, txt: &Tensor, timestep: f64) -> Result<Tensor> {
        let dev = vid.device().clone();
        let txt = self.txt_in.forward(txt)?;
        let (mut vid, vid_shape) = self.vid_in.forward(vid)?;
        let emb = self
            .emb_in
            .forward(timestep, &dev)?
            .reshape(((), self.vid_dim, 2, 3))?;
        let mut txt = txt;

        let lt = txt.dim(1)?;
        let a0 = &self.blocks[0].attn;
        let cache_even = build_window_cache(
            &a0.freqs,
            vid_shape,
            a0.window,
            false,
            a0.rope_pixel,
            a0.rope_on_text,
            lt,
            &dev,
        )?;
        let cache_odd = build_window_cache(
            &a0.freqs,
            vid_shape,
            a0.window,
            true,
            a0.rope_pixel,
            a0.rope_on_text,
            lt,
            &dev,
        )?;
        for (i, block) in self.blocks.iter().enumerate() {
            let cache = if i % 2 == 1 { &cache_odd } else { &cache_even };
            let (v, t) = block.forward(&vid, &txt, &emb, cache)?;
            vid = v;
            txt = t;
        }
        if self.use_output_ada {
            vid = nn::rms_norm(&vid, self.vid_out_norm.as_ref().unwrap(), self.eps)?;
            let scale = emb_param(&emb, 0, 1)?.broadcast_add(self.out_scale.as_ref().unwrap())?;
            let shift = emb_param(&emb, 0, 0)?.broadcast_add(self.out_shift.as_ref().unwrap())?;
            vid = vid.broadcast_mul(&scale)?.broadcast_add(&shift)?;
        }
        self.vid_out.forward(&vid, vid_shape)
    }

    /// Quantize every DiT Linear to `quant` (`Q4_0`/`Q8_0`), Linear-only — sc-5927. The VAE stays
    /// dense (it has no Linears here); `vid_in.proj` (in=132) is auto-skipped by the block-
    /// divisibility predicate, matching the reference. Idempotent / safe to call once after load.
    pub fn quantize(&mut self, quant: Quant) -> CResult<()> {
        self.vid_in.quantize(quant)?;
        self.txt_in.quantize(quant)?;
        self.emb_in.quantize(quant)?;
        for block in &mut self.blocks {
            block.quantize(quant)?;
        }
        self.vid_out.quantize(quant)
    }
}

#[cfg(test)]
mod quant_tests {
    use super::*;

    /// Cosine similarity of two flattened tensors.
    fn cosine(a: &Tensor, b: &Tensor) -> f32 {
        let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (p, r) in a.iter().zip(b.iter()) {
            dot += (*p as f64) * (*r as f64);
            na += (*p as f64) * (*p as f64);
            nb += (*r as f64) * (*r as f64);
        }
        (dot / (na.sqrt() * nb.sqrt() + 1e-12)) as f32
    }

    /// A `[out,in]` `Linear` with `in_dim` a multiple of the 32-wide block quantizes and forwards
    /// near-losslessly at Q8 / coherently at Q4 vs the dense f32 result — the per-Linear analog of the
    /// full-DiT quant smoke, runnable on CPU with no weights (mirrors candle-gen-lens's gate).
    fn quant_roundtrip(quant: Quant, min_cos: f32) {
        let dev = Device::Cpu;
        let (in_dim, out_dim) = (64usize, 96usize); // in=64 = two Q4_0/Q8_0 blocks per row
        let w = Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev).unwrap();
        let b = Tensor::randn(0f32, 1f32, out_dim, &dev).unwrap();
        let mut lin = Linear(QLinear::from_dense(DenseLinear::Transposed {
            weight_t: nn::transpose_weight(&w).unwrap(),
            bias: Some(b),
        }));
        // a 3-D activation (B,S,in) exercises the leading-dim flatten in the quant forward.
        let x = Tensor::randn(0f32, 1f32, (2usize, 5usize, in_dim), &dev).unwrap();
        let dense = lin.forward(&x).unwrap();

        lin.quantize(quant).unwrap();
        assert!(lin.0.is_quantized(), "must be quantized");
        assert_eq!(
            lin.0.matmul_strategy(),
            Some(candle_gen::quant::MatmulStrategy::Int8Fast),
            "SeedVR2 uses candle's int8 QMatMul forward"
        );
        let q = lin.forward(&x).unwrap();
        assert_eq!(
            q.dims(),
            dense.dims(),
            "shape preserved through quant forward"
        );

        let cos = cosine(&dense, &q);
        assert!(cos > min_cos, "{quant:?} cosine {cos:.5} ≤ {min_cos}");
    }

    #[test]
    fn q8_is_near_lossless() {
        quant_roundtrip(Quant::Q8, 0.999);
    }

    #[test]
    fn q4_stays_coherent() {
        quant_roundtrip(Quant::Q4, 0.95);
    }

    /// A Linear whose `in_features` is not a multiple of the block (the `vid_in.proj` in=132 case)
    /// stays dense — the reference predicate. (132 % 32 = 4.)
    #[test]
    fn quantize_skips_indivisible_in_features() {
        let dev = Device::Cpu;
        let w = Tensor::randn(0f32, 1f32, (64usize, 132usize), &dev).unwrap();
        let mut lin = Linear(QLinear::from_dense(DenseLinear::Transposed {
            weight_t: nn::transpose_weight(&w).unwrap(),
            bias: None,
        }));
        lin.quantize(Quant::Q8).unwrap();
        assert!(
            !lin.0.is_quantized(),
            "in=132 must stay dense (132 % 32 ≠ 0)"
        );
    }

    /// `quantize` is idempotent — a second call on an already-quantized Linear is a no-op, not a panic
    /// (the transformer's quantize pass runs uniformly over every Linear).
    #[test]
    fn quantize_is_idempotent() {
        let dev = Device::Cpu;
        let w = Tensor::randn(0f32, 1f32, (64usize, 32usize), &dev).unwrap();
        let mut lin = Linear(QLinear::from_dense(DenseLinear::Transposed {
            weight_t: nn::transpose_weight(&w).unwrap(),
            bias: None,
        }));
        lin.quantize(Quant::Q8).unwrap();
        lin.quantize(Quant::Q8).unwrap(); // no-op, must not error
        assert!(lin.0.is_quantized());
    }

    /// sc-9039: the block pre-norms now reuse a single `ones` weight cached at load rather than
    /// allocating a fresh `Tensor::ones` on every `rms_plain` call in the hot loop. This must be
    /// bit-identical — `nn::rms_norm` is deterministic given (weight, eps), and a cached unit weight
    /// equals a freshly-allocated one exactly.
    #[test]
    fn rms_plain_cached_ones_is_bit_identical() {
        let dev = Device::Cpu;
        let dim = 2560usize; // 3B vid_dim
        let eps = 1e-6;
        let x = Tensor::randn(0f32, 1f32, (1usize, 4usize, dim), &dev).unwrap();

        // Cached unit weight (built once, as `Block::ones` is at load).
        let ones = Tensor::ones(dim, DType::F32, &dev).unwrap();
        let cached = rms_plain(&x, &ones, eps).unwrap();

        // A freshly-allocated unit weight — the old per-call behaviour.
        let fresh_ones = Tensor::ones(dim, DType::F32, &dev).unwrap();
        let fresh = rms_plain(&x, &fresh_ones, eps).unwrap();

        let diff = (cached - fresh)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(
            diff, 0.0,
            "cached-ones rms_plain must match a fresh one exactly"
        );
    }
}
