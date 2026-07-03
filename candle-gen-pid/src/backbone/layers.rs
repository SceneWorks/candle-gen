//! PixDiT leaf modules + host helpers: RMSNorm (via [`crate::nn::rms`]), SwiGLU `FeedForward`, GELU
//! `Mlp`, `TimestepConditioner`, the patch/pixel token embedders, `FinalLayer`, the 2-D sin/cos pixel
//! position table, and the unfold/fold patchify pair. Faithful port of the `modules.py` /
//! `pixeldit_c2i.py` blocks merged into `pixeldit_official.py`. Runs f32 throughout.

use candle_gen::candle_core::{Device, Tensor};
use candle_gen::candle_nn::{Linear, Module};
use candle_gen::{Result, Weights};

use crate::nn::{linear, rms};

/// All PixDiT RMSNorms use eps = 1e-6.
pub const RMS_EPS: f32 = 1e-6;

/// SwiGLU `FeedForward`: `w2(silu(w1(x)) · w3(x))`. Inner width is baked into the loaded weight
/// shapes; all three projections are bias-less.
pub struct FeedForward {
    w1: Linear,
    w2: Linear,
    w3: Linear,
}

impl FeedForward {
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            w1: linear(w, &format!("{prefix}.w1"))?,
            w2: linear(w, &format!("{prefix}.w2"))?,
            w3: linear(w, &format!("{prefix}.w3"))?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.w1.forward(x)?.silu()?;
        Ok(self.w2.forward(&(gate * self.w3.forward(x)?)?)?)
    }
}

/// `Mlp`: `fc2(gelu(fc1(x)))` with exact (erf) GELU and biased projections (pixel-stream FFN).
pub struct Mlp {
    fc1: Linear,
    fc2: Linear,
}

impl Mlp {
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            fc1: linear(w, &format!("{prefix}.fc1"))?,
            fc2: linear(w, &format!("{prefix}.fc2"))?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.fc2.forward(&self.fc1.forward(x)?.gelu_erf()?)?)
    }
}

/// `concat([cos(t·freqs), sin(t·freqs)])`, `freqs[i] = exp(-ln(max_period)·i/half)`, `half = dim/2`.
/// `t`: `[N]` → `[N, dim]` (dim even). cos-then-sin, matching the reference.
pub fn timestep_embedding(t: &Tensor, dim: i32, max_period: f32) -> Result<Tensor> {
    let half = (dim / 2) as usize;
    let lp = (max_period as f64).ln();
    let freqs: Vec<f32> = (0..half)
        .map(|i| (-lp * i as f64 / half as f64).exp() as f32)
        .collect();
    let freqs = Tensor::from_vec(freqs, (1, half), t.device())?;
    let n = t.dims1()?;
    let t = t.reshape((n, 1))?;
    let args = t.broadcast_mul(&freqs)?; // [n, half]
    Ok(Tensor::cat(&[&args.cos()?, &args.sin()?], 1)?)
}

/// `TimestepConditioner`: sinusoidal embedding (size 256, **max_period = 10**) → Linear → SiLU →
/// Linear. The `max_period=10` is a PixDiT-specific value (not the usual 10000).
pub struct TimestepConditioner {
    mlp0: Linear,
    mlp2: Linear,
    freq_size: i32,
}

impl TimestepConditioner {
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            mlp0: linear(w, &format!("{prefix}.mlp.0"))?,
            mlp2: linear(w, &format!("{prefix}.mlp.2"))?,
            freq_size: 256,
        })
    }

    /// `t`: `[N]` → `[N, hidden]`.
    pub fn forward(&self, t: &Tensor) -> Result<Tensor> {
        let emb = timestep_embedding(t, self.freq_size, 10.0)?;
        let h = self.mlp0.forward(&emb)?.silu()?;
        Ok(self.mlp2.forward(&h)?)
    }
}

/// `PatchTokenEmbedder`: a Linear `proj` with an optional trailing RMSNorm. `s_embedder` has no
/// norm; `y_embedder` carries `norm.weight`.
pub struct PatchTokenEmbedder {
    proj: Linear,
    norm: Option<Tensor>,
}

impl PatchTokenEmbedder {
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let norm_key = format!("{prefix}.norm.weight");
        let norm = if w.contains(&norm_key) {
            Some(w.require(&norm_key)?)
        } else {
            None
        };
        Ok(Self {
            proj: linear(w, &format!("{prefix}.proj"))?,
            norm,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.proj.forward(x)?;
        match &self.norm {
            Some(nw) => rms(&x, nw, RMS_EPS),
            None => Ok(x),
        }
    }
}

/// `FinalLayer`: RMSNorm then a biased Linear to the output channels.
pub struct FinalLayer {
    norm: Tensor,
    linear: Linear,
}

impl FinalLayer {
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            norm: w.require(&format!("{prefix}.norm.weight"))?,
            linear: linear(w, &format!("{prefix}.linear"))?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.linear.forward(&rms(x, &self.norm, RMS_EPS)?)?)
    }
}

/// Host 2-D sin/cos pixel position table `[H·W, embed_dim]` (row-major over `(H, W)`, `W` fastest),
/// matching `get_2d_sincos_pos_embed_from_grid`. Computed in f64, cast f32. First half encodes the
/// **w** coordinate, second half the **h** coordinate (the reference's swapped `emb_h`/`emb_w`).
pub fn sincos_2d_pos(embed_dim: i32, h: i32, w: i32, device: &Device) -> Result<Tensor> {
    let d = (embed_dim / 2) as usize; // per-axis dim
    let half = d / 2; // omega length
    let omega: Vec<f64> = (0..half)
        .map(|k| 1.0 / 10000f64.powf(k as f64 / (d as f64 / 2.0)))
        .collect();
    let oned = |pos: f64, out: &mut [f32]| {
        for k in 0..half {
            let a = pos * omega[k];
            out[k] = a.sin() as f32;
            out[half + k] = a.cos() as f32;
        }
    };
    let n = (h * w) as usize;
    let mut buf = vec![0f32; n * embed_dim as usize];
    for i in 0..h {
        for j in 0..w {
            let p = (i * w + j) as usize;
            let base = p * embed_dim as usize;
            oned(j as f64, &mut buf[base..base + d]); // first half: w coordinate (j)
            oned(i as f64, &mut buf[base + d..base + 2 * d]); // second half: h coordinate (i)
        }
    }
    Ok(Tensor::from_vec(buf, (n, embed_dim as usize), device)?)
}

/// `F.unfold(x, kernel=p, stride=p).transpose(1,2)`: `[B, C, H, W]` → `[B, L, C·p²]`, patch order
/// row-major over `(Hs, Ws)` with `Ws` fastest, inner flatten `(C, pH, pW)` C-outermost.
pub fn unfold_patches(x: &Tensor, patch: i32) -> Result<Tensor> {
    let (b, c, h, w) = x.dims4()?;
    let p = patch as usize;
    let (hs, ws) = (h / p, w / p);
    Ok(x.reshape((b, c, hs, p, ws, p))?
        .permute((0, 2, 4, 1, 3, 5))?
        .contiguous()?
        .reshape((b, hs * ws, c * p * p))?)
}

/// Inverse of [`unfold_patches`] for the final fold: `[B, C, P², L]` → `[B, C, H, W]` (non-overlapping
/// stride = kernel, a pure scatter). `P²` axis is `(pH, pW)`, `L` is `(Hs, Ws)`.
pub fn fold_patches(x: &Tensor, c: i32, hs: i32, ws: i32, patch: i32) -> Result<Tensor> {
    let b = x.dim(0)?;
    let (c, hs, ws, p) = (c as usize, hs as usize, ws as usize, patch as usize);
    Ok(x.reshape((b, c, p, p, hs, ws))?
        .permute((0, 1, 4, 2, 5, 3))?
        .contiguous()?
        .reshape((b, c, hs * p, ws * p))?)
}

/// `PixelTokenEmbedder` image-mode forward: `[B, C, H, W]` → per-pixel Linear → add the 2-D sin/cos
/// position table → patchify to `[B·L, P², D]`. Port of `PixelTokenEmbedder.forward` (dim==4 branch).
pub struct PixelTokenEmbedder {
    proj: Linear,
    dim: i32,
}

impl PixelTokenEmbedder {
    pub fn from_weights(w: &Weights, prefix: &str, dim: i32) -> Result<Self> {
        Ok(Self {
            proj: linear(w, &format!("{prefix}.proj"))?,
            dim,
        })
    }

    pub fn forward(&self, x: &Tensor, h: i32, w: i32, patch: i32) -> Result<Tensor> {
        let b = x.dim(0)?;
        let (hh, ww, p) = (h as usize, w as usize, patch as usize);
        let (hs, ws) = (hh / p, ww / p);
        // [B,C,H,W] -> [B,H,W,C] -> proj -> [B,H,W,D]
        let xt = x.permute((0, 2, 3, 1))?.contiguous()?;
        let xp = self.proj.forward(&xt)?;
        let pos =
            sincos_2d_pos(self.dim, h, w, x.device())?.reshape((1, hh, ww, self.dim as usize))?;
        let xp = xp.broadcast_add(&pos)?;
        // [B,Hs,p,Ws,p,D] -> [B,Hs,Ws,p,p,D] -> [B*L, P2, D]
        Ok(xp
            .reshape((b, hs, p, ws, p, self.dim as usize))?
            .permute((0, 1, 3, 2, 4, 5))?
            .contiguous()?
            .reshape((b * hs * ws, p * p, self.dim as usize))?)
    }
}
