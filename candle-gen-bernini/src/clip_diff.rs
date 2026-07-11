//! The Bernini planner's flow-matching ViT diffusion head (`DiffLoss_FM` / `SimpleMLPAdaLN`,
//! `bernini/models/diffloss_fm.py`) + its `FlowMatchScheduler` (`bernini/models/scheduler.py`) —
//! candle sibling of `mlx-gen-bernini/src/clip_diff.rs` (sc-5139).
//!
//! The planner's MAR loop (see [`crate::mar::sample_vit_embed`]) samples a target ViT embedding by
//! running this small AdaLN MLP as a flow-matching denoiser conditioned on `c` = the connector's
//! `for_gen`/`for_vit` projection of the planner hidden states. Inference-only: the train-time
//! `diffusion_batch_mul` is dropped, and the `eps`/`rest` output split is vestigial (out channels ==
//! in channels == 3584, so `rest` is empty).
//!
//! `SimpleMLPAdaLN`: `input_proj`(in→width) + `time_embed`(GLIDE sinusoidal→MLP) + `cond_embed`(z→width)
//! → `y = t+c` → N adaLN-zero `ResBlock`s → `FinalLayer`. LayerNorm reductions run in f32 (the parity
//! goldens are f32).
//!
//! Validated bit-near against `tests/fixtures/clip_diff_golden.safetensors` (the same synthetic golden
//! the MLX lane asserts): `for_gen`/`for_vit` (via [`crate::connector`]), the net forward, and a full
//! triple-CFG `sample()` denoise — all ~5e-3 f32 matmul floor.

use candle_gen::candle_core::{DType, Tensor, D};
use candle_gen::candle_nn::{Linear, Module, VarBuilder};
use candle_gen::{CandleError, Result as CResult};

use crate::nn::{layer_norm, lin_bias};

const LN_EPS: f64 = 1e-6;

// ---------------------------------------------------------------------------
// FlowMatchScheduler (inference) — analog of candle-gen-wan's flow scheduler.
// ---------------------------------------------------------------------------

/// The clip-diff flow-matching scheduler: a shifted linear σ schedule + an Euler velocity step.
/// Host-side `f32` σ/timestep vectors (tiny — `num_inference_steps` entries).
pub struct FlowMatchScheduler {
    sigmas: Vec<f32>,
    timesteps: Vec<f32>,
    num_train: f32,
    shift: f32,
    sigma_min: f32,
    sigma_max: f32,
    extra_one_step: bool,
}

impl FlowMatchScheduler {
    /// Bernini clip-diff defaults: `sigma_min 0.003/1.002`, `sigma_max 1.0`, `num_train 1000`.
    pub fn new(shift: f32, extra_one_step: bool) -> Self {
        let mut s = Self {
            sigmas: Vec::new(),
            timesteps: Vec::new(),
            num_train: 1000.0,
            shift,
            sigma_min: 0.003 / 1.002,
            sigma_max: 1.0,
            extra_one_step,
        };
        s.set_timesteps(100);
        s
    }

    /// Build the σ schedule for `steps` inference steps (`denoising_strength = 1.0`): a linspace
    /// `[sigma_max … sigma_min]` (one extra step then dropped when `extra_one_step`), shifted by
    /// `shift·σ / (1 + (shift-1)·σ)`; `timesteps = σ · num_train`.
    pub fn set_timesteps(&mut self, steps: usize) {
        let sigma_start = self.sigma_max; // denoising_strength 1.0 → sigma_min + (max-min)·1 = max
        let n = if self.extra_one_step {
            steps + 1
        } else {
            steps
        };
        let mut sigmas: Vec<f32> = (0..n)
            .map(|i| {
                let frac = if n <= 1 {
                    0.0
                } else {
                    i as f32 / (n as f32 - 1.0)
                };
                sigma_start + (self.sigma_min - sigma_start) * frac
            })
            .collect();
        if self.extra_one_step {
            sigmas.pop(); // [:-1]
        }
        let shift = self.shift;
        for s in &mut sigmas {
            *s = shift * *s / (1.0 + (shift - 1.0) * *s);
        }
        self.timesteps = sigmas.iter().map(|&s| s * self.num_train).collect();
        self.sigmas = sigmas;
    }

    pub fn timesteps(&self) -> &[f32] {
        &self.timesteps
    }

    /// Euler velocity step at schedule index `i`: `sample + v · (σ_{i+1} − σ_i)` (σ_{last+1} = 0).
    pub fn step(&self, model_output: &Tensor, i: usize, sample: &Tensor) -> CResult<Tensor> {
        let sigma = self.sigmas[i];
        let sigma_next = if i + 1 >= self.sigmas.len() {
            0.0
        } else {
            self.sigmas[i + 1]
        };
        Ok((sample + (model_output * (sigma_next - sigma) as f64)?)?)
    }
}

// ---------------------------------------------------------------------------
// SimpleMLPAdaLN building blocks.
// ---------------------------------------------------------------------------

/// `x*(1+scale) + shift` (DiT adaLN modulation).
fn modulate(x: &Tensor, shift: &Tensor, scale: &Tensor) -> CResult<Tensor> {
    Ok(((x * (scale + 1.0)?)? + shift)?)
}

/// GLIDE sinusoidal timestep embedding `[N, dim]` (f32): `half = dim/2`,
/// `freqs[k] = exp(-ln(max_period)·k/half)`, `emb = cat(cos(t·freqs), sin(t·freqs))`. `dim` is even
/// (256), so the odd-pad branch of the reference is unreachable and omitted.
fn timestep_embedding(t: &Tensor, dim: usize, max_period: f32) -> CResult<Tensor> {
    let half = dim / 2;
    let ln = max_period.ln();
    let freqs: Vec<f32> = (0..half)
        .map(|k| (-ln * k as f32 / half as f32).exp())
        .collect();
    let dev = t.device();
    let freqs = Tensor::from_vec(freqs, (1, half), dev)?; // [1, half]
    let n = t.dim(0)?;
    let t_col = t.to_dtype(DType::F32)?.reshape((n, 1))?;
    let args = t_col.broadcast_mul(&freqs)?; // [N, half]
    Ok(Tensor::cat(&[&args.cos()?, &args.sin()?], 1)?)
}

struct TimestepEmbedder {
    mlp0: Linear,
    mlp2: Linear,
    freq_size: usize,
}

impl TimestepEmbedder {
    fn new(vb: &VarBuilder) -> CResult<Self> {
        Ok(Self {
            mlp0: lin_bias(vb, "mlp.0")?,
            mlp2: lin_bias(vb, "mlp.2")?,
            freq_size: 256,
        })
    }

    /// `mlp(timestep_embedding(t))`, with the sinusoidal embedding cast to `dtype` before the MLP
    /// (the reference `t_freq.to(t.dtype)`).
    fn forward(&self, t: &Tensor, dtype: DType) -> CResult<Tensor> {
        let freq = timestep_embedding(t, self.freq_size, 10000.0)?.to_dtype(dtype)?;
        Ok(self.mlp2.forward(&self.mlp0.forward(&freq)?.silu()?)?)
    }
}

struct ResBlock {
    in_ln_w: Tensor,
    in_ln_b: Tensor,
    mlp0: Linear,
    mlp2: Linear,
    adaln: Linear, // adaLN_modulation.1 (Linear width→3·width); SiLU applied in forward
}

impl ResBlock {
    fn new(vb: &VarBuilder) -> CResult<Self> {
        Ok(Self {
            in_ln_w: vb.get_unchecked("in_ln.weight")?,
            in_ln_b: vb.get_unchecked("in_ln.bias")?,
            mlp0: lin_bias(vb, "mlp.0")?,
            mlp2: lin_bias(vb, "mlp.2")?,
            adaln: lin_bias(vb, "adaLN_modulation.1")?,
        })
    }

    /// `shift,scale,gate = adaLN(silu(y))`; `h = mlp(modulate(LN(x), shift, scale))`; `x + gate·h`.
    fn forward(&self, x: &Tensor, y: &Tensor) -> CResult<Tensor> {
        let mods = self.adaln.forward(&y.silu()?)?;
        let w = mods.dim(D::Minus1)? / 3;
        let shift = mods.narrow(D::Minus1, 0, w)?;
        let scale = mods.narrow(D::Minus1, w, w)?;
        let gate = mods.narrow(D::Minus1, 2 * w, w)?;
        let h = layer_norm(x, Some(&self.in_ln_w), Some(&self.in_ln_b), LN_EPS)?;
        let h = modulate(&h, &shift, &scale)?;
        let h = self.mlp2.forward(&self.mlp0.forward(&h)?.silu()?)?;
        Ok((x + gate.mul(&h)?)?)
    }
}

struct FinalLayer {
    linear: Linear,
    adaln: Linear, // adaLN_modulation.1 (Linear width→2·width)
}

impl FinalLayer {
    fn new(vb: &VarBuilder) -> CResult<Self> {
        Ok(Self {
            linear: lin_bias(vb, "linear")?,
            adaln: lin_bias(vb, "adaLN_modulation.1")?,
        })
    }

    /// `shift,scale = adaLN(silu(c))`; `linear(modulate(LN_noaffine(x), shift, scale))`.
    fn forward(&self, x: &Tensor, c: &Tensor) -> CResult<Tensor> {
        let mods = self.adaln.forward(&c.silu()?)?;
        let w = mods.dim(D::Minus1)? / 2;
        let shift = mods.narrow(D::Minus1, 0, w)?;
        let scale = mods.narrow(D::Minus1, w, w)?;
        let h = layer_norm(x, None, None, LN_EPS)?; // norm_final: elementwise_affine=False
        let h = modulate(&h, &shift, &scale)?;
        Ok(self.linear.forward(&h)?)
    }
}

/// `SimpleMLPAdaLN`: the clip-diff denoiser network.
struct SimpleMlpAdaLn {
    time_embed: TimestepEmbedder,
    cond_embed: Linear,
    input_proj: Linear,
    res_blocks: Vec<ResBlock>,
    final_layer: FinalLayer,
}

impl SimpleMlpAdaLn {
    fn new(vb: &VarBuilder, depth: usize) -> CResult<Self> {
        let rvb = vb.pp("res_blocks");
        let res_blocks = (0..depth)
            .map(|i| ResBlock::new(&rvb.pp(i)))
            .collect::<CResult<Vec<_>>>()?;
        Ok(Self {
            time_embed: TimestepEmbedder::new(&vb.pp("time_embed"))?,
            cond_embed: lin_bias(vb, "cond_embed")?,
            input_proj: lin_bias(vb, "input_proj")?,
            res_blocks,
            final_layer: FinalLayer::new(&vb.pp("final_layer"))?,
        })
    }

    /// `x` `[N, in]`, `t` `[N]` (or `[1]`), `c` `[N, z]` → `[N, out]`.
    fn forward(&self, x: &Tensor, t: &Tensor, c: &Tensor) -> CResult<Tensor> {
        let mut h = self.input_proj.forward(x)?;
        let te = self.time_embed.forward(t, h.dtype())?;
        let ce = self.cond_embed.forward(c)?;
        let y = te.broadcast_add(&ce)?;
        for block in &self.res_blocks {
            h = block.forward(&h, &y)?;
        }
        self.final_layer.forward(&h, &y)
    }
}

/// The flow-matching ViT diffusion head: `SimpleMLPAdaLN` + its `FlowMatchScheduler`.
pub struct DiffLossFm {
    net: SimpleMlpAdaLn,
    scheduler: FlowMatchScheduler,
    in_channels: usize,
}

impl DiffLossFm {
    /// Build from a `VarBuilder` rooted at the net namespace (`net.*` for the sc-5144 layout). `depth`
    /// = number of res blocks (16), `in_channels` = 3584, `shift` = 2.0.
    pub fn new(vb: VarBuilder, depth: usize, in_channels: usize, shift: f32) -> CResult<Self> {
        Ok(Self {
            net: SimpleMlpAdaLn::new(&vb, depth)?,
            scheduler: FlowMatchScheduler::new(shift, true),
            in_channels,
        })
    }

    /// Raw denoiser forward (no CFG).
    pub fn forward(&self, x: &Tensor, t: &Tensor, c: &Tensor) -> CResult<Tensor> {
        self.net.forward(x, t, c)
    }

    /// Split `x` `[k·N, C]` into `k` equal row-chunks along dim 0.
    fn split_rows(x: &Tensor, k: usize) -> CResult<Vec<Tensor>> {
        let n = x.dim(0)? / k;
        (0..k)
            .map(|i| -> CResult<Tensor> { Ok(x.narrow(0, i * n, n)?) })
            .collect()
    }

    /// Standard CFG over a 2-tiled batch: `uncond + cfg·(cond − uncond)`.
    fn forward_with_cfg(&self, x: &Tensor, t: &Tensor, c: &Tensor, cfg: f32) -> CResult<Tensor> {
        let half = &Self::split_rows(x, 2)?[0];
        let combined = Tensor::cat(&[half, half], 0)?;
        let out = self.net.forward(&combined, t, c)?;
        let p = Self::split_rows(&out, 2)?; // cond, uncond
        let half_eps = (&p[1] + ((&p[0] - &p[1])? * cfg as f64)?)?;
        Ok(Tensor::cat(&[&half_eps, &half_eps], 0)?)
    }

    /// Triple CFG over a 3-tiled batch (txt/img guidance):
    /// `uncond + img·(imgcond − uncond) + txt·(cond − imgcond)`.
    fn forward_with_txt_img_cfg(
        &self,
        x: &Tensor,
        t: &Tensor,
        c: &Tensor,
        txt_cfg: f32,
        img_cfg: f32,
    ) -> CResult<Tensor> {
        let part = &Self::split_rows(x, 3)?[0];
        let combined = Tensor::cat(&[part, part, part], 0)?;
        let out = self.net.forward(&combined, t, c)?;
        let p = Self::split_rows(&out, 3)?; // cond, uncond, imgcond
        let img_term = ((&p[2] - &p[1])? * img_cfg as f64)?;
        let txt_term = ((&p[0] - &p[2])? * txt_cfg as f64)?;
        let part_eps = ((&p[1] + img_term)? + txt_term)?;
        Ok(Tensor::cat(&[&part_eps, &part_eps, &part_eps], 0)?)
    }

    /// Denoise a target ViT embedding from `noise_base` `[N, in]`, conditioned on `z`. Mirrors
    /// `DiffLoss_FM.sample`: `img_cfg.is_some() && cfg>1` → triple CFG (z tiled ×3, noise ×3);
    /// `cfg>1` → standard CFG (×2); else plain. `z` must already be tiled to match the chosen mode
    /// (×3 / ×2 / ×1), as the reference's caller does. Returns the tiled samples `[mode·N, in]`.
    pub fn sample(
        &mut self,
        z: &Tensor,
        cfg: f32,
        num_steps: usize,
        img_cfg: Option<f32>,
        noise_base: &Tensor,
    ) -> CResult<Tensor> {
        self.scheduler.set_timesteps(num_steps);
        let dtype = z.dtype();
        // Guard the pre-tiling contract (F-081): a mismatched z would fail with an opaque matmul
        // shape error rather than this clear one.
        let tiles = if img_cfg.is_some() && cfg > 1.0 {
            3
        } else if cfg > 1.0 {
            2
        } else {
            1
        };
        if z.dim(0)? != tiles * noise_base.dim(0)? {
            return Err(CandleError::Msg(format!(
                "bernini clip_diff sample: conditioning z rows {} must be {tiles}× the noise base rows {}",
                z.dim(0)?,
                noise_base.dim(0)?
            )));
        }
        let refs: Vec<&Tensor> = (0..tiles).map(|_| noise_base).collect();
        let mut samples = Tensor::cat(&refs, 0)?.to_dtype(dtype)?;

        let timesteps: Vec<f32> = self.scheduler.timesteps().to_vec();
        for (i, &ts) in timesteps.iter().enumerate() {
            let t = Tensor::from_vec(vec![ts], (1,), z.device())?.to_dtype(dtype)?;
            let pred = match (img_cfg, cfg > 1.0) {
                (Some(img), true) => self.forward_with_txt_img_cfg(&samples, &t, z, cfg, img)?,
                (None, true) => self.forward_with_cfg(&samples, &t, z, cfg)?,
                _ => self.net.forward(&samples, &t, z)?,
            };
            samples = self.scheduler.step(&pred, i, &samples)?;
        }
        Ok(samples)
    }

    pub fn in_channels(&self) -> usize {
        self.in_channels
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Device;

    /// FlowMatchScheduler σ schedule: shift 2.0 / extra_one_step → first σ is the shifted σ_max,
    /// monotonically decreasing, `timesteps = σ·1000`, and `step` is the Euler velocity update.
    #[test]
    fn scheduler_schedule_and_step() {
        let sched = FlowMatchScheduler::new(2.0, true);
        // shift·1/(1+(shift-1)·1) = 2/2 = 1.0 for σ_max=1.
        assert!(
            (sched.sigmas[0] - 1.0).abs() < 1e-6,
            "first σ = shifted σ_max"
        );
        for w in sched.sigmas.windows(2) {
            assert!(w[1] < w[0], "σ strictly decreasing");
        }
        assert_eq!(sched.timesteps.len(), 100);
        assert!((sched.timesteps[0] - sched.sigmas[0] * 1000.0).abs() < 1e-3);

        let sample = Tensor::from_vec(vec![1.0f32, 2.0], (1, 2), &Device::Cpu).unwrap();
        let v = Tensor::from_vec(vec![0.5f32, -0.5], (1, 2), &Device::Cpu).unwrap();
        let out = sched.step(&v, 0, &sample).unwrap();
        let d = (sched.sigmas[1] - sched.sigmas[0]) as f64;
        let got: Vec<f32> = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((got[0] as f64 - (1.0 + 0.5 * d)).abs() < 1e-5);
        assert!((got[1] as f64 - (2.0 - 0.5 * d)).abs() < 1e-5);
    }

    /// `set_timesteps(3)` yields 3 steps (extra_one_step drops the tail of a 4-point linspace).
    #[test]
    fn scheduler_step_count() {
        let mut sched = FlowMatchScheduler::new(2.0, true);
        sched.set_timesteps(3);
        assert_eq!(sched.timesteps().len(), 3);
    }
}
