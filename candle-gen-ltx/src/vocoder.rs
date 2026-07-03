//! LTX-2.3 **vocoder** (sc-5495) — candle (NCL) port of `mlx-gen-ltx` `vocoder.rs`: mel/STFT-domain
//! spectrogram → waveform. The shipped 2.3 path is `VocoderWithBwe` — a BigVGAN core (16 kHz) →
//! stored-STFT mel → a BigVGAN bandwidth-extension generator (16 → 48 kHz) → linear-interp skip of
//! the core output, summed and clipped.
//!
//! Loaded from the dense `ltx-2.3-22b-distilled.safetensors` under the `vocoder.` prefix (same
//! checkpoint as the rest — no separate download). Conv weights are checkpoint-native PyTorch layout
//! (`Conv1d [O, I, k]`, `ConvTranspose1d [I, O, k]`), consumed directly by candle's NCL conv ops.
//! Everything runs **f32** (a post-sampling quality island). The STFT is the checkpoint's stored
//! `forward_basis`/`mel_basis` matmuls (no FFT). `x**2` is `x·x` here (candle has no `power`).

use candle_gen::candle_core::{DType, Device, Result, Tensor};

use crate::config::{VocoderConfig, VocoderGenConfig};

const LRELU_SLOPE: f64 = 0.1;

fn get(vb: &Vb, key: &str) -> Result<Tensor> {
    vb.inner
        .get_unchecked(key)?
        .to_dtype(DType::F32)?
        .contiguous()
}

fn get_opt(vb: &Vb, key: &str) -> Result<Option<Tensor>> {
    if vb.inner.contains_tensor(key) {
        Ok(Some(get(vb, key)?))
    } else {
        Ok(None)
    }
}

/// Load `{prefix}.weight` (f32), erroring loudly if a `{prefix}.scales` sibling is present — the vocoder
/// convs are never MLX-affine-packed, so a `.scales` sibling would mean a tier packed a leaf we would
/// otherwise load as u32-code garbage (sc-9417 guard, mirroring `quant::guard_no_scales`).
fn get_conv_weight(vb: &Vb, prefix: &str) -> Result<Tensor> {
    crate::quant::guard_no_scales(&vb.inner, prefix, DType::F32)?.contiguous()
}

/// Thin wrapper carrying the f32 `VarBuilder` (rooted at `vocoder`) + the device, so the loaders can
/// pull raw tensors by full key (the vocoder's PyTorch names don't map onto candle's nn modules).
struct Vb {
    inner: candle_gen::candle_nn::VarBuilder<'static>,
    device: Device,
}

/// `max(x, slope·x)` (LeakyReLU).
fn leaky(x: &Tensor, slope: f64) -> Result<Tensor> {
    x.maximum(&x.affine(slope, 0.0)?)
}

/// Replicate-pad the last (time) axis of a 3-D `(B, C, L)` tensor (edge values).
fn replicate_pad_l(x: &Tensor, left: usize, right: usize) -> Result<Tensor> {
    if left == 0 && right == 0 {
        return Ok(x.clone());
    }
    let (b, c, l) = x.dims3()?;
    let mut parts: Vec<Tensor> = Vec::new();
    if left > 0 {
        parts.push(
            x.narrow(2, 0, 1)?
                .broadcast_as((b, c, left))?
                .contiguous()?,
        );
    }
    parts.push(x.clone());
    if right > 0 {
        parts.push(
            x.narrow(2, l - 1, 1)?
                .broadcast_as((b, c, right))?
                .contiguous()?,
        );
    }
    Tensor::cat(&parts, 2)
}

/// 1-D conv (NCL), weight `[O, I, k]`, optional bias.
struct Conv1d {
    w: Tensor,
    b: Option<Tensor>,
    stride: usize,
    padding: usize,
    dilation: usize,
}

impl Conv1d {
    fn load(vb: &Vb, prefix: &str, stride: usize, padding: usize, dilation: usize) -> Result<Self> {
        Ok(Self {
            w: get_conv_weight(vb, prefix)?,
            b: get_opt(vb, &format!("{prefix}.bias"))?,
            stride,
            padding,
            dilation,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = x.conv1d(&self.w, self.padding, self.stride, self.dilation, 1)?;
        match &self.b {
            Some(b) => y.broadcast_add(&b.reshape((1, b.dim(0)?, 1))?),
            None => Ok(y),
        }
    }
}

/// 1-D transposed conv (NCL), weight `[I, O, k]`, optional bias.
struct ConvT1d {
    w: Tensor,
    b: Option<Tensor>,
    stride: usize,
    padding: usize,
}

impl ConvT1d {
    fn load(vb: &Vb, prefix: &str, stride: usize, padding: usize) -> Result<Self> {
        Ok(Self {
            w: get_conv_weight(vb, prefix)?,
            b: get_opt(vb, &format!("{prefix}.bias"))?,
            stride,
            padding,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = x.conv_transpose1d(&self.w, self.padding, 0, self.stride, 1, 1)?;
        match &self.b {
            Some(b) => y.broadcast_add(&b.reshape((1, b.dim(0)?, 1))?),
            None => Ok(y),
        }
    }
}

/// `x + sin²(exp(α)·x) / (exp(β) + 1e-6)` (`_SnakeCore`; log-scale α/β over channels). NCL.
struct SnakeCore {
    alpha: Tensor, // (1, C, 1)
    beta: Tensor,
}

impl SnakeCore {
    fn load(vb: &Vb, prefix: &str) -> Result<Self> {
        let alpha = get(vb, &format!("{prefix}.alpha"))?;
        let beta = get(vb, &format!("{prefix}.beta"))?;
        let c = alpha.dim(0)?;
        Ok(Self {
            alpha: alpha.exp()?.reshape((1, c, 1))?,
            beta: beta.exp()?.reshape((1, c, 1))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let s = x.broadcast_mul(&self.alpha)?.sin()?;
        let num = s.sqr()?;
        let den = self.beta.affine(1.0, 1e-6)?;
        x.broadcast_add(&num.broadcast_div(&den)?)
    }
}

/// Kaiser-sinc depth-wise filter (`_SnakeFilter`) — a stored `(1,1,taps)` kernel applied per channel
/// (same filter to every channel) along the time axis. NCL `(B, C, L)`.
struct SnakeFilter {
    filter: Tensor, // (1, 1, taps)
}

impl SnakeFilter {
    fn load(vb: &Vb, key: &str) -> Result<Self> {
        Ok(Self {
            filter: get(vb, key)?,
        })
    }

    fn apply_filter(&self, x: &Tensor, stride: usize) -> Result<Tensor> {
        let taps = self.filter.dim(2)?;
        let even = taps % 2 == 0;
        let pad_left = taps / 2 - usize::from(even);
        let pad_right = taps / 2;
        let (b, c, _l) = x.dims3()?;
        let x_padded = replicate_pad_l(x, pad_left, pad_right)?; // (B, C, L+pad)
        let total = x_padded.dim(2)?;
        // depth-wise: fold C into the batch, single-channel conv with the shared (1,1,taps) kernel.
        let x_flat = x_padded.reshape((b * c, 1, total))?;
        let out = x_flat.conv1d(&self.filter, 0, stride, 1, 1)?; // (B*C, 1, T_out)
        let t_out = out.dim(2)?;
        out.reshape((b, c, t_out))
    }
}

/// 2× upsample via zero-insert conv-transpose with the kaiser-sinc filter (`_SnakeUpsample`). NCL.
struct SnakeUpsample {
    filter: Tensor, // (1, 1, taps)
}

impl SnakeUpsample {
    fn load(vb: &Vb, key: &str) -> Result<Self> {
        Ok(Self {
            filter: get(vb, key)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let taps = self.filter.dim(2)?;
        let ratio = 2usize;
        let pad = taps / ratio - 1;
        let pad_left = pad * ratio + (taps - ratio) / 2;
        let pad_right = pad * ratio + (taps - ratio).div_ceil(2);
        let (b, c, _l) = x.dims3()?;
        let x_padded = replicate_pad_l(x, pad, pad)?; // (B, C, L+2pad)
        let total = x_padded.dim(2)?;
        let x_flat = x_padded.reshape((b * c, 1, total))?;
        let out = x_flat.conv_transpose1d(&self.filter, 0, 0, ratio, 1, 1)?; // (B*C, 1, T_up)
        let out = (out * ratio as f64)?;
        let t_up = out.dim(2)?;
        let mut out = out.reshape((b, c, t_up))?;
        // out[:, :, pad_left:] then [:, :, :-pad_right].
        out = out.narrow(2, pad_left, t_up - pad_left)?;
        if pad_right > 0 {
            let cur = out.dim(2)?;
            out = out.narrow(2, 0, cur - pad_right)?;
        }
        Ok(out)
    }
}

/// BigVGAN anti-aliased SnakeBeta activation: 2× upsample → SnakeBeta → low-pass + 2× downsample.
struct SnakeBeta {
    act: SnakeCore,
    up: SnakeUpsample,
    down: SnakeFilter,
}

impl SnakeBeta {
    fn load(vb: &Vb, prefix: &str) -> Result<Self> {
        Ok(Self {
            act: SnakeCore::load(vb, &format!("{prefix}.act"))?,
            up: SnakeUpsample::load(vb, &format!("{prefix}.upsample.filter"))?,
            down: SnakeFilter::load(vb, &format!("{prefix}.downsample.lowpass.filter"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.up.forward(x)?;
        let x = self.act.forward(&x)?;
        self.down.apply_filter(&x, 2)
    }
}

/// A vocoder residual block (BigVGAN `AMPBlock1` or HiFi-GAN `ResBlock1`/`ResBlock2`).
enum ResBlock {
    Amp1 {
        convs1: Vec<Conv1d>,
        convs2: Vec<Conv1d>,
        acts1: Vec<SnakeBeta>,
        acts2: Vec<SnakeBeta>,
    },
    Hifi1 {
        convs1: Vec<Conv1d>,
        convs2: Vec<Conv1d>,
    },
    Hifi2 {
        convs: Vec<Conv1d>,
    },
}

impl ResBlock {
    fn load(vb: &Vb, prefix: &str, kind: &str, kernel: i32, dilations: &[i32]) -> Result<Self> {
        let pad = |k: i32, d: i32| ((k - 1) * d / 2) as usize;
        match kind {
            "amp1" => {
                let mut convs1 = Vec::new();
                let mut convs2 = Vec::new();
                let mut acts1 = Vec::new();
                let mut acts2 = Vec::new();
                for (i, &d) in dilations.iter().enumerate() {
                    convs1.push(Conv1d::load(
                        vb,
                        &format!("{prefix}.convs1.{i}"),
                        1,
                        pad(kernel, d),
                        d as usize,
                    )?);
                    convs2.push(Conv1d::load(
                        vb,
                        &format!("{prefix}.convs2.{i}"),
                        1,
                        pad(kernel, 1),
                        1,
                    )?);
                    acts1.push(SnakeBeta::load(vb, &format!("{prefix}.acts1.{i}"))?);
                    acts2.push(SnakeBeta::load(vb, &format!("{prefix}.acts2.{i}"))?);
                }
                Ok(ResBlock::Amp1 {
                    convs1,
                    convs2,
                    acts1,
                    acts2,
                })
            }
            "2" => {
                let mut convs = Vec::new();
                for (i, &d) in dilations.iter().enumerate() {
                    convs.push(Conv1d::load(
                        vb,
                        &format!("{prefix}.convs.{i}"),
                        1,
                        pad(kernel, d),
                        d as usize,
                    )?);
                }
                Ok(ResBlock::Hifi2 { convs })
            }
            _ => {
                let mut convs1 = Vec::new();
                let mut convs2 = Vec::new();
                for (i, &d) in dilations.iter().enumerate() {
                    convs1.push(Conv1d::load(
                        vb,
                        &format!("{prefix}.convs1.{i}"),
                        1,
                        pad(kernel, d),
                        d as usize,
                    )?);
                    convs2.push(Conv1d::load(
                        vb,
                        &format!("{prefix}.convs2.{i}"),
                        1,
                        pad(kernel, 1),
                        1,
                    )?);
                }
                Ok(ResBlock::Hifi1 { convs1, convs2 })
            }
        }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            ResBlock::Amp1 {
                convs1,
                convs2,
                acts1,
                acts2,
            } => {
                let mut x = x.clone();
                for i in 0..convs1.len() {
                    let xt = acts1[i].forward(&x)?;
                    let xt = convs1[i].forward(&xt)?;
                    let xt = acts2[i].forward(&xt)?;
                    let xt = convs2[i].forward(&xt)?;
                    x = (xt + x)?;
                }
                Ok(x)
            }
            ResBlock::Hifi1 { convs1, convs2 } => {
                let mut x = x.clone();
                for i in 0..convs1.len() {
                    let xt = convs1[i].forward(&leaky(&x, LRELU_SLOPE)?)?;
                    let xt = convs2[i].forward(&leaky(&xt, LRELU_SLOPE)?)?;
                    x = (xt + x)?;
                }
                Ok(x)
            }
            ResBlock::Hifi2 { convs } => {
                let mut x = x.clone();
                for c in convs {
                    let xt = c.forward(&leaky(&x, LRELU_SLOPE)?)?;
                    x = (xt + x)?;
                }
                Ok(x)
            }
        }
    }
}

/// A HiFi-GAN / BigVGAN generator (`Vocoder` / `BigVGANVocoder`).
pub struct Generator {
    conv_pre: Conv1d,
    ups: Vec<ConvT1d>,
    resblocks: Vec<ResBlock>,
    act_post: Option<SnakeBeta>, // BigVGAN only
    conv_post: Conv1d,
    num_kernels: usize,
    bigvgan: bool,
    use_tanh_at_final: bool,
    apply_final_activation: bool,
}

impl Generator {
    /// Build from `vb` under `prefix` (`"vocoder.vocoder"` for the core, `"vocoder.bwe_generator"`
    /// for BWE — the dense checkpoint nests the core generator one level deeper than the mlx bundle).
    fn load(vb: &Vb, prefix: &str, cfg: &VocoderGenConfig) -> Result<Self> {
        let bigvgan = cfg.is_bigvgan();
        let kind = if bigvgan {
            "amp1"
        } else if cfg.resblock == "2" {
            "2"
        } else {
            "1"
        };
        let num_upsamples = cfg.upsample_rates.len();
        let num_kernels = cfg.resblock_kernel_sizes.len();

        let conv_pre = {
            let w = get(vb, &format!("{prefix}.conv_pre.weight"))?;
            let k = w.dim(2)? as usize;
            Conv1d::load(vb, &format!("{prefix}.conv_pre"), 1, k / 2, 1)?
        };
        let mut ups = Vec::with_capacity(num_upsamples);
        for (i, (&stride, &k)) in cfg
            .upsample_rates
            .iter()
            .zip(cfg.upsample_kernel_sizes.iter())
            .enumerate()
        {
            ups.push(ConvT1d::load(
                vb,
                &format!("{prefix}.ups.{i}"),
                stride as usize,
                ((k - stride) / 2) as usize,
            )?);
        }
        let mut resblocks = Vec::with_capacity(num_upsamples * num_kernels);
        let mut idx = 0;
        for _ in 0..num_upsamples {
            for (&k, dil) in cfg
                .resblock_kernel_sizes
                .iter()
                .zip(cfg.resblock_dilation_sizes.iter())
            {
                resblocks.push(ResBlock::load(
                    vb,
                    &format!("{prefix}.resblocks.{idx}"),
                    kind,
                    k,
                    dil,
                )?);
                idx += 1;
            }
        }
        let act_post = if bigvgan {
            Some(SnakeBeta::load(vb, &format!("{prefix}.act_post"))?)
        } else {
            None
        };
        let conv_post = {
            let w = get(vb, &format!("{prefix}.conv_post.weight"))?;
            let k = w.dim(2)? as usize;
            Conv1d::load(vb, &format!("{prefix}.conv_post"), 1, k / 2, 1)?
        };
        Ok(Self {
            conv_pre,
            ups,
            resblocks,
            act_post,
            conv_post,
            num_kernels,
            bigvgan,
            use_tanh_at_final: cfg.use_tanh_at_final,
            apply_final_activation: cfg.apply_final_activation,
        })
    }

    /// `(B, C, T, F)` mel/feature input → NCL `(B, C·F, T)` → `conv_pre`.
    fn pre(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, t, f) = x.dims4()?;
        // (B,C,T,F) → (B,C,F,T) → (B, C·F, T).
        let x = x
            .permute((0, 1, 3, 2))?
            .reshape((b, c * f, t))?
            .contiguous()?;
        self.conv_pre.forward(&x)
    }

    /// Per stage: optional leaky pre-act → transposed-conv upsample → mean of `num_kernels` outputs.
    fn up_loop(&self, mut x: Tensor) -> Result<Tensor> {
        for i in 0..self.ups.len() {
            if !self.bigvgan {
                x = leaky(&x, LRELU_SLOPE)?;
            }
            x = self.ups[i].forward(&x)?;
            let start = i * self.num_kernels;
            let outs: Vec<Tensor> = (start..start + self.num_kernels)
                .map(|idx| self.resblocks[idx].forward(&x))
                .collect::<Result<Vec<_>>>()?;
            x = Tensor::stack(&outs, 0)?.mean(0)?;
        }
        Ok(x)
    }

    /// `x` is the mel/feature input `(B, C, T, F)` (stereo `C=2`). Returns the waveform `(B, C_out, T)`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.pre(x)?;
        let mut x = self.up_loop(x)?;
        if self.bigvgan {
            x = self.act_post.as_ref().unwrap().forward(&x)?;
            x = self.conv_post.forward(&x)?;
            if self.apply_final_activation {
                x = if self.use_tanh_at_final {
                    x.tanh()?
                } else {
                    x.clamp(-1.0, 1.0)?
                };
            }
        } else {
            x = leaky(&x, 0.01)?;
            x = self.conv_post.forward(&x)?;
            x = x.tanh()?;
        }
        Ok(x) // (B, C_out, T) — already NCL
    }
}

/// Stored windowed-STFT + mel basis (`_MelSTFT` / `_STFTBasis`) for the BWE mel computation.
struct MelStft {
    forward_basis: Tensor, // (2·n_freq, 1, win)
    mel_basis: Tensor,     // (n_mels, n_freq)
}

/// Log-mel of a (left-padded) flattened waveform `(BC, T)`. `forward_basis` is the `(2·n_freq,1,win)`
/// STFT kernel (stacked real+imag rows), `mel_basis` the `(n_mels, n_freq)` filterbank. Gathers all
/// `win`-length windows `(BC, n_frames, win)` then one batched matmul against the basis.
fn stft_log_mel(
    x: &Tensor,
    forward_basis: &Tensor,
    mel_basis: &Tensor,
    hop: usize,
    win: usize,
    device: &Device,
) -> Result<Tensor> {
    let (bc, total) = x.dims2()?;
    let n_freq2 = forward_basis.dim(0)?;
    let n_freq = n_freq2 / 2;
    let n_frames = ((total.saturating_sub(win)) / hop + 1).max(1);

    // Gather every sliding window at once: idx[i, k] = i·hop + k.
    let mut idx = Vec::with_capacity(n_frames * win);
    for i in 0..n_frames {
        for k in 0..win {
            idx.push((i * hop + k) as u32);
        }
    }
    let idx = Tensor::from_vec(idx, n_frames * win, device)?;
    let windows = x.index_select(&idx, 1)?.reshape((bc, n_frames, win))?; // (BC, n_frames, win)

    // STFT: windows @ basisᵀ, batched.
    let basis_t = forward_basis.squeeze(1)?.t()?.contiguous()?; // (win, 2·n_freq)
    let spec = windows
        .reshape((bc * n_frames, win))?
        .matmul(&basis_t)?
        .reshape((bc, n_frames, n_freq2))?;

    let real = spec.narrow(2, 0, n_freq)?;
    let imag = spec.narrow(2, n_freq, n_freq)?;
    let magnitude = (real.sqr()? + imag.sqr()?)?.sqrt()?; // (BC, n_frames, n_freq)

    let mel_basis_t = mel_basis.t()?.contiguous()?; // (n_freq, n_mels)
    let n_mels = mel_basis_t.dim(1)?;
    let mel = magnitude
        .reshape((bc * n_frames, n_freq))?
        .matmul(&mel_basis_t)?
        .reshape((bc, n_frames, n_mels))?;
    mel.maximum(1e-5f64)?.log()
}

/// BigVGAN core + bandwidth-extension (`VocoderWithBWE`). The shipped 2.3 vocoder (48 kHz).
pub struct VocoderWithBwe {
    vocoder: Generator,
    bwe_generator: Generator,
    mel_stft: MelStft,
    input_sr: usize,
    output_sr: usize,
    hop: usize,
    win: usize,
    device: Device,
}

impl VocoderWithBwe {
    fn load(vb: &Vb, cfg: &VocoderConfig) -> Result<Self> {
        let bwe_cfg = cfg
            .bwe
            .as_ref()
            .expect("VocoderWithBwe requires a bwe config");
        Ok(Self {
            // In the dense checkpoint the core generator is nested under `vocoder.vocoder.*` (the
            // VocoderWithBWE module's `.vocoder` submodule); the BWE under `vocoder.bwe_generator.*`.
            vocoder: Generator::load(vb, "vocoder.vocoder", &cfg.core)?,
            bwe_generator: Generator::load(vb, "vocoder.bwe_generator", bwe_cfg)?,
            mel_stft: MelStft {
                forward_basis: get(vb, "vocoder.mel_stft.stft_fn.forward_basis")?,
                mel_basis: get(vb, "vocoder.mel_stft.mel_basis")?,
            },
            input_sr: cfg.bwe_input_sample_rate as usize,
            output_sr: cfg.bwe_output_sample_rate as usize,
            hop: cfg.bwe_hop_length as usize,
            win: cfg.bwe_win_length as usize,
            device: vb.device.clone(),
        })
    }

    /// Log-mel from a waveform `(B, C, T)` → `(B, C, n_mels, T_frames)`.
    fn compute_mel(&self, audio: &Tensor) -> Result<Tensor> {
        let (b, c, t) = audio.dims3()?;
        let mut x = audio.reshape((b * c, t))?;
        let left_pad = self.win.saturating_sub(self.hop);
        if left_pad > 0 {
            x = x.pad_with_zeros(1, left_pad, 0)?;
        }
        if x.dim(1)? < self.win {
            x = x.pad_with_zeros(1, 0, self.win - x.dim(1)?)?;
        }
        let mel_bt = stft_log_mel(
            &x,
            &self.mel_stft.forward_basis,
            &self.mel_stft.mel_basis,
            self.hop,
            self.win,
            &self.device,
        )?; // (B*C, T_frames, n_mels)
        let (n_frames, n_mels) = (mel_bt.dim(1)?, mel_bt.dim(2)?);
        // (B*C, T_frames, n_mels) → (B, C, n_mels, T_frames).
        mel_bt
            .reshape((b, c, n_frames, n_mels))?
            .permute((0, 1, 3, 2))?
            .contiguous()
    }

    /// Linear-interp upsample of the skip connection to the BWE rate (`_upsample_skip`). `(B,C,T)`.
    fn upsample_skip(&self, x: &Tensor) -> Result<Tensor> {
        let ratio = (self.output_sr / self.input_sr).max(1);
        if ratio <= 1 {
            return Ok(x.clone());
        }
        let (_b, _c, t) = x.dims3()?;
        let t_out = t * ratio;
        let mut floor_idx = Vec::with_capacity(t_out);
        let mut ceil_idx = Vec::with_capacity(t_out);
        let mut frac = Vec::with_capacity(t_out);
        for i in 0..t_out {
            let pos = i as f64 / ratio as f64;
            let fl = (pos.floor() as i64).clamp(0, (t - 1) as i64) as usize;
            let cl = ((pos.floor() as i64) + 1).clamp(0, (t - 1) as i64) as usize;
            floor_idx.push(fl as u32);
            ceil_idx.push(cl as u32);
            frac.push((pos - pos.floor()) as f32);
        }
        let fl = Tensor::from_vec(floor_idx, t_out, &self.device)?;
        let cl = Tensor::from_vec(ceil_idx, t_out, &self.device)?;
        let frac = Tensor::from_vec(frac, (1, 1, t_out), &self.device)?;
        let lo = x.index_select(&fl, 2)?;
        let hi = x.index_select(&cl, 2)?;
        lo.broadcast_add(&frac.broadcast_mul(&(hi - &lo)?)?)
    }

    /// `(B, C, T_low)` low → mel → BWE residual + linear-interp skip, summed and clipped.
    pub fn forward(&self, mel_spec: &Tensor) -> Result<Tensor> {
        let low = self.vocoder.forward(mel_spec)?; // (B, C, T_low)
        let mel_from_low = self.compute_mel(&low)?; // (B, C, n_mels, T_frames)
        let mel_for_bwe = mel_from_low.permute((0, 1, 3, 2))?.contiguous()?; // (B, C, T, n_mels)
        let residual = self.bwe_generator.forward(&mel_for_bwe)?; // (B, C, T_high)
        let skip = self.upsample_skip(&low)?;
        let target = residual.dim(2)?.min(skip.dim(2)?);
        let residual = residual.narrow(2, 0, target)?;
        let skip = skip.narrow(2, 0, target)?;
        (residual + skip)?.clamp(-1.0, 1.0)
    }
}

/// The selected LTX vocoder (`load_vocoder`): HiFi-GAN / BigVGAN core, or the core + BWE wrapper.
pub enum LtxVocoder {
    Plain(Generator),
    Bwe(Box<VocoderWithBwe>),
}

impl LtxVocoder {
    /// Build from a `VarBuilder` (the f32 builder at the **checkpoint root**) + the [`VocoderConfig`].
    pub fn load(
        vb: candle_gen::candle_nn::VarBuilder<'static>,
        device: &Device,
        cfg: &VocoderConfig,
    ) -> Result<Self> {
        let vb = Vb {
            inner: vb,
            device: device.clone(),
        };
        if cfg.bwe.is_some() {
            Ok(LtxVocoder::Bwe(Box::new(VocoderWithBwe::load(&vb, cfg)?)))
        } else {
            Ok(LtxVocoder::Plain(Generator::load(
                &vb,
                "vocoder.vocoder",
                &cfg.core,
            )?))
        }
    }

    /// Mel `(B, C, T, F)` → waveform `(B, C_out, T)`.
    pub fn forward(&self, mel_spec: &Tensor) -> Result<Tensor> {
        match self {
            LtxVocoder::Plain(g) => g.forward(mel_spec),
            LtxVocoder::Bwe(v) => v.forward(mel_spec),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replicate_pad_edges() {
        // (1, 1, 3) = [5, 7, 9] → pad (2, 1) replicates first/last: [5,5,5,7,9,9].
        let x = Tensor::from_vec(vec![5.0f32, 7.0, 9.0], (1, 1, 3), &Device::Cpu).unwrap();
        let y = replicate_pad_l(&x, 2, 1).unwrap();
        assert_eq!(y.dims(), &[1, 1, 6]);
        let v = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(v, vec![5.0, 5.0, 5.0, 7.0, 9.0, 9.0]);
    }

    #[test]
    fn leaky_relu_slope() {
        let x = Tensor::from_vec(vec![-2.0f32, 0.0, 3.0], 3, &Device::Cpu).unwrap();
        let y = leaky(&x, 0.1).unwrap().to_vec1::<f32>().unwrap();
        assert!((y[0] - -0.2).abs() < 1e-6);
        assert_eq!(y[1], 0.0);
        assert_eq!(y[2], 3.0);
    }

    #[test]
    fn stft_log_mel_shape() {
        // bc=2, total=40, win=8, hop=4 → n_frames = (40-8)/4+1 = 9; n_freq=3, n_mels=4.
        let dev = Device::Cpu;
        let x = Tensor::randn(0.0f32, 1.0, (2, 40), &dev).unwrap();
        let fb = Tensor::randn(0.0f32, 1.0, (6, 1, 8), &dev).unwrap(); // 2·n_freq=6
        let mb = Tensor::randn(0.0f32, 1.0, (4, 3), &dev)
            .unwrap()
            .abs()
            .unwrap();
        let mel = stft_log_mel(&x, &fb, &mb, 4, 8, &dev).unwrap();
        assert_eq!(mel.dims(), &[2, 9, 4]);
    }
}
