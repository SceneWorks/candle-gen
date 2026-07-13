//! DC-AE (deep-compression autoencoder) **f32 image path** and the **EfficientViT GLU** ReLU
//! linear-attention block — gating spike sc-11777 (epic 11776), the candle/CUDA mirror of the
//! mlx-gen-sana port (mlx-gen #612, spike sc-8486).
//!
//! Scope is the image path only (no 3D/temporal ops). The whole autoencoder runs **f32** (the
//! checkpoint is f32 and the linear-attention `1/(Σ+eps)` normalizer is f32 in the reference
//! regardless) — the port-playbook's "f32 or f32/split" convention; the deep-compression decode is a
//! single 32× image decode, so f32 memory is not the concern a video VAE's is.
//!
//! **Layout: channels-first NCHW throughout** (candle-native, and the diffusers-native checkpoint
//! layout). This is the key simplification vs the mlx port, which works in MLX-native NHWC and
//! transposes `movedim(1,-1)` around every channel-wise Linear/RMSNorm and back to NCHW for the
//! multi-head attention reshape. Here the conv weights load `[O, I/groups, kH, kW]` as-is (no
//! transpose), the per-channel `trms2d`/`Linear` ops act on axis 1, and the attention reshape is
//! already channels-first — so this port is *fewer* ops than the reference while computing the
//! identical result (the element ordering into the `[B, groups, 3·head_dim, H·W]` attention reshape
//! is the same, because the mlx port transposes to that same NCHW view first).
//!
//! Block fidelity (component-for-component vs the mlx-gen-sana reference, which is itself the faithful
//! diffusers port):
//!  - [`ResBlock`]: `conv1 → SiLU → conv2(no-bias) → trms2d(channel) → + residual`.
//!  - [`EfficientVitBlock`]: `LinearAttn → GluMbConv`.
//!  - [`LinearAttn`] (`SanaMultiscaleLinearAttention`): per-pixel `to_q/k/v`(no bias) → multiscale
//!    depthwise+grouped QKV projections → per-head `ReLU(Q),ReLU(K)` linear attention with a
//!    `1/(Σ+eps)` normalizer (the algebraically-identical numerator/denominator split of the
//!    reference's ones-row `F.pad`, same f32 sums) → `to_out`(no bias) → trms2d → + residual.
//!  - [`GluMbConv`] (`GLUMBConv`): `conv_inverted(1×1) → SiLU → conv_depth(3×3 depthwise) → gated
//!    SiLU → conv_point(1×1 no-bias) → trms2d → + residual`.
//!  - [`UpBlock`] (`DCUpBlock2d`, interpolate): `nearest-upsample → conv`, + a channel shortcut
//!    `repeat_interleave → pixel_shuffle`.
//!  - [`DownBlock`] (`DCDownBlock2d`, the encoder mirror): `pixel_unshuffle → conv`, + a channel
//!    shortcut `pixel_unshuffle → channel-average`.

// The pure tensor helpers + `forward` methods return candle-core's `Result` (candle ops' native
// error). The weight-loaders return `candle_gen::Result` (its `CandleError` is what `Weights::require`
// yields, and candle-core errors `?`-convert into it); those are annotated `candle_gen::Result`
// explicitly.
use candle_gen::candle_core::{DType, Result, Tensor, D};
use candle_gen::candle_nn::ops::silu;
use candle_gen::candle_nn::{Conv2d, Conv2dConfig, Module};
use candle_gen::Weights;

use crate::config::{BlockType, DcAeConfig};

// ---------------------------------------------------------------------------------------------------
// Shared primitives
// ---------------------------------------------------------------------------------------------------

/// A conv whose on-disk weight is the diffusers-native `[O, I/groups, kH, kW]`, consumed by candle's
/// NCHW `Conv2d` as-is (no transpose — the MLX port transposes to `[O, kH, kW, I/groups]`). Stored
/// f32.
fn conv(
    w: &Weights,
    prefix: &str,
    stride: usize,
    padding: usize,
    groups: usize,
    bias: bool,
) -> candle_gen::Result<Conv2d> {
    let weight = w
        .require(&format!("{prefix}.weight"))?
        .to_dtype(DType::F32)?;
    let b = if bias {
        Some(w.require(&format!("{prefix}.bias"))?.to_dtype(DType::F32)?)
    } else {
        None
    };
    let cfg = Conv2dConfig {
        padding,
        stride,
        dilation: 1,
        groups,
        cudnn_fwd_algo: None,
    };
    Ok(Conv2d::new(weight, b, cfg))
}

/// A per-pixel channel `Linear` (no bias) stored as a 2D `[out, in]` weight, realised as a 1×1
/// NCHW conv (numerically identical: a pointwise channel projection). `to_q/k/v/out` in the
/// linear-attention block are these.
fn channel_linear(w: &Weights, prefix: &str) -> candle_gen::Result<Conv2d> {
    let weight = w
        .require(&format!("{prefix}.weight"))?
        .to_dtype(DType::F32)?;
    let (out, inn) = weight.dims2()?;
    let weight = weight.reshape((out, inn, 1, 1))?;
    Ok(Conv2d::new(weight, None, Conv2dConfig::default()))
}

/// Channel-wise (per-pixel) RMSNorm over the channel axis, computed in f32 — the DC-AE `trms2d`
/// ("trimmed RMS", `norm_type = rms_norm`). `weight`/`bias` are per-channel `[C]`, broadcast as
/// `[1, C, 1, 1]`. Mirrors the mlx `rms_norm` (which reduces over the NHWC last axis = channels);
/// in NCHW that reduction is axis 1.
struct Trms2d {
    weight: Tensor, // [1, C, 1, 1]
    bias: Tensor,   // [1, C, 1, 1]
    eps: f64,
}

impl Trms2d {
    fn load(w: &Weights, prefix: &str, eps: f32) -> candle_gen::Result<Self> {
        let weight = w
            .require(&format!("{prefix}.weight"))?
            .to_dtype(DType::F32)?;
        let bias = w.require(&format!("{prefix}.bias"))?.to_dtype(DType::F32)?;
        let c = weight.elem_count();
        Ok(Self {
            weight: weight.reshape((1, c, 1, 1))?,
            bias: bias.reshape((1, c, 1, 1))?,
            eps: eps as f64,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let xf = x.to_dtype(DType::F32)?;
        let var = xf.sqr()?.mean_keepdim(1)?; // mean over channel axis
        let denom = (var + self.eps)?.sqrt()?;
        let normed = xf.broadcast_div(&denom)?;
        normed
            .broadcast_mul(&self.weight)?
            .broadcast_add(&self.bias)
    }
}

/// `PixelShuffle(r)` (NCHW, channel→space): `[B, C·r², H, W] → [B, C, H·r, W·r]`, viewing the
/// `C·r²` channels as `(C, r, r)` (C slowest) — exactly `torch.nn.PixelShuffle` / the mlx NHWC
/// upsampler. Used by [`UpBlock`]'s channel shortcut.
fn pixel_shuffle(x: &Tensor, r: usize) -> Result<Tensor> {
    let (b, c, h, wd) = x.dims4()?;
    let out_c = c / (r * r);
    x.reshape((b, out_c, r, r, h, wd))?
        .permute((0, 1, 4, 2, 5, 3))? // [B, out_c, H, r, W, r]
        .reshape((b, out_c, h * r, wd * r))
}

/// `PixelUnshuffle(r)` (NCHW, space→channel): `[B, C, H·r, W·r] → [B, C·r², H, W]` — the exact
/// inverse of [`pixel_shuffle`]. Used by [`DownBlock`]'s conv + channel shortcut.
fn pixel_unshuffle(x: &Tensor, r: usize) -> Result<Tensor> {
    let (b, c, h, wd) = x.dims4()?;
    let (oh, ow) = (h / r, wd / r);
    x.reshape((b, c, oh, r, ow, r))?
        .permute((0, 1, 3, 5, 2, 4))? // [B, C, r, r, oh, ow]
        .reshape((b, c * r * r, oh, ow))
}

/// `repeat_interleave` along the channel axis: each channel duplicated `r` times in place
/// (`c0,c0,…,c1,c1,…`). The [`UpBlock`] channel-shortcut expander (inverse of [`channel_average`]).
fn repeat_interleave_channel(x: &Tensor, r: usize) -> Result<Tensor> {
    let (b, c, h, wd) = x.dims4()?;
    x.reshape((b, c, 1, h, wd))?
        .broadcast_as((b, c, r, h, wd))?
        .reshape((b, c * r, h, wd))
}

/// Average each contiguous group of `in_c / out_c` channels into one (`out_c` groups) — the inverse
/// of [`repeat_interleave_channel`], the [`DownBlock`] channel-shortcut reducer.
fn channel_average(x: &Tensor, out_c: usize) -> Result<Tensor> {
    let (b, c, h, wd) = x.dims4()?;
    let g = c / out_c;
    x.reshape((b, out_c, g, h, wd))?.mean(2)
}

// ---------------------------------------------------------------------------------------------------
// ResBlock
// ---------------------------------------------------------------------------------------------------

/// `ResBlock` (`norm_type = rms_norm`, `act_fn = silu`): `conv1 → SiLU → conv2(no-bias) → trms2d →
/// + residual`.
struct ResBlock {
    conv1: Conv2d,
    conv2: Conv2d,
    norm: Trms2d,
}

impl ResBlock {
    fn load(w: &Weights, prefix: &str, eps: f32) -> candle_gen::Result<Self> {
        Ok(Self {
            conv1: conv(w, &format!("{prefix}.conv1"), 1, 1, 1, true)?,
            conv2: conv(w, &format!("{prefix}.conv2"), 1, 1, 1, false)?,
            norm: Trms2d::load(w, &format!("{prefix}.norm"), eps)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.conv1.forward(x)?;
        let h = silu(&h)?;
        let h = self.conv2.forward(&h)?;
        let h = self.norm.forward(&h)?;
        h + x
    }
}

// ---------------------------------------------------------------------------------------------------
// EfficientViT linear-attention (the shared hard primitive)
// ---------------------------------------------------------------------------------------------------

/// The core ReLU-**linear**-attention kernel (O(N), softmax-free) shared by the DC-AE EfficientViT
/// block and (story 2) the SANA Linear-DiT trunk. Inputs are per-head `[B, groups, head_dim, N]`
/// with `q`/`k` **already ReLU'd**; `v` un-activated. Computes, for every query position `i`:
///
/// `out[:,i] = Σ_n v[:,n] · (φ(q_i)·φ(k_n)) / (Σ_n φ(q_i)·φ(k_n) + eps)`
///
/// via the associative form `num = (v·kᵀ)·q`, `den = (Σ_n k)·q` — the algebraic identity of the
/// reference's ones-row `F.pad(value)` normalizer, in the same f32 sums. O(N) because the `[hd,hd]`
/// (and `[1,hd]`) contractions are formed first, never the `[N,N]` score matrix.
pub(crate) fn relu_linear_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    eps: f64,
) -> Result<Tensor> {
    let k_t = k.transpose(D::Minus1, D::Minus2)?.contiguous()?; // [B, g, N, hd]
    let num = v.matmul(&k_t)?.matmul(q)?; // (v·kᵀ)·q = [B, g, hd, N]
    let k_sum = k.sum_keepdim(D::Minus1)?.transpose(D::Minus1, D::Minus2)?; // [B, g, 1, hd]
    let den = k_sum.matmul(q)?; // [B, g, 1, N]
    num.broadcast_div(&(den + eps)?) // broadcast the [.,.,1,N] denominator over head_dim
}

/// One multiscale QKV projection: depthwise `proj_in` (kernel `k`, groups = channels) → grouped 1×1
/// `proj_out` (groups = 3·num_heads). Both bias-free. Operates on the concatenated `[B, 3·inner, H, W]`
/// qkv.
struct MultiscaleProj {
    proj_in: Conv2d,
    proj_out: Conv2d,
}

impl MultiscaleProj {
    fn load(
        w: &Weights,
        prefix: &str,
        kernel: usize,
        channels: usize,
        num_heads: usize,
    ) -> candle_gen::Result<Self> {
        Ok(Self {
            proj_in: conv(
                w,
                &format!("{prefix}.proj_in"),
                1,
                kernel / 2,
                channels,
                false,
            )?,
            proj_out: conv(w, &format!("{prefix}.proj_out"), 1, 0, 3 * num_heads, false)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.proj_out.forward(&self.proj_in.forward(x)?)
    }
}

/// `SanaMultiscaleLinearAttention` (residual). For SANA-1.0 `inner == in_channels` (head_dim 32,
/// scale mult 1.0), so `num_heads = channels / head_dim`.
struct LinearAttn {
    to_q: Conv2d,
    to_k: Conv2d,
    to_v: Conv2d,
    to_out: Conv2d,
    projs: Vec<MultiscaleProj>,
    norm: Trms2d,
    head_dim: usize,
    num_heads: usize,
    attn_eps: f64,
}

impl LinearAttn {
    fn load(
        w: &Weights,
        prefix: &str,
        cfg: &DcAeConfig,
        channels: usize,
    ) -> candle_gen::Result<Self> {
        let head_dim = cfg.attention_head_dim as usize;
        let num_heads = channels / head_dim; // mult 1.0 → inner == channels
        let mut projs = Vec::new();
        for (i, k) in cfg.qkv_multiscales.iter().enumerate() {
            projs.push(MultiscaleProj::load(
                w,
                &format!("{prefix}.to_qkv_multiscale.{i}"),
                *k as usize,
                3 * channels, // the proj operates over the concatenated [q,k,v] = 3·inner channels
                num_heads,
            )?);
        }
        Ok(Self {
            to_q: channel_linear(w, &format!("{prefix}.to_q"))?,
            to_k: channel_linear(w, &format!("{prefix}.to_k"))?,
            to_v: channel_linear(w, &format!("{prefix}.to_v"))?,
            to_out: channel_linear(w, &format!("{prefix}.to_out"))?,
            projs,
            norm: Trms2d::load(w, &format!("{prefix}.norm_out"), cfg.norm_eps)?,
            head_dim,
            num_heads,
            attn_eps: cfg.attn_eps as f64,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, _c, h, wd) = x.dims4()?;
        let hd = self.head_dim;
        // qkv (channels-first), then concat the multiscale projections along the channel axis.
        let q = self.to_q.forward(x)?;
        let k = self.to_k.forward(x)?;
        let v = self.to_v.forward(x)?;
        let qkv = Tensor::cat(&[&q, &k, &v], 1)?; // [B, 3·inner, H, W]
        let mut multi = vec![qkv.clone()];
        for proj in &self.projs {
            multi.push(proj.forward(&qkv)?);
        }
        let hidden = Tensor::cat(&multi.iter().collect::<Vec<_>>(), 1)?; // [B, C_tot, H, W]

        // The channel axis C_tot = 3·hd·groups reshapes to [B, groups, 3·hd, H·W]; groups =
        // num_heads·(1+scales). (No transpose: candle is already in the mlx port's post-transpose
        // NCHW view, so the element ordering into this reshape is identical.)
        let hw = h * wd;
        let groups = self.num_heads * (1 + self.projs.len());
        let hidden = hidden.reshape((b, groups, 3 * hd, hw))?;
        // chunk(3) over the 3·hd axis → q,k,v each [B, groups, hd, HW]
        let q = hidden.narrow(2, 0, hd)?.relu()?;
        let k = hidden.narrow(2, hd, hd)?.relu()?;
        let v = hidden.narrow(2, 2 * hd, hd)?.contiguous()?;

        let out = relu_linear_attention(&q.contiguous()?, &k.contiguous()?, &v, self.attn_eps)?;

        // → [B, inner·(1+scales), H, W], to_out, trms2d, residual.
        let out = out.reshape((b, groups * hd, h, wd))?;
        let out = self.to_out.forward(&out)?;
        let out = self.norm.forward(&out)?;
        out + x
    }
}

// ---------------------------------------------------------------------------------------------------
// GLUMBConv
// ---------------------------------------------------------------------------------------------------

/// `GLUMBConv` (`rms_norm`, residual, expand_ratio 4).
struct GluMbConv {
    conv_inverted: Conv2d, // 1×1, in → 2·hidden
    conv_depth: Conv2d,    // 3×3 depthwise, 2·hidden → 2·hidden
    conv_point: Conv2d,    // 1×1 no-bias, hidden → out
    norm: Trms2d,
    hidden: usize,
}

impl GluMbConv {
    fn load(w: &Weights, prefix: &str, channels: usize, eps: f32) -> candle_gen::Result<Self> {
        let hidden = 4 * channels;
        Ok(Self {
            conv_inverted: conv(w, &format!("{prefix}.conv_inverted"), 1, 0, 1, true)?,
            conv_depth: conv(w, &format!("{prefix}.conv_depth"), 1, 1, 2 * hidden, true)?,
            conv_point: conv(w, &format!("{prefix}.conv_point"), 1, 0, 1, false)?,
            norm: Trms2d::load(w, &format!("{prefix}.norm"), eps)?,
            hidden,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.conv_inverted.forward(x)?;
        let h = silu(&h)?;
        let h = self.conv_depth.forward(&h)?;
        // chunk(2) over the channel axis → gate.
        let a = h.narrow(1, 0, self.hidden)?;
        let g = h.narrow(1, self.hidden, self.hidden)?;
        let h = (a * silu(&g)?)?;
        let h = self.conv_point.forward(&h)?;
        let h = self.norm.forward(&h)?;
        h + x
    }
}

// ---------------------------------------------------------------------------------------------------
// Block dispatch + up/down stages
// ---------------------------------------------------------------------------------------------------

/// An `EfficientViTBlock`: linear attention then the GLU mix-conv.
struct EfficientVitBlock {
    attn: LinearAttn,
    conv_out: GluMbConv,
}

impl EfficientVitBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.conv_out.forward(&self.attn.forward(x)?)
    }
}

enum Block {
    Res(ResBlock),
    // Boxed: the EfficientViT block (linear-attn + GLU) is much larger than `ResBlock`, so an unboxed
    // variant would bloat every `Block` (clippy::large_enum_variant).
    Evit(Box<EfficientVitBlock>),
}

impl Block {
    fn load(
        w: &Weights,
        prefix: &str,
        cfg: &DcAeConfig,
        ty: BlockType,
        ch: usize,
    ) -> candle_gen::Result<Self> {
        Ok(match ty {
            BlockType::Res => Block::Res(ResBlock::load(w, prefix, cfg.norm_eps)?),
            BlockType::EfficientVit => Block::Evit(Box::new(EfficientVitBlock {
                attn: LinearAttn::load(w, &format!("{prefix}.attn"), cfg, ch)?,
                conv_out: GluMbConv::load(w, &format!("{prefix}.conv_out"), ch, cfg.norm_eps)?,
            })),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Block::Res(b) => b.forward(x),
            Block::Evit(b) => b.forward(x),
        }
    }
}

/// `DCUpBlock2d` (interpolate path): `nearest-upsample → conv`, + the `repeat_interleave →
/// pixel_shuffle` channel shortcut. `in_ch` = deeper stage channels, `out_ch` = this stage's.
struct UpBlock {
    conv: Conv2d,
    repeats: usize,
}

impl UpBlock {
    fn load(w: &Weights, prefix: &str, in_ch: usize, out_ch: usize) -> candle_gen::Result<Self> {
        Ok(Self {
            conv: conv(w, &format!("{prefix}.conv"), 1, 1, 1, true)?,
            repeats: out_ch * 4 / in_ch,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_b, _c, h, wd) = x.dims4()?;
        let up = self.conv.forward(&x.upsample_nearest2d(h * 2, wd * 2)?)?;
        let shortcut = pixel_shuffle(&repeat_interleave_channel(x, self.repeats)?, 2)?;
        up + shortcut
    }
}

/// `DCDownBlock2d` (the encoder mirror): `pixel_unshuffle → conv`, + the `pixel_unshuffle →
/// channel-average` shortcut. `in_ch` = this stage's channels, `out_ch` = deeper stage's.
struct DownBlock {
    conv: Conv2d,
    out_ch: usize,
}

impl DownBlock {
    fn load(w: &Weights, prefix: &str, out_ch: usize) -> candle_gen::Result<Self> {
        Ok(Self {
            // Consumes the space→channel unshuffled `in_ch·4` map (padding 1 keeps the 3×3 same-size).
            conv: conv(w, &format!("{prefix}.conv"), 1, 1, 1, true)?,
            out_ch,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let unshuffled = pixel_unshuffle(x, 2)?; // [B, in·4, H/2, W/2]
        let down = self.conv.forward(&unshuffled)?;
        let shortcut = channel_average(&unshuffled, self.out_ch)?;
        down + shortcut
    }
}

struct Stage {
    upsample: Option<UpBlock>,
    downsample: Option<DownBlock>,
    blocks: Vec<Block>,
}

// ---------------------------------------------------------------------------------------------------
// Decoder (the spike's GO/NO-GO deliverable)
// ---------------------------------------------------------------------------------------------------

/// The full DC-AE **decoder** (image path, f32). Faithful component port of diffusers `AutoencoderDC`
/// decode (`mit-han-lab/dc-ae-f32c32-sana-1.0`) via the mlx-gen-sana reference.
pub struct DcAeDecoder {
    cfg: DcAeConfig,
    conv_in: Conv2d,
    in_shortcut_repeats: usize,
    stages: Vec<Stage>, // storage order shallow(0)→deep(n-1); decode iterates deep→shallow
    norm_out: Trms2d,
    conv_out: Conv2d,
}

impl DcAeDecoder {
    pub fn from_weights(w: &Weights, cfg: DcAeConfig) -> candle_gen::Result<Self> {
        let n = cfg.num_stages();
        let deepest = cfg.block_out_channels[n - 1] as usize;
        let conv_in = conv(w, "decoder.conv_in", 1, 1, 1, true)?;

        let mut stages = Vec::with_capacity(n);
        for i in 0..n {
            let ch = cfg.block_out_channels[i] as usize;
            // Stages 0..n-1 carry an upsample at storage slot `.0`; the deepest stage does not, so its
            // blocks start at slot `.0`. Block weights live under `decoder.up_blocks.{i}.{slot}`.
            let has_up = i + 1 < n;
            let upsample = if has_up {
                Some(UpBlock::load(
                    w,
                    &format!("decoder.up_blocks.{i}.0"),
                    cfg.block_out_channels[i + 1] as usize,
                    ch,
                )?)
            } else {
                None
            };
            let offset = usize::from(has_up);
            let mut blocks = Vec::new();
            for j in 0..cfg.layers_per_block[i] {
                let prefix = format!("decoder.up_blocks.{i}.{}", j as usize + offset);
                blocks.push(Block::load(w, &prefix, &cfg, cfg.block_types[i], ch)?);
            }
            stages.push(Stage {
                upsample,
                downsample: None,
                blocks,
            });
        }

        Ok(Self {
            in_shortcut_repeats: deepest / cfg.latent_channels as usize,
            conv_in,
            stages,
            norm_out: Trms2d::load(w, "decoder.norm_out", cfg.norm_eps)?,
            conv_out: conv(w, "decoder.conv_out", 1, 1, 1, true)?,
            cfg,
        })
    }

    /// Decode a latent `[B, latent_channels, h, w]` (NCHW, diffusers-native; **already un-scaled** by
    /// the caller) into an image `[B, 3, H=32·h, W=32·w]` (NCHW, f32).
    pub fn decode(&self, latent: &Tensor) -> Result<Tensor> {
        let latent = latent.to_dtype(DType::F32)?;
        let shortcut = repeat_interleave_channel(&latent, self.in_shortcut_repeats)?;
        let mut h = (self.conv_in.forward(&latent)? + shortcut)?;
        for stage in self.stages.iter().rev() {
            if let Some(up) = &stage.upsample {
                h = up.forward(&h)?;
            }
            for block in &stage.blocks {
                h = block.forward(&h)?;
            }
        }
        let h = self.norm_out.forward(&h)?;
        let h = h.relu()?;
        self.conv_out.forward(&h)
    }

    pub fn config(&self) -> &DcAeConfig {
        &self.cfg
    }
}

// ---------------------------------------------------------------------------------------------------
// Encoder (round-trip only — the spike's secondary check)
// ---------------------------------------------------------------------------------------------------

/// A compact **encoder** — the structural mirror of [`DcAeDecoder`] built from the same primitives
/// (`ResBlock` / `EfficientVitBlock` / `DownBlock` ConvPixelUnshuffle). It exists only "as far as the
/// spike needs" for a round-trip reconstruction check; the decoder is the parity deliverable. Reads
/// `encoder.*` keys so the diffusers checkpoint layout would load, but the spike only smoke-tests its
/// composition (shape/finite), since there is no encoder in the mlx reference to parity against.
pub struct DcAeEncoder {
    conv_in: Conv2d,
    stages: Vec<Stage>, // shallow(0)→deep(n-1); encode iterates shallow→deep
    conv_out: Conv2d,
    out_shortcut_group: usize,
}

impl DcAeEncoder {
    pub fn from_weights(w: &Weights, cfg: &DcAeConfig) -> candle_gen::Result<Self> {
        let n = cfg.num_stages();
        let conv_in = conv(w, "encoder.conv_in", 1, 1, 1, true)?;
        let mut stages = Vec::with_capacity(n);
        for i in 0..n {
            let ch = cfg.block_out_channels[i] as usize;
            let mut blocks = Vec::new();
            for j in 0..cfg.layers_per_block[i] {
                let prefix = format!("encoder.down_blocks.{i}.{j}");
                blocks.push(Block::load(w, &prefix, cfg, cfg.block_types[i], ch)?);
            }
            // Stages 0..n-1 carry a downsample at slot `.{layers}`; the deepest does not.
            let downsample = if i + 1 < n {
                Some(DownBlock::load(
                    w,
                    &format!("encoder.down_blocks.{i}.{}", cfg.layers_per_block[i]),
                    cfg.block_out_channels[i + 1] as usize,
                )?)
            } else {
                None
            };
            stages.push(Stage {
                upsample: None,
                downsample,
                blocks,
            });
        }
        let deepest = cfg.block_out_channels[n - 1] as usize;
        Ok(Self {
            conv_in,
            stages,
            conv_out: conv(w, "encoder.conv_out", 1, 1, 1, true)?,
            out_shortcut_group: deepest / cfg.latent_channels as usize,
        })
    }

    /// Encode an image `[B, 3, H, W]` into a latent `[B, latent_channels, H/32, W/32]` (NCHW, f32).
    pub fn encode(&self, image: &Tensor) -> Result<Tensor> {
        let image = image.to_dtype(DType::F32)?;
        let mut h = self.conv_in.forward(&image)?;
        for stage in &self.stages {
            for block in &stage.blocks {
                h = block.forward(&h)?;
            }
            if let Some(down) = &stage.downsample {
                h = down.forward(&h)?;
            }
        }
        // conv_out to latent channels, plus a channel-average shortcut (mirror of the decoder's
        // conv_in repeat_interleave shortcut).
        let shortcut = channel_average(&h, h.dim(1)? / self.out_shortcut_group)?;
        self.conv_out.forward(&h)?.broadcast_add(&shortcut)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Device;
    use std::collections::HashMap;

    /// A deterministic pseudo-random tensor (linear-congruential fill) — reproducible on any backend,
    /// no rand dep.
    fn det(shape: &[usize], seed: u64, dev: &Device) -> Tensor {
        let n: usize = shape.iter().product();
        let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // top bits → [-1, 1)
            let u = ((s >> 33) as f64) / ((1u64 << 31) as f64) - 1.0;
            v.push(u as f32);
        }
        Tensor::from_vec(v, shape, dev).unwrap()
    }

    #[test]
    fn pixel_shuffle_unshuffle_roundtrip() {
        let dev = Device::Cpu;
        let x = det(&[2, 12, 4, 6], 7, &dev); // C=12 = 3·2²
        let up = pixel_shuffle(&x, 2).unwrap();
        assert_eq!(up.dims(), &[2, 3, 8, 12]);
        let back = pixel_unshuffle(&up, 2).unwrap();
        assert_eq!(back.dims(), x.dims());
        let d = (&x - &back)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            d < 1e-6,
            "pixel_shuffle∘pixel_unshuffle must be identity; max|Δ|={d}"
        );
    }

    #[test]
    fn repeat_interleave_then_average_is_identity() {
        let dev = Device::Cpu;
        let x = det(&[1, 4, 2, 2], 11, &dev);
        let rep = repeat_interleave_channel(&x, 3).unwrap();
        assert_eq!(rep.dims(), &[1, 12, 2, 2]);
        // Exact interleave: input channel c → output channels [3c, 3c+1, 3c+2], all equal x[c].
        // Row-major flat index for (channel, h, w) in a [1,C,2,2] tensor is channel*4 + h*2 + w.
        let xv = x.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let rv = rep.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for copy in 0..3usize {
            // input channel 1, pixel (0,0) = xv[4]; its copies land in output channels 3,4,5.
            assert_eq!(rv[(3 + copy) * 4], xv[4]);
        }
        let avg = channel_average(&rep, 4).unwrap();
        let d = (&x - &avg)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            d < 1e-6,
            "channel_average∘repeat_interleave must be identity; max|Δ|={d}"
        );
    }

    #[test]
    fn trms2d_matches_manual() {
        // Build a Trms2d with weight=1, bias=0 and compare to a hand-computed channel-RMS-norm.
        let dev = Device::Cpu;
        let c = 5usize;
        let norm = Trms2d {
            weight: Tensor::ones((1, c, 1, 1), DType::F32, &dev).unwrap(),
            bias: Tensor::zeros((1, c, 1, 1), DType::F32, &dev).unwrap(),
            eps: 1e-5,
        };
        let x = det(&[1, c, 1, 1], 3, &dev);
        let got = norm
            .forward(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let xv = x.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let ms: f32 = xv.iter().map(|v| v * v).sum::<f32>() / c as f32;
        let denom = (ms + 1e-5f32).sqrt();
        for (g, v) in got.iter().zip(xv.iter()) {
            assert!(
                (g - v / denom).abs() < 1e-5,
                "trms2d channel norm mismatch: {g} vs {}",
                v / denom
            );
        }
    }

    #[test]
    fn relu_linear_attention_matches_quadratic() {
        // The O(N) associative kernel must equal the explicit O(N²) softmax-free ReLU attention
        // (materializing the [N,N] score matrix) — an independent computation path proving the
        // numerator/denominator split algebra of the "shared hard primitive".
        let dev = Device::Cpu;
        let (b, g, hd, n) = (1usize, 2usize, 3usize, 5usize);
        let q = det(&[b, g, hd, n], 21, &dev).relu().unwrap(); // φ(q) ≥ 0
        let k = det(&[b, g, hd, n], 22, &dev).relu().unwrap(); // φ(k) ≥ 0
        let v = det(&[b, g, hd, n], 23, &dev);
        let eps = 1e-15f64;

        let fast = relu_linear_attention(&q, &k, &v, eps).unwrap();

        // Explicit: Score[i,m] = φ(q_i)·φ(k_m) = qᵀk [N,N]; out = (v·Scoreᵀ) / (Σ_m Score[i,m] + eps).
        let score = q
            .transpose(D::Minus1, D::Minus2)
            .unwrap()
            .matmul(&k)
            .unwrap(); // [b,g,N,N]
        let num = v
            .matmul(
                &score
                    .transpose(D::Minus1, D::Minus2)
                    .unwrap()
                    .contiguous()
                    .unwrap(),
            )
            .unwrap(); // [b,g,hd,N]
        let den = score
            .sum_keepdim(D::Minus1)
            .unwrap()
            .transpose(D::Minus1, D::Minus2)
            .unwrap(); // [b,g,1,N]
        let slow = num.broadcast_div(&(den + eps).unwrap()).unwrap();

        let d = (&fast - &slow)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            d < 1e-5,
            "O(N) linear attention must equal the explicit quadratic form; max|Δ|={d}"
        );
    }

    #[test]
    fn decoder_smoke_tiny_cpu() {
        // A random-weight forward through every primitive on the tiny config: finite, right shape,
        // non-degenerate (not a constant field). This is the CPU analogue of the GPU 1024² smoke.
        let dev = Device::Cpu;
        let cfg = DcAeConfig::tiny_test();
        let w = synthetic_weights(&cfg, /*decoder*/ true, /*encoder*/ false, &dev);
        let dec = DcAeDecoder::from_weights(&w, cfg.clone()).unwrap();
        let comp = cfg.spatial_compression() as usize; // ×8 for tiny
        let latent = det(&[1, cfg.latent_channels as usize, 3, 3], 99, &dev);
        let img = dec.decode(&latent).unwrap();
        assert_eq!(img.dims(), &[1, 3, 3 * comp, 3 * comp]);
        let flat = img.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(flat.iter().all(|v| v.is_finite()), "non-finite decode");
        let (lo, hi) = flat
            .iter()
            .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
        assert!(
            hi - lo > 1e-4,
            "decode output is constant — graph degenerate: [{lo}, {hi}]"
        );
    }

    #[test]
    fn encode_decode_roundtrip_shape_tiny_cpu() {
        // Round-trip geometry: encode an image to the latent grid, decode back to the image grid.
        // Proves the encoder/decoder down/up-sampling are exact geometric inverses and the shared
        // primitives compose in both directions (shape + finite; no parity, no reference encoder).
        let dev = Device::Cpu;
        let cfg = DcAeConfig::tiny_test();
        let w = synthetic_weights(&cfg, true, true, &dev);
        let enc = DcAeEncoder::from_weights(&w, &cfg).unwrap();
        let dec = DcAeDecoder::from_weights(&w, cfg.clone()).unwrap();
        let comp = cfg.spatial_compression() as usize;
        let img = det(&[1, 3, 2 * comp, 2 * comp], 5, &dev); // → 2×2 latent grid
        let latent = enc.encode(&img).unwrap();
        assert_eq!(latent.dims(), &[1, cfg.latent_channels as usize, 2, 2]);
        let recon = dec.decode(&latent).unwrap();
        assert_eq!(recon.dims(), img.dims());
        assert!(
            recon
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .all(|v| v.is_finite()),
            "non-finite round-trip reconstruction"
        );
    }

    /// The **GPU 1024² decode smoke + f32 memory-profile determination** — the spike's required
    /// single-pass-vs-tiled deliverable. Builds the FULL `sana_f32c32` decoder (real layer shapes ⇒
    /// representative ~weight+activation footprint) with synthetic weights on CUDA sm_120, decodes a
    /// `[1,32,32,32]` latent to a `[1,3,1024,1024]` image single-pass, asserts finite/non-degenerate,
    /// and prints the peak device VRAM so the run records whether f32 fits without the spatial-tiled
    /// (blend_v/blend_h) fallback. cuda-gated; needs `--release` (and an idle GPU for a clean baseline).
    ///
    /// Run:
    ///   cargo test -p candle-gen-sana --lib --features cuda --release -- --ignored --nocapture \
    ///       gpu_decode_1024_memory_profile
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "GPU 1024² f32 memory-profile smoke — run on CUDA sm_120 with --release"]
    fn gpu_decode_1024_memory_profile() {
        use candle_gen::testkit::VramProbe;
        let dev = Device::new_cuda(0).expect("cuda device");
        let cfg = DcAeConfig::sana_f32c32();

        let mut probe = VramProbe::start(0);
        let load = probe.phase();
        let w = synthetic_weights(&cfg, /*decoder*/ true, /*encoder*/ false, &dev);
        let dec = DcAeDecoder::from_weights(&w, cfg.clone()).unwrap();
        probe.end_load(load);

        let latent = det(&[1, cfg.latent_channels as usize, 32, 32], 1234, &dev);
        let run = probe.phase();
        let img = dec.decode(&latent).unwrap();
        let _ = img.sum_all().unwrap().to_scalar::<f32>().unwrap(); // force full eval before sampling
        probe.end_gen(run);

        assert_eq!(img.dims(), &[1, 3, 1024, 1024], "1024² decode shape");
        let flat = img
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(
            flat.iter().all(|v| v.is_finite()),
            "non-finite 1024² decode"
        );
        let (lo, hi) = flat
            .iter()
            .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
        assert!(
            hi - lo > 1e-3,
            "1024² decode is constant — graph degenerate: [{lo}, {hi}]"
        );
        println!(
            "DC-AE f32 1024² single-pass decode OK on CUDA: range=[{lo:.4}, {hi:.4}]  VRAM: {}",
            probe.report()
        );
    }

    // -- synthetic weight-map builders (deterministic; cover every key `from_weights` requires) -----

    struct Emit<'a> {
        map: HashMap<String, Tensor>,
        seed: u64,
        dev: &'a Device,
    }
    impl<'a> Emit<'a> {
        fn new(dev: &'a Device) -> Self {
            Self {
                map: HashMap::new(),
                seed: 1,
                dev,
            }
        }
        fn t(&mut self, shape: &[usize], key: String) {
            self.seed += 1;
            self.map.insert(key, det(shape, self.seed, self.dev));
        }
        fn conv(&mut self, p: &str, o: usize, i: usize, k: usize, bias: bool) {
            self.t(&[o, i, k, k], format!("{p}.weight"));
            if bias {
                self.t(&[o], format!("{p}.bias"));
            }
        }
        fn norm(&mut self, p: &str, c: usize) {
            self.t(&[c], format!("{p}.weight"));
            self.t(&[c], format!("{p}.bias"));
        }
        fn block(&mut self, cfg: &DcAeConfig, prefix: &str, ty: BlockType, ch: usize) {
            let hd = cfg.attention_head_dim as usize;
            match ty {
                BlockType::Res => {
                    self.conv(&format!("{prefix}.conv1"), ch, ch, 3, true);
                    self.conv(&format!("{prefix}.conv2"), ch, ch, 3, false);
                    self.norm(&format!("{prefix}.norm"), ch);
                }
                BlockType::EfficientVit => {
                    let heads = ch / hd;
                    let scales = cfg.qkv_multiscales.len();
                    // to_q/k/v: [out,in] 2D channel-linears; to_out: inner·(1+scales) → ch.
                    for name in ["to_q", "to_k", "to_v"] {
                        self.t(&[ch, ch], format!("{prefix}.attn.{name}.weight"));
                    }
                    self.t(
                        &[ch, ch * (1 + scales)],
                        format!("{prefix}.attn.to_out.weight"),
                    );
                    for (i, k) in cfg.qkv_multiscales.iter().enumerate() {
                        let ch3 = 3 * ch;
                        // proj_in: depthwise (groups=3ch) → [3ch,1,k,k]; proj_out: grouped 1×1 (groups=3·heads).
                        self.t(
                            &[ch3, 1, *k as usize, *k as usize],
                            format!("{prefix}.attn.to_qkv_multiscale.{i}.proj_in.weight"),
                        );
                        self.t(
                            &[ch3, ch3 / (3 * heads), 1, 1],
                            format!("{prefix}.attn.to_qkv_multiscale.{i}.proj_out.weight"),
                        );
                    }
                    self.norm(&format!("{prefix}.attn.norm_out"), ch);
                    let hidden = 4 * ch;
                    self.conv(
                        &format!("{prefix}.conv_out.conv_inverted"),
                        2 * hidden,
                        ch,
                        1,
                        true,
                    );
                    self.t(
                        &[2 * hidden, 1, 3, 3],
                        format!("{prefix}.conv_out.conv_depth.weight"),
                    );
                    self.t(&[2 * hidden], format!("{prefix}.conv_out.conv_depth.bias"));
                    self.conv(
                        &format!("{prefix}.conv_out.conv_point"),
                        ch,
                        hidden,
                        1,
                        false,
                    );
                    self.norm(&format!("{prefix}.conv_out.norm"), ch);
                }
            }
        }
    }

    /// Build a synthetic (deterministic pseudo-random) weight map covering every key the decoder and/or
    /// encoder require for `cfg`, then load it through the real [`Weights`] path (`from_map`). Lets the
    /// CPU smokes exercise the true load + forward without the ~1.25 GB real checkpoint.
    fn synthetic_weights(cfg: &DcAeConfig, decoder: bool, encoder: bool, dev: &Device) -> Weights {
        let mut e = Emit::new(dev);
        let n = cfg.num_stages();
        let lat = cfg.latent_channels as usize;

        if decoder {
            let deepest = cfg.block_out_channels[n - 1] as usize;
            e.conv("decoder.conv_in", deepest, lat, 3, true);
            for i in 0..n {
                let ch = cfg.block_out_channels[i] as usize;
                let has_up = i + 1 < n;
                let offset = usize::from(has_up);
                if has_up {
                    let in_ch = cfg.block_out_channels[i + 1] as usize;
                    e.conv(&format!("decoder.up_blocks.{i}.0.conv"), ch, in_ch, 3, true);
                }
                for j in 0..cfg.layers_per_block[i] {
                    let prefix = format!("decoder.up_blocks.{i}.{}", j as usize + offset);
                    e.block(cfg, &prefix, cfg.block_types[i], ch);
                }
            }
            e.norm("decoder.norm_out", cfg.block_out_channels[0] as usize);
            e.conv(
                "decoder.conv_out",
                3,
                cfg.block_out_channels[0] as usize,
                3,
                true,
            );
        }
        if encoder {
            e.conv(
                "encoder.conv_in",
                cfg.block_out_channels[0] as usize,
                3,
                3,
                true,
            );
            for i in 0..n {
                let ch = cfg.block_out_channels[i] as usize;
                for j in 0..cfg.layers_per_block[i] {
                    e.block(
                        cfg,
                        &format!("encoder.down_blocks.{i}.{j}"),
                        cfg.block_types[i],
                        ch,
                    );
                }
                if i + 1 < n {
                    let out_ch = cfg.block_out_channels[i + 1] as usize;
                    // DownBlock conv consumes the space→channel unshuffled ch·4 map → out_ch.
                    e.conv(
                        &format!("encoder.down_blocks.{i}.{}.conv", cfg.layers_per_block[i]),
                        out_ch,
                        ch * 4,
                        3,
                        true,
                    );
                }
            }
            e.conv(
                "encoder.conv_out",
                lat,
                cfg.block_out_channels[n - 1] as usize,
                3,
                true,
            );
        }

        Weights::from_map(e.map)
    }
}
