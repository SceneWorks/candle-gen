//! BiSeNet face parsing (PuLID-FLUX `face_features_image`, sc-5492) — the candle (Windows/CUDA) twin
//! of `mlx-gen-face`'s `bisenet.rs`, a port of facexlib's `parsing_bisenet.pth`.
//!
//! PuLID whitens non-face regions before the EVA-CLIP crop: `parse(x)[0]` → 19-class argmax mask →
//! background labels `[0,16,18,7,8,9,14,15]` become white, the rest grayscale
//! (`pipeline_flux.py:167-177`). This replaces facexlib's torch BiSeNet — the last torch holdout on
//! the candle PuLID lane.
//!
//! ## Architecture (facexlib `parsing/{bisenet,resnet}.py`, num_class = 19)
//! - **ResNet18 backbone**: `conv1(7×7,s2)` → maxpool(3×3,s2,p1) → 4 stages of 2 `BasicBlock`s
//!   (stages 2/3/4 block-0 stride-2 + 1×1 downsample) → `feat8`/`feat16`/`feat32` (1/8,1/16,1/32).
//! - **ContextPath**: `ARM(256→128)` + `ARM(512→128)` (ConvBNReLU → 1×1 conv_atten → sigmoid → mul),
//!   a global-avg `conv_avg(512→128)`, nearest upsamples, and `conv_head16/32`.
//! - **FeatureFusionModule(256→256)**: channel-concat `feat_res8`+`feat_cp8`, ConvBNReLU, then an
//!   SE-style attention (`conv1→relu→conv2→sigmoid`, `feat*atten + feat`).
//! - **conv_out head**: ConvBNReLU(256→256) → 1×1 → 19 logits @ 64², then **bilinear upsample to
//!   512² with `align_corners=True`** (hand-rolled exactly) → argmax. (`conv_out16`/`conv_out32` are
//!   training-only aux heads; PuLID uses `[0]`, so they're omitted.)
//!
//! Every conv is bias-less + BN; BNs are folded into the convs at conversion. The mask is a coarse
//! argmax → tolerant; parity is mask IoU ≈ 1.0. Input: **NCHW** `[B,3,512,512]`, `(rgb/255 - mean) /
//! std` with ImageNet mean/std. f32, OIHW weights (transposed from the file's OHWI at load — candle is
//! NCHW where MLX is NHWC).

use candle_gen::candle_core::{Device, Tensor};
use candle_gen::{CandleError, Result};
use candle_nn::ops::sigmoid;

use crate::common::{Conv, ConvW, Weights};

/// PuLID background labels → whitened in the `face_features_image`.
pub const BG_LABELS: [u32; 8] = [0, 16, 18, 7, 8, 9, 14, 15];
/// ImageNet normalization (the facexlib parse-net preprocessing).
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Global average pool over the NCHW spatial axes → `[B,C,1,1]`.
fn global_avg(x: &Tensor) -> Result<Tensor> {
    Ok(x.mean_keepdim((2, 3))?)
}

/// `3×3` stride-2 pad-1 max pool over NCHW. Hand-rolled via strided `index_select` over a zero-padded
/// map (overlapping windows; the input is ReLU'd ≥ 0 so 0-pad ≡ PyTorch's -inf pad — every output
/// window contains ≥ 1 real pixel) — basic ops only, GPU-safe on sm_120 (no maxpool kernel needed).
fn maxpool_3x3_s2(x: &Tensor) -> Result<Tensor> {
    let (_b, _c, h, w) = x.dims4()?;
    let oh = (h - 1) / 2 + 1;
    let ow = (w - 1) / 2 + 1;
    let p = x.pad_with_zeros(2, 1, 1)?.pad_with_zeros(3, 1, 1)?; // [B,C,H+2,W+2]
    let dev = x.device();
    let mut acc: Option<Tensor> = None;
    for dy in 0..3usize {
        let rows: Vec<u32> = (0..oh).map(|i| (dy + 2 * i) as u32).collect();
        let pr = p.index_select(&Tensor::from_vec(rows, oh, dev)?, 2)?; // [B,C,oh,W+2]
        for dx in 0..3usize {
            let cols: Vec<u32> = (0..ow).map(|j| (dx + 2 * j) as u32).collect();
            let win = pr.index_select(&Tensor::from_vec(cols, ow, dev)?, 3)?; // [B,C,oh,ow]
            acc = Some(match acc {
                None => win,
                Some(a) => a.maximum(&win)?,
            });
        }
    }
    acc.ok_or_else(|| CandleError::Msg("face bisenet maxpool: empty pooling window".into()))
}

/// Per-axis `align_corners=True` linear interpolation matrix `[out_n, in_n]`: row `i` maps output pixel
/// `i` to source `i·(in-1)/(out-1)` with bilinear weights `(1-f, f)` on `(floor, floor+1)`.
fn interp_matrix(in_n: usize, out_n: usize, dev: &Device) -> Result<Tensor> {
    let mut m = vec![0f32; out_n * in_n];
    let denom = (out_n.max(2) - 1) as f32; // (out_n - 1).max(1)
    for i in 0..out_n {
        let src = i as f32 * (in_n - 1) as f32 / denom;
        let x0 = src.floor() as usize;
        let x1 = (x0 + 1).min(in_n - 1);
        let f = src - x0 as f32;
        m[i * in_n + x0] += 1.0 - f;
        m[i * in_n + x1] += f; // += so x0 == x1 (last row) sums to 1
    }
    Ok(Tensor::from_vec(m, (out_n, in_n), dev)?)
}

/// Bilinear upsample of NCHW `x` to `out_h × out_w` with `align_corners=True`, applied as two separable
/// matmuls (`Wy · x` over rows, `x · Wxᵀ` over cols). Bit-faithful to `F.interpolate(mode=bilinear,
/// align_corners=True)`.
fn upsample_bilinear_ac(x: &Tensor, out_h: usize, out_w: usize) -> Result<Tensor> {
    let (b, c, h, w) = x.dims4()?;
    let dev = x.device();
    let wy = interp_matrix(h, out_h, dev)?; // [out_h, h]
    let wx = interp_matrix(w, out_w, dev)?; // [out_w, w]
    let x2 = x.contiguous()?.reshape((b * c, h, w))?;
    // rows: [b*c,out_h,h] · [b*c,h,w] → [b*c,out_h,w]
    let wy_b = wy
        .unsqueeze(0)?
        .broadcast_as((b * c, out_h, h))?
        .contiguous()?;
    let mid = wy_b.matmul(&x2)?;
    // cols: [b*c,out_h,w] · [b*c,w,out_w] → [b*c,out_h,out_w]
    let wxt = wx
        .t()?
        .unsqueeze(0)?
        .broadcast_as((b * c, w, out_w))?
        .contiguous()?;
    let out = mid.matmul(&wxt)?;
    Ok(out.reshape((b, c, out_h, out_w))?)
}

/// ResNet18 `BasicBlock`: `conv1(stride)→relu→conv2 (+ downsample) → relu`.
struct BasicBlock {
    conv1: Conv,
    conv2: Conv,
    downsample: Option<Conv>,
    stride: usize,
}
impl BasicBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let r = self.conv1.forward_relu(x, self.stride, 1)?;
        let r = self.conv2.forward(&r, 1, 1)?;
        let shortcut = match &self.downsample {
            Some(ds) => ds.forward(x, self.stride, 0)?,
            None => x.clone(),
        };
        Ok((r + shortcut)?.relu()?)
    }
}

/// AttentionRefinementModule: `ConvBNReLU → global-avg → 1×1 conv_atten → sigmoid → mul`.
struct Arm {
    conv: Conv,
    conv_atten: Conv, // bn_atten folded in
}
impl Arm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let feat = self.conv.forward_relu(x, 1, 1)?;
        let atten = self.conv_atten.forward(&global_avg(&feat)?, 1, 0)?;
        let atten = sigmoid(&atten)?;
        Ok(feat.broadcast_mul(&atten)?)
    }
}

/// BiSeNet (19-class face parsing).
pub struct BiSeNet {
    // ResNet18 backbone
    conv1: Conv,
    layers: Vec<Vec<BasicBlock>>,
    // ContextPath
    arm16: Arm,
    arm32: Arm,
    conv_head32: Conv,
    conv_head16: Conv,
    conv_avg: Conv,
    // FeatureFusionModule
    ffm_convblk: Conv,
    ffm_conv1: ConvW,
    ffm_conv2: ConvW,
    // output head
    conv_out_conv: Conv,
    conv_out_out: ConvW,
}

const STAGES: [(i32, usize); 4] = [(64, 1), (128, 2), (256, 2), (512, 2)]; // (out_chan, block0 stride)

impl BiSeNet {
    /// Load from the converted `bisenet_parsing.safetensors` (the same file the MLX sibling loads).
    pub(crate) fn from_weights(w: &Weights) -> Result<Self> {
        let mut layers = Vec::with_capacity(4);
        for (li, &(_oc, stride)) in STAGES.iter().enumerate() {
            let l = li + 1;
            let mut blocks = Vec::with_capacity(2);
            for b in 0..2 {
                let p = format!("resnet.layer{l}.{b}");
                let (s, ds) = if b == 0 {
                    (
                        stride,
                        if l > 1 {
                            Some(Conv::load(w, &format!("{p}.downsample"))?)
                        } else {
                            None
                        },
                    )
                } else {
                    (1, None)
                };
                blocks.push(BasicBlock {
                    conv1: Conv::load(w, &format!("{p}.conv1"))?,
                    conv2: Conv::load(w, &format!("{p}.conv2"))?,
                    downsample: ds,
                    stride: s,
                });
            }
            layers.push(blocks);
        }
        Ok(Self {
            conv1: Conv::load(w, "resnet.conv1")?,
            layers,
            arm16: Arm {
                conv: Conv::load(w, "arm16.conv")?,
                conv_atten: Conv::load(w, "arm16.conv_atten")?,
            },
            arm32: Arm {
                conv: Conv::load(w, "arm32.conv")?,
                conv_atten: Conv::load(w, "arm32.conv_atten")?,
            },
            conv_head32: Conv::load(w, "conv_head32")?,
            conv_head16: Conv::load(w, "conv_head16")?,
            conv_avg: Conv::load(w, "conv_avg")?,
            ffm_convblk: Conv::load(w, "ffm.convblk")?,
            ffm_conv1: ConvW::load(w, "ffm.conv1")?,
            ffm_conv2: ConvW::load(w, "ffm.conv2")?,
            conv_out_conv: Conv::load(w, "conv_out.conv")?,
            conv_out_out: ConvW::load(w, "conv_out.conv_out")?,
        })
    }

    fn run_layer(&self, idx: usize, x: &Tensor) -> Result<Tensor> {
        let mut h = x.clone();
        for blk in &self.layers[idx] {
            h = blk.forward(&h)?;
        }
        Ok(h)
    }

    /// ResNet18 backbone → `(feat8, feat16, feat32)`.
    fn resnet(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let x = self.conv1.forward_relu(x, 2, 3)?;
        let x = maxpool_3x3_s2(&x)?;
        let x = self.run_layer(0, &x)?; // layer1
        let feat8 = self.run_layer(1, &x)?; // layer2 (1/8)
        let feat16 = self.run_layer(2, &feat8)?; // layer3 (1/16)
        let feat32 = self.run_layer(3, &feat16)?; // layer4 (1/32)
        Ok((feat8, feat16, feat32))
    }

    /// ContextPath → `(feat_res8, feat_cp8)` (the `feat_cp16` aux output is unused by `parse[0]`).
    fn context_path(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let (feat8, feat16, feat32) = self.resnet(x)?;
        let s32 = feat32.dim(2)?;
        let s16 = feat16.dim(2)?;
        let s8 = feat8.dim(2)?;

        let avg = self.conv_avg.forward_relu(&global_avg(&feat32)?, 1, 0)?; // [B,128,1,1]
        let avg_up = avg.upsample_nearest2d(s32, s32)?; // → feat32 spatial

        let feat32_sum = (self.arm32.forward(&feat32)? + avg_up)?;
        let feat32_up = feat32_sum.upsample_nearest2d(s16, s16)?;
        let feat_cp16 = self.conv_head32.forward_relu(&feat32_up, 1, 1)?;

        let feat16_sum = (self.arm16.forward(&feat16)? + feat_cp16)?;
        let feat16_up = feat16_sum.upsample_nearest2d(s8, s8)?;
        let feat_cp8 = self.conv_head16.forward_relu(&feat16_up, 1, 1)?;

        Ok((feat8, feat_cp8))
    }

    /// FeatureFusionModule(fsp, fcp).
    fn ffm(&self, fsp: &Tensor, fcp: &Tensor) -> Result<Tensor> {
        let fcat = Tensor::cat(&[fsp, fcp], 1)?; // channel-concat (NCHW axis 1)
        let feat = self.ffm_convblk.forward_relu(&fcat, 1, 0)?;
        let atten = self.ffm_conv1.forward(&global_avg(&feat)?)?.relu()?;
        let atten = sigmoid(&self.ffm_conv2.forward(&atten)?)?;
        let feat_atten = feat.broadcast_mul(&atten)?;
        Ok((feat_atten + feat)?)
    }

    /// Parse → 19-class logits, NCHW `[B, 19, H, W]` (bilinear-upsampled to the input size).
    pub fn parse_logits(&self, x: &Tensor) -> Result<Tensor> {
        let (_b, _c, h, w) = x.dims4()?;
        let (feat_res8, feat_cp8) = self.context_path(x)?;
        let feat_fuse = self.ffm(&feat_res8, &feat_cp8)?;
        let out = self.conv_out_conv.forward_relu(&feat_fuse, 1, 1)?;
        let out = self.conv_out_out.forward(&out)?; // [B,19,64,64]
        upsample_bilinear_ac(&out, h, w)
    }

    /// Parse → argmax class mask, `[B, H, W]` (`u32`).
    pub fn parse_mask(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.parse_logits(x)?.argmax(1)?)
    }
}

/// Normalize an RGB NCHW `[B,3,H,W]` `[0,1]` image to the BiSeNet parse-net input (ImageNet mean/std).
pub fn to_parse_input(rgb01: &Tensor) -> Result<Tensor> {
    let dev = rgb01.device();
    let mean = Tensor::from_vec(MEAN.to_vec(), (1, 3, 1, 1), dev)?;
    let std = Tensor::from_vec(STD.to_vec(), (1, 3, 1, 1), dev)?;
    Ok(rgb01.broadcast_sub(&mean)?.broadcast_div(&std)?)
}

/// PuLID `face_features_image`: `where(mask ∈ BG_LABELS, white(1.0), gray(rgb01))`, NCHW `[B,3,H,W]`.
/// `rgb01` is the un-normalized `[0,1]` aligned crop (NCHW); `mask` is [`BiSeNet::parse_mask`] output
/// (`[B,H,W]`). The bg-whiten + luma grayscale is pure host arithmetic (read both back, recombine).
pub fn face_features_image(rgb01: &Tensor, mask: &Tensor) -> Result<Tensor> {
    let (b, _c, h, w) = rgb01.dims4()?;
    // RGB as host HWC-interleaved f32; mask as host labels.
    let rgb = rgb01
        .to_device(&Device::Cpu)?
        .permute((0, 2, 3, 1))? // [B,H,W,3]
        .contiguous()?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let m = mask
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<u32>()?;
    let n = b * h * w;
    let mut out = vec![0f32; n * 3];
    for i in 0..n {
        let v = if BG_LABELS.contains(&m[i]) {
            1.0
        } else {
            0.299 * rgb[i * 3] + 0.587 * rgb[i * 3 + 1] + 0.114 * rgb[i * 3 + 2]
        };
        out[i * 3] = v;
        out[i * 3 + 1] = v;
        out[i * 3 + 2] = v;
    }
    // Back to NCHW on the source device.
    let hwc = Tensor::from_vec(out, (b, h, w, 3), &Device::Cpu)?;
    Ok(hwc
        .permute((0, 3, 1, 2))?
        .contiguous()?
        .to_device(rgb01.device())?)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `interp_matrix` rows are convex (weights sum to 1) and align the corners: row 0 → source 0,
    /// last row → source in-1.
    #[test]
    fn interp_matrix_align_corners_rows_sum_to_one() {
        let dev = Device::Cpu;
        let (in_n, out_n) = (4usize, 8usize);
        let m = interp_matrix(in_n, out_n, &dev).unwrap();
        let rows = m.to_vec2::<f32>().unwrap();
        for (i, row) in rows.iter().enumerate() {
            let s: f32 = row.iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "row {i} sums to {s}");
        }
        // align_corners: first output maps fully to source 0, last to source in-1.
        assert!((rows[0][0] - 1.0).abs() < 1e-6);
        assert!((rows[out_n - 1][in_n - 1] - 1.0).abs() < 1e-6);
    }

    /// `maxpool_3x3_s2` halves a ramp's spatial dims to `(h-1)/2+1` and (on a ≥0 input) returns the
    /// window maxima — basic-ops path, no maxpool kernel.
    #[test]
    fn maxpool_3x3_s2_shape_and_max() {
        let dev = Device::Cpu;
        // [1,1,4,4] ascending 0..15 → pooled [1,1,2,2]. With pad-1, the top-left output window spans
        // padded rows/cols {0,1,2} (i.e. real rows {0,1} cols {0,1} + a zero border) → max = 5; the
        // bottom-right window spans real rows/cols {2,3} → max = 15.
        let x = Tensor::arange(0f32, 16f32, &dev)
            .unwrap()
            .reshape((1, 1, 4, 4))
            .unwrap();
        let y = maxpool_3x3_s2(&x).unwrap();
        assert_eq!(y.dims(), &[1, 1, 2, 2]);
        let v = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(v[0], 5.0);
        assert_eq!(v[3], 15.0);
    }

    /// `upsample_bilinear_ac` to the same size is an identity (each interp matrix is the identity), so
    /// a known map is preserved bit-for-bit.
    #[test]
    fn upsample_bilinear_ac_same_size_identity() {
        let dev = Device::Cpu;
        let x = Tensor::arange(0f32, 12f32, &dev)
            .unwrap()
            .reshape((1, 1, 3, 4))
            .unwrap();
        let y = upsample_bilinear_ac(&x, 3, 4).unwrap();
        assert_eq!(y.dims(), &[1, 1, 3, 4]);
        let (a, b) = (
            x.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            y.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        );
        for (p, q) in a.iter().zip(&b) {
            assert!((p - q).abs() < 1e-4);
        }
    }

    /// `face_features_image` whitens background labels (1.0 everywhere) and grayscales foreground; the
    /// three output channels are equal (gray).
    #[test]
    fn face_features_whitens_bg_and_grays_fg() {
        let dev = Device::Cpu;
        // 1×1 image, foreground (label 1 ∉ BG) red → luma 0.299.
        let rgb = Tensor::from_vec(vec![1f32, 0.0, 0.0], (1, 3, 1, 1), &dev).unwrap();
        let fg_mask = Tensor::from_vec(vec![1u32], (1, 1, 1), &dev).unwrap();
        let fg = face_features_image(&rgb, &fg_mask)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!((fg[0] - 0.299).abs() < 1e-5 && fg[0] == fg[1] && fg[1] == fg[2]);
        // Background label 0 ∈ BG → white.
        let bg_mask = Tensor::from_vec(vec![0u32], (1, 1, 1), &dev).unwrap();
        let bg = face_features_image(&rgb, &bg_mask)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(bg, vec![1.0, 1.0, 1.0]);
    }
}
