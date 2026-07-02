//! MLX-packed → GGML repack seam (sc-9085 spike; grows into the shared packed-load module,
//! sc-9086, epic 9083).
//!
//! The hosted quant tiers (epic 8506, e.g. `SceneWorks/z-image-turbo-mlx`) store each quantized
//! Linear as the MLX packed triple `{base}.weight` (u32 codes) + `{base}.scales` + `{base}.biases`:
//! **group-wise affine** quantization along the input dimension at group size 64, dequantized as
//! `w = scale · q + bias` with the codes packed LSB-first (4-bit: element *k* of a row is nibble
//! `k % 8` of u32 `k / 8`; 8-bit: byte `k % 4` of u32 `k / 4`).
//!
//! GGML's `Q4_1` is the **same affine form** over 32-element blocks (`block_q4_1` = f16 `d` +
//! f16 `m` + 16 nibble bytes; element `j` in the low nibble of `qs[j]`, element `j + 16` in the
//! high nibble — `w = d · q + m`). One MLX group therefore splits **losslessly** into two `Q4_1`
//! blocks sharing `d = scale`, `m = bias`: [`repack_mlx_q4_to_q4_1`] is a pure nibble permutation
//! plus a bf16 → f16 cast of the per-group scale/bias (exact whenever the bf16 value's exponent is
//! in f16 range — its 7-bit mantissa always fits f16's 10; the real-weight spike test censuses the
//! exceptions). The repacked [`QTensor`] feeds the existing `QLinear` dequant-on-forward machinery
//! (sc-7702: the weight is dequantized to a dense matmul; the int8 activation fast path stays off).
//!
//! 8-bit has no affine GGML container (`Q8_0`/`Q8_1` are symmetric, no bias/min), so the MLX Q8
//! tier cannot be repacked losslessly: [`dequant_mlx_q8`] materializes the exact MLX grid values
//! and the caller re-quantizes to `Q8_0` (`QTensor::quantize`), a second 8-bit rounding the spike
//! measured at 0.56 % mean / 0.87 % worst relative RMS on the real z-image Q8 tier (~10× below
//! Q4's inherent error) — the accepted Q8 path per the sc-9085 decision; an exact affine custom
//! dequant path stays an option if a Q8 A/B ever shows the gap.

use std::borrow::Cow;

use candle_core::quantized::{GgmlDType, QStorage, QTensor};
use candle_core::{DType, Device, Result, Tensor};
use half::f16;

/// MLX's default (and the hosted tiers' only) quantization group size along the input dim.
pub const MLX_GROUP_SIZE: usize = 64;

/// GGML `Q4_1` block: 32 elements in 20 bytes (f16 `d`, f16 `m`, 16 nibble bytes).
const Q4_1_BLOCK: usize = 32;
const Q4_1_BLOCK_BYTES: usize = 20;

/// Derive the quant bit-width from an MLX packed pair's shapes (the mlx-gen `packed_bits`
/// convention): `scales` is `[out, in/64]` ⇒ `in = cols · 64`; the u32-packed `weight` is
/// `[out, in·bits/32]` ⇒ `bits = wq_cols · 32 / in`. Exact for group-aligned Q4/Q8 packs, so the
/// tiers ship no bit-width manifest.
pub fn mlx_packed_bits(wq_cols: usize, scales_cols: usize) -> usize {
    let in_dim = scales_cols * MLX_GROUP_SIZE;
    wq_cols * 32 / in_dim
}

/// Whether an f32 value survives the f32 → f16 → f32 round-trip exactly — the only lossy step the
/// Q4 repack can have (bf16 scales/biases whose exponent falls outside f16's range). Used by the
/// spike census; production repack proceeds regardless (the deviation is one f16 ulp of a scale).
pub fn f16_exact(x: f32) -> bool {
    f16::from_f32(x).to_f32() == x
}

/// The exact affine grid values an MLX pack represents, computed the way candle's `Q4_1`/CPU
/// dequant computes them (`f32(scale) · q + f32(bias)` per element, f32 accumulate): the repack's
/// loss-free reference. `codes` are the unpacked per-element quant codes of one row-major
/// `[out, in]` tensor; `scales`/`biases` are the per-group f32 values.
fn affine_grid(codes: &[u8], scales: &[f32], biases: &[f32], in_dim: usize) -> Vec<f32> {
    let groups_per_row = in_dim / MLX_GROUP_SIZE;
    codes
        .iter()
        .enumerate()
        .map(|(i, &q)| {
            let (row, col) = (i / in_dim, i % in_dim);
            let g = row * groups_per_row + col / MLX_GROUP_SIZE;
            scales[g] * q as f32 + biases[g]
        })
        .collect()
}

/// Unpack MLX u32-packed 4-bit codes (LSB-first nibbles) into one `u8` code per element.
fn unpack_mlx_q4(wq: &[u32], out_dim: usize, in_dim: usize) -> Vec<u8> {
    let words_per_row = in_dim / 8;
    let mut codes = vec![0u8; out_dim * in_dim];
    for row in 0..out_dim {
        for (w, &word) in wq[row * words_per_row..(row + 1) * words_per_row]
            .iter()
            .enumerate()
        {
            let base = row * in_dim + w * 8;
            for nib in 0..8 {
                codes[base + nib] = ((word >> (4 * nib)) & 0xF) as u8;
            }
        }
    }
    codes
}

/// An MLX packed triple pulled to CPU vectors: u32 code words + f32 scales/biases + the tensor's
/// `[out, in]` dims.
type MlxParts = (Vec<u32>, Vec<f32>, Vec<f32>, usize, usize);

/// Pull an MLX packed triple to CPU vectors, validating the group-64 4-bit shape contract
/// (`wq [out, in/8]`, `scales`/`biases` `[out, in/64]`).
fn q4_parts(wq: &Tensor, scales: &Tensor, biases: &Tensor) -> Result<MlxParts> {
    let (out_dim, wq_cols) = wq.dims2()?;
    let (s_rows, s_cols) = scales.dims2()?;
    let in_dim = s_cols * MLX_GROUP_SIZE;
    if s_rows != out_dim || biases.dims2()? != (s_rows, s_cols) || wq_cols * 8 != in_dim {
        candle_core::bail!(
            "not an MLX group-64 Q4 pack: wq {:?}, scales {:?}, biases {:?}",
            wq.shape(),
            scales.shape(),
            biases.shape()
        );
    }
    let cpu = Device::Cpu;
    let wq = wq.to_device(&cpu)?.flatten_all()?.to_vec1::<u32>()?;
    let scales = scales
        .to_device(&cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let biases = biases
        .to_device(&cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    Ok((wq, scales, biases, out_dim, in_dim))
}

/// Repack an MLX group-64 affine **Q4** triple into a GGML **`Q4_1`** [`QTensor`] on `device` —
/// lossless up to the bf16 → f16 scale/bias cast (see the module docs). The result plugs straight
/// into the per-crate `QLinear::Quantized` dequant-on-forward path.
pub fn repack_mlx_q4_to_q4_1(
    wq: &Tensor,
    scales: &Tensor,
    biases: &Tensor,
    device: &Device,
) -> Result<QTensor> {
    let (wq, scales, biases, out_dim, in_dim) = q4_parts(wq, scales, biases)?;
    let codes = unpack_mlx_q4(&wq, out_dim, in_dim);

    let blocks = out_dim * in_dim / Q4_1_BLOCK;
    let mut bytes = Vec::with_capacity(blocks * Q4_1_BLOCK_BYTES);
    let groups_per_row = in_dim / MLX_GROUP_SIZE;
    for row in 0..out_dim {
        for g in 0..groups_per_row {
            let d = f16::from_f32(scales[row * groups_per_row + g]);
            let m = f16::from_f32(biases[row * groups_per_row + g]);
            let group = &codes[row * in_dim + g * MLX_GROUP_SIZE..][..MLX_GROUP_SIZE];
            // One MLX group of 64 = two consecutive Q4_1 blocks of 32 with the same d/m.
            for block in group.chunks_exact(Q4_1_BLOCK) {
                bytes.extend_from_slice(&d.to_le_bytes());
                bytes.extend_from_slice(&m.to_le_bytes());
                for j in 0..Q4_1_BLOCK / 2 {
                    bytes.push(block[j] | (block[j + Q4_1_BLOCK / 2] << 4));
                }
            }
        }
    }

    // MUST be `Cow::Borrowed`: candle's `as_t_slice` takes the Cow by value and returns a slice
    // borrowed from it, so an `Owned` cow's backing Vec is dropped before `from_data` copies the
    // blocks out — a use-after-free that reads freed memory (garbage weights, found in the sc-9085
    // spike). Borrowed keeps `bytes` alive across the call; `from_data` clones into its own Vec.
    let storage = QStorage::from_data(Cow::Borrowed(&bytes), device, GgmlDType::Q4_1)?;
    QTensor::new(storage, (out_dim, in_dim))
}

/// The exact f32 grid values of an MLX Q4 pack — the repack's loss-free reference (spike
/// verification; not a load path).
pub fn dequant_mlx_q4_reference(wq: &Tensor, scales: &Tensor, biases: &Tensor) -> Result<Tensor> {
    let (wq, scales, biases, out_dim, in_dim) = q4_parts(wq, scales, biases)?;
    // Reference through the same f16 cast the repack bakes in, so "lossless" is measured against
    // what a Q4_1 block can represent (the f16-exactness census reports the cast's own deviation).
    let scales: Vec<f32> = scales.iter().map(|&s| f16::from_f32(s).to_f32()).collect();
    let biases: Vec<f32> = biases.iter().map(|&b| f16::from_f32(b).to_f32()).collect();
    let codes = unpack_mlx_q4(&wq, out_dim, in_dim);
    Tensor::from_vec(
        affine_grid(&codes, &scales, &biases, in_dim),
        (out_dim, in_dim),
        &Device::Cpu,
    )
}

/// Materialize an MLX group-64 affine **Q8** triple (`wq [out, in/4]` u32, LSB-first bytes) as its
/// exact f32 grid values. 8-bit has no affine GGML container, so the Q8 tier path is
/// dequant-then-`QTensor::quantize(…, Q8_0)` — this is the dequant half (and the error reference
/// the spike measures the `Q8_0` re-quantization against).
pub fn dequant_mlx_q8(wq: &Tensor, scales: &Tensor, biases: &Tensor) -> Result<Tensor> {
    let (out_dim, wq_cols) = wq.dims2()?;
    let (s_rows, s_cols) = scales.dims2()?;
    let in_dim = s_cols * MLX_GROUP_SIZE;
    if s_rows != out_dim || biases.dims2()? != (s_rows, s_cols) || wq_cols * 4 != in_dim {
        candle_core::bail!(
            "not an MLX group-64 Q8 pack: wq {:?}, scales {:?}, biases {:?}",
            wq.shape(),
            scales.shape(),
            biases.shape()
        );
    }
    let cpu = Device::Cpu;
    let words = wq.to_device(&cpu)?.flatten_all()?.to_vec1::<u32>()?;
    let scales = scales
        .to_device(&cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let biases = biases
        .to_device(&cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    let mut codes = vec![0u8; out_dim * in_dim];
    for (w, &word) in words.iter().enumerate() {
        for b in 0..4 {
            codes[w * 4 + b] = ((word >> (8 * b)) & 0xFF) as u8;
        }
    }
    Tensor::from_vec(
        affine_grid(&codes, &scales, &biases, in_dim),
        (out_dim, in_dim),
        &Device::Cpu,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pack per-element 4-bit codes into MLX u32 words (LSB-first nibbles) — the test-side inverse
    /// of `unpack_mlx_q4`.
    fn pack_mlx_q4(codes: &[u8]) -> Vec<u32> {
        codes
            .chunks_exact(8)
            .map(|c| {
                c.iter()
                    .enumerate()
                    .fold(0u32, |acc, (i, &q)| acc | ((q as u32 & 0xF) << (4 * i)))
            })
            .collect()
    }

    /// Position-dependent codes + per-group-distinct f16-exact scales/biases: dequantizing the
    /// repacked `Q4_1` tensor must reproduce the MLX affine grid EXACTLY. The position-dependence
    /// makes the assertion sensitive to any nibble/block/group ordering mistake on either side of
    /// the permutation.
    #[test]
    fn q4_repack_is_lossless_and_order_sensitive() -> Result<()> {
        let (out_dim, in_dim) = (4, 128); // 2 groups/row, 4 Q4_1 blocks/row
        let codes: Vec<u8> = (0..out_dim * in_dim)
            .map(|i| ((i * 7 + i / 13) % 16) as u8)
            .collect();
        let groups = out_dim * in_dim / MLX_GROUP_SIZE;
        // Exactly f16-representable, distinct per group, including negatives.
        let scales: Vec<f32> = (0..groups).map(|g| 0.0625 * (g as f32 + 1.0)).collect();
        let biases: Vec<f32> = (0..groups).map(|g| -0.5 - 0.25 * g as f32).collect();
        assert!(scales.iter().chain(biases.iter()).all(|&x| f16_exact(x)));

        let wq = Tensor::from_vec(pack_mlx_q4(&codes), (out_dim, in_dim / 8), &Device::Cpu)?;
        let s = Tensor::from_vec(
            scales.clone(),
            (out_dim, in_dim / MLX_GROUP_SIZE),
            &Device::Cpu,
        )?;
        let b = Tensor::from_vec(
            biases.clone(),
            (out_dim, in_dim / MLX_GROUP_SIZE),
            &Device::Cpu,
        )?;
        assert_eq!(mlx_packed_bits(in_dim / 8, in_dim / MLX_GROUP_SIZE), 4);

        let qt = repack_mlx_q4_to_q4_1(&wq, &s, &b, &Device::Cpu)?;
        assert_eq!(qt.dtype(), GgmlDType::Q4_1);
        let got = qt
            .dequantize(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let want = affine_grid(&codes, &scales, &biases, in_dim);
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert_eq!(g, w, "element {i}: repacked dequant {g} != MLX grid {w}");
        }
        // And the reference helper agrees with the hand-computed grid.
        let reference = dequant_mlx_q4_reference(&wq, &s, &b)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(reference, want);
        Ok(())
    }

    /// The Q8 dequant helper reproduces the MLX 8-bit affine grid exactly (byte order + grouping).
    #[test]
    fn q8_dequant_matches_grid() -> Result<()> {
        let (out_dim, in_dim) = (2, 128);
        let codes: Vec<u8> = (0..out_dim * in_dim)
            .map(|i| ((i * 31 + 5) % 256) as u8)
            .collect();
        let words: Vec<u32> = codes
            .chunks_exact(4)
            .map(|c| {
                c.iter()
                    .enumerate()
                    .fold(0u32, |acc, (i, &q)| acc | ((q as u32) << (8 * i)))
            })
            .collect();
        let groups = out_dim * in_dim / MLX_GROUP_SIZE;
        let scales: Vec<f32> = (0..groups).map(|g| 0.03125 * (g as f32 + 1.0)).collect();
        let biases: Vec<f32> = (0..groups).map(|g| -1.0 + 0.125 * g as f32).collect();

        let wq = Tensor::from_vec(words, (out_dim, in_dim / 4), &Device::Cpu)?;
        let s = Tensor::from_vec(
            scales.clone(),
            (out_dim, in_dim / MLX_GROUP_SIZE),
            &Device::Cpu,
        )?;
        let b = Tensor::from_vec(
            biases.clone(),
            (out_dim, in_dim / MLX_GROUP_SIZE),
            &Device::Cpu,
        )?;
        assert_eq!(mlx_packed_bits(in_dim / 4, in_dim / MLX_GROUP_SIZE), 8);

        let got = dequant_mlx_q8(&wq, &s, &b)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let want = affine_grid(&codes, &scales, &biases, in_dim);
        assert_eq!(got, want);
        Ok(())
    }
}
