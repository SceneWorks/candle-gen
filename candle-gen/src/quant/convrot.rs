//! **ConvRot online rotation** (sc-9601, epic 9083) — the activation-side leg the sc-9300 consume path
//! was missing. A community INT8-ConvRot checkpoint (`krea2_turbo_int8_convrot.safetensors`, a ComfyUI
//! export of the arXiv 2512.03673 method) does NOT store the canonical weight `W`; it stores the
//! **rotated** weight `RHT(W) = W·R` quantized to int8, where `R` is a *regular* Hadamard transform
//! applied block-diagonally in groups of `N₀` (default 256) along the input dim `K`. The stored codes
//! alone reconstruct `X·(W·R)ᵀ`, which is noise (the sc-9300 A/B NO-GO, PSNR ≈ 8 dB). Recovering the
//! true `X·Wᵀ` needs the **same** rotation applied online to the activation:
//!
//! ```text
//! X·Wᵀ = Σ_g (X_g·R)·(W_g·R)ᵀ = Σ_g RHT(X_g)·RHT(W_g)ᵀ     (R orthogonal, R·Rᵀ = I; per K-block g)
//! ```
//!
//! So the forward applies `RHT(X)` (this module) and then the plain int8 IGEMM against the stored
//! rotated weight codes — no weight un-rotation, no stored rotation matrix.
//!
//! # The transform R (recovered from the paper + the checkpoint)
//! `R` is the **regular Hadamard matrix (RHT)** of order `N₀`, NOT the Sylvester/Walsh–Hadamard (which
//! is why the sc-9300 spike's power-of-two block-Hadamard trials failed to recover `W`). It is built by
//! the Kronecker power of the *regular* order-4 Hadamard
//!
//! ```text
//! H₄ = [ 1  1  1 -1        H_{4^{k+1}} = H_{4^k} ⊗ H₄
//!        1  1 -1  1
//!        1 -1  1  1
//!       -1  1  1  1 ]
//! ```
//!
//! normalized by `1/√N₀` (so `H·Hᵀ = N₀·I` ⇒ `R = H/√N₀` is orthogonal). `H₄` is symmetric, so `R` is a
//! **symmetric orthogonal involution** (`R = Rᵀ = R⁻¹`) — the same right-multiply un-rotates. The paper's
//! group sizes are powers of four (16, 64, 256, 1024); the checkpoint uses `convrot_groupsize = 256`
//! (`= 4⁴`), read per-projection from its `comfy_quant` descriptor.
//!
//! Numerically verified against `krea/Krea-2-Turbo`: dequantizing the stored `blocks.0.attn.wq` codes and
//! right-multiplying each 256-block by this `R` recovers the canonical `to_q` at cosine `0.99996`
//! (absmax 1.8596 vs 1.8594); the full int8 forward with the online `RHT(X)` reaches cosine `0.99991`
//! against the f32 reference linear, versus `0.10` without it.

use candle_core::{DType, Device, Result, Tensor};

/// The regular order-4 Hadamard `H₄` (symmetric, `H₄·H₄ᵀ = 4·I`), the Kronecker seed for every RHT.
const H4: [[f32; 4]; 4] = [
    [1.0, 1.0, 1.0, -1.0],
    [1.0, 1.0, -1.0, 1.0],
    [1.0, -1.0, 1.0, 1.0],
    [-1.0, 1.0, 1.0, 1.0],
];

/// Whether `n` is a power of four (`16, 64, 256, 1024, …`) — the group sizes the RHT Kronecker
/// construction produces. `1` (a single `4⁰`) is also accepted (an identity rotation).
pub fn is_power_of_four(n: usize) -> bool {
    n != 0 && n.is_power_of_two() && n.trailing_zeros().is_multiple_of(2)
}

/// Build the normalized regular Hadamard `R = H_{N₀} / √N₀` (`[N₀, N₀]`, f32) on `device`, where
/// `H_{N₀}` is the Kronecker power of [`H4`]. `group_size` must be a power of four (the RHT construction;
/// the checkpoint uses 256). `R` is symmetric and orthogonal (`R·Rᵀ = I`), so applying it to the
/// activation online inverts the same rotation folded into the stored weight.
///
/// Built once per `(group_size, device)` and cached by the caller (the matrix is tiny — 256² f32 =
/// 256 KiB — but the Kronecker build shouldn't repeat across a 12 B DiT's 224 projections × N steps).
pub fn regular_hadamard(group_size: usize, device: &Device) -> Result<Tensor> {
    if !is_power_of_four(group_size) {
        candle_core::bail!(
            "ConvRot regular Hadamard order {group_size} is not a power of four (16, 64, 256, 1024, …)"
        );
    }
    // Kronecker-power H₄ up to `group_size` on the host, then normalize by 1/√group_size.
    let mut h: Vec<f32> = vec![1.0];
    let mut n = 1usize;
    while n < group_size {
        let mut next = vec![0f32; n * 4 * n * 4];
        let nn = n * 4;
        for (i, hi) in h.iter().enumerate() {
            let (r, c) = (i / n, i % n);
            for (br, row) in H4.iter().enumerate() {
                for (bc, &hb) in row.iter().enumerate() {
                    // Kronecker: element (r,c) of H scales the H₄ block at (br,bc).
                    next[(r + br * n) * nn + (c + bc * n)] = hi * hb;
                }
            }
        }
        h = next;
        n = nn;
    }
    let inv_sqrt = 1.0f32 / (group_size as f32).sqrt();
    for v in &mut h {
        *v *= inv_sqrt;
    }
    Tensor::from_vec(h, (group_size, group_size), device)
}

/// Apply the block-diagonal ConvRot rotation `RHT(x) = x·R` to the last (`K`) dim of `x`, in groups of
/// `r`'s order `N₀`: reshape the `K` axis into `K/N₀` blocks of `N₀`, right-multiply each by `r`
/// (`[N₀, N₀]`), and reshape back. `K` must be a multiple of `N₀`. Computed in **f32** for accuracy
/// (the Hadamard's `±1/√N₀` entries lose precision in bf16) and returned in f32 — the downstream int8
/// activation quant reads it in f32 anyway.
///
/// This is the online leg: with the stored weight already `W·R`, `RHT(x)·(W·R)ᵀ = x·Wᵀ`.
pub fn convrot_rotate(x: &Tensor, r: &Tensor) -> Result<Tensor> {
    let n0 = r.dim(0)?;
    let dims = x.dims().to_vec();
    let k = *dims
        .last()
        .expect("convrot_rotate: activation has a last dim");
    if !k.is_multiple_of(n0) {
        candle_core::bail!("ConvRot rotate: K ({k}) is not a multiple of group size ({n0})");
    }
    let lead: usize = dims[..dims.len() - 1].iter().product();
    // [lead·(K/N₀), N₀] @ [N₀, N₀] → [lead·(K/N₀), N₀], then restore [.., K].
    let blocks = lead * (k / n0);
    let xr = x
        .to_dtype(DType::F32)?
        .reshape((blocks, n0))?
        .matmul(&r.to_dtype(DType::F32)?)?
        .reshape(dims)?;
    Ok(xr)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn to_vec2(t: &Tensor) -> Vec<Vec<f32>> {
        t.to_dtype(DType::F32).unwrap().to_vec2::<f32>().unwrap()
    }

    #[test]
    fn power_of_four_predicate() {
        for n in [1usize, 4, 16, 64, 256, 1024, 4096] {
            assert!(is_power_of_four(n), "{n} is 4^k");
        }
        for n in [0usize, 2, 8, 32, 128, 512, 3, 6] {
            assert!(!is_power_of_four(n), "{n} is not 4^k");
        }
    }

    /// `R = H/√N₀` is a symmetric orthogonal involution for every supported order.
    #[test]
    fn regular_hadamard_is_orthogonal_involution() -> Result<()> {
        let dev = Device::Cpu;
        for &g in &[4usize, 16, 64, 256] {
            let r = regular_hadamard(g, &dev)?;
            assert_eq!(r.dims(), &[g, g]);
            let rrt = r.matmul(&r.t()?)?;
            let eye = Tensor::from_vec(
                (0..g * g)
                    .map(|i| if i / g == i % g { 1f32 } else { 0f32 })
                    .collect::<Vec<_>>(),
                (g, g),
                &dev,
            )?;
            let max_off = rrt.sub(&eye)?.abs()?.max_all()?.to_scalar::<f32>()?;
            assert!(max_off < 1e-5, "R·Rᵀ ≠ I for order {g}: max|.| = {max_off}");
            // Symmetric ⇒ its own inverse.
            let asym = r.sub(&r.t()?)?.abs()?.max_all()?.to_scalar::<f32>()?;
            assert!(asym < 1e-6, "R not symmetric for order {g}: {asym}");
        }
        Ok(())
    }

    /// The order-4 matrix is exactly the regular `H₄/2` (not the Sylvester `[[1,1],[1,-1]]⊗…`).
    #[test]
    fn order4_is_regular_not_sylvester() -> Result<()> {
        let r = regular_hadamard(4, &Device::Cpu)?;
        let got = to_vec2(&r);
        let want = [
            [0.5, 0.5, 0.5, -0.5],
            [0.5, 0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5, 0.5],
        ];
        for (gr, wr) in got.iter().zip(&want) {
            for (g, w) in gr.iter().zip(wr) {
                assert!((g - w).abs() < 1e-6, "regular H₄/2 mismatch: {g} vs {w}");
            }
        }
        Ok(())
    }

    /// Rotating twice with the involution `R` is the identity (`RHT(RHT(x)) = x`), and the round trip
    /// `RHT(x)·(W·R)ᵀ == x·Wᵀ` holds block-diagonally over a `K` that spans multiple groups.
    #[test]
    fn rotate_round_trip_reconstructs_linear() -> Result<()> {
        let dev = Device::Cpu;
        let (g, groups) = (16usize, 3usize);
        let k = g * groups; // 48, spans 3 blocks
        let r = regular_hadamard(g, &dev)?;

        // Double rotation is identity.
        let x = Tensor::randn(0f32, 1f32, (5, k), &dev)?;
        let twice = convrot_rotate(&convrot_rotate(&x, &r)?, &r)?;
        let back = twice.sub(&x)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(back < 1e-4, "RHT∘RHT ≠ identity: max|.| = {back}");

        // RHT(x)·(W·R)ᵀ == x·Wᵀ, where W·R is the per-block right rotation of W (the stored weight).
        let out = 7usize;
        let w = Tensor::randn(0f32, 1f32, (out, k), &dev)?;
        let wr = convrot_rotate(&w, &r)?; // stored rotated weight
        let xr = convrot_rotate(&x, &r)?;
        let got = xr.matmul(&wr.t()?)?;
        let want = x.matmul(&w.t()?)?;
        let err = got.sub(&want)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(err < 1e-3, "ConvRot round trip max abs err {err}");
        Ok(())
    }
}
