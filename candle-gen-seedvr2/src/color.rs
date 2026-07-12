//! SeedVR2 post-decode color correction — candle port of `mlx-gen-seedvr2/src/color.rs` (a faithful
//! port of `SeedVR2Util.apply_color_correction`): wavelet reconstruction (content high-freq + style
//! low-freq) then a LAB color transfer (histogram-match the a/b chroma + partial L) to the LR style.
//! Pure host f32 arithmetic (B=1); only the tensor I/O at the boundary differs from the MLX version.
//! Operates on `(1,3,H,W)` in [-1,1] → `(1,3,H,W)` in [-1,1].

use candle_gen::candle_core::{DType, Result, Tensor};
use rayon::prelude::*;

const KERNEL: [[f32; 3]; 3] = [
    [0.0625, 0.125, 0.0625],
    [0.125, 0.25, 0.125],
    [0.0625, 0.125, 0.0625],
];

#[inline]
fn clampi(v: i64, lo: i64, hi: i64) -> usize {
    v.clamp(lo, hi) as usize
}

/// Dilated 3×3 wavelet blur of one `H×W` channel (clamp padding), dilation = `radius`.
///
/// The dominant post-decode cost (≈270 MAC/pixel across a 5-level decomposition, ×3 channels,
/// ×2 decomps, per output frame). Every output pixel is an independent 3×3 gather, so the row loop
/// is data-parallel: each row reads only the shared (immutable) input and writes its own output
/// slice. The per-pixel accumulation order is unchanged, so the result is bit-identical to the
/// serial version — only the wall-clock cost moves off the single host thread (sc-11232 / F-094).
fn wavelet_blur(img: &[f32], h: i32, w: i32, radius: i32) -> Vec<f32> {
    let mut r = radius.max(1);
    let max_safe = (h.min(w) / 8).max(1);
    if r > max_safe {
        r = max_safe;
    }
    let (hh, ww) = (h as i64, w as i64);
    let wu = w as usize;
    let mut out = vec![0f32; (h * w) as usize];
    // One rayon task per output row; `par_chunks_mut(wu)` disjointly partitions `out` by row.
    out.par_chunks_mut(wu)
        .enumerate()
        .for_each(|(row_idx, row)| {
            let y = row_idx as i64;
            for (x, dst) in row.iter_mut().enumerate() {
                let x = x as i64;
                let mut acc = 0f32;
                for (ky, dy) in [-1i64, 0, 1].iter().enumerate() {
                    let yy = clampi(y + dy * r as i64, 0, hh - 1);
                    for (kx, dx) in [-1i64, 0, 1].iter().enumerate() {
                        let xx = clampi(x + dx * r as i64, 0, ww - 1);
                        acc += KERNEL[ky][kx] * img[yy * wu + xx];
                    }
                }
                *dst = acc;
            }
        });
    out
}

/// 5-level wavelet decomposition of one channel → `(high_freq, low_freq)`.
fn wavelet_decomp(img: &[f32], h: i32, w: i32) -> (Vec<f32>, Vec<f32>) {
    let n = (h * w) as usize;
    let mut high = vec![0f32; n];
    let mut cur = img.to_vec();
    for i in 0..5 {
        let radius = 1 << i; // 1,2,4,8,16
        let low = wavelet_blur(&cur, h, w, radius);
        for k in 0..n {
            high[k] += cur[k] - low[k];
        }
        cur = low;
    }
    (high, cur)
}

fn srgb_to_linear(x: f32) -> f32 {
    if x > 0.04045 {
        ((x + 0.055) / 1.055).powf(2.4)
    } else {
        x / 12.92
    }
}
fn linear_to_srgb(x: f32) -> f32 {
    if x > 0.0031308 {
        1.055 * x.max(0.0).powf(1.0 / 2.4) - 0.055
    } else {
        12.92 * x
    }
}

const EPS: f32 = 6.0 / 29.0;
fn lab_f(t: f32) -> f32 {
    let eps3 = EPS * EPS * EPS;
    if t > eps3 {
        t.cbrt()
    } else {
        let kappa = (29.0f32 / 3.0).powi(3);
        (kappa * t + 16.0) / 116.0
    }
}
fn lab_finv(f: f32) -> f32 {
    if f > EPS {
        f * f * f
    } else {
        let kappa = (29.0f32 / 3.0).powi(3);
        (116.0 * f - 16.0) / kappa
    }
}

fn rgb_to_lab(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let (rl, gl, bl) = (srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b));
    let mut x = 0.4124564 * rl + 0.3575761 * gl + 0.1804375 * bl;
    let y = 0.2126729 * rl + 0.7151522 * gl + 0.0721750 * bl;
    let mut z = 0.0193339 * rl + 0.119_192 * gl + 0.9503041 * bl;
    x /= 0.95047;
    z /= 1.08883;
    let (fx, fy, fz) = (lab_f(x), lab_f(y), lab_f(z));
    (116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz))
}

fn lab_to_rgb(l: f32, a: f32, b: f32) -> (f32, f32, f32) {
    let fy = (l + 16.0) / 116.0;
    let fx = a / 500.0 + fy;
    let fz = fy - b / 200.0;
    let x = lab_finv(fx) * 0.95047;
    let y = lab_finv(fy);
    let z = lab_finv(fz) * 1.08883;
    let rl = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z;
    let gl = -0.969_266 * x + 1.8760108 * y + 0.0415560 * z;
    let bl = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z;
    (linear_to_srgb(rl), linear_to_srgb(gl), linear_to_srgb(bl))
}

/// Histogram match `source` onto `reference` (equal length) by rank (numpy stable-argsort port).
fn hist_match(source: &[f32], reference: &[f32]) -> Vec<f32> {
    let n = source.len();
    let mut src_idx: Vec<usize> = (0..n).collect();
    src_idx.sort_by(|&i, &j| source[i].total_cmp(&source[j])); // stable, NaN-safe
    let mut ref_sorted = reference.to_vec();
    ref_sorted.sort_by(f32::total_cmp);
    let mut inv = vec![0usize; n];
    for (rank, &p) in src_idx.iter().enumerate() {
        inv[p] = rank;
    }
    (0..n).map(|p| ref_sorted[inv[p]]).collect()
}

/// `content`/`style`: `(1,3,H,W)` in [-1,1]. Returns the corrected `(1,3,H,W)` in [-1,1].
pub fn apply_color_correction(
    content: &Tensor,
    style: &Tensor,
    luminance_weight: f32,
) -> Result<Tensor> {
    let (_b, _c, h32, w32) = content.dims4()?;
    let (h, w) = (h32 as i32, w32 as i32);
    let n = (h * w) as usize;
    let dt = content.dtype();
    let dev = content.device().clone();
    let c = content
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let s = style
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    // 1. wavelet reconstruction: content high-freq + style low-freq, per channel. The three
    // channels are independent, so `par_chunks_mut(n)` hands each its own output slice; each blur
    // inside `wavelet_decomp` is itself row-parallel. Per-element math is unchanged (bit-identical).
    let mut recon = vec![0f32; 3 * n];
    recon.par_chunks_mut(n).enumerate().for_each(|(ch, dst)| {
        let (chigh, _) = wavelet_decomp(&c[ch * n..(ch + 1) * n], h, w);
        let (_, slow) = wavelet_decomp(&s[ch * n..(ch + 1) * n], h, w);
        for (d, (hi, lo)) in dst.iter_mut().zip(chigh.iter().zip(slow.iter())) {
            *d = (hi + lo).clamp(-1.0, 1.0);
        }
    });

    // 2. to LAB (content from `recon`, style from the original). The transfer is per-pixel with
    // three `powf`/`cbrt`-heavy conversions, so map the pixels in parallel into interleaved LAB
    // triples, then deinterleave into the channel-major arrays `hist_match` needs. The interleave→
    // deinterleave shuffle is memory-bound and cheap next to the transcendentals it parallelizes.
    let to01 = |v: f32| ((v + 1.0) * 0.5).clamp(0.0, 1.0);
    let lab_of = |src: &[f32]| -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let interleaved: Vec<[f32; 3]> = (0..n)
            .into_par_iter()
            .map(|p| {
                let (l, a, b) = rgb_to_lab(to01(src[p]), to01(src[n + p]), to01(src[2 * n + p]));
                [l, a, b]
            })
            .collect();
        let (mut l, mut a, mut b) = (vec![0f32; n], vec![0f32; n], vec![0f32; n]);
        for (p, &[lp, ap, bp]) in interleaved.iter().enumerate() {
            l[p] = lp;
            a[p] = ap;
            b[p] = bp;
        }
        (l, a, b)
    };
    let (c_l, c_a, c_b) = lab_of(&recon);
    let (s_l, s_a, s_b) = lab_of(&s);

    // 3. histogram-match chroma; partial L blend. The 2–3 histogram matches are independent sorts;
    // run them concurrently with `rayon::join` (each match stays serial internally, so its rank
    // mapping is identical to the single-threaded version).
    let (matched_a, (matched_b, out_l)) = rayon::join(
        || hist_match(&c_a, &s_a),
        || {
            rayon::join(
                || hist_match(&c_b, &s_b),
                || {
                    if luminance_weight < 1.0 {
                        let matched_l = hist_match(&c_l, &s_l);
                        (0..n)
                            .map(|p| {
                                luminance_weight * c_l[p] + (1.0 - luminance_weight) * matched_l[p]
                            })
                            .collect()
                    } else {
                        c_l.clone()
                    }
                },
            )
        },
    );

    // 4. back to RGB → [-1,1], (1,3,H,W). Per-pixel and independent: map in parallel into
    // interleaved RGB, then scatter serially into the channel-major output buffer.
    let rgb: Vec<[f32; 3]> = (0..n)
        .into_par_iter()
        .map(|p| {
            let (r, g, b) = lab_to_rgb(out_l[p], matched_a[p], matched_b[p]);
            [
                r.clamp(0.0, 1.0) * 2.0 - 1.0,
                g.clamp(0.0, 1.0) * 2.0 - 1.0,
                b.clamp(0.0, 1.0) * 2.0 - 1.0,
            ]
        })
        .collect();
    let mut out = vec![0f32; 3 * n];
    for (p, &[r, g, b]) in rgb.iter().enumerate() {
        out[p] = r;
        out[n + p] = g;
        out[2 * n + p] = b;
    }
    Tensor::from_vec(out, (1, 3, h as usize, w as usize), &dev)?.to_dtype(dt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Device;

    /// Independent, fully single-threaded reference reproducing the pre-vectorization algorithm
    /// (sc-11232 / F-094 baseline). Kept separate from the parallel production code so the parity
    /// test compares against a genuinely serial implementation, not a refactor of the same loops.
    fn wavelet_blur_serial(img: &[f32], h: i32, w: i32, radius: i32) -> Vec<f32> {
        let mut r = radius.max(1);
        let max_safe = (h.min(w) / 8).max(1);
        if r > max_safe {
            r = max_safe;
        }
        let (hh, ww) = (h as i64, w as i64);
        let wu = w as usize;
        let mut out = vec![0f32; (h * w) as usize];
        for y in 0..h as i64 {
            for x in 0..w as i64 {
                let mut acc = 0f32;
                for (ky, dy) in [-1i64, 0, 1].iter().enumerate() {
                    let yy = clampi(y + dy * r as i64, 0, hh - 1);
                    for (kx, dx) in [-1i64, 0, 1].iter().enumerate() {
                        let xx = clampi(x + dx * r as i64, 0, ww - 1);
                        acc += KERNEL[ky][kx] * img[yy * wu + xx];
                    }
                }
                out[(y * ww + x) as usize] = acc;
            }
        }
        out
    }

    fn wavelet_decomp_serial(img: &[f32], h: i32, w: i32) -> (Vec<f32>, Vec<f32>) {
        let n = (h * w) as usize;
        let mut high = vec![0f32; n];
        let mut cur = img.to_vec();
        for i in 0..5 {
            let radius = 1 << i;
            let low = wavelet_blur_serial(&cur, h, w, radius);
            for k in 0..n {
                high[k] += cur[k] - low[k];
            }
            cur = low;
        }
        (high, cur)
    }

    /// The exact pre-F-094 host hot loop, operating on channel-major `[-1,1]` slices.
    fn reference_serial(c: &[f32], s: &[f32], h: i32, w: i32, luminance_weight: f32) -> Vec<f32> {
        let n = (h * w) as usize;
        let mut recon = vec![0f32; 3 * n];
        for ch in 0..3 {
            let (chigh, _) = wavelet_decomp_serial(&c[ch * n..(ch + 1) * n], h, w);
            let (_, slow) = wavelet_decomp_serial(&s[ch * n..(ch + 1) * n], h, w);
            for k in 0..n {
                recon[ch * n + k] = (chigh[k] + slow[k]).clamp(-1.0, 1.0);
            }
        }
        let to01 = |v: f32| ((v + 1.0) * 0.5).clamp(0.0, 1.0);
        let (mut c_l, mut c_a, mut c_b) = (vec![0f32; n], vec![0f32; n], vec![0f32; n]);
        let (mut s_l, mut s_a, mut s_b) = (vec![0f32; n], vec![0f32; n], vec![0f32; n]);
        for p in 0..n {
            let (l, a, b) = rgb_to_lab(to01(recon[p]), to01(recon[n + p]), to01(recon[2 * n + p]));
            c_l[p] = l;
            c_a[p] = a;
            c_b[p] = b;
            let (l, a, b) = rgb_to_lab(to01(s[p]), to01(s[n + p]), to01(s[2 * n + p]));
            s_l[p] = l;
            s_a[p] = a;
            s_b[p] = b;
        }
        let matched_a = hist_match(&c_a, &s_a);
        let matched_b = hist_match(&c_b, &s_b);
        let out_l: Vec<f32> = if luminance_weight < 1.0 {
            let matched_l = hist_match(&c_l, &s_l);
            (0..n)
                .map(|p| luminance_weight * c_l[p] + (1.0 - luminance_weight) * matched_l[p])
                .collect()
        } else {
            c_l
        };
        let mut out = vec![0f32; 3 * n];
        for p in 0..n {
            let (r, g, b) = lab_to_rgb(out_l[p], matched_a[p], matched_b[p]);
            out[p] = r.clamp(0.0, 1.0) * 2.0 - 1.0;
            out[n + p] = g.clamp(0.0, 1.0) * 2.0 - 1.0;
            out[2 * n + p] = b.clamp(0.0, 1.0) * 2.0 - 1.0;
        }
        out
    }

    // Small deterministic LCG so the fixture is reproducible without pulling in rand.
    fn fixture(seed: u64, len: usize) -> Vec<f32> {
        let mut x = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        (0..len)
            .map(|_| {
                x ^= x >> 12;
                x ^= x << 25;
                x ^= x >> 27;
                let u = (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32 / (1u32 << 24) as f32;
                u * 2.0 - 1.0 // [-1, 1]
            })
            .collect()
    }

    fn parity_case(h: usize, w: usize, luminance_weight: f32) {
        let dev = Device::Cpu;
        let n = h * w;
        let c = fixture(1, 3 * n);
        let s = fixture(9999, 3 * n);
        let content = Tensor::from_vec(c.clone(), (1, 3, h, w), &dev).unwrap();
        let style = Tensor::from_vec(s.clone(), (1, 3, h, w), &dev).unwrap();

        let got = apply_color_correction(&content, &style, luminance_weight)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let want = reference_serial(&c, &s, h as i32, w as i32, luminance_weight);

        assert_eq!(got.len(), want.len());
        let mut max_abs = 0f32;
        for (g, r) in got.iter().zip(want.iter()) {
            max_abs = max_abs.max((g - r).abs());
        }
        // Rayon only redistributes independent, per-element work; the arithmetic per pixel is
        // unchanged, so the vectorized result is bit-identical to the serial reference.
        assert!(
            max_abs <= 1e-6,
            "vectorized color correction diverged from serial reference: max |Δ| = {max_abs} (h={h}, w={w}, lw={luminance_weight})"
        );
    }

    #[test]
    fn vectorized_matches_serial_partial_luminance() {
        // luminance_weight < 1.0 exercises the extra L-channel histogram match + blend branch.
        parity_case(12, 10, 0.5);
    }

    #[test]
    fn vectorized_matches_serial_full_luminance() {
        // luminance_weight == 1.0 keeps L unchanged (the `else` branch).
        parity_case(9, 13, 1.0);
    }
}
