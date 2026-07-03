//! Unified `FaceAnalysis` — the one entry point that orchestrates SCRFD + ArcFace, the candle twin
//! of mlx-gen-face's `face.rs` (mirroring insightface `app.get()`). BiSeNet face-parsing (PuLID's
//! `face_features_image`) is intentionally NOT ported here: it is PuLID-only and lands with the PuLID
//! provider (sc-5492), keeping this sc-5490 slice the shared detect/embed core that InstantID and the
//! Phase-5 kps_extract surface both need.
//!
//! Pipeline (zero Python): detector blob (cv2-faithful resize-to-fit 640 + pad + normalize) → SCRFD
//! detect → 5-pt `norm_crop` 112² → glintr100 embedding → `Vec<Face>` sorted largest-first.

use candle_gen::candle_core::{Device, Tensor};
use candle_gen::{CandleError, Result};

use crate::align;
use crate::bisenet::{self, BiSeNet};
use crate::common::Weights;
use crate::iresnet::ArcFace;
use crate::scrfd::{Detection, Scrfd, DET_SIZE};

/// One detected face — mirrors insightface's `Face` fields the consumers use.
#[derive(Clone, Debug)]
pub struct Face {
    /// `[x1, y1, x2, y2]` in original-image pixels.
    pub bbox: [f32; 4],
    /// 5 landmarks (L-eye, R-eye, nose, L-mouth, R-mouth) in original-image pixels.
    pub kps: [[f32; 2]; 5],
    /// SCRFD detection confidence.
    pub det_score: f32,
    /// Raw 512-d glintr100 recognition embedding (un-normalized; L2-normalize for cosine).
    pub embedding: Vec<f32>,
}

/// Bounding-box area of a detection (for largest-first ordering).
fn det_area(d: &Detection) -> f32 {
    (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1])
}

/// cv2 `resize` `INTER_LINEAR` for an RGB `u8` HWC image — the SCRFD detector preprocessing. Faithful
/// fixed-point bilinear (half-pixel coords, 11-bit weights, two integer passes, `>>22` with rounding),
/// identical to the MLX sibling.
pub fn resize_bilinear_cv2(
    src: &[u8],
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
) -> Result<Vec<u8>> {
    const C: usize = 3;
    // Public boundary: reject an undersized source buffer with a typed error rather than
    // aborting the process on caller-supplied input (sc-9025 / F-041).
    if src.len() < in_h * in_w * C {
        return Err(CandleError::Msg(format!(
            "resize_bilinear_cv2: src buffer of {} bytes too small for {in_h}×{in_w}×3",
            src.len()
        )));
    }
    const BITS: i64 = 11;
    const SCALE: f64 = (1i64 << BITS) as f64; // 2048

    let coeffs = |in_n: usize, out_n: usize| {
        let scale = in_n as f64 / out_n as f64;
        let mut ofs = Vec::with_capacity(out_n);
        let mut a = Vec::with_capacity(out_n);
        for d in 0..out_n {
            let f = (d as f64 + 0.5) * scale - 0.5;
            let mut s = f.floor() as i64;
            let mut fr = f - s as f64;
            if s < 0 {
                s = 0;
                fr = 0.0;
            }
            if s >= in_n as i64 - 1 {
                s = in_n as i64 - 1;
                fr = 0.0;
            }
            let s1 = (s + 1).min(in_n as i64 - 1);
            let w1 = (fr * SCALE).round_ties_even() as i64;
            let w0 = ((1.0 - fr) * SCALE).round_ties_even() as i64;
            ofs.push((s as usize, s1 as usize));
            a.push((w0, w1));
        }
        (ofs, a)
    };

    let (xofs, xa) = coeffs(in_w, out_w);
    let (yofs, ya) = coeffs(in_h, out_h);

    // The vertical pass only reads the source rows named in `yofs`, so resample just those rows.
    let mut needed: Vec<usize> = yofs.iter().flat_map(|&(s0, s1)| [s0, s1]).collect();
    needed.sort_unstable();
    needed.dedup();
    let mut row_of = vec![usize::MAX; in_h];
    for (hi, &sy) in needed.iter().enumerate() {
        row_of[sy] = hi;
    }

    // Horizontal pass over the needed source rows → int (value·2048).
    let mut hbuf = vec![0i64; needed.len() * out_w * C];
    for (hi, &sy) in needed.iter().enumerate() {
        for (dx, (&(sx, sx1), &(w0, w1))) in xofs.iter().zip(&xa).enumerate() {
            for ch in 0..C {
                hbuf[(hi * out_w + dx) * C + ch] = src[(sy * in_w + sx) * C + ch] as i64 * w0
                    + src[(sy * in_w + sx1) * C + ch] as i64 * w1;
            }
        }
    }

    // Vertical pass → uint8, (acc + 2^21) >> 22.
    let mut out = vec![0u8; out_h * out_w * C];
    for (dy, (&(sy0, sy1), &(v0, v1))) in yofs.iter().zip(&ya).enumerate() {
        let (r0, r1) = (row_of[sy0], row_of[sy1]);
        for dx in 0..out_w {
            for ch in 0..C {
                let acc =
                    hbuf[(r0 * out_w + dx) * C + ch] * v0 + hbuf[(r1 * out_w + dx) * C + ch] * v1;
                out[(dy * out_w + dx) * C + ch] = (((acc + (1 << 21)) >> 22).clamp(0, 255)) as u8;
            }
        }
    }
    Ok(out)
}

/// Build the SCRFD detector blob from an RGB `u8` image: insightface-faithful resize-to-fit 640
/// (aspect-preserving) → top-left pad to 640² → `(rgb − 127.5) / 128`. Returns the **NCHW**
/// `[1,3,640,640]` f32 blob (MLX returns NHWC) and `det_scale` (= `new_h / h`).
pub fn detector_blob(img: &[u8], h: usize, w: usize, device: &Device) -> Result<(Tensor, f32)> {
    // Public boundary: reject an undersized image buffer with a typed error rather than
    // aborting the process on caller-supplied input (sc-9025 / F-041).
    if img.len() < h * w * 3 {
        return Err(CandleError::Msg(format!(
            "detector_blob: img buffer of {} bytes too small for {h}×{w}×3",
            img.len()
        )));
    }
    let det = DET_SIZE;
    let im_ratio = h as f64 / w as f64;
    let (new_w, new_h) = if im_ratio > 1.0 {
        ((det as f64 / im_ratio) as usize, det)
    } else {
        (det, (det as f64 * im_ratio) as usize)
    };
    // Beyond ~640:1 the minor side truncates to 0 (e.g. new_w = (640 / im_ratio) as usize == 0):
    // the blob would stay all-padding, detection returns an empty list, and det_scale maps
    // coordinates through 1/0. Reject such degenerate aspect ratios explicitly (sc-9026, F-042),
    // mirroring the zero-dimension guard in `detect`.
    if new_w == 0 || new_h == 0 {
        return Err(CandleError::Msg(format!(
            "detector_blob: degenerate aspect ratio {h}×{w} resizes the minor side to 0 \
             (new {new_h}×{new_w}); image is too far from square for face detection"
        )));
    }
    let det_scale = new_h as f32 / h as f32;
    let resized = resize_bilinear_cv2(img, h, w, new_h, new_w)?;

    // top-left into a 640² canvas; normalize (rgb-127.5)/128.
    let norm = |v: u8| (v as f32 - 127.5) / 128.0;
    let mut blob = vec![norm(0); det * det * 3]; // padded region = normalized 0
    for y in 0..new_h {
        for x in 0..new_w {
            for ch in 0..3 {
                blob[(y * det + x) * 3 + ch] = norm(resized[(y * new_w + x) * 3 + ch]);
            }
        }
    }
    let nhwc = Tensor::from_vec(blob, (1, det, det, 3), device)?;
    Ok((nhwc.permute((0, 3, 1, 2))?.contiguous()?, det_scale))
}

/// The native face-analysis stack: SCRFD + ArcFace (+ optional BiSeNet for the PuLID crop path).
pub struct FaceAnalysis {
    scrfd: Scrfd,
    arcface: ArcFace,
    parser: Option<BiSeNet>,
    device: Device,
    /// Detection score / NMS thresholds (insightface defaults: 0.5 / 0.4).
    pub det_thresh: f32,
    pub nms_thresh: f32,
}

impl FaceAnalysis {
    /// Build the detection + recognition stack from already-loaded sub-models on `device`. For the
    /// PuLID crop path, add the BiSeNet parser with [`FaceAnalysis::with_parser`].
    pub fn new(scrfd: Scrfd, arcface: ArcFace, device: Device) -> Self {
        Self {
            scrfd,
            arcface,
            parser: None,
            device,
            det_thresh: 0.5,
            nms_thresh: 0.4,
        }
    }

    /// Attach the BiSeNet parser (enables [`FaceAnalysis::face_features_image`]) — the PuLID-FLUX
    /// `face_features_image` path (sc-5492). InstantID / the Phase-5 kps surface don't need it.
    pub(crate) fn with_parser(mut self, bisenet_weights: &Weights) -> Result<Self> {
        self.parser = Some(BiSeNet::from_weights(bisenet_weights)?);
        Ok(self)
    }

    /// Detect every face in an RGB `u8` image, sorted **largest-first** (insightface `app.get()`
    /// order). No ArcFace forward is run — consumers that need only the box/landmarks use this.
    pub fn detect(&self, img: &[u8], h: usize, w: usize) -> Result<Vec<Detection>> {
        // A zero dimension makes `detector_blob` compute `det_scale = new_h / 0 = NaN`; reject first.
        if h == 0 || w == 0 {
            return Err(CandleError::Msg(format!(
                "face detect: image has a zero dimension ({h}×{w})"
            )));
        }
        if img.len() < h * w * 3 {
            return Err(CandleError::Msg(format!(
                "face detect: img buffer of {} bytes too small for {h}×{w}×3",
                img.len()
            )));
        }
        let (blob, det_scale) = detector_blob(img, h, w, &self.device)?;
        let mut dets = self
            .scrfd
            .detect(&blob, det_scale, self.det_thresh, self.nms_thresh)?;
        dets.sort_by(|a, b| det_area(b).total_cmp(&det_area(a)));
        Ok(dets)
    }

    /// Align + ArcFace-embed a single [`detect`](Self::detect) result into a [`Face`] — one
    /// `[1,3,112,112]` recognition forward (embed-on-demand for the largest face).
    pub fn embed(&self, img: &[u8], h: usize, w: usize, det: &Detection) -> Result<Face> {
        let crop = align::norm_crop(img, h, w, &det.kps)?;
        let emb = self
            .arcface
            .forward(&align::to_arcface_input(&[crop], &self.device)?)?;
        Ok(Face {
            bbox: det.bbox,
            kps: det.kps,
            det_score: det.score,
            embedding: emb.flatten_all()?.to_vec1::<f32>()?,
        })
    }

    /// Detect → align → embed every face, sorted **largest-first**. Runs ONE batched
    /// `[N,3,112,112]` ArcFace forward (iresnet100 has no cross-batch ops, so each row is identical
    /// to the per-face forward).
    pub fn analyze(&self, img: &[u8], h: usize, w: usize) -> Result<Vec<Face>> {
        let dets = self.detect(img, h, w)?;
        if dets.is_empty() {
            return Ok(Vec::new());
        }
        let crops: Vec<Vec<u8>> = dets
            .iter()
            .map(|d| align::norm_crop(img, h, w, &d.kps))
            .collect::<Result<Vec<_>>>()?;
        let emb = self
            .arcface
            .forward(&align::to_arcface_input(&crops, &self.device)?)?;
        let flat = emb.flatten_all()?.to_vec1::<f32>()?;
        let dim = flat.len() / dets.len(); // [N, 512] → 512 per row
        Ok(dets
            .iter()
            .enumerate()
            .map(|(i, d)| Face {
                bbox: d.bbox,
                kps: d.kps,
                det_score: d.score,
                embedding: flat[i * dim..(i + 1) * dim].to_vec(),
            })
            .collect())
    }

    /// PuLID `face_features_image`: facexlib 512² align of `face` → BiSeNet parse → background
    /// whitened, foreground grayscale. Returns NCHW `[1,3,512,512]` f32 in `[0,1]` on the stack's
    /// device. Requires [`FaceAnalysis::with_parser`].
    pub fn face_features_image(
        &self,
        img: &[u8],
        h: usize,
        w: usize,
        face: &Face,
    ) -> Result<Tensor> {
        let parser = self.parser.as_ref().ok_or_else(|| {
            CandleError::Msg("face_features_image requires a BiSeNet parser (with_parser)".into())
        })?;
        let crop = align::align_face_512(img, h, w, &face.kps)?; // 512² RGB u8
        let rgb01 = u8_to_rgb01_nchw(&crop, 512, 512, &self.device)?;
        let mask = parser.parse_mask(&bisenet::to_parse_input(&rgb01)?)?;
        bisenet::face_features_image(&rgb01, &mask)
    }
}

/// RGB `u8` HWC → NCHW `[1,3,H,W]` f32 in `[0,1]` (the BiSeNet parse-net / `face_features_image` input
/// layout — candle is channels-first where the MLX sibling is channels-last).
fn u8_to_rgb01_nchw(crop: &[u8], h: usize, w: usize, device: &Device) -> Result<Tensor> {
    let data: Vec<f32> = crop.iter().map(|&v| v as f32 / 255.0).collect();
    let hwc = Tensor::from_vec(data, (1, h, w, 3), device)?;
    Ok(hwc.permute((0, 3, 1, 2))?.contiguous()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The public boundary must return a typed `Err` (not panic) on an undersized buffer (sc-9025).
    #[test]
    fn resize_rejects_undersized_buffer() {
        let src = vec![0u8; 4 * 4 * 3 - 1];
        let err = resize_bilinear_cv2(&src, 4, 4, 8, 8).unwrap_err();
        assert!(
            err.to_string().contains("too small for 4×4×3"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn resize_same_size_is_identity() {
        let src: Vec<u8> = (0..5 * 3 * 3).map(|i| (i * 37 % 256) as u8).collect();
        let out = resize_bilinear_cv2(&src, 5, 3, 5, 3).unwrap();
        assert_eq!(out, src);
    }

    #[test]
    fn resize_constant_preserved_on_tall_downscale() {
        let (in_h, w) = (200usize, 4usize);
        let src = vec![123u8; in_h * w * 3];
        let out = resize_bilinear_cv2(&src, in_h, w, 8, w).unwrap();
        assert_eq!(out.len(), 8 * w * 3);
        assert!(out.iter().all(|&v| v == 123), "constant must be preserved");
    }

    // sc-9026 (F-042): a degenerate aspect ratio truncates the minor side to 0, which used to yield
    // an all-padding blob + det_scale through 1/0 and a silent "no face". It must now error.
    #[test]
    fn detector_blob_rejects_extreme_wide_ratio() {
        let device = Device::Cpu;
        let (h, w) = (1usize, 700usize); // im_ratio ≈ 1/700 → new_h = (640 * 1/700) as usize == 0
        let img = vec![0u8; h * w * 3];
        let err = detector_blob(&img, h, w, &device).unwrap_err();
        assert!(
            err.to_string().contains("degenerate aspect ratio"),
            "expected degenerate-ratio rejection, got: {err}"
        );
    }

    #[test]
    fn detector_blob_rejects_extreme_tall_ratio() {
        let device = Device::Cpu;
        let (h, w) = (700usize, 1usize); // im_ratio == 700 → new_w = (640 / 700) as usize == 0
        let img = vec![0u8; h * w * 3];
        let err = detector_blob(&img, h, w, &device).unwrap_err();
        assert!(
            err.to_string().contains("degenerate aspect ratio"),
            "expected degenerate-ratio rejection, got: {err}"
        );
    }

    // A normal image still produces a full [1,3,640,640] blob; the common (square) path is unchanged:
    // det == new_h == new_w, det_scale == 1.0, and the blob is non-empty.
    #[test]
    fn detector_blob_square_image_unchanged() {
        let device = Device::Cpu;
        let (h, w) = (128usize, 128usize);
        let img: Vec<u8> = (0..h * w * 3).map(|i| (i * 31 % 256) as u8).collect();
        let (blob, det_scale) = detector_blob(&img, h, w, &device).unwrap();
        assert_eq!(blob.dims(), &[1, 3, DET_SIZE, DET_SIZE]);
        assert_eq!(det_scale, DET_SIZE as f32 / h as f32);
        let flat = blob.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(
            flat.iter().any(|&v| v != (0.0f32 - 127.5) / 128.0),
            "a real square image must not produce an all-padding blob"
        );
    }

    // A moderate (non-degenerate) wide ratio still resizes to a valid non-zero minor side.
    #[test]
    fn detector_blob_moderate_ratio_ok() {
        let device = Device::Cpu;
        let (h, w) = (64usize, 256usize); // im_ratio = 0.25 → new_h = 160, new_w = 640
        let img = vec![200u8; h * w * 3];
        let (blob, det_scale) = detector_blob(&img, h, w, &device).unwrap();
        assert_eq!(blob.dims(), &[1, 3, DET_SIZE, DET_SIZE]);
        assert!(det_scale > 0.0 && det_scale.is_finite());
    }
}
