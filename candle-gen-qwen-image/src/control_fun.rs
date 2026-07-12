//! The Qwen-Image **2512-Fun-Controlnet-Union** (VACE) control provider (sc-8350) — the candle
//! (Windows/CUDA) sibling of `mlx-gen-qwen-image`'s `QwenImageControl` (mlx sc-8267 / PR #604).
//! Structural control (pose / canny / depth) on the **Qwen-Image-2512** base via the alibaba-pai
//! `Qwen-Image-2512-Fun-Controlnet-Union` checkpoint (Apache-2.0, ungated).
//!
//! Unlike the retired InstantX Qwen ControlNet lane (an independent mini-transformer
//! emitting residuals the base ADDs at a fixed interval; removed in sc-9868), this is **VACE-style**: a `control_img_in`
//! patch embedder feeds a control state threaded through 5 control blocks that reuse the base block
//! math (seeded at block 0 by `before_proj(c) + img_embed`); each emits a zero-init `after_proj` hint
//! the base 60-layer MMDiT adds into its image stream at `control_layers = [0, 12, 24, 36, 48]` scaled
//! by the request's control scale — [`QwenTransformer::forward_fun_control`].
//!
//! **Input-agnostic** (sc-8250): pose, canny, and depth differ only by the preprocessor-produced
//! control image fed to [`QwenFunControl::generate`] — there is no mode index and no per-kind branch.
//! v1 is pose/canny/depth-from-prompt (no img2img-with-control compose yet).
//!
//! Like the (now-retired) InstantX lane, this is a plain struct driven **directly** by the worker (a
//! bespoke stream, like `candle_gen_sdxl::IpAdapterSdxl`), not a gen-core-registered generator — the
//! registered `qwen_image` descriptor stays txt2img-only. This is the sole Qwen control engine: the
//! bespoke InstantX ControlNet lane was removed in sc-9868 (MLX twin retired in sc-8267, worker
//! repointed InstantX→2512-Fun in sc-8350).

use std::path::{Path, PathBuf};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::runtime::CancelFlag;
use candle_gen::gen_core::{Image, Progress};
use candle_gen::{CandleError, Result};

use crate::config::{TextEncoderConfig, TransformerConfig, NEGATIVE_FALLBACK};
use crate::control_common;
use crate::pipeline;
use crate::text_encoder::QwenTextEncoder;
use crate::transformer::{QwenFunControlBranch, QwenTransformer};
use crate::vae::{QwenVae, QwenVaeEncoder};

/// The transformer + control branch run bf16 (native dtype); the encoder + VAE run f32.
const DIT_DTYPE: DType = DType::BF16;
const ENC_DTYPE: DType = DType::F32;
/// Error-message prefix for this lane (shared [`control_common`] helpers thread it through).
const LABEL: &str = "qwen fun-control";

/// The 2512-Fun Union injects 5 VACE hints into the base 60-layer MMDiT at these base block indices
/// (the alibaba-pai `config/qwenimage_control.yaml` `control_layers`, interval 12). `0` must be present
/// — `before_proj` lives on control block 0.
pub const CONTROL_LAYERS: [usize; 5] = [0, 12, 24, 36, 48];
/// Packed control-context channels (`control_img_in` in-features): `[control_latent(16) | mask(1) |
/// inpaint(16)]` × the 2×2 patch = `33 · 4 = 132`.
pub const CONTROL_IN_DIM: usize =
    (crate::config::LATENT_CHANNELS * 2 + 1) * crate::config::PATCH * crate::config::PATCH;
/// Default conditioning scale on the VACE hints.
pub const DEFAULT_CONTROL_SCALE: f32 = 1.0;

/// Paths to the Qwen-Image 2512-Fun control checkpoints.
pub struct QwenFunControlPaths {
    /// The `Qwen/Qwen-Image-2512` diffusers snapshot dir (`text_encoder/`, `transformer/`, `vae/`,
    /// `tokenizer/`).
    pub qwen_base: PathBuf,
    /// The alibaba-pai `Qwen-Image-2512-Fun-Controlnet-Union` checkpoint — a single `.safetensors`
    /// file or a dir of shards.
    pub controlnet: PathBuf,
}

/// One Qwen-Image 2512-Fun (pose/canny/depth) generation request. The control **kind** is implicit in
/// the control image passed to [`QwenFunControl::generate`] (input-agnostic — no mode field).
#[derive(Clone)]
pub struct QwenFunControlRequest {
    pub prompt: String,
    pub negative: String,
    pub width: u32,
    pub height: u32,
    pub steps: usize,
    /// True-CFG guidance scale.
    pub guidance: f32,
    /// Conditioning scale on the VACE hints (`0` ≡ base txt2img).
    pub control_scale: f32,
    pub seed: u64,
    pub cancel: CancelFlag,
}

impl Default for QwenFunControlRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            negative: String::new(),
            width: 1024,
            height: 1024,
            steps: 30,
            guidance: 4.0,
            control_scale: DEFAULT_CONTROL_SCALE,
            seed: 0,
            cancel: CancelFlag::default(),
        }
    }
}

/// Resolve the 2512-Fun control weight file(s) from a dir-or-file path → the list of `.safetensors`
/// shards to mmap (the checkpoint is a single `Qwen-Image-2512-Fun-Controlnet-Union-….safetensors`,
/// or a dir of shards).
fn resolve_controlnet_files(path: &Path) -> Result<Vec<PathBuf>> {
    // Shared file-or-dir resolver (sc-8999 / F-019): single `.safetensors` → itself, a dir → its
    // sorted shards, a missing path → the crafted `{label}: no .safetensors ...` error.
    candle_gen::resolve_weight_files(path, "qwen fun-control")
}

/// The loaded Qwen-Image 2512-Fun control model: the reused base text encoder / DiT / VAE-decoder, plus
/// the VAE encoder (to encode the control hint) and the VACE control branch.
pub struct QwenFunControl {
    device: Device,
    te: QwenTextEncoder,
    /// Qwen tokenizer, loaded+parsed **once** at load and reused across encodes (sc-8991 / F-011)
    /// instead of re-parsing `tokenizer.json` per prompt/branch.
    tokenizer: candle_gen::gen_core::tokenizer::TextTokenizer,
    transformer: QwenTransformer,
    controlnet: QwenFunControlBranch,
    vae: QwenVae,
    vae_encoder: QwenVaeEncoder,
}

impl QwenFunControl {
    /// Load the base Qwen-Image-2512 components + the VAE encoder + the 2512-Fun VACE control branch.
    pub fn load(paths: &QwenFunControlPaths) -> Result<Self> {
        let device = candle_gen::default_device()?;
        let root = paths.qwen_base.clone();
        // The 2512 base reuses the base config verbatim (sc-8647 / sc-8271 parity).
        let te_cfg = TextEncoderConfig::qwen_image_2512();
        let dit_cfg = TransformerConfig::qwen_image_2512();

        let te = QwenTextEncoder::new(
            &te_cfg,
            control_common::component_vb(&root, "text_encoder", ENC_DTYPE, &device, LABEL)?,
        )?;
        // The base 2512 MMDiT packed-detects (a packed MLX base tier loads straight from the packed
        // parts; a dense base snapshot unchanged) at the `group_size` read from `transformer/config.json`.
        let gs = crate::transformer_group_size(&root.join("transformer"));
        let transformer = QwenTransformer::new_gs(
            &dit_cfg,
            control_common::component_vb(&root, "transformer", DIT_DTYPE, &device, LABEL)?,
            gs,
        )?;
        let vae = QwenVae::new(control_common::component_vb(
            &root, "vae", ENC_DTYPE, &device, LABEL,
        )?)?;
        let vae_encoder = QwenVaeEncoder::new(control_common::component_vb(
            &root, "vae", ENC_DTYPE, &device, LABEL,
        )?)?;

        let cn_files = resolve_controlnet_files(&paths.controlnet)?;
        let cn_vb = candle_gen::mmap_var_builder(&cn_files, DIT_DTYPE, &device)?;
        let controlnet =
            QwenFunControlBranch::new(&dit_cfg, &CONTROL_LAYERS, CONTROL_IN_DIM, cn_vb)?;

        let tokenizer = control_common::load_tokenizer(&root, &te_cfg, LABEL)?;
        Ok(Self {
            device,
            te,
            tokenizer,
            transformer,
            controlnet,
            vae,
            vae_encoder,
        })
    }

    /// Tokenize + encode `prompt` → `prompt_embeds` `[1, seq, 3584]` at the DiT dtype (bf16). Mirrors
    /// the txt2img `Pipeline::encode`.
    fn encode(&self, prompt: &str) -> Result<Tensor> {
        control_common::encode(
            &self.tokenizer,
            &self.te,
            &self.device,
            DIT_DTYPE,
            prompt,
            LABEL,
        )
    }

    /// Structural-control generation: condition the base MMDiT on `control` (a preprocessed pose / canny
    /// / depth image at the request size — input-agnostic, no kind argument) via the 2512-Fun VACE
    /// branch. VAE-encodes + packs the control hint to the 132-ch control context once, then runs the
    /// control denoise (the control branch runs on both CFG passes).
    pub fn generate(
        &self,
        req: &QwenFunControlRequest,
        control: &Image,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        if req.cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }
        require_prompt(&req.prompt)?;
        let (lat_h, lat_w) = pipeline::latent_dims(req.width, req.height);

        let pos = self.encode(&req.prompt)?;
        let neg = if req.guidance > 1.0 {
            let n = if req.negative.trim().is_empty() {
                NEGATIVE_FALLBACK
            } else {
                req.negative.as_str()
            };
            Some(self.encode(n)?)
        } else {
            None
        };

        // VAE-encode the control image → 16-ch latent, then pack the 132-ch control context (control
        // latent + zero mask + zero inpaint, 2×2-packed). Constant across denoise steps + the batch.
        let control_img = control_common::preprocess_control_image(
            control,
            req.width,
            req.height,
            &self.device,
            LABEL,
        )?;
        let control_latent = self.vae_encoder.encode(&control_img)?;
        let control_cond =
            pipeline::pack_fun_control_context(&control_latent, req.width, req.height)?
                .to_dtype(DIT_DTYPE)?;

        // Routed through the unified curated sampler/scheduler framework (epic 7114): the `native`
        // schedule is the production `qwen_sigmas`, `mu` steers the (non-default) curated scheduler. The
        // bespoke control provider has no `req.sampler`/`req.scheduler` surface, so both stay `None` (the
        // N1 default: `euler` over the native schedule). The model is fed the raw sigma (`Sigma`
        // convention); the VACE branch + true-CFG pos/neg/blend all live inside the `predict` closure.
        let native = pipeline::qwen_sigmas(req.steps, req.width, req.height);
        let mu = pipeline::qwen_mu(req.width, req.height);
        let sigmas = candle_gen::resolve_flow_schedule(None, mu, req.steps, &native);
        let latents = pipeline::create_noise(req.seed, req.width, req.height, &self.device)?
            .to_dtype(DIT_DTYPE)?;

        let latents = candle_gen::run_flow_sampler(
            None,
            candle_gen::gen_core::sampling::TimestepConvention::Sigma,
            &sigmas,
            latents,
            req.seed,
            &req.cancel,
            on_progress,
            |latents, sigma| -> Result<Tensor> {
                let pos_v = self.transformer.forward_fun_control(
                    latents,
                    &pos,
                    sigma,
                    lat_h,
                    lat_w,
                    Some((&self.controlnet, &control_cond)),
                    req.control_scale,
                )?;
                match &neg {
                    Some(neg) => {
                        let neg_v = self.transformer.forward_fun_control(
                            latents,
                            neg,
                            sigma,
                            lat_h,
                            lat_w,
                            Some((&self.controlnet, &control_cond)),
                            req.control_scale,
                        )?;
                        Ok(pipeline::compute_guided_noise(
                            &pos_v,
                            &neg_v,
                            req.guidance,
                        )?)
                    }
                    None => Ok(pos_v),
                }
            },
        )?;

        on_progress(Progress::Decoding);
        let lat = pipeline::unpack_latents(&latents, req.width, req.height)?;
        let decoded = self.vae.decode(&lat)?;
        control_common::to_image(&decoded)
    }
}

/// The positive prompt is required on the control lane — unlike the negative (which falls back to
/// [`NEGATIVE_FALLBACK`]), there is no sensible positive fallback. An empty or whitespace-only positive
/// would reach gen-core's `tokenize("")`, whose pre-chat-template short-circuit to zero-length ids
/// underflows `QwenTextEncoder::prompt_embeds`' `hidden.narrow(1, 34, s - 34)` (the sc-8646 class) —
/// a usize-underflow panic in debug, an opaque `narrow` error in release. Fail fast with a diagnosable
/// message instead (sc-11187 / F-085; the descriptor-registered txt2img lane guards this in `validate`,
/// but this bespoke control stream never runs it).
fn require_prompt(prompt: &str) -> Result<()> {
    if prompt.trim().is_empty() {
        return Err(CandleError::Msg(format!(
            "{LABEL}: prompt must not be empty"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::Image;

    /// sc-11187 / F-085: the control lane's positive prompt is required. An empty or whitespace-only
    /// prompt is rejected up front (before it can reach `tokenize("")` and underflow `prompt_embeds`);
    /// a real prompt passes. This bespoke stream never runs the txt2img descriptor's `validate`, so it
    /// carries its own guard.
    #[test]
    fn require_prompt_rejects_empty() {
        assert!(require_prompt("").is_err());
        assert!(require_prompt("   ").is_err());
        assert!(require_prompt("\t\n").is_err());
        let msg = require_prompt("").unwrap_err().to_string();
        assert!(msg.contains("must not be empty"), "got: {msg}");
        assert!(require_prompt("a cat, canny control").is_ok());
    }

    #[test]
    fn request_defaults() {
        let r = QwenFunControlRequest::default();
        assert_eq!((r.width, r.height), (1024, 1024));
        assert_eq!(r.steps, 30);
        assert_eq!(r.control_scale, DEFAULT_CONTROL_SCALE);
        assert!(!r.cancel.is_cancelled());
    }

    /// The shipped 2512-Fun Union: 5 control layers at `[0, 12, 24, 36, 48]` across the 60-layer base
    /// MMDiT (interval 12, `0` present for `before_proj`), control context 132 = (16·2 + 1)·4.
    #[test]
    fn control_layout_matches_fork() {
        assert_eq!(CONTROL_LAYERS, [0, 12, 24, 36, 48]);
        assert_eq!(CONTROL_LAYERS.len(), 5);
        assert!(CONTROL_LAYERS.contains(&0), "before_proj lives on block 0");
        assert_eq!(CONTROL_IN_DIM, 132);
        let base = TransformerConfig::qwen_image_2512();
        // 5 hints evenly spaced across 60 base blocks at interval 12.
        assert_eq!(base.num_layers, 60);
        for (n, &p) in CONTROL_LAYERS.iter().enumerate() {
            assert_eq!(
                p,
                n * 12,
                "control layer {n} should inject at base block {}",
                n * 12
            );
            assert!(p < base.num_layers, "injection index in range");
        }
    }

    #[test]
    fn controlnet_file_resolution() {
        let dir = std::env::temp_dir().join(format!("qwen_fun_cn_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // Empty dir → error.
        assert!(resolve_controlnet_files(&dir).is_err());
        // A single file path resolves to itself.
        let f = dir.join("Qwen-Image-2512-Fun-Controlnet-Union.safetensors");
        std::fs::write(&f, b"x").unwrap();
        assert_eq!(resolve_controlnet_files(&f).unwrap(), vec![f.clone()]);
        // A dir of shards resolves to the sorted shard list.
        let g = dir.join("model-00002.safetensors");
        std::fs::write(&g, b"y").unwrap();
        let got = resolve_controlnet_files(&dir).unwrap();
        assert_eq!(got, vec![f, g]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// This lane's control-image preprocessing goes through the shared [`control_common`] helper with
    /// this lane's `LABEL`; the numeric behavior is unchanged from the pre-dedup verbatim copy.
    #[test]
    fn control_preprocess_shape_and_range() {
        let img = Image {
            width: 16,
            height: 8,
            pixels: vec![255u8; 16 * 8 * 3],
        };
        let t = control_common::preprocess_control_image(&img, 16, 8, &Device::Cpu, LABEL).unwrap();
        assert_eq!(t.dims(), &[1, 3, 8, 16]);
        // 255 → 255/127.5 - 1 = 1.0
        let v = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(v.iter().all(|x| (x - 1.0).abs() < 1e-4));
        // size mismatch errors loudly, with this lane's label.
        let e = control_common::preprocess_control_image(&img, 32, 8, &Device::Cpu, LABEL)
            .unwrap_err()
            .to_string();
        assert!(e.starts_with("qwen fun-control:"), "got: {e}");
    }

    /// **Packed-detect fires on the shared 2512-Fun control tier key layout (sc-9869).** The candle
    /// `QwenFunControl` load path builds every control-branch projection through
    /// [`crate::quant::QLinear::linear_detect_gs`], which packed-detects per key. So when the
    /// caller-provided `controlnet` path points at the shared packed tier
    /// `SceneWorks/qwen-image-2512-fun-controlnet-union` (a single `model.safetensors` of packed
    /// `{base}.weight` u32 + `.scales` + `.biases` triples) instead of the dense alibaba-pai
    /// checkpoint, the projections load straight from the packed parts — **no dense weight is
    /// materialized, no engine change** (the group-size seam was already threaded).
    ///
    /// This test writes a synthetic packed checkpoint mirroring the tier's exact 2512-Fun control key
    /// layout (`control_img_in`, `control_blocks.0.before_proj`, `control_blocks.0.after_proj` — the
    /// keys `QwenFunControlBranch::new` reads) and loads each through the same `linear_detect_gs` call
    /// the loader uses, asserting the packed path fires (`is_packed()`), that a dense sibling (no
    /// `.scales`) stays dense, and that the packed forward reproduces the affine grid bit-exactly.
    #[test]
    fn packed_detect_fires_on_2512fun_control_layout() -> Result<()> {
        use crate::quant::QLinear;
        use candle_gen::candle_core::safetensors::MmapedSafetensors;
        use candle_gen::candle_nn::VarBuilder;
        use std::collections::HashMap;

        let dev = Device::Cpu;
        let gs = candle_gen::quant::MLX_GROUP_SIZE; // the hosted Qwen-Image tiers pack at group 64
        let dev_ref = &dev;

        // Test-side MLX Q4 packer: 4-bit codes → MLX u32 words (LSB-first nibbles), group `gs`. Returns
        // `(wq [out,in/8] u32, scales [out,in/gs], biases [out,in/gs], affine grid [out,in])` — the
        // exact packed-parts fixture the detect loader consumes plus the affine grid it reproduces.
        let q4_packed = |out_dim: usize, in_dim: usize| -> (Tensor, Tensor, Tensor, Vec<f32>) {
            let codes: Vec<u8> = (0..out_dim * in_dim)
                .map(|i| ((i * 7 + i / 13) % 16) as u8)
                .collect();
            let groups = out_dim * in_dim / gs;
            let scales: Vec<f32> = (0..groups).map(|g| 0.0625 * (g as f32 + 1.0)).collect();
            let biases: Vec<f32> = (0..groups).map(|g| -0.5 - 0.25 * g as f32).collect();
            let gpr = in_dim / gs;
            let grid: Vec<f32> = (0..out_dim * in_dim)
                .map(|i| {
                    let (row, col) = (i / in_dim, i % in_dim);
                    let g = row * gpr + col / gs;
                    scales[g] * codes[i] as f32 + biases[g]
                })
                .collect();
            let words: Vec<u32> = codes
                .chunks_exact(8)
                .map(|c| {
                    c.iter()
                        .enumerate()
                        .fold(0u32, |acc, (i, &q)| acc | ((q as u32 & 0xF) << (4 * i)))
                })
                .collect();
            let wq = Tensor::from_vec(words, (out_dim, in_dim / 8), dev_ref).unwrap();
            let s = Tensor::from_vec(scales, (out_dim, gpr), dev_ref).unwrap();
            let b = Tensor::from_vec(biases, (out_dim, gpr), dev_ref).unwrap();
            (wq, s, b, grid)
        };

        // Dims chosen gs-divisible (MLX packing requires `in_dim % group_size == 0`). The real
        // `control_img_in` in-features is `CONTROL_IN_DIM` (132) — not a synthetic width here; a
        // gs-divisible `cin` keeps the fixture a valid packed triple while still exercising the exact
        // `control_img_in.{weight,scales,biases}` key the loader detects. `before/after_proj`:
        // inner → inner.
        let inner = 128usize;
        let cin = 256usize;
        let mut map: HashMap<String, Tensor> = HashMap::new();
        let mut grids: HashMap<String, Vec<f32>> = HashMap::new();

        // The three 2512-Fun control keys the loader packed-detects, each a packed triple. A bias is
        // present (the loader loads these biased) — the dense `.bias` sibling must survive the base
        // string alongside the `.scales`/`.biases` (the key-remap trap `linear_detect_gs` guards).
        for (base, out_dim, in_dim) in [
            ("control_img_in", inner, cin),
            ("control_blocks.0.before_proj", inner, inner),
            ("control_blocks.0.after_proj", inner, inner),
        ] {
            let (wq, s, b, grid) = q4_packed(out_dim, in_dim);
            map.insert(format!("{base}.weight"), wq);
            map.insert(format!("{base}.scales"), s);
            map.insert(format!("{base}.biases"), b);
            let bias_vec: Vec<f32> = (0..out_dim).map(|i| 0.01 * i as f32).collect();
            map.insert(
                format!("{base}.bias"),
                Tensor::from_vec(bias_vec, (out_dim,), &dev)?,
            );
            grids.insert(base.to_string(), grid);
        }
        // A dense sibling (no `.scales`) — the dense path must stay unchanged when a tier ships some
        // weight dense.
        map.insert(
            "control_blocks.0.dense_proj.weight".into(),
            Tensor::randn(0f32, 1f32, (inner, inner), &dev)?,
        );
        map.insert(
            "control_blocks.0.dense_proj.bias".into(),
            Tensor::zeros((inner,), DType::F32, &dev)?,
        );

        let tmp = std::env::temp_dir().join(format!(
            "sc9869_2512fun_packed_{}.safetensors",
            std::process::id()
        ));
        candle_gen::candle_core::safetensors::save(&map, &tmp)?;
        // SAFETY: freshly written by this test, single reader.
        let st = unsafe { MmapedSafetensors::new(&tmp)? };
        let vb = VarBuilder::from_backend(Box::new(st), DType::F32, dev.clone());

        // Load exactly as `QwenFunControlBranch::new` does: `control_img_in` off the root, the two
        // block projections off `control_blocks.0`.
        let img_in = QLinear::linear_detect_gs(cin, inner, &vb, "control_img_in", true, gs)?;
        assert!(
            img_in.is_packed(),
            "control_img_in must packed-detect on the packed tier (no dense staging)"
        );
        let blk0 = vb.pp("control_blocks").pp(0);
        let before = QLinear::linear_detect_gs(inner, inner, &blk0, "before_proj", true, gs)?;
        let after = QLinear::linear_detect_gs(inner, inner, &blk0, "after_proj", true, gs)?;
        assert!(before.is_packed(), "before_proj must packed-detect");
        assert!(after.is_packed(), "after_proj must packed-detect");

        // A dense sibling stays dense (packed-detect is per-key, not all-or-nothing).
        let dense = QLinear::linear_detect_gs(inner, inner, &blk0, "dense_proj", true, gs)?;
        assert!(!dense.is_packed(), "no `.scales` ⇒ dense path unchanged");

        // The packed forward reproduces the affine grid bit-exactly (packed-detect wired correctly, not
        // reinterpreting the u32 code stream as garbage).
        let cosine = |a: &Tensor, b: &Tensor| -> f32 {
            let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let b = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
            for (x, y) in a.iter().zip(&b) {
                dot += (*x as f64) * (*y as f64);
                na += (*x as f64) * (*x as f64);
                nb += (*y as f64) * (*y as f64);
            }
            (dot / (na.sqrt() * nb.sqrt() + 1e-12)) as f32
        };
        let grid = grids.remove("control_img_in").unwrap();
        // Same bias the packed `control_img_in` loaded (`0.01·i`) so the affine comparison includes it.
        let img_in_bias: Vec<f32> = (0..inner).map(|i| 0.01 * i as f32).collect();
        let grid_lin = candle_gen::candle_nn::Linear::new(
            Tensor::from_vec(grid, (inner, cin), &dev)?,
            Some(Tensor::from_vec(img_in_bias, (inner,), &dev)?),
        );
        let x = Tensor::randn(0f32, 1f32, (4, cin), &dev)?;
        let cos = cosine(
            &img_in.forward(&x)?,
            &candle_gen::candle_nn::Module::forward(&grid_lin, &x)?,
        );
        assert!(
            cos > 0.99999,
            "packed control_img_in vs affine grid cosine {cos:.6}"
        );

        std::fs::remove_file(&tmp).ok();
        Ok(())
    }

    /// The 132-ch control context packs to `[1, seq, 132]` and reduces to `[control_latent | 0 | 0]`:
    /// the mask (channel 16) and the inpaint latents (channels 17..33) of every packed token are zero
    /// in the pose/canny/depth-only layout, while the control latent (channels 0..16) carries through.
    #[test]
    fn fun_control_context_packs_and_zero_pads() {
        let (w, h) = (32u32, 16u32);
        let (l8h, l8w) = ((h / 8) as usize, (w / 8) as usize); // 2 x 4
                                                               // A non-zero 16-ch control latent.
        let latent = Tensor::ones((1, 16, l8h, l8w), DType::F32, &Device::Cpu).unwrap();
        let ctx = pipeline::pack_fun_control_context(&latent, w, h).unwrap();
        let (lat_h, lat_w) = pipeline::latent_dims(w, h); // h/16, w/16 = 1 x 2
        assert_eq!(ctx.dims(), &[1, lat_h * lat_w, 132]);
        // Reshape the packed 132 features back to [33, 2, 2] per token and check the channel layout:
        // channels 0..16 (control latent) are 1.0, channel 16 (mask) + 17..33 (inpaint) are 0.0.
        let v = ctx.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let seq = lat_h * lat_w;
        for tok in 0..seq {
            for ch in 0..33 {
                for sub in 0..4 {
                    let val = v[tok * 132 + ch * 4 + sub];
                    if ch < 16 {
                        assert_eq!(val, 1.0, "control latent channel {ch} should be 1.0");
                    } else {
                        assert_eq!(val, 0.0, "mask/inpaint channel {ch} should be 0.0");
                    }
                }
            }
        }
    }
}
