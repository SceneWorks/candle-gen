//! [`PidEngine`] — the load-once, decode-many entry point a PiD-eligible provider holds (epic 7840,
//! candle dup sc-7853). It owns the heavy weights (the `PixDiT` student checkpoint + the Gemma-2
//! caption encoder) and the per-latent-space [`PidConfig`], and mints a per-generation [`PidDecoder`]
//! bound to that generation's caption + degrade σ + seed via [`PidEngine::decoder`].
//!
//! A PiD decoder is tied to a *latent space*, not a model, so the engine is parameterized by a
//! backbone tag (`"qwenimage"`, `"flux"`, …) resolved against the [`crate::registry`]. The released
//! students all share the `sr4x` `PixDiT` topology; only the LQ latent-channel count + grid
//! compression differ per space. Runs f32 throughout.

use std::path::{Path, PathBuf};

use candle_gen::candle_core::{DType, Device};
use candle_gen::gen_core::runtime::CancelFlag;
use candle_gen::gen_core::sampling::flow_capture_plan;
use candle_gen::gen_core::{GenerationRequest, PidWeights, WeightsSource};
use candle_gen::{CandleError, Result, Weights};

use crate::caption::CaptionEncoder;
use crate::config::{PidConfig, SamplerConfig};
use crate::decoder::PidDecoder;
use crate::gemma2::{Gemma2, Gemma2Config};
use crate::lq::PidNet;
use crate::registry::{lookup, BackboneSpec};
use crate::sampler::Sampler;

/// Filename of the merged Gemma-2-2b-it checkpoint inside the gemma snapshot dir; falls back to
/// loading every `*.safetensors` shard in the dir when absent.
const GEMMA_MERGED_FILE: &str = "gemma-2-2b-it.safetensors";

/// Env override for the PiD decode memory-budget ceiling (`nvidia-smi` total × 0.85 otherwise).
const PID_BUDGET_ENV: &str = "PID_DECODE_BUDGET_GIB";

/// A loaded PiD decoder engine for one latent space — built once, reused across generations.
pub struct PidEngine {
    /// The converted student checkpoint (f32), retained so [`Self::decoder`] can rebuild a [`PidNet`]
    /// per generation (cheap — candle `Tensor` clones share storage).
    weights: Weights,
    /// Per-latent-space backbone config (`sr4x` topology + the space's LQ latent-channel count).
    cfg: PidConfig,
    /// The released 4-step SDE distill sampler config.
    sampler_cfg: SamplerConfig,
    /// The Gemma-2-2b caption encoder (loaded once; the projection runs per caption).
    caption: CaptionEncoder,
    /// Key prefix for [`PidNet::from_weights`] — `""` for the converted checkpoint (the EMA export
    /// pre-strips the `net.` nesting).
    ckpt_prefix: &'static str,
}

impl PidEngine {
    /// Build from explicit paths: the converted PiD checkpoint (a single `.safetensors`), the
    /// `gemma-2-2b-it` snapshot dir (weights + `tokenizer.json`), the backbone latent-space tag
    /// (e.g. `"qwenimage"`), and the compute `device`. Errors on an unknown/out-of-scope backbone.
    pub fn load(
        checkpoint: &Path,
        gemma_dir: &Path,
        backbone: &str,
        device: &Device,
    ) -> Result<Self> {
        let spec = lookup(backbone).ok_or_else(|| {
            CandleError::Msg(format!(
                "pid: unknown/out-of-scope backbone {backbone:?} (no PiD latent-space mapping)"
            ))
        })?;
        let cfg = config_for_spec(&spec);

        // The PiD net runs f32 (the parity target + the dense-GEMM-safe path).
        let weights = Weights::from_file(checkpoint, device, DType::F32)?;

        // Gemma: prefer the merged single-file checkpoint, else load the snapshot dir's shards.
        let merged = gemma_dir.join(GEMMA_MERGED_FILE);
        let gw = if merged.is_file() {
            Weights::from_file(&merged, device, DType::F32)?
        } else {
            let files = candle_gen::sorted_safetensors(gemma_dir, "pid gemma encoder")?;
            Weights::from_files(&files, device, DType::F32)?
        };
        let gemma = Gemma2::from_weights(&gw, "model.", &Gemma2Config::gemma_2_2b())?;
        let caption = CaptionEncoder::new(gemma, gemma_dir.join("tokenizer.json"))?;

        Ok(Self {
            weights,
            cfg,
            sampler_cfg: SamplerConfig::distill_4step(),
            caption,
            ckpt_prefix: "",
        })
    }

    /// Build from a [`PidWeights`] load-spec component (the gen-core seam) for the given backbone tag
    /// on `device`. `checkpoint` must be a [`WeightsSource::File`] (the converted `.safetensors`);
    /// `gemma` must be a [`WeightsSource::Dir`] (the snapshot dir).
    pub fn from_spec(pid: &PidWeights, backbone: &str, device: &Device) -> Result<Self> {
        let checkpoint = file_path(&pid.checkpoint, "pid checkpoint")?;
        let gemma_dir = dir_path(&pid.gemma, "pid gemma encoder")?;
        Self::load(&checkpoint, &gemma_dir, backbone, device)
    }

    /// Spatial SR factor baked into the student (4× for every released backbone).
    pub fn scale(&self) -> i32 {
        self.cfg.sr_scale
    }

    /// VAE spatial compression (latent grid → pixel grid; 8 for the catalog VAEs).
    pub fn vae_compression(&self) -> i32 {
        self.cfg.latent_spatial_down_factor
    }

    /// The backbone config (`patch_size`/`hidden_size`/…) — used by the decode memory-budget guard
    /// ([`crate::budget::guard`]) at the resolve seam.
    pub fn config(&self) -> &PidConfig {
        &self.cfg
    }

    /// Mint a per-generation [`PidDecoder`] bound to one caption. `sigma` is the LQ degrade level
    /// (0 for a clean-latent decode of a fully-denoised latent); `seed` drives the sampler's noise +
    /// per-step ε. Rebuilds the [`PidNet`] from the retained weights (cheap) and encodes the caption.
    pub fn decoder(&self, caption: &str, sigma: f32, seed: u64) -> Result<PidDecoder> {
        let net = PidNet::from_weights(&self.weights, self.ckpt_prefix, &self.cfg)?;
        let caption_embs = self.caption.encode(caption)?;
        Ok(PidDecoder::new(
            net,
            Sampler::new(&self.sampler_cfg),
            caption_embs,
            sigma,
            self.cfg.sr_scale,
            self.cfg.latent_spatial_down_factor,
            seed,
        ))
    }
}

/// Resolve the decode seam for one generation (epic 7840) — the shared entry point every PiD-eligible
/// candle provider calls. It lives here in `candle-gen-pid` rather than in a provider crate because the
/// providers don't share a dependency edge (Z-Image depends on neither Qwen-Image nor FLUX), but they
/// all depend on this one.
///
/// When `req.use_pid` is set, mint a per-generation [`PidDecoder`] bound to the prompt — a **clean σ=0
/// decode of the fully-denoised latent**, seeded from `base_seed`; the caller passes it (as a
/// `&dyn LatentDecoder`) to its decode call site in place of the native VAE. Errors (rather than
/// silently falling back) if PiD was requested but the model was not loaded with `LoadSpec::pid`. When
/// the flag is unset, returns `None` and the caller uses the native VAE — the byte-exact default path.
///
/// This is the **clean σ=0** entry: it always decodes the fully-denoised latent. The `from_ldm`
/// early-stop x_t-capture (σ>0) is threaded via [`resolve_pid_decoder_at_sigma`] for wired flow-match
/// spaces; any other latent space routing through this function rejects a
/// [`pid_capture_sigma`](GenerationRequest::pid_capture_sigma) request rather than silently dropping it.
pub fn resolve_pid_decoder(
    pid: Option<&PidEngine>,
    req: &GenerationRequest,
    base_seed: u64,
    model_id: &str,
) -> Result<Option<PidDecoder>> {
    if req.use_pid && req.pid_capture_sigma.is_some() {
        return Err(CandleError::Msg(format!(
            "{model_id}: pid_capture_sigma (from_ldm early-stop) is not wired for this latent space \
             yet — the flow-match qwenimage space (Qwen-Image / Krea) uses \
             resolve_pid_decoder_at_sigma; the flux / flux2 and variance-preserving SDXL siblings are \
             follow-ons"
        )));
    }
    resolve_pid_decoder_at_sigma(pid, req, base_seed, model_id, 0.0)
}

/// `from_ldm`-aware variant of [`resolve_pid_decoder`]: mint the per-generation [`PidDecoder`] at an
/// explicit degrade `capture_sigma` (the **achieved** σ of a partially-denoised `x_k`, in the
/// flow-match frame). `0.0` reproduces the clean-latent decode. The caller is responsible for actually
/// truncating its denoise schedule to the matching step (see [`flow_capture_for_request`]), so the
/// latent it later hands to [`PidDecoder::decode`] really sits at this σ — this function only binds σ
/// into the decoder. Same `use_pid`/loaded-engine contract as [`resolve_pid_decoder`].
pub fn resolve_pid_decoder_at_sigma(
    pid: Option<&PidEngine>,
    req: &GenerationRequest,
    base_seed: u64,
    model_id: &str,
    capture_sigma: f32,
) -> Result<Option<PidDecoder>> {
    resolve_pid_decoder_for_fields(
        pid,
        req.use_pid,
        &req.prompt,
        req.count,
        req.width,
        req.height,
        &req.cancel,
        base_seed,
        model_id,
        capture_sigma,
    )
}

/// Field-parameterized sibling of [`resolve_pid_decoder_at_sigma`] for the providers that carry a
/// bespoke request type instead of a [`GenerationRequest`] (InstantID / PuLID / z-image-control /
/// sdxl-IP, plus the flux2-control / flux2-edit / kolors-control / sdxl-edit siblings). It applies the
/// **same** memory-budget guard (F-013 sc-9095) and spatial-tiling plan (sc-10087) as the registered
/// seam — the whole point of sc-11242 / F-091: before this, every bespoke `pid_decoder_for` copy minted
/// the decoder via `engine.decoder(…).with_cancel(…)` with no `budget::guard` and no `with_tiling`, so a
/// large `use_pid` decode (e.g. 1536² → 6144² at 4×) ran the whole-image forward and reproduced the CUDA
/// sysmem-fallback silent hang the guard/tiling were built to prevent. Route all such copies through
/// here so they are budgeted and tiled, and the copy-drift collapses to one implementation.
///
/// `capture_sigma` is `0.0` for a clean-latent decode (every current bespoke caller runs a full denoise
/// to σ=0). Returns `None` when `use_pid` is unset; errors loudly when `use_pid` is set but no engine is
/// loaded (never a silent VAE fallback).
#[allow(clippy::too_many_arguments)]
pub fn resolve_pid_decoder_for_fields(
    pid: Option<&PidEngine>,
    use_pid: bool,
    prompt: &str,
    count: u32,
    width: u32,
    height: u32,
    cancel: &CancelFlag,
    base_seed: u64,
    model_id: &str,
    capture_sigma: f32,
) -> Result<Option<PidDecoder>> {
    if !use_pid {
        return Ok(None);
    }
    let engine = pid.ok_or_else(|| {
        CandleError::Msg(format!(
            "{model_id}: use_pid was requested but no PiD decoder is loaded (load with LoadSpec::pid)"
        ))
    })?;
    // Memory budget + tiling (F-013 sc-9095, tiling sc-10087). PiD super-resolves in pixel space by
    // `engine.scale()`, so a `max_size`-legal `width × height` decodes at `(width·scale) ×
    // (height·scale)` — a 1536² request → 6144², whose whole-image forward exhausts VRAM (→ the CUDA
    // sysmem-fallback silent hang). We now **tile** rather than refuse: size the tile against the
    // machine's safe budget (`nvidia-smi` total × 0.85, or the `PID_DECODE_BUDGET_GIB` override) and only
    // refuse when even a minimum tile plus the resident output-resolution buffers won't fit.
    let safe_gib = candle_gen::vae_tiling::safe_budget_gib(PID_BUDGET_ENV, 0.85, 16.0);
    crate::budget::guard(
        model_id,
        count,
        width,
        height,
        engine.scale(),
        engine.config(),
        safe_gib,
    )?;
    let scale = engine.scale();
    let (th, tw) = (
        (height * scale as u32) as i32,
        (width * scale as u32) as i32,
    );
    let cfg = engine.config();
    let plan = crate::budget::plan_tile_edge(
        count.max(1) as i32,
        th,
        tw,
        cfg.patch_size,
        cfg.hidden_size,
        safe_gib,
    );
    // Thread the request's cancel flag into the minted decoder so the 4-step decode honors a cancel
    // per sampler step (F-006) — the `LatentDecoder::decode` trait signature carries no flag.
    let mut decoder = engine
        .decoder(prompt, capture_sigma, base_seed)?
        .with_cancel(cancel.clone());
    if !plan.whole_fits {
        decoder = decoder.with_tiling(plan.edge, plan.overlap);
    }
    Ok(Some(decoder))
}

/// Resolve the `from_ldm` early-stop for one **flow-match** generation: fold `req.use_pid` +
/// [`req.pid_capture_sigma`](GenerationRequest::pid_capture_sigma) together with the schedule into the
/// two values a wired site needs — the decoder's degrade σ and how many schedule entries to denoise.
///
/// Returns `(capture_sigma, keep)`: pass `capture_sigma` to [`resolve_pid_decoder_at_sigma`] and run the
/// denoise over `&sigmas[..keep]`. The clean path yields `(0.0, sigmas.len())` — the full schedule, σ=0
/// — whenever PiD is off, no capture is requested, or the requested ceiling would stop the denoise
/// at/before the img2img `start_step` (no benefit). `start_step` is `0` for txt2img / edit / control.
pub fn flow_capture_for_request(
    req: &GenerationRequest,
    sigmas: &[f32],
    start_step: usize,
) -> (f32, usize) {
    let plan = req
        .use_pid
        .then(|| flow_capture_plan(sigmas, req.pid_capture_sigma))
        .flatten();
    match plan {
        Some(c) if c.keep > start_step + 1 => (c.sigma, c.keep),
        _ => (0.0, sigmas.len()),
    }
}

/// Assemble the per-latent-space [`PidConfig`] for a resolved backbone spec. The released students
/// share the `sr4x` PixDiT topology; only the LQ latent-channel count, the latent grid's spatial
/// compression, and the SR scale differ per latent space (16-ch/8× for qwen/flux/sd3, 4-ch/8× for
/// sdxl, 128-ch/16× for flux2 — see the registry `FLUX2` note, sc-7847).
///
/// `sr_scale` is threaded from [`BackboneSpec::pid_scale`] (F-141, sc-11246) rather than left at the
/// hard-coded `sr4x()` `4`. Every released student is 4× — the value `sr4x()` already defaults to — so
/// this is a byte-for-byte no-op today; but the registry deliberately carries `pid_scale` per space, and
/// an 8× student would otherwise silently size its noise, output geometry, and the budget guard for 4×
/// (a wrong-resolution decode, not a load error). `sr_scale` flows into the decoder's output resolution
/// (`zH·vae_compression·scale`) and the LQ upsample ratio (`(sr_scale·lsdf)/patch`).
fn config_for_spec(spec: &BackboneSpec) -> PidConfig {
    let mut cfg = PidConfig::sr4x();
    cfg.lq_latent_channels = spec.latent_channels;
    cfg.latent_spatial_down_factor = spec.latent_spatial_down_factor;
    cfg.sr_scale = spec.pid_scale;
    cfg
}

/// Extract the single-file path from a [`WeightsSource`], rejecting a directory.
fn file_path(src: &WeightsSource, what: &str) -> Result<PathBuf> {
    match src {
        WeightsSource::File(p) => Ok(p.clone()),
        WeightsSource::Dir(_) => Err(CandleError::Msg(format!(
            "{what}: expected the converted .safetensors file, got a directory"
        ))),
    }
}

/// Extract the directory path from a [`WeightsSource`], rejecting a single file.
fn dir_path(src: &WeightsSource, what: &str) -> Result<PathBuf> {
    match src {
        WeightsSource::Dir(p) => Ok(p.clone()),
        WeightsSource::File(_) => Err(CandleError::Msg(format!(
            "{what}: expected a snapshot directory, got a single file"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn err_string<T>(r: Result<T>) -> String {
        match r {
            Ok(_) => panic!("expected an error"),
            Err(e) => e.to_string(),
        }
    }

    // --- F-141 (sc-11246): sr_scale is threaded from BackboneSpec.pid_scale --------------------------

    #[test]
    fn config_threads_pid_scale_and_default_is_identity() {
        // Every released student ships pid_scale=4, the same value sr4x() hard-codes — so threading is a
        // byte-for-byte no-op today: the assembled config must equal PidConfig::sr4x() with only the two
        // pre-existing per-space fields (latent channels / grid compression) diverging, never sr_scale.
        let baseline = PidConfig::sr4x();
        for backbone in ["qwenimage", "flux", "sd3", "sdxl", "flux2"] {
            let spec = lookup(backbone).unwrap();
            let cfg = config_for_spec(&spec);
            assert_eq!(
                cfg.sr_scale, spec.pid_scale,
                "{backbone}: sr_scale must come from the spec, not the hard-coded 4"
            );
            assert_eq!(
                cfg.sr_scale, baseline.sr_scale,
                "{backbone}: released students are all 4× — threading reproduces current behavior"
            );
            assert_eq!(cfg.lq_latent_channels, spec.latent_channels);
            assert_eq!(
                cfg.latent_spatial_down_factor,
                spec.latent_spatial_down_factor
            );
        }
    }

    #[test]
    fn nondefault_pid_scale_changes_decode_geometry() {
        // A hypothetical 8× student: threading pid_scale sizes the decode for 8×, not the hard-coded 4×.
        // sr_scale drives the decoder's output resolution (target_hw = zH·vae_compression·scale) and the
        // LQ upsample ratio ((sr_scale·lsdf)/patch), so a different pid_scale is a materially different
        // decode — proven here on the geometry those consumers compute from sr_scale.
        let mut spec = lookup("flux").unwrap();
        spec.pid_scale = 8;
        let cfg8 = config_for_spec(&spec);
        assert_eq!(
            cfg8.sr_scale, 8,
            "non-default pid_scale must be threaded through"
        );

        spec.pid_scale = 4;
        let cfg4 = config_for_spec(&spec);
        assert_ne!(
            cfg8.sr_scale, cfg4.sr_scale,
            "a non-default pid_scale yields a different sr_scale"
        );
        // Output pixel side for a fixed latent grid side `zside` differs between the two scales.
        let zside = 32;
        let out8 = zside * cfg8.latent_spatial_down_factor * cfg8.sr_scale;
        let out4 = zside * cfg4.latent_spatial_down_factor * cfg4.sr_scale;
        assert_ne!(out8, out4, "output geometry tracks sr_scale");
    }

    #[test]
    fn unknown_backbone_errors() {
        let err = err_string(PidEngine::load(
            Path::new("/nonexistent/ckpt.safetensors"),
            Path::new("/nonexistent/gemma"),
            "dinov2", // out-of-scope (vision-encoder latent, not a VAE latent)
            &Device::Cpu,
        ));
        assert!(err.contains("out-of-scope backbone"), "got: {err}");
    }

    #[test]
    fn from_spec_rejects_swapped_sources() {
        // checkpoint must be a File, gemma must be a Dir — a swap is rejected before any load.
        let swapped = PidWeights {
            checkpoint: WeightsSource::Dir("/nonexistent/ckpt".into()),
            gemma: WeightsSource::Dir("/nonexistent/gemma".into()),
        };
        let err = err_string(PidEngine::from_spec(&swapped, "qwenimage", &Device::Cpu));
        assert!(err.contains("converted .safetensors file"), "got: {err}");
    }

    #[test]
    fn resolve_pid_decoder_off_is_none() {
        let req = GenerationRequest {
            prompt: "a fox".into(),
            ..Default::default()
        };
        assert!(resolve_pid_decoder(None, &req, 0, "some_model")
            .unwrap()
            .is_none());
    }

    #[test]
    fn resolve_pid_decoder_requested_without_engine_errors() {
        let req = GenerationRequest {
            prompt: "a fox".into(),
            use_pid: true,
            ..Default::default()
        };
        let err = err_string(resolve_pid_decoder(None, &req, 0, "some_model"));
        assert!(err.contains("no PiD decoder is loaded"), "got: {err}");
    }

    #[test]
    fn resolve_pid_decoder_rejects_capture_sigma_for_unwired_space() {
        let req = GenerationRequest {
            prompt: "a fox".into(),
            use_pid: true,
            pid_capture_sigma: Some(0.2),
            ..Default::default()
        };
        let err = err_string(resolve_pid_decoder(None, &req, 0, "flux"));
        assert!(
            err.contains("not wired for this latent space"),
            "got: {err}"
        );
    }

    #[test]
    fn resolve_pid_decoder_ignores_capture_sigma_when_pid_off() {
        let req = GenerationRequest {
            prompt: "a fox".into(),
            use_pid: false,
            pid_capture_sigma: Some(0.2),
            ..Default::default()
        };
        assert!(resolve_pid_decoder(None, &req, 0, "flux")
            .unwrap()
            .is_none());
    }

    // --- Field-parameterized shared seam (sc-11242 / F-091) -----------------------------------------
    // The bespoke providers (InstantID / PuLID / z-image-control / sdxl-IP, and the flux2-control /
    // flux2-edit / kolors-control / sdxl-edit siblings) now funnel their `pid_decoder_for` through
    // `resolve_pid_decoder_for_fields`, which applies the same budget guard + tiling as the registered
    // seam. Their per-lane `use_pid` decodes therefore honor the guard/tiling exactly like the guarded
    // path — the numeric guard/plan behavior itself is covered by the `crate::budget` tests. These
    // assert the shared funnel's off/loaded-engine contract that every bespoke lane inherits.

    #[test]
    fn field_helper_off_is_none() {
        // use_pid unset → native VAE (None), regardless of size — every bespoke lane inherits this.
        assert!(resolve_pid_decoder_for_fields(
            None,
            false,
            "a fox",
            1,
            1536,
            1536,
            &CancelFlag::default(),
            0,
            "instantid",
            0.0,
        )
        .unwrap()
        .is_none());
    }

    #[test]
    fn field_helper_requested_without_engine_errors() {
        // use_pid set but no engine loaded → the shared loud refusal (never a silent VAE fallback).
        let err = err_string(resolve_pid_decoder_for_fields(
            None,
            true,
            "a fox",
            1,
            1536,
            1536,
            &CancelFlag::default(),
            0,
            "pulid",
            0.0,
        ));
        assert!(err.contains("no PiD decoder is loaded"), "got: {err}");
    }

    // Reachability guard for F-091: `resolve_pid_decoder_at_sigma` (the registered seam) and the
    // field helper share one implementation, so a bespoke lane cannot drift back to an unguarded
    // whole-image decode without also breaking the registered path. This asserts the delegation stays
    // wired — `resolve_pid_decoder_at_sigma` must agree with the field helper on the off/error contract.
    #[test]
    fn request_seam_delegates_to_field_helper() {
        let req = GenerationRequest {
            prompt: "a fox".into(),
            use_pid: true,
            width: 1536,
            height: 1536,
            count: 1,
            ..Default::default()
        };
        let via_req = err_string(resolve_pid_decoder_at_sigma(
            None,
            &req,
            0,
            "sdxl ip-adapter",
            0.0,
        ));
        let via_fields = err_string(resolve_pid_decoder_for_fields(
            None,
            true,
            &req.prompt,
            req.count,
            req.width,
            req.height,
            &req.cancel,
            0,
            "sdxl ip-adapter",
            0.0,
        ));
        assert_eq!(via_req, via_fields);
    }
}
