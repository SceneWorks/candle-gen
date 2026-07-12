//! Bernini renderer knobs (the `bernini_renderer.json` sidecar the converter emits) + the guidance-mode
//! enum / resolution + the CLI-default guidance scalars. The candle sibling of
//! `mlx-gen-bernini/src/config.rs` + the renderer half of `forward.rs`'s `Mode`.

use std::path::Path;

use candle_gen::gen_core::{self, GenerationRequest};
use candle_gen::{CandleError, Result as CResult};

use candle_gen_wan::config::{MAX_AREA_14B, SIZE_MULTIPLE_14B};

/// One renderer guidance mode (the renderer half of the upstream `cli.GUIDANCE_MODES`; the `*_wapg`
/// ViT-planner modes are full-Bernini only and out of scope for the renderer, sc-10995).
///
/// **Part-1 scope (sc-10994):** only the text-conditioned modes ([`Mode::T2v`], [`Mode::T2vApg`]) run
/// on candle — those are the raw caption→pixel render. The packed source-id conditioning modes
/// (`v2v`/`v2v_chain`/`v2v_apg`/`r2v_apg`/`rv2v`) additionally need the token-axis packed forward +
/// per-source RoPE on candle-gen-wan (a follow-up); [`Mode::needs_conditioning`] flags them so the
/// pipeline rejects them with an actionable message rather than silently rendering text-only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    T2v,
    T2vApg,
    V2v,
    V2vChain,
    V2vApg,
    R2vApg,
    Rv2v,
}

impl Mode {
    pub fn from_name(s: &str) -> Option<Mode> {
        Some(match s {
            "t2v" => Mode::T2v,
            "t2v_apg" => Mode::T2vApg,
            "v2v" => Mode::V2v,
            "v2v_chain" => Mode::V2vChain,
            "v2v_apg" => Mode::V2vApg,
            "r2v_apg" => Mode::R2vApg,
            "rv2v" => Mode::Rv2v,
            _ => return None,
        })
    }

    /// Whether this mode routes through APG (x-space) vs a plain weighted velocity sum.
    pub fn is_apg(self) -> bool {
        matches!(self, Mode::T2vApg | Mode::V2vApg | Mode::R2vApg)
    }

    /// Whether this mode consumes packed source-id conditioning (video/image latents) — i.e. everything
    /// except the two text-only modes. These need the candle-gen-wan packed forward (sc-10994 follow-up).
    pub fn needs_conditioning(self) -> bool {
        !matches!(self, Mode::T2v | Mode::T2vApg)
    }
}

/// Bernini renderer inference knobs, read from the converter's `bernini_renderer.json` sidecar (else
/// the upstream `BerniniRendererConfig` defaults).
#[derive(Clone, Debug)]
pub struct BerniniKnobs {
    /// High→low expert switch boundary (× `num_train_timesteps`).
    pub switch_dit_boundary: f32,
    /// UniPC flow-shift (the reference builds the scheduler with `flow_shift = config.shift`).
    pub shift: f32,
    pub use_src_id_rotary_emb: bool,
    pub interpolate_src_id: bool,
    pub max_trained_src_id: f64,
    pub max_sequence_length: usize,
}

impl Default for BerniniKnobs {
    fn default() -> Self {
        Self {
            switch_dit_boundary: 0.875,
            shift: 3.0,
            use_src_id_rotary_emb: true,
            interpolate_src_id: true,
            max_trained_src_id: 5.0,
            max_sequence_length: 512,
        }
    }
}

impl BerniniKnobs {
    /// Read `<root>/bernini_renderer.json`; any missing field falls back to the default. A **missing**
    /// sidecar is a legitimate snapshot shape (→ all defaults); a **present but malformed** sidecar is a
    /// damaged download and surfaces an error naming the file rather than silently downgrading the knobs
    /// (F-145; mirrors krea/boogu's `read_optional_json`, F-073).
    pub fn from_dir(root: &Path) -> CResult<Self> {
        let d = Self::default();
        let v = read_optional_json(
            &root.join("bernini_renderer.json"),
            "bernini_renderer knobs",
        )?
        .unwrap_or(serde_json::Value::Null);
        let f = |k: &str, dv: f32| {
            v.get(k)
                .and_then(serde_json::Value::as_f64)
                .map(|x| x as f32)
                .unwrap_or(dv)
        };
        let b = |k: &str, dv: bool| v.get(k).and_then(serde_json::Value::as_bool).unwrap_or(dv);
        let i = |k: &str, dv: i64| v.get(k).and_then(serde_json::Value::as_i64).unwrap_or(dv);
        Ok(Self {
            switch_dit_boundary: f("switch_dit_boundary", d.switch_dit_boundary),
            shift: f("shift", d.shift),
            use_src_id_rotary_emb: b("use_src_id_rotary_emb", d.use_src_id_rotary_emb),
            interpolate_src_id: b("interpolate_src_id", d.interpolate_src_id),
            max_trained_src_id: f("max_trained_src_id", d.max_trained_src_id as f32) as f64,
            // Clamp to >=0 before the usize cast: a negative `max_sequence_length` in JSON would wrap
            // to a huge usize and drive an unbounded allocation downstream (F-080, mirrored from mlx).
            max_sequence_length: i("max_sequence_length", d.max_sequence_length as i64).max(0)
                as usize,
        })
    }
}

/// Read an **optional** JSON manifest, distinguishing "genuinely absent" (→ `Ok(None)`, use the
/// documented default) from "present but corrupt" (→ `Err`, naming the file). A missing sidecar/config
/// is a legitimate snapshot shape; an I/O error or malformed JSON on a file that *is* present signals a
/// damaged/partial download that must surface rather than silently swap defaults for real config
/// (F-145; mirrors krea/boogu's `read_optional_json`, F-073). Shared by the three Bernini reads
/// (`bernini_renderer.json`, `bernini_planner.json`, `qwen2_5_vl_config.json`).
pub(crate) fn read_optional_json(path: &Path, who: &str) -> CResult<Option<serde_json::Value>> {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(ref e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => {
            return Err(CandleError::Msg(format!(
                "{who}: read {}: {e}",
                path.display()
            )))
        }
    };
    let v = serde_json::from_slice(&bytes).map_err(|e| {
        CandleError::Msg(format!(
            "{who}: parse {} (corrupt snapshot?): {e}",
            path.display()
        ))
    })?;
    Ok(Some(v))
}

/// Shared request-geometry validation for **both** Bernini entry points (the full `bernini` pipeline and
/// the `bernini_renderer`), hoisted so the full pipeline can't drift from the renderer's guards (F-095):
/// reject an explicit `steps == 0`, a `width`/`height` that is not a multiple of [`SIZE_MULTIPLE_14B`],
/// an area over [`MAX_AREA_14B`], and a `num_frames` that is not `1 + 4·k`. `id` prefixes the message so
/// each provider names itself. Without these, a 328-px request dies with an opaque shape error at the
/// first denoise step and `steps: Some(0)` is silently promoted to 1.
pub fn validate_bernini_geometry(id: &str, req: &GenerationRequest) -> gen_core::Result<()> {
    if req.steps == Some(0) {
        return Err(gen_core::Error::Msg(format!(
            "{id}: steps must be >= 1 (an explicit 0 renders undenoised noise)"
        )));
    }
    if !req.width.is_multiple_of(SIZE_MULTIPLE_14B) || !req.height.is_multiple_of(SIZE_MULTIPLE_14B)
    {
        return Err(gen_core::Error::Msg(format!(
            "{id}: width/height must be multiples of {SIZE_MULTIPLE_14B} (got {}x{})",
            req.width, req.height
        )));
    }
    let area = req.width as usize * req.height as usize;
    if area > MAX_AREA_14B {
        return Err(gen_core::Error::Msg(format!(
            "{id}: width×height ({}×{} = {area} px) exceeds the max area {MAX_AREA_14B} px \
             (704×1280); reduce the resolution",
            req.width, req.height
        )));
    }
    if let Some(f) = req.frames {
        if f == 0 || f % 4 != 1 {
            return Err(gen_core::Error::Msg(format!(
                "{id}: num_frames must be 1 + 4·k (got {f})"
            )));
        }
    }
    Ok(())
}

/// Reject a resolved-[`Mode`]/conditioning mismatch for the renderer (F-096): a mode that consumes
/// packed source conditioning ([`Mode::needs_conditioning`]) with no source present would silently
/// render text-only, and a text-only mode with sources present would VAE-encode then silently drop them.
/// `has_source` is `has_video || has_image`. Free fn so it is unit-testable without weights.
pub fn check_mode_conditioning(id: &str, mode: Mode, has_source: bool) -> gen_core::Result<()> {
    if mode.needs_conditioning() && !has_source {
        return Err(gen_core::Error::Msg(format!(
            "{id}: guidance mode {mode:?} needs source conditioning (a reference image / video clip) \
             but none was provided — pass conditioning or select a text-only mode (t2v/t2v_apg)"
        )));
    }
    if !mode.needs_conditioning() && has_source {
        return Err(gen_core::Error::Msg(format!(
            "{id}: guidance mode {mode:?} is text-only but source conditioning was provided (it would \
             be silently ignored) — select a conditioning mode or drop the conditioning"
        )));
    }
    Ok(())
}

/// CLI/gradio default guidance scalars (`bernini/cli.py add_common_args` + `run_*.sh`). A request's
/// `guidance` overrides `omega_txt`; the rest are fixed defaults until the worker surfaces them.
pub struct Defaults;
impl Defaults {
    pub const STEPS: usize = 40;
    pub const NUM_FRAMES: usize = 81;
    pub const OMEGA_VID: f32 = 1.25;
    pub const OMEGA_IMG: f32 = 4.5;
    pub const OMEGA_TXT: f32 = 4.0;
    pub const OMEGA_SCALE: f32 = 0.8;
    pub const ETA: f32 = 0.5;
    pub const MOMENTUM: f32 = 0.0;
    pub const NORM_THRESHOLD: f32 = 50.0;
    pub const FPS: u32 = 16;
}

/// Resolve the guidance [`Mode`] from the request's `video_mode` (a renderer **guidance mode** name
/// preferred, else a **task_type** name) plus which conditioning is present. With no `video_mode`,
/// default by conditioning: video+images ⇒ `rv2v`, video ⇒ `v2v_apg`, images ⇒ `v2v`, none ⇒ `t2v_apg`.
/// Byte-for-byte the mlx `resolve_mode` (so the renderer's mode dispatch stays parity-checked).
pub fn resolve_mode(video_mode: Option<&str>, has_video: bool, has_image: bool) -> Mode {
    if let Some(s) = video_mode {
        if let Some(m) = Mode::from_name(s) {
            return m;
        }
        if let Some(m) = task_to_mode(s) {
            return m;
        }
    }
    match (has_video, has_image) {
        (true, true) => Mode::Rv2v,
        (true, false) => Mode::V2vApg,
        (false, true) => Mode::V2v,
        (false, false) => Mode::T2vApg,
    }
}

/// The upstream task_type → guidance_mode table (`gradio_demo.py` RENDERER_TASK_DEFAULTS). Used as a
/// fallback when `video_mode` is a task name rather than a guidance-mode name.
fn task_to_mode(task: &str) -> Option<Mode> {
    Some(match task {
        "t2i" | "t2v" => Mode::T2vApg,
        "i2i" => Mode::V2v,
        "v2v" | "mv2v" | "ads2v" => Mode::V2vApg,
        "r2v" => Mode::R2vApg,
        "rv2v" => Mode::Rv2v,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_resolution_prefers_guidance_then_task_then_conditioning() {
        // Explicit guidance-mode name.
        assert_eq!(resolve_mode(Some("rv2v"), false, false), Mode::Rv2v);
        assert_eq!(resolve_mode(Some("t2v_apg"), false, false), Mode::T2vApg);
        // Task name fallback (t2i/t2v → t2v_apg, r2v → r2v_apg).
        assert_eq!(resolve_mode(Some("t2i"), false, false), Mode::T2vApg);
        assert_eq!(resolve_mode(Some("r2v"), false, true), Mode::R2vApg);
        // Conditioning-driven defaults.
        assert_eq!(resolve_mode(None, false, false), Mode::T2vApg);
        assert_eq!(resolve_mode(None, true, false), Mode::V2vApg);
        assert_eq!(resolve_mode(None, false, true), Mode::V2v);
        assert_eq!(resolve_mode(None, true, true), Mode::Rv2v);
        // "t2v" as a guidance-mode name is the plain mode (from_name wins over the task table).
        assert_eq!(resolve_mode(Some("t2v"), false, false), Mode::T2v);
    }

    #[test]
    fn text_modes_need_no_conditioning_the_rest_do() {
        assert!(!Mode::T2v.needs_conditioning());
        assert!(!Mode::T2vApg.needs_conditioning());
        for m in [
            Mode::V2v,
            Mode::V2vChain,
            Mode::V2vApg,
            Mode::R2vApg,
            Mode::Rv2v,
        ] {
            assert!(m.needs_conditioning(), "{m:?} consumes conditioning");
        }
        // Only the *_apg modes route through x-space APG.
        assert!(Mode::T2vApg.is_apg());
        assert!(!Mode::T2v.is_apg());
    }

    #[test]
    fn knobs_default_when_sidecar_missing() {
        // A genuinely-absent sidecar is a legitimate snapshot shape → all defaults, not an error.
        let k = BerniniKnobs::from_dir(Path::new("/nonexistent")).expect("missing sidecar is Ok");
        assert_eq!(k.switch_dit_boundary, 0.875);
        assert_eq!(k.shift, 3.0);
        assert_eq!(k.max_trained_src_id, 5.0);
        assert_eq!(k.max_sequence_length, 512);
    }

    /// F-145: a **present but malformed** sidecar surfaces an error naming the file, rather than
    /// silently falling back to defaults (which is indistinguishable from a damaged download).
    #[test]
    fn knobs_malformed_sidecar_errors() {
        let dir = std::env::temp_dir().join(format!("bernini_cfg_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bernini_renderer.json");
        std::fs::write(&path, b"{ this is not json ]").unwrap();
        let err = BerniniKnobs::from_dir(&dir).expect_err("malformed sidecar must error");
        let msg = err.to_string();
        assert!(
            msg.contains("bernini_renderer.json"),
            "names the file: {msg}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn read_optional_json_absent_vs_corrupt() {
        // Absent → Ok(None).
        assert!(read_optional_json(Path::new("/no/such/file.json"), "x")
            .unwrap()
            .is_none());
        // Present + valid → Ok(Some).
        let dir = std::env::temp_dir().join(format!("bernini_roj_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let ok = dir.join("ok.json");
        std::fs::write(&ok, br#"{"a":1}"#).unwrap();
        assert!(read_optional_json(&ok, "x").unwrap().is_some());
        // Present + corrupt → Err naming the file.
        let bad = dir.join("bad.json");
        std::fs::write(&bad, b"nope").unwrap();
        let e = read_optional_json(&bad, "who").unwrap_err().to_string();
        assert!(e.contains("bad.json") && e.contains("who"), "{e}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// F-096: a conditioning mode with no source, and a text-only mode with a source, both reject.
    #[test]
    fn mode_conditioning_mismatch_rejects() {
        // Conditioning mode, no source → error.
        assert!(check_mode_conditioning("id", Mode::V2vApg, false).is_err());
        assert!(check_mode_conditioning("id", Mode::Rv2v, false).is_err());
        // Conditioning mode with a source → ok.
        assert!(check_mode_conditioning("id", Mode::V2vApg, true).is_ok());
        // Text-only mode with no source → ok.
        assert!(check_mode_conditioning("id", Mode::T2vApg, false).is_ok());
        assert!(check_mode_conditioning("id", Mode::T2v, false).is_ok());
        // Text-only mode with a source → error (would be silently dropped).
        assert!(check_mode_conditioning("id", Mode::T2vApg, true).is_err());
    }

    /// F-095: the shared geometry guard rejects zero steps, off-grid sizes, over-area, bad frame counts.
    #[test]
    fn geometry_guard_rejects_bad_requests() {
        let base = GenerationRequest {
            prompt: "x".into(),
            width: 256,
            height: 256,
            ..Default::default()
        };
        assert!(validate_bernini_geometry("id", &base).is_ok());
        // explicit zero steps
        assert!(validate_bernini_geometry(
            "id",
            &GenerationRequest {
                steps: Some(0),
                ..base.clone()
            }
        )
        .is_err());
        // off-grid size (328 not a multiple of SIZE_MULTIPLE_14B)
        assert!(validate_bernini_geometry(
            "id",
            &GenerationRequest {
                width: 328,
                ..base.clone()
            }
        )
        .is_err());
        // over the max-area envelope
        assert!(validate_bernini_geometry(
            "id",
            &GenerationRequest {
                width: 1280,
                height: 1024,
                ..base.clone()
            }
        )
        .is_err());
        // frames not ≡ 1 (mod 4)
        assert!(validate_bernini_geometry(
            "id",
            &GenerationRequest {
                frames: Some(16),
                ..base.clone()
            }
        )
        .is_err());
        assert!(validate_bernini_geometry(
            "id",
            &GenerationRequest {
                frames: Some(17),
                ..base.clone()
            }
        )
        .is_ok());
    }
}
