//! Bernini renderer knobs (the `bernini_renderer.json` sidecar the converter emits) + the guidance-mode
//! enum / resolution + the CLI-default guidance scalars. The candle sibling of
//! `mlx-gen-bernini/src/config.rs` + the renderer half of `forward.rs`'s `Mode`.

use std::path::Path;

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
    /// Read `<root>/bernini_renderer.json`; any missing field falls back to the default.
    pub fn from_dir(root: &Path) -> Self {
        let d = Self::default();
        let v: serde_json::Value = std::fs::read(root.join("bernini_renderer.json"))
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or(serde_json::Value::Null);
        let f = |k: &str, dv: f32| {
            v.get(k)
                .and_then(serde_json::Value::as_f64)
                .map(|x| x as f32)
                .unwrap_or(dv)
        };
        let b = |k: &str, dv: bool| v.get(k).and_then(serde_json::Value::as_bool).unwrap_or(dv);
        let i = |k: &str, dv: i64| v.get(k).and_then(serde_json::Value::as_i64).unwrap_or(dv);
        Self {
            switch_dit_boundary: f("switch_dit_boundary", d.switch_dit_boundary),
            shift: f("shift", d.shift),
            use_src_id_rotary_emb: b("use_src_id_rotary_emb", d.use_src_id_rotary_emb),
            interpolate_src_id: b("interpolate_src_id", d.interpolate_src_id),
            max_trained_src_id: f("max_trained_src_id", d.max_trained_src_id as f32) as f64,
            // Clamp to >=0 before the usize cast: a negative `max_sequence_length` in JSON would wrap
            // to a huge usize and drive an unbounded allocation downstream (F-080, mirrored from mlx).
            max_sequence_length: i("max_sequence_length", d.max_sequence_length as i64).max(0)
                as usize,
        }
    }
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
        let k = BerniniKnobs::from_dir(Path::new("/nonexistent"));
        assert_eq!(k.switch_dit_boundary, 0.875);
        assert_eq!(k.shift, 3.0);
        assert_eq!(k.max_trained_src_id, 5.0);
        assert_eq!(k.max_sequence_length, 512);
    }
}
