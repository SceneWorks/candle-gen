//! Shared test-support helpers (sc-9055 / F-069) — the single home for the PPM read/write, cosine,
//! env-path, GPU peak-VRAM, and HF-Hub-cache resolution helpers that had been hand-copied into ~16
//! `#[cfg(test)]` validation modules across the provider crates.
//!
//! Why one home: the copies had already **drifted** — two PPM header tokenizers (one comment-tolerant,
//! one not), an f32- vs f64-accumulating cosine, and (F-071 / sc-9057) HF-cache resolvers that variously
//! did or did not honour `$HF_HOME`. A comment-bearing PPM passed some harnesses and failed others, and a
//! methodology fix had to be mirrored by hand. Concentrating them here makes the behaviour canonical.
//!
//! Behaviour is preserved for every caller:
//! * [`read_ppm`] is the **comment-tolerant** tokenizer (a strict superset — the non-tolerant callers only
//!   ever read comment-free `P6` files written by [`write_ppm`], for which the two agree byte-for-byte).
//! * [`cosine`] is the full normalized cosine (`0.0` when either input is the zero vector); [`cosine_dot`]
//!   is the bare dot product for callers whose inputs are already L2-normalized (SDXL/Kolors/FLUX IP).
//! * [`hf_snapshot_dir`] resolves the HF Hub cache in the canonical order `$HF_HUB_CACHE` → `$HF_HOME/hub`
//!   → `<home>/.cache/huggingface/hub` (`USERPROFILE` on Windows, then `HOME`), matching the sc-9057 fix.
//!
//! Gated behind the crate `testkit` feature so this test-only surface (and its `std::process` /
//! `nvidia-smi` dependency) never compiles into a production build. Provider crates enable it as a
//! dev-dependency feature: `candle-gen = { path = "...", features = ["testkit"] }` under
//! `[dev-dependencies]`, or `candle-gen/testkit` in a test-only feature.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

use crate::gen_core::Image;

// ---------------------------------------------------------------------------------------------------
// Env paths
// ---------------------------------------------------------------------------------------------------

/// A required env-var path for an opt-in real-weight test. Panics with a clear message if unset —
/// these tests are `#[ignore]`d and only run when the caller exports the env var.
pub fn env_path(key: &str) -> PathBuf {
    PathBuf::from(std::env::var(key).unwrap_or_else(|_| {
        panic!("set ${key} (see the test module docs for the real-weight run)")
    }))
}

/// An optional env-var path — `None` when unset (for tests that skip gracefully rather than panic).
pub fn env_path_opt(key: &str) -> Option<PathBuf> {
    std::env::var(key).ok().map(PathBuf::from)
}

// ---------------------------------------------------------------------------------------------------
// PPM image IO (codec-less — the harnesses own their image IO)
// ---------------------------------------------------------------------------------------------------

/// Minimal binary-`P6` PPM reader — `P6 <w> <h> <maxval>` header then `w*h*3` raw RGB bytes. Tolerant
/// of a single (or several) `#`-comment line and arbitrary header whitespace; enough for hand-prepared
/// reference images (the `image` dep in these crates is built codec-less).
///
/// This is the comment-tolerant tokenizer; on the comment-free files [`write_ppm`] produces it agrees
/// byte-for-byte with the older whitespace-only readers it replaced.
pub fn read_ppm(path: &Path) -> Image {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let mut i = 0usize;
    let mut tok = || -> String {
        // skip whitespace + comment lines
        loop {
            while i < bytes.len() && bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            if i < bytes.len() && bytes[i] == b'#' {
                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                }
            } else {
                break;
            }
        }
        let start = i;
        while i < bytes.len() && !bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        String::from_utf8_lossy(&bytes[start..i]).to_string()
    };
    assert_eq!(tok(), "P6", "{} is not a binary (P6) PPM", path.display());
    let w: usize = tok().parse().expect("ppm width");
    let h: usize = tok().parse().expect("ppm height");
    let _max: usize = tok().parse().expect("ppm maxval");
    i += 1; // single whitespace after maxval, before the pixel block
    let pixels = bytes[i..i + w * h * 3].to_vec();
    Image {
        width: w as u32,
        height: h as u32,
        pixels,
    }
}

/// Write a binary-`P6` PPM (`P6\n<w> <h>\n255\n<rgb bytes>`). Convert to PNG out-of-band for viewing.
pub fn write_ppm(path: &Path, img: &Image) {
    let mut out = format!("P6\n{} {}\n255\n", img.width, img.height).into_bytes();
    out.extend_from_slice(&img.pixels);
    std::fs::write(path, out).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
}

/// Mean absolute per-byte difference between two equal-size RGB renders (the
/// injection-changes-the-output sanity metric). Panics on a size mismatch.
pub fn mean_abs_diff(a: &Image, b: &Image) -> f32 {
    assert_eq!(a.pixels.len(), b.pixels.len(), "render size mismatch");
    let sum: u64 = a
        .pixels
        .iter()
        .zip(&b.pixels)
        .map(|(x, y)| (*x as i32 - *y as i32).unsigned_abs() as u64)
        .sum();
    sum as f32 / a.pixels.len() as f32
}

// ---------------------------------------------------------------------------------------------------
// Cosine similarity
// ---------------------------------------------------------------------------------------------------

/// Cosine similarity of two equal-length embeddings, normalizing internally (inputs need NOT be
/// pre-normalized). Returns `0.0` when either input is the zero vector.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

/// Bare dot product of two equal-length vectors — the cosine metric when both inputs are already
/// L2-normalized (the SDXL / Kolors / FLUX IP-adapter feature extractors normalize before comparing).
pub fn cosine_dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

// ---------------------------------------------------------------------------------------------------
// HF Hub cache resolution (F-071 / sc-9057 — honour $HF_HOME, not just the Unix ~/.cache default)
// ---------------------------------------------------------------------------------------------------

/// The candidate HF Hub cache roots, in resolution order: `$HF_HUB_CACHE`, then `$HF_HOME/hub`, then the
/// user-home `.cache/huggingface/hub` default (`USERPROFILE` on Windows, then `HOME`).
///
/// The Windows-primary dev box keeps the cache at `D:\.cache\huggingface` via `HF_HOME`, where `HOME` is
/// usually unset — resolvers that only consulted `$HOME/.cache/huggingface` silently missed it (F-071).
pub fn hf_cache_roots() -> Vec<PathBuf> {
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Ok(c) = std::env::var("HF_HUB_CACHE") {
        roots.push(PathBuf::from(c));
    }
    if let Ok(h) = std::env::var("HF_HOME") {
        roots.push(PathBuf::from(h).join("hub"));
    }
    for home_var in ["USERPROFILE", "HOME"] {
        if let Ok(home) = std::env::var(home_var) {
            roots.push(PathBuf::from(home).join(".cache/huggingface/hub"));
        }
    }
    roots
}

/// Resolve the first existing `snapshots/<rev>/` directory for an HF repo under the [`hf_cache_roots`],
/// or `None` if the repo isn't cached anywhere. `repo` is the `owner/name` form (e.g.
/// `"openai/clip-vit-large-patch14"`) — it is normalized to the `models--owner--name` cache dir.
pub fn hf_snapshot_dir(repo: &str) -> Option<PathBuf> {
    let repo_dir = format!("models--{}", repo.replace('/', "--"));
    for snapshots in hf_cache_roots()
        .into_iter()
        .map(|r| r.join(&repo_dir).join("snapshots"))
    {
        let Ok(revs) = std::fs::read_dir(&snapshots) else {
            continue;
        };
        if let Some(dir) = revs
            .filter_map(std::result::Result::ok)
            .map(|e| e.path())
            .find(|p| p.is_dir())
        {
            return dir.into();
        }
    }
    None
}

/// [`hf_snapshot_dir`] that panics with an actionable message if the repo isn't cached — for tests that
/// require the weights present.
pub fn require_hf_snapshot_dir(repo: &str) -> PathBuf {
    hf_snapshot_dir(repo).unwrap_or_else(|| {
        panic!(
            "{repo} snapshot not cached under any HF cache root \
             (HF_HUB_CACHE / HF_HOME/hub / <home>/.cache/huggingface/hub)"
        )
    })
}

// ---------------------------------------------------------------------------------------------------
// GPU peak-VRAM sampler (device-level `nvidia-smi memory.used`) — used by the video-VAE decode sweeps
// ---------------------------------------------------------------------------------------------------

pub use gpu_peak::{used_mib, PeakSampler};
pub use vram_probe::{VramProbe, VramReport};

mod vram_probe {
    //! sc-9094 — the per-tier VRAM measuring harness (epic 9083's packed-load rollout). Wraps the
    //! device-level [`PeakSampler`] into the three phase quantities the manifest's per-variant
    //! `minMemoryGb` gate is derived from:
    //!
    //! * **load peak** — the transient high-water mark *during* model load (weights → device,
    //!   packed-repack, CPU-staging). For flux2-dev this is the headline: the dense CPU-stage path
    //!   peaked ~105 GB; the packed Q4 load lands the quantized footprint on-device directly.
    //! * **steady resident** — device VRAM after load settles, *before* denoise — the persistent
    //!   weight + component footprint a job holds for its whole lifetime.
    //! * **overall peak** — the max across the whole generate (load + denoise + VAE decode). This is
    //!   the number the card must physically hold; `minMemoryGb` = this + headroom.
    //!
    //! All three are **device-level** `nvidia-smi memory.used` deltas over a recorded `baseline`
    //! (WDDM reports per-process `used_memory` as `[N/A]`, and the card must fit the *whole* device's
    //! resident bytes anyway). Run on an otherwise-idle GPU; the report prints the baseline so a
    //! non-zero pre-run residency is visible. The sampler is an in-process helper thread (part of the
    //! measurement, not a background job) polling every ~40 ms — fast enough to catch a multi-hundred-ms
    //! load/decode transient.
    //!
    //! Usage from a provider example (load and generate are separate phases, so their peaks separate):
    //! ```ignore
    //! let mut probe = VramProbe::start(0);          // records the idle baseline
    //! let load = probe.phase();                     // sample across load
    //! let gen = registry::load(id, &spec)?;         //   ... weights → device ...
    //! probe.end_load(load);                         // load peak recorded; steady sampled now
    //! let run = probe.phase();                      // sample across generate
    //! let out = gen.generate(&req, &mut cb)?;       //   ... denoise + decode ...
    //! probe.end_gen(run);                           // overall peak recorded
    //! println!("{}", probe.report());               // load-peak / steady / overall-peak (GB)
    //! ```

    use super::gpu_peak::{used_mib, PeakSampler};

    /// MiB → GB (10⁹ bytes — the manifest's `minMemoryGb` is base-10 GB, matching the MLX footprint
    /// numbers). `1 MiB = 2²⁰ bytes`.
    fn mib_to_gb(mib: u64) -> f64 {
        (mib as f64) * (1024.0 * 1024.0) / 1.0e9
    }

    /// The three phase quantities (GB) plus the idle baseline they were measured over.
    #[derive(Clone, Copy, Debug)]
    pub struct VramReport {
        /// Device VRAM already resident before the run (GB) — should be ~0 on an idle GPU.
        pub baseline_gb: f64,
        /// Transient high-water mark during load, over baseline (GB).
        pub load_peak_gb: f64,
        /// Resident VRAM after load settles, before denoise, over baseline (GB).
        pub steady_gb: f64,
        /// Max over the whole generate (load + denoise + decode), over baseline (GB).
        pub peak_gb: f64,
    }

    impl std::fmt::Display for VramReport {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(
                f,
                "load-peak {:.1} GB | steady {:.1} GB | overall-peak {:.1} GB (baseline {:.1} GB)",
                self.load_peak_gb, self.steady_gb, self.peak_gb, self.baseline_gb
            )
        }
    }

    /// A phase-scoped [`PeakSampler`] the caller starts around a load or generate call and hands back
    /// to the matching `end_*` to fold its peak into the report.
    pub struct Phase(PeakSampler);

    /// The per-run VRAM probe. [`start`](Self::start) records the idle baseline; each phase is bracketed
    /// by [`phase`](Self::phase) → the work → `end_load` / `end_gen`.
    pub struct VramProbe {
        gpu: usize,
        baseline_mib: u64,
        load_peak_mib: u64,
        steady_mib: u64,
        overall_peak_mib: u64,
    }

    impl VramProbe {
        /// Record the idle device baseline (used MiB) for GPU ordinal `gpu`. A missing `nvidia-smi`
        /// yields a 0 baseline (the deltas then read as absolute used, still monotonically useful).
        pub fn start(gpu: usize) -> Self {
            let baseline = used_mib(gpu).unwrap_or(0);
            Self {
                gpu,
                baseline_mib: baseline,
                load_peak_mib: baseline,
                steady_mib: baseline,
                overall_peak_mib: baseline,
            }
        }

        /// Begin sampling a phase (load or generate). Keep the returned [`Phase`] alive across the work
        /// and pass it to the matching `end_*`.
        pub fn phase(&self) -> Phase {
            Phase(PeakSampler::start(self.gpu))
        }

        /// Close the **load** phase: fold its peak into `load_peak`, and sample the settled resident
        /// (`steady`) right now (load done, denoise not started). Also seeds the overall peak.
        pub fn end_load(&mut self, phase: Phase) {
            let load_peak = phase.0.stop();
            self.load_peak_mib = self.load_peak_mib.max(load_peak);
            self.overall_peak_mib = self.overall_peak_mib.max(load_peak);
            // Steady = the instantaneous resident after load, before any denoise allocation.
            if let Some(m) = used_mib(self.gpu) {
                self.steady_mib = m;
                self.overall_peak_mib = self.overall_peak_mib.max(m);
            }
        }

        /// Close the **generate** phase: fold its peak into the overall peak.
        pub fn end_gen(&mut self, phase: Phase) {
            let gen_peak = phase.0.stop();
            self.overall_peak_mib = self.overall_peak_mib.max(gen_peak);
        }

        /// The three phase quantities in GB, over the idle baseline (clamped at 0 — a slightly-lower
        /// late sample must not read as negative usage).
        pub fn report(&self) -> VramReport {
            let over = |m: u64| mib_to_gb(m.saturating_sub(self.baseline_mib));
            VramReport {
                baseline_gb: mib_to_gb(self.baseline_mib),
                load_peak_gb: over(self.load_peak_mib),
                steady_gb: over(self.steady_mib),
                peak_gb: over(self.overall_peak_mib),
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn mib_to_gb_is_base10_gb() {
            // 1024 MiB = 2³⁰ bytes ≈ 1.0737 GB (base-10).
            assert!((mib_to_gb(1024) - 1.0737).abs() < 1e-3);
        }

        #[test]
        fn report_is_delta_over_baseline_and_nonnegative() {
            // A probe with a synthetic baseline: deltas subtract it, and a below-baseline sample
            // clamps to 0 rather than going negative.
            let mut p = VramProbe {
                gpu: 0,
                baseline_mib: 1000,
                load_peak_mib: 5000,
                steady_mib: 3000,
                overall_peak_mib: 6000,
            };
            let r = p.report();
            assert!((r.load_peak_gb - mib_to_gb(4000)).abs() < 1e-6);
            assert!((r.steady_gb - mib_to_gb(2000)).abs() < 1e-6);
            assert!((r.peak_gb - mib_to_gb(5000)).abs() < 1e-6);
            // A late sample below baseline must not underflow.
            p.steady_mib = 500;
            assert_eq!(p.report().steady_gb, 0.0);
        }
    }
}

mod gpu_peak {
    //! sc-7148 — shared `nvidia-smi` peak-VRAM sampler for the video-VAE decode sweeps. Polls
    //! device-level `memory.used` in a background thread and tracks the max while a decode runs.
    //!
    //! Device-level (not per-process) is deliberate: Windows WDDM reports per-process `used_memory` as
    //! `[N/A]`, and the budgeted decode's safe ceiling is *total* VRAM × 0.85, so the honest "did it
    //! fit" quantity is the whole device's used bytes during the decode. Run the sweep on an
    //! otherwise-idle GPU (the harness prints the pre-decode `baseline` so you can confirm nothing else
    //! was resident).

    use std::process::Command;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::Arc;
    use std::thread::{self, JoinHandle};
    use std::time::Duration;

    /// nvidia-smi poll cadence. ~40 ms is well under a multi-second VAE decode (so the peak is
    /// captured) while keeping the subprocess-spawn overhead negligible.
    const POLL: Duration = Duration::from_millis(40);

    /// Device-level used VRAM (MiB) for GPU ordinal `gpu` via `nvidia-smi`, or `None` if the query
    /// fails.
    pub fn used_mib(gpu: usize) -> Option<u64> {
        let out = Command::new("nvidia-smi")
            .args([
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                "-i",
                &gpu.to_string(),
            ])
            .output()
            .ok()?;
        if !out.status.success() {
            return None;
        }
        String::from_utf8_lossy(&out.stdout)
            .lines()
            .next()?
            .trim()
            .parse::<u64>()
            .ok()
    }

    /// Background sampler: from [`PeakSampler::start`] until [`PeakSampler::stop`], polls
    /// `used_mib(gpu)` every [`POLL`] and keeps the running max (MiB).
    pub struct PeakSampler {
        stop: Arc<AtomicBool>,
        peak: Arc<AtomicU64>,
        handle: Option<JoinHandle<()>>,
    }

    impl PeakSampler {
        pub fn start(gpu: usize) -> Self {
            let stop = Arc::new(AtomicBool::new(false));
            let peak = Arc::new(AtomicU64::new(0));
            let (s, p) = (stop.clone(), peak.clone());
            let handle = thread::spawn(move || {
                while !s.load(Ordering::Relaxed) {
                    if let Some(m) = used_mib(gpu) {
                        p.fetch_max(m, Ordering::Relaxed);
                    }
                    thread::sleep(POLL);
                }
                // One last sample after the stop signal — the true peak may land in the final window.
                if let Some(m) = used_mib(gpu) {
                    p.fetch_max(m, Ordering::Relaxed);
                }
            });
            Self {
                stop,
                peak,
                handle: Some(handle),
            }
        }

        /// Signal the sampler thread to stop, join it, and return the peak used VRAM (MiB).
        pub fn stop(mut self) -> u64 {
            self.stop.store(true, Ordering::Relaxed);
            if let Some(h) = self.handle.take() {
                let _ = h.join();
            }
            self.peak.load(Ordering::Relaxed)
        }
    }
}
