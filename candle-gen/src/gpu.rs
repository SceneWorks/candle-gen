//! Shared VRAM-budget probe: the trusted-path `nvidia-smi` resolver used by the video-VAE decode
//! tilers (`candle-gen-seedvr2`, `candle-gen-wan`, `candle-gen-ltx`).
//!
//! # Why a resolver, not a bare `Command::new("nvidia-smi")` (sc-9014 / F-030)
//!
//! The budget probes used to spawn `Command::new("nvidia-smi")` **unqualified**. On Windows,
//! `CreateProcessW` searches the application directory (and the legacy current directory) *before*
//! `PATH`, so a `nvidia-smi.exe` planted next to the worker binary — or anywhere earlier on a
//! hijacked `PATH` — would run with the worker's privileges and silently rewrite the VRAM budget.
//! Because the probe runs on **every** budgeted decode, that is a repeatable, hard-to-observe
//! injection point.
//!
//! This module resolves `nvidia-smi` from a **trusted absolute location** (System32 /
//! `CUDA_PATH\bin` on Windows, the standard `/usr/bin` etc. on Unix) once, caches the result in a
//! [`OnceLock`], and only ever executes that absolute path. If no trusted binary is found the probe
//! degrades cleanly (returns `None` → the caller falls back to its env override / conservative
//! default) rather than executing whatever happens to be first on `PATH`.
//!
//! The whole query stays **best-effort**: any failure (no CUDA, sandboxed, non-zero exit, unparsable
//! output) yields `None`, exactly matching the previous behaviour when `nvidia-smi` was absent.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

/// Cached resolution of the trusted `nvidia-smi` absolute path. `None` = not found at any trusted
/// location on this host (so the probe degrades to the caller's fallback). Resolved once per process.
static NVIDIA_SMI_PATH: OnceLock<Option<PathBuf>> = OnceLock::new();

/// Candidate **absolute** locations for `nvidia-smi`, in priority order, for the current OS. Only
/// these trusted directories are ever considered — a bare `PATH` lookup is deliberately NOT among
/// them (that is the F-030 hijack vector). The CUDA-toolkit `bin` dirs are included because a CUDA
/// dev/deploy box always has the driver utilities there even if System32 somehow does not.
fn candidate_paths() -> Vec<PathBuf> {
    let mut out = Vec::new();

    #[cfg(windows)]
    {
        // The driver installs nvidia-smi.exe into System32 on every NVIDIA Windows host — the
        // canonical trusted location. `%SystemRoot%` (usually C:\Windows) is not attacker-writable.
        if let Some(root) = std::env::var_os("SystemRoot") {
            out.push(Path::new(&root).join("System32").join("nvidia-smi.exe"));
        }
        // CUDA toolkit bin (dev boxes): CUDA_PATH\bin\nvidia-smi.exe.
        if let Some(cuda) = std::env::var_os("CUDA_PATH") {
            out.push(Path::new(&cuda).join("bin").join("nvidia-smi.exe"));
        }
    }

    #[cfg(not(windows))]
    {
        // The NVIDIA Linux driver installs nvidia-smi into /usr/bin (some distros /usr/local/bin).
        out.push(PathBuf::from("/usr/bin/nvidia-smi"));
        out.push(PathBuf::from("/usr/local/bin/nvidia-smi"));
        // CUDA toolkit bin, if the env points at it.
        if let Some(cuda) = std::env::var_os("CUDA_PATH") {
            out.push(Path::new(&cuda).join("bin").join("nvidia-smi"));
        }
    }

    out
}

/// Pick the first candidate that exists on disk. Split out so it is unit-testable without touching
/// the real host: [`resolve_nvidia_smi`] wraps this over [`candidate_paths`].
fn first_existing(candidates: &[PathBuf]) -> Option<PathBuf> {
    candidates.iter().find(|p| p.exists()).cloned()
}

/// The trusted absolute `nvidia-smi` path for this host, or `None` if none of the trusted locations
/// contains it. Cached in a [`OnceLock`] — resolved (and the filesystem probed) at most once.
pub fn resolve_nvidia_smi() -> Option<&'static Path> {
    NVIDIA_SMI_PATH
        .get_or_init(|| first_existing(&candidate_paths()))
        .as_deref()
}

/// Total VRAM (GiB) of the smallest visible CUDA device, read from a **trusted absolute**
/// `nvidia-smi` (never a bare `PATH` lookup — see the module note / sc-9014). The MIN across devices
/// is conservative on a heterogeneous box. `None` when no trusted `nvidia-smi` exists or the query
/// fails — the caller then falls back to its env override / conservative default.
///
/// This is the shared implementation the SeedVR2 / Wan / LTX budget probes route through (the F-030
/// de-dup that had been tracked as a follow-up).
pub fn nvidia_smi_min_total_gib() -> Option<f64> {
    let exe = resolve_nvidia_smi()?;
    let out = Command::new(exe)
        .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    parse_min_total_gib(&String::from_utf8_lossy(&out.stdout))
}

/// Parse the `--query-gpu=memory.total --format=csv,noheader,nounits` output (one MiB value per
/// line, one line per GPU) into the MIN total across GPUs, in GiB. Split out for unit testing.
fn parse_min_total_gib(text: &str) -> Option<f64> {
    let min_mb = text
        .lines()
        .filter_map(|line| line.trim().parse::<f64>().ok())
        .filter(|&mb| mb > 0.0)
        .fold(f64::INFINITY, f64::min);
    min_mb.is_finite().then_some(min_mb / 1024.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidate_paths_are_all_absolute_and_trusted() {
        // Every candidate must be an ABSOLUTE path (the whole point of F-030): a relative or bare
        // name would reintroduce the process-search-order hijack. There must also be at least one
        // candidate on a normal host (SystemRoot on Windows / /usr/bin on Unix are always present).
        let cands = candidate_paths();
        assert!(!cands.is_empty(), "expected at least one trusted candidate");
        for p in &cands {
            assert!(p.is_absolute(), "candidate must be absolute, got {p:?}");
            assert!(
                p.file_name()
                    .is_some_and(|n| n.to_string_lossy().starts_with("nvidia-smi")),
                "candidate must end in the nvidia-smi binary, got {p:?}"
            );
        }
    }

    #[test]
    fn first_existing_picks_the_first_present_path() {
        // With a made-up non-existent dir first and a guaranteed-existent one second, the resolver
        // must skip the missing one and return the real one — proving it validates on disk rather
        // than blindly returning candidate[0] (or falling through to PATH).
        let missing = PathBuf::from(if cfg!(windows) {
            r"C:\definitely\not\here\nvidia-smi.exe"
        } else {
            "/definitely/not/here/nvidia-smi"
        });
        // A path that is guaranteed to exist on the test host.
        let present = std::env::current_exe().expect("current exe path");
        let picked = first_existing(&[missing.clone(), present.clone()]);
        assert_eq!(picked.as_deref(), Some(present.as_path()));
    }

    #[test]
    fn first_existing_is_none_when_nothing_exists() {
        // No trusted candidate present → None (degrade cleanly to the caller's fallback, never PATH).
        let a = PathBuf::from(if cfg!(windows) {
            r"C:\nope\a\nvidia-smi.exe"
        } else {
            "/nope/a/nvidia-smi"
        });
        let b = PathBuf::from(if cfg!(windows) {
            r"C:\nope\b\nvidia-smi.exe"
        } else {
            "/nope/b/nvidia-smi"
        });
        assert_eq!(first_existing(&[a, b]), None);
    }

    #[test]
    fn parse_min_total_gib_takes_min_across_gpus() {
        // Two GPUs: 24576 MiB and 49152 MiB → min = 24576 MiB = 24 GiB.
        let got = parse_min_total_gib("24576\n49152\n").expect("parses");
        assert!((got - 24.0).abs() < 1e-9, "expected 24.0 GiB, got {got}");
    }

    #[test]
    fn parse_min_total_gib_none_on_empty_or_garbage() {
        assert_eq!(parse_min_total_gib(""), None);
        assert_eq!(parse_min_total_gib("no gpus\n\n"), None);
        // Zero/negative are filtered out (a driver quirk) → None, not 0.
        assert_eq!(parse_min_total_gib("0\n"), None);
    }
}
