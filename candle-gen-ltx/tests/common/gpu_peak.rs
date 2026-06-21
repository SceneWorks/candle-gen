//! sc-7148 — shared `nvidia-smi` peak-VRAM sampler for the video-VAE decode sweeps. Polls device-level
//! `memory.used` in a background thread and tracks the max while a decode runs.
//!
//! Device-level (not per-process) is deliberate: Windows WDDM reports per-process `used_memory` as
//! `[N/A]`, and the budgeted decode's safe ceiling is *total* VRAM × 0.85, so the honest "did it fit"
//! quantity is the whole device's used bytes during the decode. Run the sweep on an otherwise-idle GPU
//! (the harness prints the pre-decode `baseline` so you can confirm nothing else was resident).
//!
//! Included via `#[path = "common/gpu_peak.rs"]` so cargo does not build it as its own test binary
//! (only top-level `tests/*.rs` are test targets). Identical copy in candle-gen-ltx and candle-gen-wan.

#![allow(dead_code)]

use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

/// nvidia-smi poll cadence. ~40 ms is well under a multi-second VAE decode (so the peak is captured)
/// while keeping the subprocess-spawn overhead negligible.
const POLL: Duration = Duration::from_millis(40);

/// Device-level used VRAM (MiB) for GPU ordinal `gpu` via `nvidia-smi`, or `None` if the query fails.
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

/// Background sampler: from [`start`](Self::start) until [`stop`](Self::stop), polls `used_mib(gpu)`
/// every [`POLL`] and keeps the running max (MiB).
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
