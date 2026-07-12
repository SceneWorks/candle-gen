#!/usr/bin/env bash
# sc-11278 (F-103 / F-031, epic 11146): poison-tolerant-lock guard.
#
# THE TRAP: the shared generator/component caches live behind a `Mutex` on a long-lived
# `Arc<dyn Generator>` (`&self`). A panic *while a lock is held* (e.g. a CUDA OOM lifted to a panic
# mid-decode / mid-load) poisons that `Mutex`. A plain `.lock().unwrap()` / `.lock().expect(…)` then
# panics forever on every subsequent lock — one transient failure wedges the worker lane into a
# permanent panic loop until the process restarts. The workspace standardized on the poison-tolerant
# `candle_gen::lock_recover` (sc-9015) precisely to remove this class; the RoPE/geometry cache wave
# (sc-8992) and several component locks re-seeded it (sc-11278).
#
# This gate fails the build if any `.lock().unwrap()` / `.lock().expect(…)` survives under
# `candle-gen*/src`, so the poison class cannot be re-introduced. Two intentional exemptions:
#   * `vm.data().lock()` — candle's OWN `VarMap` internal `Mutex`, touched only on load / merge /
#     quantize (not our cache, not held across a render), so it is out of this finding's scope.
#   * `candle-gen/src/sync.rs` + `candle-gen/src/lib.rs` — the helper's home: their prose/doc-comments
#     and the helper's own poison-recovery unit tests reference the literal pattern on purpose.
set -euo pipefail

cd "$(dirname "$0")/.."

hits=$(grep -rnE '\.lock\(\)\.(unwrap|expect)\(' --include='*.rs' candle-gen*/src \
  | grep -v 'vm\.data()\.lock()' \
  | grep -v '^candle-gen/src/sync\.rs:' \
  | grep -v '^candle-gen/src/lib\.rs:' \
  || true)

if [ -n "$hits" ]; then
  echo "error (sc-11278 / F-103): panicking .lock().unwrap()/.lock().expect(...) on a shared"
  echo "cache/component mutex. A panic while the lock is held poisons it and wedges the worker into a"
  echo "permanent panic loop. Use candle_gen::lock_recover instead (see candle-gen/src/sync.rs)."
  echo "Offending sites:"
  echo "$hits"
  exit 1
fi

echo "check-lock-poison: OK (no panicking cache/component locks under candle-gen*/src)."
