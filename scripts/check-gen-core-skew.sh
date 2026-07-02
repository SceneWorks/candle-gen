#!/usr/bin/env bash
# sc-4482 (epic 3720): version-skew guard for the backend-neutral gen-core contract.
#
# THE TRAP: everything is git-SHA-pinned. If `mlx-gen` (macOS) resolves `sceneworks-gen-core`
# at rev A while the worker's direct dep resolves rev B, cargo silently builds BOTH. The
# provider crates (linked `as _`, no type contact with worker code) register into rev A's
# `inventory` registry while the worker queries rev B's. The symptom is "engine not found" at
# RUNTIME, not a compile error.
#
# This gate fails the build if more than one distinct `sceneworks-gen-core` resolution exists in
# the root package's dependency graph. It is reusable: pass the root package as $1 (default
# `sceneworks-worker`); the candle-gen repo wires this same script against its own worker package
# (epic 3672). `--target all` is REQUIRED so the macOS-only `mlx-gen` transitive gen-core is
# resolved even when this runs on a Linux CI lane (otherwise the skew is invisible off-macOS).
#
# Self-test: `check-gen-core-skew.sh --self-test` exercises the verdict logic on canned input
# (one-resolution => pass, two-resolutions => fail) so CI proves the gate actually fires without
# needing a deliberately-broken pin checked in.
set -euo pipefail

CRATE="sceneworks-gen-core"

# Decide from a newline-delimited list of distinct resolution lines on stdin.
# Exit 0 iff exactly one resolution; otherwise print the skew explanation and exit 1.
evaluate() {
  local pkg="$1"
  local lines=()
  local line
  while IFS= read -r line; do
    [ -n "$line" ] && lines+=("$line")
  done
  local count=${#lines[@]}

  if [ "$count" -eq 1 ]; then
    echo "OK: exactly one ${CRATE} in ${pkg}'s build graph: ${lines[0]}"
    return 0
  fi

  if [ "$count" -eq 0 ]; then
    echo "ERROR (sc-4482): ${CRATE} was not found in ${pkg}'s build graph at all." >&2
    echo "Expected the worker to depend on ${CRATE} (the backend-neutral gen-core contract)." >&2
    return 1
  fi

  {
    echo "ERROR (sc-4482 version skew): found ${count} distinct ${CRATE} resolutions in ${pkg}'s build graph:"
    printf '  %s\n' "${lines[@]}"
    cat <<'MSG'

Two gen-core revs => provider crates register into ONE inventory registry while the worker
queries ANOTHER => the engine is "not found" at RUNTIME (not a compile error, because provider
crates link `as _` and their types never meet the worker). Align the pins: the `mlx-gen` git rev
and the direct `sceneworks-gen-core` git rev in crates/sceneworks-worker/Cargo.toml MUST be
identical (and any candle-gen pin must match too). Bump them together.
MSG
  } >&2
  return 1
}

self_test() {
  local rc=0
  echo "self-test: single resolution should PASS"
  if printf '%s\n' "sceneworks-gen-core v0.1.0 (git+https://example/repo?rev=AAA#AAA)" \
      | evaluate "self-test" >/dev/null; then
    echo "  ok"
  else
    echo "  FAIL: single resolution was rejected"; rc=1
  fi

  echo "self-test: two distinct resolutions should FAIL"
  if printf '%s\n%s\n' \
      "sceneworks-gen-core v0.1.0 (git+https://example/repo?rev=AAA#AAA)" \
      "sceneworks-gen-core v0.1.0 (git+https://example/repo?rev=BBB#BBB)" \
      | evaluate "self-test" >/dev/null 2>&1; then
    echo "  FAIL: skew was NOT detected"; rc=1
  else
    echo "  ok"
  fi

  echo "self-test: zero resolutions should FAIL"
  if printf '' | evaluate "self-test" >/dev/null 2>&1; then
    echo "  FAIL: missing dependency was NOT detected"; rc=1
  else
    echo "  ok"
  fi

  if [ "$rc" -eq 0 ]; then echo "self-test: PASS"; else echo "self-test: FAIL"; fi
  return "$rc"
}

if [ "${1:-}" = "--self-test" ]; then
  self_test
  exit $?
fi

# Default to this repo's core package. candle-gen is the Windows/CUDA sibling of mlx-gen (epic
# 3672/3720); both must resolve EXACTLY ONE `sceneworks-gen-core`. Pass another package as $1 to
# check a different root (e.g. `candle-gen-sdxl`, or the SceneWorks worker from its own tree).
PKG="${1:-candle-gen}"

# Flatten the tree (`--prefix none`), strip the ` (*)` dedupe marker cargo appends to repeated
# nodes, keep only the contract crate, and unique-sort. Each unique (version + source) line is one
# distinct resolution; two revs differ in the `#<rev>` source fragment.
#
# `--color never` overrides CI's `CARGO_TERM_COLOR=always` (which would otherwise wrap the ` (*)`
# marker in ANSI codes so the `(*)$` strip misses it and a deduped node looks "distinct" — a false
# skew). The ESC-strip sed is a portable backstop (works on BSD + GNU sed).
esc=$(printf '\033')

# sc-9022 (F-038): run `cargo tree` on its own and capture stdout, stderr, and exit code
# SEPARATELY. Previously this was `cargo tree … 2>/dev/null | sed | … | evaluate`, which discarded
# cargo's stderr AND let the pipeline's exit status come from `evaluate` (the last stage). So a
# genuine RESOLUTION failure (network/fetch, auth, a bad or unknown pin, unknown package) produced
# zero matching lines, and `evaluate` mis-reported "${CRATE} was not found in the build graph" — the
# WRONG diagnosis, sending developers to hunt for pin skew when cargo could not resolve the graph at
# all. Now a nonzero `cargo tree` exit is surfaced with cargo's own stderr and a distinct message,
# and only successful output is fed into the skew verdict.
#
# NOTE on `set -e`: a failing command substitution in a plain assignment (`var=$(cmd)`) DOES abort
# under `set -euo pipefail` when the script runs from a file, so we can't just read `$?` afterwards
# — the script would exit first with cargo's raw code and NONE of the diagnostics below. We disable
# errexit only for the capture, record the exit code, then restore it.
tree_err_file=$(mktemp 2>/dev/null || echo "${TMPDIR:-/tmp}/gen-core-skew-cargo.$$")
set +e
tree_out=$(cargo tree -p "$PKG" --target all --color never --prefix none 2>"$tree_err_file")
tree_rc=$?
set -e
tree_err=$(cat "$tree_err_file" 2>/dev/null)
rm -f "$tree_err_file"

if [ "$tree_rc" -ne 0 ]; then
  {
    echo "ERROR (sc-4482 / sc-9022): 'cargo tree -p ${PKG} --target all' FAILED (exit ${tree_rc})."
    echo "This is a dependency RESOLUTION failure (network/fetch, auth, a bad/unknown pin, unknown"
    echo "package), NOT a version skew and NOT a missing ${CRATE}. Fix the resolution error below"
    echo "before the skew gate can run. cargo's own diagnostics:"
    echo "---- cargo tree stderr ----"
    if [ -n "$tree_err" ]; then
      printf '%s\n' "$tree_err"
    else
      echo "(cargo produced no stderr output)"
    fi
    echo "---------------------------"
  } >&2
  exit "$tree_rc"
fi

printf '%s\n' "$tree_out" \
  | sed "s/${esc}\\[[0-9;]*m//g" \
  | sed 's/ (\*)$//' \
  | grep -E "^${CRATE} v" \
  | sort -u \
  | evaluate "$PKG"
