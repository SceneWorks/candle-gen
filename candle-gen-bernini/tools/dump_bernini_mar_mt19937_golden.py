"""sc-11671: torch+numpy independent-oracle golden for the Bernini MAR reveal-order / per-step-noise
MT19937 bit-parity (`crate::rng`).

The reference `sample_vit_embed` (`_vendor/bernini/pipeline.py`) draws two things from two distinct
MT19937 generators:
  - the reveal **permutation** `order` via numpy legacy `np.random.shuffle` (MT19937 + Fisher–Yates over
    `random_interval`), and
  - the per-step flow-match **base noise** via `torch.randn` (torch CPU MT19937 → `normal_fill`
    Box–Muller), drawn sequentially across the planning steps inside the `revealed.sum() != 0` branch.

This dumps both for a FIXED seed + shapes so the Rust reimplementation (`candle-gen-bernini/src/rng.rs`)
can be asserted bit-exact (permutation, integer) / tight-tol (noise, f32). Pure torch + numpy — no
Bernini model math — so it is a genuinely independent oracle.

Env (this box):
  uv venv --python 3.12 .venv-golden
  uv pip install --python .venv-golden/Scripts/python.exe numpy torch --index-url https://download.pytorch.org/whl/cpu
  .venv-golden/Scripts/python.exe candle-gen-bernini/tools/dump_bernini_mar_mt19937_golden.py

Fixture -> candle-gen-bernini/tests/fixtures/mar_mt19937_golden.safetensors
"""

from __future__ import annotations

import math
import os

import numpy as np
import torch
from safetensors.torch import save_file

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(REPO_ROOT, "tests", "fixtures", "mar_mt19937_golden.safetensors")

# Fixed seed + shapes. n_query/planning_step span single-token reveals, the reference's `sum==0` skip
# (a lone token-0 reveal), multi-token reveals, and (in_channels=24 → np·24) the `normal_fill` tail
# recompute (size % 16 != 0). in_channels stays >= 16 so every draw is the torch `normal_fill` regime.
SEED = 1234
N_QUERY = 60
PLANNING_STEP = 25
IN_CHANNELS = 24


def mar_schedule(n_query: int, planning_step: int, order: list[int]) -> list[list[int]]:
    """Port of `crate::mar::mar_schedule` — sorted revealed positions per MaskGIT step."""
    out: list[list[int]] = []
    prev = n_query
    for step in range(planning_step):
        ratio = math.cos(math.pi / 2.0 * (step + 1) / planning_step)
        raw = math.floor(n_query * ratio)
        mask_len = max(min(raw, prev - 1), 1)
        if step >= planning_step - 1:
            revealed = order[:prev]
        else:
            revealed = order[mask_len:prev]
        out.append(sorted(revealed))
        prev = mask_len
    return out


def main() -> None:
    # --- reveal permutation: numpy legacy RandomState(seed).shuffle(arange) ---
    rs = np.random.RandomState(SEED)
    order_arr = np.arange(N_QUERY)
    rs.shuffle(order_arr)
    order = order_arr.tolist()

    schedule = mar_schedule(N_QUERY, PLANNING_STEP, order)

    # --- per-step FM noise: one torch generator, sequential torch.randn on non-skip steps ---
    g = torch.Generator().manual_seed(SEED)
    out: dict[str, torch.Tensor] = {}
    out["order"] = torch.tensor(order, dtype=torch.int32).contiguous()

    noise_steps: list[int] = []
    for step, revealed in enumerate(schedule):
        if sum(revealed) == 0:  # empty or {token 0} alone -> reference `nonzero().sum()==0` skip
            continue
        np_step = len(revealed)
        noise = torch.randn(np_step, IN_CHANNELS, generator=g, dtype=torch.float32)
        out[f"noise.{step}"] = noise.contiguous()
        noise_steps.append(step)

    meta = {
        "seed": str(SEED),
        "n_query": str(N_QUERY),
        "planning_step": str(PLANNING_STEP),
        "in_channels": str(IN_CHANNELS),
        "noise_steps": ",".join(map(str, noise_steps)),
        "order": ",".join(map(str, order)),
    }
    os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
    save_file(out, FIXTURE, metadata=meta)
    print(f"wrote {FIXTURE}  ({len(out)} tensors)")
    print(f"  order[:8]={order[:8]}  noise steps={noise_steps}")
    print(f"  numpy={np.__version__}  torch={torch.__version__}")


if __name__ == "__main__":
    main()
