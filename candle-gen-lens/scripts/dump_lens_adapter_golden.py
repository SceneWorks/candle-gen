#!/usr/bin/env python
"""Dump LoRA + LoKr adapter goldens for the Lens DiT (candle-gen sc-5116).

Builds **deterministic synthetic** PEFT adapters (LoRA, then LoKr) on the authoritative vendor
`LensTransformer2DModel` (SceneWorks `_vendor/lens/transformer.py`), targeting the exact trainer
modules (`lens_train_runner.DEFAULT_LORA_TARGET_MODULES` = `img_qkv` / `txt_qkv` / `to_out.0` /
`to_add_out`), saves each in the **on-disk format the trainer ships** (diffusers `save_lora_adapter`
for LoRA; `get_peft_model_state_dict` + `networkType=lokr` metadata for LoKr — the same code paths as
`lens_train_runner`), and dumps the base + adapter-applied DiT outputs over fixed synthetic inputs.

The Rust gate (`tests/adapter_parity.rs`) loads the SAME adapter files via `merge_adapters`, folds the
delta into the dense `transformer/` weights, and asserts the merged DiT matches these outputs (f32,
tight — LoRA/LoKr is a linear-merge delta), and that a scale-0 apply is a bit-exact no-op.

`img_qkv` / `txt_qkv` are **fused** `[3·inner, inner]` projections that the trainer targets as one
module each, so the LoRA/LoKr delta spans the whole fused weight — there is no q/k/v split.

Run with the transformers-5.8 lens-venv (needs `peft` installed — `pip install peft` if missing;
loads the ~16 GB f32 transformer on CPU):
  & "C:\\Users\\Michael\\AppData\\Roaming\\SceneWorks\\python\\lens-venv\\Scripts\\python.exe" \\
      candle-gen-lens\\scripts\\dump_lens_adapter_golden.py [out_dir]

Default out_dir: .scratch/lens-adapter-goldens/  (not committed — large + regenerable). Writes
  lens_lora_adapter.safetensors, lens_lokr_adapter.safetensors, lens_adapter_golden.safetensors
"""
from __future__ import annotations

import glob
import importlib.util
import json
import os
import sys
from pathlib import Path

import peft
import torch
from peft.utils import get_peft_model_state_dict
from safetensors.torch import save_file

# The vendor DiT class. Prefer the main worktree checkout; fall back to the desktop python-src copy.
VENDOR_CANDIDATES = [
    r"D:\repos\SceneWorks\apps\worker\scene_worker\_vendor\lens\transformer.py",
    r"D:\repos\SceneWorks\apps\desktop\python-src\scene_worker\_vendor\lens\transformer.py",
    os.path.expanduser(
        r"~\AppData\Local\Programs\SceneWorks\python-src\scene_worker\_vendor\lens\transformer.py"
    ),
]

FRAME, H_LAT, W_LAT = 1, 16, 16
TXT_LEN = 120
TARGET_MODULES = ["img_qkv", "txt_qkv", "to_out.0", "to_add_out"]
RANK = 8
ALPHA = 8  # alpha/rank = 1 → internal scaling 1.0 (the Rust merge uses scale 1.0 to match)
DECOMPOSE_FACTOR = -1


def load_model_cls():
    vendor = next((p for p in VENDOR_CANDIDATES if os.path.exists(p)), None)
    if vendor is None:
        sys.exit("no vendor lens/transformer.py found; tried:\n  " + "\n  ".join(VENDOR_CANDIDATES))
    print(f"vendor transformer: {vendor}")
    spec = importlib.util.spec_from_file_location("lens_transformer", vendor)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.LensTransformer2DModel


def find_transformer() -> str:
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    matches = sorted(
        glob.glob(str(hub / "models--microsoft--Lens-Turbo" / "snapshots" / "*" / "transformer"))
    )
    if not matches:
        sys.exit(f"no microsoft/Lens-Turbo transformer snapshot under {hub}")
    return matches[-1]


def randomize_adapter(model, seed: int, std: float) -> None:
    """Overwrite the freshly-attached adapter params with seeded gaussians so the delta is non-zero
    (PEFT inits LoRA-B / one LoKr factor to zero → a no-op delta otherwise)."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_" in name or "lokr_" in name or "hada_" in name:
                p.copy_(torch.randn(p.shape, generator=gen, dtype=p.dtype) * std)


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".scratch/lens-adapter-goldens")
    out_dir.mkdir(parents=True, exist_ok=True)

    LensTransformer2DModel = load_model_cls()
    tdir = find_transformer()
    print(f"transformer: {tdir}\nloading (f32, CPU)…", flush=True)
    model = LensTransformer2DModel.from_pretrained(tdir, torch_dtype=torch.float32).to("cpu").eval()

    img_len = FRAME * H_LAT * W_LAT
    n_text = len(model.config.selected_layer_index)
    enc_dim = model.config.enc_hidden_dim
    print(f"img_len={img_len} txt_len={TXT_LEN} n_text={n_text} enc_dim={enc_dim}")

    torch.manual_seed(0)
    hidden_states = torch.randn(1, img_len, model.config.in_channels, dtype=torch.float32)
    feats = [torch.randn(1, TXT_LEN, enc_dim, dtype=torch.float32) for _ in range(n_text)]
    timestep = torch.rand(1, dtype=torch.float32)  # in [0, 1]
    text_mask = torch.ones(1, TXT_LEN, dtype=torch.bool)
    img_shapes = [(FRAME, H_LAT, W_LAT)]

    def forward():
        with torch.no_grad():
            return model(hidden_states, feats, text_mask, timestep, img_shapes)

    tensors = {
        "hidden_states": hidden_states.contiguous(),
        "timestep": timestep.contiguous(),
        "grid_fhw": torch.tensor([FRAME, H_LAT, W_LAT], dtype=torch.int64),
        "base_out": forward().contiguous(),
    }
    for i, f in enumerate(feats):
        tensors[f"feat_{i}"] = f.contiguous()
    print("base forward done", flush=True)

    # --- LoRA: attach (gaussian), randomize B, save via diffusers, forward ---
    lora_cfg = peft.LoraConfig(
        r=RANK, lora_alpha=ALPHA, init_lora_weights="gaussian", target_modules=TARGET_MODULES
    )
    model.add_adapter(lora_cfg)
    randomize_adapter(model, seed=20260613, std=0.02)
    model.save_lora_adapter(
        str(out_dir), weight_name="lens_lora_adapter.safetensors", safe_serialization=True
    )
    tensors["lora_out"] = forward().contiguous()
    model.delete_adapters(model.active_adapters())
    print("LoRA done", flush=True)

    # --- LoKr: attach, randomize, save via get_peft_model_state_dict + metadata, forward ---
    lokr_cfg = peft.LoKrConfig(
        r=RANK,
        alpha=ALPHA,
        decompose_factor=DECOMPOSE_FACTOR,
        init_weights=True,
        target_modules=TARGET_MODULES,
    )
    model.add_adapter(lokr_cfg)
    randomize_adapter(model, seed=20260614, std=0.05)
    lokr_state = {
        k: v.detach().cpu().contiguous() for k, v in get_peft_model_state_dict(model).items()
    }
    save_file(
        lokr_state,
        str(out_dir / "lens_lokr_adapter.safetensors"),
        metadata={
            "format": "pt",
            "networkType": "lokr",
            "rank": str(RANK),
            "alpha": str(ALPHA),
            "decomposeFactor": str(DECOMPOSE_FACTOR),
            "targetModules": json.dumps(TARGET_MODULES),
        },
    )
    tensors["lokr_out"] = forward().contiguous()
    print("LoKr done", flush=True)

    dst = out_dir / "lens_adapter_golden.safetensors"
    save_file(tensors, str(dst))
    print(
        f"wrote {dst} + lens_lora_adapter.safetensors + lens_lokr_adapter.safetensors "
        f"(out={tuple(tensors['base_out'].shape)})"
    )


if __name__ == "__main__":
    main()
