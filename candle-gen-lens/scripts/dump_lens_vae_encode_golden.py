#!/usr/bin/env python
"""Dump a Lens VAE **encode** golden (candle-gen sc-5147 task 2).

Runs the cached `microsoft/Lens` VAE (a diffusers `AutoencoderKLFlux2`) through the vendor
`lens_train_runner._encode_latents` math, in **float32**, over a synthetic RGB image. The Rust side
loads the same `vae/` checkpoint into the shared `candle_gen_flux2::Flux2Vae` (with the encoder) and
runs the Lens encode shim; this golden lets it check the packed DiT latent `[1, S, 128]` matches.

The reference `_encode_latents` (deterministic **mean**, not `.sample()`, so the golden is reproducible):
  z   = vae.encode(x).latent_dist.mean            # [1,32,H/8,W/8] — NEURAL encoder
  xp  = _patchify(z)                              # [1,128,H/16,W/16] (channel = c·4 + p1·2 + p2)
  xp  = (xp - mean) / std                         # bn-normalize, std = sqrt(running_var + batch_norm_eps)
  lat = _unpatchify(xp)                           # [1,32,H/8,W/8]
  x0  = rearrange(lat,"b c (h p1)(w p2)->b (h w)(c p1 p2)", p1=2,p2=2)   # [1, S, 128]

Run (from the worktree root) with the lens-venv:
  & "C:\\Users\\Michael\\AppData\\Roaming\\SceneWorks\\python\\lens-venv\\Scripts\\python.exe" \\
      candle-gen-lens\\scripts\\dump_lens_vae_encode_golden.py [out_dir]

Default out_dir: .scratch/lens-vae-encode-goldens/  (not committed — regenerable).
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import torch
from diffusers import AutoencoderKLFlux2
from safetensors.torch import save_file

IMG_H, IMG_W = 256, 256  # → packed grid (H/16, W/16) = (16, 16), S = 256


def _patchify(latents: torch.Tensor) -> torch.Tensor:
    b, c, h, w = latents.shape
    latents = latents.view(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4)
    return latents.reshape(b, c * 4, h // 2, w // 2)


def _unpatchify(latents: torch.Tensor) -> torch.Tensor:
    b, c, h, w = latents.shape
    latents = latents.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3)
    return latents.reshape(b, c // 4, h * 2, w * 2)


def find_vae() -> str:
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    m = sorted(glob.glob(str(hub / "models--microsoft--Lens" / "snapshots" / "*" / "vae")))
    if not m:
        m = sorted(glob.glob(str(hub / "models--microsoft--Lens-Turbo" / "snapshots" / "*" / "vae")))
    if not m:
        sys.exit("no microsoft/Lens vae snapshot found")
    return m[-1]


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".scratch/lens-vae-encode-goldens")
    out_dir.mkdir(parents=True, exist_ok=True)

    vdir = find_vae()
    print(f"vae: {vdir}\nloading (f32, CPU)…", flush=True)
    vae = AutoencoderKLFlux2.from_pretrained(vdir, torch_dtype=torch.float32).to("cpu").eval()

    torch.manual_seed(0)
    image = (torch.rand(1, 3, IMG_H, IMG_W, dtype=torch.float32) * 2.0 - 1.0).contiguous()

    with torch.no_grad():
        z = vae.encode(image).latent_dist.mean
        bn = vae.bn
        mean = bn.running_mean.view(1, -1, 1, 1)
        std = torch.sqrt(bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps)
        xp = (_patchify(z) - mean) / std
        lat = _unpatchify(xp)
        lh, lw = lat.shape[2] // 2, lat.shape[3] // 2
        x0 = (
            lat.view(1, 32, lh, 2, lw, 2)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(1, lh * lw, 128)
        )

    tensors = {
        "image": image.contiguous(),  # [1, 3, H, W] input in [-1, 1]
        "x0": x0.contiguous(),  # [1, S, 128] packed DiT latent (posterior mean)
        "grid_hw": torch.tensor([lh, lw], dtype=torch.int64),
    }
    dst = out_dir / "lens_vae_encode_golden.safetensors"
    save_file(tensors, str(dst))
    print(f"wrote {dst}  (image={tuple(image.shape)}, x0={tuple(x0.shape)}, grid=({lh},{lw}))")


if __name__ == "__main__":
    main()
