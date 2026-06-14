#!/usr/bin/env python
"""Dump a FLUX.2 VAE **encode** golden (candle-gen sc-5147 task 1).

Runs a diffusers `AutoencoderKLFlux2` encoder, in **float32**, over a synthetic RGB image and records
both the plain posterior **mean** (`vae.encode(x).latent_dist.mean`) and the **packed, bn-normalized**
transformer-space latent (`_encode_latents`' deterministic part). The Rust side loads the same `vae/`
checkpoint into `candle_gen_flux2::vae::Flux2Vae::new_with_encoder` and checks `encode` /
`encode_packed` match.

The packed math (the inverse of `LensImagePipeline._decode`'s bn-denorm + 2×2 unpatchify):
  z   = vae.encode(x).latent_dist.mean         # [1,32,H/8,W/8]
  xp  = _patchify(z)                            # [1,128,H/16,W/16]  (channel = c·4 + p1·2 + p2)
  xp  = (xp - mean) / std                       # bn-normalize, std = sqrt(running_var + batch_norm_eps)

We compare the deterministic **mean** (not `latent_dist.sample()`) so the golden is reproducible.

Run (from the worktree root) with the lens-venv (diffusers ≥ 0.37):
  & "C:\\Users\\Michael\\AppData\\Roaming\\SceneWorks\\python\\lens-venv\\Scripts\\python.exe" \\
      candle-gen-flux2\\scripts\\dump_flux2_vae_encode_golden.py [vae_dir] [out_dir]

`vae_dir` defaults to the cached `black-forest-labs/FLUX.2-klein-9B` `vae/` (or, failing that, the
`microsoft/Lens` one — same AutoencoderKLFlux2 arch). Default out_dir: .scratch/flux2-vae-encode-goldens/.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import torch
from diffusers import AutoencoderKLFlux2
from safetensors.torch import save_file

# A small square image keeps the golden tiny while exercising all 3 downsamples (256 → 32 → packed 16).
IMG_H, IMG_W = 256, 256


def _patchify(latents: torch.Tensor) -> torch.Tensor:
    b, c, h, w = latents.shape
    latents = latents.view(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4)
    return latents.reshape(b, c * 4, h // 2, w // 2)


def find_vae() -> str:
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    for repo in ("models--black-forest-labs--FLUX.2-klein-9B", "models--microsoft--Lens"):
        m = sorted(glob.glob(str(hub / repo / "snapshots" / "*" / "vae")))
        if m:
            return m[-1]
    sys.exit("no FLUX.2-klein-9B / microsoft--Lens vae snapshot found")


def main() -> None:
    args = sys.argv[1:]
    vdir = args[0] if len(args) > 0 else find_vae()
    out_dir = Path(args[1]) if len(args) > 1 else Path(".scratch/flux2-vae-encode-goldens")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"vae: {vdir}\nloading (f32, CPU)…", flush=True)
    vae = AutoencoderKLFlux2.from_pretrained(vdir, torch_dtype=torch.float32).to("cpu").eval()

    torch.manual_seed(0)
    # RGB image in [-1, 1], NCHW.
    image = (torch.rand(1, 3, IMG_H, IMG_W, dtype=torch.float32) * 2.0 - 1.0).contiguous()

    with torch.no_grad():
        dist = vae.encode(image).latent_dist
        mean = dist.mean  # [1, 32, H/8, W/8] — deterministic posterior mean

        bn = vae.bn
        bn_mean = bn.running_mean.view(1, -1, 1, 1)
        bn_var = bn.running_var.view(1, -1, 1, 1)
        std = torch.sqrt(bn_var + vae.config.batch_norm_eps)
        packed = (_patchify(mean) - bn_mean) / std  # [1, 128, H/16, W/16]

    tensors = {
        "image": image.contiguous(),  # [1, 3, H, W] input in [-1, 1]
        "mean": mean.contiguous(),  # [1, 32, H/8, W/8] posterior mean
        "packed": packed.contiguous(),  # [1, 128, H/16, W/16] bn-normalized transformer latent
    }
    dst = out_dir / "flux2_vae_encode_golden.safetensors"
    save_file(tensors, str(dst))
    print(
        f"wrote {dst}  (image={tuple(image.shape)}, mean={tuple(mean.shape)}, "
        f"packed={tuple(packed.shape)})"
    )


if __name__ == "__main__":
    main()
