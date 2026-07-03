#!/usr/bin/env python
"""SD3.5 diffusers component-parity REFERENCE generator (sc-9076, epic 8979 F-001/F-002 follow-up).

Computes bit-reference intermediate tensors for the candle `candle-gen-sd3` port to validate against,
component-by-component, using HuggingFace `diffusers` + `transformers` as the ground truth. The Rust
side (`tests/component_parity.rs`) runs the SAME fixed prompt / fixed inputs through `candle-gen-sd3`
and compares to the tensors this script dumps (cosine + max-abs-diff, documented tolerances).

Covered components (the F-001 / F-002 bug surfaces plus the broader conditioning + DiT chain):
  * CLIP-L  pooled  [1, 768]   -- F-002: pooled taken at the FIRST EOS token (HF `argmax` pooling),
                                  NOT the last pad slot. `prompt_embeds[0]` == projected pooled.
  * CLIP-bigG pooled [1, 1280] -- same F-002 surface for the second CLIP encoder.
  * CLIP-L  penultimate hidden [1, 77, 768]   -- `hidden_states[-2]` (feeds the joint context).
  * CLIP-bigG penultimate hidden [1, 77, 1280].
  * T5-XXL hidden [1, T5_LEN, 4096].
  * aggregated pooled  [1, 2048]  = cat(clip_l_pooled, clip_g_pooled).
  * aggregated context [1, 77+T5_LEN, 4096] = cat(pad(cat(clip_l, clip_g)), t5) on the seq axis.
  * DiT one-step velocity [1, 16, H, W] -- runs the full `SD3Transformer2DModel` forward once at a
                                  fixed timestep on a fixed deterministic latent. This exercises the
                                  F-001 final joint-block context-AdaLN (scale/shift order) end-to-end:
                                  a swapped final-block AdaLN scrambles the predicted velocity.

Everything is computed in float32 on CPU for a deterministic, launch-portable reference. Weights are
resolved from the local HF cache (set $HF_HOME=D:/.cache/huggingface).

The generator is variant-agnostic (`--model`/`--tag`). The committed golden pairs (sc-9076 + sc-9580):
    * sd35_large       -> stabilityai/stable-diffusion-3.5-large
    * sd35_medium      -> stabilityai/stable-diffusion-3.5-medium        (MMDiT-X, dual-attention)
    * sd35_large_turbo -> stabilityai/stable-diffusion-3.5-large-turbo   (guidance-distilled)

Run (from the crate root, in the parity venv — torch(CPU)+diffusers+transformers+sentencepiece):
    HF_HOME=D:/.cache/huggingface python tests/parity/gen_reference.py \
        --model stabilityai/stable-diffusion-3.5-medium --tag sd35_medium \
        --out tests/parity/reference

Outputs `<out>/<tag>_reference.safetensors` (all tensors, f32) + `<out>/<tag>_manifest.json` (the fixed
prompt, seed, timestep, latent shape, tolerances, tensor list) so the Rust harness is fully driven by
committed metadata (no magic constants duplicated across the two sides).
"""

import argparse
import json
import os
from pathlib import Path

import torch
from safetensors.torch import save_file


# ------------------------------------------------------------------------------------------------
# Fixed harness inputs -- MUST match the constants in tests/component_parity.rs.
# ------------------------------------------------------------------------------------------------
PROMPT = "a photograph of an astronaut riding a horse on the surface of the moon, golden hour"
SEED = 20240703
TIMESTEP = 500.0  # DiT timestep in the [0, 1000] convention (diffusers scales sigma*1000; candle too)
T5_LEN = 256      # SD3.5 default T5 sequence length (tokenizer_max_length for T5 here)
CLIP_LEN = 77
# Latent geometry for the one-step DiT parity: 64x64 image / 8 => 8x8 latent, 16 channels.
LATENT_H = 8
LATENT_W = 8
LATENT_CH = 16

# Documented parity tolerances per component: a **cosine floor** (the correctness guard — a swapped
# AdaLN or wrong pooling token collapses cosine far below 1) plus a **max-abs ceiling** (drift guard).
#
# The max-abs ceilings are set relative to each tensor's dynamic range, NOT a uniform small constant:
# candle's f32 matmul-accumulation order differs from torch's, and **OpenCLIP bigG has a pathological
# high-magnitude activation channel** (its penultimate hidden reaches ~66, and once that channel is
# concatenated + padded into the joint context the context absmax is ~850). A few-tenths max-abs on a
# ~66-magnitude tensor is <0.5% relative drift, so the bigG-derived components (clip_g_*, pooled,
# context) get magnitude-appropriate ceilings while their cosine floors stay tight (>= 0.9999). CLIP-L
# and T5 have well-behaved ranges and hold a few-1e-3 max-abs. The DiT accumulates across 24-38 joint
# blocks so its band is the widest, but cosine must stay ~1 (the F-001 final-AdaLN guard).
TOLERANCES = {
    "clip_l_pooled":      {"cosine_min": 0.9999, "max_abs": 5e-3},
    "clip_g_pooled":      {"cosine_min": 0.9999, "max_abs": 1e-1},   # bigG range ~4.4 (projected)
    "clip_l_penultimate": {"cosine_min": 0.9999, "max_abs": 5e-3},
    "clip_g_penultimate": {"cosine_min": 0.9999, "max_abs": 5e-1},   # bigG range ~66 (outlier ch)
    "t5_hidden":          {"cosine_min": 0.9995, "max_abs": 1e-2},
    "pooled":             {"cosine_min": 0.9999, "max_abs": 1e-1},   # cat(clip_l, clip_g) -> bigG range
    "context":            {"cosine_min": 0.9995, "max_abs": 5e-1},   # bigG outlier ch padded in (~850)
    "dit_velocity":       {"cosine_min": 0.999,  "max_abs": 5e-2},
}


def deterministic_latent():
    """Fixed CPU-seeded N(0,1) latent [1, 16, H, W] -- same construction the Rust side uses
    (StdRng-seeded normal), so the DiT sees an identical input. We seed torch's default CPU RNG and
    also emit the exact tensor so the Rust side loads it rather than re-deriving the RNG stream."""
    g = torch.Generator(device="cpu").manual_seed(SEED)
    return torch.randn(1, LATENT_CH, LATENT_H, LATENT_W, generator=g, dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="stabilityai/stable-diffusion-3.5-large")
    ap.add_argument("--out", default="tests/parity/reference")
    ap.add_argument("--tag", default="sd35_large", help="output basename tag")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from transformers import (
        CLIPTextModelWithProjection,
        CLIPTokenizer,
        T5EncoderModel,
        T5TokenizerFast,
    )
    from diffusers import SD3Transformer2DModel

    device = torch.device("cpu")
    dtype = torch.float32
    tensors = {}

    print(f"[ref] model={args.model} device={device} dtype={dtype}")

    # -- CLIP-L (text_encoder / tokenizer) --------------------------------------------------------
    print("[ref] loading CLIP-L ...")
    tok_l = CLIPTokenizer.from_pretrained(args.model, subfolder="tokenizer")
    te_l = CLIPTextModelWithProjection.from_pretrained(
        args.model, subfolder="text_encoder", torch_dtype=dtype
    ).to(device).eval()

    # -- CLIP-bigG (text_encoder_2 / tokenizer_2) -------------------------------------------------
    print("[ref] loading CLIP-bigG ...")
    tok_g = CLIPTokenizer.from_pretrained(args.model, subfolder="tokenizer_2")
    te_g = CLIPTextModelWithProjection.from_pretrained(
        args.model, subfolder="text_encoder_2", torch_dtype=dtype
    ).to(device).eval()

    def clip_embeds(tok, te):
        ti = tok(
            PROMPT,
            padding="max_length",
            max_length=CLIP_LEN,
            truncation=True,
            return_tensors="pt",
        )
        ids = ti.input_ids.to(device)
        with torch.no_grad():
            out = te(ids, output_hidden_states=True)
        pooled = out[0]                    # projected pooled (EOS-position) [1, embed] -- F-002 path
        penult = out.hidden_states[-2]     # [1, 77, embed]
        return ids, pooled.float(), penult.float()

    with torch.no_grad():
        ids_l, pooled_l, penult_l = clip_embeds(tok_l, te_l)
        ids_g, pooled_g, penult_g = clip_embeds(tok_g, te_g)
    tensors["clip_l_pooled"] = pooled_l.contiguous()
    tensors["clip_g_pooled"] = pooled_g.contiguous()
    tensors["clip_l_penultimate"] = penult_l.contiguous()
    tensors["clip_g_penultimate"] = penult_g.contiguous()
    tensors["clip_l_input_ids"] = ids_l.to(torch.int32).contiguous()
    tensors["clip_g_input_ids"] = ids_g.to(torch.int32).contiguous()
    print(f"[ref] clip_l pooled {tuple(pooled_l.shape)} penult {tuple(penult_l.shape)}")
    print(f"[ref] clip_g pooled {tuple(pooled_g.shape)} penult {tuple(penult_g.shape)}")

    del te_l, te_g

    # -- T5-XXL (text_encoder_3 / tokenizer_3) ----------------------------------------------------
    print("[ref] loading T5-XXL ...")
    tok_t5 = T5TokenizerFast.from_pretrained(args.model, subfolder="tokenizer_3")
    te_t5 = T5EncoderModel.from_pretrained(
        args.model, subfolder="text_encoder_3", torch_dtype=dtype
    ).to(device).eval()
    ti5 = tok_t5(
        PROMPT,
        padding="max_length",
        max_length=T5_LEN,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        t5_hidden = te_t5(ti5.input_ids.to(device))[0].float()
    tensors["t5_hidden"] = t5_hidden.contiguous()
    tensors["t5_input_ids"] = ti5.input_ids.to(torch.int32).contiguous()
    print(f"[ref] t5_hidden {tuple(t5_hidden.shape)}")
    del te_t5

    # -- aggregated pooled + context (the diffusers encode_prompt combination) --------------------
    pooled = torch.cat([pooled_l, pooled_g], dim=-1)                     # [1, 2048]
    clip_ctx = torch.cat([penult_l, penult_g], dim=-1)                   # [1, 77, 2048]
    clip_ctx = torch.nn.functional.pad(clip_ctx, (0, t5_hidden.shape[-1] - clip_ctx.shape[-1]))
    context = torch.cat([clip_ctx, t5_hidden], dim=-2)                   # [1, 77+T5_LEN, 4096]
    tensors["pooled"] = pooled.contiguous()
    tensors["context"] = context.contiguous()
    print(f"[ref] pooled {tuple(pooled.shape)} context {tuple(context.shape)}")

    # -- DiT one-step velocity --------------------------------------------------------------------
    # Exercises the whole MMDiT including the F-001 final context-AdaLN. Fixed latent + fixed
    # timestep; the diffusers SD3Transformer2DModel returns the model velocity as `sample`.
    print("[ref] loading SD3Transformer2DModel ...")
    dit = SD3Transformer2DModel.from_pretrained(
        args.model, subfolder="transformer", torch_dtype=dtype
    ).to(device).eval()
    latent = deterministic_latent()
    timestep = torch.tensor([TIMESTEP], dtype=dtype, device=device)
    with torch.no_grad():
        vel = dit(
            hidden_states=latent,
            timestep=timestep,
            encoder_hidden_states=context,
            pooled_projections=pooled,
            return_dict=True,
        ).sample.float()
    tensors["dit_latent_in"] = latent.contiguous()
    tensors["dit_velocity"] = vel.contiguous()
    print(f"[ref] dit_velocity {tuple(vel.shape)}")
    del dit

    # -- write --------------------------------------------------------------------------------------
    st_path = out / f"{args.tag}_reference.safetensors"
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(st_path))
    manifest = {
        "model": args.model,
        "tag": args.tag,
        "prompt": PROMPT,
        "seed": SEED,
        "timestep": TIMESTEP,
        "t5_len": T5_LEN,
        "clip_len": CLIP_LEN,
        "latent_shape": [1, LATENT_CH, LATENT_H, LATENT_W],
        "tolerances": TOLERANCES,
        "tensors": {k: list(v.shape) for k, v in tensors.items()},
        "reference_file": st_path.name,
        "note": (
            "SD3.5 diffusers component-parity reference (sc-9076). Regenerate with "
            "tests/parity/gen_reference.py in the sd35env venv."
        ),
    }
    (out / f"{args.tag}_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[ref] wrote {st_path} ({st_path.stat().st_size} bytes) + manifest")


if __name__ == "__main__":
    main()
