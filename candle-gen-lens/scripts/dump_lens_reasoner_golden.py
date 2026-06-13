#!/usr/bin/env python
"""Dump a greedy-generation golden for the Lens PromptReasoner (candle-gen sc-5118).

Runs the authoritative vendor ``LensGptOssEncoder`` (the `GptOssForCausalLM` subclass that ships in the
`Lens-Turbo` `text_encoder/`) as a **generating** model — `generate(do_sample=False)` — over the harmony
reasoner prompt (the rewriter system instruction + `reasoning_effort="low"` + the generation prompt),
and dumps the prompt `input_ids` + the greedily generated token ids. The Rust gate
(`tests/reasoner_parity.rs`) reproduces the template byte-exactly, runs its own KV-cache greedy decode,
and checks the token stream against torch.

Generation is `do_sample=False` (greedy / argmax) for determinism — the vendor default temperature 0.7
samples, which can't be parity-checked. `max_new_tokens` is small (the gate compares the *leading*
greedy tokens; cross-build bf16 candle-vs-torch argmax can diverge on a late near-tie, exactly like the
encoder e2e, so the gate is prefix-based + a teacher-forced cache-equivalence check).

Strings (prompt / date) are stored as uint8 tensors (candle's `safetensors::load` returns tensors, not
metadata), mirroring the e2e golden's `date_utf8`.

Run (from the worktree root) with the transformers-5.8 lens-venv:
  & "C:\\Users\\Michael\\AppData\\Roaming\\SceneWorks\\python\\lens-venv\\Scripts\\python.exe" \\
      candle-gen-lens\\scripts\\dump_lens_reasoner_golden.py [out_dir]

Default out_dir: .scratch/lens-reasoner-goldens/  (not committed — regenerable).
"""
from __future__ import annotations

import datetime
import glob
import importlib.util
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoConfig, AutoTokenizer, Mxfp4Config

VENDOR_TE_CANDIDATES = [
    r"D:\repos\SceneWorks\apps\worker\scene_worker\_vendor\lens\text_encoder.py",
    r"D:\repos\SceneWorks\apps\desktop\python-src\scene_worker\_vendor\lens\text_encoder.py",
]

PROMPT = "a cat on a skateboard"
MAX_NEW_TOKENS = 24
DEVICE = "cuda"

# The vendor PromptReasoner system prompt (verbatim) + the local-path suffix.
SYSTEM_PROMPT = """
You are a prompt rewriter for a text-to-image model.
Your task is to convert the user's input into a single, precise, descriptive image prompt suitable for a text-to-image model.
Follow these rules strictly:

1. The output must be a clear and accurate description of a single image scene, written in the style of a text-to-image prompt.
  - Do not include explanations, reasoning, commentary, or meta text.
  - Do not ask questions.
  - Do not output multiple options.
  - Do not use uncertain, speculative, or alternative wording such as "maybe", "possibly", "perhaps", "or", "might", or "could".

2. Preserve the user's intended scene faithfully.
  - Do not change the objects, entities, attributes, actions, relationships, or core setting explicitly described by the user.
  - You may add reasonable visual details only when they help make the image concrete and coherent.
  - Any added details must be consistent with the user's description and must not introduce new important objects or alter the meaning.

3. If the image contains many main subjects of the same kind, describe each subject in detail, including humans, animals, objects, and any other prominent elements.
  - For each subject, include its appearance, color, size, shape, material, pose, expression, and position if applicable in the scene.
  - Make sure every main subject is clearly distinguishable from the others, such as in a scene with "4 dogs," describing each dog separately.

4. The output must fully cover the scene implied by the user's input.
  - Include the main subjects, relevant attributes, actions, spatial relationships, environment, and visible details necessary to render the scene.
  - If the user input is already sufficiently detailed and already suitable for image generation, keep it unchanged or only make minimal edits for fluency and clarity.

5. Resolve content that requires simple inference into explicit visual results when the result is unambiguous and visually representable.
  - Example: if the user says "the answer to 2+2 is written on the blackboard", output should explicitly describe "the blackboard shows 2+2=4".
  - Use only direct, necessary inference that is clearly implied by the user input.
  - Do not invent hidden facts, backstory, or ambiguous details.

6. Language rule:
  - If the user input is not in English, output in the same language.
  - Otherwise, output in English.

7. Output format:
  - Output exactly one final rewritten prompt.
  - Do not use bullet points, numbering, JSON, XML, Markdown, or quotation marks unless they are part of the scene itself.

Your goal is to produce a prompt that is concrete, visual, faithful to the user intent, and directly usable as input to a text-to-image model.
""".strip()


def find_snapshot() -> str:
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    snaps = sorted(
        p
        for p in glob.glob(str(hub / "models--microsoft--Lens-Turbo" / "snapshots" / "*"))
        if os.path.isdir(p)
    )
    if not snaps:
        sys.exit("no microsoft/Lens-Turbo snapshot found")
    return snaps[-1]


def load_encoder_cls():
    vendor_te = next((p for p in VENDOR_TE_CANDIDATES if os.path.isfile(p)), None)
    if vendor_te is None:
        sys.exit("no _vendor/lens/text_encoder.py found")
    spec = importlib.util.spec_from_file_location("lens_text_encoder", vendor_te)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.LensGptOssEncoder


def u8(s: str) -> torch.Tensor:
    return torch.tensor(list(s.encode("utf-8")), dtype=torch.uint8)


@torch.no_grad()
def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".scratch/lens-reasoner-goldens")
    out_dir.mkdir(parents=True, exist_ok=True)
    snap = find_snapshot()
    print(f"snapshot: {snap}", flush=True)

    tok = AutoTokenizer.from_pretrained(os.path.join(snap, "tokenizer"))
    system_prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        "Keep any reasoning private. The visible answer must contain only the final rewritten prompt."
    )
    conversation = [
        {"role": "system", "content": system_prompt, "thinking": None},
        {"role": "user", "content": PROMPT, "thinking": None},
    ]
    text = tok.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True, reasoning_effort="low"
    )
    input_ids = tok(text, return_tensors="pt", add_special_tokens=True).input_ids

    cfg = AutoConfig.from_pretrained(os.path.join(snap, "text_encoder"))
    cfg._attn_implementation = "eager"
    cfg._experts_implementation = "eager"
    print("loading text_encoder (MXFP4 → bf16)…", flush=True)
    model = (
        load_encoder_cls()
        .from_pretrained(
            os.path.join(snap, "text_encoder"),
            config=cfg,
            quantization_config=Mxfp4Config(dequantize=True),
            torch_dtype=torch.bfloat16,
            device_map=DEVICE,
        )
        .eval()
    )
    # NOTE: do NOT call set_selected_layers — the generate path must hit the stock LM forward, not the
    # feature-capture override.

    current_date = datetime.date.today().isoformat()
    print(f"input_ids L={input_ids.shape[1]}  date={current_date}", flush=True)
    print(f"greedy generate ({MAX_NEW_TOKENS} tokens)…", flush=True)
    out_ids = model.generate(
        input_ids.to(DEVICE),
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
    )
    new_tokens = out_ids[0, input_ids.shape[1]:]
    decoded = tok.decode(new_tokens, skip_special_tokens=False)
    print("generated:", repr(decoded), flush=True)

    tensors = {
        "input_ids": input_ids.to(torch.int64).cpu(),
        "new_tokens": new_tokens.to(torch.int64).cpu().reshape(1, -1),
        "prompt_utf8": u8(PROMPT),
        "date_utf8": u8(current_date),
    }
    dst = out_dir / "lens_reasoner_golden.safetensors"
    save_file(tensors, str(dst))
    print(f"wrote {dst}  (L={input_ids.shape[1]}, new={new_tokens.shape[0]})")


if __name__ == "__main__":
    main()
