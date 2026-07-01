# Full Codebase Review — candle-gen — 2026-07-01

## Executive summary

- **Repository at a glance:** Rust workspace, 26 crates (1 core + 25 model-family provider crates), ~378 source files, ~124k LOC of Rust. Rust-native diffusion/vision model inference on the candle ML framework; the Windows/CUDA sibling of mlx-gen, sharing the backend-neutral `gen_core` contract.
- **Coverage:** All 26 workspace crates (src/, examples/, tests/, manifests), `scripts/`, `.github/workflows/ci.yml`, and the root workspace config were reviewed by eight parallel deep-review passes plus a cross-cutting duplication scan. Excluded: `vendor/candle-kernels` (deliberate local fork of upstream, reviewed for provenance only), `target/`, binary test fixtures. See Coverage notes.
- **Headline:** The codebase is unusually well-documented, parity-disciplined, and honest at its API boundaries — error handling and capability validation are strong, and most raw duplication is a sanctioned porting pattern. The top risks are (1) **two verified numerical-parity defects in the SD3.5 port** (an AdaLN scale/shift swap in the final joint block and CLIP pooled conditioning taken from the wrong token position) that silently degrade every SD3.5 render, (2) **hard-won fixes not propagating across sibling crates** (the i32 attention-overflow guard exists in flux2 only; the quant-seam matmul strategy diverges between twins), and (3) **copy-paste pipeline scaffolding** (~25 copies of the loader helper, 8 copies of the adapter-merge skeleton, triplicated provider plumbing) that has already produced at least one behavioral drift bug (Kolors scheduler routing).
- **Counts:** Critical: 0 | High: 3 | Medium: 29 | Low: 35 | Info: 11 (78 findings).

## Critical findings

None found. No exploitable security vulnerability, data-loss risk, or production-blocking defect was identified.

## High findings

#### [F-001] Fix the swapped scale/shift in the SD3.5 final joint block's context AdaLN
- **Category:** bad-pattern
- **Severity:** High
- **Location:** `candle-gen-sd3/src/transformer.rs:648-654`
- **Finding:** Both branches of `if self.context_pre_only { … } else { … }` are byte-identical: `modulate(&layer_norm(txt)?, &cm[1], &cm[0])`. For the 6-chunk AdaLN-Zero case (shift first) `scale = cm[1]` is correct, but the final block's `norm1_context` is the 2-chunk diffusers `AdaLayerNormContinuous`, whose order — documented in this same file at lines 13, 232, and 697-702, and pinned by the sc-7881 `norm_out` fix — is `scale, shift = chunk(2)` with **scale first**. The pre-only branch should be `modulate(…, &cm[0], &cm[1])`; as written it applies `(1+shift)·LN + scale`. **Verified during synthesis**: `AdaLayerNormZero::forward` (lines 253-256) returns chunks in raw linear order with no reordering.
- **Impact:** The normalized text tokens feed the last joint attention's K/V, so every SD3.5 Large/Turbo/Medium render numerically diverges from the diffusers reference in the final block. Subtle enough to survive the C6 coherence eyeball (unlike the `norm_out` twin bug, which scrambled renders) — exactly the epic-7841 "AdaLN bug magnet" class.
- **Suggested fix:** Change the `context_pre_only` branch to `modulate(&layer_norm(txt)?, &cm[0], &cm[1])`, collapse the now-meaningful `if/else`, and add a 2-chunk `norm1_context` order unit test mirroring `adaln_continuous_chunk_order_is_scale_then_shift`. Re-validate with a component-level diffusers parity run.
- **Confidence:** High (code-level inconsistency is certain; visual magnitude unmeasured without weights)

#### [F-002] Pool the SD3.5 CLIP hidden at the first EOS, not the last padding token
- **Category:** bad-pattern
- **Severity:** High
- **Location:** `candle-gen-sd3/src/conditioning.rs:150-158` (with `fit_clip_tokens` at 139-148 and the `pad_id = eos_id` call at 268)
- **Finding:** `eos_position` uses `Iterator::max_by_key`, which returns the **last** maximal element on ties. `encode_clip` pads rows with the EOS id (`fit_clip_tokens(ids, eos_id, eos_id)`), so for every prompt shorter than 77 tokens all pad slots tie at the max id and the pooled hidden is taken from position 76 (trailing pad) instead of the first EOS. Torch/HF `argmax` — which diffusers' pooled `text_embeds` lookup relies on — returns the **first** occurrence, and the hidden states differ under causal attention. The unit test `eos_position_is_argmax` pads with id 9 ≠ EOS, so it never exercises the tie production always hits. **Verified during synthesis.**
- **Impact:** The pooled CLIP-L/bigG conditioning — which drives all AdaLN modulation via `temb` — is taken from the wrong sequence position for essentially every real prompt, on all three SD3.5 variants.
- **Suggested fix:** Return the first maximal index (e.g. `ids.iter().position(|&v| v == eos_id)` with an argmax fallback), and extend the test to a row padded with EOS.
- **Confidence:** High

#### [F-003] Add the i32 attention-scores overflow guard to chroma, lens, and the flux1 IP DiT
- **Category:** bad-pattern
- **Severity:** High
- **Location:** `candle-gen-chroma/src/transformer.rs:109-116` (+ `config.rs:211`, max_size 2048); `candle-gen-lens/src/transformer.rs:284-291` (+ `lib.rs:594-596`, max_size 2080); `candle-gen-flux/src/ip_dit.rs:55-69`; guard reference: `candle-gen-flux2/src/transformer.rs:71-127`
- **Finding:** flux2's `attention_budgeted` documents (sc-5487) that candle CUDA kernels index the `[B,H,Sq,Sk]` scores tensor with i32 — above ~2.147B elements the tail "silently corrupts" — and chunks queries to stay under 1.0B. The identical unguarded SDPA in chroma, lens, and the vendored flux1 `IpFlux` can exceed that budget at their **advertised** max sizes: chroma at 2048² (24 heads × ~16.9k joint seq ≈ 6.8B elements); lens at its largest 1440-base buckets with the always-on CFG batch of 2 (≈ 3.8B elements).
- **Impact:** Large-resolution renders on CUDA can silently produce garbage-attention images with no error — the exact failure mode sc-5487 already debugged once in flux2's edit path — while the descriptors advertise these sizes as supported.
- **Suggested fix:** Lift `attention_budgeted` into `candle-gen` and reuse it in chroma/lens/flux1-IP; or lower `max_size` on the affected descriptors until guarded.
- **Confidence:** Medium (overflow mechanism verified by flux2's in-repo analysis and test; corruption at those sizes not re-reproduced in this review)

## Medium findings

#### [F-004] Kolors txt2img silently ignores a scheduler-only curated request
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `candle-gen-kolors/src/pipeline.rs:151-154,180-195` (vs `control.rs:285-295`, `ip_provider.rs:287-297`)
- **Finding:** `Pipeline::render` engages the curated path only on a curated *sampler* name; `req.scheduler` is never consulted. The control and IP providers carry the same block **plus** an explicit `scheduler_curated` check. The descriptor advertises the curated scheduler menu and `validate` accepts e.g. `scheduler: Some("karras")` with the default sampler — but txt2img then renders with the native schedule.
- **Impact:** A validated `karras`/`sgm_uniform` txt2img request silently renders with the wrong σ-schedule — the "false-capability trap" this crate's own docs warn against, produced by drift among three copies of the same routing block.
- **Suggested fix:** Add the `scheduler_curated` check to `Pipeline::render`, or extract the routing decision into one shared helper (see F-021).
- **Confidence:** High

#### [F-005] SCAIL-2 advertises MultiReference conditioning it silently drops
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `candle-gen-scail2/src/pipeline.rs:65-70` (descriptor), `pipeline.rs:343` (`additional: Vec::new()`)
- **Finding:** The descriptor lists `ConditioningKind::MultiReference`, so `validate_request` accepts multi-reference requests, but `Scail2::run` hardcodes `additional: Vec::new()` — the engine layer's `CharacterRef` support is never wired. The module doc says multi-reference "awaits the worker request contract", yet the capability is already advertised.
- **Impact:** A multi-reference request validates, renders for minutes, and silently ignores the extra characters.
- **Suggested fix:** Map MultiReference conditioning onto `Scail2Job.additional`, or drop `MultiReference` from the descriptor until the worker contract lands (cite the tracking story).
- **Confidence:** High

#### [F-006] SCAIL-2 `build_segments` drops trailing driving frames and can underflow
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `candle-gen-scail2/src/generate.rs:160-177`
- **Finding:** In the multi-segment branch, windows stop when `start + len > total`, so up to `len − overlap − 1` (≤75 with the shipped 81/5 defaults) trailing driving frames are never generated, with no warning. Separately, `let stride = len - overlap;` underflows in `usize` when a direct engine consumer builds a `Scail2Job` (all fields `pub`) with `segment_overlap >= segment_len`.
- **Impact:** Output video is silently shorter than the driving clip for most clips > 81 frames; pathological job values panic (debug) or wrap (release) instead of erroring.
- **Suggested fix:** Emit a final shortened/overlapping tail segment (or document the drop and return the kept frame count), and validate `segment_overlap < segment_len` at the top of `generate`.
- **Confidence:** Medium (drop behavior verified from code; upstream `scail.py` parity for the tail not verified)

#### [F-007] FLUX.2 edit/control providers accept an empty prompt (the sc-8646 bug class)
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `candle-gen-flux2/src/edit_provider.rs:143-186`, `candle-gen-flux2/src/control_provider.rs:142-166` (vs the guards at `candle-gen-flux/src/control_provider.rs:317-319` and `candle-gen-flux2/src/lib.rs:407-411`)
- **Finding:** The registered flux2 txt2img path and the flux1 control provider reject empty prompts; the bespoke `Flux2Edit::generate` and `Flux2Control::generate` do not. `gen_core::TextTokenizer::tokenize("")` short-circuits to a (1, 0) encoding before the chat template runs, so an empty prompt reaches the TE as a zero-length sequence.
- **Impact:** A worker bug or user edge case produces a deep, confusing tensor-shape error (or degenerate conditioning) instead of a clean validation error.
- **Suggested fix:** Add the same `req.prompt.trim().is_empty()` guard at the top of both providers' `generate`.
- **Confidence:** High

#### [F-008] FLUX.1 control-image encode uses device RNG with a silently-ignored `set_seed`
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `candle-gen-flux/src/control_provider.rs:391-409` (esp. line 400, `let _ = self.device.set_seed(seed);`)
- **Finding:** `encode_control_latent` seeds the CUDA device RNG (error discarded) and lets the candle AutoEncoder *sample* the posterior. The crate's own determinism contract (sc-3673, cited in this file) states device `randn` is not launch-portable — the reason every other latent uses CPU-seeded `StdRng`.
- **Impact:** The control latent is deterministic per-launch at best, non-deterministic if `set_seed` fails silently, and never launch-portable — undermining the seed-reproducibility guarantee the surrounding code maintains.
- **Suggested fix:** Encode with the posterior mean (matching `Flux2Vae::encode_packed` and the boogu precedent), or sample with CPU-seeded `StdRng` noise; at minimum propagate the `set_seed` error.
- **Confidence:** Medium

#### [F-009] `read_scalar` panics on a malformed `.alpha` tensor in user-supplied adapter files (6 crates)
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `candle-gen-sdxl/src/adapters.rs:192-194`, `candle-gen-z-image/src/adapters.rs:170-172`, `candle-gen-sd3/src/adapters.rs:285-287`, `candle-gen-scail2/src/adapters.rs:154-156`, `candle-gen-qwen-image/src/adapters.rs:146-148` (main LoRA path at 205), `candle-gen-krea/src/adapters.rs:231-233` (both paths, 279 and 360)
- **Finding:** `read_scalar` does `to_vec1::<f32>()?[0]`, which panics on a size-0 `.alpha` tensor. In sdxl, scail2, and qwen-image the non-panicking `read_scalar_opt` exists in the same file — written precisely for this case — but the main kohya/PEFT LoRA path still calls the panicking variant on downloaded community adapters (qwen-image guards only its third-party LyCORIS path; krea has no safe variant at all).
- **Impact:** A malformed/truncated third-party LoRA crashes the worker thread instead of surfacing a typed error or a skipped key.
- **Suggested fix:** Route the `.alpha` read through `read_scalar_opt` semantics (`.first().copied().ok_or_else(…)`) in all six copies — or once, in the shared skeleton of F-018.
- **Confidence:** High

#### [F-010] Full checkpoints eagerly loaded onto the device to fetch a single tensor (3 sites)
- **Category:** efficiency
- **Severity:** Medium
- **Location:** `candle-gen-sd3/src/conditioning.rs:340-357`, `candle-gen-sdxl/src/conditioning.rs:103-115`, `candle-gen-clip/src/lib.rs:169-186,270-283`
- **Finding:** All three sites call `candle_core::safetensors::load(file, device)` — materializing **every** tensor of a CLIP checkpoint on the GPU — solely to extract `text_projection.weight`, immediately after (sd3, sdxl) the same files were already read by `build_clip_transformer`. The clip crate's image path additionally loads the entire file (including the unused text tower) eagerly.
- **Impact:** ~1.7 GB (sd3: CLIP-L + bigG) / ~1.4 GB (sdxl: bigG) of avoidable transient VRAM plus a second full disk read on every component load — on crates that elsewhere CPU-stage precisely to kill load transients (sc-8504).
- **Suggested fix:** Fetch the single tensor through an mmapped `VarBuilder`/`SafeTensors::tensor` (or load to CPU and upload one tensor).
- **Confidence:** High

#### [F-011] Tokenizers re-loaded and re-parsed from disk on every prompt encode (7 crates)
- **Category:** efficiency
- **Severity:** Medium
- **Location:** `candle-gen-wan/src/lib.rs:131-141` + `wan14b.rs:232-242` + `model_vace.rs:121-131` + `training.rs:158-167` (once **per dataset item** during caching); `candle-gen-z-image/src/pipeline.rs:330-345` + `control.rs:690-704` + `edit.rs:282-292`; `candle-gen-flux/src/pipeline.rs:386-406`; `candle-gen-flux2/src/lib.rs:195-205`; `candle-gen-sdxl/src/pipeline.rs:262-283`; `candle-gen-ideogram/src/pipeline.rs:178-199`; `candle-gen-scail2/src/generate.rs:270-279`; `candle-gen-qwen-image/src/lib.rs:164-181` + `control.rs:181-199` + `control_fun.rs:205-223` (while the same crate's `QwenEdit` caches correctly)
- **Finding:** These paths call `TextTokenizer::from_file`/`Tokenizer::from_file` inside the per-request encode, re-parsing multi-megabyte `tokenizer.json` files (UMT5/Qwen unigram models are the worst) once or twice per generate — despite each crate carefully caching the heavy TE/DiT/VAE components. Chroma, lens, kolors, and LTX show the correct pattern (tokenizer cached in `Components`).
- **Impact:** Tens-to-hundreds of ms of blocking file I/O + JSON parse per request (per branch under CFG), and O(dataset) redundant parses in the wan trainer; failures also surface late instead of at load.
- **Suggested fix:** Load the tokenizer once into each crate's `Components` (or a `OnceLock`), mirroring the sibling crates.
- **Confidence:** High

#### [F-012] Step-invariant tensors rebuilt inside the denoise/frame hot loops (8 crates)
- **Category:** efficiency
- **Severity:** Medium
- **Location:** `candle-gen-ideogram/src/transformer/model.rs:110-133,157-211`; `candle-gen-flux2/src/transformer.rs:680-684` (+ `pos_embed.rs:41-62`); `candle-gen-lens/src/transformer.rs:540-545` (+ `rope.rs:52-120`); `candle-gen-ltx/src/transformer.rs:726-758` (+ `rope.rs:201-268`); `candle-gen-scail2/src/model.rs:479-526`; `candle-gen-qwen-image/src/transformer.rs:490-492,555-557,688-690,839-841`; `candle-gen-krea/src/transformer/mod.rs:143-151`; `candle-gen-boogu/src/transformer/mod.rs:182-191`
- **Finding:** Multiple DiT `forward`s recompute, per denoise step (×2 under CFG): host-built RoPE cos/sin tables (LTX builds **four** — ~4.7M trig evaluations per call), text/image conditioning projections, and attention masks. Ideogram is the worst case: it rebuilds a `[B,1,L,L]` segment mask in a single-threaded host loop (~1 GiB at 2048²), round-trips the `indicator` tensor device→host per forward, and then `broadcast_add`s the mask onto the scores in each of 34 blocks — even though this pipeline always passes uniform segment ids, making the mask provably all-zeros. All of these depend only on fixed geometry, not σ. Chroma and wan demonstrate the correct hoisted pattern.
- **Impact:** Redundant host compute, H2D transfers, and GPU-idle bubbles multiplied by 8–48 steps; for Ideogram at large resolutions, tens of GB of avoidable allocation/transfer per render.
- **Suggested fix:** Hoist mask/RoPE/conditioning construction out of `forward` into per-render prepared state (a "prepared conditioning" struct), and skip the mask add when all segment ids are equal.
- **Confidence:** High

#### [F-013] CFG/negative-branch work runs even when guidance disables it (3 crates)
- **Category:** efficiency
- **Severity:** Medium
- **Location:** `candle-gen-lens/src/lib.rs:279-324`; `candle-gen-sd3/src/pipeline.rs:296-312,401-427`; `candle-gen-wan/src/wan14b.rs:335-342,388-394`
- **Finding:** Lens always builds the `[pos; neg]` batch-2 conditioning and runs the uncond half each step even at `guidance == 1.0` — the **default** for `lens_turbo` — where `cfg_rescale` reduces exactly to `cond`. SD3.5 Large/Medium likewise run two DiT forwards per step for an explicit `guidance: Some(1.0)`. Wan 14B unconditionally UMT5-encodes the negative prompt (a 24-layer forward over 512 tokens) and projects it through both experts even when the denoise loop never uses it (`guidance <= 1.0`); the 5B path gates this correctly, as do Z-Image base and SenseNova.
- **Impact:** Up to 2× DiT compute per step on the most latency-sensitive distilled paths, for mathematically identical output.
- **Suggested fix:** Skip the uncond encode/forward when effective guidance is 1.0 (verify against the mlx twins for parity intent first).
- **Confidence:** Medium (numerics are exact; deliberate bit-parity retention is the only open question)

#### [F-014] SAM3 reads back all 200 query masks (~66 MB) from the device every video frame
- **Category:** efficiency
- **Severity:** Medium
- **Location:** `candle-gen-sam3/src/video.rs:289-327`
- **Finding:** `run_detection` copies the full `[1,200,288,288]` `pred_masks` tensor to host (`to_vec1`, ~16.6M f32) per frame, then keeps only the handful of queries whose score passed `SCORE_THRESH_DET`.
- **Impact:** A large synchronous PCIe transfer per frame that scales with `num_queries`, not detections — measurable latency in the person-track pipeline.
- **Suggested fix:** Compute `probs` first, then `narrow`/`index_select` only the kept query rows before the host readback.
- **Confidence:** High

#### [F-015] SAM3 per-object memory banks grow without bound over a video
- **Category:** efficiency
- **Severity:** Medium
- **Location:** `candle-gen-sam3/src/video.rs:196-206,586-621` (also `461-466,494-500`)
- **Finding:** Every frame appends a `FrameMem` with `maskmem_features` + `maskmem_pos_enc` (~2.7 MB of device tensors) to `non_cond` for every live object; nothing evicts entries older than what `gather_memory` can read (7 spatial / 16 pointer frames back). `unmatched_frames`/`overlap_pairs` bookkeeping also grows unpruned.
- **Impact:** ~2.4 GB of VRAM for 300 frames × 3 objects; long clips exhaust VRAM mid-job on the person-track lane.
- **Suggested fix:** Evict `non_cond` entries older than `max(NUM_MASKMEM, MAX_OBJ_PTRS)` frames (keeping `object_pointer` for the pointer window); cap the hotstart bookkeeping.
- **Confidence:** Medium (growth is certain from the code; incident severity depends on production clip lengths)

#### [F-016] SAM3 rebuilds constant geometry tensors per object per frame (and re-encodes the constant text prompt)
- **Category:** efficiency
- **Severity:** Medium
- **Location:** `candle-gen-sam3/src/tracker.rs:1424,1453,1473-1479,1505-1527`; `detr.rs:516`; `video.rs:289-297` → `model.rs:149-158`; also `vision.rs:457-492` (the always-computed, always-discarded 36² FPN level, see `model.rs:199-201` / `tracker.rs:626-636`)
- **Finding:** `decode_tracked_frame`/`decode_mask_conditioning_frame` rebuild `dense_pe(72)` (host coordinate loop + Gaussian matmul) and the 288→1008 bilinear resize matrix (~1.2 MB host build + upload) once per tracked object per frame; the detector rebuilds its 72² sine PE per frame; the fixed concept prompt is re-encoded through the 24-layer CLIP text tower every frame; and `fpn_from_backbone` always computes the scale-0.5 FPN branch both consumers discard.
- **Impact:** Repeated host compute + H2D uploads inside the multi-object video hot loop, linear in objects × frames.
- **Suggested fix:** Cache the deterministic tables at model load (keyed by grid size); encode the text once per `propagate`; add an FPN variant that skips the discarded level.
- **Confidence:** High

#### [F-017] SeedVR2 re-transposes and copies every Linear weight on every forward
- **Category:** efficiency
- **Severity:** Medium
- **Location:** `candle-gen-seedvr2/src/nn.rs:11-16` (used by `dit.rs:57-59`, `vae.rs:30-32`)
- **Finding:** Every dense `Linear::forward` calls `w.t()?.contiguous()?`, materializing a transposed copy of the full weight matrix per call — ~10+ Linears per block × 32/36 blocks per step. The sibling sam3 `Linear` stores `weight_t` pre-transposed at load (a drift between two copies of the same seam, see F-025).
- **Impact:** Several GB of transient transposed-weight allocation and extra kernel launches per upscale step/tile — pure overhead.
- **Suggested fix:** Pre-transpose in `Linear::load` (as sam3 does), or drop the `.contiguous()` and let candle's matmul consume the transposed view (candle-nn's own `Linear` does).
- **Confidence:** High

#### [F-018] Adapter-merge skeleton copy-pasted across 8 crates (~7,300 LOC total)
- **Category:** redundant
- **Severity:** Medium
- **Location:** `candle-gen-{krea,lens,qwen-image,scail2,sd3,sdxl,wan,z-image}/src/adapters.rs`
- **Finding:** `PEFT_PREFIXES`, `LOKR_SUFFIXES`, `AdapterFile`/`read_adapter`, `strip_peft_prefix`, `build_kohya_table`, `Role`/triple classification, `read_scalar`, `merge_into`, `declares_lokr`, and the `merge_adapters` shell (with its "no target matched" loud-error contract) are byte-near-identical across eight crates; only the per-model stem/target resolution is family-specific. The copies have already drifted on behavior, not just style: `classify_lora_key` vs `split_lora_role` shapes; panicking vs non-panicking scalar reads (F-009); an `eprintln!` in sd3/qwen-edit only (F-051); qwen-image checks LoKr via `wmeta::is_lokr_network_type`/`parse_rank_alpha` while krea string-compares `networkType == "lokr"` directly; LoHa support exists on one side only.
- **Impact:** Every format fix (new PEFT prefix, alpha-handling, report semantics) must land 8×; drift is already observable.
- **Suggested fix:** Lift the format-parsing/merge-report skeleton into `candle_gen::train::lora` (both trainer-reconstruction halves already live there), parameterized by a `resolve_targets` closure per family.
- **Confidence:** High

#### [F-019] The "sorted `.safetensors` in dir → unsafe mmap → VarBuilder" loader is duplicated ~34 times workspace-wide
- **Category:** redundant
- **Severity:** Medium
- **Location:** Representative: `candle-gen-flux/src/{pipeline.rs:190-204,ip_provider.rs:151-165,control_provider.rs:132-146}`, `candle-gen-flux2/src/{lib.rs:140-151,control_provider.rs:244-266}`, `candle-gen-chroma/src/{pipeline.rs:94-109,text.rs:79-93}`, `candle-gen-lens/src/{lib.rs:124-145,reasoner.rs:78-98}`, `candle-gen-kolors/src/{pipeline.rs:112-127,control.rs:126-141,ip_provider.rs:136-151}`, `candle-gen-scail2/src/pipeline.rs:93-117,146-162`, `candle-gen-ideogram/src/pipeline.rs:71-98`, `candle-gen-wan/src/{lib.rs,wan14b.rs,model_vace.rs}` (3 copies), `candle-gen-z-image/src/{pipeline.rs,edit.rs,control.rs}` (3 copies), `candle-gen-qwen-image/src/{lib.rs:125-149,edit.rs:92-119,control.rs:100-128,control_fun.rs:122-150,vision_language.rs:116-144}` (5 copies in one crate), `candle-gen-krea/src/{loader.rs:31-54,vae.rs:30-45}`, `candle-gen-boogu/src/{loader.rs:22-45,pipeline.rs:98-114}`
- **Finding:** The same `component_vb`-shaped helper (list a snapshot subdir, sort, error-if-empty, `unsafe { VarBuilder::from_mmaped_safetensors }`) is re-implemented per provider and often per provider *variant*, with error-string drift and one behavioral drift: flux2's `control_var_builder` places its "no .safetensors" check where the single-file arm can never trigger it, so a missing control checkpoint surfaces as a raw mmap error instead of the crafted message flux1 gets right (`candle-gen-flux2/src/control_provider.rs:244-266` vs `candle-gen-flux/src/control_provider.rs:150-164`).
- **Impact:** ~25 copies of an `unsafe`-adjacent load path; any improvement (shard handling, better errors, the SAFETY invariant) must be applied everywhere; the unsafe surface is scattered instead of one audited function.
- **Suggested fix:** One `component_vb(dir, dtype, device, label)` in `candle-gen` (which already hosts `train::flow_match::component_vb`); fix the flux2 single-file check in the process.
- **Confidence:** High

#### [F-020] Wan quadruplicates its tokenize→pad-512→UMT5-encode routine and component loader
- **Category:** redundant
- **Severity:** Medium
- **Location:** `candle-gen-wan/src/lib.rs:93-173`, `wan14b.rs:133-161,230-267`, `model_vace.rs:80-104,119-156`, `training.rs:151-192`
- **Finding:** The tokenize → empty-guard (sc-7078) → UMT5-encode → zero-pad/truncate-to-512 routine (including the identical sc-7078 comment) and `component_vb` exist in four copies across the 5B, 14B, VACE, and trainer modules; the trainer copy has already drifted (extra `to_dtype(F32)`).
- **Impact:** ~200 duplicated lines guarding a known correctness trap (the 512-pad collapse); a future tokenizer fix must land 4× — exactly the bug class sc-7078 was.
- **Suggested fix:** Crate-private `umt5_encode_padded(…)` + shared `component_vb`, called from all four sites.
- **Confidence:** High

#### [F-021] Kolors triplicates ~150 lines of pipeline scaffolding across its three entry points
- **Category:** redundant
- **Severity:** Medium
- **Location:** `candle-gen-kolors/src/control.rs:386-433`, `ip_provider.rs:370-429` vs `pipeline.rs:305-351`
- **Finding:** `encode`, `build_time_ids`, `initial_noise`, `decode`, and the curated-vs-native routing block are copy-pasted verbatim across `Pipeline`, `KolorsControl`, and `IpAdapterKolors`; the routing drift is F-004's root cause.
- **Impact:** Three-way maintenance of identical numerics; the seam already produced the crate's one behavioral bug.
- **Suggested fix:** Extract a `KolorsCommon` (encode/time_ids/noise/decode + routing) shared by the three.
- **Confidence:** High

#### [F-022] Z-Image triplicates loader/decode/preprocess/tokenizer plumbing across pipeline/edit/control
- **Category:** redundant
- **Severity:** Medium
- **Location:** `candle-gen-z-image/src/pipeline.rs:246-295,334-345,667-715`; `edit.rs:59-94,280-360,362-391`; `control.rs:64-98,689-752,771-821,899-927`
- **Finding:** Near-identical copies of `component_vb`, VAE decode→RGB8, `[0,255]→[-1,1]` preprocessing, the Qwen tokenizer constants, `text_embeddings`/`encode_cap`, and (pipeline vs control) `uncond_embeddings` with the sc-8646 fix; `init_time_step` exists twice with different signatures. The sc-8646 class of fix already had to land in two places.
- **Impact:** Any tokenizer-policy or VAE-scale fix must be replicated 2–3×; drift risk is demonstrated, not hypothetical.
- **Suggested fix:** A crate-private `common.rs` for the shared plumbing; `edit`/`control` call into `crate::pipeline` items (several are already `pub(crate)`).
- **Confidence:** High

#### [F-023] FLUX.1 triplicates its component-loading stack and the `IpFlux` forward body
- **Category:** redundant
- **Severity:** Medium
- **Location:** Loading: `candle-gen-flux/src/pipeline.rs:137-204` vs `ip_provider.rs:136-262` vs `control_provider.rs:116-280` (+ the CPU-seeded noise block ×3). Forwards: `candle-gen-flux/src/ip_dit.rs:593-641,651-715,731-821`
- **Finding:** The CLIP/T5/VAE load, config parse, parity-critical constants, and seeded-noise block are copy-pasted across the three flux1 providers. In `ip_dit.rs`, `forward`, `forward_injected`, and `forward_control` triplicate a ~60-line body where `forward_control(.., None)` is documented byte-identical to `forward_injected`, which with `injector = None` is byte-identical to `forward`.
- **Impact:** Parity-critical constants and the guidance/embedding preamble can drift independently across three copies each, in a file deliberately kept in lockstep with upstream candle.
- **Suggested fix:** A `flux1::components` module for the loads; implement the two specialized forwards as thin wrappers over `forward_control`.
- **Confidence:** High

#### [F-024] FLUX.2 triplicates the quant-vs-dense TE+DiT loader
- **Category:** redundant
- **Severity:** Medium
- **Location:** `candle-gen-flux2/src/lib.rs:157-184` vs `edit_provider.rs:108-139` vs `control_provider.rs:99-138`
- **Finding:** The `match quant { Some(q) => CPU-stage → quantize_onto(GPU), None => dense }` block for `Qwen3TextEncoder` + `Flux2Transformer` is copy-pasted in three loaders. (The staging strategy itself is deliberate; the triplication is not.)
- **Impact:** A staging-strategy change (e.g. pre-quantized snapshot consumption) must be replicated three times.
- **Suggested fix:** `Pipeline::load_te_and_dit()` called from all three.
- **Confidence:** High

#### [F-025] Four copies of the dense-or-quantized `QLinear` seam with load-bearing drift
- **Category:** redundant
- **Severity:** Medium
- **Location:** `candle-gen-flux2/src/quant.rs:26-113` vs `candle-gen-lens/src/quant.rs:37-107`; `candle-gen-sam3/src/common.rs:99-218` vs `candle-gen-seedvr2/src/dit.rs:28-118` + `quant.rs:14-28`
- **Finding:** Four same-shaped `Dense|Quantized` Linear seams (with identical `ggml_dtype` mappings — lens's copy even calls itself "the single source of truth") have diverged on load-bearing behavior: lens deliberately dequantizes-on-forward (sc-7702, the gpt-oss outlier fix) while **flux2 routes through `QMatMul::forward`** — the int8 `fast_mmq` activation-quant path lens documents as corrupting under outliers; sam3 pre-transposes the weight at load while seedvr2 re-transposes per forward (F-017).
- **Impact:** A future model with outlier activations on flux2's seam reproduces the sc-7702 black-render failure; maintainers must know which of four identically-named types they are touching.
- **Suggested fix:** One shared `QLinear` in `candle-gen` with an explicit `MatmulStrategy { Int8Fast, DequantDense }` knob and one `ggml_dtype`, documenting when each is safe.
- **Confidence:** Medium (flux2-dev was GPU-validated, so its activations presumably tolerate `fast_mmq` today; the latent risk is the point)

#### [F-026] Budgeted VAE-tiling machinery duplicated 1:1 between wan and ltx
- **Category:** redundant
- **Severity:** Medium
- **Location:** `candle-gen-wan/src/vae.rs:502-578,658-806` vs `candle-gen-ltx/src/vae.rs:221-296,322-459`
- **Finding:** `decode_tiled` (~80 lines of tile/narrow/blend/accumulate/normalize), `nvidia_smi_total_gib`, the 0.85 safe-frac, the env-override budget resolver, and the `TilingBudgetError` mapping are byte-near-identical in both crates (and per their own comments, echoed in seedvr2). Both files say "de-dupe into candle-gen core is a tracked follow-up" **without a story id** — contrary to the repo's own follow-ups-go-in-stories convention.
- **Impact:** Three-way drift risk in genuinely tricky blending/geometry code; z16 already lacks the budgeted entry z48 gained.
- **Suggested fix:** Lift a generic `decode_tiled(decode_fn, plan, …)` + budget resolver into `candle_gen`, parameterized by cost model and env-var name; file the story and cite it.
- **Confidence:** High

#### [F-027] Dead `LtxDiT` and its RoPE helpers survive the AV fold-in
- **Category:** dead-code
- **Severity:** Medium
- **Location:** `candle-gen-ltx/src/transformer.rs:288-375`, `candle-gen-ltx/src/rope.rs:61-125,300-317`
- **Finding:** `LtxDiT` is `pub` with zero references anywhere in the workspace (the provider builds only `AvDiT` since sc-5495); `precompute_split_freqs` and `video_rope` are referenced only by it. Workspace-wide grep confirms no example, test, or crate consumes them.
- **Impact:** ~180 lines of unmaintained model code sharing internals (`Attention`/`VideoBlock`) that keep evolving for the AV path; misleads readers about which DiT is live.
- **Suggested fix:** Delete `LtxDiT`, `video_rope`, and `precompute_split_freqs`; keep `VideoBlock`, which `AvDiT` reuses.
- **Confidence:** High

#### [F-028] SVD dtype documentation contradicts the code in three places
- **Category:** readability
- **Severity:** Medium
- **Location:** `candle-gen-svd/src/lib.rs:17-18,68-70` (vs the actual f32 defaults at 72-85); `candle-gen-svd/src/transformer.rs:75-76,268-269`
- **Finding:** The crate doc says the UNet + image encoder run **fp16**; `Components::load`'s doc says "fp16 on CUDA / f32 on CPU"; `transformer.rs` claims both "bf16 on CUDA" and "fp16 on CUDA" in two comments — while the body deliberately defaults everything to **f32** (fp16/bf16 only via `SVD_FORCE_*` env vars, with a detailed and correct rationale about fp16 NaNs).
- **Impact:** Anyone sizing VRAM or debugging precision is misled at the module-doc level; the valuable f32 rationale is buried under contradicting headers.
- **Suggested fix:** Rewrite the three doc sites to state f32-default + env-gated experimental overrides.
- **Confidence:** High

#### [F-072] Boogu Turbo advertises a sampler menu it silently ignores
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `candle-gen-boogu/src/lib.rs:61,226-232`, `candle-gen-boogu/src/pipeline.rs:177-217`
- **Finding:** `descriptor_turbo()` advertises `samplers = ["lcm", "euler_ancestral", "dpmpp_sde"]`, so `validate_request` accepts those values — but `render_turbo` never reads `req.sampler` or `req.scheduler`; it always runs the bespoke DMD predict/renoise loop. The Base and Edit paths route `req.sampler` into `run_flow_sampler` correctly.
- **Impact:** A request selecting `euler_ancestral` vs `lcm` on `boogu_image_turbo` validates and produces byte-identical output — the same false-capability trap as F-004/F-005.
- **Suggested fix:** Route the Turbo loop through the curated-sampler framework, or shrink the advertised menu to the single native loop until wired (cite the story if this is a deliberate mlx-mirror stopgap).
- **Confidence:** High (that the value is ignored)

#### [F-073] Silent error-swallowing config reads downgrade behavior on corrupted snapshots
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `candle-gen-qwen-image/src/edit.rs:183-189` (`read_zero_cond_t`); same pattern at `candle-gen-krea/src/config.rs:100-106` and `candle-gen-krea/src/text_encoder.rs:104-119`
- **Finding:** `read_zero_cond_t` chains `.ok()`/`.and_then()` and defaults to `false` on *any* failure — missing file, I/O error, malformed JSON, wrong type. Only the legitimately-absent-key case (2509 snapshots) should default; a corrupted/partially-downloaded `-2511` snapshot silently switches the modulation to the single-timestep 2509 math. Krea's `model_index.json` reads silently fall back to defaults the same way.
- **Impact:** Degraded renders with no error or log line from a damaged snapshot — exactly the "renders wrong output silently" failure mode the adapter-merge code was designed to prevent.
- **Suggested fix:** Distinguish `NotFound`/absent-key (→ default) from read/parse errors (→ `Err` or at least a logged warning).
- **Confidence:** High

#### [F-074] Qwen-Image's Fun-Controlnet lane is a wholesale copy of the InstantX lane
- **Category:** redundant
- **Severity:** Medium
- **Location:** `candle-gen-qwen-image/src/control_fun.rs:64-93,122-150,205-223,321-366` vs the byte-identical counterparts in `candle-gen-qwen-image/src/control.rs:48-77,100-128,181-199,308-353`
- **Finding:** The Fun-Controlnet lane duplicated the InstantX lane's entire scaffolding verbatim (request struct, loader, prompt encoder, image preprocessor, image converter), changing only the branch forward call and error prefixes — `preprocess_control_image` and `to_image` are character-identical apart from the label.
- **Impact:** Any preprocessing fix must land twice inside one crate; the InstantX retirement (Phase B, sc-8246) must untangle the shared helpers anyway.
- **Suggested fix:** Extract a private `control_common` module parameterized by label.
- **Confidence:** High

#### [F-075] Qwen-Image edit header warns of a missing attention-overflow fix that already landed
- **Category:** readability
- **Severity:** Medium
- **Location:** `candle-gen-qwen-image/src/edit.rs:17-19` (fix at `transformer.rs:111-161`; validated by `edit_validate.rs:221-267`)
- **Finding:** The module doc says a joint sequence over ~2.1B score elements "would silently corrupt — keep the output ≤ ~1536² until the shared `JointAttention` gains query-row chunking (the FLUX.2 fix)." That chunking exists (`ATTN_SCORES_BUDGET` + `attention_budgeted`, sc-6217), and the 1536² edit validation exists *because* it landed.
- **Impact:** Gives operators a false resolution ceiling and implies a live silent-corruption bug in a guarded path — the most dangerous kind of doc rot in a parity-driven codebase (and the mirror image of F-003, where the fix is genuinely missing).
- **Suggested fix:** Rewrite the note to state the chunking exists, citing sc-6217/`ATTN_SCORES_BUDGET`.
- **Confidence:** High

## Low findings

#### [F-029] Runtime HF-hub downloads on the SDXL render path are unpinned
- **Category:** security
- **Severity:** Low
- **Location:** `candle-gen-sdxl/src/pipeline.rs:181-187` (used at `pipeline.rs:272,333`, `loaders.rs:80,90`, `conditioning.rs:93-96`, `training.rs:154-157,559`)
- **Finding:** `hf_get` resolves the fp16-fix VAE (`madebyollin/sdxl-vae-fp16-fix`) and both CLIP tokenizers from the hub's mutable default revision at runtime, with no revision pin or digest check.
- **Impact:** A compromised or force-pushed upstream silently changes production weights/tokenization; an upstream deletion (cf. this project's own Lens takedown history) breaks cold-cache generation at request time.
- **Suggested fix:** Pin by commit SHA via `repo_with_revision` (or vendor the artifacts into the deployed snapshot layout) and surface a clear offline-cache error.
- **Confidence:** High

#### [F-030] `nvidia-smi` invoked via PATH/process-search-order from library decode paths
- **Category:** security
- **Severity:** Low
- **Location:** `candle-gen-seedvr2/src/video.rs:73-88`, `candle-gen-wan/src/vae.rs:720-735`, `candle-gen-ltx/src/vae.rs:376-391`
- **Finding:** Budget probes spawn `Command::new("nvidia-smi")` unqualified. On Windows, `CreateProcessW` searches the application directory (and legacy cwd) before `PATH`, so a planted binary runs with the worker's privileges; a hijacked PATH silently changes the VRAM budget. The subprocess also runs on every budgeted decode.
- **Impact:** Low-likelihood, deployment-dependent escalation vector; environment-dependent budget behavior that is hard to observe.
- **Suggested fix:** Resolve the absolute path once and cache (`OnceLock`), or prefer a CUDA-API query (`cudaMemGetInfo`); log which budget source won.
- **Confidence:** Medium

#### [F-031] Mutex poison converts one panic into permanent panics on shared generator caches (6 crates)
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `candle-gen-sdxl/src/lib.rs:243-246`; `candle-gen-seedvr2/src/lib.rs:130,213`; `candle-gen-wan/src/{conv3d.rs:87-118, vae16.rs:109-127, lib.rs:288-291, wan14b.rs:494-498, model_vace.rs:294-298}`; `candle-gen-ltx/src/lib.rs:338-342`; `candle-gen-svd/src/lib.rs:261-264`
- **Finding:** Component and streaming-conv caches lock with `.unwrap()`/`.expect()`. A panic while holding the lock (e.g. a CUDA OOM-turned-panic mid-decode) poisons the mutex, after which every subsequent `generate` on the shared `Arc<dyn Generator>` panics instead of returning `Err`.
- **Impact:** One transient failure can wedge a long-lived worker lane into a panic loop until restart.
- **Suggested fix:** `lock().unwrap_or_else(|e| e.into_inner())` — the cached state is overwrite-on-miss and safe to reuse — or map poisoning to a typed error.
- **Confidence:** Medium (requires a panic-while-locked to trigger; the pattern is verified)

#### [F-032] Bespoke providers accept `steps == 0` and silently decode noise (3 crates)
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `candle-gen-sdxl/src/ip_provider.rs:188-277`; `candle-gen-instantid/src/model.rs:333-381`; `candle-gen-scail2/src/pipeline.rs:264-268,348`
- **Finding:** The registered `SdxlGenerator::validate` rejects `steps: Some(0)` explicitly, but the worker-driven IP-Adapter, InstantID, and SCAIL-2 entry points have no steps floor: an empty schedule means the noise (or prior) is VAE-decoded unchanged.
- **Impact:** A misconfigured request burns GPU time and returns garbage instead of a fast typed error — on SCAIL-2, minutes of video render.
- **Suggested fix:** Add a `steps >= 1` check at the top of the bespoke `generate*` entry points, mirroring the registered path.
- **Confidence:** High

#### [F-033] Optimizer-name normalization silently substitutes via substring matching
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `candle-gen/src/train/optim.rs:25-42`
- **Finding:** `normalize` classifies by `contains`: `"adamax"`, `"radam"`, `"nadam"` all contain `"adam"` and silently train as plain Adam; `"adamw8bit"` maps to full-precision AdamW.
- **Impact:** A worker request trains with a different optimizer than requested instead of failing validation — inconsistent with the trainer's own strict `timestep_type`/`loss_type` rejection.
- **Suggested fix:** Match an explicit alias whitelist (exact after separator-stripping); reject the rest.
- **Confidence:** High

#### [F-034] Final partial gradient-accumulation flush is under-scaled
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `candle-gen/src/train/flow_match.rs:624-636` (divisor at 344-350)
- **Finding:** When training ends (or cancels) with `k < accum` pending micro-grads, the flush still averages by `1/accum`, so the tail update is weighted `k/accum` of a normal update rather than a true mean of the `k` grads.
- **Impact:** The last optimizer update of any run with `steps % accum != 0` is silently weaker than intended; marginal on a converged LoRA, which is why it hasn't been noticed.
- **Suggested fix:** Track the pending micro-count and divide the flush by it, or document the down-weighting as deliberate.
- **Confidence:** Medium

#### [F-035] Adapter freeze/thaw visitor errors are swallowed with `let _ =` around preview rendering
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `candle-gen/src/train/flow_match.rs:579-582,604-607`
- **Finding:** `let _ = dit.visit_lora_mut(…)` discards the visitor `Result` for both the freeze and the thaw pass. If a `LoraHost` impl ever fails mid-walk, training resumes with some adapters still detached — their grads silently become `None`.
- **Impact:** The same silent-grad failure class the fused-ops rule guards against, reintroduced at the harness level.
- **Suggested fix:** Propagate with `?` (the function already returns `Result`).
- **Confidence:** High

#### [F-036] JoyCaption cancellation is only observed at token boundaries
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `candle-gen-joycaption/src/lib.rs:169-205`
- **Finding:** The gen-core `CancelFlag` is mirrored onto core-llm's flag only inside the `StreamEvent::Token` handler; a cancel during the long vision-tower + prompt prefill isn't seen until the first token.
- **Impact:** Worst-case cancel latency equals the full LLaVA prefill; GPU work continues for an abandoned job.
- **Suggested fix:** Bridge the flag so it is polled during prefill (shared flag type or wrapper), or document the boundary.
- **Confidence:** Medium

#### [F-037] CLIP embedder weights fallback picks an arbitrary first `*.safetensors` with no shard support
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `candle-gen-clip/src/lib.rs:247-265`
- **Finding:** When `model.safetensors` is absent, the loader takes the first `.safetensors` in unsorted `read_dir` order; a sharded snapshot would load one arbitrary shard and fail later with a bare missing-tensor error. The depth loader (`candle-gen-depth/src/common.rs:43-62`) sorts and merges shards correctly.
- **Impact:** A resharded upstream snapshot produces a confusing, machine-dependent error.
- **Suggested fix:** Sort candidates; merge shards (like depth) or error explicitly on multiple files.
- **Confidence:** High

#### [F-038] Skew-gate script masks the real `cargo tree` failure
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `scripts/check-gen-core-skew.sh:109-114` (message at 39-43)
- **Finding:** `cargo tree … 2>/dev/null` discards stderr; if `cargo tree` itself fails (fetch/auth/pin error), `evaluate` sees zero lines and reports "sceneworks-gen-core was not found in the build graph" — the wrong diagnosis.
- **Impact:** CI failures point developers at pin alignment when the problem is resolution, exactly when the lockstep protocol is under pressure.
- **Suggested fix:** Capture output and exit code separately; on nonzero exit print cargo's stderr with a distinct message.
- **Confidence:** High

#### [F-039] `package-cuda.ps1` shadows the PowerShell automatic variable `$matches`
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `scripts/package-cuda.ps1:79-90`
- **Finding:** The DLL-collection loop assigns `$matches = Get-ChildItem …`; `$Matches` is the automatic variable populated by `-match`, so any later `-match` in scope clobbers the list (PSScriptAnalyzer `PSAvoidAssignmentToAutomaticVariable`).
- **Impact:** A future edit adding a `-match` inside the loop silently corrupts the shipped-bundle DLL list.
- **Suggested fix:** Rename to `$dlls`.
- **Confidence:** High

#### [F-040] `control_scale == 0.0` silently remapped to 0.7 in the flux1 control provider
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `candle-gen-flux/src/control_provider.rs:335-341` (doc at 64-67)
- **Finding:** A request with `control_scale: 0.0` steers at the default 0.7, while one layer down `forward_control` and the parity tests define scale 0 as byte-identical to the base forward; `1e-9` and `0.0` behave completely differently.
- **Impact:** Callers cannot express "control off"; the API contradicts the engine's own ablation semantics and can mask worker bugs.
- **Suggested fix:** Make the field `Option<f32>` (None → default, Some(0.0) → no-op) or document loudly.
- **Confidence:** High

#### [F-041] Library-runtime panics on public entry points (lens assert, face preprocessing, scail2 mask)
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `candle-gen-lens/src/transformer.rs:515-521` (`assert_eq!` in the DiT forward); `candle-gen-face/src/face.rs:48-52,123-127` + `align.rs:200-204` (`assert!` in `pub` preprocessing); `candle-gen-scail2/src/preprocess.rs:40-43,73-81` (`assert!` + opaque reshape on misaligned input)
- **Finding:** These `pub` functions panic on invalid input instead of returning the crates' own typed errors; the registered trait paths pre-validate, but direct callers (InstantID/PuLID consume the face APIs; future conditioning paths hit the lens assert) get process aborts.
- **Impact:** A malformed buffer from a future direct caller aborts the worker (or poisons a mutex — F-031) instead of a catchable error.
- **Suggested fix:** Convert to `bail!`/`Error::Msg` at the pub boundaries; keep asserts on `pub(crate)` internals.
- **Confidence:** High

#### [F-042] Degenerate aspect ratios silently produce an empty face-detector blob
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `candle-gen-face/src/face.rs:128-136`
- **Finding:** `new_w = (640 / im_ratio) as usize` truncates to 0 beyond ~640:1 aspect; the blob stays all-padding, detection returns an empty list, and `det_scale == 0` maps coordinates through `1/0`.
- **Impact:** Degenerate inputs get a silent "no face" instead of a diagnosable rejection in the identity providers.
- **Suggested fix:** Reject images whose computed `new_w`/`new_h` round to 0, mirroring the existing zero-dimension guard.
- **Confidence:** Medium

#### [F-043] Wan 14B discards its `MergeReport`, and LTX silently ignores `req.steps`
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `candle-gen-wan/src/wan14b.rs:183` (vs the "surfaced, never silently dropped" contract in `adapters.rs:58-64`); `candle-gen-ltx/src/lib.rs:249-311,357-378`
- **Finding:** `build_expert` drops the adapter merge report, so partial LoRA matches vanish (only the zero-match case errors). Separately, LTX's `render` never reads `req.steps` (fixed distilled σ schedule) and `validate` neither rejects nor documents a supplied override.
- **Impact:** A half-matching community LoRA "applies" with weaker effect and no diagnostic; a `steps: 30` LTX request runs 8 steps with no feedback.
- **Suggested fix:** Log/surface the report at the call site; reject or document non-default `steps` on the LTX descriptor.
- **Confidence:** High

#### [F-044] `MAX_AREA_14B` is documented as a cap but never enforced
- **Category:** dead-code
- **Severity:** Low
- **Location:** `candle-gen-wan/src/config.rs:215-222` (no consumer; `wan14b.rs:513-545` validates only per-edge/multiple-of-16)
- **Finding:** The constant and its docs claim a 704×1280 area cap "like the 5B", but nothing references it; a 1280×1280 request (2.2× the documented area) passes validation onto two resident 14B experts.
- **Impact:** Either a missing guard against far-over-envelope runs or an actively misleading constant.
- **Suggested fix:** Enforce the area cap in `validate`, or delete the constant and its claims.
- **Confidence:** High

#### [F-045] SenseNova hardcodes `timestep_shift = 3.0`, shadowing the parsed config field
- **Category:** dead-code
- **Severity:** Low
- **Location:** `candle-gen-sensenova/src/config.rs:113-136,160-183`; consumer `candle-gen-sensenova/src/lib.rs:71,187-199`
- **Finding:** `NeoChatConfig` parses `timestep_shift` (and a dozen other generation-math fields) that nothing reads; the pipeline uses `DEFAULT_TIMESTEP_SHIFT = 3.0`, so a checkpoint shipping a different shift silently renders with 3.0 unless the request overrides it.
- **Impact:** Footgun for future checkpoint variants; dead config surface implies configurability that doesn't exist.
- **Suggested fix:** Route `req.scheduler_shift.unwrap_or(cfg-or-product-default)` with a comment on why 3.0 wins, or mark the fields reference-only.
- **Confidence:** Medium

#### [F-046] SD3.5 CLIP-tokenizer parity tests silently no-op off one workstation
- **Category:** dead-code
- **Severity:** Low
- **Location:** `candle-gen-sd3/src/clip_tokenizer.rs:177-184,207-255,261-308`
- **Finding:** All three synthesis-parity tests key off hardcoded `D:\sd35\…` paths and early-return when absent, so the claimed "byte-for-byte equivalent, asserted in the crate tests" guarantee is vacuously green everywhere else.
- **Impact:** A `tokenizers`-crate bump could silently regress the sc-8500 synthesized tokenizer with zero automated coverage.
- **Suggested fix:** Vendor a tiny vocab/merges fixture, or gate via env var + `#[ignore]` so skips are visible.
- **Confidence:** High

#### [F-047] Legacy per-crate Euler/schedule remnants after the sampler unification
- **Category:** dead-code
- **Severity:** Low
- **Location:** `candle-gen-flux2/src/pipeline.rs:136-150` (+ discarded tuple at `lib.rs:253`, `edit_provider.rs:204`, `control_provider.rs:177`); `candle-gen-lens/src/schedule.rs:52-63`; `candle-gen-ltx/src/scheduler.rs:1-21`
- **Finding:** `euler_step`/`timesteps` in flux2 and lens are called only by their own unit tests since the epic-7114 unification; flux2's call sites also discard half of `schedule()`'s tuple while recomputing `compute_mu` twice; LTX's `scheduler.rs` module is fully orphaned.
- **Impact:** Misleading API surface (a "legacy loop" nothing runs) inviting fixes to the wrong seam.
- **Suggested fix:** Delete the legacy helpers (candle-gen's sampler tests cover the N1 equivalence); have `schedule()` return only sigmas.
- **Confidence:** High

#### [F-048] `flash-attn` remains a no-op feature alias forwarded by every crate
- **Category:** dead-code
- **Severity:** Low
- **Location:** `candle-gen/Cargo.toml:38-40` (forwarded by all provider crates)
- **Finding:** The feature enables no code anywhere in the workspace ("wired in a later slice" since the Phase-1 scaffold); every new provider crate cargo-cults another forwarding line.
- **Impact:** Consumers enabling `flash-attn` get silently identical binaries; ~26 lines of ritual per-crate plumbing.
- **Suggested fix:** Wire `candle-flash-attn` behind it (cite the story) or delete the feature until scheduled.
- **Confidence:** High

#### [F-049] Unused heavyweight dependencies declared (and denied by their own comments) in 3 crates
- **Category:** dead-code
- **Severity:** Low
- **Location:** `candle-gen-ideogram/Cargo.toml:17-19` (+ feature forwarding at 32-34); `candle-gen-qwen-image/Cargo.toml:14-19` (`candle-transformers` + `tokenizers`); `candle-gen-krea/Cargo.toml:18` (`candle-transformers`)
- **Finding:** All three crates declare `candle-transformers` (qwen-image also `tokenizers`) with zero usages — grep-confirmed — and in ideogram and qwen-image the *adjacent comment literally says the crate has no candle-transformers reference*. The `metal`/`cuda`/`flash-attn` features are dutifully forwarded to the unused dep.
- **Impact:** Three crates pay the compile time of a large crate for nothing; the contradictory comments actively mislead.
- **Suggested fix:** Delete the deps and their feature forwards; `cargo check` per-crate to confirm.
- **Confidence:** High

#### [F-050] Unused/duplicated config surface in Ideogram, SCAIL-2, and Boogu
- **Category:** dead-code
- **Severity:** Low
- **Location:** `candle-gen-ideogram/src/config.rs:25,38-43,55` (`IDEOGRAM_4_FP8_REPO`, `DEFAULT_*`, `RES_MIN/MAX/MULTIPLE`, `DEFAULT_MU` — unreferenced; `RES_MIN`/`RES_MAX` shadow independently hardcoded descriptor limits at `lib.rs:155-156`); `candle-gen-scail2/src/config.rs:55-56,85` (`max_trained_src_id`, documented as driving >N-reference interpolation that `model.rs:463-501` never implements); `candle-gen-boogu/src/config.rs:29-41,92-115,133-143` (`multiple_of`, `ffn_dim_multiplier`, `axes_lens`, `instruction_feat_dim`, `num_instruction_feat_layers`, `reduce_type` — parsed, never consumed; a snapshot with `reduce_type: "concat"` loads silently and runs mean-shaped weights)
- **Finding:** Dead constants and parsed-but-unread config fields, several of which duplicate live limits or promise unimplemented behavior.
- **Impact:** Drift hazard between constants and descriptors; misleading parity claims (>5 references, alternate reduce types) that fail deep instead of loudly.
- **Suggested fix:** Delete, or wire into the descriptor/validation as the single source of truth (`reduce_type != "mean"` → error); cite stories for the unimplemented paths.
- **Confidence:** High

#### [F-051] Adapter merge reports printed to stderr from library code (sd3, qwen-edit) while twins stay silent
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `candle-gen-sd3/src/pipeline.rs:286-289` and `candle-gen-qwen-image/src/edit.rs:174-177` (vs the silent twin at `candle-gen-z-image/src/pipeline.rs:286-295`)
- **Finding:** Two merge paths write the report straight to stderr via `eprintln!`; the z-image twin is silent, and everywhere else the report is discarded (see also F-043 for wan). No provider surfaces it consistently.
- **Impact:** Unstructured, uncapturable stderr noise in the worker; inconsistent observability across providers.
- **Suggested fix:** Return `MergeReport` to the caller consistently (krea's `merge_into_weights` already does); drop the prints.
- **Confidence:** High

#### [F-052] Grad-clipping and Prodigy issue hundreds of blocking GPU→CPU scalar syncs per step
- **Category:** efficiency
- **Severity:** Low
- **Location:** `candle-gen/src/train/optim.rs:48-71,343-377`
- **Finding:** `clip_grad_norm` does one `to_scalar` device sync per var; Prodigy's pass 1 does two per var per micro-step. With ~224–336 adapter vars (Krea/Z-Image surfaces), that is hundreds of stalls per optimizer update.
- **Impact:** Milliseconds of pure CPU-GPU stall per training step, growing with the target surface.
- **Suggested fix:** Accumulate the reductions on-device and read back once per pass.
- **Confidence:** High

#### [F-053] Streaming VAE decode accumulates frames via repeated `Tensor::cat` (O(T²) copy traffic)
- **Category:** efficiency
- **Severity:** Low
- **Location:** `candle-gen-wan/src/vae.rs:461-469`, `candle-gen-wan/src/vae16.rs:394-401,440-453`
- **Finding:** The per-frame streaming loops grow the output with `cat(&[acc, new])` each iteration, re-copying the accumulated full-resolution video per frame and briefly holding old+new copies.
- **Impact:** ~10× the output size in copy traffic for a 21-latent-frame decode; inflates the decode-stage VRAM peak the streaming design exists to bound.
- **Suggested fix:** Collect into a `Vec<Tensor>` and cat once (as svd's decode already does).
- **Confidence:** High

#### [F-054] IP-Adapter K/V projections of constant image tokens recomputed per block per step
- **Category:** efficiency
- **Severity:** Low
- **Location:** `candle-gen-flux/src/ip_adapter.rs:154-187`
- **Finding:** `double_block_residual` re-projects the fixed 4-token image-prompt tensor through `k_proj`/`v_proj` for each of 19 double blocks on every denoise step, though the tokens never change (documented at 128-129).
- **Impact:** 19 × steps redundant matmuls per render; trivially hoistable.
- **Suggested fix:** Precompute per-block K/V in `FluxIpInjector::new`.
- **Confidence:** High

#### [F-055] Fresh no-affine LayerNorms (device `ones` allocations) constructed inside hot loops
- **Category:** efficiency
- **Severity:** Low
- **Location:** `candle-gen-flux/src/control.rs:325-344` (per stream × 6 blocks × steps); `candle-gen-seedvr2/src/dit.rs:124-129` (`rms_plain` `Tensor::ones` per call, ~128+ per DiT step)
- **Finding:** Both paths build a fresh norm (including a device `ones` alloc + fill) on every invocation inside per-step loops; the equivalent AdaLN base norms in the same files are constructed once at load.
- **Impact:** Hundreds of needless small allocations/kernel launches per render/tile.
- **Suggested fix:** Store the norm/ones tensor on the block/transformer at load, or use the weight-free functional layer-norm helper the sibling crates use.
- **Confidence:** High

#### [F-056] Kolors runs `encoder_hid_proj` inside the per-step UNet forward on one of three paths
- **Category:** efficiency
- **Severity:** Low
- **Location:** `candle-gen-kolors/src/unet.rs:307-308` (vs the hoisted projection at `control.rs:263`, `ip_provider.rs:272`)
- **Finding:** The txt2img UNet re-projects the step-invariant ChatGLM3 context every denoise step; the control/IP providers project once up front.
- **Impact:** ~4 GFLOPs/step waste plus an asymmetric seam that makes the three lanes harder to diff-review.
- **Suggested fix:** Hoist the projection into `Pipeline::render`, matching the other two paths.
- **Confidence:** High

#### [F-057] `has_diff_patch_keys` reads whole adapter files despite claiming "reads only the header"
- **Category:** efficiency
- **Severity:** Low
- **Location:** `candle-gen-scail2/src/adapters.rs:317-330`
- **Finding:** The doc says header-only; the implementation is `std::fs::read(path)` on bundles that can be hundreds of MB, just to enumerate tensor names.
- **Impact:** Avoidable multi-hundred-MB read on the worker's routing path; the comment misstates the cost.
- **Suggested fix:** Read the 8-byte header length then only `8 + header_len` bytes, or fix the comment.
- **Confidence:** High

#### [F-058] SeedVR2 load path holds raw fp16 and cast copies of all weights simultaneously
- **Category:** efficiency
- **Severity:** Low
- **Location:** `candle-gen-seedvr2/src/pipeline.rs:126-141`, `weights.rs:51-57`
- **Finding:** `load` keeps `vae_raw`/`dit_raw` alive across `convert_*(..).cast(dt)`, so peak load VRAM is ~2× (bf16) to ~3× (f32) checkpoint size — on the order of an extra 14 GB transient for the 7B DiT.
- **Impact:** Load-time OOM headroom much tighter than steady-state on mid-range cards.
- **Suggested fix:** `drop()` the raw maps eagerly, or cast tensor-by-tensor.
- **Confidence:** Medium

#### [F-059] Duplicated small utilities across crates: `image_seed` (6 copies), seeded-noise/salt (4 sites), `to_image` (4), Lanczos reimpl, SDXL β constants (3), `repeat_kv` (12 files), rotate/sine-PE/mask helpers (sam3/seedvr2), depth `Weights` fork, `text_key_mask` twins
- **Category:** redundant
- **Severity:** Low
- **Location:** `image_seed`: `candle-gen-{chroma,flux,kolors,sd3,sdxl,z-image}/src/pipeline.rs`; noise/salt: `candle-gen-sdxl/src/{denoise.rs:63-67,edit_provider.rs:326-330,pipeline.rs:424-428,training.rs:245-250}` + `STEP_RNG_SALT` ×3; `to_image`: flux/flux2/chroma/lens; Lanczos: `candle-gen-instantid/src/resample.rs:14-30` (vs `gen_core::imageops`); β constants: `candle-gen-sdxl/src/{pipeline.rs:100-102,sampler.rs:40-42}` + `candle-gen-instantid/src/model.rs:77-79`; `repeat_kv`: 12 text-encoder/DiT files; sam3 internal twins: `detr.rs:667-707` vs `model.rs:268-276`/`tracker.rs:668-693`, `vision.rs:117-130` vs `tracker.rs:849-862`; depth `Weights`: `candle-gen-depth/src/common.rs:23-89` vs `candle-gen-sdxl/src/weights.rs:16-44`
- **Finding:** The sc-7792 consolidation (`candle_gen::cached` + `for_each_seed`/`image_seed`) is **not present on this tree** — it never merged — so the determinism-critical `image_seed` and CPU-seeded-noise conventions still exist in 6/4 copies, alongside a long tail of reimplemented helpers, several parity-critical (RoPE rotation, sine PE, β schedule).
- **Impact:** A single divergent copy of a determinism/parity convention silently breaks cross-provider reproducibility; each new crate copies the block again.
- **Suggested fix:** Land the sc-7792 branch (or re-cut it), then migrate; hoist `Weights`, the noise draw + salt, `to_image`, and the SDXL constants into `candle-gen`; use `gen_core::imageops` for Lanczos.
- **Confidence:** High

#### [F-060] `candle-gen-sdxl` has become the de-facto commons crate
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `candle-gen-sdxl/src/weights.rs` (imported by `candle-gen-pulid`, `candle-gen-flux`); `candle-gen-pulid/Cargo.toml:26`
- **Finding:** PuLID (a FLUX-family provider) depends on the entire ~12k-LOC SDXL crate solely for the 73-line backend-generic `Weights` map (its own Cargo.toml comment admits this); feature flags must be forwarded through the unrelated crate.
- **Impact:** Compile-time bloat, false coupling (SDXL churn rebuilds pulid/flux), and gravity pulling shared utilities into the wrong crate.
- **Suggested fix:** Hoist `Weights` into `candle-gen`; re-export from the SDXL crate for compatibility.
- **Confidence:** High

#### [F-061] SDXL trainer duplicates the exported UNet config; bespoke decode path lacks the tiling the registered path has
- **Category:** redundant
- **Severity:** Low
- **Location:** `candle-gen-sdxl/src/training.rs:88-116` (verbatim duplicate of `unet/controlnet.rs:40-64`); `candle-gen-sdxl/src/denoise.rs:150-168` vs `pipeline.rs:602-637`
- **Finding:** `training.rs::sdxl_unet_config()` is an exact duplicate of the exported one (risking train-vs-inference config drift after a candle re-pin). `denoise::decode_image` duplicates `Pipeline::decode`'s post-process but lacks the sc-4987 VAE tiling, so all bespoke providers decode 1024² monolithically while the registered path tiles.
- **Impact:** Config drift risk; divergent peak-VRAM behavior between the two SDXL lanes at identical resolutions.
- **Suggested fix:** Import the exported config; fold the post-process into one helper with a tiling mode.
- **Confidence:** High

#### [F-076] Qwen-Image txt2img defaults to 4 steps on a non-distilled 20B model
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `candle-gen-qwen-image/src/config.rs:9-11` (used at `lib.rs:190-193`)
- **Finding:** `DEFAULT_STEPS = 4`, with a comment admitting "production callers pass ~20–50 (Qwen-Image T2I is not distilled)". A valid request that omits `steps` gets a visually broken 4-step render of an undistilled base model.
- **Impact:** Any caller relying on engine defaults (a new worker lane, an example, a conformance profile) silently gets garbage-quality renders.
- **Suggested fix:** Default to a production step count (30/50, like `QwenEditRequest` and boogu) or require `steps` in `validate` — unless fork parity demands 4, in which case document that loudly.
- **Confidence:** High (behavior); Medium (that changing it is desired)

#### [F-077] Krea/Boogu prompt-length caps enforced only by opaque tensor errors
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `candle-gen-krea/src/pipeline.rs:44-46` + `text_encoder.rs:148-150`; `candle-gen-boogu/src/pipeline.rs:52-53` + `text_encoder.rs:67-70`
- **Finding:** Both crates tokenize via gen-core's `encode_ids` (documented: no padding, no truncation — caller owns length policy), but nothing enforces the `MAX_TEXT_TOKENS` (1024/1280) that sizes the RoPE table; an over-long prompt fails deep in `Rotary::text`'s `narrow` with a candle shape error mid-generate. Qwen-Image right-truncates and is safe.
- **Impact:** A pathologically long prompt yields "narrow invalid args" instead of "prompt exceeds N tokens" — hard to diagnose from worker logs.
- **Suggested fix:** Truncate to the cap or return a descriptive length error in the tokenizer wrappers.
- **Confidence:** High

## Informational

#### [F-078] Adapter files are read fully into memory with no size cap
- **Category:** security
- **Severity:** Info
- **Location:** `candle-gen-qwen-image/src/adapters.rs:90-98`, `candle-gen-krea/src/adapters.rs:105-113` (pattern shared by the other `read_adapter` copies — see F-018)
- **Finding:** `read_adapter` does `std::fs::read(path)` of a worker-supplied adapter path with no size cap before parsing, unlike the mmap'd base-weight paths. No injection/traversal risk — paths come from the trusted worker.
- **Impact:** A multi-GB "adapter" file is fully buffered before validation; worst case is memory pressure, not code execution.
- **Suggested fix:** Optional: mmap adapters like base weights, or sanity-cap the size before `fs::read` (fold into the F-018 shared skeleton).
- **Confidence:** High

#### [F-062] `unsafe` mmap hygiene is inconsistent (undocumented SAFETY at ~8 of ~30 sites)
- **Category:** security
- **Severity:** Info
- **Location:** `candle-gen-clip/src/lib.rs:172-175`; `candle-gen-sdxl/src/loaders.rs:44,91,123` + `training.rs:431,557-563`; others carry the house SAFETY comment
- **Finding:** All ~30 `unsafe { VarBuilder::from_mmaped_safetensors }` sites follow the standard candle read-only-weights pattern (no misuse found), but several lack the one-line SAFETY comment the house style uses, and the invariant is restated per site rather than owned by one audited helper (see F-019).
- **Impact:** None today; future edits could drop the reasoning silently.
- **Suggested fix:** Centralize via F-019's helper; meanwhile add the standard comment at the bare sites.
- **Confidence:** High

#### [F-063] Vendored sliced-attention path is unreachable and upstream-broken
- **Category:** dead-code
- **Severity:** Info
- **Location:** `candle-gen-sdxl/src/unet/attention.rs:157-183,232-243`
- **Finding:** Every config sets `sliced_attention_size: None`, and the branch (byte-identical to upstream candle @65ecb58) would error at runtime if enabled (`Tensor::stack` to 4-D then `dims3()`).
- **Impact:** None today; a future "enable attention slicing for VRAM" attempt hits an immediate shape error.
- **Suggested fix:** One-line comment noting unreachable + upstream-broken; optionally file upstream.
- **Confidence:** High

#### [F-064] Shard merges silently overwrite duplicate keys
- **Category:** bad-pattern
- **Severity:** Info
- **Location:** `candle-gen-sam3/src/common.rs:35-73` (pattern shared by other `from_dir` loaders)
- **Finding:** Later shards silently overwrite earlier tensors with the same key; a stray safetensors file in a snapshot dir can shadow real weights with no diagnostic.
- **Impact:** Wrong model behavior from a polluted checkpoint dir, with nothing pointing at the cause.
- **Suggested fix:** Warn/error when `HashMap::insert` returns `Some` during the merge.
- **Confidence:** High

#### [F-065] Stale scaffold-era docs on the core crate
- **Category:** readability
- **Severity:** Info
- **Location:** `candle-gen/src/lib.rs:16-17`, `candle-gen/Cargo.toml:5`
- **Finding:** The crate doc still says "Phase 1 is a scaffold … the real SDXL pipeline lands in a later slice"; the workspace now ships ~25 production providers, a training harness, and the unified sampler.
- **Impact:** First paragraph a new reader sees misstates maturity.
- **Suggested fix:** Drop the scaffold caveat.
- **Confidence:** High

#### [F-066] Doc/comment contradictions and stale claims (grab bag)
- **Category:** readability
- **Severity:** Info
- **Location:** `candle-gen-instantid/src/kps.rs:10-11` (cites a nonexistent `tests/instantid_kps.rs` parity test — the evidence lives in mlx-gen); `candle-gen-ltx/src/config.rs:6-8` + `lib.rs:394-395` (audio described as deferred though shipped); `candle-gen-ltx/src/vae.rs:472-476` ("placeholder constants" comment on CUDA-calibrated sc-7148 anchors); `candle-gen-lens/src/vae.rs:22-24` ("velocity" doc on what is the final denoised latents); `candle-gen-face/src/lib.rs:114-124` (`image_dims` doc describes a bbox); `candle-gen-face/src/bisenet.rs:36-39` (hand-rolled sigmoid with a false "avoids a candle_nn dep" comment — the crate depends on candle-nn and its sibling module imports the real one); `candle-gen-sd3/src/conditioning.rs:110-127` (`zeroed_outputs` doc contradicts the actual CFG path; test-only consumer); `candle-gen-qwen-image/src/control_fun.rs:6` vs `18-20` (InstantX lane called "retired" then "kept intact" 13 lines apart); `candle-gen-krea/src/adapters.rs:510-511` (doc links `attention_surface_keys`, renamed to `merge_surface_keys` in sc-8776); `candle-gen-qwen-image/src/vl_tokenizer.rs:74-81` (undocumented `as u8` truncation vs PIL's round-to-nearest in a module claiming PIL-exactness)
- **Finding:** Each comment contradicts adjacent code or cites evidence that isn't there.
- **Impact:** Misleads maintainers at parity-sensitive boundaries.
- **Suggested fix:** Fix the ten doc sites; use `candle_nn::ops::sigmoid` in bisenet; round (or document truncation) in `vl_tokenizer`.
- **Confidence:** High

#### [F-067] Naming/structure nits that read as bugs
- **Category:** readability
- **Severity:** Info
- **Location:** `candle-gen-flux2/src/text_encoder.rs:244-253` (`Qwen3TextEncoder` also loads/runs the Mistral dev tower); `candle-gen-sensenova/src/t2i.rs:422-431` (tensor named `cond` fed to the *uncond* pass — correct but uncommented); `candle-gen-wan/src/vace.rs:190-206,294-296,450-452` (width token count computed as `wl / ph`, correct only because the patch is square); `candle-gen-z-image/src/dit.rs:108-111,156-159` (constructor sizes K/V by `n_kv_heads`, forward reshapes by `n_heads` — GQA half-support); `candle-gen-z-image/src/control.rs:498` vs `573-577` (`steps == 0` → 1 in Turbo, 50 in base, undocumented); `candle-gen-sam3/src/video.rs:410-427` (new-detection predicate computed twice); `candle-gen/src/train/flow_match.rs:536-561` (`micro` always equals `step`)
- **Finding:** Each site is functionally correct today but requires the reader to prove it; several are latent traps for variant ports (asymmetric patch, GQA).
- **Impact:** "Is this a bug?" tax on parity-critical loops.
- **Suggested fix:** Renames, one-line comments, `debug_assert_eq!(ph, pw)`, and unified `steps == 0` semantics.
- **Confidence:** High

#### [F-068] Example-target naming convention breaches in ideogram
- **Category:** readability
- **Severity:** Info
- **Location:** `candle-gen-ideogram/examples/{render_edit.rs,convert_fp8.rs,ideogram-render.rs}`
- **Finding:** Unprefixed example names (`render_edit`, `convert_fp8`) breach the `<crate>-<verb>` convention that exists because example binaries share one output path and collide (LNK1104) in the CUDA gate.
- **Impact:** Latent link collision when another crate adds a same-named example.
- **Suggested fix:** Rename to `ideogram-edit` / `ideogram-convert-fp8`.
- **Confidence:** High

#### [F-069] Test-harness helper duplication (PPM/cosine/env/gpu_peak)
- **Category:** redundant
- **Severity:** Info
- **Location:** `read_ppm`/`write_ppm`/`env_path`/`cosine` copies across ~16 `#[cfg(test)]` validation modules (sdxl ×2, instantid ×2, pulid, kolors ×2, qwen-image ×4, z-image ×3, flux); `tests/common/gpu_peak.rs` byte-duplicated between wan and ltx
- **Finding:** Test-only duplication with observable drift (two PPM header tokenizers — one comment-tolerant, one not; f32 vs f64 cosine accumulate). All modules are properly `#[cfg(test)]`-gated (verified), so no production impact.
- **Impact:** A comment-bearing PPM passes some harnesses and fails others; methodology fixes must be mirrored by hand.
- **Suggested fix:** Shared test-support module (testkit or a `candle-gen` dev feature).
- **Confidence:** High

#### [F-070] Debug scaffolding retained from closed investigations
- **Category:** dead-code
- **Severity:** Info
- **Location:** `candle-gen-flux2/src/lib.rs:232,266-315,326-354` (FLUX2_DEBUG-gated `dbg_stats` probes in the render path); `candle-gen-flux2/examples/{flux2-qmm-probe.rs,flux2-te-probe.rs}`; `candle-gen-sam3/src/video.rs:405,680` (`Assoc.empty_trk` populated, never read); `candle-gen-sam3/src/vision.rs:457-492` (the always-discarded 36² FPN level — see F-016); assorted `let _ =` silenced fields in wan/ltx/svd (`transformer.rs:59,109`, `gemma.rs:52,260`, `svd/vae.rs:443-445`); `candle-gen-qwen-image/src/edit.rs:214,408` (`te_cfg` kept alive only by `let _ = &self.te_cfg`); `candle-gen-boogu/src/{tokenizer.rs:102-116,text_encoder.rs:257-272,transformer/rope.rs:89-95}` (single-reference wrappers unused in-crate after the sc-7645 multi-ref generalization — verify no worker/parity consumer before removing)
- **Finding:** Env-gated probes and silenced fields left from closed stories (sc-7457 et al.); runtime cost is negligible but the idiom is inconsistent with the rest of the workspace.
- **Impact:** Reader noise; `empty_trk` in particular may be a silent behavioral gap vs the reference association logic rather than dead code.
- **Suggested fix:** Remove (or promote `dbg_stats` to a documented shared diagnostic); confirm `empty_trk` against the reference `_associate_det_trk` and either wire or delete it.
- **Confidence:** Medium

#### [F-071] Real-weight CLIP tests resolve the HF cache via `$HOME` on a Windows-primary repo
- **Category:** bad-pattern
- **Severity:** Info
- **Location:** `candle-gen-clip/src/lib.rs:471-473,521-523`
- **Finding:** Both `#[ignore]` tests do `std::env::var("HOME").expect(..)` and assume `~/.cache/huggingface`; on the primary dev box `HOME` is often unset and the cache lives at `D:\.cache\huggingface`.
- **Impact:** The documented validation lane fails out-of-the-box on the machine it's meant for.
- **Suggested fix:** Honor `HF_HOME` with a `USERPROFILE`/`HOME` fallback.
- **Confidence:** High

## Themes and systemic observations

1. **Hard-won fixes don't propagate across sibling crates.** The three High findings and several Mediums are all instances of one crate learning a lesson its twins never received: flux2's i32 attention budget (F-003), lens's dequant-on-forward quant seam vs flux2's `fast_mmq` path (F-025), sd3's `norm_out` AdaLN fix vs the unfixed final-block twin in the same file (F-001), sam3's pre-transposed Linear vs seedvr2's per-forward transpose (F-017). The porting discipline is per-crate; the workspace needs a small shared kernel (`candle_gen::mmdit`-style helpers: budgeted attention, QLinear, RoPE builders, norm/modulate) so a fix becomes a family-wide guarantee.

2. **Copy-paste pipeline scaffolding is the dominant debt, and it has already produced behavioral bugs.** ~25 copies of the component loader (F-019), 8 of the adapter-merge skeleton (F-018), triplicated provider plumbing in kolors/z-image/flux1/flux2 (F-021–F-024), quadruplicated wan encode (F-020). The Kolors scheduler-routing bug (F-004) is direct drift damage; the flux2 control error-path regression (in F-019) is another. Much of this was *planned* to be consolidated — sc-7792's shared `cached`/`image_seed` helpers exist but never merged to this tree (F-059) — so the first step is landing/refreshing that branch, then extending it.

3. **The bespoke (worker-driven) providers form a second, less-guarded lane beside the registered `Generator` paths.** Registered paths get `validate` (steps/size/prompt floors), VAE tiling, cached tokenizers, capability honesty; the bespoke lanes variously lack empty-prompt guards (F-007), steps floors (F-032), tiling (F-061), advertised-capability enforcement (F-004, F-005), and determinism discipline (F-008). A shared "provider entry contract" checklist (or moving the floors into shared helpers) would mechanize the house value the crates' own docs call the "false-capability trap".

4. **Hot loops carry avoidable invariant recomputation, inconsistently across siblings.** Tokenizer re-parsing per request (F-011), per-step RoPE/mask/conditioning rebuilds (F-012), ungated CFG branches (F-013), per-frame constant rebuilds and readbacks in the video crates (F-014–F-016). In each case at least one sibling crate (chroma, wan, kolors, lens) already demonstrates the correct hoisted/cached pattern — this is convergence work, not research.

5. **Error discipline is strong at API boundaries, softer inside.** Typed errors, loud capability rejection, and catchable budget errors are the norm; the residue is panics on untrusted file input (F-009), poisoned-mutex `expect`s (F-031), `assert!` on pub preprocessing (F-041), and silently discarded reports/visitor errors (F-035, F-043). All are localized and cheap to fix.

6. **Migration waves leave archaeology.** Each big unification (AV fold-in, curated samplers, sampler policies) stranded a layer of dead code and stale docs (F-027, F-047, F-063, F-065, F-066, F-070). A short "sweep the strandline" pass at the end of each epic — the repo already has the story discipline for it — would keep the reading surface honest.

## Coverage notes

- **Reviewed:** all 26 workspace crates' `src/`, `Cargo.toml`s, examples, and test files (env-gated GPU harnesses verified `#[ignore]`/`#[cfg(test)]`-gated and skimmed rather than line-by-line-read in some crates); `scripts/check-cuda.ps1`, `scripts/package-cuda.ps1`, `scripts/check-gen-core-skew.sh`; `.github/workflows/ci.yml`; root `Cargo.toml` and `rust-toolchain.toml`. Review was performed by eight parallel deep passes (one per crate group) plus a workspace-wide cross-cutting duplication/`unsafe`/TODO scan; the two High-severity SD3.5 findings were independently re-verified against the source during synthesis.
- **Excluded:** `vendor/candle-kernels` (a documented, deliberate fork of the pinned upstream rev differing only in three `-gencode` lines — provenance checked, contents not re-reviewed); `target/` build output; binary test fixtures (`candle-gen-sdxl/tests/fixtures/`, `candle-gen-seedvr2/data/neg_embed.safetensors`).
- **Not performed:** a dependency CVE audit of `Cargo.lock` (no `cargo audit`/`cargo deny` run — the manifests pin candle to a git rev and keep a small direct-dep surface, but a lockfile advisory scan is worth adding to CI); re-reproduction of the F-003 attention overflow on hardware; A/B renders quantifying the visual magnitude of F-001/F-002 (both are code-verified; magnitude needs a diffusers component-parity run); bodies of ~15 `#[ignore]`d real-weight parity tests beyond structure/gating checks.
- **Known-deliberate patterns excluded by design** (documented decisions, not findings): per-crate trainer `compute_loss_grads` duplication (sc-7787), dequant-on-forward quant strategy where applied (sc-7702), composable-op training code (fused ops have no backward), CPU-seeded noise (sc-3673 determinism contract), the conv3d-via-taps twins (different padding semantics), the gen-core SHA-pin lockstep protocol, and the wan 512-token pad.
