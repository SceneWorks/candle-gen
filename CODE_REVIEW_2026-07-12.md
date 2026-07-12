# Full Codebase Review — candle-gen — 2026-07-12

## Executive summary

- **Repository at a glance:** Rust workspace, 29 crates (1 shared-commons core + 28 model-family provider crates), ~510 source files, ~182k LOC. Rust-native diffusion/vision model inference on the candle ML framework; the Windows/CUDA sibling of mlx-gen, sharing the backend-neutral `gen_core` contract (SHA-pinned in lockstep with the SceneWorks worker).
- **Coverage:** All 29 workspace crates (src/, examples/, tests/, manifests), `scripts/`, `.github/workflows/ci.yml`, root workspace config, and `vendor/candle-kernels` (provenance re-verified by full diff against the pinned upstream rev). Eleven parallel deep-review passes plus a workspace-wide cross-cutting scan; the highest-severity findings were independently re-verified against source during synthesis. See Coverage notes.
- **Prior review:** This is the second full review; the first (`CODE_REVIEW_2026-07-01.md`, 78 findings) triggered a large story-tagged remediation wave (roughly sc-8981–sc-9570). **71 of 78 prior findings are fully fixed, 7 are partial, 0 are untouched** — including both prior High-severity SD3.5 parity bugs and the i32 attention guard. See "Prior-review remediation status" below.
- **Headline:** The consolidation the last review asked for genuinely landed — one audited loader, one quant seam, one adapter-merge skeleton, shared seeds/tiling/testkit — and error discipline improved measurably. The top risks today are (1) **two blocking defects in the never-reviewed Bernini full pipeline** (host-device and dtype contract violations that fail every CUDA `bernini` generate and every conditioned request, before its pending GPU validation), (2) **five remaining unguarded-attention sites that silently corrupt at advertised request sizes** (stock SDXL UNet, FLUX.1 VAE, the boogu/krea Qwen3-VL vision tower, krea's grounded TE, sensenova prefill) — the same i32-overflow class the last review flagged, re-found in paths the sweep missed and in new code, and (3) **a silent training-quality bug in the Wan MoE trainer** that locks each expert to a disjoint half of any even-sized dataset. A systemic pattern also re-emerged: the sc-8992 RoPE-cache wave re-introduced the poisoned-mutex class at ~12 new sites, and new crates re-grow previously-fixed bug classes (steps==0 floors, per-forward RoPE rebuilds, silent config swallows).
- **Counts (new findings):** Critical: 0 | High: 4 | Medium: 21 | Low: 47 | Info: 13 (85 findings, numbered F-079 onward to avoid colliding with the 2026-07-01 report's F-001–F-078, which code comments already cite).

## Prior-review remediation status (2026-07-01 → 2026-07-12)

**Fully fixed (71):** F-001 (sc-8981, AdaLN order pinned by test), F-002 (sc-8982, first-EOS pooling), F-003 as scoped (sc-8983/9116/9570 — shared `sdpa_budgeted_bhsd/flat` in `candle-gen/src/attention.rs`, adopted by chroma/lens/flux-IP/qwen/sdxl-vendored/sd3/ltx/seedvr2-VAE; *but the class re-appears at five unswept sites — see F-081*), F-004–F-011, F-013–F-017, F-019–F-029, F-031 as scoped (sc-9015 `lock_recover`/`cached`; *class re-introduced in new code — F-103*), F-032 as scoped (*missed lanes — F-102*), F-033, F-036–F-048, F-050–F-058, F-060, F-061, F-063–F-078. Nearly every fix carries a story tag and a pinned regression test.

**Partial (7):**

| Prior | Status | What remains |
|---|---|---|
| F-018 | PARTIAL | Merge core hoisted to `candle-gen/src/train/merge.rs`; but per-crate `adapters.rs` count grew 8→10 (+anima, +ideogram; ~8,400 LOC), `strip_peft_prefix`/`PEFT_PREFIXES` still copied in krea/lens/sd3/z-image, 8 crates keep their own `merge_adapters` shell. |
| F-030 | PARTIAL | Production PATH-hijack closed (`candle-gen/src/gpu.rs` trusted resolver); testkit still spawns bare `nvidia-smi` (F-113), and `nvidia_smi_min_total_gib()` re-spawns the subprocess on every budgeted decode (only the path is cached). |
| F-034 | PARTIAL | Shared driver passes the actual pending micro-count (sc-9018); the wan MoE trainer's tail flush still passes nominal `accum` (`candle-gen-wan/src/training.rs:640-644` → wrapper 670-683), under-weighting each expert's final update when `steps/2 % accum != 0` or on mid-run cancel. |
| F-035 | PARTIAL | Fixed in the shared driver; the wan trainer's preview freeze/thaw copies still discard visitor results with `let _ =` (`training.rs:588-593,614-619`) — benign today (infallible closures), same pattern. |
| F-049 | PARTIAL | The three cited crates are clean; the identical unused-dep pattern is now live in flux2/scail2/wan/z-image/svd/qwen-image/pid (F-104). |
| F-059 | PARTIAL | Determinism-critical utilities centralized (`seed.rs`, `weights.rs`); cosmetic duplication is *growing*: `to_image` 4→12 copies, `repeat_kv` 12→16 files, InstantID Lanczos and the SDXL β constants (7 files) remain. |
| F-062 | PARTIAL | mmap `unsafe` owned by one audited file; ~57 SAFETY tags for ~63 unsafe sites — the one untagged file is `candle-gen/src/quant/cublaslt.rs` (8 unsafe uses incl. `unsafe impl Send/Sync` at :160-161, prose-only rationale). |

**Vendored-fork provenance:** `vendor/candle-kernels` was diffed in full against the pinned upstream rev (`c1e6756a89`). The delta is exactly the two documented changes (sc-7544 gencode block in `build.rs`; sc-9601 i32-cast block in `src/cast.cu`) — the fork is not drifting, though its docs contradict themselves about the delta count (part of F-148).

## Critical findings

None found. No exploitable security vulnerability, data-loss risk, or defect blocking currently-shipped production paths was identified.

## High findings

#### [F-079] Move the full-Bernini planner's host-built tensors to the compute device
- **Category:** bad-pattern
- **Severity:** High
- **Location:** `candle-gen-bernini/src/process.rs:211,239`; `candle-gen-bernini/src/assembly.rs:37-41`; consumed at `candle-gen-bernini/src/bernini.rs:428-448` and `candle-gen-bernini/src/mar.rs:266-268`
- **Finding:** `mrope_position_ids` and `build_attention_mask_4d` hardcode `Device::Cpu`, and `format_mllm_inputs_embeds` builds `input_ids` on CPU; `build_stream` stores them in `StreamState` with no `to_device` anywhere in the crate (the mask gets a `to_dtype` at `bernini.rs:448` but never a device move — **verified during synthesis**). On CUDA, the backbone's `embed_tokens.index_select` (CPU indexes vs CUDA table), `mrope_cos_sin` (CPU cos/sin vs CUDA q/k), and `scores.broadcast_add(mask)` all hard-error with candle's `DeviceMismatchBinaryOp`.
- **Impact:** Every `generate` on the registered `bernini` engine fails at the first planner forward on the repo's target platform (Windows/CUDA) — including plain t2i. All bernini tests are CPU-only and the GPU smoke covers only `bernini_renderer`, so nothing catches this before the pending sc-11003 GPU validation.
- **Suggested fix:** In `build_stream`, move `pos`/`mask` to the backbone device alongside the existing dtype cast, and give `format_mllm_inputs_embeds` the backbone device for its `input_ids` tensor (or accept a `&Device` parameter).
- **Confidence:** High

#### [F-080] Fix the Bernini vision tower's bf16-weights / f32-activations dtype contract
- **Category:** bad-pattern
- **Severity:** High
- **Location:** `candle-gen-bernini/src/vision.rs:450-492` (no input cast; f32 `plan.rope` → `cos`/`sin`; f32 `additive_mask`); weights loaded bf16 at `candle-gen-bernini/src/bernini.rs:87,201-204`; f32 pixels from `candle-gen-bernini/src/vit_preprocess.rs:197-216,223-240`
- **Finding:** `BerniniPlanner::load` builds `VisionTower` from a `PLANNER_DTYPE = BF16` VarBuilder, but its inputs are f32 (`normalized_frame`/`pack_patches`), its RoPE tables are f32, and its block masks are f32 — with zero `to_dtype` calls anywhere in `vision.rs` (**verified during synthesis**). Candle matmul/binary ops hard-error on mixed dtypes, so `patch_embed.forward(f32 pixels)` against bf16 weights fails immediately. The parity test loads the fixture as F32, so it never exercises the production bf16 path.
- **Impact:** Every conditioned full-`bernini` request (Reference / MultiReference / VideoClip — all advertised in the descriptor at `bernini.rs:514-517`) errors at the first ViT encode, on CPU and CUDA alike.
- **Suggested fix:** In `VisionTower::forward`, cast `pixel_values` to the tower dtype and cast `cos`/`sin`/masks to the activation dtype (mirror `Qwen25VlText::mrope_cos_sin`'s `to_dtype(dtype)` and `bernini.rs:448`'s mask cast); add a bf16 leg to the parity test.
- **Confidence:** High

#### [F-081] Five attention sites remain unguarded against the i32 scores overflow at advertised request sizes
- **Category:** bad-pattern
- **Severity:** High
- **Location:** (a) stock dense-path SDXL UNet: `candle-gen-sdxl/src/pipeline.rs:552-583` (Stock arm, `build_unet_with_adapters` :665-678) with `max_size: 2048` at `candle-gen-sdxl/src/lib.rs:414-416`; (b) FLUX.1 VAE mid-block: `candle-gen-flux/src/pipeline.rs:294-296,342,1093-1106` (stock `AutoEncoder` + packed `AutoEncoderKL` decode) with `max_size: 2048` at `candle-gen-flux/src/lib.rs:309-311`, plus the control-image encode at `candle-gen-flux/src/control_provider.rs:165-179`; (c) boogu Qwen3-VL vision tower (used by boogu edit **and** the new `krea_2_edit`): `candle-gen-boogu/src/vision/mod.rs:119-145` with `MAX_PIXELS = 16_777_216` at `candle-gen-boogu/src/vision/preprocess.rs:29-31`; (d) krea image-grounded TE: `candle-gen-krea/src/text_encoder.rs:243-271` with `MAX_EDIT_TOKENS = 8192` at `candle-gen-krea/src/pipeline.rs:79`; (e) sensenova VQA/interleave prefill: `candle-gen-sensenova/src/qwen3.rs:509-513` with uncapped `preprocess_image` at `candle-gen-sensenova/src/t2i.rs:458-482`
- **Finding:** The sc-9116 sweep (fixing F-003) guarded the listed DiT/VAE sites via the shared `candle_gen::sdpa_budgeted_*` helpers, but these five paths compute full unchunked scores tensors that exceed `i32::MAX` elements (~2.147B — candle's CUDA kernels index scores with i32; the tail silently corrupts) within their *advertised, `validate`-accepted* envelopes: stock SDXL at ≥ ~1664² (2048²: `2·10·16384² ≈ 5.4e9`); FLUX.1 VAE single-head spatial attention at 2048² (HW=65536 → 65536² ≈ 4.3e9) in upstream candle-transformers code the sweep couldn't touch; the boogu ViT at one ~3.0 MP reference (`16·11585² ≈ 2.15e9`, admission allows 16.7 MP — runs *before* any token cap in both edit pipelines); krea's grounded TE at exactly its inclusive 8192-token cap (`32·8192² = 2^31`); sensenova understanding prefills at ~8.2k tokens (one 4096×2048 source image), which the sweep triaged as "bounded seq" without accounting for image-bearing prefixes. (Sites (a)–(d) verified against source and the pinned candle rev during synthesis.)
- **Impact:** Valid requests silently produce garbage output on CUDA with no error — corrupted attention in a dense 2048² SDXL render, a garbage FLUX.1 decode *after* a correct denoise, subtly wrong edits from corrupted vision grounding on the brand-new krea/boogu edit engines, wrong VQA answers — the exact failure mode sc-5487 took days to debug once. The larger cases OOM instead (still without a typed rejection).
- **Suggested fix:** (a) route dense SDXL loads through the vendored UNet (already pinned bit-identical by `vendored_unet_matches_stock_forward`); (b) vendor the two FLUX.1 VAE mid-block attention forwards onto `sdpa_budgeted_flat` the way `flux2/src/vae.rs::MidAttention` does; (c) drop `sdpa_budgeted_bhsd` into `Block::attention` (the drop-in already used one module over) and/or lower `MAX_PIXELS`; (d) pass the causal mask into `sdpa_budgeted_bhsd` or make the cap exclusive and ≤ ~5,500; (e) budget `attention_cached` (the helper supports the `[1,1,q,k]` mask) or cap prefill length. Where a guard can't land promptly, lower the advertised `max_size` instead.
- **Confidence:** High for the boogu/krea sites (new code, thresholds verified); Medium for SDXL/FLUX.1/sensenova (mechanism and thresholds verified against source and the in-repo sc-9116 analysis; corruption not re-reproduced on hardware in this review)

#### [F-082] Fix the expert↔dataset parity lock in the Wan MoE trainer
- **Category:** bad-pattern
- **Severity:** High
- **Location:** `candle-gen-wan/src/training.rs:531-536`
- **Finding:** The alternating-expert train loop selects the expert by step parity (`ei = if step % 2 == 1 { 0 } else { 1 }`) and the dataset item by `(step - 1) % cache.len()`. For an **even** dataset size, `(step-1) % N` keeps the same parity as the expert index forever, so the high-noise expert trains exclusively on even-indexed items and the low-noise expert exclusively on odd-indexed items — each expert sees a disjoint half of the dataset for the entire run. **Verified during synthesis.**
- **Impact:** For the very common even-sized LoRA dataset (10–20 images — including this crate's own 2-item e2e fixtures), each expert's adapter silently trains on only half the user's images, degrading subject fidelity with no error or warning, on every such training job. (Rated High rather than Medium because the corruption is silent, affects the most common input shape, and there is no signal that anything went wrong.)
- **Suggested fix:** Decouple the item index from the expert parity — e.g. index by the per-expert micro counter (`cache[(ex.micro as usize) % cache.len()]`) or by `((step - 1) / 2) % cache.len()` — and check whether the mlx `WanMoeTrainer` twin shares the bug so the fix lands in lockstep.
- **Confidence:** High

## Medium findings

#### [F-083] Move the SDXL trainer preview off the native DDIM loop that sc-10826 removed for ghosting
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `candle-gen-sdxl/src/training.rs:255-302` (`render_one_preview`, native `DDIMScheduler` at 274-297)
- **Finding:** sc-10826 (commit 2e09eeb) deleted the native candle-transformers `DDIMScheduler` inference loop from `pipeline.rs` because it "rendered a ghosted, translucent double-exposure (guidance-invariant)" — but only in `lib.rs`/`pipeline.rs`. The sc-8650 trainer preview still drives the identical native loop with the same default config.
- **Impact:** SDXL LoRA/LoKr training previews render through the known-ghosting solver, so a healthy adapter looks broken mid-run — misleading exactly the progress feedback previews exist to provide.
- **Suggested fix:** Rebuild `render_one_preview` on the curated path (`DiscreteModelSampling::sdxl` + `candle_gen::run_curated_sampler(Some("ddim"), …)`), mirroring `Pipeline::denoise_curated`.
- **Confidence:** Medium (identical component/config as the removed loop; the ghost was diagnosed behaviorally, not root-caused)

#### [F-084] `load_instantid_unet_with_adapters` skips the packed-tier fork its sibling loaders take
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `candle-gen-sdxl/src/loaders.rs:77-92` (line 83 hardcodes the dense `.fp16` filename)
- **Finding:** sc-10813 packed-detected `load_instantid_unet` (via `instantid_unet_file`, :61-66) and sc-9528 gave the registered txt2img lane a packed+adapter fold (`pipeline.rs:638-655`), but the adapter variant of the InstantID/edit/IP loader still resolves only `unet/diffusion_pytorch_model.fp16.safetensors`. On a packed q4/q8 tier with adapters it fails with a misleading "snapshot is missing …fp16.safetensors" error even though the tier is present and the same combination works on the registered lane.
- **Impact:** InstantID-lane LoRA jobs against a packed tier hard-fail with a wrong diagnosis — classic sibling-entry-point copy drift.
- **Suggested fix:** Fork on `detect_packed_unet` and reuse `packed_adapters::fold_adapters_into_packed_map`; or return an explicit "adapters on a packed tier are not wired on this lane" error.
- **Confidence:** High

#### [F-085] Guard empty-string prompts on the qwen txt2img negative branch and the control_fun positive branch
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `candle-gen-qwen-image/src/lib.rs:291-295` (resident), `candle-gen-qwen-image/src/lib.rs:469-474` (sequential), `candle-gen-qwen-image/src/control_fun.rs:193`
- **Finding:** gen-core's `TextTokenizer::tokenize("")` short-circuits to empty ids **before** the chat template runs (the sc-8646 class; verified at the pinned rev). txt2img uses `req.negative_prompt.as_deref().unwrap_or(NEGATIVE_FALLBACK)`, so `Some("")` reaches `tokenize("")` unguarded; `control_fun`'s positive prompt has no empty guard at all. Zero-length ids then hit `QwenTextEncoder::prompt_embeds`' `hidden.narrow(1, 34, s - 34)` with `s = 0` — a usize-underflow panic in debug, an opaque `narrow` error in release. The in-crate siblings (`edit.rs:405-411`, `control_fun.rs:195-199` negative-side) guard exactly this.
- **Impact:** A cleared UI text field serialized as `""` fails the render with an undiagnosable error on a path the descriptor advertises (`supports_negative_prompt: true`); the same request works on the edit/control siblings.
- **Suggested fix:** Apply the sibling `trim().is_empty() → NEGATIVE_FALLBACK` guard in `render`/`render_sequential`; add an empty-prompt error at the top of `QwenFunControl::generate`.
- **Confidence:** High

#### [F-086] Qwen packed additive install silently mis-scales an untagged LyCORIS LoKr declared as `AdapterKind::Lokr`
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `candle-gen-qwen-image/src/adapters.rs:466-476` (guard) vs `candle-gen-qwen-image/src/adapters.rs:271-279` (dense route)
- **Finding:** The dense path routes any untagged `lokr_*` file through `merge_lokr_thirdparty` (per-module LyCORIS scaling) *before* consulting `spec.kind`. The packed path's rejection guard is `!af.declares_lokr() && spec.kind != AdapterKind::Lokr && keys_contain_lokr(…)` — so the same untagged file with `spec.kind == Lokr` slips past into `resolve_lokr_file`, where `parse_rank_alpha(None, None)` yields `(1.0, 1.0)` and per-module `.alpha` tensors are dropped as skipped keys.
- **Impact:** The identical `(file, spec)` renders correctly scaled on a dense Edit tier and silently mis-scaled on the packed q4/q8 tier — precisely the outcome the guard's own comment says it exists to prevent.
- **Suggested fix:** Drop the `spec.kind != AdapterKind::Lokr` clause: reject whenever the file is undeclared yet carries `lokr_*` keys, matching the dense path's file-metadata-is-authoritative stance.
- **Confidence:** Medium

#### [F-087] Boogu edit's 1280-token cap makes advertised reference sizes unservable, with a misleading error
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `candle-gen-boogu/src/pipeline.rs:62` (`MAX_TEXT_TOKENS = 1280`) + `candle-gen-boogu/src/tokenizer.rs:89-98,133-160`; advertised surface `candle-gen-boogu/src/lib.rs:58,214-216,245-253` (max_size 2048, up to 5 references)
- **Finding:** The edit template embeds one `<|image_pad|>` per merged vision token (image px / 1024), and the whole rendered string is checked against the 1280-token RoPE cap — but the grounded path never uses that RoPE table (it builds fresh MRoPE tables sized to S, `text_encoder.rs:316-318`). A single ≥1152² reference emits ≥1296 pads and can never pass; five 512² references can't either — while `descriptor_edit` advertises `max_size 2048` and `MAX_EDIT_REFERENCES = 5`. The failure message blames the *prompt* for the reference images.
- **Impact:** Validated multi-reference or large-reference boogu edits deterministically fail at generate time (after the vision tower has already burned GPU time) with an error that misdirects diagnosis; the advertised 5-reference capability only works for ≤ ~448² thumbnails.
- **Suggested fix:** Give the boogu edit encode its own larger cap (mirroring krea's `MAX_EDIT_TOKENS`, sized against a budgeted TE attention per F-081), and name the reference-token count in the error.
- **Confidence:** High

#### [F-088] `krea_2_edit` silently drops `req.use_pid`
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `candle-gen-krea/src/pipeline.rs:395-455` (no `resolve_pid_decoder`; native decode at :453) vs `candle-gen-krea/src/pipeline.rs:242-247,323-328` and the boogu template `candle-gen-boogu/src/pipeline.rs:361-366`
- **Finding:** `render` and `render_base` resolve the PiD decode seam and *error* when `use_pid` is requested but not loaded; `render_edit` never consults `req.use_pid` and always decodes with the native VAE. The shared `build` still accepts `spec.pid` for the edit id (`lib.rs:364-422`), so an edit generator can be loaded with a PiD engine that is then unreachable.
- **Impact:** A worker request asking for the 4× PiD super-resolving decode on `krea_2_edit` validates, renders, and silently returns a native-resolution image — the descriptor-accepts/render-drops trap, drifted from both its txt2img siblings and the boogu path it was templated on.
- **Suggested fix:** Call `candle_gen_pid::resolve_pid_decoder(…)` in `render_edit` and branch the decode like `render_base`; or reject `spec.pid`/`use_pid` for the edit id until wired.
- **Confidence:** High

#### [F-089] The single-entry RoPE cache thrashes on every CFG step (krea Raw/Edit, boogu Base/Edit)
- **Category:** efficiency
- **Severity:** Medium
- **Location:** `candle-gen-krea/src/transformer/mod.rs:126-155` (+ table build `rope.rs:104-141`); `candle-gen-boogu/src/transformer/mod.rs:114-149` (+ `rope.rs:152-193`)
- **Finding:** The sc-8992 cache holds exactly one geometry keyed on `cap_len`. Under true CFG the cond and uncond contexts almost always have different token counts, so every step alternates two geometries and **misses on every forward** — each miss re-runs the host trig loop plus an H2D upload. CFG is the default for `krea_2_raw` (52 steps ⇒ 104 rebuilds), `krea_2_edit` (edit tables at 2048²/2-ref ≈ 6.5M trig evals + ~25 MB upload each), and boogu Base/Edit. Only the single-context Turbo paths benefit today.
- **Impact:** The hoist landed to fix F-012 is inert on the highest-step production paths; tens of ms of host compute + transfer re-injected per step, worst on the new edit path where tables are largest.
- **Suggested fix:** Make the cache hold two entries (or a small map keyed by the geometry tuple) so pos/neg tables coexist across steps.
- **Confidence:** High

#### [F-090] Propagate the sc-9028 over-area guard to the wan-vace and scail2 14B lanes
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `candle-gen-wan/src/model_vace.rs:274-315` (validate; `max_size: 1280` at 354-355), `candle-gen-scail2/src/pipeline.rs:87-88,232-241`
- **Finding:** F-044's fix added the `MAX_AREA_14B` (704×1280) cap to the A14B `validate` because far-over-envelope requests fail opaquely (OOM). The Wan2.1-VACE-14B provider (14B DiT + 96-ch control stream) and SCAIL-2 (14B DiT run **f32** ≈ 56 GiB, packed sequence >2× the plain token count) both advertise `max_size: 1280` with no area check — a 1280×1280×81-frame request validates and runs.
- **Impact:** The exact incident class sc-9028 closed for the A14B remains open on the two *heavier* sibling lanes: minutes of GPU time ending in an opaque CUDA OOM instead of a fast rejection.
- **Suggested fix:** Enforce an area cap in both `validate`s (reusing `MAX_AREA_14B` or a per-model envelope), mirroring `wan14b.rs:642-651`.
- **Confidence:** Medium (guard absence certain; exact OOM threshold per lane unmeasured)

#### [F-091] PiD budget guard and tiling do not reach the four bespoke `pid_decoder_for` copies
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `candle-gen-instantid/src/model.rs:302-317`, `candle-gen-pulid/src/pulid_flux.rs:230-244`, `candle-gen-z-image/src/control.rs:480`, `candle-gen-sdxl/src/ip_provider.rs:221` (vs the guarded seam at `candle-gen-pid/src/engine.rs:196-228`)
- **Finding:** `resolve_pid_decoder_at_sigma` gained the memory-budget guard (sc-9095) and the spatial-tiling plan (sc-10087) — but only the registered providers call it. The bespoke `pid_decoder_for` copies mint the decoder via `engine.decoder(…).with_cancel(…)` directly, with no `budget::guard` and no `with_tiling`, so their `use_pid` decodes always run the whole-image forward.
- **Impact:** A large InstantID/PuLID/z-image-control/sdxl-IP `use_pid` render (e.g. 1536² → 6144² at 4×, ~11 GiB estimated peak by pid's own model) on a smaller GPU reproduces exactly the CUDA sysmem-fallback silent hang sc-10087 was built to prevent — with no typed refusal, because the budget backstop is also skipped.
- **Suggested fix:** Add a field-parameterized variant of `resolve_pid_decoder_at_sigma` in `candle-gen-pid` (taking prompt/count/size/cancel/use_pid instead of `GenerationRequest`) and route all four bespoke copies through it — collapsing the copy-drift at the same time.
- **Confidence:** High (guard/tiling absence certain; incident frequency depends on GPU size and requested resolutions)

#### [F-092] SeedVR2 video upscale ignores cancellation and reports fake progress for its whole multi-chunk run
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `candle-gen-seedvr2/src/lib.rs:187-209`; `candle-gen-seedvr2/src/pipeline.rs:404-461,466-502`
- **Finding:** The video path checks `req.cancel` exactly once before the run and emits a single `Progress::Step { current: 1, total: 1 }`; `generate_video`'s chunk loop, the per-frame fallback, and `generate_video_tiled`'s loops take no cancel flag or progress callback. sam3's `propagate` gained exactly this contract in sc-8972, so the crate is internally inconsistent about the gen-core video cancel contract.
- **Impact:** A worker cancel during a long clip (minutes to hours) is not honored until the whole upscale completes, and the UI gets no real progress for the entire run.
- **Suggested fix:** Thread `&req.cancel` and a per-chunk progress callback into the three generate loops (mirror `Sam3VideoModel::propagate`), reporting `Progress::Step { current: chunk_idx, total: plan.len() }`.
- **Confidence:** High

#### [F-093] `Sam3VideoModel::propagate` leaks per-video session state across calls with no reset
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `candle-gen-sam3/src/video.rs:85-101,162-185`
- **Finding:** `propagate` mutates persistent session state (`obj_ids`, `banks`, `first_frame`, `unmatched_frames`, `keep_alive`, `overlap_pairs`, `removed`, `last_occluded`, `max_obj_id`) but never clears it; only `from_weights` initializes it, and no `reset()` exists. A second `propagate` on the same instance starts against banks/hotstart bookkeeping from the previous clip.
- **Impact:** Any caller that caches the model (attractive — construction loads ~445M params) and runs a second clip gets silently corrupted tracking, not an error.
- **Suggested fix:** Reset the session fields at the top of `propagate` (or add `pub fn reset_session(&mut self)`), keeping the loaded weights.
- **Confidence:** Medium (hazard certain; whether production reuses one instance depends on the worker)

#### [F-094] SeedVR2 per-frame color correction is a single-threaded host hot loop that can dominate video wall time
- **Category:** efficiency
- **Severity:** Medium
- **Location:** `candle-gen-seedvr2/src/color.rs:134-199` (called per output frame from `pipeline.rs:353-372`)
- **Finding:** `apply_color_correction` runs two 5-level dilated wavelet decompositions × 3 channels (≈270 multiply-adds/pixel), three full-image LAB conversions with `powf`, and 2–3 O(n log n) `hist_match` sorts — all single-threaded host f32 — per output frame, serialized against the GPU pipeline.
- **Impact:** For a 300-frame 4K upscale this adds tens of minutes of idle-GPU host time; the post-process, not the one-step DiT, becomes the bottleneck.
- **Suggested fix:** Rayon-parallelize per frame and per channel (independent computations, per-frame numerics unchanged), or move the wavelet blurs to device tensors.
- **Confidence:** Medium (cost math certain; exact share of wall time depends on GPU speed)

#### [F-095] Restore the renderer's request-validation guards in the full-Bernini `validate`
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `candle-gen-bernini/src/bernini.rs:584-602` (validate), `:624-632` (`steps.max(1)`), vs `candle-gen-bernini/src/pipeline.rs:467-504`
- **Finding:** Copy-drift between the two sibling entry points: `BerniniRenderer::validate` rejects `steps == Some(0)`, non-multiple-of-`SIZE_MULTIPLE_14B` sizes, and `> MAX_AREA_14B`; the full `Bernini::validate` checks none of these. `generate_impl` truncates `height/8` and drops rows, so a 328-px request dies with an opaque shape error at denoise step 1, and `steps: Some(0)` is silently promoted to 1.
- **Impact:** Invalid `bernini` requests fail deep with unhelpful tensor errors (or silently render 1 step) instead of the crafted messages the sibling engine already has.
- **Suggested fix:** Hoist a shared `validate_bernini_geometry(req)` used by both entry points.
- **Confidence:** High

#### [F-096] Wire `Mode::needs_conditioning` — conditioning-mode/source mismatches are silent
- **Category:** dead-code
- **Severity:** Medium
- **Location:** `candle-gen-bernini/src/config.rs:44-50` (only self-tests reference it); `candle-gen-bernini/src/pipeline.rs:232-272,344-347`
- **Finding:** `Mode::needs_conditioning` is documented (config.rs:13-14 promises "the pipeline rejects them with an actionable message rather than silently rendering text-only") but is never called outside its own unit test. An explicit `video_mode="v2v"/"rv2v"` with no conditioning silently renders a plain text-only result; conversely `"t2v"` with a `Reference` attached VAE-encodes the source and then silently drops it.
- **Impact:** The advertised conditioning surface can be silently ignored or silently absent — users get a text-only render labeled as v2v; the exact "validate accepts, render drops" class this repo polices.
- **Suggested fix:** After `resolve_mode`, error when `mode.needs_conditioning()` and no sources are present; reject/warn on the inverse. Mirror in `resolve_vit_mode` for the full pipeline.
- **Confidence:** High

#### [F-097] Cache (and serialize) the full-Bernini components across generates
- **Category:** efficiency
- **Severity:** Medium
- **Location:** `candle-gen-bernini/src/bernini.rs:536-541` (no cache), `:615-925` (per-request loads: VAE :655, planner :672, UMT5 :827-834, both 14B experts :875-890)
- **Finding:** Unlike `BerniniRenderer` (Mutex-cached `Components`, pipeline.rs:106-163) and every sibling provider, the full `Bernini` reloads the Qwen2.5-VL planner, UMT5-xxl (f32), both 14B experts, and the z16 VAE from disk on every `generate`, and nothing serializes concurrent calls — two callers would each mmap+upload ~50 GB of expert weights simultaneously.
- **Impact:** Minutes of avoidable disk→VRAM load per request, plus a concurrent-request OOM hazard the cached siblings structurally avoid.
- **Suggested fix:** Cache at minimum `{UMT5, high, low, vae, tokenizer}` behind `candle_gen::cached`; keep the planner load-use-drop inside the request (the staging is deliberate for peak VRAM).
- **Confidence:** High

#### [F-098] Hoist step-invariant RoPE tables and source patch-embeds out of the Bernini packed-forward hot loop
- **Category:** efficiency
- **Severity:** Medium
- **Location:** `candle-gen-bernini/src/forward.rs:66-118` (`embed_segment`/`velocity`); host trig loop at `candle-gen-wan/src/rope.rs:43-80`
- **Finding:** `PackedForward::velocity` rebuilds `WanRope::cos_sin` (single-threaded host f64 trig over `L·64` entries) plus `apply_source_id` for the target and every source on every call — and the guidance chains make 2–4 `velocity` calls per denoise step; conditioning sources are also re-patch-embedded per call. All of this depends only on geometry and source-id, not σ. At 704×1280×81f (L≈74k) a 40-step Rv2v render rebuilds ~160 tables host-side.
- **Impact:** Seconds-to-tens-of-seconds of avoidable per-render host compute and H2D transfer, plus GPU idle bubbles, multiplied by the 4-forward guidance chains — the F-012 class re-introduced in a new crate (wan itself hoists these).
- **Suggested fix:** Cache `(cos, sin)` per `(grid, source_id)` and per-source `(tokens, cos, sin)` in a per-render prepared struct built once before the loop.
- **Confidence:** High

#### [F-099] The Bernini tier converter materializes essentially the whole 168 GB package in RAM
- **Category:** efficiency
- **Severity:** Medium
- **Location:** `candle-gen-bernini/src/convert.rs:168-235` (`extract_components`), `:283-307` (`write_expert`)
- **Finding:** `extract_components` calls `candle_core::safetensors::load(shard, &Device::Cpu)` per shard (full materialization, not mmap) and accumulates all routed tensors — both F32 experts (~114 GB) plus the planner (~33 GB) — into `HashMap`s held simultaneously until each component is written. Peak RSS ≈ 150 GB before the first write.
- **Impact:** The offline (but needed — sc-11003) tier build will OOM on typical ≤128 GB workstations.
- **Suggested fix:** Route via header-only key scans, then write one component at a time using mmap'd loads (the shard→component mapping is prefix-pure).
- **Confidence:** Medium

#### [F-100] Remove the per-call `stream.synchronize()` from the int8 IGEMM hot path
- **Category:** efficiency
- **Severity:** Medium
- **Location:** `candle-gen/src/quant/cublaslt.rs:272` (inside `matmul_int8_staged`, :239-283)
- **Finding:** `matmul_int8_staged` ends with a full `self.stream.synchronize()` on every GEMM. It runs on candle's own device stream, so stream ordering already guarantees correctness for downstream candle ops, and the host-fold paths read back via `clone_dtoh` (which synchronizes itself). The fp8 twin (`matmul_fp8_staged`, :320-373) has no such sync.
- **Impact:** The ConvRot resident forward drains the entire pipeline once per int8 projection per denoise step (~224 projections × steps × CFG on the Krea-2 12B DiT), defeating async enqueue-ahead; it also biases the `convrot_w8a8_bench` int8 column vs the sync-free fp8 column.
- **Suggested fix:** Drop the synchronize (or move it into the two host-fold callers only); re-run the CUDA-gated `cublaslt_8bit_numerics` to confirm the stream-ordered guards suffice.
- **Confidence:** Medium (the sync's existence and cost are certain; removal needs the one-test confirmation)

#### [F-101] Refresh the README's Status and Layout sections — they describe the repo circa 7 model families ago
- **Category:** readability
- **Severity:** Medium
- **Location:** `README.md:10-142`
- **Finding:** The Status narrative stops at the 7th family (JoyCaption) and asserts as "deferred" many things that shipped long ago: Wan "TI2V/I2V, VACE, LoRA, quantization, and tiling are deferred" (all shipped), Qwen-Image "Edit / ControlNet / Lightning / LoRA / quantization are deferred" (shipped), FLUX.2 "edit variants … LoRA, and quantization are deferred" (shipped). The Layout tree lists 9 of the 29 workspace crates.
- **Impact:** The repo's front door (and the published core crate's `readme`) actively misstates current capability — features that are GPU-validated and worker-wired are described as absent.
- **Suggested fix:** Replace the per-family changelog-style Status block with a family/capability matrix, and regenerate the Layout tree from `[workspace] members`. The Packaging section is accurate — keep it.
- **Confidence:** High

#### [F-102] The steps==0 / request-floor sweep (F-032) missed five bespoke lanes
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `candle-gen-kolors/src/control.rs:305-337` + `candle-gen-kolors/src/ip_provider.rs:266-292` (curated branch feeds `req.steps` into gen-core's `schedule_sigmas`, which clamps `steps.max(1)` — a `steps: 0` request silently renders 1 step); `candle-gen-sdxl/src/edit_provider.rs:229-243` (explicit `steps: 0` with default strength 0.8 produces an empty schedule — the 80%-noised source is decoded and returned as the "edit"); `candle-gen-flux/src/ip_provider.rs:356-363` (validate checks only the prompt — no steps floor, no multiple-of-16 size check, unlike its three siblings); `candle-gen-pulid/src/pulid_flux.rs:269-320` (`get_schedule(0, …)` yields `[NaN]`, zero sampler steps, pure seeded noise decoded; no size floor either); `candle-gen-anima/src/lib.rs:254,289-307` (`anima_sigmas` clamps `steps.max(1)` — silent 1-step render)
- **Finding:** The sc-9016/F-032 remediation added `steps == 0` rejection to the bespoke sdxl-IP, InstantID, and SCAIL-2 entry points, but five other lanes — including one brand-new crate (anima) — still accept it, each degrading differently (silent 1-step render, noise decode, or garbled pseudo-edit).
- **Impact:** A misconfigured worker request burns GPU time and returns garbage labeled success instead of the fast typed error its sibling lanes give; per-lane behavior diverges on identical input.
- **Suggested fix:** Add the `reject_zero_steps`-style guard (plus size floors where absent) to all five lanes; longer-term, hoist a shared bespoke-provider request-floor helper so new lanes get it by construction.
- **Confidence:** High

#### [F-103] The poisoned-mutex class (F-031) re-appeared at ~12 new sites — most introduced by the sc-8992 RoPE-cache wave
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** RoPE/geometry caches: `candle-gen-lens/src/transformer.rs:540`, `candle-gen-flux2/src/transformer.rs:570,613`, `candle-gen-krea/src/transformer/mod.rs:133`, `candle-gen-boogu/src/transformer/mod.rs:126`, `candle-gen-scail2/src/model.rs:357`, `candle-gen-ltx/src/transformer.rs:589`, `candle-gen-sam3/src/tracker.rs:319-349`, `candle-gen-sam3/src/detr.rs:526`; component/encoder locks: `candle-gen-sd3/src/pipeline.rs:351` (`.expect` across the T5/CLIP encode), `candle-gen-flux/src/pipeline.rs:1066` (T5 `.expect` across a 24-layer forward), `candle-gen-chroma/src/pipeline.rs:236`, `candle-gen-z-image/src/lib.rs:166-178` + `candle-gen-z-image/src/base.rs:77-88` (lock held across `load_components`)
- **Finding:** These sites `lock().unwrap()`/`.expect()` mutexes on `Arc`-shared cached generators/models. A panic while a lock is held (CUDA OOM-turned-panic mid-forward or mid-load) poisons it, after which every subsequent render on the long-lived worker instance panics. The workspace standardized on the poison-tolerant `candle_gen::lock_recover`/`cached` (sc-9015) precisely to remove this failure mode — the sc-8992 cache wave and several pre-existing sites never adopted it (z-image's own `base.rs:96` uses `lock_recover` three lines away).
- **Impact:** One transient failure wedges a cached generator into a permanent panic loop until worker restart — the incident mode sc-9015 was raised to prevent, now re-seeded across 8 crates.
- **Suggested fix:** Replace with `candle_gen::lock_recover` (all protected states are overwrite-on-miss caches or reloadable components, safe to reuse after a panic). Consider a clippy-style grep in CI for `lock().unwrap()`/`lock().expect` under `candle-gen*/src`.
- **Confidence:** High (pattern verified at each site; triggering requires a panic while locked)

## Low findings

#### [F-104] Unused dependencies with stale justification comments across ten manifests
- **Category:** dead-code
- **Severity:** Low
- **Location:** `candle-gen-flux2/Cargo.toml:24,27,38-39` (`candle-transformers` + feature forwards, `tokenizers`, `rand_distr`); `candle-gen-flux/Cargo.toml:44` (`rand_distr`); `candle-gen-wan/Cargo.toml:16-24` (`tokenizers`, `rand_distr`, `safetensors`); `candle-gen-scail2/Cargo.toml:20-28` (`tokenizers`, `rand`, `rand_distr`, `safetensors`); `candle-gen-svd/Cargo.toml:14-19` (`rand_distr`); `candle-gen-qwen-image/Cargo.toml:20,25` (`rand_distr`, `safetensors`); `candle-gen-z-image/Cargo.toml:35,38,42` (`tokenizers`, `rand_distr`; `safetensors` is dev-only-used); `candle-gen-pid/Cargo.toml:20-21` (`image` with png/jpeg features)
- **Finding:** All grep-verified unreferenced (tokenization went through gen-core, noise through `candle_gen::seeded_normal_vec` (sc-9452), adapter metadata through the shared `read_adapter` (sc-8998)) — several with comments describing code paths that no longer exist. flux2's `candle-transformers` forces the large crate into every build purely via feature plumbing; pid's `image` taxes the ~9 crates that link it.
- **Impact:** The F-049 pattern recurring at ten new locations after the original three were fixed; needless compile surface and actively misleading manifests.
- **Suggested fix:** Delete the lines (moving z-image's `safetensors` to dev-deps); add a `cargo machete`/`cargo udeps` CI step to end the recurrence class.
- **Confidence:** High

#### [F-105] Shared dependency versions pinned per-crate instead of via `[workspace.dependencies]`
- **Category:** redundant
- **Severity:** Low
- **Location:** root `Cargo.toml:17-22` (policy statement) vs `candle-gen/Cargo.toml:22` (`safetensors`), ≥8 crates' direct `serde_json = "1"`, ~20 crates' `image = { version = "0.25", … }` with feature-list variance, `candle-gen-instantid/Cargo.toml:36` (direct `rand`)
- **Finding:** The root manifest declares itself the "single source of truth for the deps shared across the workspace", yet `safetensors`/`image`/`half` were never centralized and several crates re-declare `serde_json`/`rand` directly instead of `{ workspace = true }`.
- **Impact:** A major-version bump of `image`/`safetensors` is an N-file edit — exactly the pre-centralization state the root comment says it fixed. Bites at the next major bump.
- **Suggested fix:** Add `safetensors`, `image` (superset features), `half` to `[workspace.dependencies]`; convert direct declarations.
- **Confidence:** High

#### [F-106] A new wave of near-identical per-crate packed-load wrapper enums is accreting over the shared quant seam
- **Category:** redundant
- **Severity:** Low
- **Location:** local `QLinear { Dense, Packed }` wrappers: `candle-gen-boogu/src/quant.rs:31`, `candle-gen-chroma/src/quant.rs:42`, `candle-gen-ideogram/src/quant.rs:49`, `candle-gen-krea/src/quant.rs:42`, `candle-gen-ltx/src/quant.rs:72`; `QEmbedding { Dense, Packed }` wrappers: `candle-gen-flux/src/quant.rs:47-82`, `candle-gen-z-image/src/quant.rs:44`, `candle-gen-flux2/src/quant.rs:61-105`; plus a character-identical `q4_packed` test-fixture packer in flux/flux2 and four more copies inside `candle-gen` itself (`quant/adapt.rs:456-491`, `quant/mod.rs:886-921`, `quant/repack.rs:375-385`, `train/lora.rs:1016-1040`); ~2,900 LOC across 12 per-crate `quant.rs` files
- **Finding:** The F-025 fix centralized the load-bearing logic, but each packed-tier adoption re-wrote the same thin `Dense|Packed` enum, `.scales`-sibling detect, and `is_packed()` hook — five `QLinear` wrappers and three `QEmbedding` wrappers differing only in docs and default group size, plus seven copies of the MLX-Q4 test packer.
- **Impact:** Six types named `QLinear` again coexist (the "which identically-named type am I touching" tax F-025 called out); a detect-logic fix must be re-verified per copy.
- **Suggested fix:** Add a generic `candle_gen::quant::detect::{QLinear, QEmbedding}` parameterized by group-size source and a single testkit fixture packer; migrate the wrappers.
- **Confidence:** High

#### [F-107] Step-invariant RoPE/grids still rebuilt per forward in flux1, z-image (vendored), and anima
- **Category:** efficiency
- **Severity:** Low
- **Location:** `candle-gen-flux/src/ip_dit.rs:739-742` + `candle-gen-flux/src/control.rs:509-512` (the control lane builds the same `pe` **twice** per step) + `candle-gen-flux/src/packed_dit.rs:609`; `candle-gen-z-image/src/control.rs:303-353` + `candle-gen-z-image/src/packed_dit.rs:551-572` + `candle-gen-z-image/src/dit.rs:514-544` (host coordinate grids + three RoPE table sets per call, ×2 under base-mode CFG, and per training step); `candle-gen-anima/src/transformer.rs:477-485` (Cosmos RoPE host trig ≈1.2M entries ×2 at 1536² per forward, plus a geometry-only `Tensor::zeros` mask channel)
- **Finding:** The F-012 class at sites the sc-8992 wave didn't reach: tables depending only on geometry are rebuilt every DiT forward (flux1: every denoise step ×25; z-image base-CFG: 100 rebuilds per render; anima: ~480 rebuilds per count-8 batch). The proven in-repo fix pattern (`Flux2RopeCache` / qwen `RopeCache`) applies directly.
- **Impact:** Repeated host compute and H2D uploads inside denoise hot loops — pure overhead growing with resolution; doubled on the flux1 control lane.
- **Suggested fix:** Port the ids/geometry-keyed cache onto `IpFlux`/`PackedFluxDit` (threading one `pe` into both control and base), the vendored z-image DiTs, and `CosmosDiT`.
- **Confidence:** High

#### [F-108] RoPE/prepared-cond cache keys force device→host readbacks every forward (ltx, ideogram)
- **Category:** efficiency
- **Severity:** Low
- **Location:** `candle-gen-ltx/src/transformer.rs:503-598` (key = `flatten_all()?.to_vec1::<f32>()` of the full position grid — ~268 KB–1 MB synchronous D2H per forward, twice per step with audio; the comment claims "a few hundred floats"); `candle-gen-ideogram/src/transformer/model.rs:185-196` (three `to_vec1` readbacks of indicator/segment/position ids per forward, ~0.5 MB per step ×2 under asymmetric CFG)
- **Finding:** Both sc-8992 caches validate hits by reading back to host tensors that the pipeline itself constructed host-side one call earlier, injecting a blocking GPU sync into the denoise hot loop to compare keys that were already available cheaply.
- **Impact:** A needless per-step pipeline stall retained after the F-012 fix (small vs the DiT forward, but pure overhead), plus a misleading cost comment in ltx.
- **Suggested fix:** Key on `(dims, fps)` / pass a prepared-cond handle from the pipeline instead of reading tensor contents back.
- **Confidence:** High

#### [F-109] `AdaptLinear` re-casts/transposes frozen adapter factors on every forward
- **Category:** efficiency
- **Severity:** Low
- **Location:** `candle-gen/src/quant/adapt.rs:69-81` (`Adapter::residual`), `:202-221` (`LokrFactors::residual`, esp. `.t()?.contiguous()?` at :205)
- **Finding:** The inference-side additive residuals re-run `a.to_dtype(xd)`/`b.to_dtype(xd)` and (LoKr) `w2.to_dtype(xd)?.t()?.contiguous()?` on every forward, though the factors are frozen after `push_lora`/`push_lokr_structured` and the activation dtype is fixed per run.
- **Impact:** Step-invariant allocations + kernel launches × adapted-projections × steps × CFG — tens of thousands of extra launches per Wan/Anima/qwen-edit Lightning render on a packed tier.
- **Suggested fix:** Pre-cast the factors and pre-transpose `w2` at push time (AdaptLinear is inference-only, so no differentiability constraint).
- **Confidence:** High

#### [F-110] `QEmbedding` dequantizes the entire vocab table on every forward
- **Category:** efficiency
- **Severity:** Low
- **Location:** `candle-gen/src/quant/mod.rs:729-741`
- **Finding:** The quantized embedding forward materializes the full `[vocab, hidden]` dense table (~1.5–2.2 GB f32 for a Qwen-class TE) and then index-selects a handful of rows, once per prompt encode (twice under CFG).
- **Impact:** A multi-GB transient allocation + dequant kernel per request, eroding the packed tier's VRAM headroom precisely during TE encode.
- **Suggested fix:** Dequantize only the needed rows (index-select the `wq/scales/biases` rows before repack), or cache the dense table across a request's CFG branches.
- **Confidence:** High (behavior); Medium (priority)

#### [F-111] Give `AdaptLinear::detect` a group-size-aware variant
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `candle-gen/src/quant/adapt.rs:323-356`
- **Finding:** `detect` recovers `in_features` as `scales.dims()[1] * MLX_GROUP_SIZE` and repacks at hardcoded group 64, with no `detect_gs` twin — although the crate's own docs (`repack.rs:12-18`) state the group size is not recoverable from shapes, and every sibling loader grew an explicit-`gs` entry point for the group-32 tiers.
- **Impact:** Pointing an `AdaptLinear::detect`-based loader at a group-32 tier derives 2× the true `in_features` and fails with a misleading "unsupported bit-width" message; the asymmetry invites a copy-paste port that assumes 64.
- **Suggested fix:** Add `detect_gs(vb, name, group_size)`; make `detect` the documented group-64 wrapper.
- **Confidence:** High

#### [F-112] `CublasLt::run` leaks descriptors/layouts on early-error paths
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `candle-gen/src/quant/cublaslt.rs:529-647` (creates at :547,:580-587; destroys only at :641-645)
- **Finding:** `run` creates a matmul desc, three matrix layouts, and a preference object, but every fallible call between creation and the trailing `destroy_*` block returns early with `?` — a failing layout creation or heuristic search leaks up to five device-side handles.
- **Impact:** A worker that retries after transient cublasLt errors accumulates leaked descriptor objects for the process lifetime.
- **Suggested fix:** Wrap the handles in small `Drop` guards.
- **Confidence:** High

#### [F-113] Route the testkit VRAM sampler through the trusted nvidia-smi resolver
- **Category:** security
- **Severity:** Low
- **Location:** `candle-gen/src/testkit.rs:384-403` (also `candle-gen-lens/tests/encoder_quant_parity.rs:76`)
- **Finding:** `gpu_peak::used_mib` spawns `Command::new("nvidia-smi")` unqualified — the exact process-search-order vector F-030 closed for production in this same crate. The fixed resolver (`crate::gpu::resolve_nvidia_smi`) sits one module away and is not used.
- **Impact:** Test-only, but the CUDA gate runs this routinely on the dev box; a planted `nvidia-smi.exe` on the PATH executes with the developer's privileges during every gated push.
- **Suggested fix:** Call the trusted resolver and return `None` when unresolved; delete the duplicate query code.
- **Confidence:** High

#### [F-114] SDXL LoKr metadata `rank`/`alpha` parse failures silently default
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `candle-gen-sdxl/src/adapters.rs:224-234`
- **Finding:** `merge_lokr_file` reads `rank`/`alpha` from file metadata with `.and_then(|s| s.parse().ok()).unwrap_or(…)` — a *present but malformed* value (e.g. `"rank": "4x"`) silently becomes `rank = 1.0`, changing the merged delta's scale with no error, on user-supplied third-party files.
- **Impact:** A corrupt/nonstandard LoKr merges at the wrong strength and renders silently mis-adapted output — the same untrusted-input class F-009 fixed for `.alpha` tensors.
- **Suggested fix:** Distinguish absent (default) from unparseable (typed error naming the key), matching `read_scalar_opt` semantics.
- **Confidence:** High

#### [F-115] `EulerAncestralSampler::new(1, …)` produces a NaN σ table
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `candle-gen-sdxl/src/sampler.rs:67-93` (line 84: `i as f64 / (n - 1) as f64`)
- **Finding:** The public constructor rejects `train_steps == 0` but at `train_steps == 1` computes `0.0/0.0 = NaN`, yielding `sigmas = [0.0, NaN]`. The Kolors sibling handles this corner explicitly (`candle-gen-kolors/src/sampler.rs:52-56`).
- **Impact:** Latent — only `sdxl()` (1000) is used today; a future pub caller gets silent NaN latents.
- **Suggested fix:** Mirror the Kolors guard or reject `train_steps < 2`.
- **Confidence:** High

#### [F-116] SDXL `lightning` renders silently drop a validated `scheduler` selection
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `candle-gen-sdxl/src/pipeline.rs:703-738` (lightning branch never consults `req.scheduler`) vs the unconditional scheduler menu at `candle-gen-sdxl/src/lib.rs:409-412`
- **Finding:** `validate` accepts `sampler: "lightning"` combined with any advertised scheduler, but the lightning path uses its own fixed trailing schedule and ignores `req.scheduler` entirely — the F-004 shape in miniature.
- **Impact:** A `lightning + karras` request validates and renders byte-identically to `lightning` alone — a quiet false-capability edge on the axis this codebase explicitly polices.
- **Suggested fix:** Reject a non-default `scheduler` when `sampler == "lightning"` in `validate` (or document + honor it).
- **Confidence:** High

#### [F-117] Boogu Turbo's DMD renoise RNG stream collides with sibling batch-image seeds
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `candle-gen-boogu/src/pipeline.rs:263-267,479-491`; convention at `candle-gen/src/seed.rs:47-56`
- **Finding:** Batch image *i* renders at `base_seed + i` and its step-*s* renoise draws from `StdRng(base_seed + i + s)` — byte-identical to image *i+s*'s initial-noise stream in the same batch. This is precisely the collision `STEP_RNG_SALT` was hoisted into core to prevent; boogu's DMD loop predates the salt and never adopted it.
- **Impact:** Images within a `count > 1` Turbo batch share correlated noise draws (image *i*'s first renoise **is** image *i+1*'s initial latent), reducing batch diversity in a hard-to-attribute way.
- **Suggested fix:** Key the renoise stream with `STEP_RNG_SALT` — after confirming the mlx twin so the (output-changing) fix lands lockstep.
- **Confidence:** High (collision arithmetic); Medium (visual impact)

#### [F-118] MRoPE/vision-splice machinery duplicated verbatim between the boogu and krea text encoders
- **Category:** redundant
- **Severity:** Low
- **Location:** `candle-gen-krea/src/text_encoder.rs:186-212,274-284,432-644` vs `candle-gen-boogu/src/text_encoder.rs:43-71,135-145,280-445`
- **Finding:** `Rotary`, `repeat_kv`, `image_blocks`, `replace_seq`/`slice_seq`, `mrope_positions`, `mrope_cos_sin` (self-described "ported verbatim from candle_gen_boogu"), and `causal_mask` are near-byte-identical in both crates — yet krea already depends on boogu for the vision tower, so the shared path existed.
- **Impact:** ~250 lines of parity-critical Qwen3-VL grounding logic (interleaved-MRoPE axis selection is a known drift magnet) must now be fixed twice.
- **Suggested fix:** Export the grounding helpers from boogu (or hoist to `candle-gen`) and have krea's TE consume them.
- **Confidence:** High

#### [F-119] Krea control-checkpoint meta read indexes `[0]` on a possibly-empty tensor
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `candle-gen-krea/src/control.rs:323-326`
- **Finding:** `from_checkpoint` reads the `meta.inject_offset` tensor with `to_vec1::<f32>()?[0]`, which panics on a size-0 tensor in a malformed/truncated control-branch checkpoint — the class F-009 fixed for adapters; the hardened `read_scalar` is already a dependency.
- **Impact:** A corrupt studio-trained overlay crashes the worker thread at `Krea2Control::load` instead of returning a typed error.
- **Suggested fix:** `candle_gen::train::merge::read_scalar(META_INJECT_OFFSET, t)? as usize`.
- **Confidence:** High

#### [F-120] `MAX_TEXT_TOKENS` defined twice in krea src (plus three example copies)
- **Category:** redundant
- **Severity:** Low
- **Location:** `candle-gen-krea/src/pipeline.rs:71` and `candle-gen-krea/src/training.rs:99` (consumers split between them); redefined in `examples/krea-control-train.rs:43`, `krea-control-infer.rs:36`, `krea-control-diag.rs:30`
- **Finding:** The same 1024 RoPE-table cap exists as two independent crate constants with different consumers (`control_provider.rs`/`control_trainer.rs` import the *training* one), so raising the inference cap would silently leave the control/training lanes sized differently.
- **Impact:** Latent train-vs-infer cap drift; a mismatch surfaces as the opaque `narrow` error sc-9047 eliminated.
- **Suggested fix:** One `pub(crate)` constant, imported everywhere including examples.
- **Confidence:** High

#### [F-121] ConvRot forward panics via `expect` inside `OnceLock::get_or_init`
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `candle-gen-krea/src/quant.rs:108-124`
- **Finding:** The lazily-built CUDA `Int8Linear` uses `.expect(…)` inside `get_or_init`, so a cublasLt initialization failure aborts the sampler thread mid-render instead of returning the crate's typed error (the ConvRot load path otherwise errors cleanly).
- **Impact:** Contained (documented non-shipping variant) but a library-runtime panic (F-041 class) on a path a worker can reach through `LoadSpec::text_encoder`.
- **Suggested fix:** Build the `Int8Linear` eagerly in `QLinear::convrot_int8`, where `?` is available.
- **Confidence:** High

#### [F-122] Reduce the Wan ComfyUI dequant's double-resident host map (and fix its memory claim)
- **Category:** efficiency
- **Severity:** Low
- **Location:** `candle-gen-wan/src/comfyui.rs:126-168` (doc claim at :170-173); callers `candle-gen-wan/src/wan14b.rs:175-206`
- **Finding:** `dequant_scaled_fp8_map` takes ownership of `src` but iterates it by reference while inserting into `out`, so the entire fp8 source (~14 GB per A14B expert) and the full bf16 output (~28 GB) are resident simultaneously. The doc's claim that "peak host memory stays one tensor, not the whole expert" is true only of the f32 intermediate.
- **Impact:** ~42 GB transient host-RAM peak per expert load (twice, sequentially) on the in-place ComfyUI lane.
- **Suggested fix:** Drain the source map (`into_iter`/`remove` per key) so each source tensor drops as its converted twin is inserted; reword the doc.
- **Confidence:** High

#### [F-123] Dead default constants: `DEFAULT_FRAMES_VACE` and all four SVD defaults
- **Category:** dead-code
- **Severity:** Low
- **Location:** `candle-gen-wan/src/config.rs:252`; `candle-gen-svd/src/config.rs` (four `DEFAULT_*`) vs `candle-gen-svd/src/pipeline.rs:34-47`
- **Finding:** `DEFAULT_FRAMES_VACE` has no consumer (VACE derives frames from the ControlClip). SVD's four `DEFAULT_*` consts are unreferenced — `SvdParams::default()` independently hardcodes the same values (25/25/7/7).
- **Impact:** A future default change edits one copy and silently leaves the other; the dead VACE const implies a frames default that doesn't exist.
- **Suggested fix:** Delete the VACE const; have `SvdParams::default()` read the config consts.
- **Confidence:** High

#### [F-124] wan-vace silently ignores `req.frames`
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `candle-gen-wan/src/model_vace.rs:140-167` (render derives everything from the clip), `:274-315` (validate never checks `frames`)
- **Finding:** The VACE output frame count comes solely from the ControlClip; a request carrying `frames: Some(33)` with a 17-frame clip validates and renders 17 frames with no diagnostic — the silently-ignored-request-knob class F-043 fixed for LTX `steps`.
- **Impact:** A caller setting `frames` on `wan_vace` gets a different-length video than requested with zero feedback.
- **Suggested fix:** Reject `req.frames` when it disagrees with `clip.frames.len()`, mirroring the sc-9027 LTX treatment.
- **Confidence:** High

#### [F-125] SCAIL-2 multi-segment progress restarts from 1 per segment
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `candle-gen-scail2/src/generate.rs:391-427` (esp. :424-427; segment loop at :346)
- **Finding:** `Progress::Step` uses `total = steps` and `current = i+1` per segment, so for driving clips > 81 frames the reported progress runs 1→N and then jumps back per segment, with `Decoding` firing mid-run.
- **Impact:** Percent-complete goes backwards on exactly the long jobs where progress matters most.
- **Suggested fix:** `total = steps × segments.len()`, `current = seg_idx·steps + i + 1`.
- **Confidence:** High

#### [F-126] `WanVae16::encode` silently drops trailing frames on unaligned clip lengths
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `candle-gen-wan/src/vae16.rs:434-461` (chunk math at :438-452)
- **Finding:** `num_chunks = 1 + (t-1)/4` consumes exactly `1 + 4·(num_chunks-1)` frames, so for `t % 4 != 1` the trailing frames are never encoded and vanish from the latent. All in-repo callers pre-align, but the method is `pub` and `Scail2Job` is fully `pub`.
- **Impact:** A misaligned clip encodes shorter, surfacing later as a confusing shape mismatch instead of the loud typed error this codebase's public boundaries standardize on.
- **Suggested fix:** Return a typed error when `(t - 1) % 4 != 0`.
- **Confidence:** High

#### [F-127] wan14b-txt2video example silently drops `--lora-high/--lora-low` in ComfyUI mode
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `candle-gen-wan/examples/wan14b-txt2video.rs:68-115`
- **Finding:** The example parses the LoRA flags into `adapters`, but the ComfyUI branch calls `load_from_comfyui_experts(…)` (which takes no adapters) — a command combining both flag sets silently renders unadapted, despite the header advertising the lightx2v Lightning pair as the main LoRA use case.
- **Impact:** A GPU-validation A/B combining the sc-10671 in-place path with a Lightning distill produces base-model output with no warning, invalidating the eyeball comparison.
- **Suggested fix:** Error (or loud warning) when both flag sets are supplied.
- **Confidence:** High

#### [F-128] SeedVR2 chunked video assembly holds ~2–3× the full decoded clip in host RAM
- **Category:** efficiency
- **Severity:** Low
- **Location:** `candle-gen-seedvr2/src/pipeline.rs:443-460`; `candle-gen-seedvr2/src/video.rs:155-175`
- **Finding:** `generate_video` accumulates every chunk's RGB8 frames (including duplicated overlap frames) into `chunk_frames`, then `assemble_overlap` clones each frame into the output — the full video exists twice-plus at peak.
- **Impact:** A 300-frame 4K clip is ~7.5 GB of frames; peak host RSS hits ~15–17 GB for data that could stream chunk-by-chunk.
- **Suggested fix:** Blend incrementally as chunks arrive (move, don't clone).
- **Confidence:** High

#### [F-129] sam3 hotstart `keep_alive` and `FrameMem.object_score` are write-only state
- **Category:** dead-code
- **Severity:** Low
- **Location:** `candle-gen-sam3/src/video.rs:39-41,97,477-501` (`keep_alive`); `:68,218,285,643-647` (`object_score`)
- **Finding:** `keep_alive` is initialized, incremented, and decremented every frame but never read for removal/suppression (removal is driven solely by `unmatched_frames`/`overlap_pairs`); `object_score` is written in three places and read nowhere.
- **Impact:** ~30 lines of inert per-frame bookkeeping; if the `transformers` reference gates track removal on keep-alive, this is a silent behavioral divergence rather than just dead code.
- **Suggested fix:** Wire `keep_alive` into removal to match the reference (verify against `modeling_sam3_video.py` and the mlx twin) or delete both fields with a comment citing the check.
- **Confidence:** Medium

#### [F-130] LTX config carries dead fields and a pointless keep-alive shim
- **Category:** dead-code
- **Severity:** Low
- **Location:** `candle-gen-ltx/src/config.rs:56-59,72-74,178,200,206-210`; `candle-gen-ltx/src/lib.rs:560-563`
- **Finding:** `TransformerConfig.in_channels`/`out_channels`/`adaln_coeff` and `GemmaConfig.sliding_window` are set but never read (`AvStream` hardcodes `coeff: 9`); `GemmaConfig::num_hidden_states()` has no caller; `_defaults_referenced()` "references" three items that are all `pub` (no lint would fire) — the shim itself is dead.
- **Impact:** Dead config surface implies configurability that doesn't exist (the F-050 footgun): a checkpoint variant changing `adaln_coeff` would silently do nothing.
- **Suggested fix:** Delete the unread fields + shim, or wire `adaln_coeff` into `AvStream::load`.
- **Confidence:** High

#### [F-131] LTX `validate` accepts unbounded frame counts, bypassing every memory budget except the VAE's
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `candle-gen-ltx/src/lib.rs:438-445` (frames check), `:344-376` (unbudgeted denoise)
- **Finding:** `validate` only checks `frames % 8 == 1` with no upper bound; only the VAE decode stage is budget-guarded. A `frames: 2001` request at 1280² passes validation and OOMs in the 22B AvDiT denoise loop instead of the catchable `TilingBudgetError`-style rejection the decode stage provides.
- **Impact:** Worker-facing requests can drive an uncatchable allocation failure in the denoise stage.
- **Suggested fix:** Add a max-frames (or max token count) cap in `validate`, sized against the envelope validated on the target GPU.
- **Confidence:** Medium

#### [F-132] Qwen sequential env override never evicts the resident components cache (mixed-mode double residency)
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `candle-gen-qwen-image/src/lib.rs:639-646` (with `sequential_offload_enabled`, :547-551)
- **Finding:** `CANDLE_GEN_OFFLOAD` is re-read on every `generate`. If a prior request ran resident (populating `self.components`), a later sequential request loads a second TE/DiT/VAE while the cached `Arc` set stays alive — peak ≈ resident + sequential working set, the opposite of the flag's purpose. The edit sibling captures the env once at load, so the lanes also disagree on sampling time.
- **Impact:** Flipping the env mid-process makes the "low-peak" path the highest-peak path — a likely OOM in exactly the constrained-VRAM situation sequential exists for. Exposure low today (only the A/B harness drives the env).
- **Suggested fix:** Take/clear the components cache when the sequential branch is taken, or capture the env once at `load`.
- **Confidence:** High

#### [F-133] Z-Image trainer keeps a fourth copy of the tokenizer policy that common.rs was created to centralize
- **Category:** redundant
- **Severity:** Low
- **Location:** `candle-gen-z-image/src/training.rs:288-300` vs `candle-gen-z-image/src/common.rs:78-92`
- **Finding:** `training::load_tokenizer` re-declares the exact `TokenizerConfig` that `common::tokenizer_config()`/`build_tokenizer` centralize — and common.rs's stated purpose is that the entry points "can never drift on the tokenization policy". The trainer's caption encode is parity-critical with inference.
- **Impact:** A tokenizer-policy change (the sc-8646 class) lands in common.rs but silently misses the trainer, training adapters against differently-tokenized captions.
- **Suggested fix:** Replace with `common::build_tokenizer(root, "z_image trainer")`.
- **Confidence:** High

#### [F-134] Qwen crate holds three byte-identical `to_image` copies and a duplicated tokenizer-config block
- **Category:** redundant
- **Severity:** Low
- **Location:** `candle-gen-qwen-image/src/lib.rs:527-540`, `candle-gen-qwen-image/src/edit.rs:625-638` vs the shared home `candle-gen-qwen-image/src/control_common.rs:115-128`; tokenizer config duplicated `lib.rs:240-249` vs `:397-406`
- **Finding:** `control_common::to_image` is already the shared `pub(crate)` home (created by the F-074 dedup), but `lib.rs` and `edit.rs` kept private copies; the small tokenizer-config block is duplicated between the resident and sequential loaders.
- **Impact:** A postprocess or tokenizer-policy fix must land multiple times inside one crate — the copy-drift class F-074 closed for the control lanes.
- **Suggested fix:** Point `lib.rs`/`edit.rs` at the shared helper; extract one `tokenizer_config()`.
- **Confidence:** High

#### [F-135] Adapters on a packed MLX tier fail with a misleading "wrong adapter format" error (sd3, lens)
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `candle-gen-sd3/src/pipeline.rs:310-337`, `candle-gen-lens/src/lib.rs:195-207`
- **Finding:** With adapters present, both crates CPU-load `transformer/` and merge even when the tier is packed (u32 codes + `.scales`). Every delta shape-mismatches the packed `.weight`, `merged == 0`, and `no_target_matched` fires with a message blaming the adapter's key format. sd3's own comment acknowledges the merge/packed incompatibility but only uses it to pick the quant path.
- **Impact:** A valid LoRA paired with a hosted `*-mlx` packed snapshot is diagnosed as a malformed adapter, sending the user down the wrong debugging path.
- **Suggested fix:** Check the packed config first and return "LoRA/LoKr merge requires a dense tier; this snapshot is a pre-quantized MLX pack".
- **Confidence:** High

#### [F-136] Skip the all-zero joint attention mask on the lens cond-only path
- **Category:** efficiency
- **Severity:** Low
- **Location:** `candle-gen-lens/src/lib.rs:302-311` (unguided branch) with `:378-391` and `candle-gen-lens/src/transformer.rs:669-685`
- **Finding:** When guidance is off (the `lens_turbo` default), `encode_prompt` returns an all-ones mask → all-zero additive tensor → `denoise` still passes `Some(mask)`, forcing a broadcast add onto the `[B,H,Sq,Sk]` scores (the model's largest tensor) in each of 48 blocks every step. `forward` documents `text_valid: None` as the skip path.
- **Impact:** ~192 avoidable full-scores kernels per default 4-step turbo render — pure overhead on the crate's most common configuration.
- **Suggested fix:** Return `None` for the mask in the `!guided` branch (the guided path's zero-mask uncond half is load-bearing; keep it).
- **Confidence:** High

#### [F-137] Convert the lens trainer twin's pub-API `assert_eq!` to a typed error
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `candle-gen-lens/src/dit_train.rs:399-405`
- **Finding:** `LensTransformerTrain::forward_pre_main` (pub) still `assert_eq!`s the text-feature layer count, while its inference twin was converted to a typed error by the F-041 fix (sc-9025). The two forwards are documented lockstep mirrors.
- **Impact:** A malformed cached-feature set aborts the training process instead of surfacing a catchable error — inconsistent with the crate's own F-041 remediation.
- **Suggested fix:** Mirror the inference twin's typed error.
- **Confidence:** High

#### [F-138] PuLID embeds every detected face but uses only the largest
- **Category:** efficiency
- **Severity:** Low
- **Location:** `candle-gen-pulid/src/pulid_flux.rs:248-265`
- **Finding:** `compute_id_embedding` calls `FaceAnalysis::analyze` — which norm-crops and ArcFace-embeds **all** detections in one batched forward — then keeps only `faces.first()`. The detect-then-embed-largest pattern exists in `candle-gen-face` and is what InstantID uses.
- **Impact:** For a group-photo reference, N−1 wasted host warp-affines plus an N-row iresnet100 forward per generation, plus a behavioral asymmetry between the two identity providers.
- **Suggested fix:** Replace with `detect(…)` + `embed(…)` on the largest detection.
- **Confidence:** High

#### [F-139] Depth estimator panics on zero-dimension images and unchecked buffers at pub boundaries
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `candle-gen-depth/src/lib.rs:94-98,121-129`; `candle-gen-depth/src/preprocess.rs:27-51`
- **Finding:** `estimate_control_rgb8` validates only `rgb.len() != w*h*3`, so a 0×0 image with an empty buffer passes and reaches `resize_rgb8_to_unit`, where `.min(in_h - 1)` underflows (debug panic) and `rgb[0]` indexes an empty slice (release panic). The pub `preprocess_rgb8` performs no buffer-size check at all. The face crate guards both cases with typed errors — the F-041/F-042 class re-appearing in the sibling preprocessor.
- **Impact:** A malformed worker image aborts the worker thread on the auto-depth Fun-Controlnet lane.
- **Suggested fix:** Reject zero dimensions and undersized buffers with `CandleError::Msg`, mirroring `face detect`.
- **Confidence:** High

#### [F-140] FLUX dev time-shift constants / `flow_mu` now maintained in three places
- **Category:** redundant
- **Severity:** Low
- **Location:** `candle-gen-pulid/src/pulid_flux.rs:50-51,129-133` vs `candle-gen-flux/src/pipeline.rs:99-100,113-119` and `candle-gen-flux/src/ip_provider.rs:47`
- **Finding:** `BASE_SHIFT`/`MAX_SHIFT` and the `flow_mu` linear map are parity-critical schedule constants copied three times; PuLID's copy documents itself as "the candle-gen-flux `flow_mu` twin", and PuLID already depends on `candle-gen-flux` (where `flow_mu` is `pub(crate)`).
- **Impact:** A dev-schedule constant fix must land in three files; drift silently desynchronizes PuLID's scheduler axis from the base FLUX pipeline it is documented to match.
- **Suggested fix:** Export `flow_mu`/the shift constants from `candle-gen-flux` (or hoist next to `resolve_flow_schedule` in `candle-gen`).
- **Confidence:** High

#### [F-141] `PidEngine::load` does not thread `BackboneSpec.pid_scale` into the config
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `candle-gen-pid/src/engine.rs:67-69` (spec fields at `registry.rs:56-57`)
- **Finding:** The engine copies `latent_channels` and `latent_spatial_down_factor` from the registry spec onto `PidConfig::sr4x()` but not `pid_scale` — `cfg.sr_scale` stays the hard-coded 4, though the registry deliberately carries `pid_scale` per space ("4× or 8×").
- **Impact:** If an 8× student ships (the config documents the possibility), the engine silently sizes noise, output geometry, and the budget guard for 4× — a wrong-resolution decode rather than a load error.
- **Suggested fix:** `cfg.sr_scale = spec.pid_scale;` beside the other two copies (or a `debug_assert_eq!` until an 8× student exists).
- **Confidence:** High

#### [F-142] Dead configuration surface in candle-gen-pid
- **Category:** dead-code
- **Severity:** Low
- **Location:** `candle-gen-pid/src/config.rs:43-55,170-198` (re-exported at `lib.rs:39`)
- **Finding:** `CaptionConfig` is publicly exported with zero consumers, and `PidConfig.use_text_rope`/`rope_mode`/`lq_in_channels` are constructed but never read — `rope_1d_text` is applied unconditionally regardless of `use_text_rope`.
- **Impact:** The F-050 footgun: a variant port setting `use_text_rope: false` compiles and silently runs the NTK+text-RoPE path anyway.
- **Suggested fix:** Honor the flags (error on unsupported combinations) or delete the fields/struct and note the fixed policy.
- **Confidence:** High

#### [F-143] `Qwen25VlText::forward` retains every hidden state when only `[-2]` is needed
- **Category:** efficiency
- **Severity:** Low
- **Location:** `candle-gen-bernini/src/qwen2_5_vl.rs:322-363`
- **Finding:** `forward` pushes all 29 `[B,L,3584]` hidden states into a `Vec` to mimic HF `output_hidden_states`, but its only caller uses index `len-2`. In the MAR loop this runs 3× per planning step (75+ calls per generate), each holding ~1 GB extra bf16 activations.
- **Impact:** Avoidable peak-VRAM pressure in the planner stage, stacked on the 7B backbone weights.
- **Suggested fix:** Track only the previous-layer hidden state; keep the full-Vec API for tests behind a separate method.
- **Confidence:** High

#### [F-144] Bernini MAR loop burns 3 backbone forwards on steps with nothing to reveal
- **Category:** efficiency
- **Severity:** Low
- **Location:** `candle-gen-bernini/src/mar.rs:259-273`
- **Finding:** `sample_vit_embed` runs the cond/uncond/imgcond backbone passes *before* the `revealed.sum() == 0` skip. When `n_query < planning_step`, the trailing steps have empty reveal sets yet still execute three full 7B forwards whose outputs are discarded.
- **Impact:** Up to ~3×(planning_step − n_query) wasted 7B forwards per generate for small targets.
- **Suggested fix:** Hoist the skip check (which depends only on the precomputed schedule) above the three `stream_for_vit` calls.
- **Confidence:** High

#### [F-145] Bernini sidecar/config reads silently swallow malformed JSON (F-073 class, new sites)
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `candle-gen-bernini/src/config.rs:81-106` (`BerniniKnobs::from_dir`); `candle-gen-bernini/src/bernini.rs:124-147` (`PlannerKnobs::from_dir`), `:150-170` (`read_mrope_config`)
- **Finding:** All three loaders chain `.ok()?.and_then(…ok())` and fall back to defaults on a corrupted sidecar, indistinguishable from "file absent". A damaged `qwen2_5_vl_config.json` silently swaps MRoPE sections and token ids; a damaged renderer sidecar silently flips `switch_dit_boundary`/`shift`.
- **Impact:** A corrupted snapshot degrades quality (wrong expert boundary, wrong MRoPE geometry) with zero signal — the downgrade class F-073 recorded, in a crate where the mrope config is load-bearing.
- **Suggested fix:** Distinguish `NotFound` (defaults OK) from parse failure (error naming the file), as the krea/boogu loaders now do.
- **Confidence:** High

#### [F-146] Anima re-encodes the fixed prompt per image in the count loop
- **Category:** efficiency
- **Severity:** Low
- **Location:** `candle-gen-anima/src/lib.rs:265-282`; `candle-gen-anima/src/pipeline.rs:127-131`
- **Finding:** `AnimaPipeline::generate` calls `encode_prompt` (Qwen3-0.6B forward + conditioner) for cond and uncond inside the per-seed closure, so a `count = 8` request runs 16 seed-independent text encodes instead of 2. Sibling providers hoist the encode out of `for_each_image_seed`.
- **Impact:** ~14 redundant TE+conditioner forwards per max-count batch.
- **Suggested fix:** Compute `cond`/`uncond` once and pass them into the per-seed closure.
- **Confidence:** High

#### [F-147] `check-cuda.ps1` sets `CUDA_COMPUTE_CAP` with a trailing space via unquoted cmd `set`
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `scripts/check-cuda.ps1:63`
- **Finding:** The inner cmd line is `set CUDA_COMPUTE_CAP=$ComputeCap && …`; cmd's unquoted `set` includes everything up to `&&`, so the value is `"80 "` (trailing space). It works today only because cudaforge's `GpuArch::parse` happens to `trim()` (verified in `cudaforge-0.1.6/src/compute_cap.rs:37`). The same line already uses the safe quoted form for `CUDA_PATH`.
- **Impact:** A latent, environment-dependent break of the repo's primary local CUDA gate; also a copy-paste trap for the next script.
- **Suggested fix:** `set "CUDA_COMPUTE_CAP=$ComputeCap" && …`.
- **Confidence:** High

#### [F-148] Vendored-fork docs disagree with themselves about the delta count
- **Category:** readability
- **Severity:** Low
- **Location:** `vendor/candle-kernels/VENDORED.md:18-20`; `vendor/candle-kernels/build.rs:46`
- **Finding:** VENDORED.md's heading still reads "The only change vs upstream" immediately above the corrected text "There are **two** changes"; `build.rs:46` still claims its gencode block is "THE ONLY change" although the sc-9601 `cast.cu` block landed 2026-07-03. (The actual delta was re-verified this review: exactly the two documented changes, nothing undocumented.)
- **Impact:** The next re-vendor (mandatory on every candle pin bump) is guided by these docs; a maintainer trusting the build.rs claim could re-apply only the gencode block, silently dropping the i32 casts and reintroducing the INT8-ConvRot regression.
- **Suggested fix:** Retitle the section and update the build.rs comment to "one of two changes — see VENDORED.md".
- **Confidence:** High

#### [F-149] Add a dependency-advisory scan to CI
- **Category:** security
- **Severity:** Low
- **Location:** `.github/workflows/ci.yml` (absent lane); `Cargo.lock`
- **Finding:** CI has fmt/clippy/check/test/skew lanes but no `cargo audit`/`cargo deny` step — the July review's coverage notes already flagged this. The lockfile carries a meaningful network/parsing surface (hf-hub/ureq/rustls-tls, tokenizers, image png/jpeg decoders) that processes remote-origin bytes at runtime.
- **Impact:** A RUSTSEC advisory in e.g. the PNG decoder or rustls goes unnoticed; everything is pinned, so nothing self-heals.
- **Suggested fix:** Add a non-blocking (or scheduled) `cargo audit` job on the committed `Cargo.lock`; promote to blocking once triaged.
- **Confidence:** High

#### [F-150] Doc/comment contradictions and stale claims (grab bag)
- **Category:** readability
- **Severity:** Low
- **Location:** `candle-gen-qwen-image/src/lib.rs:18-19,661-662` (crate header still says Edit/ControlNet/Lightning/LoRA/quant are "deferred and rejected" — all shipped in this same crate); `candle-gen-sdxl/src/sampler.rs:3-8` + `candle-gen-sdxl/tests/conformance.rs:29-34` + `candle-gen-sdxl/Cargo.toml:5` + `candle-gen-sdxl/src/unet/controlnet.rs:37-39` (four pre-sc-10826/F-061 claims); `candle-gen-scail2/src/model.rs:13-14` (f32 14B DiT called "~28 GiB"; it is ~56 GiB — `pipeline.rs:123-129` has it right); `candle-gen-lens/src/transformer.rs:116-131` (orphaned doc of the removed budget constant fused onto the `attention` doc); `candle-gen-flux2/src/pos_embed.rs:8-9` (RoPE formula typo: `out1 = i·cos + r·cos` should be `+ r·sin`); `candle-gen-flux2/examples/flux2-txt2img.rs:6-7,71` (says klein rejects `--quant`; stale since sc-11031); `candle-gen-bernini/src/vit_guidance.rs:12` (shape doc contradicts call sites); `CODEGRAPH.md:10-16` (describes the 29-crate lockstep-pinned workspace as a self-contained "boogu" system)
- **Finding:** Nine sites where load-bearing documentation contradicts the code or the current architecture — the F-066 class regrown after that grab bag was fixed.
- **Impact:** In a codebase whose comments are explicitly load-bearing for parity work, each of these misdirects the next reader; the scail2 VRAM figure and flux2 formula are the kind of "doc" that gets trusted during ports.
- **Suggested fix:** One-line edits at each site; regenerate or drop CODEGRAPH.md.
- **Confidence:** High

## Informational

#### [F-151] ~10 env vars silently change production rendering behavior, with no central registry
- **Category:** bad-pattern
- **Severity:** Info
- **Location:** `CANDLE_GEN_OFFLOAD` (flux2 `lib.rs:663-672`, flux `lib.rs:613`/`pipeline.rs:983`, qwen `lib.rs:548`); `PID_PIXEL_POS_ABS` (`candle-gen-pid/src/backbone/layers.rs:168` — switches forward-path numerics); `SVD_FORCE_F16`/`SVD_FORCE_BF16`/`SVD_DEBUG` (`candle-gen-svd/src/lib.rs:84-86,398`); `SENSENOVA_DISTILL_LORA` (`distill.rs:106`); `WAN_VAE_BUDGET_GIB`/`SEEDVR2_BUDGET_GIB`/LTX equivalent; `LTX_GEMMA_DIR` (`lib.rs:165`); `HF_HOME`/`HF_HUB_CACHE`. (~28 further vars are test-harness-only.)
- **Finding:** Each is individually documented at its site, but there is no single inventory, and two (`PID_PIXEL_POS_ABS`, `SVD_FORCE_*`) change output numerics rather than just performance/residency.
- **Impact:** A stray env var in a worker deployment changes renders or VRAM behavior with nothing in logs pointing at the cause.
- **Suggested fix:** Document the production-affecting set in the README (or `docs/env.md`) and log the chosen value at load time where a var wins.
- **Confidence:** High

#### [F-152] Drop the redundant `cargo check --workspace` CI step
- **Category:** efficiency
- **Severity:** Info
- **Location:** `.github/workflows/ci.yml:45-46`
- **Finding:** The check step runs after `cargo clippy --workspace --all-targets -- -D warnings`, which already type-checks a strict superset with the same feature set.
- **Impact:** Redundant minutes on both matrix lanes per push; extra target-dir churn on the disk-flaky ubuntu lane.
- **Suggested fix:** Delete the step (keep the macOS-only `--features metal` check, which clippy does not cover).
- **Confidence:** High

#### [F-153] Merge-report observability residue: anima discards reports; ideogram prints to stderr
- **Category:** bad-pattern
- **Severity:** Info
- **Location:** `candle-gen-anima/src/loader.rs:184,201` (`let _report =`); `candle-gen-ideogram/src/pipeline.rs:176` + `candle-gen-ideogram/src/adapters.rs:99-103` (unconditional `eprintln!` of merge counts)
- **Finding:** Two new crates each landed on a different side of the F-043/F-051 question the workspace already ratified (silent library-side merges, sc-9035): anima silently drops the skipped-key counts; ideogram prints unstructured chatter from library load paths.
- **Impact:** Partially-degraded adapter applies are invisible in one crate and uncapturable noise in the other; both diverge from the ratified convention.
- **Suggested fix:** Align both with the sc-9035 convention (silent, with the report available to callers).
- **Confidence:** High

#### [F-154] Empty-prompt guards drifted: `is_empty()` vs `trim().is_empty()` across siblings
- **Category:** bad-pattern
- **Severity:** Info
- **Location:** `candle-gen-krea/src/lib.rs:191`, `candle-gen-boogu/src/lib.rs:124` (`is_empty`) vs `candle-gen-chroma/src/lib.rs:82` and `candle-gen-krea/src/control_provider.rs:223` (`trim().is_empty()`)
- **Finding:** Two registered validators accept a whitespace-only prompt while chroma and krea's own control provider reject it — behavioral drift among otherwise-identical validate blocks.
- **Impact:** Inconsistent request acceptance across the family for the same degenerate input.
- **Suggested fix:** Standardize on `trim().is_empty()`.
- **Confidence:** High

#### [F-155] Residual panic-prone reads/asserts on first-party inputs (scrfd, draw_kps)
- **Category:** bad-pattern
- **Severity:** Info
- **Location:** `candle-gen-face/src/scrfd.rs:104-108` (`flatten_all()?.to_vec1::<f32>()?[0]` on the per-level reg scale — panics on a zero-element tensor); `candle-gen-instantid/src/kps.rs:514-519` (pub `draw_kps` asserts `kps.len() >= 5`; exported at the crate root though in-crate callers pre-validate)
- **Finding:** Two low-likelihood F-009/F-041-class residues on first-party (not community) inputs.
- **Impact:** A malformed in-house conversion or a future direct caller gets an abort instead of a typed error.
- **Suggested fix:** `.first().copied().ok_or_else(…)` for scrfd; `Result` or a documented `# Panics` section for `draw_kps`.
- **Confidence:** High

#### [F-156] `single_safetensors` silently hands one shard to the SD3 CLIP builder
- **Category:** bad-pattern
- **Severity:** Info
- **Location:** `candle-gen-sd3/src/conditioning.rs:422-426`
- **Finding:** Returns the first *sorted* shard as "the" checkpoint. Stock SD3.5 CLIP encoders are single-file so this is currently safe, but a re-sharded snapshot would fail with a confusing "tensor not found" deep in the builder rather than a clear error at resolution.
- **Impact:** Future-proofing only.
- **Suggested fix:** Error when `files.len() > 1`, or thread the full list through an mmap VarBuilder like the T5 branch.
- **Confidence:** High

#### [F-157] Flow-match driver misreports a zero-step config as "cancelled"
- **Category:** bad-pattern
- **Severity:** Info
- **Location:** `candle-gen/src/train/flow_match.rs:637-639`
- **Finding:** `run_flow_match_training` never validates `cfg.steps`; the `steps_run == 0 ⇒ CandleError::Canceled` guard is also reached by a `steps: 0` config from a trainer that skipped `validate_flow_match_request`, surfacing a config error as a typed cancellation (which workers treat as user-initiated). Latent — all three adopters validate upstream.
- **Suggested fix:** Add a `cfg.steps > 0` check before the loop so the guard only ever means cancellation.
- **Confidence:** High

#### [F-158] `Int8Linear`'s retained `w_i8` is not "small" — document or drop it after staging
- **Category:** readability
- **Severity:** Info
- **Location:** `candle-gen/src/quant/eight_bit_linear.rs:83-88,140-166`
- **Finding:** After `from_per_channel_parts` pre-stages the codes to a device buffer, the struct keeps the full `(N, K)` source tensor with a comment claiming it "holds only the small CPU source (kept for shape queries)" — it is the entire weight's codes in the caller's carry dtype; only its dims are used.
- **Impact:** Misleading doc; latent multi-GB host retention for future callers.
- **Suggested fix:** Store `(n, k)` dims instead of the tensor (or fix the comment).
- **Confidence:** High

#### [F-159] `AdaptLinear` re-grows small duplicates of existing core seams — inside the commons itself
- **Category:** redundant
- **Severity:** Info
- **Location:** `candle-gen/src/quant/adapt.rs:35-51` (`Base`) vs `candle-gen/src/train/lora.rs:115-143` (`LoraBase`); `adapt.rs:144-164` (LoKr leg resolution) vs `lora.rs:746-764` (`reconstruct_lokr_delta`)
- **Finding:** PR #425's hoist introduced a third `Dense|Packed` base enum with identical bodies to the trainer's `LoraBase`, and byte-identical LoKr leg-resolution match arms (including identical error strings). The scale-convention difference between the two LoKr paths is deliberate and documented (sc-10578); the leg resolution is not part of that difference.
- **Impact:** The drift pattern re-seeding inside the crate that exists to prevent it: a leg-resolution fix must now land in two places.
- **Suggested fix:** Extract a shared `resolve_lokr_factors(…)`; consider unifying `LoraBase` with `adapt::Base`.
- **Confidence:** High

#### [F-160] Depth config carries five unread hyperparameter fields
- **Category:** dead-code
- **Severity:** Info
- **Location:** `candle-gen-depth/src/config.rs:22-48`
- **Finding:** `num_channels`, `mlp_ratio` (+ `intermediate_size()`), `neck_hidden_sizes`, `fusion_hidden_size`, `head_hidden_size` are never read — every affected shape rides the loaded checkpoint tensors.
- **Impact:** A Base/Large plug-in editing just these fields would appear configurable while changing nothing.
- **Suggested fix:** Mark reference-only or use in load-time shape assertions.
- **Confidence:** High

#### [F-161] Bernini guidance-mode names silently degrade the planner's system prompt
- **Category:** bad-pattern
- **Severity:** Info
- **Location:** `candle-gen-bernini/src/bernini.rs:621`; `candle-gen-bernini/src/template.rs:35-45`
- **Finding:** `resolve_vit_mode` accepts either a guidance-mode name (`rv2v_wapg`) or a task name (`rv2v`), but the raw string is passed to `BerniniTemplate` as the task: a guidance-mode name falls through the `system_prompt` match to the generic "You are a helpful assistant.", changing planner conditioning relative to the equivalent task-name request with no warning. (Also noted: `vit_one_step`'s `V2vApg` arm creates a fresh `MomentumBuffer::new(0.0)` per step — inert at momentum 0.0, a silent no-op if a nonzero default ever ships; `candle-gen-bernini/src/forward.rs:444-459`.)
- **Impact:** Two spellings of the same intent produce different renders, silently.
- **Suggested fix:** Map the resolved `VitMode` back to its canonical task name for templating; thread a persistent momentum buffer.
- **Confidence:** Medium

#### [F-162] Anima real-weight tests resolve the HF cache via `$HOME` (F-071 class, new sites)
- **Category:** bad-pattern
- **Severity:** Info
- **Location:** `candle-gen-anima/tests/parity_real_weights.rs:44,386`
- **Finding:** `split_files()` globs `$HOME/.cache/huggingface/hub/…` on a Windows-primary repo whose actual cache lives under `HF_HOME` (`D:\.cache\huggingface`) — the pattern F-071 fixed for the CLIP tests via `testkit::hf_cache_roots`.
- **Impact:** The gated parity lane silently can't find weights on the very box it's meant to run on.
- **Suggested fix:** Route through `candle_gen::testkit::hf_cache_roots`.
- **Confidence:** High

#### [F-163] `fold_adapters_into_packed_map` detects adapted layers by full-tensor bit-compare instead of the merge report
- **Category:** redundant
- **Severity:** Info
- **Location:** `candle-gen-sdxl/src/packed_adapters.rs:138-171`
- **Finding:** To learn which packed Linears the fold touched, the code snapshots every dequantized grid pre-merge and runs `tensors_differ` (a full f32 subtract + `max_all` per packed Linear, ~2.6B elements total) post-merge — because `MergeReport` records only counts, not which keys merged.
- **Impact:** An avoidable full extra CPU pass over the dequantized UNet on every packed+adapter load.
- **Suggested fix:** Extend `MergeReport` (or `merge_into`) to record merged base keys and repartition from that set.
- **Confidence:** High

## Themes and systemic observations

1. **The July remediation wave was real, disciplined, and nearly complete.** 71 of 78 prior findings are fixed, almost all with story tags and pinned regression tests; the big consolidations (one audited mmap loader, one QLinear seam with an explicit strategy knob, one adapter-merge skeleton, shared seed/salt, shared VAE tiling, shared testkit) all landed and were adopted. The remediation model — findings → stories → per-crate sweeps — demonstrably works and should be reused for this report.

2. **The failure mode has shifted from "fixes don't propagate" to "sweeps have edges, and new code re-grows old bugs."** Every prior *listed* site got its fix, but: the i32 attention sweep missed the stock (non-vendored) SDXL UNet, upstream FLUX.1 VAE code, and three newer surfaces (F-081); the steps==0 sweep missed five bespoke lanes including a brand-new crate (F-102); the PiD budget/tiling guard reached only the registered lanes, not the four bespoke copies (F-091); the area-cap fix stayed on one of three 14B lanes (F-090). Sweeps should end with a class-wide grep/checklist, and new crates need the checklist at PR time — bernini and anima each shipped several already-fixed bug classes (steps floor, silent config swallow, per-forward RoPE, `$HOME` cache, merge-report handling).

3. **One improvement wave planted the seeds of a regression wave.** The sc-8992 RoPE-cache hoists (fixing F-012) introduced `Mutex` caches at ~10 sites — almost all with `lock().unwrap()` (F-103, re-seeding the F-031 poison class), two with device→host readback keys (F-108), and two with a single-entry key that thrashes under CFG, the default on those paths (F-89). When a pattern is stamped across crates, the stamp itself deserves a review pass.

4. **Never-reviewed code carries the highest-severity defects.** All four Highs are in code that landed after (or was missed by) the last review: the bernini planner/vision tower (F-079/F-080, CPU-only tests hid both), the boogu ViT now on the krea edit path (F-081c), and the Wan MoE trainer loop (F-082). The pending sc-11003 GPU validation would have caught F-079 the hard way; a pre-validation review pass on large new subsystems is cheaper.

5. **The "descriptor advertises what the code drops" class persists at a low boil.** krea_2_edit drops `use_pid` (F-088), boogu edit can't serve its advertised reference sizes (F-087), SDXL lightning ignores a validated scheduler (F-116), bernini's `needs_conditioning` was never wired (F-096), wan-vace ignores `req.frames` (F-124). The repo already names this the "false-capability trap"; a conformance-test pattern (for each advertised capability × entry point, assert it is either honored or rejected) would mechanize it.

6. **Duplication is now bimodal.** Load-bearing duplication is down sharply (loader ~34→1, QLinear 4→1 seam, merge skeleton 8→1 core). Cosmetic/utility duplication is *growing with each new family*: `to_image` 4→12, `repeat_kv` 12→16, five new thin `QLinear` wrappers + three `QEmbedding` wrappers (F-106), verbatim MRoPE grounding in krea/boogu (F-118), `flow_mu` ×3 (F-140). Each is individually trivial; collectively they are how the next Kolors-scheduler-style drift bug gets planted.

7. **Config and doc honesty needs a standing sweep.** The README misstates capability at the front door (F-101), dead config fields that imply nonexistent configurability recur in new crates (F-130, F-142, F-160), and nine doc-contradiction sites regrew after the F-066 cleanup (F-150). The strandline sweep the last review recommended per epic is still worth institutionalizing.

## Coverage notes

- **Reviewed:** all 29 workspace crates' `src/`, `Cargo.toml`s, examples, and test files; `scripts/check-cuda.ps1`, `scripts/package-cuda.ps1`, `scripts/check-gen-core-skew.sh`; `.github/workflows/ci.yml`; root `Cargo.toml` + `rust-toolchain.toml`; `README.md`/`CODEGRAPH.md` accuracy. Eleven parallel deep passes (one per crate group, one infra/cross-cutting) each verified the prior review's findings in its area and performed a fresh six-lens pass, with extra depth on post-2026-07-01 code (bernini, krea edit/control, ComfyUI load-in-place, sequential residency, packed/quant work). The four High findings and the key Bernini/SDXL/flux/boogu/wan claims were independently re-verified against source during synthesis; agents additionally verified load-bearing claims against the pinned candle-core/candle-transformers sources and the pinned gen-core rev (e.g. the sc-11028 offset-view guard was confirmed complete against candle's `to_device`/`force_contiguous` semantics).
- **Vendored fork:** `vendor/candle-kernels` provenance re-verified by full recursive diff against the pinned upstream rev (CRLF-normalized): exactly the two documented deltas. Kernel contents not line-reviewed (unchanged upstream code).
- **Excluded:** `target/` build output; binary test fixtures; the bodies of `#[ignore]`d real-weight GPU parity harnesses (structure, gating, and env handling reviewed; numerics not re-executed).
- **Not performed:** any build/test/clippy execution (read-only review; the repo's CUDA gate is expensive and shared); a `cargo audit`/`cargo deny` scan of `Cargo.lock` (still recommended — F-149); hardware re-reproduction of the F-081 overflow corruptions (thresholds verified analytically against kernel indexing, as in the prior review); mlx-gen twin parity comparison (not in this tree) — noted where a fix must land in lockstep (F-082, F-117); `cargo udeps` whole-graph confirmation of F-104 (grep-verified per crate instead).
- **Known-deliberate patterns excluded by design** (documented decisions, not findings): per-crate trainer `compute_loss_grads` duplication (sc-7787), the dequant-on-forward vs Int8Fast strategy split (explicit `MatmulStrategy` knob, GPU-validated per crate), CPU-seeded noise (sc-3673), composable-op training code (fused ops have no backward), the gen-core SHA-pin lockstep protocol and its long pin-history comment, wan's 512-token pad, and the qwen sequential lane's by-design per-request reloads (sc-10867).
