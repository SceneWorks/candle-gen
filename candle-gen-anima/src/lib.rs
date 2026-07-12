//! # candle-gen-anima
//!
//! The **Anima** provider crate for [`candle-gen`](candle_gen) — the candle (Windows/CUDA) sibling of
//! `mlx-gen-anima` (epic 10512, sc-10525). Three variants share **one architecture** and differ only
//! in the DiT weights file:
//! - **`anima_base`** — the base model (30 steps, CFG 4.5),
//! - **`anima_aesthetic`** — the aesthetic fine-tune (30 steps, CFG 4.5),
//! - **`anima_turbo`** — the merged CFG-free few-step student (10 steps, CFG 1.0).
//!
//! ## Architecture (transcribed from the MLX port; no candle-transformers reference)
//! - **DiT** — the **Cosmos-Predict2** `CosmosTransformer3DModel` (28 layers, hidden 2048 = 16×128,
//!   patch `(1,2,2)`, adaLN-LoRA 256, 3-axis NTK RoPE `rope_scale (1,4,4)`, `concat_padding_mask` ⇒
//!   **17-channel** patch-embed input) — [`transformer::CosmosDiT`].
//! - **Text conditioner** — the **`AnimaTextConditioner`** (bundled under `{prefix}.llm_adapter.*`):
//!   `nn.Embedding(32128, 1024)` over T5 ids → 6 × [self-attn → cross-attn into Qwen3 states → GELU
//!   MLP] → out_proj + RMSNorm, right-padded to **512** — [`conditioner::AnimaTextConditioner`].
//! - **Text encoder** — **Qwen3-0.6B base** (`last_hidden_state`, GQA 16/8 handled with `repeat_kv`) —
//!   [`text_encoder::AnimaQwen3`].
//! - **VAE** — the **Qwen-Image** `AutoencoderKLQwenImage`, reusing [`vae::QwenVae`] (from
//!   `candle-gen-qwen-image`) via the original→diffusers key rename [`vae::convert_vae_key`].
//! - **Scheduler** — `FlowMatchEulerDiscreteScheduler` static `shift=3.0`, `sigmas = linspace(1, 1/N, N)`
//!   ([`pipeline::anima_sigmas`]); default solver the recommended `er_sde` (sc-10519), carried by the
//!   `441ecec` gen-core pin ([`pipeline::DEFAULT_SAMPLER`]).
//!
//! **`backend = "candle"`, `mac_only = false`** — this crate is what lets the manifest drop the
//! `macOnly: true` gate the MLX-only port carried (sc-10523 wires it worker-side).
//!
//! **Surface:** txt2img at the single-file dense checkpoint, with **LoRA/LoKr injection** (448 DiT + 60
//! conditioner targets folded at load, stacked + mixed, strict routing — [`adapters`]). Q4/Q8 candle
//! quant tiers are the counterpart of MLX sc-10517 (see the `quant` gap note in [`loader`]).

pub mod adapt;
pub mod adapters;
pub mod conditioner;
pub mod config;
pub mod loader;
pub mod nn;
pub mod pipeline;
pub mod rope;
pub mod text_encoder;
pub mod tokenizer;
pub mod transformer;
pub mod vae;

pub use conditioner::AnimaTextConditioner;
pub use config::{ConditionerConfig, DitConfig, Qwen3Config, Variant};
pub use loader::{detect_dit_prefix, AnimaComponents};
pub use pipeline::{anima_sigmas, AnimaPipeline, GenOptions, DEFAULT_SAMPLER};
pub use text_encoder::AnimaQwen3;
pub use transformer::CosmosDiT;
pub use vae::{load_vae, QwenVae};

use std::sync::{Arc, Mutex};

use candle_gen::candle_core::Device;
use candle_gen::gen_core::{
    self, Capabilities, GenerationOutput, GenerationRequest, Generator, LoadSpec, Modality,
    ModelDescriptor, Progress, Quant, WeightsSource,
};

use crate::config::RES_MULTIPLE;

/// The candle quant tiers Anima advertises — Q4 + Q8 (the counterpart of MLX sc-10517). The DiT loads
/// packed (dequant-dense per step, CPU-capable); the conditioner / Qwen3 TE / VAE stay dense bf16.
const ANIMA_QUANTS: &[Quant] = &[Quant::Q4, Quant::Q8];

const MAX_COUNT: u32 = 8;
const RES_MIN: u32 = 512;
/// Above ~1920 px/side the Cosmos RoPE would index out of its trained range; `rope.rs` **rejects**
/// such a request rather than clamping. The shipped ceiling is 1536² (post-patch 96 < the 120-position
/// max_size), so the guard is unreachable via the normal path. See [`crate::rope`].
const RES_MAX: u32 = 1536;

/// Build the descriptor for a variant. Turbo is the merged CFG-free student (no guidance / negative
/// prompt); Base/Aesthetic run true classifier-free guidance.
fn descriptor_for(variant: Variant) -> ModelDescriptor {
    let cfg_capable = variant.uses_cfg();
    ModelDescriptor {
        id: variant.id(),
        family: "anima",
        backend: "candle",
        modality: Modality::Image,
        capabilities: Capabilities {
            supports_negative_prompt: cfg_capable,
            supports_guidance: cfg_capable,
            supports_true_cfg: false,
            conditioning: vec![],
            // LoRA/LoKr injection is wired (the candle counterpart of MLX sc-10521): every trained
            // adapter's 448 DiT + 60 conditioner targets fold at load, stacked + mixed, strict routing
            // (`adapters::apply_anima_adapters`). Weight-level fold, validated bit-exact on CPU.
            supports_lora: true,
            supports_lokr: true,
            // Rectified-flow over the unified curated-sampler framework (epic 7114). The native default
            // (req.sampler == None) is the recommended er_sde solver; the full curated menu is advertised.
            samplers: candle_gen::curated_sampler_names(),
            schedulers: candle_gen::curated_scheduler_names(),
            supported_guidance_methods: vec![],
            min_size: RES_MIN,
            max_size: RES_MAX,
            max_count: MAX_COUNT,
            // The whole point of the candle port: Anima is no longer Mac-only.
            mac_only: false,
            // Q4 + Q8 (the candle counterpart of MLX sc-10517): the DiT packed-detects and runs the
            // dequant-dense forward (CPU-capable — NOT the CUDA-only int8 fast GEMM); conditioner /
            // Qwen3 TE / VAE stay dense bf16. A pre-packed tier is a real, loadable snapshot.
            supported_quants: ANIMA_QUANTS,
            supports_kv_cache: false,
            requires_sigma_shift: true,
        },
    }
}

pub fn descriptor_base() -> ModelDescriptor {
    descriptor_for(Variant::Base)
}
pub fn descriptor_aesthetic() -> ModelDescriptor {
    descriptor_for(Variant::Aesthetic)
}
pub fn descriptor_turbo() -> ModelDescriptor {
    descriptor_for(Variant::Turbo)
}

/// A loaded Anima generator: the cached descriptor + variant + lazily-built pipeline (mirrors the
/// candle-gen-qwen-image lazy component cache).
pub struct Anima {
    descriptor: ModelDescriptor,
    variant: Variant,
    root: WeightsSource,
    device: Device,
    /// LoRA/LoKr adapters to bake onto the DiT + conditioner at pipeline build (empty for the plain
    /// model). Captured at load; folded lazily when the pipeline is first assembled.
    adapters: Vec<gen_core::AdapterSpec>,
    /// Lazily-built, shared pipeline behind the shared read-through cache ([`candle_gen::cached`],
    /// sc-7792). The slot holds an `Arc<AnimaPipeline>` — cheap to clone — so `pipeline()` can return
    /// an owned handle and **release the cache lock before the denoise** (see [`Anima::pipeline`]),
    /// matching every sibling candle-gen provider. Before sc-10608 this was a bespoke
    /// `Mutex<Option<AnimaPipeline>>` whose guard was held *across* the whole generation.
    pipeline: Mutex<Option<Arc<AnimaPipeline>>>,
}

pub fn load_base(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_variant(spec, Variant::Base)
}
pub fn load_aesthetic(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_variant(spec, Variant::Aesthetic)
}
pub fn load_turbo(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_variant(spec, Variant::Turbo)
}

fn load_variant(spec: &LoadSpec, variant: Variant) -> gen_core::Result<Box<dyn Generator>> {
    let id = variant.id();
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "{id}: candle Anima is txt2img only (no control / IP-adapter)"
        )));
    }
    // A Q4/Q8 tier is a **pre-packed** snapshot (the worker points `spec.weights` at it; the DiT
    // packed-detects it at load). LoRA/LoKr on a packed tier is now wired (sc-10640): the loader builds
    // the packed model and installs each adapter as a **forward-time residual** (`y = xW_packed +
    // scale·(xA)B`, epic 10043 prior art) rather than folding into the codes — so no rejection here. The
    // dense-checkpoint fold and the packed residual are both handled in `loader::AnimaComponents::load`
    // (a LoKr/LoHa on a packed tier still errors, in the loader — sc-10713).
    //
    // A requested Q4/Q8 tier MUST be an actually-packed checkpoint (u32 codes + `.scales`). Anima ships
    // no packed tier yet, and `lin()` packed-detects PER-TENSOR — so a `quantize = Q8` request against a
    // DENSE DiT would silently build bf16 and return success (a tier downgrade the caller never sees).
    // Assert the DiT is packed; otherwise reject naming the requested tier and what was found. Same
    // runtime-lie class as sc-10515 (advertising a tier the load can't honor).
    if let Some(q) = spec.quantize {
        if !loader::dit_is_packed(&spec.weights, variant).map_err(gen_core::Error::from)? {
            return Err(gen_core::Error::Unsupported(format!(
                "{id}: {q:?} tier requested but the DiT checkpoint is DENSE (no packed `.scales` \
                 tensors) — Anima ships no packed Q4/Q8 tier yet; load the dense tier (no quantize)"
            )));
        }
    }
    // LoRA/LoKr adapters (`spec.adapters`) are accepted — folded onto the DiT + conditioner when the
    // pipeline is assembled (`adapters::apply_anima_adapters`).
    let device = candle_gen::default_device().map_err(gen_core::Error::from)?;
    Ok(Box::new(Anima {
        descriptor: descriptor_for(variant),
        variant,
        root: spec.weights.clone(),
        device,
        adapters: spec.adapters.clone(),
        pipeline: Mutex::new(None),
    }))
}

impl Anima {
    /// Lazily assemble (and cache) the pipeline on first `generate`, returning a **shared owned
    /// handle**.
    ///
    /// ## Concurrency decision (sc-10608)
    ///
    /// This adopts the shared [`candle_gen::cached`] read-through cache (sc-7792) that every sibling
    /// candle-gen provider uses. The consequence — the whole point of the story — is a deliberate
    /// change to the lock/denoise relationship:
    ///
    /// - **Before:** a bespoke `Mutex<Option<AnimaPipeline>>` whose guard was held **across the entire
    ///   denoise**. A second `generate` on the same loaded model blocked on the cache mutex until the
    ///   first finished all its steps — concurrent generation was serialized *by accident*.
    /// - **Now:** `cached()` holds the lock only across the (idempotent) build and a cheap `Arc` clone,
    ///   then drops the guard. `generate` runs the denoise on the returned `Arc<AnimaPipeline>`
    ///   **outside the lock**, so a second caller only waits to clone the `Arc`, not for the first
    ///   generation to finish.
    ///
    /// Running the denoise outside the lock is **safe** here — this is verified deliberately, not
    /// inherited from the siblings:
    /// 1. `AnimaPipeline` / `AnimaComponents` are stateless forwards over **immutable, `Arc`-backed
    ///    candle weights** — every `forward`/`encode`/`decode` in the generate path takes `&self`. The
    ///    only `&mut self` methods (`visit_adaptable_mut` adapter installers) run at load, never during
    ///    generation. No `RefCell`/`Cell`/scratch buffer is mutated per step, and each `generate` draws
    ///    its own noise + tensors. There is therefore no shared mutable state for two concurrent
    ///    generations to race on.
    /// 2. `Arc<AnimaPipeline>: Send + Sync` is compiler-enforced: the cache is `Mutex<Option<Arc<…>>>`,
    ///    and `Anima` must be `Sync` to satisfy the `Generator` bound, which requires the pipeline to be
    ///    `Send + Sync` (pinned by `pipeline_handle_is_send_and_sync`).
    /// 3. The candle `Device` internally synchronizes GPU submission, so concurrent forwards are
    ///    serialized at the driver — safe, never corrupting.
    ///
    /// The lock-release contract is pinned by `cache_lock_is_released_before_generation` below;
    /// returning an owned `Arc` (not a `MutexGuard`) makes "hold the lock across the denoise"
    /// structurally impossible to reintroduce without changing this signature.
    fn pipeline(&self) -> gen_core::Result<Arc<AnimaPipeline>> {
        Ok(candle_gen::cached(&self.pipeline, || {
            AnimaPipeline::from_source(&self.root, self.variant, &self.device, &self.adapters)
                .map(Arc::new)
        })?)
    }
}

impl Generator for Anima {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        validate_request(&self.descriptor, req)
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        validate_request(&self.descriptor, req)?;
        // Build-or-clone the shared pipeline, then run the denoise on the returned `Arc` **outside**
        // the cache lock (sc-10608 — see `Anima::pipeline` for the concurrency decision).
        let pipeline = self.pipeline()?;

        let base_seed = req.seed.unwrap_or_else(gen_core::default_seed);
        let steps = req.steps.unwrap_or(self.variant.default_steps()) as usize;
        let guidance = if self.variant.uses_cfg() {
            req.guidance.unwrap_or(self.variant.default_guidance())
        } else {
            1.0
        };
        let negative = req.negative_prompt.clone().unwrap_or_default();

        // Shared batch frame (sc-7792): the `0..count` loop + per-image `image_seed(base_seed, n)`
        // derivation + `Vec` collect that every provider repeats. The model body stays hand-written in
        // the closure (captures `on_progress` + the borrowed pipeline).
        let images = candle_gen::for_each_image_seed(base_seed, req.count, |seed| {
            let opts = GenOptions {
                width: req.width,
                height: req.height,
                steps,
                guidance,
                seed,
                sampler: req.sampler.clone(),
            };
            pipeline.generate(
                &req.prompt,
                &negative,
                self.variant,
                &opts,
                &req.cancel,
                on_progress,
            )
        })?;
        Ok(GenerationOutput::Images(images))
    }
}

/// Capability-driven request validation (testable without loaded weights): non-empty prompt, size a
/// multiple of 16, on top of the shared [`Capabilities::validate_request`] floor.
pub(crate) fn validate_request(
    desc: &ModelDescriptor,
    req: &GenerationRequest,
) -> gen_core::Result<()> {
    let id = desc.id;
    if req.prompt.is_empty() {
        return Err(gen_core::Error::Msg(format!(
            "{id}: prompt must not be empty"
        )));
    }
    // Reject an explicit `steps: Some(0)` loudly: `anima_sigmas` clamps `steps.max(1)`, so a 0 silently
    // becomes a single-step render rather than the fast typed error its sibling bespoke lanes give
    // (`reject_zero_steps`, sc-9016, F-032; swept here by sc-11182, F-102). A `None` legitimately falls
    // through to `variant.default_steps()`.
    if req.steps == Some(0) {
        return Err(gen_core::Error::Msg(format!(
            "{id}: steps must be >= 1 (an explicit 0 renders a single step of undenoised noise)"
        )));
    }
    desc.capabilities.validate_request(id, req)?;
    if !req.width.is_multiple_of(RES_MULTIPLE) || !req.height.is_multiple_of(RES_MULTIPLE) {
        return Err(gen_core::Error::Msg(format!(
            "{id}: {}x{} must be a multiple of {RES_MULTIPLE}",
            req.width, req.height
        )));
    }
    Ok(())
}

// Link-time registration of all three variants.
candle_gen::register_generators! {
    descriptor_base => load_base,
    descriptor_aesthetic => load_aesthetic,
    descriptor_turbo => load_turbo,
}

/// Force-link hook (keeps the `inventory::submit!` registrations from being dead-stripped).
pub fn force_link() {}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::registry;

    fn req(w: u32, h: u32) -> GenerationRequest {
        GenerationRequest {
            prompt: "an anime girl with silver hair, detailed".into(),
            width: w,
            height: h,
            ..Default::default()
        }
    }

    #[test]
    fn three_variants_registered_as_candle() {
        for id in ["anima_base", "anima_aesthetic", "anima_turbo"] {
            let g = registry::load(
                id,
                &LoadSpec::new(WeightsSource::Dir("/nonexistent".into())),
            )
            .unwrap_or_else(|_| panic!("id {id} not registered"));
            assert_eq!(g.descriptor().id, id);
            assert_eq!(g.descriptor().family, "anima");
            assert_eq!(g.descriptor().backend, "candle");
        }
    }

    #[test]
    fn descriptors_surface() {
        let b = descriptor_base();
        assert_eq!(b.id, "anima_base");
        assert_eq!(b.backend, "candle");
        assert_eq!(b.modality, Modality::Image);
        assert!(b.capabilities.supports_guidance);
        assert!(b.capabilities.supports_negative_prompt);
        assert!(b.capabilities.requires_sigma_shift);
        // The candle port removes the Mac-only gate.
        assert!(!b.capabilities.mac_only);
        // LoRA/LoKr injection is wired; Q4/Q8 tiers are advertised (packed-detect, dequant-dense).
        assert!(b.capabilities.supports_lora && b.capabilities.supports_lokr);
        assert_eq!(b.capabilities.supported_quants, &[Quant::Q4, Quant::Q8]);
        assert_eq!(b.capabilities.min_size, 512);
        assert_eq!(b.capabilities.max_size, 1536);
        // Turbo is the CFG-free merged student.
        let t = descriptor_turbo();
        assert!(!t.capabilities.supports_guidance);
        assert!(!t.capabilities.supports_negative_prompt);
        // The default flow solver (er_sde on the 441ecec gen-core pin) is a real curated sampler.
        assert_eq!(pipeline::DEFAULT_SAMPLER, "er_sde");
        assert!(
            b.capabilities.samplers.contains(&pipeline::DEFAULT_SAMPLER),
            "er_sde advertised in the curated menu (441ecec gen-core pin carries the ErSde solver)"
        );
    }

    #[test]
    fn validate_rejects_bad_requests() {
        assert!(validate_request(&descriptor_base(), &GenerationRequest::default()).is_err()); // empty prompt
        assert!(validate_request(&descriptor_base(), &req(1000, 1024)).is_err()); // not mult of 16
        assert!(validate_request(&descriptor_base(), &req(256, 256)).is_err()); // below min
        assert!(validate_request(&descriptor_base(), &req(2048, 2048)).is_err()); // above max
                                                                                  // Explicit `steps: Some(0)` is rejected (sc-11182, F-102) — it would otherwise clamp to a
                                                                                  // silent 1-step render in `anima_sigmas`; `None` (the default) is fine.
        let zero_steps = GenerationRequest {
            steps: Some(0),
            ..req(1024, 1024)
        };
        let err = validate_request(&descriptor_base(), &zero_steps).unwrap_err();
        assert!(err.to_string().contains("steps must be >= 1"), "{err}");
        assert!(validate_request(&descriptor_base(), &req(1024, 1024)).is_ok());
        assert!(validate_request(&descriptor_base(), &req(1536, 1536)).is_ok());
    }

    /// Write a minimal **dense** DiT split_files layout (one anchor tensor, NO `.scales` codes) so the
    /// quant-guard can header-detect it as dense. Returns the split_files root.
    fn write_dense_split_files() -> std::path::PathBuf {
        use candle_gen::candle_core::{DType, Device, Tensor};
        let root = std::env::temp_dir().join(format!("anima_quant_guard_{}", std::process::id()));
        let dm = root.join("diffusion_models");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&dm).unwrap();
        let mut m = std::collections::HashMap::new();
        m.insert(
            "net.x_embedder.proj.1.weight".to_string(),
            Tensor::zeros((2, 2), DType::F32, &Device::Cpu).unwrap(),
        );
        candle_gen::candle_core::safetensors::save(&m, dm.join(Variant::Base.dit_filename()))
            .unwrap();
        root
    }

    /// Write a minimal **packed** DiT split_files layout (an anchor tensor WITH a `.scales`/`.biases`
    /// sibling) so the quant-guard header-detects it as packed. Returns the split_files root.
    fn write_packed_split_files() -> std::path::PathBuf {
        use candle_gen::candle_core::{DType, Device, Tensor};
        let root = std::env::temp_dir().join(format!("anima_packed_guard_{}", std::process::id()));
        let dm = root.join("diffusion_models");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&dm).unwrap();
        let mut m = std::collections::HashMap::new();
        // The anchor `.weight` (u32 codes) + `.scales`/`.biases` — enough for the header-only packed
        // detect (`dit_path_is_packed` looks only for a `.scales` sibling).
        m.insert(
            "net.x_embedder.proj.1.weight".to_string(),
            Tensor::zeros((2, 2), DType::U32, &Device::Cpu).unwrap(),
        );
        m.insert(
            "net.x_embedder.proj.1.scales".to_string(),
            Tensor::zeros((2, 1), DType::F32, &Device::Cpu).unwrap(),
        );
        m.insert(
            "net.x_embedder.proj.1.biases".to_string(),
            Tensor::zeros((2, 1), DType::F32, &Device::Cpu).unwrap(),
        );
        candle_gen::candle_core::safetensors::save(&m, dm.join(Variant::Base.dit_filename()))
            .unwrap();
        root
    }

    #[test]
    fn load_accepts_lora_and_packed_combo_but_rejects_quant_on_dense() {
        use candle_gen::gen_core::{AdapterKind, AdapterSpec};
        let root = write_dense_split_files();
        let base = WeightsSource::Dir(root.clone());
        let lora_spec = || {
            vec![AdapterSpec::new(
                "/lora.safetensors".into(),
                1.0,
                AdapterKind::Lora,
            )]
        };

        // A plain LoRA load SUCCEEDS (lazily built; the fold happens at first generate). Advertising the
        // capability and then rejecting at load would be a lie.
        assert!(load_base(&LoadSpec::new(base.clone()).with_adapters(lora_spec())).is_ok());

        // A Q4/Q8 request against a DENSE checkpoint must be REJECTED at load, not silently downgraded to
        // bf16 and returned Ok (the sc-10525 blocker: a tier the load can't honor). The message names the
        // requested tier and that the DiT is dense.
        for q in [Quant::Q4, Quant::Q8] {
            let err = load_base(&LoadSpec::new(base.clone()).with_quant(q))
                .err()
                .expect("Q-tier on a dense DiT must error");
            let gen_core::Error::Unsupported(msg) = &err else {
                panic!("expected Unsupported, got {err:?}");
            };
            assert!(
                msg.contains(&format!("{q:?}")) && msg.contains("DENSE"),
                "message must name the tier + dense: {msg}"
            );
        }

        // Q8 + LoRA against a DENSE checkpoint still errors — but for the **tier-mismatch** reason (Q8 on
        // a dense DiT), NOT a packed+adapter combo rejection. That combo rejection was REMOVED in sc-10640
        // (the combo is now wired via forward-time residuals); the guard that fires here is the same dense
        // tier-mismatch as the no-adapter case above, so the message names DENSE.
        let dense_combo = LoadSpec::new(base.clone())
            .with_quant(Quant::Q8)
            .with_adapters(lora_spec());
        let gen_core::Error::Unsupported(msg) = load_base(&dense_combo).err().expect("err") else {
            panic!("expected Unsupported");
        };
        assert!(
            msg.contains("DENSE"),
            "dense-tier mismatch (Q8 on dense), not a packed-combo rejection: {msg}"
        );

        // sc-10640: Q4/Q8 + LoRA on a **packed** checkpoint is now ACCEPTED at load (built lazily; the
        // residual install runs at first generate). This is exactly the combo that used to be rejected.
        let packed_root = write_packed_split_files();
        let packed = WeightsSource::Dir(packed_root.clone());
        assert!(
            load_base(
                &LoadSpec::new(packed)
                    .with_quant(Quant::Q8)
                    .with_adapters(lora_spec())
            )
            .is_ok(),
            "packed tier + LoRA must be accepted at load (sc-10640) — no combo rejection"
        );

        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&packed_root);
    }

    /// The pipeline handle must be `Send + Sync` so the denoise can run outside the cache lock
    /// (sc-10608). `Anima` is `Sync` (required by the `Generator` bound), its cache is
    /// `Mutex<Option<Arc<AnimaPipeline>>>`, and `Mutex<Option<Arc<T>>>` is only `Sync` when
    /// `T: Send + Sync`. A change that made `AnimaPipeline` non-`Sync` would break `Anima: Sync` — this
    /// static assertion fails to compile first, with a clear pointer to the reason.
    #[test]
    fn pipeline_handle_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Arc<AnimaPipeline>>();
        assert_send_sync::<Anima>();
    }

    /// Pins the concurrency decision of sc-10608: the pipeline cache lock is **released before the
    /// denoise**, so a second request can obtain the pipeline while the first is mid-generation. Before
    /// sc-10608, `Anima` held the cache guard **across** the whole generation, serializing concurrent
    /// callers by accident; adopting [`candle_gen::cached`] (which returns an owned `Arc` and drops the
    /// guard) removes that.
    ///
    /// This drives the exact mechanism `Anima::pipeline` now uses — `candle_gen::cached()` over a
    /// `Mutex<Option<Arc<T>>>` — with a stand-in payload, because building a real `AnimaPipeline` needs
    /// full weights (exercised on the gated real-weights lane, not this CPU unit test). Returning an
    /// owned `Arc` (not a `MutexGuard`) is what makes the pre-sc-10608 "hold the lock across the
    /// denoise" shape structurally impossible to reintroduce without changing `pipeline()`'s signature.
    #[test]
    fn cache_lock_is_released_before_generation() {
        use std::sync::mpsc;
        use std::sync::{Arc, Barrier, Mutex};
        use std::time::Duration;

        // The shared read-through cache anima uses (`Mutex<Option<Arc<T>>>`); `u32` stands in for
        // `Arc<AnimaPipeline>`.
        let slot: Arc<Mutex<Option<Arc<u32>>>> = Arc::new(Mutex::new(None));

        let in_gen = Arc::new(Barrier::new(2)); // A now holds its handle and is "mid-generation"
        let end_gen = Arc::new(Barrier::new(2)); // A may finish only after B acquired concurrently
        let (slot_a, in_a, end_a) = (Arc::clone(&slot), Arc::clone(&in_gen), Arc::clone(&end_gen));

        let a = std::thread::spawn(move || {
            // Build-or-clone via the shared accessor path, then hold ONLY the returned `Arc` across the
            // "denoise" — never the cache guard (the sc-10608 contract).
            let handle = candle_gen::cached(&slot_a, || Ok::<_, ()>(Arc::new(7u32))).unwrap();
            in_a.wait();
            end_a.wait(); // if B were blocked on the cache lock, this rendezvous would never complete
            *handle
        });

        let (tx, rx) = mpsc::channel();
        let (slot_b, in_b, end_b) = (Arc::clone(&slot), Arc::clone(&in_gen), Arc::clone(&end_gen));
        std::thread::spawn(move || {
            in_b.wait(); // A is holding its pipeline handle right now.
                         // A second request must obtain the pipeline WHILE A is mid-generation. The
                         // pre-sc-10608 "guard held across the denoise" shape would block this until A
                         // finished; `cached()` returns immediately (and hits the cache — the panic
                         // asserts no double-build).
            let handle = candle_gen::cached(&slot_b, || -> Result<Arc<u32>, ()> {
                panic!("cache hit expected, not a rebuild")
            })
            .unwrap();
            let _ = tx.send(*handle);
            end_b.wait();
        });

        // B must acquire the pipeline promptly while A is still mid-generation — not after A releases.
        // A held-across-denoise regression would make B block and this `recv_timeout` fire.
        let got = rx.recv_timeout(Duration::from_secs(2)).expect(
            "second caller must obtain the pipeline while the first is mid-generation — the cache lock \
             must NOT be held across the denoise (sc-10608)",
        );
        assert_eq!(got, 7, "second caller got the cached pipeline, no rebuild");
        assert_eq!(a.join().unwrap(), 7);
        assert!(
            slot.try_lock().is_ok(),
            "cache lock is free after generation, never wedged across a denoise"
        );
    }
}
