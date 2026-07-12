//! Krea 2 pose-**ControlNet** trainer — the gen_core [`Trainer`] adapter (sc-10163, epic 10159 B2).
//!
//! The ControlNet Training Studio drives control-branch training through the *same*
//! `gen_core::load_trainer(id, spec).train(req)` path LoRA uses. This registers a
//! [`KreaControlTrainer`] under [`KREA_2_CONTROL_ID`] that:
//!
//! 1. encodes each `(target, control_image, caption)` [`TrainingItem`](gen_core::train::TrainingItem)
//!    into a [`ControlSample`] (VAE latents + caption embedding, CPU-resident),
//! 2. builds a [`ControlBranch`] over the frozen Krea base and drives the reusable [`ControlTrainer`]
//!    (sc-8462) — the proven S0 recipe — mapping its [`TrainEvent`](crate::control_train::TrainEvent)
//!    stream onto [`TrainingProgress`], and
//! 3. writes the trained branch as the output overlay, its `kind` taken from `config.control_type`
//!    (so registration describes it correctly).
//!
//! It reuses the LoRA trainer's encode helpers (`encode_caption`, `load_image_tensor`) so train-time
//! conditioning matches inference. Backend-neutral core ([`ControlTrainer`] knows no gen_core); this
//! file is the thin candle adapter. The MLX twin (epic 10159 B5 / sc-8465) mirrors it.

use std::path::PathBuf;

use candle_gen::candle_core::{DType, Device};
use candle_gen::gen_core::train::{
    Trainer, TrainerDescriptor, TrainingOutput, TrainingProgress, TrainingRequest,
};
use candle_gen::gen_core::{self, LoadSpec, Modality, WeightsSource};
use candle_gen::train::dataset::{bucket_resolution, load_image_tensor};
use candle_gen::train::flow_match;
use candle_gen::{CandleError, Result};

use candle_gen_qwen_image::vae::QwenVaeEncoder;

use crate::config::Krea2Config;
use crate::control::{ControlBranch, DEFAULT_INJECT_OFFSET};
use crate::control_train::{ControlSample, ControlTrainConfig, ControlTrainer};
use crate::loader::Weights;
use crate::pipeline::MAX_TEXT_TOKENS;
use crate::text_encoder::{KreaTeConfig, KreaTextEncoder};
use crate::tokenizer::KreaTokenizer;
use crate::train_dit::KreaTrainDit;
use crate::training::encode_caption;

/// Registry id for the candle Krea pose-ControlNet trainer. Distinct from the LoRA trainer's
/// `krea_2_raw` — this trains a control branch, not a LoRA. The output overlay applies at
/// `krea_2_turbo` inference (the deployed base).
pub const KREA_2_CONTROL_ID: &str = "krea_2_control";

/// The frozen base the produced overlay is applied on (recorded in the overlay meta).
const OVERLAY_BASE_MODEL: &str = "krea_2_turbo";

/// Control-branch depth: copy the first N of 28 single-stream DiT blocks (the S0-proven value).
const N_CONTROL_BLOCKS: usize = 7;

const LABEL: &str = "krea control trainer";

/// Identity + capabilities: `backend = "candle"`, and it is neither a LoRA nor a LoKr trainer.
pub fn control_trainer_descriptor() -> TrainerDescriptor {
    TrainerDescriptor {
        id: KREA_2_CONTROL_ID,
        family: "krea_2",
        backend: "candle",
        modality: Modality::Image,
        supports_lora: false,
        supports_lokr: false,
    }
}

/// A loaded candle Krea control trainer. Loading is **lazy** (no file I/O — the encoders and DiT are
/// built inside [`train`](Trainer::train)), mirroring [`crate::training::KreaTrainer`].
pub struct KreaControlTrainer {
    descriptor: TrainerDescriptor,
    root: PathBuf,
    device: Device,
}

/// Construct the (lazy) control trainer from a [`LoadSpec`] whose `weights` is the Krea snapshot
/// directory (`tokenizer/ text_encoder/ transformer/ vae/`).
pub fn load_control_trainer(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    let root =
        match &spec.weights {
            WeightsSource::Dir(p) => p.clone(),
            WeightsSource::File(_) => return Err(CandleError::Msg(
                "krea_2_control trainer expects a snapshot directory (tokenizer/ text_encoder/ \
                 transformer/ vae/), not a single .safetensors file"
                    .into(),
            )),
        };
    Ok(Box::new(KreaControlTrainer {
        descriptor: control_trainer_descriptor(),
        root,
        device: candle_gen::default_device()?,
    }))
}

// Link-time self-registration into gen-core's trainer registry (kept linked by `crate::force_link`).
candle_gen::register_trainer! { control_trainer_descriptor => load_control_trainer }

impl KreaControlTrainer {
    /// Reject requests this trainer cannot serve: an empty dataset, an item without a control image,
    /// or a request with no `control_type` — the control-specific preconditions the gen_core contract
    /// now carries (a LoRA trainer would ignore these fields).
    fn validate_inner(&self, req: &TrainingRequest) -> Result<()> {
        if req.items.is_empty() {
            return Err(CandleError::Msg(format!("{LABEL}: empty dataset")));
        }
        if req.config.control_type.is_none() {
            return Err(CandleError::Msg(format!(
                "{LABEL}: control training requires config.control_type (e.g. \"pose\")"
            )));
        }
        if let Some(i) = req
            .items
            .iter()
            .position(|it| it.control_image_path.is_none())
        {
            return Err(CandleError::Msg(format!(
                "{LABEL}: item {i} ({}) has no control_image_path — control training needs a \
                 conditioning image per item",
                req.items[i].image_path.display()
            )));
        }
        Ok(())
    }

    fn train_inner(
        &self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> Result<TrainingOutput> {
        on_progress(TrainingProgress::Preparing);
        let device = &self.device;
        let cpu = Device::Cpu;
        let edge = bucket_resolution(req.config.resolution);
        let compute_dtype = flow_match::parse_compute_dtype(&req.config.train_dtype);

        // ── encode (target, control, caption) → CPU-resident ControlSamples ──
        // The VAE encoder + text encoder are loaded only for the cache pass, then dropped before the
        // DiT (the working set) loads. Samples live on the CPU (dataset-size-independent VRAM); the
        // trainer copies each to the device for its micro-step.
        let vae_encoder = QwenVaeEncoder::new(flow_match::component_vb(
            &self.root,
            "vae",
            device,
            DType::F32,
            LABEL,
        )?)?;
        let tokenizer = KreaTokenizer::from_snapshot(&self.root, device)?;
        let te_cfg = KreaTeConfig::from_snapshot(&self.root)?;
        let te_w = Weights::from_dir(&self.root.join("text_encoder"), device, DType::F32)?;
        let text_encoder =
            KreaTextEncoder::load(&te_w, "language_model", &te_cfg, MAX_TEXT_TOKENS)?;

        let total = req.items.len() as u32;
        let mut samples: Vec<ControlSample> = Vec::with_capacity(req.items.len());
        for (i, item) in req.items.iter().enumerate() {
            if req.cancel.is_cancelled() {
                break;
            }
            on_progress(TrainingProgress::Caching {
                current: i as u32 + 1,
                total,
            });
            let control_path = item
                .control_image_path
                .as_ref()
                .expect("validate_inner ensured every item has a control image");
            let target = load_image_tensor(&item.image_path, edge, device)?;
            let control = load_image_tensor(control_path, edge, device)?;
            // Latents stay f32 (the flow-match mix runs f32); the caption stack is stored bf16 (the
            // DiT casts it to bf16 at forward anyway — identical values, half the RAM).
            let x0 = vae_encoder.encode(&target)?.to_device(&cpu)?;
            let ctrl = vae_encoder.encode(&control)?.to_device(&cpu)?;
            let cap = encode_caption(&tokenizer, &text_encoder, &item.caption)?
                .to_dtype(DType::BF16)?
                .to_device(&cpu)?;
            samples.push(ControlSample { x0, ctrl, cap });
        }
        drop(text_encoder);
        drop(vae_encoder);
        if samples.is_empty() {
            // Cancelled before a single item cached — nothing to train.
            return Err(CandleError::Msg(format!(
                "{LABEL}: cancelled during caching"
            )));
        }

        // ── frozen base DiT + trainable control branch ──
        on_progress(TrainingProgress::LoadingModel);
        let dit_cfg = Krea2Config::from_snapshot(&self.root)?;
        let dit_w = Weights::from_dir(&self.root.join("transformer"), device, compute_dtype)?;
        let dit = KreaTrainDit::load(&dit_w, &dit_cfg)?;
        let branch = ControlBranch::from_base(
            &dit_w,
            &dit_cfg,
            N_CONTROL_BLOCKS,
            compute_dtype,
            DEFAULT_INJECT_OFFSET,
        )?;
        drop(dit_w);

        // ── map the gen_core config → ControlTrainConfig ──
        // gen_core `steps` = total micro-steps; an optimizer update fires every `gradient_accumulation`
        // of them, so the trainer's update budget = steps / accumulation.
        let batch = req.config.gradient_accumulation.max(1);
        let max_steps = (req.config.steps / batch).max(1);
        let ctrl_cfg = ControlTrainConfig {
            lr: req.config.learning_rate,
            batch,
            max_steps,
            warmup_steps: req.config.lr_warmup_steps,
            timestep_type: req.config.timestep_type.clone(),
            seed: req.config.seed,
            grad_checkpoint: req.config.gradient_checkpointing,
            mae: flow_match::is_mae(&req.config),
            compute_dtype,
            save_every: req.config.save_every,
            resolution: edge,
            control_type: req.config.control_type.clone(),
            ..ControlTrainConfig::default()
        };

        let mut trainer = ControlTrainer::new(
            dit,
            branch,
            samples,
            ctrl_cfg,
            req.output_dir.clone(),
            0,
            device.clone(),
        )?;

        // ── train: drive the loop via the public single-step API so we own cancel + progress mapping
        //    (the neutral ControlTrainer stays gen_core-agnostic). ──
        let mut last_loss = f32::NAN;
        let mut ran_updates = 0u32;
        for _ in 0..max_steps {
            if req.cancel.is_cancelled() {
                break;
            }
            let rep = trainer.step()?;
            if !rep.loss.is_finite() {
                return Err(CandleError::Msg(format!(
                    "{LABEL}: non-finite loss at update {}",
                    rep.step
                )));
            }
            last_loss = rep.loss;
            ran_updates = rep.step;
            on_progress(TrainingProgress::Training {
                step: rep.step,
                total: max_steps,
                loss: rep.loss,
            });
            if req.config.save_every > 0 && rep.step.is_multiple_of(req.config.save_every) {
                trainer.save_checkpoint(rep.step)?;
                on_progress(TrainingProgress::Checkpoint { step: rep.step });
            }
        }

        // ── save the final overlay to output_dir/file_name ──
        on_progress(TrainingProgress::Saving);
        let adapter_path = req.output_dir.join(&req.file_name);
        trainer.save_overlay(&adapter_path, OVERLAY_BASE_MODEL)?;
        Ok(TrainingOutput {
            adapter_path,
            // gen_core counts micro-steps; the trainer counts optimizer updates of `batch` each.
            steps: ran_updates * batch,
            final_loss: last_loss,
        })
    }
}

impl Trainer for KreaControlTrainer {
    fn descriptor(&self) -> &TrainerDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &TrainingRequest) -> gen_core::Result<()> {
        self.validate_inner(req).map_err(Into::into)
    }

    fn train(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> gen_core::Result<TrainingOutput> {
        self.validate_inner(req)?;
        self.train_inner(req, on_progress).map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::registry;
    use candle_gen::gen_core::runtime::CancelFlag;
    use candle_gen::gen_core::train::{TrainingConfig, TrainingItem};

    /// The control trainer self-registers and resolves through gen-core's registry as the candle Krea
    /// control trainer (distinct id from the LoRA `krea_2_raw`); `load` is lazy so a nonexistent dir
    /// still resolves.
    #[test]
    fn control_trainer_registers_as_candle() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let t = registry::load_trainer(KREA_2_CONTROL_ID, &spec)
            .expect("candle krea control trainer is registered");
        assert_eq!(t.descriptor().id, KREA_2_CONTROL_ID);
        assert_eq!(t.descriptor().family, "krea_2");
        assert_eq!(t.descriptor().backend, "candle");
        assert!(
            !t.descriptor().supports_lora,
            "control trainer is not a LoRA trainer"
        );
        assert!(!t.descriptor().supports_lokr);
    }

    /// `validate` enforces the control-specific preconditions the LoRA path lacks: a non-empty
    /// dataset, a `control_type`, and a control image on every item.
    #[test]
    fn validate_requires_control_image_and_type() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let t = registry::load_trainer(KREA_2_CONTROL_ID, &spec).unwrap();

        let ok = TrainingRequest {
            items: vec![TrainingItem::with_control(
                "/img.png".into(),
                "x".into(),
                "/pose.png".into(),
            )],
            config: TrainingConfig {
                control_type: Some("pose".into()),
                ..Default::default()
            },
            output_dir: "/out".into(),
            file_name: "a.safetensors".into(),
            trigger_words: vec![],
            cancel: CancelFlag::new(),
        };
        assert!(
            t.validate(&ok).is_ok(),
            "a well-formed control request validates"
        );

        let bad = |mutate: &dyn Fn(&mut TrainingRequest)| {
            let mut r = ok.clone();
            mutate(&mut r);
            assert!(t.validate(&r).is_err());
        };
        bad(&|r| r.items.clear());
        bad(&|r| r.config.control_type = None);
        bad(&|r| r.items = vec![TrainingItem::captioned("/img.png".into(), "x".into())]);
    }
}
