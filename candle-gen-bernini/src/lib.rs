//! # candle-gen-bernini
//!
//! ByteDance **Bernini** renderer — the candle (Windows/CUDA + Linux/NVIDIA) sibling of the
//! `mlx-gen-bernini` renderer (sc-4706), part 1 of epic 6562 (the epic splits the Bernini provider into
//! renderer-first then planner; the code seam is that the crate registers the renderer engine now and
//! the full pipeline later).
//!
//! The Bernini renderer **is** Wan2.2-T2V-A14B, finetuned: a **dual-expert MoE** (two complete
//! `WanTransformer3DModel` checkpoints — a high-noise `transformer/` and a low-noise `transformer_2/`)
//! driven by a single flow-match scheduler that picks the high expert while the integer timestep is
//! `≥ switch_dit_boundary·1000` and the low expert below it. It reuses the [`candle_gen_wan`] foundation
//! wholesale — the z16 VAE ([`candle_gen_wan::vae16::WanVae16`]), the UMT5 text encoder
//! ([`candle_gen_wan::text_encoder::Umt5Encoder`]), the dual-expert
//! [`WanTransformer`](candle_gen_wan::transformer::WanTransformer) with its sc-10025 packed-detect Q4/Q8
//! load, the flow/UniPC scheduler ([`candle_gen_wan::scheduler`]), the 3-axis RoPE
//! ([`candle_gen_wan::rope::WanRope`]), and the noise/frames glue — with two Bernini deltas layered on:
//!
//!   1. **The Bernini knobs** ([`config::BerniniKnobs`], from the `bernini_renderer.json` sidecar): the
//!      expert-switch boundary (0.875) and the UniPC flow shift (3.0).
//!   2. **APG guidance** ([`guidance`]): the caption-only render defaults to `t2v_apg` — Adaptive
//!      Projected Guidance in x-space (the reference's `resolve_mode(None,false,false)`), with plain CFG
//!      (`t2v`) selectable via `video_mode="t2v"`. On the first low-noise step all omegas are scaled once
//!      by `OMEGA_SCALE`.
//!
//! ## Status / scope
//! **sc-10994 (renderer):** the **caption→pixel renderer** (t2v / t2v_apg), Q4/Q8 packed streaming load
//! of the two experts, and the candle turnkey tier [`convert`]er are implemented; `bernini_renderer`
//! self-registers.
//!
//! **sc-10995 (planner slice):** the framework-independent planner seams are ported + CPU-golden
//! parity-tested against the same fixtures the MLX lane asserts (`tests/*_parity.rs`): the Qwen2.5-VL
//! text backbone ([`qwen2_5_vl::Qwen25VlText`] — MRoPE table + penultimate hidden state), the MRoPE
//! host-side input shaping ([`process`] — `generate_unified_inputs` / position ids / flex mask), the
//! [`connector::MlpConnector`] (`for_gen`/`for_vit`), the planner→renderer [`mar`] handoff
//! (`four_streams` / `post_process_input_embeds` / `mar_schedule`), and the ViT-conditioned
//! [`vit_guidance`] combine modes. These are **library modules** — the full `bernini` end-to-end
//! generator (the MAR sampling loop `mar::sample_vit_embed`, the vision tower, the clip-diff
//! flow-matching head, the packed-conditioning renderer forward, and the `bernini` registration) is the
//! precise follow-up beyond this slice; nothing here fakes an end-to-end pipeline.
//!
//! `backend = "candle"`, `mac_only = false`.

pub mod config;
pub mod connector;
pub mod convert;
pub mod guidance;
pub mod mar;
mod nn;
pub mod pipeline;
pub mod process;
pub mod qwen2_5_vl;
pub mod vit_guidance;

pub use config::{resolve_mode, BerniniKnobs, Defaults, Mode};
pub use connector::MlpConnector;
pub use convert::{build_bernini_candle_tier, route_bernini_expert_key};
pub use guidance::{apg_delta, normalized_guidance, normalized_guidance_chain, MomentumBuffer};
pub use mar::{
    feat_to_renderer, four_streams, mar_schedule, post_process_input_embeds, FourStreams,
    RendererFeat,
};
pub use pipeline::{descriptor, force_link, load, MODEL_ID};
pub use process::{
    build_attention_mask_4d, generate_unified_inputs, mrope_position_ids, MRopeConfig,
};
pub use qwen2_5_vl::{Qwen25VlText, QwenVlTextConfig};
pub use vit_guidance::{rv2v_chain, vae_txt_vit};
