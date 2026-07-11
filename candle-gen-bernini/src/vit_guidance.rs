//! The Bernini renderer's **ViT-conditioned** guidance-combine modes (candle sibling of
//! `mlx-gen-bernini/src/vit_guidance.rs`, sc-5142) — the velocity combination half of the
//! `BerniniPipeline`-only guidance modes. These take the per-stream target predictions (each = a DiT
//! forward over a given packed-latent variant + one of the planner's 4 prompt-embed streams) and
//! combine them into the step's noise prediction.
//!
//! Two delta families:
//!   - **plain** ([`vae_txt_vit`] w/ `apg=false`, [`rv2v_chain`] w/ `apg=false`) — raw `to − from` deltas.
//!   - **APG** (`apg=true`) — each delta v-space-projected against a per-mode reference (see
//!     [`crate::guidance::apg_delta`]).
//!
//! All predictions are `[1, n_target, C]` (the target-sliced packed-token velocity, batch 1).

use candle_gen::candle_core::Tensor;
use candle_gen::Result as CResult;

use crate::guidance::apg_delta;

/// Fixed APG projection scales for the ViT-conditioned modes (`apg_delta` defaults).
const PARALLEL_SCALE: f32 = 0.2;
const ORTHOGONAL_SCALE: f32 = 1.0;

/// `base + Σ ω·delta` accumulator.
fn combine(base: &Tensor, terms: &[(f32, Tensor)]) -> CResult<Tensor> {
    let mut acc = base.clone();
    for (w, d) in terms {
        acc = (acc + (d * *w as f64)?)?;
    }
    Ok(acc)
}

/// Optionally APG-project a delta against `reference`; identity when `apg` is false.
fn maybe_apg(delta: Tensor, reference: &Tensor, apg: bool) -> CResult<Tensor> {
    if apg {
        apg_delta(&delta, reference, PARALLEL_SCALE, ORTHOGONAL_SCALE)
    } else {
        Ok(delta)
    }
}

/// `vae_txt_vit` (`apg=false`) / `vae_txt_vit_wapg` (`apg=true`) — the primary full-Bernini mode
/// (t2i / edit). Three cumulative deltas over the VAE-conditioned predictions, the APG reference being
/// the **higher-conditioned** ("to") prediction of each delta:
///
///   `base + ω_img·Δ(img←base) + ω_txt·Δ(txt←img) + ω_tgt·Δ(vit←txt)`
#[allow(clippy::too_many_arguments)]
pub fn vae_txt_vit(
    base: &Tensor,
    img: &Tensor,
    txt: &Tensor,
    vit: &Tensor,
    omega_img: f32,
    omega_txt: f32,
    omega_tgt: f32,
    apg: bool,
) -> CResult<Tensor> {
    let d_img = maybe_apg((img - base)?, img, apg)?;
    let d_txt = maybe_apg((txt - img)?, txt, apg)?;
    let d_vit = maybe_apg((vit - txt)?, vit, apg)?;
    combine(
        base,
        &[(omega_img, d_img), (omega_txt, d_txt), (omega_tgt, d_vit)],
    )
}

/// `rv2v_wapg` (`apg=false`) / `r2v_wapg` (`apg=true`) — the 5-prediction reference chain over the
/// video / image / text / ViT conditioning. The APG reference being the **lower-conditioned** ("from")
/// prediction of each delta:
///
///   `base + ω_vid·Δ(V←base) + ω_img·Δ(VI←V) + ω_txt·Δ(VTI←VI) + ω_tgt·Δ(VTIC←VTI)`
#[allow(clippy::too_many_arguments)]
pub fn rv2v_chain(
    base: &Tensor,
    eps_v: &Tensor,
    eps_vi: &Tensor,
    eps_vti: &Tensor,
    eps_vtic: &Tensor,
    omega_vid: f32,
    omega_img: f32,
    omega_txt: f32,
    omega_tgt: f32,
    apg: bool,
) -> CResult<Tensor> {
    let d_vid = maybe_apg((eps_v - base)?, base, apg)?;
    let d_img = maybe_apg((eps_vi - eps_v)?, eps_v, apg)?;
    let d_txt = maybe_apg((eps_vti - eps_vi)?, eps_vi, apg)?;
    let d_vit = maybe_apg((eps_vtic - eps_vti)?, eps_vti, apg)?;
    combine(
        base,
        &[
            (omega_vid, d_vid),
            (omega_img, d_img),
            (omega_txt, d_txt),
            (omega_tgt, d_vit),
        ],
    )
}
