//! Bernini planner **MAR handoff** — the planner→renderer feature composition (candle sibling of the
//! handoff half of `mlx-gen-bernini/src/mar.rs`, sc-5140). This ships the pieces that turn a computed
//! planner hidden state into the 4 renderer conditioning streams, plus the MaskGIT reveal schedule and
//! the sequence-axis scatter primitive — the pure/host + connector-only surface that is golden-testable
//! without the vision tower or the clip-diff flow-matching head.
//!
//! The full MAR sampling loop (`sample_vit_embed` in the MLX reference) additionally drives the
//! Qwen2.5-VL backbone × 3 streams and the clip-diff denoiser per reveal step; that loop (and the
//! vision tower + clip-diff modules it needs) is the follow-up beyond this slice — see the crate docs.

use candle_gen::candle_core::Tensor;
use candle_gen::Result as CResult;

use crate::connector::MlpConnector;

fn mask_to_idx(mask: &[bool], want: bool) -> Vec<u32> {
    mask.iter()
        .enumerate()
        .filter(|&(_, &m)| m == want)
        .map(|(i, _)| i as u32)
        .collect()
}

/// `post_process_input_embeds` (inference): set the gen-ViT slots of `input_embeds` `[1, L, H]` to
/// `mask_token` `[1, 1, H]` (broadcast). `gen_mask[i]` marks a gen-ViT slot. Implemented as
/// `x·(1−g) + mask_token·g` (no in-place scatter), matching the reference exactly.
pub fn post_process_input_embeds(
    input_embeds: &Tensor,
    gen_mask: &[bool],
    mask_token: &Tensor,
) -> CResult<Tensor> {
    let l = gen_mask.len();
    let g: Vec<f32> = gen_mask
        .iter()
        .map(|&m| if m { 1.0 } else { 0.0 })
        .collect();
    let gv =
        Tensor::from_vec(g, (1, l, 1), input_embeds.device())?.to_dtype(input_embeds.dtype())?;
    let keep = (1.0 - &gv)?;
    Ok((input_embeds.broadcast_mul(&keep)? + mask_token.broadcast_mul(&gv)?)?)
}

/// `feat_from_planner_to_renderer` (inference): `for_gen` over all tokens + the txt/vit sub-masks.
pub struct RendererFeat {
    /// `connector.for_gen(hidden)` — `[1, L, gen_dim]`.
    pub contexts: Tensor,
    /// Token positions that are **not** gen-ViT (text + input-vit).
    pub txt_idx: Vec<u32>,
    /// Token positions that **are** gen-ViT.
    pub vit_idx: Vec<u32>,
}

/// Run the renderer-feature projection over `hidden` `[1, L, H]`; `gen_mask` marks the gen-ViT slots.
pub fn feat_to_renderer(
    hidden: &Tensor,
    gen_mask: &[bool],
    connector: &MlpConnector,
) -> CResult<RendererFeat> {
    Ok(RendererFeat {
        contexts: connector.for_gen(hidden)?,
        txt_idx: mask_to_idx(gen_mask, false),
        vit_idx: mask_to_idx(gen_mask, true),
    })
}

/// The 4 renderer conditioning streams.
pub struct FourStreams {
    pub wtxt_wvit: Tensor,
    pub wtxt_wovit: Tensor,
    pub wotxt_wvit: Tensor,
    pub wotxt_wovit: Tensor,
}

/// Gather rows along the sequence axis (dim 1) at `idx`.
fn take_seq(a: &Tensor, idx: &[u32]) -> CResult<Tensor> {
    let sel = Tensor::from_vec(idx.to_vec(), (idx.len(),), a.device())?;
    Ok(a.index_select(&sel, 1)?)
}

/// Build the 4 streams from the cond + uncond planner hidden states (the `sample_vit_embed` tail,
/// `else`/`masked_tgt_embed_with_qwen_txt_vit_tokens` branch).
pub fn four_streams(
    cond_hidden: &Tensor,
    cond_gen_mask: &[bool],
    uncond_hidden: &Tensor,
    uncond_gen_mask: &[bool],
    connector: &MlpConnector,
) -> CResult<FourStreams> {
    let c = feat_to_renderer(cond_hidden, cond_gen_mask, connector)?;
    let u = feat_to_renderer(uncond_hidden, uncond_gen_mask, connector)?;
    Ok(FourStreams {
        wtxt_wovit: take_seq(&c.contexts, &c.txt_idx)?,
        wotxt_wvit: take_seq(&c.contexts, &c.vit_idx)?,
        wotxt_wovit: take_seq(&u.contexts, &u.txt_idx)?,
        wtxt_wvit: c.contexts,
    })
}

/// The MaskGIT/MAR reveal schedule: for each of `planning_step` steps, the **sorted** target-token
/// positions revealed (un-masked) on that step (candle port of the mask bookkeeping in the reference
/// `sample_vit_embed`):
///
///   - `mask_ratio = cos(π/2·(s+1)/N)` (f64, like numpy); `mask_len = floor(n_query·ratio)`.
///   - clamp `mask_len = max(1, min(masked_now − 1, mask_len))` (always leave ≥1 masked, never grow).
///   - the still-masked set is always a prefix `order[:mask_len]`, so the newly-revealed chunk is
///     `order[mask_len : prev_mask_len]`; the **last** step reveals the whole remaining masked set.
///   - positions are returned **ascending** (the reference gathers/scatters by sorted token position).
///
/// `order` is the seeded reveal permutation of `[0, n_query)`. The chunks across steps are disjoint and
/// cover every token exactly once.
pub fn mar_schedule(n_query: i32, planning_step: usize, order: &[i32]) -> Vec<Vec<i32>> {
    let mut out = Vec::with_capacity(planning_step);
    let mut prev = n_query;
    for step in 0..planning_step {
        let ratio = (std::f64::consts::PI / 2.0 * (step + 1) as f64 / planning_step as f64).cos();
        let raw = (n_query as f64 * ratio).floor() as i32;
        let mask_len = raw.min(prev - 1).max(1);
        let revealed: &[i32] = if step >= planning_step - 1 {
            &order[..prev as usize]
        } else {
            &order[mask_len as usize..prev as usize]
        };
        let mut sorted: Vec<i32> = revealed.to_vec();
        sorted.sort_unstable();
        out.push(sorted);
        prev = mask_len;
    }
    out
}

/// Overwrite the rows of `base` `[1, L, H]` at positions `idx` with `src` `[1, n, H]` (row `j` ←
/// `src[0, j]`), leaving the other rows untouched. `idx.len() == n`. Implemented as a pure row gather
/// over `concat([base; src])`, so it is bit-exact (a one-hot matmul would pick up the f32 matmul floor).
///
/// The MAR sampling loop that drives this per reveal step (`sample_vit_embed` in the reference) is the
/// follow-up beyond this slice; the primitive + its unit test ship now so the loop can reuse it.
#[allow(dead_code)]
pub(crate) fn scatter_rows(base: &Tensor, idx: &[u32], src: &Tensor) -> CResult<Tensor> {
    let (_, l, h) = base.dims3()?;
    let n = idx.len();
    let stacked = Tensor::cat(&[&base.reshape((l, h))?, &src.reshape((n, h))?], 0)?; // [L+n, H]

    let mut gi: Vec<u32> = (0..l as u32).collect();
    for (j, &pos) in idx.iter().enumerate() {
        gi[pos as usize] = (l + j) as u32;
    }
    let sel = Tensor::from_vec(gi, (l,), base.device())?;
    Ok(stacked.index_select(&sel, 0)?.reshape((1, l, h))?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Device;

    /// The reveal schedule chunks are disjoint and cover every token exactly once (ascending order).
    #[test]
    fn mar_schedule_covers_all_tokens_once() {
        let n = 12i32;
        let steps = 5usize;
        let order: Vec<i32> = (0..n).rev().collect(); // an arbitrary permutation
        let sched = mar_schedule(n, steps, &order);
        assert_eq!(sched.len(), steps);
        let mut seen = vec![0u32; n as usize];
        for chunk in &sched {
            // ascending
            let mut sorted = chunk.clone();
            sorted.sort_unstable();
            assert_eq!(&sorted, chunk, "chunk positions must be ascending");
            for &p in chunk {
                seen[p as usize] += 1;
            }
        }
        assert!(
            seen.iter().all(|&c| c == 1),
            "every token revealed exactly once"
        );
    }

    /// scatter_rows overwrites exactly the indexed rows, leaving the others bit-identical.
    #[test]
    fn scatter_rows_overwrites_indexed_only() {
        let base = Tensor::from_vec(
            (0..8u32).map(|v| v as f32).collect::<Vec<_>>(),
            (1, 4, 2),
            &Device::Cpu,
        )
        .unwrap();
        let src =
            Tensor::from_vec(vec![100f32, 101.0, 200.0, 201.0], (1, 2, 2), &Device::Cpu).unwrap();
        let out = scatter_rows(&base, &[1u32, 3], &src).unwrap();
        let v: Vec<f32> = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // rows: 0 untouched [0,1]; 1 -> [100,101]; 2 untouched [4,5]; 3 -> [200,201]
        assert_eq!(v, vec![0.0, 1.0, 100.0, 101.0, 4.0, 5.0, 200.0, 201.0]);
    }

    /// post_process sets exactly the gen slots to the mask token.
    #[test]
    fn post_process_sets_gen_slots() {
        let emb =
            Tensor::from_vec(vec![1f32, 2.0, 3.0, 4.0, 5.0, 6.0], (1, 3, 2), &Device::Cpu).unwrap();
        let mask_token = Tensor::from_vec(vec![-1f32, -2.0], (1, 1, 2), &Device::Cpu).unwrap();
        let out = post_process_input_embeds(&emb, &[false, true, false], &mask_token).unwrap();
        let v: Vec<f32> = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(v, vec![1.0, 2.0, -1.0, -2.0, 5.0, 6.0]);
    }
}
