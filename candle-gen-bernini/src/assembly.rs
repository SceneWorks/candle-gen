//! The planner-input assembly glue that brackets the MAR loop — `format_mllm_inputs_embeds`
//! (`bernini.py`) before it, and the UMT5 `concat_with_zero_init` (`pipeline.__call__`) after it.
//! Candle sibling of `mlx-gen-bernini/src/assembly.rs` (sc-5140).
//!
//!   - [`format_mllm_inputs_embeds`] — embed the token ids, then `masked_scatter` the ViT visual
//!     features into the visual slots (`visual_input_mask | visual_output_mask`). The gen-output slots
//!     get placeholder features here; [`crate::mar::post_process_input_embeds`] then overwrites them
//!     with the `mask_token` before the loop starts.
//!   - [`concat_with_zero_init`] — the renderer streams' text combine: prepend the UMT5 prompt embeds
//!     to each planner stream, then zero-pad / truncate to `max_sequence_length` (512). The UMT5
//!     encoder itself is candle-gen-wan's; this is only the prepend + pad/truncate mechanics.
//!
//! These are exact host/tensor ops (gather, scatter, concat, pad, slice) → validated bit-for-bit
//! against `tests/fixtures/assembly_golden.safetensors` (the same golden the MLX lane asserts).

use candle_gen::candle_core::Tensor;
use candle_gen::{CandleError, Result as CResult};

use crate::mar::scatter_rows;
use crate::qwen2_5_vl::Qwen25VlText;

/// `format_mllm_inputs_embeds` (`bernini.py`): `embed_tokens(input_ids)` `[1, L, H]`, then scatter the
/// `visual_embeds` `[n, H]` (input-ViT + target-ViT features, in sequence order) into the visual slots
/// (`visual_input_mask | visual_output_mask`). With no visual features it's just the token embedding.
///
/// `input_ids` is the flat id list (length `L`); the two masks are per-token booleans of length `L`.
/// The scatter is row-order: visual feature `j` fills the `j`-th visual slot in ascending position,
/// matching torch `masked_scatter` (which fills `True` positions in flattened order).
pub fn format_mllm_inputs_embeds(
    backbone: &Qwen25VlText,
    input_ids: &[i64],
    visual_embeds: Option<&Tensor>,
    visual_input_mask: &[bool],
    visual_output_mask: &[bool],
) -> CResult<Tensor> {
    let l = input_ids.len();
    let ids = Tensor::from_vec(
        input_ids.to_vec(),
        (1, l),
        &candle_gen::candle_core::Device::Cpu,
    )?;
    let embeds = backbone.embed(&ids)?; // [1, L, H]

    let ve = match visual_embeds {
        Some(v) if v.dim(0)? > 0 => v,
        _ => return Ok(embeds),
    };

    let visual_idx: Vec<u32> = (0..l)
        .filter(|&i| visual_input_mask[i] || visual_output_mask[i])
        .map(|i| i as u32)
        .collect();
    let n = visual_idx.len();
    if n != ve.dim(0)? {
        return Err(CandleError::Msg(format!(
            "format_mllm_inputs_embeds: {n} visual slots but {} visual features",
            ve.dim(0)?
        )));
    }
    let h = embeds.dim(2)?;
    scatter_rows(&embeds, &visual_idx, &ve.reshape((1, n, h))?)
}

/// `concat_with_zero_init` (`pipeline.__call__`): prepend the UMT5 prompt embeds `[1, T, W]` to a
/// planner stream `[1, S, W]`, then zero-pad (or truncate) the result to `max_sequence_length` tokens
/// → `[1, max_seq, W]`. Padding appends zero rows on the sequence axis (the reference `feat.new_zeros`).
pub fn concat_with_zero_init(
    t5_embeds: &Tensor,
    stream: &Tensor,
    max_seq: usize,
) -> CResult<Tensor> {
    let combined = Tensor::cat(&[t5_embeds, stream], 1)?; // [1, T+S, W]
    pad_and_truncate(&combined, max_seq)
}

/// Zero-pad (append) or truncate `feat` `[1, S, W]` to exactly `max_seq` tokens on the sequence axis.
pub fn pad_and_truncate(feat: &Tensor, max_seq: usize) -> CResult<Tensor> {
    let (b, s, w) = feat.dims3()?;
    if s < max_seq {
        let zeros = Tensor::zeros((b, max_seq - s, w), feat.dtype(), feat.device())?;
        Ok(Tensor::cat(&[feat, &zeros], 1)?)
    } else if s > max_seq {
        Ok(feat.narrow(1, 0, max_seq)?)
    } else {
        Ok(feat.clone())
    }
}
