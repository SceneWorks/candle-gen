//! SDXL dual-CLIP conditioning (sc-5491, epic 5480) — the candle twin of
//! `mlx-gen-sdxl::pipeline::encode_conditioning`, the piece the txt2img [`crate::pipeline`] does NOT
//! build (it uses only the final CLIP hidden state and no pooled text-embeds, relying on the stock
//! candle UNet). The InstantID UNet ([`crate::UNet2DConditionModel::forward_instantid`]) needs the real
//! SDXL micro-conditioning, so this assembles it exactly as diffusers does:
//!
//! - **cross-attention conditioning** `[B, 77, 2048]` = `cat(penultimate(CLIP-L)[768],
//!   penultimate(CLIP-bigG)[1280])` — the *second-to-last* encoder-layer hidden state (`hidden_states[-2]`,
//!   pre-final-layer-norm) of each encoder, via candle's
//!   [`ClipTextTransformer::forward_until_encoder_layer`] at `until_layer = -2`;
//! - **pooled text-embeds** `[B, 1280]` = `text_projection(finalₙₒᵣₘ(CLIP-bigG)[eos])` — the bigG
//!   final-layer-norm hidden at the EOS position (the arg-max token, EOS being the highest id) projected
//!   through `text_projection` (the `CLIPTextModelWithProjection` head candle's `ClipTextTransformer`
//!   omits, loaded here from the same `text_encoder_2` checkpoint).
//!
//! **CFG batch order is uncond-first** (`[negative, prompt]`) to match the candle txt2img +
//! [`crate::denoise`] convention (`eps_uncond + cfg·(eps_cond − eps_uncond)`, chunk 0 = uncond) — NOT
//! the mlx crate's positive-first order. The InstantID glue batches the face tokens uncond-first too.

use std::path::Path;

use candle_core::{DType, Device, IndexOp, Tensor, D};
use tokenizers::Tokenizer;

use candle_gen::quant::QLinear;
use candle_gen::{CandleError, Result};

// The vendored, packed-detecting CLIP tower + its config (sc-9527): a pure superset of the stock
// `stable_diffusion::clip` — a dense snapshot loads exactly as before, a packed MLX tier loads its
// dual CLIP straight from the packed parts.
use crate::clip::{self, ClipTextTransformer};
use crate::pipeline::{hf_get, snapshot_file, Clip};

/// Pad/truncate-check a token id list to exactly `max_len`: error if longer (parity with the txt2img
/// path's hard reject — a silently truncated prompt drops conditioning), else right-pad with `pad_id`.
/// Factored out (and the EOS pool below) so the token bookkeeping is unit-testable without CLIP weights.
fn pad_tokens(mut ids: Vec<u32>, pad_id: u32, max_len: usize) -> Result<Vec<u32>> {
    if ids.len() > max_len {
        return Err(CandleError::Msg(format!(
            "sdxl conditioning: prompt is {} tokens > the {max_len}-token CLIP limit",
            ids.len()
        )));
    }
    ids.resize(max_len, pad_id);
    Ok(ids)
}

/// The EOS position of a CLIP token row = the arg-max token id (EOS = `<|endoftext|>` = 49407 is the
/// highest id, and SDXL pads with `"!"` so there is exactly one EOS). diffusers pools the bigG hidden
/// state here for the `text_embeds`. Returns 0 for an empty row (degenerate; never happens for a real
/// 77-token row that always carries BOS+EOS).
fn eos_position(ids: &[u32]) -> usize {
    ids.iter()
        .enumerate()
        .max_by_key(|(_, &v)| v)
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Pool the bigG final-layer-norm hidden `final_g` `[B, 77, 1280]` at each row's EOS position and run it
/// through `text_projection` → pooled text-embeds `[B, 1280]`. `token_rows[b]` is row `b`'s padded ids
/// (for the EOS lookup). Split out so the gather/projection is testable with synthetic tensors.
fn pool_eos(
    final_g: &Tensor,
    token_rows: &[Vec<u32>],
    text_projection: &QLinear,
) -> Result<Tensor> {
    let mut rows = Vec::with_capacity(token_rows.len());
    for (b, ids) in token_rows.iter().enumerate() {
        rows.push(final_g.i((b, eos_position(ids)))?); // [1280]
    }
    let eos_hidden = Tensor::stack(&rows, 0)?; // [B, 1280]
    Ok(text_projection.forward(&eos_hidden)?)
}

/// A loaded SDXL dual-CLIP conditioner: both text encoders + the bigG `text_projection` + the two
/// tokenizers. Built once per model; `encode` is called per request.
pub struct SdxlConditioner {
    tok_l: Tokenizer,
    tok_g: Tokenizer,
    clip_l: ClipTextTransformer,
    clip_g: ClipTextTransformer,
    /// bigG `CLIPTextModelWithProjection` head (`text_projection.weight`, `[1280, 1280]`, no bias) —
    /// a packed-detecting [`QLinear`] (sc-9527): packed straight from the MLX triple, else dense.
    text_projection: QLinear,
    cfg_l: clip::Config,
    cfg_g: clip::Config,
    device: Device,
}

/// Resolve one CLIP encoder's weight file + packed group size (sc-9527): the packed
/// `text_encoder{,_2}/model.safetensors` + its config group size on a packed MLX tier, else the dense
/// `.fp16` file at the default group 64. The vendored tower's `linear_detect_gs` takes the packed or
/// dense path per the presence of `.scales` siblings; the returned group size only matters on the
/// packed branch.
fn clip_weight_file(
    root: &Path,
    which: &Clip,
    dense_sub: &str,
) -> Result<(std::path::PathBuf, usize)> {
    match crate::pipeline::detect_packed_clip(root, which)? {
        Some((packed_file, group_size)) => Ok((packed_file, group_size)),
        None => Ok((
            snapshot_file(root, dense_sub)?,
            candle_gen::quant::MLX_GROUP_SIZE,
        )),
    }
}

/// Build one vendored, packed-detecting CLIP tower for `which` and return it alongside the VarBuilder
/// it was built from (so a caller — the bigG `text_projection` — can reuse the same mmap) and the
/// packed group size.
fn load_clip_tower_with_vb(
    root: &Path,
    which: &Clip,
    device: &Device,
    dtype: DType,
) -> Result<(ClipTextTransformer, candle_nn::VarBuilder<'static>, usize)> {
    let (_tok_repo, dense_sub) = which.sources();
    let (file, group_size) = clip_weight_file(root, which, dense_sub)?;
    let vb = candle_gen::mmap_var_builder(&[file], dtype, device)?;
    let tower = ClipTextTransformer::new_gs(vb.clone(), &which.vendored_config(), group_size)?;
    Ok((tower, vb, group_size))
}

/// As [`load_clip_tower_with_vb`] but drops the VarBuilder (the CLIP-L encoder has no projection head).
fn load_clip_tower(
    root: &Path,
    which: &Clip,
    device: &Device,
    dtype: DType,
) -> Result<ClipTextTransformer> {
    Ok(load_clip_tower_with_vb(root, which, device, dtype)?.0)
}

impl SdxlConditioner {
    /// Load both CLIP encoders from the SDXL snapshot (`text_encoder/` = CLIP-L, `text_encoder_2/` =
    /// bigG), the bigG `text_projection`, and the two model-agnostic tokenizers (cached via `hf-hub`,
    /// exactly as the txt2img path). `dtype` is the compute dtype (f16 for production). Both encoders
    /// and the projection head packed-detect through the vendored tower (sc-9527), so a packed MLX tier
    /// loads its dual CLIP straight from the packed parts; a dense snapshot is byte-identical.
    pub fn load(root: &Path, device: &Device, dtype: DType) -> Result<Self> {
        let cfg_l = clip::Config::sdxl();
        let cfg_g = clip::Config::sdxl2();
        let (tok_l_repo, _l_sub) = Clip::L.sources();
        let (tok_g_repo, _g_sub) = Clip::BigG.sources();

        let tok_l = Tokenizer::from_file(hf_get(tok_l_repo, "tokenizer.json")?)
            .map_err(|e| CandleError::Msg(format!("load tokenizer {tok_l_repo}: {e}")))?;
        let tok_g = Tokenizer::from_file(hf_get(tok_g_repo, "tokenizer.json")?)
            .map_err(|e| CandleError::Msg(format!("load tokenizer {tok_g_repo}: {e}")))?;

        // sc-9527: build each encoder through the vendored, packed-detecting tower. On a packed MLX
        // tier the weights live under `text_encoder{,_2}/model.safetensors` (with `.scales` siblings);
        // on a dense snapshot they live under the `.fp16` name and the vendored tower takes the dense
        // path (byte-identical to the stock builder, pinned by `clip::tests::vendored_dense_matches_stock`).
        let clip_l = load_clip_tower(root, &Clip::L, device, dtype)?;
        let (clip_g, g_vb, g_group_size) =
            load_clip_tower_with_vb(root, &Clip::BigG, device, dtype)?;

        // The bigG `text_projection.weight` (`CLIPTextModelWithProjection` pooled head) lives at the bare
        // top-level key of the SAME bigG checkpoint the tower just read. It packs on a packed tier, so
        // load it packed-detecting from the same VarBuilder (no separate materialization) — dense
        // (`text_projection.weight`) on a dense snapshot.
        let text_projection = clip::text_projection(&g_vb, cfg_g.projection_dim, g_group_size)?;

        Ok(Self {
            tok_l,
            tok_g,
            clip_l,
            clip_g,
            text_projection,
            cfg_l,
            cfg_g,
            device: device.clone(),
        })
    }

    /// Tokenize `text` through `tok`, padded to the encoder's `max_position_embeddings` with the config
    /// pad token (`"!"` for SDXL; EOS otherwise — the candle txt2img rule). Returns the padded id row.
    fn tokenize(&self, tok: &Tokenizer, cfg: &clip::Config, text: &str) -> Result<Vec<u32>> {
        let pad_token = cfg
            .pad_with
            .clone()
            .unwrap_or_else(|| "<|endoftext|>".into());
        let pad_id = *tok
            .get_vocab(true)
            .get(pad_token.as_str())
            .ok_or_else(|| CandleError::Msg(format!("pad token {pad_token:?} not in vocab")))?;
        let ids = tok
            .encode(text, true)
            .map_err(|e| CandleError::Msg(format!("tokenize: {e}")))?
            .get_ids()
            .to_vec();
        pad_tokens(ids, pad_id, cfg.max_position_embeddings)
    }

    /// Encode `prompt` (+ `negative` under CFG) into the SDXL conditioning `[B, 77, 2048]` and pooled
    /// text-embeds `[B, 1280]`. With CFG the batch is **uncond-first** (`[negative, prompt]`), matching
    /// [`crate::denoise`]. Without CFG (`cfg_on = false`) a single cond row.
    pub fn encode(&self, prompt: &str, negative: &str, cfg_on: bool) -> Result<(Tensor, Tensor)> {
        // Token rows, uncond-first under CFG.
        let texts: Vec<&str> = if cfg_on {
            vec![negative, prompt]
        } else {
            vec![prompt]
        };

        // Encode **one row at a time** (batch 1) and stack along the batch dim afterwards — NOT a single
        // batched-2 CLIP forward. candle-transformers' stock CLIP builds its causal attention mask as
        // `[B, S, S]` and `broadcast_add`s it onto the per-head scores `[B, H, S, S]`; that only aligns
        // when `B == 1` (the mask's batch dim broadcasts against the head dim). At `B >= 2` it panics
        // (`shape mismatch in broadcast_add, lhs [2, H, 77, 77], rhs [2, 77, 77]`). The stock SD pipeline
        // dodges this by running uncond/cond as separate passes, so do the same here.
        let mut penult_rows: Vec<Tensor> = Vec::with_capacity(texts.len());
        let mut pooled_rows: Vec<Tensor> = Vec::with_capacity(texts.len());
        for text in &texts {
            let row_l = self.tokenize(&self.tok_l, &self.cfg_l, text)?;
            let row_g = self.tokenize(&self.tok_g, &self.cfg_g, text)?;
            let ids_l = Tensor::new(row_l.as_slice(), &self.device)?
                .reshape((1, self.cfg_l.max_position_embeddings))?;
            let ids_g = Tensor::new(row_g.as_slice(), &self.device)?
                .reshape((1, self.cfg_g.max_position_embeddings))?;

            // Penultimate hidden (`hidden_states[-2]`, pre-final-norm) from each encoder; the bigG `.0`
            // is its final-norm hidden (for the pooled head). `usize::MAX` = the plain causal mask (no
            // padding truncation), matching the txt2img path.
            let (_final_l, penult_l) =
                self.clip_l
                    .forward_until_encoder_layer(&ids_l, usize::MAX, -2)?;
            let (final_g, penult_g) =
                self.clip_g
                    .forward_until_encoder_layer(&ids_g, usize::MAX, -2)?;
            penult_rows.push(Tensor::cat(&[&penult_l, &penult_g], D::Minus1)?); // [1, 77, 2048]
                                                                                // pool the single-row bigG final hidden at its EOS, then project → [1, 1280].
            pooled_rows.push(pool_eos(
                &final_g,
                std::slice::from_ref(&row_g),
                &self.text_projection,
            )?);
        }

        let conditioning = Tensor::cat(&penult_rows, 0)?; // [B, 77, 2048]
        let pooled = Tensor::cat(&pooled_rows, 0)?; // [B, 1280]
        Ok((conditioning, pooled))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// `pad_tokens`: right-pads to `max_len` with the pad id, and rejects an over-long prompt (no silent
    /// truncation — a dropped tail loses conditioning).
    #[test]
    fn pad_tokens_pads_and_rejects_overflow() {
        let p = pad_tokens(vec![1, 2, 3], 9, 6).unwrap();
        assert_eq!(p, vec![1, 2, 3, 9, 9, 9]);
        // Exactly max is fine.
        assert_eq!(pad_tokens(vec![1, 2, 3], 9, 3).unwrap(), vec![1, 2, 3]);
        // Over max errors.
        assert!(pad_tokens(vec![1, 2, 3, 4], 9, 3).is_err());
    }

    /// `eos_position` = the arg-max id (EOS is the highest CLIP id). Finds the EOS even with padding
    /// after it, and the BOS (id 49406) before it.
    #[test]
    fn eos_position_is_argmax() {
        // [BOS=49406, tok, tok, EOS=49407, pad=256, pad=256] → EOS at index 3.
        let row = [49406u32, 320, 540, 49407, 256, 256];
        assert_eq!(eos_position(&row), 3);
        // A row whose EOS is last.
        assert_eq!(eos_position(&[49406, 1, 2, 3, 49407]), 4);
    }

    /// `pool_eos` gathers the per-row EOS hidden and projects it — shape `[B, out]`, and it actually
    /// selects the EOS row (not row 0). Synthetic `final_g` with a distinctive EOS-position vector.
    #[test]
    fn pool_eos_gathers_eos_and_projects() {
        let dev = Device::Cpu;
        // B=1, S=4, H=3. EOS at index 2 (token ids put the max there).
        let token_rows = vec![vec![49406u32, 10, 49407, 256]];
        // final_g: rows 0..4 are [r,r,r]; the EOS row (index 2) is [2,2,2].
        let data: Vec<f32> = (0..4).flat_map(|r| vec![r as f32; 3]).collect();
        let final_g = Tensor::from_vec(data, (1, 4, 3), &dev).unwrap();
        // Identity projection (3→3) so the output is exactly the gathered EOS hidden — wrapped as a
        // dense `QLinear` (the sc-9527 packed-detecting projection type; here the dense arm).
        let eye =
            Tensor::from_vec(vec![1f32, 0., 0., 0., 1., 0., 0., 0., 1.], (3, 3), &dev).unwrap();
        let proj = QLinear::from_dense(candle_gen::quant::DenseLinear::Linear(
            candle_nn::Linear::new(eye, None),
        ));
        let pooled = pool_eos(&final_g, &token_rows, &proj).unwrap();
        assert_eq!(pooled.dims(), &[1, 3]);
        let v = pooled.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(v, vec![2.0, 2.0, 2.0]); // the EOS-position (index 2) hidden, projected by identity
    }
}
