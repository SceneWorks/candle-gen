//! LTX-2.3 **MLX packed-tier ingestion** (sc-9545, sc-9089 umbrella) — the split-file / subfolder
//! resolver + key-remap layer that lets the merged sc-9417 packed-detect seam ([`crate::quant`]) fire on
//! the REAL `SceneWorks/ltx-2.3-mlx` q4/q8 tier weights, with **no dense staging**.
//!
//! ## The gap sc-9417 left (why the wiring alone could not render)
//!
//! sc-9417 wired [`crate::quant::qlinear`] / [`crate::quant::qembedding`] to packed-DETECT on a
//! `{key}.scales` sibling, validated with SYNTHETIC fixtures on the crate's own key layout. But the
//! crate's dense loader ([`crate::Pipeline::load_components`]) consumes a **single bundled** Lightricks
//! checkpoint (`ltx-2.3-22b-distilled.safetensors`, keys under `model.diffusion_model.*` with the dense
//! Lightricks spelling), whereas the hosted MLX tier ships **split per-component** safetensors in a
//! `q4/` (or `q8/`) subfolder whose `transformer.safetensors` uses **remapped** keys. So the packed
//! `.scales` siblings live under different names than the crate asks for, and the DiT body, connectors,
//! VAE, and gemma each live in a different file. This module bridges that: it resolves the tier's files
//! and rewrites the crate's key requests to the tier's names via candle's [`Rename`] backend, so the
//! *existing* loaders ([`crate::transformer::AvDiT::new`], [`crate::connector`], [`crate::vae`],
//! [`crate::gemma`]) load straight from the packed/dense tier parts unchanged.
//!
//! ## The real q4 tier layout (hf-header audit, sc-9545)
//!
//! ```text
//! <snapshot>/q4/               (or q8/)
//!   quantize_config.json       { "quantization": { "bits": 4, "group_size": 64 } }
//!   config.json embedded_config.json split_model.json
//!   transformer.safetensors    PACKED — DiT body, 1344 `.scales`; keys REMAPPED (see below)
//!   connector.safetensors      DENSE  — `*_embeddings_connector.*` + `text_embedding_projection.*`,
//!                                       crate-native spelling (`to_out.0`, `ff.net.0.proj`, `ff.net.2`)
//!   vae_decoder.safetensors    DENSE  — `up_blocks.*` / `conv_in` / `conv_out` / `per_channel_statistics`,
//!                                       conv weights CHANNELS-LAST `[O,kt,kh,kw,I]`, stats `mean`/`std`
//!   vae_encoder audio_vae vocoder upsampler   DENSE  (not needed for the T2V DiT render)
//! <snapshot>/gemma/            DENSE  — standard `language_model.model.*` 5-shard set + tokenizer.json
//! ```
//!
//! ## The transformer key remap (crate spelling → tier spelling), unambiguous per the audit
//!
//! | crate loader asks for | tier `transformer.safetensors` |
//! |---|---|
//! | `…attn*.to_out.0.{weight,scales,biases,bias}` | `…attn*.to_out.…` |
//! | `…ff.net.0.proj.…` | `…ff.proj_in.…` |
//! | `…ff.net.2.…` | `…ff.proj_out.…` |
//! | `…emb.timestep_embedder.linear_1.…` | `…linear1.…` |
//! | `…emb.timestep_embedder.linear_2.…` | `…linear2.…` |
//! | `model.diffusion_model.<X>` | `<X>` (the DiT sits at the file root) |
//!
//! The tier transformer carries **zero** keys in the crate spelling (audited: 0 `net.0.proj`, 0
//! `to_out.0`, 0 `linear_1`), so the rewrite is total and never collides. The connector file *does* use
//! the crate spelling natively, so [`remap_transformer_key`] is applied only to the DiT builder, never
//! the connector one.
//!
//! ## group_size from config (sc-9545 AC)
//!
//! [`TierPaths::packed_config`] reads `quantize_config.json`'s `quantization.group_size` via the shared
//! [`candle_gen::quant::PackedConfig`]; [`TierPaths::validate_group_size`] asserts it equals the
//! [`crate::quant::GROUP_SIZE`] the loaders repack at (64), failing loudly rather than silently
//! mis-repacking if a future tier ever ships a different group.

use std::path::{Path, PathBuf};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::var_builder::Rename;
use candle_gen::candle_nn::VarBuilder;
use candle_gen::{CandleError, Result as CResult};

/// The resolved file set of a packed LTX tier subfolder (`.../q4` or `.../q8`) + its sibling `gemma/`.
pub struct TierPaths {
    /// The `q4/` (or `q8/`) subfolder holding the split per-component safetensors.
    pub tier_dir: PathBuf,
    /// The Gemma-3-12B encoder snapshot dir (the tier's sibling `gemma/`, or an override).
    pub gemma_dir: PathBuf,
}

impl TierPaths {
    /// Detect a packed tier at `dir`: a directory that directly holds `transformer.safetensors` **and**
    /// `quantize_config.json` (the MLX split-tier marker). Returns `None` for the dense single-bundle
    /// layout so [`crate::Pipeline`] keeps the legacy path unchanged.
    ///
    /// `gemma_override` (from `LoadSpec::text_encoder` / `$LTX_GEMMA_DIR`) wins for the Gemma dir; else
    /// the tier's sibling `gemma/` (one level up from the `q4/` subdir) is used.
    pub fn detect(dir: &Path, gemma_override: Option<&Path>) -> Option<Self> {
        let marker = dir.join("transformer.safetensors");
        let cfg = dir.join("quantize_config.json");
        if !(marker.is_file() && cfg.is_file()) {
            return None;
        }
        let gemma_dir = gemma_override
            .map(Path::to_path_buf)
            .or_else(|| {
                // The tier nests `<snapshot>/{q4,q8,gemma}`; from `<snapshot>/q4` the gemma sibling is
                // `../gemma`.
                dir.parent().map(|p| p.join("gemma")).filter(|g| g.is_dir())
            })
            .unwrap_or_else(|| dir.join("gemma"));
        Some(Self {
            tier_dir: dir.to_path_buf(),
            gemma_dir,
        })
    }

    fn file(&self, name: &str) -> CResult<PathBuf> {
        let p = self.tier_dir.join(name);
        if !p.is_file() {
            return Err(CandleError::Msg(format!(
                "ltx tier: missing `{name}` in {} (expected a split MLX tier: transformer / connector \
                 / vae_decoder / vae_encoder / audio_vae / vocoder)",
                self.tier_dir.display()
            )));
        }
        Ok(p)
    }

    /// Parse the tier's `quantize_config.json` → the shared [`candle_gen::quant::PackedConfig`]
    /// (`quantization.{bits, group_size}`). Errors if the file is absent/unparseable or carries no
    /// `quantization` block (a packed tier always has one).
    pub fn packed_config(&self) -> CResult<candle_gen::quant::PackedConfig> {
        let p = self.tier_dir.join("quantize_config.json");
        let text = std::fs::read_to_string(&p)
            .map_err(|e| CandleError::Msg(format!("ltx tier: read {}: {e}", p.display())))?;
        let json: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| CandleError::Msg(format!("ltx tier: parse {}: {e}", p.display())))?;
        candle_gen::quant::PackedConfig::from_config(&json).ok_or_else(|| {
            CandleError::Msg(format!(
                "ltx tier: {} has no `quantization.bits` — not a packed tier config",
                p.display()
            ))
        })
    }

    /// Read + **validate** the tier's `group_size` against the [`crate::quant::GROUP_SIZE`] the packed
    /// loaders repack at (sc-9545 AC). The MLX-packed→GGML repack ([`candle_gen::quant`]) is done at a
    /// fixed group; if the tier ever ships a different group the repack would mis-align, so fail loudly
    /// rather than render garbage. Returns the validated group size.
    pub fn validate_group_size(&self) -> CResult<usize> {
        let cfg = self.packed_config()?;
        let g = cfg.group_size as usize;
        if g != crate::quant::GROUP_SIZE {
            return Err(CandleError::Msg(format!(
                "ltx tier: quantize_config.json group_size {g} != the loader's repack group {} — the \
                 MLX→GGML repack would mis-align. A tier at a new group needs the group threaded into \
                 the packed loaders (candle_gen::quant::*_gs already accepts it).",
                crate::quant::GROUP_SIZE
            )));
        }
        Ok(g)
    }

    /// The DiT VarBuilder over the tier's **packed** `transformer.safetensors`, with the crate→tier key
    /// remap applied so [`crate::transformer::AvDiT::new`] (which asks for `model.diffusion_model.<X>`
    /// with the dense Lightricks spelling) resolves the tier's rootless remapped keys — firing the
    /// packed-detect seam on the real `.scales` siblings. Loaded at `dtype` (bf16).
    pub fn dit_vb(&self, dtype: DType, device: &Device) -> CResult<VarBuilder<'static>> {
        let inner =
            candle_gen::mmap_var_builder(&[self.file("transformer.safetensors")?], dtype, device)?;
        Ok(rename_vb(inner, dtype, device, remap_transformer_key))
    }

    /// The connector + text-projection VarBuilder over the tier's **dense** `connector.safetensors`.
    /// Its keys already use the crate spelling (`to_out.0`, `ff.net.0.proj`, `ff.net.2`), so only the
    /// `model.diffusion_model.` prefix (which the crate prepends for the connectors) is stripped — the
    /// text projection is at the file root, the connectors too.
    pub fn connector_vb(&self, dtype: DType, device: &Device) -> CResult<VarBuilder<'static>> {
        let inner =
            candle_gen::mmap_var_builder(&[self.file("connector.safetensors")?], dtype, device)?;
        Ok(rename_vb(inner, dtype, device, strip_diffusion_prefix))
    }

    /// The video-VAE decoder VarBuilder over the tier's **dense** `vae_decoder.safetensors`, remapped so
    /// [`crate::vae::LtxVideoVae::new`] (which asks for `vae.decoder.<X>` and `vae.per_channel_statistics.
    /// {mean-of-means,std-of-means}`) resolves the tier's rootless `<X>` / `per_channel_statistics.{mean,
    /// std}`. Conv weights are additionally **permuted** from the tier's channels-last `[O,kt,kh,kw,I]` to
    /// the crate's PyTorch `[O,I,kt,kh,kw]` on load (see [`VaeRemapBackend`]).
    pub fn vae_vb(&self, dtype: DType, device: &Device) -> CResult<VarBuilder<'static>> {
        let inner =
            candle_gen::mmap_var_builder(&[self.file("vae_decoder.safetensors")?], dtype, device)?;
        Ok(VarBuilder::from_backend(
            Box::new(VaeRemapBackend { inner }),
            dtype,
            device.clone(),
        ))
    }

    /// The Gemma-3-12B encoder VarBuilder rooted at `language_model.model.` over the tier's sibling
    /// `gemma/` shards. The tier ships Gemma **dense** with the standard `language_model.model.*` keys
    /// (matches the crate exactly), so no remap — just the sorted-shard resolve.
    pub fn gemma_vb(&self, dtype: DType, device: &Device) -> CResult<VarBuilder<'static>> {
        let files = candle_gen::sorted_safetensors(&self.gemma_dir, "ltx tier gemma")?;
        Ok(candle_gen::mmap_var_builder(&files, dtype, device)?.pp("language_model.model"))
    }

    /// The Gemma tokenizer path (`gemma/tokenizer.json`).
    pub fn tokenizer_path(&self) -> PathBuf {
        self.gemma_dir.join("tokenizer.json")
    }
}

/// Wrap `inner` in candle's [`Rename`] backend applying `f` to every requested key. Boxed as a
/// `Renamer` fn so the DiT / connector remaps share one path.
fn rename_vb(
    inner: VarBuilder<'static>,
    dtype: DType,
    device: &Device,
    f: fn(&str) -> String,
) -> VarBuilder<'static> {
    let renamer: Box<dyn Fn(&str) -> String + Send + Sync> = Box::new(f);
    VarBuilder::from_backend(Box::new(Rename::new(inner, renamer)), dtype, device.clone())
}

/// Rewrite a crate DiT key (what [`crate::transformer`] asks for) to the tier's `transformer.safetensors`
/// spelling. Strips the `model.diffusion_model.` prefix (the DiT is at the file root) and applies the
/// four projection renames. Order matters: strip the prefix first, then the sub-key renames.
pub fn remap_transformer_key(key: &str) -> String {
    let k = key
        .strip_prefix("model.diffusion_model.")
        .unwrap_or(key)
        .to_string();
    // `attn*.to_out.0.<suffix>` → `attn*.to_out.<suffix>` (candle `Linear` under a `.0` ModuleList slot).
    let k = k.replace(".to_out.0.", ".to_out.");
    // FeedForward: `ff.net.0.proj.<suffix>` → `ff.proj_in.<suffix>`, `ff.net.2.<suffix>` → `ff.proj_out`.
    let k = k
        .replace(".net.0.proj.", ".proj_in.")
        .replace(".net.2.", ".proj_out.");
    // AdaLayerNormSingle timestep MLP: `linear_1`/`linear_2` → `linear1`/`linear2`.
    k.replace(".linear_1.", ".linear1.")
        .replace(".linear_2.", ".linear2.")
}

/// Strip the crate's `model.diffusion_model.` connector prefix — the tier's `connector.safetensors`
/// roots the connectors + text projection at the file top, in the crate's own key spelling.
fn strip_diffusion_prefix(key: &str) -> String {
    key.strip_prefix("model.diffusion_model.")
        .unwrap_or(key)
        .to_string()
}

/// A `SimpleBackend` for the tier's `vae_decoder.safetensors` that (1) strips the crate's `vae.` /
/// `vae.decoder.` prefix, (2) renames the `mean-of-means`/`std-of-means` stats to the tier's
/// `mean`/`std`, and (3) **permutes** every conv weight from the tier's channels-last `[O,kt,kh,kw,I]`
/// to the crate's `[O,I,kt,kh,kw]`. Kept a bespoke backend (not [`Rename`]) because the conv permute is a
/// tensor transform, not just a key rewrite.
struct VaeRemapBackend {
    inner: VarBuilder<'static>,
}

impl VaeRemapBackend {
    /// crate VAE key → tier `vae_decoder.safetensors` key. Returns `(tier_key, is_conv_weight)`.
    fn remap(key: &str) -> (String, bool) {
        // The crate roots the VAE at `vae.` then descends into `decoder.` for the block body and
        // `per_channel_statistics.` for the stats — the tier drops the `vae.`/`vae.decoder.` wrappers.
        let k = key
            .strip_prefix("vae.decoder.")
            .or_else(|| key.strip_prefix("vae."))
            .unwrap_or(key);
        // Stats: crate `per_channel_statistics.mean-of-means` → tier `.mean`; `std-of-means` → `.std`.
        let k = k
            .replace(
                "per_channel_statistics.mean-of-means",
                "per_channel_statistics.mean",
            )
            .replace(
                "per_channel_statistics.std-of-means",
                "per_channel_statistics.std",
            );
        // A conv weight (`….conv.weight`) is channels-last in the tier and must be permuted on load.
        let is_conv_weight = k.ends_with(".conv.weight");
        (k, is_conv_weight)
    }

    /// Permute a tier conv weight `[O,kt,kh,kw,I]` → crate `[O,I,kt,kh,kw]`.
    fn permute_conv(w: Tensor) -> candle_gen::candle_core::Result<Tensor> {
        if w.rank() == 5 {
            w.permute((0, 4, 1, 2, 3))?.contiguous()
        } else {
            // A 1-D conv (rare) or already-torch-layout weight: leave as-is.
            Ok(w)
        }
    }
}

impl candle_gen::candle_nn::var_builder::SimpleBackend for VaeRemapBackend {
    fn get(
        &self,
        s: candle_gen::candle_core::Shape,
        name: &str,
        _h: candle_gen::candle_nn::Init,
        dtype: DType,
        dev: &Device,
    ) -> candle_gen::candle_core::Result<Tensor> {
        // Shape-checked reads are only used for non-conv leaves (the crate reads convs via
        // `get_unchecked`, then infers dims from the permuted shape), so no permute is needed here.
        let (k, _) = Self::remap(name);
        self.inner
            .get_with_hints_dtype(s, &k, Default::default(), dtype)?
            .to_device(dev)
    }

    fn get_unchecked(
        &self,
        name: &str,
        dtype: DType,
        dev: &Device,
    ) -> candle_gen::candle_core::Result<Tensor> {
        let (k, is_conv) = Self::remap(name);
        let t = self.inner.get_unchecked_dtype(&k, dtype)?.to_device(dev)?;
        if is_conv {
            Self::permute_conv(t)
        } else {
            Ok(t)
        }
    }

    fn contains_tensor(&self, name: &str) -> bool {
        let (k, _) = Self::remap(name);
        self.inner.contains_tensor(&k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The transformer remap turns every crate DiT key into the audited tier spelling (sc-9545) — the
    /// exact `to_out.0`/`ff.net.*`/`linear_*` → `to_out`/`proj_in`/`proj_out`/`linear1/2` rewrites the
    /// hf-header audit of `SceneWorks/ltx-2.3-mlx` q4 found, plus the `model.diffusion_model.` strip.
    #[test]
    fn transformer_remap_matches_real_tier_layout() {
        // attn `to_out.0` (packed triple + dense bias) → `to_out`.
        assert_eq!(
            remap_transformer_key(
                "model.diffusion_model.transformer_blocks.0.attn1.to_out.0.scales"
            ),
            "transformer_blocks.0.attn1.to_out.scales"
        );
        assert_eq!(
            remap_transformer_key(
                "model.diffusion_model.transformer_blocks.0.attn1.to_out.0.weight"
            ),
            "transformer_blocks.0.attn1.to_out.weight"
        );
        // FeedForward `net.0.proj` / `net.2` → `proj_in` / `proj_out`.
        assert_eq!(
            remap_transformer_key(
                "model.diffusion_model.transformer_blocks.5.ff.net.0.proj.scales"
            ),
            "transformer_blocks.5.ff.proj_in.scales"
        );
        assert_eq!(
            remap_transformer_key("model.diffusion_model.transformer_blocks.5.ff.net.2.weight"),
            "transformer_blocks.5.ff.proj_out.weight"
        );
        // audio_ff too (the `.net.*` rewrite is prefix-agnostic).
        assert_eq!(
            remap_transformer_key(
                "model.diffusion_model.transformer_blocks.5.audio_ff.net.0.proj.bias"
            ),
            "transformer_blocks.5.audio_ff.proj_in.bias"
        );
        // AdaLayerNormSingle timestep MLP linear_1/2 → linear1/2.
        assert_eq!(
            remap_transformer_key(
                "model.diffusion_model.adaln_single.emb.timestep_embedder.linear_1.weight"
            ),
            "adaln_single.emb.timestep_embedder.linear1.weight"
        );
        assert_eq!(
            remap_transformer_key(
                "model.diffusion_model.adaln_single.emb.timestep_embedder.linear_2.bias"
            ),
            "adaln_single.emb.timestep_embedder.linear2.bias"
        );
        // Un-nested keys (patchify_proj, to_q, scale_shift_table, gate) only lose the prefix.
        assert_eq!(
            remap_transformer_key("model.diffusion_model.patchify_proj.weight"),
            "patchify_proj.weight"
        );
        assert_eq!(
            remap_transformer_key("model.diffusion_model.transformer_blocks.0.attn1.to_q.scales"),
            "transformer_blocks.0.attn1.to_q.scales"
        );
        assert_eq!(
            remap_transformer_key(
                "model.diffusion_model.transformer_blocks.0.attn1.to_gate_logits.weight"
            ),
            "transformer_blocks.0.attn1.to_gate_logits.weight"
        );
    }

    /// The connector prefix strip leaves the crate-native connector spelling intact (the tier connector
    /// file uses `to_out.0` / `ff.net.*` natively, so ONLY the `model.diffusion_model.` prefix is removed
    /// — the transformer projection renames must NOT be applied here).
    #[test]
    fn connector_strip_keeps_native_spelling() {
        assert_eq!(
            strip_diffusion_prefix(
                "model.diffusion_model.video_embeddings_connector.transformer_1d_blocks.0.attn1.to_out.0.weight"
            ),
            "video_embeddings_connector.transformer_1d_blocks.0.attn1.to_out.0.weight"
        );
        assert_eq!(
            strip_diffusion_prefix(
                "model.diffusion_model.audio_embeddings_connector.transformer_1d_blocks.0.ff.net.0.proj.weight"
            ),
            "audio_embeddings_connector.transformer_1d_blocks.0.ff.net.0.proj.weight"
        );
        // The text projection is already at the file root (no prefix to strip).
        assert_eq!(
            strip_diffusion_prefix("text_embedding_projection.video_aggregate_embed.weight"),
            "text_embedding_projection.video_aggregate_embed.weight"
        );
    }

    /// The VAE remap drops the `vae.`/`vae.decoder.` wrapper, renames the stats, and flags conv weights
    /// for the channels-last→torch permute — matching the audited tier `vae_decoder.safetensors` layout.
    #[test]
    fn vae_remap_matches_real_tier_layout() {
        let (k, conv) =
            VaeRemapBackend::remap("vae.decoder.up_blocks.0.res_blocks.0.conv1.conv.weight");
        assert_eq!(k, "up_blocks.0.res_blocks.0.conv1.conv.weight");
        assert!(conv, "a `.conv.weight` must be flagged for the permute");
        let (k, conv) = VaeRemapBackend::remap("vae.decoder.conv_in.conv.bias");
        assert_eq!(k, "conv_in.conv.bias");
        assert!(!conv, "a bias is not permuted");
        let (k, _) = VaeRemapBackend::remap("vae.per_channel_statistics.mean-of-means");
        assert_eq!(k, "per_channel_statistics.mean");
        let (k, _) = VaeRemapBackend::remap("vae.per_channel_statistics.std-of-means");
        assert_eq!(k, "per_channel_statistics.std");
    }

    /// The permute turns a tier channels-last conv `[O,kt,kh,kw,I]` into the crate `[O,I,kt,kh,kw]`.
    #[test]
    fn conv_permute_channels_last_to_torch() -> candle_gen::candle_core::Result<()> {
        use candle_gen::candle_core::Device;
        // [O=2, kt=3, kh=3, kw=3, I=4] → [2,4,3,3,3].
        let w = Tensor::arange(0f32, (2 * 3 * 3 * 3 * 4) as f32, &Device::Cpu)?
            .reshape((2, 3, 3, 3, 4))?;
        let p = VaeRemapBackend::permute_conv(w.clone())?;
        assert_eq!(p.dims(), &[2, 4, 3, 3, 3]);
        // Spot-check via flat buffers: p[o,i,t,h,ww] == w[o,t,h,ww,i].
        let wv = w.flatten_all()?.to_vec1::<f32>()?; // strides [O]:108 [kt]:36 [kh]:12 [kw]:4 [I]:1
        let pv = p.flatten_all()?.to_vec1::<f32>()?; // strides [O]:108 [I]:27 [kt]:9 [kh]:3 [kw]:1
        let w_idx = |o: usize, t: usize, h: usize, ww: usize, i: usize| {
            o * 108 + t * 36 + h * 12 + ww * 4 + i
        };
        let p_idx = |o: usize, i: usize, t: usize, h: usize, ww: usize| {
            o * 108 + i * 27 + t * 9 + h * 3 + ww
        };
        for o in 0..2 {
            for i in 0..4 {
                for t in 0..3 {
                    for h in 0..3 {
                        for ww in 0..3 {
                            assert_eq!(pv[p_idx(o, i, t, h, ww)], wv[w_idx(o, t, h, ww, i)]);
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
