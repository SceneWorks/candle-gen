//! Krea 2's **Qwen3-VL-4B-Instruct** condition encoder (text path only — the vision tower is unused
//! for text-to-image). A 36-layer decoder-only LM; the hidden states at the 12 evenly-spaced indices
//! `text_encoder_select_layers = [2,5,…,35]` are **stacked** (not aggregated here) into
//! `[B, L, 12, 2560]` — the exact `context` the DiT's `TextFusionTransformer` consumes (sc-7569). The
//! learned aggregation lives in the DiT, NOT here. Port of `mlx-gen-krea`'s `text_encoder/`,
//! structured like `candle-gen-boogu`'s Qwen3-VL text encoder.
//!
//! GQA (32 query / 8 kv heads, **decoupled** head_dim 128 so q_proj is 4096-wide while hidden is
//! 2560), bias-less q/k/v/o, **per-head q/k RMSNorm**, HF half-split RoPE (θ = 5e6), SwiGLU MLP,
//! pre-norm causal decoder blocks. Runs in **f32** — the parity-grade precision for this exact encoder
//! in the sibling boogu/ideogram ports; the DiT casts the features down to bf16.
//!
//! HF `hidden_states` indexing: `hidden_states[i]` is the state after running `i` decoder layers
//! (`hidden_states[0]` = the raw embedding), so the reference's `select_hidden = [2,5,…,35]` capture
//! the OUTPUT of 0-indexed layers `[1,4,…,34]`. The final `language_model.norm` is never applied (all
//! selected layers are pre-final-norm), and only `max+1` layers are run.

use std::path::Path;

use candle_gen::candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_gen::candle_nn::ops::softmax_last_dim;
use candle_gen::candle_nn::rotary_emb::rope;

use crate::loader::{embedding_detect, linear_detect, rmsnorm, Weights};
use crate::quant::{QEmbedding, QLinear};

/// Qwen3-VL-4B text-tower architecture (verified from the published `text_encoder/config.json`:
/// `qwen3_vl_text`, hidden 2560, 36 layers, GQA 32/8, head_dim 128, FFN 9728, eps 1e-6) + the Krea
/// conditioning policy (which hidden-state layers to stack, how many template-prefix tokens to drop).
#[derive(Debug, Clone, PartialEq)]
pub struct KreaTeConfig {
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    /// HF `output_hidden_states` indices the pipeline stacks (`model_index.json`
    /// `text_encoder_select_layers`): `hidden_states[i]` = the LM state after running `i` layers.
    pub select_hidden: Vec<usize>,
    /// Leading template-prefix tokens dropped from the conditioning (`Qwen3VLConditioner`'s
    /// `prompt_template_encode_start_idx`); the system-instruction prefix tokenizes to this many.
    pub prefix_tokens: usize,
    /// `<|image_pad|>` id (the vision-embed splice placeholder) — image-grounded edit path only
    /// (epic 10871 / sc-10880). Standard Qwen3-VL `151655` (confirmed for Krea, sc-10875).
    pub image_token_id: u32,
    /// Qwen3-VL `text_config.rope_parameters.mrope_section` — the per-axis (T/H/W) frequency counts over
    /// `head_dim/2 = 64` used by the 3-D interleaved MRoPE on the image-grounded path (`[24, 20, 20]`).
    pub mrope_section: [usize; 3],
}

impl KreaTeConfig {
    pub fn qwen3_vl_4b() -> Self {
        Self {
            num_layers: 36,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 5_000_000.0,
            select_hidden: vec![2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35],
            prefix_tokens: 34,
            image_token_id: 151655,
            mrope_section: [24, 20, 20],
        }
    }

    /// Parse `<root>/text_encoder/config.json` (`text_config`) + `<root>/model_index.json`
    /// (`text_encoder_select_layers`); missing scalars fall back to [`Self::qwen3_vl_4b`].
    pub fn from_snapshot(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref();
        let path = root.join("text_encoder").join("config.json");
        let text = std::fs::read_to_string(&path).map_err(|e| {
            candle_gen::candle_core::Error::Msg(format!("krea te: read {}: {e}", path.display()))
        })?;
        let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| {
            candle_gen::candle_core::Error::Msg(format!("krea te: parse {}: {e}", path.display()))
        })?;
        let tc = v.get("text_config").unwrap_or(&v);
        let d = Self::qwen3_vl_4b();
        let u = |k: &str, dflt: usize| {
            tc.get(k)
                .and_then(serde_json::Value::as_u64)
                .map(|n| n as usize)
                .unwrap_or(dflt)
        };

        let mut cfg = Self {
            num_layers: u("num_hidden_layers", d.num_layers),
            num_heads: u("num_attention_heads", d.num_heads),
            num_kv_heads: u("num_key_value_heads", d.num_kv_heads),
            head_dim: u("head_dim", d.head_dim),
            rms_norm_eps: tc
                .get("rms_norm_eps")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(d.rms_norm_eps),
            // `text_config.rope_theta` is null on disk; honor `rope_parameters`/`rope_scaling` if set,
            // else the qwen3_vl_text default (5e6).
            rope_theta: tc
                .get("rope_parameters")
                .or_else(|| tc.get("rope_scaling"))
                .and_then(|r| r.get("rope_theta"))
                .or_else(|| tc.get("rope_theta"))
                .and_then(serde_json::Value::as_f64)
                .map(|n| n as f32)
                .unwrap_or(d.rope_theta),
            select_hidden: d.select_hidden.clone(),
            prefix_tokens: d.prefix_tokens,
            // `image_token_id` is a top-level config field; `mrope_section` lives under
            // `rope_parameters`/`rope_scaling`. Both fall back to the standard Qwen3-VL values (sc-10875).
            image_token_id: v
                .get("image_token_id")
                .and_then(serde_json::Value::as_u64)
                .map(|n| n as u32)
                .unwrap_or(d.image_token_id),
            mrope_section: tc
                .get("rope_parameters")
                .or_else(|| tc.get("rope_scaling"))
                .and_then(|r| r.get("mrope_section"))
                .and_then(serde_json::Value::as_array)
                .and_then(|a| read_mrope_section(a))
                .unwrap_or(d.mrope_section),
        };

        // `text_encoder_select_layers` lives in the pipeline manifest. A genuinely-absent
        // `model_index.json` keeps the reference `select_hidden` default; a *present-but-corrupt*
        // manifest (I/O error or malformed JSON) errors loudly rather than silently downgrading to the
        // default on a damaged snapshot (sc-9010 / F-073).
        if let Some(mv) = read_optional_model_index(&root.join("model_index.json"))? {
            if let Some(arr) = mv
                .get("text_encoder_select_layers")
                .and_then(|a| a.as_array())
            {
                let sel: Vec<usize> = arr
                    .iter()
                    .filter_map(|x| x.as_u64().map(|n| n as usize))
                    .collect();
                if !sel.is_empty() {
                    cfg.select_hidden = sel;
                }
            }
        }
        Ok(cfg)
    }
}

/// Read the **optional** `model_index.json`, distinguishing "genuinely absent" (→ `Ok(None)`, keep
/// the reference default) from "present but corrupt" (→ `Err`, name the file). A missing manifest is a
/// legitimate snapshot shape; an I/O error or malformed JSON on a manifest that *is* present signals a
/// damaged/partial download that must surface rather than silently downgrade behavior (sc-9010 /
/// F-073).
fn read_optional_model_index(path: &Path) -> Result<Option<serde_json::Value>> {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(ref e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => {
            return Err(candle_gen::candle_core::Error::Msg(format!(
                "krea te: read {}: {e}",
                path.display()
            )))
        }
    };
    let v = serde_json::from_str(&text).map_err(|e| {
        candle_gen::candle_core::Error::Msg(format!(
            "krea te: parse {} (corrupt snapshot?): {e}",
            path.display()
        ))
    })?;
    Ok(Some(v))
}

/// Parse a JSON `mrope_section` array into `[t, h, w]` (exactly three positive counts); any other shape
/// falls back (`None`) to the [`KreaTeConfig::qwen3_vl_4b`] default.
fn read_mrope_section(a: &[serde_json::Value]) -> Option<[usize; 3]> {
    if a.len() != 3 {
        return None;
    }
    let mut out = [0usize; 3];
    for (i, x) in a.iter().enumerate() {
        out[i] = x.as_u64()? as usize;
    }
    Some(out)
}

/// HF half-split RoPE table (θ over `head_dim`), built once for the max sequence length (f32).
struct Rotary {
    cos: Tensor,
    sin: Tensor,
}

impl Rotary {
    fn new(head_dim: usize, theta: f32, max_seq: usize, device: &Device) -> Result<Self> {
        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / theta.powf(i as f32 / head_dim as f32))
            .collect();
        let n = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, n), device)?;
        let t = Tensor::arange(0u32, max_seq as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_seq, 1))?;
        let freqs = t.matmul(&inv_freq)?; // (max_seq, head_dim/2)
        Ok(Self {
            cos: freqs.cos()?,
            sin: freqs.sin()?,
        })
    }

    fn text(&self, seq: usize) -> Result<(Tensor, Tensor)> {
        Ok((self.cos.narrow(0, 0, seq)?, self.sin.narrow(0, 0, seq)?))
    }
}

struct Attention {
    q_proj: QLinear,
    k_proj: QLinear,
    v_proj: QLinear,
    o_proj: QLinear,
    q_norm: Tensor,
    k_norm: Tensor,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    eps: f64,
}

impl Attention {
    fn load(w: &Weights, prefix: &str, cfg: &KreaTeConfig) -> Result<Self> {
        Ok(Self {
            q_proj: linear_detect(w, &format!("{prefix}.q_proj"), false)?,
            k_proj: linear_detect(w, &format!("{prefix}.k_proj"), false)?,
            v_proj: linear_detect(w, &format!("{prefix}.v_proj"), false)?,
            o_proj: linear_detect(w, &format!("{prefix}.o_proj"), false)?,
            q_norm: w.get(&format!("{prefix}.q_norm.weight"))?,
            k_norm: w.get(&format!("{prefix}.k_norm.weight"))?,
            n_heads: cfg.num_heads,
            n_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
            eps: cfg.rms_norm_eps,
        })
    }

    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let (nh, nkv, hd) = (self.n_heads, self.n_kv_heads, self.head_dim);

        let q = self.q_proj.forward(x)?.reshape((b, s, nh, hd))?;
        let k = self.k_proj.forward(x)?.reshape((b, s, nkv, hd))?;
        let v = self.v_proj.forward(x)?.reshape((b, s, nkv, hd))?;
        // Per-head q/k RMSNorm over the head dim, then transpose to [B, H, S, D].
        let q = rmsnorm(&q, &self.q_norm, self.eps)?
            .transpose(1, 2)?
            .contiguous()?;
        let k = rmsnorm(&k, &self.k_norm, self.eps)?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let q = rope(&q, cos, sin)?;
        let k = rope(&k, cos, sin)?;
        let k = repeat_kv(&k, nh / nkv)?;
        let v = repeat_kv(&v, nh / nkv)?;

        let scale = (hd as f64).powf(-0.5);
        let scores = (q.contiguous()?.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
        let scores = scores.broadcast_add(mask)?;
        let probs = softmax_last_dim(&scores)?;
        let o = probs.matmul(&v)?; // [B, nh, S, D]
        let o = o.transpose(1, 2)?.contiguous()?.reshape((b, s, nh * hd))?;
        self.o_proj.forward(&o)
    }
}

/// Repeat each kv head `groups` times along the head axis ([B,nkv,S,D] → [B,nkv·groups,S,D]).
fn repeat_kv(x: &Tensor, groups: usize) -> Result<Tensor> {
    if groups == 1 {
        return Ok(x.clone());
    }
    let (b, nkv, s, d) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((b, nkv, groups, s, d))?
        .contiguous()?
        .reshape((b, nkv * groups, s, d))
}

struct Mlp {
    gate: QLinear,
    up: QLinear,
    down: QLinear,
}

impl Mlp {
    fn load(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            gate: linear_detect(w, &format!("{prefix}.gate_proj"), false)?,
            up: linear_detect(w, &format!("{prefix}.up_proj"), false)?,
            down: linear_detect(w, &format!("{prefix}.down_proj"), false)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gated = (self.gate.forward(x)?.silu()? * self.up.forward(x)?)?;
        self.down.forward(&gated)
    }
}

struct DecoderLayer {
    input_ln: Tensor,
    post_ln: Tensor,
    attn: Attention,
    mlp: Mlp,
    eps: f64,
}

impl DecoderLayer {
    fn load(w: &Weights, prefix: &str, cfg: &KreaTeConfig) -> Result<Self> {
        Ok(Self {
            input_ln: w.get(&format!("{prefix}.input_layernorm.weight"))?,
            post_ln: w.get(&format!("{prefix}.post_attention_layernorm.weight"))?,
            attn: Attention::load(w, &format!("{prefix}.self_attn"), cfg)?,
            mlp: Mlp::load(w, &format!("{prefix}.mlp"))?,
            eps: cfg.rms_norm_eps,
        })
    }

    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let h = (x + self
            .attn
            .forward(&rmsnorm(x, &self.input_ln, self.eps)?, cos, sin, mask)?)?;
        &h + self.mlp.forward(&rmsnorm(&h, &self.post_ln, self.eps)?)?
    }
}

/// The Krea Qwen3-VL-4B text-path condition encoder.
pub struct KreaTextEncoder {
    embed_tokens: QEmbedding,
    layers: Vec<DecoderLayer>,
    rotary: Rotary,
    /// 0-indexed decoder-layer OUTPUT indices to capture (= `select_hidden[i] - 1`), in stack order.
    out_layers: Vec<usize>,
    prefix_tokens: usize,
    // ── image-grounded edit path (epic 10871 / sc-10880) ────────────────────────────────────────
    head_dim: usize,
    rms_norm_eps: f64,
    rope_theta: f32,
    /// Qwen3-VL MRoPE per-axis (T/H/W) frequency counts over `head_dim/2` for the 3-D interleaved rope.
    mrope_section: [usize; 3],
    /// `<|image_pad|>` id — the vision-embed splice placeholder.
    image_token_id: u32,
    device: Device,
}

impl KreaTextEncoder {
    /// Load from the `text_encoder` weights under `prefix` (`"language_model"`). The final
    /// `{prefix}.norm.weight` is intentionally not loaded. `max_seq` sizes the RoPE table.
    pub fn load(w: &Weights, prefix: &str, cfg: &KreaTeConfig, max_seq: usize) -> Result<Self> {
        let out_layers: Vec<usize> = cfg
            .select_hidden
            .iter()
            .map(|&s| {
                s.checked_sub(1).ok_or_else(|| {
                    candle_gen::candle_core::Error::Msg(
                        "krea te: select_hidden index 0 has no layer output".into(),
                    )
                })
            })
            .collect::<Result<_>>()?;
        let max_layer = *out_layers.iter().max().unwrap_or(&0);
        if max_layer >= cfg.num_layers {
            return Err(candle_gen::candle_core::Error::Msg(format!(
                "krea te: select_hidden needs layer {max_layer} but the encoder has {} layers",
                cfg.num_layers
            )));
        }

        let embed_tokens = embedding_detect(w, &format!("{prefix}.embed_tokens"))?;

        let mut layers = Vec::with_capacity(max_layer + 1);
        for i in 0..=max_layer {
            layers.push(DecoderLayer::load(w, &format!("{prefix}.layers.{i}"), cfg)?);
        }
        Ok(Self {
            embed_tokens,
            layers,
            rotary: Rotary::new(cfg.head_dim, cfg.rope_theta, max_seq.max(1), w.device())?,
            out_layers,
            prefix_tokens: cfg.prefix_tokens,
            head_dim: cfg.head_dim,
            rms_norm_eps: cfg.rms_norm_eps,
            rope_theta: cfg.rope_theta,
            mrope_section: cfg.mrope_section,
            image_token_id: cfg.image_token_id,
            device: w.device().clone(),
        })
    }

    /// `input_ids`: `[1, S]` u32. Returns the stacked conditioning `[1, S - prefix_tokens, num_select,
    /// hidden]` (the DiT's `context`), f32. The final norm is never applied; only layers up to
    /// `max(out_layers)` are run. Causal (decoder-only); no padding (the candle tokenizer emits none).
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (b, s) = input_ids.dims2()?;
        let (cos, sin) = self.rotary.text(s)?;
        let mask = causal_mask(b, s, &self.device)?;

        let mut hidden = self.embed_tokens.forward(input_ids)?.to_dtype(DType::F32)?;
        let mut saved: Vec<(usize, Tensor)> = Vec::with_capacity(self.out_layers.len());
        for (i, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden, &cos, &sin, &mask)?;
            if self.out_layers.contains(&i) {
                saved.push((i, hidden.clone()));
            }
        }
        self.stack_and_trim(&saved)
    }

    /// Image-grounded condition encoding for the edit path (epic 10871 / sc-10880). The image-only
    /// sibling of [`forward`](Self::forward): the vision tower's merged embeds are spliced over each
    /// reference's `<|image_pad|>` block, the decoder runs under the 3-D **interleaved MRoPE** (each
    /// reference's grid advancing the shared position counter), and each reference's `deepstack[k]`
    /// features are injected at its block after LM layers 0/1/2 — mirroring `Qwen3VLTextModel` with one
    /// `<|image_pad|>` block per reference. The same select-layer stack + template-prefix drop tail as
    /// [`forward`](Self::forward) produces the DiT's `context` `[1, S − prefix, num_select, hidden]`.
    ///
    /// - `input_ids`: `[1, S]` u32 from [`crate::tokenizer::KreaTokenizer::encode_with_images`] (the edit
    ///   template with `<|vision_start|><|image_pad|>×n<|vision_end|>` per reference).
    /// - `image_embeds[k]`: the k-th reference's merged vision embeds `[n_k, hidden]`.
    /// - `deepstack[k]`: its 3 deepstack features (each `[n_k, hidden]`).
    /// - `grids[k]`: its patch grid `[t, h, w]`.
    ///
    /// The block order must match the reference order (the template emits the references' vision blocks
    /// in order, before the instruction). `b = 1`.
    pub fn forward_with_images(
        &self,
        input_ids: &Tensor,
        image_embeds: &[Tensor],
        deepstack: &[Vec<Tensor>],
        grids: &[[i32; 3]],
    ) -> Result<Tensor> {
        let (b, s) = input_ids.dims2()?;
        let ids: Vec<u32> = input_ids.i(0)?.to_dtype(DType::U32)?.to_vec1::<u32>()?;

        // Contiguous `<|image_pad|>` blocks, in order; block k carries reference k.
        let blocks = image_blocks(&ids, self.image_token_id);
        if blocks.len() != image_embeds.len() {
            return Err(candle_gen::candle_core::Error::Msg(format!(
                "krea te: {} image-token blocks in input_ids but {} reference embeds",
                blocks.len(),
                image_embeds.len()
            )));
        }

        // Token embeddings (f32), then splice each reference's vision embeds over its block. Each
        // replacement is the same length as the block, so earlier splices don't shift later indices.
        let mut hidden = self.embed_tokens.forward(input_ids)?.to_dtype(DType::F32)?;
        for (k, &(start, len)) in blocks.iter().enumerate() {
            if image_embeds[k].dim(0)? != len {
                return Err(candle_gen::candle_core::Error::Msg(format!(
                    "krea te: reference {k} has {} vision tokens but its image block is {len}",
                    image_embeds[k].dim(0)?
                )));
            }
            let img = image_embeds[k].unsqueeze(0)?.to_dtype(hidden.dtype())?; // [1, n_k, hidden]
            hidden = replace_seq(&hidden, &img, start, start + len, s)?;
        }

        // 3-D interleaved MRoPE (per-image grids) + causal mask.
        let (pt, ph, pw) = mrope_positions(&ids, self.image_token_id, grids);
        let (cos, sin) = self.mrope_cos_sin(&pt, &ph, &pw)?;
        let mask = causal_mask(b, s, &self.device)?;

        let mut saved: Vec<(usize, Tensor)> = Vec::with_capacity(self.out_layers.len());
        for (i, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden, &cos, &sin, &mask)?;
            // Deepstack: after LM layers 0/1/2, add each reference's layer-i feature at its block.
            for (k, &(start, len)) in blocks.iter().enumerate() {
                if i < deepstack[k].len() {
                    let ds = deepstack[k][i].unsqueeze(0)?.to_dtype(hidden.dtype())?; // [1, n_k, hidden]
                    let mid = (slice_seq(&hidden, start, start + len)? + ds)?;
                    hidden = replace_seq(&hidden, &mid, start, start + len, s)?;
                }
            }
            if self.out_layers.contains(&i) {
                saved.push((i, hidden.clone()));
            }
        }
        self.stack_and_trim(&saved)
    }

    /// Stack the captured select layers on a NEW axis 2 → `[b, s, n, hidden]` (reference
    /// `torch.stack([hidden_states[i] for i in select], dim=2)`), then drop the leading template-prefix
    /// tokens. Shared by [`forward`](Self::forward) and [`forward_with_images`](Self::forward_with_images)
    /// so the text and image-grounded paths stack identically.
    fn stack_and_trim(&self, saved: &[(usize, Tensor)]) -> Result<Tensor> {
        let pick = |idx: usize| -> Result<Tensor> {
            saved
                .iter()
                .find(|(k, _)| *k == idx)
                .map(|(_, v)| v.clone())
                .ok_or_else(|| {
                    candle_gen::candle_core::Error::Msg(format!(
                        "krea te: hidden state {idx} not captured"
                    ))
                })
        };
        let expanded: Vec<Tensor> = self
            .out_layers
            .iter()
            .map(|&idx| pick(idx)?.unsqueeze(2))
            .collect::<Result<_>>()?;
        let stacked = Tensor::cat(&expanded, 2)?; // [b, s, n, hidden]

        // Drop the leading template-prefix tokens (the system instruction).
        let n = stacked.dim(1)?;
        if self.prefix_tokens >= n {
            return Err(candle_gen::candle_core::Error::Msg(format!(
                "krea te: prompt has {n} tokens but the {} template-prefix tokens leave nothing",
                self.prefix_tokens
            )));
        }
        stacked.narrow(1, self.prefix_tokens, n - self.prefix_tokens)
    }

    /// Build the interleaved-MRoPE `cos`/`sin` `[s, head_dim/2]` (f32). Each of the `head_dim/2`
    /// frequencies takes its position from the T/H/W axis per the Qwen3-VL interleave: within the first
    /// `mrope_section[1]·3` indices, `j%3==1 → H`, `j%3==2 → W`, else `T` (the tail stays `T`). Ported
    /// verbatim from `candle_gen_boogu`'s text encoder (Krea shares the section `[24,20,20]`, head_dim
    /// 128, and θ = 5e6).
    fn mrope_cos_sin(&self, pt: &[i64], ph: &[i64], pw: &[i64]) -> Result<(Tensor, Tensor)> {
        let s = pt.len();
        let hd = self.head_dim;
        let half = hd / 2;
        let sec_h = self.mrope_section[1] * 3;
        let sec_w = self.mrope_section[2] * 3;
        let inv: Vec<f32> = (0..half)
            .map(|j| (self.rope_theta as f64).powf(-(2.0 * j as f64) / hd as f64) as f32)
            .collect();

        let mut freqs = vec![0f32; s * half];
        for (i, ((&t, &h), &w)) in pt.iter().zip(ph).zip(pw).enumerate() {
            for j in 0..half {
                let pos = if j < sec_h && j % 3 == 1 {
                    h
                } else if j < sec_w && j % 3 == 2 {
                    w
                } else {
                    t
                };
                freqs[i * half + j] = pos as f32 * inv[j];
            }
        }
        let freqs = Tensor::from_vec(freqs, (s, half), &self.device)?;
        Ok((freqs.cos()?, freqs.sin()?))
    }

    /// The RMS-norm eps (exposed for the parity harness); the config value threaded at load.
    pub fn rms_norm_eps(&self) -> f64 {
        self.rms_norm_eps
    }
}

/// Slice `[1, s, d]` along the sequence axis (axis 1) to `[start, end)`.
fn slice_seq(x: &Tensor, start: usize, end: usize) -> Result<Tensor> {
    x.narrow(1, start, end - start)
}

/// Replace `x[:, start:end, :]` with `repl` (`[1, end-start, d]`) via concat of the surrounding slices.
fn replace_seq(x: &Tensor, repl: &Tensor, start: usize, end: usize, s: usize) -> Result<Tensor> {
    let before = x.narrow(1, 0, start)?;
    let after = x.narrow(1, end, s - end)?;
    Tensor::cat(&[&before, repl, &after], 1)
}

/// Contiguous runs of `image_token_id` in `ids`, returned as `(start, len)` in input-id order. Each run
/// is one reference image's `<|image_pad|>` block.
fn image_blocks(ids: &[u32], image_token_id: u32) -> Vec<(usize, usize)> {
    let mut blocks = Vec::new();
    let mut i = 0usize;
    while i < ids.len() {
        if ids[i] == image_token_id {
            let start = i;
            while i < ids.len() && ids[i] == image_token_id {
                i += 1;
            }
            blocks.push((start, i - start));
        } else {
            i += 1;
        }
    }
    blocks
}

/// Vision spatial merge — the LM sees one token per `merge²` patches (Qwen3-VL `spatial_merge_size`).
const SPATIAL_MERGE: i64 = 2;

/// 3-D MRoPE positions per token (mirrors `get_rope_index`): text tokens advance `(i, i, i)`; the k-th
/// image block (at running offset `cur`) takes `grids[k] = [t, h, w]`, gets `t = cur`, `h = cur + row`,
/// `w = cur + col` over its `(h/merge)×(w/merge)` merged grid, then advances `cur += max(h, w) / merge`.
/// Multiple image blocks consume `grids` in order. Ported from `candle_gen_boogu`'s text encoder.
fn mrope_positions(
    ids: &[u32],
    image_token_id: u32,
    grids: &[[i32; 3]],
) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
    let (mut pt, mut ph, mut pw) = (Vec::new(), Vec::new(), Vec::new());
    let mut cur = 0i64;
    let mut i = 0usize;
    let mut img_k = 0usize;
    while i < ids.len() {
        if ids[i] == image_token_id {
            let g = grids[img_k];
            let (llm_h, llm_w) = (g[1] as i64 / SPATIAL_MERGE, g[2] as i64 / SPATIAL_MERGE);
            let step = (g[1].max(g[2]) as i64) / SPATIAL_MERGE;
            for idx in 0..(llm_h * llm_w) {
                pt.push(cur);
                ph.push(cur + idx / llm_w);
                pw.push(cur + idx % llm_w);
            }
            cur += step;
            i += (llm_h * llm_w) as usize;
            img_k += 1;
        } else {
            pt.push(cur);
            ph.push(cur);
            pw.push(cur);
            cur += 1;
            i += 1;
        }
    }
    (pt, ph, pw)
}

/// Additive causal mask `[B, 1, S, S]` (f32): `0` where query `i` may attend key `j` (`j ≤ i`),
/// `-inf` otherwise. No padding term (the candle tokenizer emits no padding).
fn causal_mask(b: usize, s: usize, device: &Device) -> Result<Tensor> {
    let mut data = vec![0f32; b * s * s];
    for bi in 0..b {
        for i in 0..s {
            for j in (i + 1)..s {
                data[(bi * s + i) * s + j] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_vec(data, (b, 1, s, s), device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_layers_map_to_zero_indexed_outputs() {
        let cfg = KreaTeConfig::qwen3_vl_4b();
        assert_eq!(cfg.select_hidden.len(), 12);
        assert_eq!(cfg.select_hidden.first().copied(), Some(2));
        assert_eq!(cfg.select_hidden.last().copied(), Some(35));
        // The OUTPUT-of-layer mapping is `select - 1`: captures layers 1..34.
        let out: Vec<usize> = cfg.select_hidden.iter().map(|s| s - 1).collect();
        assert_eq!(out.first().copied(), Some(1));
        assert_eq!(out.last().copied(), Some(34));
        assert!(*out.iter().max().unwrap() < cfg.num_layers);
    }

    fn te_snapshot_tmp(name: &str) -> std::path::PathBuf {
        let tmp = std::env::temp_dir().join(format!(
            "krea_te_{name}_{}_{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(tmp.join("text_encoder")).unwrap();
        // A minimal valid text_encoder/config.json (missing scalars default to qwen3_vl_4b).
        std::fs::write(
            tmp.join("text_encoder").join("config.json"),
            br#"{"text_config": {}}"#,
        )
        .unwrap();
        tmp
    }

    #[test]
    fn from_snapshot_defaults_select_when_model_index_absent() {
        // No model_index.json → keep the reference select_hidden default.
        let tmp = te_snapshot_tmp("idx_absent");
        let cfg = KreaTeConfig::from_snapshot(&tmp).unwrap();
        assert_eq!(cfg.select_hidden, KreaTeConfig::qwen3_vl_4b().select_hidden);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn from_snapshot_reads_present_select_layers() {
        let tmp = te_snapshot_tmp("idx_present");
        std::fs::write(
            tmp.join("model_index.json"),
            br#"{"text_encoder_select_layers": [1, 2, 3]}"#,
        )
        .unwrap();
        let cfg = KreaTeConfig::from_snapshot(&tmp).unwrap();
        assert_eq!(cfg.select_hidden, vec![1, 2, 3]);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn from_snapshot_errors_on_corrupt_model_index() {
        // model_index.json present but malformed (partial download) → error, NOT silent default.
        let tmp = te_snapshot_tmp("idx_corrupt");
        std::fs::write(tmp.join("model_index.json"), b"{ not json").unwrap();
        assert!(KreaTeConfig::from_snapshot(&tmp).is_err());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn te_config_carries_grounding_defaults() {
        let c = KreaTeConfig::qwen3_vl_4b();
        assert_eq!(c.image_token_id, 151655);
        assert_eq!(c.mrope_section, [24, 20, 20]);
    }

    const IMG: u32 = 151655;

    #[test]
    fn image_blocks_finds_runs_in_order() {
        // text, text, [4 image], text, [2 image], text.
        let ids = [9u32, 9, IMG, IMG, IMG, IMG, 9, IMG, IMG, 9];
        assert_eq!(image_blocks(&ids, IMG), vec![(2, 4), (7, 2)]);
    }

    #[test]
    fn mrope_positions_advance_across_two_images() {
        // Block 0 ↔ grid [1,4,4] (merged 2×2 = 4 tokens, t-step max(4,4)/2 = 2);
        // block 1 ↔ grid [1,4,2] (merged 2×1 = 2 tokens, t-step max(4,2)/2 = 2). The scene-then-person
        // fixed order (sc-10878) is exactly the grid order fed here.
        let ids = [9u32, 9, IMG, IMG, IMG, IMG, 9, IMG, IMG, 9];
        let grids = [[1, 4, 4], [1, 4, 2]];
        let (pt, ph, pw) = mrope_positions(&ids, IMG, &grids);
        assert_eq!(pt.len(), ids.len());
        assert_eq!((pt[0], pt[1]), (0, 1));
        assert_eq!(&pt[2..6], &[2, 2, 2, 2]);
        assert_eq!(&ph[2..6], &[2, 2, 3, 3]);
        assert_eq!(&pw[2..6], &[2, 3, 2, 3]);
        assert_eq!(pt[6], 4);
        assert_eq!(&pt[7..9], &[5, 5]);
        assert_eq!(&ph[7..9], &[5, 6]);
        assert_eq!(&pw[7..9], &[5, 5]);
        assert_eq!(pt[9], 7);
    }
}
