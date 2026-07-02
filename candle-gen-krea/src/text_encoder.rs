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

use candle_gen::candle_core::{DType, Device, Result, Tensor};
use candle_gen::candle_nn::ops::softmax_last_dim;
use candle_gen::candle_nn::rotary_emb::rope;
use candle_gen::candle_nn::{Embedding, Linear, Module};

use crate::loader::{linear, rmsnorm, Weights};

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
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
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
            q_proj: linear(w, &format!("{prefix}.q_proj"), false)?,
            k_proj: linear(w, &format!("{prefix}.k_proj"), false)?,
            v_proj: linear(w, &format!("{prefix}.v_proj"), false)?,
            o_proj: linear(w, &format!("{prefix}.o_proj"), false)?,
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
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl Mlp {
    fn load(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            gate: linear(w, &format!("{prefix}.gate_proj"), false)?,
            up: linear(w, &format!("{prefix}.up_proj"), false)?,
            down: linear(w, &format!("{prefix}.down_proj"), false)?,
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
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    rotary: Rotary,
    /// 0-indexed decoder-layer OUTPUT indices to capture (= `select_hidden[i] - 1`), in stack order.
    out_layers: Vec<usize>,
    prefix_tokens: usize,
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

        let embed_weight = w.get(&format!("{prefix}.embed_tokens.weight"))?;
        let hidden = embed_weight.dim(1)?;
        let embed_tokens = Embedding::new(embed_weight, hidden);

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

        // Stack the captured layers (in `out_layers` order) on a NEW axis 2 → [b, s, n, hidden],
        // matching the reference `torch.stack([hidden_states[i] for i in select], dim=2)`.
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
}
