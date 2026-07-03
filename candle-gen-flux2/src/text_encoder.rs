//! FLUX.2's decoder-LM text encoder. Two checkpoints share this graph: klein's **Qwen3** (36 layers,
//! hidden 4096, θ=1e6, eps 1e-6, per-head q/k-norm, `model.*` keys) and dev's **Mistral** (the
//! language tower of a `Mistral3ForConditionalGeneration`: hidden 5120, θ=1e9, eps 1e-5, **no**
//! q/k-norm, `language_model.model.*` keys). Their intermediate hidden states — Qwen3 layers
//! (9, 18, 27) → `[B, S, 12288]`, Mistral layers (10, 20, 30) → `[B, S, 15360]` — are concatenated
//! into the transformer's `prompt_embeds`. Port of `mlx-gen-flux2`'s `text_encoder/` module (which
//! likewise unifies both behind a single `qk_norm` flag).
//!
//! Both: GQA (32 query / 8 kv heads), **bias-less** q/k/v/o projections, HF half-split RoPE, SwiGLU
//! MLP, pre-norm residual blocks. The prompt path runs only up to `max(out_layers)` layers (higher
//! layers cannot influence the kept states), applies **no** final norm, and concatenates the three
//! saved states on the feature axis. Runs in **f32** (the transformer's x/context embedders require
//! f32 input). The per-head q/k RMSNorm is the Qwen3 addition — gated by `te_qk_norm` (klein on,
//! dev off).

use candle_gen::candle_core::{DType, Device, Result, Tensor, D};
use candle_gen::candle_nn::{
    ops::softmax_last_dim, rms_norm, rotary_emb::rope, Module, RmsNorm, VarBuilder,
};
use candle_gen::gen_core::Quant;

use crate::config::Flux2Config;
use crate::quant::{rms_norm_to, QEmbedding, QLinear};

/// HF half-split RoPE table (θ over `head_dim`), built once for the max sequence length.
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

    /// Rows of the precomputed cos/sin tables — the max sequence length this Rotary was sized for.
    /// A `narrow(0, 0, seq)` beyond this fails opaquely, so [`Flux2PromptEncoder::prompt_embeds`]
    /// validates `seq` against it up front (sc-9386, F-077 sibling).
    fn max_seq(&self) -> Result<usize> {
        self.cos.dim(0)
    }

    fn apply(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        let (_, _, seq, _) = q.dims4()?;
        let cos = self.cos.narrow(0, 0, seq)?;
        let sin = self.sin.narrow(0, 0, seq)?;
        let q = rope(&q.contiguous()?, &cos, &sin)?;
        let k = rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q, k))
    }

    /// Move the precomputed RoPE tables to `device` (CPU-staged dev quant path).
    fn to_device(&self, device: &Device) -> Result<Self> {
        Ok(Self {
            cos: self.cos.to_device(device)?,
            sin: self.sin.to_device(device)?,
        })
    }
}

struct Attention {
    q_proj: QLinear,
    k_proj: QLinear,
    v_proj: QLinear,
    o_proj: QLinear,
    /// Per-head q/k RMSNorm over the head dim — `Some` for Qwen3 (klein), `None` for Mistral (dev).
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    /// RMSNorm eps, kept so the q/k norms can be rebuilt on the GPU by the CPU-staged quant path.
    eps: f64,
}

impl Attention {
    fn new(cfg: &Flux2Config, vb: VarBuilder) -> Result<Self> {
        let h = cfg.te_hidden_size;
        let (nh, nkv, hd) = (cfg.te_n_heads, cfg.te_n_kv_heads, cfg.te_head_dim);
        // Mistral (dev) has no `q_norm`/`k_norm` weights — only build them when the variant carries
        // per-head q/k-norm, so loading the dev tower doesn't look for absent keys.
        let (q_norm, k_norm) = if cfg.te_qk_norm {
            (
                Some(rms_norm(hd, cfg.te_rms_norm_eps, vb.pp("q_norm"))?),
                Some(rms_norm(hd, cfg.te_rms_norm_eps, vb.pp("k_norm"))?),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            q_proj: QLinear::linear_detect(h, nh * hd, &vb, "q_proj", false)?,
            k_proj: QLinear::linear_detect(h, nkv * hd, &vb, "k_proj", false)?,
            v_proj: QLinear::linear_detect(h, nkv * hd, &vb, "v_proj", false)?,
            o_proj: QLinear::linear_detect(nh * hd, h, &vb, "o_proj", false)?,
            q_norm,
            k_norm,
            n_heads: nh,
            n_kv_heads: nkv,
            head_dim: hd,
            eps: cfg.te_rms_norm_eps,
        })
    }

    fn quantize_onto(&mut self, quant: Quant, device: &Device) -> Result<()> {
        self.q_proj.quantize_onto(quant, device)?;
        self.k_proj.quantize_onto(quant, device)?;
        self.v_proj.quantize_onto(quant, device)?;
        self.o_proj.quantize_onto(quant, device)?;
        if let Some(n) = &self.q_norm {
            self.q_norm = Some(rms_norm_to(n, self.eps, device)?);
        }
        if let Some(n) = &self.k_norm {
            self.k_norm = Some(rms_norm_to(n, self.eps, device)?);
        }
        Ok(())
    }

    fn forward(&self, x: &Tensor, rotary: &Rotary, mask: &Tensor) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let (nh, nkv, hd) = (self.n_heads, self.n_kv_heads, self.head_dim);

        // Project, reshape to [B, H, S, D]. Per-head q/k RMSNorm (over the head_dim axis, before
        // RoPE) is Qwen3-only; for Mistral (dev) q/k pass straight through.
        let q = self.q_proj.forward(x)?.reshape((b, s, nh, hd))?;
        let k = self.k_proj.forward(x)?.reshape((b, s, nkv, hd))?;
        let v = self.v_proj.forward(x)?.reshape((b, s, nkv, hd))?;
        let q = match &self.q_norm {
            Some(n) => n.forward(&q)?,
            None => q,
        }
        .transpose(1, 2)?; // [B, nh, S, D]
        let k = match &self.k_norm {
            Some(n) => n.forward(&k)?,
            None => k,
        }
        .transpose(1, 2)?; // [B, nkv, S, D]
        let v = v.transpose(1, 2)?.contiguous()?;

        let (q, k) = rotary.apply(&q, &k)?;
        // GQA: repeat kv heads to query-head count.
        let k = repeat_kv(&k, nh / nkv)?;
        let v = repeat_kv(&v, nh / nkv)?;

        let scale = (hd as f64).powf(-0.5);
        let scores = (q.contiguous()?.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
        let scores = scores.broadcast_add(mask)?; // [B, nh, S, S] + [B, 1, S, S]
        let probs = softmax_last_dim(&scores)?;
        let o = probs.matmul(&v)?; // [B, nh, S, D]
        let o = o.transpose(1, 2)?.reshape((b, s, nh * hd))?;
        self.o_proj.forward(&o)
    }
}

/// Repeat each kv head `groups` times along the head axis ([B, nkv, S, D] → [B, nkv·groups, S, D]).
fn repeat_kv(x: &Tensor, groups: usize) -> Result<Tensor> {
    if groups == 1 {
        return Ok(x.clone());
    }
    let (b, nkv, s, d) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((b, nkv, groups, s, d))?
        .reshape((b, nkv * groups, s, d))
}

struct Mlp {
    gate: QLinear,
    up: QLinear,
    down: QLinear,
}

impl Mlp {
    fn new(cfg: &Flux2Config, vb: VarBuilder) -> Result<Self> {
        let (h, i) = (cfg.te_hidden_size, cfg.te_intermediate_size);
        Ok(Self {
            gate: QLinear::linear_detect(h, i, &vb, "gate_proj", false)?,
            up: QLinear::linear_detect(h, i, &vb, "up_proj", false)?,
            down: QLinear::linear_detect(i, h, &vb, "down_proj", false)?,
        })
    }

    fn quantize_onto(&mut self, quant: Quant, device: &Device) -> Result<()> {
        self.gate.quantize_onto(quant, device)?;
        self.up.quantize_onto(quant, device)?;
        self.down.quantize_onto(quant, device)
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let g = self.gate.forward(x)?.silu()?;
        let u = self.up.forward(x)?;
        self.down.forward(&(g * u)?)
    }
}

struct DecoderLayer {
    input_ln: RmsNorm,
    post_ln: RmsNorm,
    attn: Attention,
    mlp: Mlp,
    eps: f64,
}

impl DecoderLayer {
    fn new(cfg: &Flux2Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            input_ln: rms_norm(
                cfg.te_hidden_size,
                cfg.te_rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            post_ln: rms_norm(
                cfg.te_hidden_size,
                cfg.te_rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            attn: Attention::new(cfg, vb.pp("self_attn"))?,
            mlp: Mlp::new(cfg, vb.pp("mlp"))?,
            eps: cfg.te_rms_norm_eps,
        })
    }

    fn quantize_onto(&mut self, quant: Quant, device: &Device) -> Result<()> {
        self.attn.quantize_onto(quant, device)?;
        self.mlp.quantize_onto(quant, device)?;
        self.input_ln = rms_norm_to(&self.input_ln, self.eps, device)?;
        self.post_ln = rms_norm_to(&self.post_ln, self.eps, device)?;
        Ok(())
    }

    fn forward(&self, x: &Tensor, rotary: &Rotary, mask: &Tensor) -> Result<Tensor> {
        let h = (x + self
            .attn
            .forward(&self.input_ln.forward(x)?, rotary, mask)?)?;
        &h + self.mlp.forward(&self.post_ln.forward(&h)?)?
    }
}

/// The FLUX.2 decoder-LM prompt-embeds encoder. Backbone varies by variant (Qwen3 for klein,
/// Mistral for dev) — hence the variant-neutral name; the assembly (`DecoderLayer`s + `Rotary`) is
/// shared and dispatched off `Flux2Config`, so this single type loads/runs either tower.
pub struct Flux2PromptEncoder {
    embed_tokens: QEmbedding,
    layers: Vec<DecoderLayer>,
    rotary: Rotary,
    out_layers: [usize; 3],
    max_run: usize,
}

impl Flux2PromptEncoder {
    /// Build under `cfg.te_prefix` (klein Qwen3: `model`; dev Mistral: `language_model.model`). The
    /// final `…norm` and `lm_head` are intentionally not loaded — `prompt_embeds` uses the
    /// pre-final-norm intermediate states only. Only the first `max(out_layers)` layers are
    /// constructed (higher layers cannot affect the kept states).
    pub fn new(cfg: &Flux2Config, vb: VarBuilder) -> Result<Self> {
        let model = vb.pp(cfg.te_prefix);
        let embed_tokens = QEmbedding::detect(
            &model,
            "embed_tokens",
            cfg.te_vocab_size,
            cfg.te_hidden_size,
        )?;
        let max_run = *cfg.te_out_layers.iter().max().unwrap();
        let mut layers = Vec::with_capacity(max_run);
        let vb_layers = model.pp("layers");
        for i in 0..max_run {
            layers.push(DecoderLayer::new(cfg, vb_layers.pp(i))?);
        }
        let rotary = Rotary::new(
            cfg.te_head_dim,
            cfg.te_rope_theta,
            cfg.max_sequence_length.max(1),
            vb.device(),
        )?;
        Ok(Self {
            embed_tokens,
            layers,
            rotary,
            out_layers: cfg.te_out_layers,
            max_run,
        })
    }

    /// Fold every projection to `Q4_0`/`Q8_0` **onto `device`** and carry the token embedding, RoPE
    /// tables, and RMSNorms there too (CPU-staged dev quant path, sc-7457). Call after building the
    /// dense encoder on the CPU; afterwards the encoder runs on `device`. The token embedding stays
    /// full precision (a lookup, not a matmul) and is only moved to `device`.
    pub fn quantize(&mut self, quant: Quant, device: &Device) -> Result<()> {
        // The token embedding stays full precision when dense (a lookup, not a matmul) and is moved to
        // `device`; when it loaded packed (MLX tier), it already lives on `device` and `to_device` is a
        // no-op.
        self.embed_tokens.to_device(device)?;
        self.rotary = self.rotary.to_device(device)?;
        for layer in &mut self.layers {
            layer.quantize_onto(quant, device)?;
        }
        Ok(())
    }

    /// `input_ids` / `attention_mask`: `[B, S]` (ids u32, mask 1=real/0=pad). Returns `prompt_embeds`
    /// `[B, S, 3·hidden]` (f32): the layer-9/18/27 hidden states concatenated on the feature axis.
    /// Hidden-state index 0 = embeddings; index k = output of layer k-1.
    pub fn prompt_embeds(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let (b, s) = input_ids.dims2()?;
        // The RoPE cos/sin tables are precomputed to a fixed `max_sequence_length`; a longer sequence
        // would `narrow(0, 0, seq)` past the table end and fail with an opaque candle shape error deep
        // in `Rotary::apply`. Reject it up front with a clear length message (sc-9386, mirroring the
        // F-077 fix in krea/boogu). NOTE the public `Flux2Pipeline::encode` path already right-truncates
        // the prompt to `max_sequence_length` via the gen-core tokenizer, so this can only fire for a
        // caller that hands `prompt_embeds` raw over-length ids directly.
        check_seq_len(s, self.rotary.max_seq()?)?;
        let mask = build_mask(attention_mask, b, s, input_ids.device())?;
        let mut hidden = self.embed_tokens.forward(input_ids)?.to_dtype(DType::F32)?;

        let mut saved: Vec<(usize, Tensor)> = Vec::with_capacity(3);
        if self.out_layers.contains(&0) {
            saved.push((0, hidden.clone()));
        }
        for (i, layer) in self.layers.iter().take(self.max_run).enumerate() {
            hidden = layer.forward(&hidden, &self.rotary, &mask)?;
            let idx = i + 1;
            if self.out_layers.contains(&idx) {
                saved.push((idx, hidden.clone()));
            }
        }
        let pick = |idx: usize| -> Result<Tensor> {
            saved
                .iter()
                .find(|(k, _)| *k == idx)
                .map(|(_, v)| v.clone())
                .ok_or_else(|| {
                    candle_gen::candle_core::Error::Msg(format!(
                        "flux2 te: state {idx} not captured"
                    ))
                })
        };
        let [a, b_, c] = self.out_layers;
        Tensor::cat(&[pick(a)?, pick(b_)?, pick(c)?], D::Minus1)
    }
}

/// Validate a token-sequence length against the RoPE-table cap (sc-9386, F-077 sibling): a sequence
/// longer than `max_seq` — the rows the cos/sin tables were precomputed for — returns a clear,
/// actionable message naming the cap and the actual length, instead of the opaque `narrow` tensor
/// shape error that would otherwise surface deep in `Rotary::apply` mid-encode. Pure so it is
/// unit-testable without a real snapshot / weights.
fn check_seq_len(seq: usize, max_seq: usize) -> Result<()> {
    if seq > max_seq {
        return Err(candle_gen::candle_core::Error::Msg(format!(
            "flux2 te: prompt has {seq} tokens, exceeds max_sequence_length={max_seq} \
             (the RoPE table is sized to this cap)"
        )));
    }
    Ok(())
}

/// Additive attention mask `[B, 1, S, S]` (f32): `0` where a query `i` may attend key `j` (causal
/// `j <= i` AND `j` not padding), `-inf` otherwise. Built host-side.
fn build_mask(attention_mask: &Tensor, b: usize, s: usize, device: &Device) -> Result<Tensor> {
    let am: Vec<i64> = attention_mask
        .to_dtype(DType::I64)?
        .flatten_all()?
        .to_vec1::<i64>()?;
    let mut data = vec![0f32; b * s * s];
    for bi in 0..b {
        for i in 0..s {
            for j in 0..s {
                let allowed = j <= i && am[bi * s + j] == 1;
                if !allowed {
                    data[(bi * s + i) * s + j] = f32::NEG_INFINITY;
                }
            }
        }
    }
    Tensor::from_vec(data, (b, 1, s, s), device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_seq_len_rejects_over_cap_with_clear_message() {
        // An over-length sequence returns an actionable length error naming the cap and the actual
        // length — NOT the opaque tensor `narrow` error that would surface deep in `Rotary::apply`
        // (sc-9386, F-077 sibling).
        let err = check_seq_len(513, 512).unwrap_err().to_string();
        assert!(err.contains("513"), "names the actual length: {err}");
        assert!(
            err.contains("max_sequence_length=512"),
            "names the cap: {err}"
        );
        assert!(!err.contains("narrow"), "not an opaque tensor error: {err}");
    }

    #[test]
    fn check_seq_len_accepts_at_and_below_cap() {
        // At-limit and below-limit sequences pass validation (normal prompts are unaffected).
        assert!(check_seq_len(512, 512).is_ok());
        assert!(check_seq_len(1, 512).is_ok());
        assert!(check_seq_len(0, 512).is_ok());
    }

    #[test]
    fn rotary_max_seq_reports_table_rows_and_narrows_within_cap() -> Result<()> {
        // The guard reads `max_seq` straight off the precomputed table; a within-cap sequence still
        // narrows (byte-identically to before) while an over-cap one is what `check_seq_len` rejects.
        let dev = Device::Cpu;
        let cap = 8usize;
        let rot = Rotary::new(4, 1e4, cap, &dev)?;
        assert_eq!(rot.max_seq()?, cap);
        // within-cap narrow succeeds
        let q = Tensor::zeros((1, 2, cap - 1, 4), DType::F32, &dev)?;
        let (rq, _) = rot.apply(&q, &q)?;
        assert_eq!(rq.dims4()?, (1, 2, cap - 1, 4));
        // the guard would reject a seq past the cap
        assert!(check_seq_len(cap + 1, rot.max_seq()?).is_err());
        Ok(())
    }
}
