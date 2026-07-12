//! Native Qwen2.5-VL-7B planner **LLM backbone** — a stateless feature extractor (candle sibling of
//! `mlx-gen-bernini/src/qwen2_5_vl.rs`, sc-5132). Port of the text decoder of the Bernini planner's
//! `Qwen2_5_VLModel` (architecturally stock HF Qwen2.5-VL). The planner runs this as a single forward
//! over `inputs_embeds` and taps `hidden_states[-2]` (the penultimate residual stream — the input to
//! the final decoder layer, pre-final-norm). There is **no** token generation, KV-cache, or `lm_head`.
//!
//! Deltas vs a stock Qwen2 decoder: attention `q/k/v_proj` carry a **bias** while `o_proj` does not,
//! there is **no q/k-norm**, and the rotary is the net-new **3D multimodal RoPE**
//! (`apply_multimodal_rotary_pos_emb`) driven by externally supplied `(3, L)` position ids + an
//! externally supplied additive 4D attention mask (the planner's flex mask — text/in-vit causal,
//! gen-target bidirectional — not a hardcoded causal mask).
//!
//! Validated bit-near against `tests/fixtures/qwen_backbone_golden.safetensors` (the same synthetic
//! two-layer golden the MLX lane asserts): the MRoPE table (trig only, ~1e-4 f32 floor) and the
//! penultimate hidden state through the residual stack (~5e-3 f32 matmul floor).

use candle_gen::candle_core::{DType, Device, Tensor, D};
use candle_gen::candle_nn::{ops::softmax_last_dim, VarBuilder};
use candle_gen::quant::AdaptLinear as QLinear;
use candle_gen::{CandleError, Result as CResult};

use crate::nn::rms_norm;

/// Text-decoder config for the Qwen2.5-VL-7B planner backbone (the LLM half of `mllm/config.json`).
#[derive(Clone, Debug)]
pub struct QwenVlTextConfig {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    /// The 3 MRoPE channel sections (temporal, height, width); sum·2 = head_dim.
    pub mrope_section: [usize; 3],
}

impl Default for QwenVlTextConfig {
    /// Qwen2.5-VL-7B-Instruct (the Bernini planner base).
    fn default() -> Self {
        Self {
            hidden_size: 3584,
            num_layers: 28,
            num_heads: 28,
            num_kv_heads: 4,
            head_dim: 128,
            intermediate_size: 18944,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            mrope_section: [16, 24, 24],
        }
    }
}

impl QwenVlTextConfig {
    /// Read from a `qwen2_5_vl_config.json` (the snapshot copy of `mllm/config.json`). The text fields
    /// live at the top level; `head_dim = hidden_size / num_attention_heads`.
    pub fn from_config_json(path: &std::path::Path) -> CResult<Self> {
        let v: serde_json::Value = serde_json::from_slice(
            &std::fs::read(path).map_err(|e| CandleError::Msg(format!("read config: {e}")))?,
        )
        .map_err(|e| CandleError::Msg(format!("parse {}: {e}", path.display())))?;
        let i =
            |k: &str, d: u64| v.get(k).and_then(serde_json::Value::as_u64).unwrap_or(d) as usize;
        let f = |k: &str, d: f64| v.get(k).and_then(serde_json::Value::as_f64).unwrap_or(d);
        let hidden_size = i("hidden_size", 3584);
        let num_heads = i("num_attention_heads", 28);
        let mrope = v
            .get("rope_scaling")
            .and_then(|r| r.get("mrope_section"))
            .and_then(serde_json::Value::as_array)
            .map(|a| {
                let g = |idx: usize, d: usize| {
                    a.get(idx)
                        .and_then(serde_json::Value::as_u64)
                        .map(|x| x as usize)
                        .unwrap_or(d)
                };
                [g(0, 16), g(1, 24), g(2, 24)]
            })
            .unwrap_or([16, 24, 24]);
        Ok(Self {
            hidden_size,
            num_layers: i("num_hidden_layers", 28),
            num_heads,
            num_kv_heads: i("num_key_value_heads", 4),
            head_dim: hidden_size / num_heads,
            intermediate_size: i("intermediate_size", 18944),
            rms_norm_eps: f("rms_norm_eps", 1e-6),
            rope_theta: f("rope_theta", 1_000_000.0),
            mrope_section: mrope,
        })
    }
}

/// Per-layer attention: `q/k/v_proj` carry bias, `o_proj` does not (Qwen2.5-VL; no q/k-norm).
///
/// sc-11062: each projection is a **packed-detecting** [`QLinear`] ([`AdaptLinear`]): when the tier ships
/// the MLX-packed triple (`{proj}.scales`/`.biases` present — the q4/q8 planner the converter emits), the
/// packed base loads with no dense weight materialized; a dense bf16 tier (no `.scales`) takes the dense
/// arm byte-identically to the old `candle_nn::Linear` read (the CPU parity goldens are unaffected). Only
/// the LLM text linears quantize — the vision tower / connector / clip_diff head stay dense (mirrors the
/// MLX lane's conservative planner quant policy, sc-5146).
struct Attn {
    q: QLinear,
    k: QLinear,
    v: QLinear,
    o: QLinear,
}

impl Attn {
    fn new(vb: &VarBuilder, cfg: &QwenVlTextConfig) -> CResult<Self> {
        let h = cfg.hidden_size;
        let qd = cfg.num_heads * cfg.head_dim;
        let kvd = cfg.num_kv_heads * cfg.head_dim;
        Ok(Self {
            q: QLinear::linear_detect(h, qd, vb, "q_proj", true)?,
            k: QLinear::linear_detect(h, kvd, vb, "k_proj", true)?,
            v: QLinear::linear_detect(h, kvd, vb, "v_proj", true)?,
            o: QLinear::linear_detect(qd, h, vb, "o_proj", false)?,
        })
    }
}

/// SwiGLU MLP (bias-free), the stock Qwen2 MLP. Packed-detecting like [`Attn`] (sc-11062).
struct Mlp {
    gate: QLinear,
    up: QLinear,
    down: QLinear,
}

impl Mlp {
    fn new(vb: &VarBuilder, cfg: &QwenVlTextConfig) -> CResult<Self> {
        let h = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        Ok(Self {
            gate: QLinear::linear_detect(h, inter, vb, "gate_proj", false)?,
            up: QLinear::linear_detect(h, inter, vb, "up_proj", false)?,
            down: QLinear::linear_detect(inter, h, vb, "down_proj", false)?,
        })
    }

    fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        let gated = (self.gate.forward(x)?.silu()? * self.up.forward(x)?)?;
        Ok(self.down.forward(&gated)?)
    }
}

struct Layer {
    input_ln: Tensor,
    post_ln: Tensor,
    attn: Attn,
    mlp: Mlp,
}

impl Layer {
    fn new(vb: &VarBuilder, cfg: &QwenVlTextConfig) -> CResult<Self> {
        Ok(Self {
            input_ln: vb.get_unchecked("input_layernorm.weight")?,
            post_ln: vb.get_unchecked("post_attention_layernorm.weight")?,
            attn: Attn::new(&vb.pp("self_attn"), cfg)?,
            mlp: Mlp::new(&vb.pp("mlp"), cfg)?,
        })
    }
}

/// HF half-split rotary `rotate_half`: `cat(-x[d/2:], x[:d/2])` on the last axis.
fn rotate_half(x: &Tensor) -> CResult<Tensor> {
    let d = x.dim(D::Minus1)?;
    let half = d / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, d - half)?;
    Ok(Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?)
}

/// The native Qwen2.5-VL-7B text decoder, run as a stateless penultimate-hidden-state extractor.
pub struct Qwen25VlText {
    embed_tokens: Tensor,
    layers: Vec<Layer>,
    norm: Tensor,
    cfg: QwenVlTextConfig,
}

impl Qwen25VlText {
    /// Build from a `VarBuilder` **already rooted at the model namespace** (`model.*` for the snapshot
    /// layout, `w.model.*` for the golden fixture). Reads `embed_tokens.weight`, `layers.{i}.*`,
    /// `norm.weight`.
    pub fn new(cfg: QwenVlTextConfig, vb: VarBuilder) -> CResult<Self> {
        let lvb = vb.pp("layers");
        let layers = (0..cfg.num_layers)
            .map(|i| Layer::new(&lvb.pp(i), &cfg))
            .collect::<CResult<Vec<_>>>()?;
        Ok(Self {
            embed_tokens: vb.get_unchecked("embed_tokens.weight")?,
            layers,
            norm: vb.get_unchecked("norm.weight")?,
            cfg,
        })
    }

    pub fn config(&self) -> &QwenVlTextConfig {
        &self.cfg
    }

    /// The device the backbone weights live on. Planner-side host tensors (token ids, MRoPE position
    /// ids, the additive attention mask) must be moved onto this device before the forward, otherwise
    /// `embed_tokens.index_select`, `mrope_cos_sin`, and `scores.broadcast_add(mask)` hard-error with
    /// `DeviceMismatchBinaryOp` on CUDA (sc-11148 / F-079).
    pub fn device(&self) -> &Device {
        self.embed_tokens.device()
    }

    /// Token embedding: `input_ids` `[B,L]` (int) → `[B,L,hidden]` (the reference's
    /// `get_input_embeddings()`).
    pub fn embed(&self, input_ids: &Tensor) -> CResult<Tensor> {
        let (b, l) = input_ids.dims2()?;
        // The token ids are host-assembled on CPU (`format_mllm_inputs_embeds`); the embedding weight
        // lives on the model device (CUDA when rendering). `index_select` requires both operands on the
        // same device — move the indices to the weight's device (a no-op on CPU, so the CPU parity
        // goldens stay bit-identical). sc-11003: the planner GPU path was never exercised (CPU-only
        // parity), so this device skew only surfaces on-device.
        let flat = input_ids
            .reshape((b * l,))?
            .to_dtype(DType::U32)?
            .to_device(self.embed_tokens.device())?;
        let h = self.embed_tokens.dim(1)?;
        let g = self.embed_tokens.index_select(&flat, 0)?;
        Ok(g.reshape((b, l, h))?)
    }

    /// Assemble the multimodal rotary `(cos, sin)` for the layer apply, each `[1, L, head_dim]` in
    /// `dtype`. `position_ids` is `[3, L]` int (temporal / height / width rows).
    ///
    /// Mirrors `Qwen2_5_VLRotaryEmbedding` + the channel interleave of
    /// `apply_multimodal_rotary_pos_emb`: one rotary table per axis (`inv_freq[j] =
    /// theta^(-2j/head_dim)`, `emb = cat(freqs, freqs)`), then the `head_dim` channels are stitched from
    /// the three axes by the doubled `mrope_section` `[16,24,24,16,24,24]`, chunk `i` taking axis `i%3`.
    pub fn mrope_cos_sin(&self, position_ids: &Tensor, dtype: DType) -> CResult<(Tensor, Tensor)> {
        let dev = position_ids.device();
        let hd = self.cfg.head_dim;
        let half = hd / 2;
        let theta = self.cfg.rope_theta;
        let inv_freq: Vec<f32> = (0..half)
            .map(|j| (1.0 / theta.powf((2 * j) as f64 / hd as f64)) as f32)
            .collect();
        let inv = Tensor::from_vec(inv_freq, (1, half), dev)?; // [1, half]

        let l = position_ids.dim(1)?;
        let pos_f32 = position_ids.to_dtype(DType::F32)?; // [3, L]

        // Doubled-section widths [16,24,24,16,24,24]; chunk i taken from axis i%3.
        let s = self.cfg.mrope_section;
        let doubled = [s[0], s[1], s[2], s[0], s[1], s[2]];

        // One rotary table per axis (its own position row), sliced into the 6 channel pieces.
        let mut cos_pieces: Vec<Vec<Tensor>> = Vec::with_capacity(3);
        let mut sin_pieces: Vec<Vec<Tensor>> = Vec::with_capacity(3);
        for row in 0..3 {
            let p = pos_f32.narrow(0, row, 1)?.reshape((l, 1))?; // [L, 1]
            let freqs = p.matmul(&inv)?; // [L, half]
            let emb = Tensor::cat(&[&freqs, &freqs], 1)?; // [L, head_dim]
            let cos = emb.cos()?.reshape((1, l, hd))?; // [1, L, head_dim]
            let sin = emb.sin()?.reshape((1, l, hd))?;
            let mut cp = Vec::with_capacity(6);
            let mut sp = Vec::with_capacity(6);
            let mut off = 0usize;
            for &w in doubled.iter() {
                cp.push(cos.narrow(2, off, w)?);
                sp.push(sin.narrow(2, off, w)?);
                off += w;
            }
            cos_pieces.push(cp);
            sin_pieces.push(sp);
        }
        // Stitch: channel chunk `i` is taken from axis `i % 3` (`apply_multimodal_rotary_pos_emb`).
        let cos_sel: Vec<Tensor> = (0..6).map(|i| cos_pieces[i % 3][i].clone()).collect();
        let sin_sel: Vec<Tensor> = (0..6).map(|i| sin_pieces[i % 3][i].clone()).collect();
        let cos = Tensor::cat(&cos_sel.iter().collect::<Vec<_>>(), 2)?.to_dtype(dtype)?;
        let sin = Tensor::cat(&sin_sel.iter().collect::<Vec<_>>(), 2)?.to_dtype(dtype)?;
        Ok((cos, sin))
    }

    /// Apply MRoPE to a `[B, L, H, head_dim]` projection given assembled `cos`/`sin` `[1, L, head_dim]`
    /// (broadcast over the head axis): `x*cos + rotate_half(x)*sin`.
    fn apply_mrope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> CResult<Tensor> {
        let cos = cos.unsqueeze(2)?; // [1, L, 1, head_dim]
        let sin = sin.unsqueeze(2)?;
        Ok((x.broadcast_mul(&cos)? + rotate_half(x)?.broadcast_mul(&sin)?)?)
    }

    /// Expand `[B,L,Hkv,D]` → `[B,L,Hkv*groups,D]` (GQA repeat).
    fn repeat_kv(x: &Tensor, groups: usize) -> CResult<Tensor> {
        if groups == 1 {
            return Ok(x.clone());
        }
        let (b, l, hkv, d) = x.dims4()?;
        let x = x.unsqueeze(3)?; // [B,L,Hkv,1,D]
        let x = x.broadcast_as((b, l, hkv, groups, d))?;
        Ok(x.reshape((b, l, hkv * groups, d))?.contiguous()?)
    }

    /// Eager attention with an external additive 4D `mask` (`[1,1,L,L]` broadcast): q/k/v project →
    /// reshape to heads → MRoPE q,k → GQA → `softmax(q·kᵀ/√d + mask)·v` → o_proj.
    fn attention(
        &self,
        x: &Tensor,
        a: &Attn,
        cos: &Tensor,
        sin: &Tensor,
        mask: &Tensor,
    ) -> CResult<Tensor> {
        let (b, s, _) = x.dims3()?;
        let hd = self.cfg.head_dim;
        let (nh, nkv) = (self.cfg.num_heads, self.cfg.num_kv_heads);

        let q = a.q.forward(x)?.reshape((b, s, nh, hd))?;
        let k = a.k.forward(x)?.reshape((b, s, nkv, hd))?;
        let v = a.v.forward(x)?.reshape((b, s, nkv, hd))?;
        let q = Self::apply_mrope(&q, cos, sin)?;
        let k = Self::apply_mrope(&k, cos, sin)?;

        let groups = nh / nkv;
        let q = q.permute((0, 2, 1, 3))?.contiguous()?; // [B,H,L,D]
        let k = Self::repeat_kv(&k, groups)?
            .permute((0, 2, 1, 3))?
            .contiguous()?;
        let v = Self::repeat_kv(&v, groups)?
            .permute((0, 2, 1, 3))?
            .contiguous()?;

        let scale = (hd as f64).powf(-0.5);
        let scores = (q.matmul(&k.transpose(D::Minus1, D::Minus2)?.contiguous()?)? * scale)?;
        // mask is [1,1,L,L] (broadcast over B and H).
        let scores = scores.broadcast_add(mask)?;
        let weights = softmax_last_dim(&scores)?;
        let out = weights
            .matmul(&v)?
            .permute((0, 2, 1, 3))?
            .contiguous()?
            .reshape((b, s, nh * hd))?;
        Ok(a.o.forward(&out)?)
    }

    /// Full stateless forward. `embeds` `[B,L,hidden]`; `position_ids` `[3,L]` int; `mask` an additive
    /// 4D attention mask (`[1,1,L,L]`, `0`/`-inf`).
    ///
    /// Returns **all** hidden states exactly as HF `output_hidden_states=True`: `[embeds, layer0_out,
    /// …, layer_{N-2}_out, final_norm(layer_{N-1}_out)]` — `N+1` entries. The planner's penultimate tap
    /// is [`Self::penultimate`] (`[-2]` = the input to the final decoder layer).
    pub fn forward(
        &self,
        embeds: &Tensor,
        position_ids: &Tensor,
        mask: &Tensor,
    ) -> CResult<Vec<Tensor>> {
        let (cos, sin, mask) = self.prepare_rotary_and_mask(embeds, position_ids, mask)?;
        let eps = self.cfg.rms_norm_eps;
        let mut hidden = embeds.clone();
        let mut all = Vec::with_capacity(self.layers.len() + 1);
        for layer in &self.layers {
            all.push(hidden.clone()); // HF appends the pre-layer hidden state
            hidden = self.run_layer(&hidden, layer, &cos, &sin, &mask, eps)?;
        }
        all.push(rms_norm(&hidden, &self.norm, eps)?);
        Ok(all)
    }

    /// Assemble the MRoPE `(cos, sin)` and normalize the additive attention `mask` to a 4D
    /// `[B?,1,L,L]`-broadcastable shape — the per-forward setup shared by [`Self::forward`] and
    /// [`Self::penultimate`].
    ///
    /// `embeds` is authoritative for the device (it came off the embedding weight). The host-assembled
    /// `position_ids` (mrope) and additive `mask` are built on CPU, so move them onto the embeds device
    /// before they meet GPU tensors in `mrope_cos_sin` / `broadcast_add` (a no-op on CPU → parity
    /// goldens unchanged). sc-11003 planner-on-GPU device fix.
    fn prepare_rotary_and_mask(
        &self,
        embeds: &Tensor,
        position_ids: &Tensor,
        mask: &Tensor,
    ) -> CResult<(Tensor, Tensor, Tensor)> {
        let dev = embeds.device();
        let position_ids = position_ids.to_device(dev)?;
        let mask = mask.to_device(dev)?;
        let (cos, sin) = self.mrope_cos_sin(&position_ids, embeds.dtype())?;
        let mask = match mask.rank() {
            3 => mask.unsqueeze(1)?, // [1,L,L] -> [1,1,L,L]
            4 => mask,
            r => {
                return Err(CandleError::Msg(format!(
                    "qwen2_5_vl: attention mask must be 3D or 4D, got {r}D"
                )))
            }
        };
        Ok((cos, sin, mask))
    }

    /// One decoder layer over the residual stream: pre-attn RMSNorm → self-attention residual →
    /// post-attn RMSNorm → MLP residual. Factored so [`Self::forward`] and [`Self::penultimate`] run the
    /// identical block math.
    fn run_layer(
        &self,
        hidden: &Tensor,
        layer: &Layer,
        cos: &Tensor,
        sin: &Tensor,
        mask: &Tensor,
        eps: f64,
    ) -> CResult<Tensor> {
        let normed = rms_norm(hidden, &layer.input_ln, eps)?;
        let hidden = (hidden + self.attention(&normed, &layer.attn, cos, sin, mask)?)?;
        let normed = rms_norm(&hidden, &layer.post_ln, eps)?;
        Ok((&hidden + layer.mlp.forward(&normed)?)?)
    }

    /// The planner feature: the penultimate hidden state `hidden_states[-2]` (the residual stream
    /// feeding the final decoder layer, pre-final-norm) — `[B,L,hidden]`.
    ///
    /// F-143: only `[-2]` is ever consumed, so this walks the residual stack keeping a single hidden
    /// state instead of retaining all `N+1` `[B,L,hidden]` states like [`Self::forward`] does (~1 GB
    /// bf16 apiece for the 29-layer 7B backbone, allocated 75+× per generate). `hidden_states[-2]` is
    /// the input to the *final* decoder layer, i.e. the residual stream after the first `N-1` layers —
    /// the last layer's output and the final norm only affect `[-1]`, so they are skipped entirely. The
    /// result is bit-identical to `forward(...)[len-2]` (see the `penultimate_matches_forward` golden);
    /// [`Self::forward`] is retained as the full-Vec HF-parity API for tests.
    pub fn penultimate(
        &self,
        embeds: &Tensor,
        position_ids: &Tensor,
        mask: &Tensor,
    ) -> CResult<Tensor> {
        let (cos, sin, mask) = self.prepare_rotary_and_mask(embeds, position_ids, mask)?;
        let eps = self.cfg.rms_norm_eps;
        // `hidden_states[-2]` with an empty stack is `embeds` itself (all = [embeds, norm(embeds)]).
        // The final layer + final norm only affect `[-1]`, so they are dropped from `leading` and never
        // run.
        let Some((_last, leading)) = self.layers.split_last() else {
            return Ok(embeds.clone());
        };
        let mut hidden = embeds.clone();
        for layer in leading {
            hidden = self.run_layer(&hidden, layer, &cos, &sin, &mask, eps)?;
        }
        Ok(hidden)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Device;
    use std::collections::HashMap;

    /// The default config matches Qwen2.5-VL-7B (head_dim derived = 128, GQA groups = 7).
    #[test]
    fn config_shapes() {
        let c = QwenVlTextConfig::default();
        assert_eq!(c.head_dim, c.hidden_size / c.num_heads);
        assert_eq!(c.head_dim, 128);
        assert_eq!(c.num_heads / c.num_kv_heads, 7);
        assert_eq!(c.mrope_section.iter().sum::<usize>() * 2, c.head_dim);
    }

    /// rotate_half is the NeoX half-split: `[a,b,c,d] → [-c,-d,a,b]`.
    #[test]
    fn rotate_half_neox() {
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 1, 1, 4), &Device::Cpu).unwrap();
        let r = rotate_half(&x).unwrap();
        let got: Vec<f32> = r.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(got, vec![-3.0, -4.0, 1.0, 2.0]);
    }

    /// MRoPE channel stitch: for a text segment all three position rows are equal, so the assembled
    /// cos/sin must equal a plain 1D rotary table (the reference note: "the three rotary position index
    /// of text embedding is always the same → no difference with modern LLMs").
    #[test]
    fn mrope_text_equals_1d() {
        let cfg = QwenVlTextConfig::default();
        let backbone = Qwen25VlText {
            embed_tokens: Tensor::zeros((8, cfg.hidden_size), DType::F32, &Device::Cpu).unwrap(),
            layers: Vec::new(),
            norm: Tensor::ones((cfg.hidden_size,), DType::F32, &Device::Cpu).unwrap(),
            cfg: cfg.clone(),
        };
        let l = 5usize;
        let row: Vec<i64> = (0..l as i64).collect();
        let mut data = Vec::new();
        for _ in 0..3 {
            data.extend_from_slice(&row);
        }
        let pos = Tensor::from_vec(data, (3, l), &Device::Cpu).unwrap();
        let (cos, _sin) = backbone.mrope_cos_sin(&pos, DType::F32).unwrap();
        assert_eq!(cos.dims(), &[1, l, cfg.head_dim]);

        // A plain 1D rotary table over the same positions.
        let half = cfg.head_dim / 2;
        let inv: Vec<f32> = (0..half)
            .map(|j| (1.0 / cfg.rope_theta.powf((2 * j) as f64 / cfg.head_dim as f64)) as f32)
            .collect();
        let inv = Tensor::from_vec(inv, (1, half), &Device::Cpu).unwrap();
        let p = Tensor::from_vec(
            row.iter().map(|&x| x as f32).collect::<Vec<_>>(),
            (l, 1),
            &Device::Cpu,
        )
        .unwrap();
        let freqs = p.matmul(&inv).unwrap();
        let emb = Tensor::cat(&[&freqs, &freqs], 1).unwrap();
        let cos1d = emb.cos().unwrap().reshape((1, l, cfg.head_dim)).unwrap();

        let diff = (&cos - &cos1d)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            diff < 1e-6,
            "text MRoPE must equal 1D rotary, max|Δ|={diff}"
        );
    }

    /// A tiny synthetic backbone (head_dim 8 = 2 heads × 4, 1 kv head, mrope [1,2,1]) with `n` real
    /// decoder layers, runnable on CPU — the residual stack without the 7B weights.
    fn tiny_backbone(n: usize, dev: &Device) -> (Qwen25VlText, QwenVlTextConfig) {
        let cfg = QwenVlTextConfig {
            hidden_size: 16,
            num_layers: n,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 8,
            intermediate_size: 32,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            mrope_section: [1, 2, 1],
        };
        let mut m: HashMap<String, Tensor> = HashMap::new();
        let mut put = |k: &str, shape: &[usize]| {
            m.insert(
                k.to_string(),
                Tensor::randn(0f32, 0.2f32, shape, dev).unwrap(),
            );
        };
        let h = cfg.hidden_size;
        let (nh, nkv, hd) = (cfg.num_heads, cfg.num_kv_heads, cfg.head_dim);
        put("embed_tokens.weight", &[8, h]);
        put("norm.weight", &[h]);
        for i in 0..n {
            let b = format!("layers.{i}");
            put(&format!("{b}.input_layernorm.weight"), &[h]);
            put(&format!("{b}.post_attention_layernorm.weight"), &[h]);
            put(&format!("{b}.self_attn.q_proj.weight"), &[nh * hd, h]);
            put(&format!("{b}.self_attn.q_proj.bias"), &[nh * hd]);
            put(&format!("{b}.self_attn.k_proj.weight"), &[nkv * hd, h]);
            put(&format!("{b}.self_attn.k_proj.bias"), &[nkv * hd]);
            put(&format!("{b}.self_attn.v_proj.weight"), &[nkv * hd, h]);
            put(&format!("{b}.self_attn.v_proj.bias"), &[nkv * hd]);
            put(&format!("{b}.self_attn.o_proj.weight"), &[h, nh * hd]);
            put(
                &format!("{b}.mlp.gate_proj.weight"),
                &[cfg.intermediate_size, h],
            );
            put(
                &format!("{b}.mlp.up_proj.weight"),
                &[cfg.intermediate_size, h],
            );
            put(
                &format!("{b}.mlp.down_proj.weight"),
                &[h, cfg.intermediate_size],
            );
        }
        let vb = VarBuilder::from_tensors(m, DType::F32, dev);
        (Qwen25VlText::new(cfg.clone(), vb).unwrap(), cfg)
    }

    fn max_abs(a: &Tensor, b: &Tensor) -> f32 {
        (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    /// F-143: the memory-lean [`Qwen25VlText::penultimate`] must return **bit-identical** results to the
    /// full-Vec `forward(...)[len-2]` it replaces — the last decoder layer and the final norm only feed
    /// `[-1]`, so skipping them cannot perturb `[-2]`. Checked across a multi-layer stack and the
    /// empty-stack edge (where `[-2]` is `embeds` itself).
    #[test]
    fn penultimate_matches_forward() {
        let dev = Device::Cpu;
        let l = 6usize;
        // MRoPE position ids [3, L] (all axes share the text row) + an all-attend additive mask.
        let row: Vec<i64> = (0..l as i64).collect();
        let mut pdata = Vec::new();
        for _ in 0..3 {
            pdata.extend_from_slice(&row);
        }
        let mk_inputs = |cfg: &QwenVlTextConfig| {
            let embeds = Tensor::randn(0f32, 1f32, (1, l, cfg.hidden_size), &dev).unwrap();
            let pos = Tensor::from_vec(pdata.clone(), (3, l), &dev).unwrap();
            let mask = Tensor::zeros((1, 1, l, l), DType::F32, &dev).unwrap();
            (embeds, pos, mask)
        };
        // `hidden_states[-2]` is only defined with ≥2 hidden states, i.e. ≥1 decoder layer.
        for n in [1usize, 2, 3] {
            let (bb, cfg) = tiny_backbone(n, &dev);
            let (embeds, pos, mask) = mk_inputs(&cfg);
            let all = bb.forward(&embeds, &pos, &mask).unwrap();
            assert_eq!(all.len(), n + 1, "forward keeps N+1 hidden states");
            let want = &all[all.len() - 2];
            let got = bb.penultimate(&embeds, &pos, &mask).unwrap();
            assert_eq!(got.dims(), want.dims());
            assert_eq!(
                max_abs(&got, want),
                0.0,
                "penultimate (n={n}) must equal forward()[-2] bit-for-bit"
            );
        }
        // Degenerate empty stack: `forward` yields a single state, so `[-2]` is undefined; the lean
        // path returns `embeds` (all = [embeds, norm(embeds)] → the pre-final-norm state is `embeds`).
        let (bb0, cfg0) = tiny_backbone(0, &dev);
        let (embeds, pos, mask) = mk_inputs(&cfg0);
        let got0 = bb0.penultimate(&embeds, &pos, &mask).unwrap();
        assert_eq!(max_abs(&got0, &embeds), 0.0, "empty stack → embeds");
    }

    // --- sc-11062: packed-detect planner LLM linears --------------------------------------------

    /// Build the MLX-packed triple (`{name}.weight` u32 codes + `.scales` + `.biases`) for a random
    /// dense `[out, in]` weight at group 64, alongside the affine grid it dequantizes to (the exact
    /// reference the packed forward reproduces). `in` must be a multiple of 64.
    fn pack_into(
        map: &mut HashMap<String, Tensor>,
        name: &str,
        out_dim: usize,
        in_dim: usize,
        dev: &Device,
    ) -> Tensor {
        let w = Tensor::randn(0f32, 0.1f32, (out_dim, in_dim), dev).unwrap();
        let (wq, scales, biases) =
            candle_gen::quant::pack_mlx_affine(&w, 4, 64).expect("pack q4 group-64");
        // The affine grid the packed codes dequantize to (scale·q + bias) — the packed forward target.
        let grid = candle_gen::quant::dequant_mlx_q4_reference_gs(&wq, &scales, &biases, 64)
            .expect("dequant grid")
            .to_device(dev)
            .unwrap();
        map.insert(format!("{name}.weight"), wq);
        map.insert(format!("{name}.scales"), scales);
        map.insert(format!("{name}.biases"), biases);
        grid
    }

    fn cosine(a: &Tensor, b: &Tensor) -> f32 {
        let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (x, y) in a.iter().zip(&b) {
            dot += (*x as f64) * (*y as f64);
            na += (*x as f64) * (*x as f64);
            nb += (*y as f64) * (*y as f64);
        }
        (dot / (na.sqrt() * nb.sqrt() + 1e-12)) as f32
    }

    /// sc-11062: when the tier ships the MLX-packed triple for the Qwen2.5-VL LLM linears, [`Attn`] and
    /// [`Mlp`] load **packed** (no dense weight materialized) and their forward reproduces the affine
    /// grid the pack represents; a dense tier (no `.scales`) loads dense. Group-64-aligned dims (hidden
    /// 64, intermediate 128) mirror the real 7B backbone (hidden 3584 = 56·64, intermediate 18944 =
    /// 296·64, kv 512 = 8·64 all group-64-aligned).
    #[test]
    fn attn_and_mlp_load_packed_when_scales_present() {
        use candle_gen::candle_core::safetensors::MmapedSafetensors;
        let dev = Device::Cpu;
        let cfg = QwenVlTextConfig {
            hidden_size: 64,
            num_layers: 1,
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 64,
            intermediate_size: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            mrope_section: [16, 8, 8],
        };
        let (h, qd, kvd, inter) = (
            cfg.hidden_size,
            cfg.num_heads * cfg.head_dim,
            cfg.num_kv_heads * cfg.head_dim,
            cfg.intermediate_size,
        );

        let mut map: HashMap<String, Tensor> = HashMap::new();
        let attn = "self_attn";
        let grids = [
            (
                "q",
                pack_into(&mut map, &format!("{attn}.q_proj"), qd, h, &dev),
            ),
            (
                "k",
                pack_into(&mut map, &format!("{attn}.k_proj"), kvd, h, &dev),
            ),
            (
                "v",
                pack_into(&mut map, &format!("{attn}.v_proj"), kvd, h, &dev),
            ),
            (
                "o",
                pack_into(&mut map, &format!("{attn}.o_proj"), h, qd, &dev),
            ),
            ("gate", pack_into(&mut map, "mlp.gate_proj", inter, h, &dev)),
            ("up", pack_into(&mut map, "mlp.up_proj", inter, h, &dev)),
            ("down", pack_into(&mut map, "mlp.down_proj", h, inter, &dev)),
        ];
        // q/k/v carry a dense bias (the tier ships it alongside the packed triple).
        for (proj, out) in [("q_proj", qd), ("k_proj", kvd), ("v_proj", kvd)] {
            map.insert(
                format!("{attn}.{proj}.bias"),
                Tensor::randn(0f32, 0.1f32, (out,), &dev).unwrap(),
            );
        }

        let tmp = std::env::temp_dir().join(format!(
            "sc11062_planner_packed_{}.safetensors",
            std::process::id()
        ));
        candle_gen::candle_core::safetensors::save(&map, &tmp).unwrap();
        // SAFETY: freshly written, single-reader for the test.
        let st = unsafe { MmapedSafetensors::new(&tmp).unwrap() };
        let vb = VarBuilder::from_backend(Box::new(st), DType::F32, dev.clone());

        let a = Attn::new(&vb.pp(attn), &cfg).expect("packed attn");
        let m = Mlp::new(&vb.pp("mlp"), &cfg).expect("packed mlp");
        for (name, packed) in [
            ("q", a.q.is_packed()),
            ("k", a.k.is_packed()),
            ("v", a.v.is_packed()),
            ("o", a.o.is_packed()),
            ("gate", m.gate.is_packed()),
            ("up", m.up.is_packed()),
            ("down", m.down.is_packed()),
        ] {
            assert!(packed, "{name} must load packed when `.scales` present");
        }

        // The bias-free packed projections' forward reproduces their affine grid (dequant-on-forward).
        // q/k/v carry a bias the packed base folds in, so they are compared only for packed detection.
        use candle_gen::candle_nn::Module;
        let bias_free: [(&str, &QLinear); 4] = [
            ("o", &a.o),
            ("gate", &m.gate),
            ("up", &m.up),
            ("down", &m.down),
        ];
        for (name, lin) in bias_free {
            let grid = &grids.iter().find(|(n, _)| *n == name).unwrap().1;
            let in_dim = grid.dim(1).unwrap();
            let x = Tensor::randn(0f32, 1f32, (3, in_dim), &dev).unwrap();
            let dense = candle_gen::candle_nn::Linear::new(grid.clone(), None);
            let got = lin.forward(&x).unwrap();
            let want = dense.forward(&x).unwrap();
            let cos = cosine(&got, &want);
            assert!(
                cos > 0.9999,
                "{name} packed forward vs grid cosine {cos:.6}"
            );
        }

        // A dense tier (no `.scales`) still loads dense — the byte-identical legacy path.
        let mut dmap: HashMap<String, Tensor> = HashMap::new();
        dmap.insert(
            format!("{attn}.q_proj.weight"),
            Tensor::randn(0f32, 0.1f32, (qd, h), &dev).unwrap(),
        );
        dmap.insert(
            format!("{attn}.q_proj.bias"),
            Tensor::randn(0f32, 0.1f32, (qd,), &dev).unwrap(),
        );
        dmap.insert(
            format!("{attn}.k_proj.weight"),
            Tensor::randn(0f32, 0.1f32, (kvd, h), &dev).unwrap(),
        );
        dmap.insert(
            format!("{attn}.k_proj.bias"),
            Tensor::randn(0f32, 0.1f32, (kvd,), &dev).unwrap(),
        );
        dmap.insert(
            format!("{attn}.v_proj.weight"),
            Tensor::randn(0f32, 0.1f32, (kvd, h), &dev).unwrap(),
        );
        dmap.insert(
            format!("{attn}.v_proj.bias"),
            Tensor::randn(0f32, 0.1f32, (kvd,), &dev).unwrap(),
        );
        dmap.insert(
            format!("{attn}.o_proj.weight"),
            Tensor::randn(0f32, 0.1f32, (h, qd), &dev).unwrap(),
        );
        let dtmp = std::env::temp_dir().join(format!(
            "sc11062_planner_dense_{}.safetensors",
            std::process::id()
        ));
        candle_gen::candle_core::safetensors::save(&dmap, &dtmp).unwrap();
        // SAFETY: freshly written, single-reader.
        let dst = unsafe { MmapedSafetensors::new(&dtmp).unwrap() };
        let dvb = VarBuilder::from_backend(Box::new(dst), DType::F32, dev.clone());
        let da = Attn::new(&dvb.pp(attn), &cfg).expect("dense attn");
        assert!(!da.q.is_packed(), "no `.scales` ⇒ dense q");
        assert!(!da.o.is_packed(), "no `.scales` ⇒ dense o");

        std::fs::remove_file(&tmp).ok();
        std::fs::remove_file(&dtmp).ok();
    }
}
