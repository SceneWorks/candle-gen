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
use candle_gen::candle_nn::{ops::softmax_last_dim, Linear, Module, VarBuilder};
use candle_gen::{CandleError, Result as CResult};

use crate::nn::{lin, lin_bias, rms_norm};

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
struct Attn {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
}

impl Attn {
    fn new(vb: &VarBuilder) -> CResult<Self> {
        Ok(Self {
            q: lin_bias(vb, "q_proj")?,
            k: lin_bias(vb, "k_proj")?,
            v: lin_bias(vb, "v_proj")?,
            o: lin(vb, "o_proj")?,
        })
    }
}

/// SwiGLU MLP (bias-free), the stock Qwen2 MLP.
struct Mlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl Mlp {
    fn new(vb: &VarBuilder) -> CResult<Self> {
        Ok(Self {
            gate: lin(vb, "gate_proj")?,
            up: lin(vb, "up_proj")?,
            down: lin(vb, "down_proj")?,
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
    fn new(vb: &VarBuilder) -> CResult<Self> {
        Ok(Self {
            input_ln: vb.get_unchecked("input_layernorm.weight")?,
            post_ln: vb.get_unchecked("post_attention_layernorm.weight")?,
            attn: Attn::new(&vb.pp("self_attn"))?,
            mlp: Mlp::new(&vb.pp("mlp"))?,
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
            .map(|i| Layer::new(&lvb.pp(i)))
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
        let flat = input_ids.reshape((b * l,))?.to_dtype(DType::U32)?;
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
        let (cos, sin) = self.mrope_cos_sin(position_ids, embeds.dtype())?;
        // Normalize the mask to a 4D [B?,1,L,L]-broadcastable shape.
        let mask = match mask.rank() {
            3 => mask.unsqueeze(1)?, // [1,L,L] -> [1,1,L,L]
            4 => mask.clone(),
            r => {
                return Err(CandleError::Msg(format!(
                    "qwen2_5_vl: attention mask must be 3D or 4D, got {r}D"
                )))
            }
        };
        let eps = self.cfg.rms_norm_eps;
        let mut hidden = embeds.clone();
        let mut all = Vec::with_capacity(self.layers.len() + 1);
        for layer in &self.layers {
            all.push(hidden.clone()); // HF appends the pre-layer hidden state
            let normed = rms_norm(&hidden, &layer.input_ln, eps)?;
            hidden = (&hidden + self.attention(&normed, &layer.attn, &cos, &sin, &mask)?)?;
            let normed = rms_norm(&hidden, &layer.post_ln, eps)?;
            hidden = (&hidden + layer.mlp.forward(&normed)?)?;
        }
        all.push(rms_norm(&hidden, &self.norm, eps)?);
        Ok(all)
    }

    /// The planner feature: the penultimate hidden state `hidden_states[-2]` (the residual stream
    /// feeding the final decoder layer, pre-final-norm) — `[B,L,hidden]`.
    pub fn penultimate(
        &self,
        embeds: &Tensor,
        position_ids: &Tensor,
        mask: &Tensor,
    ) -> CResult<Tensor> {
        let all = self.forward(embeds, position_ids, mask)?;
        Ok(all[all.len() - 2].clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Device;

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
}
