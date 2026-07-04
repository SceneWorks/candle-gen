//! Vendored, **packed-detecting** SDXL CLIP text-encoder tower (sc-9527, sc-9089j follow-up to the
//! sc-9416 UNet packed-load).
//!
//! A faithful copy of candle-transformers `stable_diffusion::clip::ClipTextTransformer` at the
//! workspace candle pin (`c1e6756`), vendored for the SAME reason the UNet was (sc-9416): the
//! `SceneWorks/sdxl-base-mlx` q4/q8 tiers pack the dual CLIP text encoders — each
//! `text_encoder{,_2}/config.json` carries a `quantization` block and every attention / MLP `Linear`
//! under `model.safetensors` is stored as the packed MLX triple `{weight u32, scales, biases}` — but
//! the stock `ClipTextTransformer` builds its projections from opaque `candle_nn::Linear` that cannot
//! consume packed u32 codes. This copy routes every Linear on the packed surface (attention
//! `q/k/v/out_proj`, MLP `fc1/fc2`, and the bigG final `text_projection`) through the shared
//! [`candle_gen::quant`] seam ([`QLinear::linear_detect_gs`]): a **pure superset** of the stock
//! tower — absent a `.scales` sibling it takes the plain dense path unchanged (so a dense diffusers
//! snapshot is byte-identical to the stock build, pinned by the vendored-vs-stock parity test), and
//! present it builds the quantized projection straight from the packed parts (no dense staging).
//!
//! Only the Linear surface is swapped; the token/position embeddings, the LayerNorms and the entire
//! forward (last-hidden-state [`ClipTextTransformer::forward_with_mask`] AND the penultimate
//! [`ClipTextTransformer::forward_until_encoder_layer`] the InstantID conditioner uses) are kept
//! byte-faithful to upstream so this copy can be re-diffed on a candle re-pin. The MLX SDXL tiers
//! pack the CLIP embeddings **dense** (only the Linears carry `.scales`), so the embeddings stay
//! `candle_nn::Embedding`.
//!
//! **`group_size` is threaded** from the parsed [`candle_gen::quant::PackedConfig`] through every block
//! constructor (per sc-9474/sc-9410) rather than hardcoded: the SDXL tiers pack at the default 64
//! today, but the seam repacks at whatever the component `config.json` declares.

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn as nn;
use candle_nn::{Module, VarBuilder};

use candle_gen::quant::{QLinear, MLX_GROUP_SIZE};

/// CLIP activation (byte-faithful copy of the stock `clip::Activation`).
#[derive(Debug, Clone, Copy)]
pub enum Activation {
    QuickGelu,
    Gelu,
    GeluErf,
}

impl Module for Activation {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Activation::QuickGelu => xs * nn::ops::sigmoid(&(xs * 1.702f64)?)?,
            Activation::Gelu => xs.gelu(),
            Activation::GeluErf => xs.gelu_erf(),
        }
    }
}

/// The CLIP text config — a local copy of the stock `clip::Config` (whose fields are private, so the
/// vendored tower cannot read them off it). The `sdxl()` / `sdxl2()` builders reproduce the exact
/// values the stock `Config::sdxl()` / `Config::sdxl2()` use (CLIP-L / OpenCLIP bigG), so a vendored
/// tower built from either is numerically identical to the stock one on shared weights.
#[derive(Debug, Clone)]
pub struct Config {
    pub vocab_size: usize,
    /// aka `hidden_size`.
    pub embed_dim: usize,
    pub activation: Activation,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    /// The character to use for padding; EOS when `None`.
    pub pad_with: Option<String>,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    /// The pooled `text_projection` output dim (bigG's `CLIPTextModelWithProjection` head).
    pub projection_dim: usize,
}

impl Config {
    /// CLIP-L / ViT-L (`text_encoder/`) — matches the stock `clip::Config::sdxl()`.
    pub fn sdxl() -> Self {
        Self {
            vocab_size: 49408,
            embed_dim: 768,
            intermediate_size: 3072,
            max_position_embeddings: 77,
            pad_with: Some("!".to_string()),
            num_hidden_layers: 12,
            num_attention_heads: 12,
            projection_dim: 768,
            activation: Activation::QuickGelu,
        }
    }

    /// OpenCLIP bigG (`text_encoder_2/`) — matches the stock `clip::Config::sdxl2()`.
    pub fn sdxl2() -> Self {
        Self {
            vocab_size: 49408,
            embed_dim: 1280,
            intermediate_size: 5120,
            max_position_embeddings: 77,
            pad_with: Some("!".to_string()),
            num_hidden_layers: 32,
            num_attention_heads: 20,
            projection_dim: 1280,
            activation: Activation::Gelu,
        }
    }
}

/// Token + position embeddings — kept **dense** (`candle_nn::Embedding`): the MLX SDXL tiers pack only
/// the Linear surface, the embeddings ship as plain bf16 tables (byte-faithful to stock).
#[derive(Debug)]
struct ClipTextEmbeddings {
    token_embedding: nn::Embedding,
    position_embedding: nn::Embedding,
    position_ids: Tensor,
}

impl ClipTextEmbeddings {
    fn new(vs: VarBuilder, c: &Config) -> Result<Self> {
        let token_embedding = nn::embedding(c.vocab_size, c.embed_dim, vs.pp("token_embedding"))?;
        let position_embedding = nn::embedding(
            c.max_position_embeddings,
            c.embed_dim,
            vs.pp("position_embedding"),
        )?;
        let position_ids =
            Tensor::arange(0u32, c.max_position_embeddings as u32, vs.device())?.unsqueeze(0)?;
        Ok(ClipTextEmbeddings {
            token_embedding,
            position_embedding,
            position_ids,
        })
    }
}

impl Module for ClipTextEmbeddings {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let token_embedding = self.token_embedding.forward(xs)?;
        let position_embedding = self.position_embedding.forward(&self.position_ids)?;
        token_embedding.broadcast_add(&position_embedding)
    }
}

/// CLIP self-attention. The four projections (`q/k/v/out_proj`) packed-detect through
/// [`QLinear::linear_detect_gs`]; the forward is byte-faithful to stock (`shape` reshape, f32
/// score/softmax, causal-mask broadcast-add).
#[derive(Debug)]
struct ClipAttention {
    k_proj: QLinear,
    v_proj: QLinear,
    q_proj: QLinear,
    out_proj: QLinear,
    head_dim: usize,
    scale: f64,
    num_attention_heads: usize,
}

impl ClipAttention {
    fn new(vs: VarBuilder, c: &Config, group_size: usize) -> Result<Self> {
        let embed_dim = c.embed_dim;
        let num_attention_heads = c.num_attention_heads;
        // All four CLIP projections are biased (`.bias` sibling in the checkpoint); `linear_detect_gs`
        // additionally loads the packed `.scales`/`.biases` triple when present, else the dense path.
        let k_proj =
            QLinear::linear_detect_gs(embed_dim, embed_dim, &vs, "k_proj", true, group_size)?;
        let v_proj =
            QLinear::linear_detect_gs(embed_dim, embed_dim, &vs, "v_proj", true, group_size)?;
        let q_proj =
            QLinear::linear_detect_gs(embed_dim, embed_dim, &vs, "q_proj", true, group_size)?;
        let out_proj =
            QLinear::linear_detect_gs(embed_dim, embed_dim, &vs, "out_proj", true, group_size)?;
        let head_dim = embed_dim / num_attention_heads;
        let scale = (head_dim as f64).powf(-0.5);
        Ok(ClipAttention {
            k_proj,
            v_proj,
            q_proj,
            out_proj,
            head_dim,
            scale,
            num_attention_heads,
        })
    }

    fn shape(&self, xs: &Tensor, seq_len: usize, bsz: usize) -> Result<Tensor> {
        xs.reshape((bsz, seq_len, self.num_attention_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()
    }

    fn forward(&self, xs: &Tensor, causal_attention_mask: &Tensor) -> Result<Tensor> {
        let in_dtype = xs.dtype();
        let (bsz, seq_len, embed_dim) = xs.dims3()?;
        let query_states = (self.q_proj.forward(xs)? * self.scale)?;
        let proj_shape = (bsz * self.num_attention_heads, seq_len, self.head_dim);
        let query_states = self
            .shape(&query_states, seq_len, bsz)?
            .reshape(proj_shape)?
            .to_dtype(DType::F32)?;
        let key_states = self
            .shape(&self.k_proj.forward(xs)?, seq_len, bsz)?
            .reshape(proj_shape)?
            .to_dtype(DType::F32)?;
        let value_states = self
            .shape(&self.v_proj.forward(xs)?, seq_len, bsz)?
            .reshape(proj_shape)?
            .to_dtype(DType::F32)?;
        let attn_weights = query_states.matmul(&key_states.transpose(1, 2)?)?;

        let src_len = key_states.dim(1)?;
        let attn_weights = attn_weights
            .reshape((bsz, self.num_attention_heads, seq_len, src_len))?
            .broadcast_add(causal_attention_mask)?;
        let attn_weights =
            attn_weights.reshape((bsz * self.num_attention_heads, seq_len, src_len))?;
        let attn_weights = nn::ops::softmax(&attn_weights, D::Minus1)?;

        let attn_output = attn_weights.matmul(&value_states)?.to_dtype(in_dtype)?;
        let attn_output = attn_output
            .reshape((bsz, self.num_attention_heads, seq_len, self.head_dim))?
            .transpose(1, 2)?
            .reshape((bsz, seq_len, embed_dim))?;
        self.out_proj.forward(&attn_output)
    }

    /// Test-only: whether every attention projection loaded packed (a pre-quantized MLX tier).
    #[cfg(test)]
    fn all_packed(&self) -> bool {
        self.q_proj.is_quantized()
            && self.k_proj.is_quantized()
            && self.v_proj.is_quantized()
            && self.out_proj.is_quantized()
    }
}

/// CLIP MLP (`fc1` → activation → `fc2`); both projections packed-detect.
#[derive(Debug)]
struct ClipMlp {
    fc1: QLinear,
    fc2: QLinear,
    activation: Activation,
}

impl ClipMlp {
    fn new(vs: VarBuilder, c: &Config, group_size: usize) -> Result<Self> {
        let fc1 = QLinear::linear_detect_gs(
            c.embed_dim,
            c.intermediate_size,
            &vs,
            "fc1",
            true,
            group_size,
        )?;
        let fc2 = QLinear::linear_detect_gs(
            c.intermediate_size,
            c.embed_dim,
            &vs,
            "fc2",
            true,
            group_size,
        )?;
        Ok(ClipMlp {
            fc1,
            fc2,
            activation: c.activation,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.fc1.forward(xs)?;
        self.fc2.forward(&self.activation.forward(&xs)?)
    }

    #[cfg(test)]
    fn all_packed(&self) -> bool {
        self.fc1.is_quantized() && self.fc2.is_quantized()
    }
}

#[derive(Debug)]
struct ClipEncoderLayer {
    self_attn: ClipAttention,
    layer_norm1: nn::LayerNorm,
    mlp: ClipMlp,
    layer_norm2: nn::LayerNorm,
}

impl ClipEncoderLayer {
    fn new(vs: VarBuilder, c: &Config, group_size: usize) -> Result<Self> {
        let self_attn = ClipAttention::new(vs.pp("self_attn"), c, group_size)?;
        let layer_norm1 = nn::layer_norm(c.embed_dim, 1e-5, vs.pp("layer_norm1"))?;
        let mlp = ClipMlp::new(vs.pp("mlp"), c, group_size)?;
        let layer_norm2 = nn::layer_norm(c.embed_dim, 1e-5, vs.pp("layer_norm2"))?;
        Ok(ClipEncoderLayer {
            self_attn,
            layer_norm1,
            mlp,
            layer_norm2,
        })
    }

    fn forward(&self, xs: &Tensor, causal_attention_mask: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let xs = self.layer_norm1.forward(xs)?;
        let xs = self.self_attn.forward(&xs, causal_attention_mask)?;
        let xs = (xs + residual)?;

        let residual = &xs;
        let xs = self.layer_norm2.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        xs + residual
    }
}

#[derive(Debug)]
struct ClipEncoder {
    layers: Vec<ClipEncoderLayer>,
}

impl ClipEncoder {
    fn new(vs: VarBuilder, c: &Config, group_size: usize) -> Result<Self> {
        let vs = vs.pp("layers");
        let mut layers: Vec<ClipEncoderLayer> = Vec::new();
        for index in 0..c.num_hidden_layers {
            let layer = ClipEncoderLayer::new(vs.pp(index.to_string()), c, group_size)?;
            layers.push(layer)
        }
        Ok(ClipEncoder { layers })
    }

    fn forward(&self, xs: &Tensor, causal_attention_mask: &Tensor) -> Result<Tensor> {
        let mut xs = xs.clone();
        for layer in self.layers.iter() {
            xs = layer.forward(&xs, causal_attention_mask)?;
        }
        Ok(xs)
    }
}

/// The vendored, packed-detecting CLIP text transformer. Its Linear surface routes through
/// [`candle_gen::quant`]; the forward contract (both [`Self::forward_with_mask`] and
/// [`Self::forward_until_encoder_layer`]) is byte-faithful to the stock
/// `stable_diffusion::clip::ClipTextTransformer`.
#[derive(Debug)]
pub struct ClipTextTransformer {
    embeddings: ClipTextEmbeddings,
    encoder: ClipEncoder,
    final_layer_norm: nn::LayerNorm,
}

impl ClipTextTransformer {
    /// Build at the default MLX group size (64).
    pub fn new(vs: VarBuilder, c: &Config) -> Result<Self> {
        Self::new_gs(vs, c, MLX_GROUP_SIZE)
    }

    /// Build at an explicit MLX packed `group_size` (threaded from the component `config.json`'s
    /// `quantization.group_size`, sc-9474/sc-9410). The dense path ignores it.
    pub fn new_gs(vs: VarBuilder, c: &Config, group_size: usize) -> Result<Self> {
        let vs = vs.pp("text_model");
        let embeddings = ClipTextEmbeddings::new(vs.pp("embeddings"), c)?;
        let encoder = ClipEncoder::new(vs.pp("encoder"), c, group_size)?;
        let final_layer_norm = nn::layer_norm(c.embed_dim, 1e-5, vs.pp("final_layer_norm"))?;
        Ok(ClipTextTransformer {
            embeddings,
            encoder,
            final_layer_norm,
        })
    }

    // https://github.com/huggingface/transformers/blob/674f750a57431222fa2832503a108df3badf1564/src/transformers/models/clip/modeling_clip.py#L678
    fn build_causal_attention_mask(
        bsz: usize,
        seq_len: usize,
        mask_after: usize,
        device: &Device,
    ) -> Result<Tensor> {
        let mask: Vec<_> = (0..seq_len)
            .flat_map(|i| {
                (0..seq_len).map(move |j| {
                    if j > i || j > mask_after {
                        f32::MIN
                    } else {
                        0.
                    }
                })
            })
            .collect();
        let mask = Tensor::from_slice(&mask, (seq_len, seq_len), device)?;
        mask.broadcast_as((bsz, seq_len, seq_len))
    }

    /// Last-hidden-state forward (post-`final_layer_norm`) — the txt2img [`crate::pipeline`] path.
    pub fn forward_with_mask(&self, xs: &Tensor, mask_after: usize) -> Result<Tensor> {
        let (bsz, seq_len) = xs.dims2()?;
        let xs = self.embeddings.forward(xs)?;
        let causal_attention_mask =
            Self::build_causal_attention_mask(bsz, seq_len, mask_after, xs.device())?;
        let xs = self.encoder.forward(&xs, &causal_attention_mask)?;
        self.final_layer_norm.forward(&xs)
    }

    /// Returns `(final_layer_norm(last), hidden_states[until_layer])` — the penultimate hidden path
    /// the InstantID [`crate::conditioning`] conditioner uses (`until_layer = -2`).
    pub fn forward_until_encoder_layer(
        &self,
        xs: &Tensor,
        mask_after: usize,
        until_layer: isize,
    ) -> Result<(Tensor, Tensor)> {
        let (bsz, seq_len) = xs.dims2()?;
        let xs = self.embeddings.forward(xs)?;
        let causal_attention_mask =
            Self::build_causal_attention_mask(bsz, seq_len, mask_after, xs.device())?;

        let mut xs = xs.clone();
        let mut intermediate = xs.clone();

        let until_layer = if until_layer < 0 {
            self.encoder.layers.len() as isize + until_layer
        } else {
            until_layer
        } as usize;

        for (layer_id, layer) in self.encoder.layers.iter().enumerate() {
            xs = layer.forward(&xs, &causal_attention_mask)?;
            if layer_id == until_layer {
                intermediate = xs.clone();
            }
        }

        Ok((self.final_layer_norm.forward(&xs)?, intermediate))
    }

    /// Test-only: whether every projection on the transformer's Linear surface loaded packed.
    #[cfg(test)]
    pub(crate) fn all_projections_packed(&self) -> bool {
        self.encoder
            .layers
            .iter()
            .all(|l| l.self_attn.all_packed() && l.mlp.all_packed())
    }
}

impl Module for ClipTextTransformer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward_with_mask(xs, usize::MAX)
    }
}

/// Load the bigG `text_projection` (the bare top-level `text_projection.weight`, `[1280, 1280]`, no
/// bias — `CLIPTextModelWithProjection`'s pooled head) as a **packed-detecting** [`QLinear`]: packed
/// straight from the `{weight u32, scales, biases}` triple when the MLX tier packs it (`.scales`
/// present), else the dense `text_projection.weight`. `projection_dim` is the square in/out dim.
pub fn text_projection(
    vs: &VarBuilder,
    projection_dim: usize,
    group_size: usize,
) -> Result<QLinear> {
    QLinear::linear_detect_gs(
        projection_dim,
        projection_dim,
        vs,
        "text_projection",
        false,
        group_size,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::safetensors::MmapedSafetensors;
    use candle_nn::{VarBuilder, VarMap};
    use candle_transformers::models::stable_diffusion::clip as stock;
    use std::collections::HashMap;

    const GS: usize = 64;

    /// A tiny CLIP config that exercises every vendored path cheaply on CPU (group 64 divides
    /// `embed_dim` / `intermediate_size`; 2 layers).
    fn tiny_cfg() -> Config {
        Config {
            vocab_size: 64,
            embed_dim: 64,
            intermediate_size: 128,
            max_position_embeddings: 16,
            pad_with: Some("!".to_string()),
            num_hidden_layers: 2,
            num_attention_heads: 4,
            projection_dim: 64,
            activation: Activation::QuickGelu,
        }
    }

    /// The stock CLIP config whose public shape our vendored `Config::sdxl()` (CLIP-L: 12 layers,
    /// 768/3072, QuickGelu) mirrors exactly. The stock `Config` fields are private, so we can only
    /// build it via its named constructors.
    fn stock_cfg_l() -> stock::Config {
        stock::Config::sdxl()
    }

    /// The stock CLIP config whose public shape our vendored `Config::sdxl2()` (OpenCLIP bigG: 32
    /// layers, 1280/5120, Gelu, 20 heads) mirrors exactly. Same private-fields constraint — the bigG
    /// arm therefore exercises the FULL 32-layer tower (no reduced-depth stock ctor exists), which
    /// still runs cheaply enough on CPU for a `--lib` test.
    fn stock_cfg_g() -> stock::Config {
        stock::Config::sdxl2()
    }

    /// Cosine similarity over all elements (f64) — the canonical `candle_gen::quant` packed-vs-dense
    /// parity metric (`packed_qlinear_forward_matches_dense_grid` asserts `> 0.99999`). The MLX Q4→Q4_1
    /// repack rounds group scales to f16, so an f32-grid reference deviates by a small per-group
    /// quantization of the *scale* — cosine (direction, not absolute scale) is the metric the shared
    /// seam uses for exactly this reason, matching the sc-9416 UNet packed surface convention.
    fn cosine(a: &Tensor, b: &Tensor) -> f32 {
        let a = a.to_dtype(DType::F32).unwrap().flatten_all().unwrap();
        let b = b.to_dtype(DType::F32).unwrap().flatten_all().unwrap();
        let a = a.to_vec1::<f32>().unwrap();
        let b = b.to_vec1::<f32>().unwrap();
        let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (x, y) in a.iter().zip(b.iter()) {
            dot += (*x as f64) * (*y as f64);
            na += (*x as f64) * (*x as f64);
            nb += (*y as f64) * (*y as f64);
        }
        (dot / (na.sqrt() * nb.sqrt() + 1e-12)) as f32
    }

    /// Pack an `[out, in]` weight as an MLX Q4 triple (LSB-first nibbles, per-group affine
    /// scales/biases) — mirrors the sc-9416 UNet packed-detect test's `pack` helper.
    fn pack(map: &mut HashMap<String, Tensor>, base: &str, out_f: usize, in_f: usize, bias: bool) {
        let dev = Device::Cpu;
        let codes: Vec<u8> = (0..out_f * in_f)
            .map(|i| ((i * 5 + 3) % 16) as u8)
            .collect();
        let words: Vec<u32> = codes
            .chunks_exact(8)
            .map(|c| {
                c.iter()
                    .enumerate()
                    .fold(0u32, |acc, (i, &q)| acc | ((q as u32 & 0xF) << (4 * i)))
            })
            .collect();
        let groups = out_f * in_f / GS;
        let scales: Vec<f32> = (0..groups).map(|g| 0.03125 * (g as f32 + 1.0)).collect();
        let biases: Vec<f32> = (0..groups).map(|g| -0.25 - 0.1 * g as f32).collect();
        let gpr = in_f / GS;
        map.insert(
            format!("{base}.weight"),
            Tensor::from_vec(words, (out_f, in_f / 8), &dev).unwrap(),
        );
        map.insert(
            format!("{base}.scales"),
            Tensor::from_vec(scales, (out_f, gpr), &dev).unwrap(),
        );
        map.insert(
            format!("{base}.biases"),
            Tensor::from_vec(biases, (out_f, gpr), &dev).unwrap(),
        );
        if bias {
            map.insert(
                format!("{base}.bias"),
                Tensor::zeros((out_f,), DType::F32, &dev).unwrap(),
            );
        }
    }

    /// The exact affine grid the pack represents (so a dense reference can be built from the SAME
    /// numbers the packed path repacks) — `[out, in]`.
    fn grid(out_f: usize, in_f: usize) -> Vec<f32> {
        let codes: Vec<u8> = (0..out_f * in_f)
            .map(|i| ((i * 5 + 3) % 16) as u8)
            .collect();
        let groups_per_row = in_f / GS;
        let scale = |g: usize| 0.03125 * (g as f32 + 1.0);
        let bias = |g: usize| -0.25 - 0.1 * g as f32;
        (0..out_f * in_f)
            .map(|i| {
                let (row, col) = (i / in_f, i % in_f);
                let g = row * groups_per_row + col / GS;
                scale(g) * codes[i] as f32 + bias(g)
            })
            .collect()
    }

    fn dense_lin(
        map: &mut HashMap<String, Tensor>,
        base: &str,
        out_f: usize,
        in_f: usize,
        bias: bool,
    ) {
        map.insert(
            format!("{base}.weight"),
            Tensor::from_vec(grid(out_f, in_f), (out_f, in_f), &Device::Cpu).unwrap(),
        );
        if bias {
            map.insert(
                format!("{base}.bias"),
                Tensor::zeros((out_f,), DType::F32, &Device::Cpu).unwrap(),
            );
        }
    }

    /// Fill one encoder layer's Linear surface (attn `q/k/v/out_proj` + MLP `fc1/fc2`), either packed
    /// or dense, at the SAME affine grid — the pack/dense switch is `packed`.
    fn layer_linears(map: &mut HashMap<String, Tensor>, prefix: &str, c: &Config, packed: bool) {
        let e = c.embed_dim;
        let put = |map: &mut HashMap<String, Tensor>, base: &str, o: usize, i: usize| {
            if packed {
                pack(map, base, o, i, true);
            } else {
                dense_lin(map, base, o, i, true);
            }
        };
        for p in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            put(map, &format!("{prefix}.self_attn.{p}"), e, e);
        }
        put(map, &format!("{prefix}.mlp.fc1"), c.intermediate_size, e);
        put(map, &format!("{prefix}.mlp.fc2"), e, c.intermediate_size);
    }

    /// The dense (norm/embedding) surface, shared by the packed and dense checkpoints — random but
    /// identical (seeded from a fixed grid) so packed-vs-dense parity is isolated to the Linears.
    fn norms_and_embeds(map: &mut HashMap<String, Tensor>, c: &Config) {
        let dev = Device::Cpu;
        map.insert(
            "text_model.embeddings.token_embedding.weight".into(),
            Tensor::from_vec(
                grid(c.vocab_size, c.embed_dim),
                (c.vocab_size, c.embed_dim),
                &dev,
            )
            .unwrap(),
        );
        map.insert(
            "text_model.embeddings.position_embedding.weight".into(),
            Tensor::from_vec(
                grid(c.max_position_embeddings, c.embed_dim),
                (c.max_position_embeddings, c.embed_dim),
                &dev,
            )
            .unwrap(),
        );
        let ln = |map: &mut HashMap<String, Tensor>, base: &str| {
            map.insert(
                format!("{base}.weight"),
                Tensor::ones((c.embed_dim,), DType::F32, &dev).unwrap(),
            );
            map.insert(
                format!("{base}.bias"),
                Tensor::zeros((c.embed_dim,), DType::F32, &dev).unwrap(),
            );
        };
        for l in 0..c.num_hidden_layers {
            ln(map, &format!("text_model.encoder.layers.{l}.layer_norm1"));
            ln(map, &format!("text_model.encoder.layers.{l}.layer_norm2"));
        }
        ln(map, "text_model.final_layer_norm");
    }

    fn build_checkpoint(c: &Config, packed: bool) -> HashMap<String, Tensor> {
        let mut map = HashMap::new();
        norms_and_embeds(&mut map, c);
        for l in 0..c.num_hidden_layers {
            layer_linears(
                &mut map,
                &format!("text_model.encoder.layers.{l}"),
                c,
                packed,
            );
        }
        map
    }

    fn vb_from_map(
        map: HashMap<String, Tensor>,
        tag: &str,
    ) -> (VarBuilder<'static>, std::path::PathBuf) {
        let tmp = std::env::temp_dir().join(format!(
            "sc9527_clip_{tag}_{}.safetensors",
            std::process::id()
        ));
        candle_core::safetensors::save(&map, &tmp).unwrap();
        // SAFETY: we just wrote this file and nothing else touches it during the test.
        let st = unsafe { MmapedSafetensors::new(&tmp).unwrap() };
        let vb = VarBuilder::from_backend(Box::new(st), DType::F32, Device::Cpu);
        (vb, tmp)
    }

    // ---- (a) packed-detect fires on the CLIP key layout ---------------------------------------

    /// A `.scales`-sibling CLIP checkpoint routes EVERY projection (attn q/k/v/out_proj, MLP fc1/fc2)
    /// to the packed path; a dense one (no `.scales`) falls back. Covers both encoders' key layouts
    /// (CLIP-L `sdxl()` shape and bigG `sdxl2()` shape at tiny dims).
    #[test]
    fn packed_detect_fires_on_clip_layout() -> Result<()> {
        // A CLIP-L-shaped tiny config (QuickGelu, 4 heads, 2 layers) and a bigG-shaped one (Gelu, 8
        // heads, 3 layers) — the two encoders' distinct layer/head/activation layouts. `embed_dim` /
        // `intermediate_size` stay multiples of the group 64 so the synthetic pack tiles cleanly (the
        // real tiers pad these to 768/1280 and 3072/5120). Both are packed then dense-detected.
        let bigg = Config {
            num_attention_heads: 8,
            num_hidden_layers: 3,
            activation: Activation::Gelu,
            ..tiny_cfg()
        };
        for c in [tiny_cfg(), bigg] {
            let (vb_p, tmp_p) = vb_from_map(build_checkpoint(&c, true), "detect_packed");
            let packed = ClipTextTransformer::new_gs(vb_p, &c, GS)?;
            assert!(
                packed.all_projections_packed(),
                "every CLIP Linear must load packed on a `.scales` checkpoint"
            );

            let (vb_d, tmp_d) = vb_from_map(build_checkpoint(&c, false), "detect_dense");
            let dense = ClipTextTransformer::new_gs(vb_d, &c, GS)?;
            assert!(
                !dense.all_projections_packed(),
                "a dense (no `.scales`) checkpoint must fall back to dense Linears"
            );
            std::fs::remove_file(&tmp_p).ok();
            std::fs::remove_file(&tmp_d).ok();
        }
        Ok(())
    }

    // ---- (b) packed-vs-dense encode parity -----------------------------------------------------

    /// The packed CLIP forward matches the dense CLIP forward built from the SAME affine grid — within
    /// the sc-9416 UNet packed-vs-dense tolerance (the repack is lossless and both forwards
    /// dequant-to-dense-matmul). Runs both the last-hidden and the penultimate paths.
    #[test]
    fn packed_vs_dense_encode_parity() -> Result<()> {
        let c = tiny_cfg();
        let (vb_p, tmp_p) = vb_from_map(build_checkpoint(&c, true), "parity_packed");
        let (vb_d, tmp_d) = vb_from_map(build_checkpoint(&c, false), "parity_dense");
        let packed = ClipTextTransformer::new_gs(vb_p, &c, GS)?;
        let dense = ClipTextTransformer::new_gs(vb_d, &c, GS)?;
        assert!(packed.all_projections_packed());
        assert!(!dense.all_projections_packed());

        let ids = Tensor::from_vec(
            (0..c.max_position_embeddings as u32).collect::<Vec<_>>(),
            (1, c.max_position_embeddings),
            &Device::Cpu,
        )?;

        // Last hidden state.
        let y_p = packed.forward_with_mask(&ids, usize::MAX)?;
        let y_d = dense.forward_with_mask(&ids, usize::MAX)?;
        let cos = cosine(&y_p, &y_d);
        assert!(cos > 0.99999, "packed vs dense last-hidden cosine {cos:.6}");

        // Penultimate hidden state (the conditioning path).
        let (_, penult_p) = packed.forward_until_encoder_layer(&ids, usize::MAX, -2)?;
        let (_, penult_d) = dense.forward_until_encoder_layer(&ids, usize::MAX, -2)?;
        let cos2 = cosine(&penult_p, &penult_d);
        assert!(
            cos2 > 0.99999,
            "packed vs dense penultimate cosine {cos2:.6}"
        );

        std::fs::remove_file(&tmp_p).ok();
        std::fs::remove_file(&tmp_d).ok();
        Ok(())
    }

    /// The packed bigG `text_projection` forward matches the dense one on the same grid — the pooled
    /// head packs too (sc-9527 AC: the bigG final text projection packed-detects).
    #[test]
    fn packed_text_projection_matches_dense() -> Result<()> {
        let dim = 64usize;
        let mut mp = HashMap::new();
        pack(&mut mp, "text_projection", dim, dim, false);
        let (vb_p, tmp_p) = vb_from_map(mp, "tp_packed");
        let mut md = HashMap::new();
        dense_lin(&mut md, "text_projection", dim, dim, false);
        let (vb_d, tmp_d) = vb_from_map(md, "tp_dense");

        let tp_p = text_projection(&vb_p, dim, GS)?;
        let tp_d = text_projection(&vb_d, dim, GS)?;
        assert!(
            tp_p.is_quantized(),
            "packed text_projection must load packed"
        );
        assert!(!tp_d.is_quantized(), "dense text_projection falls back");

        let x = Tensor::randn(0f32, 1f32, (2, dim), &Device::Cpu)?;
        let cos = cosine(&tp_p.forward(&x)?, &tp_d.forward(&x)?);
        assert!(
            cos > 0.99999,
            "packed vs dense text_projection cosine {cos:.6}"
        );
        std::fs::remove_file(&tmp_p).ok();
        std::fs::remove_file(&tmp_d).ok();
        Ok(())
    }

    // ---- (c) vendored-dense vs stock-candle_transformers parity --------------------------------

    /// The vendored tower on a DENSE checkpoint is bit-identical to the stock candle-transformers CLIP
    /// built from the SAME `VarMap` weights — the guard that swapping `candle_nn::Linear` →
    /// `QLinear::linear_detect_gs` (dense fallback) changed nothing numerically.
    ///
    /// Covers BOTH encoders the SDXL conditioner uses, on their real shapes:
    ///   * CLIP-L (`Config::sdxl()` vs `stock::Config::sdxl()`): 12 layers, 768/3072, QuickGelu;
    ///   * OpenCLIP bigG (`Config::sdxl2()` vs `stock::Config::sdxl2()`): 32 layers, 1280/5120, Gelu.
    ///
    /// A per-encoder activation/eps/dim divergence on the bigG arm (different activation + dims than
    /// CLIP-L) would be caught here. And for BOTH encoders it compares BOTH forward paths the
    /// conditioner actually reads — the last-hidden `forward` AND the penultimate
    /// `forward_until_encoder_layer(.., -2)` (InstantID/refiner conditioning path).
    #[test]
    fn vendored_dense_matches_stock() -> Result<()> {
        // One arm: build the vendored tower first (populating the VarMap with random dense weights —
        // no `.scales`, so the dense fallback fires), then the stock tower reads the SAME parameters,
        // and compare last-hidden + penultimate to the vendored tolerance bar (1e-4).
        fn assert_arm(vendored_cfg: &Config, stock_cfg: &stock::Config) -> Result<(f32, f32)> {
            let dev = Device::Cpu;
            let vm = VarMap::new();
            let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
            let vendored = ClipTextTransformer::new(vb.clone(), vendored_cfg)?;
            let stock_model = stock::ClipTextTransformer::new(vb, stock_cfg)?;

            let ids = Tensor::from_vec((0..77u32).collect::<Vec<_>>(), (1, 77), &dev)?;

            // Last hidden state (`final_layer_norm` output).
            let y_v = vendored.forward(&ids)?;
            let y_s = stock_model.forward(&ids)?;
            assert_eq!(y_v.dims(), y_s.dims());
            let diff_last = (y_v - y_s)?.abs()?.max_all()?.to_scalar::<f32>()?;

            // Penultimate hidden state (`until_layer = -2`) — the conditioning path.
            let (_, penult_v) = vendored.forward_until_encoder_layer(&ids, usize::MAX, -2)?;
            let (_, penult_s) = stock_model.forward_until_encoder_layer(&ids, usize::MAX, -2)?;
            assert_eq!(penult_v.dims(), penult_s.dims());
            let diff_penult = (penult_v - penult_s)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?;

            Ok((diff_last, diff_penult))
        }

        // CLIP-L (`text_encoder/`): QuickGelu, 12 layers, 768/3072.
        let (l_last, l_penult) = assert_arm(&Config::sdxl(), &stock_cfg_l())?;
        assert!(
            l_last < 1e-4,
            "vendored CLIP-L last-hidden diverged from stock by {l_last}"
        );
        assert!(
            l_penult < 1e-4,
            "vendored CLIP-L penultimate diverged from stock by {l_penult}"
        );

        // OpenCLIP bigG (`text_encoder_2/`): Gelu, 32 layers, 1280/5120, 20 heads — the arm that would
        // catch a per-encoder activation/eps/dim divergence CLIP-L cannot.
        let (g_last, g_penult) = assert_arm(&Config::sdxl2(), &stock_cfg_g())?;
        assert!(
            g_last < 1e-4,
            "vendored bigG last-hidden diverged from stock by {g_last}"
        );
        assert!(
            g_penult < 1e-4,
            "vendored bigG penultimate diverged from stock by {g_penult}"
        );

        Ok(())
    }
}
