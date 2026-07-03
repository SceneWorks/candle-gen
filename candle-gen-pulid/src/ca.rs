//! PerceiverAttentionCA ×20 + the FLUX DiT injection schedule (sc-5492). Candle port of
//! `mlx-gen-pulid`'s `ca.rs` (= `pulid/encoders_transformer.py PerceiverAttentionCA` + the
//! `flux/model.py` injection schedule).
//!
//! Each [`PerceiverAttentionCA`] cross-attends the **image** tokens (queries, dim=3072) onto the
//! IDFormer `id_embedding` (keys/values, kv_dim=2048): `img += id_weight · ca(id_embedding, img)`. 20 of
//! them are injected into the FLUX DiT — after every 2nd double block (10) and every 4th single block
//! (10) — via the generic [`candle_gen_flux::DitImageInjector`] post-block seam (no PuLID code in the
//! flux crate). `ca_idx` runs 0..9 across the double injections then 10..19 across the single ones,
//! exactly matching the reference's shared running counter.
//!
//! The conditioning path is **f32** (the EVA tower + IDFormer + these CA modules), while the candle FLUX
//! DiT image stream is bf16, so [`PerceiverAttentionCA::forward`] casts the incoming image tokens to f32,
//! computes the residual in f32, and the DiT seam casts it back to the image dtype before adding (the
//! `r.to_dtype(img.dtype())` in `IpFlux::forward_injected`). LayerNorm ε=1e-5; SDPA `scale =
//! dim_head^-0.5`, softmax in f32.

use candle_core::{DType, Tensor, D};
use candle_nn::ops::softmax_last_dim;
use candle_nn::{LayerNorm, Linear, Module};

use candle_gen::weights::Weights;
use candle_gen::Result as GenResult;
use candle_gen_flux::DitImageInjector;

const EPS: f64 = 1e-5;

fn join(p: &str, leaf: &str) -> String {
    format!("{p}.{leaf}")
}

fn layer_norm(w: &Weights, prefix: &str) -> GenResult<LayerNorm> {
    Ok(LayerNorm::new(
        w.require(&format!("{prefix}.weight"))?,
        w.require(&format!("{prefix}.bias"))?,
        EPS,
    ))
}

/// Cross-attention block: q from `latents` (image tokens, dim=3072), k/v from `x` (id_embedding,
/// kv_dim=2048). All linears bias-free. Shares its structure with the IDFormer's `PerceiverAttention`;
/// the only behavioral difference is the k/v source (here `x` alone, the id_embedding).
pub struct PerceiverAttentionCA {
    norm1: LayerNorm, // over kv_dim (id_embedding)
    norm2: LayerNorm, // over dim (image)
    to_q: Linear,
    to_kv: Linear,
    to_out: Linear,
    heads: usize,
    dim_head: usize,
}

impl PerceiverAttentionCA {
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        heads: usize,
        dim_head: usize,
    ) -> GenResult<Self> {
        let lin = |leaf: &str| -> GenResult<Linear> {
            Ok(Linear::new(
                w.require(&join(prefix, &format!("{leaf}.weight")))?,
                None,
            ))
        };
        Ok(Self {
            norm1: layer_norm(w, &join(prefix, "norm1"))?,
            norm2: layer_norm(w, &join(prefix, "norm2"))?,
            to_q: lin("to_q")?,
            to_kv: lin("to_kv")?,
            to_out: lin("to_out")?,
            heads,
            dim_head,
        })
    }

    /// `id_embedding`: `[B, 32, kv_dim]`; `img`: `[B, S, dim]` → residual `[B, S, dim]` (f32). `img`
    /// arrives in the DiT's dtype (bf16); it is cast to f32 for the f32 conditioning math.
    pub fn forward(&self, id_embedding: &Tensor, img: &Tensor) -> candle_core::Result<Tensor> {
        let img = img.to_dtype(DType::F32)?;
        let x = self.norm1.forward(id_embedding)?;
        let lat = self.norm2.forward(&img)?;
        let (b, s, _dim) = lat.dims3()?;
        let n_kv = x.dim(1)?;
        let (h, hd) = (self.heads, self.dim_head);

        let q = self.to_q.forward(&lat)?; // [B, S, inner]
        let kv = self.to_kv.forward(&x)?; // [B, 32, inner*2]
        let parts = kv.chunk(2, D::Minus1)?;
        let to_heads = |t: &Tensor, n: usize| -> candle_core::Result<Tensor> {
            t.reshape((b, n, h, hd))?.transpose(1, 2)?.contiguous()
        };
        let q = to_heads(&q, s)?;
        let k = to_heads(&parts[0], n_kv)?;
        let v = to_heads(&parts[1], n_kv)?;

        let scale = (hd as f64).powf(-0.5);
        let scores = (q.matmul(&k.transpose(D::Minus1, D::Minus2)?.contiguous()?)? * scale)?;
        let attn = softmax_last_dim(&scores)?.matmul(&v)?;
        let out = attn
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, s, h * hd))?;
        self.to_out.forward(&out)
    }
}

/// The 20 CA modules + the bound `id_embedding`/`id_weight`, implementing the FLUX DiT injection
/// schedule as a [`DitImageInjector`]. Build it for a given id_embedding (from the IDFormer) and inject
/// during the denoise via [`candle_gen_flux::IpFlux::forward_injected`].
pub struct PulidCa {
    ca: Vec<PerceiverAttentionCA>,
    id_embedding: Tensor,
    id_weight: f64,
    double_interval: usize,
    single_interval: usize,
    n_double_inject: usize,
}

impl PulidCa {
    /// `prefix` = `"pulid_ca"`. `num_double_blocks`/`num_single_blocks` are the FLUX DiT block counts
    /// (19 / 38) — used to size the module count and the shared ca_idx base. `heads`/`dim_head` =
    /// 16/128 (the PuLID CA shape).
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        id_embedding: Tensor,
        id_weight: f64,
        num_double_blocks: usize,
        num_single_blocks: usize,
    ) -> GenResult<Self> {
        let double_interval = 2usize;
        let single_interval = 4usize;
        let n_double_inject = num_double_blocks.div_ceil(double_interval);
        let n_single_inject = num_single_blocks.div_ceil(single_interval);
        let num_ca = n_double_inject + n_single_inject;
        let ca = (0..num_ca)
            .map(|i| PerceiverAttentionCA::from_weights(w, &join(prefix, &i.to_string()), 16, 128))
            .collect::<GenResult<Vec<_>>>()?;
        Ok(Self {
            ca,
            id_embedding,
            id_weight,
            double_interval,
            single_interval,
            n_double_inject,
        })
    }

    /// The number of CA modules (= 20 for the FLUX 19-double / 38-single schedule).
    pub fn num_ca(&self) -> usize {
        self.ca.len()
    }

    fn scaled(&self, r: Tensor) -> candle_core::Result<Tensor> {
        r * self.id_weight
    }
}

impl DitImageInjector for PulidCa {
    fn after_double(
        &self,
        block_idx: usize,
        img_hidden: &Tensor,
    ) -> candle_core::Result<Option<Tensor>> {
        if self.id_weight == 0.0 || !block_idx.is_multiple_of(self.double_interval) {
            return Ok(None); // bit-identical to plain FLUX at a non-injection block / 0 weight
        }
        let ca_idx = block_idx / self.double_interval;
        Ok(Some(self.scaled(
            self.ca[ca_idx].forward(&self.id_embedding, img_hidden)?,
        )?))
    }

    fn injects_after_single(&self, block_idx: usize) -> bool {
        self.id_weight != 0.0 && block_idx.is_multiple_of(self.single_interval)
    }

    fn after_single(
        &self,
        block_idx: usize,
        img_tokens: &Tensor,
    ) -> candle_core::Result<Option<Tensor>> {
        if self.id_weight == 0.0 || !block_idx.is_multiple_of(self.single_interval) {
            return Ok(None);
        }
        // Continue the shared counter after the double injections.
        let ca_idx = self.n_double_inject + block_idx / self.single_interval;
        Ok(Some(self.scaled(
            self.ca[ca_idx].forward(&self.id_embedding, img_tokens)?,
        )?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use std::collections::HashMap;

    /// Build a tiny `PerceiverAttentionCA` (heads=1, dim_head=2 ⇒ inner=2; img dim=3, kv_dim=4) from a
    /// random weight map, exercising the candle CA forward math at CPU scale.
    fn tiny_ca(prefix: &str, dev: &Device) -> PerceiverAttentionCA {
        let (dim, kv_dim, inner) = (3usize, 4usize, 2usize);
        let r = |rows: usize, cols: usize| Tensor::randn(0f32, 1f32, (rows, cols), dev).unwrap();
        let v = |n: usize| Tensor::randn(0f32, 1f32, n, dev).unwrap();
        let mut m: HashMap<String, Tensor> = HashMap::new();
        m.insert(format!("{prefix}.norm1.weight"), v(kv_dim));
        m.insert(format!("{prefix}.norm1.bias"), v(kv_dim));
        m.insert(format!("{prefix}.norm2.weight"), v(dim));
        m.insert(format!("{prefix}.norm2.bias"), v(dim));
        m.insert(format!("{prefix}.to_q.weight"), r(inner, dim)); // [out=inner, in=dim]
        m.insert(format!("{prefix}.to_kv.weight"), r(2 * inner, kv_dim));
        m.insert(format!("{prefix}.to_out.weight"), r(dim, inner));
        let w = Weights::from_map(m);
        PerceiverAttentionCA::from_weights(&w, prefix, 1, 2).unwrap()
    }

    /// The injection schedule + shapes: `after_double` fires on even block indices, `after_single` on
    /// multiples of 4 (continuing the shared counter), and an `id_weight = 0` PulidCa injects nothing.
    #[test]
    fn injection_schedule_and_shapes() {
        let dev = Device::Cpu;
        let (dim, kv_dim, s) = (3usize, 4usize, 5usize);
        let id_embedding = Tensor::randn(0f32, 1f32, (1, 6, kv_dim), &dev).unwrap();
        let img = Tensor::randn(0f32, 1f32, (1, s, dim), &dev).unwrap();
        // Two tiny CAs: ca[0] for the double seam, ca[1] for the single seam (n_double_inject = 1).
        let pulid = PulidCa {
            ca: vec![tiny_ca("a", &dev), tiny_ca("b", &dev)],
            id_embedding: id_embedding.clone(),
            id_weight: 1.0,
            double_interval: 2,
            single_interval: 4,
            n_double_inject: 1,
        };

        // after_double: even idx → residual [1,S,dim]; odd idx → None.
        let r0 = pulid
            .after_double(0, &img)
            .unwrap()
            .expect("inject at block 0");
        assert_eq!(r0.dims(), &[1, s, dim]);
        assert!(pulid.after_double(1, &img).unwrap().is_none());

        // after_single: multiples of 4 inject (ca index n_double_inject + idx/4); others skip.
        assert!(pulid.injects_after_single(0));
        assert!(!pulid.injects_after_single(1));
        assert!(pulid.injects_after_single(4));
        let s0 = pulid
            .after_single(0, &img)
            .unwrap()
            .expect("single inject at 0");
        assert_eq!(s0.dims(), &[1, s, dim]);

        // id_weight = 0 ⇒ no injection anywhere (the no-id ablation arm).
        let off = PulidCa {
            ca: vec![tiny_ca("a", &dev)],
            id_embedding,
            id_weight: 0.0,
            double_interval: 2,
            single_interval: 4,
            n_double_inject: 1,
        };
        assert!(off.after_double(0, &img).unwrap().is_none());
        assert!(!off.injects_after_single(0));
    }
}
