//! Synthetic-weight smoke tests (no real checkpoint, CPU): build a *tiny* PidNet + Gemma-2 from
//! randomly-initialized weights at the correct shapes and run a forward. This exercises every
//! reshape / permute / matmul / conv / attention path in the candle port — catching shape bugs that
//! the pure registry/budget unit tests can't — without needing the 1.36 B real checkpoint or a GPU.
//! Numerics are meaningless here (random weights); the assertions are on shapes + finiteness.

use std::collections::HashMap;

use candle_gen::candle_core::{Device, Tensor};
use candle_gen::Weights;
use candle_gen_pid::config::PidConfig;
use candle_gen_pid::gemma2::{Gemma2, Gemma2Config};
use candle_gen_pid::lq::PidNet;

/// A tiny PixDiT/PidNet config with the real topology but minimal dims. Geometry is self-consistent:
/// `patch_size=2, lsdf=2, sr_scale=2` ⇒ LQ upsample ratio `(2·2)/2 = 2`, and a pixel side
/// `H = zH · lsdf · sr_scale` keeps the upsampled latent grid equal to the patch grid.
fn tiny_cfg() -> PidConfig {
    PidConfig {
        in_channels: 3,
        num_groups: 2,
        hidden_size: 8,
        pixel_hidden_size: 4,
        pixel_attn_hidden_size: 8,
        pixel_num_groups: 2,
        patch_depth: 2,
        pixel_depth: 1,
        patch_size: 2,
        txt_embed_dim: 6,
        txt_max_length: 5,
        use_text_rope: true,
        text_rope_theta: 10000.0,
        rope_mode: candle_gen_pid::config::RopeMode::NtkAware,
        rope_ref_h: 4,
        rope_ref_w: 4,
        lq_in_channels: 0,
        lq_latent_channels: 4,
        lq_hidden_dim: 8, // divisible by the ResBlock GroupNorm's 4 groups
        lq_num_res_blocks: 1,
        lq_interval: 2,
        sr_scale: 2,
        latent_spatial_down_factor: 2,
    }
}

struct Builder {
    map: HashMap<String, Tensor>,
    dev: Device,
}

impl Builder {
    fn new() -> Self {
        Self {
            map: HashMap::new(),
            dev: Device::Cpu,
        }
    }
    /// Insert a random tensor of the given shape under `key`.
    fn w(&mut self, key: &str, shape: &[usize]) {
        let t = Tensor::randn(0f32, 1f32, shape, &self.dev).unwrap();
        self.map.insert(key.to_string(), t);
    }
    fn into_weights(self) -> Weights {
        Weights::from_map(self.map)
    }
}

fn build_pidnet_weights(cfg: &PidConfig) -> Weights {
    let mut b = Builder::new();
    let h = cfg.hidden_size as usize;
    let hd = cfg.head_dim() as usize;
    let ng = cfg.num_groups as usize;
    let p2 = (cfg.patch_size * cfg.patch_size) as usize;
    let pd = cfg.pixel_hidden_size as usize;
    let pad = cfg.pixel_attn_hidden_size as usize;
    let phd = cfg.pixel_head_dim() as usize;
    let ff = 16usize; // arbitrary SwiGLU inner width
    let pmlp = 8usize; // arbitrary pixel MLP inner width
    let txt = cfg.txt_embed_dim as usize;
    let lqh = cfg.lq_hidden_dim as usize;
    let lqc = cfg.lq_latent_channels as usize;

    // --- patch blocks (MMDiTBlockT2I) ---
    for i in 0..cfg.patch_depth {
        let p = format!("patch_blocks.{i}");
        for n in ["norm_x1", "norm_y1", "norm_x2", "norm_y2"] {
            b.w(&format!("{p}.{n}.weight"), &[h]);
        }
        b.w(&format!("{p}.attn.qkv_x.weight"), &[3 * ng * hd, h]);
        b.w(&format!("{p}.attn.qkv_y.weight"), &[3 * ng * hd, h]);
        for n in ["q_norm_x", "k_norm_x", "q_norm_y", "k_norm_y"] {
            b.w(&format!("{p}.attn.{n}.weight"), &[hd]);
        }
        b.w(&format!("{p}.attn.proj_x.weight"), &[h, ng * hd]);
        b.w(&format!("{p}.attn.proj_y.weight"), &[h, ng * hd]);
        for m in ["mlp_x", "mlp_y"] {
            b.w(&format!("{p}.{m}.w1.weight"), &[ff, h]);
            b.w(&format!("{p}.{m}.w3.weight"), &[ff, h]);
            b.w(&format!("{p}.{m}.w2.weight"), &[h, ff]);
        }
        b.w(&format!("{p}.adaLN_modulation_img.0.weight"), &[6 * h, h]);
        b.w(&format!("{p}.adaLN_modulation_txt.0.weight"), &[6 * h, h]);
    }

    // --- pixel blocks (PiTBlock) ---
    for i in 0..cfg.pixel_depth {
        let p = format!("pixel_blocks.{i}");
        b.w(&format!("{p}.compress_to_attn.weight"), &[pad, p2 * pd]);
        b.w(&format!("{p}.expand_from_attn.weight"), &[p2 * pd, pad]);
        b.w(&format!("{p}.norm1.weight"), &[pd]);
        b.w(&format!("{p}.norm2.weight"), &[pd]);
        b.w(
            &format!("{p}.attn.qkv.weight"),
            &[3 * cfg.pixel_num_groups as usize * phd, pad],
        );
        b.w(&format!("{p}.attn.q_norm.weight"), &[phd]);
        b.w(&format!("{p}.attn.k_norm.weight"), &[phd]);
        b.w(
            &format!("{p}.attn.proj.weight"),
            &[pad, cfg.pixel_num_groups as usize * phd],
        );
        b.w(&format!("{p}.mlp.fc1.weight"), &[pmlp, pd]);
        b.w(&format!("{p}.mlp.fc2.weight"), &[pd, pmlp]);
        b.w(&format!("{p}.adaLN_modulation.0.weight"), &[p2 * 6 * pd, h]);
    }

    // --- embedders / conditioner / final ---
    b.w(
        "pixel_embedder.proj.weight",
        &[pd, cfg.in_channels as usize],
    );
    b.w(
        "s_embedder.proj.weight",
        &[h, cfg.in_channels as usize * p2],
    );
    b.w("t_embedder.mlp.0.weight", &[h, 256]);
    b.w("t_embedder.mlp.2.weight", &[h, h]);
    b.w("y_embedder.proj.weight", &[h, txt]);
    b.w("y_embedder.norm.weight", &[h]);
    b.w("y_pos_embedding", &[1, cfg.txt_max_length as usize, h]);
    b.w("final_layer.norm.weight", &[pd]);
    b.w("final_layer.linear.weight", &[cfg.in_channels as usize, pd]);

    // --- LQ adapter (lq_proj) ---
    b.w("lq_proj.latent_proj.0.weight", &[lqh, lqc, 3, 3]);
    b.w("lq_proj.latent_proj.0.bias", &[lqh]);
    b.w("lq_proj.latent_proj.2.weight", &[lqh, lqh, 3, 3]);
    b.w("lq_proj.latent_proj.2.bias", &[lqh]);
    for r in 0..cfg.lq_num_res_blocks {
        let rp = format!("lq_proj.latent_proj.{}", r + 3);
        b.w(&format!("{rp}.block.0.weight"), &[lqh]);
        b.w(&format!("{rp}.block.0.bias"), &[lqh]);
        b.w(&format!("{rp}.block.2.weight"), &[lqh, lqh, 3, 3]);
        b.w(&format!("{rp}.block.2.bias"), &[lqh]);
        b.w(&format!("{rp}.block.3.weight"), &[lqh]);
        b.w(&format!("{rp}.block.3.bias"), &[lqh]);
        b.w(&format!("{rp}.block.5.weight"), &[lqh, lqh, 3, 3]);
        b.w(&format!("{rp}.block.5.bias"), &[lqh]);
    }
    let num_outputs = cfg.num_lq_outputs();
    for i in 0..num_outputs {
        b.w(&format!("lq_proj.output_heads.{i}.weight"), &[h, lqh]);
        b.w(
            &format!("lq_proj.gate_modules.{i}.content_proj.weight"),
            &[h, 2 * h],
        );
        b.w(&format!("lq_proj.gate_modules.{i}.log_alpha"), &[h]);
    }

    b.into_weights()
}

#[test]
fn pidnet_forward_shapes_and_finite() {
    let cfg = tiny_cfg();
    let w = build_pidnet_weights(&cfg);
    let net = PidNet::from_weights(&w, "", &cfg).expect("build tiny PidNet");

    let dev = Device::Cpu;
    // pixel side = zH · lsdf · sr_scale = 1 · 2 · 2 = 4; latent grid 1×1.
    let x = Tensor::randn(0f32, 1f32, &[1, 3, 4, 4], &dev).unwrap();
    let t = Tensor::from_vec(vec![500.0f32], (1,), &dev).unwrap();
    let y = Tensor::randn(
        0f32,
        1f32,
        &[1, cfg.txt_max_length as usize, cfg.txt_embed_dim as usize],
        &dev,
    )
    .unwrap();
    let lq = Tensor::randn(
        0f32,
        1f32,
        &[1, cfg.lq_latent_channels as usize, 1, 1],
        &dev,
    )
    .unwrap();
    let sigma = Tensor::from_vec(vec![0.0f32], (1,), &dev).unwrap();

    let out = net
        .forward(&x, &t, &y, &lq, &sigma)
        .expect("PidNet forward");
    assert_eq!(
        out.dims(),
        &[1, 3, 4, 4],
        "predicts a pixel tensor at the input resolution"
    );
    let flat = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!(
        flat.iter().all(|v| v.is_finite()),
        "forward output is finite"
    );
}

fn gemma_tiny_cfg() -> Gemma2Config {
    Gemma2Config {
        hidden_size: 8,
        num_layers: 1,
        num_heads: 2,
        num_kv_heads: 1,
        head_dim: 4,
        intermediate_size: 16,
        rope_theta: 10000.0,
        attn_softcap: 50.0,
        query_pre_attn_scalar: 4.0,
        rms_eps: 1e-6,
    }
}

fn build_gemma_weights(cfg: &Gemma2Config) -> Weights {
    let mut b = Builder::new();
    let h = cfg.hidden_size as usize;
    let vocab = 10usize;
    b.w("model.embed_tokens.weight", &[vocab, h]);
    b.w("model.norm.weight", &[h]);
    let l = "model.layers.0";
    for n in [
        "input_layernorm",
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
        "post_feedforward_layernorm",
    ] {
        b.w(&format!("{l}.{n}.weight"), &[h]);
    }
    let qh = cfg.num_heads as usize * cfg.head_dim as usize;
    let kh = cfg.num_kv_heads as usize * cfg.head_dim as usize;
    b.w(&format!("{l}.self_attn.q_proj.weight"), &[qh, h]);
    b.w(&format!("{l}.self_attn.k_proj.weight"), &[kh, h]);
    b.w(&format!("{l}.self_attn.v_proj.weight"), &[kh, h]);
    b.w(&format!("{l}.self_attn.o_proj.weight"), &[h, qh]);
    let im = cfg.intermediate_size as usize;
    b.w(&format!("{l}.mlp.gate_proj.weight"), &[im, h]);
    b.w(&format!("{l}.mlp.up_proj.weight"), &[im, h]);
    b.w(&format!("{l}.mlp.down_proj.weight"), &[h, im]);
    b.into_weights()
}

#[test]
fn gemma2_forward_shapes_and_finite() {
    let cfg = gemma_tiny_cfg();
    let w = build_gemma_weights(&cfg);
    let gemma = Gemma2::from_weights(&w, "model.", &cfg).expect("build tiny Gemma2");

    let dev = Device::Cpu;
    let ids = Tensor::from_vec(vec![1u32, 2, 3, 0, 0], (1, 5), &dev).unwrap();
    let mask = Tensor::from_vec(vec![1f32, 1.0, 1.0, 0.0, 0.0], (1, 5), &dev).unwrap();
    let out = gemma.forward(&ids, Some(&mask)).expect("Gemma2 forward");
    assert_eq!(out.dims(), &[1, 5, cfg.hidden_size as usize]);
    let flat = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!(flat.iter().all(|v| v.is_finite()), "gemma output is finite");
}
