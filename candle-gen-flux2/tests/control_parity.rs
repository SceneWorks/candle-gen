//! sc-7460: exact-correctness gates for the candle FLUX.2-dev Fun-Controlnet-Union (VACE) control
//! branch, on an in-memory tiny synthetic base transformer + control branch (random small weights, no
//! checkpoint). Mirrors the merged mlx-gen `control_parity.rs` (sc-2292). These prove the *mechanism*
//! (hint injection + the `control_context_scale` knob) is wired correctly via two invariants:
//!
//!   (a) **`scale = 0` is the base forward.** With `control_context_scale = 0` every control hint is
//!       multiplied by 0 and added (`+0`), so `forward_with_control(scale = 0)` is byte-identical to the
//!       plain `forward` — regardless of the (here random) control weights.
//!
//!   (b) **`scale > 0` actually injects.** With non-zero control weights + `scale = 0.8` the output
//!       *differs* from the base forward (the hints flow into the base image stream) and stays finite —
//!       proving the control branch is a real contribution, not a silent no-op.

use std::collections::HashMap;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen_flux2::config::Flux2Config;
use candle_gen_flux2::pipeline::{prepare_grid_ids, prepare_text_ids};
use candle_gen_flux2::{
    Flux2ControlBranch, Flux2ControlTransformer, Flux2Transformer, CONTROL_IN_DIM,
};

const TS: f32 = 500.0;

/// Tiny config: `inner = num_heads·head_dim = 2·8 = 16`, `num_double_layers = 1` → `control_layers =
/// [0]` → one control block injecting at base double block 0. The TE fields are unused by the
/// transformer but the struct requires them.
fn tiny() -> Flux2Config {
    Flux2Config {
        num_double_layers: 1,
        num_single_layers: 1,
        num_heads: 2,
        head_dim: 8,
        in_channels: 4,
        out_channels: 4,
        joint_attention_dim: 12,
        mlp_ratio: 3.0,
        timestep_channels: 16,
        axes_dim: [2, 2, 2, 2],
        rope_theta: 2000.0,
        te_hidden_size: 4,
        te_intermediate_size: 12,
        te_n_layers: 1,
        te_n_heads: 2,
        te_n_kv_heads: 1,
        te_head_dim: 2,
        te_rope_theta: 1_000_000.0,
        te_rms_norm_eps: 1e-6,
        te_qk_norm: false,
        te_vocab_size: 16,
        te_prefix: "model",
        te_out_layers: [0, 1, 2],
        max_sequence_length: 512,
        num_latent_channels: 1,
        vae_scale_factor: 8,
    }
}

fn randn(mean: f32, std: f32, shape: &[usize], dev: &Device) -> Tensor {
    Tensor::randn(mean, std, shape, dev).unwrap()
}

/// A bias-less `[out, in]` projection (`{key}.weight`), small enough to keep the tiny forward finite.
fn lin(m: &mut HashMap<String, Tensor>, key: &str, out: usize, inn: usize, dev: &Device) {
    m.insert(format!("{key}.weight"), randn(0.0, 0.05, &[out, inn], dev));
}

/// A bias-carrying `[out, in]` projection (`{key}.weight` + `{key}.bias`) — the control branch's
/// `control_img_in` / `before_proj` / `after_proj`.
fn lin_b(m: &mut HashMap<String, Tensor>, key: &str, out: usize, inn: usize, dev: &Device) {
    lin(m, key, out, inn, dev);
    m.insert(format!("{key}.bias"), randn(0.0, 0.02, &[out], dev));
}

/// An RMSNorm weight (`{key}.weight`), near 1.0 so the norm is meaningful.
fn norm(m: &mut HashMap<String, Tensor>, key: &str, dim: usize, dev: &Device) {
    m.insert(
        format!("{key}.weight"),
        (randn(0.0, 0.02, &[dim], dev) + 1.0).unwrap(),
    );
}

/// A full FLUX.2 double block's weights under `p` (`attn.*` + `ff.*` + `ff_context.*`), diffusers
/// naming (`attn.to_out.0` read natively by the candle `DoubleBlock`). Shared by the base blocks and
/// the control blocks.
fn double_block(
    m: &mut HashMap<String, Tensor>,
    p: &str,
    inner: usize,
    hd: usize,
    ff: usize,
    dev: &Device,
) {
    let a = format!("{p}.attn");
    for n in [
        "to_q",
        "to_k",
        "to_v",
        "add_q_proj",
        "add_k_proj",
        "add_v_proj",
        "to_add_out",
    ] {
        lin(m, &format!("{a}.{n}"), inner, inner, dev);
    }
    lin(m, &format!("{a}.to_out.0"), inner, inner, dev);
    for n in ["norm_q", "norm_k", "norm_added_q", "norm_added_k"] {
        norm(m, &format!("{a}.{n}"), hd, dev);
    }
    for ffn in ["ff", "ff_context"] {
        lin(m, &format!("{p}.{ffn}.linear_in"), 2 * ff, inner, dev);
        lin(m, &format!("{p}.{ffn}.linear_out"), inner, ff, dev);
    }
}

fn base_weights(cfg: &Flux2Config, dev: &Device) -> HashMap<String, Tensor> {
    let inner = cfg.inner_dim();
    let hd = cfg.head_dim;
    let ff = (cfg.mlp_ratio * inner as f32) as usize;
    let smlp = cfg.single_mlp_hidden();
    let mut m = HashMap::new();
    lin(&mut m, "x_embedder", inner, cfg.in_channels, dev);
    lin(
        &mut m,
        "context_embedder",
        inner,
        cfg.joint_attention_dim,
        dev,
    );
    lin(
        &mut m,
        "time_guidance_embed.timestep_embedder.linear_1",
        inner,
        cfg.timestep_channels,
        dev,
    );
    lin(
        &mut m,
        "time_guidance_embed.timestep_embedder.linear_2",
        inner,
        inner,
        dev,
    );
    lin(
        &mut m,
        "double_stream_modulation_img.linear",
        3 * 2 * inner,
        inner,
        dev,
    );
    lin(
        &mut m,
        "double_stream_modulation_txt.linear",
        3 * 2 * inner,
        inner,
        dev,
    );
    lin(
        &mut m,
        "single_stream_modulation.linear",
        3 * inner,
        inner,
        dev,
    );
    for i in 0..cfg.num_double_layers {
        double_block(
            &mut m,
            &format!("transformer_blocks.{i}"),
            inner,
            hd,
            ff,
            dev,
        );
    }
    for i in 0..cfg.num_single_layers {
        let a = format!("single_transformer_blocks.{i}.attn");
        lin(
            &mut m,
            &format!("{a}.to_qkv_mlp_proj"),
            3 * inner + 2 * smlp,
            inner,
            dev,
        );
        lin(&mut m, &format!("{a}.to_out"), inner, inner + smlp, dev);
        norm(&mut m, &format!("{a}.norm_q"), hd, dev);
        norm(&mut m, &format!("{a}.norm_k"), hd, dev);
    }
    lin(&mut m, "norm_out.linear", 2 * inner, inner, dev);
    lin(&mut m, "proj_out", cfg.out_channels, inner, dev);
    m
}

fn control_weights(cfg: &Flux2Config, dev: &Device) -> HashMap<String, Tensor> {
    let inner = cfg.inner_dim();
    let hd = cfg.head_dim;
    let ff = (cfg.mlp_ratio * inner as f32) as usize;
    let mut m = HashMap::new();
    // `control_img_in`: CONTROL_IN_DIM (260) → inner, bias-carrying, stays dense.
    lin_b(&mut m, "control_img_in", inner, CONTROL_IN_DIM, dev);
    let places = cfg.control_layer_places();
    for i in 0..places.len() {
        let p = format!("control_transformer_blocks.{i}");
        double_block(&mut m, &p, inner, hd, ff, dev);
        lin_b(&mut m, &format!("{p}.after_proj"), inner, inner, dev);
        if i == 0 {
            lin_b(&mut m, &format!("{p}.before_proj"), inner, inner, dev);
        }
    }
    m
}

struct Fixture {
    base: Flux2Transformer,
    control: Flux2ControlTransformer,
    hidden: Tensor,
    encoder: Tensor,
    img_ids: Vec<[i64; 4]>,
    txt_ids: Vec<[i64; 4]>,
    control_context: Tensor,
}

impl Fixture {
    fn build() -> Self {
        let dev = Device::Cpu;
        let cfg = tiny();
        // Two byte-identical bases (same weight map): one plain, one wrapped in the control branch.
        let bw = base_weights(&cfg, &dev);
        let base =
            Flux2Transformer::new(&cfg, VarBuilder::from_tensors(bw.clone(), DType::F32, &dev))
                .unwrap();
        let base2 =
            Flux2Transformer::new(&cfg, VarBuilder::from_tensors(bw, DType::F32, &dev)).unwrap();
        let branch = Flux2ControlBranch::new(
            &cfg,
            VarBuilder::from_tensors(control_weights(&cfg, &dev), DType::F32, &dev),
        )
        .unwrap();
        let control = Flux2ControlTransformer::new(base2, branch);

        let (lat_h, lat_w) = (2usize, 2usize);
        let target_seq = lat_h * lat_w;
        let txt_seq = 3usize;
        Self {
            base,
            control,
            hidden: randn(0.0, 1.0, &[1, target_seq, cfg.in_channels], &dev),
            encoder: randn(0.0, 1.0, &[1, txt_seq, cfg.joint_attention_dim], &dev),
            img_ids: prepare_grid_ids(lat_h, lat_w),
            txt_ids: prepare_text_ids(txt_seq),
            // The control context shares the target image grid (same seq), width = CONTROL_IN_DIM.
            control_context: randn(0.0, 0.3, &[1, target_seq, CONTROL_IN_DIM], &dev),
        }
    }

    fn base_forward(&self) -> Tensor {
        self.base
            .forward(
                &self.hidden,
                &self.encoder,
                &self.img_ids,
                &self.txt_ids,
                TS,
                None,
            )
            .unwrap()
    }

    fn control_forward(&self, scale: f32) -> Tensor {
        self.control
            .forward(
                &self.hidden,
                &self.encoder,
                &self.img_ids,
                &self.txt_ids,
                TS,
                None, // tiny base has no guidance embedder
                &self.control_context,
                scale,
            )
            .unwrap()
    }
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
    let av = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let bv = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    av.iter()
        .zip(&bv)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

#[test]
fn control_in_dim_is_260_for_dev() {
    // The dev control context width: control latent (128) + mask (4) + inpaint latent (128).
    let c = Flux2Config::dev();
    let in_ch = c.in_channels;
    let mask_ch = in_ch / c.num_latent_channels;
    assert_eq!(in_ch + mask_ch + in_ch, CONTROL_IN_DIM);
    assert_eq!(CONTROL_IN_DIM, 260);
}

#[test]
fn control_layers_are_every_other_double_block() {
    assert_eq!(Flux2Config::dev().control_layer_places(), vec![0, 2, 4, 6]);
    assert_eq!(tiny().control_layer_places(), vec![0]);
}

#[test]
fn scale_zero_equals_base_forward() {
    let f = Fixture::build();
    let base = f.base_forward();
    let control0 = f.control_forward(0.0);
    assert_eq!(control0.dims(), base.dims());
    let d = max_abs_diff(&base, &control0);
    assert!(
        d == 0.0,
        "control_context_scale = 0 must be byte-identical to the base forward (hints ×0); max|Δ| = {d}"
    );
}

#[test]
fn scale_nonzero_injects_and_stays_finite() {
    let f = Fixture::build();
    let base = f.base_forward();
    let controlled = f.control_forward(0.8);
    assert_eq!(controlled.dims(), base.dims());
    let d = max_abs_diff(&base, &controlled);
    assert!(
        d > 1e-6,
        "scale = 0.8 must change the output (hints injected); max|Δ| = {d}"
    );
    let cv = controlled.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!(
        cv.iter().all(|x| x.is_finite()),
        "controlled output must be finite"
    );
}
