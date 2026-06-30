//! sc-8412: exact-correctness gates for the candle FLUX.1-dev Fun-Controlnet-Union control branch, on
//! an in-memory tiny synthetic base ([`IpFlux`]) + control branch ([`FluxControlNet`]) — random small
//! weights, no checkpoint. Mirrors the merged candle-gen-flux2 `control_parity.rs` (sc-7460) and the mlx
//! `control_parity` gates. These prove the *mechanism* (the 6→19 residual injection + the `control_scale`
//! knob) is wired correctly via two invariants:
//!
//!   (a) **`scale = 0` is the base forward.** With `control_scale = 0` every control residual is
//!       multiplied by 0 and added (`+0`), so the control forward is byte-identical to the plain base
//!       forward — regardless of the (here random) control weights.
//!
//!   (b) **`scale > 0` actually injects.** With non-zero control weights + `scale = 0.8` the output
//!       *differs* from the base forward (the residuals flow into the base image stream) and stays finite
//!       — proving the control branch is a real contribution, not a silent no-op.
//!
//! The base is the BFL-layout vendored [`IpFlux`]; the control branch is the diffusers-layout
//! [`FluxControlNet`]. They share the tiny FLUX [`Config`]'s inner dims, so the 6 control residuals
//! (here 2) land cleanly on the base double stream at `interval = ceil(depth / num_residuals)`.

use std::collections::HashMap;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::{VarBuilder, VarMap};
use candle_gen_flux::control::{FluxControlNet, FluxControlNetConfig, FluxControlTransformer};
use candle_gen_flux::{FluxConfig, IpFlux};

/// A tiny base FLUX config: inner 16 (heads 2 · head_dim 8), `axes_dim` summing to 8, with the real
/// fixed input widths (in 64 / context 4096 / vec 768) the control branch's input projections require.
/// `depth = 6` double blocks so a 2-residual branch injects at `ceil(6/2) = 3` (after blocks 0, 3).
fn tiny_cfg() -> FluxConfig {
    FluxConfig {
        in_channels: 64,
        vec_in_dim: 768,
        context_in_dim: 4096,
        hidden_size: 16,
        mlp_ratio: 2.0,
        num_heads: 2,
        depth: 6,
        depth_single_blocks: 2,
        axes_dim: vec![2, 2, 4],
        theta: 10_000,
        qkv_bias: true,
        guidance_embed: true,
    }
}

/// Snapshot a VarMap into a name→Tensor map (so two `from_tensors` builders share identical weights).
fn snapshot(vm: &VarMap) -> HashMap<String, Tensor> {
    vm.data()
        .lock()
        .unwrap()
        .iter()
        .map(|(k, v)| (k.clone(), v.as_tensor().clone()))
        .collect()
}

struct Fixture {
    base: IpFlux,
    control: FluxControlTransformer,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    pooled: Tensor,
    timesteps: Tensor,
    guidance: Tensor,
    control_cond: Tensor,
}

impl Fixture {
    fn build() -> Self {
        let dev = Device::Cpu;
        let dtype = DType::F32;
        let cfg = tiny_cfg();

        // Two byte-identical bases (same weight map): one plain, one wrapped in the control branch.
        let base_vm = VarMap::new();
        {
            // Force-create the base weights by building once, then snapshot.
            let vb = VarBuilder::from_varmap(&base_vm, dtype, &dev);
            let _ = IpFlux::new(&cfg, vb).unwrap();
        }
        let bw = snapshot(&base_vm);
        let base = IpFlux::new(&cfg, VarBuilder::from_tensors(bw.clone(), dtype, &dev)).unwrap();
        let base2 = IpFlux::new(&cfg, VarBuilder::from_tensors(bw, dtype, &dev)).unwrap();

        // The control branch (2 layers) over the same tiny config.
        let ctrl_cfg = FluxControlNetConfig {
            num_layers: 2,
            supports_guidance: true,
        };
        let ctrl_vm = VarMap::new();
        {
            let vb = VarBuilder::from_varmap(&ctrl_vm, dtype, &dev);
            let _ = FluxControlNet::new(&cfg, &ctrl_cfg, vb).unwrap();
        }
        let cw = snapshot(&ctrl_vm);
        let branch =
            FluxControlNet::new(&cfg, &ctrl_cfg, VarBuilder::from_tensors(cw, dtype, &dev))
                .unwrap();
        let control = FluxControlTransformer::new(base2, branch);

        // A 2×2 latent grid (img_seq = 4) + a 3-token text sequence. The FLUX position ids carry 3 axes.
        let (lat_h, lat_w) = (2usize, 2usize);
        let img_seq = lat_h * lat_w;
        let txt_seq = 3usize;
        let b = 1usize;

        // Build the FLUX img_ids (axis 0 = 0, axis 1 = row, axis 2 = col) and zero txt_ids — the same
        // geometry the candle `State` builds.
        let mut ids = vec![0f32; img_seq * 3];
        for h in 0..lat_h {
            for w in 0..lat_w {
                let row = h * lat_w + w;
                ids[row * 3 + 1] = h as f32;
                ids[row * 3 + 2] = w as f32;
            }
        }
        let img_ids = Tensor::from_vec(ids, (b, img_seq, 3), &dev).unwrap();
        let txt_ids = Tensor::zeros((b, txt_seq, 3), dtype, &dev).unwrap();

        Self {
            base,
            control,
            img: Tensor::randn(0f32, 1f32, (b, img_seq, 64), &dev).unwrap(),
            img_ids,
            txt: Tensor::randn(0f32, 1f32, (b, txt_seq, 4096), &dev).unwrap(),
            txt_ids,
            pooled: Tensor::randn(0f32, 1f32, (b, 768), &dev).unwrap(),
            timesteps: Tensor::full(0.5f32, b, &dev).unwrap(),
            guidance: Tensor::full(3.5f32, b, &dev).unwrap(),
            control_cond: Tensor::randn(0f32, 0.3f32, (b, img_seq, 64), &dev).unwrap(),
        }
    }

    fn base_forward(&self) -> Tensor {
        self.base
            .forward(
                &self.img,
                &self.img_ids,
                &self.txt,
                &self.txt_ids,
                &self.timesteps,
                &self.pooled,
                Some(&self.guidance),
                None,
            )
            .unwrap()
    }

    fn control_forward(&self, scale: f64) -> Tensor {
        self.control
            .forward(
                &self.img,
                &self.img_ids,
                &self.txt,
                &self.txt_ids,
                &self.timesteps,
                &self.pooled,
                Some(&self.guidance),
                &self.control_cond,
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
fn residual_count_and_interval() {
    let f = Fixture::build();
    // 2 control residuals over the tiny 6-block base → interval ceil(6/2) = 3.
    assert_eq!(f.control.num_residuals(), 2);
    assert_eq!(f.control.residual_interval(), 3);
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
        "control_scale = 0 must be byte-identical to the base forward (residuals ×0); max|Δ| = {d}"
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
        "scale = 0.8 must change the output (residuals injected); max|Δ| = {d}"
    );
    let cv = controlled.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!(
        cv.iter().all(|x| x.is_finite()),
        "controlled output must be finite"
    );
}
