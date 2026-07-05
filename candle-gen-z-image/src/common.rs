//! Crate-private shared Z-Image plumbing (sc-9002 / F-022): the loader, VAE decode → RGB8, the
//! `[0,255] → [-1,1]` image preprocess, and the Qwen tokenizer policy that the three Z-Image entry
//! points — the registered txt2img [`crate::pipeline`], the [`crate::edit`] img2img provider, and the
//! [`crate::control`] Fun-ControlNet provider — used to each carry a near-identical copy of.
//!
//! Before this module the tokenizer constants (`QWEN_PAD_TOKEN_ID`, `TOKENIZER_MAX_LEN`, the
//! `ChatTemplate::QwenInstruct` config), the `[0,255] → [-1,1]` CHW normalize, the deterministic
//! VAE-encode **mean** (`(mean − shift) · scale`), and the `postprocess_image` → RGB8 decode all lived
//! in triplicate — and the sc-8646 empty-uncond fix already had to land in two of them. Any future
//! tokenizer-policy or VAE-scale fix now lands **once** here.
//!
//! Genuine per-entry-point differences are preserved as explicit parameters, NOT flattened:
//!
//! * **Resize policy** ([`ResizePolicy`]): pipeline resizes LANCZOS only when off-size, edit always
//!   resizes LANCZOS to the render size, control **requires** an exact-size control image (no silent
//!   stretch of a pose skeleton). Each site passes its own policy.
//! * **Text-encoder handle**: the txt2img pipeline runs the dense-or-packed [`crate::pipeline::TextEnc`]
//!   enum, edit/control run the stock `ZImageTextEncoder` directly. So the shared tokenizer helpers
//!   return **token ids**, and [`encode_ids`] runs *any* encoder via a `forward` closure — the encode
//!   math (batch axis, dtype cast) is shared, the model handle is the caller's.
//! * **Encoded-context shape**: edit/pipeline use the raw 16-channel encode **mean** for the img2img
//!   init latent; control appends a zero mask + zero inpaint group to reach the 33-channel
//!   Fun-Controlnet-Union layout. So [`encode_mean`] returns the shared 16-ch mean and control does its
//!   own channel-cat on top.

use candle_gen::candle_core::{DType, Device, IndexOp, Tensor};
use candle_gen::gen_core::imageops::resize_lanczos_u8;
use candle_gen::gen_core::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use candle_gen::gen_core::Image;
use candle_gen::{CandleError, LatentDecoder, Result};
use candle_gen_pid::PidDecoder;
use candle_transformers::models::z_image::sampling::postprocess_image;
use candle_transformers::models::z_image::vae::{AutoEncoderKL, Encoder as VaeEncoder};
use rand::{rngs::StdRng, SeedableRng};

/// VAE spatial downscale — the latent is image/8 per side (the 4-stage AutoencoderKL has 3
/// downsamplers). Matches `mlx-gen-z-image`'s `SPATIAL_SCALE`.
pub(crate) const SPATIAL_SCALE: u32 = 8;

/// DiT patch size on each spatial axis (`Config::z_image_turbo().all_patch_size[0]`). The flow-match
/// `mu` shift is computed from the post-patchify image sequence length.
pub(crate) const PATCH_SIZE: u32 = 2;

/// Z-Image latent channel count (the VAE's `latent_channels` and the DiT's `in_channels`).
pub(crate) const LATENT_CHANNELS: usize = 16;

/// Qwen3 pad token id (`<|endoftext|>`). Only consulted when padding to a fixed length, which the
/// Z-Image tokenizer config does not do (`pad_to_max_length: false`); carried for correctness/parity
/// with the mlx loader.
pub(crate) const QWEN_PAD_TOKEN_ID: i32 = 151643;

/// Right-truncation cap for prompt tokenization (HF single-sequence truncation). Z-Image prompts are
/// short; 512 is generous and never engages in practice.
pub(crate) const TOKENIZER_MAX_LEN: usize = 512;

/// The Z-Image image encoder's dtype (f32): the VAE encode path runs f32, then the mean is cast to the
/// compute dtype (bf16) for the init/control latent.
pub(crate) const ENC_DTYPE: DType = DType::F32;

/// How a site fits a source image to the render size before the `[0,255] → [-1,1]` normalize — a
/// genuine per-entry-point difference (preserved, not flattened, sc-9002).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ResizePolicy {
    /// LANCZOS-resize to the render size, but skip the resize when the source is already at the render
    /// size (the base img2img / `Reference` path — [`crate::pipeline::encode_reference`]).
    ResizeIfNeeded,
    /// Always LANCZOS-resize to the render size (the [`crate::edit`] img2img path — the worker pre-fits,
    /// but resizing here keeps the provider robust to an off-size source).
    ResizeAlways,
    /// Require the image already at the render size; error otherwise (the [`crate::control`] pose path —
    /// the worker renders the skeleton at the target size, and a silent stretch would distort the pose).
    RequireExact,
}

/// The shared Z-Image Qwen tokenizer config (chat template + max-length policy). One home so the three
/// entry points can never drift on the tokenization policy (`pad_to_max_length: false` is load-bearing
/// for the sc-8646 empty-uncond path — see [`uncond_ids`]).
pub(crate) fn tokenizer_config() -> TokenizerConfig {
    TokenizerConfig {
        max_length: TOKENIZER_MAX_LEN,
        pad_token_id: QWEN_PAD_TOKEN_ID,
        chat_template: ChatTemplate::QwenInstruct,
        pad_to_max_length: false,
    }
}

/// Build the Z-Image Qwen tokenizer from `root/tokenizer/tokenizer.json`. `label` names the site in the
/// error (`"z-image"`, `"z-image edit"`, `"z-image control"`).
pub(crate) fn build_tokenizer(root: &std::path::Path, label: &str) -> Result<TextTokenizer> {
    TextTokenizer::from_file(root.join("tokenizer/tokenizer.json"), tokenizer_config())
        .map_err(|e| CandleError::Msg(format!("{label}: load tokenizer: {e}")))
}

/// Prompt → the Qwen chat-template token ids (non-empty), erroring on an empty tokenization. Shared by
/// every conditional encode path; the caller runs its own text encoder over the ids via [`encode_ids`].
///
/// `tok` is the cached tokenizer ([`build_tokenizer`]) the caller holds on its `Components` — loaded
/// once, reused across encodes (sc-8991 / F-011) rather than re-parsing `tokenizer.json` per prompt.
pub(crate) fn prompt_ids(tok: &TextTokenizer, prompt: &str, label: &str) -> Result<Vec<i32>> {
    let out = tok
        .tokenize(prompt)
        .map_err(|e| CandleError::Msg(format!("{label}: tokenize: {e}")))?;
    if out.ids.is_empty() {
        return Err(CandleError::Msg(format!("{label}: empty prompt")));
    }
    Ok(out.ids)
}

/// Negative prompt → the Qwen chat-template token ids for the **unconditional** CFG branch (sc-8646).
///
/// The empty-string case must NOT route through [`prompt_ids`]: gen-core's [`TextTokenizer::tokenize`]
/// short-circuits an empty prompt to a `(1, 0)` sequence **before** the chat template is applied (the
/// config has `pad_to_max_length = false`), so an empty negative prompt would trip the empty-`ids`
/// guard and error. Instead the QwenInstruct scaffolding around `""` is rendered via
/// [`TextTokenizer::encode_chat_ids`] — `<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n` — which
/// tokenizes to the non-empty role-marker sequence the reference `mlx-gen-z-image` feeds its uncond
/// branch. A non-empty negative prompt takes the ordinary [`prompt_ids`] path.
pub(crate) fn uncond_ids(
    tok: &TextTokenizer,
    negative_prompt: &str,
    label: &str,
) -> Result<Vec<i32>> {
    if !negative_prompt.is_empty() {
        return prompt_ids(tok, negative_prompt, label);
    }
    // `add_special_tokens = true` mirrors `tokenize`'s `encode(text, true)`. For Qwen this only governs
    // the auto-added BOS/EOS (Qwen adds none), so the ids equal the templated tokens.
    let ids = tok
        .encode_chat_ids("", true)
        .map_err(|e| CandleError::Msg(format!("{label}: tokenize uncond: {e}")))?;
    if ids.is_empty() {
        return Err(CandleError::Msg(format!(
            "{label}: unconditional embedding tokenized to an empty sequence"
        )));
    }
    Ok(ids)
}

/// Token `ids` → `cap_feats` `(seq, 2560)` at `dtype`: build the `(1, L)` input, run the caller's text
/// encoder `forward`, squeeze the batch axis, cast to the compute dtype. The reference `prepare_inputs`
/// does the SEQ_MULTI_OF padding + attention mask downstream, so every id here is a valid token.
///
/// `forward` is the caller's encoder — the dense-or-packed [`crate::pipeline::TextEnc`] enum or the
/// stock `ZImageTextEncoder` — so the encode math is shared while the model handle stays the caller's.
pub(crate) fn encode_ids<F>(
    ids: &[i32],
    device: &Device,
    dtype: DType,
    forward: F,
) -> Result<Tensor>
where
    F: FnOnce(&Tensor) -> candle_gen::candle_core::Result<Tensor>,
{
    // candle embeddings index with u32; the chat-template ids are small non-negative Qwen ids.
    let ids: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
    let len = ids.len();
    let input_ids = Tensor::from_vec(ids, (1, len), device)?;
    let enc = forward(&input_ids)?; // (1, L, 2560)
    Ok(enc.squeeze(0)?.to_dtype(dtype)?) // (L, 2560)
}

/// Deterministic, launch-portable initial latent noise (sc-3673): N(0,1) from a fixed-algorithm CPU
/// RNG (`StdRng`, ChaCha) seeded by `seed`, built on CPU then moved to `device` at `dtype`. NOT
/// candle's device `randn` (its seed→noise mapping is not launch-portable). Shared by all three render
/// loops so generation is a pure function of `(seed, request)`.
pub(crate) fn seed_noise(
    seed: u64,
    lat_h: usize,
    lat_w: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let n = LATENT_CHANNELS * lat_h * lat_w;
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = candle_gen::seeded_normal_vec(&mut rng, n);
    Ok(
        Tensor::from_vec(noise, (1, LATENT_CHANNELS, lat_h, lat_w), &Device::Cpu)?
            .to_device(device)?
            .to_dtype(dtype)?,
    )
}

/// An RGB8 image → `[1, 3, H, W]` f32 in `[-1, 1]` (the VAE encoder's input range), fit to the render
/// `width × height` per the [`ResizePolicy`], then normalized `p/127.5 − 1.0` HWC → CHW. `label` names
/// the site in buffer/size errors. The output is at [`ENC_DTYPE`] (f32).
pub(crate) fn preprocess_image(
    image: &Image,
    width: u32,
    height: u32,
    policy: ResizePolicy,
    device: &Device,
    label: &str,
) -> Result<Tensor> {
    let (iw, ih) = (image.width as usize, image.height as usize);
    if image.pixels.len() != iw * ih * 3 {
        return Err(CandleError::Msg(format!(
            "{label}: image buffer {} != {iw}x{ih}x3",
            image.pixels.len()
        )));
    }
    let (rw, rh) = (width as usize, height as usize);
    let resized: Vec<f32> = match policy {
        ResizePolicy::RequireExact => {
            if image.width != width || image.height != height {
                return Err(CandleError::Msg(format!(
                    "{label}: image {iw}x{ih} must match the request {width}x{height}"
                )));
            }
            image.pixels.iter().map(|&p| p as f32).collect()
        }
        ResizePolicy::ResizeIfNeeded if (ih, iw) == (rh, rw) => {
            image.pixels.iter().map(|&p| p as f32).collect()
        }
        ResizePolicy::ResizeIfNeeded | ResizePolicy::ResizeAlways => {
            resize_lanczos_u8(&image.pixels, ih, iw, rh, rw)? // HWC f32 [0,255]
        }
    };
    // [0,255] → [-1,1], HWC → CHW.
    let mut data = vec![0f32; 3 * rh * rw];
    for y in 0..rh {
        for x in 0..rw {
            for c in 0..3 {
                data[c * rh * rw + y * rw + x] = resized[(y * rw + x) * 3 + c] / 127.5 - 1.0;
            }
        }
    }
    Ok(Tensor::from_vec(data, (1, 3, rh, rw), device)?.to_dtype(ENC_DTYPE)?)
}

/// Deterministic VAE-encode **mean** of a preprocessed `[1, 3, H, W]` f32 image → latent
/// `(1, 16, H/8, W/8)` at `out_dtype`: run the raw `Encoder` (candle's `AutoEncoderKL::encode` samples
/// via the *device* RNG, not launch-portable — sc-3673), take the distribution mean (not a sampled
/// draw), and map to latent space as `(mean − shift) · scale`. The 16-ch mean the img2img init and the
/// control context both build on (control appends its own mask/inpaint groups downstream).
pub(crate) fn encode_mean(
    encoder: &VaeEncoder,
    img: &Tensor,
    shift: f64,
    scale: f64,
    out_dtype: DType,
) -> Result<Tensor> {
    let moments = img.apply(encoder)?; // (1, 32, H/8, W/8) — [mean | logvar]
    let mean = moments.chunk(2, 1)?[0].clone(); // (1, 16, H/8, W/8)
    let latents = ((mean - shift)? * scale)?;
    Ok(latents.to_dtype(out_dtype)?)
}

/// VAE-decode the final latents `(1, 16, 1, h, w)` to an RGB8 [`Image`]. The VAE applies its own
/// `/scaling_factor + shift_factor` un-scale inside `decode`; `postprocess_image` maps the `[-1, 1]`
/// output to `[0, 255]` u8. Byte-identical across the three entry points.
pub(crate) fn decode(
    vae: &AutoEncoderKL,
    pid: Option<&PidDecoder>,
    latents: &Tensor,
) -> Result<Image> {
    // Drop the singleton frame axis: (1, 16, 1, h, w) -> (1, 16, h, w). This is the exact NCHW latent
    // the VAE decode receives — and the same one PiD consumes (a zero-transform seam, epic 7840 /
    // sc-7853). When a PiD decoder resolved, the `zimage-turbo`-tagged `flux`-student (Z-Image aliases
    // the FLUX.1 latent space) super-resolves; else the native VAE (its own `/scaling + shift` un-scale
    // is applied inside `decode`). PiD emits a larger `[1,3,4H,4W]` tensor; `postprocess_image` reads
    // the size from the tensor (never `latent*8`).
    let latents = latents.squeeze(2)?;
    let decoded = match pid {
        Some(pid) => pid.decode(&latents)?,
        None => vae.decode(&latents)?.to_dtype(DType::F32)?, // (1, 3, H, W) in [-1, 1]
    };
    let img = postprocess_image(&decoded)? // u8 (1, 3, H, W)
        .i(0)?
        .to_device(&Device::Cpu)?;
    let (c, h, w) = img.dims3()?;
    if c != 3 {
        return Err(CandleError::Msg(format!("expected 3 channels, got {c}")));
    }
    let pixels = img.permute((1, 2, 0))?.flatten_all()?.to_vec1::<u8>()?;
    Ok(Image {
        width: w as u32,
        height: h as u32,
        pixels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shared preprocess resizes/fits per policy and maps `[0,255] → [-1,1]` in CHW f32. A solid
    /// white source ⇒ all ≈ 1.0; a truncated buffer errors; `RequireExact` rejects an off-size image
    /// while the resize policies fit it. (GPU-free — CPU tensors only.)
    #[test]
    fn preprocess_resize_and_normalize() {
        let white = |w: u32, h: u32| Image {
            width: w,
            height: h,
            pixels: vec![255u8; (w * h * 3) as usize],
        };

        // ResizeAlways: 8x8 → 16x16, 255 → 1.0.
        let t = preprocess_image(
            &white(8, 8),
            16,
            16,
            ResizePolicy::ResizeAlways,
            &Device::Cpu,
            "z-image test",
        )
        .unwrap();
        assert_eq!(t.dims(), &[1, 3, 16, 16]);
        let v = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(v.iter().all(|x| (x - 1.0).abs() < 1e-3));

        // ResizeIfNeeded at the render size is a no-op resize (still normalizes to 1.0).
        let t2 = preprocess_image(
            &white(16, 16),
            16,
            16,
            ResizePolicy::ResizeIfNeeded,
            &Device::Cpu,
            "z-image test",
        )
        .unwrap();
        assert_eq!(t2.dims(), &[1, 3, 16, 16]);
        assert!(t2
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|x| (x - 1.0).abs() < 1e-3));

        // RequireExact: exact size ok, off-size errors (no silent stretch).
        assert!(preprocess_image(
            &white(16, 16),
            16,
            16,
            ResizePolicy::RequireExact,
            &Device::Cpu,
            "z-image test"
        )
        .is_ok());
        assert!(preprocess_image(
            &white(16, 16),
            32,
            16,
            ResizePolicy::RequireExact,
            &Device::Cpu,
            "z-image test"
        )
        .is_err());

        // A truncated buffer errors on every policy.
        let bad = Image {
            width: 8,
            height: 8,
            pixels: vec![0u8; 8 * 8 * 3 - 1],
        };
        assert!(preprocess_image(
            &bad,
            8,
            8,
            ResizePolicy::ResizeAlways,
            &Device::Cpu,
            "z-image test"
        )
        .is_err());
    }

    /// The tokenizer policy is the single shared config: chat template QwenInstruct, no
    /// pad-to-max-length (load-bearing for the sc-8646 empty-uncond path), the Qwen pad id and cap.
    #[test]
    fn tokenizer_config_is_the_shared_policy() {
        let c = tokenizer_config();
        assert_eq!(c.max_length, TOKENIZER_MAX_LEN);
        assert_eq!(c.pad_token_id, QWEN_PAD_TOKEN_ID);
        assert!(matches!(c.chat_template, ChatTemplate::QwenInstruct));
        assert!(
            !c.pad_to_max_length,
            "pad_to_max_length must stay false (sc-8646 empty-uncond depends on the short-circuit)"
        );
    }

    /// `encode_ids` builds a `(1, L)` u32 input, runs the caller's `forward`, and squeezes the batch
    /// axis to `(L, hidden)` at the requested dtype — model-agnostic (a stub encoder here). GPU-free.
    #[test]
    fn encode_ids_shapes_and_dtype() {
        // Stub "encoder": maps (1, L) ids → (1, L, 4) of ones, so we exercise the batch-axis squeeze +
        // dtype cast without a real model.
        let ids = vec![3i32, 5, 7];
        let out = encode_ids(&ids, &Device::Cpu, DType::F32, |input| {
            let (b, l) = input.dims2()?;
            Tensor::ones((b, l, 4), DType::F32, input.device())
        })
        .unwrap();
        assert_eq!(out.dims(), &[3, 4]); // batch axis squeezed → (L, hidden)
        assert_eq!(out.dtype(), DType::F32);
    }

    /// `seed_noise` is deterministic and launch-portable: same seed ⇒ byte-identical draw; the shape is
    /// the 16-channel latent prior at the given latent geometry. GPU-free (CPU→CPU).
    #[test]
    fn seed_noise_is_deterministic() {
        let a = seed_noise(42, 4, 6, &Device::Cpu, DType::F32).unwrap();
        let b = seed_noise(42, 4, 6, &Device::Cpu, DType::F32).unwrap();
        assert_eq!(a.dims(), &[1, LATENT_CHANNELS, 4, 6]);
        let (va, vb) = (
            a.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            b.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        );
        assert_eq!(va, vb, "same seed ⇒ identical noise");
        // A different seed diverges.
        let c = seed_noise(43, 4, 6, &Device::Cpu, DType::F32).unwrap();
        assert_ne!(va, c.flatten_all().unwrap().to_vec1::<f32>().unwrap());
    }
}
