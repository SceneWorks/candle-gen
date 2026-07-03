//! sc-7459: real-weights smoke for the candle single-file → diffusers converter
//! ([`candle_gen_flux2::convert_and_assemble`]) — the candle twin of mlx-gen-flux2's
//! `convert_real_weights.rs` (sc-3136). `#[ignore]`d — needs BOTH the base
//! `black-forest-labs/FLUX.2-klein-9B` diffusers snapshot AND the wikeeyang
//! `Flux2-Klein-9B-True-V2` `*-bf16.safetensors` single file:
//!
//! ```text
//! cargo test -p candle-gen-flux2 --test convert_real_weights -- --ignored --nocapture
//! ```
//!
//! The committed `convert` unit tests prove the remap *structure* (qkv split order, adaLN half-swap,
//! key renames) on synthetic tensors; this proves the *whole assembly* on the real fine-tune: the
//! converter's internal base-validation guard passes (produced keyset+shapes == base's), the borrowed
//! vae/text_encoder/tokenizer/scheduler resolve, and the assembled `transformer/` loads through the
//! production candle [`Flux2Transformer::new`] (every diffusers key resolves at the right shape) — i.e.
//! the dir is exactly what the `flux2_klein_9b` loader consumes via the worker's `modelPath` seam.
//!
//! Set `CANDLE_FLUX2_TRUE_V2_OUT=<dir>` to keep the converted dir for a follow-on GPU render
//! (`flux2-txt2img --snapshot <dir>`); otherwise it converts to a temp dir and cleans up.

use std::path::PathBuf;

use candle_gen::candle_core::{DType, Device};
use candle_gen::candle_nn::VarBuilder;
use candle_gen_flux2::config::Flux2Config;
use candle_gen_flux2::convert_and_assemble;
use candle_gen_flux2::transformer::Flux2Transformer;

/// Base FLUX.2-klein-9B diffusers snapshot (env `CANDLE_FLUX2_SNAPSHOT` or the HF cache).
///
/// F-069/F-071 (sc-9055/sc-9057): the HF-cache roots come from the shared
/// [`candle_gen::testkit::hf_cache_roots`] so this honours `$HF_HUB_CACHE` / `$HF_HOME` (not just the
/// Unix `$HOME/.cache` default). The base-specific `transformer/` predicate is kept.
fn base_snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("CANDLE_FLUX2_SNAPSHOT") {
        return PathBuf::from(p);
    }
    for snaps in candle_gen::testkit::hf_cache_roots()
        .into_iter()
        .map(|r| r.join("models--black-forest-labs--FLUX.2-klein-9B/snapshots"))
    {
        let Ok(revs) = std::fs::read_dir(&snaps) else {
            continue;
        };
        if let Some(dir) = revs
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .find(|p| p.is_dir() && p.join("transformer").is_dir())
        {
            return dir;
        }
    }
    panic!("no base FLUX.2-klein-9B snapshot with transformer/ under any HF cache root")
}

/// wikeeyang true_v2 single-file transformer, bf16 (env `CANDLE_FLUX2_TRUE_V2_FILE` or the HF cache).
/// This is the exact file the SceneWorks manifest's `convertSourceFile` targets.
///
/// F-069/F-071: HF-cache roots via the shared [`candle_gen::testkit::hf_cache_roots`] (honours
/// `$HF_HUB_CACHE` / `$HF_HOME`). The specific `*-bf16.safetensors` filename predicate is kept.
fn true_v2_bf16_file() -> PathBuf {
    if let Ok(p) = std::env::var("CANDLE_FLUX2_TRUE_V2_FILE") {
        return PathBuf::from(p);
    }
    for snaps in candle_gen::testkit::hf_cache_roots()
        .into_iter()
        .map(|r| r.join("models--wikeeyang--Flux2-Klein-9B-True-V2/snapshots"))
    {
        let Ok(revs) = std::fs::read_dir(&snaps) else {
            continue;
        };
        for snap in revs.filter_map(|e| e.ok()).map(|e| e.path()) {
            let file = snap.join("Flux2-Klein-9B-True-v2-bf16.safetensors");
            if file.is_file() {
                return file;
            }
        }
    }
    panic!(
        "missing wikeeyang true_v2 Flux2-Klein-9B-True-v2-bf16.safetensors under any HF cache root"
    )
}

#[test]
#[ignore = "needs base FLUX.2-klein-9B snapshot + wikeeyang true_v2 bf16 single file (~35 GB total)"]
fn convert_assembles_loadable_diffusers_dir() {
    let base = base_snapshot();
    let source = true_v2_bf16_file();
    // Persist to an explicit out dir (for a follow-on render) when CANDLE_FLUX2_TRUE_V2_OUT is set,
    // else a temp dir we clean up.
    let (out, keep) = match std::env::var("CANDLE_FLUX2_TRUE_V2_OUT") {
        Ok(p) => (PathBuf::from(p), true),
        Err(_) => (
            std::env::temp_dir().join("candle_flux2_true_v2_convert_out"),
            false,
        ),
    };
    let _ = std::fs::remove_dir_all(&out); // idempotent: clear any prior run

    // Convert + assemble. The internal base-validation guard asserts produced keyset+shapes match the
    // base diffusers transformer exactly — so a returned Ok is already structural proof.
    let assembled = convert_and_assemble(&source, &base, &out).expect("convert + assemble");
    assert_eq!(assembled, out);

    // The converted transformer is a real file with its borrowed config.json.
    let tf = out.join("transformer");
    assert!(
        tf.join("diffusion_pytorch_model.safetensors").is_file(),
        "converted transformer safetensors written"
    );
    assert!(
        tf.join("config.json").is_file(),
        "transformer config.json copied"
    );

    // Borrowed components resolve (hardlink tree on Windows, symlink on unix) + carry their files.
    for sub in ["vae", "text_encoder", "tokenizer", "scheduler"] {
        assert!(
            out.join(sub).is_dir(),
            "{sub} borrowed component resolves to a dir"
        );
    }
    assert!(
        out.join("model_index.json").is_file(),
        "model_index.json copied"
    );
    assert!(
        out.join("tokenizer/tokenizer.json").is_file(),
        "borrowed tokenizer.json resolves"
    );
    assert!(
        std::fs::read_dir(out.join("text_encoder"))
            .unwrap()
            .filter_map(|e| e.ok())
            .any(|e| e.path().extension().and_then(|s| s.to_str()) == Some("safetensors")),
        "borrowed text_encoder weights resolve"
    );

    // The converted transformer loads through the production candle loader: every diffusers key (and
    // the loader's expected shapes) resolves from the assembled dir. This is the "loadable" proof the
    // `flux2_klein_9b` engine relies on via the worker's `modelPath` seam.
    let mut files: Vec<PathBuf> = std::fs::read_dir(&tf)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors"))
        .collect();
    files.sort();
    // SAFETY: mmap of read-only weight files; standard candle loading path.
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&files, DType::F32, &Device::Cpu)
            .expect("VarBuilder over converted transformer")
    };
    Flux2Transformer::new(&Flux2Config::klein_9b(), vb)
        .expect("converted transformer loads via Flux2Transformer::new");

    println!(
        "flux2 true_v2 convert + assemble OK: loadable diffusers dir at {}",
        out.display()
    );
    if !keep {
        let _ = std::fs::remove_dir_all(&out);
    }
}
