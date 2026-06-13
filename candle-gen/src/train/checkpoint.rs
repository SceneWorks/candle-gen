//! Adapter-checkpoint naming (sc-5165). Intermediate checkpoints are written every `save_every`
//! micro-steps; the step is zero-padded so the files sort lexically. Mirrors the MLX
//! `checkpoint_filename`.

/// Strip a trailing `.safetensors` from an adapter file name to get the stem used for intermediate
/// checkpoints (`my_style.safetensors` → `my_style`).
pub fn file_stem(file_name: &str) -> &str {
    file_name.strip_suffix(".safetensors").unwrap_or(file_name)
}

/// `{stem}-step{step:06}.safetensors` — the intermediate-checkpoint file name at micro-step `step`.
pub fn checkpoint_filename(stem: &str, step: u32) -> String {
    format!("{stem}-step{step:06}.safetensors")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_zero_padded_and_sortable() {
        assert_eq!(file_stem("my_style.safetensors"), "my_style");
        assert_eq!(file_stem("noext"), "noext");
        assert_eq!(
            checkpoint_filename("my_style", 500),
            "my_style-step000500.safetensors"
        );
        // Lexical sort matches numeric order.
        assert!(checkpoint_filename("s", 90) < checkpoint_filename("s", 100));
    }
}
