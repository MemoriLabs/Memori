//! Token-length based text splitting. Ensures large documents don't crash the ONNX session context window.

use tokenizers::Tokenizer;

/// Splits `text` into token windows of at most `chunk_size` tokens.
///
/// Returns `None` if the text fits within a single window — the caller should reuse the original
/// owned string with zero allocations. Returns `Some(chunks)` with at least two entries when the
/// text was split.
pub fn chunk_text_by_tokens(
    text: &str,
    tokenizer: &Tokenizer,
    chunk_size: usize,
) -> Option<Vec<String>> {
    if chunk_size == 0 {
        log::warn!("chunk_size is 0, treating as single chunk");
        return None;
    }

    if text.trim().is_empty() {
        return None;
    }

    let encoding = match tokenizer.encode(text, false) {
        Ok(e) => e,
        Err(e) => {
            log::warn!("Failed to tokenize text: {}, treating as single chunk", e);
            return None;
        }
    };

    let ids = encoding.get_ids();

    if ids.len() <= chunk_size {
        return None;
    }

    let mut chunks = Vec::new();

    for chunk_ids in ids.chunks(chunk_size) {
        match tokenizer.decode(chunk_ids, true) {
            Ok(decoded) => {
                if !decoded.is_empty() {
                    chunks.push(decoded);
                }
            }
            Err(e) => {
                log::warn!("Failed to decode chunk: {}", e);
            }
        }
    }

    if chunks.len() <= 1 {
        log::warn!("Chunking produced <= 1 results, treating as single chunk");
        None
    } else {
        Some(chunks)
    }
}
