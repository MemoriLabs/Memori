//! Input cleanup and byte-packing helpers for embeddings.

/// Drops strings that are completely empty or contain only invisible characters
/// (whitespace, control characters like \0, or zero-width formatting spaces).
/// This ensures the embedder never wastes CPU cycles on semantic noise.
pub fn prepare_text_inputs(mut texts: Vec<String>) -> Vec<String> {
    texts.retain(|t| {
        // Keep the string if it contains at least one character that is NOT
        // whitespace, NOT a control character, and NOT a zero-width space.
        t.chars()
            .any(|c| !c.is_whitespace() && !c.is_control() && c != '\u{200B}')
    });
    texts
}

/// Fallback generator: creates zeroed vectors of length `dim`.
/// Used when the ONNX model panics, ensuring we don't break the user's data pipeline schema.
pub fn zero_vectors(count: usize, dim: usize) -> Vec<Vec<f32>> {
    vec![vec![0.0; dim]; count]
}

/// Formats a flat embedding as little-endian `f32` bytes for native database storage.
///
/// This is a pure-Rust equivalent to Python's `struct.pack('<...f', ...)`
/// and prevents us from needing heavy database-specific serialization crates.
pub fn format_embedding_for_db(embedding: &[f32]) -> Vec<u8> {
    embedding.iter().flat_map(|&f| f.to_le_bytes()).collect()
}

/// Parses little-endian `f32` bytes from native database storage into a flat embedding.
///
/// This is the inverse of `format_embedding_for_db`.
pub fn parse_embedding_from_db(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

/// Parses a batch of little-endian `f32` bytes back into `(flat_buffer, [rows, cols])`.
///
/// `dim` must match the dimension used when the embeddings were stored.
pub fn parse_embedding_batch_from_db(bytes: &[u8], dim: usize) -> (Vec<f32>, [usize; 2]) {
    let flat: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    let rows = if dim > 0 { flat.len() / dim } else { 0 };
    (flat, [rows, dim])
}
