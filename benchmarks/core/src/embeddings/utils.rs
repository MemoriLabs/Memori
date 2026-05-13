//! Input cleanup and byte-packing helpers for embeddings.

/// Drops strings that contain no visible characters (whitespace, control, or
/// zero-width space).
pub fn prepare_text_inputs(mut texts: Vec<String>) -> Vec<String> {
    texts.retain(|t| {
        t.chars()
            .any(|c| !c.is_whitespace() && !c.is_control() && c != '\u{200B}')
    });
    texts
}

/// Returns `count` zero vectors of length `dim`. Used as a fallback when
/// embedding generation fails so callers receive a well-formed output buffer.
pub fn zero_vectors(count: usize, dim: usize) -> Vec<Vec<f32>> {
    vec![vec![0.0; dim]; count]
}

/// Encodes a flat embedding as little-endian `f32` bytes for database storage.
pub fn format_embedding_for_db(embedding: &[f32]) -> Vec<u8> {
    embedding.iter().flat_map(|&f| f.to_le_bytes()).collect()
}

/// Decodes little-endian `f32` bytes into a flat embedding. Inverse of
/// [`format_embedding_for_db`]. Trailing bytes that don't form a complete
/// `f32` are ignored.
pub fn parse_embedding_from_db(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

/// Decodes a batch of little-endian `f32` bytes into `(flat_buffer, [rows, cols])`.
/// `dim` must match the dimension used when the embeddings were stored.
pub fn parse_embedding_batch_from_db(bytes: &[u8], dim: usize) -> (Vec<f32>, [usize; 2]) {
    let flat: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    let rows = flat.len().checked_div(dim).unwrap_or(0);
    (flat, [rows, dim])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepare_text_inputs_drops_empty_and_whitespace_only_strings() {
        let inputs = vec![
            "hello".to_string(),
            "".to_string(),
            "   ".to_string(),
            "\t\n".to_string(),
            "\u{200B}".to_string(),
            "world".to_string(),
        ];
        let filtered = prepare_text_inputs(inputs);
        assert_eq!(filtered, vec!["hello".to_string(), "world".to_string()]);
    }

    #[test]
    fn prepare_text_inputs_keeps_strings_with_any_visible_char() {
        let inputs = vec!["  a  ".to_string(), "\u{200B}b".to_string()];
        let filtered = prepare_text_inputs(inputs);
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn zero_vectors_has_expected_shape() {
        let v = zero_vectors(3, 4);
        assert_eq!(v.len(), 3);
        assert!(v.iter().all(|row| row.len() == 4));
        assert!(v.iter().flatten().all(|&x| x == 0.0));
    }

    #[test]
    fn format_and_parse_embedding_roundtrip() {
        let embedding = vec![0.0_f32, 1.0, -1.0, 0.5, 0.25];
        let bytes = format_embedding_for_db(&embedding);
        assert_eq!(bytes.len(), embedding.len() * 4);

        let parsed = parse_embedding_from_db(&bytes);
        assert_eq!(parsed, embedding);
    }

    #[test]
    fn parse_embedding_from_db_drops_trailing_partial_bytes() {
        let mut bytes = format_embedding_for_db(&[1.0_f32, 2.0]);
        bytes.push(0xAB);
        let parsed = parse_embedding_from_db(&bytes);
        assert_eq!(parsed, vec![1.0, 2.0]);
    }

    #[test]
    fn parse_embedding_batch_from_db_produces_correct_shape() {
        let rows: [[f32; 3]; 2] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let flat: Vec<u8> = rows
            .iter()
            .flat_map(|r| format_embedding_for_db(r))
            .collect();

        let (buffer, shape) = parse_embedding_batch_from_db(&flat, 3);
        assert_eq!(shape, [2, 3]);
        assert_eq!(buffer, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn parse_embedding_batch_with_zero_dim_yields_empty_rows() {
        let (_, shape) = parse_embedding_batch_from_db(&[0u8; 8], 0);
        assert_eq!(shape, [0, 0]);
    }
}
