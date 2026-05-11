pub mod mysql;
pub mod postgresql;
pub mod sqlite;

use sha2::{Digest, Sha256};
use uuid::Uuid;

pub fn new_uuid() -> String {
    Uuid::new_v4().to_string()
}

/// Stable content-addressable key used to deduplicate facts, subjects, and predicates.
/// SHA-256 is chosen for collision resistance — two identical strings always produce the
/// same `uniq`, so ON CONFLICT clauses can increment `num_times` instead of inserting duplicates.
pub fn generate_uniq(inputs: &[&str]) -> String {
    let mut hasher = Sha256::new();
    for input in inputs {
        hasher.update(input.as_bytes());
    }
    format!("{:x}", hasher.finalize())
}

/// Convert a `Vec<f32>` embedding to raw little-endian bytes for BLOB/BYTEA storage.
pub fn embedding_to_bytes(embedding: &[f32]) -> Vec<u8> {
    embedding.iter().flat_map(|f| f.to_le_bytes()).collect()
}
