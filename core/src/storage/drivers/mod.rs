pub mod mysql;
pub mod postgresql;
pub mod sqlite;

use sha2::{Digest, Sha256};
use uuid::Uuid;

/// Parses a triple entity node — either a plain string name or an `{name, type}` object.
/// Used by `knowledge_graph_create` across all dialect drivers.
pub(super) fn read_triple_entity(v: Option<&serde_json::Value>) -> (String, String) {
    match v {
        Some(serde_json::Value::String(s)) => (s.clone(), "entity".to_string()),
        Some(serde_json::Value::Object(map)) => {
            let name = map
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let typ = map
                .get("type")
                .and_then(|v| v.as_str())
                .unwrap_or("entity")
                .to_string();
            (name, typ)
        }
        _ => (String::new(), "entity".to_string()),
    }
}

pub fn new_uuid() -> String {
    Uuid::new_v4().to_string()
}

/// Stable content-addressable key used to deduplicate facts, subjects, and predicates.
/// Matches the Python SDK: join all terms, strip non-alphanumeric characters, lowercase, then SHA-256.
/// This ensures `uniq` values are consistent across the Rust and Python SDKs.
pub fn generate_uniq(inputs: &[&str]) -> String {
    let joined: String = inputs
        .iter()
        .flat_map(|s| s.chars())
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_lowercase())
        .collect();
    let mut hasher = Sha256::new();
    hasher.update(joined.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Convert a `Vec<f32>` embedding to raw little-endian bytes for BLOB/BYTEA storage.
pub fn embedding_to_bytes(embedding: &[f32]) -> Vec<u8> {
    embedding.iter().flat_map(|f| f.to_le_bytes()).collect()
}
