use crate::search::FactId;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingRow {
    pub id: FactId,
    #[serde(default)]
    pub content_embedding: Vec<f32>,
    #[serde(default, alias = "content_embedding_base64", alias = "embedding_b64")]
    pub content_embedding_b64: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateFactRow {
    pub id: FactId,
    pub content: String,
    pub date_created: String,
    #[serde(default)]
    pub summaries: Vec<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WriteOp {
    pub op_type: String,
    pub payload: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WriteBatch {
    pub ops: Vec<WriteOp>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WriteAck {
    pub written_ops: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankedFact {
    pub id: FactId,
    pub content: String,
    pub similarity: f32,
    pub rank_score: f32,
    pub date_created: String,
    #[serde(default)]
    pub summaries: Vec<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FetchEmbeddingsRequest {
    pub entity_id: String,
    pub limit: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FetchFactsByIdsRequest {
    pub ids: Vec<FactId>,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error, Serialize, Deserialize)]
#[error("host storage error ({code}): {message}")]
pub struct HostStorageError {
    pub code: String,
    pub message: String,
}

impl HostStorageError {
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
        }
    }
}
