use napi::Either;
use napi::bindgen_prelude::Float32Array;
use napi_derive::napi;
use serde::{Deserialize, Serialize};

// --- Core API Structs ---

#[napi(object)]
#[derive(Serialize, Deserialize)]
pub struct NapiRetrievalRequest {
    pub entity_id: String,
    pub query_text: String,
    pub dense_limit: u32,
    pub limit: u32,
}

#[napi(object)]
#[derive(Serialize, Deserialize)]
pub struct NapiRecallSummary {
    pub content: String,
    pub date_created: String,
    pub entity_fact_id: Option<i64>,
    pub fact_id: Option<i64>,
}

#[napi(object)]
#[derive(Serialize, Deserialize)]
pub struct NapiRecallObject {
    pub id: i64,
    pub content: String,
    pub rank_score: Option<f64>,
    pub similarity: Option<f64>,
    pub date_created: Option<String>,
    pub summaries: Option<Vec<NapiRecallSummary>>,
}

#[napi(object)]
#[derive(Serialize, Deserialize)]
pub struct NapiMessage {
    pub role: String,
    pub content: String,
}

#[napi(object)]
#[derive(Serialize, Deserialize)]
pub struct NapiAugmentationInput {
    pub entity_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub process_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversation_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversation_messages: Option<Vec<NapiMessage>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub llm_provider: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub llm_model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub llm_provider_sdk_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub framework: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub platform_provider: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub storage_dialect: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub storage_cockroachdb: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sdk_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub use_mock_response: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fact_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

// --- Storage Bridge Structs ---

#[napi(object)]
pub struct NapiEmbeddingRow {
    pub id: Either<i64, String>,
    pub content_embedding: Float32Array,
}

#[napi(object)]
#[derive(Serialize)]
pub struct NapiCandidateSummaryRow {
    pub content: String,
    pub date_created: String,
}

#[napi(object)]
pub struct NapiCandidateFactRow {
    pub id: Either<i64, String>,
    pub content: String,
    pub date_created: String,
    pub summaries: Option<Vec<NapiCandidateSummaryRow>>,
}

#[napi(object)]
pub struct NapiWriteAck {
    pub written_ops: u32,
}
