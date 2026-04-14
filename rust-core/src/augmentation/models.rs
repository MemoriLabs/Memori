use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AugmentationInput {
    pub entity_id: String,
    #[serde(default)]
    pub process_id: Option<String>,
    #[serde(default)]
    pub conversation_id: Option<String>,
    #[serde(default)]
    pub conversation_messages: Vec<ConversationMessage>,
    #[serde(default)]
    pub system_prompt: Option<String>,
    #[serde(default)]
    pub llm_provider: Option<String>,
    #[serde(default)]
    pub llm_model: Option<String>,
    #[serde(default)]
    pub llm_provider_sdk_version: Option<String>,
    #[serde(default)]
    pub framework: Option<String>,
    #[serde(default)]
    pub platform_provider: Option<String>,
    #[serde(default)]
    pub storage_dialect: Option<String>,
    #[serde(default)]
    pub storage_cockroachdb: Option<bool>,
    #[serde(default)]
    pub sdk_version: Option<String>,
    #[serde(default)]
    pub use_mock_response: bool,
    #[serde(default)]
    pub mock_response: Option<serde_json::Value>,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub fact_id: Option<String>,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AugmentationPayload {
    pub conversation: AugmentationConversation,
    pub meta: AugmentationMeta,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AugmentationConversation {
    pub messages: Vec<ConversationMessage>,
    pub summary: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AugmentationMeta {
    pub attribution: AugmentationAttribution,
    pub framework: AugmentationFramework,
    pub llm: AugmentationLlm,
    pub platform: AugmentationPlatform,
    pub sdk: AugmentationSdk,
    pub storage: AugmentationStorage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AugmentationAttribution {
    pub entity: AugmentationAttributionEntity,
    pub process: AugmentationAttributionProcess,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AugmentationAttributionEntity {
    pub id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AugmentationAttributionProcess {
    pub id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AugmentationLlm {
    pub model: AugmentationLlmModel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AugmentationLlmModel {
    pub provider: Option<String>,
    pub sdk: AugmentationLlmSdk,
    pub version: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AugmentationLlmSdk {
    pub version: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AugmentationSdk {
    pub lang: String,
    pub version: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AugmentationFramework {
    pub provider: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AugmentationPlatform {
    pub provider: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AugmentationStorage {
    pub cockroachdb: bool,
    pub dialect: Option<String>,
}
