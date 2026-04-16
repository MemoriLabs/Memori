use crate::OrchestratorError;
use crate::augmentation::models::{
    AugmentationAttribution, AugmentationAttributionEntity, AugmentationAttributionProcess,
    AugmentationConversation, AugmentationFramework, AugmentationInput, AugmentationLlm,
    AugmentationLlmModel, AugmentationLlmSdk, AugmentationMeta, AugmentationPayload,
    AugmentationPlatform, AugmentationSdk, AugmentationStorage, ConversationMessage,
};
use crate::network::{ApiError, ApiSubdomain, MemoriClient};
use crate::storage::{WriteBatch, WriteOp};
use sha2::{Digest, Sha256};

pub async fn run_advanced_augmentation(
    input: &AugmentationInput,
) -> Result<WriteBatch, OrchestratorError> {
    let payload = build_payload(input);
    if log::log_enabled!(log::Level::Trace) {
        if let Ok(payload_json) = serde_json::to_string(&payload) {
            log::trace!("augmentation request payload: {payload_json}");
        }
    }

    let response = if input.use_mock_response {
        log::debug!("augmentation using mock response");
        input
            .mock_response
            .clone()
            .unwrap_or_else(default_mock_augmentation_response)
    } else {
        log::debug!("augmentation calling Memori API: sdk/augmentation");
        let client = MemoriClient::new(ApiSubdomain::Default)?;
        let raw_response = client.augmentation_raw_async(&payload).await?;
        log::trace!("augmentation raw response body: {raw_response}");
        serde_json::from_str::<serde_json::Value>(&raw_response).map_err(|e| {
            OrchestratorError::ApiError(ApiError::Network(format!(
                "failed to parse augmentation response body as JSON: {e}"
            )))
        })?
    };
    Ok(build_write_batch_from_response(input, response))
}

pub fn build_payload(input: &AugmentationInput) -> AugmentationPayload {
    let mut messages = input.conversation_messages.clone();
    if messages.is_empty() {
        if let Some(content) = input.content.clone() {
            messages.push(ConversationMessage {
                role: "assistant".to_string(),
                content,
            });
        }
    }

    AugmentationPayload {
        conversation: AugmentationConversation {
            messages,
            summary: None,
        },
        meta: AugmentationMeta {
            attribution: AugmentationAttribution {
                entity: AugmentationAttributionEntity {
                    id: hash_id(&input.entity_id),
                },
                process: AugmentationAttributionProcess {
                    id: hash_id(input.process_id.as_deref().unwrap_or("")),
                },
            },
            framework: AugmentationFramework {
                provider: input.framework.clone(),
            },
            llm: AugmentationLlm {
                model: AugmentationLlmModel {
                    provider: input.llm_provider.clone(),
                    sdk: AugmentationLlmSdk {
                        version: input.llm_provider_sdk_version.clone(),
                    },
                    version: input.llm_model.clone(),
                },
            },
            platform: AugmentationPlatform {
                provider: input.platform_provider.clone(),
            },
            sdk: AugmentationSdk {
                lang: "python".to_string(),
                version: input.sdk_version.clone(),
            },
            storage: AugmentationStorage {
                cockroachdb: input.storage_cockroachdb.unwrap_or(false),
                dialect: input.storage_dialect.clone(),
            },
        },
    }
}

fn hash_id(value: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(value.as_bytes());
    format!("{:x}", hasher.finalize())
}

pub fn build_write_batch_from_response(
    input: &AugmentationInput,
    response: serde_json::Value,
) -> WriteBatch {
    let mut ops = Vec::new();

    if let Some(facts) = extract_facts(&response) {
        ops.push(WriteOp {
            op_type: "entity_fact.create".to_string(),
            payload: serde_json::json!({
                "entity_id": input.entity_id,
                "conversation_id": input.conversation_id,
                "facts": facts,
            }),
        });
    }

    if let Some(triples) = response
        .pointer("/entity/semantic_triples")
        .or_else(|| response.pointer("/entity/triples"))
        .cloned()
    {
        ops.push(WriteOp {
            op_type: "knowledge_graph.create".to_string(),
            payload: serde_json::json!({
                "entity_id": input.entity_id,
                "semantic_triples": triples,
            }),
        });
    }

    if let Some(attrs) = response.pointer("/process/attributes").cloned() {
        ops.push(WriteOp {
            op_type: "process_attribute.create".to_string(),
            payload: serde_json::json!({
                "process_id": input.process_id,
                "attributes": attrs,
            }),
        });
    }

    if let Some(summary) = response.pointer("/conversation/summary").cloned() {
        ops.push(WriteOp {
            op_type: "conversation.update".to_string(),
            payload: serde_json::json!({
                "conversation_id": input.conversation_id,
                "summary": summary,
            }),
        });
    }

    if ops.is_empty() {
        if let Some(content) = input.content.clone() {
            ops.push(WriteOp {
                op_type: "upsert_fact".to_string(),
                payload: serde_json::json!({
                    "entity_id": input.entity_id,
                    "fact_id": input.fact_id,
                    "content": content,
                    "metadata": input.metadata,
                }),
            });
        }
    }

    WriteBatch { ops }
}

fn extract_facts(response: &serde_json::Value) -> Option<Vec<String>> {
    if let Some(arr) = response
        .pointer("/entity/facts")
        .and_then(|value| value.as_array())
    {
        let mut facts = Vec::new();
        for item in arr {
            if let Some(text) = item.as_str() {
                facts.push(text.to_string());
                continue;
            }
            if let Some(content) = item.get("content").and_then(|v| v.as_str()) {
                facts.push(content.to_string());
            }
        }
        if !facts.is_empty() {
            return Some(facts);
        }
    }

    let triples = response
        .pointer("/entity/triples")
        .or_else(|| response.pointer("/entity/semantic_triples"))
        .and_then(|value| value.as_array())?;
    let mut facts = Vec::new();
    for triple in triples {
        if let Some(content) = triple.get("content").and_then(|v| v.as_str()) {
            facts.push(content.to_string());
            continue;
        }

        let subject = read_triple_value(triple.get("subject"));
        let predicate = triple.get("predicate").and_then(|v| v.as_str());
        let object = read_triple_value(triple.get("object"));
        if let (Some(s), Some(p), Some(o)) = (subject, predicate, object) {
            facts.push(format!("{s} {p} {o}"));
        }
    }
    if facts.is_empty() { None } else { Some(facts) }
}

fn read_triple_value(value: Option<&serde_json::Value>) -> Option<String> {
    match value {
        Some(serde_json::Value::String(s)) => Some(s.to_string()),
        Some(serde_json::Value::Object(map)) => map
            .get("name")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
        _ => None,
    }
}

fn default_mock_augmentation_response() -> serde_json::Value {
    serde_json::json!({
        "conversation": {
            "summary": "The conversation states that the user's favorite color is blue."
        },
        "entity": {
            "triples": [
                {
                    "content": "The user's favorite color is blue.",
                    "subject": { "name": "self", "type": "person" },
                    "predicate": "has favorite color",
                    "object": { "name": "blue", "type": "concept" }
                }
            ]
        },
        "process": {
            "attributes": [
                "Image input interpretation",
                "Markdown/plaintext formatting control",
                "Concise information-dense writing",
                "Knowledge cutoff adherence",
                "Relevant context filtering",
                "Prompt capability extraction",
                "Structured JSON schema output"
            ]
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_id_is_deterministic_and_sha256_length() {
        let a = hash_id("entity-1");
        let b = hash_id("entity-1");
        assert_eq!(a, b);
        assert_eq!(a.len(), 64);
    }

    #[test]
    fn hash_id_changes_with_input() {
        assert_ne!(hash_id("entity-1"), hash_id("entity-2"));
    }

    #[test]
    fn build_payload_hashes_ids_and_fills_defaults_when_missing() {
        let input = AugmentationInput {
            entity_id: "entity-1".to_string(),
            process_id: None,
            conversation_id: None,
            conversation_messages: Vec::new(),
            system_prompt: None,
            llm_provider: None,
            llm_model: None,
            llm_provider_sdk_version: None,
            framework: None,
            platform_provider: None,
            storage_dialect: None,
            storage_cockroachdb: None,
            sdk_version: None,
            use_mock_response: true,
            mock_response: None,
            session_id: None,
            fact_id: None,
            content: Some("fact content".to_string()),
            metadata: serde_json::json!({}),
        };
        let payload = build_payload(&input);
        assert_eq!(payload.meta.attribution.entity.id, hash_id("entity-1"));
        assert_eq!(payload.meta.attribution.process.id, hash_id(""));
        assert!(!payload.meta.storage.cockroachdb);
        assert_eq!(payload.conversation.messages.len(), 1);
        assert_eq!(payload.conversation.messages[0].role, "assistant");
        assert_eq!(payload.conversation.messages[0].content, "fact content");
    }

    #[test]
    fn build_write_batch_falls_back_to_upsert_fact_when_response_empty() {
        let input = AugmentationInput {
            entity_id: "entity-1".to_string(),
            process_id: None,
            conversation_id: None,
            conversation_messages: Vec::new(),
            system_prompt: None,
            llm_provider: None,
            llm_model: None,
            llm_provider_sdk_version: None,
            framework: None,
            platform_provider: None,
            storage_dialect: None,
            storage_cockroachdb: None,
            sdk_version: None,
            use_mock_response: true,
            mock_response: None,
            session_id: None,
            fact_id: Some("fact-1".to_string()),
            content: Some("manual fact".to_string()),
            metadata: serde_json::json!({ "source": "manual" }),
        };
        let batch = build_write_batch_from_response(&input, serde_json::json!({}));
        assert_eq!(batch.ops.len(), 1);
        assert_eq!(batch.ops[0].op_type, "upsert_fact");
        assert_eq!(
            batch.ops[0].payload.get("content").and_then(|v| v.as_str()),
            Some("manual fact")
        );
    }
}
