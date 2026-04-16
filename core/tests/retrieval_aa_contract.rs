use std::sync::Mutex;

use base64::Engine;
use engine_orchestrator::augmentation::{
    AugmentationInput, ConversationMessage, build_payload, build_write_batch_from_response,
};
use engine_orchestrator::embeddings::format_embedding_for_db;
use engine_orchestrator::retrieval::{RetrievalRequest, format_recall_output, run_retrieval};
use engine_orchestrator::search::FactId;
use engine_orchestrator::storage::{
    CandidateFactRow, EmbeddingRow, HostStorageError, StorageBridge, WriteAck, WriteBatch,
};

struct MockStorageBridge {
    writes: Mutex<Vec<WriteBatch>>,
    requested_fact_count: Mutex<usize>,
}

struct InvalidEmbeddingBridge;
struct DimensionMismatchBridge;

impl StorageBridge for InvalidEmbeddingBridge {
    fn fetch_embeddings(
        &self,
        _entity_id: &str,
        _limit: usize,
    ) -> Result<Vec<EmbeddingRow>, HostStorageError> {
        Ok(vec![EmbeddingRow {
            id: FactId::Int(1),
            content_embedding: Vec::new(),
            content_embedding_b64: Some("AQ==".to_string()),
        }])
    }

    fn fetch_facts_by_ids(
        &self,
        _ids: &[FactId],
    ) -> Result<Vec<CandidateFactRow>, HostStorageError> {
        Ok(Vec::new())
    }

    fn write_batch(&self, batch: &WriteBatch) -> Result<WriteAck, HostStorageError> {
        Ok(WriteAck {
            written_ops: batch.ops.len(),
        })
    }
}

impl StorageBridge for DimensionMismatchBridge {
    fn fetch_embeddings(
        &self,
        _entity_id: &str,
        _limit: usize,
    ) -> Result<Vec<EmbeddingRow>, HostStorageError> {
        Ok(vec![EmbeddingRow {
            id: FactId::Int(1),
            content_embedding: vec![1.0, 0.0, 0.2],
            content_embedding_b64: None,
        }])
    }

    fn fetch_facts_by_ids(
        &self,
        _ids: &[FactId],
    ) -> Result<Vec<CandidateFactRow>, HostStorageError> {
        Ok(Vec::new())
    }

    fn write_batch(&self, batch: &WriteBatch) -> Result<WriteAck, HostStorageError> {
        Ok(WriteAck {
            written_ops: batch.ops.len(),
        })
    }
}

impl MockStorageBridge {
    fn new() -> Self {
        Self {
            writes: Mutex::new(Vec::new()),
            requested_fact_count: Mutex::new(0),
        }
    }
}

impl StorageBridge for MockStorageBridge {
    fn fetch_embeddings(
        &self,
        _entity_id: &str,
        _limit: usize,
    ) -> Result<Vec<EmbeddingRow>, HostStorageError> {
        let mut rows = Vec::new();
        for i in 1..=8 {
            let emb = if i == 1 {
                vec![1.0, 0.0]
            } else {
                vec![0.1, 0.9]
            };
            let encoded =
                base64::engine::general_purpose::STANDARD.encode(format_embedding_for_db(&emb));
            rows.push(EmbeddingRow {
                id: FactId::Int(i),
                content_embedding: Vec::new(),
                content_embedding_b64: Some(encoded),
            });
        }
        Ok(rows)
    }

    fn fetch_facts_by_ids(
        &self,
        ids: &[FactId],
    ) -> Result<Vec<CandidateFactRow>, HostStorageError> {
        *self
            .requested_fact_count
            .lock()
            .expect("requested_fact_count mutex poisoned") = ids.len();

        let mut rows = Vec::new();
        for id in ids {
            rows.push(CandidateFactRow {
                id: id.clone(),
                content: match id {
                    FactId::Int(1) => "rust memory safety language".to_string(),
                    _ => "javascript frontend language".to_string(),
                },
                date_created: "2026-01-01".to_string(),
                summaries: Vec::new(),
            });
        }
        Ok(rows)
    }

    fn write_batch(&self, batch: &WriteBatch) -> Result<WriteAck, HostStorageError> {
        let mut guard = self.writes.lock().expect("writes mutex poisoned");
        guard.push(batch.clone());
        Ok(WriteAck {
            written_ops: batch.ops.len(),
        })
    }
}

#[test]
fn retrieval_pipeline_runs_dense_then_rerank() {
    let bridge = MockStorageBridge::new();
    let request = RetrievalRequest {
        entity_id: "user-1".to_string(),
        query_text: "rust language".to_string(),
        dense_limit: 100,
        limit: 2,
    };

    let ranked = run_retrieval(&bridge, &request, &[1.0, 0.0]).expect("retrieval should succeed");
    assert_eq!(ranked.len(), 2);
    assert_eq!(ranked[0].id, FactId::Int(1));
    assert_eq!(
        *bridge
            .requested_fact_count
            .lock()
            .expect("requested_fact_count mutex poisoned"),
        8
    );
}

#[test]
fn retrieval_rejects_invalid_base64_embedding_bytes() {
    let bridge = InvalidEmbeddingBridge;
    let request = RetrievalRequest {
        entity_id: "user-1".to_string(),
        query_text: "rust language".to_string(),
        dense_limit: 10,
        limit: 2,
    };

    let result = run_retrieval(&bridge, &request, &[1.0, 0.0]);
    assert!(result.is_err());
    let message = result.expect_err("expected retrieval error").to_string();
    assert!(message.contains("invalid_embedding_bytes"));
}

#[test]
fn recall_output_includes_id_and_scores() {
    let ranked = vec![engine_orchestrator::storage::RankedFact {
        id: FactId::Int(10),
        content: "A ranked fact".to_string(),
        similarity: 0.9,
        rank_score: 1.2,
        date_created: "2026-01-01".to_string(),
        summaries: Vec::new(),
    }];
    let output = format_recall_output(&ranked);
    assert!(output.contains("id=10"));
    assert!(output.contains("similarity=0.9000"));
    assert!(output.contains("rank_score=1.2000"));
    assert!(output.contains("A ranked fact"));
}

#[test]
fn retrieval_rejects_embedding_dimension_mismatch() {
    let bridge = DimensionMismatchBridge;
    let request = RetrievalRequest {
        entity_id: "user-1".to_string(),
        query_text: "rust language".to_string(),
        dense_limit: 10,
        limit: 2,
    };

    let result = run_retrieval(&bridge, &request, &[1.0, 0.0]);
    assert!(result.is_err());
    let message = result.expect_err("expected retrieval error").to_string();
    assert!(message.contains("invalid_embedding_dimension"));
}

#[test]
fn augmentation_pipeline_builds_write_batch() {
    let bridge = MockStorageBridge::new();
    let input = AugmentationInput {
        entity_id: "user-1".to_string(),
        process_id: Some("proc-1".to_string()),
        conversation_id: Some("conv-1".to_string()),
        conversation_messages: vec![
            ConversationMessage {
                role: "user".to_string(),
                content: "remember I prefer concise responses".to_string(),
            },
            ConversationMessage {
                role: "assistant".to_string(),
                content: "Noted".to_string(),
            },
        ],
        system_prompt: None,
        llm_provider: Some("openai".to_string()),
        llm_model: Some("gpt-4.1-mini".to_string()),
        llm_provider_sdk_version: Some("1.2.3".to_string()),
        framework: Some("pytest".to_string()),
        platform_provider: Some("local".to_string()),
        storage_dialect: Some("sqlite".to_string()),
        storage_cockroachdb: Some(false),
        sdk_version: Some("0.1.0".to_string()),
        use_mock_response: true,
        mock_response: Some(serde_json::json!({
            "entity": {
                "facts": ["prefers concise responses"],
                "semantic_triples": [{"subject": "user", "predicate": "prefers", "object": "concise"}]
            },
            "process": { "attributes": { "phase": "test" } },
            "conversation": { "summary": "user preference recorded" }
        })),
        session_id: None,
        fact_id: Some("fact-1".to_string()),
        content: Some("new memory fact".to_string()),
        metadata: serde_json::json!({ "source": "unit-test" }),
    };

    let payload = build_payload(&input);
    assert_eq!(payload.meta.attribution.entity.id.len(), 64);
    assert_eq!(payload.meta.attribution.process.id.len(), 64);
    assert_eq!(payload.meta.llm.model.sdk.version.as_deref(), Some("1.2.3"));
    assert_eq!(payload.meta.storage.dialect.as_deref(), Some("sqlite"));

    let batch = build_write_batch_from_response(&input, input.mock_response.clone().expect("mock"));
    assert_eq!(batch.ops.len(), 4);
    assert_eq!(batch.ops[0].op_type, "entity_fact.create");
    assert_eq!(batch.ops[1].op_type, "knowledge_graph.create");
    assert_eq!(batch.ops[2].op_type, "process_attribute.create");
    assert_eq!(batch.ops[3].op_type, "conversation.update");

    let ack = bridge.write_batch(&batch).expect("write should succeed");
    assert_eq!(ack.written_ops, 4);
}

#[test]
fn augmentation_pipeline_derives_facts_from_triples_when_missing() {
    let input = AugmentationInput {
        entity_id: "user-1".to_string(),
        process_id: Some("proc-1".to_string()),
        conversation_id: Some("conv-1".to_string()),
        conversation_messages: vec![],
        system_prompt: None,
        llm_provider: Some("openai".to_string()),
        llm_model: Some("gpt-4.1-mini".to_string()),
        llm_provider_sdk_version: Some("1.2.3".to_string()),
        framework: Some("pytest".to_string()),
        platform_provider: Some("local".to_string()),
        storage_dialect: Some("sqlite".to_string()),
        storage_cockroachdb: Some(false),
        sdk_version: Some("0.1.0".to_string()),
        use_mock_response: true,
        mock_response: Some(serde_json::json!({
            "entity": {
                "triples": [
                    {
                        "content": "The user prefers Spring.",
                        "subject": {"name": "self", "type": "person"},
                        "predicate": "prefers",
                        "object": {"name": "Spring", "type": "concept"}
                    }
                ]
            },
            "conversation": { "summary": "The user likes Spring." }
        })),
        session_id: None,
        fact_id: Some("fact-1".to_string()),
        content: Some("new memory fact".to_string()),
        metadata: serde_json::json!({ "source": "unit-test" }),
    };

    let batch = build_write_batch_from_response(&input, input.mock_response.clone().expect("mock"));
    let fact_op = batch
        .ops
        .iter()
        .find(|op| op.op_type == "entity_fact.create")
        .expect("entity_fact.create should exist");
    let facts = fact_op
        .payload
        .get("facts")
        .and_then(|v| v.as_array())
        .expect("facts should be an array");
    assert_eq!(facts.len(), 1);
    assert_eq!(facts[0].as_str(), Some("The user prefers Spring."));
}

#[test]
fn augmentation_pipeline_accepts_current_aa_response_contract() {
    let input = AugmentationInput {
        entity_id: "user-1".to_string(),
        process_id: Some("proc-1".to_string()),
        conversation_id: Some("conv-1".to_string()),
        conversation_messages: vec![],
        system_prompt: None,
        llm_provider: Some("openai".to_string()),
        llm_model: Some("gpt-4o-mini".to_string()),
        llm_provider_sdk_version: Some("2.8.1".to_string()),
        framework: Some("langchain".to_string()),
        platform_provider: Some("local".to_string()),
        storage_dialect: Some("sqlite".to_string()),
        storage_cockroachdb: Some(false),
        sdk_version: Some("3.2.8".to_string()),
        use_mock_response: true,
        mock_response: Some(serde_json::json!({
            "conversation": {
                "summary": "The conversation states that the user's favorite color is blue."
            },
            "entity": {
                "triples": [
                    {
                        "content": "The user's favorite color is blue.",
                        "object": { "name": "blue", "type": "concept" },
                        "predicate": "has favorite color",
                        "subject": { "name": "self", "type": "person" }
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
        })),
        session_id: None,
        fact_id: None,
        content: None,
        metadata: serde_json::json!({}),
    };

    let batch = build_write_batch_from_response(&input, input.mock_response.clone().expect("mock"));
    assert_eq!(batch.ops.len(), 4);
    assert!(
        batch
            .ops
            .iter()
            .any(|op| op.op_type == "entity_fact.create")
    );
    assert!(
        batch
            .ops
            .iter()
            .any(|op| op.op_type == "knowledge_graph.create")
    );
    assert!(
        batch
            .ops
            .iter()
            .any(|op| op.op_type == "process_attribute.create")
    );
    assert!(
        batch
            .ops
            .iter()
            .any(|op| op.op_type == "conversation.update")
    );

    let fact_op = batch
        .ops
        .iter()
        .find(|op| op.op_type == "entity_fact.create")
        .expect("entity_fact.create should exist");
    let facts = fact_op
        .payload
        .get("facts")
        .and_then(|v| v.as_array())
        .expect("facts should be an array");
    assert_eq!(
        facts[0].as_str(),
        Some("The user's favorite color is blue.")
    );
}
