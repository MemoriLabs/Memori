use crate::bridge::*;
use crate::types::*;
use dashmap::DashMap;
use engine_orchestrator::EngineOrchestrator;
use engine_orchestrator::search::FactId;
use engine_orchestrator::storage::{CandidateFactRow, EmbeddingRow, RankedFact, WriteAck};
use napi::bindgen_prelude::*;
use napi::threadsafe_function::ThreadsafeFunction;
use napi_derive::napi;
use std::panic::catch_unwind;
use std::sync::atomic::AtomicU32;
use std::sync::{Arc, Mutex};

#[napi]
pub struct MemoriEngine {
    pub(crate) inner: Arc<EngineOrchestrator>,
    pub(crate) pending_embeddings: PendingEmbeddingsMap,
    pub(crate) pending_facts: PendingFactsMap,
    pub(crate) pending_writes: PendingWritesMap,
}

#[napi]
impl MemoriEngine {
    #[napi(constructor)]
    pub fn new(
        model_name: Option<String>,
        fetch_embeddings_cb: ThreadsafeFunction<(u32, String)>,
        fetch_facts_by_ids_cb: ThreadsafeFunction<(u32, String)>,
        write_batch_cb: ThreadsafeFunction<(u32, String)>,
    ) -> Result<Self> {
        let pending_embeddings = Arc::new(DashMap::new());
        let pending_facts = Arc::new(DashMap::new());
        let pending_writes = Arc::new(DashMap::new());

        let bridge = Arc::new(NodeStorageBridge {
            fetch_embeddings_tsfn: Mutex::new(Some(fetch_embeddings_cb)),
            fetch_facts_by_ids_tsfn: Mutex::new(Some(fetch_facts_by_ids_cb)),
            write_batch_tsfn: Mutex::new(Some(write_batch_cb)),
            pending_embeddings: pending_embeddings.clone(),
            pending_facts: pending_facts.clone(),
            pending_writes: pending_writes.clone(),
            next_id: AtomicU32::new(1),
        });

        let inner = EngineOrchestrator::new_with_storage(model_name.as_deref(), Some(bridge))
            .map_err(|e| Error::from_reason(e.to_string()))?;

        Ok(Self {
            inner: Arc::new(inner),
            pending_embeddings,
            pending_facts,
            pending_writes,
        })
    }

    #[napi]
    pub fn resolve_embeddings_callback(&self, id: u32, result: Vec<NapiEmbeddingRow>) {
        let rows: Vec<EmbeddingRow> = result
            .into_iter()
            .map(|r| EmbeddingRow {
                id: match r.id {
                    Either::A(num) => FactId::Int(num),
                    Either::B(s) => FactId::String(s),
                },
                content_embedding: r.content_embedding.to_vec(),
                content_embedding_b64: None,
            })
            .collect();

        // DashMap returns an Option<(K, V)> tuple on removal
        if let Some((_, tx)) = self.pending_embeddings.remove(&id) {
            let _ = tx.send(rows);
        }
    }

    #[napi]
    pub fn resolve_facts_callback(&self, id: u32, result: Vec<NapiCandidateFactRow>) {
        let rows: Vec<CandidateFactRow> = result
            .into_iter()
            .map(|r| {
                let id_val = match r.id {
                    Either::A(num) => serde_json::json!(num),
                    Either::B(s) => serde_json::json!(s),
                };
                let mut obj = serde_json::Map::new();
                obj.insert("id".to_string(), id_val);
                obj.insert("content".to_string(), serde_json::json!(r.content));
                obj.insert(
                    "date_created".to_string(),
                    serde_json::json!(r.date_created),
                );
                if let Some(sums) = r.summaries {
                    obj.insert("summaries".to_string(), serde_json::to_value(sums).unwrap());
                }
                serde_json::from_value(serde_json::Value::Object(obj)).unwrap()
            })
            .collect();

        if let Some((_, tx)) = self.pending_facts.remove(&id) {
            let _ = tx.send(rows);
        }
    }

    #[napi]
    pub fn resolve_write_callback(&self, id: u32, result: NapiWriteAck) {
        if let Some((_, tx)) = self.pending_writes.remove(&id) {
            let ack = WriteAck {
                written_ops: result.written_ops as usize,
            };
            let _ = tx.send(ack);
        }
    }

    #[napi]
    pub fn embed_texts(&self, texts: Vec<String>) -> Result<Vec<Float32Array>> {
        let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
            let (flat_vectors, shape) = self.inner.embed(texts);
            let mut out = Vec::with_capacity(shape[0]);
            let dim = shape[1];
            for chunk in flat_vectors.chunks(dim) {
                out.push(Float32Array::new(chunk.to_vec()));
            }
            Ok(out)
        }));

        match result {
            Ok(Ok(arr)) => Ok(arr),
            Ok(Err(e)) => Err(e),
            Err(_) => Err(Error::from_reason(
                "Rust panicked during embed_texts!".to_string(),
            )),
        }
    }

    #[napi]
    pub async fn retrieve(&self, request: NapiRetrievalRequest) -> Result<Vec<NapiRecallObject>> {
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let req = serde_json::from_value(serde_json::to_value(&request).unwrap())
                .map_err(|e| Error::from_reason(format!("Invalid retrieval request: {}", e)))?;
            let results: Vec<RankedFact> = inner
                .retrieve(req)
                .map_err(|e| Error::from_reason(e.to_string()))?;
            let napi_results = results
                .into_iter()
                .map(|r| {
                    let id = match r.id {
                        FactId::Int(n) => Either::A(n),
                        FactId::String(s) => Either::B(s),
                    };
                    let summaries = if r.summaries.is_empty() {
                        None
                    } else {
                        Some(
                            r.summaries
                                .into_iter()
                                .map(|s| NapiRecallSummary {
                                    content: s["content"].as_str().unwrap_or("").to_string(),
                                    date_created: s["date_created"]
                                        .as_str()
                                        .unwrap_or("")
                                        .to_string(),
                                    entity_fact_id: fact_id_from_json(&s["entity_fact_id"]),
                                    fact_id: fact_id_from_json(&s["fact_id"]),
                                })
                                .collect(),
                        )
                    };
                    NapiRecallObject {
                        id,
                        content: r.content,
                        rank_score: Some(r.rank_score as f64),
                        similarity: Some(r.similarity as f64),
                        date_created: Some(r.date_created),
                        summaries,
                    }
                })
                .collect();
            Ok(napi_results)
        })
        .await
        .map_err(|e| Error::from_reason(e.to_string()))?
    }

    #[napi]
    pub async fn recall(&self, request: NapiRetrievalRequest) -> Result<String> {
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let req = serde_json::from_value(serde_json::to_value(&request).unwrap())
                .map_err(|e| Error::from_reason(format!("Invalid recall request: {}", e)))?;
            inner
                .recall(req)
                .map_err(|e| Error::from_reason(e.to_string()))
        })
        .await
        .map_err(|e| Error::from_reason(e.to_string()))?
    }

    #[napi]
    pub fn submit_augmentation(&self, input: NapiAugmentationInput) -> Result<String> {
        let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
            let core_input = serde_json::from_value(serde_json::to_value(&input).unwrap())
                .map_err(|e| Error::from_reason(format!("Invalid augmentation input: {}", e)))?;
            let accepted = self
                .inner
                .submit_augmentation(core_input)
                .map_err(|e| Error::from_reason(e.to_string()))?;
            Ok(accepted.job_id.to_string())
        }));

        match result {
            Ok(Ok(id)) => Ok(id),
            Ok(Err(e)) => Err(e),
            Err(_) => Err(Error::from_reason(
                "Rust panicked during augmentation submit!".to_string(),
            )),
        }
    }

    #[napi]
    pub async fn wait_for_augmentation(&self, timeout_ms: Option<u32>) -> Result<bool> {
        let timeout = timeout_ms.map(|ms| std::time::Duration::from_millis(ms as u64));
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || inner.wait_for_augmentation(timeout))
            .await
            .map_err(|e| Error::from_reason(e.to_string()))?
            .map_err(|e| Error::from_reason(e.to_string()))
    }

    #[napi]
    pub fn shutdown(&self) {
        self.inner.shutdown();
    }
}

fn fact_id_from_json(v: &serde_json::Value) -> Option<Either<i64, String>> {
    match v {
        serde_json::Value::Number(n) => n.as_i64().map(Either::A),
        serde_json::Value::String(s) => Some(Either::B(s.clone())),
        _ => None,
    }
}
