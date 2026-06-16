use crate::bridge::NodeConnectionFactory;
use crate::types::*;
use engine_orchestrator::EngineOrchestrator;
use engine_orchestrator::search::FactId;
use engine_orchestrator::storage::models::RankedFact;
use engine_orchestrator::storage::{Dialect, RustStorageManager, StorageBridge, WriteBatch};
use napi::bindgen_prelude::*;
use napi::threadsafe_function::ThreadsafeFunction;
use napi_derive::napi;
use std::panic::catch_unwind;
use std::sync::Arc;

#[napi]
pub struct MemoriEngine {
    pub(crate) inner: Arc<EngineOrchestrator>,
    pub(crate) factory: Arc<NodeConnectionFactory>,
    pub(crate) storage_manager: Arc<RustStorageManager>,
}

#[napi]
impl MemoriEngine {
    /// `dialect` is the SQL dialect of the user's connection (`"sqlite"`, `"postgresql"`, `"cockroachdb"`, `"mysql"`).
    #[napi(constructor)]
    pub fn new(
        env: Env,
        model_name: Option<String>,
        storage_call_cb: ThreadsafeFunction<(u32, String)>,
        dialect: String,
    ) -> Result<Self> {
        // Unref so the TSFN doesn't keep the Node event loop alive when idle.
        unsafe {
            napi::sys::napi_unref_threadsafe_function(env.raw(), storage_call_cb.raw());
        }

        let dialect_enum = dialect.parse::<Dialect>().map_err(Error::from_reason)?;

        let factory = Arc::new(NodeConnectionFactory::new(storage_call_cb, dialect));
        let storage_manager = Arc::new(RustStorageManager::new(factory.clone(), dialect_enum));

        let inner = EngineOrchestrator::new_with_storage(
            model_name.as_deref(),
            Some(storage_manager.clone()),
        )
        .map_err(|e| Error::from_reason(e.to_string()))?;

        let inner = Arc::new(inner);
        // Weak reference breaks the cycle: EngineOrchestrator → storage_manager → embedder → EngineOrchestrator.
        let embed_handle = Arc::downgrade(&inner);
        storage_manager.set_embedder(Box::new(move |texts: Vec<String>| {
            let Some(engine) = embed_handle.upgrade() else {
                return vec![];
            };
            let (flat, shape) = engine.embed(texts);
            if shape[0] == 0 || shape[1] == 0 {
                return vec![];
            }
            let dim = shape[1];
            flat.chunks(dim).map(|c| c.to_vec()).collect()
        }));

        Ok(Self {
            inner,
            factory,
            storage_manager,
        })
    }

    /// Called by TS to unblock a pending Rust storage call.
    ///
    /// `result_json` is one of:
    ///   `{ "conn_id": N }` (for acquire), `{ "rows": [...] }` (for execute),
    ///   `{ "ok": true }` (for begin/commit/rollback/close),
    ///   or `{ "error": { "code": "...", "message": "..." } }`.
    #[napi]
    pub fn resolve_storage_call(&self, id: u32, result_json: String) {
        self.factory.resolve(id, result_json);
    }

    /// Runs database migrations. Must be called once after construction.
    #[napi]
    pub async fn build(&self) -> Result<()> {
        let storage = self.storage_manager.clone();
        tokio::task::spawn_blocking(move || {
            storage
                .build()
                .map_err(|e| Error::from_reason(e.to_string()))
        })
        .await
        .map_err(|e| Error::from_reason(e.to_string()))?
    }

    /// For immediate writes before the augmentation pipeline completes (e.g. conversation messages).
    #[napi]
    pub async fn write_batch(&self, json: String) -> Result<NapiWriteAck> {
        let storage = self.storage_manager.clone();
        tokio::task::spawn_blocking(move || {
            let batch: WriteBatch = serde_json::from_str(&json)
                .map_err(|e| Error::from_reason(format!("invalid batch JSON: {e}")))?;
            storage
                .write_batch(&batch)
                .map_err(|e| Error::from_reason(e.to_string()))
                .map(|ack| NapiWriteAck {
                    written_ops: ack.written_ops as u32,
                })
        })
        .await
        .map_err(|e| Error::from_reason(e.to_string()))?
    }

    /// Returns conversation messages for the given session ID as a JSON array of
    /// `{ role, content }` objects. Returns `"[]"` when no storage is configured.
    #[napi]
    pub async fn get_conversation_history(&self, session_id: String) -> Result<String> {
        let storage = self.storage_manager.clone();
        tokio::task::spawn_blocking(move || {
            storage
                .get_conversation_history(&session_id)
                .map_err(|e| Error::from_reason(e.to_string()))
                .and_then(|messages| {
                    serde_json::to_string(&messages).map_err(|e| Error::from_reason(e.to_string()))
                })
        })
        .await
        .map_err(|e| Error::from_reason(e.to_string()))?
    }

    #[napi]
    pub async fn embed_texts(&self, texts: Vec<String>) -> Result<Vec<Float32Array>> {
        let inner = self.inner.clone();
        // Run ONNX inference on a blocking thread so the Node event loop stays free.
        let chunks: Vec<Vec<f32>> = tokio::task::spawn_blocking(move || {
            let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
                let (flat, shape) = inner.embed(texts);
                if shape[0] == 0 || shape[1] == 0 {
                    return Ok(vec![]);
                }
                let dim = shape[1];
                Ok::<Vec<Vec<f32>>, Error>(flat.chunks(dim).map(|c| c.to_vec()).collect())
            }));
            match result {
                Ok(Ok(v)) => Ok(v),
                Ok(Err(e)) => Err(e),
                Err(_) => Err(Error::from_reason("Rust panicked during embed_texts!")),
            }
        })
        .await
        .map_err(|e| Error::from_reason(e.to_string()))??;
        Ok(chunks.into_iter().map(Float32Array::new).collect())
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
                "Rust panicked during augmentation submit!",
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
