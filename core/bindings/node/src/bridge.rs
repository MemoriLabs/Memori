use dashmap::DashMap;
use engine_orchestrator::search::FactId;
use engine_orchestrator::storage::{
    CandidateFactRow, EmbeddingRow, HostStorageError, StorageBridge, WriteAck, WriteBatch,
};
use napi::threadsafe_function::{ErrorStrategy, ThreadsafeFunction, ThreadsafeFunctionCallMode};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;
use tokio::sync::oneshot;
use tokio::time::timeout;

const JS_CALLBACK_TIMEOUT: Duration = Duration::from_secs(30);

pub type PendingEmbeddingsMap = Arc<DashMap<u32, oneshot::Sender<Vec<EmbeddingRow>>>>;
pub type PendingFactsMap = Arc<DashMap<u32, oneshot::Sender<Vec<CandidateFactRow>>>>;
pub type PendingWritesMap = Arc<DashMap<u32, oneshot::Sender<WriteAck>>>;

pub struct NodeStorageBridge {
    pub fetch_embeddings_tsfn: ThreadsafeFunction<(u32, String), ErrorStrategy::Fatal>,
    pub fetch_facts_by_ids_tsfn: ThreadsafeFunction<(u32, String), ErrorStrategy::Fatal>,
    pub write_batch_tsfn: ThreadsafeFunction<(u32, String), ErrorStrategy::Fatal>,
    pub pending_embeddings: PendingEmbeddingsMap,
    pub pending_facts: PendingFactsMap,
    pub pending_writes: PendingWritesMap,
    pub next_id: AtomicU32,
}

impl StorageBridge for NodeStorageBridge {
    fn fetch_embeddings(
        &self,
        entity_id: &str,
        limit: usize,
    ) -> std::result::Result<Vec<EmbeddingRow>, HostStorageError> {
        let payload = serde_json::json!({ "entity_id": entity_id, "limit": limit }).to_string();
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = oneshot::channel();

        self.pending_embeddings.insert(id, tx);

        let status = self
            .fetch_embeddings_tsfn
            .call((id, payload), ThreadsafeFunctionCallMode::NonBlocking);

        // Fail gracefully if the TS function queue fails, preventing thread lockup
        if status != napi::Status::Ok {
            self.pending_embeddings.remove(&id);
            return Err(HostStorageError::new(
                "NAPI_ERR",
                "Failed to queue JS callback",
            ));
        }

        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                match timeout(JS_CALLBACK_TIMEOUT, rx).await {
                    Ok(Ok(rows)) => Ok(rows),
                    Ok(Err(_)) => Err(HostStorageError::new("NAPI_ERR", "Channel dropped")),
                    Err(_) => Err(HostStorageError::new(
                        "TIMEOUT",
                        "fetchEmbeddings JS callback did not respond within 30s",
                    )),
                }
            })
        })
    }

    fn fetch_facts_by_ids(
        &self,
        ids: &[FactId],
    ) -> std::result::Result<Vec<CandidateFactRow>, HostStorageError> {
        let payload = serde_json::json!({ "ids": ids }).to_string();
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = oneshot::channel();

        self.pending_facts.insert(id, tx);

        let status = self
            .fetch_facts_by_ids_tsfn
            .call((id, payload), ThreadsafeFunctionCallMode::NonBlocking);

        if status != napi::Status::Ok {
            self.pending_facts.remove(&id);
            return Err(HostStorageError::new(
                "NAPI_ERR",
                "Failed to queue JS callback",
            ));
        }

        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                match timeout(JS_CALLBACK_TIMEOUT, rx).await {
                    Ok(Ok(rows)) => Ok(rows),
                    Ok(Err(_)) => Err(HostStorageError::new("NAPI_ERR", "Channel dropped")),
                    Err(_) => Err(HostStorageError::new(
                        "TIMEOUT",
                        "fetchFactsByIds JS callback did not respond within 30s",
                    )),
                }
            })
        })
    }

    fn write_batch(&self, batch: &WriteBatch) -> std::result::Result<WriteAck, HostStorageError> {
        let payload = serde_json::to_string(batch)
            .map_err(|e| HostStorageError::new("JSON_ERR", e.to_string()))?;
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = oneshot::channel();

        self.pending_writes.insert(id, tx);

        let status = self
            .write_batch_tsfn
            .call((id, payload), ThreadsafeFunctionCallMode::NonBlocking);

        if status != napi::Status::Ok {
            self.pending_writes.remove(&id);
            return Err(HostStorageError::new(
                "NAPI_ERR",
                "Failed to queue JS callback",
            ));
        }

        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                match timeout(JS_CALLBACK_TIMEOUT, rx).await {
                    Ok(Ok(ack)) => Ok(ack),
                    Ok(Err(_)) => Err(HostStorageError::new("NAPI_ERR", "Channel dropped")),
                    Err(_) => Err(HostStorageError::new(
                        "TIMEOUT",
                        "writeBatch JS callback did not respond within 30s",
                    )),
                }
            })
        })
    }
}
