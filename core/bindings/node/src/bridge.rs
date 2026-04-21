use engine_orchestrator::search::FactId;
use engine_orchestrator::storage::{
    CandidateFactRow, EmbeddingRow, HostStorageError, StorageBridge, WriteAck, WriteBatch,
};
use napi::threadsafe_function::{ErrorStrategy, ThreadsafeFunction, ThreadsafeFunctionCallMode};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::oneshot;

pub type PendingEmbeddingsMap = Arc<Mutex<HashMap<u32, oneshot::Sender<Vec<EmbeddingRow>>>>>;
pub type PendingFactsMap = Arc<Mutex<HashMap<u32, oneshot::Sender<Vec<CandidateFactRow>>>>>;
pub type PendingWritesMap = Arc<Mutex<HashMap<u32, oneshot::Sender<WriteAck>>>>;

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
        self.pending_embeddings.lock().unwrap().insert(id, tx);

        self.fetch_embeddings_tsfn
            .call((id, payload), ThreadsafeFunctionCallMode::NonBlocking);

        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                rx.await
                    .map_err(|_| HostStorageError::new("NAPI_ERR", "Channel dropped"))
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
        self.pending_facts.lock().unwrap().insert(id, tx);

        self.fetch_facts_by_ids_tsfn
            .call((id, payload), ThreadsafeFunctionCallMode::NonBlocking);

        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                rx.await
                    .map_err(|_| HostStorageError::new("NAPI_ERR", "Channel dropped"))
            })
        })
    }

    fn write_batch(&self, batch: &WriteBatch) -> std::result::Result<WriteAck, HostStorageError> {
        let payload = serde_json::to_string(batch)
            .map_err(|e| HostStorageError::new("JSON_ERR", e.to_string()))?;
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = oneshot::channel();
        self.pending_writes.lock().unwrap().insert(id, tx);

        self.write_batch_tsfn
            .call((id, payload), ThreadsafeFunctionCallMode::NonBlocking);

        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                rx.await
                    .map_err(|_| HostStorageError::new("NAPI_ERR", "Channel dropped"))
            })
        })
    }
}
