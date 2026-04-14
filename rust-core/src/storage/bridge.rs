use crate::search::FactId;
use crate::storage::models::{
    CandidateFactRow, EmbeddingRow, HostStorageError, WriteAck, WriteBatch,
};

pub trait StorageBridge: Send + Sync {
    fn fetch_embeddings(
        &self,
        entity_id: &str,
        limit: usize,
    ) -> Result<Vec<EmbeddingRow>, HostStorageError>;

    fn fetch_facts_by_ids(&self, ids: &[FactId])
    -> Result<Vec<CandidateFactRow>, HostStorageError>;

    fn write_batch(&self, batch: &WriteBatch) -> Result<WriteAck, HostStorageError>;
}
