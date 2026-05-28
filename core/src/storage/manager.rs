use std::sync::Arc;

use parking_lot::RwLock;

use crate::search::FactId;
use crate::storage::bridge::StorageBridge;
use crate::storage::builder;
use crate::storage::connection::ConnectionFactory;
use crate::storage::dialect::Dialect;
use crate::storage::drivers::{mysql, postgresql, sqlite};
use crate::storage::models::{
    CandidateFactRow, EmbeddingRow, HostStorageError, WriteAck, WriteBatch,
};

type EmbedFn = Box<dyn Fn(Vec<String>) -> Vec<Vec<f32>> + Send + Sync>;

/// Implements [`StorageBridge`] entirely in Rust.
///
/// Owns all SQL logic, migration running, and transaction orchestration.
/// Delegates raw execution to the host language (TS or Python) via the
/// [`ConnectionFactory`] — no connection is held between calls.
pub struct RustStorageManager {
    factory: Arc<dyn ConnectionFactory>,
    dialect: Dialect,
    /// Wired in after construction to avoid a circular Arc dependency.
    /// Mirrors TS's `StorageManager.setEmbedder()`.
    embed: RwLock<Option<EmbedFn>>,
}

impl RustStorageManager {
    pub fn new(factory: Arc<dyn ConnectionFactory>, dialect: Dialect) -> Self {
        Self {
            factory,
            dialect,
            embed: RwLock::new(None),
        }
    }

    pub fn set_embedder(&self, f: EmbedFn) {
        *self.embed.write() = Some(f);
    }

    fn embed_texts(&self, texts: Vec<String>) -> Vec<Vec<f32>> {
        if let Some(embedder) = self.embed.read().as_ref() {
            embedder(texts)
        } else {
            vec![]
        }
    }

    // ── connection helper ─────────────────────────────────────────────────────

    /// Acquires a connection, runs `f`, then closes it — even on error.
    fn with_conn<T>(
        &self,
        f: impl FnOnce(
            &dyn crate::storage::connection::StorageConnection,
        ) -> Result<T, HostStorageError>,
    ) -> Result<T, HostStorageError> {
        let conn = self.factory.acquire()?;
        let result = f(&*conn);
        conn.close();
        result
    }

    // ── dispatch helpers ──────────────────────────────────────────────────────

    fn do_entity_create(
        &self,
        conn: &dyn crate::storage::connection::StorageConnection,
        external_id: &str,
    ) -> Result<Option<i64>, HostStorageError> {
        match &self.dialect {
            Dialect::Sqlite => sqlite::entity_create(conn, external_id),
            Dialect::Postgresql | Dialect::Cockroachdb => {
                postgresql::entity_create(conn, external_id)
            }
            Dialect::Mysql => mysql::entity_create(conn, external_id),
        }
    }

    fn do_process_create(
        &self,
        conn: &dyn crate::storage::connection::StorageConnection,
        external_id: &str,
    ) -> Result<Option<i64>, HostStorageError> {
        match &self.dialect {
            Dialect::Sqlite => sqlite::process_create(conn, external_id),
            Dialect::Postgresql | Dialect::Cockroachdb => {
                postgresql::process_create(conn, external_id)
            }
            Dialect::Mysql => mysql::process_create(conn, external_id),
        }
    }

    fn do_session_create(
        &self,
        conn: &dyn crate::storage::connection::StorageConnection,
        uuid: &str,
        entity_id: Option<i64>,
        process_id: Option<i64>,
    ) -> Result<Option<i64>, HostStorageError> {
        match &self.dialect {
            Dialect::Sqlite => sqlite::session_create(conn, uuid, entity_id, process_id),
            Dialect::Postgresql | Dialect::Cockroachdb => {
                postgresql::session_create(conn, uuid, entity_id, process_id)
            }
            Dialect::Mysql => mysql::session_create(conn, uuid, entity_id, process_id),
        }
    }

    fn do_session_get_id(
        &self,
        conn: &dyn crate::storage::connection::StorageConnection,
        uuid: &str,
    ) -> Result<Option<i64>, HostStorageError> {
        match &self.dialect {
            Dialect::Sqlite => sqlite::session_get_id(conn, uuid),
            Dialect::Postgresql | Dialect::Cockroachdb => postgresql::session_get_id(conn, uuid),
            Dialect::Mysql => mysql::session_get_id(conn, uuid),
        }
    }

    fn do_conversation_get_id_by_session(
        &self,
        conn: &dyn crate::storage::connection::StorageConnection,
        session_id: i64,
    ) -> Result<Option<i64>, HostStorageError> {
        match &self.dialect {
            Dialect::Sqlite => sqlite::conversation_get_id_by_session(conn, session_id),
            Dialect::Postgresql | Dialect::Cockroachdb => {
                postgresql::conversation_get_id_by_session(conn, session_id)
            }
            Dialect::Mysql => mysql::conversation_get_id_by_session(conn, session_id),
        }
    }

    fn do_conversation_create(
        &self,
        conn: &dyn crate::storage::connection::StorageConnection,
        session_id: i64,
        timeout: i64,
    ) -> Result<Option<i64>, HostStorageError> {
        match &self.dialect {
            Dialect::Sqlite => sqlite::conversation_create(conn, session_id, timeout),
            Dialect::Postgresql | Dialect::Cockroachdb => {
                postgresql::conversation_create(conn, session_id, timeout)
            }
            Dialect::Mysql => mysql::conversation_create(conn, session_id, timeout),
        }
    }

    fn do_conversation_update(
        &self,
        conn: &dyn crate::storage::connection::StorageConnection,
        id: i64,
        summary: &str,
    ) -> Result<(), HostStorageError> {
        match &self.dialect {
            Dialect::Sqlite => sqlite::conversation_update(conn, id, summary),
            Dialect::Postgresql | Dialect::Cockroachdb => {
                postgresql::conversation_update(conn, id, summary)
            }
            Dialect::Mysql => mysql::conversation_update(conn, id, summary),
        }
    }

    fn do_conversation_message_create(
        &self,
        conn: &dyn crate::storage::connection::StorageConnection,
        conversation_id: i64,
        role: &str,
        content: &str,
    ) -> Result<(), HostStorageError> {
        match &self.dialect {
            Dialect::Sqlite => {
                sqlite::conversation_message_create(conn, conversation_id, role, content)
            }
            Dialect::Postgresql | Dialect::Cockroachdb => {
                postgresql::conversation_message_create(conn, conversation_id, role, content)
            }
            Dialect::Mysql => {
                mysql::conversation_message_create(conn, conversation_id, role, content)
            }
        }
    }

    fn do_entity_fact_create(
        &self,
        conn: &dyn crate::storage::connection::StorageConnection,
        entity_id: i64,
        facts: &[String],
        embeddings: &[Vec<f32>],
        conversation_id: Option<i64>,
    ) -> Result<(), HostStorageError> {
        match &self.dialect {
            Dialect::Sqlite => {
                sqlite::entity_fact_create(conn, entity_id, facts, embeddings, conversation_id)
            }
            Dialect::Postgresql | Dialect::Cockroachdb => {
                postgresql::entity_fact_create(conn, entity_id, facts, embeddings, conversation_id)
            }
            Dialect::Mysql => {
                mysql::entity_fact_create(conn, entity_id, facts, embeddings, conversation_id)
            }
        }
    }

    fn do_entity_fact_create_without_embedding(
        &self,
        conn: &dyn crate::storage::connection::StorageConnection,
        entity_id: i64,
        content: &str,
    ) -> Result<(), HostStorageError> {
        match &self.dialect {
            Dialect::Sqlite => {
                sqlite::entity_fact_create_without_embedding(conn, entity_id, content)
            }
            Dialect::Postgresql | Dialect::Cockroachdb => {
                postgresql::entity_fact_create_without_embedding(conn, entity_id, content)
            }
            Dialect::Mysql => mysql::entity_fact_create_without_embedding(conn, entity_id, content),
        }
    }

    fn do_knowledge_graph_create(
        &self,
        conn: &dyn crate::storage::connection::StorageConnection,
        entity_id: i64,
        triples: &[serde_json::Value],
    ) -> Result<(), HostStorageError> {
        match &self.dialect {
            Dialect::Sqlite => sqlite::knowledge_graph_create(conn, entity_id, triples),
            Dialect::Postgresql | Dialect::Cockroachdb => {
                postgresql::knowledge_graph_create(conn, entity_id, triples)
            }
            Dialect::Mysql => mysql::knowledge_graph_create(conn, entity_id, triples),
        }
    }

    fn do_process_attribute_create(
        &self,
        conn: &dyn crate::storage::connection::StorageConnection,
        process_id: i64,
        attributes: &[String],
    ) -> Result<(), HostStorageError> {
        match &self.dialect {
            Dialect::Sqlite => sqlite::process_attribute_create(conn, process_id, attributes),
            Dialect::Postgresql | Dialect::Cockroachdb => {
                postgresql::process_attribute_create(conn, process_id, attributes)
            }
            Dialect::Mysql => mysql::process_attribute_create(conn, process_id, attributes),
        }
    }

    fn do_entity_fact_get_embeddings(
        &self,
        conn: &dyn crate::storage::connection::StorageConnection,
        entity_id: i64,
        limit: usize,
    ) -> Result<Vec<EmbeddingRow>, HostStorageError> {
        match &self.dialect {
            Dialect::Sqlite => sqlite::entity_fact_get_embeddings(conn, entity_id, limit),
            Dialect::Postgresql | Dialect::Cockroachdb => {
                postgresql::entity_fact_get_embeddings(conn, entity_id, limit)
            }
            Dialect::Mysql => mysql::entity_fact_get_embeddings(conn, entity_id, limit),
        }
    }

    fn do_entity_fact_get_by_ids(
        &self,
        conn: &dyn crate::storage::connection::StorageConnection,
        ids: &[FactId],
    ) -> Result<Vec<CandidateFactRow>, HostStorageError> {
        match &self.dialect {
            Dialect::Sqlite => sqlite::entity_fact_get_by_ids(conn, ids),
            Dialect::Postgresql | Dialect::Cockroachdb => {
                postgresql::entity_fact_get_by_ids(conn, ids)
            }
            Dialect::Mysql => mysql::entity_fact_get_by_ids(conn, ids),
        }
    }

    // ── write_batch internals ─────────────────────────────────────────────────

    /// Clones the batch and injects pre-computed embeddings into any op that needs
    /// them but doesn't already carry them. Must be called outside the DB transaction
    /// window — ONNX inference here avoids holding a write lock or inflating the
    /// CockroachDB transaction window (which increases 40001 retry frequency).
    fn precompute_embeddings(&self, batch: &WriteBatch) -> WriteBatch {
        let ops = batch
            .ops
            .iter()
            .map(|op| match op.op_type.as_str() {
                "entity_fact.create" if !op.payload["fact_embeddings"].is_array() => {
                    let facts: Vec<String> = op.payload["facts"]
                        .as_array()
                        .map(|a| {
                            a.iter()
                                .filter_map(|v| v.as_str().map(String::from))
                                .collect()
                        })
                        .unwrap_or_default();
                    if facts.is_empty() {
                        return op.clone();
                    }
                    let embeddings = self.embed_texts(facts);
                    let mut new_op = op.clone();
                    new_op.payload["fact_embeddings"] = serde_json::json!(embeddings);
                    new_op
                }
                "upsert_fact" if op.payload.get("content_embedding").is_none() => {
                    if let Some(content) = op.payload["content"].as_str() {
                        let embedding = self
                            .embed_texts(vec![content.to_string()])
                            .into_iter()
                            .next()
                            .unwrap_or_default();
                        let mut new_op = op.clone();
                        new_op.payload["content_embedding"] = serde_json::json!(embedding);
                        return new_op;
                    }
                    op.clone()
                }
                _ => op.clone(),
            })
            .collect();
        WriteBatch { ops }
    }

    fn execute_batch_ops(
        &self,
        conn: &dyn crate::storage::connection::StorageConnection,
        batch: &WriteBatch,
    ) -> Result<(), HostStorageError> {
        for op in &batch.ops {
            match op.op_type.as_str() {
                // TS/BYODB-only: the Python SDK persists conversation messages through its own
                // augmentation path and does not use write_batch for this op.
                "conversation_message.create" => {
                    let conv_id_str = op.payload["conversation_id"]
                        .as_str()
                        .unwrap_or("")
                        .to_string();
                    let session_id = self
                        .do_session_create(conn, &conv_id_str, None, None)?
                        .ok_or_else(|| {
                            HostStorageError::new("INTERNAL", "do_session_create returned no id")
                        })?;
                    let conv_id = self
                        .do_conversation_create(conn, session_id, 30)?
                        .ok_or_else(|| {
                            HostStorageError::new(
                                "INTERNAL",
                                "do_conversation_create returned no id",
                            )
                        })?;

                    if let Some(messages) = op.payload["messages"].as_array() {
                        for msg in messages {
                            let role = msg["role"].as_str().unwrap_or("");
                            let content = msg["content"].as_str().unwrap_or("");
                            self.do_conversation_message_create(conn, conv_id, role, content)?;
                        }
                    }
                }
                "entity_fact.create" => {
                    let entity_id_str = op.payload["entity_id"].as_str().unwrap_or("").to_string();
                    let internal_entity_id = self
                        .do_entity_create(conn, &entity_id_str)?
                        .ok_or_else(|| {
                            HostStorageError::new("INTERNAL", "do_entity_create returned no id")
                        })?;

                    let facts: Vec<String> = op.payload["facts"]
                        .as_array()
                        .map(|a| {
                            a.iter()
                                .filter_map(|v| v.as_str().map(String::from))
                                .collect()
                        })
                        .unwrap_or_default();

                    let embeddings =
                        if let Some(raw_embs) = op.payload["fact_embeddings"].as_array() {
                            // Embeddings provided by the caller (e.g. from a manual write call).
                            raw_embs
                                .iter()
                                .map(|emb| {
                                    emb.as_array()
                                        .map(|arr| {
                                            arr.iter()
                                                .filter_map(|v| v.as_f64().map(|f| f as f32))
                                                .collect()
                                        })
                                        .unwrap_or_default()
                                })
                                .collect()
                        } else if !facts.is_empty() {
                            self.embed_texts(facts.clone())
                        } else {
                            vec![]
                        };

                    let internal_conv_id =
                        if let Some(conv_id_str) = op.payload["conversation_id"].as_str() {
                            if !conv_id_str.is_empty() {
                                let session_id = self
                                    .do_session_create(
                                        conn,
                                        conv_id_str,
                                        Some(internal_entity_id),
                                        None,
                                    )?
                                    .ok_or_else(|| {
                                        HostStorageError::new(
                                            "INTERNAL",
                                            "do_session_create returned no id",
                                        )
                                    })?;
                                self.do_conversation_create(conn, session_id, 30)?
                            } else {
                                None
                            }
                        } else {
                            None
                        };

                    self.do_entity_fact_create(
                        conn,
                        internal_entity_id,
                        &facts,
                        &embeddings,
                        internal_conv_id,
                    )?;
                }
                "knowledge_graph.create" => {
                    let entity_id_str = op.payload["entity_id"].as_str().unwrap_or("").to_string();
                    let internal_entity_id = self
                        .do_entity_create(conn, &entity_id_str)?
                        .ok_or_else(|| {
                            HostStorageError::new("INTERNAL", "do_entity_create returned no id")
                        })?;
                    let triples = op.payload["semantic_triples"]
                        .as_array()
                        .map(Vec::as_slice)
                        .unwrap_or(&[]);
                    self.do_knowledge_graph_create(conn, internal_entity_id, triples)?;
                }
                "process_attribute.create" => {
                    let process_id_str =
                        op.payload["process_id"].as_str().unwrap_or("").to_string();
                    if process_id_str.is_empty() {
                        continue;
                    }
                    let internal_process_id = self
                        .do_process_create(conn, &process_id_str)?
                        .ok_or_else(|| {
                            HostStorageError::new("INTERNAL", "do_process_create returned no id")
                        })?;
                    let attributes: Vec<String> = match op.payload["attributes"].as_array() {
                        Some(arr) => arr
                            .iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect(),
                        None => op.payload["attributes"]
                            .as_object()
                            .map(|o| {
                                o.values()
                                    .filter_map(|v| v.as_str().map(String::from))
                                    .collect()
                            })
                            .unwrap_or_default(),
                    };
                    self.do_process_attribute_create(conn, internal_process_id, &attributes)?;
                }
                "conversation.update" => {
                    let conv_id_str = op.payload["conversation_id"]
                        .as_str()
                        .unwrap_or("")
                        .to_string();
                    let session_id = self
                        .do_session_create(conn, &conv_id_str, None, None)?
                        .ok_or_else(|| {
                            HostStorageError::new("INTERNAL", "do_session_create returned no id")
                        })?;
                    let conv_id = self
                        .do_conversation_create(conn, session_id, 30)?
                        .ok_or_else(|| {
                            HostStorageError::new(
                                "INTERNAL",
                                "do_conversation_create returned no id",
                            )
                        })?;
                    let summary = op.payload["summary"].as_str().unwrap_or("");
                    self.do_conversation_update(conn, conv_id, summary)?;
                }
                "upsert_fact" => {
                    let entity_id_str = op.payload["entity_id"].as_str().unwrap_or("").to_string();
                    let internal_entity_id = self
                        .do_entity_create(conn, &entity_id_str)?
                        .ok_or_else(|| {
                            HostStorageError::new("INTERNAL", "do_entity_create returned no id")
                        })?;
                    if let Some(content) = op.payload["content"].as_str() {
                        // Read the embedding pre-computed by precompute_embeddings (outside the tx).
                        // Falls back to embed_texts only if somehow missing (defensive).
                        let pre = op.payload["content_embedding"].as_array().map(|a| {
                            a.iter()
                                .filter_map(|v| v.as_f64().map(|f| f as f32))
                                .collect::<Vec<f32>>()
                        });
                        let embeddings = pre
                            .map(|e| vec![e])
                            .unwrap_or_else(|| self.embed_texts(vec![content.to_string()]));
                        if let Some(embedding) =
                            embeddings.into_iter().next().filter(|e| !e.is_empty())
                        {
                            self.do_entity_fact_create(
                                conn,
                                internal_entity_id,
                                &[content.to_string()],
                                &[embedding],
                                None,
                            )?;
                        } else {
                            self.do_entity_fact_create_without_embedding(
                                conn,
                                internal_entity_id,
                                content,
                            )?;
                        }
                    }
                }
                unknown => {
                    return Err(HostStorageError::new(
                        "UNSUPPORTED_OP",
                        format!("unsupported write op type: {unknown}"),
                    ));
                }
            }
        }
        Ok(())
    }
}

impl StorageBridge for RustStorageManager {
    fn build(&self) -> Result<(), HostStorageError> {
        let conn = self.factory.acquire()?;
        let result = builder::run(&*conn, &self.dialect);
        conn.close();
        result
    }

    fn get_conversation_history(
        &self,
        session_id: &str,
    ) -> Result<Vec<serde_json::Value>, HostStorageError> {
        self.with_conn(|conn| {
            let session_internal_id = match self.do_session_get_id(conn, session_id)? {
                Some(id) => id,
                None => return Ok(vec![]),
            };
            let conv_id = match self.do_conversation_get_id_by_session(conn, session_internal_id)? {
                Some(id) => id,
                None => return Ok(vec![]),
            };
            let messages = match &self.dialect {
                Dialect::Sqlite => sqlite::conversation_messages_read(conn, conv_id)?,
                Dialect::Postgresql | Dialect::Cockroachdb => {
                    postgresql::conversation_messages_read(conn, conv_id)?
                }
                Dialect::Mysql => mysql::conversation_messages_read(conn, conv_id)?,
            };
            Ok(messages
                .into_iter()
                .map(|(role, content)| serde_json::json!({ "role": role, "content": content }))
                .collect())
        })
    }

    fn fetch_embeddings(
        &self,
        entity_id: &str,
        limit: usize,
    ) -> Result<Vec<EmbeddingRow>, HostStorageError> {
        self.with_conn(|conn| {
            let internal_id = match &self.dialect {
                Dialect::Sqlite => sqlite::entity_get_id(conn, entity_id)?,
                Dialect::Postgresql | Dialect::Cockroachdb => {
                    postgresql::entity_get_id(conn, entity_id)?
                }
                Dialect::Mysql => mysql::entity_get_id(conn, entity_id)?,
            };
            match internal_id {
                Some(id) => self.do_entity_fact_get_embeddings(conn, id, limit),
                None => Ok(vec![]),
            }
        })
    }

    fn fetch_facts_by_ids(
        &self,
        ids: &[FactId],
    ) -> Result<Vec<CandidateFactRow>, HostStorageError> {
        self.with_conn(|conn| self.do_entity_fact_get_by_ids(conn, ids))
    }

    /// Executes all write ops in a single transaction per the Python `connection_context` model.
    ///
    /// CockroachDB can return serialization error code `40001` under concurrent load.
    /// The correct response is to retry the entire transaction with a fresh connection,
    /// which is what the retry loop here does.
    fn write_batch(&self, batch: &WriteBatch) -> Result<WriteAck, HostStorageError> {
        if batch.ops.is_empty() {
            return Ok(WriteAck { written_ops: 0 });
        }

        // Embed outside the transaction so ONNX inference doesn't extend the lock window.
        let batch = self.precompute_embeddings(batch);

        let op_count = batch.ops.len();
        const MAX_RETRIES: u32 = 5;
        let mut last_err: Option<HostStorageError> = None;

        for attempt in 0..=MAX_RETRIES {
            let conn = self.factory.acquire()?;

            if let Err(e) = conn.begin() {
                conn.close();
                return Err(e);
            }

            // Fold commit into the result so a 40001 at commit time is retried like any
            // other serialization failure — CockroachDB commonly rejects at commit.
            let result = self
                .execute_batch_ops(&*conn, &batch)
                .and_then(|_| conn.commit());

            match result {
                Ok(()) => {
                    conn.close();
                    return Ok(WriteAck {
                        written_ops: op_count,
                    });
                }
                Err(e) => {
                    let _ = conn.rollback();
                    conn.close();

                    // CockroachDB serialization failure — retry with exponential backoff.
                    // 50ms, 100ms, 200ms, 400ms, 800ms (capped at 1000ms).
                    // Production callers should add random jitter via the `rand` crate.
                    if e.code == "40001" && self.dialect.is_cockroachdb() && attempt < MAX_RETRIES {
                        let backoff =
                            std::time::Duration::from_millis((50 * 2_u64.pow(attempt)).min(1000));
                        std::thread::sleep(backoff);
                        last_err = Some(e);
                        continue;
                    }

                    return Err(e);
                }
            }
        }

        Err(last_err
            .unwrap_or_else(|| HostStorageError::new("ERR", "write_batch exhausted retries")))
    }

    fn shutdown(&self) {
        self.factory.shutdown();
    }
}
