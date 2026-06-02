use std::sync::Arc;

use rand::Rng;

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

/// SQL storage bridge backed by a user-supplied [`ConnectionFactory`]. No connection is held between calls.
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

    fn entity_create(
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

    fn process_create(
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

    fn session_create(
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

    fn session_get_id(
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

    fn conversation_get_id_by_session(
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

    fn conversation_create(
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

    fn conversation_update(
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

    fn conversation_message_create(
        &self,
        conn: &dyn crate::storage::connection::StorageConnection,
        conversation_id: i64,
        role: &str,
        msg_type: &str,
        content: &str,
    ) -> Result<(), HostStorageError> {
        match &self.dialect {
            Dialect::Sqlite => {
                sqlite::conversation_message_create(conn, conversation_id, role, msg_type, content)
            }
            Dialect::Postgresql | Dialect::Cockroachdb => postgresql::conversation_message_create(
                conn,
                conversation_id,
                role,
                msg_type,
                content,
            ),
            Dialect::Mysql => {
                mysql::conversation_message_create(conn, conversation_id, role, msg_type, content)
            }
        }
    }

    fn entity_fact_create(
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

    fn require_entity_id(
        &self,
        conn: &dyn crate::storage::connection::StorageConnection,
        external_id: &str,
    ) -> Result<i64, HostStorageError> {
        self.entity_create(conn, external_id)?
            .ok_or_else(|| HostStorageError::new("INTERNAL", "entity_create returned no id"))
    }

    fn knowledge_graph_create(
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

    fn process_attribute_create(
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

    fn entity_fact_get_embeddings(
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

    fn entity_fact_get_by_ids(
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

    // Must run before the transaction — ONNX inference inside a tx inflates the CockroachDB lock window.
    fn precompute_embeddings(&self, batch: &WriteBatch) -> WriteBatch {
        let ops = batch
            .ops
            .iter()
            .map(|op| match op.op_type.as_str() {
                "entity_fact.create" => {
                    let facts_arr = op.payload["facts"].as_array();
                    let embs_arr = op.payload["fact_embeddings"].as_array();
                    match (facts_arr, embs_arr) {
                        (None, _) => op.clone(),
                        (Some(facts), Some(embs)) if embs.len() == facts.len() => {
                            // Aligned — only re-filter if blank facts need stripping.
                            let has_blanks = facts
                                .iter()
                                .any(|f| f.as_str().map(|s| s.trim().is_empty()).unwrap_or(false));
                            if !has_blanks {
                                return op.clone();
                            }
                            let (f_out, e_out): (Vec<_>, Vec<_>) = facts
                                .iter()
                                .zip(embs.iter())
                                .filter(|(f, _)| {
                                    f.as_str().map(|s| !s.trim().is_empty()).unwrap_or(false)
                                })
                                .map(|(f, e)| (f.clone(), e.clone()))
                                .unzip();
                            let mut new_op = op.clone();
                            new_op.payload["facts"] = serde_json::json!(f_out);
                            new_op.payload["fact_embeddings"] = serde_json::json!(e_out);
                            new_op
                        }
                        (Some(facts), _) => {
                            // No embeddings supplied, or misaligned count — zip would truncate
                            // valid facts. Filter blank facts and compute embeddings outside
                            // the transaction window instead.
                            let filtered: Vec<String> = facts
                                .iter()
                                .filter_map(|v| {
                                    let s = v.as_str()?;
                                    if s.trim().is_empty() {
                                        None
                                    } else {
                                        Some(s.to_string())
                                    }
                                })
                                .collect();
                            if filtered.is_empty() {
                                return op.clone();
                            }
                            let embeddings = self.embed_texts(filtered.clone());
                            let mut new_op = op.clone();
                            new_op.payload["facts"] = serde_json::json!(filtered);
                            new_op.payload["fact_embeddings"] = serde_json::json!(embeddings);
                            new_op
                        }
                    }
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

    // Stringify a scalar JSON value — matches Python's str() used in _normalize_attributes.
    fn scalar_to_string(v: &serde_json::Value) -> Option<String> {
        match v {
            serde_json::Value::String(s) => Some(s.clone()),
            serde_json::Value::Number(n) => Some(n.to_string()),
            serde_json::Value::Bool(b) => Some(b.to_string()),
            _ => None,
        }
    }

    // Accepts both JSON string and integer values so a numeric ID sent from the
    // TS bridge never silently becomes an empty string via `.as_str()`.
    fn coerce_id_str(v: &serde_json::Value) -> String {
        if let Some(s) = v.as_str() {
            s.to_string()
        } else if let Some(n) = v.as_i64() {
            n.to_string()
        } else if let Some(n) = v.as_u64() {
            n.to_string()
        } else {
            String::new()
        }
    }

    fn execute_batch_ops(
        &self,
        conn: &dyn crate::storage::connection::StorageConnection,
        batch: &WriteBatch,
    ) -> Result<usize, HostStorageError> {
        let mut applied: usize = 0;
        for op in &batch.ops {
            match op.op_type.as_str() {
                // TS/BYODB-only: Python persists messages through its own augmentation path.
                "conversation_message.create" => {
                    let conv_id = Self::coerce_id_str(&op.payload["conversation_id"]);
                    if conv_id.is_empty() {
                        continue;
                    }
                    let messages = match op.payload["messages"].as_array() {
                        Some(msgs) if !msgs.is_empty() => msgs,
                        _ => continue,
                    };
                    let session_id = self
                        .session_create(conn, &conv_id, None, None)?
                        .ok_or_else(|| {
                            HostStorageError::new("INTERNAL", "session_create returned no id")
                        })?;
                    let conv_id =
                        self.conversation_create(conn, session_id, 30)?
                            .ok_or_else(|| {
                                HostStorageError::new(
                                    "INTERNAL",
                                    "conversation_create returned no id",
                                )
                            })?;
                    for msg in messages {
                        let role = msg["role"].as_str().unwrap_or("");
                        let msg_type = msg["type"].as_str().unwrap_or("text");
                        let content = msg["content"].as_str().unwrap_or("");
                        self.conversation_message_create(conn, conv_id, role, msg_type, content)?;
                    }
                }
                "entity_fact.create" => {
                    let entity_id = Self::coerce_id_str(&op.payload["entity_id"]);
                    if entity_id.is_empty() {
                        continue;
                    }

                    let facts: Vec<String> = op.payload["facts"]
                        .as_array()
                        .map(|a| {
                            a.iter()
                                .filter_map(|v| {
                                    let s = v.as_str()?;
                                    if s.trim().is_empty() {
                                        None
                                    } else {
                                        Some(s.to_string())
                                    }
                                })
                                .collect()
                        })
                        .unwrap_or_default();

                    if facts.is_empty() {
                        continue;
                    }

                    let entity_id = self.require_entity_id(conn, &entity_id)?;

                    let embeddings = {
                        let raw_embs =
                            op.payload["fact_embeddings"].as_array().ok_or_else(|| {
                                HostStorageError::new(
                                    "INTERNAL",
                                    "entity_fact.create: fact_embeddings missing; \
                                     precompute_embeddings must run before execute_batch_ops",
                                )
                            })?;
                        let deserialized: Vec<Vec<f32>> = raw_embs
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
                            .collect();
                        if deserialized.is_empty() || deserialized.len() != facts.len() {
                            // Misaligned — mirrors Python's recovery: re-embed all facts from scratch.
                            let recomputed = self.embed_texts(facts.clone());
                            if recomputed.len() == facts.len() {
                                recomputed
                            } else {
                                vec![]
                            }
                        } else {
                            deserialized
                        }
                    };

                    let conv_id = {
                        let conv_id_str = Self::coerce_id_str(&op.payload["conversation_id"]);
                        if !conv_id_str.is_empty() {
                            let session_id = self
                                .session_create(conn, &conv_id_str, Some(entity_id), None)?
                                .ok_or_else(|| {
                                    HostStorageError::new(
                                        "INTERNAL",
                                        "session_create returned no id",
                                    )
                                })?;
                            self.conversation_create(conn, session_id, 30)?
                        } else {
                            None
                        }
                    };

                    self.entity_fact_create(conn, entity_id, &facts, &embeddings, conv_id)?;
                }
                "knowledge_graph.create" => {
                    let entity_id = Self::coerce_id_str(&op.payload["entity_id"]);
                    if entity_id.is_empty() {
                        continue;
                    }
                    let entity_id = self.require_entity_id(conn, &entity_id)?;
                    let triples = op.payload["semantic_triples"]
                        .as_array()
                        .map(Vec::as_slice)
                        .unwrap_or(&[]);
                    if triples.is_empty() {
                        continue;
                    }
                    self.knowledge_graph_create(conn, entity_id, triples)?;
                }
                "process_attribute.create" => {
                    let process_id = Self::coerce_id_str(&op.payload["process_id"]);
                    if process_id.is_empty() {
                        continue;
                    }
                    let process_id = self.process_create(conn, &process_id)?.ok_or_else(|| {
                        HostStorageError::new("INTERNAL", "process_create returned no id")
                    })?;
                    let attributes: Vec<String> = match op.payload["attributes"].as_array() {
                        // Python: [str(x) for x in raw if str(x).strip()]
                        Some(arr) => arr
                            .iter()
                            .filter_map(|v| {
                                let s = Self::scalar_to_string(v)?;
                                if s.trim().is_empty() { None } else { Some(s) }
                            })
                            .collect(),
                        // Python: [f"{k}:{v}" for k, v in raw.items()]
                        None => op.payload["attributes"]
                            .as_object()
                            .map(|o| {
                                o.iter()
                                    .filter_map(|(k, v)| {
                                        Some(format!("{}:{}", k, Self::scalar_to_string(v)?))
                                    })
                                    .collect()
                            })
                            .unwrap_or_default(),
                    };
                    if attributes.is_empty() {
                        continue;
                    }
                    self.process_attribute_create(conn, process_id, &attributes)?;
                }
                "conversation.update" => {
                    let conv_id = Self::coerce_id_str(&op.payload["conversation_id"]);
                    let summary = op.payload["summary"].as_str().unwrap_or("");
                    if conv_id.is_empty() || summary.is_empty() {
                        continue;
                    }
                    // Look up only — don't create, or a stale ID could spawn an empty conversation that swallows the summary.
                    let session_id = match self.session_get_id(conn, &conv_id)? {
                        Some(id) => id,
                        None => continue,
                    };
                    let conv_id = match self.conversation_get_id_by_session(conn, session_id)? {
                        Some(id) => id,
                        None => continue,
                    };
                    self.conversation_update(conn, conv_id, summary)?;
                }
                "upsert_fact" => {
                    let entity_id = Self::coerce_id_str(&op.payload["entity_id"]);
                    if entity_id.is_empty() {
                        continue;
                    }
                    let content = match op.payload["content"].as_str() {
                        Some(c) if !c.trim().is_empty() => c,
                        _ => continue,
                    };
                    let entity_id = self.require_entity_id(conn, &entity_id)?;
                    // Embedding was pre-computed outside the tx; absent means no embedder is set.
                    let embeddings: Vec<Vec<f32>> = op.payload["content_embedding"]
                        .as_array()
                        .and_then(|a| {
                            let e: Vec<f32> = a
                                .iter()
                                .filter_map(|v| v.as_f64().map(|f| f as f32))
                                .collect();
                            if e.is_empty() { None } else { Some(vec![e]) }
                        })
                        .unwrap_or_default();
                    self.entity_fact_create(
                        conn,
                        entity_id,
                        &[content.to_string()],
                        &embeddings,
                        None,
                    )?;
                }
                unknown => {
                    return Err(HostStorageError::new(
                        "UNSUPPORTED_OP",
                        format!("unsupported write op type: {unknown}"),
                    ));
                }
            }
            applied += 1;
        }
        Ok(applied)
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
            let session_internal_id = match self.session_get_id(conn, session_id)? {
                Some(id) => id,
                None => return Ok(vec![]),
            };
            let conv_id = match self.conversation_get_id_by_session(conn, session_internal_id)? {
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
                Some(id) => self.entity_fact_get_embeddings(conn, id, limit),
                None => Ok(vec![]),
            }
        })
    }

    fn fetch_facts_by_ids(
        &self,
        ids: &[FactId],
    ) -> Result<Vec<CandidateFactRow>, HostStorageError> {
        self.with_conn(|conn| self.entity_fact_get_by_ids(conn, ids))
    }

    fn write_batch(&self, batch: &WriteBatch) -> Result<WriteAck, HostStorageError> {
        if batch.ops.is_empty() {
            return Ok(WriteAck { written_ops: 0 });
        }

        // Embed outside the transaction so ONNX inference doesn't extend the lock window.
        let batch = self.precompute_embeddings(batch);

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
                .and_then(|applied| conn.commit().map(|_| applied));

            match result {
                Ok(applied) => {
                    conn.close();
                    return Ok(WriteAck {
                        written_ops: applied,
                    });
                }
                Err(e) => {
                    let _ = conn.rollback();
                    conn.close();

                    // SQLSTATE 40001 (serialization failure) — retry with exponential backoff
                    // plus up to 50% random jitter to de-correlate concurrent retriers.
                    // Applies to CockroachDB, CockroachDB via pg adapter (reported as postgresql),
                    // and PostgreSQL under REPEATABLE READ / SERIALIZABLE isolation.
                    // Base: 50ms, 100ms, 200ms, 400ms, 800ms (capped at 1000ms).
                    if e.code == "40001" && attempt < MAX_RETRIES {
                        let base_ms = (50 * 2_u64.pow(attempt)).min(1000);
                        let jitter_ms = rand::thread_rng().gen_range(0..=base_ms / 2);
                        let backoff = std::time::Duration::from_millis(base_ms + jitter_ms);
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
