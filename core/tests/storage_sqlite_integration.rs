//! End-to-end tests for [`RustStorageManager`] against a real SQLite database.
//!
//! Exercises `build()`, `write_batch()`, and `fetch_embeddings()` through the
//! [`ConnectionFactory`] / [`StorageConnection`] traits — the same path the Node
//! bridge uses via the `storageCall` protocol.

use std::sync::{Arc, Mutex};

use base64::{Engine as _, engine::general_purpose::STANDARD};
use engine_orchestrator::search::FactId;
use engine_orchestrator::storage::{
    ConnectionFactory, Dialect, HostStorageError, RustStorageManager, SqlBind, StorageBridge,
    StorageConnection, WriteBatch, WriteOp,
};
use rusqlite::{Connection, Row, ToSql, types::ValueRef};
use uuid::Uuid;

fn db_err(e: rusqlite::Error) -> HostStorageError {
    HostStorageError::new("ERR", e.to_string())
}

fn bind_to_sql(bind: &SqlBind) -> Result<Box<dyn ToSql>, HostStorageError> {
    Ok(match bind {
        SqlBind::Null => Box::new(rusqlite::types::Null),
        SqlBind::Int(n) => Box::new(*n),
        SqlBind::Float(f) => Box::new(*f),
        SqlBind::Text(s) => Box::new(s.clone()),
        SqlBind::Bytes(b64) => {
            let bytes = STANDARD
                .decode(b64)
                .map_err(|e| HostStorageError::new("ERR", e.to_string()))?;
            Box::new(bytes)
        }
    })
}

fn row_to_json(row: &Row<'_>) -> rusqlite::Result<serde_json::Value> {
    let mut obj = serde_json::Map::new();
    for (i, name) in row.as_ref().column_names().iter().enumerate() {
        let val = match row.get_ref(i)? {
            ValueRef::Null => serde_json::Value::Null,
            ValueRef::Integer(n) => serde_json::json!(n),
            ValueRef::Real(f) => serde_json::json!(f),
            ValueRef::Text(s) => serde_json::Value::String(String::from_utf8_lossy(s).into_owned()),
            ValueRef::Blob(b) => serde_json::Value::String(STANDARD.encode(b)),
        };
        obj.insert((*name).to_string(), val);
    }
    Ok(serde_json::Value::Object(obj))
}

struct RusqliteConnection {
    conn: Arc<Mutex<Connection>>,
}

struct RusqliteFactory {
    conn: Arc<Mutex<Connection>>,
}

impl RusqliteFactory {
    fn new() -> Result<Self, HostStorageError> {
        let uri = format!("file:memori_{}?mode=memory&cache=shared", Uuid::new_v4());
        let conn = Connection::open(&uri).map_err(db_err)?;
        conn.execute_batch("PRAGMA foreign_keys = ON;")
            .map_err(db_err)?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }
}

impl StorageConnection for RusqliteConnection {
    fn execute(
        &self,
        sql: &str,
        binds: Vec<SqlBind>,
    ) -> Result<Vec<serde_json::Value>, HostStorageError> {
        let conn = self.conn.lock().unwrap();
        let params: Vec<Box<dyn ToSql>> =
            binds.iter().map(bind_to_sql).collect::<Result<_, _>>()?;
        let param_refs: Vec<&dyn ToSql> = params.iter().map(|p| p.as_ref()).collect();

        let mut stmt = conn.prepare(sql).map_err(db_err)?;
        if stmt.readonly() {
            let rows = stmt
                .query_map(rusqlite::params_from_iter(param_refs.iter()), row_to_json)
                .map_err(db_err)?;
            rows.collect::<Result<Vec<_>, _>>().map_err(db_err)
        } else {
            stmt.execute(rusqlite::params_from_iter(param_refs.iter()))
                .map_err(db_err)?;
            Ok(vec![])
        }
    }

    fn begin(&self) -> Result<(), HostStorageError> {
        self.conn
            .lock()
            .unwrap()
            .execute_batch("BEGIN")
            .map_err(db_err)
    }

    fn commit(&self) -> Result<(), HostStorageError> {
        self.conn
            .lock()
            .unwrap()
            .execute_batch("COMMIT")
            .map_err(db_err)
    }

    fn rollback(&self) -> Result<(), HostStorageError> {
        self.conn
            .lock()
            .unwrap()
            .execute_batch("ROLLBACK")
            .map_err(db_err)
    }

    fn close(&self) {}
}

impl ConnectionFactory for RusqliteFactory {
    fn acquire(&self) -> Result<Box<dyn StorageConnection>, HostStorageError> {
        Ok(Box::new(RusqliteConnection {
            conn: Arc::clone(&self.conn),
        }))
    }

    fn dialect(&self) -> &str {
        "sqlite"
    }
}

fn make_manager() -> RustStorageManager {
    let factory = RusqliteFactory::new().expect("sqlite factory");
    RustStorageManager::new(Arc::new(factory), Dialect::Sqlite)
}

#[test]
fn build_applies_sqlite_migrations() {
    let factory = Arc::new(RusqliteFactory::new().expect("factory"));
    let manager = RustStorageManager::new(factory.clone(), Dialect::Sqlite);
    manager.build().expect("build should succeed");

    let conn = factory.acquire().expect("connection");
    let rows = conn
        .execute("SELECT num FROM memori_schema_version", vec![])
        .expect("schema version read");
    assert!(
        !rows.is_empty(),
        "schema_version should be populated after build"
    );
}

#[test]
fn write_batch_upsert_and_fetch_embeddings() {
    let manager = make_manager();
    manager.build().expect("build");

    manager.set_embedder(Box::new(|texts: Vec<String>| {
        texts.into_iter().map(|_| vec![1.0_f32, 0.0, 0.0]).collect()
    }));

    let batch = WriteBatch {
        ops: vec![WriteOp {
            op_type: "upsert_fact".to_string(),
            payload: serde_json::json!({
                "entity_id": "test-entity",
                "content": "User likes autumn weather",
            }),
        }],
    };
    let ack = manager.write_batch(&batch).expect("write_batch");
    assert_eq!(ack.written_ops, 1);

    let embeddings = manager
        .fetch_embeddings("test-entity", 10)
        .expect("fetch_embeddings");
    assert_eq!(embeddings.len(), 1);
    assert!(matches!(embeddings[0].id, FactId::Int(_)));
    assert!(
        embeddings[0]
            .content_embedding_b64
            .as_deref()
            .is_some_and(|s| !s.is_empty()),
        "embedding blob should round-trip as base64"
    );
}

#[test]
fn get_conversation_history_returns_empty_for_unknown_session() {
    let manager = make_manager();
    manager.build().expect("build");

    let history = manager
        .get_conversation_history("no-such-session")
        .expect("history");
    assert!(history.is_empty());
}

#[test]
fn conversation_message_create_and_history_round_trip() {
    let manager = make_manager();
    manager.build().expect("build");

    let batch = WriteBatch {
        ops: vec![WriteOp {
            op_type: "conversation_message.create".to_string(),
            payload: serde_json::json!({
                "conversation_id": "sess-abc",
                "messages": [
                    { "role": "user", "content": "Hello!" },
                    { "role": "assistant", "content": "Hi there!" },
                ]
            }),
        }],
    };
    let ack = manager.write_batch(&batch).expect("write_batch");
    assert_eq!(ack.written_ops, 1);

    let history = manager
        .get_conversation_history("sess-abc")
        .expect("history");
    assert_eq!(history.len(), 2);
    assert_eq!(history[0]["role"], "user");
    assert_eq!(history[0]["content"], "Hello!");
    assert_eq!(history[1]["role"], "assistant");
    assert_eq!(history[1]["content"], "Hi there!");
}

#[test]
fn augmentation_batch_entity_fact_knowledge_graph_process_attribute_conversation_update() {
    let manager = make_manager();
    manager.build().expect("build");

    manager.set_embedder(Box::new(|texts: Vec<String>| {
        texts.into_iter().map(|_| vec![1.0_f32, 0.0, 0.0]).collect()
    }));

    let batch = WriteBatch {
        ops: vec![
            WriteOp {
                op_type: "entity_fact.create".to_string(),
                payload: serde_json::json!({
                    "entity_id": "aug-entity",
                    "facts": ["User prefers dark mode", "User is a developer"],
                    "conversation_id": "aug-sess",
                }),
            },
            WriteOp {
                op_type: "knowledge_graph.create".to_string(),
                payload: serde_json::json!({
                    "entity_id": "aug-entity",
                    "semantic_triples": [
                        { "subject": "User", "predicate": "prefers", "object": "dark mode" }
                    ],
                }),
            },
            WriteOp {
                op_type: "process_attribute.create".to_string(),
                payload: serde_json::json!({
                    "process_id": "aug-process",
                    "attributes": ["code-assistant", "rust-developer"],
                }),
            },
            WriteOp {
                op_type: "conversation.update".to_string(),
                payload: serde_json::json!({
                    "conversation_id": "aug-sess",
                    "summary": "User is a developer who prefers dark mode.",
                }),
            },
        ],
    };
    let ack = manager.write_batch(&batch).expect("write_batch");
    assert_eq!(ack.written_ops, 4);

    let embeddings = manager
        .fetch_embeddings("aug-entity", 10)
        .expect("fetch_embeddings");
    assert_eq!(
        embeddings.len(),
        2,
        "two facts should be stored with embeddings"
    );
}

#[test]
fn entity_fact_blank_facts_are_filtered_before_embedding() {
    // Regression: precompute_embeddings() must filter blank/whitespace facts
    // before calling embed_texts(). The Rust embedding pipeline silently drops
    // blank inputs, so an unfiltered call returns fewer vectors than facts,
    // misaligning embeddings with the wrong facts (e.g. "likes coffee" gets no
    // embedding and the blank slot gets the "likes coffee" vector).
    let manager = make_manager();
    manager.build().expect("build");

    let received_texts: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(vec![]));
    let captured = Arc::clone(&received_texts);

    manager.set_embedder(Box::new(move |texts: Vec<String>| {
        *captured.lock().unwrap() = texts.clone();
        texts.into_iter().map(|_| vec![0.1_f32, 0.2, 0.3]).collect()
    }));

    let batch = WriteBatch {
        ops: vec![WriteOp {
            op_type: "entity_fact.create".to_string(),
            payload: serde_json::json!({
                "entity_id": "blank-test-entity",
                "facts": ["likes tea", "   ", "likes coffee"],
            }),
        }],
    };
    let ack = manager.write_batch(&batch).expect("write_batch");
    assert_eq!(ack.written_ops, 1);

    // Embedder should have been called with only the two non-blank facts.
    let texts_sent = received_texts.lock().unwrap().clone();
    assert_eq!(
        texts_sent,
        vec!["likes tea", "likes coffee"],
        "blank/whitespace facts must be filtered before embed_texts is called"
    );

    // Both non-blank facts should have been stored with their embeddings.
    let embeddings = manager
        .fetch_embeddings("blank-test-entity", 10)
        .expect("fetch_embeddings");
    assert_eq!(
        embeddings.len(),
        2,
        "only the two non-blank facts should be stored with embeddings"
    );
}
