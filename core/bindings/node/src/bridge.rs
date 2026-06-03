use dashmap::DashMap;
use engine_orchestrator::storage::{
    ConnectionFactory, HostStorageError, SqlBind, StorageConnection,
};
use napi::threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;
use tokio::sync::oneshot;
use tokio::time::timeout;

const JS_CALLBACK_TIMEOUT: Duration = Duration::from_secs(30);

pub type PendingStorageMap = Arc<DashMap<u32, oneshot::Sender<serde_json::Value>>>;

/// Shared TSFN state cloned into every [`NodeConnection`] that the factory produces.
struct Inner {
    storage_call_tsfn: Mutex<Option<ThreadsafeFunction<(u32, String)>>>,
    pending: PendingStorageMap,
    next_id: AtomicU32,
}

impl Inner {
    /// Fire-and-forget: sends `close` to TS without waiting for acknowledgement.
    /// Used by [`NodeConnection::close`] since close errors are non-fatal and blocking
    /// a Tokio worker thread for a pool-return operation is unnecessary.
    fn send_close(&self, conn_id: u32) {
        let Ok(payload_str) = serde_json::to_string(&serde_json::json!({
            "op": "close",
            "conn_id": conn_id,
        })) else {
            return;
        };
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        // No pending entry — TS will call resolveStorageCall, which silently discards
        // the result when the id is absent from the map.
        if let Some(tsfn) = self
            .storage_call_tsfn
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
        {
            let _ = tsfn.call(
                Ok((id, payload_str)),
                ThreadsafeFunctionCallMode::NonBlocking,
            );
        }
    }

    fn call(&self, payload: serde_json::Value) -> Result<serde_json::Value, HostStorageError> {
        let payload_str = serde_json::to_string(&payload)
            .map_err(|e| HostStorageError::new("JSON_ERR", e.to_string()))?;
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = oneshot::channel();
        self.pending.insert(id, tx);

        let status = {
            if let Some(tsfn) = self
                .storage_call_tsfn
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .as_ref()
            {
                tsfn.call(
                    Ok((id, payload_str)),
                    ThreadsafeFunctionCallMode::NonBlocking,
                )
            } else {
                napi::Status::Closing
            }
        };

        if status != napi::Status::Ok {
            self.pending.remove(&id);
            return Err(HostStorageError::new(
                "NAPI_ERR",
                "failed to queue JS storage callback",
            ));
        }

        // block_in_place lets Tokio know this thread is about to block, keeping the scheduler
        // correct even if Inner::call is ever reached from a regular async worker thread.
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                match timeout(JS_CALLBACK_TIMEOUT, rx).await {
                    Ok(Ok(value)) => {
                        if let Some(err) = value.get("error") {
                            let code = err["code"].as_str().unwrap_or("ERR").to_string();
                            let msg = err["message"]
                                .as_str()
                                .unwrap_or("unknown error")
                                .to_string();
                            Err(HostStorageError::new(code, msg))
                        } else {
                            Ok(value)
                        }
                    }
                    Ok(Err(_)) => Err(HostStorageError::new("NAPI_ERR", "storage channel dropped")),
                    Err(_) => {
                        self.pending.remove(&id);
                        Err(HostStorageError::new(
                            "TIMEOUT",
                            "storage JS callback did not respond within 30s",
                        ))
                    }
                }
            })
        })
    }
}

/// Implements [`ConnectionFactory`] by delegating to a single TypeScript
/// `storageCall(id, payloadJson)` ThreadsafeFunction.
///
/// Protocol — all payloads/results are JSON strings:
///   acquire  → `{ "op": "acquire" }`                                   → `{ "conn_id": N }`
///   execute  → `{ "op": "execute", "conn_id": N, "sql": "…", "binds": […] }` → `{ "rows": […] }`
///   begin    → `{ "op": "begin",   "conn_id": N }`                     → `{ "ok": true }`
///   commit   → `{ "op": "commit",  "conn_id": N }`                     → `{ "ok": true }`
///   rollback → `{ "op": "rollback","conn_id": N }`                     → `{ "ok": true }`
///   close    → `{ "op": "close",   "conn_id": N }`                     → `{ "ok": true }`
///
///   On any error: `{ "error": { "code": "…", "message": "…" } }`
///
/// TS resolves each call via `engine.resolveStorageCall(id, resultJson)`.
pub struct NodeConnectionFactory {
    inner: Arc<Inner>,
    dialect_str: String,
}

impl NodeConnectionFactory {
    pub fn new(storage_call_tsfn: ThreadsafeFunction<(u32, String)>, dialect_str: String) -> Self {
        Self {
            inner: Arc::new(Inner {
                storage_call_tsfn: Mutex::new(Some(storage_call_tsfn)),
                pending: Arc::new(DashMap::new()),
                next_id: AtomicU32::new(1),
            }),
            dialect_str,
        }
    }

    /// Called by `engine.resolveStorageCall` to unblock the waiting Rust thread.
    pub fn resolve(&self, id: u32, result_json: String) {
        if let Some((_, tx)) = self.inner.pending.remove(&id) {
            let value: serde_json::Value = serde_json::from_str(&result_json).unwrap_or(
                serde_json::json!({ "error": { "code": "JSON_ERR", "message": "invalid JSON from TS" } }),
            );
            let _ = tx.send(value);
        }
    }
}

impl ConnectionFactory for NodeConnectionFactory {
    fn acquire(&self) -> Result<Box<dyn StorageConnection>, HostStorageError> {
        let result = self.inner.call(serde_json::json!({ "op": "acquire" }))?;
        let conn_id = result["conn_id"]
            .as_u64()
            .map(|n| n as u32)
            .ok_or_else(|| HostStorageError::new("NAPI_ERR", "acquire returned no conn_id"))?;
        Ok(Box::new(NodeConnection {
            conn_id,
            inner: self.inner.clone(),
        }))
    }

    fn dialect(&self) -> &str {
        &self.dialect_str
    }

    fn shutdown(&self) {
        // Drop the TSFN first so no new calls can be queued.
        let _ = self
            .inner
            .storage_call_tsfn
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .take();
        // Unblock any Rust threads waiting in block_in_place for a JS callback.
        // Without this they would hang until the 30-second timeout fires per call.
        let keys: Vec<u32> = self.inner.pending.iter().map(|e| *e.key()).collect();
        for key in keys {
            if let Some((_, tx)) = self.inner.pending.remove(&key) {
                let _ = tx.send(serde_json::json!({
                    "error": { "code": "SHUTDOWN", "message": "engine is shutting down" }
                }));
            }
        }
    }
}

/// A single checked-out connection. Calls TS for every SQL operation.
pub struct NodeConnection {
    conn_id: u32,
    inner: Arc<Inner>,
}

impl StorageConnection for NodeConnection {
    fn execute(
        &self,
        sql: &str,
        binds: Vec<SqlBind>,
    ) -> Result<Vec<serde_json::Value>, HostStorageError> {
        let result = self.inner.call(serde_json::json!({
            "op": "execute",
            "conn_id": self.conn_id,
            "sql": sql,
            "binds": binds,
        }))?;
        result["rows"]
            .as_array()
            .cloned()
            .ok_or_else(|| HostStorageError::new("NAPI_ERR", "execute response missing rows array"))
    }

    fn begin(&self) -> Result<(), HostStorageError> {
        self.inner
            .call(serde_json::json!({ "op": "begin", "conn_id": self.conn_id }))?;
        Ok(())
    }

    fn commit(&self) -> Result<(), HostStorageError> {
        self.inner
            .call(serde_json::json!({ "op": "commit", "conn_id": self.conn_id }))?;
        Ok(())
    }

    fn rollback(&self) -> Result<(), HostStorageError> {
        self.inner
            .call(serde_json::json!({ "op": "rollback", "conn_id": self.conn_id }))?;
        Ok(())
    }

    fn close(&self) {
        self.inner.send_close(self.conn_id);
    }
}
