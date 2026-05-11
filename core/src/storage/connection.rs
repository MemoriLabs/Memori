use crate::storage::models::HostStorageError;
use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde::{Deserialize, Serialize};

mod string_i64 {
    use serde::{self, Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(val: &i64, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&val.to_string())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<i64, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        s.parse::<i64>().map_err(serde::de::Error::custom)
    }
}

/// A single bind parameter for a parameterised SQL statement.
///
/// `Bytes` are base64-encoded for JSON transport across the NAPI/PyO3 boundary;
/// the host-side adapter decodes them back to a native buffer before execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "t", content = "v")]
pub enum SqlBind {
    #[serde(rename = "null")]
    Null,
    #[serde(rename = "int", with = "string_i64")]
    Int(i64),
    #[serde(rename = "float")]
    Float(f64),
    #[serde(rename = "text")]
    Text(String),
    /// Raw bytes (e.g. embedding vectors), base64-encoded for JSON transport.
    #[serde(rename = "bytes")]
    Bytes(String),
}

impl SqlBind {
    /// Encode raw bytes as a base64 `Bytes` bind value.
    pub fn bytes(data: &[u8]) -> Self {
        Self::Bytes(STANDARD.encode(data))
    }

    pub fn id_or_null(id: Option<i64>) -> Self {
        match id {
            Some(n) => Self::Int(n),
            None => Self::Null,
        }
    }

    pub fn text_or_null(s: Option<&str>) -> Self {
        match s {
            Some(v) => Self::Text(v.to_string()),
            None => Self::Null,
        }
    }
}

/// A single live connection to the user's database pool.
///
/// Acquired via [`ConnectionFactory::acquire`]. Caller is responsible for
/// calling [`StorageConnection::close`] when finished — this releases the
/// connection back to the pool rather than destroying it.
pub trait StorageConnection: Send + Sync {
    fn execute(
        &self,
        sql: &str,
        binds: Vec<SqlBind>,
    ) -> Result<Vec<serde_json::Value>, HostStorageError>;
    fn begin(&self) -> Result<(), HostStorageError>;
    fn commit(&self) -> Result<(), HostStorageError>;
    fn rollback(&self) -> Result<(), HostStorageError>;
    fn close(&self);
}

/// Creates on-demand connections backed by the user's connection pool.
///
/// Each call to [`acquire`] checks out one connection. Callers must close it
/// when done so the pool can reclaim it. No connection is held between calls —
/// matching the Python `connection_context` pattern.
pub trait ConnectionFactory: Send + Sync {
    fn acquire(&self) -> Result<Box<dyn StorageConnection>, HostStorageError>;
    /// The SQL dialect detected from the user's connection (e.g. `"sqlite"`).
    fn dialect(&self) -> &str;
    fn shutdown(&self) {}
}

/// Parses a DB row's `id` field to an `i64`, tolerating both integer and string JSON values.
pub fn read_id(row: &serde_json::Value, field: &str) -> Option<i64> {
    let v = &row[field];
    v.as_i64()
        .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
        .or_else(|| v.as_f64().map(|f| f as i64))
}
