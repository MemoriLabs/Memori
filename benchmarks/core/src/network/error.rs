use reqwest::StatusCode;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ApiError {
    #[error("Validation error (422): {message}")]
    Validation {
        message: String,
        details: Option<serde_json::Value>,
    },
    #[error("Request rejected (433): {message}")]
    Rejected {
        message: String,
        details: Option<serde_json::Value>,
    },
    #[error("Quota exceeded (429): {0}")]
    QuotaExceeded(String),
    #[error("Client error ({status_code}): {message}")]
    Client {
        status_code: StatusCode,
        message: String,
        details: Option<serde_json::Value>,
    },
    #[error(
        "Memori API request failed due to an SSL/TLS certificate error. This is often caused by corporate proxies/SSL inspection. Try updating your CA certificates. Details: {0}"
    )]
    Ssl(String),
    #[error("Network or timeout error: {0}")]
    Network(String),
    #[error("Configuration error: {0}")]
    Config(String),
}

impl PartialEq for ApiError {
    fn eq(&self, other: &Self) -> bool {
        self.to_string() == other.to_string()
    }
}

impl Eq for ApiError {}
