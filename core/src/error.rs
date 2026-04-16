use crate::network::ApiError;
use crate::storage::HostStorageError;
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum OrchestratorError {
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("command not supported: {0}")]
    UnsupportedCommand(String),
    #[error("postprocess queue is full")]
    QueueFull,
    #[error("background runtime unavailable: {0}")]
    BackgroundUnavailable(String),
    #[error("model error: {0}")]
    ModelError(String),
    #[error(transparent)]
    ApiError(#[from] ApiError),
    #[error("storage bridge unavailable")]
    StorageUnavailable,
    #[error("storage bridge error: {0}")]
    StorageBridge(HostStorageError),
}

impl OrchestratorError {
    pub fn status_code(&self) -> u32 {
        match self {
            Self::InvalidInput(_) => 1,
            Self::UnsupportedCommand(_) => 2,
            Self::QueueFull => 3,
            Self::BackgroundUnavailable(_) => 4,
            Self::ModelError(_) => 5,
            Self::ApiError(_) => 6,
            Self::StorageUnavailable => 7,
            Self::StorageBridge(_) => 8,
        }
    }
}
