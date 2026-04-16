use std::time::Duration;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum SubmitError<J> {
    #[error("runtime is not started")]
    NotRunning,
    #[error("runtime is shutting down")]
    ShuttingDown,
    #[error("runtime has stopped")]
    Stopped,
    #[error("job queue is full")]
    QueueFull(J),
}

#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error("runtime was already started")]
    AlreadyStarted,
    #[error("invalid configuration: {0}")]
    InvalidConfig(&'static str),
    #[error("failed to build async runtime")]
    RuntimeBuild(#[from] std::io::Error),
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum FlushError {
    #[error("timed out after {0:?} waiting for jobs to complete")]
    Timeout(Duration),
    #[error("runtime is not running")]
    NotRunning,
}
