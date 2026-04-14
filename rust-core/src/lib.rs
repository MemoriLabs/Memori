//! The central nervous system of the Memori Engine.
//!
//! This library manages two distinct execution contexts:
//! 1. A synchronous, high-performance math pipeline for local Embeddings.
//! 2. An asynchronous, heavily-bounded worker pool (`WorkerRuntime`) for background I/O tasks.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use crate::augmentation::{AugmentationInput, run_advanced_augmentation};
use crate::embeddings::{SentenceTransformersEmbedder, embed_texts};
use crate::models::{AugmentationJob, PostprocessJob};
use crate::retrieval::RetrievalRequest;
use crate::storage::{RankedFact, StorageBridge};

pub mod augmentation;
pub mod embeddings;
mod error;
mod models;
pub mod network;
pub mod retrieval;
pub mod runtime;
pub mod search;
pub mod storage;

pub use error::OrchestratorError;
pub use runtime::{FlushError, RuntimeConfig, RuntimeError, SubmitError, WorkerRuntime};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PostprocessAccepted {
    pub job_id: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AugmentationAccepted {
    pub job_id: u64,
}

/// The main application state.
///
/// We hold the `SentenceTransformersEmbedder` inside an `Arc` so its massive ML model
/// is natively tied to the lifecycle of the engine and cleanly dropped when the engine is destroyed.
#[derive(Clone)]
pub struct EngineOrchestrator {
    embedder: Arc<SentenceTransformersEmbedder>,
    postprocess_runtime: WorkerRuntime<PostprocessJob>,
    augmentation_runtime: WorkerRuntime<AugmentationJob>,
    storage_bridge: Option<Arc<dyn StorageBridge>>,
}

static NEXT_JOB_ID: AtomicU64 = AtomicU64::new(1);

impl EngineOrchestrator {
    /// Initializes the engine, downloading the specified ONNX model to the local cache if necessary.
    ///
    /// # Errors
    /// Returns `OrchestratorError::ModelError` if the model cannot be found, downloaded, or initialized.
    pub fn new(model_name: Option<&str>) -> Result<Self, OrchestratorError> {
        Self::new_with_storage(model_name, None)
    }

    pub fn new_with_storage(
        model_name: Option<&str>,
        storage_bridge: Option<Arc<dyn StorageBridge>>,
    ) -> Result<Self, OrchestratorError> {
        let embedder = SentenceTransformersEmbedder::new(model_name)
            .map_err(|e: anyhow::Error| OrchestratorError::ModelError(e.to_string()))?;

        let postprocess_runtime = init_postprocess_runtime()?;
        let augmentation_runtime = init_augmentation_runtime(storage_bridge.clone())?;

        Ok(Self {
            embedder: Arc::new(embedder),
            postprocess_runtime,
            augmentation_runtime,
            storage_bridge,
        })
    }

    /// Synchronously processes text embeddings.
    ///
    /// Note: This function prioritizes pipeline continuity. If the underlying ML model
    /// fails to evaluate a batch, it will fall back to sequential processing. If all
    /// fallbacks fail, it returns degraded zero-vectors rather than crashing the FFI process.
    pub fn embed(&self, texts: Vec<String>) -> (Vec<f32>, [usize; 2]) {
        embed_texts(&self.embedder, texts)
    }

    pub fn execute(&self, command: &str) -> Result<String, OrchestratorError> {
        execute_command(command)
    }

    pub fn hello_world(&self) -> String {
        "hello world".to_string()
    }

    pub fn postprocess_request(
        &self,
        payload: &str,
    ) -> Result<PostprocessAccepted, OrchestratorError> {
        validate_postprocess_payload(payload)?;
        let job_id = NEXT_JOB_ID.fetch_add(1, Ordering::Relaxed);
        self.postprocess_runtime
            .submit(PostprocessJob {
                job_id,
                payload: payload.to_string(),
            })
            .map_err(map_submit_error)?;
        Ok(PostprocessAccepted { job_id })
    }

    pub fn retrieve(
        &self,
        request: RetrievalRequest,
    ) -> Result<Vec<RankedFact>, OrchestratorError> {
        if request.query_text.trim().is_empty() {
            return Err(OrchestratorError::InvalidInput(
                "query_text cannot be empty".to_string(),
            ));
        }
        if request.entity_id.trim().is_empty() {
            return Err(OrchestratorError::InvalidInput(
                "entity_id cannot be empty".to_string(),
            ));
        }
        if request.dense_limit == 0 || request.limit == 0 {
            return Ok(Vec::new());
        }

        let bridge = self
            .storage_bridge
            .as_deref()
            .ok_or(OrchestratorError::StorageUnavailable)?;
        let (flat_query, shape) = self.embed(vec![request.query_text.clone()]);
        if shape[0] == 0 || shape[1] == 0 {
            return Ok(Vec::new());
        }
        let query_embedding = &flat_query[..shape[1]];
        if query_embedding.iter().all(|value| *value == 0.0) {
            return Err(OrchestratorError::ModelError(
                "failed to generate a valid query embedding".to_string(),
            ));
        }
        retrieval::run_retrieval(bridge, &request, query_embedding)
    }

    pub fn recall(&self, request: RetrievalRequest) -> Result<String, OrchestratorError> {
        let ranked = self.retrieve(request)?;
        Ok(retrieval::format_recall_output(&ranked))
    }

    pub fn submit_augmentation(
        &self,
        input: AugmentationInput,
    ) -> Result<AugmentationAccepted, OrchestratorError> {
        validate_augmentation_input(&input)?;
        let job_id = NEXT_JOB_ID.fetch_add(1, Ordering::Relaxed);
        self.augmentation_runtime
            .submit(AugmentationJob { job_id, input })
            .map_err(map_submit_error)?;
        Ok(AugmentationAccepted { job_id })
    }

    pub fn wait_for_augmentation(
        &self,
        timeout: Option<Duration>,
    ) -> Result<bool, OrchestratorError> {
        match timeout {
            Some(limit) => match self.augmentation_runtime.flush_for(limit) {
                Ok(()) => Ok(true),
                Err(FlushError::Timeout(_)) => Ok(false),
                Err(err) => Err(map_flush_error(err)),
            },
            None => self
                .augmentation_runtime
                .flush()
                .map(|()| true)
                .map_err(map_flush_error),
        }
    }
}

impl Drop for EngineOrchestrator {
    fn drop(&mut self) {
        self.postprocess_runtime.shutdown();
        self.augmentation_runtime.shutdown();
    }
}

fn init_postprocess_runtime() -> Result<WorkerRuntime<PostprocessJob>, OrchestratorError> {
    let postprocess_runtime = WorkerRuntime::new(
        RuntimeConfig {
            queue_capacity: 512,
            max_concurrency: 2,
            worker_threads: Some(1),
            ..Default::default()
        },
        |job: PostprocessJob| async move {
            log::info!(
                "[orchestrator postprocess worker] job {} accepted",
                job.job_id
            );
            tokio::time::sleep(Duration::from_millis(35)).await;
            log::info!(
                "[orchestrator postprocess worker] job {} processed payload ({} bytes)",
                job.job_id,
                job.payload.len()
            );
        },
    )
    .map_err(|e| OrchestratorError::BackgroundUnavailable(e.to_string()))?;

    postprocess_runtime
        .start()
        .map_err(|e| OrchestratorError::BackgroundUnavailable(e.to_string()))?;

    Ok(postprocess_runtime)
}

fn init_augmentation_runtime(
    storage_bridge: Option<Arc<dyn StorageBridge>>,
) -> Result<WorkerRuntime<AugmentationJob>, OrchestratorError> {
    let augmentation_runtime = WorkerRuntime::new(
        RuntimeConfig {
            queue_capacity: 512,
            max_concurrency: 2,
            worker_threads: Some(1),
            ..Default::default()
        },
        move |job: AugmentationJob| {
            let bridge = storage_bridge.clone();
            async move {
                match run_advanced_augmentation(&job.input).await {
                    Ok(batch) => match bridge {
                        Some(storage) => {
                            if let Err(error) = storage.write_batch(&batch) {
                                log::error!(
                                    "[orchestrator augmentation worker] job {} write failed: {}",
                                    job.job_id, error
                                );
                            } else {
                                log::info!(
                                    "[orchestrator augmentation worker] job {} persisted {} op(s)",
                                    job.job_id,
                                    batch.ops.len()
                                );
                            }
                        }
                        None => {
                            log::warn!(
                                "[orchestrator augmentation worker] job {} ignored: no storage bridge",
                                job.job_id
                            );
                        }
                    },
                    Err(error) => {
                        log::error!(
                            "[orchestrator augmentation worker] job {} augmentation failed: {}",
                            job.job_id, error
                        );
                    }
                };
            }
        },
    )
    .map_err(|e| OrchestratorError::BackgroundUnavailable(e.to_string()))?;

    augmentation_runtime
        .start()
        .map_err(|e| OrchestratorError::BackgroundUnavailable(e.to_string()))?;

    Ok(augmentation_runtime)
}

fn map_submit_error<T>(err: SubmitError<T>) -> OrchestratorError {
    match err {
        SubmitError::NotRunning | SubmitError::ShuttingDown | SubmitError::Stopped => {
            OrchestratorError::BackgroundUnavailable("runtime is not accepting jobs".to_string())
        }
        SubmitError::QueueFull(_job) => OrchestratorError::QueueFull,
    }
}

fn map_flush_error(err: FlushError) -> OrchestratorError {
    match err {
        FlushError::Timeout(limit) => {
            OrchestratorError::BackgroundUnavailable(format!("timed out after {limit:?}"))
        }
        FlushError::NotRunning => {
            OrchestratorError::BackgroundUnavailable("runtime is not running".to_string())
        }
    }
}

fn execute_command(command: &str) -> Result<String, OrchestratorError> {
    let trimmed = command.trim();
    if trimmed.is_empty() {
        return Err(OrchestratorError::InvalidInput(
            "command cannot be empty".to_string(),
        ));
    }

    match trimmed {
        "ping" => Ok("pong".to_string()),
        other => Err(OrchestratorError::UnsupportedCommand(other.to_string())),
    }
}

fn validate_postprocess_payload(payload: &str) -> Result<(), OrchestratorError> {
    if payload.trim().is_empty() {
        return Err(OrchestratorError::InvalidInput(
            "postprocess payload cannot be empty".to_string(),
        ));
    }

    Ok(())
}

fn validate_augmentation_input(input: &AugmentationInput) -> Result<(), OrchestratorError> {
    if input.entity_id.trim().is_empty() {
        return Err(OrchestratorError::InvalidInput(
            "augmentation entity_id cannot be empty".to_string(),
        ));
    }
    let has_message = input
        .conversation_messages
        .iter()
        .any(|m| !m.content.trim().is_empty() && !m.role.trim().is_empty());
    let has_content = input
        .content
        .as_ref()
        .map(|s| !s.trim().is_empty())
        .unwrap_or(false);
    if !has_message && !has_content {
        return Err(OrchestratorError::InvalidInput(
            "augmentation requires conversation_messages or content".to_string(),
        ));
    }
    Ok(())
}
