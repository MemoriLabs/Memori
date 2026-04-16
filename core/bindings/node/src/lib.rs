//! napi-rs bindings over [`engine_orchestrator::EngineOrchestrator`].
//!
//! This crate is a thin adapter: it deserialises JSON payloads coming from the
//! Node SDK into engine types, invokes the engine, and serialises the result
//! back out. All business logic lives in the root `engine-orchestrator` crate.

#![forbid(unsafe_code)]

use std::sync::Arc;
use std::sync::mpsc;
use std::time::Duration;

use engine_orchestrator::augmentation::AugmentationInput;
use engine_orchestrator::retrieval::RetrievalRequest;
use engine_orchestrator::search::FactId;
use engine_orchestrator::storage::{
    CandidateFactRow, EmbeddingRow, FetchEmbeddingsRequest, FetchFactsByIdsRequest,
    HostStorageError, StorageBridge, WriteAck, WriteBatch,
};
use engine_orchestrator::{EngineOrchestrator, OrchestratorError};
use napi::threadsafe_function::{ErrorStrategy, ThreadsafeFunction, ThreadsafeFunctionCallMode};
use napi::{Error, JsFunction, Result, Status};
use napi_derive::napi;

struct NodeStorageBridge {
    fetch_embeddings_cb: ThreadsafeFunction<String, ErrorStrategy::Fatal>,
    fetch_facts_by_ids_cb: ThreadsafeFunction<String, ErrorStrategy::Fatal>,
    write_batch_cb: ThreadsafeFunction<String, ErrorStrategy::Fatal>,
}

impl NodeStorageBridge {
    const CALLBACK_TIMEOUT: Duration = Duration::from_secs(30);

    fn call_json(
        callback: &ThreadsafeFunction<String, ErrorStrategy::Fatal>,
        payload_json: String,
    ) -> std::result::Result<String, HostStorageError> {
        let (tx, rx) = mpsc::sync_channel::<String>(1);
        let status = callback.call_with_return_value(
            payload_json,
            ThreadsafeFunctionCallMode::Blocking,
            move |value: String| {
                let _ = tx.send(value);
                Ok(())
            },
        );
        if status != Status::Ok {
            return Err(HostStorageError::new(
                "node_callback_failed",
                format!("callback status: {status:?}"),
            ));
        }
        rx.recv_timeout(Self::CALLBACK_TIMEOUT)
            .map_err(|e| match e {
                mpsc::RecvTimeoutError::Timeout => HostStorageError::new(
                    "node_callback_timeout",
                    format!(
                        "callback did not return within {}s",
                        Self::CALLBACK_TIMEOUT.as_secs()
                    ),
                ),
                mpsc::RecvTimeoutError::Disconnected => {
                    HostStorageError::new("node_callback_channel_closed", "callback channel closed")
                }
            })
    }
}

impl StorageBridge for NodeStorageBridge {
    fn fetch_embeddings(
        &self,
        entity_id: &str,
        limit: usize,
    ) -> std::result::Result<Vec<EmbeddingRow>, HostStorageError> {
        let request = FetchEmbeddingsRequest {
            entity_id: entity_id.to_string(),
            limit,
        };
        let payload = serde_json::to_string(&request)
            .map_err(|e| HostStorageError::new("serialization_error", e.to_string()))?;
        let result = Self::call_json(&self.fetch_embeddings_cb, payload)?;
        serde_json::from_str::<Vec<EmbeddingRow>>(&result)
            .map_err(|e| HostStorageError::new("deserialization_error", e.to_string()))
    }

    fn fetch_facts_by_ids(
        &self,
        ids: &[FactId],
    ) -> std::result::Result<Vec<CandidateFactRow>, HostStorageError> {
        let request = FetchFactsByIdsRequest { ids: ids.to_vec() };
        let payload = serde_json::to_string(&request)
            .map_err(|e| HostStorageError::new("serialization_error", e.to_string()))?;
        let result = Self::call_json(&self.fetch_facts_by_ids_cb, payload)?;
        serde_json::from_str::<Vec<CandidateFactRow>>(&result)
            .map_err(|e| HostStorageError::new("deserialization_error", e.to_string()))
    }

    fn write_batch(&self, batch: &WriteBatch) -> std::result::Result<WriteAck, HostStorageError> {
        let payload = serde_json::to_string(batch)
            .map_err(|e| HostStorageError::new("serialization_error", e.to_string()))?;
        let result = Self::call_json(&self.write_batch_cb, payload)?;
        serde_json::from_str::<WriteAck>(&result)
            .map_err(|e| HostStorageError::new("deserialization_error", e.to_string()))
    }
}

#[napi]
pub struct MemoriEngine {
    inner: EngineOrchestrator,
}

#[napi]
impl MemoriEngine {
    #[napi(constructor)]
    pub fn new(model_name: Option<String>) -> Result<Self> {
        let inner = EngineOrchestrator::new(model_name.as_deref())
            .map_err(orchestrator_error_to_napi_error)?;
        Ok(Self { inner })
    }

    #[napi]
    pub fn execute(&self, command: String) -> Result<String> {
        self.inner
            .execute(&command)
            .map_err(orchestrator_error_to_napi_error)
    }

    #[napi]
    pub fn hello_world(&self) -> String {
        self.inner.hello_world()
    }

    #[napi]
    pub fn core_postprocess_request(&self, payload: String) -> Result<String> {
        self.inner
            .postprocess_request(&payload)
            .map(|accepted| accepted.job_id.to_string())
            .map_err(orchestrator_error_to_napi_error)
    }
}

#[napi]
pub struct EngineHandle {
    orchestrator: EngineOrchestrator,
}

#[napi]
impl EngineHandle {
    #[napi(constructor)]
    pub fn new(
        model_name: Option<String>,
        fetch_embeddings_cb: JsFunction,
        fetch_facts_by_ids_cb: JsFunction,
        write_batch_cb: JsFunction,
    ) -> Result<Self> {
        let fetch_embeddings_tsfn = fetch_embeddings_cb
            .create_threadsafe_function::<String, String, _, ErrorStrategy::Fatal>(0, |ctx| {
                Ok(vec![ctx.value])
            })
            .map_err(|e| Error::new(Status::GenericFailure, e.to_string()))?;

        let fetch_facts_tsfn = fetch_facts_by_ids_cb
            .create_threadsafe_function::<String, String, _, ErrorStrategy::Fatal>(0, |ctx| {
                Ok(vec![ctx.value])
            })
            .map_err(|e| Error::new(Status::GenericFailure, e.to_string()))?;

        let write_batch_tsfn = write_batch_cb
            .create_threadsafe_function::<String, String, _, ErrorStrategy::Fatal>(0, |ctx| {
                Ok(vec![ctx.value])
            })
            .map_err(|e| Error::new(Status::GenericFailure, e.to_string()))?;

        let bridge = NodeStorageBridge {
            fetch_embeddings_cb: fetch_embeddings_tsfn,
            fetch_facts_by_ids_cb: fetch_facts_tsfn,
            write_batch_cb: write_batch_tsfn,
        };

        let orchestrator =
            EngineOrchestrator::new_with_storage(model_name.as_deref(), Some(Arc::new(bridge)))
                .map_err(orchestrator_error_to_napi_error)?;
        Ok(Self { orchestrator })
    }

    #[napi]
    pub fn execute(&self, command: String) -> Result<String> {
        self.orchestrator
            .execute(&command)
            .map_err(orchestrator_error_to_napi_error)
    }

    #[napi]
    pub fn hello_world(&self) -> Result<String> {
        Ok(self.orchestrator.hello_world())
    }

    #[napi]
    pub fn core_postprocess_request(&self, payload: String) -> Result<String> {
        self.orchestrator
            .postprocess_request(&payload)
            .map(|accepted| accepted.job_id.to_string())
            .map_err(orchestrator_error_to_napi_error)
    }

    #[napi]
    pub fn retrieve(&self, request_json: String) -> Result<String> {
        let request: RetrievalRequest = serde_json::from_str(&request_json)
            .map_err(|e| Error::new(Status::InvalidArg, e.to_string()))?;
        let ranked = self
            .orchestrator
            .retrieve(request)
            .map_err(orchestrator_error_to_napi_error)?;
        serde_json::to_string(&ranked)
            .map_err(|e| Error::new(Status::GenericFailure, e.to_string()))
    }

    #[napi]
    pub fn recall(&self, request_json: String) -> Result<String> {
        let request: RetrievalRequest = serde_json::from_str(&request_json)
            .map_err(|e| Error::new(Status::InvalidArg, e.to_string()))?;
        self.orchestrator
            .recall(request)
            .map_err(orchestrator_error_to_napi_error)
    }

    #[napi]
    pub fn submit_augmentation(&self, input_json: String) -> Result<String> {
        let input: AugmentationInput = serde_json::from_str(&input_json)
            .map_err(|e| Error::new(Status::InvalidArg, e.to_string()))?;
        self.orchestrator
            .submit_augmentation(input)
            .map(|accepted| accepted.job_id.to_string())
            .map_err(orchestrator_error_to_napi_error)
    }
}

#[napi]
pub fn execute(command: String) -> Result<String> {
    let orchestrator = EngineOrchestrator::new(None).map_err(orchestrator_error_to_napi_error)?;
    orchestrator
        .execute(&command)
        .map_err(orchestrator_error_to_napi_error)
}

#[napi]
pub fn hello_world() -> Result<String> {
    let orchestrator = EngineOrchestrator::new(None).map_err(orchestrator_error_to_napi_error)?;
    Ok(orchestrator.hello_world())
}

#[napi]
pub fn core_postprocess_request(payload: String) -> Result<String> {
    let orchestrator = EngineOrchestrator::new(None).map_err(orchestrator_error_to_napi_error)?;
    orchestrator
        .postprocess_request(&payload)
        .map(|accepted| accepted.job_id.to_string())
        .map_err(orchestrator_error_to_napi_error)
}

fn orchestrator_error_to_napi_error(error: OrchestratorError) -> Error {
    let status = match error.status_code() {
        1 | 2 => Status::InvalidArg,
        3 => Status::QueueFull,
        _ => Status::GenericFailure,
    };
    Error::new(status, error.to_string())
}
