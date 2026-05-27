//! PyO3 bindings over [`engine_orchestrator::EngineOrchestrator`].
//!
//! This crate is a thin adapter: it deserialises JSON payloads coming from the
//! Python SDK into engine types, invokes the engine, and serialises the result
//! back out. All business logic lives in the root `engine-orchestrator` crate.

#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::mpsc;
use std::time::Duration;

use engine_orchestrator::augmentation::AugmentationInput;
use engine_orchestrator::retrieval::RetrievalRequest;
use engine_orchestrator::search::FactId;
use engine_orchestrator::storage::{
    CandidateFactRow, EmbeddingRow, FetchEmbeddingsRequest, FetchFactsByIdsRequest,
    HostStorageError, StorageBridge, WriteAck, WriteBatch,
};
use engine_orchestrator::{
    EmbeddingEngine as CoreEmbeddingEngine, EngineOrchestrator, OrchestratorError,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;

struct PythonStorageBridge {
    fetch_embeddings_cb: Py<PyAny>,
    fetch_facts_by_ids_cb: Py<PyAny>,
    write_batch_cb: Py<PyAny>,
}

impl PythonStorageBridge {
    const CALLBACK_TIMEOUT: Duration = Duration::from_secs(30);

    fn call_json_callback(
        callback: &Py<PyAny>,
        payload_json: String,
    ) -> Result<String, HostStorageError> {
        let callback: Py<PyAny> = Python::attach(|py| callback.clone_ref(py));
        let (tx, rx) = mpsc::sync_channel(1);
        std::thread::spawn(move || {
            let result = Python::attach(|py| {
                let value = callback
                    .call1(py, (payload_json,))
                    .map_err(|e| HostStorageError::new("python_callback_failed", e.to_string()))?;
                value
                    .extract::<String>(py)
                    .map_err(|e| HostStorageError::new("python_callback_bad_return", e.to_string()))
            });
            let _ = tx.send(result);
        });
        match rx.recv_timeout(Self::CALLBACK_TIMEOUT) {
            Ok(result) => result,
            Err(mpsc::RecvTimeoutError::Timeout) => Err(HostStorageError::new(
                "python_callback_timeout",
                format!(
                    "callback did not return within {}s",
                    Self::CALLBACK_TIMEOUT.as_secs()
                ),
            )),
            Err(mpsc::RecvTimeoutError::Disconnected) => Err(HostStorageError::new(
                "python_callback_channel_closed",
                "callback channel closed",
            )),
        }
    }
}

impl StorageBridge for PythonStorageBridge {
    fn fetch_embeddings(
        &self,
        entity_id: &str,
        limit: usize,
    ) -> Result<Vec<EmbeddingRow>, HostStorageError> {
        let request = FetchEmbeddingsRequest {
            entity_id: entity_id.to_string(),
            limit,
        };
        let payload = serde_json::to_string(&request)
            .map_err(|e| HostStorageError::new("serialization_error", e.to_string()))?;
        let result = Self::call_json_callback(&self.fetch_embeddings_cb, payload)?;
        serde_json::from_str::<Vec<EmbeddingRow>>(&result)
            .map_err(|e| HostStorageError::new("deserialization_error", e.to_string()))
    }

    fn fetch_facts_by_ids(
        &self,
        ids: &[FactId],
    ) -> Result<Vec<CandidateFactRow>, HostStorageError> {
        let request = FetchFactsByIdsRequest { ids: ids.to_vec() };
        let payload = serde_json::to_string(&request)
            .map_err(|e| HostStorageError::new("serialization_error", e.to_string()))?;
        let result = Self::call_json_callback(&self.fetch_facts_by_ids_cb, payload)?;
        serde_json::from_str::<Vec<CandidateFactRow>>(&result)
            .map_err(|e| HostStorageError::new("deserialization_error", e.to_string()))
    }

    fn write_batch(&self, batch: &WriteBatch) -> Result<WriteAck, HostStorageError> {
        let payload = serde_json::to_string(batch)
            .map_err(|e| HostStorageError::new("serialization_error", e.to_string()))?;
        let result = Self::call_json_callback(&self.write_batch_cb, payload)?;
        serde_json::from_str::<WriteAck>(&result)
            .map_err(|e| HostStorageError::new("deserialization_error", e.to_string()))
    }
}

#[pyclass]
pub struct MemoriEngine {
    inner: EngineOrchestrator,
}

#[pymethods]
impl MemoriEngine {
    #[new]
    #[pyo3(signature = (model_name=None))]
    fn new(model_name: Option<&str>) -> PyResult<Self> {
        let inner = EngineOrchestrator::new(model_name)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn execute(&self, command: &str) -> PyResult<String> {
        self.inner
            .execute(command)
            .map_err(orchestrator_error_to_py_err)
    }

    fn hello_world(&self) -> String {
        self.inner.hello_world()
    }

    fn core_postprocess_request(&self, payload: &str) -> PyResult<u64> {
        self.inner
            .postprocess_request(payload)
            .map(|accepted| accepted.job_id)
            .map_err(orchestrator_error_to_py_err)
    }

    fn embed_texts(&self, py: Python<'_>, texts: Vec<String>) -> Vec<Vec<f32>> {
        reshape_embedding_result(py.detach(|| self.inner.embed(texts)))
    }
}

#[pyclass]
struct NativeEmbedder {
    inner: CoreEmbeddingEngine,
}

#[pymethods]
impl NativeEmbedder {
    #[new]
    #[pyo3(signature = (model_name=None))]
    fn new(model_name: Option<&str>) -> PyResult<Self> {
        let inner = CoreEmbeddingEngine::new(model_name)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn embed_texts(&self, py: Python<'_>, texts: Vec<String>) -> Vec<Vec<f32>> {
        reshape_embedding_result(py.detach(|| self.inner.embed(texts)))
    }
}

#[pyclass]
struct EngineHandle {
    orchestrator: EngineOrchestrator,
}

#[pymethods]
impl EngineHandle {
    #[new]
    fn new(
        model_name: Option<String>,
        fetch_embeddings_cb: Py<PyAny>,
        fetch_facts_by_ids_cb: Py<PyAny>,
        write_batch_cb: Py<PyAny>,
    ) -> PyResult<Self> {
        let bridge = PythonStorageBridge {
            fetch_embeddings_cb,
            fetch_facts_by_ids_cb,
            write_batch_cb,
        };
        let orchestrator =
            EngineOrchestrator::new_with_storage(model_name.as_deref(), Some(Arc::new(bridge)))
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { orchestrator })
    }

    fn execute(&self, command: &str) -> PyResult<String> {
        self.orchestrator
            .execute(command)
            .map_err(orchestrator_error_to_py_err)
    }

    fn hello_world(&self) -> String {
        self.orchestrator.hello_world()
    }

    fn core_postprocess_request(&self, payload: &str) -> PyResult<u64> {
        self.orchestrator
            .postprocess_request(payload)
            .map(|accepted| accepted.job_id)
            .map_err(orchestrator_error_to_py_err)
    }

    fn embed_texts(&self, py: Python<'_>, texts: Vec<String>) -> Vec<Vec<f32>> {
        reshape_embedding_result(py.detach(|| self.orchestrator.embed(texts)))
    }

    fn retrieve(&self, py: Python<'_>, request_json: &str) -> PyResult<String> {
        let request: RetrievalRequest =
            serde_json::from_str(request_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ranked = py
            .detach(|| self.orchestrator.retrieve(request))
            .map_err(orchestrator_error_to_py_err)?;
        serde_json::to_string(&ranked).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn recall(&self, py: Python<'_>, request_json: &str) -> PyResult<String> {
        let request: RetrievalRequest =
            serde_json::from_str(request_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
        py.detach(|| self.orchestrator.recall(request))
            .map_err(orchestrator_error_to_py_err)
    }

    fn submit_augmentation(&self, input_json: &str) -> PyResult<u64> {
        let input: AugmentationInput =
            serde_json::from_str(input_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
        self.orchestrator
            .submit_augmentation(input)
            .map(|accepted| accepted.job_id)
            .map_err(orchestrator_error_to_py_err)
    }

    #[pyo3(signature = (timeout_ms=None))]
    fn wait_for_augmentation(&self, py: Python<'_>, timeout_ms: Option<u64>) -> PyResult<bool> {
        let timeout = timeout_ms.map(Duration::from_millis);
        py.detach(|| self.orchestrator.wait_for_augmentation(timeout))
            .map_err(orchestrator_error_to_py_err)
    }
}

#[pyfunction]
fn execute(command: &str) -> PyResult<String> {
    let orchestrator =
        EngineOrchestrator::new(None).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    orchestrator
        .execute(command)
        .map_err(orchestrator_error_to_py_err)
}

#[pyfunction]
fn hello_world() -> PyResult<String> {
    let orchestrator =
        EngineOrchestrator::new(None).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(orchestrator.hello_world())
}

#[pyfunction]
fn core_postprocess_request(payload: &str) -> PyResult<u64> {
    let orchestrator =
        EngineOrchestrator::new(None).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    orchestrator
        .postprocess_request(payload)
        .map(|accepted| accepted.job_id)
        .map_err(orchestrator_error_to_py_err)
}

#[pyfunction]
#[pyo3(signature = (texts, model_name=None))]
fn embed_texts(
    py: Python<'_>,
    texts: Vec<String>,
    model_name: Option<&str>,
) -> PyResult<Vec<Vec<f32>>> {
    let model_name = model_name.map(str::to_owned);
    let result: Result<Vec<Vec<f32>>, OrchestratorError> = py.detach(move || {
        let engine = cached_embedding_engine(model_name.as_deref())?;
        Ok(reshape_embedding_result(engine.embed(texts)))
    });
    result.map_err(orchestrator_error_to_py_err)
}

fn cached_embedding_engine(model_name: Option<&str>) -> Result<CoreEmbeddingEngine, OrchestratorError> {
    static CACHE: Mutex<Option<HashMap<Option<String>, CoreEmbeddingEngine>>> = Mutex::new(None);

    let key = model_name.map(str::to_owned);
    let mut guard = CACHE
        .lock()
        .map_err(|_| OrchestratorError::ModelError("embedding engine cache lock poisoned".into()))?;
    let cache = guard.get_or_insert_with(HashMap::new);
    if !cache.contains_key(&key) {
        let engine = CoreEmbeddingEngine::new(model_name)?;
        cache.insert(key.clone(), engine);
    }
    Ok(cache.get(&key).expect("embedding engine cache entry").clone())
}

fn reshape_embedding_result((flat_vectors, shape): (Vec<f32>, [usize; 2])) -> Vec<Vec<f32>> {
    let rows = shape[0];
    let dim = shape[1];
    if rows == 0 || dim == 0 {
        return Vec::new();
    }

    flat_vectors
        .chunks(dim)
        .take(rows)
        .map(|chunk| chunk.to_vec())
        .collect()
}

fn orchestrator_error_to_py_err(error: OrchestratorError) -> PyErr {
    match error.status_code() {
        1 | 2 => PyValueError::new_err(error.to_string()),
        3 => PyRuntimeError::new_err("postprocess queue is full"),
        _ => PyRuntimeError::new_err(error.to_string()),
    }
}

#[pymodule(gil_used = true)]
fn memori_python(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<MemoriEngine>()?;
    module.add_class::<NativeEmbedder>()?;
    module.add_class::<EngineHandle>()?;
    module.add_function(wrap_pyfunction!(execute, module)?)?;
    module.add_function(wrap_pyfunction!(hello_world, module)?)?;
    module.add_function(wrap_pyfunction!(core_postprocess_request, module)?)?;
    module.add_function(wrap_pyfunction!(embed_texts, module)?)?;
    Ok(())
}
