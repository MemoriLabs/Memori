use tokio::runtime::Handle;

/// Configuration for [`crate::WorkerRuntime`].
#[derive(Clone)]
pub struct RuntimeConfig {
    /// Capacity of the internal bounded job queue (`try_send` fails when full).
    pub queue_capacity: usize,
    /// Maximum handler futures executing at once (I/O overlap cap).
    pub max_concurrency: usize,
    /// Tokio worker threads when [`Self::tokio_handle`] is [`None`]; ignored if a handle is set.
    pub worker_threads: Option<usize>,
    /// If set, [`crate::WorkerRuntime::start`] schedules work on this runtime
    /// instead of creating a nested one (recommended for applications that already own a Tokio runtime).
    pub tokio_handle: Option<Handle>,
    /// Shutdown behavior (v1: drain only).
    pub shutdown: ShutdownPolicy,
}

impl std::fmt::Debug for RuntimeConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RuntimeConfig")
            .field("queue_capacity", &self.queue_capacity)
            .field("max_concurrency", &self.max_concurrency)
            .field("worker_threads", &self.worker_threads)
            .field(
                "tokio_handle",
                &self.tokio_handle.as_ref().map(|_| "<Handle>"),
            )
            .field("shutdown", &self.shutdown)
            .finish()
    }
}

/// How [`crate::WorkerRuntime::shutdown`] behaves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ShutdownPolicy {
    /// Finish all accepted jobs (queued and in-flight) before stopping.
    #[default]
    Drain,
}

impl RuntimeConfig {
    /// Validates fields; used by [`crate::WorkerRuntime::new`].
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.queue_capacity < 1 {
            return Err("queue_capacity must be at least 1");
        }
        if self.max_concurrency < 1 {
            return Err("max_concurrency must be at least 1");
        }
        if let Some(n) = self.worker_threads {
            if n < 1 {
                return Err("worker_threads must be at least 1 when set");
            }
        }
        Ok(())
    }
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            queue_capacity: 256,
            max_concurrency: 32,
            worker_threads: None,
            tokio_handle: None,
            shutdown: ShutdownPolicy::Drain,
        }
    }
}
