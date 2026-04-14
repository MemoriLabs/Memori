//! In-process background job runtime: bounded queue, capped concurrent async handlers, flush/shutdown.
//!
//! Use one [`WorkerRuntime`] per pool (for example separate pools for API I/O vs database writes).
//! `queue_capacity` bounds memory and applies backpressure via [`WorkerRuntime::submit`].
//! `max_concurrency` limits how many handler futures run at once while awaiting I/O.
//!
//! Prefer setting [`RuntimeConfig::tokio_handle`] to your application's
//! [`tokio::runtime::Handle`] so pools share one Tokio runtime instead of nesting dedicated ones.
//!
//! ## Lifecycle
//!
//! 1. [`WorkerRuntime::new`] - validate config, build handle (not running).
//! 2. [`WorkerRuntime::start`] - dedicated Tokio runtime + dispatcher.
//! 3. [`WorkerRuntime::submit`] - non-blocking enqueue (`try_send`); on a full queue returns
//!    [`SubmitError::QueueFull`] with the job so callers can retry or drop explicitly.
//! 4. [`WorkerRuntime::flush`] / [`WorkerRuntime::flush_for`] - wait until accepted jobs finish.
//! 5. [`WorkerRuntime::shutdown`] - stop accepting, drain, tear down (idempotent).
//!
//! ## Deadlock
//!
//! Do not call [`WorkerRuntime::flush`] or [`WorkerRuntime::shutdown`] from inside a job running
//! on the same runtime's worker threads.

mod config;
mod errors;
mod state;
mod worker;

pub use config::{RuntimeConfig, ShutdownPolicy};
pub use errors::{FlushError, RuntimeError, SubmitError};
pub use worker::WorkerRuntime;
