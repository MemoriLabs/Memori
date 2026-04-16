use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc as std_mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use tokio::runtime::{Builder, Handle, Runtime};
use tokio::sync::{Notify, Semaphore, mpsc};
use tokio::task::JoinHandle;

use crate::runtime::config::RuntimeConfig;
use crate::runtime::errors::{FlushError, RuntimeError, SubmitError};
use crate::runtime::state::{Lifecycle, LifecycleState};

pub(crate) type BoxFuture = Pin<Box<dyn Future<Output = ()> + Send>>;
pub(crate) type JobHandler<J> = Arc<dyn Fn(J) -> BoxFuture + Send + Sync>;

struct OutstandingGuard<J: Send + 'static> {
    inner: Arc<Inner<J>>,
}

impl<J: Send + 'static> Drop for OutstandingGuard<J> {
    fn drop(&mut self) {
        let prev = self.inner.outstanding.fetch_sub(1, Ordering::SeqCst);
        if prev == 1 {
            self.inner.wake_waiters();
        }
    }
}

enum ShutdownStage {
    NotShutdown,
    InProgress,
    Done,
}

struct Inner<J: Send + 'static> {
    config: RuntimeConfig,
    lifecycle: Lifecycle,
    job_tx: Mutex<Option<mpsc::Sender<J>>>,
    outstanding: AtomicUsize,
    flush_lock: Mutex<()>,
    flush_cvar: Condvar,
    drain_notify: Notify,
    runtime: Mutex<Option<Runtime>>,
    dispatcher_join: Mutex<Option<JoinHandle<()>>>,
    semaphore: Arc<Semaphore>,
    handler: JobHandler<J>,
    ops: Mutex<()>,
    shutdown_stage: Mutex<ShutdownStage>,
    shutdown_cvar: Condvar,
}

impl<J: Send + 'static> Inner<J> {
    fn new(config: RuntimeConfig, handler: JobHandler<J>) -> Self {
        let max_concurrency = config.max_concurrency;
        Self {
            config,
            lifecycle: Lifecycle::new(),
            job_tx: Mutex::new(None),
            outstanding: AtomicUsize::new(0),
            flush_lock: Mutex::new(()),
            flush_cvar: Condvar::new(),
            drain_notify: Notify::new(),
            runtime: Mutex::new(None),
            dispatcher_join: Mutex::new(None),
            semaphore: Arc::new(Semaphore::new(max_concurrency)),
            handler,
            ops: Mutex::new(()),
            shutdown_stage: Mutex::new(ShutdownStage::NotShutdown),
            shutdown_cvar: Condvar::new(),
        }
    }

    fn wake_waiters(&self) {
        self.flush_cvar.notify_all();
        self.drain_notify.notify_waiters();
    }

    fn notify_zero_if_needed(&self) {
        if self.outstanding.load(Ordering::Acquire) == 0 {
            self.wake_waiters();
        }
    }
}

/// Background worker runtime with a bounded internal queue and capped concurrent async handlers.
#[derive(Clone)]
pub struct WorkerRuntime<J: Send + 'static> {
    inner: Arc<Inner<J>>,
}

impl<J: Send + 'static> WorkerRuntime<J> {
    /// Creates a runtime in the "not started" state. Validates [`RuntimeConfig`].
    pub fn new<F, Fut>(config: RuntimeConfig, handler: F) -> Result<Self, RuntimeError>
    where
        F: Fn(J) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        config.validate().map_err(RuntimeError::InvalidConfig)?;
        let f = Arc::new(handler);
        let handler: JobHandler<J> = Arc::new(move |job: J| {
            let f = f.clone();
            Box::pin(async move { f(job).await })
        });
        Ok(Self {
            inner: Arc::new(Inner::new(config, handler)),
        })
    }

    /// Starts the dedicated Tokio runtime and dispatcher. Errors if already started.
    pub fn start(&self) -> Result<(), RuntimeError> {
        let _ops = self.inner.ops.lock().unwrap();
        match self.inner.lifecycle.load() {
            LifecycleState::Running => return Err(RuntimeError::AlreadyStarted),
            LifecycleState::ShuttingDown | LifecycleState::Stopped => {
                return Err(RuntimeError::AlreadyStarted);
            }
            LifecycleState::NotStarted => {}
        }

        let (tx, rx) = mpsc::channel(self.inner.config.queue_capacity);
        *self.inner.job_tx.lock().unwrap() = Some(tx);

        let sem = self.inner.semaphore.clone();
        let handler = self.inner.handler.clone();
        let inner = self.inner.clone();

        let join = if let Some(ref h) = self.inner.config.tokio_handle {
            h.spawn(async move {
                run_dispatcher(rx, sem, handler, inner).await;
            })
        } else {
            let worker_threads = self
                .inner
                .config
                .worker_threads
                .unwrap_or_else(default_worker_threads);

            let runtime = Builder::new_multi_thread()
                .worker_threads(worker_threads)
                .enable_all()
                .build()?;

            let join = runtime.spawn(async move {
                run_dispatcher(rx, sem, handler, inner).await;
            });

            *self.inner.runtime.lock().unwrap() = Some(runtime);
            join
        };

        *self.inner.dispatcher_join.lock().unwrap() = Some(join);
        self.inner.lifecycle.store(LifecycleState::Running);
        Ok(())
    }

    /// Enqueues a job without blocking. Fails when the queue is full or the runtime is not running.
    pub fn submit(&self, job: J) -> Result<(), SubmitError<J>> {
        let job_tx = self.inner.job_tx.lock().unwrap();
        match self.inner.lifecycle.load() {
            LifecycleState::NotStarted => return Err(SubmitError::NotRunning),
            LifecycleState::Running => {}
            LifecycleState::ShuttingDown => return Err(SubmitError::ShuttingDown),
            LifecycleState::Stopped => return Err(SubmitError::Stopped),
        }
        let Some(tx) = job_tx.as_ref() else {
            return Err(SubmitError::NotRunning);
        };
        match tx.try_send(job) {
            Ok(()) => {
                drop(job_tx);
                self.inner.outstanding.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
            Err(mpsc::error::TrySendError::Full(job)) => Err(SubmitError::QueueFull(job)),
            Err(mpsc::error::TrySendError::Closed(_)) => Err(SubmitError::ShuttingDown),
        }
    }

    /// Blocks until all accepted jobs have finished (queued and in-flight).
    pub fn flush(&self) -> Result<(), FlushError> {
        self.flush_impl(None)
    }

    /// Like [`flush`](Self::flush) but returns [`FlushError::Timeout`] after `timeout`.
    pub fn flush_for(&self, timeout: Duration) -> Result<(), FlushError> {
        self.flush_impl(Some(timeout))
    }

    fn flush_impl(&self, timeout: Option<Duration>) -> Result<(), FlushError> {
        match self.inner.lifecycle.load() {
            LifecycleState::NotStarted => return Err(FlushError::NotRunning),
            LifecycleState::Running | LifecycleState::ShuttingDown => {}
            LifecycleState::Stopped => {
                if self.inner.outstanding.load(Ordering::Acquire) == 0 {
                    return Ok(());
                }
            }
        }

        let timed = timeout.map(|dur| (Instant::now() + dur, dur));
        let mut guard = self.inner.flush_lock.lock().unwrap();
        loop {
            if self.inner.outstanding.load(Ordering::Acquire) == 0 {
                return Ok(());
            }
            if let Some((deadline, limit)) = timed {
                let now = Instant::now();
                if now >= deadline {
                    return Err(FlushError::Timeout(limit));
                }
                let wait = deadline - now;
                let (g, _) = self.inner.flush_cvar.wait_timeout(guard, wait).unwrap();
                guard = g;
            } else {
                guard = self.inner.flush_cvar.wait(guard).unwrap();
            }
        }
    }

    /// Stops accepting jobs, drains work per [`crate::runtime::config::ShutdownPolicy`], and tears down the runtime.
    /// Safe to call multiple times.
    pub fn shutdown(&self) {
        {
            let mut stage = self.inner.shutdown_stage.lock().unwrap();
            loop {
                match *stage {
                    ShutdownStage::Done => return,
                    ShutdownStage::InProgress => {
                        stage = self.inner.shutdown_cvar.wait(stage).unwrap();
                    }
                    ShutdownStage::NotShutdown => match self.inner.lifecycle.load() {
                        LifecycleState::Stopped => {
                            *stage = ShutdownStage::Done;
                            self.inner.shutdown_cvar.notify_all();
                            return;
                        }
                        LifecycleState::NotStarted => {
                            self.inner.lifecycle.store(LifecycleState::Stopped);
                            *stage = ShutdownStage::Done;
                            self.inner.shutdown_cvar.notify_all();
                            return;
                        }
                        _ => {
                            *stage = ShutdownStage::InProgress;
                            break;
                        }
                    },
                }
            }
        }

        {
            let mut job_tx = self.inner.job_tx.lock().unwrap();
            self.inner.lifecycle.store(LifecycleState::ShuttingDown);
            *job_tx = None;
        }

        let runtime = self.inner.runtime.lock().unwrap().take();
        let join = self.inner.dispatcher_join.lock().unwrap().take();

        match (runtime, join) {
            (Some(rt), join) => {
                rt.block_on(async {
                    if let Some(j) = join {
                        let _ = j.await;
                    }
                });
                rt.shutdown_timeout(Duration::from_secs(30));
            }
            (None, Some(join)) => {
                if let Some(h) = self.inner.config.tokio_handle.clone() {
                    wait_dispatcher_on_handle(h, join);
                }
            }
            (None, None) => {}
        }

        self.inner.lifecycle.store(LifecycleState::Stopped);

        let mut stage = self.inner.shutdown_stage.lock().unwrap();
        *stage = ShutdownStage::Done;
        self.inner.shutdown_cvar.notify_all();
        drop(stage);

        self.inner.notify_zero_if_needed();
    }
}

fn default_worker_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

fn wait_dispatcher_on_handle(handle: Handle, join: JoinHandle<()>) {
    let (done_tx, done_rx) = std_mpsc::sync_channel::<()>(1);
    handle.spawn(async move {
        let _ = join.await;
        let _ = done_tx.send(());
    });
    let _ = done_rx.recv();
}

async fn run_dispatcher<J: Send + 'static>(
    mut rx: mpsc::Receiver<J>,
    sem: Arc<Semaphore>,
    handler: JobHandler<J>,
    inner: Arc<Inner<J>>,
) {
    while let Some(job) = rx.recv().await {
        let permit = match sem.clone().acquire_owned().await {
            Ok(p) => p,
            Err(_) => break,
        };
        let handler = handler.clone();
        let inner_for_task = inner.clone();
        tokio::spawn(async move {
            let _guard = OutstandingGuard {
                inner: inner_for_task,
            };
            let _permit = permit;
            handler(job).await;
        });
    }

    loop {
        if inner.outstanding.load(Ordering::Acquire) == 0 {
            break;
        }
        inner.drain_notify.notified().await;
    }
}
