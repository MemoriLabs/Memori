use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::Duration;

use engine_orchestrator::{FlushError, RuntimeConfig, RuntimeError, SubmitError, WorkerRuntime};
use tokio::sync::Barrier;

fn test_config(queue_capacity: usize, max_concurrency: usize) -> RuntimeConfig {
    RuntimeConfig {
        queue_capacity,
        max_concurrency,
        worker_threads: Some(2),
        ..Default::default()
    }
}

#[test]
fn start_uses_shared_tokio_handle() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let mut cfg = test_config(8, 2);
    cfg.tokio_handle = Some(rt.handle().clone());
    let w: WorkerRuntime<u32> = WorkerRuntime::new(cfg, |_j| async {}).unwrap();
    w.start().unwrap();
    w.submit(0).unwrap();
    w.flush().unwrap();
    w.shutdown();
    drop(rt);
}

#[test]
fn submit_fails_when_not_started() {
    let rt: WorkerRuntime<u32> =
        WorkerRuntime::new(RuntimeConfig::default(), |_j| async {}).unwrap();
    assert!(matches!(rt.submit(1), Err(SubmitError::NotRunning)));
}

#[test]
fn start_twice_errors() {
    let rt: WorkerRuntime<u32> =
        WorkerRuntime::new(RuntimeConfig::default(), |_j| async {}).unwrap();
    rt.start().unwrap();
    assert!(matches!(rt.start(), Err(RuntimeError::AlreadyStarted)));
    rt.shutdown();
}

#[test]
fn queue_full_when_saturated() {
    let config = test_config(1, 1);
    let rt: WorkerRuntime<u32> = WorkerRuntime::new(config, |_j| async {
        tokio::time::sleep(Duration::from_millis(500)).await;
    })
    .unwrap();
    rt.start().unwrap();
    assert!(rt.submit(0).is_ok());
    thread::sleep(Duration::from_millis(50));
    assert!(rt.submit(1).is_ok());
    assert!(matches!(rt.submit(2), Err(SubmitError::QueueFull(2))));
    rt.shutdown();
}

#[test]
fn flush_waits_for_completion() {
    let done = Arc::new(AtomicUsize::new(0));
    let d = done.clone();
    let config = test_config(8, 4);
    let rt: WorkerRuntime<u32> = WorkerRuntime::new(config, move |_j| {
        let d = d.clone();
        async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            d.fetch_add(1, Ordering::SeqCst);
        }
    })
    .unwrap();
    rt.start().unwrap();
    rt.submit(0).unwrap();
    rt.submit(1).unwrap();
    rt.flush().unwrap();
    assert_eq!(done.load(Ordering::SeqCst), 2);
    rt.shutdown();
}

#[test]
fn flush_for_times_out() {
    let config = test_config(8, 1);
    let rt: WorkerRuntime<u32> = WorkerRuntime::new(config, |_j| async {
        tokio::time::sleep(Duration::from_secs(10)).await;
    })
    .unwrap();
    rt.start().unwrap();
    rt.submit(0).unwrap();
    assert!(matches!(
        rt.flush_for(Duration::from_millis(50)),
        Err(FlushError::Timeout(_))
    ));
    rt.shutdown();
}

#[test]
fn shutdown_is_idempotent() {
    let rt: WorkerRuntime<u32> = WorkerRuntime::new(test_config(4, 2), |_j| async {}).unwrap();
    rt.start().unwrap();
    rt.shutdown();
    rt.shutdown();
    assert!(matches!(rt.submit(0), Err(SubmitError::Stopped)));
    assert!(matches!(rt.start(), Err(RuntimeError::AlreadyStarted)));
}

#[test]
fn submit_rejected_after_shutdown() {
    let rt: WorkerRuntime<u32> = WorkerRuntime::new(test_config(4, 2), |_j| async {}).unwrap();
    rt.start().unwrap();
    rt.shutdown();
    assert!(matches!(rt.submit(0), Err(SubmitError::Stopped)));
}

#[test]
fn concurrent_submit() {
    let count = Arc::new(AtomicUsize::new(0));
    let c = count.clone();
    let config = test_config(64, 8);
    let rt: WorkerRuntime<u32> = WorkerRuntime::new(config, move |_j| {
        let c = c.clone();
        async move {
            c.fetch_add(1, Ordering::SeqCst);
        }
    })
    .unwrap();
    rt.start().unwrap();
    let handles: Vec<_> = (0..32u32)
        .map(|i| {
            let r = rt.clone();
            thread::spawn(move || {
                let _ = r.submit(i);
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    rt.flush().unwrap();
    assert_eq!(count.load(Ordering::SeqCst), 32);
    rt.shutdown();
}

#[test]
fn overlap_up_to_max_concurrency() {
    let in_flight = Arc::new(AtomicUsize::new(0));
    let max_seen = Arc::new(AtomicUsize::new(0));
    let barrier = Arc::new(Barrier::new(3));

    let ifl = in_flight.clone();
    let ms = max_seen.clone();
    let b = barrier.clone();

    let config = test_config(16, 3);
    let rt: WorkerRuntime<u32> = WorkerRuntime::new(config, move |_j| {
        let ifl = ifl.clone();
        let ms = ms.clone();
        let b = b.clone();
        async move {
            let n = ifl.fetch_add(1, Ordering::SeqCst) + 1;
            loop {
                let cur = ms.load(Ordering::SeqCst);
                if n <= cur {
                    break;
                }
                if ms
                    .compare_exchange(cur, n, Ordering::SeqCst, Ordering::SeqCst)
                    .is_ok()
                {
                    break;
                }
            }
            b.wait().await;
            ifl.fetch_sub(1, Ordering::SeqCst);
        }
    })
    .unwrap();

    rt.start().unwrap();
    for i in 0..3u32 {
        rt.submit(i).unwrap();
    }
    thread::sleep(Duration::from_millis(50));
    assert!(max_seen.load(Ordering::SeqCst) >= 3);
    rt.shutdown();
}
