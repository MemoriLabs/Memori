use std::sync::atomic::{AtomicU8, Ordering};

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LifecycleState {
    NotStarted = 0,
    Running = 1,
    ShuttingDown = 2,
    Stopped = 3,
}

impl TryFrom<u8> for LifecycleState {
    type Error = ();

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::NotStarted),
            1 => Ok(Self::Running),
            2 => Ok(Self::ShuttingDown),
            3 => Ok(Self::Stopped),
            _ => Err(()),
        }
    }
}

pub(crate) struct Lifecycle {
    raw: AtomicU8,
}

impl Lifecycle {
    pub(crate) fn new() -> Self {
        Self {
            raw: AtomicU8::new(LifecycleState::NotStarted as u8),
        }
    }

    pub(crate) fn load(&self) -> LifecycleState {
        LifecycleState::try_from(self.raw.load(Ordering::Acquire))
            .unwrap_or(LifecycleState::Stopped)
    }

    pub(crate) fn store(&self, state: LifecycleState) {
        self.raw.store(state as u8, Ordering::Release);
    }
}
