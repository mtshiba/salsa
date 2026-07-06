use std::fmt;
use std::panic::{self, UnwindSafe};

/// A panic payload indicating that execution of a salsa query was cancelled.
///
/// This can occur for a few reasons:
/// *
/// *
/// *
#[derive(Debug)]
#[non_exhaustive]
pub enum Cancelled {
    /// The query was operating but the local database execution has been cancelled.
    Local,

    /// The query was operating on revision R, but there is a pending write to move to revision R+1.
    PendingWrite,

    /// The query was blocked on another thread, and that thread panicked.
    PropagatedPanic,

    /// The query attempted to claim a query on another thread and that would have formed a
    /// cross-thread cycle. To keep fixpoint iteration deterministic, the requesting side
    /// backs out entirely (releasing all of its claims) and retries from the top level once
    /// the winning thread has finalized the cycle.
    CycleLoser,
}

impl Cancelled {
    #[cold]
    pub(crate) fn throw(self) -> ! {
        // We use resume and not panic here to avoid running the panic
        // hook (that is, to avoid collecting and printing backtrace).
        panic::resume_unwind(Box::new(self));
    }

    /// Runs `f`, and catches any salsa cancellation.
    pub fn catch<F, T>(f: F) -> Result<T, Cancelled>
    where
        F: FnOnce() -> T + UnwindSafe,
    {
        match panic::catch_unwind(f) {
            Ok(t) => Ok(t),
            Err(payload) => match payload.downcast() {
                Ok(cancelled) => Err(*cancelled),
                Err(payload) => panic::resume_unwind(payload),
            },
        }
    }
}

impl std::fmt::Display for Cancelled {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let why = match self {
            Cancelled::Local => "local cancellation request",
            Cancelled::PendingWrite => "pending write",
            Cancelled::PropagatedPanic => "propagated panic",
            Cancelled::CycleLoser => "losing a cross-thread cycle race",
        };
        f.write_str("cancelled because of ")?;
        f.write_str(why)
    }
}

impl std::error::Error for Cancelled {}
