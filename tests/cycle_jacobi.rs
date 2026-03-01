#![cfg(feature = "inventory")]

//! Test that verifies Jacobi iteration semantics for fixpoint cycle convergence.
//!
//! # What Jacobi iteration means
//!
//! In Jacobi iteration, all non-head cycle participants see **previous-iteration values**
//! (snapshots) of other participants, not freshly computed values from the current iteration.
//! This is in contrast to Gauss-Seidel iteration, where participants see the latest computed
//! values immediately.
//!
//! # How this test verifies Jacobi
//!
//! The cycle structure is:
//!
//! ```text
//! head ──→ query_a ──→ head  (direct cycle)
//!   └────→ query_b ──→ query_a  (B reads A's value)
//!                   └──→ head    (cycle back)
//! ```
//!
//! `head` calls `query_a` first, then `query_b`. Both are non-head cycle participants.
//! `query_b` reads `query_a`'s value.
//!
//! **With Jacobi**: When `query_b` reads `query_a`, it sees `query_a`'s *previous-iteration*
//! value (snapshot), causing `query_b` to lag behind `query_a` by one iteration step. This
//! results in more iterations before convergence.
//!
//! **With Gauss-Seidel**: `query_b` would see `query_a`'s *current-iteration* value immediately
//! (since `query_a` was already re-executed earlier in the same iteration), converging faster.
//!
//! ## Iteration trace (Jacobi)
//!
//! ```text
//! Iter  head_prov  A_new  A_snap→seen  B_new  B_snap→seen  head_new
//!   0       0        1      (none)→1     1     (none)→1       2
//!   1       2        3       1→1        1       1→1          2  (A changed → continue)
//!   2       2        3       3→3        3       1→1          3  (B,head changed → continue)
//!   3       3        3       3→3        3       3→3          3  (all converged → finalize)
//! ```
//!
//! Finalizes at IterationCount(3). With Gauss-Seidel this would be IterationCount(2) because
//! `query_b` would see `query_a`'s new value (3) immediately in iteration 1, and head would
//! jump to 3 one iteration earlier.

mod common;
use common::{ExecuteValidateLoggerDatabase, LogDatabase};
use expect_test::expect;
use salsa::Database as Db;

const MAX: u32 = 3;

#[salsa::input]
struct MyInput {
    field: u32,
}

/// Cycle head: returns `min(A + B, MAX)`.
/// Calls `query_a` first, then `query_b`.
#[salsa::tracked(cycle_initial = cycle_initial)]
fn head(db: &dyn Db, input: MyInput) -> u32 {
    let a = query_a(db, input);
    let b = query_b(db, input);
    (a + b).min(MAX)
}

fn cycle_initial(_db: &dyn Db, _id: salsa::Id, _input: MyInput) -> u32 {
    0
}

/// A = min(head + 1, MAX). Direct cycle participant (calls head).
#[salsa::tracked(cycle_initial = cycle_initial)]
fn query_a(db: &dyn Db, input: MyInput) -> u32 {
    (head(db, input) + 1).min(MAX)
}

/// B reads A's value. With Jacobi, B sees A's *previous-iteration* snapshot.
/// B also calls head to participate in the cycle.
#[salsa::tracked(cycle_initial = cycle_initial)]
fn query_b(db: &dyn Db, input: MyInput) -> u32 {
    let _ = head(db, input);
    query_a(db, input)
}

/// Verify that the iteration count reflects Jacobi semantics (snapshot-based
/// value propagation) rather than Gauss-Seidel (immediate propagation).
#[test]
fn jacobi_snapshot_causes_propagation_delay() {
    let db = ExecuteValidateLoggerDatabase::default();
    let input = MyInput::new(&db, 1);

    // Final value should be MAX regardless of iteration method.
    assert_eq!(head(&db, input), MAX);

    // The iteration count proves Jacobi: IterationCount(3) instead of
    // Gauss-Seidel's IterationCount(2). The extra iteration is caused by
    // query_b seeing query_a's previous-iteration snapshot, delaying
    // convergence by one step.
    db.assert_logs(expect![[r#"
        [
            "salsa_event(WillExecute { database_key: head(Id(0)) })",
            "salsa_event(WillExecute { database_key: query_a(Id(0)) })",
            "salsa_event(WillExecute { database_key: query_b(Id(0)) })",
            "salsa_event(WillIterateCycle { database_key: head(Id(0)), iteration_count: IterationCount(1) })",
            "salsa_event(WillExecute { database_key: query_a(Id(0)) })",
            "salsa_event(WillExecute { database_key: query_b(Id(0)) })",
            "salsa_event(WillIterateCycle { database_key: head(Id(0)), iteration_count: IterationCount(2) })",
            "salsa_event(WillExecute { database_key: query_a(Id(0)) })",
            "salsa_event(WillExecute { database_key: query_b(Id(0)) })",
            "salsa_event(WillIterateCycle { database_key: head(Id(0)), iteration_count: IterationCount(3) })",
            "salsa_event(WillExecute { database_key: query_a(Id(0)) })",
            "salsa_event(WillExecute { database_key: query_b(Id(0)) })",
            "salsa_event(DidFinalizeCycle { database_key: head(Id(0)), iteration_count: IterationCount(3) })",
        ]"#]]);
}
