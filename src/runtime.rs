use self::dependency_graph::DependencyGraph;
use crate::durability::Durability;
use crate::function::{SyncGuard, SyncOwner};
use crate::key::DatabaseKeyIndex;
use crate::sync::Mutex;
use crate::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use crate::sync::thread::{self, ThreadId};
use crate::table::Table;
use crate::zalsa::Zalsa;
use crate::{Cancelled, Event, EventKind, Revision};

mod dependency_graph;

#[cfg_attr(feature = "persistence", derive(serde::Serialize, serde::Deserialize))]
pub struct Runtime {
    /// Set to true when the current revision has been cancelled.
    /// This is done when we an input is being changed. The flag
    /// is set back to false once the input has been changed.
    #[cfg_attr(feature = "persistence", serde(skip))]
    revision_cancelled: AtomicBool,

    /// Distinguishes provisional cycle results created before and after cancelling other handles
    /// within the same revision. Reset when the revision advances.
    #[cfg_attr(feature = "persistence", serde(skip))]
    cancellation_count: AtomicU8,

    /// In-flight cycle groups (GODE), see [`cycle_groups::CycleGroups`].
    #[cfg_attr(feature = "persistence", serde(skip))]
    cycle_groups: cycle_groups::CycleGroups,

    /// Stores the "last change" revision for values of each duration.
    /// This vector is always of length at least 1 (for Durability 0)
    /// but its total length depends on the number of durations. The
    /// element at index 0 is special as it represents the "current
    /// revision".  In general, we have the invariant that revisions
    /// in here are *declining* -- that is, `revisions[i] >=
    /// revisions[i + 1]`, for all `i`. This is because when you
    /// modify a value with durability D, that implies that values
    /// with durability less than D may have changed too.
    revisions: [Revision; Durability::LEN],

    /// The dependency graph tracks which runtimes are blocked on one
    /// another, waiting for queries to terminate.
    #[cfg_attr(feature = "persistence", serde(skip))]
    dependency_graph: Mutex<DependencyGraph>,

    /// Data for instances
    #[cfg_attr(feature = "persistence", serde(skip))]
    table: Table,
}

#[derive(Copy, Clone, Debug)]
pub(super) enum WaitResult {
    Completed,
    Panicked,
    Cancelled,
    /// The blocked thread must back out of its entire computation (releasing all of its
    /// claims) and retry from the top level: it is part of a cross-thread cycle whose
    /// winner requested the chain to yield. See `Cancelled::CycleLoser`.
    BackOut,
}

#[derive(Debug)]
pub(crate) enum BlockResult<'me> {
    /// The query is running on another thread.
    Running(Running<'me>),

    /// Blocking resulted in a cycle.
    ///
    /// The lock is hold by the current thread or there's another thread that is waiting on the current thread,
    /// and blocking this thread on the other thread would result in a deadlock/cycle.
    ///
    /// `same_thread` is `true` when the lock is held by the current thread itself (an
    /// intra-thread cycle over the query stack). `false` means the cycle spans threads:
    /// another thread transitively waits on us; `owner` is the thread holding the
    /// contested key (used for the driver-priority back-out decision).
    Cycle { same_thread: bool, owner: ThreadId },
}

pub(crate) enum BlockTransferredResult<'me> {
    /// The current thread is the owner of the transferred query
    /// and it can claim it if it wants to.
    ImTheOwner,

    /// The query is owned/running on another thread.
    OwnedBy(Box<BlockOnTransferredOwner<'me>>),

    /// The query has transferred its ownership to another query previously but that query has
    /// since then completed and released the lock.
    Released,
}

pub(super) struct BlockOnTransferredOwner<'me> {
    dg: crate::sync::MutexGuard<'me, DependencyGraph>,
    /// The query that we're trying to claim.
    database_key: DatabaseKeyIndex,
    /// The thread that currently owns the lock for the transferred query.
    other_id: ThreadId,
    /// The current thread that is trying to claim the transferred query.
    thread_id: ThreadId,
}

impl<'me> BlockOnTransferredOwner<'me> {
    /// Block on the other thread to complete the computation.
    pub(super) fn block(self, query_mutex_guard: SyncGuard<'me>) -> BlockResult<'me> {
        // Cycle in the same thread.
        if self.thread_id == self.other_id {
            return BlockResult::Cycle {
                same_thread: true,
                owner: self.other_id,
            };
        }

        if self.dg.depends_on(self.other_id, self.thread_id) {
            crate::tracing::debug!(
                "block_on: cycle detected for {:?} in thread {thread_id:?} on {:?}",
                self.database_key,
                self.other_id,
                thread_id = self.thread_id
            );
            return BlockResult::Cycle {
                same_thread: false,
                owner: self.other_id,
            };
        }

        BlockResult::Running(Running(Box::new(BlockedOnInner {
            dg: self.dg,
            query_mutex_guard,
            database_key: self.database_key,
            other_id: self.other_id,
            thread_id: self.thread_id,
        })))
    }
}

pub struct Running<'me>(Box<BlockedOnInner<'me>>);

struct BlockedOnInner<'me> {
    dg: crate::sync::MutexGuard<'me, DependencyGraph>,
    query_mutex_guard: SyncGuard<'me>,
    database_key: DatabaseKeyIndex,
    other_id: ThreadId,
    thread_id: ThreadId,
}

/// Registry of in-flight cycle groups (GODE: Group-Owned Detached Evaluation).
///
/// A cycle group is created when a thread closes a dependency cycle; that thread backs
/// out to its top level and re-evaluates the strongly connected component *detached*
/// (with an empty query stack), as the group's single owner. Every other thread that
/// touches the group's provisional state parks here until the group completes. The group
/// records its members so that an aborted evaluation can be discarded wholesale by
/// removing the members' memos (leaving no observable leftovers).
pub(crate) mod cycle_groups {
    use rustc_hash::FxHashMap;

    use crate::key::DatabaseKeyIndex;
    use crate::sync::thread::ThreadId;
    use crate::sync::{Arc, Mutex};

    pub(crate) struct CycleGroups {
        state: Mutex<GroupsState>,
    }

    #[derive(Default)]
    struct GroupsState {
        /// Live groups by root key.
        groups: FxHashMap<DatabaseKeyIndex, Arc<GroupState>>,
        /// Member key -> root key of the live group it belongs to.
        member_index: FxHashMap<DatabaseKeyIndex, DatabaseKeyIndex>,
        /// Member key -> cycle heads it read. Links members into connected
        /// components: a group is a batch of every cycle met during one evaluation,
        /// and canonicalization must scope to one component (see `component_of`).
        member_heads: FxHashMap<DatabaseKeyIndex, Vec<DatabaseKeyIndex>>,
        /// Owner thread -> root key of the live group it is evaluating.
        owner_index: FxHashMap<ThreadId, DatabaseKeyIndex>,
    }

    pub(crate) struct GroupState {
        pub(crate) root: DatabaseKeyIndex,
        pub(crate) owner: ThreadId,
    }

    /// An arbitrary but run-stable total order over query keys, used to pick canonical
    /// cycle-group entries and to arbitrate group-vs-group conflicts.
    pub(crate) fn key_order(key: DatabaseKeyIndex) -> (u32, u64) {
        (key.ingredient_index().as_u32(), key.key_index().as_bits())
    }

    impl Default for CycleGroups {
        fn default() -> Self {
            Self {
                state: Mutex::new(GroupsState::default()),
            }
        }
    }

    impl CycleGroups {
        /// Registers a new group rooted at `root`, owned by the current thread. Returns
        /// `None` if a live group already covers `root` (as root or member).
        pub(crate) fn register(
            &self,
            root: DatabaseKeyIndex,
            owner: ThreadId,
        ) -> Option<Arc<GroupState>> {
            let mut state = self.state.lock();
            if state.groups.contains_key(&root) || state.member_index.contains_key(&root) {
                return None;
            }
            let group = Arc::new(GroupState { root, owner });
            state.groups.insert(root, Arc::clone(&group));
            state.member_index.insert(root, root);
            state.owner_index.insert(owner, root);
            Some(group)
        }

        /// Records `member` as part of the live group rooted at `root`, linked to the
        /// cycle `heads` it read (itself, if the member is a head). Re-recording
        /// replaces the links: components must reflect the converged dependency
        /// structure, not couplings from earlier iterations that have since vanished
        /// (a conditional cycle edge would otherwise keep two components glued
        /// together and make canonicalization restart forever).
        pub(crate) fn add_member(
            &self,
            root: DatabaseKeyIndex,
            member: DatabaseKeyIndex,
            heads: impl IntoIterator<Item = DatabaseKeyIndex>,
        ) {
            let mut state = self.state.lock();
            debug_assert!(state.groups.contains_key(&root));
            state.member_index.insert(member, root);
            state.member_heads.insert(member, heads.into_iter().collect());
        }

        /// The members of `head`'s connected component within the group rooted at
        /// `root`: the closure of the member–head links. Superset of the strongly
        /// connected component `head` belongs to, and disjoint from independent
        /// cycles that merely share the batch; a function of the dependency
        /// structure, so stable across runs and entry points.
        pub(crate) fn component_of(
            &self,
            root: DatabaseKeyIndex,
            head: DatabaseKeyIndex,
        ) -> Vec<DatabaseKeyIndex> {
            let state = self.state.lock();
            if std::env::var("GODE_DEBUG").is_ok() {
                let ledger: Vec<_> = state
                    .member_heads
                    .iter()
                    .filter(|(m, _)| state.member_index.get(m) == Some(&root))
                    .collect();
                eprintln!("[ledger] root={root:?} head={head:?} {ledger:?}");
            }
            let mut component = vec![head];
            let mut frontier = vec![head];
            while let Some(current) = frontier.pop() {
                for (&member, heads) in &state.member_heads {
                    if state.member_index.get(&member) != Some(&root) {
                        continue;
                    }
                    if component.contains(&member) {
                        continue;
                    }
                    if heads.contains(&current) || member == current {
                        component.push(member);
                        frontier.push(member);
                    }
                }
                // Heads this member read also join the component.
                if let Some(heads) = state.member_heads.get(&current) {
                    for &linked in heads {
                        if !component.contains(&linked) {
                            component.push(linked);
                            frontier.push(linked);
                        }
                    }
                }
            }
            component
        }

        /// The live group that `key` belongs to, if any.
        pub(crate) fn group_of(&self, key: DatabaseKeyIndex) -> Option<Arc<GroupState>> {
            let state = self.state.lock();
            let root = state.member_index.get(&key)?;
            state.groups.get(root).cloned()
        }

        /// The root of the live group owned (evaluated) by `thread`, if any.
        pub(crate) fn root_owned_by(&self, thread: ThreadId) -> Option<DatabaseKeyIndex> {
            self.state.lock().owner_index.get(&thread).copied()
        }

        /// The members of the live group rooted at `root` (excluding queries that were
        /// never recorded).
        pub(crate) fn members_of(&self, root: DatabaseKeyIndex) -> Vec<DatabaseKeyIndex> {
            let state = self.state.lock();
            state
                .member_index
                .iter()
                .filter_map(|(member, r)| (*r == root).then_some(*member))
                .collect()
        }

        /// Completes the group rooted at `root`, removing it from the registry. Threads
        /// blocked on the root's claim are woken through the ordinary claim-release
        /// machinery; the registry itself has no waiters.
        pub(crate) fn complete(&self, root: DatabaseKeyIndex) {
            let mut state = self.state.lock();
            let GroupsState {
                member_index,
                member_heads,
                ..
            } = &mut *state;
            member_index.retain(|member, r| {
                let keep = *r != root;
                if !keep {
                    member_heads.remove(member);
                }
                keep
            });
            state.owner_index.retain(|_, r| *r != root);
            state.groups.remove(&root);
        }
    }
}

/// The outcome of blocking on another thread's computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BlockOutcome {
    /// The other thread completed the computation.
    Completed,
    /// The other thread was cancelled; retry the claim.
    Cancelled,
    /// The current thread must back out of its computation entirely (see
    /// `WaitResult::BackOut`).
    BackOut,
}

impl Running<'_> {
    /// Blocks on the other thread to complete the computation.
    ///
    /// # Panics
    ///
    /// If the other thread panics, this function will panic as well.
    #[must_use]
    pub(crate) fn block_on(self, zalsa: &Zalsa) -> BlockOutcome {
        let BlockedOnInner {
            dg,
            query_mutex_guard,
            database_key,
            other_id,
            thread_id,
        } = *self.0;

        zalsa.event(&|| {
            Event::new(EventKind::WillBlockOn {
                other_thread_id: other_id,
                database_key,
            })
        });

        crate::tracing::debug!(
            "block_on: thread {thread_id:?} is blocking on {database_key:?} in thread {other_id:?}",
        );

        let result =
            DependencyGraph::block_on(dg, thread_id, database_key, other_id, query_mutex_guard);

        match result {
            WaitResult::Panicked => {
                // If the other thread panicked, then we consider this thread
                // cancelled. The assumption is that the panic will be detected
                // by the other thread and responded to appropriately.
                Cancelled::PropagatedPanic.throw()
            }
            WaitResult::Cancelled => BlockOutcome::Cancelled,
            WaitResult::Completed => BlockOutcome::Completed,
            WaitResult::BackOut => BlockOutcome::BackOut,
        }
    }
}

impl std::fmt::Debug for Running<'_> {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fmt.debug_struct("Running")
            .field("database_key", &self.0.database_key)
            .field("other_id", &self.0.other_id)
            .field("thread_id", &self.0.thread_id)
            .finish()
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Stamp {
    pub durability: Durability,
    pub changed_at: Revision,
}

pub fn stamp(revision: Revision, durability: Durability) -> Stamp {
    Stamp {
        durability,
        changed_at: revision,
    }
}

impl Default for Runtime {
    fn default() -> Self {
        Runtime {
            revisions: [Revision::start(); Durability::LEN],
            revision_cancelled: Default::default(),
            cancellation_count: Default::default(),
            cycle_groups: Default::default(),
            dependency_graph: Default::default(),
            table: Default::default(),
        }
    }
}

impl std::fmt::Debug for Runtime {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fmt.debug_struct("Runtime")
            .field("revisions", &self.revisions)
            .field("revision_cancelled", &self.revision_cancelled)
            .field("dependency_graph", &self.dependency_graph)
            .finish()
    }
}

impl Runtime {
    #[inline]
    pub(crate) fn current_revision(&self) -> Revision {
        self.revisions[0]
    }

    /// Reports that an input with durability `durability` changed.
    /// This will update the 'last changed at' values for every durability
    /// less than or equal to `durability` to the current revision.
    ///
    /// # Panics
    ///
    /// Panics if `durability` is [`Durability::NEVER_CHANGE`].
    pub(crate) fn report_tracked_write(&mut self, durability: Durability) {
        assert_ne!(
            durability,
            Durability::NEVER_CHANGE,
            "never-changing inputs cannot be mutated"
        );
        let new_revision = self.current_revision();
        self.revisions[1..=durability.index()].fill(new_revision);
    }

    /// The revision in which values with durability `d` may have last
    /// changed.  For D0, this is just the current revision. But for
    /// higher levels of durability, this value may lag behind the
    /// current revision. If we encounter a value of durability Di,
    /// then, we can check this function to get a "bound" on when the
    /// value may have changed, which allows us to skip walking its
    /// dependencies.
    #[inline]
    pub(crate) fn last_changed_revision(&self, d: Durability) -> Revision {
        match self.revisions.get(d.index()) {
            Some(&revision) => revision,
            None => never_changed_revision(),
        }
    }

    pub(crate) fn load_cancellation_flag(&self) -> bool {
        self.revision_cancelled.load(Ordering::Acquire)
    }

    pub(crate) fn cycle_groups(&self) -> &cycle_groups::CycleGroups {
        &self.cycle_groups
    }

    pub(crate) fn cancellation_count(&self) -> u8 {
        self.cancellation_count.load(Ordering::Acquire)
    }

    pub(crate) fn set_cancellation_flag(&self) {
        crate::tracing::trace!("set_cancellation_flag");
        self.revision_cancelled.store(true, Ordering::Release);
    }

    pub(crate) fn reset_cancellation_flag(&self) {
        self.revision_cancelled.store(false, Ordering::Release);
    }

    pub(crate) fn bump_cancellation_count(&mut self) -> bool {
        let count = self.cancellation_count.get_mut();
        let Some(next) = count.checked_add(1) else {
            return true;
        };
        *count = next;
        false
    }

    /// Returns the [`Table`] used to store the value of salsa structs
    #[inline]
    pub(crate) fn table(&self) -> &Table {
        &self.table
    }

    pub(crate) fn table_mut(&mut self) -> &mut Table {
        &mut self.table
    }

    /// Increments the "current revision" counter and clears
    /// the cancellation flag.
    ///
    /// This should only be done by the storage when the state is "quiescent".
    pub(crate) fn new_revision(&mut self) -> Revision {
        let r_old = self.current_revision();
        let r_new = r_old.next();
        self.revisions[0] = r_new;
        *self.cancellation_count.get_mut() = 0;
        crate::tracing::info!("new_revision: {r_old:?} -> {r_new:?}");
        r_new
    }

    /// Decides which side of a cross-thread cycle must back out, preferring to keep the
    /// established cycle driver alive.
    ///
    /// The current thread wants `contested_key` (computed by `owner`), which would close a
    /// cycle `owner -> ... -> me` of blocked threads. Walk that chain and classify each
    /// held key: a key whose memo is a current provisional fixpoint state marks its owner
    /// as a *driver* (mid cycle iteration). The driver wins: with "requester always
    /// loses", an unrelated thread that merely holds a query the driver's evaluation
    /// reaches would keep killing the driver, re-forming the cycle over and over (each
    /// re-formation ratchets the inherited iteration stamp until the iteration cap).
    ///
    /// Returns `Some(wake_key)` if the current thread should win: `wake_key` is the key it
    /// holds on the cycle whose waiters must be woken with [`WaitResult::BackOut`].
    /// Returns `None` if the current thread must back out itself (also the safe fallback
    /// whenever the chain cannot be walked, e.g. because it changed concurrently).
    pub(crate) fn cross_cycle_decision(
        &self,
        zalsa: &Zalsa,
        owner: ThreadId,
        contested_key: DatabaseKeyIndex,
        my_root: Option<DatabaseKeyIndex>,
    ) -> Option<DatabaseKeyIndex> {
        let me = thread::current().id();
        let dg = self.dependency_graph.lock();
        dg.cross_cycle_walk(zalsa, self, me, owner, contested_key, my_root)
    }

    /// Block until `other_id` completes executing `database_key`, or return `BlockResult::Cycle`
    /// immediately in case of a cycle.
    ///
    /// `query_mutex_guard` is the guard for the current query's state;
    /// it will be dropped after we have successfully registered the
    /// dependency.
    ///
    /// # Propagating panics
    ///
    /// If the thread `other_id` panics, then our thread is considered
    /// cancelled, so this function will panic with a `Cancelled` value.
    pub(crate) fn block<'a>(
        &'a self,
        database_key: DatabaseKeyIndex,
        other_id: ThreadId,
        query_mutex_guard: SyncGuard<'a>,
    ) -> BlockResult<'a> {
        let thread_id = thread::current().id();
        // Cycle in the same thread.
        if thread_id == other_id {
            return BlockResult::Cycle {
                same_thread: true,
                owner: other_id,
            };
        }

        let dg = self.dependency_graph.lock();

        if dg.depends_on(other_id, thread_id) {
            crate::tracing::debug!(
                "block_on: cycle detected for {database_key:?} in thread {thread_id:?} on {other_id:?}"
            );
            return BlockResult::Cycle {
                same_thread: false,
                owner: other_id,
            };
        }

        BlockResult::Running(Running(Box::new(BlockedOnInner {
            dg,
            query_mutex_guard,
            database_key,
            other_id,
            thread_id,
        })))
    }

    /// Tries to claim ownership of a transferred query where `thread_id` is the current thread and `query`
    /// is the query (that had its ownership transferred) to claim.
    ///
    /// For this operation to be reasonable, the caller must ensure that the sync table lock on `query` is not released
    /// before this operation completes.
    pub(super) fn block_transferred(
        &self,
        query: DatabaseKeyIndex,
        current_id: ThreadId,
    ) -> BlockTransferredResult<'_> {
        let dg = self.dependency_graph.lock();

        let owner_thread = dg.thread_id_of_transferred_query(query, None);

        let Some(owner_thread_id) = owner_thread else {
            // The query transferred its ownership but the owner has since then released the lock.
            return BlockTransferredResult::Released;
        };

        if owner_thread_id == current_id || dg.depends_on(owner_thread_id, current_id) {
            BlockTransferredResult::ImTheOwner
        } else {
            // Lock is owned by another thread, wait for it to be released.
            BlockTransferredResult::OwnedBy(Box::new(BlockOnTransferredOwner {
                dg,
                database_key: query,
                other_id: owner_thread_id,
                thread_id: current_id,
            }))
        }
    }

    /// Invoked when this runtime completed computing `database_key` with
    /// the given result `wait_result`.
    /// This function unblocks any dependent queries and allows them
    /// to continue executing.
    pub(crate) fn unblock_queries_blocked_on(
        &self,
        database_key: DatabaseKeyIndex,
        wait_result: WaitResult,
    ) {
        self.dependency_graph
            .lock()
            .unblock_runtimes_blocked_on(database_key, wait_result);
    }

    /// Unblocks all transferred queries that are owned by `database_key` recursively.
    ///
    /// Invoked when a query completes that has been marked as transfer target (it has
    /// queries that transferred their lock ownership to it) with the given `wait_result`.
    ///
    /// This function unblocks any dependent queries and allows them to continue executing. The
    /// query `database_key` is not unblocked by this function.
    #[cold]
    pub(crate) fn unblock_transferred_queries_owned_by(
        &self,
        database_key: DatabaseKeyIndex,
        wait_result: WaitResult,
    ) {
        self.dependency_graph
            .lock()
            .unblock_runtimes_blocked_on_transferred_queries_owned_by(database_key, wait_result);
    }

    /// Removes the ownership transfer of `query`'s lock if it exists.
    ///
    /// If `query` has transferred its lock ownership to another query, this function will remove that transfer,
    /// so that `query` now owns its lock again.
    #[cold]
    pub(super) fn undo_transfer_lock(&self, query: DatabaseKeyIndex) {
        self.dependency_graph.lock().undo_transfer_lock(query);
    }

    /// Transfers ownership of the lock for `query` to `new_owner_key`.
    ///
    /// For this operation to be reasonable, the caller must ensure that the sync table lock on `query` is not released
    /// and that `new_owner_key` is currently blocked on `query`. Otherwise, `new_owner_key` might
    /// complete before the lock is transferred, leaving `query` locked forever.
    pub(super) fn transfer_lock(
        &self,
        query: DatabaseKeyIndex,
        new_owner_key: DatabaseKeyIndex,
        new_owner_id: SyncOwner,
        guard: SyncGuard,
    ) -> bool {
        let dg = self.dependency_graph.lock();
        DependencyGraph::transfer_lock(
            dg,
            query,
            thread::current().id(),
            new_owner_key,
            new_owner_id,
            guard,
        )
    }

    #[cfg(feature = "persistence")]
    pub(crate) fn deserialize_from(&mut self, other: &mut Runtime) {
        // The only field that is serialized is `revisions`.
        self.revisions = other.revisions;
    }
}

#[cold]
#[inline(never)]
fn never_changed_revision() -> Revision {
    Revision::start()
}

#[cfg(test)]
mod tests {
    use super::Runtime;
    use crate::Revision;

    #[test]
    fn cancellation_count_overflow_requires_revision_bump() {
        let mut runtime = Runtime::default();

        for _ in 0..u8::MAX {
            assert!(!runtime.bump_cancellation_count());
        }

        assert!(runtime.bump_cancellation_count());
        runtime.new_revision();

        assert_eq!(runtime.current_revision(), Revision::start().next());
        assert_eq!(runtime.cancellation_count(), 0);
    }
}
