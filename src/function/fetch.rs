use crate::cycle::{CycleRecoveryStrategy, IterationStamp};
use crate::function::eviction::EvictionPolicy;
use crate::function::memo::Memo;
use crate::function::execute::DisableLocalCancellationGuard;
use crate::function::sync::ClaimResult;
use crate::function::{Configuration, IngredientImpl, Reentrancy};
use crate::zalsa::{MemoIngredientIndex, Zalsa};
use crate::zalsa_local::{QueryRevisions, ZalsaLocal};
use crate::{Cancelled, DatabaseKeyIndex, Id};

impl<C> IngredientImpl<C>
where
    C: Configuration,
{
    #[inline]
    pub fn fetch<'db>(
        &'db self,
        db: &'db C::DbView,
        zalsa: &'db Zalsa,
        zalsa_local: &'db ZalsaLocal,
        id: Id,
    ) -> &'db C::Output<'db> {
        zalsa.unwind_if_revision_cancelled(zalsa_local);

        let database_key_index = self.database_key_index(id);

        #[cfg(feature = "detailed-trace")]
        let _span = crate::tracing::debug_span!("fetch", query = ?database_key_index).entered();

        // A query that lost a cross-thread cycle race unwinds its entire stack (see
        // `Cancelled::CycleLoser`); the retry must happen where the stack is empty.
        // Only the top-level entry point can catch it.
        let memo = if zalsa_local.query_stack_is_empty() {
            // A cycle group that converged from a non-canonical entry is re-evaluated
            // here, detached, from its minimum member (GODE I-D) before the original
            // query is retried.
            let mut pending_canonical: Option<crate::DatabaseKeyIndex> = None;
            loop {
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    if let Some(canonical) = pending_canonical {
                        // The re-evaluation continues a fixpoint iteration that upstream
                        // completes before honoring local cancellation; defer it for
                        // exactly that scope (dropped before the retry below).
                        let _deferral = DisableLocalCancellationGuard::new(zalsa_local);
                        let ingredient = zalsa.lookup_ingredient(canonical.ingredient_index());
                        // SAFETY: `db` is the database this ingredient belongs to.
                        unsafe {
                            ingredient.fetch_detached(
                                zalsa,
                                crate::database::RawDatabase::from(db),
                                zalsa_local,
                                canonical.key_index(),
                            );
                        }
                    }
                    self.refresh_memo(db, zalsa, zalsa_local, id)
                })) {
                    Ok(memo) => break memo,
                    Err(payload) => {
                        pending_canonical = None;
                        // GODE: any unwind that reaches the top level while this thread
                        // owns a live cycle group means the group's evaluation died
                        // (panic, propagated panic, or a back-out). Discard its
                        // provisional state wholesale: remove every member memo (leaving
                        // the never-computed state) and wake parked threads.
                        if let Some(root) = zalsa_local.current_cycle_group() {
                            let groups = zalsa.runtime().cycle_groups();
                            for member in groups.members_of(root) {
                                zalsa
                                    .lookup_ingredient(member.ingredient_index())
                                    .remove_memo(zalsa, member.key_index());
                            }
                            groups.complete(root);
                            zalsa_local.set_current_cycle_group(None);
                        }
                        match payload.downcast::<Cancelled>() {
                        Ok(cancelled) => match *cancelled {
                            Cancelled::CycleLoser => {
                                crate::tracing::debug!(
                                    "{database_key_index:?}: lost a cross-thread cycle race; retrying from the top level"
                                );
                                zalsa_local.end_cycle_backout();
                                // Give the winning thread time to finalize the cycle.
                                std::thread::yield_now();
                                continue;
                            }
                            Cancelled::CycleCanonicalize { canonical } => {
                                crate::tracing::debug!(
                                    "{database_key_index:?}: canonical cycle re-evaluation from {canonical:?}"
                                );
                                zalsa_local.end_cycle_backout();
                                pending_canonical = Some(canonical);
                                continue;
                            }
                            other => other.throw(),
                        },
                        Err(payload) => std::panic::resume_unwind(payload),
                    }
                    }
                }
            }
        } else {
            self.refresh_memo(db, zalsa, zalsa_local, id)
        };

        // SAFETY: We just refreshed the memo so it is guaranteed to contain a value now.
        let memo_value = unsafe { memo.value.as_ref().unwrap_unchecked() };

        self.eviction.record_use(id);

        let revisions = &memo.header.revisions;
        zalsa_local.report_tracked_read(
            database_key_index,
            revisions.durability,
            revisions.changed_at,
            memo.header.cycle_heads(),
            #[cfg(feature = "accumulator")]
            revisions.accumulated().is_some(),
            #[cfg(feature = "accumulator")]
            &revisions.accumulated_inputs,
        );

        memo_value
    }

    #[inline(always)]
    pub(super) fn refresh_memo<'db>(
        &'db self,
        db: &'db C::DbView,
        zalsa: &'db Zalsa,
        zalsa_local: &'db ZalsaLocal,
        id: Id,
    ) -> &'db Memo<'db, C> {
        let memo_ingredient_index = self.memo_ingredient_index(zalsa, id);

        loop {
            // Keep the hot and cold probes in distinct control-flow blocks. Using `or_else`
            // here can outline both into one function, making hot hits pay for the cold path's
            // stack frame.
            if let Some(memo) = self.fetch_hot(zalsa, id, memo_ingredient_index) {
                return memo;
            }

            if let Some(memo) = self.fetch_cold(zalsa, zalsa_local, db, id, memo_ingredient_index) {
                return memo;
            }
        }
    }

    #[inline(always)]
    fn fetch_hot<'db>(
        &'db self,
        zalsa: &'db Zalsa,
        id: Id,
        memo_ingredient_index: MemoIngredientIndex,
    ) -> Option<&'db Memo<'db, C>> {
        let memo = self.get_memo_from_table_for(zalsa, id, memo_ingredient_index)?;

        memo.value.as_ref()?;

        let database_key_index = self.database_key_index(id);

        let can_shallow_update = memo
            .header
            .shallow_verify_memo(zalsa, database_key_index, true);

        if can_shallow_update.yes() && !memo.header.may_be_provisional() {
            memo.header
                .update_shallow(zalsa, database_key_index, can_shallow_update);

            // SAFETY: memo is present in memo_map and we have verified that it is
            // still valid for the current revision.
            unsafe { Some(self.extend_memo_lifetime(memo)) }
        } else {
            None
        }
    }

    fn fetch_cold<'db>(
        &'db self,
        zalsa: &'db Zalsa,
        zalsa_local: &'db ZalsaLocal,
        db: &'db C::DbView,
        id: Id,
        memo_ingredient_index: MemoIngredientIndex,
    ) -> Option<&'db Memo<'db, C>> {
        let database_key_index = self.database_key_index(id);
        // Try to claim this query: if someone else has claimed it already, go back and start again.
        let claim_guard = match self
            .sync_table
            .try_claim(zalsa, zalsa_local, id, Reentrancy::Allow)
        {
            ClaimResult::Claimed(guard) => guard,
            ClaimResult::Running(blocked_on) => {
                if blocked_on.block_on(zalsa) == crate::runtime::BlockOutcome::BackOut {
                    cycle_loser_unwind(zalsa_local, database_key_index)
                }
                return None;
            }
            ClaimResult::Cycle { same_thread: true, .. } => {
                return Some(self.fetch_cold_cycle(
                    zalsa,
                    zalsa_local,
                    db,
                    id,
                    database_key_index,
                    memo_ingredient_index,
                ));
            }
            ClaimResult::Cycle {
                same_thread: false,
                owner,
                ..
            } => {
                // Joining a cross-thread cycle would let two threads iterate one strongly
                // connected component concurrently, making the fixpoint trace (and with
                // non-monotone recovery functions, the converged values) depend on thread
                // scheduling. One side must back out entirely and retry from the top
                // level. The winner is decided by group roles (see `cross_cycle_walk`):
                // a group owner beats non-group threads, and between two group owners
                // the smaller root key wins — a total order, so the outcome does not
                // depend on which side detected the cycle first.
                let my_root = zalsa_local.current_cycle_group();
                match zalsa.runtime().cross_cycle_decision(
                    zalsa,
                    owner,
                    database_key_index,
                    my_root,
                ) {
                    Some(wake_key) => {
                        crate::tracing::debug!(
                            "{database_key_index:?}: cross-thread cycle; driver wakes {wake_key:?}"
                        );
                        zalsa
                            .runtime()
                            .unblock_queries_blocked_on(wake_key, crate::runtime::WaitResult::BackOut);
                        std::thread::yield_now();
                        return None;
                    }
                    None => cycle_loser_unwind(zalsa_local, database_key_index),
                }
            }
        };

        // GODE: never execute or reuse another thread's in-flight provisional cycle
        // state. Every provisional memo belongs to a registered cycle group; if that
        // group is live and owned by another thread, park on the group until it
        // finalizes (or aborts) and then retry. A provisional memo without a live group
        // is a leftover of a real panic and falls through to normal (failing)
        // verification and re-execution.
        if let Some(old_memo) = self.get_memo_from_table_for(zalsa, id, memo_ingredient_index) {
            if old_memo.header.may_be_provisional()
                && old_memo.header.verified_at.load() == zalsa.current_revision()
                && !old_memo.header.cycle_heads().is_empty()
            {
                if let Some(group) = zalsa.runtime().cycle_groups().group_of(database_key_index) {
                    if group.owner != crate::sync::thread::current().id() {
                        // Park by blocking on the group root's claim, which the owner
                        // holds for the entire evaluation. Unlike a condvar, this
                        // registers an edge in the dependency graph, so a deadlock
                        // (the owner needing a query this thread holds) is detected and
                        // resolved through the ordinary cross-thread cycle rules instead
                        // of hanging.
                        drop(claim_guard);
                        let root = group.root;
                        let root_ingredient = zalsa.lookup_ingredient(root.ingredient_index());
                        match root_ingredient.wait_for(zalsa, root.key_index()) {
                            crate::function::WaitForResult::Running(running) => {
                                if running.block_on(zalsa) == crate::runtime::BlockOutcome::BackOut
                                {
                                    cycle_loser_unwind(zalsa_local, database_key_index)
                                }
                            }
                            crate::function::WaitForResult::Cycle { .. } => {
                                // The group (transitively) waits on a query this thread
                                // holds: the non-group side yields.
                                cycle_loser_unwind(zalsa_local, database_key_index)
                            }
                            crate::function::WaitForResult::Available => {
                                // Between the registry lookup and the claim probe the
                                // group completed (or its owner is between claims);
                                // retry.
                                std::thread::yield_now();
                            }
                        }
                        return None;
                    }
                }
            }
        }

        // Now that we've claimed the item, check again to see if there's a "hot" value.
        let opt_old_memo = self.get_memo_from_table_for(zalsa, id, memo_ingredient_index);

        if let Some(old_memo) = opt_old_memo {
            if old_memo.value.is_some()
                && old_memo
                    .header
                    .verify_memo(db.into(), &claim_guard, C::CYCLE_STRATEGY, true)
            {
                // SAFETY: memo is present in memo_map and we have verified that it is
                // still valid for the current revision.
                return unsafe { Some(self.extend_memo_lifetime(old_memo)) };
            }
        }

        self.execute(db, claim_guard, opt_old_memo)
    }

    #[cold]
    #[inline(never)]
    fn fetch_cold_cycle<'db>(
        &'db self,
        zalsa: &'db Zalsa,
        zalsa_local: &'db ZalsaLocal,
        db: &'db C::DbView,
        id: Id,
        database_key_index: DatabaseKeyIndex,
        memo_ingredient_index: MemoIngredientIndex,
    ) -> &'db Memo<'db, C> {
        // no provisional value; create/insert/return initial provisional value
        match C::CYCLE_STRATEGY {
            // SAFETY: We do not access the query stack reentrantly.
            CycleRecoveryStrategy::Panic => unsafe {
                zalsa_local.with_query_stack_unchecked(|stack| {
                    panic!(
                        "dependency graph cycle when querying {database_key_index:#?}, \
                    set cycle_fn/cycle_initial to fixpoint iterate.\n\
                    Query stack:\n{stack:#?}",
                    );
                })
            },
            CycleRecoveryStrategy::Fixpoint | CycleRecoveryStrategy::FallbackImmediate => {
                // GODE: the first cycle hit on this thread forms a cycle group; nested or
                // repeated hits during the same evaluation join it. The group records its
                // members so an aborted or panicked evaluation can be discarded wholesale,
                // and so other threads can park on the group instead of observing its
                // provisional state.
                let group_root = match zalsa_local.current_cycle_group() {
                    Some(root) => Some(root),
                    None => {
                        let registered = zalsa
                            .runtime()
                            .cycle_groups()
                            .register(database_key_index, crate::sync::thread::current().id());
                        if registered.is_some() {
                            zalsa_local.set_current_cycle_group(Some(database_key_index));
                            Some(database_key_index)
                        } else {
                            // A live group already covers this key. This should not happen
                            // while we hold the (same-thread) claim; fall back to legacy
                            // behavior rather than corrupting the registry.
                            None
                        }
                    }
                };
                if let Some(root) = group_root {
                    // The contested key is a cycle head by construction.
                    zalsa.runtime().cycle_groups().add_member(
                        root,
                        database_key_index,
                        [database_key_index],
                    );
                }

                let cancellation_count = zalsa.runtime().cancellation_count();
                // check if there's a provisional value for this query
                // Note we don't `validate_may_be_provisional` the memo here as we want to reuse an
                // existing provisional memo if it exists
                let memo_guard = self.get_memo_from_table_for(zalsa, id, memo_ingredient_index);
                if let Some(memo) = &memo_guard {
                    let revisions = &memo.header.revisions;
                    // Ideally, we'd use the last provisional memo even if it wasn't a cycle head in the last iteration
                    // but that would require inserting itself as a cycle head, which either requires clone
                    // on the value OR a concurrent `Vec` for cycle heads.
                    if memo.header.verified_at.load() == zalsa.current_revision()
                        && memo.value.is_some()
                        && revisions.iteration().cancellation_count() == cancellation_count
                        && revisions.cycle_heads().contains(&database_key_index)
                    {
                        revisions
                            .cycle_heads()
                            .remove_all_except(database_key_index);

                        crate::tracing::debug!(
                            "hit cycle at {database_key_index:#?}, \
                                returning last provisional value: {:#?}",
                            revisions
                        );

                        // SAFETY: memo is present in memo_map.
                        return unsafe { self.extend_memo_lifetime(memo) };
                    }
                }

                crate::tracing::debug!(
                    "hit cycle at {database_key_index:#?}, \
                    inserting and returning fixpoint initial value"
                );

                let iteration = memo_guard
                    .and_then(|old_memo| {
                        let revisions = &old_memo.header.revisions;
                        if old_memo.header.verified_at.load() == zalsa.current_revision()
                            && old_memo.value.is_some()
                            && revisions.iteration().cancellation_count() == cancellation_count
                        {
                            Some(revisions.iteration())
                        } else {
                            None
                        }
                    })
                    .unwrap_or_else(|| IterationStamp::initial(cancellation_count));
                let revisions = QueryRevisions::fixpoint_initial(database_key_index, iteration);

                let initial_value = C::cycle_initial(db, id, C::id_to_input(zalsa, id));
                self.insert_memo(
                    zalsa,
                    id,
                    Memo::new(Some(initial_value), zalsa.current_revision(), revisions),
                    memo_ingredient_index,
                )
            }
        }
    }
}

/// Unwinds the current thread out of all of its queries because it lost a cross-thread
/// cycle race. The local cancellation token is set first so that the claims released
/// during unwinding report `WaitResult::Cancelled` (waiters retry) rather than
/// `WaitResult::Panicked` (waiters propagate the panic).
#[cold]
#[inline(never)]
pub(super) fn cycle_loser_unwind(zalsa_local: &ZalsaLocal, database_key_index: DatabaseKeyIndex) -> ! {
    crate::tracing::debug!(
        "{database_key_index:?}: cross-thread cycle detected; unwinding as the losing side"
    );
    zalsa_local.begin_cycle_backout();
    Cancelled::CycleLoser.throw()
}
