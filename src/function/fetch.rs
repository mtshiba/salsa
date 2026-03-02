use crate::cycle::{CycleRecoveryStrategy, IterationCount};
use crate::function::eviction::EvictionPolicy;
use crate::function::memo::Memo;
use crate::function::sync::ClaimResult;
use crate::function::{Configuration, IngredientImpl, Reentrancy};
use crate::zalsa::{MemoIngredientIndex, Zalsa};
use crate::zalsa_local::{QueryRevisions, ZalsaLocal};
use crate::{tracing, DatabaseKeyIndex, Id};

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

        #[cfg(debug_assertions)]
        let _span = crate::tracing::debug_span!("fetch", query = ?database_key_index).entered();

        let memo = self.refresh_memo(db, zalsa, zalsa_local, id);

        // FIXPOINT JACOBI: Return the previous-iteration snapshot for Fixpoint-strategy
        // provisional queries. This ensures every Fixpoint query in a cycle sees the same
        // snapshot values regardless of execution order, making fixpoint iteration fully
        // deterministic.
        //
        // Non-Fixpoint (Panic) queries return their fresh table values. Their values are
        // computed from Fixpoint snapshots (deterministic inputs), so they are also
        // deterministic. Returning fresh values instead of snapshots avoids premature
        // convergence in mixed Fixpoint/Panic cycles.
        //
        // Snapshots are: cycle_initial at iteration 1 (set in fetch_cold/execute.rs),
        // previous iteration's table value at iteration 2+ (set in fetch_cold).
        // If no snapshot exists (iteration 0, SCC discovery), fall through to table value.
        //
        // Always report METADATA (cycle_heads, durability, changed_at) from the LATEST memo,
        // not the snapshot. Snapshot memos have stale cycle_heads with old iteration counts.
        // Finalized memos (may_be_provisional=false) skip this path entirely.
        let memo_value = if zalsa_local.is_jacobi_mode()
            && memo.may_be_provisional()
            && C::CYCLE_STRATEGY == CycleRecoveryStrategy::Fixpoint
        {
            let memo_ingredient_index = self.memo_ingredient_index(zalsa, id);
            if let Some(snapshot) =
                self.get_jacobi_snapshot(zalsa, id, memo_ingredient_index)
            {
                // SAFETY: Snapshot was saved before re-execution, guaranteed to have a value.
                unsafe { snapshot.value.as_ref().unwrap_unchecked() }
            } else {
                // SAFETY: We just refreshed the memo so it is guaranteed to contain a value now.
                unsafe { memo.value.as_ref().unwrap_unchecked() }
            }
        } else {
            // SAFETY: We just refreshed the memo so it is guaranteed to contain a value now.
            unsafe { memo.value.as_ref().unwrap_unchecked() }
        };

        self.eviction.record_use(id);

        // Always report from the latest memo (not the snapshot) to get correct cycle_heads.
        zalsa_local.report_tracked_read(
            database_key_index,
            memo.revisions.durability,
            memo.revisions.changed_at,
            memo.cycle_heads(),
            #[cfg(feature = "accumulator")]
            memo.revisions.accumulated().is_some(),
            #[cfg(feature = "accumulator")]
            &memo.revisions.accumulated_inputs,
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
            if let Some(memo) = self
                .fetch_hot(zalsa, id, memo_ingredient_index)
                .or_else(|| self.fetch_cold(zalsa, zalsa_local, db, id, memo_ingredient_index))
            {
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

        let can_shallow_update = self.shallow_verify_memo(zalsa, database_key_index, memo);

        if can_shallow_update.yes() && !memo.may_be_provisional() {
            self.update_shallow(zalsa, database_key_index, memo, can_shallow_update);

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
                let _ = blocked_on.block_on(zalsa);
                return None;
            }
            ClaimResult::Cycle { .. } => {
                return Some(self.fetch_cold_cycle(
                    zalsa,
                    zalsa_local,
                    db,
                    id,
                    database_key_index,
                    memo_ingredient_index,
                ));
            }
        };

        // Now that we've claimed the item, check again to see if there's a "hot" value.
        let opt_old_memo = self.get_memo_from_table_for(zalsa, id, memo_ingredient_index);

        if let Some(old_memo) = opt_old_memo {
            if old_memo.value.is_some() {
                let can_shallow_update =
                    self.shallow_verify_memo(zalsa, database_key_index, old_memo);
                if can_shallow_update.yes()
                    && self.validate_may_be_provisional(
                        zalsa,
                        zalsa_local,
                        database_key_index,
                        old_memo,
                    )
                {
                    self.update_shallow(zalsa, database_key_index, old_memo, can_shallow_update);

                    // SAFETY: memo is present in memo_map and we have verified that it is
                    // still valid for the current revision.
                    return unsafe { Some(self.extend_memo_lifetime(old_memo)) };
                }

                // JACOBI: Save snapshot before re-execution so we can return
                // the previous iteration's value to other queries.
                if zalsa_local.is_jacobi_mode()
                    && old_memo.may_be_provisional()
                    && old_memo.verified_at.load() == zalsa.current_revision()
                {
                    if self.get_jacobi_snapshot(zalsa, id, memo_ingredient_index).is_none() {
                        if C::CYCLE_STRATEGY == CycleRecoveryStrategy::Fixpoint {
                            // Fixpoint queries: use cycle_initial as snapshot for
                            // deterministic iteration 1.
                            tracing::debug!(
                                "{database_key_index:?}: saving initial Jacobi snapshot (cycle_initial)"
                            );
                            self.set_jacobi_initial_snapshot(db, zalsa, id, memo_ingredient_index);
                        } else {
                            // Non-Fixpoint queries (Panic, FallbackImmediate) that participate
                            // in a cycle: save old memo as snapshot for convergence tracking.
                            tracing::debug!(
                                "{database_key_index:?}: saving Jacobi snapshot (old memo)"
                            );
                            self.set_jacobi_snapshot(zalsa, id, memo_ingredient_index, old_memo);
                        }
                    } else {
                        tracing::debug!(
                            "{database_key_index:?}: saving Jacobi snapshot before re-execution"
                        );
                        self.set_jacobi_snapshot(zalsa, id, memo_ingredient_index, old_memo);
                    }
                }

                let verify_result = self.deep_verify_memo(db, zalsa, old_memo, database_key_index);

                if verify_result.is_unchanged() {
                    // SAFETY: memo is present in memo_map and we have verified that it is
                    // still valid for the current revision.
                    return unsafe { Some(self.extend_memo_lifetime(old_memo)) };
                }
            }
        }

        let result = self.execute(db, claim_guard, opt_old_memo);

        // JACOBI: After re-execution, check if the non-head participant converged.
        // Only check Fixpoint-strategy queries: non-Fixpoint queries (e.g. Panic)
        // may create tracked structs whose Ids oscillate across iterations,
        // causing spurious non-convergence. Fixpoint queries define explicit
        // convergence semantics via cycle_initial/cycle_fn.
        if zalsa_local.is_jacobi_mode() && C::CYCLE_STRATEGY == CycleRecoveryStrategy::Fixpoint {
            if let Some(snapshot) = self.get_jacobi_snapshot(zalsa, id, memo_ingredient_index) {
                if let Some(new_memo) =
                    self.get_memo_from_table_for(zalsa, id, memo_ingredient_index)
                {
                    if let (Some(old_v), Some(new_v)) =
                        (snapshot.value.as_ref(), new_memo.value.as_ref())
                    {
                        if !C::values_equal(old_v, new_v) {
                            tracing::debug!(
                                "{database_key_index:?}: Jacobi non-head value changed"
                            );
                            zalsa_local.set_jacobi_all_converged(false);
                        }
                    }
                }
            }
        }

        result
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
                // check if there's a provisional value for this query
                // Note we don't `validate_may_be_provisional` the memo here as we want to reuse an
                // existing provisional memo if it exists
                let memo_guard = self.get_memo_from_table_for(zalsa, id, memo_ingredient_index);
                if let Some(memo) = &memo_guard {
                    // Ideally, we'd use the last provisional memo even if it wasn't a cycle head in the last iteration
                    // but that would require inserting itself as a cycle head, which either requires clone
                    // on the value OR a concurrent `Vec` for cycle heads.
                    if memo.verified_at.load() == zalsa.current_revision()
                        && memo.value.is_some()
                        && memo.revisions.cycle_heads().contains(&database_key_index)
                    {
                        memo.revisions
                            .cycle_heads()
                            .remove_all_except(database_key_index);

                        crate::tracing::debug!(
                            "hit cycle at {database_key_index:#?}, \
                                returning last provisional value: {:#?}",
                            memo.revisions
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
                        if old_memo.verified_at.load() == zalsa.current_revision()
                            && old_memo.value.is_some()
                        {
                            Some(old_memo.revisions.iteration())
                        } else {
                            None
                        }
                    })
                    .unwrap_or(IterationCount::initial());
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
