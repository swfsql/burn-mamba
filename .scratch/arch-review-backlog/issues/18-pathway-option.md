# 18: The pathway is an option of the call, not a tag on the cache

Status: needs-triage

Blocked by: None

From the architecture review of 2026-11-16. Line numbers are those of commit
`7807d31b`. The review compiled nothing.

**Files:** `src/mamba3/cache.rs`, `src/mamba3/double_ssd/cache.rs`,
`src/mamba3/single_ssd/cache.rs`, `src/mamba3/mamba3.rs:1572-1630,1758-1787`,
`src/mamba3/plant_frame.rs:24`, `src/mamba3/ssd_path.rs`,
`src/family/adapters.rs` (`Mamba3Caches::from_slots`),
`src/family/cache_traversal.rs`, `src/family/tests/block_contract.rs:46-48`.

**Problem:** `Mamba3DoubleSsdCache` and `Mamba3SingleSsdCache` have the same fields.
`src/mamba3/cache.rs` says that they have the same values at each call boundary, and
that a cache exists only there. So the cache type holds no state of its own. It only
selects the pathway. The cost of this selector:

- 2 cache structs, the enums `Mamba3Cache` and `Mamba3Caches`, and 8 `From` impls.
- `Mamba3Caches::from_slots` guesses the pathway from slot 0.
- The origin of a packed call converts to the pathway of the cache.
- `plant_frame.rs:24` already uses one cache type for both pathways, and single-SSD
  converts at its entry and at its exit.
- `step` runs the double-SSD recurrence for both variants.
- The header of `src/family/tests/block_contract.rs` records a limit: "No contract
  starts from a double-SSD cache", so "no setting reaches the double-SSD pathway".
- More than ten tests call `.single_ssd().expect(…)` on a returned cache.

**Direction:** One `Mamba3Cache` struct (`ssm_bhpr` and the frame cache).
`Mamba3SsdPath`, the `Options` of the block, holds the pathway beside the SSD path.
The block contracts get settings for double-SSD. The default stays single-SSD.

**ADR:** This contradicts ADR-0003 ("selected by the cache type"). The ADR protects
against a mix of the two accumulators in the middle of a sequence. No cache exists in
the middle of a sequence, so a caller cannot reach that mix. The two pathways stay.
If the user agrees, a new ADR supersedes ADR-0003.

**Cost:** A break of the public API (the cache types and `Mamba3SsdPath`). Saved
caches are run-time state, so no record changes.

**Related:** Ticket 22 gives `Mamba3SsdPath` the struct form, which has a place for
the pathway. `../burn-stack/.scratch/arch-review-backlog/issues/12-one-cache-stack.md`
is smaller after this ticket.
