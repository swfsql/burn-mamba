# 04: Batch B, delete the moved tests of `Layers`, the gradient horizon, `Layer` and the bidirectional stacks

Status: resolved

Blocked by: `02-first-contracts.md`,
`../burn-stack/.scratch/container-tests-home/issues/04-batch-b-layers.md`

Spec: `../spec.md`, sections "The sort rule" and "Coverage". Run only one session on the
two repos.

**What to build:** the burn-mamba tests of this batch test no container. The coverage of
both repos keeps each item of the record.

- Read the sorted list under `## Comments` of burn-stack ticket 04.
- Delete each test of `layers.rs`, `layer.rs` and `bidi.rs` in `src/family/tests/` that
  the list marks rule 1 or rule 2. Keep each test of rule 3.
- The memory probe moved to burn-stack. Delete it here.
- Add each contract that burn-stack ticket 04 added to the contract tests of ticket 02,
  on each block setting.
- Delete a test file that has no test left, and its `mod` line.

**Pin:** move the burn-stack pin to the pushed commit of burn-stack ticket 04.

**Gate:** run the coverage compare against the record of ticket 01. Both test suites
together must cover each item of the record. Write the result under `## Comments`. If an
item is lost, stop and ask the user. The fix is usually a new contract in burn-stack.

- [x] Each test of rule 1 or 2 in the list is deleted. Each test of rule 3 stays.
- [x] The new contracts run on each block setting.
- [x] The coverage compare lists no lost item of the record.
- [x] `cargo test --lib -- --test-threads=1` passes on `flex`. `scripts/check.sh` passes.

## Comments

### From burn-stack ticket 04

- The sorted list marks each of the 25 tests rule 1. No test is rule 2 or rule 3.
- burn-stack ticket 04 added no contract (its section "No new contract"). So the contract
  tests of ticket 02 get no new contract.
- The memory probe `grad_horizon_memory_probe` is in
  `../burn-stack/src/reference/tests/grad_horizon.rs`, over a wide `RefBlock`.

### The implementation

- The pin moved from `a236fe3f` to burn-stack `d8d8d351`, the pushed commit that resolves
  its ticket 04. The version of burn-stack stays 0.41.0, so there is no version bump.
- `src/family/tests/layers.rs` (18 tests, with the memory probe), `layer.rs` (5 tests)
  and `bidi.rs` (2 tests) held only tests of rule 1. The three files and their `mod` lines
  are deleted.
- The header of `src/family/tests/mod.rs` names the properties of the files that stay. It
  does not name the gradient reachability across a `grad_horizon` cut.
- No other doc of burn-mamba names a deleted test or the probe.
- This ticket adds no test, so it has no red check.
- `cargo test --lib -- --test-threads=1` on `flex`: 350 passed, 3 ignored. Before the
  change: 374 passed, 4 ignored. The deletion removes 24 tests and the ignored probe.
- `scripts/check.sh` passed on the pin, with no warning. Its coverage step gave the
  burn-mamba totals of the record.

### Coverage result

The gate ran with the `[patch]` of burn-stack on, at burn-stack `d8d8d351`. The commands
are the four steps of `tmp/coverage-record/record.md`. The script is
`tmp/coverage-record/gate-04.sh`, the reports are in `tmp/coverage-record/after-04/`, and
the log is `tmp/coverage-record/gate-04.log`. The first run passed:

| `src/` of | lib tests of | lines | branches | conditions |
|---|---|---|---|---|
| burn-mamba | burn-mamba | 7713 of 7898 | 189 of 216 | 82 of 108 |
| burn-stack | burn-stack | 3867 of 4415 | 303 of 432 | 121 of 216 |
| burn-stack | burn-mamba | 3354 of 6660 | 219 of 692 | 73 of 346 |
| burn-stack | both | 4100 of 6670 | 326 of 696 | 135 of 348 |

- Step 1: the compare with the record lists no lost item, and no covered item on a
  changed line. The totals of burn-mamba equal those of the record.
- Step 4: the compare with the record lists no lost item. It lists 13 lines and 2
  branches on changed lines, in `src/block_contract.rs` (lines 371–386). These are the
  lines that burn-stack ticket 05 changed. The gate of ticket 05 listed the same items.
  The totals of both suites equal those of the gate of ticket 05.
- No burn-mamba test except the deleted ones reached 49 lines and 8 branches of
  burn-stack, in `src/modules/{bidi,layer,layers,mlp}.rs` and `src/utils/schedule/mod.rs`.
  The compare of `burn-stack-by-burn-mamba` from `after-05` to `after-04` lists them, in
  `tmp/coverage-record/after-04/burn-stack-by-burn-mamba-since-05.md`. The burn-stack tests
  cover each of them, so the union keeps them.

### The review (`tmp/review-ct04-standards.md`, `tmp/review-ct04-spec.md`)

- Fixed:
  - The header of `src/family/tests/mod.rs` has no semicolon and no passive in the edited
    sentence. It uses the glossary term "Muon plan", and "properties" in place of
    "contracts", because "block contract" is a term of burn-stack.
  - The last bullet of the coverage result had two readings. Its compare had no saved
    output. The output is now in `after-04/`.
  - The status and the checkboxes of this ticket.
- Kept:
  - The first sentence of the header of `src/family/tests/mod.rs`, the File Map line of
    `src/family/tests/` and the seam row. Ticket 06 rewrites them for what stays.
- Not a defect:
  - The spec review said that the compares check the conditions only as totals. That is
    false. `coverage.py` counts a condition as covered when the tests cover its two
    outcomes. The compare checks each outcome of each condition of the base. So a lost
    condition shows as a lost branch, at the span of the condition.
  - A test showed it. `tmp/ct04/lost_condition_demo.py` sets to zero the false outcome of
    one covered condition (`src/block_contract.rs` 165:16) in a copy of
    `after-04/burn-stack-by-both.json`. The compare of the copy lists 1 lost branch, and
    134 of 348 conditions in place of 135. The compares of this gate list no lost branch,
    so they lose no condition.
  - The compare does not name a lost condition as a condition. The ticket
    `../burn-stack/.scratch/coverage-lost-conditions/issues/01-the-compare-names-each-lost-condition.md`
    adds that.
