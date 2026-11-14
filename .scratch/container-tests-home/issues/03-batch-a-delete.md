# 03: Batch A, delete the moved class-marker tests

Status: resolved

Blocked by: `02-first-contracts.md`,
`../burn-stack/.scratch/container-tests-home/issues/03-batch-a-class-markers.md`

Spec: `../spec.md`, sections "The sort rule" and "Coverage". Run only one session on the
two repos.

**What to build:** the class-marker tests of burn-mamba test no container. The coverage
of both repos keeps each item of the record.

- Read the sorted list under `## Comments` of burn-stack ticket 03.
- Delete each test of `src/family/tests/class.rs` that the list marks rule 1 or rule 2.
  Also delete each test that the list marks "3, repeat": the user chose this in burn-stack
  ticket 03, because a kept test of burn-mamba makes its checks. Keep each other test of
  rule 3.
- Add each contract that burn-stack ticket 03 added to the contract tests of ticket 02,
  on each block setting.
- Delete a test file that has no test left, and its `mod` line.

**Pin:** move the burn-stack pin to the pushed commit of burn-stack ticket 03.

**Gate:** run the coverage compare against the record of ticket 01. Both test suites
together must cover each item of the record. Write the result under `## Comments`. If an
item is lost, stop and ask the user. The fix is usually a new contract in burn-stack.

- [x] Each test of rule 1 or 2 in the list is deleted. Each test of rule 3 stays.
- [x] The new contracts run on each block setting.
- [x] The coverage compare lists no lost item of the record.
- [x] `cargo test --lib -- --test-threads=1` passes on `flex`. `scripts/check.sh` passes.

## Comments

### From burn-stack ticket 03

- The sorted list marks each of the 29 tests of `src/family/tests/class.rs` rule 1 or
  "3, repeat". `prime_through_the_enum` is "1, and 3, repeat". No test is rule 2.
- The user chose to delete each "3, repeat" (the section "The rule-3 repeats" of that
  ticket). So no test of `class.rs` stays.
- The new contract is `zero_cache_equals_no_cache`. Add it to the contract tests, on each
  block setting.

### The implementation

- The pin moved from `d8d8d351` to burn-stack `5416d97e`, the pushed commit of its
  ticket 03. The version of burn-stack stays 0.41.0, so there is no version bump.
- The change deletes `src/family/tests/class.rs` (29 tests) and its `mod` line.
- `block_contract.rs` runs `zero_cache_equals_no_cache` on each of the 17 block settings.
- The header of `src/family/tests/mod.rs` stays. It still names the class-marker
  placement, because two tests that stay check it. `prime_and_class_tokens_through_the_enum`
  in `block.rs` checks the positions and the cursors. `packed.rs` checks the opening slots
  of the class latents.
- No other doc of burn-mamba names a deleted test.
- `cargo test --lib -- --test-threads=1` on `flex`: 322 passed, 3 ignored. Before the
  change: 350 passed, 3 ignored. The change deletes 29 tests and adds 1.
- `scripts/check.sh` passed on the pin, with no warning (`tmp/ct03/check.log`). Its
  coverage step gave the burn-mamba totals of the record.

### The red check

The contract test had a real red step: at the old pin, the build failed, because
`block_contract::zero_cache_equals_no_cache` did not exist. Then two temporary breaks ran
in one build (`tmp/ct03/red/breaks-1-2.log`). The tree keeps neither break.

| Break | Settings that fail |
|---|---|
| The `Block::zero_cache` of Mamba-3 gives a state of ones | The 11 Mamba-3 settings and `MambaBlock`, on the forward half |
| The `step` of Mamba-1 with no cache starts from a state of ones | Mamba-1, on the step half |

The 4 Mamba-2 settings passed, because no break changes Mamba-2. Each failure names its
setting and the half of the contract.

A `forward` or a `step` with no cache starts from the inherent `zero_cache` of its family.
So a break inside the inherent `zero_cache` changes both sides of the contract, and the
contract does not fail. The first break is in the `Block` impl, because there the zero
cache can differ from the no-cache path.

### Coverage result

The gate ran with the `[patch]` of burn-stack on, at burn-stack `5416d97e`. The commands
are the four steps of `tmp/coverage-record/record.md`. The script is
`tmp/coverage-record/gate-03.sh`, the reports are in `tmp/coverage-record/after-03/`, and
the log is `tmp/coverage-record/gate-03.log`. The first run passed:

| `src/` of | lib tests of | lines | branches | conditions |
|---|---|---|---|---|
| burn-mamba | burn-mamba | 7713 of 7898 | 189 of 216 | 82 of 108 |
| burn-stack | burn-stack | 3905 of 4441 | 312 of 432 | 128 of 216 |
| burn-stack | burn-mamba | 3303 of 6686 | 203 of 692 | 59 of 346 |
| burn-stack | both | 4126 of 6696 | 327 of 696 | 136 of 348 |

- Step 1: the compare with the record lists no lost item, and no covered item on a
  changed line. The totals of burn-mamba equal those of the record. The sorted list of
  burn-stack ticket 03 asks this gate for an arm of `MambaBlock` that only a
  `family_net_config_*` test runs. Step 1 shows that no such arm exists: the deletion
  loses no item of `src/family/`.
- Step 4: the compare with the record lists no lost item. It lists 13 lines, 2 branches
  and 1 condition on changed lines, in `src/block_contract.rs` (lines 371–386 of the
  record, the condition 385:29). These are the lines of `muon_projections_name_existing_rank_2_weights`
  that burn-stack ticket 05 changed. The gates of tickets 04 and 05 listed the same lines
  and branches. Since burn-stack `65334b6`, the compare also names each condition. So it
  names the condition of the two branches. The gate lost no condition.
- Of the burn-mamba tests, only the deleted tests reached 77 lines, 16 branches and 14
  conditions of burn-stack, in `src/modules/{bidi,layers,network}.rs` and
  `src/utils/class/mod.rs`. The compare of `burn-stack-by-burn-mamba` from `after-04` to
  `after-03` lists them, in
  `tmp/coverage-record/after-03/burn-stack-by-burn-mamba-since-04.md`. The burn-stack
  tests cover each of them, so the union keeps them.

### The review (`tmp/review-ct03-standards.md`, `tmp/review-ct03-spec.md`)

- Fixed:
  - The header of `src/family/tests/mod.rs` keeps its base text. The first form of this
    change deleted the class-marker placement from it, but `block.rs` and `packed.rs`
    still check that property.
  - The red check writes "a state of ones". The glossary term is "state", and "SSM state"
    is _Avoid_.
  - The changed lines of step 4 are lines of the record. The condition 385:29 is a
    covered item on a changed line, not a lost item.
  - The coverage result answers the question of the sorted list about an arm of
    `MambaBlock`.
- Kept:
  - "with the `[patch]` on" and "the union". `tmp/coverage-record/record.md` and the
    tickets 04 and 05 use these words.
- `scripts/check.sh` ran before the fixes. The fixes change only this ticket and the
  header of `src/family/tests/mod.rs`, which went back to its base text.
