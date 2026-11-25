# 06: Batch D, delete the moved tests of `MambaBlock`, the untied copies, packed rows, the H-Net, the captured step, the configs and the reset alignment

Status: resolved

Blocked by: `02-first-contracts.md`,
`../burn-stack/.scratch/container-tests-home/issues/06-batch-d-block-and-the-rest.md`

Spec: `../spec.md`, sections "The sort rule" and "Coverage". Run only one session on the
two repos.

**What to build:** the burn-mamba tests of this batch test no container. The coverage of
both repos keeps each item of the record.

- Read the sorted list under `## Comments` of burn-stack ticket 06.
- Delete each test of `block.rs`, `untied.rs`, `packed.rs`, `hnet.rs`, `capture.rs`,
  `config.rs` and `reset_align.rs` in `src/family/tests/` that the list marks rule 1 or
  rule 2. Keep each test of rule 3.
- Add each contract that burn-stack ticket 06 added to the contract tests of ticket 02,
  on each block setting.
- Delete a test file that has no test left, and its `mod` line.
- The header of `src/family/tests/mod.rs` says that these tests check the containers
  against real blocks. Rewrite it for what stays: the block contracts on each setting,
  and the tests that only Mamba needs.
- After the gate, delete the paragraph about the gate of the move in the "Coverage gate"
  section of `docs/agents/testing.md`. The record of ticket 01 goes when tickets 03–06
  are resolved.

**Pin:** move the burn-stack pin to the pushed commit of burn-stack ticket 06.

**Gate:** run the coverage compare against the record of ticket 01. Both test suites
together must cover each item of the record. Write the result under `## Comments`. If an
item is lost, stop and ask the user. The fix is usually a new contract in burn-stack.

- [x] Each test of rule 1 or 2 in the list is deleted. Each test of rule 3 stays.
- [x] The new contracts run on each block setting.
- [x] The header of the family tests and the File Map line describe what stays.
- [x] The coverage compare lists no lost item of the record. Its one lost line is a
      closing brace with no coverage region (see "Coverage result").
- [x] `docs/agents/testing.md` has no paragraph about the gate of the move.
- [x] `cargo test --lib -- --test-threads=1` passes on `flex`. `scripts/check.sh` passes.

## Comments

### From ticket 02 (the review of the contract tests)

- `src/family/tests/block_contract.rs` has its own copies of the config builders
  `m1`/`m2`/`m3` of `block.rs`. If `block.rs` keeps tests of rule 3, put one copy in
  `src/family/tests/mod.rs` for both files.
- The "limits" section of the header of `block_contract.rs` names `capture.rs` as the
  test that compares each cache field directly. If this batch moves or deletes that
  test, update that line.

### The implementation

- Commit `0724e3e6` moved the pin to burn-stack `cf08a89`, the first pushed commit of
  its ticket 06. The gate of this ticket added a follow-up commit to burn-stack ticket 06
  (see "Coverage result"). The pin moves to that pushed commit, `f2927ae`. The version
  of burn-stack stays 0.41.0, so there is no version bump.
- The deletions follow the sorted list of burn-stack ticket 06. 20 tests go, and 23
  tests of rule 3 stay:

  | File | Deleted | Stays |
  |---|---|---|
  | `block.rs` | `record_round_trip`, `cache_stack_and_traversal`, `captured_step_over_flat_caches`, `stack_knobs`, `untied_through_the_enum`, `muon_plan_matches_the_family_type` | the other 10 tests |
  | `untied.rs` | the three `*_untied_stack_equals_the_stack_of_its_application_views`, `mamba3_untied_tail_trains_one_copy_per_application` | `mamba3_layout_follows_the_untie_list` |
  | `packed.rs` | all 4. The change deletes the file and its `mod` line. | none |
  | `hnet.rs` | all 4. The change deletes the file and its `mod` line. | none |
  | `capture.rs` | `mamba2_assign_moves_every_field`, `mamba1_assign_moves_every_field` | the two Mamba-3 tests |
  | `config.rs` | none | all 5 |
  | `reset_align.rs` | none | all 5 |

- `cache_stack_and_traversal` was rule 2 and rule 3. Its rule-3 part is now two small
  tests in `block.rs`. `the_flat_caches_keep_empty_slots` checks that `MambaCaches`
  keeps a slot of `None`. `the_zero_cache_of_mamba3_is_single_ssd` checks the zero
  cache of a Mamba-3 block in the enum, through `Block::zero_cache`.
- `capture.rs` and `untied.rs` now hold only Mamba-3 tests. So their `mod` lines have
  the gate `mamba3`, and `capture.rs` has no inner `mod mamba3`.
- `block_contract.rs` runs the four new contracts on each block setting, with the bound
  `Block::Caches: CacheTensors`. It has 4 new untied settings. They come from the probe
  of burn-stack ticket 06 and from the deleted tests:
  - Mamba-1,
  - Mamba-2 on `Minimal`,
  - Mamba-3 on `Minimal`, with `mimo_rank` 2 and the output norm,
  - `MambaBlock` over Mamba-3, with `DtBias` and `D` untied, on the default path.
- The `MambaBlock` setting runs on each enabled family, not only on the first. The
  first gate run chose this (see "Coverage result"). So the settings are 23.
- `src/family/tests/mod.rs` holds the one copy of `D_MODEL`, `m1`, `m2_of_width`, `m2`
  and `m3`, for `block.rs`, `block_contract.rs` and `untied.rs`. The `mamba3_config` of
  `untied.rs` was `m3` with the defaults of `ngroups` and `mimo_rank`.
- The docs that describe what stays:
  - the header of `src/family/tests/mod.rs`, and the File Map line of
    `src/family/tests/`,
  - the seam row "Block contracts" of `docs/agents/testing.md`,
  - the headers of `block.rs`, `untied.rs`, `capture.rs` and `config.rs`,
  - the "limits" line of `block_contract.rs`.

  A doc in `config.rs` named `hnet.rs`. It now names `hnet_mixes_stage_and_main_family`.
- The lib tests are 311. The family tests are 40, and they pass on `flex`.

### The red check

One run with four temporary breaks in `src/`, reverted after the run:

| Break | Test that fails | Setting in the message |
|---|---|---|
| `Mamba2::untied_params` does not name `d_h` | `untied_stack_equals_the_stack_of_its_application_views` | `Mamba-2, untied, Minimal(Some(4))` |
| `MambaBlock::untied_params` gives no parameter for Mamba-3 | `untied_stack_equals_the_stack_of_its_application_views` | `MambaBlock, untied, Mamba3(SerialRecalculated(None))` |
| `MambaCaches::from_slots` drops each slot of `None` | `the_flat_caches_keep_empty_slots` | — |
| The `Block::zero_cache` of Mamba-3 gives a double-SSD cache | `the_zero_cache_of_mamba3_is_single_ssd` | — |

The untied contract gave one line for each broken setting, and no line for the others.

### Coverage result

The gate ran with the `[patch]` of burn-stack on. The commands are the four steps of
`tmp/coverage-record/record.md`. The script is `tmp/coverage-record/gate-06.sh`, and the
reports are in `tmp/coverage-record/after-06/`.

The first run (`tmp/coverage-record/gate-06-first.log`) lost these items:

- burn-mamba, lines 183–184 and 203 of `src/family/block.rs`: the Mamba-2 arms of
  `MambaBlock::block_forward_packed` and `MambaBlock::reset_align`. Only
  `packed_mamba2_matches_alone` reached them. The `MambaBlock` setting of the contracts
  ran only on Mamba-3. The user chose: the `MambaBlock` setting runs on each enabled
  family.
- burn-stack, the branch `157:9 false` of `src/optim/spec.rs` and its condition:
  `is_whole_muon` on a tiled spec. Only `MuonPlan::build` with an untied weight reaches
  it, and only `mamba3_untied_tail_trains_one_copy_per_application` did that build. The
  user chose rule 1: `muon_updates_an_untied_projection_one_copy_at_a_time` of burn-stack
  also runs one training step through the plan. The comments of burn-stack ticket 06
  give the change and its red check.
- burn-stack, line 232 of `src/block_contract.rs`. It is the closing brace with no
  coverage region that the comments of burn-stack ticket 06 record. It is not lost code.

The final run (`tmp/coverage-record/gate-06.log`) is on both fixes:

| `src/` of | lib tests of | lines | branches | conditions |
|---|---|---|---|---|
| burn-mamba | burn-mamba | 7729 of 7898 | 189 of 216 | 82 of 108 |
| burn-stack | burn-stack | 4088 of 4626 | 317 of 438 | 130 of 219 |
| burn-stack | burn-mamba | 3121 of 6871 | 191 of 698 | 52 of 349 |
| burn-stack | both | 4309 of 6881 | 331 of 702 | 137 of 351 |

- Step 1 lists no lost item and no item on a changed line.
- Step 4 lists one lost line: line 232 of `src/block_contract.rs`, the closing brace
  above. It lists 32 lines, 4 branches and 2 conditions on changed lines, all in
  `src/block_contract.rs`. They are the items that burn-stack tickets 05 and 06 changed
  and checked:
  - ticket 05: 13 lines, 2 branches and 1 condition of
    `muon_projections_name_existing_rank_2_weights`,
  - ticket 06: 19 lines, 2 branches and 1 condition.

The record has 7713 of 7898 lines for burn-mamba, and 4057 of 6627 lines for burn-stack
by both suites. Tickets 03–06 are now all resolved, so the record and its `.discard` tag
go to `/tmp`. The change deletes the paragraph about the gate of the move from
`docs/agents/testing.md`.

### The review

Four reports, one for each axis and each repo: `tmp/review-ct06-standards.md`,
`tmp/review-ct06-spec.md`, `tmp/review-ct06-bs-standards.md` and
`tmp/review-ct06-bs-spec.md`. The fixes:

- burn-stack, `muon_updates_an_untied_projection_one_copy_at_a_time`: the training step
  has a learning rate of 1, so the red check fails also under `dev-f16`. The copies must
  part after the step, as in the deleted test. One check compares each copy, for the
  `Segmented` step and for the training step. The comments of burn-stack ticket 06 give
  the red check in fp32 and fp16.
- burn-stack, the docs: the fix deletes the seam row "Containers with real blocks". The
  line of `CLAUDE.md` about the tests of burn-mamba names the block contracts. The spec
  says: "The test-seam tables of both repos change with the module and with the
  burn-mamba ticket."
- The header of `block_contract.rs` said that the untied Mamba-3 setting unties each
  tensor that it can untie. It does not untie the tensors of the Kalman gain and of the
  tropical readout. The deleted test did not untie them either. The header now names
  what each untied setting unties.
- The header of `capture.rs` said that both contracts check the traversal through the
  outputs. `cache_stack_moves_every_slot` compares the slots through the traversal. The
  header also named a capture prototype that the repo does not have. It now names the
  capture of the examples.
- `config.rs`: `hnet_mixes_stage_and_main_family` runs one pair of families, not each
  pair.
- One phrase for one concept: "what only Mamba needs". The header of the family tests,
  the seam row and the File Map line use it.
- `mamba_block_settings` has one list of the settings, with no second tuple.
- The numbers of the changed lines in "Coverage result" now include the conditions.

Not changed:

- The untied Mamba-3 setting does not get the Kalman gain and the tropical readout. The
  spec runs the suite on each setting that a family test uses, and no deleted test
  untied those tensors.
- The names `m1`, `m2` and `m3` stay, as the comment of ticket 02 names them.
- `capture.rs` keeps its own width of 16. Its config is not a shared config.
