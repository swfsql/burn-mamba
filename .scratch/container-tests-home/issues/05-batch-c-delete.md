# 05: Batch C, delete the moved tests of Multi-Gate Residuals and the optimizer

Status: resolved

Blocked by: `02-first-contracts.md`,
`../burn-stack/.scratch/container-tests-home/issues/05-batch-c-multi-gate-and-optimizer.md`

Spec: `../spec.md`, sections "The sort rule" and "Coverage". Run only one session on the
two repos.

**What to build:** the burn-mamba tests of this batch test no container and no module of
burn-stack. The coverage of both repos keeps each item of the record.

- Read the sorted list under `## Comments` of burn-stack ticket 05.
- Delete each test of `multi_gate.rs` and `optim.rs` in `src/family/tests/` that the
  list marks rule 1 or rule 2. Keep each test of rule 3.
- Add each contract that burn-stack ticket 05 added to the contract tests of ticket 02,
  on each block setting.
- Delete a test file that has no test left, and its `mod` line.

**Pin:** move the burn-stack pin to the pushed commit of burn-stack ticket 05.

**Gate:** run the coverage compare against the record of ticket 01. Both test suites
together must cover each item of the record. Write the result under `## Comments`. If an
item is lost, stop and ask the user. The fix is usually a new contract in burn-stack.

- [x] Each test of rule 1 or 2 in the list is deleted. Each test of rule 3 stays.
- [x] The new contracts run on each block setting.
- [x] The coverage compare lists no lost item of the record.
- [x] `cargo test --lib -- --test-threads=1` passes on `flex`. `scripts/check.sh` passes.

## Comments

### From burn-stack ticket 05

- The new contract is `muon_projections_select_only_weights_of_the_block`. Add it to
  the contract tests, on each block setting.
- The user chose: the count of the Muon specs of each family is rule 3. The four
  `*_plan_fits_the_model` tests are rule 2, but their counts stay. Move the counts
  (Mamba-3: 2, Mamba-2: 2, Mamba-1: 3, with no MLP and no untied weight) into one small
  rule-3 test, beside `mamba3_scalar_channels_stay_on_adamw`.

### The implementation

- The pin moved from `d534e7d3` to burn-stack `a236fe3f`, the pushed commit of its
  ticket 05. The version of burn-stack stays 0.41.0, so there is no version bump.
- `src/family/tests/multi_gate.rs` had 17 tests, all of rule 1. The file and its `mod`
  line are deleted.
- `src/family/tests/optim.rs` had 16 tests. The 15 tests of rule 1 or 2 are deleted.
  `mamba3_scalar_channels_stay_on_adamw` (rule 3) stays.
- The new rule-3 test `each_family_gives_a_fixed_count_of_muon_specs` holds the counts
  of the four `*_plan_fits_the_model` tests. It reads the plan of a network config with
  no feed-forward, no `CatLinear` merge and no untied parameter. The configs are Mamba-3
  in a `MambaBidiLayersConfig` with the `mean` merge, Mamba-2 in a `MambaVocabNetConfig`,
  and Mamba-1 in a `MambaLatentNetConfig`. The user chose this form at the gate (see
  below). A failure names the family, its network config and the paths of its specs.
- `block_contract.rs` runs `muon_projections_select_only_weights_of_the_block` on each
  of the 17 block settings.
- The red check used two temporary changes of the family code. A plain path
  (`ProjSpec::path_whole`) for the `out_proj` spec of Mamba-2 failed the new contract on
  the 4 Mamba-2 settings, with one line for each setting. A Mamba-1 list without
  `out_proj` failed the count test. Both changes are reverted.
- The "Block contracts" seam row of `docs/agents/testing.md` names
  `src/family/tests/optim.rs`, because its tests now check only the Mamba specs.
- `scripts/check.sh` passes on the pin: 374 lib tests (404 before, 32 deleted, 2 new).

### Coverage result

The gate ran with the `[patch]` of burn-stack on, at burn-stack `a236fe3f`. The commands
are the four steps of `tmp/coverage-record/record.md`. The script is
`tmp/coverage-record/gate-05.sh`, and the reports are in `tmp/coverage-record/after-05/`.

The first run lost 7 lines of burn-mamba, all in `src/family/config.rs`. Each line is a
one-line forwarder that only the deleted tests reached:

- line 74, the Mamba-1 arm of `MambaBlockConfig::muon_projections`
  (`mamba1_plan_fits_the_model`). The `MambaBlock` setting of the contracts uses only
  Mamba-3.
- lines 126–128, `MambaVocabNetConfig::muon_plan` (`mamba2_plan_fits_the_model`,
  `vocab_plan_holds_the_mlp`).
- lines 150–152, `MambaBidiLayersConfig::muon_plan` (the two `bidi_plan_*` tests).

No block contract reaches these lines, because a contract takes a block config, not a
network config. The user chose to read the counts through the plans of the network
configs, as the old tests did. The final run of the four steps lists no lost item:

| `src/` of | lib tests of | lines | branches | conditions |
|---|---|---|---|---|
| burn-mamba | burn-mamba | 7713 of 7898 | 189 of 216 | 82 of 108 |
| burn-stack | both | 4100 of 6670 | 326 of 696 | 135 of 348 |

The record has the same totals for burn-mamba. For burn-stack it has 4057 of 6627 lines,
324 of 694 branches and 134 of 347 conditions. The compare of step 4 lists no lost item.
It lists 13 lines and 2 branches on changed lines: the lines of
`muon_projections_name_existing_rank_2_weights` that burn-stack ticket 05 changed. That
ticket checked them. The log of the final run is `tmp/coverage-record/gate-05.log`, and
the log of the first run is `tmp/coverage-record/gate-05-first.log`.

### The review (`tmp/review-ct05-standards.md`, `tmp/review-ct05-spec.md`)

- Fixed:
  - The header of `src/family/tests/bidi.rs` named the deleted `multi_gate` tests. It
    now names their home in burn-stack.
  - The doc of the count test said that the count pins the list of each family. A swap
    of two specs keeps the count. The doc now says that a change of the count fails the
    test. It also gives the reason for the network configs: they reach
    the forwarders of `src/family/config.rs`, and no contract reaches them.
  - The glossary term "untied parameter", the name `network` for the label of a network
    config, and no second `use` of the prelude.
  - The seam row writes the path from the root of the repo, and it names what
    `optim.rs` catches.
  - Step 4 ran on an earlier form of the count test, and `check.sh` ran before the last
    edit of `optim.rs`. The four steps and `check.sh` ran again on the final tree.
- Kept:
  - The header of `src/family/tests/mod.rs` and the File Map line of `src/family/tests/`
    still say that the other files test the containers. Ticket 06 rewrites both for what
    stays.
  - The `MambaBlock` setting of the contracts uses one family, as ticket 02 chose. The
    arms of Mamba-1 and Mamba-2 in `MambaBlockConfig::muon_projections` are one-line
    forwarders, and the count test reaches them.
  - The count test reads the plans of containers. The user chose this form at the gate.
