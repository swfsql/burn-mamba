# 06: Batch D, delete the moved tests of `MambaBlock`, the untied copies, packed rows, the H-Net, the captured step, the configs and the reset alignment

Status: ready-for-agent

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

- [ ] Each test of rule 1 or 2 in the list is deleted. Each test of rule 3 stays.
- [ ] The new contracts run on each block setting.
- [ ] The header of the family tests and the File Map line describe what stays.
- [ ] The coverage compare lists no lost item of the record.
- [ ] `docs/agents/testing.md` has no paragraph about the gate of the move.
- [ ] `cargo test --lib -- --test-threads=1` passes on `flex`. `scripts/check.sh` passes.

## Comments

### From ticket 02 (the review of the contract tests)

- `src/family/tests/block_contract.rs` has its own copies of the config builders
  `m1`/`m2`/`m3` of `block.rs`. If `block.rs` keeps tests of rule 3, put one copy in
  `src/family/tests/mod.rs` for both files.
- The "limits" section of the header of `block_contract.rs` names `capture.rs` as the
  test that compares each cache field directly. If this batch moves or deletes that
  test, update that line.
