# 05: Batch C, delete the moved tests of Multi-Gate Residuals and the optimizer

Status: ready-for-agent

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

- [ ] Each test of rule 1 or 2 in the list is deleted. Each test of rule 3 stays.
- [ ] The new contracts run on each block setting.
- [ ] The coverage compare lists no lost item of the record.
- [ ] `cargo test --lib -- --test-threads=1` passes on `flex`. `scripts/check.sh` passes.
