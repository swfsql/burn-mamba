# 02: The first block contracts on each block setting

Status: ready-for-agent

Blocked by: `../burn-stack/.scratch/container-tests-home/issues/02-block-contract-suite.md`

Spec: `../spec.md`, sections "Block settings in burn-mamba" and "Testing Decisions".
Run only one session on the two repos.

**What to build:** burn-mamba proves "my blocks ⇒ block contracts" for the seven first
contracts, on each block setting that its family tests use.

- A new test module in `src/family/tests/` calls the contracts of burn-stack.
- It has one `#[test]` for each contract. The test runs the contract on each block
  setting. It collects each failure with the name of its setting, and at the end it
  fails with all of them. So one failure hides no other.
- The block settings:
  - Mamba-1.
  - Mamba-2 and Mamba-3 on the SSD paths `default`, `Minimal`, `Serial` and
    `SerialRecalculated`.
  - Mamba-3 with the rotations `Real`, `Complex` and `Quaternion`.
  - Mamba-3 with the tap patterns `None`, `Vertical`,
    `VerticalPlusHorizontalCarryOver` and `default`.
  - Mamba-3 with `mimo_rank` 1 and 2, and with `micro_steps` 2.
  - `MambaBlock` with one family.
- Each family gives its fp32 tolerance. The contract passes it through `dtype_tol`.
- If a contract fails on a setting, stop and ask the user. The failure can be a bug of
  the family or a wrong contract.
- This ticket deletes no test.

**Pin:** move the burn-stack pin to the pushed commit of burn-stack ticket 02. Follow
"The burn-stack pin" of `docs/agents/issue-tracker.md`.

**Docs:**

- `docs/agents/testing.md`: the "Containers with real blocks" row becomes "Block
  contracts". It names the new module.
- The File Map line of `src/family/tests/` names the block contracts.

- [ ] Each of the seven contracts runs on each block setting above.
- [ ] A failure names its contract and its setting.
- [ ] `cargo test --lib -- --test-threads=1` passes on `flex`. `scripts/check.sh` passes.
- [ ] The seam row and the File Map line name the block contracts.
