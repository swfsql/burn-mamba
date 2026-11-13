# 02: The first block contracts on each block setting

Status: resolved

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

- [x] Each of the seven contracts runs on each block setting above.
- [x] A failure names its contract and its setting.
- [x] `cargo test --lib -- --test-threads=1` passes on `flex`. `scripts/check.sh` passes.
- [x] The seam row and the File Map line name the block contracts.

## Comments

### From burn-stack ticket 02 (the review of the contract module)

- A window of a contract has 7 tokens. The default Mamba chunk is longer, so
  with it no window holds a chunk boundary. Give each SSD path a short chunk
  (for example `Serial(Some(4))`), so that a scan error between chunks shows.
- Each contract starts from no cache. On Mamba-3, the cache variant selects
  the double-SSD pathway, so no contract reaches that pathway. The settings
  above leave it out too. Batch D decides who keeps the double-SSD check of
  `src/family/tests/capture.rs`.
- `cache_traversal_moves_every_field` reads outputs, not fields. A field that
  one more step does not change stays unseen. `*_assign_moves_every_field`
  compares each field directly.

### The implementation

- The pin moved from `12ddd437` to burn-stack `d534e7d3`, the pushed commit of its
  ticket 02. The version of burn-stack stays 0.41.0, so there is no version bump.
- `src/family/tests/block_contract.rs` has one test for each contract, with the name of
  the contract. A trait `Contract` and the macro `contract_tests!` pass each generic
  contract function to `on_each_setting`. That function catches the panic of each
  setting, and at the end it panics with one line for each failed setting:
  `- <setting>: block contract `<name>`: <check>`.
- The 17 settings: Mamba-1, Mamba-2 on 4 paths, Mamba-3 on 11 settings, and
  `MambaBlock` over the first enabled family (Mamba-3, then Mamba-2, then Mamba-1). The
  name of a Mamba-3 setting gives its path, rotation, tap pattern, `mimo_rank` and
  `micro_steps`.
- Mamba-3 has a base (`SerialRecalculated`, `Complex2D`, the default tap pattern,
  `mimo_rank` 1, `micro_steps` 1). Each other setting changes the path, the rotation,
  the tap pattern, `mimo_rank` or `micro_steps` of the base. So `Complex`, the default
  tap pattern and `mimo_rank` 1 are the base itself.
- At `micro_steps` 1, `Vertical` and `VerticalPlusHorizontalCarryOver` are the default
  tap pattern. So their settings have `micro_steps` 2, as in `capture.rs` and
  `packed.rs`.
- Each SSD path has a chunk length of 4 tokens (`Some(4 · micro_steps)` write positions
  on Mamba-3). The `default` paths keep the optimal chunk length of 32. On them, only the
  packed contract crosses a chunk boundary.
- Each family has a tolerance of `1e-4`. Every contract passed on every setting at the
  first run, so no failure needed a question.
- A run with the three tolerances set to `-1` showed that each setting runs. Each
  contract that uses the tolerance failed on 17 settings. The packed contract failed on
  14, because Mamba-1 and the two `Minimal` paths return early. Each failure line named
  its contract and its setting. The Muon contract takes no tolerance and passed.
- `scripts/check.sh` passes: 404 lib tests (397 before). The coverage of `src/` stays
  the same as in the record: lines 7713 of 7898, branches 189 of 216, conditions 82 of
  108.

### The review (`tmp/review-ct02-standards.md`, `tmp/review-ct02-spec.md`)

- Fixed: the seam row describes no plan, uses noun phrases and gives the coverage
  clause of the spec. The module header no longer says that each setting changes one
  value. It says "the default SSD path" and "chunk length", and it gives the reason for
  the optimal chunk length of the default path. All Mamba-3 paths get their chunk length
  from one helper (`m3_path`). The panic text has no count. `This` became
  `ThisContract`.
- Kept, with a note in ticket 06: the copies of the config builders `m1`/`m2`/`m3` of
  `block.rs`, and the pointer to `capture.rs` in the header of the module. Ticket 06
  sorts both files.
