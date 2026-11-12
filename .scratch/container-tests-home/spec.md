# Container tests: each test in the repo that owns its behavior

Status: ready-for-agent

Source: `../burn-stack/.scratch/arch-review-backlog/issues/05-container-tests-home.md`.
Its `## Comments` hold the triage that this spec follows.

## Problem Statement

burn-mamba is older than burn-stack, and its tests grew first. So 130 tests in its
family tests still check the containers of burn-stack, with a Mamba block in the place
of `RefBlock`. ADR-0005 says that `RefBlock` tests the containers. Today a regression
in a container can pass in burn-stack and fail only when burn-mamba moves its pin.

A second family crate gets two bad choices: copy these tests, or test less than
burn-mamba. Also, no repo measures its test coverage. A move of tests can lose
coverage, and no one sees it.

## Solution

Each test goes to the repo that owns the behavior that it checks:

- burn-stack tests each container with `RefBlock`, and each of its modules alone.
- burn-stack gives the block contracts as public test functions under `test-helpers`.
  Each family crate calls the whole suite on each block setting that it ships. It tests
  only its own family code. burn-mamba is not special.
- Each repo measures the line, branch and condition coverage of its own `src/` by its own
  tests, in each run of its check script. A compare mode shows each covered item that a
  change lost.
- The move has a gate: after each batch, both repos together keep each covered item that
  they had before the move.

## User Stories

1. As a burn-stack maintainer, I want each container test in burn-stack, so that a
   container regression fails before a pin move.
2. As a burn-stack maintainer, I want each test that uses no block beside its module, so
   that I find it next to its code.
3. As a burn-stack maintainer, I want one `RefBlock` test file for each mechanism, so
   that no file holds every container.
4. As a burn-stack maintainer, I want `RefBlock` to grow through its config when a moved
   test needs it, so that ADR-0005 holds.
5. As a family-crate maintainer, I want one public function for each block contract, so
   that every family crate checks the same contracts.
6. As a family-crate maintainer, I want each contract to take my config, options and
   tolerance, so that my block passes within its numerics.
7. As a family-crate maintainer, I want the block itself to tell a contract to skip it,
   so that I keep no list of skips.
8. As a family-crate maintainer, I want a failure to name its contract and setting, so
   that one failure hides no other.
9. As a family-crate maintainer, I want no container test code in my crate, so that I
   repeat no check of burn-stack.
10. As a family-crate maintainer, I want a contract to use a container only when it must,
    so that a failure points at my block.
11. As a burn-stack maintainer, I want the suite to run on `RefBlock`, so that the suite
    and the container tests cannot drift apart.
12. As a burn-stack maintainer, I want each contract to fail on a block that breaks it, so
    that no contract passes because it cannot fail.
13. As a burn-mamba maintainer, I want the suite on each block setting that the tests use
    today, so that no setting loses its checks.
14. As a burn-mamba maintainer, I want the tests that only Mamba needs to stay, so that
    the family code keeps its own checks.
15. As a reviewer, I want one line for each test in each batch, so that I can check that
    the move dropped no check.
16. As a reviewer, I want a coverage compare after each batch, so that I see each covered
    item that the move lost.
17. As a maintainer of either repo, I want coverage in each check run, so that coverage
    stays visible after the move.
18. As a new family-crate maintainer, I want to run the burn-stack coverage script on my
    crate, so that I write no copy.
19. As a family-crate maintainer, I want lost coverage to show a property that no contract
    states, so that I add a contract.
20. As an agent on burn-stack ticket 02, I want the container tests in burn-stack first,
    so that my change meets its tests.
21. As a burn-stack maintainer, I want the horizon memory probe over `RefBlock`, so that
    it measures the horizon and not a family.
22. As a reader of ADR-0005, I want it to name the block contracts, so that a real block
    in the suite looks correct.
23. As a reader of either seam table, I want a row for the block contracts, so that I know
    where each test goes.

## Implementation Decisions

### Terms

**Block contract** and **Family crate** are in the glossary of burn-stack.

### Who proves what

- burn-stack proves "block contracts ⇒ the containers work".
- Each family crate proves "my blocks ⇒ block contracts".
- No family crate tests a container. When a family crate needs a property that no
  contract states, the fix is a new contract in burn-stack.

### The block contract module (burn-stack)

- A new top-level module `block_contract`, beside `reference`. It has the gate of
  `reference`: the tests of burn-stack, or the `test-helpers` feature.
- One public function for each contract, generic over the block config. Its inputs are
  the config, the block options and an fp32 tolerance.
- The function passes the tolerance through `dtype_tol`. The check of a value that can
  be large uses `max_rel_diff`. A failure panics with a message that names the contract.
- A contract checks the bare block when it can. It puts the block in a container only
  when the property needs one. Examples: the untied application views, and the run on
  the inner backend under a cut.
- A contract that does not apply to a block returns early. The block gives the reason
  through the `Block` trait, for example a `reset_align` of `None`.
- The first contracts come from the `Block` docs and the current tests:
  - the forward equals the step unrolled,
  - a window split through the cache equals one window,
  - padded rows equal each row alone,
  - packed segments equal each segment alone,
  - a record round trip keeps the block,
  - the cache traversal moves every field of the cache,
  - the Muon projections name rank-2 weights that exist.

  Each batch adds the contracts that its sort finds.
- burn-stack runs each contract on `RefBlock` with a near-zero tolerance. It deletes its
  own `RefBlock` tests that repeat a contract.
- A private test block breaks one contract at a time. Each contract function has one
  test that expects its panic on that block. `RefBlock` does not change for this.
- ADR-0005 gets one consequence, in the same change as the module: "A family crate tests
  its own blocks with the block contracts, through `test-helpers`. It does not test the
  containers again."

### New homes in burn-stack

- The 14 tests that use no block go to the tests of their modules: Multi-Gate Residuals
  (6), the class plans (3) and `Segmented` (5).
- The container tests go to the `RefBlock` tests, with one file for each mechanism.
- The memory probe of the gradient horizon moves over a `RefBlock` that is large enough
  for the activations to set the peak memory.
- `RefBlock` grows through its config when a moved test needs it. The identity-prefix
  test needs no growth, because `out_proj` is already public.

### The sort rule

Each batch asks these questions of each test, in order:

1. Does it test a container? That is, can `RefBlock`, or no block, take the place of the
   Mamba block? Then it moves to burn-stack, or burn-stack drops it as a repeat.
2. Does every family crate need the same test for its own blocks? Then it becomes a
   block contract, and burn-mamba calls the contract.
3. Does only Mamba need it? Then it stays in burn-mamba.

The first step of each batch writes a list under the `## Comments` of its ticket. The
list has one line for each test: the rule number and the new home. When the rule does
not place a test, the batch stops and asks the user.

### Batches

Each batch reads about 2–3k lines, with the matching tests of burn-stack:

- A: the class-marker tests.
- B: the tests of `Layers` and the gradient horizon, of `Layer`, and of the
  bidirectional stacks.
- C: the tests of Multi-Gate Residuals and of the optimizer.
- D: the tests of `MambaBlock`, the untied copies, the packed rows, the H-Net, the
  captured step, the configs and the reset alignment.

### Block settings in burn-mamba

The suite runs on each setting that a family test uses today:

- Mamba-1.
- Mamba-2 and Mamba-3 on the SSD paths `default`, `Minimal`, `Serial` and
  `SerialRecalculated`.
- Mamba-3 with the rotations `Real`, `Complex` and `Quaternion`.
- Mamba-3 with the tap patterns `None`, `Vertical`, `VerticalPlusHorizontalCarryOver`
  and `default`.
- Mamba-3 with `mimo_rank` 1 and 2, and with `micro_steps` 2.

Each batch adds the settings that its files use to its list.

### Coverage

- One script in burn-stack takes the path of a crate. It measures the line, branch and
  condition coverage of the `src/` of that crate by its own lib tests. It uses
  `cargo-llvm-cov` on nightly.
- Nightly rustc has no MC/DC: `-Zcoverage-options` accepts only `block`, `branch` and
  `condition`. Condition coverage is the nearest to MC/DC.
- The check script of each repo runs it by default. burn-mamba runs the copy of
  burn-stack through its path in the box.
- A compare mode lists each item that a base commit covered and the current tree does
  not cover. It has the form of `compare_reports.py`. There is no stored baseline.
- The gate of the move starts with a record before the first batch. The record holds the
  coverage of `src/` in both repos by both test suites. burn-mamba uses its `[patch]`
  path for it. After each batch, both suites together must cover each item of the
  record.
- `cargo-llvm-cov` may not report burn-stack from a run of burn-mamba, because burn-stack
  is not in its workspace. The coverage work checks this first. If it cannot report
  burn-stack, the gate of the move is the lists of the batches and the review.
- The user installs `cargo-llvm-cov` and the nightly `llvm-tools-preview` through the box
  build. The coverage work starts after that.

### Order

Each ticket changes one repo. The tickets are in `issues/` beside this spec (burn-mamba)
and in the `issues/` of the same slug in burn-stack.

- burn-stack 01: the coverage script. burn-mamba 01: its run, and the record.
- burn-stack 02: the block contract module. burn-mamba 02: the first contracts on each
  block setting.
- Batches A–D: each batch has a burn-stack ticket (03–06) that moves the tests, and a
  burn-mamba ticket (03–06) that deletes them. The coverage gate runs in the burn-mamba
  ticket, so it measures a real deletion.
- Each burn-mamba ticket from 02 on moves the pin.

The burn-stack batch tickets block burn-stack ticket `02-residuals-transition.md`.

## Testing Decisions

- A good test checks behavior at a public interface: a container, a module alone or a
  block through the `Block` trait. It reads a field only when the field is public by
  design, as the fields of `RefBlock` are.
- The seams:
  - **Containers with `RefBlock`** (burn-stack, an existing seam) gets the container
    tests that move.
  - **One module alone** (burn-stack, an existing seam) gets the 14 tests that use no
    block. So the optimizer module gets tests of its own.
  - **Block contracts** (new) replaces **Containers with real blocks**. In burn-stack,
    each contract function passes on `RefBlock` and fails on the private test block. In
    each family crate, one test for each contract runs it on each block setting, and a
    failure names its setting. It catches a
    block that breaks a contract, and a contract function that cannot fail. It misses a
    container need that no contract states, and the coverage report shows that need.
  - The coverage compare is a gate, not a seam. It works like the bench gates.
- The test-seam tables of both repos change with the module and with the burn-mamba
  ticket.
- Prior art:
  - The `RefBlock` tests: `block_forward_equals_step_unrolled` and
    `block_forward_splits_into_windows_through_the_cache`.
  - `padded_rows_match_each_row_alone` in the H-Net tests.
  - `a_start_off_the_reset_alignment_panics`, for a test that expects a panic.
  - `compare_reports.py`, for the form of the compare mode.
  - The burn-mamba family tests are the sources of the move.
- The tests run on `flex`. A test takes its device from `test_device()`. A tolerance goes
  through `dtype_tol`.

## Out of Scope

- A change of the behavior of a container, or of the interface of `Layers` (burn-stack
  ticket 02).
- The family tests outside the family test directory: form agreement, forward/step
  parity inside a family, and a mechanism against its direct form. Some of them can
  overlap a block contract. A later ticket can sort them with the same rule.
- MC/DC, which nightly rustc does not have.
- A stored coverage baseline or a coverage ratchet.
- The installation of the coverage tools.
- A claim about training quality, and a run on a backend other than `flex`.

## Further Notes

- burn-stack adds public items only under `test-helpers`, and burn-mamba changes only
  its tests and scripts. So no ticket breaks a public API, and no version bump applies.
- The batch sizes come from the line counts of today. A batch that grows past about 3k
  lines splits in two.
- The coverage record goes to the `tmp/` of burn-mamba, with a `.discard` tag that names
  the end of the move.
