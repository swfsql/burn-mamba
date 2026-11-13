# 01: The coverage run, and the record before the move

Status: resolved

Blocked by: `../burn-stack/.scratch/container-tests-home/issues/01-coverage-script.md`

Spec: `../spec.md`, section "Coverage". Run only one session on the two repos.

**What to build:** burn-mamba reports its coverage in each check run. A record holds the
coverage of both repos before any test moves. The batch tickets 03–06 gate on it.

- `scripts/check.sh` runs `../burn-stack/scripts/coverage.py` on burn-mamba by default.
  When the tools are missing, it prints a note and skips the coverage, as the prose
  check does.
- Check whether `cargo llvm-cov` can report the `src/` of burn-stack from a run of
  the burn-mamba tests, with the `[patch]` path. Write the answer under `## Comments`.
- The record holds:
  - the coverage of the `src/` of burn-stack by the burn-stack tests,
  - the coverage of the `src/` of burn-stack by the burn-mamba tests, if the tool can
    report it,
  - the coverage of the `src/` of burn-mamba by the burn-mamba tests.
- The record names the commit of each repo. It goes to `tmp/`, with a `.discard` tag:
  "discard when tickets 03–06 of `.scratch/container-tests-home/` are all resolved".
- If the tool cannot report burn-stack from here, the gate of the move is the sorted
  lists of the batches and the review. Write this under `## Comments`, and tell the user.

**Docs:** `docs/agents/testing.md` names the coverage report and the gate of this move,
beside the bench gates.

- [x] `scripts/check.sh` runs the coverage by default, and skips it with a note when the
      tools are missing.
- [x] `## Comments` holds the answer about the report of burn-stack.
- [x] The record exists in `tmp/`, with its commits and its `.discard` tag.
- [x] The test-seam doc names the coverage gate.

## Comments

### The report of burn-stack from a run of burn-mamba

Yes, `cargo llvm-cov` (0.9.1) can report the `src/` of burn-stack from a run of the
burn-mamba tests, with the `[patch]` path. Three facts of the tool apply:

- The default run instruments only the crates of the workspace. Its wrapper
  (`RUSTC_WRAPPER=cargo-llvm-cov`) gives `-Cinstrument-coverage` only to the crates in an
  env list. `--dep-coverage burn-stack` adds `burn_stack` to that list.
- Cargo does not track that list. So a burn-stack build with no instrumentation stays in
  `target/llvm-cov-target/`, and the run uses it. Delete it first:
  `cargo +nightly clean -p burn-stack --target-dir target/llvm-cov-target`.
- `cargo llvm-cov report --dep-coverage burn-stack` finds no file of the path dependency.
  The plain report has the files of burn-stack, because its default ignore regex skips
  only the cargo registry, the git checkouts and the toolchain.

`coverage.py` counts only the `src/` of the crate that it runs. So the throwaway script
`tmp/coverage-record/dep_report.py` does the run above, and counts the `src/` of
burn-stack with the functions of `coverage.py`. It also adds the counts of two reports
(the two suites), and compares two saved reports with the compare of `coverage.py`. A
second run with the script gave the same items as the first run: no lost item and no
changed item, in both directions.

This ticket changes only burn-mamba, so the script stays in `tmp/`. A burn-stack ticket
can move it into `coverage.py` as two options. One counts the `src/` of a `[patch]`
dependency. The other adds a saved report.

### The review

- `/code-review` (`tmp/review-ct01-standards.md`, `tmp/review-ct01-spec.md`) found no hard
  violation. The fixes: `docs/agents/testing.md` cites the coverage rules of burn-stack
  and does not copy them. It names `dep_report.py` as the exception for the move. The
  record writes the result under `## Comments`, as tickets 03–06 do. Ticket 06 deletes
  the paragraph about the gate of the move. The header of `check.sh` names the coverage.
- `check.sh` runs the lib tests twice: once on stable, and once under the coverage. So the
  flaky test `siso_matches_mimo_backward` has two chances to fail the check. burn-stack
  does the same.

### The record

The record is `tmp/coverage-record/record.md`, with the tag `tmp/coverage-record.discard`.
It names the commits (burn-mamba `58a4e2f3`, burn-stack `d534e7d3`), the reports and the
commands of the gate.

| `src/` of | lib tests of | lines | branches | conditions |
|---|---|---|---|---|
| burn-stack | burn-stack | 3639 of 4372 | 257 of 430 | 93 of 215 |
| burn-stack | burn-mamba | 3227 of 6617 | 247 of 690 | 95 of 345 |
| burn-stack | both | 4057 of 6627 | 324 of 694 | 134 of 347 |
| burn-mamba | burn-mamba | 7713 of 7898 | 189 of 216 | 82 of 108 |

Only the burn-mamba tests cover 418 lines and 67 branches of burn-stack.
`tmp/coverage-record/burn-stack-only-by-burn-mamba.md` lists them. A batch can lose them.

The gate of each batch ticket (03–06): the two compares of the record list no lost
item. The sorted list of the batch and the review stay as the second check.

### The check run

- `check_coverage` runs after `cargo test --lib`. Without `../burn-stack/scripts/coverage.py`
  (CI), or without `cargo-llvm-cov`, it prints a note and returns 0. A copy of the
  function showed both notes. The second test used a `cargo` that has no subcommands.
- The first coverage run failed on the known flaky test `siso_matches_mimo_backward`. The
  second run passed.
- `./scripts/check.sh` passed on the pin, with the coverage step. That step gave the
  burn-mamba totals of the record: lines 7713 of 7898, branches 189 of 216, conditions 82
  of 108.
