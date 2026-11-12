# 01: The coverage run, and the record before the move

Status: ready-for-agent

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

- [ ] `scripts/check.sh` runs the coverage by default, and skips it with a note when the
      tools are missing.
- [ ] `## Comments` holds the answer about the report of burn-stack.
- [ ] The record exists in `tmp/`, with its commits and its `.discard` tag.
- [ ] The test-seam doc names the coverage gate.
