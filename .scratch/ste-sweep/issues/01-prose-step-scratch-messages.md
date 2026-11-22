# 01: The prose step reads `.scratch/` and the message strings

Status: ready-for-agent

Blocked by: none

Spec: `../spec.md`, section "The prose step of `scripts/check.sh`".

**Context:** this ticket goes first in burn-mamba. Tickets 02–09 are the
batches of the sweep, and this ticket blocks each one. So the prose step guards
each batch. `../burn-stack/.scratch/ste-sweep/issues/01-prose-step-scratch-messages.md`
makes the same change in burn-stack.

**What to build:** the prose step of `scripts/check.sh` also checks the
tickets and the strings of `panic!`, `expect` and `assert!` calls.

- The step also reads the files under `.scratch/`.
- The step runs the linter with `--parts docs,comments,messages`, on both
  sides of the compare.
- The rule stays. A changed file fails when it has more hard findings than at
  the merge base with `origin/main`. A new file starts at zero.
- The comments of the script describe what the step reads. Each doc that
  names the prose step agrees with them.
- The ticket plants no violation to prove the change (spec, "Testing
  Decisions").

**References:**

- The prose step of `../burn-stack/scripts/check.sh` has the same shape. Keep
  the two steps equal.
- The new tickets of `.scratch/ste-sweep/` are new files, so the extended step
  reads them. They start at zero hard findings.

- [ ] The prose step reads `.scratch/` and the message strings.
- [ ] The comments of `scripts/check.sh` and each doc that names the step
      describe what it reads.
- [ ] `./scripts/check.sh` passes. Run it in the background.

## Comments
