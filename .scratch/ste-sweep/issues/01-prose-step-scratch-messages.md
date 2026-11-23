# 01: The prose step reads `.scratch/` and the message strings

Status: resolved

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

- [x] The prose step reads `.scratch/` and the message strings.
- [x] The comments of `scripts/check.sh` and each doc that names the step
      describe what it reads.
- [x] `./scripts/check.sh` passes. Run it in the background.

## Comments

### The step of burn-stack

`check_prose` is a copy of the function of `../burn-stack/scripts/check.sh`,
from its comment to its last line. A `diff` of the two functions gives no
output.

### The counts of the prose step

The merge base is `a195b208`. The step reads these files:

| File | Hard findings before | Hard findings after |
|---|---|---|
| `.scratch/ste-sweep/issues/01-prose-step-scratch-messages.md` | 0 | 0 |
| `.scratch/ste-sweep/issues/08-sweep-notes-root-docs.md` | 0 | 0 |
| `docs/agents/code-review.md` | 0 | 0 |
| `examples/tiny-stories/README.md` | 8 | 8 |

The reference "The new tickets of `.scratch/ste-sweep/` are new files" does
not apply in burn-mamba. The spec and the tickets are in the merge base, so the
step reads only the tickets that change. Each ticket has zero hard findings.
`spec.md` has one hard finding at line 300: the history rule matches the
example that the line quotes for that rule. Ticket 09 sweeps `.scratch/`.

### The message strings

With `--parts docs,comments,messages`, 21 of the 193 tracked `.rs` files get 24
more hard findings. The total goes from 417 to 441. All 24 are in the message
strings. Both sides of the compare count them, so an unchanged message does not
fail the step.

| File | Without messages | With messages |
|---|---|---|
| `examples/common/context.rs` | 11 | 13 |
| `examples/common/long_stories.rs` | 4 | 6 |
| `examples/hnet-stories/training.rs` | 0 | 1 |
| `examples/mnist-ae/training.rs` | 3 | 4 |
| `examples/mnist-class/training.rs` | 1 | 2 |
| `examples/reset/{majority,quintic,rotor}/training.rs` | 0 | 1 each |
| `examples/reset/{spinor,spinor-product,swap}/training.rs` | 0 | 1 each |
| `examples/tally/shared/training.rs` | 0 | 1 |
| `examples/tiny-stories/context.rs` | 0 | 2 |
| `examples/tiny-stories/contrast.rs` | 3 | 4 |
| `examples/tiny-stories/inference.rs` | 2 | 3 |
| `examples/tiny-stories/training.rs` | 1 | 2 |
| `src/mamba2/ssd/serial_recalculated/combined_backward/tests.rs` | 2 | 3 |
| `src/mamba3/double_ssd/ssd/serial.rs` | 5 | 6 |
| `src/mamba3/rotation/mod.rs` | 11 | 12 |
| `src/mamba3/single_ssd/ssd/serial.rs` | 5 | 6 |
| `src/row_end.rs` | 0 | 1 |

The report of a failure also uses `--parts docs,comments,messages`. So it lists
each finding that the count has.

The calls that the linter reads are in the burn-stack ticket, from a probe
file. The comment of `check_prose` gives four of them as examples.

### The docs that name the step

- `CLAUDE.md:16` says "prose" and does not say what the step reads. Every
  session loads `CLAUDE.md`, so the line stays short and does not change.
- `docs/agents/code-review.md:8` names the rules of `ste-lint.py` that
  `scripts/check.sh` covers. This change adds three sentences to it: the files
  and the strings that the step reads, and the rule of the step. The step runs
  each hard rule of the linter, so the list of rules gets "for example".
- Line 10 of tickets 02–09 names the step. It agrees with the step.
- `.scratch/check-script-followups/issues/01-prose-ratchet-reads-scratch.md`
  asks for the same step. This change meets each of its criteria. Its
  `Status:` stays `needs-triage` for the maintainer.

### The `tmp/` directories

After the review, the maintainer selects a skip of each `tmp/` directory, at
any depth, so the step reads what the sweep reads. A second commit in each repo
adds `grep -v -E '(^|/)tmp/'` to the list of files, and a sentence to the
comment of `check_prose`. In burn-mamba, 13 tracked `.md` files are under
`examples/{hnet-stories,tiny-stories}/tmp/`, and `.gitignore` has the root
`tmp/`. `docs/agents/code-review.md:8` and the section "The prose step of
`scripts/check.sh`" of the spec also give the skip.
