# 01: The phrasal-verb rule flags the common verbs that it misses

Status: ready-for-agent

Blocked by: none

**Context:** the STE sweep of both repos found two misses of the linter of the
`asd-ste100` skill. This ticket fixes the first one, and ticket 02 fixes the
other. The two tickets change different rules, so they do not block each
other. Both tickets block each open sweep ticket: `.scratch/ste-sweep/` 02 to
09, and `../burn-stack/.scratch/ste-sweep/` 04 and 05. After both tickets,
`../burn-stack/.scratch/ste-sweep/issues/06-recount-swept-batches.md` counts
the batches that are already swept again.

**Access:** `../asd-ste100` holds a copy of the skill. The user approved access
to it. In each command, reach it by its absolute path, because the access hook
reads that path. In a repo file, write only the relative path `../asd-ste100`.
`scripts/check.sh` of both repos runs the installed copy of the skill, not this
copy. The user installs the change.

**What to build:** the phrasal-verb rule flags `run out`, `run into`,
`blow up`, `cut off`, `carry out` and `give up` as hard findings, as it flags
`go back`. It flags each tense, and a noun or a pronoun between the verb and
the particle. The sweep found two of them: `run out` in
`../burn-stack/src/examples/training.rs` (burn-stack ticket 03) and `blow up`
in `../burn-stack/src/modules/loss/bce.rs` (burn-stack ticket 02).

## The repro

Run the linter on a `.md` file with these lines, one paragraph each:

| Line | Finding of the linter |
|---|---|
| `A loop sees it go back to the first batch.` | `go back`, hard (the reference) |
| `The session goes back to the first batch.` | `goes back`, hard (the reference) |
| `A loop sees it run out when it checks the budget.` | none |
| `The budget runs out after 600 training steps.` | none |
| `The budget ran out at the last epoch.` | none |
| `The loop runs out of batches.` | none |
| `A loop sees the budget run out.` | none |
| `The loop runs into a wall.` | none |
| `The gradient blows up at step 10.` | none |
| `The detach cuts off the graph.` | none |
| `The batcher carries out the padding.` | none |
| `The session gives up the lock.` | none |
| `The loop picks up the next batch.` | `picks up`, advisory (a technical phrasal verb) |

## Rules

- Each new pair is a hard finding (`phrasal-verb-particle`). A pair that is a
  fixed technical term, as `set up`, stays advisory
  (`phrasal-verb-technical`). Give the reason for each pair that goes to the
  advisory list.
- The message of each finding gives a plain verb, as the message of `go back`
  does.
- `run over` stays out of this ticket. In both repos, "a scan runs over the
  row" means "iterates over the row". A flag on it needs its own decision.

## References

- The `SKILL.md` of the skill describes the phrasal-verb rules. Its text must
  stay true after the change.
- The `--selftest` option of the linter holds the self-tests. Add the cases of
  the repro there.

- [ ] The linter flags each line of the repro that has no finding, in each of
      its tenses, as a hard finding with a plain verb.
- [ ] The lines with `go back`, `goes back` and `picks up` keep their findings.
- [ ] `--selftest` holds the cases of the repro, and it passes.
- [ ] The `SKILL.md` of the skill describes the rule as it is.
- [ ] `## Comments` lists the new hits in each tracked `.rs` and `.md` file of
      both repos, outside each `tmp/`, with the count of each file. Run the
      linter with `--enable history --parts docs,comments,messages`, as
      `scripts/check.sh` does.

## Comments
