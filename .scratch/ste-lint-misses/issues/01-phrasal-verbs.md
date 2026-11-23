# 01: The phrasal-verb rule flags the common verbs that it misses

Status: resolved

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

- [x] The linter flags each line of the repro that has no finding, in each of
      its tenses, as a hard finding with a plain verb.
- [x] The lines with `go back`, `goes back` and `picks up` keep their findings.
- [x] `--selftest` holds the cases of the repro, and it passes.
- [x] The `SKILL.md` of the skill describes the rule as it is.
- [x] `## Comments` lists the new hits in each tracked `.rs` and `.md` file of
      both repos, outside each `tmp/`, with the count of each file. Run the
      linter with `--enable history --parts docs,comments,messages`, as
      `scripts/check.sh` does.

## Comments

### The change in `../asd-ste100`

- `PARTICLE_PAIRS` holds the six pairs. The particle rule tries them first,
  then the verbs of `PARTICLE_VERBS`. A pair gives a finding also before a
  determiner (`cuts off the graph`). A pronoun or a noun can stand between
  the verb and the particle. The noun has one word, or two words after a
  determiner (`Cut a carried cache off`).
- `up to` after a pair is no exception, unlike the `up to` of
  `PARTICLE_VERBS` (`takes up to five seconds`). `gives up to five slots` can
  also read as the phrasal verb, so it gives a finding. So does
  `given the tokens up to t`. No line of the two repos has such a `give`.
- Each pair gives its plain verbs in the message:

  | Pair | Plain verbs |
  |---|---|
  | `run out` | end, stop |
  | `run into` | hit, meet |
  | `blow up` | diverge, increase |
  | `cut off` | cut, remove, stop |
  | `carry out` | do, apply |
  | `give up` | stop, release |

- No pair goes to the advisory list. None of the six is a fixed technical
  term: each one has a plain verb that keeps the meaning.
- The `SKILL.md` and the `README.md` of the skill name the six pairs. The
  `SKILL.md` quotes them, so its baseline in the `README.md` is 37 (it was
  31).
- The linter flags the two finds of the sweep in the files before the sweep:
  `run out` in `../burn-stack/src/examples/training.rs` and `blow up` in
  `../burn-stack/src/modules/loss/bce.rs`. The sweep rewrote both files, so
  the files of today have no hit.

### The new hits

The scan reads 493 files with `--enable history --parts docs,comments,messages`.
It finds 18 new hits of a pair in 15 files, and it keeps each finding of the
old linter. Three findings of `give up` get the message of the pair:
`info/mamba-3/architecture-deltas.md` (2) and `src/mamba3/helpers/tests.rs` (1).

This ticket file is in the scan. It puts each quoted pattern in a code span, so
it has no new hit of a pair. It has one new `long-sentence` hit at line 66. There, the linter counts the
`[x]` of a checked box as a word. The `[ ]` of an open box is no word.

In this repo:

| File | Hits | Words |
|---|---|---|
| `examples/reset/majority/tests.rs` | 1 | `run into` (a false finding, see below) |
| `examples/tally/record/dataset.rs` | 1 | `runs out` |
| `examples/tiny-stories/contrast.rs` | 1 | `runs out` |
| `info/mamba-3/mimo-as-batch.md` | 1 | `give up` |
| `info/mamba-3/trapezoid-as-integration.md` | 1 | `give up` |

In `../burn-stack`:

| File | Hits | Words |
|---|---|---|
| `.scratch/glossary-followups/issues/02-prose-sweep.md` | 1 | `runs out` (in a quote) |
| `.scratch/glossary-followups/issues/05-prose-sweep-examples.md` | 1 | `cut off` (in a quote) |
| `.scratch/ste-sweep/issues/02-sweep-modules.md` | 1 | `blow up` (in a quote) |
| `.scratch/ste-sweep/issues/03-sweep-examples.md` | 2 | `run out` (in quotes) |
| `src/examples/mnist/classify.rs` | 1 | `runs out` |
| `src/examples/tiny_stories/dataset.rs` | 1 | `ran out` (a message) |
| `src/examples/tiny_stories/lm.rs` | 3 | `cut off`, `Cut a carried cache off`, `runs out` |
| `src/modules/cache.rs` | 1 | `cut off` |
| `src/modules/layers.rs` | 1 | `runs out` |
| `src/utils/padding.rs` | 1 | `ran out` |

### The false finding

`examples/reset/majority/tests.rs` line 278 has `leaks the pre-reset run into
long-prefix`. There, "run" is a noun. A noun cue after an adjective cannot
tell this line from `sees the budget run out` of the repro, where "run" is a
verb. That needs part-of-speech tagging.

A numeral or `each` before the noun
gives the same false finding (`two runs out of ten`), but no line of the two
repos has it. The code comment of `PARTICLE_PAIRS` and the `README.md` of the
skill name this limit. A sweep ticket can write the sentence of `tests.rs`
again without `run into`.

### The review

The standards review and the spec review found these, and the change fixes
them:

- The `README.md` said that one of the six pairs gives a finding before a
  determiner. Each pair does.
- The `SKILL.md` did not say that the pairs give a finding before a
  determiner.
- The self-tests did not check `run over`, `up to` and the cap of the noun.

Three findings stay open on purpose:

- `give` has no `up to` exception, but the old rule had one. The user chose
  this: a sentence with `gives up to` has two readings.

- `The team carries out an evaluation` gives two hard findings:
  `phrasal-verb-particle` and `nominalization`. Both rules apply, and the
  message of the nominalization gives the better rewrite (`evaluates`).
- With a noun between, a message can name the wrong verbs:
  `runs each row into the buffer` gets "hit, meet". The ticket asks for the
  noun between, and the finding itself is true.
