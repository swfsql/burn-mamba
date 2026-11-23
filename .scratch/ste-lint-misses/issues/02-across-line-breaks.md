# 02: The multi-word rules match across a line break

Status: ready-for-agent

Blocked by: none

**Context:** the STE sweep of both repos found two misses of the linter of the
`asd-ste100` skill. Ticket 01 fixes the first one, and this ticket fixes the
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

**What to build:** three rules also match when their words are on two lines of
one paragraph or one comment. These rules are the passive (`passive-voice`),
the compound tense (`present-perfect`) and the history phrases of two or more
words (`history`). The phrasal-verb rule matches across a line break already, and it
is the reference. The sweep found the miss in burn-stack ticket 03: "only the
row groups up to the requested rows are / decompressed", in the header of
`../burn-stack/src/examples/tiny_stories/dataset.rs`. After a rewrap put both
words on one line, the linter flagged the passive.

## The repro

Run the linter with `--enable history` on each form. A `/` marks the line
break. Each form is one paragraph of a `.md` file, one `///` doc comment and
one `//` comment of a `.rs` file:

| Text | On one line | On two lines |
|---|---|---|
| `… up to the requested rows are / decompressed.` | `are decompressed`, advisory | none |
| `The run has / finished.` | `has finished`, advisory | none |
| `The cache is no / longer valid.` | `no longer`, hard | none |
| `… and then the cursor goes / back to the first window.` | `goes back`, hard | `goes back`, hard (the reference) |
| `The cache was / previously valid.` | `previously`, hard | `previously`, hard (one word) |

## Rules

- The words of two different paragraphs never match. The same holds for two
  comments with a line of code between them.
- A finding across a line break gives the line and the column of its first
  word, as the phrasal-verb rule does.
- Each rule keeps its kind: the passive and the compound tense stay advisory,
  and the history phrases stay hard.

## References

- The `SKILL.md` of the skill describes the rules. Its text must stay true
  after the change.
- The `--selftest` option of the linter holds the self-tests. Add the cases of
  the repro there, in the `.md` form and the two `.rs` forms.

- [ ] Each form on two lines gets the same finding as the form on one line, in
      a `.md` paragraph, a `///` doc comment and a `//` comment.
- [ ] Two paragraphs, or two comments with code between them, give no match.
- [ ] `--selftest` holds the cases of the repro, and it passes.
- [ ] The `SKILL.md` of the skill describes the rules as they are.
- [ ] `## Comments` lists the new hits in each tracked `.rs` and `.md` file of
      both repos, outside each `tmp/`, with the count of each file. It gives
      the hard hits (the history phrases) apart from the advisory ones. Run
      the linter with `--enable history --parts docs,comments,messages`, as
      `scripts/check.sh` does.

## Comments
