# 02: The multi-word rules match across a line break

Status: resolved

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

- [x] Each form on two lines gets the same finding as the form on one line, in
      a `.md` paragraph, a `///` doc comment and a `//` comment.
- [x] Two paragraphs, or two comments with code between them, give no match.
- [x] `--selftest` holds the cases of the repro, and it passes.
- [x] The `SKILL.md` of the skill describes the rules as they are.
- [x] `## Comments` lists the new hits in each tracked `.rs` and `.md` file of
      both repos, outside each `tmp/`, with the count of each file. It gives
      the hard hits (the history phrases) apart from the advisory ones. Run
      the linter with `--enable history --parts docs,comments,messages`, as
      `scripts/check.sh` does.

## Comments

### The change in `../asd-ste100`

- Each rule reads the first four words of the next line of its paragraph or
  comment. Before the change, only the particle rule read them. A match must
  start on its own line, so a finding gives the line and the column of its
  first word. Each rule keeps its kind.
- The ticket names three rules, but four more rules had the same miss: the
  soft phrasal verbs (`reach` / `out`), the technical phrasal verbs (`sets` /
  `up`), the nominalization (`perform an` / `analysis`) and `blazing` /
  `fast`. The change fixes them too, because each rule reads the same text. In
  the two repos, they give one new hit: `gives back`, advisory.
- The exception of the passive also reads the next line. So `is used` /
  `for training` gives no finding, as on one line. The two repos lose no
  finding.
- Four more cases make the text of two lines read as the text of one line:
  - The spaces at the end of a line (a Markdown hard break) do not stop a
    match. A pattern with one literal space (`no longer`) needs this.
  - A code span that is open at the end of a line hides the words of the next
    line. Before the change, the particle rule read the code of such a span
    as prose.
  - A heading has no next line in its paragraph, and a next line that opens a
    deeper quote starts a new paragraph. Before the change, the particle rule
    read the next line in both cases. A line with fewer `>` is a lazy line of
    the paragraph, so the rules read it.
  - The `description:` of a YAML front matter reads the next line of its
    value.
- The self-tests hold the repro in the three forms (`.md`, `///`, `//`), with
  the line, the column and the kind of each finding. They also hold two
  paragraphs, two comments with code between them, a doc comment before a
  comment, and the cases above.
- The `SKILL.md` says that the words of one finding can be on two lines of a
  paragraph or a comment. The `README.md` says the same, and it gives the
  limits. The baseline of the `SKILL.md` stays 37.
- The linter flags the miss that the sweep found, in the file before the sweep:
  `are decompressed` at line 31 of
  `../burn-stack/src/examples/tiny_stories/dataset.rs`, at the parent of
  burn-stack commit `7f6259b`. The sweep rewrote the sentence, so the file of
  today has no hit.

### The new hits

The scan reads 480 files with `--enable history --parts docs,comments,messages`:
295 in this repo and 185 in `../burn-stack`. It skips each path with a `tmp/`
part. It finds 65 new hits in 52 files, and it keeps each finding of the old
linter. No compound tense gives a new hit.

The one hard hit is a quote in a spec. `scripts/check.sh` skips `.scratch/`.
Its words also give an advisory passive, so the table of this repo below has
the same row.

| File | Hits | Words |
|---|---|---|
| `.scratch/ste-sweep/spec.md` | 1 | `were added` (300) |

Of the 64 advisory hits, 63 are passives and one is the technical phrasal
verb `gives back`. Some passives are adjectives (`are unchanged`,
`is unbiased`). The rule flags these words on one line too.

In this repo, 41 hits in 33 files:

| File | Hits | Words |
|---|---|---|
| `.scratch/arch-review-backlog/issues/04-k-backward.md` | 1 | `is shared` (53) |
| `.scratch/arch-review-backlog/issues/06-cache-field-lists.md` | 1 | `is reverted` (230) |
| `.scratch/arch-review-backlog/issues/11-cache-traversal.md` | 1 | `is written` (34) |
| `.scratch/ste-sweep/spec.md` | 1 | `were added` (300) |
| `benches/layer.rs` | 1 | `is measured` (3) |
| `examples/hnet-stories/inference.rs` | 1 | `is given` (88) |
| `examples/reset/README.md` | 1 | `is scored` (325) |
| `examples/reset/rotor/dataset.rs` | 1 | `are decorrelated` (139) |
| `examples/reset/spinor-product/tests.rs` | 4 | `is unused` (172), `is encoded` (409), `is fitted` (557), `is renormalised` (881) |
| `examples/reset/spinor/tests.rs` | 1 | `is fitted` (504) |
| `examples/reset/swap/tests.rs` | 1 | `is fitted` (469) |
| `examples/tally/drift/tests.rs` | 1 | `is projected` (7) |
| `examples/tiny-stories/contrast.rs` | 1 | `is separated` (697) |
| `examples/tiny-stories/contrast/tests.rs` | 1 | `is padded` (365) |
| `examples/tiny-stories/inference.rs` | 2 | `is given` (56), `is set` (190) |
| `examples/tiny-stories/inference/tests.rs` | 1 | `is held` (101) |
| `examples/tiny-stories/model.rs` | 1 | `is scored` (31) |
| `info/mamba-3/mimo-as-batch.md` | 1 | `is known` (538) |
| `info/mamba-3/rotation-as-optimization.md` | 1 | `is conjugated` (136) |
| `src/mamba3/double_ssd/double_ssd/tests.rs` | 2 | `are untouched` (317), `are unrolled` (667) |
| `src/mamba3/helpers.rs` | 1 | `is dropped` (470) |
| `src/mamba3/mamba3.rs` | 1 | `is enabled` (710) |
| `src/mamba3/positive/scan.rs` | 2 | `is allowed` (7), `is allowed` (11) |
| `src/mamba3/quat_scan/quat_scan.rs` | 1 | `is unbiased` (131) |
| `src/mamba3/rotation/mod.rs` | 1 | `is detached` (802) |
| `src/mamba3/rotation/rope.rs` | 2 | `are unchanged` (124), `are unchanged` (130) |
| `src/mamba3/rotation/tests.rs` | 1 | `is pinned` (2079) |
| `src/mamba3/scan_options.rs` | 1 | `is rounded` (127) |
| `src/mamba3/single_ssd/single_ssd/tests.rs` | 1 | `are untouched` (442) |
| `src/mamba3/single_ssd/ssd/serial.rs` | 1 | `are zeroed` (234) |
| `src/mamba3/single_ssd/ssd/serial_recalculated/diag.rs` | 1 | `is replaced` (8) |
| `src/mamba3/single_ssd/ssd/serial_recalculated/serial_recalculated.rs` | 2 | `is defined` (16), `are zeroed` (324) |
| `src/mamba3/tap_pattern.rs` | 1 | `is transported` (13) |

In `../burn-stack`, 23 hits in 19 files:

| File | Hits | Words |
|---|---|---|
| `.scratch/arch-review-backlog/issues/13-untied-tiling.md` | 1 | `are untied` (188) |
| `.scratch/glossary-followups/issues/01-docs-redo.md` | 1 | `is listed` (91) |
| `.scratch/glossary-followups/issues/09-examples-renames.md` | 1 | `are permitted` (10) |
| `.scratch/ste-sweep/issues/03-sweep-examples.md` | 1 | `is bucketed` (120) |
| `src/examples/cli.rs` | 1 | `was saved` (454) |
| `src/examples/tiny_stories/dataset.rs` | 2 | `is truncated` (722), `is scored` (787) |
| `src/examples/tiny_stories/lm.rs` | 1 | `is truncated` (183) |
| `src/examples/training.rs` | 1 | `is sized` (222) |
| `src/modules/bidi.rs` | 1 | `is written` (8) |
| `src/modules/hnet/smooth.rs` | 1 | `is clamped` (49) |
| `src/modules/layer.rs` | 1 | `is called` (35) |
| `src/modules/layers.rs` | 2 | `is suppressed` (295), `is gated` (300) |
| `src/modules/loss/l2warp.rs` | 1 | `is pulled` (4) |
| `src/modules/network.rs` | 1 | `is rounded` (314) |
| `src/modules/residuals.rs` | 1 | `is untracked` (309) |
| `src/utils/capture/weights.rs` | 1 | `gives back` (13) |
| `src/utils/class/mod.rs` | 1 | `are inserted` (17) |
| `src/utils/mod.rs` | 1 | `are spliced` (39) |
| `src/utils/padding.rs` | 3 | `are spliced` (22), `is scattered` (26), `is untouched` (216) |

### The review

The standards review and the spec review found these, and the change fixes
them:

- The words of a heading and the next line matched, and so did the words of
  a line and a deeper quote. The particle rule had this leak before the
  change, and the change gave it to each rule. No hit of the scan came from
  it, and a scan after the fix gives the same hits.
- The new sentences used "goes on", a phrasal verb. They now say "is open at
  the end of a line".
- `check` read the state of the main loop to get its next-line words. Now
  each call gives these words.
- The sentence of the `SKILL.md` came after the sentences on phrasal verbs, so
  a reader could apply it to those rules only. It now starts with "For each
  rule".
- Smaller points: the name `line_end` replaces `end`, which was next to
  `m.end()`. The self-tests have no `chr(10)`. This ticket says why one row is
  in two tables, and "the miss that the sweep found" replaces "the find of the
  sweep".

These findings stay open on purpose:

- The linter uses "goes on" for "continues" in about 20 more comments, and
  its `README.md` uses it once. A rename of all of them is outside this
  ticket. The linter does not flag "go on", because `on` is not in
  `PARTICLES`.
- One phrasal verb on two lines can give two findings: `come` / `back up`
  gives `come back up` on the first line and `back up` (technical) on the
  second line. The old linter gives the same two findings. A fix must map
  the words of the next line to its columns.
- A backslash hard break (`no\` / `longer`) gives no finding. The ticket does
  not ask for it.
- `check` takes 11 parameters. It took 11 before the change too.
