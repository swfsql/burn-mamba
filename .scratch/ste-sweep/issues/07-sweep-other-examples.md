# 07: Sweep: the other examples

Status: resolved

Blocked by: 01, `.scratch/ste-lint-misses/` 01 and 02 (the linter)

Spec: `../spec.md`. Follow its method, its text that does not change, its kept
findings and its rules for messages and copies.

**Context:** ticket 01 extends the prose step of `scripts/check.sh`. Tickets
02–08 are the batches of burn-mamba, and they do not block each other. Ticket
09 sweeps `.scratch/` last.

**What to build:** the prose of the examples outside `examples/reset/` follows
the `asd-ste100` skill.

- Scope: `examples/`, outside `examples/reset/` and outside each `tmp/`
  directory, 55 files. These are `examples/README.md` and the examples
  common, hnet-stories, mnist-ae, mnist-class, tally and tiny-stories.
- At the start: 98 hard findings (13 in messages) and 100 advisory findings.
- The example tests can match a message. Keep each matched substring.
- A CLI flag, a knob name and a value in a README stay.

**References:**

- Spec: "The method of a batch", "Text that does not change", "Kept
  findings", "Messages and copies" and "A doc that disagrees with the code".
- The terms: `GLOSSARY.md`, `../burn-stack/GLOSSARY.md`, ADR-0008 for a
  quoted source word, and `docs/source-terms.md` before a source word.
- Prior art: `../burn-stack/.scratch/glossary-followups/issues/05-prose-sweep-examples.md`.

- [x] `## Comments` gives the count of hard findings before and after.
- [x] The linter finds zero hard findings in the scope, except the kept
      findings that `## Comments` lists.
- [x] Each substring of a changed message that a test of either repo matches
      stays.
- [x] `## Comments` lists each doc that disagrees with the code, with its
      evidence. The user decides each one.
- [x] `./scripts/check.sh` passes. Run it in the background.

## Comments

A line number of an old phrase is its line at `70922a1a`. A line number of a
new phrase is its line in this change. The paths are relative to `examples/`,
except where a path starts with `examples/` or `../`.

### The counts

The linter runs with `--enable history --parts docs,comments,messages` on the
55 tracked `.rs` and `.md` files of the scope.

| Findings | Before | After |
|---|---|---|
| Hard | 95 | 2 |
| Hard, in messages | 8 | 1 |
| Advisory | 103 | 58 |

The ticket gives 98 hard findings (13 in messages) and 100 advisory findings
at the start. Ticket 06 removed 5 of them: one semicolon in the training-budget
message of each `training.rs` of the scope. That leaves 93 hard findings (8 in
messages). At `70922a1a`, the updated linter finds 95 hard findings (8 in
messages) and 103 advisory findings. So it finds 2 hard findings and 3
advisory findings more than the counts of the ticket.

### The kept findings

| Location | Rule | Match | Reason |
|---|---|---|---|
| `common/context.rs:547` | `long-sentence` (21 words, cap 20) | the `format!` of `context-use.json` | False positive. The string is a JSON template of keys and placeholders, with no prose. A change of a key changes the file format. |
| `tiny-stories/inference.rs:30` | `synonym-rotation` | `starts` and `launch` | False positive. `launch` (line 19) is a noun: the GPU work that the host enqueues. `starts` (line 30) tells the cache that a call of `generate` begins from. The two words name two different actions. |

### The messages

A search of both repos covers the `should_panic(expected = …)` attributes,
the `contains(…)` checks and the string compares. No test matches a part of a
changed message. The old strings are at `70922a1a`.

| Location | New string | Finding |
|---|---|---|
| `common/context.rs:494` | `re-mention: {} letters ({} words) in {…} stories. The words are far-part words of {…}+ letters, in the own prompt and in < {…} of the stories` | long sentence (hard) |
| `common/long_stories.rs:240` | `whole valid: … ({…} characters, {…} stories longer than {cut}). Bits/char {…} (chars 0..{cut} {…}, past {cut} {…})` | long sentence, semicolon (hard) |
| `tiny-stories/contrast.rs:784` | `contrast: … per training step. Prompt …, skip {}. Weight {} (…). lm {}, one-sided {}` | long sentence (hard) |
| `tiny-stories/context.rs:26` and `:44`, `tiny-stories/inference.rs:85` | `no trained model in the artifacts directory. Run with --training first` | semicolon (hard) |
| `tally/depth/tests.rs:268`, `tally/drift/tests.rs:234`, `tally/record/tests.rs:362` | `… reached {…}, so the task does not need …` | em dash |
| `tally/drift/tests.rs:310` | `the linear state matched the gain on knife ({knife:.4}), so this rung has no content` | em dash |
| `tally/record/tests.rs:401` | `f32 held the whole exponential span ({worst}), so the range caveat is wrong` | em dash |
| `tally/record/tests.rs:483` | `a latch for {k} fits at this width, so the rung is not about the register` | em dash |

- The three asserts "reached … so the task does not need …" take the form of
  the reset ladder (ticket 06).
- The first three rows change the printed output of `--context-use`,
  `--long-stories` and the start of the contrast. No test, script or doc of
  either repo reads these lines. Outside the scope,
  `examples/hnet-stories/tmp/1m-a/long.sh:48` greps `^whole valid`. That
  prefix stays. The JSON files do not change.

### The copies

- `The weights that Muon trains (and where the fused ones split).` replaces
  `Which weights Muon took over …` in its three copies: the `training.rs` of
  `mnist-ae`, `mnist-class` and `tiny-stories`. The words come from
  `mnist-class/README.md` ("The weights that Muon trains").
- The doc of `Valid` has two near copies, in `mnist-ae/training.rs` and
  `mnist-class/training.rs`. Both become active ("It captures its forward
  …"), and `falls back to eager` becomes "runs eagerly instead" in both.
- The CUDA paragraphs of `mnist-class/README.md` and `mnist-ae/README.md`
  split at the same place.
- The docs of `PROMPTS` and `PROMPT_TEMPERATURE` have one copy in
  `tiny-stories/inference.rs` and one in `hnet-stories/inference.rs`. Both
  copies take "without a `--prompt` flag" and "without a `--temperature`
  flag".
- "A separator character as a seed would be out of distribution" has a near
  copy in `tiny-stories/README.md` and in `tiny-stories/model.rs`. Both split
  at the colon.
- The `knife` family has two near copies, in `tally/README.md` and
  `tally/drift/dataset.rs`. In both, `is blended in` becomes "the estimate
  mixes the prior with new evidence", and `is aimed again` becomes "the
  generator aims … again".
- `mnist-ae/model.rs:331` (`Held once per virtual layer instead of shared by
  all of them.`) is a near copy of the comment of `mnist-class/model.rs:56`.
  It stays: its text is shorter, and its paragraph has no finding.
- No verbatim copy of a changed paragraph or message is in burn-stack.

### Other word changes

- Each phrasal verb in a paragraph with a finding becomes one verb:
  - `took over` (hard, three copies): "trains", see above.
  - `comes back` (hard, `tally/depth/tests.rs:9`): "returns".
  - `runs out` (hard): "needs more range than f32 has", the words of
    `tally/README.md` (`tally/record/dataset.rs:14`).
    `A new shuffle starts after its last entry` (`tiny-stories/contrast.rs:735`).
  - `goes back into` (hard, `tiny-stories/contrast.rs:47`): "reaches".
  - `go up` (hard, `tiny-stories/contrast/tests.rs:109`): "rise", the verb of
    the test name `raises_delta`.
  - `rules out` (`tally/record/tests.rs:408`): "excludes", as in ticket 06.
  - `carries over from` (`common/context.rs:31`): "keeps from".
  - `leave it out` (`tally/shared/data.rs:13`): "exclude it".
  - `drifts down to the run` (`tally/record/dataset.rs:85`): "falls to the
    level of the run".
  - `cleans up` (`tiny-stories/README.md:198`): "cleans".
  - `read … out of` (`mnist-ae/model.rs:134`): "read … from".
- One word, one meaning:
  - `mnist-ae/training.rs:180` (hard): `every small validation check` becomes
    "Each small validation", because the file uses `validate` for the action.
  - `tiny-stories/README.md:90` (hard): the linter pairs `launched` (the GPU
    work) with `start` (the first character of a story). That is the false
    positive of the kept finding. The passive `are not launched again` changes
    anyway, for its actor: `The host does not enqueue their launches again.`
- A sentence with three or more parts becomes a list. Examples:
  - the README topics of `tiny-stories` and `hnet-stories` in
    `examples/README.md`,
  - the flags of `hnet-stories/cli.rs`,
  - the shortcuts of `tally/README.md` and the order of the `knife` family,
  - the line of the profile in `tiny-stories/contrast/tests.rs`.

### Advisory findings

- A passive becomes active where the actor is relevant. Examples:
  - "`enc_to_z` projects the pooled summary" (`mnist-ae/model.rs`),
  - "The block projects B/C once per group" (`mnist-class/model.rs`),
  - "The loop walks a story in windows" (`tiny-stories/README.md`),
  - "The example writes the corpus knobs" (`tiny-stories/main.rs`).
- An em dash between two clauses becomes a period, a comma with "so" or a
  colon.
- The present perfect `has seen` (`mnist-ae/model.rs`, twice) becomes "sees".
- The advisory findings that stay are of these kinds:
  - a passive that tells a state, for example "is scored", "is unscored",
    "is unused" and "is untied",
  - a passive whose actor the sentence does not need, because the sentence is
    about the object. Examples are "is padded" (`tiny-stories/README.md:99`),
    "is detached" (`tiny-stories/contrast.rs:27`), "is shared", "is held"
    and "are threaded" (`tiny-stories/inference.rs`),
  - the false positive "a gap is elapsed time" (`tally/README.md:198`,
    `tally/drift/dataset.rs:15`). There, `elapsed` is an adjective.
  - an em dash in a short comment of a weight table (`tally/depth/tests.rs`,
    `tally/drift/tests.rs`) or of a config line (`a scalar state — nothing to
    rotate`),
  - `Set up` in the doc of `launch` (`mnist-class/main.rs:53`,
    `tally/shared/mod.rs:30`). The other examples give the same doc with
    `Wire up`, which the linter does not read (see
    [For the linter](#for-the-linter)). One later change can give all these
    copies one verb.

### Docs that disagree with the code

The sweep changes neither side. The user decides each one.

1. `tiny-stories/cli.rs:1` and `tiny-stories/main.rs:24` list four groups of
   the flags of the example: the corpus knobs, the scan options, the
   training-step profiler and the VRAM cap. The `Cli` struct (`cli.rs:31`) and
   the `HELP` text (`cli.rs:14`) have more groups:
   - `--pack-align` and `--grad-checkpointing`,
   - the prompting flags `--prompt`, `--continuations` and `--temperature`,
   - the context flags `--context-use`, `--context-batch`,
     `--context-stories` and `--long-stories`,
   - the `--contrast*` flags.

   A suggestion: give the two lists the groups of the `Cli` struct.
2. The header of `hnet-stories/cli.rs` lists six groups of flags. This change
   gives the same groups as a list. The `HELP` text has two more flags:
   `--chunk-log` (`cli.rs:29`, the chunk ratios of each stage during training)
   and `--long-stories` (`cli.rs:36`, a measure apart from the context use).
   A suggestion: add the two flags to the list.

### Terms

- `tiny-stories/context.rs:3` says "The lead rows are the class latents".
  `common/context.rs` and `common/long_stories.rs` say "lead positions" for
  the same thing. `../burn-stack/GLOSSARY.md` gives
  `row (for a place on the time axis)` as an _Avoid_ word for a **Position**.
  The paragraph has no finding, so it stays.
- `tiny-stories/context/tests.rs:1` says "[`score`] reads the right rows" and
  "the reference reads its last row", for places on the time axis. The test
  code names the same axis `rows` (`context/tests.rs:66`). So the words stay.
  A change of the term also renames the local `rows`.
- `tiny-stories/README.md:187` and the help of `--pack-align`
  (`tiny-stories/cli.rs:18`) say "start each story" of a packed row. There, a
  story is a **Segment**, and `../burn-stack/GLOSSARY.md` gives `story` as an
  _Avoid_ word for a segment. Elsewhere, the README uses "story" for the item
  of the corpus.
- `mnist-ae/training.rs:180` says "Each small validation", because the file
  uses `validate` for the action (see
  [Other word changes](#other-word-changes)). Five docs say "every small
  validation check": `mnist-ae/inference.rs:6`, `mnist-ae/README.md:49`,
  `mnist-class/inference.rs:8`, `mnist-class/README.md:13` and
  `tiny-stories/README.md:225`. Their paragraphs have no finding, so they stay.
  One later change can give the example docs one name.

### For the linter

Three suggestions. Each one agrees with STE.

1. The synonym rule pairs `start` with `launch`, also when `launch` is a
   noun. The probe `It replays one recording, instead of a new launch. A second
   call starts from a zero cache.` gives one hard finding. STE gives each word
   one part of speech. A suggestion: count a rotation only when both words are
   verbs.
2. The strict cap reads a JSON template as a sentence. The `format!` of
   `context-use.json` (`common/context.rs:547`) gives a sentence of 21 words. A
   suggestion: skip a message string that has no word outside its keys and
   placeholders, for example a string that starts with `{{"`.
3. The phrasal-verb rule misses these verbs. A probe of six sentences gives no
   phrasal-verb finding: `Wire up`, `carries over`, `leave it out`,
   `masks them out`, `is blended in` and `cleans up`. A suggestion: add
   `carry over`, `leave … out` and `blend in` to the hard list. Add `wire up`,
   `mask … out` and `clean up` to the technical list.

These misses stay, in paragraphs with no hard finding:

- `Wire up` in the doc of `launch` (`mnist-ae/main.rs:59`,
  `hnet-stories/main.rs:58` and the six reset rungs),
- `masks them out` and `leaves out` (`tally/shared/training.rs:261-264`).

### The check

`./scripts/check.sh` passes on the final tree (exit 0). The log has no warning
and no failed test.

### The review

The two agents of `/mattpocock-skills:code-review` read `git diff 70922a1a`,
one after the other. This change holds the fixes of their findings:

- Each line keeps the width of its file. The doc of `remention_marks` and the
  header of `common/long_stories.rs` had a line of 100 characters or more.
- No rewrite adds a fact:
  - `tally/README.md` keeps "So the `C` port …" in the paragraph of the gate.
    The words "through the gate or the `final_norm`" are gone.
  - The comment of the untied tensors (`mnist-class/model.rs:56`) is a list,
    with no count.
  - The two table cells of `tally/README.md` give "11 gap decays" and
    "11 × 11 × 5" with no new noun.
- The list of the positions with no score keeps its full reading, in
  `tally/README.md` and `tally/shared/data.rs`. The new words are "Such a
  position is a reset, a tie or a symbol that fixes the answer by itself".
- Each actor can act:
  - "The code below measures the far part" (`common/context.rs`),
  - "the block subtracts that estimate" (`tally/drift/tests.rs`),
  - "The first batch records the forward, the backward and the update"
    (`mnist-ae/README.md`, the words of `mnist-class/README.md`).
- The header of `tally/drift/dataset.rs` keeps the link between the geometric
  discount and the small margin.
- `gen_plateau` names "the high value" in its last part, and `drifts down`
  becomes one verb.
- The log lines start each sentence with a word: `re-mention: {} letters …`,
  `Bits/char …`. The contrast line has no label `Row:`, and a period puts
  `lm` and `one-sided` apart from `Weight`.
- The notes of `tiny-stories/README.md` give the untied head its own item, so
  the list stays tight.
- `## Comments` explains the 2 hard findings that the updated linter adds. It
  lists the kind of passive that stays, the second doc that disagrees
  (`hnet-stories/cli.rs`), the near copy of `mnist-ae/model.rs:331` and the
  term "small validation check".
