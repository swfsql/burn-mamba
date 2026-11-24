# 06: Sweep: the reset examples

Status: resolved

Blocked by: 01, `.scratch/ste-lint-misses/` 01 and 02 (the linter)

Spec: `../spec.md`. Follow its method, its text that does not change, its kept
findings and its rules for messages and copies.

**Context:** ticket 01 extends the prose step of `scripts/check.sh`. Tickets
02–08 are the batches of burn-mamba, and they do not block each other. Ticket
09 sweeps `.scratch/` last.

**What to build:** the prose of the reset examples follows the `asd-ste100`
skill.

- Scope: `examples/reset/`, outside each `tmp/` directory, 44 files.
- At the start: 157 hard findings (6 in messages) and 124 advisory findings.
- `examples/reset/quintic/HANDOFF.md` has 75 hard findings. It is the handoff
  of an open thread. Keep each fact, each number and each open question.
- The example tests can match a message. Keep each matched substring.

**References:**

- Spec: "The method of a batch", "Text that does not change", "Kept
  findings", "Messages and copies" and "A doc that disagrees with the code".
- The terms: `GLOSSARY.md`, `../burn-stack/GLOSSARY.md`, ADR-0008 for a
  quoted source word, and `docs/source-terms.md` before a source word.
- `examples/README.md`: the examples and their knobs.
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

A line number of an old phrase is its line at `dc063402`. A line number of a
new phrase is its line in this change.

### The counts

The linter runs with `--enable history --parts docs,comments,messages` on the
44 tracked `.rs` and `.md` files of the scope.

| Findings | Before | After |
|---|---|---|
| Hard | 158 | 1 |
| Hard, in messages | 6 | 0 |
| Advisory | 132 | 86 |

The ticket gives 157 hard findings and 124 advisory findings at the start. At
`dc063402`, the updated linter finds 158 hard findings (6 in messages) and 132
advisory findings. `examples/reset/quintic/HANDOFF.md` goes from 75 hard
findings to 0.

### The kept finding

| Location | Rule | Match | Reason |
|---|---|---|---|
| `examples/reset/majority/tests.rs:278` | `phrasal-verb-particle` | `run into` | False positive. In `leaks the pre-reset run into long-prefix`, the word `run` is a noun, and `into` belongs to `leaks … into`. |

### The messages

A search of both repos covers the `should_panic(expected = …)` attributes,
the `contains(…)` checks and the string compares. No test matches a part of
a changed message. The old strings are at `dc063402`. The paths of the table
are relative to `examples/reset/`.

| Location | New string | Finding |
|---|---|---|
| the `training.rs` of each rung | `the training reached its budget, so it stops` | semicolon (hard) |
| the `main.rs` of each rung | `the flags enable neither training nor inference` | passive |
| `quintic/dataset.rs:315` | `d and t generate A₅` | passive |
| `quintic/main.rs:122` | `the persisted model config is for the other group. Pass the matching --group` | passive, em dash |
| `majority/tests.rs:328`, `rotor/tests.rs:557` and `:609`, `spinor/tests.rs:671`, `spinor-product/tests.rs:757`, `swap/tests.rs:780` | `… reached {…}, so the task does not need …` | em dash |
| `swap/tests.rs:722` | `the two lifts did not cancel ({worst_cancel:.3}). Check the double cover` | em dash |

### The copies

- The two messages of `training.rs` and `main.rs` have verbatim copies in
  five other examples: `hnet-stories`, `mnist-ae`, `mnist-class`,
  `tiny-stories` and `tally/shared`. These copies change in this ticket too.
  So ticket 07 starts 5 hard findings lower: one semicolon in each of its
  `training.rs` files.
- Three near copies of the "reached … — the task does not need …" assert are
  in the scope of ticket 07: `examples/tally/depth/tests.rs:268`,
  `examples/tally/drift/tests.rs:233` and `examples/tally/record/tests.rs:361`.
  They stay. Ticket 07 can give them the form of the table above.
- In the scope, these sets of near copies get one form:
  - `Every weight is written in closed form` has seven copies: five test docs
    and the module headers of `spinor-product/tests.rs` and `swap/tests.rs`.
    Each copy becomes "The test writes every weight in closed form", the
    words of `README.md`.
  - The doc of `LEVELS` ("That is a finite partition of the output space of
    the block, so …") splits in `spinor/tests.rs`, `spinor-product/tests.rs`
    and `swap/tests.rs`.
  - The doc of `D_MODEL` ("Two, the floor for a three-symbol alphabet")
    splits in `spinor/tests.rs` and `swap/tests.rs`.
  - "The axis is projected per head" (`spinor/model.rs`) and "The generators
    are projected per head" (`README.md`) become active: "The block projects
    …".

### History phrases

Two phrases of the history rule change. In each one, the phrase compares two
configurations, not two states of the code.

- `spinor-product/tests.rs:230`: `the turn that the write can no longer sit
  before` becomes "the turn, because in a single micro-step the write cannot
  sit before it".
- `swap/tests.rs:793`: `is no longer linearly readable` becomes "is not
  linearly readable".

`quintic/HANDOFF.md` is the work log of an open thread. Its state phrases stay,
for example "state unchanged since `6a4e6ef`" and "Nothing in `src/` changed".
The history rule does not flag them.

### Other word changes

- Each phrasal verb in a paragraph with a finding becomes one verb:
  - `keeps it off` (hard, `swap/tests.rs:923`) becomes "disables it".
  - `pass it on` (`spinor-product/tests.rs:870`) becomes "give it to the
    layer above".
  - `gets past this bound` (`swap/tests.rs:792`) becomes "escapes this bound".
  - `rules out` (`quintic/HANDOFF.md`, twice) becomes "excludes".
  - `so the leak is priced in` (`quintic/HANDOFF.md`, Next steps) becomes
    "so that training pays for the leak", the words of `README.md:1058`. Its
    copy `Training at 96 prices it in` (Findings, item 3) changes with it.
- `README.md:24` keeps the verb `switch … off`. The file uses it four more
  times, in paragraphs with no finding. One verb for this action in each file
  wins over one plain verb in one sentence. The new sentence is "The rung
  switches the residual off". See [For the linter](#for-the-linter).
- One word, one meaning:
  - `quintic/tests.rs` uses `erases` for the wipe of the state, and `cancel`
    for two lifts that sum to zero. So `the norm removes the scale` becomes
    "the norm undoes the scale".
  - `spinor-product/tests.rs` uses `fix` for "hold" (`It is fixed at 1`,
    line 50). So `No scale can fix the rotation` (line 236) takes the verb of
    the module header, "repair". `the counts fix the element up to a sign`
    (line 1174) becomes "determine".
  - `quintic/HANDOFF.md` uses `delete` for the file. So
    `needed the leak removed by hand` becomes "needed a hand edit that stops
    the leak".

### Advisory findings

- A passive becomes active where the actor is relevant. Examples: "The tests
  measure the ceilings above on all four components" (`README.md`) and
  "[`handmade`] and [`handmade_composed`] share them"
  (`spinor-product/tests.rs`).
- An em dash between two clauses becomes a period, a comma with "so" or a
  colon.
- The advisory findings that stay are of these kinds:
  - an em dash in a heading, a table cell or a short comment of a weight
    table,
  - a passive that tells a state, for example "is fixed at 1", "is scored"
    and "are decorrelated",
  - the present perfect "has stalled" and "has accumulated", which tell the
    current state,
  - a false positive of `phrasal-verb-technical` at `rotor/tests.rs:21`:
    `± turn the phase by one detent. The two heads read the accumulated turn
    on two axes`. The two words of the match are in two sentences.

### Docs that disagree with the code

The sweep changes neither side. The user decides each one.

1. `examples/reset/README.md:392` and `:442` say that the trained
   `--rotation complex` run of `reset-spinor` gets more than the counts
   ceiling on `random`. Line 442 says "even more than the 71.4% of a table
   given the exact counts". But the table of the same section gives 71.3% for
   that run (`:375`) and 71.4% for the counts ceiling (`:372`). The doc of
   `counts_ceiling_is_the_abelian_limit` (`spinor/tests.rs:687`) agrees with
   the table: the run "does not clear this ceiling either". A suggestion:
   measure the run again, or make the two sentences agree with the table.
2. `examples/reset/quintic/HANDOFF.md`: the heading "A. reset-quintic (state
   unchanged since `6a4e6ef`)" and the file list of "Files and surface" have
   no `examples/reset/quintic/cli.rs`. Commit `9f8267f7` ("Examples: no env
   vars, one cli.rs per example") adds that file after `7437ba2e`, the last
   rewrite of the handoff. The CLI list of the handoff agrees with the flags
   of `cli.rs`, and `--train-length` defaults to `SEQ_LENGTH = 96`. A
   suggestion: add `cli.rs` to the file list. A change of the heading also
   changes its anchor.

### For the linter

Two suggestions. Each one agrees with STE Rule 9.3.

1. The phrasal-verb rule reads the noun `run` and the preposition `into` as
   one phrasal verb. The probe
   `A decay near 1 leaks the pre-reset run into the next family.` gives one
   hard finding. A suggestion: do not read `run` as a verb after a determiner
   or an adjective.
2. The phrasal-verb rule misses verbs with a particle when a word stands
   between them. A probe of five sentences gives no finding: `pass it on`,
   `gets past it`, `keeps changelogs out of`, `prices it in` and
   `Switching it off`. A suggestion: add `pass … on`, `get past`,
   `keep … out` and `price … in` to the hard list. Add `switch … off` to the
   technical list.

These misses stay. Each one is in a paragraph with no finding, or it keeps
the verb of its file:

- `No residual or embedding gets past it` (`majority/tests.rs:337`),
- `keeps changelogs out of the docs proper` (`quintic/HANDOFF.md:4`),
- `Switching it off is structural`, with five copies: `README.md:38`, and
  the `model.rs` of `majority` (line 49), `rotor` (line 49), `spinor`
  (line 79) and `swap` (line 63),
- `switch … off` and `switch … on` in `README.md` (lines 24, 176, 266 and
  272).

When the linter reads `switch … off`, one later change can give these copies
one plain verb, for example "disable".

### Terms

`quintic/HANDOFF.md` uses `step` alone for one transition of the recurrence:
`Per step`, `One step of Rotor4D`, the `per-step transition` of the probe
table, and more. It also uses `6000 steps` for training steps.
`GLOSSARY.md` gives `step (alone)` as an _Avoid_ word for a **Micro-step**.
`../burn-stack/GLOSSARY.md` also gives it for a **Step**. The words stay, and
the rewrite adds no bare `step`. The user decides whether the handoff takes
"micro-step" and "training step".

### Follow-ups

The two messages of the table above have one copy in each of 11 `main.rs`
and 11 `training.rs` files. So one change of their words touches 22 files.
A function in `examples/common/` beside `common::cli::HELP` can hold each
message once. That is a code change, outside this sweep.

### The check

`./scripts/check.sh` passes on the final tree (exit 0). The log has no
warning and no failed test.

### The review

The two agents of `/mattpocock-skills:code-review` read `git diff dc063402`,
one after the other. This change holds the fixes of their findings:

- The seven copies of the closed-form doc keep the original claim: "The test
  writes every weight in closed form". The words "has a closed form" claim
  less.
- `quintic/main.rs:92`: each sentence of the LR comment has a verb.
- `quintic/tests.rs:340`: "undoes" in place of "cancels", because the file
  uses `cancel` for the two lifts.
- `spinor-product/tests.rs:411`: "share them", for the plural "The parts".
- The training-budget message gets a subject in its 11 copies.
- `README.md:364` and `:532`: "The gap is in the rotation channels" keeps
  the link between the parameter counts and the channels.
- `README.md:640`: the colon stays. The sentence after it is the content of
  "the sharper statement".
- `spinor/tests.rs:681`: "When a block writes …, it can see no more than this"
  prevents a wrong first parse.
- `quintic/HANDOFF.md`: the cause of the long-length failure is the mixing,
  not the head. The line on `:603` of the DeltaProduct TeX names no actor,
  because the paper measures DeltaNet[−1,1] there ("For $S_5$, even 10
  layers proved insufficient").
- No new paragraph starts with a reference to the paragraph before it
  (`spinor-product/tests.rs:1146`, `quintic/model.rs:38`,
  `spinor/model.rs:44`, `README.md:703` and the "This `B`" of
  `spinor-product/tests.rs:231`).
- `README.md:24` keeps the verb of its file (see
  [Other word changes](#other-word-changes)).
- The claim about the near copies covers only the sets that it lists. The LR
  comments of the other rungs are different texts.

### The decisions of the user

1. Docs that disagree, item 1: the two sentences change, so that they agree
   with the table.
   - `README.md:392` gives "On `random`, it gets more than the twin (71.3%
     against 60.6%), for the reason that `reset-swap` gives. It stays just
     below the counts ceiling (71.4%)."
   - `README.md:442` gives "It does better on `random`: 71.3%, close to the
     71.4% of a table given the exact counts."
2. Docs that disagree, item 2: `cli.rs` joins the file list of the handoff.
   The heading stays.
3. Terms: the handoff takes "micro-step" for one transition of a crate kind,
   and "training step" for one step of the optimiser. In the probe, one step
   of the recurrence reads one token (`M[tok]`). So the probe table says
   "per-token transition". These uses of `step` stay, because each one has a
   different sense:
   - `real step` and `step algebra` (lines 147 and 148), the step size of
     `info/mamba-3/rotation-as-optimization.md`,
   - the verb `steps` (line 43),
   - the heading `Next steps (B), in order` and `step B2` of the plan,
   - the `step` method of the block in "Forward/`step`/gradient parity tests".
     It is a code span now.
4. Follow-ups: `examples/common/mod.rs` holds each message once.
   `print_nothing_to_run` prints the message and the `--help` text.
   `print_budget_reached` prints the other message. The 11 launchers and the
   11 training loops call them, and the output of each example stays the
   same.
