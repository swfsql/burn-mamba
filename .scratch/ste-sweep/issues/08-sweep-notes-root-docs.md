# 08: Sweep: the notes and the root docs

Status: resolved

Blocked by: 01, `.scratch/ste-lint-misses/` 01 and 02 (the linter)

Spec: `../spec.md`. Follow its method, its text that does not change, its kept
findings and its rules for messages and copies.

**Context:** ticket 01 extends the prose step of `scripts/check.sh`. Tickets
02–08 are the batches of burn-mamba, and they do not block each other. Ticket
09 sweeps `.scratch/` last. burn-stack ticket 05 sweeps the root docs of
burn-stack.

**What to build:** the prose of the derivation notes, the docs, the benches
and the root files follows the `asd-ste100` skill.

- Scope: `info/`, `docs/`, `benches/`, `CLAUDE.md`, `README.md`,
  `GLOSSARY.md` and `build.rs`, 27 files. `bench.md` and `kernels.md` stay
  (spec, "Out of Scope").
- At the start: 76 hard findings and 60 advisory findings. No message has a
  finding.
- Each note of `info/` has a float64 check in `scripts/`, with check numbers
  that match its sections. Keep each section, each heading and each number.
- An ADR keeps each decision, each option and each reason.
- `GLOSSARY.md` keeps its terms, their meanings and its _Avoid_ entries.
- Each rule of `CLAUDE.md` keeps its exact meaning. Read
  `/mattpocock-skills:writing-for-agents` before you change `CLAUDE.md`.
- A known kept finding: `CLAUDE.md:65` quotes a banned phrase of the history
  rule.
- `CLAUDE.md` and `../burn-stack/CLAUDE.md` share some rules word for word.
  Record each changed copy under `## Comments` of burn-stack ticket 05.

**References:**

- Spec: "The method of a batch", "Text that does not change", "Kept
  findings", "Messages and copies" and "A doc that disagrees with the code".
- The terms: `GLOSSARY.md`, `../burn-stack/GLOSSARY.md`, ADR-0008 for a
  quoted source word, and `docs/source-terms.md` before a source word.
- Prior art: `../burn-stack/.scratch/glossary-followups/issues/02-prose-sweep.md`.

- [x] `## Comments` gives the count of hard findings before and after.
- [x] The linter finds zero hard findings in the scope, except the kept
      findings that `## Comments` lists.
- [x] `## Comments` lists each doc that disagrees with the code, with its
      evidence. The user decides each one.
- [x] `./scripts/check.sh` passes. Run it in the background.

## Comments

### From burn-stack ticket 05

- Burn-stack ticket 05 gives the **Waiting** rule of `../burn-stack/CLAUDE.md`
  the words of the rule of `CLAUDE.md`, with one difference. The burn-stack
  rule says "one part of a chain", and this rule says "one step of a chain".
  **Step** is a term of `../burn-stack/GLOSSARY.md`: the recurrent call. Its
  _Avoid_ entry is "step (alone, for one recurrence step or for one training
  step)". So the copy of this repo can also say "part". Then the two copies
  are equal.
- For burn-stack ticket 05, the user selects zero hard findings. No code span
  hides a phrase from the linter. A record can remove a historical phrase,
  with a hint that the git history keeps it.
- So the rule on changelog prose of `../burn-stack/CLAUDE.md` gives "an old
  state next to the current one" in place of a quote of a banned phrase. The
  rule of this repo has the same quote. With these words, it can lose the
  kept finding that the body of this ticket names.
- Burn-stack ticket 05 closes before this ticket runs. For each shared rule
  that this ticket changes, write a new burn-stack ticket in
  `../burn-stack/.scratch/ste-sweep/issues/`. That ticket applies the change
  to `../burn-stack/CLAUDE.md`. The record of burn-stack ticket 05 gives the
  reason.

### The counts

The linter runs with `--enable history --parts docs,comments,messages` on the
27 tracked `.rs` and `.md` files of the scope.

| Findings | Before | After |
|---|---|---|
| Hard | 77 | 9 |
| Hard, in messages | 0 | 0 |
| Advisory | 63 | 45 |

The ticket gives 76 hard and 60 advisory findings at the start, at
`9a8958ec`. A probe lints the scope at `9a8958ec` with the installed linter.
It finds 78 hard and 63 advisory findings. So the two tickets of
`.scratch/ste-lint-misses/` add 2 hard findings and 3 advisory findings.
Ticket 03 (`27b4f4d9`) splits the long sentence of ADR-0005. That leaves 77
hard findings at `f3ad86c7`.

A line number of an old phrase is its line at `f3ad86c7`. A line number of a
new phrase is its line in this change. The paths of `info/` notes give only
the file name. The git history of this ticket gives each old phrase.

### The kept findings

Eight kept findings are false positives. One keeps a precision
(`benches/layer.rs:212`).

| Location | Rule | Match | Reason |
|---|---|---|---|
| `README.md:272` | `synonym-rotation` | `launches` and `starts` (`:185`) | `launches` is a noun: the kernel launches that `./kernels.sh` counts. `starts` tells the first call of a loop. Ticket 07 lists the same misread. |
| `benches/layer.rs:212` | `synonym-rotation` | `starts` and `launch` (`:227`) | Precision. `launch` is the GPU term: a backend enqueues a kernel. `starts` tells when criterion takes its first timed sample. One word for both actions loses the GPU term. Both words are verbs, so suggestion 1 of ticket 07 does not clear this finding. |
| `architecture-deltas.md:101` | `synonym-rotation` | `erases` and `removes` | `removes` (`:42`, `:43`, `:374`) names a part of the Mamba-2 block that Mamba-3 does not have. `erases` tells what a norm hides: a rescaling of `y` or of the gate. The heading of §7 says `erases`, and the table row `:101` copies the heading. |
| `trapezoid-as-integration.md:343` | `synonym-rotation` | `stop` and `terminates` (`:249`, `:315`) | `terminates` tells that the expansion of §5 is finite. `stop being an integrator` tells that the taps get another role at `u > 1`. |
| `gate-as-positive-system.md:395` | `long-paragraph` | the reference of Baccelli et al. | The linter reads the period of each author initial as the end of a sentence. |
| `mimo-as-batch.md:533` | `long-paragraph` | the reference of Siems et al. | The same misread. |
| `trapezoid-as-integration.md:528` | `long-paragraph` | the reference of Siems et al. | The same misread. |
| `rotation-as-optimization.md:798` | `long-paragraph` | the reference of Siems et al. | The same misread, and the abbreviation `Prop.` |
| `rotation-as-optimization.md:823` | `long-paragraph` | the references of Mescheder et al. and Daskalakis et al. | The same misread. |

The kept finding of the ticket body (`CLAUDE.md:65`) is gone (see
[The copies](#the-copies)).

### The copies

- The **Waiting** rule of `CLAUDE.md` says "one part of a chain", the word of
  `../burn-stack/CLAUDE.md`. So the two copies are equal. The reason is the
  _Avoid_ entry of **Step** (see
  [From burn-stack ticket 05](#from-burn-stack-ticket-05)).
- The rule on changelog prose of `CLAUDE.md` gives "no old state next to the
  current one" in place of the quote of a banned phrase. These are the words
  of `../burn-stack/CLAUDE.md`. The rule keeps its meaning.
- The two rules on changelog prose are not word-for-word copies, before or
  after this change. Each one keeps its other words.
- `../burn-stack/CLAUDE.md` needs no change, so this ticket writes no
  burn-stack ticket.
- No other changed paragraph has a copy in burn-stack. The Notation section
  of `../burn-stack/CLAUDE.md` has its own shorter text.
- In the scope, two notes get the same change in three places:
  - "is not studied further, as orthogonal to …", in the abstracts of
    `mimo-as-batch.md` and `trapezoid-as-integration.md`,
  - "Otherwise the change cannot use the pathway" in §10.4 of
    `mimo-as-batch.md`, and "… cannot use that pathway" in §10.2 of
    `trapezoid-as-integration.md`. Each keeps the determiner of its old text.
  - the one-line summary that "is half correct", in §3 of both notes. Both
    split at the colon.

### Other word changes

- Each phrasal verb in a paragraph with a finding becomes one verb:
  - `architecture-deltas.md:72` (hard): "What Mamba-3 loses".
  - `mimo-as-batch.md:478` and `trapezoid-as-integration.md:477` (hard):
    "Otherwise the change cannot use the pathway" (or "that pathway").
  - `architecture-deltas.md:536` (hard): "This crate intentionally does not
    use that axis".
  - `rotation-as-optimization.md:548` (hard): "returns".
  - `mimo-as-batch.md:50` and `trapezoid-as-integration.md:250`: "restores".
  - `gate-as-positive-system.md:370`: "enable `κ`".
  - `gate-as-positive-system.md:294`, a miss of the linter: "new evidence
    enters the estimate".
  - `mimo-as-batch.md:37` and `trapezoid-as-integration.md:36`, a miss of
    the linter: "is not studied further, as orthogonal". The words keep both
    parts of the old verb: no more study, and the reason.
- One word, one meaning:
  - In `architecture-deltas.md`, Mamba-3 `removes` a part of the block. Four
    old words change: `:42`, `:43`, `:305` and `:325`. A norm `erases` a
    rescaling, so `:383` says `erases exactly that`.
  - `architecture-deltas.md:518` (hard): "check", the verb of the script
    (`:31` in this change).
  - `trapezoid-as-integration.md:240`: the second installment "never
    subtracts". The next sentence gives the same fact: no negative weight.
  - `rotation-as-optimization.md:590` (hard): the paragraph splits, and the
    linter gives no finding there after the split. `deletes` stays. The
    rank-one `erase` of the delta rule is a different action.
- History: `architecture-deltas.md:197` (hard) says "So, unlike the
  unnormalised score, the score `C^⊤B` is not a product of two learned
  scales". The unnormalised score of §3.3 is the other side of the compare.
- A sentence with three or more parts becomes a list:
  - the reset rungs in `README.md`,
  - the three prior readings in the novelty note of
    `trapezoid-as-integration.md`,
  - the standard facts in the novelty note of `rotation-as-optimization.md`.
- A sentence with no subject gets one: "The script verifies this over the
  whole closed disc" (§4) and "The script verifies it" (§8), in
  `rotation-as-optimization.md`. More such sentences stay, in paragraphs with
  no finding. Examples are `mimo-as-batch.md:319` and
  `trapezoid-as-integration.md:255`.
- Each long paragraph splits by topic. Each split keeps the section, its
  heading and its number. In a list item, the second paragraph has the indent
  of the item.

### Advisory findings

- A passive becomes active where the actor is relevant. Examples:
  - "The script verifies …", in six places of the notes,
  - "The bench measures it" (`benches/layer.rs:3`),
  - "The initialisation sets the masks" (`mimo-as-batch.md` §6),
  - "The block adds the bias before the rotation" (`architecture-deltas.md`
    §4),
  - "The product `∏ d` over `u` micro-steps transports a **lag-`u`** tap"
    (`gate-as-positive-system.md` §3).
- The advisory findings that stay are of these kinds:
  - a passive that tells a state, for example "is fixed", "is pinned", "is
    unbounded", "are widened" and "is learned",
  - a passive whose actor the text does not name, for example "is
    documented", "are compiled in", "is resolved" and "is measured, not
    argued",
  - the heading "6.2 Where the decay is substituted" of
    `gate-as-positive-system.md`, and the result title "The bias is added
    before the rotation" of `architecture-deltas.md`,
  - `has run` in the doc of `sync` (`benches/layer.rs:163`). The completion of
    each queued operation is the point.
  - the em dashes of the link titles in `README.md`. They are the titles of
    the talks and the repositories.

### Docs that disagree with the code

The sweep finds no doc that disagrees with the code. It checks the facts of
the changed paragraphs that name code:

- §8 and §9.4 of `architecture-deltas.md` against
  `Mamba3ScanOptions::optimal_chunk_len` (`src/mamba3/scan_options.rs:110`).
  At `N = 128` and `P = 64`, the rule gives 96, 64 and 32 for the ranks 1, 2
  and 4, as §8 says.
- ADR-0007 against `src/mamba3/plant_frame.rs` and its tests in
  `src/mamba3/plant_frame/tests.rs`.
- §10.4 of `mimo-as-batch.md` against the key scale `scale_bnlh` of
  `src/mamba3/single_ssd/ssd/serial.rs`. The Spec review makes this check.

The other changed paragraphs that name code keep the words that name it.
Examples are §7 of `mimo-as-batch.md`, §10.2 of
`trapezoid-as-integration.md` and the header of `benches/layer.rs`. So the
sweep does not check them again.

### Open for the user

1. Two sentences of `architecture-deltas.md` tell that the note changed the
   library:
   - `:85`: "§8 is the one section that changed the library instead of
     describing it",
   - `:530`: "This is the one place where the note changed behaviour instead
     of describing it".

   The linter does not flag them, so they stay. The rule on changelog prose
   of `CLAUDE.md` asks to cut such prose. But the spec of the sweep keeps
   each fact. Each sentence holds a fact: the chunk schedule of §8 is a
   choice of this crate. The user decides.
2. The body of this ticket asks for a record of each changed copy in
   burn-stack ticket 05. That ticket is closed, and each ticket changes one
   repo. The two changed rules take the words of `../burn-stack/CLAUDE.md`,
   so burn-stack needs no change and gets no new ticket (see
   [The copies](#the-copies)). Item 1 of "Open for the user" of burn-stack
   ticket 05 stays open. The user can close it, or ask for a record in
   burn-stack.

### For the linter

Three suggestions. Each one agrees with STE.

1. The linter ends a sentence at the period of an author initial (`J.`,
   `G. J.`, `J.-P.`) and of the abbreviation `Prop.`. Five reference items get
   a `long-paragraph` finding. A probe of the reference of Baccelli et al.
   alone gives the same finding. A suggestion: do not end a sentence at a
   period after one capital letter, or after a known abbreviation.
2. The phrasal-verb rule misses six verbs. A probe of their sentences gives
   no phrasal-verb finding. The verbs are at these locations:
   - `mimo-as-batch.md:37` and `trapezoid-as-integration.md:36` at
     `f3ad86c7`. This change replaces that verb.
   - `rotation-as-optimization.md:470` (the lead-in of the two verified
     facts), in a paragraph with no finding,
   - `docs/source-terms.md:3` (the first word of the doc), in a paragraph
     with no finding,
   - `mimo-as-batch.md:58` (the rank and the initialisation),
   - `architecture-deltas.md:448` (the balance of the FLOPs),
   - `README.md:50` (the cumulative rotation and the recurrence).

   A suggestion: add the first three verbs to the hard list. The other three
   are terms of math, so they go to the technical list.
3. Suggestion 1 of ticket 07 also fixes `README.md:272`: count a rotation
   only when both words are verbs.

### The review

The two agents of `/mattpocock-skills:code-review` read `git diff f3ad86c7`,
one after the other. Their reports are `tmp/ste-sweep-08/review-standards.md`
and `tmp/ste-sweep-08/review-spec.md`. This change holds the fixes of their
findings:

- Three ragged wraps join (`architecture-deltas.md` §8 and its preface, and
  item 2 of the abstract of `rotation-as-optimization.md`).
- The record gives these facts correctly:
  - each pathway sentence, with its own determiner,
  - six misses of the linter,
  - six passives that become "The script verifies …",
  - the current line numbers.
- The split of the gauge item of §8(iii) in `rotation-as-optimization.md`
  keeps the condition of all three results. A list after "With `β` on that
  side" gives them.
- "So the generator shares no subexpression …" replaces a garden path.
- `mimo-as-batch.md` §9 names `A(x_i)` as the subject, in place of an `It`
  that can read as the script.
- `architecture-deltas.md` §3 keeps "That is the whole mechanism behind the
  stability claim of the source". The quote moves to the next sentence.
- The two abstracts say "is not studied further, as orthogonal to …". That
  keeps both parts of the old verb.
- Item 2 of the abstract of `rotation-as-optimization.md` adds no actor:
  "That is descent in the real part of the state and *ascent* in the
  imaginary part".
- The lead-in at `rotation-as-optimization.md:470` keeps its words. Its
  paragraph has no finding, so the change is out of scope.
- The reset list of `README.md` says "The rungs, in order", the order of
  `examples/README.md`. No item has the word "then".
- The kept finding of `benches/layer.rs:212` gives the reason "Precision".
- The list of checks of code adds §10.4 of `mimo-as-batch.md`.

These findings stay:

- `check` replaces `confirm` in §9.4 of `architecture-deltas.md`. The bench
  measures the direction, and the file uses `check` for that action.
- The two sentences of `architecture-deltas.md` that tell a change of the
  library are open for the user (item 1 of
  [Open for the user](#open-for-the-user)).
- No record goes to burn-stack (item 2 of
  [Open for the user](#open-for-the-user)).

### The check

`./scripts/check.sh` passes on the final tree (exit 0). The log has no
warning and no failed test.

### Work files

`tmp/ste-sweep-08/` holds these files:

- the file list of the scope, and the lint reports before and after,
- the lint script, the width script and the probe of `9a8958ec`,
- the two probe files of the linter,
- the review briefs and the review reports,
- the logs of `./scripts/check.sh` and the commit message.
