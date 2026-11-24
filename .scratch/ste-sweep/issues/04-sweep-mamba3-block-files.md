# 04: Sweep: the Mamba-3 block files

Status: resolved

Blocked by: 01, `.scratch/ste-lint-misses/` 01 and 02 (the linter)

Spec: `../spec.md`. Follow its method, its text that does not change, its kept
findings and its rules for messages and copies.

**Context:** ticket 01 extends the prose step of `scripts/check.sh`. Tickets
02–08 are the batches of burn-mamba, and they do not block each other. Ticket
09 sweeps `.scratch/` last.

**What to build:** the prose of the Mamba-3 block files follows the
`asd-ste100` skill.

- Scope: each file of `src/mamba3/` outside the directories of tickets 02 and
  03, 12 files. Examples are `mamba3.rs`, the helpers, the tap pattern, the
  plant frame, the scan options and the cache.
- At the start: 57 hard findings and 71 advisory findings. No message has a
  finding.
- The header of `mamba3.rs` is the math reference of Mamba-3 (`CLAUDE.md`).
  Keep each formula, each symbol, each section and each shape suffix.

**References:**

- Spec: "The method of a batch", "Text that does not change", "Kept
  findings", "Messages and copies" and "A doc that disagrees with the code".
- The terms: `GLOSSARY.md`, `../burn-stack/GLOSSARY.md`, ADR-0008 for a
  quoted source word, and `docs/source-terms.md` before a source word.
- Prior art: `../burn-stack/.scratch/glossary-followups/issues/02-prose-sweep.md`.

- [x] `## Comments` gives the count of hard findings before and after.
- [x] The linter finds zero hard findings in the scope, except the kept
      findings that `## Comments` lists.
- [x] Each substring of a changed message that a test of either repo matches
      stays.
- [x] `## Comments` lists each doc that disagrees with the code, with its
      evidence. The user decides each one.
- [x] `./scripts/check.sh` passes. Run it in the background.

## Comments

### The counts

The linter runs with `--enable history --parts docs,comments,messages` on the
12 tracked `.rs` files of the scope. The scope has no `.md` file.

| Findings | Before | After |
|---|---|---|
| Hard | 57 | 0 |
| Hard, in messages | 0 | 0 |
| Advisory | 75 | 33 |

The ticket gives 71 advisory findings at the start. At `03dc6761`, the
updated linter finds 75. Its count of hard findings agrees with the ticket.

No hard finding stays. Each advisory finding that stays is one of these:

- an em dash in a heading, a section banner, a table or a list of terms,
- the em dash of the doc "State rank — the size of the `r` axis of the state"
  (`src/mamba3/mamba3.rs:383` and `:835`). `src/mamba1/mamba1.rs:114` and
  `src/mamba2/mamba2.rs:323` have the same doc, so all four copies keep it.
- a passive that tells a state, for example "is closed", "is unchanged" or
  "is enabled",
- a passive with no relevant actor: "is reduced"
  (`src/mamba3/helpers.rs:89`), "is dropped" (`:476`) and "is transported"
  (`src/mamba3/tap_pattern.rs`, 3 times). The tap pattern uses "transport"
  as its word for the decay across a gap.
- a false positive at `src/mamba3/mamba3.rs:879`. In "its write is `u`
  staggered outer products", the verb "is" has no passive participle.

None of these em dashes joins two sentences.

Two state passives get the value of the code in place of the participle. Each
new form agrees with the type of the field:

- "`out_norm` is set" becomes "is `Some`" (`src/mamba3/plant_frame.rs:369`,
  an `Option<RmsNormGated>`),
- "`siso_specialization_decode` is enabled" becomes "is `true`"
  (`src/mamba3/mamba3.rs:717`, a `bool`).

### The check

`./scripts/check.sh` passes on the final tree (exit 0). The log has no
warning and no failed test.

### The messages

One message changes. Its em dash joins two sentences, so the method splits
it. A search of the tests of both repos finds no match for a part of it. The
search covers `should_panic(expected = …)`, the `contains(…)` checks and the
string compares.

| Location | New string |
|---|---|
| `src/mamba3/mamba3.rs:1295` | `{:?} rotates nothing at state_rank = {} and rotation_fraction = {}. Use RotationKind::Real1D` |

The old string is at `03dc6761`.

### The copies

These paragraphs have copies in the scope. Each set changes together:

- The doc of `Mamba3::tap_pattern_spec` and the doc of `TapPatternSpec`
  (`src/mamba3/tap_pattern.rs:345`) are near copies. Both get the same
  split.
- The two docs of a `#[module(skip)]` field in `src/mamba3/mamba3.rs` (`:417`
  and `:456`) both get "excludes it from the record". The same file has
  "keeps "content" channels out of the turn" (`:988`), which the linter does
  not flag. It gets "excludes … from" too, so the file has one verb for this
  action.
- The four shape docs with "initialised to" in `src/mamba3/mamba3.rs` get
  one form: "Shape: …. The init sets it to …".

`src/mamba2/mamba2.rs:683`, the doc of `Mamba2::forward_packed`, is a near
copy of the doc of `Mamba3::forward_packed`. It gets the same paragraph
break, after "up to the next start". Its words do not change. This file is in
the scope of ticket 05. Its hard findings go from 7 to 6, so ticket 05 starts
with 1 hard finding fewer than its count. Burn-stack has no copy of a changed
paragraph.

### History phrases

`CLAUDE.md` says to delete changelog prose. The sweep changes these phrases,
and the old text is at `03dc6761`. Each fact about the current code stays.

| Location | Old phrase | New phrase |
|---|---|---|
| The doc of `reference_prefix_sum`, `src/mamba3/helpers/tests.rs` | "if the op `prefix_sum` replaced is ever dropped" | "if Burn ever drops, deprecates or changes that op" |
| The doc of `prefix_sum_matches_cumsum`, the same file | "the op [`prefix_sum`] replaced" | "the op that [`prefix_sum`] replaces" |

### Docs that disagree with the code

The sweep changes neither the claim of these docs nor the code. The user
decides each one.

1. `src/mamba3/helpers/tests.rs:81`: the section banner calls `prefix_sum`
   "the log-depth inclusive scan". The header of `src/mamba3/helpers.rs`
   calls it "the blocked inclusive scan" (line 18). A suggestion for the
   banner: "the blocked inclusive scan".

   The evidence, in `src/mamba3/helpers.rs`:

   - Without a period, the function makes one `cumsum` in each block and
     one `cumsum` over the block totals (lines 218 and 225).
   - Its doc says that "the whole is a few ops for any `len`" (line 119).
   - The same doc gives the Hillis–Steele doubling of `quat_scan` as the
     other method (line 141).
   - On the circle, the function calls itself on the block totals (line
     244). So only in that case does the depth grow with `len`, as
     `log len / log block`.
2. `src/mamba3/helpers.rs:349`: the doc of `interior_gap_decay` names "the
   `v` slots of the cache". The cache holds the `x` of each slot
   (`x_buhp`, `src/mamba3/cache.rs:112` and `:118`). The doc of
   `tail_decay` also says "The cache holds the `x` of the tap FIFO"
   (`src/mamba3/helpers.rs:384`). The sweep changes the form of the
   sentence, and the name `v` stays. A suggestion: "the `x` slots of the
   cache".

### A term

`GLOSSARY.md` gives "weight" as an _Avoid_ word for a **Mass**.

- `src/mamba3/tap_pattern.rs:22`, in a paragraph with a hard finding: "Their
  *weight* is not a second decision" becomes "Their *mass*". The next
  sentence names the mass `Δₚ` that the two scalars split.
- `src/mamba3/helpers.rs:604`, the comment of a passive finding: "the left
  endpoint is unweighted" becomes "has no mass".
- Two paragraphs with no finding keep the word: "the quadrature weights"
  (`src/mamba3/tap_pattern.rs:45`), and "the weight that the plant wrote"
  with "its weighted mean" (`:160`). Ticket 03 keeps the second use at
  `src/mamba3/positive/kalman.rs:14`.

Tickets 02 and 03 keep "weight" in their rewrites. This ticket changes the
word in the two places above. The review notes that "the quadrature weights"
(`:45`) and the *mass* of `:22` now name the same quantity in one header. The
user decides between the two changes and the old words.

### For the linter

Two suggestions. Each one agrees with STE.

1. The sentence rule ends a sentence at the abbreviation `Prop.`. STE
   permits only approved abbreviations. A suggestion: report the
   abbreviation itself, and do not count its period as the end of a
   sentence. The evidence:

   - At `03dc6761`, the paragraph at `src/mamba3/mamba3.rs:51` has 5
     sentences, and the linter counts 6. The words `(paper Prop.
     *Complex-to-Real SSM Equivalence*)` count as two sentences.
   - A test file with 6 short sentences and `(paper Prop. *Name*)` gives a
     long-paragraph finding. The same file with "Proposition" gives none.
2. The passive rule reads "is `u` staggered" (`src/mamba3/mamba3.rs:879`)
   as "is     staggered", with a gap of spaces for the code span. Suggestion
   2 of ticket 03 also covers this case: read a code span as one noun.
