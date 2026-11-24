# 03: Sweep: the Mamba-3 mechanisms

Status: resolved

Blocked by: 01, `.scratch/ste-lint-misses/` 01 and 02 (the linter)

Spec: `../spec.md`. Follow its method, its text that does not change, its kept
findings and its rules for messages and copies.

**Context:** ticket 01 extends the prose step of `scripts/check.sh`. Tickets
02–08 are the batches of burn-mamba, and they do not block each other. Ticket
09 sweeps `.scratch/` last.

**What to build:** the prose of the rotation, the positive systems, the
quaternion scan and MambaProduct follows the `asd-ste100` skill.

- Scope: `src/mamba3/rotation/`, `src/mamba3/positive/`,
  `src/mamba3/quat_scan/` and `src/mamba3/product/`, 14 files.
- At the start: 101 hard findings (1 in a message) and 120 advisory findings.
  `src/mamba3/rotation/tests.rs` alone has 39 hard findings.
- Tests match two panic messages of the rotation. Keep each matched
  substring.
- A known false positive: at `src/mamba3/positive/tests.rs:825`, the linter
  reads "keeps … out" across a gap of spaces.
- The module headers hold the math of each mechanism. Keep each formula, each
  symbol and each shape suffix.

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
14 tracked `.rs` files of the scope. The scope has no `.md` file.

| Findings | Before | After |
|---|---|---|
| Hard | 101 | 3 |
| Hard, in messages | 1 | 0 |
| Advisory | 127 | 40 |

The ticket gives 120 advisory findings at the start. At `a33f7cc2`, the
updated linter finds 127. Its count of hard findings agrees with the ticket.

[The kept findings](#the-kept-findings) gives the 3 hard findings that stay.
Each advisory finding that stays is one of these:

- an em dash in a heading, a section banner, or a list of terms or shapes,
- a passive that tells a state, for example "is unchanged" or "is pinned",
- a passive with no relevant actor, for example "The group is given
  directly" in `src/mamba3/product/mod.rs:96`.

None of these em dashes joins two sentences.

### The check

`./scripts/check.sh` passes on the final tree (exit 0). The log has no
warning and no failed test.

### The kept findings

Each one is a false positive.

1. `src/mamba3/positive/scan.rs:284`, synonym rotation of `start` and
   `launch`. In this file, `launch` is only a noun: "a one-launch in-block
   scan" (line 247) and "mostly launches, not bytes" (line 256). These are the
   kernels that a round of the scan sends to the GPU. The verb `start` names
   the first position of a segment or a product. Ticket 02 keeps the same pair
   at `src/mamba3/double_ssd/ssd/serial.rs:263`.
2. `src/mamba3/positive/tests.rs:840`, phrasal verb `keeps … out`. This is the
   known false positive of the spec and the ticket. The linter reads the code
   span `0·∞ = NaN` between the verb and "out" as a gap of spaces. In this
   sentence, "out of the forward" is a phrase of place, not the particle of a
   phrasal verb.
3. `src/mamba3/rotation/mod.rs:490`, phrasal verb `keeps … away`. The linter
   reads the code span `norm` between the verb and "away" as a gap of spaces,
   as in item 2. In this sentence, "away from 0" is a phrase of place. It also
   tells a margin: the next paragraph of the comment gives "a norm floor of
   `√div_eps`". A rewrite to "above 0" loses the margin.

### The messages

Four messages change. The first one has the hard finding. The other three
have an em dash or a passive that the method changes. A search of the tests of
both repos finds no match for a part of any of them. The search covers
`should_panic(expected = …)`, the `contains(…)` checks and the string
compares.

| Location | New string |
|---|---|
| `src/mamba3/rotation/mod.rs:852` | `a quaternion kind always turns at least one block. A real transition is RotationKind::Real1D` |
| `src/mamba3/rotation/tests.rs:1332` | `{label}: the scan's branch moved. Pick the lengths again so that both branches stay covered` |
| `src/mamba3/rotation/tests.rs:1414` | `the continuation must take the blocked branch. Otherwise, the carry-in that this test exercises is not the blocked one. Lengthen the continuation` |
| `src/mamba3/rotation/tests.rs:2012` | `{kind:?}: the block ignores rotation_range: {d}` |

The old strings are at `a33f7cc2`. The two panic messages that tests match do
not change:

- "expected Quaternion" (`src/mamba3/rotation/mod.rs:378`), which
  `src/mamba3/rotation/tests.rs:1560` matches,
- "multiple of 4", which `src/mamba3/rotation/tests.rs:2077` matches. The
  test sets `state_rank = 4` and `rotation_fraction = 0.5`, so it panics at
  `src/mamba3/mamba3.rs:1307`, outside the scope. The messages at
  `src/mamba3/rotation/mod.rs:685` and `:727` also hold the substring, and
  they do not change.

`../burn-stack/src/modules/misc/prefix_sum.rs:104` has the substring "must
take the blocked branch" in its own assert message. That assert is not a test
match, and the new string keeps the substring.

### The copies

No changed paragraph has a verbatim copy outside the scope. A search for the
changed phrases finds the banner "Split-prefill continuity" in
`src/mamba1/mamba1/tests.rs`, `src/mamba2/mamba2/tests.rs` and
`src/mamba3/double_ssd/double_ssd/tests.rs`. The text under each banner is
different from the changed doc. Burn-stack has no copy.

`docs/adr/0005-mambaproduct-sequence-fold.md:9` has a near copy of a changed
sentence of `src/mamba3/product/mod.rs`: "the decay is not a control of its
own: it comes from the step size". Ticket 02 also changes near copies, so
that each set keeps one form. The ADR gets the same split, and its other
words stay. This file is in the scope of ticket 08. Its hard findings go from
1 to 0.

### History phrases

`CLAUDE.md` deletes changelog prose. The sweep changes these phrases, and the
old text is at `a33f7cc2`. Each fact about the current code stays. Two deleted
parts tell only history: "it replaced" and "before this one, was all of
them". A past tense about an alternative becomes a "would" form.

| Location | Old phrase | New phrase |
|---|---|---|
| The banner of section 3b, `src/mamba3/rotation/tests.rs` | "the readable token-by-token reference it replaced" | "the readable token-by-token reference" |
| The doc of `complex_angle_scan_forward_step_parity` | "`Tensor::cumsum` once, the blocked scan now, whatever replaces it" | "`Tensor::cumsum`, the blocked scan, or a later replacement" |
| The same doc | "made the scan's cost matter in the first place" | "makes the cost of the scan matter" |
| The doc of `complex_angle_scan_split_prefill_parity` | "which, before this one, was all of them: the existing split-prefill tests" | "The other split-prefill tests" |
| The doc of `saturated_generator_keeps_a_live_axis_gradient` | "Squashing each component separately froze both at once" | "would freeze both at once" |
| The doc of `a_normalised_quaternion_has_an_unbiased_norm` | "a division by `√Σq²` gave a mean of `+2e-4`" | "would give a mean" |
| The assert of `rotation_dim`, `src/mamba3/rotation/mod.rs` | "the cache was sized for exactly that many" | "the cache holds exactly that many" |

### Docs that disagree with the code

The sweep finds no doc that disagrees with the code.

Many rewrites change a passive to an active verb. Each one names an actor that
the old text did not name. Examples are "the block forms `αₜ`", "the scan
detaches the norm" and "The test duplicates Φⱼ". The code next to each
sentence gives the actor. Where the actor is not clear, the passive stays, for
example "The factors are stacked along one block axis". Two actors need a
look at a test or a match arm:

- The doc of `quaternion_rotation_field_survives_record_roundtrip` says that
  the value of the field comes from the module that loads the record. The
  test loads the record into `cfg.init(&device)` and reads
  `RotationKind::Quaternion4D` back (`src/mamba3/rotation/tests.rs:1048`).
- The doc of `rotate_bc_forward` says that the function returns `B`/`C` and
  the accumulator as they are for `Real1D`. The match arm returns
  `(b_bsmhr, c_btmhr, prev.expect_real())`.

### A term

`GLOSSARY.md` gives "weight" as an _Avoid_ word for a **Mass**. The rewritten
sentence at `src/mamba3/positive/kalman.rs:14` keeps the word of the old text:
`Λ` is "the total weight of every sample that the plant wrote". Ticket 02
records the same use in other files, and its reason applies here. A change of
the word is a change of the term in many files, and the test name
`the_precision_is_the_plants_weight_on_ones` uses it too.

### For the linter

The kept findings suggest two changes to the linter. Each one agrees with
STE.

1. Suggestion 3 of ticket 02 also clears `scan.rs:284`: count the pair
   `start` and `launch` only when both words are verbs.
2. The findings at `tests.rs:840` and `rotation/mod.rs:490` show the code
   span as a gap of spaces (`keeps             out`, `keeps        away`).
   So the phrasal-verb rule reads the verb and the particle as two words next
   to each other. A suggestion: read a code span as one noun. Then the rule
   reads "keeps X out" as a verb, a noun and a particle. It flags this form
   only for the six pairs of its list.
