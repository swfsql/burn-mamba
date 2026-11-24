# 02: Sweep: the Mamba-3 pathways

Status: resolved

Blocked by: 01, `.scratch/ste-lint-misses/` 01 and 02 (the linter)

Spec: `../spec.md`. Follow its method, its text that does not change, its kept
findings and its rules for messages and copies.

**Context:** ticket 01 extends the prose step of `scripts/check.sh`. Tickets
02–08 are the batches of burn-mamba, and they do not block each other. Ticket
09 sweeps `.scratch/` last.

**What to build:** the prose of the single-SSD and double-SSD pathways follows
the `asd-ste100` skill.

- Scope: `src/mamba3/single_ssd/` and `src/mamba3/double_ssd/`, 29 files.
- At the start: 92 hard findings (2 in messages) and 121 advisory findings.
- The module headers hold the math of each SSD path. Keep each formula, each
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
29 tracked `.rs` files of the scope. The scope has no `.md` file.

| Findings | Before | After |
|---|---|---|
| Hard | 92 | 4 |
| Hard, in messages | 2 | 0 |
| Advisory | 128 | 54 |

The ticket gives 121 advisory findings at the start. At `1b94af91`, the
updated linter finds 128. Its count of hard findings agrees with the ticket.

[The kept findings](#the-kept-findings) gives the 4 hard findings that stay.
Each advisory finding that stays is one of these:

- an em dash in a heading, a section banner or a list of terms,
- a passive that tells a state (`double_ssd/ssd/ssd_path/mod.rs:16` and
  `single_ssd/single_ssd/mod.rs:217`).

None of these em dashes joins two sentences.

### The check

`./scripts/check.sh` passes on the final tree (exit 0). The log has no
warning and no failed test.

### The kept findings

Each one is a false positive.

1. `src/mamba3/double_ssd/double_ssd/mod.rs:409` and `:549`, long sentence
   (26 and 32 words). The linter reads the banner line of a section and the
   first sentence under it as one sentence. An example of a banner line is
   `── The tail of the plant frame, on a read axis of length 1 ──`. The
   first sentence under each banner has 20 words or fewer.
2. `src/mamba3/single_ssd/ssd/diag.rs:88`, long sentence (28 words). The two
   comment lines are shape maps, for example
   `c_bnlmhr [b, n, l, m, h, r] -> c_bnlhmr [b, n, l, h, m, r]`. They are not
   prose, but the linter reads them as one sentence.
3. `src/mamba3/double_ssd/ssd/serial.rs:263`, synonym rotation of `start` and
   `launch`. In this file, `launches` is a noun for the kernels that the K4
   loop sends to the GPU (line 245). `start` names the first position of a
   segment, a call or a chunk. **Start state** is a term of `GLOSSARY.md`. So
   the two words do not name the same action.

### The messages

The two hard findings in messages are one message in two files. A search of
the tests of both repos finds no match for a part of either string. The
search covers `should_panic(expected = …)`, the `contains(…)` checks and the
string compares.

| Location | New string |
|---|---|
| `src/mamba3/double_ssd/ssd/serial.rs:52` | `double_ssd_serial does not support init_state_hpr yet. Use double_ssd_minimal instead.` |
| `src/mamba3/single_ssd/ssd/serial.rs:66` | `single_ssd_serial does not support init_state_hpr yet. Use single_ssd_minimal instead.` |

The old strings are at `1b94af91`. Each new string keeps "yet" and the advice
to use the `Minimal` path. The double-SSD string also names the functions of
its pathway (see [The decisions of the user](#the-decisions-of-the-user)).

### The copies

These paragraphs have copies in the scope. Each set changes together:

- The doc of `cfg_real1d_scalar` and the comment on the `Real1D` accumulator,
  in the tests of both pathways.
- The doc of the tolerance of the `state_rank = 1` rung, in the tests of both
  pathways.
- The doc of the read-axis test, the comment on the full `C` and the comment
  on the read positions of the test, in both `ssd_path/tests.rs`.
- The header of both `ssd/mod.rs`, and the doc of `c_bntmhr` in both
  `ssd_path/mod.rs`.
- The comment on the strict-upper mask, in `single_ssd/ssd/serial.rs` and
  `single_ssd/ssd/serial_recalculated/serial_recalculated.rs`.

The spec changes all copies of a paragraph in the same ticket. So these files
outside the scope also change. Each one is in the scope of ticket 05.

| File | Copies | Hard before | Hard after |
|---|---|---|---|
| `src/mamba1/mamba1/tests.rs` | the docs of `Heads`, `run_with_grads`, `check_grads_match`, `param_input` and `run_step_matches_forward`, and the comment of the guard on the random initial state | 9 | 4 |
| `src/mamba2/mamba2/tests.rs` | the same five docs and the same comment | 9 | 4 |
| `src/mamba2/ssd/ssd_path/tests.rs` | the comments on `init_state_hpr`, on the tolerance of the gradients and on `nchunks=1` | 5 | 3 |
| `src/standard_ssd/combined_backward.rs` | the comment on the two axes of the chunk | 0 | 0 |

Three of these are near copies:

- the doc of `Heads` in Mamba-1 and Mamba-2 (it names "the final cache"),
- the Mamba-1 doc of `run_step_matches_forward` (it says "equals"),
- the comment on the tolerance of the gradients in Mamba-2 (it says "the
  summations").

The rule of the spec names verbatim copies. The sweep also changes these near
copies, so that each set keeps one form. Each copy keeps its own words. So
ticket 05 starts with 12 fewer hard findings than its count. Burn-stack has no
copy of a changed paragraph.

### Docs that disagree with the code

The sweep changes only the form of these three docs. The names and the claim
that disagree with the code stay. The user decides each one.

1. `src/mamba3/single_ssd/single_ssd/tests.rs:166`: the doc of `RunGrads` says
   that the struct mirrors "the equivalent struct in `mamba3::tests`". The
   crate has no module `mamba3::tests`. `src/mamba3/mamba3.rs` has no
   `mod tests`, and `src/mamba3/` has no `tests.rs`. The other `RunGrads`
   structs are in `src/mamba3/double_ssd/double_ssd/tests.rs:24`,
   `src/mamba1/mamba1/tests.rs:15` and `src/mamba2/mamba2/tests.rs:15`.
2. `src/mamba3/double_ssd/ssd/serial.rs:52`: the message names `ssd_serial`
   and `ssd_minimal`. These are the functions of Mamba-2
   (`src/mamba2/ssd/serial.rs:43`, `src/mamba2/ssd/minimal.rs:95`). The
   function that asserts is `double_ssd_serial`
   (`src/mamba3/double_ssd/ssd/serial.rs:44`). The `Minimal` function of this
   pathway is `double_ssd_minimal` (`src/mamba3/double_ssd/ssd/minimal.rs:41`).
   The single-SSD message names `single_ssd_serial` and `single_ssd_minimal`,
   and these functions exist.
3. `src/mamba3/double_ssd/ssd/ssd_path/tests.rs:109`: the comment names
   `ssd_serial` and `ssd_serial_recalculated`. The functions of this pathway
   are `double_ssd_serial` and `double_ssd_serial_recalculated`
   (`src/mamba3/double_ssd/ssd/serial_recalculated/serial_recalculated.rs:32`).
   The copy at `src/mamba2/ssd/ssd_path/tests.rs:115` names the functions of
   Mamba-2, so it agrees with its code.

### The decisions of the user

1. The doc of `RunGrads` names `double_ssd::double_ssd::tests`. The history
   supports this choice. At `18a3feb3`, the tests of `src/mamba3/mamba3.rs`,
   with their `struct RunGrads`, move to `src/mamba3/double_ssd/double_ssd.rs`.
   Burn-stack has no `RunGrads`.
2. The message names `double_ssd_serial` and `double_ssd_minimal`. No test of
   either repo matches a part of it.
3. The comment names `double_ssd_serial` and `double_ssd_serial_recalculated`.

### A term

`GLOSSARY.md` gives "weight" as an _Avoid_ word for a **Mass**. Two rewritten
sentences keep the word of the old text:

- `src/mamba3/single_ssd/token_band.rs:104`: the γ-diagonal of the kernel
  "already fixed its weight". This weight is the mass `γ` of the read.
- `src/mamba3/single_ssd/ssd/mod.rs:7`: the kernel applies "the trapezoid
  weights". These are `γ` and `scale`, and `scale` is a sum of masses.

The header of `token_band.rs` and other files of `src/mamba3/` also use
"weight" for `γ` and for the sum `Δ̃`. So a change of the word is a change of
the term in many files. The sweep keeps it, and a separate change can follow
the rename rule of `CLAUDE.md`.

### For the linter

The kept findings suggest three changes to the linter. Each one agrees with
STE.

1. A banner line of a section has no end mark, so the linter joins it with the
   next sentence. A suggestion: end a sentence at a comment line that starts
   or ends with a run of box-drawing characters (`─`, `═`). A heading is not a
   part of a sentence.
2. A comment line that holds only a shape map or a formula has no end mark
   too. A suggestion: end a sentence at the line break of a comment line when
   most of its tokens are identifiers, brackets and operators such as `->`. A
   formula is not a sentence.
3. The pair `start` and `launch` of the synonym rule also matches the noun
   `launches` of a GPU kernel. A suggestion: count the pair only when both
   words are verbs. STE gives each word one part of speech.
