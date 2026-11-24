# 05: Sweep: the other `src/` modules

Status: resolved

Blocked by: 01, `.scratch/ste-lint-misses/` 01 and 02 (the linter)

Spec: `../spec.md`. Follow its method, its text that does not change, its kept
findings and its rules for messages and copies.

**Context:** ticket 01 extends the prose step of `scripts/check.sh`. Tickets
02–08 are the batches of burn-mamba, and they do not block each other. Ticket
09 sweeps `.scratch/` last.

**What to build:** the prose of Mamba-1, Mamba-2, the standard SSD scan, the
family choice and the shared modules follows the `asd-ste100` skill.

- Scope: `src/mamba1/`, `src/mamba2/`, `src/standard_ssd/`, `src/family/`,
  and the files at the root of `src/` with their test directories, 45 files.
- At the start: 50 hard findings (2 in messages) and 98 advisory findings.
- The header of `mamba2.rs` is the math reference of Mamba-2 (`CLAUDE.md`).
  Keep each formula, each symbol, each section and each shape suffix.
- Tests match panic messages of the family choice, packing, the row end, the
  Mamba-2 block and its SSD path. Keep each matched substring.

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
45 tracked `.rs` files of the scope. The scope has no `.md` file.

| Findings | Before | After |
|---|---|---|
| Hard | 37 | 0 |
| Hard, in messages | 2 | 0 |
| Advisory | 88 | 63 |

The ticket gives 50 hard findings and 98 advisory findings at the start.
Ticket 04 fixes 1 hard finding of `src/mamba2/mamba2.rs`. At `245ca737`,
the updated linter finds 37 hard findings (2 in messages) and 88 advisory
findings.

No hard finding stays. Each advisory finding that stays is one of these:

- an em dash in a heading, a section banner or a list of terms,
- the em dash of the doc "State rank — the size of the `r` axis of the state"
  (`src/mamba1/mamba1.rs:114` and `src/mamba2/mamba2.rs:323`). Its near copy
  at `src/mamba2/mamba2.rs:196` starts with "`state_rank` —". Ticket 04 keeps
  the same doc in `src/mamba3/mamba3.rs`.
- a passive that tells a state: "`M` is fixed at the type level"
  (`src/family/mod.rs:4`), "if Δ is untied" (`src/mamba2/mamba2.rs:717`
  and `:1088`), and "the high-level `Tensor` is pinned to"
  (`src/mamba2/ssd/serial_recalculated/combined_backward/tests.rs:9`). The
  crate uses "pinned to `Dispatch`" for this state, for example at
  `src/standard_ssd/kernels.rs:16`.
- a false positive: "run-time state" (`src/family/block.rs:42`) and "the work
  is batched matrix multiplications" (`src/mamba2/mamba2.rs:631`). In each
  one, the participle is part of an adjective.

None of these em dashes joins two sentences. Two other em dashes change,
although neither one joins two sentences:

- An em dash in parentheses becomes a colon: "(fully depthwise: each channel
  is independent)" (`src/mamba2/mamba2.rs:133`).
- An em dash adds a second idea to a list of positions
  (`src/padding/tests.rs:312`). The second idea becomes its own sentence:
  "Batch-wide, `End` comes after the padding."

### The check

`./scripts/check.sh` passes on the final tree (exit 0). The log has no
warning and no failed test.

### The review

The two agents of `/mattpocock-skills:code-review` read `git diff 245ca737`.
The fixes of their findings are in this change:

- The list of `src/padding.rs:13` stays a full list: "These fields are …",
  as in `src/row_end.rs:4`.
- The paragraph of `src/standard_ssd/mod.rs:72` splits at its second topic.
- "is pinned to" stays (`combined_backward/tests.rs:9`), and two "so that"
  clauses keep their purpose (`src/mamba2/ssd/ssd_path/tests.rs`).
- A comment names `forward_packed` in place of "This step"
  (`src/mamba2/mamba2.rs:726`).
- The doc of `run_split_matches_full` keeps "that thread" as in its Mamba-3
  copy. The sentence has 23 words.
- "no column of the initial window" replaces "no sample" (see Terms 5).

### The messages

Two messages change. A search of the tests of both repos covers
`should_panic(expected = …)`, the `contains(…)` checks and the string
compares.

| Location | New string | Test match |
|---|---|---|
| `src/row_end.rs:132` | `a row-end read in a call with resets needs the origin field. The last segment of a row with a reset starts from it` | `src/row_end/tests.rs:170` matches "needs the origin field". The substring stays. |
| `src/mamba2/ssd/serial_recalculated/combined_backward/tests.rs:156` | `the oracle identity of d_da_local fails: max abs diff = {diff}` | none |

The old strings are at `245ca737`.

### The copies

These paragraphs have copies in the scope or in a swept file. Each set has
one form:

- The doc of `build_init_cache`, the doc of `run_split_matches_full` and the
  `seq_len` comment of `run_step_matches_forward` are near copies in
  `src/mamba1/mamba1/tests.rs` and `src/mamba2/mamba2/tests.rs`. Each pair
  gets the same split. The two docs take the form of their Mamba-3 copies in
  `src/mamba3/double_ssd/double_ssd/tests.rs`, which ticket 02 swept. Each
  copy keeps the words of its family, for example the fields of its cache.
- The list of the dials of `padded_mamba3_is_each_row_alone`
  (`src/padding/tests.rs:232`) takes the list form of
  `packed_mamba3_streams_are_the_batch_of_their_sequences`
  (`src/packing/tests.rs:554`).
- In `src/mamba2/mamba2.rs`, two comments "drop" the oldest column of the
  conv window (`:809` and `:1117`). At `245ca737`, the doc of `step` gives a
  different verb for the same action (line 1061). It gets "drops" too, so
  the file has one verb for this action. The comment of the zero-pad columns
  of Step 5 also gets "Drop" (`:958`).

Burn-stack has no copy of a changed paragraph. The padding contract of
`../burn-stack/src/modules/mod.rs:120` already has the split form of
`src/padding.rs:4`.

### History phrases

`CLAUDE.md` says to delete changelog prose. At `245ca737`, the comment of
the zero-pad columns (`src/mamba2/mamba2.rs:956`) has a passive in the past
tense. Its new text is "Drop the zero-pad columns that Step 5 adds." The
fact about the current code stays.

### Two readings

At `245ca737`, two comments have two readings. In each one, the code gives
the reading that the new text states.

1. `src/standard_ssd/kernels.rs:439`: the doc of
   `k4_ssd_state_passing_backward` has a phrasal verb (line 440). In one
   reading, `decay` is a common factor of the `(p, r)` sum. In the other
   reading, `d_decay` is the result of the sum. The code gives the second
   reading: `d_decay_bhn` is the `(p, r)` sum of
   `d_intra_chunk_state_bnhpr · chunk_input_state_bnhpr` (line 503). The new
   doc says "So `d_decay` is the `(p, r)` sum of `d_sᵢ₊₁·sᵢ`".
2. `src/mamba2/mamba2.rs:1197`: the comment of the B expansion in `step`
   says "as the SSD forward path does" (line 1194). "SSD path" is a term of
   `GLOSSARY.md`, but no SSD path expands B. `forward_packed` expands B and C
   with `gqa_expand_to_heads` (line 924), before it calls the SSD path at
   Step 6. The new comment says "as `forward` does before the SSD scan".

### Docs that disagree with the code

The sweep changes neither the claim of this doc nor the code. The user
decides it.

1. `src/padding.rs:8`: "The family zeroes the decay and the writes of that
   micro-step". `GLOSSARY.md` defines the decay as the factor
   `α = exp(Δ·A)`. At a padded position, this factor is 1, not 0:

   - Mamba-1 and Mamba-2 set `Δ = 0` (`src/mamba1/mamba1.rs:490` and
     `src/mamba2/mamba2.rs:886`), so `Ā = exp(0·A) = 1`.
   - Mamba-3 sets `da = 0` and `alpha = da.exp()`. Its doc says "no decay
     (`da = 0`, `α = 1`)" (`src/mamba3/helpers.rs:472`).
   - The next sentence of the same doc says "`Δ = 0` gives `Ā = 1`".

   A suggestion: "The family zeroes `Δ·A` and the writes of that micro-step,
   so the decay is 1".

### Terms

1. `src/mamba2/mamba2.rs:848` and `:849`, in a paragraph with a hard finding:
   `GLOSSARY.md` gives "state projection" as an _Avoid_ word for a **Write**.
   "(state input proj)" becomes "(the `B` of the write)", and "(state output
   proj)" becomes "(the `C` of the readout)".
2. `src/padding/tests.rs:234` and `src/packing/tests.rs:556` say "every tap
   pattern family". In `../burn-stack/GLOSSARY.md`, a **Family** is Mamba-1,
   Mamba-2 or Mamba-3. The words stay. A suggestion: "every kind of tap
   pattern".
3. `src/lib.rs:29` lists "class tokens" in what burn-stack holds. Burn-stack
   also holds the class latents, and `../burn-stack/GLOSSARY.md` names both
   **Class markers**. The words stay. A suggestion: "class markers".
4. `src/mamba1/mamba1/tests.rs:224` says "`seq_len >= d_conv`". `d_conv` is
   the Python name. The code and the Mamba-2 copy of the comment use
   `conv_kernel`. The words stay. A suggestion: `seq_len >= conv_kernel`.
5. `GLOSSARY.md` defines a **Sample** as the product `B xᵀ` of one write
   position. Some docs also use "samples" for the entries of a conv window:
   the **Row end** entry of `GLOSSARY.md`, the header of `src/row_end.rs`
   and `src/padding.rs:13`. The words of these docs stay. The new comment of
   `seq_len` in `src/mamba1/mamba1/tests.rs:225` and
   `src/mamba2/mamba2/tests.rs:207` says "no column of the initial window",
   as `src/mamba2/mamba2.rs:1117` does. The user decides whether the
   glossary gives the second sense or the docs change.

### The decisions of the user

1. The doc of `src/padding.rs:8` gets the suggestion. It also says that the
   decay is a no-op. The new text: "The family zeroes `Δ·A` and the writes of
   that micro-step where the discretisation forms them. So the decay is 1,
   and it is a no-op."
2. Terms 2 to 5 keep their words.
3. Term 1 keeps its change.
4. The two suggestions for the linter stay suggestions for the skill.

### For the linter

Two suggestions. Each one agrees with STE.

1. The sentence rule reads a section banner as the start of the next
   sentence. A banner is a heading, not a sentence. A suggestion: end a
   sentence at a comment line that box-drawing rules (`── … ──`) frame. The
   evidence: a probe file with the banner `// ── Step 3: Split xbc into the
   three parts x, B and C ──…` and one comment line of 16 words gives one
   long sentence of 27 words.
2. The passive rule reads a participle that is part of an adjective:
   "run-time state" (`src/family/block.rs:42`) and "is batched matrix
   multiplications" (`src/mamba2/mamba2.rs:631`). A suggestion: do not read
   a word in a hyphenated compound as a participle. Also, do not read a
   participle that a noun follows as a passive.
