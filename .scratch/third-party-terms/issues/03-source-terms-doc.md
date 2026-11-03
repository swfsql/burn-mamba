# 03: The map from Mamba source terms to our terms

Status: resolved

Blocked by: None (can start immediately)

**What to build:** `docs/source-terms.md`. An agent that reads a Mamba paper or a
reference implementation looks up a source word there. It finds our term, the
difference, and the wrong port that the word can cause.

The input is `research.md`. It is a best-effort draft, so each of its rows is a
lead and not a fact. Ticket 04 checks each row against its citations. Load the
`asd-ste100` and `mattpocock-skills:writing-for-agents` skills before you write.
The user approved reads of `../papers/` and `../py/` for this ticket.

**Expected length.** One session, under 150k tokens. `research.md` is about 30k
tokens. Do not check rows against the sources here, because that work is large
and has its own ticket. Read a cited line only if you cannot understand a row
without it.

## Decisions from the grill

**Our terms stay.** This doc renames nothing (ADR-0008). `Block` stays. The first
row of the doc says that the official class `Block` is our layer, and that
`mixer` is our block.

**Placement test.** A word goes in `GLOSSARY.md` if an agent that reads only our
code and docs needs it. A word goes in this doc if only an agent that reads the
sources needs it.

**Sources of this doc.** The Mamba papers and blogs, the official code and
mamba3-minimal: research parts 2.1, 2.2 and 2.3. The H-Net and Multi-Gate
Residuals rows go to the doc of burn-stack, whichever crate owns our term.

**Sections.**

1. **False friends and splits.** This section is the focus. Write one row for
   each source word, in the order of risk of research part 3. Each row gives the
   source word, what it means in each source, our term and the wrong port. The
   input is the rows that research marks `false friend` or `split`.
2. **Source terms with no term of ours.** Research part 4, for example `λ_t`,
   which is also `trap`, `lam` and "mixing factor". Also the names of the
   `renamed` rows that a reader cannot decode alone: `k_state`, `v_state`,
   `zxBCdtAtrap`, `cu_seqlens`. Drop a `renamed` row with a name that explains
   itself: `InferenceCache`, `generate`.
3. **Where the sources disagree with each other.** See the seed list below. The
   columns are the concept, each source, and "Our code follows".
4. **Letters.** Research part 5.3, and the spelling in each source of the cells
   that research 5.1 marks as incomplete.

**Citations.** A source citation has the form `file:line`. The doc header gives
the path aliases and pins the commits: official code `a14b1df`, mamba3-minimal
`15ba998`. Our own code is cited by item name, with no line number.

**"Our code follows".** This ticket leaves each cell as `?`. Ticket 04 fills the
cells in a new session. Do not check our code for these cells here.

## Seed list for section 3

Research part 2.2 and part 5 give the citations. Add each other disagreement that
you find as you write.

- **Cumulative angle.** The names `angle_dt_state`, `Angles_Cumsum` and
  `cum_angle`. The official kernel accumulates `+Δθ`, mamba3-minimal accumulates
  `−Δθ`, and the paper writes `R^⊤`.
- **Bound on the angle.** The official code has `tanh(angle)·π·dt`. mamba3-minimal
  and the paper have no bound.
- **Rotated share of the state.** The official `rope_fraction` is 0.5 or 1.0.
  mamba3-minimal and the paper always rotate the whole state.
- **Pair layout.** The official `rotate_pairwise`, "Interleaved rotary" and
  "Halved rotary".
- **Tap cache.** The official code keeps `k_state` and `v_state`. mamba3-minimal
  keeps the product `prev_Bx`.
- **SSD calls.** The official code has one call in its combined kernel.
  mamba3-minimal has its "two-SSD decomposition".
- **SISO.** The official code has separate kernels. mamba3-minimal has a separate
  branch and a separate `ssd` function.
- **Chunk length.** `Q` in Mamba-2, `C` in Mamba-3, `chunk_size`, `block_len`.
- **`B` and the step size.** `B` in Mamba-2 holds the step size. `B` in Mamba-3
  does not.
- **`A` and `B`.** They are discrete in Mamba-2 and continuous in Mamba-3.
- **The split `λ`.** `trap` is pre-sigmoid in `forward` and post-sigmoid in
  `_preprocess`. The other names are `Trap`, `lam` and "mixing factor".
- **Norm and residual.** The official `Block` runs "Add -> LN -> Mixer", and no
  code wraps `Mamba3` in a `Block`. The mamba3-minimal model loop runs both adds.

## Links

- The File Map line of `docs/` in `CLAUDE.md`: add `source-terms.md`.
- Do not add other links. Ticket 04 adds them after its check, so no agent reads
  an unchecked doc.

## Acceptance criteria

- [x] `docs/source-terms.md` has the four sections.
- [x] Its first row covers the official `Block` and `mixer`.
- [x] Each word of research part 3 that a Mamba source uses has a row.
- [x] Each cell of "Our code follows" holds `?`.
- [x] Each source citation has the form `file:line`, and no citation of our own
      code has a line number.
- [x] The File Map has the file. `docs/agents/domain.md` and "Where to look" do
      not name it yet.
- [x] `scripts/ste-lint.py` of the `asd-ste100` skill gives no hard violation on
      the doc.

## Comments

The doc follows the structure of the reviewed `docs/source-terms.md` of
burn-stack. These choices go past the text of the ticket:

- **The glossaries are newer than `research.md`.** Research part 4 marks some rows
  as "no term" that now have a term: `Δ_t` is Step size, `α_t` is Decay, `GatedMLP`
  is Feed-forward. These rows moved to "Source names that a reader cannot decode".
  Research also gives "none" for some rows whose item exists now, such as
  `Mamba3::b_norm`. The doc names these items. `hidden_states` maps to Residual
  stream, as the burn-stack doc does for its "hidden state".
- **Rows of section 3 past the seed list:** "The coefficient `β`", "What
  `ssm_state` holds", "Feed-forward part", "Form with no rotation" and "Count of
  heads".
- **The seed "The split `λ`" is the row "The share `λ`".** The doc defines "split"
  as a kind of source word, and **Mass** defines a mass as a share.
- **Section 3 has one column "Papers".** The Mamba-1 and Mamba-2 papers apply to
  only a few rows, so a cell gives the name of the paper.
- **Spellings past the ticket:** the rows `i`, `h`, `m` and `v`. Research 5.1 marks
  them as true for one source only, not as incomplete. Their cells of the Notation
  table have the same gap.
- **Facts in two sections.** A word can have a row in sections 1 or 2 and a concept
  in section 3, because the ticket asks for both. So some facts repeat, for example
  the two forms of `trap`. A lookup of a word must show the fact that causes its
  wrong port.
- **Rows of `research.md` with no row here:** the `renamed` rows with a name that
  explains itself. They are `InferenceCache`, `generate`, `MambaLMHeadModel`,
  `Mamba3LMHeadModel`, "gradient step" and "Mamba architecture". "Process prompt"
  also explains itself, but it has a row because its tail differs from **Prefill**.
- **Citations that differ from `research.md`.** `TIL/mamba3_mimo.py:255` is `DT`,
  so the `Trap` citation is `:256`, and the row for `Δ_t` uses `:255`. These
  citations are new, from lines that I read: `MM/mamba3.py:76`, `:178`, `:373`,
  `:676`, `:677`, `:678`, and `SSM/modules/mamba2.py:85`.

This ticket did not check the rows against their citations. A script checked only
that each cited file and line exists. Ticket 04 does the check of the content.
