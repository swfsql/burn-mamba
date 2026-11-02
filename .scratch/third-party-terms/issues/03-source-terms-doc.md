# 03: The map from Mamba source terms to our terms

Status: ready-for-agent

Blocked by: None (can start immediately)

**What to build:** `docs/source-terms.md`. An agent that reads a Mamba paper or a
reference implementation looks up a source word there. It finds our term, the
difference, and the wrong port that the word can cause.

The input is `research.md`. Load the `asd-ste100` and
`mattpocock-skills:writing-for-agents` skills before you write. The user approved
reads of `../papers/` and `../py/` for this ticket.

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

- `docs/agents/domain.md`: tell an agent to read this doc before it reads or
  ports from a source.
- "Where to look" in `CLAUDE.md`: add the doc.
- The File Map line of `docs/` in `CLAUDE.md`: add `source-terms.md`.

## Acceptance criteria

- [ ] `docs/source-terms.md` has the four sections.
- [ ] Its first row covers the official `Block` and `mixer`.
- [ ] Each word of research part 3 that a Mamba source uses has a row.
- [ ] Each cell of "Our code follows" holds `?`.
- [ ] Each source citation has the form `file:line`, and no citation of our own
      code has a line number.
- [ ] `docs/agents/domain.md` and `CLAUDE.md` have the links, and the File Map has
      the file.
- [ ] `scripts/ste-lint.py` of the `asd-ste100` skill gives no hard violation on
      the doc.
