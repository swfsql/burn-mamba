# 04: Check the map of Mamba source terms

Status: resolved

Blocked by: 03

**What to build:** Each row of `docs/source-terms.md` matches its citations. Each
row of section 3 says which reading our code follows. Then the links tell agents
to read the doc.

## Decisions from the grill

**How to run it.** Run this ticket in a new session, with several agents. The user
approved reads of `../papers/` and `../py/` for this ticket.

**Scope.**

- Each row of each section, against each of its citations. `research.md` is a
  best-effort draft, so no row counts as checked.
- Each row of section 3. Fill its "Our code follows" cell.

**Out of scope.** The doc of burn-stack. It has its own check ticket.

## Work pipeline

One agent is reliable up to about 150k tokens. The doc has about 100 rows, which
is too much for one agent. So the lead splits the work before any agent starts.

1. **Split.** The lead reads only `docs/source-terms.md` and numbers its rows. It
   puts each citation into a batch for its source group:
   - The Mamba-1 and Mamba-2 papers, and the blogs.
   - The Mamba-3 paper.
   - The official modules and models.
   - The official kernels.
   - mamba3-minimal.
   - Section 3, because each of its rows spans several sources and our code.
2. **Agents.** One agent takes one batch of at most 25 rows. A source group can
   need more than one batch. Expect 6 to 8 agents.
3. **Reads.** An agent reads the cited lines, with enough context to decide. It
   does not read a whole file unless a row needs that.
4. **Limit.** Each agent stays under 150k tokens. At the limit, it stops and
   reports the rows that it did not finish. The lead gives them to a new agent.
5. **Report.** For each citation, an agent gives `confirmed`, `wrong` or
   `not found`, and the lines that prove it. For a wrong row, it also gives the
   corrected text. A batch agent does not edit the doc.
6. **Merge.** The lead edits the doc from the reports. The lead does not read the
   sources again.

## Section 3

1. Read the cited source lines. Do not trust the text of the row.
2. Read our code for the same behaviour. If reading does not settle the row, run
   a small program in `/tmp`.
3. Do not change code in this ticket. If our code follows no source, or a
   difference looks like a bug, write that in the cell and report it to the user.
4. Delete the sentence of the section intro that defines `?`. For each "No
   citation" cell, read the source. Give a citation if the source has a statement
   on the concept. The sentence that defines "No citation" stays only if a cell
   still holds it.

**The first row to check.** Part 1 of `research.md` defines the path aliases of
this paragraph. The official kernel accumulates `+Δθ` (`TRI/angle_dt.py:104`).
mamba3-minimal accumulates `−Δθ` (`MM/mamba3.py:367`). The official rotary step
has a `conjugate` flag, and `SSM/modules/mamba3.py:339` passes `False`. Find the
sign of our code. Say if the difference changes any output, or only the sign of
the learned angles.

## Links

Add these after the merge:

- `docs/agents/domain.md`: tell an agent to read this doc before it reads or
  ports from a source.
- "Where to look" in `CLAUDE.md`: add the doc.

## Acceptance criteria

- [x] Each citation of the doc has a report from an agent.
- [x] No cell of "Our code follows" holds `?`.
- [x] Each corrected row cites the lines that prove it.
- [x] `docs/agents/domain.md` and `CLAUDE.md` have the links.
- [x] `## Comments` at the end of this file gives the number of rows checked and
      corrected, and each open question for the user.
- [x] `git diff` shows no change under `src/`.

## Comments

### Counts

- **Rows checked:** 107 of 107. Row L41 has no citation of its own. It points to
  the letter tables.
- **Citations checked:** 577. 555 `confirmed`, 21 `wrong`, 1 `not found`. A script
  compared the verdict lines with the list of citations, and no citation has no
  verdict.
- **Rows corrected:** 61, and 17 of them are section 3. A row can change with no
  `wrong` verdict, because an agent also tested the uncited claims and the negative
  claims of a row.
- **Citations of the doc after the merge:** 560 unique. 134 are new and 7 are gone.
  Each new citation comes from a report. A script checked that each cited file and
  line exists.
- **Agents:** 12 for the batches, not 6 to 8. The cap of 25 rows forces this, because
  a row counts once for each source group that it cites. One more agent answered one
  question from the review. A harness rule refuses a report file from a subagent, so
  each report came as the final message of its agent. The user removed the reports
  and the numeric checks from the repo after the merge.
- **Agents over 150k tokens:** `s3-rotation` (197k), `s3-paths` (191k) and
  `s3-coefficients` (168k). The user accepted up to 200k.

### The first row: the sign of the angle

Our code follows the official code. It accumulates `+Δ·ϱ·π·tanh(ϑ)` and turns `B`
and `C` by `R(+θ)`, in forward and in step. The paper and mamba3-minimal turn by
`R(−θ)`.

With the same weights, the outputs differ. A checkpoint cannot load unchanged across
the two groups. A sign flip of the angle channels of the in-projection makes the
outputs equal, so the difference is only the sign of the learned angles. The flip
needs these conditions:

1. The bound on the angle is odd and is equal on both sides. `π·tanh` and the
   identity are odd. The bound of mamba3-minimal differs from the official bound, so
   the flip alone does not make them equal.
2. The flip also covers the bias of the angle channels.
3. `B` and `C` turn in one direction.
4. Forward and step turn in one direction.
5. The group is abelian. So this flip applies to `Complex2D`, not to `Quaternion4D`
   or `Rotor4D`.

Trained models also match only if the start law and the optimizer are symmetric in
the sign. SGD with momentum stays mirrored in a numeric check. Adam and Muon had no
run. A float64 model of the conventions gave these conditions.

`handmade_block_solves_every_family` of `reset-rotor` pins the direction in the real
block. Its hand-built decoder reads `(cos φ, −sin φ)`. It gives 100% on each family
with our sign, and 33.8% to 40.1% with a flipped `ϑ`, in a transient mutation.

### Questions for the user, with the answers

1. **A sign in the header of `src/mamba3/mamba3.rs`, section 2.** The header gave
   `ρₜ = R(+Δₜ·ϱ·π·tanh(ϑₜ))` with `hₜ = αₜ ρₜ hₜ₋₁ + Δₜ Bₜ xₜᵀ`, and then said that
   `B̃ₜ = Rₜ Bₜ` absorbs this form. These two statements did not match. Absorption of
   that `ρₜ` gives `B̃ₜ = Rₜᵀ Bₜ`, the form of the paper. Our code applies `R(+θ)`, as
   the official kernels do, and that equals a direct form with `ρₜ = R(−…)`.
   **Answer:** the code stays, and the docs change. A follow-up change corrects the
   header, and it adds a note that the sources turn in opposite directions. That
   change touches only doc comments under `src/`. The check found no bug.
2. **The path of `utils/`.** The checkout of the official code held
   `mamba_ssm/utils/` at `mamba_ssm/ops/utils/`, as an uncommitted change.
   **Answer:** the user restored the checkout, and the doc cites `mamba_ssm/utils/`.
3. **The size of the doc.** The doc grew from 45.7 KB to about 64 KB, and some
   "Differences" text of section 3 restates the module headers.
   **Answer:** no trim.

### Reports for the user, with no decision needed

No difference looks like a bug. Each case below follows no source, and each one is a
selectable option or a recorded decision.

- `Mamba3Config::rotation_range` has default `2.0`, and the official value is `1.0`.
- `Mamba3Config::rotation_fraction` has default `1.0`, and the official value is
  `0.5`.
- `optimal_chunk_len` has no source. The source defaults are fixed values.
- The tap slot axis, `helpers::tail_decay` and `helpers::interior_gap_decay` serve a
  lag above 1, and no source has this lag.
- `Gain::Kalman` sets `TrapezoidCoeffs::da`, so `da` is not `Δ·A` under this member.
- `Residuals::MultiGate` has no outer skip. It comes from another paper.
- SISO is `mimo_rank` = 1 of one implementation (ADR-0004). So an official block with
  `is_mimo=True` and `mimo_rank=1` has no equal here.
- `Complex2D` turns by `R(+θ)`, and `Quaternion4D` and `Rotor4D` turn by the inverse.
  A comparison of weights across these kinds must know this.

### Findings about a source, with no doc change

- **`SSM/modules/mamba3.py:147`.** The one-token branch can fail. It passes a 3-axis
  `u` to `step`, and `step` takes `(batch, d_model)`. Mamba-2 removes the axis at
  `SSM/modules/mamba2.py:281`. An agent read this and did not run it, so row L52
  stays as it is.
- **`M2/systems.tex:116`.** The paper sets `A_t = 0` on the last token of a sequence.
  In its recurrence, that value does not cut the state that leaves that token. Row
  L100 now says this. An agent read this from the equation and did not run it.

### Optional findings

An agent found no fault in these rows. The lead still added each finding, because it
gives a word or a proof that a port needs:

- L79: `MIMO_V`, `MIMO_Z`, `MIMO_O`, and the `φ` of the appendix.
- L85 and L87: the proof lines of the count and of the `Δ` projection.
- L160 and L184: the spellings of the Cute kernel. The row checks of these two rows
  did not cover this word, so its proof is only the cited line.
- L185 and L188: the spellings of the Tilelang kernels.

The other optional findings stayed out, so that the doc stays small.

### Review

The review ran on the two axes before the commit. The fixes of its findings are in
the doc:

- **Standards:** 6 source words in our own voice, and 4 judgement calls on words.
- **Spec:** L109, L110, L136 and L140. The MIMO forward was not in the first check of
  L40 and L134, so one agent read it. It writes an accumulator, in the same form as
  SISO, and it takes no input state.
