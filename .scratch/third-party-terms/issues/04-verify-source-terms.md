# 04: Check the map of Mamba source terms

Status: ready-for-agent

Blocked by: 03

**What to build:** Each row of section 3 of `docs/source-terms.md` says which
reading our code follows. Each row that states a behaviour matches the sources
and our code.

## Decisions from the grill

**How to run it.** Run this ticket in a new session. The user permits several
agents for this ticket. The user approved reads of `../papers/` and `../py/` for
this ticket.

**Scope.**

- Each row of section 3. Fill its "Our code follows" cell.
- Each row in any section that states how a source or our code behaves. An
  example: `trap` is pre-sigmoid in `forward` and post-sigmoid in `_preprocess`.

**Out of scope.**

- A row that only says that a word means another concept.
- The doc of burn-stack. It cites only papers.

## Method

1. Read the cited source lines. Do not trust the text of the row or of
   `research.md`.
2. Read our code for the same behaviour. If reading does not settle the row, run
   a small program in `/tmp`.
3. Correct each wrong row.
4. Do not change code in this ticket. If our code follows no source, or a
   difference looks like a bug, write that in the cell and report it to the user.

**The first row to check.** Part 1 of `research.md` defines the path aliases of
this paragraph. The official kernel accumulates `+Δθ`
(`TRI/angle_dt.py:104`). mamba3-minimal accumulates `−Δθ` (`MM/mamba3.py:367`).
The official rotary step has a `conjugate` flag, and `SSM/modules/mamba3.py:339`
passes `False`. Find the sign of our code. Say if the difference changes any
output, or only the sign of the learned angles.

## Acceptance criteria

- [ ] No cell of "Our code follows" holds `?`.
- [ ] Each corrected row cites the lines that prove it.
- [ ] `## Comments` at the end of this file gives the number of rows checked and
      corrected, and each open question for the user.
- [ ] `git diff` shows no change under `src/`.
