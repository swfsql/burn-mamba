# 04: Check the map of Mamba source terms

Status: ready-for-agent

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

- [ ] Each citation of the doc has a report from an agent.
- [ ] No cell of "Our code follows" holds `?`.
- [ ] Each corrected row cites the lines that prove it.
- [ ] `docs/agents/domain.md` and `CLAUDE.md` have the links.
- [ ] `## Comments` at the end of this file gives the number of rows checked and
      corrected, and each open question for the user.
- [ ] `git diff` shows no change under `src/`.
