# 09: Sweep: the tickets

Status: ready-for-agent

Blocked by: 01, 02, 03, 04, 05, 06, 07, 08, `.scratch/ste-lint-misses/` 01 and 02 (the linter)

Spec: `../spec.md`. Follow its method, its text that does not change and its
kept findings.

**Context:** ticket 01 extends the prose step of `scripts/check.sh` to
`.scratch/`. Tickets 02–08 sweep the rest of burn-mamba and write their
records under `## Comments` here. This ticket goes last, so it also reads those
records.

**What to build:** the prose of the tickets and the specs follows the
`asd-ste100` skill.

- Scope: all of `.scratch/`, resolved tickets and the tickets of this sweep
  included, 55 files before this sweep.
- At the start: 46 hard findings and 109 advisory findings. The tickets of
  this sweep add their own findings, if any.
- `## Comments` and `## Answer` are records. A history phrase there stays only
  as a "Ticket record" (spec, "Kept findings").
- Keep the state of each checkbox, each `Status:` line and each `Blocked by:`
  line. The title of a ticket is a heading.

**References:**

- The spec sections "The method of a batch", "Text that does not change" and
  "Kept findings".
- The terms: `GLOSSARY.md`, `../burn-stack/GLOSSARY.md`, ADR-0008 for a
  quoted source word, and `docs/source-terms.md` before a source word.
- `docs/agents/issue-tracker.md`: the shape of a ticket.
- Prior art: `../burn-stack/.scratch/glossary-followups/issues/02-prose-sweep.md`.

- [ ] `## Comments` gives the count of hard findings before and after.
- [ ] The linter finds zero hard findings in the scope, except the kept
      findings that `## Comments` lists.
- [ ] `## Comments` lists the false positives of the whole sweep in one place,
      with the tickets of 02–08 as the source.
- [ ] `./scripts/check.sh` passes. Run it in the background.

## Comments
