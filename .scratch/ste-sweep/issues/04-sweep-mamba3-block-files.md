# 04: Sweep: the Mamba-3 block files

Status: ready-for-agent

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

- [ ] `## Comments` gives the count of hard findings before and after.
- [ ] The linter finds zero hard findings in the scope, except the kept
      findings that `## Comments` lists.
- [ ] Each substring of a changed message that a test of either repo matches
      stays.
- [ ] `## Comments` lists each doc that disagrees with the code, with its
      evidence. The user decides each one.
- [ ] `./scripts/check.sh` passes. Run it in the background.

## Comments
