# 05: Sweep: the other `src/` modules

Status: ready-for-agent

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

- [ ] `## Comments` gives the count of hard findings before and after.
- [ ] The linter finds zero hard findings in the scope, except the kept
      findings that `## Comments` lists.
- [ ] Each substring of a changed message that a test of either repo matches
      stays.
- [ ] `## Comments` lists each doc that disagrees with the code, with its
      evidence. The user decides each one.
- [ ] `./scripts/check.sh` passes. Run it in the background.

## Comments
