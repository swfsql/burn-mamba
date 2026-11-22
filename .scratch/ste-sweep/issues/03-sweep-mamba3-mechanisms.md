# 03: Sweep: the Mamba-3 mechanisms

Status: ready-for-agent

Blocked by: 01

Spec: `../spec.md`. Follow its method, its text that does not change, its kept
findings and its rules for messages and copies.

**Context:** ticket 01 extends the prose step of `scripts/check.sh`. Tickets
02–08 are the batches of burn-mamba, and they do not block each other. Ticket
09 sweeps `.scratch/` last.

**What to build:** the prose of the rotation, the positive systems, the
quaternion scan and MambaProduct follows the `asd-ste100` skill.

- Scope: `src/mamba3/rotation/`, `src/mamba3/positive/`,
  `src/mamba3/quat_scan/` and `src/mamba3/product/`, 14 files.
- At the start: 101 hard findings (1 in a message) and 120 advisory findings.
  `src/mamba3/rotation/tests.rs` alone has 39 hard findings.
- Tests match two panic messages of the rotation. Keep each matched
  substring.
- A known false positive: at `src/mamba3/positive/tests.rs:825`, the linter
  reads "keeps … out" across a gap of spaces.
- The module headers hold the math of each mechanism. Keep each formula, each
  symbol and each shape suffix.

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
