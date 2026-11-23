# 06: Sweep: the reset examples

Status: ready-for-agent

Blocked by: 01, `.scratch/ste-lint-misses/` 01 and 02 (the linter)

Spec: `../spec.md`. Follow its method, its text that does not change, its kept
findings and its rules for messages and copies.

**Context:** ticket 01 extends the prose step of `scripts/check.sh`. Tickets
02–08 are the batches of burn-mamba, and they do not block each other. Ticket
09 sweeps `.scratch/` last.

**What to build:** the prose of the reset examples follows the `asd-ste100`
skill.

- Scope: `examples/reset/`, outside each `tmp/` directory, 44 files.
- At the start: 157 hard findings (6 in messages) and 124 advisory findings.
- `examples/reset/quintic/HANDOFF.md` has 75 hard findings. It is the handoff
  of an open thread. Keep each fact, each number and each open question.
- The example tests can match a message. Keep each matched substring.

**References:**

- Spec: "The method of a batch", "Text that does not change", "Kept
  findings", "Messages and copies" and "A doc that disagrees with the code".
- The terms: `GLOSSARY.md`, `../burn-stack/GLOSSARY.md`, ADR-0008 for a
  quoted source word, and `docs/source-terms.md` before a source word.
- `examples/README.md`: the examples and their knobs.
- Prior art: `../burn-stack/.scratch/glossary-followups/issues/05-prose-sweep-examples.md`.

- [ ] `## Comments` gives the count of hard findings before and after.
- [ ] The linter finds zero hard findings in the scope, except the kept
      findings that `## Comments` lists.
- [ ] Each substring of a changed message that a test of either repo matches
      stays.
- [ ] `## Comments` lists each doc that disagrees with the code, with its
      evidence. The user decides each one.
- [ ] `./scripts/check.sh` passes. Run it in the background.

## Comments
