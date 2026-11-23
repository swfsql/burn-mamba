# 08: Sweep: the notes and the root docs

Status: ready-for-agent

Blocked by: 01, `.scratch/ste-lint-misses/` 01 and 02 (the linter)

Spec: `../spec.md`. Follow its method, its text that does not change, its kept
findings and its rules for messages and copies.

**Context:** ticket 01 extends the prose step of `scripts/check.sh`. Tickets
02–08 are the batches of burn-mamba, and they do not block each other. Ticket
09 sweeps `.scratch/` last. burn-stack ticket 05 sweeps the root docs of
burn-stack.

**What to build:** the prose of the derivation notes, the docs, the benches
and the root files follows the `asd-ste100` skill.

- Scope: `info/`, `docs/`, `benches/`, `CLAUDE.md`, `README.md`,
  `GLOSSARY.md` and `build.rs`, 27 files. `bench.md` and `kernels.md` stay
  (spec, "Out of Scope").
- At the start: 76 hard findings and 60 advisory findings. No message has a
  finding.
- Each note of `info/` has a float64 check in `scripts/`, with check numbers
  that match its sections. Keep each section, each heading and each number.
- An ADR keeps each decision, each option and each reason.
- `GLOSSARY.md` keeps its terms, their meanings and its _Avoid_ entries.
- Each rule of `CLAUDE.md` keeps its exact meaning. Read
  `/mattpocock-skills:writing-for-agents` before you change `CLAUDE.md`.
- A known kept finding: `CLAUDE.md:65` quotes a banned phrase of the history
  rule.
- `CLAUDE.md` and `../burn-stack/CLAUDE.md` share some rules word for word.
  Record each changed copy under `## Comments` of burn-stack ticket 05.

**References:**

- Spec: "The method of a batch", "Text that does not change", "Kept
  findings", "Messages and copies" and "A doc that disagrees with the code".
- The terms: `GLOSSARY.md`, `../burn-stack/GLOSSARY.md`, ADR-0008 for a
  quoted source word, and `docs/source-terms.md` before a source word.
- Prior art: `../burn-stack/.scratch/glossary-followups/issues/02-prose-sweep.md`.

- [ ] `## Comments` gives the count of hard findings before and after.
- [ ] The linter finds zero hard findings in the scope, except the kept
      findings that `## Comments` lists.
- [ ] `## Comments` lists each doc that disagrees with the code, with its
      evidence. The user decides each one.
- [ ] `./scripts/check.sh` passes. Run it in the background.

## Comments

### From burn-stack ticket 05

- Burn-stack ticket 05 gives the **Waiting** rule of `../burn-stack/CLAUDE.md`
  the words of the rule of `CLAUDE.md`, with one difference. The burn-stack
  rule says "one part of a chain", and this rule says "one step of a chain".
  **Step** is a term of `../burn-stack/GLOSSARY.md`: the recurrent call. Its
  _Avoid_ entry is "step (alone, for one recurrence step or for one training
  step)". So the copy of this repo can also say "part". Then the two copies
  are equal.
- Burn-stack ticket 05 closes before this ticket runs. For each shared rule
  that this ticket changes, write a new burn-stack ticket in
  `../burn-stack/.scratch/ste-sweep/issues/`. That ticket applies the change
  to `../burn-stack/CLAUDE.md`. The record of burn-stack ticket 05 gives the
  reason.
