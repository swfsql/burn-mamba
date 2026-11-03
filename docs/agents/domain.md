# Domain Docs

How the engineering skills read the domain documentation of this repo when they explore the code.

## Before exploring, read these

- **`GLOSSARY.md`** at the repo root.
- **`../burn-stack/GLOSSARY.md`**: it defines the terms of the containers (block, layer, cache, row, segment, …).
- **`docs/adr/`**: read the ADRs that touch the area of your work.
- The **Notation** table of `CLAUDE.md` and the module headers of `src/mamba2/mamba2.rs` and `src/mamba3/mamba3.rs`. They define the math and the shape suffixes.

If `GLOSSARY.md` or `docs/adr/` does not exist, **continue silently**. Do not report that it is missing. Do not propose to create it in advance. The `/domain-modeling` skill (through `/grill-with-docs` and `/improve-codebase-architecture`) creates these files when a term or a decision is resolved.

## Before you read or port from a source

Read **`docs/source-terms.md`** before you read a source or port from it. It maps the words of the Mamba papers, the Mamba blog, the official code and mamba3-minimal to our terms. It also gives the wrong port that each word can cause, and the reading that our code follows where the sources disagree. The words of the H-Net and Multi-Gate Residuals papers are in `../burn-stack/docs/source-terms.md`.

## File structure

This repo is single-context. There is no `GLOSSARY-MAP.md`.

```
/
├── GLOSSARY.md
├── docs/adr/
│   └── 0001-<decision-slug>.md
├── docs/source-terms.md
└── src/
```

## Use the glossary's vocabulary

When your output names a domain concept (in an issue title, a refactor proposal, a hypothesis, a test name), use the term that `GLOSSARY.md` defines. Do not change to a synonym that the glossary tells you to avoid.

If the glossary does not have the concept, that is a signal. Either you invent language that the project does not use (think again), or the glossary has a real gap (record it for `/domain-modeling`).

## Flag ADR conflicts

If your output contradicts an ADR, say so explicitly. Do not silently override it:

> _Contradicts ADR-0001 (no custom kernels), but worth reopening because…_
