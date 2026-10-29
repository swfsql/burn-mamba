# 01: Docs redo: a short CLAUDE.md, a short File Map, no files.md

Status: ready-for-agent

Blocked by: None (can start immediately)

**What to build:** An agent that starts a session reads a short CLAUDE.md that
points to where each kind of knowledge lives. Today, CLAUDE.md (657 lines) and
files.md (399 lines) repeat much of the glossary, the ADRs and the module
headers. After this ticket, files.md does not exist, and its non-obvious
decisions are in the module headers.

## Decisions from the grill

- Load the `mattpocock-skills:writing-for-agents` and `asd-ste100` skills
  before you write. Read `GLOSSARY.md`, `../burn-stack/GLOSSARY.md` and
  `docs/adr/` first, and use their terms in all new prose.
- CLAUDE.md keeps only what every session needs:
  - a summary of the project in about 3 lines,
  - the build and test commands, the feature flags and the test-device rules,
  - the rules for edits, waiting, memory and commit messages,
  - a "where to look" map. It lists the two glossaries, `docs/adr/`, the
    module headers, `info/`, `docs/agents/` and the external references under
    `../`. The math is in the headers of `mamba2.rs` and `mamba3.rs`.
  - the Notation table,
  - a short File Map.
- The File Map has 40 lines or fewer. It has one line for each directory, and
  one line for each file whose name does not tell what it holds. It has no
  `tests.rs` lines.
- Notation: the `s` entry becomes "write-axis length (= tokens·u)". Keep the
  current identifier names everywhere: tickets 02 and 03 rename them.
- The Architecture and Key Design Decisions sections go to their sources.
  Terms go to the glossary, decisions with a trade-off to an ADR, and
  mechanisms and math to the module header of their file. Delete a paragraph only after you
  find its content in one of these places. If you find it in none, move it to
  the module header. If it holds a new term or a new decision, also add a note
  under `## Comments` for the user. Do not edit the glossary or the ADRs.
- files.md: move each non-obvious decision into the module header of its file.
  Then move files.md out of the repo with `mv files.md /tmp/`. Use no `rm` and
  no git command. Fix every link to files.md (README.md, CLAUDE.md, a handoff in
  `examples/reset/`).
- The block-agnostic rule follows ADR-0002. burn-stack code is block-agnostic
  by default, and its docs can cite a family. A rare exception is named after
  the mechanism. burn-deltanet is a second consumer.
- The doc-maintenance rules:
  - Keep the docs small, and never use them as a changelog (keep this rule).
  - A change that adds, deletes or renames a file updates its File Map line
    in the same change.
  - A new term goes to `GLOSSARY.md` (through `/domain-modeling`). A new
    decision with a trade-off goes to an ADR.
  - Delete the `tmp.md` rule and the "do not read big files at the end" rule.
- Custom Commands: `mv` and `cp` are allowed to move and copy files. Content
  edits still use the Edit/Write tools. Keep the other rules.
- Do not edit README.md or examples/README.md, except for the files.md links.
  Ticket 04 sweeps their prose.

## Acceptance criteria

- [ ] CLAUDE.md has about 200 lines or fewer, and its File Map has 40 lines or
      fewer.
- [ ] files.md is not in the repo. Each of its non-obvious decisions is in a
      module header, or is listed under `## Comments`.
- [ ] Each deleted CLAUDE.md paragraph has its content in a glossary, an ADR or
      a module header, or is listed under `## Comments`.
- [ ] `rg 'files\.md'` finds no link outside `.scratch/`. No doc links to a
      CLAUDE.md anchor that is gone.
- [ ] `cargo check` passes, and `cargo doc --all --no-deps` has no warnings.
- [ ] The asd-ste100 lint (`ste-lint.py`) passes on CLAUDE.md.

## Comments
