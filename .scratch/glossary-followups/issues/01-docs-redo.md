# 01: Docs redo: a short CLAUDE.md, a short File Map, no files.md

Status: resolved

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
  - A name and its docs change in the same change. A rename of an identifier
    updates every doc that names it or its concept: comments, messages, the
    READMEs, `info/`, `GLOSSARY.md` and the ADRs. A change of a term in the
    docs also renames each identifier that uses the term in that sense.
  - Delete the `tmp.md` rule and the "do not read big files at the end" rule.
- Custom Commands: `mv` and `cp` are allowed to move and copy files. Content
  edits still use the Edit/Write tools. Keep the other rules.
- Do not edit README.md or examples/README.md, except for the files.md links.
  Ticket 04 sweeps their prose.

## Acceptance criteria

- [x] CLAUDE.md has about 200 lines or fewer, and its File Map has 40 lines or
      fewer.
- [x] files.md is not in the repo. Each of its non-obvious decisions is in a
      module header, or is listed under `## Comments`.
- [x] Each deleted CLAUDE.md paragraph has its content in a glossary, an ADR or
      a module header, or is listed under `## Comments`.
- [x] `rg 'files\.md'` finds no link outside `.scratch/`. No doc links to a
      CLAUDE.md anchor that is gone.
- [x] `cargo check` passes, and `cargo doc --all --no-deps` has no warnings.
- [x] The asd-ste100 lint (`ste-lint.py`) passes on CLAUDE.md.

## Comments

Notes from the implementation, for the user:

- **Item docs count as "found".** Most decisions of files.md and many
  paragraphs of CLAUDE.md were already in the docs of their own file. Such a
  doc is the module header, an item doc, or a code comment beside the code. I
  did not copy them into the module header, to keep one source of truth.
  Examples: the `tanh` reason of `rotation_range` and the CPU/GPU trade-off of
  `siso_specialization_decode` are in their config docs. The order of the `A`
  floor is in a comment in `helpers.rs`. Object if you want each of them in
  the header.
- **Moved into module headers or comments** (in no doc before):
  - `src/lib.rs`: "The backend is a run-time value" (the `Dispatch` pin, no
    `<B>` on the library types).
  - `src/mamba1/mamba1.rs`: `A` does not depend on the input, and its init.
  - `src/unified/mod.rs`: the model keeps its fused projections, and only the
    Muon plan splits them. The item docs of the lists hold the details.
  - `src/packing/tests.rs`: why the packed check has a tolerance.
  - Each `#[backend_extension]` site: all sites must have the same list. At
    the request of the user, `build.rs` now sets a `cube_backend` cfg (as
    burn does), and each site gates its `Cube` arm with it.
- **The block-agnostic rule** of ADR-0002 is now a bullet under "Rules → Code".
  The summary stays at three lines.
- **Found in burn-stack docs, not in this repo.** I deleted these CLAUDE.md
  sentences: the flag logit and `missing_lm_head` (`modules/network.rs`), the
  loader worker that breaks a capture (`examples/device.rs`), "prime then
  step" (its glossary, `modules/hnet/network.rs`). The list of the captured
  passes of each example is in `examples/README.md`.
- **Not decisions.** files.md described the coverage of the test files, the
  defaults of the configs, and the content of each `info/` note. The test
  headers, the `#[config(default = …)]` lines and the header of each note hold
  them. The convention of the `scripts/` checks is now in "Where to look".
- **A changed rule.** CLAUDE.md said "Do not read or write outside the root".
  The user decided the new rule: read and write are free in the root and in
  `../{burn-stack,burn,cubecl,cubek}/`. To read `../py/`, ask first. I also
  put each other path outside these directories (for example, `../papers/`)
  under "ask first".
- **Kept**: the `## Agent skills` section in its `###` form, because
  `setup-matt-pocock-skills` updates it in place.
- **Left for ticket 04**: old prose in the headers that I did not write
  ("Kalman gate", "SSM state", "slot" for a row, "folded axis"). The notation
  table in the `mamba3.rs` header still says "`sequence` length, **folded**"
  for `s`.
- No new term and no new decision came up.
