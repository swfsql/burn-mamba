# 01: The prose ratchet also reads `.scratch/`

Status: needs-triage

Blocked by: None (can start immediately)

**What to build:** `check_prose` of `scripts/check.sh` reads each changed `.rs`
and `.md` file, also the files of `.scratch/`. At HEAD, the line
`grep -v '^\.scratch/'` drops them. So a spec or a ticket can get a new hard
STE violation, and the check does not fail.

The maintainer made this decision in a burn-mamba-example session. That repo
has a copy of `check_prose` that reads `.scratch/`
(`../burn-mamba-example/scripts/check.sh`). The burn-stack tracker has a ticket
for the same step in its own check:
`../burn-stack/.scratch/check-script-followups/issues/01-prose-ratchet.md`.

## Acceptance criteria

- [ ] `check_prose` reads the changed `.md` files of `.scratch/`.
- [ ] `scripts/check.sh` passes. A file of `.scratch/` that fails gets a fix of
      its prose, or a comment in its ticket.

## Notes

- The ratchet compares each file with `origin/main`. A file that is new since
  `origin/main` starts at zero hard violations.
- A name next to its synonym is a hard violation (`synonym-rotation`). For
  example, the glossary term `Erase` in a text that also says `delete`. A code
  span around the name stops the violation. The spec of the docs overhaul of
  burn-mamba-example needed two.

## Comments
