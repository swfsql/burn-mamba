# 01: Notation and doc rules

Status: resolved

Blocked by: None (can start immediately)

**What to build:** The Notation table gives the correct symbols of the papers.
`CLAUDE.md` tells an agent how a doc can use a source word, and how to read a
letter in a formula. Research: `research.md`, part 5.

Line numbers are those of commit `2039187`. Load the `asd-ste100` and
`mattpocock-skills:writing-for-agents` skills before you edit `CLAUDE.md`.

## Decisions from the grill

**The `_Avoid_` rule.** Add one line to the Docs rules of `CLAUDE.md`: `_Avoid_`
applies to our own words. A doc can use a source word in quotes or backticks
when it cites that source. ADR-0008 records this decision.

**Wrong cells of the Paper column.**

| Place | Row | Old | New |
|---|---|---|---|
| `CLAUDE.md:166` | `m` | `M` | `R` |
| `CLAUDE.md:169` | `l` | `Q` | `Q` (Mamba-2), `C` (Mamba-3) |
| `src/mamba3/mamba3.rs:198` | `m` | `M` | `R` |
| `src/mamba3/mamba3.rs:201` | `l` | `Q` | `C` |

`src/mamba2/mamba2.rs:80` keeps `Q`, because that table is of Mamba-2.

**Our own letter for the rank.** Our comments keep upper-case `M` for the MIMO
rank (ADR-0004, `info/mamba-3/architecture-deltas.md`). The Notation rule makes
an upper-case `R` read as a relation of `r`, our state rank. Add a note at the
tables of `CLAUDE.md` and `src/mamba3/mamba3.rs`: the `R` of the paper is the
`M` of our comments.

**Letters in formulas.** Add to the Notation paragraph of `CLAUDE.md`: a letter
in a formula is a paper symbol, not a shape letter. Then `hₜ` is the state, and
the suffix `_bhl` still has `h` = `nheads`. At `src/mamba3/mamba3.rs:80`, "the
cached `h`" becomes "the cached state".

**The other cells stay.** Research 5.1 marks the cells of `b`, `s`, `t`, `i`,
`h`, `g`, `p`, `r`, `a` and `v` as incomplete. Do not change them. Their
spelling in each source goes to the Letters section of `docs/source-terms.md`
(ticket 03).

## Acceptance criteria

- [x] `CLAUDE.md` and `src/mamba3/mamba3.rs` show the new cells and the note.
- [x] `src/mamba2/mamba2.rs:80` still shows `Q`.
- [x] `CLAUDE.md` has the `_Avoid_` rule and the rule for letters in formulas.
- [x] `rg -n 'cached .h.' src/mamba3/mamba3.rs` finds no hit.
- [x] `cargo check` and `cargo doc --all --no-deps` give no warnings.

## Comments

- The two field docs of `mimo_rank` in `src/mamba3/mamba3.rs` also said
  "Paper: `M`". They are the same error as the table cell, so they now say
  `R`.
- The cached-state line keeps "SSD core" on one line, so the search of ticket
  02 finds it.
- The symbols `R` and `C` come from the citations of `research.md`. This ticket
  did not read the TeX again.
- Open question from the review: the rule for letters in formulas has the
  wording of this ticket. Some formulas use a shape letter as an index, for
  example `B̄ₜ[m]` and `Rₜ,ᵤ` in `src/mamba3/mamba3.rs`.
- The paper `C` of the chunk length is also the readout matrix `C`. Ticket 03
  records this clash (research 5.3).
