# Agent briefs

`/mattpocock-skills:triage` writes an agent brief from the template of its
`AGENT-BRIEF.md`. A brief in burn-mamba or in burn-stack also has the sections below,
each one when its condition holds.

The implementer of a brief starts cold. It has the ticket, `CLAUDE.md` and the repos,
but not the triage conversation. So the brief holds each fact and each reason that the
conversation found.

| Section | When | What it holds |
|---|---|---|
| **Context** | The ticket is one of a split, or a blocker gives it an interface | The order of the tickets, the path of each one, and what each blocker delivers before this ticket starts |
| **Move or removal** | Most of the diff deletes code | Whether the capability moves or goes. For a move: where it goes, and a check that the old way and the new way agree, run before the deletion |
| **Names that stay** | The diff deletes a name, and other names look like it | A table of the look-alike names to keep, with what each one is. Then the exact list of the items that go |
| **References** | Always | The module headers of the mechanism, and the tests and the harnesses to copy. The seam of each new test, from `docs/agents/testing.md` of the repo of the ticket. For an official reading: the source files, after `docs/source-terms.md` |
| **Burn autodiff rules** | The change trains a tensor, keeps a graph in a cache, moves a cache to another backend, or tests a gradient | The rules and the traps of the burn-stack doc `docs/burn/autodiff-semantics.md` that apply, by section |
| **Rejected options** | The ticket asks for an ADR | Each option that the triage rejected, with its reason, in the brief or in its **Triage notes**. The ADR gives them as its considered options |

Each check of the acceptance criteria names its tolerance (a constant or a
`dtype_tol(…)` value), and the command that runs it.
