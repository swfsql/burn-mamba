# 09: One training loop for the reset ladder

Status: needs-triage

Blocked by: None

From the architecture review of 2026-11-03. Line numbers are those of commit
`42ca3d0e`.

**Files:** `examples/reset/*/training.rs`. Precedent: `examples/tally/shared/training.rs:45,138`.

**Problem:** Against majority (284 code lines), a `diff` gives 39 lines for rotor,
spinor and swap, 41 for spinor-product and 55 for quintic. A reader needs that `diff`
to find what a rung changes. The common part holds the resume seed, `loader_device`,
the checkpoint pair, the final-validation rule, the budget stop and `end_epoch`.

**Direction:** One shared module of the reset ladder holds the loop. Each rung keeps
its dataset, model, scoring and tests. About 1200 lines go.

**Against it:** An example is also documentation, so a rung reader opens a second
file. All-rung behaviour edits stopped after `Trainer` arrived. The pressure is low.
