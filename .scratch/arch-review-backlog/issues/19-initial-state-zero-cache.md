# 19: The learnable initial state is the state of the zero cache

Status: needs-triage

Blocked by: 18

From the architecture review of 2026-11-16. Line numbers are those of commit
`7807d31b`. The review compiled nothing.

**Files:** `src/mamba2/mamba2.rs:194,955,1086`,
`src/mamba2/ssd/{minimal,serial}.rs`,
`src/mamba2/ssd/serial_recalculated/serial_recalculated.rs:43`,
`src/mamba3/double_ssd/double_ssd/mod.rs:263,288`,
`src/mamba3/single_ssd/single_ssd/mod.rs:373`,
`src/mamba3/{double_ssd,single_ssd}/ssd/` (the 2 input bundles and the 6 scans),
`src/family/tests/block_contract.rs:27-31`, `GLOSSARY.md` ("Learnable initial state").

**Problem:** `init_state_hpr` goes through 3 input bundles to 9 scans. The 3 `Minimal`
scans add it to the initial state. The 6 serial scans panic ("not yet implemented").
So the default SSD path panics when `has_learnable_init_state` is on. Two more facts
break block contracts when the state is not zero:

- `step` does not read it (`src/mamba2/mamba2.rs:1086`), so the forward is not equal to
  the step unrolled.
- Each forward adds it to the state of its cache. So a forward in two windows adds it
  two times, and is not equal to one forward.

The block contracts leave `InitState` out for this reason. A grep found no test and no
example that sets the option.

**Direction:** The zero cache holds `h₀`, broadcast over the rows. Each SSD path, both
pathways, `step` and each segment of a packed row with no origin then start from it.
A carried cache already holds its state, so no call adds `h₀` to it. The input bundles
and the scans lose the field, and the 6 panics go.

**This changes a documented behaviour.** The glossary says that each forward adds it
to the state of its cache. The official code does the same with `init_states`
(`../py/state-spaces/mamba/`). At the init (`h₀ = 0`), the values do not change. The
user decides this at triage. The glossary entry changes with the code.

**Also in burn-stack:** `zero_caches_for` (`../burn-stack/src/modules/layers.rs:799`)
builds each slot from application 0. An untied `InitState` needs the application of
each slot. Split this ticket by repo at triage.

**Not checked:** The gradient of `h₀` through `zero_cache` under a gradient horizon
and in a captured step.
