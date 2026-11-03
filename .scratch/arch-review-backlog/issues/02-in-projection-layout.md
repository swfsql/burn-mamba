# 02: One in-projection layout

Status: needs-triage

Blocked by: `01-plant-frame.md`. It needs its own design round before an agent
starts.

From the architecture review of 2026-11-03. Line numbers are those of commit
`42ca3d0e`.

**Files:** `src/mamba3/mamba3.rs:501-577,1276-1294,1316-1386`,
`src/mamba3/double_ssd/double_ssd/mod.rs:138-181,678-733`,
`src/mamba3/single_ssd/single_ssd/mod.rs:180-225`.

**Problem:** The order of the channel groups is in code at 6 sites: 3 splits,
`split_positive`, `d_in_proj_tail` and `muon_projections`. It is in prose at 7 more.
`Δ, A, λ, μ, noise, a, b` all have width `u·nheads`, so a swap gives no shape error.
A zero-width group "would silently vanish" (`D:146-149`). `assert_plan_fits` checks
only width sums. A grep found no test that ties the Muon order to the split order.

**Direction:** One ordered description of the channel groups, derived from the
config: name, width, axis, optimizer, head or tail. The widths, the Muon plan and a
split into named streams read it.

**Tests:** Each group reaches its stream. The Muon names match the split.

**Cost:** The same tensor ops. `split_into` has a fixed arity, so optional groups
keep `narrow`.
