# 08: One agreement harness

Status: needs-triage

Blocked by: None

From the architecture review of 2026-11-03. Line numbers are those of commit
`42ca3d0e`.

**Files:** `src/mamba2/ssd/ssd_path/tests.rs:16-304`,
`src/mamba3/double_ssd/ssd/ssd_path/tests.rs:16-320`,
`src/mamba3/single_ssd/ssd/ssd_path/tests.rs:13-365`,
`../burn-stack/src/utils/test_helpers.rs:105`.

**Problem:** "Agree" has two definitions: Mamba-2 compares with `max_abs_diff`, and
single-SSD with `max_rel_diff`. Each file defines the same 6 helpers.
`check_grads_match_two_paths!` is fixed at two alternates. A new SSD path needs 3
test bodies and a macro change. No `ssd_path` test sets `restarts: Some` or
`seed_bnhpr: Some`.

**Direction:** One test module runs a list of SSD paths on fresh graphs with one
loss. It compares outputs, final state and gradients with the simple form. Each tree
keeps its input maker, gradient list and shape cases.

**Cost:** Gradient rank varies per field, so the harness needs a macro or host
read-backs first (`src/mamba2/ssd/ssd_path/tests.rs:159`). No production cost.
