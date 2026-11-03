# 01: One plant frame around the three scans

Status: needs-triage

Blocked by: None. It needs its own design round before an agent starts.

From the architecture review of 2026-11-03. Line numbers are those of commit
`42ca3d0e`. The review compiled nothing here.

**Files:** `src/mamba3/double_ssd/double_ssd/mod.rs:77-267,497-585,658-778,823-893`,
`src/mamba3/single_ssd/single_ssd/mod.rs:116-345,561-636`.

**Problem:** One plant option costs three edits: the double-SSD forward, the
single-SSD forward and `step`. A reader must diff 300 lines to find what a pathway
changes. Normalised, 114 of 126 pre-scan lines and 44 of 53 tail lines of double-SSD
occur in single-SSD. `step_project` and `step_finish` are a third copy, one rank
lower. The copies drift: `D:241-248` asserts the shapes of B and C, and single-SSD
does not.

**Direction:** One routine from `project_in` to the rotated B, C and masses. One
routine from the readout to `out_proj`. `step` enters at `tokens = 1`, as the
comments at `D:742-744` and `D:945-948` already describe. Each pathway keeps only
its middle.

**Decided with the user:**

- ADR-0007 is amended in this work. Independence covers the scans. The pre-scan
  work and the tail are one module with their own tests.
- The user runs `./bench.sh` before and after. Decode kernels are shape-sensitive
  (`helpers.rs:771-774`). If decode slows down, `step` keeps its own frame, and only
  the two forwards share one.

**Do this before `02-in-projection-layout.md`,** so that only one split remains to
convert.

**Trade-off:** A wrong split or gate then fails both sides alike, so parity cannot
see it. The frame needs its own tests.
