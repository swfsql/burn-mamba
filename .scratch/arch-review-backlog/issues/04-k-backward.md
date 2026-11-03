# 04: One K1–K3 backward for both pathways

Status: needs-triage

Blocked by: None

From the architecture review of 2026-11-03. Line numbers are those of commit
`42ca3d0e`. Not compiled.

**Files:** `src/mamba3/double_ssd/ssd/serial_recalculated/combined_backward.rs:256-306,412-543`,
`src/mamba3/single_ssd/ssd/serial_recalculated/combined_backward.rs:418-539`.

**Problem:** 566 lines hide a difference of a few terms: `y_diag`, `scale`, the strict
mask and the seeds. Normalised, 194 of 271 double-SSD lines occur in single-SSD. The
forward K1–K4 and `k4_ssd_state_passing_backward` are already shared. The K1 suffix
sum, the K2 backward and BLUE are copies. K3 differs only in `k_scaled` and the
`d_scale` split. A K3 fix needs two edits (`e43b32e3`, `0ac8e539`). Single-SSD lacks
three `san` guards that double-SSD has (`d_c_blue`, `diff`, `d_da_blue`).

**Direction:** The K1, K2, K3 and BLUE backwards move beside their forwards in
`double_ssd/ssd/serial_recalculated/serial_recalculated.rs`. Each
`combined_backward` becomes a short assembly.

**ADR-0007 holds:** each fast form still meets autodiff on its own simple form.

**Cost:** Helpers must take tensors by value and drop them at the same points. A
clone that a caller holds raises the GPU peak. The user checks it with `./bench.sh`.

**A later step, worth exploring:** Mamba-2 joins this core. By reading, with no
numeric check, its K1–K5 are the double-SSD K1–K5 at `m = 1`, `u = 1`, with
`v = dt·x` and `da = dt·A`, plus the D skip. It needs a Mamba-2 bench by the user,
and the fast form stops mirroring `_mamba_chunk_scan_combined_bwd` line by line.
