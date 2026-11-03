# 07: One row-end read for padded and packed rows

Status: needs-triage

Blocked by: None

From the architecture review of 2026-11-03. Line numbers are those of commit
`42ca3d0e`.

**Files:** `src/mamba3/mamba3.rs:645-709,1647,1856-1927`, `src/mamba2/mamba2.rs:840-868`,
`src/mamba1/mamba1.rs:442-453`, `src/packing.rs:288-315`, `src/padding.rs:39-57`.

**Problem:** `match packed { None => window(..), Some(p) => restart_window(..) }`
occurs at `M3:674-683`, `M3:1899-1912`, `M3:1913-1918` and `mamba2.rs:857-866`. Each
site also does `last_origin`, then `Tensor::cat`. A mistake reads a stale carry with
a correct shape.

**Direction:** One reader, built once per call from the padding and the resets. It
gives the last `L` real positions of a stream, from its carry and origin. 5 fields in
3 families read through it. It keeps the plain `narrow` when no row has padding,
because `cat` and gather cost memory.

**Also:** `src/mamba3/mamba3.rs:1647` has a private `Packed`, with the name of
`burn_stack::utils::Packed`. Give it another name in this work.
