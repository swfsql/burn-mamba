# 21: One value for the geometry of a Mamba-3 block

Status: needs-triage

Blocked by: None

From the architecture review of 2026-11-16. Line numbers are those of commit
`7807d31b`.

**Files:** `src/mamba3/mamba3.rs:377-464,467-512,1101-1207,1463-1499`.

**Problem:** `Mamba3` and `Mamba3Config` each compute the same facts:
`in_proj_layout` (`:501` and `:1183`, two equal bodies), `num_rotation_blocks`,
`d_inner` and `nheads`. The block also stores four counts that follow from the config:
`rotation_dim`, `num_rotation_angles`, `num_rotation_channels` and `num_quat_blocks`.
`tap_pattern_spec` and `rotation_spec` build two more bundles from the same fields. A
new option of the plant changes both sides.

**Direction:** One value holds the geometry and the non-parameter settings of a block.
The config computes it once. The block holds it as one `#[module(skip)]` field, and
`in_proj_layout`, the two specs and the fresh caches read it.

**Not checked:** Which of the plain fields of `Mamba3` (`dt_limit`, `a_floor`,
`rotation_range`, the counts) a record holds now, and so whether saved records change.
Mamba-2 and Mamba-1 can have the same shape. That was not read.

**Cost:** A break of the public fields of `Mamba3`.
