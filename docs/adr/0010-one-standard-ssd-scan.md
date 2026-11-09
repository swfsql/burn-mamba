# Mamba-2 and double-SSD call one standard SSD scan

The `SerialRecalculated` paths of Mamba-2 and of double-SSD call one standard SSD
scan on primitives, in `src/standard_ssd/`. The scan has the axes of Mamba-3:
`mimo_rank` and the read axis. Mamba-2 forms `v = Δ·x` and `da = Δ·A`, calls the
scan with a length-1 `m` axis and `read_stride = 1`, and adds its chain rule and
its `D` skip. A fix to K4 or to the resets is then one fix, and a change to the
batching of the backward reaches both callers.

The cost is the link to the reference. The fast form of Mamba-2 does not mirror
`_mamba_chunk_scan_combined_bwd` line by line, so a reader cannot compare its
statements with that kernel.

## Considered Options

- **Two copies of the scan, one in each family**: each copy can follow its own
  source. A K4 or reset fix then needs a second derivation, and a batching
  change reaches only one family.
- **Mamba-2 as a case of double-SSD in the `mamba3` module**: the `mamba2` feature
  would then need `mamba3`.

## Consequences

- The scan is outside the three family modules. It compiles when the `mamba2` or
  the `mamba3` feature is on, and `mamba2` does not imply `mamba3`.
- ADR-0007 holds for each family. `Minimal` and `Serial` of Mamba-2 do not use the
  scan, so they stay the check on its fast form. A reader can still compare them
  with the reference.
- A bug in the scan fails both callers alike. The agreement tests of each family
  show it, because each family has simple forms that do not use the scan.
- The backward of single-SSD assembles the backwards of K1–K5 of the scan. One K5
  backward serves both K5 forwards, with the scale and the mask diagonal as
  arguments. `Minimal` and `Serial` of single-SSD use autodiff, so they stay the
  check on that backward.
