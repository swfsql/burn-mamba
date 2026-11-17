# Two Mamba-3 pathways, selected by the scan options

Mamba-3 keeps two pathways for its trapezoidal recurrence. Double-SSD (one
standard SSD call for each term of the trapezoid) is simple and easy to check.
Single-SSD (one SSD call, the form of the official kernels) uses about half the
training memory. Both stay, because each one is the check on the other
(ADR-0007). The scan options of the call select the pathway
(`Mamba3ScanOptions::pathway`), and the default is single-SSD.

One cache type, `Mamba3Cache`, serves both pathways. In the middle of a
sequence, the single-SSD accumulator has a different meaning from the state. But
a cache exists only at a call boundary, and there the accumulator equals the
state. So a cache from a forward of one pathway continues its sequence on either
pathway.

## Considered Options

- **A cache type for each pathway**: the type of the cache selects the pathway.
  It stops a mix of the two accumulators in the middle of a sequence, but no
  caller can reach that mix. It costs the `From` conversions between the two
  types, and block contracts that cannot reach double-SSD.

## Consequences

- `step` has no pathway. It runs the double-SSD recurrence for a cache from
  either pathway.
- The block contracts run each `Mamba3` setting on both pathways, except the
  untied setting. The `MambaBlock` settings run single-SSD.
