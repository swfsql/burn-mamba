# Two Mamba-3 pathways, selected by the cache type

Mamba-3 keeps two pathways for its trapezoidal recurrence. Double-SSD (one
standard SSD call for each term of the trapezoid) is simple and easy to check.
Single-SSD (one SSD call, the form of the official kernels) uses about half the
training memory. The cache variant that the caller gives (`DoubleSsd` or
`SingleSsd`) selects the pathway, and no cache selects single-SSD. In the middle
of a sequence, the single-SSD accumulator has a different meaning. So it has its
own cache type, and a chunked pass cannot mix the two.

## Consequences

- The two accumulators are equal at call boundaries, so the caches convert with
  `From`.
- `step` runs the double-SSD recurrence for both cache variants.
