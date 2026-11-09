# Keep each fast form beside its simple form, and prove them equal

burn-mamba has three SSD paths and two Mamba-3 pathways. Each fast form
(`SerialRecalculated` with its recompute backward, single-SSD) stays next to a
simple form (`Minimal`, double-SSD) that is easy to check against the paper.
Tests assert that the forms agree on outputs, final caches and gradients. So the
simple form is the check on the fast form, and the fast form gives the memory
savings.

This independence covers the scans. Each simple form keeps its own scan, apart
from each fast form. The fast forms of Mamba-2 and double-SSD share one scan
(ADR-0010), and each one meets its own simple form in the tests. The work
around the scans, the entry and the tail of the plant frame, is one module. A
bug in that module fails all forms alike, and agreement cannot show it. So the
module has its own tests against its definitions.

## Consequences

The defaults are the fast forms: `SerialRecalculated(None)`, and single-SSD
when the caller gives no cache.
