# Keep each fast form beside its simple form, and prove them equal

This minimal reference has three SSD paths and two Mamba-3 pathways. Each fast
form (`SerialRecalculated` with its recompute backward, single-SSD) stays next
to a simple form (`Minimal`, double-SSD) that is easy to check against the
paper. Tests assert that the forms agree on outputs, final caches and
gradients. So the simple form stays the readable reference, and the fast form
gives the memory savings.

## Consequences

The defaults are the fast forms: `SerialRecalculated(None)`, and single-SSD
when the caller gives no cache.
