# Mix rotation kinds, each with its own fraction

Status: needs-triage

## Idea

Today, one rotation kind turns a prefix of the state, and the rotation fraction
is 0.5 or 1. The rest of the state does not turn (`Real1D`).

Replace the single fraction with a list of `(rotation kind, fraction)` parts.
The fractions of a list add up to 1. Each part turns its own share of the state
with its own kind. Example: `Rotor4D`@0.2 + `Quaternion4D`@0.2 +
`Complex2D`@0.2 + `Real1D`@0.4.

The single rotation fraction then becomes deprecated.

## Open questions

- How the parts map onto the state dimensions (order, alignment to the block
  size of each kind).
- What the in-projection and the Muon specs look like for a mixed list.

## Comments
