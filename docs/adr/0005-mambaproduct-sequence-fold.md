# MambaProduct is a sequence fold with full-size micro-steps

MambaProduct runs `u = micro_steps` full Mamba-3 micro-steps for each token. The
micro-steps are `u` consecutive positions of the existing recurrence, so there
is no new kernel and no cache change. Only the writes fold: C, the gate `z`,
the `D` skip and the output stay at token resolution.

Each micro-step has its own `Δ`, so it has its own decay. In Mamba, the decay
is not a control of its own: it comes from the step size (`α = exp(ΔA)`, and
`Δ` also weights the write and paces the rotation). So `α ≡ 1` on the interior
micro-steps, as in DeltaProduct, would also stop the rotation. The micro-steps are full-size: the interval of a
token is `u`× longer, and that buys the reach.

## Considered Options

- **The mechanism of DeltaProduct** (non-commuting rank-one factors). The
  curvature of Mamba is isotropic, so each factor is a scalar and cannot
  rotate. The rotation comes from the `RotationKind` instead
  (`info/mamba-3/rotation-as-optimization.md`).
- **Subdivided micro-steps** (`Δⱼ = Δ/u`). They stay reachable from the
  full-size micro-steps, so the full-size form is a superset.
