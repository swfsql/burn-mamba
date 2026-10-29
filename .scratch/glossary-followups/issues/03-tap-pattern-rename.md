# 03: Rename the tap pattern

Status: ready-for-agent

Blocked by: 01

**What to build:** The code and the docs separate two concepts. The trapezoid
is the discretisation rule of Mamba-3. The tap pattern is the choice of which
taps exist. The two taps have one name each: the lag-1 tap and the lag-`u` tap (see
`GLOSSARY.md`: Trapezoid, Tap, Lag, Lag-1 tap, Lag-`u` tap, Tap pattern). A user
sets `tap_pattern` in a Mamba-3 config.

## Rename map (decided in the grill)

| Old | New |
|---|---|
| the enum `Trapezoid` | `TapPattern` |
| the config field `trapezoid`, `with_trapezoid` | `tap_pattern`, `with_tap_pattern` |
| `TrapezoidSpec`, `trapezoid_spec` | `TapPatternSpec`, `tap_pattern_spec` |
| the module `trapezoid` (`mv`, then fix the `mod` and `use` lines) | `tap_pattern` |
| `has_interior_tap` | `has_lag1_tap` |
| prose "interior tap", "horizontal tap", "near tap" | "lag-1 tap" |
| prose "far tap", "vertical tap" | "lag-`u` tap" |
| prose "trapezoid" where it means the choice | "tap pattern" |

Keep these names:
- The enum member names (`HorizontalReset`, `HorizontalCarryOver`, `Vertical`,
  `VerticalPlusHorizontal…`, `None`).
- The names of the rule itself: `TrapezoidCoeffs`, `trapezoidal_coefficients`,
  "trapezoidal", and prose "trapezoid" where it means the rule.
- The math symbols (`νⁱⁿᵗ`, `νᶠᵃʳ`).

Do not touch the experiment directories `examples/*/tmp/`. Old configs stop
loading, and that is correct (no config compatibility).

## Method

- A scripted replacement is allowed for the identifiers in this ticket (the
  user approved it). Review every hit with `rg` afterwards.
- Edit the prose with the Edit tool, site by site.
- Update every doc mention: CLAUDE.md (its File Map line too), module headers,
  `info/`, README.md, examples/README.md.

## Acceptance criteria

- [ ] `rg -w 'Trapezoid|trapezoid|TrapezoidSpec|trapezoid_spec|with_trapezoid|has_interior_tap'`
      over the code finds no hit. Prose "trapezoid" is left only where it means
      the rule.
- [ ] `rg -i 'interior tap|far tap|horizontal tap|vertical tap'` finds no hit.
- [ ] `cargo check`, `cargo test --lib -- --test-threads=1`,
      `cargo test --examples -j 1 -- --test-threads=1` pass (flex).
- [ ] `cargo doc --all --no-deps` has no warnings.

## Comments
