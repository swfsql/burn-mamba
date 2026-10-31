# 03: Rename the tap pattern

Status: resolved

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

- [x] `rg -w 'Trapezoid|trapezoid|TrapezoidSpec|trapezoid_spec|with_trapezoid|has_interior_tap'`
      over the code finds no hit. Prose "trapezoid" is left only where it means
      the rule.
- [x] `rg -i 'interior tap|far tap|horizontal tap|vertical tap'` finds no hit.
- [x] `cargo check`, `cargo test --lib -- --test-threads=1`,
      `cargo test --examples -j 1 -- --test-threads=1` pass (flex).
- [x] `cargo doc --all --no-deps` has no warnings.

## Comments

- The rename also covers the field `trapezoid` of `Mamba3` and of the two
  cache configs (`Mamba3DoubleSsdCacheConfig`, `Mamba3SingleSsdCacheConfig`),
  and the test locals `trapezoid`. All of them are `tap_pattern` now.
- These identifiers used an old tap term, so they follow the term:
  - `interior_tap_crosses_tokens` becomes `lag1_tap_crosses_tokens`.
  - `far_tap_crosses_tokens` becomes `tap_crosses_tokens`. The local
    `far_open_1s1` (also the parameter of `kalman::LogMasses::new`) becomes
    `tap_open_1s1`. In this code, "far tap" meant the tap at `tap_lag`. Under
    `HorizontalReset`, that tap is a lag-1 tap, so "lag-`u` tap" is the wrong
    name for it. A bare "tap" names it, as in `tap_lag` and `tap_slots`. The
    prose writes "the tap at `tap_lag`".
- These names keep "interior" or "far". Each one stays, for the reason given:
  - `nu_interior` (a field of `TrapezoidCoeffs`), `nu_interior_bsh`,
    `nu_interior_buh`, `far_shifted_bsh`, `interior_shifted_bsh`,
    `interior_bsh`, the local `interior` (the `α·νⁱⁿᵗ` term) of
    `src/mamba3/positive/tests.rs`, and `nu_int`, `nu_far` in the scripts. They
    spell the math symbols `νⁱⁿᵗ` and `νᶠᵃʳ`, which this ticket keeps.
  - `interior_gap_decay`: the inside of the gap of a tap, not a tap.
  - "Interior" for a point inside the `tanh` range, or for the micro-steps
    inside a token (ADR 0005, `product/mod.rs`, the script), and the far part
    of the context scores (`examples/common/context.rs`). These are other
    meanings.
  - The names from the member names: the `vertical_*` and `horizontal_*` test
    names, `cfg_vertical`, the script labels (`vertical (lag u)`,
    `reset-horizontal`, …), and "the tap graph is "vertical"" (why the member
    has the name `Vertical`).
- Prose "trapezoid" stays where it means the rule: its taps, masses, tap slots
  and coefficients, and "the trapezoid is on/off" (trapezoidal against plain
  exponential-Euler). The note `info/mamba-3/trapezoid-as-integration.md` and
  its script keep their names, because they are about the rule.
- The prose names the extra tap of a two-tap member "the lag-1 tap of a
  two-tap member" (or pattern). It does not write "the second tap", because
  "second tap" already means the `β` tap ("No second tap" under `None`).
- The new name `has_lag1_tap` is `false` for `HorizontalCarryOver` and
  `HorizontalReset`, and `lag1_tap_crosses_tokens` is `false` for
  `HorizontalCarryOver`. Their one tap is a lag-1 tap, but it is the tap at
  `tap_lag`. The docs say this. A name such as `has_two_taps` would not need
  the note. This question is open for the user.
- The script `scripts/mamba-3/trapezoid_as_integration.py` follows the prose.
  Its labels "(today)" and "today's formula" lose the changelog word. It still
  passes (74 checks).
- The acceptance `rg -i 'interior tap|far tap|horizontal tap|vertical tap'`
  finds only the _Avoid_ lines of `GLOSSARY.md`.
