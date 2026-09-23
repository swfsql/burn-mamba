# The `tally` ladder

Three examples on symbol streams. Each isolates what a **positive system**
buys a Mamba-3 block. A positive system is a second, per-head scalar
recurrence beside the plant: it reads only the input and sets the coefficients
of the plant (`src/mamba3/positive/`, `info/kalman/gate-as-positive-system.md`).

The [`reset`](../reset/README.md) ladder climbs the transition group of the
*plant*. This ladder holds the plant at the bottom rung of that ladder, and
climbs what is **beside** it:

| rung | what it asks for | the system it needs | measured |
|---|---|---|---|
| [`tally-depth`](#tally-depth) | count with a floor | a tropical register, `Tropical::MaxPlus` | needed: trained stock reaches 83% |
| [`tally-drift`](#tally-drift) | age evidence by how much of it there is | a Kalman gate, `Gain::KalmanProjectedNoise` | needed: the best stock arm measured reaches 90.7%, trained 83.8% |
| [`tally-record`](#tally-record) | hold a running maximum over a **wide** alphabet | the same register, for its range | needed at 12 values: trained stock reaches 90.1%, the register 95.4% at equal budget |

The (max, +) semiring of the register buys two things, and neither is the
maximum itself:

- **Growth** (`tally-depth`): a floor needs `a > 0`, a decay above one. The
  decay of the plant, `α = exp(Δ·A)`, is at most one.
- **Range** (`tally-record`): a linear state can also compute a maximum, as a
  sum in the exponential domain. But that route needs a channel with a range
  that f32 does not have, once the alphabet is wide enough.

## The shared shape

Every rung reads a symbol stream and puts **every** position in one of two
classes. A position whose class is not a function of the history (a reset, a
tie, a symbol that fixes the answer by itself) is **not scored**.

The three rungs share these settings:

- **The plant is at the floor of the `reset` ladder**: `state_rank = 1`,
  `RotationKind::Real1D`, `Trapezoid::None`. So a rung measures the positive
  system, not the SSM. In the hand-built `tally-depth` and `tally-record`
  blocks, the state of the plant stays at zero.
- **`ignore_last_residual`**: the head sees only the block.
- **A reference head** with a constant output. The `final_norm` of the network
  then keeps the *direction* of (answer, reference), and the two-class head
  reads its sign, as `reset-majority` explains.

`tally-depth` runs at `d_model = 2` (the floor for three symbols).
`tally-record` and `tally-drift` run at `d_model = 3`. Each `model.rs` records
why.

Every rung closes the three shortcuts of the `reset` ladder: the label is not
a function of the current symbol, Mamba-3 has no short convolution to widen a
window with, and the residual is off.

## What a classifier gets for free

The `reset` ladder selects the smallest `d_model` that allows a construction.
Here the width must also make the *shortcuts* impossible, and there are more
of them than expected. The tests measure two, and each one removes or demotes
a rung:

- **The gate cross-multiplies.** A decision like "is `v` above the mean" never
  needs a division: `sign(v·n − Σ)` is the same decision, and a sum head, a
  count head and the per-token gate of the block give it. The `final_norm` of
  the network does the same job another way: it turns `(Σ, n)` into a
  direction, that is a ratio. So the `C` port (read `η/Λ` instead of `η`) is a
  real capability, but a classifier gets it **for free**. That is why this
  ladder has no "mean" rung. `tally-drift` is what survives: a discount
  *inside* the recurrence, which no readout can undo.
- **The embedding is free.** A block can make any one function of the symbol
  an affine channel, with a suitable placement of the embeddings. That is how
  a stock block reaches a maximum over a small alphabet. It cannot make a dozen
  independent functions (the latch route), and it cannot stretch one channel
  past the seven digits of f32. `tally-record` measures that limit.

## Usage

```bash
# training and running inference in flex (fp32) — <rung> ∈ depth|drift|record
cargo run --release --example tally-<rung> -- --training --inference

# the ablation arm: the same block without its positive system
cargo run --release --example tally-<rung> -- --training --inference -- --stock

# the claims below, measured: the hand-built solutions and the sweeps
cargo test --release --example tally-<rung> -- --nocapture
```

`--stock` changes a **fresh** model config only (on reload, the saved config
wins).

- See `burn-mamba/Cargo.toml` for other features or backend information.
- See `burn-mamba/examples/README.md` for the CLI usage overview.

## Reading the tables

The conventions of the `reset` ladder apply, plus three of this ladder:

- **Every ablation gets the best readout that it allows**, swept over a grid.
  So the number bounds the ablated *architecture*, not one fit of it.
- The sweeps are coarse, and each sweeps one *family* of readouts. So an
  ablation column is a **lower bound** on the ablated block. That is why every
  rung also has a **trained** `--stock` row at equal budget. On `tally-record`,
  training finds an arm that the sweeps do not have (the exponential encoding),
  and the alphabet is wide enough that this arm needs more range than f32 has.
- An ablation that the block cannot express at all (for example, a second
  head that cross-multiplies) is computed in `f64` from the same streams, and
  its label says so.
- Short trained runs (for example, 1000 batches) are at equal budget. They
  compare arms, not ceilings. The hand-built row carries exactness.

---

## tally-depth

The rung that needs the register. The model reads `(` / `)` / `R`. At every `)`
it reports whether the bracket depth is **still positive**. A `)` at depth 0
closes nothing (the depth does not go negative):

```text
  symbols   (  (  )  )  )  (  )  R  )  (  )
  depth     1  2  1  0  0  1  0  0  0  1  0
  target    .  .  i  o  o  .  o  .  o  .  o
```

Only the `)` positions are scored: after a `(` the depth is at least one, for
any history.

<details>
<summary>Why this task</summary>

The clamp is the point of the rung. `cₜ = max(cₜ₋₁ + aₜ, 0)` is Lindley's
recursion. It is **linear in the (max, +) semiring**, so one register computes
it exactly. In the exponential domain it is `Λₜ = e^{aₜ}·Λₜ₋₁ + e^{bₜ}`, a
linear recurrence whose decay `e^{a}` is **above one** on a `(`. The plant
cannot have that: `α = exp(Δ·A)` with `A < 0` is at most one, by construction.

The rung closes two more shortcuts:

- **A running sum.** The `floor` family pushes the sum below zero, where the
  depth stops. On its scored positions, the sum and the depth disagree a
  quarter of the time.
- **A decayed sum** (a multiplicative decrement). It never reaches zero, so it
  imitates the floor, until a deep excursion (the `deep` family).

The two families defeat opposite ends of the decay range. So read the *worst
family* column.

</details>

<details>
<summary>Measured</summary>

78 parameters. Chance is 50%.

| | random | floor | deep | worst |
|---|---|---|---|---|
| best per-symbol lookup (no memory) | 78.6% | 74.6% | 72.4% | 72.4% |
| best **selective gate**, register off (8 decays × 3 writes × 9 thresholds) | 91.9% | 96.1% | 98.5% | **86.5%** |
| trained, register off (`-- --stock`), 80 epochs | 95.7% | 98.5% | 83.1% | **83.1%** |
| **hand-built** register block, no training | **100%** | **100%** | **100%** | **100%** |
| trained, 80 epochs | **100%** | **100%** | **100%** | **100%** |

The three column maxima of the sweep come from three *different* points. A
decay of 1 (the plain sum) reaches 98.5% on `deep` and 74.4% on `floor`. A
decay of 0.7 reaches 96.1% on `floor` and 78.6% on `deep`. No single setting
has both. The trained stock arm lands in the same band. The register needs
none of this.

`handmade_register_solves_every_family` writes every weight in closed form.
The `Δ` of the plant is `1e-12` in both heads, so its state is zero throughout,
and the register is the whole model.

</details>

<details>
<summary>Notes</summary>

- **The floor of the register is `b`, not zero.** `b` is the floor that the
  *token itself* guarantees: `S` after a `(` (the depth is at least one), `0`
  after a `)`. With `b = 0` everywhere, the first token costs one: a fresh
  register holds "the max of nothing", so the count would lag by one.
- **The scale is the temperature.** The soft maximum exceeds the hard one by
  at most `ln(T + 1)` ≈ 3.5 at `T = 32`, in register units. So a bracket is
  worth `S = 12`, and the threshold is at `S/2`. There is no temperature knob:
  the scale of the in-projection is the temperature.

</details>

---

## tally-drift

The rung of the Kalman gate. Its answer is a **measurement, not a wall**. The
model reads 33 values (from `−2` to `+2` in steps of `1/8`) and a gap symbol
`~`. At every value, it reports whether the value is **above the running
estimate of the level**. The level is a random walk that moves only during
gaps. A gap is elapsed time: it adds doubt without evidence.

```text
  symbols   +1   +1   +2   ~    ~    -1   +1
  estimate  1.0  1.0  1.33 1.33 1.33 0.0  0.36
  target     .    .    +    .    .    -    +
```

(`Q_GAP = 0.5`: the two gaps shrink the weight of three values to `0.75`, so
the `-1` pulls the estimate to `0`. A value on its estimate is not scored.)

<details>
<summary>Why this task</summary>

The optimal estimate is the information-form filter. Its whole content is
**how much the past is still worth** after `k` gap tokens:

```text
  Λ ← Λ / (1 + k·q·Λ)
```

This is hyperbolic in `k` and saturating in `Λ`: after a long run, the past is
worth `1/(k·q)` votes, for any length of the run. A projected decay can only
discount geometrically, `αᵏ·Λ`: a *product* of the two quantities, where the
filter has a sum of their reciprocals. (The script's §3.7 puts the best
geometric fit at a relative error of 0.89.)

The gate computes the filter exactly, because the recurrence of the gate
**is** the filter: `dₜ = αₜ/(1 + qₜ·αₜ·Λₜ₋₁)`. The construction uses `Δ = 1`,
`q ≈ 0` at a value (evidence, no doubt), and `Δ ≈ 0`, `q = q_gap` at a gap
(doubt, no evidence). `Gain::KalmanProjectedNoise` exists for this separation.
The tied arm (`q ∝ Δ`) cannot make it.

The rung closes three more shortcuts:

- **A running mean.** Ignoring the gaps changes the answer at 6–9% of the
  scored positions, and at 24% on `knife`.
- **The estimate read as `η/Λ` alone.** A gap scales `η` and `Λ` together, so
  it does *not* change the estimate. Only the weight of the prior changes, and
  that shows only when new evidence is blended in.
- **A stream of ordinary observations.** They land far enough from the mean
  that every arm agrees, so a family of them measures nothing.

The last two points give the construction of the `knife` family: a short run,
a gap, one new observation (its pull depends on the weight of the prior, the
disputed quantity), and then a **burst** of probes, each a sixteenth of a unit
above or below the estimate of that moment. Every probe is itself evidence, so
each next probe is aimed again. The burst makes the *scored* positions the ones
that the estimate decides.

</details>

<details>
<summary>Measured</summary>

219 parameters. Chance is 50%.

| | random | straddle | chain | knife | worst |
|---|---|---|---|---|---|
| best per-symbol lookup (no memory) | 78.3% | 78.1% | 79.5% | 66.0% | 66.0% |
| the join at **`κ = 0`** (the same block and read, projected decay; 11 gap decays) | 98.9% | 99.2% | 99.2% | 90.7% | **90.7%** |
| best **linear (sum, count) + the cross-multiplication of the gate** (f64; 11 × 11 × 5) | 98.9% | 99.0% | 99.3% | 93.0% | **93.0%** |
| trained, gate off (`-- --stock`), 1000 batches | 95.1% | 94.6% | 95.0% | 83.8% | **83.8%** |
| trained, 1000 batches | 98.7% | 98.2% | 98.2% | 91.8% | **91.8%** |
| **hand-built** Kalman block, no training | **100%** | **100%** | **100%** | **100%** | **100%** |

Read the `knife` column. The estimate decides its scored positions. The other
columns show that an ordinary stream does not measure this at all. On `knife`:

- the gate is exact,
- the best projected decay reaches 90.7%,
- the best linear state that cross-multiplies reaches 93.0%,
- the two trained arms are 8 points apart at equal budget.

On the other families, all four arms are within a point of each other.

That spread is the real content of the rung. The capability is not "estimate
a level" (a linear state does that well). It is **a discount inside the
recurrence**. It shows only where the weight of the prior decides, and there
it is worth 7–10 points. The dynamics behind it (a bounded recovery after
strong evidence, a confidence that contracts) are in
`info/kalman/gate-as-positive-system.md`.

</details>

<details>
<summary>Notes</summary>

- **`a_floor = 1e-8`.** The estimator holds an unweighted sum across a whole
  sequence, and the floor of the block on `|A|` is the only thing that decays
  it. At the default `1e-4`, that leak is comparable to the margins of this
  task.
- **The first value of a sequence is not scored.** With an empty prior, it
  *is* its own posterior mean, so it cannot be above or below it.
- **The bursts make the separation, not the probe precision.** A coarser grid
  (a quarter unit) with a wider probe offset (0.13) moves the best linear arm
  by under a point. Scoring *bursts* of probes, instead of a stream where
  ordinary observations outnumber them, moves it by 7. `tally-depth` applies
  the same discipline when it scores only its `)` tokens.
- **Wider families do not help.** Wider run and gap lengths make the
  comparisons obvious again. The numbers above are at the setting with the
  largest separation.

</details>

---

## tally-record

The **range** rung. The model reads twelve values (`1`…`12`) and a reset `R`.
At every value, it reports whether the value is a new maximum since the last
`R` (strictly).

```text
  symbols   3  1  4  4  2  R  2  5  1  6
  max       3  3  4  4  4  -  2  5  5  6
  target    +  -  +  -  -  .  +  +  -  +
```

<details>
<summary>Why this task, and why the width of the alphabet is the rung</summary>

A maximum is `max(cₜ₋₁, vₜ)`, so one register holds it for any alphabet
size, and the hand-built block is exact.

The interesting part is the route *without* a register. A maximum is also a
**sum in the exponential domain**: at decay one, `Σ exp(S·vₛ)` exceeds
`exp(S·vₜ)` exactly when an earlier value matched or beat `vₜ`, if `e^S` is
larger than the sequence length. That test is linear in that domain, so no
logarithm is necessary (`scripts/kalman/gate_as_positive_system.py` §5.4
checks the identity in f64).

**The cost of this route is range.** It must hold `e^{S·(v_max − v_min)}` in
one in-projection channel. Every channel is an affine read of the *same*
embedding, so the whole block shares one mantissa of resolution: more heads
do not buy more.

- At six values and 32 tokens, the span is `e^{20} ≈ 5·10⁸`, and a trained
  stock block solves the task (~100%).
- At twelve values and 64 tokens (this rung), the span is
  `e^{45.8} ≈ 8·10¹⁹`. f32 holds about `10⁷`, and the hand-built arm falls to
  59.8% on `climb`.

The rung closes three more shortcuts:

- **A comparison against a decaying average.** The average is *below* the
  maximum. So `edge` (every value within one step of the maximum) defeats it
  by construction, and `plateau` defeats it (it sets the maximum early and
  never approaches it again).
- **The exponential encoding**: range, above.
- **One latch per value, selected by the gate.** It is exact, but it needs
  one state scalar per value and one channel per threshold. `d_model = 3`
  gives about three independent channels: the tests report a residual of
  `3.6e-15` for a channel that the embedding spreads, and `0.36`–`0.46` for
  each of the twelve indicators.

</details>

<details>
<summary>Measured</summary>

165 parameters. Chance is 50%.

| | random | plateau | climb | bands | edge | worst |
|---|---|---|---|---|---|---|
| best per-symbol lookup (no memory) | 70.2% | 78.7% | 67.3% | 74.7% | 61.6% | 61.6% |
| best **decaying comparison**, register off (8 × 4 × 4 × 7 grid) | 89.8% | 84.5% | 92.3% | 87.5% | 83.1% | **79.5%** |
| hand-built **exponential-domain** arm, register off, in f32 | 88.7% | 69.6% | 59.8% | 86.5% | 66.8% | **59.8%** |
| trained, register off (`-- --stock`), 1000 batches | 97.2% | 96.9% | 90.1% | 97.6% | 93.6% | **90.1%** |
| trained, 1000 batches | 98.5% | 97.8% | 97.9% | 97.7% | 95.4% | **95.4%** |
| **hand-built** register block, no training | **100%** | **100%** | **100%** | **100%** | **100%** | **100%** |

The trained rows did not converge. Only the arm with the register has an exact
solution to converge *to*.

`edge` and `bands` are the families that separate the arms:

- `edge` puts every token within one step of the maximum, so an approximately
  right answer stops paying.
- `bands` keeps each segment in a quarter of the alphabet, so the model needs
  resolution at the bottom as well as at the top.

</details>
