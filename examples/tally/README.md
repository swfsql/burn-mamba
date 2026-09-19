# The `tally` ladder

Three examples on symbol streams, each isolating what a **positive system**
buys a Mamba-3 block — a second, per-head scalar recurrence that reads only the
input and sets the plant's coefficients (`src/mamba3/positive/`,
`info/kalman/gate-as-positive-system.md`).

Where the [`reset`](../reset/README.md) ladder climbs the *plant's* transition
group as the state grows, this one holds the plant at that ladder's bottom rung
(`state_rank = 1`, `RotationKind::Real1D`, `Trapezoid::None`, no residual) — in
two of the three rungs its state never moves at all — and climbs what sits
**beside** it:

| rung | what it asks for | the system it needs | measured |
|---|---|---|---|
| [`tally-depth`](#tally-depth) | count with a floor | a tropical register, `Tropical::MaxPlus` | needed: trained stock reaches 83% |
| [`tally-drift`](#tally-drift) | age evidence by how much of it there is | a Kalman gate, `Gain::KalmanProjectedNoise` | needed: the best stock arm measured reaches 90.7%, trained 83.8% |
| [`tally-record`](#tally-record) | hold a running maximum over a **wide** alphabet | the same register, for its range | needed at 12 values: trained stock reaches 90.1% against the register's 95.4% at equal budget |

The third rung is the one that says *why* the (max, +) semiring helps, because it
is the one that can be switched off by shrinking the task. A maximum is a **sum
in the exponential domain** — `Σ exp(S·vₛ)` at decay one, with the record test
linear in that domain — so a linear state computes it exactly, and at six values
and 32 tokens a trained stock block does (~100%). What that route costs is **range**: it
needs `e^{S·(v_max − v_min)}` inside one channel, which at twelve values and 64
tokens is `e^{45.8} ≈ 8·10¹⁹` and f32 has seven digits. The register carries
`S·v` instead, and the rung separates.

So the semiring buys two things, and neither of them is the maximum itself:
**range** (here) and **growth** — `a > 0` is a decay above one, which the plant's
`α = exp(Δ·A) ≤ 1` cannot be, and which a floor needs (`tally-depth`).

## The shared shape

Every rung reads a symbol stream and classifies **every** position into two
classes. Positions whose class is not a function of the history — a reset, a
tie, a symbol that fixes the answer by itself — are **not scored**.

All three run one block with the plant at the `reset` ladder's floor (so what a
rung measures is the positive system, not the SSM), `ignore_last_residual` (so
the head sees the block alone), and a **reference head** whose output is
constant: the network's `final_norm` then keeps the *direction* of
(answer, reference) and the two-class head reads its sign, exactly as
`reset-majority` explains.

The three shortcuts every rung closes are `reset`'s: the label is not a function
of the current symbol, Mamba-3 has no short convolution to widen a window with,
and the residual is off.

## What a classifier gets for free (and why one rung is missing)

The `reset` ladder picks the smallest `d_model` that makes a construction
possible. Here the width also has to make the *shortcuts* impossible, and there
are more of them than expected. Two are measured in the tests, and both are
reasons a rung is absent or demoted:

- **The gate cross-multiplies.** Deciding "is `v` above the mean" never needs a
  division: `sign(v·n − Σ)` is the same decision, and a sum head, a count head
  and the block's per-token gate produce it. The network's `final_norm` does the
  same job a second way, turning `(Σ, n)` into a direction, i.e. a ratio. So the
  `C`-port — reading `η/Λ` instead of `η` — is a real capability that is
  **free to a classifier**, and there is no "mean" rung here. `tally-drift` is
  what survives: a discount applied *inside* the recurrence, which no readout
  can undo.
- **The embedding is free.** A block can make one function of the symbol an
  affine channel by placing the embeddings to suit — which is how a stock block
  reaches a maximum, at a small enough alphabet. What it cannot do is make a
  dozen independent ones (the latch route), nor stretch one channel past f32's
  seven digits, which is what `tally-record` prices.

## Usage

```bash
# training and running inference in flex (fp32) — <rung> ∈ depth|record|drift
cargo run --release --example tally-<rung> -- --training --inference

# the ablation arm: the same block with its positive system removed
cargo run --release --example tally-<rung> -- --training --inference -- --stock

# the claims below, measured: the hand-built solutions and the sweeps
cargo test --release --example tally-<rung> -- --nocapture
```

- See `burn-mamba/Cargo.toml` for other features or backend information.
- See `burn-mamba/examples/README.md` for the CLI usage overview.

## Reading the tables

Same conventions as the `reset` ladder, plus two of this ladder's own:

- **every ablation is given the best readout it admits**, swept over a grid, so
  the number bounds the ablated *architecture* rather than one fitting of it;
- the sweeps are coarse and each is one *family* of readouts, so an ablation
  column is a **lower bound** on the ablated block — which is why every rung also
  carries a **trained** `--stock` row at equal budget. On `tally-record` training
  finds an arm the sweeps have no term for (the exponential encoding below), and
  the alphabet is sized so that arm costs more than f32 has;
- where an ablation is not expressible in the block at all — a second head that
  cross-multiplies, say — it is computed in `f64` from the same streams and
  labelled as such.

---

## tally-depth

The rung the register is needed for. The model reads `(` / `)` / `R` and
reports, at every `)`, whether the bracket depth is **still positive** — where a
`)` at depth 0 closes nothing rather than going negative:

```text
  symbols   (  (  )  )  )  (  )  R  )  (  )
  depth     1  2  1  0  0  1  0  0  0  1  0
  target    .  .  i  o  o  .  o  .  o  .  o
```

Only `)` positions are scored: after a `(` the depth is at least one whatever
the history.

<details>
<summary>Why this task</summary>

The clamp is the rung. `cₜ = max(cₜ₋₁ + aₜ, 0)` is Lindley's recursion —
**linear in the (max, +) semiring**, so one register computes it exactly. In the
exponential domain it is `Λₜ = e^{aₜ}·Λₜ₋₁ + e^{bₜ}`, a linear recurrence whose
decay `e^{a}` is **above one** on a `(`. That is the part the plant cannot have:
`α = exp(Δ·A)` with `A < 0` is at most one, by construction. Beyond the three
shared rows:

| shortcut | why it is closed |
|---|---|
| a running **sum** | the `floor` family drives the sum below zero, where the depth stops; on its scored positions the sum disagrees with the depth a quarter of the time |
| a **decayed** sum (a multiplicative decrement) | it never reaches zero, so it imitates the floor — until a deep excursion, which is the `deep` family |

The two families defeat opposite ends of the decay range, which is what makes
the *worst family* the number to read.

</details>

<details>
<summary>Measured</summary>

78 parameters. Chance is 50%.

| | random | floor | deep | worst |
|---|---|---|---|---|
| best per-symbol lookup (no memory at all) | 78.6% | 74.6% | 72.4% | 72.4% |
| best **selective gate**, register off (8 decays × 3 writes × 9 thresholds) | 91.9% | 96.1% | 98.5% | **86.5%** |
| trained, register off (`-- --stock`), 80 epochs | 95.7% | 98.5% | 83.1% | **83.1%** |
| **hand-built** register block, no training | **100%** | **100%** | **100%** | **100%** |
| trained, 80 epochs | **100%** | **100%** | **100%** | **100%** |

The sweep's three column-maxima come from three *different* points: a decay of 1
(the plain sum) reaches 98.5% on `deep` and 74.4% on `floor`; a decay of 0.7
reaches 96.1% on `floor` and 78.6% on `deep`. No single setting has both, the
trained stock arm lands in the same band, and the register needs none of it.

`handmade_register_solves_every_family` writes every weight down in closed form;
the plant's `Δ` is `1e-12` in both heads, so its state is zero throughout and
the register is the whole model.

</details>

<details>
<summary>Notes</summary>

- **The register's floor is `b`, not zero.** `b` is the floor the *token itself*
  guarantees — `S` after a `(` (the depth is at least one), `0` after a `)`.
  Writing `b = 0` everywhere costs the first token: a fresh register carries "the
  max of nothing", so the count would lag by one.
- **The scale is the temperature.** The soft maximum exceeds the hard one by at
  most `ln(T + 1)` ≈ 3.5 at `T = 32`, in register units, so a bracket is worth
  `S = 12` and the threshold sits at `S/2`. There is no temperature knob: the
  in-projection's own scale is it.

</details>

---

## tally-drift

The Kalman gate's rung, and the one whose answer is a **measurement rather than
a wall**. The model reads seventeen values on a quarter-unit grid and a gap
symbol `~`, and reports at every value whether it is **above the running
estimate of the level**. The level is a random walk that moves only during gaps:
a gap is elapsed time, so it adds doubt without adding evidence.

```text
  symbols    +1  +1  +2   ~   ~  -1  +0
  estimate  1.0 1.0 1.3   …   … 0.4 0.4
  target      .   -   +   .   .   -   +
```

<details>
<summary>Why this task</summary>

The optimal estimate is the information-form filter, and its whole content is
**how much the past is still worth** after `k` gap tokens:

```text
  Λ ← Λ / (1 + k·q·Λ)
```

hyperbolic in `k`, saturating in `Λ`: after a long run the past is worth
`1/(k·q)` votes no matter how long the run was. A projected decay can only
discount geometrically, `αᵏ·Λ` — a *product* of the two quantities where the
filter has a sum of their reciprocals (the script's §3.7 puts the best geometric
fit 0.89 relative error away). The gate computes the filter exactly, because the
gate's recurrence **is** the filter: `dₜ = αₜ/(1 + qₜ·αₜ·Λₜ₋₁)`, with `Δ = 1`,
`q ≈ 0` at a value (evidence, no doubt) and `Δ ≈ 0`, `q = q_gap` at a gap (doubt,
no evidence) — the separation `Gain::KalmanProjectedNoise` exists for, the tied
arm's `q ∝ Δ` having no way to make it.

| shortcut | why it is closed |
|---|---|
| a running mean | ignoring the gaps changes the answer at 6–9% of scored positions, and at 24% on `knife` |
| the **estimate** read as `η/Λ` alone | a gap scales `η` and `Λ` together, so it leaves the estimate *unchanged*; only the prior's weight moves, and that shows when new evidence is blended in |
| a stream of **ordinary** observations | they land far enough from the mean that every arm agrees — so a family of them measures nothing, whoever is right |

The last two rows are the `knife` family's construction: a short run, a gap, one
fresh observation (whose pull is set by the prior's weight — the quantity in
dispute), and then a **burst** of probes, each a sixteenth of a unit either side
of the estimate as it then stands. Every probe is itself evidence, so the next
is aimed afresh, and the burst is what makes the *scored* positions be the ones
the estimate decides.

</details>

<details>
<summary>Measured</summary>

219 parameters. Chance is 50%.

| | random | straddle | chain | knife | worst |
|---|---|---|---|---|---|
| best per-symbol lookup (no memory at all) | 78.3% | 78.1% | 79.5% | 66.0% | 66.0% |
| the join at **`κ = 0`** (the scaled member: same block, same read, projected decay; 11 gap decays) | 98.9% | 99.2% | 99.2% | 90.7% | **90.7%** |
| best **linear (sum, count) + the gate's cross-multiplication** (f64; 11 × 11 × 5) | 98.9% | 99.0% | 99.3% | 93.0% | **93.0%** |
| trained, gate off (`-- --stock`), 1000 batches | 95.1% | 94.6% | 95.0% | 83.8% | **83.8%** |
| trained, 1000 batches | 98.7% | 98.2% | 98.2% | 91.8% | **91.8%** |
| **hand-built** Kalman block, no training | **100%** | **100%** | **100%** | **100%** | **100%** |

Read the `knife` column: it is the one whose scored positions the estimate
decides, and every other column is there to show that an ordinary stream does
not measure this at all. On `knife` the gate is exact, the best projected decay
reaches 90.7%, the best linear state that cross-multiplies 93.0%, and the two
trained arms sit 8 points apart at equal budget. On the other families all four
arms are within a point of each other.

That spread is the rung's real content. The capability is not "estimate a
level" — a linear state does that well — it is **applying a discount inside the
recurrence**, which shows up only where the prior's weight decides, and there it
is worth 7–10 points. The dynamics behind it (a bounded recovery after strong
evidence, a confidence that contracts) are in
`info/kalman/gate-as-positive-system.md`.

The two trained rows are short runs (1000 batches, ~4 minutes each) at equal
budget, so they compare arms rather than establish ceilings; the hand-built row
is what carries exactness.

</details>

<details>
<summary>Notes</summary>

- **`a_floor = 1e-8`.** The estimator holds an unweighted sum across a whole
  sequence, and the block's floor on `|A|` is the only thing that decays it; at
  the default `1e-4` that leak is comparable to the margins this task turns on.
- **The first value of a sequence is unscored.** With an empty prior it *is* its
  own posterior mean, so there is nothing to be above or below.
- **Tightening the probe is not what did it.** Going from a quarter-unit value
  grid to an eighth, and the probe offset from 0.13 to 0.06, moved the best
  linear arm by under a point. What moved it 7 was scoring *bursts* of probes
  instead of a stream in which ordinary observations outnumber them — the same
  discipline `tally-depth` applies by scoring only its `)` tokens.
- **Widening the families does not help either.** Spreading run and gap lengths
  further makes the comparisons obvious again; the numbers above are at the
  setting that maximised the separation.

</details>

---

## tally-record

The **range** rung. The model reads twelve values and a reset `R`, and reports at
every value whether it is a new maximum since the last `R` (strictly).

```text
  symbols   3  1  4  4  2  R  2  5  1  6
  max       3  3  4  4  4  -  2  5  5  6
  target    +  -  +  -  -  .  +  +  -  +
```

<details>
<summary>Why this task — and why the alphabet's width is the rung</summary>

A maximum is `max(cₜ₋₁, vₜ)`, so one register holds it at any alphabet size and
the hand-built block is exact. The interesting part is the route that does *not*
use a register: a maximum is also a **sum in the exponential domain**, since
`Σ exp(S·vₛ)` at decay one exceeds `exp(S·vₜ)` exactly when some earlier value
matched or beat `vₜ`, provided `e^S` beats the sequence length. That test is
linear in that domain, so no logarithm is ever taken —
`scripts/kalman/gate_as_positive_system.py` §5.4 checks the identity in f64.

**The route is priced in range.** It must hold `e^{S·(v_max − v_min)}` inside one
in-projection channel, and every channel is an affine read of the *same*
embedding, so the whole block shares one mantissa's worth of resolution — more
heads do not buy more. At six values and 32 tokens that span is `e^{20} ≈ 5·10⁸`
and a trained stock block solves the task (~100%). At twelve values and 64
tokens it is `e^{45.8} ≈ 8·10¹⁹`, f32 holds about
`10⁷`, and the hand-built arm collapses to 59.8% on `climb`.

| shortcut | why it is closed |
|---|---|
| comparing against a decaying **average** | the average sits *below* the maximum, so `edge` — where every value is within one step of it — defeats it by construction, and `plateau` defeats it by setting the maximum early and never approaching it again |
| the **exponential** encoding | range, above |
| one **latch** per value, selected by the gate | it is exact — at one state scalar per value and one channel per threshold. `d_model = 3` gives about three independent channels: the tests report `3.6e-15` residual for a channel the embedding is spread by and `0.36`–`0.46` for each of the twelve indicators |

</details>

<details>
<summary>Measured</summary>

165 parameters. Chance is 50%.

| | random | plateau | climb | bands | edge | worst |
|---|---|---|---|---|---|---|
| best per-symbol lookup (no memory at all) | 70.2% | 78.7% | 67.3% | 74.7% | 61.6% | 61.6% |
| best **decaying comparison**, register off (8 × 4 × 4 × 7 grid) | 89.8% | 84.5% | 92.3% | 87.5% | 83.1% | **79.5%** |
| hand-built **exponential-domain** arm, register off, in f32 | 88.7% | 69.6% | 59.8% | 86.5% | 66.8% | **59.8%** |
| trained, register off (`-- --stock`), 1000 batches | 97.2% | 96.9% | 90.1% | 97.6% | 93.6% | **90.1%** |
| trained, 1000 batches | 98.5% | 97.8% | 97.9% | 97.7% | 95.4% | **95.4%** |
| **hand-built** register block, no training | **100%** | **100%** | **100%** | **100%** | **100%** | **100%** |

The two trained rows are short runs at equal budget (1000 batches, ~4 minutes
each), so they compare arms rather than establish ceilings — neither has
converged, and only the register's arm has an exact solution to converge *to*.

`edge` and `bands` are the families that make the arms part: `edge` puts every
token within one step of the maximum, so being approximately right stops paying,
and `bands` confines each segment to a quarter of the alphabet, so resolution is
needed at the bottom as well as the top.

</details>
