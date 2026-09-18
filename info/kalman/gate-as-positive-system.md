# Gate as Positive System

### What a second, input-only scalar recurrence buys a Mamba-3 block, and what it does not

> Reference note for `burn-mamba`. It is the authority for this crate's decisions
> around `src/mamba3/positive/` — `Mamba3Config::{gain, tropical}`, the log-semiring
> scan, and the three ports those systems reach the plant through.
>
> The other three notes classify the terms of the *plant's* local objective:
> [`rotation-as-optimization.md`](../mamba-3/rotation-as-optimization.md) the quadratic term,
> [`trapezoid-as-integration.md`](../mamba-3/trapezoid-as-integration.md) the linear term along
> time, [`mimo-as-batch.md`](../mamba-3/mimo-as-batch.md) the linear term along rank. This one
> is about something else entirely: a system that is **not** the plant, running
> beside it, whose output sets the plant's coefficients.
>
> Every numbered claim below is checked in float64 by
> [`scripts/kalman/gate_as_positive_system.py`](../../scripts/kalman/gate_as_positive_system.py)
> (29 checks, section numbers match). The script depends only on `numpy` and the
> equations reproduced here — not on the crate — so the results stand
> independently of the implementation. The accuracy tables are from
> `examples/tally/`, whose tests recompute them.

---

## Abstract

A Mamba-3 head is a **linear parameter-varying** plant: linear in its state, with
coefficients `ρₜ = g(uₜ)` read off the token. This note is about making that
schedule *dynamic* — `ρₜ = g(uₜ, σₜ₋₁)` for a second state `σ` — without losing
the chunkwise pass. The condition is one-way: `σ` may read the input and never the
plant, so the cascade still scans.

Two members of that shape are built (§3, §4). Both are **nonnegative 2×2 matrices
acting projectively on a nonnegative 2-vector**, i.e. positive linear systems,
carried in log coordinates where one scan serves both. The first computes a
Kalman gain — a decay that reads how much evidence the head holds — and the
second a soft (max, +) register: counters with a floor, running maxima, resets.

The audit is in §5, and it is stricter than it looks: a classifier gets a
surprising amount of a positive system's output for free, because the block's
per-token gate cross-multiplies and its final norm divides. What survives is
narrow and specific — **growth** (a decay above one, which `α = exp(Δ·A) ≤ 1`
cannot be), **range** (log coordinates where the linear form needs `e^{S·v}`),
and **a discount applied inside the recurrence** that no readout can undo — and
each of the three is worth 5–10 points against the best stock arm on a task built
to isolate it, against nothing at all on a task that is not.

---

## 1. Scope and results

1. Both members are positive linear systems and share one log-semiring scan
   (§2). The projective read is shift-invariant, so the scan renormalises for
   free, and a doubling prefix equals the sequential fold.
2. The Kalman member's information form **is** the covariance-form Kalman filter
   of a random walk observed through the block's own write (§3.1–3.2); `κ = 0`
   returns the projected decay exactly (§3.3).
3. Its precision has a ceiling `Λₜ < 1/qₜ + mₜ` whatever the history (§3.4) and
   contracts the Hilbert distance at a rate with **no `α` in it** (§3.5); at
   `q = 0, α ≡ 1` the gain collapses like `1/t`, which is what the ceiling
   prevents (§3.6). Under the trapezoid's two installments all of this holds
   with `γ` for `m`, and `Λ` is the plant's own weight on ones (§3.9).
4. Its ageing is hyperbolic in elapsed time and saturating in evidence held; no
   geometric discount fits the family (§3.7: best fit 0.89 relative error).
5. The tropical member is the same scan with a positive log-decay: above the hard
   (max, +) recursion and within `ln(t + 1)` of it (§4.1–4.2), with the
   projection's scale as its temperature (§4.3), computing Lindley recursions and
   running maxima (§4.4).
6. **What a classifier gets free** (§5): `sign(v − Σ/n) = sign(v·n − Σ)`, so the
   read-side port is not a capability gap on a decision task; a gap leaves the
   estimate unchanged and moves only its weight; and a maximum is a sum in the
   exponential domain, so what a running maximum needs from the register is its
   *range*, not the maximum.
7. What is left is measured, not argued: against 100% for the positive systems,
   the best stock arms found reach 83% (a floor), 90.1% (a maximum over twelve
   values) and 83.8–93.0% (a discount inside the recurrence) — and ~100% on
   versions of the last two that do not isolate the capability (§6).

---

## 2. Positive systems, in log coordinates

A member is a per-head scalar recurrence written as a nonnegative matrix acting
projectively:

```text
  (M ⊗ s)ᵢ = log Σⱼ exp(Mᵢⱼ + sⱼ)        the log-semiring product
  read(s)  = s₀ − s₁                     the projective read
```

Ordinary arithmetic on positive numbers *is* this semiring in log coordinates
(§2.1), so nothing is approximated by working there — and two things are gained.
The read ignores a common shift of the matrix (§2.2), so a prefix product may be
renormalised at every combine and never drifts; and the two members' natural
entries (`1/α` at a reset token, `e^{a}` for a growing register) stay finite,
where a linear form overflows.

The product is associative, so the prefix products come from a doubling scan —
`⌈log₂ T⌉` full-width combines, the schedule `rotation::quat_cumprod` already
uses — and it equals the sequential fold (§2.3), which is what the crate's
`positive::scan::fold` exists to be tested against.

Two shapes occur:

| shape | matrix | carries |
|---|---|---|
| projective | all four entries | a log-ratio (`ln Λ`) |
| affine | `[[a, b], [−∞, 0]]` | an absolute value (`c`) |

The affine one admits no shift (its lower row is fixed), and needs none: its read
is absolute rather than a ratio.

---

## 3. The Kalman member

Mamba's update is an exponential moving average of a rank-one setpoint whose gain
`1 − α` is *projected*. A Kalman filter has the same form with the gain
*computed* from a variance that accumulates evidence. In information form
(`Λ = 1/P`, `η = Λ·S`) the per-head scalar filter is an SSD whose decay is
computed:

```text
  dₜ = αₜ / (1 + qₜ·αₜ·Λₜ₋₁)
  Λₜ = dₜ·Λₜ₋₁ + mₜ                       the same recurrence on ones
  ηₜ = dₜ·ηₜ₋₁ + (the block's own write)   the plant, unchanged
```

with `mₜ = Δₜ` the step's mass when it has no left endpoint (`Trapezoid::None`;
§3.9 is the split) and `qₜ = κₕ·Δₜ·exp(rₜ)` the doubt it injects. Its matrix is

```text
  M = log [[α(1 + q·m),  m],
           [α·q,         1]]
```

**§3.1–3.2 — it is the textbook filter.** The recurrence above reproduces the
covariance-form Kalman filter of a random walk (`a² = 1/α`, process noise `q`,
measurement precision `m`), in both its precision and its mean, and the matrix
generates the same precision. The information form is not an approximation of the
filter; it is the filter in the coordinates the plant already uses.

**§3.3 — containment.** At `q = 0` the computed decay is `α` exactly, so `κ = 0`
is the stock block bit for bit. In the implementation the correction is a
`softplus` of a floored `ln q`, which is exactly zero there, and the crate's tests
assert the whole block is unchanged.

**§3.4 — a ceiling.** `Λₜ < 1/qₜ + mₜ` whatever came before. Strong evidence
therefore dominates the readout for a *bounded* time rather than one growing with
its strength — the property the scaled member (`q = 0`, decay projected) does not
have, where recovery costs `ln(excess)/(−ln λ)` and grows with the excess.

**§3.5 — a contraction.** For `q, m > 0` the matrix is entrywise positive, so by
Birkhoff's theorem it contracts the Hilbert distance `|ln Λ − ln Λ'|` by at least
`tanh(¼·ln(1 + 1/(q·m)))` per step. The rate has **no `α`** in it: a long-memory
head forgets its confidence's initialisation as fast as a short one. That is what
makes a cache's initial `Λ`, and any rounding in it, harmless.

**§3.6 — what the ceiling prevents.** At `q = 0` with `α ≡ 1` the same recurrence
is an exact precision-weighted running mean, and its gain falls like `1/t`: the
head stops writing. Any `q > 0` rules it out.

**§3.7 — the shape of ageing.** After `k` uninformative steps the filter's prior
is worth `Λ/(1 + k·q·Λ)`: hyperbolic in `k`, saturating in `Λ`. A projected decay
gives `αᵏ·Λ`, a product of the two where the filter has a sum of reciprocals. No
`(α, scale)` fits the family over a spread of both: the best is 0.89 relative
error. This is the one capability the plant cannot imitate, and `tally-drift` is
the task that measures what it is worth.

### 3.8 The two arms

`q = κₕ·Δₜ` (the **tied** arm, `Gain::Kalman`) is the Brownian reading — doubt
grows with elapsed time — and costs **zero in-projection channels**: one per-head
`κ`. Its limit is that `q` and `m` both scale with `Δ`, so it has no
*uninformative* token: a step that injects little doubt also carries little
evidence. `Gain::KalmanProjectedNoise` multiplies `q` by a projected `exp(rₜ)`,
which buys exactly that separation — a gap — at one channel per (head,
micro-step).

### 3.9 Under the trapezoid

With a `β` tap the plant pays each sample in two installments: `γₜ = λₜ·Δₜ` at
its own step, and `νₜ₊₁ = (1 − λₜ₊₁)·Δₜ₊₁` one step later, transported by that
step's decay (`β = ν·d`). Counting the late installment before the predict,

```text
  Lₜ = Λₜ₋₁ + νₜ,    dₜ = αₜ / (1 + qₜ·αₜ·Lₜ),    Λₜ = dₜ·Lₜ + γₜ
```

is still one Möbius map — shift by `ν`, predict, shift by `γ`:

```text
  M = log [[α(1 + qγ),  αν + γ(1 + qαν)],
           [αq,         1 + qαν       ]]
```

and it makes `Λ` **exactly** the plant's recurrence on ones: the total weight of
every sample written (a fresh cache's zero tap slot among them), with `η/Λ`
their weighted mean. The ceiling becomes `Λₜ < 1/qₜ + γₜ`; the contraction's
exact rate uses `γ + αν(1 + qγ) ≥ γ`, so `tanh(¼·ln(1 + 1/(q·γ)))` is still an
`α`-free bound. Without a tap, `γ = Δ` and `ν = 0`, and this is §3's element.

A **lag-`u`** tap (the `Vertical*` patterns at `u > 1`) is transported across
its whole gap, `∏ d` over `u` steps, which reads `u` earlier precisions — no
2×2 map does. The gate enters it like a lag-1 installment and over-counts by
`νₜ·(dₜ − ∏ d)`, so `Λ` bounds the plant's weight from above instead of
equalling it.

---

## 4. The tropical member

The affine element with its log-decay allowed to be **positive**:

```text
  cₜ = log(exp(cₜ₋₁ + aₜ) + exp(bₜ))   →   max(cₜ₋₁ + aₜ, bₜ)
```

**§4.1–4.2.** The soft value is above the hard one and within `ln(t + 1)` of it,
in the projection's units. **§4.3.** Scaling `(a, b)` by `s` is the same as
running at temperature `1/s`, so the temperature is not a knob: the in-projection
owns it, and a construction picks its scale so the margin clears `ln(T + 1)`.

**§4.4.** `(a, b) = (±1, 0)` is the Lindley recursion — a counter clamped at zero,
i.e. an integrator with anti-windup; `(0, v)` is a running maximum; `(−∞, b)` a
reset. All exact, all in one scalar.

The member's *distinguishing* property is in the exponential domain, where the
same recursion reads `Λₜ = e^{aₜ}·Λₜ₋₁ + e^{bₜ}`. That is a linear recurrence
whose **decay is above one** whenever `a > 0`. A Mamba head's decay is
`α = exp(Δ·A)` with `A < 0`: at most one, structurally. So a floor — which needs
the state to grow *and* be caught from below — is the part of the semiring the
plant cannot reach, and a maximum, whose `a` is zero, is not (§5.4).

---

## 5. The ports, and what a classifier gets for free

A positive system reaches the plant at one of three places: the **decay**
(`ln dₜ` replaces `ln αₜ` where it is formed, so every consumer follows), the
**read** (`y ← y·(Λ + ε)^(−ωₕ)`, which turns the information `η` into the
estimate `η/Λ`), or the **readout** (`y ← y + c·eₕ`). Which of those is a real
capability depends on what the block is asked for, and the audit is less generous
than it looks.

**§5.1 — the gate cross-multiplies.** `sign(v − Σ/n) = sign(v·n − Σ)`. A block
with a sum head, a count head and its per-token gate decides any comparison
against a mean without dividing; the network's final RMSNorm does the same job a
second way, turning `(Σ, n)` into a direction. **The read port is therefore free
to a classifier** — it buys a *value*, not a decision. (`examples/tally` has no
"mean" rung for this reason.)

**§5.2 — a gap does not move the estimate.** Scaling `η` and `Λ` by the same
factor leaves `η/Λ` fixed and lowers only the weight behind it. So a discount's
effect appears one step later, when new evidence is blended in — which is why a
task that means to measure it must put fresh evidence after the gap and probe
after that.

**§5.3 — the size of the drift separation.** On a stream whose probes sit at the
estimate's own edge, the best geometric-discount arm reaches ~96% in f64 (and
97.7% in the crate's f32 example sweep) against the filter's 100%. Real, and
small.

**§5.4 — a maximum is a sum in the exponential domain.** `Σ exp(S·vₛ)` at decay
one decides "is `v` a new maximum" exactly, provided `e^S` exceeds the sequence
length — no logarithm anywhere. So a running maximum does not need the semiring;
what it needs is the semiring's **range**, since that arm spans
`e^{S·(v_max − v_min)}` (1.5e15 in the script's setting, against f32's ~1e7)
where the register spans `S·v`. The dial is the alphabet: `examples/tally/record`
ties at six values and separates at twelve, and every in-projection channel being
an affine read of one embedding is why more heads do not buy more digits.

---

## 6. Consequences for this crate

### 6.1 What is structural

`Gain` and `Tropical` are structural, like `Trapezoid` and `RotationKind`: a
Kalman member allocates `κ` and `ω` per head (the output norm keeps `ω`, which
scales the SSD's readout against the `D` skip), one cache slot, and — on the projected-noise arm — one in-projection channel per (head,
micro-step); a register allocates two channels, a slot and `eₕ`. Stock is the
default and is recovered *exactly*, not approximately, at `κ = 0` / `e = 0`.

### 6.2 Where the decay is substituted

Inside `helpers::trapezoidal_coefficients`, between `da` and `α`. That is not a
convenience: the trapezoid's transports, the tap slots' carried decay and
single-SSD's key scale all read the log-decay, and substituting it where it is
*formed* makes them consistent by construction rather than by audit.

### 6.3 What the ladder measured

| task | what it needs | best stock arm found | the positive system |
|---|---|---|---|
| a counter with a floor (`tally-depth`) | growth (§4) | 83% trained, 86.5% swept | **100%** |
| a discount inside the recurrence (`tally-drift`) | §3.7 | 83.8% trained, 90.7–93.0% swept | **100%** (91.8% trained) |
| a running maximum over 12 values (`tally-record`) | range (§5.4) | 90.1% trained, 79.5% swept | **100%** (95.4% trained) |

The trained columns are short runs at equal budget, so they compare arms rather
than establish ceilings; the hand-built column is the exactness claim. Two
further numbers are the ladder's real content, both ties: the maximum ties at
six values (a small enough alphabet fits the exponential
encoding), and the drift task ties everywhere except where the estimate actually
decides — 98–99% on ordinary streams against 90.7% on the probe bursts. A
capability that only shows under a family built for it is still a capability, but
the family is not optional. The tables are `examples/tally/README.md`'s; the
rungs' tests recompute them.

### 6.4 What does not follow

- **Not a state-tracking claim.** These systems are commutative and scalar; they
  add nothing to what `RotationKind` classifies. A register holds a counter, not
  a group element.
- **Not an argument from optimality.** The Kalman member is the optimal filter of
  a model the data need not follow; what §3 establishes is that the block can
  *be* that filter, and §5 is how much that is worth on a decision task.
- **Not yet a training claim.** Everything above is representability plus three
  tiny trained rungs. Whether a computed decay helps a language model at scale is
  untested here; the one thing the crate can promise is that the join is exact,
  so the question is answerable by turning `κ` on.

---

## 7. Reproduction

```bash
python3 scripts/kalman/gate_as_positive_system.py      # 29 checks, float64
cargo test --lib positive -- --test-threads=1   # the implementation's own suite
cargo test --release --example tally-depth -- --nocapture   # and the ladder's
```

---

## References

- R. E. Kalman, *A New Approach to Linear Filtering and Prediction Problems*,
  1960 — the filter §3.1 checks against, in its covariance form.
- A. H. Jazwinski, *Stochastic Processes and Filtering Theory*, 1970 — the
  fading-memory filter, which is this note's scaled member (`q = 0`).
- G. Birkhoff, *Extensions of Jentzsch's theorem*, 1957 — the contraction of §3.5,
  via the Hilbert projective metric.
- F. Baccelli, G. Cohen, G. J. Olsder, J.-P. Quadrat, *Synchronization and
  Linearity*, 1992 — (max, +) linear systems; §4's members are theirs.
- L. Farina, S. Rinaldi, *Positive Linear Systems*, 2000 — the class both members
  belong to.
- Prior art on the *scaled* member, which several architectures already build per
  key channel: ABC (Peng et al., 2022) and LightNet (2024) form a cumulative
  log-normaliser and decay by its increment; RWKV-4's WKV is (from memory,
  unverified) the same object with a current-token bonus. The **added** member (feedback on confidence) and the
  tropical readout are what this note adds; a full literature check is still
  owed before any novelty claim beyond that.
