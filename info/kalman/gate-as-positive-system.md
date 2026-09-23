# Gate as Positive System

### What a second, input-only scalar recurrence buys a Mamba-3 block, and what it does not

> Reference note for `burn-mamba`. It is the authority for the decisions of this
> crate around `src/mamba3/positive/`: `Mamba3Config::{gain, tropical}`, the
> log-semiring scan, and the three ports through which those systems reach the
> plant.
>
> The other three notes classify the terms of the local objective of the
> *plant*:
>
> - [`rotation-as-optimization.md`](../mamba-3/rotation-as-optimization.md):
>   the quadratic term,
> - [`trapezoid-as-integration.md`](../mamba-3/trapezoid-as-integration.md):
>   the linear term along time,
> - [`mimo-as-batch.md`](../mamba-3/mimo-as-batch.md): the linear term along
>   rank.
>
> This note is about a different thing: a system that is **not** the plant. It
> runs beside the plant, and its output sets the coefficients of the plant.
>
> [`scripts/kalman/gate_as_positive_system.py`](../../scripts/kalman/gate_as_positive_system.py)
> checks every numbered claim below in float64 (29 checks, with the same
> section numbers). The script depends only on `numpy` and on the equations in
> this note, not on the crate. So the results do not depend on the
> implementation. The accuracy tables are from `examples/tally/`, whose tests
> compute them again.

---

## Abstract

A Mamba-3 head is a **linear parameter-varying** plant. It is linear in its
state, and it reads its coefficients `ρₜ = g(uₜ)` from the token. This note
makes that schedule *dynamic*, `ρₜ = g(uₜ, σₜ₋₁)` for a second state `σ`, and
keeps the chunkwise pass. The condition goes in one direction only: `σ` may
read the input, but never the plant. Then the cascade still scans.

This note builds two members of that shape (§3, §4). Both are **nonnegative
2×2 matrices that act projectively on a nonnegative 2-vector**, that is,
positive linear systems. Both are carried in log coordinates, where one scan
serves both of them:

- The first member computes a Kalman gain: a decay that reads how much
  evidence the head holds.
- The second member is a soft (max, +) register: counters with a floor,
  running maxima, resets.

The audit is in §5, and it is stricter than it looks. A classifier gets a
large part of the output of a positive system free: the per-token gate of the
block cross-multiplies, and its final norm divides. What survives is narrow
and specific:

- **growth**: a decay above one, which `α = exp(Δ·A) ≤ 1` cannot be,
- **range**: log coordinates, where the linear form needs `e^{S·v}`,
- **a discount applied inside the recurrence**, which no readout can undo.

Each of the three is worth 5–10 points against the best stock arm on a task
built to isolate it. On a task that does not isolate it, it is worth nothing.

---

## 1. Scope and results

1. Both members are positive linear systems, and they share one log-semiring
   scan (§2). The projective read is shift-invariant, so the scan renormalises
   free. A doubling prefix equals the sequential fold.
2. The information form of the Kalman member **is** the covariance-form Kalman
   filter of a random walk, observed through the write of the block itself
   (§3.1–3.2). At `κ = 0` it gives the projected decay exactly (§3.3).
3. Its precision has a ceiling `Λₜ < 1/qₜ + mₜ` for any history (§3.4). It
   contracts the Hilbert distance at a rate with **no `α` in it** (§3.5). At
   `q = 0, α ≡ 1` the gain collapses like `1/t`, and the ceiling prevents this
   (§3.6). Under the two installments of the trapezoid, all of this holds with
   `γ` in place of `m`, and `Λ` is the weight of the plant on ones (§3.9).
4. Its ageing is hyperbolic in the elapsed time and saturates in the evidence
   held. No geometric discount fits the family (§3.7: the best fit has a
   relative error of 0.89).
5. The tropical member is the same scan with a positive log-decay (§4.1–4.4):
   - It is above the hard (max, +) recursion and within `ln(t + 1)` of it.
   - The scale of the projection is its temperature.
   - It computes Lindley recursions and running maxima.
6. **What a classifier gets free** (§5):
   - `sign(v − Σ/n) = sign(v·n − Σ)`, so the read-side port is not a gap in
     capability on a decision task.
   - A gap does not change the estimate. It moves only the weight of the
     estimate.
   - A maximum is a sum in the exponential domain. So a running maximum needs
     the *range* of the register, not its maximum.
7. The remainder is measured, not argued. The positive systems reach 100%. The
   best stock arms that we found reach 83% (a floor), 90.1% (a maximum over
   twelve values) and 83.8–93.0% (a discount inside the recurrence). On
   versions of the last two that do not isolate the capability, the stock
   arms reach ~100% (§6).

---

## 2. Positive systems, in log coordinates

A member is a per-head scalar recurrence, written as a nonnegative matrix that
acts projectively:

```text
  (M ⊗ s)ᵢ = log Σⱼ exp(Mᵢⱼ + sⱼ)        the log-semiring product
  read(s)  = s₀ − s₁                     the projective read
```

Ordinary arithmetic on positive numbers *is* this semiring in log coordinates
(§2.1). So the log coordinates approximate nothing, and they give two
advantages:

- The read ignores a common shift of the matrix (§2.2). So a prefix product
  can be renormalised at every combine, and it never drifts.
- The natural entries of the two members stay finite, where a linear form
  overflows: `1/α` at a reset token, `e^{a}` for a growing register.

The product is associative, so a doubling scan gives the prefix products. It
does `⌈log₂ T⌉` full-width combines, with the schedule that
`rotation::quat_cumprod` also uses. It equals the sequential fold (§2.3). The
crate has `positive::scan::fold` for this reason: the tests compare the scan
with it.

Two shapes occur:

| shape | matrix | carries |
|---|---|---|
| projective | all four entries | a log-ratio (`ln Λ`) |
| affine | `[[a, b], [−∞, 0]]` | an absolute value (`c`) |

The affine shape accepts no shift (its lower row is fixed). It needs none,
because its read is absolute, not a ratio.

---

## 3. The Kalman member

The update of Mamba is an exponential moving average of a rank-one setpoint,
and its gain `1 − α` is *projected*. A Kalman filter has the same form, but it
*computes* the gain from a variance that accumulates evidence. In information
form (`Λ = 1/P`, `η = Λ·S`), the per-head scalar filter is an SSD with a
computed decay:

```text
  dₜ = αₜ / (1 + qₜ·αₜ·Λₜ₋₁)
  Λₜ = dₜ·Λₜ₋₁ + mₜ                       the same recurrence on ones
  ηₜ = dₜ·ηₜ₋₁ + (the block's own write)   the plant, unchanged
```

Here `mₜ = Δₜ` is the mass of the step when it has no left endpoint
(`Trapezoid::None`, §3.9 gives the split). `qₜ = κₕ·Δₜ·exp(rₜ)` is the doubt
that the step injects. Its matrix is:

```text
  M = log [[α(1 + q·m),  m],
           [α·q,         1]]
```

**§3.1–3.2: it is the textbook filter.** The recurrence above gives the
covariance-form Kalman filter of a random walk (`a² = 1/α`, process noise `q`,
measurement precision `m`), in its precision and in its mean. The matrix gives
the same precision. The information form is not an approximation of the
filter. It is the filter, in the coordinates that the plant already uses.

**§3.3: containment.** At `q = 0` the computed decay is `α` exactly, so
`κ = 0` is the stock block, bit for bit. In the implementation, the correction
is a `softplus` of a floored `ln q`, which is exactly zero there. The tests of
the crate assert that the whole block does not change.

**§3.4: a ceiling.** `Λₜ < 1/qₜ + mₜ`, whatever came before. So strong
evidence dominates the readout for a *bounded* time, not for a time that grows
with its strength. The scaled member (`q = 0`, projected decay) does not have
this property: there, recovery costs `ln(excess)/(−ln λ)`, which grows with
the excess.

**§3.5: a contraction.** For `q, m > 0` the matrix is entrywise positive. So,
by the theorem of Birkhoff, it contracts the Hilbert distance
`|ln Λ − ln Λ'|` by at least `tanh(¼·ln(1 + 1/(q·m)))` per step. The rate has
**no `α`** in it: a long-memory head forgets the initialisation of its
confidence as fast as a short one. For this reason, the initial `Λ` of a
cache, and any rounding in it, cause no harm.

**§3.6: what the ceiling prevents.** At `q = 0` with `α ≡ 1`, the same
recurrence is an exact precision-weighted running mean, and its gain falls
like `1/t`: the head stops writing. Any `q > 0` prevents this.

**§3.7: the shape of ageing.** After `k` uninformative steps, the prior of the
filter is worth `Λ/(1 + k·q·Λ)`. This is hyperbolic in `k` and saturates in
`Λ`. A projected decay gives `αᵏ·Λ`: a product of the two, where the filter
has a sum of reciprocals. No `(α, scale)` fits the family over a spread of
both: the best fit has a relative error of 0.89. This is the one capability
that the plant cannot copy, and `tally-drift` is the task that measures its
value.

### 3.8 The two arms

- `q = κₕ·Δₜ` (the **tied** arm, `Gain::Kalman`) is the Brownian reading:
  doubt grows with the elapsed time. It costs **zero in-projection channels**,
  only one per-head `κ`. Its limit: `q` and `m` both scale with `Δ`, so it has
  no *uninformative* token. A step that injects little doubt also carries
  little evidence.
- `Gain::KalmanProjectedNoise` multiplies `q` by a projected `exp(rₜ)`. This
  buys exactly that separation (a gap), at one channel per (head,
  micro-step).

### 3.9 Under the trapezoid

With a `β` tap, the plant pays each sample in two installments:

- `γₜ = λₜ·Δₜ` at its own step,
- `νₜ₊₁ = (1 − λₜ₊₁)·Δₜ₊₁` one step later, transported by the decay of that
  step (`β = ν·d`).

Count the late installment before the predict:

```text
  Lₜ = Λₜ₋₁ + νₜ,    dₜ = αₜ / (1 + qₜ·αₜ·Lₜ),    Λₜ = dₜ·Lₜ + γₜ
```

This is still one Möbius map (shift by `ν`, predict, shift by `γ`):

```text
  M = log [[α(1 + qγ),  αν + γ(1 + qαν)],
           [αq,         1 + qαν       ]]
```

With it, `Λ` is **exactly** the recurrence of the plant on ones: the total
weight of every sample written (the zero tap slot of a fresh cache is one of
them). `η/Λ` is their weighted mean. The ceiling becomes `Λₜ < 1/qₜ + γₜ`. The
exact rate of the contraction uses `γ + αν(1 + qγ) ≥ γ`, so
`tanh(¼·ln(1 + 1/(q·γ)))` is still an `α`-free bound. Without a tap, `γ = Δ`
and `ν = 0`, and this is the element of §3.

A **lag-`u`** tap (the `Vertical*` patterns at `u > 1`) is transported across
its whole gap, `∏ d` over `u` steps. That product reads `u` earlier
precisions, and no 2×2 map does that. The gate adds it like a lag-1
installment and over-counts by `νₜ·(dₜ − ∏ d)`. So `Λ` is an upper bound of
the weight of the plant, not equal to it.

---

## 4. The tropical member

The tropical member is the affine element, with a log-decay that can be
**positive**:

```text
  cₜ = log(exp(cₜ₋₁ + aₜ) + exp(bₜ))   →   max(cₜ₋₁ + aₜ, bₜ)
```

**§4.1–4.2.** The soft value is above the hard one and within `ln(t + 1)` of
it, in the units of the projection.

**§4.3.** To scale `(a, b)` by `s` is the same as to run at temperature `1/s`.
So the temperature is not a knob: the in-projection owns it. A construction
selects its scale so that the margin is more than `ln(T + 1)`.

**§4.4.** `(a, b) = (±1, 0)` is the Lindley recursion: a counter clamped at
zero, that is, an integrator with anti-windup. `(0, v)` is a running maximum.
`(−∞, b)` is a reset. All are exact, and all fit in one scalar.

The property that *distinguishes* the member is in the exponential domain.
There, the same recursion reads `Λₜ = e^{aₜ}·Λₜ₋₁ + e^{bₜ}`. That is a linear
recurrence whose **decay is above one** when `a > 0`. The decay of a Mamba
head is `α = exp(Δ·A)` with `A < 0`: structurally, at most one. A floor needs
the state to grow *and* a lower limit that catches it. So a floor is the part
of the semiring that the plant cannot reach. A maximum, whose `a` is zero, is
not that part (§5.4).

---

## 5. The ports, and what a classifier gets free

A positive system reaches the plant at one of three places:

- the **decay**: `ln dₜ` replaces `ln αₜ` where the block forms it, so every
  consumer follows,
- the **read**: `y ← y·(Λ + ε)^(−ωₕ)`, which turns the information `η` into
  the estimate `η/Λ`,
- the **readout**: `y ← y + c·eₕ`.

Which of those is a real capability depends on the task of the block. The
audit is less generous than it looks.

**§5.1: the gate cross-multiplies.** `sign(v − Σ/n) = sign(v·n − Σ)`. A block
with a sum head, a count head and its per-token gate can compare against a
mean without a division. The final RMSNorm of the network does the same job a
second way: it turns `(Σ, n)` into a direction. **So a classifier gets the
read port free.** The port buys a *value*, not a decision. (For this reason,
`examples/tally` has no "mean" rung.)

**§5.2: a gap does not move the estimate.** To scale `η` and `Λ` by the same
factor keeps `η/Λ` and decreases only the weight behind it. So the effect of a
discount shows one step later, when new evidence is blended in. A task that
measures it must thus put fresh evidence after the gap, and probe after that.

**§5.3: the size of the drift separation.** On a stream whose probes are at
the edge of the estimate itself, the best geometric-discount arm reaches ~96%
in f64 (and 97.7% in the f32 example sweep of the crate). The filter reaches
100%. The separation is real and small.

**§5.4: a maximum is a sum in the exponential domain.** `Σ exp(S·vₛ)` at decay
one decides exactly "is `v` a new maximum", if `e^S` is more than the sequence
length. It needs no logarithm. So a running maximum does not need the
semiring. It needs the **range** of the semiring: that arm spans
`e^{S·(v_max − v_min)}` (1.5e15 in the setting of the script, against the
~1e7 of f32), where the register spans `S·v`. The dial is the alphabet:
`examples/tally/record` ties at six values and separates at twelve. Every
in-projection channel is an affine read of one embedding, and for this reason
more heads do not buy more digits.

---

## 6. Consequences for this crate

### 6.1 What is structural

`Gain` and `Tropical` are structural, like `Trapezoid` and `RotationKind`:

- A Kalman member allocates `κ` and `ω` per head, one cache slot, and (on the
  projected-noise arm) one in-projection channel per (head, micro-step). The
  output norm keeps `ω`, which scales the readout of the SSD against the `D`
  skip.
- A register allocates two channels, a slot and `eₕ`.

Stock is the default. At `κ = 0` / `e = 0` the block gives stock *exactly*,
not approximately.

### 6.2 Where the decay is substituted

The block substitutes the decay inside `helpers::trapezoidal_coefficients`,
between `da` and `α`. That is not only a convenience. The transports of the
trapezoid, the carried decay of the tap slots and the key scale of single-SSD
all read the log-decay. When the block substitutes it where it is *formed*,
they are consistent by construction, not by audit.

### 6.3 What the ladder measured

| task | what it needs | best stock arm found | the positive system |
|---|---|---|---|
| a counter with a floor (`tally-depth`) | growth (§4) | 83% trained, 86.5% swept | **100%** |
| a discount inside the recurrence (`tally-drift`) | §3.7 | 83.8% trained, 90.7–93.0% swept | **100%** (91.8% trained) |
| a running maximum over 12 values (`tally-record`) | range (§5.4) | 90.1% trained, 79.5% swept | **100%** (95.4% trained) |

The trained columns are short runs at an equal budget. So they compare arms,
and they do not find ceilings. The hand-built column is the exactness claim.
Two more numbers are the real content of the ladder, and both are ties:

- The maximum ties at six values: a sufficiently small alphabet fits the
  exponential encoding.
- The drift task ties everywhere except where the estimate really decides:
  98–99% on ordinary streams, against 90.7% on the probe bursts.

A capability that shows only under a family built for it is still a
capability. But the family is necessary. The tables are those of
`examples/tally/README.md`, and the tests of the rungs compute them again.

### 6.4 What does not follow

- **Not a state-tracking claim.** These systems are commutative and scalar.
  They add nothing to what `RotationKind` classifies. A register holds a
  counter, not a group element.
- **Not an argument from optimality.** The Kalman member is the optimal
  filter of a model that the data do not have to follow. §3 shows that the
  block can *be* that filter. §5 shows how much that is worth on a decision
  task.
- **Not yet a training claim.** Everything above is representability plus
  three tiny trained rungs. This note does not test whether a computed decay
  helps a language model at scale. The crate can promise one thing: the join
  is exact. So to answer the question, turn `κ` on.

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
  1960: the filter that §3.1 checks against, in its covariance form.
- A. H. Jazwinski, *Stochastic Processes and Filtering Theory*, 1970: the
  fading-memory filter, which is the scaled member of this note (`q = 0`).
- G. Birkhoff, *Extensions of Jentzsch's theorem*, 1957: the contraction of
  §3.5, through the Hilbert projective metric.
- F. Baccelli, G. Cohen, G. J. Olsder, J.-P. Quadrat, *Synchronization and
  Linearity*, 1992: (max, +) linear systems. The members of §4 are theirs.
- L. Farina, S. Rinaldi, *Positive Linear Systems*, 2000: the class of both
  members.
- Prior art on the *scaled* member, which several architectures already build
  per key channel:
  - ABC (Peng et al., 2022) and LightNet (2024) form a cumulative
    log-normaliser and decay by its increment.
  - RWKV-4's WKV is (from memory, unverified) the same object with a
    current-token bonus.

  This note adds the **added** member (feedback on confidence) and the
  tropical readout. A full literature check is still necessary before any
  novelty claim beyond that.
