# The `reset` ladder

Four examples on the **same shape of stream**, each the smallest task its block
is *needed* for and that the rung below cannot solve. Read together they isolate,
one at a time, what each piece of the SSM recurrence actually buys — plus a fifth
that keeps the stream and moves the *other* dial, `micro_steps`:

| rung | what only that block can do | the state it needs | the group it tracks |
|---|---|---|---|
| [`reset-majority`](#reset-majority) | forget on command | a **real** transition, `RotationKind::Real1D` | — (a sign) |
| [`reset-rotor`](#reset-rotor) | count modulo `k` | a **complex** transition, `RotationKind::Complex2D` | `Z₃` |
| [`reset-spinor`](#reset-spinor) | compose in a non-abelian group | a **quaternion** transition, `Quaternion4D` | `Q₈` |
| [`reset-swap`](#reset-swap) | hold a group with more than one involution | a **two-sided** `SO(4)` transition, `Rotor4D` | `S₃` |
| [`spinor-product`](#spinor-product) | apply **two** group elements in one token | two recurrence steps, `micro_steps = 2` | `Q₈`, two generators per token |

## The shared shape

Every rung reads a stream of two "turn" symbols plus a reset `R`, and reports at
**every** position what the turns have done since the last `R`. The alphabet and
the readout change; the skeleton does not. Each is one block at the smallest size
that admits an exact solution, with the residual switched off
(`ignore_last_residual`) so the classification head sees the block alone.

The three shortcuts every rung closes:

| shortcut | why it is closed |
|---|---|
| read the current symbol | the label is not a function of it |
| read a fixed window | Mamba-3 has no short convolution at all |
| read the residual | `ignore_last_residual` — the head sees the block alone |

Each rung then closes one more, and *that* is what the rung is about. It is the
last row of each "Why this task" table below.

## Usage

```bash
# training and running inference in flex (fp32) — <rung> ∈ majority|rotor|spinor|swap
cargo run --release --example reset-<rung> -- --training --inference

# the claims below, measured: the hand-built exact solution and the ablations
cargo test --release --example reset-<rung> -- --nocapture
```

All four rungs are one Mamba-3 block; the upper three (whose transition actually
rotates) also take a downstream `--rotation` flag, forwarded after
a second `--`; it selects the rotation baked into a **fresh** model config (a
persisted one wins on reload), so each rung can be run as its own ablation:

```bash
cargo run --release --example reset-spinor -- --training --inference -- --rotation complex
cargo run --release --example reset-swap   -- --training --inference -- --rotation quaternion
```

The coda is its own target, with `--micro-steps` in place of `--rotation`:

```bash
cargo run --release --example spinor-product -- --training --inference
cargo run --release --example spinor-product -- --training --inference -- --micro-steps 1
```

- See `burn-mamba/Cargo.toml` for other features or backend information.
- See `burn-mamba/examples/README.md` for the CLI usage overview.

## Reading the tables

The "Measured" tables come from each rung's `tests.rs`, except the `trained` rows,
which are the training runs above. Two conventions hold:

- **Every ablation is given the best readout it admits**, so the number bounds the
  ablated *architecture* rather than one fitting of it. From `reset-rotor` up that
  readout is computed exhaustively — every cut of a scalar channel into intervals,
  every cut of an output plane into sectors, every table over a fine partition —
  and only the knob the ablation leaves free is swept. `reset-majority`, whose
  block has one scalar channel and one gain, sweeps both on a grid.
- From `reset-spinor` up, **every lookup table and linear probe is fitted on one
  split and scored on another**, so the ceilings are what a model could actually
  reach, not memorised answer keys.

---

## reset-majority

The smallest task that a **selective decay actually has to solve** — and that
nothing else in the model can. All four rungs are one Mamba-3 block; this is the
one whose rotation group is trivial, so the decay is all it has.

The model reads `+` / `-` / `R` and reports the sign of the running vote **since
the last `R`**:

```text
  symbols    +  +  -  +  +  R  -  -  +  -  -  +  R  +  -
  vote      +1 +2 +1 +2 +3 +0 -1 -2 -1 -2 -3 -2 +0 +1 +0
  target     p  p  p  p  p  .  n  n  n  n  n  n  .  p  .
```

Positions where the vote is exactly zero have no sign to report and are not
scored (`.`).

<details>
<summary>Why this task</summary>

A Mamba-3 block at `d_model = 2`, `state_rank = 1`, `RotationKind::Real1D`
unrolls to two data-dependent scalar recurrences and a sign-like readout — see
`model.rs` for the derivation. `Real1D` pairs nothing, so it is the one rotation
kind that admits an odd `state_rank`: here the state *is* one real scalar per
head. Beyond the three shared rows:

| shortcut | why it is closed |
|---|---|
| a **fixed** decay | a reset must erase its past outright *and* the votes after it stay unweighted |

The eval set pins that down from both sides:

- **`long-prefix`** — a long same-sign run, one `R`, then a majority of one vote
  the other way. Any decay near 1 leaks the buried run through.
- **`long-suffix`** — an early `R`, then `b+1` votes one way followed by `b` the
  other, so the majority is decided by the *oldest* post-reset tokens. Any decay
  away from 1 lets the recent block outvote them.

</details>

<details> 
<summary>Measured</summary>

70 parameters. Chance is 50%.

| | random | long-prefix | long-suffix |
|---|---|---|---|
| best per-symbol lookup (no memory at all) | 77.3% | 82.3% | 55.2% |
| best **fixed**-decay block (10 decays × 6 gains) | 91.0% | 82.6% | 98.8% |
| — the same, worst family per decay | \<71% | | |
| **hand-built** selective block, no training | **100%** | **100%** | **100%** |
| trained, 80 epochs | **100%** | **100%** | **100%** |

The row that matters is the third: no fixed decay clears 71% on its worst family,
which is where a model with *no memory whatsoever* already sits. Turning
selectivity on — one channel of the block's `A` projection — takes it to 100%.

`handmade_block_solves_every_family` writes every weight down in closed form from
the unrolled recurrence (no fitting anywhere), and `no_fixed_decay_solves_the_task`
re-runs that same block with `RESET`'s selectivity switched off — `A` made
input-independent, the one changed knob — sweeping the decay and the readout gain.

</details>

<details> 
<summary>Notes</summary>

- **The selective solution is a basin you have to find.** With a constant LR
  training either reaches it or stalls near the memoryless ceiling depending on the
  init; the cosine schedule (warmup to 3e-2, annealed to 1e-4) gets there on most
  seeds. A run that ends around 80% has stalled — restart it with a different
  `seed` in the training config.
- **Ties are unscored on purpose.** Asking for a third "vote is exactly zero" class
  turns a sign readout into an exact-zero detector; it costs most of the accuracy
  and tests calibration rather than memory.

</details>

---

## reset-rotor

The corollary one rung up: the smallest task that a **rotating transition
actually has to solve** — and that a real one (the rung below, or a Mamba-2 block)
cannot, at any size.

Same stream, read differently. The model sees `+` / `-` / `R` and reports where a
**three-detent rotor** stands: the running turn count since the last `R`, taken
**mod 3**.

```text
  symbols    R  +  +  +  -  +  +  R  -  -  -  -  +
  turns     +0 +1 +2 +3 +2 +3 +4 +0 -1 -2 -3 -4 -3
  detent     0  1  2  0  2  0  1  0  2  1  0  2  0
```

Every position is scored. Every sequence opens with an `R` — the token that anchors
the rotor, and (see `model.rs`) the one that gives the block its phase reference.

<details> 
<summary>Why this task</summary>

`reset-majority` isolates the one thing a selective SSM has that a linear one does
not: a **data-dependent decay**. This isolates the one thing a rotating transition
has that a real one does not: a **complex transition** — the data-dependent rotation that
Mamba-3 absorbs into `B`/`C` (the "RoPE trick"). At `d_model = 2`, `state_rank = 2`,
`per_head_dim = 1` the block unrolls to two heads sharing one rotating pair.

| shortcut | why it is closed |
|---|---|
| hold the turn count in a **real** state | the label is *periodic* in the count, and a linear readout cuts the count axis into three intervals at most |
| a **fixed** rotation (vanilla RoPE) | its phase measures *positions* since the reset, not turns |

The eval set pins those down from both sides:

- **`drift`** — one reset, then a strongly biased walk: the turn count runs out to
  `±31`, sweeping the detents over and over. Holding that count is easy; no
  three-interval readout of it can report a residue that alternates across
  sixty-odd values.
- **`balanced`** — one reset, then a shuffled bag of equally many `+` and `-`: the
  count stays inside `±9` but its order is random, so nothing keyed to the position
  since the reset predicts it.

`random` (resets at ~⅛ of positions) is the family where both shortcuts are partly
available — after a reset, "three steps in" almost gives the answer — which is why
it is reported separately rather than averaged in.

</details>

<details> 
<summary>Measured</summary>

86 parameters. Chance is 33.3%.

| | random | drift | balanced |
|---|---|---|---|
| best per-symbol lookup (no memory at all) | 49.1% | 36.5% | 37.4% |
| best predictor of (symbol, steps since the reset) | 59.0% | 46.5% | 46.0% |
| best **fixed-rotation** block (12 angles) | 51.8% | 40.7% | 42.1% |
| best **rotation-free** block (7 decays) | 59.5% | 37.4% | 41.9% |
| **hand-built** rotating block, no training | **100%** | **100%** | **100%** |
| trained, 80 epochs | **100%** | **100%** | **100%** |

The two ablation rows are per-family *best cases*, and neither clears 43% on the
adversarial families — barely above the memoryless table. Turning the rotation on,
and letting one in-projection channel drive it, takes the same block to 100%.

`handmade_block_solves_every_family` writes every weight down in closed form;
`no_fixed_rotation_solves_the_task` re-runs that block with `ϑ`'s data dependence
removed, sweeping the per-step angle; `no_real_state_solves_the_task` switches the
rotation off entirely (`RotationKind::Real1D`, i.e. a real transition) and sweeps
the decay of a block that holds the turn count.

The `positional` row is the ablation idea taken to the limit: the best table lookup
from `(symbol, steps since the reset)`. It bounds every block that writes its state
at the reset and reads what has accumulated since — under an input-independent
rotation both the phase and the decay of what it reads are functions of exactly
that number — and it needs no model at all.

</details>

<details> 
<summary>Notes</summary>

- **The rotor is the state.** `R` writes `B` into the state at whatever phase the
  sequence has reached and wipes what was there (`A(R) ≈ −20`); `±` write nothing at
  all (`x(±) = 0` exactly) and only turn the phase by `±2π/3`. The readout is
  `Cᵀ R(θ_R − θₜ) B`, a function of the rotation accumulated *since the write* —
  which is why the absolute phase never has to be reset, and never is: it drifts on
  forever, folded mod `2π` by `wrap_angle`.
- **Two heads, two axes.** They share `Δ` (hence the same angle) and differ only in
  the per-head bias `c_bias_hmr`, which puts their `C` a quarter turn apart. That is
  what makes `(y₀, y₁) ∝ (cos φ, −sin φ)` — one axis alone could not tell `+1` from
  `−1` detent, since their cosines agree.
- **The reset is inherited, not new.** Erasing on `R` is `reset-majority`'s
  selectivity, and Mamba-3 gets it through the data-dependent `A` rather than
  through `Δ` (which here stays at 1 for every symbol, so the per-step angle is
  `π·tanh(ϑ)` outright).
- **Training finds it** — on every seed tried (0, 1, 2), with the same cosine
  schedule as `reset-majority`, and without the seed hunting that one needs. Not at
  the same moment, though: two seeds were exact on all three families by epoch ~35,
  while the third sat at 73–86% on `drift` until epoch 50 and then snapped to 100%
  within five epochs. A run short of 100% halfway through has not necessarily
  stalled — the rotation locks onto the detents late.
- **`d_model = 2`, not 1.** With a 2-D token every projection is an independent
  affine functional of the symbol, so `ϑ` can read the turn direction while `x` and
  `A` read the reset flag — which is what makes the closed-form solution
  constructible.

</details>

---

## reset-spinor

One rung further up: the smallest task that a **quaternion** Mamba-3 block has to
solve, and that the abelian rotation it ships with cannot.

Same shape of stream, two turns and a reset — but the turns **do not commute**. The
model reads `i` / `j` / `R` and reports the running product in the quaternion group
`Q₈` since the last reset:

```text
  symbols   R   i   j   i   j   R   j   i   i
  state     1   i   k  -j  -1   1   j  -k  -i
  target    0   1   3   6   4   0   2   7   5
```

Classes are `1, i, j, k, -1, -i, -j, -k` in that order. Every position is scored,
and every sequence opens with an `R` — the token that writes the identity into the
state.

<details> 
<summary>Why this task</summary>

`Q₈` is the smallest non-abelian group of unit quaternions — which is to say the
smallest group that `RotationKind::Quaternion4D`'s state space contains and
`Complex2D`'s does not.

| shortcut | why it is closed |
|---|---|
| hold a count in a **real** state | the label is periodic in each generator (`i⁴ = 1`) — the `reset-rotor` argument |
| an **abelian** rotation | its cumulative rotation is a `cumsum` of angles, i.e. a function of the symbol *counts*, and `ij = k` while `ji = −k` |

That last row is the point. `Q₈`'s commutator subgroup is `{±1}`, so the counts pin
the answer down to a sign and no further: what an abelian state can carry is exactly
the abelianisation `Q₈/{±1} ≅ Z₂×Z₂`, and the missing bit *is* the task. The eval
families put that under different pressure:

- **`shuffle`** — one reset, then a shuffled bag of equally many `i`s and `j`s. The
  counts are fixed by construction, so only the order is left: 91% of positions sit
  on a `(#i, #j)` cell that carries both signs.
- **`runs`** — one reset, then long blocks of one symbol. Still non-commutative, but
  a word of a few blocks is nearly determined by its counts, so this is the family
  every order-blind model does best on.
- **`random`** — resets at ~⅛ of positions, so many words are short, and a short word
  is often pinned by its counts alone.

</details>

<details> 
<summary>Measured</summary>

328 parameters (278 for the abelian twin — the quaternion block projects a rotation
axis per head, twelve channels against the abelian two, which are shared across
heads). Chance is 12.5%.

| | random | shuffle | runs |
|---|---|---|---|
| best per-symbol lookup (no memory at all) | 34.2% | 17.4% | 18.7% |
| best readout of the **abelian twin's** state | 59.8% | 51.8% | 52.7% |
| best predictor of `(#i, #j)` since the reset | 71.4% | 55.8% | 77.1% |
| **hand-built** quaternion block, no training | **100%** | **100%** | **100%** |
| trained, `--rotation quaternion` | **100%** | **100%** | **100%** |
| trained, `--rotation complex` | 64.6% | 35.4% | 39.5% |

The last two rows are the same model, the same data and the same schedule, with one
enum changed: the quaternion run is exact on all three families (`16384/16384` each,
every element at 100%), and the abelian one is not close.

The middle two rows bound what an order-blind model can do, from two directions. The
twin turns by a half-turn per symbol, exactly as the quaternion block does, so its
state carries the abelianisation `Q₈/{±1}` and then has to guess a sign — about 50%.
A table handed the *exact* counts does better, because a `cumsum` of angles can
resolve more of them than parity, but it too is stuck wherever both signs occur.
Both are ceilings for a block that writes its state at the reset and reads the
rotation accumulated since, which is what this construction does under either
rotation; the trained `--rotation complex` row covers an abelian block free to do
something else entirely, and lands between them.

`handmade_block_solves_every_family` writes every weight down in closed form;
`abelian_rotation_loses_the_order` rebuilds the same block with
`RotationKind::Complex2D` and reports it twice — through the identical head, and
through the best table over a fine partition of its output space;
`counts_ceiling_is_the_abelian_limit` needs no model at all;
`labels_are_the_quaternion_group` checks the dataset really is the `Q₈` word problem
(`ij = k`, `ji = −k`, `i⁴ = 1`).

</details>

<details> 
<summary>Notes</summary>

- **The group is the state.** Left multiplication by a unit quaternion is
  orthogonal, so a state written at step `τ` and read at step `t` gives
  `⟨C, (Qₜ ⊗ Q_τ*) ⊗ B⟩`, and `Qₜ ⊗ Q_τ* = qₜ ⊗ ⋯ ⊗ q_{τ+1}` is the group word
  itself — newest factor on the left, which is the convention the dataset's labels
  use. Writing `B = 1` (the group's unit) makes the readout the four components of
  that element, and the four heads — set apart only by the per-head bias
  `c_bias_hmr` — read one component each.
- **The abelian twin is given the better `B`.** With `B = (1,0,0,0)` its second
  rotated pair would multiply zero and it would carry one parity instead of two; it
  gets `(1,0,1,0)` so its state is the *whole* abelianisation. Its "same head" column
  is near chance only because the nearest-element decoder is matched to a quaternion,
  not to a pair of parities — the meaningful number is the best-readout column
  beside it.
- **Everything abelian stays under the counts ceiling.** The hand-built twin reads
  parities and gets ~50–60%; the trained model, free to wire itself any way it likes,
  gets 64.6% on `random` — better, and still under the 71.4% a table given the exact
  counts reaches. Three mechanisms, one wall, and it is not at 100%: that wall is
  what "the transition is abelian" costs, regardless of how the block is put
  together.
- **`Rotor4D` is a strict superset, and runs here too** (`-- --rotation rotor`). Left
  multiplication is *isoclinic* — it turns both invariant planes of a 4-block by the
  same angle — so `Quaternion4D` and `Complex2D` are actually **incomparable**: the
  quaternion state cannot express two independent per-pair angles. The full `SO(4)`
  kind (two-sided `q ⊗ v ⊗ p̄`, `p = 1` recovering this one) contains both, at twice
  the rotation channels. `Q₈` needs none of that extra reach — the ladder's point is
  the *smallest* state that solves each rung — and it costs: the exact solution needs
  `p ≡ 1` on the turns, which the same schedule does not find (see
  [`reset-swap`](#reset-swap), where the second factor is what the task is *for*).
- **Nothing here is `reset-rotor`'s job.** The reset (a selective decay) and the
  periodicity (a rotation) are inherited unchanged; the only new requirement is that
  the two turns fail to commute. That is why the ablation is a single enum knob
  rather than a different model.

</details>

---

## reset-swap

One rung further up again: the smallest task that a **two-sided** (`SO(4)`) Mamba-3
rotation has to solve, and that the left-isoclinic quaternion one rung below cannot
present to its own readout.

Same shape of stream — two turns and a reset — but the turns are now **swaps**. The
model reads `s` / `t` / `R` and reports how three items are ordered: the running word
in the symmetric group `S₃` since the last reset.

```text
  symbols   R   s   t   s   t   R   t   s   s
  order    abc bac bca cba acb abc acb cab bca
  target    0   2   3   5   1   0   1   4   3
```

Classes are the six orders `abc, acb, bac, bca, cab, cba` in that order. Every
position is scored, and every sequence opens with an `R` — the token that writes the
identity into the state.

<details> 
<summary>Why this task</summary>

`S₃` is the smallest non-abelian group, so everything `reset-spinor` argues applies
here unchanged: the label is not a function of the current symbol, it is periodic in
each generator, and it is not a function of how many `s`s and `t`s went by, because
`st ≠ ts`. What makes it the *next* rung is one extra fact:

> `S₃` has **three** elements of order two — the three swaps. Every finite subgroup
> of `SU(2)` has exactly **one**: `−1`.

So `S₃` does not embed in `SU(2)` at all, and the only homomorphism `S₃ → SU(2)`
sends every odd permutation to `−1` — the sign character, and nothing more. A
left-isoclinic transition has two ways to respond and neither works:

| what the `Quaternion4D` block can do | why it is not enough |
|---|---|
| be a homomorphic image of the word | the image is `{±1}`: the parity, which the counts already give |
| track the **double cover** `2D₃` instead | it works, and puts the answer in the state — as `±W`, two **antipodal** vectors sharing one label, which no linear readout can merge |

Two-sided, the block reaches conjugation `v ↦ q v q̄` — that is `SO(3) ⊂ SO(4)`, where
`±q` act *identically*, the double cover collapses, and the three swaps are three
honest half-turns about three axes `60°` apart. The group itself is then the state,
and a linear head reads it straight off.

The eval families are `reset-spinor`'s three, with `s`/`t` for `i`/`j` — except
that here `s² = 1`, so a run only alternates and `runs` is nearly wasted motion,
which is what makes it the family the counts come closest to deciding.

</details>

<details> 
<summary>Measured</summary>

378 parameters (318 for the left-isoclinic twin — the rotor projects a left and a
right axis per head, twelve channels each against six). Chance is 16.7%.

| | random | shuffle | runs |
|---|---|---|---|
| best per-symbol lookup (no memory at all) | 40.7% | 22.7% | 29.7% |
| best predictor of the **sign character** since the reset | 49.7% | 37.2% | 46.0% |
| best predictor of `(#s, #t)` since the reset | 68.7% | 47.1% | 76.8% |
| best readout of the **abelian twin's** state | 68.5% | 48.8% | 76.4% |
| best **linear** readout of the **left-isoclinic twin's** state | 55.8% | 53.0% | 26.3% |
| best **table** readout of that same state | **100%** | **100%** | **100%** |
| **hand-built** `Rotor4D` block, no training | **100%** | **100%** | **100%** |
| **trained**, `--rotation rotor` | **100%** | **100%** | **100%** |
| trained, `--rotation quaternion` | 80.0% | 50.3% | 77.3% |
| trained, `--rotation complex` | 79.0% | 46.5% | 71.7% |

The last three rows are the same model, the same data and the same schedule, with
one enum changed: the two-sided run is exact on all three families (`16384/16384`
each, every permutation at 100%, and already exact by epoch 40), and neither of the
other two is close. Note how little the middle rung buys over the bottom one —
80/50/77 against 79/46/72 — which is what the theory says it should be: on `S₃`,
`SU(2)`'s only homomorphic image is the sign character, and the counts already
contain it.

The trained rows clear the `(#s, #t)` ceiling on `random` and `runs` because a
*trained* block need not write only at the reset: writing at every token lets a
decaying trace carry recency, which pins short words down without any group
structure at all. `shuffle` is the family that closes that escape — one reset, then a
long word whose counts are fixed by construction — and there the two ablations sit at
50% and 46% while the two-sided block is exact.

The middle three rows are the finding. The left-isoclinic state is not
information-poor — a lookup table over a fine partition of its output recovers the
permutation *exactly*, because `2D₃` has twelve elements and the map from words to
them is injective. It is the **linear** column that collapses, and it collapses for a
reason the same table makes visible: averaged over the positions carrying one label,
that state cancels to nothing. `‖mean output‖ / rms output` per class comes out at
**0.184 / 0.057 / 0.016** for the three families, against **1.000** for the two-sided
block. The two lifts `±W` occur about equally often, so every linear functional of
the state averages to zero on each class.

`handmade_rotor_solves_every_family` writes every weight down in closed form;
`left_isoclinic_carries_a_double_cover` rebuilds the same block with
`RotationKind::Quaternion4D` and reports it four ways — through the identical head,
through the best linear readout, through the best table, and by the cancellation
statistic; `abelian_rotation_loses_the_order` does the same one rung further down;
`counts_and_parity_ceilings` needs no model at all;
`labels_are_the_symmetric_group` and `the_lift_of_a_swap_squares_to_minus_one` check
the dataset really is the `S₃` word problem and that the obstruction is what it is
claimed to be (`q² = −1`, yet `q v q̄` squares to the identity).

</details>

<details> 
<summary>Notes</summary>

- **Conjugation is the point, not "more parameters".** The hand-built solution ties
  the right factor to the left (`p = q`) and uses no other freedom of `SO(4)`. What it
  buys is not reach but *quotienting*: `q` and `−q` are two lifts of one rotation, and
  conjugation cannot tell them apart. That is exactly the ambiguity a left-multiplying
  state is stuck with.
- **Why `60°`.** The two axes must be a *third* of a turn apart: two half-turns about
  axes `θ` apart compose to a rotation by `2θ`, and `s∘t` has order 3, so `2θ = 120°`.
  Getting this wrong gives a different (usually infinite) group, and the dataset test
  would catch it.
- **The real axis is dead weight.** Conjugation fixes it, so head 0 reports a constant
  and the useful state is three-dimensional. `state_rank = 4` is not slack — it is the
  smallest block a quaternion rotation acts on, and `SO(3)` arrives inside it as the
  part that moves.
- **The extra factor is not always free.** In `reset-spinor` the same `Rotor4D` block
  trains *worse* than the quaternion one, because `Q₈` wants `p ≡ 1` exactly and a
  spurious right rotation compounds with the word. Here the task wants the right
  factor — tied to the left, as conjugation — and the same block, schedule and
  optimiser reach 100% by epoch 40. Extra capacity pays where the task asks for it and
  costs where it does not; both rungs are worth reading together.
- **This is the one rung whose ablation is not about lost information.**
  `reset-rotor`'s real state and `reset-spinor`'s abelian state genuinely cannot
  represent their targets; here the left-isoclinic state *can*, and the wall is the
  readout. That makes it the sharper statement about the block: a Mamba block's output
  is linear in its state, so a state that needs unfolding is a state the block cannot
  report.

</details>

---

## spinor-product

The coda: the only one that leaves the block's transition *group* alone and moves
`micro_steps` instead. It is [`reset-spinor`](#reset-spinor)'s task with the
stream **packed two symbols per token** — a composition one recurrence step
cannot do and two can.

The alphabet gains the third unit and a hold (`i` / `j` / `k` / `.` / `R`), a
token is an ordered *pair* of them, and the target is the running product in `Q₈`
after **both**:

```text
  token       R.      ij      .k      jk      ii      k.
  state      1  1    i -k   -k  1    j -i    1  i    j  j
  target        1       -k       1      -i       i       j
```

Sequences are 32 tokens (64 symbols); every token is scored, and every sequence
opens with an `R`.

<details>
<summary>Why this task</summary>

A Mamba-3 step's transition is `α·R`, and `R = exp(ϑ)` where the generator `ϑ`
is a projection of the token — an **affine functional** of it. A token here is
two one-hot slots, so at `u = 1`

```text
   ϑ(a, b) = v_a + w_b          generators ADD
   token   = q_b ⊗ q_a          the group MULTIPLIES
```

and those are the same thing only when the two commute. `MambaProduct` runs `u`
full recurrence steps per token (`micro_steps`, `burn_mamba::mamba3::product`),
each with its own `x`, `B`, `Δ`, `A`, `λ` and rotation, so at `u = 2` the token's
transition is `exp(w_b) ⊗ exp(v_a)` — the product itself.

| shortcut | why it is closed |
|---|---|
| compose the token's two symbols in **one** step | generators add where the group multiplies — and the alphabet leaves no way to reparameterise around it (below) |

Two properties of the alphabet make that exact rather than merely likely, and
both are load-bearing:

- **The hold pins the axes.** With `.` in the alphabet a token can carry one
  turn alone, so `(i, .)` forces `exp(v_i) = i` and `(., j)` forces
  `exp(w_j) = j`. The only generators inside the block's bound
  (`‖ϑ‖ < rotation_range·π = 2π`) with those images lie along `±x̂` and `±ŷ` —
  the axes of `i` and `j`. Every sum of them lies in the `xy`-plane, and `exp` of
  a vector in a plane has **zero** component along the axis orthogonal to it.
  The token `(i, j)` needs `j·i = −k`, which is nothing *but* that component. No
  weights, no scale and no squash can bend this: they change `‖ϑ‖`, never its
  direction (`one_step_generators_add_and_cannot_reach_k`).
- **The third unit keeps the pairs non-abelian.** Over `i`/`j` alone every
  two-turn token composes into `⟨k⟩ ≅ Z₄`, which commutes — one rotation per
  token is then enough to carry a whole word, and a `micro_steps = 1` model
  trains to 98–100% on that two-unit variant. With three units, `(i,j) ↦ k` and
  `(i,k) ↦ j` do not commute and no abelian reparameterisation exists.

The eval families are `reset-spinor`'s, over the three units and with holds mixed
into every word — the mixture is what puts single-turn and two-turn tokens side
by side. `shuffle` is one reset then a shuffled bag of equal `i`/`j`/`k` plus a
quarter holds (the counts fixed, the word the whole sequence: nothing in it but
composition); `runs` is blocks of one symbol (where the counts come closest to
deciding); `random` resets at ~1/16 of symbols (shorter words, the ones a
decaying trace can carry without any group structure).

Length is part of the question here, and inference reports every family at 32
tokens and at three times that: a block that composes each token exactly tracks
the group for as long as you run it, while one that only approximates the
composition compounds its error with every token.

</details>

<details>
<summary>Measured</summary>

1044 parameters at `u = 2` (720 at `u = 1` — `u` widens only the per-micro-step
in-projection segments). Chance is 12.5%.

| | random | shuffle | runs |
|---|---|---|---|
| best per-token lookup (no memory at all) | 30.9% | 16.4% | 21.1% |
| best predictor of `(#i, #j, #k)` since the reset | 62.9% | 52.0% | 66.6% |
| the same construction at `u = 1`, same head | 26.7% | 16.3% | 20.9% |
| — the same, best readout of its state | 19.2% | 13.6% | 13.7% |
| **hand-built** `u = 2` block, no training | **100%** | **100%** | **100%** |
| **trained**, `--micro-steps 2` | **100%** | **100%** | **100%** |
| trained, `--micro-steps 1`, seed 0 | 73.8% | 44.3% | 62.0% |
| trained, `--micro-steps 1`, seed 1 | 68.1% | 54.0% | 56.8% |

The trained rows are the same model, the same data and the same schedule with
one integer changed. `u = 2` is **exact** — `16384/16384` on every family, every
group element at 100%, and there by epoch 40 — while neither `u = 1` seed clears
the row three above it, the ceiling for a model that sees only how many of each
symbol went by. That is what "cannot compose the token" costs: the block falls
back on something the counts already contain.

The same models at **96 tokens**, three times the length they were trained on:

| at 96 tokens | random | shuffle | runs |
|---|---|---|---|
| **hand-built** `u = 2` block | **100%** | **100%** | **100%** |
| **trained**, `--micro-steps 2` | **100%** | **100%** (49150/49152) | **100%** |
| trained, `--micro-steps 1`, seed 0 | 70.7% | 26.2% | 37.4% |
| trained, `--micro-steps 1`, seed 1 | 66.3% | 41.5% | 41.2% |

The `u = 2` model extrapolates to two errors in fifty thousand positions, because
what it learned is the group; both `u = 1` models lose ground exactly where the
words get longer.

The two `u = 1` construction rows in the first table are the twin of the
hand-built solution, swept over the generator scale and reported both through the
identical head and through the best table over a fine partition of its output
space. They land *below* the order-blind ceiling, which is the point rather than
a defect: a rotation that is wrong is worse than no rotation at all, and
everything else in the construction — the reset, the write, the decay — folds
into a single step perfectly well. Only the composition does not.

`handmade_product_block_solves_every_family` writes every weight down in closed
form (no fitting anywhere); `one_step_cannot_compose_a_token` re-runs that same
construction at `micro_steps = 1`; `one_step_generators_add_and_cannot_reach_k`
is the obstruction above, computed over all six ordered pairs of distinct units;
`counts_ceiling_is_the_order_blind_limit` needs no model at all;
`labels_are_the_paired_quaternion_word_problem` checks the dataset really is the
`Q₈` word problem read in pairs.

</details>

<details>
<summary>Notes</summary>

- **The hand-built solution is `reset-spinor`'s, folded.** Each micro-step is a
  plain Mamba-3 step over the slot it reads: `R` writes the identity quaternion
  and wipes (`A ≈ −20`), `i` / `j` / `k` turn the cumulative rotation by
  half-turns about the three axes, `.` does nothing, and the four heads read the
  four components of the relative quaternion. `micro_steps` needed no new
  construction, because it is not a new kernel — the micro-steps are folded into
  the sequence axis and the recurrence runs at length `tokens · u`.
- **The dial is the transition, not the memory.** The state is one
  `[nheads, per_head_dim, state_rank]` matrix at every `u`; what widens is the
  in-projection (720 → 1044 parameters here) and the recurrence work. That is
  DeltaProduct's trade, and the reason it is a *dial* rather than a size.
- **The obstruction is about the token→rotation map, not information.** Both
  `u = 1` and `u = 2` can express the *same set* of per-token transitions (a
  scalar times a rotation — `(∏αⱼ)·R₂R₁` is one of each). What `u = 2` changes is
  that the map from token to that rotation stops having to be `exp ∘ affine`.
  A trained `u = 1` block has one more string to pull than the hand-built twin —
  the layer's pre-`RmsNorm` divides by a per-token scalar, and a bias then bends
  the sum slightly off the plane — so it can *approximate*, which is why the
  trained rows are measured rather than argued. Approximating is also why the
  words are 32 tokens long: over a handful of tokens a near-miss per token still
  scores well (at 16 tokens a `u = 1` model reaches 95%), and over a long word it
  compounds into the numbers above. The exact solution is the only one whose
  accuracy does not depend on how long you run it.
- **What `u` buys is decided by the `RotationKind`.** This is the non-abelian
  case, where the gain is a per-token transition the parameterisation otherwise
  cannot name. Under `Complex2D` the same pairing would cost nothing — a sum of
  angles *is* the composite rotation of an abelian group, which is exactly why
  the two-unit version of this task is solvable at `u = 1` — and under `Real1D`
  there is no rotation to compose, so `u` widens only the write.
- **Two symbols per token, not two tokens.** The stream is `reset-spinor`'s
  length in symbols; only the packing changes. So this is not a harder word
  problem — it is the same one, asked at half the number of recurrence steps a
  stock block would take.
- **`step_infinite` is unavailable here**, as it is for every non-abelian kind
  at `u > 1`: the read-to-write relative rotation `P⁻ᵗQⱼPᵗ⁻ⁿ⁻¹` depends on `n`
  alone only if `Qⱼ` commutes with `P`, so the constant-token output is
  almost-periodic rather than convergent. The block asserts rather than
  approximating.

</details>

---

## Notes shared by the rungs

- **Half-turns sit in the interior.** Both `reset-spinor` and `reset-swap` need a
  generator of order two, i.e. a `180°` turn. The block bounds one step to
  `rotation_range · π · Δ`, defaulting to 2 — one full traverse of the rotation group
  per unit `Δ`, which for `SU(2)` is every element (its period is `4π`, because `q`
  and `−q` turn the state differently). So a half-turn is `tanh(‖ϑ‖) = 1/2`, a point
  with a live gradient, rather than `tanh`'s asymptote, where f32's derivative is
  exactly zero and no optimiser could ever arrive. It is the same `TURN_RAW` constant
  in both. The bound is on the generator's *magnitude*, so the axis is exactly the
  direction the projection names — and the generators are projected per head, so the
  heads need not agree on one.
- **Each solution is a basin you have to find.** All five use the same cosine
  schedule (warmup to 3e-2, annealed to 1e-4). Accuracy climbs, falls back, and snaps
  to exact partway through — `reset-rotor`'s slowest seed by epoch 50, `reset-spinor`
  by epoch 55, `reset-swap` and `spinor-product` by epoch 40 — so a run still short of
  100% at the halfway mark has not necessarily stalled. Only `reset-majority` is
  seed-sensitive enough to need a restart.
- **Each rung inherits the ones below.** The reset is always `reset-majority`'s
  selective decay and the periodicity is always `reset-rotor`'s rotation; a rung adds
  exactly one requirement, which is why every ablation is one knob rather than a
  different model — an enum for the rungs, an integer for the coda.
