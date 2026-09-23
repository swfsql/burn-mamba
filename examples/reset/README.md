# The `reset` ladder

Four examples on the **same shape of stream**. Each is the smallest task that
its block is *needed* for, and that the rung below cannot solve. Together they
isolate, one at a time, what each piece of the SSM recurrence buys. A fifth
example keeps the stream and moves the *other* dial, `micro_steps`. A sixth
asks the question of `reset-swap` one group size up, where the answer splits:

| rung | what only that block can do | the state it needs | the group it tracks |
|---|---|---|---|
| [`reset-majority`](#reset-majority) | forget on command | a **real** transition, `RotationKind::Real1D` | — (a sign) |
| [`reset-rotor`](#reset-rotor) | count modulo `k` | a **complex** transition, `RotationKind::Complex2D` | `Z₃` |
| [`reset-spinor`](#reset-spinor) | compose in a non-abelian group | a **quaternion** transition, `Quaternion4D` | `Q₈` |
| [`reset-swap`](#reset-swap) | hold a group with more than one involution | a **two-sided** `SO(4)` transition, `Rotor4D` | `S₃` |
| [`spinor-product`](#spinor-product) | apply **two** group elements in one token | two recurrence steps, `micro_steps = 2` | `Q₈`, two generators per token |
| [`reset-quintic`](#reset-quintic) | hold a **non-solvable** group, and one that no single layer can hold | `Rotor4D`, and for `S₅` a **second layer** | `A₅` and `S₅` |

## The shared shape

Every rung reads a stream of two "turn" symbols plus a reset `R`. At **every**
position, it reports what the turns did since the last `R`. The alphabet and
the readout change, but the skeleton does not. Each rung is one block, at the
smallest size that allows an exact solution, with the residual switched off
(`ignore_last_residual`), so the classification head sees only the block.

The four ladder rungs share three settings:

- **`d_model = 2`**, the floor for a three-symbol alphabet. The pre-`RmsNorm`
  of the layer puts a 1-D token on `±1`, so one dimension carries only two
  symbols. Three points of `ℝ²` are affinely independent, so every
  in-projection channel can take any value on the three symbols.
- **`Trapezoid::None`**. Every construction below pins `λ ≈ 1`, so the `β` tap
  does nothing useful. Switching it off is structural: no `λ` segment in the
  in-projection, no tap slot in the cache, one SSD call instead of two.
- **`nheads = 2`**. A head is a *readout*, not a piece of state: every head
  holds its own copy of the same state, and differs only in the `C` that reads
  that copy. Each of these labels needs two projections.

So what changes up the ladder is the **state**, not the model width:
`state_rank` 1 → 2 → 4 → 4, and the rotation kind with it. Otherwise the rungs
are independent: each takes the smallest setting that *it* allows.
`spinor-product` differs on two counts: it keeps four heads and the trapezoid.
Each `model.rs` records its own floor and what breaks below it.

The three shortcuts that every rung closes:

| shortcut | why it is closed |
|---|---|
| read the current symbol | the label is not a function of it |
| read a fixed window | Mamba-3 has no short convolution at all |
| read the residual | `ignore_last_residual` — the head sees the block alone |

Each rung then closes one more, and *that* is the point of the rung. It is the
last row of each "Why this task" table below.

## Usage

```bash
# training and running inference in flex (fp32) — <rung> ∈ majority|rotor|spinor|swap|quintic
cargo run --release --example reset-<rung> -- --training --inference

# the claims below, measured: the hand-built exact solution and the ablations
cargo test --release --example reset-<rung> -- --nocapture
```

`reset-spinor`, `reset-swap` and `reset-quintic` also take their own
`--rotation` flag, after a second `--`. It selects the rotation of a **fresh**
model config (on reload, the saved config wins). So each of these rungs can
run as its own ablation:

```bash
cargo run --release --example reset-spinor -- --training --inference -- --rotation complex
cargo run --release --example reset-swap   -- --training --inference -- --rotation quaternion
```

The coda is its own target, with `--micro-steps` in place of `--rotation`, plus
`--layers` for the depth contrast (the other way to put two rotations in a token):

```bash
cargo run --release --example spinor-product -- --training --inference
cargo run --release --example spinor-product -- --training --inference -- --micro-steps 1
cargo run --release --example spinor-product -- --training --inference -- --micro-steps 1 --layers 2
```

- See `burn-mamba/Cargo.toml` for other features or backend information.
- See `burn-mamba/examples/README.md` for the CLI usage overview.

## Reading the tables

The "Measured" tables come from the `tests.rs` of each rung. The `trained` rows
are the exception: they come from the training runs above. Two conventions
apply:

- **Every ablation gets the best readout that it accepts.** So the number bounds
  the ablated *architecture*, not one fit of it. From `reset-rotor` up, a search
  over all candidates computes that readout: every cut of a scalar channel into
  intervals, every cut of an output plane into sectors, every table over a fine
  partition. Only the knob that the ablation leaves free is swept. The block of
  `reset-majority` has one scalar channel and one gain, and its test sweeps both
  on a grid.
- From `reset-spinor` up, **every lookup table and linear probe is fitted on one
  split and scored on another**. So the ceilings are what a model could reach,
  not memorised answer keys.

---

## reset-majority

The smallest task that only a **selective decay** can solve. Nothing else in
the model can solve it. Every rung is a Mamba-3 block. In this one the rotation
group is trivial, so the decay is all that the block has.

The model reads `+` / `-` / `R` and reports the sign of the running vote **since
the last `R`**:

```text
  symbols    +  +  -  +  +  R  -  -  +  -  -  +  R  +  -
  vote      +1 +2 +1 +2 +3 +0 -1 -2 -1 -2 -3 -2 +0 +1 +0
  target     p  p  p  p  p  .  n  n  n  n  n  n  .  p  .
```

A position where the vote is exactly zero has no sign to report. The model is
not scored there (`.`).

<details>
<summary>Why this task</summary>

A Mamba-3 block at `d_model = 2`, `state_rank = 1`, `RotationKind::Real1D`
unrolls to two data-dependent scalar recurrences and a sign-like readout.
`model.rs` has the derivation. `Real1D` pairs nothing, so it is the one rotation
kind that accepts an odd `state_rank`. Here the state *is* one real scalar per
head. In addition to the three shared rows:

| shortcut | why it is closed |
|---|---|
| a **fixed** decay | a reset must erase its past fully, *and* the votes after it must keep equal weights |

The eval set tests that from both sides:

- **`long-prefix`**: a long same-sign run, one `R`, then a majority of one vote
  the other way. Any decay near 1 lets the buried run leak through.
- **`long-suffix`**: an early `R`, then a block of votes one way and a block
  one or two votes shorter the other way. So the *oldest* post-reset tokens
  decide the majority. Any decay away from 1 lets the recent block outvote
  them.

</details>

<details> 
<summary>Measured</summary>

64 parameters. Chance is 50%.

| | random | long-prefix | long-suffix |
|---|---|---|---|
| best per-symbol lookup (no memory at all) | 77.3% | 82.3% | 55.2% |
| best **fixed**-decay block (10 decays × 6 gains) | 91.0% | 82.6% | 98.8% |
| — the same, worst family per decay | \<71% | | |
| **hand-built** selective block, no training | **100%** | **100%** | **100%** |
| trained, 80 epochs | **100%** | **100%** | **100%** |

The row that matters is the third. No fixed decay gets more than 71% on its
worst family, and a model with *no memory at all* already gets that much. One
selective channel of the `A` projection of the block takes it to 100%.

`handmade_block_solves_every_family` writes every weight in closed form from the
unrolled recurrence (there is no fitting). `no_fixed_decay_solves_the_task` runs
that same block with the selectivity of `RESET` switched off: `A` is
input-independent, and that is the only changed knob. It sweeps the decay and
the readout gain.

</details>

<details> 
<summary>Notes</summary>

- **The selective solution is a basin that training must find.** With a
  constant LR, training either finds it or stalls near the memoryless ceiling.
  The init decides which. The cosine schedule (warmup to 3e-2, annealed to 1e-4)
  finds it on most seeds. A run that ends near 80% has stalled. Restart it with
  a different `seed` in the training config.
- **Ties are unscored on purpose.** A third "vote is exactly zero" class turns a
  sign readout into an exact-zero detector. That costs most of the accuracy, and
  it tests calibration, not memory.

</details>

---

## reset-rotor

The same argument one rung up: the smallest task that only a **rotating
transition** can solve. A real transition (the rung below, or a Mamba-2 block)
cannot solve it at any size.

The stream is the same, but the model reads it differently. It sees `+` / `-` /
`R` and reports the position of a **three-detent rotor**: the running turn count
since the last `R`, **mod 3**.

```text
  symbols    R  +  +  +  -  +  +  R  -  -  -  -  +
  turns     +0 +1 +2 +3 +2 +3 +4 +0 -1 -2 -3 -4 -3
  detent     0  1  2  0  2  0  1  0  2  1  0  2  0
```

Every position is scored. Every sequence starts with an `R`. That token anchors
the rotor, and it gives the block its phase reference (see `model.rs`).

<details> 
<summary>Why this task</summary>

`reset-majority` isolates the one thing that a selective SSM has and a linear
one does not: a **data-dependent decay**. This rung isolates the one thing that
a rotating transition has and a real one does not: a **complex transition**.
That is the data-dependent rotation that Mamba-3 absorbs into `B`/`C` (the "RoPE
trick"). At `d_model = 2`, `state_rank = 2`, `per_head_dim = 1`, the block
unrolls to two heads that share one rotating pair.

| shortcut | why it is closed |
|---|---|
| hold the turn count in a **real** state | the label is *periodic* in the count, and a linear readout cuts the count axis into three intervals at most |
| a **fixed** rotation (vanilla RoPE) | its phase measures *positions* since the reset, not turns |

The eval set tests these from both sides:

- **`drift`**: one reset, then a strongly biased walk. The turn count goes as
  far as `±31`, and passes each detent many times. It is easy to hold that
  count. But no three-interval readout of it can report a residue that
  alternates across more than sixty values.
- **`balanced`**: one reset, then a shuffled bag of equally many `+` and `-`
  (one extra `+` at an odd length). The count stays within `±10`, but its
  order is random. So nothing that depends on the position since the reset
  predicts it.

In `random` (resets at ~⅛ of positions), both shortcuts partly work: after a
reset, "three steps in" almost gives the answer. So this family is reported
separately and not averaged in.

</details>

<details> 
<summary>Measured</summary>

80 parameters. Chance is 33.3%.

| | random | drift | balanced |
|---|---|---|---|
| best per-symbol lookup (no memory at all) | 49.1% | 36.5% | 37.4% |
| best predictor of (symbol, steps since the reset) | 59.0% | 46.5% | 46.0% |
| best **fixed-rotation** block (12 angles) | 51.8% | 40.7% | 42.1% |
| best **rotation-free** block (7 decays) | 63.8% | 37.7% | 59.3% |
| **hand-built** rotating block, no training | **100%** | **100%** | **100%** |
| trained, 80 epochs | **100%** | **100%** | **100%** |

The two ablation rows are per-family *best cases*. On `drift`, neither gets
more than 41%, only slightly above the memoryless table. The rotation-free
block gets 59% on `balanced`, where the count stays small, but at the same
decays it stays at 38% on `drift`. Switch the rotation on, let one
in-projection channel drive it, and the same block gets 100%.

- `handmade_block_solves_every_family` writes every weight in closed form.
- `no_fixed_rotation_solves_the_task` runs that block without the data
  dependence of `ϑ`, and sweeps the per-step angle.
- `no_real_state_solves_the_task` switches the rotation off fully
  (`RotationKind::Real1D`, a real transition). It sweeps the decay of a block
  that holds the turn count.

The `positional` row takes the ablation idea to its limit: the best table lookup
from `(symbol, steps since the reset)`. It needs no model at all. It bounds every
block that writes its state at the reset and reads what has accumulated since.
Under an input-independent rotation, the phase and the decay of what such a block
reads are functions of exactly that number.

</details>

<details> 
<summary>Notes</summary>

- **The rotor is the state.** `R` writes `B` into the state at the current phase,
  and erases what was there (`A(R) ≈ −20`). `±` write nothing (`x(±) = 0`
  exactly) and only turn the phase by `±2π/3`. The readout is
  `Cᵀ R(θ_R − θₜ) B`, a function of the rotation accumulated *since the write*.
  So the absolute phase never has to be reset, and it never is. It drifts
  forever, folded mod `2π` by `wrap_angle`.
- **Two heads, two axes.** The heads share `Δ` (so the same angle). They differ
  only in the per-head bias `c_bias_hmr`, which puts their `C` a quarter turn
  apart. That gives `(y₀, y₁) ∝ (cos φ, −sin φ)`. One axis alone cannot tell
  detent `+1` from detent `−1`, because their cosines are equal.
- **The reset is inherited, not new.** Erasing on `R` is the selectivity of
  `reset-majority`. Mamba-3 gets it through the data-dependent `A`, not through
  `Δ`. Here `Δ` stays at 1 for every symbol, so the per-step angle is
  `2π·tanh(ϑ)` directly (at the default `rotation_range = 2`).
- **Training finds it** with the cosine schedule of `reset-majority`, and with
  no seed search. The rotation can lock onto the detents late. So a run below
  100% at the halfway mark has not necessarily stalled.
- **`d_model = 2`, not 1.** With a 2-D token, every projection is an independent
  affine functional of the symbol. So `ϑ` can read the turn direction while `x`
  and `A` read the reset flag. That is what makes the closed-form solution
  possible.

</details>

---

## reset-spinor

One rung up: the smallest task that only a **quaternion** Mamba-3 block can
solve. The default abelian rotation of the block cannot solve it.

The stream has the same shape, two turns and a reset. But the turns **do not
commute**. The model reads `i` / `j` / `R` and reports the running product in the
quaternion group `Q₈` since the last reset:

```text
  symbols   R   i   j   i   j   R   j   i   i
  state     1   i  -k   j  -1   1   j   k  -j
  target    0   1   7   2   4   0   2   3   6
```

The classes are `1, i, j, k, -1, -i, -j, -k`, in that order. Every position is
scored. Every sequence starts with an `R`, the token that writes the identity
into the state.

<details> 
<summary>Why this task</summary>

`Q₈` is the smallest non-abelian group of unit quaternions. So it is the
smallest group that the state space of `RotationKind::Quaternion4D` contains and
that of `Complex2D` does not.

| shortcut | why it is closed |
|---|---|
| hold a count in a **real** state | the label is periodic in each generator (`i⁴ = 1`) — the `reset-rotor` argument |
| an **abelian** rotation | its cumulative rotation is a `cumsum` of angles, i.e. a function of the symbol *counts*, and `ij = k` while `ji = −k` |

That last row is the point. The commutator subgroup of `Q₈` is `{±1}`, so the
counts fix the answer only up to a sign. An abelian state can carry exactly the
abelianisation `Q₈/{±1} ≅ Z₂×Z₂`, and the missing bit *is* the task. The eval
families test that with different pressure:

- **`shuffle`**: one reset, then a shuffled bag of equally many `i`s and `j`s
  (one extra `i` at an odd length). The construction fixes the counts, so only
  the order is left. 91% of positions
  are on a `(#i, #j)` cell that carries both signs.
- **`runs`**: one reset, then long blocks of one symbol. The word is still
  non-commutative. But its counts nearly decide a word of a few blocks, so every
  order-blind model does best on this family.
- **`random`**: resets at ~⅛ of positions. So many words are short, and the
  counts alone often decide a short word.

</details>

<details> 
<summary>Measured</summary>

134 parameters. The abelian twin has 122: the quaternion block projects a
rotation axis per head (six channels), and the abelian block projects two
channels that the heads share. Chance is 12.5%.

| | random | shuffle | runs |
|---|---|---|---|
| best per-symbol lookup (no memory at all) | 34.2% | 17.4% | 18.7% |
| best readout of the **abelian twin's** state | 60.6% | 51.9% | 53.5% |
| best predictor of `(#i, #j)` since the reset | 71.4% | 55.8% | 77.1% |
| **hand-built** quaternion block, no training | **100%** | **100%** | **100%** |
| trained, `--rotation quaternion` | **100%** | **100%** | **100%** |
| trained, `--rotation complex` | 71.3% | 53.6% | 58.1% |

The last two rows use the same model, the same data and the same schedule. Only
one enum changes. The quaternion run is exact on all three families
(`16384/16384` each, every element at 100%). The abelian run is far from that.

The two middle rows bound an order-blind model from two directions:

- The twin turns by a half-turn per symbol, the same as the quaternion block.
  So its state carries the abelianisation `Q₈/{±1}`, and it must guess a sign
  (about 50%).
- A table given the *exact* counts does better, because a `cumsum` of angles can
  resolve more of them than parity. But it also fails wherever both signs occur.

Both are ceilings for a block that writes its state at the reset and reads the
rotation accumulated since. This construction does that under either rotation.
The trained `--rotation complex` row is an abelian block that is free to do
something different, so these ceilings do *not* bound it. It gets more than the
counts ceiling on `random`, for the reason that [`reset-swap`](#reset-swap)
gives. The ceilings above it are what bound the architecture.

- `handmade_block_solves_every_family` writes every weight in closed form.
- `abelian_rotation_loses_the_order` builds the same block with
  `RotationKind::Complex2D`. It reports the block twice: through the identical
  head, and through the best table over a fine partition of its output space.
- `counts_ceiling_is_the_abelian_limit` needs no model at all.
- `labels_are_the_quaternion_group` checks that the dataset is the `Q₈` word
  problem (`ij = k`, `ji = −k`, `i⁴ = 1`).

</details>

<details> 
<summary>Notes</summary>

- **The group is the state.** Left multiplication by a unit quaternion is
  orthogonal. So a state written at step `τ` and read at step `t` gives
  `⟨C, (Qₜ ⊗ Q_τ*) ⊗ B⟩`, and `Qₜ ⊗ Q_τ* = qₜ ⊗ ⋯ ⊗ q_{τ+1}` is the group word.
  The newest factor is on the left, as in the labels of the dataset. With
  `B = 1` (the unit of the group), the state is that element. The heads read it,
  and only the per-head bias `c_bias_hmr` makes them different.
- **Two heads, because `nheads` counts readouts and not state.** Every head holds
  its own copy of the same quaternion (same `B`, same `ᾱ`, same rotation). The
  heads differ only in the `C` that reads that copy. So the question is how many
  *projections* the label needs. The answer is two:
  - The eight elements of `Q₈` land on eight directions `45°` apart in a plane
    (`eᵣ ↦` the unit vector at `r·45°`). They are distinct, equidistant and in
    convex position, which is what a linear eight-way head needs.
  - Head `h` reads the `h`-th coordinate of that map. `d_inner = d_model = 2`
    makes `out_proj` the identity, and the head is a sector decoder.

  This is exact at 134 parameters. Four heads would cost 212. `reset-swap` gets
  the same answer from the same argument. `spinor-product` is where it fails.
- **The ceilings above are measured on all four components.** The readout of the
  model is a linear map of the state. So a table over a fine partition of the
  whole 4-vector is at least as good as every readout that the model could have.
  `tests.rs` runs the probe twice, with the two heads on two components each.
- **The abelian twin gets the better `B`.** With `B = (1,0,0,0)`, its second
  rotated pair would multiply zero, and it would carry one parity instead of two.
  It gets `(1,0,1,0)`, so its state is the *whole* abelianisation. Its "same
  head" column is **0%**, and that number means nothing. The sector decoder
  matches a quaternion, not a pair of parities. On this plane, a pair of parities
  can land only on the four odd sectors, and its label is never there. The
  meaningful number is the best-readout column next to it.
- **`shuffle` is the wall, and everything abelian hits it.**
  - The hand-built twin reads parities and gets ~50–60% on every family.
  - The trained model can wire itself in any way. It does better on `random`,
    even more than the 71.4% of a table given the exact counts. A *trained*
    block can write at every token, not only at the reset. Then a decaying trace
    carries recency, and recency identifies short words with no group structure
    at all.
  - `shuffle` closes that escape: one reset, then a long word whose counts the
    construction fixes. There the trained model gets 53.6%, and the quaternion
    block is exact.

  Three mechanisms hit one wall. That wall is the cost of an abelian transition,
  however the block is built.
- **`Rotor4D` is a strict superset, and it runs here too**
  (`-- --rotation rotor`). Left multiplication is *isoclinic*: it turns both
  invariant planes of a 4-block by the same angle. So `Quaternion4D` and
  `Complex2D` are **incomparable**: the quaternion state cannot express two
  independent per-pair angles. The full `SO(4)` kind (two-sided `q ⊗ v ⊗ p̄`,
  where `p = 1` gives this one) contains both, at twice the rotation channels.
  `Q₈` does not need that extra reach: the ladder asks for the *smallest* state
  that solves each rung. And the extra reach has a cost. The exact solution needs
  `p ≡ 1` on the turns, and the same schedule does not find it. In
  [`reset-swap`](#reset-swap), the task needs the second factor.
- **Nothing here is the job of `reset-rotor`.** The reset (a selective decay) and
  the periodicity (a rotation) stay the same. The only new requirement is that
  the two turns do not commute. That is why the ablation is one enum knob and not
  a different model.

</details>

---

## reset-swap

One more rung up: the smallest task that only a **two-sided** (`SO(4)`) Mamba-3
rotation can solve. The left-isoclinic quaternion of the rung below cannot give
the answer to its own readout.

The stream has the same shape, two turns and a reset. But the turns are
**swaps**. The model reads `l` / `r` / `R` and reports the order of three items:
the running word in the symmetric group `S₃` since the last reset.

```text
  symbols   R   l   r   l   r   R   r   l   l
  order    abc bac bca cba cab abc acb cab acb
  target    0   2   4   5   3   0   1   3   1
```

`l` swaps the left pair of positions, and `r` swaps the right pair. The
classes are the six orders `abc, acb, bac, cab, bca, cba`, in that order (the
lexicographic order of the arrays `perm`, where `perm[x]` is the position of
item `x`). Every position is scored. Every sequence starts with an `R`, the
token that writes the identity into the state.

<details> 
<summary>Why this task</summary>

`S₃` is the smallest non-abelian group. So every argument of `reset-spinor`
applies here too:

- The label is not a function of the current symbol.
- It is periodic in each generator.
- It is not a function of the counts of `l` and `r`, because `lr ≠ rl`.

One more fact makes it the *next* rung:

> `S₃` has **three** elements of order two: the three swaps. Every finite
> subgroup of `SU(2)` has exactly **one**: `−1`.

So `S₃` does not embed in `SU(2)`. The only homomorphism `S₃ → SU(2)` sends every
odd permutation to `−1`: it is the sign character, and nothing more. A
left-isoclinic transition has two possible responses, and neither works:

| what the `Quaternion4D` block can do | why it is not enough |
|---|---|
| be a homomorphic image of the word | the image is `{±1}`: the parity, which the counts already give |
| track the **double cover** `2D₃` instead | this puts the answer in the state, but as `±W`: two **antipodal** vectors that share one label, and no linear readout can merge them |

A two-sided block reaches conjugation `v ↦ q v q̄`, which is `SO(3) ⊂ SO(4)`.
There `±q` act *identically*, so the double cover collapses. The three swaps are
then true half-turns about three axes `60°` apart. The group itself is the state,
and a linear head reads it directly.

The eval families are the three of `reset-spinor`, with `l`/`r` for `i`/`j`. One
difference: here `l² = 1`, so a run only alternates, and most of `runs` does
nothing. That is why the counts come closest to deciding `runs`.

</details>

<details> 
<summary>Measured</summary>

146 parameters. The left-isoclinic twin has 128: the rotor projects a left and a
right axis per head (six channels each), and the twin projects one axis (three).
Chance is 16.7%.

| | random | shuffle | runs |
|---|---|---|---|
| best per-symbol lookup (no memory at all) | 40.7% | 22.7% | 29.7% |
| best predictor of the **sign character** since the reset | 49.7% | 37.2% | 46.0% |
| best predictor of `(#l, #r)` since the reset | 68.7% | 47.1% | 76.8% |
| best readout of the **abelian twin's** state | 68.7% | 49.2% | 75.3% |
| best **linear** readout of the **left-isoclinic twin's** state | 55.8% | 53.0% | 26.3% |
| best **table** readout of that same state | **100%** | **100%** | **100%** |
| **hand-built** `Rotor4D` block, no training | **100%** | **100%** | **100%** |
| **trained**, `--rotation rotor` | **100%** | **100%** | **100%** |
| trained, `--rotation quaternion` | 70.9% | 43.4% | 51.0% |
| trained, `--rotation complex` | 75.4% | 45.9% | 70.9% |

The last three rows use the same model, the same data and the same schedule.
Only one enum changes. The two-sided run is exact on all three families
(`16384/16384` each, every permutation at 100%, exact by epoch 20). Neither of
the other two is close. The middle rung buys less than nothing over the bottom
one: 71/43/51 against 75/46/71. The theory predicts that. On `S₃`, the only
homomorphic image in `SU(2)` is the sign character, and the counts already
contain it. So the extra structure only adds something more to get wrong.

The trained abelian row gets more than the `(#l, #r)` ceiling on `random`. A
*trained* block can write at every token, not only at the reset. Then a decaying
trace carries recency, and recency identifies short words with no group
structure at all. `shuffle` closes that escape: one reset, then a long word whose
counts the construction fixes. There the two ablations get 43% and 46%, and the
two-sided block is exact.

The middle three rows are the finding:

- The left-isoclinic state has all the information. A lookup table over a fine
  partition of its output recovers the permutation *exactly*: `2D₃` has twelve
  elements, and the map from words to them is injective.
- The **linear** column collapses, and the same table shows why. Averaged over
  the positions of one label, that state cancels to nothing.
  `‖mean output‖ / rms output` per class is **0.184 / 0.057 / 0.016** for the
  three families, against **1.000** for the two-sided block.
- The two lifts `±W` occur about equally often. So every linear functional of the
  state averages to zero on each class.

The tests:

- `handmade_rotor_solves_every_family` writes every weight in closed form.
- `left_isoclinic_carries_a_double_cover` builds the same block with
  `RotationKind::Quaternion4D`. It reports the block in four ways: through the
  identical head, through the best linear readout, through the best table, and
  with the cancellation statistic.
- `abelian_rotation_loses_the_order` does the same one rung lower.
- `counts_and_parity_ceilings` needs no model at all.
- `left_isoclinic_with_final_norm` shows that the wall is the linear head (see
  the notes).
- `labels_are_the_symmetric_group` checks that the dataset is the `S₃` word
  problem. `the_lift_of_a_swap_squares_to_minus_one` checks the obstruction
  (`q² = −1`, but `q v q̄` squares to the identity).

</details>

<details> 
<summary>Notes</summary>

- **Conjugation is the point, not "more parameters".** The hand-built solution
  ties the right factor to the left (`p = q`) and uses no other freedom of
  `SO(4)`. What it buys is not reach but a *quotient*: `q` and `−q` are two lifts
  of one rotation, and conjugation cannot tell them apart. A left-multiplying
  state cannot resolve exactly that ambiguity.
- **Why `60°`.** The two axes must be a *third* of a turn apart. Two half-turns
  about axes `θ` apart compose to a rotation by `2θ`, and `l∘r` has order 3, so
  `2θ = 120°`. A wrong angle gives a different (usually infinite) group, and the
  dataset test would catch it.
- **The real axis is dead weight.** Conjugation fixes it, so head 0 reports a
  constant, and the useful state is three-dimensional. `state_rank = 4` is not
  slack. It is the smallest block that a quaternion rotation acts on, and `SO(3)`
  is the part of it that moves.
- **The head needs only two of the three axes.** `d_model = 2` (the floor for a
  three-symbol alphabet), so the two heads read the `x` and `y` of the rotation.
  A half-turn about an axis of that plane flips `z`. So `z` carries only the
  parity that the counts already give. The rest is the six orbit points on one
  circle, in six distinct directions, and a nearest-point decoder over them is
  exact. As in `reset-spinor`, `nheads` counts readouts, not state. Both heads
  hold the same rotated vector, and differ only in the `C` that reads it. The
  ceilings above are measured on all four components (`tests.rs` runs the probe
  twice). A table over four components is at least as good as every linear
  readout of them, and the head of the model is such a readout.
- **Two heads are enough here only because the trapezoid is off.** With the `β`
  tap on, the same cut trained to 57 / 39 / 50%. So four heads looked necessary.
  Without the tap, it gets 100% by epoch 20. Remember this when you read an
  ablation: it bounds the configuration that it ran at, not the knob in general.
  In `spinor-product`, two heads really fail (61 / 52 / 53% with the tap,
  46 / 30 / 33% without).
- **The extra factor is not always free.** In `reset-spinor`, the same `Rotor4D`
  block trains *worse* than the quaternion one. `Q₈` needs `p ≡ 1` exactly, and a
  spurious right rotation compounds with the word. Here the task needs the right
  factor, tied to the left as conjugation. The same block, schedule and optimiser
  get 100% by epoch 20. Extra capacity helps where the task needs it, and costs
  where it does not. Read the two rungs together.
- **This is the one rung whose ablation is not about lost information.** The real
  state of `reset-rotor` and the abelian state of `reset-spinor` cannot represent
  their targets. Here the left-isoclinic state *can*, and the wall is the readout.
  That is the sharper statement about the block: the output of a Mamba block is
  linear in its state, so the block cannot report a state that needs unfolding.
- **So `final_norm: false` is necessary.** A final `RmsNorm` over the output
  `(e, y)`, with `e` constant, is even in `y` and merges `±W`. With that norm,
  the same hand-built `Quaternion4D` block is exact on all three families (130
  parameters). With the same weights and no norm, it is near chance
  (10 / 15 / 10%).

</details>

---

## spinor-product

The coda. It is the only example that keeps the transition *group* of the block
and changes `micro_steps` instead. It is the task of
[`reset-spinor`](#reset-spinor), with the stream **packed two symbols per
token**. One recurrence step cannot compose the two symbols, and two steps can.

The alphabet adds the third unit and a hold (`i` / `j` / `k` / `.` / `R`). A
token is an ordered *pair* of them. The target is the running product in `Q₈`
after **both**:

```text
  token       R.      ij      .k      jk      ii      k.
  state      1  1    i -k   -k  1    j -i    1  i    j  j
  target        1       -k       1      -i       i       j
```

A sequence is 32 tokens (64 symbols). Every token is scored. Every sequence
starts with an `R`.

<details>
<summary>Why this task</summary>

The transition of a Mamba-3 step is `α·R`, with `R = exp(ϑ)`. The generator `ϑ`
is a projection of the token: an **affine functional** of it. Here a token is
two one-hot slots, so at `u = 1`:

```text
   ϑ(a, b) = v_a + w_b          generators ADD
   token   = q_b ⊗ q_a          the group MULTIPLIES
```

The two are the same only when the two symbols commute. `MambaProduct`
(`micro_steps`, `burn_mamba::mamba3::product`) runs `u` full recurrence steps
per token. Each step has its own `x`, `B`, `Δ`, `A`, `λ` and rotation. So at
`u = 2`, the transition of the token is `exp(w_b) ⊗ exp(v_a)`: the product
itself.

| shortcut | why it is closed |
|---|---|
| compose the two symbols of the token in **one** step | generators add where the group multiplies, and the alphabet leaves no reparameterisation that avoids it (below) |

Two properties of the alphabet make that exact, not only likely. Both are
necessary:

- **The hold pins the axes.** With `.` in the alphabet, a token can carry one
  turn alone. So `(i, .)` forces `exp(v_i) = i`, and `(., j)` forces
  `exp(w_j) = j`. Inside the bound of the block
  (`‖ϑ‖ < rotation_range·π = 2π`), the only generators with those images are
  along `±x̂` and `±ŷ`, the axes of `i` and `j`. Every sum of them is in the
  `xy`-plane. `exp` of a vector in a plane has **zero** component along the axis
  orthogonal to that plane. The token `(i, j)` needs `j·i = −k`, which is only
  that component. No weights, no scale and no squash can change this: they change
  `‖ϑ‖`, never its direction (`one_step_generators_add_and_cannot_reach_k`).
- **The third unit keeps the pairs non-abelian.** Over `i`/`j` alone, every
  two-turn token composes into `⟨k⟩ ≅ Z₄`, which commutes. Then one rotation per
  token is enough to carry a whole word, and a `micro_steps = 1` model trains to
  98–100% on that two-unit variant. With three units, `(i,j) ↦ k` and
  `(i,k) ↦ j` do not commute, and no abelian reparameterisation exists.

The eval families are those of `reset-spinor`, over the three units, with holds
mixed into every word. The mixture puts single-turn and two-turn tokens next to
each other.

- `shuffle`: one reset, then a shuffled bag of equal `i`/`j`/`k` plus a quarter
  of holds. The counts are fixed and the word is the whole sequence, so only
  composition is left.
- `runs`: blocks of one symbol. Here the counts come closest to deciding.
- `random`: resets at ~1/16 of symbols. These are shorter words, which a decaying
  trace can carry with no group structure.

Length is part of the question here. Inference reports every family at 32
tokens and at three times that. A block that composes each token exactly tracks
the group for as long as it runs. A block that only approximates the composition
adds error with every token.

</details>

<details>
<summary>Measured</summary>

1044 parameters at `u = 2`, 720 at `u = 1` (`u` widens only the per-micro-step
in-projection segments). Chance is 12.5%.

| | random | shuffle | runs |
|---|---|---|---|
| best per-token lookup (no memory at all) | 30.9% | 16.4% | 21.1% |
| best predictor of `(#i, #j, #k)` since the reset | 62.9% | 52.0% | 66.6% |
| the same construction at `u = 1`, same head | 39.6% | 18.5% | 27.6% |
| — the same, best readout of its state | 40.9% | 18.1% | 26.0% |
| **hand-built** `u = 2` block, no training | **100%** | **100%** | **100%** |
| **hand-built** `u = 1` block **handed the pair's product** | **100%** | **100%** | **100%** |
| **trained**, `--micro-steps 2` | **100%** | **100%** | **100%** |
| trained, `--micro-steps 1` | 73.8% | 44.3% | 62.0% |
| trained, `--micro-steps 1 --layers 2` | 43.6% | 29.4% | 35.3% |

The first two trained rows use the same model, the same data and the same
schedule. Only one integer changes. `u = 2` is **exact**: `16384/16384` on every
family, every group element at 100%, by epoch 40. The `u = 1` run gets less than
the counts row on `shuffle` and `runs`. That row is the ceiling for a model that
sees only how many of each symbol went by. That is the cost of "cannot compose
the token": the block uses something that the counts already contain instead.

The same models at **96 tokens**, three times their training length:

| at 96 tokens | random | shuffle | runs |
|---|---|---|---|
| **hand-built** `u = 2` block | **100%** | **100%** | **100%** |
| **hand-built** `u = 1` block **handed the pair's product** | **100%** | **100%** | **100%** |
| **trained**, `--micro-steps 2` | **100%** | **100%** (49150/49152) | **100%** |
| trained, `--micro-steps 1` | 70.7% | 26.2% | 37.4% |

The `u = 2` model extrapolates with two errors in fifty thousand positions,
because it learned the group. The `u = 1` model loses accuracy exactly where the
words get longer.

**The other way to put two rotations in a token: a second layer.**
`--layers 2 --micro-steps 1` has 1280 parameters against the 1044 of `u = 2`, and
two blocks and two states against one. The rotation of a second layer is `exp` of
an affine functional of the layer *below*, not of the token. So the axis pinning
that closes `u = 1` does not apply to it. The second layer only needs the product
of the pair as input (the hand-built rows above are exact at both lengths with
it). The layer below must compute that product, as a learned feature. At the
schedule of the ladder, the trained two-layer model stays below the counts row
on every family.

The two `u = 1` construction rows in the first table are the twin of the
hand-built solution. They sweep the generator scale, and report both through the
identical head and through the best table over a fine partition of the output
space. They land *below* the order-blind ceiling, and that is the point, not a
defect. A wrong rotation is worse than no rotation. Everything else in the
construction (the reset, the write, the decay) folds into a single step
correctly. Only the composition does not.

- `handmade_product_block_solves_every_family` writes every weight in closed
  form (there is no fitting).
- `one_step_cannot_compose_a_token` runs that same construction at
  `micro_steps = 1`.
- `one_step_generators_add_and_cannot_reach_k` is the obstruction above,
  computed over all six ordered pairs of distinct units.
- `a_second_layer_only_helps_by_composing_the_pair` is the hand-built row for the
  depth contrast. It is the same one-step block, and it reads the *effect* of the
  token (its group element, plus whether it resets) instead of its two symbols.
- `counts_ceiling_is_the_order_blind_limit` needs no model at all.
- `labels_are_the_paired_quaternion_word_problem` checks that the dataset is the
  `Q₈` word problem read in pairs.

</details>

<details>
<summary>Notes</summary>

- **The hand-built solution is the one of `reset-spinor`, folded.** Each
  micro-step is a plain Mamba-3 step over the slot that it reads:
  - `R` writes the identity quaternion and erases the state (`A ≈ −20`).
  - `i` / `j` / `k` turn the cumulative rotation by half-turns about the three
    axes.
  - `.` does nothing.
  - The four heads read the four components of the relative quaternion.

  `micro_steps` needed no new construction, because it is not a new kernel. The
  micro-steps fold into the sequence axis, and the recurrence runs at length
  `tokens · u`.
- **The dial is the transition, not the memory.** The state is one
  `[nheads, per_head_dim, state_rank]` matrix at every `u`. The in-projection
  widens (720 → 1044 parameters here), and the recurrence does more work. That
  is the trade of DeltaProduct, and the reason it is a *dial* and not a size.
- **The obstruction is about the token→rotation map, not information.**
  - `u = 1` and `u = 2` can express the *same set* of per-token transitions: a
    scalar times a rotation (`(∏αⱼ)·R₂R₁` is one of each). At `u = 2`, the map
    from the token to that rotation does not have to be `exp ∘ affine`.
  - A trained `u = 1` block has one more freedom than the hand-built twin. The
    pre-`RmsNorm` of the layer divides by a per-token scalar, and a bias then
    moves the sum slightly off the plane. So it can *approximate*, and the
    trained rows are measured, not argued.
  - The approximation is also why the words are 32 tokens long. Over a few
    tokens, a small error per token still scores well (at 16 tokens, a `u = 1`
    model gets 95%). Over a long word, the errors compound into the numbers
    above.

  The exact solution is the only one whose accuracy does not depend on the
  length of the run.
- **Depth is a different resource, and here a worse one.**
  - A second layer also applies a second rotation per token, but to a **second
    state**. The word must end in the state of the last layer. So the per-token
    rotation of that layer must still be the whole product, and the layer below
    must compute it as a *feature*. That feature is a function of the pair with no
    additive form, so a bilinear one, and the `C·B` term of the block is exactly
    that.
  - Nothing forbids that feature. But a learned product is approximate, so it
    can track the group only while the words stay short.
  - `u = 2` costs fewer parameters (1044 vs 1280), has one state instead of two,
    and is exact at every length.

  The dial and the depth are not interchangeable. The dial changes the
  token→transition *map*. Depth adds another map after it.
- **The `RotationKind` decides what `u` buys.** This is the non-abelian case.
  Here `u` gives a per-token transition that the parameterisation cannot express
  otherwise. Under `Complex2D`, the same pairing costs nothing: a sum of angles
  *is* the composite rotation of an abelian group. That is why the two-unit
  version of this task is solvable at `u = 1`. Under `Real1D`, there is no
  rotation to compose, so `u` widens only the write.
- **Two symbols per token, not two tokens.** The stream has the length of
  `reset-spinor` in symbols. Only the packing changes. So this is not a harder
  word problem. It is the same problem, at half the number of recurrence steps of
  a stock block.
- **This is the rung that keeps the trapezoid, and the one that cannot shrink.**
  - The five `reset-*` rungs run at `Trapezoid::None` (their constructions pin
    `λ ≈ 1`, so the `β` tap is dead weight) and at two heads.
  - Here the construction also pins `λ ≈ 1`. Without the tap, it *still* gets
    100% at 32 tokens, but only 83% at 96. The 96-token column is what this rung
    reports.
  - Two heads cost more (61 / 52 / 53% with the tap, 46 / 30 / 33% without).
  - `d_model = 8` is a floor, not a habit. Each slot needs four independent
    indicator channels (`i`, `j`, `k`, `R`, with `.` as the reference). So its
    five symbols must be affinely independent, which takes four dimensions. The
    direction spaces of the two slots must not overlap, or one functional could
    not read a slot alone.

</details>

---

## reset-quintic

The question of the ladder, one size up. The block of `reset-swap` holds `S₃`.
Does a Mamba-3 block hold the symmetric group of **five** items, the group of
the unsolvable quintic? Half of it:

- `A₅` (the sixty even arrangements, the rotation group of the icosahedron) is
  one `Rotor4D` block, exact at the width of `reset-swap`.
- `S₅` is not the transition group of **any** Mamba-3 layer, of any rotation
  kind and any size. It needs a second layer.

The stream has the same shape, two turns and a reset, over five items. It uses
one of two alphabets (`--group`):

```text
  S₅ (the default):  s = (0 1),  c = (0 1 2 3 4)                        120 classes
  symbols  R      s      c      s      c      c      R      c      s
  order    abcde  bacde  ebacd  beacd  dbeac  cdbea  abcde  eabcd  aebcd

  A₅ (--group a5):   d = (0 1)(2 3),  t = (0 2 4),  d∘t of order 5       60 classes
  symbols  R      d      t      d      t      t      R      t      d
  order    abcde  badce  eabcd  aecbd  deabc  cedba  abcde  ebadc  bedac
```

A turn `g` moves the item at position `p` to position `g(p)`. So `s` swaps the
first two positions, and `c` moves every item one position to the right (the
last item goes to the front). `d` swaps the positions `0 ↔ 1` and `2 ↔ 3`, and
`t` moves the items at positions `0, 2, 4` to `2, 4, 0`.

The classes are the elements of the group in the lexicographic order of the
arrays `perm`, where `perm[x]` is the position of item `x` (`abcde` is 0).
Every position is scored. Every sequence starts with an `R`. A sequence is
**96** symbols long, three times the 32 of the ladder (the notes give the
reason).

```bash
cargo run --release --example reset-quintic -- --training --inference                 # S₅, two layers
cargo run --release --example reset-quintic -- --training --inference -- --group a5   # A₅, one layer
cargo run --release --example reset-quintic -- --training --inference -- --group s5 --layers 1
cargo test --release --example reset-quintic -- --nocapture
```

<details>
<summary>Why this task</summary>

**`A₅` is the icosahedron.** `d` is a half-turn about an edge axis. `t` is a
third-turn about a face axis `20.9°` from it (`cos = φ/√3`). Their product is a
fifth-turn about a vertex: `⟨d, t | d² = t³ = (dt)⁵ = 1⟩`, which *is* `A₅`. So
the construction of `reset-swap` applies without change. Conjugation
`v ↦ q v q̄` is `SO(3)` inside a `Rotor4D` 4-block, `R` writes a reference
vector, and the state is its orbit: sixty points instead of six. What changes is
how much less every cheaper state can carry:

| shortcut | why it is closed |
|---|---|
| an **abelian** rotation | `A₅` is perfect (`[A₅, A₅] = A₅`), so its only abelian image is trivial: an abelian state tracks *nothing* of it, not even the sign of `reset-swap` |
| a **stack** of simpler layers | `A₅` is simple and non-abelian, so no cascade of solvable machines reaches it (Krohn–Rhodes): its word problem is NC¹-complete (Barrington) |
| a **left-isoclinic** rotation | the double cover `2I`: the lift of `d` squares to `−1`, every element arrives as `±W`, and a linear head cannot merge antipodes (the wall of `reset-swap` again) |

**`S₅` is not a group of block rotations.** The per-token transition of a layer
is a scalar times a block-diagonal rotation: a product of `SO(4)`s, of which
`Complex2D` and `Quaternion4D` are subgroups. So a layer that tracks a group
exactly, at every length, holds it as a *quotient* of a group of such rotations.
`S₅` is not such a quotient, and one pair of elements shows why:

- `c` and `s∘c∘s` are five-cycles that are conjugate in `S₅` (by `s`) but
  **not** in `A₅`. In the icosahedron, they turn by `72°` and `144°`.
- Conjugation by a rotation never changes an angle. In a 4-block `v ↦ q v p̄`,
  conjugation by another rotation keeps the angles of `q` and of `p`. Every map
  that exchanges them is a reflection.
- So in any group of block rotations, an odd element can act on the `A₅` inside
  only as some element of `A₅` already does. The odd elements of `S₅` do not act
  like that. And the normal subgroups of `S₅` are `1`, `A₅`, `S₅`. So a layer
  follows all of `S₅`, or at most its **sign**.

Size does not change this. No `state_rank`, head count, `mimo_rank` or
`micro_steps` (a product of rotations is a rotation) changes it. It is also
exactly what a reflection buys. In the 4-D standard representation of `S₅`, a
swap *is* a reflection. A Householder transition (as in the delta rule) has that
one-layer route, and a Mamba transition does not.

**Two layers can.** Write `σ = sᵉ ∘ a`, with its sign `e` and its even part `a`.

1. The first layer holds `e` (a half-turn on `s`).
2. Then `a` steps by `a ← sᵉ'∘g∘sᵉ · a`. On `s` this is the identity. On `c`, it
   is `c` (`e = 0`) or `s∘c∘s` (`e = 1`): the `72°`/`144°` pair, selected by the
   sign from the layer below.
3. The second layer reads that sign and holds `a` as the `A₅` block does.

That is the Krohn–Rhodes cascade `S₅ = A₅ ⋊ C₂`, one layer per factor.

</details>

<details>
<summary>Measured</summary>

**`A₅`** — chance 1.67%, at 96 symbols:

| | random | shuffle | runs |
|---|---|---|---|
| best per-symbol lookup (no memory at all) | 30.8% | 4.9% | 11.2% |
| best predictor of `(#d, #t)` since the reset | 57.4% | 11.7% | 41.6% |
| **hand-built** `Rotor4D` block, `reset-swap`'s width (308 params) | **100%** | **100%** | **100%** |
| the same construction, left-isoclinic, same head | 15.7% | 3.1% | 6.2% |
| **trained**, `Rotor4D` (608 params) | **100%** | **100%** | **100%** (49151/49152) |
| trained, `--rotation quaternion` | 66.0% | 14.2% | 43.3% |
| trained, `--rotation complex` | 66.0% | 14.4% | 42.2% |

The same models at **288 symbols**:

| at 288 symbols | random | shuffle | runs |
|---|---|---|---|
| **hand-built** | **100%** | **100%** | **100%** |
| **trained**, `Rotor4D` | **100%** | 99.5% | 99.3% |
| trained, `--rotation quaternion` / `complex` | 64.7 / 64.5% | 5.7 / 5.8% | 16.8 / 18.2% |

**`S₅`** — chance 0.83%, at 96 symbols:

| | random | shuffle | runs |
|---|---|---|---|
| best per-symbol lookup (no memory at all) | 27.9% | 3.5% | 6.8% |
| best predictor of the sign since the reset | 23.4% | 4.1% | 9.6% |
| best predictor of `(#s, #c)` since the reset | 52.2% | 8.6% | 32.5% |
| **hand-built** two-layer stack (902 params, also 100% at 288) | **100%** | **100%** | **100%** |
| trained, two layers (1200 params) | 81.6% | 36.1% | 54.0% |
| trained, `--layers 1` (908 params) | 78.0% | 14.0% | 33.6% |

At 288 symbols, the two trained rows fall to 76.3 / 12.7 / 23.4% and
76.6 / 5.3 / 12.1%. Neither tracks the group. The two-layer model only carries
more recency. The trained rows get more than the counts ceiling on `random` for
the reason of `reset-swap`. A block that writes at every token keeps a decaying
trace of the last few symbols, and that trace identifies short words with no
group at all.

- `handmade_a5_rotor_solves_every_family` and
  `handmade_s5_two_layers_solve_every_family` write every weight in closed form,
  and score both lengths.
- `left_isoclinic_carries_the_binary_icosahedral_group` is the one-enum twin.
- `s5_needs_a_reflection_one_layer_does_not_have` checks the facts of the
  obstruction: `A₅` is simple and perfect, the normal subgroups of `S₅`, the
  `72°`/`144°` pair is conjugate only in `S₅`, and two-sided conjugation keeps
  both angles.
- `counts_and_sign_ceilings` needs no model.
- `labels_are_the_alternating_and_symmetric_groups` and
  `a5_is_the_icosahedral_rotation_group` check the datasets and the embedding.
- `learned_rotations` (ignored, set `QUINTIC_ARTIFACTS` to a run directory)
  reads the angles of a trained first layer from its weights.

</details>

<details>
<summary>Notes</summary>

- **Training finds the group one head at a time, hence the trained width.**
  - Every head carries its own rotation. In every trained `A₅` block inspected,
    the icosahedral rotation (`180.0°` and `120.0°`, axes `20.9°` or `69.1°`
    apart) was on one head. The other heads held something smaller (a `Z₃`, a
    recency trace).
  - In the hand-built block, every head reads the *same* state. A trained block
    gives the class head a single projection of it. At the hand-built width, that
    stays below exact.
  - `mimo_rank` is the cure, not more heads. The ranks of one head share its
    state and its transition. So two channels at rank two give the head that
    finds the group two readouts of it, and the second head is room to miss.
    `model::DEFAULT_WIDTH` is `d_model 4`, two heads, rank 2.
- **The learned solution is not always conjugation.** The learned left and right
  axes are `69.1°` apart on one side and `20.9°` on the other. `q` and `p` run in
  the two Galois-twin copies of the binary icosahedral group, which is the
  *four*-dimensional representation of `A₅`. `(−1, −1)` acts trivially, so the
  double cover still collapses. This is also in `SO(4)` and not in `SU(2)`.
  `Rotor4D` holds both.
- **Why 96 symbols.** Trained on the 32 symbols of the ladder, the same model
  turns by the right angles and holds with `ᾱ ≈ 0.99` per step. Over 32 steps
  that costs nothing. But the state shrinks slowly, and the head (which has
  biases) misreads it on long words. At 96 symbols, training pays for the leak.
  The rest (`ᾱ = 0.991` on `d`) costs half a point at 288. The hand-built head decodes *directions* with no bias, so the floor of
  the block (`ᾱ = e^(−a_floor·Δ)`) never affects it. `REF_POINT` makes the
  shadow of the orbit on the two readouts point in sixty distinct directions.
- **The widths are floors, argued in `model::floor_width`.**
  - `A₅` at `d_model = 2`: that is the floor of the three-symbol alphabet, and
    two readouts are enough.
  - `S₅` needs `d_model = 3`. The second layer must tell `s`, `c` at either sign
    and `R` apart. Its write must be zero except on `R`. Its turn must be zero on
    `s` and take two non-parallel values on the two `c`s. Four points of `ℝ²` are
    affinely dependent, so two dimensions are not enough.
  - The construction also embeds `c` as the zero vector. So after the pre-norm,
    the sign from layer 1 is a pure direction. Without that, the slow decay of
    the sign moves the turns of the second layer away from `72°`/`144°`.
- **`A₅` trains, the quotient does not.**
  - `A₅` is exact at the training length. The left-isoclinic and abelian twins
    get the same numbers, as the double cover and perfectness predict.
  - `S₅` is expressible: the hand-built stack is exact at every length. But no
    trained two-layer run found it. The variations tried (seed, schedule,
    epochs, depth, width, word length) all stalled between the one-layer rows
    and the `A₅` rows.
  - `learned_rotations` shows why on the first layer. It learns `c` as a
    fifth-turn and gives `s` no half-turn. So it tries to hold the group itself,
    which it cannot, instead of the sign that the layer above needs.
  - Both layers start identical, and the sign helps only after the second layer
    reads it. Gradient descent must find the whole cascade at once.

</details>

---

## Notes shared by the rungs

- **Half-turns are in the interior.** `reset-spinor` and `reset-swap` both need a
  generator of order two, a `180°` turn.
  - The block bounds one step to `rotation_range · π · Δ`, with a default of 2.
    That is one full traverse of the rotation group per unit `Δ`. For `SU(2)`,
    that is every element: its period is `4π`, because `q` and `−q` turn the
    state differently.
  - So a half-turn is `tanh(‖ϑ‖) = 1/2`, a point with a live gradient. It is not
    at the asymptote of `tanh`, where the f32 derivative is exactly zero and no
    optimiser can arrive. Both rungs use the same `TURN_RAW` constant.
  - The bound is on the *magnitude* of the generator, so the axis is exactly the
    direction of the projection. The generators are projected per head, so the
    heads do not have to agree on one axis.
- **Each solution is a basin that training must find.** Every rung uses the same
  cosine schedule (warmup to 3e-2, annealed to 1e-4). Accuracy increases, falls,
  and then jumps to exact during the run. So a run below 100% at the halfway
  mark has not necessarily stalled. Only `reset-majority` is sensitive enough to
  the seed to need a restart.
- **Each rung inherits the rungs below.** The reset is always the selective decay
  of `reset-majority`. The periodicity is always the rotation of `reset-rotor`. A
  rung adds exactly one requirement. That is why every ablation is one knob and
  not a different model: an enum for the rungs, an integer for the coda.
