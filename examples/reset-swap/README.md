# Reset-Swap

The corollary of [`reset-spinor`](../reset-spinor/README.md), one rung further
up: the smallest task that a **two-sided** (`SO(4)`) Mamba-3 rotation has to
solve, and that the left-isoclinic quaternion one rung below cannot present to
its own readout.

Same shape of stream — two turns and a reset — but the turns are now **swaps**.
The model reads `s` / `t` / `R` and reports, at every position, how three items
are ordered: the running word in the symmetric group `S₃` since the last reset.

```text
  symbols   R   s   t   s   t   R   t   s   s
  order    abc bac bca cba acb abc acb cab bca
  target    0   2   3   5   1   0   1   4   3
```

Classes are the six orders `abc, acb, bac, bca, cab, cba` in that order. Every
position is scored, and every sequence opens with an `R` — the token that writes
the identity into the state.

## Usage

```bash
# training and running inference in flex (fp32)
cargo run --release --example reset-swap -- --training --inference

# the ablations: the same model one and two rungs down
cargo run --release --example reset-swap -- --training --inference -- --rotation quaternion
cargo run --release --example reset-swap -- --training --inference -- --rotation complex

# the claims below, measured
cargo test --release --example reset-swap -- --nocapture
```

- See `burn-mamba/Cargo.toml` for other features or backend information.
- See `burn-mamba/examples/README.md` for the CLI usage overview.

## Why this task

The four `reset-*` examples are a ladder, each isolating one thing its block has
that the rung below does not:

| example | what only that block can do | the state it needs |
|---|---|---|
| `reset-majority` | forget on command | a **selective decay** |
| `reset-rotor` | count modulo `k` | a **complex** transition (a rotation) |
| `reset-spinor` | compose in a non-abelian group | a **quaternion** transition |
| `reset-swap` | hold a group with more than one involution | a **two-sided** (`SO(4)`) transition |

`S₃` is the smallest non-abelian group, so everything `reset-spinor` argues
applies here unchanged: the label is not a function of the current symbol, it is
periodic in each generator, and it is not a function of how many `s`s and `t`s
went by, because `st ≠ ts`. What makes it the *next* rung is one extra fact:

> `S₃` has **three** elements of order two — the three swaps. Every finite
> subgroup of `SU(2)` has exactly **one**: `−1`.

So `S₃` does not embed in `SU(2)` at all, and the only homomorphism `S₃ → SU(2)`
sends every odd permutation to `−1` — the sign character, and nothing more. A
left-isoclinic transition has two ways to respond and neither works:

| what the `Quaternion4D` block can do | why it is not enough |
|---|---|
| be a homomorphic image of the word | the image is `{±1}`: the parity, which the counts already give |
| track the **double cover** `2D₃` instead | it works, and puts the answer in the state — as `±W`, two **antipodal** vectors sharing one label, which no linear readout can merge |

Two-sided, the block reaches conjugation `v ↦ q v q̄` — that is `SO(3) ⊂ SO(4)`,
where `±q` act *identically*, the double cover collapses, and the three swaps are
three honest half-turns about three axes `60°` apart. The group itself is then
the state, and a linear head reads it straight off.

## Measured

378 parameters (318 for the left-isoclinic twin — the rotor projects a left and
a right axis per head, twelve channels each against six). Chance is 16.7%.

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

The last three rows are the same model, the same data and the same schedule,
with one enum changed: the two-sided run is exact on all three families
(`16384/16384` each, every permutation at 100%, and already exact by epoch 40),
and neither of the other two is close. Note how little the middle rung buys over
the bottom one — 80/50/77 against 79/46/72 — which is what the theory says it
should be: on `S₃`, `SU(2)`'s only homomorphic image is the sign character, and
the counts already contain it.

The trained rows clear the `(#s, #t)` ceiling on `random` and `runs` because a
*trained* block need not write only at the reset: writing at every token lets a
decaying trace carry recency, which pins short words down without any group
structure at all. `shuffle` is the family that closes that escape — one reset,
then a long word whose counts are fixed by construction — and there the two
ablations sit at 50% and 46% while the two-sided block is exact.

The middle three rows are the finding. The left-isoclinic state is not
information-poor — a lookup table over a fine partition of its output recovers
the permutation *exactly*, because `2D₃` has twelve elements and the map from
words to them is injective. It is the **linear** column that collapses, and it
collapses for a reason the same table makes visible: averaged over the positions
carrying one label, that state cancels to nothing. `‖mean output‖ / rms output`
per class comes out at **0.184 / 0.057 / 0.016** for the three families, against
**1.000** for the two-sided block. The two lifts `±W` occur about equally often,
so every linear functional of the state averages to zero on each class.

`tests.rs` produces every row (the training runs below are separate).
`handmade_rotor_solves_every_family` writes every weight down in closed form (no
fitting anywhere); `left_isoclinic_carries_a_double_cover` rebuilds the same
block with `RotationKind::Quaternion4D` and reports it four ways — through the
identical head, through the best linear readout, through the best table, and by
the cancellation statistic; `counts_and_parity_ceilings` needs no model at all;
`labels_are_the_symmetric_group` and `the_lift_of_a_swap_squares_to_minus_one`
check the dataset really is the `S₃` word problem and that the obstruction is
what it is claimed to be (`q² = −1`, yet `q v q̄` squares to the identity).

Every lookup table and linear probe here is **fitted on one split and scored on
another**, so the ceilings are what a model could actually reach, not memorised
answer keys.

## Notes

- **Conjugation is the point, not "more parameters".** The hand-built solution
  ties the right factor to the left (`p = q`) and uses no other freedom of
  `SO(4)`. What it buys is not reach but *quotienting*: `q` and `−q` are two
  lifts of one rotation, and conjugation cannot tell them apart. That is exactly
  the ambiguity a left-multiplying state is stuck with.
- **Half-turns sit in the interior**, as one rung down. A transposition has order
  two, so its rotation is a `180°` turn; with the default `rotation_range = 2`
  that is `tanh(‖ϑ‖) = 1/2`, where the gradient is alive rather than on `tanh`'s
  asymptote. It is the same `TURN_RAW` constant `reset-spinor` uses.
- **Why `60°`.** The two axes must be a *third* of a turn apart: two half-turns
  about axes `θ` apart compose to a rotation by `2θ`, and `s∘t` has order 3, so
  `2θ = 120°`. Getting this wrong gives a different (usually infinite) group, and
  the dataset test would catch it.
- **The real axis is dead weight.** Conjugation fixes it, so head 0 reports a
  constant and the useful state is three-dimensional. `state_rank = 4` is not
  slack — it is the smallest block a quaternion rotation acts on, and `SO(3)`
  arrives inside it as the part that moves.
- **The extra factor is not always free.** In `reset-spinor` the same `Rotor4D`
  block trains *worse* than the quaternion one, because `Q₈` wants `p ≡ 1`
  exactly and a spurious right rotation compounds with the word. Here the task
  wants the right factor — tied to the left, as conjugation — and the same
  block, schedule and optimiser reach 100% by epoch 40. Extra capacity pays where
  the task asks for it and costs where it does not; both rungs are worth reading
  together.
- **This is the one rung whose ablation is not about lost information.**
  `reset-rotor`'s real state and `reset-spinor`'s abelian state genuinely cannot
  represent their targets; here the left-isoclinic state *can*, and the wall is
  the readout. That makes it the sharper statement about the block: a Mamba
  block's output is linear in its state, so a state that needs unfolding is a
  state the block cannot report.
