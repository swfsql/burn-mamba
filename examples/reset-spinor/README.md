# Reset-Spinor

The corollary of [`reset-rotor`](../reset-rotor/README.md), one rung further up:
the smallest task that a **quaternion** Mamba-3 block has to solve, and that the
abelian rotation it ships with cannot.

Same shape of stream, two turns and a reset — but the turns **do not commute**.
The model reads `i` / `j` / `R` and reports, at every position, the running
product in the quaternion group `Q₈` since the last reset:

```text
  symbols   R   i   j   i   j   R   j   i   i
  state     1   i   k  -j  -1   1   j  -k  -i
  target    0   1   3   6   4   0   2   7   5
```

Classes are `1, i, j, k, -1, -i, -j, -k` in that order. Every position is
scored, and every sequence opens with an `R` — the token that writes the
identity into the state.

## Usage

```bash
# training and running inference in flex (fp32)
cargo run --release --example reset-spinor -- --training --inference

# the ablation: the identical model with the abelian rotation
cargo run --release --example reset-spinor -- --training --inference -- --rotation complex

# the claims below, measured
cargo test --release --example reset-spinor -- --nocapture
```

- See `burn-mamba/Cargo.toml` for other features or backend information.
- See `burn-mamba/examples/README.md` for the CLI usage overview.

## Why this task

The three `reset-*` examples are a ladder, each isolating one thing its block
has that the rung below does not:

| example | what only that block can do | the state it needs |
|---|---|---|
| `reset-majority` | forget on command | a **selective decay** |
| `reset-rotor` | count modulo `k` | a **complex** transition (a rotation) |
| `reset-spinor` | compose in a non-abelian group | a **quaternion** transition |

`Q₈` is the smallest non-abelian group of unit quaternions — which is to say the
smallest group that `RotationKind::Quaternion4D`'s state space contains and
`Complex2D`'s does not. The task is built to need exactly that:

| shortcut | why it is closed |
|---|---|
| read the current symbol | the label is not a function of it |
| read a fixed window | Mamba-3 has no short convolution at all |
| read the residual | `ignore_last_residual` — the head sees the block alone |
| hold a count in a **real** state | the label is periodic in each generator (`i⁴ = 1`) — the `reset-rotor` argument |
| an **abelian** rotation | its cumulative rotation is a `cumsum` of angles, i.e. a function of the symbol *counts*, and `ij = k` while `ji = −k` |

That last row is the point. `Q₈`'s commutator subgroup is `{±1}`, so the counts
pin the answer down to a sign and no further: what an abelian state can carry is
exactly the abelianisation `Q₈/{±1} ≅ Z₂×Z₂`, and the missing bit *is* the task.
The eval families put that under different pressure:

- **`shuffle`** — one reset, then a shuffled bag of equally many `i`s and `j`s.
  The counts are fixed by construction, so only the order is left: 91% of
  positions sit on a `(#i, #j)` cell that carries both signs.
- **`runs`** — one reset, then long blocks of one symbol. Still non-commutative,
  but a word of a few blocks is nearly determined by its counts, so this is the
  family every order-blind model does best on.
- **`random`** — resets at ~⅛ of positions, so many words are short, and a short
  word is often pinned by its counts alone.

## Measured

283 parameters (278 for the abelian twin — it spends one fewer in-projection
channel on the rotation). Chance is 12.5%.

| | random | shuffle | runs |
|---|---|---|---|
| best per-symbol lookup (no memory at all) | 34.2% | 17.4% | 18.7% |
| best readout of the **abelian twin's** state | 59.9% | 52.6% | 52.6% |
| best predictor of `(#i, #j)` since the reset | 71.4% | 55.8% | 77.1% |
| **hand-built** quaternion block, no training | **100%** | **100%** | **100%** |
| trained, `--rotation quaternion` | **100%** | **100%** | **100%** |
| trained, `--rotation complex` | 70.1% | 38.3% | 41.7% |

The last two rows are the same model, the same data and the same schedule, with
one enum changed: the quaternion run is exact on all three families
(`16384/16384` each, every element at 100%), and the abelian one is not close.

The middle two rows bound what an order-blind model can do, from two
directions: the abelian twin's state carries the abelianisation and then has to
guess a sign (~50%), while a table given the *exact* counts — far more than a
sum of angles holds — is still stuck wherever both signs occur. Both are
ceilings for a block that writes its state at the reset and reads the rotation
accumulated since, which is what this construction does under either rotation;
the trained `--rotation complex` row is what covers an abelian block free to do
something else entirely.

`tests.rs` produces the first four rows (the last two are the two training
runs above). `handmade_block_solves_every_family` writes every
weight down in closed form (no fitting anywhere); `abelian_rotation_loses_the_order`
rebuilds the same block with `RotationKind::Complex2D` and reports it twice —
through the identical head, and through the best table over a fine partition of
its output space; `counts_ceiling_is_the_abelian_limit` needs no model at all;
`labels_are_the_quaternion_group` checks the dataset really is the `Q₈` word
problem (`ij = k`, `ji = −k`, `i⁴ = 1`).

Every lookup table here is **fitted on one split and scored on another**, so the
ceilings are what a model could actually reach, not memorised answer keys.

## Notes

- **The group is the state.** Left multiplication by a unit quaternion is
  orthogonal, so a state written at step `τ` and read at step `t` gives
  `⟨C, (Qₜ ⊗ Q_τ*) ⊗ B⟩`, and `Qₜ ⊗ Q_τ* = qₜ ⊗ ⋯ ⊗ q_{τ+1}` is the group word
  itself — newest factor on the left, which is the convention the dataset's
  labels use. Writing `B = 1` (the group's unit) makes the readout the four
  components of that element, and the four heads — set apart only by the
  per-head bias `c_bias_hmr` — read one component each.
- **Half-turns are exactly representable.** `i` needs a generator of `π`, i.e.
  `tanh(ϑ) = 1`. `tanh` saturates to `1.0` in f32 well below `ϑ = 20`, so the
  generator is `π` to the last bit and the per-step quaternion is `i` up to
  `cos(π/2) ≈ 4e-8`.
- **The abelian twin is given the better `B`.** With `B = (1,0,0,0)` its second
  rotated pair would multiply zero and it would carry one parity instead of two;
  it gets `(1,0,1,0)` so its state is the *whole* abelianisation. Its "same
  head" column is near chance only because the nearest-element decoder is
  matched to a quaternion, not to a pair of parities — the meaningful number is
  the best-readout column beside it.
- **The trained abelian model beats its hand-built twin — and stops at the
  counts.** On `random` it reaches 70.1% against the 71.4% counts ceiling, well
  above the 59.9% its write-once-and-read twin manages, so training finds
  something the construction does not (writing on every symbol makes the state
  an order-*dependent* sum). It buys nothing on the other two families, and
  nowhere does it clear what a table given the exact counts already gets.
- **Training the quaternion model is a basin, as one rung down.** Accuracy
  climbs to ~99/96/83% by epoch 30, falls back, and only settles at exact around
  epoch 60 — with the same cosine schedule as the other two examples (warmup to
  3e-2, annealed to 1e-4).
- **Nothing here is `reset-rotor`'s job.** The reset (a selective decay) and the
  periodicity (a rotation) are inherited unchanged; the only new requirement is
  that the two turns fail to commute. That is why the ablation is a single enum
  knob rather than a different model.
