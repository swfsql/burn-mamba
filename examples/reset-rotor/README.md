# Reset-Rotor

The corollary of [`reset-majority`](../reset-majority/README.md), one family up:
the smallest task that a **single Mamba-3 block actually has to solve** — and
that a Mamba-2 block, at any size, cannot.

Same stream, read differently. The model sees `+` / `-` / `R` and reports, at
every position, where a **three-detent rotor** stands: the running turn count
since the last `R`, taken **mod 3**.

```text
  symbols   R  +  +  +  -  +  +  R  -  -  -  -  +
  turns     0  1  2  3  2  3  4  0 -1 -2 -3 -4 -3
  detent    0  1  2  0  2  0  1  0  2  1  0  2  0
```

Every position is scored. Every sequence opens with an `R` — the token that
anchors the rotor, and (see `model.rs`) the one that gives the block its phase
reference.

## Usage

```bash
# training and running inference in flex (fp32)
cargo run --release --example reset-rotor -- --training --inference

# the claims below, measured
cargo test --release --example reset-rotor -- --nocapture
```

- See `burn-mamba/Cargo.toml` for other features or backend information.
- See `burn-mamba/examples/README.md` for the CLI usage overview.

## Why this task

`reset-majority` isolates the one thing a Mamba-2 block has that a linear SSM
does not: a **selective decay**. This isolates the one thing a Mamba-3 block has
that a Mamba-2 block does not: a **complex transition** — the data-dependent
rotation that Mamba-3 absorbs into `B`/`C` (the "RoPE trick"). At `d_model = 2`,
`state_rank = 2`, `per_head_dim = 1` the block unrolls to two heads sharing one
rotating pair, and the task is built to need exactly that:

| shortcut | why it is closed |
|---|---|
| read the current symbol | the label is not a function of it |
| read a fixed window | Mamba-3 has no short convolution at all |
| read the residual | `ignore_last_residual` — the head sees the block alone |
| hold the turn count in a **real** state | the label is *periodic* in the count, and a linear readout cuts the count axis into three intervals at most |
| a **fixed** rotation (vanilla RoPE) | its phase measures *positions* since the reset, not turns |

The last two rows are the point, and the eval set pins them down from both
sides:

- **`drift`** — one reset, then a strongly biased walk: the turn count runs out
  to `±31`, sweeping the detents over and over. Holding that count is easy; no
  three-interval readout of it can report a residue that alternates across
  sixty-odd values.
- **`balanced`** — one reset, then a shuffled bag of equally many `+` and `-`:
  the count stays inside `±9` but its order is random, so nothing keyed to the
  position since the reset predicts it.

`random` (resets at ~⅛ of positions) is the family where both shortcuts are
partly available — after a reset, "three steps in" almost gives the answer —
which is why it is reported separately rather than averaged in.

## Measured

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
adversarial families — barely above the memoryless table, which is where a model
with no memory whatsoever already sits. Turning the rotation on, and letting one
in-projection channel drive it, takes the same block to 100%.

`tests.rs` produces all of it. `handmade_block_solves_every_family` writes every
weight down in closed form from the unrolled recurrence (no fitting anywhere);
`no_fixed_rotation_solves_the_task` re-runs that block with `ϑ`'s data
dependence removed, sweeping the per-step angle; `no_real_state_solves_the_task`
switches the rotation off entirely (`rope_fraction = 0`, i.e. a real transition)
and sweeps the decay of a block that holds the turn count.

The two ablations do **not** sweep the readout — they are given the best one
they admit, computed exactly: every cut of the scalar channel into three
intervals, every cut of the output plane into three sectors. Only the knob the
ablation leaves free is swept. The `positional` row is the same idea taken to
the limit: the best table lookup from `(symbol, steps since the reset)`, which
upper-bounds *any* input-independent phase — and every decay it could be read
through — without running a model at all.

## Notes

- **The rotor is the state.** `R` writes `B` into the state at whatever phase
  the sequence has reached and wipes what was there (`A(R) ≈ −20`); `±` write
  nothing at all (`x(±) = 0` exactly) and only turn the phase by `±2π/3`. The
  readout is `Cᵀ R(θ_R − θₜ) B`, a function of the rotation accumulated *since
  the write* — which is why the absolute phase never has to be reset, and never
  is: it drifts on forever, folded mod `2π` by `wrap_angle`.
- **Two heads, two axes.** They share `Δ` (hence the same angle) and differ only
  in the per-head bias `c_bias_hmr`, which puts their `C` a quarter turn apart.
  That is what makes `(y₀, y₁) ∝ (cos φ, −sin φ)` — one axis alone could not
  tell `+1` from `−1` detent, since their cosines agree.
- **The reset is inherited, not new.** Erasing on `R` is `reset-majority`'s
  selectivity, and Mamba-3 gets it through the data-dependent `A` rather than
  through `Δ` (which here stays at 1 for every symbol, so the per-step angle is
  `π·tanh(ϑ)` outright).
- **Training finds it** — on every seed tried (0, 1, 2), with the same cosine
  schedule as `reset-majority` (warmup to 3e-2, annealed to 1e-4), and without
  the seed hunting that one needs. Not at the same moment, though: two seeds
  were exact on all three families by epoch ~35, while the third sat at 73–86%
  on `drift` until epoch 50 and then snapped to 100% within five epochs. A run
  short of 100% halfway through has not necessarily stalled — the rotation locks
  onto the detents late.
- **`d_model = 2`, not 1.** With a 2-D token every projection is an independent
  affine functional of the symbol, so `ϑ` can read the turn direction while `x`
  and `A` read the reset flag — which is what makes the closed-form solution
  constructible.
