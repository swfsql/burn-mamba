# Reset-Majority

The smallest task that a **single Mamba-2 block actually has to solve** — and
that nothing else in the model can.

The model reads a stream of `+` / `-` / `R` symbols and reports, at every
position, the sign of the running vote **since the last `R`**:

```text
  symbols   + + - + + R - - + - - + R + -
  vote      1 2 1 2 3 0 -1 -2 -1 -2 -3 -2 0 1 0
  target    p p p p p .  n  n  n  n  n  n .  p .
```

Positions where the vote is exactly zero have no sign to report and are not
scored (`.`).

## Usage

```bash
# training and running inference in flex (fp32)
cargo run --release --example reset-majority -- --training --inference

# the claims below, measured
cargo test --release --example reset-majority -- --nocapture
```

- See `burn-mamba/Cargo.toml` for other features or backend information.
- See `burn-mamba/examples/README.md` for the CLI usage overview.

## Why this task

A Mamba-2 block at `d_model = 2`, `state_rank = 1`, `conv_kernel = 1` unrolls to
two data-dependent scalar recurrences and a sign-like readout — see the
`model.rs` docs for the derivation. The task is built to need exactly that, and
nothing around it:

| shortcut | why it is closed |
|---|---|
| read the current symbol | the label is not a function of it |
| read a fixed window | `conv_kernel = 1` — no convolution at all |
| read the residual | `ignore_last_residual` — the head sees the block alone |
| a **fixed** decay | a reset must erase its past outright *and* the votes after it stay unweighted |

That last row is the point, and the eval set pins it down from both sides:

- **`long-prefix`** — a long same-sign run, one `R`, then a majority of one vote
  the other way. Any decay near 1 leaks the buried run through.
- **`long-suffix`** — an early `R`, then `b+1` votes one way followed by `b` the
  other, so the majority is decided by the *oldest* post-reset tokens. Any decay
  away from 1 lets the recent block outvote them.

## Measured

62 parameters. Chance is 50%.

| | random | long-prefix | long-suffix |
|---|---|---|---|
| best per-symbol lookup (no memory at all) | 77.3% | 82.3% | 55.2% |
| best **fixed**-decay block (10 decays × 6 gains) | 85.9% | 82.6% | 98.8% |
| — the same, worst family per decay | \<71% | | |
| **hand-built** selective block, no training | **100%** | **100%** | **100%** |
| trained, 80 epochs | **100%** | **100%** | **100%** |

The row that matters is the third: no fixed decay clears 71% on its worst
family, which is where a model with *no memory whatsoever* already sits. Turning
selectivity on — one channel of the block's `Δ` projection — takes it to 100%.

`tests.rs` produces all of it: `handmade_block_solves_every_family` writes every
weight down in closed form from the unrolled recurrence (no fitting anywhere),
and `no_fixed_decay_solves_the_task` re-runs that same block with `RESET`'s
selectivity switched off, sweeping the decay and the readout gain.

## Notes

- **The selective solution is a basin you have to find.** With a constant LR
  training either reaches it or stalls near the memoryless ceiling depending on
  the init; the cosine schedule (warmup to 3e-2, annealed to 1e-4) gets there on
  most seeds. A run that ends around 80% has stalled — restart it with a
  different `seed` in the training config.
- **Ties are unscored on purpose.** Asking for a third "vote is exactly zero"
  class turns a sign readout into an exact-zero detector; it costs most of the
  accuracy and tests calibration rather than memory.
