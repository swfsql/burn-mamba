# 10: Follow the training terms of burn-stack

Status: resolved

Blocked by: `../burn-stack/.scratch/glossary-followups/issues/10-training-terms.md`

**What to build:** burn-mamba builds against the burn-stack commit of its
ticket 10 (crate version 0.37.0). That commit adds three entries to its
glossary: the training step, the optimizer update and the LR schedule. It also
renames fields of the LR schedules. This crate does not compile against that
rev until it makes the same renames.

The prose of this repo also says "schedule" alone for an LR schedule or for a
layer schedule. After burn-stack ticket 10, the glossary of burn-stack has
`_Avoid_: schedule (alone)` under both terms. The prose then writes the full
term.

The user permits the break of saved configs and the move of the pin. The pin
needs the rev on GitHub, and the user pushes it. The help copy in
`examples/README.md` stays, because burn-stack ticket 10 changes no text of
`cli::HELP`.

## Rename map

| Old (burn-stack) | New |
|---|---|
| field `CosineAnnealingLr::total_steps` (a config key) | `total_training_steps` |
| field `CosineAnnealingLr::warmup_steps` (a config key), `with_warmup_steps` | `warmup_training_steps`, `with_warmup_training_steps` |
| field `LinearLr::steps` (a config key) | `LinearLr::training_steps` |

## Code sites

The compiler finds each site of a burn-stack name. These tracked files use an
old name:

- `Cargo.toml`: the two `burn-stack` dependencies, at rev `1915e69`.
- `with_warmup_steps`: the `main.rs` of `mnist-class`, `mnist-ae`,
  `tiny-stories`, `hnet-stories` and of each `reset/` example (`majority`,
  `rotor`, `spinor`, `spinor-product`, `swap`, `quintic`), and
  `examples/tally/shared/mod.rs`.

The local `total_steps` is a name of burn-mamba, and the compiler does not find
it. It holds the argument of `CosineAnnealingLr::new`. Rename it to
`total_training_steps` in the `main.rs` of each `reset/` example and in
`examples/tally/shared/mod.rs`.

The experiment directories `examples/*/tmp/` record past runs. In
`hnet-stories/tmp/1m-a/`, `tiny-stories/tmp/1m-b/` and
`tiny-stories/tmp/1m-cased/`, the scripts `gen.py`, `decay.py`, `lrprof.py`,
`lrseq.py` and `seq.py` write or read the old keys. After the move, a config
with an old key does not load. The past runs stay as they are, as in tickets
06, 08 and 09. The user decides if the scripts change.

## Prose sites

Line numbers are those of commit `5bf01b7`.

"Schedule" for an LR schedule (24 lines). "Cosine schedule" is "cosine LR
schedule", as in the burn-stack test
`the_batch_size_rescales_a_cosine_lr_schedule`.

- `examples/mnist-class/README.md:97,101`
- `examples/reset/README.md:179,293,369,451,539,618,733,759,1051,1079`
- `examples/reset/quintic/main.rs:92`
- `examples/reset/majority/main.rs:64`
- `examples/reset/rotor/main.rs:68`
- `examples/reset/spinor-product/model.rs:53`
- `examples/reset/spinor-product/tests.rs:1147`
- `examples/tiny-stories/main.rs:92,99,101,106,121`
- `examples/hnet-stories/main.rs:98`
- `examples/hnet-stories/README.md:57`

The lines of `examples/reset/README.md` that say "the same schedule" can mean
more than the LR schedule. Read each line with the config of its rung. If a
line means more, write what it means.

"Schedule" for a layer schedule:

- `README.md:89`: "with a configurable schedule".
- `examples/mnist-ae/model.rs:206`: "`StridedStretched` schedule".

The verb "step" for an optimizer update:

- `examples/mnist-class/training.rs:135`: "stepped by a `Trainer`". Write
  "trained by a `Trainer`".
- `src/unified/tests/untied.rs:183`: "is stepped on its own". Write "is updated
  on its own".

## These stay

"Schedule" has other senses here, and these hits do not change:

- The chunk-length schedule: `README.md:293`,
  `info/mamba-3/architecture-deltas.md`, `src/mamba3/ssd_path/tests.rs`.
- The fixed frequencies of a position embedding: `README.md:56`.
- The cleanups of `--max-vram`: `examples/tiny-stories/training.rs:174,178`,
  `examples/hnet-stories/training.rs:115`.
- The weight of the contrast: `examples/tiny-stories/contrast.rs:32`.
- The order of accumulate and mix: `src/unified/tests/multi_gate.rs:482`.
- The order of a scan or of a cut: `src/mamba3/positive/scan.rs:15,249`,
  `src/packing/tests.rs:427`,
  `src/mamba3/single_ssd/ssd/ssd_path/tests.rs:347`,
  `info/kalman/gate-as-positive-system.md:36,119`.
- The past runs in `examples/*/tmp/`.

The verb "step" stays where it means the recurrent call. Examples are
`README.md:181` and `src/unified/tests/class.rs`.

"Chunkwise" stays. The Comments of ticket 09 say that it can mean a window, but
that is not correct. Every hit means the chunkwise SSD. The sites in the
examples name a `forward` that runs this SSD, as the row `forward()` of the
table in `README.md` does. The 256 characters of the prefill are windows, and
ticket 09 fixed that site.

## Method

1. Move the two `burn-stack` dependencies of `Cargo.toml` to the rev of
   burn-stack ticket 10, and update `Cargo.lock`.
2. Use the Edit tool. `replace_all` is correct for an identifier in one file.
   Edit each prose site on its own, because "schedule" has the other senses
   above.
3. Review the hits afterwards:

   ```bash
   rg -n -w schedule -g '!**/tmp/**' -g '!.scratch/**' examples src *.md docs info \
     | rg -v -i -e 'LR[- ]schedules?' -e 'layer schedule'
   ```

4. Bump the crate version to 0.75.0 in `Cargo.toml`, as tickets 06, 08 and 09
   did. Put the bump, `Cargo.lock`, the renames and the prose in one commit.

## Acceptance criteria

- [x] `rg -e total_steps -e warmup_steps -e 'cosine schedule' -e 'stepped by a'`
      over `src`, `examples` (without `tmp/`) and the markdown files finds
      nothing.
- [x] Each hit of the command in step 3 is in [These stay](#these-stay). A hit
      can also be the second line of a wrapped term ("LR" ends the line before).
- [x] The help copy in `examples/README.md` is equal to the output of `-h`.
- [x] `cargo check`, `cargo test --lib -- --test-threads=1`,
      `cargo test --examples -j 1 -- --test-threads=1` pass (flex).
- [x] `cargo doc --all --no-deps` has no warnings.

## Comments

- The pin is rev `9f5e057`, the later commit of burn-stack ticket 10.
- The word for each "schedule" of `examples/reset/` comes from the config of
  its rung. Every rung has the same LR schedule (5120 training steps).
  - "Training config", in the three paragraphs of `README.md` that say that
    only one enum or integer changes. For `reset-spinor` and `reset-swap`, the
    `training_config.json` of the compared runs are equal, byte for byte. The
    `--micro-steps 1` row of `spinor-product` has no saved run at `max_lr` 0.03,
    so `main.rs` is the only evidence for that row.
  - "Training config", also in `spinor-product/model.rs:53`. The list above
    has this line as an LR schedule. But the sentence compares whole runs, and
    `h2-a`, `base-a` and `reset-swap` `d2-a` have equal configs.
  - "LR schedule", in the `p ≡ 1` line of `README.md`. `spinor/tmp/` has no
    `Rotor4D` run, and the sentence does not say that nothing else changes.
  - "LR schedule", in the two "LR schedule of the ladder" lines (`README.md`,
    `spinor-product/tests.rs`). These lines limit the claim to that learning
    rate. `l2-s0` and `l2-s1` (`max_lr` 0.03) stay below the counts row. At
    `max_lr` 0.01, `l2-lr1e-2-s0` gets 100 / 99.99 / 100% at the training
    length, and seeds 1 to 3 stay below the counts row.
- The review found these items, and the user approved these changes:
  - The locals `schedule` of `src/unified/tests/untied.rs` are `layer_schedule`.
    They hold a layer schedule, and `CLAUDE.md` asks that an identifier follows
    a change of its term. [These stay](#these-stay) kept them before the review.
  - The local `iterations_per_epoch` of `mnist-class`, `mnist-ae`,
    `tiny-stories` and `hnet-stories` is `training_steps_per_epoch`, because it
    counts training steps. The trailing comment of the two `mnist` warmups moves
    to its own line, to keep these lines under 100 characters.
  - `examples/reset/README.md` gives the four two-layer seeds at
    `--max-lr 1e-2`.
- The review found these items, and this ticket does not change them:
  - The doc of `mamba3_untied_tail_trains_one_copy_per_application` has one
    sentence of 32 words and a passive clause. This ticket sets its wording.
  - Seven examples repeat the same `CosineAnnealingLr` setup. Each one is its
    own target and reads alone.
  - `examples/tiny-stories/README.md:140` says "training iteration" for one
    batch. A batch there holds many windows, so this is not a training step.
- The user keeps the scripts of `examples/*/tmp/` as they are.
- This ticket has two commits.
- `cargo test --lib -- --test-threads=1`: 356 tests pass.
  `cargo test --examples -j 1 -- --test-threads=1`: 56 tests pass.
