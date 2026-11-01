# 09: Follow the examples renames of burn-stack

Status: ready-for-agent

Blocked by: None (can start immediately)

**What to build:** burn-mamba builds against the burn-stack commit of its
ticket 09 (crate version 0.36.0). That commit renamed the identifiers and
CLI flags of the example scaffolding that conflict with the glossary of
burn-stack. The ticket is
`../burn-stack/.scratch/glossary-followups/issues/09-examples-renames.md`.
burn-mamba does not compile against that rev until it makes the same
renames. One commit before it, `189fc71`, changed the prose of the example
scaffolding, the CLI help included (burn-stack ticket 05).

## Rename map

| Old (burn-stack) | New |
|---|---|
| flag `--no-graph` | `--eager` |
| `AppArgs::graphs()` | `AppArgs::capture()` |
| `Trainer::step` | `Trainer::train_step` |
| `StepFlags` | `TrainingStepFlags` |
| `LmModel::optim_step` (a trait method) | `LmModel::optimizer_update` |
| `Session::begin_step` | `Session::begin_training_step` |
| `Session::skip_steps` | `Session::skip_training_steps` |
| field `TrainingProgress::step` (key `"step"` of `progress.json`) | `training_step` |
| key `"step"` of `metrics.jsonl` | `"training_step"` |
| `lm::Bucket`, config key `TinyStoriesConfig::bucket` | `lm::Bucketing`, `bucketing` |
| config keys `Bucket::slots`, `Bucket::chars` | `Bucketing::buckets`, `Bucketing::batch_chars` |
| flag `--bucket-chars` | `--batch-chars` |

## Sites

The compiler finds each Rust site. These tracked files use an old name:

- `Cargo.toml`: the two `burn-stack` dependencies, at rev `e40fcf5`.
- `AppArgs::graphs()`: the `training.rs` of `mnist-class`, `mnist-ae`,
  `tally/shared` and of each `reset/` example (`majority`, `rotor`,
  `spinor`, `spinor-product`, `swap`, `quintic`), and the `main.rs` of
  `tiny-stories` and `hnet-stories`.
- `Trainer::step` (`trainer.step(…)`): the same `training.rs` files, except
  those of `tiny-stories` and `hnet-stories`.
- `Session::begin_step`: the same `training.rs` files, except those of
  `mnist-class`, `tiny-stories` and `hnet-stories`.
- `StepFlags`: `examples/tiny-stories/cli.rs`, `examples/hnet-stories/cli.rs`.
- `LmModel::optim_step`: `examples/tiny-stories/training.rs`,
  `examples/hnet-stories/training.rs`.
- `TrainingProgress::step`: `examples/tiny-stories/training.rs`
  (`progress.step + 1`).
- The flag `--no-graph` in docs and help text: the `training.rs`,
  `main.rs`, `cli.rs` and `README.md` files of the examples above, and
  `examples/tiny-stories/inference.rs`.

The prose that follows these names:

- `examples/README.md` (about lines 188 to 262) has a copy of `cli::HELP`.
  It has the wording of burn-stack rev `e40fcf5` ("captured CUDA graphs",
  "the LR-schedule step", "schedule step", `--no-graph`). Ticket 04 left it
  until the pin moves. Copy the help text of the new rev
  (`cargo run --example <name> -- -h` prints it).
- `examples/tiny-stories/contrast.rs` says "the schedule step" for the count
  that is now `TrainingProgress::training_step`. Write "training step". The
  sites are the docs of `ContrastConfig::start`, `weight_at`, and the field
  `step` and `new` of the contrast feed. The field and the parameters `step`
  there are names of burn-mamba. Rename them to `training_step` too, so that
  they match the field of burn-stack.

The experiment directories `examples/*/tmp/` record past runs:

- About 30 scripts read the key `"step"` of `metrics.jsonl` or of
  `progress.json`. Examples are `agg.py`, `summ.py`, `lrprof.py`,
  `stage.py`, `screen.sh` and `extend.sh` of `hnet-stories/tmp/1m-a/`,
  `tiny-stories/tmp/1m-b/` and `tiny-stories/tmp/1m-cased/`.
- `hnet-stories/tmp/1m-a/gen.py` writes the config key `"bucket"` with
  `"slots"` and `"chars"`. `probe.sh` passes `--bucket-chars`.

After the move, a new run writes `"training_step"`, and a config with
`"bucket"` does not load. The past runs stay as they are, as in tickets 06
and 08. The user decides if the scripts change.

## Method

1. Move the two `burn-stack` dependencies of `Cargo.toml` to the rev of
   burn-stack ticket 09, and update `Cargo.lock`.
2. Use the Edit tool for the renames. `replace_all` is correct for an
   identifier in one file. Edit `step` and `graph` in prose site by site,
   because these words also name the step call and the autodiff graph.
   Review every hit with `rg` afterwards.
3. Bump the crate version in `Cargo.toml`, as tickets 06 and 08 did. Put the
   bump, `Cargo.lock` and the renames in one commit.

## Acceptance criteria

- [ ] `rg -e no-graph -e '\.graphs\(\)' -e '\bStepFlags' -e optim_step
      -e begin_step -e skip_steps -e bucket-chars -e 'CUDA graph'
      -e 'schedule step'` over
      `src`, `examples` (without `tmp/`) and the markdown files finds
      nothing. (`\bStepFlags` has a word boundary, because the new name
      `TrainingStepFlags` holds the old one.)
- [ ] `cargo check`, `cargo test --lib -- --test-threads=1`,
      `cargo test --examples -j 1 -- --test-threads=1` pass (flex).
- [ ] `cargo doc --all --no-deps` has no warnings.

## Comments
