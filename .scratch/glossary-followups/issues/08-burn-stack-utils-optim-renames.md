# 08: Follow the utils and optim renames of burn-stack

Status: resolved

Blocked by: None (can start immediately)

**What to build:** burn-mamba builds against burn-stack rev
`e40fcf54efc261f3d67082bb28667d30185bd7e6` (crate version 0.35.0). That
commit renamed the identifiers of the utils and optim modules that conflict
with the glossary of burn-stack (its ticket 08:
`../burn-stack/.scratch/glossary-followups/issues/08-utils-optim-renames.md`).
burn-mamba does not compile against that rev until it makes the same renames.

## Rename map

| Old (burn-stack) | New |
|---|---|
| `ProjSegment` | `SubProjection` |
| field `ProjSpec::segments` | `sub_projections` |
| `MuonPlan::without_segment` | `without_sub_projection` |
| `BlockState` (`optim::segmented`) | `SubProjectionState` |
| field `SegmentedState::blocks` | `states` |
| `Lr::Sequence` (a config key) | `Lr::Chain` |
| `LrSegment` | `LrLink` |
| `class_chunk_plan` | `class_window_plan` |
| module `utils::graph` | `utils::capture` |

Two burn-stack commits before that rev also change the glossary and the docs:
`f4c12d8` (the prose of utils and optim) and `6896ed1` (the glossary entries
**Recording** and **Captured step**).

## Sites

These tracked files use an old name:

- `Cargo.toml`: the two `burn-stack` dependencies, at rev `d39c57c`.
- `ProjSegment`: `src/mamba1/mamba1.rs`, `src/mamba2/mamba2.rs`,
  `src/mamba3/mamba3.rs`, `src/unified/tests/optim.rs`.
- `ProjSpec::segments`: `src/mamba3/tap_pattern.rs` (two sites),
  `src/mamba3/product/tests.rs`, `src/mamba3/positive/tests.rs`,
  `src/unified/tests/optim.rs`.
- `BlockState`, `SegmentedState::blocks`, `without_segment`:
  `src/unified/tests/optim.rs`.
- `class_chunk_plan`: `src/unified/tests/class.rs`.
- `utils::graph`: a doc link in `src/unified/capture.rs`.

The compiler finds each of them. Other `segments` of burn-mamba (the
`p.segments` of a packed row, `examples/common/long_stories.rs`) are the
**segment** of the glossary. They stay.

The padding tests of burn-mamba copy the pattern of the burn-stack test
`check_each_slot_alone`, which is now `check_each_row_alone`. The copies are
`check_each_slot_alone`, `padded_mamba1_is_each_slot_alone`,
`padded_mamba2_is_each_slot_alone` and `padded_mamba3_is_each_slot_alone`
(`src/padding/tests.rs`). "Slot" names a row there. Rename them to
`each_row_alone`, as ticket 06 did for the H-Net copy.

The experiment directory `examples/hnet-stories/tmp/1m-a/` writes and reads
the key `"Sequence"`: `gen.py`, `lrprof.py`, `lrseq.py`, `decay.py`, and the
`training_config.json` of past runs. After the move, a config with this key
does not load. The past runs are records, so they stay as they are, as in
ticket 06. A new run from these scripts needs the key `"Chain"`. The user
decides if the scripts change.

## Method

1. Move the two `burn-stack` dependencies of `Cargo.toml` to rev
   `e40fcf54efc261f3d67082bb28667d30185bd7e6`, and update `Cargo.lock`.
2. Use the Edit tool for the renames. `replace_all` is correct for an
   identifier in one file. Edit `segments` site by site, because the word
   also names the segments of a packed row. Review every hit with `rg`
   afterwards.
3. Bump the crate version in `Cargo.toml`, as ticket 06 did. Put the bump,
   `Cargo.lock` and the renames in one commit.

## Acceptance criteria

- [x] `rg -e ProjSegment -e without_segment -e BlockState -e LrSegment
      -e 'Lr::Sequence' -e class_chunk_plan -e 'utils::graph'
      -e each_slot_alone` over `src`, `examples` (without `tmp/`) and the
      markdown files finds nothing.
- [x] `cargo check`, `cargo test --lib -- --test-threads=1`,
      `cargo test --examples -j 1 -- --test-threads=1` pass (flex).
- [x] `cargo doc --all --no-deps` has no warnings.

## Comments

- The commit holds the bump of `Cargo.toml` (crate version 0.73.0, made by
  the user), `Cargo.lock` (burn-stack 0.35.0) and the renames.
- `cargo check --all-targets` has no warnings. `cargo test --lib`: 356 tests
  pass, 4 are ignored. The example tests pass.
- The docs and the identifiers of the same concept change with the names:
  - "Segment" and "block" for a part of the weight of a fused projection are
    now "sub-projection". This includes the `muon_projections` of the three
    families, the in-projection docs of `Mamba2` and `Mamba3`, the Muon,
    tap-pattern, product and rotation tests, the docs of `examples/reset/`,
    `examples/mnist-class/README.md` and
    `info/mamba-3/architecture-deltas.md`. The import alias `Seg` is now
    `Sub`.
  - Where the text means the slices of the projected input, it says
    "channels". An example is the doc of `split_positive`.
  - Five tests named a sub-projection "block" or "segment". They are now
    `all_adamw_sub_projections_equal_plain_adamw`,
    `one_muon_sub_projection_equals_plain_muon`,
    `per_sub_projection_muon_differs_from_fused_muon`,
    `sgd_fallback_sub_projections_are_stateless_sgd`
    (`src/unified/tests/optim.rs`) and
    `in_proj_repeats_only_the_per_micro_step_sub_projections`
    (`src/mamba3/product/tests.rs`). The test helper `append_dead_segment`
    (`src/mamba3/tap_pattern.rs`) is now `append_dead_sub_projections`.
  - `src/unified/tests/class.rs` says "window", not "chunk", for a window.
  - `src/padding/tests.rs` says "row" for a row of the batch, and "position"
    or "real token" for a position. Its fifth test is now
    `padded_mamba3_network_closes_each_row_at_its_own_end`.
  - The headers of `src/unified/capture.rs` and of its tests say "captured
    step", not "graph capture".
- The scripts of `examples/hnet-stories/tmp/1m-a/` do not change (the
  decision of the user). `examples/tiny-stories/tmp/1m-b/` and
  `examples/tiny-stories/tmp/1m-cased/` also write the key `"Sequence"`, and
  the `ORIENTATION.md` of `1m-b` names `Lr::Sequence`. These are records too,
  so they do not change. A new run from these scripts needs the key
  `"Chain"`.
- Ticket 04 gets these sites. The list is not complete:
  - "Segment" for a slice of the projected input, which is not a linear map:
    `src/mamba3/helpers.rs`, `src/mamba3/product/mod.rs` (also two panic
    messages) and the forwards of both pathways. They need a term for this
    slice.
  - "Segment" and "chunk" for a window: `src/mamba1/mamba1.rs` ("a sequence in
    segments", "chunked prefill", "a prior segment"),
    `src/unified/tests/layers.rs`, `src/mamba3/rotation/tests.rs`,
    `src/mamba3/product/mod.rs` and `PREFILL_CHUNK` of
    `examples/tiny-stories/inference.rs`.
  - "Graph" for a recording or a captured step: `README.md` ("CUDA graph
    replay") and the training docs of `examples/`. They name the flag
    `--no-graph` and `AppArgs::graphs` of burn-stack, and burn-stack keeps
    these names for now.
