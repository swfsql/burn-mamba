# 08: Follow the utils and optim renames of burn-stack

Status: ready-for-agent

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

- [ ] `rg -e ProjSegment -e without_segment -e BlockState -e LrSegment
      -e 'Lr::Sequence' -e class_chunk_plan -e 'utils::graph'
      -e each_slot_alone` over `src`, `examples` (without `tmp/`) and the
      markdown files finds nothing.
- [ ] `cargo check`, `cargo test --lib -- --test-threads=1`,
      `cargo test --examples -j 1 -- --test-threads=1` pass (flex).
- [ ] `cargo doc --all --no-deps` has no warnings.

## Comments
