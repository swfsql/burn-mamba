# 06: Follow the H-Net renames of burn-stack

Status: ready-for-agent

Blocked by: None (can start immediately)

**What to build:** burn-mamba builds against burn-stack rev
`d39c57cc289c367990f224b591573ee80b445cc3` (crate version 0.34.0). That
commit renamed the H-Net identifiers that conflict with the glossary of
burn-stack (its ticket 07:
`../burn-stack/.scratch/glossary-followups/issues/07-hnet-renames.md`).
The `hnet-stories` example does not compile against that rev until
burn-mamba makes the same renames.

## Rename map

| Old | New |
|---|---|
| `Routing::boundary_bs` (burn-stack) | `chunk_boundary_bs` |
| `HNetVocabShape::n_start_tokens`, `with_n_start_tokens` (burn-stack) | `n_start_embeddings`, `with_n_start_embeddings` |
| `HNetShape::smooth_block`, `with_smooth_block` (burn-stack) | `smooth_scan_block`, `with_smooth_scan_block` |
| the constant `N_START_TOKENS` (`examples/hnet-stories/model.rs`) | `N_START_EMBEDDINGS` |
| the local `boundary` (`examples/hnet-stories/inference.rs`) | `chunk_boundary` |
| the test `mamba3_padded_slots_match_each_slot_alone` (a copy of the burn-stack test `padded_slots_match_each_slot_alone`) | `mamba3_padded_rows_match_each_row_alone` |
| prose "start rows" for the start embeddings (`examples/hnet-stories/README.md`, the line with `n_start_tokens`) | "start embeddings" (**start embedding** is a term of `../burn-stack/GLOSSARY.md`) |

## Sites

These tracked files use an old name:

- `Cargo.toml`: the two `burn-stack` dependencies, at rev `4383373`.
- `examples/hnet-stories/inference.rs`: `r.boundary_bs`, the local
  `boundary`.
- `examples/hnet-stories/model.rs`: `with_n_start_tokens`, `N_START_TOKENS`.
- `examples/hnet-stories/README.md`: "start rows" and `n_start_tokens`.
- `src/unified/tests/hnet.rs`: the test name.

No tracked file uses `smooth_block`.

Do not touch the experiment directories `examples/*/tmp/` (their `gen.py` and
the `model_config.json` of each run). They use `smooth_block` and
`n_start_tokens`, and they record past runs. A saved config with the old keys
is not compatible. That is correct: no config compatibility, as in ticket 05.

The other H-Net prose ("main network", "inner network", "start rows" outside
the line above) is in the scope of ticket 04. It is not part of this ticket.

## Method

1. Move the two `burn-stack` dependencies of `Cargo.toml` to rev
   `d39c57cc289c367990f224b591573ee80b445cc3`, and update `Cargo.lock`.
2. Use the Edit tool for the renames. `replace_all` is correct for an
   identifier in one file. Review every hit with `rg` afterwards.
3. Edit the prose with the Edit tool, site by site.
4. Bump the crate version in `Cargo.toml`, as ticket 05 did. Put the bump,
   `Cargo.lock` and the renames in one commit.

## Acceptance criteria

- [ ] `rg -e '\bboundary_bs' -e n_start_tokens -e N_START_TOKENS
      -e smooth_block -e padded_slots` over `src`, `examples` (without
      `tmp/`) and the markdown files finds nothing. (The `\b` keeps the new
      name `chunk_boundary_bs` out of the hits.)
- [ ] `cargo check`, `cargo test --lib -- --test-threads=1`,
      `cargo test --examples -j 1 -- --test-threads=1` pass (flex).
- [ ] `cargo doc --all --no-deps` has no warnings.

## Comments
