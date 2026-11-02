# 07: One name for the rotated width: rotation_dim

Status: resolved

Blocked by: None (can start immediately)

**What to build:** One name for the number of leading `state_rank` entries
that the rotation turns. Today the crate has two names for this value:
`rotation_dim` and `rotation_width`. After this ticket, the crate uses only
`rotation_dim`.

`rotation_dim` stays because it is the public name: `Mamba3::rotation_dim`,
`Mamba3Config::rotation_dim()`, `RotationSpec::rotation_dim`,
`apply_rope_partial` and the Notation table of CLAUDE.md and of
`src/mamba3/mamba3.rs`. `rotation_width` is local to
`src/mamba3/rotation/mod.rs`.

## Scope (all in `src/mamba3/rotation/mod.rs`)

- The parameter `rotation_width` of `rotate_blocks_partial` and of
  `rotate_blocks_two_sided_partial`, and their docs.
- The alias `let rotation_width = rotation_dim;` in `rotate_bc_forward`.
  Delete it, and pass `rotation_dim` at the call sites.
- The panic text "cache/block rotation width mismatch". Write
  `rotation_dim`. First, find the tests that match the text
  (`should_panic(expected = …)`).

## Acceptance criteria

- [x] `rg -i 'rotation.width'` over `src`, `examples` (without `tmp/`),
      `benches`, `info` and the markdown files finds nothing.
- [x] `cargo check`, `cargo test --lib -- --test-threads=1`,
      `cargo test --examples -j 1 -- --test-threads=1` pass (flex).
- [x] `cargo doc --all --no-deps` has no warnings.

## Comments

- The Standards review of ticket 02 found this ("one word, one meaning").
- The search of the first criterion does not include `.scratch/`, because
  this ticket and ticket 02 name the old identifier.
- No test matched the panic text, and no config holds the old name.
- The same change replaces the prose "rotated width" with `rotation_dim`: the
  comment in `rotate_bc_forward`, the rustdoc of `Mamba3::rotation_spec` and
  the comment in the validation of `Mamba3Config` (`src/mamba3/mamba3.rs`).
  The criterion search does not find this prose, and the rename rule of
  CLAUDE.md includes it.
