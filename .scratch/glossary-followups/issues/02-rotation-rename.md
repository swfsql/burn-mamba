# 02: Rename rope to rotation

Status: ready-for-agent

Blocked by: 01

**What to build:** The code and the docs call the rotation "rotation". "RoPE"
names only the RoPE trick: the technique that moves the accumulated rotation
into B and C (see `GLOSSARY.md`: Rotation, RoPE trick, Rotation fraction). A
user sets `rotation_fraction` in a Mamba-3 config.

## Rename map (decided in the grill)

| Old | New |
|---|---|
| `rope_fraction`, `with_rope_fraction` | `rotation_fraction`, `with_rotation_fraction` |
| `rope_dim` | `rotation_dim` |
| `rope_width` | `rotation_width` |
| `num_rope_angles` | `num_rotation_angles` (its doc keeps "Python: `num_rope_angles`") |
| test and helper names where "rope" means the extent of the rotation (`*_rope_half*`, `full_rope`, `partial_rope`, `cfg_rope_half*`, `production_rope`) | the same name with `rotation` |
| prose "RoPE" that means the rotation | "rotation" |

Keep these names. They name the RoPE trick: `apply_rope`, `apply_rope_partial`,
the module `rope.rs`, and its locals (`x_rope`, `x_h1_rope`, `x_h2_rope`,
`x_rope_rotated`). A name that means "B or C after the RoPE trick" also stays.

Do not touch the experiment directories `examples/*/tmp/`. They record past
runs. Old configs stop loading, and that is correct (no config compatibility).

## Method

- A scripted replacement is allowed for the identifiers in this ticket (the
  user approved it). Review every hit with `rg` afterwards.
- Edit the prose with the Edit tool, site by site.
- Update every doc mention: CLAUDE.md (the Notation entry `a`), module
  headers, `info/`, README.md, examples/README.md.

## Acceptance criteria

- [ ] `rg -i rope` over `src`, `examples` (without `tmp/`), `benches`, `info`
      and the markdown files finds only the keep-list names, "RoPE trick", and
      citations of the Python name.
- [ ] `cargo check`, `cargo test --lib -- --test-threads=1`,
      `cargo test --examples -j 1 -- --test-threads=1` pass (flex).
- [ ] `cargo doc --all --no-deps` has no warnings.

## Comments
