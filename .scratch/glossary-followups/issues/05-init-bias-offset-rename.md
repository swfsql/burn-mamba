# 05: Rename init_bias_step to init_bias_offset

Status: ready-for-agent

Blocked by: None (can start immediately)

**What to build:** burn-mamba builds again against burn-stack. burn-stack
renamed the field `init_bias_step` of Multi-Gate Residuals to
`init_bias_offset` (its ticket 02:
`../burn-stack/.scratch/glossary-followups/issues/02-prose-sweep.md`). In the
glossary of burn-stack, "step" names only the step call. The field is the
offset between the initial gate biases of two consecutive residual streams.
Until burn-mamba makes this rename, it does not compile against burn-stack.

## Rename map

| Old | New |
|---|---|
| `init_bias_step` (a field of `MultiGateResidualConfig` and of `ResidualsConfig::MultiGate`) | `init_bias_offset` |
| `with_init_bias_step` | `with_init_bias_offset` |
| the test `init_bias_step_ramps_the_gates` | `init_bias_offset_ramps_the_gates` |
| prose "step" for this offset (for example, the doc of that test) | "offset" |

## Sites

These tracked files use the old name:

- `src/unified/tests/layers.rs`
- `src/unified/tests/multi_gate.rs`
- `src/unified/tests/packed.rs`
- `examples/mnist-class/model.rs`
- `examples/tiny-stories/model.rs`
- `examples/tiny-stories/test_net.rs`

Do not touch the experiment directories `examples/*/tmp/` (their `gen.py` and
the `model_config.json` of each run). They record past runs. A saved config
with `ResidualsConfig::MultiGate` stops loading, because the new field has no
default there. That is correct: no config compatibility, as in ticket 02.

burn-deltanet also sets the field, at two sites
(`examples/tiny-stories/model.rs`, `src/unified/tests/layers.rs`). That repo
needs the same change in a session of its own. It is not part of this ticket.

## Method

- Use the Edit tool. `replace_all` is correct for the identifier in each
  file. Review every hit with `rg` afterwards.
- Edit the prose with the Edit tool, site by site.

## Acceptance criteria

- [x] `rg init_bias_step` over `src`, `examples` (without `tmp/`) and the
      markdown files finds nothing.
- [x] `cargo check`, `cargo test --lib -- --test-threads=1`,
      `cargo test --examples -j 1 -- --test-threads=1` pass (flex).
- [x] `cargo doc --all --no-deps` has no warnings.

## Comments

- Done against burn-stack rev `4383373`. The commit also holds that bump of
  `Cargo.toml` (crate version 0.71.0) and of `Cargo.lock`.
