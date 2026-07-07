# tmp.md — absorbed

The grokking study (setup, knobs, findings, conclusions, reproduction
commands) now lives in `examples/grokking/README.md` as a standalone report.
Per user decision, CLAUDE.md is **not** updated (the work is exclusive to the
grokking example).

Remainder — cross-example changes files.md may still want entries for:

- `examples/common/cli.rs` — optim-load workaround (burn bug,
  `info/optim-load.md`): `load_model` re-stamps persisted `ParamId`s onto the
  loaded module (`restore_param_ids`/`ParamIdStamper`, matched by dotted
  path); `load_optim` now takes `model: &impl Module` and prunes orphaned
  optimizer-state entries before `from_bytes`. All 5 examples'
  `load_or_save_optim(&cfg, &model)` call sites updated.
- `examples/common/model/mod.rs` — added
  `impl ModelConfigExt for MambaVocabNetConfig`.
- `Cargo.toml` — dev-deps added: `rand_chacha = "0.9.0"` (deterministic
  splits); `burn-pack` (same git rev, `std`) for the ParamId workaround.
- `examples/README.md` — grokking entry added (done in-place).

Run artifacts from the study remain under `examples/grokking/tmp/` (kept, per
no-deletion policy).
