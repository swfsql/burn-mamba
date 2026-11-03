# 02: The `family` module, and configs in the `{ shape, block }` form

Status: ready-for-agent

Blocked by: `01-follow-the-removals.md`

Spec: `../spec.md`. Run only one session on the two repos.

**What to build:**

First commit: rename `src/unified/` to `src/family/`, and `src/unified/cache.rs` to
`src/family/adapters.rs`. Update every doc that names them: `CLAUDE.md` with its
File Map, `src/lib.rs`, `docs/agents/testing.md`, the READMEs, `info/`, and the
comments. `rg -n "unified"` lists them.

Then the configs:

- `MambaLatentNetConfig`, `MambaVocabNetConfig` and `MambaBidiLayersConfig` become
  concrete structs: `{ shape: LatentShape, block: MambaBlockConfig }`,
  `{ shape: VocabShape, block: MambaBlockConfig }` and
  `{ shape: BidiShape, block: MambaBlockConfig }`.
- `MambaBlockConfig` moves from `hnet.rs` to a new `src/family/config.rs`, with the
  three structs and their `ModelConfigExt` impls.
- `MambaBlockConfig` gets a method that gives the default `MambaSsdPath` of its
  family. It replaces the `default_path` helper of `tests/hnet.rs`.
- `init` builds the network enums of today from the shape. They go in ticket 04.
- `muon_plan` delegates to the shape. Check that the MLP stays in the plan.
- The 33 enum literals in `src`, `examples` and `benches` become builder calls.

`tiny-stories` and the other networks can now set an `InitPolicy`, because the
shape holds it. Add no policy to any example here: their behaviour does not change.

**Tests:**

- A config round trip through JSON, for each container and each family.
- The existing tests pass after their literals change.

**Done when:**

- `cargo check` and `cargo doc --all --no-deps` have no warnings.
- `cargo test --lib -- --test-threads=1` passes, with the same test names under
  `family::`.
- `cargo test --examples -j 1 -- --test-threads=1` passes.
