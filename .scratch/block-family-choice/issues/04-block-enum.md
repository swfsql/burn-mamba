# 04: `MambaBlock`, the one run-time family choice

Status: ready-for-agent

Blocked by: `03-family-reset-alignments.md`

Spec: `../spec.md`. Run only one session on the two repos.

**What to build:** Start from `../spike/lib.rs`, a copy of the throwaway project. It
compiles with only public items. `tmp/spike-block-enum/` has the project itself, if
this box has it.

- Add `src/family/block.rs`: `MambaBlock`, `MambaCache`, the flat `MambaCaches`, and
  their `Block`, `CacheStack` and `CacheTensors` impls.
- `MambaBlockConfig` implements `BlockConfig`.
- `src/family/config.rs` gets `MambaHNetVocabNetConfig`, and all `init` bodies call
  the shape directly.
- `MambaLatentNet`, `MambaVocabNet`, `MambaBidiLayers` and `MambaHNetVocabNet`
  become type aliases.
- Delete `src/family/network.rs`, `bidi.rs` and `hnet.rs`, with
  `with_family_pairs!`, `MambaHNetCaches`, `MambaBlockConfig::family` and
  `pack_align`.
- Callers read `net.layers.class_latents.len()` and
  `net.layers.only_start_latents()` directly. They call `reset_align` on the first
  block.
- `let MambaLatentNet::Mamba3(net) = &mut model` in tests and examples goes. They
  read the network fields directly, and match `MambaBlock::Mamba3(block)` only
  where they reach a block.
- `MambaCaches::detach` of `tiny-stories` becomes `CacheStack::detach`.
- `docs/adr/0009-family-choice-at-the-block.md` records this decision. Delete its
  line that names this ticket, because the code then follows it.
- Update the File Map, `examples/README.md`, the header of `src/lib.rs`, and each
  doc that names a deleted item. `src/family/mod.rs` still says that the module
  wraps monomorphisations in enums, so rewrite that header.
- Go to 0.76.0. The commit message states the clean break: saved configs and
  records stop loading, and no migration exists.
- Move `../spike/` to `/tmp`, and commit its removal. The code now holds its
  content. `tmp/spike-block-enum.discard` tags the throwaway project for the same
  move.

**Tests:** add `src/family/tests/block.rs` from `../spike/arch.rs`. Keep the tests
that only the enum can break:

- `module_surface`, `param_paths_add_only_the_variant`, `record_round_trip`,
  `record_of_another_family_panics`,
- `block_output_is_exact`, `network_output_and_gradients_are_exact`,
- `cache_stack_and_traversal`, `stack_knobs`, `untied_through_the_enum`,
- `hnet_mixes_stage_and_main_family`, `muon_plan_matches_the_family_type`,
  `captured_step_over_flat_caches`,
- `wrong_path_panics`, `wrong_cache_panics`, `mixed_stack_rules`.

Add `prime` and class tokens through the enum. Leave out the packed, vocab, bidi
and trainer tests of the spike, because existing tests cover them. Use the test
device rules of `CLAUDE.md`: `test_device()`, `dtype_tol`, and `max_rel_diff` for
values that can be large.

Also run the crate with each family alone. The spike did this with
`--no-default-features --features mamba1` (and `mamba2`, `mamba3`).

**Done when:**

- `cargo check` and `cargo doc --all --no-deps` have no warnings.
- `cargo test --lib -- --test-threads=1` passes on flex, with each family alone
  and with all families.
- `cargo test --examples -j 1 -- --test-threads=1` passes.
- `cargo run --example reset-majority -- --training --inference` runs.

## Comments

The implementation of ticket 02 found this:

- `MambaBidiLayersConfig::muon_plan` of the spike delegates to
  `BidiShape::muon_plan`, and that method gives only the block specs. So the
  `CatLinear.weight` spec is lost, and `assert_plan_fits` does not see it. Keep the
  spec in the `muon_plan` of `src/family/config.rs`. The test
  `bidi_plan_holds_the_cat_linear_merge` guards it.
- `init` and `muon_plan` stay inherent methods, and the `ModelConfigExt` impls
  forward to them. The prelude of burn-stack does not export the trait, and each
  `.init(&device)` of the tests and examples needs one of the two.
