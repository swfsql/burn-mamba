# 01: Follow the removals of burn-stack

Status: ready-for-agent

Blocked by: `../burn-stack/.scratch/block-family-choice/issues/01-remove-remote-backend.md`, `../burn-stack/.scratch/block-family-choice/issues/02-remove-per-layer-class-latents.md`

Spec: `../spec.md`. Run only one session on the two repos.

**What to build:**

- **The remote backend**: delete `backend-remote` from `Cargo.toml` and the word
  `remote` from the backend list of `CLAUDE.md`. Delete each other mention that
  `rg -i "remote"` finds, including in `bench.sh` and `kernels.sh`. Keep the
  mentions that mean a git remote.
- **Per-layer class latents**: delete the tests of the removed behaviour in
  `src/unified/tests/class.rs`, `multi_gate.rs` and `layers.rs`. Examples:
  `per_layer_class_latents_step_matches_forward`,
  `prime_runs_a_per_layer_latent_mamba3`,
  `layers_multi_gate_per_layer_class_latents_step_matches_forward`,
  `layers_per_layer_middle_and_end_latents_step_matches_forward`,
  `layer_prime_returns_the_latent_and_its_delta`. Read each test before you delete
  it. A test of a stack-level latent stays.
- **The pin**: move both `rev` values over the new burn-stack commits.

**Done when:**

- `cargo check` and `cargo doc --all --no-deps` have no warnings.
- `cargo test --lib -- --test-threads=1` passes. The only missing tests are those
  of the removed behaviour.
- `cargo test --examples -j 1 -- --test-threads=1` passes.

## Comments

A review of burn-stack ticket `02-remove-per-layer-class-latents.md` found more
work for this ticket:

- Three docs of this repo need an update:
  - `src/unified/network.rs:96` says "all three marker levels". Two levels stay.
  - `src/unified/tests/optim.rs:238` and `examples/mnist-ae/model.rs:289` name
    `straight_block`/`reverse_block`. `BidiLayers` holds its blocks under
    `real_layers.N.block`.
- Five tests set a stack-level latent and a per-layer latent. Delete only their
  per-layer lines and their asserts on `class.per_layer`:
  - `prime_on_a_network_covers_every_class_level`
  - `class_markers_split_forward_matches_single_forward`
  - `latent_network_multi_gate_class_markers_prime_step_matches_forward`
  - `layers_multi_gate_class_latents_split_forward_matches_single`
  - `layers_multi_gate_prime_then_step_matches_step`

The implementation of burn-stack ticket `02-remove-per-layer-class-latents.md`
adds this work:

- `class_prime_plan` has no `window_len` parameter now. In
  `src/unified/tests/class.rs`, delete the second argument (`0`) of each call.
  In `class_prime_plan_emits_only_what_waits_for_the_next_token`, delete the
  block that passes `1`. It tests the tokens that a lower level handed up, and
  only the per-layer latents did that.
- `the_carry_tracks_class_latents_spliced_below_the_cut` in
  `src/unified/tests/layers.rs` tests only per-layer latents. Delete it. The
  burn-stack test `grad_horizon_keeps_a_class_latent_of_the_stack_training`
  covers a class latent of a stack below a cut.
- The tests over `run_prime_cut` in `src/unified/tests/layers.rs` use only
  per-layer latents. They are the only tests that run `prime` under a cut. Port
  the `run_prime_cut(0)` case to a class latent of the stack. The
  `run_prime_cut(3)` case has no port: the stream of a prime now always crosses
  the cut.
- `latent_network_multi_gate_class_markers_prime_step_matches_forward` and the
  matching test of `src/unified/tests/class.rs` need more than the deletion of
  their per-layer lines. Without the layer latent, the sequence is `S N u0 …`.
  So `seq + 3` becomes `seq + 2`, `row(2)` becomes `row(1)` and `row(t + 3)`
  becomes `row(t + 2)`. Change their comments and their "three markers" message
  too.
