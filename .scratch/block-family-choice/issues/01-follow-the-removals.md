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
