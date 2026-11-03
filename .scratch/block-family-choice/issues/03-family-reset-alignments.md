# 03: The reset alignment of each family

Status: ready-for-agent

Blocked by: `02-family-module-and-configs.md`, `../burn-stack/.scratch/block-family-choice/issues/03-reset-align-and-checked-packed.md`

Spec: `../spec.md`. Run only one session on the two repos.

**What to build:**

- `impl Block for Mamba2` and `impl Block for Mamba3` give `reset_align`, with the
  math of `MambaVocabNet::pack_align` (`src/family/network.rs`). Mamba-3 divides by
  its micro-steps. Mamba-1 keeps the default `None`.
- `pack_align` delegates to the first block. It goes in ticket 04.
- `examples/tiny-stories/training.rs:70-76` and `packed.rs` pass the alignment to
  `Packed::from_starts`.
- `src/packing.rs`: `chunk_resets` keeps its debug-only check. Its doc already says
  that the check reads the device.
- `packed.rs` keeps only the parity of each family. Delete a test only when
  `reference/tests.rs` of burn-stack holds the same check.
- Move the pin.

**Done when:**

- `cargo check` and `cargo doc --all --no-deps` have no warnings.
- `cargo test --lib -- --test-threads=1` passes.
- `cargo test --examples -j 1 -- --test-threads=1` passes.
