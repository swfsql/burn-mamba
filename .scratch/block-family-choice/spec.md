# The family choice at the block

## Problem

`src/unified/` has one enum for each container, with one arm for each family.
Each body only forwards. So each new container method needs a new pass-through,
and each new stack knob needs an edit in each arm of each config enum. Only
`MambaHNetVocabNetConfig` has the `{ shape, block }` form of burn-stack
ADR-0007.

Packed rows put three unchecked invariants on the caller, and no burn-stack test
reaches them. `RefBlock` panics on a packed forward, against burn-stack ADR-0005.

## Decisions

All three families stay equally selectable, because third-party projects use
them (ADR-0009). The examples stay on the enum form for the same reason.

- `MambaBlock` is one enum over the three blocks. It fills `Block`.
- `MambaCaches` is a flat list of slots. Each slot has its own family tag, so
  `CacheStack::from_slots` guesses nothing.
- `MambaLatentNet`, `MambaVocabNet`, `MambaBidiLayers` and `MambaHNetVocabNet`
  stay as type aliases of the generic containers. No shim keeps the old variants.
- Each config is a concrete `{ shape, block }` struct in burn-mamba. Burn's
  `#[derive(Config)]` takes no generics.
- `MambaBlockConfig` gets a method that gives the default `MambaSsdPath` of its
  family.
- One family per stack stays the rule. The existing mismatch panics enforce it,
  so this change adds no check for it.
- This is a clean break. Saved configs and saved records stop loading, and Burn
  reports the missing tensors. No migration code exists.
- `src/unified/` becomes `src/family/`: `block.rs`, `config.rs`, `capture.rs` and
  `adapters.rs` (the `impl Block` of each family). `network.rs`, `bidi.rs` and
  `hnet.rs` go.
- `Block::reset_align` replaces `MambaVocabNet::pack_align`. It gives the reset
  alignment for given `Options`. Its default is `None`, which means no packed
  rows.
- `Packed::from_starts` takes the alignment and checks each start. `Packed` stores
  `lead`, and `place` compares it with the table. Both checks read only host
  integers, so they run in every build.
- `chunk_resets` keeps its check for debug builds only, because that check reads
  the device.
- The fields of `Packed` become `pub(crate)`, so no caller can skip the checks.
- `RefBlock` gets a packed forward and a `reset_align` option (default 1).
- The remote backend goes from both crates.
- Per-layer class latents go. Nothing uses them. The code that supports them
  adapts: it loses that support, and it becomes simpler where the removal allows
  it.
- burn-stack goes to 0.38.0 and burn-mamba to 0.76.0.

## Evidence

A throwaway project holds `MambaBlock`, the flat caches and the four configs, over
the pinned burn-stack and only public items. `spike/` has a copy of its source. It
has no type-system blocker. Its 25 tests pass on flex with all families, and each
family also passes alone.

- A variant adds one segment to each parameter path, and nothing else.
- Outputs of the enum and of the family type are exactly equal, at block level
  and at network level. So are the `in_proj` gradients of a Mamba-3 network.
- A record of the current enum does not load into the new form, and the reverse
  also fails.

The spike did not run a GPU, a hardware recording or a bench. This change holds
no captured-step logic, so flex is the correct backend for its tests.

## Test seams

- **The enum**: a new `src/family/tests/block.rs`, from the suite of the spike. It
  keeps only the tests of what the enum can break. These are the derives, records,
  the path segment, exact outputs, the flat caches, the mismatch panics and the
  mixed-stack rule. Add `prime` and class tokens. Leave out the tests of the spike
  that only repeat a container test.
- **Packed rows**: the container checks go to `reference/tests.rs` of burn-stack.
  `packed.rs` here keeps only the parity of each family.
- **The baseline**: 450 lib tests on flex, with 443 passing and 7 ignored. burn-stack
  has 90, and burn-mamba has 360. No existing test may start to fail for a reason
  other than a removed item.

## Tickets

Run only one session on the two repos until the last ticket is complete. So each
ticket can use the `[patch]` of `Cargo.toml` (see `docs/agents/issue-tracker.md`).
Before each burn-mamba commit, push the burn-stack commits and move the pin.

Run the tickets in this order:

1. `../burn-stack/.scratch/block-family-choice/issues/01-remove-remote-backend.md`
2. `../burn-stack/.scratch/block-family-choice/issues/02-remove-per-layer-class-latents.md`
3. `issues/01-follow-the-removals.md`
4. `issues/02-family-module-and-configs.md`
5. `../burn-stack/.scratch/block-family-choice/issues/03-reset-align-and-checked-packed.md`
6. `issues/03-family-reset-alignments.md`
7. `issues/04-block-enum.md`

## After these tickets

The other findings of the review are in `.scratch/arch-review-backlog/` of each
repo. All have the status `needs-triage`. The preferred order is:

1. The plant frame and the in-projection layout, with their own design round
   (`01`, `02` of this repo).
2. The recompute work and the unused traits (`03`, `04`, `05` of this repo).
3. The container tests, then `Residuals` (`05`, `02` of burn-stack).
4. The others, as the user selects.
