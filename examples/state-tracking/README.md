# State tracking on A₅ — abelian RoPE vs. quaternion rotation

A harness for the capability that **motivates** the **quaternion**
(`RotationKind::Quaternion4D`) rotation versus Mamba-3's default **abelian**
(`RotationKind::Complex2D`, `SO(2)`/complex RoPE) rotation.

The task is the **word problem of the alternating group `A₅`** (the rotation
group of the icosahedron, the smallest *non-solvable* group): the model reads a
leading reference token then a sequence of `A₅` generators (one-hot) and must
output, at every position, the cumulative product so far — a 60-way
classification (chance ≈ `1.7%`).

By Barrington's theorem this is `NC¹`-complete. A single-layer linear SSM with
abelian (`SO(2)`) transitions is confined to the solvable / `TC⁰` regime and
cannot *compose* it; the non-abelian `SU(2)` quaternion rotation can represent
the binary icosahedral group `2I = SL(2,5)` (a double cover of `A₅`), so in
principle it can.

This example uses the shared example harness (`common/`), like `fibonacci` and
`mnist-class`: `dataset.rs` (the A₅ generator/running-product dataset),
`model.rs` (`model_config(rotation)` returning a `MambaLatentNetConfig::Mamba3`),
`training.rs` (cross-entropy over **every** position + a per-position accuracy
readout), `inference.rs` (per-position eval accuracy), and `main.rs`.

## What to expect — please read this

This is an honest, runnable **harness**, *not* a tuned benchmark, and the headline
result is nuanced:

- **The metric matters.** Watch the **per-position accuracy** (printed every
  validation pass and at inference), not the per-token average. Position `t` has
  only `≤ NUM_GENERATORS^t` reachable input prefixes, so the early positions are
  solved by pure **memorisation** by *both* rotations — averaging hides this. The
  abelian/non-abelian distinction only lives in the **deep** positions, where the
  running product genuinely has to be composed.

- **At this tiny scale the gap is modest.** With the default deliberately tiny,
  single-layer, fibonacci-scale model, both rotations track to a similar depth
  (both memorise short prefixes; with `NUM_TRAIN` sequences the "memorisation
  frontier" reaches depth ≈ `log_NUM_GENERATORS(NUM_TRAIN)`). The quaternion shows
  only a **small edge in the deepest positions**, and only after **extended
  training**. A single short run will mostly show both curves tracking together.

- **A clean, dramatic gap needs scale.** Surfacing the full capability
  (quaternion ≈100% deep, abelian at chance) requires more model capacity
  (width/depth) and much more training than this demo affords — consistent with
  the Mamba-3 paper's state-tracking experiments. Empirically, a single tiny
  layer does not learn strong `A₅` composition at accessible training budgets.

The design choices that make the comparison *valid* (so the harness is correct
even though the gap is modest):

- **Reference/anchor token.** The Mamba-3 rotation rotates both `B` and `C`, so
  the SSD readout only ever sees the *relative* rotation `Rₜ⋯Rᵢ₊₁` between
  positions — never the *absolute* running product `Pₜ`. A fixed reference token
  at position 0 (rotation learned to identity, `P₀ = I`) anchors the readout so
  `Cₜᵀ Pₜ B₀ x₀` carries the absolute product. Without it *both* rotations are
  confined to relative information and collapse at the same shallow depth.

- **Per-position metric** to expose where each rotation actually fails.

- **A regularised, constant-LR recipe** (weight decay + constant LR) so resuming
  is a seamless continuation of the slow, grokking-style transition.

## Run

```bash
cargo run --release --example state-tracking --features backend-flex -- --training --inference -- --rotation complex
cargo run --release --example state-tracking --features backend-flex -- --training --inference -- --rotation quaternion
```

`--rotation complex|quaternion` (default `complex`) selects the rotation baked
into a fresh model config; it is forwarded after the trailing `--` via the
harness's `extra_args`. All the standard harness flags apply (`--training`,
`--inference`, `--artifacts-path`, `--training-config`, `--model-config`,
`--remove-artifacts`; see `common/cli.rs` `HELP`). Once a model config is
persisted into an artifacts directory it wins on reload, so `--rotation` only
takes effect when a *new* config is created.

**Training longer (resume).** The deep-position behaviour only emerges over many
epochs. Re-run with the same `--artifacts-path` to continue training from the
saved weights/optimizer — the constant LR makes this a clean continuation:

```bash
# first run prints e.g. `new artifacts directory: "/tmp/...-state-tracking-XXXX-0"`
A=/tmp/...-state-tracking-XXXX-0
cargo run --release --example state-tracking --features backend-flex -- --training --inference --artifacts-path $A
# repeat to keep training; the per-position curve deepens
```

Epochs / batch size / LR / weight decay live in the (persisted)
`training_config.json`; edit it or `main.rs`. For heavier runs use
`--features backend-cuda` and scale up the model (`model.rs`), `NUM_TRAIN` /
`SEQ_LENGTH` (`dataset.rs`), and epochs.

## See also

- `src/mamba3/rotation.rs` — the quaternion rotation (algebra, cumulative scan,
  `RotationKind` / `RotationState`) and its tests. The module header derives the
  relative-rotation readout `Cₜᵀ (Rₜ⋯Rᵢ₊₁) Bᵢ` that motivates the anchor token.
- The Mamba-3 paper's "Complex-Valued SSMs" / state-tracking sections.
