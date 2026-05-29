# State tracking on A₅ — abelian RoPE vs. quaternion rotation

A demo of the capability gap that motivates the **quaternion**
(`RotationKind::Quaternion4D`) rotation versus Mamba-3's default **abelian**
(`RotationKind::Complex2D`, `SO(2)`/complex RoPE) rotation.

The task is the **word problem of the alternating group `A₅`** (the rotation
group of the icosahedron, the smallest *non-solvable* group): the model reads a
sequence of `A₅` generators (one-hot) and must output, at every position, the
cumulative product so far — a 60-way classification (chance ≈ `1.7%`).

By Barrington's theorem this is `NC¹`-complete. A single-layer linear SSM with
abelian (`SO(2)`) transitions is confined to the solvable / `TC⁰` regime and
cannot track it; the non-abelian `SU(2)` quaternion rotation can represent the
binary icosahedral group `2I = SL(2,5)` (a double cover of `A₅`), so it can.

This example uses the shared example harness (`common/`), like `fibonacci` and
`mnist-class`: a `dataset.rs` (the `A₅` generator/running-product dataset), a
`model.rs` (`model_config(rotation)` over the common `MyMamba3Network`), a
`training.rs` (cross-entropy over **every** position), an `inference.rs`
(per-token eval accuracy), and a `main.rs` (`launch()`). The model is
deliberately **tiny** (fibonacci-scale) so it runs quickly on CPU.

## Run

```bash
# train + report eval accuracy; --rotation is forwarded after the trailing --
cargo run --release --example state-tracking --features backend-flex -- --training --inference -- --rotation complex
cargo run --release --example state-tracking --features backend-flex -- --training --inference -- --rotation quaternion
```

`--rotation complex|quaternion` (default `complex`) selects the rotation baked
into a fresh model config; it is forwarded after the trailing `--` via the
harness's `extra_args`. All the standard harness flags apply (`--training`,
`--inference`, `--artifacts-path`, `--training-config`, `--model-config`,
`--remove-artifacts`); see `common/cli.rs` `HELP`. Note that once a model config
is persisted into an artifacts directory it wins on reload, so `--rotation` only
takes effect when a *new* config is created (e.g. the default fresh temp dir).

Epochs / batch size / LR live in the (persisted) `TrainingConfig`; edit the
defaults in `main.rs` or supply a `--training-config`.

Compare the two final per-token accuracies. The quaternion run climbs well above
the complex run, which tends to plateau near chance. For a wider, cleaner gap,
scale the model / sequence length / `num_epochs` up and run on GPU
(`--features backend-cuda`) — this is a demonstration of the capability, not a
tuned benchmark.

## See also

- `src/mamba3/rotation.rs` — the quaternion rotation (algebra, cumulative scan,
  `RotationKind` / `RotationState`) and its tests.
- The Mamba-3 paper's "Complex-Valued SSMs" section, which derives the
  abelian/complex RoPE that this generalises.
