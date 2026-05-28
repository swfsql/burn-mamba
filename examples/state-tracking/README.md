# State tracking on A₅ — abelian RoPE vs. quaternion rotation

A self-contained demo of the capability gap that motivates the **quaternion**
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

This example is intentionally minimal — one file, a hand-rolled training loop
(no `burn::train::Learner`, no dataloader), and a **tiny** model
(fibonacci-scale) so it runs quickly on CPU.

## Run

```bash
cargo run --release --example state-tracking --features backend-flex -- --rotation complex
cargo run --release --example state-tracking --features backend-flex -- --rotation quaternion
```

Flags: `--rotation complex|quaternion` (default `complex`), `--epochs N`
(default `30`).

Compare the two final per-token accuracies. The quaternion run climbs above the
complex run, which tends to plateau. For a wider, cleaner gap, run on GPU
(`--features backend-cuda`) and scale up the model / sequence length / epochs in
`main.rs` — this is a demonstration of the capability, not a tuned benchmark.

## See also

- `src/mamba3/rotation.rs` — the quaternion rotation (algebra, cumulative scan,
  `RotationKind` / `RotationState`) and its tests.
- The Mamba-3 paper's "Complex-Valued SSMs" section, which derives the
  abelian/complex RoPE that this generalises.
