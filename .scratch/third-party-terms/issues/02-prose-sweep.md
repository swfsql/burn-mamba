# 02: Sweep source words from our own prose

Status: resolved

Blocked by: None (can start immediately)

**What to build:** Our prose uses the terms of `GLOSSARY.md`. It does not use a
source word in the sense of the source. A quote of a source stays (ADR-0008).

Line numbers are those of commit `2039187`. Load the `asd-ste100` skill before
you edit. Edit each line by hand with the Edit tool, because the correct word
depends on the line.

## Decisions from the grill

| Old prose | New prose | Size and rule |
|---|---|---|
| "complex transition" | "rotation" | 11 lines in 8 files. Each line works with "rotation", so no new term is necessary. |
| "SSD core" | "SSD scan" | 11 lines in 8 files. |
| "right-endpoint weight" for `γ` | "the mass of the current sample" | `src/mamba3/mamba3.rs:28`, `src/mamba3/helpers.rs:584`. |
| "left-endpoint weight" for `β` | "the coefficient of the tap sample" | `src/mamba3/mamba3.rs:27`. `β` is not a mass, so "mass" is wrong here. |
| "SSD layer" | "block" or "plant" | 5 lines: `src/mamba2/mamba2.rs:3`, `:9`, `:55`, `:103`, `src/mamba2/ssd/minimal.rs:41`. |
| "Mamba-N layer" that means our block | "Mamba-N block" | `src/mamba1/cache.rs:25`, `src/mamba2/cache.rs:88`, `src/mamba3/double_ssd/cache.rs:80`, `src/mamba3/single_ssd/cache.rs:81`, `src/mamba3/mamba3.rs:227`. |
| "state-input" | "sample" | 6 lines, all in `info/`. |
| "SSM recurrence" | "recurrence of the plant" | `src/mamba2/ssd/serial.rs:349`, `src/mamba3/double_ssd/ssd/serial.rs:321`, `examples/reset/README.md:5`. |
| "selective SSM" that means our plant | "plant" | `src/mamba2/mamba2.rs:9`. Read the other 9 lines. |
| "Pre-LN residual layers" | "Pre-LN layers" | `README.md:75`. |

Read each line of these groups before you change it:

- **"Mamba-N layer" that is correct.** It means a layer with a Mamba-N block, so
  it stays. `examples/mnist-ae/model.rs:15` and
  `examples/reset/quintic/model.rs:67` are examples. Also read
  `examples/README.md:22`, `examples/reset/README.md:870` and
  `info/mamba-3/architecture-deltas.md:50`.
- **"selective SSM" that is correct.** It stays where it names the published
  model or a class of models: `README.md:36`, `src/lib.rs:18`,
  `src/mamba1/mamba1.rs:3`, `:56`.

## Limits

- `initial state` and `initial cache` stay. `_Avoid_` applies to them only for
  an origin.
- Do not change the experiment directories `examples/*/tmp/`. They record past
  runs.
- No identifier holds one of these words, so this ticket renames no identifier.
- The message at `examples/reset/rotor/tests.rs:611` changes. No test matches
  its text.
- No doc links to the title at `src/mamba3/mamba3.rs:49`, so the title can
  change. It becomes "Rotation", and it keeps its reference to the paper section.
- `info/mamba-3/rotation-as-optimization.md:76` becomes "The rotation lets `ηρ`
  fill the disc".

## Acceptance criteria

Run each `rg` from the repo root with
`--glob '!.scratch' --glob '!target' --glob '!GLOSSARY.md' --glob '!**/tmp/**'`.

- [x] `rg -i 'complex transition'` finds only quotes of a source.
- [x] `rg -i 'ssd core'` finds no hit.
- [x] `rg -i 'endpoint weight'` finds no hit in `src/mamba3/`.
- [x] `rg -i 'ssd layer'` finds only quotes of a source.
- [x] `rg -i 'state-input'` finds only quotes of a source.
- [x] `rg -i 'ssm recurrence'` finds only quotes of a source.
- [x] `rg -i 'pre-ln residual'` finds no hit in `README.md`.
- [x] `rg -i 'layer' src/mamba1/cache.rs src/mamba2/cache.rs src/mamba3/double_ssd/cache.rs src/mamba3/single_ssd/cache.rs`
      finds no "Mamba-N layer".
- [x] `cargo check` and `cargo doc --all --no-deps` give no warnings.
- [x] `cargo check --examples -j 1` passes.

## Comments

- The acceptance `rg` reads one line at a time. The sweep also changed the forms
  that it cannot see: the wrapped phrases at `README.md:279`,
  `info/mamba-3/rotation-as-optimization.md:338`, `:728`,
  `examples/reset/rotor/tests.rs:283`, `src/mamba2/mod.rs:3`, the emphasis at
  `examples/reset/spinor/main.rs:9`, and "state input" at
  `src/mamba3/tap_pattern.rs:3`.
- "state-input" also had 4 lines in `scripts/`, not only the 6 lines in `info/`.
  The acceptance `rg` includes them, so they changed too.
- "Each line works with rotation" is false at two lines. Proposition 3 gives
  `α·R(φ)`, so the whole transition is the momentum, not only `R(φ)`.
  `info/mamba-3/rotation-as-optimization.md:426` and `README.md:280` say
  "rotating transition".
- `info/mamba-3/architecture-deltas.md:50` and `:241` mean our block, so they say
  "block". `:235` and `scripts/mamba-3/architecture_deltas.py:232` name the read
  `y_t`, so they say "plant".
- `src/mamba2/mamba2.rs:44` said "the layer" for the SSD layer. It says "plant".
- The banners "cache for a single layer" of the four cache files say "block", to
  match the doc line below them.
- These lines did not change. The ticket does not list them, so the user decides:
  - `examples/reset/README.md:13`: `**complex** transition`. Its column compares
    the rotation kinds.
  - "complex-valued state transition": `README.md:40`, `src/lib.rs:22`, `:60`,
    `src/mamba2/mamba2.rs:55`, `src/mamba3/mamba3.rs:9`, `src/mamba3/mod.rs:7`.
  - "layer" that means our block: `src/mamba2/cache.rs:13`,
    `src/mamba3/mamba3.rs:108`, `src/mamba3/rotation/mod.rs:21`.
  - "residual layers": `src/mamba1/mod.rs:10`, `src/mamba2/mod.rs:15`.
  - "endpoint weight" out of `src/mamba3/`:
    `examples/reset/spinor-product/tests.rs:66`,
    `scripts/mamba-3/mimo_as_batch.py:22`, `:23`,
    `scripts/mamba-3/trapezoid_as_integration.py:19`, `:20`.
  - `src/mamba2/mamba2.rs:892`: "state input proj". It names `B`, so "sample" does
    not fit.
