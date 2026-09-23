# files.md

A per-file **signature reference**: what each important file defines, and the
non-obvious decisions to know before you edit it. For the architecture and the
file tree, see `CLAUDE.md`. For the notation, see its
[Notation](./CLAUDE.md#notation) section. The module headers carry the detailed
math. This file points to them and does not repeat them.

This file covers **this** crate only. The block-generic composition layer
(`Layer`/`Layers`/networks/bidi/multi-gate/class tokens/schedules/norms/
losses/Muon) is the sibling crate `../burn-stack/` (see its `CLAUDE.md`).

Keep this file minimal (see CLAUDE.md → *Documentation Maintenance*): one short
entry per important file, no changelog. It omits trivial `mod.rs` glue and
`tests.rs` files.

> Burn 0.22 pins the high-level `Tensor` (every `Module`) to the global
> `Dispatch` backend, so library types are **not** backend-generic (no `<B>`).
> Only the custom-backward internals stay generic over `B` (`F<B,D>`/`Mask<B>`,
> the `Backward<B,_>` nodes, the `Autodiff<B>` ext impls).

---

## `src/lib.rs`

Feature-gated module declarations (`mamba{1,2,3}`, `unified`), the `prelude`
(the family preludes, the `Mamba*` unified types, `burn_stack::prelude::*`), and
`pub use burn_stack`. `#![warn(missing_docs)]`.

## `src/padding.rs`

The family half of right padding (the `pad` of `forward`):

- `real_len_b`,
- `window`: the per-slot `narrow`, as a fixed-shape gather. Every "last
  samples" cache field is read with it, at the end of its slot.
- `fill_padded`,
- `repeat_rows`: token mask → folded axis.

`tests.rs` checks every family, both Mamba-3 pathways across the dials, and a
network that ends with an `End` latent, against each slot run alone (outputs,
caches, gradients). A fully padded slot returns its cache.

---

## Mamba-1 (`src/mamba1/`): no SSD, no backend-ext trait

- **`mamba1.rs`**: `Mamba1` + `Mamba1Config` + `Mamba1Untied`.
  - `A` is input-**independent**, initialised from `log(arange(1..=state_rank))`.
  - `forward`: in_proj → causal conv (left-padded from `cache.conv_bik`) → SiLU
    → sequential `selective_scan` (ZOH A, Euler B) → SiLU gate → out_proj.
    `step` slides the conv window by hand (and accepts a conv without bias).
  - Padding: `Δ = 0` in `ssm`, the conv window read at the end of each slot.
  - `muon_projections()` (feature `optim`): `in_proj [x|res]`,
    `x_proj [dt*|B|C]`, `out_proj` (`*` = fallback optimizer).
- **`cache.rs`**: `Mamba1Cache` (`conv_bik` window + `ssm_bir` state) /
  `Mamba1Caches` (one per virtual layer, `n_caches`).

## Mamba-2 (`src/mamba2/`)

- **`mamba2.rs`**: `Mamba2` + `Mamba2Config` (`state_rank` 128, `per_head_dim`
  64, `ngroups` 1, `expand` 2) + `Mamba2Untied`.
  - Only `forward` uses the SSD path (through `Mamba2BackendExt`). `step` is
    the pure recurrence with a manual conv-window slide.
  - `init_state_hpr` (optional): each `forward` **adds** it to the incoming
    cache state. Only `Minimal` supports it (the serial paths panic). `step`
    does not read it.
  - Padding: `Δ = 0` (the same identity step as the chunk padding), the conv
    window read at the end of each slot.
  - `muon_projections()`: `in_proj [z|x|B|C|dt*]` (`xbc` split further: the
    conv is shared, the linear map is not), `out_proj`.
  - `InProjTail` moves `dt` into `in_proj_tail` (`project_in` joins the two).
- **`cache.rs`**: `Mamba2Cache` = `conv_bvk` window + `ssm_bhpr` (the
  O(p·r) state). Zero-init is correct (`h₀ = 0`).
- **`ssd/ssd_path.rs`**: `Mamba2SsdPath{Minimal|Serial|SerialRecalculated}
  (Option<chunk>)`, default `SerialRecalculated(None)`. `Mamba2SsdInput`
  (pre-processed, B/C GQA-expanded). `optimal_chunk_len ≈ √(r·p)`, a multiple
  of 32, capped at 512. `run()` dispatches.
- **`ssd/minimal.rs`**: the clearest reference, 4 steps (`Y_diag`, chunk
  state, inter-chunk scan, state → output). Autodiff backward.
- **`ssd/serial.rs`**: the same math as a serial chunk loop (mirrors Triton
  K1–K5). Autodiff backward. Numbered op comments per kernel.
- **`ssd/serial_recalculated/`**: the custom backward (recomputes
  intermediates, ~⅓ less memory). `serial_recalculated.rs` defines
  `Mamba2BackendExt` (default body = K1–K5 on primitives). `backward.rs`
  registers the `Autodiff<B>` node. `combined_backward.rs` is the gradient
  math (7 inputs). The `#[backend_extension]` list is the same at all four
  extension sites: one `Cube` arm for every cubecl `backend-*` feature
  (mirrors burn's `cube_backend` cfg), plus `Flex`/`NdArray`/`LibTorch`/
  `Autodiff`.

## Mamba-3 (`src/mamba3/`)

- **`mamba3.rs`**: `Mamba3` + `Mamba3Config` + `Mamba3Untied`. The module
  header is the math reference (§1–5 + notation table).
  - `state_rank` must be even unless `Real1D` (which accepts the scalar `1`).
    `init` asserts that a rotating kind turns ≥ 1 pair, and that the quaternion
    kinds turn whole 4-blocks.
  - MIMO init: `mimo_{x,z,o}_hmp` = `1/M`, `1`, `1/M`, so a MIMO block *is* its
    SISO block at init (`info/mamba-3/mimo-as-batch.md`).
  - Defaults: `rope_fraction` 1, `rotation_range` 2 (per factor for `Rotor4D`),
    `trapezoid` `HorizontalCarryOver`, `micro_steps` 1, `gain` `Projected`,
    `tropical` `None`. The reference model needs `rotation_range = 1`,
    `rope_fraction = 0.5`.
  - In-proj layout `[z | x·u | B·u | C | Δ·u | A·u | λ·u | μ·u | rotation·u |
    r·u | a·u | b·u]`: only the per-micro-step segments widen. The optional
    segments are last: `split_positive` removes `r`/`(a, b)`, then
    `helpers::split_trailing` removes rotation, `μ`, `λ` (in that order).
    `muon_projections()` mirrors this, with **one segment per micro-step**
    (independent maps for Muon). `without_segment` still drops a whole stream.
  - `InProjTail` moves the `d_in_proj_tail()` segments into `in_proj_tail` for
    any application count, so one tiled Muon spec fits every real layer.
  - `forward`/`step` **dispatch by cache variant** (missing ⇒ SingleSsd).
    `save_tap_slots` and `positive_tail` read the FIFO and the carries at the
    end of each slot. `positive_read` applies the positive ports **before** the
    `D` skip, so `has_outproj_norm` keeps them.
  - `init_state_hpr`: as in Mamba-2 (`Minimal` only, `step` does not read it).
  - Two performance-only `mimo_rank == 1` flags (`#[module(skip)]`, identical
    values/grads): `siso_specialization` (chunkwise γ-correction, wins
    everywhere) and `siso_specialization_decode` (per-token sites, wins on GPU,
    loses on CPU).
- **`mod.rs`**: `Mamba3BackendExt: Mamba3DoubleSsdBackendExt +
  Mamba3SingleSsdBackendExt`, wired by `backend_macros`.
- **`helpers.rs`**: helpers that both pathways and both modes share (see its
  header list). Non-obvious:
  - `trapezoidal_coefficients` is rank-3 (MambaProduct gives `step` a `u`
    axis). It returns **untransported** masses. A Kalman gain computes its
    decay **between `da` and `α`**, so every consumer reads the computed decay.
    `TrapezoidCoeffs::padded` makes padded positions the identity step.
  - A closed tap gives its mass **back** (`λ ← 1`, `μ ← 0`, by
    `token_start_gate`), so the gated patterns are bit-exact degeneracies.
  - The `A` floor is `-softplus(x).clamp(a_floor, ∞)`: the clamp must bind the
    positive softplus *before* the unary minus. The other order pins
    `A ≡ +a_floor` (a growing state).
  - `split_trailing`: `split_with_sizes` **drops** a zero-length segment, so an
    optional segment cannot be one more entry in the main split.
  - `prefix_sum` is **blocked** (runs of `∛(2·len)`, clamped `16..=256`).
    Burn's `cumsum` is `O(len²)` on cubecl. Blocking beats Hillis–Steele
    doubling at every length measured (numbers in the doc).
  - The read axis: `read_rows` / `read_causal_mask` (the identity at
    `stride = 1`), and `mod prim` twins on `F<B,_>` + `scatter_read_rows` for
    the recompute backwards.
- **`cache.rs`**: `Mamba3Cache{DoubleSsd|SingleSsd}` / `Mamba3Caches` enums,
  `from_vec` (**empty ⇒ SingleSsd**). The cross-pathway `From` impls are
  field-identity moves, valid at call boundaries (`scaleₜ = γₜ` there).
- **`ssd_path.rs`**: `Mamba3SsdPath` (pathway-agnostic, `From` into both).
  `optimal_chunk_len(r, p, m, u)` divides the `√(r·p)` rule by `m`, then rounds
  **up to a multiple of `u`**: `m` widens both chunk axes, `u` only the write
  axis (`chunk_tokens = chunk_len/u` read rows,
  `info/mamba-3/architecture-deltas.md` §8). `backward_chunk_group = ⌈n/2⌉`
  (rationale in the doc).
- **`trapezoid.rs`**: `Trapezoid`, the tap pattern (a closed 2×3 lattice,
  meaningful only at `u > 1`), and `TrapezoidSpec`. `tap_lag(u)` is the one
  knob of the single-tap members (shift, key-scale offset, FIFO depth).
  `has_interior_tap(u)` folds at `u = 1`. The module header states the one
  mass rule. `None` is structural: no `λ`, no `β`, no tap slots, one SSD call.

### `mamba3/double_ssd/`

- **`double_ssd/mod.rs`**: `forward_double_ssd` / `step_double_ssd`. One SSD
  call for `γ` + one per tap (shift-before-chunking by the lag of the tap,
  `β = ν·α` times `interior_gap_decay`). `step` solves the `u`-position block
  **in closed form** (one `mimo_outer_sum` per side over a fused
  `u·mimo_rank` axis), so its launch count does not depend on `u`. It reuses
  `forward`'s `shift_stream`/`interior_gap_decay`/`save_tap_slots`. Step
  helpers (`pub(crate)`): `StepProjection`/`step_project`, `step_readout`
  (`_siso`/`_mimo`), `step_finish`.
- **`cache.rs`**: `Mamba3DoubleSsdCache`: `ssm_bhpr`, the tap FIFO
  (`k_state_bumhr`/`v_state_buhp`, **oldest first**), `rotation`,
  `log_precision_bh`, `tropical_bh`. The FIFO's `k` is stored **as rotated**.
  Its `x` is **pre-scaled by the decay since its position** (this carries a
  lag-`u` gap across a call boundary). Absent fields are `None`, not zeros.
- **`ssd/`**: `Mamba3DoubleSsdInput` (**MIMO-first**, `v` pre-scaled by γ/β,
  `C` + `read_stride` on the read axis). The same three algorithms as Mamba-2,
  with `mimo_rank` fused into the chunk. `serial_recalculated/` owns the K1–K4
  primitives that **both** pathways' backwards use, including
  `k4_ssd_state_passing_backward` (the only walk in either backward).

### `mamba3/single_ssd/`

- **`single_ssd/mod.rs`**: `forward_single_ssd`: one SSD call with key scale
  `scaleₜ = γₜ + νₜ₊ₗₐ₉ (+ νⁱⁿᵗₜ₊₁)`, a strict mask + same-step γ correction,
  and a **boundary-β seed** folded into the initial state. Under
  `Trapezoid::None` it delegates to `forward_double_ssd`. `step_single_ssd`
  converts to a double-ssd cache, runs `step_double_ssd`, and converts back.
- **`token_band.rs`**: the lag-`u` correction band, as one intra-token
  contraction outside the kernel. It takes the **lag-`u` mass alone**. The
  pathways agree on everything a caller can observe, but not on the `u−1`
  partial sums per token that the read axis never computes (the module header
  explains why).
- **`cache.rs`**: `Mamba3SingleSsdCache`: the same fields as the double-ssd
  cache, but `ssm_bhpr` is the accumulator `h'`. Under `Trapezoid::None`,
  `h' ≡ h` everywhere.
- **`ssd/`**: `Mamba3SingleSsdInput`: raw `v` and `scale_bnlh` on the write
  axis, `C` / `gamma_bnth` / `read_stride` on the read axis, and
  `siso_specialization`. `diag.rs`: `y_diag_correction` (SISO-branched).
  `serial_recalculated/diag.rs` is its `F<B,D>` twin + analytic backward. The
  flag reaches the backward through the `Backward` node's `State`.

### `mamba3/product/`

`unfold_micro_bs` / `unfold_micro_b` (a `u`-wide segment → `u` positions, one
reshape) and `last_micro4` (the `x` of the `D` skip). The module header is the
reference for MambaProduct: the dial versus DeltaProduct's mechanism, what
`u` buys per `RotationKind`, the read/write axis split, and the closed-form
`step`.

### `mamba3/rotation/`

- **`mod.rs`**: `RotationKind{Real1D|Complex2D|Quaternion4D|Rotor4D}`,
  `RotationState{Real|Angle|Quaternion|Rotor}` (`identity(kind, …)`),
  `RotationSpec`, and the one entry point `rotate_bc_forward` (both pathways
  and `step`). The quaternion algebra, `quat_cumprod`, and the two-sided
  `Rotor4D` helpers. Non-obvious:
  - The generator bounds its **magnitude** (`bound_rotation_vector`), so the
    axis does not depend on the size of the projection.
  - `safe_norm` is scale-free: `‖r‖²` of raw channels overflows f16 at
    `|r| ≈ 250`, and `∞` gives a *zero* rotation.
  - The quaternion generators are **per head**.
  - `rotate_bc_forward` renormalises the scan's prefixes.
  - `Real` holds a tensor-less `NoRotation`: the `Module` derive wants one
    field per variant.
- **`rope.rs`**: `wrap_angle` (mod `2π`, offset `detach`ed) and
  `apply_rope`/`apply_rope_partial` (interleaved pairs for SISO,
  half-and-half for MIMO).

### `mamba3/positive/`

Per-head scalar systems beside the plant. Math and audit:
`info/kalman/gate-as-positive-system.md`.

- **`mod.rs`**: `Gain::{Projected, Kalman, KalmanProjectedNoise}`,
  `Tropical::{None, MaxPlus}`, `LOG_ZERO` (`ln 0` kept finite),
  `fresh_slots`.
- **`scan.rs`**: `lse` (exact next to `LOG_ZERO`, splits a tie's gradient),
  the `Mobius` (projective, shifted per combine) and `Affine` elements,
  `prefix` (Hillis–Steele: its doc says why it is not blocked), `fold` (the
  reference).
- **`kalman.rs`**: `LogMasses` (from the **pre-activations**: `ln` of an
  underflowed mass is a NaN gradient), `gate` (the decay, exact `ln α` at
  `κ = 0`). `Λ` is exact for a lag-1 tap, an upper bound at lag `u`.
- **`tropical.rs`**: `register`, the affine scan.

### `mamba3/quat_scan/`

The memory-efficient cumulative-product scan (a recompute backward, like SSD
`SerialRecalculated`). `quat_scan.rs`: `Mamba3QuatScanBackendExt` + the SoA
`Quat` helper (no per-step `narrow`/`cat`) + `quat_cumprod_recalculated`.
`backward.rs`: a `Backward<B,2>` that saves only `q` + `init` (the exact
unit-quaternion VJP, parallel ops only).

---

## The unified API (`src/unified/`)

- **`mod.rs`**: `MambaSsdPath` (+ `mamba{2,3}_default()`). The module header
  argues why the MIMO 3-D tensors are **diagonals** and not stacked matrices
  for Muon.
- **`cache.rs`**: `MambaCaches` (plain runtime state, **not** a `Module`) +
  `detach()`, and the `Block` / `BlockConfig` / `CacheStack` impls of each
  family. `cache_to_inner`/`cache_from_inner` are written by hand:
  `Module::map` does **nothing** on plain `Tensor` fields.
- **`capture.rs`**: `impl CacheTensors` (burn-stack's) for every family's
  caches and `MambaCaches`. The single-ssd cache uses the double-ssd traversal
  through the `From` move. Optional fields, pathway, family and rotation kind
  must match (else panic).
- **`network.rs`**: `MambaLatentNet` / `MambaVocabNet` + their `*Config`
  (wrapping `burn_stack::modules::{LatentNetwork, VocabNetwork}`), and the
  `ModelConfigExt` impls.
- **`bidi.rs`**: `MambaBidiLayers` + `MambaBidiLayersConfig` (wrapping
  `burn_stack::modules::BidiLayers`).
- **`tests/`**: the burn-stack containers tested against **real** blocks:
  `layer`, `layers` (`grad_horizon`), `multi_gate`, `bidi`, `class` (marker
  placement, forward/step/prime parity), `optim` (each plan fits its model and
  never selects a boundary weight, by `burn_stack::optim::BLOCK_CONTAINERS`),
  `untied`, `capture`.

## Benchmarks (`benches/layer.rs`, `bench.sh`, `kernels.sh`)

Criterion single-block benches (`forward`/`train`/`step`) for all three
families. The Mamba-3 cases compare the SISO-specialization flags and sweep the
rotation ladder. **The user runs them, not an agent.** Each case builds its
block, input, seed cache and warm-up *inside* the criterion closure (so a `--`
filter isolates one case) and *outside* the timed region. `bench.sh` runs the
backend configurations and writes `bench.md`. `kernels.sh` counts kernel
launches per case (cubecl's profiling logger) and writes `kernels.md`. Its
header explains how to read a count mismatch.

## Notes (`info/`) and their checks (`scripts/`)

Each note has a sibling script: float64 `numpy`, numbered checks that match
the section numbers, no import of the crate, non-zero exit on failure. Cite a
note. Do not restate it.

- **`info/mamba-3/rotation-as-optimization.md`** (69 checks): the
  optimization reading of the complex transition (the *quadratic* term), and
  `micro_steps` versus DeltaProduct.
- **`info/mamba-3/trapezoid-as-integration.md`** (74 checks): the trapezoid
  (the *linear* term along time): `λ` as operator splitting, the
  two-installment collapse under single-ssd, and the 2×3 tap lattice at
  `u > 1`. `trapezoid.rs` owns the parameterisation.
- **`info/mamba-3/mimo-as-batch.md`** (54 checks): `mimo_rank` (the linear
  term along *rank*): the minibatch reading, the cost of the value tying, why
  the rotation must be shared, and `MambaProduct(u=M) ⊇ MIMO(M)`.
- **`info/mamba-3/architecture-deltas.md`** (35 checks): the block *outside*
  the recurrence: BCNorm and the `B`/`C` bias floor, the deleted short conv,
  data-dependent `A`, the output-norm placements, and §8 (the chunk-length
  divisor).
- **`info/kalman/gate-as-positive-system.md`** (29 checks): `positive/`: the
  shared log-semiring scan, the Kalman ceiling and contraction, the max-plus
  register, and what a classifier gets for free (measured on
  `examples/tally/`).
