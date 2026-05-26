# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

A Rust library implementing [Mamba-1](https://arxiv.org/abs/2312.00752),
[Mamba-2](https://arxiv.org/abs/2405.21060), and
[Mamba-3](https://arxiv.org/abs/2603.15569) SSM (Structured State Space Model)
architectures on top of the [Burn](https://github.com/tracel-ai/burn/) deep
learning framework. The goal is a **minimal, readable reference
implementation** that ports the official CUDA/Triton kernels down to standard
Burn tensor ops — not a production-optimized one. There are no custom kernels;
everything is expressed with Burn's portable tensor operations so the same code
runs on every backend (CPU, WGPU, CUDA, Metal, LibTorch, …).

## Build & Test Commands

```bash
# Type-check (no backend needed for the lib surface, but tests/examples need one)
cargo check

# Run tests (flex is the preferred test/check backend)
cargo test --features "backend-flex"

# Build docs
cargo doc --all --no-deps

# Run an example (fibonacci = small Mamba-2, flex backend, fp32)
cargo run --example fibonacci --features "backend-flex" -- --training --inference

# Run the MNIST example (sequential MNIST classifier, Mamba-3, CUDA; long-running)
cargo run --example mnist-class --features "backend-cuda" -- --training --inference
```

- **Feature flags select the backend**: `backend-flex` (preferred for
  checks/tests), `backend-cpu`, `backend-wgpu`, `backend-metal`,
  `backend-vulkan`, `backend-cuda`, `backend-rocm`, `backend-tch-cpu`,
  `backend-tch-gpu`, `backend-remote`, `backend-ndarray` (+ BLAS variants,
  deprecated). Exactly one backend feature must be enabled or compilation fails
  with an explicit error (the `_dev-has-backend` signal).
- The `mamba1`, `mamba2`, `mamba3`, and `autodiff` features are on by default.
  `mamba2`/`mamba3` imply `autodiff` (needed for their custom backward).
- `cubecl` / `fusion` enable the memory-saving custom backward on those backend
  families. `dev-f16`, `dev-simd`, `dev-autotune` are example/test conveniences.
- `bacon.toml` configures [bacon](https://github.com/Canop/bacon) jobs (watch
  check/test); `cubecl.toml` configures the CubeCL runtime.

---

## File Map

The structure lists the files within the project. `refs/` and `doc/` are **external
reference material** (paper TeX, official Python implementation, third-party
minimal impl) and are intentionally not analyzed here — see
[Extra References](#extra-references).

```text
.
├── Cargo.toml                         # crate manifest: features (backends, mamba1/2/3, autodiff), deps
├── README.md                          # public readme: usage, feature list, learning links, references
├── cubecl.toml                        # CubeCL runtime config
├── bacon.toml                         # bacon watch-runner jobs (check/test)
├── examples
│   ├── README.md                      # how the examples are structured + CLI usage
│   ├── common                         # infrastructure shared by all examples
│   │   ├── README.md
│   │   ├── backend.rs                 # runtime backend + float/int dtype selection from feature flags (MainBackend, MainDevice)
│   │   ├── cli.rs                     # arg parsing (AppArgs), artifact dir mgmt, config load/save, train/infer flow
│   │   ├── mnist
│   │   │   ├── dataset.rs             # sequential-MNIST dataset loading/batching (pixels as a sequence)
│   │   │   └── mod.rs
│   │   ├── mod.rs                     # re-exports; ModelConfigExt trait (Config → Module factory)
│   │   ├── model
│   │   │   ├── bidi.rs                # bidirectional wrapper networks (Mamba2/3 BidiLayers) for examples
│   │   │   ├── mod.rs                 # ModelConfigExt; re-exports of the example networks
│   │   │   └── model.rs               # MyMamba2Network / MyMamba3Network (in_proj → Layers → out_proj) + config helpers
│   │   ├── optim.rs                   # optimizer config wrapper (OptimConfigExt)
│   │   └── training.rs                # generic training loop, TrainingConfig, LR scheduler glue
│   ├── fibonacci                      # smallest example: Mamba-2 on a fibonacci-like synthetic sequence
│   │   ├── README.md
│   │   ├── dataset.rs                 # synthetic fibonacci-like sequence generator
│   │   ├── inference.rs               # autoregressive generation via step()
│   │   ├── main.rs                    # launch(): wires backend, configs, train/infer (Mamba2BackendExt bound)
│   │   ├── model.rs                   # example model_config() for the fibonacci task
│   │   └── training.rs                # training entry for the fibonacci task
│   └── mnist-class                    # sequential-MNIST digit classifier: Mamba-3
│       ├── README.md
│       ├── main.rs                    # launch(): Mamba3BackendExt bound, cosine-annealing LR; inference is a stub
│       ├── model.rs                   # model_config() for the classifier
│       └── training.rs                # training entry (classification head on last timestep)
├── refs                               # EXTERNAL references (not analyzed)
│   │── VikramLex/mamba3-minimal       # unofficial Mamba-3 minimal impl (grain of salt) — basis of the double-ssd decomposition
│   │── mamba-3-paper                  # Mamba-3 paper TeX project
│   └── state-spaces/mamba             # official Mamba-1/2/3 Python implementation (authoritative)
├── src
│   ├── lib.rs                         # crate root: module decls, prelude, DENY_NAN/DENY_INF sanity flags
│   ├── mamba1                         # Mamba-1: original selective SSM (conv1d + sequential selective scan)
│   │   ├── cache.rs                   # Mamba1Cache(s): conv window + SSM state, and configs
│   │   ├── layer.rs                   # Mamba1Layer (Pre-LN residual block); NO Layers-stack / virtual-layer scheduling
│   │   ├── mamba1.rs                  # Mamba1 block + Mamba1Config; forward() (selective_scan) and step()
│   │   ├── mod.rs                     # module + prelude re-exports
│   │   └── network.rs                 # Mamba1Network (embedding → plain Vec<Mamba1Layer> → norm → LM head)
│   ├── mamba2                         # Mamba-2: Structured State Space Duality (SSD)
│   │   ├── bidi
│   │   │   ├── mod.rs
│   │   │   └── naive
│   │   │       ├── layer.rs           # Mamba2BidiLayers / Mamba2BidiLayerPair (forward + reversed pass)
│   │   │       ├── mod.rs
│   │   │       └── output_merge.rs    # OutputMerge: Mean | CatLinear merge of the two directions
│   │   ├── cache.rs                   # Mamba2Cache(s): conv window (bvk) + SSM state (bhpr)
│   │   ├── layer.rs                   # Mamba2Layer / Mamba2Layers (Pre-LN residual + virtual layers)
│   │   ├── mamba2.rs                  # Mamba2 block + Mamba2Config; chunkwise forward() and recurrent step()
│   │   ├── mod.rs                     # module + prelude (incl. Mamba2BackendExt, SsdPath/SsdInput)
│   │   ├── network.rs                 # Mamba2Network (full LM)
│   │   └── ssd                        # chunkwise SSD algorithms (the heart of Mamba-2)
│   │       ├── minimal.rs             # matmul/segsum SSD; autodiff backward
│   │       ├── mod.rs                 # re-exports backend-ext traits + SsdInput/SsdPath
│   │       ├── serial.rs              # serial-over-chunks hybrid SSD; autodiff backward
│   │       ├── serial_recalculated
│   │       │   ├── backward.rs        # registered custom Backward node (autodiff op)
│   │       │   ├── combined_backward.rs # recompute-based gradient math (memory-efficient)
│   │       │   ├── mod.rs             # wiring + BackendExt trait exports
│   │       │   └── serial_recalculated.rs # forward + Mamba2BackendExt impl
│   │       └── ssd_path.rs            # Mamba2SsdPath enum, Mamba2SsdInput struct, run() dispatch, optimal chunk_len
│   ├── mamba3                         # Mamba-3: trapezoidal SSD + data-dependent RoPE + MIMO
│   │   ├── bidi
│   │   │   ├── mod.rs
│   │   │   └── naive
│   │   │       ├── layer.rs           # Mamba3BidiLayers / Mamba3BidiLayerPair
│   │   │       ├── mod.rs
│   │   │       └── output_merge.rs    # OutputMerge (Mean | CatLinear)
│   │   ├── cache.rs                   # Mamba3Cache / Mamba3Caches ENUMS dispatching DoubleSsd vs SingleSsd
│   │   ├── double_ssd                 # double-pass trapezoidal decomposition (γ-SSD + β-SSD); VikramLex-style
│   │   │   ├── cache.rs               # Mamba3DoubleSsdCache(s): ssm/k_state/v_state/cum_angle (NO conv cache)
│   │   │   ├── double_ssd.rs          # forward_double_ssd / step_double_ssd; apply_rope / apply_rope_partial
│   │   │   ├── mod.rs
│   │   │   └── ssd                    # standard SSD kernels (reused by both γ and β passes)
│   │   │       ├── minimal.rs         # matmul/segsum MIMO-first SSD; autodiff backward
│   │   │       ├── mod.rs
│   │   │       ├── serial.rs          # serial-over-chunks SSD; autodiff backward
│   │   │       ├── serial_recalculated
│   │   │       │   ├── backward.rs
│   │   │       │   ├── combined_backward.rs
│   │   │       │   ├── mod.rs
│   │   │       │   └── serial_recalculated.rs
│   │   │       └── ssd_path.rs        # Mamba3DoubleSsdPath / Mamba3DoubleSsdInput (v_bnlmhp, da, b/c_bnlmhr, …)
│   │   ├── helpers.rs                 # shared forward/step helpers: trapezoid coeffs, QK-norm+GQA+bias, MIMO-V build
│   │   ├── layer.rs                   # Mamba3Layer / Mamba3Layers (Pre-LN residual, virtual layers, zero-cache factories)
│   │   ├── mamba3.rs                  # Mamba3 block + Mamba3Config; forward()/step() dispatch by cache variant
│   │   ├── mod.rs                     # module + prelude; Mamba3BackendExt aggregating both ssd ext traits (macros)
│   │   ├── network.rs                 # Mamba3Network (full LM; tied/untied LM head, vocab padding)
│   │   ├── single_ssd                 # single-pass official-kernel trapezoidal form (≈½ training memory)
│   │   │   ├── cache.rs               # Mamba3SingleSsdCache(s): same fields, DIFFERENT ssm semantics (h')
│   │   │   ├── mod.rs
│   │   │   ├── single_ssd.rs          # forward_single_ssd (scale + boundary-β seed); step_single_ssd (via double-ssd cache round-trip)
│   │   │   └── ssd
│   │   │       ├── minimal.rs
│   │   │       ├── mod.rs
│   │   │       ├── serial.rs
│   │   │       ├── serial_recalculated
│   │   │       │   ├── backward.rs
│   │   │       │   ├── combined_backward.rs
│   │   │       │   ├── mod.rs
│   │   │       │   └── serial_recalculated.rs
│   │   │       └── ssd_path.rs        # Mamba3SingleSsdPath / Mamba3SingleSsdInput (adds gamma_bnlh, scale_bnlh)
│   │   └── ssd_path.rs                # Mamba3SsdPath: pathway-agnostic algo selector; From<>/Into<> both sub-paths
│   ├── schedule.rs                    # Schedule + BidiSchedule: virtual-layer → real-weight index mapping
│   └── utils
│       ├── backend_macros.rs          # macros emitting per-backend BackendExt impls + autodiff marker traits
│       ├── combined_grad.rs           # flatten/unflatten (y, final_state) into one tracked tensor for custom backward
│       ├── gqa.rs                     # gqa_expand_to_heads: replicate per-group B/C across heads_per_group
│       ├── log_sigmoid.rs             # numerically-stable log-sigmoid (custom, incl. fp16)
│       ├── loss
│       │   ├── bce.rs                 # binary cross-entropy
│       │   ├── cross_entropy.rs       # cross-entropy
│       │   ├── mod.rs
│       │   └── mse.rs                 # mean squared error
│       ├── mod.rs                     # utils root: stable_max, div_eps (per-dtype epsilon)
│       ├── primitive.rs               # FloatTensor primitive ↔ Tensor<B,D> conversion helper
│       ├── rms_norm.rs                # RmsNorm (last-dim, fp16-safe); used as QK-Norm in Mamba-3
│       ├── rms_norm_gated.rs          # RmsNormGated: norm + SiLU(z) gate (Mamba-2 out norm; Mamba-3 optional out norm)
│       ├── sanity.rs                  # sanity(): optional NaN/Inf guards gated by DENY_NAN/DENY_INF
│       ├── scheduler.rs               # LR schedulers (CosineAnnealing+warmup, Constant) — example use
│       ├── segsum.rs                  # stable segment-sum → 1-semiseparable mask (log-space prefix-sum diff)
│       ├── silu.rs                    # SiLU activation (custom, fp16-aware)
│       ├── softplus.rs                # softplus activation (custom, fp16-aware)
│       ├── split.rs                   # split_into: array-typed split_with_sizes for clean destructuring
│       └── test_helpers.rs            # max_abs_diff + grad-comparison macros used across tests
└── files.md                           # more in-depth exploration over some of the files listed above
```

### Files.md

This contains more information about the most important files in the project.

---

## Architecture (detailed)

### Layer → Network hierarchy (all three families)

Each model family follows the same composition, top to bottom:

```text
{Model}Network    embedding → {Model}Layers → final RMSNorm → LM head → logits
{Model}Layers     a stack of N (virtual) layers over M real weight sets
{Model}Layer      Pre-LN residual:  y = x·residual_scale + Block(RMSNorm(x))
{Model} (Block)   the SSM core itself (mamba1.rs / mamba2.rs / mamba3.rs)
```

- `layer.rs` wraps the core block with an input RMSNorm and a residual add, and
  the `{Model}Layers` stack owns **virtual-layer scheduling** (see
  [Virtual layers](#virtual-layer-scheduling)) plus `ignore_first/last_residual`
  flags (zero out the first/last residual when composing with other module
  types). **Exception: Mamba-1** has no `Mamba1Layers` stack and no
  virtual-layer support — `Mamba1Network` simply holds a `Vec<Mamba1Layer>` and
  loops over it. Virtual layers and the `…Layers` stack exist only for Mamba-2
  and Mamba-3.
- `network.rs` assembles the LM: token `Embedding` → layer stack → `norm_f`
  (final RMSNorm) → LM head. The LM head can be **tied** to the (transposed)
  embedding weights (`missing_lm_head = true` ⇒ `lm_head: None`) or a separate
  `Linear`. Vocab size is rounded up to `pad_vocab_size_multiple` for GPU
  alignment.

### Dual execution modes

Every block / layer / network exposes two methods:

- **`forward()`** — parallel (chunkwise) mode for training and prompt prefill.
  Linear in sequence but expressed as batched GEMMs that exploit tensor cores.
- **`step()`** — recurrent mode for token-by-token decoding. O(state) per token,
  constant memory, no growing KV cache.

`forward()` from any initial cache is mathematically equal to the recurrent
`step()` unrolling from that same cache — this parity (outputs **and** final
cache **and** gradients) is what the per-block test suites assert, and it
subsumes the chunked-prefill (split-vs-full) guarantee.

### Caches

A cache carries the streaming state between calls (prefill→decode, or chunked
prefill). Mamba-1/2 caches hold a conv window + SSM state. **Mamba-3 has no conv
cache** (the short convolution is removed); see the Mamba-3 section.

### SSD algorithm selection (Mamba-2 & Mamba-3)

The chunkwise scan in Mamba-2/3 is pluggable via an `…SsdPath` enum, each
variant carrying an optional chunk length (`None` ⇒ optimal default
≈ `√(state_rank · per_head_dim)`, rounded to a multiple of 32, capped at 512):

| Variant | Algorithm | Backward |
|---------|-----------|----------|
| `Minimal(chunk)` | mostly batched matmuls + `segsum` mask | **autodiff** |
| `Serial(chunk)` | serial loop over chunks + matmuls (mirrors the 5 Triton kernels K1–K5) | **autodiff** |
| `SerialRecalculated(chunk)` | serial loop, **custom memory-efficient backward** (recompute; mirrors `ssd_combined.py`) | **custom** (saves ~⅓ training memory) |

`Default` = `SerialRecalculated(None)`. `SsdPath::run(input)` dispatches to
`ssd_minimal` / `ssd_serial` / `ssd_serial_recalculated`. All three are exact
reformulations of the same SSD and must agree on values **and** gradients
(asserted by the `ssd_path.rs` tests).

#### Backend extension traits

Each SSD family defines a `…BackendExt` trait (e.g. `Mamba2BackendExt`,
`Mamba3DoubleSsdBackendExt`, `Mamba3SingleSsdBackendExt`). The default trait
body works for every plain Burn backend; only `Autodiff<B>` gets the custom
backward, implemented in the `serial_recalculated/backward.rs` +
`combined_backward.rs` pair. `utils/backend_macros.rs` emits the per-backend
"use default impl" blocks (`impl_ssd_backend_ext_for_burn_backends!`) and the
autodiff marker trait (`decl_ssd_autodiff_backend_ext!`). `Mamba3BackendExt`
aggregates both the double- and single-ssd ext traits.

The `serial_recalculated` custom backward must hand Burn a single tracked
tensor; `utils/combined_grad.rs` flattens `(y, final_state)` into one 1-D tensor
and splits it back afterwards.

### Mamba-1 (`src/mamba1/`)

Original selective SSM. `mamba1.rs`: in-projection → causal depthwise `conv1d`
(left-padded from the conv cache window for strict causality) → SiLU → the
`x_proj`/`dt_proj` selective projections → a **sequential `selective_scan`**
(ZOH for A, Euler for B) → SiLU gate → out-projection. `step()` is the
single-token recurrence sharing the same cache (`conv` window + `ssm` state).

### Mamba-2 (`src/mamba2/`)

Structured State Space Duality. `mamba2.rs` is heavily documented (read its
module header for the full SSD math). Pipeline in `forward()`:

1. **In-projection** `d_model → [z | xbc | dt_raw]` (one linear, enables tensor
   parallelism with a single all-reduce).
2. **Causal depthwise Conv1d** + SiLU over `xbc` (left-padded from cache).
3. **Split** `xbc → (x, B, C)`.
4. **Discretise**: `Δ = softplus(dt_raw + dt_bias)` (clamped); `Ā = exp(Δ·A)`,
   `A = -exp(a_log)` per head; `B̄ = Δ·B` (Euler).
5. **Zero-pad** sequence to a multiple of `chunk_len` (Δ=0 ⇒ Ā=1, B̄=0, so the
   state passes through unchanged → final state is exact).
6. **GQA-expand** B/C from `ngroups` to `nheads` (`utils/gqa.rs`), then run the
   selected **SSD path**.
7. **Gated RMSNorm** with the `z` gate; **out-projection**.

`step()` runs the pure recurrence `hₜ = Āₜ hₜ₋₁ + B̄ₜ xₜᵀ`, `yₜ = Cₜᵀ hₜ + D xₜ`.
Optional learnable initial state `init_state_hpr`.

### Mamba-3 (`src/mamba3/`)

Mamba-3 extends Mamba-2 with three independent additions (each works alone or
combined): **trapezoidal discretisation**, **data-dependent RoPE** on B/C, and
**MIMO** (multiple-input multiple-output) rank expansion. Read the
`mamba3.rs` module header for the full combined math.

#### Differences from Mamba-2

| Aspect | Mamba-2 | Mamba-3 |
|--------|---------|---------|
| Recurrence | 2-term `h = Ā h + B̄ x` | 3-term trapezoidal `h = α h + β B₋₁x₋₁ + γ Bₜxₜ` |
| `A` | fixed per-head `a_log_h` | data-dependent `Aₜ = −softplus(dd_A)`, clamped `≤ −a_floor` |
| `λ` (trapezoid split) | absent | per-head data-dependent `λ = σ(λ̂)` |
| Short conv | present | **removed** (no conv cache) |
| B/C norm | post-SSD gated RMSNorm | **QK-Norm** (`RmsNorm` over `state_rank`) applied **before** SSD |
| B/C bias | none | learnable `[nheads, mimo_rank, state_rank]`, init = 1 |
| Positional | none | data-dependent cumulative RoPE on B/C |
| MIMO | none | `mimo_rank > 1`: parallel B/C rank channels + `mimo_x/z/o` projections |
| Out gate | gated RMSNorm | SiLU gate, or optional per-head gated RMSNorm (`has_outproj_norm`) |

#### Trapezoidal coefficients (`helpers::trapezoidal_coefficients`)

```text
Δₜ = softplus(dd_dt + dt_bias)        (clamped to dt_limit)
Aₜ = −softplus(dd_A)                  (clamped ≤ −a_floor)
αₜ = exp(Δₜ·Aₜ)                       — decay
βₜ = (1 − λₜ)·Δₜ·αₜ                   — left-endpoint weight (Bₜ₋₁xₜ₋₁)
γₜ = λₜ·Δₜ                            — right-endpoint weight (Bₜxₜ)
```

Setting `λ ≡ 1` collapses to the Mamba-2 (Euler) form.

#### In-projection layout

```text
d_in_proj = 2·d_inner + 2·ngroups·state_rank·mimo_rank + 3·nheads + num_rope_angles
split:      [ z | x | B_raw | C_raw | dd_dt | dd_A | λ_raw | θ ]
```

`B_raw`/`C_raw` carry the `mimo_rank · ngroups · state_rank` channels each.
`ngroups < nheads` shares B/C across heads (GQA-style).

#### Data-dependent RoPE (`apply_rope` / `apply_rope_partial`, in `double_ssd.rs`)

Angles are projected (`num_rope_angles = rope_dim/2`), squashed by
`tanh(θ)·π`, scaled by Δ per head, then **cumulatively summed** along the
sequence (continued from the cache's `cum_angle`):

```text
cum_angle[t] = cum_angle[t-1] + Δₜ · π · tanh(θₜ)
```

Pairing convention depends on the path: **SISO (mimo_rank == 1)** uses
interleaved/NeoX pairing `(0,1),(2,3),…` (`rotate_pairwise = true`); **MIMO**
uses half-and-half/GPT-J pairing (`rotate_pairwise = false`). With
`rope_fraction = 0.5` only the first `rope_dim` entries are rotated, the rest
pass through (`apply_rope_partial`).

#### Two SSD pathways — the central Mamba-3 design point

The trapezoidal recurrence is realised by **two interchangeable algorithms**,
selected at runtime by which **cache variant** is supplied. `Mamba3Cache` and
`Mamba3Caches` are **enums** (`DoubleSsd | SingleSsd`); `Mamba3::forward`/`step`
match on the variant and delegate. A missing cache defaults to **SingleSsd**.

- **Double-SSD** (`src/mamba3/double_ssd/`, VikramLex-style): splits the
  trapezoid into two **standard** SSD calls that reuse the Mamba-2-like kernels:
  - γ-SSM: `hᵞₜ = αₜ hᵞₜ₋₁ + γₜ Bₜ xₜ`   (current token)
  - β-SSM: `hᵝₜ = αₜ hᵝₜ₋₁ + βₜ Bₜ₋₁ xₜ₋₁` (previous token, "shift-before-chunking")
  - `hₜ = hᵞₜ + hᵝₜ`. Simple and easy to verify, but ~2× the intra-chunk and
    chunk-state memory. Both passes feed `da = Δ·A` so the SSD computes
    `exp(Δ·A) = α`. `step()` runs this recurrence directly.

- **Single-SSD** (`src/mamba3/single_ssd/`, official Triton-SISO /
  Tilelang-MIMO form): one SSD call using `scaleₜ = γₜ + (1−λₜ₊₁)·Δₜ₊₁` as the
  key scale, a strict lower-triangular intra-chunk mask, a same-step γ
  correction, and a **boundary β seed** `(1−λ₀)·Δ₀·Kₜ₋₁⊗xₜ₋₁` folded into the
  initial state. ≈ half the training memory of double-ssd. Its cache's SSM
  accumulator `h'` has **different semantics** than the double-ssd state
  mid-sequence — hence a distinct `Mamba3SingleSsdCache` type so the two can't be
  mixed inside a chunked pass. At sequence boundaries (where caches are stored)
  `scaleₜ = γₜ`, so `h'` coincides with the double-ssd state; the two caches
  therefore convert via field-identity `From` impls (`src/mamba3/cache.rs`).
  `step_single_ssd` decodes by converting to the double-ssd cache, running
  `step_double_ssd`, and converting back.

`Mamba3SsdPath` (top-level) is pathway-agnostic; it converts to/from
`Mamba3DoubleSsdPath` / `Mamba3SingleSsdPath` via `From`. The SSD inputs differ:
double feeds pre-scaled `v_bnlmhp` (already × γ or β); single feeds raw `v` plus
`gamma_bnlh` and `scale_bnlh` so the kernel applies them internally.

#### Shared forward/step helpers (`helpers.rs`)

Three rank-generic helpers used by both pathways and both modes:
`trapezoidal_coefficients` (Δ/α/β/γ/da), `qk_norm_expand_bias`
(QK-Norm → GQA-expand groups→heads → add `[nheads, mimo_rank, state_rank]`
bias), and `build_v_with_mimo` (`v = x ⊙ mimo_x`, inserting the `mimo_rank`
axis; identity when SISO).

#### Mamba-3 cache fields (both variants)

`ssm_bhpr` (SSM hidden state), `k_state_bmhr` (previous-token B per rank, for the
β term), `v_state_bhp` (previous-token x), `cum_angle_bha` (cumulative RoPE
angle). The `ssm_bhpr` **semantics differ** between Double and Single (see
above). No conv cache.

### Virtual layer scheduling (`src/schedule.rs`)

`{Model}Layers` can run `n_virtual_layers` logical passes over `n_real_layers`
weight sets (e.g. 48 logical from 12 real). Each virtual layer keeps its **own
cache** but shares parameters. `Schedule` maps virtual→real index:
`Cyclic` (wrap), `Stretched` (block-repeat), `Custom(Vec)`. For bidirectional
stacks, `BidiSchedule` pairs forward/backward layers: `StridedCyclic`,
`StridedStretched`, `SymmetricCyclic`, `SymmetricStretched`, `Custom` (even
virtual indices = straight →, odd = reverse ←).

### Bidirectional support (`*/bidi/naive/`)

For non-autoregressive tasks. A `{Model}BidiLayerPair` runs a straight pass (→)
and a reversed pass (← via `flip` on the sequence axis, then flip back), then
merges with `OutputMerge` (`Mean` or `CatLinear`). `{Model}BidiLayers` stacks
pairs with `BidiSchedule`. Both Mamba-2 (`src/mamba2/bidi/`) and Mamba-3
(`src/mamba3/bidi/`) support it.

### Utilities (`src/utils/`)

Custom activations/norms exist because Burn either lacks them or needs an
fp16-stable variant: `silu`, `softplus`, `log_sigmoid`, `rms_norm` (also used as
Mamba-3 QK-Norm), `rms_norm_gated`. The fp16 RMSNorm paths avoid computing `x²`
directly (overflow) by normalising against `max(|x|)` first. Other helpers:
`segsum` (1-semiseparable mask via log-space prefix-sum differences),
`gqa` (group→head expansion), `split` (typed array split),
`combined_grad`/`backend_macros`/`primitive` (custom-backward plumbing),
`sanity` (NaN/Inf guards gated by the crate-level `DENY_NAN`/`DENY_INF`
constants in `lib.rs`), `scheduler` (LR schedules), and `loss/` (bce,
cross_entropy, mse). `div_eps`/`stable_max` (in `utils/mod.rs`) give per-dtype
numerical constants.

### Examples (`examples/`)

`examples/common/` holds shared infra: `backend.rs` (compile-time backend +
dtype selection from features), `cli.rs` (`AppArgs`, artifact dir, config
load/save, train→infer flow), `training.rs`/`optim.rs` (generic loop + optimizer
config), `model/` (`MyMamba2Network` / `MyMamba3Network` = `in_proj` → `Layers`
→ `out_proj`, plus the `*_block_config`/`*_layers_config` builders and the
bidirectional wrappers), and `mnist/` (sequential-MNIST dataset). Each concrete
example (`fibonacci/`, `mnist-class/`) supplies its own
`main.rs` (`launch()`), `model.rs` (`model_config()`), `training.rs`, and—where
relevant—`dataset.rs`/`inference.rs`. `fibonacci` is the smallest Mamba-2 demo;
`mnist-class` is a Mamba-3 sequential-MNIST classifier (inference is a stub).

---

## Key Design Decisions

- **No optimized kernels** — relies entirely on Burn's portable tensor ops, so
  the same code runs on every backend.
- **Two Mamba-3 SSD pathways** — double-ssd (simple, verifiable) vs single-ssd
  (official-kernel form, ~½ training memory). The cache type is what selects the
  pathway; the SSM accumulators differ mid-sequence, but coincide at boundaries,
  so the two caches inter-convert via field-identity `From` impls. `step()` runs
  the recurrent (double-ssd) form for both; single-ssd decoding round-trips
  through the double-ssd cache.
- **Three SSD algorithm variants** (`Minimal`/`Serial`/`SerialRecalculated`),
  the last with a custom recompute backward for memory savings; all proven
  equal on values + gradients by tests.
- **Bidirectional support** in both Mamba-2 and Mamba-3.
- **Burn 0.21+.0** — a recent version; APIs may differ from older online docs.
- The root of the project is `/shared/claude/burn-mamba/`; do not read/write
  outside it.
- Whenever source file is created/removed/updated, its entry should be
  added/removed/updated from the [File Map](#file-map) and `files.md`.
  [File Map](#file-map) should always list all source files, while `files.md`
  refers only to the most important or impactful files. Some updates naturally
  don't require any changes to their content.

---

## Notation

Tensor names carry a suffix encoding their shape, and the codebase is
**deliberately verbose/pedantic** about this — it is *desired*, and is backed by
frequent shape `assert`s and shape commentary. Conventions:

- If a single operation produces a tensor whose name already carries the shape
  suffix, no extra shape comment is needed (this also covers expansions, where
  the shape is explicit in the code).
- In commentary, a shape can be written underscore-style (`_bhl` instead of
  `[batch, nheads, chunk_len]`). When explaining a specific dimension it is fine
  to spell it out by name (`batch` instead of `b`) — verbosity is not a problem.
- Only when a shape deserves special attention (usually when an upper-case
  letter is involved) should a comment expand it explicitly to `[...]` form. The
  `[]` form is also natural when describing indexing.
- When context is clear, *paper* style (upper-case symbols `A, B, C, H, Y, L, …`)
  may appear in **documentation/comments**, but **code must never** use the
  reference nomenclature for actual identifiers — that would make the style
  internally inconsistent.
- **Lower-case** letters are the base dimensions (table below). **Upper-case**
  letters denote a *relation* of the lower-case ones (offset, multiple, concat,
  stacking): e.g. `X` may be `x+1`, `x−1`, `x*2`; `S` is the padded sequence;
  `K` is `conv_kernel − 1`; `XY` may be `x+y` or `x*y`. Upper-case may also
  refer to paper elements.

Internal-code dimension keys (the "Paper"/"Python" columns map to the Mamba
papers and `refs/state-spaces/mamba`):

| Letter | Dimension | Paper | Python | Typical |
|--------|-----------|-------|--------|---------|
| `b` | `batch` | — | `batch` | varies |
| `s` | `sequence` length | `T` | `seqlen` | varies |
| `d` | `d_model` | `D` | `d_model` | 768, 1024 |
| `i` | `d_inner` = `expand`·`d_model` | `E·D` | `d_inner` | 2·`d_model` |
| `h` | `nheads` | `H` | `nheads` | `d_inner`/`per_head_dim` |
| `p` | `per_head_dim` | `P` | `headdim` | 64, 128 |
| `r` | `state_rank` | `N` | `d_state` | 64, 128, 256 |
| `m` | `mimo_rank` (Mamba-3) | `M` | `mimo_rank` | 1–8 |
| `n` | `nchunks` = `sequence`/`chunk_len` | — | `nchunks` | varies |
| `g` | `ngroups` | `G` | `ngroups` / `num_bc_heads` | 1 … `nheads` |
| `l` | `chunk_len` | `Q` | `chunk_size` | 64 … 256 |
| `a` | `num_rope_angles` = `rope_dim`/2 (Mamba-3) | — | `num_rope_angles` | varies |
| `v` | `conv_dim` = `d_inner`+2·`ngroups`·`state_rank` (Mamba-2) | — | `conv_dim` | — |
| `k` | `conv_kernel` (Mamba-1/2) | — | `d_conv` | 4 |

Per-file module headers (notably `mamba2.rs` and `mamba3.rs`) carry their own
notation tables tailored to that file — consult them first when editing.

## Extra References

- **Mamba-3 paper** (TeX project): `refs/mamba-3-paper/`.
- **Official Mamba-1/2/3 implementation** (Python, authoritative; a clone of the
  authors' GitHub): `refs/state-spaces/mamba/`. The Triton SISO and Tilelang
  MIMO kernels here are the reference for the single-ssd path.
- **Mamba-3 minimal** (unofficial):
  `refs/VikramLex/mamba3-minimal/` — the basis of the double-ssd decomposition.
