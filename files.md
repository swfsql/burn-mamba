# files.md

A per-file **signature reference** for `burn-mamba`: what each important file
contains, the items it defines and their purpose, and the key
decisions/characteristics worth knowing before editing it. For the high-level
architecture and the full annotated tree, see `CLAUDE.md`; for notation, see
its [Notation](./CLAUDE.md#notation) section.

Conventions: tensor names carry shape suffixes (`_bsh` = `[batch, sequence,
nheads]`, etc.). Dimension keys: `b`atch, `s`equence, `d`_model,
`i`=d_inner, `h`eads, `p`er_head_dim, `r`=state_rank, `m`=mimo_rank,
`n`chunks, `g`roups, `l`=chunk_len, `a`=num_rope_angles, `v`=conv_dim,
`k`=conv_kernel.

Files not listed here are either trivial `mod.rs` re-export glue or test-only.

As of Burn 0.22 the high-level `Tensor` (and therefore every `Module`) is pinned
to the global `Dispatch` backend, so library types are **no longer
backend-generic** — `struct Mamba2`, `Mamba2Cache`, … carry no `<B>`. The
backend is chosen at runtime by the `Device`. The only items still generic over a
backend `B` are the custom-backward internals (`F<B, D>` / `Mask<B>` primitive
wrappers, the `Backward<B, _>` nodes, and the `Autodiff<B>` ext impls).

---

## Crate root

### `src/lib.rs`
Crate root. Declares the feature-gated modules (`mamba1`/`2`/`3`, `schedule`,
`utils`) and the top-level `prelude`, and carries the crate-level overview doc.
Enables `#![warn(missing_docs)]`, so any undocumented public item surfaces as a
build warning. Defines two crate-wide sanity constants:
- `pub const DENY_NAN: bool` / `pub const DENY_INF: bool` — when `true`, the
  `utils::sanity` guards actually run NaN/Inf checks (both `false` by default,
  so the checks are no-ops in release).

### `Cargo.toml`
Manifest. Edition 2024, depends on `burn` `0.22.0-pre.1` (pinned
to a git rev; the `extension` feature provides `#[backend_extension]`). The
`[features]` section is the source of truth for backend selection
(`backend-*`), the model toggles (`mamba1`/`2`/`3`, all default-on), `autodiff`
(required by mamba2/3), and `cubecl`/`fusion`. With the Dispatch architecture,
each `backend-*` feature simply enables the corresponding `burn/<backend>`;
several may be compiled in at once (the old `_dev-has-backend` compile-error gate
is gone — examples pick the runtime `Device` from the enabled feature). A `TODO`
notes the `dev-*`/`backend-*` features leaking into the lib for cubecl needs.

---

## Mamba-1 (`src/mamba1/`)

The original selective SSM. Mamba-1 is the simplest family: **no SSD, no
backend-ext trait, no bidirectional support.** It does share the `Layers`
stack with virtual-layer scheduling and the cache-threaded network with
Mamba-2/3.

### `mamba1/mamba1.rs`
The core block. Read its module header for the file-local notation table.
- `struct Mamba1` — params: `in_proj` (`d_model → 2·d_inner`), depthwise
  causal `conv1d`, `x_proj` (`d_inner → dt_rank + 2·state_rank`), `dt_proj`
  (`dt_rank → d_inner`), `a_log [d_inner, state_rank]`, `d [d_inner]`, `out_proj`.
- `struct Mamba1Config` — `d_model`, `state_rank` (16), `conv_kernel` (4),
  `expand` (2), dt init params, `has_proj_bias`/`has_conv_bias`, optional
  `dt_rank`/`d_inner`. PyTorch-style uniform init; A initialised from
  `arange(1..=state_rank)` then `log`. (Field names mirror Mamba-2/3.)
- `forward(x, cache) -> (y, cache)` — in_proj → causal conv (left-padded from
  `cache.conv_bik`) → SiLU → `ssm()` selective scan → SiLU gate → out_proj.
  Threads a `Mamba1Cache` so a sequence can be processed in segments.
- `ssm()` / `selective_scan()` — the **sequential** scan: ZOH discretisation for
  A (`exp(Δ·A)`), Euler for B (`Δ·B`); loops over the sequence stacking outputs.
- `step()` / `ssm_step()` / `selective_scan_step()` — single-token recurrence
  sharing the same cache (rolling conv window + `ssm_bir` state update).

Key characteristic: A is input-**independent** here (unlike Mamba-2's per-head
scalar and Mamba-3's data-dependent A).

### `mamba1/cache.rs`
- `struct Mamba1Cache` — `conv_bik [b, i, k]` (conv window) + `ssm_bir [b, i, r]`
  (SSM state), plain `Tensor`s (updated by reassignment). `sanity()` guards.
- `struct Mamba1Caches` — `Vec<Mamba1Cache>`, one per **virtual** layer;
  helpers `caches_len`/`from_vec`/`into_options`/`from_options`.
- `*Config` factories (zero-init), `new_from_block_config` (`n_real_caches`).

### `mamba1/layer.rs`
- `struct Mamba1Layers` — the stack: `n_real_layers` weight sets,
  `n_virtual_layers: Option<(usize, Schedule)>` (`module(skip)`), `real_layers`,
  `ignore_first/last_residual`. `forward`/`step` loop over virtual indices,
  mapping each to a real layer via the schedule, each with its own cache.
  No SSD path (Mamba-1 has no backend-ext trait).
- `struct Mamba1Layer` — `RmsNorm` + `Mamba1` block. `forward`/`step` apply
  Pre-LN residual `y = x·residual_scale + Block(norm(x))`, threading a cache.

### `mamba1/network.rs`
- `struct Mamba1Network` — `embedding` → `Mamba1Layers` → `norm_f` → optional
  tied/untied `lm_head`. Targets the `state-spaces/mamba-130m` text models.
- `forward(tokens, caches) -> (logits, caches)` and `step(token, caches)` (both
  take `Option<Mamba1Caches>` and return the updated caches); vocab padding to
  `pad_vocab_size_multiple`; weight-tied LM head when `missing_lm_head`.

---

## Mamba-2 (`src/mamba2/`)

Structured State Space Duality (SSD). Adds the pluggable chunkwise SSD, the
backend-ext trait, and bidirectional support (the `Layers` stack with virtual
scheduling is now shared with Mamba-1).

### `mamba2/mamba2.rs`
The core SSD block. **Read its module header** for the full SSD math (recurrence
↔ 1-semiseparable attention duality) and its file-local notation table.
- `struct Mamba2` — `in_proj` (`d_model → d_inner + conv_dim + nheads`),
  depthwise causal `conv1d` (over `conv_dim = d_inner + 2·ngroups·state_rank`),
  per-head `dt_bias_h`/`a_log_h`/`d_h`, gated `norm` (`RmsNormGated`),
  `out_proj`, optional learnable `init_state_hpr`, plus `state_rank`/`ngroups`.
- `struct Mamba2Config` — fully documented fields with Paper/Python aliases:
  `d_model`, `state_rank` (128), `conv_kernel` (4), `expand` (2), `per_head_dim`
  (64), `ngroups` (1), `a_init_range`, `dt_*`, `dt_limit`, bias flags,
  `has_learnable_init_state`. Derived: `d_inner`, `nheads`, `conv_dim`.
- `forward(input, cache, ssd_path) -> (out, cache)` — 8 steps: in-proj split
  `[z | xbc | dt_raw]` → causal conv+SiLU → split `(x, B, C)` → discretise
  (`Δ=softplus(dt_raw+dt_bias)`, `Ā=exp(Δ·A)` with `A=-exp(a_log)`, `B̄=Δ·B`) →
  zero-pad to a multiple of `chunk_len` → GQA-expand B/C → **run selected SSD
  path** → gated RMSNorm with `z` → out-proj. Zero-pad is exact (Δ=0⇒Ā=1,B̄=0).
- `step(input, cache)` — pure recurrence `hₜ = Āₜ hₜ₋₁ + B̄ₜ xₜᵀ`,
  `yₜ = Cₜᵀ hₜ + D xₜ`, with manual conv-window slide. Only `forward` touches the
  SSD path (dispatched through `Mamba2BackendExt`, impl'd for `Dispatch`); `step`
  is the plain recurrence.

### `mamba2/cache.rs`
- `struct Mamba2Cache` — `conv_bvk [b, v, k]` (rolling conv window) +
  `ssm_bhpr [b, h, p, r]` (O(p·r) compressed state — the SSM memory advantage
  over a growing KV-cache). `sanity()` guards.
- `struct Mamba2Caches` — `Vec`, one per **virtual** layer; helpers
  `into_options`/`from_options` (take-without-clone threading).
- `*Config` factories, `new_from_block_config`. Zero-init is correct (no prior
  tokens; `h₀=0`).

### `mamba2/layer.rs`
- `struct Mamba2Layers` — the stack: `n_real_layers` weight sets,
  `n_virtual_layers: Option<(usize, Schedule)>` (`module(skip)`), `real_layers`,
  `ignore_first/last_residual`. `forward`/`step` loop over virtual indices,
  mapping each to a real layer via the schedule, each with its own cache.
- `struct Mamba2Layer` — single `RmsNorm` + `Mamba2`; Pre-LN residual with a
  `residual_scale` (0.0 suppresses first/last residual).

### `mamba2/network.rs`
Full LM: `embedding → Mamba2Layers → norm_f → lm_head`, tied/untied head, vocab
padding. `forward(tokens, caches, ssd_path)` / `step(token, caches)`.

### `mamba2/ssd/ssd_path.rs`
The SSD selector and input bundle.
- `enum Mamba2SsdPath { Minimal(Option<usize>), Serial(_), SerialRecalculated(_) }`
  — each carries an optional `chunk_len`. `Default = SerialRecalculated(None)`.
  Variant docs map to the reference Triton kernels (`ssd_minimal.py`, the 5
  K1–K5 kernels, `ssd_combined.py`).
- `struct Mamba2SsdInput` — pre-processed inputs: `x_bnlhp`, `dt_bnlh`,
  `a_decay_h` (= A, negative), `b_bnlhr`/`c_bnlhr` (already GQA-expanded),
  `d_h`, `initial_state_bhpr`, optional `init_state_hpr`. `sanity()`.
- `optimal_default(state_rank, per_head_dim)` ≈ `√(r·p)` rounded to a multiple of
  32, capped 512. `core_optimal`/`chunked_optimal`/… convenience ctors,
  `chunk_len_or_optimal`, and `run(input)` which dispatches to the three impls.

### `mamba2/ssd/minimal.rs`
- `Mamba2SsdInput::ssd_minimal() -> (y_bnlhp, final_state_bhpr)`. The 4-step
  chunkwise algorithm, **autodiff backward** (no custom op):
  1. Intra-chunk `Y_diag = (L ∘ C Bᵀ)·X` with `L = exp(segsum(Δ·A))`.
  2. Per-chunk state (zero-init) via decayed outer products.
  3. Inter-chunk state scan (segsum over chunks) → per-chunk init state + final.
  4. State→output `Y_off = exp(a_cum)·(C·h_prev)`; `Y = Y_diag + Y_off + D·X`.
  Steps 1/2/4 are batched GEMMs (tensor cores); step 3 is a short scan over
  `nchunks`. The clearest reference implementation of the SSD.

### `mamba2/ssd/serial.rs`
- `Mamba2SsdInput::ssd_serial() -> (y, final_state)`. Same math, but a **serial
  loop over chunks** plus matmuls (mirrors the 5 Triton kernels). **Autodiff
  backward.** Lower peak memory than `minimal` for the intra-chunk products.

### `mamba2/ssd/serial_recalculated/`
The memory-efficient path with a **custom backward**.
- `serial_recalculated.rs` — defines the **`Mamba2BackendExt` trait**
  (`fn ssd_serial_recalculated(...) -> (FloatTensor, FloatTensor)` on raw
  primitives) whose **default body replicates `ssd_serial`** for plain backends;
  and `Mamba2SsdInput::ssd_serial_recalculated()` which lowers to primitives and
  calls the trait. Asserts `init_state_hpr.is_none()` (not yet supported here).
- `backward.rs` — the `impl Mamba2BackendExt for Autodiff<B, C>`: registers a
  `Backward<B, 7>` node (`CombinedKernelsBackward`) that **recomputes** the
  forward intermediates during the backward pass instead of stashing them
  (saves ~⅓ training memory). Feature-gated by `autodiff`.
- `combined_backward.rs` — the actual gradient math (`CombinedGrads`) for all 7
  inputs, recompute-based.
- The two outputs `(y, final_state)` are flattened into one tracked 1-D tensor
  (see `utils/combined_grad.rs`) because `prep.finish` accepts a single tensor.

### `mamba2/bidi/naive/`
- `layer.rs` — `Mamba2BidiLayers` (stack over `BidiSchedule`) and
  `Mamba2BidiLayerPair` (a straight → block + a reversed ← block via `flip` on
  the sequence axis, then flip-back), merged per pair. Forward-only (training /
  non-autoregressive tasks).
- `output_merge.rs` — `enum OutputMerge { Mean(NoOp), CatLinear(Linear) }` (+
  `OutputMergeConfig`): how the two directions combine (average vs.
  `Linear([2·d_model → d_model])` over the concatenation).

---

## Mamba-3 (`src/mamba3/`)

Extends Mamba-2 with **trapezoidal discretisation**, **data-dependent RoPE** on
B/C, and **MIMO** rank expansion. The defining structural choice: **two
interchangeable SSD pathways** (`double_ssd` / `single_ssd`) selected by the
cache variant.

### `mamba3/mamba3.rs`
The core block + the pathway dispatcher. **Read its module header** for the full
combined math (trapezoid + RoPE + MIMO) and the file-local notation table.
- `struct Mamba3` — `in_proj` (size = `d_in_proj`, see below), per-head
  `dt_bias_h`/`d_h`, `b_norm`/`c_norm` (QK-Norm `RmsNorm` over `state_rank`),
  `b_bias_hmr`/`c_bias_hmr` (`[h, m, r]`, init 1), optional MIMO
  `mimo_x_hmp`/`mimo_z_hmp`/`mimo_o_hmp` (`Some` only when `mimo_rank>1`),
  optional `out_norm` (`RmsNormGated`), `out_proj`, optional `init_state_hpr`,
  plus `state_rank`/`ngroups`/`num_rope_angles`/`rope_dim`/`mimo_rank`.
- `struct Mamba3Config` — `d_model`, `state_rank` (128, **must be even** for RoPE
  pairing), `expand`, `per_head_dim`, `ngroups`, `mimo_rank` (1=SISO), `a_floor`,
  `dt_*`, `dt_limit`, `rope_fraction` (0.5 or 1.0), `has_proj_bias`,
  `has_learnable_init_state`, `has_outproj_norm`. Derived: `d_inner`, `nheads`,
  `rope_dim`, `num_rope_angles`, and
  `d_in_proj = 2·d_inner + 2·ngroups·state_rank·mimo_rank + 3·nheads + num_rope_angles`
  (split `[z | x | B_raw | C_raw | dd_dt | dd_A | λ_raw | θ]`).
- `forward(input, cache, ssd_path)` / `step(input, cache)` — **dispatch by cache
  variant**: missing cache ⇒ defaults to **SingleSsd**; otherwise matches
  `Mamba3Cache::{DoubleSsd, SingleSsd}` and delegates to
  `forward_double_ssd`/`forward_single_ssd` (or `step_double_ssd`/
  `step_single_ssd`).

### `mamba3/mod.rs`
Wires the backend-ext trait aggregation. Defines `Mamba3BackendExt:
Backend + Mamba3DoubleSsdBackendExt + Mamba3SingleSsdBackendExt`, then uses the
`decl_ssd_autodiff_backend_ext!` and `impl_ssd_backend_ext_for_burn_backends!`
macros (`utils/backend_macros.rs`) to emit per-backend impls and the autodiff
marker. The `backwards` submodule blanket-impls the aggregate for `Autodiff<B>`.

### `mamba3/helpers.rs`
Rank-generic helpers shared by both pathways **and** both modes (forward at
rank 5, step at rank 4):
- `struct TrapezoidCoeffs<D>` + `trapezoidal_coefficients(...)` —
  `Δ=softplus(dd_dt+dt_bias)`, `A=-softplus(dd_A)` clamped `≤ -a_floor`,
  `da=Δ·A`, `α=exp(da)`, `β=(1-λ)·Δ·α`, `γ=λ·Δ` with `λ=σ(λ_raw)`.
- `qk_norm_expand_bias(...)` — RmsNorm(over `state_rank`) → GQA-expand
  groups→heads → add `[h,m,r]` bias.
- `build_v_with_mimo(...)` — `v = x ⊙ mimo_x` inserting the `mimo_rank` axis;
  identity (size-1 axis) when SISO.

### `mamba3/cache.rs`
The pathway-tagged cache **enums** — the heart of the dual-pathway design.
- `enum Mamba3Cache { DoubleSsd(Mamba3DoubleSsdCache), SingleSsd(Mamba3SingleSsdCache) }`
  (per layer) and `enum Mamba3Caches { ... }` (per network).
- Conversions `From`/`Into` both ways, `double_ssd()`/`single_ssd()` extractors,
  `into_options`/`from_options`/`from_vec` (peeks the first element; **empty ⇒
  SingleSsd**). The enum lets one dispatch entry accept/return either family.
- **Cross-pathway cache conversions**: field-identity `From` impls between
  `Mamba3SingleSsdCache`↔`Mamba3DoubleSsdCache` (and the plural `…Caches`). Valid
  because at a sequence boundary `scaleₜ = γₜ`, so the single-ssd accumulator `h'`
  equals the double-ssd state `h`. Used by `step_single_ssd`.

### `mamba3/double_ssd/double_ssd.rs`
The **double-pass trapezoidal** forward/step (VikramLex-style) + the RoPE
utilities. Splits the trapezoid into two **standard** SSD calls:
γ-SSM (current token, scaled by γ) + β-SSM (previous token, scaled by β,
"shift-before-chunking"), summed. Simple/verifiable but ~2× SSD memory. Its
`step_double_ssd` recurrence is reused (via cache conversion) for single-ssd
decoding too.
- `forward_double_ssd(input, cache, Mamba3DoubleSsdPath)` — 11 steps: in-proj
  split → trapezoid coeffs → reshape x → QK-norm B/C → cumulative RoPE angles →
  build shifted (prev-token) inputs → scale by γ/β → pad → two
  `Mamba3DoubleSsdInput::run` calls → sum → unpad → D-skip + gate/gated-norm +
  MIMO rank merge → out-proj → update cache.
- `step_double_ssd(input, cache)` — single-token recurrence
  `hₜ = αₜ hₜ₋₁ + Σₘ βₜ Bₜ₋₁[m]⊗(xₜ₋₁⊙mimo_x) + Σₘ γₜ Bₜ[m]⊗(xₜ⊙mimo_x)`.
- `apply_rope(x, angles, rotate_pairwise)` — rotates the last dim in pairs.
  `rotate_pairwise=true` = interleaved/NeoX (SISO Triton); `false` =
  half-and-half/GPT-J (MIMO Tilelang).
- `apply_rope_partial(x, angles, rope_dim, rotate_pairwise)` — rotates only the
  first `rope_dim` entries (partial RoPE, `rope_fraction=0.5`); falls back to
  `apply_rope` when `rope_dim == state_rank`, and is the **identity** when
  `rope_dim == 0` (`rope_fraction=0`, RoPE disabled). Used by **both** pathways
  (single_ssd imports it from here).
- `wrap_angle(angles)` — reduces angles mod `2π` into `[−π, π]` with a `detach`ed
  offset (value-exact, gradient identity) before `sin`/`cos` and when storing the
  `cum_angle` accumulator, preserving low-bit-float precision. Used by **both**
  pathways.

### `mamba3/double_ssd/cache.rs`
- `struct Mamba3DoubleSsdCache` — `ssm_bhpr` (trapezoidal hidden state),
  `k_state_bmhr` (previous-token B per rank, for the β term), `v_state_bhp`
  (previous-token x), `cum_angle_bha` (cumulative RoPE angle). **No conv cache**
  (Mamba-3 has no short convolution). `+ Caches`/`*Config` factories.

### `mamba3/double_ssd/ssd/ssd_path.rs`
- `enum Mamba3DoubleSsdPath` — same three variants as Mamba-2, `From<Mamba3SsdPath>`.
- `struct Mamba3DoubleSsdInput` — **MIMO-first** SSD input: `v_bnlmhp`
  (already × γ or β), `da_bnlh` (= Δ·A), `b_bnlmhr`/`c_bnlmhr` (QK-normed,
  RoPE-applied, bias-added, per-head, per-rank), `initial_state_bhpr`, optional
  `init_state_hpr`. `run()` dispatches to `double_ssd_minimal/serial/serial_recalculated`.

### `mamba3/double_ssd/ssd/{minimal,serial,serial_recalculated/*}.rs`
The same three SSD algorithms as Mamba-2, **adapted to MIMO-first** inputs
(extra `mimo_rank` axis fused into the chunk reshape). `serial_recalculated/`
defines `Mamba3DoubleSsdBackendExt` (+ autodiff marker) and the custom recompute
backward, mirroring the Mamba-2 structure (`backward.rs` registers the node,
`combined_backward.rs` holds the gradient math).

### `mamba3/single_ssd/single_ssd.rs`
The **single-pass** trapezoidal forward (official Triton-SISO / Tilelang-MIMO
form), ≈ half the training memory of double-ssd.
- `forward_single_ssd(input, cache, Mamba3SingleSsdPath)` — one SSD call using
  `scaleₜ = γₜ + (1−λₜ₊₁)·Δₜ₊₁` as the key scale, a strict lower-triangular
  intra-chunk mask + same-step γ correction (inside the kernel), and a
  **boundary-β seed** `(1−λ₀)·Δ₀·Kₜ₋₁⊗xₜ₋₁` folded into the initial state from
  the cache. Saves last-token B/x and `cum_angle` into the cache.
- `step_single_ssd(...)` — converts the single-ssd cache to a double-ssd cache,
  runs `step_double_ssd` (the pure recurrent form, matching the official
  `mamba3_siso_step` / `mamba3_step_fn` kernels), then converts back. Lossless
  because the two accumulators coincide at the per-token boundary.

### `mamba3/single_ssd/cache.rs`
- `struct Mamba3SingleSsdCache` — same four fields as the double cache, **but
  `ssm_bhpr` carries different semantics**: the single-ssd accumulator
  `h'ₜ = αₜ h'ₜ₋₁ + scaleₜ Bₜ⊗xₜ`, giving correct output everywhere except the
  diagonal (patched by the in-kernel γ correction). The distinct type **prevents
  feeding a double-ssd cache into single-ssd mid-sequence** (which would silently
  corrupt state); at boundaries the accumulators coincide, so the explicit
  cross-pathway `From` conversions in `mamba3/cache.rs` are lossless.
  `+ Caches`/`*Config`.

### `mamba3/single_ssd/ssd/ssd_path.rs`
- `enum Mamba3SingleSsdPath` + `struct Mamba3SingleSsdInput` — like the
  double input but feeds **raw `v`** plus `gamma_bnlh` and `scale_bnlh` (the
  kernel applies the scaling internally, rather than the caller pre-scaling V).
  Defines `Mamba3SingleSsdBackendExt`. Same `minimal/serial/serial_recalculated`
  trio under `ssd/`.

### `mamba3/ssd_path.rs`
- `enum Mamba3SsdPath { Minimal(_), Serial(_), SerialRecalculated(_) }` — the
  **pathway-agnostic** algorithm selector exposed to users. `Default =
  SerialRecalculated(None)`. `From<Mamba3DoubleSsdPath>` and
  `From<Mamba3SingleSsdPath>` (and the reverse `From`s live in each sub-path) let
  the top-level path convert to whichever pathway the cache selects.

### `mamba3/rotation.rs`
- Self-contained **reference + verification** of the quaternion (`k = 4`)
  rotational state — the non-abelian generalisation of Mamba-3's data-dependent
  RoPE. Mamba-3 ships the abelian `SO(2)` case (`double_ssd.rs::apply_rope`,
  cumulative *angles* via `cumsum`); this module implements the next rung:
  per-step unit quaternions whose cumulative rotation lives in
  `SU(2) ⊂ SO(4)` (non-commuting → richer state-tracking, up to `NC¹`).
- Public ops: `quat_mul`/`quat_conj`/`quat_normalize` (algebra on the trailing
  `(w,x,y,z)` axis), `quat_from_scaled_axis` (data-dependent **materialise**
  step — axis·angle → unit quaternion via the exp map, the analogue of RoPE's
  `Δ·π·tanh(θ)`; identity at `Δ→0`), `quat_to_rot4` (the `4×4` left-isoclinic
  matrix), `quat_cumprod` (the **associative scan** that replaces RoPE's
  `cumsum`, with a cross-chunk **carry** — the analogue of `cum_angle`), and
  `rotate_state_rank_blocks` (apply a per-block quaternion to a `state_rank`
  axis, used as `B̄ = rotate(B, conj(Qcum))`).
- Together these are the `materialise → scan → apply` pipeline (the engine the
  abelian RoPE specialises to `cumsum` + sin/cos).
- Integration scaffolding (default-off): `RotationKind { Complex2D | Quaternion4D }`
  (a `Mamba3Config` field, default `Complex2D` ⇒ unchanged) and `RotationState
  { Angle(Tensor<3>) | Quaternion(Tensor<4>) }` (the cache accumulator variant,
  a `#[derive(Module)]` enum). `Mamba3Config::d_in_proj` branches via
  `num_rotation_channels` (`num_rope_angles` for Complex2D, `3·num_quat_blocks`
  quaternion generators for Quaternion4D). Forward/step don't branch yet —
  building a `Quaternion4D` block asserts "not yet wired" (next step: swap
  `RotationState` in for the caches' `cum_angle_bha` + wire forward/step).
- Key properties, proved by the tests: the RoPE *factoring*
  (`Cₜᵀ(Rₜ⋯Rᵢ₊₁)Bᵢ = C̄ₜᵀB̄ᵢ`) survives **non-commutativity** (so the
  scalar-decay SSD core is unchanged; only `cumsum`→scan changes), and the `k=2`
  single-axis restriction reproduces the **production** `apply_rope` exactly
  (cross-validation against the current pathway). NOT wired into the `Mamba3`
  block — it is a tested math reference for that larger change.

### `mamba3/layer.rs`
- `struct Mamba3Layers` / `Mamba3Layer` — same shape as Mamba-2 (virtual
  scheduling, `ignore_first/last_residual`, Pre-LN residual with
  `residual_scale`). Plus four free `make_zero_caches_{double,single}_ssd_{2d,3d}`
  helpers used to lazily allocate caches matching the chosen pathway.

### `mamba3/network.rs`
Full Mamba-3 LM (`embedding → Mamba3Layers → norm_f → tied/untied lm_head`,
vocab padding). `forward`/`step` thread `Mamba3Caches`.

### `mamba3/bidi/naive/{layer,output_merge}.rs`
- `Mamba3BidiLayers` / `Mamba3BidiLayerPair` — straight + reversed Mamba-3 passes
  merged per pair; stack uses `BidiSchedule`. The pair clones each layer's `norm`
  + `mamba_block` and an `OutputMerge`.
- `output_merge.rs` — `OutputMerge { Mean, CatLinear }` (identical role to the
  Mamba-2 one).

---

## Scheduling (`src/schedule.rs`)

Virtual-layer → real-weight index mapping (no tensors; pure index arithmetic).
- `enum Schedule { Cyclic, Stretched, Custom(Vec<usize>) }` with
  `real_idx(virtual_idx, virtual_len, real_len)`: `Cyclic` wraps
  (`v % real_len`), `Stretched` block-repeats (`v·real_len/virtual_len`),
  `Custom` indexes a table.
- `enum BidiSchedule { StridedCyclic, StridedStretched, SymmetricCyclic,
  SymmetricStretched, Custom }` — pairs forward/backward layers (even virtual
  indices = straight →, odd = reverse ←). Doc comments include worked examples.

---

## Utilities (`src/utils/`)

### `utils/mod.rs`
Per-dtype numerical helper used by the norms: `div_eps(dtype: DType) -> f32` (a
dtype-tuned epsilon between the raw underflow bound and machine epsilon, matched
on the runtime `DType`). Declares all util submodules. (Burn 0.22 dropped
`B::FloatElem`, so the old `<B>`-generic `div_eps`/`div_eps_f32`/`stable_max`
were replaced by this single `DType`-taking function.)

### `utils/rms_norm.rs` & `utils/rms_norm_gated.rs`
- `RmsNorm` — last-dim RMS normalisation with a learnable `gamma` (init 1). Used
  both as a generic norm and as Mamba-3's **QK-Norm** on B/C.
- `RmsNormGated` — RMSNorm fused with a SiLU gate; `norm_before_gate` toggles
  `norm(x)·σ(z)` vs `norm(x·σ(z))`. Mamba-2's output norm; Mamba-3's optional
  `out_norm`.
- Both have **fp16-safe paths**: instead of computing `x²` directly (overflows
  on large widths), they normalise against `max(|x|)` first. `config.epsilon` is
  retired (commented out) in favour of `div_eps`.

### `utils/silu.rs`, `utils/softplus.rs`, `utils/log_sigmoid.rs`
Custom activations Burn either lacks or needs fp16-stable variants of.
- `Silu` — `x·σ(x) = x/(1+exp(-x))` (a unit `Module`).
- `softplus`, `log_sigmoid` — numerically-stable implementations.

### `utils/segsum.rs`
- `segsum(x) -> [..., s, s]` — **stable segment sum** building the
  1-semiseparable mask: works in log-space as `cumsum[i] − cumsum[j]` (avoids
  underflow of long product chains), upper triangle set to `-∞` so `exp(...)`
  gives the strict-causal lower-triangular `L`. The backbone of `ssd_minimal`.

### `utils/gqa.rs`
- `gqa_expand_to_heads::<D, DP1>(t, group_dim, nheads)` — replicates each
  group's B/C slice across `heads_per_group = nheads/ngroups`. `DP1 = D+1` is a
  caller-supplied const (Rust can't yet express the constraint). Panics if
  `nheads % ngroups != 0`.

### `utils/split.rs`
- `split_into::<D, N>(t, [sizes; N], dim) -> [Tensor; N]` — array-typed
  `split_with_sizes`, enabling `let [z, x, b, c, ...] = split_into(...)`
  destructuring instead of `parts.next().unwrap()` chains. Used by every
  in-projection split.

### `utils/combined_grad.rs`
- `flatten_pair` / `unflatten_pair` — flatten `(y, final_state)` into one 1-D
  tracked tensor and split it back. Needed because Burn's `prep.finish` takes a
  single tracked tensor; used by every `serial_recalculated` custom backward.

### `utils/backend_macros.rs`
- `impl_ssd_backend_ext_for_burn_backends!($trait)` — emits the per-backend
  "use the default impl" blocks (NdArray, Flex, LibTorch, Remote, CubeCL,
  Fusion), each feature-gated.
- `decl_ssd_autodiff_backend_ext!($auto, $ext, ...)` — declares the autodiff
  marker trait + blanket impl for `Autodiff<B>`. Cuts the per-family
  backend-ext boilerplate (Mamba-3 uses these; Mamba-2 hand-rolls the same).

### `utils/fprim.rs`
- `F<B, const D>` — a rank-tagged newtype over a backend's `FloatTensor<B>`
  primitive that mirrors the slice of the high-level `Tensor` method API
  (`matmul`/`permute`/`reshape`/`squeeze_dim`/`unsqueeze_dim(s)`/`expand`/`exp`/
  `sum_dim`/`cumsum`/`slice`/`narrow`/`triu`/`mask_fill`/`stack`/`cat`/`zeros`/
  `full`, plus `+ - * neg`). Because Burn 0.22's `Tensor` is pinned to the global
  `Dispatch` backend, both the trait default forward bodies
  (`*/serial_recalculated/serial_recalculated.rs`, run under a generic `B`) and
  the custom `Backward<B, _>` nodes cannot build a `Tensor`; this wrapper keeps
  the forward K-kernels and the recompute-backward gradient math
  (`*/serial_recalculated/combined_backward.rs`) reading like the original tensor
  code while operating on `B::float_*` primitives.
- `Mask<B>` + `san(&F)` — companion bool-mask wrapper for `mask_fill`, and the
  primitive analogue of `sanity::sanity`.

### `utils/sanity.rs`
- `sanity(&t)` / `sanity_nan(&t)` — optional NaN/Inf guards, **no-ops unless**
  `DENY_NAN`/`DENY_INF` (in `lib.rs`) are `true`. Called liberally (`san(&...)`)
  through the forward paths.

### `utils/scheduler.rs`
LR schedulers for the examples (copied/adapted from `burn-jepa`):
`enum Lr { CosineAnnealing(CosineAnnealingLr), Constant(ConstantLr) }` with
`get_lr(step)` (cosine annealing + linear warmup, or constant).

### `utils/loss/{bce,cross_entropy,mse}.rs`
Loss functions (binary cross-entropy, cross-entropy, mean squared error) used by
the example training loops.

### `utils/test_helpers.rs` (test-only)
- `max_abs_diff(a, b) -> f32` and the `check_grads_match_two_paths!` macro,
  shared by the SSD-path agreement tests in Mamba-2 and Mamba-3 (assert the
  three SSD algorithms agree on outputs **and** gradients).

---

## Examples (`examples/`)

The examples are **Dispatch-only**: no module carries a backend type generic.
The backend is selected at runtime by constructing a `Device`; autodiff and dtype
are device properties (`device.clone().autodiff()`,
`device.configure((FloatDType::F16, IntDType::I32))`).

### `examples/common/device.rs`
Runtime dtype selection + record format (replaces the old compile-time
`backend.rs`). The backend itself is picked by `Device::default()` directly in
each `main.rs` — it resolves to the enabled `backend-*` feature (each enables the
matching `burn/<backend>`), honouring the `BURN_DEVICE` env override and a
built-in priority list when several are compiled in. This module provides:
- `configure_dtype(&mut Device)` — installs fp16/i32 device defaults under
  `dev-f16`; a no-op otherwise.
- `RecorderTy` — the on-disk record format (`NamedMpkFileRecorder`,
  half-precision under `dev-f16`, full otherwise).
- `FloatElement` — the host scalar matching the runtime float dtype
  (`burn::tensor::f16` under `dev-f16`, else `f32`); used for `to_vec` reads.

### `examples/common/cli.rs`
`struct AppArgs` + parsing (training/inference flags, artifact dir, config
load/save). Manages the train→infer flow and config persistence (model/optim
save/load via `RecorderTy`, all non-generic over the backend). The `HELP` const
is the CLI help text.

### `examples/common/model/{mod,model,bidi}.rs`
- `mod.rs` — the `ModelConfigExt` trait (`init(&Device) -> Self::Model`; no
  backend generic).
- `model.rs` — the example networks `MyMamba2Network` / `MyMamba3Network`
  (`in_proj` → `{2,3}Layers` → `out_proj`) with `*NetworkConfig`, plus the
  `*_block_config` / `*_layers_config` builder helpers.
- `bidi.rs` — bidirectional variants (`MyMamba{2,3}BidiNetwork`) wrapping the
  `…BidiLayers`.

### `examples/common/training.rs` and `examples/common/mnist/`
`TrainingConfig` (epochs/batch/LR/seed + `AdamWConfig`) and the
`optimizer_config(dtype)` helper (AdamW defaults with a dtype-sized epsilon); the
sequential-MNIST dataset loader (`MnistDataset`/`MnistBatcher`, pixels as a
length-784 sequence). The training dataloader is built with
`.set_device(autodiff_device)` so batches match the model's (autodiff) backend;
the validation loader uses the inner device.

### `examples/fibonacci/`
Smallest demo: a tiny **Mamba-2** model on a fibonacci-like synthetic sequence.
`main.rs` (`launch(app_args)`: picks the `Device`, derives the autodiff device,
wires configs + train/infer), `model.rs` (`model_config()`), `dataset.rs`,
`training.rs`, `inference.rs` (next-value prediction; reads values as
`FloatElement`). Runs end-to-end on fp32 and `dev-f16` (the latter goes NaN —
accepted fp16 instability).

### `examples/mnist-class/`
Sequential-MNIST digit classifier using **Mamba-3** (cosine-annealing LR).
`main.rs` (`launch(app_args)`) / `model.rs` / `training.rs`; **inference is a
stub** ("not yet implemented").

