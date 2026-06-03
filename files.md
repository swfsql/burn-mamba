# files.md

A per-file **signature reference**: what each important file defines and the
non-obvious decisions worth knowing before editing it. For the architecture and file
tree see `CLAUDE.md`; for notation see its [Notation](./CLAUDE.md#notation) section.
The detailed per-family math lives in the `mamba2.rs` / `mamba3.rs` module headers.

Keep this file minimal (see CLAUDE.md → *Documentation Maintenance*): one terse entry
per important file, no changelog. Trivial `mod.rs` glue and `tests.rs` are omitted.

Shape keys: `b`atch `s`equence `d`_model `i`=d_inner `h`eads `p`er_head_dim
`r`=state_rank `m`=mimo_rank `n`chunks `g`roups `l`=chunk_len `a`=num_rope_angles
`v`=conv_dim `k`=conv_kernel.

> Burn 0.22 pins the high-level `Tensor` (every `Module`) to the global `Dispatch`
> backend, so library types are **not** backend-generic (no `<B>`). Only the
> custom-backward internals stay generic over `B` (`F<B,D>`/`Mask<B>`, the
> `Backward<B,_>` nodes, the `Autodiff<B>` ext impls).

---

## `src/lib.rs`
Feature-gated module decls + `prelude` + crate overview. `#![warn(missing_docs)]`.
Crate guards `DENY_NAN`/`DENY_INF` (both `false` ⇒ the `sanity` checks are no-ops).

---

## Mamba-1 (`src/mamba1/`) — simplest family: no SSD, no backend-ext trait

- **`mamba1.rs`** — `Mamba1` block + `Mamba1Config`. A is input-**independent** (unlike
  Mamba-2/3). `forward`: in_proj → causal conv (left-padded from `cache.conv_bik`) →
  SiLU → sequential `selective_scan` (ZOH A, Euler B) → SiLU gate → out_proj.
  `step` shares the cache. A init from `arange(1..=state_rank).log()`.
- **`cache.rs`** — `Mamba1Cache` (`conv_bik` window + `ssm_bir` state) / `Mamba1Caches`
  (`Vec`, one per virtual layer; `into_options`/`from_options`, zero-init factories).

## Mamba-2 (`src/mamba2/`)

- **`mamba2.rs`** — `Mamba2` + `Mamba2Config` (`state_rank` 128, `per_head_dim` 64,
  `ngroups` 1, `expand` 2). `forward` per CLAUDE.md; only `forward` touches the SSD
  path (via `Mamba2BackendExt`), `step` is the pure recurrence with a manual
  conv-window slide. Optional learnable `init_state_hpr`.
- **`cache.rs`** — `Mamba2Cache` = `conv_bvk` window + `ssm_bhpr` (the O(p·r) compressed
  state — the memory win over a growing KV-cache). Zero-init correct (`h₀=0`).
- **`ssd/ssd_path.rs`** — `Mamba2SsdPath{Minimal|Serial|SerialRecalculated}(Option<chunk>)`,
  `Default = SerialRecalculated(None)`; `Mamba2SsdInput` (pre-processed
  `x_bnlhp`/`dt_bnlh`/`a_decay_h`/GQA-expanded `b,c_bnlhr`/…); `optimal_default ≈ √(r·p)`;
  `run()` dispatches.
- **`ssd/minimal.rs`** — clearest reference: 4 steps (intra-chunk `Y_diag=(L∘CBᵀ)X`,
  `L=exp(segsum(Δ·A))`; per-chunk state; inter-chunk scan; state→output). Autodiff bwd.
- **`ssd/serial.rs`** — same math as a serial chunk loop (mirrors Triton K1–K5); lower
  peak memory. Autodiff bwd.
- **`ssd/serial_recalculated/`** — custom backward (recomputes intermediates, ~⅓ less
  memory). `serial_recalculated.rs` defines `Mamba2BackendExt` (default body = `ssd_serial`
  on primitives; asserts `init_state_hpr.is_none()`); `backward.rs` registers the
  `Autodiff<B>` node; `combined_backward.rs` is the recompute gradient math (7 inputs).

## Mamba-3 (`src/mamba3/`)

- **`mamba3.rs`** — `Mamba3` + `Mamba3Config` (`state_rank` **even** for RoPE pairing;
  `mimo_rank` 1=SISO; `rope_fraction`; `rotation: RotationKind`; `a_floor`). Fields:
  QK-norm `b_norm`/`c_norm`, `b/c_bias_hmr` (init 1), optional `mimo_{x,z,o}_hmp` and
  `out_norm`. Derived `d_in_proj` (split `[z|x|B_raw|C_raw|dd_dt|dd_A|λ_raw|θ]`).
  `forward`/`step` **dispatch by cache variant** (missing ⇒ SingleSsd).
- **`mod.rs`** — `Mamba3BackendExt: Mamba3DoubleSsdBackendExt + Mamba3SingleSsdBackendExt`,
  wired via `backend_macros`.
- **`helpers.rs`** — rank-generic, shared by both pathways/modes: `trapezoidal_coefficients`
  (`Δ/A/da/α/β/γ`, `λ=σ`), `qk_norm_expand_bias`, `build_v_with_mimo`.
- **`cache.rs`** — the pathway-tagged `Mamba3Cache{DoubleSsd|SingleSsd}` / `Mamba3Caches`
  enums; extractors; `from_vec`/`from_options` (**empty ⇒ SingleSsd**). The cross-pathway
  `From` impls are field-identity, valid because at a boundary `scaleₜ=γₜ` so single-ssd
  `h'` equals double-ssd `h`.
- **`ssd_path.rs`** — pathway-agnostic `Mamba3SsdPath` (`Default=SerialRecalculated(None)`);
  `From` both sub-paths so it converts to whichever pathway the cache selects.

### `mamba3/double_ssd/`
- **`double_ssd/mod.rs`** — `forward_double_ssd`/`step_double_ssd` + the RoPE utilities.
  Splits the trapezoid into γ-SSM (current ×γ) + β-SSM (prev ×β, shift-before-chunking),
  summed; ~2× SSD memory. `step_double_ssd` is reused (via cache conversion) for
  single-ssd decoding. `apply_rope`/`apply_rope_partial` (rotate last-dim pairs;
  interleaved/NeoX SISO vs half-and-half/GPT-J MIMO; identity when `rope_dim==0`) and
  `wrap_angle` are used by **both** pathways.
- **`cache.rs`** — `Mamba3DoubleSsdCache`: `ssm_bhpr` (trapezoidal state), `k_state_bmhr`
  (prev-token B, β term), `v_state_bhp` (prev-token x), `rotation` (`RotationState`). No conv.
- **`ssd/ssd_path.rs` + `ssd/*`** — `Mamba3DoubleSsdPath`; `Mamba3DoubleSsdInput` is
  **MIMO-first** (`v_bnlmhp` already ×γ/β, `da_bnlh`, `b/c_bnlmhr`). Same three algorithms
  as Mamba-2 with the `mimo_rank` axis fused into the chunk reshape;
  `serial_recalculated/` defines `Mamba3DoubleSsdBackendExt` + custom backward.

### `mamba3/single_ssd/`
- **`single_ssd/mod.rs`** — `forward_single_ssd`: one SSD call with key scale
  `scaleₜ = γₜ + (1−λₜ₊₁)·Δₜ₊₁`, strict-lower-triangular intra-chunk mask + same-step γ
  correction (in-kernel), and a **boundary-β seed** folded into the initial state.
  `step_single_ssd` converts to a double-ssd cache, runs `step_double_ssd`, converts back.
- **`cache.rs`** — `Mamba3SingleSsdCache`: same four fields but `ssm_bhpr` carries
  `h'ₜ = αₜh'ₜ₋₁ + scaleₜ Bₜ⊗xₜ` (correct except the diagonal, patched in-kernel). The
  distinct type prevents mixing a double-ssd cache into single-ssd mid-sequence.
- **`ssd/ssd_path.rs` + `ssd/*`** — `Mamba3SingleSsdPath` + `Mamba3SingleSsdInput` (raw `v`
  + `gamma_bnlh` + `scale_bnlh`, scaled in-kernel); `Mamba3SingleSsdBackendExt`; same trio.

### `mamba3/rotation/` (`mod.rs`)
The quaternion (`k=4`) **non-abelian** generalisation of RoPE (`SU(2) ⊂ SO(4)`).
Algebra (`quat_mul`/`conj`/`normalize`), `quat_from_scaled_axis` (data-dependent
materialise via the exp map), `quat_cumprod` (associative **scan** replacing `cumsum`,
with a cross-chunk carry), `rotate_state_rank_blocks` (`B̄ = rotate(B, conj(Qcum))`).
Wiring: `RotationKind{Complex2D|Quaternion4D}` (config) + `RotationState{Angle|Quaternion}`
(cache); forward/step branch via `rotate_bc_forward`/`rotate_bc_step`; runs on both
pathways. Tests: the RoPE factoring survives non-commutativity, and `k=2` reproduces
the production `apply_rope`.

### `mamba3/quat_scan/`
Memory-efficient cumprod scan (recompute backward, like SSD `SerialRecalculated`).
**`quat_scan.rs`**: `Mamba3QuatScanBackendExt` (default body uses the `Quat`
struct-of-arrays helper — `(w,x,y,z)` separate so the Hamilton product is fusible
element-wise math, no per-step `narrow`/`cat`) + `quat_cumprod_recalculated(q,init) ->
(cum, final_carry)` (single-output node; `final_carry = cum[:,−1]`). **`backward.rs`**:
`Backward<B,2>` saving only `q`+`init`, recomputing the prefix product, exact
unit-quaternion VJP with parallel ops only.

---

## Composition modules (`src/modules/`)

Generic over `M = Mamba1|Mamba2|Mamba3`; the single home for layer/network composition
plus shared NN blocks.

- **`mod.rs`** — `trait MambaBlock` (assoc. `Cache`/`Caches: CacheStack`/`SsdPath`,
  `block_forward`/`block_step`, `zero_caches_{2d,3d}`; Mamba-1's `SsdPath=()`),
  `trait MambaBlockConfig` (`d_model()`+`init_block`), and `enum MambaSsdPath`
  (`Mamba1|Mamba2(_)|Mamba3(_)` + `mamba{2,3}_default()`).
- **`layer.rs`** — `Layer<M>`: Pre-LN residual `y = x·residual_scale + M(RMSNorm(x))`.
- **`layers.rs`** — `Layers<M>`: `n_real_layers` weight sets, `n_virtual_layers:
  Option<(usize, Schedule)>`, `ignore_first/last_residual`, `residuals: Residuals`; loops
  virtual→real per the schedule, each with its own cache. `LayersBuilder`
  (`with_residuals`).
- **`multi_gate.rs`** — Multi-Gate Residuals: `enum Residuals{Standard|MultiGate}` (+
  `ResidualsConfig`) for `Layers`. `MultiGateResidual` (one per layer) gates `n_stream`
  streams toward the layer output then attention-pools them; point-wise over `(b,s)` so
  `forward`==`step`. Independent sigmoid gate only. Math in the module header.
- **`network.rs`** — `LatentNetwork<M>` (linear in/out) and `VocabNetwork<M>` (embedding →
  `norm_f` → tied/untied LM head, vocab padded). Both build on the same `Layers<M>`.
  Runtime enums `MambaLatentNet`/`MambaVocabNet` (+ concrete `*Config` enums — Config
  derive is not generic-aware); `forward`/`step` **panic on a family-mismatched
  cache/path**. `*Builder`s carry `with_class_{tokens,latents}`; the `*Config` enum
  variants carry a `residuals: ResidualsConfig` (plain additive vs Multi-Gate).
- **`bidi.rs`** — `BidiLayerPair<M>` (straight + reversed-via-`flip`) and `BidiLayers<M>`
  (stacks pairs with a `BidiSchedule`); `OutputMerge{Mean(NoOp)|CatLinear(Linear)}`;
  runtime `MambaBidiLayers`. Forward-only.
- **`cache.rs`** — `trait CacheStack` (collection iface `slot_count`/`into_slots`/
  `from_slots`, impl'd for `Mamba{1,2,3}Caches`) + `enum MambaCaches` (**plain runtime
  state**, not a `Module`).
- **`norm/`** — `RmsNorm` (also Mamba-3 QK-Norm) + `RmsNormGated` (RMSNorm × SiLU gate,
  `norm_before_gate` toggle). **fp16-safe**: normalise against `max(|x|)` to avoid `x²`
  overflow; epsilon from `div_eps`.
- **`activation/`** — `Silu`, `softplus`, `log_sigmoid` (fp16-aware variants Burn lacks).
- **`misc/`** — `gqa_expand_to_heads` (group→head replicate; `DP1=D+1` caller const),
  `segsum` (stable log-space 1-semiseparable mask; backbone of `ssd_minimal`),
  `split_into` (array-typed `split_with_sizes` → `let [z,x,b,c,…]=…`), `sanity` guards.
- **`loss/`** — bce, cross_entropy, mse (example training).

## Utilities (`src/utils/`)

- **`mod.rs`** — `div_eps(dtype) -> f32`: per-dtype safe-division epsilon (geometric mean
  of a scaled min-exponent and machine epsilon). Used by the norms.
- **`class/`** — learnable `[CLS]`-style tokens/latents. `ClassToken` (networks),
  `ClassLatent` (layer containers); markers stored as `#[module(skip)]` + one
  `Option<Param<Tensor<2>>>`. `ClassMarker` + `insert_class_markers` place
  `Start|Middle|End|Custom` relative to length `L` (Start@0, Middle@L/2, End@L,
  Custom@idx; ties keep `Vec` order). `step` injects via cursors (`Start`/`Custom` only;
  `Middle`/`End` panic for the cursored level).
- **`schedule/`** — `Schedule{Cyclic|Stretched|Custom}` (`real_idx`) and
  `BidiSchedule{Strided*/Symmetric*/Custom}` (even virtual = →, odd = ←).
- **`scheduler/`** — `Lr{CosineAnnealing|Constant}` (`get_lr(step)`; cosine + warmup).
- **`backend_macros.rs`** — `impl_ssd_backend_ext_for_burn_backends!` (per-backend default
  blocks) + `decl_ssd_autodiff_backend_ext!` (autodiff marker + `Autodiff<B>` blanket).
- **`combined_grad.rs`** — `flatten_pair`/`unflatten_pair`: `(y, final_state)` into one
  tracked tensor and back (`prep.finish` takes a single tensor).
- **`fprim.rs`** — `F<B, const D>`: rank-tagged `FloatTensor<B>` newtype mirroring the
  `Tensor` method API, so the generic-`B` forward kernels and `Backward<B,_>` nodes
  (which can't build a `Dispatch` `Tensor`) read like tensor code over `B::float_*`.
  `Mask<B>` + `san(&F)` accompany it.
- **`test_helpers.rs`** (test-only) — `max_abs_diff` + `check_grads_match_two_paths!`,
  shared by the SSD-path agreement tests.
