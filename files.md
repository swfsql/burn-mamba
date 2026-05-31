# files.md

A per-file **signature reference** for `burn-mamba`: what each important file
defines and the non-obvious decisions worth knowing before editing it. For the
high-level architecture and the file tree see `CLAUDE.md`; for notation see its
[Notation](./CLAUDE.md#notation) section.

Shape-suffix keys: `b`atch, `s`equence, `d`_model, `i`=d_inner, `h`eads,
`p`er_head_dim, `r`=state_rank, `m`=mimo_rank, `n`chunks, `g`roups,
`l`=chunk_len, `a`=num_rope_angles, `v`=conv_dim, `k`=conv_kernel.

Files not listed are trivial `mod.rs` glue or test-only. As of Burn 0.22 the
high-level `Tensor` (every `Module`) is pinned to the global `Dispatch` backend,
so library types are **not backend-generic** (`struct Mamba2`, `Mamba2Cache`, …
carry no `<B>`); the backend is a runtime `Device`. Only the custom-backward
internals stay generic over `B` (`F<B,D>` / `Mask<B>` wrappers, the
`Backward<B,_>` nodes, the `Autodiff<B>` ext impls).

---

## Crate root — `src/lib.rs`

Declares the feature-gated family modules + `modules`/`utils`, the `prelude`, and
the crate overview. Enables `#![warn(missing_docs)]`. Defines the crate-wide
guards `DENY_NAN` / `DENY_INF` (both `false` by default ⇒ the `sanity` checks
compile to no-ops).

---

## Mamba-1 (`src/mamba1/`)

The simplest family: **no SSD, no backend-ext trait.** Layer stack / network /
bidi come from `src/modules/`; this module is just the block + its cache.

### `mamba1/mamba1.rs`
The core block (read its header for the local notation table). A is input-**independent**
here (unlike Mamba-2's per-head scalar and Mamba-3's data-dependent A).
- `Mamba1` — `in_proj` (`d_model→2·d_inner`), depthwise causal `conv1d`, `x_proj`
  (`d_inner→dt_rank+2·state_rank`), `dt_proj`, `a_log [d_inner, state_rank]`,
  `d`, `out_proj`. A init from `arange(1..=state_rank).log()`.
- `Mamba1Config` — `d_model`, `state_rank` (16), `conv_kernel` (4), `expand` (2),
  dt init, bias flags, optional `dt_rank`/`d_inner`. Field names mirror Mamba-2/3.
- `forward(x, cache) -> (y, cache)` — in_proj → causal conv (left-padded from
  `cache.conv_bik`) → SiLU → `selective_scan` (ZOH A, Euler B) → SiLU gate → out_proj.
- `step()` — single-token recurrence sharing the cache.

### `mamba1/cache.rs`
- `Mamba1Cache` — `conv_bik [b,i,k]` window + `ssm_bir [b,i,r]` state (plain
  `Tensor`s, updated by reassignment).
- `Mamba1Caches` — `Vec`, one per **virtual** layer; `into_options`/`from_options`
  threading + `*Config` zero-init factories.

---

## Mamba-2 (`src/mamba2/`)

SSD. Adds the pluggable chunkwise scan + the backend-ext trait.

### `mamba2/mamba2.rs`
The core SSD block — **read its header** for the recurrence↔1-semiseparable-attention
duality and the local notation table.
- `Mamba2` — `in_proj` (`d_model → d_inner + conv_dim + nheads`), depthwise causal
  `conv1d` over `conv_dim = d_inner + 2·ngroups·state_rank`, per-head
  `dt_bias_h`/`a_log_h`/`d_h`, gated `norm`, `out_proj`, optional `init_state_hpr`.
- `Mamba2Config` — `d_model`, `state_rank` (128), `conv_kernel` (4), `expand` (2),
  `per_head_dim` (64), `ngroups` (1), `dt_*`/`dt_limit`, bias flags. Derived:
  `d_inner`, `nheads`, `conv_dim`.
- `forward(input, cache, ssd_path)` — in-proj split `[z|xbc|dt_raw]` → conv+SiLU →
  split `(x,B,C)` → discretise → zero-pad to a `chunk_len` multiple (exact:
  Δ=0⇒Ā=1,B̄=0) → GQA-expand B/C → run SSD path → gated RMSNorm(z) → out-proj.
- `step(input, cache)` — pure recurrence with manual conv-window slide; only
  `forward` touches the SSD path (via `Mamba2BackendExt`).

### `mamba2/cache.rs`
- `Mamba2Cache` — `conv_bvk [b,v,k]` window + `ssm_bhpr [b,h,p,r]` (the O(p·r)
  compressed state — the SSM memory advantage over a growing KV-cache).
- `Mamba2Caches` — `Vec`, one per virtual layer; zero-init is correct (`h₀=0`).

### `mamba2/ssd/ssd_path.rs`
- `Mamba2SsdPath { Minimal | Serial | SerialRecalculated }(Option<chunk_len>)`,
  `Default = SerialRecalculated(None)`. Variant docs map to the reference Triton
  kernels.
- `Mamba2SsdInput` — pre-processed `x_bnlhp`, `dt_bnlh`, `a_decay_h` (= A, negative),
  GQA-expanded `b_bnlhr`/`c_bnlhr`, `d_h`, `initial_state_bhpr`, optional
  `init_state_hpr`.
- `optimal_default(state_rank, per_head_dim)` ≈ `√(r·p)`, rounded to mult-of-32,
  capped 512. `run(input)` dispatches to the three impls.

### `mamba2/ssd/minimal.rs`
- `ssd_minimal() -> (y_bnlhp, final_state_bhpr)` — the clearest reference: 4 steps
  (intra-chunk `Y_diag = (L∘CBᵀ)X` with `L = exp(segsum(Δ·A))`; per-chunk state
  via decayed outer products; inter-chunk state scan; state→output). Steps 1/2/4
  are batched GEMMs, step 3 a short scan over `nchunks`. **Autodiff backward.**

### `mamba2/ssd/serial.rs`
- `ssd_serial()` — same math, a **serial loop over chunks** + matmuls (mirrors the
  5 Triton kernels); lower peak memory than `minimal`. **Autodiff backward.**

### `mamba2/ssd/serial_recalculated/`
The memory-efficient path with a **custom backward** (recomputes forward
intermediates instead of stashing them, ~⅓ less training memory).
- `serial_recalculated.rs` — defines `Mamba2BackendExt` (default body = `ssd_serial`
  on primitives); `ssd_serial_recalculated()` lowers to primitives. Asserts
  `init_state_hpr.is_none()` (unsupported here).
- `backward.rs` — `impl Mamba2BackendExt for Autodiff<B>`: registers the
  `CombinedKernelsBackward` node (`autodiff`-gated).
- `combined_backward.rs` — the recompute-based gradient math for all 7 inputs.
- `(y, final_state)` are flattened into one tracked tensor (`utils/combined_grad.rs`).

---

## Mamba-3 (`src/mamba3/`)

Extends Mamba-2 with **trapezoidal discretisation**, **data-dependent RoPE** on
B/C, and **MIMO**. The defining choice: **two interchangeable SSD pathways**
(`double_ssd` / `single_ssd`) selected by the cache variant.

### `mamba3/mamba3.rs`
Core block + pathway dispatcher — **read its header** for the combined math.
- `Mamba3` — `in_proj` (size `d_in_proj`), per-head `dt_bias_h`/`d_h`, `b_norm`/`c_norm`
  (QK-Norm over `state_rank`), `b_bias_hmr`/`c_bias_hmr` (init 1), optional MIMO
  `mimo_{x,z,o}_hmp` (only `mimo_rank>1`), optional `out_norm`, `out_proj`,
  optional `init_state_hpr`.
- `Mamba3Config` — `d_model`, `state_rank` (128, **even** for RoPE pairing),
  `expand`, `per_head_dim`, `ngroups`, `mimo_rank` (1=SISO), `a_floor`, `dt_*`,
  `rope_fraction` (0.5/1.0), `rotation: RotationKind`, bias/init/out-norm flags.
  Derived `d_in_proj = 2·d_inner + 2·ngroups·state_rank·mimo_rank + 3·nheads +
  num_rotation_channels` (split `[z|x|B_raw|C_raw|dd_dt|dd_A|λ_raw|θ]`).
- `forward`/`step` — **dispatch by cache variant**: missing ⇒ SingleSsd; else
  match `Mamba3Cache::{DoubleSsd,SingleSsd}` and delegate.

### `mamba3/mod.rs`
Defines `Mamba3BackendExt: Mamba3DoubleSsdBackendExt + Mamba3SingleSsdBackendExt`
and uses the `backend_macros` to emit per-backend impls + the autodiff marker.

### `mamba3/helpers.rs`
Rank-generic helpers shared by both pathways and both modes:
- `trapezoidal_coefficients` → `Δ/A/da/α/β/γ` (`λ = σ(λ_raw)`).
- `qk_norm_expand_bias` — RmsNorm(over `state_rank`) → GQA-expand → add `[h,m,r]` bias.
- `build_v_with_mimo` — `v = x ⊙ mimo_x` inserting the `mimo_rank` axis (identity
  size-1 axis when SISO).

### `mamba3/cache.rs`
The pathway-tagged cache **enums** (heart of the dual-pathway design):
`Mamba3Cache { DoubleSsd | SingleSsd }` (per layer), `Mamba3Caches` (per network).
Conversions both ways, `double_ssd()`/`single_ssd()` extractors,
`into_options`/`from_options`/`from_vec` (**empty ⇒ SingleSsd**). The
**cross-pathway** `From` impls are field-identity, valid because at a sequence
boundary `scaleₜ = γₜ` so the single-ssd accumulator `h'` equals the double-ssd
state `h`; used by `step_single_ssd`.

### `mamba3/double_ssd/double_ssd/mod.rs`
The **double-pass** forward/step (VikramLex-style) + the RoPE utilities. Splits
the trapezoid into γ-SSM (current, ×γ) + β-SSM (previous, ×β, shift-before-chunking),
summed. Simple/verifiable but ~2× SSD memory; its `step_double_ssd` recurrence is
reused (via cache conversion) for single-ssd decoding too.
- `forward_double_ssd(input, cache, path)` — in-proj split → trapezoid coeffs →
  QK-norm B/C → RoPE → build shifted prev-token inputs → scale γ/β → pad → two
  `Mamba3DoubleSsdInput::run` → sum → unpad → D-skip + gate/MIMO merge → out-proj.
- `apply_rope` / `apply_rope_partial` — rotate the last dim in pairs;
  `rotate_pairwise=true` interleaved/NeoX (SISO), `false` half-and-half/GPT-J (MIMO).
  `apply_rope_partial` rotates only the first `rope_dim` (identity when
  `rope_dim==0`). Used by **both** pathways (single_ssd imports them).
- `wrap_angle` — mod-`2π` into `[−π,π]` with a `detach`ed offset (value-exact,
  gradient identity), before sin/cos and when storing `cum_angle`.

### `mamba3/double_ssd/cache.rs`
- `Mamba3DoubleSsdCache` — `ssm_bhpr` (trapezoidal hidden state), `k_state_bmhr`
  (prev-token B per rank, β term), `v_state_bhp` (prev-token x), `rotation`
  (`RotationState`). **No conv cache.**

### `mamba3/double_ssd/ssd/ssd_path.rs` + `ssd/{minimal,serial,serial_recalculated}`
- `Mamba3DoubleSsdPath` (three variants, `From<Mamba3SsdPath>`); `Mamba3DoubleSsdInput`
  is **MIMO-first**: `v_bnlmhp` (already ×γ or ×β), `da_bnlh` (= Δ·A), `b/c_bnlmhr`
  (QK-normed, RoPE-applied, bias-added), `initial_state_bhpr`, optional `init_state_hpr`.
- The three SSD algorithms mirror Mamba-2, adapted to MIMO-first inputs (extra
  `mimo_rank` axis fused into the chunk reshape). `serial_recalculated/` defines
  `Mamba3DoubleSsdBackendExt` + the custom recompute backward.

### `mamba3/single_ssd/single_ssd/mod.rs`
The **single-pass** forward (official Triton-SISO / Tilelang-MIMO form), ≈½ the
training memory.
- `forward_single_ssd(input, cache, path)` — one SSD call with
  `scaleₜ = γₜ + (1−λₜ₊₁)·Δₜ₊₁` key scale, a strict lower-triangular intra-chunk
  mask + same-step γ correction (in-kernel), and a **boundary-β seed**
  `(1−λ₀)·Δ₀·Kₜ₋₁⊗xₜ₋₁` folded into the initial state from the cache.
- `step_single_ssd` — converts to a double-ssd cache, runs `step_double_ssd`,
  converts back; lossless because the accumulators coincide at the boundary.

### `mamba3/single_ssd/cache.rs`
- `Mamba3SingleSsdCache` — same four fields as the double cache, **but `ssm_bhpr`
  has different semantics**: `h'ₜ = αₜ h'ₜ₋₁ + scaleₜ Bₜ⊗xₜ` (correct everywhere
  except the diagonal, patched by the in-kernel γ correction). The distinct type
  prevents feeding a double-ssd cache into single-ssd mid-sequence; the boundary
  `From` conversions (in `mamba3/cache.rs`) are lossless.

### `mamba3/single_ssd/ssd/ssd_path.rs` + `ssd/*`
- `Mamba3SingleSsdPath` + `Mamba3SingleSsdInput` — like the double input but feeds
  **raw `v`** plus `gamma_bnlh` and `scale_bnlh` (the kernel scales internally).
  Defines `Mamba3SingleSsdBackendExt`; same minimal/serial/serial_recalculated trio.

### `mamba3/ssd_path.rs`
- `Mamba3SsdPath { Minimal | Serial | SerialRecalculated }` — the
  **pathway-agnostic** selector exposed to users (`Default = SerialRecalculated(None)`).
  `From<Mamba3DoubleSsdPath>` / `From<Mamba3SingleSsdPath>` let it convert to
  whichever pathway the cache selects.

### `mamba3/rotation/` (`mod.rs` + `tests.rs`)
Self-contained reference + verification of the quaternion (`k=4`) rotational
state — the **non-abelian** generalisation of Mamba-3's RoPE (per-step unit
quaternions in `SU(2) ⊂ SO(4)`; non-commuting ⇒ richer state-tracking).
- Algebra: `quat_mul`/`quat_conj`/`quat_normalize`; `quat_from_scaled_axis` (the
  data-dependent **materialise** — axis·angle → unit quaternion via the exp map,
  the analogue of `Δ·π·tanh(θ)`); `quat_to_rot4`; `quat_cumprod` (the associative
  **scan** replacing RoPE's `cumsum`, with a cross-chunk **carry**; Hillis–Steele,
  `O(log seq)` depth — `forward` instead calls `quat_scan`'s recompute-backward
  variant); `rotate_state_rank_blocks` (`B̄ = rotate(B, conj(Qcum))`).
- Block wiring: `RotationKind { Complex2D | Quaternion4D }` (a `Mamba3Config` field,
  default Complex2D) and `RotationState { Angle(Tensor<3>) | Quaternion(Tensor<4>) }`
  (the cache accumulator). `num_rotation_channels` = `num_rope_angles` (Complex2D)
  or `3·num_quat_blocks` (Quaternion4D). forward/step branch via
  `rotate_bc_forward`/`rotate_bc_step`. Quaternion4D runs on **both** SSD pathways.
- Tests prove the RoPE *factoring* (`Cₜᵀ(Rₜ⋯Rᵢ₊₁)Bᵢ = C̄ₜᵀB̄ᵢ`) survives
  non-commutativity (so the scalar-decay SSD core is unchanged), and the `k=2`
  restriction reproduces the production `apply_rope` exactly.

### `mamba3/quat_scan/`
The **memory-efficient** quaternion cumprod scan: a custom recompute backward
mirroring the SSD `SerialRecalculated` design.
- `quat_scan.rs` — `Mamba3QuatScanBackendExt` (default body runs the scan via the
  `Quat` **struct-of-arrays** helper: `(w,x,y,z)` as separate tensors so the
  Hamilton product is fusible element-wise math with no per-step `narrow`/`cat`),
  `quat_prefix_product_soa`, and `quat_cumprod_recalculated(q, init) -> (cum,
  final_carry)`. Single-output node (`final_carry = cum[:, −1]`, a thin autodiff
  slice — no two-output `combined_grad` plumbing).
- `backward.rs` — the `Autodiff<B>` `Backward<B,2>` node: saves only `q`+`init`,
  recomputes the prefix product `P`, evaluates the **exact unit-quaternion VJP**
  with parallel ops only (`S[t] = Σ_{s≥t} conj(Pₛ)⊗d_cum[s]`, `G = P⊗S`,
  `d_q[t] = G[t]⊗conj(cum[t−1])`, `d_init = S[0]`). No token loop. Tests assert
  equality with `quat_cumprod` on values + grads.

---

## Composition modules (`src/modules/`)

The single home for composing per-family blocks into layers and full networks
(generic over `M = Mamba1 | Mamba2 | Mamba3`), plus the shared neural building
blocks. Replaces the former per-family `layer.rs`/`network.rs`/`bidi/` copies.

### `modules/mod.rs` — traits + the family-tagged path
- `trait MambaBlock: Module` — the block interface the generic layers delegate to:
  associated `Cache` / `Caches: CacheStack` / `SsdPath` (Mamba-1 uses `()`),
  `block_forward`/`block_step`, `zero_caches_{2d,3d}`.
- `trait MambaBlockConfig: Config` — `d_model()` + `init_block(device)`, letting
  the builders construct a stack without knowing the family.
- `enum MambaSsdPath { Mamba1 | Mamba2(_) | Mamba3(_) }` + `mamba{2,3}_default()`.

### `modules/layer.rs` / `modules/layers.rs`
- `Layer<M>` — Pre-LN residual: `y = x·residual_scale + M(RMSNorm(x))`.
- `Layers<M>` — the stack: `n_real_layers` weight sets, `n_virtual_layers:
  Option<(usize, Schedule)>`, `ignore_first/last_residual`. `forward`/`step` loop
  over virtual indices, mapping each to a real layer via the schedule, each with
  its own cache. `LayersBuilder` constructs it (non-serde).

### `modules/network.rs`
- `LatentNetwork<M>` — `in_proj (input_size→d_model) → Layers<M> → out_proj` for
  feature/regression tasks.
- `VocabNetwork<M>` — `Embedding → Layers<M> → norm_f → LM head`; head tied
  (embeddingᵀ when `missing_lm_head`) or untied; vocab padded to
  `pad_vocab_size_multiple`. Both build on the **same** `Layers<M>` core.
- `enum MambaLatentNet` / `MambaVocabNet` (`#[derive(Module)]`) + matching
  `#[derive(Config)]` `*Config` enums (concrete per-family — Config derive is not
  generic-aware). `forward`/`step` dispatch and **panic on a family-mismatched
  cache/ssd_path**. Plain `*Builder` factories with `with_class_{tokens,latents}`.

### `modules/bidi.rs`
- `BidiLayerPair<M>` — a straight (→) + reversed (← via `flip`, then flip-back)
  pass; `BidiLayers<M>` stacks pairs with a `BidiSchedule`. Forward-only
  (non-autoregressive); available to all three families.
- `enum OutputMerge { Mean(NoOp) | CatLinear(Linear) }` — direction merge
  (average vs `Linear([2·d_model→d_model])` over the concatenation).
- `enum MambaBidiLayers` (+ `Config`) — the runtime family-selection wrapper.

### `modules/cache.rs`
- `trait CacheStack` — the per-network cache *collection* interface
  (`slot_count`/`into_slots`/`from_slots`); implemented for `Mamba{1,2,3}Caches`.
- `enum MambaCaches { Mamba1(_) | Mamba2(_) | Mamba3(_) }` — **plain runtime state**
  (not a `Module`), threaded through `forward`/`step`.

### `modules/norm/{rms_norm,rms_norm_gated}.rs`
- `RmsNorm` — last-dim RMS norm with learnable `gamma`; also Mamba-3's **QK-Norm**.
- `RmsNormGated` — RMSNorm fused with a SiLU gate (`norm_before_gate` toggles
  `norm(x)·σ(z)` vs `norm(x·σ(z))`). Mamba-2's output norm; Mamba-3's optional one.
- Both have **fp16-safe paths**: normalise against `max(|x|)` first instead of
  computing `x²` (which overflows on large widths). Epsilon comes from `div_eps`.

### `modules/activation/{silu,softplus,log_sigmoid}.rs`
Custom activations Burn lacks or needs fp16-stable variants of: `Silu`
(`x·σ(x)`, a unit `Module`), `softplus`, `log_sigmoid`.

### `modules/misc/{gqa,segsum,split,sanity}.rs`
- `gqa_expand_to_heads::<D, DP1>(t, group_dim, nheads)` — replicate each group's
  B/C across `nheads/ngroups` (`DP1 = D+1` is a caller const; panics if not divisible).
- `segsum(x) -> [..,s,s]` — **stable** segment sum (log-space `cumsum[i]−cumsum[j]`,
  upper triangle `-∞`) building the 1-semiseparable mask; backbone of `ssd_minimal`.
- `split_into::<D, N>(t, [sizes; N], dim) -> [Tensor; N]` — array-typed
  `split_with_sizes`, enabling `let [z, x, b, c, …] = split_into(...)`.
- `sanity` / `sanity_nan` — NaN/Inf guards, no-ops unless `DENY_NAN`/`DENY_INF`.

### `modules/loss/{bce,cross_entropy,mse}.rs`
Loss functions used by the example training loops.

---

## Utilities (`src/utils/`)

Lower-level plumbing (and `div_eps`).

### `utils/mod.rs`
- `div_eps(dtype: DType) -> f32` — a per-dtype epsilon for safe division, chosen
  per float format as the geometric mean (log10) of a scaled min-exponent and
  machine epsilon (comfortably above the underflow floor, negligible vs typical
  activations). Used by the norms. Panics on non-float dtypes.

### `utils/class/` (`mod.rs` + `tests.rs`)
Learnable `[CLS]`-style class tokens/latents spliced into the sequence.
`ClassToken` markers live on the networks, `ClassLatent` on layer containers (see
the CLAUDE.md class-tokens section). Each container stores markers as
`#[module(skip)]` metadata + one `Option<Param<Tensor<2>>>` (`[num_markers,
width]`). The `ClassMarker` trait + `insert_class_markers` place `Start | Middle
| End | Custom(index)` relative to length `L` (Start@0, Middle@`L/2`, End@`L`,
Custom@`index`, last; ties keep `Vec` order). `step` injects via position cursors;
`Start`/`Custom` inject in `step`, `Middle`/`End` panic for the cursored level.

### `utils/schedule/` (`mod.rs` + `tests.rs`)
Virtual-layer → real-weight index mapping (pure index arithmetic).
- `Schedule { Cyclic | Stretched | Custom(Vec) }` — `real_idx(v, virtual_len,
  real_len)`: `Cyclic` wraps, `Stretched` block-repeats, `Custom` indexes a table.
- `BidiSchedule { StridedCyclic | StridedStretched | SymmetricCyclic |
  SymmetricStretched | Custom }` — pairs forward/backward layers (even virtual = →,
  odd = ←).

### `utils/scheduler/`
LR schedulers for the examples: `Lr { CosineAnnealing | Constant }` with
`get_lr(step)` (cosine annealing + linear warmup, or constant).

### `utils/backend_macros.rs`
- `impl_ssd_backend_ext_for_burn_backends!($trait)` — emits the per-backend
  "use the default impl" blocks (each feature-gated).
- `decl_ssd_autodiff_backend_ext!(...)` — declares the autodiff marker trait +
  blanket impl for `Autodiff<B>`. Cuts the per-family backend-ext boilerplate.

### `utils/combined_grad.rs`
`flatten_pair` / `unflatten_pair` — flatten `(y, final_state)` into one tracked
1-D tensor and split it back; needed because Burn's `prep.finish` takes a single
tracked tensor.

### `utils/fprim.rs`
- `F<B, const D>` — a rank-tagged newtype over `FloatTensor<B>` mirroring the
  slice of the `Tensor` method API used by the kernels (matmul/permute/reshape/…,
  `+ - * neg`). Because Burn 0.22's `Tensor` is pinned to `Dispatch`, neither the
  trait default forward bodies (run under a generic `B`) nor the `Backward<B,_>`
  nodes can build a `Tensor`; this wrapper keeps the forward K-kernels and the
  recompute-backward gradient math reading like tensor code over `B::float_*`.
- `Mask<B>` + `san(&F)` — bool-mask wrapper for `mask_fill`, and the primitive
  analogue of `sanity`.

### `utils/test_helpers.rs` (test-only)
`max_abs_diff(a, b)` + the `check_grads_match_two_paths!` macro, shared by the
SSD-path agreement tests (the three algorithms agree on outputs **and** grads).
