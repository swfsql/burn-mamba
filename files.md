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
  `muon_projections()` (feature `optim`) names the Muon-eligible weights and their
  column seams: `in_proj [x|res]`, `x_proj [dt*|B|C]`, `out_proj` (`*` = AdamW's).
- **`cache.rs`** — `Mamba1Cache` (`conv_bik` window + `ssm_bir` state) / `Mamba1Caches`
  (`Vec`, one per virtual layer; `into_options`/`from_options`, zero-init factories).

## Mamba-2 (`src/mamba2/`)

- **`mamba2.rs`** — `Mamba2` + `Mamba2Config` (`state_rank` 128, `per_head_dim` 64,
  `ngroups` 1, `expand` 2). `forward` per CLAUDE.md; only `forward` touches the SSD
  path (via `Mamba2BackendExt`), `step` is the pure recurrence with a manual
  conv-window slide. Optional learnable `init_state_hpr`.
  `muon_projections()`: `in_proj [z|x|B|C|dt*]` (`xbc` split further — the conv is
  shared, the linear map is not), `out_proj`.
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
  `mimo_rank` 1=SISO; `rope_fraction` `0.5|1` (default 1, full); `rotation: RotationKind`;
  `rotation_range` (default 2, the per-step bound in half-turns per unit Δ, applied to
  **each** quaternion factor — both defaults ship the full rotation, and the reference's
  narrower `1`/`0.5` are asked for explicitly); `a_floor`). `rotation_spec()` bundles the
  three rotation fields; `num_rotation_blocks()` = `num_quat_blocks · quat_factors` (the
  projection/scan block axis, doubled for `Rotor4D`) drives `num_rotation_channels()`;
  `zero_rotation_state()` is the one fresh-cache accumulator, shared by every pathway.
  Under `Real1D` every rotation count is `0`; `init` asserts any other kind turns ≥ 1 pair,
  and `muon_projections()` omits the (then absent) rotation segment. Fields:
  QK-norm `b_norm`/`c_norm`, `b/c_bias_hmr` (init 1), optional `mimo_{x,z,o}_hmp` and
  `out_norm`. Derived `d_in_proj` (split `[z|x|B_raw|C_raw|dd_dt|dd_A|λ_raw|θ]`),
  mirrored by `muon_projections()` as `in_proj [z|x|B|C|dt*|A*|λ*|rotation]` + `out_proj`.
  `forward`/`step` **dispatch by cache variant** (missing ⇒ SingleSsd).
  Two performance-only `mimo_rank == 1` knobs (default on, `#[module(skip)]`, identical
  values/grads): `siso_specialization` for the chunkwise γ-correction (threaded down as a
  field of the SSD input bundle, into the ext trait and the `Backward` node's state) and
  `siso_specialization_decode` for the per-token sites (`use_siso_decode_kernels()`).
  Split because the first wins on every backend and the second only on GPU.
- **`mod.rs`** — `Mamba3BackendExt: Mamba3DoubleSsdBackendExt + Mamba3SingleSsdBackendExt`,
  wired via `backend_macros`.
- **`helpers.rs`** — rank-generic, shared by both pathways/modes: `trapezoidal_coefficients`
  (`Δ/A/da/α/β/γ`, `λ=σ`), `qk_norm_expand_bias`, `build_v_with_mimo`, `mimo_outer_sum`
  (`Σₘ v[m]⊗k[m]` state contribution; step + boundary seed; `_siso` broadcast vs `_mimo`
  matmul, per `siso_specialization_decode`), `split_rotation_channels` (peels the in-proj's
  trailing rotation columns, `None` under `Real1D`; it cannot be one more entry in the main
  `split_into` because `split_with_sizes` **drops** a zero-length segment). Non-obvious: the
  `A` floor is `-softplus(x).clamp(a_floor, ∞)` — the clamp must bind the **positive**
  softplus before the unary minus (`A ≤ −a_floor` ⇒ `α < 1`); clamping after negation
  instead pins `A ≡ +a_floor` (data-independent growth).
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
  single-ssd decoding; it is factored through pub(crate) `StepProjection`/`step_project`
  (in-proj → coeffs → QK-norm, pre-rotation; its `rot_ba` is `None` under `Real1D`),
  `step_readout` (state×C einsum, `_siso`/`_mimo` branches) and `step_finish`
  (D-skip, gate/gated-norm, MIMO aggregation, out-proj), shared with
  `step_constant`. `apply_rope`/`apply_rope_partial` (rotate last-dim pairs;
  interleaved/NeoX SISO vs half-and-half/GPT-J MIMO; `rope_dim > 0` required) and
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
  + `gamma_bnlh` + `scale_bnlh`, scaled in-kernel, + `siso_specialization`);
  `Mamba3SingleSsdBackendExt`; same trio.
- **`ssd/diag.rs`** — `y_diag_correction`, the same-step γ term all three algorithms add
  back over the strict-lower mask; branches on `mimo_rank == 1 && siso_specialization` (at
  1 the `m×m` Gram is a scalar, so both matmuls collapse to a reduction + broadcast).
  `serial_recalculated/diag.rs` is the `F<B,D>` twin plus the analytic backward
  (`DiagGrads`, `d_gamma` whole not partial); the flag reaches it through the ext-trait
  method and the `Backward` node's `State`, so backward replays the forward's branch.
  Both keep the two branches separately callable for the `m=1` comparison tests.

### `mamba3/rotation/` (`mod.rs`)
"RoPE" here is the *transition's* imaginary part (`hₜ = αₜRₜhₜ₋₁`) factored onto B/C, not
a positional code — see the `mamba3.rs` header. This module holds the **non-abelian**
generalisations: quaternion (`k=4`, `SU(2)`) and the full `SO(4)`.
Algebra (`quat_mul`/`conj`/`normalize`), `quat_from_scaled_axis` (data-dependent
materialise via the exp map), `quat_cumprod` (associative **scan** replacing `cumsum`,
with a cross-chunk carry), `rotate_state_rank_blocks` (`B̄ = rotate(B, conj(Qcum))`) and
its two-sided `…_two_sided` / `rotate_blocks_two_sided_partial` / `split_rotor`.
Wiring: `RotationKind{Real1D|Complex2D|Quaternion4D|Rotor4D}` (config, `.quat_factors()`
= 1|1|1|2) + `RotationState{Real|Angle|Quaternion|Rotor}` (cache; `identity(kind,…)` builds
any of them, `quat_stack(kind)` unwraps either quaternion one and rejects a mismatched
variant; `Real` holds a tensor-less `NoRotation` — Burn's enum `Module` derive wants exactly
one field per variant)
+ `RotationSpec{kind,rope_dim,range}` (from `Mamba3::rotation_spec()`);
forward/step dispatch via `rotate_bc_forward`/`rotate_bc_step`; runs on both pathways.
`Rotor4D` is the two-sided `v ↦ q⊗v⊗p̄`, i.e. every element of `SO(4) ≅ (SU(2)×SU(2))/±1`.
The conjugation reverses the right-hand order **twice**, so `Tₜ = pₜ⊗⋯⊗p₁` is the *same*
left fold as `Qₜ`: both factors stack on one block axis and the generator split, the scan
and the normalisation run once over `2·blocks`, unbranched — only the application
(`B̄ᵢ = Qᵢ*⊗Bᵢ⊗Tᵢ`, conjugate on the left factor only) differs. It is the ceiling for
`k=4`, and the only kind containing the abelian one: `L_q` is *isoclinic* (both invariant
planes turn by the same angle), so `Quaternion4D` cannot express two independent per-pair
angles; two-sided the planes turn by `a∓b`. `p=q` gives the adjoint `SO(3)`.
`Real1D` is the trivial group at the bottom of the ladder, and it is structural, not a
zeroed knob: no in-projection channels (so `rotate_bc_forward`/`_step` take an
`Option<Tensor>` and hand `prev` straight back), no accumulator, `B`/`C` untouched. It is
what a rotation ablation selects — `rope_fraction` has no `0` setting.
`forward`, `step` and `step_infinite` all derive the per-step rotation from one pair of
helpers — `angle_increment` (`Δ·range·π·tanh(ϑ)`, shared across heads) and
`generator_increment` (`Δ·range·π·tanh(‖r‖)·r̂`, **per head** and block, channels laid out
`[head][block][xyz]` — for `Rotor4D` the block axis is `[left…|right…]`, so the bound
applies per factor and each independently sweeps all of `SU(2)`).
Non-obvious: the quaternion generator bounds its **magnitude** (`bound_rotation_vector`),
so the axis is the projection's direction at any scale — a per-component squash would
make the reachable set a cube and tie the axis to the projection's size; `safe_norm`
forms norms scale-free, since `‖r‖²` over raw in-projection channels overflows f16 at
`|r|≈250` and `∞` divides back to a *zero* rotation; the rotated width comes from
`rope_dim`, asserted equal to the accumulator's `blocks·4` rather than read off it, and a
partial quaternion rotation must land on whole 4-blocks;
`rotate_bc_forward` renormalises the scan's prefixes (`step` normalises per step) so a
drifted product turns B/C without rescaling them.
Tests: the RoPE factoring survives non-commutativity (and, against materialised
`L_q·R_p̄` matmuls, two-sidedness), `k=2` reproduces the production `apply_rope`,
`range=1` reproduces the reference angle, `Real1D` equals any kind whose generator is
zeroed (and projects/caches nothing), the half-turn is reachable with a live gradient (at
`range=1` it is not — f32's `tanh'` is exactly 0 there), a zero right generator reproduces
`Quaternion4D` on B, C and the accumulator, a shared axis turns the two planes by `a∓b`,
and gradient reaches the right factor's channels.

### `mamba3/quat_scan/`
Memory-efficient cumprod scan (recompute backward, like SSD `SerialRecalculated`).
**`quat_scan.rs`**: `Mamba3QuatScanBackendExt` (default body uses the `Quat`
struct-of-arrays helper — `(w,x,y,z)` separate so the Hamilton product is fusible
element-wise math, no per-step `narrow`/`cat`) + `quat_cumprod_recalculated(q,init) ->
(cum, final_carry)` (single-output node; `final_carry = cum[:,−1]`). **`backward.rs`**:
`Backward<B,2>` saving only `q`+`init`, recomputing the prefix product, exact
unit-quaternion VJP with parallel ops only.

### `mamba3/step_constant/` (`mod.rs`)
Constant-input closed form on `Mamba3`: `step_infinite` (stationary fixed-point
output; no cache in/out — the state orbits, the cumulative rotation cancels in the
readout, factor `(γ+βP⁻¹)(1−αP⁻¹)⁻¹`). Per RoPE pair that factor is
`(γ+βe^{−iθ̂})/(1−αe^{−iθ̂})`; per quaternion block the same in the abelian subalgebra
of the constant per-step `q`; unrotated channels use the scalar series `(β+γ)/(1−α)`.
`Rotor4D` leaves that commutative subalgebra, so its factor is the `ℝ⁴` resolvent by
**Cayley–Hamilton**: `c₁ = 2(k₁+k₂)`, `c₂ = 2+4k₁k₂` from the two plane cosines
`k₁,₂ = cos(a∓b)` (the half-angles alone fix them; the axes only fix the planes), the
cubic Horner-evaluated **on the vector** (no `4×4` materialised) and the determinant
formed factor-wise as `(1−α)²+2α(1−kᵢ)`, which the expanded quartic would lose to
cancellation. Denominators floored by `div_eps`. `Real1D` is the scalar series alone. All
four rotation kinds, both SSD pathways. The per-step
increment comes from `rotation`'s shared helpers, so the fixed point cannot drift from
the recurrence it is the limit of.

---

## Composition modules (`src/modules/`)

Generic over `M = Mamba1|Mamba2|Mamba3`; the single home for layer/network composition
plus shared NN blocks.

- **`mod.rs`** — `trait MambaBlock` (assoc. `Cache`/`Caches: CacheStack`/`SsdPath`,
  `block_forward`/`block_step`, `block_step_infinite` with a panicking default —
  only Mamba-3 overrides, `zero_caches_{2d,3d}`; Mamba-1's
  `SsdPath=()`),
  `trait MambaBlockConfig` (`d_model()`+`init_block`+`muon_projections()`), and
  `enum MambaSsdPath`
  (`Mamba1|Mamba2(_)|Mamba3(_)` + `mamba{2,3}_default()`). `MambaBlock`'s
  `ModuleDisplay + AutodiffModule` supertraits are what make the generic containers
  themselves `Module`/`AutodiffModule` (needed by `Layers::grad_horizon`); every family
  gets them from `#[derive(Module)]`.
- **`layer.rs`** — `Layer<M>`: Pre-LN `M(RMSNorm(x))`; the outer residual and class-latent
  insert are applied by `Layers`. `insert_latents(x, Option<&mut ClassCursor>)` is `pub`
  (a bare-`Layer` caller needs it too — `Layers` splices its layers' latents itself, since
  under MultiGate the rows must also enter the streams); `step` takes the same cursor and
  returns its last emitted token (`step_one`, the injection-free body, is `pub(crate)` for
  the cascade, which has already placed the markers `step`'s cursorless guard rejects). `prime(batch, cache, cursor)`
  steps the latents waiting for the next token without one, returning `Option<(delta,
  latent)>` — the row comes back with the delta because the residual is the caller's — and
  the cache untouched (`None` included) when none were waiting. Cursorless `step_infinite`
  mirrors `step`. Optional `norm2`+`mlp` (allocated together) add a second
  Pre-LN sub-block with a residual of its own; the methods therefore return the layer's
  **total delta** `h₁ + mlp(norm2(x + h₁))`, so `Layers`' single add yields both residuals
  — matching `mamba_ssm`'s `Block` when `d_intermediate > 0`. `mlp_residual` keeps the
  mixer-only path clone-free.
- **`mlp.rs`** — `GatedMlp` + `GatedMlpConfig`: SwiGLU `fc2(v ⊙ silu(g))` where
  `[v|g] = fc1(x)` (value half **first** — the checkpoint's fused layout). `hidden` rounds
  `d_intermediate` **up** to `multiple_of` (128). Rank-generic (point-wise).
- **`layers.rs`** — `Layers<M>`: `n_real_layers` weight sets, `n_virtual_layers:
  Option<(usize, Schedule)>`, `residuals`; loops virtual→real per the schedule, each with
  its own cache; owns the outer residual (`skip_residual`/`ignore_first/last_residual` —
  which govern only that outer add, not the feed-forward's inner one).
  `LayersBuilder` (`with_residuals`, `with_ignore_{first,last}_residual`, `with_mlp`,
  `with_grad_horizon`).
  `grad_horizon: Some(K)` back-propagates only the **top `K`** virtual layers; the rest
  run on `AutodiffModule::valid(self)` and are lifted back with `Tensor::from_inner` at
  the boundary, together with the token stream, its MultiGate stream sets and the
  prefix's cache slots (the suffix's stay tracked, so a cache carried between calls still
  transports gradient). One `grad_cut` decides for `forward` and the shared `cascade`
  alike, so all three entry points cut on the same virtual layers; it returns `0` off the
  autodiff backend — keyed on the **module's** device, `prime` having no input tensor —
  because `.inner()`/`.valid()` panic there. Under weight sharing a real layer straddles
  the cut and collects the gradient of its tracked applications only. The stack **input**
  is re-attached *straight-through* at the boundary — added exactly once, on the autodiff
  side (earlier would be a tracked input to the prefix: backend mismatch, and the end of
  the memory saving). The carry shadows `x`'s *shape*, taking zero rows wherever a prefix
  layer splices class latents, and rides the `cascade` the same way. Without it a cut
  severs the input's only path (it enters at the bottom) and `in_proj`/the embedding
  never trains. Those splice positions take a **ghost** row — value zero, from the
  *tracked* table — so a per-layer class latent below the cut trains too; class
  embeddings are learnable input rows at every level, while the layer's transform
  stays undifferentiated. Under MultiGate the carry goes to **every** stream as
  well as the pooled token (that is where the residual lives; the aggregator's
  weights sum to one, so this does not double-count).
  `forward`/`step`/`prime` take `Option<&mut ClassCursors>` (stack-level + per-virtual-
  layer); `step` cascades the stack latents and each layer's own up the stack in
  `forward`'s token order, returning the last token emitted. `prime(batch, caches,
  cursors)` opens that same private `cascade` with a token-less stream (`prime = true`
  switches each layer to `class_prime_plan`), returning `(Option<Tensor<2>>,
  Option<Caches>)`: the last latent emitted, and caches untouched when nothing ran — a
  partly primed cacheless stack is completed with zero caches for the layers that stepped
  nothing (exactly the state they hold). MultiGate hosts class latents at both levels: the
  `cascade` carries one `[b, k, d]` stream set per token and splices a latent into every
  stream, as `forward` does into its `[b, s, k, d]`. Cascade tokens go through
  `Layer::step_one`, not the cursorless `Layer::step`, so per-layer `Middle`/`End` place.
  Cursorless `step_infinite` mirrors `step` (incl. MultiGate; same residual/skip flags).
- **`multi_gate.rs`** — `Residuals{Standard|MultiGate}` (+`ResidualsConfig`) for `Layers`:
  MultiGate routes up to `n_stream` depth-streams per real/virtual layer
  (`per_virtual_layer`) in **two phases** — while fewer than `n_stream` exist the layer
  output is *appended* as a new stream (`accumulate`/`accumulate_step`), after which they
  are gate-mixed (`forward`/`step`); both end in the shared `attn_pool` (any stream count).
  Seeding `n` copies of the input instead is an unbreakable symmetry (identical streams ⇒
  identical gates and grads ⇒ one lerped stream). Point-wise, so `forward`==`step`.
  A class marker enters the token stream *and* all `k` streams; identical streams score
  alike, so the convex pool hands the row on unchanged — as the additive skip would.
  `depth_init_bias(n_mixing_layers, n_stream)` is the paper's carry bias over the *mixing*
  layers only; `init_bias_step` ramps it per stream (`0` = the paper's uniform init).
  Math, and why a convex mean-pool wants a norm before the head, in the header.
- **`network.rs`** — `LatentNetwork<M>` (linear in/out, **optional** pre-`out_proj`
  `norm_f` via `final_norm` — shared readout `head()`) and `VocabNetwork<M>` (embedding →
  unconditional `norm_f` → tied/untied LM head, vocab padded). Both build on the same
  `Layers<M>`. The embedding keeps Burn's `N(0,1)` `EmbeddingConfig` default (no
  initializer knob), so a **tied** head opens at logit variance `d_model`.
  Runtime enums `MambaLatentNet`/`MambaVocabNet` (+ concrete `*Config` enums — Config
  derive is not generic-aware); `forward`/`step` **panic on a family-mismatched
  cache/path**; `step_infinite` mirrors `step` (enums included;
  Mamba-3 only, panic otherwise). Both take `Option<&mut ClassCursors>`, the network's own
  class tokens riding the `network` cursor and the stack's the rest (each class token is a
  full network pass; `step_one` is the one-token body). `prime` (enums included) covers
  all three levels: the network's due class tokens each run a full pass, then
  `Layers::prime` flushes whatever is still waiting above them; `VocabNetwork::prime`
  returns the primed latent's logits.
  `*Builder`s carry `with_class_{tokens,latents}`; the `*Config` enum variants carry
  `class_latents` (stack level, `d_model`-wide) — `MambaLatentNetConfig` additionally
  `class_tokens` (network level, `input_size`-wide) — plus `residuals: ResidualsConfig`
  (plain additive vs Multi-Gate), `final_norm`, `ignore_first/last_residual` and
  `mlp: Option<GatedMlpConfig>`, and build a `MuonPlan` via `muon_plan()` (block + MLP
  weights; the boundary embedding/head/projections and the class tables are deliberately
  absent — they stay on AdamW).
- **`bidi.rs`** — `BidiLayerPair<M>` (straight + reversed-via-`flip`, merged) and
  `BidiLayers<M>` (stacks pairs with a `BidiSchedule`, adds the residual, runs pairs **by
  reference** via `bidi_pair_forward` — never clones a block, as a cloned un-materialised
  `Param` resamples); `OutputMerge{Mean(NoOp)|CatLinear(Linear)}`; runtime
  `MambaBidiLayers`. Forward-only, `forward` taking `Option<&mut ClassCursors>` (pairs take
  a single-level `ClassCursor`). MultiGate threads its streams **per pair**, same
  accumulate-then-mix schedule as `Layers` (stack latents are spliced before the seed, so
  they ride along). `muon_plan()` adds the per-pair
  `CatLinear.weight` merge to the block specs.
- **`cache.rs`** — `trait CacheStack` (collection iface `slot_count`/`into_slots`/
  `from_slots`, plus per-slot `cache_to_inner`/`cache_from_inner` for
  `Layers::grad_horizon`; impl'd for `Mamba{1,2,3}Caches`) + `enum MambaCaches` (**plain
  runtime state**, not a `Module`). The conversions are hand-written per family because
  `Module::map` is a no-op on plain `Tensor` fields, which is all a cache holds — a
  derived version would silently skip every tensor.
- **`norm/`** — `RmsNorm` (also Mamba-3 QK-Norm) + `RmsNormGated` (RMSNorm × SiLU gate,
  `norm_before_gate` toggle). **fp16-safe**: normalise against `max(|x|)` to avoid `x²`
  overflow; epsilon from `div_eps`.
- **`activation/`** — `Silu`, `softplus`, `log_sigmoid` (dtype-aware variants Burn
  lacks). `softplus` = identity above a per-dtype precision threshold (f64 38 / f32 18 /
  bf16 7 / f16 9), else `log1p(eˣ)` on a `clamp_max`ed input (so `eˣ` never overflows);
  `log_sigmoid` = `−softplus(−x)`, which keeps its large-negative tail (`log σ(x) → x`)
  finite.
- **`misc/`** — `gqa_expand_to_heads` (group→head replicate; `DP1=D+1` caller const),
  `segsum` (stable log-space 1-semiseparable mask; backbone of `ssd_minimal`),
  `split_into` (array-typed `split_with_sizes` → `let [z,x,b,c,…]=…`), `sanity` guards,
  `rope` (`wrap_angle`/`apply_rope{,_partial}`, Mamba-3 only).
- **`loss/`** — bce, cross_entropy, mse (example training).

## Optimizer (`src/optim/`, feature `optim`)

Muon needs one linear map per parameter; the fused projections are several, and the
rank-2 assert makes a wrong group a panic. So the plan is an **allowlist** and the
splitting happens in the optimizer, leaving the forward's single fused GEMM alone.

- **`spec.rs`** — `ProjSegment{name,width,muon}` (`muon()`/`adamw()`) and
  `ProjSpec{path,scope,segments}` (`block`/`block_whole`/`path`/`path_whole`, `width`,
  `has_muon`, `is_whole_muon`, `predicates`, `param_group`). `ProjScope::Block` matches
  the path under each `BLOCK_CONTAINERS` entry (`mamba_block`/`straight_block`/
  `reverse_block`), so one plan covers plain, virtual-layer and bidi stacks alike.
- **`segmented.rs`** — `Segmented` (`Optimizer`): splits weight+grad along `dim`
  (1 = a `Linear`'s output axis), steps each block with its own `Muon`/`AdamW`,
  concatenates. Exact — AdamW is element-wise, Newton–Schulz per-matrix. `RecordState`
  for `SegmentedState` is **hand-written**: the derive has no `Vec<nested>`, and
  unflatten cannot see the spec, so a block's kind is recovered from its leaf names
  (`momentum.velocity` vs `momentum.moment_1/2`, which never overlap).
- **`mod.rs`** — `MuonPlan{specs}` (`new`/`extend`/`with_mlp`/`without_segment`) and
  `build(&AdamWConfig, &MuonConfig)`: `adamw.init()` is the fallback group (and fixes
  the gradient clipping every Muon group reuses), then one group per Muon-owning spec —
  stock `Muon` for a whole-matrix spec, `Segmented` otherwise. `muon_config(wd)` defaults
  to `AdjustLrFn::MatchRmsAdamW`, whose update RMS is `0.2·lr`, so Muon and AdamW share
  one learning rate. Header records what is excluded and why — rank ≠ 2, embedding-like
  boundary weights, per-head scalar channels (Δ/`A`/`λ`), and the MIMO tensors (the paper
  parameterises them as element-wise scales, not the `DPR` stack of maps, so they are
  diagonals; MIMO's real matrix expansion is B/C, already inside `in_proj`).
- **`report.rs`** — `MuonPlan::describe(&impl Module)`: one line per parameter (path,
  shape, owner, segments with `*` on AdamW's) plus the share on Muon.

## Utilities (`src/utils/`)

- **`mod.rs`** — `div_eps(dtype) -> f32`: per-dtype safe-division epsilon (geometric mean
  of a scaled min-exponent and machine epsilon). Used by the norms.
- **`class/`** — learnable `[CLS]`-style tokens/latents. `ClassToken` (networks),
  `ClassLatent` (layer containers); markers stored as `#[module(skip)]` + one
  `Option<Param<Tensor<2>>>`. `ClassMarker` (`insert_pos`, `group_rank`, `needs_full_len`,
  `closes_sequence`) places `Start|Middle|End|Custom` against length `L` (Start@0,
  Middle@L/2, End@L, Custom@idx; ties keep `Vec` order). `Start`/`Middle`/`Custom` precede
  the token at their index — so a `Custom(k ≥ L)` lands only if the caller streams past
  `L`, still before the next token; `End` alone closes (`closes_sequence`), trailing the
  last token, and is what a closing `step` returns.
  `ClassCursor{offset, full_len}` is one level's placement state; `ClassCursors{full_len,
  network, stack, per_layer}` the whole hierarchy (`new(full_len)`/`stream()`; `per_layer`
  self-sizes; `fit`/`enter`/`leave` internal — the inner level's sequence is longer by the
  markers this level splices in). `class_chunk_plan` is the single placement decision —
  `(at, marker)` pairs for the next `chunk_len` user tokens, advancing the cursor — feeding
  `insert_class_markers` (a `forward` chunk) and `class_row` (one `step` token), so both
  calls place identically and a sequence splits anywhere. `insert_class_markers`' tensor
  half is `splice_class_rows<D>` (sequence axis 1, row broadcast over the rest) —
  rank-generic so one placement serves `[b, s, d]` and the MultiGate streams
  `[b, s, k, d]`; `class_emb_table` is the checked `[markers, width]` table it reads. `class_prime_plan` is its
  `prime` twin (same `class_plan` body): at the chunk's trailing edge it emits the markers
  waiting for the *next* user token instead of leaving them, never `End`, and nothing once
  the cursor is at the announced end — which is what keeps a `Custom(k ≥ L)` from trailing
  (only a `step`, which *has* the token, may land it). `class_emb_width` sizes a prime's
  rows, having no token to read the width from. `assert_full_len_known` guards
  `Middle`/`End`. `class_marker_output_indices` reports a position past the emitted
  sequence for a marker that never lands.
- **`schedule/`** — `Schedule{Cyclic|Stretched|Custom}` (`real_idx`) and
  `BidiSchedule{Strided*/Symmetric*/Custom}` (even virtual = →, odd = ←).
- **`scheduler/`** — `Lr{CosineAnnealing|Constant}` (`get_lr(step)`; cosine + warmup).
- **`backend_macros.rs`** — `impl_ssd_backend_ext_for_burn_backends!` (per-backend default
  blocks) + `decl_ssd_autodiff_backend_ext!` (autodiff marker + `Autodiff<B>` blanket).
- **`combined_grad.rs`** — `flatten_pair`/`unflatten_pair`: `(y, final_state)` into one
  tracked tensor and back (`prep.finish` takes a single tensor).
- **`detach.rs`** — `detach_params(module)`: clears `require_grad` and re-roots every
  `Param`. Cuts gradients but **frees no memory** — Burn registers untracked ops in the
  graph anyway, so their activations stay retained (measured: 3144 MB vs 208 MB for an
  inner-backend prefix at 64 virtual layers, and only the latter is flat in depth). The
  module header carries the numbers; `Layers::grad_horizon` uses the inner backend
  instead.
- **`fprim.rs`** (`mamba2`/`mamba3` only) — `F<B, const D>`: rank-tagged `FloatTensor<B>`
  newtype mirroring the
  `Tensor` method API, so the generic-`B` forward kernels and `Backward<B,_>` nodes
  (which can't build a `Dispatch` `Tensor`) read like tensor code over `B::float_*`.
  `Mask<B>` + `san(&F)` accompany it.
- **`test_helpers.rs`** (test-only) — `max_abs_diff` + `check_grads_match_two_paths!`,
  shared by the SSD-path agreement tests.

## Benchmarks (`benches/layer.rs`, `bench.sh`, `kernels.sh`)

Criterion single-block benches (`forward`/`train`/`step`) over all three families; the
Mamba-3 cases pair the SISO-specialization flags head-to-head and sweep the rotation ladder
(`real1d`/`quaternion4d`/`rotor4d` against `siso`'s `Complex2D`).
**Run by the user, not by an agent.** Each case builds its block, input and warm-up
*inside* the criterion closure, so a `--` filter really isolates one case.
`bench.sh` drives the backend configurations — flex and CUDA share one build,
fusion needs its own — and writes `bench.md`.
`kernels.sh` reuses those builds to count kernel launches per case — cubecl's
profiling logger at `basic` totals the launches between syncs, and a count is
exact, so one `--test` iteration suffices — and writes `kernels.md`. All carry
their own rationale.
