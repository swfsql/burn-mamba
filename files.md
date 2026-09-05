# files.md

A per-file **signature reference**: what each important file defines and the
non-obvious decisions worth knowing before editing it. For the architecture and file
tree see `CLAUDE.md`; for notation see its [Notation](./CLAUDE.md#notation) section.
The detailed per-family math lives in the `mamba2.rs` / `mamba3.rs` module headers.

Covers **this** crate only. The block-generic composition layer (`Layer`/`Layers`/
networks/bidi/multi-gate/class tokens/schedules/norms/losses/Muon) is the sibling
crate `../burn-stack/`; see its `CLAUDE.md` File Map.

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
Feature-gated module decls (`mamba{1,2,3}`, `unified`) + `prelude` + crate overview.
`#![warn(missing_docs)]`. `pub use burn_stack;` re-exports the composition crate, and
`prelude` re-exports `burn_stack::prelude::*` alongside the `Mamba*` unified types.
The `DENY_NAN`/`DENY_INF` guards live in `burn_stack`.

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

- **`mamba3.rs`** — `Mamba3` + `Mamba3Config` (`state_rank` **even** unless `Real1D`;
  `mimo_rank` 1=SISO, the rank dial (`info/mimo-as-batch.md`; `mimo_{x,z,o}_hmp` init
  `1/M`, `1`, `1/M`, so a MIMO block *is* its SISO block at init — key `mean_m B[m]`,
  query `mean_m C[m]`, `D/M`, and a rank-**one** write);
  `micro_steps` (`u`, default 1 = stock) — MambaProduct, see
  `mamba3/product/`; `trapezoid: Trapezoid` (the `β` tap pattern; `init` panics on the
  unimplemented ones); `rope_fraction` `0.5|1` (default 1, full); `rotation: RotationKind`;
  `rotation_range` (default 2, the per-step bound in half-turns per unit Δ, applied to
  **each** quaternion factor — both defaults ship the full rotation, and the reference's
  narrower `1`/`0.5` are asked for explicitly); `a_floor`). `rotation_spec()` bundles the
  three rotation fields; `num_rotation_blocks()` = `num_quat_blocks · quat_factors` (the
  projection/scan block axis, doubled for `Rotor4D`) drives `num_rotation_channels()`;
  `zero_rotation_state()` is the one fresh-cache accumulator, shared by every pathway.
  Under `Real1D` every rotation count is `0` and `state_rank` may be odd (scalar `1`);
  `init` asserts any other kind turns ≥ 1 pair over an even `state_rank`, and
  `muon_projections()` omits the (then absent) rotation segment. Fields:
  QK-norm `b_norm`/`c_norm`, `b/c_bias_hmr` (init 1), optional `mimo_{x,z,o}_hmp` and
  `out_norm`. Derived `d_in_proj` (split `[z|x·u|B_raw·u|C_raw|dd_dt·u|dd_A·u|λ_raw·u|θ·u]`
  — only the per-micro-step segments widen; the two trailing segments are the optional
  ones, peeled by `split_trailing` in layout order: `rotation_channels_total()` = `u·`
  `num_rotation_channels()` (`0` under `Real1D`), then `lambda_channels_total()` = `u·nheads`
  (`0` under `Trapezoid::None`)), mirrored by `muon_projections()` as
  `in_proj [z|x|B|C|dt*|A*|λ*|rotation]` with each `u`-wide stream emitted as `u`
  same-named segments (independent maps to Muon; `without_segment` still drops the whole
  stream) + `out_proj`.
  `forward`/`step` **dispatch by cache variant** (missing ⇒ SingleSsd).
  Two performance-only `mimo_rank == 1` knobs (default on, `#[module(skip)]`, identical
  values/grads): `siso_specialization` for the chunkwise γ-correction (threaded down as a
  field of the SSD input bundle, into the ext trait and the `Backward` node's state) and
  `siso_specialization_decode` for the per-token sites (`use_siso_decode_kernels()`).
  Split because the first wins on every backend and the second only on GPU.
- **`mod.rs`** — `Mamba3BackendExt: Mamba3DoubleSsdBackendExt + Mamba3SingleSsdBackendExt`,
  wired via `backend_macros`.
- **`helpers.rs`** — rank-generic, shared by both pathways/modes: `trapezoidal_coefficients`
  (`Δ/A/da/α/γ` + **optional** `β`; a `None` `lambda_raw` is `Trapezoid::None` and yields
  `β = None`, `γ = Δ` by sharing `Δ`'s tensor), `qk_norm_expand_bias`, `build_v_with_mimo`,
  `mimo_outer_sum` (`Σₘ v[m]⊗k[m]` state contribution; step + boundary seed, which fuses the
  tap slots into `m`; `_siso` broadcast vs `_mimo` matmul, per `siso_specialization_decode`),
  the tap-lag trio `shift_stream` / `interior_gap_decay` / `tail_decay` (the lag-`L` shift
  with the cache's slots as prefix; `Πᵈ⁼¹‥ᴸ⁻¹αₚ₋ᵈ`, the gap transport `β`'s own `αₚ` does not
  cover, front **zero**-padded because the cache's `x` slots already carry that part; and
  that part — `Πᵣ₌q₊₁ᔆ⁻¹αᵣ` for the last `L` positions. All three are `None`/no-ops at
  `L = 1`, so the default pattern is byte-identical), `split_trailing` (peels an
  optional trailing in-proj segment — rotation under `Real1D`, then `λ` under
  `Trapezoid::None`, which is why those two are last; it cannot be one more entry in the main
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
- **`trapezoid.rs`** — `Trapezoid`, the trapezoid's **tap pattern**: which earlier sample
  the `β` tap reads. A choice that exists only at `micro_steps > 1`
  (`info/trapezoid-as-integration.md` §§8–9), and a structural one — it picks
  whether the in-proj spends `λ` channels and, through `tap_lag()`, everything else.
  **`tap_lag(u)` is the one knob the two implemented patterns share**: `0` no tap /
  `1` `HorizontalCarryOver` (the default) / `u` `Vertical`, and it *is* `tap_slots()`
  (a lag-`L` tap needs the last `L` positions live), the pre-chunking shift, the
  single-SSD key-scale offset and `step`'s FIFO depth. So the two are one algorithm
  read at two lags, and coincide at `u = 1`.
  `Vertical` = lag `u`: the tap reads the *same micro-step of the previous token*, so
  all taps cross (vs `1/u`) and the pattern is `u` token-rate filters. What is vertical
  is the **tap graph**, not the scan — the state keeps the one flattened chain, which is
  what keeps it causal and `forward` ≡ `step`. Closed form at token resolution:
  `hₜ = Aₜ(hₜ₋₁ + Σⱼ νₜ,ⱼ ṽₜ₋₁,ⱼ) + Σⱼ γₜ,ⱼ ṽₜ,ⱼ` — the `u=1` trapezoid with its left
  endpoint promoted from rank 1 to rank `u`, `A` untouched (still `λ`-free).
  `None` = `λ ≡ 1` ⇒ `β = 0`, `γ = Δ`, with nothing
  paid for the absent term anywhere — no `λ` in-proj/Muon segment, no `β` tensor, no tap
  slots, **one** SSD call in `forward`, one outer product in `step`; the two pathways
  coincide, so `forward_single_ssd` delegates.
  `assert_implemented()` is called from `Mamba3Config::init`, so the rest fail at
  construction. Degeneracies at
  `u = 1`: `Vertical` = `HorizontalCarryOver`, `HorizontalReset` = `None`.
  `VerticalPlusHorizontalReset` is the least settled: a 2-D tap graph (`(t−1,j)` and
  `(t,j−1)`) that still collapses to one scalar per sample, both taps being lags on the
  one folded chain.

### `mamba3/double_ssd/`
- **`double_ssd/mod.rs`** — `forward_double_ssd`/`step_double_ssd` + the RoPE utilities.
  Splits the trapezoid into γ-SSM (current ×γ) + β-SSM (tapped ×β, shift-before-chunking
  by `tap_lag`, `β` picking up `interior_gap_decay` so it carries the whole gap), summed;
  ~2× SSD memory — except under `Trapezoid::None`, where the β side (shift,
  scale, second SSD call, tap slots) is skipped outright and this is one plain SSD pass.
  `forward` folds the micro-steps in right after the split and
  collapses back after the SSD (so between them `s` counts micro-steps and `tokens` is the
  only token-resolution name); `step` loops the recurrence `u` times and reads out once,
  driving the tap FIFO **tap → decay the survivors → push** so a slot's `x` accumulates
  exactly the `α`s between its own position and the one that taps it (at lag 1 nothing
  survives a tap, so nothing is ever decayed and `β = (1−λ)Δα` is the whole coefficient).
  `step_double_ssd` is reused (via cache conversion) for
  single-ssd decoding; it is factored through pub(crate) `StepProjection`/`step_project`
  (in-proj → coeffs → QK-norm, pre-rotation; per-micro-step streams keep a `u` axis that
  `MicroProjection`/`StepProjection::micro(j)` peels, `z`/`C` are per token; `rot_ba` is
  `None` under `Real1D`), `step_readout` (state×C einsum, `_siso`/`_mimo` branches) and
  `step_finish`
  (D-skip, gate/gated-norm, MIMO aggregation, out-proj). `rotation/rope.rs`'s `apply_rope`/`apply_rope_partial` (rotate
  last-dim pairs; interleaved/NeoX SISO vs half-and-half/GPT-J MIMO; `rope_dim > 0`
  required) and `wrap_angle` are used by **both** pathways.
- **`cache.rs`** — `Mamba3DoubleSsdCache`: `ssm_bhpr` (trapezoidal state), the tap FIFO
  `k_state_bumhr`/`v_state_buhp` (`tap_slots` deep, **oldest first**), `rotation`
  (`RotationState`). No conv. The FIFO's `k` is stored **as rotated at its own position**
  (the relative-rotation factoring reconstructs the transport) and its `x` **pre-scaled by
  the decay since that position** — which is what carries a lag-`u` gap across a call
  boundary, and is a no-op at lag 1. The two tap slots are `Option`s — absent, not zeroed,
  when the `Trapezoid` has no β tap (`sanity()` asserts they agree); the `*CacheConfig`s
  carry `trapezoid` + `micro_steps` to size them.
- **`ssd/ssd_path.rs` + `ssd/*`** — `Mamba3DoubleSsdPath`; `Mamba3DoubleSsdInput` is
  **MIMO-first** (`v_bnlmhp` already ×γ/β, `da_bnlh`, `b/c_bnlmhr`). Same three algorithms
  as Mamba-2 with the `mimo_rank` axis fused into the chunk reshape;
  `serial_recalculated/` defines `Mamba3DoubleSsdBackendExt` + custom backward.

### `mamba3/single_ssd/`
- **`single_ssd/mod.rs`** — `forward_single_ssd`: one SSD call with key scale
  `scaleₜ = γₜ + (1−λₜ₊ₗₐ₉)·Δₜ₊ₗₐ₉` (a sample's two trapezoid installments share a transport
  and collapse to this one scalar — `info/trapezoid-as-integration.md` §5; the pathway
  exists because they do, and §9's collapse theorem is why the wider lag changes nothing
  here), strict-lower-triangular intra-chunk mask + same-step γ
  correction (in-kernel) widened by `token_band` at lag > 1, and a **boundary-β seed** over
  the cache's `lag` slots folded into the initial state. Same
  micro-step fold/collapse as the double pathway; everything between (trapezoid, `scale`,
  QK-norm, rotation, seed, chunking) runs at micro-step resolution unchanged.
  `step_single_ssd` converts to a double-ssd cache, runs `step_double_ssd`, converts back.
  Under `Trapezoid::None` the whole body is skipped: `scale ≡ γ` makes this the double-ssd
  form (strict mask + full diagonal correction = inclusive mask), so `forward_single_ssd`
  delegates rather than reassemble what it masked.
- **`token_band.rs`** — the lag-`u` correction band, as **one intra-token contraction
  outside the kernel**. `scale` is wrong for the `lag` reads before a tap is paid: the
  diagonal at lag 1 (in-kernel, `ssd/diag.rs`), a `u`-wide band at lag `u`. Non-obvious and
  the reason no mask, chunk-length constraint or cross-chunk term is needed: at the only
  reads that survive (`j = u−1`) that band **is the token**. Equivalence with double-SSD
  therefore holds on everything a caller observes (output + every cache field) but not on
  the `u−1` per-token partial sums `last_micro5` discards — correcting those would need the
  band's cross-chunk and boundary-seed parts, where it is already `scale`-weighted and
  cannot be un-weighted. The state is exact at every position in both pathways.
- **`cache.rs`** — `Mamba3SingleSsdCache`: same four fields (same tap-FIFO layout and
  decayed-`x` convention, hence the field-identity `From`) but `ssm_bhpr` carries
  `h'ₜ = αₜh'ₜ₋₁ + scaleₜ Bₜ⊗xₜ` (correct except the `lag`-wide band, patched in-kernel at
  lag 1 and by `token_band` beyond). The
  distinct type prevents mixing a double-ssd cache into single-ssd mid-sequence — a
  distinction that vanishes under `Trapezoid::None`, where `h' ≡ h` everywhere.
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

### `mamba3/product/` (`mod.rs`)

**MambaProduct** — DeltaProduct's *dial*, not its mechanism (`info/rotation-as-optimization.md`):
`u = Mamba3Config::micro_steps` full Mamba-3 steps
per token, so a token's transition is the **product** `(∏ⱼαⱼ)·R_{u−1}⋯R₀` and its write a
sum of `u` outer products staggered along it. Evaluated by folding the micro-steps into the
**sequence axis** — no new kernel, no cache change (`u` buys transition expressiveness, not
memory) — with the read `C`, the gate `z`, the `D` skip and the output per token.
`unfold_micro_bs`/`unfold_micro_b` reinterpret a `u`-wide in-proj segment as `u` positions
(a pure reshape: the projection already lays micro-steps out contiguously),
`repeat_micro_bs` broadcasts the per-token `C` across the group so its *last* copy carries
the right cumulative rotation, `last_micro5`/`last_micro4` collapse `y`/`x` back.
Why it is a Mamba-3 dial and not a Mamba-2 one: the curvature is isotropic, so every
per-micro-step factor is a scalar and scalars commute — `u` micro-writes provably collapse
into one decay-weighted rank-`u` write, and the rotation must come from the *step size*
(`RotationKind`) rather than from the product. The same is true
here under `Real1D` — the sequential reading of the cell `mimo_rank` occupies jointly —
while `Complex2D` gains `u`× the per-token angle reach with a live gradient at every factor
(a single step at the `rotation_range` bound sits on `tanh`'s asymptote), and the
non-abelian kinds gain a product no single bounded step can express.
What `u` buys is bought by inflating the token's effective interval `u`×, not subdividing
it; the consistent alternative (`Δⱼ = Δ/u`) is inside the model's reach (`dt_limit` has no
lower floor **by default** — it is a configurable clamp), so this is a superset of it.
Non-obvious: unlike DeltaProduct there is no forget gate to keep at token rate — Mamba
fuses decay, write weight and rotation rate into one `Δ` — so every micro-step decays; a
scalar decay composes, so this costs nothing, and pinning `α ≡ 1` would silence the
rotation with it. What `u > 1` does change is the trapezoid's taps, which now live on the
folded chain and so become a *choice* (`trapezoid.rs`): at the default lag 1 they straddle
*micro-steps*, so only `1/u` still cross a token and the interior ones pair two projections
of *one* token (`info/trapezoid-as-integration.md` §§8–9, with the tap lattice `u>1` opens;
`λ` is per micro-step, so `λ=1` on the interior recovers the `u=1` semantics); at
`Vertical`'s lag `u` all of them cross. The tap cache is the FIFO that choice needs.
Tests: helper round-trips; `d_in_proj`/Muon-tiling arithmetic (`u=1` is stock); forward≡step
on both pathways × all four kinds × `u∈{2,3}`; split-prefill continuity; forward≡step
**gradients** (input + `in_proj`); per-micro-step gradient liveness (a dropped or
mis-ordered micro-step passes every value test).

### `mamba3/rotation/` (`mod.rs`, `rope.rs`)

`rope.rs` is the mechanical half: `wrap_angle` (reduce mod `2π`, offset `detach`ed so
the value is exact and fp16 stays stable over long sequences) and
`apply_rope`/`apply_rope_partial` (rotate last-dim pairs over a `rope_dim` prefix;
interleaved/NeoX pairing for SISO, half-and-half/GPT-J for MIMO). Not a positional
encoding — the angles are the imaginary part of the *state transition*, factored out of
the state and into B/C.
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
`forward` and `step` both derive the per-step rotation from one pair of
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

---

## The unified API (`src/unified/`)

Where the three families meet `burn-stack`'s block-generic containers, plus the
runtime-selectable enums that pick a family at run time (and panic on a
family-mismatched cache or SSD path). The containers themselves are `burn-stack`.

- **`mod.rs`** — `enum MambaSsdPath` (`Mamba1|Mamba2(_)|Mamba3(_)` +
  `mamba{2,3}_default()`). Module header carries the Muon *"why the 3-D tensors are not
  stacked matrices"* argument: `mimo_x`/`mimo_z`/`mimo_o` are learnable per-head
  **diagonals** (`DP + PR`, not the `DPR` stack of maps the paper avoids
  instantiating), so orthogonalising one would constrain a set of gains; MIMO's real
  R-fold matrix expansion is B/C, already inside `in_proj` and already Muon's.
- **`cache.rs`** — `enum MambaCaches` (plain runtime state, **not** a `Module`: caches
  are threaded through `forward`/`step`, never recorded or optimised) and the three
  per-family plug-ins: `impl CacheStack for Mamba{1,2,3}Caches`, `impl Block for
  Mamba{1,2,3}` (`Options` = that family's `*SsdPath`; `()` for Mamba-1, which has no
  chunking), `impl BlockConfig for
  Mamba{1,2,3}Config`. `cache_to_inner`/`cache_from_inner` are spelled out per family
  rather than derived: `Module::map` is a **no-op on plain `Tensor` fields**, which is
  all a cache holds, so a `Module`-based conversion would silently skip every one of
  them — and `Tensor::inner` panics off autodiff, so `Layers::grad_horizon` must check
  `Device::is_autodiff` first. `MambaCaches::detach()` is `CacheStack::detach` (values
  kept, graph dropped) dispatched over the runtime tag: the enum cannot implement
  `CacheStack` itself (its slot type would have to be a fourth enum), and a caller
  carrying a cache across a gradient boundary holds the enum, not the family type.
- **`network.rs`** — `MambaLatentNet`/`MambaVocabNet` + `#[derive(Config)]` `*Config`,
  wrapping `burn_stack::modules::{LatentNetwork, VocabNetwork}`. The variant field is
  still named `mamba_block`, so saved `model_config.json` files keep loading.
- **`bidi.rs`** — `MambaBidiLayers` + `MambaBidiLayersConfig`, wrapping
  `burn_stack::modules::BidiLayers`.
- **`tests/`** — the burn-stack containers exercised against **real** blocks (burn-stack
  tests them against its own reference block): `layer` (the two residuals of an `mlp`
  layer), `layers` (`grad_horizon` reachability + shared-weight gradients),
  `multi_gate`, `bidi`, `class` (marker placement, forward/step/prime parity), `optim`
  (each family's plan fits its model and never selects a boundary weight — the
  boundary test keys off `burn_stack::optim::BLOCK_CONTAINERS`, not a spelling).

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

## Notes (`info/`) and their checks (`scripts/`)

- **`info/rotation-as-optimization.md`** — the reference for the `micro_steps`/DeltaProduct
  relationship and for the optimization reading of the complex transition. Derives: a real
  step on any real loss cannot rotate; the three views of Mamba-3's rotation that can
  (complex step size / descent-ascent on a harmonic potential / momentum); DeltaProduct's
  fourth route (composing non-commuting rank-one curvatures) and why isotropy forbids it
  here; the (curvature rank × step algebra) 2×2 the `RotationKind` table follows from.
  Cite it rather than restating it.
- **`scripts/rotation_as_optimization.py`** — float64 `numpy` check of all 66 of that
  document's numbered claims, section numbers matching. Encodes the recurrence from the
  equations and never imports the crate, so it is independent of the implementation
  (which the Rust suites cover). Runs standalone; non-zero exit on failure.
- **`info/trapezoid-as-integration.md`** — the reference for the trapezoidal
  discretisation and its interaction with `micro_steps`; companion to the above, which
  classifies the *quadratic* term while this classifies the *linear* one. Derives: the
  rank-two target, and that `λ` touches only the linear term (so the isotropy arguments
  survive); `λ` as an operator-splitting parameter (Lie–Trotter at 1, Strang at ½, which
  *derives* the paper's `O(Δ³)` condition); the two-installment collapse `Δ̃ₛ`, which **is**
  single-ssd's key scale; the augmented state as a feedforward buffer (`spec = spec(M)∪{0}`)
  and its exact difference from momentum; the token-level `A(x_i)`/`B(x_{i−1},x_i)` unroll,
  in which `λ` never appears in `A`; the collapse theorem for arbitrary tap lags and the
  vertical/reset-horizontal/carry-over-horizontal lattice it opens at `u>1`.
  Cite it rather than restating it.
- **`scripts/trapezoid_as_integration.py`** — same contract as the above: float64 `numpy`,
  54 checks, section numbers matching, standalone, non-zero exit on failure.
- **`info/mimo-as-batch.md`** — the reference for `mimo_rank` and its interaction with
  `micro_steps` and `RotationKind`; third of the trio, classifying the linear term along
  *rank* where the trapezoid note classifies it along *time*. Derives: the rank-`M`
  objective (a minibatch of `M`, ridge split `ρ/M`; `G` stays `(1−α)I`, so the isotropy
  arguments survive); the value tying — `M` free keys but one value through `M` fixed
  masks — priced at `(M−1)P − M² + 1` lost write dimensions; the paper's `M²`-SISO
  identity as *isotropy*, not a property of MIMO (it fails for both a sequential and a
  jointly-solved rank-`M` delta rule); the init identity above; why the rotation **must**
  be shared across ranks (one state ⇒ one transition; per-rank angles have no state-space
  preimage and break the time-invariance of the same-step `M²` Gram); and the containment
  `MambaProduct(u=M) ⊇ MIMO(M)`, the two dials being one dial with different things tied.
  Cite it rather than restating it.
- **`scripts/mimo_as_batch.py`** — same contract as the above: float64 `numpy`, 54 checks,
  section numbers matching, standalone, non-zero exit on failure.
