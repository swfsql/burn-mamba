# mamba3.md — plan: physical-frame state moments / state-PR for the Mamba-3 `forward()`

Design plan for extending the Mamba-2 state-moments feature
(`src/modules/state_moments.rs` + `src/mamba2/ssd/moments.rs` + the
`forward_with_state_moments(_grad)` cascade) to Mamba-3. Grounded in the
current `src/mamba3/` code **and** the paper's complex-SSM sections
(`../papers/mamba-3/structure/methods_complex.tex`,
`../papers/mamba-3/appendix/complex.tex`).

**The feature** is the Hermitian PR of the **physical-frame** state moment
`M_phys` — both the penalty and the shipped diagnostic (no extra shipped
observables). **The main execution model is `M_phys` + the single-SSD
pathway + a `SerialRecalculated`-style custom recompute backward** (memory
at scale, composing with the SSD's own custom backward). The cache-frame
closed-form moments (`M̃`) and the rotation-stripped content moments
(`M°`) exist **only as test-gated verifiers**.

## 0. What carries over unchanged

- **`StateMoments` accumulators.** The de-rotated (physical) states are
  plain real tensors of the Mamba-2 state shape, so `{m2_bhrr, m1_bhr,
  count}`, `merge`, `pool_batch` are reused verbatim on them. All Mamba-3
  novelty is isolated in (a) producing per-token *de-rotated* states
  chunk-locally (§2) and (b) the Hermitian PR recombination (§1).
- **The upper cascade.** `MambaBlock::block_forward_with_state_moments`
  and `_grad` already exist with panicking defaults; the two overrides in
  `modules/cache.rs` light up `Layer`/`Layers`/networks/runtime enums
  with zero further changes.
- **The example glue, almost.** `diagnostics::state_pr_forward` needs a
  `MambaCaches::Mamba3` arm; the penalty plumbing is unchanged — the
  Mamba-3 arm penalises (and logs) `PR_ℂ(M_phys)` only.

## 1. The object: the physical-frame Hermitian moment

Per the paper (Props. *complex-to-real* / *rope-trick-trap*; proofs in
`appendix/complex.tex`), the Mamba-3 state is **complex**,
`cₜ ∈ ℂ^{p × r/2}` with transition `e^{Δ(A+iθ)}`; the implemented real
`ssm_bhpr` is its realification (each rotation plane = one complex
coordinate's `(Re, Im)` pair; the output reads `Re(Cᴴh)`, hence the
`[C; −Ĉ]` sign). The cache/tilde state `h̃ₜ` — what `step()` carries,
rotations absorbed into B̃/C̃ — relates to the **physical** state by the
cumulative phase: `cₜ = h̃ₜ Dₜ`, `Dₜ = diag(e^{iφ_{t,a}})`. The combined
real recurrence (mamba3.rs §4), with `B̃ = R·B` the RoPE-rotated,
QK-normed, bias-added key:

```text
hₜ = αₜ hₜ₋₁ + βₜ Σₘ B̃ₜ₋₁[m] ⊗ (xₜ₋₁ ⊙ mimo_x[m]) + γₜ Σₘ B̃ₜ[m] ⊗ (xₜ ⊙ mimo_x[m])
```

**Definition.** `M_phys = Σₜ cₜᴴcₜ` (pooling `p` and `t`):

```text
M_phys[a,b] = Σₜ e^{−iφ_{t,a}} G̃ₜ[a,b] e^{+iφ_{t,b}} ,        G̃ₜ = h̃ₜᴴh̃ₜ
            = b₍·a₎ᴴ ( Λ_aᴴ Λ_b ∘ XX ) b₍·b₎ ,   Λ_a[t,j] = L[t,j]·e^{−i(φ_{t,a}−φ_{j,a})}
```

The second line (raw keys `b₍·a₎` = the plane-`a` components over write
positions `j`, value Gram `XX`, decay mask `L`) is the "two-sided RoPE
trick": B's rotation survives only as the **write→read relative phase
over the decay window**, and **C's rotation cancels identically**
(`yₜ = C̄ₜᴴh̃ₜ = Cₜᴴcₜ`) — `M_phys` is the covariance that raw,
un-rotated C reads.

**Why it is the penalty (rank-honest).**

- Within-plane rotation is free: a rotating single-plane rank-1 conveyor
  reads `PR ≡ 1` (phases cancel in `ccᴴ`).
- Rank *created* by rotation is charged: a writer spread over `k` planes
  at different rates genuinely occupies `k` complex dims of retained
  memory — `PR → k`, weighted by the decay window (only what is still
  remembered counts). No frame or per-token gauge can collapse this
  (single-token PR is unitary-invariant and already exceeds 1 once
  retained writes have been rotated apart).
- No phantom rank: the cache-frame pooled moment additionally charges
  global frame *drift* — a **static** physical state under ongoing
  rotation spins in the cache frame (`h̃ₜ = Dₜ†v`), decohering `M̃`
  toward the number of active planes while the memory content is
  unchanged; `M_phys` reads its true rank. This is why `M̃` is only a
  test verifier, not the shipped metric.
- `θ ≡ 0` ⇒ `Dₜ = I` ⇒ `M_phys = M̃` = exactly the Mamba-2 moment: the
  penalty is a strict generalisation of the one already proven to grok
  on Mamba-2.

**PR.** Pair the `r` axis per the layout into `c = x + iy`; then
`M = A + iS` with `A = Σ(xxᵀ + yyᵀ)` (symmetric), `S = Σ(yxᵀ − xyᵀ)`
(antisymmetric) — both linear recombinations of the *de-rotated* states'
real `m2_bhrr` sub-blocks (`m1_bhr` pairs likewise for `center`); and
`PR_ℂ = (tr A)² / (‖A‖²_F + ‖S‖²_F)` (Hermitian: `tr M = tr A` real,
`tr M² = Σ|M_ab|²`). Hermitian, **not** realified: one complex direction
is one readout channel (`y = Re(Cᴴh)`); the ×2 realified count is a
representation artifact, unrelated to rotation-created rank. Blocks:
`rope_fraction < 1` ⇒ unrotated dims stay a *real* block, with the
cross-coherences filling a mixed Hermitian matrix `[[M, X],[Xᴴ, Σuuᵀ]]`
(PR over that; still all recombinations of `m2_bhrr`); `Quaternion4D` ⇒
same formulas with 4-blocks (quaternionic Hermitian `M`: diagonal real,
`tr(M²) = Σ‖M_ab‖²` with 4-component norms).

Test-only relatives (§5): the **cache-frame** `M̃` (closed-form Gram
reduction, no state materialisation — the Mamba-2 derivation over the
combined injections) and the **content** `M°` (same closed form fed raw,
source-frame B — fully rotation-blind; `PR° ≡ 1` for any constant writer,
including multi-frequency ones). All three share `tr` (frame-invariant),
which is the cheap cross-check.

## 2. Algorithm: serial chunkwise recompute (there is no closed form)

**Obstruction.** In `M_phys[a,b]` the token index couples to the matrix
*entry* through `e^{i(φ_{t,b}−φ_{t,a})}`, so the per-token phases do not
factor out of the Gram kernel; every exact factorisation of
`Λ_aᴴΛ_b ∘ XX` reproduces the per-token state as an intermediate. Exact
`M_phys` therefore costs materialised states — affordable
**chunk-locally**, in exactly the `SerialRecalculated` discipline. Per
chunk (serial over `n`, small accumulators carried):

1. **Combined injections** (the Mamba-2-derivation gift, unchanged): the
   trapezoid is one scalar-decay SSD with a `2·mimo_rank` channel axis —
   `x̂ = concat_m(v_γ, v_β)`, `b̂ = concat_m(b̃, b̃_prev)`, shared
   log-decay `da` and mask `L = exp(segsum(da))`; the β stream is
   "shift-before-chunking" with the cache's `(k_state, v_state)` as its
   first element. The data-dependent γ/β and `Aₜ` are already absorbed
   into the pre-scaled `v` and `da`.
2. **States**: `h̃[t] = dₜ·h₋ + Σⱼ L[t,j] · x̂ⱼ ⊗ b̂ⱼ` — a channel
   pre-contraction plus one batched GEMM, ≈ `l²·p·r` per `(b,n,h)`
   (a `min(p,r)`× factor over SSD step 1); transient memory = **one
   chunk** of states `[b,h,l,p,r]`.
3. **De-rotate**: apply the *forward* cumulative rotation to the states'
   `r` axis (`v = (Π R)·h̃`) — reuse the rotation module's `apply`; the
   cumulative angles/quaternions are already computed before being
   absorbed into B̃/C̃, the seam only exports them. No per-channel
   special-casing: β writes physically enter one step rotated, which the
   already-rotated `b̂` channels + end de-rotation reproduce exactly.
4. **Accumulate** the plain real `m2/m1` sums of the de-rotated states,
   masked to `valid_len` (pads: `Δ=0` ⇒ identity decay, zero write, and
   `Δθ=0` ⇒ identity rotation — confirm).

Boundary carry `h₋` per chunk comes from the pathway's own
chunk-boundary states (§3), with the cache's initial state counted
exactly once — taken from the cache directly, not from the per-stream
input bundles.

**Backward — the main execution model.** Plain autodiff over the serial
loop retains *every* chunk's states (the full per-token trajectory) —
acceptable only at study scale. The shipped mode is a **custom recompute
backward** node in the `SerialRecalculated` pattern (as `quat_scan` /
`mamba2/ssd/serial_recalculated`): the forward stores inputs plus the
small `(m2, m1)` accumulators; the backward re-materialises one chunk of
states at a time and accumulates input grads (`combined_grad.rs`-style
flattening for the multi-output). Gradients flow to `x̂`/`b̂`/`da`, the
**angles** (the de-rotation is θ-differentiable — this is what lets the
penalty shape the rotation itself), and the initial state.

## 3. Pathway: single-SSD is the main target

The moments never touch the SSD kernel — they read pre-SSD tensors — but
the seam differs per pathway:

- **Single-SSD (primary)**: its accumulator `h′` has different
  mid-sequence semantics, but the chunk-boundary states coincide with the
  true states (the field-identity `From` conversions in
  `mamba3/cache.rs`). At the seam, rebuild the γ/β-scaled shifted
  injections from the pre-kernel tensors (raw `v` + `gamma_bnlh` +
  `scale_bnlh`; reading says they suffice — confirm at implementation),
  export the cumulative rotation, and run §2 off the boundary states.
- **Double-SSD**: the seam is right before the two `.run()` calls —
  `input_gamma`/`input_beta` are exactly the needed bundles. Kept working
  (dispatch is by cache variant) but serves mainly the cross-pathway
  parity tests.
- **Step side**: the cache stores `ssm_bhpr` (tilde frame) *and* the
  `RotationState` accumulator, so the physical state is
  `apply(rotation_state, ssm_bhpr)` — used by the stepwise diagnostic
  and the parity tests.

## 4. API / file plan

```text
src/modules/state_moments.rs   + StatePairing descriptor (block size 2|4,
                               layout interleaved|half-half, rotated range)
                               and StateMoments::pr_complex(&StatePairing)
                               — assembles A/S (+ mixed blocks) from
                               m2_bhrr/m1_bhr; real pr() untouched (Mamba-2)
src/mamba3/rotation/…          export StatePairing + the per-token cumulative
                               rotation at the seam (single source of truth
                               with the rotation's own pairing/application)
src/mamba3/moments.rs          Mamba3MomentsInput { xhat_bnlMhp, bhat_bnlMhr
                               (M = 2·mimo_rank), da_bnlh, rotation (angles |
                               quaternions), initial_state_bhpr, init_state_hpr }
                               ::state_moments_phys(valid_len) -> StateMoments
                               (of de-rotated states) — serial recompute
                               forward; custom backward on the cubecl/fusion
                               families via backend_macros, plain-autodiff
                               fallback elsewhere (+ ::detached())
src/mamba3/mamba3.rs           forward_with_state_moments(_grad): cache-variant
                               dispatch; single-ssd seam primary, double-ssd
                               seam for parity; private `_impl` refactor like
                               Mamba-2
src/modules/cache.rs           impl_mamba3: override the two trait methods
examples/grokking/diagnostics  state_pr_forward: MambaCaches::Mamba3 arm;
                               penalty = PR_ℂ(M_phys) (single observable)
```

`Mamba2SsdInput::state_moments` stays as-is (Mamba-2's real closed-form
reference; also the θ≡0 degeneracy target).

## 5. Test plan

Test-gated reference implementations (not shipped):
- **Brute force**: the literal trapezoid recurrence per token (α/β/γ,
  `k/v_state` seeding, rotated B̃), states then rotated to the physical
  frame.
- **Cache-frame `M̃`**: the Mamba-2-style 3-term closed-form Gram
  reduction over the combined injections — the independent
  no-materialisation verifier.
- **Content `M°`**: the same closed form fed raw (source-frame) B.

Ladder:
1. **SSD-level values + grads**: `state_moments_phys` vs brute force;
   grads wrt `x̂/b̂/da/angles/initial_state` through a fixed moments
   loss. Cases: padded `valid_len`, zero + random initial state,
   learnable init.
2. **Trace identity**: `tr M_phys = tr M̃` exactly (frame-invariant) —
   cheap cross-verification of recompute vs closed form on random inputs.
3. **Custom backward ≡ plain autodiff** on the recompute node
   (repo-standard grad-comparison macros).
4. **Degeneracies**: `θ ≡ 0` ⇒ `M_phys ≡ M̃ ≡` the Mamba-2 moments
   (pins de-rotation + pairing); `λ ≡ 1` ⇒ β channels contribute exactly
   nothing (pins channel bookkeeping).
5. **Rank-honesty suite** (constructed inputs): (a) single-plane rotating
   conveyor ⇒ `PR_ℂ(M_phys) ≡ 1`; (b) static physical state under
   ongoing rotation ⇒ `M_phys` rank-true while `M̃` inflates; (c)
   `k`-plane constant writer with long memory ⇒ `PR_ℂ(M_phys) → k`
   while `PR(M°) ≡ 1`.
6. **Block/network parity vs real `step()`** (physical frame via
   `apply(rotation_state, cache)`): `Complex2D` `rope_fraction` 0.5 and
   1.0; `Quaternion4D`; `mimo_rank` 1 and 4; both cache variants in;
   padded seq; streamed-merge. Grad counterpart on the default
   (single-ssd) path — the rotation cumsum/scan gradient through the
   de-rotation is the genuinely new coverage.
7. Existing suites stay green — the seam refactors are the only touch to
   proven code.

## 6. Risks / open questions

- **Custom-backward composition**: the moments node is a *second
  consumer* of the rotation producers (cumsum / `quat_scan`) and of the
  pre-SSD tensors, alongside the SSD's own custom backward — one node
  with two downstream uses should compose in Burn (it did for the
  Mamba-2 `_grad` independent-subgraph argument); verify early.
- **Chunk-state transient** `[b,h,l,p,r]`: bounded but real; the
  chunk-len selector may want a lower cap while moments are on
  (benchmark).
- **Single-SSD injection reconstruction**: confirm the pre-kernel tensors
  suffice to rebuild the shifted β injections without touching the
  kernel-form `scale`/strict-mask machinery.
- **Pairing single source of truth**: `pr_complex` must consume the very
  layout the rotation applies (SISO interleaved/NeoX vs MIMO
  half-and-half/GPT-J; `rope_fraction` boundary) — export `StatePairing`
  from `mamba3/rotation`, never re-derive it in the moments code.
- **Quaternion realification convention**: verify the 4-block layout /
  multiplication side before wiring the quaternionic recombination (the
  §1 formulas assume it); adjust assembly signs, not the accumulation.
- **Pad rotation**: confirm `Δθ = 0` at pads (angles are Δ-scaled) so
  pads are full identity steps for the de-rotation too.
- **Sample-count semantics** unchanged (`count = valid_len · p` per
  `(b,h)` — MIMO ranks share the state).

## 7. Milestones

1. `StatePairing` + `pr_complex` on `StateMoments`, with recombination
   unit tests vs brute-force complex (and quaternionic) moments.
2. `Mamba3MomentsInput::state_moments_phys`: serial recompute forward
   with **plain-autodiff backward** first (correctness at study scale) +
   ladder 1–2, 4–5 (brute force, `M̃`/`M°` verifiers, all test-gated).
3. **Custom recompute backward** — the at-scale execution model —
   + ladder 3.
4. Block seams: single-SSD primary + double-SSD parity; trait overrides,
   network cascade, step-side physical parity (ladder 6).
5. Example: `MambaCaches::Mamba3` arm, penalty = `PR_ℂ(M_phys)`;
   optional grokking Mamba-3 arm (README "Open threads" item) once
   running.
