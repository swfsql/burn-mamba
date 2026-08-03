# mamba3.md — plan: state moments / state-PR for the Mamba-3 `forward()`

Design exploration for extending the Mamba-2 state-moments feature
(`src/modules/state_moments.rs` + `src/mamba2/ssd/moments.rs` + the
`forward_with_state_moments(_grad)` cascade) to Mamba-3. Written before
implementation; grounded in the current `src/mamba3/` code.

## 0. What carries over unchanged

- **`StateMoments` itself.** The Mamba-3 hidden state has the same shape as
  Mamba-2's (`ssm_bhpr`, `[batch, nheads, per_head_dim, state_rank]` — MIMO
  ranks share one state), so `{m2_bhrr, m1_bhr, count}`, `merge`,
  `pool_batch`, `pr(center)` all apply verbatim.
- **The whole upper cascade.** `MambaBlock::block_forward_with_state_moments`
  and `_grad` already exist with panicking defaults; implementing the two
  overrides for `Mamba3` in `modules/cache.rs` lights up
  `Layer`/`Layers`/networks/runtime enums with zero further changes.
- **The example glue, almost.** `diagnostics::state_pr_forward` only needs a
  `MambaCaches::Mamba3` match arm (read `ssm_bhpr` from either cache
  variant) and the `state_pr_penalty` path works as-is.

## 1. What the Mamba-3 state *is* (and which frame we measure it in)

The combined recurrence (mamba3.rs §4), with `B̃ = R·B` the RoPE-rotated,
QK-normed, bias-added key:

```text
hₜ = αₜ hₜ₋₁ + βₜ Σₘ B̃ₜ₋₁[m] ⊗ (xₜ₋₁ ⊙ mimo_x[m]) + γₜ Σₘ B̃ₜ[m] ⊗ (xₜ ⊙ mimo_x[m])
```

Two consequences:

**Frame semantics (the one real decision).** The cache state `ssm_bhpr` —
what `step()` exposes and what the stepwise diagnostic samples — lives in
the *cumulative-rotation frame* (rotations are absorbed into B̃/C̃, angles
continued from the cache accumulator). We define the moments as moments of
**exactly this cache-frame state**. Rationale:

- It is the only definition that satisfies the parity contract
  (`forward` moments ≡ `step`-loop cache accumulation) as-is.
- It falls out for free: the SSD inputs already carry the rotated B̃, so the
  closed form consumes them like Mamba-2 consumes B. Works identically for
  `Complex2D` and `Quaternion4D` (no rotation-specific code); `wrap_angle`
  is value-exact so wrapping is invisible here.
- Invariance facts worth documenting in code:
  - `tr Σ` (hence norms) is frame-invariant — rotations are orthogonal on
    the `r` axis and the per-token Gram trace survives conjugation.
  - **Single-token** PR (the diagnostic's `final_*` columns) is
    frame-invariant — de-rotation at fixed `t` is one fixed rotation of all
    samples.
  - **Pooled-over-time** `tr(Σ²)` (hence pooled PR) is frame-dependent:
    de-rotating each token conjugates its Gram by a *different* `R̄ₜ`.
- The de-rotated ("physical write-directions") alternative is **not**
  expressible in the same closed form: `h_t^{local} = h̃ₜ R̄ₜ` makes every
  cross-`(j,j′)` kernel entry `t`-dependent, breaking the separability the
  Gram reduction needs. If ever wanted: approximate per-chunk (de-rotate by
  the chunk-start frame — exact per chunk, frames mixed only across
  chunks). Deferred; document, don't build.

**Interpretation caveat** for the state-PR diagnostic/penalty on Mamba-3:
RoPE deliberately *spreads* B over rotation orbits, so a healthy pooled PR
in the cache frame is upward-biased vs Mamba-2 (a single physical
direction swept through an orbit reads as rank ≥ 2). Per-head trends over
training remain meaningful; absolute cross-family comparisons are not.

## 2. The closed form generalises with one trick: combined injections

The double-SSD decomposition is the derivation gift. `forward_double_ssd`
already splits the trapezoid into **two standard scalar-decay SSD calls
sharing the same log-decay** `da_bnlh`:

- γ-stream: `Mamba3DoubleSsdInput { v = γ·x (mimo-built), b = B̃, da, … }`
- β-stream: `v = β·x_shifted, b = B̃_shifted, da` — "shift-before-chunking",
  with the cache's `(k_state_bmhr, v_state_bhp)` as the shifted stream's
  first element.

Since both streams share `αₜ` and states are additive, the true state is a
*single* SSD state with a doubled injection channel:

```text
hₜ = dₜ·h₋ + Σ_{j≤t} L[t,j] · Σ_{c ∈ 2m channels} x̂_{j,c} ⊗ b̂_{j,c}
x̂ = concat_m(v_γ, v_β),   b̂ = concat_m(b, b_prev)      (channel axis: 2·mimo_rank)
```

So the Mamba-2 derivation applies **verbatim** with the injection index
flattened from `l` to `l·(2m)` — same mask `L = exp(segsum(da))`, same
three terms:

```text
Σₜ hₜᵀhₜ = (Σₜ dₜ²)·h₋ᵀh₋                                  (carry)
         + h₋ᵀP + Pᵀh₋ ,  P = Σ_c X̂_cᵀ diag(w) B̂_c          (cross)
         + Σ_{c,c′} B̂_cᵀ (K ∘ X̂_c X̂_{c′}ᵀ) B̂_{c′}           (input²)
```

with `w = dᵀL`, `K = LᵀL`, and the first moment likewise
(`u = 1ᵀL`). The `c,c′` double sum is a `2m × 2m` loop of `[l,l]`-kernel
GEMMs (or one flattened `[2ml, 2ml]` kernel — for `l = 64, m ≤ 4` that is
≤ 512², acceptable; pick per benchmark). No new math beyond Mamba-2 —
notably the trapezoid's γ/β *data-dependence* and the data-dependent `Aₜ`
are already absorbed into the pre-scaled `v` and `da`.

Boundary states `h₋` per chunk: same Steps-2–3 recompute as Mamba-2's
moments, once, over the **combined** injections (not per stream — one scan,
one `initial_state`). Pad masking identical (`Δ=0` pads ⇒ identity steps).

**Initial-state ownership caveat:** in `forward_double_ssd` the cache's
`ssm_bhpr` is fed to one stream and zeros to the other (states add). The
moments recompute must count it exactly once — take it from the cache
directly, not from the per-stream input bundles.

## 3. Pathway strategy (double-SSD vs single-SSD)

The moments computation is **pathway-agnostic by construction** (it reads
pre-SSD tensors, like Mamba-2's). The pathway only decides *which tensors
the block has already materialised at the seam*:

- **Double-SSD** (milestone 1): the seam is right before the two `.run()`
  calls — `input_gamma`/`input_beta` are exactly the needed bundles.
  Detach both for the diagnostic variant; leave attached for `_grad`.
- **Single-SSD** (milestone 2): its kernel form feeds raw `v` +
  `gamma_bnlh` + `scale_bnlh` and its accumulator `h′` has different
  mid-sequence semantics — but the moments never touch the kernel. At the
  seam, reconstruct the γ/β-scaled shifted injections (cheap elementwise ×
  + the shift concat, all tensors present pre-kernel) and run the same
  moments fn. The **incoming** cache state converts exactly at call
  boundaries (`h′` ≡ true state there — the existing field-identity `From`
  conversions in `mamba3/cache.rs`).
- `Mamba3::forward` dispatches by cache variant; `forward_with_state_moments`
  mirrors that dispatch. Milestone 1 may `unimplemented!` the single-SSD
  arm with a clear message.

## 4. API / file plan

```text
src/mamba3/moments.rs          Mamba3MomentsInput { xhat_bnlMhp, bhat_bnlMhr,   (M = 2·mimo_rank)
                               da_bnlh, initial_state_bhpr, init_state_hpr }
                               ::state_moments(valid_len) -> StateMoments
                               (+ ::detached(); built from the two double-ssd
                               bundles or from single-ssd pre-kernel tensors)
src/mamba3/mamba3.rs           forward_with_state_moments(_grad) — cache-variant
                               dispatch; forward_double_ssd gains the seam via a
                               private `_impl(with_moments: Option<bool>)` like Mamba-2
src/modules/cache.rs           impl_mamba3: override the two trait methods
examples/grokking/diagnostics  state_pr_forward: add the MambaCaches::Mamba3 arm
```

`Mamba2SsdInput::state_moments` stays separate (its `m = 1`, single-stream
shape doesn't warrant unifying; the shared math is ~100 lines each and the
Mamba-2 file doubles as the readable reference derivation).

## 5. Test plan (mirror the Mamba-2 ladder)

1. **SSD-level, values + grads** (`mamba3/moments/tests.rs`): brute-force =
   the *literal trapezoid recurrence* (α/β/γ per token, `k/v_state` seeding
   for the β boundary, on already-rotated B̃) vs closed form; grads wrt
   `x̂/b̂/da/initial_state` through a fixed moments loss. Cases: padded
   `valid_len`, zero + random initial state, learnable init.
2. **Block-level vs real `step()`**: `forward_with_state_moments` vs
   step-loop `ssm_bhpr` accumulation. Axes to cover (each is one config, not
   a matrix): `Complex2D` `rope_fraction` 0.5 and 1.0; `Quaternion4D`;
   `mimo_rank` 1 and 4; **both cache variants in** (double + converted
   single); padded seq; streamed-merge. Plus the grad counterpart on the
   default path (upstream here = in-proj/QK-norm/rotation — the rotation
   cumsum/scan gradient is the genuinely new coverage).
3. **Degeneracy check**: configure `λ ≡ 1` (β ≡ 0) — the β channels must
   contribute exactly nothing (moments equal the γ-only computation); this
   pins the channel bookkeeping.
4. **PR end-to-end**: `pool_batch().pr()` vs brute-force PR over explicitly
   collected states (cache frame), as in Mamba-2.
5. Existing suites stay green — the `forward_double_ssd` →
   `forward_impl` refactor is the only touch to proven code.

## 6. Risks / open questions

- **Rotation-gradient cost in `_grad` mode**: the moments subgraph retains
  the rotated B̃ producers (cumsum or quaternion scan). `quat_scan`'s
  recompute-backward is a custom node — confirm a *second* consumer of its
  output composes (it should: one node, two downstream uses), else fall
  back to plain scan for the moments branch.
- **Memory**: the `Σ_{c,c′}` kernel is `(2m)²` `[b,n,h,l,l]` Hadamards (or
  one `[2ml, 2ml]`); fine for the grokking scale, benchmark before enabling
  on big MIMO configs.
- **Sample-count semantics**: unchanged (`count = valid_len · p` per
  `(b,h)`) — MIMO ranks share the state, so `m` does not multiply samples.
- **Where γ/β live in single-SSD forward**: verify the pre-kernel tensors
  really suffice to rebuild the shifted β injections without touching the
  kernel-form `scale`/strict-mask machinery (reading says yes; confirm at
  implementation).

## 7. Milestones

1. `Mamba3MomentsInput` + closed form + SSD-level tests (double-ssd inputs).
2. Block seam in `forward_double_ssd` (+ `_grad`), trait overrides,
   block/network parity tests, `λ≡1` and rotation/MIMO axes.
3. Single-SSD seam (reconstruct injections; cache `From`-conversion in).
4. Example: `state_pr_forward` Mamba-3 arm; optional grokking Mamba-3 arm
   (README "Open threads" item) once the penalty story on Mamba-2 is read.
