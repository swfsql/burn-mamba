# Trapezoid as Integration

### What the second discretisation tap of Mamba-3 is, and what it composes with

> Reference note for `burn-mamba`. It is the authority for the decisions of
> this crate around the exponential-trapezoidal discretisation
> (`helpers::trapezoidal_coefficients`, `single_ssd/ssd/diag.rs`). It is also
> the authority for how that discretisation interacts with
> `Mamba3Config::micro_steps`.
>
> It is the companion of [`rotation-as-optimization.md`](rotation-as-optimization.md):
>
> - That note classifies the **quadratic** term of the local objective: the
>   transition, and the algebra of its step size.
> - This note classifies the **linear** term: the write, and the quadrature
>   that makes it.
>
> The split is exact (§3). For this reason, the two notes do not overlap, and
> `RotationKind` and `λ` are independent dials.
>
> [`scripts/mamba-3/trapezoid_as_integration.py`](../../scripts/mamba-3/trapezoid_as_integration.py)
> checks every numbered claim below in float64 (74 checks, with the same
> section numbers). The script depends only on `numpy` and on the equations in
> this note, not on the crate. So the results do not depend on the
> implementation.

---

## Abstract

In one common reading, the state of a linear RNN is a fast weight that is fit
online. In that reading, the step of Mamba-2 is one gradient step on an
isotropic ridge, pulled toward a rank-one target. The exponential-trapezoidal
discretisation of Mamba-3 is usually described as an improvement to
second-order accuracy, or as a two-tap FIR filter on the gradient stream. After
that, it is put aside as orthogonal to everything else.

It is orthogonal, and this note makes the orthogonality precise, not assumed.
Four results:

1. **The trapezoid changes the objective, but only its linear term.** The
   quadratic form stays `G = (1−α)I`: isotropic and `λ`-free. The target
   inflates from rank one to **rank two**. Its second component is the write of
   the previous token, transported into the current frame. So everything that
   the companion note derives from isotropy stays true without change.
2. **`λ` is an operator-splitting parameter.** One trapezoidal step is exactly
   *data half-step → ridge integrated exactly → data half-step*. `λ = 1` is
   Lie–Trotter (and is Mamba-2), `λ = ½` is symmetric **Strang**, and
   `λ = σ(u_t)` is a learned interpolation. This *derives* the error condition
   of the source, `λ = ½ + O(Δ) ⟹ O(Δ³)`, and does not only quote it: Strang
   is second-order only when the split is symmetric.
3. **The step of each sample is paid in two installments, and they
   collapse.** Both installments use the same transport. So sample `s` carries
   one scalar weight `Δ̃_s = λ_sΔ_s + (1−λ_{s+1})Δ_{s+1}`. That scalar is not
   only an observation about the recurrence. It is literally the single-SSD
   composite key scale of the crate, and the same-step γ-correction is its
   diagonal exception.
4. **The tap lattice generalises, and the collapse makes it cheap.** For a tap
   at any lag `d`, transported across *its own* gap, the collapse still holds:
   one scalar per sample, for any lags. So multi-tap trapezoids keep the
   single-SSD form, and only the correction band widens to `max(d)`. So a
   wider tap set is free on the algorithm side. It defines a closed 2×3
   lattice, which exists only at `micro_steps > 1`:
   - a lag-1 tap: absent, gated to within a token, or unrestricted,
   - times a lag-`u` tap: absent or present.

With the unroll of the companion note, the token-level form of Mamba-3 is
`h_i = A(x_i)h_{i−1} + B(x_{i−1}, x_i)`, and **`λ` never appears in `A`**. The
trapezoid and `micro_steps` fill disjoint halves of the normal form of
DeltaProduct.

---

## 1. Scope and results

Like its companion, this note is about the *state update* only. The gate, the
skip, the normalisations and the read are outside the classified object.

| § | claim |
|---|---|
| 3 | The objective, in both display forms. The trapezoid inflates the target to rank two, and does not touch the quadratic term. |
| 4 | `λ` is the splitting parameter: Lie–Trotter at `1`, Strang at `½`. This derives the `O(Δ³)` condition. |
| 5 | The two-installment identity, and its identity with the single-SSD key scale. |
| 6 | The augmented state is a feedforward buffer: `spec(Â) = spec(M) ∪ {0}`. The precise difference from momentum. |
| 7 | The DeltaProduct-style unroll: `A(x_i)`, `B(x_{i−1}, x_i)`, and why the write is not a function of the current token alone. |
| 8 | What `u > 1` does to the semantics of the taps, and what learning can recover. |
| 9 | The tap lattice: the general collapse theorem, its six members, and their cost. |
| 10 | Consequences for this crate. |

This note proposes no change of behaviour to the library. §10 lists what it
does change: documentation, and the reasons for two existing defaults.

---

## 2. Setup and conventions

The note uses one head. The recurrence, with the state in the `h ∈ ℝ^{N×P}`
orientation of the paper (the companion note uses the transposed fast-weight
`S`, and nothing below depends on the orientation):

$$h_t=\alpha_t h_{t-1}+\beta_t\,v_{t-1}+\gamma_t\,v_t,\qquad v_t:=B_tx_t^\top,\qquad y_t=C_t^\top h_t+Dx_t$$

$$\alpha_t=e^{\Delta_tA_t},\qquad \beta_t=(1-\lambda_t)\Delta_t\,\alpha_t,\qquad \gamma_t=\lambda_t\Delta_t,\qquad \lambda_t=\sigma(\hat\lambda_t)\in(0,1)$$

This agrees exactly with `helpers::trapezoidal_coefficients`. `λ ≡ 1`
collapses to Mamba-2.

With a rotation, the scalar `α_t` becomes `M_t = α_tR_t`, with `R_t`
orthogonal, and **the β tap carries `M_t`**: the older sample is
parallel-transported into the current frame before the weight. For this
reason, the caches store the tapped `(B, x)` **without the mass of the tap**,
and apply `ν` at time `t`. They do not store a pre-weighted contribution. The
slot is otherwise not raw:

- `B` carries its own rotation.
- At lag `u` (§9), `x` carries its own decay to the call boundary. This lets a
  tap span a call.

Two conventions come from the companion note:

- The product of step size and curvature is pinned, but its factors are not:
  `η_tρ_t = 1−μ_t` is a **gauge freedom**. Where a claim below depends on one
  factor, the script checks it in at least three gauges.
- "The transition is the quadratic term, the write is the linear term" is the
  standard reading of the correspondence. §3 uses it and does not derive it
  again.

---

## 3. The objective: a rank-two target

**Expanded form.**

$$\mathcal{L}_t(S)=\underbrace{\frac{\rho_t}{2}\lVert S\rVert_F^2}_{\text{quadratic: }G=\rho I}\;-\;\underbrace{\frac{1}{\eta_t}\Big(\gamma_t\big\langle Sk_t,x_t\big\rangle+\beta_t\big\langle Sk_{t-1},x_{t-1}\big\rangle\Big)}_{\text{linear: a two-sample write}},\qquad \eta_t\rho_t=1-\alpha_t$$

**Proximal form.**

$$\mathcal{L}_t(S)=\frac{\rho_t}{2}\Big\lVert S-\underbrace{\frac{\gamma_t\,x_tk_t^\top+\beta_t\,x_{t-1}k_{t-1}^\top}{1-\alpha_t}}_{T_t,\ \operatorname{rank}\,2}\Big\rVert_F^2+\text{const}$$

`S_t = S_{t−1} − η_t∇L_t(S_{t−1})` gives the recurrence exactly, verified in
three gauges. `λ ≡ 1` gives `β = 0`, and the row of Mamba-2 without change.

So: **Mamba-2 regresses toward a rank-one target. The trapezoid regresses
toward a rank-two target, whose second component is the write of the
previous token, transported.**

The scope of the change is the load-bearing part. The script checks it on the
object of the claim, and does not only assert it. The **homogeneous** response
(a nonzero initial state, the input switched off) does not depend on `λ`. The
**driven** response does. `G = (1−α_t)I` (isotropic) for any `λ`. Two
consequences:

- The trapezoid does not touch any argument that the companion note builds on
  isotropy. In particular, Proposition 4 (isotropic curvature ⟹ commuting
  per-micro-step factors ⟹ `u` can change only the write) holds with the
  trapezoid on.
- The usual one-line summary, *"the trapezoid does not change the
  objective"*, is half correct and worth a split: the **quadratic term** does
  not change, but the objective does. The stronger version makes the
  trapezoid look as if it is outside the framework.

---

## 4. `λ` is the splitting parameter

Split the gradient flow of `L_t` into its state-dependent part (the ridge,
linear in `S`) and its state-independent part (the data term, a translation).
One trapezoidal step is exactly, and bit-exactly:

```text
  S′  = S       + (1−λ_t)Δ_t · v_{t−1}      data half-step, left endpoint
  S″  = α_t S′                              ridge, integrated EXACTLY
  S_t = S″      +    λ_t Δ_t · v_t          data half-step, right endpoint
```

That is an **exponential integrator with a split forcing**. The only
state-dependent part of the objective is integrated in closed form. The data
term is state-independent, so it is a pure translation, and it gets a
quadrature. The whole discretisation table is the choice of that quadrature.

| `λ` | splitting | order |
|---|---|---|
| `1` | Lie–Trotter (ridge, then data) | 1st: **this is Mamba-2 / exponential-Euler** |
| `0` | the reversed Lie–Trotter (data, then ridge) | 1st |
| `½` | symmetric **Strang** | 2nd |
| `σ(u_t)` | a learned, per-token, per-head interpolation | — |

All four are verified. This derives the error remark of the source, and does
not only repeat it: **Strang splitting is second-order only when the split is
symmetric**. That is exactly the condition `λ_t = ½ + O(Δ_t)` that the
appendix needs for an `O(Δ³)` local truncation error. The script measures the
state-input quadrature alone, against the exact integral, so that the
right-hand approximation of the transition does not contaminate the fit. The
fitted orders are **1.93** for exponential-Euler and **3.05** for the
trapezoid at `λ = ½`.

Then the ablation that prefers an unconstrained `λ` over `½` is clear. The
block gets a splitting parameter, and it chooses not to use it for symmetry.
Accuracy is available, and `λ` is not used for it.

---

## 5. Two installments, and the single-SSD key scale

Sample `s` reaches the state by two paths: `γ_s` at step `s`, and `β_{s+1}` at
step `s+1`. Their sum:

$$h_T=\sum_{s\le T}w_s\;M_{s+1:T}\,v_s,\qquad w_s=\tilde\Delta_s:=\lambda_s\Delta_s+(1-\lambda_{s+1})\Delta_{s+1}\ (s<T),\qquad w_T=\lambda_T\Delta_T$$

This is exact. It stays true without change for a **non-commuting** rotational
transition, because `β_{s+1}` carries `M_{s+1}`, and the common transport
factors out.

**This is not a lookahead.** The recurrence is strictly causal. `Δ̃_s` is a
retrospective decomposition of a state that is already computed, and it is
*complete* only at `s+1`. The correct reading: the step of sample `s` is
**paid in two installments**, and this buys a causal degree of freedom:

> The trapezoid decouples what the output at time `t` sees of token `t` from
> what the state finally keeps of it. The read `y_t = C_t^⊤h_t` sees only
> `λ_tΔ_t` of the fresh token. The remainder lands after that read. Mamba-2 has
> one number for both.

That is a **read-after-write** statement. It is the one knob in the corpus
that puts a fraction between the write and the read of the same token.

**The collapse is the implementation.** `single_ssd/single_ssd/mod.rs` scales
the key by `scaleₜ = γₜ + (1 − λₜ₊₁)·Δₜ₊₁` (at lag 1), which *is* `Δ̃ₜ`. The
intra-chunk path masks out the `s = t` entry, and `single_ssd/ssd/diag.rs`
adds it back at `γₜ`. The single-SSD pathway exists **because** the two
installments share a transport. §9 turns this into a design constraint.

Two bookkeeping facts, both verified:

- `λ = 1` gives `Δ̃ = Δ` (one installment, Mamba-2).
- `λ = ½` at constant `Δ` gives `Δ̃ = Δ`. The classical trapezoid keeps the
  total mass. It changes *when* the mass lands, not how much.

Finally, `β, γ ≥ 0` always, and `Δ̃_s ≥ λ_sΔ_s`. The second installment adds
to the first and never removes. The trapezoid is a **write-side** mechanism,
and it makes no negative weight. This is a characterisation, not a
scorecard. State tracking is the job of the rotation (companion note §§4, 8),
and no trapezoid variant in this note aims at it.

---

## 6. The augmented state is a feedforward buffer

The expansion of §5 terminates: each sample appears in exactly two terms and
never again (§7). So the buffer `w_t := v_t` gives back a first-order Markov
form:

$$\begin{bmatrix}S_t\\w_t\end{bmatrix}=\underbrace{\begin{bmatrix}M_t&(1-\lambda_t)\Delta_tM_t\\ \mathbf 0&\mathbf 0\end{bmatrix}}_{\hat A_t}\begin{bmatrix}S_{t-1}\\w_{t-1}\end{bmatrix}+\begin{bmatrix}\lambda_t\Delta_t\,v_t\\v_t\end{bmatrix},\qquad \boxed{\operatorname{spec}(\hat A_t)=\operatorname{spec}(M_t)\cup\{0\}^N}$$

Verified, together with the equivalence of the augmented recurrence and the
trapezoid. The `(2,1)` block is **zero: the buffer reads the input, never the
state.**

This is the exact difference from momentum, and it sharpens the standard
description "structurally the same move as momentum". The companion matrix of
heavy ball, `[[1+β−ηρ, −β], [1, 0]]`, has a nonzero `(2,1)` entry. That entry
is precisely where §6 of the companion note gets its complex eigenvalues. So:

> The trapezoid is momentum without the feedback: a velocity buffer that
> accumulates inputs, but never the state. It adds `N` zero eigenvalues and
> nothing else.

Two consequences:

- The trapezoid stays inside the chunkable algebra, and the momentum of
  Titans does not. The auxiliary state reads the inputs but not the state, so
  the transition still does not depend on `S_{t−1}`.
- The mechanism stays in the write: a feedforward buffer cannot move the
  spectrum.

---

## 7. MambaProduct: the unroll

The `micro_steps` (`u`) of the companion note runs `u` full Mamba-3 steps per
token, folded into the sequence axis. Run the derivation of DeltaProduct on
it, with `v_{i,0} := v_{i−1,u}` and `M_{a:b} := M_{i,b}⋯M_{i,a}`:

$$\boxed{\;\begin{aligned}
A(x_i)&=M_{1:u}=\Big(\textstyle\prod_{j=1}^{u}\alpha_{i,j}\Big)\,R_{i,u}\cdots R_{i,1}\\[4pt]
B(x_{i-1},x_i)&=\underbrace{(1-\lambda_{i,1})\Delta_{i,1}\;A(x_i)\,v_{i-1,u}}_{\text{previous token, whole product}}
+\underbrace{\sum_{j=1}^{u-1}\tilde\Delta_{i,j}\;M_{(j+1):u}\,v_{i,j}}_{\text{interior, fully paid}}
+\underbrace{\lambda_{i,u}\Delta_{i,u}\,v_{i,u}}_{\text{freshest, part-paid}}
\end{aligned}\;}$$

Verified against the folded recurrence. Compared with the
`A(x_i) = ∏(I − β_jk_jk_j^⊤)`, `B(x_i) = Σ_j(∏_{k>j}…)β_jk_jv_j^⊤` of
DeltaProduct, there are three structural differences:

1. **The samples enter `A` in DeltaProduct, and cannot in MambaProduct.** The
   `A` of Mamba contains no key, no value and **no `λ`**. This is verified on
   the homogeneous token map under two different `λ` schedules. That is
   isotropy (companion note §7) stated as a normal form: **the trapezoid adds
   nothing to `A`.**
2. **The partial-product shape of `B` is shared, but it tells nothing.** Any
   `u`-step unroll of an affine recurrence gives it. It is not evidence of a
   shared mechanism.
3. **`B` takes two tokens.** Exactly one term carries `x_{i−1}`, and the
   *whole* token product `A(x_i)` transports it. Verified deterministically:
   with `x_i` fixed, a change of only `x_{i−1}` moves `B` by precisely the `β`
   term.

The third puts Mamba-3 outside the form that the delta-rule literature
classifies, `S_t = S_{t−1}M_t + u_tk_t^⊤`, where the write is a function of
`x_t` alone. The reason is not the boring one. A rank-`r` write generalises
that form trivially, but a two-token input window does not. The augmentation
of §6 is the standard way back into the form.

**Expand the cache away.** Substitute until no cached term remains. This gives
the sum of §5 at micro-step resolution, and the expansion **terminates at
depth 2**: each sample appears in exactly two terms. There is no regress: the
structure is FIR, of depth two, at any `u`.

---

## 8. What `u > 1` does to the taps

At `u = 1`, every tap spans a token boundary. At `u > 1`, the taps split, and
the split is uneven:

| tap | pairs | what it is | share |
|---|---|---|---|
| `j = 1` | micro-step `u` of `t−1` ↔ micro-step `1` of `t` | the cross-token filter | `1/u` |
| `j = 2…u` | micro-step `j−1` ↔ `j`, **both inside token `t`** | not a temporal filter | `(u−1)/u` |

Both verified. The boundary keeps three things, and together they show that
the cross-token role is not degraded:

- The tap still carries a full projection of token `t−1`.
- `A(x_i)` transports it, that is, **the same operator that moves the incoming
  state**, a scalar apart (verified by a perturbation of each in turn).
- The same sample has both roles, as at `u = 1`.

What really changes is the interior taps. They pair **two projections of the
same token**. That is not a two-point quadrature of a data stream, but a second
key/value pair from one input, which is the shape of MIMO. So at `u > 1`,
`(u−1)/u` of the taps of the trapezoid stop being an integrator. They become
more within-token write rank, in the slot that `mimo_rank` and `micro_steps`
already fill.

**Learning can recover this.** `λ` is an independent per-micro-step channel
(the in-projection lays out `λ·u`), so the block can specialise. Two limits,
verified exactly:

- `λ_{i,j} = 1` for `j ≥ 2` keeps **one cross-token trapezoid tap per token,
  and plain exponential-Euler inside the token**: the `u = 1` tap semantics on
  top of a `u`-step transition.
- `λ ≡ 1` is plain MambaProduct, with the trapezoid off.

So the change of semantics is inside the parameterisation, and the fold does
not impose it. What this does *not* handle: nothing distinguishes `λ_{i,1}`
from the other `u−1` channels, although only the first has the cross-token
job. The specialisation is reachable, but nothing encourages it.

---

## 9. The tap lattice

§8 raises a design question that exists only at `u > 1`: **which** earlier
sample(s) must the `β` tap read? The answer needs one generalisation, done
correctly.

At lag 1, the tap coefficient is `(1−λ_p)Δ_p·α_p`, and `α_p` is the transition
**over the gap between the two samples**. So the faithful generalisation of a
tap at lag `d` transports it across its own gap:

$$\text{tap at lag }d:\qquad \nu_p\cdot M_{p-d+1:p}\,v_{p-d}$$

The alternative uses the lag-1 coefficient again at lag `d`: one micro-step of
transport across a gap of `d` micro-steps. That breaks the collapse of §5, and
seems to delete the single-SSD pathway. This comes from the wrong transport,
not from a property of any design.

> **Collapse theorem.** For any set of taps, each at its own lag `d` with the
> coefficient `ν` and transported across its own gap, the total contribution
> of sample `s` to `h_T` is
> `[γ_s + Σ_taps ν_{s+d}] · M_{s+1:T} v_s`: **one scalar per sample, for any
> lags.** So any multi-tap trapezoid keeps the single-SSD form, and only the
> correction band widens to `max(d)`.

Verified for five tap sets, combinations included. More taps for the
trapezoid is a standing suggestion in this literature (the banded factor of
the mask is the obvious thing to widen). The theorem says what it costs:
nothing on the algorithm side, if the transport rule above is kept.

**The six members.** The tap set is a choice about two taps, so the lattice is
a product: a lag-1 (*horizontal*) tap that is absent, gated to within a token,
or unrestricted, times a lag-`u` (*vertical*) tap that is absent or present.
All six collapse, and all six are really different models at `u > 1`. The
degenerate cases rank them:

| member | taps | crossings of a token, per token | at `u = 1` |
|---|---|---|---|
| `None` | — | `0` | no trapezoid, at any `u` |
| `HorizontalReset` | lag 1, closed at the first micro-step of each token | `0` | is `None` |
| `HorizontalCarryOver` (default) | lag 1, always | `1` | *is* the baseline |
| `Vertical` | lag `u`, always | `u` | is `HorizontalCarryOver` |
| `VerticalPlusHorizontalReset` | both, the lag-1 tap closed | `u` | is `HorizontalCarryOver` |
| `VerticalPlusHorizontalCarryOver` | both, ungated | `u+1` | is `HorizontalCarryOver` |

Every degeneracy is verified:

- `HorizontalReset` alone has no cross-token path at all. So it cannot do the
  job that the trapezoid was introduced for. It is a component, not an
  alternative.
- `Vertical` gives the `u = 1` semantics again at every micro-step: `u`
  parallel filters at token resolution, one per micro-step channel, with a
  `u`-slot tap cache.
- The two-tap members give each job its own coefficient, and do not make them
  share one `λ`. A second per-(head, micro-step) mass `μ` mixes them. That
  makes `VerticalPlusHorizontalCarryOver` the **join**: it *is*
  `HorizontalCarryOver` at `μ ≡ 1` and `Vertical` at `μ ≡ 0`, both exactly.
  So the choice that the rest of the column makes at configuration time, the
  join makes by descent.

**The cost of the lag-`u` tap**, all verified:

- The key scale is `γ_s + (1−λ_{s+u})Δ_{s+u}`: the lag-1 formula with the
  index shifted `s+1 → s+u`.
- The same-step γ-correction generalises from a **diagonal** to a
  **`u`-wide band** (subtract the mass that is still unpaid where `t−s < u`).
  At `u = 1` the band *is* the diagonal, so `diag.rs` does not change. The
  band does not have to enter the kernel. A folded pass keeps only the reads
  at the last micro-step of each token, and at those reads the band *is* that
  token. So it is one intra-token contraction afterwards, with no mask change,
  no chunk-length constraint and no cross-chunk term.
- Double-SSD shifts by `u`, not by `1`.
- The tap cache becomes a `u`-deep FIFO, and the cross-chunk seed needs `u`
  positions.

**The cost of the second tap** is one more per-(head, micro-step) scalar
channel. On the single-SSD pathway it costs nothing else. The collapse gives
one scalar per sample for any number of taps. So the key scale gets a term,
and the pass count does not. The band stays the lag-`u` mass alone (a lag-1
installment has already landed at every read that the band covers). On the
double-SSD pathway, the two taps have different shifts and cannot share a
pass, so it runs **three** SSD calls. That asymmetry is the honest price of
the join, and it shows where the join belongs.

The map is not a ranking. Its purpose is the invariant: **protect the
collapse**. The single-SSD pathway is built on it, and it stays true for any
tap set that transports each tap over its own gap.

How to *parameterise* the members is a separate question, and this note does
not settle it. The header of `src/mamba3/trapezoid.rs` does. The checks for
that choice (mass conservation, the fallback of the gate, and the two-tap key
scale) are under this section in the script.

---

## 10. Consequences for this crate

### 10.1 What the trapezoid is, and how to describe it

The trapezoid is a quadrature of the state-input integral, or equivalently an
operator splitting with the parameter `λ` (§4). It is entirely in the linear
term of the local objective (§3). "A two-tap FIR filter on the gradient
stream" is an accurate description and the correct one-line summary.
"Structurally the same move as momentum" needs the qualifier of §6: the entry
that makes momentum what it is, is exactly the entry that the trapezoid does
not have.

### 10.2 The `Δ̃` collapse is load-bearing, not decorative

The composite key scale of `single_ssd` *is* `Δ̃`, and `diag.rs` is its
diagonal exception (§5). A future change to which samples the write
integrates must keep the collapse, or give up that pathway. §9 states the
condition that keeps it: transport each tap over its own gap.

### 10.3 The trapezoid and `micro_steps` are disjoint dials

`λ` never appears in `A(x_i)`. `u` multiplies factors in `A` and terms in `B`
(§7). That is the structural reason why the fold needed no special case, and
it is the precise sense in which the two compose. One nuance is worth a note
where `micro_steps` is documented. Under the default member, only `1/u` of the
taps still cross a token at `u > 1`, and the interior taps change kind (§8).
The lattice of §9 exists to make that choice.

### 10.4 What does not follow

- No numerical behaviour changes. Every claim here is about the recurrence as
  implemented.
- The trapezoid is not a route to state tracking, and this note does not
  analyse it as one. Its weights are non-negative (§5), and its augmentation
  is nilpotent (§6). The circulating component stays with the rotation, as the
  companion note says.
- The reading says nothing about the read. `C` never appears in any objective
  here, exactly as in the companion note.

---

## 11. Reproduction

```bash
python3 scripts/mamba-3/trapezoid_as_integration.py
```

`numpy` only, float64 throughout, 74 checks. It exits non-zero on failure. The
section numbers in its output are the same as in this document. The script
encodes the recurrence of §2 directly, and never imports the crate. So the
Rust test suites assert, separately, that it agrees with the implementation
(`src/mamba3/double_ssd/`, `src/mamba3/single_ssd/`,
`src/mamba3/product/tests.rs`).

---

## References

- *Mamba-3*. arXiv:2603.15569. §*Exponential-Trapezoidal Discretization* and
  its appendix give Proposition 1, the discretisation table, the mask
  factorisation `L = L₁L₂`, and the ablation of the `λ` parameterisation used
  in §4.
- T. Dao, A. Gu. *Transformers are SSMs*. arXiv:2405.21060. The SSD form that
  contains the mask factorisation.
- J. Siems, T. Carstensen, A. Zela, F. Hutter, M. Pontil, R. Grazzi.
  *DeltaProduct: Improving State-Tracking in Linear RNNs via Householder
  Products*, 2025. The unroll that §7 mirrors, and the `A(x_i)`/`B(x_i)`
  normal form that §7 compares with.
- G. Strang. *On the construction and comparison of difference schemes*, 1968.
  The splitting of §4: second order needs the symmetric split.
- E. Hairer, C. Lubich, G. Wanner. *Geometric Numerical Integration*, 2006.
  Exponential integrators and splitting order conditions.
- B. T. Polyak. *Some methods of speeding up the convergence of iteration
  methods*, 1964. Heavy ball, the contrast in §6.
- E. Süli, D. Mayers. *An Introduction to Numerical Analysis*, 2003. The
  trapezoidal rule as the source states it.

**Note on novelty.** The equations of the trapezoid, the mask factorisation
`L = L₁L₂` and the error rate are from the source paper. Three readings are
prior, and none is a new result in optimization or numerical analysis alone:
the added term as a two-tap FIR filter on the gradient stream, the observation
that under a complex transition the older tap arrives parallel-transported,
and a wider band as the natural extension. This note assembles:

- `λ` identified as an operator-splitting parameter, which derives the error
  condition of the source and does not only quote it (§4),
- the two-installment collapse, and its identity with the key scale of this
  crate (§5),
- the feedforward-buffer characterisation, and its exact difference from
  momentum (§6),
- the token-level unroll, and the disjointness from `micro_steps` that follows
  because `λ` is absent from `A` (§7),
- the collapse theorem, which prices the wider band at zero (§9).
