# Trapezoid as Integration

### What Mamba-3's second discretisation tap is, and what it composes with

> Reference note for `burn-mamba`. It is the authority for this crate's decisions
> around the exponential-trapezoidal discretisation
> (`helpers::trapezoidal_coefficients`, `single_ssd/ssd/diag.rs`) and for how that
> discretisation interacts with `Mamba3Config::micro_steps`.
>
> Companion to [`rotation-as-optimization.md`](rotation-as-optimization.md). That
> note classifies the **quadratic** term of the local objective — the transition,
> and the algebra its step size lives in. This one classifies the **linear** term —
> the write, and the quadrature that produces it. The split is exact (§3), which is
> why the two notes do not overlap and why `RotationKind` and `λ` are independent
> dials.
>
> Every numbered claim below is checked in float64 by
> [`scripts/trapezoid_as_integration.py`](../scripts/trapezoid_as_integration.py)
> (54 checks, section numbers match). The script depends only on `numpy` and on the
> equations reproduced here — not on the crate — so the results stand independently
> of the implementation.

---

## Abstract

Under the reading in which a linear RNN's state is a fast weight being fit online,
Mamba-2's step is one gradient step on an isotropic ridge pulled toward a rank-one
target. Mamba-3's exponential-trapezoidal discretisation is usually described as a
second-order accuracy improvement, or as a two-tap FIR filter on the gradient
stream, and then set aside as orthogonal to everything else.

It is orthogonal, and this note makes the orthogonality precise rather than
assumed. Four results:

1. **The trapezoid changes the objective, but only its linear term.** The quadratic
   form stays `G = (1−α)I` — isotropic, `λ`-free — and the target inflates from
   rank one to **rank two**, its second component being the previous token's write
   transported into the current frame. Everything the companion note derives from
   isotropy therefore survives verbatim.
2. **`λ` is an operator-splitting parameter.** One trapezoidal step is exactly
   *data half-step → ridge integrated exactly → data half-step*. `λ = 1` is
   Lie–Trotter (and is Mamba-2), `λ = ½` is symmetric **Strang**, and `λ = σ(u_t)`
   is a learned interpolation. This *derives* the source's `λ = ½ + O(Δ) ⟹ O(Δ³)`
   error condition instead of quoting it: Strang is second-order only when the
   split is symmetric.
3. **Each sample's step is paid in two installments, and they collapse.** Because
   both installments ride the same transport, sample `s` carries one scalar weight
   `Δ̃_s = λ_sΔ_s + (1−λ_{s+1})Δ_{s+1}`. That scalar is not an observation about
   the recurrence — it is literally the crate's single-SSD composite key scale, and
   the same-step γ-correction is its diagonal exception.
4. **The tap lattice generalises, and the collapse is what makes it cheap.** For a
   tap at any lag `d`, transported across *its own* gap, the collapse still holds:
   one scalar per sample, whatever the lags. So multi-tap trapezoids keep the
   single-SSD form; only the correction band widens to `max(d)`. A wider tap set is
   therefore free on the algorithm side, and it defines a three-point design space
   — *vertical*, *reset-horizontal*, *carry-over-horizontal* — that only exists at
   `micro_steps > 1`.

Combined with the companion note's unroll, Mamba-3's token-level form is
`h_i = A(x_i)h_{i−1} + B(x_{i−1}, x_i)`, in which **`λ` never appears in `A`**. The
trapezoid and `micro_steps` occupy disjoint halves of DeltaProduct's normal form.

---

## 1. Scope and results

This note concerns the *state update* only, as its companion does: the gate, the
skip, the normalisations and the read are outside the object being classified.

| § | claim |
|---|---|
| 3 | The objective, both display forms. The trapezoid inflates the target to rank two and leaves the quadratic term alone. |
| 4 | `λ` is the splitting parameter: Lie–Trotter at `1`, Strang at `½`. Derives the `O(Δ³)` condition. |
| 5 | The two-installment identity, and its identity with the single-SSD key scale. |
| 6 | The augmented state is a feedforward buffer: `spec(Â) = spec(M) ∪ {0}`. The precise difference from momentum. |
| 7 | The DeltaProduct-style unroll: `A(x_i)`, `B(x_{i−1}, x_i)`, and why the write is not a function of the current token alone. |
| 8 | What `u > 1` does to the tap semantics, and what is recoverable by learning. |
| 9 | The tap lattice: the general collapse theorem, the three patterns, and their cost. |
| 10 | Consequences for this crate. |

Nothing here proposes a behavioural change to the library. §10 lists what it does
change: documentation, and the reasons behind two existing defaults.

---

## 2. Setup and conventions

One head. The recurrence, with the state in the paper's `h ∈ ℝ^{N×P}` orientation
(the companion note uses the transposed fast-weight `S`; nothing below depends on
which):

$$h_t=\alpha_t h_{t-1}+\beta_t\,v_{t-1}+\gamma_t\,v_t,\qquad v_t:=B_tx_t^\top,\qquad y_t=C_t^\top h_t+Dx_t$$

$$\alpha_t=e^{\Delta_tA_t},\qquad \beta_t=(1-\lambda_t)\Delta_t\,\alpha_t,\qquad \gamma_t=\lambda_t\Delta_t,\qquad \lambda_t=\sigma(\hat\lambda_t)\in(0,1)$$

matching `helpers::trapezoidal_coefficients` exactly. `λ ≡ 1` collapses to Mamba-2.

Where a rotation is present the scalar `α_t` becomes `M_t = α_tR_t` with `R_t`
orthogonal, and **the β tap carries `M_t`** — the older sample is parallel-
transported into the current frame before being weighted. This is why the caches
store the raw `(B_{t−1}, x_{t−1})` and re-weight at time `t` rather than storing a
pre-decayed contribution.

Two conventions carried over from the companion note. The step-size/curvature
product is pinned but its factors are not: `η_tρ_t = 1−μ_t` is a **gauge freedom**,
and every claim below is checked in at least three gauges where it depends on one.
And "the transition is the quadratic term, the write is the linear term" is the
standard reading of the correspondence, which §3 uses without re-deriving.

---

## 3. The objective — a rank-two target

**Expanded form.**

$$\mathcal{L}_t(S)=\underbrace{\frac{\rho_t}{2}\lVert S\rVert_F^2}_{\text{quadratic: }G=\rho I}\;-\;\underbrace{\frac{1}{\eta_t}\Big(\gamma_t\big\langle Sk_t,x_t\big\rangle+\beta_t\big\langle Sk_{t-1},x_{t-1}\big\rangle\Big)}_{\text{linear: a two-sample write}},\qquad \eta_t\rho_t=1-\alpha_t$$

**Proximal form.**

$$\mathcal{L}_t(S)=\frac{\rho_t}{2}\Big\lVert S-\underbrace{\frac{\gamma_t\,x_tk_t^\top+\beta_t\,x_{t-1}k_{t-1}^\top}{1-\alpha_t}}_{T_t,\ \operatorname{rank}\,2}\Big\rVert_F^2+\text{const}$$

`S_t = S_{t−1} − η_t∇L_t(S_{t−1})` reproduces the recurrence exactly, verified in
three gauges. Setting `λ ≡ 1` gives `β = 0` and recovers Mamba-2's row verbatim.

So: **Mamba-2 regresses toward a rank-one target; the trapezoid regresses toward a
rank-two target whose second component is the previous token's write, transported.**

The scope of the change is the load-bearing part, and it is checked on the object
of the claim rather than asserted: the **homogeneous** response (nonzero initial
state, input switched off) is independent of `λ`, while the **driven** response is
not. `G = (1−α_t)I` — isotropic — whatever `λ` does. Two consequences:

- Every argument the companion note builds on isotropy is untouched by the
  trapezoid. In particular Proposition 4 (isotropic curvature ⟹ commuting
  per-micro-step factors ⟹ `u` can only change the write) holds with the trapezoid
  switched on.
- The usual one-line summary — *"the trapezoid does not change the objective"* — is
  half right and worth splitting: the **quadratic term** does not change; the
  objective does. Reading the stronger version is what makes the trapezoid look
  like it falls outside the framework altogether.

---

## 4. `λ` is the splitting parameter

Split the gradient flow of `L_t` into its state-dependent part (the ridge, linear
in `S`) and its state-independent part (the data term, a translation). One
trapezoidal step is exactly, and bit-exactly:

```text
  S′  = S       + (1−λ_t)Δ_t · v_{t−1}      data half-step, left endpoint
  S″  = α_t S′                              ridge, integrated EXACTLY
  S_t = S″      +    λ_t Δ_t · v_t          data half-step, right endpoint
```

That is an **exponential integrator with a split forcing**: the only
state-dependent part of the objective is integrated in closed form, and the data
term — which is state-independent, hence a pure translation — is given a
quadrature. The whole discretisation table is the choice of that quadrature.

| `λ` | splitting | order |
|---|---|---|
| `1` | Lie–Trotter (ridge, then data) | 1st — **this is Mamba-2 / exponential-Euler** |
| `0` | the reversed Lie–Trotter (data, then ridge) | 1st |
| `½` | symmetric **Strang** | 2nd |
| `σ(u_t)` | a learned, per-token, per-head interpolation | — |

All four verified. This derives the source's error remark rather than restating it:
**Strang splitting is second-order only when the split is symmetric**, which is
exactly the condition `λ_t = ½ + O(Δ_t)` that the appendix needs for `O(Δ³)` local
truncation error. Measured on the state-input quadrature in isolation (against the
exact integral, so the transition's own right-hand approximation does not
contaminate the fit): fitted orders **1.93** for exponential-Euler and **3.05** for
the trapezoid at `λ = ½`.

The ablation that prefers unconstrained `λ` over `½` is then legible: the block is
being given a splitting parameter and is choosing not to spend it on symmetry.
Accuracy is available and is not what `λ` is used for.

---

## 5. Two installments, and the single-SSD key scale

Sample `s` reaches the state by two paths — `γ_s` at step `s`, and `β_{s+1}` at
step `s+1`. Summing them:

$$h_T=\sum_{s\le T}w_s\;M_{s+1:T}\,v_s,\qquad w_s=\tilde\Delta_s:=\lambda_s\Delta_s+(1-\lambda_{s+1})\Delta_{s+1}\ (s<T),\qquad w_T=\lambda_T\Delta_T$$

Exact, and it survives a **non-commuting** rotational transition verbatim, because
`β_{s+1}` carries `M_{s+1}` and the common transport factors out.

**This is not a lookahead.** The recurrence is strictly causal; `Δ̃_s` is a
retrospective decomposition of an already-computed state, and is only *complete* at
`s+1`. The right reading is that sample `s`'s step is **paid in two installments**,
and the causal degree of freedom this buys is:

> The trapezoid decouples what the output at time `t` sees of token `t` from what
> the state eventually keeps of it. The read `y_t = C_t^⊤h_t` sees only `λ_tΔ_t` of
> the fresh token; the rest lands after that read. Mamba-2 has one number for both.

That is a **read-after-write** statement, and it is the one knob in the corpus that
puts a fraction between the write and the read of the same token.

**The collapse is the implementation.** `single_ssd/single_ssd/mod.rs` scales the
key by `scaleₜ = γₜ + (1 − λₜ₊₁)·Δₜ₊₁`, which *is* `Δ̃ₜ`, and `single_ssd/ssd/diag.rs`
adds back the `s = t` entry at `γₜ` after the intra-chunk path masks it out. The
single-SSD pathway exists **because** the two installments share a transport. §9
turns this into a design constraint.

Two bookkeeping facts, both verified: `λ = 1` gives `Δ̃ = Δ` (one installment,
Mamba-2), and `λ = ½` at constant `Δ` gives `Δ̃ = Δ` — the classical trapezoid
preserves total mass, redistributing *when* the mass lands rather than how much.

Finally, `β, γ ≥ 0` always and `Δ̃_s ≥ λ_sΔ_s`: the second installment is a top-up,
never a rollback. The trapezoid is a **write-side** mechanism and manufactures no
negative weight. This is a characterisation, not a scorecard — state tracking is
the rotation's job (companion note §§4, 8), and no trapezoid variant in this note
is aimed at it.

---

## 6. The augmented state is a feedforward buffer

Because the expansion of §5 terminates (each sample appears in exactly two terms
and never again — §7), first-order Markov form is restorable by carrying the buffer
`w_t := v_t`:

$$\begin{bmatrix}S_t\\w_t\end{bmatrix}=\underbrace{\begin{bmatrix}M_t&(1-\lambda_t)\Delta_tM_t\\ \mathbf 0&\mathbf 0\end{bmatrix}}_{\hat A_t}\begin{bmatrix}S_{t-1}\\w_{t-1}\end{bmatrix}+\begin{bmatrix}\lambda_t\Delta_t\,v_t\\v_t\end{bmatrix},\qquad \boxed{\operatorname{spec}(\hat A_t)=\operatorname{spec}(M_t)\cup\{0\}^N}$$

Verified, along with the equivalence of the augmented recurrence to the trapezoid.
The `(2,1)` block is **zero: the buffer reads the input, never the state.**

This is the exact difference from momentum, and it sharpens the standard
"structurally the same move as momentum" description. Heavy ball's companion matrix
`[[1+β−ηρ, −β], [1, 0]]` has a nonzero `(2,1)` entry, and that entry is precisely
where the companion note's §6 gets its complex eigenvalues. So:

> The trapezoid is momentum with the feedback removed — a velocity buffer that
> accumulates inputs but never the state. It adds `N` zero eigenvalues and nothing
> else.

Two consequences. It is why the trapezoid stays inside the chunkable algebra where
Titans' momentum does not: the auxiliary state reads the inputs but not the state,
so the transition still does not depend on `S_{t−1}`. And it is why the mechanism
is confined to the write: a feedforward buffer cannot move the spectrum.

---

## 7. MambaProduct — the unroll

The companion note's `micro_steps` (`u`) runs `u` full Mamba-3 steps per token,
folded into the sequence axis. Running DeltaProduct's own derivation on it, with
`v_{i,0} := v_{i−1,u}` and `M_{a:b} := M_{i,b}⋯M_{i,a}`:

$$\boxed{\;\begin{aligned}
A(x_i)&=M_{1:u}=\Big(\textstyle\prod_{j=1}^{u}\alpha_{i,j}\Big)\,R_{i,u}\cdots R_{i,1}\\[4pt]
B(x_{i-1},x_i)&=\underbrace{(1-\lambda_{i,1})\Delta_{i,1}\;A(x_i)\,v_{i-1,u}}_{\text{previous token, whole product}}
+\underbrace{\sum_{j=1}^{u-1}\tilde\Delta_{i,j}\;M_{(j+1):u}\,v_{i,j}}_{\text{interior, fully paid}}
+\underbrace{\lambda_{i,u}\Delta_{i,u}\,v_{i,u}}_{\text{freshest, part-paid}}
\end{aligned}\;}$$

Verified against the folded recurrence. Set beside DeltaProduct's
`A(x_i) = ∏(I − β_jk_jk_j^⊤)`, `B(x_i) = Σ_j(∏_{k>j}…)β_jk_jv_j^⊤`, three
structural differences:

1. **The samples enter `A` in DeltaProduct and cannot in MambaProduct.** Mamba's
   `A` contains no key, no value and **no `λ`** — verified on the homogeneous token
   map under two different `λ` schedules. That is isotropy (companion note §7)
   stated as a normal form: **the trapezoid contributes nothing to `A`.**
2. **The partial-product shape of `B` is shared but uninformative** — it is what
   any `u`-step unroll of an affine recurrence gives, and is not evidence of a
   shared mechanism.
3. **`B` takes two tokens.** Exactly one term carries `x_{i−1}`, and it is
   transported by the *whole* token product `A(x_i)`. Verified deterministically:
   holding `x_i` fixed and moving only `x_{i−1}` moves `B` by precisely the `β`
   term.

The third puts Mamba-3 outside the form the delta-rule literature classifies,
`S_t = S_{t−1}M_t + u_tk_t^⊤` with the write a function of `x_t` alone — and not for
the boring reason: a rank-`r` write generalises that form trivially, a two-token
input window does not. §6's augmentation is the standard way back in.

**Expanding the cache away.** Substituting until no cached term remains gives the
§5 sum at micro-step resolution, and the expansion **terminates at depth 2** — each
sample appears in exactly two terms. There is no regress: the structure is FIR,
depth two, at any `u`.

---

## 8. What `u > 1` does to the taps

At `u = 1` every tap spans a token boundary. At `u > 1` the taps split, and the
split is uneven:

| tap | pairs | what it is | share |
|---|---|---|---|
| `j = 1` | micro-step `u` of `t−1` ↔ micro-step `1` of `t` | the cross-token filter | `1/u` |
| `j = 2…u` | micro-step `j−1` ↔ `j`, **both inside token `t`** | not a temporal filter | `(u−1)/u` |

Both verified. Three things are preserved at the boundary, and together they say
the cross-token role is not degraded: the tap still carries a full projection of
token `t−1`; it is transported by `A(x_i)`, i.e. **the same operator that moves the
incoming state**, a scalar apart (verified by perturbing each in turn); and the
same sample serves both roles, as at `u = 1`.

What genuinely changes is the interior taps. They pair **two projections of the
same token** — not a two-point quadrature of a data stream but a second key/value
pair drawn from one input, which is MIMO's shape. So at `u > 1`, `(u−1)/u` of the
trapezoid's taps stop being an integrator and become extra within-token write rank,
in the slot `mimo_rank` and `micro_steps` already occupy.

**This is recoverable by learning.** `λ` is an independent per-micro-step channel
(the in-projection lays out `λ·u`), so the block can specialise. Two limits,
verified exactly:

- `λ_{i,j} = 1` for `j ≥ 2` leaves **one cross-token trapezoid tap per token and
  plain exponential-Euler inside it** — the `u = 1` tap semantics on top of a
  `u`-step transition;
- `λ ≡ 1` is plain MambaProduct, trapezoid off.

So the change in semantics is inside the parameterisation rather than imposed by
the fold. What is *not* handled is that nothing distinguishes `λ_{i,1}` from the
other `u−1` channels, although only the first has the cross-token job: the
specialisation is reachable but not encouraged.

---

## 9. The tap lattice

§8 raises a design question that only exists at `u > 1`: **which** earlier sample
should the `β` tap read? Answering it needs one generalisation done correctly.

At lag 1 the tap coefficient is `(1−λ_p)Δ_p·α_p`, and `α_p` is the transition
**over the gap between the two samples**. So the faithful generalisation of a tap
at lag `d` transports it across its own gap:

$$\text{tap at lag }d:\qquad \nu_p\cdot M_{p-d+1:p}\,v_{p-d}$$

Reusing the lag-1 coefficient at lag `d` instead — one micro-step of transport
across a `d`-micro-step gap — breaks the §5 collapse and looks like it deletes the
single-SSD pathway. That is an artifact of the wrong transport, not a property of
any design.

> **Collapse theorem.** For any set of taps, each at its own lag `d` with
> coefficient `ν` and transported across its own gap, sample `s`'s total
> contribution to `h_T` is
> `[γ_s + Σ_taps ν_{s+d}] · M_{s+1:T} v_s` — **one scalar per sample, whatever the
> lags.** Hence any multi-tap trapezoid keeps the single-SSD form, and only the
> correction band widens to `max(d)`.

Verified for five tap sets, including combinations. Widening the trapezoid to more
taps is a standing suggestion in this literature — the mask's banded factor is the
obvious thing to widen — and the theorem says what it costs: nothing on the
algorithm side, provided the transport rule above is respected.

**The three patterns.** All three collapse, all three are genuinely different
models, and the degenerate cases rank them:

| pattern | tap | cross-token taps | at `u = 1` |
|---|---|---|---|
| **carry-over-horizontal** (today) | lag 1, always | `1/u` | — it *is* the baseline |
| **reset-horizontal** | lag 1, suppressed at `j = 1` | **none** | degenerates to no trapezoid |
| **vertical** | lag `u`, always | `u/u` | coincides with carry-over-horizontal |

Both degeneracies verified. Reset-horizontal alone has no cross-token path at all,
so it cannot do the job the trapezoid was introduced for; it is a component, not an
alternative. Vertical restores the `u = 1` semantics at every micro-step: `u`
parallel filters at token resolution, one per micro-step channel, with a `u`-slot
tap cache.

**Cost of vertical**, all verified:

- key scale `= γ_s + (1−λ_{s+u})Δ_{s+u}` — today's formula with the index shifted
  `s+1 → s+u`;
- the same-step γ-correction generalises from a **diagonal** to a **`u`-wide band**
  (subtract `Δ̃_s − γ_s` where `t−s < u`); at `u = 1` the band *is* the diagonal,
  i.e. `diag.rs` unchanged;
- double-SSD shifts by `u` instead of `1` (`double_ssd/double_ssd/mod.rs` builds the β term by
  a `narrow` plus a prepended cached element);
- the tap cache gains a `u` dimension, and the cross-chunk seed needs `u` positions.

All bounded; nothing structural is lost. The combinations are also live —
`vertical + reset-horizontal` gives each job its own coefficient instead of making
them share one `λ`, at the cost of one extra per-micro-step scalar channel and a
third nonzero per row in the mask's banded factor.

None of this is a recommendation. It is a map of what the lattice costs, and the
one line worth carrying out of it is that **the collapse is the invariant to
protect**: it is what the single-SSD pathway is built on, and it survives any tap
set that transports each tap over its own gap.

---

## 10. Consequences for this crate

### 10.1 What the trapezoid is, and how to describe it

A quadrature of the state-input integral, equivalently an operator splitting whose
parameter is `λ` (§4), living entirely in the linear term of the local objective
(§3). Describing it as "a two-tap FIR filter on the gradient stream" is accurate
and is the right one-line summary; describing it as "structurally the same move as
momentum" needs §6's qualifier, since the entry that makes momentum momentum is
exactly the one the trapezoid does not have.

### 10.2 The `Δ̃` collapse is load-bearing, not decorative

`single_ssd`'s composite key scale *is* `Δ̃`, and `diag.rs` is its diagonal
exception (§5). Any future change to which samples the write integrates has to
preserve the collapse or give up that pathway — §9 states the condition that keeps
it (transport each tap over its own gap).

### 10.3 The trapezoid and `micro_steps` are disjoint dials

`λ` never appears in `A(x_i)`; `u` multiplies factors in `A` and terms in `B`
(§7). That is the structural reason the fold needed no special-casing, and it is
the precise sense in which the two compose. The one nuance worth documenting where
`micro_steps` is documented: at `u > 1` only `1/u` of the taps still cross a token,
and the interior ones change kind (§8).

### 10.4 What does not follow

- No numerical behaviour changes. Every claim here is about the recurrence as
  implemented.
- The trapezoid is not a route to state tracking and is not analysed as one. Its
  weights are non-negative (§5) and its augmentation is nilpotent (§6); the
  circulating component remains the rotation's, per the companion note.
- The reading says nothing about the read. `C` never appears in any objective here,
  exactly as in the companion note.

---

## 11. Reproduction

```bash
python3 scripts/trapezoid_as_integration.py
```

`numpy` only; float64 throughout; 54 checks; exits non-zero on failure. Section
numbers in its output match this document's. The script encodes the recurrence from
§2 directly and never imports the crate, so agreement between it and the
implementation is asserted separately, by the Rust test suites
(`src/mamba3/double_ssd/`, `src/mamba3/single_ssd/`, `src/mamba3/product/tests.rs`).

---

## References

- *Mamba-3*. arXiv:2603.15569. §*Exponential-Trapezoidal Discretization* and its
  appendix carry Proposition 1, the discretisation table, the mask factorisation
  `L = L₁L₂`, and the `λ` parameterisation ablation used in §4.
- T. Dao, A. Gu. *Transformers are SSMs*. arXiv:2405.21060. The SSD form the mask
  factorisation lives in.
- J. Siems, T. Carstensen, A. Zela, F. Hutter, M. Pontil, R. Grazzi.
  *DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products*,
  2025. The unroll §7 mirrors, and the `A(x_i)`/`B(x_i)` normal form it is set
  against.
- G. Strang. *On the construction and comparison of difference schemes*, 1968.
  The splitting of §4; second order requires the symmetric split.
- E. Hairer, C. Lubich, G. Wanner. *Geometric Numerical Integration*, 2006.
  Exponential integrators and splitting order conditions.
- B. T. Polyak. *Some methods of speeding up the convergence of iteration methods*,
  1964. Heavy ball; the contrast in §6.
- E. Süli, D. Mayers. *An Introduction to Numerical Analysis*, 2003. The
  trapezoidal rule as the source states it.

**Note on novelty.** The trapezoid's equations, the mask factorisation `L = L₁L₂`
and the error rate are the source paper's. Reading the added term as a two-tap FIR
filter on the gradient stream, observing that under a complex transition the older
tap arrives parallel-transported, and proposing a wider band as the natural
extension are all prior, and none is a new result in optimization or numerical
analysis on its own. What this note assembles is: `λ` identified as an
operator-splitting parameter, which derives the source's own error condition rather
than quoting it (§4); the two-installment collapse and its identity with the key
scale this crate ships (§5); the feedforward-buffer characterisation and its exact
difference from momentum (§6); the token-level unroll, and the disjointness from
`micro_steps` that follows from `λ` being absent in `A` (§7); and the collapse
theorem, which prices the wider band at zero (§9).
