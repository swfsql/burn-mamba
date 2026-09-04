# MIMO as Batch Size

### What Mamba-3's rank dial does to the local objective, and what it composes with

> Reference note for `burn-mamba`. It is the authority for this crate's decisions
> around `Mamba3Config::mimo_rank` (`helpers::{build_v_with_mimo, mimo_outer_sum}`,
> the `mimo_x_hmp` / `mimo_z_hmp` / `mimo_o_hmp` masks) and for how the rank dial
> interacts with `Mamba3Config::micro_steps` and `RotationKind`.
>
> Third of three, one per part of the local objective.
> [`rotation-as-optimization.md`](rotation-as-optimization.md) classifies the
> **quadratic** term — the transition, and the algebra its step size lives in.
> [`trapezoid-as-integration.md`](trapezoid-as-integration.md) classifies the
> **linear** term along *time* — the write, and the quadrature that produces it.
> This one classifies the linear term along *rank* — how many samples the write
> carries, and what is tied between them. All three splits are exact, which is why
> the three notes do not overlap.
>
> Every numbered claim below is checked in float64 by
> [`scripts/mimo_as_batch.py`](../scripts/mimo_as_batch.py) (54 checks, section
> numbers match). The script depends only on `numpy` and on the equations
> reproduced here — not on the crate — so the results stand independently of the
> implementation.

---

## Abstract

Under the reading in which a linear RNN's state is a fast weight being fit online,
Mamba-2's step is one gradient step on an isotropic ridge pulled toward a rank-one
target. Mamba-3's MIMO extension is introduced for a hardware reason — decoding is
memory-bound, and widening the outer product to a matrix product raises arithmetic
intensity at almost no extra memory traffic — and is then usually set aside as
orthogonal to the modelling story.

It is orthogonal, and this note makes the orthogonality precise rather than assumed.
Five results:

1. **MIMO changes the objective, but only its linear term.** The quadratic form
   stays `G = (1−α)I` — isotropic, `M`-free — and the target inflates from rank one
   to **rank `M`** (rank `2M` with the trapezoid). Everything the companion notes
   derive from isotropy therefore survives verbatim.
2. **The batch is `M` free keys and one value.** Only `B`/`C` are widened
   projections; the `M` values are `M` *fixed diagonal images* of a single vector.
   The tying costs exactly `(M−1)P − M² + 1` dimensions of the rank-`M` write
   manifold — and `micro_steps` pays that back exactly.
3. **"MIMO = `R²` SISO SSMs" is isotropy again, not a property of MIMO.** The
   decomposition into standalone SISO models needs the transition to be a function
   of *no* sample. Mamba's is; a sequential rank-`M` delta rule's is not, and
   neither is a jointly-solved one.
4. **At initialisation a MIMO block *is* its SISO block** — key `B̄ = mean_m B^{(m)}`,
   query `C̄ = mean_m C^{(m)}`, skip `D/M`, and a write of rank **one**. Rank is
   learned, not built in.
5. **The rotation is shared across ranks by necessity.** `M` ranks share one state,
   so they share its transition; a per-rank rotation has no state-space preimage at
   all. MIMO ranks are rotation-**synchronous**, micro-steps rotation-**staggered**.

Set beside `micro_steps`, the two turn out to be one dial with different things
tied: `MambaProduct(u = M)` reproduces a whole `MIMO(M)` trajectory exactly, and the
converse fails by the dimension count of result 2.

---

## 1. Scope and results

This note concerns the *state update* only, as its companions do: the gate, the
skip, the normalisations and the read are outside the object being classified. That
exclusion costs more here than in the other two notes, and §10.6 says how much:
`M` buys `M` reads as well as `M` writes, and the reads never enter any objective.

| § | claim |
|---|---|
| 3 | The objective, both display forms. The target inflates to rank `M`; `G` does not move. |
| 4 | The batch: `M` free keys, one value through `M` fixed masks. The tying cost, in closed form. |
| 5 | Why the ranks decompose into standalone SISO models, and why the delta-rule family cannot. |
| 6 | Initialisation: a MIMO block starts on the SISO manifold. |
| 7 | MIMO and the rotation: one state, therefore one transition. |
| 8 | MIMO and the trapezoid: the `Δ̃` collapse is rank-blind. |
| 9 | MIMO and MambaProduct: what each dial unties, and the containment. |

Nothing here proposes a behavioural change to the library. §10 lists what it does
change: documentation, and the reasons behind three existing decisions.

---

## 2. Setup and conventions

One head. The state in the fast-weight orientation `S ∈ ℝ^{P×N}` (rows indexed by
value channel, columns by state channel), as
[`rotation-as-optimization.md`](rotation-as-optimization.md) uses;
[`trapezoid-as-integration.md`](trapezoid-as-integration.md) uses the transposed
`h ∈ ℝ^{N×P}` and nothing below depends on which. Collect the rank channels into
matrices:

$$B_t=\big[b_t^{(1)}\cdots b_t^{(M)}\big]\in\mathbb{R}^{N\times M},\qquad
V_t=\big[v_t^{(1)}\cdots v_t^{(M)}\big]\in\mathbb{R}^{P\times M},\qquad
v_t^{(m)}=\text{mimo\_x}[m]\odot x_t$$

$$\boxed{\;S_t=\alpha_tS_{t-1}+\beta_t\,V_{t-1}B_{t-1}^\top+\gamma_t\,V_tB_t^\top\;}
\qquad Y_t=S_tC_t\in\mathbb{R}^{P\times M}$$

$$\alpha_t=e^{\Delta_tA_t},\qquad \beta_t=(1-\lambda_t)\Delta_t\alpha_t,\qquad \gamma_t=\lambda_t\Delta_t$$

`M = 1` and `λ ≡ 1` is Mamba-2. Three facts about the parameterisation are
load-bearing and are the source's own, adopted here without change:

- `B` and `C` are **widened projections**: `d_model → ngroups·state_rank·mimo_rank`,
  so the `M` keys and the `M` queries are independent linear functions of the token,
  at an additive parameter cost.
- `x`, `z` and the output are **not** widened. One SISO projection is produced and
  then element-wise rescaled per rank by a learnable, **data-independent** vector
  (`mimo_x`, `mimo_z`, `mimo_o`), specifically to keep the parameter count additive
  rather than multiplicative.
- `B` and `C` are RMS-normalised (QK-norm) per rank channel *before* the recurrence.

The rotation is carried in the gauge the implementation runs — transition the plain
scalar `α_t`, cumulative rotation absorbed into `B`/`C` — so every quantity here is
real. Two conventions carried over from the companions: the step-size/curvature
product is pinned but its factors are not (`η_tρ_t = 1−α_t` is a **gauge freedom**,
and every claim that depends on one is checked in three gauges), and "the transition
is the quadratic term, the write is the linear term" is the standard reading of the
correspondence, used without re-deriving.

---

## 3. The objective — a rank-`M` target

**Expanded / `⟨·,·⟩` form.**

$$\mathcal{L}_t(S)=\underbrace{\frac{\rho_t}{2}\lVert S\rVert_F^2}_{\text{quadratic: }G=\rho I}
-\underbrace{\frac{1}{\eta_t}\Big(\gamma_t\big\langle SB_t,\,V_t\big\rangle_F
+\beta_t\big\langle SB_{t-1},\,V_{t-1}\big\rangle_F\Big)}_{\text{linear: a }2M\text{-sample write}},
\qquad \eta_t\rho_t=1-\alpha_t$$

$$=\ \sum_{m=1}^{M}\Big[\frac{\rho_t}{2M}\lVert S\rVert_F^2
-\frac{1}{\eta_t}\Big(\gamma_t\big\langle Sb_t^{(m)},v_t^{(m)}\big\rangle
+\beta_t\big\langle Sb_{t-1}^{(m)},v_{t-1}^{(m)}\big\rangle\Big)\Big]$$

**Proximal form.**

$$\mathcal{L}_t(S)=\frac{\rho_t}{2}\Big\lVert S-\underbrace{\frac{\gamma_t\,V_tB_t^\top+\beta_t\,V_{t-1}B_{t-1}^\top}{1-\alpha_t}}_{T_t,\ \operatorname{rank}\,2M}\Big\rVert_F^2+\text{const}$$

`S_t = S_{t−1} − η_t∇L_t(S_{t−1})` reproduces the recurrence exactly, verified in
three gauges. The second line is the reason for this note's title: **MIMO is a
minibatch.** `L_t` is a sum of `M` per-sample objectives, each structurally Mamba-2's
own and each carrying `ρ_t/M` of the ridge — the standard convention in which a
regulariser is shared across a batch rather than replicated.

Target rank, measured (`P = N = 8`, `M = 3`): `1` for Mamba-2, `min(P,N,M) = 3` for
MIMO without the trapezoid, `min(P,N,2M) = 6` with it. So the two write-side dials
**multiply**: taps `×` ranks.

The scope of the change is the load-bearing part, and it is checked on the object of
the claim rather than asserted: the **state-to-state map** — how a perturbation of
`S` propagates with the input switched off — is exactly `∏_tα_t·I` at `M = 1, 2, 3, 5`
and for any MIMO masks. So `G = (1−α_t)I` whatever `M` does. Two consequences:

- Every argument the companion notes build on isotropy is untouched by MIMO. In
  particular `rotation-as-optimization.md`'s Proposition 4 — isotropic curvature ⟹
  commuting per-micro-step factors ⟹ `u` can only change the write — holds with
  MIMO switched on, which is what makes §9 possible.
- The one-line summary *"MIMO does not change the objective"* is half right, in
  exactly the way the trapezoid's is: the **quadratic term** does not change; the
  objective does, and its target gains rank.

---

## 4. The batch — `M` free keys and one value

The minibatch reading needs one qualification, and it is easy to miss because it
lives in the source's appendix rather than its equations. Only the `M` **keys** are
independent projections. The `M` **values** are

$$v_t^{(m)}=D_m\,x_t,\qquad D_m:=\operatorname{diag}(\text{mimo\_x}[m])\ \text{fixed}$$

so the batch is `M` samples whose targets are `M` *fixed linear images* of one
data-dependent vector. They span `M` dimensions generically — verified — but from
only `P` degrees of freedom, not `MP`. Moving the mask to the other side of the
pairing says what that is:

$$\big\langle Sb_t^{(m)},\,D_mx_t\big\rangle=\big\langle (D_mS)\,b_t^{(m)},\,x_t\big\rangle$$

> **One target, `M` keys, `M` fixed measurement gains.** Not `M` observations: `M`
> masked views of one shared state, each fitted to the same value through its own
> key. MIMO buys rank in **key** space and nothing in value space.

The cost is exactly quantifiable. Measure the dimension of the reachable write
manifold (the Jacobian rank of the parameterisation), against a free rank-`M` write
`Σ_m v^{(m)}b^{(m)\top}` with `M` unconstrained values:

| `P` | `N` | `M` | MIMO(`M`) | `u = M` free values | free rank-`M` | deficit | `(M−1)P − M² + 1` |
|---|---|---|---|---|---|---|---|
| 8 | 8 | 3 | 31 | 39 | 39 | 8 | 8 |
| 10 | 8 | 3 | 33 | 45 | 45 | 12 | 12 |
| 8 | 8 | 2 | 23 | 28 | 28 | 5 | 5 |
| 12 | 10 | 4 | 51 | 72 | 72 | 21 | 21 |
| 6 | 8 | 2 | 21 | 24 | 24 | 3 | 3 |
| 9 | 7 | 5 | 43 | 55 | 55 | 12 | 12 |

**The value tying costs exactly `(M−1)P − M² + 1` dimensions of the rank-`M` write
manifold**, in every shape checked, and `u = M` free values reach the whole manifold.
§9 turns the middle column into a statement about `micro_steps`.

This is a characterisation, not a complaint: the tying is what keeps the parameter
count additive (`DP + PR` per head instead of `DPR`), which is the trade MIMO exists
to make. It is the honest reason to expect MIMO to behave like added rank rather
than like added data.

---

## 5. Why the ranks decompose — isotropy, not a property of MIMO

The source's training story is that a rank-`M` MIMO SSM equals `M²` SISO SSMs:
`M` write-channels summed into a shared state, read out `M` ways, so the SISO kernel
can be invoked as a black box. Verified here, over a full trajectory with the
trapezoid and the rotation on:

$$S_t=\sum_{m=1}^{M}S_t^{(m)},\qquad
y_t^{(i)}=\sum_{j=1}^{M}\mathsf{SSM}\big(\alpha,\Delta,B^{(j)},C^{(i)},x^{(j)}\big)_t$$

with each `S^{(m)}` a **standalone** SISO run from `S_0/M` driven only by its own
`(b^{(m)}, v^{(m)})`. Two statements have to be kept apart here, because conflating
them makes the result look trivial:

- **(S1)** the state splits linearly across the writes, given a shared transition.
  True of *any* affine recurrence, including a product of Householders. Vacuous.
- **(S2)** each part is a **standalone model of the same family**, driven only by
  its own sample. This needs the transition to be a function of **no** sample.

(S2) is the black-box claim, and Mamba satisfies it because its transition is
`α_t = e^{Δ_tA_t}` — no key, no value. Neither delta-rule shape does, verified:

| family | transition | (S2)? |
|---|---|---|
| Mamba-3 MIMO | `α_t`, a scalar | **yes** |
| sequential rank-`M` delta rule | `∏_m(I − β_mk_mk_m^\top)` | no — holds every key |
| jointly-solved (block-reflector) rank-`M` | `I − ηK(K^\top K)^{-1}K^\top` | no — holds every key |

So the "`M²` SISO SSMs" identity is **isotropy again**, the same premise that gives
`rotation-as-optimization.md` its Proposition 4. One premise, two conclusions: a
scalar transition means micro-steps cannot rotate, *and* means rank channels
decompose. The corresponding surface fact is that MIMO's write is
permutation-invariant in the rank index (verified) where a product of rank-one
erases is not (verified) — the delta-rule family must **sequence** its `M` pairs or
solve them jointly, and MIMO simply **sums** them.

---

## 6. Initialisation — a MIMO block starts as its SISO block

The masks are initialised `mimo_x = mimo_o = 1/M`, `mimo_z = 1`. Then every value is
`x_t/M`, so

$$V_tB_t^\top=\frac{1}{M}\,x_t\Big(\sum_m b_t^{(m)}\Big)^{\!\top}=x_t\,\bar B_t^\top,
\qquad \bar B_t=\operatorname{mean}_m b_t^{(m)}$$

and the merge `Σ_m mimo_o[m] ⊙ silu(z ⊙ mimo_z[m]) ⊙ y^{(m)}` collapses to
`silu(z_t) ⊙ (S_t\bar C_t + D x_t/M)`. Verified on state and output over a full
trajectory:

> **At initialisation a MIMO block is exactly its SISO block** with key
> `B̄ = mean_m B^{(m)}`, query `C̄ = mean_m C^{(m)}`, and the `D` skip scaled by
> `1/M`. The rank-`M` write is **rank one**, and reaches rank `M` as soon as
> `mimo_x` differentiates.

Two readings. First, this is the `M` counterpart of `micro_steps = 1` being stock
byte for byte: the dial starts on the manifold it generalises, so a MIMO run that
does not beat its SISO baseline may simply not have left it, and `mimo_x` is the
parameter that leaves it. Second, `1/M` is the minibatch **average** convention —
the write mass is held at the SISO value rather than scaled with the batch, i.e. the
"no linear-scaling-rule step-size bump" choice, with the learnable mask free to
interpolate away from it.

---

## 7. MIMO and the rotation — one state, therefore one transition

The `M` ranks share **one** state `S ∈ ℝ^{P×N}`, and the rotation acts on that
state's `N` axis. So there is exactly one rotation per step no matter how many ranks
write into it — the rotation is a property of the *state's coordinates*, the rank
index a property of the *samples*, and they cannot meet. The implementation says the
same thing operationally: the cumulative angles are projected per (head, plane) and
**broadcast** over the rank axis.

The invariant that makes this checkable is the same-step read/write Gram. For any
transition of the form `α ×` isometry, `R̄^\top R̄ = I` gives

$$\tilde C_t^{(i)\top}\tilde B_t^{(m)}=C_t^{(i)\top}B_t^{(m)}\qquad\text{for all }t,\ \text{all }M^2\text{ entries}$$

Verified. It is a *time-invariant*: the same-step coupling of read `i` to write `m`
never moves, while the cross-token Gram does — that is where the accumulated
rotation lives, and it is what makes it a transition rather than an encoding.

Giving each rank its own cumulative rotation breaks that invariant, and the way it
breaks is diagnostic. The `i = m` diagonal stays invariant; the **off-diagonal —
MIMO's `M²` cross terms — drifts**, because it is the difference of two independent
angle clocks:

| per-rank angle spread | off-diagonal drift at `t = 20` |
|---|---|
| `0` | `0.000` (recovers the shared rotation exactly) |
| `0.05` | `0.178` |
| `0.2` | `1.450` |
| `1.0` | `4.314` |

And the underlying reason is structural, not a matter of degree:

> **Per-rank rotation has no state-space preimage.** `S_t = α_tS_{t−1} + Σ_m V^{(m)}(\bar R^{(m)}b^{(m)})^\top`
> ungauges to a rotational transition only if a single orthogonal `Q_t` equals
> `\bar R_t^{(m)}` for every `m`. Verified: with shared angles the best single
> orthogonal frame fits to `3.3e-16`; with per-rank angles it is off by `3.36` in
> Frobenius norm.

So per-rank angles are not a member of the rotation ladder at all — the transition
would remain the real scalar `α_t`, and every transition-level property (`Real1D`'s
row: no state tracking, the descent bound, `step_infinite`) would go with it. They
would also buy nothing on the write: measured against a QK-normed key, a per-rank
rotation adds **zero** directions to the reachable write map, because QK-norm removes
the key's *magnitude* and never its direction, and the direction was already free.

**MIMO and `RotationKind` are therefore orthogonal by construction**, and the
broadcast in the implementation is the only thing they can be. The trade the source
makes — `M` ranks over one state, so that bytes stay flat — is exactly what forbids
per-rank angles; the alternative is per-rank *states*, which is `M` heads.

---

## 8. MIMO and the trapezoid — the collapse is rank-blind

`γ_t` and `β_t` are per-head **scalars**, shared across ranks. So the two-installment
identity of [`trapezoid-as-integration.md`](trapezoid-as-integration.md) §5 goes
through with the rank-`M` write substituted for the rank-one one, unchanged:

$$S_T=\sum_{s\le T}\tilde\Delta_s\;\Big(\textstyle\prod_{r=s+1}^{T}\alpha_r\Big)\;V_sB_s^\top,
\qquad \tilde\Delta_s=\lambda_s\Delta_s+(1-\lambda_{s+1})\Delta_{s+1}$$

Verified. `Δ̃` carries **no rank index**, which is why `single_ssd`'s composite key
scale is a `[b, n, l, h]` tensor and needs no `m` axis, and why the same-step
γ-correction is unchanged at any `mimo_rank`. `λ ≡ 1` gives `Δ̃ = Δ` with MIMO on,
exactly as without it.

The two dials touch nothing of each other: the trapezoid extends the linear term
along **time** (a second tap, at lag 1, transported), MIMO extends it along **rank**
(`M` samples at one time). Their only interaction is the multiplication of target
ranks recorded in §3.

---

## 9. MIMO and MambaProduct — what each dial unties

`micro_steps` (`u`) runs `u` full Mamba-3 steps per token, folded into the sequence
axis. Both dials widen the write; they differ in what is left tied:

| dial | keys per token | values per token | step sizes | combination |
|---|---|---|---|---|
| trapezoid (2 taps) | 0 new — reuses `b_{t−1}` | 0 new — reuses `x_{t−1}` | splits `Δ` into two installments | summed, transported |
| **MIMO** (`M`) | `M` free | **0 free** — `M` fixed diagonal images of one `x` | **none** — shares `Δ`, `A`, `λ`, and the rotation | summed, **parallel**, order-free |
| **micro-steps** (`u`) | `u·M` free | **`u` free** (the in-projection lays out `x·u`) | `u` free (`Δ`, `A`, `λ`, rotation each `×u`) | composed, **sequential**, order-dependent |

Each row is checked: MIMO's write is permutation-invariant in `m`, the micro-step
fold is not (two permutations); bumping a single `Δ_{i,j}` changes the token map,
where a per-rank scale has nowhere to come from except the value mask.

**The token-level normal form.** With `M_{a:b} := M_{i,b}⋯M_{i,a}`, the unroll of §7
of the trapezoid note gives `S_i = A(x_i)S_{i−1} + B(x_{i−1}, x_i)` with

$$A(x_i)=\Big(\textstyle\prod_{j=1}^{u}\alpha_{i,j}\Big)R_{i,u}\cdots R_{i,1}$$

and `A(x_i)` verified — on the homogeneous token map, which is what it *is* —
independent of `λ`, of the keys, of the values, and of `M`. That is isotropy stated
as a normal form, and it is why **the rank dial cannot enter `A` any more than the
trapezoid can**. `B(x_{i−1}, x_i)` collects `(u+1)·M` outer products, `M` of them
carrying the previous token, and has rank exactly that where the shape allows it
(measured `6` at `u = M = 2`, `P = N = 16`).

**The containment.** With free values available, `u` micro-steps can reproduce what
`M` ranks do, and constructively: take `α_{i,j} = α_i^{1/u}` so the token transition
matches, and `γ_{i,j} = Δ_i/∏_{k>j}α_{i,k}` so the staggered decay cancels and all
`u` writes land with equal weight. Verified over a whole trajectory:

> **`MambaProduct(u = M)` reproduces a whole `MIMO(M)` trajectory exactly.** The
> converse fails by §4's dimension count — the tied family is a proper submanifold,
> short by `(M−1)P − M² + 1`.

So the two are **one dial with different things tied**, which is the content behind
their sharing a cell at `Real1D`: with a real, commuting step algebra all `u` can
change is the write, and the write is what `M` widens. They stop coinciding in two
places. Upward, at the non-abelian kinds, where `u` also multiplies factors in `A`
and `M` still cannot. Downward, in cost: MIMO is parallel and holds state bytes flat
(that is the whole reason it exists), where `u` costs `u×` the recurrence and pays
for its freedom with `(u−1)·(d_inner + bc + 3·nheads + rot)` extra in-projection
columns.

**Under the rotation the two interleave rather than mix.** A token's `u·M` keys form
`u` **rigid groups of `M`**: within a micro-step the `M` keys' Gram is invariant
(§7), and successive groups are staggered by the accumulated product `R_{i,j}⋯R_{i,1}`.
Both verified. MIMO ranks are rotation-synchronous; micro-steps are
rotation-staggered.

---

## 10. Consequences for this crate

### 10.1 What MIMO is, and how to describe it

A minibatch of `M` samples per token, sharing one step size and one transition, with
`M` free keys and `M` values tied to one vector through fixed diagonal masks (§3,
§4). Describing it as "extra write rank" is accurate; describing it as "extra data"
is not. The hardware motive is worth keeping attached rather than treated as an
aside — raising the batch for arithmetic intensity is the same trade batch size
makes in ordinary SGD, and it carries the same expectation of diminishing returns in
`M`.

### 10.2 The initialisation identity is worth knowing

`mimo_rank > 1` begins life *exactly* as its SISO counterpart, at rank one (§6). Any
comparison of a MIMO configuration against SISO is a comparison against that run's
own starting point, and `mimo_x` is the only parameter that leaves it.

### 10.3 The rotation is broadcast over ranks by necessity

`rotate_bc_forward` expands the cumulative-angle tensor over the `m` axis, and
`num_rotation_channels()` has no `mimo_rank` factor in any branch. §7 is why that is
forced rather than convenient: one state, one transition; per-rank angles have no
state-space preimage, break the time-invariance of the `M²` same-step Gram, and add
no reachable write. Worth stating wherever the rotation is documented, because the
opposite reading — "the ranks could rotate at different rates" — is the natural
first guess.

### 10.4 The `Δ̃` collapse is rank-blind, and that is a constraint

`single_ssd`'s composite key scale is a `[b, n, l, h]` tensor precisely because
`γ`/`β` are per-head scalars (§8). Any future change that makes the write's
per-sample weight depend on the rank index has to carry that index into the scale
tensor and the same-step correction, or give up the pathway.

### 10.5 `micro_steps` contains MIMO on the write

§9's containment is the precise version of what `src/mamba3/product/` documents as
"the sequential reading of the cell `mimo_rank` occupies jointly". The difference is
tying and cost, in both directions: `u` unties the values and the step sizes and
pays `u×` the recurrence; `M` ties them and pays nothing in bytes. Documented where
`micro_steps` is documented.

### 10.6 What does not follow

- No numerical behaviour changes. Every claim here is about the recurrence as
  implemented.
- The reading says nothing about the read, and here that omission is larger than in
  the companion notes: `M` widens `C` as well as `B`, and `C`, `mimo_z` and
  `mimo_o` never appear in any objective above. Half of what the rank dial buys is
  outside this reading entirely.
- MIMO is not a route to state tracking and is not analysed as one. It leaves `G`
  isotropic and the transition scalar; the circulating component remains the
  rotation's.

---

## 11. Reproduction

```bash
python3 scripts/mimo_as_batch.py
```

`numpy` only; float64 throughout; 54 checks; exits non-zero on failure. Section
numbers in its output match this document's. The script encodes the recurrence from
§2 directly and never imports the crate, so agreement between it and the
implementation is asserted separately, by the Rust test suites
(`src/mamba3/helpers.rs`, `src/mamba3/single_ssd/`, `src/mamba3/product/tests.rs`).

---

## References

- *Mamba-3*. arXiv:2603.15569. §*Multi-Input, Multi-Output* and its appendix carry
  the recurrence, the `M²`-SISO equivalence, the chunked-algorithm FLOP argument and
  the mask parameterisation used throughout.
- T. Dao, A. Gu. *Transformers are SSMs: Generalized Models and Efficient Algorithms
  Through Structured State Space Duality*. arXiv:2405.21060. The rank-one baseline
  §3 reduces to.
- J. Siems, T. Carstensen, A. Zela, F. Hutter, M. Pontil, R. Grazzi. *DeltaProduct:
  Improving State-Tracking in Linear RNNs via Householder Products*, 2025. The
  sequential rank-`M` write §5 contrasts against, and the dial §9 compares with.
- I. Schlag, K. Irie, J. Schmidhuber. *Linear Transformers Are Secretly Fast Weight
  Programmers*, 2021. The fast-weight orientation of §2.

**Note on novelty.** The recurrence, the `M²`-SISO equivalence and the mask
parameterisation are the source paper's, and reading a rank-`M` write as a minibatch
is the obvious first move once the fast-weight correspondence is in hand. What this
note assembles is: the objective itself, with the ridge split that makes the
minibatch reading exact and the check that `G` is `M`-free (§3); the identification
of the value tying as `M` fixed measurement gains on one target, priced in closed
form at `(M−1)P − M² + 1` (§4); the observation that the `M²`-SISO decomposition is
isotropy rather than a property of MIMO, with the two delta-rule counterexamples
(§5); the initialisation identity (§6); the no-preimage argument for why the
rotation must be shared, and the same-step Gram invariant that detects it (§7); and
the containment of `MIMO(M)` inside `MambaProduct(u = M)`, which turns two dials into
one dial with different things tied (§9).
