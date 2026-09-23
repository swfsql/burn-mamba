# MIMO as Batch Size

### What the rank dial of Mamba-3 does to the local objective, and what it composes with

> Reference note for `burn-mamba`. It is the authority for the decisions of
> this crate around `Mamba3Config::mimo_rank`
> (`helpers::{build_v_with_mimo, mimo_outer_sum}`, the `mimo_x_hmp` /
> `mimo_z_hmp` / `mimo_o_hmp` masks). It is also the authority for how the
> rank dial interacts with `Mamba3Config::micro_steps` and `RotationKind`.
>
> It is the third of three notes, one per part of the local objective:
>
> - [`rotation-as-optimization.md`](rotation-as-optimization.md) classifies the
>   **quadratic** term: the transition, and the algebra of its step size.
> - [`trapezoid-as-integration.md`](trapezoid-as-integration.md) classifies the
>   **linear** term along *time*: the write, and the quadrature that makes it.
> - This note classifies the linear term along *rank*: how many samples the
>   write carries, and what is tied between them.
>
> All three splits are exact. For this reason, the three notes do not overlap.
>
> [`scripts/mamba-3/mimo_as_batch.py`](../../scripts/mamba-3/mimo_as_batch.py)
> checks every numbered claim below in float64 (54 checks, with the same
> section numbers). The script depends only on `numpy` and on the equations in
> this note, not on the crate. So the results do not depend on the
> implementation.

---

## Abstract

In one common reading, the state of a linear RNN is a fast weight that is fit
online. In that reading, the step of Mamba-2 is one gradient step on an
isotropic ridge, pulled toward a rank-one target. Mamba-3 introduces its MIMO
extension for a hardware reason: decoding is memory-bound, and to widen the
outer product to a matrix product raises the arithmetic intensity at almost no
added memory traffic. After that, MIMO is usually put aside as orthogonal to
the modelling story.

It is orthogonal, and this note makes the orthogonality precise, not assumed.
Five results:

1. **MIMO changes the objective, but only its linear term.** The quadratic
   form stays `G = (1−α)I`: isotropic and `M`-free. The target inflates from
   rank one to **rank `M`** (rank `2M` with the trapezoid). So everything that
   the companion notes derive from isotropy stays true without change.
2. **The batch is `M` free keys and one value.** Only `B`/`C` are widened
   projections. The `M` values are `M` *fixed diagonal images* of a single
   vector. The tying costs exactly `(M−1)P − M² + 1` dimensions of the
   rank-`M` write manifold, and `micro_steps` gives that back exactly.
3. **"MIMO = `R²` SISO SSMs" is isotropy again, not a property of MIMO.** The
   decomposition into standalone SISO models needs a transition that is a
   function of *no* sample. The transition of Mamba is such a function. The
   transition of a sequential rank-`M` delta rule is not, and neither is that
   of a jointly-solved one.
4. **At initialisation, a MIMO block *is* its SISO block**, with the key
   `B̄ = mean_m B^{(m)}`, the query `C̄ = mean_m C^{(m)}`, the skip `D/M`, and a
   write of rank **one**. The rank is learned, not built in.
5. **The ranks must share the rotation.** `M` ranks share one state, so they
   share its transition. A per-rank rotation has no state-space preimage at
   all. MIMO ranks are rotation-**synchronous**, and micro-steps are
   rotation-**staggered**.

Beside `micro_steps`, the two are one dial with different things tied.
`MambaProduct(u = M)` reproduces a whole `MIMO(M)` trajectory exactly. The
converse fails by the dimension count of result 2.

---

## 1. Scope and results

Like its companions, this note is about the *state update* only. The gate,
the skip, the normalisations and the read are outside the classified object.
That exclusion costs more here than in the other two notes, and §10.6 says
how much: `M` buys `M` reads as well as `M` writes, and the reads never enter
any objective.

| § | claim |
|---|---|
| 3 | The objective, in both display forms. The target inflates to rank `M`, and `G` does not move. |
| 4 | The batch: `M` free keys, and one value through `M` fixed masks. The cost of the tying, in closed form. |
| 5 | Why the ranks decompose into standalone SISO models, and why the delta-rule family cannot. |
| 6 | Initialisation: a MIMO block starts on the SISO manifold. |
| 7 | MIMO and the rotation: one state, so one transition. |
| 8 | MIMO and the trapezoid: the `Δ̃` collapse is rank-blind. |
| 9 | MIMO and MambaProduct: what each dial unties, and the containment. |

This note proposes no change of behaviour to the library. §10 lists what it
does change: documentation, and the reasons for three existing decisions.

---

## 2. Setup and conventions

The note uses one head. The state is in the fast-weight orientation
`S ∈ ℝ^{P×N}` (the rows are value channels, the columns are state channels),
as [`rotation-as-optimization.md`](rotation-as-optimization.md) uses.
[`trapezoid-as-integration.md`](trapezoid-as-integration.md) uses the
transposed `h ∈ ℝ^{N×P}`, and nothing below depends on the orientation.
Collect the rank channels into matrices:

$$B_t=\big[b_t^{(1)}\cdots b_t^{(M)}\big]\in\mathbb{R}^{N\times M},\qquad
V_t=\big[v_t^{(1)}\cdots v_t^{(M)}\big]\in\mathbb{R}^{P\times M},\qquad
v_t^{(m)}=\text{mimo\_x}[m]\odot x_t$$

$$\boxed{\;S_t=\alpha_tS_{t-1}+\beta_t\,V_{t-1}B_{t-1}^\top+\gamma_t\,V_tB_t^\top\;}
\qquad Y_t=S_tC_t\in\mathbb{R}^{P\times M}$$

$$\alpha_t=e^{\Delta_tA_t},\qquad \beta_t=(1-\lambda_t)\Delta_t\alpha_t,\qquad \gamma_t=\lambda_t\Delta_t$$

`M = 1` and `λ ≡ 1` is Mamba-2. Three facts about the parameterisation are
load-bearing. They are from the source, and this note uses them without
change:

- `B` and `C` are **widened projections**:
  `d_model → ngroups·state_rank·mimo_rank`. So the `M` keys and the `M`
  queries are independent linear functions of the token, at an additive
  parameter cost.
- `x`, `z` and the output are **not** widened. The block makes one SISO
  projection, and then rescales it element-wise per rank by a learnable,
  **data-independent** vector (`mimo_x`, `mimo_z`, `mimo_o`). The purpose is to
  keep the parameter count additive, not multiplicative.
- The block RMS-normalises `B` and `C` (QK-norm) per rank channel, *before*
  the recurrence.

The note carries the rotation in the gauge of the implementation: the
transition is the plain scalar `α_t`, and `B`/`C` absorb the cumulative
rotation. So every quantity here is real. Two conventions come from the
companions:

- The product of step size and curvature is pinned, but its factors are not
  (`η_tρ_t = 1−α_t` is a **gauge freedom**). The script checks every claim
  that depends on one factor in three gauges.
- "The transition is the quadratic term, the write is the linear term" is the
  standard reading of the correspondence. This note uses it and does not
  derive it again.

---

## 3. The objective: a rank-`M` target

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

`S_t = S_{t−1} − η_t∇L_t(S_{t−1})` gives the recurrence exactly, verified in
three gauges. The second line is the reason for the title of this note:
**MIMO is a minibatch.** `L_t` is a sum of `M` per-sample objectives. Each has
the structure of the objective of Mamba-2, and each carries `ρ_t/M` of the
ridge. This is the standard convention, where a batch shares a regulariser
and does not replicate it.

The measured target rank (`P = N = 8`, `M = 3`) is `1` for Mamba-2,
`min(P,N,M) = 3` for MIMO without the trapezoid, and `min(P,N,2M) = 6` with
it. So the two write-side dials **multiply**: taps `×` ranks.

The scope of the change is the load-bearing part. The script checks it on the
object of the claim, and does not only assert it. The **state-to-state map**
(how a perturbation of `S` propagates with the input switched off) is exactly
`∏_tα_t·I` at `M = 1, 2, 3, 5` and for any MIMO masks. So `G = (1−α_t)I`
for any `M`. Two consequences:

- MIMO does not touch any argument that the companion notes build on
  isotropy. In particular, Proposition 4 of `rotation-as-optimization.md`
  (isotropic curvature ⟹ commuting per-micro-step factors ⟹ `u` can change
  only the write) holds with MIMO on. This makes §9 possible.
- The one-line summary *"MIMO does not change the objective"* is half
  correct, in the same way as the summary of the trapezoid: the **quadratic
  term** does not change. The objective changes, and its target gets more
  rank.

---

## 4. The batch: `M` free keys and one value

The minibatch reading needs one qualification. It is easy to miss, because
it is in the appendix of the source, not in its equations. Only the `M`
**keys** are independent projections. The `M` **values** are:

$$v_t^{(m)}=D_m\,x_t,\qquad D_m:=\operatorname{diag}(\text{mimo\_x}[m])\ \text{fixed}$$

So the batch is `M` samples whose targets are `M` *fixed linear images* of
one data-dependent vector. They generically span `M` dimensions (verified),
but from only `P` degrees of freedom, not `MP`. Move the mask to the other
side of the pairing to see what that is:

$$\big\langle Sb_t^{(m)},\,D_mx_t\big\rangle=\big\langle (D_mS)\,b_t^{(m)},\,x_t\big\rangle$$

> **One target, `M` keys, `M` fixed measurement gains.** Not `M`
> observations: `M` masked views of one shared state. Each view is fit to the
> same value through its own key. MIMO buys rank in **key** space and nothing
> in value space.

The cost has an exact value. Measure the dimension of the reachable write
manifold (the Jacobian rank of the parameterisation), against a free rank-`M`
write `Σ_m v^{(m)}b^{(m)\top}` with `M` unconstrained values:

| `P` | `N` | `M` | MIMO(`M`) | `u = M` free values | free rank-`M` | deficit | `(M−1)P − M² + 1` |
|---|---|---|---|---|---|---|---|
| 8 | 8 | 3 | 31 | 39 | 39 | 8 | 8 |
| 10 | 8 | 3 | 33 | 45 | 45 | 12 | 12 |
| 8 | 8 | 2 | 23 | 28 | 28 | 5 | 5 |
| 12 | 10 | 4 | 51 | 72 | 72 | 21 | 21 |
| 6 | 8 | 2 | 21 | 24 | 24 | 3 | 3 |
| 9 | 7 | 5 | 43 | 55 | 55 | 12 | 12 |

**The value tying costs exactly `(M−1)P − M² + 1` dimensions of the rank-`M`
write manifold**, in every shape checked. `u = M` free values reach the whole
manifold. §9 turns the middle column into a statement about `micro_steps`.

This is a characterisation, not a complaint. The tying keeps the parameter
count additive (`DP + PR` per head, not `DPR`), and that is the trade that
MIMO exists to make. It is the honest reason to expect MIMO to behave like
added rank, not like added data.

---

## 5. Why the ranks decompose: isotropy, not a property of MIMO

The training story of the source is that a rank-`M` MIMO SSM equals `M²` SISO
SSMs: `M` write-channels summed into a shared state, read out `M` ways. So
the SISO kernel can be used as a black box. This note verifies it over a full
trajectory, with the trapezoid and the rotation on:

$$S_t=\sum_{m=1}^{M}S_t^{(m)},\qquad
y_t^{(i)}=\sum_{j=1}^{M}\mathsf{SSM}\big(\alpha,\Delta,B^{(j)},C^{(i)},x^{(j)}\big)_t$$

Here each `S^{(m)}` is a **standalone** SISO run from `S_0/M`, driven only by
its own `(b^{(m)}, v^{(m)})`. Keep two statements apart. If you mix them, the
result looks trivial:

- **(S1)** Given a shared transition, the state splits linearly across the
  writes. This is true of *any* affine recurrence, a product of Householders
  too. It is vacuous.
- **(S2)** Each part is a **standalone model of the same family**, driven only
  by its own sample. This needs a transition that is a function of **no**
  sample.

(S2) is the black-box claim. Mamba satisfies it, because its transition is
`α_t = e^{Δ_tA_t}`: no key, no value. Neither delta-rule shape satisfies it
(verified):

| family | transition | (S2)? |
|---|---|---|
| Mamba-3 MIMO | `α_t`, a scalar | **yes** |
| sequential rank-`M` delta rule | `∏_m(I − β_mk_mk_m^\top)` | no: holds every key |
| jointly-solved (block-reflector) rank-`M` | `I − ηK(K^\top K)^{-1}K^\top` | no: holds every key |

So the "`M²` SISO SSMs" identity is **isotropy again**: the same premise that
gives `rotation-as-optimization.md` its Proposition 4. One premise has two
conclusions: with a scalar transition, micro-steps cannot rotate, *and* rank
channels decompose. The matching surface fact: the write of MIMO is
permutation-invariant in the rank index (verified), and a product of rank-one
erases is not (verified). The delta-rule family must **sequence** its `M`
pairs or solve them jointly. MIMO only **sums** them.

---

## 6. Initialisation: a MIMO block starts as its SISO block

The masks are initialised to `mimo_x = mimo_o = 1/M`, `mimo_z = 1`. Then every
value is `x_t/M`, so:

$$V_tB_t^\top=\frac{1}{M}\,x_t\Big(\sum_m b_t^{(m)}\Big)^{\!\top}=x_t\,\bar B_t^\top,
\qquad \bar B_t=\operatorname{mean}_m b_t^{(m)}$$

The merge `Σ_m mimo_o[m] ⊙ silu(z ⊙ mimo_z[m]) ⊙ y^{(m)}` collapses to
`silu(z_t) ⊙ (S_t\bar C_t + D x_t/M)`. Verified on the state and the output,
over a full trajectory:

> **At initialisation, a MIMO block is exactly its SISO block**, with the key
> `B̄ = mean_m B^{(m)}`, the query `C̄ = mean_m C^{(m)}`, and the `D` skip
> scaled by `1/M`. The rank-`M` write is **rank one**. It reaches rank `M`
> when `mimo_x` differentiates.

Two readings:

- This is the `M` counterpart of `micro_steps = 1`, which is stock byte for
  byte. The dial starts on the manifold that it generalises. So a MIMO run
  that does not beat its SISO baseline possibly did not leave that manifold,
  and `mimo_x` is the parameter that leaves it.
- `1/M` is the minibatch **average** convention. The write mass stays at the
  SISO value, and does not scale with the batch. That is the "no step-size
  increase from the linear scaling rule" choice, and the learnable mask is
  free to interpolate away from it.

---

## 7. MIMO and the rotation: one state, so one transition

The `M` ranks share **one** state `S ∈ ℝ^{P×N}`, and the rotation acts on the
`N` axis of that state. So there is exactly one rotation per step, for any
number of ranks that write into it. The rotation is a property of the
*coordinates of the state*, and the rank index is a property of the
*samples*. The two cannot meet. The implementation shows the same thing in
its operations: the cumulative angles are per (head, plane), and they are
**broadcast** over the rank axis. What the block *projects* is per plane at
`Complex2D` (shared by the heads, then `Δ`-scaled per head), and per (head,
block) at the quaternion kinds. Neither carries a rank index.

The same-step read/write Gram makes this checkable. For any transition of the
form `α ×` isometry, `R̄^\top R̄ = I` gives:

$$\tilde C_t^{(i)\top}\tilde B_t^{(m)}=C_t^{(i)\top}B_t^{(m)}\qquad\text{for all }t,\ \text{all }M^2\text{ entries}$$

Verified. It is a *time-invariant*: the same-step coupling of read `i` to
write `m` never moves. The cross-token Gram moves: the accumulated rotation
is there, and this makes it a transition, not an encoding.

Give each rank its own cumulative rotation, and that invariant breaks. The
way it breaks is diagnostic. The `i = m` diagonal stays invariant. The
**off-diagonal (the `M²` cross terms of MIMO) drifts**, because it is the
difference of two independent angle clocks:

| per-rank angle spread | off-diagonal drift at `t = 20` |
|---|---|
| `0` | `0.000` (recovers the shared rotation exactly) |
| `0.05` | `0.178` |
| `0.2` | `1.450` |
| `1.0` | `4.314` |

The underlying reason is structural, not a matter of degree:

> **Per-rank rotation has no state-space preimage.**
> `S_t = α_tS_{t−1} + Σ_m V^{(m)}(\bar R^{(m)}b^{(m)})^\top` ungauges to a
> rotational transition only if a single orthogonal `Q_t` equals
> `\bar R_t^{(m)}` for every `m`. Verified: with shared angles, the best
> single orthogonal frame fits to `3.3e-16`. With per-rank angles, it is off by
> `3.36` in Frobenius norm.

So per-rank angles are not a member of the rotation ladder at all. The
transition stays the real scalar `α_t`, and every transition-level property
goes with it (the row of `Real1D`: no state tracking, the descent bound). They
also buy nothing on the write. Against a QK-normed key, a per-rank rotation
adds **zero** directions to the reachable write map (measured). QK-norm
removes the *magnitude* of the key, never its direction, and the direction
was already free.

**So MIMO and `RotationKind` are orthogonal by construction**, and the
broadcast in the implementation is the only possible form. The trade of the
source (`M` ranks over one state, so that the bytes stay flat) is exactly what
forbids per-rank angles. The alternative is per-rank *states*, which is `M`
heads.

---

## 8. MIMO and the trapezoid: the collapse is rank-blind

`γ_t` and `β_t` are per-head **scalars**, shared across ranks. So the
two-installment identity of
[`trapezoid-as-integration.md`](trapezoid-as-integration.md) §5 holds without
change, with the rank-`M` write in place of the rank-one write:

$$S_T=\sum_{s\le T}\tilde\Delta_s\;\Big(\textstyle\prod_{r=s+1}^{T}\alpha_r\Big)\;V_sB_s^\top,
\qquad \tilde\Delta_s=\lambda_s\Delta_s+(1-\lambda_{s+1})\Delta_{s+1}$$

Verified. `Δ̃` carries **no rank index**. For this reason, the composite key
scale of `single_ssd` is a `[b, n, l, h]` tensor with no `m` axis, and the
same-step γ-correction does not change at any `mimo_rank`. `λ ≡ 1` gives
`Δ̃ = Δ` with MIMO on, exactly as without it.

The two dials do not touch each other. The trapezoid extends the linear term
along **time** (a second tap, at lag 1, transported). MIMO extends it along
**rank** (`M` samples at one time). Their only interaction is the
multiplication of target ranks in §3.

---

## 9. MIMO and MambaProduct: what each dial unties

`micro_steps` (`u`) runs `u` full Mamba-3 steps per token, folded into the
sequence axis. Both dials widen the write. They differ in what stays tied:

| dial | keys per token | values per token | step sizes | combination |
|---|---|---|---|---|
| trapezoid (2 taps) | 0 new: reuses `b_{t−1}` | 0 new: reuses `x_{t−1}` | splits `Δ` into two installments | summed, transported |
| **MIMO** (`M`) | `M` free | **0 free**: `M` fixed diagonal images of one `x` | **none**: shares `Δ`, `A`, `λ`, and the rotation | summed, **parallel**, order-free |
| **micro-steps** (`u`) | `u·M` free | **`u` free** (the in-projection lays out `x·u`) | `u` free (`Δ`, `A`, `λ`, rotation each `×u`) | composed, **sequential**, order-dependent |

The script checks each row:

- The write of MIMO is permutation-invariant in `m`. The micro-step fold is
  not (two permutations).
- A change of a single `Δ_{i,j}` changes the token map. A per-rank scale can
  come only from the value mask.

**The token-level normal form.** With `M_{a:b} := M_{i,b}⋯M_{i,a}`, the unroll
of §7 of the trapezoid note gives `S_i = A(x_i)S_{i−1} + B(x_{i−1}, x_i)`,
with:

$$A(x_i)=\Big(\textstyle\prod_{j=1}^{u}\alpha_{i,j}\Big)R_{i,u}\cdots R_{i,1}$$

The script verifies `A(x_i)` on the homogeneous token map (which is what it
*is*): it does not depend on `λ`, on the keys, on the values, or on `M`. That
is isotropy stated as a normal form. It is why **the rank dial cannot enter
`A`, as the trapezoid cannot**. `B(x_{i−1}, x_i)` collects `(u+1)·M` outer
products, `M` of which carry the previous token. Where the shape permits, it
has exactly that rank (measured: `6` at `u = M = 2`, `P = N = 16`).

**The containment.** With free values, `u` micro-steps can reproduce what `M`
ranks do, by construction:

- Take `α_{i,j} = α_i^{1/u}`, so that the token transition matches.
- Take `γ_{i,j} = Δ_i/∏_{k>j}α_{i,k}`, so that the staggered decay cancels and
  all `u` writes land with equal weight.

Verified over a whole trajectory:

> **`MambaProduct(u = M)` reproduces a whole `MIMO(M)` trajectory exactly.**
> The converse fails by the dimension count of §4. The tied family is a proper
> submanifold, short by `(M−1)P − M² + 1`.

So the two are **one dial with different things tied**. This is the content
behind their shared cell at `Real1D`: with a real, commuting step algebra, `u`
can change only the write, and the write is what `M` widens. They stop
agreeing in two places:

- Upward, at the non-abelian kinds: there `u` also multiplies factors in `A`,
  and `M` still cannot.
- Downward, in cost. MIMO is parallel and keeps the state bytes flat (that is
  the whole reason it exists). `u` costs `u×` the recurrence, and it pays for
  its freedom with `(u−1)·(d_inner + bc + 3·nheads + rot)` added in-projection
  columns.

**Under the rotation, the two interleave and do not mix.** The `u·M` keys of a
token form `u` **rigid groups of `M`**. Within a micro-step, the Gram of the
`M` keys is invariant (§7). The accumulated product `R_{i,j}⋯R_{i,1}` staggers
the successive groups. Both are verified. MIMO ranks are rotation-synchronous,
and micro-steps are rotation-staggered.

---

## 10. Consequences for this crate

### 10.1 What MIMO is, and how to describe it

MIMO is a minibatch of `M` samples per token. The samples share one step size
and one transition. They have `M` free keys, and `M` values tied to one vector
through fixed diagonal masks (§3, §4). "Extra write rank" is an accurate
description. "Extra data" is not. Keep the hardware motive attached, and do
not treat it as an aside. To raise the batch for arithmetic intensity is the
same trade that batch size makes in ordinary SGD. It carries the same
expectation of diminishing returns in `M`.

### 10.2 The initialisation identity is worth knowing

A block with `mimo_rank > 1` starts *exactly* as its SISO counterpart, at rank
one (§6). So any comparison of a MIMO configuration against SISO compares it
against the starting point of that run. `mimo_x` is the only parameter that
leaves it.

### 10.3 The ranks must share the rotation

`rotate_bc_forward` expands the cumulative-angle tensor over the `m` axis, and
`num_rotation_channels()` has no `mimo_rank` factor in any branch. §7 shows
that this is necessary, not only convenient. One state has one transition.
Per-rank angles have no state-space preimage, break the time-invariance of the
`M²` same-step Gram, and add no reachable write. Say this wherever the
rotation is documented, because the opposite reading ("the ranks could rotate
at different rates") is the natural first guess.

### 10.4 The `Δ̃` collapse is rank-blind, and that is a constraint

The composite key scale of `single_ssd` is a `[b, n, l, h]` tensor because
`γ`/`β` are per-head scalars (§8). A future change that makes the per-sample
weight of the write depend on the rank index must carry that index into the
scale tensor and into the same-step correction. Otherwise it must give up the
pathway.

### 10.5 `micro_steps` contains MIMO on the write

The containment of §9 is the precise version of what `src/mamba3/product/`
documents as "the *sequential* reading of the cell that `mimo_rank` fills
*jointly*". The difference is tying and cost, in both directions:

- `u` unties the values and the step sizes, and pays `u×` the recurrence.
- `M` ties them, and pays nothing in bytes.

The crate documents this where it documents `micro_steps`.

### 10.6 What does not follow

- No numerical behaviour changes. Every claim here is about the recurrence as
  implemented.
- The reading says nothing about the read, and here that omission is larger
  than in the companion notes. `M` widens `C` as well as `B`, and `C`,
  `mimo_z` and `mimo_o` never appear in any objective above. Half of what the
  rank dial buys is outside this reading.
- MIMO is not a route to state tracking, and this note does not analyse it as
  one. It keeps `G` isotropic and the transition scalar. The circulating
  component stays with the rotation.

---

## 11. Reproduction

```bash
python3 scripts/mamba-3/mimo_as_batch.py
```

`numpy` only, float64 throughout, 54 checks. It exits non-zero on failure. The
section numbers in its output are the same as in this document. The script
encodes the recurrence of §2 directly, and never imports the crate. So the
Rust test suites assert, separately, that it agrees with the implementation
(`src/mamba3/helpers/tests.rs`, `src/mamba3/single_ssd/`,
`src/mamba3/product/tests.rs`).

---

## References

- *Mamba-3*. arXiv:2603.15569. §*Multi-Input, Multi-Output* and its appendix
  give the recurrence, the `M²`-SISO equivalence, the FLOP argument of the
  chunked algorithm, and the mask parameterisation used throughout.
- T. Dao, A. Gu. *Transformers are SSMs: Generalized Models and Efficient
  Algorithms Through Structured State Space Duality*. arXiv:2405.21060. The
  rank-one baseline of §3.
- J. Siems, T. Carstensen, A. Zela, F. Hutter, M. Pontil, R. Grazzi.
  *DeltaProduct: Improving State-Tracking in Linear RNNs via Householder
  Products*, 2025. The sequential rank-`M` write that §5 contrasts against,
  and the dial that §9 compares with.
- I. Schlag, K. Irie, J. Schmidhuber. *Linear Transformers Are Secretly Fast
  Weight Programmers*, 2021. The fast-weight orientation of §2.

**Note on novelty.** The recurrence, the `M²`-SISO equivalence and the mask
parameterisation are from the source paper. To read a rank-`M` write as a
minibatch is the obvious first move, once the fast-weight correspondence is
known. This note assembles:

- the objective itself, with the ridge split that makes the minibatch reading
  exact, and the check that `G` is `M`-free (§3),
- the identification of the value tying as `M` fixed measurement gains on one
  target, priced in closed form at `(M−1)P − M² + 1` (§4),
- the observation that the `M²`-SISO decomposition is isotropy, not a
  property of MIMO, with the two delta-rule counterexamples (§5),
- the initialisation identity (§6),
- the no-preimage argument for why the ranks must share the rotation, and the
  same-step Gram invariant that detects it (§7),
- the containment of `MIMO(M)` inside `MambaProduct(u = M)`, which turns two
  dials into one dial with different things tied (§9).
