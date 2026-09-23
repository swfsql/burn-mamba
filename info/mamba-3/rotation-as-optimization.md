# Rotation as Optimization

### What a Mamba-3 step optimizes, and what `micro_steps` really composes

> Reference note for `burn-mamba`. It is the authority for the decisions of
> this crate around `Mamba3Config::micro_steps` (`src/mamba3/product/`) and its
> relation to DeltaProduct. It is also the authority for the
> optimization-theoretic reading of the complex transition
> (`src/mamba3/rotation/`).
>
> [`scripts/mamba-3/rotation_as_optimization.py`](../../scripts/mamba-3/rotation_as_optimization.py)
> checks every numbered claim below in float64 (69 checks, with the same
> section numbers). The script depends only on `numpy` and on the equations in
> this note, not on the crate. So the results do not depend on the
> implementation.

---

## Abstract

Linear RNNs have a reading in which the recurrent state is a fast-weight
matrix that is fit online, one gradient step per token. The transition is the
quadratic term of a local objective, and the write is its linear term. Under
that reading, the complex ("rotational") transition of Mamba-3 is usually
declared out of scope: a rotation has complex eigenvalues, and no objective in
any metric makes one.

We show that this declaration comes from the assumption that the step size is
a positive real number. Relax exactly one premise, in any of three different
ways, and the rotation is inside the framework again, exactly and with no
approximation:

1. **The step size is complex** (`η ∈ ℂ`, or `ℍ`), on the unchanged convex
   objective of Mamba-2. Its real part is the descent. Its imaginary part is
   circulation, tangent to the level sets. `|μ| ≤ 1` guarantees
   `Re η > 0` (descent).
2. **The goal is a saddle, not a minimum**: descent in the real part of the
   state and *ascent* in the imaginary part, on a harmonic potential.
   Harmonic functions have no minima, so this is forced, not chosen. The
   rotation is the standard cycling of gradient descent–ascent.
3. **The method is momentum**: heavy ball on the *same* Mamba-2 objective,
   with momentum `β = α²` and step `ηρ = |1−μ|²`, has exactly the eigenvalues
   `μ, μ̄`. The imaginary half of the state is the velocity buffer.

A fourth escape exists, and it belongs to the delta-rule family: **compose
several steps whose Hessians do not commute**. That is DeltaProduct, and it
needs direction-dependent (rank-one) curvature. The curvature of Mamba is
isotropic, so its per-step transitions commute, and *no* product of them can
leave the real axis. This is the precise reason why the mechanism of
DeltaProduct has no instance in Mamba-3. It is also the precise sense in
which the `micro_steps` of this crate is a different construction that
reaches the same place.

The resulting design space is a 2×2 over (**rank of the curvature**) ×
(**algebra of the step size**). Mamba-2, DeltaNet/DeltaProduct and
Mamba-3/MambaProduct fill three cells, and the fourth is empty. Every claim
of the `micro_steps` documentation about what `u > 1` buys per `RotationKind`
follows from one line of that table.

---

## 1. Scope and results

This note is about the *state update* only: the map from the previous state
and the current token to the next state. Everything that a block wraps around
it (the gate, the skip, the output projection, the normalisations) is outside
the classified object. The read is outside too: the query never appears in
any objective below.

The results, in the order of their derivation:

| § | claim |
|---|---|
| 3 | No first-order method with a real step size, on any real-valued loss, can make a rotation. This is the obstruction, stated exactly. |
| 4 | A step size relaxed to `ℂ` gives Mamba-3 exactly, on the objective of Mamba-2. Descent is guaranteed, with `\|arg η\| ≤ arcsin α`. |
| 4.5 | The real constraint of Mamba-2 is a *step-size cap*: `ηρ ∈ (0,1)`, always short of a Newton step. The complex transition lets `ηρ` fill the disc. Parity is on the real axis at `ηρ = 2`. Mod-`k` has the circulation fraction `cos(π/k)`. |
| 5 | The same recurrence is gradient descent–ascent on a harmonic (saddle-only) potential. |
| 6 | The same recurrence is heavy-ball momentum on the unchanged objective, with the velocity folded into the imaginary part of the state. |
| 7 | The delta-rule escape: composition. Verified sharp: rotation occurs iff both micro-steps overshoot. |
| 8 | The design space, and what `micro_steps` is. |
| 8b | The empty cell: what a rotational delta rule needs. A `ℂ`-Hermitian curvature, and the tied ordering (with its `ℍ` side condition). |
| 9 | Consequences for this crate. One of them changes an implementation claim: the exponential integrator is load-bearing for state tracking, not only for accuracy. |

This note proposes no change of behaviour to the library. §9 lists what it
does change: documentation, attribution, and the reasons for three existing
defaults.

---

## 2. Setup and conventions

### 2.1 The recurrence

The note uses one head, and drops the trapezoid and MIMO until §9 (they are
orthogonal). Take Mamba-3 with a complex transition, under the
**exponential-Euler** discretisation: the exponential on the transition, and
the held right endpoint on the state-input (`α = e^{ΔA}`, `γ = Δ`). Mamba-1
and Mamba-2 really implement that scheme, and the discretisation table of the
Mamba-3 paper names it. The same table notes that the Mamba-1 paper reports
ZOH (`γ = A⁻¹(e^{ΔA} − I)`), but that its released implementation does not
use it. §5 depends on this distinction.

$$h_t = e^{\Delta_t(A_t + i\theta_t)}\,h_{t-1} + \Delta_t\,B_t\,x_t^\top,
\qquad y_t = \operatorname{Re}\big(C_t^\top h_t\big)$$

Here `h ∈ ℂ^{n}` per value channel, `n = N/2`, `A_t ∈ ℝ` is a scalar (per
head), `θ_t ∈ ℝ^n` is per state channel, and `B_t = B + iB̂`, `C_t = C + iĈ`
are complex projections. Write:

$$\mu_{t,j} := e^{\Delta_t(A_t + i\theta_{t,j})} = \alpha_t\,e^{i\varphi_{t,j}},
\qquad \alpha_t = e^{\Delta_t A_t}\in(0,1],\qquad \varphi_{t,j} = \Delta_t\theta_{t,j}$$

In real form, `μ` acts as `α·R(φ)`, with `R` the 2×2 rotation. This is the
*complex-to-real SSM equivalence* of the source paper. It is the reason why
the `state_rank` `N` of the crate carries `N/2` rotating pairs.

### 2.2 Fast-weight orientation

For a comparison with the delta-rule literature, transpose to the fast-weight
convention: `S ∈ ℂ^{P×n}`, the rows are value channels and the columns are
complex state channels. Then the `(B, x, C)` of Mamba have the roles of (key,
value, query):

$$\boxed{\;S_t = S_{t-1}\,\mathrm{D}(\mu_t) + \Delta_t\,x_t k_t^\top\;}
\qquad k_t := B_t,\quad x_t\in\mathbb{R}^P,\quad q_t := \overline{C_t}$$

The conjugate on `q` is not cosmetic. With the real inner product on `ℂ`:

$$\langle a, b\rangle := \operatorname{Re}(\bar a\, b) \quad\text{(the Euclidean one on }\mathbb{R}^2)$$

the read is the honest pairing `y_t = ⟨q_t, S_t⟩`, and the inner model is
`f_S(k) = Re(S k̄)`. That is exactly why the *real* read vector of the source
paper is `[C; −Ĉ]`, and its real write vector is `[B; +B̂]`: the read is
conjugated, and the write is not. If you invert this, you get a sign error in
`y` and nowhere else.

The gradient of a real-valued function of a complex variable is the Riesz
representative for `⟨·,·⟩`, that is, `∇ := 2 ∂/∂S̄` (Wirtinger). For
`L(S) = ½ρ‖S‖²` this gives `∇L = ρS`, as it should.

### 2.3 The framework that this note extends

This note extends the "linear RNN as test-time optimizer" correspondence. For
a recurrence `S_t = S_{t-1}M_t + u_tk_t^⊤`, the update field is affine. Its
integral along a ray gives:

$$\mathcal{L}_t(S) = \frac{1}{2\eta}\operatorname{tr}\!\big(SG_tS^\top\big) - \frac{1}{\eta}\big\langle Sk_t, u_t\big\rangle,
\qquad G_t := I - M_t$$

This is exact **iff `G_t` is symmetric**, that is, iff `M_t` is. With a
symmetric positive definite preconditioner `P` on key space, the condition
widens: an objective exists in *some* metric iff `M_t` is diagonalizable with
real eigenvalues. Two standard readings follow, and the note uses them below:

- **The transition is the quadratic term.** All the identity of the model is
  in `G`.
- **The write is the linear term.** Every model has the same write, so it
  distinguishes nothing.

For orientation, the two baselines, in both display forms:

$$\mathcal{L}^{\text{DeltaNet}}_t(S)=\tfrac12\lVert Sk_t-v_t\rVert^2
=\underbrace{\tfrac12\lVert Sk_t\rVert^2}_{\text{quadratic: }G=k k^\top}
-\underbrace{\langle Sk_t,v_t\rangle}_{\text{linear: write}}
+\underbrace{\tfrac12\lVert v_t\rVert^2}_{\text{const}}$$

$$\mathcal{L}^{\text{Mamba-2}}_t(S)=\underbrace{\frac{a_t}{2}\lVert S\rVert_F^2}_{\text{quadratic: }G=aI}
-\underbrace{\langle Sk_t,x_t\rangle}_{\text{linear: write}}
\;=\;\frac{a_t}{2}\Big\lVert S-\frac{x_tk_t^\top}{a_t}\Big\rVert_F^2+\text{const},
\qquad \eta_t=\Delta_t,\ a_t=-A_t$$

DeltaNet regresses toward a **vector** target, in the seminorm from `kk^⊤`. It
does not see `S` off `k`. Mamba-2 regresses toward a **rank-one matrix**
target, in the Frobenius metric, and penalises every direction equally.
*Rank-one, direction-dependent curvature* against *isotropic curvature* is the
distinction that decides everything in §7 and §8.

---

## 3. The obstruction

> **Proposition 1.** Let `L` be any real-valued, twice-differentiable loss,
> and let `η ∈ ℝ`. Then the transition from one step of preconditioned
> gradient descent, `I − η∇²L`, has a real spectrum. So no such step can make
> a rotation.

*Proof.* `∇²L` is symmetric, so it is orthogonally diagonalizable with real
eigenvalues. `I − η∇²L` has the same eigenvectors. ∎

Verified over 2000 random (loss, step) draws, with steps of both signs (§3 of
the script).

Two direct corollaries are worth a statement, because both are tempting
escapes that do not work:

- **A complex loss, with the step size left real, does not help.** The
  Hessian of a real-valued quadratic on `ℂ^n` is Hermitian. So a "complex
  delta rule" factor `I − βkk^*` still has a real spectrum (verified). Read
  this as a statement about the *step*, not about the loss. The same factor,
  with `β` on the complex disc, rotates in 100% of draws. §8(iii) calls this
  cell the interesting extrapolation.
- **A change of metric does not help.** An SPD preconditioner `P` gives a
  symmetric `G = (I−M)P^{-1}` only when `M` is diagonalizable over `ℝ`, and a
  rotation is not. Preconditioned DeltaNet is this corollary, built. It
  approximates the key Gram by a diagonal and preconditions the write key
  with it. So `I − βPkkᵀ` is similar to a symmetric PSD perturbation, and the
  spectrum stays real. It sharpens the curvature on which the delta rule
  descends. It does not leave the real column of the table of §8.

So Proposition 1 is the correct frame: it says precisely which premise must
break. It has four premises, and the note gives one section to each.

| premise | broken by | § |
|---|---|---|
| the step size is a real scalar | a step in `ℂ` or `ℍ` | 4 |
| the objective is minimized | descent–ascent on a saddle | 5 |
| the update depends only on the current state | momentum: a larger state space | 6 |
| one step per token | a composition of `u` steps | 7 |

The third is the loosest fit, and §6 says so. Heavy ball does not contradict
Proposition 1 at all: it makes its rotation by *enlarging the state space*,
not by a rotation on the given one. It gets its section because it lands on
the same transition, not because it escapes the same statement. This note
also does not claim that the list is exhaustive. These are four constructions
that work, over premises that permit other breaks.

---

## 4. View I: the step size is complex

### 4.1 The construction

Keep the state complex and the update `ℂ`-linear. A real-valued objective
whose gradient step is `ℂ`-linear must have a *Hermitian* quadratic part,
that is, a real curvature `ρ_j > 0` per channel. Take:

$$\boxed{\;
\mathcal{L}_t(S)=\underbrace{\frac12\sum_j\rho_{t,j}\lVert S_{:,j}\rVert^2}_{\text{quadratic: transition}}
-\underbrace{\big\langle \operatorname{Re}(S\bar w_t),\ x_t\big\rangle}_{\text{linear: write}}
\;=\;\frac12\sum_j\rho_{t,j}\Big\lVert S_{:,j}-\frac{x_tw_{t,j}}{\rho_{t,j}}\Big\rVert^2+\text{const}\;}$$

Its structure is *identical* to that of Mamba-2: convex, separable, isotropic
per channel, a proximal pull toward a rank-one target. Now let the step size
be a complex number per channel:

$$S_t = S_{t-1} - \nabla\mathcal{L}_t(S_{t-1})\,\mathrm{D}(\eta_t),\qquad \eta_{t,j}\in\mathbb{C}$$

A match against the recurrence gives two conditions and nothing else:

$$\rho_{t,j}\,\eta_{t,j} = 1-\mu_{t,j},\qquad \eta_{t,j}\,w_{t,j} = \Delta_t\,k_{t,j}$$

Only the products are determined. The split into `(ρ, η, w)` is a gauge
freedom, verified exact in three different gauges. Two gauges are natural:

**Gauge (a): the curvature of Mamba-2.** `ρ_{t,j} = Re(1−μ_{t,j})/Δ_t`. Then:

$$\eta_{t,j} = \Delta_t\big(1 + i\tan\psi_{t,j}\big),\qquad \psi_{t,j} := \arg(1-\mu_{t,j}),
\qquad \operatorname{Re}\eta_{t,j} = \Delta_t\ \text{exactly}$$

**Gauge (b): an unchanged step length.** `ρ_{t,j} = |1−μ_{t,j}|/Δ_t`. This
gives `η = Δ_t e^{iψ}` and `w = e^{−iψ}k`: the key of the objective is the
physical key rotated by `−ψ`, with the same modulus.

In continuous time, gauge (a) is `Δ`-free, exactly as the gauge of Mamba-2 is:

$$\rho_j = a,\qquad \eta_j = 1 - i\frac{\theta_j}{a},\qquad w_j = \frac{a}{a-i\theta_j}k_j,
\qquad S^\star_{:,j} = \frac{x\,k_j}{a-i\theta_j}$$

and `S*` is the true steady state of the ODE, as it must be.

### 4.2 The Helmholtz split is the polar form of `η`

The usual structural statement about a rotational transition splits the
generator into a symmetric part (dissipative, "the objective") and a skew part
(circulating, "invisible"). In the complex frame, that split is literally the
real/imaginary split of a single complex number (verified in the 2×2 real
form):

$$G = I - \alpha R^\top = \underbrace{\operatorname{Re}(1-\mu)}_{1-\alpha\cos\varphi}\,I
\;-\;\underbrace{\operatorname{Im}(1-\mu)}_{-\alpha\sin\varphi}\,J,
\qquad J=\begin{bmatrix}0&-1\\1&0\end{bmatrix}$$

So the recovered objective already had the correct curvature: `ρ = Re(1−μ)/Δ`
in gauge (a) *is* the symmetric part that the classification recovers. **The
skew part is not a missing piece of the objective. It is the imaginary part of
the learning rate.** The same recurrence, written as a flow:

$$\dot S = -(\operatorname{Re}\eta)\,\nabla\mathcal{L} \;-\; (\operatorname{Im}\eta)\,i\nabla\mathcal{L},
\qquad \langle i\nabla\mathcal{L},\,\nabla\mathcal{L}\rangle = 0$$

This is a gradient part plus a part tangent to the level sets of *the same*
`L`. It is the dissipative-Hamiltonian (port-Hamiltonian) form
`ẋ = (J − R)∇H`, where `L` is both the Lyapunov function and the Hamiltonian.

### 4.3 Descent is free

The symmetric part of the multiplication by `η` is **exactly** `Re(η)·I`
(verified). So:

$$\big\langle \nabla\mathcal{L},\ \text{step}\big\rangle = -\sum_j \operatorname{Re}(\eta_{t,j})\,\lVert\nabla\mathcal{L}_{:,j}\rVert^2$$

and the step is a descent direction iff `Re η > 0`, that is, `|ψ| < π/2`. This
is the operational content of "the preconditioner must be positive definite".
For Mamba-3 it costs nothing:

> **Proposition 2.** For `|μ| = α ≤ 1`, `Re(1−μ) = 1 − α cos φ ≥ 0` and
> `|arg(1−μ)| ≤ arcsin α`, with equality at `cos φ = α`.

*Proof.* `1 − μ` is on the circle of radius `α` about `1`. For `α < 1` the
origin is outside that circle, and the tangent lines from the origin subtend
`arcsin α`. ∎

Verified over the whole closed disc. So **every admissible Mamba-3 parameter
gives a real descent step**, with an exact accounting. This holds for the
abelian kind here, and for all four kinds by Proposition 2′ in §8, which needs
only a transition of the form `α ×` an isometry:

$$\text{descent fraction} = \cos\psi = \frac{1-\alpha\cos\varphi}{|1-\alpha e^{i\varphi}|},
\qquad \text{circulation fraction} = \sin\psi$$

### 4.4 What the rotation buys, in optimizer coordinates

`ηρ = 1 − μ`, so the reachable set of the product of step and curvature is the
unit disc about `1`. A comparison of the families in that one coordinate is
the most compressed statement in this note:

| model | `ηρ` | reading |
|---|---|---|
| Mamba-2 | `1 − α ∈ (0,1)` | **always undershoots**: cannot even reach the Newton step `ηρ = 1` |
| DeltaNet | `β ∈ (0,2)` | the full classical range, with overshoot |
| Mamba-3, `φ = 0` | `1 − α ∈ (0,1)` | Mamba-2 |
| Mamba-3, `φ = π` | `1 + α → 2` | overshoot at the classical stability edge `2/ρ`, `μ → −1` |
| Mamba-3, `φ ∉ {0,π}` | off the real axis | real circulation, `ψ ≠ 0` |

> **The restriction of Mamba-2 is not "no rotations". It is a step-size
> cap.** `α = e^{ΔA} ∈ (0,1)` forces `ηρ ∈ (0,1)`: strictly less than one
> Newton step, and far from the classical stability limit at `2`. The complex
> transition lets `ηρ` leave that interval.

Two consequences sharpen the usual story about state tracking:

- **Parity is not in the non-integrable regime.** Parity wants `μ = −1`, that
  is, `ψ = 0`: *pure descent*, at exactly `η = 2/ρ`, whose textbook behaviour
  is a period-2 oscillation. (To reach `φ ≈ π` needs `|A| ≪ θ`, which drives
  `ψ → 0`. So the coupling of `α` and `φ` through `Δ` makes this stronger, not
  weaker.) What Mamba-2 does not have here is *overshoot*, not "an objective
  for rotation".
- **Mod-`k` is in that regime, by a measurable amount.** At `α → 1` and
  `φ = 2π/k`:

  $$|\psi| = \frac{\pi}{2}-\frac{\pi}{k},\qquad \text{circulation fraction} = \cos\frac{\pi}{k}$$

  This gives `0` at `k=2`, `0.5` at `k=3`, `0.707` at `k=4`, and `→1` as
  `k→∞`. The fraction of the update that no objective can explain is a closed
  form in the order of the tracked group.

---

## 5. View II: the goal is a saddle

View I relaxes the *step*. The alternative is to keep a real step and relax
the *goal*. Treat the two real coordinates of a complex channel as two
players:

$$F_t(S) = \operatorname{Re}\Big(-\tfrac12\operatorname{tr}\big(S\,\mathrm{D}(\tilde A_t)\,S^\top\big)
- x_t^\top S k_t\Big),\qquad \tilde A_t = A_t + i\theta_t$$

and run **descent in `Re S`, ascent in `Im S`**. That field is exactly the
complex flow of Mamba-3, and its exponential over one interval is exactly the
transition. Both are verified. The script verifies the second with a
real-form generator built from the field and then exponentiated, so it shares
no subexpression with the transition that it is compared against.

The sign is the part that needs care, because a reader who does this by hand
gets it wrong. For `F = Re(φ)` with `φ` holomorphic, Cauchy–Riemann gives
`∂F/∂z_r = Re(φ′)` and `∂F/∂z_i = −Im(φ′)`. So the descent–ascent field
`(−∂F/∂z_r, +∂F/∂z_i)` is `−φ′(z)` as a complex number. **The ascent in the
imaginary part exactly cancels the conjugation that a descent in the real
part alone would introduce.** Descent in both gives `−conj(φ′)`, which is not
a flow of this system.

One precision: only the *homogeneous* part is the exact flow map. An exact
integral of the forcing is ZOH, `γ = A⁻¹(e^{ΔA} − I)`, where Mamba holds the
right endpoint at `γ = Δ` (§2.1). So this is the approximation that
Mamba-1/-2 already make, and that the min–max reading inherits. The min–max
reading does not introduce it. The two agree to `O(Δ)`.

This is more than a reparameterisation, for two reasons:

- `F` is the real part of a holomorphic function, so it is **harmonic**
  (verified numerically: the Laplacian is zero). By the maximum principle, a
  harmonic function has **no minima at all**: every critical point is a
  saddle. So the min–max reading is not a choice of style. A minimization
  reading of this potential does not exist.
- The rotation is then an entirely standard phenomenon: **gradient
  descent–ascent cycles on saddles**. In this dictionary, the decay `a` is the
  strong convexity/concavity of the two players, and `θ` is their bilinear
  coupling. Their ratio is precisely the angle of §4, `tan ψ = θ/a`. GDA
  converges when convexity dominates, and orbits when coupling dominates.
  `ψ → π/2` is the pure state-tracking limit.

Note the direction of the literature: extragradient and optimistic-GDA
methods exist specifically to *suppress* this cycling. Mamba-3 wants it, and
the exponential integrator (§9.5) keeps it exact.

---

## 6. View III: the method is momentum

The third relaxation keeps both the objective and the real step size, and
changes the *order* of the method. Heavy ball on the unchanged objective of
Mamba-2:

$$z_{t+1} = z_t - \eta\nabla\mathcal{L}(z_t) + \beta\,(z_t - z_{t-1})$$

has the companion matrix `[[1+β−ηρ, −β], [1, 0]]` on `(z_t, z_{t−1})`.

> **Proposition 3.** For any `α ∈ (0,1)` and `φ ∉ {0, π}`, set `β = α²` and
> `ηρ = |1−μ|² = 1 + α² − 2α cos φ`. Then the eigenvalues of the companion
> matrix are exactly `μ` and `μ̄`. So the matrix is similar to `α·R(φ)`.

Verified over 500 random `(α, φ, ρ)` draws, exact to `1e-9`, with the closed
form `ηρ = |1−μ|²` to `1e-15`.

So the complex transition **is** momentum, with the velocity buffer folded
into the imaginary part of the state, not held as a second tensor. The usual
price of momentum is a second state slot. Mamba-3 pays it: it declares half
of the state imaginary.

Two precisions, so that this is not overclaimed:

- The *transition* is exactly that of heavy ball, up to a change of basis.
  The *write* of Mamba-3 is more general. `B` and `B̂` are independent
  projections, so the input drives both the iterate and the velocity. Heavy
  ball drives only the iterate.
- The correspondence has no `φ ∈ {0, π}` member, and for this reason
  Proposition 3 excludes those values. The map does not degrade there.
  Instead, a companion matrix is **non-derogatory** (its minimal polynomial
  is its characteristic polynomial). So it is never similar to a scalar pair
  `diag(α, α)`, and that is exactly a *non-rotating* Mamba-3 pair. At `φ = 0`,
  the parameters above give a defective Jordan block, `rank(C − αI) = 1`,
  with transient growth to `2.84` before the decay. The Mamba-3 pair is
  diagonal and monotone. So momentum is a reading of the *rotating* channels
  only. It does not extend along `rope_fraction` to the unrotated channels,
  which never enter the correspondence.

Views I, II and III are three readings of one recurrence. I and II are the
same mechanism (`Im η` is the ascent direction). III is really a different
method that lands on the same transition.

---

## 7. View IV: a composition of non-commuting steps

The fourth escape from Proposition 1 does not touch the loss, the step or the
method. It uses this fact: **a product of symmetric matrices is not
necessarily symmetric**. So `u ≥ 2` ordinary steps can compose into a
rotation, although no single step can. This is the mechanism of DeltaProduct.

DeltaProduct takes `u` delta-rule micro-steps per token on `u` different
`(k_j, v_j)` pairs. This gives the transition:

$$M_t = \alpha_t\prod_{j=1}^{u}\big(I - \beta_{t,j}k_{t,j}k_{t,j}^\top\big)$$

Two verified facts pin down when this works:

- **Rank-one curvature rotates, and sharply.** At `u = 2`, `d_k = 6`,
  `β ∼ U(0,2)`, ~16–18% of random draws have a complex spectrum. In **every**
  rotating draw, both `β_j > 1` (4000 draws, zero counterexamples). A factor
  `I − βkk^⊤` has the eigenvalue `1 − β` along `k`. So `β > 1` means that
  the micro-step *overshoots its own minimizer*. **Two overshoots make the
  rotation.** This is the same coordinate that §4.4 uses, reached by a
  different construction.
- **Isotropic curvature never rotates.** With `∇²L = ρI`, every factor is
  `(1 − η_jρ_j)I`, a scalar. Scalars commute, so the product is a scalar:
  `0 / 2000` draws at `u = 3` get a complex spectrum.

The second fact is the load-bearing one for this crate. It deserves a
statement as the proposition that it is:

> **Proposition 4.** If the local curvature is isotropic (`∇²L_j = ρ_jI`),
> then the per-micro-step transitions commute, and every product of them is a
> real scalar multiple of the identity. No number of micro-steps can make a
> rotation, and `u` can change only the linear term, that is, the write.

That is the exact reason why the mechanism of DeltaProduct has **no
instance** in Mamba-3. The usual informal version, "Mamba has no erase", names
a symptom. Isotropy is the cause. The proposition also derives (it does not
only assert) that `u` micro-writes under a shared scalar transition collapse
into a single rank-`u` write, which is MIMO.

---

## 8. The design space

Write both families in one form. Both are `u` first-order steps per token:

$$\boxed{\;M_t=\prod_{j=1}^{u}\Big(I-\eta_{t,j}\,\nabla^2\mathcal{L}_{t,j}\Big)\;}$$

They differ in two independent dials: the **rank of the curvature** and the
**algebra of the step size**.

|  | curvature `ρI` (isotropic) | curvature `kkᵀ` (rank-one) |
|---|---|---|
| **`η ∈ ℝ`** | Mamba-2, and Mamba-3 at `Real1D`: `u` collapses to a rank-`u` write (MIMO) | DeltaNet, **DeltaProduct**, Preconditioned DeltaNet |
| **`η ∈ ℂ`, `ℍ`** | **Mamba-3, MambaProduct** | *empty*: a "rotational delta rule" |

Three things follow directly. They are exactly the claims of the
`micro_steps` documentation, which cites this note for their derivation.

**(i) Why `micro_steps` does not exist on Mamba-2.** Proposition 4: isotropic
curvature, a real step, commuting factors. `u` buys only the write, and the
write is the job of MIMO.

**(ii) Why the `RotationKind` split falls exactly where it does.** With
isotropic curvature, the transition factors are elements of the step algebra.
So what `u` buys is a property of that algebra alone:

| kind | step algebra | factors | what `u > 1` buys |
|---|---|---|---|
| `Real1D` | `ℝ` | commute, real | the write only: `u` staggered rank-1 writes, that is, the job of `mimo_rank`, read sequentially |
| `Complex2D` | `ℂ` | commute, phases **add** | `u`× the per-token angle reach, with a live gradient at every factor |
| `Quaternion4D`, `Rotor4D` | `ℍ`, two-sided | **do not commute** | a per-token transition that no single bounded step can express |

Verified: over `ℂ`, the product depends only on the sum of the angles (it is
order-free). Over `ℍ`, the same generators in the reverse order give a
different product, and the product also leaves `exp(Σ generators)`.

**(iii) The empty cell is real, and it is the interesting extrapolation.**
It is rank-one curvature with a complex step: a delta rule whose transition
carries a data-dependent rotation. The rank-one erase conjugates through a
rotational gauge, `P*(I − βkk^H)P = I − β(P*k)(P*k)^H`. This needs only a
linear isometry `P`, so it is true without change for the non-abelian kinds.
It has one **side** condition, which is invisible over `ℂ` but not over `ℍ`:

- The gauge acts on the left (`v ↦ qv`, or `v ↦ qvp̄`). So the step must
  multiply on the *opposite* side of the key, `v − kβ⟨k,v⟩`. Then `β` never
  has to commute with `q`, the right factor of the two-sided kind cancels
  against its own inverse, and the identity is exact for both quaternion
  kinds.
- If `β` is on the same side as the gauge, it comes back conjugated, `q̄βq`:
  the same key, a different step.

All three statements are verified.

That is **not** sufficient to conclude "no new chunkwise algorithm". That
conclusion is true of only one of the two orderings. The recurrence is
affine. The gauge pins the *write* key to `P_t*k_t` for any order of the
step, but the *erase* key follows the rotation:

| order within a step | erase key | write key | |
|---|---|---|---|
| rotate, then erase | `P_t*k_t` | `P_t*k_t` | **tied**: the existing WY/tied-key kernel applies without change |
| erase, then rotate | `P_{t−1}*k_t` | `P_t*k_t` | **untied**: the generalized-DPLR shape, which that kernel does not compute |

Verified in both the left- and the right-multiplication orientations. The
rows above name the *temporal* order. A product does not survive
transposition without a reversal. So "rotate first" is the leftmost matrix
factor in one convention and the rightmost in the other. If you read the
equation of one convention in the order of the other, the tied model becomes
the untied one. This is a real failure mode, not a hypothetical one.

In both cases, the erase factor stays a proper tied Householder. So the
`‖·‖₂ ≤ 1` bound and the `u > 1` stability argument stay true in both orders.
The wrong order costs the kernel, not the guarantee. Mamba-3 has no erase, so
nothing here constrains this crate. It constrains anyone who builds the cell.

The cell has a second **precondition** that is easy to miss. A complex step
rotates a rank-one curvature only if that curvature is `ℂ`-Hermitian. The
*measurement* decides that, not the type of the target: a loss that
penalises only `Re(k^HS)` gives the real curvature, whatever it compares
against. A penalty on both components of the residual gives `kk^H`, and that
forces a complex regression *target* too. With a real target, the curvature
is `KKᵀ` over `ℝ^{2n}`, and since `KᵀJK = 0`:

$$M = I - (aI + bJ)KK^\top \;=\; \begin{bmatrix}1-a & 0\\ -b & 1\end{bmatrix}
\quad\text{on } \operatorname{span}\{K, JK\}$$

This is triangular, with the eigenvalues `1−a` and `1`, both **real**. It is
non-normal: a shear, without the norm bound. Verified: `0%` of draws rotate
with a real target (`max‖M‖₂ = 1.41`), and `100%` with a complex one
(`‖M‖₂ = 1`). So to tie the imaginary half of the target to its real half, or
to set it to zero, is not a cheaper variant of this cell. It deletes the
mechanism. The value `x` of Mamba-3 is real, and this has no effect on it,
because the isotropic `ρI` is `ℂ`-Hermitian free. The precondition is
specific to the rank-one column.

A block in that cell also carries **two** step sizes, so two phases, at two
different rates. This answers the question whether "the" rotation belongs
per token or per micro-step:

| term | curvature | step | phase | rate |
|---|---|---|---|---|
| ridge `(ρ/2)‖S‖²` | isotropic | `η` | `arg η`, as in Mamba-3 | once per **token** |
| data fit `½‖kᴴS − v‖²` | rank-one over `ℂ` | `β` | `arg(1−β)` | once per **micro-step** |

The ridge step is per token, because the `u` corrective steps share one
interval (§7). `β` is per micro-step, because there are `u` of them. The two
are independent. The phase of the decay acts on every plane, and the phase of
the erase acts only in the plane that its own key spans. So for `n ≥ 2`, no
single decay phase with real erase gates can reproduce a token with two erase
phases. Each phase already uses an existing gate at the correct rate, so
neither needs a configuration knob.

The ladder `ℝ ⊂ ℂ ⊂ ℍ` is uniform. This is worth a record, because it is why
the rotation code of the crate has no branches:

| algebra | `η = (1−μ)/ρ` | symmetric part of `v ↦ ηv` | descent condition |
|---|---|---|---|
| `ℝ` | positive scalar | `η·I` | `η > 0` |
| `ℂ` | complex scalar | `Re(η)·I` | `Re η > 0` |
| `ℍ` | quaternion | `Re(η)·I` | `Re η > 0` |
| two-sided `(q, p)` | pair | not a multiple of `I`, trace `= 4·Re(q)Re(p)` | `α < 1` (below) |

All verified. The bound of Proposition 2, `|arg| ≤ arcsin α`, holds without
change over `ℍ`.

The two-sided row does not fit the pattern. `v ↦ qvp̄` is not a
multiplication by a scalar in any algebra, and its symmetric part is not a
multiple of `I`. So `Re(η) > 0` has nothing to attach to. Its normalised trace
is `Re(q)Re(p)`, which is negative about half the time. It is tempting to read
that as a descent failure that needs a sign constraint. **It is not**: the
trace is the *average* of `⟨Tv, v⟩` over directions, not the condition. The
condition is a statement about eigenvalues, and it holds uniformly:

> **Proposition 2′ (descent, all four kinds).** Let `M = αT`, with `T` any
> isometry and `α ∈ (0,1)`. Then `sym(I − M) = I − α·T_sym ≽ (1−α)I ≻ 0`.
>
> *Proof.* `‖T‖₂ = 1`, so `|⟨Tv, v⟩| ≤ ‖v‖²`, and every eigenvalue of `T_sym`
> is in `[−1, 1]`. So every eigenvalue of `I − αT_sym` is at least `1 − α`. ∎

Verified, also on the draws where `Re(q)Re(p) < 0` (`λ_min > 0` in all
20 000). So **descent is free for every `RotationKind`, with a uniform margin
`1 − α`**. The conformality property (the first bullet below) buys it. The
per-algebra `Re(η) > 0` rows are the sharper single-sided special case
(`1 − α cos φ ≥ 1 − α`), not a separate condition.

Two properties of *this* enlargement are load-bearing. They are why the
escape is "the step size joins a normed division algebra". The strictly
larger "the preconditioner does not have to be symmetric" is vacuous alone,
because any `M` is `I − GP` for a symmetric `G` and a general `P`:

- **Every kind is `α × isometry`, so it is normal, so `‖M‖₂ = ρ(M) = α`
  exactly.** This includes `Rotor4D`, whose `v ↦ qvp̄` is in `SO(4)`. For a
  general non-symmetric preconditioner, the gap between the norm and the
  spectral radius is not only present but *unbounded*. Conjugate a rotation
  by an ill-conditioned `D`: the spectrum stays on the unit circle, and
  `‖M‖₂` grows with `cond(D)` (verified: `‖M‖₂/ρ(M) = 35` at
  `cond(D) = 50`). That loses the bound that a *time-varying* product needs.
  The per-factor spectral radius does not bound `M_t⋯M_1`, but a
  submultiplicative `‖M_t‖₂ ≤ 1` does. Conformality makes the state-tracking
  norm argument exact, not asymptotic.
- **Commutativity is not what makes the RoPE trick work.** The trick needs
  only a transition of the form *scalar × group element*. `α` is a scalar, so
  it commutes with everything. The cumulative rotation telescopes as
  `R_{i+1..t} = R̄_t R̄_i^{-1}`, by associativity and invertibility.
  Commutativity collapses that telescoping scan into a closed-form **cumsum of
  angles**. Without commutativity, the non-abelian kinds need a scan with a
  cross-chunk accumulator. The carry exists *because* the factors do not
  commute. This is also why the trick works without change for
  `Quaternion4D` and `Rotor4D`.

---

## 9. Consequences for this crate

### 9.1 What `micro_steps` is, and how to describe it

`Mamba3Config::micro_steps` (`u`) runs `u` full Mamba-3 recurrence steps per
token. Each step takes its own `x`, `Δ`, `A`, `λ`, `B` and rotation from its
own slice of the input projection. The block evaluates them with the
micro-steps folded into the sequence axis. By §7–§8, this is *not* the
mechanism of DeltaProduct carried over (that mechanism has no instance here).
It is the same construction with the other dial turned:

> DeltaProduct builds a non-commuting product from the **curvature** (a
> different rank-one Hessian per micro-step). The curvature of Mamba-3 is
> isotropic, so `micro_steps` builds one from the **step size**. Same
> equation, orthogonal dial.

The name `micro_steps` is accurate. "Product" is also accurate (at the
quaternion kinds, the per-token transition really is a product of `u`
non-commuting factors), on the condition that it is not read as "product of
Householders". Cite DeltaProduct as the source of the *dial* (`u` first-order
steps per token, for the expressiveness of the transition, not for memory),
not of the *mechanism*.

### 9.2 The reach claim needs no external attribution

At `Complex2D`, the `u` factors commute and their phases add. So `u`
micro-rotations, each well inside the per-step bound, compose to `u`× the
reach, with a live gradient at every factor. This is the real payoff at the
abelian kind, and it needs no external attribution. It follows from §8(ii),
plus one fact: a single rotation *at* the `rotation_range` bound is on the
asymptote of `tanh`, where the f32 gradient is exactly zero.

State the mechanism honestly. The token gets `u` full-size steps: its
effective interval becomes longer, not subdivided. The consistent alternative
(`Δ_j = Δ/u`: the same per-token transition, which buys only the staggered
writes and the non-abelian order) is inside the reachable set of the model,
because `dt_limit` has no lower floor by default. So this is a superset, not
an error. (`dt_limit` is a configurable clamp. A run that raises the floor
above `Δ/u` loses that alternative.)

### 9.3 `micro_steps` against `mimo_rank`

Both widen the write. §7 says why they are not redundant, and why the
difference disappears at `Real1D`. With isotropic curvature, the transition
factors do not depend on the key at all. So at `Real1D`, `u` can change only
the linear term. A sequential epoch of `u` samples with decay-staggered
weights then fills the cell that `mimo_rank` fills jointly, as a minibatch of
`M`. `u` becomes structurally different exactly when the step algebra is
non-trivial.

### 9.4 The trapezoid composes with all of it

The trapezoidal discretisation is a *two-sample* linear term under the same
complex step. The rotation of this step **parallel-transports the older
sample into the current frame**, before the weight:

$$\text{linear term} \ \propto\ \gamma_t\,x_tk_t^\top \;+\; \beta_t\,x_{t-1}\big(e^{i\varphi_t}\odot k_{t-1}\big)^\top$$

Verified exact over a full trajectory. The usual reading of the trapezoid as
a two-tap FIR filter on the gradient stream stays true under the complex
transition. It is a *transported* filter, not a naive one. This is also why
the caches of the crate store the tapped `(B, x)` **without the mass of the
tap**, and apply `β` at time `t`. They do not store a pre-weighted
contribution.

### 9.5 The exponential integrator is load-bearing (implementation claim)

State tracking needs a transition that preserves the norm exactly in the
undamped limit. Forward Euler on a pure rotation has the modulus
`|1 + iΔθ|`, which is already `1.28` at `Δθ = 0.8`: it spirals outward.
Exponential-Euler gives the modulus exactly `1.0`.

So the choice of the exponential discretisation (and not forward Euler) is
not only about second-order accuracy, which is how it is usually presented.
**It makes the transition orthogonal, and so it makes parity exact, not
drifting.** Say this wherever the discretisation is documented. It does not
depend on anything else in this note.

### 9.6 What does not follow

- No numerical behaviour changes. Nothing here is a bug report. Every
  proposition is about the recurrence as implemented.
- The three views do not make the rotation "just optimization". They locate
  it precisely: it is the imaginary part of a step size, or equivalently an
  ascent direction, or equivalently a velocity. Under all three, the query
  still never appears in any objective, and the reading still says nothing
  about what the block reads back out.

---

## 10. Reproduction

```bash
python3 scripts/mamba-3/rotation_as_optimization.py
```

`numpy` only, float64 throughout, 69 checks. It exits non-zero on failure. The
section numbers in its output are the same as in this document. The script
encodes the recurrence of §2 directly, and never imports the crate. So the
Rust test suites assert, separately, that it agrees with the implementation
(`src/mamba3/product/tests.rs`, `src/mamba3/rotation/tests.rs`).

---

## References

**Architectures.**

- A. Gu, T. Dao. *Mamba: Linear-Time Sequence Modeling with Selective State
  Spaces*. arXiv:2312.00752.
- T. Dao, A. Gu. *Transformers are SSMs: Generalized Models and Efficient
  Algorithms Through Structured State Space Duality*. arXiv:2405.21060.
- *Mamba-3*. arXiv:2603.15569. §*Complex-Valued SSMs* and its appendix give
  the complex-to-real equivalence and the RoPE-trick propositions used in §2.
- J. Siems, T. Carstensen, A. Zela, F. Hutter, M. Pontil, R. Grazzi.
  *DeltaProduct: Improving State-Tracking in Linear RNNs via Householder
  Products*, 2025. The construction of §7, and its Prop. 1.3 spectral
  condition.
- I. Schlag, K. Irie, J. Schmidhuber. *Linear Transformers Are Secretly Fast
  Weight Programmers*, 2021. The delta rule as an online learner.
- S. Yang et al. *Parallelizing Linear Transformers with the Delta Rule over
  Sequence Length*, 2024.
- *Preconditioned DeltaNet: Curvature-aware Sequence Modeling for Linear
  Recurrences*, 2026. The metric corollary of §3, built: a diagonal
  approximation to the key Gram, which preconditions the write key.
- J. Su et al. *RoFormer: Enhanced Transformer with Rotary Position
  Embedding*. arXiv:2104.09864. The original form of the RoPE trick.

**State tracking and expressivity.**

- R. Grazzi et al. *Unlocking State-Tracking in Linear RNNs Through Negative
  Eigenvalues*, 2025.
- W. Merrill et al. *The Illusion of State in State-Space Models*, 2024.
- Y. Sarrof et al. *The Expressive Capacity of State Space Models*, 2024.

**Optimization.**

- B. T. Polyak. *Some methods of speeding up the convergence of iteration
  methods*, 1964. Heavy ball, §6.
- L. Mescheder, S. Nowozin, A. Geiger. *The Numerics of GANs*, 2017, and
  C. Daskalakis et al. *Training GANs with Optimism*, 2018. The cycling of
  gradient descent–ascent on saddles, and the extragradient/optimistic fixes
  that §5 refers to.
- A. van der Schaft, D. Jeltsema. *Port-Hamiltonian Systems Theory: An
  Introductory Overview*, 2014. The `ẋ = (J − R)∇H` form of §4.2.
- The Cartan–Dieudonné theorem: every element of `O(n)` is a product of at
  most `n` reflections. The expressivity argument behind the rank-one route of
  §7.

**Note on novelty.** Nothing in §§3–7 is a new result in optimization. Each
is a standard fact: the symmetry of Hessians, Wirtinger calculus, the
harmonicity of `Re` of a holomorphic function, GDA cycling, the complex
eigenvalues of heavy ball, products of symmetric matrices. The contribution of
this note is the assembly:

- these four constructions each break a different premise of Proposition 1
  (§3),
- three of them describe the same Mamba-3 recurrence,
- the resulting 2×2 derives the design decisions in `src/mamba3/product/` and
  `src/mamba3/rotation/`.
