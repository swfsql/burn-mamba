# Rotation as Optimization

### What a Mamba-3 step optimizes, and what `micro_steps` actually composes

> Reference note for `burn-mamba`. It is the authority for this crate's decisions
> around `Mamba3Config::micro_steps` (`src/mamba3/product/`) and its relationship
> to DeltaProduct, and for the optimization-theoretic reading of the complex
> transition (`src/mamba3/rotation/`).
>
> Every numbered claim below is checked in float64 by
> [`scripts/rotation_as_optimization.py`](../scripts/rotation_as_optimization.py)
> (66 checks, section numbers match). The script depends only on `numpy` and on
> the equations reproduced here — not on the crate — so the results stand
> independently of the implementation.

---

## Abstract

Linear RNNs admit a reading in which the recurrent state is a fast-weight matrix
being fit online, one gradient step per token: the transition is the quadratic
term of a local objective and the write is its linear term. Under that reading
Mamba-3's complex ("rotational") transition is usually declared out of scope,
because a rotation has complex eigenvalues and no objective in any metric
produces one.

We show the declaration is an artifact of assuming the step size is a positive
real number. Relaxing exactly one premise — in any of three different ways —
puts the rotation back inside the framework, exactly and with no approximation:

1. **the step size is complex** (`η ∈ ℂ`, or `ℍ`), on Mamba-2's unchanged convex
   objective. Its real part is the descent, its imaginary part is circulation
   tangent to the level sets, and `Re η > 0` — descent — is guaranteed by
   `|μ| ≤ 1`;
2. **the goal is a saddle, not a minimum**: descent in the real part of the state
   and *ascent* in the imaginary part, on a harmonic potential. Harmonic
   functions have no minima, so this is forced rather than chosen, and the
   rotation is the standard cycling of gradient descent–ascent;
3. **the method is momentum**: heavy ball on the *same* Mamba-2 objective, with
   momentum `β = α²` and step `ηρ = |1−μ|²`, has exactly the eigenvalues `μ, μ̄`.
   The imaginary half of the state is the velocity buffer.

A fourth escape exists and is the delta-rule family's: **compose several steps
whose Hessians do not commute**. That is DeltaProduct, and it requires
direction-dependent (rank-one) curvature. Mamba's curvature is isotropic, so its
per-step transitions commute and *no* product of them can leave the real axis.
This is the precise reason DeltaProduct's mechanism has no instance in Mamba-3,
and the precise sense in which this crate's `micro_steps` is a different
construction that reaches the same place.

The resulting design space is a 2×2 over (**rank of the curvature**) ×
(**algebra of the step size**), in which Mamba-2, DeltaNet/DeltaProduct and
Mamba-3/MambaProduct occupy three cells and the fourth is empty. Every claim the
`micro_steps` documentation makes about what `u > 1` buys per `RotationKind`
follows from one line of that table.

---

## 1. Scope and results

This note concerns the *state update* only: the map from the previous state and
the current token to the next state. Everything a block wraps around it — the
gate, the skip, the output projection, the normalisations — is outside the
object being classified, as is the read: the query never appears in any objective
below.

Results, in the order they are derived:

| § | claim |
|---|---|
| 3 | No first-order method with a real step size, on any real-valued loss, can produce a rotation. This is the obstruction, stated exactly. |
| 4 | Relaxing the step size to `ℂ` reproduces Mamba-3 exactly, on Mamba-2's objective. Descent is guaranteed, with `\|arg η\| ≤ arcsin α`. |
| 4.5 | Mamba-2's real constraint is a *step-size cap*: `ηρ ∈ (0,1)`, always short of a Newton step. The complex transition is what lets `ηρ` fill the disc. Parity sits on the real axis at `ηρ = 2`; mod-`k` has circulation fraction `cos(π/k)`. |
| 5 | The same recurrence is gradient descent–ascent on a harmonic (saddle-only) potential. |
| 6 | The same recurrence is heavy-ball momentum on the unchanged objective, with the velocity folded into the imaginary part of the state. |
| 7 | The delta-rule escape: composition. Verified sharp — rotation appears iff both micro-steps overshoot. |
| 8 | The design space, and what `micro_steps` is. |
| 9 | Consequences for this crate, including one that changes an implementation claim: the exponential integrator is load-bearing for state tracking, not merely for accuracy. |

Nothing here proposes a behavioural change to the library. §9 lists what it does
change: documentation, attribution, and the reasons behind three existing
defaults.

---

## 2. Setup and conventions

### 2.1 The recurrence

One head, dropping the trapezoid and MIMO until §9 (they are orthogonal). Mamba-3
with a complex transition, under the **exponential-Euler** discretisation: the
exponential on the transition, the held right endpoint on the state-input
(`α = e^{ΔA}`, `γ = Δ`). That is the scheme Mamba-1 and Mamba-2 actually
implement, and the Mamba-3 paper's discretisation table is what names it — the
same table notes that Mamba-1's paper reports ZOH (`γ = A⁻¹(e^{ΔA} − I)`) while
its released implementation does not use it. §5 turns on the distinction.

$$h_t = e^{\Delta_t(A_t + i\theta_t)}\,h_{t-1} + \Delta_t\,B_t\,x_t^\top,
\qquad y_t = \operatorname{Re}\big(C_t^\top h_t\big)$$

with `h ∈ ℂ^{n}` per value channel, `n = N/2`, `A_t ∈ ℝ` scalar (per head),
`θ_t ∈ ℝ^n` per state channel, and `B_t = B + iB̂`, `C_t = C + iĈ` complex
projections. Write

$$\mu_{t,j} := e^{\Delta_t(A_t + i\theta_{t,j})} = \alpha_t\,e^{i\varphi_{t,j}},
\qquad \alpha_t = e^{\Delta_t A_t}\in(0,1],\qquad \varphi_{t,j} = \Delta_t\theta_{t,j}$$

Realified, `μ` acts as `α·R(φ)` with `R` the 2×2 rotation — this is the source
paper's *complex-to-real SSM equivalence*, and the reason the crate's `state_rank`
`N` carries `N/2` rotating pairs.

### 2.2 Fast-weight orientation

To compare with the delta-rule literature, transpose to the fast-weight
convention: `S ∈ ℂ^{P×n}`, rows indexed by value channel, columns by complex
state channel, so that Mamba's `(B, x, C)` play the roles of (key, value, query):

$$\boxed{\;S_t = S_{t-1}\,\mathrm{D}(\mu_t) + \Delta_t\,x_t k_t^\top\;}
\qquad k_t := B_t,\quad x_t\in\mathbb{R}^P,\quad q_t := \overline{C_t}$$

The conjugate on `q` is not cosmetic. With the real inner product on `ℂ`

$$\langle a, b\rangle := \operatorname{Re}(\bar a\, b) \quad\text{(the Euclidean one on }\mathbb{R}^2)$$

the read is the honest pairing `y_t = ⟨q_t, S_t⟩` and the inner model is
`f_S(k) = Re(S k̄)`. That is exactly why the source paper's *real* read vector is
`[C; −Ĉ]` while its real write vector is `[B; +B̂]`: the read is conjugated, the
write is not. Anything that gets this backwards will produce a sign error in `y`
and nowhere else.

Gradients of real-valued functions of a complex variable are taken as the Riesz
representative for `⟨·,·⟩`, i.e. `∇ := 2 ∂/∂S̄` (Wirtinger). For
`L(S) = ½ρ‖S‖²` this gives `∇L = ρS`, as it should.

### 2.3 The framework being extended

The reading this note extends is the "linear RNN as test-time optimizer"
correspondence. For a recurrence `S_t = S_{t-1}M_t + u_tk_t^⊤` the update field is
affine, and integrating it along a ray recovers

$$\mathcal{L}_t(S) = \frac{1}{2\eta}\operatorname{tr}\!\big(SG_tS^\top\big) - \frac{1}{\eta}\big\langle Sk_t, u_t\big\rangle,
\qquad G_t := I - M_t$$

exact **iff `G_t` is symmetric**, i.e. iff `M_t` is. Allowing a symmetric positive
definite preconditioner `P` on key space widens this to: an objective exists in
*some* metric iff `M_t` is diagonalizable with real eigenvalues. Two standard
readings follow and are used below: **the transition is the quadratic term** (all
model identity lives in `G`), and **the write is the linear term** (every model
has the same one, so it distinguishes nothing).

For orientation, the two baselines in both display forms:

$$\mathcal{L}^{\text{DeltaNet}}_t(S)=\tfrac12\lVert Sk_t-v_t\rVert^2
=\underbrace{\tfrac12\lVert Sk_t\rVert^2}_{\text{quadratic: }G=k k^\top}
-\underbrace{\langle Sk_t,v_t\rangle}_{\text{linear: write}}
+\underbrace{\tfrac12\lVert v_t\rVert^2}_{\text{const}}$$

$$\mathcal{L}^{\text{Mamba-2}}_t(S)=\underbrace{\frac{a_t}{2}\lVert S\rVert_F^2}_{\text{quadratic: }G=aI}
-\underbrace{\langle Sk_t,x_t\rangle}_{\text{linear: write}}
\;=\;\frac{a_t}{2}\Big\lVert S-\frac{x_tk_t^\top}{a_t}\Big\rVert_F^2+\text{const},
\qquad \eta_t=\Delta_t,\ a_t=-A_t$$

DeltaNet regresses toward a **vector** target in the seminorm induced by `kk^⊤`,
blind to `S` off `k`. Mamba-2 regresses toward a **rank-one matrix** target in the
Frobenius metric, penalising every direction equally. *Rank-one, direction-dependent
curvature* versus *isotropic curvature* is the distinction that decides everything
in §7 and §8.

---

## 3. The obstruction

> **Proposition 1.** Let `L` be any real-valued, twice-differentiable loss and let
> `η ∈ ℝ`. Then the transition induced by one step of preconditioned gradient
> descent, `I − η∇²L`, has a real spectrum. Consequently no such step can realise
> a rotation.

*Proof.* `∇²L` is symmetric, hence orthogonally diagonalizable with real
eigenvalues; `I − η∇²L` shares its eigenvectors. ∎

Verified over 2000 random (loss, step) draws with steps of both signs
(§3 of the script).

Two immediate corollaries worth stating, because both are tempting escapes that
do not work:

- **Complexifying the loss, with the step size left real, does not help.** The
  Hessian of a real-valued quadratic on `ℂ^n` is Hermitian, so a "complex delta
  rule" factor `I − βkk^*` still has a real spectrum. Verified. Read this as a
  statement about the *step*, not about the loss: the same factor with `β` on the
  complex disc rotates in 100% of draws, which is the cell §8(iii) calls the
  interesting extrapolation.
- **Changing the metric does not help.** An SPD preconditioner `P` yields
  `G = (I−M)P^{-1}` symmetric only when `M` is diagonalizable over `ℝ`, which a
  rotation is not.

Proposition 1 is therefore the right frame: it says precisely which premise must
break. It has four, and the note takes one section each.

| premise | broken by | § |
|---|---|---|
| the step size is a real scalar | a step in `ℂ` or `ℍ` | 4 |
| the objective is minimized | descent–ascent on a saddle | 5 |
| the update depends only on the current state | momentum — a larger state space | 6 |
| one step per token | composing `u` of them | 7 |

The third is the loosest fit and is flagged as such in §6: heavy ball does not
contradict Proposition 1 at all, since it produces its rotation by *enlarging the
state space* rather than by rotating on the given one. It earns its section by
landing on the same transition, not by escaping the same statement. Nor is the
list claimed exhaustive — these are four constructions that work, over premises
that admit other breaks.

---

## 4. View I — the step size is complex

### 4.1 The construction

Keep the state complex and the update `ℂ`-linear. A real-valued objective whose
gradient step is `ℂ`-linear must have a *Hermitian* quadratic part, i.e. real
curvature `ρ_j > 0` per channel. Take

$$\boxed{\;
\mathcal{L}_t(S)=\underbrace{\frac12\sum_j\rho_{t,j}\lVert S_{:,j}\rVert^2}_{\text{quadratic: transition}}
-\underbrace{\big\langle \operatorname{Re}(S\bar w_t),\ x_t\big\rangle}_{\text{linear: write}}
\;=\;\frac12\sum_j\rho_{t,j}\Big\lVert S_{:,j}-\frac{x_tw_{t,j}}{\rho_{t,j}}\Big\rVert^2+\text{const}\;}$$

— structurally *identical* to Mamba-2's: convex, separable, isotropic per
channel, a proximal pull toward a rank-one target — and let the step size be a
complex number per channel:

$$S_t = S_{t-1} - \nabla\mathcal{L}_t(S_{t-1})\,\mathrm{D}(\eta_t),\qquad \eta_{t,j}\in\mathbb{C}$$

Matching against the recurrence gives two conditions and nothing else:

$$\rho_{t,j}\,\eta_{t,j} = 1-\mu_{t,j},\qquad \eta_{t,j}\,w_{t,j} = \Delta_t\,k_{t,j}$$

Only the products are determined — the split into `(ρ, η, w)` is a gauge freedom,
verified exact in three different gauges. Two are natural:

**Gauge (a) — Mamba-2's own curvature.** `ρ_{t,j} = Re(1−μ_{t,j})/Δ_t`. Then

$$\eta_{t,j} = \Delta_t\big(1 + i\tan\psi_{t,j}\big),\qquad \psi_{t,j} := \arg(1-\mu_{t,j}),
\qquad \operatorname{Re}\eta_{t,j} = \Delta_t\ \text{exactly}$$

**Gauge (b) — unchanged step length.** `ρ_{t,j} = |1−μ_{t,j}|/Δ_t`, giving
`η = Δ_t e^{iψ}` and `w = e^{−iψ}k` — the objective's key is the physical key
rotated by `−ψ`, modulus preserved.

In continuous time gauge (a) is `Δ`-free, exactly as Mamba-2's is:

$$\rho_j = a,\qquad \eta_j = 1 - i\frac{\theta_j}{a},\qquad w_j = \frac{a}{a-i\theta_j}k_j,
\qquad S^\star_{:,j} = \frac{x\,k_j}{a-i\theta_j}$$

and `S*` is the true steady state of the ODE, as it must be.

### 4.2 The Helmholtz split is the polar form of `η`

The usual structural statement about a rotational transition splits the generator
into symmetric (dissipative, "the objective") and skew (circulating, "invisible")
parts. In the complex frame that split is literally the real/imaginary split of a
single complex number — verified in the 2×2 realification:

$$G = I - \alpha R^\top = \underbrace{\operatorname{Re}(1-\mu)}_{1-\alpha\cos\varphi}\,I
\;-\;\underbrace{\operatorname{Im}(1-\mu)}_{-\alpha\sin\varphi}\,J,
\qquad J=\begin{bmatrix}0&-1\\1&0\end{bmatrix}$$

So the recovered objective already had the right curvature: `ρ = Re(1−μ)/Δ` in
gauge (a) *is* the symmetric part that the classification recovers. **The skew
part is not a piece of the objective that went missing. It is the imaginary part
of the learning rate.** The same recurrence, written as a flow:

$$\dot S = -(\operatorname{Re}\eta)\,\nabla\mathcal{L} \;-\; (\operatorname{Im}\eta)\,i\nabla\mathcal{L},
\qquad \langle i\nabla\mathcal{L},\,\nabla\mathcal{L}\rangle = 0$$

a gradient part plus a part tangent to the level sets of *the same* `L` — the
dissipative-Hamiltonian (port-Hamiltonian) form `ẋ = (J − R)∇H`, with `L` serving
as both Lyapunov function and Hamiltonian.

### 4.3 Descent is free

Multiplication by `η` has symmetric part **exactly** `Re(η)·I` (verified), so

$$\big\langle \nabla\mathcal{L},\ \text{step}\big\rangle = -\sum_j \operatorname{Re}(\eta_{t,j})\,\lVert\nabla\mathcal{L}_{:,j}\rVert^2$$

and the step is a descent direction iff `Re η > 0`, i.e. `|ψ| < π/2`. This is the
operational content of "the preconditioner must be positive definite", and for
Mamba-3 it costs nothing:

> **Proposition 2.** For `|μ| = α ≤ 1`, `Re(1−μ) = 1 − α cos φ ≥ 0` and
> `|arg(1−μ)| ≤ arcsin α`, attained at `cos φ = α`.

*Proof.* `1 − μ` lies on the circle of radius `α` about `1`; the origin is outside
it for `α < 1`, and the tangent lines from the origin subtend `arcsin α`. ∎

Verified over the whole closed disc. So **every admissible Mamba-3 parameter gives
a genuine descent step** — for the abelian kind here, and for all four by
Proposition 2′ in §8, which needs only that the transition is `α ×` an isometry —
with an exact accounting:

$$\text{descent fraction} = \cos\psi = \frac{1-\alpha\cos\varphi}{|1-\alpha e^{i\varphi}|},
\qquad \text{circulation fraction} = \sin\psi$$

### 4.4 What the rotation buys, in optimizer coordinates

Since `ηρ = 1 − μ`, the reachable set of the step-curvature product is the unit
disc about `1`. Comparing families in that one coordinate is the most compressed
statement in this note:

| model | `ηρ` | reading |
|---|---|---|
| Mamba-2 | `1 − α ∈ (0,1)` | **always undershoots** — cannot even reach the Newton step `ηρ = 1` |
| DeltaNet | `β ∈ (0,2)` | the full classical range, overshoot included |
| Mamba-3, `φ = 0` | `1 − α ∈ (0,1)` | Mamba-2 |
| Mamba-3, `φ = π` | `1 + α → 2` | overshoot at the classical stability edge `2/ρ`; `μ → −1` |
| Mamba-3, `φ ∉ {0,π}` | off the real axis | genuine circulation, `ψ ≠ 0` |

> **Mamba-2's restriction is not "no rotations". It is a step-size cap.**
> `α = e^{ΔA} ∈ (0,1)` forces `ηρ ∈ (0,1)`: strictly less than one Newton step,
> let alone the classical stability limit at `2`. The complex transition is what
> lets `ηρ` leave that interval.

Two consequences that sharpen the usual story about state tracking:

- **Parity is not in the non-integrable regime.** Parity wants `μ = −1`, i.e.
  `ψ = 0`: *pure descent*, at exactly `η = 2/ρ`, whose textbook behaviour is
  period-2 oscillation. (Reaching `φ ≈ π` requires `|A| ≪ θ`, which drives
  `ψ → 0`, so the coupling of `α` and `φ` through `Δ` reinforces rather than
  spoils this.) What Mamba-2 lacks here is *overshoot*, not "an objective for
  rotation".
- **Mod-`k` is, and quantifiably.** At `α → 1` and `φ = 2π/k`,

  $$|\psi| = \frac{\pi}{2}-\frac{\pi}{k},\qquad \text{circulation fraction} = \cos\frac{\pi}{k}$$

  giving `0` at `k=2`, `0.5` at `k=3`, `0.707` at `k=4`, `→1` as `k→∞`. The
  fraction of the update that no objective can account for is a closed form in
  the order of the group being tracked.

---

## 5. View II — the goal is a saddle

View I relaxes the *step*. The alternative is to keep a real step and relax the
*goal*. Treat the two real coordinates of a complex channel as two players:

$$F_t(S) = \operatorname{Re}\Big(-\tfrac12\operatorname{tr}\big(S\,\mathrm{D}(\tilde A_t)\,S^\top\big)
- x_t^\top S k_t\Big),\qquad \tilde A_t = A_t + i\theta_t$$

and run **descent in `Re S`, ascent in `Im S`**. That field is exactly the
Mamba-3 complex flow, and exponentiating it over one interval is exactly the
transition — both verified, the second by building the realified generator from
the field and exponentiating it, so it shares no subexpression with the
transition it is compared against.

The sign is the part worth pausing on, because it is what a reader redoing this
by hand gets wrong. For `F = Re(φ)` with `φ` holomorphic, Cauchy–Riemann gives
`∂F/∂z_r = Re(φ′)` and `∂F/∂z_i = −Im(φ′)`, so the descent–ascent field
`(−∂F/∂z_r, +∂F/∂z_i)` is `−φ′(z)` as a complex number: **ascending in the
imaginary part is exactly what cancels the conjugation that descending in the
real part alone would introduce.** Descending in both would give `−conj(φ′)`,
which is not a flow of this system.

One precision. Only the *homogeneous* part is the exact flow map. Integrating the
forcing exactly would be ZOH, `γ = A⁻¹(e^{ΔA} − I)`, where Mamba holds the right
endpoint at `γ = Δ` (§2.1). So this is the approximation Mamba-1/-2 already make
and the min–max reading inherits, not one it introduces; the two agree to `O(Δ)`.

This is more than a re-parameterisation, for two reasons.

- `F` is the real part of a holomorphic function, hence **harmonic** (verified
  numerically: Laplacian zero). By the maximum principle a harmonic function has
  **no minima at all** — every critical point is a saddle. So the min–max reading
  is not a stylistic choice; a minimization reading of this potential does not
  exist.
- The rotation is then an entirely standard phenomenon: **gradient descent–ascent
  cycles on saddles**. In this dictionary the decay `a` is the strong
  convexity/concavity of the two players, `θ` is their bilinear coupling, and
  their ratio is precisely §4's angle, `tan ψ = θ/a`. GDA converges when
  convexity dominates and orbits when coupling does — `ψ → π/2` is the pure
  state-tracking limit.

Worth noting the direction of the literature: extragradient and optimistic-GDA
methods exist specifically to *suppress* this cycling. Mamba-3 wants it, and the
exponential integrator (§9.5) is what keeps it exact.

---

## 6. View III — the method is momentum

The third relaxation keeps both the objective and the real step size and changes
the *order* of the method. Heavy ball on Mamba-2's unchanged objective,

$$z_{t+1} = z_t - \eta\nabla\mathcal{L}(z_t) + \beta\,(z_t - z_{t-1})$$

has companion matrix `[[1+β−ηρ, −β], [1, 0]]` on `(z_t, z_{t−1})`.

> **Proposition 3.** For any `α ∈ (0,1)` and `φ ∉ {0, π}`, setting
> `β = α²` and `ηρ = |1−μ|² = 1 + α² − 2α cos φ` makes the companion matrix's
> eigenvalues exactly `μ` and `μ̄`; the matrix is therefore similar to `α·R(φ)`.

Verified over 500 random `(α, φ, ρ)` draws, exact to `1e-9`, with the closed form
`ηρ = |1−μ|²` to `1e-15`.

So the complex transition **is** momentum, with the velocity buffer folded into
the imaginary part of the state rather than held as a second tensor. Momentum's
usual price is a second state slot; Mamba-3 pays it by declaring half of the
state imaginary.

Two precisions, so this is not overclaimed.

The *transition* is exactly heavy ball's up to a change of basis. Mamba-3's
*write* is more general — `B` and `B̂` are independent projections, so the input
drives both the iterate and the velocity, where heavy ball drives only the
iterate.

And the correspondence has no `φ ∈ {0, π}` member, which is why Proposition 3
excludes them. It is not that the map degrades there: a companion matrix is
**non-derogatory** (its minimal polynomial is its characteristic polynomial), so
it is never similar to a scalar pair `diag(α, α)` — which is exactly what a
*non-rotating* Mamba-3 pair is. At `φ = 0` the parameters above give a defective
Jordan block, `rank(C − αI) = 1`, with transient growth to `2.84` before decaying,
where the Mamba-3 pair is diagonal and monotone. So momentum is a reading of the
*rotating* channels only; it does not extend along `rope_fraction` to the
unrotated ones, which never enter the correspondence at all.

Views I, II and III are three readings of one recurrence. I and II are the same
mechanism (`Im η` is the ascent direction); III is genuinely a different method
that lands on the same transition.

---

## 7. View IV — composing non-commuting steps

The fourth escape from Proposition 1 does not touch the loss, the step or the
method. It uses the fact that **a product of symmetric matrices need not be
symmetric**: `u ≥ 2` ordinary steps can compose into a rotation even though no
single one can. This is DeltaProduct's mechanism.

DeltaProduct takes `u` delta-rule micro-steps per token on `u` different
`(k_j, v_j)` pairs, giving the transition

$$M_t = \alpha_t\prod_{j=1}^{u}\big(I - \beta_{t,j}k_{t,j}k_{t,j}^\top\big)$$

Two verified facts pin down when this works:

- **Rank-one curvature rotates, and sharply.** At `u = 2`, `d_k = 6`,
  `β ∼ U(0,2)`: ~16–18% of random draws have a complex spectrum, and in **every**
  rotating draw both `β_j > 1` (4000 draws, zero counterexamples). Since a factor
  `I − βkk^⊤` has eigenvalue `1 − β` along `k`, `β > 1` means that micro-step
  *overshoots its own minimizer*. **Rotation is manufactured out of two
  overshoots** — which is the same coordinate §4.4 uses, reached by a different
  construction.
- **Isotropic curvature never rotates.** With `∇²L = ρI` every factor is
  `(1 − η_jρ_j)I`, a scalar. Scalars commute, so the product is a scalar:
  `0 / 2000` draws at `u = 3` acquire a complex spectrum.

The second is the load-bearing one for this crate, and it deserves stating as the
proposition it is:

> **Proposition 4.** If the local curvature is isotropic (`∇²L_j = ρ_jI`), then
> the per-micro-step transitions commute and every product of them is a real
> scalar multiple of the identity. No number of micro-steps can produce a
> rotation, and the only thing `u` can change is the linear term — i.e. the
> write.

That is the exact reason DeltaProduct's mechanism has **no instance** in Mamba-3.
The usual informal version, "Mamba has no erase", names a symptom; isotropy is
the cause. It also derives, rather than asserts, the statement that `u`
micro-writes under a shared scalar transition collapse into a single rank-`u`
write — which is MIMO.

---

## 8. The design space

Write both families in one form. Both are `u` first-order steps per token:

$$\boxed{\;M_t=\prod_{j=1}^{u}\Big(I-\eta_{t,j}\,\nabla^2\mathcal{L}_{t,j}\Big)\;}$$

They differ in two independent dials — the **rank of the curvature** and the
**algebra of the step size**:

|  | curvature `ρI` (isotropic) | curvature `kkᵀ` (rank-one) |
|---|---|---|
| **`η ∈ ℝ`** | Mamba-2 — `u` collapses to a rank-`u` write (MIMO) | DeltaNet, **DeltaProduct** |
| **`η ∈ ℂ`, `ℍ`** | **Mamba-3, MambaProduct** | *empty* — a "rotational delta rule" |

Three things follow directly, and they are exactly the claims the `micro_steps`
documentation currently asserts without derivation.

**(i) Why `micro_steps` does not exist on Mamba-2.** Proposition 4: isotropic
curvature, real step, commuting factors. `u` buys only the write, and the write is
MIMO's job.

**(ii) Why the `RotationKind` split falls exactly where it does.** With isotropic
curvature the transition factors are elements of the step algebra, so what `u`
buys is a property of that algebra alone:

| kind | step algebra | factors | what `u > 1` buys |
|---|---|---|---|
| `Real1D` | `ℝ` | commute, real | the write only — `u` staggered rank-1 writes, i.e. `mimo_rank`'s job read sequentially |
| `Complex2D` | `ℂ` | commute, phases **add** | `u`× the per-token angle reach, with a live gradient at every factor |
| `Quaternion4D`, `Rotor4D` | `ℍ`, two-sided | **do not commute** | a per-token transition no single bounded step expresses |

Verified: over `ℂ` the product depends only on the sum of the angles (order-free);
over `ℍ` the same generators in the reverse order give a different product, and
the product also leaves `exp(Σ generators)`.

**(iii) The empty cell is real and is the interesting extrapolation.** Rank-one
curvature with a complex step — a delta rule whose transition carries a
data-dependent rotation. The rank-one erase does conjugate through a rotational
gauge, `P*(I − βkk^H)P = I − β(P*k)(P*k)^H`, which needs only that `P` is a
linear isometry and therefore survives the non-abelian kinds unchanged.

That is **not** enough to conclude "no new chunkwise algorithm", which an earlier
version of this section claimed and which is true of only one of the two orderings.
The recurrence is affine, and the gauge pins the *write* key to `P_t*k_t` however
the step is ordered, while the *erase* key follows the rotation:

| ordering within a step | erase key | write key | |
|---|---|---|---|
| rotate, then erase | `P_t*k_t` | `P_t*k_t` | **tied** — the existing WY/tied-key kernel applies verbatim |
| erase, then rotate | `P_{t−1}*k_t` | `P_t*k_t` | **untied** — the generalized-DPLR shape, which that kernel does not compute |

Verified, in both the left- and right-multiplication orientations — the rows above
name the *temporal* order, and a product does not survive transposition unreversed,
so "rotate first" is the leftmost matrix factor in one convention and the rightmost
in the other. Reading one convention's equation in the other's order turns the tied
model into the untied one, which is a live failure mode rather than a hypothetical.

The erase factor stays a proper tied Householder in both cases, so the `‖·‖₂ ≤ 1`
bound and the `u > 1` stability argument survive either way — what the wrong
ordering costs is the kernel, not the guarantee. Since Mamba-3 has no erase,
nothing here constrains this crate; it constrains anyone building the cell.

It has a **precondition** that is easy to miss: a complex step rotates a rank-one
curvature only if that curvature is `ℂ`-Hermitian, which forces the regression
*target* to be complex too. With a real target the curvature is `KKᵀ` over
`ℝ^{2n}`, and since `KᵀJK = 0`,

$$M = I - (aI + bJ)KK^\top \;=\; \begin{bmatrix}1-a & 0\\ -b & 1\end{bmatrix}
\quad\text{on } \operatorname{span}\{K, JK\}$$

— triangular, eigenvalues `1−a` and `1`, both **real**, and non-normal: a shear,
with the norm bound gone. Verified: `0%` of draws rotate with a real target
(`max‖M‖₂ = 1.41`), `100%` with a complex one (`‖M‖₂ = 1`). So tying the target's
imaginary half to its real half, or zeroing it, is not a cheaper variant of this
cell — it deletes the mechanism. Mamba-3's own value `x` is real and this does not
bite, because isotropic `ρI` is `ℂ`-Hermitian for free; the precondition is
specific to the rank-one column.

A block in that cell also carries **two** step sizes, hence two phases, at two
different rates — which dissolves the question of whether "the" rotation belongs
per token or per micro-step:

| term | curvature | step | phase | rate |
|---|---|---|---|---|
| ridge `(ρ/2)‖S‖²` | isotropic | `η` | `arg η` — Mamba-3's | once per **token** |
| data fit `½‖kᴴS − v‖²` | rank-one over `ℂ` | `β` | `arg(1−β)` | once per **micro-step** |

The ridge step is per token because the `u` corrective steps share one interval
(§7); `β` is per micro-step because there are `u` of them. The two are
independent: the decay's phase acts on every plane, the erase's only in the plane
its own key spans, so for `n ≥ 2` no single decay phase with real erase gates
reproduces a token carrying two erase phases. Each phase already rides an existing
gate at the right rate, so neither needs a configuration knob.

The ladder `ℝ ⊂ ℂ ⊂ ℍ` is uniform in a way worth recording, because it is why the
crate's rotation code is unbranched:

| algebra | `η = (1−μ)/ρ` | symmetric part of `v ↦ ηv` | descent condition |
|---|---|---|---|
| `ℝ` | positive scalar | `η·I` | `η > 0` |
| `ℂ` | complex scalar | `Re(η)·I` | `Re η > 0` |
| `ℍ` | quaternion | `Re(η)·I` | `Re η > 0` |
| two-sided `(q, p)` | pair | not a multiple of `I`; trace `= 4·Re(q)Re(p)` | `α < 1` (below) |

All verified; Proposition 2's bound `|arg| ≤ arcsin α` holds verbatim over `ℍ`.

The two-sided row is the one that does not fit the pattern — `v ↦ qvp̄` is not
multiplication by a scalar in any algebra, and its symmetric part is not a multiple
of `I`, so `Re(η) > 0` has nothing to attach to. Its normalised trace is
`Re(q)Re(p)`, which is negative about half the time, and it is tempting to read that
as descent failing and needing a sign constraint. **It is not**: the trace is the
*average* of `⟨Tv, v⟩` over directions, not the condition. The condition is an
eigenvalue statement, and it is satisfied uniformly:

> **Proposition 2′ (descent, all four kinds).** Let `M = αT` with `T` any isometry
> and `α ∈ (0,1)`. Then `sym(I − M) = I − α·T_sym ≽ (1−α)I ≻ 0`.
>
> *Proof.* `‖T‖₂ = 1`, so `|⟨Tv, v⟩| ≤ ‖v‖²` and every eigenvalue of `T_sym` lies in
> `[−1, 1]`; hence every eigenvalue of `I − αT_sym` is at least `1 − α`. ∎

Verified, including on the draws where `Re(q)Re(p) < 0` (`λ_min > 0` in every one of
20 000). So **descent is free for every `RotationKind`, with a uniform margin
`1 − α`** — the conformality bullet above is what buys it, and the per-algebra
`Re(η) > 0` rows are the sharper single-sided specialisation (`1 − α cos φ ≥ 1 − α`),
not a separate condition.

Two properties of *this* enlargement are load-bearing, and they are why the escape
is "the step size joins a normed division algebra" rather than the strictly larger
"the preconditioner need not be symmetric" — which is vacuous on its own, since any
`M` is `I − GP` for symmetric `G` and general `P`:

- **Every kind is `α × isometry`, hence normal, hence `‖M‖₂ = ρ(M) = α` exactly.**
  This includes `Rotor4D`, whose `v ↦ qvp̄` is in `SO(4)`. For a general
  non-symmetric preconditioner the gap between norm and spectral radius is not
  merely present but *unbounded*: conjugating a rotation by an ill-conditioned `D`
  leaves the spectrum on the unit circle while `‖M‖₂` grows with `cond(D)`
  (verified: `‖M‖₂/ρ(M) = 35` at `cond(D) = 50`). That forfeits the bound a
  *time-varying* product needs:
  boundedness of `M_t⋯M_1` does not follow from per-factor spectral radius, but
  does follow from a submultiplicative `‖M_t‖₂ ≤ 1`. Conformality is what makes the
  state-tracking norm argument exact rather than asymptotic.
- **Commutativity is not what makes the RoPE trick work.** The trick needs only
  that the transition is *scalar × group element*: `α` is scalar so it commutes
  with everything, and the cumulative rotation telescopes as
  `R_{i+1..t} = R̄_t R̄_i^{-1}` by associativity and invertibility. Commutativity is
  what collapses that telescoping scan into a closed-form **cumsum of angles**; its
  absence is exactly why the non-abelian kinds need a scan with a cross-chunk
  accumulator. The carry exists *because* the factors do not commute, not despite
  it — which is also why the trick survives `Quaternion4D` and `Rotor4D` unchanged.

---

## 9. Consequences for this crate

### 9.1 What `micro_steps` is, and how to describe it

`Mamba3Config::micro_steps` (`u`) runs `u` full Mamba-3 recurrence steps per
token, each drawing its own `Δ`, `A`, `λ`, `B` and rotation from its own slice of
the input projection, evaluated by folding the micro-steps into the sequence axis.
By §7–§8, this is *not* DeltaProduct's mechanism carried over — that mechanism has
no instance here — but it is the same construction with the other dial turned:

> DeltaProduct builds a non-commuting product out of the **curvature** (a
> different rank-one Hessian per micro-step). Mamba-3's curvature is isotropic, so
> `micro_steps` builds one out of the **step size** instead. Same equation,
> orthogonal dial.

The name `micro_steps` is accurate. "Product" remains accurate — the per-token
transition genuinely is a product of `u` non-commuting factors at the quaternion
kinds — provided it is not read as "product of Householders". DeltaProduct should
be cited as the source of the *dial* (`u` first-order steps per token, transition
expressiveness rather than memory) and not of the *mechanism*.

### 9.2 The reach claim, attributed to itself

At `Complex2D` the `u` factors commute and their phases add, so `u` micro-rotations
each comfortably inside the per-step bound compose to `u`× the reach with a live
gradient at every factor. This is the real payoff at the abelian kind and it needs
no external attribution: it follows from §8(ii) plus the fact that a single
rotation *at* the `rotation_range` bound sits on `tanh`'s asymptote, where the f32
gradient is exactly zero.

The mechanism should be stated honestly: it is bought by giving the token `u`
full-size steps, i.e. by inflating its effective interval, not by subdividing it.
The consistent alternative (`Δ_j = Δ/u`, same per-token transition, buying only the
staggered writes and the non-abelian ordering) is inside the model's reachable set —
`dt_limit` has no lower floor by default — so this is a superset, not an error.
(It is a configurable clamp, so a run that raises the floor above `Δ/u` gives that
up.)

### 9.3 `micro_steps` versus `mimo_rank`

Both widen the write. §7 says why they are not redundant and why the difference
vanishes at `Real1D`: with isotropic curvature the transition factors do not depend
on the key at all, so at `Real1D` all `u` can change is the linear term, and a
sequential epoch of `u` samples with decay-staggered weights is what `mimo_rank`
occupies jointly as a minibatch of `M`. `u` becomes structurally distinct exactly
when the step algebra is non-trivial.

### 9.4 The trapezoid composes with all of it

The trapezoidal discretisation is a *two-sample* linear term under the same complex
step, with the older sample **parallel-transported into the current frame** by this
step's rotation before being weighted:

$$\text{linear term} \ \propto\ \gamma_t\,x_tk_t^\top \;+\; \beta_t\,x_{t-1}\big(e^{i\varphi_t}\odot k_{t-1}\big)^\top$$

Verified exact over a full trajectory. The usual reading of the trapezoid as a
two-tap FIR filter on the gradient stream survives the complex transition; it is a
*transported* filter, not a naive one. This is also why the crate's caches store the
raw previous-token `(B, x)` and re-weight them at time `t` rather than storing a
pre-decayed contribution.

### 9.5 The exponential integrator is load-bearing (implementation claim)

State tracking requires the transition to be exactly norm-preserving in the
undamped limit. Forward Euler on a pure rotation has modulus `|1 + iΔθ|`, which is
`1.28` already at `Δθ = 0.8`: it spirals outward. Exponential-Euler gives modulus
exactly `1.0`.

So the choice of exponential (rather than forward-Euler) discretisation is not only
about second-order accuracy, which is how it is usually presented — **it is what
makes the transition orthogonal, and therefore what makes parity exact rather than
drifting.** This is worth stating wherever the discretisation is documented, and it
is independent of everything else in this note.

### 9.6 What does not follow

- No numerical behaviour changes. Nothing here is a bug report; every proposition
  is about the recurrence as implemented.
- The three views do not make the rotation "just optimization". They locate it
  precisely: it is the imaginary part of a step size, equivalently an ascent
  direction, equivalently a velocity. Under all three, the query still never
  appears in any objective, and the reading still says nothing about what is read
  back out.

---

## 10. Reproduction

```bash
python3 scripts/rotation_as_optimization.py
```

`numpy` only; float64 throughout; 66 checks; exits non-zero on failure. Section
numbers in its output match this document's. The script encodes the recurrence
from §2 directly and never imports the crate, so agreement between it and the
implementation is asserted separately, by the Rust test suites
(`src/mamba3/product/tests.rs`, `src/mamba3/rotation/tests.rs`).

---

## References

**Architectures.**

- A. Gu, T. Dao. *Mamba: Linear-Time Sequence Modeling with Selective State
  Spaces*. arXiv:2312.00752.
- T. Dao, A. Gu. *Transformers are SSMs: Generalized Models and Efficient
  Algorithms Through Structured State Space Duality*. arXiv:2405.21060.
- *Mamba-3*. arXiv:2603.15569. §*Complex-Valued SSMs* and its appendix carry the
  complex-to-real equivalence and the RoPE-trick propositions used in §2.
- J. Siems, T. Carstensen, A. Zela, F. Hutter, M. Pontil, R. Grazzi.
  *DeltaProduct: Improving State-Tracking in Linear RNNs via Householder
  Products*, 2025. §7's construction and its Prop. 1.3 spectral condition.
- I. Schlag, K. Irie, J. Schmidhuber. *Linear Transformers Are Secretly Fast
  Weight Programmers*, 2021. The delta rule as an online learner.
- S. Yang et al. *Parallelizing Linear Transformers with the Delta Rule over
  Sequence Length*, 2024.
- J. Su et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding*.
  arXiv:2104.09864. The RoPE trick's original form.

**State tracking and expressivity.**

- R. Grazzi et al. *Unlocking State-Tracking in Linear RNNs Through Negative
  Eigenvalues*, 2025.
- W. Merrill et al. *The Illusion of State in State-Space Models*, 2024.
- Y. Sarrof et al. *The Expressive Capacity of State Space Models*, 2024.

**Optimization.**

- B. T. Polyak. *Some methods of speeding up the convergence of iteration
  methods*, 1964. Heavy ball; §6.
- L. Mescheder, S. Nowozin, A. Geiger. *The Numerics of GANs*, 2017, and
  C. Daskalakis et al. *Training GANs with Optimism*, 2018. Gradient
  descent–ascent cycling on saddles, and the extragradient/optimistic fixes
  referenced in §5.
- A. van der Schaft, D. Jeltsema. *Port-Hamiltonian Systems Theory: An
  Introductory Overview*, 2014. The `ẋ = (J − R)∇H` form of §4.2.
- Cartan–Dieudonné theorem: every element of `O(n)` is a product of at most `n`
  reflections. The expressivity argument behind §7's rank-one route.

**Note on novelty.** Nothing in §§3–7 is a new result in optimization; each is a
standard fact (symmetry of Hessians, Wirtinger calculus, harmonicity of `Re` of a
holomorphic function, GDA cycling, heavy ball's complex eigenvalues, products of
symmetric matrices). The contribution of this note is the assembly — that these
four constructions each break a different premise of Proposition 1 (§3), that
three of them describe the same Mamba-3 recurrence, and that the resulting 2×2 derives the design
decisions in `src/mamba3/product/` and `src/mamba3/rotation/` which were
previously stated as observations.
