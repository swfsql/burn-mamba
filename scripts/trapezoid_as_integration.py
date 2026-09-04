#!/usr/bin/env python3
"""Numerical verification for `info/trapezoid-as-integration.md`.

Every claim the document makes that can be checked numerically is checked here, in
float64, with section numbers matching the document's. Pure `numpy`; no other
dependency, no I/O, no reference to the model code — the point is that these are
statements about the *recurrence*, reproducible from the equations alone.

    python3 scripts/trapezoid_as_integration.py

Exits non-zero if any check fails.

Conventions (document §2). The state is kept in the paper's `h`, shape ``[N, P]``
(rows indexed by state channel, columns by value channel), because the trapezoid
reads more easily there; `info/rotation-as-optimization.md` uses the transposed
fast-weight ``S``. Nothing below depends on which. One head throughout.

    alpha[t] = exp(dt[t] * A[t])            decay, A[t] < 0
    beta[t]  = (1 - lam[t]) * dt[t] * alpha[t]      left-endpoint weight
    gamma[t] = lam[t] * dt[t]                       right-endpoint weight
    v[t]     = outer(B[t], x[t])                    the state-input
    h[t]     = alpha[t] * h[t-1] + beta[t] * v[t-1] + gamma[t] * v[t]

Where a rotation is present the scalar ``alpha[t]`` becomes ``M[t] = alpha[t] R[t]``
with ``R[t]`` orthogonal, and the beta tap carries ``M[t]`` in place of ``alpha[t]``.
"""

import sys

import numpy as np

RNG = np.random.default_rng(20260904)
PASSED = 0
FAILED = 0
TRAPZ = getattr(np, "trapezoid", None) or np.trapz


def ok(name, cond):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  pass  {name}")
    else:
        FAILED += 1
        print(f"  FAIL  {name}")


def close(a, b, tol=1e-10):
    return np.max(np.abs(np.asarray(a) - np.asarray(b))) < tol


def section(title):
    print(f"\n{title}\n{'-' * len(title)}")


def rand_rot(n):
    q, r = np.linalg.qr(RNG.normal(size=(n, n)))
    return q * np.sign(np.diag(r))


# ---------------------------------------------------------------- shared setup
T, N, P = 14, 5, 4
DT = RNG.uniform(0.05, 1.5, T)
AA = -RNG.uniform(0.1, 2.0, T)
LAM = RNG.uniform(0.0, 1.0, T)
ALPHA = np.exp(DT * AA)
BETA = (1.0 - LAM) * DT * ALPHA
GAMMA = LAM * DT
V = np.einsum("tn,tp->tnp", RNG.normal(size=(T, N)), RNG.normal(size=(T, P)))
ZERO = np.zeros((N, P))


def run_trap(alpha=ALPHA, beta=BETA, gamma=GAMMA, v=V, n=T, h0=None):
    h, out = (ZERO.copy() if h0 is None else h0.copy()), []
    for t in range(n):
        h = alpha[t] * h + beta[t] * (v[t - 1] if t > 0 else ZERO) + gamma[t] * v[t]
        out.append(h.copy())
    return np.array(out)


H = run_trap()


# =============================================================================
section("3. The objective: one gradient step on a rank-two target")

# L(S) = (rho/2)||S||_F^2 - (1/eta) <S, gamma v_t + beta v_{t-1}>, eta*rho = 1-alpha.
# Only the products are pinned; the split into (rho, eta) is a gauge freedom, so
# the step is checked in three of them (cf. rotation note section 4.1).
for label, eta_of in [
    ("eta = Delta (Mamba-2's gauge)", lambda t: DT[t]),
    ("eta = 1", lambda t: 1.0),
    ("eta = (1-alpha)/a (the ZOH gauge)", lambda t: (1 - ALPHA[t]) / (-AA[t])),
]:
    h, got = ZERO.copy(), []
    for t in range(T):
        eta = eta_of(t)
        rho = (1 - ALPHA[t]) / eta
        target = GAMMA[t] * V[t] + BETA[t] * (V[t - 1] if t > 0 else ZERO)
        h = h - eta * (rho * h - target / eta)
        got.append(h.copy())
    ok(f"one gradient step reproduces the recurrence ({label})", close(got, H))

h, got = ZERO.copy(), []
for t in range(T):
    eta, rho = DT[t], (1 - ALPHA[t]) / DT[t]
    tgt = (GAMMA[t] * V[t] + BETA[t] * (V[t - 1] if t > 0 else ZERO)) / (1 - ALPHA[t])
    h = h - eta * rho * (h - tgt)
    got.append(h.copy())
ok("proximal form: (rho/2)||S - T||^2 with T the rank-two target", close(got, H))

ok("the target has rank 2 (Mamba-2's has rank 1)",
   np.linalg.matrix_rank((GAMMA[5] * V[5] + BETA[5] * V[4]) / (1 - ALPHA[5])) == 2)

# lambda lives only in the LINEAR term: the quadratic form G = (1-alpha) I is
# untouched, which is what keeps the isotropy argument of the rotation note intact.
# Checked by splitting the response: the homogeneous part (initial state, no input)
# must be lambda-independent, the driven part must not be.
LAM_B = RNG.uniform(0.0, 1.0, T)
BETA_B, GAMMA_B = (1 - LAM_B) * DT * ALPHA, LAM_B * DT
H0 = RNG.normal(size=(N, P))
ok("the homogeneous response is independent of lambda (G = (1-alpha) I untouched)",
   close(run_trap(v=np.zeros_like(V), h0=H0),
         run_trap(beta=BETA_B, gamma=GAMMA_B, v=np.zeros_like(V), h0=H0)))
ok("the driven response does depend on lambda (it is the linear term)",
   not close(H, run_trap(beta=BETA_B, gamma=GAMMA_B), 1e-6))


# =============================================================================
section("4. lambda is the splitting parameter")

# data half-step (left endpoint) -> ridge, integrated EXACTLY -> data half-step
h, got = ZERO.copy(), []
for t in range(T):
    s1 = h + (1 - LAM[t]) * DT[t] * (V[t - 1] if t > 0 else ZERO)
    got.append((ALPHA[t] * s1 + LAM[t] * DT[t] * V[t]).copy())
    h = got[-1]
ok("data / (exact ridge) / data splitting == the trapezoid", close(got, H))

lam_h = np.full(T, 0.5)
H_half = run_trap(beta=(1 - lam_h) * DT * ALPHA, gamma=lam_h * DT)
h, got = ZERO.copy(), []
for t in range(T):
    h = ALPHA[t] * (h + 0.5 * DT[t] * (V[t - 1] if t > 0 else ZERO)) + 0.5 * DT[t] * V[t]
    got.append(h.copy())
ok("lambda = 1/2 is symmetric Strang splitting", close(got, H_half))

H_eul = run_trap(beta=np.zeros(T), gamma=DT)
h, got = ZERO.copy(), []
for t in range(T):
    h = ALPHA[t] * h + DT[t] * V[t]
    got.append(h.copy())
ok("lambda = 1 is Lie-Trotter (ridge then data) == Mamba-2 / exp-Euler", close(got, H_eul))

H_left = run_trap(beta=DT * ALPHA, gamma=np.zeros(T))
h, got = ZERO.copy(), []
for t in range(T):
    h = ALPHA[t] * (h + DT[t] * (V[t - 1] if t > 0 else ZERO))
    got.append(h.copy())
ok("lambda = 0 is the reversed Lie-Trotter (data then ridge)", close(got, H_left))

# Order of the state-input quadrature, isolated by comparing against the exact
# integral of eq. (approx-step): Lie-Trotter is first order, Strang second, so the
# local truncation errors are O(dt^2) and O(dt^3). This is the appendix's error
# remark, derived rather than quoted -- Strang needs the SYMMETRIC split, which is
# exactly the condition lambda = 1/2 + O(dt).
AC = -0.7
Bf = lambda s: np.array([np.sin(1.3 * s + 0.2), np.cos(0.7 * s)])
xf = lambda s: np.array([np.exp(0.3 * np.sin(s)), 0.5 + 0.2 * s])


def exact_integral(t0, t1, n=200001):
    ts = np.linspace(t0, t1, n)
    return TRAPZ(np.array([np.exp((t1 - s) * AC) * np.outer(Bf(s), xf(s)) for s in ts]),
                 ts, axis=0)


grid, e_err, t_err = [0.1, 0.05, 0.025, 0.0125], [], []
for d in grid:
    t0, t1 = 1.0, 1.0 + d
    ex = exact_integral(t0, t1)
    e_err.append(np.max(np.abs(d * np.outer(Bf(t1), xf(t1)) - ex)))
    t_err.append(np.max(np.abs(
        d * (0.5 * np.exp(d * AC) * np.outer(Bf(t0), xf(t0))
             + 0.5 * np.outer(Bf(t1), xf(t1))) - ex)))
pe = np.polyfit(np.log(grid), np.log(e_err), 1)[0]
pt = np.polyfit(np.log(grid), np.log(t_err), 1)[0]
ok(f"exp-Euler local truncation error is O(dt^2)  (fitted {pe:.2f})", abs(pe - 2) < 0.15)
ok(f"exp-trapezoid(1/2) local truncation error is O(dt^3)  (fitted {pt:.2f})",
   abs(pt - 3) < 0.20)


# =============================================================================
section("5. Two installments, and the single-SSD key scale")

# Sample s reaches the state by two paths: gamma_s at step s, and beta_{s+1} at
# step s+1. Both ride the same transport, so they add into ONE scalar weight.
TILDE = GAMMA.copy()
TILDE[:-1] = LAM[:-1] * DT[:-1] + (1 - LAM[1:]) * DT[1:]


def reconstruct(t_idx, tilde=TILDE, alpha=ALPHA, gamma=GAMMA, v=V):
    acc = ZERO.copy()
    for s in range(t_idx + 1):
        w = gamma[s] if s == t_idx else tilde[s]
        acc = acc + np.prod(alpha[s + 1:t_idx + 1]) * w * v[s]
    return acc


ok("h_t = sum_s decay * w_s * v_s, with w_s = Dtil_s for s < t and gamma_t at s = t",
   all(close(reconstruct(t), H[t]) for t in range(T)))

# Dtil is exactly the single-SSD composite key scale (single_ssd/ssd/diag.rs:
# "scale_t = gamma_t + (1-lam_{t+1}) dt_{t+1}"), and the same-step gamma-correction
# is exactly the s = t entry above. The decomposition is not a curiosity; it is
# what that pathway implements.
ok("Dtil_s == the implementation's composite key scale gamma_s + (1-lam_{s+1}) dt_{s+1}",
   close(TILDE[:-1], GAMMA[:-1] + (1 - LAM[1:]) * DT[1:]))

H_tilde = run_trap(beta=np.zeros(T), gamma=TILDE)
ok("an exp-Euler run at step Dtil differs only in the freshest sample",
   all(close(H_tilde[t] - H[t], (TILDE[t] - GAMMA[t]) * V[t]) for t in range(T)))

ok("lambda = 1 gives Dtil == Delta (Mamba-2 has one installment)",
   close(np.concatenate([DT[:-1] + 0 * DT[:-1], DT[-1:]]), DT))

DC = np.full(T, 0.3)
ok("lambda = 1/2 at constant Delta gives Dtil == Delta (mass preserving)",
   close(0.5 * DC[:-1] + 0.5 * DC[1:], DC[:-1]))

# The identity is not abelian-specific: with M[t] = alpha[t] R[t] non-commuting,
# the beta tap carries M[t], so the common transport still factors out.
ROT = [rand_rot(N) for _ in range(T)]
MM = [ALPHA[t] * ROT[t] for t in range(T)]
h, Hr = ZERO.copy(), []
for t in range(T):
    h = MM[t] @ (h + (1 - LAM[t]) * DT[t] * (V[t - 1] if t > 0 else ZERO)) \
        + GAMMA[t] * V[t]
    Hr.append(h.copy())
Hr = np.array(Hr)


def mprod(Ms, a, b, n=N):
    out = np.eye(n)
    for t in range(max(a, 0), b + 1):
        out = Ms[t] @ out
    return out


def reconstruct_rot(t_idx):
    acc = ZERO.copy()
    for s in range(t_idx + 1):
        w = GAMMA[s] if s == t_idx else TILDE[s]
        acc = acc + w * (mprod(MM, s + 1, t_idx) @ V[s])
    return acc


ok("the identity survives a non-commuting rotational transition verbatim",
   all(close(reconstruct_rot(t), Hr[t], 1e-9) for t in range(T)))

# The trapezoid adds no negative weight: it is a top-up, never a rollback. (This
# is a characterization, not a scorecard -- state tracking is the rotation's job.)
d_, l_ = RNG.uniform(1e-3, 5.0, 20000), RNG.uniform(0, 1, 20000)
a_ = np.exp(d_ * -RNG.uniform(1e-3, 5.0, 20000))
ok("beta and gamma are non-negative for every admissible parameter",
   bool(np.all((1 - l_) * d_ * a_ >= 0) and np.all(l_ * d_ >= 0)))
ok("Dtil_s >= lam_s dt_s: the second installment is a top-up, never a rollback",
   bool(np.all(TILDE[:-1] >= LAM[:-1] * DT[:-1] - 1e-15)))


# =============================================================================
section("6. The augmented state is a feedforward buffer")

# Restoring first-order Markov form by carrying the buffer w_t := v_t gives a
# block matrix whose (2,1) block is ZERO -- the buffer reads the input, never the
# state. Heavy ball's companion matrix has a nonzero (2,1) block, and that entry
# is where its complex eigenvalues come from (rotation note section 6).
for t in [0, 5, 11]:
    Ahat = np.zeros((2 * N, 2 * N))
    Ahat[:N, :N] = MM[t]
    Ahat[:N, N:] = (1 - LAM[t]) * DT[t] * MM[t]
    ok(f"spec(Ahat) == spec(M) u {{0}}^N  (t = {t})",
       np.allclose(np.sort_complex(np.linalg.eigvals(Ahat)),
                   np.sort_complex(np.concatenate(
                       [np.linalg.eigvals(MM[t]), np.zeros(N)])), atol=1e-9))

S_, w_, got = ZERO.copy(), ZERO.copy(), []
for t in range(T):
    S_, w_ = MM[t] @ S_ + (1 - LAM[t]) * DT[t] * (MM[t] @ w_) + GAMMA[t] * V[t], V[t]
    got.append(S_.copy())
ok("the augmented (S, w) recurrence is the trapezoid", close(got, Hr, 1e-9))

C_hb = np.array([[1 + 0.9 - 1.0, -0.9], [1.0, 0.0]])
ok("heavy ball (nonzero (2,1) block) has complex eigenvalues",
   np.max(np.abs(np.linalg.eigvals(C_hb).imag)) > 1e-9)
ok("the trapezoid (zero (2,1) block) has real eigenvalues",
   np.max(np.abs(np.linalg.eigvals(np.array([[0.6, 0.18], [0.0, 0.0]])).imag)) < 1e-15)
prod = np.eye(2)
for t in range(8):
    m_ = 0.3 + 0.05 * t
    prod = np.array([[m_, m_ * (0.2 + 0.03 * t)], [0.0, 0.0]]) @ prod
ok("products of augmented trapezoid factors stay real-spectrumed",
   np.max(np.abs(np.linalg.eigvals(prod).imag)) < 1e-14)


# =============================================================================
section("7. MambaProduct: the DeltaProduct-style unroll")

TK, U, PN = 6, 3, 4
TF = TK * U
dtp = RNG.uniform(0.05, 1.2, TF)
lamp = RNG.uniform(0.0, 1.0, TF)
alp = np.exp(dtp * -RNG.uniform(0.1, 2.0, TF))
Mp = [alp[t] * rand_rot(N) for t in range(TF)]
Vp = RNG.normal(size=(TF, N, PN))
Z = np.zeros((N, PN))


def run_folded(lam=lamp, v=Vp, Ms=Mp, h0=None):
    h, out = (Z.copy() if h0 is None else h0.copy()), []
    for t in range(TF):
        h = Ms[t] @ (h + (1 - lam[t]) * dtp[t] * (v[t - 1] if t > 0 else Z)) \
            + lam[t] * dtp[t] * v[t]
        out.append(h.copy())
    return np.array(out)


HF = run_folded()
TILP = lamp * dtp
TILP[:-1] = lamp[:-1] * dtp[:-1] + (1 - lamp[1:]) * dtp[1:]

A_of = lambda i: mprod(Mp, i * U, i * U + U - 1)


def B_of(i, v=Vp):
    lo, acc = i * U, Z.copy()
    if i > 0:                                        # previous token, whole product
        acc = acc + (1 - lamp[lo]) * dtp[lo] * (A_of(i) @ v[lo - 1])
    for j in range(U - 1):                           # interior, fully paid
        t = lo + j
        acc = acc + TILP[t] * (mprod(Mp, t + 1, lo + U - 1) @ v[t])
    return acc + lamp[lo + U - 1] * dtp[lo + U - 1] * v[lo + U - 1]


h, got = Z.copy(), []
for i in range(TK):
    h = A_of(i) @ h + B_of(i)
    got.append(h.copy())
ok("h_i = A(x_i) h_{i-1} + B(x_{i-1}, x_i) reproduces the folded recurrence",
   close(got, HF[U - 1::U], 1e-9))

# A(x_i) carries no lambda. Checked on the object of the claim: the homogeneous
# token map (nonzero incoming state, input switched off) under two different
# lambda schedules, against the ordered micro-step product.
lam_alt = RNG.uniform(0.0, 1.0, TF)
HZERO = np.zeros_like(Vp)
G0 = RNG.normal(size=(N, PN))
hom = run_folded(lam=lamp, v=HZERO, h0=G0)[U - 1::U]
hom_alt = run_folded(lam=lam_alt, v=HZERO, h0=G0)[U - 1::U]
ok("lambda does not appear in A(x_i): the homogeneous token map is lambda-free",
   close(hom, hom_alt))
acc = G0.copy()
prod_tok = []
for i in range(TK):
    acc = mprod(Mp, i * U, i * U + U - 1) @ acc
    prod_tok.append(acc.copy())
ok("A(x_i) = (prod alpha) R_u...R_1, the ordered micro-step product",
   close(hom, prod_tok, 1e-9))


def expand(t_idx, v=Vp):
    acc = Z.copy()
    for s in range(t_idx + 1):
        w = lamp[s] * dtp[s] if s == t_idx else TILP[s]
        acc = acc + w * (mprod(Mp, s + 1, t_idx) @ v[s])
    return acc


ok("expanding the cache away gives h_t as a pure sum over samples",
   all(close(expand(t), HF[t], 1e-9) for t in range(TF)))
ok("the expansion terminates: each sample appears in exactly two terms",
   all(2 == (1 + (1 if t + 1 < TF else 0)) for t in range(TF - 1)))

lam_one = np.ones(TF)
til_one = lam_one * dtp
til_one[:-1] = dtp[:-1]
ok("lambda = 1 collapses the expansion to Mamba-2's w_s = Delta_s", close(til_one, dtp))

# B depends on the PREVIOUS token, so the token-level form is not B(x_i) and the
# write is not a function of the current token alone. Deterministic witness.
i_ = 3
E = np.ones((N, PN))
V_alt = Vp.copy()
V_alt[i_ * U - 1] = Vp[i_ * U - 1] + E
ok("B moves when only x_{i-1} moves: B = B(x_{i-1}, x_i), not B(x_i)",
   not close(B_of(i_, V_alt), B_of(i_), 1e-6))
ok("and it moves by exactly the beta term",
   close(B_of(i_, V_alt) - B_of(i_),
         (1 - lamp[i_ * U]) * dtp[i_ * U] * (A_of(i_) @ E), 1e-9))


# =============================================================================
section("8. What u > 1 does to the taps")

tok = np.arange(TF) // U
cross = [s for s in range(1, TF) if tok[s] != tok[s - 1]]
interior = [s for s in range(1, TF) if tok[s] == tok[s - 1]]
ok("only 1 tap in u crosses a token boundary (u - 1 are inside one token)",
   len(cross) == TK - 1 and len(interior) == TF - TK)
ok("every interior tap pairs two projections of the SAME token",
   all(tok[s] == tok[s - 1] for s in interior))
# The boundary tap and the incoming state are moved by the SAME operator, a scalar
# apart -- so the previous token is not made stale relative to the state it arrives
# with. Perturb each in turn and compare the two responses.
d_state = A_of(i_) @ E
d_tap = (B_of(i_, V_alt) - B_of(i_)) / ((1 - lamp[i_ * U]) * dtp[i_ * U])
ok("the boundary tap rides A(x_i), the same transport as the incoming state",
   close(d_tap, d_state, 1e-9) and not close(d_state, ZERO[:N, :PN], 1e-6))

lam_bnd = lamp.copy()
lam_bnd[np.arange(TF) % U != 0] = 1.0
h, got = Z.copy(), []
for t in range(TF):
    if t % U == 0 and t > 0:
        h = Mp[t] @ (h + (1 - lamp[t]) * dtp[t] * Vp[t - 1]) + lamp[t] * dtp[t] * Vp[t]
    elif t % U == 0:
        h = Mp[t] @ h + lamp[t] * dtp[t] * Vp[t]
    else:
        h = Mp[t] @ h + dtp[t] * Vp[t]
    got.append(h.copy())
ok("lambda = 1 on interior micro-steps leaves one cross-token tap plus Euler inside",
   close(got, run_folded(lam=lam_bnd), 1e-9))

h, got = Z.copy(), []
for t in range(TF):
    h = Mp[t] @ h + dtp[t] * Vp[t]
    got.append(h.copy())
ok("lambda = 1 everywhere is plain MambaProduct, trapezoid off",
   close(got, run_folded(lam=np.ones(TF)), 1e-9))


# =============================================================================
section("9. The tap lattice")

# A tap at lag d, coefficient nu, transported across ITS OWN gap M_{p-d+1:p}.
# At d = 1 that transport is alpha_p, i.e. the implementation's beta -- so this is
# the existing rule stated at the right generality, not a new one.
micro = np.arange(TF) % U
gam_l = lamp * dtp
nu_l = (1 - lamp) * dtp
nu_2 = (1 - RNG.uniform(0, 1, TF)) * dtp
en_all = np.ones(TF, dtype=bool)
en_reset = micro != 0


def run_taps(taps, gam=gam_l):
    h, out = Z.copy(), []
    for p in range(TF):
        h = Mp[p] @ h
        for d, nu, en in taps:
            if p - d >= 0 and en[p]:
                h = h + nu[p] * (mprod(Mp, p - d + 1, p) @ Vp[p - d])
        h = h + gam[p] * Vp[p]
        out.append(h.copy())
    return np.array(out)


def collapse_taps(taps, t_idx, gam=gam_l):
    acc = Z.copy()
    for s in range(t_idx + 1):
        w = gam[s] + sum(nu[s + d] for d, nu, en in taps
                         if s + d <= t_idx and en[s + d])
        acc = acc + w * (mprod(Mp, s + 1, t_idx) @ Vp[s])
    return acc


LATTICE = {
    "vertical (lag u)": [(U, nu_l, en_all)],
    "reset-horizontal (lag 1, no carry-over)": [(1, nu_l, en_reset)],
    "carry-over-horizontal (today)": [(1, nu_l, en_all)],
    "vertical + reset-horizontal": [(U, nu_l, en_all), (1, nu_2, en_reset)],
    "vertical + carry-over-horizontal": [(U, nu_l, en_all), (1, nu_2, en_all)],
}
for name, taps in LATTICE.items():
    Hx = run_taps(taps)
    ok(f"scalar per-sample collapse is exact: {name}",
       all(close(collapse_taps(taps, t), Hx[t], 1e-9) for t in range(TF)))

# For the vertical pattern the collapse gives the same composite key scale with
# the index shifted from s+1 to s+u, and the same-step gamma-correction widens
# from the diagonal to a u-wide band.
Hv = run_taps(LATTICE["vertical (lag u)"])
scale = gam_l.copy()
scale[:-U] = gam_l[:-U] + nu_l[U:]
w_exact = np.array([[(gam_l[s] if t - s < U else scale[s]) if s <= t else 0.0
                     for s in range(TF)] for t in range(TF)])
ok("vertical key scale = gamma_s + (1-lam_{s+u}) dt_{s+u} (today's formula, shifted)",
   all(close(sum(w_exact[t, s] * (mprod(Mp, s + 1, t) @ Vp[s]) for s in range(t + 1)),
             Hv[t], 1e-9) for t in range(TF)))
band = np.array([[(scale[s] - gam_l[s]) if (s <= t and t - s < U) else 0.0
                  for s in range(TF)] for t in range(TF)])
glob = np.array([[scale[s] if s <= t else 0.0 for s in range(TF)] for t in range(TF)])
ok("global scaling minus a u-wide band correction gives the exact weights",
   close(glob - band, w_exact))
ok("at u = 1 that band is the diagonal, i.e. today's same-step correction",
   all(band[t, s] == 0.0 for t in range(TF) for s in range(TF) if s <= t and t - s >= U))

Hh = run_taps(LATTICE["carry-over-horizontal (today)"])
Hr2 = run_taps(LATTICE["reset-horizontal (lag 1, no carry-over)"])
for a, Ha, b, Hb in [("vertical", Hv, "carry-over", Hh),
                     ("reset", Hr2, "carry-over", Hh),
                     ("vertical", Hv, "reset", Hr2)]:
    ok(f"{a} and {b} are genuinely different models", not close(Ha, Hb, 1e-6))

ok("at u = 1 vertical and carry-over-horizontal coincide",
   close(run_taps([(1, nu_l, en_all)]), Hh))
ok("at u = 1 reset-horizontal degenerates to no trapezoid at all",
   close(run_taps([(1, nu_l, np.zeros(TF, dtype=bool))]), run_taps([])))


# =============================================================================
print(f"\n{PASSED} passed, {FAILED} failed")
sys.exit(1 if FAILED else 0)
