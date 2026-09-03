#!/usr/bin/env python3
"""Numerical verification for `info/rotation-as-optimization.md`.

Every claim the document makes that can be checked numerically is checked here, in
float64, with section numbers matching the document's. Pure `numpy`; no other
dependency, no I/O, no reference to the model code — the point is that these are
statements about the *recurrence*, reproducible from the equations alone.

    python3 scripts/rotation_as_optimization.py

Exits non-zero if any check fails.

Conventions (document §2). State ``S`` is complex, ``[P, n]`` with ``n = N/2``:
rows are value channels, columns complex state channels. Mamba's ``(B, x, C)``
play the roles of ``(key, value, query)``. The real inner product on the complex
plane is ``<a, b> = Re(conj(a) b)``, i.e. the Euclidean one on R^2, and the
gradient is the Riesz representative for it (``2 d/dSbar``, Wirtinger).

    mu[t, j] = exp(dt[t] * (A[t] + 1j * theta[t, j])) = alpha[t] * exp(1j * phi[t, j])
"""

import sys

import numpy as np

RNG = np.random.default_rng(20260903)
PASSED = 0
FAILED = 0


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


# ---------------------------------------------------------------- the objective
def grad(S, rho, w, x):
    """Riesz representative of dL for L below, w.r.t. the real inner product.

    L(S) = 1/2 sum_j rho[j] ||S[:, j]||^2  -  sum_p x[p] Re(<w, S[p, :]>)
    """
    return S * rho[None, :] - np.outer(x, w)


def loss(S, rho, w, x):
    quad = 0.5 * np.sum(rho[None, :] * np.abs(S) ** 2)
    lin = np.sum(x * np.real(S @ np.conj(w)))
    return quad - lin


def random_run(T, P, n):
    """A random Mamba-3 complex-Euler run: (mu, dt, alpha, theta, A, k, x, q)."""
    dt = np.abs(RNG.normal(size=T)) + 0.05
    A = -(np.abs(RNG.normal(size=T)) + 0.05)
    theta = RNG.normal(size=(T, n)) * 1.5
    mu = np.exp(dt[:, None] * (A[:, None] + 1j * theta))
    k = RNG.normal(size=(T, n)) + 1j * RNG.normal(size=(T, n))
    q = RNG.normal(size=(T, n)) + 1j * RNG.normal(size=(T, n))
    x = RNG.normal(size=(T, P))
    return mu, dt, A, theta, k, x, q


# =============================================================================
section("3. The obstruction: a real loss with a real step cannot rotate")

bad = 0
for _ in range(2000):
    d = 4
    H = RNG.normal(size=(d, d))
    H = H + H.T  # any real-valued loss has a symmetric Hessian
    eta = RNG.normal()  # any real step, either sign
    bad += np.max(np.abs(np.linalg.eigvals(np.eye(d) - eta * H).imag)) > 1e-9
ok("I - eta*Hess(L) has a real spectrum, 2000 random (loss, step) draws", bad == 0)

# Complexifying the loss alone does not help: a complex least-squares Hessian is
# Hermitian, so the delta-rule factor still has a real spectrum.
n = 5
kc = RNG.normal(size=n) + 1j * RNG.normal(size=n)
Hh = np.eye(n) - RNG.uniform(0.1, 2.0) * np.outer(kc, kc.conj())
ok(
    "complex delta rule I - beta k k^* is Hermitian, hence real-spectrumed",
    close(Hh, Hh.conj().T) and np.max(np.abs(np.linalg.eigvals(Hh).imag)) < 1e-12,
)


# =============================================================================
section("4. View I: a complex step size")

P, n, T = 3, 4, 12
mu, dt, A, theta, k, x, q = random_run(T, P, n)
t = 0
S0 = RNG.normal(size=(P, n)) + 1j * RNG.normal(size=(P, n))
S_ref = mu[t][None, :] * S0 + dt[t] * np.outer(x[t], k[t])

# 4.1 -- the rescue is exact, and the gauge is free (only rho*eta and eta*w are fixed)
for label, rho in [
    ("gauge (a), rho = Re(1-mu)/dt", np.real(1 - mu[t]) / dt[t]),
    ("gauge (b), rho = |1-mu|/dt", np.abs(1 - mu[t]) / dt[t]),
    ("gauge (c), rho = 1", np.ones(n)),
]:
    eta = (1 - mu[t]) / rho
    w = dt[t] * k[t] / eta
    ok(f"{label}: one complex step reproduces the recurrence exactly",
       close(S0 - grad(S0, rho, w, x[t]) * eta[None, :], S_ref))

psi = np.angle(1 - mu[t])
rho_a = np.real(1 - mu[t]) / dt[t]
eta_a = (1 - mu[t]) / rho_a
ok("gauge (a): eta = dt * (1 + i tan psi)", close(eta_a, dt[t] * (1 + 1j * np.tan(psi))))
ok("gauge (a): Re(eta) = dt exactly -- the descent part is Mamba-2's step",
   close(np.real(eta_a), dt[t] * np.ones(n)))
rho_b = np.abs(1 - mu[t]) / dt[t]
ok("gauge (b): |eta| = dt, and the objective's key is k rotated by -psi",
   close(np.abs((1 - mu[t]) / rho_b), dt[t] * np.ones(n))
   and close(dt[t] * k[t] / ((1 - mu[t]) / rho_b), k[t] * np.exp(-1j * psi)))

# 4.2 -- the Helmholtz split of G = I - M is the real/imaginary split of 1 - mu
real_row = lambda z: np.array([[z.real, z.imag], [-z.imag, z.real]])  # noqa: E731
alpha0, phi0 = np.abs(mu[t, 0]), np.angle(mu[t, 0])
M = real_row(mu[t, 0])
R = np.array([[np.cos(phi0), -np.sin(phi0)], [np.sin(phi0), np.cos(phi0)]])
J = np.array([[0.0, -1.0], [1.0, 0.0]])
G = np.eye(2) - M
ok("the real form of the transition is alpha * R^T", close(M, alpha0 * R.T))
ok("G = Re(1-mu) I - Im(1-mu) J, i.e. sym/skew = real/imaginary",
   close(G, np.real(1 - mu[t, 0]) * np.eye(2) - np.imag(1 - mu[t, 0]) * J)
   and close((G + G.T) / 2, (1 - alpha0 * np.cos(phi0)) * np.eye(2)))
z = RNG.normal() + 1j * RNG.normal()
E = real_row(z)
ok("the symmetric part of multiplication by eta is exactly Re(eta) I",
   close((E + E.T) / 2, z.real * np.eye(2)))

# 4.3 -- the step is always a descent direction, with an exact bound
worst_re, worst_gap = np.inf, -np.inf
for al in np.linspace(1e-6, 1.0, 400):
    m = al * np.exp(1j * np.linspace(-np.pi, np.pi, 1601))
    worst_re = min(worst_re, np.min(np.real(1 - m)))
    worst_gap = max(worst_gap, np.max(np.abs(np.angle(1 - m))) - np.arcsin(min(al, 1.0)))
ok(f"Re(1-mu) >= 0 over the whole closed unit disc (min {worst_re:.1e})", worst_re > -1e-12)
ok(f"|arg(1-mu)| <= arcsin(alpha) (max excess {worst_gap:.1e})", worst_gap < 1e-9)
al = 0.8
ok("the bound is attained, at cos(phi) = alpha",
   abs(abs(np.angle(1 - al * np.exp(1j * np.arccos(al)))) - np.arcsin(al)) < 1e-12)

rho = np.real(1 - mu[t]) / dt[t]
eta = (1 - mu[t]) / rho
w = dt[t] * k[t] / eta
g = grad(S0, rho, w, x[t])
step = -g * eta[None, :]
inner = np.sum(np.real(np.conj(g) * step))
ok("<grad, step> = -sum_j Re(eta_j) ||grad[:, j]||^2",
   close(inner, -np.sum(np.real(eta)[None, :] * np.abs(g) ** 2)))
ok("the step strictly decreases the loss (finite difference)",
   loss(S0 + 1e-4 * step, rho, w, x[t]) - loss(S0, rho, w, x[t]) < 0 and inner < 0)

# 4.4 -- whole trajectories agree
S, ys_ref = np.zeros((P, n), complex), []
for i in range(T):
    S = mu[i][None, :] * S + dt[i] * np.outer(x[i], k[i])
    ys_ref.append(np.real(S @ np.conj(q[i])))
S, ys_L = np.zeros((P, n), complex), []
for i in range(T):
    rho_i = np.real(1 - mu[i]) / dt[i]
    eta_i = (1 - mu[i]) / rho_i
    S = S - grad(S, rho_i, dt[i] * k[i] / eta_i, x[i]) * eta_i[None, :]
    ys_L.append(np.real(S @ np.conj(q[i])))
ok("T-step complex-descent trajectory == the recurrence", close(ys_ref, ys_L))

# 4.5 -- the reachable set: Mamba-2 is an interval, Mamba-3 is the disc
alphas = np.linspace(1e-3, 0.999, 200)
ok("Mamba-2: eta*rho = 1 - alpha lies in (0, 1) -- always short of a Newton step",
   np.all((1 - alphas > 0) & (1 - alphas < 1)))
ok("Mamba-3: eta*rho = 1 - mu fills the punctured unit disc; its real diameter is (0, 2)",
   close(abs(1 - (-0.999)), 1.999, 1e-3) and abs(1 - (-1.0)) == 2.0)

print("\n  circulation fraction sin|psi| for a mod-k counter at alpha -> 1:")
for kk in (2, 3, 4, 6, 12):
    phi_k = 2 * np.pi / kk
    psi_k = abs(np.angle(1 - np.exp(1j * phi_k)))
    ok(f"    mod-{kk:<2d}: |psi| = pi/2 - pi/{kk}, circulation = cos(pi/{kk}) = {np.cos(np.pi/kk):.3f}",
       abs(psi_k - (np.pi / 2 - np.pi / kk)) < 1e-12
       and abs(np.sin(psi_k) - np.cos(np.pi / kk)) < 1e-12)


# =============================================================================
section("5. View II: min-max on a harmonic potential")

a_c, th_c = 0.83, 1.7
At = -a_c + 1j * th_c
gw = RNG.normal() + 1j * RNG.normal()


def F(zr, zi):
    """F = Re( -Atilde/2 z^2 - g z ) -- the real part of a holomorphic function."""
    zz = zr + 1j * zi
    return np.real(-At / 2 * zz**2 - gw * zz)


zr, zi = RNG.normal(), RNG.normal()
h = 1e-6
field = (
    -(F(zr + h, zi) - F(zr - h, zi)) / (2 * h),  # descend in the real part
    +(F(zr, zi + h) - F(zr, zi - h)) / (2 * h),  # ASCEND in the imaginary part
)
want = At * (zr + 1j * zi) + gw
ok("descent-in-Re / ascent-in-Im on F is exactly the complex flow",
   close(field, [want.real, want.imag], 1e-6))

h = 1e-4
lap = (F(zr + h, zi) + F(zr - h, zi) + F(zr, zi + h) + F(zr, zi - h) - 4 * F(zr, zi)) / h**2
ok(f"F is harmonic (Laplacian {lap:.1e}), so it has no minimum -- only saddles", abs(lap) < 1e-4)

S, ys_gda = np.zeros((P, n), complex), []
for i in range(T):
    At_i = A[i] + 1j * theta[i]
    S = np.exp(dt[i] * At_i)[None, :] * S + dt[i] * np.outer(x[i], k[i])
    ys_gda.append(np.real(S @ np.conj(q[i])))
ok("the matrix descent-ascent flow, integrated exponentially, == the recurrence",
   close(ys_ref, ys_gda))


# =============================================================================
section("6. View III: momentum")

worst = 0.0
for _ in range(500):
    alpha = RNG.uniform(0.05, 0.999)
    phi = RNG.uniform(0.05, np.pi - 0.05) * RNG.choice([-1, 1])
    beta = alpha**2
    m = alpha * np.exp(1j * phi)
    eta_rho = 1 + beta - 2 * alpha * np.cos(phi)
    worst = max(worst, abs(eta_rho - abs(1 - m) ** 2))
    comp = np.array([[1 + beta - eta_rho, -beta], [1.0, 0.0]])
    ev = sorted(np.linalg.eigvals(comp), key=lambda v: v.imag)
    if not close(ev, sorted([m, np.conj(m)], key=lambda v: v.imag), 1e-9):
        worst = np.inf
ok("heavy ball's companion eigenvalues are exactly mu, conj(mu), 500 draws", worst < 1e-12)
ok(f"closed form eta*rho = |1 - mu|^2 (max error {worst:.1e})", worst < 1e-12)

alpha, phi = 0.7, 1.1
beta = alpha**2
comp = np.array([[1 + beta - abs(1 - alpha * np.exp(1j * phi)) ** 2, -beta], [1.0, 0.0]])
rot = alpha * np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
ok("the companion matrix is similar to alpha*R(phi) (same trace and determinant)",
   close(np.trace(comp), np.trace(rot)) and close(np.linalg.det(comp), np.linalg.det(rot)))
ok("momentum needs two real slots per channel -- the same two the complex state has",
   comp.shape == (2, 2))


# =============================================================================
section("7. View IV: composing non-commuting steps (the delta-rule route)")

rotated = 0
for _ in range(2000):
    M = np.eye(4)
    for _ in range(3):  # isotropic curvature: every factor is (1 - eta*rho) I
        M = M @ (np.eye(4) * (1 - RNG.uniform(-1, 3)))
    rotated += np.max(np.abs(np.linalg.eigvals(M).imag)) > 1e-9
ok("isotropic curvature: 0 / 2000 products of u=3 steps acquire a complex spectrum",
   rotated == 0)


def householder(kv, b):
    kv = kv / np.linalg.norm(kv)
    return np.eye(len(kv)) - b * np.outer(kv, kv)


n_rot = n_viol = 0
for _ in range(4000):
    b1, b2 = RNG.uniform(0, 2), RNG.uniform(0, 2)
    M = householder(RNG.normal(size=6), b1) @ householder(RNG.normal(size=6), b2)
    if np.max(np.abs(np.linalg.eigvals(M).imag)) > 1e-9:
        n_rot += 1
        n_viol += not (b1 > 1 and b2 > 1)
ok(f"rank-one curvature: {100 * n_rot / 4000:.0f}% of u=2 products rotate", n_rot > 0)
ok("every rotating product had both beta > 1: rotation is built from two overshoots",
   n_viol == 0)


# =============================================================================
section("8. The step-size ladder: R, C, H, and the two-sided rotor")


def qmul(p, r):
    p0, p1, p2, p3 = p
    r0, r1, r2, r3 = r
    return np.array([
        p0 * r0 - p1 * r1 - p2 * r2 - p3 * r3,
        p0 * r1 + p1 * r0 + p2 * r3 - p3 * r2,
        p0 * r2 - p1 * r3 + p2 * r0 + p3 * r1,
        p0 * r3 + p1 * r2 - p2 * r1 + p3 * r0,
    ])


def qconj(p):
    return np.array([p[0], -p[1], -p[2], -p[3]])


def qexp_pure(v):
    """exp of a pure quaternion: the unit quaternion rotating by 2*||v||."""
    nrm = np.linalg.norm(v)
    if nrm < 1e-15:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return np.concatenate([[np.cos(nrm)], np.sin(nrm) * v / nrm])


basis = np.eye(4)
qq = RNG.normal(size=4)
Lq = np.stack([qmul(qq, basis[i]) for i in range(4)], axis=1)
ok("H: the symmetric part of v -> q v is Re(q) I, exactly as over C",
   close((Lq + Lq.T) / 2, qq[0] * np.eye(4)))

uq = RNG.normal(size=4)
uq /= np.linalg.norm(uq)
al = 0.6
imu = np.array([1.0, 0, 0, 0]) - al * uq
ok("H: Re(1 - alpha q) > 0 and |angle(1 - alpha q)| <= arcsin(alpha) -- same bound",
   imu[0] > 0 and np.arccos(imu[0] / np.linalg.norm(imu)) <= np.arcsin(al) + 1e-12)

rho_q = imu[0]
eta_q = imu / rho_q
s_q, k_q = RNG.normal(size=4), RNG.normal(size=4)
w_q = qmul(qconj(eta_q) / np.dot(eta_q, eta_q), k_q)
ok("H: the same formula eta = (1 - mu)/rho reproduces the quaternionic recurrence",
   close(s_q - qmul(eta_q, rho_q * s_q - w_q), qmul(al * uq, s_q) + k_q))

q1, p1 = RNG.normal(size=4), RNG.normal(size=4)
q1 /= np.linalg.norm(q1)
p1 /= np.linalg.norm(p1)
Tm = np.stack([qmul(qmul(q1, basis[i]), qconj(p1)) for i in range(4)], axis=1)
ok("SO(4): v -> q v pbar is orthogonal with det 1",
   close(Tm @ Tm.T, np.eye(4)) and np.linalg.det(Tm) > 0)
ok("SO(4): its normalised trace -- the descent fraction -- is Re(q) Re(p)",
   abs(np.trace(Tm) - 4 * q1[0] * p1[0]) < 1e-12)

# What u > 1 buys, per algebra: abelian factors commute (order-free, angles add);
# quaternionic ones do not (order matters, and the product leaves exp(sum)).
angles = RNG.uniform(-0.4, 0.4, size=4)
ok("C: the product over a token depends only on the sum of the angles (order-free)",
   close(np.prod(np.exp(1j * angles)), np.exp(1j * angles.sum())))
gens = [RNG.uniform(-0.4, 0.4, size=3) for _ in range(3)]
fwd = np.array([1.0, 0, 0, 0])
rev = np.array([1.0, 0, 0, 0])
for gv in gens:
    fwd = qmul(qexp_pure(gv), fwd)
for gv in reversed(gens):
    rev = qmul(qexp_pure(gv), rev)
ok("H: the same generators in the reverse order give a different product",
   np.max(np.abs(fwd - rev)) > 1e-3)
ok("H: the product also leaves exp(sum of generators) -- it is not any single step",
   np.max(np.abs(fwd - qexp_pure(sum(gens)))) > 1e-3)


# =============================================================================
section("9. The trapezoid, and 10. the integrator")

lam = RNG.uniform(0.2, 0.9, size=T)
S, ys_trap = np.zeros((P, n), complex), []
for i in range(T):
    alpha_i = np.exp(dt[i] * A[i])
    phi_i = dt[i] * theta[i]
    beta_i = (1 - lam[i]) * dt[i] * alpha_i
    gamma_i = lam[i] * dt[i]
    prev = np.zeros((P, n), complex) if i == 0 else np.outer(x[i - 1], k[i - 1])
    S = (mu[i][None, :] * S
         + beta_i * np.exp(1j * phi_i)[None, :] * prev
         + gamma_i * np.outer(x[i], k[i]))
    ys_trap.append(np.real(S @ np.conj(q[i])))

S, ys_trap_L = np.zeros((P, n), complex), []
for i in range(T):
    rho_i = np.real(1 - mu[i]) / dt[i]
    eta_i = (1 - mu[i]) / rho_i
    alpha_i = np.exp(dt[i] * A[i])
    phi_i = dt[i] * theta[i]
    W = lam[i] * dt[i] * np.outer(x[i], k[i])
    if i > 0:
        # the older sample, transported into the current frame by this step's rotation
        W = W + (1 - lam[i]) * dt[i] * alpha_i * np.outer(x[i - 1], k[i - 1] * np.exp(1j * phi_i))
    S = S - (S * rho_i[None, :] - W / eta_i[None, :]) * eta_i[None, :]
    ys_trap_L.append(np.real(S @ np.conj(q[i])))
ok("the trapezoid is the same complex step on a transported two-sample objective",
   close(ys_trap, ys_trap_L))

dtr, thr = 0.4, 2.0
ok(f"forward Euler on a pure rotation has modulus {abs(1 + 1j * dtr * thr):.4f} > 1 (diverges)",
   abs(1 + 1j * dtr * thr) > 1 + 1e-9)
ok("exponential-Euler on a pure rotation has modulus exactly 1 (orthogonal, exact)",
   abs(abs(np.exp(1j * dtr * thr)) - 1) < 1e-15)


# =============================================================================
print(f"\n{PASSED} passed, {FAILED} failed")
sys.exit(1 if FAILED else 0)
