#!/usr/bin/env python3
"""Numerical verification for `info/mimo-as-batch.md`.

Every claim the document makes that can be checked numerically is checked here, in
float64, with section numbers matching the document's. Pure `numpy`; no other
dependency, no I/O, no reference to the model code — the point is that these are
statements about the *recurrence*, reproducible from the equations alone.

    python3 scripts/mimo_as_batch.py

Exits non-zero if any check fails.

Conventions (document §2). The state is kept in the fast-weight orientation
``S``, shape ``[P, N]`` (rows indexed by value channel, columns by state channel),
as `info/rotation-as-optimization.md` does; `info/trapezoid-as-integration.md`
uses the transposed ``h``. Nothing below depends on which. One head throughout,
and the rotation is carried in the gauge the implementation runs — transition the
plain scalar ``alpha``, cumulative rotation absorbed into ``B``/``C`` — so every
quantity here is real.

    alpha[t] = exp(dt[t] * A[t])                        decay, A[t] < 0
    beta[t]  = (1 - lam[t]) * dt[t] * alpha[t]          left-endpoint weight
    gamma[t] = lam[t] * dt[t]                           right-endpoint weight
    v[t, m]  = mimo_x[m] * x[t]                         the M tied values
    B[t, m]                                             the M free keys
    W[t]     = sum_m outer(v[t, m], B[t, m])            the rank-M write
    S[t]     = alpha[t] * S[t-1] + beta[t] * W[t-1] + gamma[t] * W[t]
"""

import sys

import numpy as np

RNG = np.random.default_rng(20260904)
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


def block_rot(angles, n):
    """Block-diagonal SO(2)^{n/2} rotation from `n // 2` plane angles."""
    r = np.zeros((n, n))
    for j, a in enumerate(angles):
        c, s = np.cos(a), np.sin(a)
        r[2 * j:2 * j + 2, 2 * j:2 * j + 2] = [[c, -s], [s, c]]
    return r


def jac_rank(f, theta, eps=1e-6):
    """Dimension of the image manifold of `f` at `theta`, by finite differences."""
    base = f(theta).ravel()
    jac = np.zeros((base.size, theta.size))
    for i in range(theta.size):
        pert = theta.copy()
        pert[i] += eps
        jac[:, i] = (f(pert) - base.reshape(f(theta).shape)).ravel() / eps
    return np.linalg.matrix_rank(jac, tol=1e-4)


# =============================================================================
# Shared trajectory: one head, M ranks, trapezoid on, rotation on (gauged frame).
# =============================================================================
N, P, M, T = 8, 8, 3, 9
NPAIR = N // 2

dt = RNG.uniform(0.05, 1.5, T)
a_coef = -RNG.uniform(0.1, 2.0, T)
alpha = np.exp(dt * a_coef)
lam = RNG.uniform(0.05, 0.95, T)
beta = (1.0 - lam) * dt * alpha
gamma = lam * dt

step_angle = RNG.uniform(-1.0, 1.0, (T, NPAIR)) * dt[:, None]
cum_angle = np.cumsum(step_angle, axis=0)
RBAR = np.stack([block_rot(cum_angle[t], N) for t in range(T)])

B_raw = RNG.normal(size=(T, M, N))
C_raw = RNG.normal(size=(T, M, N))
B = np.einsum("tij,tmj->tmi", RBAR, B_raw)      # rotated keys, as the kernel sees them
C = np.einsum("tij,tmj->tmi", RBAR, C_raw)
x = RNG.normal(size=(T, P))
mimo_x = RNG.normal(size=(M, P))
v = np.einsum("tp,mp->tmp", x, mimo_x)          # the M tied values
D_SKIP = RNG.normal(size=P)
z = RNG.normal(size=(T, P))

ZERO = np.zeros((P, N))


def write(t, val=None, key=None):
    """Rank-M write of step `t` (empty before the sequence starts)."""
    if t < 0:
        return ZERO
    return np.einsum("mp,mn->pn", v[t] if val is None else val[t],
                     B[t] if key is None else key[t])


def run(s0=None, val=None, key=None):
    """The trajectory; returns S[0..T] with S[0] = s0."""
    s = ZERO.copy() if s0 is None else s0.copy()
    out = [s.copy()]
    for t in range(T):
        s = (alpha[t] * s + beta[t] * write(t - 1, val, key)
             + gamma[t] * write(t, val, key))
        out.append(s.copy())
    return np.stack(out)


S0 = RNG.normal(size=(P, N))
S = run(S0)


# =============================================================================
section("3. The objective — a rank-M target, and only the linear term moves")
# =============================================================================
# L_t(S) = (rho/2)||S||_F^2 - (1/eta)( gamma <S B_t, V_t>_F + beta <S B_{t-1}, V_{t-1}>_F )
# with eta * rho = 1 - alpha.  Only the product is pinned; the split is a gauge.
for label, mult in [("eta = Delta", 1.0), ("eta = Delta/2", 0.5), ("eta = 3.7 Delta", 3.7)]:
    err = 0.0
    for t in range(T):
        eta = mult * dt[t]
        rho = (1.0 - alpha[t]) / eta
        grad = rho * S[t] - (gamma[t] * write(t) + beta[t] * write(t - 1)) / eta
        err = max(err, np.max(np.abs(S[t] - eta * grad - S[t + 1])))
    ok(f"one gradient step reproduces the MIMO recurrence, gauge {label}", err < 1e-11)

err_prox = 0.0
ranks_trap, ranks_euler = [], []
for t in range(1, T):
    eta = dt[t]
    rho = (1.0 - alpha[t]) / eta
    probe = RNG.normal(size=(P, N))
    target = (gamma[t] * write(t) + beta[t] * write(t - 1)) / (1.0 - alpha[t])
    expanded = (rho / 2 * np.sum(probe ** 2)
                - (gamma[t] * np.sum(probe * write(t))
                   + beta[t] * np.sum(probe * write(t - 1))) / eta)
    proximal = rho / 2 * np.sum((probe - target) ** 2) - rho / 2 * np.sum(target ** 2)
    err_prox = max(err_prox, abs(expanded - proximal))
    ranks_trap.append(np.linalg.matrix_rank(target, tol=1e-9))
    ranks_euler.append(np.linalg.matrix_rank(gamma[t] * write(t), tol=1e-9))
ok("proximal form equals the expanded form up to a constant", err_prox < 1e-9)
ok(f"target rank is min(P, N, M) = {min(P, N, M)} without the trapezoid",
   all(r == min(P, N, M) for r in ranks_euler))
ok(f"target rank is min(P, N, 2M) = {min(P, N, 2 * M)} with it",
   all(r == min(P, N, 2 * M) for r in ranks_trap))

# The minibatch decomposition: M per-sample objectives, each carrying rho/M of the ridge.
err_mb = 0.0
for t in range(T):
    eta = dt[t]
    rho = (1.0 - alpha[t]) / eta
    probe = RNG.normal(size=(P, N))
    per_sample = 0.0
    for m in range(M):
        lin = gamma[t] * np.sum(probe * np.outer(v[t, m], B[t, m]))
        if t > 0:
            lin += beta[t] * np.sum(probe * np.outer(v[t - 1, m], B[t - 1, m]))
        per_sample += rho / (2 * M) * np.sum(probe ** 2) - lin / eta
    joint = (rho / 2 * np.sum(probe ** 2)
             - (gamma[t] * np.sum(probe * write(t))
                + beta[t] * np.sum(probe * write(t - 1))) / eta)
    err_mb = max(err_mb, abs(per_sample - joint))
ok("L = sum_m L^(m), each carrying rho/M of the ridge (a minibatch of M)", err_mb < 1e-9)

# G is M-free.  Checked on the object of the claim: the state-to-state map, i.e.
# how a perturbation of S propagates, at any M and any MIMO parameters.
probe_e = RNG.normal(size=(P, N))
step_maps = []
for m_alt in (1, 2, 3, 5):
    v_alt = np.einsum("tp,mp->tmp", x, RNG.normal(size=(m_alt, P)))
    b_alt = RNG.normal(size=(T, m_alt, N))

    def run_alt(s0, v_a=v_alt, b_a=b_alt):
        s = s0.copy()
        for t in range(T):
            w = lambda u: (np.einsum("mp,mn->pn", v_a[u], b_a[u]) if u >= 0 else ZERO)
            s = alpha[t] * s + beta[t] * w(t - 1) + gamma[t] * w(t)
        return s

    step_maps.append(run_alt(S0 + probe_e) - run_alt(S0))
ok("the state-to-state map is prod(alpha) * I at every M: G = (1-alpha) I, M-free",
   all(close(sm, np.prod(alpha) * probe_e) for sm in step_maps))


# =============================================================================
section("4. The batch — M free keys and one value, replicated through fixed masks")
# =============================================================================
ok("the M values are M fixed diagonal images of one vector",
   close(v[4], np.stack([mimo_x[m] * x[4] for m in range(M)])))
ok("so they span M dimensions generically, from only P degrees of freedom",
   np.linalg.matrix_rank(v[4], tol=1e-9) == min(M, P))

# The tying cost, measured as the dimension of the reachable rank-M write manifold.
print("   write-manifold dimensions (Jacobian rank of the parameterisation):")
print(f"   {'P':>3}{'N':>4}{'M':>3} | {'MIMO(M)':>8}{'u=M steps':>11}{'free rank-M':>13}"
      f" | {'deficit':>8}{'(M-1)P-M^2+1':>14}")
for (p, n, m) in [(8, 8, 3), (10, 8, 3), (8, 8, 2), (12, 10, 4), (6, 8, 2), (9, 7, 5)]:
    masks = [np.diag(RNG.normal(size=p)) for _ in range(m)]

    def f_mimo(th, p=p, n=n, m=m, masks=masks):
        return sum(np.outer(masks[j] @ th[:p], th[p:].reshape(m, n)[j]) for j in range(m))

    def f_steps(th, p=p, n=n, m=m, masks=masks):
        vals = th[:m * p].reshape(m, p)
        return sum(np.outer(masks[0] @ vals[j], th[m * p:].reshape(m, n)[j]) for j in range(m))

    def f_free(th, p=p, n=n, m=m):
        vals = th[:m * p].reshape(m, p)
        return sum(np.outer(vals[j], th[m * p:].reshape(m, n)[j]) for j in range(m))

    d_mimo = jac_rank(f_mimo, RNG.normal(size=p + m * n))
    d_steps = jac_rank(f_steps, RNG.normal(size=m * p + m * n))
    d_free = jac_rank(f_free, RNG.normal(size=m * p + m * n))
    print(f"   {p:>3}{n:>4}{m:>3} | {d_mimo:>8}{d_steps:>11}{d_free:>13}"
          f" | {d_free - d_mimo:>8}{(m - 1) * p - m * m + 1:>14}")
    ok(f"P={p} N={n} M={m}: tying costs exactly (M-1)P - M^2 + 1 dimensions",
       d_free - d_mimo == (m - 1) * p - m * m + 1)
    ok(f"P={p} N={n} M={m}: u = M free values reach the whole rank-M manifold",
       d_steps == d_free)


# =============================================================================
section("5. Why the ranks decompose — isotropy, not a property of MIMO")
# =============================================================================
sub = np.zeros((P, N))
for m in range(M):
    s = S0 / M
    for t in range(T):
        prev = np.outer(v[t - 1, m], B[t - 1, m]) if t > 0 else ZERO
        s = alpha[t] * s + beta[t] * prev + gamma[t] * np.outer(v[t, m], B[t, m])
    sub += s
ok("MIMO = a sum of M standalone SISO runs, each driven by its own (B^(m), x^(m))",
   close(sub, S[-1], 1e-11))

err_r2 = 0.0
per_rank = []
for m in range(M):
    s = S0 / M
    traj = [s.copy()]
    for t in range(T):
        prev = np.outer(v[t - 1, m], B[t - 1, m]) if t > 0 else ZERO
        s = alpha[t] * s + beta[t] * prev + gamma[t] * np.outer(v[t, m], B[t, m])
        traj.append(s.copy())
    per_rank.append(np.stack(traj))
for t in range(T):
    for i in range(M):
        direct = S[t + 1] @ C[t, i]
        summed = sum(per_rank[j][t + 1] @ C[t, i] for j in range(M))
        err_r2 = max(err_r2, np.max(np.abs(direct - summed)))
ok("hence y^(i) = sum_j SSM(alpha, Delta, B^(j), C^(i), x^(j)): the R^2-SISO identity",
   err_r2 < 1e-11)

# The premise is that the transition carries no sample. Both delta-rule shapes fail it.
K = RNG.normal(size=(T, M, N))
K /= np.linalg.norm(K, axis=2, keepdims=True)
bta = RNG.uniform(0.2, 1.5, (T, M))


def delta_sequential():
    s = np.zeros((P, N))
    for t in range(T):
        for m in range(M):
            s = (s @ (np.eye(N) - bta[t, m] * np.outer(K[t, m], K[t, m]))
                 + bta[t, m] * np.outer(v[t, m], K[t, m]))
    return s


def delta_sequential_standalone():
    tot = np.zeros((P, N))
    for m in range(M):
        s = np.zeros((P, N))
        for t in range(T):
            s = (s @ (np.eye(N) - bta[t, m] * np.outer(K[t, m], K[t, m]))
                 + bta[t, m] * np.outer(v[t, m], K[t, m]))
        tot += s
    return tot


def block_reflector(eta=0.7):
    s = np.zeros((P, N))
    for t in range(T):
        kt = K[t].T
        gram_inv = np.linalg.inv(kt.T @ kt)
        s = s @ (np.eye(N) - eta * kt @ gram_inv @ kt.T) + eta * v[t].T @ gram_inv @ kt.T
    return s


def block_reflector_standalone(eta=0.7):
    tot = np.zeros((P, N))
    for m in range(M):
        s = np.zeros((P, N))
        for t in range(T):
            kt = K[t, m:m + 1].T
            gram_inv = np.linalg.inv(kt.T @ kt)
            s = s @ (np.eye(N) - eta * kt @ gram_inv @ kt.T) + eta * v[t, m:m + 1].T @ gram_inv @ kt.T
        tot += s
    return tot


ok("a sequential rank-M delta rule does NOT decompose (its transition holds every key)",
   not close(delta_sequential(), delta_sequential_standalone(), 1e-6))
ok("nor does the jointly-solved (block-reflector) one, for the same reason",
   not close(block_reflector(), block_reflector_standalone(), 1e-6))

perm = RNG.permutation(M)
ok("MIMO's write is permutation-invariant in the rank index",
   close(run(S0, val=v[:, perm], key=B[:, perm])[-1], S[-1], 1e-11))
prod_fwd = np.linalg.multi_dot([np.eye(N) - bta[0, j] * np.outer(K[0, j], K[0, j])
                                for j in range(M)])
prod_rev = np.linalg.multi_dot([np.eye(N) - bta[0, j] * np.outer(K[0, j], K[0, j])
                                for j in reversed(range(M))])
ok("a product of rank-one erases is not: the delta family must sequence, MIMO sums",
   not close(prod_fwd, prod_rev, 1e-6))


# =============================================================================
section("6. Initialisation — a MIMO block starts as its SISO block")
# =============================================================================
mx0 = np.full((M, P), 1.0 / M)
mo0 = np.full((M, P), 1.0 / M)
mz0 = np.ones((M, P))
v0 = np.einsum("tp,mp->tmp", x, mx0)
silu = lambda u: u / (1.0 + np.exp(-u))


def run_mimo_block():
    s = np.zeros((P, N))
    outs = []
    for t in range(T):
        prev = np.einsum("mp,mn->pn", v0[t - 1], B[t - 1]) if t > 0 else ZERO
        s = alpha[t] * s + beta[t] * prev + gamma[t] * np.einsum("mp,mn->pn", v0[t], B[t])
        y = np.stack([s @ C[t, m] + D_SKIP * v0[t, m] for m in range(M)])
        outs.append(sum(mo0[m] * silu(z[t] * mz0[m]) * y[m] for m in range(M)))
    return s, np.stack(outs)


def run_siso_block(key_bar, query_bar, d_scale):
    s = np.zeros((P, N))
    outs = []
    for t in range(T):
        prev = np.outer(x[t - 1], key_bar[t - 1]) if t > 0 else ZERO
        s = alpha[t] * s + beta[t] * prev + gamma[t] * np.outer(x[t], key_bar[t])
        outs.append(silu(z[t]) * (s @ query_bar[t] + d_scale * D_SKIP * x[t]))
    return s, np.stack(outs)


s_mimo, out_mimo = run_mimo_block()
s_siso, out_siso = run_siso_block(B.mean(1), C.mean(1), 1.0 / M)
ok("at init the state equals the SISO state with key = mean_m B^(m)", close(s_mimo, s_siso))
ok("at init the output equals the SISO output with query = mean_m C^(m) and D/M",
   close(out_mimo, out_siso))
ok("so the rank-M write is rank ONE at initialisation",
   np.linalg.matrix_rank(np.einsum("mp,mn->pn", v0[3], B[3]), tol=1e-9) == 1)
mx_split = mx0 + RNG.normal(size=(M, P)) * 0.3
ok("and reaches rank M as soon as mimo_x differentiates",
   np.linalg.matrix_rank(np.einsum("mp,mn->pn",
                                   np.einsum("p,mp->mp", x[3], mx_split), B[3]),
                         tol=1e-9) == min(M, P, N))


# =============================================================================
section("7. MIMO and the rotation — one state, therefore one transition")
# =============================================================================
err_gram = 0.0
for t in range(T):
    gram = np.einsum("in,mn->im", C[t], B[t])
    gram_raw = np.einsum("in,mn->im", C_raw[t], B_raw[t])
    err_gram = max(err_gram, np.max(np.abs(gram - gram_raw)))
ok("shared rotation: every one of the M^2 same-step read/write Grams is time-invariant",
   err_gram < 1e-11)
ok("cross-token Grams do move — that is where the accumulated rotation shows up",
   not close(C[5] @ B[1].T, C_raw[5] @ B_raw[1].T, 1e-3))

TL = 40
B_l = RNG.normal(size=(TL, M, N))
C_l = RNG.normal(size=(TL, M, N))
per_rank_step = RNG.normal(size=(TL, M, NPAIR)) * 0.6
per_rank_cum = np.cumsum(per_rank_step, axis=0)
diag_err, off_err = [], []
for t in range(TL):
    rots = [block_rot(per_rank_cum[t, m], N) for m in range(M)]
    g = np.array([[(rots[i] @ C_l[t, i]) @ (rots[m] @ B_l[t, m]) for m in range(M)]
                  for i in range(M)])
    g0 = C_l[t] @ B_l[t].T
    diag_err.append(max(abs(g[i, i] - g0[i, i]) for i in range(M)))
    off_err.append(max(abs(g[i, m] - g0[i, m]) for i in range(M) for m in range(M) if i != m))
ok("per-rank rotation: the i = m diagonal is still invariant", max(diag_err) < 1e-11)
ok("but the off-diagonal — MIMO's R^2 cross terms — drifts within a few tokens",
   off_err[3] > 1e-1)
print("   off-diagonal drift vs per-rank angle spread, at t = 20:")
shared_walk = np.cumsum(RNG.normal(size=(TL, NPAIR)) * 0.6, axis=0)
for spread in (0.0, 0.05, 0.2, 1.0):
    walk = np.cumsum(RNG.normal(size=(TL, M, NPAIR)) * spread, axis=0) + shared_walk[:, None, :]
    rots = [block_rot(walk[20, m], N) for m in range(M)]
    drift = np.mean([abs((rots[i] @ C_l[20, i]) @ (rots[m] @ B_l[20, m]) - C_l[20, i] @ B_l[20, m])
                     for i in range(M) for m in range(M) if i != m])
    print(f"     spread {spread:>4} -> {drift:7.3f}")
    if spread == 0.0:
        ok("spread 0 recovers the shared rotation exactly (it is a superset)", drift < 1e-11)

rots = [block_rot(per_rank_cum[10, m], N) for m in range(M)]
u_svd, _, vt_svd = np.linalg.svd(sum(rots) / M)
q_best = u_svd @ vt_svd
ok("no single orthogonal Q fits the M per-rank frames: no state-space preimage",
   max(np.linalg.norm(rots[m] - q_best) for m in range(M)) > 1e-1)
shared = [block_rot(per_rank_cum[10, 0], N)] * M
u_svd, _, vt_svd = np.linalg.svd(sum(shared) / M)
ok("with a shared rotation it fits exactly — the ranks share the state's transition",
   max(np.linalg.norm(shared[m] - u_svd @ vt_svd) for m in range(M)) < 1e-11)

# And a per-rank rotation reaches no write a free key does not already reach.
rms = lambda u: u / np.sqrt(np.mean(u ** 2) + 1e-8)
masks = [np.diag(mimo_x[m]) for m in range(M)]
n_base = 1 + P + M * N


def w_qk(th):
    keys = th[1 + P:n_base].reshape(M, N)
    return th[0] * sum(np.outer(masks[m] @ th[1:1 + P], rms(keys[m])) for m in range(M))


def w_qk_rot(th):
    keys = th[1 + P:n_base].reshape(M, N)
    ang = th[n_base:].reshape(M, NPAIR)
    return th[0] * sum(np.outer(masks[m] @ th[1:1 + P], block_rot(ang[m], N) @ rms(keys[m]))
                       for m in range(M))


ok("a per-rank rotation adds 0 directions to the write map: QK-norm never removed direction",
   jac_rank(w_qk_rot, RNG.normal(size=n_base + M * NPAIR))
   == jac_rank(w_qk, RNG.normal(size=n_base)))


# =============================================================================
section("8. MIMO and the trapezoid — the Delta-tilde collapse is rank-blind")
# =============================================================================
d_tilde = np.array([gamma[s] + ((1.0 - lam[s + 1]) * dt[s + 1] if s + 1 < T else 0.0)
                    for s in range(T)])
recon = np.prod(alpha) * S0
for s in range(T):
    recon = recon + d_tilde[s] * np.prod(alpha[s + 1:T]) * write(s)
ok("h_T = sum_s Delta-tilde_s * alpha_(s+1:T) * W_s, with W_s the rank-M write",
   close(recon, S[-1], 1e-9))
ok("Delta-tilde carries no rank index: gamma and beta are per-head scalars",
   d_tilde.shape == (T,))
lam_one = np.ones(T)
gam_one, bet_one = lam_one * dt, (1.0 - lam_one) * dt * alpha
s_euler = ZERO.copy()
for t in range(T):
    s_euler = alpha[t] * s_euler + bet_one[t] * write(t - 1) + gam_one[t] * write(t)
d_tilde_one = np.array([gam_one[s] + ((1.0 - lam_one[s + 1]) * dt[s + 1] if s + 1 < T else 0.0)
                        for s in range(T)])
ok("lambda = 1 gives Delta-tilde = Delta (one installment), MIMO unchanged",
   close(d_tilde_one, dt))


# =============================================================================
section("9. MIMO and MambaProduct — what each dial unties")
# =============================================================================
U = 3
dt_u = RNG.uniform(0.05, 1.2, (T, U))
a_u = -RNG.uniform(0.1, 2.0, (T, U))
al_u = np.exp(dt_u * a_u)
lam_u = RNG.uniform(0.05, 0.95, (T, U))
gam_u = lam_u * dt_u
bet_u = (1.0 - lam_u) * dt_u * al_u
x_u = RNG.normal(size=(T, U, P))                       # u FREE values per token
B_u = RNG.normal(size=(T, U, M, N))                    # u * M free keys per token
v_u = np.einsum("tup,mp->tump", x_u, mimo_x)


def fold(t_end=T):
    """Micro-steps folded into the sequence axis, exactly as the block evaluates them."""
    s = np.zeros((P, N))
    prev_w = ZERO
    for t in range(t_end):
        for j in range(U):
            w = np.einsum("mp,mn->pn", v_u[t, j], B_u[t, j])
            s = al_u[t, j] * s + bet_u[t, j] * prev_w + gam_u[t, j] * w
            prev_w = w
    return s


s_ref = fold()
for name, j_perm in [("swap the first two", np.array([1, 0, 2])),
                     ("reverse", np.array([2, 1, 0]))]:
    v_p, B_p = v_u[:, j_perm], B_u[:, j_perm]
    s_p = np.zeros((P, N))
    prev_w = ZERO
    for t in range(T):
        for j in range(U):
            w = np.einsum("mp,mn->pn", v_p[t, j], B_p[t, j])
            s_p = al_u[t, j] * s_p + bet_u[t, j] * prev_w + gam_u[t, j] * w
            prev_w = w
    ok(f"micro-steps are order-dependent ({name}), where MIMO ranks are not",
       not close(s_p, s_ref, 1e-6))

# The token-level normal form: A(x_i) holds no rank index, no key, no value, no lambda.
# Checked on the homogeneous token map, which is what A(x_i) *is*.
def token_map(seed_state, lam_sched=lam_u, val=v_u, key=B_u, ranks=M):
    s = seed_state.copy()
    prev_w = ZERO
    bet_s = (1.0 - lam_sched) * dt_u * al_u
    gam_s = lam_sched * dt_u
    for j in range(U):
        w = np.einsum("mp,mn->pn", val[3, j, :ranks], key[3, j, :ranks])
        s = al_u[3, j] * s + bet_s[3, j] * prev_w + gam_s[3, j] * w
        prev_w = w
    return s


a_token = np.prod(al_u, axis=1)
lam_alt = RNG.uniform(0.05, 0.95, (T, U))
v_alt2 = np.einsum("tup,mp->tump", RNG.normal(size=(T, U, P)), mimo_x)
B_alt2 = RNG.normal(size=(T, U, M, N))
base_map = token_map(S0 + probe_e) - token_map(S0)
ok("A(x_i) = prod_j alpha_(i,j): the homogeneous token map is exactly that scalar",
   close(base_map, a_token[3] * probe_e))
ok("A(x_i) does not see lambda",
   close(token_map(S0 + probe_e, lam_sched=lam_alt) - token_map(S0, lam_sched=lam_alt),
         base_map))
ok("A(x_i) does not see the keys, the values, or M",
   close(token_map(S0 + probe_e, val=v_alt2, key=B_alt2, ranks=1)
         - token_map(S0, val=v_alt2, key=B_alt2, ranks=1), base_map))

# B(x_(i-1), x_i): (u+1) * M outer products.  Measured where the bound bites.
PB, NB, MB, UB = 16, 16, 2, 2
mx_b = RNG.normal(size=(MB, PB))
xb = RNG.normal(size=(2, UB, PB))
Bb = RNG.normal(size=(2, UB, MB, NB))
dtb = RNG.uniform(0.1, 1.0, (2, UB))
alb = np.exp(dtb * -RNG.uniform(0.2, 1.5, (2, UB)))
lamb = RNG.uniform(0.1, 0.9, (2, UB))
gamb, betb = lamb * dtb, (1.0 - lamb) * dtb * alb
vb = np.einsum("tup,mp->tump", xb, mx_b)


def fold_b(t_end):
    s = np.zeros((PB, NB))
    prev_w = np.zeros((PB, NB))
    for t in range(t_end):
        for j in range(UB):
            w = np.einsum("mp,mn->pn", vb[t, j], Bb[t, j])
            s = alb[t, j] * s + betb[t, j] * prev_w + gamb[t, j] * w
            prev_w = w
    return s


b_token = fold_b(2) - np.prod(alb[1]) * fold_b(1)
ok(f"B(x_(i-1), x_i) has rank exactly (u+1) * M = {(UB + 1) * MB} at P = N = {PB}",
   np.linalg.matrix_rank(b_token, tol=1e-9) == (UB + 1) * MB)

# Containment: MambaProduct(u = M) reproduces a whole MIMO(M) trajectory.
lam_one_t = np.ones(T)
gam_t, bet_t = lam_one_t * dt, (1.0 - lam_one_t) * dt * alpha
s_mimo_traj = ZERO.copy()
for t in range(T):
    s_mimo_traj = alpha[t] * s_mimo_traj + gam_t[t] * write(t)
s_prod_traj = ZERO.copy()
for t in range(T):
    al_c = np.full(M, alpha[t] ** (1.0 / M))
    carry = np.array([np.prod(al_c[j + 1:]) for j in range(M)])
    dt_c = dt[t] / carry                      # lambda = 1, so gamma_(t,j) = dt_c[j]
    for j in range(M):
        s_prod_traj = al_c[j] * s_prod_traj + dt_c[j] * np.outer(v[t, j], B[t, j])
ok("MambaProduct(u = M) reproduces a whole MIMO(M) trajectory exactly",
   close(s_prod_traj, s_mimo_traj, 1e-9))
# (the converse fails by dimension, and section 4 is where that is measured)

# The tying table, checked by perturbation: what MIMO shares and micro-steps do not.
def fold_scalar(dt_s, a_s, lam_s):
    s = np.zeros((P, N))
    prev_w = ZERO
    al_s = np.exp(dt_s * a_s)
    for j in range(U):
        w = np.einsum("mp,mn->pn", v_u[3, j], B_u[3, j])
        s = al_s[j] * s + (1.0 - lam_s[j]) * dt_s[j] * al_s[j] * prev_w + lam_s[j] * dt_s[j] * w
        prev_w = w
    return s


base_fold = fold_scalar(dt_u[3], a_u[3], lam_u[3])
bumped = dt_u[3].copy()
bumped[1] *= 1.1
ok("micro-steps carry their own Delta: bumping Delta_j alone changes the token map",
   not close(fold_scalar(bumped, a_u[3], lam_u[3]), base_fold, 1e-6))
scaled_ranks = v[3] * np.array([1.0, 1.3, 0.7])[:, None]
ok("MIMO ranks share one Delta: a per-rank scale must come from the value mask instead",
   not close(np.einsum("mp,mn->pn", scaled_ranks, B[3]),
             np.einsum("mp,mn->pn", v[3], B[3]), 1e-6))

# The rotation: M ranks travel rigidly, u micro-steps are staggered.
step_rots = [block_rot(RNG.normal(size=NPAIR) * 0.5, N) for _ in range(U)]
cum_rots = [step_rots[0]]
for j in range(1, U):
    cum_rots.append(step_rots[j] @ cum_rots[-1])
groups = [np.stack([cum_rots[j] @ B_l[0, m] for m in range(M)]) for j in range(U)]
ok("within one micro-step the M keys stay a rigid frame (Gram fixed)",
   all(close(groups[j] @ groups[j].T, B_l[0] @ B_l[0].T, 1e-11) for j in range(U)))
ok("across micro-steps they are staggered by the accumulated rotation",
   not close(groups[2] @ groups[0].T, B_l[0] @ B_l[0].T, 1e-3))


# =============================================================================
print(f"\n{PASSED} passed, {FAILED} failed")
sys.exit(1 if FAILED else 0)
