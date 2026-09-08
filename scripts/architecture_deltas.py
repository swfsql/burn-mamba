#!/usr/bin/env python3
"""Numerical verification for `info/architecture-deltas.md`.

Every claim the document makes that can be checked numerically is checked here, in
float64, with section numbers matching the document's. Pure `numpy`; no other
dependency, no I/O, no reference to the model code — the point is that these are
statements about the *block*, reproducible from its definition alone.

    python3 scripts/architecture_deltas.py

Exits non-zero if any check fails.

Conventions. One head, SISO unless a check says otherwise. `N` is the state rank
(the axis BCNorm normalises and the biases live on), `P` the head dimension, `L` a
short sequence. The recurrence and its coefficients are the ones the two companion
notes fix:

    alpha[t] = exp(dt[t] * A[t])                 A[t] <= -a_floor
    beta[t]  = (1 - lam[t]) * dt[t] * alpha[t]
    gamma[t] = lam[t] * dt[t]

and the block's B/C path, in the order the reference kernel and this crate both run
it (`mamba3.py` -> `mamba3_siso_fwd.py`; `helpers::qk_norm_expand_bias` then
`rotate_bc_forward`):

    B = rope(rmsnorm(B_raw) + B_bias)            norm, then bias, then rotation
    C = rope(rmsnorm(C_raw) + C_bias)
"""

import math
import sys

import numpy as np

RNG = np.random.default_rng(20260905)
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


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300)


def section(title):
    print(f"\n{title}")


# ---------------------------------------------------------------------------
# Block primitives
# ---------------------------------------------------------------------------

EPS = 1e-5


def rmsnorm(v, gain=1.0, eps=EPS):
    """RMSNorm over the last axis, as `RMSNormGated` runs it with no gate."""
    v = np.asarray(v, float)
    r = np.sqrt(np.mean(v * v, axis=-1, keepdims=True) + eps)
    return gain * v / r


def silu(x):
    return x / (1.0 + np.exp(-x))


def rope_pairs(v, theta, rope_dim):
    """Rotate the first `rope_dim` coordinates of `v` as interleaved 2-blocks.

    `theta` has one angle per pair. The tail (`N - rope_dim` coordinates) passes
    through untouched — this is `rope_fraction < 1`.
    """
    v = np.asarray(v, float).copy()
    for j in range(rope_dim // 2):
        a, b = v[2 * j], v[2 * j + 1]
        c, s = np.cos(theta[j]), np.sin(theta[j])
        v[2 * j], v[2 * j + 1] = c * a - s * b, s * a + c * b
    return v


def coefficients(dt, a_raw, lam_raw, a_floor=1e-4):
    """The trapezoid's three coefficients from the raw projections.

    `A = -softplus(a_raw)` clamped to `<= -a_floor`, `dt = softplus(dt_raw)`,
    `lam = sigmoid(lam_raw)` — exactly `mamba3.py`'s parameterisation.
    """
    A = -np.logaddexp(0.0, a_raw)
    A = np.minimum(A, -a_floor)
    lam = 1.0 / (1.0 + np.exp(-lam_raw))
    alpha = np.exp(dt * A)
    beta = (1.0 - lam) * dt * alpha
    gamma = lam * dt
    return alpha, beta, gamma, A, lam


# ---------------------------------------------------------------------------
# 3. BCNorm pins the score's scale
# ---------------------------------------------------------------------------

section("3. BCNorm (QK-norm on B and C)")

N = 128
b_raw = RNG.normal(size=N)
c_raw = RNG.normal(size=N)

b = rmsnorm(b_raw)
ok("3.1  RMSNorm output has unit RMS, hence ||B|| = sqrt(N)",
   abs(np.linalg.norm(b) - np.sqrt(N)) < 1e-3)

# Exact but for the norm's `eps`, which is why the tolerances here are 1e-4 and
# not machine epsilon: every invariance in this file holds to O(eps / rms^2).
ok("3.2  BCNorm is invariant to the projection's scale (x1e3)",
   rel(rmsnorm(1e3 * b_raw), rmsnorm(b_raw)) < 1e-4)

# Without the norm, the score is the product of two learned scales.
unnormed = lambda s: float((s * c_raw) @ (s * b_raw))
normed = lambda s: float(rmsnorm(s * c_raw) @ rmsnorm(s * b_raw))
ok("3.3  unnormed score scales as s^2; normed score does not",
   abs(unnormed(10.0) / unnormed(1.0) - 100.0) < 1e-6
   and abs(normed(10.0) / normed(1.0) - 1.0) < 1e-4)

# The score's *magnitude* becomes a function of the shape alone.
S = 4000
B_s = rmsnorm(RNG.normal(size=(S, N)))
C_s = rmsnorm(RNG.normal(size=(S, N)))
data_scores = np.einsum("sn,sn->s", C_s, B_s)
ok("3.4  normed data-data score has mean ~0 and sd ~sqrt(N)",
   abs(data_scores.mean()) < 0.2 * np.sqrt(N)
   and abs(data_scores.std() / np.sqrt(N) - 1.0) < 0.10)


# ---------------------------------------------------------------------------
# 4. The B/C biases — an affine score, and a positive floor at init
# ---------------------------------------------------------------------------

section("4. B/C biases")

bias_b = np.ones(N)
bias_c = np.ones(N)

Bt = B_s + bias_b
Ct = C_s + bias_c
full = np.einsum("sn,sn->s", Ct, Bt)
t_dd = np.einsum("sn,sn->s", C_s, B_s)          # data  . data
t_db = C_s @ bias_b                              # data  . bias
t_bd = B_s @ bias_c                              # bias  . data
t_bb = float(bias_c @ bias_b)                    # bias  . bias
ok("4.1  the biased score is exactly the four-term affine expansion",
   close(full, t_dd + t_db + t_bd + t_bb, tol=1e-9))

ok("4.2  bias-bias term is exactly N; the other three are O(sqrt(N))",
   abs(t_bb - N) < 1e-9
   and abs(t_dd.std() / np.sqrt(N) - 1.0) < 0.10
   and abs(t_db.std() / np.sqrt(N) - 1.0) < 0.10
   and abs(t_bd.std() / np.sqrt(N) - 1.0) < 0.10)

dev = np.abs(full / N - 1.0)
ok("4.3  at init the score sits within ~30% of the constant N",
   dev.mean() < 0.30)


def deviation(n):
    Bx = rmsnorm(RNG.normal(size=(S, n))) + 1.0
    Cx = rmsnorm(RNG.normal(size=(S, n))) + 1.0
    sc = np.einsum("sn,sn->s", Cx, Bx)
    return np.abs(sc / n - 1.0).mean()


d32, d128 = deviation(32), deviation(128)
ok("4.4  that deviation shrinks as 1/sqrt(N) (32 vs 128 -> factor ~2)",
   abs(d32 / d128 - 2.0) < 0.25)

ok("4.5  with the ones bias every pair scores positive; without it, half do",
   (full > 0).mean() > 0.999 and abs((t_dd > 0).mean() - 0.5) < 0.05)

# The paper's four bias initialisations, by the floor each produces.
floors = {}
for name, draw in (
    ("1.0", lambda: np.ones(N)),
    ("0.0", lambda: np.zeros(N)),
    ("U(0,1)", lambda: RNG.uniform(0.0, 1.0, size=N)),
    ("U(-1,1)", lambda: RNG.uniform(-1.0, 1.0, size=N)),
):
    vals = [float(draw() @ draw()) for _ in range(200)]
    floors[name] = float(np.mean(vals))
ok("4.6  floor ordering 1.0 > U(0,1) >> U(-1,1) ~ 0.0, and 1.0 = N, U(0,1) = N/4",
   abs(floors["1.0"] - N) < 1e-9
   and abs(floors["U(0,1)"] - N / 4) < 0.15 * N
   and abs(floors["0.0"]) < 1e-9
   and abs(floors["U(-1,1)"]) < 0.10 * N)
print(f"        floors: {', '.join(f'{k}={v:.1f}' for k, v in floors.items())}")

# 4.7 The bias is added *before* the rotation, so the floor is not flat.
f = 0.5
rope_dim = int(f * N) // 2 * 2
n_pairs = rope_dim // 2
theta_t = RNG.uniform(-np.pi, np.pi, size=n_pairs)
theta_s = RNG.uniform(-np.pi, np.pi, size=n_pairs)
pre_t = rope_pairs(np.ones(N), theta_t, rope_dim)
pre_s = rope_pairs(np.ones(N), theta_s, rope_dim)
predicted = (N - rope_dim) + 2.0 * np.cos(theta_t - theta_s).sum()
ok("4.7  bias-before-rope: floor = (N - rope_dim) + 2*sum cos(theta_t - theta_s)",
   abs(float(pre_t @ pre_s) - predicted) < 1e-9)

post = float(rope_pairs(np.zeros(N), theta_t, rope_dim) @ np.ones(N) + np.ones(N) @ np.ones(N))
ok("4.8  bias-after-rope would instead be the flat constant N",
   abs(post - N) < 1e-9)

many = []
for _ in range(2000):
    tt = RNG.uniform(-np.pi, np.pi, size=n_pairs)
    ss = RNG.uniform(-np.pi, np.pi, size=n_pairs)
    many.append(float(rope_pairs(np.ones(N), tt, rope_dim)
                      @ rope_pairs(np.ones(N), ss, rope_dim)))
ok("4.9  averaged over rotation lag the floor decays to the unrotated part (N - rope_dim)",
   abs(np.mean(many) - (N - rope_dim)) < 0.05 * N)

# 4.10 At init the layer's read is, to O(1/sqrt(N)), the bias-only read.
L, P = 24, 64
dt = np.logaddexp(0.0, RNG.normal(size=L) - 4.0)
alpha, beta, gamma, A, lam = coefficients(
    dt, RNG.normal(size=L), RNG.normal(size=L))
Bm = rmsnorm(RNG.normal(size=(L, N))) + 1.0
Cm = rmsnorm(RNG.normal(size=(L, N))) + 1.0
V = RNG.normal(size=(L, P))
decay = np.zeros((L, L))
for t in range(L):
    for s in range(t + 1):
        decay[t, s] = np.prod(alpha[s + 1:t + 1]) * gamma[s]
score = (Cm @ Bm.T) * decay
score_bias_only = N * decay
y = score @ V
y_bias_only = score_bias_only @ V
ok("4.10 at init the read is the bias-only (pure decay) read to within ~1/sqrt(N)",
   rel(y, y_bias_only) < 4.0 / np.sqrt(N))


# ---------------------------------------------------------------------------
# 5. No short convolution
# ---------------------------------------------------------------------------

section("5. The removed short convolution, and its two replacements")

L = 32
dt = np.logaddexp(0.0, RNG.normal(size=L) - 2.0)
alpha, beta, gamma, A, lam = coefficients(
    dt, RNG.normal(size=L), RNG.normal(size=L))
v = RNG.normal(size=(L, N))                      # the state-input B_t x_t

h3 = np.zeros(N)
three_term = []
for t in range(L):
    h3 = alpha[t] * h3 + (beta[t] * v[t - 1] if t > 0 else 0.0) + gamma[t] * v[t]
    three_term.append(h3.copy())

h2 = np.zeros(N)
filtered = []
for t in range(L):
    vt = gamma[t] * v[t] + (beta[t] * v[t - 1] if t > 0 else 0.0)
    h2 = alpha[t] * h2 + vt
    filtered.append(h2.copy())
ok("5.1  the 3-term recurrence is a 2-term one on a width-2 filtered state-input",
   close(np.array(three_term), np.array(filtered), tol=1e-12))

samples = []
for _ in range(5000):
    a, b, g, _, _ = coefficients(
        np.logaddexp(0.0, RNG.normal() - 2.0), RNG.normal(), RNG.normal())
    samples.append((b, g))
samples = np.array(samples)
ok("5.2  the implicit filter's taps are strictly positive: it can smooth, never difference",
   (samples > 0).all())
ok("5.3  their ratio beta/gamma is unbounded, so the filter spans the positive quadrant",
   (samples[:, 0] / samples[:, 1]).min() < 0.05
   and (samples[:, 0] / samples[:, 1]).max() > 5.0)


# ---------------------------------------------------------------------------
# 6. Data-dependent A unties decay from write weight
# ---------------------------------------------------------------------------

section("6. Data-dependent A")

A_FLOOR = 1e-4


def jac_rank(fn, x, eps=1e-6):
    x = np.asarray(x, float)
    cols = []
    for i in range(x.size):
        e = np.zeros_like(x)
        e[i] = eps
        cols.append((fn(x + e) - fn(x - e)) / (2 * eps))
    return np.linalg.matrix_rank(np.stack(cols, axis=1), tol=1e-8)


A_head = -3.0                                    # Mamba-2: one learned scalar per head
m2 = lambda p: np.array([np.exp(p[0] * A_head), p[0]])          # p = [dt]
m3 = lambda p: np.array([np.exp(p[0] * -np.logaddexp(0.0, p[1])), p[0]])  # p = [dt, a_raw]
ok("6.1  Mamba-2's (alpha, gamma) locus is one-dimensional per head",
   jac_rank(m2, np.array([0.3])) == 1)
ok("6.2  Mamba-3's is two-dimensional: A adds exactly one dof per (token, head)",
   jac_rank(m3, np.array([0.3, 0.5])) == 2)

# lam = 1 throughout this section, so gamma = dt exactly and the comparison with
# Mamba-2 (which is lam = 1) is like for like.
LAM_ONE = np.full(40000, 30.0)
dts = np.logaddexp(0.0, RNG.normal(size=40000) - 1.0)
a_raws = RNG.normal(size=40000) * 4.0
al, _, ga, _, _ = coefficients(dts, a_raws, LAM_ONE)
ok("6.3  the floor caps the reachable decay: alpha <= exp(-a_floor * gamma)",
   (al <= np.exp(-A_FLOOR * ga) + 1e-12).all())

g0 = 0.5
sweep = np.array([coefficients(np.array([g0]), np.array([r]), np.array([30.0]))[0][0]
                  for r in np.linspace(-30.0, 30.0, 4000)])
ok("6.4  at fixed gamma, alpha sweeps (0, exp(-a_floor*gamma)]",
   sweep.min() < 1e-6 and abs(sweep.max() - np.exp(-A_FLOOR * g0)) < 1e-6)

# Same experiment on a token stream: among tokens that happen to share a write
# weight, how much decay is still available?
bucket = np.abs(ga - g0) < 0.01 * g0
al2 = np.exp(ga[bucket] * A_head)
al3 = al[bucket]
ok("6.5  among tokens of equal gamma, Mamba-2's alpha is a point and Mamba-3's is a range",
   bucket.sum() > 20 and np.ptp(al3) > 0.5 and np.ptp(al3) > 50 * np.ptp(al2))
print(f"        gamma ~ {g0}: alpha spread  Mamba-2 {np.ptp(al2):.2e}   "
      f"Mamba-3 {np.ptp(al3):.2f}")


# ---------------------------------------------------------------------------
# 7. The optional output norm: what each placement erases
# ---------------------------------------------------------------------------

section("7. Output-norm placement")

P = 64
y = RNG.normal(size=P)
z = RNG.normal(size=P)
g = silu(z)
c = 3.0

post_gate = lambda yy, gg: rmsnorm(yy * gg)
pre_gate = lambda yy, gg: rmsnorm(yy) * gg
no_norm = lambda yy, gg: yy * gg

ok("7.1  post-gate norm erases a positive rescaling of the gate",
   rel(post_gate(y, c * g), post_gate(y, g)) < 1e-3)
ok("7.2  pre-gate norm keeps it (scales by c)",
   close(pre_gate(y, c * g), c * pre_gate(y, g), tol=1e-9))
ok("7.3  post-gate and pre-gate both erase a rescaling of y",
   rel(post_gate(c * y, g), post_gate(y, g)) < 1e-3
   and rel(pre_gate(c * y, g), pre_gate(y, g)) < 1e-3)
ok("7.4  no norm keeps both, so the block's output scale is set upstream",
   rel(no_norm(c * y, g), no_norm(y, g)) > 1.0
   and rel(no_norm(y, c * g), no_norm(y, g)) > 1.0)

# Grouping: per-head (group_size = P) vs whole-d_inner normalisation differ as soon
# as the heads' scales differ — the grouped one equalises them.
H = 4
Y = RNG.normal(size=(H, P)) * np.array([1.0, 10.0, 0.1, 3.0])[:, None]
grouped = rmsnorm(Y)                              # per head
flat = rmsnorm(Y.reshape(-1)).reshape(H, P)       # over d_inner
head_rms = lambda M: np.sqrt((M ** 2).mean(axis=1))
ok("7.5  grouped RMSNorm equalises per-head scale; the flat one preserves the spread",
   head_rms(grouped).std() < 1e-3 and head_rms(flat).std() > 0.5)


# ---------------------------------------------------------------------------
# 8. Chunk length under MIMO
# ---------------------------------------------------------------------------

section("8. Chunk length under MIMO")


def flops(T, C, N_, P_, R):
    """The paper's chunked SSD FLOP count, generalised to MIMO rank R."""
    intra = (T / C) * (2 * (C * R) ** 2 * N_ + 2 * (C * R) ** 2 * P_)
    inter = (T / C) * (4 * N_ * P_ * (C * R) + 2 * N_ * P_)
    return intra + inter


T, Nn, Pp = 8192, 64, 64
ok("8.1  SISO at C = N = P costs ~8*T*N^2",
   abs(flops(T, Nn, Nn, Pp, 1) / (8 * T * Nn ** 2) - 1.0) < 0.02)

R = 4
ok("8.2  MIMO at C = N/R keeps the cost at ~R x SISO",
   abs(flops(T, Nn // R, Nn, Pp, R) / (8 * T * R * Nn ** 2) - 1.0) < 0.05)

intra = lambda C, r: (T / C) * (2 * (C * r) ** 2 * Nn + 2 * (C * r) ** 2 * Pp)
ok("8.3  at fixed C the intra-chunk term scales exactly as R^2",
   abs(intra(Nn, R) / intra(Nn, 1) - R ** 2) < 1e-9)

ratio_scaled = flops(T, Nn // R, Nn, Pp, R) / flops(T, Nn, Nn, Pp, 1)
ratio_unscaled = flops(T, Nn, Nn, Pp, R) / flops(T, Nn, Nn, Pp, 1)
ok("8.4  so leaving C at its SISO value costs ~2.5x the C = N/R schedule here",
   abs(ratio_scaled - R) < 0.25 and ratio_unscaled / ratio_scaled > 2.0)
print(f"        R={R}: C=N/R -> {ratio_scaled:.2f}x,  C=N -> {ratio_unscaled:.2f}x")

# The crate's schedule (`Mamba3SsdPath::optimal_chunk_len`): the same square-root
# rule, divided by the rank (which widens both of a chunk's axes) and put back on
# the 32 grid, then rounded up to a whole number of tokens (`micro_steps` widens
# only the write axis, so it subdivides the chunk instead of shortening it).
def optimal_chunk_len(state_rank, per_head_dim, mimo_rank=1, micro_steps=1):
    u = max(micro_steps, 1)
    folded = -(-math.isqrt(state_rank * per_head_dim) // max(mimo_rank, 1))
    folded = ((folded + 31) // 32) * 32
    folded = min(max(folded, 32), 512)
    return -(-folded // u) * u


ok("8.5  the schedule is 96 / 64 / 32 at rank 1 / 2 / >=4, and u leaves the width alone",
   optimal_chunk_len(128, 64) == 96
   and optimal_chunk_len(128, 64, 2, 1) == 64
   and optimal_chunk_len(128, 64, 4, 1) == 32
   and all(96 <= optimal_chunk_len(128, 64, 1, u) < 96 + u
           and optimal_chunk_len(128, 64, 1, u) % u == 0
           for u in (1, 2, 3, 4, 5, 8, 16)))

fused = lambda m, u: optimal_chunk_len(128, 64, m, u) * m
ok("8.6  it keeps the fused axis near the SISO target instead of scaling with the rank",
   fused(1, 1) == 96 and fused(4, 1) == 128 and 96 * 4 == 384)
print(f"        fused chunk*m:  m=1 -> {fused(1, 1)},  m=4 -> {fused(4, 1)}"
      f"  (unscaled would be {96 * 4})")

# The materialised intra-chunk score is [batch, nchunks, heads, (C/u)*m, C*m]: the
# read axis carries the tokens, the write axis the micro-steps. Over sequence*u
# folded positions that is batch*s*heads * C*m^2 elements -- no u at all.
score_mem = lambda m, u: optimal_chunk_len(128, 64, m, u) * m ** 2   # per token, per head
ok("8.7  the score carries no u: memory per token is C*m^2, flat to the token rounding",
   all(1.0 <= score_mem(m, u) / score_mem(m, 1)
       < 1.0 + u / optimal_chunk_len(128, 64, m, 1)
       for m in (1, 2, 4, 8) for u in (1, 2, 3, 4, 8))
   and all(score_mem(m, u) == score_mem(m, 1)
           for m in (1, 2, 4, 8) for u in (1, 2, 4, 8)
           if optimal_chunk_len(128, 64, m, 1) % u == 0))

# The alternative -- divide by the product, as if a chunk of C positions were C/u
# tokens' worth of work. It doubles the exponent on nchunks and, once 32 floors the
# division, fails to hold the memory it was spent on anyway.
def divide_by_product(state_rank, per_head_dim, mimo_rank=1, micro_steps=1):
    n = -(-math.isqrt(state_rank * per_head_dim) // max(mimo_rank * micro_steps, 1))
    return min(max(((n + 31) // 32) * 32, 32), 512)

nchunks = lambda sched, u: u / sched(128, 64, 1, u)          # per token, up to `s`
ok("8.8  dividing by u too puts nchunks at u^2 and still lets the score grow 2.67x at u=8",
   abs(nchunks(optimal_chunk_len, 8) / nchunks(optimal_chunk_len, 1) - 8) < 0.1
   and abs(nchunks(divide_by_product, 8) / nchunks(divide_by_product, 1) - 24) < 0.1
   and abs(divide_by_product(128, 64, 1, 8) * 1 ** 2 * 8 / 96 - 8.0 / 3.0) < 1e-9
   and score_mem(1, 8) == score_mem(1, 1))
print(f"        nchunks vs u=1 at u=8:  read axis -> "
      f"{nchunks(optimal_chunk_len, 8) / nchunks(optimal_chunk_len, 1):.0f}x,  "
      f"divide-by-product -> "
      f"{nchunks(divide_by_product, 8) / nchunks(divide_by_product, 1):.0f}x")
print(f"        score memory vs u=1 at u=8:  read axis -> 1.00x,  "
      f"divide-by-product -> {divide_by_product(128, 64, 1, 8) * 8 / 96:.2f}x")


# ---------------------------------------------------------------------------

print(f"\n{PASSED} passed, {FAILED} failed")
sys.exit(1 if FAILED else 0)
