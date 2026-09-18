#!/usr/bin/env python3
"""Numerical verification for `info/kalman/gate-as-positive-system.md`.

Every claim the document makes that can be checked numerically is checked here, in
float64, with section numbers matching the document's. Pure `numpy`; no other
dependency, no I/O, no reference to the model code — the point is that these are
statements about the *recurrences*, reproducible from the equations alone.

    python3 scripts/gate_as_positive_system.py

Exits non-zero if any check fails.

Conventions (document §2). A positive system is a nonnegative 2x2 matrix acting
projectively on a nonnegative 2-vector, carried in log coordinates:

    (M (x) s)_i = log sum_j exp(M_ij + s_j)             the log-semiring product
    read(s)     = s_0 - s_1                             the projective read

Its two members, per head and per position:

    Kalman    M = log [[alpha*(1 + q*m), m], [alpha*q, 1]]      read = log Lambda
    tropical  M =     [[a, b], [-inf, 0]]                       read = c

with `alpha = exp(dt*A)` the block's projected decay, `m = dt` the step's mass and
`q = kappa*dt*exp(r)` the doubt it injects. The Kalman member's plant runs

    d[t]      = alpha[t] / (1 + q[t]*alpha[t]*Lambda[t-1])      the computed decay
    Lambda[t] = d[t]*Lambda[t-1] + m[t]                         precision
    eta[t]    = d[t]*eta[t-1] + m[t]*T[t]                       information
    S[t]      = eta[t] / Lambda[t]                              the estimate
"""

import sys

import numpy as np

RNG = np.random.default_rng(20260917)
PASSED = 0
FAILED = 0
NEG = -1e30  # the log-semiring zero, kept finite


def ok(name, cond):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"ok   {name}")
    else:
        FAILED += 1
        print(f"FAIL {name}")


def close(a, b, tol=1e-9):
    return bool(np.max(np.abs(np.asarray(a) - np.asarray(b))) <= tol * (1 + np.max(np.abs(b))))


# ---------------------------------------------------------------------------
# §2  Positive systems, in log coordinates
# ---------------------------------------------------------------------------


def log_mul(x, y):
    """The log-semiring matrix product."""
    return np.logaddexp.reduce(x[..., :, :, None] + y[..., None, :, :], axis=-2)


def log_apply(m, s):
    return np.logaddexp.reduce(m + s[..., None, :], axis=-1)


def check_semiring():
    x, y = np.log(RNG.random((2, 2)) + 0.1), np.log(RNG.random((2, 2)) + 0.1)
    ok("2.1 log-semiring product is the ordinary product of the exponentials",
       close(np.exp(log_mul(x, y)), np.exp(x) @ np.exp(y)))

    s = np.log(RNG.random(2) + 0.1)
    shift = 3.7
    read = lambda v: v[0] - v[1]
    ok("2.2 the projective read ignores a common shift of the matrix",
       close(read(log_apply(x + shift, s)), read(log_apply(x, s))))

    # A prefix product by doubling equals the sequential fold, for both members.
    def fold(ms):
        out = [ms[0]]
        for t in range(1, len(ms)):
            out.append(log_mul(ms[t], out[-1]))
        return np.array(out)

    def doubling(ms):
        acc, off, n = ms.copy(), 1, len(ms)
        ident = np.array([[0.0, NEG], [NEG, 0.0]])
        while off < n:
            shifted = np.concatenate([np.repeat(ident[None], off, 0), acc[: n - off]])
            acc = log_mul(acc, shifted)
            off *= 2
        return acc

    ms = np.log(RNG.random((17, 2, 2)) + 0.05)
    ok("2.3 doubling prefix == sequential fold (projective)", close(doubling(ms), fold(ms), 1e-9))
    affine = np.stack([np.stack([RNG.normal(size=13), RNG.normal(size=13)], -1),
                       np.stack([np.full(13, NEG), np.zeros(13)], -1)], -2)
    ok("2.3 doubling prefix == sequential fold (affine)",
       close(doubling(affine)[:, 0, :], fold(affine)[:, 0, :], 1e-9))


# ---------------------------------------------------------------------------
# §3  The Kalman member
# ---------------------------------------------------------------------------


def gate(alpha, m, q, lam0=0.0, targets=None):
    """The information-form recurrence; returns (d, Lambda, eta)."""
    lam, eta = lam0, 0.0
    ds, lams, etas = [], [], []
    for t in range(len(alpha)):
        d = alpha[t] / (1.0 + q[t] * alpha[t] * lam)
        lam = d * lam + m[t]
        eta = d * eta + (m[t] * targets[t] if targets is not None else 0.0)
        ds.append(d)
        lams.append(lam)
        etas.append(eta)
    return np.array(ds), np.array(lams), np.array(etas)


def check_kalman():
    n = 60
    alpha = np.exp(-RNG.random(n) * 0.3)
    m = RNG.random(n) + 0.2
    q = RNG.random(n) * 0.4 + 0.01
    targets = RNG.normal(size=n)

    # 3.1 The information form is the covariance-form Kalman filter of a random
    # walk with process noise q/(a^2 scaling) observed with precision m.
    d, lam, eta = gate(alpha, m, q, lam0=1.3, targets=targets)
    p, mean = 1 / 1.3, 0.0
    cov_lam, cov_mean = [], []
    for t in range(n):
        a2 = 1.0 / alpha[t]
        p_prior = a2 * p + q[t]
        k = p_prior / (p_prior + 1.0 / m[t])
        p = (1 - k) * p_prior
        mean += k * (targets[t] - mean)
        cov_lam.append(1 / p)
        cov_mean.append(mean)
    ok("3.1 information form == covariance-form filter (precision)", close(lam, cov_lam))
    ok("3.1 information form == covariance-form filter (mean)", close(eta / lam, cov_mean))

    # 3.2 The matrix of §2 generates the same precision.
    logm = np.stack(
        [
            np.stack([np.log(alpha * (1 + q * m)), np.log(m)], -1),
            np.stack([np.log(alpha * q), np.zeros(n)], -1),
        ],
        -2,
    )
    s = np.array([np.log(1.3), 0.0])
    by_matrix = []
    for t in range(n):
        s = log_apply(logm[t], s)
        by_matrix.append(np.exp(s[0] - s[1]))
    ok("3.2 the Möbius matrix generates the precision", close(by_matrix, lam, 1e-8))

    # 3.3 Containment: kappa = 0 is the projected decay, bit for bit.
    d0, _, _ = gate(alpha, m, np.zeros(n), lam0=1.3)
    ok("3.3 q = 0 ⇒ d = alpha exactly", np.array_equal(d0, alpha))

    # 3.4 The ceiling, whatever the history.
    _, lam_hi, _ = gate(alpha, m, q, lam0=1e6)
    ok("3.4 Lambda[t] < 1/q[t] + m[t] from any start", bool(np.all(lam_hi < 1 / q + m)))

    # 3.5 Birkhoff: the contraction rate has no alpha in it.
    _, lam_a, _ = gate(alpha, m, q, lam0=0.5)
    _, lam_b, _ = gate(alpha, m, q, lam0=50.0)
    dist = np.abs(np.log(lam_a) - np.log(lam_b))
    bound = np.abs(np.log(0.5) - np.log(50.0)) * np.cumprod(np.tanh(0.25 * np.log1p(1 / (q * m))))
    ok("3.5 the Hilbert distance contracts at the Birkhoff rate", bool(np.all(dist <= bound + 1e-12)))

    # 3.6 Gain collapse is what the ceiling prevents: at q = 0 and alpha = 1 the
    # head stops writing, the gain falling like 1/t.
    _, lam_flat, _ = gate(np.ones(n), np.ones(n), np.zeros(n))
    ok("3.6 q = 0, alpha = 1 ⇒ the gain collapses like 1/t", close(1 / lam_flat, 1 / np.arange(1, n + 1)))

    # 3.7 Hyperbolic vs geometric ageing: no single (alpha, scale) fits the
    # filter's discount over a range of evidence and elapsed time.
    ks = np.arange(1, 9)
    lams0 = np.array([2.0, 5.0, 12.0, 30.0])
    q_gap = 0.5
    exact = np.array([[l / (1 + k * q_gap * l) for k in ks] for l in lams0])
    best = np.inf
    for a in np.linspace(0.01, 0.99, 99):
        geo = np.array([[a**k * l for k in ks] for l in lams0])
        scale = np.sum(geo * exact) / np.sum(geo * geo)
        best = min(best, np.max(np.abs(scale * geo - exact) / exact))
    print(f"     best geometric fit to the hyperbolic discount: {best:.2f} relative error")
    ok("3.7 a geometric discount cannot be the filter's", best > 0.3)


# ---------------------------------------------------------------------------
# §4  The tropical member
# ---------------------------------------------------------------------------


def tropical(a, b, c0=NEG, tau=1.0):
    c, out = c0, []
    for t in range(len(a)):
        c = tau * np.logaddexp((c + a[t]) / tau, b[t] / tau)
        out.append(c)
    return np.array(out)


def hard(a, b, c0=-np.inf):
    c, out = c0, []
    for t in range(len(a)):
        c = max(c + a[t], b[t])
        out.append(c)
    return np.array(out)


def check_tropical():
    n = 40
    a = RNG.integers(-3, 4, n).astype(float)
    b = RNG.integers(-3, 4, n).astype(float) * 4

    soft, exact = tropical(a, b), hard(a, b)
    gap = soft - exact
    ok("4.1 the soft register is above the hard one", bool(np.all(gap > -1e-12)))
    ok("4.2 and by at most ln(t + 1)", bool(np.all(gap <= np.log(np.arange(1, n + 1) + 1) + 1e-9)))

    # 4.3 Scale is the temperature: the same recursion at scale s is the tau = 1/s one.
    s = 8.0
    ok("4.3 the projection's scale is the temperature",
       close(tropical(s * a, s * b) / s, tropical(a, b, tau=1 / s), 1e-9))

    # 4.4 The named members, at a scale where the soft value rounds to the hard one.
    opens = RNG.integers(0, 2, n) * 2 - 1  # +-1
    d, out = 0, []
    for step in opens:
        d = max(d + int(step), 0)
        out.append(d)
    depth = np.array(out, float)
    counter = tropical(s * opens, np.zeros(n), tau=1.0) / s
    ok("4.4 (a, b) = (±1, 0) is the Lindley recursion (a counter with a floor)",
       close(np.round(counter), depth, 1e-9))

    values = RNG.normal(size=n)
    running_max = np.maximum.accumulate(values)
    ok("4.4 (a, b) = (0, v) is a running maximum",
       close(tropical(np.zeros(n), s * values) / s, running_max, 5e-2))


# ---------------------------------------------------------------------------
# §5  The ports, and what a classifier gets for free
# ---------------------------------------------------------------------------


def check_ports():
    # 5.1 A decision never needs the division: cross-multiplication is the same
    # comparison, and a per-token gate can form it.
    n = 500
    sums = RNG.normal(size=n) * 5
    counts = RNG.random(n) * 9 + 1
    v = RNG.normal(size=n)
    ok("5.1 sign(v - S/n) == sign(v*n - S)",
       bool(np.all(np.sign(v - sums / counts) == np.sign(v * counts - sums))))

    # 5.2 A gap scales eta and Lambda together: the estimate does not move, only
    # the weight behind it does.
    alpha = np.ones(6)
    m = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 1.0])
    q = np.array([0.0, 0.0, 0.0, 0.5, 0.5, 0.0])
    targets = np.array([1.0, 2.0, 0.0, 0.0, 0.0, -1.0])
    _, lam, eta = gate(alpha, m, q, targets=targets)
    est = eta / lam
    ok("5.2 a gap leaves the estimate unchanged", close(est[2], est[4]))
    ok("5.2 …and lowers the weight behind it", bool(lam[4] < lam[2]))

    # 5.3 The size of the drift separation, measured rather than asserted: an
    # estimate built from a geometric discount, compared with the filter's, on a
    # stream whose probes sit at the estimate's own edge.
    def stream(rng, n_tokens, q_gap=0.5):
        syms, lam, eta, level = [], 0.0, 0.0, rng.normal()
        while len(syms) < n_tokens:
            for _ in range(rng.integers(3, 13)):
                x = round((level + rng.normal()) * 4) / 4
                syms.append(("v", x))
                lam, eta = lam + 1, eta + x
            k = rng.integers(1, 7)
            for _ in range(k):
                syms.append(("gap", 0.0))
                d = 1 / (1 + q_gap * lam)
                lam, eta = lam * d, eta * d
            level += 0.8 * np.sqrt(k) * rng.normal()
            x = round((level + rng.normal()) * 4) / 4
            syms.append(("v", x))
            lam, eta = lam + 1, eta + x
            mean = eta / lam if lam > 0 else 0.0
            probe = round((mean + (0.13 if rng.random() < 0.5 else -0.13)) * 4) / 4
            syms.append(("v", probe))
            lam, eta = lam + 1, eta + probe
        return syms[:n_tokens]

    rng = np.random.default_rng(7)
    syms = stream(rng, 6000)
    lam = eta = 0.0
    labels = []
    for kind, x in syms:
        if kind == "gap":
            d = 1 / (1 + 0.5 * lam)
            lam, eta = lam * d, eta * d
            labels.append(None)
        else:
            lam, eta = lam + 1, eta + x
            mean = eta / lam
            labels.append(None if abs(x - mean) < 1e-2 else (x > mean))

    best = 0.0
    for a_gap in np.linspace(0.0, 1.0, 21):
        for b_val in (1.0, 0.97, 0.95, 0.9):
            s = c = 0.0
            hit = tot = 0
            for (kind, x), lab in zip(syms, labels):
                if kind == "gap":
                    s, c = s * a_gap, c * a_gap
                else:
                    s, c = b_val * s + x, b_val * c + 1
                if lab is not None:
                    tot += 1
                    hit += (x * c - s > 0) == lab
            best = max(best, hit / tot)
    print(f"     best geometric-discount arm on a knife stream: {100 * best:.2f}% (the filter is exact)")
    ok("5.3 the filter's edge over a geometric discount is real but small",
       0.9 < best < 0.995)

    # 5.4 What the tropical member does *not* buy: a maximum is a sum in the
    # exponential domain, so a plain linear state with write exp(S*v), decay 1
    # and the same cross-multiplied comparison decides "is v a new maximum"
    # exactly — no logarithm anywhere. The register's advantages over it are
    # range (it spans S*v where this spans exp(S*v)) and, for a counter with a
    # floor, growth: `a > 0` is a decay above one, which `alpha = exp(dt*A) <= 1`
    # cannot be.
    vals = RNG.integers(1, 7, 400).astype(float)
    scale = np.log(len(vals)) + 1.0
    u = np.exp(scale * (vals - 6.0))
    h, exact_hits = 0.0, 0
    for t, x in enumerate(u):
        record = vals[t] > np.max(vals[:t]) if t else True
        exact_hits += (x > h) == record
        h += x
    ok("5.4 a linear state with an exponential write decides 'new maximum' exactly",
       exact_hits == len(vals))
    print(f"     that arm's span is exp(S*(v_max - v_min)) = {np.max(u) / np.min(u):.1e}"
          f" — f32 holds ~1e7, which is the range caveat")


if __name__ == "__main__":
    check_semiring()
    check_kalman()
    check_tropical()
    check_ports()
    print(f"\n{PASSED} passed, {FAILED} failed")
    sys.exit(1 if FAILED else 0)
