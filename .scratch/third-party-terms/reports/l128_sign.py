#!/usr/bin/env python3
"""L128: the sign of the accumulated angle and the direction of the rotation.

A small float64 block: an in-projection, the trapezoid, the rotation of B and C.
`run` takes a full convention. Each check prints the max abs difference of the output
y against a reference. A value near 1e-13 means equal. A value near 1 means different.
"""
import numpy as np

D_MODEL, N, HEADS, P, T = 5, 8, 2, 3, 30
PI = np.pi


def pi_tanh(x):
    return PI * np.tanh(x)


def identity(x):
    return x


def not_odd(x):
    return PI * (np.tanh(x) + 0.25)


def rot(x, ang, layout, n_ang):
    """R(ang) on the pairs of the last axis, as all sources write it."""
    c, s = np.cos(ang), np.sin(ang)
    out = x.copy()
    if layout == "interleaved":
        i0, i1 = np.arange(n_ang) * 2, np.arange(n_ang) * 2 + 1
    else:
        i0, i1 = np.arange(n_ang), np.arange(n_ang) + N // 2
    out[..., i0] = c * x[..., i0] - s * x[..., i1]
    out[..., i1] = s * x[..., i0] + c * x[..., i1]
    return out


def make(seed, proj_bias):
    r = np.random.default_rng(seed)
    n_ang = N // 2
    widths = [HEADS * P, N, N, HEADS, HEADS, n_ang]
    W = r.normal(size=(sum(widths), D_MODEL)) / np.sqrt(D_MODEL)
    b = r.normal(size=sum(widths)) * 0.5 if proj_bias else np.zeros(sum(widths))
    u = r.normal(size=(T, D_MODEL))
    return dict(W=W, b=b, u=u, widths=widths,
                b_bias=r.normal(size=(HEADS, N)), c_bias=r.normal(size=(HEADS, N)))


def theta_slice(m):
    return slice(sum(m["widths"][:-1]), None)


def flip_theta(m, weights=True, bias=True):
    m = dict(m)
    W, b = m["W"].copy(), m["b"].copy()
    s = theta_slice(m)
    if weights:
        W[s] *= -1
    if bias:
        b[s] *= -1
    m["W"], m["b"] = W, b
    return m


def flip_pair_second(m, layout, n_ang):
    """S = diag(1, -1) on each pair, on B, C and their biases. Theta stays."""
    m = dict(m)
    idx = np.arange(n_ang) * 2 + 1 if layout == "interleaved" else np.arange(n_ang) + N // 2
    W = m["W"].copy()
    b = m["b"].copy()
    bb, cb = m["b_bias"].copy(), m["c_bias"].copy()
    off_b = m["widths"][0]
    off_c = off_b + N
    W[off_b + idx] *= -1
    W[off_c + idx] *= -1
    b[off_b + idx] *= -1
    b[off_c + idx] *= -1
    bb[:, idx] *= -1
    cb[:, idx] *= -1
    m["W"], m["b"], m["b_bias"], m["c_bias"] = W, b, bb, cb
    return m


def run(m, cum=+1, dir_b=+1, dir_c=+1, bound=pi_tanh, layout="interleaved", n_ang=N // 2,
        step_from=None, step_dir=None, wrap=None):
    """y of the block. After `step_from`, the step uses `step_dir` on B and C."""
    z = m["u"] @ m["W"].T + m["b"]
    w = np.cumsum(m["widths"])[:-1]
    x, B, C, dt, lam, th = np.split(z, w, axis=-1)
    x = x.reshape(T, HEADS, P)
    dt = np.logaddexp(0, dt) * 0.3
    lam = 1 / (1 + np.exp(-lam))
    A = -np.array([0.2, 1.1])
    th = th[:, :n_ang]
    h = np.zeros((HEADS, P, N))
    ang = np.zeros((HEADS, n_ang))
    prev = np.zeros((HEADS, P, N))
    ys = np.zeros((T, HEADS, P))
    for t in range(T):
        ang = ang + cum * dt[t][:, None] * bound(th[t])[None, :]
        if wrap == "floor":
            ang = ang - 2 * PI * np.floor(ang / (2 * PI))
        if wrap == "round":
            ang = ang - 2 * PI * np.round(ang / (2 * PI))
        db, dc = dir_b, dir_c
        if step_from is not None and t >= step_from:
            db, dc = step_dir, step_dir
        Bh = np.stack([rot(B[t] + m["b_bias"][i], db * ang[i], layout, n_ang) for i in range(HEADS)])
        Ch = np.stack([rot(C[t] + m["c_bias"][i], dc * ang[i], layout, n_ang) for i in range(HEADS)])
        alpha = np.exp(dt[t] * A)
        beta = (1 - lam[t]) * dt[t] * alpha
        gamma = lam[t] * dt[t]
        write = Bh[:, None, :] * x[t][:, :, None]
        h = alpha[:, None, None] * h + beta[:, None, None] * prev + gamma[:, None, None] * write
        prev = write
        ys[t] = np.einsum("hpn,hn->hp", h, Ch)
    return ys


def direct(m, rho_sign, bound=pi_tanh):
    """The direct form: h = alpha rho h + beta rho B_prev x_prev + gamma B x, y = C^T h."""
    z = m["u"] @ m["W"].T + m["b"]
    w = np.cumsum(m["widths"])[:-1]
    x, B, C, dt, lam, th = np.split(z, w, axis=-1)
    x = x.reshape(T, HEADS, P)
    dt = np.logaddexp(0, dt) * 0.3
    lam = 1 / (1 + np.exp(-lam))
    A = -np.array([0.2, 1.1])
    h = np.zeros((HEADS, P, N))
    prev = np.zeros((HEADS, P, N))
    ys = np.zeros((T, HEADS, P))
    for t in range(T):
        a = rho_sign * dt[t][:, None] * bound(th[t])[None, :]
        for i in range(HEADS):
            h[i] = rot(h[i], a[i], "interleaved", N // 2)
            prev[i] = rot(prev[i], a[i], "interleaved", N // 2)
        alpha = np.exp(dt[t] * A)
        beta = (1 - lam[t]) * dt[t] * alpha
        gamma = lam[t] * dt[t]
        Bh = B[t] + m["b_bias"]
        Ch = C[t] + m["c_bias"]
        write = Bh[:, None, :] * x[t][:, :, None]
        h = alpha[:, None, None] * h + beta[:, None, None] * prev + gamma[:, None, None] * write
        prev = write
        ys[t] = np.einsum("hpn,hn->hp", h, Ch)
    return ys


def d(a, b):
    return np.abs(a - b).max()


def line(name, value):
    print(f"{name:<74} {value:.3e}")


def checks_abelian():
    m = make(0, proj_bias=False)
    mb = make(1, proj_bias=True)
    off = run(m)
    print("Reference: official = +accumulated angle, R(+). Same weights unless stated.")
    line("A1 minimal (-accumulated, R(+)), same weights", d(run(m, cum=-1), off))
    line("A2 paper (+accumulated, R^T) vs minimal, same weights", d(run(m, dir_b=-1, dir_c=-1), run(m, cum=-1)))
    line("A3 minimal, flipped theta weights", d(run(flip_theta(m), cum=-1), off))
    line("A4 paper, flipped theta weights", d(run(flip_theta(m), dir_b=-1, dir_c=-1), off))
    offb = run(mb)
    line("A5 with a projection bias: minimal, flip weights and bias", d(run(flip_theta(mb), cum=-1), offb))
    line("A6 with a projection bias: minimal, flip weights only", d(run(flip_theta(mb, bias=False), cum=-1), offb))
    print("The bound. Each row compares with the official form that uses the same bound.")
    line("B1 identity bound (odd), minimal, flipped theta", d(run(flip_theta(m), cum=-1, bound=identity), run(m, bound=identity)))
    line("B2 non-odd bound, minimal, flipped theta", d(run(flip_theta(m), cum=-1, bound=not_odd), run(m, bound=not_odd)))
    s = flip_pair_second(m, "interleaved", N // 2)
    line("B3 non-odd bound, minimal, S on B and C, theta stays", d(run(s, cum=-1, bound=not_odd), run(m, bound=not_odd)))
    line("B4 pi*tanh vs identity, minimal, flipped theta (the L129 difference)", d(run(flip_theta(m), cum=-1, bound=identity), off))
    print("Consistency. A flip of theta cannot absorb a mixed direction.")
    mix = run(m, dir_b=+1, dir_c=-1)
    line("C1 B by R(+), C by R(-), no flip", d(mix, off))
    line("C2 B by R(+), C by R(-), flipped theta", d(run(flip_theta(m), dir_b=+1, dir_c=-1), off))
    tail = T // 2
    fwd_step = run(m, step_from=tail, step_dir=-1)
    line("C3 forward R(+), step R(-) (conjugate=True), no flip", d(fwd_step, off))
    line("C4 forward R(+), step R(-), flipped theta", d(run(flip_theta(m), step_from=tail, step_dir=-1), off))
    line("C5 forward R(+), step R(+) (conjugate=False)", d(run(m, step_from=tail, step_dir=+1), off))
    line("C6 minimal forward and step: forward R(+) on -angle, step same", d(run(m, cum=-1, step_from=tail, step_dir=+1), run(m, cum=-1)))
    print("Layout, partial rotation, wrap.")
    for layout in ("interleaved", "half"):
        for n_ang in (N // 2, N // 4):
            a = run(m, layout=layout, n_ang=n_ang)
            line(f"D1 {layout}, {n_ang} angles: minimal, same weights", d(run(m, cum=-1, layout=layout, n_ang=n_ang), a))
            line(f"D2 {layout}, {n_ang} angles: minimal, flipped theta", d(run(flip_theta(m), cum=-1, layout=layout, n_ang=n_ang), a))
    line("D3 wrap to [0, 2pi) (official) vs [-pi, pi] (ours)", d(run(m, wrap="floor"), run(m, wrap="round")))
    line("D4 no wrap vs wrap", d(run(m), run(m, wrap="round")))
    print("Direct form against the RoPE form.")
    line("E1 direct rho=R(+a) vs paper (R^T)", d(direct(m, +1), run(m, dir_b=-1, dir_c=-1)))
    line("E2 direct rho=R(+a) vs minimal", d(direct(m, +1), run(m, cum=-1)))
    line("E3 direct rho=R(+a) vs official and ours (R(+theta))", d(direct(m, +1), off))
    line("E4 direct rho=R(-a) vs official and ours (R(+theta))", d(direct(m, -1), off))


def checks_layout_permutation():
    """A fixed permutation of B and C maps the half layout to the interleaved layout."""
    m = make(0, proj_bias=False)
    perm = np.empty(N, dtype=int)  # half index for each interleaved index
    perm[0::2] = np.arange(N // 2)
    perm[1::2] = np.arange(N // 2) + N // 2
    p = dict(m)
    W = m["W"].copy()
    off_b = m["widths"][0]
    off_c = off_b + N
    W[off_b + perm] = m["W"][off_b + np.arange(N)]
    W[off_c + perm] = m["W"][off_c + np.arange(N)]
    bb, cb = m["b_bias"].copy(), m["c_bias"].copy()
    bb[:, perm] = m["b_bias"]
    cb[:, perm] = m["c_bias"]
    p["W"], p["b_bias"], p["c_bias"] = W, bb, cb
    line("F1 half layout with permuted B and C vs interleaved", d(run(p, layout="half"), run(m)))
    line("F2 half layout, same B and C vs interleaved", d(run(m, layout="half"), run(m)))


def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def qconj(q):
    return q * np.array([1, -1, -1, -1])


def qexp(g):
    a = np.linalg.norm(g)
    return np.concatenate([[np.cos(a)], np.sin(a) * g / a])


def checks_quaternion():
    """Non-abelian: the direction is not free, and a flip of the generators cannot absorb it."""
    r = np.random.default_rng(3)
    t = 20
    gs = r.normal(size=(t, 3)) * 0.5
    bs, cs, xs = r.normal(size=(t, 4)), r.normal(size=(t, 4)), r.normal(size=t)
    alpha = 0.9

    def direct_form(g):
        h = np.zeros(4)
        y = []
        for i in range(t):
            h = alpha * qmul(qexp(g[i]), h) + bs[i] * xs[i]
            y.append(cs[i] @ h)
        return np.array(y)

    def factored(g, use_conj):
        cum = np.array([1.0, 0, 0, 0])
        h = np.zeros(4)
        y = []
        for i in range(t):
            cum = qmul(qexp(g[i]), cum)
            q = qconj(cum) if use_conj else cum
            h = alpha * h + qmul(q, bs[i]) * xs[i]
            y.append(qmul(q, cs[i]) @ h)
        return np.array(y)

    ref = direct_form(gs)
    line("G1 quaternion: direct L_q vs factored conj(Q) (our code)", d(factored(gs, True), ref))
    line("G2 quaternion: direct L_q vs factored Q (no conj)", d(factored(gs, False), ref))
    line("G3 quaternion: factored conj(Q(g)) vs factored Q(-g) (flip of generators)", d(factored(-gs, False), factored(gs, True)))
    fixed_axis = np.tile(np.array([1.0, 0, 0]), (t, 1)) * r.normal(size=(t, 1)) * 0.5
    line("G4 one fixed axis (abelian): conj(Q(g)) vs Q(-g)", d(factored(-fixed_axis, False), factored(fixed_axis, True)))


def checks_training():
    """Equal losses along SGD with momentum when the start differs by a flip of theta."""
    def loss(m, cum):
        return np.mean((run(m, cum=cum) - target) ** 2)

    def grad(m, cum):
        g = np.zeros_like(m["W"])
        e = 1e-6
        for i in range(m["W"].shape[0]):
            for j in range(m["W"].shape[1]):
                a, c = dict(m), dict(m)
                a["W"], c["W"] = m["W"].copy(), m["W"].copy()
                a["W"][i, j] += e
                c["W"][i, j] -= e
                g[i, j] = (loss(a, cum) - loss(c, cum)) / (2 * e)
        return g

    target = np.random.default_rng(9).normal(size=(T, HEADS, P))
    m0 = make(5, proj_bias=False)
    a, c = dict(m0), flip_theta(m0)
    va, vc = np.zeros_like(a["W"]), np.zeros_like(a["W"])
    s = theta_slice(m0)
    for it in range(6):
        ga, gc = grad(a, +1), grad(c, -1)
        va, vc = 0.9 * va + ga, 0.9 * vc + gc
        a["W"] = a["W"] - 0.02 * va
        c["W"] = c["W"] - 0.02 * vc
    w = c["W"].copy()
    w[s] *= -1
    line("H1 after 6 SGD steps: official weights vs minimal weights with theta flipped back", np.abs(a["W"] - w).max())
    line("H2 after 6 SGD steps: official loss vs minimal loss", abs(loss(a, +1) - loss(c, -1)))
    line("H3 after 6 SGD steps: loss drop of the official run (not zero = it trained)", loss(m0, +1) - loss(a, +1))


if __name__ == "__main__":
    checks_abelian()
    checks_layout_permutation()
    checks_quaternion()
    checks_training()
