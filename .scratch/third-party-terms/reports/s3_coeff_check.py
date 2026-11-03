# Check: the official forward form (no decay in the scale) equals the paper
# recurrence (decay in beta), up to the accumulator offset.
import numpy as np
rng = np.random.default_rng(0)
T = 12
dt = rng.uniform(0.01, 2.0, T)
A = -rng.uniform(0.01, 2.0, T)
lam = rng.uniform(0.0, 1.0, T)
bx = rng.normal(size=T)
alpha = np.exp(dt * A)
beta = (1 - lam) * dt * alpha
gamma = lam * dt
# paper
h = np.zeros(T)
prev = 0.0
for t in range(T):
    h[t] = alpha[t] * prev + (beta[t] * bx[t - 1] if t > 0 else 0.0) + gamma[t] * bx[t]
    prev = h[t]
# official forward: scale = gamma_t + nu_{t+1}, with nu_T = 0, no decay in scale
nu_shift = np.append((dt[1:] * (1 - lam[1:])), 0.0)
scale = gamma + nu_shift
s = np.zeros(T)
prev = 0.0
for t in range(T):
    s[t] = alpha[t] * prev + scale[t] * bx[t]
    prev = s[t]
h2 = s - nu_shift * bx
print("max |h - (s - nu_shift*bx)|:", np.abs(h - h2).max())
print("s[-1] - h[-1]:", s[-1] - h[-1])
print("max |s - h| mid-sequence:", np.abs(s - h).max())
