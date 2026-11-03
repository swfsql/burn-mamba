# Check: official Add -> LN -> Mixer order, mamba3-minimal order and our delta fold give one value.
import numpy as np
rng = np.random.default_rng(1)
n, d, layers = 8, 4, 5
M = [rng.normal(size=(d, d)) for _ in range(layers)]
F = [rng.normal(size=(d, d)) for _ in range(layers)]
norm = lambda v: v / np.sqrt((v * v).mean(-1, keepdims=True) + 1e-5)
mixer = lambda i, v: np.tanh(v @ M[i])
mlp = lambda i, v: np.tanh(v @ F[i])
x0 = rng.normal(size=(n, d))
# official Block chain with an mlp, then the last add of MixerModel.forward
hs, res = x0, None
for i in range(layers):
    res = hs + res if res is not None else hs
    hs = mixer(i, norm(res))
    res = hs + res
    hs = mlp(i, norm(res))
official = hs + res
# mamba3-minimal
x = x0
for i in range(layers):
    x = mixer(i, norm(x)) + x
    x = x + mlp(i, norm(x))
minimal = x
# our Layer delta + Layers add
x = x0
for i in range(layers):
    h1 = mixer(i, norm(x))
    h2 = mlp(i, norm(x + h1))
    x = (h1 + h2) + x
ours = x
print(np.abs(official - minimal).max(), np.abs(ours - minimal).max())
