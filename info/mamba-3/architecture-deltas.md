# Architecture Deltas

### What Mamba-3 changed outside the SSM core, and what each change bought

> Reference note for `burn-mamba`. It is the authority for the decisions of
> this crate around the parts of the Mamba-3 block that are *not* the
> recurrence:
>
> - BCNorm (`b_norm`/`c_norm`),
> - the `B`/`C` biases (`b_bias_hmr`/`c_bias_hmr`),
> - the absent short convolution,
> - the optional output norm (`has_outproj_norm`),
> - the data-dependent `A` (`dd_A`, `a_floor`).
>
> It is the companion of the trio that classifies the recurrence itself:
> [`trapezoid-as-integration.md`](trapezoid-as-integration.md),
> [`rotation-as-optimization.md`](rotation-as-optimization.md) and
> [`mimo-as-batch.md`](mimo-as-batch.md). Those three cover the linear term,
> the quadratic term, and the rank. This note has every Mamba-3 change that is
> *not* one of those three. The division is by *what the note classifies*:
>
> - The trio takes the **algebra** of the state update and never leaves it.
> - This note takes the block around it, plus the **parameterisation** of the
>   scalar channels that the trio takes as given.
>
> §§5–6 are where that second half reaches into the recurrence: for the sign of
> the taps of the trapezoid, and for the locus that data-dependent `A` opens.
> Neither changes an algebraic claim of the trio.
>
> [`scripts/mamba-3/architecture_deltas.py`](../../scripts/mamba-3/architecture_deltas.py)
> checks every numbered claim below in float64 (35 checks, with the same
> section numbers). The script depends only on `numpy` and on the definitions
> in this note, not on the crate.

---

## Abstract

The three core changes get the method sections of the paper. The block around
them was rebuilt at the same time, and the rebuild is not incidental. Mamba-3:

- deletes the short causal convolution and its activation,
- deletes the post-gate gated RMSNorm of Mamba-2,
- adds RMS normalisation on `B` and `C`,
- adds a learnable per-(head, channel) bias to each of them,
- makes `A` data-dependent.

Four results:

1. **At initialisation, a Mamba-3 layer is a convolution.** BCNorm pins
   `‖B‖ = √N`. So the ones-initialised biases add a score term of exactly
   `N`, and the data adds a term of standard deviation `√N` (§4.2). The read
   is thus the *bias-only* read (a decay-weighted causal average) to within
   `O(1/√N)` (§4.10). This is the quantitative content of the sentence of the
   source "we hypothesize that these biases induce a convolution-like
   behavior". It also predicts the ablation table of the source. The floor is
   `⟨b_C, b_B⟩`: `N` at the ones init, `N/4` at `U(0,1)`, and `0` at the zero
   init and at the symmetric init. This is the same order as the measured
   perplexities (§4.6).
2. **The bias is added before the rotation, so the floor is a locality prior,
   not a constant.** The bias–bias term is
   `(N − rope_dim) + 2Σⱼcos(θ_{t,j} − θ_{s,j})` (§4.7). It is maximal at zero
   rotation-lag, and over large lags its average is the unrotated remainder
   `N − rope_dim` (§4.9). A bias added *after* the rotation gives the flat
   constant `N` (§4.8). So `rope_fraction` is also the fraction of the
   convolutional floor that the rotation localises. This couples two dials
   that look independent.
3. **The two replacements for the short convolution are both low-pass.** The
   trapezoid is exactly a width-2 filter on the *state-input* `B_tx_t` (§5.1).
   Its taps `β, γ` are strictly positive for every input (§5.2), and the bias
   floor is positive by construction. A learned depthwise conv can have a
   negative tap, but the pair that replaced it cannot. What Mamba-3 gave up is
   the difference of adjacent tokens, not their smoothing.
4. **Data-dependent `A` buys exactly one degree of freedom per (token,
   head).** In Mamba-2, `(α, γ) = (e^{ΔA_h}, Δ)` traces a one-dimensional locus
   per head (§6.1). One scalar sets the memory horizon *and* the write weight:
   this is the tying that its own preliminaries describe. The locus of Mamba-3
   is two-dimensional (§6.2), with the cap `α ≤ e^{−a_floor·Δ}` from the floor
   (§6.3). The source reports the change as performance-neutral, made for
   consistency. So it is a choice of parameterisation, not a claimed win.

§7 gives the mechanism behind the table of output-norm placements, which the
source calls "unintuitive". §8 is the one section that changed the library
instead of describing it. The default chunk length divides the square-root
rule by `mimo_rank` (the advice of the source). Then it only *subdivides* it by
`micro_steps`, because the two dials widen a chunk along different axes.

---

## 1. Scope and results

| § | claim |
|---|---|
| 2 | The two blocks side by side, and what the source measured. |
| 3 | BCNorm pins the scale of the score to the shape, not to the learned scale of the projections. |
| 4 | The biases make the score affine. At init the data-independent part dominates, and the rotation localises it. |
| 5 | The short convolution and its two replacements. Both replacements are strictly positive filters. |
| 6 | Data-dependent `A` unties the decay from the write weight, up to the floor cap. |
| 7 | What each output-norm placement erases. |
| 8 | The chunk-length schedule: why `mimo_rank` divides it (FLOPs), and why `micro_steps` only subdivides it (the read axis of a chunk). |
| 9 | Consequences for this crate. |

§§3–7 propose no change of behaviour. They document three existing defaults
and give the reasons for them. §8 is the exception. It is the argument for the
divisor of `Mamba3SsdPath::optimal_chunk_len`, which changes the default chunk
length for blocks with `mimo_rank > 1` or `micro_steps > 1`. The exact values
do not change, and a SISO, `u = 1` block does not change.

---

## 2. Setup and conventions

### 2.1 The two blocks

`N` is the state rank, `P` the head dimension, `G` the number of `B`/`C`
groups (multi-value attention: in both families, the heads of a group share
`B`/`C`). The note uses one head throughout.

| | Mamba-2 | Mamba-3 |
|---|---|---|
| in-projection | `[z ǀ x ǀ B ǀ C ǀ Δ]` | `[z ǀ x ǀ B ǀ C ǀ Δ ǀ A ǀ λ ǀ μ* ǀ θ]` |
| short conv | depthwise, width 4, over `x‖B‖C` | **none** |
| activation on `x`, `B`, `C` | `SiLU` after the conv | **none** |
| `B`, `C` normalisation | none | **RMSNorm over `N`**, then a learnable bias, then the rotation |
| `A` | one learned scalar per head | **projected per token**, `A = −softplus(·)` clamped `≤ −a_floor` |
| discretisation | `h = αh₋₁ + Δ·Bx` | trapezoidal `h = αh₋₁ + βB₋₁x₋₁ + γBx` |
| transition | real | real × rotation (RoPE trick) |
| rank | SISO | `mimo_rank ≥ 1` |
| output | gated RMSNorm(`y`, `z`) → out-proj | `y ⊙ SiLU(z)` → out-proj, with an **optional** norm |
| skip | `D ⊙ x` per head | unchanged |

`*` `μ` is a segment of this crate, not of the source. It is the mass that
splits the left endpoint between two taps. Only a two-tap `Trapezoid` has it
([`trapezoid-as-integration.md`](trapezoid-as-integration.md) §9), and it
folds away at `micro_steps = 1`. Every per-micro-step segment widens by `u`.
`z` and `C` do not.

Rows 2–4 together have a consequence that is worth a separate statement:
**the only nonlinearities left on the `B`/`C`/`x` path are the normalisation
itself and the squashing of the scalar channels** (`softplus` on `Δ` and `A`,
`σ` on `λ` and `μ`, `tanh` on `θ`). Values reach the SSD core as raw
projections. The `SiLU` of the gate is the only pointwise activation of the
block.

The `B`/`C` path, in the order of the reference kernel (`mamba3.py` →
`mamba3_siso_fwd.py`) and of this crate (`helpers::qk_norm_expand_bias`, then
`rotate_bc_forward`):

$$\tilde B_t=\mathrm{rope}\big(\mathrm{rmsnorm}(B^{\text{raw}}_t)+b_B\big),\qquad
\tilde C_t=\mathrm{rope}\big(\mathrm{rmsnorm}(C^{\text{raw}}_t)+b_C\big)$$

with `b_B, b_C ∈ ℝ^{H×M×N}` learnable, initialised to **ones**. The norm is
first, the bias second, the rotation third. §4.7–4.8 show that the third
boundary is load-bearing, and §9.1 records that the crate agrees with it.

### 2.2 What the source measured

All architecture ablations are 440M parameters at Chinchilla-optimal tokens,
with the FineWeb-Edu test perplexity. They are separate runs, so the two "no
bias" rows below do not agree exactly (16.49 vs 16.52). Read the order, not
the digits.

| variant | ppl ↓ | | bias init | ppl ↓ | | `B` bias | `C` bias | ppl ↓ |
|---|---|---|---|---|---|---|---|---|
| − bias − trapezoid | 16.68 | | `1.0`, trained | **15.72** | | ✗ | ✗ | 16.52 |
| − bias | 16.49 | | `1.0`, frozen | 15.80 | | ✓ | ✗ | 16.68 |
| **Mamba-3** | **15.72** | | `U(0,1)` | 15.76 | | ✗ | ✓ | 15.98 |
| **+ short conv** | 15.85 | | `U(−1,1)` | 16.07 | | ✓ | ✓ | **15.69** |
| | | | `0.0` | 16.57 | | | | |

The tables directly support three readings:

- To add the short convolution again to the finished model *hurts* (15.85 vs
  15.72). It is redundant, not only optional.
- A frozen ones-bias recovers most of the gain (15.80). So the bias does more
  work as an initialisation than as a learned parameter.
- The biases are synergistic: `B` alone is worse than neither.

The source also reports these results, which this note does not derive again:

- Data-dependent RoPE takes parity from 0.90 → 100.0, and bracketed modular
  arithmetic from 0.88 → 87.75, where standard RoPE and no RoPE both fail.
- MIMO at `R = 4` adds 1.2 points of downstream average over SISO at matched
  parameters.
- Mamba-3 at half the state size matches the perplexity of Mamba-2.
- Mamba-3 (SISO) decodes slightly *faster* than Mamba-2, although it has more
  terms. The source attributes this to its kernels, and does not divide it per
  component.

---

## 3. BCNorm pins the scale of the score

RMSNorm over the `N` axis with unit gain sends any `B ≠ 0` to a vector of RMS
1, so `‖B‖ = √N` (§3.1). It is invariant to any positive rescaling of the
projection that made it (§3.2). So the score `C^⊤B` is no longer a product of
two learned scales. Under a `10×` reparameterisation of both projections, the
unnormalised score moves by `100×`, and the normalised one does not move
(§3.3). The shape fixes the magnitude of the score: mean `0`, standard
deviation `√N` over random data (§3.4).

That is the whole mechanism behind the stability claim of the source ("BCNorm
is also able to stabilize large-scale runs, resulting in the removal of the
post-gate RMSNorm"). Mamba-2 controlled the output scale of the block at the
*end*, after the SSD. Mamba-3 controls the two factors that set it at the
*start*. The state is a sum of outer products of `B` with values, and the read
pairs it against `C`. So to pin both operands pins the whole path, except the
values. When that exception matters, the optional output norm (§7) is there
for it.

It also makes §4 possible: a bias of fixed size has a meaning only against an
operand of fixed size. Bias and norm are one change, not two.

---

## 4. The biases: an affine score, and a convolution at init

With `B̃ = B + b_B` and `C̃ = C + b_C`, the score expands into four terms
(§4.1):

$$\tilde C_t^\top\tilde B_s=\underbrace{C_t^\top B_s}_{\text{data-data}}
+\underbrace{C_t^\top b_B}_{\text{query-side}}
+\underbrace{b_C^\top B_s}_{\text{key-side}}
+\underbrace{b_C^\top b_B}_{\text{constant}}$$

Three of them are `O(√N)`. The fourth, at the ones init, is exactly `N`
(§4.2). The asymmetry is the point:

- At `N = 128` the score is within ~30% of the constant `N` (§4.3).
- That deviation decreases as `1/√N` (§4.4).
- Every (query, key) pair scores **positive**, where the unbiased score is
  positive half the time (§4.5).

Through the decay mask, the read of the layer at initialisation is the
bias-only read, to within `O(1/√N)` (§4.10):

$$y_t\;\approx\;N\sum_{s\le t}\Big(\textstyle\prod_{i=s+1}^{t}\alpha_i\Big)\gamma_s\,v_s$$

This is a causal, data-independent, exponentially-weighted moving average: a
convolution whose kernel is the decay profile. The layer *starts* as a
smoother and learns its content-dependence on top of that. It does not start
from a zero-mean score that must find locality.

**The ablation table follows from the constant term.** `⟨b_C, b_B⟩` is `N`
for the ones init, `N/4` for `U(0,1)`, and `0` for the zero init and (in
expectation) for `U(−1,1)` (§4.6). That is a *weak* order. The measured
`15.72 < 15.76 < 16.07 < 16.57` refines it and does not contradict it. The
floor ranks the first two and ties the last two, and variance breaks the tie.
A symmetric init keeps the two `O(√N)` cross terms of the ones init, but none
of its floor. The zero init adds neither. The weak order carries the
summary of the source, that the model "is not very sensitive to the
initialization of the biases as long as they are positive". Positivity is not
a convention: without it, the four-term expansion has no floor.

**The rotation localises the floor.** The bias is added before the rotation.
So each bias vector is rotated by its own cumulative angles, and the constant
term becomes (§4.7):

$$b_C^\top R_t^\top R_s b_B=(N-\texttt{rope\_dim})+2\sum_{j}\cos\big(\theta_{t,j}-\theta_{s,j}\big)$$

It is maximal at zero rotation-lag, and its average decays to the unrotated
remainder `N − rope_dim` as the lag grows (§4.9). A bias added after the
rotation gives the flat constant `N` at every lag (§4.8). This has two
consequences:

- The convolutional floor is a *locality* prior in the time of the
  accumulated rotation, not a uniform prior.
- `rope_fraction` sets how much of the floor is localised. At
  `rope_fraction = 1` no flat component survives. At `0.5` (the reference)
  half of it survives.

That is a real coupling between the bias and the rotation, and only the order
causes it.

**Why `C` matters more than `B`.** In the presence table, `C`-only (15.98)
is better than `B`-only (16.68), which is *worse* than neither. The expansion
shows at least that the two are not interchangeable:

- `b_B` enters the state. The block writes it once per token, and then it
  decays with everything else. So `b_B` alone gives every write a
  data-independent direction, with no matching probe.
- `b_C` enters the read, and the block applies it fresh at every step to the
  whole accumulated state. So `b_C` alone gives every read a data-independent
  probe of the running sum.

This is a reading of the asymmetry, not a derivation of it. Nothing here
predicts the *sign* of the `B`-only row, and the checks in §4 do not cover it.

---

## 5. The short convolution, and its two replacements

Mamba-2 (and GDN, and most linear models) applies a depthwise causal
convolution of width 4, plus `SiLU`, to `x‖B‖C` before the recurrence.
Mamba-3 has neither, and to add the convolution again makes the model worse
(15.85 vs 15.72). The source attributes this to the trapezoid plus the
biases. §4 priced the biases. The half of the trapezoid is its own definition,
written again:

$$h_t=\alpha_th_{t-1}+\beta_tv_{t-1}+\gamma_tv_t
=\alpha_th_{t-1}+\tilde v_t,\qquad \tilde v_t=\gamma_tv_t+\beta_tv_{t-1}$$

So the three-term recurrence is a two-term recurrence that runs on a width-2
filtered state-input (§5.1). The filter is different from the deleted
convolution on three axes at the same time:

- It is **data-dependent** (`β`, `γ` are per-token functions of `Δ`, `A`,
  `λ`). The conv is fixed.
- It acts on `B_tx_t` **inside** the recurrence. The conv acts on `x_t`
  outside it.
- It is one scalar pair per (head, step). The conv is a separate 4-tap kernel
  per channel.

[`trapezoid-as-integration.md`](trapezoid-as-integration.md) §§4–5 derives
`β` and `γ`. Cite it: this note needs only their sign.

**Both replacements are strictly positive.** `β = (1−λ)Δα` and `γ = λΔ`, with
`λ ∈ (0,1)`, `Δ > 0`, `α > 0`. So the taps are positive for every input
(§5.2). Their ratio is unbounded, so the pair sweeps the whole positive
quadrant and nothing else (§5.3). The bias floor is positive by §4. So the two
mechanisms that replaced the convolution are both **low-pass**. They can
average adjacent state-inputs in any proportion, but they cannot subtract one
from the other. A learned depthwise kernel can. This is the one capability
that the deletion really costs. It is worth an exact name, because "the
convolution is redundant" is a claim about a trained 440M language model, not
about the function classes.

---

## 6. Data-dependent `A`

In Mamba-2, the per-head `A_h` is learned and input-independent. So the pair
that the recurrence really uses is:

$$(\alpha_t,\gamma_t)=\big(e^{\Delta_tA_h},\ \Delta_t\big)$$

This is a **one-dimensional** locus per head (§6.1): the Jacobian in the
single free variable `Δ_t` has rank 1. The preliminaries of the source
describe the consequence as a feature: "a larger `Δ_t` forgets faster and
up-weights the current token more strongly". It is a feature, but it is also a
constraint: a token cannot ask for a strong write *and* a long memory.

Mamba-3 projects `A_t` per token, `A_t = −softplus(\cdot)`, clamped to
`≤ −a_floor`. The locus becomes two-dimensional (§6.2): exactly one more
degree of freedom per (token, head). The floor adds one inequality (§6.3):

$$\alpha_t\le e^{-\texttt{a\_floor}\cdot\gamma_t}$$

At fixed `γ`, the reachable `α` is the half-open interval
`(0, e^{−a_floor·γ}]`, and the `A` projection alone sweeps it (§6.4). On a
token stream, tokens with the same write weight (to within 1%) have an `α`
spread of `~1.0` under Mamba-3, and of `~0.007` under Mamba-2 (§6.5).

The source says that data-dependent `A` performs *similarly* to
data-independent `A`, and that it chose it "for consistency so that all SSM
parameters are data-dependent". That is the honest framing to keep. It is a
choice of parameterisation with a clean interpretation, not one of the three
measured advances. A comparison of Mamba-2 with Mamba-3 must not attribute
anything to it.

---

## 7. What each output-norm placement erases

Mamba-2 ends its block with a gated RMSNorm over `d_inner`, applied *after*
the gate. Mamba-3 removes it and ends with the bare `y ⊙ SiLU(z)`. For hybrid
models, the source adds it again, as a **pre-gate, per-head-grouped** norm. It
reports that this buys length-generalised retrieval, at a small cost in
in-context retrieval, and it does not call any row of its table the best. The
three placements are not variants of one operation. Each erases a different
thing:

| placement | rescaling of `y` | rescaling of the gate `SiLU(z)` | check |
|---|---|---|---|
| post-gate (Mamba-2) | erased | **erased** | §7.1, §7.3 |
| pre-gate (Mamba-3 optional) | erased | kept | §7.2, §7.3 |
| none (Mamba-3 default) | kept | kept | §7.4 |

The middle row is the interesting one. A norm before the gate keeps the
*magnitude* of the gate as a live signal. A norm after the gate keeps only its
direction. A retrieval head needs a gate that can attenuate its own output, to
suppress a position. A norm after the gate removes exactly that. This is a
mechanism for the observation of the source, not a prediction of its table.

Grouping is the second axis, and it is independent. A per-head norm makes the
scales of the heads equal. One norm over `d_inner` keeps their spread (§7.5).
A model whose heads have intentionally different magnitudes loses that under
grouping. A model where the scale of a single head can grow without limit is
safe under grouping.

The default `has_outproj_norm = false` is the pure-Mamba-3 setting. It is
correct for the default of this crate, because BCNorm (§3) does the
stabilisation upstream.

---

## 8. Chunk length under MIMO and `micro_steps`

The chunked algorithm of SSD costs `intra + inter` FLOPs per chunk. At
`C = N = P` and rank 1, the total is `≈ 8TN²` (§8.1). Under MIMO, the
intra-chunk term scales with `(CR)²` (exactly `R²` at fixed `C`, §8.3), and
the inter-chunk term scales with `R`. The remedy of the source is to halve the
chunk as the rank grows, `C_MIMO = C_SISO / R`. This makes the total
`≈ 8TRN²` again, that is, `R×` SISO and not `R²×` (§8.2). Its reference
implementation gives the same advice in a comment on the `chunk_size`
argument ("64 for SISO, 64/mimo_rank for MIMO"). At `N = P = 64`, `R = 4`,
the two schedules differ by `2.5×` (§8.4).

This crate fuses the rank onto the chunk axis: `[L·M, L·M]` intra-chunk
matrices in `single_ssd/ssd/serial.rs`. (§8.7 gives the general `[T·M, L·M]`,
where `T` is the read rows of the chunk. At `u = 1`, the two are the same.) So
the same arithmetic applies without change, and
`Mamba3SsdPath::optimal_chunk_len` implements it: `√(N·P)` divided by
`mimo_rank`, then rounded up to a multiple of 32 and clamped to `32..=512`.
At `N = 128`, `P = 64`, that is `96 / 64 / 32` for a rank of `1 / 2 / ≥4`
(§8.5). So the fused axis `C·M` is 128 for `M = 4`, where the unscaled chunk
puts it at 384 (§8.6). A SISO block does not change: the divisor is 1.

**`micro_steps` is the other widening, and it is a different one.**
Micro-steps fold into the sequence axis, but they fold only the *writes*.
Each of the `u` micro-steps of a token writes to the state, and the SSD
**reads** the token once, at its last micro-step. So a chunk has two axes:
`C` writes by `C/u` reads. Its score is `[batch, nchunks, heads, (C/u)·M,
C·M]`, that is, `batch · sequence · heads · C · M²` elements, with no `u` in
it (§8.7). The rank is on *both* axes, so it divides the chunk. `u` is on one
axis, so it only subdivides the chunk. Hence the second half of the schedule:
round the width up to a multiple of `u`. Then a chunk is a whole number of
tokens, and its read rows are a contiguous run of them: a reshape, not a
gather.

The alternative is to also divide by `u`, because a chunk of `C` positions
holds `C/u` tokens. It is worse on both counts:

- FLOPs do not prefer either schedule: `u` multiplies the intra- and
  inter-chunk terms alike, and cancels out of the balance.
- A chunk that shrinks as `1/u`, over a sequence that grows as `u`, puts
  `nchunks` at `u²`. That doubles the exponent of the serial scan of the
  current schedule, and the backward of both pathways walks it chunk by
  chunk.
- It does not even buy the memory that it costs. The 32 grid floors the
  division, so past a fold of 3 the chunk stops shrinking, and the score grows
  with `u` in any case: `2.67×` at `u = 8`, where a fixed width gives exactly
  `1×` (§8.8).

The write side stays linear in `u`, and this cannot be reduced: the state
recurrence really takes `u` steps per token. The read side is `u`-invariant in
FLOPs and in memory: the score, the state-to-output product, and the QK-norm
and rotation of `C`. The block evaluates it only where the readout occurs
(`helpers::read_rows`, and the kernels take the stride as
`Mamba3*SsdInput::read_stride`). A model that wants a different trade can
give an explicit chunk length (`Mamba3SsdPath::SerialRecalculated(Some(n))`).
It has priority over the schedule, and the block rounds it up to a whole
number of tokens.

Two caveats on the whole exercise:

- `optimal_chunk_len` is a rule of thumb about matmul shapes on real
  backends, not a FLOP minimiser. For this reason it keeps the 32 grid and its
  floor, and does not follow the division down to 24.
- The FLOP model above is that of the source, measured against fused GPU
  kernels. On some backends, the kernel-launch count dominates the cost of
  this crate (`kernels.sh`). There, fewer and larger chunks can win, against
  the FLOPs and the memory.

The schedule is the defensible default, not a measured optimum. `bench.sh` is
the tool that can settle it.

---

## 9. Consequences for this crate

### 9.1 The order of the `B`/`C` path is load-bearing

The order is `RMSNorm → GQA-expand → bias → rotation`. The crate does this
with `helpers::qk_norm_expand_bias` followed by `rotate_bc_forward`, in both
pathways. The reference does the same with `mamba3.py` + `mamba3_siso_fwd.py`
(the norm in the module, the bias and then the rotary inside the kernel):

- The bias–rotation boundary changes the behaviour (§4.7 vs §4.8). To move the
  bias after the rotation replaces a locality prior with a flat one.
- The norm–bias boundary matters for the same reason that BCNorm and the
  biases are one change (§3): a bias of size 1 is calibrated against an
  operand of RMS 1.

### 9.2 The ones-init is a mechanism, not a placeholder

`Initializer::Ones` on `b_bias_hmr`/`c_bias_hmr` makes the block start as a
convolution (§4.10). It is measurably better than zeros (15.72 vs 16.57). Do
not change it to a zero-mean initialiser. Keep both biases, not only the bias
of `B` (§4, last paragraph). They are 3-D per-head parameters and stay on
AdamW: `muon_projections()` names only the projection matrices, as it should.

### 9.3 The optional output norm is one row of the table of the source

`has_outproj_norm = true` builds `RmsNormGatedConfig::new(per_head_dim)` with
`norm_before_gate(true)`: pre-gate and per-head-grouped. That is exactly the
"Pre-Gate Grouped RMS" row of the source, the row that it recommends for
hybrids. The config cannot reach the other three rows: `Mamba3Config::init`
fixes both the placement and the grouping. The norm of Mamba-2
(`is_norm_before_gate = false`, over `d_inner`) is the "Post-Gate Default RMS"
row. So the crate spans the two ends of that table across the two families.
The source explicitly leaves open whether Mamba-3 should expose the other two
rows, and nothing here argues for it.

### 9.4 The chunk schedule divides by `mimo_rank`, and only subdivides by `micro_steps`

See §8. `Mamba3SsdPath::optimal_chunk_len` divides the square-root rule by the
rank, rounds it up to a multiple of 32, and then rounds it *up* to a whole
number of tokens. The rank half is the advice of the source (its kernels
carry it as a comment on `chunk_size`). The different treatment of
`micro_steps` is from this crate. It follows because the chunk has a read
axis that the rank widens and `u` does not (§8). For a SISO, `u = 1` block,
the default path is the same byte for byte.

This is the one place where the note changed behaviour instead of describing
it. It is a heuristic that changes a heuristic: the exact values do not
change, and the cost moves. `bench.sh` can confirm the direction on a given
backend.

### 9.5 What belongs to the caller, not to this crate

The macro-architecture of the source is Llama-style: Mamba-3 blocks that
alternate with SwiGLU MLPs under pre-norm. Hybrids use a 5:1 interleave with
NoPE self-attention. MIMO models are parameter-matched with a smaller MLP
width (4096 → 3824 at 1.5B). None of that is the business of this crate:
`burn-stack` owns the layer composition, and it is block-agnostic by
construction. Two training-side details are easy to lose, so this note
records them:

- The reference marks `dt_bias` and `D` as no-weight-decay. It does **not**
  mark the `B`/`C` biases. So under the usual convention, weight decay pulls
  them toward the zero init, which ablates worst. The source does not comment
  on this.
- The kernels that the latency tables measure are fused
  Triton/TileLang/CuTe. That is the axis that this crate intentionally gives
  up (`CLAUDE.md` → *No optimized kernels*).

### 9.6 What does not follow

- Not that the convolution is useless in general (§5). It is redundant
  *given* the trapezoid and the biases, in a trained 440M LM. The function
  classes are different, and the difference has a sign.
- Not that the biases are an attention-sink mechanism. The floor is uniform
  over positions before the rotation modulates it (§4.7). A sink is
  position-selective. The resemblance is worth a note, but this note verified
  nothing about it.
- Not that data-dependent `A` contributes to the reported gains (§6). The
  source says that it does not.
- Not that the removal of the post-gate norm is safe at every scale (§7). The
  source adds it again for hybrids, and it reports competing trade-offs among
  the placements.

---

## 10. Reproduction

```
python3 scripts/mamba-3/architecture_deltas.py
```

Pure `numpy`, float64, 35 checks, with the same section numbers as this
document. It exits non-zero on failure. The script defines the `B`/`C` path of
the block and the coefficients of the trapezoid from scratch. It does not
import or call the crate. So a failure means that one of the statements above
is wrong, not that the implementation drifted.

---

## References

- Mamba-3 (`../papers/mamba/mamba-3/`):
  - §*Mamba-3 Architecture* (BCNorm, the biases, the removed convolution and
    post-gate norm),
  - the remark of the preliminaries on data-dependent `A`,
  - §*Multi-Input, Multi-Output* (the chunked FLOP count and the `C/R`
    schedule),
  - the architecture ablations and the hybrid norm table of the appendices.
- Reference implementation:
  - `../py/state-spaces/mamba/mamba_ssm/modules/mamba3.py` (module-level
    norm, bias init, `A_floor`, `is_outproj_norm`, the `chunk_size` advice),
  - `ops/triton/mamba3/mamba3_siso_fwd.py` (the bias and then the rotary,
    inside the kernel).
- Companion notes:
  - [`trapezoid-as-integration.md`](trapezoid-as-integration.md) (`β`, `γ`
    and the two-installment collapse),
  - [`rotation-as-optimization.md`](rotation-as-optimization.md) (the rotation
    and its data-dependence),
  - [`mimo-as-batch.md`](mimo-as-batch.md) (the rank, the value tying, and why
    the ranks share the rotation).
