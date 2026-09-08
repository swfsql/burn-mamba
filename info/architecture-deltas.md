# Architecture Deltas

### What Mamba-3 changed outside the SSM core, and what each change bought

> Reference note for `burn-mamba`. It is the authority for this crate's decisions
> around the parts of the Mamba-3 block that are *not* the recurrence: BCNorm
> (`b_norm`/`c_norm`), the `B`/`C` biases (`b_bias_hmr`/`c_bias_hmr`), the absent
> short convolution, the optional output norm (`has_outproj_norm`), and the
> data-dependent `A` (`dd_A`, `a_floor`).
>
> Companion to the trio that classifies the recurrence itself —
> [`trapezoid-as-integration.md`](trapezoid-as-integration.md),
> [`rotation-as-optimization.md`](rotation-as-optimization.md),
> [`mimo-as-batch.md`](mimo-as-batch.md). Those three cover the linear term, the
> quadratic term, and the rank. Everything Mamba-3 changed that is *not* one of
> those three is here, and the division is clean: this note never touches the state
> update, and the other three never touch the block around it.
>
> Every numbered claim below is checked in float64 by
> [`scripts/architecture_deltas.py`](../scripts/architecture_deltas.py) (35 checks,
> section numbers match). The script depends only on `numpy` and on the definitions
> reproduced here — not on the crate.

---

## Abstract

The three core changes get the paper's method sections; the block around them was
rebuilt at the same time, and the rebuild is not incidental. Mamba-3 deletes the
short causal convolution and its activation, deletes Mamba-2's post-gate gated
RMSNorm, adds RMS normalisation on `B` and `C`, adds a learnable per-(head,
channel) bias to each of them, and makes `A` data-dependent. Four results:

1. **At initialisation a Mamba-3 layer is a convolution.** BCNorm pins `‖B‖ = √N`,
   so the ones-initialised biases contribute a score term of exactly `N` while the
   data contributes a term of standard deviation `√N` (§4.2). The read is therefore
   the *bias-only* read — a decay-weighted causal average — to within `O(1/√N)`
   (§4.10). This is the quantitative content of the source's "we hypothesize that
   these biases induce a convolution-like behavior", and it predicts its own
   ablation table: the floor is `⟨b_C, b_B⟩`, which is `N` at the ones init, `N/4`
   at `U(0,1)`, and `0` at both the zero and the symmetric init — the same ordering
   as the measured perplexities (§4.6).
2. **The bias is added before the rotation, so the floor is a locality prior, not a
   constant.** The bias–bias term is `(N − rope_dim) + 2Σⱼcos(θ_{t,j} − θ_{s,j})`
   (§4.7): maximal at zero rotation-lag, averaging to the unrotated remainder
   `N − rope_dim` over large lags (§4.9). Had the bias been added *after* the
   rotation it would have been the flat constant `N` (§4.8). So `rope_fraction` is
   also the fraction of the convolutional floor that the rotation localises — a
   coupling between two dials that look independent.
3. **The two replacements for the short convolution are both low-pass.** The
   trapezoid is exactly a width-2 filter on the *state-input* `B_tx_t` (§5.1) whose
   taps `β, γ` are strictly positive for every input (§5.2), and the bias floor is
   positive by construction. A learned depthwise conv can have a negative tap; the
   pair that replaced it cannot. What Mamba-3 gave up is differencing adjacent
   tokens, not smoothing them.
4. **Data-dependent `A` buys exactly one degree of freedom per (token, head).**
   Mamba-2's `(α, γ) = (e^{ΔA_h}, Δ)` traces a one-dimensional locus per head
   (§6.1): one scalar sets the memory horizon *and* the write weight, which is the
   tying its own preliminaries describe. Mamba-3's locus is two-dimensional (§6.2),
   capped by the floor at `α ≤ e^{−a_floor·Δ}` (§6.3). The source reports the change
   as performance-neutral and made for consistency, so it is a parameterisation
   choice and not a claimed win.

§7 gives the mechanism behind the output-norm placement table the source calls
"unintuitive". §8 is the one section that changed the library rather than
describing it: the default chunk length divides the square-root rule by
`mimo_rank` — the source's own advice — and then only *subdivides* it by
`micro_steps`, the two dials widening a chunk along different axes.

---

## 1. Scope and results

| § | claim |
|---|---|
| 2 | The two blocks side by side, and what the source measured. |
| 3 | BCNorm pins the score's scale to the shape, not to the projections' learned scale. |
| 4 | The biases make the score affine; at init the data-independent part dominates, and the rotation localises it. |
| 5 | The short convolution and its two replacements; both are strictly positive filters. |
| 6 | Data-dependent `A` unties decay from write weight, up to the floor cap. |
| 7 | What each output-norm placement erases. |
| 8 | The chunk-length schedule: why `mimo_rank` divides it (FLOPs) and why `micro_steps` only subdivides it (a chunk's read axis). |
| 9 | Consequences for this crate. |

§§3–7 propose no behavioural change: they document, and give the reasons behind
three existing defaults. §8 is the exception — it is the argument behind
`Mamba3SsdPath::optimal_chunk_len`'s divisor, which changes the default chunk
length for `mimo_rank > 1` or `micro_steps > 1` blocks (exact values unchanged; a
SISO, `u = 1` block is untouched).

---

## 2. Setup and conventions

### 2.1 The two blocks

`N` is the state rank, `P` the head dimension, `G` the number of `B`/`C` groups
(multi-value attention: `B`/`C` are shared across the heads of a group in both
families). One head throughout.

| | Mamba-2 | Mamba-3 |
|---|---|---|
| in-projection | `[z ǀ x ǀ B ǀ C ǀ Δ]` | `[z ǀ x ǀ B ǀ C ǀ Δ ǀ A ǀ λ ǀ θ]` |
| short conv | depthwise, width 4, over `x‖B‖C` | **none** |
| activation on `x`, `B`, `C` | `SiLU` after the conv | **none** |
| `B`, `C` normalisation | none | **RMSNorm over `N`**, then a learnable bias, then the rotation |
| `A` | one learned scalar per head | **projected per token**, `A = −softplus(·)` clamped `≤ −a_floor` |
| discretisation | `h = αh₋₁ + Δ·Bx` | trapezoidal `h = αh₋₁ + βB₋₁x₋₁ + γBx` |
| transition | real | real × rotation (RoPE trick) |
| rank | SISO | `mimo_rank ≥ 1` |
| output | gated RMSNorm(`y`, `z`) → out-proj | `y ⊙ SiLU(z)` → out-proj; the norm is **optional** |
| skip | `D ⊙ x` per head | unchanged |

The consequence of rows 2–4 together is worth stating on its own: **the only
nonlinearities left on the `B`/`C`/`x` path are the normalisation itself and the
squashing of the scalar channels** (`softplus` on `Δ` and `A`, `σ` on `λ`, `tanh`
on `θ`). Values reach the SSD core as raw projections; the gate's `SiLU` is the
block's only pointwise activation.

The `B`/`C` path, in the order the reference kernel runs it (`mamba3.py` →
`mamba3_siso_fwd.py`) and this crate with it (`helpers::qk_norm_expand_bias`, then
`rotate_bc_forward`):

$$\tilde B_t=\mathrm{rope}\big(\mathrm{rmsnorm}(B^{\text{raw}}_t)+b_B\big),\qquad
\tilde C_t=\mathrm{rope}\big(\mathrm{rmsnorm}(C^{\text{raw}}_t)+b_C\big)$$

with `b_B, b_C ∈ ℝ^{H×M×N}` learnable, initialised to **ones**. Norm first, bias
second, rotation third; §4.7–4.8 show the third boundary is load-bearing and §9.1
records that the crate matches it.

### 2.2 What the source measured

All architecture ablations are 440M parameters at Chinchilla-optimal tokens,
FineWeb-Edu test perplexity. They are separate runs, so the two "no bias" rows
below disagree slightly (16.49 vs 16.52); read the ordering, not the digits.

| variant | ppl ↓ | | bias init | ppl ↓ | | `B` bias | `C` bias | ppl ↓ |
|---|---|---|---|---|---|---|---|---|
| − bias − trapezoid | 16.68 | | `1.0`, trained | **15.72** | | ✗ | ✗ | 16.52 |
| − bias | 16.49 | | `1.0`, frozen | 15.80 | | ✓ | ✗ | 16.68 |
| **Mamba-3** | **15.72** | | `U(0,1)` | 15.76 | | ✗ | ✓ | 15.98 |
| **+ short conv** | 15.85 | | `U(−1,1)` | 16.07 | | ✓ | ✓ | **15.69** |
| | | | `0.0` | 16.57 | | | | |

Three readings the tables support directly. Re-adding the short convolution to the
finished model *hurts* (15.85 vs 15.72) — it is redundant, not merely optional. A
frozen ones-bias recovers most of the gain (15.80), so the bias is doing more work
as an initialisation than as a learned parameter. And the biases are synergistic:
`B` alone is worse than neither.

Elsewhere in the source, and not re-derived here: data-dependent RoPE takes parity
from 0.90 → 100.0 and bracketed modular arithmetic from 0.88 → 87.75 where standard
RoPE and no RoPE both fail; MIMO at `R = 4` adds 1.2 points of downstream average
over SISO at matched parameters; Mamba-3 at half the state size matches Mamba-2's
perplexity; and Mamba-3 (SISO) decodes slightly *faster* than Mamba-2 despite the
extra terms (the source attributes this to its kernels, and does not decompose it
per component).

---

## 3. BCNorm pins the score's scale

RMSNorm over the `N` axis with unit gain sends any `B ≠ 0` to a vector of RMS 1,
hence `‖B‖ = √N` (§3.1), and is invariant to any positive rescaling of the
projection that produced it (§3.2). The score `C^⊤B` therefore stops being a
product of two learned scales — under a `10×` reparameterisation of both
projections the unnormalised score moves by `100×` and the normalised one does not
(§3.3) — and becomes a quantity whose magnitude is fixed by the shape: mean `0`,
standard deviation `√N` over random data (§3.4).

That is the whole mechanism behind the source's stability claim ("BCNorm is also
able to stabilize large-scale runs, resulting in the removal of the post-gate
RMSNorm"). Mamba-2 controlled the block's output scale at the *end*, after the SSD;
Mamba-3 controls the two factors that set it at the *start*. The state is a sum of
outer products of `B` with values, and the read pairs it against `C`, so pinning
both operands pins the whole path — with the exception of the values, which is what
the optional output norm (§7) is for when the exception matters.

It is also what makes §4 possible: a bias of fixed size is only meaningful against
an operand of fixed size. Bias and norm are one change, not two.

---

## 4. The biases: an affine score, and a convolution at init

With `B̃ = B + b_B` and `C̃ = C + b_C` the score expands into four terms (§4.1):

$$\tilde C_t^\top\tilde B_s=\underbrace{C_t^\top B_s}_{\text{data-data}}
+\underbrace{C_t^\top b_B}_{\text{query-side}}
+\underbrace{b_C^\top B_s}_{\text{key-side}}
+\underbrace{b_C^\top b_B}_{\text{constant}}$$

Three of them are `O(√N)`; the fourth, at the ones init, is exactly `N` (§4.2).
The asymmetry is the point. At `N = 128` the score sits within ~30% of the constant
`N` (§4.3), that deviation shrinks as `1/√N` (§4.4), and every (query, key) pair
scores **positive** where the unbiased score would be positive half the time
(§4.5). Propagated through the decay mask, the layer's read at initialisation is
the bias-only read to within `O(1/√N)` (§4.10):

$$y_t\;\approx\;N\sum_{s\le t}\Big(\textstyle\prod_{i=s+1}^{t}\alpha_i\Big)\gamma_s\,v_s$$

which is a causal, data-independent, exponentially-weighted moving average — a
convolution whose kernel is the decay profile. The layer *starts* as a smoother and
learns its content-dependence on top of that, rather than starting from a
zero-mean score and having to discover locality.

**The ablation table falls out of the constant term.** `⟨b_C, b_B⟩` is `N` for the
ones init, `N/4` for `U(0,1)`, `0` for the zero init and `0` in expectation for
`U(−1,1)` (§4.6) — matching the measured order `15.72 < 15.76 < 16.07 < 16.57` and
the source's own summary that the model "is not very sensitive to the
initialization of the biases as long as they are positive". Positivity is not a
convention: it is what makes the four-term expansion have a floor at all. A
symmetric init has the same *variance* as the ones init and none of its effect.

**The rotation localises the floor.** Because the bias is added before the rotation,
both bias vectors are rotated by their own cumulative angles, and the constant term
becomes (§4.7)

$$b_C^\top R_t^\top R_s b_B=(N-\texttt{rope\_dim})+2\sum_{j}\cos\big(\theta_{t,j}-\theta_{s,j}\big)$$

— maximal at zero rotation-lag, decaying to the unrotated remainder
`N − rope_dim` on average as the lag grows (§4.9). Adding the bias after the
rotation would have given the flat constant `N` regardless of lag (§4.8). Two
consequences. The convolutional floor is a *locality* prior in accumulated-rotation
time rather than a uniform one; and `rope_fraction` sets how much of it is
localised — at `rope_fraction = 1` no flat component survives at all, at the
reference's `0.5` half of it does. That is a real coupling between the bias and the
rotation, and it exists only because of the ordering.

**Why `C` matters more than `B`.** The presence table shows `C`-only (15.98) beating
`B`-only (16.68), which is *worse* than neither. The expansion at least shows the
two are not interchangeable — `b_B` enters the state, written once per token and
then decayed with everything else, while `b_C` enters the read, applied fresh at
every step to the whole accumulated state, so `b_C` alone gives every read a
data-independent probe of the running sum while `b_B` alone gives every write a
data-independent direction with no matching probe. Offered as a reading of the
asymmetry, not as a derivation of it: nothing here predicts the *sign* of the
`B`-only row, and the checks in §4 do not cover it.

---

## 5. The short convolution, and its two replacements

Mamba-2 (and GDN, and most linear models) applies a depthwise causal convolution of
width 4, plus `SiLU`, to `x‖B‖C` before the recurrence. Mamba-3 has neither, and
re-adding the convolution makes the model worse (15.85 vs 15.72). The source
attributes this to the trapezoid plus the biases; §4 priced the biases, and the
trapezoid's half is a restatement of its own definition:

$$h_t=\alpha_th_{t-1}+\beta_tv_{t-1}+\gamma_tv_t
=\alpha_th_{t-1}+\tilde v_t,\qquad \tilde v_t=\gamma_tv_t+\beta_tv_{t-1}$$

so the three-term recurrence is a two-term one run on a width-2 filtered
state-input (§5.1). The filter differs from the deleted convolution on three axes
at once: it is **data-dependent** (`β`, `γ` are per-token functions of `Δ`, `A`,
`λ`) where the conv is fixed; it acts on `B_tx_t` **inside** the recurrence where
the conv acts on `x_t` outside it; and it is one scalar pair per (head, step) where
the conv is a separate 4-tap kernel per channel. The derivation of `β` and `γ` is
[`trapezoid-as-integration.md`](trapezoid-as-integration.md) §§4–5 — cite it, this
note only needs their sign.

**Both replacements are strictly positive.** `β = (1−λ)Δα` and `γ = λΔ` with
`λ ∈ (0,1)`, `Δ > 0`, `α > 0`: the taps are positive for every input (§5.2), and
their ratio is unbounded, so the pair sweeps the whole positive quadrant and
nothing else (§5.3). The bias floor is positive by §4. So the two mechanisms that
replaced the convolution are both **low-pass**: they can average adjacent
state-inputs in any proportion, and they cannot subtract one from the other. A
learned depthwise kernel can. This is the one capability the deletion actually
costs, and it is worth naming precisely, because "the convolution is redundant" is
a claim about a trained 440M language model and not about the function classes.

---

## 6. Data-dependent `A`

Mamba-2's per-head `A_h` is learned and input-independent, so the pair that the
recurrence actually consumes is

$$(\alpha_t,\gamma_t)=\big(e^{\Delta_tA_h},\ \Delta_t\big)$$

a **one-dimensional** locus per head (§6.1): the Jacobian in the single free
variable `Δ_t` has rank 1. Its own preliminaries describe the consequence — "a
larger `Δ_t` forgets faster and up-weights the current token more strongly" — as a
feature, and it is one, but it is also a constraint: a token cannot ask to be
written strongly *and* remembered long.

Mamba-3 projects `A_t` per token, `A_t = −softplus(\cdot)` clamped to `≤ −a_floor`.
The locus becomes two-dimensional (§6.2) — exactly one extra degree of freedom per
(token, head) — subject to one inequality the floor imposes (§6.3):

$$\alpha_t\le e^{-\texttt{a\_floor}\cdot\gamma_t}$$

and at fixed `γ` the reachable `α` is the half-open interval `(0, e^{−a_floor·γ}]`,
swept by the `A` projection alone (§6.4). Measured on a token stream, tokens
sharing a write weight to within 1% have an `α` spread of `~1.0` under Mamba-3 and
`~0.007` under Mamba-2 (§6.5).

The source's own remark is that data-dependent `A` performs *similarly* to
data-independent `A` and was chosen "for consistency so that all SSM parameters are
data-dependent". That is the honest framing to carry: it is a parameterisation
choice with a clean interpretation, not one of the three measured advances, and a
Mamba-2 → Mamba-3 comparison should not attribute anything to it.

---

## 7. What each output-norm placement erases

Mamba-2 ends its block with a gated RMSNorm over `d_inner`, applied *after* the
gate. Mamba-3 removes it and ends with the bare `y ⊙ SiLU(z)`; the source restores
it — as a **pre-gate, per-head-grouped** norm — for hybrid models, reporting that
it buys length-generalised retrieval at a small cost in in-context retrieval, and
declining to call any row of its table best. The three placements are not variants
of one operation; each erases something different:

| placement | rescaling of `y` | rescaling of the gate `SiLU(z)` | check |
|---|---|---|---|
| post-gate (Mamba-2) | erased | **erased** | §7.1, §7.3 |
| pre-gate (Mamba-3 optional) | erased | kept | §7.2, §7.3 |
| none (Mamba-3 default) | kept | kept | §7.4 |

The middle row is the interesting one: normalising before the gate keeps the gate's
*magnitude* as a live signal, while normalising after it keeps only its direction.
A gate that can attenuate its own output is exactly what a retrieval head needs to
suppress a position, and it is exactly what a norm placed after it takes away —
which is a mechanism for the source's observation, not a prediction of its table.

Grouping is the second axis and is independent: a per-head norm equalises the
heads' scales, while one norm over `d_inner` preserves their spread (§7.5). A
model whose heads have deliberately different magnitudes loses that under grouping;
one where a single head can blow up is protected by it.

The default `has_outproj_norm = false` is the pure-Mamba-3 setting and is right for
this crate's default, with BCNorm (§3) doing the stabilising upstream.

---

## 8. Chunk length under MIMO and `micro_steps`

SSD's chunked algorithm costs `intra + inter` FLOPs per chunk; at `C = N = P` and
rank 1 the total is `≈ 8TN²` (§8.1). Under MIMO the intra-chunk term scales with
`(CR)²` — exactly `R²` at fixed `C` (§8.3) — while the inter-chunk term scales with
`R`. The source's remedy is to halve the chunk as the rank grows,
`C_MIMO = C_SISO / R`, which returns the total to `≈ 8TRN²`, i.e. `R×` SISO and not
`R²×` (§8.2). Its reference implementation carries the same advice as a comment on
the `chunk_size` argument ("64 for SISO, 64/mimo_rank for MIMO"). At `N = P = 64`,
`R = 4` the two schedules differ by `2.5×` (§8.4).

This crate fuses the rank onto the chunk axis (`[L·M, L·M]` intra-chunk matrices in
`single_ssd/ssd/serial.rs`), so the same arithmetic applies verbatim, and
`Mamba3SsdPath::optimal_chunk_len` implements it: `√(N·P)` divided by `mimo_rank`,
then rounded onto the 32 grid and clamped to `32..=512`. At `N = 128`, `P = 64` that
is `96 / 64 / 32` for a rank of `1 / 2 / ≥4` (§8.5), so the fused axis `C·M` lands at
128 for `M = 4` where the unscaled chunk would put it at 384 (§8.6). A SISO block is
unaffected: the divisor is 1 and the value is what it always was.

**`micro_steps` is the other widening, and it is not the same one.** Micro-steps fold
into the sequence axis, but they fold only the *writes*: each of a token's `u`
micro-steps writes to the state, while the token is **read** once, at its last one.
A chunk therefore has two axes — `C` writes by `C/u` reads — and its score is
`[batch, nchunks, heads, (C/u)·M, C·M]`, i.e. `batch · sequence · heads · C · M²`
elements with no `u` in it at all (§8.7). The rank appears on *both* axes and so
divides the chunk; `u` appears on one and so only subdivides it. Hence the second
half of the schedule: round the width up to a multiple of `u`, which makes a chunk a
whole number of tokens — and makes its read rows a contiguous run of them, a reshape
rather than a gather.

The alternative is to divide by `u` as well, on the reading that a chunk of `C`
positions holds `C/u` tokens. It is worse on both counts. FLOPs are indifferent to
it either way (`u` multiplies the intra- and inter-chunk terms alike and cancels out
of the balance), but a chunk shrinking as `1/u` over a sequence growing as `u` puts
`nchunks` at `u²` — twice the serial scan the current schedule runs, and both
pathways' backwards walk it chunk by chunk. And it does not even buy the memory it
is spent on: the 32 grid floors the division, so past a fold of 3 the chunk stops
shrinking and the score grows with `u` regardless — `2.67×` at `u = 8` where holding
the width fixed is exactly `1×` (§8.8).

What remains linear in `u` is the write side, and irreducibly so: the state
recurrence really does take `u` steps per token. The read side — the score, the
state-to-output product, and `C`'s own QK-norm and rotation — is `u`-invariant in
both FLOPs and memory, because it is only ever evaluated where the readout happens
(`helpers::read_rows`; the kernels take the stride as `Mamba3*SsdInput::read_stride`).
A model that wants a different trade can pass an explicit chunk length
(`Mamba3SsdPath::SerialRecalculated(Some(n))`), which wins over the schedule and is
rounded up to a whole number of tokens.

Two caveats on the whole exercise. `optimal_chunk_len` is a rule of thumb about
matmul shapes on real backends, not a FLOP-minimiser — which is why the 32 grid and
its floor are kept rather than following the division down to 24. And the FLOP model
above is the source's, measured against fused GPU kernels; this crate's cost is
dominated by kernel-launch counts on some backends (`kernels.sh`), where fewer,
larger chunks can win despite the FLOPs and the memory. The schedule is the
defensible default, not a measured optimum: `bench.sh` is what would settle it.

---

## 9. Consequences for this crate

### 9.1 The `B`/`C` path's order is load-bearing

`RMSNorm → GQA-expand → bias → rotation`, which is what
`helpers::qk_norm_expand_bias` followed by `rotate_bc_forward` does in both
pathways, and what `mamba3.py` + `mamba3_siso_fwd.py` do (norm in the module, bias
then rotary inside the kernel). The bias–rotation boundary is the one that changes
behaviour (§4.7 vs §4.8): moving the bias after the rotation would replace a
locality prior with a flat one. The norm–bias boundary matters for the same reason
BCNorm and the biases are one change (§3): a bias of size 1 is calibrated against
an operand of RMS 1.

### 9.2 The ones-init is a mechanism, not a placeholder

`Initializer::Ones` on `b_bias_hmr`/`c_bias_hmr` is what makes the block start as a
convolution (§4.10) and is measurably better than zeros (15.72 vs 16.57). Do not
"fix" it toward a zero-mean initialiser. Both biases are wanted, not just `B`'s
(§4, last paragraph). They are 3-D per-head parameters and stay on AdamW:
`muon_projections()` names only the two projection matrices, as it should.

### 9.3 The optional output norm is one row of the source's table

`has_outproj_norm = true` builds `RmsNormGatedConfig::new(per_head_dim)` with
`norm_before_gate(true)`: pre-gate and per-head-grouped, i.e. exactly the source's
"Pre-Gate Grouped RMS" row, the one it recommends for hybrids. The other three rows
are not reachable from the config — the placement and the grouping are both fixed
in `Mamba3Config::init`. Mamba-2's own norm (`is_norm_before_gate = false`, over
`d_inner`) is the "Post-Gate Default RMS" row, so the crate spans the two ends of
that table across the two families. Whether Mamba-3 should expose the other two is
an open question the source explicitly leaves open; nothing here argues for it.

### 9.4 The chunk schedule divides by `mimo_rank`, and only subdivides by `micro_steps`

§8. `Mamba3SsdPath::optimal_chunk_len` divides the square-root rule by the rank,
rounds onto the 32 grid, and then rounds *up* to a whole number of tokens. The rank
half is the source's own advice (its kernels carry it as a comment on `chunk_size`);
treating `micro_steps` as a different widening is this crate's, and follows from the
chunk having a read axis the rank widens and `u` does not (§8). The default path for
a SISO, `u = 1` block is byte for byte what it was.

This is the one place where the note changed behaviour rather than describing it,
and it is a heuristic changing a heuristic: exact values, unchanged; cost, moved.
`bench.sh` is what would confirm the direction on a given backend.

### 9.5 What belongs to the caller, not to this crate

The source's macro-architecture is Llama-style: Mamba-3 blocks alternating with
SwiGLU MLPs under pre-norm, and for hybrids a 5:1 interleave with NoPE
self-attention; MIMO models are parameter-matched by shrinking the MLP width
(4096 → 3824 at 1.5B). None of that is this crate's business — `burn-stack` owns
layer composition, and it is block-agnostic by construction. Two training-side
details are worth recording anyway because they are easy to lose: the reference
marks `dt_bias` and `D` as no-weight-decay and, notably, does **not** mark the
`B`/`C` biases, so under the usual convention they are decayed toward the zero init
that ablates worst — the source does not comment on this; and the
kernels the latency tables measure are fused Triton/TileLang/CuTe, which is the
axis this crate deliberately gives up (`CLAUDE.md` → *No optimized kernels*).

### 9.6 What does not follow

- Not that the convolution is useless in general (§5): it is redundant *given* the
  trapezoid and the biases, in a trained 440M LM. The function classes differ, and
  the difference has a sign.
- Not that the biases are an attention-sink mechanism. The floor is uniform over
  positions before the rotation modulates it (§4.7); a sink is position-selective.
  The resemblance is worth noting and nothing was verified about it here.
- Not that data-dependent `A` contributes to the reported gains (§6): the source
  says it does not.
- Not that removing the post-gate norm is safe at every scale (§7): the source
  restores it for hybrids and reports competing trade-offs among the placements.

---

## 10. Reproduction

```
python3 scripts/architecture_deltas.py
```

Pure `numpy`, float64, 35 checks, section numbers matching this document, non-zero
exit on failure. The script defines the block's `B`/`C` path and the trapezoid's
coefficients from scratch; it does not import or call the crate, so a failure means
one of the statements above is wrong, not that the implementation drifted.

---

## References

- Mamba-3 (`../papers/mamba/mamba-3/`): §*Mamba-3 Architecture* (BCNorm, the
  biases, the removed convolution and post-gate norm), the preliminaries' remark on
  data-dependent `A`, §*Multi-Input, Multi-Output* (the chunked FLOP count and the
  `C/R` schedule), and the appendices' architecture ablations and hybrid norm table.
- Reference implementation: `../py/state-spaces/mamba/mamba_ssm/modules/mamba3.py`
  (module-level norm, bias init, `A_floor`, `is_outproj_norm`, the `chunk_size`
  advice) and `ops/triton/mamba3/mamba3_siso_fwd.py` (bias then rotary, inside the
  kernel).
- Companion notes: [`trapezoid-as-integration.md`](trapezoid-as-integration.md)
  (`β`, `γ` and the two-installment collapse),
  [`rotation-as-optimization.md`](rotation-as-optimization.md) (the rotation and
  its data-dependence), [`mimo-as-batch.md`](mimo-as-batch.md) (the rank, the value
  tying, and why the rotation is shared across ranks).
