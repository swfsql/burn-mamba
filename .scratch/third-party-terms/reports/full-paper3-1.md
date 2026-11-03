Batch `paper3-1` is complete. It has 66 citations: 61 confirmed, 4 wrong, 1 not found. Rows with a correction: L45, L46, L47, L48, L54, L77, L81. All rows are finished.

No report file exists, as you instructed. `coverage.py` reads `report-paper3-1.md`, so it reports this batch as absent until you save the lines below to that path. The verdict lines below are in the exact form. I checked that each `Old` text of a correction occurs in `docs/source-terms.md`.

L39 | `M3/structure/preliminaries.tex:18` | confirmed | 18: "is the hidden state".
L39 | `M3/structure/related.tex:16` | confirmed | 16: "the recurrent state represents a compressed summary". See Extra findings.
L42 | `M3/structure/preliminaries.tex:28` | confirmed | 28: "The core of the Mamba-2 layer".
L42 | `M3/structure/intro.tex:31` | confirmed | 31: "the core of our \textbf{Mamba-3} layer".
L42 | `M3/structure/experiments.tex:39` | confirmed | 39: "to the Mamba-3 layer".
L42 | `M3/structure/methods_mimo.tex:71` | confirmed | 71: "for the entire layer".
L42 | `M3/appendix/mimo.tex:23` | confirmed | 23: "the layer output".
L42 | `M3/structure/methods_arch.tex:14` | confirmed | 14: "The SSD layer is replaced".
L43 | `M3/structure/methods_arch.tex:10` | confirmed | 10: "alternating Mamba-3 and SwiGLU blocks with pre-norm".
L45 | `M3/structure/preliminaries.tex:39` | confirmed | 39: "each SSM within the layer".
L45 | `M3/structure/methods_mimo.tex:21` | confirmed | 21: "stacking the SISO recurrence ... with $P$ copies". This is one recurrence for each channel. See the correction.
L45 | `M3/structure/methods_mimo.tex:66` | confirmed | 66: "number of SSMs of its SISO counterpart".
L45 | `M3/appendix/mimo.tex:29` | confirmed | 29: "the set of independent SISO SSMs within each head". See the correction.
L46 | `M3/structure/methods_complex.tex:15` | confirmed | 15: "data-dependent rotary embeddings".
L46 | `M3/structure/methods_complex.tex:16` | confirmed | 16: "data-dependent RoPE embeddings".
L46 | `M3/structure/methods_complex.tex:62` | confirmed | 62: "a data-dependent rotary embedding applied on the $\bB,\bC$ components".
L46 | `M3/structure/methods_complex.tex:73` | confirmed | 73: "vanilla RoPE ... a fixed frequency schedule".
L46 | `M3/structure/methods_complex.tex:80` | confirmed | 80: "Rotary Embedding Equivalence". It names our rotation. See the correction.
L46 | `M3/structure/methods_complex.tex:96` | confirmed | 96: "data-dependent RoPE ... standard RoPE".
L46 | `M3/structure/methods_arch.tex:6` | confirmed | 6: "data-dependent RoPE embeddings".
L46 | `M3/structure/experiments.tex:54` | confirmed | 54: "Mamba-3 with standard RoPE".
L46 | `M3/structure/intro.tex:40` | confirmed | 40: "the efficient RoPE-like calculation".
L46 | `M3/tables/ablations.tex:38` | confirmed | 38: "Mamba-3 (w/ Std. RoPE)".
L46 | `M3/tables/ablations.tex:39` | confirmed | 39: "Mamba-3 (w/o RoPE)". "RoPE" alone names the rotation.
L46 | `M3/tables/retrieval.tex:6` | confirmed | 6: "The Transformer baseline uses RoPE". "RoPE" alone names the fixed-frequency encoding.
L47 | `M3/structure/methods_complex.tex:13` | confirmed | 13: "we begin with complex SSMs".
L47 | `M3/structure/methods_complex.tex:30` | wrong | 30: "equivalent to a real-valued SSM". Its equation at M3/structure/methods_complex.tex:33 holds the rotation `\bR_t`. The proof of the sense with no rotation is at M3/structure/methods_complex.tex:96: "real-valued SSMs with and without standard RoPE cannot".
L47 | `M3/structure/methods_complex.tex:58` | confirmed | 58: "the discretized complex SSM". The line also gives "an equivalent real SSM" with rotation matrices. See the correction.
L47 | `M3/structure/methods_complex.tex:96` | confirmed | 96: "our complex SSM ... real-valued SSMs".
L47 | `M3/structure/experiments.tex:8` | confirmed | 8: "complex transitions".
L47 | `M3/structure/experiments.tex:54` | confirmed | 54: "Mamba-3 without RoPE".
L48 | `M3/structure/methods_trap.tex:42` | wrong | 42: the word "Trapezoidal" is present, but the table row has no $\lambda$. Its $\alpha$ at line 43 is not $e^{\Delta A}$, and it has no $\beta$ (line 44). The proof for `λ = 1/2` is at M3/structure/methods_trap.tex:142, for "classical trapezoid rule" only.
L48 | `M3/structure/methods_trap.tex:121` | confirmed | 121: "classical trapezoidal rule ... simply averages the interval endpoints". The exact phrase "classical trapezoid rule" is at lines 142 and 168.
L48 | `M3/structure/methods_trap.tex:142` | confirmed | 142: "the classical trapezoid rule, which is recovered when $\lambda_t = \tfrac{1}{2}$".
L49 | `M3/structure/methods_trap.tex:138` | confirmed | 138: "$\beta_t \coloneqq (1-\lambda_t)\Delta_t e^{\Delta_t A_t}$, $\gamma_t \coloneqq \lambda_t \Delta_t$".
L49 | `M3/structure/methods_trap.tex:200` | confirmed | 200: "convolution with weights $\beta, \gamma$".
L51 | `M3/appendix/discretization.tex:153` | confirmed | 153: "a data-dependent gate". Line 154 gives "$\lambda_t = \sigma(u_t)$".
L53 | `M3/structure/preliminaries.tex:6` | confirmed | 6: "the input sequence length as $T$".
L54 | `M3/structure/methods_arch.tex:15` | confirmed | 15: "employs the SISO SSM by default".
L54 | `M3/structure/methods_mimo.tex:21` | wrong | 21: "the SISO recurrence", with no "SISO SSM", no "SISO variant" and no default form. The proof is at M3/structure/methods_mimo.tex:67: "conversion of a SISO SSM to a rank $R$ MIMO SSM".
L54 | `M3/appendix/mimo.tex:32` | confirmed | 32: "parameter match the SISO variants".
L56 | `M3/appendix/results.tex:183` | confirmed | 183: "in two halves ... between segments".
L58 | `M3/appendix/results.tex:110` | confirmed | 110: "a comparable fusion structure ... a Triton kernel fused with".
L58 | `M3/appendix/results.tex:119` | confirmed | 119: "Kernel DSL and fusion structure".
L61 | `M3/structure/methods_mimo.tex:53` | confirmed | 53: "a \emph{chunked} algorithm".
L62 | `M3/appendix/results.tex:46` | confirmed | 46: "state size usage of Mamba variants".
L62 | `M3/appendix/results.tex:175` | confirmed | 175: "For Mamba variants". Line 174 lists "Mamba-3 1.5B (SISO), Mamba-2 1.5B".
L62 | `M3/structure/intro.tex:45` | confirmed | 45: "Both of our Mamba-3 variants".
L62 | `M3/tables/decode_cudagraph_timing.tex:7` | confirmed | 7: "across different Mamba variants".
L72 | `M3/structure/methods_trap.tex:138` | confirmed | 138: "$\lambda_t \in [0,1]$ is a data-dependent scalar".
L72 | `M3/structure/methods_trap.tex:146` | confirmed | 146: "the trapezoidal parameter satisfies".
L73 | `M3/structure/methods_complex.tex:23` | confirmed | 23: "\boldsymbol{\theta}(t)". The discrete form is at line 44.
L76 | `M3/structure/methods_arch.tex:19` | confirmed | 19: "following the $\bB,\bC$ projection, mirroring the QKNorm".
L76 | `M3/structure/methods_arch.tex:20` | confirmed | 20: "BC normalization (BCNorm) or QK normalization (QKNorm)".
L77 | `M3/structure/methods_arch.tex:24` | not found | 24: "$\bB, \bC$ Biases" has no "BC bias". The exact phrase is at M3/structure/experiments.tex:49 and M3/tables/ablations.tex:23.
L77 | `M3/structure/methods_arch.tex:25` | confirmed | 25: "learnable, head-specific, channel-wise biases".
L79 | `M3/appendix/mimo.tex:11` | confirmed | 11: "(\bW_X,\bX'_t)".
L79 | `M3/appendix/mimo.tex:13` | confirmed | 13: "weights $\bW_{Z'}$ and $\bW_Z$".
L79 | `M3/appendix/mimo.tex:25` | confirmed | 25: "(\bW_{O'},\bY'_t)".
L81 | `M3/structure/methods_trap.tex:64` | wrong | 64: "Exponential-Trapezoidal", the row of Mamba-3. The proof is at M3/structure/methods_trap.tex:59: "Exponential-Euler", with "Mamba-1, -2" at line 63.
L81 | `M3/structure/methods_trap.tex:100` | confirmed | 100: "Exponential-Euler (Mamba-1/-2)". The line does not support "ZOH". See the correction.
L82 | `M3/structure/preliminaries.tex:62` | confirmed | 62: "is a structured mask".
L82 | `M3/structure/methods_trap.tex:171` | confirmed | 171: "composed with a 2-band matrix".
L83 | `M3/structure/methods_mimo.tex:56` | confirmed | 56: "intra-chunk computation ... inter-chunk computation".
L84 | `M3/appendix/results.tex:183` | confirmed | 183: "a state passing approach".
L88 | `M3/appendix/results.tex:15` | confirmed | 15: "(pre- vs post-gate) ... pre-gate, grouped RMS".

## Row corrections

### L45

Cell: Its meaning in the source
Old: Papers: the source counts several SSMs in one layer, one for each head or channel.
New: Papers: the source counts several SSMs in one layer, one for each head or channel. The Mamba-3 paper also uses "SSM" for the whole plant, as in "a single head of a typical SSM".
Proof: `M3/structure/methods_mimo.tex:20`: "Consider a single head of a typical SSM".

Cell: Our term
Old: **Head** for one of them.
New: **Head** for one of them when the source counts one SSM for each head. The Mamba-3 paper also counts one SSM for each channel of a head. That SSM has no term of ours.
Proof: `M3/appendix/mimo.tex:29`: "SISO SSMs within each head". `M3/structure/methods_mimo.tex:21`: "$P$ copies sharing the same $\alpha_t, \Delta_t$". Our **Head** has its own step size and decay (`GLOSSARY.md`), so one copy is not a head.

Cell: Citation
Old: `M3/structure/preliminaries.tex:39`, `M3/structure/methods_mimo.tex:21`, `M3/structure/methods_mimo.tex:66`, `M3/appendix/mimo.tex:29`
New: `M3/structure/preliminaries.tex:31`, `M3/structure/preliminaries.tex:39`, `M3/structure/methods_mimo.tex:20`, `M3/structure/methods_mimo.tex:21`, `M3/structure/methods_mimo.tex:66`, `M3/appendix/mimo.tex:29`, `M3/structure/methods_trap.tex:160`
Proof: `M3/structure/preliminaries.tex:31`: "the SSM recurrence". `M3/structure/methods_trap.tex:160`: "the core recurrence". The row shows no Mamba-3 citation for these two words.

### L46

Cell: Its meaning in the source
Old: Mamba-3 paper: "data-dependent rotary embeddings" and "data-dependent RoPE" name our rotation.
New: Mamba-3 paper: "data-dependent rotary embeddings", "data-dependent RoPE", "Rotary Embedding" and "rotary position embeddings" name our rotation.
Proof: `M3/appendix/results.tex:110`: "For Mamba-3 ... fused with rotary position embeddings". `M3/structure/methods_complex.tex:80`: "Rotary Embedding Equivalence". The row gives these words only to the official code, and our **Rotary position embedding** is the fixed-frequency encoding.

Cell: Citation
Old: `M3/tables/retrieval.tex:6`
New: `M3/tables/retrieval.tex:6`, `M3/appendix/results.tex:110`
Proof: `M3/appendix/results.tex:110`: "rotary position embeddings".

### L47

Cell: Its meaning in the source
Old: Mamba-3 paper: two forms of the model. "Complex SSM" has the rotation. "Real-valued SSM" and "without RoPE" have no rotation.
New: Mamba-3 paper: "complex SSM" has the rotation. "Without RoPE" has no rotation. "Real-valued SSM" has two meanings. In a comparison, it has no rotation. In the equivalence proposition, it is the real form of the complex SSM, and its transition holds the rotation matrix `R_t`.
Proof: `M3/structure/methods_complex.tex:30`: "equivalent to a real-valued SSM". `M3/structure/methods_complex.tex:33`: "\bR_t\,\vh_{t-1}". `M3/structure/methods_complex.tex:96`: "real-valued SSMs with and without standard RoPE". "Two forms" also fails: `M3/tables/ablations.tex:37` to `M3/tables/ablations.tex:39` list three forms of Mamba-3.

### L48

Cell: Its meaning in the source
Old: Mamba-3 paper: "Trapezoidal" and "classical trapezoid rule" name the classic rule with a fixed average, at `λ = 1/2`.
New: Mamba-3 paper: "classical trapezoid rule" names the classic rule with a fixed average of the endpoints. Our recurrence gives this average of the state input at `λ = 1/2`. "Trapezoidal", in the table of discretizations, is the form of S4. Its `α` is `(I−ΔA/2)⁻¹(I+ΔA/2)`, not `exp(ΔA)`. It has no `β`. So it is not our recurrence at `λ = 1/2`.
Proof: `M3/structure/methods_trap.tex:43`: "$(I-\frac{\Delta}{2}A)^{-1}\,(I+\frac{\Delta}{2}A)$". `M3/structure/methods_trap.tex:46`: "S4". `M3/structure/methods_trap.tex:138`: "$\alpha_t \coloneqq e^{\Delta_t A_t}$". `M3/structure/methods_trap.tex:142`: "recovered when $\lambda_t = \tfrac{1}{2}$".

### L54

Cell: Citation
Old: `M3/structure/methods_mimo.tex:21`
New: `M3/structure/methods_mimo.tex:67`
Proof: `M3/structure/methods_mimo.tex:67`: "a SISO SSM to a rank $R$ MIMO SSM". Change this citation in L54 only. The same citation is also in L45 and in a third row, outside this batch.

### L77

Cell: Citation
Old: `M3/structure/methods_arch.tex:24`
New: `M3/structure/experiments.tex:49`
Proof: `M3/structure/experiments.tex:49`: "the introduction of BC bias".

### L81

Cell: Meaning
Old: The discretizations of Mamba-1 and Mamba-2.
New: Mamba-3 paper: "exponential-Euler" is the discretization of the released Mamba-1 and Mamba-2. "ZOH" is the discretization that the Mamba-1 paper reports. The released code does not use it.
Proof: `M3/structure/methods_trap.tex:74`: "While the Mamba-1 paper reports ZOH discretization, the implementation follows". `M3/structure/methods_trap.tex:101`: "actually used an additional approximation in the released implementation". The write terms differ: `A_t⁻¹(exp(Δ_t A_t) − I)` at `M3/structure/methods_trap.tex:96` and `Δ_t` at `M3/structure/methods_trap.tex:107`.

Cell: Citation
Old: `M3/structure/methods_trap.tex:64`, `M3/structure/methods_trap.tex:100`
New: `M3/structure/methods_trap.tex:59`, `M3/structure/methods_trap.tex:74`, `M3/structure/methods_trap.tex:100`, `M3/structure/methods_trap.tex:101`
Proof: `M3/structure/methods_trap.tex:59`: "Exponential-Euler". `M3/structure/methods_trap.tex:64`: "Exponential-Trapezoidal".

## Extra findings

- L39: `M3/structure/related.tex:16` holds the word, but its sentence is on test-time training. A line on SSMs is `M3/structure/related.tex:45`: "Such SISO models have a significantly larger recurrent state".
- L42, negative claim "no word for a Pre-LN residual": the Mamba-3 paper passes. "pre-norm" occurs only at `M3/structure/methods_arch.tex:10`. "residual" occurs only for the gate, at `M3/appendix/mimo.tex:13` and `M3/appendix/mimo.tex:22`.
- L42, source word "layer": the row explains only the phrases that name our block. The paper also has "MLP layers" (`M3/appendix/mimo.tex:32`, `M3/structure/experiments.tex:22`), "RMSNorm layer" (`M3/structure/methods_arch.tex:21`) and "$24$ layers" (`M3/appendix/results.tex:175`). No sentence of the row is false.
- L46: no cited line shows "RoPE" alone for the RoPE trick. The closest is `M3/structure/intro.tex:40`: "RoPE-like calculation". The phrase is defined at `M3/structure/methods_complex.tex:68`: "as the ``RoPE trick.''".
- L61: a word search for "mode" and "modes" in `structure/`, `appendix/` and `tables/` of the Mamba-3 paper has no result. So `"Mode" also names forms with no chunks.` holds only for the Mamba-2 paper (`M2/efficient.tex:9`), although the sentence before it names both papers. Suggested text: `Mamba-2 paper: "mode" also names forms with no chunks.`
- L73: `θ_t` is at `M3/structure/methods_complex.tex:44`, which also shows the angle `Δ_t θ_t`, the proof of "rate". "Projected" is at `M3/structure/methods_arch.tex:16`: "produced by data-dependent projections".
- L79, source word "Phi": the Mamba-3 paper has only a lower-case `$\phi$`, at `M3/appendix/mimo.tex:22`. It is the gate function `$\bY_t \odot \text{SiLU}(\bZ_t)$`, not the down-projection. A reader of the paper can confuse the two.
- Words with no result in the Mamba-3 paper. Other batches must find them. L42: "selective SSM layer", "S6 layer", "core SSD layer". L49: "Trapezoidal coefficients", "left-endpoint coefficient", "right-endpoint coefficient". L72: "trapezoidal interpolation parameter", "Mixing factor", "convex-combination modulator". L79: "Psi", "Zeta". L83: "diagonal blocks", "low-rank blocks".