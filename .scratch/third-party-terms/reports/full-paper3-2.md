Batch `paper3-2`: 28 confirmed, 4 wrong, 0 not found, out of 32 citations. Rows with a correction: L97, L109, L110, L114, L171. All rows are finished.

**The report file does not exist.** The harness refused the Write call to `/tmp/source-terms-check/report-paper3-2.md`, with this message: "Subagents should return findings as text, not write report files." I did not write the file through Bash, because that would bypass the refusal. The full report is below instead. To use it, save the text between the `=====` lines to that path. Until then, `coverage.py` lists `paper3-2` under "no report". I did not run `coverage.py` on this text. I counted the verdicts by hand.

===== START OF REPORT =====
L97 | `M3/structure/methods_trap.tex:120` | wrong | 120: "convex combination of both interval endpoints". The line holds the word, but the word names two samples, not only the earlier sample. The right endpoint is the current sample. The proof is at M3/structure/methods_trap.tex:102: "holding the (right) endpoint constant".
L97 | `M3/structure/methods_trap.tex:156` | wrong | 156: "a width-2 convolution on $\vv_t$ (weighted by $\beta, \gamma$)". The quoted word is on the line before. The proof is at M3/structure/methods_trap.tex:155: "convolution of size two on the state-input".
L107 | `M3/appendix/mimo.tex:13` | confirmed | 13: "we obtain the residual gate term $\bZ_t$".
L107 | `M3/appendix/mimo.tex:22` | confirmed | 22: "the residual function $\phi$ ... $\phi(\bY_t, \bZ_t):=\bY_t \odot \text{SiLU}(\bZ_t)$". This is a gate, not a skip.
L109 | `M3/structure/preliminaries.tex:19` | confirmed | 19: "$\bB(t)x(t)$ as the \emph{state-input}".
L109 | `M3/structure/preliminaries.tex:85` | confirmed | 85: "$\bB_t$ represents the continuous parameter, whereas in Mamba-2, $\bB_t$ represents the discretized parameter which is equivalent to $\gamma_t \bB_t$".
L109 | `M3/structure/methods_trap.tex:156` | confirmed | 156: "the state-input $\vv_t = \bB_tx_t$".
L109 | `M3/structure/methods_mimo.tex:59` | wrong | 59: "state projection $\bB \vx^\top$". The line holds the word, but the word names the product, not `B`. The row says that "state projection" names `B`.
L110 | `M3/structure/methods_mimo.tex:31` | confirmed | 31: "modifying the output projection as $\bC_t \in \mathbb{R}^{N} \to \bC_t \in \mathbb{R}^{N \times R}$".
L110 | `M3/structure/methods_mimo.tex:59` | wrong | 59: "state emission $\bC^\top \vh$". The line holds the word, but the word names the product, not `C`. The row says that all its words are `C`.
L111 | `M3/structure/methods_trap.tex:1` | confirmed | 1: "\subsection{Exponential-Trapezoidal Discretization}".
L111 | `M3/structure/methods_trap.tex:119` | confirmed | 119: "we introduce a \emph{generalized trapezoidal rule}".
L112 | `M3/structure/methods_trap.tex:5` | confirmed | 5: "the data is only observed at discrete time steps".
L112 | `M3/structure/preliminaries.tex:49` | confirmed | 49: "vectorizes the time-step recurrence".
L112 | `M3/structure/related.tex:16` | confirmed | 16: "recurrent steps update the state". The sentence is about test-time training models, not about a Mamba model.
L113 | `M3/structure/preliminaries.tex:20` | confirmed | 20: "For discrete sequences with step size $\Delta_t$". The data dependence is at line 29.
L114 | `M3/structure/preliminaries.tex:32` | confirmed | 32: "$\alpha_t \coloneqq e^{\Delta_t A_t}\in(0,1)$".
L115 | `M3/structure/methods_arch.tex:10` | confirmed | 10: "alternating Mamba-3 and SwiGLU blocks with pre-norm".
L159 | `M3/appendix/discretization.tex:154` | confirmed | 154: "$\lambda_t = \sigma(u_t)$, where $u_t$ is a learned projection of the current input token".
L161 | `M3/structure/preliminaries.tex:18` | confirmed | 18: "$\vh(t)\!\in\!\mathbb{R}^N$ is the hidden state".
L161 | `M3/appendix/mimo.tex:18` | confirmed | 18: "$\bH_t \;=\; a_t\,\bH_{t-1} \;+\; \bB_t \bX_t^\top \in \mathbb{R}^{N\times P}$". Line 16 calls this "The state update".
L167 | `M3/structure/methods_trap.tex:156` | confirmed | 156: "$\vv_t = \bB_tx_t$".
L170 | `M3/structure/methods_mimo.tex:31` | confirmed | 31: "the output projection as $\bC_t$".
L170 | `M3/structure/methods_mimo.tex:53` | confirmed | 53: "the sequence is divided into chunks of length $C$".
L171 | `M3/structure/preliminaries.tex:85` | confirmed | 85: "$\bB_t$ represents the continuous parameter". The line proves the claim for `B` only. See the correction for `A`.
L172 | `M3/structure/methods_trap.tex:138` | confirmed | 138: "$\beta_t \coloneqq (1-\lambda_t)\Delta_t e^{\Delta_t A_t}$".
L181 | `M3/structure/preliminaries.tex:6` | confirmed | 6: "We denote the input sequence length as $T$".
L181 | `M3/appendix/results.tex:179` | confirmed | 179: "Sequence lengths were swept over $L \in \{512, 1024, 2048, 4096, 16384\}$".
L186 | `M3/structure/methods_mimo.tex:25` | confirmed | 25: "$\bB_t \in \mathbb{R}^{N} \to \bB_t \in \mathbb{R}^{N \times R}$".
L186 | `M3/structure/methods_mimo.tex:31` | confirmed | 31: "acquires an extra rank $R$".
L186 | `M3/structure/methods_mimo.tex:67` | confirmed | 67: "a rank $R$ MIMO SSM".
L186 | `M3/appendix/mimo.tex:5` | confirmed | 5: "denote $P, R\in\mathbb{N}$ as the head dimension and MIMO rank".

## Row corrections

### L97

Cell: Meaning
Old: The earlier sample of the write.
New: The earlier sample of the write. In the Mamba-3 paper, an "interval endpoint" is either sample, at `t−1` or at `t`. "Convolution of size two" reads both samples, with the weights `β` and `γ`.
Proof: M3/structure/methods_trap.tex:102 "holding the (right) endpoint constant". M3/structure/methods_trap.tex:120 "both interval endpoints". M3/structure/methods_trap.tex:156 "width-2 convolution on $\vv_t$ (weighted by $\beta, \gamma$)".

Cell: Citation
Old: `M3/structure/methods_trap.tex:120`, `M3/structure/methods_trap.tex:156`
New: `M3/structure/methods_trap.tex:102`, `M3/structure/methods_trap.tex:120`, `M3/structure/methods_trap.tex:134`, `M3/structure/methods_trap.tex:155`, `M3/structure/methods_trap.tex:156`
Proof: M3/structure/methods_trap.tex:155 "convolution of size two on the state-input". M3/structure/methods_trap.tex:134 "$\beta_t \bB_{t-1} x_{t-1}$" is the proof of the tap at `t−1`.

### L109

Cell: Meaning
Old: "State projection" and "expansion" name `B`.
New: "State projection" names the product `B xᵀ`. "Expansion" names `B` in the Mamba-2 glossary. "State expansion" and "expansion factor" name `N`, which is our `r`.
Proof: M3/structure/methods_mimo.tex:59 "state projection $\bB \vx^\top$". M2/glossary.tex:24 "input projection (expansion) matrix". M3/tables/mimo_arith_intensity.tex:8 "$N$ is the state size (expansion factor)". M3/structure/related.tex:53 "its state expansion is kept at Mamba-1/-2 levels".

Cell: Our term
Old: **Write** for the other words.
New: **Write** for "state projection" and "expansion". `r` (`state_rank`) for "state expansion" and "expansion factor".
Proof: M3/tables/mimo_arith_intensity.tex:8 "$N$ is the state size (expansion factor)". M2/background.tex:131 also has "state expansion factor".

Cell: Citation
Old: `M3/structure/methods_mimo.tex:59`, `M2/glossary.tex:24`
New: `M3/structure/methods_mimo.tex:59`, `M3/structure/related.tex:53`, `M3/tables/mimo_arith_intensity.tex:8`, `M2/glossary.tex:24`
Proof: The two new lines are in the Meaning proof above.

### L110

Cell: Meaning
Old: `C` of the plant.
New: `C` of the plant. "State emission" names the product `Cᵀ h`. "Tensor contraction" and `contract` name einsum notation, not `C`. In the Mamba-3 paper, "output projection" also names `out_proj`, which runs after the gate.
Proof: M3/structure/methods_mimo.tex:59 "state emission $\bC^\top \vh$". M3/appendix/mimo.tex:5 "tensor contraction notation". M3/appendix/results.tex:114 "OP = output projection". M3/appendix/results.tex:127 "\torch{IP}, \triton{Conv}, \triton{SSM}, \triton{Gate}, \torch{OP}". M3/structure/experiments.tex:39 "pre-output projection RMSNorm". SSM/modules/mamba3.py:249 `out = self.out_proj(y.to(x.dtype))`.

Cell: Citation
Old: `M3/structure/methods_mimo.tex:59`, `M2/glossary.tex:25`
New: `M3/structure/methods_mimo.tex:59`, `M3/appendix/mimo.tex:5`, `M3/appendix/results.tex:114`, `M3/appendix/results.tex:127`, `M3/structure/experiments.tex:39`, `M2/glossary.tex:25`
Proof: The new lines are in the Meaning proof above.

### L114

Cell: Meaning
Old: `α_t` is `exp(Δ_t A_t)`, the scalar decay of the transition.
New: `α_t` is `exp(Δ_t A_t)`, the scalar decay of the transition. The Mamba-3 appendix also calls `A(t)` "a scalar decay".
Proof: M3/appendix/discretization.tex:8 "$A(t)\in\mathbb{R}$ is a scalar decay".

Cell: Citation
Old: `M3/structure/preliminaries.tex:32`
New: `M3/structure/preliminaries.tex:32`, `M3/appendix/discretization.tex:8`
Proof: The new line is in the Meaning proof above.

### L171

Cell: Citation
Old: `M3/structure/preliminaries.tex:85`
New: `M3/structure/preliminaries.tex:43`, `M3/structure/preliminaries.tex:85`
Proof: M3/structure/preliminaries.tex:43 "the overall discrete transition $\alpha_t \coloneqq e^{\Delta_t A_t}$". This line shows that `A_t` is continuous in the Mamba-3 paper. Line 85 names only `B`.

## Extra findings

- L97, source word "β term". The Mamba-3 paper does not have this phrase. `rg -i 'beta term'` found no line. The phrase is at MM/mamba3.py:95, which the row cites. This is not a fault.
- L97, claim "Each source has one tap only, at `t−1`". The claim holds for the Mamba-3 paper. M3/structure/methods_trap.tex:134: "$\beta_t \bB_{t-1} x_{t-1}$". A search for a multi-step rule (`Simpson`, `Adams`, `Runge`, `Heun`, `two-tap`) found no line.
- L97, source word "convolution of size two". The phrase is also at M3/appendix/discretization.tex:70.
- L107, source word `z`. The cited Mamba-3 lines show only `Z_t`. The lower-case word is at M3/structure/methods_mimo.tex:68: "the gate $\vz$". This is not a fault.
- L107, claim about "residual". The word is only at M3/appendix/mimo.tex:13 and M3/appendix/mimo.tex:22 in the paper. The paper has no other sense of the word, so the claim holds.
- L111, claim "with one tap at `t−1`". The cited lines do not show the lag. The proof is at M3/structure/methods_trap.tex:134. The row can add this citation.
- L112, negative claim "A source has exactly one for each token". The claim holds for the Mamba-3 paper. A search for `micro`, `sub-step`, `multi-step` and `intermediate step` found no line. M3/structure/preliminaries.tex:7: "$x_t$ for the input at time $t$".
- L114, source word "decay". M3/structure/preliminaries.tex:32 does not hold the word. It is at M3/structure/preliminaries.tex:50: "the state-transition decay". The correction above covers the other use, at M3/appendix/discretization.tex:8.
- L138, not in this batch. The row has the same gap as L171. It cites M3/structure/preliminaries.tex:85 for `A`, and that line names only `B`.
- L159, source word `u`. The input letter is also in the paper. M3/appendix/mimo.tex:5: "consider the input $\bU_t \in \mathbb{R}^{D}$". This is not a fault.
- L161, claim "`H` is the state matrix". The other uses agree: M3/appendix/discretization.tex:67 and M3/tables/mimo_arith_intensity.tex:24. A search found no `H` as a count of heads in the paper.
- L181, claim "one Mamba-3 appendix: `L`". The claim holds. `L` as a length is only in M3/appendix/results.tex, at lines 179 and 186. Each other `L` that the searches found is the SSD mask, for example M3/structure/preliminaries.tex:62 and M3/appendix/discretization.tex:57.
- L181, negative claim "No source separates `s` from `t`". The claim holds for the Mamba-3 paper. A search for `micro` and `sub-step` found no line. M3/appendix/discretization.tex:60 uses the index letters `S` and `T`, but these do not name different lengths. M2/ssm.tex:229: "the dimension $\mathtt{S}$ is equal to $\mathtt{T}$".
===== END OF REPORT =====