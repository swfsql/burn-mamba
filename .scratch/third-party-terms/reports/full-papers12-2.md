Batch `papers12-2`: 15 confirmed, 1 wrong, 0 not found, out of 16 citations.
Rows with a correction: L169, L180, L181. All rows are finished.

**The report file does not exist.** `/tmp/source-terms-check/report-papers12-2.md` was not written. The Write tool refused it with "Subagents should return findings as text, not write report files". I did not write the file through Bash, because that would bypass the refusal. A script reads the verdict lines, so save the text below the line to that path. The text is in the exact report format.

The one wrong verdict is L180, `M2/efficient.tex:173`: the `B` on that line is a slot of the `BMM` cost notation, not the batch of the model.

---

L163 | `M2/glossary.tex:22` | confirmed | 22: $M$ & Sequence transformation \textbf{matrix}
L165 | `M2/architecture.tex:181` | confirmed | 181: "one can create $\mathtt{G}$ independent K and V heads", the group count of the grouped head pattern.
L165 | `M2/glossary.tex:31` | confirmed | 31: $G$ & Attention \textbf{Gram} matrix
L166 | `M2/glossary.tex:32` | confirmed | 32: $L$ & (Structured) mask matrix
L166 | `M1/method.tex:87` | confirmed | 87: \Require $x : \mathtt{(B, L, D)}$. M1/background.tex:9 names it: "batch size $B$ and sequence length $L$".
L169 | `M2/glossary.tex:28` | confirmed | 28: $Q$ & Attention \textbf{query} matrix
L171 | `M2/glossary.tex:23` | confirmed | 23: $A$ & Discrete SSM recurrent (state) matrix. The word "Discrete" is on the line of `A` only. The proof for `B` is M2/background.tex:35: "h_{t} &= A_t h_{t-1} + B_t x_t".
L172 | `../papers/multi-gate-res/tex/main.tex:193` | confirmed | 193: \bm{s}_i' = (1-\beta_i)\odot \bm{s}_i + \beta_i \odot \mathcal{F}_l(\bm{h}_l). Line 190: "each of the residual streams could be updated".
L180 | `M1/method.tex:87` | confirmed | 87: \Require $x : \mathtt{(B, L, D)}$. M1/background.tex:116 names it: "of batch size $B$ and length $L$".
L180 | `M2/efficient.tex:173` | wrong | 173: "$\mathsf{BMM}(\mathtt{B}, \mathtt{M}, \mathtt{N}, \mathtt{K})$ ... with batch dimension $\mathtt{B}$". This `B` is a slot of a cost notation. It is not the batch of the model. M2/efficient.tex:184 fills the slot with the chunk count: "$\mathsf{BMM}(\mathtt{T}/\mathtt{Q}, \mathtt{Q}, \mathtt{Q}, \mathtt{N})$". `\mathtt{B}` occurs only at lines 173, 176 and 177, and all three define this notation. The TeX of the paper has no symbol for the batch of the model. Its code listing uses `batch` at M2/code.py:14 and `b` at M2/code.py:25.
L181 | `M2/glossary.tex:15` | confirmed | 15: $\mathtt{T}$ & \textbf{Time} axis or \textbf{target} sequence axis
L181 | `M1/method.tex:87` | confirmed | 87: \Require $x : \mathtt{(B, L, D)}$. A search of M1 for `$T$` and `\mathtt{T}` gives no result.
L182 | `M1/method.tex:227` | confirmed | 227: "expanding the model dimension $D$ by a controllable expansion factor $E$". Line 228 has $3ED^2$.
L182 | `M2/systems.tex:15` | confirmed | 15: "$W^{(x)}, W^{(z)} \in \mathbb{R}^{d \times ed}$ where $e$ is the expansion factor (typically 2)". The search for `ed` and `\mathtt{E}` finds the form only in systems.tex, so "in one section" holds.
L183 | `M2/architecture.tex:134` | confirmed | 134: "The number of heads is defined as $\mathtt{H} = \mathtt{D} / \mathtt{P}$."
L187 | `M2/architecture.tex:181` | confirmed | 181: "one can create $\mathtt{G}$ independent K and V heads, where $1 < \mathtt{G}$ and $\mathtt{G}$ divides $\mathtt{H}$".

## Row corrections

### L169

Cell: Citation
Old: `M2/glossary.tex:28`, `SSM/modules/mamba3.py:186`
New: `M2/efficient.tex:59`, `M2/glossary.tex:28`, `SSM/modules/mamba3.py:186`
Proof: M2/efficient.tex:59: "decomposing into chunks of length $\mathtt{Q}=3$". The row names `Q` as the chunk length, and no citation of the row shows that use in the paper.

### L180

Cell: Spelling in each source
Old: Mamba-1 and Mamba-2 papers: `B`.
New: Mamba-1 paper: `B`. Mamba-2 paper: no symbol for the batch of the model. Its code listing uses `batch` and `b`. Its cost notation `BMM` has a batch slot `B`. Most SSD costs fill that slot with `T/Q`, which is our `n`.
Proof: M2/efficient.tex:173: "with batch dimension $\mathtt{B}$". M2/efficient.tex:184: "$\mathsf{BMM}(\mathtt{T}/\mathtt{Q}, \mathtt{Q}, \mathtt{Q}, \mathtt{N})$". M2/code.py:14: "X: (batch, length, n_heads, d_head)". M2/code.py:25: "b (c l) ... -> b c l ...". M2/efficient.tex:152 includes code.py in the paper. M2/efficient.tex:200 fills the slot with `1`, so the text says "most".

Cell: Citation
Old: `M1/method.tex:87`, `M2/efficient.tex:173`, `SSM/modules/ssd_minimal.py:86`
New: `M1/method.tex:87`, `M2/code.py:14`, `M2/code.py:25`, `M2/efficient.tex:173`, `M2/efficient.tex:184`, `SSM/modules/ssd_minimal.py:86`
Proof: The same lines.

### L181

Cell: Spelling in each source
Old: Mamba-1 paper and one Mamba-3 appendix: `L`. No source separates `s` from `t`.
New: Mamba-1 paper, one section of the Mamba-2 paper and one Mamba-3 appendix: `L`. No source separates `s` from `t`. Mamba-2 has `S` and `T` for the source and target axes of attention. Its SSM sets `S` = `T`.
Proof: M2/systems.tex:15: "a single input $u \in \mathbb{R}^{L \times d}$". M2/attention.tex:38: "$\mathtt{S}$ and $\mathtt{T}$ represent \emph{source} and \emph{target} sequence lengths". M2/attention.tex:56: "separating the notation for dimensions (e.g.\ $\mathtt{S}$ and $\mathtt{T}$)". M2/ssd.tex:44 gives the SSM form `(T,S)`, and M2/ssd.tex:49 says "where $\mathtt{S}=\mathtt{T}$". A reader who sees `(T,S)` at M2/ssd.tex:44 can map these letters to our `t` and `s`.

Cell: Citation
Old: `M2/glossary.tex:15`, `M3/structure/preliminaries.tex:6`, `M1/method.tex:87`, `M3/appendix/results.tex:179`
New: `M2/glossary.tex:15`, `M3/structure/preliminaries.tex:6`, `M1/method.tex:87`, `M2/systems.tex:15`, `M3/appendix/results.tex:179`, `M2/attention.tex:38`, `M2/ssd.tex:49`
Proof: The same lines.

## Extra findings

- L181, negative claim "No source separates `s` from `t`". A search of M1, M2 and BLOG for `micro`, `sub-step`, `write axis` and `read axis` gives no result. For these sources the claim holds for our two axes. The `S` and `T` letters of M2 are in the correction above. This check does not cover the other sources of the row.
- L181, sequence-length `L` in M2. In systems.tex, the form occurs from line 15 to line 60. A search outside systems.tex for `L` near the words length, axis, sequence or layers finds only the layer count and one mask use (M2/attention.tex:245). That search is a filter, not a read of each `L`.
- L166, a third meaning of `L`. M2/experiments.tex:240: "a Transformer with $L$ layers". M2/experiments.tex:242 has "$\frac{L}{2}$ SSD layers". The row lists only the mask matrix and the sequence length. The sentence defines the letter, so a wrong port is not likely. I give no correction.
- L166, uncited M2 use. M2/systems.tex:15 also shows `L` as a sequence length. The row cites only M1 for that meaning.
- L169, BLOG use of the chunk-length `Q`. BLOG/part3.md:86: "blocks of size $ \mathtt{Q} \times \mathtt{Q} $". The blog also uses `Q` as the query at BLOG/part2.md:151: "Y = Q \cdot (K^\top V)".
- L165, L169 and L180, typeface. M2 separates dimensions from matrices by typeface. M2/glossary.tex:7 and M2/glossary.tex:8 separate the "tensor dimensions" from the "Matrices and tensors". `\mathtt{G}`, `\mathtt{Q}` and `\mathtt{B}` are dimensions. `$G$`, `$Q$` and `$B$` are matrices. The rows do not state this rule. A reader can use it to separate the two meanings.
- L171, a possible exception to "Discrete in Mamba-2". M2/systems.tex:21 and M2/systems.tex:41 have "$SSM_{A, B, C, \Delta}$", and M2/systems.tex:39 has "\Delta, B, C &= \text{projection}(u)". `\Delta` occurs only in systems.tex. The paper does not define this form, so these lines do not prove that `A` and `B` are continuous there. They can be continuous. Other parts of the paper use the discrete form: M2/background.tex:35, M2/scan.tex:5. I give no correction.
- L171, Mamba-1. The row makes no claim about M1. M1 has both forms: M1/background.tex:38 "h'(t) &= \A h(t) + \B x(t)" and M1/background.tex:48 "h_{t} &= \dA h_{t-1} + \dB x_t". M1/method.tex:119 gives the map: "\dA, \dB ... \gets \mathsf{discretize}(\dt, \A, \B)".
- L180, the comment "(B, T, Q, D, P) in the paper". A search of the M2 TeX files and BLOG does not find this tuple. The row quotes only the comment, so the row is correct. A reader cannot find the tuple in the paper.
- L182, a false friend that the doc does not list. "State expansion factor" in M2 is `N`, our `r`. It is not the `e` of this row. M2/background.tex:131: "We also call it the \textbf{state expansion factor}". M2/efficient.tex:15: "state expansion factor $\mathtt{N}$". M2/related.tex:30: "the SSM state size (also called state expansion factor)". In docs/source-terms.md, a search for `state expansion` gives no result. If the lead wants this fact in L182, add this sentence to the cell "Spelling in each source" after the Mamba-2 sentence: The "state expansion factor" of the Mamba-2 paper is `N`, our `r`. Add the citation `M2/background.tex:131`.
- L183, the formula of the cited line. M2/architecture.tex:134 gives `H` = `D` / `P`, with the model dimension `D`. The Notation table of CLAUDE.md gives `h` = `d_inner` / `per_head_dim`. The row gives only the spelling, so the row is correct. A reader who ports the formula from this line can leave out the expansion factor.