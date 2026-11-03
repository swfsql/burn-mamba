Batch `papers12-1`: 57 citations. 53 confirmed, 4 wrong, 0 not found. All rows are finished. Corrections: L42, L45, L61, L80, L98, L100, L109, L110, L157. I wrote no report file, as you instructed. The report follows.

```text
L39 | `M1/background.tex:27` | confirmed | 27: "an implicit latent state h(t)".
L39 | `M2/related.tex:30` | confirmed | 30: "N is the SSM state size".
L42 | `M2/architecture.tex:26` | confirmed | 26: "inner sequence mixing layer (i.e. outside the core SSD layer)".
L42 | `M2/architecture.tex:36` | confirmed | 36: "the SSD layer is viewed as a map from A, X, B, C to Y".
L42 | `M2/architecture.tex:167` | confirmed | 167: "The selective SSM (S6) layer of the Mamba architecture".
L42 | `BLOG/part1.md:29` | confirmed | 29: "SSD model refers to a specific standalone layer, like attention or an SSM". The literal "SSD layers" is at :31.
L42 | `BLOG/part1.md:55` | confirmed | 55: "its core "S6" layer".
L43 | `M1/method.tex:223` | confirmed | 223: "interleaved with an MLP (multi-layer perceptron) block".
L43 | `M1/method.tex:237` | confirmed | 237: "two stacks of the block", "interleaved MHA ... and MLP blocks".
L45 | `M1/background.tex:8` | confirmed | 8: "Structured SSMs independently map each channel". The plural names the class, not a count. See Row corrections.
L50 | `M2/efficient.tex:27` | confirmed | 27: "block decomposition of the matrix".
L50 | `M2/efficient.tex:40` | confirmed | 40: "for some block size Q".
L50 | `M2/efficient.tex:103` | confirmed | 103: "The j-th block represents ... for the range R".
L50 | `M2/code.py:11` | confirmed | 11: def ssd(X, A, B, C, block_len=64, initial_states=None):
L50 | `M2/code.py:24` | confirmed | 24: # Rearrange into blocks/chunks
L50 | `BLOG/part3.md:95` | confirmed | 95: "split the sequence of input into blocks (or chunks) of size Q".
L51 | `M1/method.tex:277` | confirmed | 277: \label{thm:gating}, the start of the theorem. N=1 is at :278.
L51 | `M1/method.tex:290` | confirmed | 290: "g_t = sigma(Linear(x_t)) (the gate)".
L51 | `M1/method.tex:332` | confirmed | 332: "It generalizes RNN gates (e.g. g_t in ...)".
L53 | `M2/glossary.tex:15` | confirmed | 15: "Time axis or target sequence axis".
L53 | `M2/background.tex:116` | confirmed | 116: "T represents the sequence or time axis".
L55 | `M2/efficient.tex:108` | confirmed | 108: "supposing that the initial state (to the chunk) is 0".
L55 | `M2/systems.tex:102` | confirmed | 102: "each worker takes an initial state".
L56 | `M2/scan.tex:191` | confirmed | 191: "states are passed from one local segment to another".
L56 | `M1/appendix.tex:369` | confirmed | 369: "S=34021 segments of length 2^17 ... that cover the genome".
L56 | `M2/code.py:1` | confirmed | 1: def segsum(x). The index-range difference is at :6.
L57 | `M1/method.tex:313` | confirmed | 313: "selective models can simply reset their state at any time".
L57 | `M1/method.tex:320` | confirmed | 320: "reset their state at boundaries (e.g. Delta_t to infinity, or ... when g_t to 1)".
L58 | `M1/method.tex:201` | confirmed | 201: "the fused selective scan layer".
L59 | `M2/architecture.tex:70` | confirmed | 70: "The parameters may be tied across heads".
L61 | `M2/efficient.tex:9` | wrong | 9: "an algorithm for computing SSD models". The quoted word is not on the line or in its context. The proof is at M2/efficient.tex:30: "implementation of the SSD algorithm".
L61 | `M2/scan.tex:114` | confirmed | 114: "A state-passing mode where information is propagated forward in chunks".
L61 | `M2/scan.tex:118` | wrong | 118: \end{itemize}. The proof is at M2/scan.tex:113: "A dilated mode where information is propagated 1, 2, 4, 8, ... steps at a time".
L75 | `M2/architecture.tex:66` | confirmed | 66: "defining one head of the sequence transformation". "Multihead patterns" is at :67, in context.
L75 | `M2/architecture.tex:70` | confirmed | 70: "leading to a head pattern".
L75 | `M2/architecture.tex:162` | confirmed | 162: "multi-value attention (MVA) of multi-input SSM (MIS) pattern".
L75 | `M2/architecture.tex:185` | confirmed | 185: "grouped-input SSM (GIS), or synonymously grouped-value attention (GVA)".
L80 | `M1/method.tex:108` | confirmed | 108: \caption{SSM + Selection (S6)}
L80 | `M1/method.tex:397` | confirmed | 397: "abbreviate selective SSMs as S6 models".
L80 | `M2/architecture.tex:167` | confirmed | 167: "The selective SSM (S6) layer".
L83 | `M2/efficient.tex:101` | confirmed | 101: \subsection{Diagonal Blocks}
L83 | `M2/efficient.tex:111` | confirmed | 111: \subsection{Low-Rank Blocks}
L84 | `M2/scan.tex:114` | confirmed | 114: "A state-passing mode".
L84 | `M2/scan.tex:191` | confirmed | 191: "we call this "state-passing"".
L98 | `M2/architecture.tex:28` | wrong | 28: \paragraph{Parallel Parameter Projections.} The quote drops a word. The proof is at M2/architecture.tex:41: "adopting parallel projections for the A, B, C, X inputs".
L98 | `M2/architecture.tex:37` | confirmed | 37: "in parallel with a single projection at the beginning of the block".
L100 | `M2/systems.tex:110` | confirmed | 110: \label{subsec:varlen}, under "Variable Length" at :109.
L100 | `M2/systems.tex:115` | confirmed | 115: "variable sequence lengths by simply treating the whole batch as one long sequence".
L100 | `M2/systems.tex:116` | wrong | 116: "setting A_t = 0 for tokens t at the end of one sequence". This value is on the last token, not an index or an offset, and not on the start of a sequence. See Row corrections.
L109 | `M2/glossary.tex:24` | confirmed | 24: "State space model input projection (expansion) matrix".
L110 | `M2/glossary.tex:25` | confirmed | 25: "State space model output projection (contraction) matrix".
L110 | `M2/architecture.tex:152` | confirmed | 152: "the C parameter which controls the SSM state contraction".
L157 | `M1/method.tex:87` | confirmed | 87: x : (B, L, D).
L157 | `M1/method.tex:116` | confirmed | 116: \B : (B, L, N) gets s_B(x).
L158 | `M2/glossary.tex:16` | confirmed | 16: "Source sequence axis (in attention)".
L159 | `M2/systems.tex:15` | confirmed | 15: "a single input u in R^{L x d}".
L161 | `M2/ssm.tex:247` | confirmed | 247: "contracting the hidden state H by the output matrix C".
```

## Row corrections

### L42

Cell: Its meaning in the source
Old: Papers, blog and `SSR`: "SSD layer", "selective SSM layer", "S6 layer" and "core SSD layer" name the recurrence inside the block.
New: Papers, blog and `SSR`: "SSD layer", "selective SSM layer", "S6 layer" and "core SSD layer" name the recurrence inside the block. Exception: in the Mamba-1 appendix, "selective SSM layer" names our block, with the projections, the convolution and the activation.
Proof: M1/appendix.tex:112: "the entire selective SSM block (input projection, convolution, activation, scan, output projection)". M1/appendix.tex:117: "As a result, the selective SSM layer has the same memory requirement".

Cell: Citation
Old: `M2/architecture.tex:26`
New: `M2/systems.tex:51`, `M1/appendix.tex:112`, `M1/appendix.tex:117`, `M2/architecture.tex:26`
Proof: M2/systems.tex:51: "the TP Mamba-2 layer can be written as". Lines 53 to 61 hold the projections, conv1d, the SSM, the gate, groupnorm and the output. No current citation of the row shows "Mamba-2 layer".

### L45

Cell: Its meaning in the source
Old: Papers: the source counts several SSMs in one layer, one for each head or channel.
New: Mamba-3 paper: the source counts several SSMs in one layer, one for each head or channel. Mamba-1 and Mamba-2 papers: one SSM applies independently to each channel. The Mamba-1 appendix counts one selective SSM for each layer.
Proof: M1/background.tex:117: "the SSM is applied independently to each channel". M2/background.tex:119: "viewing the input as P independent sequences and applying the SSM to each". M1/appendix.tex:124: "two layers of selective SSMs".

Cell: Citation
Old: `M1/background.tex:8`
New: `M1/background.tex:8`, `M1/background.tex:117`, `M1/appendix.tex:124`, `M2/background.tex:119`
Proof: the same lines.

### L61

Cell: Citation
Old: `M2/efficient.tex:9`
New: `M2/efficient.tex:30`
Proof: M2/efficient.tex:30: "implementation of the SSD algorithm".

Cell: Citation
Old: `M2/scan.tex:118`
New: `M2/scan.tex:113`
Proof: M2/scan.tex:113: "A dilated mode". This mode has no chunks.

### L80

Cell: Meaning
Old: The state recurrence of Mamba-1.
New: The state recurrence of Mamba-1. The SSD layer of Mamba-2 is a special case of the selective SSM.
Proof: M2/background.tex:185: "The state space dual (SSD) layer can be defined as a special case of the selective SSM". BLOG/part2.md:19 says the same.

Cell: Citation
Old: `M2/architecture.tex:167`
New: `M2/architecture.tex:167`, `M2/background.tex:185`, `M1/method.tex:165`
Proof: M1/method.tex:165: "Overview of Selective Scan". No current citation of the row shows "selective scan".

### L98

Cell: Citation
Old: `M2/architecture.tex:28`
New: `M2/architecture.tex:41`
Proof: M2/architecture.tex:41: "adopting parallel projections".

### L100

Cell: Meaning
Old: The source holds an index or an offset, not a mark.
New: The official code holds an index or an offset, not a mark. The Mamba-2 paper and the blog set `A_t = 0` on the last token of a sequence. In the recurrence `h_t = A_t h_{t-1} + B_t x_t`, that value cuts the state that enters the last token. It does not cut the state that leaves the last token. A **Reset** is on the first token of the next sequence.
Proof: M2/systems.tex:116: "A_t = 0 for tokens t at the end of one sequence". BLOG/part4.md:61: "for tokens at the end of each sequence". M2/background.tex:35: "h_{t} &= A_t h_{t-1} + B_t x_t". M2/code.py:6 agrees: the entry (i, j) sums `A` over j+1 to i. I read this from the equation and the listing. I did not run the code.

Cell: Citation
Old: `M2/systems.tex:116`
New: `M2/systems.tex:116`, `BLOG/part4.md:61`, `M2/background.tex:35`
Proof: the same lines.

### L109

Cell: Meaning
Old: "Expansion" is not the `expand` of `d_inner`.
New: "Expansion" of `B` is not the `expand` of `d_inner`. "Expansion factor" (`E`, `e`) is the `expand` in the Mamba-1 paper and in one section of the Mamba-2 paper. "State expansion factor" `N` is `state_rank`.
Proof: M1/method.tex:227: "a controllable expansion factor E". M2/systems.tex:15: "e is the expansion factor (typically 2)". M2/background.tex:131: "We also call it the state expansion factor".

Cell: Citation
Old: `M2/glossary.tex:24`
New: `M2/glossary.tex:24`, `M2/background.tex:61`, `M1/method.tex:227`, `M2/systems.tex:15`, `M2/background.tex:131`
Proof: M2/background.tex:61: "B = f_B(Delta, B-ring)". This line proves the uncited claim "In Mamba-2, `B` holds the step size". M2/glossary.tex:24 does not show it.

### L110

Cell: Meaning
Old: `C` of the plant.
New: `C` of the plant. "Output projection" also names the last linear map of the block, `W^{(o)}`. That map is `out_proj`, not the readout.
Proof: M2/systems.tex:15: "output projection matrix W^{(o)} in R^{ed x d}". M2/architecture.tex:47: "right before the final output projection". M1/method.tex:229: "ED^2 for output projection".

Cell: Our term
Old: **Readout**
New: **Readout** for `C`. `out_proj` for the last linear map of the block.
Proof: src/mamba2/mamba2.rs:184: `pub out_proj: Linear`. The glossaries have no term for this map.

Cell: Citation
Old: `M2/architecture.tex:152`
New: `M2/architecture.tex:152`, `M2/systems.tex:15`, `M2/architecture.tex:47`, `M1/method.tex:229`
Proof: the same lines.

### L157

Cell: Source use
Old: `B` is both the batch and the input projection in the Mamba-1 paper.
New: `B` is both the batch and the matrix `B` of the SSM in the Mamba-1 paper. The "input projections" of that paper are other maps, at the entry of the block.
Proof: M1/method.tex:229: "2ED^2 for input projections". M1/method.tex:232 counts the projections for `Delta`, `B`, `C` separately, as SSM parameters.

## Extra findings

- L39: the uncited source words are present. "hidden state": M1/background.tex:118, M2/scan.tex:24. "recurrent state": M2/scan.tex:187.
- L42: the negative claim holds for M1, M2 and the blog. A search for residual, Pre-LN, pre-norm and LayerNorm found no name for a norm and residual around one block. The closest line is M1/method.tex:235: "interleaved with standard normalization and residual connections".
- L42: "entire layer" and "layer output" are not in M1, M2 or the blog. Only M3 can support them.
- L42: "SSD layers" also counts the units of a stack. M2/experiments.tex:199: "Mamba-2: 64 SSD layers". M2/experiments.tex:202: "28 SSD layers and 4 attention layers, interleaving with 32 gated MLP layers". The count of plants is the same, so I made no correction. The lead can add a note.
- L53: "Papers and official code: one axis" does not cover the `S` of Mamba-2. M2/glossary.tex:16 gives a source axis. M2/attention.tex:38 separates `S` and `T`. M2/attention.tex:46 gives `S=T` as a case of self-attention. L158 states the fact, so a pointer to L158 is sufficient.
- L55: `initial_states` is also in the Mamba-2 listing, at M2/code.py:11 and M2/code.py:41 to 43. The row names only the official code for this word. No wrong fact follows.
- L56: "segments" also names a chunk in the blog. BLOG/part1.md:247: "splits the sequence into segments". `segsum` is also at BLOG/part3.md:116. The row does not name the blog.
- L58: M1 also holds the kernel words. M1/method.tex:170: "kernel fusion". M1/method.tex:205: "the fused kernel". M1/method.tex:190 to 192 give the operations in one pass.
- L61: "chunked algorithm" is not in M1, M2 or the blog. It is at M3/structure/methods_mimo.tex:52 and :53. The exact "SSD algorithm" is also at M2/intro.tex:55.
- L83: M2/efficient.tex:166 is a stronger line for three of the words: "Diagonal blocks represent intra-chunk computations and the off-diagonal blocks represent inter-chunk computations". M2 pairs "inter-chunk" with "off-diagonal blocks" here, and "Low-Rank Blocks" is the subsection name.
- L84: the unhyphenated form is at M2/scan.tex:353: "The state passing (chunkwise) algorithm". It is also at BLOG/part3.md:93.
- L100: in M2, "varlen" is only the label at M2/systems.tex:110. A search of M1, M2 and the blog found no "varlen mode".
- L109: "input projection" has the same split as "output projection", and the row does not list it. M2/glossary.tex:24 uses it for `B`. M2/systems.tex:15 uses it for `W^{(x)}, W^{(z)}`.
- L109: M2/systems.tex:39 lists "Delta, B, C = projection(u)" as separate outputs. M2/background.tex:66 says that the paper omits the discretization step after that section. The claim "`B` holds the step size" rests on M2/background.tex:61 only.
- L161: "`h` is the state" holds for my sources: M1/background.tex:27, M2/background.tex:11, BLOG/part1.md:47.