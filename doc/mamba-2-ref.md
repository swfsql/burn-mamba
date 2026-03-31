# Mamba-2: A Complete Technical Reference

> A standalone, formula-focused technical document covering the SSD model, SSD framework, SSD algorithm, architecture, systems optimizations, and kernel implementation.

---

## 1. Overview and Motivation

Mamba-2 addresses two core limitations of Mamba-1:

**Conceptual gap.** Mamba-1's selective SSM lacked a formal connection to the dominant attention paradigm. Mamba-2 establishes an exact mathematical duality between a restricted class of SSMs and a form of linear attention, unifying these two families under the *Structured State Space Duality (SSD)* framework.

**Hardware efficiency.** Mamba-1's hardware-aware selective scan does not leverage tensor cores (the matrix-multiply units that provide up to 16× more throughput than scalar arithmetic on modern GPUs). By recasting the SSM computation as matrix multiplications, Mamba-2 enables significantly faster training.

| GPU   | BF16 Matmul | FP32 Arithmetic |
|-------|-------------|-----------------|
| A100  | 312 TFLOPS  | 19 TFLOPS       |
| H100  | 989 TFLOPS  | 67 TFLOPS       |

The central object of Mamba-2 is the **SSD layer**, which can be computed either as a structured recurrence (the *linear mode*, efficient for inference) or as a masked matrix multiplication (the *quadratic mode*, efficient for training via tensor cores). A hybrid *chunkwise* algorithm combines both to get the best of each.

---

## 2. The SSD Model

### 2.1 The Linear (Recurrent) Mode

The SSD layer is built on the **selective state space model (SSM)**:

$$
h_t = A_t\, h_{t-1} + B_t\, x_t \tag{1}
$$
$$
y_t = C_t^\top h_t \tag{2}
$$

**Dimensions and types:**

| Symbol | Shape | Description |
|--------|-------|-------------|
| $x_t$ | scalar ($\mathbb{R}$) or $\mathbb{R}^\mathtt{P}$ (per head) | Input at time $t$ |
| $h_t$ | $\mathbb{R}^\mathtt{N}$ | Hidden state (state dimension $\mathtt{N}$) |
| $y_t$ | scalar or $\mathbb{R}^\mathtt{P}$ | Output at time $t$ |
| $A_t$ | $\mathbb{R}^{(\mathtt{N} \times \mathtt{N})}$ in general; $\mathbb{R}^{()}$ in SSD | State transition matrix (scalar in SSD) |
| $B_t$ | $\mathbb{R}^\mathtt{N}$ | Input projection |
| $C_t$ | $\mathbb{R}^\mathtt{N}$ | Output projection |

**The SSD restriction on $A$.** The single defining modification of SSD relative to general selective SSMs is:

> **SSD constrains $A_t$ to be a *scalar times identity* matrix.**

This means that instead of a diagonal matrix of shape $\mathtt{(T, N)}$, $A_t$ is a single scalar $a_t \in \mathbb{R}$, with $A_t = a_t \cdot I_N$. Consequently, $A$ has shape $\mathtt{(T)}$.

**Multi-head formulation.** For a head with $\mathtt{P}$ channels sharing the same $(A, B, C)$ parameters:

$$
Y^{\mathtt{(T,P)}} = \mathsf{SSM}\!\left(A^{\mathtt{(T)}},\, B^{\mathtt{(T,N)}},\, C^{\mathtt{(T,N)}}\right)\!\left(X^{\mathtt{(T,P)}}\right) \tag{3}
$$

Each of the $\mathtt{P}$ channels of $X$ is processed by the same SSM dynamics independently. The full model consists of multiple such heads, analogous to multi-head attention, typically with $\mathtt{P} = 64$ or $\mathtt{P} = 128$.

### 2.2 The Quadratic (Attention) Mode

Given the same parameters $(A^{\mathtt{(T)}}, B^{\mathtt{(T,N)}}, C^{\mathtt{(T,N)}})$, define the **1-semiseparable (1-SS) mask matrix** $L \in \mathbb{R}^{\mathtt{T} \times \mathtt{T}}$:

$$
L = \begin{bmatrix}
1 &   &   &   \\
a_1 & 1 &   &   \\
a_2 a_1 & a_2 & 1 &   \\
\vdots & \vdots & \ddots & \ddots \\
a_{T-1} \cdots a_1 & a_{T-1} \cdots a_2 & \cdots & a_{T-1} & 1
\end{bmatrix}
\tag{4}
$$

In compact notation, the $(i, j)$ entry of $L$ (for $i \geq j$) is the **cumulative product**:

$$
L_{ij} = a_{i:j}^\times := a_i \cdot a_{i-1} \cdots a_{j+1} \quad (i \geq j), \qquad L_{ij} = 0 \quad (i < j) \tag{5}
$$

with the convention $L_{ii} = 1$ (empty product).

The **attention-like (quadratic) form** of the SSD layer is then:

$$
M = L \circ CB^\top \in \mathbb{R}^{\mathtt{(T,T)}} \tag{6}
$$
$$
Y = M \cdot X \tag{7}
$$

where $\circ$ denotes elementwise (Hadamard) product, and $CB^\top$ is the standard $\mathtt{(T,T)}$ outer product over the state dimension $\mathtt{N}$, i.e.:

$$
(CB^\top)_{ij} = C_i^\top B_j \in \mathbb{R} \tag{8}
$$

This is the SSD layer expressed as a **masked linear attention** computation:

$$
Y = (L \circ CB^\top)\, X \tag{9}
$$

which is directly analogous to causal linear attention $Y = (L_{\text{causal}} \circ QK^\top)\, V$ under the renaming $(C, B, X) \mapsto (Q, K, V)$.

### 2.3 State Space Duality

The **duality claim** is that equations (1)–(2) and (6)–(7) define exactly the same function:

$$
(A^{\mathtt{(T)}},\, B^{\mathtt{(T,N)}},\, C^{\mathtt{(T,N)}},\, X^{\mathtt{(T,P)}}) \;\mapsto\; Y^{\mathtt{(T,P)}}
$$

**Proof (direct).** Unrolling the recurrence (1)–(2), the SSM defines a matrix transformation $Y = MX$ where:

$$
M_{ij} = C_i^\top A_{i:j}^\times B_j := C_i^\top (A_i \cdots A_{j+1})\, B_j \tag{10}
$$

Because every $A_t = a_t \cdot I$ is a scalar multiple of identity, the product $A_{i:j}^\times = a_{i:j}^\times \cdot I$ is also a scalar times identity. Thus:

$$
M_{ij} = C_i^\top (a_{i:j}^\times \cdot I)\, B_j = a_{i:j}^\times\, (C_i^\top B_j) = L_{ij} \cdot (CB^\top)_{ij}
$$

which is exactly $(L \circ CB^\top)_{ij}$. $\square$

The full SSM matrix representation for scalar-identity $A$ is:

$$
M = \begin{bmatrix}
C_0^\top B_0 &   &   &   \\
a_1 C_1^\top B_0 & C_1^\top B_1 &   &   \\
a_2 a_1 C_2^\top B_0 & a_2 C_2^\top B_1 & C_2^\top B_2 &   \\
\vdots & \vdots & \ddots & \ddots
\end{bmatrix}
\tag{11}
$$

This matrix $M$ is a **lower-triangular 1-semiseparable matrix**: every submatrix fully contained in the lower-triangular portion is rank-1 (low-rank with rank $\leq \mathtt{N}$, but here exactly rank 1 because $A$ is scalar).

### 2.4 Comparison to Mamba-1 and Attention

**SSD vs. Mamba-1 (S6):**

| Property | Mamba-1 (S6) | Mamba-2 (SSD) |
|----------|--------------|----------------|
| Structure of $A_t$ | Diagonal, shape $\mathtt{(T,N)}$ | Scalar × identity, shape $\mathtt{(T)}$ |
| Head dimension $\mathtt{P}$ | 1 (each channel is an independent SSM) | $\mathtt{P} > 1$ (e.g., 64 or 128) |
| Total state per head | $\mathtt{N}$ (each channel independently) | $\mathtt{P} \times \mathtt{N}$ (all channels share one recurrence) |
| Typical state dim $\mathtt{N}$ | 16 | 64 to 256+ |
| Uses tensor cores | No | Yes (via quadratic/chunkwise form) |

The scalar-identity constraint on $A$ means that, within a single head, all $\mathtt{N}$ state dimensions share the same scalar decay $a_t$ at each step, and all $\mathtt{P}$ channels share these dynamics. This weight-tying is the key that enables the attention-dual form.

**SSD vs. standard attention:**

| Property | Standard Attention | SSD |
|----------|--------------------|-----|
| Attention matrix | $\text{softmax}(QK^\top)$ | $L \circ CB^\top$ (no softmax) |
| State/context size | $\mathtt{T}$ (grows with sequence) | $\mathtt{N}$ (constant) |
| Causal mask | Binary lower-triangular $L_{\text{causal}}$ | 1-SS matrix $L$ with input-dependent entries |
| Positional encoding | Separate (RoPE, etc.) | Implicitly via $L$: the weight $a_{i:j}^\times$ discounts by distance |
| Training FLOPs | $O(\mathtt{T}^2 \mathtt{N})$ | $O(\mathtt{T}\mathtt{N}^2)$ |
| Inference FLOPs | $O(\mathtt{T}\mathtt{N})$ | $O(\mathtt{N}^2)$ per step |

The mask $L$ acts as **input-dependent relative positional encoding**: the attention score $C_i^\top B_j$ is attenuated by $a_{i:j}^\times = a_i \cdots a_{j+1}$, a "discount factor" that depends on the inputs and scales with the separation between positions $i$ and $j$.

---

## 3. The SSD Framework

The SSD framework consists of two independent theoretical perspectives on the SSD model, each generalizing it in a different direction.

### 3.1 Framework 1: Structured Matrix Transformations

#### Matrix Sequence Transformations

A **matrix sequence transformation** (or "matrix mixer") is a sequence model $X \mapsto Y$ of the form:

$$
Y = M(X) \cdot X
$$

where $M$ is a (possibly input-dependent) $\mathtt{T} \times \mathtt{T}$ matrix. SSMs are matrix transformations: unrolling the recurrence gives $M_{ij} = C_i^\top A_{i:j}^\times B_j$ as in (10).

#### Semiseparable Matrices

The SSM matrix $M$ is a **lower-triangular semiseparable matrix**, defined by the property that every submatrix fully contained in the strictly lower-triangular part has rank $\leq \mathtt{r}$ for some rank parameter $\mathtt{r}$. For general (diagonal) $A$, the rank is $\mathtt{r} = \mathtt{N}$. For scalar-identity $A$ (SSD), the rank is $\mathtt{r} = 1$ — a **1-semiseparable matrix**.

The general semiseparable matrix entry for a diagonal-$A$ SSM is:

$$
M_{ij} = C_i^\top \left(\prod_{k=j+1}^{i} \text{diag}(a_k)\right) B_j, \quad i \geq j \tag{12}
$$

where $\text{diag}(a_k)$ is the diagonal matrix with entries $a_k \in \mathbb{R}^\mathtt{N}$.

For the SSD case (scalar $A$), this simplifies to equation (10).

#### Central Takeaway

> All algorithms for computing SSMs correspond to **structured matrix multiplication algorithms** on semiseparable matrices.

The two modes of SSD are:
- **Linear mode:** a *sequential* structured matrix multiplication algorithm, exploiting the recurrent structure.
- **Quadratic mode:** the *naive* dense matrix multiplication algorithm, materializing $M$ explicitly.

### 3.2 Framework 2: Structured Masked Attention

#### Kernel Attention and Linear Attention

**Kernel attention** defines a sequence transformation via:

$$
Y = (QK^\top) \cdot V
$$

where $Q, K \in \mathbb{R}^{\mathtt{(T,N)}}$, $V \in \mathbb{R}^{\mathtt{(T,P)}}$. When $\mathtt{N}$ is small (or constant), this can be linearized via associativity of matrix multiplication:

$$
Y = Q \cdot (K^\top V)
$$

reducing from $O(\mathtt{T}^2 \mathtt{N})$ to $O(\mathtt{T}\mathtt{N}^2)$ time.

**Causal linear attention** extends this with a lower-triangular causal mask $L_{\text{causal}}$:

$$
Y = (L_{\text{causal}} \circ QK^\top) \cdot V \tag{13}
$$

where $L_{\text{causal}}$ has all-ones below the diagonal. The efficient form uses cumulative sums:

$$
Y = Q \cdot \mathsf{cumsum}(K^\top V)
$$

This is because multiplication by $L_{\text{causal}}$ encodes exactly a cumulative sum: $y = L_{\text{causal}} \cdot x \iff y = \mathsf{cumsum}(x)$.

#### Structured Masked Attention (SMA)

The SMA framework generalizes causal linear attention by allowing $L$ to be **any structured matrix**. Formally, SMA is the **four-way tensor contraction**:

$$
Y = \mathsf{contract}(\mathtt{TN}, \mathtt{SN}, \mathtt{SP}, \mathtt{TS} \to \mathtt{TP})(Q, K, V, L) \tag{14}
$$

**Quadratic (standard) reduction order** — contract $(Q,K)$ first, then apply $L$, then contract with $V$:

$$
\begin{aligned}
G &= \mathsf{contract}(\mathtt{TN}, \mathtt{SN} \to \mathtt{TS})(Q, K) \\
M &= \mathsf{contract}(\mathtt{TS}, \mathtt{TS} \to \mathtt{TS})(G, L) \\
Y &= \mathsf{contract}(\mathtt{TS}, \mathtt{SP} \to \mathtt{TP})(M, V)
\end{aligned}
\tag{15}
$$

**Linear (efficient) reduction order** — contract $(V,K)$ first, multiply by $L$, then contract with $Q$:

$$
\begin{aligned}
Z &= \mathsf{contract}(\mathtt{SP}, \mathtt{SN} \to \mathtt{SPN})(V, K) \\
H &= \mathsf{contract}(\mathtt{TS}, \mathtt{SPN} \to \mathtt{TPN})(L, Z) \\
Y &= \mathsf{contract}(\mathtt{TN}, \mathtt{TPN} \to \mathtt{TP})(Q, H)
\end{aligned}
\tag{16}
$$

The second line of (16) is a matrix multiplication by $L$, which is efficient when $L$ is structured (admits subquadratic matrix-vector multiplication). The two reduction orders are equivalent by the **associativity of tensor contractions** — a generalization of matrix multiplication associativity.

#### SSD as 1-SS Structured Masked Attention

The SSD model is the special case of SMA where:

$$
L = \begin{bmatrix}
1 & & \\
a_1 & 1 & \\
a_2 a_1 & a_2 & 1 & \\
\vdots & & \ddots
\end{bmatrix}
\tag{17}
$$

This is the **1-semiseparable mask** derived from the scalar SSM parameters $\{a_t\}$. Computing $y = Lx$ corresponds to a scalar linear recurrence:

$$
y_t = a_t y_{t-1} + x_t \tag{18}
$$

which is exactly the SSM update (1) when $B_t = 1$ and $C_t = 1$.

**Structured mask taxonomy:**

| Mask $L$ | Sequence Model |
|----------|----------------|
| All-ones lower-triangular | Causal Linear Attention |
| Lower-triangular with constant decay $\lambda^{i-j}$ | Retentive Network (RetNet) |
| 1-semiseparable (input-dependent decay) | SSD (Mamba-2) |
| Toeplitz | Toeplitz Structured Attention |
| DFT matrix | Fourier Structured Attention |

---

## 4. The SSD Algorithm

The SSD algorithm combines both the linear and quadratic modes to achieve efficient training with tensor cores while maintaining linear complexity in sequence length.

### 4.1 Block Matrix Decomposition View

Partition the semiseparable matrix $M$ into blocks of size $\mathtt{Q} \times \mathtt{Q}$ (the chunk size). The block structure of a semiseparable matrix is:

$$
M = \begin{bmatrix}
D_0 & & \\
B_1 A_{10} C_0^\top & D_1 & \\
B_2 A_{20} C_0^\top & B_2 A_{21} C_1^\top & D_2 & \\
\vdots & & \ddots
\end{bmatrix}
\tag{19}
$$

where:
- **Diagonal blocks** $D_k$: each is itself a smaller $\mathtt{Q} \times \mathtt{Q}$ semiseparable matrix (the *intra-chunk* interaction).
- **Off-diagonal blocks** $B_j A_{jk} C_k^\top$: each factors as a **low-rank matrix** — specifically a product of an $\mathtt{Q} \times \mathtt{N}$ matrix, an $\mathtt{N} \times \mathtt{N}$ scalar, and an $\mathtt{N} \times \mathtt{Q}$ matrix.

The four computational steps, color-coded by role:

**Step 1 (Orange — Diagonal blocks):** Compute $Y_{\text{diag}} = D_k X_k$ for each chunk $k$ using the quadratic (attention) form of SSD within the chunk.

**Step 2 (Green — Input-to-State):** Compute the final state of each chunk assuming zero initial state:

$$
\text{state}_k = \sum_{t \in \text{chunk}_k} \underbrace{a_{k,\text{end}:t}^\times}_{\text{decay}} \cdot B_t x_t \tag{20}
$$

This is a batched matrix multiplication over all chunks in parallel.

**Step 3 (Yellow — State-to-State):** Run a scalar SSM recurrence on the chunk-boundary states. The chunk-to-chunk transition scalar is the cumulative product of all $a_t$ within a chunk:

$$
A_k^{\text{chunk}} = \prod_{t \in \text{chunk}_k} a_t \tag{21}
$$

The recurrence over chunks computes the *true* (cumulative) state at each chunk boundary:

$$
h_k^* = A_k^{\text{chunk}} \cdot h_{k-1}^* + \text{state}_k \tag{22}
$$

**Step 4 (Blue — State-to-Output):** For each chunk $k$, compute the contribution of the true initial state $h_{k-1}^*$ to the outputs within that chunk:

$$
Y_{\text{off}, t} = C_t^\top \left(a_{t:k\text{-start}}^\times \cdot h_{k-1}^*\right) \tag{23}
$$

This is again a batched matrix multiplication.

**Final output:**

$$
Y = Y_{\text{diag}} + Y_{\text{off}} \tag{24}
$$

### 4.2 Chunking and State Passing View

An equivalent interpretation in terms of the sequence directly:

Split the input sequence $X \in \mathbb{R}^{\mathtt{(T,P)}}$ into $\mathtt{T}/\mathtt{Q}$ chunks of length $\mathtt{Q}$.

**Step 1 — Intra-chunk outputs:** For each chunk, compute the local outputs as if the initial state were zero. Uses the quadratic (attention-mode) formula:

$$
Y_{\text{intra}, k} = L_k \circ (C_k B_k^\top) \cdot X_k \tag{25}
$$

where $L_k$ is the $\mathtt{Q} \times \mathtt{Q}$ 1-SS mask matrix for chunk $k$.

**Step 2 — Chunk states:** For each chunk $k$, compute the final SSM state assuming zero initial state. In matrix form over the chunk:

$$
s_k = \sum_{t=1}^{\mathtt{Q}} e^{A_{k,\mathtt{Q}:t}} B_{k,t} x_{k,t} \tag{26}
$$

This is computed as a batched matmul: $s_k \in \mathbb{R}^{\mathtt{(N,P)}}$.

**Step 3 — State passing (inter-chunk scan):** Compute the true global state at each chunk boundary by running the SSM recurrence on the chunk states. Let $\bar{A}_k = \exp\!\left(\sum_{t \in k} a_t\right)$ be the cumulative decay over chunk $k$. Then:

$$
h_k = \bar{A}_k \cdot h_{k-1} + s_k \tag{27}
$$

This is a sequential scan over $\mathtt{T}/\mathtt{Q}$ elements (much shorter than $\mathtt{T}$).

**Step 4 — State-to-output conversion:** For each chunk $k$ with true initial state $h_{k-1}$, compute the additional output contribution:

$$
Y_{\text{state}, k, t} = C_{k,t}^\top \left(e^{A_{k,t:0}} h_{k-1}\right) \tag{28}
$$

again using a batched matmul.

**Complexity analysis:**

| Step | Time Complexity | Uses Matmuls? | Parallelizable? |
|------|----------------|---------------|-----------------|
| 1 (Intra-chunk) | $O(\mathtt{T}\mathtt{Q}\mathtt{N})$ | Yes | Fully |
| 2 (Chunk states) | $O(\mathtt{T}\mathtt{N}\mathtt{P}/\mathtt{Q})$ | Yes | Fully |
| 3 (State passing) | $O(\mathtt{T}/\mathtt{Q} \cdot \mathtt{N}^2)$ | No (scan) | Sequential |
| 4 (State-to-output) | $O(\mathtt{T}\mathtt{N}\mathtt{P}/\mathtt{Q})$ | Yes | Fully |

Step 3 operates on a sequence of length $\mathtt{T}/\mathtt{Q}$ rather than $\mathtt{T}$, making it a small fraction of the total runtime for typical chunk sizes ($\mathtt{Q} = 64$–$256$).

**Asymptotic complexity of full SSD algorithm:**

$$
\text{FLOPs} = O\!\left(\mathtt{T}\mathtt{Q}\mathtt{N} + \frac{\mathtt{T}}{\mathtt{Q}}\mathtt{N}^2\mathtt{P}\right)
$$

Optimal chunk size balances these two terms: $\mathtt{Q}^* = \sqrt{\mathtt{N}\mathtt{P}}$.

For fixed $\mathtt{Q}$, total FLOPs scale as $O(\mathtt{T}\mathtt{N}^2)$, matching the linear SSM recurrence in asymptotic order — but now primarily via tensor-core-amenable matmuls.

### 4.3 Reference Implementation

The minimal SSD implementation in PyTorch (approximately 25 lines):

```python
def segsum(x):
    """Segment sum: computes log of 1-SS matrix entries.
    exp(segsum(A)) produces a 1-semiseparable matrix."""
    T = x.size(-1)
    x_cumsum = torch.cumsum(x, dim=-1)
    x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum

def ssd(X, A, B, C, block_len=64, initial_states=None):
    """
    X: (batch, length, n_heads, d_head)     — input
    A: (batch, length, n_heads)             — scalar log-decay per head
    B: (batch, length, n_heads, d_state)    — state input projection
    C: (batch, length, n_heads, d_state)    — state output projection
    Returns Y: (batch, length, n_heads, d_head)
    """
    # Rearrange into chunks
    X, A, B, C = [rearrange(x, "b (c l) ... -> b c l ...", l=block_len)
                  for x in (X, A, B, C)]
    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)

    # Step 1: Intra-chunk outputs (diagonal blocks)
    L = torch.exp(segsum(A))
    Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)

    # Step 2: Chunk states (green blocks — B terms)
    decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)

    # Step 3: Inter-chunk SSM recurrence (yellow blocks — A terms)
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # Step 4: State-to-output per chunk (blue blocks — C terms)
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum('bclhn,bchpn,bhcl->bclhp', C, states, state_decay_out)

    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
    return Y, final_state
```

**Key einsum shapes:**
- `bclhn`: batch, chunk, position-in-chunk, head, state-dim — for $C$ or $B$
- `bcshp`: batch, chunk, position-in-chunk, head, head-dim — for $X$ or $Y$
- `bhcls`: batch, head, chunk, row-in-chunk, col-in-chunk — for the 1-SS mask $L$
- `bchpn`: batch, chunk, head, head-dim, state-dim — for chunk states

### 4.4 Numerical Stability: The Segment Sum

Computing the 1-SS matrix $L$ requires evaluating all pairwise cumulative products $a_{i:j}^\times = a_i \cdots a_{j+1}$.

**Problem with direct products:** cumulative products can underflow exponentially (e.g., $0.9^{1000} \approx 2.6 \times 10^{-46}$).

**Problem with log-space differences:** computing $a_{i:j}^\times = e^{(\log a_i + \cdots)} = e^{\text{cumsum}(\log a)[i] - \text{cumsum}(\log a)[j]}$ via pairwise differences of cumsums suffers from **catastrophic cancellation** when the cumsums are large.

**Solution — the `segsum` primitive:** Compute independent partial cumsums without subtraction. The key insight is to produce the segment sums $\sum_{k=j}^{i-1} \log a_k$ for all pairs $(i,j)$ using batched cumsums without subtraction, then exponentiate:

```python
def segsum(x):
    T = x.size(-1)
    x_cumsum = torch.cumsum(x, dim=-1)
    # x_segsum[..., i, j] = cumsum[i] - cumsum[j]  (for i >= j)
    x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
    # The stable version avoids subtraction by direct construction;
    # the above is equivalent, stabilized by masking and the fact
    # that we only exponentiate after the subtraction
    mask = torch.tril(torch.ones(T, T, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum
```

The 1-SS matrix is then $L = \exp(\mathsf{segsum}(\log A))$.

**Note:** In the actual Mamba-2 code, the $A$ parameter is maintained in log-space (i.e., the model learns $\log a_t$ directly), so the `segsum` is applied to the raw parameter values.

### 4.5 Discretization

Mamba-2 works directly with *discrete* SSM parameters $A$ and $B$. These correspond to what earlier structured SSM papers (S4, Mamba-1) called $\bar{A}$ and $\bar{B}$ — the parameters after applying a **zero-order hold (ZOH) discretization** to a continuous-time system.

The ZOH discretization maps continuous parameters $(A_c, B_c, \Delta)$ to:

$$
\bar{A} = e^{\Delta A_c} \tag{29}
$$
$$
\bar{B} = A_c^{-1}(e^{\Delta A_c} - I)\, B_c \tag{30}
$$

In Mamba-1, these formulas were implemented with a heuristic: $\bar{A} = e^{\Delta A_c}$ (exact ZOH) but $\bar{B} = \Delta B_c$ (Euler approximation), because $e^x - 1 \approx x$ for small $x$.

In Mamba-2, the model directly learns $A \in \mathbb{R}^{\mathtt{(T)}}$ (the log-decay, so $a_t = e^{A_t} \in (0,1)$) and $B \in \mathbb{R}^{\mathtt{(T,N)}}$ as the discrete-time parameters. The continuous-time parameterization can be used by applying (29)–(30) as a preprocessing step.

---

## 5. Mamba-2 Architecture

### 5.1 Input Projections and Parameter Structure

The key architectural change from Mamba-1 is the **parallel projection** structure: all SSM parameters $(A, B, C)$ and the input $X$ are computed from a single linear projection of the layer input $u \in \mathbb{R}^{\mathtt{D}}$, rather than sequentially. This is motivated by the connection to attention and enables tensor parallelism.

**Projection dimensions:**

| Parameter | Shape | Description |
|-----------|-------|-------------|
| $X$ | $\mathtt{(T, H, P)}$ | Input, $\mathtt{H}$ heads of dim $\mathtt{P}$ |
| $A$ (log-decay) | $\mathtt{(T, H)}$ | Scalar log-decay per head |
| $B$ | $\mathtt{(T, G, N)}$ | State input projection, $\mathtt{G}$ groups |
| $C$ | $\mathtt{(T, G, N)}$ | State output projection, $\mathtt{G}$ groups |
| $\Delta$ | $\mathtt{(T, H)}$ | Discretization step size |
| $Z$ | $\mathtt{(T, H, P)}$ | Multiplicative gate |

Here $\mathtt{H}$ is the number of heads, $\mathtt{G}$ is the number of groups ($\mathtt{G} \leq \mathtt{H}$, with $B$ and $C$ shared across $\mathtt{H}/\mathtt{G}$ heads in each group), $\mathtt{P}$ is the head dimension, $\mathtt{N}$ is the state dimension.

**Grouped structure (analogous to grouped-query attention):** Multiple heads share the same $B$ and $C$ projections. A group of $\mathtt{H}/\mathtt{G}$ heads applies the same $B_t$ and $C_t$ but with different $X$, $A$, and $Z$ projections.

### 5.2 Full Layer Operations

The complete sequence of operations in a single Mamba-2 layer:

| Step | Operation | Formula |
|------|-----------|---------|
| 1 | Input LayerNorm | $u' = \text{LayerNorm}(u)$ |
| 2 | In-projection | $(X, Z, A_{\log}, B, C) = u' W_{\text{in}}^\top$ |
| 3 | Activation / discretization | $A = \text{softplus}(\Delta) \cdot A_{\log}$ (or similar) |
| 4 | Short depthwise convolution | $X', B', C' = \text{Conv1d}(X, B, C)$ then nonlinearity |
| 5 | SSD computation | $Y = \text{SSD}(X', A, B', C')$ |
| 6 | Multiplicative gate | $Y' = Y \odot \sigma(Z)$ |
| 7 | Output LayerNorm | $Y'' = \text{LayerNorm}(Y')$ |
| 8 | Out-projection | $\text{output} = Y'' W_{\text{out}}^\top$ |

The **depthwise convolution** (step 4) mixes the last few tokens locally before the SSM. It is applied separately to $X$, $B$, and $C$ with shared or separate filters.

**Parameter count comparison (single SSD layer vs. attention layer):**

For a model of width $\mathtt{D} = \mathtt{H} \cdot \mathtt{P}$:
- Attention $QKV$ projection: $3\mathtt{D}^2$ parameters
- SSD in-projection for $X, Z, B, C$: roughly $2\mathtt{D}^2 + 2\mathtt{D}\mathtt{N}\mathtt{G}/\mathtt{H} + \mathtt{D}$ parameters
- Both have the same out-projection cost $\mathtt{D}^2$

---

## 6. Systems Optimizations

### 6.1 Tensor Parallelism

In Mamba-1, applying Tensor Parallelism (TP) required 2 all-reduces per layer because some SSM parameters depended on inner activations (not just the layer input). In Mamba-2, because all parameters $(A, B, C, X, Z)$ are computed in a single parallel projection from the layer input, TP can be applied straightforwardly:

- **In-projection matrix** $W_{\text{in}}$ is column-sharded across $\mathtt{TP}$ devices.
- **Out-projection matrix** $W_{\text{out}}$ is row-sharded.
- **GroupNorm** uses a number of groups divisible by TP degree, so normalization is independent per device.

Result: **1 all-reduce per layer** (down from 2 in Mamba-1), matching the Transformer cost.

### 6.2 Sequence Parallelism

For very long sequences that cannot fit on a single device, the sequence dimension is split across devices. Mamba-2 supports two forms:

**Form 1 — Residual/normalization SP:** Replaces the all-reduce in TP with reduce-scatter → LayerNorm → all-gather. This is identical to Transformer SP and applies directly.

**Form 2 — Context parallelism (CP) for the SSD itself:** Uses the chunking structure of the SSD algorithm directly. Each GPU $k$ receives a contiguous chunk of the input sequence.

The protocol is:
1. Each GPU independently computes its local intra-chunk outputs and chunk states (Steps 1–2 of the SSD algorithm).
2. GPUs communicate their boundary states sequentially using point-to-point send/receive:

$$
h_k^* = \bar{A}_k h_{k-1}^* + s_k \tag{31}
$$

3. Each GPU updates its output with the received state (Step 4).

This is exactly the SSD state-passing step (Step 3) but distributed across GPUs instead of across chunks within a GPU. Communication is $O(\mathtt{N}\mathtt{P})$ per GPU boundary (state size), far smaller than attention's $O(\mathtt{T}\mathtt{N})$ KV cache communication.

### 6.3 Variable-Length Sequences

For batches with sequences of different lengths (common in fine-tuning and inference), SSMs have a natural solution: treat the whole batch as one long concatenated "sequence" and set the transition $A_t = 0$ at the boundaries between sequences.

Formally, for a token at the end of sequence $s$ (last position before the next sequence begins):

$$
a_{t_{\text{boundary}}} = 0 \tag{32}
$$

This zeroes out the state, preventing information from leaking across sequence boundaries, without requiring padding tokens. The resulting computation is identical to processing each sequence independently, but in a single batched operation.

---

## 7. Kernel Implementation: Fused SSD

### 7.1 The Five SSD Kernels

The original Triton implementation of the SSD prefill expresses the algorithm as five separate GPU kernels:

| Kernel | Operation | Key Formula |
|--------|-----------|-------------|
| **Chunk Cumsum** | Computes $\Delta$ and cumulative log-decay $A_{\text{cum}}$ within each chunk | $A_{\text{cum},t} = \sum_{s=1}^{t} a_s$ |
| **BMM (Chunk Intra)** | Computes intra-chunk 1-SS mask × $CB^\top$ product | $L_k \circ C_k B_k^\top$ via batched matmul |
| **Chunk State** | Computes chunk-boundary states assuming zero initial state | $s_k = \sum_t e^{A_{\text{cum},\mathtt{Q}} - A_{\text{cum},t}} B_t x_t$ |
| **State Passing** | Runs SSM recurrence across chunk boundaries | $h_k = e^{A_{\text{cum},k}} h_{k-1} + s_k$ |
| **Chunk Scan** | Computes final output combining intra-chunk and state contributions | $Y_k = Y_{\text{intra},k} + C_k (e^{A_{\text{cum}}} h_{k-1})$ |

### 7.2 Kernel Dependency Graph

The dependencies between the five kernels, per chunk:

```
Chunk Cumsum (all chunks, independent)
        │
        ▼
Chunk State (chunk k, independent across chunks)
        │
        ▼
State Passing (chunk k depends on chunk k-1: sequential!)
        │
        ▼
Chunk Scan (chunk k)
        ▲
        │
BMM (all chunks, independent)
```

- Chunk Cumsum, BMM, and Chunk State are **fully parallel** across chunks.
- State Passing is **strictly sequential**: chunk $k$ cannot begin until chunk $k-1$ is complete.
- Chunk Scan depends on both State Passing and BMM outputs.

### 7.3 Fusion Strategy and Synchronization

Fusing all five kernels into a single Triton kernel eliminates the implicit global synchronization between kernel launches. The key challenge is replacing implicit sync with explicit synchronization.

**Threadblock assignment:** The fused kernel assigns each threadblock to process a complete chunk through Chunk State → State Passing → Chunk Scan. This maximizes data reuse: intermediate results (the chunk state) remain in registers or L1 cache between steps.

**State Passing serialization via atomics:** Without the original kernel-boundary synchronization, State Passing correctness must be enforced via **atomic ordering**. Chunk $k$'s threadblock busy-waits on an atomic flag set by chunk $k-1$'s threadblock.

**Serialization overhead analysis:** Let $f$ be the fraction of combined runtime attributable to State Passing, and $n$ be the number of chunks processed concurrently. Naive serialization would give:

$$
\text{slowdown} = f \cdot n + (1-f) \cdot 1
$$

However, the State Passing synchronization *overlaps* with Chunk State and Chunk Scan computation in other chunks. The actual overhead is approximately:

$$
\text{effective time} = \text{SP compute time} + \max\!\left(\text{other compute time},\, \text{SP sync time}\right)
$$

For $f = 1/7$ (State Passing is 1/7 of total) and $n = 8$:

$$
\frac{1}{7} + \max\!\left(\frac{6}{7}, \frac{8}{7} \cdot \frac{1}{7} \cdot 7\right) = \frac{1}{7} + \frac{6}{7} \approx 1.14\times \text{ slowdown}
$$

In practice, Nsight Compute profiling shows that **fewer than 3% of warp stalls** are caused by State Passing synchronization — the latency is effectively hidden.

**Chunk Cumsum and BMM handling:** These two fast steps are handled by assigning the first few threadblocks to them. Later threadblocks (handling Chunk State / State Passing / Chunk Scan) wait for these prerequisites via atomics before proceeding.

**Threadblock dispatch order:**
```
If Chunk Cumsum work unassigned → assign to Chunk Cumsum
Else if BMM work unassigned     → assign to BMM  
Else                            → assign to (Chunk State → State Passing → Chunk Scan) for one chunk
```

### 7.4 Key Optimizations

**Threadblock reordering for serialization hiding:** Launch threadblocks for all `nheads` before advancing to the next chunk in the batch dimension. This means $\mathtt{H}$ threadblocks make progress simultaneously in State Passing instead of 1, reducing effective serialization by factor $\mathtt{H}$.

**Cache hints:** 
- Output tensors: marked low-priority (evict-first) since they are write-once.
- Shared tensors (e.g., $CB$ matrices shared across grouped heads): marked high-priority to reduce eviction.
- Release semantics on sync atomics to avoid unnecessary L1 cache flushes.

**Conditional separation:** The special-case handling for the first chunk (reading initial states) and last chunk (writing final states) is moved outside the fused kernel using PyTorch slice assignments. This reduces register pressure inside the main kernel.

**Intermediate datatypes:** Some computations (e.g., applying $A$ decay to $B$ in Chunk Scan) use fp16 instead of fp32, giving approximately 16% speedup with negligible accuracy loss.

**Chunk size tuning:** The optimal chunk size shifts from $\mathtt{Q} = 256$ (original unfused kernels) to $\mathtt{Q} = 128$ (fused kernel), because the fused kernel benefits from reduced register pressure at smaller chunk sizes.

**Fused kernel performance:**

| Metric | Value |
|--------|-------|
| SSD speedup (vs. unfused) | **1.50×–2.51×** on A100 and H100 |
| End-to-end speedup (Mamba-2 2.7B, seq=128K) | **~8–13%** |
| End-to-end speedup (seq=1K, batch=1) | **~20%** (dominated by launch overhead reduction) |
| Compute utilization | ~40–50% |
| Memory (L2) utilization | ~65–75% |
| Warp stalls from State Passing sync | <3% |

The fused kernel is primarily **L2-bound** rather than DRAM-bound, meaning that further optimization requires either algorithmic changes or tuning block sizes to improve cache behavior.

---

## 8. Complexity and Properties Summary

### Computational Complexity

| Mode | Time | Memory | Hardware |
|------|------|--------|----------|
| Linear (recurrent, inference) | $O(\mathtt{T}\mathtt{N}^2)$ per sequence | $O(\mathtt{N}^2)$ state | Scalar arithmetic |
| Quadratic (attention, naïve) | $O(\mathtt{T}^2\mathtt{N})$ | $O(\mathtt{T}^2)$ attention matrix | Tensor cores |
| Chunkwise SSD (training) | $O(\mathtt{T}\mathtt{N}^2)$ | $O(\mathtt{T}\mathtt{N})$ | Tensor cores |

### State Size

The SSM maintains a **constant-size state** $h \in \mathbb{R}^{\mathtt{N} \times \mathtt{P}}$ per head, regardless of sequence length. This contrasts with:
- Standard attention: KV cache grows as $O(\mathtt{T} \cdot \mathtt{N})$.
- SSD: state is always $O(\mathtt{N} \cdot \mathtt{P})$ regardless of $\mathtt{T}$.

This makes inference memory scale as $O(\mathtt{N})$ (per head), which is the primary efficiency advantage of SSMs at long context.

### The Sequence Length Scaling Advantage

- Self-attention compute and memory: $O(\mathtt{T}^2)$ — quadruples when sequence doubles.
- SSD compute: $O(\mathtt{T})$ — doubles when sequence doubles.
- SSD memory for states: $O(1)$ in sequence length.

### Key Model Properties

**Expressivity:** The 1-SS mask $L$ provides **input-dependent positional discounting**: the influence of position $j$ on position $i$ is weighted by $a_{i:j}^\times$, which depends on the input sequence (unlike fixed positional encodings). This is the SSM analogue of "selectivity."

**Selectivity mechanics:** When $a_t \approx 0$ at some time $t$, the state is effectively reset — information before $t$ is discarded. When $a_t \approx 1$, the state persists. The model learns $a_t$ as a function of the input, enabling input-dependent forgetting and remembering.

**Recurrent inference:** During autoregressive generation, the SSD layer operates purely in linear mode. Each new token requires only:

$$
h_t = a_t \cdot h_{t-1} + B_t x_t, \quad y_t = C_t^\top h_t
$$

with $O(\mathtt{N}^2)$ operations per step (dominated by the outer product $B_t x_t^\top \in \mathbb{R}^{\mathtt{N} \times \mathtt{P}}$ and the dot product $C_t^\top h_t$). There is no KV cache.

---

## 9. Key Formulas Quick Reference

### Core Model Equations

| Name | Formula |
|------|---------|
| SSM recurrence | $h_t = A_t h_{t-1} + B_t x_t,\quad y_t = C_t^\top h_t$ |
| SSD constraint | $A_t = a_t \cdot I_\mathtt{N}$ (scalar) |
| 1-SS mask entry | $L_{ij} = a_{i:j}^\times = \prod_{k=j+1}^{i} a_k$ |
| Attention-form SSD | $Y = (L \circ CB^\top)\, X$ |
| Matrix form entry | $M_{ij} = a_{i:j}^\times \cdot C_i^\top B_j$ |

### SSD Algorithm (Chunkwise)

| Step | Formula |
|------|---------|
| Intra-chunk 1-SS mask | $L_k = \exp\!\big(\mathsf{segsum}(A_k)\big)$ |
| Intra-chunk output | $Y_{\text{intra},k} = (L_k \circ C_k B_k^\top) X_k$ |
| Chunk state | $s_k = \mathsf{einsum}(B_k,\, e^{A_{\text{cum},k,-1} - A_{\text{cum},k}},\, X_k)$ |
| Chunk decay | $\bar{A}_k = \exp\!\left(\sum_{t \in k} a_t\right)$ |
| Inter-chunk recurrence | $h_k = \bar{A}_k h_{k-1} + s_k$ |
| State-to-output | $Y_{\text{state},k,t} = C_{k,t}^\top \left(e^{A_{\text{cum},k,t}}\, h_{k-1}\right)$ |
| Final output | $Y = Y_{\text{intra}} + Y_{\text{state}}$ |

### Segment Sum

| Formula |
|---------|
| $\mathsf{segsum}(x)_{ij} = \sum_{k=j}^{i-1} x_k \quad (i \geq j),\quad -\infty \quad (i < j)$ |
| 1-SS matrix: $L = \exp(\mathsf{segsum}(\log A))$ |
| Implementation: $\mathsf{segsum}(x)_{ij} = \mathsf{cumsum}(x)_i - \mathsf{cumsum}(x)_j$ (with tril mask) |

### Discretization (ZOH)

| Formula |
|---------|
| $\bar{A} = e^{\Delta A_c}$ |
| $\bar{B} = A_c^{-1}(e^{\Delta A_c} - I)\, B_c$ |
| Mamba-1/2 approximation: $\bar{B} \approx \Delta B_c$ (Euler) |

### SMA Duality

| Form | Computation order |
|------|------------------|
| Quadratic | $(Q,K) \to G$; $G \circ L \to M$; $(M,V) \to Y$ |
| Linear | $(V,K) \to Z$; $(L,Z) \to H$; $(Q,H) \to Y$ |

### Complexity

| Quantity | Value |
|----------|-------|
| Training FLOPs (SSD) | $O(\mathtt{T}\mathtt{N}^2)$ |
| Inference FLOPs per step | $O(\mathtt{N}^2)$ |
| State memory | $O(\mathtt{N} \cdot \mathtt{P})$ per head (constant in $\mathtt{T}$) |
| Optimal chunk size | $\mathtt{Q}^* = \sqrt{\mathtt{N}\mathtt{P}}$ |
