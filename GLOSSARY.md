# burn-mamba

The Mamba-1, Mamba-2 and Mamba-3 blocks for burn-stack, written only with
portable Burn tensor ops.

The terms of the containers (family, sequence, token, row, cache, segment, …)
are in the [burn-stack glossary](../burn-stack/GLOSSARY.md). The families of
this crate are Mamba-1, Mamba-2 and Mamba-3.

A source can use another word for a term here, or use one of our words with another
meaning. These docs map the words of the sources to our terms:

- The Mamba papers, the Mamba blog, the official code and mamba3-minimal:
  [`docs/source-terms.md`](docs/source-terms.md).
- The H-Net and Multi-Gate Residuals papers:
  [`../burn-stack/docs/source-terms.md`](../burn-stack/docs/source-terms.md).

## Language

### The plant

**Plant**:
The state recurrence of a block: its transition, its write and its readout.
_Avoid_: SSM core, system (alone)

**Head**:
One of the parallel recurrences of a Mamba-2 or Mamba-3 plant. Each head has
its own state, step size and decay.
_Avoid_: SSM (for one head)

**B/C group**:
A set of heads that share one `B` and one `C`.
_Avoid_: group (alone)

**State**:
The `hₜ` of the plant: what the plant carries from one micro-step to the next.
_Avoid_: hidden state, latent state, SSM state, recurrent state

**Transition**:
The part of the plant that carries the state to the next micro-step. Its
factors are the decay and, in Mamba-3, the rotation.

**Write**:
The part of the plant that adds samples to the state.
_Avoid_: state projection

**Sample**:
The product `B xᵀ` of one write position, before its mass.
_Avoid_: state-input

**Readout**:
The part of the plant that reads the state through `C`.
_Avoid_: state emission, contraction

**Step size**:
The data-dependent `Δ` of one micro-step. It scales the write, sets the decay
and paces the rotation.

**Decay**:
The factor `α = exp(Δ·A)` of the transition, with `A < 0`. A gain can set it.

**In-projection**:
The fused projection at the entry of a block. It gives `x`, the gate and, in
Mamba-2 and Mamba-3, the data-dependent parameters of the plant.

**Channel group**:
One named part of the output of the in-projection. A per-micro-step group holds
`u` sub-projections. A per-token group holds one.

**Learnable initial state**:
A trained state that each forward adds to the state of its cache. Mamba-2 and
Mamba-3 have it. Only the `Minimal` SSD path supports it.
_Avoid_: origin

**Gate**:
The SiLU output gate `z` of a block.
_Avoid_: gate (for the gain, or for a tap that closes at token boundaries)

**Plant frame**:
The work of a Mamba-3 block around the plant. The entry ends at the rotated `B`
and `C`, the masses and the tap slots. The tail starts after the readout and
ends at `out_proj`.

### SSD evaluation

**SSD**:
Structured State Space Duality: the matrix form of the recurrence. An SSD scan
computes it chunk by chunk.
_Avoid_: SSD core

**SSD path**:
The method that computes the chunkwise SSD scan: `Minimal`, `Serial` or
`SerialRecalculated`. All SSD paths give the same values and gradients.
_Avoid_: algorithm, variant, mode

**Pathway**:
How Mamba-3 maps its trapezoidal recurrence onto SSD calls. Double-SSD makes
one SSD call for each term of the trapezoid. Single-SSD makes one SSD call for
all terms. Each pathway works with each SSD path.
_Avoid_: algorithm, variant, mode, path

**Accumulator**:
The running sum that single-SSD keeps in place of the state. It is equal to the
state only at call boundaries.
_Avoid_: state (for the accumulator)

### Mamba-3 options

**MambaProduct**:
The Mamba-3 option that runs `u` micro-steps for each token.
_Avoid_: DeltaProduct (a different mechanism)

**MIMO**:
The Mamba-3 option in which the write and the readout of each head have rank
`m > 1` and share one state.

**MIMO rank**:
The rank `m` of the write and of the readout of each head.

**SISO**:
MIMO with rank 1. It is not a separate form.

### Time axes

**Micro-step**:
One recurrence step of the plant. Under MambaProduct, a token has `u`
micro-steps. Every other token has one.
_Avoid_: sub-step, inner step, step (alone)

**Write axis**:
The time axis with one entry for each micro-step.
_Avoid_: folded axis, sequence axis

**Read axis**:
The time axis with one entry for each token.

### The trapezoid

**Trapezoid**:
The discretisation rule of Mamba-3. It writes the current sample and the
samples at its taps into the state.

**Tap**:
An earlier sample that the `β` term of the write reads.

**Lag**:
The number of write positions between a tap and the current sample.

**Lag-1 tap**:
The tap one write position back: the previous micro-step.
_Avoid_: horizontal tap, interior tap, near tap

**Lag-`u` tap**:
The tap `u` write positions back: the same micro-step of the previous token. At
`u = 1`, it is the lag-1 tap.
_Avoid_: vertical tap, far tap

**Tap pattern**:
The choice of which taps exist, and whether the lag-1 tap closes at token
boundaries.
_Avoid_: trapezoid (for the choice)

**Mass**:
The share of the step size `Δ` that the write of one micro-step gives to one
sample. The masses of a micro-step add up to `Δ`.
_Avoid_: weight, coefficient (for a mass)

**Trapezoid coefficient**:
One of `α`, `β`, `γ` in the recurrence of the trapezoid. `γ` is a mass. `β` is
a tap mass times the decay, so it is not a mass.

### Rotation

**Rotation**:
The orthogonal factor of the Mamba-3 state transition. It turns the state by a
data-dependent amount at each micro-step. It is not a positional encoding.
_Avoid_: RoPE, positional encoding, complex transition

**RoPE trick**:
The technique that moves the accumulated rotation out of the state and into B
and C. The SSD scan then sees only a scalar decay.
_Avoid_: RoPE (as a name for the rotation)

**Rotary position embedding**:
The positional encoding of transformers: a fixed rotation whose angle counts
positions. The rotation of Mamba-3 is different, because its angle depends on
the data.
_Avoid_: vanilla RoPE, standard RoPE, RoPE

**Rotation kind**:
The group of the rotated part of the state: `Real1D`, `Complex2D`,
`Quaternion4D` or `Rotor4D`. `Real1D` is the trivial group, so "no rotation"
means `Real1D`.
_Avoid_: RoPE kind

**Rotation fraction**:
The share of the state that the rotation turns. The rest of the state does not
turn.
_Avoid_: RoPE fraction

**Rotation range**:
The bound on the angle of one micro-step, in units of `π·Δ`.

### Positive systems

**Positive system**:
A scalar recurrence for each head, next to the plant. It reads only the
in-projection and sets coefficients of the plant.
_Avoid_: side system, gate

**Gain**:
The positive system that sets the decay of the plant.
_Avoid_: gate, Kalman gate

**Tropical register**:
The positive system that adds a soft maximum `max(c + a, b)` to the readout.
_Avoid_: max-plus gate

### Packing

**Origin**:
The cache that each segment of a packed call starts from. With no origin, a
segment starts fresh. It is not the learnable initial state.
_Avoid_: initial cache, initial state (for an origin)
